"""End-to-end smoke test against a REAL uvicorn server over HTTP.

Demonstrates the complete annotation flow for two concurrent orders:

  1. create A and B with the same prefix;
  2. both stream in lockstep (batched) until 8 unacked tokens -> both pause;
  3. keep reading B (auto-ack) while A stays stopped  -> B keeps producing,
     A stays frozen;
  4. re-fetch A's unacknowledged fragment and prove byte-identical replay;
  5. ack A -> credit freed -> A resumes from its exact old progress;
  6. exercise rejected / duplicate / late acknowledgements.

Run:  python scripts/smoke_http.py
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import time

# Make ``app`` importable when the script is run directly (python scripts/...).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import httpx

from app.engine import MAX_BACKLOG


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


PORT = _free_port()
BASE = f"http://127.0.0.1:{PORT}"
PREFIX = "The quick brown fox"
LENGTH = 40


def banner(title: str) -> None:
    print("\n" + "=" * 68)
    print(f"  {title}")
    print("=" * 68)


async def wait_state(c: httpx.AsyncClient, oid: str, *, generated: int | None = None,
                     status: str | None = None, timeout: float = 15.0) -> dict:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        info = (await c.get(f"{BASE}/api/orders/{oid}")).json()
        if generated is not None and info["generated"] < generated:
            await asyncio.sleep(0.05)
            continue
        if status is not None and info["status"] != status:
            await asyncio.sleep(0.05)
            continue
        return info
    raise TimeoutError(f"order {oid} never reached target state")


async def main() -> None:
    # Launch a REAL uvicorn server in a subprocess (same entrypoint as Docker).
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app",
         "--host", "127.0.0.1", "--port", str(PORT), "--log-level", "warning"],
        cwd=ROOT,
    )
    deadline = time.monotonic() + 20
    async with httpx.AsyncClient(timeout=10) as c:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError("uvicorn subprocess exited early")
            try:
                if (await c.get(f"{BASE}/api/health")).status_code == 200:
                    break
            except httpx.TransportError:
                await asyncio.sleep(0.2)
        else:
            raise RuntimeError("server did not become ready")

    try:
        async with httpx.AsyncClient(timeout=10) as c:
            banner("1. 创建两路订单（相同前缀，长度 %d）" % LENGTH)
            for oid in ("A", "B"):
                r = await c.post(f"{BASE}/api/orders",
                                 json={"order_id": oid, "prefix": PREFIX, "length": LENGTH})
                r.raise_for_status()
                print(f"  order {oid}: {r.json()}")

            banner(f"2. 两路合批推进，各自攒满 {MAX_BACKLOG} 个未确认 token 后暂停")
            a = await wait_state(c, "A", generated=8)
            b = await wait_state(c, "B", generated=8)
            print(f"  A: generated={a['generated']} backlog={a['backlog']} status={a['status']}")
            print(f"  B: generated={b['generated']} backlog={b['backlog']} status={b['status']}")
            assert a["status"] == "paused" and b["status"] == "paused"
            health = (await c.get(f"{BASE}/api/health")).json()
            print(f"  health: tick={health['tick_count']} last_batch_size={health['last_batch_size']}")

            banner("3. A 停读不确认；B 持续读并确认 → B 继续出字，A 冻结")
            async def reader_b():
                high = -1
                for _ in range(400):
                    info = (await c.get(f"{BASE}/api/orders/B")).json()
                    if info["generated"] > high:
                        await c.post(f"{BASE}/api/orders/B/ack",
                                     json={"seq": info["generated"] - 1})
                        high = info["generated"]
                    if info["status"] == "done":
                        return
                    await asyncio.sleep(0.03)

            rb = asyncio.create_task(reader_b())
            b = await wait_state(c, "B", generated=16)
            a = (await c.get(f"{BASE}/api/orders/A")).json()
            print(f"  A 停读后: generated={a['generated']} status={a['status']} (应保持 8/paused)")
            print(f"  B 续读后: generated={b['generated']} status={b['status']} (应 >=16)")
            assert a["generated"] == 8 and a["status"] == "paused"
            assert b["generated"] >= 16
            await rb

            banner("4. 原样重取 A 的未确认片段（两次取文本必须逐字一致）")
            f1 = (await c.get(f"{BASE}/api/orders/A/tokens?start=0&end=8")).json()
            f2 = (await c.get(f"{BASE}/api/orders/A/tokens?start=0&end=8")).json()
            f3 = (await c.get(f"{BASE}/api/orders/A/tokens?start=3&end=6")).json()
            print(f"  首次取 seq 0..7 : {f1['text']!r}")
            print(f"  重取 seq 0..7   : {f2['text']!r}")
            print(f"  重取 seq 3..5   : {f3['text']!r}")
            assert f1["text"] == f2["text"], "重取必须原样返回"
            assert f3["text"] == f1["text"][3:6]

            banner("5. 确认规则：越界拒绝 / 重复与迟到不多腾额度")
            bad = await c.post(f"{BASE}/api/orders/A/ack", json={"seq": 99})
            print(f"  越界确认 seq=99 -> HTTP {bad.status_code} {bad.json()['detail']}")
            assert bad.status_code == 409
            ack = (await c.post(f"{BASE}/api/orders/A/ack", json={"seq": 3})).json()
            print(f"  确认 seq=3 -> backlog={ack['backlog']} freed={ack['credit_freed']} status={ack['status']}")
            dup = (await c.post(f"{BASE}/api/orders/A/ack", json={"seq": 3})).json()
            late = (await c.post(f"{BASE}/api/orders/A/ack", json={"seq": 0})).json()
            print(f"  重复确认 seq=3 -> freed={dup['credit_freed']} (应为 0)")
            print(f"  迟到确认 seq=0 -> freed={late['credit_freed']} (应为 0)")
            assert dup["credit_freed"] == 0 and late["credit_freed"] == 0

            banner("6. A 额度恢复后续读，从原进度继续（前 8 字不变）")
            before = f1["text"]
            a = await wait_state(c, "A", generated=12)
            again = (await c.get(f"{BASE}/api/orders/A/tokens?start=0&end=12")).json()
            print(f"  续读后 seq 0..11: {again['text']!r}")
            print(f"  前 8 字与停读前一致: {again['text'][:8] == before}")
            assert again["text"][:8] == before

            banner("7. 持续确认直到 A 全部完成")
            for _ in range(400):
                info = (await c.get(f"{BASE}/api/orders/A")).json()
                if info["generated"] >= 1:
                    await c.post(f"{BASE}/api/orders/A/ack",
                                 json={"seq": info["generated"] - 1})
                if info["status"] == "done":
                    break
                await asyncio.sleep(0.03)
            final = (await c.get(f"{BASE}/api/orders/A")).json()
            full = (await c.get(f"{BASE}/api/orders/A/tokens?start=0&end=40")).json()
            print(f"  A 最终: generated={final['generated']}/{final['total']} status={final['status']}")
            print(f"  完整续写（接在前缀 {PREFIX!r} 之后）: {full['text']!r}")
            assert final["status"] == "done" and final["generated"] == LENGTH

            banner("全部断言通过 ✓")
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except (ProcessLookupError, PermissionError):
            # Some sandboxes disallow signalling the child; it exits with the
            # parent process group anyway, and it only served loopback traffic.
            pass
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except (ProcessLookupError, PermissionError):
                pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:  # noqa: BLE001
        print(f"\nSMOKE TEST FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
