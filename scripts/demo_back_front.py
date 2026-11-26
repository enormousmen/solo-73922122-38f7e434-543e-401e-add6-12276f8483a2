"""API demo: fetch the BACK slice first, then fill the FRONT gap.

Starts a real uvicorn subprocess (same entrypoint as Docker) and drives the
HTTP protocol with printed request/response pairs:

  1. create one order and wait until 8 unconfirmed tokens are buffered/paused;
  2. fetch only the back slice  seq [4,8)   ("先取后段");
  3. try to ack seq=7 anyway                 -> HTTP 409, no credit freed;
  4. fetch the front slice seq [0,4)         ("再补前段");
  5. ack seq=7 once                          -> 200, frees all 8 credits;
  6. re-fetch [0,8)                          -> byte-identical original;
  7. duplicate ack seq=7                     -> credit_freed=0;
  8. empty slice [0,0) on a fresh order fills no gap -> ack seq=0 is 409.

Usage:
    python scripts/demo_back_front.py            # launches its own uvicorn
    python scripts/demo_back_front.py --base http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import asyncio
import os
import socket
import subprocess
import sys
import tempfile
import time
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import httpx  # noqa: E402

PREFIX = "The quick brown fox"
LENGTH = 40


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def show(req: str, response: httpx.Response) -> dict:
    try:
        data = response.json()
    except ValueError:
        data = {"raw": response.text}
    print(f"\n$ {req}")
    print(f"-> HTTP {response.status_code}")
    for key in ("start", "end", "text", "next_seq", "generated", "status",
                "acked_seq", "backlog", "credit_freed", "ack_applied",
                "delivered_contig_seq", "detail"):
        if isinstance(data, dict) and key in data:
            print(f"     {key} = {data[key]!r}")
    return data if isinstance(data, dict) else {}


async def wait_generated(c: httpx.AsyncClient, base: str, oid: str, n: int) -> dict:
    for _ in range(400):
        info = (await c.get(f"{base}/api/orders/{oid}")).json()
        if info["generated"] >= n:
            return info
        await asyncio.sleep(0.05)
    raise TimeoutError(f"{oid} did not reach {n} tokens")


async def run(base: str) -> None:
    tag = uuid.uuid4().hex[:6]
    fid, eid = f"DF{tag}", f"DE{tag}"
    async with httpx.AsyncClient(timeout=10) as c:
        # 1. order + full 8-token window --------------------------------
        r = await c.post(f"{base}/api/orders",
                         json={"order_id": fid, "prefix": PREFIX, "length": LENGTH})
        r.raise_for_status()
        await wait_generated(c, base, fid, 8)
        info = (await c.get(f"{base}/api/orders/{fid}")).json()
        print(f"order {fid} ready: generated={info['generated']} "
              f"backlog={info['backlog']} status={info['status']}")
        assert info["backlog"] == 8 and info["status"] == "paused"

        # 2. back slice only --------------------------------------------
        back = show(f"GET /api/orders/{fid}/tokens?start=4&end=8   # 先取后段",
                    await c.get(f"{base}/api/orders/{fid}/tokens?start=4&end=8"))
        assert back["text"]

        # 3. ack the tail with a front gap -> 409 ------------------------
        blocked = show(f"POST /api/orders/{fid}/ack {{\"seq\":7}}      # 前段缺失",
                       await c.post(f"{base}/api/orders/{fid}/ack", json={"seq": 7}))
        assert "detail" in blocked
        assert (await c.get(f"{base}/api/orders/{fid}")).json()["acked_seq"] == -1

        # 4. fill the front gap -----------------------------------------
        front = show(f"GET /api/orders/{fid}/tokens?start=0&end=4   # 再补前段",
                      await c.get(f"{base}/api/orders/{fid}/tokens?start=0&end=4"))
        full = (await c.get(f"{base}/api/orders/{fid}/tokens?start=0&end=8")).json()
        assert front["text"] + back["text"] == full["text"]

        # 5. one cumulative ack now covers front + early back -----------
        ack = show(f"POST /api/orders/{fid}/ack {{\"seq\":7}}      # 一并确认",
                   await c.post(f"{base}/api/orders/{fid}/ack", json={"seq": 7}))
        assert ack["credit_freed"] == 8 and ack["status"] == "active"

        # 6. re-fetch is still the original text ------------------------
        again = (await c.get(f"{base}/api/orders/{fid}/tokens?start=0&end=8")).json()
        print(f"\nre-fetch [0,8) = {again['text']!r}\nidentical = "
              f"{again['text'] == full['text']}")
        assert again["text"] == full["text"]

        # 7. duplicate ack frees nothing --------------------------------
        dup = show(f"POST /api/orders/{fid}/ack {{\"seq\":7}}      # 重复确认",
                   await c.post(f"{base}/api/orders/{fid}/ack", json={"seq": 7}))
        assert dup["credit_freed"] == 0 and dup["ack_applied"] is False

        # 8. empty slice cannot fill a gap -------------------------------
        await c.post(f"{base}/api/orders",
                     json={"order_id": eid, "prefix": "z", "length": 10})
        await wait_generated(c, base, eid, 8)
        empty = await c.get(f"{base}/api/orders/{eid}/tokens?start=0&end=0")
        show(f"GET /api/orders/{eid}/tokens?start=0&end=0  # 空片", empty)
        bad = await c.post(f"{base}/api/orders/{eid}/ack", json={"seq": 0})
        show(f"POST /api/orders/{eid}/ack {{\"seq\":0}}      # 空片不补缺口", bad)
        assert bad.status_code == 409

        print("\nDEMO ASSERTIONS PASSED ✓")


async def wait_ready(c: httpx.AsyncClient, base: str, deadline: float) -> None:
    while time.monotonic() < deadline:
        try:
            if (await c.get(f"{base}/api/health")).status_code == 200:
                return
        except httpx.TransportError:
            await asyncio.sleep(0.2)
    raise RuntimeError("server did not become ready")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None,
                     help="Base URL of a running server (e.g. http://127.0.0.1:8000). "
                          "If omitted, a local uvicorn subprocess is started.")
    args = ap.parse_args()

    if args.base:
        asyncio.run(run(args.base.rstrip("/")))
        return

    port = free_port()
    base = f"http://127.0.0.1:{port}"
    # Route the child's logs away from our stdout pipe: otherwise a surviving
    # child would hold the pipe's write end open after this script exits.
    log = open(os.path.join(tempfile.gettempdir(), "demo_back_front_uvicorn.log"),
               "wb")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
    )
    try:
        async def _boot_and_run():
            async with httpx.AsyncClient(timeout=10) as c:
                await wait_ready(c, base, time.monotonic() + 25)
            await run(base)
        asyncio.run(_boot_and_run())
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except (ProcessLookupError, PermissionError):
                pass
        except (ProcessLookupError, PermissionError):
            # Some sandboxes disallow signalling the child; it is loopback-only
            # and exits with its parent session.
            pass
        log.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"\nDEMO FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
