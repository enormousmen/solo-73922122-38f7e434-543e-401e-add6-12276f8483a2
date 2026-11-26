"""End-to-end API tests using FastAPI's ASGI transport (no network needed)."""

import asyncio

from app.engine import MAX_BACKLOG


async def wait_generated(client, oid, n):
    for _ in range(500):
        info = (await client.get(f"/api/orders/{oid}")).json()
        if info["generated"] >= n:
            return info
        await asyncio.sleep(0.02)
    raise TimeoutError(f"order {oid} did not reach {n} tokens")


async def test_health_and_ui(client):
    r = await client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["max_backlog"] == MAX_BACKLOG
    ui = await client.get("/")
    assert ui.status_code == 200 and "双路续写" in ui.text


async def test_full_two_stream_demo_flow(client):
    # 1. create two orders with identical prefix
    for oid in ("A", "B"):
        r = await client.post("/api/orders",
                              json={"order_id": oid, "prefix": "The quick brown fox",
                                    "length": 40})
        assert r.status_code == 201, r.text

    # 2. both start in lockstep until each holds 8 unacked tokens, then pause
    a = await wait_generated(client, "A", 8)
    b = await wait_generated(client, "B", 8)
    assert a["backlog"] == 8 and a["status"] == "paused"
    assert b["backlog"] == 8 and b["status"] == "paused"

    # 3. B keeps reading: a background worker acknowledges everything as it
    #    arrives, so B retains budget and keeps generating.  A stays stopped.
    async def reader_b():
        high = -1
        for _ in range(600):
            info = (await client.get("/api/orders/B")).json()
            if info["generated"] > high:
                await client.post("/api/orders/B/ack",
                                  json={"seq": info["generated"] - 1})
                high = info["generated"]
            if info["status"] == "done":
                return
            await asyncio.sleep(0.02)

    task = asyncio.create_task(reader_b())
    b = await wait_generated(client, "B", 14)
    a = (await client.get("/api/orders/A")).json()
    assert a["generated"] == 8 and a["status"] == "paused", "stopped A must freeze"
    assert b["generated"] >= 14
    await task

    # 4. re-fetch A's unacknowledged fragment: identical bytes every time
    f1 = (await client.get("/api/orders/A/tokens?start=0&end=8")).json()
    f2 = (await client.get("/api/orders/A/tokens?start=0&end=8")).json()
    f3 = (await client.get("/api/orders/A/tokens?start=3&end=6")).json()
    assert f1["text"] == f2["text"]
    assert f3["text"] == f1["text"][3:6]

    # 5. ack beyond delivered high-water is refused (HTTP 409)
    bad = await client.post("/api/orders/A/ack", json={"seq": 50})
    assert bad.status_code == 409

    # 6. acknowledge seq 0..3 -> credit freed, order becomes active again
    ack = (await client.post("/api/orders/A/ack", json={"seq": 3})).json()
    assert ack["credit_freed"] == 4 and ack["backlog"] == 4
    assert ack["status"] == "active"

    # 7. duplicate / late acks free nothing
    dup = (await client.post("/api/orders/A/ack", json={"seq": 3})).json()
    late = (await client.post("/api/orders/A/ack", json={"seq": 0})).json()
    assert dup["credit_freed"] == 0 and late["credit_freed"] == 0

    # 8. A resumes from the exact old progress, extending the retained tokens
    before = f1["text"]
    a = await wait_generated(client, "A", 12)
    again = (await client.get("/api/orders/A/tokens?start=0&end=12")).json()
    assert again["text"][:8] == before  # old bytes preserved
    assert len(again["text"]) == 12

    # 9. ack everything delivered so A can run to completion
    async def drain_a():
        for _ in range(600):
            info = (await client.get("/api/orders/A")).json()
            if info["generated"] >= 1:
                await client.post("/api/orders/A/ack",
                                  json={"seq": info["generated"] - 1})
            if info["status"] == "done":
                return
            await asyncio.sleep(0.02)

    await asyncio.wait_for(drain_a(), timeout=15)
    final = (await client.get("/api/orders/A")).json()
    assert final["generated"] == 40 and final["status"] == "done"

    # 10. fetch ranges: 404 on unknown order / future seq
    assert (await client.get("/api/orders/NOPE")).status_code == 404
    assert (await client.get(
        "/api/orders/A/tokens?start=999&end=1000")).status_code == 404


async def test_validation_errors(client):
    r = await client.post("/api/orders",
                          json={"order_id": "bad id!", "prefix": "", "length": 10})
    assert r.status_code == 400
    r = await client.post("/api/orders",
                          json={"order_id": "Z", "prefix": "", "length": 0})
    assert r.status_code == 422  # pydantic rejects length <= 0
    r = await client.get("/api/orders/Z/tokens?start=0&end=1")
    assert r.status_code == 404
