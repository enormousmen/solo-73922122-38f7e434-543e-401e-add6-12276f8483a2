"""Tests for the delivery-contiguity ack protocol.

Covers the issue where fetching only a back slice (e.g. seq 4..7) and acking
its tail wrongly freed credit for the never-returned front slice:

* ack past a gap in *delivered* seqs -> HTTP 409 / AckRejected, state intact;
* fill the front gap and the early back slice can be acked together;
* only fully, successfully sent response bodies mark seqs delivered — empty
  slices, failed sends and cancelled streams fill no gap;
* re-fetch is byte-identical; duplicate/late acks free no extra credit;
* the per-order delivery ledger stays bounded to MAX_BACKLOG slots;
* an in-flight response on one order never blocks the other order.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from app.engine import AckRejected, GenerationEngine, MAX_BACKLOG
from app.main import fetch_tokens
from app.model import TinyARModel


async def _ticks(eng: GenerationEngine, n: int) -> None:
    for _ in range(n):
        await eng.step()


# --------------------------------------------------------------------- #
# Engine-level protocol
# --------------------------------------------------------------------- #
async def test_back_slice_only_cannot_ack_tail():
    eng = GenerationEngine(model=TinyARModel(), tick_interval=999)
    await eng.create_order("O", "The quick brown fox", length=40)
    await _ticks(eng, MAX_BACKLOG)
    order = eng.orders["O"]
    assert order.generated == 8 and order.backlog == 8

    await eng.fetch("O", 4, 8)  # only the back slice
    assert order.delivered_seqs == [4, 5, 6, 7]
    with pytest.raises(AckRejected):
        await eng.acknowledge("O", 7)

    # rejected ack mutates nothing: progress + credit untouched
    assert order.acked_seq == -1
    assert order.backlog == 8
    assert order.status.value == "paused"


async def test_front_gap_filled_then_early_back_slice_acks_together():
    eng = GenerationEngine(model=TinyARModel(), tick_interval=999)
    await eng.create_order("O", "The quick brown fox", length=40)
    await _ticks(eng, MAX_BACKLOG)

    back = await eng.fetch("O", 4, 8)
    with pytest.raises(AckRejected):
        await eng.acknowledge("O", 7)

    front = await eng.fetch("O", 0, 4)
    # the two slices really are contiguous pieces of the same original text
    assert (front["text"] + back["text"]) == (await eng.fetch("O", 0, 8))["text"]

    ack = await eng.acknowledge("O", 7)  # one cumulative ack covers both
    assert ack["ack_applied"] is True
    assert ack["credit_freed"] == 8
    assert ack["acked_seq"] == 7
    assert ack["backlog"] == 0
    assert ack["status"] == "active"


async def test_incremental_ack_keeps_out_of_order_tail():
    eng = GenerationEngine(model=TinyARModel(), tick_interval=999)
    await eng.create_order("O", "hello", length=40)
    await _ticks(eng, MAX_BACKLOG)

    await eng.fetch("O", 4, 8)   # back arrives first ...
    await eng.fetch("O", 0, 4)   # ... then the front gap is filled

    first = await eng.acknowledge("O", 3)
    assert first["credit_freed"] == 4
    # old seqs 4..7 must remain on record against the shifted window
    assert eng.orders["O"].delivered_seqs == [4, 5, 6, 7]

    second = await eng.acknowledge("O", 7)
    assert second["credit_freed"] == 4
    assert second["backlog"] == 0

    dup = await eng.acknowledge("O", 7)
    assert dup["ack_applied"] is False and dup["credit_freed"] == 0
    late = await eng.acknowledge("O", 2)
    assert late["ack_applied"] is False and late["credit_freed"] == 0


async def test_empty_slice_marks_no_sequence():
    eng = GenerationEngine(model=TinyARModel(), tick_interval=999)
    await eng.create_order("O", "x", length=10)
    await _ticks(eng, MAX_BACKLOG)

    empty = await eng.fetch("O", 0, 0)
    assert empty["text"] == "" and empty["start"] == empty["end"] == 0
    assert eng.orders["O"].delivered_seqs == []
    with pytest.raises(AckRejected):
        await eng.acknowledge("O", 0)


async def test_delivery_ledger_never_exceeds_window():
    eng = GenerationEngine(model=TinyARModel(), tick_interval=999)
    await eng.create_order("O", "some prefix", length=200)

    # Drive a long lifetime: generate, deliver one seq at a time, ack it.
    for _ in range(200):
        await eng.step()
        o = eng.orders["O"]
        nxt = o.acked_seq + 1
        if nxt < o.generated:
            await eng.fetch("O", nxt, nxt + 1)
            await eng.acknowledge("O", nxt)
        # the delivery record is bounded by the unconfirmed window ...
        assert len(o.delivered_seqs) <= MAX_BACKLOG
        assert o.delivered_mask < (1 << MAX_BACKLOG)
    # ... while the replay text buffer is still retained in full.
    assert eng.orders["O"].generated == 200
    full = await eng.fetch("O", 0, 200)
    assert len(full["text"]) == 200


async def test_replayed_slice_is_byte_identical_and_remarks():
    eng = GenerationEngine(model=TinyARModel(), tick_interval=999)
    await eng.create_order("O", "repeatable prefix", length=16)
    await _ticks(eng, 8)
    first = await eng.fetch("O", 0, 8)
    again = await eng.fetch("O", 0, 8)
    assert first["text"] == again["text"]
    # re-fetching keeps every slot marked (idempotent)
    assert eng.orders["O"].delivered_seqs == list(range(8))


# --------------------------------------------------------------------- #
# HTTP API
# --------------------------------------------------------------------- #
async def _wait_generated(client, oid, n):
    for _ in range(500):
        info = (await client.get(f"/api/orders/{oid}")).json()
        if info["generated"] >= n:
            return info
        await asyncio.sleep(0.02)
    raise TimeoutError(f"order {oid} did not reach {n}")


async def test_http_back_then_front_ack_protocol(client):
    r = await client.post("/api/orders",
                          json={"order_id": "D", "prefix": "The quick brown fox",
                                "length": 40})
    assert r.status_code == 201
    await _wait_generated(client, "D", 8)

    back = await client.get("/api/orders/D/tokens?start=4&end=8")
    assert back.status_code == 200 and len(back.json()["text"]) == 4

    blocked = await client.post("/api/orders/D/ack", json={"seq": 7})
    assert blocked.status_code == 409
    info = (await client.get("/api/orders/D")).json()
    assert info["acked_seq"] == -1 and info["backlog"] == 8

    front = await client.get("/api/orders/D/tokens?start=0&end=4")
    assert front.status_code == 200
    ok = await client.post("/api/orders/D/ack", json={"seq": 7})
    assert ok.status_code == 200
    body = ok.json()
    assert body["ack_applied"] is True and body["credit_freed"] == 8
    assert body["backlog"] == 0 and body["status"] == "active"


async def test_http_empty_slice_then_ack_is_409(client):
    await client.post("/api/orders", json={"order_id": "E", "prefix": "x",
                                           "length": 10})
    await _wait_generated(client, "E", 8)
    empty = await client.get("/api/orders/E/tokens?start=0&end=0")
    assert empty.status_code == 200 and empty.json()["text"] == ""
    bad = await client.post("/api/orders/E/ack", json={"seq": 0})
    assert bad.status_code == 409


async def test_http_duplicate_and_late_ack_free_nothing(client):
    await client.post("/api/orders", json={"order_id": "U", "prefix": "x",
                                           "length": 20})
    await _wait_generated(client, "U", 8)
    assert (await client.get("/api/orders/U/tokens?start=0&end=8")).status_code == 200
    first = (await client.post("/api/orders/U/ack", json={"seq": 7})).json()
    assert first["credit_freed"] == 8
    dup = (await client.post("/api/orders/U/ack", json={"seq": 7})).json()
    late = (await client.post("/api/orders/U/ack", json={"seq": 2})).json()
    assert dup["credit_freed"] == 0 and dup["ack_applied"] is False
    assert late["credit_freed"] == 0 and late["ack_applied"] is False


# --------------------------------------------------------------------- #
# Transport completion: failed send / cancellation fill no gap
# --------------------------------------------------------------------- #
async def _ready_order(client, oid="S"):
    await client.post("/api/orders", json={"order_id": oid, "prefix": "abc",
                                          "length": 40})
    await _wait_generated(client, oid, 8)


async def test_failed_response_send_does_not_mark_delivery(client):
    await _ready_order(client, "F")
    response = await fetch_tokens("F", 0, 8)  # raw StreamingResponse

    async def send_that_fails_on_body(message):
        if message["type"] == "http.response.body":
            raise OSError("simulated broken pipe")

    with contextlib.suppress(BaseException):
        await response({"type": "http", "http_version": "1.1"}, None,
                       send_that_fails_on_body)
    await asyncio.sleep(0)

    blocked = await client.post("/api/orders/F/ack", json={"seq": 7})
    assert blocked.status_code == 409


async def test_cancelled_body_stream_does_not_mark_delivery(client):
    await _ready_order(client, "C")
    response = await fetch_tokens("C", 0, 8)
    iterator = response.body_iterator

    chunk = await iterator.__anext__()  # body handed over, mark not reached yet
    assert chunk
    await iterator.aclose()             # client gone / task cancelled
    await asyncio.sleep(0)

    blocked = await client.post("/api/orders/C/ack", json={"seq": 7})
    assert blocked.status_code == 409


# --------------------------------------------------------------------- #
# Concurrency: an in-flight response on A must not block B
# --------------------------------------------------------------------- #
async def test_other_order_fetches_acks_and_generates_while_a_sends(client):
    for oid in ("A", "B"):
        await client.post("/api/orders",
                          json={"order_id": oid, "prefix": "The quick brown fox",
                                "length": 40})
    await _wait_generated(client, "A", 8)
    await _wait_generated(client, "B", 8)

    # Open A's slice response and stop at the yielded body: it is still
    # "being sent" and has not marked delivery yet.
    response_a = await fetch_tokens("A", 0, 8)
    it_a = response_a.body_iterator
    await it_a.__anext__()

    try:
        # B fetches + confirms + keeps generating completely independently.
        got_b = await client.get("/api/orders/B/tokens?start=0&end=8")
        assert got_b.status_code == 200
        ack_b = await client.post("/api/orders/B/ack", json={"seq": 7})
        assert ack_b.status_code == 200 and ack_b.json()["credit_freed"] == 8
        b_after = await _wait_generated(client, "B", 9)
        assert b_after["generated"] >= 9

        # A is untouched (still paused at 8, unacked).
        a_info = (await client.get("/api/orders/A")).json()
        assert a_info["generated"] == 8 and a_info["acked_seq"] == -1
    finally:
        # Resume A's generator so its post-send mark runs.
        with contextlib.suppress(StopAsyncIteration):
            await it_a.__anext__()

    ack_a = await client.post("/api/orders/A/ack", json={"seq": 7})
    assert ack_a.status_code == 200 and ack_a.json()["credit_freed"] == 8
