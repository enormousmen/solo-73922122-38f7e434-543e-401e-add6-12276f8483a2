"""Engine-level tests: batching, 8-token backlog, pause/resume, ack rules."""

import pytest

from app.engine import (
    AckRejected,
    EngineError,
    GenerationEngine,
    MAX_BACKLOG,
    OrderNotFound,
    SeqOutOfRange,
)
from app.model import TinyARModel


@pytest.fixture()
def gen_engine():
    return GenerationEngine(model=TinyARModel(), tick_interval=0.01)


async def run_ticks(eng, n):
    for _ in range(n):
        await eng.step()


async def test_backlog_cap_pauses_order_and_other_keeps_generating(gen_engine):
    await gen_engine.create_order("A", "hello", length=40)
    await gen_engine.create_order("B", "world", length=40)

    # First 8 ticks: both have budget -> batch size 2, generation in lockstep
    for _ in range(MAX_BACKLOG):
        assert await gen_engine.step() == 2
    a, b = gen_engine.orders["A"], gen_engine.orders["B"]
    assert a.generated == 8 and b.generated == 8
    assert a.backlog == 8 and a.status.value == "paused"

    # A stops reading and never acks.  B keeps acknowledging, so it retains
    # budget and continues alone while A stays frozen at its old progress.
    for _ in range(5):
        # New protocol: the slice must be returned before its tail is ackable.
        await gen_engine.fetch("B", 0, b.generated)
        await gen_engine.acknowledge("B", b.generated - 1)
        assert await gen_engine.step() == 1
    a, b = gen_engine.orders["A"], gen_engine.orders["B"]
    assert a.generated == 8, "paused order must not advance"
    assert b.generated == 13 and b.status.value in ("active", "paused")
    assert a.status.value == "paused"


async def test_ack_frees_credit_and_resumes_from_same_progress(gen_engine):
    await gen_engine.create_order("A", "hello", length=30)
    await run_ticks(gen_engine, 8)
    a = gen_engine.orders["A"]
    frozen_tokens = list(a.tokens)
    assert a.status.value == "paused"

    # New protocol: seq 0..3 must be returned before they are acknowledged.
    await gen_engine.fetch("A", 0, 4)
    # Acknowledge the first 4 contiguous tokens (seq 0..3)
    info = await gen_engine.acknowledge("A", 3)
    assert info["backlog"] == 4
    assert info["status"] == "active"

    for _ in range(4):
        assert await gen_engine.step() == 1
    a = gen_engine.orders["A"]
    # Pre-pause tokens are untouched and new tokens simply extend them
    assert a.tokens[:8] == frozen_tokens
    assert a.generated == 12
    assert a.backlog == 8  # 4 remaining old + 4 new
    assert a.status.value == "paused"


async def test_ack_beyond_delivered_high_water_is_rejected(gen_engine):
    await gen_engine.create_order("A", "x", length=20)
    await run_ticks(gen_engine, 3)  # generated seq 0..2
    with pytest.raises(AckRejected):
        await gen_engine.acknowledge("A", 5)
    with pytest.raises(AckRejected):
        await gen_engine.acknowledge("A", 99)
    # rejected ack must not mutate state
    assert gen_engine.orders["A"].acked_seq == -1


async def test_duplicate_and_late_acks_free_no_extra_credit(gen_engine):
    await gen_engine.create_order("A", "x", length=20)
    await run_ticks(gen_engine, 8)
    await gen_engine.fetch("A", 0, 4)  # deliver seq 0..3 before acking them
    first = await gen_engine.acknowledge("A", 3)
    assert first["credit_freed"] == 4 and first["ack_applied"] is True

    duplicate = await gen_engine.acknowledge("A", 3)
    assert duplicate["credit_freed"] == 0 and duplicate["ack_applied"] is False

    older = await gen_engine.acknowledge("A", 1)  # late ack behind high-water
    assert older["credit_freed"] == 0 and older["ack_applied"] is False
    assert gen_engine.orders["A"].acked_seq == 3
    assert gen_engine.orders["A"].backlog == 4


async def test_refetch_returns_identical_bytes(gen_engine):
    await gen_engine.create_order("A", "repeatable", length=16)
    await run_ticks(gen_engine, 6)
    r1 = await gen_engine.fetch("A", 0, 6)
    r2 = await gen_engine.fetch("A", 2, 5)
    r1_again = await gen_engine.fetch("A", 0, 6)
    assert r1["text"] == r1_again["text"]
    assert r2["text"] == r1["text"][2:5]
    # partial range that extends beyond what is generated is clipped
    r3 = await gen_engine.fetch("A", 4, 999)
    assert r3["end"] == 6 and r3["text"] == r1["text"][4:]


async def test_fetch_not_yet_generated_seq_raises(gen_engine):
    await gen_engine.create_order("A", "x", length=10)
    with pytest.raises(SeqOutOfRange):
        await gen_engine.fetch("A", 0, 1)  # nothing generated yet
    await run_ticks(gen_engine, 2)
    with pytest.raises(SeqOutOfRange):
        await gen_engine.fetch("A", 5, 6)


async def test_unknown_order_and_bad_inputs(gen_engine):
    with pytest.raises(OrderNotFound):
        await gen_engine.get_order("NOPE")
    await gen_engine.create_order("A", "x", length=5)
    with pytest.raises(EngineError):
        await gen_engine.create_order("A", "y", length=5)  # duplicate id
    with pytest.raises(EngineError):
        await gen_engine.create_order("C", "z", length=0)


async def test_order_completes_and_stops_consuming_budget(gen_engine):
    await gen_engine.create_order("A", "end", length=5)
    await run_ticks(gen_engine, 5)
    a = gen_engine.orders["A"]
    assert a.generated == 5 and a.status.value == "done"
    assert await gen_engine.step() == 0
    assert a.generated == 5


async def test_two_orders_batched_generation_matches_reference_decode(gen_engine):
    """Batched engine output must equal single-stream greedy reference.

    Both orders acknowledge as they go so the 8-token backlog never pauses
    them; the point of this test is numerical equivalence of the batched
    forward pass, not back-pressure (covered elsewhere).
    """
    model = gen_engine.model
    ref_state = model.new_state("fox")
    ref = [model.generate_batch([ref_state])[0] for _ in range(10)]

    await gen_engine.create_order("A", "fox", length=10)
    await gen_engine.create_order("B", "other", length=10)
    for k in range(10):
        assert await gen_engine.step() == 2
        # New protocol: deliver seq k, then acknowledge it, so credit frees
        # continuously and neither order pauses.
        await gen_engine.fetch("A", k, k + 1)
        await gen_engine.fetch("B", k, k + 1)
        await gen_engine.acknowledge("A", k)
        await gen_engine.acknowledge("B", k)
    assert gen_engine.orders["A"].tokens == ref
