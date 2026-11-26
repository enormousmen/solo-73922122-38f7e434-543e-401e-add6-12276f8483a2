"""Generation engine: orders, back-pressure credits and batched scheduling.

Two annotation orders are served concurrently.  A background ticker wakes up
at a fixed cadence; every tick, every order that

* still has tokens left to generate **and**
* currently has fewer than ``MAX_BACKLOG`` unacknowledged tokens,

is advanced by exactly one decoding step.  When both orders are eligible they
are decoded together in one batched NumPy forward pass.

An order whose backlog is full is *paused*: it is skipped until the annotator
acknowledges tokens, freeing credit.  Pausing never discards progress — when
the order resumes, generation simply continues from the exact model state it
reached before the pause.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum

from .model import ModelState, TinyARModel

logger = logging.getLogger("generation")

MAX_BACKLOG = 8
TICK_INTERVAL = 0.12  # seconds between generation steps


class OrderStatus(str, Enum):
    ACTIVE = "active"      # has budget and/or remaining tokens to produce
    PAUSED = "paused"      # backlog full, waiting for acknowledgements
    DONE = "done"          # requested length fully generated


@dataclass
class Order:
    order_id: str
    total: int
    tokens: list[int] = field(default_factory=list)          # seq -> token id
    acked_seq: int = -1                                       # highest contiguous ack
    state: ModelState | None = None
    last_batch_size: int = 0                                 # diagnostics

    @property
    def generated(self) -> int:
        return len(self.tokens)

    @property
    def backlog(self) -> int:
        """Number of delivered-but-unacknowledged tokens."""
        return self.generated - (self.acked_seq + 1)

    @property
    def has_budget(self) -> bool:
        return self.backlog < MAX_BACKLOG

    @property
    def has_remaining(self) -> bool:
        return self.generated < self.total

    @property
    def status(self) -> OrderStatus:
        if self.has_remaining:
            return OrderStatus.ACTIVE if self.has_budget else OrderStatus.PAUSED
        return OrderStatus.DONE


class EngineError(Exception):
    """Base error; ``status_code`` mirrors the HTTP status to return."""

    status_code = 400


class OrderNotFound(EngineError):
    status_code = 404


class AckRejected(EngineError):
    """Ack points past the highest contiguous sequence already delivered."""

    status_code = 409


class SeqOutOfRange(EngineError):
    status_code = 404


class GenerationEngine:
    def __init__(self, model: TinyARModel | None = None,
                 tick_interval: float = TICK_INTERVAL) -> None:
        self.model = model or TinyARModel()
        self.tick_interval = tick_interval
        self.orders: dict[str, Order] = {}
        self._lock: asyncio.Lock = asyncio.Lock()
        self._task: asyncio.Task | None = None
        self._stop: asyncio.Event = asyncio.Event()
        self.tick_count = 0
        self.last_batch_size = 0

    # --------------------------------------------------------------- #
    # Lifecycle
    # --------------------------------------------------------------- #
    def start(self) -> None:
        # Recreate loop-bound primitives when re-entering on a different event
        # loop (pytest-asyncio gives every test a brand-new loop): an
        # Event/Lock/Task bound to a closed loop must not be reused.
        if self._task is not None and not self._task.done():
            return
        self._task = None
        self._lock = asyncio.Lock()
        self._stop = asyncio.Event()
        self._task = asyncio.create_task(self._run(), name="generation-ticker")

    async def stop(self) -> None:
        if self._task is not None:
            self._stop.set()
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None

    async def _run(self) -> None:
        try:
            while not self._stop.is_set():
                await self.step()
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self.tick_interval)
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            raise

    async def step(self) -> int:
        """One scheduler tick. Returns the batch size used (0, 1 or 2)."""
        async with self._lock:
            eligible = [o for o in self.orders.values()
                        if o.has_remaining and o.has_budget]
            if not eligible:
                self.last_batch_size = 0
                return 0
            next_ids = self.model.generate_batch([o.state for o in eligible])
            for order, token_id in zip(eligible, next_ids):
                order.tokens.append(token_id)
                order.last_batch_size = len(eligible)
            self.last_batch_size = len(eligible)
            self.tick_count += 1
            logger.info("tick=%d batch=%d orders=%s",
                        self.tick_count, len(eligible),
                        {o.order_id: o.generated for o in eligible})
            return len(eligible)

    # --------------------------------------------------------------- #
    # Public API (all mutating operations hold the lock)
    # --------------------------------------------------------------- #
    async def create_order(self, order_id: str, prefix: str, length: int) -> Order:
        if length <= 0:
            raise EngineError("length must be a positive integer")
        if len(prefix) > 4096:
            raise EngineError("prefix too long (max 4096 chars)")
        async with self._lock:
            if order_id in self.orders:
                raise EngineError(f"order {order_id!r} already exists")
            order = Order(order_id=order_id, total=length,
                          state=self.model.new_state(prefix))
            self.orders[order_id] = order
            return self._snapshot(order)

    async def fetch(self, order_id: str, start: int, end: int) -> dict:
        """Return tokens with sequence numbers in ``[start, end)``.

        The fragment is served from the retained token buffer, so fetching a
        range again returns byte-identical text (re-fetch of unacked data).
        """
        async with self._lock:
            order = self._require_order(order_id)
            if start < 0 or end < start:
                raise EngineError("invalid range: require 0 <= start <= end")
            available = order.generated
            if start >= available:
                raise SeqOutOfRange(
                    f"seq {start} not generated yet (have 0..{available - 1})")
            end = min(end, available)
            ids = order.tokens[start:end]
            return {
                "order_id": order_id,
                "start": start,
                "end": end,
                "text": self.model.decode(ids),
                "next_seq": end,
                "generated": available,
                "total": order.total,
                "status": order.status.value,
                "replayed": True,  # identical bytes on every call
            }

    async def acknowledge(self, order_id: str, seq: int) -> dict:
        async with self._lock:
            order = self._require_order(order_id)
            delivered_max = order.generated - 1
            if seq > delivered_max:
                # Refuse to acknowledge beyond the contiguous high-water mark.
                raise AckRejected(
                    f"ack seq {seq} exceeds max delivered seq {delivered_max}")
            before = order.acked_seq
            applied = seq > before
            if applied:
                # New contiguous acknowledgement: free credit.
                order.acked_seq = seq
            # Duplicate (seq <= before) or late ack frees nothing.
            result = order_to_dict(order)
            result["acked_seq_before"] = before
            result["ack_applied"] = applied
            result["credit_freed"] = order.acked_seq - before
            return result

    async def get_order(self, order_id: str) -> Order:
        async with self._lock:
            return self._snapshot(self._require_order(order_id))

    async def list_orders(self) -> list[Order]:
        async with self._lock:
            return [self._snapshot(o) for o in self.orders.values()]

    # --------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------- #
    def _require_order(self, order_id: str) -> Order:
        order = self.orders.get(order_id)
        if order is None:
            raise OrderNotFound(f"unknown order {order_id!r}")
        return order

    def _snapshot(self, order: Order) -> Order:
        return Order(
            order_id=order.order_id,
            total=order.total,
            tokens=list(order.tokens),
            acked_seq=order.acked_seq,
            state=None,  # model state is never exposed
            last_batch_size=order.last_batch_size,
        )


def order_to_dict(order: Order) -> dict:
    return {
        "order_id": order.order_id,
        "total": order.total,
        "generated": order.generated,
        "acked_seq": order.acked_seq,
        "backlog": order.backlog,
        "max_backlog": MAX_BACKLOG,
        "status": order.status.value,
        "last_batch_size": order.last_batch_size,
    }
