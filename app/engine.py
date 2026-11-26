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
    # Delivery ledger, bounded to the MAX_BACKLOG unconfirmed window.  Bit i
    # means "seq acked_seq+1+i was carried by a response the server finished
    # sending successfully".  It is an 8-bit mask (one slot per credit), never
    # a structure that grows with ``total``.
    delivered_mask: int = 0

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

    # ----------------------------------------------------------- #
    # Delivery bookkeeping (constant-size, one bit per credit)
    # ----------------------------------------------------------- #
    def mark_delivered(self, start: int, end: int) -> None:
        """Record absolute seq numbers in ``[start, end)`` as fully sent.

        Only seqs inside the current unconfirmed window have a slot.  Seqs at
        or below the acked watermark are already confirmed and irrelevant;
        seqs beyond the window cannot occur because the back-pressure cap stops
        generation exactly at the window boundary.
        """
        base = self.acked_seq + 1
        for seq in range(max(start, base), end):
            idx = seq - base
            if 0 <= idx < MAX_BACKLOG:
                self.delivered_mask |= 1 << idx

    def can_ack_through(self, seq: int) -> bool:
        """Whether ``acked_seq+1 .. seq`` are all on record as delivered."""
        count = seq - self.acked_seq  # number of seqs that would be newly acked
        if count <= 0:
            return True
        if count > MAX_BACKLOG:
            return False
        low_bits = (1 << count) - 1
        return (self.delivered_mask & low_bits) == low_bits

    def advance_watermark(self, seq: int) -> None:
        """Move the acked watermark to ``seq`` and align the delivery window.

        The low ``count`` bits are the contiguous prefix just confirmed.
        Shifting them out keeps any out-of-order tail (a back slice received
        early) recorded against the new base.
        """
        count = seq - self.acked_seq
        if count > 0:
            self.delivered_mask >>= count
            self.acked_seq = seq

    @property
    def delivered_seqs(self) -> list[int]:
        """Absolute seqs currently marked delivered (at most MAX_BACKLOG)."""
        base = self.acked_seq + 1
        return [base + i for i in range(MAX_BACKLOG)
                if self.delivered_mask & (1 << i)]

    @property
    def delivered(self) -> set[int]:
        """Set view of :attr:`delivered_seqs` (bounded to the window)."""
        return set(self.delivered_seqs)

    @property
    def delivery(self) -> list[bool]:
        """Fixed ``MAX_BACKLOG``-slot view of the window (bit i -> base+i)."""
        return [bool(self.delivered_mask & (1 << i))
                for i in range(MAX_BACKLOG)]

    @property
    def delivered_contig_seq(self) -> int:
        """Highest seq reachable by a contiguous delivered prefix.

        Equals ``acked_seq`` when the very next seq is missing.  An ack may
        only advance up to this sequence number.
        """
        run = 0
        bits = self.delivered_mask
        while run < MAX_BACKLOG and (bits & 1):
            run += 1
            bits >>= 1
        return self.acked_seq + run


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

    def _slice_locked(self, order: Order, start: int, end: int) -> dict:
        """Build the slice payload. Caller holds ``self._lock``."""
        if start < 0 or end < start:
            raise EngineError("invalid range: require 0 <= start <= end")
        available = order.generated
        if start >= available:
            raise SeqOutOfRange(
                f"seq {start} not generated yet (have 0..{available - 1})")
        end = min(end, available)
        ids = order.tokens[start:end]
        return {
            "order_id": order.order_id,
            "start": start,
            "end": end,
            "text": self.model.decode(ids),
            "next_seq": end,
            "generated": available,
            "total": order.total,
            "status": order.status.value,
            "replayed": True,  # identical bytes on every call
        }

    async def prepare_slice(self, order_id: str, start: int, end: int) -> dict:
        """Validate and build a slice WITHOUT recording delivery.

        The HTTP layer calls this under a short lock, sends the body without
        holding the lock, and only then calls :meth:`mark_delivered` if the
        whole body was sent successfully.
        """
        async with self._lock:
            order = self._require_order(order_id)
            return self._slice_locked(order, start, end)

    # Convenience aliases for the two-phase delivery protocol.
    take_slice = prepare_slice

    async def mark_delivered(self, order_id: str, start: int, end: int) -> dict:
        """Record ``[start, end)`` as carried by a fully-sent response."""
        async with self._lock:
            order = self.orders.get(order_id)
            if order is None:
                raise OrderNotFound(f"unknown order {order_id!r}")
            order.mark_delivered(start, end)
            return order_to_dict(order)

    commit_delivery = mark_delivered
    record_delivery = mark_delivered
    mark_sent = mark_delivered

    async def fetch(self, order_id: str, start: int, end: int) -> dict:
        """Return tokens with sequence numbers in ``[start, end)``.

        The fragment is served from the retained token buffer, so fetching a
        range again returns byte-identical text (re-fetch of unacked data).

        For in-process callers receiving the whole dict at once, a successful
        return also records the slice as delivered.  The HTTP transport uses
        :meth:`prepare_slice` + :meth:`mark_delivered` instead so that a failed
        or cancelled network send is not counted.
        """
        async with self._lock:
            order = self._require_order(order_id)
            payload = self._slice_locked(order, start, end)
            order.mark_delivered(payload["start"], payload["end"])
            return payload

    async def acknowledge(self, order_id: str, seq: int) -> dict:
        async with self._lock:
            order = self._require_order(order_id)
            before = order.acked_seq

            if seq <= before:
                # Duplicate (seq == before) or late/smaller ack: idempotent,
                # frees no extra credit and is never a 409.
                applied = False
            else:
                generated_max = order.generated - 1
                if seq > generated_max:
                    # Nothing at that sequence number exists yet.
                    raise AckRejected(
                        f"ack seq {seq} exceeds max generated seq {generated_max}")
                if not order.can_ack_through(seq):
                    # The seqs after the last ack must have been *continuously
                    # returned* (fully delivered) up to ``seq``.  A gap means
                    # the annotator only holds an out-of-order back slice, so
                    # confirming its tail must not free the missing front's
                    # credit.  State is left untouched.
                    have = set(order.delivered_seqs)
                    missing = [s for s in range(before + 1, seq + 1)
                               if s not in have]
                    raise AckRejected(
                        f"ack seq {seq} refused: seqs {before + 1}..{seq} were "
                        f"not all continuously delivered (contiguous delivered "
                        f"through #{order.delivered_contig_seq}); missing "
                        f"{missing[:8]}")
                order.advance_watermark(seq)
                applied = True

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
            delivered_mask=order.delivered_mask,
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
        # Highest seq that may currently be acked without a 409, plus the
        # bounded set of out-of-order seqs already on record as delivered.
        "delivered_contig_seq": order.delivered_contig_seq,
        "delivered_seqs": order.delivered_seqs,
    }
