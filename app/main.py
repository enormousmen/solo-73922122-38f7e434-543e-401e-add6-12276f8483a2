"""FastAPI application: REST API plus the annotation web UI."""

from __future__ import annotations

import contextlib
import re
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .engine import EngineError, GenerationEngine, MAX_BACKLOG, order_to_dict
from .model import TOKENS, TinyARModel

STATIC_DIR = Path(__file__).parent / "static"
ORDER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")

engine = GenerationEngine(model=TinyARModel())


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    engine.start()
    try:
        yield
    finally:
        await engine.stop()


app = FastAPI(title="Dual-Stream Autoregressive Annotation Demo", lifespan=lifespan)


@app.exception_handler(EngineError)
async def engine_error_handler(request: Request, exc: EngineError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code,
                        content={"error": type(exc).__name__, "detail": str(exc)})


class CreateOrderBody(BaseModel):
    order_id: str = Field(..., description="Unique order id, e.g. A / B")
    prefix: str = Field("", description="Text prefix to continue from")
    length: int = Field(..., gt=0, le=10_000)


class AckBody(BaseModel):
    seq: int = Field(..., ge=0, description="Highest contiguous sequence number confirmed")


def _validate_order_id(order_id: str) -> None:
    if not ORDER_ID_RE.match(order_id):
        raise EngineError("order_id may only contain letters, digits, '_' and '-' (1-32 chars)")


# --------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


# --------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------- #
@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "tick_count": engine.tick_count,
        "last_batch_size": engine.last_batch_size,
        "vocab_size": len(TOKENS),
        "max_backlog": MAX_BACKLOG,
    }


@app.post("/api/orders", status_code=201)
async def create_order(body: CreateOrderBody) -> dict:
    _validate_order_id(body.order_id)
    order = await engine.create_order(body.order_id, body.prefix, body.length)
    return order_to_dict(order)


@app.get("/api/orders")
async def list_orders() -> dict:
    orders = await engine.list_orders()
    return {"orders": [order_to_dict(o) for o in orders]}


@app.get("/api/orders/{order_id}")
async def get_order(order_id: str) -> dict:
    return order_to_dict(await engine.get_order(order_id))


@app.get("/api/orders/{order_id}/tokens")
async def fetch_tokens(order_id: str, start: int = 0, end: int | None = None):
    if end is None:
        end = start + 1
    # Build the slice under the engine lock, then release the lock before any
    # network I/O so a slow/half-open response cannot block the other order,
    # acks or the generation ticker.
    payload = await engine.prepare_slice(order_id, start, end)
    body = JSONResponse(payload).body
    seq_start, seq_end = payload["start"], payload["end"]

    async def body_iter():
        # Yield the whole JSON body once.  Everything after the yield runs
        # only when Starlette resumes this generator after the chunk was sent
        # successfully:
        #   * normal completion  -> resume here, record delivery;
        #   * send error / client disconnect / task cancellation -> the
        #     generator is closed at the yield point (GeneratorExit /
        #     CancelledError), so the mark below never runs.  An unfinished
        #     response therefore cannot fill an acknowledgement gap.
        yield body
        await engine.mark_delivered(order_id, seq_start, seq_end)

    return StreamingResponse(
        body_iter(),
        media_type="application/json",
        headers={
            "Cache-Control": "no-store",
            "Content-Length": str(len(body)),
        },
    )


@app.post("/api/orders/{order_id}/ack")
async def acknowledge(order_id: str, body: AckBody) -> dict:
    return await engine.acknowledge(order_id, body.seq)
