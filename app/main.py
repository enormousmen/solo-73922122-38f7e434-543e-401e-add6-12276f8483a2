"""FastAPI application: REST API plus the annotation web UI."""

from __future__ import annotations

import contextlib
import re
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
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
async def fetch_tokens(order_id: str, start: int = 0, end: int | None = None) -> dict:
    if end is None:
        end = start + 1
    return await engine.fetch(order_id, start, end)


@app.post("/api/orders/{order_id}/ack")
async def acknowledge(order_id: str, body: AckBody) -> dict:
    return await engine.acknowledge(order_id, body.seq)
