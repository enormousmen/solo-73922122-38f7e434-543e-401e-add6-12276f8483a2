import httpx
import pytest
import pytest_asyncio

from app.main import app, engine


@pytest_asyncio.fixture()
async def running_engine():
    """Reset global engine state and run the background ticker (API tests)."""
    await engine.stop()
    engine.orders.clear()
    engine.tick_count = 0
    engine.last_batch_size = 0
    engine.start()
    try:
        yield
    finally:
        await engine.stop()
        engine.orders.clear()


@pytest_asyncio.fixture()
async def client(running_engine):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport,
                                 base_url="http://test") as c:
        yield c
