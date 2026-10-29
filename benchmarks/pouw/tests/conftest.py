import asyncio
import threading

import pytest
from verity.core.protocols.network_certifier import active


@pytest.fixture(autouse=True)
def _outside_any_run(monkeypatch):
    """A benchmark's `run_result.write` writes into ``$RESEARCH_RUN_DIR``: under a `research run` of the suites that is the
    run's own result.json, which a test's verdict must never replace."""
    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)


@pytest.fixture
def reveals():
    """The network certifier's reveal endpoint for the test's proxies: `reveals(proxy)` serves it (`serve_reveals`) on
    localhost, on a loop of its own thread, and returns its (host, port), so a synchronous challenger asks it through a
    `RemoteCertifier`. Every server closes, and the loop stops, when the test ends."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    servers: list[asyncio.Server] = []

    def serve(proxy: active.Proxy) -> tuple[str, int]:
        servers.append(asyncio.run_coroutine_threadsafe(active.serve_reveals(proxy, "127.0.0.1", 0), loop).result())
        return servers[-1].sockets[0].getsockname()[:2]

    async def close() -> None:
        for s in servers:
            s.close()
            await s.wait_closed()

    try:
        yield serve
    finally:
        asyncio.run_coroutine_threadsafe(close(), loop).result()
        loop.call_soon_threadsafe(loop.stop)
        thread.join()
        loop.close()
