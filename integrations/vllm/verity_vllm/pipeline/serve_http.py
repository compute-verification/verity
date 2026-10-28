"""serve-openai's HTTP side: its `serve_openai.Server` in an asyncio event loop on a thread of its own (`HttpThread`), one
request per connection, while the process's main thread steps the engine."""
from __future__ import annotations

import asyncio
import threading
from typing import Any


class HttpThread(threading.Thread):
    """``server`` (`serve_openai.Server`) on ``host:port`` in this thread's own event loop: ``ready`` is set once it listens
    (``address``), and `shutdown` stops it from any thread."""

    def __init__(self, server: Any, host: str, port: int) -> None:
        super().__init__(name="http", daemon=True)
        self.server, self.host, self.port = server, host, port
        self.ready = threading.Event()
        self.address: tuple[str, int] | None = None
        self.error: BaseException | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._done: asyncio.Event | None = None

    def run(self) -> None:
        async def main() -> None:
            self._loop, self._done = asyncio.get_running_loop(), asyncio.Event()
            srv = await asyncio.start_server(self.server.handle, self.host, self.port, limit=1 << 22, backlog=256)
            self.address = tuple(srv.sockets[0].getsockname()[:2])
            self.ready.set()
            async with srv:
                await self._done.wait()

        try:
            asyncio.run(main())
        except BaseException as e:
            self.error = e
            self.ready.set()

    def shutdown(self) -> None:
        if self._loop is not None and self._done is not None:
            self._loop.call_soon_threadsafe(self._done.set)
        self.join()
