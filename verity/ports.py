from __future__ import annotations

import socket

__all__ = ["free_port"]


def free_port() -> int:
    """A loopback port the OS has free now."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
