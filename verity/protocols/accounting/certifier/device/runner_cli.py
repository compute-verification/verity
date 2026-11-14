"""A runner's command line: the HOST:PORT its flags take, its pacer's default, and the lines it prints of itself."""

from __future__ import annotations

import json

#: when the proxy's pacer writes each egress bucket, in ms after its boundary: twice the proxy's slack, so that a tick
#: in time has handed it before
PACED_MS = 50.0


def _announce(line: dict) -> None:
    print(json.dumps(line), flush=True)


def _address(text: str) -> tuple[str, int]:
    host, _, port = text.rpartition(":")
    return host, int(port)
