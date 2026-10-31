"""What flock-verify's verdict reports of the `--registered` file it verified against (`Flock.Registered.finding`): each
registered port's value, the root its reads were opened against, and that root's domain and shape."""
from __future__ import annotations

from typing import Any, Mapping


def finding(ports: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """The finding of a file's ``ports``: a row's words null where the port names none, and their bits 16 unless named."""
    return {p: {"value": e["value"], "root": e["root"], "domain": e["domain"], "leaves": e["leaves"],
                "words": e.get("words"), "word_bits": e.get("word_bits", 16)} for p, e in ports.items()}


def opened(registered: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """A finding on each port's value and root."""
    return {p: {"value": e["value"], "root": e["root"]} for p, e in registered.items()}
