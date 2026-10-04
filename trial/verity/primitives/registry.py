"""Constructions by name. A subprotocol lists the constructions that ship with it; one with heavy dependencies ships in
its own distribution and registers under the entry-point group `verity.constructions` as `<subprotocol>/<name>`, so it
is loaded only when a configuration names it, and its dependencies are installed only where it is wanted."""

from __future__ import annotations

from collections.abc import Mapping
from importlib.metadata import entry_points

GROUP = "verity.constructions"


class NotInstalled(LookupError):
    pass


def load(subprotocol: str, name: str, shipped: Mapping[str, type]) -> object:
    if name in shipped:
        return shipped[name]()
    key = f"{subprotocol}/{name}"
    found = [ep for ep in entry_points(group=GROUP) if ep.name == key]
    if not found:
        registered = [ep.name.split("/", 1)[1] for ep in entry_points(group=GROUP) if ep.name.startswith(f"{subprotocol}/")]
        raise NotInstalled(f"{key} is not installed (available: {', '.join(sorted([*shipped, *registered]))})")
    return found[0].load()()
