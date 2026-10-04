"""Constructions by name: what the configuration names, found among what is installed.

Every construction registers itself under the entry-point group `verity.constructions` as `<subprotocol>/<name>` (the
commitment scheme as `commitments/<name>`, the challenge source as `challenges/<name>`). The protocol never imports a
construction: it loads the one the configuration names, so a construction that is not installed is a refusal, and its
dependencies are never pulled into a deployment that does not name it.
"""

from __future__ import annotations

from importlib.metadata import entry_points

GROUP = "verity.constructions"


def installed() -> tuple[str, ...]:
    return tuple(sorted(ep.name for ep in entry_points(group=GROUP)))


def load(key: str) -> object | None:
    """The construction registered as `key`, or None when nothing installed registers it."""
    found = [ep for ep in entry_points(group=GROUP) if ep.name == key]
    if not found:
        return None
    if len(found) > 1:
        raise LookupError(f"{key}: registered by {len(found)} distributions")
    return found[0].load()()
