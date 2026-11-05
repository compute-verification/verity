"""The network certifier as the integration reads it: an interface the integration owns and the certifier implements.

Each part is a `Protocol` below, listing exactly what the integration reads of it, and its accessor (`active`,
`io_tie`, `release`, `ingress_scan`, `compile_slice`, `served_rows`) gives the implementation installed for it: the entry
point of that name in the ``verity_vllm.certifier`` group, which the certifier's side declares. Nothing here names the
certifier's modules, and the certifier never imports the integration. Where a part has no implementation installed, or
more than one, its accessor refuses (`errors.Refused`), and only the paths that serve the network certifier need one:
the ``ids`` protocol (`pipeline.serve_protocol.IdsProtocol`) and the scan state (`program.predict.scan_state`)."""
from __future__ import annotations

import functools
import struct
from collections.abc import Callable
from importlib.metadata import entry_points
from typing import Any, Protocol

from verity_vllm.errors import Refused

GROUP = "verity_vllm.certifier"


class Active(Protocol):
    """The certifier's id requests: the link frame's ``HEADER`` and ``id_request``, a canonical token-id request."""

    HEADER: struct.Struct
    id_request: Callable[..., bytes]


class IoTie(Protocol):
    """The scan's egress: ``id_event(token_id, size)``, one token id as a ``size``-byte event."""

    id_event: Callable[[int, int], bytes]


class Release(Protocol):
    """The release headers the certifier puts on the server's copy of a request."""

    RELEASE_HEADER: bytes
    RUN_HEADER: bytes


class IngressScan(Protocol):
    """The certifier's ingress scan: its class and state, the registered value of its state rows (``SCAN_STATE``), and the
    scan, its state row's layout and encoding, and a window's rows of it."""

    Class: type
    State: type
    SCAN_STATE: str
    empty: Callable[..., Any]
    encode: Callable[..., list[int]]
    layout: Callable[..., dict[str, tuple[int, int]]]
    position: Callable[..., int]
    scan: Callable[..., Any]
    state_bits: Callable[..., int]
    window_state: Callable[..., list[list[int]]]


class CompileSlice(Protocol):
    """The served slice's ``statics``."""

    statics: Callable[..., Any]


class ServedRows(Protocol):
    """The served rows: a session's ``windows`` and their registration (``register``, ``Served``)."""

    Served: type
    register: Callable[..., Any]
    windows: Callable[..., Any]


@functools.cache
def _part(name: str) -> Any:
    eps = list(entry_points(group=GROUP, name=name))
    if len(eps) != 1:
        found = ", ".join(f"{ep.value} from {ep.dist.name if ep.dist else '?'}" for ep in eps) or "none"
        raise Refused(f"the {GROUP!r} entry point group has {len(eps)} {name!r} parts ({found}), not one: install exactly "
                      "one network certifier, or serve without it", what=f"verity_vllm.ports.{name}")
    return eps[0].load()


def active() -> Active:
    return _part("active")


def io_tie() -> IoTie:
    return _part("io_tie")


def release() -> Release:
    return _part("release")


def ingress_scan() -> IngressScan:
    return _part("ingress_scan")


def compile_slice() -> CompileSlice:
    return _part("compile_slice")


def served_rows() -> ServedRows:
    return _part("served_rows")


__all__ = ["GROUP", "Active", "CompileSlice", "IngressScan", "IoTie", "Release", "ServedRows", "active", "compile_slice",
           "ingress_scan", "io_tie", "release", "served_rows"]
