"""Identifier syntax (README §1): `art:<sha256 hex>`, `drv:<sha256 hex>`, attempt id == run id (`r20260922-105905-2021`).

The store checks syntax, never existence: a ref may name an artifact that lives only remotely.
"""
from __future__ import annotations

import re
from typing import Any

from . import canon

HEX64 = r"[0-9a-f]{64}"
ART_RE = re.compile(rf"^art:{HEX64}$")
DRV_RE = re.compile(rf"^drv:{HEX64}$")
RUN_RE = re.compile(r"^r\d{8}-\d{6}-[0-9a-f]{4}$")
SHA_RE = re.compile(rf"^{HEX64}$")


def art_id(manifest: Any) -> str:
    """`art:` + sha256 of the canonical manifest document (a dict, or anything with `to_doc()`)."""
    doc = manifest.to_doc() if hasattr(manifest, "to_doc") else manifest
    return "art:" + canon.sha256(doc)


def drv_id(derivation_doc: dict) -> str:
    return "drv:" + canon.sha256(derivation_doc)


def is_art(s: Any) -> bool:
    return isinstance(s, str) and bool(ART_RE.match(s))


def is_drv(s: Any) -> bool:
    return isinstance(s, str) and bool(DRV_RE.match(s))


def is_run_id(s: Any) -> bool:
    return isinstance(s, str) and bool(RUN_RE.match(s))


def is_sha256(s: Any) -> bool:
    return isinstance(s, str) and bool(SHA_RE.match(s))


def hex_of(art_or_drv: str) -> str:
    """The 64 hex digits after the prefix."""
    return art_or_drv.split(":", 1)[1]


def target_dirname(target: str) -> str:
    """Label directory name for a target id: ':' -> '_' (art:abc -> art_abc; run ids are unchanged)."""
    return target.replace(":", "_")


def kind_of_id(s: str) -> str:
    """'art' | 'drv' | 'run' | 'unknown'."""
    if is_art(s):
        return "art"
    if is_drv(s):
        return "drv"
    if is_run_id(s):
        return "run"
    return "unknown"
