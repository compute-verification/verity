from __future__ import annotations

import re
from typing import Any

from . import canon

HEX64 = r"[0-9a-f]{64}"
ART_RE = re.compile(rf"^art:{HEX64}$")
SHORT_ART_MIN_HEX = 8
SHORT_ART_RE = re.compile(rf"^art:[0-9a-f]{{{SHORT_ART_MIN_HEX},63}}$")
DRV_RE = re.compile(rf"^drv:{HEX64}$")
RUN_RE = re.compile(r"^r\d{8}-\d{6}-[0-9a-f]{4}$")
PR_RE = re.compile(r"^pr:[1-9]\d*@[0-9a-f]{40}$")
CKPT_RE = re.compile(r"^ckpt:[A-Za-z0-9][\w.-]*/[\w.-]+@[0-9a-f]{40}$")
SHA_RE = re.compile(rf"^{HEX64}$")
TASK_RE = re.compile(r"^task:[0-9a-f]{128}$")
COMMIT_RE = re.compile(r"^commit:[0-9a-f]{40}$")


def art_id(manifest: Any) -> str:
    doc = manifest.to_doc() if hasattr(manifest, "to_doc") else manifest
    return "art:" + canon.sha256(doc)


def drv_id(derivation_doc: dict) -> str:
    return "drv:" + canon.sha256(derivation_doc)


def is_art(s: Any) -> bool:
    return isinstance(s, str) and bool(ART_RE.match(s))


def is_short_art(s: Any) -> bool:
    return isinstance(s, str) and bool(SHORT_ART_RE.match(s))


def is_drv(s: Any) -> bool:
    return isinstance(s, str) and bool(DRV_RE.match(s))


def is_run_id(s: Any) -> bool:
    return isinstance(s, str) and bool(RUN_RE.match(s))


def is_pr(s: Any) -> bool:
    return isinstance(s, str) and bool(PR_RE.match(s))


def is_ckpt(s: Any) -> bool:
    return isinstance(s, str) and bool(CKPT_RE.match(s)) and "--" not in s


def is_task(s: Any) -> bool:
    return isinstance(s, str) and bool(TASK_RE.match(s))


def is_commit(s: Any) -> bool:
    return isinstance(s, str) and bool(COMMIT_RE.match(s))


def is_sha256(s: Any) -> bool:
    return isinstance(s, str) and bool(SHA_RE.match(s))


def hex_of(art_or_drv: str) -> str:
    return art_or_drv.split(":", 1)[1]


def target_dirname(target: str) -> str:
    return target.replace(":", "_").replace("/", "--")


def kind_of_id(s: str) -> str:
    if is_art(s):
        return "art"
    if is_drv(s):
        return "drv"
    if is_run_id(s):
        return "run"
    if is_pr(s):
        return "pr"
    if is_ckpt(s):
        return "ckpt"
    if is_task(s):
        return "task"
    if is_commit(s):
        return "commit"
    return "unknown"
