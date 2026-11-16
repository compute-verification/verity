"""Finding the output a layer's benchmark result is read from (`proof_layers`): a run's ``out/`` directory, or a
run id's record fetched through the evidence store (`fetch`), and which command wrote it (`detect`)."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

from verity_bench.proof_layers_result import (
    A5,
    FETCH,
    FIX,
    LAYERS_SH,
    PROVE_PATH,
    RECURSION,
    RUN_ID,
    Unreadable,
)

# ---- finding a benchmark result's output --------------------------------------------------------------------------------


def fetch(run: str) -> Path:
    """The root of `run`'s record as the evidence store holds it, with `FETCH`'s files fetched."""

    try:
        from research.store import labels_sync
        from research.store.local import LocalStore, StoreError
    except ImportError as e:
        raise Unreadable(f"{run}: the evidence store's client isn't installed here ({e}): {FIX}") from e
    store = LocalStore()
    try:
        attempt = store.get_attempt(run)
        if attempt is None:
            raise Unreadable(f"{run} is not in the evidence store this machine reads: {FIX}")
        outputs = attempt.get("outputs") or {}
        art = outputs.get("run_record") or outputs.get("run_files")
        if not art:
            raise Unreadable(f"{run} kept no record")
        labels_sync.read_through(store, [run, art], lambda m: print(f"prove layers: {m}", file=sys.stderr))
        if why := store.withdrawn(art) or store.withdrawn(run):
            raise Unreadable(f"{run} was withdrawn ({why}): nothing of it is read")
        return Path(store.fetch(art, paths=list(FETCH)))
    except StoreError as e:
        raise Unreadable(f"{run}'s record can't be fetched: {e}: {FIX}") from e


def detect(d: Path) -> str | None:
    """Which command's output `d` holds, by its files; None where it holds none of them."""

    if (d / "recursion_e2e.json").is_file():
        return RECURSION
    if (d / "audit.log").is_file() or any(d.glob("*/audit.log")):
        return PROVE_PATH
    if (d / "layers.txt").is_file() and (d / "a6" / "summary.json").is_file():
        return LAYERS_SH
    if any((d / f).is_file() for f in ("summary.txt", "a5.log", "a5/a5.log", "a5/a5/summary.json", "a5/summary.json")):
        return A5
    return None


def locate(given: str, fetcher: Callable[[str], Path]) -> Path:
    """The directory `given` names: a directory (its ``out/`` where it holds no output itself), or a run id's fetched
    ``out/``."""

    p = Path(given)
    if p.is_dir():
        return p / "out" if detect(p) is None and (p / "out").is_dir() else p
    if RUN_ID.fullmatch(given):
        root = fetcher(given)
        return root / "out" if (root / "out").is_dir() else root
    raise Unreadable(f"{given} is neither a directory nor a run id")
