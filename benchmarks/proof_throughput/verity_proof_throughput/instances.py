"""``proof-throughput/v1``: the frozen instance sets, one tier per workload, pinned by ``instances/<workload>.json``.

A class-statement workload's instances are the lanes ``verity_flock.class_statement.class_lanes`` gives its one coordinate
class: the synthetic ``Gemm_v2`` Call of ``n_max`` outputs (``Workload.definitions``) evaluated on uniform words drawn from
``partition_units.derive(SOURCE, DOMAIN, {program digest, seed 0})``, the x row shared by the Call's coordinates and each
weight row its own. The point job stages exactly these (``--definitions``, ``--batch fixed --n B``, seed 0), so B proves
ids ``[0, B)``. The words are uniform 16-bit encodings, non-finite ones included (the statement's domain is ``total``): a
Boolean circuit's proving cost doesn't depend on the values it carries.

The manifest pins each lane's digest (SHA-256 of ``proof-throughput/lane/v1``, then every input word and every output word
as u32 little-endian, ports in order), the digest of every ladder prefix, the recipe and the code it came from. A result
names the manifest by its file's SHA-256 (``contract.instances_ref``); the point job recomputes its prefix before proving
(``check_prefix``).

    python -m verity_proof_throughput instances --workload bf16-k6144 [--write]
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from .workloads import Workload, ladder

DATASET = "proof-throughput/v1"
SCHEMA = "proof-throughput/instances/v1"
SEED = 0
PARTITION = "q-word"
LANE_TAG = b"proof-throughput/lane/v1\0"
ROOT = Path(__file__).resolve().parents[1] / "instances"


def manifest_path(w: Workload) -> Path:
    return ROOT / f"{w.name}.json"


def manifest_sha256(w: Workload) -> str:
    return hashlib.sha256(manifest_path(w).read_bytes()).hexdigest()


def load(w: Workload) -> dict[str, Any]:
    return json.loads(manifest_path(w).read_text())


def lane_digests(ins: list[list[int]], want: list[list[int]]) -> list[str]:
    """Each lane's digest: the tag, then its input words and its output words, each a u32 little-endian."""
    import numpy as np

    cols = np.asarray(ins + want, dtype=np.uint32).T.copy()
    return [hashlib.sha256(LANE_TAG + row.astype("<u4").tobytes()).hexdigest() for row in cols]


def prefix_digest(lanes: list[str], b: int) -> str:
    return hashlib.sha256("".join(lanes[:b]).encode()).hexdigest()


def class_lanes(w: Workload, n: int) -> dict[str, Any]:
    """The first ``n`` instances of the workload's coordinate class, as the point job stages them."""
    from verity.ml.circuits.codec import program_digest
    from verity_flock import class_statement as CS
    from verity_flock import partition_units as PU

    if w.stager != "class-statement":
        raise ValueError(f"{w.name}: its instances come from {w.stager}, not class_lanes")
    if not 1 <= n <= w.n_max:
        raise ValueError(f"{w.name}: n = {n} outside [1, {w.n_max}]")
    program, _calls, _prov = CS.distinct_calls(None, None, w.definitions(), None)
    units = PU.partition_units(program, PARTITION)
    classes = units.classes()
    if len(classes) != 1:
        raise ValueError(f"{w.name}: {len(classes)} coordinate classes, not one")
    (shape, cl), = classes.items()
    pdig = program_digest(program)
    key = PU.derive(PU.SOURCE, PU.DOMAIN, {"program": pdig, "seed": SEED})
    ins, want, _rows = CS.class_lanes(units, cl, key, n, CS.broadcast_params(units))
    return {"shape": shape, "definition": str(cl.representative.definition), "program_digest": pdig, "ins": ins, "want": want}


def _git_head() -> str | None:
    if os.environ.get("RESEARCH_SOURCE_SHA"):
        return os.environ["RESEARCH_SOURCE_SHA"]
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=Path(__file__).parent).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _source_sha256() -> dict[str, str]:
    import verity_flock

    base = Path(verity_flock.__file__).parent
    return {f"verity_flock/{f}": hashlib.sha256((base / f).read_bytes()).hexdigest()
            for f in ("class_statement.py", "partition_units.py")}


def build(w: Workload, now: dt.datetime | None = None) -> dict[str, Any]:
    """The workload's manifest, regenerated from the recipe."""
    got = class_lanes(w, w.n_max)
    lanes = lane_digests(got["ins"], got["want"])
    outs = [v for port in got["want"] for v in port]
    return {
        "schema": SCHEMA, "dataset": DATASET, "tier": w.name, "n": w.n_max, "K": w.K, "vu": w.vu, "profile": w.profile,
        "recipe": {"definitions": w.definitions(), "partition": PARTITION, "seed": SEED, "stager": w.stager,
                   "lanes": "verity_flock.class_statement.class_lanes of the one coordinate class, key "
                            "partition_units.derive(SOURCE, DOMAIN, {program: program_digest, seed}); lane i is coordinate i",
                   "shape": got["shape"], "definition": got["definition"], "program_digest": got["program_digest"],
                   "domain": "total: uniform 16-bit words, non-finite encodings included"},
        "lane_digest": "sha256(b'proof-throughput/lane/v1\\0' || every input word then every output word, each u32 LE)",
        "generated": {"utc": (now or dt.datetime.now(dt.timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ"), "git_head": _git_head(),
                      "research_run_id": os.environ.get("RESEARCH_RUN_ID"), "source_sha256": _source_sha256()},
        "output_words": {"distinct": len(set(outs)), "count": len(outs)},
        "prefixes": {str(b): prefix_digest(lanes, b) for b in ladder(w.n_max)},
        "lanes": lanes,
    }


def check_prefix(w: Workload, b: int) -> dict[str, Any]:
    """Recompute ids ``[0, b)`` and compare their prefix digest with the manifest's."""
    pinned = load(w)
    got = class_lanes(w, b)
    lanes = lane_digests(got["ins"], got["want"])
    want = pinned["prefixes"].get(str(b)) or prefix_digest(pinned["lanes"], b)
    have = prefix_digest(lanes, b)
    return {"ok": have == want and got["program_digest"] == pinned["recipe"]["program_digest"], "B": b, "prefix": have,
            "pinned": want, "manifest_sha256": manifest_sha256(w)}


def instances_ref(w: Workload, b: int) -> dict[str, Any] | None:
    """The fingerprint's ``instances`` for B: ids ``[0, B)`` of the workload's tier; None while it has no manifest."""
    from verity_numerical.bench import contract

    if not manifest_path(w).exists():
        return None
    return contract.instances_ref(w.name, 0, b, manifest_sha256(w), dataset=DATASET)
