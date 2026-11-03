"""``proof-throughput/v2``: the frozen populations, one per workload, pinned by ``instances/<workload>.json``.

A class-statement workload's population is ``n_max`` independent instances of its VU Definition (``Workload.population``,
core's ``population_program``), each instance one unit of ``Q_template_instance`` v0 (``PARTITION``), so the population is
one stratum. Its lanes are the ones ``verity_flock.class_statement.class_lanes`` gives that one unit class: uniform words sampled
from ``partition_units.derive(SOURCE, DOMAIN, {program digest, seed 0})``, each instance its own x and weight row. The point
job stages exactly these (``--program-module``, ``--batch fixed --n n_max``, seed 0) once, whatever B, and each session proves
the B units the verifier samples by the stratified law (``law``: core's ``one_stage.draw.derive`` of ``stratified:B`` over the
program and query, which every verifier derives for itself). The words are uniform 16-bit encodings, non-finite ones
included (the statement's domain is ``total``): a Boolean circuit's proving cost doesn't depend on the values it carries.

The manifest pins each lane's digest (SHA-256 of ``proof-throughput/lane/v1``, then every input word and every output word
as u32 little-endian, ports in order), the population's digest, the recipe and the code it came from. A result names the
manifest by its file's SHA-256 (``population_ref``); the point job recomputes the population before proving (``check``).

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

from .workloads import Workload

DATASET = "proof-throughput/v2"
SCHEMA = "proof-throughput/instances/v2"
SEED = 0
PARTITION = "q-template-instance"
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


def population_digest(lanes: list[str]) -> str:
    return hashlib.sha256("".join(lanes).encode()).hexdigest()


def units(w: Workload) -> Any:
    """The population's ``PartitionUnits``: its program under ``PARTITION``."""
    from verity_flock import partition_units as PU

    return PU.partition_units(w.population(), PARTITION)


def class_lanes(w: Workload, n: int | None = None) -> dict[str, Any]:
    """The population's first ``n`` instances (default all), as the point job stages them: lane t's words depend on t alone."""
    from verity.ml.circuits.codec import program_digest
    from verity_flock import class_statement as CS
    from verity_flock import partition_units as PU

    if w.stager != "class-statement":
        raise ValueError(f"{w.name}: its instances come from {w.stager}, not class_lanes")
    us = units(w)
    classes = us.classes()
    if len(classes) != 1:
        raise ValueError(f"{w.name}: {len(classes)} unit classes, not one")
    (shape, cl), = classes.items()
    pdig = program_digest(us.program)
    key = PU.derive(PU.SOURCE, PU.DOMAIN, {"program": pdig, "seed": SEED})
    ins, want, _rows = CS.class_lanes(us, cl, key, w.n_max if n is None else n, CS.broadcast_params(us))
    return {"shape": shape, "definition": str(cl.representative.definition), "program_digest": pdig, "ins": ins, "want": want}


def partition_object(w: Workload, circuit_meta: dict[str, Any]) -> dict[str, Any]:
    """The ``verity/partition/v1`` object the workload's statement binds, derived from its own program as the verifier must
    (``flock-verify --partition``), refused unless it is the one the circuit's META names."""
    from verity.ml.circuits import partition_object as PO

    us = units(w)
    obj = PO.build(us.program, us.query)
    want = ((circuit_meta.get("partition") or {}).get("digest"), circuit_meta.get("program_sha512"))
    if (PO.digest(obj), obj["program"]) != want:
        raise ValueError(f"{w.name}: the circuit binds partition {want[0]} and program {want[1]}, not the workload's "
                         f"{PO.digest(obj)} and {obj['program']}")
    return obj


def program_file(w: Workload) -> bytes:
    """The program as the verifier reads it (``flock-verify --program``): its canonical descriptor encoding."""
    from verity.ml.circuits.codec import canonical_json, encode_program

    return canonical_json(encode_program(w.population()))


def law(w: Workload, b: int) -> dict[str, Any]:
    """The verifier's stratified law for B: ``stratified:B`` expanded over the population's program and query (core's
    ``one_stage.draw.derive``), the object ``flock-circuit serve --draw-law`` draws each session's units by."""
    from verity.core.protocols.sampled_proofs.one_stage import draw as D

    us = units(w)
    return D.derive(us.program, us.query, {"law": "stratified", "k": b})


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
    got = class_lanes(w)
    lanes = lane_digests(got["ins"], got["want"])
    outs = [v for port in got["want"] for v in port]
    return {
        "schema": SCHEMA, "dataset": DATASET, "tier": w.name, "n": w.n_max, "K": w.K, "vu": w.vu, "profile": w.profile,
        "recipe": {"population": f"population_program({w.vu}, {w.n_max})", "partition": PARTITION, "seed": SEED,
                   "stager": w.stager,
                   "lanes": "verity_flock.class_statement.class_lanes of the one unit class, key "
                            "partition_units.derive(SOURCE, DOMAIN, {program: program_digest, seed}); lane i is instance i",
                   "shape": got["shape"], "definition": got["definition"], "program_digest": got["program_digest"],
                   "domain": "total: uniform 16-bit words, non-finite encodings included"},
        "lane_digest": "sha256(b'proof-throughput/lane/v1\\0' || every input word then every output word, each u32 LE)",
        "generated": {"utc": (now or dt.datetime.now(dt.timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ"), "git_head": _git_head(),
                      "research_run_id": os.environ.get("RESEARCH_RUN_ID"), "source_sha256": _source_sha256()},
        "output_words": {"distinct": len(set(outs)), "count": len(outs)},
        "population_digest": population_digest(lanes),
        "lanes": lanes,
    }


def check(w: Workload) -> dict[str, Any]:
    """Recompute the population and compare its digest with the manifest's."""
    pinned = load(w)
    got = class_lanes(w)
    have = population_digest(lane_digests(got["ins"], got["want"]))
    return {"ok": have == pinned["population_digest"] and got["program_digest"] == pinned["recipe"]["program_digest"],
            "n": w.n_max, "population": have, "pinned": pinned["population_digest"], "manifest_sha256": manifest_sha256(w)}


def population_ref(w: Workload) -> dict[str, Any] | None:
    """The fingerprint's ``population``: ids ``[0, n_max)`` of the workload's tier, the units each session's sample is taken
    from (the contract's ``instances`` names the B ids a statement proves, which a sample doesn't fix); None without a manifest."""
    if not manifest_path(w).exists():
        return None
    return {"dataset": DATASET, "tier": w.name, "range": [0, w.n_max], "manifest_sha256": manifest_sha256(w)}
