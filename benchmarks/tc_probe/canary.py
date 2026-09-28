"""Canary: the fixed small tile set that says a pinned instruction still computes its model's words.  Run it on the anchor after any
driver / ptxas / toolkit change, on a second physical device of the SKU (TRUSTED_STABLE), or whenever anyone asks "did it break":

    python benchmarks/tc_probe/tc_probe.py --instruction sm89.mma.m16n8k32.e4m3 --out $RESEARCH_RUN_DIR --canary [--sass-check]

A canary is < 1 minute on the device, exits non-zero on any mismatch against a declared model, and under ``research run`` writes a
``tc-evidence/v1`` ``result.json`` whose ``tc_evidence.canary`` holds the counts and whose ``workload_fingerprint.family`` is
``canary``; the lane then labels the artifact ``family=canary instruction=<id> hardware=<sku family> model=<name>
model_agrees=<true|false>`` (the ``research data label`` lines are printed at the end; ``trust.py`` reads them).

The tile set is a pure function of the instruction id, the repository fixtures and :data:`CANARY_SEED` (:func:`canary_plan`,
:func:`canary_tiles`, witnessed by :func:`canary_digest`):

* the first :data:`CANARY_TILES` tiles of every finite-sweep family ``tc_probe.families`` defines for the format, drawn by the
  same generator as ``--sweep`` with ``n_random = CANARY_N_RANDOM`` (a small sweep, in the sweep's order, so a canary tile is a
  sweep tile of a known seed);
* every regression tile of every *resolved* discrepancy-log entry naming the instruction or a Target over it
  (:data:`trust.REGRESSION_TILES`): ``family:<name>`` tiles join the family set, ``replay:<fragment>`` files
  are replayed in full (golden records; the tc-total special captures for BF16).

TODO(pod): the device half (``run_canary``: ``Probe.tiles`` + ``model_eval`` + ``replay_golden``) has not been executed on
silicon from this laptop.  First pod run per anchor: ``research run --tool tc_probe --project verity -- python
benchmarks/tc_probe/tc_probe.py --instruction <id> --out $RESEARCH_RUN_DIR --canary --sass-check``, check the wall time (< 60 s) and
that ``probe_results.json["canary"]["digest"]`` equals :func:`canary_digest` computed offline, then label the artifact.  BF16
anchors have no ``adder_width_cancellation`` / ``acc_only_lsb_walk`` families yet (they are FP8-only in ``families()``): the
trust framework's PINNED criterion P3 lists them as missing for every BF16 instruction until ``families()`` yields them for BF16
and the anchors are re-swept.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:  # `python benchmarks/tc_probe/tc_probe.py` puts HERE first; a test importing this module may not
    sys.path.insert(0, str(HERE))

import tc_probe as P  # noqa: E402

from trust import REGRESSION_TILES, _load_discrepancies, subjects  # noqa: E402

CANARY_SEED = 20260922
CANARY_N_RANDOM = 64      # families() n_random for the canary draws: every family then has >= CANARY_TILES tiles
CANARY_TILES = 4          # tiles kept per family (M x 8 elements each: 512 for mma, 2048 for wgmma)
REPLAY_ROOTS = ("verity/ml/circuits/ops/tests/fixtures/golden", "verity/ml/circuits/ops/tests/fixtures/tc-total-2026-09-07", "verity/ml/circuits/silicon/captures")


@dataclass(frozen=True)
class CanaryPlan:
    instruction: str
    seed: int
    n_random: int
    tiles_per_family: int
    families: tuple[str, ...]                 # in the sweep's order
    regression: dict[str, tuple[str, ...]]    # resolved D-entry -> its tiles (as in REGRESSION_TILES)
    replays: tuple[str, ...]                  # repo-relative replay files, in D-entry order
    unresolved: tuple[str, ...]               # regression tiles this format / repo cannot realise (reported, not silently dropped)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["regression"] = {k: list(v) for k, v in self.regression.items()}
        return d

    def elements(self, M: int) -> int:
        return len(self.families) * self.tiles_per_family * M * P.N


def _repo_root(start: Path | None = None) -> Path | None:
    here = (start or HERE).resolve()
    for p in [here, *here.parents]:
        if (p / "verity" / "ml" / "circuits" / "silicon" / "discrepancy_log.json").is_file():
            return p
    return None


def subject_names(instruction_id: str) -> set[str]:
    """The instruction id and every Target name over it: what a discrepancy-log ``profile`` may say (the trust framework's own
    resolution, so the canary's regression set is exactly the dossier's)."""
    try:
        return {instruction_id} | subjects(instruction_id).names
    except KeyError:  # an instruction tc_probe measures that the registry does not list yet
        return {instruction_id}


def anchor_token(instruction_id: str) -> str | None:
    """The anchor SKU family of the instruction (``a100``, ``rtx4090``, ``h100``) for choosing among per-device capture files."""
    try:
        return subjects(instruction_id).anchor_family
    except KeyError:
        return None


def resolved_regression_tiles(instruction_id: str, repo: Path | None = None) -> dict[str, tuple[str, ...]]:
    """``D-id -> tiles`` for every *resolved* discrepancy-log entry naming the instruction (open ones are not regression tests yet;
    excluded ones have no tile by construction)."""
    names = subject_names(instruction_id)
    out: dict[str, tuple[str, ...]] = {}
    for e in _load_discrepancies(repo if repo is not None else _repo_root()):
        if e.get("profile") in names and e.get("status") == "resolved":
            out[e["id"]] = tuple(REGRESSION_TILES.get(e["id"], ()))
    return out


def family_names(fmt: P.Format, M: int, K: int, n_random: int = CANARY_N_RANDOM) -> list[str]:
    """The family names ``tc_probe.families`` yields for a format, in order (a dry run of the generator)."""
    return [name for name, *_ in P.families(np.random.default_rng(0), n_random, fmt, M, K)]


def _find_replay(repo: Path | None, fragment: str, ins: P.Instruction) -> str | None:
    """A repo-relative replay file whose path contains ``fragment`` and whose records fit the instruction's format; among several
    (the tc-total captures are per device) the one under the instruction's anchor directory (``a100``, ``4090``, ...)."""
    if repo is None:
        return None
    found: list[str] = []
    for root in REPLAY_ROOTS:
        base = repo / root
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            rel = p.relative_to(repo).as_posix()
            if not p.is_file() or fragment not in rel:
                continue
            if rel.endswith("mma_special.reduced.json.gz"):
                if ins.fmt is P.BF16 and ins.m == 16:
                    found.append(rel)
                continue
            if rel.endswith(".json"):
                try:
                    head = json.loads(p.read_text())
                except ValueError:
                    continue
                if isinstance(head, dict) and head.get("records") and head.get("k", ins.k) == ins.k and head.get("operand_bits", ins.fmt.bits) == ins.fmt.bits:
                    found.append(rel)
    if not found:
        return None
    tok = anchor_token(ins.id) or ""
    digits = "".join(ch for ch in tok if ch.isdigit())  # rtx4090 -> 4090 (the capture directories are named by number)
    for rel in found:
        parts = rel.lower().split("/")
        if tok and (tok in parts or (digits and digits in parts)):
            return rel
    return found[0]


def canary_plan(instruction_id: str, repo: Path | str | None = None) -> CanaryPlan:
    """The canary tile set of an instruction: deterministic given the id, the repo fixtures and the constants above."""
    ins = P.INSTRUCTIONS[instruction_id]
    repo_path = Path(repo).resolve() if repo is not None else _repo_root()
    available = family_names(ins.fmt, ins.m, ins.k)
    regression = resolved_regression_tiles(instruction_id, repo_path)
    wanted = set(available)                                       # one canary tile batch from every family the format has
    replays: list[str] = []
    unresolved: list[str] = []
    for did, tiles in regression.items():
        for t in tiles:
            kind, _, what = t.partition(":")
            if kind == "family":
                if what in available:
                    wanted.add(what)
                else:
                    unresolved.append(f"{did}:{t}")
            elif kind == "replay":
                f = _find_replay(repo_path, what, ins)
                if f is None:
                    unresolved.append(f"{did}:{t}")
                elif f not in replays:
                    replays.append(f)
    return CanaryPlan(instruction=instruction_id, seed=CANARY_SEED, n_random=CANARY_N_RANDOM, tiles_per_family=CANARY_TILES,
                      families=tuple(f for f in available if f in wanted), regression=regression, replays=tuple(replays),
                      unresolved=tuple(unresolved))


def canary_tiles(plan: CanaryPlan, fmt: P.Format, M: int, K: int) -> Iterator[tuple[str, np.ndarray, np.ndarray, np.ndarray, dict | None]]:
    """``(family, A, Bt, C, meta)`` for every family of the plan: the first ``tiles_per_family`` tiles of the family drawn by
    ``tc_probe.families`` with the canary seed.  Slicing after the draw keeps every family's tiles independent of which families
    the plan keeps (the generator consumes the same random stream either way)."""
    rng = np.random.default_rng(plan.seed)
    keep = set(plan.families)
    for name, A, Bt, C, meta in P.families(rng, plan.n_random, fmt, M, K):
        if name not in keep:
            continue
        n = min(plan.tiles_per_family, A.shape[0])
        yield name, np.ascontiguousarray(A[:n]), np.ascontiguousarray(Bt[:n]), np.ascontiguousarray(C[:n]), meta


def canary_digest(plan: CanaryPlan, fmt: P.Format, M: int, K: int) -> str:
    """sha256 over every canary tile (A, Bt, C bytes in plan order): the determinism witness recorded next to the counts."""
    h = hashlib.sha256()
    for name, A, Bt, C, _ in canary_tiles(plan, fmt, M, K):
        h.update(name.encode())
        h.update(A.tobytes())
        h.update(Bt.tobytes())
        h.update(C.tobytes())
    return h.hexdigest()


def compare(D: np.ndarray, results: list[tuple[np.ndarray, np.ndarray]], model_names: list[str]) -> dict[str, int]:
    """Mismatches per model between the hardware words ``D`` and each model's ``(words, domain_errors)``."""
    return {name: int(((model != D) | err).sum()) for name, (model, err) in zip(model_names, results)}


def run_canary(probe: "P.Probe", ins: P.Instruction, plan: CanaryPlan, out: Path, procs: int, repo: Path | str | None = None,
               tiles_fn=None, eval_fn=None, replay_fn=None) -> dict:
    """The device half: run every canary tile and replay, compare against the instruction's models, write ``canary_tiles.npz``.
    ``tiles_fn(A, Bt, C) -> D``, ``eval_fn(A, Bt, C) -> [(words, errs) per model]`` and ``replay_fn(path) -> replay dict`` default
    to the probe, the model pool and ``replay_golden`` (injectable so the bookkeeping is testable without a GPU).
    TODO(pod): first execution on silicon pending."""
    import multiprocessing as mp

    M, K, fmt = ins.m, ins.k, ins.fmt
    model_names = [m.name for m in ins.models]
    declared = [m.name for m in ins.models if m.declared]
    tiles_fn = tiles_fn or probe.tiles
    pool = None
    if eval_fn is None:
        pool = mp.Pool(procs, initializer=P._init, initargs=(P.model_descs(ins),))
        eval_fn = lambda A, Bt, C: P.model_eval(pool, A, Bt, C, procs)  # noqa: E731
    t0 = time.time()
    fams: dict[str, dict] = {}
    saved: dict[str, np.ndarray] = {}
    try:
        for name, A, Bt, C, _meta in canary_tiles(plan, fmt, M, K):
            D = tiles_fn(A, Bt, C)
            results = eval_fn(A, Bt, C)
            mism = compare(D, results, model_names)
            asserted = P.family_asserted(ins, name)
            first = []
            any_mism = np.zeros(D.shape, np.bool_)
            for model, err in results:
                any_mism |= (model != D) | err
            for t, i, j in list(zip(*np.nonzero(any_mism)))[: P.FIRST_MISMATCHES]:
                rec = {"tile": int(t), "i": int(i), "j": int(j), **P.hexrec(C[t, i, j], A[t, i], Bt[t, j], D[t, i, j], fmt.hex_width), "model": {}}
                for mname, (model, err) in zip(model_names, results):
                    rec["model"][mname] = "error" if err[t, i, j] else f"{int(model[t, i, j]):08x}"
                first.append(rec)
            fams[name] = {"tiles": int(A.shape[0]), "elements": int(A.shape[0]) * M * P.N, "asserted": asserted, "mismatches": mism, "first_mismatches": first}
            for arr, tag in ((A, "A"), (Bt, "Bt"), (C, "C"), (D, "D")):
                saved[f"{name}_{tag}"] = arr
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    replays = []
    repo_path = Path(repo).resolve() if repo is not None else _repo_root()
    for rel in plan.replays:
        path = (repo_path / rel) if repo_path is not None else Path(rel)
        if replay_fn is not None:
            r = replay_fn(path)
        elif rel.endswith("mma_special.reduced.json.gz"):
            r = P.replay_golden(probe, path, P.special_all_finite_records(path), ins.models[0].pipeline.name)
        else:
            r = P.replay_golden(probe, path)
        r["file"] = rel
        replays.append(r)
    np.savez_compressed(out / "canary_tiles.npz", **saved)
    totals = {m: sum(f["mismatches"][m] for f in fams.values() if f["asserted"]) for m in model_names}
    report = {
        "plan": plan.to_dict(),
        "digest": canary_digest(plan, fmt, M, K),
        "families": fams,
        "replays": replays,
        "elements": sum(f["elements"] for f in fams.values()),
        "asserted_elements": sum(f["elements"] for f in fams.values() if f["asserted"]),
        "total_mismatches_by_model": totals,
        "model_agrees": {m: totals[m] == 0 and all(r["mismatches"] == 0 and r["other_positions_nonzero"] == 0 for r in replays) for m in model_names},
        "declared_models": declared,
        "seconds": round(time.time() - t0, 1),
    }
    report["failures"] = canary_failures(report)
    return report


def canary_failures(report: dict) -> list[str]:
    """What makes a canary fail: any mismatch against a declared model, any replay mismatch."""
    out = []
    for m in report["declared_models"]:
        if report["total_mismatches_by_model"].get(m):
            out.append(f"canary mismatches vs {m}={report['total_mismatches_by_model'][m]}")
    for r in report["replays"]:
        if r["mismatches"] or r["other_positions_nonzero"]:
            out.append(f"canary replay {r['file']} mismatches={r['mismatches']} other_positions_nonzero={r['other_positions_nonzero']}")
    return out


def evidence_labels(ins: P.Instruction, report: dict, identity: dict, *, evidence: str, family: str | None = None) -> list[dict]:
    """The label contract of a ``tc-evidence/v1`` artifact (``trust.py`` reads exactly these keys): ``instruction``,
    ``hardware`` (the SKU family), ``model`` (the declared pipelines), ``model_agrees``, ``evidence`` (``sweep`` / ``canary``),
    ``device`` (the GPU UUID, S1's device identity) and, for canaries, ``family=canary``.  ``report`` is a sweep or canary report
    with ``model_agrees`` per model (a canary's also lists ``declared_models``)."""

    declared = report.get("declared_models") or [m.name for m in ins.models if m.declared] or [m.name for m in ins.models]
    agrees = all(report["model_agrees"].get(m, False) for m in declared)
    labels = [{"key": "family", "value": family}] if family else []
    return labels + [
        {"key": "instruction", "value": ins.id},
        {"key": "hardware", "value": sku_family(identity.get("device"))},
        {"key": "model", "value": ",".join(declared)},
        {"key": "model_agrees", "value": "true" if agrees else "false"},
        {"key": "evidence", "value": evidence},
        {"key": "device", "value": identity.get("gpu_uuid")},
    ]


def result_labels(ins: P.Instruction, report: dict, identity: dict) -> list[dict]:
    """The label contract for a canary artifact."""
    return evidence_labels(ins, report, identity, evidence="canary", family="canary")


def sku_family(device: str | None) -> str | None:
    from trust import sku_family as _f

    return _f(device)


def label_commands(art_id: str, labels: list[dict], by: str) -> list[str]:
    return [f"research data label {art_id} {lab['key']} {lab['value']} --by {by}" for lab in labels if lab["value"]]


if __name__ == "__main__":  # offline: print the plan and the digest for an instruction (no device)
    import argparse

    ap = argparse.ArgumentParser(description="canary plan / digest (offline)")
    ap.add_argument("instruction")
    ap.add_argument("--repo", default=None)
    ns = ap.parse_args()
    plan = canary_plan(ns.instruction, ns.repo)
    ins = P.INSTRUCTIONS[ns.instruction]
    print(json.dumps({**plan.to_dict(), "elements": plan.elements(ins.m), "digest": canary_digest(plan, ins.fmt, ins.m, ins.k)}, indent=1))
