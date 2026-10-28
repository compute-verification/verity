"""verity_vllm/check/gates.py — the frozen invariant set G1..G7 over the JSON artifacts of one run directory (torch-free).

THE GATES AND THE CONSTANTS BELOW ARE FROZEN: they change only through an approved decision recorded in
`out/capture/decisions.md`, never by editing this file alone.  Anything that is a heuristic (per-step histogram
equality, linear instance growth, the family-reference match, the overhead ladder) is a DIAGNOSTIC reported in the
card (`diagnostics()`), never a gate.

A *run directory* is what `bench/run_config.py` produces for one configuration (model × engine arguments × κ):

    RUN_DIR/run.json                 the driver's manifest (config_id, profile, case, workload, C, laptop, snapshot_steps,
                                     artifact path overrides)             — read by `RunArtifacts`
    RUN_DIR/stages.json              per-stage {status, seconds, returncode, log}   — a failed stage becomes evidence
    RUN_DIR/capture/{log.jsonl.gz, tokens.json, versions.json, stats.json, snapshots.json, cos_sin_cache.npy}
    RUN_DIR/control/tokens.json      the `--no-observer` control run: THE token reference of every gate
    RUN_DIR/resolution.json          `bench.resolve_log --out RUN_DIR/resolution`
    RUN_DIR/fold_summary.json        this module's `FOLD_SUMMARY_SCHEMA` (written twice: `fold_summary_2.json`)
    RUN_DIR/census/census.json       GPU census;  RUN_DIR/census_reconcile.json  its torch-free reconciliation
    RUN_DIR/nonint/noninterference.json
    RUN_DIR/golden.json              `verity_vllm.properties.golden --check` (integrator)
    RUN_DIR/quarantine.json          differential tests of newly registered definitions (when the run uses any)
    RUN_DIR/holdouts.json            written by the ORCHESTRATOR: the same gates G1..G6 over holdout workloads
    RUN_DIR/compare.json             `verity_vllm.check.fold_compare` (tokens, boundary, subcircuits, structure)
    RUN_DIR/wiring.json, replay_a.json, replay_chain.json, replay_b_step<N>.json, root_policy.json, provenance.json
                                     no producer in this tree: a gate that reads one is skipped with the file missing as
                                     its reason

Gates:
    G1 observation fidelity    non-interference PASS, tokens equal across a/b/c AND to the control run, every boundary
                               hash equal, steps == max_tokens, layers == expected_layers; census residue 0, ambiguous 0,
                               every declared kernel's launch count equal to its declaring events
    G2 semantic closure        0 Unsupported, 0 Unresolved, 0 ambiguous pattern matches
    G3 root closure            0 unknown roots, 0 opaque/prescribed fallbacks, root_policy pass with no new root class
    G4 primitive exactness     tier-a replay pass, k >= TIER_A_K per (specialization, regime) stratum, every captured
                               specialization compared; quarantine differential tests all pass when new definitions are used
    G5 composition exactness   (D82) wiring.json: EVERY operand read of EVERY instance reproduced by the independent
                               last-writer map (storage identity + generation + byte range + stream order), 0 uncovered,
                               one stream, 0 writes outside the schema-declared write set;  replay_a.json: the verifier-
                               seeded stratified sample passes and every served token at every snapshot step (>=
                               MIN_DECODE_TOKEN_STEPS decode steps) is reproduced from its observed logits;
                               replay_chain.json: >= CHAIN_MIN_SLICES connected slices replayed with computed intermediates,
                               all pass.  Full replay (replay_b_*) is reported when present, never gating.
    G6 canonicality+regression two independent folds -> one digest; golden.json pass
    G7 holdout generalization  holdouts.json present, >= 1 holdout, G1..G6 pass in every holdout

    G8 VU partition            STRUCTURAL, outside the frozen set (query track): `query_use.json`
                               (`verity-ir/query-use/v1`, the partition track's use configuration + verdict document);
                               `skipped` ("no query-use document") when absent, pass iff verdict.partition.status == "exact"
                               and verdict.width.status == "ok" (no violators, no unsupported) and verdict.status == "ok";
                               `fail` (with the codes) on overlap / gap / violation / `not-ready` / `fail` / a binding
                               mismatch with `query.json`; `unsupported` (a fifth status, lead 22:22Z) when the evaluator
                               could not decide -- neither pass nor fail, the run verdict INCOMPLETE, never REJECTED.
                               A run without the document keeps today's verdict: G8 `skipped` never makes it INCOMPLETE.

    R1 reference integrity     FAIL-CLOSED (`REF_INTEGRITY_RULE`; explicitly authorised checker
                               addition, outside the frozen set): `fold/construction errors > 0 ⇒ the Program is ineligible
                               as an accepted reference`.  Evidence = `fold_summary.json` (`errors`, `unresolved`,
                               `unsupported`; `error_classes` / `per_step` must agree with `errors`), `fold_summary_2.json`
                               when present, `resolution.json` `totals.errors` / `totals.unsupported` when present.  Any
                               count > 0 -> `fail` (verdict REJECTED).  Missing summary, missing / non-integer count or
                               inconsistent evidence -> `unverified` (verdict INCOMPLETE): missing error evidence is never
                               read as zero.  Never `skipped`.  `fold_integrity(...)` is the pure rule over the dicts so
                               any consumer of a fold Program (e.g. the AOT roundtrip's `--accepted`) can apply it.

Every gate is a pure function `gate_Gx(run: RunArtifacts) -> GateResult`; nothing here imports torch, replay, analytic,
root_policy or golden.  A missing artifact makes the gate `skipped` (with the producing stage's log as evidence), a
malformed one `error`; a gate never raises.  `evaluate_gates(run_dir)` runs the seven gates, G8 and R1; `verdict(results)` folds
them into ACCEPTED / REJECTED / INCOMPLETE (G1..G7 as before; G8 / R1 contribute only `fail` -> REJECTED, `error` /
`unsupported` / `unverified` -> INCOMPLETE; a record that predates them keeps the verdict of G1..G7).

The fold summary the driver writes (`run_config --worker fold`) is defined here as `FOLD_SUMMARY_SCHEMA` /
`FOLD_SUMMARY_FIELDS` (also reproduced in `out/gen/cards/SCHEMA.md`).
"""

from __future__ import annotations

import importlib
import json
import math
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from verity_vllm.check.codes import VerificationCode as VC
from verity_vllm.check.result import MISSING_INPUT, CheckResult
from verity_vllm.check.verdict import match_verdict

# ---------------------------------------------------------------------------------------------------------
# FROZEN CONSTANTS — part of the accepted invariant set; change only through an approved decision in
# `out/capture/decisions.md`.
# ---------------------------------------------------------------------------------------------------------

#: G4 — the tier-a body replay must evaluate at least this many instances per specialization (`k`).
TIER_A_K = 3
#: G5 — served tokens must be reproduced at step 0 and at least this many decode steps (KV transitions in between).
MIN_DECODE_TOKEN_STEPS = 2
#: G5 — at least this many connected chained slices must pass.
CHAIN_MIN_SLICES = 1
#: G5 — the regimes every specialization present in both must be sampled in (tier a strata).
REGIMES = ("prefill", "decode")
#: Analytic-roots tolerance (bf16 ULPs) used by `root_policy` validators; recorded here so the card can quote it.
COS_SIN_MAX_ULP = 1
#: Diagnostic only (never gating): the cumulative instance count over decode steps is called "linear" when the
#: least-squares residual is within this many instances.
DIAG_MAX_LINEAR_RESIDUAL = 0.5
#: The gate ids, in report order.
GATES = ("G1", "G2", "G3", "G4", "G5", "G6", "G7")
GATE_NAMES = {
    "G1": "observation fidelity", "G2": "semantic closure", "G3": "root closure", "G4": "primitive exactness",
    "G5": "composition exactness", "G6": "canonicality + regression", "G7": "holdout generalization",
}
#: The gates a holdout must pass (G7 is the holdout gate itself).
HOLDOUT_GATES = ("G1", "G2", "G3", "G4", "G5", "G6")
#: Structural gates outside the frozen G1..G7 set (evaluated and reported, verdict rule in `verdict`).
STRUCTURAL_GATES = ("G8",)
GATE_NAMES["G8"] = "VU partition"
#: Integrity gates outside the frozen set: the fail-closed reference-integrity rule.
INTEGRITY_GATES = ("R1",)
GATE_NAMES["R1"] = "reference integrity (fail-closed)"
#: The rule, versioned: quoted in every R1 result and in `gates.json["constants"]`; bump the date on any change of the rule.
REF_INTEGRITY_RULE = {
    "id": "R1", "version": "2026-09-18",
    "statement": "fold/construction errors > 0 => the Program is ineligible as an accepted reference; missing or inconsistent "
                 "error evidence => unverified (never zero)",
    "evidence": ["fold_summary.json: errors, unresolved, unsupported (+ error_classes / per_step consistency)",
                 "fold_summary_2.json (when present): the same", "resolution.json (when present): totals.errors, totals.unsupported"],
}
#: Every gate `evaluate_gates` runs, in report order.
ALL_GATES = GATES + STRUCTURAL_GATES + INTEGRITY_GATES
#: G8 — the schema of the query-use document (`verity.ir/query_artifact.py`, partition owner).
QUERY_USE_SCHEMA = "verity-ir/query-use/v1"
QUERY_USE_KIND = "vu-partition"

GATES_SCHEMA = "verity-gen/gates/v1"
FOLD_SUMMARY_SCHEMA = "verity-gen/fold-summary/v1"

#: Default artifact locations relative to the run directory (`run.json["artifacts"]` may override any of them with an
#: absolute path, e.g. an existing capture supplied to the driver with `--log/--snapshots`).
ARTIFACTS: dict[str, str] = {
    "run": "run.json",
    "stages": "stages.json",
    "log": "capture/log.jsonl.gz",
    "snapshots": "capture/snapshots.json",
    "tokens": "capture/tokens.json",
    "versions": "capture/versions.json",
    "stats": "capture/stats.json",
    "cos_sin": "capture/cos_sin_cache.npy",
    "control_tokens": "control/tokens.json",
    "resolution": "resolution.json",
    "fold_summary": "fold_summary.json",
    "fold_summary_2": "fold_summary_2.json",
    "fold_summary_other": "fold_summary_other.json",
    "census": "census/census.json",
    "census_reconcile": "census_reconcile.json",
    "noninterference": "nonint/noninterference.json",
    "compare": "compare.json",
    "wiring": "wiring.json",
    "replay_a": "replay_a.json",
    "replay_chain": "replay_chain.json",
    "replay_c": "replay_c.json",
    "root_policy": "root_policy.json",
    "golden": "golden.json",
    "quarantine": "quarantine.json",
    "holdouts": "holdouts.json",
    "query_use": "query_use.json",
    "query_doc": "query.json",                    # G8 binding: the companion query document (`verity-ir/query/v1`, qpart 22:14Z)
    "provenance": "provenance.json",
    "gates": "gates.json",
    "card": "card.json",
    "m0": "m0/m0.json",
    "spend": "spend.json",
    "gpu_done": "gpu_done.json",
}

#: Which driver stage produces each artifact (for the evidence trail when the artifact is missing).
ARTIFACT_STAGE: dict[str, str] = {
    "log": "capture", "snapshots": "capture", "tokens": "capture", "versions": "capture", "stats": "capture",
    "cos_sin": "capture", "control_tokens": "control", "resolution": "resolve", "fold_summary": "fold",
    "fold_summary_2": "fold_2", "census": "census", "census_reconcile": "census_reconcile",
    "noninterference": "nonint_compare", "compare": "compare", "replay_a": "replay_a", "replay_c": "replay_c",
    "root_policy": "root_policy", "golden": "golden",
}


def replay_b_key(step: int) -> str:
    return f"replay_b_step{int(step)}"


def replay_b_rel(step: int) -> str:
    return f"replay_b_step{int(step)}.json"


# ---------------------------------------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------------------------------------


@dataclass
class GateResult:
    id: str
    name: str
    status: str                                   # "pass" | "fail" | "skipped" | "error" | "unsupported" (G8 only: undecided)
    evidence: list[str] = field(default_factory=list)   # artifact paths consulted (present or missing) and stage logs
    detail: dict = field(default_factory=dict)
    reason: str | None = None

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "GateResult":
        return cls(id=d["id"], name=d.get("name", GATE_NAMES.get(d["id"], "")), status=d["status"],
                   evidence=list(d.get("evidence") or []), detail=dict(d.get("detail") or {}), reason=d.get("reason"))

    def result(self) -> CheckResult:
        """The gate as a `CheckResult` named by its id: `pass` holds, `fail` is a mismatch, `skipped` did not run, any other status is undecided."""
        code = {"pass": VC.ACCEPTED, "fail": VC.CHECK_MISMATCH, "skipped": None}.get(self.status, MISSING_INPUT)
        return CheckResult(name=self.id, code=code, claim=self.name, scope=dict(self.detail), reason=self.reason)


class ArtifactError(Exception):
    """A present artifact that cannot be read as the JSON layout the gate expects."""


class _Skip(Exception):
    """Raised inside a gate when an input is unavailable; the wrapper turns it into a `skipped` result."""

    def __init__(self, reason: str, evidence: list[str] | None = None, detail: dict | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.evidence = list(evidence or [])
        self.detail = dict(detail or {})


# ---------------------------------------------------------------------------------------------------------
# the run directory
# ---------------------------------------------------------------------------------------------------------


class RunArtifacts:
    """The JSON artifacts of one run directory, located through `run.json["artifacts"]` overrides and `ARTIFACTS`."""

    def __init__(self, run_dir: str | Path, manifest: dict | None = None) -> None:
        self.run_dir = Path(run_dir)
        if manifest is None:
            p = self.run_dir / ARTIFACTS["run"]
            manifest = _read_json(p) if p.exists() else {}
            if not isinstance(manifest, dict):
                raise ArtifactError(f"{p}: run manifest is not a JSON object")
        self.manifest: dict = manifest
        self._cache: dict[str, Any] = {}
        self._stages: dict[str, dict] | None = None

    # locating ----------------------------------------------------------------------------------------

    def path(self, key: str) -> Path:
        over = (self.manifest.get("artifacts") or {}).get(key)
        if over:
            rel = over
        elif key in ARTIFACTS:
            rel = ARTIFACTS[key]
        elif key.startswith("replay_b_step"):
            rel = key + ".json"
        else:
            raise KeyError(key)
        p = Path(rel)
        return p if p.is_absolute() else self.run_dir / p

    def exists(self, key: str) -> bool:
        return self.path(key).exists()

    def rel(self, key: str) -> str:
        """The artifact path as evidence: relative to the run directory when inside it."""
        p = self.path(key)
        try:
            return str(p.relative_to(self.run_dir))
        except ValueError:
            return str(p)

    def json(self, key: str) -> Any:
        """The parsed artifact, `None` when the file is absent; `ArtifactError` when it is not valid JSON."""
        if key in self._cache:
            return self._cache[key]
        p = self.path(key)
        if not p.exists():
            self._cache[key] = None
            return None
        try:
            v = _read_json(p)
        except (OSError, ValueError) as e:
            raise ArtifactError(f"{p}: {type(e).__name__}: {e}") from e
        self._cache[key] = v
        return v

    def need(self, key: str) -> Any:
        """The parsed artifact, or `_Skip` (missing) with the producing stage's record as the reason / evidence."""
        v = self.json(key)
        if v is not None:
            return v
        ev = [self.rel(key)]
        stage = ARTIFACT_STAGE.get(key) or (key if key.startswith("replay_b_step") else None)
        rec = self.stage(stage) if stage else None
        if rec is not None:
            why = f"stage {stage} {rec.get('status')}" + (f" (rc={rec.get('returncode')})" if rec.get("returncode") not in (None, 0) else "")
            if rec.get("reason"):
                why += f": {rec['reason']}"
            if rec.get("log"):
                why += f" — see {rec['log']}"
                ev.append(str(rec["log"]))
            raise _Skip(f"{self.rel(key)} missing; {why}", ev)
        raise _Skip(f"{self.rel(key)} missing" + (f" (stage {stage} not run)" if stage else ""), ev)

    # driver records ----------------------------------------------------------------------------------

    def stage(self, name: str | None) -> dict | None:
        if name is None:
            return None
        if self._stages is None:
            st = self.json("stages")
            recs = st.get("stages", st) if isinstance(st, dict) else (st or [])
            if isinstance(recs, dict):
                recs = list(recs.values())
            self._stages = {r["stage"]: r for r in recs if isinstance(r, dict) and "stage" in r}
        return self._stages.get(name)

    @property
    def laptop(self) -> bool:
        return bool(self.manifest.get("laptop"))

    @property
    def profile_name(self) -> str | None:
        return self.manifest.get("profile")

    @property
    def C(self) -> dict | None:
        """The profile configuration: `run.json["C"]` (values; a card-style `{field: {"value", "provenance"}}` is
        accepted too), else the profile module named in the manifest; `ArtifactError` when that profile can't be loaded.  A gate never
        reads the log itself."""
        C = self.manifest.get("C")
        if isinstance(C, dict) and C:
            return {k: (v["value"] if isinstance(v, dict) and "value" in v else v) for k, v in C.items()}
        name = self.profile_name
        if not name:
            return None
        if "C_from_profile" not in self._cache:
            try:
                self._cache["C_from_profile"] = dict(load_profile_config(name))
            except Exception as e:
                raise ArtifactError(f"profile {name!r} could not be loaded: {type(e).__name__}: {e}") from e
        return self._cache["C_from_profile"]

    def workload_requests(self) -> list[dict]:
        """`[{request_id, prompt_len, max_tokens}]` from the manifest, else from the control / capture tokens.json."""
        wl = self.manifest.get("workload") or {}
        reqs = wl.get("requests")
        if reqs:
            return [dict(r) for r in reqs]
        for key in ("control_tokens", "tokens"):
            tok = self.json(key)
            if isinstance(tok, dict) and tok.get("requests"):
                return [{"request_id": r.get("request_id"), "prompt_len": len(r.get("prompt_token_ids") or []),
                         "max_tokens": r.get("max_tokens")} for r in tok["requests"]]
        return []

    def expected_steps(self) -> int | None:
        """The engine steps a run of this workload takes.  The control run (`control/tokens.json`, the bare engine
        on the same workload — THE token reference of every gate) records its realized `timing.engine_steps`; that is the
        expectation, so declared arrivals (B8-mixed), EOS before `max_tokens` and chunked prefill are all compared like for like.
        Fallback for older control files: `max(max_tokens)` (one prefill + max_tokens-1 decode steps, every request at step 0),
        as `noninterference.compare` also falls back."""
        ctl = self.json("control_tokens")
        if isinstance(ctl, dict) and (ctl.get("timing") or {}).get("engine_steps") is not None:
            return int(ctl["timing"]["engine_steps"])
        mt = [int(r["max_tokens"]) for r in self.workload_requests() if r.get("max_tokens") is not None]
        return max(mt) if mt else None

    def tier_b_steps(self) -> list[int]:
        """The steps tier b replays (G5): the manifest's `tier_b_steps`, else every snapshot step."""
        s = self.manifest.get("tier_b_steps")
        steps = [int(x) for x in (s.split(",") if isinstance(s, str) else (s or [])) if str(x).strip()]
        return steps or self.snapshot_steps()

    def snapshot_steps(self) -> list[int]:
        """The snapshot steps of the capture: the manifest's, else the tier-b reports present."""
        s = self.manifest.get("snapshot_steps")
        if isinstance(s, str):
            steps = [int(x) for x in s.split(",") if x.strip()]
        else:
            steps = [int(x) for x in (s or [])]
        if not steps:
            steps = [int(m.group(1)) for p in self.run_dir.glob("replay_b_step*.json") for m in [re.search(r"step(\d+)\.json$", p.name)] if m]
        return sorted(set(steps))

    def control_tokens(self) -> tuple[dict[str, list[int]] | None, str]:
        """`{request_id: generated ids}` of the clean control run: `control/tokens.json`, else the non-interference
        c_bare variant (also a bare run).  Never a workload-authored accepted list."""
        ct = self.json("control_tokens")
        if isinstance(ct, dict) and ct.get("requests"):
            return {str(r["request_id"]): [int(t) for t in r.get("generated_token_ids") or []] for r in ct["requests"]}, "control/tokens.json"
        ni = self.json("noninterference")
        c = ((ni or {}).get("tokens") or {}).get("c_bare") if isinstance(ni, dict) else None
        if isinstance(c, dict) and c:
            return {str(k): [int(t) for t in v] for k, v in c.items()}, "noninterference c_bare"
        return None, "none"


def load_profile_config(name: str) -> dict:
    """`verity_vllm.observe.fold.profiles.<name>.PROFILE.config` (torch-free import)."""
    mod = importlib.import_module(name if "." in name else f"verity_vllm.observe.fold.profiles.{name}")
    prof = getattr(mod, "PROFILE", None) or getattr(mod, "profile", None)
    if prof is None:
        for attr in ("build_profile", "make_profile", "get_profile"):
            f = getattr(mod, attr, None)
            if callable(f):
                prof = f()
                break
    if prof is None:
        raise ImportError(f"{mod.__name__} exposes no PROFILE")
    return dict(prof.config)


def _read_json(p: Path) -> Any:
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------------------------------------------------
# gate plumbing
# ---------------------------------------------------------------------------------------------------------

_GATE_FNS: dict[str, Callable[[RunArtifacts], GateResult]] = {}


def _gate(gid: str):
    """Register `fn(run) -> GateResult`; the wrapper supplies id/name and converts `_Skip` / exceptions."""

    def deco(fn):
        def wrapped(run: RunArtifacts) -> GateResult:
            try:
                r = fn(run)
            except _Skip as s:
                return GateResult(gid, GATE_NAMES[gid], "skipped", s.evidence, s.detail, s.reason)
            except ArtifactError as e:
                return GateResult(gid, GATE_NAMES[gid], "error", [], {}, str(e))
            except Exception as e:  # noqa: BLE001 - a gate never raises; a bug is an `error` result
                return GateResult(gid, GATE_NAMES[gid], "error", [], {}, f"{type(e).__name__}: {e}")
            r.id, r.name = gid, GATE_NAMES[gid]
            return r

        wrapped.__name__ = fn.__name__
        wrapped.__doc__ = fn.__doc__
        _GATE_FNS[gid] = wrapped
        return wrapped

    return deco


def _result(problems: list[str], evidence: list[str], detail: dict) -> GateResult:
    return GateResult("", "", "fail" if problems else "pass", evidence, detail, "; ".join(problems) if problems else None)


def _int(x: Any, what: str) -> int:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ArtifactError(f"{what}: expected a number, got {x!r}")
    return int(x)


def _get(d: Any, path: str, what: str) -> Any:
    """`d["a"]["b"]` for path "a.b"; `ArtifactError` when a key is missing (the layout is part of the contract)."""
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            raise ArtifactError(f"{what}: key {path!r} missing")
        cur = cur[k]
    return cur


def _first(d: dict, *keys: str) -> Any:
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return None


# ---------------------------------------------------------------------------------------------------------
# G1 — observation fidelity
# ---------------------------------------------------------------------------------------------------------


@_gate("G1")
def gate_G1(run: RunArtifacts) -> GateResult:
    """Observation fidelity: `noninterference.json` PASS; tokens equal across the (a) observer, (b) hooks and (c) bare
    variants AND equal to the clean control run's tokens (computed here from `generated_token_ids`, never from a
    workload-authored accepted list); every boundary hash equal; `steps == max_tokens`; `layers == expected_layers`
    (from the file; null fails); AND the census reconciliation has residue 0, 0 ambiguous claims and 0 launch-count
    mismatches."""
    ni = run.need("noninterference")
    cr = run.need("census_reconcile")
    ev = [run.rel("noninterference"), run.rel("census_reconcile")]
    problems: list[str] = []
    detail: dict[str, Any] = {}
    # -- non-interference
    if not ni.get("PASS"):
        problems.append("noninterference PASS is false")
    n_cmp = _int(_get(ni, "hash_comparisons", "noninterference"), "hash_comparisons")
    n_eq = _int(_get(ni, "hash_equal", "noninterference"), "hash_equal")
    if n_cmp <= 0:
        problems.append("no boundary hash comparisons")
    elif n_eq != n_cmp:
        problems.append(f"boundary hashes {n_eq}/{n_cmp} equal")
    if ni.get("mismatches"):
        problems.append(f"{len(ni['mismatches'])} hash mismatches")
    if ni.get("problems"):
        problems.append(f"{len(ni['problems'])} problems recorded: {ni['problems'][:2]}")
    toks = ni.get("tokens") or {}
    variants = {k: {str(r): [int(t) for t in v] for r, v in (toks.get(k) or {}).items()} for k in ("a_observer", "b_hooks", "c_bare")}
    missing = [k for k, v in variants.items() if not v]
    if missing:
        problems.append(f"tokens missing for variants {missing}")
    control, control_src = run.control_tokens()
    if control_src == "control/tokens.json":
        ev.append(run.rel("control_tokens"))
    if control is None:
        problems.append("no control tokens (control/tokens.json and c_bare both absent)")
    else:
        for k, v in variants.items():
            if v and v != control:
                problems.append(f"{k} tokens differ from the control run's ({control_src})")
    if variants["a_observer"] and variants["b_hooks"] and variants["a_observer"] != variants["b_hooks"]:
        problems.append("(a) and (b) tokens differ")
    steps = ni.get("steps") or []
    rows = [s for s in steps if isinstance(s, dict) and "error" not in s]
    if len(rows) != len(steps):
        problems.append("a step is present in only one variant")
    bad_sampled = [r.get("step") for r in rows if r.get("sampled_eq") is not True]
    if bad_sampled:
        problems.append(f"sampled ids differ between (a) and (b) at steps {bad_sampled[:8]}")
    exp = run.expected_steps()
    if exp is None:
        exp = ni.get("expected_steps")
    if exp is None:
        problems.append("expected step count unknown (no control run step count, no workload max_tokens)")
    elif len(steps) != int(exp):
        problems.append(f"{len(steps)} steps, expected {exp} (control run engine steps, else workload max_tokens)")
    if ni.get("expected_steps") is not None and exp is not None and int(ni["expected_steps"]) != int(exp):
        problems.append(f"noninterference.expected_steps={ni['expected_steps']} disagrees with the workload's {exp}")
    exp_layers = ni.get("expected_layers")
    if exp_layers is None:
        problems.append("expected_layers is null (noninterference run without --case/--expected-layers)")
    else:
        exp_layers = int(exp_layers)
        if ni.get("layers") != exp_layers:
            problems.append(f"layers={ni.get('layers')!r} != expected_layers={exp_layers}")
        counts = sorted({(r.get("layers_a"), r.get("layers_b")) for r in rows})
        if rows and counts != [(exp_layers, exp_layers)]:
            problems.append(f"per-step layer counts {counts} != expected_layers={exp_layers} in both variants")
        try:
            C = run.C
        except ArtifactError as e:
            C = None
            problems.append(f"expected_layers={exp_layers} not checked against the profile's NL: {e}")
        if C and C.get("NL") is not None and int(C["NL"]) != exp_layers:
            problems.append(f"expected_layers={exp_layers} disagrees with the profile's NL={C['NL']}")
    detail["noninterference"] = {"PASS": ni.get("PASS"), "hash_comparisons": n_cmp, "hash_equal": n_eq, "steps": len(steps),
                                 "expected_steps": exp, "layers": ni.get("layers"), "expected_layers": ni.get("expected_layers"),
                                 "control_source": control_src, "tokens_equal_control": {k: (v == control) if control and v else None for k, v in variants.items()},
                                 "n_mismatches": len(ni.get("mismatches") or []), "n_problems": len(ni.get("problems") or [])}
    # -- census
    t = _get(cr, "totals", "census_reconcile")
    residue = _int(_get(t, "residue_kernels", "census_reconcile.totals"), "residue_kernels")
    ambiguous = _int(_get(t, "ambiguous", "census_reconcile.totals"), "ambiguous")
    if residue:
        problems.append(f"census residue {residue} kernels {t.get('residue_names')}")
    if ambiguous:
        problems.append(f"census: {ambiguous} ambiguous compute claims")
    if not cr.get("steps_aligned"):
        problems.append(f"census steps ({cr.get('census_steps')}) not aligned with log steps ({cr.get('log_steps')})")
    step_errors = [s for s in (cr.get("steps") or []) if isinstance(s, dict) and "error" in s]
    if step_errors:
        problems.append(f"{len(step_errors)} census steps without a resolved log step")
    counts = _int(_get(t, "count_mismatches", "census_reconcile.totals (a reconciliation without the launch-count gate: "
                                              "rerun `verity-vllm census --reconcile`)"), "count_mismatches")
    if counts:
        problems.append(f"census: {counts} launch counts differ from their declaring events {t.get('count_mismatch_kernels')}")
    detail["census"] = {"census_kernels": t.get("census_kernels"), "distinct_kernel_names": t.get("distinct_kernel_names"),
                        "residue_kernels": residue, "residue_names": t.get("residue_names"), "ambiguous": ambiguous,
                        "steps_aligned": cr.get("steps_aligned"), "residue_zero": cr.get("residue_zero"),
                        "count_mismatches": counts, "count_exempt": t.get("count_exempt")}
    return _result(problems, ev, detail)


# ---------------------------------------------------------------------------------------------------------
# G2 — semantic closure
# ---------------------------------------------------------------------------------------------------------


def _ambiguity(res: dict, run: RunArtifacts) -> tuple[int | None, str, str | None]:
    """(count, source, evidence key): the resolver report's ambiguity field when it has one, else the census reconcile's."""
    for path in ("totals.ambiguous", "ambiguous", "primary_report.ambiguous", "totals.ambiguous_matches", "ambiguous_matches"):
        cur: Any = res
        ok = True
        for k in path.split("."):
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok:
            n = len(cur) if isinstance(cur, list) else cur
            return _int(n, f"resolution.{path}"), f"resolution.{path}", None
    cr = run.json("census_reconcile")
    if isinstance(cr, dict) and isinstance(cr.get("totals"), dict) and "ambiguous" in cr["totals"]:
        return _int(cr["totals"]["ambiguous"], "census_reconcile.totals.ambiguous"), "census_reconcile.totals.ambiguous (the resolver report has no ambiguity field)", "census_reconcile"
    return None, "unavailable", None


@_gate("G2")
def gate_G2(run: RunArtifacts) -> GateResult:
    """Semantic closure: the resolution report has 0 Unsupported (and 0 ResolutionErrors), the fold 0 Unresolved (and no
    other fold error: any fold error leaves an instance unemitted), and 0 ambiguous pattern matches (the census
    reconciliation's ambiguity count stands in when the resolver report has no ambiguity field)."""
    res = run.need("resolution")
    fs = run.need("fold_summary")
    ev = [run.rel("resolution"), run.rel("fold_summary")]
    problems: list[str] = []
    unsupported = _int(_get(res, "totals.unsupported", "resolution"), "resolution.totals.unsupported")
    res_errors = _int(_get(res, "totals.errors", "resolution"), "resolution.totals.errors")
    fold_unsupported = _int(_get(fs, "unsupported", "fold_summary"), "fold_summary.unsupported")
    fold_unresolved = _int(_get(fs, "unresolved", "fold_summary"), "fold_summary.unresolved")
    fold_errors = _int(_get(fs, "errors", "fold_summary"), "fold_summary.errors")
    if unsupported:
        problems.append(f"{unsupported} Unsupported resolutions ({list((res.get('primary_report') or {}).get('unsupported_ops') or {})[:4]})")
    if res_errors:
        problems.append(f"{res_errors} steps with a ResolutionError")
    if fold_unsupported:
        problems.append(f"fold: {fold_unsupported} Unsupported")
    if fold_unresolved:
        problems.append(f"fold: {fold_unresolved} Unresolved reads")
    if fold_errors:
        problems.append(f"fold: {fold_errors} errors {fs.get('error_classes')}")
    steps = _int(_get(res, "totals.steps", "resolution"), "resolution.totals.steps")
    exp = run.expected_steps()
    if exp is not None and steps != exp:
        problems.append(f"resolution covers {steps} steps, the workload takes {exp}")
    if res.get("header_mismatches"):
        problems.append(f"log header disagrees with the profile's C: {res['header_mismatches']}")
    amb, amb_src, amb_key = _ambiguity(res, run)
    if amb_key:
        ev.append(run.rel(amb_key))
    if amb is None:
        raise _Skip("no ambiguity count: the resolver report has no ambiguity field and census_reconcile.json is missing", ev,
                    {"unsupported": unsupported, "fold_unresolved": fold_unresolved})
    if amb:
        problems.append(f"{amb} ambiguous pattern matches ({amb_src})")
    detail = {"unsupported": unsupported, "resolution_errors": res_errors, "fold_unsupported": fold_unsupported,
              "fold_unresolved": fold_unresolved, "fold_errors": fold_errors, "fold_error_classes": fs.get("error_classes"),
              "ambiguous": amb, "ambiguity_source": amb_src, "steps": steps, "expected_steps": exp,
              "computes": (res.get("totals") or {}).get("computes"), "compute_rows": (res.get("totals") or {}).get("compute_rows"),
              "n_instances": fs.get("n_instances"), "header_mismatches": res.get("header_mismatches")}
    return _result(problems, ev, detail)


# ---------------------------------------------------------------------------------------------------------
# G3 — root closure
# ---------------------------------------------------------------------------------------------------------


@_gate("G3")
def gate_G3(run: RunArtifacts) -> GateResult:
    """Root closure: 0 unknown roots (fold summary and resolution report), 0 opaque / prescribed fallbacks, and
    `root_policy.json` status pass with `new_root_classes == []` and no failed root validation."""
    fs = run.need("fold_summary")
    ev = [run.rel("fold_summary")]
    problems: list[str] = []
    unknown = list(fs.get("unknown_roots") or [])
    if unknown:
        problems.append(f"fold: {len(unknown)} unknown roots: {unknown[:4]}")
    res = run.json("resolution")
    if isinstance(res, dict):
        ev.append(run.rel("resolution"))
        ru = list(((res.get("roots") or {}).get("unknown")) or [])
        if ru:
            problems.append(f"resolution: {len(ru)} unknown roots: {ru[:4]}")
    opaque = _int(fs.get("opaque_fallbacks", 0), "fold_summary.opaque_fallbacks")
    if opaque:
        problems.append(f"fold: {opaque} opaque / prescribed fallbacks")
    rp = run.need("root_policy")
    ev.append(run.rel("root_policy"))
    if rp.get("status") != "pass":
        problems.append(f"root_policy status {rp.get('status')!r}" + (f": {rp['reason']}" if rp.get("reason") else ""))
    new_classes = list(rp.get("new_root_classes") or [])
    if new_classes:
        problems.append(f"{len(new_classes)} new root classes: {new_classes[:4]}")
    classes = rp.get("classes") or {}
    failed_total = sum(int((c or {}).get("failed") or 0) for c in classes.values() if isinstance(c, dict))
    if failed_total:
        problems.append(f"{failed_total} root validations failed")
    roots = rp.get("roots") or []
    bad_roots = [r for r in roots if isinstance(r, dict) and r.get("status") not in ("pass", "validated", "ok")]
    if bad_roots:
        problems.append(f"{len(bad_roots)} roots not validated: {[r.get('name') for r in bad_roots[:3]]}")
    opaque_roots = [r.get("name") for r in roots if isinstance(r, dict) and str(r.get("class", "")).lower() in ("opaque", "prescribed")]
    if opaque_roots:
        problems.append(f"{len(opaque_roots)} opaque/prescribed roots in root_policy: {opaque_roots[:3]}")
    bad_alias = [a for a in (rp.get("alias_claims") or []) if isinstance(a, dict) and a.get("status") not in ("pass", "proven", "ok")]
    if bad_alias:
        problems.append(f"{len(bad_alias)} alias claims unproven: {[a.get('name') for a in bad_alias[:3]]}")
    detail = {"unknown_roots": unknown, "opaque_fallbacks": opaque, "unbound_roots": fs.get("unbound_roots"),
              "root_policy_status": rp.get("status"), "new_root_classes": new_classes,
              "classes": {k: dict(v) if isinstance(v, dict) else v for k, v in classes.items()},
              "roots": len(roots), "roots_failed": len(bad_roots), "alias_claims": len(rp.get("alias_claims") or []),
              "alias_unproven": len(bad_alias)}
    return _result(problems, ev, detail)


# ---------------------------------------------------------------------------------------------------------
# G4 — primitive exactness (tier a replay + quarantine differential tests)
# ---------------------------------------------------------------------------------------------------------


def quarantine_specs(res: dict | None) -> list[str]:
    """Specializations the resolution report attributes to a `quarantine` registry module (heuristic over the fields the
    resolver may expose: `quarantine_specs`, `spec_modules`/`by_spec_module`, or a label mentioning `quarantine`)."""
    if not isinstance(res, dict):
        return []
    out: list[str] = []
    q = res.get("quarantine_specs")
    if isinstance(q, list):
        out += [str(x) for x in q]
    for key in ("spec_modules", "by_spec_module", "registries"):
        m = res.get(key) or (res.get("primary_report") or {}).get(key)
        if isinstance(m, dict):
            out += [str(s) for s, mod in m.items() if "quarantine" in str(mod)]
    for s in list((res.get("by_spec") or {}).keys()):
        if "quarantine" in str(s).lower():
            out.append(str(s))
    return sorted(set(out))


@_gate("G4")
def gate_G4(run: RunArtifacts) -> GateResult:
    """Primitive exactness: the tier-a replay passes with `k >= TIER_A_K` and no uncovered specialization; when the run
    uses newly registered definitions, `quarantine.json` must show every differential test passed (passed == total > 0)
    and `negative_matcher_tests: pass` — missing while the resolution report lists quarantine specs is a failure."""
    rep = run.need("replay_a")
    ev = [run.rel("replay_a")]
    problems: list[str] = []
    status, k = rep.get("status"), rep.get("k")
    uncovered = list(rep.get("uncovered_specs") or [])
    cov = rep.get("coverage") or {}
    if status != "pass":
        problems.append(f"tier a status {status!r}" + (f": {rep['reason']}" if rep.get("reason") else ""))
    if not isinstance(k, int) or isinstance(k, bool):
        problems.append(f"tier a k={k!r} is not an integer")
    elif k < TIER_A_K:
        problems.append(f"tier a k={k} < TIER_A_K={TIER_A_K}")
    if uncovered:
        problems.append(f"{len(uncovered)} uncovered specializations: {uncovered[:4]}")
    fails = {s: c for s, c in cov.items() if isinstance(c, dict) and (c.get("fail") or 0)}
    if fails:
        problems.append(f"instances failed in {len(fails)} specializations: {sorted(fails)[:4]}")
    never = sorted(s for s, c in cov.items() if isinstance(c, dict) and int(c.get("evaluated") or 0) == 0 and s not in uncovered)
    if never:
        problems.append(f"{len(never)} specializations evaluated 0 times: {never[:4]}")
    n_pass = sum(int(c.get("pass") or 0) for c in cov.values() if isinstance(c, dict))
    n_fail = sum(int(c.get("fail") or 0) for c in cov.values() if isinstance(c, dict))
    n_skip = sum(int(c.get("skipped") or 0) for c in cov.values() if isinstance(c, dict))
    detail: dict[str, Any] = {"status": status, "k": k, "seed": rep.get("seed"), "specs": len(cov), "covered": len(cov) - len(uncovered),
                              "uncovered_specs": uncovered, "pass": n_pass, "fail": n_fail, "skipped": n_skip,
                              "seconds": rep.get("seconds"), "max_rss_gb": rep.get("max_rss_gb")}
    # -- quarantine (new registered definitions)
    res = run.json("resolution")
    qspecs = quarantine_specs(res)
    q = run.json("quarantine")
    detail["quarantine_specs"] = qspecs
    if q is None:
        detail["quarantine"] = "absent"
        if qspecs:
            ev.append(run.rel("quarantine"))
            problems.append(f"quarantine.json missing while the run uses quarantine specs {qspecs[:3]}")
    else:
        ev.append(run.rel("quarantine"))
        diff = q.get("differential") or {}
        passed, total = diff.get("passed"), diff.get("total")
        neg = q.get("negative_matcher_tests")
        detail["quarantine"] = {"differential": {"passed": passed, "total": total}, "negative_matcher_tests": neg, "status": q.get("status")}
        if not isinstance(passed, int) or not isinstance(total, int) or total <= 0:
            problems.append(f"quarantine differential tests: {passed}/{total} (need total > 0)")
        elif passed != total:
            problems.append(f"quarantine differential tests {passed}/{total} passed")
        if neg != "pass":
            problems.append(f"quarantine negative_matcher_tests = {neg!r}")
        if q.get("status") not in (None, "pass"):
            problems.append(f"quarantine status {q.get('status')!r}")
    return _result(problems, ev, detail)


# ---------------------------------------------------------------------------------------------------------
# G5 — composition exactness (tier b replay of step 0 and every decode snapshot step)
# ---------------------------------------------------------------------------------------------------------

@_gate("G5")
def gate_G5(run: RunArtifacts) -> GateResult:
    """Composition exactness (D82): exhaustive provenance/structure, sampled numerical semantics.

    (a) `wiring.json` — every operand read of every instance is reproduced by the independent last-writer map
        (`n_mismatch == 0`, `uncovered == 0`, `reads > 0`), the run is on one stream, and no journaled write lies
        outside the schema-declared write set of its dispatcher op.
    (b) `replay_a.json` — status pass (the verifier-seeded stratified sample: every (specialization, regime) stratum
        has its required passing evaluations; bodies are G4's view of the same report), and `tokens`: every served
        token at every snapshot step reproduced from its observed logits, with step 0 and >= MIN_DECODE_TOKEN_STEPS
        decode steps present.  This proves TokenSelect given the logits; the logits themselves are covered by (a)+(b).
    (c) `replay_chain.json` — >= CHAIN_MIN_SLICES connected slices replayed with computed intermediate values, all
        pass, each with >= 1 chained edge.
    Full replay (`replay_b_step*.json`) is diagnostic: summarized in `detail.full_replay` when present, never gating."""
    problems: list[str] = []
    ev: list[str] = []
    detail: dict[str, Any] = {}
    # (a) exhaustive provenance
    w = run.need("wiring")
    ev.append(run.rel("wiring"))
    st = w.get("streams") or {}
    sc = w.get("schema") or {}
    detail["provenance"] = {"status": w.get("status"), "reads": w.get("reads"), "writes": w.get("writes"), "instances": w.get("instances"),
                            "n_mismatch": w.get("n_mismatch"), "uncovered": w.get("uncovered"), "single_stream": st.get("single_stream"),
                            "streams": st.get("distinct_streams"), "writes_outside_schema": sc.get("writes_outside_schema"),
                            "schema_roots": sc.get("schema_roots"), "triton_roots": sc.get("triton_roots"),
                            "declared_unwritten": sc.get("declared_unwritten"), "seconds": w.get("seconds")}
    if w.get("status") != "pass":
        problems.append(f"wiring status {w.get('status')!r}: {'; '.join(w.get('failures') or [])}")
    if (w.get("n_mismatch") or 0) > 0:
        problems.append(f"{w['n_mismatch']} operand read(s) not reproduced by the independent provenance map (first: {(w.get('mismatches') or [{}])[0]})")
    if (w.get("uncovered") or 0) > 0:
        problems.append(f"{w['uncovered']} read(s) touch bytes with no journaled producer")
    if not isinstance(w.get("reads"), int) or w.get("reads", 0) <= 0:
        problems.append("wiring: no operand reads checked")
    if st and st.get("single_stream") is False:
        problems.append(f"events on several streams {st.get('distinct_streams')}: log order is not execution order")
    if not st:
        problems.append("wiring: stream census absent (run provenance with the raw log)")
    if sc and (sc.get("writes_outside_schema") or 0) > 0:
        problems.append(f"{sc['writes_outside_schema']} journaled write(s) outside the schema-declared write set")
    # (b) sampled semantics + served tokens
    rep = run.need("replay_a")
    ev.append(run.rel("replay_a"))
    toks = rep.get("tokens") or {}
    detail["sample"] = {"status": rep.get("status"), "k": rep.get("k"), "seed": rep.get("seed"), "strata": rep.get("strata"),
                        "specs": rep.get("n_specs"), "specs_covered": rep.get("n_specs_covered"), "summary": rep.get("summary")}
    if rep.get("status") != "pass":
        problems.append(f"tier a status {rep.get('status')!r}" + (f": {rep['status_reason']}" if rep.get("status_reason") else ""))
    if not toks:
        problems.append("tier a report has no `tokens` section (served-token reproduction)")
    else:
        steps = [int(x) for x in (toks.get("steps") or [])]
        decode = [x for x in steps if x > 0]
        detail["tokens"] = {"steps": steps, "evaluated": toks.get("evaluated"), "equal": toks.get("equal"), "skipped": toks.get("skipped"),
                            "mismatch": (toks.get("mismatch") or [])[:5], "all_rows_equal": toks.get("all_rows_equal")}
        if 0 not in steps:
            problems.append("served tokens: step 0 (prefill) not checked")
        if len(decode) < MIN_DECODE_TOKEN_STEPS:
            problems.append(f"served tokens checked at {len(decode)} decode step(s) < MIN_DECODE_TOKEN_STEPS={MIN_DECODE_TOKEN_STEPS}")
        if toks.get("mismatch"):
            problems.append(f"{len(toks['mismatch'])} served token(s) not reproduced from the logits boundary: {toks['mismatch'][:3]}")
        if toks.get("skipped"):
            problems.append(f"{toks['skipped']} TokenSelect instance(s) skipped (missing logits snapshot?)")
        if not toks.get("evaluated"):
            problems.append("no TokenSelect instance evaluated")
    # (c) chained slices
    ch = run.need("replay_chain")
    ev.append(run.rel("replay_chain"))
    detail["chain"] = {"status": ch.get("status"), "n_slices": ch.get("n_slices"), "n_edges_chained": ch.get("n_edges_chained"),
                       "slices": [{"step": sl.get("step"), "status": sl.get("status"), "edges": sl.get("edges_chained"),
                                   "specs": [str(x).split("{")[0] for x in sl.get("specs") or []]} for sl in (ch.get("slices") or [])],
                       "seconds": ch.get("seconds")}
    if ch.get("status") != "pass":
        problems.append(f"chain status {ch.get('status')!r}" + (f": {ch.get('status_reason') or ch.get('reason')}" if (ch.get('status_reason') or ch.get('reason')) else ""))
    if (ch.get("n_slices") or 0) < CHAIN_MIN_SLICES:
        problems.append(f"{ch.get('n_slices') or 0} chained slice(s) < CHAIN_MIN_SLICES={CHAIN_MIN_SLICES}")
    if ch.get("n_slices") and any(int(sl.get("edges_chained") or 0) < 1 for sl in ch.get("slices") or []):
        problems.append("a chained slice chained no producer->consumer edge")
    # diagnostic: full replay when present
    full: dict[str, Any] = {}
    for p in sorted(run.run_dir.glob("replay_b_step*.json")):
        m = re.search(r"step(\d+)\.json$", p.name)
        if not m:
            continue
        rb = run.json(replay_b_key(int(m.group(1))))
        if isinstance(rb, dict):
            full[m.group(1)] = {"status": rb.get("status"), "partial": bool(rb.get("partial")), "n_instances": rb.get("n_instances"),
                                "n_diverged": rb.get("n_diverged"), "tokens_equal": (rb.get("tokens") or {}).get("equal"),
                                "n_from_earlier_steps": rb.get("n_from_earlier_steps")}
    detail["full_replay"] = full or "not run (diagnostic only)"
    return _result(problems, ev, detail)


# ---------------------------------------------------------------------------------------------------------
# G6 — canonicality + regression
# ---------------------------------------------------------------------------------------------------------


@_gate("G6")
def gate_G6(run: RunArtifacts) -> GateResult:
    """Canonicality + regression: two independent folds (separate processes) of the same log give one digest, and
    `golden.json` (`verity_vllm.properties.golden --check`) passes with every entry's digest and attribution sha as expected.
    Equal digests across two captures of one workload are NOT required (recorded as detail when available)."""
    a = run.need("fold_summary")
    b = run.need("fold_summary_2")
    ev = [run.rel("fold_summary"), run.rel("fold_summary_2")]
    problems: list[str] = []
    da, db = a.get("digest"), b.get("digest")
    if not da or not db:
        problems.append("a fold ran without Program() (no digest)" if (a.get("with_program") is False or b.get("with_program") is False) else "empty digest")
    elif da != db:
        problems.append(f"digests differ: {str(da)[:16]}… vs {str(db)[:16]}…")
    if a.get("n_instances") != b.get("n_instances"):
        problems.append(f"instance counts differ: {a.get('n_instances')} vs {b.get('n_instances')}")
    detail: dict[str, Any] = {"digest": da, "digest_2": db, "digests_equal": bool(da) and da == db, "n_instances": a.get("n_instances"),
                              "spec": a.get("spec"), "seconds": [a.get("seconds"), b.get("seconds")], "max_rss_gb": [a.get("max_rss_gb"), b.get("max_rss_gb")]}
    other = run.json("fold_summary_other")
    if isinstance(other, dict):
        ev.append(run.rel("fold_summary_other"))
        detail["other_capture"] = {"digest": other.get("digest"), "log": other.get("log"), "equal": bool(da) and other.get("digest") == da}
    g = run.need("golden")
    ev.append(run.rel("golden"))
    if g.get("status") != "pass":
        problems.append(f"golden status {g.get('status')!r}" + (f": {g['reason']}" if g.get("reason") else ""))
    entries = g.get("entries") or []
    bad = [e for e in entries if isinstance(e, dict) and (e.get("status") != "pass" or (e.get("expected_digest") is not None and e.get("expected_digest") != e.get("got_digest"))
                                                        or (e.get("expected_attribution_sha") is not None and e.get("expected_attribution_sha") != e.get("got_attribution_sha")))]
    if bad:
        problems.append(f"{len(bad)} golden entries changed: {[e.get('name') for e in bad[:4]]}")
    detail["golden"] = {"status": g.get("status"), "entries": len(entries), "changed": [e.get("name") for e in bad]}
    return _result(problems, ev, detail)


# ---------------------------------------------------------------------------------------------------------
# G7 — holdout generalization
# ---------------------------------------------------------------------------------------------------------


@_gate("G7")
def gate_G7(run: RunArtifacts) -> GateResult:
    """Holdout generalization: `holdouts.json` (written by the orchestrator) is present with at least one holdout and
    G1..G6 pass in every holdout."""
    h = run.json("holdouts")
    ev = [run.rel("holdouts")]
    if h is None:
        raise _Skip("holdouts.json missing (the orchestrator has not run the holdout workloads)", ev)
    holdouts = h.get("holdouts") or []
    problems: list[str] = []
    if not holdouts:
        problems.append("no holdouts")
    if h.get("status") not in (None, "pass"):
        problems.append(f"holdouts status {h.get('status')!r}")
    per = {}
    for ho in holdouts:
        if not isinstance(ho, dict):
            problems.append(f"malformed holdout entry {ho!r}")
            continue
        gates = ho.get("gates") or {}
        statuses = {g: (gates.get(g, {}).get("status") if isinstance(gates.get(g), dict) else gates.get(g)) for g in HOLDOUT_GATES}
        failing = sorted(g for g, s in statuses.items() if s != "pass")
        per[str(ho.get("name"))] = {"workload": ho.get("workload"), "run_dir": ho.get("run_dir"), "gates": statuses, "pass": not failing}
        if failing:
            problems.append(f"holdout {ho.get('name')}: {failing} not passing")
    detail = {"status": h.get("status"), "n_holdouts": len(holdouts), "holdouts": per}
    return _result(problems, ev, detail)


# ---------------------------------------------------------------------------------------------------------
# G8 — VU partition (structural; outside the frozen G1..G7 set)
# ---------------------------------------------------------------------------------------------------------


@_gate("G8")
def gate_G8(run: RunArtifacts) -> GateResult:
    """VU partition: `query_use.json` (`verity-ir/query-use/v1`, written by the partition track: use configuration + `VuVerdict`).
    Pass iff `verdict.partition.status == "exact"` and `verdict.width.status == "ok"` (no violators, no unsupported); fail with
    the partition codes / width violators otherwise.  Absent document -> `skipped` ("no query-use document"); a document that
    is not this schema, or lacks the verdict fields, -> `error`.  Reads only fields of the frozen document shape (the freeze
    doc); never touches G1..G7."""
    doc = run.json("query_use")
    ev = [run.rel("query_use")]
    if doc is None:
        raise _Skip("no query-use document", ev)
    what = run.rel("query_use")
    if not isinstance(doc, dict):
        raise ArtifactError(f"{what}: not a JSON object")
    if doc.get("schema") != QUERY_USE_SCHEMA:
        raise ArtifactError(f"{what}: schema {doc.get('schema')!r} != {QUERY_USE_SCHEMA!r}")
    use = doc.get("use") if isinstance(doc.get("use"), dict) else {}
    v = _get(doc, "verdict", what)
    part = _get(v, "partition", f"{what}.verdict")
    width = _get(v, "width", f"{what}.verdict")
    p_status = _get(part, "status", f"{what}.verdict.partition")
    w_status = _get(width, "status", f"{what}.verdict.width")
    codes = [str(c) for c in (part.get("codes") or [])]
    violators = list(width.get("violators") or [])
    unsupported = list(width.get("unsupported") or [])
    w_max = width.get("w_max", use.get("w_max"))
    v_status = v.get("status")                      # "ok" | "fail" | "not-ready" | "unsupported" (qpart INTERFACE NOTE 21:45Z)
    # Two classes of finding (lead's decision 22:22Z).  FAIL: a wrong partition (overlap / gap), a width violation, `not-ready` (a
    # Definition / gate wider than w_max inside the partition), the evaluator's own `fail`, a foreign document.  UNSUPPORTED: the
    # evaluator could not decide (partition / width / status `unsupported`, member classes with undecidable w_out) -- neither pass nor
    # fail: G8 is `unsupported`, the run verdict INCOMPLETE, never REJECTED on that basis.  A fail-class finding dominates.
    problems: list[str] = []
    undecided: list[str] = []
    if use.get("kind") != QUERY_USE_KIND:
        problems.append(f"use.kind {use.get('kind')!r} != {QUERY_USE_KIND!r}")
    if p_status == "unsupported":
        undecided.append("partition 'unsupported'" + (f" {codes[:6]}" if codes else ""))
    elif p_status != "exact":
        problems.append(f"partition {p_status!r}" + (f" {codes[:6]}" if codes else ""))
    if w_status == "unsupported":
        undecided.append("width 'unsupported'")
    elif w_status != "ok":
        problems.append(f"width {w_status!r}")
    if violators:
        problems.append(f"{len(violators)} member class(es) with w_out > w_max={w_max}: {violators[:4]}")
    if unsupported:
        undecided.append(f"{len(unsupported)} member class(es) with undecidable w_out: {unsupported[:4]}")
    if v_status == "unsupported":
        undecided.append("verdict.status 'unsupported'")
    elif v_status != "ok":                          # `fail` / `not-ready` (and anything unknown): surfaced verbatim, not a pass
        problems.append(f"verdict.status {v_status!r}")
    # binding: the companion query document (`RUN_DIR/query.json`, qpart 22:14Z) must carry the same binding_id and the run's digest
    binding: dict[str, Any] = {"checked": False}
    qdoc = run.json("query_doc")
    if isinstance(qdoc, dict):
        ev.append(run.rel("query_doc"))
        fs = run.json("fold_summary")
        digest = fs.get("digest") if isinstance(fs, dict) else None
        binding = {"checked": True, "query_binding_id": qdoc.get("binding_id"), "query_program_digest": qdoc.get("program_digest"),
                   "run_digest": digest, "query_id": qdoc.get("query_id")}
        if qdoc.get("binding_id") != doc.get("binding_id"):
            problems.append("binding: query.json binding_id != query_use.json binding_id")
        if digest is not None and qdoc.get("program_digest") != digest:
            problems.append("binding: query.json program_digest != the run's fold digest")
    rd = v.get("readiness_32") if isinstance(v.get("readiness_32"), dict) else {}
    detail: dict[str, Any] = {
        "schema": doc.get("schema"), "binding_id": doc.get("binding_id"), "use": {"kind": use.get("kind"), "w_max": use.get("w_max")},
        "status": v_status,                          # the evaluator's VuVerdict.status, verbatim (lead 22:22Z)
        "partition": {"status": p_status, "codes": codes[:32]},
        "width": {"status": w_status, "w_max": w_max, "max_seen": width.get("max_seen"), "n_violators": len(violators),
                  "violators": violators[:16], "n_unsupported": len(unsupported), "unsupported": unsupported[:16]},
        "count": v.get("count"), "verdict_status": v_status,
        "readiness_32": {"ok": rd.get("ok"), "max_w": rd.get("max_w"), "n_violators": len(rd.get("violators") or [])} if rd else None,
        "evaluator": doc.get("evaluator"),
        "binding": binding, "binding_checked": binding["checked"],
        "undecided": undecided,
    }
    if problems:
        return _result(problems, ev, detail)
    if undecided:
        return GateResult("", "", "unsupported", ev, detail, "unsupported (" + "; ".join(undecided) + ")")
    return _result([], ev, detail)


# ---------------------------------------------------------------------------------------------------------
# R1 — reference integrity, fail-closed (outside the frozen G1..G7 set)
# ---------------------------------------------------------------------------------------------------------

_INTEGRITY_COUNTS = ("errors", "unresolved", "unsupported")


def _count_field(d: dict, key: str, what: str, problems: list[str], unverified: list[str]) -> int | None:
    """An integer count from a summary; a missing / non-integer / negative value is *unverified* evidence (never zero)."""
    if key not in d:
        unverified.append(f"{what}.{key} missing")
        return None
    v = d[key]
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        unverified.append(f"{what}.{key} = {v!r} is not a non-negative integer")
        return None
    if v > 0:
        problems.append(f"{what}.{key} = {v}")
    return v


def fold_integrity(fold_summary: Any, fold_summary_2: Any = None, resolution: Any = None) -> tuple[str, list[str], dict]:
    """The pure R1 rule over the parsed artifacts: `("pass" | "fail" | "unverified", problems, detail)`.

    * `fail`       — any fold error / Unresolved read / Unsupported resolution (either fold summary), or any ResolutionError /
                     Unsupported in the resolution report's totals: the Program is NOT eligible as an accepted reference;
    * `unverified` — no fold summary, a count that is missing or not an integer, or inconsistent evidence (`error_classes`
                     histogram or `per_step` error sum disagreeing with `errors`): missing evidence is never read as zero;
    * `pass`       — every count present, integer and zero.
    A `fail` finding dominates `unverified`.  `fold_summary_2` / `resolution` are optional inputs (checked when present)."""
    problems: list[str] = []
    unverified: list[str] = []
    detail: dict[str, Any] = {"rule": dict(REF_INTEGRITY_RULE), "fold_summary": None, "fold_summary_2": None, "resolution": None}
    for what, fs in (("fold_summary", fold_summary), ("fold_summary_2", fold_summary_2)):
        if fs is None:
            if what == "fold_summary":
                unverified.append("fold_summary.json missing: no fold error evidence")
            continue
        if not isinstance(fs, dict):
            unverified.append(f"{what}.json is not a JSON object")
            continue
        counts = {k: _count_field(fs, k, what, problems, unverified) for k in _INTEGRITY_COUNTS}
        classes = fs.get("error_classes")
        n_err = counts["errors"]
        if isinstance(classes, dict) and n_err is not None:
            try:
                total = sum(int(v) for v in classes.values())
            except (TypeError, ValueError):
                total = None
            if total is None:
                unverified.append(f"{what}.error_classes has non-integer counts")
            elif total != n_err:
                unverified.append(f"{what}.error_classes sums to {total}, errors = {n_err}")
        per_step = fs.get("per_step")
        if isinstance(per_step, list) and per_step and n_err is not None:
            vals = [s.get("errors") for s in per_step if isinstance(s, dict)]
            if all(isinstance(v, int) and not isinstance(v, bool) for v in vals) and len(vals) == len(per_step):
                if sum(vals) != n_err:
                    unverified.append(f"{what}.per_step errors sum to {sum(vals)}, errors = {n_err}")
        detail[what] = {**counts, "error_classes": classes if isinstance(classes, dict) else None, "digest": fs.get("digest"),
                        "with_program": fs.get("with_program"), "n_instances": fs.get("n_instances")}
    if resolution is not None:
        if not isinstance(resolution, dict) or not isinstance(resolution.get("totals"), dict):
            unverified.append("resolution.json has no `totals` object")
        else:
            t = resolution["totals"]
            rc = {k: _count_field(t, k, "resolution.totals", problems, unverified) for k in ("errors", "unsupported")}
            detail["resolution"] = rc
    status = "fail" if problems else ("unverified" if unverified else "pass")
    detail["unverified"] = unverified
    return status, problems, detail


@_gate("R1")
def gate_R1(run: RunArtifacts) -> GateResult:
    """Reference integrity, fail-closed (`REF_INTEGRITY_RULE`): `fold_integrity` over `fold_summary.json`, `fold_summary_2.json`
    (when present) and `resolution.json` (when present).  Never `skipped`: a missing / unreadable summary is `unverified`."""
    ev = [run.rel("fold_summary")]
    arts: dict[str, Any] = {}
    unreadable: list[str] = []
    for key in ("fold_summary", "fold_summary_2", "resolution"):
        try:
            arts[key] = run.json(key)
        except ArtifactError as e:
            arts[key] = None
            unreadable.append(f"{run.rel(key)}: {e}")
        if key != "fold_summary" and arts[key] is not None:
            ev.append(run.rel(key))
    status, problems, detail = fold_integrity(arts["fold_summary"], arts["fold_summary_2"], arts["resolution"])
    if unreadable:
        detail["unreadable"] = unreadable
        if status == "pass":
            status = "unverified"
    if status == "fail":
        reason = "fold/construction errors > 0: " + "; ".join(problems) + f" [rule {REF_INTEGRITY_RULE['id']}/{REF_INTEGRITY_RULE['version']}]"
    elif status == "unverified":
        reason = "unverified (" + "; ".join(detail["unverified"] + unreadable) + f") [rule {REF_INTEGRITY_RULE['id']}/{REF_INTEGRITY_RULE['version']}]"
    else:
        reason = None
    return GateResult("", "", status, ev, detail, reason)


# ---------------------------------------------------------------------------------------------------------
# evaluation and verdict
# ---------------------------------------------------------------------------------------------------------


def evaluate_gates(run_dir: str | Path, *, manifest: dict | None = None) -> list[GateResult]:
    """The seven frozen gates, G8 and R1 over one run directory, in `ALL_GATES` order.  Never raises for artifact problems."""
    run = RunArtifacts(run_dir, manifest)
    return [_GATE_FNS[g](run) for g in ALL_GATES]


def verdict(results: list[GateResult]) -> str:
    """ACCEPTED iff all seven frozen gates pass; REJECTED if any gate fails; INCOMPLETE otherwise (skipped / error / missing).  Gates outside the frozen set (G8, R1)
    contribute only `fail` -> REJECTED and `error` / `unsupported` / `unverified` -> INCOMPLETE: a run without a query-use document (G8 `skipped`) or a record that
    predates G8 / R1 keeps the verdict of G1..G7.  [R9 ref] R1 `unverified` (no fold error evidence) is INCOMPLETE, never ACCEPTED — the fail-closed rule."""
    return match_verdict([r.result() for r in results], GATES)


def gates_json(results: list[GateResult], run_dir: str | Path | None = None, diagnostics_: dict | None = None) -> dict:
    return {"schema": GATES_SCHEMA, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_dir": str(run_dir) if run_dir is not None else None, "verdict": verdict(results),
            "order": [r.id for r in results], "frozen": list(GATES), "gates": {r.id: r.to_json() for r in results},
            "constants": {"TIER_A_K": TIER_A_K, "MIN_DECODE_TOKEN_STEPS": MIN_DECODE_TOKEN_STEPS, "CHAIN_MIN_SLICES": CHAIN_MIN_SLICES,
                          "REGIMES": list(REGIMES), "COS_SIN_MAX_ULP": COS_SIN_MAX_ULP,
                          "DIAG_MAX_LINEAR_RESIDUAL": DIAG_MAX_LINEAR_RESIDUAL,
                          "REF_INTEGRITY_RULE": f"{REF_INTEGRITY_RULE['id']}/{REF_INTEGRITY_RULE['version']}"},
            "diagnostics": diagnostics_}


def write_gates(run_dir: str | Path, results: list[GateResult], diagnostics_: dict | None = None) -> Path:
    p = Path(run_dir) / ARTIFACTS["gates"]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(gates_json(results, run_dir, diagnostics_), indent=1) + "\n")
    return p


def gates_markdown(results: list[GateResult]) -> str:
    L = ["| gate | name | status | reason |", "| --- | --- | --- | --- |"]
    for r in results:
        L.append(f"| {r.id} | {r.name} | **{r.status}** | {(r.reason or '').replace('|', '/')[:200]} |")
    L.append("")
    L.append(f"verdict: **{verdict(results)}**")
    return "\n".join(L)


# ---------------------------------------------------------------------------------------------------------
# DIAGNOSTICS — reported in the card, never gating
# ---------------------------------------------------------------------------------------------------------

_T_RE = re.compile(r"T=\d+,?")


def strip_T(label: str) -> str:
    """`Attention_v3{T=17,NH=12,...}` -> `Attention_v3{NH=12,...}` (the per-position attention length is workload shape)."""
    return _T_RE.sub("", label).replace(",}", "}")


def _linear_residual(xs: list[float], ys: list[float]) -> float:
    """Max |y - (a + b x)| of the least-squares line (0 when fewer than 3 points)."""
    n = len(xs)
    if n < 3:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
    a = my - b * mx
    return max(abs(y - (a + b * x)) for x, y in zip(xs, ys))


def diag_structure(fs: dict | None) -> dict:
    """Per-decode-step specialization multiset equality modulo attention `T=`, constant instance count per decode step,
    cumulative instance count linear in the step."""
    if not isinstance(fs, dict) or not isinstance(fs.get("per_step"), list) or not fs["per_step"]:
        return {"status": "unavailable"}
    per_step = fs["per_step"]
    decode = [s for s in per_step if s.get("phase") == "decode"]
    prefill = [s for s in per_step if s.get("phase") == "prefill"]
    stripped: list[Counter] = []
    for s in decode:
        c: Counter = Counter()
        for label, n in (s.get("spec_histogram") or {}).items():
            c[strip_T(label)] += int(n)
        stripped.append(c)
    counts = [int(s.get("n_instances", 0)) for s in decode]
    diffs = []
    for s, c in zip(decode, stripped):
        if stripped and c != stripped[0]:
            d = {k: [stripped[0].get(k, 0), c.get(k, 0)] for k in set(stripped[0]) | set(c) if stripped[0].get(k, 0) != c.get(k, 0)}
            diffs.append({"step": s.get("step"), "differs": dict(list(d.items())[:6])})
    xs = [float(s.get("step")) for s in decode]
    cum, acc = [], 0
    for s in per_step:
        acc += int(s.get("n_instances", 0))
        if s.get("phase") == "decode":
            cum.append(float(acc))
    resid = _linear_residual(xs, cum) if len(xs) == len(cum) and xs else float("inf")
    return {"status": "ok", "steps": len(per_step), "prefill_steps": len(prefill), "decode_steps": len(decode),
            "prefill_instances": [int(s.get("n_instances", 0)) for s in prefill], "decode_instances": sorted(set(counts)),
            "histograms_equal_mod_T": not diffs, "histogram_diffs": diffs[:3], "count_constant": len(set(counts)) <= 1,
            "linear_residual": None if math.isinf(resid) else round(resid, 4), "linear": resid <= DIAG_MAX_LINEAR_RESIDUAL,
            "decode_multiset_mod_T": dict(sorted(stripped[0].items())) if stripped else None}


def diag_family_match(cmp_: dict | None) -> dict:
    """The compare report's `structure` section (hand-authored family reference), with its unclassified extra classes."""
    st = (cmp_ or {}).get("structure") if isinstance(cmp_, dict) else None
    if not isinstance(st, dict):
        return {"status": "unavailable"}
    classes = st.get("classes") or {}
    uncl = sorted(k for k in classes if str(k).startswith("unclassified:"))
    return {"status": st.get("status"), "family": st.get("family"), "matched": st.get("matched"), "superset": st.get("superset"),
            "n_extra": st.get("n_extra"), "classes": {k: (v.get("count") if isinstance(v, dict) else v) for k, v in classes.items()},
            "unclassified": len(uncl), "unclassified_classes": uncl, "reason": st.get("reason"),
            "n_derived_nodes": st.get("n_derived_nodes"), "n_family_leaves": st.get("n_family_leaves")}


def diag_overhead(m0: dict | None, versions: dict | None) -> dict:
    """The M0 overhead ladder (step wall per rung, × baseline) with the host's clock-read cost, and the per-recorded-call
    cost with the hook's four clock reads subtracted (M6 addendum: 4 reads per recorded call)."""
    if not isinstance(m0, dict) or not isinstance(m0.get("results"), dict):
        return {"status": "unavailable"}
    host = (versions or {}).get("host") or (m0.get("versions") or {}).get("host") or {}
    clock_us = host.get("clock_read_us")
    base = m0.get("baseline_step_wall_s")
    rungs = {}
    for rung, r in m0["results"].items():
        reps = [x for x in (r.get("reps") or []) if isinstance(x, dict) and x.get("step_wall_mean_s") is not None]
        if not reps:
            continue
        step = sum(x["step_wall_mean_s"] for x in reps) / len(reps)
        calls = sum(int(x.get("recorded_calls") or 0) + int(x.get("triton_launches") or 0) for x in reps) / len(reps)
        steps_n = sum(int(x.get("observed_steps") or 0) for x in reps) / len(reps) or 1
        row: dict[str, Any] = {"label": r.get("label"), "step_wall_ms": round(step * 1e3, 2),
                               "x_baseline": round(step / base, 2) if base else None, "events_per_step": round(calls / steps_n, 1) if calls else 0}
        if base and calls:
            per_call_us = (step - base) / (calls / steps_n) * 1e6
            row["overhead_per_event_us"] = round(per_call_us, 1)
            if clock_us is not None:
                row["overhead_per_event_us_minus_clock"] = round(per_call_us - 4 * float(clock_us), 1)
        rungs[rung] = row
    return {"status": "ok", "baseline_step_wall_ms": round(base * 1e3, 2) if base else None, "reps": m0.get("reps"),
            "host": {k: host.get(k) for k in ("cpu", "nproc", "cpu_quota", "clocksource", "clock_read_us")},
            "note": "compare ladders across pods only when host.clocksource matches; `minus_clock` subtracts the hook's 4 perf_counter reads per event",
            "rungs": rungs}


def diagnostics(run: RunArtifacts | str | Path) -> dict:
    """All diagnostics over a run directory (never gating): structure heuristics, family-reference match, overhead ladder."""
    if not isinstance(run, RunArtifacts):
        run = RunArtifacts(run)

    def safe(fn, *keys):
        try:
            return fn(*[run.json(k) for k in keys])
        except ArtifactError as e:
            return {"status": "error", "reason": str(e)}

    return {"structure": safe(diag_structure, "fold_summary"), "family_match": safe(diag_family_match, "compare"),
            "overhead": safe(diag_overhead, "m0", "versions")}


# ---------------------------------------------------------------------------------------------------------
# the fold summary the driver writes (schema owned here, produced by `bench/run_config.py --worker fold`)
# ---------------------------------------------------------------------------------------------------------

FOLD_SUMMARY_FIELDS: dict[str, str] = {
    "schema": f"`{FOLD_SUMMARY_SCHEMA}`",
    "utc": "UTC time the summary was written",
    "log": "path of the raw observation log folded",
    "profile": "profile module name (`verity_vllm.observe.fold.profiles.<name>`)",
    "profile_id": "`Profile.id` of that module",
    "with_program": "true when `Program()` and the digest were built (G6 needs it)",
    "digest": "`program_digest(Program(Captured_v1{PROFILE, WORKLOAD}))` — the structure digest G6 compares (empty without Program())",
    "spec": "the derived specialization label `Captured_v1{PROFILE=..., WORKLOAD=...}`",
    "n_instances": "instances emitted by the fold (= flat body nodes)",
    "n_steps": "engine steps folded",
    "unsupported": "Unsupported resolutions seen by the fold (`ResolutionReport.unsupported`)",
    "unresolved": "fold errors of class `Unresolved` (memory reads without provenance)",
    "opaque_fallbacks": "reads served by an opaque / prescribed value instead of provenance (0 by construction; fold errors of class Opaque/Prescribed)",
    "errors": "all fold errors recorded in development mode (0 under strict)",
    "error_classes": "histogram of fold error classes",
    "error_examples": "first fold errors (<= 16)",
    "unknown_roots": "weight/buffer roots the profile classifies as `unknown` (via its `root_class`, as `resolve_log` does)",
    "unbound_roots": "roots without a weights field (the fold's `unbound root` notes; ignored fp8-scale placeholders are expected here)",
    "per_step": "list of `{step, phase, rows, n_instances, computes, spec_histogram: {label: instances}, unsupported, errors}`",
    "tokens": "`{request_id: [sampled token ids in step order]}` (the fold's TokenSelect outputs, from SampleEnd)",
    "sampled_steps": "`{request_id: [steps at which the tokens above were sampled]}`",
    "requests": "`[{request_id, prompt_len, max_tokens, n_tokens}]` from the log header and the fold",
    "params": "the Program parameters `[[name, type], ...]`",
    "ret_type": "the Program return type",
    "memory": "memory-model statistics (`MemoryModel.stats()` + storage alloc/free counts)",
    "timings": "`{fold_s, program_s, digest_s}`",
    "seconds": "wall time of the worker",
    "max_rss_gb": "maximum resident set of the worker process (D69 footprint)",
    "fold": "`compare.summarize(result)` verbatim (the layout compare.py's report has under `fold`)",
}


__all__ = [
    "TIER_A_K", "MIN_DECODE_TOKEN_STEPS", "CHAIN_MIN_SLICES", "REGIMES", "COS_SIN_MAX_ULP", "DIAG_MAX_LINEAR_RESIDUAL", "GATES", "GATE_NAMES", "HOLDOUT_GATES",
    "STRUCTURAL_GATES", "INTEGRITY_GATES", "REF_INTEGRITY_RULE", "ALL_GATES", "QUERY_USE_SCHEMA", "QUERY_USE_KIND",
    "GATES_SCHEMA", "FOLD_SUMMARY_SCHEMA", "FOLD_SUMMARY_FIELDS", "ARTIFACTS", "ARTIFACT_STAGE", "replay_b_key", "replay_b_rel",
    "GateResult", "ArtifactError", "RunArtifacts", "load_profile_config",
    "gate_G1", "gate_G2", "gate_G3", "gate_G4", "gate_G5", "gate_G6", "gate_G7", "gate_G8", "gate_R1", "fold_integrity", "quarantine_specs",
    "evaluate_gates", "verdict", "gates_json", "write_gates", "gates_markdown",
    "strip_T", "diag_structure", "diag_family_match", "diag_overhead", "diagnostics",
]
