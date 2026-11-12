"""What the Match's gates stand on: the frozen constants, the artifacts of a run directory and their reader
(`RunArtifacts`), a gate's result (`GateResult`) and the registry the gates enter themselves in (`_gate`).  `gates` hands every
name on; other code reads them through it.  Nothing here imports torch, replay, analytic, root_policy or golden."""

from __future__ import annotations

import importlib
import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from verity_vllm.check.codes import VerificationCode as VC
from verity_vllm.check.result import MISSING_INPUT, CheckResult
from verity_vllm.check.verdict_names import (
    GATE_NAMES,
    GATES,
    INTEGRITY_GATES,
    STRUCTURAL_GATES,
)

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
#: G8 — the use a query-use document (`QUERY_USE_SCHEMA`) declares.
QUERY_USE_KIND = "vu-partition"

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
    def from_json(cls, d: dict) -> GateResult:
        gid, name, status = d["id"], d.get("name", GATE_NAMES.get(d["id"], "")), d["status"]   # read first: an entry that is no object is refused here
        evidence, detail = d.get("evidence"), d.get("detail")   # a result written without either consulted no artifact and details nothing
        return cls(id=gid, name=name, status=status,
                   evidence=list(evidence) if evidence else [], detail=dict(detail) if detail else {}, reason=d.get("reason"))

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
        placed = self.manifest.get("artifacts")   # the manifest's own places, by key; a run that states none keeps every default
        over = placed.get(key) if placed else None
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
        wl = self.manifest.get("workload")
        reqs = wl.get("requests") if wl else None
        if reqs:
            return [dict(r) for r in reqs]
        for key in ("control_tokens", "tokens"):
            tok = self.json(key)
            if isinstance(tok, dict) and tok.get("requests"):
                return [{"request_id": r.get("request_id"), "prompt_len": len(ids) if (ids := r.get("prompt_token_ids")) else 0,
                         "max_tokens": r.get("max_tokens")} for r in tok["requests"]]
        return []

    def expected_steps(self) -> int | None:
        """The engine steps a run of this workload takes.  The control run (`control/tokens.json`, the bare engine
        on the same workload — THE token reference of every gate) records its realized `timing.engine_steps`; that is the
        expectation, so declared arrivals (B8-mixed), EOS before `max_tokens` and chunked prefill are all compared like for like.
        Fallback for older control files: `max(max_tokens)` (one prefill + max_tokens-1 decode steps, every request at step 0),
        as `noninterference.compare` also falls back."""
        ctl = self.json("control_tokens")
        timing = ctl.get("timing") if isinstance(ctl, dict) else None
        if timing and timing.get("engine_steps") is not None:
            return int(timing["engine_steps"])
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
            return {str(r["request_id"]): [int(t) for t in ids] if (ids := r.get("generated_token_ids")) else [] for r in ct["requests"]}, "control/tokens.json"
        ni = self.json("noninterference")
        variants = ni.get("tokens") if isinstance(ni, dict) else None   # the comparison's ids by variant; a report without them has no bare run
        c = variants.get("c_bare") if variants else None
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
