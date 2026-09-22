"""The promotion ladder C0–C5 (coverage wave Part 3 "sharpen what works means", Part 4 item 1) and `ladder.json`.

    C0  runs            the production configuration is reproducible and its execution choices are recorded
    C1  replay          replay/capture reproduces the canonical outputs exactly            (= Level 1)
    C2  representation  a typed `Program` exists and sampled canonical interfaces correspond exactly (= Level 2)
    C3  protocol        valid plan, commitments, staged sampling, negatives, artifact-only re-verification (= Level 3)
    C4  proof           one real representative obligation through the generic SP1 backend    (= Level 4)
    C5  vertical slice  the common INT pipeline runs on the port with no profile-specific protocol logic

C1–C4 are Level 1–4 of `level.py` under their public names; `level.json` keeps being written (HP/TP read and
write it) and `ladder.json` is emitted beside it. Both spellings are accepted on input (`normalize_rung("L2") ==
normalize_rung("C2") == "C2"`); output is always `C<n>`. The rungs are independent claims: `reached` is the highest
rung met, `gaps` names every lower rung not met, and a rung's status is one of `met` / `not-met` / `not-attempted` /
`not-claimed` / `absent` exactly as for the levels. `assert_claim` refuses any claim above the evidence.

**C0 evidence contract.** A production run of the port (`production/` with `production.jsonl`, `run.json`,
`manifest.json`) whose profile manifest (`verity-vllm/profile/v1`) matches the port field by field and records the
execution choices `C0_REQUIRED_FIELDS`: the attention / linear / sampler backends (`backends.*`; `backends.moe`
when the model routes), the compile mode (`compilation.enforce_eager`, `.cudagraph_mode`, `.mode`, `.custom_ops`),
`dtype`, `parallelism.tensor_parallel_size` / `.pipeline_parallel_size`, the GPU (`gpu.name`, `.compute_capability`,
`.driver`), the wheel (`vllm.commit`, `.version`, `.wheel_sha256`), the checkpoint digest (`model.repo`, `.revision`,
`.weights_sha256`), the runtime (`python.torch`, `.triton`, `.cuda`), and the declared `engine_flags` /
`engine_kwargs` — every field present and non-null — and whose production is shown reproducible
(`production_repro.json`: two independent processes agree per request on tokens and chosen logprobs, or the
recorder's in-process second stock pass is identical, the method named). **Kernel identity:** for an eager port it
is fixed by the recorded backends + wheel + GPU (the custom-op CUDA kernels ship in the wheel; the batch-invariant
Triton kernels compile deterministically from it); for a compiled port the Inductor kernel set must be recorded —
`compilation.generated_kernels` digests in the production manifest (W1c) *or* a bundle's `kernel_identity.json`
with a `production` block built from the production's Inductor cache (W1d) — else C0 is `not-met`. Kernel records
(`kernel_identity.json`, `kernels.json`) are cited whenever present.

**C5 evidence contract.** An INT-pipeline record for this port: a `run_composed` output directory (lane INT,
`e2e/run_composed.py` / `e2e/verify_composed.py`, branch `int-e2e-regression`, not merged at 8d2f159) with
`manifest.json` (`case` = the port's checkpoint role; `inputs.production_manifest` agreeing with the port on the
fields it carries), `summary.json` (`verdict: PASS`, every stage `ok`) and the second-process `reverify/reverify.json`
(`verity-vllm/e2e-composed/reverify/v1`: `fresh_verdict == recorded_verdict == PASS`, `agrees: true`) — produced by
the common command invoked with the case name and the port's declared flags only (no profile-specific protocol
logic). Until such a record is supplied (`conformance run --int-record DIR`) C5 is `not-attempted` with that pointer.

**Blocker classes (GUIDANCE §24, "why a profile did not advance").** Every rung that is not met carries a `blocker`
from the closed enum `BLOCKERS`:

    SEMANTICS_MISSING         the kernel's exact numerics are not a registered primitive/composite (the model disagrees or is absent)
    REPRESENTATION_MISSING    the canonical program / verifier description has no counterpart for the configuration (typed program,
                              semantic profile, compact representation)
    CAPTURE_MISSING           production or replay cannot record/reproduce/materialise what the rung needs
    AUTHENTICATION_MISSING    the values exist canonically but the commitment/plan/bundle machinery cannot authenticate them
    PROOF_LOWERING_MISSING    no proof-backend lowering (or no INT pipeline support) for the obligation family
    IMPLEMENTATION_BUG        a defect in verifier/adapter/harness code, not a limit of the architecture (HP's Level-3 failures:
                              profile selection and binding — fixed 2026-09-08)
    NOT_ATTEMPTED             no evidence was sought for the rung

A port may *declare* a blocker per rung in `expectations.json` (`blockers: [{"rung", "class", "detail", "resolved"?}]`);
a declaration wins over the derivation below, and a declared blocker whose rung is now met is kept as history
(`blocker_history`), which is how the matrix tells "was an implementation bug, fixed" from "never blocked". Otherwise the
class is derived: `not-attempted`/`absent` -> NOT_ATTEMPTED; C0/C1 not met -> CAPTURE_MISSING; C2 by the taxonomy outcome
(`primitive-semantics-missing` -> SEMANTICS_MISSING, `compact-representation-missing` -> REPRESENTATION_MISSING,
`replay-cannot-obtain-canonical-value` -> CAPTURE_MISSING, `commitment-layout-cannot-authenticate` -> AUTHENTICATION_MISSING;
`not-claimed` without a taxonomy reason -> REPRESENTATION_MISSING, i.e. no typed program yet; `not-met` with unrecorded
mismatches -> SEMANTICS_MISSING); C3 not met -> AUTHENTICATION_MISSING; C4 not met -> PROOF_LOWERING_MISSING; C5 not met
(an INT record supplied but not verified) -> IMPLEMENTATION_BUG. The headline `blocker` of a ladder is the class of the
first rung above `reached` that is not met (the rung the profile would advance to next).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import SCHEMA_LADDER
from . import taxonomy as TX
from .evidence import ResultsDir, _json, rel
from .level import LEVEL_NAMES, LevelResult, OverClaim

RUNGS: tuple[str, ...] = ("C0", "C1", "C2", "C3", "C4", "C5")
RUNG_NAMES: dict[str, str] = {"C0": "runs", "C1": "replay", "C2": "representation", "C3": "protocol", "C4": "proof", "C5": "vertical-slice"}
RUNG_OF_LEVEL: dict[int, str] = {1: "C1", 2: "C2", 3: "C3", 4: "C4"}
LEVEL_OF_RUNG: dict[str, int] = {v: k for k, v in RUNG_OF_LEVEL.items()}
STATUSES: tuple[str, ...] = ("met", "not-met", "not-attempted", "not-claimed", "absent")
BLOCKERS: tuple[str, ...] = ("SEMANTICS_MISSING", "REPRESENTATION_MISSING", "CAPTURE_MISSING", "AUTHENTICATION_MISSING",
                             "PROOF_LOWERING_MISSING", "IMPLEMENTATION_BUG", "NOT_ATTEMPTED")
_TAXONOMY_TO_BLOCKER: dict[str, str] = {
    TX.PRIMITIVE_SEMANTICS_MISSING: "SEMANTICS_MISSING",
    TX.COMPACT_REPRESENTATION_MISSING: "REPRESENTATION_MISSING",
    TX.REPLAY_CANNOT_OBTAIN_CANONICAL_VALUE: "CAPTURE_MISSING",
    TX.COMMITMENT_LAYOUT_CANNOT_AUTHENTICATE: "AUTHENTICATION_MISSING",
}
_DEFAULT_BLOCKER: dict[str, str] = {"C0": "CAPTURE_MISSING", "C1": "CAPTURE_MISSING", "C2": "REPRESENTATION_MISSING",
                                    "C3": "AUTHENTICATION_MISSING", "C4": "PROOF_LOWERING_MISSING", "C5": "IMPLEMENTATION_BUG"}

_RUNG_RE = re.compile(r"^\s*(?:(?P<c>[Cc])|(?P<l>[Ll](?:evel)?)\s*)?(?P<n>\d)\s*$")

INT_POINTER = ("no INT-pipeline record supplied (`conformance run --int-record DIR`: a `run_composed` output directory with "
               "`manifest.json`, `summary.json`, `reverify/reverify.json` — lane INT, `e2e/run_composed.py` / "
               "`e2e/verify_composed.py`, branch `int-e2e-regression`, not merged at 8d2f159)")
PREDATES_LADDER = "run predates the ladder (level.json only): re-run `conformance run` on the same inputs to evaluate this rung"


def normalize_rung(value: Any) -> str:
    """`"C2"`, `"c2"`, `"L2"`, `"Level 2"`, `2`, `"2"` -> `"C2"`. `L<n>` is a level (1–4); `C<n>` a rung (0–5); a bare
    integer is read as a level when 1–4 (the pre-ladder `--claim N` spelling). Anything else raises `ValueError`."""
    if isinstance(value, bool):
        raise ValueError(f"not a rung: {value!r}")
    if isinstance(value, int):
        if value in RUNG_OF_LEVEL:
            return RUNG_OF_LEVEL[value]
        raise ValueError(f"not a level 1-4: {value!r} (spell a rung as 'C<n>')")
    m = _RUNG_RE.match(str(value))
    if not m:
        raise ValueError(f"not a rung: {value!r} (expected C0..C5 or L1..L4)")
    n = int(m.group("n"))
    if m.group("c"):
        if 0 <= n <= 5:
            return f"C{n}"
        raise ValueError(f"not a rung: {value!r} (C0..C5)")
    if n in RUNG_OF_LEVEL:  # L<n> or a bare digit
        return RUNG_OF_LEVEL[n]
    raise ValueError(f"not a level: {value!r} (L1..L4)")


def rung_index(rung: str) -> int:
    return RUNGS.index(normalize_rung(rung))


@dataclass
class RungResult:
    rung: str
    met: bool
    status: str  # one of STATUSES
    reason: str | None
    evidence: list[dict[str, Any]]
    blocker: dict[str, Any] | None = None  # {"class": BLOCKERS member, "detail", "source"} for a rung that is not met

    def as_json(self) -> dict[str, Any]:
        assert self.status in STATUSES, self.status
        assert self.blocker is None or self.blocker.get("class") in BLOCKERS, self.blocker
        out: dict[str, Any] = {"rung": self.rung, "name": RUNG_NAMES[self.rung], "met": self.met, "status": self.status,
                               "reason": self.reason, "blocker": self.blocker, "evidence": self.evidence}
        if self.rung in LEVEL_OF_RUNG:
            out["level"] = LEVEL_OF_RUNG[self.rung]
        return out


def from_level(level: LevelResult | dict[str, Any]) -> RungResult:
    """A `level.py` result (object or its `level.json` record) as the rung it is."""
    if isinstance(level, LevelResult):
        return RungResult(RUNG_OF_LEVEL[level.level], level.met, level.status, level.reason, list(level.evidence))
    n = int(level["level"])
    return RungResult(RUNG_OF_LEVEL[n], bool(level.get("met")), str(level.get("status")), level.get("reason"), list(level.get("evidence") or []))


# ---------------------------------------------------------------------------------------------------------
# C0 — runs
# ---------------------------------------------------------------------------------------------------------

# (path, meaning) — every field must be present and non-null in the production run's manifest
C0_REQUIRED_FIELDS: tuple[tuple[str, str], ...] = (
    ("backends.attention", "attention backend"),
    ("backends.linear", "linear/GEMM backend"),
    ("backends.sampler", "sampler"),
    ("compilation.enforce_eager", "compile mode: eager"),
    ("compilation.cudagraph_mode", "compile mode: CUDA graphs"),
    ("compilation.mode", "compile mode: torch.compile level"),
    ("compilation.custom_ops", "compile mode: custom ops kept"),
    ("dtype", "dtype"),
    ("parallelism.tensor_parallel_size", "TP size"),
    ("parallelism.pipeline_parallel_size", "PP size"),
    ("gpu.name", "GPU"),
    ("gpu.compute_capability", "compute capability"),
    ("gpu.driver", "driver"),
    ("vllm.commit", "vLLM commit"),
    ("vllm.version", "vLLM version"),
    ("vllm.wheel_sha256", "wheel digest"),
    ("model.repo", "checkpoint repo"),
    ("model.revision", "checkpoint revision"),
    ("model.weights_sha256", "checkpoint weights digest"),
    ("python.torch", "torch"),
    ("python.triton", "triton"),
    ("python.cuda", "CUDA"),
    ("engine_flags", "declared environment flags"),
    ("engine_kwargs", "declared engine kwargs"),
)
# recorded when present; required only under the stated condition
C0_CONDITIONAL_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("backends.moe", "MoE backend", "the port's model routes (adapter case M*)"),
    ("compilation.generated_kernels", "Inductor kernel-set digests", "the port's execution mode is compiled and no bundle kernel_identity.json identifies the production kernels"),
    ("quantization", "quantization scheme", "always present, may be null"),
)
KERNEL_RECORDS: tuple[str, ...] = ("kernel_identity.json", "kernels.json")


def _get(doc: dict[str, Any], dotted: str) -> Any:
    cur: Any = doc
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def execution_choices(manifest: dict[str, Any], *, moe: bool, compiled: bool) -> dict[str, Any]:
    """The recorded execution choices of one profile manifest and the required fields it lacks."""
    choices: dict[str, Any] = {}
    missing: list[str] = []
    for path, _ in C0_REQUIRED_FIELDS:
        v = _get(manifest, path)
        choices[path] = v
        if v is None:
            missing.append(path)
    for path, _, _ in C0_CONDITIONAL_FIELDS:
        v = _get(manifest, path)
        if path == "quantization":
            choices[path] = v
            if "quantization" not in manifest:
                missing.append(path)
            continue
        if v is not None:
            choices[path] = v
        elif path == "backends.moe" and moe:
            missing.append(path)
        # `compilation.generated_kernels` (compiled ports): the kernel identity may instead be recorded by the bundle's
        # `kernel_identity.json` with a `production` block (W1d's compiled-fx rows) — decided in `rung_c0`, not per manifest
    return {"choices": choices, "missing": missing}


def _kernel_record(path: Path) -> dict[str, Any]:
    """Summary of a `kernel_identity.json` / `kernels.json` (TA1's eager kernel table or W1d's production-vs-replay Inductor table)."""
    doc = _json(path)
    rec: dict[str, Any] = {"kind": "kernel-record", "path": rel(path), "required": False}
    if isinstance(doc, dict):
        if isinstance(doc.get("production"), dict):
            rec["production"] = doc["production"]
            rec["kernels_identical"] = doc.get("kernels_identical")
            rec["production_wrapper_graph_hashes"] = doc.get("production_wrapper_graph_hashes")
            rec["identifies_production_kernels"] = True
        elif "kernels" in doc or "distinct_kernels" in doc:
            rec["summary"] = {k: doc[k] for k in ("distinct_kernels", "kernel_files", "wrappers") if k in doc}
    return rec


def rung_c0(production: dict[str, Any], dirs: list[ResultsDir], port) -> RungResult:
    """C0: production reproducible (the production half of Level 1) and every execution choice recorded in the
    production run's manifest (`C0_REQUIRED_FIELDS`); kernel-selection records cited when present."""
    ev: list[dict[str, Any]] = []
    problems: list[str] = []
    moe = port.config.checkpoint_role.upper().startswith("M")
    compiled = port.config.execution_mode != "eager"
    runs_seen = 0
    generated_kernels_recorded = False
    for rd in dirs:
        if not rd.matches_port:
            continue
        for p in rd.productions:
            runs_seen += 1
            mf = p / "manifest.json"
            if not mf.exists():
                problems.append(f"{rel(p)}: no manifest.json (execution choices not recorded)")
                ev.append({"kind": "production-manifest", "path": None, "production": rel(p), "problems": ["manifest.json missing"]})
                continue
            m = _json(mf)
            bad = port.check_manifest(m)
            xc = execution_choices(m, moe=moe, compiled=compiled)
            if xc["choices"].get("compilation.generated_kernels") is not None:
                generated_kernels_recorded = True
            run = _json(p / "run.json") if (p / "run.json").exists() else {}
            rec_problems = ([f"manifest violates the port: {'; '.join(bad)}"] if bad else []) + ([f"missing {', '.join(xc['missing'])}"] if xc["missing"] else [])
            problems += [f"{rel(mf)}: {x}" for x in rec_problems]
            ev.append({"kind": "production-manifest", "path": rel(mf), "profile_id": run.get("profile_id") or rd.profile_id,
                       "execution_choices": xc["choices"], "missing": xc["missing"] or None, "problems": rec_problems or None})
        if rd.commit is not None:
            for name in KERNEL_RECORDS:
                if (rd.commit / name).exists():
                    ev.append(_kernel_record(rd.commit / name))
    if compiled and runs_seen and not generated_kernels_recorded:
        # a compiled configuration's kernel identity is the Inductor kernel set: the manifest's digests (W1c) or the bundle's
        # production-vs-replay kernel table built from the production's Inductor cache (W1d, `kernel_identity.json`)
        kid = [e for e in ev if e["kind"] == "kernel-record" and e.get("identifies_production_kernels")]
        if kid:
            for e in kid:
                e["required"] = True
        else:
            problems.append("kernel identity of the compiled production not recorded (no compilation.generated_kernels in the production "
                            "manifest and no kernel_identity.json with a production block in a bundle of this port)")
    for p in production.get("pairs", []):
        ev.append({"kind": "production-pair", "path": None, "a": p["a"], "b": p["b"], "identical": p["identical"], "requests_shared": p["requests_shared"]})
    if not dirs or not any(rd.matches_port for rd in dirs):
        return RungResult("C0", False, "absent", "no results of this port supplied", ev)
    if production.get("status") == "absent" or runs_seen == 0:
        return RungResult("C0", False, "not-met", "no production run among the supplied results (a commit store records the replay's "
                                                  "execution choices, not a production's)", ev)
    if production.get("status") != "reproducible":
        return RungResult("C0", False, "not-met", f"production not shown reproducible ({production.get('status')}: {production.get('method')})", ev)
    if problems:
        return RungResult("C0", False, "not-met", "execution choices not fully recorded: " + "; ".join(problems), ev)
    recorded = [e for e in ev if e["kind"] == "production-manifest"]
    kernels = [e for e in ev if e["kind"] == "kernel-record"]
    kid = [e for e in kernels if e.get("required")]
    return RungResult("C0", True, "met", f"{len(recorded)} production run(s) with every execution choice recorded (backends, compile mode, dtype, "
                                         f"TP/PP, GPU+driver, wheel, checkpoint digest, runtime"
                                         + ("; Inductor kernel identity from the bundle's kernel_identity.json production block" if kid
                                            else ("; Inductor kernel-set digests in the manifest" if compiled else "")) + f"); {production.get('method')}"
                                         + (f"; kernel-selection records: {len(kernels)}" if kernels else ""), ev)


# ---------------------------------------------------------------------------------------------------------
# C5 — vertical slice (INT pipeline)
# ---------------------------------------------------------------------------------------------------------

REVERIFY_SCHEMA = "verity-vllm/e2e-composed/reverify/v1"
# production-manifest fields INT's manifest.inputs.production_manifest carries that the port declares
_C5_MANIFEST_FIELDS = ("engine_flags", "dtype", "compilation", "gpu", "model")


def check_int_record(path: Path, port) -> dict[str, Any]:
    """One `run_composed` output directory against the C5 evidence contract (module docstring)."""
    problems: list[str] = []
    manifest = _json(path / "manifest.json") if (path / "manifest.json").exists() else None
    summary = _json(path / "summary.json") if (path / "summary.json").exists() else None
    rv_path = next((q for q in (path / "reverify" / "reverify.json", path / "reverify.json") if q.exists()), None)
    reverify = _json(rv_path) if rv_path else None
    if manifest is None:
        problems.append("manifest.json missing")
    if summary is None:
        problems.append("summary.json missing")
    if reverify is None:
        problems.append("reverify/reverify.json missing (no second-process verification)")
    case = (manifest or {}).get("case") or (summary or {}).get("case")
    if case is not None and case != port.config.checkpoint_role:
        problems.append(f"case {case!r} is not the port's checkpoint role {port.config.checkpoint_role!r}")
    stages = {k: (v or {}).get("status") for k, v in ((summary or {}).get("stages") or {}).items()}
    if summary is not None:
        if summary.get("verdict") != "PASS":
            problems.append(f"summary verdict {summary.get('verdict')!r}")
        bad_stages = [k for k, v in stages.items() if v not in ("ok", "skipped", None)]
        if bad_stages:
            problems.append(f"stages not ok: {bad_stages}")
        if summary.get("partial"):
            problems.append("summary is partial")
    if reverify is not None:
        if reverify.get("schema") != REVERIFY_SCHEMA:
            problems.append(f"reverify schema {reverify.get('schema')!r} != {REVERIFY_SCHEMA!r}")
        if not (reverify.get("agrees") and reverify.get("fresh_verdict") == "PASS"):
            problems.append(f"reverify fresh_verdict {reverify.get('fresh_verdict')!r}, recorded {reverify.get('recorded_verdict')!r}, agrees {reverify.get('agrees')!r}")
    pm = ((manifest or {}).get("inputs") or {}).get("production_manifest")
    violations: list[str] = []
    if isinstance(pm, dict):
        # only the fields INT records: check them with the port's rule and keep the ones about fields that are present
        sub = {k: pm[k] for k in _C5_MANIFEST_FIELDS if k in pm}
        sub["schema"] = "verity-vllm/profile/v1"
        sub["case"] = port.config.checkpoint_role
        sub["parallelism"] = {"tensor_parallel_size": port.config.tensor_parallel_size, "pipeline_parallel_size": port.config.pipeline_parallel_size}
        sub["quantization"] = port.config.quantization
        sub["engine_kwargs"] = pm.get("engine_kwargs", port.config.engine_args)
        violations = port.check_manifest(sub)
        if violations:
            problems.append("production manifest violates the port: " + "; ".join(violations))
    command = (manifest or {}).get("command")
    return {"path": rel(path), "case": case, "command": command, "recorded_verdict": (summary or {}).get("verdict"), "stages": stages,
            "reverify": rel(rv_path) if rv_path else None, "fresh_verdict": (reverify or {}).get("fresh_verdict"), "agrees": (reverify or {}).get("agrees"),
            "reverify_rows": {k: (v or {}).get("status") for k, v in ((reverify or {}).get("rows") or {}).items()},
            "production_manifest_violations": violations, "problems": problems, "status": "verified" if not problems else "not-verified"}


def rung_c5(int_records: list[Path], port) -> RungResult:
    if not int_records:
        return RungResult("C5", False, "not-attempted", INT_POINTER, [])
    recs = [check_int_record(Path(p), port) for p in int_records]
    ev = [{"kind": "int-record", **r} for r in recs]
    good = [r for r in recs if r["status"] == "verified"]
    if good:
        return RungResult("C5", True, "met", f"{len(good)} INT-pipeline record(s) re-verified by a second process: " + ", ".join(r["path"] for r in good), ev)
    return RungResult("C5", False, "not-met", "; ".join(f"{r['path']}: {p}" for r in recs for p in r["problems"]), ev)


# ---------------------------------------------------------------------------------------------------------
# the unsupported outcome (taxonomy) of a port
# ---------------------------------------------------------------------------------------------------------


def unsupported_outcome(port, corr: dict[str, Any] | None, rungs: dict[str, RungResult]) -> tuple[str, dict[str, Any] | None]:
    """`(outcome, unsupported)` for `ladder.json`: the port's declaration first (`expectations.json`
    `unsupported_functions[0]`, then an `unsupported` family, then the taxonomy prefix of `static_config`'s
    `NotImplementedError`), else what the correspondence evidence classifies (`taxonomy.py` mapping), else
    `supported-exactly` when C2 is met, else `unclassified` (pending work is not an unsupported outcome)."""
    exp = port.expectations
    if exp.unsupported_functions:
        u = exp.unsupported_functions[0]  # the port lists them in order of importance; the first is the headline
        return u["reason"], TX.unsupported_record(u.get("rung", "C2"), u["reason"], f"{u['function']}: {u['detail']}", "expectations.json unsupported_functions[0]")
    for fam, e in exp.families.items():
        if e.status == "unsupported" and e.unsupported_reason:
            return e.unsupported_reason, TX.unsupported_record("C2", e.unsupported_reason, f"family {fam}: {e.reason}", f"expectations.json families.{fam}")
    tax = getattr(port, "level2_taxonomy", None)
    if tax:
        reason, detail = tax
        return reason, TX.unsupported_record("C2", reason, detail or port.level2_reason, "static_config.py NotImplementedError (taxonomy prefix)")
    c2 = rungs.get("C2")
    if corr and c2 is not None and c2.status == "not-met":
        s = corr.get("summary") or {}
        if int(s.get("missing-operand", 0)):
            return TX.REPLAY_CANNOT_OBTAIN_CANONICAL_VALUE, TX.unsupported_record(
                "C2", TX.REPLAY_CANNOT_OBTAIN_CANONICAL_VALUE, f"{s['missing-operand']} correspondence record(s) with a missing operand", "correspondence.json summary")
        if int(s.get("shape_mismatch", 0)):
            return TX.COMPACT_REPRESENTATION_MISSING, TX.unsupported_record(
                "C2", TX.COMPACT_REPRESENTATION_MISSING, f"{s['shape_mismatch']} shape/type mismatch(es) between captured values and the program", "correspondence.json summary")
        unclassified = int(((corr.get("missing") or {}).get("unclassified")) or 0)
        if unclassified:
            return TX.COMPACT_REPRESENTATION_MISSING, TX.unsupported_record(
                "C2", TX.COMPACT_REPRESENTATION_MISSING, f"{unclassified} captured value(s)/leaves with no counterpart in the program and no classification in expectations.json", "missing_values.json")
    if c2 is not None and c2.met:
        n = len((corr or {}).get("mismatches") or [])
        return (TX.SUPPORTED_EXACTLY + (f"; {n} recorded divergence(s) on bindings" if n else "")), None
    return "unclassified", None


# ---------------------------------------------------------------------------------------------------------
# blocker classes (why a profile did not advance)
# ---------------------------------------------------------------------------------------------------------


def _blocker(cls: str, detail: str | None, source: str) -> dict[str, Any]:
    assert cls in BLOCKERS, cls
    return {"class": cls, "detail": detail, "source": source}


def derive_blocker(rung: RungResult, unsupported: dict[str, Any] | None) -> dict[str, Any] | None:
    """The blocker class of one rung from its status and the taxonomy outcome (module docstring); `None` when met."""
    if rung.met:
        return None
    if rung.status in ("not-attempted", "absent"):
        return _blocker("NOT_ATTEMPTED", rung.reason, f"status {rung.status}")
    if rung.rung == "C2":
        reason = (unsupported or {}).get("reason")
        if reason in _TAXONOMY_TO_BLOCKER:
            return _blocker(_TAXONOMY_TO_BLOCKER[reason], (unsupported or {}).get("detail") or rung.reason, f"taxonomy outcome {reason}")
        parsed = TX.parse(rung.reason) if rung.reason else None
        if parsed and parsed[0] in _TAXONOMY_TO_BLOCKER:
            return _blocker(_TAXONOMY_TO_BLOCKER[parsed[0]], parsed[1] or rung.reason, "rung reason (taxonomy prefix)")
        if rung.status == "not-claimed":
            return _blocker("REPRESENTATION_MISSING", rung.reason, "C2 not claimed: no typed program for the configuration")
        if rung.reason and "definition-version drift" in rung.reason:
            # IT-F3: the cited evidence was computed under an earlier definition version than the port binds; the
            # comparison under the current binding has not been made, so nothing is known to be missing or wrong yet.
            return _blocker("NOT_ATTEMPTED", rung.reason, "C2 evidence predates the port's definition versions (typed.definition_version_drift)")
        return _blocker("SEMANTICS_MISSING", rung.reason, "C2 not met: compared words disagree with the registered model, not recorded on a binding")
    return _blocker(_DEFAULT_BLOCKER[rung.rung], rung.reason, f"{rung.rung} {rung.status} (default class of the rung)")


def apply_blockers(rungs: dict[str, RungResult], unsupported: dict[str, Any] | None,
                   declared: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None) -> list[dict[str, Any]]:
    """Set every rung's `blocker` (a port declaration wins over the derivation); returns the history: declared
    blockers whose rung is now met (an implementation bug that was fixed stays legible)."""
    by_rung: dict[str, dict[str, Any]] = {}
    for d in declared or ():
        by_rung[normalize_rung(d["rung"])] = d
    history: list[dict[str, Any]] = []
    for r, rung in rungs.items():
        d = by_rung.get(r)
        if rung.met:
            rung.blocker = None
            if d:
                history.append({"rung": r, "class": d["class"], "detail": d.get("detail"), "resolved": d.get("resolved"), "source": "expectations.json blockers (rung now met)"})
        elif d:
            rung.blocker = _blocker(d["class"], d.get("detail"), "expectations.json blockers")
        else:
            rung.blocker = derive_blocker(rung, unsupported)
    return history


def ensure_blockers(doc: dict[str, Any], declared: Any = None) -> dict[str, Any]:
    """A `ladder.json` document written before the blocker classes existed gets them backfilled in memory (the matrix
    reads committed records of CH2's first runs); a document that has them is returned unchanged."""
    if "blocker" in doc and all("blocker" in r for r in (doc.get("rungs") or {}).values()):
        return doc
    rungs = {r: RungResult(r, bool(rec.get("met")), str(rec.get("status")), rec.get("reason"), list(rec.get("evidence") or []))
             for r, rec in (doc.get("rungs") or {}).items()}
    if set(rungs) != set(RUNGS):
        return doc
    history = apply_blockers(rungs, doc.get("unsupported"), declared)
    doc["rungs"] = {r: rungs[r].as_json() for r in RUNGS}
    doc["blocker"] = headline_blocker(rungs, doc.get("reached"))
    doc["blocker_history"] = history
    doc["blockers_backfilled"] = True
    return doc


def headline_blocker(rungs: dict[str, RungResult], reached: str | None) -> dict[str, Any] | None:
    """The blocker of the first rung above `reached` that is not met — the rung the profile would advance to next."""
    start = RUNGS.index(reached) + 1 if reached else 0
    for r in RUNGS[start:]:
        if not rungs[r].met:
            b = rungs[r].blocker or derive_blocker(rungs[r], None)
            return {"rung": r, **b} if b else None
    return None


# ---------------------------------------------------------------------------------------------------------
# ladder.json
# ---------------------------------------------------------------------------------------------------------


def _headline(rungs: dict[str, RungResult]) -> tuple[str | None, list[str], list[str], str]:
    met = [r for r in RUNGS if rungs[r].met]
    reached = met[-1] if met else None
    gaps = [r for r in RUNGS[: RUNGS.index(reached)] if r not in met] if reached else []
    summary = (reached or "none") + (f" (gaps: {', '.join(gaps)})" if gaps else "")
    return reached, met, gaps, summary


def ladder(port_name: str, level_doc: dict[str, Any], c0: RungResult, c5: RungResult, outcome: str, unsupported: dict[str, Any] | None, *,
           utc: str, run_dir: str | None, axis: str | None, declared_blockers: Any = None) -> dict[str, Any]:
    """`ladder.json` from the level verdict (C1–C4) plus the C0/C5 rungs, the taxonomy outcome and the blocker classes."""
    rungs: dict[str, RungResult] = {"C0": c0, "C5": c5}
    for n in (1, 2, 3, 4):
        rungs[RUNG_OF_LEVEL[n]] = from_level(level_doc["levels"][str(n)])
    history = apply_blockers(rungs, unsupported, declared_blockers)
    reached, met, gaps, summary = _headline(rungs)
    claims = [f"{r} {RUNG_NAMES[r]}: {'met' if rungs[r].met else rungs[r].status}" + (f" — {rungs[r].reason}" if rungs[r].reason else "")
              + (f" [{rungs[r].blocker['class']}]" if rungs[r].blocker else "") for r in RUNGS]
    return {"schema": SCHEMA_LADDER, "port": port_name, "utc": utc, "run_dir": run_dir, "axis": axis,
            "reached": reached, "met": met, "gaps": gaps, "summary": summary,
            "blocker": headline_blocker(rungs, reached), "blocker_history": history,
            "rungs": {r: rungs[r].as_json() for r in RUNGS},
            "outcome": outcome, "unsupported": unsupported,
            "rung_of_level": {f"L{n}": r for n, r in RUNG_OF_LEVEL.items()},
            "level_json": {"level": level_doc.get("level"), "summary": level_doc.get("summary")}, "claims": claims}


def ladder_from_level(level_doc: dict[str, Any], *, axis: str | None = None, declared_blockers: Any = None) -> dict[str, Any]:
    """A ladder for a run that wrote `level.json` only (HP/TP's runs on the pre-ladder harness): C1–C4 from the
    levels, C0/C5 `not-attempted` (`PREDATES_LADDER`), no taxonomy outcome beyond what C2 says."""
    rungs: dict[str, RungResult] = {"C0": RungResult("C0", False, "not-attempted", PREDATES_LADDER, []),
                                    "C5": RungResult("C5", False, "not-attempted", INT_POINTER, [])}
    for n in (1, 2, 3, 4):
        rec = (level_doc.get("levels") or {}).get(str(n)) or {"level": n, "met": False, "status": "absent", "reason": "level.json has no record", "evidence": []}
        rungs[RUNG_OF_LEVEL[n]] = from_level({**rec, "level": n})
    reached, met, gaps, summary = _headline(rungs)
    c2 = rungs["C2"]
    outcome = TX.SUPPORTED_EXACTLY if c2.met else "unclassified"
    unsupported = None
    parsed = TX.parse(c2.reason) if c2.status == "not-claimed" else None
    if parsed:
        outcome = parsed[0]
        unsupported = TX.unsupported_record("C2", parsed[0], parsed[1] or c2.reason, "level.json level 2 reason (taxonomy prefix)")
    history = apply_blockers(rungs, unsupported, declared_blockers)
    return {"schema": SCHEMA_LADDER, "port": level_doc.get("port"), "utc": level_doc.get("utc"), "run_dir": level_doc.get("run_dir"), "axis": axis,
            "reached": reached, "met": met, "gaps": gaps, "summary": summary,
            "blocker": headline_blocker(rungs, reached), "blocker_history": history,
            "rungs": {r: rungs[r].as_json() for r in RUNGS},
            "outcome": outcome, "unsupported": unsupported, "rung_of_level": {f"L{n}": r for n, r in RUNG_OF_LEVEL.items()},
            "level_json": {"level": level_doc.get("level"), "summary": level_doc.get("summary")}, "derived_from": "level.json",
            "claims": [f"{r} {RUNG_NAMES[r]}: {'met' if rungs[r].met else rungs[r].status}" + (f" — {rungs[r].reason}" if rungs[r].reason else "")
                       + (f" [{rungs[r].blocker['class']}]" if rungs[r].blocker else "") for r in RUNGS]}


def assert_claim(ladder_doc: dict[str, Any], claimed: Any) -> None:
    """Refuse a claim the evidence does not support (`claimed` in either spelling must be a met rung)."""
    rung = normalize_rung(claimed)
    if rung not in ladder_doc.get("met", []):
        r = (ladder_doc.get("rungs") or {}).get(rung) or {}
        raise OverClaim(f"port {ladder_doc.get('port')}: {rung} claimed but the evidence supports {ladder_doc.get('summary')}"
                        f" ({rung}: {r.get('status')}: {r.get('reason')})")


def level_name(rung: str) -> str | None:
    return LEVEL_NAMES.get(LEVEL_OF_RUNG.get(normalize_rung(rung), 0))
