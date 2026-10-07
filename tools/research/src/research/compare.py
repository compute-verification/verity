"""`research compare`: two ways to set runs side by side.

`verdict(originals, replay, expect)` judges a replay against the attempts it replays, read from the evidence store by run id
(`load`: the attempt manifest, its result's meta and its outputs' manifests, local first, else the remote the store fetches from;
no payload bytes are downloaded, since manifests carry every file's sha256).  One of:

    same             every check below is equal
    within-band      nothing differs, but something only matched within the originals' own variation
    differs          something changed (each reason says what), or a metric moved and the band is unknown (one original)
    expected-change  `expect` (a `replay_expect` label value: `refused: FORM` or `blocked: GATE`) names the refusal the replay shows

The checks, each against every original:

    derivation   the replay is the same computation (the store's drv: id, which holds no source revision)
    status       state, rc and validation as the attempt records them
    outputs      every named output (all but result, run_files, run_record) equal by art id, or by content net of the run's own id:
                 the publisher stamps `meta.run_id` on typed outputs, so the same bytes from two runs never share an art id
    run files    per file by sha256: the run record's files (with the ones custody trimmed, listed by sha256) and the result's run
                 files, minus the runner's own logs and records (RUNNER_FILES).  When an attempt has no run record, only the result's
                 listed files are compared.
    metrics      the result's measurements.  Numbers: with two or more originals, inside [min - spread, max + spread], spread =
                 max - min; with one, equal.  Anything else: equal.

An item on which the originals disagree among themselves (a file, an output, a status field) cannot be checked bit for bit: it is
within-band.  `expect` given: a replay refused (not done, rc != 0 or validation failed) whose failure, class, result summary or
validation detail contains FORM (GATE) is expected-change; one refused otherwise, or not refused at all, differs.

`compare(base_dir, cand_dir)` is the older descriptive diff of two run directories.  It shows source, tool, machine, fingerprint
(expected and observed, key by key), environment, execution, instrumentation, outcome, and measurement differences; it does not
decide which differences invalidate the comparison.  Ratios are labeled candidate / baseline.  One pair of runs is one measured
ratio, not a confidence interval.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import record
from . import runs as R

SAME, WITHIN_BAND, DIFFERS, EXPECTED_CHANGE = "same", "within-band", "differs", "expected-change"
VERDICTS = (SAME, WITHIN_BAND, DIFFERS, EXPECTED_CHANGE)
EXPECT_KEY = "replay_expect"
EXPECT_RE = re.compile(r"(refused|blocked): (\S.*)")
#: outputs that are not compared by art id: the result (its meta holds the run id and the measurements, checked as metrics), and
#: the two trees whose files are compared one by one
NOT_NAMED = frozenset({"result", "run_files", "run_record"})
#: how many examples a reason line names
EXAMPLES = 5


@dataclass(frozen=True)
class Attempt:
    """What `verdict` reads of one attempt.  `files` maps a run-dir path to its sha256; `files_scope` says what they cover."""
    id: str
    doc: dict[str, Any]
    result: dict[str, Any] | None
    outputs: dict[str, str]
    contents: dict[str, str]
    files: dict[str, str]
    listed: frozenset[str]
    files_scope: str


@dataclass
class Verdict:
    verdict: str
    replay: str
    originals: list[str]
    expect: str | None
    reasons: list[str] = field(default_factory=list)
    band: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_doc(self) -> dict[str, Any]:
        return {"verdict": self.verdict, "replay": self.replay, "originals": self.originals, "expect": self.expect,
                "reasons": self.reasons, "band": self.band}


def runner_files() -> frozenset[str]:
    """The runner's own logs and records in a run dir (custody's always-kept files, the sampler's stream, the publisher's error)."""
    from .store import custody

    return custody.RUN_FILES | {"resources.jsonl", "store_publish_error.json", custody.TRIMMED_LISTING}


def load(store: Any, run_id: str) -> Attempt:
    """The attempt `run_id` from `store` (a `LocalStore`); LookupError when neither the store nor its remote has it."""
    from .store import canon, custody

    doc = store.get_attempt(run_id)
    if doc is None:
        raise LookupError(f"{run_id}: no attempt in the store at {store.root}" + (f" or on {store.remote.describe()}" if store.remote else ""))
    outs = {k: v for k, v in (doc.get("outputs") or {}).items() if v}
    mans = {name: store.get_manifest(art) for name, art in outs.items()}
    own = {art: f"@{name}" for name, art in outs.items()}
    contents = {}
    for name, m in mans.items():
        if name in NOT_NAMED:
            continue
        d = m.to_doc()
        meta = {k: v for k, v in (d.get("meta") or {}).items() if not (k == "run_id" and v == run_id)}
        refs = {k: ([own.get(x, x) for x in v] if isinstance(v, list) else own.get(v, v)) for k, v in (d.get("refs") or {}).items()}
        contents[name] = canon.sha256({**d, "meta": meta, "refs": refs})
    skip = runner_files()
    files: dict[str, str] = {}
    listed: set[str] = set()
    if "run_files" in mans and mans["run_files"].payload is not None:
        for f in mans["run_files"].payload.files or []:
            files[f.path] = f.sha256
            listed.add(f.path)
    scope = "run_files" if listed else "none"
    if "run_record" in mans and mans["run_record"].payload is not None:
        scope = "run_record"
        for f in mans["run_record"].payload.files or []:
            if f.path not in skip and not custody.is_run_file(f.path):
                files[f.path] = f.sha256
        for t in custody.trimmed_of(store, outs["run_record"], mans["run_record"]):
            if t["path"] not in skip and not custody.is_run_file(t["path"]):
                files[t["path"]] = t["sha256"]
    return Attempt(id=run_id, doc=doc, result=mans["result"].meta if "result" in mans else None, outputs=outs, contents=contents,
                   files=files, listed=frozenset(listed), files_scope=scope)


def _examples(items: list[str]) -> str:
    return ", ".join(items[:EXAMPLES]) + (f", ... ({len(items)} in all)" if len(items) > EXAMPLES else "")


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _refusal(a: Attempt) -> tuple[bool, str]:
    st = a.doc.get("status") or {}
    val = (a.result or {}).get("validation") or {}
    refused = st.get("state") != "done" or st.get("rc") not in (0, None) or st.get("validation") == "failed"
    text = " | ".join(str(x) for x in (st.get("failure"), st.get("class"), st.get("result"), val.get("detail") if isinstance(val, dict) else None) if x)
    return refused, text


class _Tally:
    def __init__(self) -> None:
        self.differs: list[str] = []
        self.within: list[str] = []
        self.notes: list[str] = []
        self.same: dict[str, int] = {}

    def categorical(self, what: str, name: str, orig: list[Any], rep: Any, show: Any = None) -> None:
        """One item: equal across the originals and in the replay (same), the originals disagree (within), else differs."""
        show = show or (lambda v: repr(v))
        label = f"{what} {name}" if name else what
        distinct = {repr(v) for v in orig}
        if len(distinct) > 1:
            self.within.append(f"{label}: varies across the originals ({len(distinct)} distinct), not checked bit for bit")
        elif repr(rep) == repr(orig[0]):
            self.same[what] = self.same.get(what, 0) + 1
        else:
            self.differs.append(f"{label}: {show(orig[0])} -> {show(rep)}")


def _short_sha(v: Any) -> str:
    return "absent" if v is None else str(v)[:12]


def verdict(originals: list[Attempt], replay: Attempt, expect: str | None = None) -> Verdict:
    """Judge `replay` against `originals` (module docstring).  ValueError when there is no original or `expect` is malformed."""
    if not originals:
        raise ValueError("verdict needs at least one original")
    m = None
    if expect is not None:
        m = EXPECT_RE.fullmatch(expect)
        if not m:
            raise ValueError(f"{EXPECT_KEY} {expect!r} is neither `refused: FORM` nor `blocked: GATE`")
    v = Verdict(verdict=SAME, replay=replay.id, originals=[o.id for o in originals], expect=expect)
    t = _Tally()

    t.categorical("derivation", "", [o.doc.get("derivation") for o in originals], replay.doc.get("derivation"))
    for k in ("state", "rc", "validation"):
        t.categorical("status", k, [(o.doc.get("status") or {}).get(k) for o in originals], (replay.doc.get("status") or {}).get(k))

    names = sorted(set(replay.contents).union(*(o.contents for o in originals)))
    for n in names:
        orig = [o.contents.get(n) for o in originals]
        if all(o.outputs.get(n) == replay.outputs.get(n) for o in originals) and replay.outputs.get(n):
            t.same["output"] = t.same.get("output", 0) + 1
            continue
        t.categorical("output", n, orig, replay.contents.get(n), show=lambda c: "absent" if c is None else "content " + c[:12])

    every = [*originals, replay]
    if all(a.files_scope == "run_record" for a in every):
        paths = sorted(set().union(*(a.files for a in every)))
    else:
        bare = [a.id for a in every if a.files_scope != "run_record"]
        t.notes.append(f"run files: only the result's listed files are compared ({_examples(bare)} has no run record)")
        paths = sorted(set().union(*(a.listed for a in every)))
    file_diffs: list[str] = []
    file_varies: list[str] = []
    for p in paths:
        orig = [o.files.get(p) for o in originals]
        if len(set(orig)) > 1:
            file_varies.append(p)
        elif replay.files.get(p) == orig[0]:
            t.same["run file"] = t.same.get("run file", 0) + 1
        else:
            file_diffs.append(f"{p} ({_short_sha(orig[0])} -> {_short_sha(replay.files.get(p))})")
    if file_diffs:
        t.differs.append(f"run files: {len(file_diffs)} differ: {_examples(file_diffs)}")
    if file_varies:
        t.within.append(f"run files: {len(file_varies)} vary across the originals, not checked bit for bit: {_examples(file_varies)}")

    def measurements(a: Attempt) -> dict[str, dict[str, Any]]:
        return {x["name"]: x for x in ((a.result or {}).get("measurements") or []) if isinstance(x, dict) and isinstance(x.get("name"), str)}

    om = [measurements(o) for o in originals]
    rm = measurements(replay)
    for name in sorted(set(rm).union(*om)):
        orig = [d.get(name) for d in om]
        rep = rm.get(name)
        if any(x is None for x in orig) or rep is None:
            t.categorical("metric", name, [x is not None for x in orig], rep is not None,
                          show=lambda present: "present" if present else "absent")
            continue
        units = {x.get("unit") for x in orig} | {rep.get("unit")}
        if len(units) > 1:
            t.differs.append(f"metric {name}: units {sorted(map(str, units))}")
            continue
        vals = [x.get("value") for x in orig]
        rv = rep.get("value")
        if not (all(_is_number(x) for x in vals) and _is_number(rv)):
            t.categorical("metric", name, vals, rv)
            continue
        lo, hi = min(vals), max(vals)
        unit = rep.get("unit")
        if len(vals) < 2:
            if rv == lo:
                t.same["metric"] = t.same.get("metric", 0) + 1
            else:
                t.differs.append(f"metric {name}: {lo} -> {rv} {unit}; band unknown: one original (a band needs at least two)")
            v.band[name] = {"n": 1, "lo": None, "hi": None, "value": rv, "unit": unit}
            continue
        spread = hi - lo
        band_lo, band_hi = lo - spread, hi + spread
        v.band[name] = {"n": len(vals), "lo": band_lo, "hi": band_hi, "value": rv, "unit": unit}
        if lo == hi == rv:
            t.same["metric"] = t.same.get("metric", 0) + 1
        elif band_lo <= rv <= band_hi:
            t.within.append(f"metric {name}: {rv} {unit} inside the band [{band_lo:.6g}, {band_hi:.6g}] of {len(vals)} originals")
        else:
            t.differs.append(f"metric {name}: {rv} {unit} outside the band [{band_lo:.6g}, {band_hi:.6g}] of {len(vals)} originals")

    if m is not None:
        refused, text = _refusal(replay)
        token = m.group(2)
        if refused and token in text:
            v.verdict = EXPECTED_CHANGE
            v.reasons = [f"expected {expect}: the replay was refused ({text})"]
            return v
        t.differs.insert(0, f"expected {expect}: " + (f"the replay was refused, but not naming {token!r} ({text})" if refused
                                                       else "the replay was not refused"))
    v.verdict = DIFFERS if t.differs else WITHIN_BAND if t.within else SAME
    same = ", ".join(f"{n} {'status field' if what == 'status' else what}{'s' if n != 1 else ''}" for what, n in t.same.items())
    v.reasons = t.differs + t.within + t.notes + ([f"equal: {same}"] if same else [])
    return v


def render_verdict(v: Verdict) -> str:
    lines = [f"{v.verdict}  replay {v.replay} against {', '.join(v.originals)}" + (f"  (expect {v.expect})" if v.expect else "")]
    lines += [f"  - {r}" for r in v.reasons]
    return "\n".join(lines)


def _get(d: Any, *path: str) -> Any:
    for k in path:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _env_view(job: dict[str, Any] | None) -> dict[str, Any]:
    e = (job or {}).get("environment") or {}
    gpu = e.get("gpu") or {}
    return {"hostname": e.get("hostname"), "host_class": e.get("host_class"), "kernel": e.get("kernel"), "python": e.get("python"),
            "cpu_count": e.get("cpu_count"), "host_memory_total_bytes": e.get("host_memory_total_bytes"),
            "cgroup_effective_memory_max_bytes": _get(e, "cgroup", "effective_memory_max_bytes"),
            "gpu_devices": [d.get("name") for d in gpu.get("devices") or []] if gpu.get("available") else "none",
            "gpu_driver": next((d.get("driver") for d in gpu.get("devices") or []), None),
            "cuda_visible_devices": gpu.get("cuda_visible_devices")}


def _dict_diff(a: dict[str, Any] | None, b: dict[str, Any] | None) -> list[tuple[str, Any, Any]]:
    a = a or {}
    b = b or {}
    out = []
    for k in sorted(set(a) | set(b)):
        if a.get(k) != b.get(k):
            out.append((k, a.get(k, "<absent>"), b.get(k, "<absent>")))
    return out


def compare(base_dir: Path, cand_dir: Path) -> dict[str, Any]:
    sb, sc = R.summarize(base_dir), R.summarize(cand_dir)
    jb, jc = record.read_job(base_dir), record.read_job(cand_dir)
    sections: dict[str, list[tuple[str, Any, Any]]] = {}
    sections["identity"] = _dict_diff(
        {"tree_sha": sb["tree_sha"], "workload_digest": sb["workload_digest"], "tool_sha": _get(sb["tool"], "research_git_sha"),
         "tool_snapshot": _get(sb["tool"], "research_snapshot"), "classifier": _get(sb["tool"], "classifier_version"),
         "machine": _get(sb["remote"], "machine"), "pod_id": _get(sb["remote"], "pod_id"), "project": sb["project"], "stage": sb["stage"]},
        {"tree_sha": sc["tree_sha"], "workload_digest": sc["workload_digest"], "tool_sha": _get(sc["tool"], "research_git_sha"),
         "tool_snapshot": _get(sc["tool"], "research_snapshot"), "classifier": _get(sc["tool"], "classifier_version"),
         "machine": _get(sc["remote"], "machine"), "pod_id": _get(sc["remote"], "pod_id"), "project": sc["project"], "stage": sc["stage"]})
    sections["fingerprint_expected"] = _dict_diff(sb["fingerprint_expected"], sc["fingerprint_expected"])
    sections["fingerprint_observed"] = _dict_diff(sb["fingerprint_observed"], sc["fingerprint_observed"])
    sections["environment"] = _dict_diff(_env_view(jb), _env_view(jc))
    sections["execution"] = _dict_diff(
        {"command": sb["command"], "timeout_s": _get(jb, "execution", "timeout_s"), "inputs": [(i.get("name"), i.get("sha256")) for i in _get(jb, "computation", "inputs") or []],
         "profile": _get(jb, "computation", "profile"), "policy_sha256": _get(jb, "execution", "policy_sha256")},
        {"command": sc["command"], "timeout_s": _get(jc, "execution", "timeout_s"), "inputs": [(i.get("name"), i.get("sha256")) for i in _get(jc, "computation", "inputs") or []],
         "profile": _get(jc, "computation", "profile"), "policy_sha256": _get(jc, "execution", "policy_sha256")})
    sections["instrumentation"] = _dict_diff(
        {"instruments": _get(jb, "telemetry", "instruments"), "cadence_s": _get(jb, "telemetry", "cadence_s"), "job_schema": sb["schema"]},
        {"instruments": _get(jc, "telemetry", "instruments"), "cadence_s": _get(jc, "telemetry", "cadence_s"), "job_schema": sc["schema"]})
    outcomes = {
        "baseline": {"state": sb["state"], "rc": sb["rc"], "class": sb["class"], "result": sb["result"], "validation": sb["validation"], "preserved": bool(sb["preserved"]), "wall_s": sb["wall_s"]},
        "candidate": {"state": sc["state"], "rc": sc["rc"], "class": sc["class"], "result": sc["result"], "validation": sc["validation"], "preserved": bool(sc["preserved"]), "wall_s": sc["wall_s"]},
    }
    mb = {m["name"]: m for m in sb["measurements"] or []}
    mc = {m["name"]: m for m in sc["measurements"] or []}
    measurements = []
    for name in sorted(set(mb) | set(mc)):
        b, c = mb.get(name), mc.get(name)
        row: dict[str, Any] = {"name": name, "baseline": b and b["value"], "candidate": c and c["value"],
                               "unit": (b or c)["unit"], "unit_mismatch": bool(b and c and b["unit"] != c["unit"])}
        if b and c and not row["unit_mismatch"] and isinstance(b["value"], (int, float)) and isinstance(c["value"], (int, float)):
            row["delta_candidate_minus_baseline"] = c["value"] - b["value"]
            row["ratio_candidate_over_baseline"] = (c["value"] / b["value"]) if b["value"] else None
        measurements.append(row)
    caveats = []
    if not (sb["result_valid"] and sc["result_valid"]):
        caveats.append("at least one run has no valid result.json; measurements above come only from valid results")
    if sb["validation"] != "passed" or sc["validation"] != "passed":
        caveats.append("at least one run did not report validation passed; a speedup on an unvalidated run is not a result")
    if sections["fingerprint_observed"]:
        caveats.append("observed fingerprints differ; whether that is the experiment or an invalid comparison is the workload contract's call")
    if sections["environment"]:
        caveats.append("environments differ; the experiment decides whether the machines are comparable")
    if any(m.get("ratio_candidate_over_baseline") is not None for m in measurements):
        caveats.append("one pair of runs gives a measured ratio, not a confidence interval")
    return {"baseline": {"id": sb["run_id"], "dir": sb["dir"]}, "candidate": {"id": sc["run_id"], "dir": sc["dir"]},
            "differences": {k: [{"field": f, "baseline": a, "candidate": b} for f, a, b in v] for k, v in sections.items()},
            "outcomes": outcomes, "measurements": measurements, "caveats": caveats}


def render(rep: dict[str, Any]) -> str:
    lines = [f"baseline   {rep['baseline']['id']}", f"candidate  {rep['candidate']['id']}", ""]
    any_diff = False
    for sec, rows in rep["differences"].items():
        if not rows:
            continue
        any_diff = True
        lines.append(f"[{sec}]")
        for r in rows:
            lines.append(f"  {r['field']:<34} {_short(r['baseline'])}  ->  {_short(r['candidate'])}")
    if not any_diff:
        lines.append("[identity/fingerprint/environment/execution/instrumentation] no recorded differences")
    lines.append("")
    lines.append("[outcomes]")
    for who in ("baseline", "candidate"):
        o = rep["outcomes"][who]
        lines.append(f"  {who:<10} {o['state']} rc={o['rc']} class={o['class']} result={o['result']} validation={o['validation']} preserved={o['preserved']} wall_s={o['wall_s']}")
    if rep["measurements"]:
        lines.append("")
        lines.append("[measurements]                          baseline        candidate       candidate/baseline")
        for m in rep["measurements"]:
            ratio = m.get("ratio_candidate_over_baseline")
            lines.append(f"  {m['name']:<36} {_num(m['baseline']):>14}  {_num(m['candidate']):>14}  {('x%.4f' % ratio) if ratio is not None else '-':>10}  {m['unit']}"
                         + ("  UNIT MISMATCH" if m["unit_mismatch"] else ""))
    if rep["caveats"]:
        lines.append("")
        lines.append("[caveats]")
        lines += [f"  - {c}" for c in rep["caveats"]]
    return "\n".join(lines)


def _short(v: Any) -> str:
    s = str(v)
    return s if len(s) <= 60 else s[:57] + "..."


def _num(v: Any) -> str:
    if v is None:
        return "-"
    return f"{v:.6g}" if isinstance(v, float) else str(v)
