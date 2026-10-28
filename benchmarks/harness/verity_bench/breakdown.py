"""Where PoUS's bandwidth audit's time over plain goes, by part, from the spans its breakdown spec's arms record
(`llama31-8b-fp8-1gpu-bandwidth-ablation`).

Memory accounting's spans arms (`benchmarks/vllm_encoded/README.md`, "Spans") time each step by part, and each timed
rep's `extras` keeps a run's parts a step (`span_per_step.<kind>.<part>_ms`, `graph` for vLLM's replayed decode steps and
`eager` for prefill and mixed steps) and its steps. A part's time an output token is its time a step times the kind's
steps over the run's generated tokens. The top-up runs beside the step, so its cost is in no part: it shows only as the
other parts' stretch, which the spans arms with the top-up off (`verity_bench.vllm.bandwidth_ablation`) separate.

The audit's time over plain (the spans arms' time per output token, paired by rep) is split into rows that sum to it:

~~~text
top-up contention     (audit-spans − audit-spans-no-topup) − (plain-spans − plain-spans-no-topup), time per output token
decode <part>         a decode step's part, audit-spans-no-topup − plain-spans-no-topup, an output token
decode, the rest      every other decode part (model, pass, between, append_wait, kv_receipts, ...), the same
prefill steps         every part of the eager steps, the same
outside every span    (audit-spans-no-topup − plain-spans-no-topup) less every part: time before a run's first mark
~~~

The credited lines' own difference (`bandwidth-audit` − `bandwidth-plain`, spans off) is beside it: the spans' marks are
in the step's time, so it is the audit's cost, and the rows say where the spans arms' time went. No row is a ratio, and
nothing here is a gate: the harness's verdicts never read spans.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from verity_bench import jobs, served_table
from verity_bench.vllm import result as result_format
from verity_bench.vllm.presets import expand_sweep, resolve_spec
from verity_bench.vllm.runner import _summary

SCHEMA = "verity-bench/bandwidth-breakdown/v1"
SPEC = "llama31-8b-fp8-1gpu-bandwidth-ablation"
PLAIN, AUDIT = "bandwidth-plain", "bandwidth-audit"
PLAIN_SPANS, AUDIT_SPANS = "bandwidth-plain-spans", "bandwidth-audit-spans"
PLAIN_OFF, AUDIT_OFF = "bandwidth-plain-spans-no-topup", "bandwidth-audit-spans-no-topup"
ARMS = (PLAIN, AUDIT, PLAIN_SPANS, AUDIT_SPANS, PLAIN_OFF, AUDIT_OFF)
SPANS_ARMS = (PLAIN_SPANS, AUDIT_SPANS, PLAIN_OFF, AUDIT_OFF)
KINDS = ("graph", "eager")
PREFIX = "span_per_step."
#: a decode step's parts given rows of their own, with what each holds on the audit (the README's span list)
DECODE_ROWS = {
    "linear": "the weights' codec decode, FP8 MMA and sketch receipts (one fused launch)",
    "lm_head": "the lm_head's decode, BF16 MMA and receipt",
    "commit": "the last pass's receipts closed: KV slots' host read, compaction, step tree",
    "attention": "paged attention with the KV words decoded inside it",
}
NOT_PARTS = ("step", "append")


def _reps(row: Mapping[str, Any], arm: str) -> list[Mapping[str, Any]]:
    return sorted(row["arms"][arm]["reps"], key=lambda rep: rep["rep"])


def _tpot_ms(rep: Mapping[str, Any]) -> float:
    return float(rep["metrics"]["time_per_output_token_s"]) * 1e3


def parts(rep: Mapping[str, Any], kind: str) -> dict[str, float]:
    """A rep's main-stream parts of `kind`, ms a step (empty when the run had no step of that kind)."""

    extras, head = rep.get("extras") or {}, f"{PREFIX}{kind}."
    return {key[len(head):-len("_ms")]: float(value) for key, value in extras.items()
            if key.startswith(head) and key.endswith("_ms") and key[len(head):-len("_ms")] not in NOT_PARTS}


def per_token(rep: Mapping[str, Any], kind: str) -> dict[str, float]:
    """A rep's parts of `kind`, ms an output token: ms a step × the kind's steps / the run's generated tokens."""

    steps = float((rep.get("extras") or {}).get(f"{PREFIX}{kind}.steps", 0.0))
    return {part: ms * steps / int(rep["generated_tokens"]) for part, ms in parts(rep, kind).items()}


def _spans_recorded(row: Mapping[str, Any], arm: str) -> None:
    for rep in _reps(row, arm):
        if not any(parts(rep, kind) for kind in KINDS):
            raise ValueError(f"{arm} at {row['point']['id']} rep {rep['rep']} recorded no spans "
                             f"(no {PREFIX}* extras): it isn't a spans arm")


def attribution(row: Mapping[str, Any]) -> dict[str, Any]:
    """One point's audit-over-plain rows (ms an output token, each a summary over the paired reps) and their sum."""

    for arm in ARMS:
        if arm not in row["arms"]:
            raise ValueError(f"the point {row['point']['id']} has no timed {arm}")
    for arm in SPANS_ARMS:
        _spans_recorded(row, arm)
    reps = {arm: _reps(row, arm) for arm in ARMS}
    n = {len(value) for value in reps.values()}
    if len(n) != 1:
        raise ValueError(f"the arms at {row['point']['id']} have different rep counts: paired rows need the same reps")
    names = [f"decode {part}" for part in DECODE_ROWS] + ["decode, the rest", "prefill steps", "outside every span"]
    values: dict[str, list[float]] = {name: [] for name in ["credited", "total", "top-up contention", *names]}
    for i in range(n.pop()):
        tpot = {arm: _tpot_ms(reps[arm][i]) for arm in ARMS}
        values["credited"].append(tpot[AUDIT] - tpot[PLAIN])
        values["total"].append(tpot[AUDIT_SPANS] - tpot[PLAIN_SPANS])
        values["top-up contention"].append((tpot[AUDIT_SPANS] - tpot[AUDIT_OFF]) - (tpot[PLAIN_SPANS] - tpot[PLAIN_OFF]))
        audit = {kind: per_token(reps[AUDIT_OFF][i], kind) for kind in KINDS}
        plain = {kind: per_token(reps[PLAIN_OFF][i], kind) for kind in KINDS}

        def diff(kind: str, part: str) -> float:
            return audit[kind].get(part, 0.0) - plain[kind].get(part, 0.0)

        decode = set(audit["graph"]) | set(plain["graph"])
        for part in DECODE_ROWS:
            values[f"decode {part}"].append(diff("graph", part))
        values["decode, the rest"].append(sum(diff("graph", part) for part in decode - set(DECODE_ROWS)))
        values["prefill steps"].append(sum(diff("eager", part) for part in set(audit["eager"]) | set(plain["eager"])))
        spanned = sum(values[name][-1] for name in names[:-1])
        values["outside every span"].append(tpot[AUDIT_OFF] - tpot[PLAIN_OFF] - spanned)
    total = _summary(values["total"])
    rows = []
    for name in ["top-up contention", *names]:
        summary = _summary(values[name])
        rows.append({"row": name, "ms_per_token": summary,
                     "share": summary["mean"] / total["mean"] if total["mean"] else None})
    return {"credited": _summary(values["credited"]), "total": total, "rows": rows,
            "sum": sum(row["ms_per_token"]["mean"] for row in rows)}


def arm_spans(row: Mapping[str, Any], arm: str) -> dict[str, Any]:
    """A spans arm's step by part at one point: each kind's steps a run, tokens a step and parts a step (ms), means over
    its reps, its time per output token, the deferred KV append beside its steps and the top-up's busy share."""

    reps = _reps(row, arm)

    def mean(values: Sequence[float]) -> float:
        return sum(values) / len(values)

    def extra(key: str) -> float | None:
        got = [(rep.get("extras") or {}).get(key) for rep in reps]
        return None if any(value is None for value in got) else mean([float(value) for value in got])

    out: dict[str, Any] = {"tpot_ms": mean([_tpot_ms(rep) for rep in reps]), "reps": len(reps),
                           "topup_busy_share": extra("span.topup_busy_share"), "filler_share": extra("filler_share"),
                           "kinds": {}}
    for kind in KINDS:
        if not all(parts(rep, kind) for rep in reps):
            continue
        names = sorted(set().union(*(parts(rep, kind) for rep in reps)))
        out["kinds"][kind] = {
            "steps": extra(f"{PREFIX}{kind}.steps"),
            "tokens_per_step": extra(f"{PREFIX}{kind}.tokens_per_step"),
            "step_ms": extra(f"{PREFIX}{kind}.step_ms"),
            "append_ms": extra(f"{PREFIX}{kind}.append_ms"),
            "parts_ms": {name: mean([parts(rep, kind).get(name, 0.0) for rep in reps]) for name in names},
        }
    return out


def point(row: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    return {"point": row["point"]["id"], "source": dict(source), **attribution(row),
            "arms": {arm: arm_spans(row, arm) for arm in SPANS_ARMS}}


def from_results(paths: Sequence[Path]) -> dict[str, Any]:
    """The breakdown of each point of local result files (`benchmark.json`)."""

    points = []
    for path in paths:
        doc = json.loads(Path(path).read_text())
        result_format.validate(doc)
        for row in doc["points"]:
            points.append(point(row, {"result": str(path), "commit": doc["identity"].get("git_commit"),
                                      "reps": doc["execution"]["reps"]}))
    return {"schema": SCHEMA, "spec": None, "points": points, "skipped": []}


def collect(store: Any, *, spec: str = SPEC) -> dict[str, Any]:
    """The breakdown of the newest passing run of each point of `spec` that times every arm (`served_table`'s rule)."""

    skipped: list[dict[str, Any]] = []
    points = []
    for sweep_point in expand_sweep(resolve_spec(spec).sweep.id):
        found = served_table._newest(store, spec, sweep_point.id, ARMS, skipped)
        if found is None:
            skipped.append({"spec": spec, "point": sweep_point.id, "run": None, "art": None, "why": "no passing run"})
            continue
        points.append(point(found.point, {"run": found.run, "art": found.art, "commit": found.commit,
                                          "reps": found.reps, "caveats": found.caveats}))
    return {"schema": SCHEMA, "spec": spec, "points": points, "skipped": skipped}


def build(*, store: Any = None, root: Path | None = None, local_only: bool = False,
          added: Callable[[str], int] | None = None, log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """`collect` after pulling the remote's labels and its attempts started since the spec was added, unless
    `local_only`."""

    root = root or jobs.repository_root()
    completion = jobs.StoreCompletion(store, root=root, local_only=local_only, log=log)
    if not local_only:
        completion.pull_labels()
        completion.pull_since((added or (lambda spec: served_table.spec_added(root, spec)))(SPEC))
    return collect(completion.store)


def _ms(summary: Mapping[str, Any]) -> str:
    ci = summary.get("ci95")
    return f"{summary['mean']:.3f}" if ci is None else f"{summary['mean']:.3f} ± {(ci[1] - ci[0]) / 2:.3f}"


def _opt(value: float | None, form: str) -> str:
    return "—" if value is None else format(value, form)


def render(doc: Mapping[str, Any]) -> str:
    lines = [
        "# Where the bandwidth audit's time goes",
        "",
        "Llama-3.1-8B FP8 on one RTX PRO 6000, the bandwidth audit's encoded stack in vLLM. Each row is ms an output "
        f"token, the mean over the point's paired reps ± its 95% half-width (Student t). `{AUDIT_SPANS}` over "
        f"`{PLAIN_SPANS}` is split into rows that sum to it: the top-up's contention from the arms with the top-up off, "
        "every other row a part of the spans arms with the top-up off. The credited lines' difference (spans off) is "
        "the audit's cost; the spans' marks add their own time to the spans arms. Nothing here is a ratio or a verdict.",
    ]
    for entry in doc["points"]:
        source = entry["source"]
        where = (f"run {source['run']}, art `{source['art'][:16]}`" if source.get("run")
                 else f"result {source.get('result')}")
        lines += ["", f"## {entry['point']} ({where}, {source.get('reps')} reps)", "",
                  "| row | ms an output token | share |", "|---|---:|---:|",
                  f"| `{AUDIT}` − `{PLAIN}` (credited lines, spans off) | {_ms(entry['credited'])} | |",
                  f"| **`{AUDIT_SPANS}` − `{PLAIN_SPANS}`** | **{_ms(entry['total'])}** | 100% |"]
        for row in entry["rows"]:
            name, part = row["row"], row["row"].removeprefix("decode ")
            label = f"{name}: {DECODE_ROWS[part]}" if part in DECODE_ROWS else name
            share = "—" if row["share"] is None else f"{row['share']:.0%}"
            lines.append(f"| {label} | {_ms(row['ms_per_token'])} | {share} |")
        lines.append(f"| sum of the rows | {entry['sum']:.3f} | |")
        arms = list(entry["arms"])
        for kind, title in (("graph", "decode steps (vLLM's CUDA graphs)"), ("eager", "prefill and mixed steps (eager)")):
            have = [arm for arm in arms if kind in entry["arms"][arm]["kinds"]]
            if not have:
                continue
            kinds = {arm: entry["arms"][arm]["kinds"][kind] for arm in have}
            top = {name: max(k["parts_ms"].get(name, 0.0) for k in kinds.values())
                   for name in set().union(*(k["parts_ms"] for k in kinds.values()))}
            names = sorted((name for name in top if top[name] >= 0.0005), key=lambda name: -top[name])
            lines += ["", f"{title}, ms a step (means over reps):", "",
                      "| part | " + " | ".join(f"`{arm}`" for arm in have) + " |",
                      "|---|" + "---:|" * len(have)]
            for name in names:
                lines.append(f"| {name} | " + " | ".join(f"{kinds[arm]['parts_ms'].get(name, 0.0):.3f}" for arm in have)
                             + " |")
            for key, label, form in (("step_ms", "**step**", ".3f"), ("append_ms", "KV append beside the step", ".3f"),
                                     ("steps", "steps a run", ".1f"), ("tokens_per_step", "rows a step", ".1f")):
                lines.append(f"| {label} | " + " | ".join(_opt(kinds[arm][key], form) for arm in have) + " |")
        lines += ["", "| arm | time per output token, ms | top-up busy / main stream | filler share |", "|---|---:|---:|---:|"]
        for arm in arms:
            spans = entry["arms"][arm]
            lines.append(f"| `{arm}` | {spans['tpot_ms']:.3f} | {_opt(spans['topup_busy_share'], '.2f')} "
                         f"| {_opt(spans['filler_share'], '.3f')} |")
    if doc["skipped"]:
        lines += ["", "Passed over:", ""]
        lines += [f"- `{s['spec']}` {s['point'] or ''}: {s['run'] or '—'}: {s['why']}" for s in doc["skipped"]]
    return "\n".join(lines) + "\n"
