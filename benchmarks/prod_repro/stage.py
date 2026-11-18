"""A prod_repro run's result as the record of one stage of the serving cost (`verity bench serving_cost`): the
stage's `kind`, its fingerprint's fields and its validation, which passes only where the run holds what the next stage
reads, each gap named.

    stage.py capture OUT --model ID --minify RUN [--exact RUN] [--reference prod-trace] [--stock prod] [--ours ARM ...]
    stage.py minify OUT --model ID

``capture``: OUT as `priced.py` reads it, on the full model: each arm of ours that served on the canonical top-k paired
lifetime for lifetime with the reference, each lifetime with a step trace, counted passes and the top-k's counts (by
step, or the lifetime's totals), and the job's measured reference over stock with an interval; its fingerprint names
the member's model and card (``served``, ``card``) as `serve_tax.member` does. ``minify``: a stand-in's
run (`serve_tax.MODELS`' ``stand_in``) holding `topk_units.py`'s `UNITS`, with both kinds of time for every pass its
pin offers, timed to its last shape; it writes the units' table, its ranges and how they were measured to
`SUMMARY`. Either fails where the run's own validation did.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "serve_tax")]
import priced  # noqa: E402 (benchmarks/prod_repro, put on sys.path above)
import serve_tax as S  # noqa: E402
from verity_config.settings import SHARED  # noqa: E402 (the tree's tools/config, which serve_tax put on sys.path)

KINDS = {stage: f"serving-cost-{stage}/v1" for stage in ("minify", "capture")}
RESULT = "research/result/v0.1"
RUN_ID = re.compile(r"r\d{8}-\d{6}-[0-9a-f]{4}")
UNITS, SUMMARY, GPUS = "units.jsonl", "units.json", "gpus.csv"
#: what `priced.counted` reads of a pass's record
PASS = frozenset({"load", "rep", "t0_unix", "wall_s"})
#: what a price can't read from a minify record yet: the top-k at prefill shapes has no unit time
UNPRICED = (("top-k", "prefill steps"),)


def capture_gaps(out: Path, reference: str = "prod-trace", stock: str = "prod",
                 ours: tuple[str, ...] = ("prod-record", "prod-record-bisect")) -> list[str]:
    gaps, refs = [], priced.lifetimes(out, reference)
    served = {arm: ds for arm in ours if (ds := priced.lifetimes(out, arm))}
    if not refs:
        gaps.append(f"no lifetime of the reference {reference}")
    if not served:
        gaps.append(f"no lifetime of {' or '.join(ours)}")
    kept = out / "topk_counters.json"
    counters = (json.loads(kept.read_text()).get("dirs") or {}) if kept.is_file() else {}
    for arm, ds in served.items():
        if refs and len(ds) != len(refs):
            gaps.append(f"{arm} has {len(ds)} lifetimes and {reference} {len(refs)}: not paired")
        for d in ds:
            meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").is_file() else {}
            mode = (meta.get("env") or {}).get("VERITY_DSA_TOPK", "")
            if mode not in priced.PASSES:
                gaps.append(f"{d.name} served on no canonical top-k (VERITY_DSA_TOPK={mode!r})")
            if not any((d / "server").glob("dsa_topk_steps.*.jsonl")) and \
                    not any(k in (counters.get(d.name) or {}) for k in priced.REMEDY):
                gaps.append(f"{d.name} kept no top-k counts (no server/dsa_topk_steps.*.jsonl, none in {kept.name})")
    for d in [*refs, *(d for ds in served.values() for d in ds)]:
        if not any((d / "server").glob(S.STEPS.format("*"))):
            gaps.append(f"{d.name} kept no public steps")
        bare = [m for m in d.glob("*/rep-*/meta.json") if not PASS.issubset(json.loads(m.read_text()))]
        if bare:
            gaps.append(f"{d.name}: {len(bare)} pass records lack {', '.join(sorted(PASS))}")
        elif not (d / "meta.json").is_file() or not priced.counted(d):
            gaps.append(f"{d.name} has no counted pass")
    below = priced.measured(out, reference, stock)
    if not (below and below.get("ci95")):
        gaps.append(f"the job measured no {reference} over {stock} with an interval")
    return gaps


def member_parts(out: Path, model: str) -> tuple[dict, list[str]]:
    """``served`` and ``card`` as `serve_tax.member` gives them, from the job's own records: the model it served and
    the cards it was leased (`GPUS`, which serve_tax's run.sh writes from nvidia-smi inside the lease)."""
    m, gaps, fields = S.MODELS.get(model), [], {}
    if m is None or m.stand_in:
        gaps.append(f"{model} is {'a stand-in' if m else 'no model of serve_tax.MODELS'}: a capture serves the full model")
    else:
        kept = out / "prod_repro.json"
        ran = json.loads(kept.read_text()).get("model") if kept.is_file() else None
        if ran != f"{m.repo}@{m.revision}":
            gaps.append(f"{kept.name} names the model {ran!r}, not {model}'s {m.repo}@{m.revision}")
        else:
            fields["served"] = m.served
    f = out / GPUS
    names = {x.split(",")[0].strip() for x in f.read_text().splitlines()[1:] if x.strip()} if f.is_file() else set()
    if len(names) != 1:
        gaps.append(f"{GPUS} names {len(names)} kinds of card, not the one the job was leased")
    else:
        from verity_bench import kit
        try:
            fields["card"] = kit.card_short(names.pop())
        except SystemExit as e:
            gaps.append(str(e))
    return fields, gaps


def minify_gaps(out: Path, model: str) -> tuple[list[str], dict | None]:
    m = S.MODELS.get(model)
    gaps = [] if m and m.stand_in else [f"{model} is not a stand-in (serve_tax.MODELS' stand_in)"]
    f = out / UNITS
    if not f.is_file():
        return [*gaps, f"no {UNITS}: topk_units.py did not run"], None
    rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    head = next((r for r in rows if "device" in r), {})
    table = priced.timed(f)
    if not any("done_s" in r for r in rows):
        gaps.append(f"{UNITS} has no last record (done_s): topk_units.py did not time every shape")
    gaps += [f"{UNITS} holds no {p} times with and without a failed row"
             for p in head.get("passes") or ("present",) if p not in table]
    summary = {"device": head.get("device"), "vllm": head.get("vllm"), "torch": head.get("torch"),
               "how": f"{head.get('how')}; each range the least and most over the distributions timed (base) and the "
                      "forced failures (failed, less base's range)",
               "units": [{"part": "top-k", "path": p, "case": case, "rows": r, "keys": k, "us": list(us)}
                         for p, t in table.items() for case in ("base", "failed") for (r, k), us in sorted(t[case].items())],
               "unpriced": [list(x) for x in UNPRICED]}
    return gaps, summary


def record(out: Path, stage: str, gaps: list[str], fields: dict, evidence: list[str]) -> list[str]:
    """Rewrites the run's result.json as ``stage``'s record: its kind, ``fields`` in its fingerprint, and its
    validation, passed where ``gaps`` is empty and the run's own validation did not fail; returns its gaps. Refuses an
    out/ that is not the run's and a result of another kind."""
    run_dir = SHARED.RESEARCH_RUN_DIR
    if not run_dir or out.resolve() != (Path(run_dir) / "out").resolve():
        raise SystemExit(f"stage {stage}: {out} is not this research run's out/: a stage's record is the run's own result")
    f = Path(run_dir) / "result.json"
    doc = json.loads(f.read_text()) if f.is_file() else {
        "schema": RESULT, "run_id": SHARED.RESEARCH_RUN_ID or Path(run_dir).name, "measurements": [],
        "measurement_files": [], "artifacts": []}
    if doc.setdefault("kind", KINDS[stage]) != KINDS[stage]:
        raise SystemExit(f"stage {stage}: {f} is a {doc['kind']} result: a run writes one kind")
    own = doc.get("validation") or {}
    if own.get("status") == "failed":
        gaps = [*gaps, f"the run's own validation failed: {own.get('detail')}"]
    doc.setdefault("workload_fingerprint", {}).update(fields)
    doc["artifacts"] = sorted({*(doc.get("artifacts") or ()), *evidence})
    doc["validation"] = {
        "status": "failed" if gaps else "passed", "checker": f"benchmarks/prod_repro/stage.py {stage}",
        "detail": "; ".join(gaps) if gaps else f"holds what the serving cost's {stage} record holds"
                  + (f"; the run's own: {own['detail']}" if own.get("detail") else ""),
        "evidence": evidence}
    tmp = f.with_name(f"{f.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    os.replace(tmp, f)
    return gaps


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="stage", required=True)
    c = sub.add_parser("capture")
    c.add_argument("out", type=Path)
    c.add_argument("--model", required=True)
    c.add_argument("--minify", required=True, metavar="RUN")
    c.add_argument("--exact", metavar="RUN")
    c.add_argument("--reference", default="prod-trace")
    c.add_argument("--stock", default="prod")
    c.add_argument("--ours", nargs="+", default=["prod-record", "prod-record-bisect"])
    m = sub.add_parser("minify")
    m.add_argument("out", type=Path)
    m.add_argument("--model", required=True)
    a = ap.parse_args(argv)
    if a.stage == "capture":
        for flag, run in (("--minify", a.minify), ("--exact", a.exact)):
            if run is not None and not RUN_ID.fullmatch(run):
                raise SystemExit(f"stage capture: {flag} {run!r} is not a run id")
        parts, named = member_parts(a.out, a.model)
        gaps = [*capture_gaps(a.out, a.reference, a.stock, tuple(a.ours)), *named]
        fields = {**parts, "minify": a.minify, **({"exact": a.exact} if a.exact else {})}
        evidence = ["out/prod_repro.json"]
    else:
        gaps, summary = minify_gaps(a.out, a.model)
        evidence = [f"out/{UNITS}"] if (a.out / UNITS).is_file() else []
        if summary:
            (a.out / SUMMARY).write_text(json.dumps(summary, indent=1) + "\n")
            evidence.append(f"out/{SUMMARY}")
        fields = {}
    gaps = record(a.out, a.stage, gaps, fields, evidence)
    print(f"stage {a.stage}: {'FAILED: ' + '; '.join(gaps) if gaps else 'passed'} ({KINDS[a.stage]})")
    return 1 if gaps else 0


if __name__ == "__main__":
    sys.exit(main())
