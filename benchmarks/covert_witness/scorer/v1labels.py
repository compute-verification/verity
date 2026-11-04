"""The labels a refresh puts on training/algorithm-policy's rows before te_core.algorithm_policy writes, stores or PUTs
them, and which of its coverage's points they leave priced (coordinator 00:47Z, 01:19Z and 01:47Z: v1cycle's labels,
not te_core's). Standard library only: `run` executes in te_core's own process (v1cycle.POLICY).

A void row is one whose rho reads 1 with a reason: te_core.formalism.published binds `none` only where every kind is
computed and none exceeds 1, so a `why` there is a void fresh-products term's (the products count diverges and no
subset prices above 1). Its 1 is no attack's cost, and its rests_on reads `UNPRICED` with that cause. A row the bound
prices (binding fresh_products, or none without a void) whose run leaves the embedding frozen (`frozen_embedding`)
rests on the broadcast norm-gain rule, under review until io-bound's FINDING: its rests_on gains `UNDER_REVIEW`.
"""
import collections
import gzip
import json
import re
from pathlib import Path

UNPRICED = "unpriced: rho >= 1 only ({})"
UNDER_REVIEW = "priced through the norm gains' rule, under review (quant 00:24Z)"
#: the note's sentence naming the labels: the first that fits the route's note, then the same with te_core's dial
#: sentence (`DIAL`) dropped, as te_core.algorithm_policy.note drops it when over
NOTE = (' rests_on: a void row\'s "unpriced: rho >= 1 only", a frozen embedding\'s "under review" (quant 00:24Z).',
        " rests_on: void rows unpriced (rho >= 1 only), frozen embeddings under review.",
        " Void rows unpriced (rests_on).")
DIAL = " Dial: rho_inputs (u_in exact at its k)."
#: a forward matmul in a trace's op log (macs_log): its weight's class, `theta` frozen or `updated` trained, and its
#: output's width
FWD_MM = re.compile(r"fwd (theta|updated) mm \[\[\d+, \d+\], \[\d+, (\d+)\]\]")


def frozen_embedding(rec: dict) -> bool | None:
    """Whether a trace record's run leaves the (tied) embedding frozen: no forward matmul onto the vocabulary, the
    widest output, reads a trained weight. None where its op log has no forward matmul."""
    ops = (rec.get("macs_log") or {}).get("ops") or {}
    mm = [(m[1], int(m[2])) for k in ops if (m := FWD_MM.fullmatch(k))]
    if not mm:
        return None
    wide = max(n for _, n in mm)
    return all(c == "theta" for c, n in mm if n == wide)


def frozen_runs(caches: Path) -> dict[str, bool | None]:
    """`frozen_embedding` of each traced run, from the first record of its trace in any of `caches`' layers."""
    out = {}
    for f in sorted(caches.glob("*/records/*.jsonl.gz")):
        run = f.name.split(".")[0]
        if out.get(run) is None:
            with gzip.open(f, "rt") as h:
                line = h.readline()
            if line.strip():
                out[run] = frozen_embedding(json.loads(line))
    return out


def labelled(p: dict, frozen: dict[str, bool | None]) -> dict[str, int]:
    """Labels te_core.algorithm_policy's table `p` in place, each row's rests_on, and counts them: `unpriced` void
    rows (`unpriced_frozen` of them with a frozen embedding), `under_review` bound-priced rows with a frozen embedding,
    `frozen_unpriced_else` frozen-embedding rows the bound doesn't price otherwise (unmeasured, Fit or a target) and
    `embedding_unknown` rows whose run has no trace here."""
    ix = {c: i for i, c in enumerate(p["columns"])}
    n = collections.Counter()
    for r in p["rows"]:
        why = (json.loads(r[ix["kinds"]]) if r[ix["kinds"]] else {}).get("why")
        fe = frozen.get(str(r[ix["record"]]).split("@")[0])
        n["embedding_unknown"] += fe is None
        if r[ix["binding_kind"]] == "none" and why:
            r[ix["rests_on"]] = UNPRICED.format(why)
            n["unpriced"] += 1
            n["unpriced_frozen"] += bool(fe)
        elif fe and r[ix["binding_kind"]] in ("fresh_products", "none"):
            r[ix["rests_on"]] = f"{r[ix['rests_on']]}; {UNDER_REVIEW}"
            n["under_review"] += 1
        elif fe:
            n["frozen_unpriced_else"] += 1
    return dict(n)


def run(ap, frozen: Path) -> int:
    """te_core.algorithm_policy's main (`ap`, the module) with its table `labelled` as `build` returns it, so the
    tables, their stored art, the console's copy and its PUT all carry the labels, and the note names them (`NOTE`);
    the counts in labels.json beside `frozen` (`frozen_runs`' JSON). Its arguments are sys.argv's."""
    import sys

    runs, build, note, n = json.loads(frozen.read_text()), ap.build, ap.note, {}

    def built(*a, **k):
        p = build(*a, **k)
        n.update(labelled(p, runs))
        (frozen.parent / "labels.json").write_text(json.dumps(n) + "\n")
        return p

    def noted(*a, **k):
        t = note(*a, **k)
        if not (n.get("unpriced") or n.get("under_review")):
            return t
        return next((x for b in (t, t.replace(DIAL, "")) for s in NOTE if len(x := b + s) <= ap.console.NOTE_MAX), t)

    ap.build, ap.note = built, noted
    return ap.main(sys.argv[1:])


def priced(d: Path) -> str | None:
    """Which of the coverage's points (tables/coverage.json's non-EMPTY cells, each a row of the labelled
    tables/algorithm-policy.json under te_core.algorithm_policy's `--out` `d`) are priced, as one sentence: the void
    ones (`UNPRICED`) are not, and those with a frozen embedding (`UNDER_REVIEW`) are under review. None without them."""
    t, c = d / "tables" / "algorithm-policy.json", d / "tables" / "coverage.json"
    if not (t.exists() and c.exists()):
        return None
    p, cov = json.loads(t.read_text()), json.loads(c.read_text())
    on = p["columns"].index("rests_on")
    points = [(x, p["rows"][x["row"]][on] or "") for x in cov if x.get("row") is not None]
    where = lambda x: f"g {x['gain_target']:g} {x['data']} rho {x['rho_decade']} ({x['method_key']} {x['record']})"
    void = [f"{where(x)}: {s}" for x, s in points if s.startswith(UNPRICED.split(" (")[0])]
    review = [where(x) for x, s in points if s.endswith(UNDER_REVIEW)]
    return (f"Of the coverage's {len(points)} points, {len(points) - len(void)} are priced"
            + (f", {len(review)} of them {UNDER_REVIEW}: " + "; ".join(review) if review else "")
            + (f". {len(void)} unpriced, a void row's 1 no attack's cost: " + "; ".join(void) if void else "") + ".")
