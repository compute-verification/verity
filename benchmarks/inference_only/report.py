"""`verity bench inference_only`: the inference-only policy of Llama 3.2 1B and its one number, the least slowdown of
training run inside the policy's inference blocks (README.md).

It prints two policies. First the served one, priced from its stated numbers (`served.toml`, or `--policy FILE`): this
tree builds no circuit of the served class, so no check runs on it, and its number is the command's. Then the class this
tree builds, which is not served (`built.py`): its policy by the integration's code, the Lean count's verdict and cost,
its number, and the record's three divisions. Then both over monitoring periods. Then the served class under the value
tie (`tied.toml`, `tied.py`), at the share of a block's work that no PoUW check credits: `--uncredited-share`, which has
no default, so without it no figure of the tied policy prints, and with it every one prints under a label saying it
rests on no PoUW verdict. With `--served OUT`, after the stated policy, a served run's record (`record.py`): the
check's verdicts in their public form (`NciCheck.Record.check`, `NciCheck.RecordCount.check`) and its cost, PoUW's
share and tie, and the value at the record's share. Then one JSON object, printed last and written to `--out`, where
the served policy's number is also written into the run's result under the metric's id (`METRIC`), unless a record
was read.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import built as B
import policy as P
import record as R
import tied as T
from rho import lean, min_cut
from verity_bench import kit
from verity_config.settings import SHARED

UNIT = ("the least slowdown of training run inside one inference block of the policy, max(1, N / limit): N the block's "
        "work and limit the most training work one block finishes (the bound's macs_a_block), in the policy's own MACs")
#: the metric's id (benchmarks/inference_only/metrics.toml), the name the served policy's value is written under in the run's result
METRIC = "training/inference-only-slowdown"
#: the metric's target, its goal's (the `goal` metrics.toml names): a slowdown of at least 1,000. The console's Metrics
#: tab shows the verdict a measurement carries beside its target, and compares nothing itself
TARGET = 1000
#: rho's command for a deployment's slowdown per monitoring period
CALCULATOR = "uv run python benchmarks/rho/periods.py"
#: the tied policy's label: what its figures are and what they rest on
TIED_LABEL = "The served class under the value tie: NOT SERVED, and every figure of it RESTS ON NO PoUW VERDICT"
#: `lean.status` as what the lock says of a statement
STANDING = {"proved": "proved in the lock", "assumption": "assumption", "draft": "no Lean statement",
            "absent": "no lock lists a proof of it"}


def verdict(value: float) -> str:
    """Where the metric's ``value`` stands against `TARGET`, a slowdown the goal wants at least that high."""
    return "within target" if value >= TARGET else "misses"


def _class(text: str) -> tuple[int, int, int]:
    try:
        slots, prompt, outputs = (int(x) for x in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r}: a class is SLOTS,PROMPT,OUTPUTS, three integers") from None
    return slots, prompt, outputs


def _share(text: str) -> float:
    refused = argparse.ArgumentTypeError(f"{text!r}: a share of a block's work is a number from 0 to 1, or a percentage "
                                         "written with %")
    try:
        s = float(text.removesuffix("%")) / (100 if text.endswith("%") else 1)
    except ValueError:
        raise refused from None
    if not 0 <= s <= 1:
        raise refused
    return s


def _n(x: float) -> str:
    """A count: whole where it is whole or large, four figures otherwise."""
    return f"{x:,.0f}" if x == int(x) or x >= 1e4 else f"{x:,.4g}"


def _g(x: float) -> str:
    """Four figures."""
    return f"{x:.4g}".replace("e+", "e")


def _commit() -> str | None:
    if sha := SHARED.RESEARCH_SOURCE_SHA:
        return sha
    got = subprocess.run(["git", "rev-parse", "HEAD"], cwd=P.ROOT, capture_output=True, text=True)
    return got.stdout.strip() if got.returncode == 0 else None


def rests_on(p: P.Policy, b: min_cut.Bound) -> str:
    """What a policy's number rests on: who says its blocks are within `c` and `k`, then the bound's own premises."""
    if p.check["verdict"] == "accepted":
        blocks = f"{p.check['executable']}, which accepts the policy"
    elif p.label == "built":
        by = "verity_vllm's block_counts" if p.division == "session" else "this benchmark's built.divisions"
        blocks = (f"the Python count of what a block reads ({by}, on the record's description), which the Lean check "
                  "does not establish")
    else:
        blocks = "the stated c, k and work, which nothing on this tree builds or checks"
    return f"{blocks}; {b.rests_on}"


def priced(p: P.Policy, sh: P.Shape, blocks: float = 1.0) -> dict[str, Any]:
    """A policy with its number: the bound's inputs and outputs on one block's work (`policy.bound`), ``blocks`` the
    blocks a session is divided into."""
    b = P.bound(p, sh, blocks)
    N = p.work[P.TOTAL] / blocks
    return {
        "label": p.label, "model": p.model, "class": list(p.cls), "division": p.division, "source": p.source,
        "c": p.c, "k": p.k, "n": p.n, "units": P.UNITS,
        "k_by_kind": [{"kind": kind, "bits": bits, "how": how} for kind, bits, how in p.parts],
        "k_unexplained": p.unexplained,
        "work": {**{x: p.work[x] / blocks for x in P.WORK}, "N": N},
        "check": p.check,
        "bound": {"statement": b.statement, "premises": list(b.premises), "N": N, "O": N * sh.outputs * min_cut.BF16 / sh.macs,
                  "size": p.c, "o": [sh.outputs, sh.macs], "b": b.b, "beta_w": b.beta_w, "beta_x": b.beta_x, "F": b.F,
                  "alpha": b.alpha, "macs_a_block": b.macs_a_block, "blocks": b.blocks},
        "value": b.slowdown, "rests_on": rests_on(p, b),
    }


def headline(row: dict[str, Any], p: P.Policy) -> list[str]:
    """The number and why, in the words a reader needs first."""
    N, limit, value = row["work"]["N"], row["bound"]["macs_a_block"], row["value"]
    holds = (f"the bound lets one block hold {_n(limit / N)} times its own work as training, so it shows no slowdown"
             if value == 1 else f"one block finishes at most 1/{_n(value)} of its own work as training")
    lines = [f"{p.model}, class {','.join(map(str, p.cls))}, {p.label}."]
    if p.outside:
        lines.append(f"The class leaves the layers' linears outside the circuit, so a block reads their outputs as fresh "
                     f"input: {p.outside:,} of its k = {p.k:,} bits.")
    lines.append(f"At k = {p.k:,} {holds} (min_cut.bound).")
    return lines


def detail(row: dict[str, Any], p: P.Policy, sh: P.Shape) -> list[str]:
    """One policy's lines: what it is, what k is made of, a block's work, the check's verdict, the bound and the
    number with what it rests on."""
    w, bd, check = row["work"], row["bound"], row["check"]
    limit = bd["macs_a_block"]
    positions = P.class_of(p.cls).positions
    width = max(len(f"{bits:,}") for bits in (*(b for _, b, _ in p.parts), p.unexplained, 0))
    out = [
        f"  source    {p.source}",
        f"  blocks    {p.division}: class {','.join(map(str, p.cls))} (slots, prompt ids, outputs), a session one slot's "
        f"{positions} positions",
        f"  c         {p.c:,} {P.UNITS['c']}",
        f"  k         {p.k:,} {P.UNITS['k']}, made of",
        *(f"              {bits:>{width},}  {kind}{': ' + how if how else ''}" for kind, bits, how in p.parts),
        f"              {p.unexplained:>{width},}  unexplained",
        f"  n         {p.n:,} {P.UNITS['n']}" + ("" if p.label != "built" else ", the stated policy's" + (
            "" if p.division == "session" else " times the division's blocks a session") + ": nothing here derives n"),
        f"  work      N = {_n(w['N'])} MACs a block, in the policy's unit: {_n(w['linears'])} linears + 2 × "
        f"({_n(w['attention'])} attention + {_n(w['lm_head'])} LM head)",
    ]
    if check["verdict"] == "not run":
        out.append(f"  check     not run: {check['why']}")
    else:
        cost = check["cost"]
        out.append(f"  check     {check['executable']} ({check['rule']}): complies: {str(check['complies']).lower()}"
                   + (f", {check['failed']} failed" if not check["complies"] else ""))
        if not check["complies"]:
            f = check["failed"]
            same = f != "c" and check["counted"]["c"] == check["policy"]["c"]
            out.append("            the Lean counts " + (f"c's {P.UNITS['c']} exactly, and " if same else "")
                       + f"{check['counted'][f]:,} {P.UNITS[f]} against {f} = {check['policy'][f]:,}: {check['why']}")
        if cost:
            out.append(f"            the count's cost: {cost['wall_s']:.1f} s wall, {cost['cpu_s']:.1f} s CPU, "
                       f"{cost['peak_mib']:,.0f} MiB peak, on an instance of {cost['instance_bytes']:,} bytes")
    out += [
        f"  bound     min_cut.bound(c, k, n, N, O, size = c): N the block's work, O = N × o × b = {_g(bd['O'])} output "
        "wires,",
        f"            o = {sh.outputs:,} output values over {sh.macs:,} MACs of the model's own forward matmuls, b = "
        f"{bd['b']:g} bits a value",
        f"            {bd['statement']}, β_w = {bd['beta_w']:g}, β_x = " + (_g(bd["beta_x"]) if bd["beta_x"] else "none")
        + f", F = {bd['F']:g}: limit {_n(limit)} MACs of training a block, N / limit = {_g(w['N'] / limit)}",
        f"            the work is {P.WORK_IS}; at most two of its three kinds of product are charged (F = 1)",
        "            size = c, so c cancels, and the count reads no n: the number is the blocks the count needs for one "
        "block's work",
        f"  number    {_n(row['value'])} = max(1, N / limit)",
        f"  rests on  {row['rests_on']}",
    ]
    return out


def division_rows(built: P.Policy, div: dict[str, dict[str, Any]], sh: P.Shape) -> list[dict[str, Any]]:
    """The built class's three divisions, each as a policy of its largest block: `n` the session policy's times the
    blocks a session, a block's work the session's shared evenly. The Lean count ran on the session Program only."""
    rows = []
    for name in B.DIVISIONS:
        d = div[name]
        per = d["blocks_a_session"]
        check = built.check if name == "session" else {
            "verdict": "not run", "why": "the integration gives the count a Program of session blocks only"}
        parts = built.parts if name == "session" else tuple((kind, bits, "") for kind, bits in d["k_by_kind"])
        p = dataclasses.replace(built, c=d["c"], k=d["k"], n=int(built.n * per), parts=parts, check=check, division=name)
        rows.append({**priced(p, sh, per), "blocks_a_session": per})
    return rows


def _table(head: Sequence[str], rows: Sequence[Sequence[str]], left: int, indent: str = "  ") -> list[str]:
    """A table's lines: the first ``left`` columns flush left, the rest flush right."""
    table = [tuple(head), *map(tuple, rows)]
    widths = [max(len(r[i]) for r in table) for i in range(len(head))]
    return [indent + "  ".join(x.ljust(w) if i < left else x.rjust(w) for i, (x, w) in enumerate(zip(r, widths))).rstrip()
            for r in table]


#: the tied section's words where its share is typed: its label, where the share is from, what the policy is, the
#: share's and the tie's premise, and what a limit is
TYPED = {
    "label": TIED_LABEL,
    "share": "The uncredited share s = {s}% of a block's work was typed on the command line (--uncredited-share). No PoUW "
             "verdict of a run states it here, so each figure holds only where PoUW's checks credit all but that share of "
             "every block's work.",
    "what": "{model}, class {cls}, with every tied value right ({file}). Nothing serves it, and this tree builds no "
            "circuit of it.",
    "premise": "the share s, typed; that PoUW ties every word it credits, which no verdict of a run from this tree shows;",
    "limit": "No limit is proved: it is arithmetic on the typed share",
}


def tied_lines(d: dict[str, Any], file: str, words_of: Mapping[str, str] = TYPED) -> list[str]:
    """The tied policy's section (`tied.priced`): the label, what the policy and its work are, the untied words and the
    uncredited share, one block's figures, and each fleet's execution phase and monitoring periods under each
    reading; in ``words_of``'s words (`TYPED`'s keys)."""
    u, un, block = d["uncredited"], d["untied"], d["block"]
    s, N, k = u["share"], d["work"]["N"], d["k"]
    width = max(len(f"{x['bits']:,}") for x in d["k_by_kind"])
    words = block["with_words"]
    out = [
        words_of["label"],
        "  " + words_of["share"].format(s=_g(100 * s)),
        "  " + words_of["what"].format(model=d["model"], cls=",".join(map(str, d["class"])), file=file),
        f"  work      the figures are for {P.WORK_IS}",
        f"  source    {d['source']}",
        f"  c         {d['c']:,} {P.UNITS['c']}",
        f"  k         {k:,} {P.UNITS['k']}, made of",
        *(f"              {x['bits']:>{width},}  {x['kind']}{': ' + x['how'] if x['how'] else ''}" for x in d["k_by_kind"]),
        f"  N         {_n(N)} MACs a block, in the policy's unit",
        f"  untied    at most {un['words']:,} untied words in a node's accepted execution phases for each {un['blocks']} "
        f"blocks of one {un['phase_seconds']:,} s phase, each at most {un['word_bits']} fresh bits in the one block that "
        f"reads it and fed by at most {un['macs_a_word']:,} MACs; one block reads at most its {un['most_a_block']:,} output "
        "words of the linears",
        f"  uncredited  s N = {_n(round(u['macs_a_block']))} MACs a block, which feed at most {_n(round(u['words_a_block']))} "
        f"output words of the linears, a word being fed by at least {u['least_fan_in']:,} MACs (the least fan-in): with "
        f"those words as fresh input a block reads k + {un['word_bits']} × {_n(round(u['words_a_block']))} = "
        f"{_n(round(u['k_with_words']))} bits, and every limit at s charges the share's MACs whole",
        "",
        "  One block (min_cut.bound), each number with its limit at s, 1 / (s + 1 / number) (min_cut.capped): a "
        "developer gets the uncredited share whole",
        f"    every word tied, k = {k:,}: number {block['number']:,.1f}, limit at s {block['limit']:,.1f}",
        f"    {T.IN_K}, k = {_n(round(words['k']))}: number {words['number']:,.1f}, limit at s "
        f"{words['limit']:,.1f}",
        "",
        "  Over one execution phase and over a monitoring period, all of a fleet's compute in the policy's blocks at "
        "capacity (min_cut.bound_charged; the slowdown calculator's fleets and periods). A node's bound on its untied "
        "words doesn't grow with the period, so one phase's figure is the weakest, and is not a period's",
    ]
    seen = {(r["where"], r["period"]): r for r in d["periods"]}
    out += _table(("where", "over one", "blocks", "the blocks' gates", "untied words at most", "a block"),
                  [(r["where"], r["period"], _g(r["blocks"]), _g(r["gates"]["blocks"]), _g(r["untied"]),
                    _g(r["untied_a_block"])) for r in seen.values()], 2, "    ")
    out += ["", f"  Each figure under its reading's name, which holds its threshold, with its limit at s = {_g(100 * s)}% "
            "and the pooled total in gates (the gates c counts): the training the blocks hold at most, at the limit"]
    out += _table(("where", "over one", "reading", "blocks charged at most", "number", "limit at s",
                   "training's gates at most", "of the blocks' gates"),
                  [(r["where"], r["period"], r["reading"], _g(r["charged"]), f"{r['number']:,.1f}",
                    f"{r['limit']:,.1f}", _g(r["gates"]["training_at_the_limit"]), _g(100 * r["share"]) + "%")
                   for r in d["periods"]], 3, "    ")
    charged, rows = (STANDING[lean.status(x)] for x in (min_cut.HE_CHARGED, min_cut.HE_ROWS))
    out += [
        f"  proved    {min_cut.HE} ({STANDING[lean.status(min_cut.HE)]}), the count one block's number is solved "
        f"from. A period's count is {min_cut.HE_CHARGED} ({charged}), and with a β_x a block {min_cut.HE_ROWS} ({rows}). "
        + words_of["limit"],
        f"  premise   {words_of['premise']} the bound on a period's untied words ({min_cut.UNTIED_WORDS}) and an untied word's bits "
        f"({min_cut.WORD_BITS}), neither a Lean statement; (H) {min_cut.HARDNESS} and (E) {min_cut.EXPORTS}, conjectures, "
        "at the circuit with the untied words as inputs; a threshold fixed before the period, with so few blocks over "
        "it in the partition the period's audit reads; and the work's structure, as above",
        "  scaled    every row: the blocks are a fleet's at its data sheet's rate, the fleets' sizes the census's; a "
        "node's bound on its untied words is one execution phase's, scaled by the blocks the node serves in a phase; "
        "the uncredited share is taken as every block's; and one block may be untied whole",
        "  not in any figure: a dropped execution phase, a voluntary row or a version the verdict doesn't cover, which "
        "nothing here bounds",
    ]
    return out


def _said(v: Mapping[str, Any]) -> str:
    """A rule's public verdict as it reads."""
    if "error" in v:
        return f"refused: error {v['error']}" + (f": {v['message']}" if "message" in v else "")
    return f"complies: {str(v['complies']).lower()}"


def served_record(files: R.Files, clock: Callable[[], float], exe: Path | None) -> dict[str, Any]:
    """Run ``files``'s record (`record.py`): the check's public verdicts and cost, PoUW's share and tie, and the value
    with its reading and mark. A value of 1 (the check refused, or no PoUW accept of record) is measured; one block's
    limit under the tie is partial; each period's is modeled."""
    got = R.read(files)
    run, pol = got["run"], got["policy"]
    pouw = R.share(got["verdicts"], pol, R.previous(files))
    with tempfile.TemporaryDirectory() as d:
        said = R.checked(files, got, Path(d), clock, exe)
    doc = {"run": run, "model": pol["serve"]["model"],
           "class": [pol["class"][x] for x in ("slots", "prompt", "outputs")], **{x: pol[x] for x in "ckn"},
           "files": {"program_bytes": files.program.stat().st_size, "circuit_bytes": files.circuit.stat().st_size,
                     "log_records": len(files.logs)},
           "check": {x: {"rule": rule, "public_form": fn, **said[x]} for x, (rule, fn) in R.PUBLIC.items()},
           "seconds": said["seconds"], "cost": said["cost"], "pouw": pouw, "tied": None, "tied_why": None}
    of_record = "c, n, the work, the check's accept, PoUW's accept and s are the record's"
    if refused := [rule for x, (rule, _) in R.PUBLIC.items() if said[x].get("complies") is not True]:
        which = "both rules" if len(refused) == 2 else refused[0]
        one = {"value": 1, "mark": "measured", "reading": f"the check refused the record ({which}), so nothing shows its "
               "blocks within c, k and n", "inputs": "the check's verdicts, the record's", "short": f"{which} refused"}
    elif pouw["share"] == 1:
        one = {"value": 1, "mark": "measured", "reading": "no PoUW accept of record, so s = 1, and every limit at s = 1 "
               "is 1 (min_cut.capped)", "inputs": f"{R.VERDICT}, the record's", "short": "no PoUW accept, s = 1"}
    else:
        try:
            t = R.tied_of(pol, run)
        except R.Refused as e:
            doc["tied_why"] = str(e)
            one = {"value": None, "mark": None, "reading": f"no figure: {e}", "inputs": of_record}
        else:
            tied = T.priced(t, P.shape(t.policy.model), pouw["share"], source=f"run {run}'s PoUW verdict of record",
                            verdict={x: pouw[x] for x in ("field", "verdict", "C", "W", "tie")})
            doc["tied"] = tied
            b = tied["block"]
            value, reading = min((b["limit"], "every word tied"), (b["with_words"]["limit"], T.IN_K))
            tie = "the tie: that PoUW ties every word it credits" + (f" ({R.NO_TIE})" if pouw["tie"]["window"] is None
                                                                       else "")
            one = {"value": value, "mark": "partial", "reading": reading,
                   "inputs": f"{of_record}, and k its policy's kinds but the linears' outputs; premises: {tie}, and "
                             f"{P.TIED.name}'s [fresh] and [untied]"}
    if one["mark"] == "measured":
        short = one.pop("short")
        periods = [{"where": r["where"], "period": r["period"], "value": 1, "reading": short, "mark": "measured"}
                   for r in P.fleet_periods(pol["n_inputs"]["work"]["theta_macs"])]
    elif doc["tied"] is not None:
        least: dict[tuple[str, str], dict[str, Any]] = {}
        for r in doc["tied"]["periods"]:
            if (key := (r["where"], r["period"])) not in least or r["limit"] < least[key]["value"]:
                least[key] = {"where": r["where"], "period": r["period"], "value": r["limit"], "reading": r["reading"],
                              "mark": "modeled"}
        periods = list(least.values())
    else:
        periods = []
    return {**doc, "block": one, "periods": periods}


def record_words(d: Mapping[str, Any]) -> dict[str, str]:
    """The tied section's words (`TYPED`'s keys) for run record ``d``'s class."""
    tie = d["pouw"]["tie"]
    tied = (R.NO_TIE if tie["window"] is None else
            f"its verdict holds its passes to window record {tie['window'][:16]}..., the run's registration's previous")
    return {
        "label": f"Run {d['run']}'s class under the value tie: c, n, the work, the check's accept, PoUW's accept and s "
                 "are the record's, and THE TIE IS A PREMISE",
        "share": "The uncredited share s = {s}% of a block's work is " + f"run {d['run']}'s: 1 - C / W, C the MACs its "
                 f"PoUW verdict of record credits ({R.VERDICT} = accept) and W its blocks' work.",
        "what": "{model}, class {cls}, with every tied value right: the record's c, n and work, and k its policy's kinds "
                "but the linears' outputs, with {file}'s [fresh]. Nothing serves it.",
        "premise": f"that PoUW ties every word it credits, which no Lean statement gives ({tied});",
        "limit": "No limit is proved: it is arithmetic on the record's share",
    }


def record_lines(d: Mapping[str, Any]) -> list[str]:
    """Run record ``d``'s lines (`served_record`): its value, the check's public verdicts and cost, PoUW's verdict, the share
    and the tie."""
    one, pouw, cost = d["block"], d["pouw"], d["cost"]
    value = "none" if one["value"] is None else f"{_n(one['value'])} ({one['mark']})"
    f, ck, case = d["files"], cost["check"], cost["case"]
    out = [f"FROM RUN {d['run']}'s RECORD: {value}, {one['reading']}",
           f"  {d['model']}, class {','.join(map(str, d['class']))}: a Program of {f['program_bytes']:,} bytes, a circuit "
           f"record of {f['circuit_bytes']:,} bytes and a registration log of {f['log_records']} records",
           f"  check     {R.CHECK}, in its public form only:",
           *(f"              {x['rule']} ({x['public_form']}): {_said(x)}" for x in d["check"].values()),
           f"            the count against the record's c = {d['c']:,}, k = {d['k']:,}, n = {d['n']:,}",
           f"  cost      in the clear, at the record's size: the check {ck['wall_s']:,.1f} s wall, {ck['cpu_s']:,.1f} s CPU, "
           f"{ck['peak_mib']:,.0f} MiB peak, on a case of {case['bytes']:,} bytes (by its own clock: read at "
           f"{d['seconds']['read']} s, the rate rule's verdict at {d['seconds']['check']} s, the count's at "
           f"{d['seconds']['count']} s); writing the case ({R.SCRIPT.name}) {case['wall_s']:,.1f} s wall, "
           f"{case['peak_mib']:,.0f} MiB peak",
           f"  PoUW      of record by {pouw['field']} = {pouw['verdict']!r}"
           + (f", from {pouw['answer_from']}" if pouw["answer_from"] else "")]
    if pouw["C"] is None:
        out += ["            no accept of record" + (" (a development accept counts as none)" if pouw["development"]
                                                      else "") + f": {pouw['why'][:400]}",
                f"  share     s = 1: no PoUW accept credits any of the blocks' work (W = {pouw['slots']} slots × "
                f"{pouw['theta_macs']:,} = {pouw['W']:,} MACs)",
                "  tie       nothing compared: there is no accept to tie"]
    else:
        tie = pouw["tie"]
        out += [f"  share     s = 1 - C / W = {_g(100 * pouw['share'])}%: C = {pouw['C']:,} MACs credited of the {pouw['macs']:,} "
                f"MACs of linears it counts, W = {pouw['slots']} slots × {pouw['theta_macs']:,} = {pouw['W']:,} MACs",
                f"  tie       {R.NO_TIE}" if tie["window"] is None else
                f"  tie       PoUW's window record {tie['window'][:16]}... = the run's registration's previous "
                f"{tie['previous'][:16]}... (the log's last record, whose chain the check holds): compared, equal"]
    if d["tied_why"]:
        out.append(f"  tied      no figure: {d['tied_why']}")
    out.append(f"  inputs    {one['inputs']}")
    if d["periods"]:
        out.append("  Over each monitoring period (and, under the tie, one execution phase), all of a fleet's compute: the "
                   "least limit over the readings, with its reading")
        out += _table(("where", "over one", "value", "reading", "mark"),
                      [(r["where"], r["period"], _n(r["value"]), r["reading"], r["mark"]) for r in d["periods"]], 2,
                      "    ")
    return out


def main(argv: Sequence[str] | None = None, check: Callable[..., dict[str, Any]] = B.lean_count, circuit: Any = None,
         commit: Callable[[], str | None] = _commit, clock: Callable[[], float] = time.perf_counter,
         record_check: Path | None = None) -> dict[str, Any]:
    """Print the report and return its JSON object. ``check`` and ``circuit`` are `built.build`'s; ``clock`` times the
    record check, and ``record_check`` is its executable (default: built here)."""
    ap = argparse.ArgumentParser(prog="verity bench inference_only", description=__doc__.split("\n\n")[0])
    ap.add_argument("--policy", type=Path, default=P.SERVED, metavar="FILE",
                    help="a stated policy to price (default: the served policy, served.toml)")
    ap.add_argument("--model", default=B.MODEL, help="the built class's checkpoint, one the integration's manifests pin")
    ap.add_argument("--class", dest="cls", type=_class, metavar="SLOTS,PROMPT,OUTPUTS",
                    help="the built class (default: the stated policy's)")
    ap.add_argument("--division", choices=B.DIVISIONS, default="session",
                    help="which of the built class's blocks its number is of")
    ap.add_argument("--stated-only", action="store_true", help="price the stated policy and build nothing")
    ap.add_argument("--uncredited-share", type=_share, metavar="S",
                    help="the share of a block's work that no PoUW check credits, from 0 to 1 or as a percentage with "
                         "%% (no default): print the served class under the value tie (tied.toml) at it, under a "
                         "label saying its figures rest on no PoUW verdict")
    ap.add_argument("--served", type=Path, metavar="OUT",
                    help="a served run's out/ (record.OUT_FILES): run the inference-only check on its record and print its "
                         "value at the share of its blocks' work that no PoUW verdict of record credits; with --program, "
                         "--circuit and --log, the files the run registered")
    ap.add_argument("--program", type=Path, metavar="FILE", help="the served run's Program")
    ap.add_argument("--circuit", type=Path, metavar="FILE", help="the served run's circuit record")
    ap.add_argument("--log", type=Path, nargs="+", metavar="FILE",
                    help="the served run's registration log, in order, its own registration last")
    ap.add_argument("--out", type=Path, default=SHARED.OUT, metavar="DIR",
                    help="write inference_only.json here (default: $OUT; unset: only printed)")
    args = ap.parse_args(argv)
    named = [x for x in ("program", "circuit", "log") if getattr(args, x)]
    if (args.served is None) != (not named) or (args.served and len(named) != 3):
        ap.error("--served OUT takes --program, --circuit and --log, and they take it")

    def say(line: str) -> None:
        print(f"[inference_only] {line}", file=sys.stderr, flush=True)

    stated = P.stated(args.policy)
    if stated.label == "tied":
        ap.error(f"{args.policy.name} is the served class under the value tie, whose number is never printed alone: "
                 "leave --policy out and state --uncredited-share S, which prints it beside its limit at that share")
    served = priced(stated, P.shape(stated.model))
    print("Inference-only: the least slowdown of training run inside the policy's inference blocks\n")
    print(f"AS SERVED: {_n(served['value'])}" if stated.label == "served" else f"AS STATED: {_n(served['value'])}")
    print("\n".join("  " + line for line in headline(served, stated)))
    print("  Not checked here: this tree builds no circuit of this class, so no check ran on it, and the number is priced "
          f"from the policy's stated numbers ({args.policy.name}).\n")
    print(f"The {stated.label} policy (stated, not built)")
    print("\n".join(detail(served, stated, P.shape(stated.model))), flush=True)
    record = None
    if args.served:
        try:
            record = served_record(R.Files(args.served, args.program, args.circuit, tuple(args.log)), clock,
                                   record_check)
        except R.Refused as e:
            ap.error(f"the record is refused: {e}")
        print("\n" + "\n".join(record_lines(record)), flush=True)
        if record["tied"] is not None:
            print("\n" + "\n".join(tied_lines(record["tied"], P.TIED.name, record_words(record))), flush=True)
    policies, divisions = [served], []
    if not args.stated_only:
        cls = args.cls or stated.cls
        try:
            policy, div = B.build(args.model, cls, stated.n, check=check, say=say, circuit=circuit)
        except B.Refused as e:
            ap.error(str(e))
        sh = P.shape(args.model)
        divisions = division_rows(policy, div, sh)
        built = next(d for d in divisions if d["division"] == args.division)
        policies.append(built)
        chosen = dataclasses.replace(policy, c=built["c"], k=built["k"], n=built["n"], check=built["check"],
                                     division=args.division,
                                     parts=tuple((x["kind"], x["bits"], x["how"]) for x in built["k_by_kind"]))
        print(f"\nThe class this tree builds: NOT SERVED. Its number: {_n(built['value'])}")
        print(f"  {args.model}, class {','.join(map(str, cls))}: the exact-int8 checkpoint with its layers' linears "
              "inside the circuit, so what registering the linears inside the circuit would give, on that checkpoint.")
        print("\n".join(detail(built, chosen, sh)))
        print("\n  The record's divisions (c and k the largest block's, a block's work the session's shared evenly; the "
              "check ran on the session's Program)")
        head = ("division", "blocks a session", "c", "k", "N a block", "limit", "number")
        table = [head] + [(d["division"], _n(d["blocks_a_session"]), f"{d['c']:,}", f"{d['k']:,}", _n(d["work"]["N"]),
                           _n(d["bound"]["macs_a_block"]), _n(d["value"])) for d in divisions]
        widths = [max(len(r[i]) for r in table) for i in range(len(head))]
        for r in table:
            print("    " + "  ".join(x.ljust(w) if i == 0 else x.rjust(w) for i, (x, w) in enumerate(zip(r, widths))))
        for d in divisions[1:]:
            kinds = "; ".join(f"{x['bits']:,} {x['kind']}" for x in d["k_by_kind"][:3])
            print(f"    {d['division']}: its largest block's k is {kinds}" + (" …" if len(d["k_by_kind"]) > 3 else ""))

    periods = [{"policy": p["label"], **row} for p in policies
               for row in P.periods(p["work"]["N"], p["bound"]["macs_a_block"])]
    print("\nOver a monitoring period, all of a fleet's compute in the policy's blocks at capacity: the training the blocks "
          "hold at most, in MACs and as a share of that compute")
    print("  (the slowdown calculator's fleets and periods, benchmarks/rho; a fleet's rate is its "
          "cards' data-sheet dense int8 rate)")
    head = ("policy", "where", "period", "blocks", "training MACs at most", "share of the compute at most")
    table = [head] + [(r["policy"], r["where"], r["period"], _g(r["blocks"]), _g(r["training_macs"]),
                       _g(100 * r["share"]) + "%") for r in periods]
    widths = [max(len(r[i]) for r in table) for i in range(len(head))]
    for r in table:
        print("  " + "  ".join(x.ljust(w) if i < 3 else x.rjust(w) for i, (x, w) in enumerate(zip(r, widths))))
    hard, exports = (lean.status(x) for x in (min_cut.HARDNESS, min_cut.EXPORTS))
    print(f"  proved    {min_cut.HE} ({lean.status(min_cut.HE)} in the lock), the count a block's limit is solved from: "
          "at k fresh bits a block, m blocks finish at most m (k + α) + (k/2) √(m O / q) of a work's MACs")
    print(f"  premise   (H) {min_cut.HARDNESS} ({hard}) at the β above and (E) {min_cut.EXPORTS} ({exports}) at b, "
          "conjectures; the work's structure (its values one word each, no two matmuls sharing both operands), taken of "
          "a training work nobody built here; and that every block reads at most k, which the Lean check accepts for no "
          "class today")
    print("  scaled    one block's ratio, multiplied through the blocks a fleet holds at its data sheet's rate: the "
          "fleets' sizes are the census's, and no fleet was measured")
    print("\nUnits: the number is the policy's block's work over the limit, both in the policy's own MACs. Not applied:")
    print("  the two thirds: at F = 1 the charged work is at most two of a training step's three kinds of product "
          "(forward, input-gradient, weight-gradient), so where the weight-gradient products are not charged a whole "
          "step slows by at least two thirds of the number; the number takes N as charged whole")
    print("  BF16: the policy's unit counts a MAC of the layers' linears as one and a BF16 MAC (attention, the LM head) "
          "as two, the card's dense int8 rate against its BF16 rate; N is not converted to either")
    print(f"A deployment's slowdown, over what it registers and what it leaves free, is the calculator's: {CALCULATOR}")

    tied = None
    if args.uncredited_share is None:
        print(f"\nThe served class under the value tie ({P.TIED.name}): no figure printed. Its figures hold only beside "
              "the share of a block's work that no PoUW check credits, and that share has no default: no PoUW verdict of "
              "a run states one here. --uncredited-share S prints them at a typed share, under a label saying they rest "
              "on no PoUW verdict.")
    else:
        policy = P.tied()
        tied = T.priced(policy, P.shape(policy.policy.model), args.uncredited_share)
        print("\n" + "\n".join(tied_lines(tied, P.TIED.name)))

    doc = {"benchmark": "inference_only", "value": served["value"], "unit": UNIT, "check": served["check"],
           "rests_on": served["rests_on"], "work_is": P.WORK_IS, "policies": policies, "divisions": divisions,
           "periods": periods, "tied": tied, "record": record, "commit": commit()}
    if run := SHARED.RESEARCH_RUN_ID:
        doc["run"] = run
    text = json.dumps(doc, sort_keys=True)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "inference_only.json").write_text(text + "\n")
        # the metric reads the stated policy until a data file the command reads names a served run of record
        if stated.label == "served" and record is None:
            kit.writer(args.out).measurement(METRIC, doc["value"], UNIT, target=TARGET, mark="modeled",
                                             verdict=verdict(doc["value"]))
    print("\n" + text)
    return doc
