"""`verity bench inference_only`: the inference-only policy of Llama 3.2 1B and its one number, the least slowdown of
training run inside the policy's inference blocks (README.md).

It prints two policies. First the served one, priced from its stated numbers (`served.toml`, or `--policy FILE`): this
tree builds no circuit of the served class, so no check runs on it, and its number is the command's. Then the class this
tree builds, which is not served (`built.py`): its policy by the integration's code, the Lean count's verdict and cost,
its number, and the record's three divisions. Then both over monitoring periods, and one JSON object, printed last and
written to `--out`.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import built as B
import policy as P
from rho import lean, min_cut
from verity_config.settings import SHARED

UNIT = ("the least slowdown of training run inside one inference block of the policy, max(1, N / limit): N the block's "
        "work and limit the most training work one block finishes (the bound's macs_a_block), in the policy's own MACs")
#: rho's command for a deployment's slowdown per monitoring period
CALCULATOR = "uv run python verity/protocols/inference_only/rho/periods.py"


def _class(text: str) -> tuple[int, int, int]:
    try:
        slots, prompt, outputs = (int(x) for x in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r}: a class is SLOTS,PROMPT,OUTPUTS, three integers") from None
    return slots, prompt, outputs


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
        "            the work is taken as training of the model's own matmul shapes, its weights outside θ, at most two "
        "of its three kinds of product charged (F = 1)",
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


def main(argv: Sequence[str] | None = None, check: Callable[..., dict[str, Any]] = B.lean_count, circuit: Any = None,
         commit: Callable[[], str | None] = _commit) -> dict[str, Any]:
    """Print the report and return its JSON object. ``check`` and ``circuit`` are `built.build`'s."""
    ap = argparse.ArgumentParser(prog="verity bench inference_only", description=__doc__.split("\n\n")[0])
    ap.add_argument("--policy", type=Path, default=P.SERVED, metavar="FILE",
                    help="a stated policy to price (default: the served policy, served.toml)")
    ap.add_argument("--model", default=B.MODEL, help="the built class's checkpoint, one the integration's manifests pin")
    ap.add_argument("--class", dest="cls", type=_class, metavar="SLOTS,PROMPT,OUTPUTS",
                    help="the built class (default: the stated policy's)")
    ap.add_argument("--division", choices=B.DIVISIONS, default="session",
                    help="which of the built class's blocks its number is of")
    ap.add_argument("--stated-only", action="store_true", help="price the stated policy and build nothing")
    ap.add_argument("--out", type=Path, default=SHARED.OUT, metavar="DIR",
                    help="write inference_only.json here (default: $OUT; unset: only printed)")
    args = ap.parse_args(argv)

    def say(line: str) -> None:
        print(f"[inference_only] {line}", file=sys.stderr, flush=True)

    stated = P.stated(args.policy)
    served = priced(stated, P.shape(stated.model))
    print("Inference-only: the least slowdown of training run inside the policy's inference blocks\n")
    print(f"AS SERVED: {_n(served['value'])}" if stated.label == "served" else f"AS STATED: {_n(served['value'])}")
    print("\n".join("  " + line for line in headline(served, stated)))
    print("  Not checked here: this tree builds no circuit of this class, so no check ran on it, and the number is priced "
          f"from the policy's stated numbers ({args.policy.name}).\n")
    print(f"The {stated.label} policy (stated, not built)")
    print("\n".join(detail(served, stated, P.shape(stated.model))), flush=True)
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
    print("  (the slowdown calculator's fleets and periods, verity/protocols/inference_only/rho; a fleet's rate is its "
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

    doc = {"benchmark": "inference_only", "value": served["value"], "unit": UNIT, "check": served["check"],
           "rests_on": served["rests_on"], "policies": policies, "divisions": divisions, "periods": periods,
           "commit": commit()}
    if run := SHARED.RESEARCH_RUN_ID:
        doc["run"] = run
    text = json.dumps(doc, sort_keys=True)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "inference_only.json").write_text(text + "\n")
    print("\n" + text)
    return doc
