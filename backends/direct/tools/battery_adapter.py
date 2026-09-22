"""Run the red team's BabyBear forgery battery against Candidate B's AIR constraints.

Every battery case (``fixtures/redteam/babybear-battery/v1``, 83 cases, expected verdict *reject*) is an adversarial
witness in the WP2 hint layout.  This adapter turns each case into B's row format -- the case's hints imposed by
name on B's witness generator, every realisation column (2-bit limbs, one-hot selectors, exclusion inverses,
materialised copies) derived from the imposed values, which is the only way a prover could fill them -- and asks
``direct-unit check-rows`` to evaluate **every AIR constraint** on the resulting trace over BabyBear (the verifier's
view; witness-generation rejections are ignored).  A case is *accepted* iff no constraint fails.

Case levels:

* ``step``   one unit on ``c``: the forged row at position 1 of a VU (``first = 0`` so ``c`` may be non-zero), followed
             by an honest zero unit whose input limbs are the forged row's output limbs (the link constraint).
* ``vu``     two units from ``c_0 = 0`` plus the epilogue: rows 0 and 1 carry the case's unit hints, rows 2..95 are
             honest zero units (the identity on a finite accumulator) carrying the word to the last row, where the
             case's epilogue hints and the claimed ``y16`` are bound by ``vu.out``.
* ``gadget`` a z3 model of one gadget: inputs pinned to the contract's premises, hints imposed, outputs bound.

Positive controls: every step/vu case is also run with honest hints (no overrides) and the honest word; every gadget
case with honest hints and the honest outputs -- all must be accepted, or the adapter is rejecting for structural
reasons.  ``--extras`` adds this lane's own forgeries (the leading-bit lower-bound hole).

    uv run --package verity-numerical python backends/direct/tools/battery_adapter.py --bin $BIN [--out report.json]
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from verity_numerical.checker import REAL
from verity_numerical.checker.ctx import IntCtx
from verity_numerical.checker.step import check_step_expr
from verity_numerical.redteam.babybear import gen_step_bb
from verity_numerical.redteam.battery_babybear import (
    Case,
    _lax_output,
    gen_step_from,
    load_battery,
    run_battery,
)

STEPS = 96
K = 16

#: Python hint tree path -> B's column name (per unit).  Hints B has no column for (``acc.f``, ``pack.low``: the
#: unlimbed words; the epilogue's ``hi``/``lo``: the public limbs) are dropped -- B reads the limbs directly.
GADGET_TAG = {
    "fp32_decode": "acc",
    "epilogue": "cast",
    "align_state": "align_state",
    "align_product": "align_prod",
    "leading_bit": "lead",
    "normalise": "norm",
}
DROP = {("acc", "f"), ("pack", "low"), ("cast", "hi"), ("cast", "lo")}


def unit_overrides(h: dict) -> dict[str, int]:
    """One unit's hint tree (``gen_step`` layout + limb hints) as B's ``name -> value`` overrides."""
    o: dict[str, int] = {}

    def put(tag: str, d: dict) -> None:
        for k, v in d.items():
            if isinstance(v, int) and (tag, k) not in DROP:
                o[f"{tag}.{k}"] = v

    put("acc", h["acc"])
    for i in range(K):
        put(f"a[{i}]", h["a"][i])
        put(f"b[{i}]", h["b"][i])
        put(f"prod[{i}]", h["prod"][i])
    for g, hg in enumerate(h["groups"]):
        t = f"g{g}"
        for k, v in hg["max"].items():
            if k == "sel":
                for i, s in enumerate(v):
                    o[f"{t}.max.sel[{i}]"] = s
            elif isinstance(v, int):
                o[f"{t}.max.{k}"] = v
        put(f"{t}.align_state", hg["acc_align"])
        for i, hq in enumerate(hg["prod_align"]):
            put(f"{t}.align_prod[{i}]", hq)
        put(f"{t}.sum", hg["sum"])
        put(f"{t}.lead", hg["lead"])
        put(f"{t}.norm", hg["norm"])
        put(f"{t}.out", hg["out"])
    put("pack", h["pack"])
    return o


def epilogue_overrides(h: dict) -> dict[str, int]:
    return {f"cast.{k}": v for k, v in h.items() if isinstance(v, int) and ("cast", k) not in DROP}


def zero_row(step: int, y16: int) -> dict:
    return {"c": None, "a": [0] * K, "b": [0] * K, "step": step, "y16": y16, "overrides": {}}


def step_job(cid: str, x: dict, h: dict | None) -> dict:
    """The unit at position 1 of a VU, then an honest zero unit consuming its output limbs."""
    row = {"c": x["c"], "a": list(x["a"]), "b": list(x["b"]), "step": 1, "y16": 0, "overrides": unit_overrides(h) if h else {}}
    return {"id": cid, "kind": "rows", "starts_trace": False, "rows": [row, zero_row(2, 0)]}


def vu_job(cid: str, x: dict, h: dict | None, y16: int) -> dict:
    n_units = len(x["a"]) // K
    assert n_units <= STEPS
    rows = []
    for s in range(STEPS):
        if s < n_units:
            hs = h["steps"][s] if h else None
            rows.append(
                {
                    "c": 0 if s == 0 else None,
                    "a": list(x["a"][K * s : K * (s + 1)]),
                    "b": list(x["b"][K * s : K * (s + 1)]),
                    "step": s,
                    "y16": y16,
                    "overrides": unit_overrides(hs) if hs else {},
                }
            )
        else:
            rows.append(zero_row(s, y16))
    if h:
        rows[-1]["overrides"].update(epilogue_overrides(h["epilogue"]))
    return {"id": cid, "kind": "rows", "starts_trace": True, "rows": rows}


def gadget_job(cid: str, gadget: str, inputs: dict, hints: dict | None, claimed: list[int]) -> dict:
    o: dict[str, int] = {}
    for k, v in (hints or {}).items():
        if gadget == "lead_norm":
            tag = "lead" if k == "l" else "norm"
        else:
            tag = GADGET_TAG[gadget]
        if (tag, k) in DROP:
            continue
        o[f"{tag}.{k}"] = v
    return {"id": cid, "kind": "gadget", "name": gadget, "inputs": inputs, "hints": o, "claimed": claimed}


def jobs_for_case(c: Case) -> tuple[dict, dict]:
    """``(the forgery, its honest control)``."""
    if c.level == "step":
        return step_job(c.id, c.inputs, c.hints), step_job(f"ctl:{c.id}", c.inputs, None)
    if c.level == "vu":
        return vu_job(c.id, c.inputs, c.hints, c.claimed["y16"]), vu_job(f"ctl:{c.id}", c.inputs, None, c.honest["y16"])
    return (
        gadget_job(c.id, c.gadget, c.inputs, c.hints, c.claimed["outputs"]),
        gadget_job(f"ctl:{c.id}", c.gadget, c.inputs, None, c.honest["outputs"]),
    )


def extra_cases(p=REAL) -> list[Case]:
    """This lane's forgeries of the leading-bit window's missing LOWER bound (LT_DIRECT's second range).

    ``l`` one larger than the bit length passes ``span - 1 - (v - lo) in [0, 2^30)`` (the only check the port
    shipped) and the group's normalisation then halves ``M`` and raises ``e1`` by one: a different output word."""
    out: list[Case] = []
    # gadget level: v = 16, l = 29 (honest l = 5): M = v >> 5 = 0
    out.append(
        Case(
            id="extra-lead-l29",
            cls="extra",
            level="gadget",
            params="REAL",
            inputs={"v": 16},
            hints={"l": 29},
            claimed={"outputs": [29]},
            honest={"outputs": [5]},
            mutation="leading_bit: v = 16, l = 29 (v < lo = 2^28): span - 1 - (v - lo) = 2^29 - 17 passes the direct bound",
            gadget="leading_bit",
            source="b-lookup",
        )
    )
    out.append(
        Case(
            id="extra-lead-norm-l29",
            cls="extra",
            level="gadget",
            params="REAL",
            inputs={"E": 0, "v": 16},
            hints={"l": 29, "clamp": 0, "M": 0, "r": 16},
            claimed={"outputs": [4, 0]},
            honest={"outputs": [-20, 8388608]},
            mutation="lead_norm: v = 16, l = 29: k = -5, div = 32, M = 0, r = 16; the group output is +0",
            gadget="lead_norm",
            source="b-lookup",
        )
    )
    # unit level: an honest instance with g0.lead.l raised by one, everything downstream re-derived honestly
    from verity_numerical.redteam.battery_babybear import honest_instances

    for name, x in honest_instances(p, limit=6):
        h0 = gen_step_bb(p, x["c"], x["a"], x["b"])
        y0 = int(check_step_expr(IntCtx(), p, x["c"], list(x["a"]), list(x["b"]), h0))
        for g in range(len(p.groups)):
            l0 = h0["groups"][g]["lead"]["l"]
            if l0 == 0 or l0 >= p.l_hi:
                continue
            h = gen_step_from(p, x["c"], x["a"], x["b"], {f"groups[{g}].lead.l": l0 + 1})
            y = _lax_output(p, "step", x, h)
            if y == y0:
                continue
            out.append(
                Case(
                    id=f"extra-step-l+1-{name}-g{g}",
                    cls="extra",
                    level="step",
                    params="REAL",
                    inputs=copy.deepcopy(x),
                    hints=h,
                    claimed={"y": y},
                    honest={"y": y0},
                    mutation=f"groups[{g}].lead.l: {l0} -> {l0 + 1}, downstream hints re-derived; word {y0:#010x} -> {y:#010x}",
                    source="b-lookup",
                )
            )
            break
        if len(out) >= 6:
            break
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", required=True, help="path to the direct-unit binary (release build)")
    ap.add_argument("--field", default="babybear")
    ap.add_argument("--out", type=Path, default=None, help="JSON report")
    ap.add_argument("--extras", action="store_true", help="add this lane's leading-bit lower-bound forgeries")
    ap.add_argument("--keep", type=Path, default=None, help="keep the jobs JSON here")
    args = ap.parse_args(argv)

    cases = load_battery()
    extras = extra_cases() if args.extras else []
    jobs, controls = [], []
    for c in cases + extras:
        j, ctl = jobs_for_case(c)
        jobs.append(j)
        # the -V_PREMISE z3 models have an input OUTSIDE the gadget's premise (v = p - 2^30 + 1 > S_MAX): the
        # honest control would be rejected on the input range itself, so there is no honest witness to control with
        if c.dropped_rule != "V_PREMISE":
            controls.append(ctl)
    payload = {"field": args.field, "jobs": jobs + controls}
    with tempfile.TemporaryDirectory() as td:
        jp = Path(td) / "jobs.json"
        rp = Path(td) / "report.json"
        jp.write_text(json.dumps(payload))
        if args.keep:
            args.keep.write_text(json.dumps(payload))
        t0 = time.time()
        subprocess.run([args.bin, "check-rows", "--field", args.field, "--in", str(jp), "--out", str(rp)], check=True)
        t_check = time.time() - t0
        rep = json.loads(rp.read_text())
    res = {r["id"]: r for r in rep["results"]}

    report = run_battery(lambda c: res[c.id]["accepted"], cases)
    print(report.render())
    ctl_failed = [cid for cid in res if cid.startswith("ctl:") and not res[cid]["accepted"]]
    print(f"controls: {len(controls) - len(ctl_failed)}/{len(controls)} honest witnesses accepted")
    for cid in ctl_failed[:10]:
        print(f"  CONTROL REJECTED {cid}: {res[cid]['first_failing']}")
    extra_accepted = [c.id for c in extras if res[c.id]["accepted"]]
    if extras:
        print(f"extras: {len(extras) - len(extra_accepted)}/{len(extras)} rejected")
        for c in extras:
            r = res[c.id]
            print(f"  {'ACCEPTED' if r['accepted'] else 'rejected'} {c.id}: {r.get('first_failing')} | {c.mutation[:110]}")
    # per case: what rejected it
    detail = []
    for c in cases + extras:
        r = res[c.id]
        detail.append(
            {
                "id": c.id,
                "class": c.cls,
                "level": c.level,
                "gadget": c.gadget,
                "dropped_rule": c.dropped_rule,
                "accepted": r["accepted"],
                "first_failing": r.get("first_failing"),
                "n_failing": r["n_failing"],
                "python_embedded": c.verdicts.get("embedded"),
                "control_accepted": res[f"ctl:{c.id}"]["accepted"] if f"ctl:{c.id}" in res else None,
            }
        )
    out = {
        "schema": "verity-direct/battery/v1",
        "field": args.field,
        "air": {"width": rep["width"], "constraints": rep["constraints"]},
        "battery": {"version": "v1", "total": report.total, "accepted": report.accepted, "by_class": report.by_class, "ok": report.ok},
        "controls": {"total": len(controls), "rejected": ctl_failed},
        "extras": {"total": len(extras), "accepted": extra_accepted},
        "seconds_check_rows": t_check,
        "cases": detail,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1) + "\n")
        print(f"wrote {args.out}")
    ok = report.ok and not ctl_failed and not extra_accepted
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
