"""machine-model cover [--card sm_NN] [--holes|--check]
                                          the coverage table (data/cover/): each opcode of NVIDIA's family tables and
                                          the SASS corpus on the H200 and the RTX PRO 6000, its envelope op, units,
                                          rate, mark, argument and source; the counts by mark (opcodes, corpus forms,
                                          corpus instructions) and every hole with its severity; --holes prints the
                                          counts and holes only; --check prints what fails the table and exits 1 if
                                          anything does"""
from __future__ import annotations

from verity_machine_model import cover as C
from verity_machine_model import theta as T
from verity_machine_model.model import device


def add(sub) -> None:
    c = sub.add_parser("cover")
    c.add_argument("--card", choices=C.DEVICES, help="one card (default: both)")
    g = c.add_mutually_exclusive_group()
    g.add_argument("--holes", action="store_true", help="the counts and the holes only")
    g.add_argument("--check", action="store_true", help="the table's problems only; exit 1 if any")


def _cell(s) -> str:
    return str(s or "").replace("|", "\\|").replace("\n", " ")


def _rate(c: C.Cover, r: C.Row, dev: str, theta: T.Theta) -> str:
    if r.envelope in ("T", "T.sp") and r.mark in ("covered", "coarse"):
        x = C.rate(c, r, dev, theta)
        w = c.envelope["device"][dev]["weight"][r.get("fmt")]
        rho = c.envelope["device"][dev]["rho"]["tensor"]
        kept = " / 2 kept" if r.envelope == "T.sp" else ""
        return f"{x:g}{kept} × {w:g} = {C.tensor_units(c, r, dev, theta):g} ≤ ρ_T {rho:g}"
    k = r.get("rate", dev)
    if k is None:
        return ""
    x = theta.rates[k]
    vals = ", ".join(f"{n} {v:g}" for n, v in (("sheet", x.sheet), ("measured", x.measured)) if v is not None)
    return f"θ {k} ({vals} {x.unit})" if vals else f"θ {k} (no value: {x.gap or x.source})"


def _counts(c: C.Cover, dev: str) -> str:
    k = C.counts(c, dev)
    ops = ", ".join(f"{k.opcodes[m]} {m}" for m in (*C.MARKS, "missing") if k.opcodes[m])
    forms = ", ".join(f"{k.forms[m]} {m}" for m in (*C.MARKS, "missing") if k.forms[m])
    total = sum(k.instructions.values())
    insts = ", ".join(f"{100 * k.instructions[m] / total:.4g}% {m}" for m in (*C.MARKS, "missing")
                      if k.instructions[m])
    return (f"{dev}: {sum(k.opcodes.values())} opcodes: {ops}; {sum(k.forms.values())} corpus forms: {forms}; "
            f"{total:,} corpus instructions: {insts}")


def run(args) -> int:
    c = C.load()
    ps = C.problems(c)
    if args.check:
        print("\n".join(ps) if ps else f"cover: every opcode, corpus form and census kind of {', '.join(C.DEVICES)} "
                                       f"has a row, and every tensor row's rate × weight is at most ρ_T")
        return 1 if ps else 0
    devs = [args.card] if args.card else list(C.DEVICES)
    ops = ", ".join(o["key"] for o in c.envelope["op"] if o["key"] != "-")
    print(f"Coverage of the envelope ISA ({ops}; data/cover/envelope.toml, provisional: MODEL-MIN owns the list)")
    for dev in devs:
        print(_counts(c, dev))
    print("\nHoles (each a gap in the coverage premise):")
    for devices, what, sev, why in C.holes(c):
        if any(d in devices.split("/") for d in devs):
            print(f"- [{sev}] {devices} {what}: {why}")
    if ps:
        print("\nProblems (machine-model cover --check):\n" + "\n".join(f"- {p}" for p in ps))
    if args.holes:
        return 1 if ps else 0
    for dev in devs:
        theta = T.theta(device(dev))
        print(f"\n## {dev}\n")
        print("| Opcode | Forms | Mark | Envelope | Units | Rate a SM-clock | Argument | Source |")
        print("|---|---|---|---|---|---|---|---|")
        for r in c.rows:
            if dev not in r.devices:
                continue
            listed = [op for op in r.ops if op in c.opcodes[dev]] or list(r.ops)
            for op in listed:
                forms = ", ".join(f"`{f}`" for f in r.forms) or _cell(r.raw.get("what"))
                if r.forms and r.raw.get("what"):
                    forms += f" ({_cell(r.raw['what'])})"
                mark = r.mark + (f" ({r.raw['severity']})" if r.mark == "hole" else "")
                print(f"| {op} | {forms} | {mark} | {r.envelope} | {_cell(r.raw.get('cost'))} | "
                      f"{_cell(_rate(c, r, dev, theta))} | {_cell(r.raw.get('argument'))} | "
                      f"{_cell(r.raw.get('source'))} |")
        print(f"\n### {dev}: units outside the SM pipes\n")
        print("| Unit | What | Mark | Envelope | Argument |")
        print("|---|---|---|---|---|")
        for u in c.units:
            if dev in u.get("devices", C.DEVICES):
                mark = u["mark"] + (f" ({u['severity']})" if u["mark"] == "hole" else "")
                print(f"| {u['key']} | {_cell(u['what'])} | {mark} | {u['envelope']} | {_cell(u['argument'])} |")
    return 1 if ps else 0
