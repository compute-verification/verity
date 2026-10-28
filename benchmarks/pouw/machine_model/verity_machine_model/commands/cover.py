"""machine-model cover [--card sm_NN] [--holes|--check]
                                          the coverage table (data/cover/): each opcode of NVIDIA's family tables and
                                          the SASS corpus on the H200 and the RTX PRO 6000, its envelope op (Accel.Op,
                                          or a proposed one) and pipe, units, rate against ρ, mark, argument
                                          and source; the counts by mark (opcodes, corpus forms, corpus instructions),
                                          with the proposed entries and on Accel.Op alone, the rows resting on proposed
                                          entries, and every hole with its severity; --holes prints the counts and holes
                                          only; --check prints what fails the table and exits 1 if anything does"""
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
    if r.envelope not in C.RATED or r.mark not in C.DOMINATED:
        k = r.get("rate", dev)
        return ", ".join(f"θ {x}" for x in C.as_list(k))
    out = []
    for d in C.demands(c, r, dev, theta):
        if d.problem:
            out.append(f"{d.rate}: {d.problem}")
        else:
            out.append(f"{d.rate} → {d.amount:.4g} ≤ ρ {d.resource} {d.rho:.4g}")
    return "; ".join(out)


def _envelope(c: C.Cover, r: C.Row) -> str:
    e = r.envelope + (f" on {r.get('pipe')}" if r.get("pipe") else "")
    fmt = C.as_list(r.get("fmt"))
    e += f" ({', '.join(fmt)})" if fmt else ""
    rests = C.rests_on(c, r)
    return e + (f"; rests on proposed {', '.join(rests)}" if rests else "")


def _counts(c: C.Cover, dev: str, proposed: bool) -> str:
    k = C.counts(c, dev, proposed)
    ops = ", ".join(f"{k.opcodes[m]} {m}" for m in (*C.MARKS, "missing") if k.opcodes[m])
    forms = ", ".join(f"{k.forms[m]} {m}" for m in (*C.MARKS, "missing") if k.forms[m])
    total = sum(k.instructions.values())
    insts = ", ".join(f"{100 * k.instructions[m] / total:.4g}% {m}" for m in (*C.MARKS, "missing")
                      if k.instructions[m])
    return (f"{sum(k.opcodes.values())} opcodes: {ops}; {sum(k.forms.values())} corpus forms: {forms}; "
            f"{total:,} corpus instructions: {insts}")


def run(args) -> int:
    c = C.load()
    ps = C.problems(c)
    if args.check:
        print("\n".join(ps) if ps else f"cover: every opcode, corpus form and census kind of {', '.join(C.DEVICES)} "
                                       f"has a row, and every dominated row's rate is at most its resource's ρ")
        return 1 if ps else 0
    devs = [args.card] if args.card else list(C.DEVICES)
    ops = c.table("op")
    accel = ", ".join(k for k, o in ops.items() if k != "-" and "proposed" not in o)
    new = [*(f"op {k}" for k, o in ops.items() if "proposed" in o),
           *(f"pipe {k}" for k, o in c.table("pipe").items() if "proposed" in o),
           *(f"fmt {k}" for k, o in c.table("fmt").items() if "proposed" in o)]
    print(f"Coverage of the envelope ISA: {c.envelope['source']}: {accel}; proposed: {', '.join(new)} "
          f"(data/cover/envelope.toml)")
    for dev in devs:
        print(f"{dev}: {_counts(c, dev, True)}")
        print(f"{dev} on Accel.Op alone (a row resting on a proposed entry a hole): {_counts(c, dev, False)}")
    print("\nRows resting on a proposed entry:")
    for r in c.rows:
        if r.mark in C.DOMINATED and (rests := C.rests_on(c, r)) and any(d in r.devices for d in devs):
            forms = f" [{', '.join(r.forms)}]" if r.forms else ""
            print(f"- {'/'.join(r.devices)} {', '.join(r.ops)}{forms}: {', '.join(rests)}")
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
        print("| Opcode | Forms | Mark | Envelope | Units | Rate a SM-clock, in the op's units, against ρ | Argument "
              "| Source |")
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
                print(f"| {op} | {forms} | {mark} | {_cell(_envelope(c, r))} | {_cell(r.raw.get('cost'))} | "
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
