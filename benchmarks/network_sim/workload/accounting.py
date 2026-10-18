"""The workload in the accounting layer's terms (`benchmarks/datacenter`, `import datacenter`), so that its `run`, `audit`
and `account` read the MLP as they read a traced circuit.

- `circuit(model)`: one word gate of width 1 for each of the Program's Boolean gates, in canonical order, so value v of
  the Program is value v of the circuit. `And2`, `Xor2` and `Not` are `and`, `xor` and `not`, a constant is a `const`,
  and an input bit is an `input` named `<parameter>.<bit>` (`label`), from the outside for the requests and registered
  for the weights.
- `datacenter(hw, clock, rate)`: a block for each unit; a compute leaf is a `gpu` with a declared scalar rate, a
  certifier a trusted `certifier`, the client `outside`, and any other leaf a `switch`, which only carries frames. A
  link's capacity is its β a tick over the clock's tick.
- `placement(circuit, hw, p, dc)`: the same leaf, step and ticks as `placement.Placement`, each device's values in
  canonical order as its cells, and the outputs delivered to the client.

Both layers derive the same crossings from one placement (`crossing_pairs`), but count their frames differently: the
accounting layer gives each crossing value its own ⌈bits / frame_bits⌉ frames, one for a Boolean value, in the bucket of
the value's tick, and keys a link by its ends, where φ packs a message's values into B-byte frames, counts them in the
bucket of the read tick, and keys each direction of a link.

    python -m workload.accounting                  # audit and account an honest run of the MLP on the mock
    python -m workload.accounting --extra-frame    # one covert frame on b0's uplink: refused
"""
from __future__ import annotations

import argparse
from collections import Counter

import datacenter as D
from datacenter.logical import Gate

from . import model as M
from . import placement as P
from .block import Context
from .hardware import OUTSIDE, Hardware

OPS = {"And2": "and", "Xor2": "xor", "Not": "not"}


def label(model: M.Model) -> dict[int, str]:
    """Each input bit's name: its root parameter's name and its index among the parameter's bits."""
    return {g: f"{name}.{j}" for pi, name in enumerate(model.params) for j, g in enumerate(model.input_gates(pi))}


def circuit(model: M.Model) -> D.Circuit:
    names = label(model)
    c = D.Circuit(names=dict(names))
    for v, g in enumerate(model.gates):
        kind = model.kind[v]
        if kind in ("run", "registered"):
            source = "outside" if kind == "run" else "registered"
            c.gates.append(Gate("input", (), 1, input=D.Input(1, source, names[v])))
        elif kind == "constant":
            c.gates.append(Gate("const", (), 1, const=model.evaluate({}, [v])[v]))
        else:
            op = OPS.get(g.prim.id.split("_")[0])
            if op is None:
                raise ValueError(f"gate {v} is a {g.prim.id}, which has no word gate")
            c.gates.append(Gate(op, tuple(g.operands), 1))
    c.outputs = tuple(model.outputs)
    return c


def inputs(model: M.Model, x: list[list[int]], weights) -> dict[str, int]:
    names = label(model)
    return {names[g]: bit for g, bit in model.bind_inputs(x, weights).items()}


def datacenter(hw: Hardware, clock: D.Clock, rate: float = 1e9) -> D.Datacenter:
    def device(x: int) -> D.Device:
        name = hw.leaves[x]
        if x in hw.cert:
            return D.Device(name, "certifier", trusted=True)
        if x == hw.client:
            return D.Device(name, "outside")
        u = hw.unit(x)
        if u != OUTSIDE and x in hw.devices(u):
            return D.Device(name, "gpu", rate={"scalar": D.Assumption(f"{name}.rate", rate, "ops/s", "declared")})
        return D.Device(name, "switch")

    return D.Datacenter(
        "netsim",
        [D.Block(u, [device(x) for x in sorted(hw.units[u])]) for u in sorted(hw.units)],
        [device(x) for x in range(len(hw.leaves)) if hw.unit(x) == OUTSIDE],
        [D.Link(hw.leaves[k.ends[0]], hw.leaves[k.ends[1]],
                D.Assumption(f"{k.name}.beta", k.beta / clock.tick, "bit/s", "declared")) for k in hw.links])


def placement(c: D.Circuit, hw: Hardware, p: P.Placement, dc: D.Datacenter) -> D.Placement:
    cells: Counter = Counter()
    cell = {}
    for v, key in enumerate(zip(p.leaf, p.step)):
        cell[v] = cells[key]
        cells[key] += 1
    devices = sorted(set(p.leaf) | {hw.client})
    return D.Placement(c, dc, {v: hw.leaves[x] for v, x in enumerate(p.leaf)}, dict(enumerate(p.step)),
                       {hw.leaves[x]: list(p.ticks[x]) for x in devices}, cell, deliver=hw.leaves[hw.client])


def crossing_pairs(dp: D.Placement) -> set[tuple[int, str]]:
    """The accounting layer's crossings as (value, reading unit), the pairs `placement.crossings` makes edges of."""
    return {(x.value.id, getattr(dp.datacenter.block_of(x.dst), "name", OUTSIDE)) for x in D.crossings(dp)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Audit and account a run of the MLP with the accounting layer.")
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--devices", type=int, default=2)
    ap.add_argument("--d", type=int, default=4)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--requests", type=int, default=3)
    ap.add_argument("--groups", type=int, default=1)
    ap.add_argument("--plan", choices=sorted(P.PLANS), default="pipeline")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--frame-bits", type=int, default=32)
    ap.add_argument("--window", type=int, default=4, help="buckets a certifier's record covers")
    ap.add_argument("--syncs", type=int, default=128, help="the sync set's size, a window's timing advice")
    ap.add_argument("--tick", type=float, default=1e-3, help="seconds a tick")
    ap.add_argument("--rate", type=float, default=1e9, help="each GPU's declared scalar ops a second")
    ap.add_argument("--extra-frame", action="store_true", help="add one covert frame on b0's uplink")
    args = ap.parse_args(argv)

    ctx = Context({"spec": {"d": args.d, "layers": args.layers, "requests": args.requests, "groups": args.groups},
                   "blocks": args.blocks, "devices": args.devices, "plan": args.plan, "frame_bytes": 64,
                   "seed": args.seed})
    clock = D.Clock(args.tick)
    c = circuit(ctx.model)
    dc = datacenter(ctx.hw, clock, args.rate)
    dp = placement(c, ctx.hw, ctx.placement, dc)
    mine = {(e.value, e.dst) for e in ctx.edges}
    print(f"{ctx.model.n} gates as a datacenter.Circuit of {c.boolean_size} Boolean gates; datacenter problems: "
          f"{dc.validate() or 'none'}; paced: {dp.paced()}")
    print(f"crossings: {len(D.crossings(dp))} in the accounting layer, {len(mine)} in placement.crossings, "
          f"{'the same' if crossing_pairs(dp) == mine else 'different'} (value, unit) pairs")
    bucket = ctx.hw.bucket_ticks
    x, ws = M.requests(ctx.spec, ctx.seed), M.weights(ctx.spec, ctx.seed)
    deviation = None
    if args.extra_frame:
        b0 = ctx.hw.leaves[ctx.hw.devices("b0")[0]]
        deviation = D.Deviation(extra={((b0, "b0.cert"), ctx.placement.tick(ctx.edges[0].value) // bucket): 1})
    trace = D.run(dp, inputs(ctx.model, x, ws), frame_bits=args.frame_bits, bucket=bucket, window=args.window,
                  clock=clock, deviation=deviation)
    answers = ctx.model.output_words(trace.values)
    print(f"answers {'equal' if answers == M.reference(ctx.spec, x, ws) else 'differ from'} the integer reference: "
          f"{answers}")
    profile = D.occupancy(dp, args.frame_bits, bucket, args.window)
    for link, row in sorted(profile.items()):
        print(f"  occupancy {link[0]}-{link[1]:<10} {[row.get(b, 0) for b in range(max(row) + 1)]}")
    claims = D.audit(trace, profile, syncs=args.syncs)
    kinds = Counter(type(k).__name__ for k in claims)
    print("claims: " + ", ".join(f"{n} {k}" for k, n in sorted(kinds.items())))
    horizon = max(t for ts in dp.ticks.values() for t in ts) + 1
    acc = D.account(claims, dp, clock, D.Interval(0, horizon))
    print(acc)
    return 0 if acc.sound else 1


if __name__ == "__main__":
    raise SystemExit(main())
