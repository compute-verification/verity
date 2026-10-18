"""The workload in the accounting layer's terms (`benchmarks/datacenter`, `import datacenter`), so that its `run`, `audit`
and `account` read the MLP as they read a traced circuit.

- `circuit(model)`: one word gate of width 1 for each of the Program's Boolean gates, in canonical order, so value v of
  the Program is value v of the circuit. `And2`, `Xor2` and `Not` are `and`, `xor` and `not`, a constant is a `const`,
  and an input bit is an `input` named `<parameter>.<bit>` (`label`), from the outside for the requests and registered
  for the weights.
- `datacenter(hw, grid, rate)`: a block for each unit; a compute leaf is a `gpu` with a declared rate in Boolean gates,
  a certifier a `certifier` (trusted by its kind), the client `outside`, and any other leaf a `switch`, which only
  carries frames. A link's capacity is its β a tick over the grid's tick.
- `placement(circuit, hw, p, dc)`: the same leaf, step and ticks as `placement.Placement`, each device's values in
  canonical order as its cells, and the outputs delivered to the client.

Both layers derive the same crossings from one placement (`crossing_pairs`) and count a crossing's frames in the bucket
of its read tick, but frame them differently: the accounting layer packs the values crossing a certifier port in a
bucket into `frame_bits`-bit frames, where φ packs each message's values into B-byte frames with headers and a `HELLO`
per connection.

    python -m workload.accounting                  # audit and account an honest run of the MLP on the mock
    python -m workload.accounting --extra-frame    # one covert frame on b0's uplink: refused
    python -m workload.accounting --assume fleet-closed   # attribute the credited work to its blocks under the premise
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
PREMISES = {D.FLEET_CLOSED.name: D.FLEET_CLOSED}


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
            c.gates.append(Gate("input", (), 1, input=D.Input(1, source)))
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


def datacenter(hw: Hardware, grid: D.Grid, rate: float = 1e9) -> D.Datacenter:
    def device(x: int) -> D.Device:
        name = hw.leaves[x]
        if x in hw.cert:
            return D.Device(name, "certifier")
        if x == hw.client:
            return D.Device(name, "outside")
        u = hw.unit(x)
        if u != OUTSIDE and x in hw.devices(u):
            return D.Device(name, "gpu", rate={"gate": D.Assumption(f"{name}.rate", rate, "gate/s", "declared")})
        return D.Device(name, "switch")

    return D.Datacenter(
        "netsim",
        [D.Block(u, [device(x) for x in sorted(hw.units[u])]) for u in sorted(hw.units)],
        [device(x) for x in range(len(hw.leaves)) if hw.unit(x) == OUTSIDE],
        [D.Link(hw.leaves[k.ends[0]], hw.leaves[k.ends[1]],
                D.Assumption(f"{k.name}.beta", k.beta / float(grid.tick), "bit/s", "declared")) for k in hw.links],
        grid)


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


def epoch(dp: D.Placement) -> D.Interval:
    """The ticks the placement uses, from the first through the last."""
    return D.Interval(0, max(t for ts in dp.ticks.values() for t in ts) + 1)


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
    ap.add_argument("--rate", type=float, default=1e9, help="each GPU's declared Boolean gates a second")
    ap.add_argument("--extra-frame", action="store_true", help="add one covert frame on b0's uplink")
    ap.add_argument("--assume", action="append", default=[], choices=sorted(PREMISES),
                    help="a named premise for `account`")
    args = ap.parse_args(argv)

    ctx = Context({"spec": {"d": args.d, "layers": args.layers, "requests": args.requests, "groups": args.groups},
                   "blocks": args.blocks, "devices": args.devices, "plan": args.plan, "frame_bytes": 64,
                   "seed": args.seed})
    grid = D.Grid(args.frame_bits, ctx.hw.bucket_ticks, args.window, args.syncs,
                  D.Assumption("tick", args.tick, "s", "declared"))
    c = circuit(ctx.model)
    dc = datacenter(ctx.hw, grid, args.rate)
    dp = placement(c, ctx.hw, ctx.placement, dc)
    mine = {(e.value, e.dst) for e in ctx.edges}
    print(f"{ctx.model.n} gates as a datacenter.Circuit of {c.boolean_size} Boolean gates; datacenter problems: "
          f"{dc.validate() or 'none'}; paced: {dp.paced()}")
    print(f"crossings: {len(D.crossings(dp))} in the accounting layer, {len(mine)} in placement.crossings, "
          f"{'the same' if crossing_pairs(dp) == mine else 'different'} (value, unit) pairs")
    x, ws = M.requests(ctx.spec, ctx.seed), M.weights(ctx.spec, ctx.seed)
    deviation = None
    if args.extra_frame:
        deviation = D.Deviation(extra={(D.Port("b0.cert", "egress"), ctx.hw.bucket(ctx.placement.tick(ctx.edges[0].value))): 1})
    plan = D.Plan(epoch(dp))
    profile = dp.public(plan)
    trace = D.run(dp, inputs(ctx.model, x, ws), plan, deviation)
    answers = ctx.model.output_words(trace.values)
    print(f"answers {'equal' if answers == M.reference(ctx.spec, x, ws) else 'differ from'} the integer reference: "
          f"{answers}")
    for port, row in sorted(profile.occupancy.items(), key=lambda kv: repr(kv[0])):
        print(f"  occupancy {port!r:<12} {[row.get(b, 0) for b in range(max(row) + 1)]}")
    claims = D.audit(trace, profile)
    kinds = Counter(type(k).__name__ for k in claims)
    print("claims: " + ", ".join(f"{n} {k}" for k, n in sorted(kinds.items())))
    acc = D.account(claims, profile, assume=tuple(PREMISES[a] for a in args.assume))
    print(acc)
    return 0 if acc.sound else 1


if __name__ == "__main__":
    raise SystemExit(main())
