"""The workload in the accounting layer's terms (`benchmarks/datacenter`, `import datacenter`): its `run`, `audit` and
`account` read the MLP's Program and verity_vllm's serving Program themselves, as they read any `verity.ml.circuits`
Program.

- `parameters(model, seed)`: each parameter's words, the requests' (run) and the weights' (registered), in leaf order.
- `datacenter(hw, cadence, kinds, rate)`: a block for each unit; a compute leaf is a `gpu` with a declared rate for each
  kind of work the Program's gates are (`datacenter.work`), a certifier a `certifier` (trusted by its kind), the client
  `outside`, and any other leaf a `switch`, which only carries frames. A link's capacity is its β a tick over the
  cadence's tick.
- `placement(model, hw, p, dc, registration)`: each Call on the device and at the step `placement.StepPlacement` gives
  its gates, the run inputs at the client, and the outputs delivered to the client. The simulator stores a registered
  value once, at its first reader, and moves it to readers in other units (Serve_v1's tied embeddings), so a value read
  in more than one unit is `held` there; every other registered value has its copy where it is read.

Both layers derive the same crossings from one placement: the accounting layer's are census values (`crossing_pairs`
expands each to its gates), and each counts a crossing's frames in the bucket of its read tick, but they frame them
differently: the accounting layer packs the values crossing a certifier port in a bucket into `frame_bits`-bit frames,
where φ packs each message's values into B-byte frames with headers and a `HELLO` per connection.

    python -m workload.accounting                  # audit and account an honest run of the MLP on the mock
    python -m workload.accounting --extra-frame    # one covert frame on b0's uplink: refused
    python -m workload.accounting --assume fleet-closed   # attribute the credited work to its blocks under the premise
    python -m workload.accounting --workload served --devices 1   # verity_vllm's serving Program, a stage a block
"""
from __future__ import annotations

import argparse
from collections import Counter

import datacenter as D
from datacenter.logical import binding, gates, layout
from verity.ml.circuits.program import REGISTRATION, RUN

from . import model as M
from . import placement as P
from .block import Context
from .hardware import OUTSIDE, DeviceGraph

PREMISES = {D.FLEET_CLOSED.name: D.FLEET_CLOSED}


def parameters(model, seed: int) -> dict[str, list[int]]:
    """Each parameter's words, one per element (`datacenter.logical.layout`: a Boolean element's bits, or a word leaf)."""
    t, out = model.inputs(seed), {}
    for pi, ((name, pt), _) in enumerate(zip(model.program.fn.signature()[0], model.params)):
        leaves, words = iter(t[g] for g in model.input_gates(pi)), []
        for widths in layout(pt):
            word, at = 0, 0
            for w in widths:
                word |= next(leaves) << at
                at += w
            words.append(word)
        out[name] = words
    return out


def split(model, seed: int) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """The run inputs' words and the registered ones'."""
    words = parameters(model, seed)
    when = dict(zip(model.params, model.program.bindings))
    return ({n: w for n, w in words.items() if when[n] == RUN}, {n: w for n, w in words.items() if when[n] == REGISTRATION})


def kinds(model) -> set[str]:
    return {k for w in D.work(model.program).values() for k in w}


def datacenter(hw: DeviceGraph, cadence: D.Cadence, kinds: set[str], rate: float = 1e9) -> D.Datacenter:
    def device(x: int) -> D.DeviceEntry:
        name = hw.leaves[x]
        if x in hw.cert:
            return D.DeviceEntry(name, "certifier")
        if x == hw.client:
            return D.DeviceEntry(name, "outside")
        u = hw.unit(x)
        if u != OUTSIDE and x in hw.devices(u):
            return D.DeviceEntry(name, "gpu", rate={k: D.Assumption(f"{name}.{k}", rate, f"{k}/s", "declared") for k in kinds})
        return D.DeviceEntry(name, "switch")

    return D.Datacenter(
        "netsim",
        [D.Zone(u, [device(x) for x in sorted(hw.units[u])]) for u in sorted(hw.units)],
        [device(x) for x in range(len(hw.leaves)) if hw.unit(x) == OUTSIDE],
        [D.LinkEntry(hw.leaves[k.ends[0]], hw.leaves[k.ends[1]],
                     D.Assumption(f"{k.name}.beta", k.beta / float(cadence.tick), "bit/s", "declared"))
         for k in hw.links],
        cadence)


def placement(model, hw: DeviceGraph, p: P.StepPlacement, dc: D.Datacenter, registration: D.Registration) -> D.Placed:
    program = model.program
    body = program.root.body
    client = hw.leaves[hw.client]
    device, step = {}, {}
    for k in range(len(body.nodes)):
        when = binding(program, k)
        if when == REGISTRATION:
            continue
        lo = body.offsets[k]
        device[k], step[k] = (client, 0) if when == RUN else (hw.leaves[p.leaf[lo]], p.step[lo])
    held = {}
    for v in D.values(program):
        units = {hw.unit(p.leaf[body.offsets[r]]) for r in v.reads}
        if binding(program, v.node) == REGISTRATION and len(units) > 1:
            held[v.id] = device[min(v.reads)]
    used = {hw.leaves.index(d) for d in device.values()} | {p.leaf[g] for g in range(model.n)}
    return D.Placed(program, dc, device, step, {hw.leaves[x]: list(p.ticks[x]) for x in used}, deliver=client,
                    registration=registration, held=held)


def crossing_pairs(dp: D.Placed) -> set[tuple[int, str]]:
    """The accounting layer's crossings as (gate, reading unit), the pairs `placement.crossings` makes edges of: each
    crossing census value expanded to its gates."""
    return {(g, getattr(dp.datacenter.block_of(x.dst), "name", OUTSIDE)) for x in D.crossings(dp)
            for g in gates(dp.program, x.value)}


def epoch(dp: D.Placed) -> D.Interval:
    """The ticks the placement uses, from the first through the last."""
    return D.Interval(0, max(t for ts in dp.ticks.values() for t in ts) + 1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Audit and account a run of the MLP with the accounting layer.")
    ap.add_argument("--workload", choices=M.WORKLOADS, default="mlp",
                    help="the MLP, or verity_vllm's serving Program for a truncated SmolLM2 (`served`)")
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--devices", type=int, default=2)
    ap.add_argument("--d", type=int, default=4, help="mlp: the width")
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--requests", type=int, default=3, help="mlp: requests")
    ap.add_argument("--groups", type=int, default=1, help="mlp: tensor groups")
    ap.add_argument("--prompt", type=int, default=2, help="served: prompt tokens")
    ap.add_argument("--steps", type=int, default=1, help="served: greedy decode steps")
    ap.add_argument("--plan", choices=sorted(P.PLANS), default="pipeline")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--frame-bits", type=int, default=32)
    ap.add_argument("--buckets", type=int, default=4, help="buckets a certifier's record covers (a window)")
    ap.add_argument("--jitter-classes", type=int, default=128, help="the sync set's size: log2 of it is a window's timing advice")
    ap.add_argument("--tick", type=float, default=1e-3, help="seconds a tick")
    ap.add_argument("--rate", type=float, default=1e9, help="each GPU's declared rate for each kind of work")
    ap.add_argument("--extra-frame", action="store_true", help="add one covert frame on b0's uplink")
    ap.add_argument("--assume", action="append", default=[], choices=sorted(PREMISES),
                    help="a named premise for `account`")
    args = ap.parse_args(argv)

    spec = ({"d": args.d, "layers": args.layers, "requests": args.requests, "groups": args.groups}
            if args.workload == "mlp" else {"layers": args.layers, "prompt": args.prompt, "steps": args.steps})
    ctx = Context({"workload": args.workload, "spec": spec, "blocks": args.blocks, "devices": args.devices,
                   "plan": args.plan, "frame_bytes": 64 if args.workload == "mlp" else 1024, "seed": args.seed})
    cadence = D.Cadence(args.frame_bits, ctx.hw.bucket_ticks, args.buckets, args.jitter_classes,
                        D.Assumption("tick", args.tick, "s", "declared"))
    run_inputs, registered = split(ctx.model, ctx.seed)
    dc = datacenter(ctx.hw, cadence, kinds(ctx.model), args.rate)
    dp = placement(ctx.model, ctx.hw, ctx.placement, dc, D.register(ctx.model.program, registered))
    mine = {(e.value, e.dst) for e in ctx.edges}
    total = Counter()
    for w in D.work(ctx.model.program).values():
        total.update(w)
    print(f"a Program of {ctx.model.n} gates, {len(D.values(ctx.model.program))} census values and work "
          f"{dict(total)}; datacenter problems: {dc.validate() or 'none'}; paced: {dp.paced()}")
    print(f"crossings: {len(D.crossings(dp))} census values in the accounting layer, {len(mine)} gate edges in "
          f"placement.crossings, {'the same' if crossing_pairs(dp) == mine else 'different'} (gate, unit) pairs")
    deviation = None
    if args.extra_frame:
        deviation = D.Deviation(extra={(D.Port("b0.cert", "egress"), ctx.hw.bucket(ctx.placement.tick(ctx.edges[0].value))): 1})
    plan = D.Plan(epoch(dp))
    profile = dp.public(plan)
    trace = D.run(dp, run_inputs, plan, deviation)
    answers = ctx.model.output_words(trace.values)
    print(f"answers {'equal' if answers == ctx.model.reference(ctx.seed) else 'differ from'} the reference: {answers}")
    for port, row in sorted(profile.occupancy.items(), key=lambda kv: repr(kv[0])):
        print(f"  occupancy {port!r:<12} {[row.get(b, 0) for b in range(max(row) + 1)]}")
    claims = D.audit(trace, profile)
    kinds_ = Counter(type(k).__name__ for k in claims)
    print("claims: " + ", ".join(f"{n} {k}" for k, n in sorted(kinds_.items())))
    acc = D.account(claims, profile, assume=tuple(PREMISES[a] for a in args.assume))
    print(acc)
    return 0 if acc.sound else 1


if __name__ == "__main__":
    raise SystemExit(main())
