"""The dependent chain (``--decode-method dependent-chain``, at every decode shape): L calls timed as one chain, where call
j's A is derived from call j - 1's y (`derive.py`'s f, one kernel on the timing stream), so no call can start before the
previous call's y is complete. The plain GEMM's decode divisor is timed the same way (f, then the GEMM), and so is every arm.

The calls cycle the shape's operand sets (distinct weights past twice the L2), L = the least multiple of the sets that is at
least `MIN_LENGTH`; call 0 derives from the last call of the previous replay. Variants, per chain:

- ``full``: f, then the call's `launch`.
- An arm whose calls have `phases` (step groups, `arm.GROUPS`; in order, the same work as `launch`) also gets, by the
  parts `arm.PARTS` makes of them (before: a_tree, forming; gemm: gemm, cleanup; after: tile_hash, screen, the movable
  groups), ``gemm`` (f, then the gemm part alone), ``before-gemm`` (f, before, gemm) and ``side`` (f, before, gemm and the
  screen on the timing stream, tile hashing (`arm.DEFERRED`) forked to a side stream after the gemm part: the served
  ``deferred`` schedule, `verity.core.protocols.pouw.serving.DEFERRED`; the chain waits for call j - S's side work before deriving call
  j's A, since call j reuses its operand set, and joins the side stream at its end, so the drain is timed once per L
  calls). v0.4's phases ``before``, ``gemm`` and ``after`` are each their part, and get no ``side``: their one ``after``
  can't be split into what the served schedule moves and what it keeps.
- ``derive-only``: f alone, per derive kind (what f costs, y resident in the L2).

The split, per call: derive = derive-only; before = before-gemm - gemm; during = gemm - derive; after = full - before-gemm;
the side chain's exposed after = side - before-gemm (the screen, and whatever of the tile hashing the side stream doesn't
hide).

The gate, before any chain is timed: its call's own gate passed; then S + 1 steps, each f on the device and a sync, the
derived A (and its scales) byte-equal to the twin's f of the source y read back, then the call launched; for an arm, its
own gate on the last call, run on the chain's derived A. The negative control, at the first chain shape: a step whose f is
skipped must fail the same check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import arm as A
import baselines as B
import derive as D
from arm import Buffer

MIN_LENGTH = 64
#: deprecated: the parts' names, v0.4's phases
PHASES = tuple(A.PARTS)
SPLIT_VARIANTS = ("gemm", "before-gemm", "side")
#: the split's times (`bench.chain_report`), each a part's
SPLIT_PARTS = {"before_ms": "before", "during_ms": "gemm", "after_ms": "after"}
Y_BYTES = {"bf16": 2, "int32": 4, "f32": 4}


def length(sets: int) -> int:
    return math.ceil(max(MIN_LENGTH, sets) / sets) * sets


@dataclass
class Link:
    """One operand set's call in a chain: the call, the A (and A's block scales) it reads, the y it writes."""

    call: object
    a: Buffer
    sf: Buffer | None
    y: Buffer
    y_src: str


class Chain:
    """A chain's body, enqueued on a stream: `enqueue(stream, n)` is n calls (the graph captures it)."""

    def __init__(self, dev, shape, links: list[Link], kind: str, variant: str):
        self.dev, self.shape, self.links, self.kind, self.variant = dev, shape, links, D.KINDS[kind], variant
        self.fork = [dev.event() for _ in links] if variant == "side" else []
        self.done = [dev.event() for _ in links] if variant == "side" else []
        self.parts = [A.phase_parts(ln.call.phases)[0] for ln in links] if variant in SPLIT_VARIANTS else []
        self.sides = [A.side_split(ln.call.phases) for ln in links] if variant == "side" else []
        if None in self.sides:
            raise A.ArmError(f"a side chain needs step-group phases with a deferred group ({', '.join(sorted(A.DEFERRED))})")

    def part(self, j: int, name: str, stream: int) -> None:
        for fn in self.parts[j % len(self.links)][name]:
            fn(stream)

    def derive(self, j: int, stream: int, src: int | None = None) -> None:
        s, d = self.links[(j - 1) % len(self.links) if src is None else src], self.links[j % len(self.links)]
        self.dev.derive(D.SRC[s.y_src], s.y.ptr, self.shape.m, self.shape.n, self.kind.id, d.a.ptr,
                        d.sf.ptr if d.sf else None, self.shape.k, stream)

    def enqueue(self, stream: int, n: int) -> None:
        dev, S = self.dev, len(self.links)
        for j in range(n):
            link = self.links[j % S]
            if self.variant == "side" and j >= S:
                dev.wait(stream, self.done[j % S])
            self.derive(j, stream)
            if self.variant == "full":
                link.call.launch(stream)
            elif self.variant == "gemm":
                self.part(j, "gemm", stream)
            elif self.variant == "before-gemm":
                self.part(j, "before", stream)
                self.part(j, "gemm", stream)
            elif self.variant == "side":
                kept, deferred = self.sides[j % S]
                self.part(j, "before", stream)
                self.part(j, "gemm", stream)
                dev.record(self.fork[j % S], stream)
                side = dev.side_stream
                dev.wait(side, self.fork[j % S])
                for fn in deferred:
                    fn(side)
                dev.record(self.done[j % S], side)
                for fn in kept:
                    fn(stream)
        if self.variant == "side" and n:
            dev.wait(stream, self.done[(n - 1) % S])

    def release(self) -> None:
        for e in self.fork + self.done:
            self.dev.event_destroy(e)
        self.fork, self.done = [], []

    # ---- the gate ----------------------------------------------------------------------------------------------------------

    def check(self, j: int, src: int | None = None) -> dict:
        """Link j's A (and scales) against the twin's f of its source's y, as the device holds them now."""
        dev, sh = self.dev, self.shape
        s, d = self.links[(j - 1) % len(self.links) if src is None else src], self.links[j % len(self.links)]
        y = dev.read(s.y.ptr, sh.m * sh.n * Y_BYTES[s.y_src])
        a = dev.read(d.a.ptr, D.a_bytes(self.kind, sh.m, sh.k))
        sf = dev.read(d.sf.ptr, d.sf.nbytes) if d.sf else None
        return D.check(self.kind.name, s.y_src, y, a, sf, sh.m, sh.n, sh.k)


def _summary(r: dict) -> dict:
    return {k: v for k, v in r.items() if k in ("ok", "a_mismatches", "scale_mismatches", "a_first", "scale_first")}


def gate(chain: Chain, ledger, base_key: tuple | None, arm=None, flags: frozenset[str] | None = None):
    """The chain's gate (the module docstring): `fn() -> (ok, detail)` for `Bench.gate`."""
    def run():
        if base_key is not None and not ledger.passed(base_key):
            return False, {"error": f"{'/'.join(map(str, base_key))}: the call's own gate did not pass"}
        dev, S, steps = chain.dev, len(chain.links), []
        for j in range(S + 1):
            chain.derive(j, dev.stream)
            dev.sync()
            r = chain.check(j)
            steps.append({"step": j, **_summary(r)})
            if not r["ok"]:
                return False, {"steps": steps}
            if chain.variant != "derive-only":
                chain.links[j % S].call.launch(dev.stream)
                dev.sync()
        detail: dict = {"steps": len(steps), "derive": chain.kind.name, "a_bytes_per_step": r["a_bytes"]}
        if arm is None:
            return True, detail
        call = chain.links[S % S].call
        g = arm.gate(dev, call, chain.shape, flags)
        A.check_gate(g, call)
        detail["arm_gate"] = {"ok": bool(g.ok), "checked": g.checked, "detail": g.detail}
        return bool(g.ok), detail
    return run


def stale_control(chain: Chain):
    """The negative control: link 1's A, not derived again after link 0 launched, against f of link 0's y now."""
    def run():
        dev = chain.dev
        chain.links[0].call.launch(dev.stream)
        dev.sync()
        r = chain.check(1, src=0)
        return r["ok"], _summary(r)
    return run


# ---- building a shape's chains -----------------------------------------------------------------------------------------------


def baseline_links(entry, fam: B.Family) -> list[Link]:
    kind = D.KINDS[D.FAMILY_KIND[fam.name]]
    return [Link(c, c.ops.a, c.ops.sa if kind.block > 1 else None, c.ops.d, "int32" if fam.out == "int32" else "bf16")
            for c in entry.calls]


def arm_links(entry, arm, opsets: list[A.OperandSet], shape) -> tuple[list[Link] | None, str | None]:
    """The arm entry's links, or why it can't be chained: its y (`outputs["y"]`, else `outputs["d"]`; `y_format` on the
    call, BF16 by default) must hold m x n words."""
    kind = D.KINDS[D.INPUT_KIND[arm.inputs]]
    links = []
    for c, o in zip(entry.calls, opsets):
        y = c.outputs.get("y") or c.outputs.get("d")
        fmt = getattr(c, "y_format", "bf16")
        if y is None:
            return None, "the call has no y output (outputs['y'] or outputs['d'])"
        if fmt not in Y_BYTES or y.nbytes < shape.m * shape.n * Y_BYTES[fmt]:
            return None, f"y ({fmt}, {y.nbytes} bytes) is not an m x n matrix of bf16, int32 or f32"
        if kind.block > 1 and o.a_scale is None:
            return None, f"inputs {arm.inputs} have block scales but the operand set has no a_scale"
        try:
            A.phase_parts(getattr(c, "phases", None))
        except A.ArmError as ex:
            return None, str(ex)
        links.append(Link(c, o.a, o.a_scale if kind.block > 1 else None, y, fmt))
    return links, None


def has_phases(links: list[Link]) -> bool:
    return all(A.phase_parts(getattr(ln.call, "phases", None)) is not None for ln in links)


def has_side(links: list[Link]) -> bool:
    return all(A.side_split(getattr(ln.call, "phases", None)) is not None for ln in links)


def split_groups(links: list[Link]) -> dict[str, list[str]] | str:
    """The step groups each of the split's times covers (`SPLIT_PARTS`), as link 0's call names them, and those the side
    stream took (``side_stream``)."""
    phases = links[0].call.phases
    if not A.phase_parts(phases)[1]:
        return "v0.4 phases: the arm's own before, gemm and after (no side stream)"
    return ({t: [g for g in A.PARTS[p] if g in phases] for t, p in SPLIT_PARTS.items()}
            | {"side_stream": [g for g in A.PARTS["after"] if g in phases and g in A.DEFERRED]})
