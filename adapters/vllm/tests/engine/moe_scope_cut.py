"""Cut one decoder layer's MoE scope out of a rank's capture log, as a log the fold reads (`verity-capture/log/v1`).

    python -m tests.engine.moe_scope_cut LOG.jsonl.gz --layer L --steps 0,1 --out CUT.jsonl.gz [--origin WHERE] [--logits]

Kept, in log order: the RunHeader; a `moe_scope_cut` Note naming the source (`--origin`, its sha256, the layer, the steps); every Root
whose allocation a kept event reads or writes (the layer's `mlp.gate`, `w13` and `w2` weights, and a shared expert's linears and the
router's score bias); and per step its StepStart, every event of the layer's `vllm.moe_forward` or `vllm.moe_forward_shared` subtree, the
top-level node right after the scope when it carries the scope's module (the block's TP all-reduce, which a rank issues after the scope,
or the shared and routed outputs' add), with `--logits` the step's `logits_processor` nodes (the lm_head shard's GEMM and its TP
all-gather), and its StepEnd.  Allocator records, the sampler and the other layers are left out: no MoE scope pattern reads them.
The output is gzip with mtime 0, so one source always cuts to the same bytes.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
from pathlib import Path
from typing import Iterable

from verity_vllm.observe.events import DispatchEnter, DispatchExit, Event, Note, Root, RunHeader, StepEnd, StepStart, TritonLaunch, views_in
from verity_vllm.observe.fold.tree import step_events
from verity_vllm.observe.log import dumps, read_log, split_steps

MOE_OPS = ("vllm.moe_forward.default", "vllm.moe_forward_shared.default")
LOGITS_MODULE = "logits_processor"


def cut(events: Iterable[Event], layer: int, steps: set[int], source: dict, *, logits: bool = False) -> list[Event]:
    module = f"model.layers.{layer}.mlp.experts"
    header: RunHeader | None = None
    roots: list[Root] = []
    bodies: list[list[Event]] = []
    for pre, step in split_steps(events):
        header = next((e for e in pre if isinstance(e, RunHeader)), header)
        roots += [e for e in pre if isinstance(e, Root)]
        if not step or step[0].step not in steps:
            continue
        se = step_events(step)
        k = next((i for i, r in enumerate(se.roots) if r.op in MOE_OPS and r.enter.module == module), None)
        if k is None:
            raise SystemExit(f"step {se.start.step}: no {' / '.join(MOE_OPS)} scope of {module}")
        nodes = [se.roots[k]] + [n for n in se.roots[k + 1:k + 2] if n.enter.module == module]
        if logits:
            nodes += [n for n in se.roots if n.enter.module == LOGITS_MODULE]
        ids = set().union(*(n.ids() for n in nodes))
        bodies.append([e for e in step if isinstance(e, (StepStart, StepEnd, Root))
                       or isinstance(e, (DispatchEnter, DispatchExit, TritonLaunch)) and e.id in ids])
    if header is None or len(bodies) != len(steps):
        raise SystemExit(f"the log has no RunHeader or lacks steps {sorted(steps)}")
    touched = {v.alloc for b in bodies for e in b for v in _views(e)}
    used = [r for r in roots if r.view.alloc in touched]
    note = Note("moe_scope_cut", {**source, "layer": layer, "module": module, "steps": sorted(steps), **({"logits": True} if logits else {}),
                                  "cut_by": "tests/engine/moe_scope_cut.py"})
    out: list[Event] = [header, note, *used]
    for b in bodies:
        out += [e for e in b if not isinstance(e, Root) or e.view.alloc in touched]
    return out


def _views(e: Event):
    if isinstance(e, DispatchEnter):
        return [*views_in(e.args), *views_in(e.kwargs)]
    if isinstance(e, DispatchExit):
        return list(views_in(e.outs))
    if isinstance(e, TritonLaunch):
        return list(views_in(e.args))
    return []


def write(events: list[Event], out: Path) -> None:
    with open(out, "wb") as f, gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as g:
        g.write(dumps(events).encode("ascii"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("log", type=Path)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--steps", required=True, help="comma-separated engine steps, e.g. 0,1")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--origin", default=None, help="where the log was recorded (default: the path given)")
    ap.add_argument("--logits", action="store_true", help="also keep each step's logits_processor nodes (the lm_head shard, its all-gather)")
    a = ap.parse_args(argv)
    source = {"origin": a.origin or str(a.log), "sha256": hashlib.sha256(a.log.read_bytes()).hexdigest()}
    events = cut(read_log(a.log), a.layer, {int(s) for s in a.steps.split(",")}, source, logits=a.logits)
    write(events, a.out)
    print(f"{a.out}: {len(events)} events, {a.out.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
