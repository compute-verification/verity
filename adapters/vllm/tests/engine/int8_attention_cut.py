"""Cut one decoder layer's int8 attention scope out of a served capture (`pipeline.m1_capture` with `--snapshot-steps`), with the
memory words its fold reads and writes, as the recorded test's fixture (`test_int8_attention_recorded`).

    python -m tests.engine.int8_attention_cut CAPTURE_DIR --layer L --steps 0,1 --out CUT.jsonl.gz [--origin WHERE]

`CUT.jsonl.gz` is a log the fold reads (`verity-capture/log/v1`), cut as `moe_scope_cut` cuts a MoE scope: the RunHeader, an
`int8_attention_cut` Note naming the source log and its sha256, and per step its StepStart, every event of the layer's
`vllm.unified_attention_with_output` subtree, and its StepEnd.  `CUT.values.npz` is the memory those events left: for every
allocation an operand or output of the scope's fold (`observe.fold.patterns.int8.Int8AttentionScope`) names, the words of exactly
the elements it names (a request's KV slots of every head, its query and output rows, kv_quant_batch's results), each from the
capture's latest snapshot in the scope that covers it, as runs of storage elements (`r<step>_<alloc>`, [start, length] rows) and
their words (`w<step>_<alloc>`).  Both files are written with mtime 0, so one capture always cuts to the same bytes.
"""
from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np

from tests.engine.moe_scope_cut import _views, write
from verity_vllm.check.fold_compare import SnapshotStore, _storage_indices
from verity_vllm.observe.fold.profiles import profile_for_case
from verity_vllm.observe.events import DispatchEnter, DispatchExit, Event, Note, Root, RunHeader, StepEnd, StepStart, TritonLaunch, ViewDesc
from verity_vllm.observe.fold.patterns.int8 import Int8AttentionScope
from verity_vllm.observe.fold.patterns.kv_cache import kv_layer_of
from verity_vllm.observe.fold.patterns.ops import op_name
from verity_vllm.observe.fold.resolve_log import collect_roots, make_context
from verity_vllm.observe.fold.resolver import Compute, Const, GatherMany, Group, Shared, resolve_step
from verity_vllm.observe.fold.tree import TreeNode, step_events
from verity_vllm.observe.log import read_log, split_steps

ROLE = "LLAMA31_8B_INT8"
WORD = {"bfloat16": np.uint16, "int8": np.uint8, "float32": np.uint32}


def profile():
    return dataclasses.replace(profile_for_case(ROLE), patterns=[Int8AttentionScope()])


def scope_of(roots: list[TreeNode], layer: int) -> TreeNode:
    module = f"model.layers.{layer}.self_attn.attn"
    hits = [n for r in roots for n in r.walk() if not n.is_triton and op_name(n.op) == "unified_attention_with_output" and kv_layer_of(n) == module]
    if len(hits) != 1:
        raise SystemExit(f"{len(hits)} unified_attention_with_output scopes of {module} in the step, not one")
    return hits[0]


def fold_scope(header: RunHeader, step: list[Event], roots: dict) -> tuple[Group, TreeNode]:
    """The scope's one resolution under the int8 profile, which must be the attention Group."""
    se = step_events(step)
    (res,) = resolve_step(se, make_context(header, se, roots, profile()))
    if not isinstance(res, Group):
        raise SystemExit(f"step {se.start.step}: the scope folds to {res!r}, not Int8AttentionScope's Group")
    return res, se.roots[0]


def rows_of(o) -> list[ViewDesc]:
    """The views an operand or output names, a GatherMany's one per gathered row."""
    if isinstance(o, GatherMany):
        v = o.view
        return [ViewDesc(v.alloc, v.offset + j * v.strides[0], v.shape[1:], v.strides[1:], v.dtype, v.device) for js in o.indices for j in js]
    if isinstance(o, Const):
        return []
    if isinstance(o, Shared):
        return [o.view]
    raise SystemExit(f"Int8AttentionScope names a {type(o).__name__} operand")


def image(group: Group, store: SnapshotStore, last: int) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """alloc -> (runs [start, length], words) of every element the group's Computes read or write, at the scope's end."""
    got: dict[int, dict[int, int]] = {}
    dtype: dict[int, type] = {}
    for c in group.parts:
        if not isinstance(c, Compute):
            continue
        for v in [w for o in (*c.operands, *c.outputs) for w in rows_of(o)]:
            s = store.find(v, before=last + 1)
            if s is None:
                raise SystemExit(f"no snapshot covers {v!r} by event {last}")
            if dtype.setdefault(v.alloc, WORD[v.dtype]) is not WORD[v.dtype]:
                raise SystemExit(f"alloc {v.alloc} is read as two word widths in the scope")
            el, words = _storage_indices(v).reshape(-1), store.extract(s, v).view(WORD[v.dtype])
            have = got.setdefault(v.alloc, {})
            for e, w in zip(el.tolist(), words.tolist()):
                if have.setdefault(e, w) != w:
                    raise SystemExit(f"alloc {v.alloc} element {e}: two words in the scope's snapshots")
    out = {}
    for a, have in got.items():
        el = np.array(sorted(have), dtype=np.int64)
        cut = np.flatnonzero(np.diff(el) != 1) + 1
        starts, ends = el[np.r_[0, cut]], el[np.r_[cut - 1, el.size - 1]]
        out[a] = (np.stack([starts, ends - starts + 1], axis=1), np.array([have[e] for e in el.tolist()], dtype=dtype[a]))
    return out


def cut(capture: Path, layer: int, steps: set[int], source: dict) -> tuple[list[Event], dict[str, np.ndarray]]:
    store = SnapshotStore(capture)
    header, roots, events, arrays = None, {}, [], {}
    root_events: list[Root] = []
    for pre, step in split_steps(read_log(capture / "log.jsonl.gz")):
        header = next((e for e in pre if isinstance(e, RunHeader)), header)
        collect_roots(pre, roots)
        root_events += [e for e in pre if isinstance(e, Root)]
        if not step or step[0].step not in steps:
            continue
        scope = scope_of(step_events(step).roots, layer)
        ids = scope.ids()
        kept = [e for e in step if isinstance(e, (StepStart, StepEnd)) or isinstance(e, (DispatchEnter, DispatchExit, TritonLaunch)) and e.id in ids]
        group, _ = fold_scope(header, kept, roots)
        n = kept[0].step
        for a, (runs, words) in image(group, store, max(ids)).items():
            arrays[f"r{n}_{a}"], arrays[f"w{n}_{a}"] = runs, words
        events.append(kept)
    if header is None or len(events) != len(steps):
        raise SystemExit(f"the capture has no RunHeader or lacks steps {sorted(steps)}")
    touched = {v.alloc for b in events for e in b for v in _views(e)}
    note = Note("int8_attention_cut", {**source, "layer": layer, "steps": sorted(steps), "cut_by": "tests/engine/int8_attention_cut.py"})
    return [header, note, *[r for r in root_events if r.view.alloc in touched], *[e for b in events for e in b]], arrays


def write_values(arrays: dict[str, np.ndarray], out: Path) -> None:
    """`np.savez_compressed`'s layout with every member stamped 1980-01-01, so the bytes depend on the words alone."""
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for k in sorted(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.ascontiguousarray(arrays[k]), allow_pickle=False)
            z.writestr(zipfile.ZipInfo(f"{k}.npy", (1980, 1, 1, 0, 0, 0)), buf.getvalue(), zipfile.ZIP_DEFLATED)


def values_path(cut_path: Path) -> Path:
    return cut_path.with_name(cut_path.name.removesuffix(".jsonl.gz") + ".values.npz")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("capture", type=Path, help="m1_capture's --out: log.jsonl.gz, snapshots.json, values/")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--steps", required=True, help="comma-separated engine steps, e.g. 0,1")
    ap.add_argument("--out", type=Path, required=True, help="the cut log, NAME.jsonl.gz; the words go to NAME.values.npz beside it")
    ap.add_argument("--origin", default=None, help="where the capture was recorded (default: the path given)")
    a = ap.parse_args(argv)
    log = a.capture / "log.jsonl.gz"
    source = {"origin": a.origin or str(log), "sha256": hashlib.sha256(log.read_bytes()).hexdigest()}
    events, arrays = cut(a.capture, a.layer, {int(s) for s in a.steps.split(",")}, source)
    write(events, a.out)
    write_values(arrays, values_path(a.out))
    raw = len(gzip.decompress(a.out.read_bytes()))
    print(f"{a.out}: {len(events)} events, {a.out.stat().st_size} bytes ({raw} raw); {values_path(a.out)}: {len(arrays) // 2} allocations, "
          f"{sum(v.nbytes for k, v in arrays.items() if k[0] == 'w')} word bytes, {values_path(a.out).stat().st_size} bytes", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
