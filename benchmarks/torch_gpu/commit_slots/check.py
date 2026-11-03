"""On a CUDA device: the GPU call's roots against the CPU reference's (README.md), as JSON to ``--out``/check.json.

1. The fixture: the edge layout at its 3 slots gives ``fixtures/commit-slots-v1.json``'s root.
2. The edge layout at a few slot counts gives the reference's step root and slot roots; at larger ones, the host twin's.
3. Two H200 layers at 17 slots give the reference's; three at 256 the host twin's; three regrouped into two groups
   (``slots_gpu.regroup``) at 17 the reference's.
4. The frozen H200 layout (52 layers) at each of ``--slots`` (4,096, 16,384 and 20,480), each layer's outputs fresh
   random bytes on the device, committed one layer at a time with two buffers reused as a serving loop reuses them: the
   reference rebuilds each sampled slot from its bytes (copied before its buffer is reused) and gets the device's slot
   root, and its step tree over the device's slot roots gets the device's step root.

Exits 1 if any check fails.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import slots as S  # noqa: E402
import slots_gpu as G  # noqa: E402
from verity.ml.commitments.identity import identity_digest_sha512  # noqa: E402

CTX = identity_digest_sha512("verity/commit-slots/check/context/v1", {"check": 1})
KEY = bytes(range(32, 64))


def _cuda(outputs: list[list[np.ndarray]]) -> list[list[torch.Tensor]]:
    return [[torch.from_numpy(a).cuda() for a in tensors] for tensors in outputs]


def _same(a: S.StepCommit, b: S.StepCommit) -> dict[str, Any]:
    bad = [s for s, (x, y) in enumerate(zip(a.slot_roots, b.slot_roots, strict=True)) if x != y]
    return {"ok": a.root == b.root and not bad, "root": a.root.hex(), "other_root": b.root.hex(), "slot_roots_differ": bad[:16]}


def fixture_check() -> dict[str, Any]:
    doc = json.loads(S.FIXTURE.read_text())
    layout, slots = S.edge_layout(), doc["slots"]
    outputs = [[np.frombuffer(bytes.fromhex(h), np.uint8).reshape(slots, -1).copy() for h in tensors] for tensors in doc["outputs"]]
    got = G.commit(G.Plan(bytes.fromhex(doc["context"]), layout), doc["step"], bytes.fromhex(doc["salt_key"]), _cuda(outputs),
                   slots)
    return {"name": "fixture", "ok": got.root.hex() == doc["root"]
            and [r.hex() for r in got.slot_roots] == [d["root"] for d in doc["per_slot"]], "root": got.root.hex()}


def against(name: str, layout: S.CommitShape, slots: int, seed: int, other: str) -> dict[str, Any]:
    outputs = S.random_outputs(layout, slots, np.random.default_rng(seed))
    got = G.commit(G.Plan(CTX, layout), seed, KEY, _cuda(outputs), slots)
    if other == "reference":
        want = S.commit(CTX, layout, seed, KEY, outputs, slots)
    else:
        want = G.commit(G.Plan(CTX, layout, "host"), seed, KEY, outputs, slots)
    return {"name": name, "slots": slots, "against": other, **_same(got, want)}


def frozen(slots: int, samples: int, seed: int) -> dict[str, Any]:
    """The frozen layout at ``slots`` slots, layer by layer from two reused buffers, checked on sampled slots."""
    layout = S.h200_layout()
    group = layout.groups[0]
    rng = np.random.default_rng(seed)
    sampled = sorted({0, slots - 1, *rng.choice(slots, size=min(samples, slots), replace=False).tolist()})
    index = torch.tensor(sampled, device="cuda")
    plan = G.Plan(CTX, layout)
    gen = torch.Generator(device="cuda")
    bufs = [[torch.empty((slots, n), dtype=torch.uint8, device="cuda") for n in group.sources] for _ in range(2)]
    done: list[Any] = [None, None]
    kept: list[list[np.ndarray]] = []
    step = 11
    sc = G.StepCommitter(plan, step, slots, KEY)
    for layer in range(len(layout.groups)):
        b = layer % 2
        if done[b] is not None:
            torch.cuda.current_stream().wait_event(done[b])
        gen.manual_seed(seed * 1000 + layer)
        for t in bufs[b]:
            t.random_(generator=gen)
        kept.append([t.index_select(0, index).cpu().numpy() for t in bufs[b]])
        done[b] = sc.add(bufs[b])
    got = sc.finish().result()
    bad = []
    for i, s in enumerate(sampled):
        root = S.commit_slot(CTX, layout, KEY, s, S.slot_rows(layout, kept, i)).root
        if root != got.slot_roots[s]:
            bad.append(s)
    step_root = S.step_tree(CTX, layout, step, got.slot_roots).root
    return {"name": "frozen-h200", "slots": slots, "layers": len(layout.groups), "committed_bytes": slots * layout.slot_bytes,
            "sampled_slots": sampled, "sampled_slots_differ": bad, "step_root_from_slot_roots": step_root == got.root,
            "ok": not bad and step_root == got.root, "root": got.root.hex()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--slots", type=int, nargs="+", default=[4096, 16384, 20480], help="the frozen layout's slot counts")
    ap.add_argument("--samples", type=int, default=4, help="random slots the reference rebuilds, beside the first and last")
    ap.add_argument("--quick", action="store_true", help="skip the frozen layout")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    edge, two, three = S.edge_layout(), S.h200_layout(layers=2), S.h200_layout(layers=3)
    results: list[dict[str, Any]] = []

    def run(fn: Any, *args: Any) -> None:
        try:
            r = fn(*args)
        except Exception as e:  # noqa: BLE001  (a failing check is a result, and the rest still run)
            r = {"name": getattr(fn, "__name__", str(fn)), "args": [str(x) for x in args], "ok": False, "error": repr(e)}
        results.append(r)
        print(json.dumps(r), flush=True)

    run(fixture_check)
    for slots in (1, 2, 5, 33, 300):
        run(against, "edge", edge, slots, slots, "reference")
    for slots in (4097, 20480):
        run(against, "edge", edge, slots, slots, "host twin")
    run(against, "h200-2-layers", two, 17, 17, "reference")
    run(against, "h200-3-layers", three, 256, 256, "host twin")
    run(against, "h200-3-layers-in-2-groups", G.regroup(three, 2), 17, 17, "reference")
    if not a.quick:
        for slots in a.slots:
            run(frozen, slots, a.samples, 5)
    props = torch.cuda.get_device_properties(0)
    doc = {"ok": all(r["ok"] for r in results), "device": props.name, "capability": f"{props.major}.{props.minor}",
           "torch": torch.__version__, "checks": results}
    (a.out / "check.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"ok": doc["ok"], "checks": len(results), "failed": [r["name"] for r in results if not r["ok"]]}))
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
