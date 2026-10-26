"""K and V read across steps at the H200 step (CMP-161): a lockstep run committed again under the prover's salt secret (or a
fixture's), its roots held to the registered ones, and drawn attention blocks of its last step s opened as M3 proper's bundle
carries them. A ``QkSoftmaxRowPad_v1`` block reads the slot's key rows of steps 0 .. s and a ``PvRow_v1`` block its value
rows: each is the committed row of that step's ``l<i>.wk.gather`` (``l<i>.wv.gather``) unit, opened against that step's root
(`replay_bundle.open_reads`), and the block's port row is those rows' bytes, zero past n, and n = s + 1 its position word.
Every opening is then checked (`replay_bundle.check_reads`) against a registration naming every step's root
(`replay_bundle.steps_provenance`), and refused against one whose step root differs, with a path or a word tampered.

    python benchmarks/e2e_h200/kv_reads.py --out OUT [--secret FILE] [--registered ROOTS.json] [--slots K] [--blocks N]
        [--workers W] [run flags as m3draw.py's]

``--secret``: m3draw's salt secret (``SHA-512(tag || secret || step)[:32]`` a step), else the run id's fixture salt keys.
``--registered``: m3draw's ``roots.json``, every step's and slot's root, which the run's must equal (exit 5 otherwise).
Writes ``OUT/kv_reads.json`` (public: the step registration naming every step's root, the blocks, their reads' public
openings, the verdicts and timings) and nothing private. Exit 0 when every root is the registered one, every block's
replay holds, every read opens and checks and every tampering is refused; 3 when one doesn't.
"""
from __future__ import annotations

import argparse
import bisect
import copy
import hashlib
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "e2e_demo")]
import run as RUN
from e2e_h200.commit import Committer, fixture_salt, fixture_secrets
from e2e_h200.driver import Driver, Params
from e2e_h200.opening import draw_slots, replay_slot
from e2e_h200.shape import SHAPES
from e2e_h200.units import Replayer
from e2e_h200.weights import Weights
from verity_e2e_demo import replay_bundle as RB
from verity_torch_loop import weights as SW

FORMAT = "verity/e2e-h200/kv-reads/v1"
#: m3draw's salt keys (``m3draw.salt_from``)
SALT_TAG = b"e2e-h200/m3draw/salt-key/v1\0"
SECRET_BYTES = 64


def salt_from(secret: bytes) -> Any:
    if len(secret) != SECRET_BYTES:
        raise ValueError(f"the salt secret is {SECRET_BYTES} bytes, not {len(secret)}")
    return lambda step: hashlib.sha512(SALT_TAG + secret + step.to_bytes(4, "big")).digest()[:32]


class SlotRows(Sequence):
    """One slot's committed rows at one step from its host tables (one row each, ``sources[group]`` its tensors' bytes):
    row g on demand, every row when iterated (`slots.slot_rows`)."""

    def __init__(self, layout: Any, sources: list[list[np.ndarray]]) -> None:
        self.layout, self.sources = layout, sources

    def __len__(self) -> int:
        return self.layout.row_count

    def __getitem__(self, g: int) -> bytes:
        lay = self.layout
        i = bisect.bisect_right(lay.group_row_base, g) - 1
        r = g - lay.group_row_base[i]
        for unit in lay.groups[i].units:
            if r < len(unit):
                return b"".join(self.sources[i][p.source][p.offset:p.offset + p.nbytes].tobytes() for p in unit[r].parts)
            r -= len(unit)
        raise IndexError(g)

    def __iter__(self) -> Any:
        rows: list[bytes] = []
        for g, src in zip(self.layout.groups, self.sources, strict=True):
            rows += [b"".join(src[p.source][p.offset:p.offset + p.nbytes].tobytes() for p in r.parts) for u in g.units for r in u]
        return iter(rows)


def slot_rows(committer: Committer, st: Any) -> SlotRows:
    """Row 0 of a replayed step's tables as `SlotRows`."""
    tables = st.host()
    sources = []
    for group in committer.groups:
        ts = []
        for t in group:
            codes, word = tables[t.op]
            ts += [codes, word] if t.quant else [codes]
        sources.append([np.ascontiguousarray(np.asarray(x)[0]).view(np.uint8).reshape(-1) for x in ts])
    return SlotRows(committer.layout, sources)


def find_row(rows: SlotRows, layout: Any, unit: int, value: bytes) -> tuple[int, int]:
    """``(rank, start)`` of ``value``'s bytes in a row of commit unit ``unit``, at an even offset."""
    for rank in range(len(layout.units[unit])):
        row = rows[layout.row_base[unit] + rank]
        at = row.find(value)
        while at >= 0 and at % 2:
            at = row.find(value, at + 1)
        if at >= 0:
            return rank, at
    raise ValueError(f"commit unit {unit} holds no row with the read bytes")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--secret", default="", help="m3draw's salt secret (0600, outside the run dir); default fixture salt keys")
    ap.add_argument("--registered", default="", help="m3draw's roots.json, which every root must equal")
    ap.add_argument("--slots", type=int, default=2, help="slots drawn (fixture coin); every step of each is rebuilt")
    ap.add_argument("--blocks", type=int, default=8, help="attention blocks drawn over the slots, softmax and pv alternating")
    ap.add_argument("--workers", type=int, default=1, help="processes rebuilding the opened (step, slot) subtrees")
    ap.add_argument("--shape", default="small", choices=sorted(SHAPES))
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--prompt-bytes", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--micro-batch", type=int, default=0)
    ap.add_argument("--attention", default=None, choices=("attn_ref", "numpy"))
    ap.add_argument("--run-id", default="e2e-h200-fixture")
    ap.add_argument("--seed", default="test")
    return ap


def root_check(results: list, registered: dict, committer: Committer) -> dict:
    regs = registered["step_roots"]
    unequal = [r.step for r, g in zip(results, regs, strict=True)
               if r.root.hex() != g["root"] or [s.hex() for s in r.slot_roots] != g["slot_roots"]]
    fields = {"context": committer.context.hex(), "layout_digest": committer.describe()["layout_digest"]}
    return {"ok": not unequal and len(results) == len(regs) and all(registered.get(k) == v for k, v in fields.items()),
            "steps": len(regs), "steps_unequal": unequal,
            "fields_unequal": [k for k, v in fields.items() if registered.get(k) != v]}


def draw_blocks(shape: Any, slots: list[int], n: int, seed: bytes) -> list[tuple[int, str, int, int]]:
    """``n`` attention blocks over ``slots`` by a fixture coin: ``(slot, family, layer, head)``, softmax and pv alternating."""
    h = hashlib.sha512(b"e2e-h200/kv-reads/blocks-fixture\0" + seed).digest()
    rng = np.random.Generator(np.random.PCG64(int.from_bytes(h[:16], "big")))
    out = []
    for i in range(n):
        layer, head = int(rng.integers(shape.layers)), int(rng.integers(shape.heads))
        out.append((slots[i % len(slots)], "softmax" if i % 2 == 0 else "pv", layer, head))
    return out


def tamper_path(reads: list[dict]) -> list[dict]:
    out = copy.deepcopy(reads)
    q = next(q for q in out if q["kind"] == "step")
    proof = q["slot"]["slot_path"]["proof"]
    proof[0] = ("1" if proof[0][0] == "0" else "0") + proof[0][1:]
    return out


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    shape = SHAPES[a.shape]
    seed = {"test": SW.TEST_SEED, "demo": SW.DEMO_SEED}.get(a.seed, a.seed.encode())
    weights = Weights(shape, seed)
    store = RUN.make_store(argparse.Namespace(store="generator", salt="", stage_lib="", workers=a.workers, run_id=a.run_id),
                           weights, a.device)
    ctx, key = fixture_secrets(a.run_id)
    if a.secret:
        path = Path(a.secret)
        if path.stat().st_mode & 0o077:
            raise SystemExit(f"{path} is readable by others: the salt secret is the prover's alone (chmod 600)")
        salt = salt_from(path.read_bytes())
    else:
        salt = fixture_salt(a.run_id)
    committer = Committer("slots", ctx, key, shape, a.device, salt)
    lay = committer.layout
    prompts = RUN.prompts_for(a.run_id, a.batch, a.prompt_bytes)
    params = Params(shape, store, a.device, True)
    t0 = time.perf_counter()
    driver = Driver(shape, params, a.batch, committer, a.device, keep=False, micro_batch=a.micro_batch or None,
                    attention=a.attention)
    results = driver.run(prompts, a.steps)
    seconds = {"run": time.perf_counter() - t0}
    print(json.dumps({"steps": len(results), "last_root": results[-1].root.hex()[:32], "seconds": seconds["run"]}), flush=True)
    rc = None
    if a.registered:
        rc = root_check(results, json.loads(Path(a.registered).read_text()), committer)
        print("ROOTCHECK " + ("EQUAL " if rc["ok"] else "UNEQUAL ") + json.dumps(rc), flush=True)
        if not rc["ok"]:
            return 5
    s = len(results) - 1
    if any(int(p) != s for p in results[s].positions):
        raise SystemExit("a lockstep run's slots are at position s at step s")

    t0 = time.perf_counter()
    slots = draw_slots(a.batch, a.slots, a.run_id.encode(), s)
    blocks = draw_blocks(shape, slots, a.blocks, a.run_id.encode())
    records: dict[int, dict[int, SlotRows]] = {t: {} for t in range(len(results))}
    items, docs, bad = [], [], 0
    for b in slots:
        op = replay_slot(shape, params, committer, results, prompts, b, s, a.device, a.attention)
        if not all(op.roots_equal):
            raise SystemExit(f"slot {b}'s replay is not its committed slot roots at steps {op.roots_equal}")
        for t in range(s + 1):
            records[t][b] = slot_rows(committer, op.driver.steps[t])
        rp = Replayer(shape, weights, op.driver.steps, op.tokens)
        fams = {f.name: f for f in rp.model_families(s)}
        for bb, fam, layer, head in blocks:
            if bb != b:
                continue
            f = fams[f"l{layer}.{fam}"]
            inputs, got, want = f.replay(0, head)
            ok = got.shape == want.shape and bool(np.array_equal(got, want))
            unit = committer.bases[f"l{layer}.{'wk' if fam == 'softmax' else 'wv'}.gather"]
            kv, d = inputs[1], inputs[1].shape[1]
            pieces = []
            for t in range(s + 1):
                rank, start = find_row(records[t][b], lay, unit, kv[t].tobytes())
                pieces.append(RB.Read(t, b, unit, rank, at=t * d, start=start, nbytes=2 * d))
            c = {"block": 0, "step": s, "slot": b, "layer": layer, "op": f.name, "unit": head}
            items.append((c, np.ascontiguousarray(kv).tobytes(), pieces))
            entry = {"slot": b, "family": f.name, "template": f.template, "head": head, "replay_ok": ok,
                     "port": "k" if fam == "softmax" else "v", "reads": len(pieces), "words": int(kv.size)}
            if fam == "softmax":
                n = np.ascontiguousarray(inputs[2]).astype("<u2").tobytes()
                items.append((c, n, [RB.Position(1, 0)]))
                entry["n"] = int(inputs[2][0])
            docs.append(entry)
            bad += not ok
    seconds["replay"] = time.perf_counter() - t0
    com = RB.Commitment(committer.context, lay, a.batch,
                        {(0, t): RB.StepRecord(results[t].salt_key, records[t], tuple(results[t].slot_roots))
                         for t in range(s + 1)})

    t0 = time.perf_counter()
    reads, payload = RB.open_reads(com, items, workers=a.workers)
    seconds["open"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    cdoc = RB.commitment_doc(com)
    reg = {"window": RB.steps_provenance(com)}
    rows = [(c, row, r) for (c, row, _), r in zip(items, reads, strict=True)]
    verdict = RB.check_reads(cdoc, rows, payload, registration=reg)
    other = copy.deepcopy(reg)
    other["window"]["commitment"]["blocks"][0]["steps"][0]["root"] = "00" * 64
    refusals = {"another step-0 root": RB.check_reads(cdoc, rows, payload, registration=other),
                "a tampered path": RB.check_reads(cdoc, [(rows[0][0], rows[0][1], tamper_path(rows[0][2]))], payload),
                "a changed word": RB.check_reads(cdoc, [(rows[0][0], bytes([rows[0][1][0] ^ 1]) + rows[0][1][1:], rows[0][2])],
                                                 payload)}
    seconds["check"] = time.perf_counter() - t0
    step_reg = RB.step_registration(com, {}, {"law": "subset", "k": 8}, run=a.run_id, program="", block=0, step=s,
                                    read_steps=True)
    refused = all(refusals.values()) and all(refusals["another step-0 root"].get(f"row {i}") for i, (_, _, p) in
                                             enumerate(items) if any(isinstance(q, RB.Read) and q.step == 0 for q in p))
    ok = not verdict and refused and not bad
    doc = {"format": FORMAT, "run_id": a.run_id, "shape": shape.name, "batch": a.batch, "steps": len(results), "step": s,
           "layout_digest": cdoc["layout_digest"], "root_check": rc, "step_registration": step_reg,
           "step_registration_digest": RB.step_digest(step_reg), "blocks": docs,
           "reads": [[{k: v for k, v in q.items() if k not in ("row", "salt")} for q in r] for r in reads],
           "payload_bytes": len(payload), "check": verdict, "refusals": refusals, "ok": ok, "seconds": seconds,
           "workers": a.workers}
    (out / "kv_reads.json").write_text(json.dumps(doc, indent=1) + "\n")
    print("VERDICT " + json.dumps({"ok": ok, "blocks": len(docs), "replay_bad": bad,
                                   "reads_opened": sum(q["kind"] == "step" for r in reads for q in r),
                                   "steps_read": sorted({q["slot"]["step"] for r in reads for q in r if q["kind"] == "step"}),
                                   "check": verdict or "clean", "refused": {k: len(v) for k, v in refusals.items()},
                                   "step_registration": RB.step_digest(step_reg)[:32], "seconds": seconds}), flush=True)
    store.close()
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
