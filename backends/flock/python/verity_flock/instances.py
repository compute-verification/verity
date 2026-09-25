"""A frozen instance set's range as the Flock prover and verifier read it, with its frame-v3 statement commitments.

The instances are B-Ligero's synthetic sets (``backends/direct/ligero/relations.py``: the recipe the frozen
``bench-instances-bf16-hopper/v1``, ``bench-instances-fp8-ada/v1`` and ``bench-instances-fp8-hopper/v1`` tiers name),
drawn exactly as ``relchain._instance_np`` draws them, with the model-recorded accumulators. The statement commits them
as B-Ligero's ``+blake3`` configuration does: tree ``a`` over the x rows and ``b`` over the W columns
(``blake3-keyed/row/v2`` row leaves), tree ``y`` over the output words (a word leaf of ``y_public`` of the final
accumulator), each under the ``hashauth.binding_digest`` domain of the set and range. So a Flock cell and a B-Ligero
cell on the same line bind the same three roots.

File (``flock-pure-instances/v1``): one JSON header line, then x rows, W columns (``row_bytes`` each, words
little-endian), the accumulators (``units`` u32 LE per VU), the circuit's output word per VU (u32 LE: the BF16 word for
an epilogue relation, else the FP32 accumulator) and the committed y word per VU (u32 LE).
"""
from __future__ import annotations

import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .lowering import PIPES

FORMAT = "flock-pure-instances/v1"
K = 1536
_DIRECT = Path(__file__).resolve().parents[3] / "direct"


def _ligero():
    if str(_DIRECT) not in sys.path:
        sys.path.insert(0, str(_DIRECT))
    from ligero import auth, hashauth
    from ligero.relations import relation
    return relation, hashauth, auth


def instances_digest(name: str, n: int) -> str:
    relation, _, _ = _ligero()
    rel = relation(name)
    return hashlib.sha256(f"{rel.name} synthetic|seed={rel.instance_seed}|n={n}|K={K}".encode()).hexdigest()


def instances_ref(name: str, total: int) -> dict:
    relation, _, _ = _ligero()
    rel = relation(name)
    ref = {"dataset": rel.instances_dataset, "tier": rel.instances_tier, "range": [0, total],
           "manifest_sha256": instances_digest(name, total), "seed": rel.instance_seed}
    if rel.instances_recipe:
        ref["recipe"] = rel.instances_recipe
    return ref


def _e4m3_operands(rng: np.random.Generator, shape) -> np.ndarray:
    """``ligero.fp8.witness.random_finite_operands`` (whose module imports torch), byte for byte."""
    w = rng.integers(0, 256, size=shape, dtype=np.int64)
    nan = (w & 0x7F) == 0x7F
    w[nan] = w[nan] & ~0x08
    return w


def _draw(args):
    name, seed, lo, hi = args
    from verity.ml.tc.silicon import tc_dot
    relation, _, _ = _ligero()
    rel = relation(name)
    operands = _e4m3_operands if rel.word_dtype == "<u1" else rel.operands
    xs, ws, accs = [], [], []
    for i in range(lo, hi):
        rng = np.random.default_rng([seed, i])
        a, b = operands(rng, (K,)), operands(rng, (K,))
        al, bl = a.tolist(), b.tolist()
        c, row = 0, []
        for s in range(rel.steps):
            c = tc_dot(rel.model, c, al[rel.k * s:rel.k * s + rel.k], bl[rel.k * s:rel.k * s + rel.k])
            row.append(c)
        xs.append(a)
        ws.append(b)
        accs.append(row)
    return xs, ws, accs


def draw(name: str, total: int, procs: int = 1, cache: str | Path | None = None):
    """``(x (total, K), w (total, K), accs (total, steps))`` of the set's first ``total`` VUs."""
    relation, _, _ = _ligero()
    rel = relation(name)
    path = Path(cache) / f"flock-{name}-{rel.instance_seed}-{total}.npz" if cache else None
    if path is not None and path.exists():
        with np.load(path) as z:
            return z["x"], z["w"], z["accs"]
    step = max(1, total // max(1, 8 * procs))
    jobs = [(name, rel.instance_seed, lo, min(total, lo + step)) for lo in range(0, total, step)]
    if procs > 1:
        with ProcessPoolExecutor(procs) as ex:
            parts = list(ex.map(_draw, jobs))
    else:
        parts = [_draw(j) for j in jobs]
    dt = np.uint16 if rel.word_dtype == "<u2" else np.uint8
    x = np.asarray([r for p in parts for r in p[0]], dtype=dt)
    w = np.asarray([r for p in parts for r in p[1]], dtype=dt)
    accs = np.asarray([r for p in parts for r in p[2]], dtype=np.uint32)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, x=x, w=w, accs=accs)
    return x, w, accs


def statement(name: str, x, w, accs) -> dict:
    """The three frame-v3 roots (and their domains) over the set's first ``len(x)`` VUs."""
    from verity.commitments.frame_v3 import FrameV3, Port, RowLeaf, WordLeaf
    from verity.commitments.indexed import RangeIndexedDomain
    from verity.commitments.merkle import CommitmentDomain
    from verity.commitments.rowleaf import ROLE_W, ROLE_X, blake3_row_digest

    relation, hashauth, auth = _ligero()
    rel = relation(name)
    total = len(x)
    word_bits = 16 if rel.word_dtype == "<u2" else 8
    ysch, ybytes = hashauth.word_schema(rel.y_bits)
    manifest = instances_digest(name, total)
    counts = {"a": total, "b": total, "y": total}
    schemas = {"a": "blake3-keyed/row/v2", "b": "blake3-keyed/row/v2", "y": ysch}
    bind = {n: hashauth.binding_digest(dataset=rel.instances_dataset, tier=rel.instances_tier, manifest_sha256=manifest,
                                       lo=0, hi=total, K=K, tree=n, schema=schemas[n]) for n in "aby"}
    dom = {n: CommitmentDomain(bind[n], auth.OWNER[n], RangeIndexedDomain(0, counts[n])) for n in "aby"}
    fv = FrameV3({"a": Port(dom["a"], RowLeaf("blake3", ROLE_X, word_bits, K)),
                  "b": Port(dom["b"], RowLeaf("blake3", ROLE_W, word_bits, K)), "y": Port(dom["y"], WordLeaf(ysch, ybytes))})
    try:
        import blake3

        def digest(row, role):
            from verity.commitments.rowleaf import blake3_row_key, row_bytes
            return blake3.blake3(row_bytes(row, word_bits), key=blake3_row_key(role)).digest()
    except ImportError:
        def digest(row, role):
            return blake3_row_digest(row, word_bits, role)
    y = [int(rel.y_public(int(a[-1]))) for a in accs]
    leaves = {"a": [fv.tree_leaf("a", r, digest(x[r].tolist(), ROLE_X)) for r in range(total)],
              "b": [fv.tree_leaf("b", r, digest(w[r].tolist(), ROLE_W)) for r in range(total)],
              "y": [fv.tree_leaf("y", r, fv.leaf_layout("y", r).digest(v.to_bytes(ybytes, "big"))) for r, v in enumerate(y)]}
    return {"roots": {n: fv.root(n, leaves[n]).hex() for n in "aby"},
            "domain_ids": {n: dom[n].domain_id.hex() for n in "aby"},
            "bindings": {n: bind[n].hex() for n in "aby"}, "owners": dict(auth.OWNER), "schemas": schemas,
            "y_bytes": ybytes, "y": y}


def write(path: str | Path, name: str, total: int, procs: int = 1, cache: str | Path | None = None) -> dict:
    """Write the set's first ``total`` VUs and their statement as a ``flock-pure-instances/v1`` file; returns the header."""
    from verity.ml.tc.cast import f32_to_bf16_word

    relation, _, _ = _ligero()
    rel = relation(name)
    pipe = PIPES[name]
    x, w, accs = draw(name, total, procs, cache)
    st = statement(name, x, w, accs)
    out = np.asarray([f32_to_bf16_word(int(a[-1])) if pipe.epilogue else int(a[-1]) for a in accs], dtype="<u4")
    header = {"format": FORMAT, "relation": name, "vus": total, "k": K, "word_bits": pipe.word_bits, "units": rel.steps,
              "row_bytes": K * pipe.word_bits // 8, "epilogue": pipe.epilogue, "instances": instances_ref(name, total),
              **{k: v for k, v in st.items() if k != "y"}}
    with open(path, "wb") as f:
        f.write(json.dumps(header, sort_keys=True).encode() + b"\n")
        f.write(np.ascontiguousarray(x.astype("<u2" if pipe.word_bits == 16 else "u1")).tobytes())
        f.write(np.ascontiguousarray(w.astype("<u2" if pipe.word_bits == 16 else "u1")).tobytes())
        f.write(np.ascontiguousarray(accs.astype("<u4")).tobytes())
        f.write(out.tobytes())
        f.write(np.asarray(st["y"], dtype="<u4").tobytes())
    return header


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("relation", choices=sorted(PIPES))
    ap.add_argument("total", type=int)
    ap.add_argument("out")
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--cache")
    a = ap.parse_args()
    h = write(a.out, a.relation, a.total, a.procs, a.cache)
    print(json.dumps({"relation": h["relation"], "vus": h["vus"], "roots": h["roots"],
                      "manifest_sha256": h["instances"]["manifest_sha256"]}))
