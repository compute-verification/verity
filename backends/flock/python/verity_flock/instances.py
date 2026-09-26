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


def _frozen(rel):
    """The relation's frozen tier (``bench-instances/v1`` ``vu-k1536`` for the A100 row), or None for a synthetic stream."""
    if not rel.frozen_tier:
        return None
    from verity_numerical.bench import instances as frozen
    return frozen


def instances_digest(name: str, n: int) -> str:
    relation, _, _ = _ligero()
    rel = relation(name)
    fz = _frozen(rel)
    if fz is not None:
        return fz.manifest_sha256(fz.DEFAULT_ROOT)
    return hashlib.sha256(f"{rel.name} synthetic|seed={rel.instance_seed}|n={n}|K={K}".encode()).hexdigest()


def instances_ref(name: str, total: int) -> dict:
    relation, _, _ = _ligero()
    rel = relation(name)
    if rel.frozen_tier:
        return {"dataset": rel.instances_dataset, "tier": rel.instances_tier, "range": [0, total],
                "manifest_sha256": instances_digest(name, total)}
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


def draw(name: str, total: int, procs: int = 1, cache: str | Path | None = None, lo: int = 0, hi: int | None = None):
    """``(x, w, accs)`` of the set's VUs ``[lo, hi)`` (default: the first ``total``)."""
    relation, _, _ = _ligero()
    rel = relation(name)
    hi = total if hi is None else hi
    fz = _frozen(rel)
    if fz is not None:
        # the frozen set: its recorded x rows, W columns and accumulators; it has no stream beyond its instances
        vu = fz.load(rel.frozen_tier, fz.DEFAULT_ROOT)
        if hi > vu.n:
            raise ValueError(f"{rel.frozen_tier} holds {vu.n} instances; [{lo}, {hi}) would repeat instances")
        K_ = vu.K
        x = np.frombuffer(vu.x, dtype=np.uint16).reshape(vu.n, K_)[lo:hi]
        w = np.frombuffer(vu.w, dtype=np.uint16).reshape(vu.n, K_)[lo:hi]
        accs = np.frombuffer(vu.acc, dtype=np.uint32).reshape(vu.n, rel.steps)[lo:hi]
        return x, w, accs
    path = Path(cache) / f"flock-{name}-{rel.instance_seed}-{lo}-{hi}.npz" if cache else None
    if path is not None and path.exists():
        with np.load(path) as z:
            return z["x"], z["w"], z["accs"]
    step = max(1, (hi - lo) // max(1, 8 * procs))
    jobs = [(name, rel.instance_seed, a, min(hi, a + step)) for a in range(lo, hi, step)]
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


SCHEMES = {"blake3": "blake3-keyed/row/v2", "sha256": "sha256/row/v1"}


def statement(name: str, x, w, accs, total: int | None = None, lo: int = 0, leaf: str = "blake3") -> dict:
    """The three frame-v3 roots (and their domains) over the set's VUs ``[lo, lo + len(x))`` of a ``total``-VU batch (one
    sub-batch's trees, bound to that range)."""
    from verity.commitments.frame_v3 import FrameV3, Port, RowLeaf, WordLeaf
    from verity.commitments.indexed import RangeIndexedDomain
    from verity.commitments.merkle import CommitmentDomain
    from verity.commitments.rowleaf import ROLE_W, ROLE_X, blake3_row_digest, sha256_row_digest

    relation, hashauth, auth = _ligero()
    rel = relation(name)
    n = len(x)
    total = n if total is None else total
    word_bits = 16 if rel.word_dtype == "<u2" else 8
    ysch, ybytes = hashauth.word_schema(rel.y_bits)
    manifest = instances_digest(name, total)
    counts = {"a": n, "b": n, "y": n}
    schemas = {"a": SCHEMES[leaf], "b": SCHEMES[leaf], "y": ysch}
    bind = {t: hashauth.binding_digest(dataset=rel.instances_dataset, tier=rel.instances_tier, manifest_sha256=manifest,
                                       lo=lo, hi=lo + n, K=K, tree=t, schema=schemas[t]) for t in "aby"}
    dom = {n: CommitmentDomain(bind[n], auth.OWNER[n], RangeIndexedDomain(0, counts[n])) for n in "aby"}
    fv = FrameV3({"a": Port(dom["a"], RowLeaf(leaf, ROLE_X, word_bits, K)),
                  "b": Port(dom["b"], RowLeaf(leaf, ROLE_W, word_bits, K)), "y": Port(dom["y"], WordLeaf(ysch, ybytes))})
    if leaf == "sha256":
        def digest(row, role):
            return sha256_row_digest(row, word_bits, role)
    else:
      try:
        import blake3

        def digest(row, role):
            from verity.commitments.rowleaf import blake3_row_key, row_bytes
            return blake3.blake3(row_bytes(row, word_bits), key=blake3_row_key(role)).digest()
      except ImportError:
        def digest(row, role):
            return blake3_row_digest(row, word_bits, role)
    y = [int(rel.y_public(int(a[-1]))) for a in accs]
    leaves = {"a": [fv.tree_leaf("a", r, digest(x[r].tolist(), ROLE_X)) for r in range(n)],
              "b": [fv.tree_leaf("b", r, digest(w[r].tolist(), ROLE_W)) for r in range(n)],
              "y": [fv.tree_leaf("y", r, fv.leaf_layout("y", r).digest(v.to_bytes(ybytes, "big"))) for r, v in enumerate(y)]}
    return {"roots": {n: fv.root(n, leaves[n]).hex() for n in "aby"},
            "domain_ids": {n: dom[n].domain_id.hex() for n in "aby"},
            "bindings": {n: bind[n].hex() for n in "aby"}, "owners": dict(auth.OWNER), "schemas": schemas,
            "y_bytes": ybytes, "y": y}


def statement_vllm_v1(name: str, x, w, accs, total: int | None = None, lo: int = 0) -> dict:
    """The three ``vllm-v1`` roots over the set's VUs ``[lo, lo + len(x))``: position-leaf trees of the x rows, the W
    columns and the output words' ``y_bytes`` little-endian bytes, each root the step-root binding of its port's
    ``StepDomain`` (PROTOCOL.md §4, §5). The domains are B-Ligero's (``ligero.vllm_tree``: ctx = the set and range per
    port), so a Flock cell and a B-Ligero ``+vllm-v1`` cell on the same line bind the same three roots."""
    from verity.commitments import vllm_v1 as core

    relation, _, _ = _ligero()
    from ligero import vllm_tree
    rel = relation(name)
    n = len(x)
    total = n if total is None else total
    word_bits = 16 if rel.word_dtype == "<u2" else 8
    yb = vllm_tree.y_bytes_of(rel.y_bits)
    manifest = instances_digest(name, total)
    y = [int(rel.y_public(int(a[-1]))) for a in accs]
    rb = "<u2" if word_bits == 16 else "u1"
    values = {"a": [np.ascontiguousarray(r.astype(rb)).tobytes() for r in x],
              "b": [np.ascontiguousarray(r.astype(rb)).tobytes() for r in w],
              "y": [v.to_bytes(yb, "little") for v in y]}
    roots, digests, ctx = {}, {}, {}
    for t in "aby":
        ctx[t] = vllm_tree.ctx_digest(dataset=rel.instances_dataset, tier=rel.instances_tier, manifest_sha256=manifest, lo=lo,
                                      hi=lo + n, tree=t)
        dom = vllm_tree.domain(t, ctx[t], n, K=K, word_bits=word_bits, y_bytes=yb)
        roots[t] = dom.bind(core.fold([core.pos_leaf(v) for v in values[t]]))
        digests[t] = core.domain_digest(dom)
    return {"scheme": core.NAME, "roots": {t: roots[t].hex() for t in "aby"}, "domain_digests": {t: digests[t].hex() for t in "aby"},
            "ctx": {t: ctx[t].hex() for t in "aby"}, "leaf": "pos-leaf", "y_bytes": yb, "y": y}


def write(path: str | Path, name: str, total: int, procs: int = 1, cache: str | Path | None = None, lo: int = 0,
          hi: int | None = None, scheme: str = "frame-v3", leaf: str = "blake3") -> dict:
    """Write VUs ``[lo, hi)`` (default all) of the set's first ``total`` and their statement as a ``flock-pure-instances/v1``
    file; returns the header. A sub-batch's instance reference keeps the whole batch's manifest and names its own range.
    ``scheme`` is the commitment scheme the header's roots are under: ``frame-v3`` or ``vllm-v1``."""
    from verity.ml.tc.cast import f32_to_bf16_word

    relation, _, _ = _ligero()
    rel = relation(name)
    pipe = PIPES[name]
    hi = total if hi is None else hi
    x, w, accs = draw(name, total, procs, cache, lo, hi)
    st = statement(name, x, w, accs, total, lo, leaf) if scheme == "frame-v3" else statement_vllm_v1(name, x, w, accs, total, lo)
    out = np.asarray([f32_to_bf16_word(int(a[-1])) if pipe.epilogue else int(a[-1]) for a in accs], dtype="<u4")
    ref = instances_ref(name, total)
    ref["range"] = [lo, hi]
    header = {"format": FORMAT, "relation": name, "vus": hi - lo, "k": K, "word_bits": pipe.word_bits, "units": rel.steps,
              "row_bytes": K * pipe.word_bits // 8, "epilogue": pipe.epilogue, "instances": ref,
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
    ap.add_argument("--scheme", choices=["frame-v3", "vllm-v1"], default="frame-v3")
    ap.add_argument("--leaf", choices=sorted(SCHEMES), default="blake3", help="frame-v3 row-leaf schema of trees a and b")
    a = ap.parse_args()
    h = write(a.out, a.relation, a.total, a.procs, a.cache, scheme=a.scheme, leaf=a.leaf)
    print(json.dumps({"relation": h["relation"], "vus": h["vus"], "roots": h["roots"],
                      "manifest_sha256": h["instances"]["manifest_sha256"]}))
