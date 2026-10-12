"""A frozen instance set's range as the Flock prover and verifier read it, with its frame-v3 statement commitments.

The instances are B-Ligero's synthetic sets (``archive/direct/ligero/relations.py``: the recipe the frozen
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
_DIRECT = Path(__file__).resolve().parents[5] / "archive" / "direct"


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
    from verity.ml.circuits.silicon.silicon import tc_dot
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
    from verity.ml.commitments.frame_v3 import FrameV3, Port, RowLeaf, WordLeaf
    from verity.ml.commitments.indexed import RangeIndexedDomain
    from verity.ml.commitments.merkle import CommitmentDomain
    from verity.ml.commitments.rowleaf import ROLE_W, ROLE_X, blake3_row_digest, sha256_row_digest

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
            from verity.ml.commitments.rowleaf import blake3_row_key, row_bytes
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
    from verity.ml.commitments import vllm_v1 as core

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
    ``scheme`` is the commitment scheme the header's roots are under: ``frame-v3`` or ``vllm-v1``. ``fp4-nvf4`` reads the frozen
    NVFP4 set (:func:`write_fp4`); a spine input set is :func:`write_set`."""
    from verity.ml.circuits.silicon.cast import f32_to_bf16_word

    if name == "fp4-nvf4":
        if scheme != "frame-v3":
            raise ValueError("fp4-nvf4 rows are frame-v3 NVFP4 row leaves")
        return write_fp4(path, total, lo, hi, leaf)

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


def _frame_v3(dataset: str, tier: str, manifest: str, lo: int, K_: int, schema_row: str, digests_a, digests_b, y, y_bits: int) -> dict:
    """frame-v3 roots over given row digests and output words, bound like :func:`statement` (``hashauth.binding_digest`` of the
    set, range, K, tree and schema)."""
    from verity.ml.commitments.frame_v3 import FrameV3, Port, WordLeaf
    from verity.ml.commitments.indexed import RangeIndexedDomain
    from verity.ml.commitments.merkle import CommitmentDomain, tree_levels

    _, hashauth, auth = _ligero()
    n = len(y)
    ysch, ybytes = hashauth.word_schema(y_bits)
    schemas = {"a": schema_row, "b": schema_row, "y": ysch}
    bind = {t: hashauth.binding_digest(dataset=dataset, tier=tier, manifest_sha256=manifest, lo=lo, hi=lo + n, K=K_, tree=t,
                                       schema=schemas[t]) for t in "aby"}
    dom = {t: CommitmentDomain(bind[t], auth.OWNER[t], RangeIndexedDomain(0, n)) for t in "aby"}
    fv = FrameV3({"y": Port(dom["y"], WordLeaf(ysch, ybytes))})
    leaves = {"a": [dom["a"].leaf(r, r, schema_row, d) for r, d in enumerate(digests_a)],
              "b": [dom["b"].leaf(r, r, schema_row, d) for r, d in enumerate(digests_b)],
              "y": [fv.tree_leaf("y", r, fv.leaf_layout("y", r).digest(int(v).to_bytes(ybytes, "big"))) for r, v in enumerate(y)]}
    return {"roots": {t: tree_levels(dom[t], leaves[t])[-1][0].hex() for t in "aby"},
            "domain_ids": {t: dom[t].domain_id.hex() for t in "aby"}, "bindings": {t: bind[t].hex() for t in "aby"},
            "owners": dict(auth.OWNER), "schemas": schemas, "y_bytes": ybytes, "y": [int(v) for v in y]}


def _write_file(path, header: dict, row_x: bytes, row_w: bytes, accs, out, y) -> dict:
    with open(path, "wb") as f:
        f.write(json.dumps(header, sort_keys=True).encode() + b"\n")
        f.write(row_x)
        f.write(row_w)
        f.write(np.ascontiguousarray(np.asarray(accs, dtype="<u4")).tobytes())
        f.write(np.asarray(out, dtype="<u4").tobytes())
        f.write(np.asarray(y, dtype="<u4").tobytes())
    return header


def _row_digests(rows_bytes: list[bytes], leaf: str, role: int) -> list[bytes]:
    from verity.ml.commitments.rowleaf import blake3_row_key
    if leaf == "sha256":
        raise ValueError("sha256 row digests need the row's word width: use _row_digests_words")
    import blake3
    key = blake3_row_key(role)
    return [blake3.blake3(b, key=key).digest() for b in rows_bytes]


def write_set(path: str | Path, relation: str, set_dir: str | Path, lo: int = 0, hi: int | None = None, leaf: str = "blake3",
              procs: int = 1, y_model: bool = False) -> dict:
    """VUs ``[lo, hi)`` of a spine input set (``vllm-vu-set/v1`` or ``input-set/v1``; its manifest's ports name the files:
    ``x.u16`` / ``w.u16`` with ``y.u16`` for BF16, ``x.u8`` / ``w.u8`` with ``y.u32``, the FP32 accumulator word, for E4M3)
    as a ``flock-pure-instances/v1`` file for ``relation`` (a :data:`lowering.PIPES` entry: its unit model chains each row, and
    the set's recorded y must equal the epilogue of the last accumulator, or the last accumulator itself; the committed y
    word is the relation's ``y_public`` of it). Rows are written per VU (the unshared statement,
    root's ruling 2026-09-26 02:30Z); the instance reference names the set and its ``content_digest``. ``y_model``: the set
    was captured on other silicon, so y is the relation's own chain output (a distinct tier; the reference counts the words
    that differ from the set's recorded y)."""
    from concurrent.futures import ProcessPoolExecutor
    from verity.ml.commitments.rowleaf import ROLE_W, ROLE_X, SCHEMA_BLAKE3_ROW, SCHEMA_SHA256_ROW, sha256_row_digest
    from verity.ml.circuits.silicon.cast import f32_to_bf16_word

    d = Path(set_dir)
    man = json.loads((d / "manifest.json").read_text())
    pipe = PIPES[relation]
    ports = man.get("ports") or {}
    dts = {"u8": "u1", "u16": "<u2", "u32": "<u4"}
    port = {r: (ports.get(r) or {"file": f"{r}.u16", "dtype": "u16"}) for r in "xwy"}
    wb = {"u1": 8, "<u2": 16}[dts[port["x"]["dtype"]]]
    if wb != pipe.word_bits or port["w"]["dtype"] != port["x"]["dtype"]:
        raise ValueError(f"{d.name}: {port['x']['dtype']} rows; {relation} takes {pipe.word_bits}-bit words")
    K_ = int(port["x"].get("words") or (man.get("relation") or {}).get("statics", {}).get("K"))
    x_all = np.fromfile(d / port["x"]["file"], dtype=dts[port["x"]["dtype"]]).reshape(-1, K_)
    n_set = x_all.shape[0]
    hi = n_set if hi is None else hi
    if hi > n_set:
        raise ValueError(f"{d.name} holds {n_set} instances; [{lo}, {hi}) would repeat instances")
    x = x_all[lo:hi]
    w = np.fromfile(d / port["w"]["file"], dtype=dts[port["w"]["dtype"]]).reshape(-1, K_)[lo:hi]
    y_rec = np.fromfile(d / port["y"]["file"], dtype=dts[port["y"]["dtype"]])[lo:hi]
    step = max(1, len(x) // max(1, 4 * procs))
    jobs = [(relation, x[i:i + step], w[i:i + step]) for i in range(0, len(x), step)]
    if procs > 1:
        with ProcessPoolExecutor(procs) as ex:
            accs = np.asarray([r for part in ex.map(_chain_rows, jobs) for r in part], dtype=np.uint32)
    else:
        accs = np.asarray([r for j in jobs for r in _chain_rows(j)], dtype=np.uint32)
    final = [f32_to_bf16_word(int(a[-1])) for a in accs] if pipe.epilogue else [int(a[-1]) for a in accs]
    bad = sum(int(f != int(v)) for f, v in zip(final, y_rec))
    if bad and not y_model:
        raise ValueError(f"{d.name}: {bad} recorded outputs differ from the {relation} model chain")
    digest = man.get("content_digest")
    if leaf == "sha256":
        da = [sha256_row_digest(r.tolist(), wb, ROLE_X) for r in x]
        db = [sha256_row_digest(r.tolist(), wb, ROLE_W) for r in w]
        schema = SCHEMA_SHA256_ROW
    else:
        da = _row_digests([np.ascontiguousarray(r).tobytes() for r in x], leaf, ROLE_X)
        db = _row_digests([np.ascontiguousarray(r).tobytes() for r in w], leaf, ROLE_W)
        schema = SCHEMA_BLAKE3_ROW
    dataset, tier = man.get("set", d.name), f"{man.get('schema', 'input-set/v1')}:{man.get('set', d.name)}"
    if y_model:
        tier += f":y={relation}"
    rel = _ligero()[0](relation) if not pipe.epilogue else None
    y_pub = [int(rel.y_public(v)) for v in final] if rel is not None else final
    st = _frame_v3(dataset, tier, digest, lo, K_, schema, da, db, y_pub, rel.y_bits if rel is not None else 16)
    ref = {"dataset": dataset, "tier": tier, "range": [lo, hi], "manifest_sha256": digest, "set_schema": man.get("schema"),
           "source": (man.get("provenance") or {}).get("source") or ("captured" if man.get("schema") == "vllm-vu-set/v1" else None),
           **({"y": {"source": f"{relation} model chain", "differ_from_set": bad}} if y_model else {})}
    if y_model:
        ref["source"] = f"{ref['source'] or 'set'} inputs, model y"
    header = {"format": FORMAT, "relation": relation, "vus": hi - lo, "k": K_, "word_bits": wb, "units": accs.shape[1],
              "row_bytes": wb * K_ // 8, "epilogue": pipe.epilogue, "instances": ref, **{k: v for k, v in st.items() if k != "y"}}
    return _write_file(path, header, np.ascontiguousarray(x).tobytes(), np.ascontiguousarray(w).tobytes(), accs, final, st["y"])


def write_synth(path: str | Path, relation: str, K_: int, lo: int, hi: int, leaf: str = "blake3", procs: int = 1) -> dict:
    """VUs ``[lo, hi)`` of the synthetic E4M3 stream at ``K_`` words (``rng([seed, i])``, two draws per VU, as
    :func:`_draw`) for an 8-bit relation with no spine input set, chained with its unit model. The manifest names the
    recipe, not a count, so every sub-batch of a batch binds the same one."""
    from concurrent.futures import ProcessPoolExecutor
    from verity.ml.commitments.rowleaf import ROLE_W, ROLE_X, SCHEMA_BLAKE3_ROW

    if leaf != "blake3":
        raise ValueError("the chunked 8-bit layouts commit keyed-BLAKE3 rows")
    rel = _ligero()[0](relation)
    pipe = PIPES[relation]
    if pipe.word_bits != 8:
        raise ValueError(f"{relation}: the synthetic stream at any K is the E4M3 one")
    seed = rel.instance_seed
    xs, ws = [], []
    for i in range(lo, hi):
        rng = np.random.default_rng([seed, i])
        xs.append(_e4m3_operands(rng, (K_,)))
        ws.append(_e4m3_operands(rng, (K_,)))
    x, w = np.asarray(xs, dtype=np.uint8), np.asarray(ws, dtype=np.uint8)
    step = max(1, len(x) // max(1, 4 * procs))
    jobs = [(relation, x[i:i + step], w[i:i + step]) for i in range(0, len(x), step)]
    if procs > 1:
        with ProcessPoolExecutor(procs) as ex:
            accs = np.asarray([r for part in ex.map(_chain_rows, jobs) for r in part], dtype=np.uint32)
    else:
        accs = np.asarray([r for j in jobs for r in _chain_rows(j)], dtype=np.uint32)
    out = [int(a[-1]) for a in accs]
    y = [int(rel.y_public(v)) for v in out]
    dataset, tier = f"flock-synthetic-{relation}-k{K_}", f"synthetic-e4m3-k{K_}"
    manifest = hashlib.sha256(f"{relation} synthetic e4m3|seed={seed}|K={K_}|rng([seed, i]), x then w".encode()).hexdigest()
    da = _row_digests([r.tobytes() for r in x], leaf, ROLE_X)
    db = _row_digests([r.tobytes() for r in w], leaf, ROLE_W)
    st = _frame_v3(dataset, tier, manifest, lo, K_, SCHEMA_BLAKE3_ROW, da, db, y, rel.y_bits)
    ref = {"dataset": dataset, "tier": tier, "range": [lo, hi], "manifest_sha256": manifest, "seed": seed, "source": "synthetic"}
    header = {"format": FORMAT, "relation": relation, "vus": hi - lo, "k": K_, "word_bits": 8, "units": accs.shape[1],
              "row_bytes": K_, "epilogue": False, "instances": ref, **{k: v for k, v in st.items() if k != "y"}}
    return _write_file(path, header, x.tobytes(), w.tobytes(), accs, out, st["y"])


def _chain_rows(args):
    relation, x, w = args
    from verity.ml.circuits.silicon.silicon import tc_dot
    from .lowering import _model
    pipe = PIPES[relation]
    model, k = _model(pipe.model), pipe.k
    out = []
    for a, b in zip(x, w):
        al, bl, c, row = a.tolist(), b.tolist(), 0, []
        for s in range(len(al) // k):
            c = tc_dot(model, c, al[k * s:k * s + k], bl[k * s:k * s + k])
            row.append(c)
        out.append(row)
    return out


def write_fp4(path: str | Path, total: int, lo: int = 0, hi: int | None = None, leaf: str = "blake3") -> dict:
    """VUs ``[lo, hi)`` of the frozen NVFP4 set (``bench-instances-nvfp4-sm120/v1``: ``ligero.fp4.chain.instances_fp4``, seed
    20260922, model-recorded accumulators) as ``flock-pure-instances/v1``: 864-byte ``nvfp4_row_bytes`` rows under
    ``blake3-keyed/row-nvfp4/v1`` or ``sha256/row-nvfp4/v1`` and a u32 word leaf of the final FP32 accumulator (``out`` = y)."""
    import types
    sys.modules.setdefault("torch", types.ModuleType("torch"))
    _ligero()
    from ligero.fp4.chain import INSTANCE_SEED, instances_digest as fp4_digest, instances_fp4
    from verity.ml.commitments.rowleaf import (ROLE_W, ROLE_X, SCHEMA_BLAKE3_ROW_NVFP4, SCHEMA_SHA256_ROW_NVFP4, blake3_row_nvfp4_digest,
                                            nvfp4_row_bytes, sha256_row_nvfp4_digest)
    from verity_numerical.bench import contract

    hi = total if hi is None else hi
    data = instances_fp4(hi - lo, start=lo)
    rows = {0: [], 1: []}
    codes = {0: [], 1: []}
    for A, B, _, _ in data:
        for r, M in ((0, A), (1, B)):
            c = [int(v) for s in M for v in s[:64]]
            sc = [int(v) for s in M for v in s[64:68]]
            codes[r].append((c, sc))
            rows[r].append(nvfp4_row_bytes(c, sc))
    accs = np.asarray([dd[2] for dd in data], dtype="<u4")
    y = [int(a[-1]) for a in accs]
    schema, dig = ((SCHEMA_SHA256_ROW_NVFP4, sha256_row_nvfp4_digest) if leaf == "sha256" else
                   (SCHEMA_BLAKE3_ROW_NVFP4, blake3_row_nvfp4_digest))
    da = [dig(c, sc, ROLE_X) for c, sc in codes[0]]
    db = [dig(c, sc, ROLE_W) for c, sc in codes[1]]
    manifest = fp4_digest(total)
    st = _frame_v3(contract.NVFP4_INSTANCES_DATASET, contract.NVFP4_INSTANCES_TIER, manifest, lo, K, schema, da, db, y, 32)
    ref = {"dataset": contract.NVFP4_INSTANCES_DATASET, "tier": contract.NVFP4_INSTANCES_TIER, "range": [lo, hi],
           "manifest_sha256": manifest, "seed": INSTANCE_SEED}
    header = {"format": FORMAT, "relation": "fp4-nvf4", "vus": hi - lo, "k": K, "word_bits": 4, "units": 24, "row_bytes": 864,
              "epilogue": False, "instances": ref, **{k: v for k, v in st.items() if k != "y"}}
    return _write_file(path, header, b"".join(rows[0]), b"".join(rows[1]), accs, y, y)


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
    ap.add_argument("--set", default=None, help="a spine input set directory (x.u16 / w.u16 / y.u16): its first TOTAL VUs")
    ap.add_argument("--y-model", action="store_true", help="with --set: y is the relation's chain output, not the set's")
    ap.add_argument("--k", type=int, default=0, help="an 8-bit relation's synthetic stream at this K (write_synth)")
    a = ap.parse_args()
    h = (write_set(a.out, a.relation, a.set, 0, a.total, a.leaf, a.procs, a.y_model) if a.set else
         write_synth(a.out, a.relation, a.k, 0, a.total, a.leaf, a.procs) if a.k else
         write(a.out, a.relation, a.total, a.procs, a.cache, scheme=a.scheme, leaf=a.leaf))
    print(json.dumps({"relation": h["relation"], "vus": h["vus"], "roots": h["roots"],
                      "manifest_sha256": h["instances"]["manifest_sha256"]}))
