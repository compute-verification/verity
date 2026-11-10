"""A spine input set's range (`input_sets`' layout) as the instance file the Flock prover and verifier read, with its frame-v3
statement commitments (``templates/gemm_coordinate.py``'s ``stage``).

The statement commits the set as B-Ligero's ``+blake3`` configuration did: tree ``a`` over the x rows and ``b`` over the W
columns (``blake3-keyed/row/v2`` row leaves), tree ``y`` over the output words (a word leaf each: the BF16 word of an
epilogue relation, else :func:`y_word`'s word of the final accumulator), each under the :func:`binding_digest` domain of
the set and range.

File (``flock-pure-instances/v1``): one JSON header line, then x rows, W columns (``row_bytes`` each, words
little-endian), the accumulators (``units`` u32 LE per VU), the circuit's output word per VU (u32 LE: the BF16 word for
an epilogue relation, else the FP32 accumulator) and the committed y word per VU (u32 LE).
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .lowering import PIPES

FORMAT = "flock-pure-instances/v1"
#: the frame-v3 binding tag and tree owners of B-Ligero's ``+blake3`` statements, which the frozen cells' roots are under
BINDING_TAG = "verity/ligero-b/auth-binding/v2h"
OWNER = {"a": -1, "b": -2, "y": -1}
SCHEMES = {"blake3": "blake3-keyed/row/v2", "sha256": "sha256/row/v1"}


def _fp8_pack(y: int) -> int:
    """The FP8 relations' chain-end word: the final FP32 accumulator with its low 10 bits, zero in every honest word, dropped."""
    if y & (2**10 - 1):
        raise ValueError("claimed FP32 word has nonzero low bits: not a tensor-core accumulator")
    return y >> 10


#: the committed chain-end word of each relation without the epilogue: its width and its word of the final FP32 accumulator
Y_WORD: dict[str, tuple[int, Callable[[int], int]]] = {"fp8-ada": (22, _fp8_pack), "fp8-hopper": (22, _fp8_pack)}


def y_word(relation: str) -> tuple[int, Callable[[int], int]]:
    try:
        return Y_WORD[relation]
    except KeyError:
        raise SystemExit(f"no committed y word for relation {relation!r}; known: {sorted(Y_WORD)}") from None


def word_schema(y_bits: int) -> tuple[str, int]:
    if y_bits <= 16:
        return "u16", 2
    if y_bits <= 32:
        return "u32", 4
    raise ValueError("output words wider than 32 bits")


def binding_digest(*, dataset: str, tier: str, manifest_sha256: str, lo: int, hi: int, K: int, tree: str, schema: str) -> bytes:
    """Identity of the committed object: the frozen input set (by manifest digest), which rows of it, the leaf schema."""
    from verity.commitments import identity_digest

    if tree not in OWNER:
        raise ValueError(f"unknown tree {tree!r}")
    return bytes.fromhex(identity_digest(BINDING_TAG, {"dataset": dataset, "tier": tier, "manifest_sha256": manifest_sha256,
                                                       "lo": int(lo), "hi": int(hi), "K": int(K), "tree": tree, "schema": schema}))


def _frame_v3(dataset: str, tier: str, manifest: str, lo: int, K_: int, schema_row: str, digests_a, digests_b, y, y_bits: int) -> dict:
    """frame-v3 roots over given row digests and output words, each tree bound by :func:`binding_digest` of the set, range, K,
    tree and schema."""
    from verity.commitments.frame_v3 import FrameV3, Port, WordLeaf
    from verity.commitments.indexed import RangeIndexedDomain
    from verity.commitments.merkle import CommitmentDomain, tree_levels

    n = len(y)
    ysch, ybytes = word_schema(y_bits)
    schemas = {"a": schema_row, "b": schema_row, "y": ysch}
    bind = {t: binding_digest(dataset=dataset, tier=tier, manifest_sha256=manifest, lo=lo, hi=lo + n, K=K_, tree=t,
                                       schema=schemas[t]) for t in "aby"}
    dom = {t: CommitmentDomain(bind[t], OWNER[t], RangeIndexedDomain(0, n), hash="sha256") for t in "aby"}
    fv = FrameV3({"y": Port(dom["y"], WordLeaf(ysch, ybytes))})
    leaves = {"a": [dom["a"].leaf(r, r, schema_row, d) for r, d in enumerate(digests_a)],
              "b": [dom["b"].leaf(r, r, schema_row, d) for r, d in enumerate(digests_b)],
              "y": [fv.tree_leaf("y", r, fv.leaf_layout("y", r).digest(int(v).to_bytes(ybytes, "big"))) for r, v in enumerate(y)]}
    return {"roots": {t: tree_levels(dom[t], leaves[t])[-1][0].hex() for t in "aby"},
            "domain_ids": {t: dom[t].domain_id.hex() for t in "aby"}, "bindings": {t: bind[t].hex() for t in "aby"},
            "owners": dict(OWNER), "schemas": schemas, "y_bytes": ybytes, "y": [int(v) for v in y]}


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
    from verity.commitments.rowleaf import blake3_row_key
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
    word is the relation's :func:`y_word` of it). Rows are written per VU (the unshared
    statement); the instance reference names the set and its ``content_digest``. ``y_model``: the set
    was captured on other silicon, so y is the relation's own chain output (a distinct tier; the reference counts the words
    that differ from the set's recorded y)."""
    from concurrent.futures import ProcessPoolExecutor
    from verity.commitments.rowleaf import ROLE_W, ROLE_X, SCHEMA_BLAKE3_ROW, SCHEMA_SHA256_ROW, sha256_row_digest
    from verity.definitions.silicon.cast import f32_to_bf16_word

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
    y_bits, y_public = (16, None) if pipe.epilogue else y_word(relation)
    y_pub = [int(y_public(v)) for v in final] if y_public is not None else final
    st = _frame_v3(dataset, tier, digest, lo, K_, schema, da, db, y_pub, y_bits)
    ref = {"dataset": dataset, "tier": tier, "range": [lo, hi], "manifest_sha256": digest, "set_schema": man.get("schema"),
           "source": (man.get("provenance") or {}).get("source") or ("captured" if man.get("schema") == "vllm-vu-set/v1" else None),
           **({"y": {"source": f"{relation} model chain", "differ_from_set": bad}} if y_model else {})}
    if y_model:
        ref["source"] = f"{ref['source'] or 'set'} inputs, model y"
    header = {"format": FORMAT, "relation": relation, "vus": hi - lo, "k": K_, "word_bits": wb, "units": accs.shape[1],
              "row_bytes": wb * K_ // 8, "epilogue": pipe.epilogue, "instances": ref, **{k: v for k, v in st.items() if k != "y"}}
    return _write_file(path, header, np.ascontiguousarray(x).tobytes(), np.ascontiguousarray(w).tobytes(), accs, final, st["y"])


def _chain_rows(args):
    relation, x, w = args
    from verity.definitions.silicon.models import tc_dot
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
