"""Operand commitment ("A-GKR, operands committed"): the statement publishes, per VU, the row digests of its x row
and W column, and the verifier checks them natively against the commitment scheme's trees
(``verifier/src/commitments.rs``; scheme ``frame-v3`` of ``verity.ml.commitments``, row leaves ``sha256/row/v1`` or
``blake3-keyed/row/v2``).  Claimed as ``R+sha256`` / ``R+blake3``.

Statement files (beside the Params-only files and ``public.bin``):

  commitment.txt  scheme, leaf, relation, the input set (dataset, tier, manifest digest), lo, vus, K, the row
                  word widths, the y word schema and count per VU (format: ``commitments.rs``); its SHA-256 is
                  absorbed into the transcript after the binding digests (``prover.Chain.commit_digest``)
  epilogue.txt    the export's epilogue with 32 more columns ``dx0..dx15 dw0..dw15``: the x and W digests as 16-bit
                  limbs (limb ``i`` = digest bytes ``2i, 2i+1`` big-endian)
  chain.txt       its ``public`` line extended by those 32 columns, so each VU's public words are its y words then
                  the two digests

The domains are B-Ligero's (``archive/direct/ligero/hashauth.py``: same binding tag and fields, owners ``a = -1``,
``b = -2``, ``y = -1``, one row per VU at rank = VU index), so for the same input set and leaf schema the roots
are B-Ligero's.  Until the hash layers bind the digest columns to the operand columns, nothing in the circuit
constrains them: the relation names have no circuit pin and the statement verifies only under
``--allow-any-circuit`` (an altered operand with honest digests is accepted: the gap the hash layers close).

Scheme ``vllm-v1`` (``--leaf vllm-v1``, claimed as ``R+vllm-v1``; ``verifier/src/vllm_v1.rs``): the published digests
are ``pos_leaf(row bytes)``, which are the tree leaves themselves; each port's root is a ``StepDomain(chunk=None)``
binding of the unpadded fold, and ``commitment.txt`` names each port's ``domain_digest``.  The program / ctx / geo /
layout digests of those domains (``vllm_domain``) are this backend's provisional mapping; the integration owns them.

    python -m gpu.commit pin RELATION --leaf sha256|blake3|vllm-v1 [--lo 0] [--vus 4096] [--instances DIR]
    python -m gpu.commit pin bf16-ampere-k2048 --leaf blake3 --input-set SET --vus 4096     # VUs [0, 4096) of an input set
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEME = "frame-v3"
VLLM = "vllm-v1"
LIMBS = 16
LEAVES = ("sha256", "blake3", VLLM)
SUFFIX = {"sha256": "+sha256", "blake3": "+blake3", VLLM: "+vllm-v1"}
VLLM_MAP_TAG = "verity/gkr-commit/vllm-v1"      # vllm_v1.rs MAP_TAG
PORTS = ("a", "b", "y")
DIGEST_COLS = [f"dx{i}" for i in range(LIMBS)] + [f"dw{i}" for i in range(LIMBS)]
OWNER = {"a": -1, "b": -2, "y": -1}
BINDING_TAG = "verity/ligero-b/auth-binding/v2h"   # hashauth.BINDING_TAG
BLAKE3_CHUNK = 1024


@dataclass(frozen=True)
class InstanceSet:
    dataset: str
    tier: str
    manifest_sha256: str
    K: int
    word_bits: int


def instance_set(relation: str, lo: int, vus: int, src: Path, frozen_manifest: Path | None = None, nvf4_k: int | None = None) -> InstanceSet:
    """What B-Ligero's commitment binds for ``relation``'s frozen set ``[lo, lo + vus)`` (``relchain.commit_vus``).
    fp4-nvf4 rows are ``x.bin``'s (``nvf4_k`` bytes per VU), not B-Ligero's lane rows: its roots are not B-Ligero's."""
    import sys

    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    if relation == "fp4-nvf4":
        from archive.direct.ligero.fp4 import chain as fc

        if nvf4_k is None:
            raise ValueError("commit: fp4-nvf4 needs its bytes per VU")
        ref = fc._instances(lo + vus)
        return InstanceSet(ref["dataset"], ref["tier"], ref["manifest_sha256"], int(nvf4_k), 8)
    from archive.direct.ligero import relchain
    from archive.direct.ligero.relations import relation as get_relation

    rel = get_relation(relation)
    if rel.frozen_tier:
        import hashlib

        man = frozen_manifest or src / "fixtures" / "bench-instances" / "v1" / "manifest.json"
        digest = hashlib.sha256(Path(man).read_bytes()).hexdigest()
    else:
        digest = relchain.instances_digest(rel, lo + vus)
    bits = {"<u2": 16, "<u1": 8}[rel.word_dtype]
    return InstanceSet(rel.instances_dataset, rel.instances_tier, digest, int(getattr(rel, "vu_words", relchain.K_VU)), bits)


def input_set(path, n: int, word_bits: int = 16):
    """``(x, w, y, iset, ref)``: VUs ``[0, n)`` of an input set (``verity_numerical.bench.input_sets``; a set is never recycled) as
    ``(n, K)`` rows and ``(n,)`` words, with the identity B-Ligero's ``commit_vus`` binds for a set (``relchain.set_ref``: dataset
    ``<schema>:<set>``, tier ``<set>``, the manifest's SHA-256), so the roots are B-Ligero's for the same set and leaf schema, and
    the fingerprint's ``instances`` entry."""
    from verity_numerical.bench.input_sets import InputSet

    s = InputSet.open(path)
    if not 0 < n <= s.n:
        raise ValueError(f"commit: input set {s.name} holds {s.n} VUs, not {n} (a set is not recycled)")
    ins = s.instances(0, n)
    dt = {8: np.uint8, 16: np.uint16}[word_bits]
    x, w = (np.ascontiguousarray(ins[p], dtype=dt) for p in ("x", "w"))
    y = np.asarray(ins["y"]).reshape(n, -1)[:, 0]
    ref = s.ref(0, n)
    return x, w, y, InstanceSet(ref["dataset"], ref["tier"], ref["manifest_sha256"], int(s.subcircuit.params["K"]), word_bits), ref


def binding(iset: InstanceSet, lo: int, hi: int, tree: str, schema: str) -> bytes:
    from verity.ml.commitments import identity_digest

    return bytes.fromhex(identity_digest(BINDING_TAG, {"dataset": iset.dataset, "tier": iset.tier, "manifest_sha256": iset.manifest_sha256,
                                                       "lo": int(lo), "hi": int(hi), "K": int(iset.K), "tree": tree, "schema": schema}))


def word_schema(y_width: int) -> str:
    return {2: "u16", 4: "u32"}[y_width]


def row_bytes(rows: np.ndarray, word_bits: int) -> np.ndarray:
    """``(n, K)`` words -> ``(n, K·word_bits/8)`` uint8, little-endian (``rowleaf.row_bytes``); refuses a word too wide."""
    rows = np.asarray(rows).astype(np.int64)
    if rows.min(initial=0) < 0 or rows.max(initial=0) >> word_bits:
        raise ValueError(f"commit: a row word exceeds {word_bits} bits")
    dt = {8: "<u1", 16: "<u2", 32: "<u4"}[word_bits]
    return np.ascontiguousarray(rows.astype(dt)).view(np.uint8).reshape(rows.shape[0], -1)


def row_digests(rows: np.ndarray, word_bits: int, role: int, leaf: str) -> list[bytes]:
    from verity.ml.commitments.frame_v3 import RowLeaf

    raw = row_bytes(rows, word_bits)
    layout = RowLeaf(leaf, role, word_bits, int(np.asarray(rows).shape[1])).layout()
    return [layout.digest(r.tobytes()) for r in raw]


def limbs(digest: bytes) -> list[int]:
    return [int.from_bytes(digest[2 * i:2 * i + 2], "big") for i in range(LIMBS)]


def vllm_domain(base_relation: str, iset: InstanceSet, lo: int, vus: int, n_y: int, y_width: int, port: str):
    """``vllm_v1.rs operand_domain``: the provisional step domain of port ``a`` / ``b`` / ``y``."""
    from verity.ml.commitments import identity_digest
    from verity.ml.commitments.vllm_v1 import StepDomain

    def d(kind: str, doc: dict) -> bytes:
        return bytes.fromhex(identity_digest(f"{VLLM_MAP_TAG}/{kind}/v0", doc))

    value, bits, words, count = {"a": ("x-row", iset.word_bits, iset.K, vus), "b": ("w-column", iset.word_bits, iset.K, vus),
                                 "y": ("y-word", 8 * y_width, 1, vus * n_y)}[port]
    return StepDomain(
        program_digest=d("program", {"relation": base_relation}),
        ctx_digest=d("ctx", {"dataset": iset.dataset, "tier": iset.tier, "manifest_sha256": iset.manifest_sha256, "lo": int(lo),
                             "hi": int(lo + vus), "port": port}),
        geo_digest=d("geo", {"vus": int(vus), "K": int(iset.K), "n_y": int(n_y)}),
        layout_digest=d("layout", {"port": port, "value": value, "word_bits": int(bits), "words": int(words), "byte_order": "little"}),
        leaf_count=int(count))


@dataclass
class Commitment:
    leaf: str
    relation: str
    iset: InstanceSet
    lo: int
    vus: int
    y_width: int
    n_y: int
    digests: dict            # "a" / "b": per-VU row digests
    roots: dict              # "a" / "b" / "y": bytes
    domains: dict | None = None   # vllm-v1: "a" / "b" / "y" domain_digest

    @property
    def scheme(self) -> str:
        return VLLM if self.leaf == VLLM else SCHEME

    def text(self) -> str:
        i = self.iset
        leaf = "pos" if self.leaf == VLLM else self.leaf
        out = (f"scheme {self.scheme}\nleaf {leaf}\nrelation {self.relation}\ninstances {i.dataset} {i.tier} {i.manifest_sha256}\n"
               f"lo {self.lo}\nvus {self.vus}\nK {i.K}\nx {i.word_bits}\nw {i.word_bits}\ny {word_schema(self.y_width)} {self.n_y}\n")
        if self.domains:
            out += "".join(f"domain_{p} {self.domains[p].hex()}\n" for p in PORTS)
        return out

    def limb_columns(self) -> np.ndarray:
        """``(vus, 32)``: each VU's x then W digest limbs (the columns ``dx0 .. dw15``)."""
        return np.array([limbs(a) + limbs(b) for a, b in zip(self.digests["a"], self.digests["b"])], dtype=np.int64)

    def public(self, y: np.ndarray) -> np.ndarray:
        """``(vus, n_y + 32)``: each VU's y words, then its digest limbs."""
        return np.concatenate([np.asarray(y, dtype=np.int64).reshape(self.vus, self.n_y), self.limb_columns()], 1)

    def info(self) -> dict:
        return {"scheme": self.scheme, "leaf": self.leaf, "relation": self.relation, "lo": self.lo, "vus": self.vus, "K": self.iset.K,
                "word_bits": self.iset.word_bits, "dataset": self.iset.dataset, "tier": self.iset.tier,
                "manifest_sha256": self.iset.manifest_sha256, "y": [word_schema(self.y_width), self.n_y],
                **{f"root_{n}": r.hex() for n, r in self.roots.items()},
                **({f"domain_{p}": d.hex() for p, d in self.domains.items()} if self.domains else {})}


def commit(leaf: str, base_relation: str, iset: InstanceSet, lo: int, x: np.ndarray, w: np.ndarray, y: np.ndarray,
           y_width: int) -> Commitment:
    """The trees of the input set: ``x`` / ``w`` ``(vus, K)`` rows, ``y`` ``(vus, n_y)`` public words."""
    from verity.ml.commitments import RangeIndexedDomain
    from verity.ml.commitments.frame_v3 import FrameV3, Port, RowLeaf, WordLeaf
    from verity.ml.commitments.merkle import CommitmentDomain
    from verity.ml.commitments.rowleaf import ROLE_W, ROLE_X

    if leaf not in LEAVES:
        raise ValueError(f"commit: leaf {leaf!r}")
    x, w = np.asarray(x).reshape(len(x), -1), np.asarray(w).reshape(len(w), -1)
    vus, K = x.shape
    if w.shape != x.shape or K != iset.K:
        raise ValueError(f"commit: rows {x.shape} / {w.shape}, the instance set has K = {iset.K}")
    y = np.asarray(y, dtype=np.int64).reshape(vus, -1)
    n_y = y.shape[1]
    if y.min(initial=0) < 0 or y.max(initial=0) >> (8 * y_width):
        raise ValueError(f"commit: a y word exceeds {word_schema(y_width)}")
    if leaf == VLLM:
        from verity.ml.commitments.vllm_v1 import domain_digest, fold, pos_leaf

        doms = {p: vllm_domain(base_relation, iset, lo, vus, n_y, y_width, p) for p in PORTS}
        dig = {"a": [pos_leaf(r.tobytes()) for r in row_bytes(x, iset.word_bits)],
               "b": [pos_leaf(r.tobytes()) for r in row_bytes(w, iset.word_bits)]}
        yleaves = [pos_leaf(int(v).to_bytes(y_width, "little")) for v in y.reshape(-1).tolist()]
        roots = {"a": doms["a"].bind(fold(dig["a"])), "b": doms["b"].bind(fold(dig["b"])), "y": doms["y"].bind(fold(yleaves))}
        return Commitment(leaf, base_relation + SUFFIX[leaf], iset, lo, vus, y_width, n_y, dig, roots,
                          {p: domain_digest(d) for p, d in doms.items()})
    rs = RowLeaf(leaf, ROLE_X, iset.word_bits, K).schema
    dom = {n: CommitmentDomain(binding(iset, lo, lo + vus, n, rs if n != "y" else word_schema(y_width)), OWNER[n],
                               RangeIndexedDomain(0, vus * (n_y if n == "y" else 1))) for n in ("a", "b", "y")}
    scheme = FrameV3({"a": Port(dom["a"], RowLeaf(leaf, ROLE_X, iset.word_bits, K)), "b": Port(dom["b"], RowLeaf(leaf, ROLE_W, iset.word_bits, K)),
                      "y": Port(dom["y"], WordLeaf(word_schema(y_width), y_width))})
    dig = {"a": row_digests(x, iset.word_bits, ROLE_X, leaf), "b": row_digests(w, iset.word_bits, ROLE_W, leaf)}
    roots = {n: scheme.root(n, [scheme.tree_leaf(n, r, d) for r, d in enumerate(dig[n])]) for n in ("a", "b")}
    ywords = y.reshape(-1).tolist()
    roots["y"] = scheme.root("y", [dom["y"].leaf(r, r, word_schema(y_width), int(v).to_bytes(y_width, "big")) for r, v in enumerate(ywords)])
    return Commitment(leaf, base_relation + SUFFIX[leaf], iset, lo, vus, y_width, n_y, dig, roots)


def extend_epilogue(text: str) -> str:
    """The epilogue circuit with the 32 digest-limb columns appended (fresh input wires after every existing wire)."""
    out, ncols, nwires = [], None, None
    for line in text.splitlines():
        t = line.split()
        if t and t[0] == "ncols":
            ncols = int(t[1])
            line = f"ncols {ncols + len(DIGEST_COLS)}"
        elif t and t[0] == "nwires":
            nwires = int(t[1])
            line = f"nwires {nwires + len(DIGEST_COLS)}"
        elif t and t[0] == "col" and t[2] in DIGEST_COLS:
            raise ValueError(f"commit: the epilogue already has a column {t[2]}")
        out.append(line)
    if ncols is None or nwires is None:
        raise ValueError("commit: the epilogue has no ncols / nwires line")
    i = max(k for k, line in enumerate(out) if line.startswith("col ") or line.startswith("ncols") or line.startswith("one "))
    out[i + 1:i + 1] = [f"col {ncols + j} {n}" for j, n in enumerate(DIGEST_COLS)]
    return "\n".join(out) + "\n"


def extend_chain(text: str) -> str:
    lines, pub = [], None
    for line in text.splitlines():
        t = line.split()
        if t and t[0] == "public":
            pub = t[1:]
        else:
            lines.append(line)
    pub = (pub or ["y16"]) + DIGEST_COLS
    return "\n".join([*lines, "public " + " ".join(pub)]) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m gpu.commit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pn = sub.add_parser("pin", help="a frozen set's pinned roots (verifier/src/commitments.rs PINS), as a Rust literal")
    pn.add_argument("relation", help="bf16-ampere | bf16-hopper | fp8-ada | fp8-hopper")
    pn.add_argument("--leaf", choices=LEAVES, required=True)
    pn.add_argument("--lo", type=int, default=0)
    pn.add_argument("--vus", type=int, default=4096)
    pn.add_argument("--instances", type=Path, default=None)
    pn.add_argument("--input-set", type=Path, default=None, help="an input set directory: VUs [0, --vus) of it (--lo must be 0)")
    pn.add_argument("--procs", type=int, default=16)
    args = ap.parse_args(argv)
    import bench_result as br

    src = Path(br.__file__).resolve().parents[2]
    if args.input_set is not None:
        if args.lo:
            raise SystemExit("gpu.commit pin --input-set: a cell commits ids [0, --vus) of its set")
        x, w, y, iset, ref = input_set(args.input_set, args.vus)
        source = (f"input set {ref['tier']} ({ref['source']}; content digest {ref['content_digest'][:16]}...) "
                  f"(python -m gpu.commit pin {args.relation} --leaf {args.leaf} --input-set ... --vus {args.vus})")
    elif args.relation == br.AMPERE_RELATION:
        if args.instances is None:
            raise SystemExit(f"gpu.commit pin {args.relation}: --instances (the {br.TIER} tier's data directory)")
        frozen = src / "fixtures" / "bench-instances" / "v1" / "manifest.json"
        x, w, y, _ = br.load_frozen(args.instances, frozen, br.TIER, args.lo, args.vus)
        source = f"fixtures/bench-instances/v1/manifest.json tier {br.TIER} (python -m gpu.commit pin {args.relation} --leaf {args.leaf})"
    elif args.relation == "fp4-nvf4":
        raise SystemExit("gpu.commit pin fp4-nvf4: not yet (its rows are x.bin's 68-byte steps, not B-Ligero's lane rows)")
    else:
        x, w, y, rel = br.load_relation(args.relation, src, args.lo, args.vus, args.procs)
        source = f"archive/direct/ligero/relchain.py instances() seed {rel.instance_seed} (python -m gpu.commit pin {args.relation} --leaf {args.leaf})"
    if args.input_set is None:
        iset = instance_set(args.relation, args.lo, args.vus, src)
    yw = 4 if np.asarray(y).dtype == np.uint32 else 2
    c = commit(args.leaf, args.relation, iset, args.lo, x, w, y, yw)
    e = c.info()
    print(json.dumps(e))
    print(f'    Pin {{\n        relation: "{c.relation}", lo: {c.lo}, vus: {c.vus},\n'
          f'        dataset: "{iset.dataset}", tier: "{iset.tier}",\n        manifest_sha256: "{iset.manifest_sha256}",\n'
          f'        k: {iset.K}, x_bits: {iset.word_bits}, w_bits: {iset.word_bits}, y_width: {yw}, n_y: {c.n_y},\n'
          f'        a: "{e["root_a"]}",\n        b: "{e["root_b"]}",\n        y: "{e["root_y"]}",\n'
          f'        source: "{source}",\n    }},')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
