"""red-team-standard-hash-2: vllm-v1 port-tree attacks on a real ``<rel>+vllm-v1`` proof (leaf/vllm_v1.py, vllm_tree.py).

Every variant is a real proof of the relation's first ``--vus`` instances, committed under ports the variant chooses; the
statement is written as the bench writes it and checked by Python (``serialize.verify_files``), Rust pinned
(``ligero-verify verify``) and reverify's commitment recomputation (``reverify.commitment_problems``, manifest ``set``).

* ``honest``: the bench's commitment: all three accept.
* ``ctx-swap``: tree a bound to port b's ctx and b to a's (roots recomputed under the verifier's domains): Python and Rust
  cannot tell (ctx is the statement's), reverify must refuse.
* ``ctx-other-set``: every port's ctx names another instance set (manifest digest): reverify must refuse.
* ``y-big-endian``: the y tree over the output words' big-endian bytes: every verifier must refuse (y leaves are theirs).
* ``bare-root``: tree a's published root is its bare tree root, not the step root: every verifier must refuse.
* ``port-transplant``: tree a's rows committed under port b's domain (layout/value of W columns), the ref named a: every
  verifier must refuse (the domain comes from the slot).
* ``count-lie``: tree a's count one larger than its leaves (a step root over the declared count): every verifier must
  refuse (the fold's shape is derived from the count).

    python -m backends.direct.ligero.redteam.rtsh_vllm_e2e --bin $LIGERO_VERIFY --relation fp8-ada-x4 --out DIR [--vus 3]

Exit 0 iff every variant has its expected outcome."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--relation", default="fp8-ada-x4")
    ap.add_argument("--vus", type=int, default=3)
    args = ap.parse_args()

    import numpy as np
    from verity.commitments.vllm_v1 import pos_leaf

    from .. import reverify as RV
    from .. import vllm_tree
    from ..auth import OWNER, TreeRef
    from ..hashauth import HashAuth
    from ..relations import relation
    from ..relchain import HashedRelationRunner, commit_bindings, instances, instances_digest
    from ..serialize import _statement_of, proof_bytes, statement_bytes, system_bytes, verify_files

    rel = relation(args.relation)
    n = args.vus
    pinned = f"{rel.name}+vllm-v1"
    data, digest = instances(rel, n), instances_digest(rel, n)
    R = HashedRelationRunner(rel, "cpu", -132.0, zk=False, mode="fiat-shamir", leaf="vllm-v1")
    x_rows, w_cols = R.committed_rows(data)
    y = [rel.y_public(int(v[3])) for v in data]
    K, wb, yb = R.vshape
    honest_b = commit_bindings(rel, R.leaf, dataset=rel.instances_dataset, tier=rel.instances_tier, manifest_sha256=digest, lo=0, hi=n)
    other_b = commit_bindings(rel, R.leaf, dataset=rel.instances_dataset, tier=rel.instances_tier, manifest_sha256="ab" * 32, lo=0, hi=n)
    out = Path(args.out)
    res: dict = {"relation": pinned, "vus": n, "vshape": [K, wb, yb]}

    def raw_tree(name, ctx, leaves, dom, root=None, count=None):
        levels = vllm_tree._levels_host(np.asarray(leaves, dtype=np.uint8).reshape(-1, 32))
        tr = object.__new__(vllm_tree.VllmTree)
        cnt = int(np.shape(leaves)[0]) if count is None else count
        tr.ref = TreeRef(name, bytes(ctx), OWNER[name], cnt, dom.bind(bytes(levels[-1][0])) if root is None else root)
        tr._levels, tr.dom = levels, dom
        return tr

    def emit(tag):
        vu = list(range(n))
        proof, pubs, lay = R.prove_vus(data, check=False, vu_ids=vu, auth=HashAuth.for_vus(R.trees, vu, vu, vu), coins=None, min_l=256)
        d = out / tag / "proofs"
        (d / "rep1").mkdir(parents=True, exist_ok=True)
        blobs = {"system.bin": system_bytes(R.sys), "rep1/sub_00.stmt": statement_bytes(_statement_of(R, pubs, lay, R.cfg(lay.l), 1)),
                 "rep1/sub_00.proof": proof_bytes(proof)}
        for p, b in blobs.items():
            (d / p).write_bytes(b)
        man = {"relation": rel.name, "authentication": "included-hash", "mode": "fiat-shamir", "set": {"total_vus": n, "instances": digest},
               "files": [{"rep": 1, "sub": 0, "proof": "rep1/sub_00.proof", "proof_sha256": _sha(blobs["rep1/sub_00.proof"]),
                          "stmt": "rep1/sub_00.stmt", "stmt_bytes": len(blobs["rep1/sub_00.stmt"])}],
               "system_file": {"path": "system.bin", "sha256": _sha(blobs["system.bin"]), "bytes": len(blobs["system.bin"])}}
        (d / "manifest.json").write_text(json.dumps(man))
        py_ok, py_why, _ = verify_files(d / "rep1" / "sub_00.stmt", d / "rep1" / "sub_00.proof")
        r = subprocess.run([args.bin, "verify", "--system", str(d / "system.bin"), "--statement", str(d / "rep1" / "sub_00.stmt"),
                            "--proof", str(d / "rep1" / "sub_00.proof")], capture_output=True, text=True)
        js = json.loads(r.stdout.strip().splitlines()[-1]) if r.stdout.strip() else {"accepted": False, "reason": r.stderr[-300:]}
        try:
            hashed, probs = RV.commitment_problems(d, man, pinned, ["rep1"])
        except Exception as e:  # noqa: BLE001
            hashed, probs = True, [f"reverify raised {type(e).__name__}: {e}"]
        row = {"python": [bool(py_ok), str(py_why)[:200]], "rust": [js.get("accepted"), js.get("system_pinned"), str(js.get("reason"))[:200]],
               "reverify_problems": probs[:3], "reverify_pass": hashed and not probs}
        print(f"{tag}: python {py_ok} {str(py_why)[:120]!r}; rust {js.get('accepted')} pinned {js.get('system_pinned')} "
              f"({str(js.get('reason'))[:120]}); reverify {'PASS' if row['reverify_pass'] else probs[:2]}", flush=True)
        return row

    R.commit(x_rows=x_rows, w_cols=w_cols, y_words=y, bindings=honest_b)
    res["honest"] = emit("honest")
    honest_trees = dict(R.trees)

    R.commit(x_rows=x_rows, w_cols=w_cols, y_words=y, bindings={"a": honest_b["b"], "b": honest_b["a"], "y": honest_b["y"]})
    res["ctx-swap"] = emit("ctx-swap")
    R.commit(x_rows=x_rows, w_cols=w_cols, y_words=y, bindings=other_b)
    res["ctx-other-set"] = emit("ctx-other-set")

    def dom_of(name, ctx, count):
        return vllm_tree.domain(name, ctx, count, K=K, word_bits=wb, y_bytes=yb)

    yw = np.asarray(y, dtype=np.int64)
    be = [pos_leaf(int(w).to_bytes(yb, "big")) for w in yw]
    R.trees = dict(honest_trees)
    R.trees["y"] = raw_tree("y", honest_b["y"], np.frombuffer(b"".join(be), dtype=np.uint8).reshape(n, 32), dom_of("y", honest_b["y"], n))
    res["y-big-endian"] = emit("y-big-endian")

    a_leaves = honest_trees["a"].levels[0]
    R.trees = dict(honest_trees)
    R.trees["a"] = raw_tree("a", honest_b["a"], a_leaves, dom_of("a", honest_b["a"], n), root=bytes(honest_trees["a"].tree_root))
    res["bare-root"] = emit("bare-root")

    R.trees = dict(honest_trees)
    R.trees["a"] = raw_tree("a", honest_b["a"], a_leaves, dom_of("b", honest_b["a"], n))
    res["port-transplant"] = emit("port-transplant")

    R.trees = dict(honest_trees)
    t = raw_tree("a", honest_b["a"], a_leaves, dom_of("a", honest_b["a"], n + 1), count=n + 1)
    t.ref = TreeRef("a", honest_b["a"], OWNER["a"], n + 1, dom_of("a", honest_b["a"], n + 1).bind(bytes(t._levels[-1][0])))
    R.trees["a"] = t
    try:
        res["count-lie"] = emit("count-lie")
    except Exception as e:  # noqa: BLE001  (the prover's multiproof cannot be built for a count its levels do not have)
        res["count-lie"] = {"prover": f"{type(e).__name__}: {e}"[:200], "python": [False, "prover refused"], "rust": [False, None, ""],
                            "reverify_pass": False}
        print(f"count-lie: prover refused ({type(e).__name__}: {str(e)[:120]})")

    def all_ok(v):
        return v["python"][0] and v["rust"][0] and v["rust"][1] and v["reverify_pass"]

    def verifiers_refuse(v):
        return not v["python"][0] and not v["rust"][0] and not v["reverify_pass"]

    exp = {"honest": all_ok(res["honest"]),
           "ctx-swap": not res["ctx-swap"]["reverify_pass"],
           "ctx-other-set": not res["ctx-other-set"]["reverify_pass"],
           "y-big-endian": verifiers_refuse(res["y-big-endian"]),
           "bare-root": verifiers_refuse(res["bare-root"]),
           "port-transplant": verifiers_refuse(res["port-transplant"]),
           "count-lie": verifiers_refuse(res["count-lie"])}
    res["expected"] = exp
    out.mkdir(parents=True, exist_ok=True)
    (out / "vllm_e2e.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps(exp))
    return 0 if all(exp.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
