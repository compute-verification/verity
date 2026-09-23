"""red-team-leaf-3 H1, binding break end to end: two committed sets that differ in VU 1's x row produce the SAME a/b trees
(identical digests and roots) yet different outputs y, and the PINNED Rust verifier accepts a proof for each.  So the
committed operand roots do not determine what the prover multiplied: a verifier holding (root_a, root_b) accepts two
different y for them.

Mechanism (steps = 96 > n = 64, the pinned fp8-ada+ajtai-n64 system, same prover patch as leaf3_ajtai_steps_e2e.py):
the Ajtai chain is acc <- X acc + B s_col, so the digest is sum_j X^(95-j) B s_j mod X^64 + 1.  Column j < 32 contributes
X^(95-j) = -X^(31-j), column j + 64 contributes +X^(31-j): equal contents in columns j and j + 64 cancel.  Set 1 puts
content P in columns 0 and 64 of VU 1's x row, set 2 puts Q != P there; digests are equal, x . w differs.

    PYTHONPATH=<ajtai-leaf tree>:<...>/packages/verity/src:<...>/backends/numerical/python \\
        python leaf3_ajtai_collide_e2e.py --bin <ligero-verify with the ajtai PINS> --out DIR

Exit 0 iff: a and b digests/roots identical across the two sets, y differs, and BOTH proofs are ACCEPTED pinned."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--relation", default="fp8-ada")
    ap.add_argument("--leaf", default="ajtai-n64")
    ap.add_argument("--k", type=int, default=32, help="words per column: 32 for fp8, 16 for bf16")
    ap.add_argument("--steps", type=int, default=96, help="> n (the leaf's ring degree); columns 0 and n then cancel")
    args = ap.parse_args()
    from backends.direct.ligero.leaf import ajtai as A0
    steps, n, k = args.steps, A0.degree(args.leaf) if hasattr(A0, "degree") else int(args.leaf.split("-n")[1]), args.k
    assert steps > n, "steps must exceed n for the wrap"

    import backends.direct.ligero.relations as R
    from backends.direct.ligero.leaf import ajtai as A

    R.K_VU = steps * k
    _orig = A.AjtaiLeaf.gadget
    def gadget(self, ctx, words, **kw):
        return _orig(self, ctx, words, **dict(kw, row_words=self.n * (self.m_col // kw["word_bits"])))
    A.AjtaiLeaf.gadget = gadget

    from backends.direct.ligero import protocol
    from backends.direct.ligero.relations import relation
    from backends.direct.ligero.relchain import HashedRelationRunner, instances, tc_dot
    from backends.direct.ligero.serialize import _statement_of, proof_bytes, statement_bytes, system_bytes

    rel = relation(args.relation)
    assert rel.steps == steps and rel.k == k

    def with_cols(vu, content):
        a, b, _, _ = vu
        a = list(a)
        a[0:k] = content
        a[n * k:(n + 1) * k] = content
        accs, c = [], 0
        for s in range(steps):
            c = tc_dot(rel.model, c, a[k * s:k * s + k], b[k * s:k * s + k])
            accs.append(c)
        return a, list(b), accs, accs[-1]

    base = instances(rel, 4)
    a1 = base[1][0]
    P, Q = a1[n * k:(n + 1) * k], a1[k:2 * k]          # two genuine operand columns of the honest row
    assert P != Q
    sets = []
    for tag, content in (("P", P), ("Q", Q)):
        data = list(base)
        data[1] = with_cols(base[1], content)
        sets.append((tag, data))

    out = Path(args.out)
    res = []
    for tag, data in sets:
        Rn = HashedRelationRunner(rel, "cpu", -128.0, zk=False, mode="fiat-shamir", leaf=args.leaf)
        Rn.commit_vus(data)
        vus = (1, 3)
        proof, pubs, lay = Rn.prove_vus([data[v] for v in vus], check=True, vu_ids=list(vus), coins=None, min_l=1024)
        ok, why = Rn.verify_vus(proof, pubs, lay, coins=None)
        d = out / tag
        d.mkdir(parents=True, exist_ok=True)
        st = _statement_of(Rn, pubs, lay, Rn.cfg(lay.l), 1)
        (d / "system.bin").write_bytes(system_bytes(Rn.sys))
        (d / "sub_00.stmt").write_bytes(statement_bytes(st))
        (d / "sub_00.proof").write_bytes(proof_bytes(proof))
        r = subprocess.run([args.bin, "verify", "--system", str(d / "system.bin"), "--statement", str(d / "sub_00.stmt"),
                            "--proof", str(d / "sub_00.proof"), "--soundness-bits", "100"], capture_output=True, text=True)
        js = json.loads(r.stdout.strip().splitlines()[-1])
        (d / "verdict.json").write_text(json.dumps(js, indent=1))
        roots = {nm: bytes(np.asarray(getattr(t, "root", t)).tobytes()).hex()[:32] for nm, t in
                 (Rn.trees.items() if isinstance(Rn.trees, dict) else enumerate(Rn.trees))}
        dig = {nm: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()[:16] for nm, v in
               (Rn.digests.items() if isinstance(Rn.digests, dict) else enumerate(Rn.digests))}
        res.append(dict(tag=tag, y=data[1][3], py=ok, rust=js["accepted"], pinned=js.get("system_pinned"),
                        sys_id=protocol.system_id(Rn.sys).hex()[:16], roots=roots, digests=dig, reason=js["reason"][:100]))
        print(f"set {tag}: VU1 y_final = {data[1][3]}; python {ok}; rust accepted {js['accepted']} pinned {js.get('system_pinned')}; "
              f"sys_id {res[-1]['sys_id']}\n   roots {roots}\n   digests {dig}")

    p, q = res
    # the y tree commits the outputs, so it differs; a/b must not
    same_ab = p["digests"] == q["digests"] and all(p["roots"][nm] == q["roots"][nm] for nm in ("a", "b"))
    diff = [nm for nm in p["roots"] if p["roots"][nm] != q["roots"][nm]]
    print(f"\ntrees whose roots differ between the sets: {diff or 'none'}; a/b identical: {same_ab}; x rows differ: True; "
          f"y differs: {p['y'] != q['y']}")
    (out / "collide.json").write_text(json.dumps(res, indent=1, default=str))
    broke = same_ab and p["rust"] and q["rust"] and p["pinned"] and q["pinned"] and p["y"] != q["y"]
    print("\n*** H1 BINDING BREAK: same a/b roots, two outputs, both accepted pinned ***" if broke else "\n(not reproduced)")
    return 0 if broke else 1


if __name__ == "__main__":
    raise SystemExit(main())
