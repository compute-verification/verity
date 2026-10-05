"""red-team-leaf-3 H1 (end-to-end): the Rust verifier never binds a hashed statement's ``steps`` (columns per VU) to the
pinned system or to the Ajtai leaf's ring degree ``n`` (verify.rs: the chain test + parse_v5 derive the whole layout from
``st.steps``; relation.rs has no steps bound; ``check_config`` only asks ``steps * n_vus <= l``).  The composed system is
column-uniform (``hashchain.compose`` adds the gadget template ONCE), so the pinned ``sys_id`` is step-agnostic.  The only
guard that ``steps <= n`` (needed for the negacyclic Ring-SIS digest to be injective, ``ajtai.py`` gadget) lives in the
PROVER at compile time -- so a malicious prover who removes it produces a proof the PINNED verifier accepts, over rows
whose Ajtai digest is no longer collision-resistant (``leaf3_ajtai_steps.py``: two distinct rows, one digest).

This drives the honest machinery at ``steps = 96 > n = 64`` against the pinned ``fp8-ada+ajtai-n64`` system (``K_VU`` bumped
to 96 * 32 = 3072, the gadget's ``steps <= n`` assert removed -- nothing else), commits two VUs, proves, and runs the
pinned Rust verifier.  Exit 0 iff the pinned verifier ACCEPTS (``system_pinned = true``) a proof whose per-VU operand rows
span 96 columns while the pin's canonical relation is 48 columns / n = 64.

    PYTHONPATH=<ajtai-leaf tree>:<...>:<...>/benchmarks/numerical/python \
        python leaf3_ajtai_steps_e2e.py --bin <ligero-verify with the ajtai PINS> --out DIR [--steps 96]
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=96, help="columns per VU (> n = 64 triggers the wrap; a divisor-free row is fine)")
    ap.add_argument("--relation", default="fp8-ada")
    ap.add_argument("--leaf", default="ajtai-n64")
    ap.add_argument("--k", type=int, default=32, help="words per column: 32 for fp8 (256 bits / 8), 16 for bf16")
    args = ap.parse_args()

    import archive.direct.ligero.relations as R
    from archive.direct.ligero.leaf import ajtai as A

    k = args.k                                      # fixed by the pinned system (256-bit column / word width)
    R.K_VU = args.steps * k                          # rel.steps = K_VU // k = args.steps (a dynamic property)
    # remove the PROVER-side guard `steps <= n` (the ONLY place it is checked; the verifier has none).  The gadget uses
    # `steps` ONLY for that assert -- the per-column template (X acc_in + B s over 256 bits, n = 64 linked rows) is
    # byte-identical for any steps -- so passing a compliant row_words to the original yields the SAME pinned system.
    _orig = A.AjtaiLeaf.gadget
    def gadget(self, ctx, words, **kw):
        kw = dict(kw, row_words=self.n * (self.m_col // kw["word_bits"]))   # steps = n: assert passes, template unchanged
        return _orig(self, ctx, words, **kw)
    A.AjtaiLeaf.gadget = gadget

    from archive.direct.ligero import protocol
    from archive.direct.ligero.relations import relation
    from archive.direct.ligero.relchain import HashedRelationRunner, instances
    from archive.direct.ligero.serialize import _statement_of, proof_bytes, statement_bytes, system_bytes

    rel = relation(args.relation)
    assert rel.steps == args.steps, f"rel.steps = {rel.steps}, wanted {args.steps}"
    Rn = HashedRelationRunner(rel, "cpu", -128.0, zk=False, mode="fiat-shamir", leaf=args.leaf)
    sys_id = protocol.system_id(Rn.sys).hex()
    print(f"{Rn.relation}: steps = {rel.steps} (leaf n = {Rn.leaf.n if hasattr(Rn.leaf, 'n') else '-'}); committed row = "
          f"{rel.steps * k} words; sys_id = {sys_id}")
    data = instances(rel, 4)
    Rn.commit_vus(data)
    coins = None
    vus = (1, 3)
    sel = [data[v] for v in vus]
    proof, pubs, lay = Rn.prove_vus(sel, check=True, vu_ids=list(vus), coins=coins, min_l=1024)
    ok, why = Rn.verify_vus(proof, pubs, lay, coins=coins)
    print(f"python verify: accepted = {ok}; {why[:120]}")
    if not ok:
        print("python rejected -- the forged layout is not internally consistent; H1 not shown end-to-end this way")
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    st = _statement_of(Rn, pubs, lay, Rn.cfg(lay.l), 1)
    (out / "system.bin").write_bytes(system_bytes(Rn.sys))
    (out / "sub_00.stmt").write_bytes(statement_bytes(st))
    (out / "sub_00.proof").write_bytes(proof_bytes(proof))
    r = subprocess.run([args.bin, "verify", "--system", str(out / "system.bin"), "--statement", str(out / "sub_00.stmt"),
                        "--proof", str(out / "sub_00.proof"), "--soundness-bits", "100"], capture_output=True, text=True)
    js = json.loads(r.stdout.strip().splitlines()[-1])
    (out / "verdict.json").write_text(json.dumps(js, indent=1))
    print(f"rust verify (PINNED): accepted = {js['accepted']}  system_pinned = {js.get('system_pinned')}")
    print(f"  reason: {js['reason'][:240]}")
    print(f"  statement.steps as the verifier read it: {js.get('statement', {}).get('steps')}")
    broke = js["accepted"] and js.get("system_pinned") is True
    print(f"\n*** pinned verifier accepts statement.steps = {rel.steps} (no steps bound) ***" if broke else "\n(not accepted pinned)")
    return 0 if broke else 1


if __name__ == "__main__":
    raise SystemExit(main())
