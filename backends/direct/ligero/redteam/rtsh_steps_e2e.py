"""red-team-standard-hash, H2 on the standard-hash leaves: a real `+<leaf>` proof whose VU spans `--steps` columns instead of
the relation's canonical ones (the honest prover with `K_VU = steps * k`; the compiled system is column-uniform, so its
sys_id is the pinned one).  Both verifiers must refuse it: Rust `ligero-verify verify` pinned (`check_vu_shape`) and Python
`serialize.verify_files` (`layout_error` + the header K check).  `--steps` equal to the canonical value is the honest control.

    python -m backends.direct.ligero.redteam.rtsh_steps_e2e --bin $LIGERO_VERIFY --leaf blake3 --steps 32 --out DIR

Exit 0 iff the outcome is the expected one: canonical steps accepted by both (pinned), any other refused by both.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--relation", default="fp8-ada")
    ap.add_argument("--leaf", default="blake3")
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--vus", type=int, default=1)
    ap.add_argument("--min-l", type=int, default=256)
    ap.add_argument("--target", type=float, default=-132.0)
    args = ap.parse_args()

    from .. import protocol
    from .. import relations as RM
    from .. import relchain as RC
    from ..relchain import HashedRelationRunner, instances
    from ..serialize import _statement_of, proof_bytes, statement_bytes, system_bytes, verify_files

    rel = RM.relation(args.relation)
    canonical = rel.steps
    sys_canonical = protocol.system_id(HashedRelationRunner(rel, "cpu", args.target, leaf=args.leaf, mode="fiat-shamir").sys).hex()
    RM.K_VU = RC.K_VU = args.steps * rel.k
    R = HashedRelationRunner(rel, "cpu", args.target, zk=False, mode="fiat-shamir", leaf=args.leaf)
    sys_forged = protocol.system_id(R.sys).hex()
    data = instances(rel, args.vus)
    R.commit_vus(data)
    proof, pubs, lay = R.prove_vus(data, check=False, vu_ids=list(range(args.vus)), min_l=args.min_l)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    st = _statement_of(R, pubs, lay, R.cfg(lay.l), 1)
    (out / "system.bin").write_bytes(system_bytes(R.sys))
    (out / "sub_00.stmt").write_bytes(statement_bytes(st))
    (out / "sub_00.proof").write_bytes(proof_bytes(proof))
    RM.K_VU = RC.K_VU = canonical * rel.k                  # the verifier side is the unpatched one
    py_ok, py_why, _ = verify_files(out / "sub_00.stmt", out / "sub_00.proof")
    r = subprocess.run([args.bin, "verify", "--system", str(out / "system.bin"), "--statement", str(out / "sub_00.stmt"),
                        "--proof", str(out / "sub_00.proof")], capture_output=True, text=True)
    js = json.loads(r.stdout.strip().splitlines()[-1])
    honest = args.steps == canonical
    meta = {"relation": R.relation, "steps": args.steps, "canonical_steps": canonical, "K_header": args.steps * rel.k,
            "sys_id": sys_forged, "same_system": sys_forged == sys_canonical, "python": [bool(py_ok), py_why[:200]],
            "rust": [js["accepted"], js.get("system_pinned"), js["reason"][:200]]}
    meta["expected"] = bool((meta["same_system"] and py_ok and js["accepted"] and js.get("system_pinned")) if honest
                            else (not py_ok and not js["accepted"]))
    (out / "steps.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta))
    return 0 if meta["expected"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
