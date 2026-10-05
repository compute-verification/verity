"""steps-pin must-reject generator (red-team-leaf-3 H2): a real proof of a registered relation whose VUs span ``--steps``
columns instead of the relation's canonical ``K_VU / k``.  The compiled system is column-uniform (the bare unit and the
hashed gadget template are both per column), so the system -- and its pinned ``sys_id`` -- is the canonical one; only
the statement's ``steps`` (and a v5 header's K = steps x k) differ.  The prover is the honest one with ``relations.K_VU``
bumped to ``steps * k`` (and relchain's copy); nothing else is patched.  Before the steps-pin fix both verifiers accepted
such a proof pinned.

    python -m archive.direct.ligero.redteam.steps_pin_forge --relation fp8-ada --leaf poseidon2 --steps 32 --out DIR
    python -m archive.direct.ligero.redteam.steps_pin_forge --relation fp8-ada --steps 32 --out DIR      # bare (v4)

Writes ``DIR/{system.bin, sub_00.stmt, sub_00.proof, forge.json}``; ``forge.json`` records the sys_id next to the
canonical-steps system's, so a reader can see the pin is the relation's.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--relation", default="fp8-ada")
    ap.add_argument("--leaf", default=None, help="hashed (v5) leaf scheme, e.g. poseidon2 / ajtai-n64; omit for the bare v4 statement")
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--vus", type=int, default=2)
    args = ap.parse_args()

    from .. import protocol
    from .. import relations as RM
    from .. import relchain as RC
    from ..relchain import HashedRelationRunner, RelationChainRunner, instances
    from ..serialize import _statement_of, proof_bytes, statement_bytes, system_bytes

    rel = RM.relation(args.relation)
    canonical = rel.steps

    def runner():
        if args.leaf is None:
            return RelationChainRunner(rel, args.device, -128.0, zk=False, mode="fiat-shamir")
        return HashedRelationRunner(rel, args.device, -128.0, zk=False, mode="fiat-shamir", leaf=args.leaf)

    sys_canonical = protocol.system_id(runner().sys).hex()
    RM.K_VU = RC.K_VU = args.steps * rel.k          # relchain holds its own K_VU (instances, the committer's row width)
    assert rel.steps == args.steps, (rel.steps, args.steps)
    R = runner()
    sys_forged = protocol.system_id(R.sys).hex()
    data = instances(rel, 2 * args.vus if args.leaf else args.vus)
    vu_ids = list(range(1, 2 * args.vus, 2)) if args.leaf else list(range(args.vus))
    sel = [data[v] for v in vu_ids]
    if args.leaf is not None:
        R.commit_vus(data)
        proof, pubs, lay = R.prove_vus(sel, check=True, vu_ids=vu_ids, min_l=1024)
    else:
        proof, pubs, lay = R.prove_vus(sel, check=True, vu_ids=vu_ids, min_l=1024)
    ok, why = R.verify_vus(proof, pubs, lay)          # the prover's own (patched) view: internally consistent
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    st = _statement_of(R, pubs, lay, R.cfg(lay.l), 1)
    (out / "system.bin").write_bytes(system_bytes(R.sys))
    (out / "sub_00.stmt").write_bytes(statement_bytes(st))
    (out / "sub_00.proof").write_bytes(proof_bytes(proof))
    meta = {"relation": R.relation if hasattr(R, "relation") else rel.name, "leaf": args.leaf, "steps": args.steps,
            "canonical_steps": canonical, "k": rel.k, "K_header": args.steps * rel.k, "n_vus": lay.n_vus, "l": lay.l,
            "sys_id": sys_forged, "sys_id_canonical_steps": sys_canonical, "same_system": sys_forged == sys_canonical,
            "patched_prover_self_verify": [bool(ok), why[:200]]}
    (out / "forge.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta))
    return 0 if ok and sys_forged == sys_canonical else 1


if __name__ == "__main__":
    raise SystemExit(main())
