"""red-team-standard-hash: the twin relabel between two circuits of one leaf (``+blake3`` / ``+blake3-xob``: the same schema,
two systems).  An honest one-VU proof under ``--leaf`` is written as is (the control), then with its statement's relation
string rewritten to ``--twin`` and paired with either system file.  Every relabelled variant must be refused by Python
``serialize.verify_files`` and must not be accepted pinned by Rust ``ligero-verify verify``.

    python -m backends.direct.ligero.redteam.rtsh_twin_relabel --bin $LIGERO_VERIFY --leaf blake3 --twin blake3-xob --out DIR

Exit 0 iff the control is accepted pinned by both and no relabelled variant is accepted (Python) or accepted pinned (Rust).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def _rust(binary: str, d: Path, system: str) -> dict:
    r = subprocess.run([binary, "verify", "--system", str(d / system), "--statement", str(d / "sub_00.stmt"),
                        "--proof", str(d / "sub_00.proof")], capture_output=True, text=True)
    try:
        js = json.loads(r.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return {"accepted": False, "pinned": None, "reason": (r.stderr or r.stdout)[-200:]}
    return {"accepted": bool(js.get("accepted")), "pinned": js.get("system_pinned"), "reason": str(js.get("reason"))[:200]}


def _python(d: Path) -> dict:
    from ..serialize import verify_files
    try:
        ok, why, _ = verify_files(d / "sub_00.stmt", d / "sub_00.proof", "cpu")
    except Exception as e:  # noqa: BLE001  (a refusal by exception is a refusal)
        return {"accepted": False, "reason": f"{type(e).__name__}: {e}"[:200]}
    return {"accepted": bool(ok), "reason": str(why)[:200]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--relation", default="fp8-ada")
    ap.add_argument("--leaf", default="blake3")
    ap.add_argument("--twin", default="blake3-xob")
    ap.add_argument("--min-l", type=int, default=256)
    ap.add_argument("--target", type=float, default=-132.0)
    args = ap.parse_args()

    from .. import protocol
    from .. import relations as RM
    from ..leaf import registry
    from ..relchain import HashedRelationRunner, instances
    from ..serialize import _statement_of, proof_bytes, statement_bytes, system_bytes

    rel = RM.relation(args.relation)
    R = HashedRelationRunner(rel, "cpu", args.target, zk=False, mode="fiat-shamir", leaf=args.leaf)
    T = HashedRelationRunner(rel, "cpu", args.target, zk=False, mode="fiat-shamir", leaf=args.twin)
    if R.leaf.schema != T.leaf.schema:
        raise SystemExit(f"{args.leaf} and {args.twin} are not twins (schemas {R.leaf.schema} / {T.leaf.schema})")
    data = instances(rel, 1)
    R.commit_vus(data)
    proof, pubs, lay = R.prove_vus(data, check=False, vu_ids=[0], min_l=args.min_l)
    st = _statement_of(R, pubs, lay, R.cfg(lay.l), 1)
    raw, pb = statement_bytes(st), proof_bytes(proof)
    t_rel = registry.relation_name(args.relation, args.twin)
    i = raw.index(st.relation.encode())
    forged = raw[:i - 4] + len(t_rel).to_bytes(4, "little") + t_rel.encode() + raw[i + len(st.relation):]
    out = Path(args.out)
    res = {"relation": st.relation, "twin_relation": t_rel, "sys_own": protocol.system_id(R.sys).hex(),
           "sys_twin": protocol.system_id(T.sys).hex(), "variants": {}}
    for name, stmt in (("control", raw), ("relabel", forged)):
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "sub_00.stmt").write_bytes(stmt)
        (d / "sub_00.proof").write_bytes(pb)
        (d / "system_own.bin").write_bytes(system_bytes(R.sys))
        (d / "system_twin.bin").write_bytes(system_bytes(T.sys))
        res["variants"][name] = {"python": _python(d), "rust_own_system": _rust(args.bin, d, "system_own.bin"),
                                 "rust_twin_system": _rust(args.bin, d, "system_twin.bin")}
    c, f = res["variants"]["control"], res["variants"]["relabel"]
    res["control_ok"] = bool(c["python"]["accepted"] and c["rust_own_system"]["accepted"] and c["rust_own_system"]["pinned"])
    res["relabel_refused"] = bool(not f["python"]["accepted"]
                                  and not (f["rust_own_system"]["accepted"] and f["rust_own_system"]["pinned"])
                                  and not (f["rust_twin_system"]["accepted"] and f["rust_twin_system"]["pinned"]))
    (out / "relabel.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res))
    return 0 if res["control_ok"] and res["relabel_refused"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
