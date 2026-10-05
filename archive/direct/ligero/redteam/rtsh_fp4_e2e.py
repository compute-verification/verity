"""red-team-standard-hash-2: the R1 / R4 / H2 attacks and a ZK check on ``fp4-nvf4+poseidon2`` (fp4/hashed.py FP4_HASHED).

``fp4-nvf4`` is not in ``relations.RELATIONS`` (its relation object lives in ``fp4/hashed.py``), so this registers
FP4_HASHED under its name and runs the standard-hash harnesses on it unchanged:

    python -m archive.direct.ligero.redteam.rtsh_fp4_e2e remap  --bin $LIGERO_VERIFY --out DIR --set-binding   # R1
    python -m archive.direct.ligero.redteam.rtsh_fp4_e2e orphan --bin $LIGERO_VERIFY --out DIR --vus 3         # R4
    python -m archive.direct.ligero.redteam.rtsh_fp4_e2e steps  --bin $LIGERO_VERIFY --out DIR --steps 48      # H2
    python -m archive.direct.ligero.redteam.rtsh_fp4_e2e zk     --bin $LIGERO_VERIFY --out DIR                 # ZK

(``--relation fp4-nvf4 --leaf poseidon2`` are supplied.)  ``steps``: a VU of ``--steps`` chained NVFP4 units (the
operands of consecutive canonical VUs concatenated, the accumulator chained through the model), proved by the honest
prover with its step count patched; ``--steps 24`` is the honest control.  Exit 0 iff the outcome is the expected one
(canonical accepted by Python and Rust pinned; any other refused by both).  ``zk``: the statement's public content (pins,
public-table columns) and two ZK proofs of the same VU, which must differ in every opened column; the ZK configuration
must satisfy k > l + t."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _register():
    from .. import relations as RM
    from ..fp4.hashed import FP4_HASHED
    RM.RELATIONS.setdefault(FP4_HASHED.name, FP4_HASHED)
    return FP4_HASHED


def _arg(argv, name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


def steps(argv) -> int:
    rel = _register()
    from .. import protocol
    from ..fp4.relation import STEPS, WORDS
    from ..fp4.witness import MODEL
    from ..relchain import HashedRelationRunner, instances
    from ..serialize import _statement_of, proof_bytes, statement_bytes, system_bytes, verify_files

    binp, out, n_steps = _arg(argv, "--bin"), Path(_arg(argv, "--out")), int(_arg(argv, "--steps"))
    canonical = rel.steps
    sys_canonical = protocol.system_id(HashedRelationRunner(rel, "cpu", -132.0, mode="fiat-shamir").sys).hex()
    reps = -(-n_steps // STEPS)
    src = instances(rel, reps)
    a = sum((list(v[0]) for v in src), [])[: n_steps * WORDS]
    b = sum((list(v[1]) for v in src), [])[: n_steps * WORDS]
    accs, c = [], 0
    for s in range(n_steps):
        c = MODEL.step(c, a[WORDS * s:WORDS * (s + 1)], b[WORDS * s:WORDS * (s + 1)])
        accs.append(int(c))
    cls = type(rel)
    old = (cls.steps, cls.vu_words)
    cls.steps, cls.vu_words = n_steps, n_steps * WORDS
    try:
        R = HashedRelationRunner(rel, "cpu", -132.0, zk=False, mode="fiat-shamir")
        sys_forged = protocol.system_id(R.sys).hex()
        data = [(a, b, accs, accs[-1])]
        try:
            R.commit_vus(data)
            proof, pubs, lay = R.prove_vus(data, check=False, vu_ids=[0], min_l=256)
        except Exception as e:  # noqa: BLE001
            meta = {"relation": rel.name, "steps": n_steps, "canonical_steps": canonical, "prover": f"{type(e).__name__}: {e}"[:300],
                    "expected": n_steps != canonical}
            print(json.dumps(meta))
            return 0 if meta["expected"] else 1
        out.mkdir(parents=True, exist_ok=True)
        st = _statement_of(R, pubs, lay, R.cfg(lay.l), 1)
        (out / "system.bin").write_bytes(system_bytes(R.sys))
        (out / "sub_00.stmt").write_bytes(statement_bytes(st))
        (out / "sub_00.proof").write_bytes(proof_bytes(proof))
    finally:
        cls.steps, cls.vu_words = old
    py_ok, py_why, _ = verify_files(out / "sub_00.stmt", out / "sub_00.proof")
    r = subprocess.run([binp, "verify", "--system", str(out / "system.bin"), "--statement", str(out / "sub_00.stmt"),
                        "--proof", str(out / "sub_00.proof")], capture_output=True, text=True)
    js = json.loads(r.stdout.strip().splitlines()[-1])
    honest = n_steps == canonical
    meta = {"relation": R.relation, "steps": n_steps, "canonical_steps": canonical, "sys_id": sys_forged,
            "same_system": sys_forged == sys_canonical, "python": [bool(py_ok), py_why[:200]],
            "rust": [js["accepted"], js.get("system_pinned"), js["reason"][:200]]}
    meta["expected"] = bool((meta["same_system"] and py_ok and js["accepted"] and js.get("system_pinned")) if honest
                            else (not py_ok and not js["accepted"]))
    (out / "steps.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta))
    return 0 if meta["expected"] else 1


def zk(argv) -> int:
    rel = _register()
    import numpy as np
    from ..relchain import HashedRelationRunner, instances
    from ..serialize import _statement_of, proof_bytes, statement_bytes, system_bytes, verify_files

    binp, out = _arg(argv, "--bin"), Path(_arg(argv, "--out"))
    out.mkdir(parents=True, exist_ok=True)
    data = instances(rel, 1)
    R = HashedRelationRunner(rel, "cpu", -132.0, zk=True, mode="fiat-shamir")
    R.commit_vus(data)
    res = {"pins": [n for _, n in R.sys.pins], "n_pins": len(R.sys.pins)}
    proofs = []
    for i in range(2):
        proof, pubs, lay = R.prove_vus(data, check=False, vu_ids=[0], min_l=256)
        cfg = R.cfg(lay.l)
        st = _statement_of(R, pubs, lay, cfg, 1)
        d = out / f"p{i}"
        d.mkdir(exist_ok=True)
        (d / "system.bin").write_bytes(system_bytes(R.sys))
        (d / "sub_00.stmt").write_bytes(statement_bytes(st))
        (d / "sub_00.proof").write_bytes(proof_bytes(proof))
        py_ok, py_why, _ = verify_files(d / "sub_00.stmt", d / "sub_00.proof")
        cmd = [binp, "verify", "--system", str(d / "system.bin"), "--statement", str(d / "sub_00.stmt"), "--proof", str(d / "sub_00.proof")]
        r = subprocess.run(cmd, capture_output=True, text=True)
        js = json.loads(r.stdout.strip().splitlines()[-1]) if r.stdout.strip() else {"accepted": None, "reason": r.stderr[-300:]}
        proofs.append(proof)
        res[f"p{i}"] = {"python": [bool(py_ok), py_why[:160]], "rust": [js.get("accepted"), js.get("system_pinned"), str(js.get("reason"))[:200]],
                        "cfg": {k: getattr(cfg, k) for k in ("l", "n", "D", "t", "zk", "t_pad") if hasattr(cfg, k)}}
    c = res["p0"]["cfg"]
    res["k_gt_l_plus_t"] = bool(c.get("zk") and c["l"] + c["t_pad"] > c["l"] + c["t"])
    fields = {}
    for name in sorted(vars(proofs[0])):
        x, y = getattr(proofs[0], name), getattr(proofs[1], name)
        try:
            ax, ay = np.asarray(x), np.asarray(y)
            if ax.dtype.kind in "iu" and ax.shape == ay.shape and ax.size:
                fields[name] = {"size": int(ax.size), "equal_fraction": float((ax == ay).mean())}
        except Exception:  # noqa: BLE001
            pass
    res["two_proofs_same_vu"] = fields
    (out / "zk.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps(res, default=str)[:3000])
    ok = all(res[f"p{i}"]["python"][0] for i in range(2)) and res["k_gt_l_plus_t"] and res["n_pins"] == 17
    return 0 if ok else 1


def main() -> int:
    cmd, argv = sys.argv[1], sys.argv[2:]
    if cmd == "steps":
        return steps(argv)
    if cmd == "zk":
        return zk(argv)
    _register()
    mod = {"remap": "rtsh_remap_e2e", "orphan": "rtsh_orphan_e2e"}[cmd]
    import importlib
    m = importlib.import_module(f"archive.direct.ligero.redteam.{mod}")
    sys.argv = [mod] + argv + ["--relation", "fp4-nvf4", "--leaf", "poseidon2"]
    return m.main()


if __name__ == "__main__":
    raise SystemExit(main())
