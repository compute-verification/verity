"""Write the red-team-leaf-2 G1 must-reject fixture ``archive/ligero-verify/fixtures/fp8-ada-ajtai-n64-g1-decoy/``: an
fp8-ada+ajtai-n64 system whose real chain is keyed by ``B = 0`` plus 2n decoy rows carrying the derived key over zero hint
rows (``ajtai_key_test._patch_zero_key``), and a proof that satisfies it (2 VUs, Fiat-Shamir, l = 1024, CPU).  Every
digest in the statement is 0.  The Rust verifier must refuse it pinned AND under ``--allow-any-system``
(``tests/relations.rs::ajtai_g1_decoy_fixture_is_refused``); the Python verifier refuses it (``ajtai_key_test``).

    PYTHONPATH=.:verity python -m archive.direct.ligero.leaf.ajtai_g1_fixture [OUT]"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from .fixtures import RUST_DIR


def main(out: Path) -> dict:
    from .. import protocol
    from ..relations import relation
    from ..relchain import HashedRelationRunner, instances
    from ..serialize import _statement_of, proof_bytes, statement_bytes, system_bytes
    from .ajtai_key_test import _patch_zero_key

    with pytest.MonkeyPatch.context() as mp:
        _patch_zero_key(mp, decoys=True)
        rel = relation("fp8-ada")
        R = HashedRelationRunner(rel, "cpu", -128.0, zk=False, mode="fiat-shamir", leaf="ajtai-n64")
        data = instances(rel, 4)
        R.commit_vus(data)
        proof, pubs, lay = R.prove_vus([data[1], data[3]], check=True, vu_ids=[1, 3], min_l=1024)
        ok, why = R.verify_vus(proof, pubs, lay)
        st = _statement_of(R, pubs, lay, R.cfg(lay.l), 1)
        out.mkdir(parents=True, exist_ok=True)
        (out / "system.bin").write_bytes(system_bytes(R.sys))
        (out / "sub_00.stmt").write_bytes(statement_bytes(st))
        (out / "sub_00.proof").write_bytes(proof_bytes(proof))
        man = {"finding": "red-team-leaf-2 G1", "relation": R.relation, "mode": "fiat-shamir", "l": lay.l, "vu_ids": [1, 3],
               "system": {"rows": R.sys.m, "sys_id": protocol.system_id(R.sys).hex(),
                          "decoy_rows": sum(1 for L in R.sys.linear if L.name.startswith("hash.decoy."))},
               "digests_all_zero": bool(not np.asarray(R.digests["a"]).any() and not np.asarray(R.digests["b"]).any()),
               "python_verdict": {"accepted": ok, "reason": why}, "expect": "rejected (pinned and --allow-any-system)"}
    (out / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    return man


if __name__ == "__main__":
    dst = Path(sys.argv[1]) if len(sys.argv) > 1 else RUST_DIR / "fixtures" / "fp8-ada-ajtai-n64-g1-decoy"
    print(json.dumps(main(dst), indent=1))
