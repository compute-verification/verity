"""Fixture + Rust-pin generation for a leaf scheme (the "make my scheme verifiable in Rust" helper of the conformance
suite).  For ``(relation, leaf)`` it composes the hashed relation on the CPU, commits a few synthetic VUs, proves ONE
sub-batch and writes what ``archive/ligero-verify`` consumes::

    DIR/system.bin  DIR/sub_00.stmt  DIR/sub_00.proof  DIR/sub_00.coins  DIR/manifest.json

and, when ``ligero-verify`` is built, runs ``system-digest`` on the system file and prints the ``leaf::PINS`` row
``("<rel>", "<leaf>", "<sys_id>", "<table_digest>")`` to paste into ``src/leaf.rs`` (Poseidon2's pins live on the
``Relation`` itself).  The Rust verifier refuses a scheme it does not know: the lane adding a scheme ports its
``schema`` / ``params_sha256`` / ``digest_elems`` / ``leaf_bytes`` into ``src/leaf.rs`` first, then pins.

    python -m archive.direct.ligero.leaf.fixtures --relation fp8-ada --leaf blake3 --out archive/ligero-verify/fixtures/fp8-ada-blake3
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from . import registry

__all__ = ["RUST_DIR", "ligero_verify_binary", "write_fixture"]

RUST_DIR = Path(__file__).resolve().parents[3] / "ligero-verify"


def ligero_verify_binary() -> Path | None:
    """The built ``ligero-verify`` (release, then debug, then PATH); ``None`` when none is built."""
    for p in (RUST_DIR / "target" / "release" / "ligero-verify", RUST_DIR / "target" / "debug" / "ligero-verify"):
        if p.is_file():
            return p
    w = shutil.which("ligero-verify")
    return Path(w) if w else None


def write_fixture(out: Path, relation: str = "fp8-ada", leaf: str | None = None, n_vus: int = 4, vus: tuple[int, ...] = (1, 3),
                  mode: str = "interactive", device: str = "cpu", min_l: int = 1024) -> dict:
    """Write the fixture set; returns the manifest (with the pins row when the Rust binary is available)."""
    from .. import protocol
    from ..relations import relation as rel_of
    from ..relchain import HashedRelationRunner, instances
    from ..serialize import _statement_of, proof_bytes, statement_bytes, system_bytes

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rel = rel_of(relation)
    R = HashedRelationRunner(rel, device, -128.0, zk=False, mode=mode, leaf=leaf)
    data = instances(rel, n_vus)
    R.commit_vus(data)
    sel = [data[v] for v in vus]
    coins = protocol.Coins.sample() if R.interactive else None
    # l >= 1024: the -128 target is met per proof (the checked-in fixtures' shape); smaller layouts land at 2^-127.7
    proof, pubs, lay = R.prove_vus(sel, check=True, vu_ids=list(vus), coins=coins, min_l=min_l)
    ok, why = R.verify_vus(proof, pubs, lay, coins=coins)
    if not ok:
        raise RuntimeError(f"{R.relation}: the fixture proof does not verify in Python: {why}")
    st = _statement_of(R, pubs, lay, R.cfg(lay.l), 1)
    (out / "system.bin").write_bytes(system_bytes(R.sys))
    (out / "sub_00.stmt").write_bytes(statement_bytes(st))
    (out / "sub_00.proof").write_bytes(proof_bytes(proof))
    if coins is not None:
        (out / "sub_00.coins").write_bytes(coins.r1 + coins.s1 + coins.r2 + coins.s2)
    # the committed rows' digests, for a Rust-side native check (what native() returned for the rows the VUs used)
    dig = {nm: np.asarray(R.digests[nm], dtype=np.int64)[list(vus)].tolist() for nm in ("a", "b")}
    man = {
        "format": {"system": "ligero-system/v3", "statement": "ligero-statement/v5", "proof": "ligero-proof/v1"},
        "relation": {"name": rel.name, "statement_relation": R.relation, "leaf": R.leaf.name, "schema": R.leaf.schema,
                     "params_sha256": R.leaf.params_sha256, "digest_elems": R.leaf.digest_elems, "y_bits": rel.y_bits},
        "authentication": R.AUTH, "mode": mode, "n_vus": len(vus), "vu_ids": list(vus), "l": lay.l,
        "system": {"rows": R.sys.m, "sys_id": protocol.system_id(R.sys).hex(), "census": R.hr.census},
        "digests": dig, "files": ["system.bin", "sub_00.stmt", "sub_00.proof"] + (["sub_00.coins"] if coins is not None else []),
        "python_verdict": {"accepted": True, "reason": why},
    }
    binary = ligero_verify_binary()
    if binary is not None:
        r = subprocess.run([str(binary), "system-digest", "--system", str(out / "system.bin")], capture_output=True, text=True)
        if r.returncode == 0:
            d = json.loads(r.stdout)
            man["rust"] = {"sys_id": d["sys_id"], "table_digest": d["table_digest"], "pinned_relation": d.get("pinned_relation")}
            # the Rust `PINS` key is the relation SUFFIX (a family member is "ajtai-n64", not "ajtai")
            man["pins_row"] = f'("{rel.name}", "{registry.suffix_of(R.leaf)}", "{d["sys_id"]}", "{d["table_digest"]}"),'
            if d["sys_id"] != man["system"]["sys_id"]:
                raise RuntimeError("Rust and Python disagree on sys_id")
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    return man


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--relation", default="fp8-ada")
    ap.add_argument("--leaf", default=None, help=f"registry name (default {registry.DEFAULT}); known: {registry.names()} + "
                                                 f"family members {[f'{p}-<params>' for p in registry.families()]}")
    ap.add_argument("--vus", default="1,3", help="comma-separated ids (of --n-vus) of the VUs the fixture proves (default 1,3)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-vus", type=int, default=4)
    ap.add_argument("--mode", default="interactive", choices=("interactive", "fiat-shamir"))
    args = ap.parse_args(argv)
    man = write_fixture(Path(args.out), args.relation, args.leaf, n_vus=args.n_vus, vus=tuple(int(v) for v in args.vus.split(",")),
                        mode=args.mode, device=args.device)
    print(json.dumps({k: man[k] for k in ("relation", "system", "rust", "pins_row") if k in man}, indent=1))
    if "pins_row" in man:
        print(f"\nleaf.rs PINS row for {man['relation']['statement_relation']}:\n    {man['pins_row']}")
    else:
        print("\n(ligero-verify not built: no table digest / pins row; `cargo build --release` in archive/ligero-verify)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
