"""hashauth.layout_error: a VU's x row and W column are fixed by its y leaf and the committed layout (red-team-standard-hash
R1), the same rule as ligero-verify's ``auth::layout_error``.  End to end (the red-team's ``rtsh_remap_e2e`` 2-VU fixture,
fp8-ada+blake3 on the CPU): VU 0's committed output is VU 1's under honest x / W roots and the statement claims VU 0 on
(x row 1, W column 1) -- refused by Python and ``$LIGERO_VERIFY`` (when set); ``reverify.commitment_problems`` (R2) flags
its y root against the recomputed instance set, a statement without a proof (R4) and a rep that claims a VU twice; the
honest dump passes all three.

    uv run pytest backends/direct/ligero/hashauth_test.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from .auth import TreeRef
from .hashauth import HashAuth, layout_error


def _auth(vu, x, w, counts) -> HashAuth:
    trees = tuple(TreeRef(n, bytes(32), o, c, bytes(32)) for n, o, c in zip(("a", "b", "y"), (-1, -2, -1), counts))
    return HashAuth(tuple(vu), tuple(x), tuple(w), trees)


def test_unshared_is_x_equals_w_equals_vu() -> None:
    assert layout_error(_auth([0, 2, 5], [0, 2, 5], [0, 2, 5], (6, 6, 6))) is None
    assert "committed layout" in layout_error(_auth([0, 1], [1, 1], [1, 1], (2, 2, 2)))       # VU 0 on VU 1's (x, W)
    assert "committed layout" in layout_error(_auth([0, 1], [0, 1], [1, 0], (2, 2, 2)))       # W columns swapped


def test_tile_is_row_major_over_the_columns() -> None:
    assert layout_error(_auth([3, 4, 5], [1, 1, 1], [0, 1, 2], (2, 3, 6))) is None             # relchain.tile_indices(2, 3, 3, 6)
    assert "committed layout" in layout_error(_auth([3, 4, 5], [1, 1, 1], [0, 2, 2], (2, 3, 6)))
    assert "committed layout" in layout_error(_auth([3], [0], [1], (3, 2, 6)))


def test_other_shapes_and_ranges_are_refused() -> None:
    assert layout_error(_auth([0], [0], [0], (1, 1, 1))) is None
    assert "neither" in layout_error(_auth([0], [0], [0], (2, 3, 5)))
    assert layout_error(_auth([6], [2], [0], (2, 3, 6))) == "auth: y: leaf index out of range"
    assert layout_error(_auth([], [], [], (0, 0, 0))) is None


LAYOUT = "auth: a VU's x row / W column is not the one its index fixes in the committed layout"


@pytest.fixture(scope="module")
def remap(tmp_path_factory):
    """``{tag: (dir, python (ok, why), manifest)}`` for the honest dump and the R1 forgery (``rtsh_remap_e2e`` with
    ``--set-binding``): 2 VUs, l = 256, trees bound to the relation's instance set as a producer binds them."""
    pytest.importorskip("torch")
    pytest.importorskip("blake3", exc_type=ImportError)
    from .relations import relation
    from .relchain import HashedRelationRunner, instances, instances_digest
    from .serialize import _statement_of, proof_bytes, statement_bytes, system_bytes

    rel, n = relation("fp8-ada"), 2
    base = instances(rel, n)
    assert int(base[0][3]) != int(base[1][3])
    digest = instances_digest(rel, n)
    bkw = {"dataset": rel.instances_dataset, "tier": rel.instances_tier, "manifest_sha256": digest}
    R = HashedRelationRunner(rel, "cpu", -132.0, zk=False, mode="fiat-shamir", leaf="blake3")
    man = {"relation": rel.name, "mode": "fiat-shamir", "set": {"total_vus": n, "instances": digest}}
    root, out = tmp_path_factory.mktemp("remap"), {}

    def emit(tag, committed, vus, xi, wi):
        R.commit_vus(committed, **bkw)
        auth = HashAuth.for_vus(R.trees, list(range(n)), xi, wi)
        proof, pubs, lay = R.prove_vus(vus, check=False, vu_ids=list(range(n)), auth=auth, coins=None, min_l=256)
        py = R.verify_vus(proof, pubs, lay, coins=None)
        d = root / tag
        (d / "rep0").mkdir(parents=True)
        (d / "system.bin").write_bytes(system_bytes(R.sys))
        (d / "rep0" / "sub_00.stmt").write_bytes(statement_bytes(_statement_of(R, pubs, lay, R.cfg(lay.l), 1)))
        (d / "rep0" / "sub_00.proof").write_bytes(proof_bytes(proof))
        out[tag] = (d, (bool(py[0]), py[1]))

    served = list(base)
    served[0] = (base[0][0], base[0][1], base[0][2], base[1][3])
    emit("honest", base, base, [0, 1], [0, 1])
    emit("forgery", served, [base[1], base[1]], [1, 1], [1, 1])
    return out, man


def _rust(d: Path) -> dict:
    exe = os.environ.get("LIGERO_VERIFY")
    if not exe:
        pytest.skip("LIGERO_VERIFY is not set")
    r = subprocess.run([exe, "verify", "--system", str(d / "system.bin"), "--statement", str(d / "rep0" / "sub_00.stmt"),
                        "--proof", str(d / "rep0" / "sub_00.proof")], capture_output=True, text=True)
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_the_honest_remap_fixture_verifies(remap) -> None:
    from .reverify import commitment_problems
    out, man = remap
    d, py = out["honest"]
    assert py == (True, "")
    assert commitment_problems(d, man, "fp8-ada+blake3", ["rep0"]) == (True, [])
    js = _rust(d)
    assert js["accepted"] and js["system_pinned"], js


def test_a_vu_on_another_vus_operands_is_refused(remap) -> None:
    out, _ = remap
    d, py = out["forgery"]
    assert not py[0] and LAYOUT in py[1], py
    js = _rust(d)
    assert not js["accepted"] and LAYOUT in js["reason"], js


def test_reverify_recomputes_the_trees_and_the_cover(remap, tmp_path) -> None:
    from .reverify import commitment_problems
    out, man = remap
    hashed, bad = commitment_problems(out["forgery"][0], man, "fp8-ada+blake3", ["rep0"])
    assert hashed and bad == ["rep0/sub_00.stmt: tree y root differ from the instance set's"], bad
    twice = shutil.copytree(out["honest"][0], tmp_path / "twice")
    shutil.copy(twice / "rep0" / "sub_00.stmt", twice / "rep0" / "sub_01.stmt")
    assert commitment_problems(twice, man, "fp8-ada+blake3", ["rep0"])[1] == [
        "rep0: 1 statement(s) without a proof, 0 proof(s) without a statement"]
    shutil.copy(twice / "rep0" / "sub_00.proof", twice / "rep0" / "sub_01.proof")
    assert commitment_problems(twice, man, "fp8-ada+blake3", ["rep0"])[1] == [
        "rep0: statements cover 2 distinct VUs of [0, 2) (2 claimed twice)"]
    listed = dict(man, files=[{"proof": "rep0/sub_00.proof", "stmt": "rep0/sub_00.stmt"}, {"stmt": "rep0/sub_01.stmt"}])
    assert "rep0: the manifest lists a statement without a proof" in commitment_problems(twice, listed, "fp8-ada+blake3", ["rep0"])[1]
    other = dict(man, set={"total_vus": 3, "instances": man["set"]["instances"]})
    assert "manifest digest" in commitment_problems(out["honest"][0], other, "fp8-ada+blake3", ["rep0"])[1][0]
