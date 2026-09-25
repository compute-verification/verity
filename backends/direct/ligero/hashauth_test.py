"""hashauth.layout_error: a VU's x row and W column are fixed by its y leaf and the committed layout (red-team-standard-hash
R1), the same rule as ligero-verify's ``auth::layout_error``.  End to end (the red-team's ``rtsh_remap_e2e`` 2-VU fixture,
fp8-ada+blake3 on the CPU): VU 0's committed output is VU 1's under honest x / W roots and the statement claims VU 0 on
(x row 1, W column 1) -- refused by Python and ``$LIGERO_VERIFY`` (when set); ``reverify.commitment_problems`` (R2) flags
its y root against the recomputed instance set, a statement without a proof (R4) and a rep that claims a VU twice; the
honest dump passes all three.  Tile layouts (a +shared v6 dump of a 2 x 3 tile written by ``bench-vu``): the dump is
recomputed from its manifest's ``set.tile`` and passes ``verify_tree``; a remapped tile VU, a wrong ``set.tile``, a root
that is not the set's and a dump without ``set.tile`` are refused.

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
    assert hashed and bad == ["rep0/sub_00.stmt: tree y root differ from the instance set's", f"rep0/sub_00.stmt: {LAYOUT}"], bad
    twice = shutil.copytree(out["honest"][0], tmp_path / "twice")
    shutil.copy(twice / "rep0" / "sub_00.stmt", twice / "rep0" / "sub_01.stmt")
    assert commitment_problems(twice, man, "fp8-ada+blake3", ["rep0"])[1] == [
        "rep0: 1 statement(s) without a proof, 0 proof(s) without a statement"]
    shutil.copy(twice / "rep0" / "sub_00.proof", twice / "rep0" / "sub_01.proof")
    assert commitment_problems(twice, man, "fp8-ada+blake3", ["rep0"])[1] == [
        "rep0: statements cover 2 distinct VUs of [0, 2) (2 claimed twice)"]
    listed = dict(man, files=[{"proof": "rep0/sub_00.proof", "stmt": "rep0/sub_00.stmt"}, {"stmt": "rep0/sub_01.stmt"}])
    assert "rep0: the manifest lists a statement without a proof" in commitment_problems(twice, listed, "fp8-ada+blake3", ["rep0"])[1]
    e = {i: {"proof": f"rep0/sub_0{i}.proof", "stmt": f"rep0/sub_0{i}.stmt"} for i in (0, 1)}
    crossed = dict(man, files=[{**e[0], "stmt": e[1]["stmt"]}, {**e[1], "stmt": e[0]["stmt"]}])
    assert commitment_problems(twice, crossed, "fp8-ada+blake3", ["rep0"])[1] == [
        "rep0: the manifest pairs a proof with another sub-batch's statement"]
    assert commitment_problems(twice, dict(man, files=[e[0], e[0], e[1]]), "fp8-ada+blake3", ["rep0"])[1] == [
        "rep0: the manifest's proofs are not the rep's proofs"]
    (twice / "rep0" / "sub_01.stmt").write_bytes(b"stmt-1")
    assert commitment_problems(twice, man, "fp8-ada+blake3", ["rep0"])[1][0].startswith("rep0/sub_01.stmt: unreadable statement")
    other = dict(man, set={"total_vus": 3, "instances": man["set"]["instances"]})
    assert "manifest digest" in commitment_problems(out["honest"][0], other, "fp8-ada+blake3", ["rep0"])[1][0]


# ---- tile layouts: a +shared (v6) dump is recomputed as the GEMM tile its manifest's `set.tile` names (fp8-ada, a 2 x 3
# tile: 6 VUs as 2 sub-batches of 3, Fiat-Shamir + ZK, written on the CPU by the bench itself) ----

TILE = (2, 3)
PINNED_SHARED = "fp8-ada+hash"          # what `system-digest` names a row-sharing pair's G system


def _rust_pair(root: Path, d: Path) -> dict:
    exe = os.environ.get("LIGERO_VERIFY")
    if not exe:
        pytest.skip("LIGERO_VERIFY is not set")
    r = subprocess.run([exe, "verify", "--system", str(root / "system.bin"), "--system-h", str(root / "system_h.bin"),
                        "--statement", str(d / "sub_00.stmt"), "--proof", str(d / "sub_00.proof"), "--proof-h", str(d / "sub_00.hproof")],
                       capture_output=True, text=True)
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def tile(tmp_path_factory):
    """``(honest dump, its manifest, forgery dir, python verdict of the forgery)``.  The honest dump is ``run bench-vu
    --auth included-hash-shared --tile 2x3 --dump-dir``.  The forgery is R1 on a tile: VU 0's committed output is VU 4's
    (x row 1, W column 1) under honest x / W roots, and its sub-batch claims VU 0 on x row 1 / W column 1 (the tile fixes
    x row 0 / W column 0)."""
    pytest.importorskip("torch")
    from .relations import relation
    from .relchain import SharedHashedRunner, tile_digest, tile_instances
    from .run import main as run_main
    from .serialize import _statement_of, proof_bytes, statement_bytes

    root = tmp_path_factory.mktemp("tile")
    d = root / "honest"
    assert run_main(["--relation", "fp8-ada", "bench-vu", "--device", "cpu", "--auth", "included-hash-shared", "--tile", "2x3",
                     "--total-vus", "6", "--batch", str(3 * 48), "--reps", "1", "--zk", "--mode", "fiat-shamir", "--pipeline", "0",
                     "--dump-dir", str(d)]) == 0
    man = json.loads((d / "manifest.json").read_text())
    rel = relation("fp8-ada")
    data, x, w = tile_instances(rel, *TILE)
    served = list(data)
    served[0] = (data[0][0], data[0][1], data[0][2], data[4][3])
    R = SharedHashedRunner(rel, "cpu", -128.0, zk=True, mode="fiat-shamir", n_proofs=2)
    R.commit_vus(served, dataset=rel.instances_dataset, tier=rel.instances_tier, manifest_sha256=tile_digest(rel, *TILE),
                 tile=TILE, x_rows=x, w_cols=w)
    auth = HashAuth.for_vus(R.trees, [0, 1, 2], [1, 0, 0], [1, 1, 2])
    pf, pubs, lay = R.prove_vus([data[4], data[1], data[2]], check=False, vu_ids=[0, 1, 2], auth=auth)
    py = R.verify_vus(pf, pubs, lay)
    f = root / "forgery"
    shutil.copytree(d, f)
    (f / "rep1" / "sub_00.stmt").write_bytes(statement_bytes(_statement_of(R, pubs, lay, R.cfg(lay.l), 2)))
    (f / "rep1" / "sub_00.proof").write_bytes(proof_bytes(pf.g))
    (f / "rep1" / "sub_00.hproof").write_bytes(proof_bytes(pf.h))
    return d, man, f, (bool(py[0]), py[1])


def test_the_bench_writes_set_tile(tile) -> None:
    from .relations import relation
    from .relchain import TILE_DRAW, TILE_LAYOUT, tile_digest
    d, man, *_ = tile
    rel = relation("fp8-ada")
    assert man["set"]["tile"] == {"nx": 2, "nw": 3, "seed": rel.instance_seed, "draw": TILE_DRAW, "layout": TILE_LAYOUT}
    assert man["set"]["instances"] == tile_digest(rel, *TILE) and man["set"]["total_vus"] == 6
    assert man["system_h_file"]["path"] == "system_h.bin"
    assert sorted(p.name for p in (d / "rep1").iterdir()) == [f"sub_0{i}.{s}" for i in (0, 1) for s in ("hproof", "proof", "stmt")]


def test_an_honest_tile_dump_is_recomputed(tile, tmp_path) -> None:
    from .reverify import Report, Verifier, _sha256, commitment_problems, committed_trees, verify_tree
    d, man, *_ = tile
    want = committed_trees(PINNED_SHARED, man)
    assert [want[n][2] for n in ("a", "b", "y")] == [2, 3, 6]
    assert commitment_problems(d, man, PINNED_SHARED, ["rep1"]) == (True, [])
    exe = os.environ.get("LIGERO_VERIFY")
    if not exe:
        pytest.skip("LIGERO_VERIFY is not set")
    rep = Report("art:test")
    verify_tree(d, man, Verifier(Path(exe), _sha256(Path(exe)), None), rep, jobs=2, params={}, asserted=None, out_dir=tmp_path)
    assert (rep.status, rep.why, rep.relation, rep.hashed, rep.tile) == ("PASS", [], PINNED_SHARED, True, "2x3"), rep.why
    assert rep.reps["rep1"]["accepted"] == 2 and rep.custody == {"files": 8, "mismatched": 0, "missing": 0}, rep.reps


def test_a_remapped_tile_vu_is_refused(tile) -> None:
    from .reverify import commitment_problems
    from .serialize import verify_files
    d, man, f, py = tile
    assert not py[0] and LAYOUT in py[1], py
    ok, why, _ = verify_files(f / "rep1" / "sub_00.stmt", f / "rep1" / "sub_00.proof", "cpu")
    assert not ok and LAYOUT in why, why
    assert commitment_problems(f, man, PINNED_SHARED, ["rep1"]) == (True, [
        "rep1/sub_00.stmt: tree y root differ from the instance set's", f"rep1/sub_00.stmt: {LAYOUT}"])
    js = _rust_pair(f, f / "rep1")
    assert not js["accepted"] and LAYOUT in js["reason"], js


def _tile_set(man: dict, **tile) -> dict:
    return dict(man, set={**man["set"], "tile": {**man["set"]["tile"], **tile}})


@pytest.mark.parametrize("edit,why", [
    ({"nx": 3, "nw": 2}, "commitment: the fp8-ada 3x2 tile (seed "),
    ({"nx": 1}, "commitment: manifest `set.tile` is 1x3 = 3 VUs, the set has total_vus = 6"),
    ({"nw": 0}, "commitment: manifest `set.tile`: nx = 2, nw = 0 are not positive integers"),
    ({"seed": 7}, "commitment: manifest `set.tile` seed 7 is not the fp8-ada instance seed"),
    ({"draw": "relchain.tile_instances/v2"}, "commitment: manifest `set.tile` names draw 'relchain.tile_instances/v2'"),
    ({"layout": "column-major"}, "commitment: manifest `set.tile` names draw"),
], ids=["swapped", "wrong-count", "zero", "other-seed", "other-draw", "other-layout"])
def test_a_wrong_set_tile_is_refused(tile, edit, why) -> None:
    from .reverify import commitment_problems
    d, man, *_ = tile
    hashed, bad = commitment_problems(d, _tile_set(man, **edit), PINNED_SHARED, ["rep1"])
    assert hashed and len(bad) == 1 and bad[0].startswith(why), bad


def test_a_consistent_manifest_of_another_tile_does_not_match_the_roots(tile) -> None:
    """A manifest naming the 3 x 2 tile with ITS digest: the set is recomputed, and the statements' trees (2 x rows, 3 W
    columns under the 2 x 3 bindings) are not its trees."""
    from .relations import relation
    from .relchain import tile_digest
    from .reverify import commitment_problems
    d, man, *_ = tile
    other = _tile_set(man, nx=3, nw=2)
    other["set"]["instances"] = tile_digest(relation("fp8-ada"), 3, 2)
    hashed, bad = commitment_problems(d, other, PINNED_SHARED, ["rep1"])
    assert hashed and bad == [f"rep1/sub_0{i}.stmt: tree {n} {what} differ from the instance set's"
                              for i in (0, 1) for n, what in (("a", "binding/count/root"), ("b", "binding/count/root"), ("y", "binding/root"))], bad


def test_a_tile_statement_whose_root_is_not_the_sets_is_refused(tile, tmp_path) -> None:
    from .reverify import commitment_problems
    from .serialize import read_statement
    d, man, *_ = tile
    e = shutil.copytree(d, tmp_path / "root")
    p = e / "rep1" / "sub_01.stmt"
    raw = bytearray(p.read_bytes())
    at = raw.find(read_statement(bytes(raw)).hash_auth.tree("a").root)
    assert at > 0
    raw[at] ^= 1
    p.write_bytes(bytes(raw))
    assert commitment_problems(e, man, PINNED_SHARED, ["rep1"]) == (True, ["rep1/sub_01.stmt: tree a root differ from the instance set's"])


def test_a_tile_dump_without_set_tile_fails_closed(tile) -> None:
    """The dumps before `set.tile` (the shared-local cells): their manifests name no tile, some the unshared set's digest."""
    from .relations import relation
    from .relchain import instances_digest
    from .reverify import commitment_problems
    d, man, *_ = tile
    s = {k: v for k, v in man["set"].items() if k != "tile"}
    closed = (True, ["commitment: 2 row-sharing (v6) statement(s) and no manifest `set.tile`: the shared-row layout is not recomputed"])
    assert commitment_problems(d, dict(man, set=s), PINNED_SHARED, ["rep1"]) == closed
    assert commitment_problems(d, dict(man, set=dict(s, instances=instances_digest(relation("fp8-ada"), 6))), PINNED_SHARED, ["rep1"]) == closed
