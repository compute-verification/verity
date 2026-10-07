"""`verity_flock.rec_shape`, outer proofs of one fixed shape: on rec-algebra's fixture session (`test_rec_vstage`'s), the cap's
statements are exactly the ones V* stages over a session of the cap's shape, and a session of another shape (another instance
count, other claims, a level whose openings are not the schedule's) is refused before anything is staged; on a small
Gemm_v2 class, in both layouts (tiles and one coordinate an instance), a statement padded to the cap (`class_statement
--real`) has the cap's shape, its real instances' outputs are the unpadded statement's, and `forge_filler` moves one bit of
one filler's output row alone."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from test_rec_vstage import SET, _fold, inner  # noqa: F401
from verity_flock import rec_open as R
from verity_flock import rec_outer as O
from verity_flock import rec_residuals as RR
from verity_flock import rec_shape as RS
from verity_flock import rec_vstage as S
from verity_flock import rec_vstar as V


def _cap(**kw) -> RS.Cap:
    base = RS.Cap.of(SET / "stage" / "circuit.txt", S.instances(SET / "stage" / "pub-4.bin"), 2 + len(_fold()["extras"]))
    return RS.Cap(**{**base.__dict__, **kw})


def test_a_cap_is_its_public_spec():
    """A cap is four public numbers, the inner circuit's META the k_log and vus_per_block, and its spec gives it back."""
    cap = _cap()
    assert RS.Cap.parse(cap.spec()) == cap and (cap.instances, cap.claims) == (4, 8)
    assert (cap.shape.m, cap.shape.k_log, cap.shape.claims) == (25, 22, 8)
    with pytest.raises(ValueError, match="instances=N"):
        RS.Cap.parse("instances=4,claims=8")


def test_the_caps_levels_are_the_sessions(inner):
    """Each level's RecOpen shape (LANES, H) and its openings (both reps' queries) follow from the cap's table shape alone, and
    are the ones the fixture session opens."""
    public, private, proofs = V.load(inner)
    sh = S.verifier(public, _fold()).sh
    lvls = O.levels(V.queries(public, private, proofs))
    assert RS.levels(_cap()) == {f"L{lv['level']}": (lv["LANES"], lv["H"], len(lv["queries"])) for lv in lvls}
    RS.check(sh, lvls, _cap())


def test_the_caps_statements_are_the_ones_v_star_stages_over_the_session(inner):
    """`outer` under the session's cap is the statements `rec_vstage.stage` stages over it: each level's RecOpen template and
    openings from the session's own levels, and each of the algebra's parts' template from the session's own table shape (the
    fold's), two instances (one a rep)."""
    public, private, proofs = V.load(inner)
    vf = S.verifier(public, _fold())
    got = {f"L{lv['level']}": {"template": R.definition(lv["LANES"], lv["H"]).id, "instances": len(lv["queries"])}
           for lv in O.levels(V.queries(public, private, proofs))}
    got |= {f"alg-p{j}": {"template": RR.definition(p.S).id, "instances": 2} for j, p in enumerate(RR.parts(RR.spec(vf.sh)))}
    assert RS.outer(_cap()) == got


@pytest.mark.parametrize("kw", [{"instances": 64}, {"claims": 9}, {"k_log": 21}])
def test_a_session_of_another_shape_is_refused_before_staging(inner, tmp_path, kw):
    """A cap of another instance count (another m), other claims or another k_log refuses the session, and nothing is staged;
    the command exits 2 with the refusal."""
    with pytest.raises(RS.Refused, match="SHAPE-REFUSED"):
        S.stage(inner, _fold(), tmp_path / "v", salts=tmp_path / "salts.bin", cap=_cap(**kw))
    assert not (tmp_path / "v").exists() and not (tmp_path / "salts.bin").exists()
    (tmp_path / "fold.json").write_text(json.dumps(_fold()))
    assert S.main(["stage", str(inner), str(tmp_path / "fold.json"), str(tmp_path / "v"), "--cap", _cap(**kw).spec()]) == 2


def test_a_level_that_is_not_the_schedules_is_refused(inner):
    """The session's table shape alone is not trusted: a level with one opening fewer than the schedule's, or another H, is
    another statement and is refused."""
    public, private, proofs = V.load(inner)
    sh = S.verifier(public, _fold()).sh
    lvls = O.levels(V.queries(public, private, proofs))
    short = [{**lvls[0], "queries": lvls[0]["queries"][1:]}, *lvls[1:]]
    with pytest.raises(RS.Refused, match="levels"):
        RS.check(sh, short, _cap())
    with pytest.raises(RS.Refused, match="levels"):
        RS.check(sh, [{**lvls[0], "H": lvls[0]["H"] + 1}, *lvls[1:]], _cap())


def test_the_filler_forgery_is_v_stars_algebra_alone(inner, tmp_path):
    """``--forge filler`` stages no level (the forged session's openings are consistent with its commitments, which the
    forgery changes) and the algebra under ``forged-filler/`` (not staged here: about 12 minutes at the fixture's m)."""
    res = S.stage(inner, _fold(), tmp_path, algebra=False, forge="filler")
    assert res["forge"] == "filler" and res["levels"] == {} and "filler" in S.FORGERIES


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    """A Gemm_v2{K=64} GemmCoordinate class staged in both layouts, 4 x 4 tiles (``tile``, `stage_tiled`) and one coordinate an
    instance (``unit``, `stage`): at 8 and at 5 instances unpadded, and padded to 8 with 5 real."""
    from verity_flock import class_statement as CS
    from verity_flock import partition_units as PU

    CS.cache_slot_circuits()
    defs = {'Gemm_v2{K=64,N=4096,DOT={"fn":"HopperBF16WgmmaDot16_v1"}}': 1}
    program, _calls, _prov = CS.distinct_calls(None, None, defs, None)
    units = PU.partition_units(program, "q-word")
    ((s, cl),) = units.classes().items()
    key = PU.derive(PU.SOURCE, PU.DOMAIN, {"program": "test", "seed": 0})
    salt, shared = hashlib.sha256(b"test").digest(), CS.broadcast_params(units)
    out = {}
    for name, lanes_n, n, real in (("plain", 8, 8, None), ("five", 5, 5, None), ("padded", 5, 8, 5)):
        d = tmp_path_factory.mktemp(f"tile-{name}")
        lanes = CS.class_lanes(units, cl, key, lanes_n * 8, shared)
        out["tile", name] = (d, n, CS.stage_tiled(units, "q-word", s, cl, lanes, d, salt, 4, 4, n, real))
        d = tmp_path_factory.mktemp(f"unit-{name}")
        lanes = CS.class_lanes(units, cl, key, lanes_n, shared)
        out["unit", name] = (d, n, CS.stage(units, "q-word", s, cl, lanes, d, salt, pad=8 if real else None))
    with pytest.raises(ValueError, match="--real 9"):
        CS.stage_tiled(units, "q-word", s, cl, CS.class_lanes(units, cl, key, 72, shared), tmp_path_factory.mktemp("bad"),
                       salt, 4, 4, 8, 9)
    with pytest.raises(ValueError, match="padded to 4"):
        CS.stage(units, "q-word", s, cl, CS.class_lanes(units, cl, key, 5, shared), tmp_path_factory.mktemp("bad"), salt, pad=4)
    return out


def _out_rows(d: Path, n: int) -> list[bytes]:
    text, data = (d / "circuit.txt").read_text(), (d / f"inst-{n}.bin").read_bytes()
    words = next(w for name, w, *_ in json.loads(text.split("\n", 2)[1][5:])["ports"] if name == "out")
    return [data[a:a + 2 * words] for a in (S._offsets(data, text, "out", k)[0] for k in range(n))]


def _header(d: Path, n: int) -> dict:
    with open(d / f"inst-{n}.bin", "rb") as f:
        return json.loads(f.readline())


@pytest.mark.slow
@pytest.mark.parametrize("layout", ["tile", "unit"])
def test_a_padded_statement_has_the_caps_shape_and_its_real_instances(staged, layout):
    """Padded to 8 instances with 5 real, the statement is the unpadded 8's shape (its circuit, its instance count), its real
    instances' output rows are the unpadded statement's, and its three fillers write one output row and add one committed
    row to each port: the padded statement's tables are the unpadded 5's and one row more."""
    (dp, _, plain), (d5, _, five), (dq, _, padded) = (staged[layout, k] for k in ("plain", "five", "padded"))
    assert (padded["real"], padded["fillers"], padded["n"]) == (5, 3, 8) and "real" not in plain
    assert padded["circuit_sha512"] == plain["circuit_sha512"] == five["circuit_sha512"] and S.instances(dq / "pub-8.bin") == 8
    assert S.shape(S.meta(dq / "circuit.txt"), 8) == S.shape(S.meta(dp / "circuit.txt"), 8)
    a, b = _out_rows(dp, 8), _out_rows(dq, 8)
    assert b[:5] == a[:5] == _out_rows(d5, 5) and b[5] == b[6] == b[7] and b[5] not in a[:5]
    h5, hq = _header(d5, 5)["shared_rows"], _header(dq, 8)["shared_rows"]
    assert hq["out"] == 8 and all(hq[p] == h5[p] + 1 for p in h5 if p != "out")


@pytest.mark.slow
@pytest.mark.parametrize("layout", ["tile", "unit"])
def test_forge_filler_moves_one_bit_of_one_fillers_output_row(staged, layout, tmp_path):
    """`forge_filler` flips bit 0 of the first filler's output row and nothing else, into a new file (a hard link to the staged
    file, as the stage cache keeps one, is left as it was); a slot that is a real instance, or a REAL that is not the
    staging's, is refused."""
    dq, _, _ = staged[layout, "padded"]
    d = tmp_path / "s"
    d.mkdir()
    for f in ("circuit.txt", "inst-8.bin", "pub-8.bin"):
        (d / f).write_bytes((dq / f).read_bytes())
    (tmp_path / "cached.bin").hardlink_to(d / "inst-8.bin")
    before = (d / "inst-8.bin").read_bytes()
    rec = RS.forge_filler(d, 8, 5)
    after = (d / "inst-8.bin").read_bytes()
    assert (rec["slot"], rec["now"]) == (5, rec["was"] ^ 1)
    diff = [i for i, (x, y) in enumerate(zip(before, after)) if x != y]
    assert len(before) == len(after) and len(diff) == 1 and before[diff[0]] ^ after[diff[0]] == 1
    assert _out_rows(d, 8)[6] == _out_rows(dq, 8)[6] and _out_rows(d, 8)[5] != _out_rows(dq, 8)[5]
    assert (tmp_path / "cached.bin").read_bytes() == before
    with pytest.raises(ValueError, match="not a filler"):
        RS.forge_filler(d, 8, 5, slot=4)
    with pytest.raises(ValueError, match="not the staging's"):
        RS.forge_filler(dq, 8, 4)
