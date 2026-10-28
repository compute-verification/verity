"""The recursive inner session's sample: the coin server answers ``Register`` with the verifier's full draw
(`verity_flock.rec_live.full_draw`, the stratified law at K = the population, so every unit) and the session record keeps
it as ``unit_draw``, which `verify` holds to its own law (``--stratified N --program --partition``). `full_draw` is checked
against Lean's own derivation (``flock-verify draw`` at K = N), and the controls `rec_inner.py forge` makes of a drawn
record (no sample, a unit dropped) against what they change. `inner_draw` reads a record's sample as the verifier does and
refuses a record without one, and `rec_vstage stage` refuses to stage V* over such a session."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from verity_flock import rec_live as RL

import test_rec_live as T

HERE = Path(__file__).resolve().parent
LEAN = HERE.parents[2] / "core" / "service" / "flock"
BIN = LEAN / ".lake" / "build" / "bin" / "flock-verify"

DRAW = {"law": "stratified", "population": 2, "k": 2, "strata": [{"template": "t", "units": [[0, 2]], "k": 2}],
        "units": [0, 1]}


def _proxy(draw=None):
    return RL.Proxy(RL.CoinServer(T._counter_rand()), T._salts(), schedule=T.SCHEDULE, draw=draw)


def test_register_is_answered_with_the_verifiers_draw_and_the_record_keeps_it():
    proxy = _proxy(DRAW)
    resp = T._session(proxy)
    assert resp[0] == (RL.RESP_DRAW, RL.canonical(DRAW)) and resp[-1][0] == RL.RESP_VERDICT
    assert proxy.session_record()["unit_draw"] == DRAW
    assert T._session(_proxy())[0] == (RL.RESP_DRAW, "null")


def test_a_draw_is_a_unit_draw_object():
    with pytest.raises(ValueError, match="unit_draw object"):
        _proxy({"law": "stratified", "units": [0]})


def _programs():
    sys.path.insert(0, str(HERE))
    import stratified_agree as SA
    from verity.core.protocols.sampled_proofs.one_stage import partition as P

    t = SA._templates(2)
    for members in ([(t[0], 5)], [(t[0], 2), (t[1], 3), (t[0], 1)]):
        prog = P.mixed_population_program(members)
        tids = list(dict.fromkeys(P.template_id(x) for x, _ in members))
        q = P.template_instance_query(tids[0]) if len(tids) == 1 else P.template_instances_query(tids)
        yield prog, q


def test_the_full_draw_is_every_unit_and_each_stratum_whole():
    from verity.core.protocols.sampled_proofs.one_stage import draw as D

    for prog, q in _programs():
        d = RL.full_draw(prog, q)
        n = d["population"]
        assert d["units"] == list(range(n)) and d["k"] == n and D.proved(d) == list(range(n))
        assert all(s["k"] == sum(hi - lo for lo, hi in s["units"]) for s in d["strata"])
        assert D.strata_problem(D.law_of(d), n) is None


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_the_full_draw_is_the_lean_verifiers_own_at_k_the_population(tmp_path):
    """`verify --stratified N` derives the law from its own program and partition object and draws by it: at K = N every
    stratum samples all its units, whatever the coins, so its sample is `full_draw`'s, byte for byte once canonical."""
    from verity.core.protocols.sampled_proofs.one_stage import partition as P
    from verity.ml.circuits.codec import canonical_json, encode_program

    subprocess.run(["lake", "build", "flock-verify"], cwd=LEAN, check=True, capture_output=True)
    for i, (prog, q) in enumerate(_programs()):
        desc, part = tmp_path / f"{i}.desc", tmp_path / f"{i}.partition.json"
        desc.write_bytes(canonical_json(encode_program(prog)))
        part.write_text(json.dumps(P.partition(prog, q)))
        want = RL.full_draw(prog, q)
        p = subprocess.run([str(BIN), "draw", "--program", str(desc), "--partition", str(part), "--stratified",
                            str(want["population"])], capture_output=True, text=True, check=True)
        got = json.loads(p.stdout.removeprefix("DRAW "))
        assert RL.canonical(got) == RL.canonical(want), (got, want)


def test_the_forged_records_lose_the_draw_or_a_unit(tmp_path):
    sys.path.insert(0, str(HERE.parent / "pod"))
    import rec_inner as RI

    for what in ("nodraw", "partial"):
        d = tmp_path / what / "prover"
        d.mkdir(parents=True)
        (d / "session.json").write_text(json.dumps({"unit_draw": DRAW}))
        assert RI.forge(what, tmp_path / what, record="prover") == {"forged": what, "record": "prover"}
        got = json.loads((d / "session.json").read_text())["unit_draw"]
        assert got == (None if what == "nodraw" else {**DRAW, "units": [0]})


def test_the_inner_draw_is_the_records_and_a_record_without_one_is_refused():
    """`inner_draw` reads the sample the proxy's record keeps, and refuses a record that keeps none, or none readable, with the
    verifier's reason (`RecSetup.noInnerDraw`; `test_vstar_difftest.test_inner_draw_agrees` holds the two readers alike)."""
    proxy = _proxy(DRAW)
    T._session(proxy)
    assert RL.inner_draw(json.dumps(proxy.session_record())) == DRAW
    undrawn = _proxy()
    T._session(undrawn)
    for record in (json.dumps(undrawn.session_record()), "{}", "[]", "not json"):
        with pytest.raises(ValueError) as e:
            RL.inner_draw(record)
        assert str(e.value) == RL.NO_INNER_DRAW, record


def test_vstage_refuses_to_stage_over_an_undrawn_inner_session(tmp_path, capsys):
    """`rec_vstage stage` over an inner session whose record holds no sample exits 2 before it reads the fold: the verifier
    refuses every session over it (`RecSetup.innerDrawn`)."""
    from verity_flock import rec_vstage as S

    undrawn = _proxy()
    T._session(undrawn)
    undrawn.write(tmp_path / "inner")
    assert S.main(["stage", str(tmp_path / "inner"), str(tmp_path / "no-fold.json"), str(tmp_path / "out")]) == 2
    assert json.loads(capsys.readouterr().out) == {"refused": RL.NO_INNER_DRAW}
    assert not (tmp_path / "out").exists()
