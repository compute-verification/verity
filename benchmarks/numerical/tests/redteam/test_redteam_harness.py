"""The harness itself: passes a sound checker, catches a weakened one, and its report is well-formed JSON."""
from __future__ import annotations

import pytest

import json
import random

from verity.ml.circuits.silicon import relation as R
from verity.ml.circuits.silicon.silicon import AMPERE_BF16_M16N8K16 as AMP
from verity_numerical.redteam import cases as C
from verity_numerical.redteam import harness as H
from verity_numerical.redteam import relation_target as RT

GOLDEN, MEASURED = C.golden_cases()


def test_silicon_target_is_trivially_sound() -> None:
    t = RT.silicon_target(AMP)
    assert H.run_completeness(t, GOLDEN).status == "PASS"
    assert t.columns(()) == []
    assert H.run_mutations(t, GOLDEN[:10]).status == "PASS"
    x = H.run_coordinated(t, GOLDEN[:30], rounds=0)
    assert x.status == "PASS" and x.detail["accepted_wrong"] == []


def test_tree_get_set_handles_dicts_lists_and_frozen_dataclasses() -> None:
    name, x = GOLDEN[0]
    w = R.step_witness(AMP, R.decode_state(x[0], AMP), list(x[1]), list(x[2]))
    path = ("groups", 1, "out", "M")
    w2 = H.tree_set(w, path, H.tree_get(w, path) + 1)
    assert H.tree_get(w2, path) == H.tree_get(w, path) + 1      # copy modified
    assert w2["groups"][1]["out"].M == w["groups"][1]["out"].M + 1
    assert w2["groups"][0] == w["groups"][0]                    # original untouched elsewhere
    p2 = ("groups", 0, "terms", 3, "aligned")
    assert H.tree_get(H.tree_set(w, p2, 5), p2) == 5 and H.tree_get(w, p2) != 5


def test_relation_columns_cover_every_integer_column() -> None:
    name, x = GOLDEN[1]
    w = R.step_witness(AMP, R.decode_state(x[0], AMP), list(x[1]), list(x[2]))
    cols = RT.relation_columns(AMP, w)
    names = {c.name for c in cols}
    for j, q in enumerate(w["prod"]):
        for k in q:
            assert f"prod.{j}.{k}" in names, k
    for gi, g in enumerate(w["groups"]):
        for k, v in g.items():
            if k == "terms":
                for i, t in enumerate(v):
                    for kk in t:
                        assert f"groups.{gi}.terms.{i}.{kk}" in names
            elif k == "out":
                for f in ("nan", "inf", "neg", "e", "M"):
                    assert f"groups.{gi}.out.{f}" in names
            elif isinstance(v, int):
                assert f"groups.{gi}.{k}" in names, k
    for c in cols:                                              # the honest value lies inside its declared range
        v = H.tree_get(w, c.path)
        assert c.lo <= v < c.hi, (c.name, v, c.lo, c.hi)


@pytest.mark.slow
def test_harness_detects_weakened_checker() -> None:
    """Drop the predicate that binds ``out.M`` of the last group: a single-column mutation is accepted with another
    output, and the harness reports SAT with the concrete input, column and both outputs."""
    weak = RT.relation_target(AMP, dropped=("g1.out.M",))
    f = H.run_mutations(weak, GOLDEN[:2], rng=random.Random(0))
    assert f.status == "SAT"
    hit = f.detail["accepted_wrong"][0]
    assert hit["column"] == "groups.1.out.M" and hit["accepted_output"] != hit["reference"]
    sound = RT.relation_target(AMP)
    assert H.run_mutations(sound, GOLDEN[:2], rng=random.Random(0)).status in ("PASS", "NOTE")


@pytest.mark.slow
def test_mutations_and_coordinated_search_do_not_break_the_integer_relation() -> None:
    t = RT.relation_target(AMP)
    sample = GOLDEN[:3] + C.structured_cases(AMP)[:8]
    f = H.run_mutations(t, sample, rng=random.Random(1))
    assert f.detail["accepted_wrong"] == [], f.detail["accepted_wrong"][:2]
    for free in f.detail["free_columns"]:          # the only free column is the sign of a zero total
        assert free["column"].endswith("neg_t"), free
    x = H.run_coordinated(t, sample[:4], rounds=60, rng=random.Random(2))
    assert x.status == "PASS"


def test_report_json(tmp_path) -> None:
    rep = H.Report(meta={"k": 1})
    rep.add(H.Finding("A1", "t", "a", "PASS", 0.5, case="c", detail={"x": (1, 2)}, reproducer="r"))
    rep.add(H.Finding("A2", "t", "b", "SAT", 1.5))
    out = tmp_path / "r.json"
    rep.write(out)
    doc = json.loads(out.read_text())
    assert doc["counts"] == {"PASS": 1, "SAT": 1} and doc["findings"][0]["detail"]["x"] == [1, 2]
    assert "A2" in rep.table()
