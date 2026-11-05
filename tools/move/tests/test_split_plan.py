"""`split_plan/plan.py`'s graph: a declaration uses what its proof term holds (Extract.lean) and what its source names
(refs.py's `R` lines), so a `simp` lemma proved by `rfl`, which leaves no trace in the term, still ties it to its module."""
from __future__ import annotations

import importlib.util
from pathlib import Path

PLAN = Path(__file__).resolve().parent.parent / "split_plan" / "plan.py"
spec = importlib.util.spec_from_file_location("plan_under_test", PLAN)
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


def test_a_reference_in_the_source_is_a_use(tmp_path):
    deps = tmp_path / "deps.tsv"
    deps.write_text("M\t0\tProofs.Toy.Setup\nM\t1\tProofs.Toy.Drawn\n"
                    "C\t0\t0\tthm\tToy.throw_bind'\t\n"
                    "C\t1\t1\tthm\tToy.drawn_ok\t\n"
                    "R\tProofs.Toy.Drawn\tToy.drawn_ok\tToy.throw_bind'\n"
                    "R\tProofs.Toy.Drawn\tToy.drawn_ok\tMathlib.not_tracked\n")
    mods, consts = P.load_deps(deps)
    assert mods == {0: "Proofs.Toy.Setup", 1: "Proofs.Toy.Drawn"}
    assert {c[3]: c[4] for c in consts} == {"Toy.throw_bind'": [], "Toy.drawn_ok": [0]}


def test_a_declaration_using_a_matcher_of_its_module_moves_with_it(tmp_path):
    """Lean reuses `liveOfJ.match_1` for an identical `match` only in `liveOfJ`'s module: a theorem there whose term
    holds it, split away from `liveOfJ`, makes its own matcher, and its `rw` stops matching."""
    deps = tmp_path / "deps.tsv"
    deps.write_text("M\t0\tProofs.Toy.NextCs\nM\t1\tProofs.Toy.Uses\n"
                    "C\t0\t0\tdef\tToy.liveOfJ\t1\n"
                    "C\t1\t0\tdef\tToy.liveOfJ.match_1\t\n"
                    "C\t2\t0\tthm\tToy.liveOfJ_points\t0,1,3\n"
                    "C\t3\t0\tthm\tToy.bind_ok\t\n"
                    "C\t4\t0\tthm\tToy.unrelated\t0\n"
                    "C\t5\t1\tthm\tToy.uses_liveOfJ\t0\n")
    mods, consts = P.load_deps(deps)
    src, imp = "Proofs.Toy.NextCs", "Proofs.Toy.Uses"
    plan = P.Plan(mods, consts, {}, {}, 1.0, {src: [], imp: [src]}.get, [src, imp])
    assert {n for _, n in plan.need(src, imp)} == {"Toy.liveOfJ", "Toy.liveOfJ_points", "Toy.bind_ok"}
