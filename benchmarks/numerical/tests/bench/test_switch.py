"""The switch digest (``bench.switch``): every frozen Table 2 number beside its replacement or the reason it left."""

from __future__ import annotations

import pytest

from verity.ml.circuits.targets.target import FIRST, FP8_ADA_MMA
from verity_numerical.bench import switch as S
from verity_numerical.bench import tables as T

pytest.importorskip("research.store.local", reason="research.store (tools/research) is not importable")
from research.store.local import LocalStore  # noqa: E402
from test_tables import hashed, put, result, store  # noqa: E402, F401  (store: the fixture)
from test_views import committed  # noqa: E402


def test_each_old_number_is_shown_beside_its_replacement_or_the_reason_it_left(store: LocalStore, capsys: pytest.CaptureFixture[str]) -> None:
    gkr = put(store, result(t_total=4.0))
    hashed_cell = put(store, hashed("poseidon2", t_total=0.69))
    new = committed(store, t_total=0.4)
    d = S.build(store)
    assert d["parity"] == [] and d["old_cells"] == len(T.build(store).cells) == 2
    rows = {r["artifact"]: r for r in d["rows"]}
    assert rows[gkr]["line"] == FIRST.name and rows[gkr]["left"]["code"] == "R" and rows[gkr]["replacements"] == []
    assert rows[gkr]["empty"] == ["results only for a weaker statement (A-GKR CPU (Rust))"]
    h = rows[hashed_cell]
    assert h["line"] == FP8_ADA_MMA.name and h["column"] == "B-Ligero + in-proof hash" and h["family"] == "B-Ligero"
    assert h["left"]["code"] == "F"                                  # the fixture preserves no proof files
    assert [c["artifact"] for c in h["replacements"]] == [new]          # the + in-proof hash column folds into its family
    assert d["added"] == []
    md = S.render_markdown(d)
    assert md.startswith(f"> **{S.LABEL}**") and "Parity (the switch's gate 2" in md and ": ok." in md
    assert f"| {FIRST.name} | A-GKR | {T.fmt_overhead(rows[gkr]['overhead'])} ({gkr[:12]}…) | R — weaker statement" in md
    assert S.main(["--root", str(store.root)]) == 0 and capsys.readouterr().out == md


def test_a_new_cell_with_no_old_number_is_listed(store: LocalStore) -> None:
    new = committed(store, t_total=0.4)
    d = S.build(store)
    assert d["rows"] == [] and [c["artifact"] for c in d["added"]] == [new]
    md = S.render_markdown(d)
    assert "(no published number before the switch —)" in md and "## New cells with no old number" in md
