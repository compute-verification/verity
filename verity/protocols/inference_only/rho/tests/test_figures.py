import json

import pytest

pytest.importorskip("matplotlib")

from rho import figures
from rho.params import Registry

REG = Registry.load()
PNG = b"\x89PNG\r\n\x1a\n"
#: one onsite and one remote scenario: every figure has what it needs, and the test stays under a few seconds
NAMES = ["onsite-h200-kimi", "remote-h200-kimi"]


@pytest.mark.parametrize("name", list(figures.FIGURES))
def test_each_figure_writes_a_png_and_the_standard_json_it_plots(tmp_path, name):
    written = figures.render(REG, tmp_path, NAMES, grid=4, only=[name])[name]
    if name in ("class-vs-k", "overheads", "dark"):
        assert written, f"{name} needs only a scenario, and drew nothing"
    assert [p.name for p in written] == ([f"{name}.png", f"{name}.json"] if written else [])
    if written:
        png, js = written
        assert png.read_bytes()[:8] == PNG
        data = json.loads(js.read_text(), parse_constant=lambda c: pytest.fail(f"{js.name} has {c}"))
        assert data


def test_class_vs_k_marks_where_each_class_turns_vacuous(tmp_path):
    figures.class_vs_k(REG, tmp_path, grid=4)
    data = json.loads((tmp_path / "class-vs-k.json").read_text())
    for panel in data["panels"]:
        for name, c in panel["classes"].items():
            ys = [float(y) for y in c["overhead"]]
            if c["vacuous_from_k"] is None:
                assert min(ys) > 1, name
            else:
                assert any(y <= 1 for y in ys), name


def test_jsonable_spells_out_what_json_cannot_hold():
    assert figures.jsonable({"a": [float("inf"), float("-inf"), float("nan"), 1.5], 2: (1,)}) == {
        "a": ["inf", "-inf", None, 1.5], "2": [1]}
