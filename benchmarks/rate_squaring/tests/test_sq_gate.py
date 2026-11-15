"""`sq_gate.py` on gate lines written here: it passes squarings that are right, and fails a wrong one and an empty gate."""

from __future__ import annotations

import io
import json

import pytest
import sq_gate

BITS, C = 512, 569
P = 2**BITS - C


def line(form: str, k: int, x0: int, xk: int | None = None) -> str:
    return f"{form} {k} {BITS} {C} {x0:x} {pow(x0, 1 << k, P) if xk is None else xk:x}\n"


def run(lines: list[str], monkeypatch, capsys) -> tuple[int, dict]:
    monkeypatch.setattr("sys.stdin", io.StringIO("".join(lines)))
    return sq_gate.main(), json.loads(capsys.readouterr().out)


def test_right_squarings_pass_and_an_input_above_p_is_reduced(monkeypatch, capsys):
    lines = [line("fold5", 1, 3), line("fold3", 7, P - 2), line("fold7", 64, (1 << 500) + 12345),
             f"fold5 3 {BITS} {C} {P + 5:x} {pow(5, 8, P):x}\n"]
    assert run(lines, monkeypatch, capsys) == (0, {"words": 4, "bad": 0, "ok": True})


@pytest.mark.parametrize("wrong", [pow(3, 4, P) + 1, pow(3, 4, P) + P, 0])
def test_one_wrong_or_unreduced_result_fails(wrong, monkeypatch, capsys):
    lines = [line("fold5", 1, 7), line("fold5", 2, 3, wrong)]
    assert run(lines, monkeypatch, capsys) == (1, {"words": 2, "bad": 1, "ok": False})


def test_a_gate_with_no_line_fails(monkeypatch, capsys):
    assert run([], monkeypatch, capsys) == (1, {"words": 0, "bad": 0, "ok": False})
