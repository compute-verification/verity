"""The grader's reading of training's slowdown code (`load_slowdown_code`) from a stand-in package in a git checkout of its own:
the commit it is at, checked against the checkout's HEAD, and each refusal with its fix; the command line (`main`) end to end
on the tiny step; and, with VERITY_GRADER_SLOWDOWN_CODE set, the tiny steps read beside the slowdown code itself, whose capture
of the same step agrees product by product, and on sm_120 prices the linears' MACs as the circuit's tensor-core dots. These
tests start git (pyproject.toml's `untraced`)."""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from conftest import CC, LF, STEP, dot_alone, k16_steps, route_factor
from test_grader import SOFTMAX, differences, fused_keys

import grader as G
from verity_config.settings import of

#: the stand-in: the modules the grader calls, at the capture it reads beside
STANDIN = {f"{G.PACKAGE}/__init__.py": 'VERSION = "stand-in"\n',
           f"{G.PACKAGE}/opnames.py": "def gemm_like(name):\n    return name in ('mm', 'bmm')\n",
           f"{G.PACKAGE}/work.py": "HE_BITS = {}\n",
           f"{G.PACKAGE}/capture/__init__.py": "",
           f"{G.PACKAGE}/capture/products.py": f"CAPTURE = {G.CAPTURE!r}\n"}
SLOWDOWN_CODE = of(__file__).VERITY_GRADER_SLOWDOWN_CODE


def _git(d: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(d), *args], check=True, capture_output=True, text=True).stdout.strip()


def _standin(d: Path, capture: str = G.CAPTURE, commit: bool = True) -> str | None:
    """The stand-in written to `d` at `capture`, committed in a git checkout of its own: its HEAD (None, `commit` false: no
    checkout)."""
    for rel, text in STANDIN.items():
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text(text.replace(G.CAPTURE, capture))
    if not commit:
        return None
    _git(d, "init", "-q")
    _git(d, "add", ".")
    _git(d, "-c", "user.name=grader", "-c", "user.email=grader@example.invalid", "-c", "commit.gpgsign=false", "commit", "-q",
         "-m", "stand-in")
    return _git(d, "rev-parse", "HEAD")


def _unload() -> None:
    for m in [m for m in sys.modules if m.split(".")[0] == G.PACKAGE]:
        del sys.modules[m]


@pytest.fixture
def fresh(monkeypatch):
    """sys.path and the slowdown code's modules as they were before the test: the loader puts its directory on sys.path and
    imports its package by name."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    before = {m: v for m, v in sys.modules.items() if m.split(".")[0] == G.PACKAGE}
    yield
    _unload()
    sys.modules.update(before)


def test_the_slowdown_code_is_read_at_the_commit_it_is_at(fresh, tmp_path, monkeypatch) -> None:
    d = tmp_path / "code"
    head = _standin(d)
    code = G.load_slowdown_code(d, head)
    assert (code.version, code.commit, code.checked, code.directory) == ("stand-in", head, True, str(d.resolve()))
    assert code.opnames.gemm_like("bmm") and not code.opnames.gemm_like("mul") and code.products.CAPTURE == G.CAPTURE
    assert G.load_slowdown_code(d, head[:7]).checked
    plain = tmp_path / "plain"
    _standin(plain, commit=False)
    _unload()
    # pytest's tmp_path may sit inside a checkout, whose HEAD git would find above `plain`
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    assert G.load_slowdown_code(plain, "a" * 40).checked is False


def test_the_slowdown_code_is_refused_with_its_fix(fresh, tmp_path) -> None:
    d, old, other = tmp_path / "code", tmp_path / "old", tmp_path / "other"
    head = _standin(d)
    for where, commit, said in ((tmp_path, head, "doesn't hold training's slowdown code"),
                                (d, "HEAD", "'HEAD' isn't a commit"),
                                (d, "0" * 40, f"is at {head[:12]}, not {'0' * 40}: check out")):
        with pytest.raises(G.Refused, match=said):
            G.load_slowdown_code(where, commit)
    with pytest.raises(G.Refused, match="captures with 4.12.8: update the grader to its capture first"):
        G.load_slowdown_code(old, _standin(old, capture="4.12.8"))
    with pytest.raises(G.Refused, match=f"is already loaded from {old.resolve()} in this process"):
        G.load_slowdown_code(other, _standin(other))


def test_the_command_line(fresh, steps, tmp_path, capsys, unit_prices) -> None:
    """`grader.py HANDOVER --slowdown-code DIR --commit SHA [--trace T] --out OUT`: what it read, written to OUT with a summary
    line, beside the trace when given one; a refusal exits 2 with its fix on stderr."""
    d, out, trace = tmp_path / "code", tmp_path / "out.json", tmp_path / "trace.json"
    head = _standin(d)
    argv = [str(steps().path.parent), "--slowdown-code", str(d), "--commit", head, "--out", str(out)]
    assert G.main(argv) == 0
    got = json.loads(out.read_text())
    assert got["format"] == G.READ and got["slowdown_code"]["checked"] and "against" not in got
    t, C = got["totals"], got["C"]["gates"]
    assert capsys.readouterr().out == (f"grader: {t['products']} products (0 of them 0 fused attention ops'), N {t['N']:,}, "
                                       f"O {t['O']:,}, |C| {C:,} ({t['gates']:,} products')\n")
    trace.write_text(json.dumps({"capture": G.CAPTURE, "gates": 500, "rows": []}))
    assert G.main([*argv, "--trace", str(trace)]) == 0
    a = json.loads(out.read_text())["against"]
    assert a["C"]["step"] == {"circuit": C, "trace": 500, "ratio": C / 500} and not a["agree"]["count"]
    assert a["C"]["products"] == {"circuit": t["gates"], "trace": 0, "ratio": None}
    capsys.readouterr()
    assert G.main([str(tmp_path), *argv[1:]]) == 2
    assert capsys.readouterr().err.startswith(f"grader: refused: cannot read a hand-over at {tmp_path / 'handover.json'}")


@pytest.mark.skipif(not SLOWDOWN_CODE, reason="set VERITY_GRADER_SLOWDOWN_CODE=DIR@COMMIT, the directory training's slowdown "
                                              "code's package sits in and its commit, to read the tiny steps beside it (about "
                                              "a minute)")
@pytest.mark.parametrize("step", ["f32", "bf16", "bf16-cublas"])
def test_the_tiny_steps_beside_the_slowdown_code(step, fresh, steps, monkeypatch, request) -> None:
    """The slowdown code's capture of the same tiny step (`capture_step`, its product rows with each GEMM at its operands'
    dtype) beside what the grader reads of the step's circuit: the same products, N, O and token dims at every (phase, op,
    dims), and the same operand widths but at the bf16 steps' attention products, which the circuit wires at the 32 bits
    torch's math path computes them in and the trace takes at the 16 bits their operands were passed at. On sm_120 the step
    keeps the card's flash attention, which the trace computes as the math path does, and they differ at attention's (phase,
    dims) alone (`differences`), in four columns: the count, by the backward's scores, which the math path saves; N and O,
    the head Definitions' multiply-adds and outputs against whole rectangles, where the scores are; and the widths of the
    products with an operand of the softmax's (`SOFTMAX`), bf16 on the card, f32 on the math path. Priced at the tensor-core
    dot alone (`dot_alone`), the linears' |C| is the trace's (its gates a bf16 MAC that dot's over 16) times their routes'
    factor (`route_factor`: the zeros grid.z slices pad their residue tiles with are dots the trace doesn't count)."""
    dtype, cc = {"f32": (None, None), "bf16": (torch.bfloat16, None), "bf16-cublas": (torch.bfloat16, CC)}[step]
    directory, _, commit = SLOWDOWN_CODE.rpartition("@")
    code = G.load_slowdown_code(Path(directory), commit)
    s = steps(dtype, cc)
    if cc:
        dot_alone(monkeypatch, s.program)
    else:
        request.getfixturevalue("unit_prices")
    capture_step = importlib.import_module(f"{G.PACKAGE}.capture.trace").capture_step
    torch.manual_seed(0)
    model = LF.LM(LF.SIZES["tiny"], ctx=2 * STEP[1], vocab=64, matmul_dtype=dtype)
    g = capture_step(model, torch.optim.AdamW(model.parameters(), **LF.ADAMW), LF.square_loss,
                     torch.zeros(*STEP, dtype=torch.long))
    h = G.load_handover(s.path)
    out = G.read(h, code)
    a = G.against(out, code.products.rows(g), code)
    if cc:
        scores = fused_keys(out, ("scores",))
        assert differences(a, out) == {"count": {k for k in scores if k[0] == "backward"}, "N": scores, "O": scores,
                                       "widths": fused_keys(out, SOFTMAX)}, a["rows"]
    else:
        assert a["agree"] == {"count": True, "N": True, "O": True, "widths": dtype is None, "token_dims": True}, a["rows"]
        assert all(a["totals"]["circuit"][x] == a["totals"]["trace"][x] for x in ("count", "N", "O"))
    if cc:
        factor = route_factor(k16_steps(h), out["products"])
        assert a["C"]["mm"]["trace"] > 0 and a["C"]["mm"]["ratio"] == pytest.approx(factor), a["C"]
