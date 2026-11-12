"""A served run's registration log registers its compiler (`serve_day.register_compiler`, which `advice.write` calls before it
binds any advice): the log a run declared its day in holds one item of kind ``builder``, core's check accepts the log with it,
and that item is what the run's advice windows are bound under: its digest the one each window's binding names the compiler by,
its ℓ at least each window's advice bits, its format the one that decodes each window's advice. A log that registers another
compiler refuses this one by core's name, and no advice is written. A run that declared no day registers nothing."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.pipeline.test_serve_advice import C, _served
from tests.pipeline.test_serve_day import PROGRAM, _day, _opts, _params
from verity.hardware import registration as G
from verity.protocols.certifier.verifier import advice as I1
from verity_vllm.engine import step_trace as ST
from verity_vllm.pipeline import advice as ADV
from verity_vllm.pipeline import advice_windows as AW
from verity_vllm.pipeline import serve_day as SD
from verity_vllm.pipeline import window_advice as WA

pytestmark = pytest.mark.usefixtures("code_versions_stood_in")

#: another registered compiler: the same inner compiler under another ℓ
OTHER = ADV.registered_compiler("test-compiler", "ab" * 64, 1 << 20)


def _declared(tmp: Path, day: int = 0) -> Path:
    """A serve dir whose server declared ``day`` in ``tmp/log.json`` before it served: three-step Commit windows, so
    requests are carried from one into the next."""
    out = tmp / "serve"
    out.mkdir()
    assert SD.declare(_opts(tmp, *_day(tmp, day, PROGRAM)), out) is not None
    return _served(out, window_steps=3)


def _view(tmp: Path) -> tuple[dict, list[G.Entry]]:
    """The log's public view under core's protocol check (`registration.check_log`), and its entries."""
    composite, log = G.log_of(json.loads((tmp / "log.json").read_text()))
    verdict = G.check_log(G.params_of(json.loads(_params(tmp).read_text())), (composite, log))
    assert verdict[0] == "accept", verdict
    return verdict[1], log


def _bindings(out: Path, rep: dict) -> dict[int, tuple[dict, bytes]]:
    """Each advice window the run wrote: its registration's binding and its advice."""
    return {int(w): (json.loads((out / "advice" / f"w{w}" / "registration.json").read_text())["binding"],
                     (out / "advice" / f"w{w}" / "private" / "advice.bin").read_bytes()) for w in rep["windows"]}


def test_a_served_runs_log_registers_the_compiler_its_advice_windows_are_bound_under(tmp_path):
    out = _declared(tmp_path)
    rep = ADV.write(out, compiler=C)
    view, log = _view(tmp_path)
    assert [type(e) for e in log] == [G.Declaration, G.Information] and [list(e.items) for e in log[1:]] == [[view["builder"]]]
    assert view["builder"] == ADV.builder_item(C) == rep["builder"]["item"] and rep["builder"]["registered"] == "now"
    assert rep["builder"]["head"] == view["head"] == log[-1].digest()
    advice, ell, digest, kind = (view["builder"][k] for k in G.BUILDER_KEYS)
    bindings = _bindings(out, rep)
    assert rep["refused"] == {} and len(bindings) > 1
    for bound, data in bindings.values():
        assert bound["compiler"] == {"name": AW.CAPPED, "digest": digest}
        assert 0 < bound["advice"]["bits"] <= 8 * len(data) <= ell
        assert WA.encode(WA.decode(data)) == data
    assert (advice, kind, digest) == (WA.SCHEMA, "builder", C.digest) and digest != C.inner.digest
    carrying = [data for _, data in bindings.values() if WA.decode(data).carried]
    assert carrying
    for data in carrying:
        with pytest.raises(ValueError, match="advice version"):
            I1.decode(data)


def test_each_later_call_reads_the_registered_compiler_again_and_leaves_the_log_as_it_is(tmp_path):
    """One call a window, as a run that binds each window's advice at its close makes them: the first registers the
    compiler, the others find it."""
    out = _declared(tmp_path)
    first, *later = sorted({r["window"] for r in ST.load(out / ST.SERVED) if r["kind"] == "window"})
    assert ADV.write(out, out / "advice" / f".w{first}", compiler=C, window=first)["builder"]["registered"] == "now"
    body = (tmp_path / "log.json").read_bytes()
    for w in later:
        rep = ADV.write(out, out / "advice" / f".w{w}", compiler=C, window=w)
        assert rep["builder"] == {"item": ADV.builder_item(C), "log": str(tmp_path / "log.json"), "head": _view(tmp_path)[0]["head"],
                                  "registered": "earlier"}
    assert (tmp_path / "log.json").read_bytes() == body and len(_view(tmp_path)[1]) == 2


def test_a_log_that_registers_another_compiler_refuses_this_one_and_no_advice_is_written(tmp_path):
    out = _declared(tmp_path)
    ADV.write(out, tmp_path / "first", compiler=C)
    body = (tmp_path / "log.json").read_bytes()
    with pytest.raises(ValueError, match="refuses the item, entry 2: duplicate-information") as e:
        ADV.write(out, compiler=OTHER)
    assert json.dumps(ADV.builder_item(C), sort_keys=True) in str(e.value) and str(tmp_path / "log.json") in str(e.value)
    assert (tmp_path / "log.json").read_bytes() == body and not (out / "advice").exists()
    assert _view(tmp_path)[0]["builder"] == ADV.builder_item(C)


def test_the_compiler_is_registered_only_in_the_log_that_holds_the_runs_declaration(tmp_path):
    out = _declared(tmp_path)
    (tmp_path / "day1").mkdir()
    other = tmp_path / "day1" / "log.json"
    SD.append(other, G.Declaration(1, PROGRAM), G.params_of(json.loads(_params(tmp_path).read_text())))
    body = other.read_bytes()
    with pytest.raises(ValueError, match="the log holds no declaration"):
        ADV.write(out, compiler=C, day_log=other)
    with pytest.raises(ValueError, match="no log there"):
        ADV.write(out, compiler=C, day_log=tmp_path / "none.json")
    assert other.read_bytes() == body and _view(tmp_path)[0]["builder"] is None and not (out / "advice").exists()
    copy = tmp_path / "copy.json"
    copy.write_bytes((tmp_path / "log.json").read_bytes())
    rep = ADV.write(out, compiler=C, day_log=copy, day_params=_params(tmp_path))
    assert rep["builder"]["log"] == str(copy) and G.log_of(json.loads(copy.read_text()))[1][-1].items == (ADV.builder_item(C),)
    assert _view(tmp_path)[0]["builder"] is None


def test_a_run_that_declared_no_day_registers_nothing(tmp_path):
    out = _served(tmp_path, window_steps=3)
    rep = ADV.write(out, compiler=C)
    assert "builder" not in rep and rep["windows"] and not (out / SD.FILE).exists()
    with pytest.raises(ValueError, match="go with a run that declared a day"):
        ADV.write(out, tmp_path / "again", compiler=C, day_log=tmp_path / "log.json")
    assert not (tmp_path / "again").exists()
