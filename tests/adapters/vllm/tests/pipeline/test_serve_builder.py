"""A served run's registration log registers its compiler (`serve_day.register_compiler`, which `advice.write` calls before it
binds any advice): the log a run declared its day in holds one item of kind ``builder``, core's check accepts the log with it,
and that item is what the run's advice windows are bound under: its digest the one each window's binding names the compiler by,
its ℓ at least each window's advice bits, its format the one that decodes each window's advice. Under the tie the item stands
in the log before the first window's sample coin. A log that registers another compiler refuses this one by core's name, and
no advice is written; a log with site entries takes the item under ``--day-params``' auditor. A run that declared no day
registers nothing."""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from tests.pipeline.test_serve_advice import C, _served
from tests.pipeline.test_serve_day import PROGRAM, _day, _opts, _params
from tests.pipeline.test_serve_h1 import PROMPTS, Later, _Server
from tests.pipeline.test_serve_tie import SEALED, WindowCommit
from tests.protocols.test_pouw_h1_commit import AUDITED, ROWS, STEPS
from tests.protocols.test_pouw_h1_served import _unit
from verity import device as S
from verity.protocols.registration.verifier import registration as G
from verity.protocols.resource_accounting.compute.pouw.prover.serving import h1 as PH
from verity.protocols.resource_accounting.network.verifier import advice as I1
from verity_vllm.engine import step_trace as ST
from verity_vllm.pipeline import advice as ADV
from verity_vllm.pipeline import advice_windows as AW
from verity_vllm.pipeline import serve_day as SD
from verity_vllm.pipeline import serve_h1 as SH
from verity_vllm.pipeline import serve_tie as TIE
from verity_vllm.pipeline import window_advice as WA

pytestmark = pytest.mark.usefixtures("code_versions_stood_in")

#: another registered compiler: the same inner compiler under another ℓ
OTHER = ADV.registered_compiler("test-compiler", "ab" * 64, 1 << 20)

#: a Site of one host with one device, its network attester and its challenger, and the auditor that accepts it
SHEET = S.Sheet(S.DeviceFigures("sim-a", 2_000_000_000, (10**15, 2 * 10**15, 2 * 10**15, 15 * 10**14, 4 * 10**15)),
                3_200_000_000_000, 1 << 25, 64 << 30)
LMS = S.SCHEMES[0]
AUDITOR = G.Auditor((SHEET,), Fraction(1, 20), (("cert0", "k0", LMS), ("chal0", "k1", LMS)),
                    hosts=(("node0", S.HostFigures(1 << 25)),))
SITE = G.site((SHEET,), (S.Loc("site0", "site", time_unit=Fraction(1, 10**9)), S.Loc("node0", "host", "site0", chip=1 << 25),
                         S.Loc("gpu0", "device", "node0", sheet=SHEET.digest()),
                         S.Loc("hbm0", "memory", "gpu0", tier="hbm", bytes=1 << 30),
                         S.Loc("cert0", "certifier", "site0", owner="auditor", key="k0", scheme=LMS),
                         S.Loc("chal0", "challenger", "site0", owner="auditor", key="k1", scheme=LMS)),
              (S.Link("up0", ("node0", "cert0"), 100_000_000_000, 1), S.Link("up1", ("cert0", S.OUTSIDE), 100_000_000_000, 1),
               S.Link("mc0", ("chal0", "node0"), 10_000_000_000, 1)))


def _declared(tmp: Path, day: int = 0) -> Path:
    """A serve dir whose server declared ``day`` in ``tmp/log.json`` before it served: three-step Commit windows, so
    requests are carried from one into the next."""
    out = tmp / "serve"
    out.mkdir()
    assert SD.declare(_opts(tmp, *_day(tmp, day, PROGRAM)), out) is not None
    return _served(out, window_steps=3)


def _view(tmp: Path, params: Path | None = None) -> tuple[dict, list[G.Entry]]:
    """The log's public view under core's protocol check (`registration.check_log`), and its entries."""
    composite, log = G.log_of(json.loads((tmp / "log.json").read_text()))
    verdict = G.check_log(G.params_of(json.loads((params or _params(tmp)).read_text())), (composite, log))
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
    assert (advice, ell, kind, digest) == (WA.SCHEMA, C.cap_bits, "builder", C.digest) and digest != C.inner.digest
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


def test_a_log_with_site_entries_takes_the_item_under_the_day_params_auditor_and_is_refused_without_it(tmp_path):
    """The served path's `verity-vllm advice` (`served_verdicts.sh`'s ``advice_of``, which passes ``--day-params`` under
    ``DAY``) on a day declared in a log whose site entries only ``--day-params``' auditor accepts: without the parameters
    the call is refused by name and writes nothing; with them it registers the item and binds the advice."""
    params = tmp_path / "site-params.json"
    params.write_text(json.dumps(G.params("onsite", AUDITOR)))
    (tmp_path / "log.json").write_bytes(G.committed("onsite", G.chain("onsite", [SITE])))
    out = tmp_path / "serve"
    out.mkdir()
    day = ["--day", "0", "--day-log", str(tmp_path / "log.json"), "--day-program", PROGRAM, "--day-params", str(params)]
    assert SD.declare(_opts(tmp_path, *day), out) is not None
    _served(out, window_steps=3)
    argv: dict[str, Any] = {"run": str(out), "compiler_name": "test-compiler", "compiler_digest": "ab" * 64}
    with pytest.raises(ADV.UsageError, match="its site entries need --day-params' auditor"):
        ADV.main(ADV.Options(**argv))
    assert not (out / "advice").exists() and _view(tmp_path, params)[0]["builder"] is None
    assert ADV.main(ADV.Options(**argv, day_params=str(params))) == 0
    view, log = _view(tmp_path, params)
    assert view["builder"] == ADV.builder_item(C) and [type(e) for e in log] == [S.Entry, G.Declaration, G.Information]
    assert (out / "advice" / "windows.json").exists()


def test_under_the_tie_the_item_stands_in_the_log_before_the_first_windows_sample_coin(tmp_path, monkeypatch):
    """Under the tie (``--h1-window-commit``) the command at each window's close (`served_verdicts.sh`'s ``tie``) runs
    `verity-vllm advice` for the window before window-commit writes R, and the window's calls_root, draw_root and sample
    coin come only once R is bound. So the first window's call registers the item while nothing of the window is bound, and
    its coin opens after; each later window's call finds it."""
    serve = tmp_path / "serve"
    serve.mkdir()
    day = SD.declare(_opts(tmp_path, *_day(tmp_path, 0, AUDITED)), serve)
    srv, runs, seen = _Server(monkeypatch), {}, []
    wc = WindowCommit(runs)

    def tie_command(argv: list[str], log: Any) -> int:
        k = int(argv[5])
        seen.append((k, SD.register_compiler(serve, ADV.builder_item(C)).registered,
                     sorted({"calls_root", "draw", "commit"} & set(runs[k].commits))))
        return wc(argv, log)

    tie = TIE.WindowTie("v3c-window-commit", serve, runner=tie_command, day=day)
    arm = SH.H1Windows(srv.model, srv.com, ROWS, STEPS, engine=object(), out=serve, unit=_unit(), prover=tmp_path / "prover",
                       closer=(closer := Later()), tie=tie)
    pending = []
    for w, prompts in enumerate(PROMPTS):
        arm.open(w)
        runs[w] = srv.serve(prompts, w)
        pending.append(arm.close(w))
        arm.sealed(w, {"window": w, **SEALED})
    closer.run()
    outs = [f.result() for f in pending]
    assert seen == [(w, "now" if w == 0 else "earlier", []) for w in range(len(PROMPTS))]
    for w, out in enumerate(outs):
        assert "error" not in out and out["commit_record"]["named"] is True, out
        draw = json.loads((serve / "h1" / f"w{w}" / PH.CALLS_FILE).read_text())["commitment"]["draw"]
        assert draw["coin"] is not None and "refused" not in draw
    view, log = _view(tmp_path)
    assert view["builder"] == ADV.builder_item(C) and [type(e) for e in log] == [G.Declaration, G.Information]
    arm.drain()
    srv.src.uninstall()
