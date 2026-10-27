"""Each stratum's circuit file of the declared window Program, built with the auditor's lowering, is what the period's
open commits (`pouw_h1_circuits.build`, `pouw_h1_rehearsal.period_open_argv`): ``--circuit-files`` carries the SHA-512
of each K's file, as infra's gate parses it (`auditor.circuit_files_of`), its ledger's open records it and core's credit
reads it (`credit.committed_files`). The real lowering (C-Flock's `UnitStager`) is the rehearsal's long CPU step, so
here a stand-in writes each file, over the served tests' stand-in unit."""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

from verity.core.protocols.pouw import credit as CR
from verity.core.protocols.pouw.circuit import h1window as HW
from verity.ml.circuits import partition_object as PO
from verity_vllm.protocols.pouw.h1 import served as SV

from tests.protocols import challenger_host as CH
from tests.protocols import pouw_h1_circuits as PC
from tests.protocols import pouw_h1_rehearsal as RH
from tests.protocols.test_pouw_h1_served import _unit

AUDITOR = Path(__file__).resolve().parents[4] / "benchmarks" / "live_challenger" / "auditor.py"
#: two linears at K = 64 and one at K = 128: two strata
SHAPES, MODULES, ROWS, STEPS = [(64, 64), (64, 128), (32, 64)], {0: "q", 1: "down", 2: "o"}, 32, 1


def _sha(b: bytes) -> str:
    return hashlib.sha512(b).hexdigest()


def _auditor(monkeypatch):
    """Infra's gate, ``auditor.py``, as a module of its own (its dataclasses read it from `sys.modules`)."""
    spec = importlib.util.spec_from_file_location("live_challenger_auditor", AUDITOR)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def program(tmp_path, monkeypatch):
    """The declared Program over the stand-in unit, written as the rehearsal writes it (``PROGRAM``, ``PROGRAM_META``),
    with `h1window.tile_unit` the stand-in's, so each K's template is the declared one's; and its descriptor."""
    monkeypatch.setattr(HW, "tile_unit", lambda k, tile, definition=None: _unit()(k))
    raw = SV.descriptor(SV.declared(SHAPES, MODULES, ROWS, STEPS, unit=_unit()))
    d = tmp_path / "program"
    d.mkdir()
    (d / RH.PROGRAM).write_bytes(gzip.compress(raw))
    (d / RH.PROGRAM_META).write_text(json.dumps({"program_digest": _sha(raw),
                                                 "layout": SV.declared_layout(SHAPES, MODULES, ROWS, STEPS)}))
    return d, raw


def test_the_period_open_commits_each_stratums_circuit_file_sha512(program, tmp_path, monkeypatch):
    """Each K of the declared layout gets one circuit file, lowered once from K's template bound to the Program and its
    Q_template_instances partition (templates by ascending K), and the period-open's ``--circuit-files`` is
    ``{k<K>: SHA-512 of K's file}``: as the gate parses it, as its open records it and as core's credit reads it. A
    second build reuses the files under the same bind, and a file's other bytes are another commitment."""
    d, raw = program
    lowered = []

    def stand_in(k, tile, bind, path):
        lowered.append(k)
        path.write_text(f"the k{k} template at {list(tile)}, bound to {json.dumps(bind, sort_keys=True)}\n")
    out = tmp_path / "circuits"
    doc = PC.build(d, out, lower=stand_in)
    assert lowered == [64, 128]
    files = {f"k{k}": _sha((out / PC.file_name(k)).read_bytes()) for k in (64, 128)}
    assert doc["files"] == files and doc["program"] == doc["bind"]["program"] == _sha(raw)
    templates = [doc["templates"]["k64"], doc["templates"]["k128"]]
    assert templates == [PO.template_id(HW.template(k, _unit()(k))) for k in (64, 128)]
    part = PO.build(json.loads(raw), PO.template_instances_query(templates))
    assert doc["bind"]["partition"] == PO.digest(part)
    assert json.loads((out / PC.MANIFEST).read_text()) == doc

    argv = RH.period_open_argv("vllm-h1-p", 1, _sha(raw), 2, None, doc["files"])
    assert argv[:9] == ["period-open", "--name", "vllm-h1-p", "--cap", "1", "--circuit", _sha(raw), "--draw-k", "2"]
    assert _auditor(monkeypatch).circuit_files_of(argv[argv.index("--circuit-files") + 1]) == files
    assert _auditor(monkeypatch).length_of(argv[argv.index("--length") + 1]) == RH.fill_length()
    ledger = CH.Ledger(tmp_path / "windows.jsonl", 1)
    ledger.period_open("vllm-h1-p", circuit=_sha(raw), draw_k=2, circuit_files=files)
    assert json.loads(ledger.lines[0])["body"]["circuit_files"] == files
    if hasattr(CR, "committed_files"):
        assert CR.committed_files(ledger.lines, "vllm-h1-p") == (files, None)

    assert PC.build(d, out, lower=stand_in) == doc and lowered == [64, 128]
    (out / PC.file_name(64)).write_text("another lowering's file\n")
    assert PC.build(d, out, lower=stand_in)["files"] == {**files, "k64": _sha(b"another lowering's file\n")}


def test_a_commit_runs_period_open_commits_its_declared_fixed_length(tmp_path, monkeypatch):
    """``pouw_h1_gpu.py --commit --peer --period P`` opens P with ``--length``, its ``--declare``'s fixed length, and
    ``--circuit-files``, each K's and the live template's, in the form the gate takes and core's credit reads back
    (`credit.committed_length`, `credit.committed_files`), so credit holds its window to both."""
    from verity_vllm.protocols.pouw import h1 as PH

    from tests.protocols import pouw_h1_gpu as GPU
    decl = PH.Static((1,) * 32, 3, 32, 4)
    files = {"k4096": "2b" * 64, "k14336": "3c" * 64, CR.LIVE: "4d" * 64}
    argv = GPU.period_open_argv("o1", "a" * 128, 7, decl, files)
    assert argv[:11] == ["period-open", "--name", "o1", "--cap", "1", "--circuit", "a" * 128, "--draw-k", "7",
                         "--rows", "32"]
    length = _auditor(monkeypatch).length_of(argv[argv.index("--length") + 1])
    assert length == GPU.fixed_length(decl) == {"max_tokens": [3] * 32, "stop": "fixed"}
    assert _auditor(monkeypatch).circuit_files_of(argv[argv.index("--circuit-files") + 1]) == files
    ledger = CH.Ledger(tmp_path / "windows.jsonl", 1)
    ledger.period_open("o1", circuit="a" * 128, draw_k=7, rows=32, length=length, circuit_files=files)
    assert CR.committed_length(ledger.lines, "o1") == ([3] * 32, None)
    assert CR.committed_files(ledger.lines, "o1") == (files, None)


def test_the_live_templates_circuit_file_is_built_beside_the_strata_and_a_commit_reads_them(program, tmp_path):
    """``--live T,S`` adds the live template's circuit file, bound to the live Program of S rounds at T rows under the
    declared Program's PREFIX (core's credit verifies each ``live@<s>`` on ``circuit_files["live"]``), kept on a second
    build. A Commit's period opens on exactly that manifest's files (`pouw_h1_gpu.period_files`): built for its circuit,
    every stratum and the live template, at its own T and S."""
    from types import SimpleNamespace

    from verity.core.protocols.sampled_proofs.one_stage import partition as P

    from tests.protocols import pouw_h1_gpu as GPU
    d, raw = program
    lived = []

    def stand_in_live(T, part, path):
        lived.append(T)
        path.write_text(f"the live template at T = {T}, bound to {P.digest(part)}\n")
    out = tmp_path / "circuits"
    doc = PC.build(d, out, lower=lambda k, tile, bind, path: path.write_text(f"k{k}\n"), live=(ROWS, STEPS),
                   lower_live=stand_in_live)
    part = PC.live_part(ROWS, STEPS, CR._prefix(raw))
    assert lived == [ROWS] and doc["files"][CR.LIVE] == _sha((out / PC.LIVE_FILE).read_bytes())
    assert doc["live_bind"] == {"T": ROWS, "S": STEPS, "partition": P.digest(part), "program": part["program"]}
    assert sorted(doc["files"]) == ["k128", "k64", CR.LIVE]
    assert PC.build(d, out, lower=lambda *_a: pytest.fail("relowered"), live=(ROWS, STEPS),
                    lower_live=stand_in_live) == doc and lived == [ROWS]

    layout = json.loads((d / RH.PROGRAM_META).read_text())["layout"]
    decl = SimpleNamespace(rows=ROWS, steps=STEPS)
    assert GPU.period_files(out, _sha(raw), layout, decl) == doc["files"]
    with pytest.raises(ValueError, match="not the period's"):
        GPU.period_files(out, "a" * 128, layout, decl)
    with pytest.raises(ValueError, match=f"not the window's {ROWS}, {STEPS + 1}"):
        GPU.period_files(out, _sha(raw), layout, SimpleNamespace(rows=ROWS, steps=STEPS + 1))
    with pytest.raises(ValueError, match="no circuit files"):
        GPU.period_files(tmp_path / "none", _sha(raw), layout, decl)
    PC.build(d, out, lower=lambda *_a: None)
    with pytest.raises(ValueError, match="live circuit file is for T = None"):
        GPU.period_files(out, _sha(raw), layout, decl)


def test_a_template_the_declared_program_does_not_define_is_lowered_for_no_stratum(program, tmp_path, monkeypatch):
    """The lowering is of the declared Program's own templates: when K's template (`h1window.template` over
    `tile_unit` at the layout's tile) is none of the descriptor's definitions, no file is built and nothing is
    committed."""
    d, _raw = program
    monkeypatch.setattr(HW, "tile_unit", lambda k, tile, definition=None: _unit((32, 16))(k))
    lowered = []
    with pytest.raises(ValueError, match=r"K = \[64, 128\]'s template .* is not one of the declared Program's"):
        PC.build(d, tmp_path / "circuits", lower=lambda k, *_a: lowered.append(k))
    assert lowered == [] and not (tmp_path / "circuits" / PC.MANIFEST).exists()
