"""`protocols.pous`: what the option refuses, where its salt comes from and how the store's package loads, on the CPU.

The model, its small point and the challenger's in-process session are `test_pous_option`'s.  Two things here are
stand-ins, each named where it is used: vLLM (`tests.fake_vllm`, for the linear layers' base type) and the period's Lean
tools (`period.serve` replaced by a function that answers as the test says).  So the tests of a monitoring period's
subprotocol hold what the option does with a tool's answer, and nothing about the tools.
"""
from __future__ import annotations

import importlib
import json
import re
import sys
from typing import Any

import pytest
import torch
from verity.protocols.accounting.pous.device import challenger_session as CS
from verity.protocols.accounting.pous.verifier import period, setup

from tests import fake_vllm
from tests.protocols.test_pous_option import (
    COINS,
    ENTRY,
    NODE,
    P2,
    SMALL,
    SOURCE,
    TEST_PROTOCOL,
    Linear,
    Model,
    _ctx,
    _in_process_party,
    _period_dir,
    ts,
)
from verity_vllm import protocols as PO
from verity_vllm.engine import hooks
from verity_vllm.protocols import pous
from verity_vllm.protocols.interface import Target

WRONG_SOURCE = "salt_source is hex, a callable of the commitment or {'command': argv}, not "
TOOLS = {"pous-run-protocol": "RUN", "pous-setup": "SETUP"}


@pytest.fixture
def toy(monkeypatch):
    """`test_pous_option`'s model at its small point, its stand-in linear type the option's."""
    monkeypatch.setattr(pous, "linear_types", lambda: (Linear,))
    monkeypatch.setattr(pous, "_base", lambda scheme: SMALL)
    return Model()


@pytest.fixture
def packages():
    """Three package names for `pous._package`, each out of ``sys.modules`` again after the test, with its modules."""
    names = ("pous_package_plain", "pous_package_with_init", "pous_package_broken")
    yield names
    for loaded in [m for m in sys.modules if m.split(".")[0] in names]:
        del sys.modules[loaded]


def _settings(source):
    return {"pous": {"salt_source": source, "experimental": True}}


def _untouched(toy, before):
    """Whether an install left nothing: no hook is live that was not, and every layer holds its plaintext."""
    held = all(torch.equal(m.weight, toy.plain["q" if name == "head" else name]) for name, m in toy.layers.items())
    return held and hooks.live() == before


def test_a_salt_source_of_another_type_is_refused_by_the_selector_and_by_the_install_itself(toy):
    assert pous.refusals(P2, _ctx(ENTRY, settings=_settings(lambda commitment: b"\x01"))) == ()
    for source, kind in ((7, "int"), ({"command": []}, "dict"), ({"command": "true"}, "dict"),
                         ({"command": ["true"], "cwd": "/"}, "dict")):
        assert pous.refusals(P2, _ctx(ENTRY, settings=_settings(source))) == (WRONG_SOURCE + kind,)
    before, target = hooks.live(), Target(model=toy, context=_ctx(ENTRY, settings=_settings(7)))
    with pytest.raises(ValueError, match=re.escape(f"{ENTRY}: {WRONG_SOURCE}int") + "$"):
        pous.install(P2, target)                                                # the adapter's own guard, no selector before it
    assert target.made == [] and _untouched(toy, before)


def test_a_callable_salt_source_is_asked_once_with_the_commitment_and_what_it_returns_makes_the_salt(toy):
    asked = []

    def source(commitment):
        asked.append(commitment)
        return bytearray.fromhex(SOURCE)

    with PO.install(Model(), _ctx(ENTRY)) as comp:                                  # the same model; the settings' hex is the same bytes
        by_hex = comp.get("pous").commitment()
    with PO.install(toy, _ctx(ENTRY, settings=_settings(source))) as comp:
        c = comp.get("pous").commitment()
        assert asked == [bytes.fromhex(c["commitment"])]
        assert (c["commitment"], c["salt"]) == (by_hex["commitment"], by_hex["salt"])
        assert comp.info()[ENTRY]["salt_source"] == "callable"


def test_a_salt_command_that_fails_is_refused_with_its_exit_code_and_its_last_words_and_leaves_nothing(toy):
    script = "import sys; sys.stdin.read(); sys.stderr.write('the auditor is away\\n'); sys.exit(3)"
    before = hooks.live()
    said = f"{ENTRY}: the salt source's command exited 3: the auditor is away"
    with pytest.raises(RuntimeError, match=re.escape(said) + "$"):
        PO.install(toy, _ctx(ENTRY, settings=_settings({"command": [sys.executable, "-c", script]})))
    assert _untouched(toy, before)


def test_c_star_answers_only_its_own_blocks_and_none_once_it_is_freed(toy):
    comp = PO.install(toy, _ctx(ENTRY))
    opt, s = comp.get("pous"), comp.get("pous").store
    blocks = s.held().blocks
    assert len(blocks) == s.params.blocks == 16 and blocks[15] == s.block(15) == opt.answer(15)
    for i in (-1, 16):
        with pytest.raises(IndexError, match=rf"block {i} is outside C\*'s 16$"):
            s.block(i)
    comp.close()
    for read in (lambda: s.block(0), lambda: blocks[0]):
        with pytest.raises(RuntimeError, match=r"C\* is freed$"):
            read()
    with pytest.raises(RuntimeError, match=re.escape(f"{ENTRY} is closed: nothing is held to audit") + "$"):
        opt.audit(NODE, COINS)


def test_the_cpu_decode_refuses_a_range_whose_last_block_holds_more_of_w(toy):
    with PO.install(toy, _ctx(ENTRY)) as comp:
        s, w = comp.get("pous").store, toy.plain["q"].numpy().tobytes()
        assert s.decode_cpu(0, 384) == w and s.decode_cpu(126, 126) == w[126:252]     # a tensor whole; one block whole
        assert any(w[100:126])
        with pytest.raises(ValueError, match="the block holding W byte 99 is nonzero past the tensor's end$"):
            s.decode_cpu(0, 100)


def test_a_weight_that_is_no_parameter_of_the_model_is_refused_and_leaves_nothing(toy, monkeypatch):
    named = [(name, p) for name, p in toy.named_parameters() if name != "k.weight"]
    monkeypatch.setattr(toy, "named_parameters", lambda: iter(named))
    before = hooks.live()
    said = f"{ENTRY}: the weights of ['k'] are no parameters of the model, so no reader could find them"
    with pytest.raises(ValueError, match=re.escape(said) + "$"):
        PO.install(toy, _ctx(ENTRY))
    assert _untouched(toy, before)


def test_weights_on_two_devices_are_refused_before_anything_is_committed(toy):
    toy.layers["k"] = Linear(torch.zeros(40, 16, dtype=torch.int8, device="meta"), torch.ones(40, 1))
    said = f"{ENTRY}: the weights are on ['cpu', 'meta'], and C* is held on one device"
    with pytest.raises(ValueError, match=re.escape(said) + "$"):
        pous.commit_weights(P2, Target(model=toy, context=_ctx(ENTRY)))


def test_an_encoder_whose_blocks_do_not_hold_w_is_caught_at_a_segments_first_block_and_leaves_nothing(toy, monkeypatch):
    encoder = pous._p2("store").CpuEncoder
    encode = encoder.encode
    monkeypatch.setattr(encoder, "encode", lambda self, s, w_seg: encode(self, s, bytes(len(w_seg))))   # codewords of zeros
    before = hooks.live()
    with pytest.raises(RuntimeError, match=r"segment 0 block 0: the reference decode of C\* is not W's payload$"):
        PO.install(toy, _ctx(ENTRY))
    assert _untouched(toy, before)


def test_the_stores_package_loads_from_its_files_with_or_without_an_init_and_never_half(tmp_path, packages):
    plain, with_init, broken = packages
    path = list(sys.path)
    for name in packages:
        (tmp_path / name).mkdir()
        (tmp_path / name / "leaf.py").write_text("VALUE = 5\n")
        (tmp_path / name / "reader.py").write_text(f"from {name}.leaf import VALUE\n")
    (tmp_path / with_init / "__init__.py").write_text("FROM_INIT = 1\n")
    (tmp_path / broken / "__init__.py").write_text("raise RuntimeError('half an init')\n")

    for name in (plain, with_init):
        pous._package(name, tmp_path / name)
        module = sys.modules[name]
        assert importlib.import_module(f"{name}.reader").VALUE == 5                    # its modules import one another by its name
        assert list(module.__path__) == [str(tmp_path / name)]
        pous._package(name, tmp_path / "nowhere")                                      # loaded once: a second call reads no file
        assert sys.modules[name] is module
    assert sys.modules[with_init].FROM_INIT == 1 and not hasattr(sys.modules[plain], "FROM_INIT")
    with pytest.raises(RuntimeError, match="half an init"):
        pous._package(broken, tmp_path / broken)
    assert broken not in sys.modules and sys.path == path


def test_the_layers_pous_encodes_are_vllms_linear_layers_that_hold_a_weight(monkeypatch):
    """vLLM is `tests.fake_vllm` here, and its ``LinearBase`` a class of this test's."""

    class LinearBase:
        def __init__(self, weight):
            self.weight = weight

    class Net:
        def __init__(self, modules):
            self.modules = modules

        def named_modules(self):
            yield "", self
            yield from self.modules.items()

    fake_vllm.install(monkeypatch).set("vllm.model_executor.layers.linear:LinearBase", LinearBase)
    monkeypatch.setattr(pous, "_base", lambda scheme: SMALL)
    assert pous.linear_types() == (LinearBase,)
    weight = torch.arange(32, dtype=torch.int8).reshape(2, 16)
    net = Net({"proj": LinearBase(weight), "bare": LinearBase(None), "other": Linear(weight.clone(), torch.ones(2, 1))})
    got = pous.commit_weights(P2, Target(model=net, context=_ctx(ENTRY)))
    assert (got["tensors"], got["bytes"]) == (1, 32)
    layout = pous._layout(P2, {"proj": weight})
    assert got["commitment"] == setup.commit(layout.params, layout.pack({"proj": weight.numpy().tobytes()})).hex()


def _lean(monkeypatch, tmp_path):
    """Stand-ins for a period's parties and tools: the challenger's session in this process (`test_pous_option`'s), and
    `period.serve` answering for the tool named ``RUN`` as ``say["run"]`` makes of the Python's answer, and for any other
    as ``say["setup"]``.  Returns ``say``, the Python's runs and what was served, each in order."""
    session, _ = _in_process_party(ts.TestSigner(b"pous-vllm", directory=tmp_path), ts.session_log(tmp_path))
    monkeypatch.setattr(CS, "session", session)
    audit, runs, served = period.audit, [], []
    say: dict[str, Any] = {"run": lambda answer: answer | {"reason": "the Lean's words"}, "setup": {"verdict": "accept"}}

    def audited(*args, **kw):
        runs.append(audit(*args, **kw))
        return runs[-1]

    def serve(exe, cases):
        served.append((exe, list(cases)))
        return [say["run"](runs[-1].answer) if exe == "RUN" else say["setup"]]

    monkeypatch.setattr(period, "audit", audited)
    monkeypatch.setattr(period, "serve", serve)
    return say, runs, served


def _subprotocol(opt, tmp_path, n):
    """Monitoring period ``n``'s subprotocol of ``opt``, its ``period.json`` naming `TOOLS`; the answer and what
    ``pous.json`` holds."""
    directory, out = _period_dir(tmp_path, n), tmp_path / f"out{n}"
    described = json.loads((directory / "period.json").read_text()) | {"tools": TOOLS}
    (directory / "period.json").write_text(json.dumps(described))
    got = opt.leg(directory, NODE, out, schemes=ts.SCHEMES, protocol=TEST_PROTOCOL)
    return got, json.loads((out / "pous.json").read_text())


def test_where_a_period_names_the_lean_tools_the_subprotocols_answer_is_theirs_and_must_be_the_pythons(toy, tmp_path, monkeypatch):
    """The tools are stand-ins (`_lean`).  The run tool is asked every period with the run's case and the setup tool at
    the first alone, with the setup's; the subprotocol answers with the run tool's answer, and refuses one that differs
    from the Python's in more than its reason."""
    say, runs, served = _lean(monkeypatch, tmp_path)
    with PO.install(toy, _ctx(ENTRY)) as comp:
        opt = comp.get("pous")
        got, doc = _subprotocol(opt, tmp_path, 0)
        assert served == [("RUN", [runs[0].case]), ("SETUP", [runs[0].setup_case])]
        assert runs[0].answer["verdict"] == "accept" and runs[0].answer["reason"] != "the Lean's words"
        assert got == runs[0].answer | {"reason": "the Lean's words", "profile": got["profile"]} == doc["answer"]
        assert doc["lean"] == {"run": runs[0].answer | {"reason": "the Lean's words"}, "setup": {"verdict": "accept"}}

        got, doc = _subprotocol(opt, tmp_path, 1)
        assert served[2:] == [("RUN", [runs[1].case])] and set(doc["lean"]) == {"run"} and got["reason"] == "the Lean's words"

        say["run"] = lambda answer: answer | {"verdict": "reject"}
        with pytest.raises(RuntimeError, match="^pous-run-protocol answered .*'verdict': 'reject'.*, the Python .*'verdict': 'accept'"):
            _subprotocol(opt, tmp_path, 2)
        assert not (tmp_path / "out2").exists()


def test_a_setup_the_lean_refuses_fails_the_first_period_whatever_the_python_said(toy, tmp_path, monkeypatch):
    """The tools are stand-ins (`_lean`)."""
    say, runs, served = _lean(monkeypatch, tmp_path)
    say["setup"] = {"verdict": "reject", "reason": "C* is not the encoding"}
    with PO.install(toy, _ctx(ENTRY)) as comp:
        with pytest.raises(RuntimeError, match="^pous-setup refused the setup the Python accepted: .*C\\* is not the encoding"):
            _subprotocol(comp.get("pous"), tmp_path, 0)
        assert runs[0].answer["verdict"] == "accept" and [exe for exe, _ in served] == ["RUN", "SETUP"]
        assert not (tmp_path / "out0").exists()
