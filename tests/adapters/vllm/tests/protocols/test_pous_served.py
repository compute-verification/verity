"""PoUS attached by its scheme's name on the build's exchange (`compose.schemes.get("pous")`, `messages.Exchange`), over a
tiny int8 model's weights, on the CPU: what it holds, what it refuses, its period and its rejects.

The weights are a toy int8 model's (`Model`), registered as a served run registers them (`RegisteredValues`, a tied weight
once), and P2 runs at the small point's 1,024-bit blocks, 8 a segment (`weight_store.geometry`), so encoding takes
milliseconds. No forward runs: a linear block is computed through the exchange with the weight decoded from ``C*``.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import itertools
import json
import re
import socket
import subprocess
import sys
from concurrent import futures
from pathlib import Path
from typing import Any

import pytest
import torch
from verity import hooks as H
from verity.protocols.compose import schemes
from verity.protocols.resource_accounting.memory.pous.attester import challenger_party as CP
from verity.protocols.resource_accounting.memory.pous.attester import challenger_session as CS
from verity.protocols.resource_accounting.memory.pous.attester import memory_attester as mc
from verity.protocols.resource_accounting.memory.pous.prover import challenger_prover as CR
from verity.protocols.resource_accounting.memory.pous.prover import period_prover
from verity.protocols.resource_accounting.memory.pous.prover import served
from verity.protocols.resource_accounting.memory.pous.prover import weight_store as ws
from verity.protocols.resource_accounting.memory.pous.verifier import period, setup
from verity.protocols.resource_accounting.memory.pous.verifier.schemes.p2 import P2Params
from verity.runtime.stream import Side

from verity_vllm import messages

P2 = "p2-16448/v4"
SOURCE = "a5" * 32
NODE, AUDITOR = "vy-dc-h200-3", "vy-auditor-2"
COINS = bytes(range(64))
SMALL = P2Params(period.W, period.C0, 8)
ROOT = Path(__file__).resolve().parents[5]


def _signer_module():
    path = ROOT / "tests" / "verity" / "protocols" / "resource_accounting" / "memory" / "pous" / "verifier" / "pous_test_signer.py"
    spec = importlib.util.spec_from_file_location("pous_test_signer", path)
    assert spec is not None and spec.loader is not None, path
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ts = _signer_module()
TEST_PROTOCOL = mc.TEST_PREFIX + period.PROTOCOL


class Linear:
    """An int8 linear as the int8_exact target holds one: its weight [N, K] and its f32 ``weight_scale`` [N, 1]."""

    def __init__(self, weight, scale):
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(scale, requires_grad=False)


class Model:
    """Three int8 linears; the head shares q's weight, as a tied head does."""

    def __init__(self) -> None:
        g = torch.Generator().manual_seed(7)

        def w(n, k):
            return torch.randint(-127, 128, (n, k), dtype=torch.int8, generator=g)

        def s(n):
            return torch.rand(n, 1, generator=g) + 0.5
        q, k = Linear(w(24, 16), s(24)), Linear(w(40, 16), s(40))
        head = Linear(w(24, 16), s(24))
        head.weight = q.weight
        self.layers = {"q": q, "k": k, "head": head}
        self.plain = {"q": q.weight.detach().clone(), "k": k.weight.detach().clone()}
        self.x = torch.randint(-127, 128, (3, 16), dtype=torch.int8, generator=g)


def _ticks(step: int = 1_000):
    counter = itertools.count(0, step)
    return lambda: next(counter)


def _in_process_party(signer, log):
    """`challenger_prover.session` with the challenger's side in a thread of this process (`challenger_party.serve`, the
    test signer, a counter clock, its own coins each session) and the prover's over a socket pair
    (`challenger_prover.answer`)."""
    calls: list[tuple] = []

    def session(party, request, respond):
        a, b = socket.socketpair()
        coins = COINS[:-1] + bytes([len(calls)])

        def challenger():
            with b, b.makefile("rwb") as f:
                return CP.serve(f, request, signer=signer, log=log, coins=coins, domain=period.SESSION_DOMAIN,
                                clock=_ticks())
        with futures.ThreadPoolExecutor(1) as pool:
            served = pool.submit(challenger)
            try:
                with a, a.makefile("rwb") as f:
                    rounds = CR.answer(f, request["block_bytes"], respond)
            finally:
                sid, served_session = served.result()
        calls.append((party, request))
        return CP.report(sid, served_session, signer.public_key) | {"host": party.host, "rounds": rounds - CS.WARMUPS}
    return session, calls


def _period_dir(tmp_path, n: int, tools: dict[str, str] | None = None):
    d = tmp_path / "periods" / f"p{n}"
    d.mkdir(parents=True)
    (d / "period.json").write_text(json.dumps({"challenger_host": AUDITOR, "challenger": {"pous": {
        "command": ["ssh", AUDITOR, "uv", "run", "python", "-m", CS.MODULE, "--device", "pous-challenger",
                    "--listen", "0.0.0.0:7710"], "address": f"{AUDITOR}:7710"}}} | ({"tools": tools} if tools else {})))
    return d


#: stand-ins for the period's Lean tools, which `_judges` answers for
JUDGES = {"pous-run-protocol": "RUN", "pous-setup": "SETUP"}


def _judges(monkeypatch) -> list:
    """`period.serve` answering for `JUDGES` as the Lean does of an honest period: the run tool with the Python's
    answer, the setup tool with an accept. Returns the Python's runs, in order."""
    audit, runs = period_prover.audit, []

    def audited(*args, **kw):
        runs.append(audit(*args, **kw))
        return runs[-1]

    monkeypatch.setattr(period_prover, "audit", audited)
    monkeypatch.setattr(period, "serve", lambda exe, cases: [runs[-1].answer if exe == "RUN" else {"verdict": "accept"}])
    return runs


POUS = schemes.get("pous")
ENTRY = POUS.option
WRONG_SOURCE = "salt_source is hex, a callable of the commitment or {'command': argv}, not "


@pytest.fixture
def toy(monkeypatch):
    monkeypatch.setattr(ws, "geometry", lambda scheme: SMALL)
    return Model()


def _weights(model: Model) -> dict[str, torch.Tensor]:
    """Each linear's weight by its registered name, a tied weight under its first holder's name only."""
    seen: dict[int, str] = {}
    for n, m in model.layers.items():
        seen.setdefault(id(m.weight), f"{n}.weight")
    return {name: next(m.weight for m in model.layers.values() if id(m.weight) == i) for i, name in seen.items()}


def _held(model: Model, **launcher):
    """``pous`` attached by name, the build answered eager, and the model's weights registered."""
    ex = messages.Exchange()
    got = ex.attach(POUS, {}, **({"salt_source": SOURCE, "experimental": True} | launcher))
    ex.answer(H.Declared("card", {"eager": True}, tuple(ex.attached), {n: dict(v) for n, v in ex.attached.items()}))
    w = _weights(model)
    ex.emit(H.RegisteredValues(tuple(w.items()), operands={n: i for i, n in enumerate(w)}))
    return got, ex


def test_a_tiny_int8_models_weights_are_held_as_p2_and_served_bit_exact(toy):
    """W is the int8 weights in the store's layout (each from a block, zero past it); C* = P2-Enc(W, salt) under the salt
    sampled from the launcher's source after the commitment; a block computed through the exchange is handed the weight
    decoded from C*, and each released weight's restore is W decoded from C*."""
    opt, ex = _held(toy)
    s = opt.store
    q, lay = s.params, s.layout
    assert (q.w, q.c0, q.blocks_per_segment, q.segments) == (1024, 105, 8, 2) and q.payload_bytes == 126
    assert [(p.name, p.offset, p.size) for p in lay.tensors] == [("q.weight", 0, 384), ("k.weight", 4 * 126, 640)]
    assert s.rows.shape == (16, 128) and s.rows.dtype == torch.uint8

    c = opt.commitment()
    w = lay.pack({"q.weight": toy.plain["q"].numpy().tobytes(), "k.weight": toy.plain["k"].numpy().tobytes()})
    assert c["commitment"] == setup.commit(q, w).hex()
    assert bytes.fromhex(c["salt"]) == setup.sample_salt(bytes.fromhex(SOURCE), bytes.fromhex(c["commitment"]))
    held = s.held()                                                                     # the verifier's setup check of C*
    period.tie(held.point, held.published, period.setup_inputs(held.setup_case))
    assert held.published.root.hex() == c["storage_root"]
    assert [b for seg in period_prover.encode(q, s.salt, w) for b in seg] == [held.blocks[i] for i in range(q.blocks)]

    released = {r.value: r for _, r in ex.requests if isinstance(r, H.ReleaseValue)}
    assert sorted(released) == ["k.weight", "q.weight"]                                 # the tied head: encoded once
    for name, r in released.items():
        assert torch.equal(r.restore(), toy.plain[name.partition(".")[0]])
    due = H.BlockDue("k", 0, {"n": 40, "k": 16}, "s8", {"x": None, "weight": torch.empty(0)}, weights={"weight": "k.weight"})
    assert torch.equal(ex.compute(due, lambda given: given["weight"]), toy.plain["k"])

    d = opt.details()
    assert d["salt_source"] == "settings" and d["tensors"] == 2 and d["used_blocks"] == 10 and d["blocks"] == 16
    assert d["certifies"] == {"certified": False, "eps": "0", "s_bits": 0} and d["gaps"]
    opt.release()
    assert s.rows is None


def test_a_salt_source_of_another_type_is_refused_by_name_before_anything_is_held(toy):
    for source, kind in ((7, "int"), ({"command": []}, "dict"), ({"command": "true"}, "dict"),
                         ({"command": ["true"], "cwd": "/"}, "dict")):
        with pytest.raises(ValueError, match=re.escape(WRONG_SOURCE + kind)):
            messages.Exchange().attach(POUS, {}, salt_source=source, experimental=True)
    with pytest.raises(ValueError, match="no salt_source"):
        messages.Exchange().attach(POUS, {}, experimental=True)


def test_a_callable_salt_source_is_asked_once_with_the_commitment_and_what_it_returns_makes_the_salt(toy):
    asked = []

    def source(commitment):
        asked.append(commitment)
        return bytearray.fromhex(SOURCE)

    by_hex = _held(Model())[0].commitment()                             # the same weights; the hex is the same bytes
    opt, _ = _held(toy, salt_source=source)
    c = opt.commitment()
    assert asked == [bytes.fromhex(c["commitment"])]
    assert (c["commitment"], c["salt"]) == (by_hex["commitment"], by_hex["salt"])
    assert opt.details()["salt_source"] == "callable"


def test_a_salt_command_is_given_the_commitment_and_its_coin_makes_the_salt(toy):
    auditor = [sys.executable, "-c", ("import json, sys; r = json.loads(sys.stdin.readline()); "
                                      "print(json.dumps({'source': r['commitment'][:64]}))")]
    opt, _ = _held(toy, salt_source={"command": auditor})
    c = opt.commitment()
    assert opt.details()["salt_source"] == "command"
    assert bytes.fromhex(c["salt"]) == setup.sample_salt(bytes.fromhex(c["commitment"][:64]), bytes.fromhex(c["commitment"]))


def test_a_salt_command_that_fails_is_refused_with_its_exit_code_and_its_last_words(toy):
    script = "import sys; sys.stdin.read(); sys.stderr.write('the auditor is away\\n'); sys.exit(3)"
    said = f"{ENTRY}: the salt source's command exited 3: the auditor is away"
    with pytest.raises(RuntimeError, match=re.escape(said) + "$"):
        _held(toy, salt_source={"command": [sys.executable, "-c", script]})


def test_c_star_answers_only_its_own_blocks_and_none_once_it_is_freed(toy):
    opt, _ = _held(toy)
    s = opt.store
    blocks = s.held().blocks
    assert len(blocks) == s.params.blocks == 16 and blocks[15] == s.block(15) == opt.answer(15) == opt.prover(())(15)
    for i in (-1, 16):
        with pytest.raises(IndexError, match=rf"block {i} is outside C\*'s 16$"):
            s.block(i)
    opt.release()
    for read in (lambda: s.block(0), lambda: blocks[0]):
        with pytest.raises(RuntimeError, match=r"C\* is freed$"):
            read()
    with pytest.raises(RuntimeError, match=re.escape(f"{ENTRY}: nothing is held to audit") + "$"):
        served.Served(P2, {}).audit(NODE, COINS)


def test_the_cpu_decode_refuses_a_range_whose_last_block_holds_more_of_w(toy):
    s, w = _held(toy)[0].store, toy.plain["q"].numpy().tobytes()
    assert s.decode_cpu(0, 384) == w and s.decode_cpu(126, 126) == w[126:252]         # a tensor whole; one block whole
    assert any(w[100:126])
    with pytest.raises(ValueError, match="the block holding W byte 99 is nonzero past the tensor's end$"):
        s.decode_cpu(0, 100)


def test_a_strided_weight_and_weights_on_two_devices_are_refused_before_anything_is_committed(toy):
    toy.layers["k"] = Linear(torch.zeros(16, 40, dtype=torch.int8).T, torch.ones(40, 1))  # FP8's column-major layout
    with pytest.raises(ValueError, match=re.escape(f"{ENTRY}: k.weight's weight is strided")):
        _held(toy)
    toy.layers["k"] = Linear(torch.zeros(40, 16, dtype=torch.int8, device="meta"), torch.ones(40, 1))
    with pytest.raises(ValueError, match=re.escape(f"{ENTRY}: the weights are on ['cpu', 'meta'], and C* is held on one "
                                                   "device") + "$"):
        _held(toy)


def test_an_encoder_whose_blocks_do_not_hold_w_is_caught_at_a_segments_first_block(toy, monkeypatch):
    encoder = ws.p2_module("store").CpuEncoder
    encode = encoder.encode
    monkeypatch.setattr(encoder, "encode", lambda self, s, w_seg: encode(self, s, bytes(len(w_seg))))   # codewords of zeros
    with pytest.raises(RuntimeError, match=r"segment 0 block 0: the reference decode of C\* is not W's payload$"):
        _held(toy)


def test_its_period_accepts_with_the_challenger_on_its_host_and_states_the_bits_it_certifies(toy, tmp_path, monkeypatch):
    """`Served.subprotocol`: the memory attester at period.json's challenger host, the prover answering from C*, the
    verifier's tie of the setup case and its check; the Lean's tools are stand-ins that answer as the Python (`_judges`)."""
    session, calls = _in_process_party(ts.TestSigner(b"pous-vllm", directory=tmp_path), ts.session_log(tmp_path))
    monkeypatch.setattr(CR, "session", session)
    _judges(monkeypatch)
    opt, _ = _held(toy)
    for n in (0, 1):
        out = tmp_path / f"out{n}"
        got = opt.subprotocol(_period_dir(tmp_path, n, JUDGES), NODE, out, side_of=Side.of, schemes=ts.SCHEMES,
                              protocol=TEST_PROTOCOL)
        assert got["verdict"] == "accept" and got["protocol"] == TEST_PROTOCOL, got["reason"]
        assert got["public"] == {"certified": False, "eps": "0", "s_bits": 0}
        assert got["profile"]["placement"]["trusted-device"] == (
            f"{mc.UNCOMMITTED} (no configuration); the memory attester ran on {AUDITOR}, not the prover's {NODE}")
        assert (out / "pous-setup.case.json").exists() == (n == 0)
    party, request = calls[0]
    assert party.host == AUDITOR and opt.periods == 2


def test_a_period_whose_tools_name_no_lean_is_undecided_where_the_python_accepts(toy, tmp_path, monkeypatch):
    session, _ = _in_process_party(ts.TestSigner(b"pous-vllm", directory=tmp_path), ts.session_log(tmp_path))
    monkeypatch.setattr(CR, "session", session)
    runs = _judges(monkeypatch)
    got = _held(toy)[0].subprotocol(_period_dir(tmp_path, 0), NODE, tmp_path / "out", side_of=Side.of,
                                    schemes=ts.SCHEMES, protocol=TEST_PROTOCOL)
    assert runs[0].answer["verdict"] == "accept", runs[0].answer["reason"]
    assert got["verdict"] == "undecided" and got["reason"] == period.NO_JUDGE and got["public"] is None


def test_a_dropped_block_rejects(toy, tmp_path):
    """A prover that no longer holds a challenged block of C* fails its round, and a setup that sent C* short a block is
    refused by the verifier's tie: both answers are rejects with the reason, never an exception or a retry."""
    signer = ts.TestSigner(b"pous-vllm", directory=tmp_path)
    run = dict(signer=signer, trusted=(signer.public_key,), schemes=ts.SCHEMES, protocol=TEST_PROTOCOL,
               clock=iter(range(0, 10**12, 1_000)).__next__)
    opt, _ = _held(toy)
    q, dropped = opt.store.params, opt.store.layout.placement("k.weight").offset // opt.store.params.payload_bytes
    honest = opt.audit(NODE, COINS, log=ts.session_log(tmp_path), **run)
    assert honest.verdict == "accept", honest.answer["reason"]

    asked = []

    def lost(blocks):
        def answer(i):
            asked.append(i)
            return bytes(q.block_bytes) if i == dropped else opt.answer(i)
        return answer
    got = opt.audit(NODE, COINS, log=ts.session_log(tmp_path), prover=lost, **run)
    assert dropped in asked and got.verdict == "reject" and got.answer["reason"] and got.public("s_bits") == 0

    held = opt.store.held()
    short = [list(seg) for seg in period.setup_inputs(held.setup_case)[2]]
    del short[1][3]
    bad = dataclasses.replace(held, setup_case=period.setup_case(q, held.published.commitment, held.published.salt, short))
    got = period_prover.audit(NODE, bad, COINS, log=ts.session_log(tmp_path), prover=opt.prover, **run)
    assert got.verdict == "reject" and got.answer["reason"] == f"setup: segment 1 of C* is not 8 blocks of {q.block_bytes} bytes"


def test_attaching_pous_by_name_loads_no_selector_and_no_other_scheme():
    """Attaching PoUS by name imports neither the adapter's selector nor any module of another scheme's package:
    useful work's, the network attester's or PoUB's."""
    ra = "verity.protocols.resource_accounting."
    probe = ("import sys\n"
             "from verity.protocols.compose import schemes\n"
             "from verity_vllm import messages\n"
             "messages.Exchange().attach(schemes.get(schemes.named('option', 'pous:p2-16448/v4')), {},\n"
             "                           salt_source='a5' * 32, experimental=True)\n"
             f"other = ({ra + 'compute'!r}, {ra + 'network'!r}, {ra + 'memory.poub'!r})\n"
             "bad = sorted(m for m in sys.modules if m == 'verity_vllm.protocols' or '.pouw' in m\n"
             "             or any(m == o or m.startswith(o + '.') for o in other))\n"
             "assert not bad, bad\n")
    got = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert got.returncode == 0, got.stderr[-2000:]


def test_weights_registered_again_after_the_commitment_are_refused(toy):
    """C* is encoded once, from the weights the run registered first: a second registration (a weight altered after the
    commitment and registered anew) is refused, and the store still holds the first."""
    model = toy
    opt, ex = _held(model)
    root = opt.store.root
    model.layers["k"].weight.data[0, 0] += 1
    w = _weights(model)
    with pytest.raises(RuntimeError, match=re.escape(f"{ENTRY}: the weights were registered twice, and C* is encoded once")):
        ex.emit(H.RegisteredValues(tuple(w.items()), operands={n: i for i, n in enumerate(w)}))
    assert opt.store.root == root
    opt.release()


def test_a_run_that_registers_no_weight_operand_is_refused_and_holds_nothing(toy):
    ex = messages.Exchange()
    opt = ex.attach(POUS, {}, salt_source=SOURCE, experimental=True)
    ex.answer(H.Declared("card", {"eager": True}, tuple(ex.attached), {n: dict(v) for n, v in ex.attached.items()}))
    with pytest.raises(ValueError, match=re.escape(f"{ENTRY}: no registered value is a block's weight operand, so W is empty")):
        ex.emit(H.RegisteredValues((("bias", torch.zeros(4)),), operands={}))
    assert opt.store is None


def test_an_int8_weight_is_held_and_its_scales_stay_outside_the_store(toy):
    """W is the int8 weights' bytes as the kernel reads them; each linear's ``weight_scale`` is not a registered weight
    operand, so it stays where it is and is no part of the store."""
    scales = {n: m.weight_scale.detach().clone() for n, m in toy.layers.items()}
    opt, _ = _held(toy)
    assert [p.name for p in opt.store.layout.tensors] == ["q.weight", "k.weight"]
    assert all(torch.equal(m.weight_scale, scales[n]) for n, m in toy.layers.items())
    opt.release()


#: the period's Lean tools, stood in for by `_lean`
TOOLS = {"pous-run-protocol": "RUN", "pous-setup": "SETUP"}


def _lean(monkeypatch, tmp_path):
    """Stand-ins for a period's parties and tools: the challenger's session in this process, and `period.serve`
    answering for the tool named ``RUN`` as ``say["run"]`` makes of the Python's answer, and for any other as
    ``say["setup"]``. Returns ``say``, the Python's runs and what was served, each in order."""
    session, _ = _in_process_party(ts.TestSigner(b"pous-vllm", directory=tmp_path), ts.session_log(tmp_path))
    monkeypatch.setattr(CR, "session", session)
    audit, runs, served_ = period_prover.audit, [], []
    say: dict[str, Any] = {"run": lambda answer: answer | {"reason": "the Lean's words"}, "setup": {"verdict": "accept"}}

    def audited(*args, **kw):
        runs.append(audit(*args, **kw))
        return runs[-1]

    def serve(exe, cases):
        served_.append((exe, list(cases)))
        return [say["run"](runs[-1].answer) if exe == "RUN" else say["setup"]]

    monkeypatch.setattr(period_prover, "audit", audited)
    monkeypatch.setattr(period, "serve", serve)
    return say, runs, served_


def _subprotocol(opt, tmp_path, n):
    """Monitoring period ``n``'s subprotocol of ``opt``, its ``period.json`` naming `TOOLS`; the answer and what
    ``pous.json`` holds."""
    directory, out = _period_dir(tmp_path, n, TOOLS), tmp_path / f"out{n}"
    got = opt.subprotocol(directory, NODE, out, side_of=Side.of, schemes=ts.SCHEMES, protocol=TEST_PROTOCOL)
    return got, json.loads((out / "pous.json").read_text())


def test_where_a_period_names_the_lean_tools_the_subprotocols_answer_is_theirs_and_must_be_the_pythons(toy, tmp_path, monkeypatch):
    """The run tool is asked every period with the run's case and the setup tool at the first alone, with the setup's;
    the subprotocol answers with the run tool's answer, and refuses one that differs from the Python's in more than its
    reason, writing nothing."""
    say, runs, served_ = _lean(monkeypatch, tmp_path)
    opt, _ = _held(toy)
    got, doc = _subprotocol(opt, tmp_path, 0)
    assert served_ == [("RUN", [runs[0].case]), ("SETUP", [runs[0].setup_case])]
    assert runs[0].answer["verdict"] == "accept" and runs[0].answer["reason"] != "the Lean's words"
    assert got == runs[0].answer | {"reason": "the Lean's words", "profile": got["profile"]} == doc["answer"]
    assert doc["lean"] == {"run": runs[0].answer | {"reason": "the Lean's words"}, "setup": {"verdict": "accept"}}

    got, doc = _subprotocol(opt, tmp_path, 1)
    assert served_[2:] == [("RUN", [runs[1].case])] and set(doc["lean"]) == {"run"} and got["reason"] == "the Lean's words"

    say["run"] = lambda answer: answer | {"verdict": "reject"}
    with pytest.raises(RuntimeError, match="^pous-run-protocol answered .*'verdict': 'reject'.*, the Python .*'verdict': 'accept'"):
        _subprotocol(opt, tmp_path, 2)
    assert not (tmp_path / "out2").exists()


def test_a_setup_the_lean_refuses_fails_the_first_period_whatever_the_python_said(toy, tmp_path, monkeypatch):
    say, runs, served_ = _lean(monkeypatch, tmp_path)
    say["setup"] = {"verdict": "reject", "reason": "C* is not the encoding"}
    opt, _ = _held(toy)
    with pytest.raises(RuntimeError, match="^pous-setup refused the setup the Python accepted: .*C\\* is not the encoding"):
        _subprotocol(opt, tmp_path, 0)
    assert runs[0].answer["verdict"] == "accept" and [exe for exe, _ in served_] == ["RUN", "SETUP"]
    assert not (tmp_path / "out0").exists()


def test_a_released_store_refuses_a_subprotocol_at_its_audit_and_writes_no_output(toy, tmp_path, monkeypatch):
    _, runs, served_ = _lean(monkeypatch, tmp_path)
    opt, _ = _held(toy)
    opt.release()
    with pytest.raises(RuntimeError, match=re.escape(f"{ENTRY}: nothing is held to audit") + "$"):
        _subprotocol(opt, tmp_path, 0)
    assert not runs and not served_ and opt.periods == 0 and not (tmp_path / "out0").exists()
