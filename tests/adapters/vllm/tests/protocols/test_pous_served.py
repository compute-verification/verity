"""PoUS attached by its scheme's name on the build's exchange (`compose.schemes.get("pous")`, `messages.Exchange`), over a
tiny int8 model's weights, on the CPU: what it holds, what it refuses, its period and its rejects.

The weights are `test_pous_option`'s model's, registered as a served run registers them (`RegisteredValues`, a tied weight
once), and P2 runs at the small point's 1,024-bit blocks, 8 a segment (`weight_store.geometry`), so encoding takes
milliseconds. No forward runs: a linear block is computed through the exchange with the weight decoded from ``C*``.
"""
from __future__ import annotations

import dataclasses
import re
import subprocess
import sys

import pytest
import torch
from verity import hooks as H
from verity.protocols.compose import schemes
from verity.protocols.resource_accounting.memory.pous.attester import memory_attester as mc
from verity.protocols.resource_accounting.memory.pous.prover import challenger_prover as CR
from verity.protocols.resource_accounting.memory.pous.prover import period_prover
from verity.protocols.resource_accounting.memory.pous.prover import served
from verity.protocols.resource_accounting.memory.pous.prover import weight_store as ws
from verity.protocols.resource_accounting.memory.pous.verifier import period, setup
from verity.runtime.stream import Side

from tests.protocols.test_pous_option import (
    AUDITOR,
    COINS,
    JUDGES,
    NODE,
    P2,
    SMALL,
    SOURCE,
    TEST_PROTOCOL,
    Linear,
    Model,
    _in_process_party,
    _judges,
    _period_dir,
    ts,
)
from verity_vllm import messages

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


def test_attaching_pous_by_name_loads_no_selector_and_nothing_of_useful_works():
    probe = ("import sys\n"
             "from verity.protocols.compose import schemes\n"
             "from verity_vllm import messages\n"
             "messages.Exchange().attach(schemes.get('pous'), {}, salt_source='a5' * 32, experimental=True)\n"
             "bad = sorted(m for m in sys.modules if m == 'verity_vllm.protocols' or '.pouw' in m)\n"
             "assert not bad, bad\n")
    got = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert got.returncode == 0, got.stderr[-2000:]
