"""`protocols.pous`: POUS as a protocol option over a tiny int8 model, on the CPU.

The model's layers are a stand-in linear type (`pous.linear_types`) holding an int8 weight [N, K] and its f32
``weight_scale`` [N, 1] as the ``int8_exact`` target's linears do, and P2 runs at the small point's 1,024-bit blocks, 8 a
segment (`pous._base`), so encoding takes milliseconds.  The layout, the commitment, the salt, the codec, the root, the
period's audit and the memory challenger are `verity.core.protocols.pous`'s and `benchmarks/pous/p2_v1/store.py`'s own; the
memory challenger's key is the PoUS tests' test signer, its clock a counter.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import itertools
import json
import socket
import sys
import types
from concurrent import futures

import pytest
import torch

from verity.core.protocols.pous import challenger_party as CP
from verity.core.protocols.pous import memory_challenger as mc
from verity.core.protocols.pous import period, setup
from verity.core.protocols.pous.schemes.p2 import P2Params
from verity_vllm import config
from verity_vllm import protocols as PO
from verity_vllm.engine import hooks
from verity_vllm.protocols import pous

P2 = "p2-16448/v4"
ENTRY = f"pous:{P2}"
SOURCE = "a5" * 32
SETTINGS = {"pous": {"salt_source": SOURCE, "experimental": True}}
NODE, AUDITOR = "vy-dc-h200-3", "vy-auditor-2"
COINS = bytes(range(64))
SMALL = P2Params(period.W, period.C0, 8)


def _signer_module():
    path = config.ROOT.parent.parent / "verity" / "core" / "protocols" / "pous" / "tests" / "pous_test_signer.py"
    spec = importlib.util.spec_from_file_location("pous_test_signer", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ts = _signer_module()
TEST_PROTOCOL = mc.TEST_PREFIX + period.PROTOCOL


class Int8Apply:
    """The plain arm's product, exactly: int32 accumulation of the s8 weight, then its per-row scale."""

    def apply(self, layer, x, bias=None):
        acc = x.to(torch.int32) @ layer.weight.to(torch.int32).T
        return acc.to(torch.float32) * layer.weight_scale.T


class Linear:
    def __init__(self, weight, scale):
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
        self.quant_method = Int8Apply()


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

    def named_modules(self):
        yield "", self
        yield from self.layers.items()

    def named_parameters(self):
        for n, m in self.layers.items():
            yield f"{n}.weight", m.weight
            yield f"{n}.weight_scale", m.weight_scale

    def named_buffers(self):
        return iter(())

    def outputs(self):
        return {n: m.quant_method.apply(m, self.x) for n, m in self.layers.items()}


@pytest.fixture
def toy(monkeypatch):
    monkeypatch.setattr(pous, "linear_types", lambda: (Linear,))
    monkeypatch.setattr(pous, "_base", lambda scheme: SMALL)
    return Model()


def _ctx(*enabled, settings=None, **kw):
    return PO.Context(stage="serve", enabled=config.parse_protocols(enabled), settings=SETTINGS if settings is None else settings,
                      **kw)


def _ticks(step: int = 1_000):
    counter = itertools.count(0, step)
    return lambda: next(counter)


def test_pous_registers_p2_v4_as_its_default_and_refuses_what_it_cannot_run():
    assert pous.NAME in config.PROTOCOLS and pous.DEFAULT_SCHEME == P2 and P2 in pous.schemes()
    assert PO.adapter("pous") is pous
    assert pous.refusals(P2, _ctx(ENTRY)) == ()
    got = pous.refusals(P2, _ctx(ENTRY, enforce_eager=False, settings={"pous": {"salt_source": SOURCE, "experimental": True,
                                                                                 "port": 1}}))
    assert any("unknown pous settings ['port']" in r for r in got) and any("compiled execution" in r for r in got)
    assert any("experimental" in r for r in pous.refusals(P2, _ctx(ENTRY, settings={"pous": {"salt_source": SOURCE}})))
    assert any("no salt_source" in r for r in pous.refusals(P2, _ctx(ENTRY, settings={"pous": {"experimental": True}})))
    assert any("not hex" in r for r in pous.refusals(P2, _ctx(ENTRY, settings={"pous": {"salt_source": "zz",
                                                                                        "experimental": True}})))
    band = pous.refusals("band-chain/d12/v1", _ctx("pous:band-chain/d12/v1"))
    assert band == (f"pous holds C* under {P2} alone, the codec its period's run protocol checks and the staging decode "
                    "reads, not band-chain/d12/v1",)
    assert not hasattr(pous, "keeps_weight_copy")                                       # POUS keeps only C*


def test_with_pous_on_the_selector_refuses_an_option_that_keeps_a_weight_copy(monkeypatch):
    _fake_pouw(monkeypatch, lambda scheme, target: None, keeps=True)
    got = PO.refusals(_ctx("pouw:ncp-v1", ENTRY))
    assert any(r.startswith("pouw:ncp-v1: keeps a weight-derived copy") for r in got)
    assert PO.refusals(_ctx("pouw:ncp-v1", settings={})) == ()


def test_a_tiny_int8_model_is_held_as_p2_and_served_bit_exact(toy):
    """W is the int8 weights in the store's layout (each from a block, zero past it); C* = P2-Enc(W, salt) under the salt
    sampled from the auditor's source after the commitment; every use decodes the layer's weight from C*, so the outputs are
    the plaintext's to the bit, and closing restores W decoded from C*."""
    before, served = hooks.live(), toy.outputs()
    comp = PO.install(toy, _ctx(ENTRY))
    opt, s = comp.get("pous"), comp.get("pous").store
    assert all(m.weight.numel() == 0 for m in toy.layers.values())                    # only C* is held
    for name, out in toy.outputs().items():
        assert torch.equal(out, served[name])                                          # decoded from C* on every use

    q, lay = s.params, s.layout                                                         # the store's frozen layout
    assert (q.w, q.c0, q.blocks_per_segment, q.segments) == (1024, 105, 8, 2) and q.payload_bytes == 126
    assert [(p.name, p.offset, p.size) for p in lay.tensors] == [("q", 0, 384), ("k", 4 * 126, 640)]
    assert s.rows.shape == (16, 128) and s.rows.dtype == torch.uint8

    c = opt.commitment()
    w = lay.pack({"q": toy.plain["q"].numpy().tobytes(), "k": toy.plain["k"].numpy().tobytes()})
    assert c["commitment"] == setup.commit(q, w).hex()
    assert bytes.fromhex(c["salt"]) == setup.draw_salt(bytes.fromhex(SOURCE), bytes.fromhex(c["commitment"]))
    held = s.held()                                                                     # the verifier's setup check of C*
    period.tie(held.point, held.published, period.setup_inputs(held.setup_case))
    assert held.published.root.hex() == c["storage_root"]
    assert [b for seg in period.encode(q, s.salt, w) for b in seg] == [held.blocks[i] for i in range(q.blocks)]

    plain = opt.weights()                                                               # tied weights: encoded once, one tensor
    assert {n: t.numpy().tobytes() for n, t in plain.items()} == {
        "q.weight": w[:384], "k.weight": w[504:1144], "head.weight": w[:384]}
    assert plain["q.weight"] is plain["head.weight"]
    view = PO.weights_view(comp, toy)                                                   # what the Commit's replay reads
    assert {n: t.numpy().tobytes() for n, t in view.named_parameters() if n.endswith(".weight")} == \
        {n: t.numpy().tobytes() for n, t in plain.items()}
    assert [type(e).__name__ for e in hooks.live(ENTRY)] == ["Service", "Patch", "Patch", "Patch", "Service"]
    wc = next(p for p in comp.record()["protocols"] if p["entry"] == ENTRY)["weights"]  # phase 1, before any salt
    assert wc == {"scheme": P2, "covers": "linears", "tensors": 2, "bytes": 1024, "commitment": c["commitment"],
                  "layout": c["layout"]}
    info = comp.info()[ENTRY]
    assert info["salt_source"] == "settings" and info["used_blocks"] == 10 and info["blocks"] == 16
    assert info["certifies"] == {"certified": False, "eps": "0", "s_bits": 0} and info["gaps"]

    comp.close()
    for name, m in toy.layers.items():
        assert torch.equal(m.weight, toy.plain["q" if name == "head" else name])
    assert toy.layers["q"].weight is toy.layers["head"].weight and hooks.live() == before
    assert all("apply" not in vars(m.quant_method) for m in toy.layers.values())
    assert opt.weights() is None and PO.weights_view(comp, toy) is toy and s.rows is None
    with pytest.raises(RuntimeError, match="closed"):
        opt.answer(0)


def test_an_int8_linear_with_its_scales_is_served_not_refused(toy):
    """The adapter's former refusal of a quantized linear is gone: an int8 weight beside its weight_scale (and the other
    quantization parameters vLLM's methods hold) is W's bytes as the kernel reads them; the scales stay outside the store,
    on the layer."""
    for m in toy.layers.values():
        m.input_scale = torch.ones(1)
    scales = {n: m.weight_scale for n, m in toy.layers.items()}
    with PO.install(toy, _ctx(ENTRY)) as comp:
        assert comp.entries == (ENTRY,) and toy.layers["k"].weight.numel() == 0
        assert all(m.weight_scale is scales[n] for n, m in toy.layers.items())
        assert not hasattr(pous, "quantized") and not hasattr(pous, "QUANT_PARAMS")
    strided = Linear(torch.zeros(16, 24, dtype=torch.int8).T, torch.ones(24, 1))       # FP8's column-major layout
    toy.layers["k"] = strided
    with pytest.raises(ValueError, match="k's weight is strided"):
        PO.install(toy, _ctx(ENTRY))


def _in_process_party(signer, log):
    """`challenger_party.session` with the challenger's side in a thread of this process (`challenger_party.serve`, the
    test signer, a counter clock, its own coins each session) and the prover's over a socket pair
    (`challenger_party.answer`)."""
    calls = []

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
                    rounds = CP.answer(f, request["block_bytes"], respond)
            finally:
                sid, served_session = served.result()
        calls.append((party, request))
        return CP.report(sid, served_session, signer.public_key) | {"host": party.host, "rounds": rounds - CP.WARMUPS}
    return session, calls


def _period_dir(tmp_path, n: int):
    d = tmp_path / "periods" / f"p{n}"
    d.mkdir(parents=True)
    (d / "period.json").write_text(json.dumps({"challenger_host": AUDITOR, "challenger": {"pous": {
        "command": ["ssh", AUDITOR, "uv", "run", "python", "-m", CP.MODULE, "--device", "pous-challenger",
                    "--listen", "0.0.0.0:7710"], "address": f"{AUDITOR}:7710"}}}))
    return d


def test_its_period_accepts_with_the_challenger_on_its_host_and_states_the_bits_it_certifies(toy, tmp_path, monkeypatch):
    """`PousOption.leg`: the memory challenger at period.json's challenger host, the prover answering from C*, the verifier's tie of
    the setup case and its check; an accepted period says plainly that this point certifies 0 bits."""
    session, calls = _in_process_party(ts.TestSigner(b"pous-vllm", directory=tmp_path), ts.session_log(tmp_path))
    monkeypatch.setattr(CP, "session", session)
    with PO.install(toy, _ctx(ENTRY)) as comp:
        opt = comp.get("pous")
        for n in (0, 1):
            out = tmp_path / f"out{n}"
            got = opt.leg(_period_dir(tmp_path, n), NODE, out, schemes=ts.SCHEMES, protocol=TEST_PROTOCOL)
            assert got["verdict"] == "accept" and got["protocol"] == TEST_PROTOCOL, got["reason"]
            assert got["public"] == {"certified": False, "eps": "0", "s_bits": 0}
            assert got["profile"]["placement"]["trusted-host"].startswith(f"holds: the memory challenger ran on {AUDITOR}")
            doc = json.loads((out / "pous.json").read_text())
            assert doc["s_bits"] == 0 and doc["rounds_answered"] == period.K + CP.WARMUPS and doc["setup"]["fresh"] == (n == 0)
            assert (out / "pous.case.json").exists() and (out / "pous-setup.case.json").exists() == (n == 0)
        party, request = calls[0]
        assert party.host == AUDITOR and party.address == (AUDITOR, 7710)
        assert request == CP.request_of(NODE, opt.store.point, opt.store.root) and opt.periods == 2


def test_a_dropped_block_rejects(toy, tmp_path):
    """A prover that no longer holds a challenged block of C* fails its round, and a setup that sent C* short a block is
    refused by the verifier's tie: both answers are rejects with the reason, never an exception or a retry."""
    signer = ts.TestSigner(b"pous-vllm", directory=tmp_path)
    run = dict(signer=signer, trusted=(signer.public_key,), schemes=ts.SCHEMES, protocol=TEST_PROTOCOL, clock=_ticks())
    with PO.install(toy, _ctx(ENTRY)) as comp:
        opt = comp.get("pous")
        q, dropped = opt.store.params, opt.store.layout.placement("k").offset // opt.store.params.payload_bytes
        honest = opt.audit(NODE, COINS, log=ts.session_log(tmp_path), **run)
        assert honest.verdict == "accept", honest.answer["reason"]

        asked = []
        def lost(blocks):
            return lambda i: asked.append(i) or (bytes(q.block_bytes) if i == dropped else opt.answer(i))
        got = opt.audit(NODE, COINS, log=ts.session_log(tmp_path), prover=lost, **run)
        assert dropped in asked and got.verdict == "reject" and got.answer["reason"] and got.s_bits == 0

        held = opt.store.held()
        short = [list(seg) for seg in period.setup_inputs(held.setup_case)[2]]
        del short[1][3]
        bad = dataclasses.replace(held, setup_case=period.setup_case(q, held.published.commitment, held.published.salt, short))
        got = period.audit(NODE, bad, COINS, log=ts.session_log(tmp_path), prover=opt.prover, **run)
        assert got.verdict == "reject" and got.answer["reason"] == f"setup: segment 1 of C* is not 8 blocks of {q.block_bytes} bytes"


def _fake_pouw(monkeypatch, install, keeps=False):
    mod = types.ModuleType("verity_vllm.protocols.fake_pouw")
    mod.NAME, mod.schemes, mod.refusals, mod.install = "pouw", lambda: ("ncp-v1",), lambda s, c: (), install
    mod.keeps_weight_copy = lambda scheme, ctx: keeps
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.setitem(PO.ADAPTERS, "pouw", "fake_pouw")


def test_pous_wraps_around_pouw_so_pouws_gemm_reads_the_decoded_weight(toy, monkeypatch):
    """PoUW's GEMM is handed the weight POUS decoded; the salt's source here is an auditor's process that reads the
    commitment and prints its coin."""
    def pouw_install(scheme, target):
        h = target.hooks(f"pouw:{scheme}")
        for m in toy.layers.values():
            h.wrap(m.quant_method, "apply", lambda orig: (lambda layer, x, bias=None: ("pouw", layer.weight.clone(), x)))
        return PO.Option(name=f"pouw:{scheme}", hooks=h)

    _fake_pouw(monkeypatch, pouw_install)
    auditor = [sys.executable, "-c", "import json, sys; r = json.loads(sys.stdin.readline()); "
               "print(json.dumps({'source': r['commitment'][:64]}))"]
    before = hooks.live()
    with PO.install(toy, _ctx("pouw:ncp-v1", ENTRY, settings={"pous": {"salt_source": {"command": auditor},
                                                                         "experimental": True}})) as comp:
        k = toy.layers["k"]
        assert comp.entries == ("pouw:ncp-v1", ENTRY) and k.weight.numel() == 0
        tag, w, x = k.quant_method.apply(k, "x")
        assert (tag, x) == ("pouw", "x") and torch.equal(w, toy.plain["k"])
        c = comp.get("pous").commitment()
        assert comp.info()[ENTRY]["salt_source"] == "command"
        assert bytes.fromhex(c["salt"]) == setup.draw_salt(bytes.fromhex(c["commitment"][:64]), bytes.fromhex(c["commitment"]))
    assert hooks.live() == before and torch.equal(toy.layers["k"].weight, toy.plain["k"])


def test_weights_that_change_after_the_commitment_are_refused_and_leave_nothing(toy, monkeypatch):
    def pouw_install(scheme, target):                                   # an inner option that alters a weight in place
        toy.layers["k"].weight.data[0, 0] += 1
        return PO.Option(name=f"pouw:{scheme}", hooks=target.hooks(f"pouw:{scheme}"))

    _fake_pouw(monkeypatch, pouw_install)
    before = hooks.live()
    with pytest.raises(ValueError, match="changed after the composition committed"):
        PO.install(toy, _ctx("pouw:ncp-v1", ENTRY))
    assert hooks.live() == before and torch.equal(toy.layers["q"].weight, toy.plain["q"])


def test_a_model_without_linears_is_refused_at_install_and_leaves_nothing(toy, monkeypatch):
    monkeypatch.setattr(pous, "linear_types", lambda: (Int8Apply,))
    before = hooks.live()
    with pytest.raises(ValueError, match="no Int8Apply layer"):
        PO.install(types.SimpleNamespace(named_modules=lambda: iter([("", object())])), _ctx(ENTRY))
    assert hooks.live() == before
