"""The logits tap (`acquire.sources.logits_tap_source`) on CPU tensors with a recording committer, over the call-boundary source's
Gemma-2-style dense Program.  With `--logits-tap 1` it claims the soft-capped logits' call boundaries and, attached just before the
call-boundary source, commits the served GEMM output, quotient and tanh: an honest forward commits the step stream the call-boundary source
alone commits, name for name, word for word and in the same order; a dishonest one commits what it served; a forward without the chain
is a collector error that commits nothing of the step; a manifest it cannot commit is refused, and the gate names it first; and a
call-boundary plan derived without the claim refuses the Commit."""
from __future__ import annotations

import os
import types

import numpy as np
import pytest

from tests.acquire.call_boundary_programs import H, IL, LG, PF, T, V
from tests.acquire.test_call_boundary_source import Recorder, _tensor, dense, pair, runtime  # noqa: F401  -- fixtures
from verity_vllm.acquire import gate as G

torch = pytest.importorskip("torch")

from verity_vllm.acquire.sources import call_boundary_source as CB  # noqa: E402
from verity_vllm.acquire.sources import logits_tap_source as LT  # noqa: E402
from verity_vllm.acquire.sources import taps  # noqa: E402

CAP = 30.0


class Softcap(torch.nn.Module):
    """Gemma-2's LogitsProcessor: the lm_head GEMM (`gemm(hidden)`, the served words), `/ cap`, tanh, `* cap`."""

    def __init__(self, gemm, tanh=True):
        super().__init__()
        self.gemm, self.tanh = gemm, tanh

    def forward(self, lm_head, hidden):
        logits = self.gemm(hidden)
        logits = logits / CAP
        if self.tanh:
            logits = torch.tanh(logits)
        return logits * CAP


def served_gemm(world, step):
    """The reference's lm_head GEMM output at `step`: what an honest served GEMM computes."""
    i = next(r["i"] for r in world.doc["rows"] if world.b.step[r["i"]] == step and r["name"] == f"{LG}/triton_launch_{step}[0]")
    return _tensor(np.array([world.vals[i][:V]], dtype=np.uint16), torch.bfloat16)


def _commit(world, monkeypatch, tap, *, gemm=None, softcap=Softcap, man=None, programs=None, plan=None):
    monkeypatch.setattr(CB, "CLAIMS", [])
    com = Recorder()
    model = runtime(["model.embed_tokens", PF[len("model."):], IL[len("model."):]],
                    {n[len("model."):]: _tensor(w, torch.bfloat16) for n, w in world.fields.items()})
    model.add_module("logits_processor", softcap(lambda h: (gemm or (lambda s: served_gemm(world, s)))(com._cur.step)))
    com.model = model
    classes = taps.attach(com, man or world.man, ("instance_outputs",), taps.TapSources(logits_tap=int(tap)), programs=programs or (world.pdir,),
                          call_boundary_plan=plan)
    for s in com.sources:
        s.install()
    return com, classes


def _drive(world, com, steps=range(1 + T), logits=lambda s: True):
    mods = dict(com.model.named_modules())
    rows = lambda calls: _tensor(np.array([world.vals[c][:H] for c in calls], dtype=np.uint16), torch.bfloat16)  # noqa: E731
    for s in steps:
        com._cur = types.SimpleNamespace(step=s)
        for src in com.sources:
            src.on_step_begin()
        mods[PF[len("model."):]](rows(world.at[(s, "e")]))
        mods[IL[len("model."):]](rows(world.at[(s, "x")]), residual=rows(world.at[(s, "e")]))
        for _ in range(int(logits(s))):
            mods["logits_processor"](mods["model.embed_tokens"], rows(world.at[(s, "y")][-1:]))
        for src in com.sources:
            src.on_step_end(com._cur)
    com._cur = None


def _stream(com):
    return [(s, n, tuple(t.shape), t.dtype, CB.words(torch, t).tobytes()) for s, n, t in com.acquired]


def test_an_honest_forward_commits_the_call_boundary_sources_step_stream(dense, monkeypatch):
    """The tap's three identities per step are the logits body's targets the call-boundary source would evaluate; committed from the
    served tensors, the step stream is the call-boundary source's alone -- the root the Commit computes over it is the same."""
    host, host_classes = _commit(dense, monkeypatch, False)
    assert [s.short_name for s in host.sources] == ["call_boundary"]
    _drive(dense, host)
    tapped, classes = _commit(dense, monkeypatch, True)
    tap, cb = tapped.sources
    assert (tap.short_name, cb.short_name) == ("logits_tap", "call_boundary") and classes == host_classes == ("instance_outputs", CB.FAMILY)
    ids = LT.manifest_sites(dense.man)
    assert len(ids) == 3 * (1 + T) and {LT.kind(r) for r in ids} == {"triton_launch", "div", "tanh"}
    assert cb.plan.identities == len(CB.manifest_sites(dense.man)) == host.sources[0].plan.identities - len(ids)
    assert not any(m == "logits_processor" for _e, m in cb.plan.bodies)
    _drive(dense, tapped)
    assert not host.errors and not tapped.errors, tapped.errors
    assert _stream(tapped) == _stream(host)
    for s in range(1 + T):
        assert {**tap.targets_at(s), **cb.targets_at(s)} == host.sources[0].targets_at(s)
        assert [n for e, n, _t in tapped.acquired if e == s][:3] == [LT.name(r) for r in ids if r["engine_step"] == s]
    assert tap.stats["tensors"] == len(ids) and tap.stats["steps"] == 1 + T and tap.stats["errors"] == 0
    assert tap.extra_step_bytes({}, None) == {"per_token_bytes": 0, "per_seq_bytes": 3 * V * 2, "tensors_per_step": 3}
    d = tap.declaration_classes()[0]
    assert (d["class"], d["how"], d["tensors"]) == (CB.FAMILY, "bound-as-leaf", len(ids))


def test_the_tap_commits_the_served_words_not_the_programs(dense, monkeypatch):
    """A served GEMM output one word off: the tap commits it and what the forward computed from it (the replay's openings of the producing
    Calls are what check them); the call-boundary source's words of the other bodies are unchanged."""
    host, _ = _commit(dense, monkeypatch, False)
    _drive(dense, host)

    def off(s):
        g = served_gemm(dense, s)
        if s == 1:
            g.view(torch.int16)[0, 3] ^= 1
        return g

    tapped, _ = _commit(dense, monkeypatch, True, gemm=off)
    _drive(dense, tapped)
    assert not tapped.errors
    got, ref = {(s, n): t for s, n, t in tapped.acquired}, {(s, n): t for s, n, t in host.acquired}
    assert list(got) == list(ref)
    bad = {k for k in got if not torch.equal(got[k].view(torch.int16), ref[k].view(torch.int16))}
    assert (1, "logits_processor/triton_launch_1/out") in bad and all(s == 1 and n.startswith("logits_processor/") for s, n in bad)
    assert torch.equal(got[(1, "logits_processor/triton_launch_1/out")], off(1))


def test_a_forward_without_the_chain_is_a_collector_error_that_commits_nothing_of_the_step(dense, monkeypatch):
    com, _ = _commit(dense, monkeypatch, True, softcap=lambda g: Softcap(g, tanh=False))
    _drive(dense, com, steps=[0])
    assert com.errors == ["logits tap: logits_processor at engine step 0 ran 1 divisions and 0 tanh, not one tanh of a quotient"]
    assert not any(n.startswith("logits_processor/") for _s, n, _t in com.acquired)
    com, _ = _commit(dense, monkeypatch, True)
    _drive(dense, com, steps=[0, 1, 2], logits=lambda s: {0: 0, 1: 2, 2: 1}[s])
    assert com.errors[0] == "logits tap: logits_processor did not run at engine step 0; the manifest names its call boundaries there"
    assert com.errors[1] == "logits tap: logits_processor ran twice at engine step 1; the Program has one body" and len(com.errors) == 2
    assert {s for s, n, _t in com.acquired if n.startswith("logits_processor/")} == {1, 2}
    com, _ = _commit(dense, monkeypatch, True, gemm=lambda s: served_gemm(dense, s).float())
    _drive(dense, com, steps=[0])
    assert len(com.errors) == 1 and "is (1, 16) torch.float32 served; the identity is 1 x 16 bf16" in com.errors[0]
    assert not any(n.startswith("logits_processor/") for _s, n, _t in com.acquired)


def test_the_gate_lists_the_tap_and_a_manifest_it_cannot_commit_is_refused_by_name(dense, pair, monkeypatch):
    """One request: the gate passes with the tap among the sources.  Two requests share each logits name at a step (B = 2) and several
    ranks are not the tap's: `attached` leaves it out, so the gate names the claimed identities, and a Commit that attaches anyway exits 3."""
    monkeypatch.setattr(CB, "CLAIMS", [LT.claim])
    rec = G.call_boundaries(dense.man, os.path.dirname(dense.pdir))
    assert rec["pass"] is True and rec["sources"] == ["logits_tap", "call_boundary"] and rec["claimed"] == 3 * (1 + T)
    assert "logits_tap" not in taps.attached(pair.man) and "logits_tap" not in taps.attached({**dense.man, "ranks": 2})
    assert "engine step 0 claims {'div': 2, 'tanh': 2, 'triton_launch': 2}" in LT.refusal(pair.man, LT.manifest_sites(pair.man))
    rec = G.call_boundaries(pair.man, pair.root)
    assert rec["pass"] is False and rec["unattached"] == len(LT.manifest_sites(pair.man)) and "claimed by logits_tap" in rec["named"]["unattached"][0]
    with pytest.raises(SystemExit) as e:
        _commit(pair, monkeypatch, True, man=pair.man, programs=(pair.root,))
    assert e.value.code == 3
    LT.claiming(False)
    assert not CB.CLAIMS and "logits_tap" not in taps.attached(dense.man) and not LT.manifest_sites(dense.man) and LT.attach(Recorder(), dense.man, ("a",)) == (("a",), None)


def test_a_call_boundary_plan_derived_without_the_claim_refuses_the_commit(dense, monkeypatch):
    """The plan's key leaves the claim out: a plan of the logits body beside the tap would commit its names twice."""
    monkeypatch.setattr(CB, "CLAIMS", [])
    plan = CB.plan_of(dense.man, dense.pdir)
    with pytest.raises(SystemExit) as e:
        _commit(dense, monkeypatch, True, plan=plan)
    assert e.value.code == 3
