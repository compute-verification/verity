"""The norm body tap (`acquire.sources.norm_body_tap_source`) on CPU tensors with a recording committer, over the call-boundary source's
Gemma-2-style dense Program, with vLLM's GemmaRMSNorm forward at both norms (`weight.float() + 1.0`, then the native `rms_norm` /
`fused_add_rms_norm` of `vllm.ir`, as an eager row runs them).  With `--norm-body-tap 1` it claims every norm call boundary and holds what
the forward computed, which the call-boundary source commits in their place: the step stream has the host's names, shapes, dtypes and
order, each identity's source recorded (the binding map's `source`); every word is the forward's own, and those CPU torch rounds as the
Program does (`weight + 1`, the residual sum, the square) are the host's; a module with both sources commits them in the manifest's order;
a forward of another form, or a value written in place after it, is a collector error that commits nothing of that module at that step;
the warm-up commits zeros and captures nothing; a manifest or model it cannot commit is refused, and the gate names it first."""
from __future__ import annotations

import json
import os
import types

import numpy as np
import pytest

from tests.acquire.call_boundary_programs import EPS, H, IL, PF, T
from tests.acquire.test_call_boundary_source import Probe, Recorder, _tensor, dense, pair, runtime  # noqa: F401  -- fixtures
from verity_vllm.acquire import gate as G
from verity_vllm.commit import binding as B

torch = pytest.importorskip("torch")

from verity_vllm.acquire.sources import call_boundary_source as CB  # noqa: E402
from verity_vllm.acquire.sources import norm_body_tap_source as NT  # noqa: E402
from verity_vllm.acquire.sources import served_tap as ST  # noqa: E402
from verity_vllm.acquire.sources import taps  # noqa: E402

EPSILON = float.fromhex(json.loads(EPS)["f64"])
ONE, RESID = PF[len("model."):], IL[len("model."):]               # the one-input norm and the x + residual norm
EXACT = ("scale", "resid", "sq")                                    # the roles CPU torch rounds as the Program does


def chain(x, weight, eps, residual=None):
    """Each value of vLLM's GemmaRMSNorm forward by its role, computed apart from the module."""
    out = {"scale": weight.float() + 1.0}
    x = x.to(torch.float32)
    if residual is not None:
        x = out["resid"] = x + residual.to(torch.float32)
    out["sq"] = x.pow(2)
    out["var"] = out["sq"].mean(dim=-1, keepdim=True)
    out["var_eps"] = out["var"] + eps
    out["rstd"] = torch.rsqrt(out["var_eps"])
    out["normed"] = x * out["rstd"]
    out["out"] = out["normed"].to(torch.float32) * out["scale"]
    return out


class GemmaRMSNorm(torch.nn.Module):
    """The served module: vLLM's forward_native over the native IR ops; `form` another forward (`plain`: x * w, a Llama norm's; `fused`:
    one kernel, no chain; `stash`: the variance kept where a later hook writes it)."""

    def __init__(self, weight, eps=EPSILON, form=None):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.variance_epsilon, self.form = eps, form

    def forward(self, x, residual=None):
        weight = self.weight.float() + 1.0
        orig = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig)
        if self.form == "fused":
            y = torch.nn.functional.rms_norm(x, (x.shape[-1],), weight, self.variance_epsilon)
        else:
            variance = x.pow(2).mean(dim=-1, keepdim=True)
            self.last = variance
            x = x * torch.rsqrt(variance + self.variance_epsilon)
            y = x.to(weight.dtype) * (self.weight.float() if self.form == "plain" else weight)
        return y.to(orig) if residual is None else (y.to(orig), residual)


def put(root, path, module):
    owner, leaf = path.rsplit(".", 1)
    m = root
    for part in owner.split("."):
        if part not in m._modules:
            m.add_module(part, Probe())
        m = m._modules[part]
    m.add_module(leaf, module)


def norm_weight(world, m):
    return _tensor(world.fields[f"model.{m}.weight"], torch.bfloat16)


def _commit(world, monkeypatch, *, norm=True, forms=None, eps=EPSILON, man=None, programs=None, plan=None, probes=()):
    monkeypatch.setattr(CB, "CLAIMS", [])
    model = runtime(["model.embed_tokens", "logits_processor", *probes],
                    {n[len("model."):]: _tensor(w, torch.bfloat16) for n, w in world.fields.items() if "layernorm" not in n})
    for m in (ONE, RESID):
        if m not in probes:
            put(model, m, GemmaRMSNorm(norm_weight(world, m), eps, (forms or {}).get(m)))
    com = Recorder(model)
    taps.attach(com, man or world.man, ("instance_outputs",), taps.TapSources(norm_body_tap=int(norm)), programs=programs or (world.pdir,),
                call_boundary_plan=plan)
    for s in com.sources:
        s.install()
    return com


def rows(world, calls):
    return _tensor(np.array([world.vals[c][:H] for c in calls], dtype=np.uint16), torch.bfloat16)


def _drive(world, com, steps=range(1 + T), runs=lambda s, m: 1):
    mods = dict(com.model.named_modules())
    for s in steps:
        com._cur = types.SimpleNamespace(step=s)
        for src in com.sources:
            src.on_step_begin()
        for _ in range(runs(s, ONE)):
            mods[ONE](rows(world, world.at[(s, "e")]))
        for _ in range(runs(s, RESID)):
            mods[RESID](rows(world, world.at[(s, "x")]), residual=rows(world, world.at[(s, "e")]))
        mods["logits_processor"](mods["model.embed_tokens"], rows(world, world.at[(s, "y")][-1:]))
        for src in com.sources:
            if hasattr(src, "on_step_end"):
                src.on_step_end(com._cur)
    com._cur = None


def replica(world, eps=EPSILON):
    """(engine step, module) -> role -> the forward's value, computed apart."""
    return {(s, m): chain(rows(world, world.at[(s, "x" if m == RESID else "e")]), norm_weight(world, m), eps,
                          rows(world, world.at[(s, "e")]) if m == RESID else None) for s in range(1 + T) for m in (ONE, RESID)}


def same(a, b):
    return np.array_equal(CB.words(torch, a), CB.words(torch, b))


def _layout(com):
    return [(s, n, tuple(t.shape), t.dtype) for s, n, t in com.acquired]


def test_an_honest_forward_commits_the_host_stream_with_the_forwards_words(dense, monkeypatch):
    host = _commit(dense, monkeypatch, norm=False)
    assert [s.short_name for s in host.sources] == ["call_boundary"]
    _drive(dense, host)
    com = _commit(dense, monkeypatch)
    tap, cb = com.sources
    assert (tap.short_name, cb.short_name) == ("norm_body_tap", "call_boundary")
    ids = NT.manifest_sites(dense.man)
    assert len(ids) == (1 + T) * (7 + 8) and {NT.role(r) for r in ids} == {*EXACT, "var", "var_eps", "rstd", "normed", "out"}
    assert set(cb.plan.bodies) == {(s, "logits_processor") for s in range(1 + T)} and cb.modules == ["logits_processor"]
    _drive(dense, com)
    assert not host.errors and not com.errors, com.errors
    assert _layout(com) == _layout(host)
    got, ref, want = {(s, n): t for s, n, t in com.acquired}, {(s, n): t for s, n, t in host.acquired}, replica(dense)
    for r in ids:
        k = (int(r["engine_step"]), ST.name(r))
        assert same(got[k], want[(k[0], ST.module_of(r))][NT.role(r)]), k
        assert NT.role(r) not in EXACT or same(got[k], ref[k]), k
    assert all(same(got[k], ref[k]) for k in got if k[1].startswith("logits_processor/"))
    tapped = {(int(r["engine_step"]), ST.name(r)) for r in ids}
    assert cb.identity_sources() == {k: "tap" if k in tapped else "host" for k in ref}
    assert tap.stats["tensors"] == len(ids) and tap.stats["steps"] == 1 + T and tap.stats["errors"] == 0
    assert tap.stats["bytes"] == sum(int(t.numel()) * t.element_size() for k, t in got.items() if k in tapped)
    d = tap.declaration_classes()[0]
    assert (d["class"], d["how"], d["source"], d["tensors"]) == (CB.FAMILY, "bound-as-leaf", "tap", len(ids))
    assert tap.extra_step_bytes({}, None)["tensors_per_step"] == len(ids) // (1 + T)


def test_the_binding_map_records_each_identitys_source_inside_its_digest(dense, monkeypatch):
    com = _commit(dense, monkeypatch)
    _drive(dense, com)
    layouts = {e: [types.SimpleNamespace(name=n, shape=list(t.shape), stream_off=0, first_leaf=0, chunk=256, nbytes=t.numel() * t.element_size(),
                                         dtype=str(t.dtype), ordinal=i) for i, (s, n, t) in enumerate(com.acquired) if s == e] for e in range(1 + T)}
    args = (layouts, [types.SimpleNamespace(root=bytes(32))] * (1 + T), bytes(32), "w" * 64)
    with_src, without = B.build_binding_map(*args, sources=B.identity_sources(com)), B.build_binding_map(*args)
    by = {x["name"]: x.get("source") for x in with_src["entries"]}
    assert {n for n, s in by.items() if s == "tap"} == {ST.name(r) for r in NT.manifest_sites(dense.man)}
    assert {n for n, s in by.items() if s == "host"} == {n for n in by if n.startswith("logits_processor/")}
    assert not any("source" in x for x in without["entries"]) and with_src["digest"] != without["digest"]
    assert B.identity_sources(Recorder()) == {}


def test_a_module_with_both_sources_commits_them_in_the_manifests_order(dense, monkeypatch):
    """The means unclaimed: the call-boundary source evaluates them beside the served values of their modules, in the host's places."""
    host = _commit(dense, monkeypatch, norm=False)
    _drive(dense, host)
    monkeypatch.setattr(NT, "ROLES", {k: v for k, v in NT.ROLES.items() if v != "var"})
    com = _commit(dense, monkeypatch)
    tap, cb = com.sources
    assert {m for _e, m in cb.plan.bodies} == {"logits_processor", ONE, RESID}
    _drive(dense, com)
    assert not com.errors, com.errors
    assert _layout(com) == _layout(host)
    sources = cb.identity_sources()
    ref = {(s, n): t for s, n, t in host.acquired}
    for s, n, t in com.acquired:
        assert sources[(s, n)] == ("host" if "/mean_" in n or n.startswith("logits_processor/") else "tap"), n
        assert sources[(s, n)] == "tap" or same(t, ref[(s, n)]), n


def test_the_tap_commits_what_the_forward_served(dense, monkeypatch):
    """Another epsilon in the served norms: their words are what that forward computed, not the Program's (the replay's openings of the
    producing Calls are what check them)."""
    com = _commit(dense, monkeypatch, eps=2 * EPSILON)
    _drive(dense, com)
    assert not com.errors
    got, want, honest = {(s, n): t for s, n, t in com.acquired}, replica(dense, 2 * EPSILON), replica(dense)
    for r in NT.manifest_sites(dense.man):
        k, v = (int(r["engine_step"]), ST.name(r)), want[(int(r["engine_step"]), ST.module_of(r))][NT.role(r)]
        assert same(got[k], v)
    assert not same(want[(0, ONE)]["var_eps"], honest[(0, ONE)]["var_eps"])


def test_a_forward_of_another_form_is_a_collector_error_that_commits_nothing_of_the_module(dense, monkeypatch):
    com = _commit(dense, monkeypatch, forms={ONE: "plain", RESID: "fused"})
    _drive(dense, com, steps=[0])
    assert com.errors == [f"norm body tap: {ONE} at engine step 0: the output's other factor is not the module weight's .float() + 1; "
                          "nothing of it is committed",
                          f"norm body tap: {RESID} at engine step 0: the forward ran 0 rsqrt, not one; nothing of it is committed"]
    assert {n.split("/")[0] for _s, n, _t in com.acquired} == {"logits_processor"}

    com = _commit(dense, monkeypatch, forms={ONE: "stash"})
    com.model.get_submodule(ONE).register_forward_hook(lambda m, _a, _o: m.last.add_(1.0))
    _drive(dense, com, steps=[0])
    assert len(com.errors) == 1 and f"{ONE}/mean_0/out of r0 @ engine step 0 was written in place after the forward computed it: nothing " \
                                   f"of {ONE} at engine step 0 is committed" in com.errors[0], com.errors
    assert not any(n.startswith(ONE) for _s, n, _t in com.acquired) and any(n.startswith(RESID) for _s, n, _t in com.acquired)

    com = _commit(dense, monkeypatch)
    _drive(dense, com, steps=[0, 1], runs=lambda s, m: {(0, ONE): 0, (1, RESID): 2}.get((s, m), 1))
    assert com.errors == [f"norm body tap: {ONE} did not run at engine step 0; the manifest names its call boundaries there",
                          f"norm body tap: {RESID} ran twice at engine step 1; the Program has one body"]
    assert {(s, n.split("/")[0]) for s, n, _t in com.acquired if not n.startswith("logits_processor/")} == {(0, RESID), (1, ONE), (1, RESID)}


def test_the_warm_up_commits_zero_words_of_the_identities_and_captures_nothing(dense, monkeypatch):
    honest = _commit(dense, monkeypatch)
    _drive(dense, honest, steps=[0])
    com = _commit(dense, monkeypatch)
    taps.warmup(com, True)
    assert all(s.placeholder for s in com.sources)
    _drive(dense, com, steps=[0])
    taps.warmup(com, False)
    assert not com.errors and _layout(com) == _layout(honest)
    assert all(not t.any() for _s, _n, t in com.acquired) and not com.sources[0]._served


def test_a_manifest_or_model_the_tap_cannot_commit_is_refused_and_the_gate_names_it(dense, pair, monkeypatch):
    monkeypatch.setattr(CB, "CLAIMS", [NT.claim])
    ids = NT.manifest_sites(dense.man)
    rec = G.call_boundaries(dense.man, os.path.dirname(dense.pdir))
    assert rec["pass"] is True and rec["sources"] == ["norm_body_tap", "call_boundary"] and rec["claimed"] == len(ids)
    assert NT.refusal({**dense.man, "ranks": 2}, ids) == "a manifest of 2 ranks (the tap serves one rank)"
    assert "names the served ['scale'] 2 times" in NT.refusal(dense.man, [*ids, ids[0]])
    assert "a manifest of 2 requests" in NT.refusal(pair.man, NT.manifest_sites(pair.man)) and "norm_body_tap" not in taps.attached(pair.man)
    rec = G.call_boundaries(pair.man, pair.root)
    assert rec["pass"] is False and "claimed by norm_body_tap" in rec["named"]["unattached"][0]
    for kw, why in (({"man": pair.man, "programs": (pair.root,)}, None), ({"probes": (ONE,)}, f"{ONE} is a Probe without a weight parameter")):
        with pytest.raises(SystemExit) as e:
            _commit(pair if "man" in kw else dense, monkeypatch, **kw)
        assert e.value.code == 3
        assert why is None or NT.model_refusal(runtime([ONE]), [ONE]) == why
    assert NT.model_refusal(runtime([]), [ONE]) == f"{ONE} is not a module of the model"
    monkeypatch.setattr(CB, "CLAIMS", [])
    with pytest.raises(SystemExit) as e:
        _commit(dense, monkeypatch, plan=CB.plan_of(dense.man, dense.pdir))   # a plan derived without the claim evaluates the norms
    assert e.value.code == 3
    NT.claiming(False)
    assert not CB.CLAIMS and not NT.manifest_sites(dense.man) and NT.attach(Recorder(), dense.man, ("a",)) == (("a",), None)
