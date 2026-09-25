"""The CPU gate and the committer adapter on a tiny module tree: the gate's plan and plan-vs-plan comparison, the REAL committer taking
its selection from the plan (`apply_plan_to_committer`), VERITY_SKIP on that selection, and the compiled rows' record ops held to it."""
from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from verity_vllm.query.manifest import format as MF  # noqa: E402
from verity_vllm.query.manifest import format as RM  # noqa: E402
from verity_vllm.commit.committer.native_host import NativeHostCommitter, SkipRefused  # noqa: E402
from verity_vllm.acquire import gate as G  # noqa: E402
from verity_vllm.correspondence.reader_for_acquire import RUNNER_PATH, AnnotationCorrespondence, runtime_tree_of  # noqa: E402
from verity_vllm.acquire.install import apply_plan_to_committer  # noqa: E402
from verity_vllm.acquire.plan import acquisition_plan  # noqa: E402

PD = "1" * 64

# module classes NAMED after the committer's class tables (its construction-time selection matches the MRO name); the tree is the test's own
Norm = type("RMSNorm", (nn.Module,), {"forward": lambda self, x, r=None: (x, x + r) if r is not None else x})
Lin = type("LinearBase", (nn.Module,), {"forward": lambda self, x: x})
Emb = type("VocabParallelEmbedding", (nn.Module,), {"forward": lambda self, ids: ids.float()})
Head = type("ParallelLMHead", (nn.Module,), {"forward": lambda self, x: x})
Logit = type("LogitsProcessor", (nn.Module,), {"forward": lambda self, x: x})
Samp = type("Sampler", (nn.Module,), {"forward": lambda self, x: x})
Plain = type("Residualizer", (nn.Module,), {"forward": lambda self, x: x})


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm, self.qkv_proj, self.o_proj, self.post = Norm(), Lin(), Lin(), Norm()
        self.mixer = Plain()


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = Emb()
        self.layers = nn.ModuleList([Layer(), Layer()])
        self.norm = Norm()
        self.lm_head = Head()
        self.logits_processor = Logit()
        self.register_buffer("layers_rope_cos_sin_cache", torch.zeros(2))


class Runner:
    def __init__(self):
        self.sampler = Samp()

    def sample(self, x):
        return x


def row(family, op, member, step=0, **extra):
    return {"family": family, "program_digest": PD, "request_id": "r0", "step": step, "invocation": 0, "op_path": op, "output_member": member,
            "element_range": [0, 4], "rank": 0, **extra}


def manifest(rows):
    fams: dict[str, int] = {}
    for r in rows:
        fams[r["family"]] = fams.get(r["family"], 0) + 1
    return {"schema": RM.SCHEMA, "program_digest": PD, "identities": rows, "populations": {"identities": len(rows), "by_family": fams},
            "manifest_digest": RM.manifest_digest(rows), "wrapper_prefix": "", "op_path_aliases": {}}


def rows_matching_v1():
    """A manifest whose Values are exactly what the class tables hook on `Model` (one Value per hooked module, the ids at the embedding)."""
    rows = [row("input_token_ids", "embed_tokens", "in", dtype="i32"), row("instance_outputs", "embed_tokens", "out")]
    for i in range(2):
        rows += [row("instance_outputs", f"layers.{i}.input_layernorm", "out"), row("instance_outputs", f"layers.{i}.qkv_proj", "0"),
                 row("instance_outputs", f"layers.{i}.o_proj", "0"), row("instance_outputs", f"layers.{i}.post", "0"), row("instance_outputs", f"layers.{i}.post", "1")]
    rows += [row("instance_outputs", "norm", "out"), row("instance_outputs", "logits_processor", "out"),
             row("sampled_token_ids", RUNNER_PATH, "sampled_token_ids", dtype="i64"), row("weights", "layers_rope_cos_sin_cache", "weight", step="*", dtype="f32")]
    return rows


def offline_committer(model, runner, acquire_inputs=None):
    c = NativeHostCommitter(None, run_id="gate-test")
    c.model, c.runner = model, runner
    c.acquire_inputs = dict(acquire_inputs or {})
    c._modules = c._select_modules()
    return c


def test_gate_plan_and_plan_compare():
    model, runner = Model(), Runner()
    rt = runtime_tree_of(model, runner, source="test")
    plan = G.run(manifest(rows_matching_v1()), rt)
    assert plan.complete and not plan.unsupported
    kinds = {k: len({m for m, _k, _s in v}) for k, v in plan.sites_by_kind().items()}
    assert kinds == {"producer_output": 11, "consumer_input": 1, "runner_return": 1, "registered_input": 1}
    doc = plan.to_json()
    assert G.compare_plans(doc, doc)["identical"]
    rows = [r for r in rows_matching_v1() if r["op_path"] != "norm"]
    other = G.run(manifest(rows), rt).to_json()
    cmp = G.compare_plans(doc, other)
    assert not cmp["identical"] and cmp["only_a"] == ["norm/out"] and not cmp["differing"]


def test_apply_plan_to_committer_replaces_the_class_tables():
    model, runner = Model(), Runner()
    rt = runtime_tree_of(model, runner, source="test")
    rows = [r for r in rows_matching_v1() if r["op_path"] != "norm"]
    rows.append(row("instance_outputs", "layers.1.mixer", RM.PROMOTED_MEMBER, **RM.promoted_annotations({"rows": 1, "width": 4, "dtype": "bf16", "spec": "Add_v1", "consumers": ["layers.1.post"]})))
    man = manifest(rows)
    plan = acquisition_plan(man, AnnotationCorrespondence(rt))
    assert plan.complete
    c = offline_committer(model, runner, acquire_inputs=MF.promoted_observation(man)["consumers"])
    before = [n for n, _m in c._modules]
    assert "norm" in before and c.plan_inputs is None
    rec = apply_plan_to_committer(c, plan)
    after = [n for n, _m in c._modules]
    assert "norm" not in after and RUNNER_PATH not in after and after == sorted(after, key=lambda n: before.index(n) if n in before else 99)
    assert set(after) == {s.module for s in plan.hooks() if s.kind == "producer_output"}
    assert c.plan_inputs == {"embed_tokens", "layers.1.post"} and rec["engine"] == "v2"
    mods = dict(model.named_modules())
    assert c._acquires_input("embed_tokens", mods["embed_tokens"]) and c._acquires_input("layers.1.post", mods["layers.1.post"])
    assert not c._acquires_input("layers.0.post", mods["layers.0.post"]) and c._input_only_modules() == []
    assert c.name.endswith("+plan")
    # a plan that does not name an armed consumer is refused
    c2 = offline_committer(model, runner, acquire_inputs={"layers.0.post": 8})
    with pytest.raises(Exception, match="arms consumer inputs the plan does not name"):
        apply_plan_to_committer(c2, plan)


def promoted_plan(model, runner):
    """v1's Values minus `norm`, plus a promoted member of `layers.1.mixer` observed at the input of `layers.1.post` (an RMSNorm)."""
    rows = [r for r in rows_matching_v1() if r["op_path"] != "norm"]
    rows.append(row("instance_outputs", "layers.1.mixer", RM.PROMOTED_MEMBER, **RM.promoted_annotations({"rows": 1, "width": 4, "dtype": "bf16", "spec": "Add_v1", "consumers": ["layers.1.post"]})))
    man = manifest(rows)
    plan = acquisition_plan(man, AnnotationCorrespondence(runtime_tree_of(model, runner, source="test")))
    assert plan.complete
    return man, plan


def engine_committer():
    """The committer's construction path (an engine: model + runner), without CUDA: no explicit arena, a given geometry digest."""
    return NativeHostCommitter(types.SimpleNamespace(model=Model(), runner=Runner()), run_id="skip-test", arena_bytes=None, geo_digest=b"g" * 32)


NORMS = ["layers.0.input_layernorm", "layers.0.post", "layers.1.input_layernorm", "layers.1.post"]


def test_verity_skip_is_applied_to_the_plan_selection(monkeypatch):
    """The plan replaces the module list, so the skip is applied to the PLAN's selection: the class's producer_output hooks go,
    a consumer-input site on a dropped module stays (the promoted member there is `layers.1.mixer`'s output), the record names the Values
    left unacquired, and the name says what was dropped."""
    monkeypatch.setenv("VERITY_SKIP", "RMSNorm")
    model, runner = Model(), Runner()
    man, plan = promoted_plan(model, runner)
    c = offline_committer(model, runner, acquire_inputs=MF.promoted_observation(man)["consumers"])
    rec = apply_plan_to_committer(c, plan)
    after = [n for n, _m in c._modules]
    assert rec["skip"]["modules"] == {"RMSNorm": NORMS} and c.skipped == {"RMSNorm": NORMS}
    assert after == rec["modules_after"] == ["embed_tokens", "layers.0.qkv_proj", "layers.0.o_proj", "layers.1.qkv_proj", "layers.1.o_proj", "logits_processor"]
    assert [n for n, _m in c._input_only_modules()] == ["layers.1.post"]
    assert rec["skip"]["values_unacquired"] == sorted(f"layers.{i}.{v}" for i in range(2) for v in ("input_layernorm/out", "post/0", "post/1"))
    assert c.name.endswith("+plan-skip_RMSNorm")
    c._check_skip()


@pytest.mark.parametrize("skip, why", [("Residualizer", "match none of the"), ("RMSNorm,Typo", r"\['Typo'\] match none of the"),
                                       ("Sampler", "is the runner's sampler")])
def test_verity_skip_that_cannot_happen_is_refused(monkeypatch, skip, why):
    """A class the plan selects no module of (the mixer carries no Value; a typo) or the runner's sampler (its ids come from the sample()
    wrap, which closes every step) is refused by name -- never a run whose name claims a skip that did not happen."""
    monkeypatch.setenv("VERITY_SKIP", skip)
    model, runner = Model(), Runner()
    man, plan = promoted_plan(model, runner)
    c = offline_committer(model, runner, acquire_inputs=MF.promoted_observation(man)["consumers"])
    with pytest.raises(SkipRefused, match=why):
        apply_plan_to_committer(c, plan)
    assert "-skip_" not in c.name and c.skipped is None


def test_install_refuses_an_unapplied_or_undone_skip(monkeypatch):
    monkeypatch.setenv("VERITY_SKIP", "RMSNorm")
    c2 = engine_committer()
    assert c2.skipped is None and "-skip_" not in c2.name
    with pytest.raises(SkipRefused, match="never applied"):
        c2.install()
    man, plan = promoted_plan(c2.model, c2.runner)
    c2.acquire_inputs = dict(MF.promoted_observation(man)["consumers"])
    apply_plan_to_committer(c2, plan)
    c2._modules.append(("layers.0.post", dict(c2.model.named_modules())["layers.0.post"]))   # re-added after the skip
    with pytest.raises(SkipRefused, match="selected again at install"):
        c2.install()
    assert not c2.installed


def recorded(monkeypatch, cs, src):
    """The names the compiled record op sees (the op body dispatches to the live source's `_record`)."""
    seen: list[str] = []
    monkeypatch.setattr(cs, "_SRC", src)
    monkeypatch.setattr(src, "_record", lambda x, nid: seen.append(src.names[nid & cs.BASE_MASK]))
    return seen


@pytest.mark.parametrize("skip", ["", "RMSNorm"])
def test_compiled_source_wraps_the_committers_plan_selection(monkeypatch, tmp_path, skip):
    """Compiled rows' record ops (wrapped before vLLM compiles) are the committer's plan selection -- not the class tables (which would
    record `norm`, a module the plan does not select) -- with or without a skip; the plan derived from the manifest at wrap time is the
    committer's plan (same digest)."""
    from verity_vllm.acquire.sources import compiled_source as cs

    cs._register_op()
    monkeypatch.setenv("VERITY_SKIP", skip)
    model, runner = Model(), Runner()
    man, plan = promoted_plan(model, runner)
    c = offline_committer(model, runner, acquire_inputs=MF.promoted_observation(man)["consumers"])
    apply_plan_to_committer(c, plan)
    src = cs.CompiledGraphSource(rows=4)
    src.use_plan(man, tmp_path)
    src.wrap_model(model, runner)
    assert src.plan.digest == plan.digest and (tmp_path / "acquisition_plan.json").is_file()
    assert src.selection["outputs"] == [n for n, _m in c._modules] and "norm" not in src.wrapped
    assert src.selection["inputs"] == ["embed_tokens", "layers.1.post"]
    assert set(src.selection["outputs"]).isdisjoint(NORMS) == bool(skip)
    src.check_selection(c)
    seen = recorded(monkeypatch, cs, src)
    x = torch.randn(2, 4)
    model.layers[1].post(x, x)
    model.layers[0].qkv_proj(x)
    assert seen == (["layers.1.post/in"] if skip else ["layers.1.post/in", "layers.1.post/0", "layers.1.post/1"]) + ["layers.0.qkv_proj"]


def test_compiled_source_refuses_without_a_plan_or_off_the_committers_selection(monkeypatch):
    from verity_vllm.acquire.sources import compiled_source as cs

    cs._register_op()
    model, runner = Model(), Runner()
    with pytest.raises(RuntimeError, match="no plan or required-value manifest was passed in"):
        cs.CompiledGraphSource(rows=4).wrap_model(model, runner)
    assert not any("forward" in m.__dict__ for _n, m in model.named_modules())
    man, plan = promoted_plan(model, runner)
    src = cs.CompiledGraphSource(rows=4)
    src.wrap_model(model, runner, plan=plan)
    c = offline_committer(model, runner)                        # the class tables' selection: not the plan's
    with pytest.raises(RuntimeError, match="not the committer's plan selection"):
        src.check_selection(c)
