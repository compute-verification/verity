"""The CPU gate's two halves on a tiny module tree: `v1_decision` is held to the REAL committer's decision (same tables, same module
tree, no engine), and `compare_sites` names every difference between v1's hook set and the plan's site set with its table."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from verity_capture.commit import required_manifest as RM  # noqa: E402
from verity_capture.experimental.commit_integ.native_host import ACQUIRE_CLASSES, ACQUIRE_INPUT_CLASSES, SKIP_CLASSES, NativeHostCommitter  # noqa: E402
from verity_vllm.acquire import gate as G  # noqa: E402
from verity_vllm.acquire.correspondence import RUNNER_PATH, AnnotationCorrespondence, runtime_tree_of  # noqa: E402
from verity_vllm.acquire.install import apply_plan_to_committer  # noqa: E402
from verity_vllm.acquire.plan import acquisition_plan  # noqa: E402
from verity_vllm.acquire.v1_decision import v1_decision  # noqa: E402

PD = "1" * 64

# module classes NAMED after v1's tables (the MRO name is what the tables match); the tree is the test's own
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
    """A manifest whose Values are exactly what v1's tables hook on `Model` (one Value per hooked module, the ids at the embedding)."""
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


def test_v1_decision_matches_committer():
    model, runner = Model(), Runner()
    rt = runtime_tree_of(model, runner, source="test")
    man = manifest(rows_matching_v1())
    d = v1_decision(rt, man)
    c = offline_committer(model, runner)
    assert d.outputs == [n for n, _m in c._modules]                                             # same modules, same (named_modules) order
    assert d.inputs == [n for n, m in c._modules if c._acquires_input(n, m)] + [n for n, _m in c._input_only_modules()]
    assert "lm_head" not in d.outputs and RUNNER_PATH in d.outputs and "layers.0.mixer" not in d.outputs
    assert d.runner == ["sampled_token_ids"] and d.weights and not d.problems
    # promoted rows arm consumer inputs exactly as commit_delta's `promoted_observation` does
    rows = rows_matching_v1() + [row("instance_outputs", "layers.0.mixer", RM.PROMOTED_MEMBER, **RM.promoted_annotations({"rows": 1, "width": 4, "dtype": "bf16", "spec": "Add_v1", "consumers": ["layers.0.post"]})),
                                 row("instance_outputs", "layers.1.mixer", RM.PROMOTED_MEMBER, **RM.promoted_annotations({"rows": 1, "width": 4, "dtype": "bf16", "spec": "Add_v1", "consumers": ["layers.1.mixer"]}))]
    man2 = manifest(rows)
    d2 = v1_decision(rt, man2)
    pia = RM.promoted_observation(man2)
    c2 = offline_committer(model, runner, acquire_inputs=pia["consumers"])
    assert d2.inputs == [n for n, m in c2._modules if c2._acquires_input(n, m)] + [n for n, _m in c2._input_only_modules()]
    assert "layers.0.post" in d2.inputs and "layers.1.mixer" in d2.inputs                      # a consumer of an acquire class, and one of none
    assert d2.causes["layers.1.mixer|consumer_input"].startswith("manifest promoted consumer of no acquire class")


def test_compare_sites_identical_when_the_manifest_names_every_v1_hook():
    model, runner = Model(), Runner()
    rt = runtime_tree_of(model, runner, source="test")
    man = manifest(rows_matching_v1())
    plan, d, cmp = G.run(man, rt, row_id="synthetic")
    assert plan.complete
    # the one difference is v1's sampler-module hook (never fires): a RESIDUAL, explained by its table
    assert [x["site"] for x in cmp["differences"]] == [[RUNNER_PATH, "producer_output", "*"]]
    assert cmp["differences"][0]["cause"].startswith("residual ACQUIRE_CLASSES")
    assert cmp["unexplained"] == [] and cmp["counts"]["only_plan"] == 0 and cmp["counts"]["only_v1"] == 1
    assert cmp["plan_output_slots"]["layers.0.post"] == ["0", "1"]
    assert cmp["counts"]["common_by_kind"] == {"producer_output": 11, "consumer_input": 1, "runner_return": 1, "registered_input": 1}


def test_compare_sites_names_differences_by_table():
    model, runner = Model(), Runner()
    rt = runtime_tree_of(model, runner, source="test")
    rows = [r for r in rows_matching_v1() if r["op_path"] != "norm"]                             # v1 hooks `norm` by class; no Value there
    rows.append(row("instance_outputs", "layers.0.mixer", "out"))                               # the manifest requires a Value of a class v1 never hooks
    rows.append(row("instance_outputs", "lm_head", "out"))                                      # ... and one of a SKIP class
    man = manifest(rows)
    plan, d, cmp = G.run(man, rt, row_id="synthetic")
    by = {tuple(x["site"]): x for x in cmp["differences"]}
    assert by[("norm", "producer_output", "*")]["in"] == "v1" and by[("norm", "producer_output", "*")]["cause"].startswith("residual ACQUIRE_CLASSES")
    # commit_delta extends v1's acquisition by the manifest's non-promoted instance_outputs op_paths: mixer and lm_head are hooked by BOTH
    assert ("layers.0.mixer", "producer_output", "*") not in by and ("lm_head", "producer_output", "*") not in by
    assert d.causes["layers.0.mixer|producer_output"].startswith("commit_delta") and d.causes["lm_head|producer_output"].startswith("commit_delta")
    assert cmp["unexplained"] == []
    # a v1 hook the plan cannot explain: an acquire class the residual pass does not see (tables without it) -> UNEXPLAINED, named
    t = {"ACQUIRE_CLASSES": set(ACQUIRE_CLASSES), "ACQUIRE_INPUT_CLASSES": set(ACQUIRE_INPUT_CLASSES), "SKIP_CLASSES": set(SKIP_CLASSES),
         "HAZARD_LEAVES": set(NativeHostCommitter.HAZARD_LEAVES), "GATHER_FLUSH_LEAVES": set(), "PRE_FLUSH_LEAVES": set()}
    d2 = v1_decision(rt, man, tables=dict(t, ACQUIRE_CLASSES=t["ACQUIRE_CLASSES"] | {"Residualizer"}))
    plan2 = acquisition_plan(man, AnnotationCorrespondence(rt), tables=t)
    cmp2 = G.compare_sites(plan2, d2, rt, tables=t)
    assert [("layers.1.mixer", "producer_output", "*")] == [tuple(k) for k in cmp2["unexplained"]]


def test_apply_plan_to_committer_replaces_the_class_tables():
    model, runner = Model(), Runner()
    rt = runtime_tree_of(model, runner, source="test")
    rows = [r for r in rows_matching_v1() if r["op_path"] != "norm"]
    rows.append(row("instance_outputs", "layers.1.mixer", RM.PROMOTED_MEMBER, **RM.promoted_annotations({"rows": 1, "width": 4, "dtype": "bf16", "spec": "Add_v1", "consumers": ["layers.1.post"]})))
    man = manifest(rows)
    plan = acquisition_plan(man, AnnotationCorrespondence(rt))
    assert plan.complete
    c = offline_committer(model, runner, acquire_inputs=RM.promoted_observation(man)["consumers"])
    before = [n for n, _m in c._modules]
    assert "norm" in before and c.plan_inputs is None
    rec = apply_plan_to_committer(c, plan)
    after = [n for n, _m in c._modules]
    assert "norm" not in after and RUNNER_PATH not in after and after == sorted(after, key=lambda n: before.index(n) if n in before else 99)
    assert set(after) == {s.module for s in plan.hooks() if s.kind == "producer_output"}
    assert c.plan_inputs == {"embed_tokens", "layers.1.post"} and rec["engine"] == "v2" and rec["modules_before"] == before
    mods = dict(model.named_modules())
    assert c._acquires_input("embed_tokens", mods["embed_tokens"]) and c._acquires_input("layers.1.post", mods["layers.1.post"])
    assert not c._acquires_input("layers.0.post", mods["layers.0.post"]) and c._input_only_modules() == []
    assert c.name.endswith("+plan")
    # a plan that does not name an armed consumer is refused
    c2 = offline_committer(model, runner, acquire_inputs={"layers.0.post": 8})
    with pytest.raises(Exception, match="arms consumer inputs the plan does not name"):
        apply_plan_to_committer(c2, plan)
