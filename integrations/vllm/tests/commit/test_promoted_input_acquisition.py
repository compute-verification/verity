"""Collector side: MANIFEST-DRIVEN input acquisition for PROMOTED members.

A value produced in a PARENT module's forward body (no module output carries it) and consumed by a CHILD leaf is required by the manifest
as the promoted member (parent, `out`) with the child named as its consumer.  The collector then acquires the child's first positional
input as `<child>/in` -- the existing input-acquisition mechanism, no longer restricted to a module class -- and the binding map binds
that committed tensor to the PRODUCER's identity (`observation: consumer-input`).  Tiny nn.Module tree on the CPU, no vllm, no CUDA: the
committer's hook installation and packed commit path are exercised with a CPU acquisition (the CUDA arena copy is the only thing
substituted).  Nothing here names a model, a vLLM class or a module path of the real case: the tree is the shape, the manifest drives it.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from verity_vllm.commit import binding as B  # noqa: E402
from verity_vllm.query.manifest.bind import Binder  # noqa: E402
from verity_vllm.query.manifest import format as RM  # noqa: E402
from verity_vllm.commit.committer.native_host import NativeHostCommitter, TensorMeta, _StepJob, _dtype_name  # noqa: E402

V, H, TOKENS = 8, 16, 3
PD = hashlib.sha256(b"program").hexdigest()


# ------------------------------------------------------------------------------------------------------------ the tiny module tree

class _Embed(torch.nn.Module):
    """The embedding leaf: an ACQUIRE class of this test (its output is a member of its own, like the real gather)."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(V, H, generator=torch.Generator().manual_seed(3)).to(torch.bfloat16), requires_grad=False)

    def forward(self, ids):
        return self.weight[ids]


class _Norm(torch.nn.Module):
    """The consumer leaf: of NO acquire class -- the committer selects it for nothing on its own."""

    def forward(self, x):
        return x * 2


class _Body(torch.nn.Module):
    """The parent: scales the embedding in its OWN forward body.  The scaled value is read by the child leaf and is the output of no
    module -- the shape of the finding (a root module's rows whose output crosses into its children)."""

    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = _Embed()
        self.register_buffer("normalizer", torch.tensor(48.0, dtype=torch.bfloat16), persistent=False)
        self.norm = _Norm()

    def forward(self, ids):
        h = self.embed_tokens(ids)
        h = h * self.normalizer
        return self.norm(h)


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Body()

    def forward(self, ids):
        return self.model(ids)


class _Runner:
    """The committer wraps `execute_model` / `sample` on the engine's runner; this one is never driven (the test calls the model)."""

    def execute_model(self, scheduler_output, *a, **kw):
        return None

    def sample(self, *a, **kw):
        return None


class _Engine:
    def __init__(self, model) -> None:
        self.model = model
        self.runner = _Runner()


def _manifest(consumers: list[str], *, spec: str = "Bf16MulScalarTensor_v1", extra_rows: list[dict] | None = None, rows: int = TOKENS) -> dict:
    """A required manifest fragment: the promoted row (model, out) at step 0 + the identities the collector binds by name."""
    ident = [{"family": "instance_outputs", "program_digest": PD, "request_id": "r0", "step": 0, "invocation": 0, "op_path": "model.embed_tokens",
              "output_member": "out", "element_range": [0, rows * H], "rank": 0, "rows": rows, "width": H, "dtype": "bf16", "spec": "Embedding_v1"},
             {"family": "instance_outputs", "program_digest": PD, "request_id": "r0", "step": 0, "invocation": 0, "op_path": "model",
              "output_member": RM.PROMOTED_MEMBER, "element_range": [0, rows * H], "rank": 0, **RM.promoted_annotations(
                  {"rows": rows, "width": H, "dtype": "bf16", "spec": spec, "consumers": consumers})}]
    return {"schema": RM.SCHEMA, "identities": ident + list(extra_rows or [])}


def _cpu_committer(model, *, acquire_classes: set[str], manifest: dict) -> tuple[NativeHostCommitter, dict]:
    """The committer armed the way `commit_delta` arms it from the manifest (binding.promoted_input_acquisition -> acquire_inputs), on a
    CPU model: the worker thread and the CUDA arena are not started (the packed commit is driven directly), the step's bytes retained on
    host."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("VERITY_RETAIN", "host")
        c = NativeHostCommitter(_Engine(model), hash_threads=1, program_digest=bytes.fromhex(PD), acquire_classes=acquire_classes,
                                staging="direct", geo_digest=hashlib.sha256(b"geo").digest())
    c._start_worker = lambda: None                                             # the worker hashes CUDA jobs; the test commits its step itself
    pia = B.promoted_input_acquisition(manifest)
    assert pia["problems"] == [], pia["problems"]
    c.acquire_inputs = dict(pia["consumers"])
    return c, pia


def _cpu_acquisition(c: NativeHostCommitter, job: _StepJob):
    """`_acquire` on the CPU: the tensor's bytes into the step job (what the arena snapshot does on the device), same metas."""
    def acquire(name: str, t, job_: _StepJob) -> None:
        src = t.detach().contiguous().reshape(-1)
        flat = src.view(torch.uint8) if src.dtype != torch.uint8 else src
        raw = bytes(flat.numpy())
        m = TensorMeta(len(job_.metas), name, _dtype_name(t), tuple(int(x) for x in t.shape), len(raw), c.chunk, 0, 0)
        m.host = memoryview(raw)
        job_.metas.append(m)
        c.x03_expected += 1
        c.x03_committed += 1
    return acquire


def _commit_packed(c: NativeHostCommitter, job: _StepJob):
    """The packed step stream (4-B aligned, the online layout) -> `_commit_step` (rows, layout digest, step root) -> `finalize()`."""
    buf = bytearray()
    for m in job.metas:
        off = (len(buf) + 3) // 4 * 4
        buf.extend(b"\0" * (off - len(buf)))
        m.stream_off = off
        buf.extend(bytes(m.host))
    mv = memoryview(bytes(buf))
    for m in job.metas:
        m.host = mv[m.stream_off:m.stream_off + m.nbytes]
    job.stream = mv
    job.t_step_end = 0.0
    job.event = type("_Landed", (), {"synchronize": lambda self: None})()       # the CUDA event of the step's copies: on the CPU they have landed
    c._cur = None
    c._commit_step(job)
    return c.finalize()


def _run_one_step(model, *, acquire_classes: set[str], manifest: dict):
    c, pia = _cpu_committer(model, acquire_classes=acquire_classes, manifest=manifest)
    c.install()
    try:
        c._begin_step()
        job = c._cur
        c._acquire = _cpu_acquisition(c, job)
        ids = torch.tensor([1, 5, 2][:TOKENS])
        out = model(ids)
        run = _commit_packed(c, job)
    finally:
        c.uninstall()
    scaled = (model.model.embed_tokens.weight[ids] * model.model.normalizer)
    return c, pia, run, {"ids": ids, "out": out, "scaled": scaled, "embedded": model.model.embed_tokens.weight[ids]}


def _bytes_of(t) -> bytes:
    return bytes(t.detach().contiguous().reshape(-1).view(torch.uint8).numpy())


def _committed_bytes(c: NativeHostCommitter, e: dict) -> bytes:
    off = int(e["stream_offset"])
    return bytes(c._gpu_blocks[int(e["identity"]["step"])][0][off:off + int(e["width_bytes"])].numpy())


# ------------------------------------------------------------------------------------------------------------ the manifest -> acquisition table

def test_the_manifests_promoted_rows_name_the_consumer_input_to_acquire_and_the_identity_it_binds():
    man = _manifest(["model.norm", "model.other"])
    pia = B.promoted_input_acquisition(man)
    assert pia["problems"] == []
    assert pia["names"] == {"model.norm/in": ["model", "out"]}                  # ONE observation point: the first consumer in Program order
    assert pia["consumers"] == {"model.norm": H * 2}                              # bf16: 2 B per element of the promoted signature's width
    assert pia["producers"]["model"] == {"consumer": "model.norm", "width": H, "dtype": "bf16", "spec": "Bf16MulScalarTensor_v1", "rows_n": 1, "member": "out"}
    assert pia["rule"] == B.PROMOTED_INPUT_RULE
    # a manifest without promoted rows drives nothing (every green Program: byte-identical arming)
    plain = {"schema": RM.SCHEMA, "identities": [r for r in man["identities"] if not r.get("promoted")]}
    assert B.promoted_input_acquisition(plain) == {"names": {}, "consumers": {}, "producers": {}, "problems": [], "rule": B.PROMOTED_INPUT_RULE}


def test_an_ambiguous_observation_point_is_refused_by_name_never_guessed():
    # (1) no consumer at all: nowhere to observe
    p = B.promoted_input_acquisition(_manifest([]))
    assert p["names"] == {} and any("names no consumer" in x for x in p["problems"]), p["problems"]
    # (2) two promoted producers sharing a first consumer: one committed `<consumer>/in` cannot bind two identities
    second = {"family": "instance_outputs", "program_digest": PD, "request_id": "r0", "step": 0, "invocation": 0, "op_path": "model.layers.0",
              "output_member": "out", "element_range": [0, TOKENS * H], "rank": 0,
              **RM.promoted_annotations({"rows": TOKENS, "width": H, "dtype": "bf16", "spec": "MeanTriton_v1", "consumers": ["model.norm"]})}
    p = B.promoted_input_acquisition(_manifest(["model.norm"], extra_rows=[second]))
    assert "model.norm/in" in p["names"] and len(p["names"]) == 1 and any("share the first consumer model.norm" in x for x in p["problems"]), p
    # (3) the consumer's `/in` is itself a manifest identity (the embedding's ids): the same name would carry two identities
    own_in = {"family": "input_token_ids", "program_digest": PD, "request_id": "r0", "step": 0, "invocation": 0, "op_path": "model.norm",
              "output_member": "in", "element_range": [0, TOKENS], "rank": 0, "rows": TOKENS, "width": 1, "dtype": "i32"}
    p = B.promoted_input_acquisition(_manifest(["model.norm"], extra_rows=[own_in]))
    assert p["names"] == {} and any("is itself a manifest identity" in x for x in p["problems"]), p["problems"]
    # (4) the rows of one promoted member disagree on their first consumer: one observation point per member
    man = _manifest(["model.norm"])
    other = dict(man["identities"][1], step=1, consumers=["model.layers.0.norm", "model.norm"])
    p = B.promoted_input_acquisition({"schema": RM.SCHEMA, "identities": man["identities"] + [other]})
    assert any("different first consumers" in x for x in p["problems"]), p["problems"]
    # (5) a width / dtype the arena cannot be sized for (a lifted word) is refused, not guessed
    man = _manifest(["model.norm"])
    man["identities"][1]["dtype"] = "l17"
    p = B.promoted_input_acquisition(man)
    assert p["names"] == {} and any("not usable for the arena bound" in x for x in p["problems"]), p["problems"]
    # commit_delta refuses the run on any problem (exit 3): the table is never partially applied
    assert p["consumers"] == {}


# ------------------------------------------------------------------------------------------------------------ the collector: acquisition at the consumer's input

@pytest.mark.parametrize("consumer_is_acquire_class", [False, True], ids=["consumer-of-no-acquire-class", "consumer-of-an-acquire-class"])
def test_a_value_produced_in_the_parents_forward_body_is_acquired_at_the_childs_input_and_bound_to_the_promoted_row(consumer_is_acquire_class):
    """The scaled embedding exists only inside the parent's forward: the embedding leaf returns the UNSCALED gather, the parent has no
    output of its own that is it, and the child leaf receives it.  Armed by the manifest, the committer installs a forward PRE-hook on the
    child (whether or not the child's class is one it selects on its own) and commits `<child>/in`; the binding map binds that tensor to
    the PROMOTED identity (model, out); coverage over the manifest passes, and the committed bytes are the parent's value bit for bit."""
    model = _Model()
    man = _manifest(["model.norm"])
    classes = {"_Embed"} | ({"_Norm"} if consumer_is_acquire_class else set())
    c, pia, run, vals = _run_one_step(model, acquire_classes=classes, manifest=man)
    # the hooks the manifest added: the consumer's pre-hook; the consumer's OUTPUT is acquired only when its class says so
    names = [m.name for m in c._layouts[0]]
    assert "model.norm/in" in names and "model.embed_tokens" in names, names
    assert ("model.norm" in names) == consumer_is_acquire_class, names
    assert c._acquires_input("model.norm", model.model.norm) and not c._acquires_input("model.embed_tokens", model.model.embed_tokens)
    assert [n for n, _m in c._input_only_modules()] == ([] if consumer_is_acquire_class else ["model.norm"])
    assert c.x03_expected == c.x03_committed == len(names) and c.errors == []
    # the binding map: `<consumer>/in` binds the PRODUCER's identity (model, out), says so, keeps the committed name
    bmap = B.build_binding_map(c._layouts, run.steps, run.run_root, PD, request_id="r0", rank=0, chunk=c.chunk, binder=Binder.of_manifest(man))
    by_name = {e["name"]: e for e in bmap["entries"] if e["kind"] == "leaf"}
    e = by_name["model.norm/in"]
    assert (e["identity"]["op_path"], e["identity"]["output_member"]) == ("model", "out") and e["observation"] == "consumer-input"
    assert e["identity"]["element_range"] == [0, TOKENS * H] and e["encoding"] == {"dtype": "bfloat16", "shape": [TOKENS, H]}
    assert (by_name["model.embed_tokens"]["identity"]["op_path"], by_name["model.embed_tokens"]["identity"]["output_member"]) == ("model.embed_tokens", "out")
    assert "observation" not in by_name["model.embed_tokens"]
    # coverage over the manifest's identities (the promoted row included) passes: one unambiguous binding each
    pop = [B.identity(PD, "r0", 0, 0, r["op_path"], r["output_member"], 0, r["element_range"][1]) for r in man["identities"]]
    cov = B.coverage_check(pop, bmap)
    assert cov["ok"] and cov["bound_n"] == cov["expected_n"] == 2 and cov["ambiguous_n"] == 0, cov
    # the committed bytes under (model, out) ARE the value produced in the parent's forward body -- not the embedding's own output
    got = _committed_bytes(c, e)
    assert got == _bytes_of(vals["scaled"]) and got != _bytes_of(vals["embedded"])
    assert _committed_bytes(c, by_name["model.embed_tokens"]) == _bytes_of(vals["embedded"])
    # and the value the model returned is the child's output, untouched by the hook
    assert torch.equal(vals["out"], vals["scaled"] * 2)
    # the openings of the promoted member's leaves verify against the run root like any other committed leaf
    lo, hi = int(e["leaf_lo"]), int(e["leaf_hi"])
    for op in c.open([(0, leaf) for leaf in range(lo, hi)]):
        assert c.verify(run, op), op


def test_without_the_manifests_binder_the_promoted_identity_is_unbound_by_name_and_the_consumer_input_is_an_extra():
    """The negative the rule closes: the same collector, the same committed tensor, but a map that does not know the manifest's promoted
    rows binds `<child>/in` to its own name (child, in) -- coverage names the promoted identity missing (fail closed), never covered by the
    stray `/in`."""
    model = _Model()
    man = _manifest(["model.norm"])
    c, _pia, run, _vals = _run_one_step(model, acquire_classes={"_Embed"}, manifest=man)
    bmap = B.build_binding_map(c._layouts, run.steps, run.run_root, PD, request_id="r0", rank=0, chunk=c.chunk)          # no binder
    pop = [B.identity(PD, "r0", 0, 0, r["op_path"], r["output_member"], 0, r["element_range"][1]) for r in man["identities"]]
    cov = B.coverage_check(pop, bmap)
    assert not cov["ok"] and cov["missing_n"] == 1 and cov["missing"][0]["op_path"] == "model" and cov["missing"][0]["output_member"] == "out", cov
    assert ["r0", 0, 0, "model.norm", "in", 0] in cov["extra"]
    # two maps of the same layouts differ ONLY by the binding (same leaves, same roots): the identity is a statement of the map, fixed
    # before the challenge, and its digest changes with it
    bmap2 = B.build_binding_map(c._layouts, run.steps, run.run_root, PD, request_id="r0", rank=0, chunk=c.chunk, binder=Binder.of_manifest(man))
    assert bmap2["digest"] != bmap["digest"] and bmap2["run_root"] == bmap["run_root"]


def test_a_consumer_that_is_not_a_module_of_the_built_model_leaves_the_identity_unbound_and_is_named():
    """The manifest names a consumer the model does not have: nothing is guessed (no hook, no other module's input), the promoted identity
    stays unbound and coverage names it; `_input_only_modules` skips it and commit_delta prints the NOTE."""
    model = _Model()
    man = _manifest(["model.layers.0.input_layernorm"])                          # not a module of this tree
    c, pia, run, _vals = _run_one_step(model, acquire_classes={"_Embed"}, manifest=man)
    assert pia["consumers"] == {"model.layers.0.input_layernorm": H * 2}
    assert c._input_only_modules() == [] and [m.name for m in c._layouts[0]] == ["model.embed_tokens"]
    bmap = B.build_binding_map(c._layouts, run.steps, run.run_root, PD, request_id="r0", rank=0, chunk=c.chunk, binder=Binder.of_manifest(man))
    pop = [B.identity(PD, "r0", 0, 0, r["op_path"], r["output_member"], 0, r["element_range"][1]) for r in man["identities"]]
    cov = B.coverage_check(pop, bmap)
    assert not cov["ok"] and cov["missing_n"] == 1 and cov["missing"][0]["op_path"] == "model"


def test_batched_requests_split_the_consumer_input_by_token_rows_like_any_per_token_member():
    """B > 1: the committed `<child>/in` is one batched tensor of every request's token rows; the promoted identity is per request, its
    request-local range at the request's `element_offset` (the per-token rule of the map), so each request's rows bind its own identity."""
    model = _Model()
    man = _manifest(["model.norm"])
    c, _pia, run, vals = _run_one_step(model, acquire_classes={"_Embed"}, manifest=man)
    rows = {0: [("r0", 0, 2), ("r1", 2, TOKENS)]}
    bmap = B.build_binding_map(c._layouts, run.steps, run.run_root, PD, request_rows=rows, rank=0, chunk=c.chunk, binder=Binder.of_manifest(man))
    ents = sorted((e for e in bmap["entries"] if e["name"] == "model.norm/in"), key=lambda e: e["identity"]["request_id"])
    assert [(e["identity"]["request_id"], e["identity"]["element_range"], e["element_offset"]) for e in ents] == [("r0", [0, 2 * H], 0), ("r1", [0, (TOKENS - 2) * H], 2 * H)]
    assert all(e["identity"]["op_path"] == "model" and e["identity"]["output_member"] == "out" for e in ents)
    for e, (lo, hi) in zip(ents, ((0, 2), (2, TOKENS))):
        pop = [B.identity(PD, e["identity"]["request_id"], 0, 0, "model", "out", 0, (hi - lo) * H)]
        assert B.coverage_check(pop, bmap)["ok"]
    # the bytes of request r1's block are the parent's value for ITS token rows
    e1 = ents[1]
    blk = _committed_bytes(c, e1)[2 * H * 2:]
    assert blk == _bytes_of(vals["scaled"][2:])
    assert np.frombuffer(blk, dtype=np.uint16).size == (TOKENS - 2) * H
