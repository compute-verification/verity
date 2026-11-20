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
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import torch
else:
    torch = pytest.importorskip("torch")

from verity_vllm.commit import binding as B
from verity_vllm.commit.committer.native_host import (
    NativeHostCommitter,
    TensorMeta,
    _dtype_name,
    _StepJob,
)
from verity_vllm.query.manifest import format as RM

V, H, TOKENS = 8, 16, 3
PD = hashlib.sha512(b"program").hexdigest()


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
                                staging="direct", geo_digest=hashlib.sha512(b"geo").digest())
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







