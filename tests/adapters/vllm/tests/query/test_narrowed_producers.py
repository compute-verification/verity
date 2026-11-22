"""A narrowed linear's two producers (`program.prune.narrowed`): on one rank the request wrapper computes its last layer's fused qkv
on the prompt rows before the last at the keys' and values' columns only, and on the last row whole, so one fx node of the
`qkv_proj` module produces rows of two widths at the prompt step.  The manifest models both (`Population.narrowed`): the whole
rows keep `out` and the narrowed rows are `out@<width>`.  Every reader of the population addresses them alike: the replay's index,
its reconciliation against the manifest (which names a manifest lacking either producer), and the Commit's binder (which binds the
module's returned tensor to the whole rows only).  The rows `dead_free` removed (the last layer's prompt rows before the last, but for
the narrowed columns) stay in the runtime tensor: the manifest says where its rows start (`runtime_rows`) and the binding map binds
the identity there, so the Commit's words of every identity are the Program's values."""
from __future__ import annotations

import copy
import gzip
import json
from types import SimpleNamespace

import numpy as np
import pytest

try:
    import torch
except ModuleNotFoundError:
    pytest.skip("could not import 'torch'", allow_module_level=True)

from tests.query.test_boolean_manifest import _write
from verity_vllm.correspondence.reader_for_query import Correspondence
from verity_vllm.definitions.registry import b1
from verity_vllm.query.program_view import from_instances
from verity_vllm.query.required import request_manifest

C = {**b1.B0_CONFIG, "H": 64, "NH": 4, "KVH": 2, "D": 16, "I": 128, "QKV": 128, "GU": 256, "NL": 2, "V": 32, "MAX_POS": 64}
LP = 4
LAST_QKV = "lm.model.layers.1.qkv_proj"


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    """A Llama-shaped greedy serve of LP prompt rows and one decode step under vLLM's row policy (the final norm over every prompt
    row, the LM head over the last), derived in the registered compiler's form (`dead_free`) and written as the Build writes a request
    Program dir; its manifest and instance sequence."""
    from verity_vllm.program.frontend import derive_examples as ex
    from verity_vllm.program.frontend.vllm_derive import derive_report

    with torch.device("meta"):
        m = ex.GreedyServe(C, 1, vllm_rows=True)
    rep = derive_report(m, (torch.empty(LP, dtype=torch.int32, device="meta"),), mode="profile", profile="b1-eager", config=C,
                        name="Serve", dead_free=True)
    assert rep.ok, rep.unsupported
    d = tmp_path_factory.mktemp("request")
    _write(d, rep)
    P = from_instances(str(d / "instances.json.gz"))
    corr = Correspondence.of(P, str(d))
    man = request_manifest(P, corr, json.loads((d / "result.json").read_text()), request_id="r0", program_dir=str(d)).manifest
    with gzip.open(d / "instances.json.gz", "rt") as f:
        inst = json.load(f)
    return d, man, inst


def _rows(man: dict, op: str) -> dict[tuple, tuple]:
    return {(r["step"], r["output_member"]): (r["rows"], r["width"], r["element_range"], r["spec"], r.get("narrowed_of"))
            for r in man["identities"] if r["op_path"] == op}


def _without(man: dict, step: int, member: str) -> dict:
    out = copy.deepcopy(man)
    out["identities"] = [r for r in out["identities"] if (r["op_path"], r["step"], r["output_member"]) != (LAST_QKV, step, member)]
    return out


def test_the_manifest_models_the_narrowed_qkv_and_the_whole_last_row_as_two_identities(build):
    """The last layer's qkv at the prompt step: the last row whole (128 columns) under `out`, the LP - 1 rows before it at their keys'
    and values' 64 columns under `out@64`; a decode step's one row and the first layer's rows are whole, under `out` alone."""
    _d, man, _inst = build
    assert man["complete"] is True and man["unmodelled"] == {}
    assert _rows(man, LAST_QKV) == {(0, "out"): (1, 128, [0, 128], "Gemm_v1", None),
                                    (0, "out@64"): (LP - 1, 64, [0, (LP - 1) * 64], "Gemm_v1", "out"),
                                    (1, "out"): (1, 128, [0, 128], "Gemm_v1", None)}
    assert _rows(man, "lm.model.layers.0.qkv_proj") == {(0, "out"): (LP, 128, [0, LP * 128], "Gemm_v1", None),
                                                        (1, "out"): (1, 128, [0, 128], "Gemm_v1", None)}
    assert not [r for r in man["identities"] if "@" in r["output_member"] and r["op_path"] != LAST_QKV]


def test_the_manifest_places_the_last_layers_prompt_identities_at_the_last_runtime_row(build):
    """`dead_free` leaves the last layer's prompt step (and the final norm, which vLLM runs over every prompt row) one row, the last:
    those identities carry `runtime_rows` [LP - 1, LP], a decode step's and the LM head's (vLLM gathers the sampled row first) none,
    and the narrowed rows the columns of the whole rows they are."""
    _d, man, _inst = build
    module = {(canon, k): alias for alias, (canon, k) in man["op_path_aliases"].items()}
    placed = {(module.get((r["op_path"], r["invocation"]), r["op_path"]), r["output_member"]): tuple(r["runtime_rows"])
              for r in man["identities"] if r.get("runtime_rows")}
    assert {r["step"] for r in man["identities"] if r.get("runtime_rows")} == {0}
    assert set(placed.values()) == {(LP - 1, LP)}
    assert {(LAST_QKV, "out"), ("lm.model.layers.1.rotary_emb", "query"), ("lm.model.norm", "0"), ("lm.model.norm", "1")} <= set(placed)
    assert all(op.startswith("lm.model.layers.1.") or op == "lm.model.norm" for op, _m in placed)
    assert {(r["op_path"], r["output_member"]): r.get("columns") for r in man["identities"] if r.get("narrowed_of")} == {
        (LAST_QKV, "out@64"): [64, 128]}


def test_the_last_layers_shared_rope_is_the_shared_modules_last_invocation(build):
    """vLLM registers one rope module for every layer (`get_rope`), under the first layer.  `dead_free` leaves the last layer's
    rope its prompt queries' last row only, which changes how many Calls its body has but not the body: the manifest still
    addresses it as the shared module's last invocation, never at a path the runtime tree has no module for."""
    _d, man, _inst = build
    assert man["op_path_aliases"] == {"lm.model.layers.1.rotary_emb": ["lm.model.layers.0.rotary_emb", 1]}
    assert not [r for r in man["identities"] if r["op_path"].startswith("lm.model.layers.1.rotary_emb")]
    assert {(r["step"], r["output_member"]): (r["element_range"], r.get("runtime_rows")) for r in man["identities"]
            if r["op_path"] == "lm.model.layers.0.rotary_emb" and r["invocation"] == 1} == {
        (0, "query"): ([0, 4 * 16], [LP - 1, LP]), (0, "key"): ([0, LP * 2 * 16], None),
        (1, "query"): ([0, 4 * 16], None), (1, "key"): ([0, 2 * 16], None)}


class _Qkv(torch.nn.Module):
    """vLLM's fused qkv linear: (the output, its bias) for the rows it is given."""

    def __init__(self, out: torch.Tensor) -> None:
        super().__init__()
        self.out = out

    def forward(self, _x: object) -> tuple:
        return self.out, None


class _Committer:
    """A committer's hook as the source sees it: the module's returned tensor first, then what the sources acquire."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model, self.name = model, "rec"
        self.errors: list[str] = []
        self.sources: list[object] = []
        self.acquired: list[tuple[str, torch.Tensor]] = []
        self._cur: SimpleNamespace | None = None
        self._plan_shape: tuple[int, int] | None = None

    def attach_source(self, src: object) -> None:
        self.sources.append(src)

    def _acquire(self, name: str, t: torch.Tensor, _job: object) -> None:
        self.acquired.append((name, t.contiguous().clone()))


def _bf16(a: np.ndarray) -> np.ndarray:
    return (a.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _metas(acquired: list) -> list:
    metas, off = [], 0
    for i, (name, t) in enumerate(acquired):
        nb = t.numel() * t.element_size()
        metas.append(SimpleNamespace(name=name, shape=tuple(t.shape), chunk=256, stream_off=off, first_leaf=off // 256, nbytes=nb,
                                     dtype="bf16", ordinal=i))
        off += -(-nb // 256) * 256
    return metas




def _plan(build, tmp_path) -> list[dict]:
    """The seal's capture plan of the fixture's one request Program (`capture_plan.rows_of_build`)."""
    from verity_vllm.check.capture_plan import rows_of_build

    d, _man, _inst = build
    (tmp_path / "build_request_r0").symlink_to(d)
    return rows_of_build(tmp_path)


def test_the_seals_binder_binds_a_narrowed_linears_return_to_its_whole_rows_at_every_step(build, tmp_path):
    """The seal binds each step's events with the capture plan's members (`serve_seal.served_binder_of`), which name the last
    layer's qkv `out` and `out@64` at one op path: its returned tensor binds `out`, never the empty member of an unbound slot,
    and every module's return binds a member the plan names."""
    from verity_vllm.pipeline.serve_seal import served_binder_of

    _d, man, _inst = build
    plan = _plan(build, tmp_path)
    members = {str(r["op_path"]): [str(m[0]) for m in r["members"]] for r in plan}
    assert members[LAST_QKV] == ["out@64", "out"] or members[LAST_QKV] == ["out", "out@64"]
    binder = served_binder_of(plan, [0, 1], {}, False, man["op_path_aliases"])
    returns = [op for op, ms in members.items() if "/" not in op and len([m for m in ms if "@" not in m]) == 1]
    for step in (0, 1):
        got = binder.bind(step, returns)
        assert not [g for g in got if g[3] == "unbound"], [g for g in got if g[3] == "unbound"]
        assert got[returns.index(LAST_QKV)] == (LAST_QKV, "out", 0, "slot")


def test_a_narrowed_linears_slot_that_binds_nothing_is_refused_by_name():
    """A capture plan that names a narrowed linear's narrowed rows and not its whole rows leaves its returned tensor nothing to
    bind: the binder refuses it by module, invocation and slot, where it would otherwise bind the empty member."""
    from verity_vllm.pipeline.serve_seal import served_binder_of

    plan = [{"op_path": LAST_QKV, "members": [("out@64", 64, "bf16")]}]
    with pytest.raises(ValueError, match=rf"step 0: {LAST_QKV} at invocation 0, slot 0: a narrowed linear's slot binds none"):
        served_binder_of(plan, [0], {}, False).bind(0, [LAST_QKV])
