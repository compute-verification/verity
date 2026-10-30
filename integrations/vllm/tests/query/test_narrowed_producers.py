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
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

try:
    import torch
except ModuleNotFoundError:
    pytest.skip("could not import 'torch'", allow_module_level=True)

from verity_vllm.acquire.sources import narrowed_source as NS
from verity_vllm.check.replay import coverage as COV, index as IX, population as RP
from verity_vllm.check.replay.evaluate import evaluate
from verity_vllm.commit.binding import build_binding_map
from verity_vllm.correspondence.reader_for_query import Correspondence
from verity.ml.definitions.registry import b1
from verity_vllm.query.manifest.bind import Binder, coverage_check
from verity_vllm.query.program_view import from_instances
from verity_vllm.query.required import request_manifest
from tests.query.test_boolean_manifest import _write

C = {**b1.B0_CONFIG, "H": 64, "NH": 4, "KVH": 2, "D": 16, "I": 128, "QKV": 128, "GU": 256, "NL": 2, "V": 32, "MAX_POS": 64}
LP = 4
LAST_QKV = "lm.model.layers.1.qkv_proj"


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    """A Llama-shaped greedy serve of LP prompt rows and one decode step under vLLM's row policy (the final norm over every prompt
    row, the LM head over the last), derived in the registered compiler's form (`dead_free`) and written as the Build writes a request
    Program dir; its manifest and instance sequence."""
    from verity_vllm.program.frontend import derive_examples as ex
    from verity_vllm.program.frontend.derive import derive_report

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


def test_the_replay_addresses_each_producer_at_its_own_identity_and_names_a_manifest_missing_either(build):
    """The replay's index reads the members off the same population: the narrowed rows at `out@64`, the whole row at `out`.  Its
    reconciliation passes against the manifest, and refuses one without either producer, naming that producer's VU."""
    d, man, inst = build
    addrs: dict[tuple, dict] = {}
    RP.population(IX.ProgramIndex(inst, program_dir=str(d)), "r0", addresses=addrs)
    assert {k[3]: (a["rows"], a["spec"]) for k, a in addrs.items() if k[:3] == ("r0", 0, LAST_QKV)} == {
        "out": (1, "Gemm_v1{K=64,N=128}"), "out@64": (LP - 1, "Gemm_v1{K=64,N=64}")}
    checks = COV.query_population(man, addrs, set(), ranks=[0])["query_checks"]
    assert checks["ok"] is True, checks
    for member in ("out", "out@64"):
        checks = COV.query_population(_without(man, 0, member), addrs, set(), ranks=[0])["query_checks"]
        assert checks["ok"] is False
        assert [(v["op_path"], v["member"]) for v in checks["vus_outside_query"]["first"]] == [(LAST_QKV, member)]


def test_the_modules_returned_tensor_binds_the_whole_rows_and_only_a_label_binds_the_narrowed(build):
    """The tuple slot of `qkv_proj`'s return binds `out`, never the narrowed rows; a committed tensor labelled `out@64` binds
    those.  With the returned tensor alone, the Commit's coverage names the narrowed rows missing, and only them."""
    _d, man, _inst = build
    binder = Binder.of_manifest(man)
    assert binder.bind(0, [LAST_QKV]) == [(LAST_QKV, "out", 0, "slot")]
    assert binder.bind(0, [LAST_QKV, f"{LAST_QKV}/out@64"]) == [(LAST_QKV, "out", 0, "slot"), (LAST_QKV, "out@64", 0, "label")]
    cov = coverage_check(man, {0: [{"name": LAST_QKV, "shape": [LP, 128]}]}, request_id="r0", max_report=10**6)
    assert [(m["op_path"], m["output_member"]) for m in cov["missing"] if m["op_path"] == LAST_QKV and m["step"] == 0] == [(LAST_QKV, "out@64")]


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
        self.model, self.name, self.errors, self.sources, self.acquired = model, "rec", [], [], []
        self._cur, self._plan_shape = None, None

    def attach_source(self, src: object) -> None:
        self.sources.append(src)

    def _acquire(self, name: str, t: torch.Tensor, _job: object) -> None:
        self.acquired.append((name, t.contiguous().clone()))


def _bf16(a: np.ndarray) -> np.ndarray:
    return (a.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _gemm(x: np.ndarray, w: np.ndarray, spec: str) -> np.ndarray:
    """A `Gemm_v1` Call's Values on token row `x` and weight rows `w`: its registered row kernel."""
    return evaluate("Gemm_v1", {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", spec)}, [[x], []], [None, w.reshape(-1)])["out"]


def _metas(acquired: list) -> list:
    metas, off = [], 0
    for i, (name, t) in enumerate(acquired):
        nb = t.numel() * t.element_size()
        metas.append(SimpleNamespace(name=name, shape=tuple(t.shape), chunk=256, stream_off=off, first_leaf=off // 256, nbytes=nb,
                                     dtype="bf16", ordinal=i))
        off += -(-nb // 256) * 256
    return metas


def test_the_committed_words_of_the_narrowed_and_the_last_rows_are_the_programs_values(build):
    """The fused GEMM's whole rows at the prompt step, through the committer's hook and the narrowed-rows source, bound by the manifest's
    binder in the binding map: the words each step-0 identity of the last layer's qkv reads are the Values its Program Calls compute,
    each from the weight rows the Call reads (the narrowed Calls the keys' and values' rows).  Row 0's words are not the last row's: the
    whole rows' identity is bound at its runtime row."""
    d, man, inst = build
    pi = IX.ProgramIndex(inst, program_dir=str(d))
    rng = np.random.default_rng(7)
    x, W = _bf16(rng.standard_normal((LP, 64))), _bf16(rng.standard_normal((128, 64)) * 0.125)
    calls = [r for r in inst["rows"] if str(r.get("name", "")).startswith(f"{LAST_QKV}/")][:LP]
    assert [r["spec"] for r in calls] == ["Gemm_v1{K=64,N=64}"] * (LP - 1) + ["Gemm_v1{K=64,N=128}"]
    program = []
    for t, r in enumerate(calls):
        (_p, _param, lo, hi), = r["args"][1]
        field, flo, fhi, _shape = pi.weight_slice(lo, hi)
        assert field == f"{LAST_QKV}.weight"
        program.append(_gemm(x[t], W.reshape(-1)[flo:fhi], r["spec"]))
    whole = np.stack([_gemm(x[t], W, "Gemm_v1{K=64,N=128}") for t in range(LP)])

    root = torch.nn.Module()
    root.lm = torch.nn.Module()
    root.lm.model = torch.nn.Module()
    root.lm.model.layers = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])
    root.lm.model.layers[1].qkv_proj = _Qkv(torch.from_numpy(whole.view(np.int16).copy()).view(torch.bfloat16))
    com = _Committer(root)
    src = NS.attach(com, man)
    assert com.sources == [src] and src.own_names == {f"{LAST_QKV}/out@64"}
    src.install()
    com._cur, com._plan_shape = SimpleNamespace(step=0), (LP, 1)
    out, _bias = root.lm.model.layers[1].qkv_proj(None)
    com.acquired.insert(0, (LAST_QKV, out.clone()))
    assert [n for n, _t in com.acquired] == [LAST_QKV, f"{LAST_QKV}/out@64"] and not com.errors

    bmap = build_binding_map({0: _metas(com.acquired)}, [], bytes(64), man["program_digest"], binder=Binder.of_manifest(man))
    words = {n: t.view(torch.int16).numpy().view(np.uint16).reshape(-1) for n, t in com.acquired}
    entry = {e["identity"]["output_member"]: e for e in bmap["entries"] if e["identity"]["op_path"] == LAST_QKV}
    want = {r["output_member"]: r["element_range"] for r in man["identities"] if r["op_path"] == LAST_QKV and r["step"] == 0}
    assert set(want) == set(entry) == {"out", "out@64"}
    got = {}
    for member, (lo, hi) in want.items():
        e = entry[member]
        elo, ehi = e["identity"]["element_range"]
        assert "refused" not in e and elo <= lo and hi <= ehi
        got[member] = words[e["name"]][e["element_offset"] + lo:e["element_offset"] + hi]
    assert (entry["out"]["element_offset"], entry["out@64"]["element_offset"]) == ((LP - 1) * 128, 0)
    assert got["out@64"].tolist() == np.concatenate(program[:LP - 1]).tolist()
    assert got["out"].tolist() == program[LP - 1].tolist()
    assert words[LAST_QKV][:128].tolist() != program[LP - 1].tolist()


def test_a_share_without_the_runtime_row_is_refused_and_coverage_names_it(build):
    """A committed qkv of one row (a committer that kept only the identity's rows) has no row LP - 1: the binding map refuses its
    entry by name, and coverage names the whole rows missing while the narrowed rows stay covered."""
    _d, man, _inst = build
    metas = [SimpleNamespace(name=LAST_QKV, shape=(1, 128), chunk=256, stream_off=0, first_leaf=0, nbytes=256, dtype="bf16", ordinal=0)]
    bmap = build_binding_map({0: metas}, [], bytes(64), man["program_digest"], binder=Binder.of_manifest(man))
    assert [e.get("refused") for e in bmap["entries"]] == [
        f"{LAST_QKV}: r0's values start at its row {LP - 1}; its share is rows [0, 1): bound to no row"]
    layouts = {0: [{"name": LAST_QKV, "shape": [1, 128]}, {"name": f"{LAST_QKV}/out@64", "shape": [LP, 64]}]}
    cov = coverage_check(man, layouts, request_id="r0", max_report=10**6)
    assert [(m["op_path"], m["output_member"]) for m in cov["missing"] if m["op_path"] == LAST_QKV and m["step"] == 0] == [(LAST_QKV, "out")]
    layouts[0][0]["shape"] = [LP, 128]
    cov = coverage_check(man, layouts, request_id="r0", max_report=10**6)
    assert not [m for m in cov["missing"] if m["op_path"] == LAST_QKV and m["step"] == 0]
