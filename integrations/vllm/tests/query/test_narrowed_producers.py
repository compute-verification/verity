"""A narrowed linear's two producers (`program.prune.narrowed`): on one rank the request wrapper computes its last layer's fused qkv
on the prompt rows before the last at the keys' and values' columns only, and on the last row whole, so one fx node of the
`qkv_proj` module produces rows of two widths at the prompt step.  The manifest models both (`Population.narrowed`): the whole
rows keep `out` and the narrowed rows are `out@<width>`.  Every reader of the population addresses them alike: the replay's index,
its reconciliation against the manifest (which names a manifest lacking either producer), and the Commit's binder (which binds the
module's returned tensor to the whole rows only)."""
from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from verity_vllm.check.replay import coverage as COV, index as IX, population as RP  # noqa: E402
from verity_vllm.correspondence.reader_for_query import Correspondence  # noqa: E402
from verity.ml.definitions.registry import b1  # noqa: E402
from verity_vllm.query.manifest.bind import Binder, coverage_check  # noqa: E402
from verity_vllm.query.program_view import from_instances  # noqa: E402
from verity_vllm.query.required import request_manifest  # noqa: E402
from tests.query.test_boolean_manifest import _write  # noqa: E402

C = {**b1.B0_CONFIG, "H": 64, "NH": 4, "KVH": 2, "D": 16, "I": 128, "QKV": 128, "GU": 256, "NL": 2, "V": 32, "MAX_POS": 64}
LP = 4
LAST_QKV = "lm.model.layers.1.qkv_proj"


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    """A Llama-shaped greedy serve of LP prompt rows and one decode step, derived in the registered compiler's form (`dead_free`)
    and written as the Build writes a request Program dir; its manifest and instance sequence."""
    from verity_vllm.program.frontend import derive_examples as ex
    from verity_vllm.program.frontend.derive import derive_report

    with torch.device("meta"):
        m = ex.GreedyServe(C, 1)
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
