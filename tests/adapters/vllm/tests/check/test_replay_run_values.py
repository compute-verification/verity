"""The replay of a `run` field: in the relation form of a DSA row past INDEX_TOPK (2,048 keys), the row's order is a value the run
decides, the `run` field `<op_path>/topk_switches[t=P]` (a `run_values` identity), and the Program checks it.  No Program declares
`run` now (a row past INDEX_TOPK selects in the mask form, `predict.dsa`), so the tests drive it from the stand-in instances below.

A request's row of 2,050 keys flows end to end: its manifest (`query.required`) names the run value, the collector
(`commit.committer.run_values`) commits the served order's Waksman settings under that name, the binding map
(`commit.binding`, `query.manifest.bind.Binder`) binds them to the request, and the replay's index (`check.replay.index`, from the
instances header's `supplied.run`, as `pipeline.build_instances.write_instances` writes it) reads them as the committed operand of the
Call that routes them (`check.replay.evaluate`).  Every check the Program returns is evaluated from committed inputs and must be 1
(`check.replay.checks`): a served order that isn't the canonical one fails the replay, and the Commit verdict, even when every sampled
VU replays bit-exactly.

No replay row kernel is registered for DsaWaksmanOrder_v1 or DsaTopKLongRow_v1, so the tests install stand-ins built from the
registry's own references (`dsa_topk.waksman_apply` over the row indices, `dsa_topk.canonical_ok`)."""
from __future__ import annotations


import numpy as np
import pytest

from tests.check.test_commit_verdict_replay_seed_of_record import (
    BASE,
    RUN_ROOT,
)
from verity_vllm.check.commit_pass import commit_verdict
from verity_vllm.check.replay import checks as CK
from verity_vllm.definitions.numerics import rows as RK
from verity_vllm.definitions.registry.dsa_topk import (
    canonical_ok,
    waksman_apply,
    waksman_switches,
)
from verity_vllm.program.predict import dsa as D

N, K, LAYER, POS = 2050, 2048, 3, 2049
ATTN = f"model.layers.{LAYER}.self_attn.attn"
WRAP = "inner.model."
LEAVES = waksman_switches(N, K)
FIELD = D.run_field(LAYER, POS)
X, Y, LOGITS = f"model.layers.{LAYER}.self_attn.indexer.x", f"model.layers.{LAYER}.self_attn.indexer.y", "lm_head.logits"
RES = {"program": {"digest": "d" * 64, "function": "f"}, "target": {"declared": {"compute_capability": [9, 0]}}}


def _doc() -> dict:
    """One step of one request: the routed order (DsaWaksmanOrder_v1 on the `run` field), the row's scores (MulVecF32_v1 of two
    registered f32 vectors), the check on both (DsaTopKLongRow_v1, the Program's one check) and the sampled token."""
    rows = [(f"DsaWaksmanOrder_v1{{N={N},TOPK={K}}}", f"{WRAP}{ATTN}/waksman_order[0]", [[["p", 0, 0, LEAVES]]]),
            (f"MulVecF32_v1{{N={N}}}", f"{WRAP}model.layers.{LAYER}.self_attn.indexer/mul[0]", [[["p", 1, 0, N]], [["p", 1, N, 2 * N]]]),
            (f"DsaTopKLongRow_v1{{N={N},TOPK={K}}}", f"{WRAP}{ATTN}/topk_long_row[0]", [[["n", 1, 0, N]], [["n", 0, 0, K]]]),
            ("TokenSelect_v1{V=4}", "token_select[0]", [[["p", 1, 2 * N, 2 * N + 4]]])]
    run = {"param": 0, "name": "run", "fields": [{"name": FIELD, "type": f"Array<{LEAVES},Value<1>>", "offset": 0, "leaves": LEAVES}]}
    return {"schema": "cb-a/instance-sequence/v2", "digest": "d" * 64, "correspondence_digest": None,
            "params": [["run", f"{{{FIELD}: Array<{LEAVES},Value<1>>}}"],
                       ["weights", f"{{{X}: Array<{N},Value<32>>, {Y}: Array<{N},Value<32>>, {LOGITS}: Array<4,Value<16>>}}"]],
            "supplied": {"run": [run]}, "checks": {"output": 1, "accept": 1, "of": [f"{ATTN} t={POS}: canonical"], "at": [[2, 0, 1]]},
            "rows": [{"i": i, "spec": s, "form": "call", "name": n, "args": a} for i, (s, n, a) in enumerate(rows)]}




@pytest.fixture
def stand_ins(monkeypatch):
    def order(st, ops):
        return {"out": np.array(waksman_apply([int(b) for b in ops[0]], list(range(int(st["N"]))), int(st["TOPK"])), dtype=np.uint32)}

    def check(st, ops):
        return {"out": np.array([canonical_ok(ops[0], ops[1], int(st["TOPK"]))], dtype=np.uint8)}

    for fam, fn in (("DsaWaksmanOrder_v1", order), ("DsaTopKLongRow_v1", check)):
        monkeypatch.setitem(RK.ROWS, fam, RK.RowKernel(fam, fn, ("out",)))
        monkeypatch.setitem(RK.EVALUATORS, fam, "the registry's reference (test stand-in)")




def _verdict(rec: dict) -> tuple[bool, str]:
    return commit_verdict({**BASE, "run_root": RUN_ROOT, "sampled_replay": rec}, True, min_openings=64)










def test_fold_keeps_a_failure_and_never_lifts_a_partial():
    ok = {"result": True, "why": None}
    refused, unevaluated = {"result": False, "why": "c0"}, {"result": None, "why": "c1"}
    assert CK.fold(True, "complete", None, None) == CK.fold(True, "complete", None, ok) == (True, "complete", None)
    assert CK.fold(True, "complete", None, refused)[:2] == CK.fold(None, "partial", "p", refused)[:2] == (False, "mismatch")
    assert CK.fold(False, "mismatch", "m", refused) == (False, "mismatch", "m")
    assert CK.fold(True, "complete", None, unevaluated)[:2] == (None, "partial")
    assert CK.fold(None, "not-evaluated", "n", unevaluated) == (None, "not-evaluated", "n")
