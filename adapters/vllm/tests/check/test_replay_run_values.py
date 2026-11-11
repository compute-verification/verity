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

import types

import numpy as np
import pytest
import torch

from tests.check.test_commit_verdict_replay_seed_of_record import BASE, ROOT_FORM, RUN_ROOT, SEED_ROOT
from verity.definitions.numerics import rows as RK
from verity.definitions.registry.dsa_topk import canonical_ok, topk_set, waksman_apply, waksman_settings, waksman_switches
from verity_vllm.check.commit_pass import commit_verdict
from verity_vllm.check.replay import checks as CK, driver as DR, index as IX
from verity_vllm.check.replay.coverage import QUERY_FAMILY_MECHANISM
from verity_vllm.commit import binding as B
from verity_vllm.commit.committer.native_host import GPU_CHUNK_BYTES, TensorMeta
from verity_vllm.commit.committer.run_values import RunValues
from verity_vllm.correspondence.reader_for_query import Correspondence
from verity_vllm.program.predict import dsa as D
from verity_vllm.query.manifest.bind import Binder
from verity_vllm.query.manifest.format import RUN_FAMILY
from verity_vllm.query.program_view import from_instances
from verity_vllm.query.required import request_manifest

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


class Row:
    """The fixture's request: its manifest, registered weights and the served row's scores, canonical order and committed values."""

    def __init__(self):
        self.doc = _doc()
        P = from_instances(self.doc)
        self.manifest = request_manifest(P, Correspondence.of(P, None, implementation_paths=[ATTN]), RES, request_id="r0").manifest
        rng = np.random.default_rng(7)
        x, y = rng.standard_normal(N).astype(np.float32), rng.standard_normal(N).astype(np.float32)
        self.scores = (x * y).view(np.uint32)
        self.weights = {X: x.view(np.uint32), Y: y.view(np.uint32), LOGITS: np.array([0x3F80, 0x4000, 0x3F00, 0], dtype=np.uint16)}
        self.canonical = sorted(topk_set([int(s) for s in self.scores], K))

    def name(self, family: str) -> str:
        r = next(r for r in self.manifest["identities"] if r["family"] == family)
        return f"{r['op_path']}/{r['output_member']}"

    def collected(self, order: list[int]) -> tuple[str, np.ndarray]:
        """What the collector commits of the served `order` at the row's step: the run value's name and bits."""
        rv = RunValues([r for r in self.manifest["identities"] if r["family"] == RUN_FAMILY], {f"model.layers.{LAYER}.self_attn": LAYER})
        rv.begin_step()
        rv.take(0, LAYER, torch.tensor([POS]), torch.tensor([order + [-1, -1]], dtype=torch.int32), 1)
        staged: list = []
        assert rv.commit(lambda n, t: staged.append((n, t))) == []
        ((name, bits),) = staged
        return name, bits.numpy()

    def committed(self, order: list[int], run_bits: np.ndarray | None = None) -> tuple[dict, object]:
        """The binding map and committed reader of a step that served `order`: the collector's run bits (or `run_bits`), the routed
        order and the scores, each bound under the manifest's name, and the sampled token."""
        run, bits = self.collected(order)
        tensors = [(run, bits if run_bits is None else np.asarray(run_bits, dtype=np.uint8)),
                   (self.name("call_boundaries"), np.array(order, dtype=np.uint32)),
                   (self.name("instance_outputs"), self.scores), (self.name("sampled_token_ids"), np.array([1], dtype=np.int64))]
        metas, off = [], 0
        for j, (n, t) in enumerate(tensors):
            m = TensorMeta(j, n, str(t.dtype), t.shape, t.nbytes, GPU_CHUNK_BYTES, off // GPU_CHUNK_BYTES, 1)
            m.stream_off, off = off, off + t.nbytes
            metas.append(m)
        bmap = B.build_binding_map({0: metas}, [types.SimpleNamespace(root=b"\1" * 32)], b"\0" * 32, "e" * 64,
                                   binder=Binder(self.manifest["identities"], rank=0))
        blobs = {n: t.tobytes() for n, t in tensors}
        return bmap, lambda step, name, b0, b1: blobs[name][b0:b1] if step == 0 and name in blobs else None

    def replay(self, order: list[int], run_bits: np.ndarray | None = None) -> dict:
        bmap, committed = self.committed(order, run_bits)
        return DR.sampled_replay({"r0": self.doc}, bmap, committed, lambda f, lo, hi: self.weights[f][lo:hi] if f in self.weights else None,
                                 arrivals={"r0": 0}, seed=SEED_ROOT, seed_source=ROOT_FORM, attention_paths=[ATTN], workers=1,
                                 manifest=self.manifest)


@pytest.fixture
def stand_ins(monkeypatch):
    def order(st, ops):
        return {"out": np.array(waksman_apply([int(b) for b in ops[0]], list(range(int(st["N"]))), int(st["TOPK"])), dtype=np.uint32)}

    def check(st, ops):
        return {"out": np.array([canonical_ok(ops[0], ops[1], int(st["TOPK"]))], dtype=np.uint8)}

    for fam, fn in (("DsaWaksmanOrder_v1", order), ("DsaTopKLongRow_v1", check)):
        monkeypatch.setitem(RK.ROWS, fam, RK.RowKernel(fam, fn, ("out",)))
        monkeypatch.setitem(RK.EVALUATORS, fam, "the registry's reference (test stand-in)")


@pytest.fixture(scope="module")
def row() -> Row:
    return Row()


def _verdict(rec: dict) -> tuple[bool, str]:
    return commit_verdict({**BASE, "run_root": RUN_ROOT, "sampled_replay": rec}, True, min_openings=64)


def test_the_replay_reads_the_run_values_the_collector_committed(row, stand_ins):
    """The manifest's run value is the collector's name, the binding map binds it to the request at the reader's step, and the replay
    routes the committed bits to the served order: every sampled VU replays, the check is 1, coverage names `run_values`' mechanism
    and the Commit passes."""
    (run,) = [r for r in row.manifest["identities"] if r["family"] == RUN_FAMILY]
    assert (run["op_path"], run["output_member"], run["element_range"]) == (ATTN, f"topk_switches[t={POS}]", [0, LEAVES])
    name, bits = row.collected(row.canonical)
    assert name == FIELD and bits.size == LEAVES and waksman_apply(list(bits), list(range(N)), K) == row.canonical
    bmap, _ = row.committed(row.canonical)
    bound = next(e["identity"] for e in bmap["entries"] if e["name"] == FIELD)
    assert (bound["request_id"], bound["step"], bound["op_path"], bound["output_member"], bound["element_range"]) == \
        ("r0", 0, ATTN, f"topk_switches[t={POS}]", [0, LEAVES])
    pi = IX.ProgramIndex(row.doc, attention_paths=[ATTN])
    f = pi.supplied_field(0, 0, LEAVES)
    assert (f["op_path"], f["member"], f["leaves"]) == (ATTN, f"topk_switches[t={POS}]", LEAVES)
    assert pi.supplied_field(0, 1, 7) is f and pi.supplied_field(0, 0, LEAVES + 1) is None

    rec = row.replay(row.canonical)
    assert (rec["result"], rec["grade"]) == (True, "complete")
    assert {k: rec["sample"][k] for k in ("picked", "evaluated", "equal")} == {"picked": 3, "evaluated": 3, "equal": 3}
    assert {k: rec["checks"][k] for k in ("result", "n", "accepted", "refused_n", "not_evaluated_n")} == \
        {"result": True, "n": 1, "accepted": 1, "refused_n": 0, "not_evaluated_n": 0}
    assert RUN_FAMILY in QUERY_FAMILY_MECHANISM
    assert {u["family"]: u["class"] for u in rec["population"]["uncovered_by_name"]}[RUN_FAMILY] == "mechanism"
    assert rec["population"]["query_checks"]["unstated_families"] == []
    ok, why = _verdict(rec)
    assert ok is True and why.startswith("PASS (") and "uncovered by name: run_values x1 (mechanism)" in why


def test_the_committed_run_values_are_what_the_replay_routes(row, stand_ins):
    """One flipped committed run bit routes another order than the committed one: the order's VU fails by name, on the run value."""
    flipped = row.collected(row.canonical)[1].copy()
    flipped[0] ^= 1
    assert waksman_apply(list(flipped), list(range(N)), K) != row.canonical
    rec = row.replay(row.canonical, flipped)
    assert rec["result"] is False and rec["sample"]["mismatch_n"] == 1
    (m,) = rec["mismatch"]
    assert m["spec"].startswith("DsaWaksmanOrder_v1") and m["inputs"] == [{"kind": "committed", "identity": ["r0", 0, ATTN, f"topk_switches[t={POS}]", [0, LEAVES]]}]


def test_a_check_that_is_not_one_refuses_the_replay_and_the_commit(row, stand_ins):
    """A served order of the top-2,048 set that isn't in ascending index order, committed honestly (its settings, its routed order):
    every sampled VU replays bit-exactly and the check is 0, so the replay fails by name, and the Commit with it."""
    served = row.canonical[1:] + row.canonical[:1]
    assert waksman_apply(waksman_settings(served, N), list(range(N)), K) == served
    rec = row.replay(served)
    assert rec["sample"]["equal"] == rec["sample"]["evaluated"] == 3
    assert (rec["result"], rec["grade"]) == (False, "mismatch")
    assert rec["why"] == f"a check of the Program is refused: check is 0, not 1: {ATTN} t={POS}: canonical (r0, step 0, row 2)"
    assert (rec["checks"]["refused_n"], rec["checks"]["refused"][0]["value"]) == (1, 0)
    ok, why = _verdict(rec)
    assert ok is False and why.startswith("C2 sampled replay: a check of the Program is refused")


def test_a_check_the_replay_cannot_evaluate_leaves_it_partial(row, stand_ins, monkeypatch):
    monkeypatch.delitem(RK.ROWS, "DsaTopKLongRow_v1")
    rec = row.replay(row.canonical)
    assert (rec["result"], rec["grade"]) == (None, "partial") and rec["checks"]["not_evaluated_n"] == 1
    assert rec["why"].startswith("a check of the Program could not be evaluated: evaluator error: KeyError('DsaTopKLongRow_v1')")


def test_fold_keeps_a_failure_and_never_lifts_a_partial():
    ok = {"result": True, "why": None}
    refused, unevaluated = {"result": False, "why": "c0"}, {"result": None, "why": "c1"}
    assert CK.fold(True, "complete", None, None) == CK.fold(True, "complete", None, ok) == (True, "complete", None)
    assert CK.fold(True, "complete", None, refused)[:2] == CK.fold(None, "partial", "p", refused)[:2] == (False, "mismatch")
    assert CK.fold(False, "mismatch", "m", refused) == (False, "mismatch", "m")
    assert CK.fold(True, "complete", None, unevaluated)[:2] == (None, "partial")
    assert CK.fold(None, "not-evaluated", "n", unevaluated) == (None, "not-evaluated", "n")
