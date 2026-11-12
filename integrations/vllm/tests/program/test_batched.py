"""`prune.batched_form`, class G's batched form of a dead-free body: its runs, what stays a Call, the references it moves, that it
reads the body as written, and that the batched body computes what the flat one does."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits.defs import Builder, CompositeDefinition, bind
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.circuits.refs import Affine, Coll, Concat, PartLog, Strided, concat
from verity.ml.circuits.types import Array, Value
from verity.ml.definitions.registry import dense
from verity.ml.definitions.replay import replayer
from verity_vllm.commit import memory_map as MM
from verity_vllm.correspondence.runtime import SCHEMA, RuntimeCorrespondence
from verity_vllm.program import prune as P
from verity_vllm.program import tp_binding as TB

F32 = Value(32)
MUL = bind(dense.MulVecF32, N=8)


def _toy(fresh: bool = False):
    """Four rows of 8 times a shared w, then rows 0·1 and 2·3, then two reads of a peer Input; `fresh`: each Call's references
    new objects, equal to the shared ones."""
    B = Builder((("x", Array(32, F32)), ("w", Array(8, F32)), ("peers_0", Array(8, F32))), Array(32, F32))
    w = Coll(Array(8, F32), Affine("p", 1, 0, 8))
    rows = [B.call(MUL, Coll(Array(8, F32), Affine("p", 0, 8 * t, 8)), Coll(Array(8, F32), Affine("p", 1, 0, 8)) if fresh else w)
            for t in range(4)]
    pairs = [B.call(MUL, rows[0], rows[1]), B.call(MUL, rows[2], rows[3])]
    peer = Coll(Array(8, F32), Affine("p", 2, 0, 8))
    reads = [B.call(MUL, peer, w), B.call(MUL, peer, w)]
    return B.finish(Coll(Array(32, F32), concat([c.refs for c in pairs + reads])))


def _checked(body):
    """`body` re-emitted through a `Builder`, whose `call`, `batch` and `finish` check each node's argument types and widths."""
    B = Builder(body.params, body.ret_type)
    for n in body.nodes:
        B.batch(n.fn, *n.args, axes=n.axes, n=n.n) if n.form == "batch" else B.call(n.fn, *n.args)
    return B.finish(Coll(body.ret_type, body.ret))


def _evaluate(body, src: str, *inputs: np.ndarray) -> np.ndarray:
    d = CompositeDefinition("BatchedFormFixture", 1, ("SRC",), lambda S, _p=body.params, _r=body.ret_type: (tuple(_p), _r),
                            replayer(body), conformance="protocol-defined", register=False)
    sd = bind(d, SRC=src)
    sd.set_body(body)
    return np.asarray(evaluate_batch(sd, *(x[None].astype(np.uint64) for x in inputs))).reshape(-1)


def _written(body) -> list:
    return [[n.form, n.fn.id, list(n.axes), n.n, [a.refs.to_json() for a in n.args]] for n in body.nodes] + [body.ret.to_json()]


def _f32(rng, n: int) -> np.ndarray:
    return np.frombuffer(rng.uniform(-2, 2, n).astype(np.float32).tobytes(), dtype=np.uint32)


def test_a_run_of_independent_calls_of_one_definition_is_one_batch_and_computes_the_same():
    flat = _toy()
    form, sizes = P.batched_form(flat)
    assert sizes == [4, 2, 1, 1] and [n.form for n in form.nodes] == ["batch", "batch", "call", "call"]
    rows, pairs = form.nodes[:2]
    assert rows.axes == (0, None) and rows.out == Array(4, MUL.signature()[1])
    assert rows.args[0].refs.to_json() == Affine("p", 0, 0, 32).to_json() and rows.args[1].refs.to_json() == Affine("p", 1, 0, 8).to_json()
    # rows 0 and 2, then rows 1 and 3, of the first batch: every other member, one Strided each
    assert pairs.axes == (0, 0) and [a.refs.to_json() for a in pairs.args] == [
        Strided("n", 0, 0, 8, 16, 1, 16).to_json(), Strided("n", 0, 8, 8, 16, 1, 16).to_json()]
    assert form.ret.to_json() == concat([Affine("n", 1, 0, 16), Affine("n", 2, 0, 8), Affine("n", 3, 0, 8)]).to_json()
    _checked(form)
    rng = np.random.default_rng(5)
    x, w, peer = _f32(rng, 32), _f32(rng, 8), _f32(rng, 8)
    assert _evaluate(form, "batched", x, w, peer).tolist() == _evaluate(flat, "flat", x, w, peer).tolist()
    assert form.gates == flat.gates and P.batched_form(form)[0] is form


def test_the_form_reads_the_body_as_written_not_its_objects():
    """A shared argument is one every member reads alike, whether the Calls share one reference object or each has its own: the
    Build's derive and `predict.build` write equal bodies with different objects."""
    assert _written(P.batched_form(_toy(fresh=True))[0]) == _written(P.batched_form(_toy())[0])


def test_a_cut_of_a_partlog_stays_a_cut_of_one_log():
    B = Builder((("x", Array(32, F32)), ("w", Array(8, F32))), Array(32, F32))
    w, log = Coll(Array(8, F32), Affine("p", 1, 0, 8)), PartLog()
    for t in range(4):
        log.append(B.call(MUL, Coll(Array(8, F32), Affine("p", 0, 8 * t, 8)), w).refs)
    reads = [B.call(bind(dense.AddScalarF32, N=8 * (t + 1), C=1.0), Coll(Array(8 * (t + 1), F32), log.cut(t + 1))) for t in range(1, 4)]
    flat = B.finish(Coll(Array(32, F32), reads[-1].refs))
    form, sizes = P.batched_form(flat)
    assert sizes == [4, 1, 1, 1]
    cuts = [n.args[0].refs for n in form.nodes[1:]]
    assert all(isinstance(c, Concat) and c.prefix is not None and c.prefix[0] is cuts[0].prefix[0] for c in cuts)
    assert cuts[-1].to_json() == {"concat": [Affine("n", 0, 8 * m, 8).to_json() for m in range(4)]}
    _checked(form)
    rng = np.random.default_rng(6)
    x, wv = _f32(rng, 32), _f32(rng, 8)
    assert _evaluate(form, "batched-log", x, wv).tolist() == _evaluate(flat, "flat-log", x, wv).tolist()


def test_a_call_that_reads_the_run_ends_it():
    B = Builder((("x", Array(8, F32)),), Array(8, F32))
    y = B.call(MUL, Coll(Array(8, F32), Affine("p", 0, 0, 8)), Coll(Array(8, F32), Affine("p", 0, 0, 8)))
    z = B.call(MUL, y, y)
    form, sizes = P.batched_form(B.finish(B.call(MUL, z, y)))
    assert sizes == [1, 1, 1] and [n.form for n in form.nodes] == ["call"] * 3


def test_the_calls_the_root_readers_take_alone_stay_calls():
    """An attention's Call holds the KV `commit.memory_map` reads from one Call, and a collective reads its peer Inputs in one
    operand of one Call (`tp_binding.rank_sites`): the form keeps both. The tables are the readers' own."""
    assert (P.KV, P.LATENT, P.PEERS) == (MM.KV, MM.LATENT, TB.PEERS)
    assert P.batched_form(_toy())[1][2:] == [1, 1]
    sparse = type("Fn", (), {"name": "SparseMlaSplit", "signature": lambda self: ((("q", None), ("kv", None)), None)})()
    paged = type("Fn", (), {"name": "Attention", "signature": lambda self: ((("q", None), ("kc", None), ("vc", None)), None)})()
    assert P.holds_kv(sparse) and P.holds_kv(paged) and not P.holds_kv(MUL)


def test_the_batched_records_keep_each_members_record_and_its_first_hint():
    flat = _toy()
    calls = [{"occ": k} for k in range(len(flat.nodes))]
    form, recs, hints = P.kept_records(flat, calls, {0: "row0", 1: "row1", 4: "pair0", 6: "peer0"}, registered=P.BATCHED)
    assert [n.form for n in form.nodes] == ["batch", "batch", "call", "call"]
    assert recs == [{"members": calls[:4]}, {"members": calls[4:6]}, calls[6], calls[7]]
    assert hints == {0: "row0", 1: "pair0", 2: "peer0"}
    assert _written(P.dead_free(flat, batched=True)[0]) == _written(form)


def test_no_reader_takes_a_batchs_correspondence_record_yet():
    doc = {"schema": SCHEMA, "rank": None, "occurrences": [], "calls": [{"members": [{"occ": None}]}]}
    with pytest.raises(ValueError, match="a batch's"):
        list(RuntimeCorrespondence.records(doc))
