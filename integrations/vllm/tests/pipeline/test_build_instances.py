"""The Build's instance sequence (`instances.json.gz`) is written a chunk of rows at a time with `json.dumps`, and its text is
`json.dump`'s of the whole document, whatever the chunk size and the file's gzip level.  Its rows store each argument in the
progressions form (instances v2, `compact.file_items_from_refs`: O(rows) for a scattered Strided where the v1 per-leaf `runs()`
is O(leaves)), and every reader -- `iter_instance_rows`, `from_instances`, `manifest_format.load_program` -- expands them back to
run lists bit-identical to what the v1 writer wrote."""
from __future__ import annotations

import gzip
import io
import json

import pytest

from verity_vllm.pipeline.build import _instance_rows, _write_instances, _write_instances_file

HEAD = {"schema": "cb-a/instance-sequence/v1", "digest": "d" * 64, "correspondence_digest": None, "params": [["prompt", "Array<8,Value<32>>"]]}
ROWS = [{"i": k, "spec": "Gemm_v1{K=64,N=8}", "form": "call", "name": f"model.layers.0.q_proj/mm\u00e9[{k}]",
         "args": [[["n", k, 0, 64]], [["p", 1, 64 * k, 64 * k + 64], ["n", 3, 0, 1]]]} for k in range(1000)]


@pytest.mark.parametrize("rows", [ROWS, [], ROWS[:1]])
@pytest.mark.parametrize("chunk", [1, 7, 4096])
def test_the_streamed_instance_sequence_is_json_dumps_text(rows, chunk):
    streamed, whole = io.StringIO(), io.StringIO()
    _write_instances(streamed, HEAD, iter(rows), chunk=chunk)
    json.dump({**HEAD, "rows": rows}, whole)
    assert streamed.getvalue() == whole.getvalue()


@pytest.mark.parametrize("rows", [ROWS, []])
def test_the_instance_file_is_that_text_gzipped(tmp_path, rows):
    p = tmp_path / "instances.json.gz"
    _write_instances_file(str(p), HEAD, iter(rows))
    with gzip.open(p, "rt") as f:
        assert f.read() == json.dumps({**HEAD, "rows": rows})


def _scattered_program():
    """A Program whose batch Call reads a SCATTERED Strided view of its parameter (inner 3, inner_stride 2: leaves 0,2,4 / 6,8,10 /
    ...) -- the per-leaf shape of the GLM-5.3 MLA-absorb transposed-weight read, at toy size."""
    from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
    from verity.ml.circuits.program import Program
    from verity.ml.circuits.refs import Coll, strided_view
    from verity.ml.circuits.types import Array, Value
    V32 = Value(32)
    Add32 = PrimitiveDefinition("TAdd32", 1, (("a", V32), ("b", V32)), V32, lambda a, b: (a + b) & 0xFFFFFFFF, register=False)
    Row3 = CompositeDefinition("TRow3", 1, (), lambda S: ((("x", Array(3, V32)),), V32),
                               lambda Bd, S, x: Bd.call(Add32, x[0], x[2]), register=False)
    views = CompositeDefinition("TViews", 1, (), lambda S: ((("M", Array(4, Array(6, V32))),), Array(4, V32)),
                                lambda Bd, S, M: Bd.batch(Row3, Coll(Array(4, Array(3, V32)), strided_view(M, 0, 3, 6, 2, 12)), axes=(0,)),
                                register=False)
    return Program(bind(views))


def _v1_rows(program) -> list[dict]:
    """What the v1 writer wrote (its expression verbatim, `runs(a.refs)` per argument), as JSON left it (lists, not tuples)."""
    from verity.ml.circuits.codec import descriptor_id
    from verity.ml.circuits.refs import runs
    return [{"i": k, "spec": descriptor_id(n.fn), "form": n.form, "name": n.name, "args": [[list(t) for t in runs(a.refs)] for a in n.args]}
            for k, n in enumerate(program.fn.body.nodes)]


def test_instance_rows_store_progressions_and_every_reader_sees_the_v1_run_lists(tmp_path):
    """The file's rows are O(rows) progressions (a "P" item for the scattered read, fewer items than the per-leaf runs), and
    `iter_instance_rows` / `load_program` of the written file return rows BIT-IDENTICAL to the v1 writer's."""
    from verity_vllm.program.compact import is_file_prog
    from verity_vllm.query.manifest.format import load_program
    program = _scattered_program()
    v1 = _v1_rows(program)
    p = tmp_path / "instances.json.gz"
    _write_instances_file(str(p), {"schema": "cb-a/instance-sequence/v2", "digest": "d" * 64, "correspondence_digest": None,
                                   "params": [[n, str(t)] for n, t in program.fn.body.params]}, _instance_rows(program))
    assert list(iter_rows_raw(p)) != v1                        # the file does NOT carry the per-leaf form ...
    from verity_vllm.query.program_view import iter_instance_rows
    assert list(iter_instance_rows(str(p))) == v1              # ... and every reader of it sees exactly the per-leaf form
    (tmp_path / "result.json").write_text("{}")
    assert load_program(str(tmp_path))["instances"]["rows"] == v1
    raw = list(iter_rows_raw(p))
    stored = [it for r in raw for a in r["args"] for it in a]
    assert any(it[0] == "P" for it in stored if is_file_prog(it))
    assert len(stored) < sum(len(a) for r in v1 for a in r["args"])


def iter_rows_raw(p):
    with gzip.open(p, "rt") as f:
        return iter(json.load(f)["rows"])


def test_from_instances_of_a_v2_file_is_the_v1_documents_view(tmp_path):
    """`from_instances` resolves the same reads from the progressions file as from the v1 document (registered specs, a scattered
    p-space weight read stored as one "P" item)."""
    from verity_vllm.program.compact import file_items_from_runs
    from verity_vllm.query.program_view import from_instances
    params = [["prompt", "Array<8,Value<32>>"], ["m.w.weight", "Array<4096,Value<32>>"]]
    v1_rows = [{"i": k, "spec": "Gemm_v1{K=64,N=8}", "form": "call", "name": f"model.layers.0.q_proj/mm[{k}]",
                "args": [[["n", k - 1, 0, 8]] if k else [["p", 0, 0, 8]],
                         [["p", 1, 64 * q + k, 64 * q + k + 1] for q in range(64)]]} for k in range(6)]  # a transposed-weight column
    v2_rows = [{**r, "args": [file_items_from_runs(map(tuple, a)) for a in r["args"]]} for r in v1_rows]
    assert any(it[0] == "P" for r in v2_rows for a in r["args"] for it in a)
    p = tmp_path / "instances.json.gz"
    _write_instances_file(str(p), {"schema": "cb-a/instance-sequence/v2", "digest": "d" * 64, "params": params}, iter(v2_rows))
    P = from_instances(str(p))
    D = from_instances({"schema": "cb-a/instance-sequence/v1", "digest": "d" * 64, "params": params, "rows": v1_rows})
    assert [(c.id, c.definition, c.form, c.name, tuple(c.reads)) for c in P.calls] == \
           [(c.id, c.definition, c.form, c.name, tuple(c.reads)) for c in D.calls]
    # the dict path expands too: the v2 rows as a loaded document resolve the same reads
    V = from_instances({"schema": "cb-a/instance-sequence/v2", "digest": "d" * 64, "params": params, "rows": v2_rows})
    assert [tuple(c.reads) for c in V.calls] == [tuple(c.reads) for c in D.calls]
