"""The Build's instance sequence (`instances.json.gz`) is written a chunk of rows at a time with `json.dumps`, and its text is
`json.dump`'s of the whole document, whatever the chunk size and the file's gzip level.  Its rows store each argument in the
progressions form (instances v2, `compact.file_items_from_refs`: O(rows) for a scattered Strided where the v1 per-leaf `runs()`
is O(leaves)), and every reader -- `iter_instance_rows`, `from_instances`, `manifest_format.load_program` -- expands them back to
run lists bit-identical to what the v1 writer wrote."""
from __future__ import annotations

import gzip
import io
import json
import zlib

import pytest

from verity_vllm.pipeline.build import _instance_rows, _write_instances, _write_instances_file
from verity_vllm.program.instances_file import write_instances_file

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


@pytest.mark.parametrize("block,threads", [(1 << 24, None), (1, 1), (97, 3), (40_000, 2)])
def test_the_parallel_gzip_is_one_member_of_the_text(tmp_path, block, threads):
    """Blocks of any size, compressed on any number of threads, inflate to the text, as one gzip member (a reader that stops at the
    first member reads it all) whose bytes depend on the text only."""
    from verity_vllm.program.instances_file import ParallelGzip
    text = json.dumps({**HEAD, "rows": ROWS})
    for body in (text, ""):
        paths = [tmp_path / f"a{block}.gz", tmp_path / f"b{block}.gz"]
        for p in paths:
            with ParallelGzip(str(p), block=block, threads=threads) as f:
                for k in range(0, len(body), 1000):
                    f.write(body[k:k + 1000])
        data = paths[0].read_bytes()
        assert zlib.decompress(data, 31).decode() == gzip.decompress(data).decode() == body
        assert data == paths[1].read_bytes()


def _attention_program(seed: int):
    """Calls reading cuts of two `PartLog`s (the dense attention's per-head key and value logs), each cut the log so far plus a few
    parts of every kind (contiguous, broadcast, empty, strided, scattered, explicit, nested), sometimes shorter than a cut before."""
    import random

    from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition
    from verity.ml.circuits.program import Program
    from verity.ml.circuits.refs import Affine, Coll, Concat, Explicit, PartLog, Strided, broadcast, concat
    from verity.ml.circuits.types import Array, Value
    V16, W, N = Value(16), 96, 10
    rng = random.Random(seed)
    Neg = PrimitiveDefinition("TINeg16", 1, (("a", V16),), V16, lambda a: -a & 0xFFFF, register=False)

    def comp(name, params, ret, body):
        return CompositeDefinition(name, 1, (), lambda S: (tuple(params), ret), body, register=False)

    pair = comp("TIPair", [("a", V16)], Array(2, V16), lambda B, S, a: Coll(Array(2, V16), concat([a.refs, B.call(Neg, a).refs])))
    sums: dict[int, object] = {}

    def summed(n):
        if n not in sums:
            sums[n] = comp(f"TISum{n}", [("x", Array(n, V16))], V16, lambda B, S, x: B.call(Neg, x[0]))
        return sums[n]

    def part(rows, t):
        k = rng.randrange(9)
        b = rng.randrange(W - 12)
        if k == 0:
            return Affine("p", 0, b, rng.randrange(1, 5))
        if k == 1:
            return broadcast("p", 0, b, rng.randrange(1, 4))
        if k == 2:
            return Affine("p", 0, b, 0)
        if k == 3:
            return Strided("p", 0, b, 2, 3, 1, 4)
        if k == 4:
            return Strided("p", 0, b, 3, 4, 2, 6)
        if k == 5:
            return Explicit(tuple(("p", 0, rng.randrange(W)) for _ in range(rng.randrange(1, 4))))
        if k == 6:
            return Concat([Affine("p", 0, b, 2), rows[rng.randrange(t + 1)].refs])
        return rows[rng.randrange(t + 1)].refs

    def body(B, S, x):
        rows = [B.call(pair, x[t]) for t in range(N)]
        keys, values, outs = PartLog(), PartLog(), []
        for t in range(N):
            keys.append(rows[t].refs)
            keys.append(part(rows, t) if rng.random() < 0.3 else Affine("p", 0, 2 * t, 2))
            values.append(rows[t].refs if rng.random() < 0.8 else part(rows, t))
            for log in (keys, values):
                j = len(log) if rng.random() < 0.85 else rng.randrange(len(log) + 1)
                refs = log.cut(j, [part(rows, t) for _ in range(rng.choice([0, 0, 1, 2]))])
                if refs.count:
                    outs.append(B.call(summed(refs.count), Coll(Array(refs.count, V16), refs)))
        return outs[-1]

    return Program(comp(f"TIAttention{seed}", [("x", Array(W, V16))], V16, body))


@pytest.mark.parametrize("seed", range(12))
def test_a_programs_rows_written_from_its_logs_are_the_json_dumps_text(tmp_path, seed):
    """`write_program_instances` writes each log cut as its log's items so far plus its own, and the file's text is the one the
    row dicts give (`instance_rows` through `json.dumps`), cut by cut, whatever the cut's tail and whether it is shorter."""
    from verity_vllm.program.instances_file import RowTexts, instance_rows, write_program_instances
    program = _attention_program(seed)
    rows = list(instance_rows(program))
    texts = RowTexts()
    assert [json.loads(t) for t in texts.rows(program)] == rows and texts.logs
    head = {"schema": "cb-a/instance-sequence/v2", "digest": "d" * 64, "params": [["x", "Array<96,Value<16>>"]]}
    p = tmp_path / "instances.json.gz"
    write_program_instances(str(p), head, program, block=4096, threads=2)
    assert gzip.decompress(p.read_bytes()).decode() == json.dumps({**head, "rows": rows})
    _read_as_json_load(p)


def _glm_attention_layer(rows: int, heads: int):
    """One GLM-5.3 TP8 layer's dense attention prefill as the derive leaves it, its nodes only: per row and head an `Attention_v4`
    call reading q, the head's key log cut at the row (k_nope's and k_pe's rows, two producers a row) and its value log."""
    from types import SimpleNamespace as NS

    from verity.ml.circuits.refs import Affine, PartLog
    keys, values, nodes = [PartLog() for _ in range(heads)], [PartLog() for _ in range(heads)], []
    for t in range(rows):
        for h in range(heads):
            keys[h].append(Affine("n", 1000 + 3 * t, 192 * h, 192 * h + 128))
            keys[h].append(Affine("n", 1001 + 3 * t, 0, 64))
            values[h].append(Affine("n", 1002 + 3 * t, 128 * h, 128 * h + 128))
            args = [Affine("n", 10 * t, 192 * h, 192 * h + 192), keys[h].cut(len(keys[h])), values[h].cut(len(values[h]))]
            nodes.append(NS(fn=f"Attention_v4{{T={t + 1},NH=1,KVH=1}}", form="call", name=f"layers.0.attn/h{h}[{t}]",
                            args=[NS(refs=r) for r in args]))
    return NS(fn=NS(body=NS(nodes=nodes)))


def test_the_writer_encodes_each_heads_items_once_not_once_a_call(monkeypatch):
    """A GLM-5.3 TP8 layer at 256 input tokens (8 heads): the row dicts encode every key item once per call that reads it, 8 x the
    sum of 1..256 (267,264 items), where `RowTexts` encodes each once and each call's last few (8,184), into the same text: 33
    times less work here, and rows/8 times less at any length.  The work is counted, not timed (no test reads a clock): the I2048
    rank's 4 h writer was that sum, 1.3e9 items."""
    from verity.ml.circuits import codec
    from verity_vllm.program import compact
    from verity_vllm.program.instances_file import RowTexts, instance_rows
    monkeypatch.setattr(codec, "descriptor_id", lambda fn: fn)
    program, encoded = _glm_attention_layer(256, 8), []
    items = compact._file_items
    monkeypatch.setattr(compact, "_file_items", lambda progs: encoded.append(len(progs)) or items(progs))
    rows = list(instance_rows(program))
    old, encoded[:] = sum(encoded), []
    texts = list(RowTexts().rows(program))
    new = sum(encoded)
    assert texts == [json.dumps(r) for r in rows]
    assert old > 8 * 256 * 257 // 2 and new <= 4 * len(rows) and old > 25 * new, (old, new)


def test_the_writer_encodes_a_weight_read_by_every_tokens_call_once(monkeypatch):
    """GLM-5.3's MLA-absorb read of a transposed weight is a scattered `Strided` (a progression per row, ~1e6 leaves) that every
    token's Call of a layer reads again, a new but equal object each time: `RowTexts` encodes each distinct read once (3 layers x 2
    reads here, against one per Call and read), into the same text, and every time once its memo is full."""
    from types import SimpleNamespace as NS

    from verity.ml.circuits import codec
    from verity.ml.circuits.refs import Affine, Concat, Strided
    from verity_vllm.program import compact, instances_file
    from verity_vllm.program.instances_file import RowTexts, instance_rows
    monkeypatch.setattr(codec, "descriptor_id", lambda fn: fn)
    nodes = []
    for layer in range(3):
        for t in range(64):
            w = Strided("p", 7 + layer, 0, 64, 1, 512, 64 * 128)
            args = [Affine("n", 100 * t + layer, 0, 512), w, Concat([Affine("p", 3, 16 * layer, 16), w])]
            nodes.append(NS(fn="MlaAbsorb{H=16}", form="call", name=f"layers.{layer}.attn/absorb[{t}]", args=[NS(refs=r) for r in args]))
    program = NS(fn=NS(body=NS(nodes=nodes)))
    calls = []
    scattered = compact._scattered_strided
    monkeypatch.setattr(compact, "_scattered_strided", lambda *a: calls.append(a) or scattered(*a))
    want = [json.dumps(r) for r in instance_rows(program)]
    old, calls[:] = len(calls), []
    assert list(RowTexts().rows(program)) == want
    assert old == 3 * 64 * 2 and len(calls) == 3 * 2, (old, len(calls))
    monkeypatch.setattr(instances_file, "SCATTERED_MEMO_BYTES", 0)
    calls[:] = []
    assert list(RowTexts().rows(program)) == want and len(calls) == old


def _read_as_json_load(p, **dump) -> None:
    from verity_vllm.program.compact import expanded_file_args
    from verity_vllm.program.instances_file import read_instances
    with gzip.open(p, "rt") as f:
        want = json.load(f)
    assert read_instances(str(p), expand=False) == want
    for r in want.get("rows") or ():
        if isinstance(r, dict) and r.get("args"):
            r["args"] = expanded_file_args(r["args"])
    assert read_instances(str(p)) == want


def _attention_rows(n: int, heads: int = 2) -> list[dict]:
    """Rows shaped like a dense attention prefill's: per row and head a key operand of the rows so far (one count-2 "N" item a row)
    and a value operand of one progression, beside a weight read."""
    rows = []
    for t in range(n):
        for h in range(heads):
            rows.append({"i": len(rows), "spec": f"Attention_v4{{T={t + 1}}}", "form": "call", "name": f"layers.0.attn/h{h}[{t}]",
                         "args": [[["n", 10_000 + t, 64 * h, 64 * h + 64]],
                                  [["N", 1000 + 7 * s, 64 * h, 64 * h + 64, 3, 0, 0, 2] for s in range(t + 1)],
                                  [["N", 5000, 64 * h, 64 * h + 64, 7, 0, 0, t + 1]] if t else [["n", 5000, 64 * h, 64 * h + 64]],
                                  [["p", 3, 4096 * h, 4096 * h + 4096]]]})
    return rows


@pytest.mark.parametrize("chunk", [1, 5, 64, 1 << 22])
def test_read_instances_is_json_load_of_the_file_whatever_its_chunks(tmp_path, monkeypatch, chunk):
    """`read_instances` is `json.load` (and, expanding, `expanded_file_args` of each row's arguments) of any instance-sequence text:
    pretty-printed or not, keys after the rows, a key named "rows" inside the header, numbers cut by a chunk boundary."""
    from verity_vllm.program import instances_file
    monkeypatch.setattr(instances_file, "_CHUNK", chunk)
    rows = _attention_rows(6) + ROWS[:5] + [{"i": 99, "args": []}, {"i": 100}]
    docs = [{**HEAD, "rows": rows}, {**HEAD, "rows": []}, {"rows": rows[:3], "after": [1.5, 1234567, None, True], "z": "rows"},
            {"params": [["rows", "Array<2,Value<8>>"]], "rows": rows[:2], "rows_n": 2}, {}]
    for k, doc in enumerate(docs):
        for dump in ({}, {"indent": 1}, {"separators": (",", ":")}):
            p = tmp_path / f"d{k}.json.gz"
            p.write_bytes(gzip.compress(json.dumps(doc, **dump).encode()))
            _read_as_json_load(p)
            q = tmp_path / f"d{k}.json"
            q.write_text(json.dumps(doc, **dump) + "\n")
            _read_as_json_load_plain(q)


def _read_as_json_load_plain(q) -> None:
    from verity_vllm.program.instances_file import read_instances
    assert read_instances(str(q), expand=False) == json.loads(q.read_text())


def test_read_instances_shares_equal_items_and_holds_a_fraction_of_json_loads_memory(tmp_path):
    """Equal argument items are one list, so a causal attention's rows (each key operand the row before's plus one item) hold each
    item once: at 192 rows of 2 heads json.load's document is about 7 times the shared one, and the ratio grows with the rows (the
    GLM-5.3 B8 rank's 8 request Programs measured 56 times their gz size, 139.5 GB, under json.load)."""
    import tracemalloc

    from verity_vllm.program.instances_file import read_instances
    p = tmp_path / "instances.json.gz"
    write_instances_file(str(p), HEAD, _attention_rows(192))
    doc = read_instances(str(p), expand=False)
    keys = [r["args"][1] for r in doc["rows"]]
    assert all(a[0] is b[0] for a, b in zip(keys, keys[2:]))

    def held(load):
        tracemalloc.start()
        d = load()
        size = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        assert d["rows"]
        return size

    def plain():
        with gzip.open(p, "rt") as f:
            return json.load(f)

    shared, whole = held(lambda: read_instances(str(p), expand=False)), held(plain)
    assert whole > 5 * shared, (whole, shared)


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
