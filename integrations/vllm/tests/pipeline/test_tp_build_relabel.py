"""A relabeled TP Build (`build --emit-ranks`, the row's `--build-relabel on`) writes each rank the files a full derive of that rank
writes, its guard fails closed by name, and with the flag off the Build is the one it was.  vLLM is not importable here, so the
ranks are `tp_toy`'s (a vocab-parallel embedding and cross-rank reduces through the construction's own `CollectiveBus`),
at W = 2 (template rank 0) and W = 4 (template rank 1, so rank 0's START = 0 is relabeled to), derived and written in-process by
the Build's own `_write_program`, `_finish` and `emit_ranks` (`_fake_build`, `build.run`'s order)."""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import functools
import gzip
import json
import os

import pytest

from tests.pipeline import tp_toy as T
from verity_vllm.pipeline import build as B
from verity_vllm.pipeline.row_driver import Stop
from verity_vllm.pipeline.row_tp import TpRow
from verity_vllm.pipeline.tp import emit_ranks, relabel_build as RB
from verity_vllm.program.frontend import relabel
from verity_vllm.program.frontend.target_profile import TargetProfile

INPUTS = {"n_open_events": 1, "distinct_paths": 1, "forbidden_hits": [], "site_packages_by_package": {}, "non_site_packages_paths": []}
SAMPLING = {"kind": "greedy", "source": "default"}
#: the telemetry two runs of the writer differ in (wall and memory), and the relabel's provenance; nothing else may differ
TELEMETRY = {"result.json": ("wall_secs_total", "max_rss_mb", "relabeled_from"),
             "construction_manifest.json": ("construction/wall_secs_total", "construction/max_rss_mb", "relabeled_from")}


def _res(W: int, r: int, record: list, wrapper: str) -> dict:
    return {"schema": "cb-a/derive_step/v3", "utc": "2026-10-03T00:00:00Z", "wrapper": wrapper, "sampling": SAMPLING,
            "tp": {"world": W, "rank": r, "collectives": {"n": len(record), "record": record}, "embedding_shards": T.embedding_shards(W, r)},
            "model_pin": {"model": "toy", "rank": r}, "attention_impls_observed": {}, "export_facts": {}, "ok": True}


def _fake_build(W: int, r: int, out: str, args: tuple, hook=None) -> None:
    """`build --tp W --rank r --out OUT ARGS` of the toy rank: the derive, the writer, the relabels (`--emit-ranks`), `_finish` and
    `emit_ranks.finish`, in `build.run`'s order.  `hook(out, write)`: the writer the relabels get (a mutation's)."""
    a = dict(zip(args[::2], args[1::2]))
    emit = emit_ranks.parse_ranks(a["--emit-ranks"], W, r, a["--emit-root"]) if "--emit-ranks" in a else None
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/inputs.json", "w") as f:
        json.dump(INPUTS, f)
    session: list | None = [] if emit else None
    rep, mod, record, inner = T.derive_rank(W, r, session=session)
    res = _res(W, r, record, a.get("--wrapper", "step"))
    write = functools.partial(B._write_program, target=TargetProfile.default(), applied={}, sampling=SAMPLING, launch=None, tp=W,
                              model_root=next(n for n, m in mod.named_modules() if m is inner))
    pre = copy.deepcopy(res) if emit else None
    write(out, res, rep, rank=r)
    if emit:
        emit_ranks.emit(emit, out, pre, session[0], shards=lambda q: T.embedding_shards(W, q), context=contextlib.nullcontext, post=None,
                        write=hook(out, write) if hook else write)
    res = B._finish(out, res, 0.0)
    if emit:
        emit_ranks.finish(emit, out, res, B._write_manifest)


def _files(d: str) -> dict[str, bytes]:
    out = {}
    for name in sorted(os.listdir(d)):
        with (gzip.open if name.endswith(".gz") else open)(os.path.join(d, name), "rb") as f:
            out[name] = f.read()
    return out


def _drop(doc: dict, path: str) -> None:
    *head, last = path.split("/")
    for k in head:
        doc = doc[k]
    doc.pop(last, None)


def assert_same_files(real: str, relabeled: str) -> None:
    """Every file byte for byte (`.gz` decompressed); `result.json` and `construction_manifest.json` with the same keys in the same
    order once the wall, memory and `relabeled_from` keys are dropped."""
    a, b = _files(real), _files(relabeled)
    assert sorted(a) == sorted(b)
    for name in a:
        if name in TELEMETRY:
            x, y = json.loads(a[name]), json.loads(b[name])
            for p in TELEMETRY[name]:
                _drop(x, p)
                _drop(y, p)
            assert json.dumps(x) == json.dumps(y), name
        else:
            assert a[name] == b[name], name


@pytest.mark.slow
@pytest.mark.parametrize("W", [2, 4])
def test_every_relabeled_rank_writes_the_bytes_its_full_derive_writes(W, tmp_path):
    t = RB.template_rank(W)
    others = [r for r in range(W) if r != t]
    tmpl = f"{tmp_path}/tmpl/build_step"
    _fake_build(W, t, tmpl, ("--emit-ranks", ",".join(map(str, others)), "--emit-root", f"{tmp_path}/rel"))
    for r in range(W):
        _fake_build(W, r, f"{tmp_path}/real/rank{r}/build_step", ())
    assert_same_files(f"{tmp_path}/real/rank{t}/build_step", tmpl)          # the template's own directory: a plain derive's
    for r in others:
        rel = f"{tmp_path}/rel/rank{r}/build_step"
        assert_same_files(f"{tmp_path}/real/rank{r}/build_step", rel)
        res = json.load(open(f"{rel}/result.json"))
        assert res["relabeled_from"] == {"rank": t, "dir": tmpl} and res["tp"]["rank"] == r
        assert json.load(open(f"{rel}/construction_manifest.json"))["relabeled_from"] == res["relabeled_from"]
        assert RB.compare_dirs(f"{tmp_path}/real/rank{r}/build_step", rel) is None
    doc = emit_ranks.read_report(tmpl)
    assert doc["template"] == {"dir": tmpl, "world": W, "rank": t, "ok": True}
    assert [(x["rank"], x["ok"], x["error"]) for x in doc["ranks"]] == [(r, True, None) for r in others]
    assert not [n for n in os.listdir(f"{tmp_path}/rel/rank{others[0]}") if n.startswith(".")], "no .relabel-tmp left"


def test_emit_ranks_takes_other_ranks_of_the_world_and_a_root():
    e = emit_ranks.parse_ranks("0,2,3", 4, 1, "/b")
    assert e.ranks == (0, 2, 3) and e.dir_of("/x/rank1/build_step/", 2) == "/b/rank2/build_step"
    for arg, world, rank, root in (("1", 4, 1, "/b"), ("4", 4, 1, "/b"), ("0,0", 4, 1, "/b"), ("0", 4, 1, None), ("1", 1, 0, "/b"), ("", 4, 1, "/b")):
        with pytest.raises(ValueError, match="--emit-ranks"):
            emit_ranks.parse_ranks(arg, world, rank, root)
    assert emit_ranks.read_report("/nonexistent/build_step") == {"schema": emit_ranks.SCHEMA, "ranks": [], "missing": True}


def test_a_relabel_leaves_the_template_derivation_as_it_was(tmp_path):
    W, t = 4, 1
    sess: list = []
    tmpl = T.derive_rank(W, t, session=sess)[0]
    blob = (json.dumps(tmpl.to_json(), default=str), json.dumps(tmpl.descriptor, sort_keys=True))
    for r in (0, 2, 3, 1):
        rep = relabel.rank_report(sess[0], r, shards=relabel.shard_map(T.embedding_shards(W, t), T.embedding_shards(W, r)))
        assert (json.dumps(rep.to_json(), default=str), json.dumps(rep.descriptor, sort_keys=True)) == \
               ((blob if r == t else (json.dumps(T.derive_rank(W, r)[0].to_json(), default=str),
                                      json.dumps(T.derive_rank(W, r)[0].descriptor, sort_keys=True))))
    plain = T.derive_rank(W, t)[0]
    assert (json.dumps(plain.to_json(), default=str), json.dumps(plain.descriptor, sort_keys=True)) == blob, "a session changes no byte"


def test_a_dead_free_template_relabels_each_rank_to_its_dead_free_derive():
    """`--dead-free` (a TP rank's request in the registered compiler's form): rank 1 written from rank 0's derivation is rank 1's own
    dead-free derive, without the reduces of the rows no logit reads, which the derive of record keeps."""
    from verity_vllm.program.prune import dead_units
    sess: list = []
    T.derive_rank(2, 0, session=sess, last=True, dead_free=True)
    rep = relabel.rank_report(sess[0], 1, shards=relabel.shard_map(T.embedding_shards(2, 0), T.embedding_shards(2, 1)))
    real, kept = T.derive_rank(2, 1, last=True, dead_free=True)[0], T.derive_rank(2, 1, last=True)[0]
    assert json.dumps(rep.descriptor, sort_keys=True) == json.dumps(real.descriptor, sort_keys=True)
    body = kept.program.fn.body
    assert not dead_units(real.program.fn.body) and "AllReduce2_v1" in {body.nodes[i].fn.id.split("{")[0] for i in dead_units(body)}


def test_the_relabel_refuses_what_it_cannot_rewrite(monkeypatch):
    W, t = 4, 1
    sess: list = []
    T.derive_rank(W, t, session=sess)
    s, shards = sess[0], relabel.shard_map(T.embedding_shards(W, t), T.embedding_shards(W, 2))
    with pytest.raises(relabel.RelabelRefused, match="not at the template's rank 2"):
        relabel.rank_report(dataclasses.replace(s, rank=(W, 2)), 3, shards=shards)
    with pytest.raises(relabel.RelabelRefused, match="family"):
        relabel.rank_report(dataclasses.replace(s, family={"matched": True}), 2, shards=shards)
    with pytest.raises(relabel.RelabelRefused, match="no rank-2 vocab range"):
        relabel.rank_report(s, 2, shards={})
    with pytest.raises(relabel.RelabelRefused, match="modules differ"):
        relabel.shard_map(T.embedding_shards(W, t), [{**T.embedding_shards(W, 2)[0], "module": "inner.other"}])
    monkeypatch.setattr(relabel, "COLLECTIVES", tuple(c for c in relabel.COLLECTIVES if c != "all_gather"))
    with pytest.raises(relabel.RelabelRefused, match="all_gather"):
        relabel.rank_report(s, 2, shards=shards)


# ------------------------------------------------------------------------------------------------ the row's side
class _Row:
    """What `relabel_build.derive` drives of a `TpRow`, with each derive `_fake_build` in-process."""
    step, request, launch_of, launch_args = TpRow.step, TpRow.request, TpRow.launch_of, TpRow.launch_args

    def __init__(self, tmp, W: int, hook=None) -> None:
        self.W, self.B, self.bdir, self.hook, self.lctx, self.aot = W, 1, f"{tmp}/build", hook, {}, None
        self.marks: list = []
        self.logs: list[str] = []
        self.verdicts: list = []
        self.rounds: list[list] = []

    def rank(self, r: int) -> str:
        return f"{self.bdir}/rank{r}"

    def tl(self, stage, ev, *kv) -> None:
        self.marks.append((stage, dict(kv)))

    def log(self, msg: str) -> None:
        self.logs.append(msg)

    def verdict(self, stage, status, detail, **_) -> None:
        self.verdicts.append((stage, status, detail))

    def derive_all(self, todo, jobs) -> None:
        self.rounds.append([(j[0], os.path.relpath(j[1], self.bdir), "--emit-ranks" in j) for j in todo])
        for j in todo:
            _fake_build(self.W, j[0], j[1], tuple(j[2:]), hook=self.hook)

    def todo(self) -> list[tuple]:
        """`TpRow.build`'s: per rank its step, then its request (one shape)."""
        out = []
        for r in range(self.W):
            os.makedirs(self.rank(r), exist_ok=True)
            out += [self.step(r), self.request(r, 4, 2)]
        return out


REQ = "build_request_LP4_T2"


def _relabeled(row: _Row) -> list[int]:
    return [r for r in range(row.W) if os.path.isfile(f"{row.rank(r)}/{REQ}/result.json")
            and "relabeled_from" in json.load(open(f"{row.rank(r)}/{REQ}/result.json"))]


@pytest.mark.parametrize("W", [2, 4])
def test_a_relabeled_build_places_every_request_after_the_step_guard_and_the_canary(W, tmp_path):
    row, t = _Row(tmp_path, W), RB.template_rank(W)
    RB.derive(row, row.todo(), 4)
    canary = W - 1
    assert row.rounds[0] == [(r, f"rank{r}/build_step", False) for r in range(W)] + [(t, "relabel_check/rank%d/build_step" % t, True)]
    assert row.rounds[1:] == [[(t, f"rank{t}/{REQ}", True), (canary, f"rank{canary}/{REQ}", False)]]
    assert _relabeled(row) == [r for r in range(W) if r not in (t, canary)]
    for r in range(W):                       # every rank's request: what its full derive writes
        _fake_build(W, r, f"{tmp_path}/real/rank{r}", ("--wrapper", "request", "--LP", "4", "--T", "2"))
        assert_same_files(f"{tmp_path}/real/rank{r}", f"{row.rank(r)}/{REQ}")
    g = json.load(open(f"{row.bdir}/relabel_check/guard.json"))
    assert all(s["ok"] for s in g["steps"].values()) and sorted(g["steps"]) == [str(r) for r in range(W)]
    assert g["canary"] == {"rank": canary, "request": REQ, "ok": True, "difference": None}
    assert [m for s, m in row.marks if s == "build.relabel_guard"] == [{"rank": r, "ok": "true", "file": "null", "path": "null"} for r in range(W)]
    assert not row.verdicts


def _rewrite_no_embedding(monkeypatch):
    monkeypatch.setattr(relabel, "shard_map", lambda tmpl, rank: {(d["START"], d["END"]): (d["START"], d["END"]) for d in tmpl})


def _range_off_by_one(monkeypatch):
    real = relabel.shard_map
    monkeypatch.setattr(relabel, "shard_map", lambda tmpl, rank: {k: (v[0] + 1, v[1] + 1) for k, v in real(tmpl, rank).items()})


def _rewrite_no_collective(monkeypatch):
    real = relabel._rank_args
    monkeypatch.setattr(relabel, "_rank_args", lambda ctx, node, t, r, world, shards:
                        real(ctx, node, t, r, world, shards) if relabel._op(node) == relabel.EMBEDDING else tuple(node.args))


def _own_at_the_next_rank(monkeypatch):
    real = relabel._rank_args
    monkeypatch.setattr(relabel, "_rank_args", lambda ctx, node, t, r, world, shards:
                        real(ctx, node, t, r if relabel._op(node) == relabel.EMBEDDING else (r + 1) % world, world, shards))


def _links_of_the_next_rank(out, write):
    return lambda d, res, rep, rank: write(d, res, rep, rank=(rank + 1) % res["tp"]["world"])


#: a rank dependence the relabel misses (an expected rewrite dropped) or gets wrong (a relabeled value perturbed)
MUTATIONS = {
    "embedding START/END not rewritten": (_rewrite_no_embedding, None),
    "vocab range (START, END) off by one": (_range_off_by_one, None),
    "own operand left at the template's index": (_rewrite_no_collective, None),
    "own operand at the next rank's index": (_own_at_the_next_rank, None),
    "tp_links.json written for the next rank": (None, _links_of_the_next_rank),
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_the_step_guard_fails_closed_by_name_and_every_request_is_derived(name, tmp_path, monkeypatch):
    patch, hook = MUTATIONS[name]
    if patch:
        patch(monkeypatch)
    W = 4
    row, t = _Row(tmp_path, W, hook=hook), RB.template_rank(W)
    RB.derive(row, row.todo(), 4)
    g = json.load(open(f"{row.bdir}/relabel_check/guard.json"))
    bad = {int(r): s for r, s in g["steps"].items() if not s["ok"]}
    assert sorted(bad) == [r for r in range(W) if r != t] and g["steps"][str(t)] == {"ok": True}
    for s in bad.values():
        d = s["difference"]
        assert d["file"] and (d.get("path") or d.get("offset") is not None), d
        if hook:
            assert (d["file"], d["path"]) == ("tp_links.json", "/rank"), d
    marks = [m for s, m in row.marks if s == "build.relabel_guard" and m["ok"] == "false"]
    assert [m["rank"] for m in marks] == sorted(bad) and all(m["file"] != "null" for m in marks)
    assert row.rounds[1] == [(r, f"rank{r}/{REQ}", False) for r in range(W)] and len(row.rounds) == 2
    assert g["canary"] is None and _relabeled(row) == [] and not os.path.exists(f"{row.bdir}/relabel_check/staged")


def test_a_request_only_dependence_fails_the_build_at_the_canary_and_places_nothing(tmp_path):
    W = 4
    hook = lambda out, write: _links_of_the_next_rank(out, write) if REQ in out else write   # noqa: E731
    row, t = _Row(tmp_path, W, hook=hook), RB.template_rank(W)
    with pytest.raises(Stop) as e:
        RB.derive(row, row.todo(), 4)
    assert e.value.rc == 10
    [(stage, status, detail)] = row.verdicts
    assert (stage, status) == ("build", "FAIL") and "relabel canary: rank 3" in detail and "tp_links.json /rank" in detail
    g = json.load(open(f"{row.bdir}/relabel_check/guard.json"))
    assert all(s["ok"] for s in g["steps"].values())
    assert g["canary"]["ok"] is False and g["canary"]["difference"]["file"] == "tp_links.json"
    assert [r for r in range(W) if os.path.isdir(f"{row.rank(r)}/{REQ}")] == [t, 3], "only the template's and the canary's derives"
    assert sorted(os.listdir(f"{row.bdir}/relabel_check/staged")) == ["rank0", "rank2", "rank3"], "the relabels stay as evidence"


def test_a_relabel_that_raises_leaves_its_rank_to_a_full_derive(tmp_path):
    W = 4

    def hook(out, write):
        def broken(d, res, rep, rank):
            raise RuntimeError(f"emit of rank {rank} failed")
        return broken if REQ in out else write

    row, t = _Row(tmp_path, W, hook=hook), RB.template_rank(W)
    RB.derive(row, row.todo(), 4)
    g = json.load(open(f"{row.bdir}/relabel_check/guard.json"))
    assert g["requests"][REQ]["emit_failed"] == {str(r): f"RuntimeError: emit of rank {r} failed" for r in (0, 2, 3)}
    assert g["canary"]["ok"] is None and g["canary"]["why"] == "not compared: its relabel failed"
    assert row.rounds[2] == [(0, f"rank0/{REQ}", False), (2, f"rank2/{REQ}", False)]
    assert _relabeled(row) == [] and all(os.path.isfile(f"{row.rank(r)}/{REQ}/result.json") for r in range(W))


def test_compare_dirs_names_the_first_difference(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        d.mkdir()
        (d / "inputs.json").write_text(json.dumps({"who": str(d)}))
        (d / "result.json").write_text(json.dumps({"ok": True, "max_rss_mb": len(str(d)), "tp": {"rank": 1}}))
    assert RB.compare_dirs(str(a), str(b)) is None, "inputs.json and telemetry are not compared"
    (b / "tp_links.json").write_text("{}")
    assert RB.compare_dirs(str(a), str(b)) == {"file": "tp_links.json", "only_in": "relabel", "files": ["tp_links.json"]}
    (a / "tp_links.json").write_text(json.dumps({"x": 1, "y": 2}))
    (b / "tp_links.json").write_text(json.dumps({"y": 2, "x": 1}))
    assert RB.compare_dirs(str(a), str(b))["path"].startswith("(same values, other bytes")
    (b / "result.json").write_text(json.dumps({"ok": True, "tp": {"rank": 2}}))
    with gzip.open(a / "descriptor.json.gz", "wb") as f:
        f.write(b'{"k": [1, 2, 3]}')
    with gzip.open(b / "descriptor.json.gz", "wb") as f:
        f.write(b'{"k": [1, 2, 4]}')
    assert RB.compare_dirs(str(a), str(b)) == {"file": "descriptor.json.gz", "path": "/k/2", "real": "3", "relabel": "4",
                                               "files": ["descriptor.json.gz", "tp_links.json", "result.json"]}, \
        "the content files first, then the summaries (result.json's /tp/rank)"
    monkeypatch.setattr(RB, "PARSE_MAX_BYTES", 15)     # the descriptor is 16 bytes decompressed (and more compressed)
    assert {k: v for k, v in RB.compare_dirs(str(a), str(b)).items() if k != "files"} == \
           {"file": "descriptor.json.gz", "offset": 13, "real": '{"k": [1, 2, 3]}', "relabel": '{"k": [1, 2, 4]}'}, "too large to parse"


def test_compare_dirs_names_a_file_it_cannot_read_as_a_difference(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        d.mkdir()
    (a / "report.json").write_text('{"rank": 1}')
    (b / "report.json").write_text('{"rank": 1')
    assert {k: v for k, v in RB.compare_dirs(str(a), str(b)).items() if k != "files"} == \
           {"file": "report.json", "offset": 10, "real": '{"rank": 1}', "relabel": '{"rank": 1'}, "not JSON: the byte offset"
    (b / "report.json").write_text('{"rank": 1}')
    with gzip.open(a / "descriptor.json.gz", "wb") as f:
        f.write(b'{"k": [1, 2, 3]}' * 64)
    (b / "descriptor.json.gz").write_bytes((a / "descriptor.json.gz").read_bytes()[:-12])
    d = RB.compare_dirs(str(a), str(b))
    assert d["file"] == "descriptor.json.gz" and d["unreadable"].startswith("EOFError") and RB._where(d) == d["unreadable"]
    (b / "result.json").write_text("{")
    (a / "result.json").write_text("{}")
    assert RB.compare_dirs(str(a), str(b))["files"] == ["descriptor.json.gz", "result.json"], "a summary that is not JSON too"
    assert RB._where({"file": "x", "only_in": "real"}) == "only in the real directory"
