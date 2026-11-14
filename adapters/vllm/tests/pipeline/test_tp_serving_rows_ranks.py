"""`verity_vllm.pipeline.tp.serving_rows` where the driver meets its ranks.  `commit` asks every rank for its window in one
collective call, under the TP run root and the roots of the world's ranks, and writes ``tp.json`` over the answers.  `rank_rows`
answers with the rank's window, or for a rank that cannot write one with the error and its traceback where the window would be.
`write` refuses a rank with no finalized run or no binding map, and a rank whose root is not the one the TP run root combines.
Then the verifier's `window_of` for a rank outside the record's world; `from_bundle`'s refusals of a bundle whose store, TP run
root or weights tree is not a rows-sha512 Commit's of record, and the Programs it loads under a Build root; and what the command
hands `from_bundle` for each rank when a checkpoint is named.  CPU-only: the LLM is a list of stand-in workers, the bundle's and
the record's readers are stood in, and no committed value is read."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from verity_vllm.check import oracle_compare as OC
from verity_vllm.check import weights_of_record as WR
from verity_vllm.pipeline import cli
from verity_vllm.pipeline import replay_bundle as RB
from verity_vllm.pipeline import serving_rows as SRW
from verity_vllm.pipeline.tp import c2 as C2
from verity_vllm.pipeline.tp import commit_replay as CR
from verity_vllm.pipeline.tp import rank_replay as RR
from verity_vllm.pipeline.tp import run_root as RT
from verity_vllm.pipeline.tp import serving_rows as SRT

ROOT = bytes(range(64))
CLAIMED = {"0": "0a" * 64, "1": "0b" * 64}
TP = {"world": 2, "tp_run_root": ROOT.hex(), "roots": CLAIMED, "root_tag": RT.TP_ROOT_TAG.decode()}
NO_RUN = "rank 1: no finalized run or no binding map (binding skipped or failed)"
OTHER_ROOT = "rank 1: its run root 0c0c0c0c0c0c0c0c is not the one the TP run root combines"


class _LLM:
    """vLLM's LLM as the driver uses it: one collective call, answered by each rank's worker in rank order."""

    def __init__(self, states: list[dict[str, Any]]) -> None:
        self.workers = [SimpleNamespace(_tp2_commit=st) for st in states]
        self.calls: list[tuple[Any, Any, tuple]] = []

    def collective_rpc(self, fn: Any, timeout: Any = None, args: tuple = ()) -> list[Any]:
        self.calls.append((fn, timeout, args))
        return [fn(w, *args) for w in self.workers]


def _args(tmp_path: Path) -> tuple[SimpleNamespace, dict[str, Any]]:
    """The driver's options naming a request file, and the request each rank gets: the file's, with the Commit's label."""
    req = {"dir": str(tmp_path / "windows"), "partition": str(tmp_path / "partition.json"), "committer": {"key": "k"}}
    (tmp_path / "request.json").write_text(json.dumps(req))
    return SimpleNamespace(serving_rows=str(tmp_path / "request.json"), rpc_timeout=7, label="the row"), {**req, "row": "the row"}


def _rc(root: str) -> SimpleNamespace:
    return SimpleNamespace(run_root=bytes.fromhex(root))


# ------------------------------------------------------------------------------------------------------- the driver and its ranks


@pytest.mark.parametrize("claimed", [CLAIMED, {**CLAIMED, "2": "0c" * 64}], ids=["the world's roots", "a root of a rank past the world"])
def test_the_driver_asks_every_rank_in_one_call_under_the_tp_run_root_and_writes_the_world(tmp_path, monkeypatch, capsys, claimed):
    seen = []

    def write(st: dict[str, Any], request: dict[str, Any], tp: dict[str, Any]) -> dict[str, Any]:
        seen.append((st["rank"], request, tp))
        return {"run": SRT.run_of(ROOT, st["rank"]), "population": 3 + st["rank"], "registration_digest": str(st["rank"]) * 20}

    monkeypatch.setattr(SRT, "write", write)
    llm, (args, request) = _LLM([{"rank": 0}, {"rank": 1}]), _args(tmp_path)
    doc = SRT.commit(llm, args, ROOT, claimed, 2)
    assert llm.calls == [(SRT.rank_rows, 7, (request, TP))] and seen == [(0, request, TP), (1, request, TP)]
    assert doc == json.loads((tmp_path / "windows" / "tp.json").read_text())
    assert doc == {"schema": SRT.SCHEMA, **TP, "runs": {"0": f"{ROOT.hex()}/rank0", "1": f"{ROOT.hex()}/rank1"}, "result": True, "why": None,
                   "windows": {str(r): {"run": f"{ROOT.hex()}/rank{r}", "population": 3 + r, "registration_digest": str(r) * 20} for r in (0, 1)}}
    assert capsys.readouterr().out == ("[tp2commit] serving rows: 2 rank windows under TP run root 0001020304050607: "
                                       "rank 0 3 units, registration 0000000000000000, rank 1 4 units, registration 1111111111111111\n")


def test_a_commit_with_a_rank_that_wrote_no_window_stops_with_every_such_rank_named_and_the_world_written(tmp_path):
    """Rank 0 finalized no run; rank 1's root is another than the one the driver combined."""
    llm, (args, _request) = _LLM([{"rank": 0, "rc": None, "bmap": {"m": 1}}, {"rank": 1, "rc": _rc("0c" * 64), "bmap": {"m": 1}}]), _args(tmp_path)
    with pytest.raises(SystemExit) as stop:
        SRT.commit(llm, args, ROOT, CLAIMED, 2)
    assert str(stop.value) == (f"fail-closed: serving rows: rank 0: LookupError('{NO_RUN.replace('rank 1', 'rank 0')}'); "
                               f"rank 1: ValueError('{OTHER_ROOT}')")
    doc = json.loads((tmp_path / "windows" / "tp.json").read_text())
    assert doc["result"] is False and doc["windows"] == {"0": None, "1": None} and doc["roots"] == CLAIMED
    assert doc["why"] == str(stop.value).removeprefix("fail-closed: serving rows: ")
    assert sorted(p.name for p in (tmp_path / "windows").iterdir()) == ["rank0", "rank1", "tp.json"]


@pytest.mark.parametrize("state, tp, kind, why", [
    ({"rc": None, "bmap": {"m": 1}}, TP, "LookupError", NO_RUN),
    ({"rc": _rc(CLAIMED["1"]), "bmap": {}}, TP, "LookupError", NO_RUN),
    ({"rc": _rc(CLAIMED["1"])}, TP, "LookupError", NO_RUN),
    ({"bmap": {"m": 1}}, TP, "LookupError", NO_RUN),
    ({"rc": _rc("0c" * 64), "bmap": {"m": 1}}, TP, "ValueError", OTHER_ROOT),
    ({"rc": _rc("0c" * 64), "bmap": {"m": 1}}, {**TP, "roots": {"0": "0c" * 64}}, "ValueError", OTHER_ROOT),
], ids=["no finalized run", "an empty binding map", "no binding map", "no run at all", "another root", "a root the driver holds for no such rank"])
def test_a_rank_that_cannot_write_its_window_answers_with_the_error_and_leaves_its_traceback_there(tmp_path, state, tp, kind, why):
    request = {"dir": str(tmp_path / "windows")}
    assert SRT.rank_rows(SimpleNamespace(_tp2_commit={"rank": 1, **state}), request, tp) == {"rank": 1, "error": f"{kind}({why!r})"}
    text = (tmp_path / "windows" / "rank1" / "error.txt").read_text()
    assert text.startswith("Traceback (most recent call last):\n") and text.endswith(f"{kind}: {why}\n")
    assert sorted(p.name for p in (tmp_path / "windows").iterdir()) == ["rank1"]


def test_a_rank_whose_root_is_the_one_combined_and_that_holds_a_binding_map_goes_on_to_its_programs_of_record(tmp_path):
    """Neither refusal holds it, so its next read is its replay configuration, which this state does not have."""
    got = SRT.rank_rows(SimpleNamespace(_tp2_commit={"rank": 1, "rc": _rc(CLAIMED["1"]), "bmap": {"m": 1}}), {"dir": str(tmp_path)}, TP)
    assert got == {"rank": 1, "error": "KeyError('replay')"}


def test_a_ranks_answer_holds_its_window_as_written_or_the_first_400_characters_of_its_error(tmp_path, monkeypatch):
    window = {"run": "r", "population": 1, "partition_digest": "p", "registration_digest": "g", "seconds": 0.5}
    monkeypatch.setattr(SRT, "write", lambda st, request, tp: window)
    worker = SimpleNamespace(_tp2_commit={"rank": "3"})
    assert SRT.rank_rows(worker, {"dir": str(tmp_path)}, TP) == {"rank": 3, "window": window} and list(tmp_path.iterdir()) == []

    def fails(st: dict[str, Any], request: dict[str, Any], tp: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("x" * 500)

    monkeypatch.setattr(SRT, "write", fails)
    (tmp_path / "rank3").mkdir()
    got = SRT.rank_rows(worker, {"dir": str(tmp_path)}, TP)
    assert got == {"rank": 3, "error": "RuntimeError('" + "x" * 386} and len(got["error"]) == 400
    assert (tmp_path / "rank3" / "error.txt").read_text().endswith("RuntimeError: " + "x" * 500 + "\n")


# ------------------------------------------------------------------------------------------------------------- the verifier's side


@pytest.mark.parametrize("rank", [2, -1, "7"], ids=["the world's size", "before the first", "past it, as text"])
def test_the_verifiers_window_of_a_rank_outside_the_records_world_is_refused_before_anything_is_read(tmp_path, rank):
    with pytest.raises(ValueError) as refused:
        SRT.window_of({"world": "2"}, rank, tmp_path / "no-such-partition.json", None)
    assert str(refused.value) == f"rank {int(rank)} is not a rank of the world of 2"


# ------------------------------------------------------------------------------------------------------- a rank's sealed bundle

REPLAY = {"programs": {"0": "/record/a", "1": "/record/b"}, "of_record": {"1": "d1"}, "requests": [{"request_id": "r0"}]}
WINDOW = {"run": f"{ROOT.hex()}/rank1", "population": 9, "partition_digest": "pd", "registration_digest": "rd", "seconds": 1.5}


def _deferred(tmp_path: Path, monkeypatch, *, scheme: str = SRT.ROWS_SCHEME, bundle_root: str = ROOT.hex(), tree: dict[str, Any] | None = None,
              weights_root: str = "0d" * 64) -> tuple[Path, dict[str, Any], SimpleNamespace]:
    """A deferred Commit of a world of 2 as `from_bundle` reads it for rank 1, with every reader of its files stood in: the opened
    bundle, the record's hold on it (`commit_replay._of_record`), the record, the required manifest, the rank's Programs, the
    committed values' reader and the window's writer.  The Commit's dir, what each stand-in was called with, and the Programs."""
    d, bdir, calls = tmp_path / "commit", tmp_path / "bundle", {}
    (bdir / "weights").mkdir(parents=True)
    d.mkdir()
    (d / C2.DEFERRED).write_text(json.dumps({"world": 2}))
    (bdir / "weights" / "tree.json").write_text(json.dumps(tree or {}))
    ctx = {"tp_run_root": bundle_root, "replay": json.loads(json.dumps(REPLAY)), "label": "the row", "code": "c0de"}
    store, run = SimpleNamespace(commit_scheme=scheme), object()
    p = SimpleNamespace(progs={"r0": "P"}, arrivals={"r0": 0}, dirs={"r0": "/record/b"}, digs={"r0": "d1"}, attn_paths={"a": 1}, tp={"t": 1},
                        index=SimpleNamespace(prefix="model."), loaded={})

    def opened(path: Any) -> dict[str, Any]:
        calls["bundle"] = path
        return {"context": ctx, "path": str(bdir), "manifest_sha256": "5a" * 32}

    def of_record(out: Path, dd: dict[str, Any], b: dict[str, Any], pair: int, rank: int, *, awaiting: bool = True) -> tuple:
        calls["of_record"] = (out, dd, b["context"], pair, rank, awaiting)
        return {}, {"run_root": CLAIMED["1"], "weight_registration": {"root": weights_root}}, store, run, {"m": 1}

    def record(commit_dir: Any) -> dict[str, Any]:
        calls["record"] = commit_dir
        return {"tp_run_root": ROOT.hex(), "root_tag": "the tag"}

    def manifest(o: Any, context: dict[str, Any]) -> None:
        calls["manifest"] = (o.required_manifest, context)

    def load(cfg: dict[str, Any], rank: int, world: int, *, definitions: bool = True) -> SimpleNamespace:
        calls["load"] = (cfg, rank, world, definitions)
        return p

    def after_commit(out_dir: str, **kw: Any) -> dict[str, Any]:
        calls["window"] = (out_dir, kw)
        return {**WINDOW, "units": 3}

    for module, name, stand_in in ((RB, "open_bundle", opened), (CR, "_of_record", of_record), (SRT, "record", record), (CR, "_manifest", manifest),
                                   (RR, "load", load), (SRW, "after_commit", after_commit), (OC, "OpenedReader", lambda *a: ("opened", *a))):
        monkeypatch.setattr(module, name, stand_in)
    return d, calls, SimpleNamespace(p=p, ctx=ctx, store=store, run=run)


def test_a_bundle_whose_store_commits_under_another_scheme_gives_no_stand_in_and_the_record_is_not_read(tmp_path, monkeypatch):
    d, calls, _ = _deferred(tmp_path, monkeypatch, scheme="vllm-v1")
    with pytest.raises(CR.Refused) as refused:
        SRT.from_bundle(d, 1, "p.json", tmp_path / "w", weights=object())
    assert str(refused.value) == "rank 1: its store commits under vllm-v1, not rows-sha512: no stand-in names a SHA-512 root"
    assert sorted(calls) == ["bundle", "of_record"] and calls["bundle"] == RB.path(d, 0, rank=1)
    assert calls["of_record"][3:] == (0, 1, False) and not (tmp_path / "w").exists()


def test_a_bundle_of_another_tp_run_root_than_the_records_gives_no_stand_in_and_no_program_is_loaded(tmp_path, monkeypatch):
    d, calls, _ = _deferred(tmp_path, monkeypatch, bundle_root="ee" * 64)
    with pytest.raises(CR.Refused) as refused:
        SRT.from_bundle(d, "1", "p.json", tmp_path / "w", pair=2, weights=object())
    assert str(refused.value) == "pair 2: the bundle's TP run root eeeeeeeeeeeeeeee is not the record's 0001020304050607"
    assert sorted(calls) == ["bundle", "of_record", "record"] and calls["bundle"] == RB.path(d, 2, rank=1) and calls["record"] == d
    assert calls["of_record"][3:] == (2, 1, False)


@pytest.mark.parametrize("build_root", [None, "build"], ids=["the record's dirs", "a Build root"])
def test_a_stand_in_reads_the_ranks_programs_under_the_build_root_when_one_is_named_and_the_bundles_configuration_else(tmp_path, monkeypatch,
                                                                                                                      build_root):
    """The Build root's ``rank<r>/`` for every rank of the world, in place of the bundle's dirs alone; the bundle's own
    configuration is not written to."""
    d, calls, made = _deferred(tmp_path, monkeypatch)
    weights, root = object(), None if build_root is None else tmp_path / build_root
    got = SRT.from_bundle(d, 1, tmp_path / "p.json", tmp_path / "w", build_root=root, weights=weights, required_manifest="req.json",
                          workers=3, binding={"compiler": "c"}, transfer_gates=("g",))
    programs = REPLAY["programs"] if root is None else {"0": str(root / "rank0"), "1": str(root / "rank1")}
    assert calls["load"] == ({**REPLAY, "programs": programs}, 1, 2, False) and made.ctx["replay"] == REPLAY
    assert calls["manifest"] == ("req.json", made.ctx) and got == WINDOW
    tp = {"world": 2, "rank": 1, "tp_run_root": ROOT.hex(), "run_root": CLAIMED["1"], "root_tag": "the tag"}
    p = made.p
    assert calls["window"] == (str(tmp_path / "w"), {
        "partition_file": str(tmp_path / "p.json"), "programs": p.progs, "bmap": {"m": 1}, "committed": ("opened", made.store, made.run),
        "weights": weights, "manifest": None, "run_root": f"{ROOT.hex()}/rank1", "row": "the row", "arrivals": p.arrivals, "program_dirs": p.dirs,
        "digests": p.digs, "committer": {"from": "replay bundle", "pair": 0, "source": "c0de"}, "workers": 3, "tp": tp,
        "index": {"attention_paths": p.attn_paths, "tp": p.tp}, "reuse": (p.index,), "stand_in": SRT.stand_in(tp, "5a" * 32),
        "binding": {"compiler": "c"}, "transfer_gates": ("g",), "tables": None})


@pytest.mark.parametrize("tree, weights_root, why", [
    ({"names": [], "root": "0d" * 64}, "0d" * 64, "the bundle's weights tree names no scheme: only a rows-sha512 tree is read"),
    ({"scheme": "vllm-v1", "names": [], "root": "0d" * 64}, "0d" * 64, "the bundle's weights tree is vllm-v1's: only a rows-sha512 tree is read"),
    ({"scheme": "rows-sha512", "names": [], "root": "0d" * 32}, "0d" * 32,
     "the committed weights root is 32 bytes, not a rows-sha512 root: only a rows-sha512 registration is read"),
], ids=["a tree that names no scheme", "another scheme's tree", "a committed root of another size"])
def test_a_bundle_whose_weights_tree_is_not_the_committed_roots_gives_no_stand_in_and_no_window_is_written(tmp_path, monkeypatch, tree,
                                                                                                           weights_root, why):
    """With no ``weights`` handed in, the tree is held to the rank's committed weights root before any tensor opens."""
    d, calls, _ = _deferred(tmp_path, monkeypatch, tree=tree, weights_root=weights_root)
    with pytest.raises(CR.Refused) as refused:
        SRT.from_bundle(d, 1, "p.json", tmp_path / "w")
    assert str(refused.value) == f"rank 1: the bundle's weights: {why}" and isinstance(refused.value.__cause__, ValueError)
    assert "window" not in calls and "load" in calls and not (tmp_path / "w").exists()


# ------------------------------------------------------------------------------------------------------------------- the command


CHECKPOINT = ["--checkpoint-dir", "ck", "--checkpoints-manifest", "pins.json", "--model", "org/model", "--revision", "r1"]


@pytest.mark.parametrize("more, ranks, index, tables", [
    ([], [0, 1, 2], None, None),
    (["--ranks", "2,0", *CHECKPOINT], [2, 0],
     {"checkpoints_manifest": "pins.json", "model": "org/model", "revision": "r1", "verify_shards": True}, None),
    (["--checkpoint-dir", "ck", "--no-verify-shards", "--tables", "t{rank}.json", "--tables-key", "key"], [0, 1, 2],
     {"checkpoints_manifest": None, "model": None, "revision": None, "verify_shards": False}, {"key": "key", "registration": "t{rank}.json"}),
    (["--tables-key", "key"], [0, 1, 2], None, None),
], ids=["every rank, no checkpoint", "two ranks in the order asked, a pinned checkpoint", "shards not hashed, tables", "a tables key and no tables"])
def test_the_command_opens_the_checkpoint_once_and_hands_it_to_each_rank_it_was_asked_for(tmp_path, monkeypatch, capsys, more, ranks, index, tables):
    asked: list[tuple] = []
    made: list[tuple[str, dict[str, Any]]] = []

    class Index:
        """`weights_of_record.CheckpointIndex`, which reads nothing here: what it was made with."""

        def __init__(self, ckpt_dir: str, **kw: Any) -> None:
            made.append((ckpt_dir, kw))

    def from_bundle(commit_dir: Any, rank: int, partition: Any, out_dir: Any, **kw: Any) -> dict[str, Any]:
        asked.append((commit_dir, rank, partition, out_dir, kw))
        return {"population": 5 + rank, "run": "ab" * 40, "registration_digest": "cd" * 32}

    monkeypatch.setattr(WR, "CheckpointIndex", Index)
    monkeypatch.setattr(SRT, "record", lambda d: {"world": 3, "of": d})
    monkeypatch.setattr(SRT, "from_bundle", from_bundle)
    o = cli.parse("tp-serving-rows", ["commit-out", "--partition", "p.json", "--out", str(tmp_path / "w"), "--build-root", "b",
                                      "--required-manifest", "req.json", "--workers", "2", *more], environ={})
    assert SRT.main(o) == 0
    assert made == ([] if index is None else [("ck", index)])
    assert [a[:4] for a in asked] == [("commit-out", r, "p.json", tmp_path / "w" / f"rank{r}") for r in ranks]
    for *_, kw in asked:
        ck = kw.pop("checkpoint")
        assert (ck is None) if index is None else isinstance(ck, Index)
        assert kw == {"build_root": "b", "required_manifest": "req.json", "workers": 2, "tables": tables}
    assert capsys.readouterr().out == "".join(f"[tp-serving-rows] rank {r}: stand-in window of {5 + r} units, run abababababababab rank {r}, "
                                              "registration cdcdcdcdcdcdcdcd\n" for r in ranks)
    assert not (tmp_path / "w").exists()
