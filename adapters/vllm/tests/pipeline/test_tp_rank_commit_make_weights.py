"""`verity_vllm.pipeline.tp.rank_worker`: the weights a rank registers when it makes its committer
(`TP2CaptureWorkerExtension.tp2_commit_make`), and its pin of them.  The rank registers the tensors the loader's post-load
hook keeps, applies the weight negatives the driver names for it, registers its weights as loaded, and compares every live
parameter with the digest composed from the pinned checkpoint by its own Program's shard map.  A record that cannot be built
is unpinned, never a pass, and a record of unpinned shards or of another Program fails by name.
CPU-only: `tests.check.test_weights_of_record`'s checkpoint and rank Programs, a model that holds the shards rank 1 of 2
serves, and `test_tp_rank_commit_make`'s stand-in committer, here one that keeps the weights as it registered them."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")

import numpy as np
import torch

from tests.check import test_weights_of_record as OF_RECORD
from tests.pipeline import test_tp_rank_commit_make as MAKE
from verity_vllm.check import weights_of_record as WR
from verity_vllm.pipeline.tp import rank_worker as RW

checkpoint = OF_RECORD.checkpoint

TP = {"world": 2, "rank": 1}
DIGEST, OTHER_DIGEST = "d" * 128, "e" * 128
DOWN, NORM = "model.layers.0.mlp.down_proj.weight", "model.norm.weight"
UNPINNED = MAKE.UNPINNED


class _Keeping(MAKE._Committer):
    """A committer that keeps every weight's bytes as they were when the rank registered them."""

    def register_weights(self) -> dict[str, Any]:
        self.words = {n: _sha(p) for n, p in self.model.named_parameters()}
        return super().register_weights()


def committer(engine: Any, run_id: str) -> _Keeping:
    return _Keeping(engine, run_id)


@pytest.fixture(autouse=True)
def _facts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(RW, "_tp_facts", lambda worker: MAKE.FACTS)


def _sha(x: Any) -> str:
    raw = x.detach().contiguous().view(torch.uint8).numpy() if isinstance(x, torch.Tensor) else np.ascontiguousarray(x)
    return hashlib.sha256(raw.tobytes()).hexdigest()


def _model(served: dict[str, np.ndarray]) -> torch.nn.Module:
    """A model whose parameters are ``served``, by name: bf16 words, and the kv scale's f32."""
    root = torch.nn.Module()
    for name, words in served.items():
        *path, leaf = name.split(".")
        mod = root
        for part in path:
            if part not in mod._modules:
                mod.add_module(part, torch.nn.Module())
            mod = mod._modules[part]
        t = torch.from_numpy(np.ascontiguousarray(words).copy())
        mod.register_parameter(leaf, torch.nn.Parameter(t.view(torch.bfloat16 if words.dtype == np.uint16 else torch.float32),
                                                        requires_grad=False))
    return root


def _program_dir(d: Path, tp: dict[str, int] | None = TP, digest: str = DIGEST) -> Path:
    """A rank's Program dir: the weights struct `tests.check.test_weights_of_record` declares for ``tp``."""
    prog, _ = OF_RECORD._program(tp=tp)
    d.mkdir(parents=True)
    with gzip.open(d / "instances.json.gz", "wt") as f:
        json.dump(prog["instances"] | {"digest": digest}, f)
    (d / "result.json").write_text(json.dumps(prog["result"] | {"program_digest": digest}))
    return d


def _shards(tensors: dict, rank: int = 1) -> dict[str, np.ndarray]:
    """What ``rank`` of 2 serves of the checkpoint."""
    tp = {"world": 2, "rank": rank}
    return OF_RECORD._served(tensors, OF_RECORD._program(tp=tp)[1], tp=tp)


@pytest.fixture
def served(checkpoint, tmp_path) -> dict[str, Any]:
    """Rank 1 of 2 serving its own shards of the checkpoint, and what the driver hands it to pin them."""
    ckpt, manifest, tensors = checkpoint
    program = _program_dir(tmp_path / "rank1" / "build_request")
    return {"model": _model(_shards(tensors)), "tensors": tensors, "program": program,
            "pin": {"checkpoint_dir": str(ckpt), "checkpoints_manifest": str(manifest), "program_dirs": {"1": str(program)}}}


def _made(model: torch.nn.Module, **kw: Any) -> tuple[dict[str, Any], dict[str, Any], Any]:
    """(the rank's answer, the state it keeps, its committer)."""
    rank = MAKE._Rank(model)
    answer = rank.tp2_commit_make(kind=f"plugin:{__name__}:committer", run_id="run", cp={}, case="case", engine_kwargs={}, hidden_so=None,
                                  hidden_ns=None, envelope=MAKE.ENVELOPE, **kw)
    return answer, rank._tp2_commit, rank._tp2_commit["com"]


# ------------------------------------------------------------------------------------------------ what the rank registers

@pytest.mark.parametrize("names, registered", [({"1": ["proj.kept"], "0": ["proj.other"]}, ["proj.kept"]), ({"0": ["proj.kept"]}, []),
                                               ({"1": None}, []), ({"1": []}, []), ({}, []), (None, [])])
def test_the_ranks_post_load_tensors_are_parameters_before_its_weights_are_registered(names, registered):
    model = MAKE._Model()
    model.proj.kept = torch.arange(3, dtype=torch.float32)
    answer, _, com = _made(model, post_load_tensors=names)
    assert answer["post_load_tensors"] == registered
    assert [n for n, _ in model.named_parameters()] == ["embed.weight", "proj.weight"] + registered
    assert sorted(com.words) == sorted(["embed.weight", "proj.weight"] + registered)
    assert answer["weight_registration"] == {"root": MAKE.ROOT, "tensors": 2 + len(registered)}


def test_a_post_load_tensor_the_model_does_not_hold_is_refused():
    with pytest.raises(TypeError, match="the Build registers the post-load tensor 'proj.kept' as a Parameter"):
        _made(MAKE._Model(), post_load_tensors={"1": ["proj.kept"]})


def test_the_weights_are_registered_as_loaded_and_the_committer_keeps_the_served_model():
    model, loaded = MAKE._Model(), torch.nn.Linear(2, 1, bias=False)
    asked = []

    def as_loaded(served: Any, *, world: int, rank: int) -> tuple[Any, dict[str, Any]]:
        asked.append((served, world, rank))
        return loaded, {"experts_as_loaded": 4}

    answer, st, com = _made(model, weights_as_loaded=as_loaded)
    assert asked == [(model, 2, 1)] and answer["weights_as_loaded"] == {"experts_as_loaded": 4}
    assert com.registered == [loaded] and sorted(com.words) == ["weight"]
    assert com.model is model and st["weights"] is loaded
    assert answer["weight_registration"] == {"root": MAKE.ROOT, "tensors": 1}


def test_without_a_loader_the_served_model_is_what_is_registered():
    model = MAKE._Model()
    answer, st, com = _made(model)
    assert answer["weights_as_loaded"] is None and com.registered == [model] and st["weights"] is model and com.model is model


# ------------------------------------------------------------------------------------------------ the flipped weight

@pytest.mark.parametrize("flip, index, after", [
    ({"rank": 1, "param": NORM, "index": 3, "value": 2.5}, 3, 2.5),
    ({"rank": "1", "param": NORM, "index": 11, "value": -4.0}, 3, -4.0),          # past the end: around again
    ({"rank": 1, "param": NORM}, 0, 1.0),
])
def test_a_flipped_weight_changes_one_element_before_the_weights_are_registered(served, flip, index, after):
    model = served["model"]
    own = _shards(served["tensors"])
    before = dict(model.named_parameters())[NORM].clone()
    answer, _, com = _made(model, flip_weight=flip)
    now = dict(model.named_parameters())[NORM]
    assert answer["flip_weight"] == {"rank": 1, "param": NORM, "index": index, "before": float(before[index]), "after": after}
    assert float(now[index]) == after != float(before[index])
    assert [i for i in range(8) if now[i] != before[i]] == [index]
    assert com.words[NORM] == _sha(now) != _sha(own[NORM])
    assert {n for n in own if com.words[n] != _sha(own[n])} == {NORM}


@pytest.mark.parametrize("flip", [None, {"rank": 0, "param": NORM, "value": 2.5}, {"param": NORM, "value": 2.5}, [1, NORM]])
def test_a_flip_that_does_not_name_this_rank_changes_nothing(served, flip):
    answer, _, com = _made(served["model"], flip_weight=flip)
    own = _shards(served["tensors"])
    assert answer["flip_weight"] is None and com.words == {n: _sha(a) for n, a in own.items()}


def test_a_flip_of_a_parameter_the_rank_does_not_hold_is_refused(served):
    with pytest.raises(RuntimeError, match="rank 1: --flip-weight names no parameter 'model.absent.weight' on this rank"):
        _made(served["model"], flip_weight={"rank": 1, "param": "model.absent.weight"})


def test_a_flipped_weight_fails_the_pin_by_its_name(served):
    answer, st, _ = _made(served["model"], flip_weight={"rank": 1, "param": NORM, "index": 3, "value": 2.5}, weights_pin=served["pin"])
    pin = answer["weights_pin"]
    assert pin["result"] is False and [m["field"] for m in pin["mismatched"]] == ["model." + NORM] and pin["missing"] == []
    assert st["weights_pin"] == pin


# ------------------------------------------------------------------------------------------------ another rank's shard

def test_a_rank_told_to_serve_another_ranks_shard_serves_it_and_fails_the_pin_by_its_name(served):
    model = served["model"]
    own, other = _shards(served["tensors"]), _shards(served["tensors"], rank=0)
    answer, _, com = _made(model, serve_shard_from={"1": [{"param": DOWN, "from_rank": 0}], "0": [{"param": DOWN, "from_rank": 1}]},
                           weights_pin=served["pin"])
    assert _sha(other[DOWN]) != _sha(own[DOWN])
    assert answer["serve_shard_from"] == [{
        "rank": 1, "param": DOWN, "from_rank": 0, "shard_rule": "rank 0/2: dim-1 slice", "sources": [DOWN],
        "served_before_sha256": _sha(own[DOWN]), "served_after_sha256": _sha(other[DOWN]), "own_shard_of_record_sha256": _sha(own[DOWN]),
        "own_equals_before": True, "after_equals_before": False}]
    assert _sha(dict(model.named_parameters())[DOWN]) == com.words[DOWN] == _sha(other[DOWN])
    assert {n for n in own if com.words[n] != _sha(own[n])} == {DOWN}
    pin = answer["weights_pin"]
    assert pin["result"] is False and [m["field"] for m in pin["mismatched"]] == ["model." + DOWN]
    assert pin["all_shards_pinned"] is True and pin["compared"] == 13


@pytest.mark.parametrize("shards", [None, {"0": [{"param": DOWN, "from_rank": 1}]}, {"1": []}, {"1": None}, [{"param": DOWN, "from_rank": 0}]])
def test_a_shard_swap_that_does_not_name_this_rank_changes_nothing(served, shards):
    answer, _, com = _made(served["model"], serve_shard_from=shards, weights_pin=served["pin"])
    assert answer["serve_shard_from"] is None and answer["weights_pin"]["result"] is True
    assert com.words == {n: _sha(a) for n, a in _shards(served["tensors"]).items()}


@pytest.mark.parametrize("swap, why", [
    ({"param": "model.absent.weight", "from_rank": 0}, "rank 1: --serve-shard-from names no parameter 'model.absent.weight' on this rank"),
    ({"param": DOWN, "from_rank": 1}, "rank 1: --serve-shard-from from_rank 1 is not another rank of world 2"),
    ({"param": DOWN, "from_rank": 2}, "rank 1: --serve-shard-from from_rank 2 is not another rank of world 2"),
    ({"param": DOWN, "from_rank": -1}, "rank 1: --serve-shard-from from_rank -1 is not another rank of world 2"),
    ({"param": NORM, "from_rank": 0}, f"rank 1: {NORM} is replicated on every rank"),
])
def test_a_shard_swap_that_names_no_other_ranks_shard_is_refused(served, swap, why):
    with pytest.raises(RuntimeError) as refused:
        _made(served["model"], serve_shard_from={"1": [swap]}, weights_pin=served["pin"])
    assert str(refused.value).startswith(why)
    assert _sha(dict(served["model"].named_parameters())[DOWN]) == _sha(_shards(served["tensors"])[DOWN])


@pytest.mark.parametrize("pin", [None, {}, {"program_dirs": {"1": "somewhere"}}, "the checkpoint"])
def test_a_shard_swap_without_the_pinned_checkpoint_is_refused(served, pin):
    with pytest.raises(RuntimeError, match="rank 1: --serve-shard-from needs --checkpoint-dir"):
        _made(served["model"], serve_shard_from={"1": [{"param": DOWN, "from_rank": 0}]}, weights_pin=pin)


# ------------------------------------------------------------------------------------------------ the pin

def test_a_rank_that_serves_its_shards_of_the_pinned_checkpoint_passes_its_pin(served):
    answer, st, _ = _made(served["model"], weights_pin=served["pin"], program_digest={"1": DIGEST, "0": OTHER_DIGEST})
    pin = answer["weights_pin"]
    assert pin["result"] is True and pin["why"] is None and (pin["mismatched"], pin["missing"], pin["unpinned"]) == ([], [], [])
    assert (pin["rank"], pin["program_dir"], pin["program_digest"]) == (1, str(served["program"]), DIGEST)
    assert pin["w2_program_of_record"] == {"built_in_process_from": str(served["program"]), "record_program_digest": DIGEST,
                                           "rank_program_digest_of_record": DIGEST}
    assert (pin["compared"], pin["live_params"], pin["committed_weights_root"]) == (13, 12, MAKE.ROOT)
    assert pin["all_shards_pinned"] is True and pin["checkpoint_entry"] == {"repo": "test/tiny", "revision": "abc"}
    assert "result_before_w1" not in pin and "result_before_w2" not in pin
    assert st["weights_pin"] == pin and st["program_digest"] == DIGEST


def test_the_pin_is_of_the_weights_as_loaded(served):
    loaded = _model(_shards(served["tensors"]) | {NORM: _shards(served["tensors"])[NORM][::-1]})
    answer, _, _ = _made(served["model"], weights_pin=served["pin"], weights_as_loaded=lambda model, *, world, rank: (loaded, None))
    assert answer["weights_pin"]["result"] is False and [m["field"] for m in answer["weights_pin"]["mismatched"]] == ["model." + NORM]


@pytest.mark.parametrize("pin", [{}, {"program_dirs": {"1": "somewhere"}}, {"checkpoint_dir": ""}, {"checkpoint_dir": None}, None, "the checkpoint"])
def test_without_a_checkpoint_of_record_the_weights_are_unpinned(served, pin):
    answer, st, _ = _made(served["model"], weights_pin=pin)
    assert answer["weights_pin"] == UNPINNED == st["weights_pin"]


def _not_evaluated(why: BaseException) -> dict[str, Any]:
    return {"rank": 1, "result": None, "why": f"rank 1: weights of record not evaluated: {why!r}"[:400], "error": repr(why)[:300]}


@pytest.mark.parametrize("dirs", [{}, {"0": "rank0"}, {"1": ""}, {"1": None}, None])
def test_a_rank_given_no_program_dir_is_not_evaluated(served, dirs):
    answer, _, _ = _made(served["model"], weights_pin=served["pin"] | {"program_dirs": dirs})
    assert answer["weights_pin"] == _not_evaluated(RuntimeError("rank 1: no rank Program dir for the weights of record"))


def test_a_build_dir_of_several_request_programs_is_pinned_by_its_first(served, tmp_path):
    build = tmp_path / "rank1_b2"
    first = _program_dir(build / "build_request_LP4_T2")
    _program_dir(build / "build_request_LP9_T1", digest=OTHER_DIGEST)
    (build / "build_request_empty").mkdir()
    answer, _, _ = _made(served["model"], weights_pin=served["pin"] | {"program_dirs": {"1": str(build)}}, program_digest=OTHER_DIGEST)
    pin = answer["weights_pin"]
    # the driver's digest names the row's workload, not this component: W2 is read on a dir that is itself a Program
    assert (pin["result"], pin["program_dir"], pin["program_digest"]) == (True, str(first), DIGEST) and "result_before_w2" not in pin
    assert pin["w2_program_of_record"] == {"built_in_process_from": str(first), "record_program_digest": DIGEST,
                                           "rank_program_digest_of_record": OTHER_DIGEST}


def test_a_dir_that_holds_no_program_is_not_evaluated(served, tmp_path):
    (tmp_path / "bare" / "build_request_empty").mkdir(parents=True)
    answer, _, _ = _made(served["model"], weights_pin=served["pin"] | {"program_dirs": {"1": str(tmp_path / "bare")}})
    assert answer["weights_pin"] == _not_evaluated(RuntimeError(f"rank 1: no rank Program (instances.json.gz) under {tmp_path / 'bare'}"))


@pytest.mark.parametrize("tp", [{"world": 2, "rank": 0}, {"world": 4, "rank": 1}, None])
def test_a_record_built_from_another_ranks_program_is_not_evaluated(served, tmp_path, tp):
    other = _program_dir(tmp_path / "other" / "build_request", tp=tp)
    answer, _, _ = _made(served["model"], weights_pin=served["pin"] | {"program_dirs": {"1": str(other)}})
    built_for = tp or {"world": 1, "rank": 0}                 # a Program that states no rank is the one rank of a world of 1
    assert answer["weights_pin"] == _not_evaluated(RuntimeError(
        f"rank 1: weights of record built from a Program of tp={built_for}, not this rank's (world 2, rank 1)"))


def test_shards_the_manifest_does_not_pin_fail_the_pin_whatever_the_bytes_say(served, tmp_path):
    ckpt = served["pin"]["checkpoint_dir"]
    unpinning = tmp_path / "unpinning.json"
    unpinning.write_text(json.dumps({"checkpoints": [{"repo": "test/tiny", "revision": "abc", "local_path": ckpt,
                                                      "files": {"model.safetensors": {"sha256": "0" * 64}}}]}))
    answer, _, _ = _made(served["model"], weights_pin=served["pin"] | {"checkpoints_manifest": str(unpinning)})
    pin = answer["weights_pin"]
    assert pin["result"] is False and pin["all_shards_pinned"] is False and pin["mismatched"] == []
    assert pin["why"].startswith("rank 1: checkpoint shards NOT pinned by manifests/checkpoints.json (all_shards_pinned=False): the record is "
                                 "over the files at the checkpoint dir, not the checkpoint of record (W1); ")
    assert len(pin["why"]) <= 400 and "result_before_w1" in pin and "result_before_w2" not in pin


def test_the_rank_fails_shards_the_manifest_does_not_pin_itself_whatever_the_check_answers(served, tmp_path, monkeypatch):
    ckpt, check = served["pin"]["checkpoint_dir"], WR.check
    unpinning = tmp_path / "unpinning.json"
    unpinning.write_text(json.dumps({"checkpoints": [{"repo": "test/tiny", "revision": "abc", "local_path": ckpt,
                                                      "files": {"model.safetensors": {"sha256": "0" * 64}}}]}))
    monkeypatch.setattr(WR, "check", lambda record, live: check(record, live) | {"result": True, "why": "the check's own words"})
    answer, _, _ = _made(served["model"], weights_pin=served["pin"] | {"checkpoints_manifest": str(unpinning)})
    pin = answer["weights_pin"]
    assert (pin["result"], pin["result_before_w1"], pin["all_shards_pinned"]) == (False, True, False)
    assert pin["why"].endswith("not the checkpoint of record (W1); the check's own words")


@pytest.mark.parametrize("digest", [OTHER_DIGEST, {"1": OTHER_DIGEST, "0": DIGEST}])
def test_a_record_of_another_program_than_the_ranks_of_record_fails_the_pin(served, digest):
    answer, _, _ = _made(served["model"], weights_pin=served["pin"], program_digest=digest)
    pin = answer["weights_pin"]
    assert (pin["result"], pin["result_before_w2"], pin["mismatched"]) == (False, True, [])
    assert pin["why"] == "rank 1: weights of record built from Program dddddddddddddddd != this rank's Program of record eeeeeeeeeeeeeeee (W2)"
    assert pin["w2_program_of_record"]["rank_program_digest_of_record"] == OTHER_DIGEST


def test_a_rank_given_no_program_digest_is_pinned_on_its_weights_alone(served):
    answer, _, _ = _made(served["model"], weights_pin=served["pin"])
    pin = answer["weights_pin"]
    assert pin["result"] is True and pin["w2_program_of_record"]["rank_program_digest_of_record"] is None


def test_the_drivers_shard_pin_is_adopted_and_one_of_another_manifest_is_refused(served, tmp_path):
    ckpt, manifest = served["pin"]["checkpoint_dir"], served["pin"]["checkpoints_manifest"]
    taken = WR.record_shard_pin(ckpt, checkpoints_manifest=manifest)
    answer, _, _ = _made(served["model"], weights_pin=served["pin"] | {"shard_pin": taken})
    assert answer["weights_pin"]["result"] is True and answer["weights_pin"]["shards"] == taken["shards"]
    elsewhere = taken | {"manifest": str(tmp_path / "another.json")}
    answer, _, _ = _made(served["model"], weights_pin=served["pin"] | {"shard_pin": elsewhere})
    assert answer["weights_pin"] == _not_evaluated(ValueError(
        f"{ckpt}: the pin was taken against manifest {tmp_path / 'another.json'}, this index reads {manifest}"))


def test_the_manifest_entry_is_found_by_the_model_the_driver_names(served, tmp_path):
    ckpt, manifest = served["pin"]["checkpoint_dir"], served["pin"]["checkpoints_manifest"]
    by_repo = tmp_path / "by_repo.json"
    entry = json.loads(Path(manifest).read_text())["checkpoints"][0]
    by_repo.write_text(json.dumps({"checkpoints": [entry | {"local_path": str(tmp_path / "moved")}]}))
    named = _made(served["model"], weights_pin=served["pin"] | {"checkpoints_manifest": str(by_repo), "model": "test/tiny"})[0]["weights_pin"]
    assert named["result"] is True and named["checkpoint_entry"] == {"repo": "test/tiny", "revision": "abc"}
    unnamed = _made(served["model"], weights_pin=served["pin"] | {"checkpoints_manifest": str(by_repo)})[0]["weights_pin"]
    assert unnamed["result"] is False and unnamed["checkpoint_entry"] is None and "result_before_w1" in unnamed
    assert Path(ckpt).is_dir()
