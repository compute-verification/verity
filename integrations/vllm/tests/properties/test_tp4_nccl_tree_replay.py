"""`tp4_nccl_tree_replay.replay` over a synthetic TP4 capture in the capture worker's layout: every all-reduce held to
`AllReduceNcclTree` (Boolean, word, model) is GO with a sealed record, the all-gather beside it is not a call; one received word off
the chain, bytes that are not their recorded sha256, a capture without batch invariance or on another GPU are NO-GO by name."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tests.properties import tp4_nccl_tree_replay as X  # noqa: E402
from verity.ml.definitions.registry import tp_allreduce as T  # noqa: E402
from verity.ml.definitions.registry import tp_allreduce_check as K  # noqa: E402
from verity_vllm.properties.record import DIR, PropertyRecord  # noqa: E402

N = 8192
NCCL_ENV = {**X.BI_PINS, "NCCL_P2P_DISABLE": "1", "NCCL_LAUNCH_MODE": "GROUP", "NCCL_NTHREADS": "1"}
#: step -> the rows M of its all-reduces; each step ends in the logits all-gather
STEPS = {0: (2, 2), 1: (1,)}


def _t(words: np.ndarray):
    return torch.from_numpy(np.ascontiguousarray(words).view(np.int16)).view(torch.bfloat16)


def _sha(t) -> str:
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _capture(tmp_path, *, flags=None, gpu=None, nccl_env=None, flip=None, stale=None):
    """The capture: per rank its meta, its rows and per step its partials and received words; `flip` = (step, k, rank) receives
    one word off the chain, `stale` = (step, k, rank) records a part sha256 that is not its bytes."""
    rng = np.random.default_rng(18)
    d = tmp_path / "capture"
    d.mkdir()
    (d / "topology.json").write_text(json.dumps({"tp": 4, "env_flags": flags if flags is not None else {"VLLM_BATCH_INVARIANT": "1"}}))
    (d / "versions.json").write_text(json.dumps({"gpu": gpu or {"name": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "cc": "12.0", "sms": 188}}))
    rows = {r: [] for r in range(X.WORLD)}
    for step, Ms in STEPS.items():
        dumps = {r: {"step": step, "part": [], "out": []} for r in range(X.WORLD)}
        for k, M in enumerate(Ms):
            parts = ((rng.standard_normal((X.WORLD, M, N)) * 0.05).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
            out = K.nccl_tree_rows(parts, (0, 1, 2, 3))
            for r in range(X.WORLD):
                recv = out.copy()
                if flip == (step, k, r):
                    recv[0, 5] ^= 1
                p, o = _t(parts[r]), _t(recv)
                dumps[r]["part"].append(p)
                dumps[r]["out"].append(o)
                rows[r].append({"rank": r, "step": step, "k": k, "kind": "all_reduce", "module": f"model.layers.{k}.mlp.down_proj",
                                "shape": [M, N], "dtype": "bfloat16", "out_shape": [M, N],
                                "part_sha256": "0" * 64 if stale == (step, k, r) else _sha(p), "out_sha256": _sha(o)})
        g = _t(np.zeros((1, 8), np.uint16))
        for r in range(X.WORLD):
            dumps[r]["part"].append(g)
            dumps[r]["out"].append(_t(np.zeros((1, 32), np.uint16)))
            rows[r].append({"rank": r, "step": step, "k": len(Ms), "kind": "all_gather", "module": "logits_processor", "shape": [1, 8],
                            "dtype": "bfloat16", "out_shape": [1, 32], "part_sha256": _sha(g), "out_sha256": ""})
            torch.save(dumps[r], d / f"coll.rank{r}.step{step}.pt")
    for r in range(X.WORLD):
        (d / f"coll.rank{r}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows[r]))
        facts = {"cuda_visible_devices": "2,3,6,7", "device": {"current": r, "name": "NVIDIA RTX PRO 6000 Blackwell Server Edition"},
                 "nccl_env": nccl_env if nccl_env is not None else NCCL_ENV, "parallel_config": {"disable_custom_all_reduce": True}}
        (d / f"coll_meta.rank{r}.json").write_text(json.dumps({"rank": r, "world_size": X.WORLD, "facts": facts}))
    return d


def _replay(tmp_path, *extra, **kw):
    out = tmp_path / "out"
    rc = X.main(["replay", "--capture", str(_capture(tmp_path, **kw)), "--out", str(out), *extra])
    return rc, json.loads((out / DIR / X.RECORD).read_text()), [json.loads(x) for x in (out / "calls.jsonl").read_text().splitlines()]


@pytest.mark.slow
def test_every_all_reduce_is_the_nccl_tree_on_bits_in_words_and_in_the_model(tmp_path, capsys) -> None:
    rc, doc, calls = _replay(tmp_path)
    assert rc == 0 and doc["ok"] is True and PropertyRecord(doc).problems() == [] and PropertyRecord(doc).name == "tp4_nccl_tree_capture"
    assert "TP4-NCCL-TREE GO" in capsys.readouterr().out
    assert doc["calls"] == 3 and doc["steps"] == [0, 1] and doc["shapes"] == {"[2, 8192]": 2, "[1, 8192]": 1}
    assert doc["rank_words"] == X.WORLD * 5 * N and doc["hash_mismatch"] == 0 and doc["ranks_differ"] == 0
    assert doc["specs"] == sorted(f"AllReduceNcclTree_v1{{WORLD=4,SIZE={m * N},CHAIN=(0,1,2,3)}}" for m in (1, 2))
    assert {o: doc["differ"][o] for o in ("boolean", "word", "model")} == {"boolean": 0, "word": 0, "model": 0}
    assert doc["differ"]["reversed"] > 0 and doc["differ"]["f32_once"] > 0
    assert doc["facts"]["device"] == list(T.RTX_PRO_6000) and doc["route"]["definition"] == "AllReduceNcclTree_v1"
    assert [(c["step"], c["k"]) for c in calls] == [(0, 0), (0, 1), (1, 0)]


@pytest.mark.parametrize("args,kw,why", [
    ((), {"flip": (1, 0, 2)}, "differ"),
    ((), {"stale": (0, 1, 3)}, "hash"),
    ((), {"flags": {}}, "uncovered"),
    ((), {"nccl_env": {**NCCL_ENV, "NCCL_ALGO": "allreduce:ring"}}, "NCCL environment"),
    ((), {"nccl_env": {k: v for k, v in NCCL_ENV.items() if k != "NCCL_P2P_DISABLE"}}, "uncovered"),
    ((), {"gpu": {"name": "NVIDIA H200", "cc": "9.0", "sms": 132}}, "not 4 RTX PRO 6000"),
    (("--no-word",), {}, "--no-word"),
], ids=["flipped-word", "stale-sha", "not-batch-invariant", "ring", "p2p-on", "h200", "no-word"])
def test_a_capture_off_the_route_or_off_the_chain_is_no_go_by_name(tmp_path, capsys, args, kw, why) -> None:
    rc, doc, _calls = _replay(tmp_path, *args, **kw)
    assert rc == 1 and doc["ok"] is False and PropertyRecord(doc).problems() == []
    assert "TP4-NCCL-TREE NO-GO" in capsys.readouterr().out
    if why == "differ":
        assert {o: doc["differ"][o] for o in ("boolean", "word", "model")} == {"boolean": 1, "word": 1, "model": 1}
        assert doc["ranks_differ"] == 1 and not doc["problems"]
    elif why == "hash":
        assert doc["hash_mismatch"] == 1 and not doc["problems"]
    elif why == "uncovered":
        assert len(doc["uncovered"]) == 3 and "no CAPTURED entry" in doc["uncovered"][0]["why"] and not doc["problems"]
    else:
        assert len(doc["problems"]) == 1 and why in doc["problems"][0], doc["problems"]
