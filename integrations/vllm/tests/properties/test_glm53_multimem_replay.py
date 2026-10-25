"""`glm53_multimem_replay` over a synthetic capture record in the capture's layout, at hidden 512 (`tp_allreduce.CAPTURED`
given a 512-wide multimem entry, FlashInfer's cap at 0): the kept words, the models' at the pinned tree, are GO on the word
Definition, the Boolean vectors and the model, the chain told apart and the probe decoded again; one kept word off the tree, a
launch the capture did not pass, pins that are not CAPTURED's, a launch on another route are NO-GO by name."""
from __future__ import annotations

import json

import numpy as np
import pytest

from tests.properties import glm53_multimem_replay as R
from tests.properties import glm53_tp8_allreduce_capture_gpu as X
from verity.ml.definitions.registry import tp_allreduce as T
from verity.ml.definitions.registry import tp_allreduce_check as K

H = 512
TREE = T.GLM53_H200_TREE
ENV = {"VLLM_BATCH_INVARIANT": "0", "VLLM_USE_DEEP_GEMM": "0", "VLLM_ALLREDUCE_USE_SYMM_MEM": "1", "VLLM_ALLREDUCE_USE_FLASHINFER": "1"}
#: (phase, M, seed): a served all-reduce, whose operands hold row 0's specials, and a probe of one period per owner
LAUNCHES = (("allreduce", 2, 9019), ("probe", 1, 9023))


@pytest.fixture(autouse=True)
def _narrow(monkeypatch):
    monkeypatch.setattr(T, "FI_MAX_BYTES", 0)
    monkeypatch.setattr(T, "CAPTURED", T.CAPTURED + (
        T.Captured("custom-off", H, "symm_mem_comm", 1, 4, "AllReduceMultimem_v1", {"TREES": (TREE,) * 8, "FTZ": False}, ()),))


def _run(tmp_path, *, flip=None, failed=None, pinned=TREE, custom=False):
    d = tmp_path / "run"
    (d / "words").mkdir(parents=True)
    case_list, cases, words = [], {}, []
    for phase, M, seed in LAUNCHES:
        c = {"phase": phase, "case": f"M{M}", "M": M, "key": f"{phase}/M{M}", "seed": seed, "spawn": "served", "H": H,
             "comm": None if custom else "symm_mem_comm"}
        ops = X.make_operands(tmp_path / "ops", c)
        out = K.multimem_rows(np.load(ops["ops"]["parts"]).reshape(X.WORLD, -1), (TREE,) * 8, False)
        if flip == phase:
            out[7] ^= 1
        np.save(d / "words" / f"{phase}_M{M}.npy", out.reshape(M, H))
        case_list.append(c)
        words.append(f"words/{phase}_M{M}.npy")
        cases[c["key"]] = {"ok": failed != phase, "totals": {"words": M * H, "mismatch": int(failed == phase), "rank_mismatch": 0}}
    doc = {"spawns": {"served": {"custom_all_reduce": custom, "env": ENV, "H": H}}, "case_list": case_list, "words": words,
           "probes": {"trees": [json.loads(json.dumps(pinned))] * 8, "ftz": False}, "cases": cases, "digest": "d" * 64}
    (d / X.RECORD).write_text(json.dumps(doc))
    return d


def _replay(tmp_path, **kw):
    out = tmp_path / "out"
    rc = R.main(["--run", str(_run(tmp_path, **kw)), "--out", str(out)])
    return rc, json.loads((out / R.OUT).read_text())


def test_the_kept_words_are_the_vectors_the_word_definition_and_the_model(tmp_path, capsys) -> None:
    rc, doc = _replay(tmp_path)
    assert rc == 0 and doc["ok"] is True and "GLM53-MULTIMEM-REPLAY GO" in capsys.readouterr().out
    assert doc["calls"] == 2 and doc["elements"] == 3 * H and doc["vectors"] == 3 * H // T.MULTIMEM_VECTOR
    assert {o: doc["differ"][o] for o in R.HELD} == {"boolean": 0, "word": 0, "model": 0} and doc["differ"]["chain"] > 0
    assert doc["specs"] == ["AllReduceMultimem_v1"] and [x["runs"] for x in doc["launches"]] == [1, 1]
    assert doc["launches"][1]["decoded"] == {"trees": True, "ftz": True} and "decoded" not in doc["launches"][0]


@pytest.mark.parametrize("kw,why", [
    ({"flip": "allreduce"}, "differ"),
    ({"failed": "probe"}, "the capture's verdict is not GO"),
    ({"pinned": T.chain(8)}, "not the record's pinned"),
    ({"custom": True}, "not AllReduceMultimem_v1"),
], ids=["flipped-word", "capture-failed", "other-pins", "other-route"])
def test_a_launch_off_the_tree_or_off_the_record_is_no_go_by_name(tmp_path, capsys, kw, why) -> None:
    rc, doc = _replay(tmp_path, **kw)
    assert rc == 1 and doc["ok"] is False and "GLM53-MULTIMEM-REPLAY NO-GO" in capsys.readouterr().out
    if why == "differ":
        assert {o: doc["differ"][o] for o in R.HELD} == {"boolean": 1, "word": 1, "model": 1} and not doc["problems"]
    else:
        assert doc["problems"] and all(why in p for p in doc["problems"]), doc["problems"]
