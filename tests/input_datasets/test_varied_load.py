"""The varied generator against `serve_load.py --varied`, byte for byte. Deleted with serve_load.py."""
from __future__ import annotations

import importlib.util
import json
import tomllib
from pathlib import Path

import pytest

import input_datasets as D

ROOT = Path(__file__).resolve().parents[2]
ONSITE = "llama32_1b_int8_varied_4_16_24_4"


@pytest.mark.parametrize("over", [{}, {"seed": 0, "count": 8, "prompt_tokens": [16, 27], "rate": 2.0, "max_tokens": 16}])
def test_the_generator_is_serve_loads_varied_load(monkeypatch, over):
    """`serve_load.py --varied`'s requests, bodies and arrival times, byte for byte, at the onsite definition's parameters
    (served_verdicts.sh's LOAD=varied defaults) and at another seed, count, prompt range, rate and max_tokens, its
    /tokenize answered from the definition's token ids."""
    spec = importlib.util.spec_from_file_location("serve_load", ROOT / "adapters/vllm/tools/served_run/serve_load.py")
    sl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sl)
    g = tomllib.loads((D.DEFINITIONS / f"{ONSITE}.toml").read_text())["generated"]
    seed = over.get("seed", g["seed"])
    p = {**g["parameters"], **{k: v for k, v in over.items() if k != "seed"}}
    bos = [p["heads"][0][0]]
    answers = {"": bos, sl.CHAT_TAIL: bos + p["tail"], sl.VARIED_FILLER: bos + p["filler"],
               **{sl.CHAT_HEAD + q: h for q, h in zip(sl.QUESTIONS, p["heads"], strict=True)}}
    monkeypatch.setattr(sl, "tokenize", lambda port, model, text: answers[text])
    want = [{"at_s": q["at_s"], "body": q["body"]} for q in sl.varied(
        0, p["model"], p["count"], rate=p["rate"], prompt_tokens=tuple(p["prompt_tokens"]), max_tokens=p["max_tokens"], seed=seed)]
    got = D.GENERATORS[(g["generator"], g["version"])](seed, p)
    assert [json.dumps(r) for r in got] == [json.dumps(r) for r in want]
    if not over:
        assert got == list(D.open(ONSITE)) and [t for t, _ in D.open(ONSITE).timed("column", name="at_s")] == [
            q["at_s"] for q in want]
