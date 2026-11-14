"""holdout.py (protected): a seed's workloads are the same workloads at every run, so a holdout can be made again from its
seed, and every request keeps to the generator's pools and off the reserved token ids."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from verity_vllm.properties import holdout

CASE, MODEL, VOCAB, SEED = "B0", "m/x", 1000, 20260917


def _generate(out: Path, seed: int = SEED) -> list[dict[str, Any]]:
    paths = holdout.generate(out, case=CASE, model=MODEL, vocab=VOCAB, n=3, seed=seed, n_requests=2)
    assert [p.parent for p in paths] == [out] * 3
    return [{"file": p.name, **json.loads(p.read_text())} for p in paths]


def test_a_seeds_workloads_are_the_ones_it_gave_before(tmp_path: Path) -> None:
    docs = _generate(tmp_path)
    assert [d["file"] for d in docs] == ["holdout_B0_0_1363946732.json", "holdout_B0_1_1302387300.json",
                                         "holdout_B0_2_1637491940.json"]
    got = [[(len(r["prompt_token_ids"]), r["max_tokens"], r["prompt_token_ids"][:3]) for r in d["requests"]] for d in docs]
    assert got == [[(31, 5, [164, 540, 449]), (65, 24, [620, 459, 844])],
                   [(33, 9, [748, 408, 384]), (96, 5, [402, 491, 656])],
                   [(128, 24, [204, 452, 621]), (63, 9, [321, 804, 220])]]
    assert docs[0]["requests"][0]["request_id"] == "8dfc6cb1-6c2e-5dcc-b730-0cad0e76770d"


def test_the_workloads_are_a_function_of_the_seed(tmp_path: Path) -> None:
    first, again, other = _generate(tmp_path / "a"), _generate(tmp_path / "b"), _generate(tmp_path / "c", SEED + 1)
    assert [d["sha256"] for d in first] == [d["sha256"] for d in again]
    assert not {d["sha256"] for d in first} & {d["sha256"] for d in other}


def test_every_request_keeps_to_the_pools_and_off_the_reserved_ids(tmp_path: Path) -> None:
    for d in _generate(tmp_path):
        assert d["prompt_lengths"] == [len(r["prompt_token_ids"]) for r in d["requests"]]
        assert len({r["request_id"] for r in d["requests"]}) == 2
        for r in d["requests"]:
            assert len(r["prompt_token_ids"]) in holdout.LENGTH_POOL and r["max_tokens"] in holdout.DECODE_POOL
            assert all(100 <= t < VOCAB - 100 for t in r["prompt_token_ids"])
