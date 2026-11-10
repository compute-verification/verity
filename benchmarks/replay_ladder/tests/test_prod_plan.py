"""The production-like plan (prod_plan.build) on a character tokenizer, and how the ladder serves a plan whose requests carry
their own sampling."""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import ladder as L
import prod_plan as PP


class CharTokenizer:
    """One token per character; the chat template wraps each message in its role's markers."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(map(chr, ids))

    def apply_chat_template(self, msgs: list[dict], add_generation_prompt: bool, tokenize: bool) -> str:
        return "".join(f"<{m['role']}>{m['content']}" for m in msgs) + "<assistant>"


def inputs(tmp: Path) -> tuple[Path, Path]:
    convs = []
    for i in range(400):
        turns = [{"from": ("human", "gpt")[j % 2], "value": f"turn {j} of conversation {i}, " + "x" * (20 + j)} for j in range(8)]
        convs.append({"id": str(i), "conversations": turns})
    sg = tmp / "sharegpt.json"
    sg.write_text(json.dumps(convs))
    c4 = tmp / "c4.json.gz"
    with gzip.open(c4, "wt") as f:
        for i in range(50):
            f.write(json.dumps({"text": f"document {i} " + "y" * 3000}) + "\n")
    return sg, c4


def test_the_plan_mixes_its_kinds_with_their_shares_lengths_and_sampling(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers",
                        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: CharTokenizer())))
    sg, c4 = inputs(tmp_path)
    out = tmp_path / "plan.jsonl"
    assert PP.main(["build", "--checkpoint", str(tmp_path), "--sharegpt", str(sg), "--c4", str(c4), "--out", str(out),
                    "--n", "200"]) == 0
    plan = [json.loads(x) for x in out.read_text().splitlines()]
    assert [r["i"] for r in plan] == list(range(200))
    kinds = {k: [r for r in plan if r["kind"] == k] for k in ("sharegpt", "multiturn", "long")}
    assert len(kinds["long"]) == 8 and 45 <= len(kinds["multiturn"]) <= 50
    assert all(PP.LONG[0] - 64 <= r["prompt_len"] <= PP.LONG[1] + 64 and r["max_tokens"] == PP.LONG_CAP for r in kinds["long"])
    assert all(r["prompt_len"] + r["max_tokens"] <= 2048 + 64 for r in kinds["sharegpt"])
    convs: dict[int, list[dict]] = {}
    for r in kinds["multiturn"]:
        convs.setdefault(r["conv"], []).append(r)
    for turns in convs.values():
        assert [t["turn"] for t in turns] == list(range(1, len(turns) + 1)) and len(turns) >= 2
        for a, b in zip(turns, turns[1:]):
            assert b["i"] - a["i"] >= PP.GAP[0] and b["prompt_ids"][:b["shared_with_previous_turn"]] == a["prompt_ids"][:b["shared_with_previous_turn"]]
            assert b["shared_with_previous_turn"] >= len(a["prompt_ids"]) - len("<assistant>")
    assert {json.dumps(r["sampling"]) for r in plan} == {'{"temperature": 0.0}', '{"temperature": 0.7}'}
    assert all(r["seeded"] == (r["sampling"]["temperature"] > 0) for r in plan)
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert sum(k["requests"] for k in meta["kinds"].values()) == 200
    at = L.S.corpus().traffic.arrivals(plan, 2.0)
    assert all(b > a for a, b in zip(at, at[1:])) and 0.7 * 100 < at[-1] < 1.3 * 100


def test_a_plan_request_carries_its_own_sampling_and_seed_and_stops_at_its_end(monkeypatch):
    monkeypatch.setattr(L.S, "corpus", lambda: SimpleNamespace(request_seed=lambda rid: len(rid)))
    real = lambda served, r, rid, sampling: {"from": "serve_tax", "ignore_eos": True}  # noqa: E731
    body = L.own_sampling(real)
    sampled = body("m", {"prompt_ids": [1, 2], "max_tokens": 9, "sampling": {"temperature": 0.7}, "seeded": True}, "abc", "greedy")
    assert (sampled["temperature"], sampled["seed"], sampled["max_tokens"], sampled["prompt"]) == (0.7, 3, 9, [1, 2])
    assert "ignore_eos" not in sampled and sampled["stream"]
    greedy = body("m", {"prompt_ids": [1], "max_tokens": 4, "sampling": {"temperature": 0.0}, "seeded": False}, "abc", "greedy")
    assert greedy["temperature"] == 0.0 and "seed" not in greedy
    assert body("m", {"prompt_ids": [1], "max_tokens": 4}, "abc", "greedy") == {"from": "serve_tax", "ignore_eos": True}


def test_kinds_counts_contexts_past_the_top_k_and_where_each_difference_starts(tmp_path, monkeypatch):
    d = tmp_path / "glm53"
    (d / "record" / L.LOAD / "rep-1").mkdir(parents=True)
    plan = [{"i": 0, "kind": "sharegpt", "prompt_len": 100, "seeded": False},
            {"i": 1, "kind": "multiturn", "prompt_len": PP.TOPK - 8, "seeded": True},
            {"i": 2, "kind": "long", "prompt_len": 3 * PP.TOPK, "seeded": False}]
    (d / "plan.jsonl").write_text("".join(json.dumps(r) + "\n" for r in plan))
    (d / "record" / L.LOAD / "rep-1" / "client.jsonl").write_text(
        "".join(json.dumps({"id": f"h{r['i']}", "i": r["i"]}) + "\n" for r in plan))
    served = {0: 50, 1: 20, 2: 10}
    monkeypatch.setattr(L.R, "window", lambda arm, load, rep: {"requests": [
        {"engine_id": f"cmpl-h{i}-0-0a1b2c3d", "served": [7] * n} for i, n in served.items()]})
    (d / "replay-1").mkdir()
    (d / "replay-1" / "replay.json").write_text(json.dumps({"requests_tokens_differing": 2, "bit_exact": False, "requests_differing": [
        {"request": "cmpl-h0-0-0a1b2c3d", "first_position": 30, "positions": 2},
        {"request": "cmpl-h2-0-0a1b2c3d", "first_position": 4, "positions": 6}]}))
    assert PP.main(["kinds", str(d)]) == 0
    k = json.loads((d / "kinds.json").read_text())
    assert {n: v["past_topk"] for n, v in k["kinds"].items()} == {"sharegpt": 0, "multiturn": 1, "long": 1}
    rows = k["replays"]["replay-1"]["differing"]
    assert [(r["kind"], r["context_at_first"], r["past_topk_at_first"]) for r in rows] == [
        ("sharegpt", 130, False), ("long", 3 * PP.TOPK + 4, True)]


def test_the_context_limit_replaces_serve_tax_s():
    assert L.max_model_len(("--max-model-len", "4096", "--seed", "0"), 40960) == ("--max-model-len", "40960", "--seed", "0")
