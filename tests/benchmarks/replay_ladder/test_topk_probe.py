"""The DSA top-k probe on a stand-in for vLLM's prefill top-k, on CPU: what it records, what sort mode rewrites, a named
replacement, the call-by-call comparison, and where a replay puts its records."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

import ladder as L
import topk_probe as T


def stock(logits, ks, ke, topk, num_rows, s0, s1, k):
    """vLLM's kernel's contract: each row's ``k`` largest logits over [ks, ke), as column indices, in no fixed order."""
    for r in range(num_rows):
        row = logits[r, ks[r]:ke[r]]
        idx = torch.topk(row, min(k, row.numel())).indices + ks[r]
        topk[r, :idx.numel()] = idx.flip(0).to(topk.dtype)


@pytest.fixture
def ops(monkeypatch):
    vllm, custom = ModuleType("vllm"), ModuleType("vllm._custom_ops")
    custom.top_k_per_row_prefill = stock
    vllm._custom_ops = custom
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm._custom_ops", custom)
    return custom


def call(ops) -> torch.Tensor:
    """One prefill of three rows, the second over fewer keys than k, with ties among the logits."""
    logits = torch.tensor([[3.0, 1, 4, 1, 5, 9, 2, 6], [5.0, 3, 5, 8, 9, 7, 9, 3], [2.0, 3, 8, 4, 6, 2, 6, 4]])
    ks, ke = torch.tensor([0, 5, 1]), torch.tensor([8, 8, 8])
    topk = torch.full((3, 4), -1, dtype=torch.int32)
    ops.top_k_per_row_prefill(logits, ks, ke, topk, 3, 0, 0, 4)
    ops.top_k_per_row_prefill.flush()
    return topk


def recs(d: Path) -> list[dict]:
    return [json.loads(x) for x in (d / "dev0.jsonl").read_text().splitlines()]


def test_record_mode_records_each_call_and_leaves_its_rows(ops, tmp_path):
    want = torch.full((3, 4), -1, dtype=torch.int32)
    stock(torch.tensor([[3.0, 1, 4, 1, 5, 9, 2, 6], [5.0, 3, 5, 8, 9, 7, 9, 3], [2.0, 3, 8, 4, 6, 2, 6, 4]]),
          torch.tensor([0, 5, 1]), torch.tensor([8, 8, 8]), want, 3, 0, 0, 4)
    T.apply(mode="record", out=str(tmp_path / "rec"), real=None)
    assert torch.equal(call(ops), want)
    (r,) = recs(tmp_path / "rec")
    assert {k: r[k] for k in ("call", "rows", "cols", "k")} == {"call": 0, "rows": 3, "cols": 8, "k": 4}
    T.apply(mode="record", out=str(tmp_path / "again"), real=None)
    assert not (tmp_path / "again").exists()
    with pytest.raises(ValueError, match="record, sort"):
        T.apply(mode="shuffle", out=str(tmp_path), real=None)


def test_sort_mode_sorts_each_row_padding_last_and_keeps_its_set_and_logits(ops, tmp_path, monkeypatch):
    T.apply(mode="record", out=str(tmp_path / "rec"), real=None)
    call(ops)
    monkeypatch.setattr(ops, "top_k_per_row_prefill", stock)
    T.apply(mode="sort", out=str(tmp_path / "sort"), real=None)
    rows = call(ops).tolist()
    assert rows[1] == [5, 6, 7, -1]
    assert all(r[:sum(x >= 0 for x in r)] == sorted(x for x in r if x >= 0) for r in rows)
    (a,), (b,) = recs(tmp_path / "rec"), recs(tmp_path / "sort")
    assert (a["set"], a["logits"], a["order"]) == (b["set"], b["logits"], b["order"])


def test_a_named_replacement_is_the_top_k_called_and_recorded(ops, tmp_path, monkeypatch):
    (tmp_path / "det_topk.py").write_text(
        "def select(logits, ks, ke, topk, num_rows, s0, s1, k):\n"
        "    for r in range(num_rows):\n"
        "        lo, hi = int(ks[r]), int(ke[r])\n"
        "        keep = sorted(range(lo, hi), key=lambda j: (-float(logits[r, j]), j))[:k]\n"
        "        topk[r, :len(keep)] = topk.new_tensor(sorted(keep))\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    T.apply(mode="record", out=str(tmp_path / "det"), real="det_topk:select")
    assert call(ops).tolist() == [[2, 4, 5, 7], [5, 6, 7, -1], [2, 3, 4, 6]]
    assert len(recs(tmp_path / "det")) == 1


def write(d: Path, dev: str, rows: list[tuple]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / dev).write_text("".join(json.dumps({"call": i, "rows": n, "cols": c, "k": 4, "logits": lg, "set": s, "order": o}) + "\n"
                                 for i, (n, c, lg, s, o) in enumerate(rows)))


def test_compare_names_the_first_call_each_checksum_differs_at(tmp_path):
    same = [(2, 2, 1.0, 10, 20), (3, 3, 2.0, 11, 21)]
    write(tmp_path / "a", "dev0.jsonl", same + [(8, 8, 3.0, 12, 22), (8, 8, 4.0, 13, 23), (8, 8, 5.0, 14, 24)])
    write(tmp_path / "b", "dev0.jsonl", same + [(8, 8, 3.0, 12, 99), (8, 8, 4.0, 98, 97), (8, 8, 6.0, 96, 95)])
    write(tmp_path / "a", "dev1.jsonl", same)
    write(tmp_path / "b", "dev1.jsonl", [same[0], (4, 4, 2.0, 11, 21)])
    c = T.compare(tmp_path / "a", tmp_path / "b")
    d0 = c["dev0.jsonl"]
    assert {k: (d0[k]["first"], d0[k]["differing"]) for k in T.CHECKS} == {"order": (2, 3), "set": (3, 2), "logits": (4, 1)}
    assert d0["order"]["first_call"]["call"] == 2 and d0["selecting_calls"] == 3 and d0["first_shape_mismatch"] is None
    d1 = c["dev1.jsonl"]
    assert d1["first_shape_mismatch"] == 1 and all(d1[k]["first"] is None for k in T.CHECKS)


def test_a_replay_probes_its_top_k_into_its_own_directory(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(L, "register", lambda model: tmp_path / "ck")
    monkeypatch.setattr(L.R, "main", lambda argv: 0)
    monkeypatch.setattr(L.R, "engine", L.R.engine)
    monkeypatch.setattr(L.P, "apply", lambda **kw: None)
    monkeypatch.setattr(L.T, "apply", lambda **kw: seen.append(kw))
    monkeypatch.setenv("LADDER_TILINGS", "")
    monkeypatch.setenv("LADDER_TOPK_PROBE_OUT", "")
    monkeypatch.delenv("LADDER_TOPK_PROBE", raising=False)
    L.replay(SimpleNamespace(model="glm53", out=str(tmp_path), index=2, logprobs=False))
    assert seen == []
    monkeypatch.setenv("LADDER_TOPK_PROBE", "sort")
    monkeypatch.setenv("LADDER_TOPK_PROBE_REAL", "det_topk:select")
    L.replay(SimpleNamespace(model="glm53", out=str(tmp_path), index=2, logprobs=False))
    assert seen == [{"mode": "sort", "out": str(tmp_path / "replay-2" / "topk"), "real": "det_topk:select"}]
