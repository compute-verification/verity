"""[R19 tp2, F-r19-tp2-1] tp2_match STRUCTURE row rule for MID-MODULE all-gathers (F-r19-moe-1: OlmoeAttention gathers q and k inside
self_attn, N tokens x 1024 per call).  Measured on row #70 (vyv-tp2, run r20260922-072919-fcdd): 66/66 sites named-equal at every one of
128 steps on both ranks, provenance 8448/8448, tokens equal -- and `structure.pass` False only because the R13 rule read the prefill's
all-gather rows as `== [#requests]` while the q/k gathers carry the prompt tokens (rows [8, 3755]).  CPU-only, torch-free.

  * positive: prefill all-gathers = the logits gather (#requests rows) + mid-module gathers (prompt-token rows) -> PASS
  * NEGATIVE: a mid-module gather with any other row count -> FAIL
  * NEGATIVE: no logits gather (only token-row gathers) -> FAIL (the sampled-row gather is required)
"""
from __future__ import annotations

import json
import types
from pathlib import Path

from verity_capture.tp import match as M

PROMPTS = [3, 5]                     # 2 requests, 8 prompt tokens
SITES = [("model.layers.0.self_attn", "all_gather", 16), ("model.layers.0.self_attn.o_proj", "all_reduce", 32), ("lm_head", "all_gather", 8)]


def _links(rank: int) -> dict:
    return {"rank": rank, "world": 2, "program_digest": f"d{rank}",
            "links": [{"k": k, "kind": kind, "shape": [1, w], "module_path": f"inner.model.{site} (Cls)"} for k, (site, kind, w) in enumerate(SITES)]}


def _recs(*, mid_rows: int | None = None, logits_rows: int | None = None) -> list[dict]:
    toks, nreq = sum(PROMPTS), len(PROMPTS)
    out = []
    for step in (0, 1):
        for k, (site, kind, w) in enumerate(SITES):
            if site == "lm_head":
                rows = (logits_rows if logits_rows is not None else nreq) if step == 0 else nreq
            else:
                rows = ((mid_rows if mid_rows is not None else toks) if step == 0 else nreq)
            out.append({"step": step, "k": k, "kind": kind, "shape": [rows, w], "module": site, "module_class": "Cls"})
    return out


def _run(tmp: Path, name: str, recs) -> dict:
    run = tmp / name
    cap = run / "capture"
    cap.mkdir(parents=True)
    for r in (0, 1):
        (cap / f"coll.rank{r}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in recs))
        (run / f"b{r}").mkdir()
        (run / f"b{r}" / "tp_links.json").write_text(json.dumps(_links(r)))
    (cap / "tokens.json").write_text(json.dumps({"requests": [{"request_id": f"r{i}", "prompt_token_ids": [1] * n, "generated_token_ids": [7, 8]}
                                                              for i, n in enumerate(PROMPTS)], "recorder": "test"}))
    M._check_main(types.SimpleNamespace(out=str(run), build_rank0=str(run / "b0"), build_rank1=str(run / "b1")))
    return json.load(open(run / "tp2_match.json"))["structure"]


def test_mid_module_all_gather_with_prompt_token_rows_passes(tmp_path):
    s = _run(tmp_path, "pos", _recs())
    assert s["pass"] is True and s["prefill_all_gather_rows_equal_requests"] is True
    assert s["prefill_all_gather_rows"]["observed"] == sorted({len(PROMPTS), sum(PROMPTS)})
    assert s["per_rank"]["0"]["prefill"]["sites_named_equal"] == len(SITES)


def test_mid_module_all_gather_with_other_rows_fails(tmp_path):
    s = _run(tmp_path, "badrows", _recs(mid_rows=sum(PROMPTS) + 1))
    assert s["pass"] is False and s["prefill_all_gather_rows_equal_requests"] is False


def test_missing_logits_all_gather_rows_fails(tmp_path):
    s = _run(tmp_path, "nologits", _recs(logits_rows=sum(PROMPTS)))       # every gather carries token rows: no sampled-row gather
    assert s["pass"] is False and s["prefill_all_gather_rows_equal_requests"] is False
