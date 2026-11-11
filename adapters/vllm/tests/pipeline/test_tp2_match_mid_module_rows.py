"""tp2_match STRUCTURE row rule for MID-MODULE all-gathers (OlmoeAttention gathers q and k inside self_attn, N tokens x 1024 per
call): a prefill step's all-gathers are the logits gather (#requests rows) and the q/k gathers, which carry the prompt tokens (rows
[8, 3755]), so a rule that reads every prefill all-gather as `== [#requests]` fails a run whose sites are all named-equal.  CPU-only, torch-free.

  * positive: prefill all-gathers = the logits gather (#requests rows) + mid-module gathers (prompt-token rows) -> PASS
  * NEGATIVE: a mid-module gather with any other row count -> FAIL
  * NEGATIVE: no logits gather (only token-row gathers) -> FAIL (the sampled-row gather is required)

The rows are those of the schedule the capture realized (`match.step_rows`): with arrivals (GLM-5.3 B8 mixed-arrivals) a step's tokens
are its prefills' prompts plus one per decoding request, and its sampled rows the requests it runs.

  * positive: two requests arriving at steps 0 and 1 -> PASS on their per-step rows
  * NEGATIVE: the same capture without the arrivals (every prompt at step 0) -> FAIL, the step named
"""
from __future__ import annotations

import json
import types
from pathlib import Path

from verity_vllm.pipeline.tp import match as M

PROMPTS = [3, 5]                     # 2 requests, 8 prompt tokens
SITES = [("model.layers.0.self_attn", "all_gather", 16), ("model.layers.0.self_attn.o_proj", "all_reduce", 32), ("lm_head", "all_gather", 8)]


def _links(rank: int) -> dict:
    return {"rank": rank, "world": 2, "program_digest": f"d{rank}",
            "links": [{"k": k, "kind": kind, "shape": [1, w], "module_path": f"inner.model.{site} (Cls)"} for k, (site, kind, w) in enumerate(SITES)]}


def _recs(*, mid_rows: int | None = None, logits_rows: int | None = None, rows: list[tuple[int, int]] | None = None) -> list[dict]:
    """Steps 0 and 1 of the two requests arriving together, or `rows`: (tokens, sampled) per step."""
    toks, nreq = sum(PROMPTS), len(PROMPTS)
    out = []
    for step, (n_tok, n_samp) in enumerate(rows or [(toks, nreq), (nreq, nreq)]):
        for k, (site, kind, w) in enumerate(SITES):
            if site == "lm_head":
                n = (logits_rows if logits_rows is not None else n_samp) if step == 0 else n_samp
            else:
                n = ((mid_rows if mid_rows is not None else n_tok) if step == 0 else n_tok)
            out.append({"step": step, "k": k, "kind": kind, "shape": [n, w], "module": site, "module_class": "Cls"})
    return out


def _run(tmp: Path, name: str, recs, first_steps: list[int] | None = None) -> dict:
    run = tmp / name
    cap = run / "capture"
    cap.mkdir(parents=True)
    for r in (0, 1):
        (cap / f"coll.rank{r}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in recs))
        (run / f"b{r}").mkdir()
        (run / f"b{r}" / "tp_links.json").write_text(json.dumps(_links(r)))
    tok = {"requests": [{"request_id": f"r{i}", "prompt_token_ids": [1] * n, "generated_token_ids": [7, 8]} for i, n in enumerate(PROMPTS)],
           "recorder": "test"}
    if first_steps:
        tok["timing"] = {"requests": {f"r{i}": {"first_step": s} for i, s in enumerate(first_steps)}}
    (cap / "tokens.json").write_text(json.dumps(tok))
    M._check_main(types.SimpleNamespace(out=str(run), build_rank0=str(run / "b0"), build_rank1=str(run / "b1")))
    return json.load(open(run / "tp2_match.json"))["structure"]


def test_mid_module_all_gather_with_prompt_token_rows_passes(tmp_path):
    s = _run(tmp_path, "pos", _recs())
    assert s["pass"] is True and s["per_rank"]["0"]["n_steps_rows_mismatching"] == 0
    assert s["per_rank"]["0"]["prefill"]["rows_all_gather"] == sorted({len(PROMPTS), sum(PROMPTS)})
    assert s["per_rank"]["0"]["prefill"]["sites_named_equal"] == len(SITES)


def test_mid_module_all_gather_with_other_rows_fails(tmp_path):
    s = _run(tmp_path, "badrows", _recs(mid_rows=sum(PROMPTS) + 1))
    assert s["pass"] is False and [b["step"] for b in s["per_rank"]["0"]["steps_rows_mismatching"]] == [0]


def test_missing_logits_all_gather_rows_fails(tmp_path):
    s = _run(tmp_path, "nologits", _recs(logits_rows=sum(PROMPTS)))       # every gather carries token rows: no sampled-row gather
    assert s["pass"] is False and s["per_rank"]["0"]["n_steps_rows_mismatching"] == 1


def test_rows_follow_the_realized_arrivals(tmp_path):
    # r0 (3 tokens) prefills at step 0; at step 1 r0 decodes while r1 (5 tokens) prefills; at step 2 r1 decodes alone
    arrived = [(3, 1), (1 + 5, 2), (1, 1)]
    s = _run(tmp_path, "arrivals", _recs(rows=arrived), first_steps=[0, 1])
    assert s["pass"] is True and s["step_rows"] == {"0": [3, 1], "1": [6, 2], "2": [1, 1]}
    s = _run(tmp_path, "together", _recs(rows=arrived))
    assert s["pass"] is False and [b["step"] for b in s["per_rank"]["0"]["steps_rows_mismatching"]] == [0, 1, 2]
