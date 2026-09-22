"""Stopping-semantics relations (contract §7 last paragraph, §8 N14): EOS, truncation, cancellation.

These are relations over the *recorded* boundary values of a request (output ids, finish_reason,
stop_reason, the output-presence record, max_tokens, eos/stop token ids) and, where available,
the committed logits rows.  They encode exactly vLLM 0.28.0's `check_stop`
(vllm/v1/core/sched/utils.py):

    if num_output < min_tokens: continue
    if last == eos_token_id:            finish_reason = "stop",   stop_reason = None
    elif last in stop_token_ids:        finish_reason = "stop",   stop_reason = last
    elif num_tokens >= max_model_len or num_output >= max_tokens:
                                        finish_reason = "length", stop_reason = None

so EOS at the very last allowed position is "stop", not "length".  Cancellation ("abort") is a
client decision: the engine emits no final output for an aborted request, so the record carries
`client_finish_reason = "abort"` and the relation can only check *internal consistency* (tokens
present up to the abort, blanks after, no stop condition met earlier); nothing about the logits
predicts an abort, and the report says so.

Each checker returns (ok: bool, findings: list[str]).
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .select import greedy_select, independent_select


def check_output_presence(rec: dict[str, Any]) -> tuple[bool, list[str]]:
    """Slots 0..n-1 hold the output ids, slots n..max_tokens-1 are blank (None) after finish."""
    f: list[str] = []
    pres = rec.get("output_presence")
    ids = rec["output_token_ids"]
    mt = rec["max_tokens"]
    if pres is None:
        return False, ["no output_presence record"]
    if len(pres) != mt:
        f.append(f"presence record has {len(pres)} slots, max_tokens={mt}")
    n = len(ids)
    if pres[:n] != ids:
        f.append("presence slots before finish do not equal output ids")
    if any(x is not None for x in pres[n:]):
        f.append("non-blank slot after finish")
    if rec.get("finish_reason") in ("stop", "length", "abort") and n == 0 and rec.get("finish_reason") != "abort":
        f.append("finished with zero output tokens")
    return not f, f


def check_finish_reason(rec: dict[str, Any], eos_token_id: int, max_model_len: int | None = None) -> tuple[bool, list[str]]:
    """finish_reason/stop_reason must be exactly what check_stop derives from the ids."""
    f: list[str] = []
    ids = rec["output_token_ids"]
    sp = rec.get("sampling", {})
    stop_ids = set((rec.get("sampling_extra") or {}).get("stop_token_ids") or sp.get("stop_token_ids") or [])
    ignore_eos = bool(sp.get("ignore_eos") or (rec.get("sampling_extra") or {}).get("ignore_eos"))
    min_tokens = int(sp.get("min_tokens", 0) or 0)
    mt = rec["max_tokens"]
    fr = rec.get("client_finish_reason") or rec.get("finish_reason")
    sr = rec.get("stop_reason")
    plen = rec.get("prompt_len", 0)

    # derive
    derived_fr, derived_sr, derived_n = None, None, None
    for i, tok in enumerate(ids):
        n = i + 1
        if n < min_tokens:
            continue
        if not ignore_eos and tok == eos_token_id:
            derived_fr, derived_sr, derived_n = "stop", None, n
            break
        if tok in stop_ids:
            derived_fr, derived_sr, derived_n = "stop", tok, n
            break
        if (max_model_len is not None and plen + n >= max_model_len) or n >= mt:
            derived_fr, derived_sr, derived_n = "length", None, n
            break
    if fr == "abort":
        if derived_fr is not None:
            f.append(f"aborted request had already met a stop condition ({derived_fr} at n={derived_n})")
        if len(ids) >= mt:
            f.append("aborted request has max_tokens outputs")
        return not f, f
    if derived_fr is None:
        f.append("no stop condition met by the recorded ids, yet request finished")
        return False, f
    if derived_n != len(ids):
        f.append(f"stop condition met at n={derived_n} but {len(ids)} tokens recorded (engine would have stopped earlier/later)")
    if fr != derived_fr:
        f.append(f"finish_reason {fr!r} != derived {derived_fr!r}")
    if sr != derived_sr:
        f.append(f"stop_reason {sr!r} != derived {derived_sr!r}")
    return not f, f


def check_eos_selection(rec: dict[str, Any], logits_row_f32: np.ndarray, eos_token_id: int, pos: int) -> tuple[bool, list[str]]:
    """At the stopping position of an EOS-terminated request, sampler.select must yield EOS:
    greedy -> argmax == eos; stochastic -> the seeded Gumbel argmax == eos.  This is
    sampler.select specialised to the stop position; it makes the EOS relation executable."""
    sp = rec.get("sampling", {})
    T = sp.get("temperature", 1.0)
    if T < 1e-5:
        r = greedy_select(logits_row_f32)
    else:
        r = independent_select(logits_row_f32, T, sp.get("top_k", 0), sp.get("top_p", 1.0), rec.get("seed"), pos)
    ok = r.predicted == eos_token_id
    return ok, ([] if ok else [f"select at stop position predicts {r.predicted}, not eos {eos_token_id} (certified={r.certified})"])


def check_no_early_eos(rec: dict[str, Any], eos_token_id: int) -> tuple[bool, list[str]]:
    """No EOS/stop id may appear before the last recorded token (else the engine would have stopped)."""
    ids = rec["output_token_ids"]
    sp = rec.get("sampling", {})
    if sp.get("ignore_eos"):
        return True, []
    stop_ids = set((rec.get("sampling_extra") or {}).get("stop_token_ids") or [])
    bad = [i for i, t in enumerate(ids[:-1]) if t == eos_token_id or t in stop_ids]
    return not bad, ([] if not bad else [f"stop token at non-final positions {bad[:5]}"])


def summarize(records: Iterable[dict[str, Any]], eos_token_id: int, max_model_len: int | None = None) -> dict[str, Any]:
    out = {"attempted": 0, "ok": 0, "by_finish_reason": {}, "failures": []}
    for rec in records:
        fr = rec.get("client_finish_reason") or rec.get("finish_reason")
        b = out["by_finish_reason"].setdefault(fr, {"attempted": 0, "ok": 0})
        checks = [check_output_presence(rec), check_finish_reason(rec, eos_token_id, max_model_len),
                  check_no_early_eos(rec, eos_token_id)]
        ok = all(c[0] for c in checks)
        out["attempted"] += 1
        b["attempted"] += 1
        out["ok"] += ok
        b["ok"] += ok
        if not ok:
            out["failures"].append({"request_id": rec["request_id"], "index": rec.get("index"),
                                    "findings": [x for c in checks for x in c[1]]})
    return out
