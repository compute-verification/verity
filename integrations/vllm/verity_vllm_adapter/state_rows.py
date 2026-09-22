"""State and identity rows R2 / R3 / R4 (contract §0/§9, W10): chunked prefill, prefix-cache hits,
induced preemption + recomputation on the eager-production profile.

Everything here is CPU-only glue around the common adapter (no vLLM / torch import):

* ``parse_engine_args`` / ``replay_engine_overrides``: declared ``LLM(...)`` overrides recorded in the
  production manifest (``engine_overrides``) and re-applied to the replay engine, so the replay goes
  through the same scheduler path (same chunk size, same KV budget) as the recorded production.
* ``build_shared_prefix_workload``: the R3 workload -- every 512/2048-token base prompt gets a partner
  that shares all but its last 64 tokens (aligned to the block size), a genuine prefix-cache HIT.
* Post-hoc checks over a capture: ``recompute_consistency`` (R4: every leaf a recompute chunk produced
  equals the leaf of the original computation of that identity), ``cached_binding_report`` (R3: every
  cached position of a hit request is bound to an existing leaf of its source request),
  ``compare_records`` (bit comparison of tokens and chosen logprobs between two production records).

Identity rules (docs/vllm-poc/state-identity.md):

* phase is a function of position: ``ph=prefill`` iff ``t < prompt_len`` (a recompute of a generated
  position after preemption is still ``ph=decode``); a replay unit whose positions straddle the prompt
  end carries ``phase="mixed"`` and ``recompute: true`` in the boundary's steps.
* a prefix-cache hit at position ``t`` of request B served from request A's KV rows is the identity
  ``r=A/t=t/ph=.../op=<attn write>`` -- valid iff A and B agree on tokens ``0..t`` (K/V at ``t`` depends on
  tokens ``0..t`` only); the boundary of B records ``cached_prefix: [{source_request_id: A, positions}]``.

    python -m verity_vllm_adapter.state_rows workload --base workload.json --out r3.json [--block-size 16]
    python -m verity_vllm_adapter.state_rows recompute-check <run_dir> --out recompute.json
    python -m verity_vllm_adapter.state_rows cached-check <run_dir> --out cached.json
    python -m verity_vllm_adapter.state_rows compare-records <label>=<production.jsonl> ... --out cmp.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

# --- declared engine overrides -----------------------------------------------------------------

# keys the replay owns (contract: replay restrictions); a production override never changes them
REPLAY_OWNED = ("enforce_eager", "enable_trace_replay", "compilation_config", "model", "tokenizer")


def _coerce(text: str) -> Any:
    low = text.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    if text[:1] in "[{":
        return json.loads(text)
    return text


def parse_engine_args(items: list[str] | None) -> dict[str, Any]:
    """``KEY=VALUE`` strings -> typed dict (ints, floats, booleans, JSON lists/objects)."""
    out: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--engine-arg expects KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        k = k.strip()
        if k in REPLAY_OWNED:
            raise SystemExit(f"--engine-arg {k}: not a declarable override (owned by the profile / the replay)")
        out[k] = _coerce(v)
    return out


def replay_engine_overrides(manifest: dict[str, Any], *, prefix_caching: bool = False) -> dict[str, Any]:
    """Production's declared overrides for the replay engine. ``enable_prefix_caching`` is dropped
    (the replay keeps it off) unless the replay opts in explicitly (R3, cross-request KV rule)."""
    out = {k: v for k, v in (manifest.get("engine_overrides") or {}).items() if k not in REPLAY_OWNED}
    out.pop("enable_prefix_caching", None)
    if prefix_caching:
        out["enable_prefix_caching"] = True
    return out


def replay_state_summary(cap_summary: dict[str, Any], trace_summary: dict[str, Any], overrides: dict[str, Any], *,
                         prefix_caching: bool) -> dict[str, Any]:
    """The state-row witnesses of a replay run (copied into replay.json)."""
    return {
        "engine_overrides_applied": overrides,
        "replay_prefix_caching": prefix_caching,
        "kv_cross_request_rule": cap_summary.get("kv_cross_request_rule"),
        "kv_reads_cross_request_resolved": cap_summary.get("kv_reads_cross_request_resolved"),
        "kv_reads_unresolved_under_own_request_rule": cap_summary.get("kv_reads_unresolved_under_own_request_rule"),
        "kv_reads_unresolved": cap_summary.get("kv_reads_unresolved"),
        "cached_positions_per_request": cap_summary.get("cached_positions_per_request"),
        "cached_sources_per_request": cap_summary.get("cached_sources_per_request"),
        "requests_with_recompute": cap_summary.get("requests_with_recompute"),
        "recompute_steps": cap_summary.get("recompute_steps"),
        "scheduler": {k: trace_summary.get(k) for k in ("engine_steps", "preemptions", "prefill_chunks_per_request",
                                                        "peak_concurrency_achieved", "max_tokens_in_a_step")},
    }


# --- R3 workload ---------------------------------------------------------------------------------


def build_shared_prefix_workload(base_path: str | Path, out_path: str | Path, *, block_size: int = 16, tail: int = 64,
                                 min_prompt_len: int = 512) -> dict[str, Any]:
    """From the frozen 16-request workload: keep every base request (sources / misses) and add, for
    each prompt of length >= ``min_prompt_len``, a partner whose prompt = source[:L - tail] + a
    ``tail``-token slice of a *different* base prompt. Sources come first in the request order.
    ``L - tail`` is a multiple of ``block_size`` so the shared prefix is exactly ``(L - tail) //
    block_size`` full blocks: the partner is a genuine prefix-cache hit of ``L - tail`` tokens."""
    from verity_vllm_sampler.workload import Request, Workload, base_request_id, load_workload

    wl = load_workload(str(base_path))
    reqs = list(wl.requests)
    donors = [r for r in reqs if r.prompt_len >= min_prompt_len]
    partners: list[Request] = []
    pairs: list[dict[str, Any]] = []
    idx = len(reqs)
    for i, src in enumerate(donors):
        L = src.prompt_len
        keep = L - tail
        if keep <= 0 or keep % block_size:
            raise SystemExit(f"{src.request_id}: L={L} tail={tail} not block aligned")
        other = donors[(i + 1) % len(donors)]
        # a tail from the middle of another prompt (never its aligned prefix -> no accidental second hit)
        off = block_size // 2 + (i * 7) % 16
        tail_ids = list(other.prompt_token_ids[off : off + tail])
        ids = list(src.prompt_token_ids[:keep]) + tail_ids
        assert len(ids) == L and ids != list(src.prompt_token_ids)
        partner = Request(
            request_id=base_request_id(wl.model, ids, src.max_tokens), index=idx, prompt_len=L, max_tokens=src.max_tokens,
            prompt_token_ids=ids, prompt_text="", text_roundtrip=False, corpus_seed=-1,
        )
        partners.append(partner)
        pairs.append({"source": src.request_id, "hit": partner.request_id, "prompt_len": L, "shared_tokens": keep,
                      "shared_blocks": keep // block_size, "expected_num_cached_tokens": keep})
        idx += 1
    all_reqs = reqs + partners
    seen = {tuple(r.prompt_token_ids) for r in all_reqs}
    assert len(seen) == len(all_reqs), "duplicate prompts"
    out = Workload(schema=wl.schema, model=wl.model, tokenizer_revision=wl.tokenizer_revision, seed=wl.seed, requests=all_reqs,
                   variants=wl.variants, seeds=wl.seeds)
    out.save(str(out_path))
    design = {
        "base_workload": str(base_path), "block_size": block_size, "tail_tokens": tail, "requests": len(all_reqs),
        "sources": len(donors), "hits": len(partners), "misses": len(all_reqs) - len(partners), "pairs": pairs,
        "order": "all base requests (sources and misses) first, then the hit partners",
    }
    Path(str(out_path) + ".design.json").write_text(json.dumps(design, indent=1) + "\n")
    return design


def build_tiny_shared_prefix_workload(tokenizer, model: str, out_path: str | Path, *, block_size: int = 16,
                                      source_len: int = 48, shared_blocks: int = 2, tail: int = 16, miss_len: int = 32,
                                      max_tokens: int = 4, seed: int = 20260907) -> dict[str, Any]:
    """The R3 workload at bundle scale (§16.8: a complete passing prefix-cache bundle under 100 MB on
    SmolLM2-135M): source A (``source_len`` tokens), hit B = A[:shared_blocks*block_size] + ``tail``
    fresh tokens (a genuine hit of exactly ``shared_blocks`` blocks), miss C (``miss_len`` tokens,
    long enough to serve as the third request of the N06 rebind variant); ``max_tokens`` each."""
    from verity_vllm_sampler.workload import SCHEMA, Request, Workload, _prompt_for, base_request_id

    keep = shared_blocks * block_size
    assert 0 < keep < source_len and tail > 0
    a_ids, a_text, a_rt = _prompt_for(tokenizer, seed, source_len)
    c_ids, c_text, c_rt = _prompt_for(tokenizer, seed + 1, miss_len)
    t_ids, _, _ = _prompt_for(tokenizer, seed + 2, tail + block_size)
    b_ids = list(a_ids[:keep]) + list(t_ids[block_size // 2 : block_size // 2 + tail])
    assert len(b_ids) == keep + tail and b_ids != list(a_ids[: len(b_ids)])
    reqs = [
        Request(base_request_id(model, a_ids, max_tokens), 0, len(a_ids), max_tokens, list(a_ids), a_text, a_rt, seed),
        Request(base_request_id(model, c_ids, max_tokens), 1, len(c_ids), max_tokens, list(c_ids), c_text, c_rt, seed + 1),
        Request(base_request_id(model, b_ids, max_tokens), 2, len(b_ids), max_tokens, b_ids, "", False, -1),
    ]
    assert len({tuple(r.prompt_token_ids) for r in reqs}) == 3
    Workload(SCHEMA, model, None, seed, reqs).save(str(out_path))
    design = {"model": model, "block_size": block_size, "requests": 3, "sources": 1, "hits": 1, "misses": 2,
              "pairs": [{"source": reqs[0].request_id, "hit": reqs[2].request_id, "prompt_len": len(b_ids), "shared_tokens": keep,
                         "shared_blocks": shared_blocks, "expected_num_cached_tokens": keep}],
              "order": "source, miss, hit", "max_tokens": max_tokens}
    Path(str(out_path) + ".design.json").write_text(json.dumps(design, indent=1) + "\n")
    return design


# --- post-hoc checks over a capture ------------------------------------------------------------


def _load_ru_index(run: Path, rid: str, step: int) -> dict[str, Any]:
    return json.loads((run / "ru" / rid / f"{step:06d}" / "index.json").read_text())


def _position_of(canonical_id: str) -> int:
    return int(next(p for p in canonical_id.split("/") if p.startswith("t=")).split("=", 1)[1])


def _entries_by_position(entries) -> dict[int, list]:
    """Group a unit's index entries by position once (a 2048-position prefill unit has ~750k entries;
    scanning it per cached position is quadratic)."""
    out: dict[int, list] = {}
    for e in entries:
        cid = e["canonical_id"] if isinstance(e, dict) else e.canonical_id
        out.setdefault(_position_of(cid), []).append(e)
    return out


def recompute_consistency(run_dir: str | Path) -> dict[str, Any]:
    """R4: for every boundary step flagged ``recompute``, every leaf of that replay unit whose position
    was computed before must hash-equal the leaf of the FIRST unit that computed the identity
    (``Boundary.step_of``). Reports counts and the first differing identity, never a verdict."""
    run = Path(run_dir)
    total = same = 0
    requests_with_recompute = 0
    steps_checked = 0
    first_diff: list[dict[str, Any]] = []
    per_request: list[dict[str, Any]] = []
    for bpath in sorted(run.glob("ru/*/boundary.json")):
        b = json.loads(bpath.read_text())
        rid = b["request_id"]
        steps = b.get("steps", [])
        rec = [s for s in steps if s.get("recompute")]
        if not rec:
            continue
        requests_with_recompute += 1
        first_step_of: dict[int, int] = {}
        for s in steps:
            for t in s["positions"]:
                first_step_of.setdefault(int(t), int(s["step"]))
        req_total = req_same = 0
        for s in rec:
            steps_checked += 1
            idx = _load_ru_index(run, rid, int(s["step"]))
            by_step: dict[int, dict[str, str]] = {}
            for e in idx["entries"]:
                cid = e["canonical_id"]
                t = _position_of(cid)
                orig = first_step_of.get(t)
                if orig is None or orig == int(s["step"]):
                    continue  # a new position of this chunk
                if orig not in by_step:
                    by_step[orig] = {x["canonical_id"]: x["leaf"] for x in _load_ru_index(run, rid, orig)["entries"]}
                total += 1
                req_total += 1
                if by_step[orig].get(cid) == e["leaf"]:
                    same += 1
                    req_same += 1
                elif len(first_diff) < 5:
                    first_diff.append({"request_id": rid, "recompute_step": int(s["step"]), "original_step": orig, "canonical_id": cid,
                                       "original_leaf": by_step[orig].get(cid), "recompute_leaf": e["leaf"]})
        per_request.append({"request_id": rid, "recompute_steps": [int(s["step"]) for s in rec], "leaves_compared": req_total,
                            "leaves_identical": req_same})
    return {
        "requests_with_recompute": requests_with_recompute, "recompute_steps_checked": steps_checked,
        "recomputed_leaves_compared": total, "recomputed_leaves_identical": same, "all_identical": total > 0 and same == total,
        "first_differing": first_diff, "per_request": per_request,
        "method": "leaf hash (SHA-256 over id, dtype, shape, bytes) of every identity in a recompute unit vs the unit that first computed it",
    }


def cached_binding_report(run_dir: str | Path) -> dict[str, Any]:
    """R3: every cached position of a hit request is bound (boundary ``cached_prefix``) to a source
    request; the source's replay unit for that position must hold the attention write identities
    the hit request's reads resolve to. Reports the binding counts and the first unbound identity."""
    run = Path(run_dir)
    hits = 0
    cached_positions = 0
    bound = 0
    missing: list[dict[str, Any]] = []
    per_request: list[dict[str, Any]] = []
    for bpath in sorted(run.glob("ru/*/boundary.json")):
        b = json.loads(bpath.read_text())
        cp = b.get("cached_prefix") or []
        if not cp:
            continue
        hits += 1
        rid = b["request_id"]
        n_pos = sum(len(c["positions"]) for c in cp)
        cached_positions += n_pos
        req_bound = 0
        for c in cp:
            src = c["source_request_id"]
            sb = json.loads((run / "ru" / src / "boundary.json").read_text())
            step_of: dict[int, int] = {}
            for s in sb.get("steps", []):
                for t in s["positions"]:
                    step_of.setdefault(int(t), int(s["step"]))
            positions_held: dict[int, set[int]] = {}
            for t in c["positions"]:
                st = step_of.get(int(t))
                if st is None:
                    if len(missing) < 5:
                        missing.append({"request_id": rid, "source": src, "t": int(t), "why": "source has no step for t"})
                    continue
                if st not in positions_held:
                    positions_held[st] = {_position_of(e["canonical_id"]) for e in _load_ru_index(run, src, st)["entries"]}
                if int(t) in positions_held[st]:
                    bound += 1
                    req_bound += 1
                elif len(missing) < 5:
                    missing.append({"request_id": rid, "source": src, "t": int(t), "why": "source unit holds no leaf at t"})
        per_request.append({"request_id": rid, "cached_positions": n_pos, "bound": req_bound,
                            "sources": sorted({c["source_request_id"] for c in cp}), "prompt_len": len(b["prompt_token_ids"])})
    return {"hit_requests": hits, "cached_positions": cached_positions, "bound_to_source_leaves": bound,
            "all_bound": cached_positions > 0 and bound == cached_positions, "first_unbound": missing, "per_request": per_request}


def cached_vs_fresh(commit_dir: str | Path, fresh_dir: str | Path) -> dict[str, Any]:
    """R3, the explicit comparison (§15.5): for every cached position ``t`` of a hit request B bound to
    source A in the cached replay (``commit_dir``), every leaf B computed at ``t`` in a replay with
    the prefix cache OFF (``fresh_dir``, values stored) must be byte-identical to A's committed leaf
    of the same layer/op/port at ``t``. Leaf hashes include the canonical id, so this is a byte
    comparison (both runs must store values). Also reports, from hashes, that A's own leaves and
    B's non-cached leaves agree between the two runs."""
    from verity_vllm.capture import CaptureReader

    cached = CaptureReader(commit_dir)
    fresh = CaptureReader(fresh_dir)
    compared = same = 0
    per_request: list[dict[str, Any]] = []
    first_diff: list[dict[str, Any]] = []
    own_same = own_compared = 0
    for rid in cached.request_ids():
        if not cached.has_boundary(rid):
            continue
        b = cached.boundary(rid)
        if not b.cached_prefix:
            continue
        fb = fresh.boundary(rid)
        req_c = req_s = 0
        by_pos: dict[tuple[str, str, int], dict[int, list]] = {}

        def entries_at(reader, r, step, t):
            key = (id(reader), r, step)
            if key not in by_pos:
                by_pos[key] = _entries_by_position(reader.ru(r, step).index.entries)
            return by_pos[key].get(t, [])

        for cp in b.cached_prefix:
            src = cp.source_request_id
            sb = cached.boundary(src)
            for t in cp.positions:
                a_step = sb.step_of(t).step
                b_step = fb.step_of(t).step
                a_unit = cached.ru(src, a_step)
                b_unit = fresh.ru(rid, b_step)
                b_leaves = {e.canonical_id: e for e in entries_at(fresh, rid, b_step, t)}
                for e in entries_at(cached, src, a_step, t):
                    bid = e.canonical_id.replace(f"/r={src}/", f"/r={rid}/", 1)
                    if bid not in b_leaves:
                        continue
                    compared += 1
                    req_c += 1
                    if a_unit.value(e.canonical_id).data == b_unit.value(bid).data:
                        same += 1
                        req_s += 1
                    elif len(first_diff) < 5:
                        first_diff.append({"source_id": e.canonical_id, "fresh_id": bid})
        # B's own (non-cached) leaves: same ids in both runs -> leaf hashes comparable
        c_leaves: dict[str, str] = {}
        for s in b.steps:
            c_leaves.update((e.canonical_id, e.leaf) for e in cached.ru(rid, s.step).index.entries)
        for fs in fb.steps:
            for e in fresh.ru(rid, fs.step).index.entries:
                if e.canonical_id in c_leaves:
                    own_compared += 1
                    own_same += c_leaves[e.canonical_id] == e.leaf
        per_request.append({"request_id": rid, "sources": sorted({c.source_request_id for c in b.cached_prefix}),
                            "cached_positions": sum(len(c.positions) for c in b.cached_prefix), "leaves_compared": req_c, "leaves_identical": req_s})
    return {
        "hit_requests": len(per_request), "cached_position_leaves_compared": compared, "cached_position_leaves_identical": same,
        "all_identical": compared > 0 and same == compared, "first_differing": first_diff,
        "hit_requests_own_leaves_compared": own_compared, "hit_requests_own_leaves_identical": own_same, "per_request": per_request,
        "method": "bytes of every leaf the hit request computed at a cached position in the cache-OFF replay vs the source request's committed "
                  "leaf of the same layer/op/port/position in the cache-ON replay; the hit request's non-cached leaves by leaf hash (same ids)",
    }


# --- production record comparison ----------------------------------------------------------------


def _load_records(path: str | Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    return {r["request_id"]: r for r in rows}


def compare_records(labelled: dict[str, str | Path]) -> dict[str, Any]:
    """Bit comparison of ``output_token_ids`` and ``chosen_logprobs`` (float64 as written) between every
    pair of production records of the same workload (prompts must agree request by request)."""
    recs = {k: _load_records(v) for k, v in labelled.items()}
    pairs = []
    for (la, a), (lb, b) in itertools.combinations(recs.items(), 2):
        common = [rid for rid in a if rid in b]
        tok_same = lp_same = 0
        diffs = []
        for rid in common:
            ra, rb = a[rid], b[rid]
            if list(ra["prompt_token_ids"]) != list(rb["prompt_token_ids"]):
                raise SystemExit(f"{rid}: prompts differ between {la} and {lb}")
            ta, tb = list(ra["output_token_ids"]), list(rb["output_token_ids"])
            pa, pb = list(ra.get("chosen_logprobs") or []), list(rb.get("chosen_logprobs") or [])
            if ta == tb:
                tok_same += 1
            if pa == pb and ta == tb:
                lp_same += 1
            elif len(diffs) < 8:
                k = next((i for i, (x, y) in enumerate(itertools.zip_longest(ta, tb)) if x != y), None)
                kl = next((i for i, (x, y) in enumerate(itertools.zip_longest(pa, pb)) if x != y), None)
                diffs.append({"request_id": rid, "prompt_len": ra["prompt_len"], "first_differing_token_index": k,
                              "first_differing_logprob_index": kl, "num_cached_tokens": [ra.get("num_cached_tokens"), rb.get("num_cached_tokens")]})
        pairs.append({"a": la, "b": lb, "requests_compared": len(common), "tokens_identical": tok_same,
                      "tokens_and_logprobs_identical": lp_same, "identical": lp_same == len(common) and len(common) > 0,
                      "differences": diffs, "profile_ids": [next(iter(a.values()))["profile_id"], next(iter(b.values()))["profile_id"]]})
    return {"records": {k: str(v) for k, v in labelled.items()}, "pairs": pairs, "all_identical": all(p["identical"] for p in pairs),
            "method": "output_token_ids list equality and chosen_logprobs (float64 as written by production.py) list equality per request"}


# --- results table for docs/vllm-poc/state-identity.md ------------------------------------------------


def _hist(d: dict[str, int] | None) -> str:
    if not d:
        return "-"
    from collections import Counter

    c = Counter(int(v) for v in d.values())
    return ", ".join(f"{k}×{n}" for k, n in sorted(c.items()))


def _fmt_count(d: dict[str, Any] | None, a: str, b: str) -> str:
    if not d:
        return "-"
    return f"{d.get(a)}/{d.get(b)}"


def results_table(labelled: dict[str, str | Path]) -> str:
    """Markdown rows (one per collected experiment directory holding ``state_row.json``, written by
    ``pod/collect_w10.sh``) for the results section of the doc; the numbers are copied, never judged."""
    head = ("| row | experiment | production overrides | witness (production) | replay | KV reads checked / unresolved "
            "| tokens+logprobs replay = production | ref1/ref2 vs commit | verify | negatives | first blocker |")
    lines = [head, "|" + "---|" * 11]
    for label, d in labelled.items():
        s = json.loads((Path(d) / "state_row.json").read_text())
        p = s.get("production") or {}
        r = s.get("replay") or {}
        sr = r.get("state_rows") or {}
        cap = r.get("capture") or {}
        t1 = s.get("table1") or {}
        cmp_ = s.get("compare") or {}
        ov = s.get("engine_overrides") or {}
        witness = [f"chunks/request {_hist(p.get('prefill_chunks_per_request'))}", f"max tokens/step {p.get('max_tokens_in_a_step')}",
                   f"steps {p.get('engine_steps')}", f"preemptions {p.get('preemptions')}",
                   f"requests with cached tokens {p.get('requests_with_cached_tokens')}"]
        replay = [f"units {cap.get('replay_units')}", f"leaves {cap.get('leaves')}", f"steps {(sr.get('scheduler') or {}).get('engine_steps')}"]
        if sr.get("kv_reads_cross_request_resolved"):
            replay.append(f"cross-request KV reads {sr['kv_reads_cross_request_resolved']}")
        if sr.get("requests_with_recompute"):
            replay.append(f"recompute units {len(sr.get('recompute_steps') or [])} in {sr['requests_with_recompute']} request(s)")
        tok = f"{_fmt_count(r.get('trajectory_followed'), 'followed', 'requests')} followed, {_fmt_count(r.get('logprobs_vs_production'), 'equal', 'requests_compared')} logprobs equal"
        comp = f"{'identical' if cmp_.get('identical') else 'DIFFERS'} ({cmp_.get('compared_scalars')} scalars)" if cmp_ else "-"
        rc = t1.get("relation_checks") or {}
        ver = (f"**{t1.get('result')}** same-runtime {_fmt_count(rc.get('same_runtime'), 'exact', 'attempted')}, "
               f"independent {_fmt_count(rc.get('independent'), 'exact', 'attempted')}, structural {(t1.get('structural') or {}).get('result')}") if t1 else "not verified"
        neg = _fmt_count(t1.get("negative"), "rejected", "attempted") if t1 else "-"
        lines.append(f"| {label} | `{Path(d).name}` | {', '.join(f'{k}={v}' for k, v in ov.items()) or 'none'} | {'; '.join(witness)} | "
                     f"{'; '.join(replay)} | {cap.get('kv_reads_checked')} / {cap.get('kv_reads_unresolved')} | {tok} | {comp} | {ver} | {neg} | {t1.get('first_blocker') if t1 else '-'} |")
    return "\n".join(lines) + "\n"


# --- CLI -------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("workload")
    w.add_argument("--base", required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--block-size", type=int, default=16)
    w.add_argument("--tail", type=int, default=64)
    for name in ("recompute-check", "cached-check"):
        p = sub.add_parser(name)
        p.add_argument("run_dir")
        p.add_argument("--out", required=True)
    c = sub.add_parser("compare-records")
    c.add_argument("records", nargs="+", metavar="LABEL=production.jsonl")
    c.add_argument("--out", required=True)
    cf = sub.add_parser("cached-vs-fresh")
    cf.add_argument("commit_dir")
    cf.add_argument("fresh_dir")
    cf.add_argument("--out", required=True)
    t = sub.add_parser("table")
    t.add_argument("dirs", nargs="+", metavar="LABEL=results/<experiment_id>")
    ty = sub.add_parser("tiny-workload")
    ty.add_argument("--tokenizer", required=True, help="local snapshot path of the checkpoint (pod: checkpoints.json local_path)")
    ty.add_argument("--model", required=True)
    ty.add_argument("--out", required=True)
    ty.add_argument("--max-tokens", type=int, default=4)
    args = ap.parse_args(argv)
    if args.cmd == "table":
        sys.stdout.write(results_table(dict(item.split("=", 1) for item in args.dirs)))
        return 0
    if args.cmd == "tiny-workload":
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        doc = build_tiny_shared_prefix_workload(tok, args.model, args.out, max_tokens=args.max_tokens)
        print(json.dumps(doc))
        return 0
    if args.cmd == "workload":
        doc = build_shared_prefix_workload(args.base, args.out, block_size=args.block_size, tail=args.tail)
    elif args.cmd == "recompute-check":
        doc = recompute_consistency(args.run_dir)
        Path(args.out).write_text(json.dumps(doc, indent=1) + "\n")
    elif args.cmd == "cached-check":
        doc = cached_binding_report(args.run_dir)
        Path(args.out).write_text(json.dumps(doc, indent=1) + "\n")
    elif args.cmd == "cached-vs-fresh":
        doc = cached_vs_fresh(args.commit_dir, args.fresh_dir)
        Path(args.out).write_text(json.dumps(doc, indent=1) + "\n")
    else:
        doc = compare_records(dict(item.split("=", 1) for item in args.records))
        Path(args.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: v for k, v in doc.items() if not isinstance(v, (list, dict))}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
