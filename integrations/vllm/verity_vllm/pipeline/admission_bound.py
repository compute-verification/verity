"""Commit admission: the HOST-retention bound of a `--retain host` Commit from the workload's DECLARED schedule, not from its
envelope.

`step_bound_bytes` (native_host) bounds ONE step by the envelope -- every request at the longest prompt, B rows at every decode step,
the FA2/FA3 M1 stream of a max-length prefill packed B-wide -- and `retained_bytes_bound` multiplies that envelope over the horizon.
That is the right bound for the DEVICE staging block (one step at a time) and 10-13x loose for the whole run's HOST retention: a
llama B16 1024/128 mixed-arrivals Commit was told 559,384 MiB against a 359,535 MiB cgroup and committed 42.5 GiB (rc pinned at
~99 GB RSS); the phi3 B8 1024/128 Commit beside it (582,111 MiB) died of the host SUM, not of its own bytes.  A bound
that loose cannot gate a launch, and a launch predicted to exceed the cgroup is not made.

This module sums the bytes the collector binds per ENGINE STEP under the declared schedule (verity-sweep/workload/v1 `requests[]`:
`prompt_len`, `max_tokens`, `arrive_step`), from the same per-module widths `step_bound_bytes` measured on the model:

    tokens_s x per_token + live_s x per_seq + per_step (+ per_step per extra prefill chunk)
    + layers x 4 B x [ sum over prompts arriving at s of M1(heads, prompt_r, NB_s)  +  M1(decode launch of the live rows, NB_s) ]

where NB_s = ceil(K_s / kBlockN) and K_s = the longest context any request live at s can reach (prompt + cap + lag) -- the geometry the
record shows: one `fa2.m1/L<k>/s<i>/g<heads>x<prompt_i>x<NB>` stream per arriving prompt, one `g<live x kv_heads>x<g>x<NB>` (GQA swapped
kernel) or `g<live x heads>x1x<NB>` stream for the decode rows, NB shared step-wide (vLLM passes the batch's max_seqlen_k to every
launch of the step).  Validated against the per-step `n_bytes` of three Commit records (llama B16 1024/128 mixed-arrivals: 181/181 steps
>= observed, total +5.8 %; llama B4 1024/128: 4 concurrent prefills; phi3 B8 256/32: hd 96, no GQA) in tests/test_admission.py.

Soundness assumptions (named on the record): every request runs to its cap plus `lag` post-EOS rows (an EOS stop before the cap only
removes bytes; the LAG is declared -- pass the Program's lag); a prompt longer than the step's token budget is chunked by the engine
(the sum over chunks of rows x NB is <= the unchunked launch; one `per_step` of chunk padding is added per extra chunk); the module
widths are those `step_bound_bytes` measured (a sizing record that predates the per-layer RoPE count is corrected here by name).

CLI (torch-free, laptop):  verity-vllm admission-bound --workload <manifest.json> --sizing <arena_sizing_*.json>
                                  [--cap-bytes N] [--lag 1] [--observed <steps.jsonl>]
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity_vllm.acquire.sources.hidden_source import CHUNK_WORDS, fa2_kblock_n, fa3_tile, mat_total_words
from verity_vllm.config import RowEngine, option, workload_arg

SCHEMA = "verity-sweep/admission/f-da-15/v1"
LEVELS_FACTOR = 1.25          # raw x1.25 for the merkle levels pinned beside the raw bytes
MiB = 1 << 20


def _hidden_words(HB: int, M: int, NB: int, BN: int, D: int) -> int:
    return -(-mat_total_words(HB, M, NB, BN, D) // CHUNK_WORDS) * CHUNK_WORDS


def _launch_hidden_words(model: dict[str, Any], *, n_seq: int, seqlen_q: int, seqlen_k: int, per_sequence: bool) -> int:
    """Words of the M1 hidden stream(s) one attention launch binds, in the geometry the record shows (see module doc)."""
    heads, kvh, hd = int(model["heads"]), int(model["kv_heads"]), int(model["head_dim"])
    g = max(1, heads // kvh)
    if str(model.get("attn_backend") or "fa2") == "fa3":
        if hd not in (64, 128):
            return 0
        _, bn = fa3_tile(hd, seqlen_q, g, 1)
        NB = -(-seqlen_k // bn)
        return _hidden_words(n_seq * kvh, seqlen_q * g, NB, bn, hd)
    bn = fa2_kblock_n(hd, 1) if hd in (64, 96, 128, 256) else 128
    NB = -(-seqlen_k // bn)
    if seqlen_q == 1 and heads > kvh:                     # FA2 decode "swapped" kernel: rows = the query heads of a KV group
        return _hidden_words(n_seq * kvh, g, NB, bn, hd)
    return _hidden_words(n_seq * heads, seqlen_q, NB, bn, hd)


def per_token_bytes_of(sizing: dict[str, Any]) -> tuple[int, str | None]:
    """The sizing record's bytes per token, corrected by name when the record predates the per-layer RotaryEmbedding count (the module
    is shared by every layer and invoked once per layer; `step_bound_bytes` counted it once -> 76,800 B/token short on Llama-3.2-1B)."""
    pt = int(sizing["per_token_bytes"])
    if sizing.get("rope_per_layer"):
        return pt, None
    rope = int((sizing.get("widths") or {}).get("RotaryEmbedding") or 0)
    layers = int((sizing.get("model") or {}).get("layers") or 1)
    if rope and layers > 1:
        return pt + rope * (layers - 1), f"RotaryEmbedding width {rope} B counted once in the sizing record: x{layers} layers (+{rope * (layers - 1)} B/token)"
    return pt, None


def scheduled_step_bytes(sizing: dict[str, Any], requests: list[dict[str, Any]], *, max_num_batched_tokens: int | None = None,
                         lag: int = 1) -> dict[str, Any]:
    """Bytes the collector binds per engine step under the declared schedule; total = the host-retention need of one `--retain host` run."""
    model = dict(sizing["model"])
    layers = int(model.get("layers") or 1)
    hidden_m1 = bool(model.get("hidden_m1", True))
    per_token, rope_note = per_token_bytes_of(sizing)
    per_seq, per_step = int(sizing["per_seq_bytes"]), int(sizing["per_step_bytes"])
    reqs = []
    for i, r in enumerate(requests):
        p = int(r.get("prompt_len") or len(r.get("prompt_token_ids") or []))
        c = int(r.get("max_tokens") or 1)
        a = int(r.get("arrive_step") or 0)
        if p <= 0 or c <= 0 or a < 0:
            raise ValueError(f"request {i}: prompt_len={p} max_tokens={c} arrive_step={a} (need prompt_len>0, max_tokens>0, arrive_step>=0)")
        reqs.append({"i": i, "prompt": p, "cap": c, "arrive": a, "end": a + c + max(0, int(lag))})   # live at steps [arrive, end)
    if not reqs:
        raise ValueError("no requests")
    horizon = max(r["end"] for r in reqs)
    steps: list[dict[str, Any]] = []
    tot = tok_b = seq_b = step_b = hid_b = 0
    prompt_tokens = sum(r["prompt"] for r in reqs)
    decode_rows = 0
    for s in range(horizon):
        pref = [r for r in reqs if r["arrive"] == s]
        dec = [r for r in reqs if r["arrive"] < s < r["end"]]
        live = pref + dec
        if not live:
            continue                                                            # an idle step binds nothing
        tokens = sum(r["prompt"] for r in pref) + len(dec)
        K = max(r["prompt"] + r["cap"] + max(0, int(lag)) for r in live)        # the longest context any live request can reach
        chunks = 1
        if max_num_batched_tokens and pref:
            budget = max(1, int(max_num_batched_tokens) - len(dec))
            chunks = max(1, math.ceil(sum(r["prompt"] for r in pref) / budget))
        hid = 0
        if hidden_m1:
            for r in pref:
                hid += _launch_hidden_words(model, n_seq=1, seqlen_q=r["prompt"], seqlen_k=K, per_sequence=True)
            if dec:
                hid += _launch_hidden_words(model, n_seq=len(dec), seqlen_q=1, seqlen_k=K, per_sequence=False)
        hb = layers * 4 * hid
        tb, sb, pb = tokens * per_token, len(live) * per_seq, per_step * chunks
        b = tb + sb + pb + hb
        tot += b; tok_b += tb; seq_b += sb; step_b += pb; hid_b += hb
        decode_rows += len(dec)
        steps.append({"step": s, "prefills": len(pref), "decodes": len(dec), "tokens": tokens, "K": K, "chunks": chunks, "bytes": b, "hidden_bytes": hb})
    peak = max(steps, key=lambda x: x["bytes"])
    return {"schema": SCHEMA, "steps": len(steps), "horizon": horizon, "total_bytes": tot, "peak_step": peak["step"], "peak_step_bytes": peak["bytes"],
            "prompt_tokens": prompt_tokens, "decode_rows": decode_rows, "requests": len(reqs), "lag": int(lag),
            "composition": {"token_bytes": tok_b, "seq_bytes": seq_b, "step_bytes": step_b, "hidden_bytes": hid_b},
            "per_token_bytes": per_token, "per_seq_bytes": per_seq, "per_step_bytes": per_step, "hidden_m1": hidden_m1,
            "attn_backend": str(model.get("attn_backend") or "fa2"), "max_num_batched_tokens": max_num_batched_tokens,
            "assumptions": [f"every request runs to its cap + {int(lag)} lag row(s); an EOS before the cap only removes bytes",
                            "per-arriving-prompt M1 stream (heads x prompt x NB) + one decode-launch stream, NB from the step's longest reachable context",
                            "a prompt over the step's token budget is chunked by the engine: rows x NB per chunk <= unchunked; one per_step of padding per extra chunk"]
                           + ([rope_note] if rope_note else []),
            "per_step": steps}


#: the windowed collector (native_collect_v2b, row_pod default `--window-mb 256`) WITHOUT bounded staging
#: copies EVERY window of every step to the pinned pool as the learning re-hash source / hit-plan hedge copy, whatever `--retain` says
#: (`--retain none` still allocates the whole pool).  The pool is a term of the run under every retention on that path.
HIDDEN_FAMILIES = ("fa2_hidden_m1", "fa3_hidden_m1")


def predicted_pool_bytes(raw: int, hidden: int, *, retain: str, levels_factor: float, windowed: bool, bounded: bool,
                         retain_exclude: set[str] | frozenset[str] | None) -> tuple[int, str]:
    """The pinned host pool one Commit run allocates, by path, with the rule that produced it (named on the record):
      unbounded windowed   raw x levels_factor under EVERY retention (hedge copies of every window)
      bounded, retain host (raw - excluded hidden) + raw x (levels_factor - 1): `--retain-exclude hidden_m1` keeps the C2 byte source
                           (the module outputs) and drops the M1 stream's chunk-aligned interior, which no C2 consumer reads (opens by
                           replay); the Merkle levels still span every leaf, the excluded ones too
      bounded, other       levels only: raw x (levels_factor - 1) (the raw bytes are hashed in the window and released; openings by replay)
      per-step blocks      raw x levels_factor for retain host, 0 otherwise (the pre-P0 path: nothing is copied unless retained)"""
    excl = set(retain_exclude or ())
    if bounded:
        if retain == "host":
            kept = raw - hidden if excl & set(HIDDEN_FAMILIES) else raw
            rule = (f"bounded staging, retain host excluding {sorted(excl)}: kept raw {kept >> 20} MiB (= raw {raw >> 20} - hidden M1 {hidden >> 20}) "
                    f"+ levels over every leaf raw {raw >> 20} MiB x {levels_factor - 1.0:.2f}" if excl else f"bounded staging, retain host: raw {raw >> 20} MiB x {levels_factor}")
            return kept + int(raw * (levels_factor - 1.0)), rule
        return int(raw * (levels_factor - 1.0)), f"bounded staging, retain {retain}: levels only, raw {raw >> 20} MiB x {levels_factor - 1.0:.2f} (raw bytes released per window, openings by replay)"
    if windowed:
        return int(raw * levels_factor), (f"unbounded windowed staging, retain {retain}: raw {raw >> 20} MiB x {levels_factor} -- every window is copied to the pinned pool "
                                          "regardless of --retain (hedge copy / learning re-hash source)")
    if retain == "host":
        return int(raw * levels_factor), f"per-step blocks, retain host: raw {raw >> 20} MiB x {levels_factor}"
    return 0, f"per-step blocks, retain {retain}: no raw bytes copied to host"


def committer_resident(workload: dict[str, Any], host_cap_bytes: int | None, *, pairs: int = 1, replay_workers: int | None = None) -> dict[str, Any]:
    """What the Commit holds beside the pinned pool at its peak, from the row planner's commit stage (`telemetry.admission.predict`):
    the C2 Programs and match oracle, the leaf digests, the binding index, the forked replay workers, the serving base and the margins.
    #23 B64 Llama: pool 183,694 MiB + resident 122,745 MiB; the pool alone fit the 239,372 MiB limit and the run OOM'd at it."""
    from verity_vllm.pipeline.telemetry import admission as plan
    cfg = plan.row_config_from_workload(workload, pairs=max(1, int(pairs)), replay_workers=replay_workers, host_limit_bytes=host_cap_bytes, stages="commit")
    cfg.host_cpus = os.cpu_count() or cfg.host_cpus
    commit = plan.predict(cfg)["commit"]
    live = dict(commit["live_at_peak"])
    pool = int(commit["pinned_pool"]["bytes"]) if live.pop("commit.pinned_pool", None) is not None else 0
    return {"bytes": int(commit["reserved_bytes"]) - pool, "peak_point": commit["peak_point"], "live_at_peak_mib": live, "margin_mib": commit["margin"],
            "rule": "row planner commit stage (telemetry.admission.predict) reserved bytes minus its pinned-pool term"}


def admission(sizing: dict[str, Any], requests: list[dict[str, Any]], host_cap_bytes: int | None, *, retain: str = "host",
              max_num_batched_tokens: int | None = None, lag: int = 1, levels_factor: float = LEVELS_FACTOR,
              envelope_bound_bytes: int | None = None, windowed: bool = True, bounded: bool = False,
              retain_exclude: set[str] | frozenset[str] | None = None, resident: dict[str, Any] | None = None) -> dict[str, Any]:
    """The admission verdict for one Commit launch: predicted = the pinned pool the run allocates (`predicted_pool_bytes`, by staging path
    and retention) + what the Commit holds beside it at its peak (`resident`, from `committer_resident`) against the container limit.  The REFUSAL domain is `--retain host`, unbounded;
    under another retention the pool is PREDICTED and named (a non-host retention on the windowed path allocates it anyway: a
    `--retain none` run allocates the 112 GiB pool), recorded as `would_exceed_by_bytes`, never refused here.  `admitted` False names the excess; the
    envelope bound is carried beside it for the record."""
    sched = scheduled_step_bytes(sizing, requests, max_num_batched_tokens=max_num_batched_tokens, lag=lag)
    raw = int(sched["total_bytes"])
    hidden = int(sched["composition"]["hidden_bytes"])
    pool, pool_rule = predicted_pool_bytes(raw, hidden, retain=retain, levels_factor=levels_factor, windowed=windowed, bounded=bounded,
                                           retain_exclude=retain_exclude)
    predicted = pool + int((resident or {}).get("bytes") or 0)
    rec = {"schema": SCHEMA, "retain": retain, "domain": "--retain host: the whole run's raw bytes pinned on host (a non-host retention is recorded, never refused)",
           "scheduled_raw_bytes": raw, "scheduled_hidden_bytes": hidden, "levels_factor": levels_factor, "predicted_host_bytes": predicted, "pool_bytes": pool, "resident": resident,
           "pool_rule": pool_rule,
           "staging": {"windowed": bool(windowed), "bounded": bool(bounded), "retain_exclude": sorted(retain_exclude or ())},
           "host_cap_bytes": host_cap_bytes, "envelope_retained_bytes_bound": envelope_bound_bytes,
           "envelope_predicted_host_bytes": (int(envelope_bound_bytes * levels_factor) if envelope_bound_bytes else None),
           "would_exceed_by_bytes": (max(0, predicted - host_cap_bytes) if host_cap_bytes is not None else None),
           "scheduled": {k: v for k, v in sched.items() if k != "per_step"}, "peak_steps": sorted(sched["per_step"], key=lambda x: -x["bytes"])[:8]}
    if retain != "host" or bounded:
        over = (f"; WARNING the pool predicted for this path ({predicted >> 20} MiB: {pool_rule}) exceeds the limit {host_cap_bytes >> 20} MiB by "
                f"{(predicted - host_cap_bytes) >> 20} MiB -- expect an OOM-kill unless the staging path changes"
                if host_cap_bytes is not None and predicted > host_cap_bytes else f"; pool predicted {predicted >> 20} MiB ({pool_rule})")
        who = f"retain={retain}" + (" + bounded staging (the bounded HOST BOUND governs)" if bounded else "")
        rec.update(admitted=True, why=f"{who}: outside the refusal domain (--retain host, unbounded), recorded not refused{over}")
    elif host_cap_bytes is None:
        rec.update(admitted=True, why="no container memory limit visible (memory.max unset): nothing to compare against; the scheduled bound is on the record")
    elif predicted > host_cap_bytes:
        rec.update(admitted=False, why=(f"predicted host {predicted >> 20} MiB (pool: raw {raw >> 20} MiB x{levels_factor} for levels = {pool >> 20} MiB; "
                                        f"resident beside it {(predicted - pool) >> 20} MiB) exceeds the "
                                        f"container memory limit {host_cap_bytes >> 20} MiB by {(predicted - host_cap_bytes) >> 20} MiB -- not launched unchanged; "
                                        f"the change that admits it is a larger container or a non-host retention, never a smaller check"))
    else:
        envp = rec["envelope_predicted_host_bytes"]
        env_note = (f"envelope bound {envp >> 20} MiB was {'over' if envp > host_cap_bytes else 'under'} it" if envp else "no envelope bound on this record")
        rec.update(admitted=True, why=f"predicted host {predicted >> 20} MiB (pool {pool >> 20} MiB = raw {raw >> 20} MiB x{levels_factor}; resident "
                                      f"{(predicted - pool) >> 20} MiB) fits the container limit "
                                      f"{host_cap_bytes >> 20} MiB with {(host_cap_bytes - predicted) >> 20} MiB to spare ({env_note})")
    return rec


def compare_observed(sched: dict[str, Any], observed: dict[int, int]) -> dict[str, Any]:
    """Soundness check of a prediction against a record's per-step `n_bytes` (commit/steps.jsonl, instrumented arm of one pair)."""
    pred = {x["step"]: x["bytes"] for x in sched["per_step"]}
    under = [(s, o, pred.get(s, 0)) for s, o in observed.items() if o > pred.get(s, 0)]
    common = [s for s in observed if s in pred]
    tot_o = sum(observed[s] for s in common); tot_p = sum(pred[s] for s in common)
    worst = max(((observed[s] / pred[s]) if pred[s] else float("inf"), s) for s in common) if common else (0.0, None)
    return {"steps_observed": len(observed), "steps_predicted": len(pred), "steps_under_predicted": len(under), "under": under[:8],
            "observed_total_bytes": tot_o, "predicted_total_bytes": tot_p, "ratio_pred_over_obs": (tot_p / tot_o) if tot_o else None,
            "worst_obs_over_pred": worst[0], "worst_step": worst[1], "sound": not under}


def observed_steps(path: str | Path, pair: int = 0, arm: str = "instrumented") -> dict[int, int]:
    out: dict[int, int] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("arm") == arm and int(r.get("pair", 0)) == pair and r.get("n_bytes") is not None:
            out[int(r["step"])] = int(r["n_bytes"])
    return out


def host_cap() -> tuple[int | None, str | None]:
    """The host memory bound of this machine and its SOURCE by name: the cgroup limit when one is visible, else the
    machine's MemTotal (a bare host is still a bound; only an unreadable machine yields None)."""
    for cg in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = Path(cg).read_text().strip()
            if v.isdigit() and int(v) < (1 << 60):
                return int(v), cg
        except OSError:
            pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024, "/proc/meminfo MemTotal (no cgroup limit visible)"
    except OSError:
        pass
    return None, None


@dataclass(frozen=True, kw_only=True)
class Options:
    workload: str = option("--workload", type=workload_arg,
                           help="verity-sweep/workload/v1 manifest (requests[] with prompt_len, max_tokens, arrive_step)")
    sizing: str = option("--sizing", help="arena_sizing_*.json of a Commit on the same model (per-module widths + geometry)")
    cap_bytes: int | None = option("--cap-bytes", type=int, default=None,
                                   help="container memory limit (default: this host's cgroup, if any)")
    lag: int = option("--lag", type=int, default=1,
                      help="post-EOS rows per request (the declared LAG; default 1 = async scheduling)")
    retain: str = option("--retain", default="host", choices=["host", "device", "none", "sample"])
    per_step_blocks: bool = option("--per-step-blocks", default=False, action="store_true",
                                   help="the collector stages per-step blocks (--window-mb 0); default: windowed (row_pod's 256 MiB ring)")
    bounded_staging: bool = option("--bounded-staging", default=False, action="store_true",
                                   help="the run uses --bounded-staging (no hedge copies; retain host keeps the non-excluded families)")
    retain_exclude: str | None = option("--retain-exclude", default=None,
                                        help="the run's --retain-exclude families (e.g. hidden_m1 = fa2_hidden_m1,fa3_hidden_m1)")
    observed: str | None = option("--observed", default=None,
                                  help="a record's commit/steps.jsonl: report per-step soundness of the prediction")
    json: bool = option("--json", default=False, action="store_true")


def main(a: Options) -> int:
    wl = json.loads(Path(a.workload).read_text())
    sizing = json.loads(Path(a.sizing).read_text())
    mnbt = RowEngine.of(wl).args.get("max_num_batched_tokens") or (sizing.get("envelope") or {}).get("max_num_batched_tokens")
    cap, cap_src = (a.cap_bytes, "--cap-bytes") if a.cap_bytes is not None else host_cap()
    # the sizing record's envelope bound describes ITS row: carry it only when the record was sized for this workload's envelope
    env = sizing.get("envelope") or {}
    sw = wl.get("sweep") or {}
    same_row = (int(env.get("batch") or 0) == int(sw.get("batch") or len(wl["requests"])) and int(env.get("max_prompt") or 0) == max(int(r.get("prompt_len") or 0) for r in wl["requests"])
                and int(env.get("max_new") or 0) == max(int(r.get("max_tokens") or 0) for r in wl["requests"]))
    excl: set[str] = set()
    for f in (a.retain_exclude or "").split(","):
        f = f.strip().removesuffix("_stream")
        if f:
            excl |= set(HIDDEN_FAMILIES) if f in ("hidden_m1", "hidden") else {f}
    rec = admission(sizing, wl["requests"], cap, retain=a.retain, max_num_batched_tokens=mnbt, lag=a.lag,
                    envelope_bound_bytes=(sizing.get("retained_bytes_bound") if same_row else None),
                    windowed=not a.per_step_blocks, bounded=a.bounded_staging, retain_exclude=excl)
    rec["sizing_record_is_this_row"] = same_row
    rec["host_cap_source"] = cap_src
    if a.observed:
        sched = scheduled_step_bytes(sizing, wl["requests"], max_num_batched_tokens=mnbt, lag=a.lag)
        rec["observed_check"] = compare_observed(sched, observed_steps(a.observed))
    if a.json:
        print(json.dumps(rec, indent=1))
        return 0 if rec["admitted"] else 3
    s = rec["scheduled"]
    print(f"{(wl.get('sweep') or {}).get('row_id') or Path(a.workload).stem}: {s['requests']} requests, {s['steps']} scheduled steps, "
          f"{s['prompt_tokens']} prompt tokens + {s['decode_rows']} decode rows (lag {s['lag']})")
    c = s["composition"]
    print(f"  scheduled raw {rec['scheduled_raw_bytes'] / 2**30:.1f} GiB = tokens {c['token_bytes'] / 2**30:.1f} + logits {c['seq_bytes'] / 2**30:.2f} + "
          f"hidden M1 {c['hidden_bytes'] / 2**30:.1f} + padding {c['step_bytes'] / 2**20:.1f} MiB; peak step {s['peak_step']} = {s['peak_step_bytes'] / 2**30:.2f} GiB")
    envs = f"{rec['envelope_predicted_host_bytes'] >> 20} MiB" if rec.get("envelope_predicted_host_bytes") else "n/a (sizing record is another row's)"
    print(f"  predicted host {rec['predicted_host_bytes'] >> 20} MiB (x{rec['levels_factor']}) vs cap {(cap or 0) >> 20} MiB; envelope bound said "
          f"{envs} -> {'ADMITTED' if rec['admitted'] else 'REFUSED'}: {rec['why']}")
    if a.observed:
        o = rec["observed_check"]
        print(f"  observed check: {o['steps_observed']} steps, under-predicted {o['steps_under_predicted']}, pred/obs total {o['ratio_pred_over_obs']:.3f}, "
              f"worst obs/pred {o['worst_obs_over_pred']:.3f} at step {o['worst_step']} -> {'SOUND' if o['sound'] else 'UNSOUND'}")
    return 0 if rec["admitted"] else 3
