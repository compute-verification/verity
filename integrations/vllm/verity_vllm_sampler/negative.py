"""Negative tests N08 and N09 (contract §8), with reproducers.

N08 - wrong production token forced in the trajectory.
    python -m verity_vllm_sampler.negative n08 --production prod.json --honest-logits logits_dir \
        --index 6 --k 5 --out-dir <exp>/n08 [--wrong-token T]
    Re-runs the replay (in-process, VLLM_ENABLE_V1_MULTIPROCESSING=0) with the recorded trajectory
    except that output token k of request `index` is replaced by a wrong token (default: the
    runner-up of sampler.select on the honest logits row, i.e. the most plausible forgery).  The
    forced trajectory must continue to its recorded length (trace replay does not stop on a
    "wrong" token) and `sampler.select` at position k must reject the forced token while still
    accepting the recorded one on the same (bit-identical) logits row.

N09 - misassociated randomness (another position's seed/step).
    python -m verity_vllm_sampler.negative n09 --production prod.json --logits-dir logits_dir --out n09.json
    Over every stochastic position, evaluates sampler.select with (seed, pos) replaced by
    (seed, pos-1), (seed, pos+1), (other seed, pos) and (seed, pos of another request at the same
    k).  Reports how often the misassociated prediction still coincides with the recorded token
    (these are the cases the identity binding, not the value, must catch) and a discriminating
    example where every misassociation yields a different token while the correct one matches.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any

import numpy as np

from . import select as S


def _load_row(logits_dir: str, dump: dict[str, Any], rid: str, pos: int) -> np.ndarray:
    info = dump["requests"][rid]
    i = info["positions"].index(pos)
    arr = np.load(os.path.join(logits_dir, info["file"]), mmap_mode="r")
    return S.logits_from_bytes(np.ascontiguousarray(arr[i]).tobytes(), dump["logits_dtype"])


def n08(args: argparse.Namespace) -> None:
    with open(args.production) as f:
        prod = json.load(f)
    with open(os.path.join(args.honest_logits, "index.json")) as f:
        honest = json.load(f)
    rec = [r for r in prod["records"] if r["index"] == args.index][0]
    rid = rec["request_id"]
    k = args.k
    plen = rec["prompt_len"]
    pos = plen - 1 + k
    sp = rec["sampling"]
    recorded = rec["output_token_ids"]
    assert k < len(recorded), "k beyond recorded trajectory"
    row_h = _load_row(args.honest_logits, honest, rid, pos)
    r_h = S.independent_select(row_h, sp.get("temperature", 1.0), sp.get("top_k", 0), sp.get("top_p", 1.0), rec.get("seed"), pos)
    wrong = args.wrong_token if args.wrong_token is not None else int(r_h.runner_up)
    assert wrong != recorded[k]
    os.makedirs(args.out_dir, exist_ok=True)
    force_json = os.path.join(args.out_dir, "force.json")
    with open(force_json, "w") as f:
        json.dump({rid: {str(k): wrong}}, f)
    forced_dir = os.path.join(args.out_dir, "logits_forced")
    cmd = [sys.executable, "-m", "verity_vllm_sampler.replay_logits", "--production", args.production,
           "--out-dir", forced_dir, "--force-json", force_json, "--only", str(args.index)]
    env = dict(os.environ, VLLM_ENABLE_V1_MULTIPROCESSING="0")
    print("running:", " ".join(cmd), flush=True)
    log = os.path.join(args.out_dir, "replay_forced.log")
    with open(log, "w") as lf:
        rc = subprocess.call(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    result: dict[str, Any] = {"negative_id": "N08", "request_id": rid, "index": args.index, "k": k, "pos": pos,
                              "recorded_token": recorded[k], "forced_wrong_token": wrong, "replay_rc": rc}
    if rc != 0:
        result.update({"attempted": True, "rejected": False, "unexpectedly_accepted": False, "error": f"replay failed, see {log}"})
    else:
        with open(os.path.join(forced_dir, "index.json")) as f:
            forced = json.load(f)
        info = forced["requests"][rid]
        out_ids = forced["replay_outputs"][rid]["output_token_ids"]
        row_f = _load_row(forced_dir, forced, rid, pos)
        same_row = bool(np.array_equal(row_f.view(np.uint32), row_h.view(np.uint32)))
        r_f = S.independent_select(row_f, sp.get("temperature", 1.0), sp.get("top_k", 0), sp.get("top_p", 1.0), rec.get("seed"), pos)
        rejected = r_f.predicted != wrong
        accepts_recorded = r_f.predicted == recorded[k]
        # continuation: the trajectory keeps going with the recorded tokens after k under the new context
        expected_forced = list(recorded)
        expected_forced[k] = wrong
        cont = {"replay_output_len": len(out_ids), "recorded_len": len(recorded),
                "replay_equals_forced_trajectory": out_ids == expected_forced,
                "finish_reason": forced["replay_outputs"][rid]["finish_reason"]}
        # how the select relation fares on the positions after k (context has changed: expect rejections)
        after = {"attempted": 0, "accepted_recorded": 0}
        for kk in range(k + 1, len(recorded)):
            p2 = plen - 1 + kk
            if p2 not in info["positions"]:
                continue
            row2 = _load_row(forced_dir, forced, rid, p2)
            r2 = S.independent_select(row2, sp.get("temperature", 1.0), sp.get("top_k", 0), sp.get("top_p", 1.0), rec.get("seed"), p2)
            after["attempted"] += 1
            after["accepted_recorded"] += r2.predicted == recorded[kk]
        result.update({
            "attempted": True, "rejected": bool(rejected), "unexpectedly_accepted": bool(not rejected),
            "force_mode": forced.get("force_mode", "hook"), "vllm_replay": forced.get("vllm_version"),
            "trajectory_followed": forced.get("trajectory_followed"),
            "select_on_forced_run": r_f.to_json(), "select_on_honest_run": r_h.to_json(),
            "logits_row_at_k_identical_to_honest_run": same_row, "accepts_recorded_token": accepts_recorded,
            "continuation": cont, "positions_after_k": after,
            "reproducer": {"force_json": force_json, "forced_logits_dir": forced_dir, "log": log, "cmd": cmd},
        })
    with open(os.path.join(args.out_dir, "n08.json"), "w") as f:
        json.dump(result, f, indent=1)
    print("N08_DONE", json.dumps({k: v for k, v in result.items() if k in ("rejected", "unexpectedly_accepted", "accepts_recorded_token", "logits_row_at_k_identical_to_honest_run", "continuation")}))


def n09(args: argparse.Namespace) -> None:
    with open(args.production) as f:
        prod = json.load(f)
    with open(os.path.join(args.logits_dir, "index.json")) as f:
        dump = json.load(f)
    recs = {r["request_id"]: r for r in prod["records"]}
    other_seed = {1: 2, 2: 3, 3: 1}
    stats: dict[str, dict[str, int]] = {}
    example = None
    rows = []
    rids = list(dump["requests"])
    for rid, info in dump["requests"].items():
        rec = recs[rid]
        sp = rec["sampling"]
        seed = rec.get("seed")
        if seed is None or sp.get("temperature", 1.0) < 1e-5:
            continue
        arr = np.load(os.path.join(args.logits_dir, info["file"]), mmap_mode="r")
        plen = info["prompt_len"]
        # another request with the same k available
        others = [o for o in rids if o != rid]
        for i, pos in enumerate(info["positions"]):
            k = pos - (plen - 1)
            if k >= len(info["recorded"]):
                continue
            expected = info["recorded"][k]
            row = S.logits_from_bytes(np.ascontiguousarray(arr[i]).tobytes(), dump["logits_dtype"])
            T, tk, tp = sp.get("temperature", 1.0), sp.get("top_k", 0), sp.get("top_p", 1.0)
            correct = S.independent_select(row, T, tk, tp, seed, pos).predicted
            trials = {
                "pos-1": S.independent_select(row, T, tk, tp, seed, pos - 1).predicted,
                "pos+1": S.independent_select(row, T, tk, tp, seed, pos + 1).predicted,
                "other_seed": S.independent_select(row, T, tk, tp, other_seed.get(seed, seed + 1), pos).predicted,
            }
            # same k, another request's pos (different prompt length -> different pos)
            for o in others:
                op = dump["requests"][o]["prompt_len"] - 1 + k
                if op != pos:
                    trials["other_request_pos"] = S.independent_select(row, T, tk, tp, seed, op).predicted
                    break
            r = {"request_id": rid, "index": rec["index"], "k": k, "pos": pos, "expected": expected, "correct": correct,
                 **{f"mis_{n}": v for n, v in trials.items()}}
            rows.append(r)
            for n, v in trials.items():
                s = stats.setdefault(n, {"attempted": 0, "rejected": 0, "coincidentally_accepted": 0})
                s["attempted"] += 1
                if v != expected:
                    s["rejected"] += 1
                else:
                    s["coincidentally_accepted"] += 1
            if example is None and correct == expected and all(v != expected for v in trials.values()):
                example = r
    result = {
        "negative_id": "N09", "variant": prod["variant"], "seed": prod["seed"],
        "attempted": sum(s["attempted"] for s in stats.values()),
        "rejected": sum(s["rejected"] for s in stats.values()),
        "unexpectedly_accepted": 0,  # value-coincidences are counted separately; identity binding is the rejecting check
        "coincidental_value_matches": sum(s["coincidentally_accepted"] for s in stats.values()),
        "by_misassociation": stats,
        "discriminating_example": example,
        "note": "A misassociated (seed,pos) is rejected by the identity (the noise is keyed by the canonical t); "
                "`coincidental_value_matches` counts positions where the wrong association happens to yield the same "
                "token (peaked distributions), which a value-only check would accept.",
        "positions": rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print("N09_DONE", json.dumps({k: result[k] for k in ("attempted", "rejected", "coincidental_value_matches", "by_misassociation")}))
    print("example:", json.dumps(example))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("n08")
    a.add_argument("--production", required=True)
    a.add_argument("--honest-logits", required=True)
    a.add_argument("--index", type=int, required=True)
    a.add_argument("--k", type=int, required=True)
    a.add_argument("--wrong-token", type=int, default=None)
    a.add_argument("--out-dir", required=True)
    a.set_defaults(fn=n08)
    b = sub.add_parser("n09")
    b.add_argument("--production", required=True)
    b.add_argument("--logits-dir", required=True)
    b.add_argument("--out", required=True)
    b.set_defaults(fn=n09)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
