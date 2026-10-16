"""window_summary.py OUT: a window run's verdict and its honest overhead against the plain layer, from the records under OUT
(window.sh's). Medians over the window's rounds.

The plain layer is cuBLASLt int8 on the same A and B, interleaved with every edited call (slice.cu). The overhead, per call:
  gpu      the edited GEMM (noise, receipts, tail) over plain
  commit   A's operand commitment: the CPU committer as run (core-s), and the GPU cost kernel's 3.6 ms-class figure
           (slice commit: the same SHA-512 work, not wired as the committer)
  proof    a round's statement and C-Flock --zk proof (core-s), over the round's calls
"""
import json
import statistics
import sys
from pathlib import Path


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def main() -> int:
    o = Path(sys.argv[1])
    ch = [json.loads(x) for x in open(o / "challenger/rounds.jsonl")]
    pr = [json.loads(x) for x in open(o / "prover/rounds.jsonl")] if (o / "prover/rounds.jsonl").exists() else []
    win = json.loads((o / "challenger/window.json").read_text())
    info = json.loads((o / "prover/prover.json").read_text()) if (o / "prover/prover.json").exists() else {}
    calls = info.get("calls") or len(ch[0].get("receipt_roots", [])) or 1
    gpu = [g for r in ch for g in r.get("gpu") or []]
    plain, edited = med([g["plain_ms"] for g in gpu]), med([g["edited_ms"] for g in gpu])
    commit_gpu = None
    if (o / "commit.jsonl").exists() and (o / "commit.jsonl").stat().st_size:
        commit_gpu = json.loads((o / "commit.jsonl").read_text().splitlines()[0])["commit_cost"]["ms"]
    p = lambda k: med([r.get(k) for r in pr])  # noqa: E731
    c = lambda k: med([r.get(k) for r in ch])  # noqa: E731
    sess = lambda k: med([(r.get("session") or {}).get(k) for r in ch])  # noqa: E731
    commit_cpu, receipt_cpu = p("a_commit_cpu_s"), p("receipt_cpu_s")
    proof_cpu = (p("stage_s") or 0) + (p("prove_cpu_s") or 0)
    w = win["window"]
    s = {
        "window": w,
        "accepted": w["failed"] == 0 and w["passed"] == w["rounds"] and w["rounds"] > 0,
        "why_failed": win["why_failed"],
        "self_checks_ok": all((r.get("self_check") or {}).get("receipts") and (r.get("self_check") or {}).get("product") for r in ch
                              if r["status"] != "skipped"),
        "coin_interval_s": {"median": med(win["coin_intervals_s"]), "min": min(win["coin_intervals_s"], default=None),
                            "max": max(win["coin_intervals_s"], default=None)},
        "calls_per_round": calls,
        "gpu_ms_per_call": {"plain": plain, "edited": edited, "edited_over_plain": edited / plain if plain else None,
                            "commit_A_kernel": commit_gpu},
        "prover_per_round_s": {k: p(k) for k in ("a_commit_s", "a_commit_cpu_s", "calls_s", "receipt_commit_s", "receipt_cpu_s",
                                                 "open_s", "tile_check_s", "stage_s", "prove_s", "prove_cpu_s",
                                                 "prove_maxrss_mb")},
        "challenger_per_round_s": {"check_s": c("check_s"), "serve_setup_s": c("serve_setup_s"), "serve_s": c("serve_s"),
                                   "verify_cpu_s": c("verify_cpu_s"), "session_verify_s": sess("verify_s"),
                                   "session_handle_s": sess("handle_s"), "serve_maxrss_mb": c("serve_maxrss_mb")},
        "circuit": win["circuit"],
    }
    if plain:
        per_call_cpu = {"commit_A": commit_cpu / calls if commit_cpu else None,
                        "receipts": receipt_cpu / calls if receipt_cpu else None, "proof": proof_cpu / calls}
        s["overhead_per_call"] = {
            "gpu_edited_over_plain": edited / plain,
            "cpu_core_s": per_call_cpu,
            "cpu_core_s_over_plain_gpu_s": sum(v for v in per_call_cpu.values() if v) / (plain / 1e3),
            "gpu_edited_plus_commit_kernel_over_plain": (edited + commit_gpu) / plain if commit_gpu else None,
        }
    print(json.dumps(s, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
