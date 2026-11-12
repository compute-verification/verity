"""Hold `sampled_accept`'s numpy reference to the kept PTX of the four kernels, off the card.

The four kernels' PTX (the run that compiled them, --ptx) are interpreted thread by thread (`ptx_sim`) on the tensors
`rejection_sample` hands them for one request of K draft tokens over V logits, in the pin's launch order; what they
store is compared word for word with `sampled_accept.sampled_accept` on the same inputs.  With --definition N the
registered `SampledAccept_v1{K,V,TILE=8192}` is evaluated on the first N cases too.

A case whose reference reports `halves_split` (lane 0 accepts a draft token that is not the last and lanes 16-31 refuse
it) is outside the Definition's domain: the kernel reads registers of lanes that stopped, which the interpreter gives as
they stand.  Such a case is reported apart and counts in neither total.

Exit status: 0 when every compared word of the cases inside the domain agrees, 1 otherwise.  --out keeps each case's
inputs and both sides' outputs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ptx_sim  # noqa: E402

from verity.definitions.registry import sampled_accept as A  # noqa: E402
from verity.definitions.registry import sampling as SA  # noqa: E402

sys.path.insert(0, str(Path("adapters/vllm").resolve()))
from tests.program import sampled_accept_vectors as SV  # noqa: E402

F = np.float32
KERNELS = ("_compute_local_logits_stats_kernel", "_rejection_kernel", "_resample_kernel", "_insert_resampled_kernel")
POISON = 0xA5A5A5A5


def launch(ptx_dir: Path, manifest: dict, logits: np.ndarray, draft, seed: int, pos, temp_bits: int) -> dict:
    K, V = len(draft), logits.shape[1]
    L = K + 1
    tiles, rtiles = -(-V // 8192), -(-V // 1024)
    heap = ptx_sim.Heap(L * V + 4 * L * tiles + 2 * rtiles + 8192)
    val = {
        "target_logits_ptr": heap.put("logits", np.ascontiguousarray(logits, F)), "target_logits_stride": V,
        "target_local_argmax_ptr": heap.empty("target_local_argmax", (L, tiles), np.int64), "target_local_argmax_stride": tiles,
        "target_local_max_ptr": heap.empty("target_local_max", (L, tiles), F), "target_local_max_stride": tiles,
        "target_local_sumexp_ptr": heap.empty("target_local_sumexp", (L, tiles), F), "target_local_sumexp_stride": tiles,
        "draft_local_max_ptr": heap.empty("draft_local_max", (L, tiles), F), "draft_local_max_stride": tiles,
        "draft_local_sumexp_ptr": heap.empty("draft_local_sumexp", (L, tiles), F), "draft_local_sumexp_stride": tiles,
        "draft_logits_stride_0": 0, "draft_logits_stride_1": 0,
        "expanded_idx_mapping_ptr": heap.put("expanded_idx_mapping", np.zeros(L, np.int64)),
        "expanded_local_pos_ptr": heap.put("expanded_local_pos", np.arange(L, dtype=np.int32)),
        "temp_ptr": heap.put("temperature", np.full(16, temp_bits, np.uint32)),
        "vocab_size": V, "num_speculative_steps": K,
        "sampled_ptr": heap.empty("sampled", (1, L), np.int64), "sampled_stride": L,
        "rejected_steps_ptr": heap.empty("num_sampled", (1,), np.int32),
        "target_rejected_logsumexp_ptr": heap.empty("target_rejected_logsumexp", (1,), F),
        "draft_rejected_logsumexp_ptr": heap.empty("draft_rejected_logsumexp", (1,), F),
        "draft_sampled_ptr": heap.put("draft_sampled", np.array([0] + [int(d) for d in draft], np.int32)),
        "cu_num_logits_ptr": heap.put("cu_num_logits", np.array([0, L], np.int32)),
        "idx_mapping_ptr": heap.put("idx_mapping", np.zeros(1, np.int64)),
        "seed_ptr": heap.put("seed", np.full(16, seed, np.int64)),
        "pos_ptr": heap.put("pos", np.array(pos, np.int64)),
        "local_residual_mass_stride": 0, "vocab_num_blocks": tiles,
        "resampled_local_argmax_ptr": heap.empty("resampled_local_argmax", (1, rtiles), np.int64), "resampled_local_argmax_stride": rtiles,
        "resampled_local_max_ptr": heap.empty("resampled_local_max", (1, rtiles), F), "resampled_local_max_stride": rtiles,
        "resample_num_blocks": rtiles,
    }
    val["rejected_step_ptr"] = val["num_sampled_ptr"] = val["rejected_steps_ptr"]
    out: dict = {"undefined": {}, "steps": {}}
    by_name = {k["name"]: k for k in manifest["kernels"]}
    for name in KERNELS:
        k = by_name[name]
        params = [val[p["name"]] for p in k["params"] if p["specialization"][0] != "constexpr"]
        grid = {"_compute_local_logits_stats_kernel": (L, tiles), "_resample_kernel": (1, rtiles)}.get(name, (1,))
        run = ptx_sim.Run((ptx_dir / name / f"{name}.ptx").read_text(), 32 * k["options"]["num_warps"], grid, params, heap.mem)
        out["undefined"][name], out["steps"][name] = run.undefined, run.steps
        if name == "_rejection_kernel":
            out["num_sampled_after_rejection"] = int(heap.get("num_sampled")[0])
            out["sampled_after_rejection"] = heap.get("sampled")[0].view(np.uint64).tolist()
    for name in ("target_local_max", "target_local_sumexp", "target_local_argmax", "draft_local_max", "draft_local_sumexp",
                 "sampled", "num_sampled", "resampled_local_argmax", "resampled_local_max", "target_rejected_logsumexp",
                 "draft_rejected_logsumexp"):
        out[name] = heap.get(name)
    return out


def compare(got: dict, ref: dict, draft) -> tuple[int, int, list[str]]:
    """(words compared, words that differ, what differed)."""
    K = len(draft)
    n = bad = 0
    notes: list[str] = []

    def eq(what, a, b):
        nonlocal n, bad
        a, b = np.asarray(a).reshape(-1), np.asarray(b).reshape(-1)
        assert a.shape == b.shape, (what, a.shape, b.shape)
        d = int((a != b).sum())
        n += a.size
        bad += d
        if d:
            notes.append(f"{what}: {d} of {a.size} differ")

    eq("tile max, draft rows", got["target_local_max"][:K].view(np.uint32), np.array(ref["max"], F).view(np.uint32))
    eq("tile sumexp, draft rows", got["target_local_sumexp"][:K].view(np.uint32), np.array(ref["sumexp"], F).view(np.uint32))
    r = ref["accepted"]
    eq("accepted count after the rejection kernel", [got["num_sampled_after_rejection"]], [r])
    eq("accepted count + 1 after the insert", got["num_sampled"], [r + 1])
    eq("appended tokens", got["sampled"][0][:r + 1], np.array(ref["appended"][:r + 1], np.int64))
    # lane 0's log-sum-exp of the last position it tested (0.0 when it tested none: a negative first draft token)
    tested = [i for i in range(K) if i <= r and draft[i] >= 0 and all(draft[j] >= 0 for j in range(i))]
    eq("lane 0's last log-sum-exp", got["target_rejected_logsumexp"].view(np.uint32), [ref["lse"][tested[-1]] if tested else 0])
    eq("the draft's log-sum-exp (a one-hot draft: 0.0)", got["draft_rejected_logsumexp"].view(np.uint32), [0])
    return n, bad, notes


def poison_report(got: dict, K: int) -> dict:
    """Which outputs no kernel stored to: the bonus row's stats, the draft's stats, the greedy argmax, the two log-sum-exps."""
    w = lambda a: np.ascontiguousarray(a).view(np.uint32)
    return {"bonus_row_stats_unwritten": bool((w(got["target_local_max"][K]) == POISON).all() and (w(got["target_local_sumexp"][K]) == POISON).all()),
            "draft_stats_unwritten": bool((w(got["draft_local_max"]) == POISON).all() and (w(got["draft_local_sumexp"]) == POISON).all()),
            "target_argmax_unwritten": bool((w(got["target_local_argmax"]) == POISON).all()),
            "resample_tiles_written": int((w(got["resampled_local_max"]) != POISON).sum())}


def one(job) -> dict:
    ptx_dir, manifest, v, definition, out, (steps, vocab, seed) = job
    if isinstance(v, int):
        v = SV.random_rows(steps, vocab, 1, seed, first=v)[0]
    name, logits, draft, seed, pos, temp, ref = (v[k] for k in ("name", "logits", "draft", "seed", "pos", "temp", "ref"))
    t0 = time.time()
    got = launch(ptx_dir, manifest, logits, draft, seed, pos, temp)
    t1 = time.time()
    n, bad, notes = compare(got, ref, draft)
    entry = {"case": name, "draft": draft, "verdict": ref["verdict"], "accepted": ref["accepted"], "appended": ref["appended"],
             "kernel_sampled": [int(x) for x in got["sampled"][0][:int(got["num_sampled"][0])]],
             "kernel_num_sampled": int(got["num_sampled"][0]), "halves_split": ref["halves_split"],
             "in_domain": ref["halves_split"] is None,
             "lse": [hex(x) for x in ref["lse"]], "lse_upper_half": [hex(x) for x in ref["lse_hi"]],
             "u": [hex(x) for x in ref["u"]], "words_compared": n, "words_differ": bad, "notes": notes,
             "undefined_shuffles": {k: v for k, v in got["undefined"].items() if v}, **poison_report(got, len(draft)),
             "seconds_ptx": round(t1 - t0, 1)}
    if definition:
        from verity.circuits.defs import bind
        from verity.circuits.evaluation.reference import evaluate

        t2 = time.time()
        d = evaluate(bind(A.SampledAccept, K=len(draft), V=logits.shape[1], TILE=A.STATS_TILE),
                     [int(w) for w in np.ascontiguousarray(logits, F).view(np.uint32).reshape(-1)],
                     [x & 0xFFFFFFFF for x in draft], seed, pos, temp)
        want = tuple(ref["verdict"]) + (ref["accepted"],) + tuple(x & 0xFFFFFFFF for x in ref["appended"])
        entry["definition"] = list(d)
        entry["definition_words_compared"] = len(want)
        entry["definition_words_differ"] = sum(int(x != y) for x, y in zip(d, want)) + abs(len(d) - len(want))
        entry["seconds_definition"] = round(time.time() - t2, 1)
    if out:
        np.savez_compressed(Path(out) / f"{name}.npz", logits=logits, draft=np.array(draft, np.int64), seed=np.int64(seed),
                            pos=np.array(pos, np.int64), temp=np.uint32(temp),
                            **{f"kernel_{k}": v for k, v in got.items() if isinstance(v, np.ndarray)},
                            ref_max=np.array(ref["max"], F), ref_sumexp=np.array(ref["sumexp"], F),
                            ref_verdict=np.array(ref["verdict"]), ref_appended=np.array(ref["appended"]))
    return entry


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptx", required=True, help="the directory the compile run kept (manifest.json and a directory a kernel)")
    ap.add_argument("--out")
    ap.add_argument("--vocab", type=int, default=154880)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--random", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--hand", action="store_true", help="the hand vectors too (tests.program.sampled_accept_vectors, from the tree's root)")
    ap.add_argument("--definition", type=int, default=0, metavar="N", help="evaluate SampledAccept_v1 on the first N cases too")
    ap.add_argument("--keep", type=int, default=24, metavar="N", help="with --out: keep the first N cases' tensors (every case's entry is kept)")
    ap.add_argument("--jobs", type=int, default=max(1, len(os.sched_getaffinity(0)) // 2))
    a = ap.parse_args()
    ptx_dir = Path(a.ptx)
    manifest = json.loads((ptx_dir / "manifest.json").read_text())
    assert manifest["call"]["vocab"] == a.vocab and manifest["call"]["steps"] == a.steps, "the PTX is specialized for its own V and K"
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
    todo = (SV.vectors(a.steps, a.vocab) if a.hand else []) + list(range(a.random))
    jobs = [(ptx_dir, manifest, v, i < a.definition, a.out if i < a.keep else None, (a.steps, a.vocab, a.seed))
            for i, v in enumerate(todo)]
    report = []
    with ProcessPoolExecutor(a.jobs) as pool:
        for entry in pool.map(one, jobs):
            report.append(entry)
            print(json.dumps(entry), flush=True)
    inside = [e for e in report if e["in_domain"]]
    outside = [e for e in report if not e["in_domain"]]
    summary = {
        "cases": len(report), "cases_in_domain": len(inside),
        "cases_in_domain_that_differ": sum(bool(e["words_differ"]) for e in inside),
        "words_compared": sum(e["words_compared"] for e in inside), "words_differ": sum(e["words_differ"] for e in inside),
        "cases_outside_domain": [{"case": e["case"], "words_compared": e["words_compared"], "words_differ": e["words_differ"],
                                  "undefined_shuffles": e["undefined_shuffles"]} for e in outside],
        "definition_cases": sum("definition" in e for e in report),
        "definition_words_compared": sum(e.get("definition_words_compared", 0) for e in report),
        "definition_words_differ": sum(e.get("definition_words_differ", 0) for e in report),
        "undefined_shuffles_in_domain": sum(bool(e["undefined_shuffles"]) for e in inside),
        "accepted_counts": {str(k): sum(e["accepted"] == k for e in report) for k in range(a.steps + 1)},
        "bonus_row_stats_unwritten": all(e["bonus_row_stats_unwritten"] for e in report),
        "draft_stats_unwritten": all(e["draft_stats_unwritten"] for e in report),
        "target_argmax_unwritten": all(e["target_argmax_unwritten"] for e in report),
        "ptx_sha256": {k["name"]: k["files"]["ptx"]["sha256"] for k in manifest["kernels"]}}
    print(json.dumps(summary))
    if a.out:
        (Path(a.out) / "ptx_check.json").write_text(json.dumps({"summary": summary, "cases": report}, indent=1) + "\n")
    return 1 if summary["words_differ"] or summary["definition_words_differ"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
