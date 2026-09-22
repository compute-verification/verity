"""Assemble the §15.5 compiled-vs-eager A/B experiment directories from pulled recorder output (laptop side, CPU only).

For one case (B1 = Qwen/Qwen2.5-1.5B, B7 = Qwen/Qwen2.5-7B) and the two profiles recorded on the pod
(``<pulled>/<case>/<prefix>-compiled``, ``<prefix>-eager`` and their ``-r2`` fresh-process repetitions)
this writes one experiment directory per profile, ``<results_root>/<case>-<profile_id>-<utc>/``:

* ``production/`` and ``production-r2/``: the recorder's records (tokens, chosen logprobs, manifests,
  steptrace, timings), copied verbatim; ``pipeline.log`` of both runs;
* ``kernels.json``: the torch.profiler kernel-identity fixture of the same model, flags and execution
  mode on this device (bench ``C1`` for the compiled profile, ``C2`` for the eager profile);
* ``table1.jsonl`` (verdict NOT_RUN), ``table2.jsonl`` (kernel paths), ``token_comparison.json`` via
  ``ab_rows``: this profile vs its r2, vs the other profile on this device, vs every ``--ref
  LABEL=path`` record from other devices;
* ``README.md``: the comparison tables (bit comparisons, no tolerance).

    python -m verity_vllm_adapter.bench.ab_collect --case B1 --pulled /tmp/pull/ab/B1 --prefix h100 \
        --model-bench /tmp/pull/bench/Qwen2.5-1.5B --results-root fixtures/results --adapter-revision <sha> \
        --ref a100-compiled=results/B1-7fad95fc27aa-20260907T1322Z/production/production.jsonl ...
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def load_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def pair_table(pairs: dict[str, Any]) -> str:
    lines = ["| pair | requests identical | positions token-equal | logprobs bit-equal (all positions) | logprobs bit-equal on common prefix | other profile |",
             "|---|---|---|---|---|---|"]
    for k, v in pairs.items():
        lines.append(f"| `{k}` | **{v['requests_identical']} / {v['requests_compared']}** | {v['positions_token_equal']} / {v['positions_compared']} "
                     f"| {v['positions_logprob_bit_equal_all']} / {v['positions_compared']} | {v['positions_logprob_bit_equal_on_common_prefix']} / {v['positions_on_common_prefix']} "
                     f"| `{v.get('other_profile_id')}` |")
    return "\n".join(lines)


def divergences(v: dict[str, Any]) -> str:
    d = [p for p in v["per_request"] if not p["identical"]]
    if not d:
        return "none"
    return "; ".join(f"`{p['request_id'][:8]}` (prompt {p['prompt_len']}, first differing output index {p['first_differing_output_index']}, "
                     f"position {p['first_differing_position']})" for p in d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--pulled", required=True, help="<pulled>/<case> directory with <prefix>-compiled, <prefix>-eager, *-r2")
    ap.add_argument("--prefix", default="h100")
    ap.add_argument("--model-bench", required=True, help="bench directory of the same model with C1/kernels.json and C2/kernels.json")
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--adapter-revision", required=True)
    ap.add_argument("--worker", default="W4d")
    ap.add_argument("--ref", action="append", default=[], help="LABEL=path/to/production.jsonl from another device (compared against BOTH profiles)")
    args = ap.parse_args()

    pulled = Path(args.pulled)
    bench = Path(args.model_bench)
    root = Path(args.results_root)
    profiles = {"compiled": bench / "C1" / "kernels.json", "eager": bench / "C2" / "kernels.json"}
    runs: dict[str, dict[str, Any]] = {}
    for prof in profiles:
        d = pulled / f"{args.prefix}-{prof}"
        run = json.loads((d / "production" / "run.json").read_text())
        runs[prof] = {"dir": d, "r2": pulled / f"{args.prefix}-{prof}-r2", "run": run,
                      "experiment_id": f"{args.case}-{run['profile_id']}-{run['utc']}"}
    written = []
    for prof, info in runs.items():
        other = "eager" if prof == "compiled" else "compiled"
        out = root / info["experiment_id"]
        out.mkdir(parents=True, exist_ok=True)
        for sub, src in (("production", info["dir"] / "production"), ("production-r2", info["r2"] / "production")):
            if (out / sub).exists():
                shutil.rmtree(out / sub)
            shutil.copytree(src, out / sub)
        shutil.copy(info["dir"] / "pipeline.log", out / "pipeline.log")
        shutil.copy(info["r2"] / "pipeline.log", out / "pipeline-r2.log")
        kern = profiles[prof]
        if kern.exists():
            shutil.copy(kern, out / "kernels.json")
        cmd = [sys.executable, "-m", "verity_vllm_adapter.bench.ab_rows", "--case", args.case,
               "--production", str(out / "production"), "--production-r2", str(out / "production-r2"),
               "--experiment-id", info["experiment_id"], "--adapter-revision", args.adapter_revision, "--worker", args.worker,
               "--profile-label", f"{args.prefix}-{prof}", "--out", str(out),
               "--other", f"{args.prefix}-{other}={runs[other]['dir'] / 'production' / 'production.jsonl'}"]
        if kern.exists():
            cmd += ["--kernels", str(out / "kernels.json")]
        for ref in args.ref:
            cmd += ["--other", ref]
        print("+", " ".join(cmd[2:]), flush=True)
        subprocess.run(cmd, check=True)

        cmp = json.loads((out / "token_comparison.json").read_text())
        man = json.loads((out / "production" / "manifest.json").read_text())
        wex = json.loads((out / "production" / "workload_exercised.json").read_text())
        recs = load_jsonl(out / "production" / "production.jsonl")
        comp = man["compilation"]
        r2_run = json.loads((out / "production-r2" / "run.json").read_text())
        lines = [
            f"# {info['experiment_id']}: {man['model']['repo'] if isinstance(man.get('model'), dict) else args.case} production only, "
            f"**{prof}** batch-invariant profile on Hopper ({man['gpu']['name']}, sm_{man['gpu']['compute_capability'].replace('.', '')})",
            "",
            f"Worker {args.worker}, branch `vllm-poc-w4d-h100-measure`, adapter `{args.adapter_revision}`, pod `{man['gpu']['pod']}`. Recorder: "
            f"`verity_vllm_adapter/pod/run_case.sh {args.case} {args.prefix}-{prof} <rev> --production-only{' --enforce-eager' if comp.get('enforce_eager') else ''}`; "
            f"a second fresh-process run of the same profile is `production-r2/` (profile `{r2_run['profile_id']}`, {r2_run['utc']}).",
            "",
            "**This is not a Table 1 verification row** (`table1.jsonl` verdict `NOT_RUN`): no replay, no commitments, no verify (`--production-only`). "
            "It is the CONTRACT §15.5 step-1 diagnostic on Hopper: the adapter's recorder (first pass, fresh engine, 16 greedy requests, "
            f"{sum(len(r['output_token_ids']) for r in recs)} generated positions) under the **{prof}** profile "
            f"(mode={comp['mode']}, cudagraph={comp['cudagraph_mode']}, custom_ops={comp.get('custom_ops')}, enforce_eager={comp.get('enforce_eager')}, "
            f"flags {json.dumps(man['engine_flags'])}), compared bit for bit with the other profile on the same GPU and with the records of the same "
            "workload on other devices already in the repo. No tolerance anywhere.",
            "",
            f"Recorder invariants: stock-vs-stock identical {json.loads((out / 'production' / 'run.json').read_text()).get('stock_vs_stock_identical')}, "
            f"cached-prefix pass identical {json.loads((out / 'production' / 'run.json').read_text()).get('cached_prefix_pass_identical')}, "
            f"in-workload prefix-cache hits {wex.get('in_workload_prefix_cache_hits')}, cached tokens from outside the recorded pass "
            f"{wex.get('cached_tokens_from_outside_the_recorded_pass')}, peak concurrency achieved {wex.get('peak_concurrency_achieved')}, "
            f"engine steps {wex.get('engine_steps')}, preemptions {wex.get('preemptions')}.",
            "",
            "## Bit comparisons (tokens as ints; chosen-token logprobs as the fp32 values the recorder serialised)",
            "",
            pair_table(cmp["pairs"]),
            "",
            "Positions after a request's first divergence follow different trajectories; they count in the *all positions* column only. "
            "The *common prefix* column restricts to positions where both records chose the same token in the same context.",
            "",
            "### Divergent requests per pair",
            "",
        ]
        for k, v in cmp["pairs"].items():
            lines.append(f"- `{k}`: {divergences(v)}")
        lines += [
            "",
            "## Kernel identity (`table2.jsonl`, `kernels.json`)",
            "",
            f"`kernels.json` is the torch.profiler pass of `verity_vllm_adapter/bench/throughput.py` for the same model under the bench configuration "
            f"{'C1 (batch-invariant, compiled + CUDA graphs)' if prof == 'compiled' else 'C2 (batch-invariant, enforce_eager)'} on this pod: the same "
            "declared flags and the same effective execution mode as this profile. `table2.jsonl` has one `kernel_path_resolved` row per op family "
            "(`checker = none`; identity fixture per §16.8, no relation executed).",
            "",
            "Files: `production/{production.jsonl,manifest.json,run.json,steptrace.json,timings.json,workload_exercised.json}`, `production-r2/` (same layout), "
            "`token_comparison.json`, `kernels.json`, `table1.jsonl`, `table2.jsonl`, `pipeline.log`, `pipeline-r2.log`. "
            "Cross-device and cross-profile findings are discussed in `docs/vllm-poc/measurement-h100.md`.",
        ]
        (out / "README.md").write_text("\n".join(lines) + "\n")
        written.append(str(out))
    print("wrote", *written, sep="\n  ")


if __name__ == "__main__":
    main()
