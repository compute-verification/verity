"""verity-vllm's tests in N parts, each a `suites.py verity-vllm` run of its own, so N jobs run the suite at once.

    python3 integrations/vllm/tools/suite_parts.py 2/8             part 2 of 8: its test modules, from integrations/vllm
    python3 integrations/vllm/tools/suite_parts.py --list 8        each part's modules and estimated seconds
    python3 integrations/vllm/tools/suite_parts.py launch          the 8 parts as `research run --on auto` jobs; their run ids
    python3 integrations/vllm/tools/suite_parts.py launch 8 -- --fresh     with suites.py options (--fresh, --quick)

The suite is CPU-bound at about 12000 CPU-s on 4-CPU jobs (13462 tests in 6 parts took 12320 CPU-s), so one 4-CPU job can't run it in under about 50 min, and `tests/program` alone, one flat directory, is half of
it. A part is a list of test modules handed to pytest (`suites.py verity-vllm -- <modules>`), so the suite, its inputs, its
guard and its configuration stay as they are. suites.py keys every pass on the pytest arguments (the suite's, each module's
and the whole run's traced pass), so a part's pass answers only for that part, never for the suite or for another part.

The parts are a function of the tree alone, so N jobs on N machines agree on them: every test module pytest collects under
the suite's `testpaths` (`modules`) goes to exactly one part, longest first, each to the part with the least so far
(`parts`). A module's estimate is its seconds in `SECONDS` (measured; a module renamed since gets `DEFAULT_S`, which only
unbalances the parts), so a stale table costs balance, never a test.

`launch` starts each part as `research run --on auto` does for any suite run, with `--jobs` the CPUs the job holds:
suites.py's default counts each process at its `TEST_PROCESS_GB` of 9 GB, which runs the suite in one process in a 16 GB job.
Each part keeps its suites.py logs in its run's `out/logs`, whose `integrations_vllm.json` `--seconds` reads.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

SUITE = Path(__file__).resolve().parents[1]
ROOT = SUITE.parents[1]
TESTPATHS = ("tests",)
#: pytest's default `python_files` and `norecursedirs` (the suite's pyproject.toml sets neither)
PYTHON_FILES = ("test_*.py", "*_test.py")
NORECURSE = ("*.egg", ".*", "_darcs", "build", "CVS", "dist", "node_modules", "venv", "{arch}", "__pycache__")
DEFAULT_PARTS = 8
#: a module SECONDS doesn't name: about the mean of those it leaves out
DEFAULT_S = 2.6
#: each test module's seconds of test time, the modules over 20 s, from a run in 8 parts (each part's 4 workers on 4 CPUs);
#: `--seconds REPORT...` prints the table from the suites.py reports of a full run or of every part of one (each part's
#: `out/logs/integrations_vllm.json`)
SECONDS: dict[str, float] = {
    "tests/protocol_options/test_pouw_h1_handoff.py": 481,
    "tests/program/test_sparse_mla_split.py": 455,
    "tests/predict/test_predict.py": 448,
    "tests/program/test_boolean_moe.py": 414,
    "tests/program/test_dsv4_qnorm_insert.py": 276,
    "tests/predict/test_batch.py": 221,
    "tests/program/test_gumbel_boolean.py": 220,
    "tests/predict/test_census.py": 197,
    "tests/program/test_boolean_dsv4_moe.py": 179,
    "tests/properties/test_glm53_tp8_allreduce_capture.py": 178,
    "tests/program/test_boolean_attention.py": 169,
    "tests/program/test_topp_words.py": 160,
    "tests/program/test_deepgemm.py": 154,
    "tests/protocol_options/test_pouw_h1.py": 153,
    "tests/protocol_options/test_pouw_h1_credit.py": 150,
    "tests/program/test_sampling_split.py": 148,
    "tests/check/test_pouw_ncp_v2_end_to_end.py": 146,
    "tests/pipeline/test_single_request_build_path.py": 138,
    "tests/program/test_bf16_gemm.py": 138,
    "tests/predict/test_steps.py": 137,
    "tests/program/test_boolean_dense.py": 137,
    "tests/program/test_dead_free.py": 134,
    "tests/store/test_remote_session.py": 111,
    "tests/program/test_sampling_shared_greedy.py": 108,
    "tests/program/test_fa2_split.py": 108,
    "tests/program/test_lifted_dense_b1.py": 108,
    "tests/properties/test_glm53_fa3_prefill_capture.py": 108,
    "tests/program/test_boolean_norms.py": 105,
    "tests/program/test_boolean_rope.py": 103,
    "tests/program/test_heldout_codec_compose.py": 97,
    "tests/engine/test_int8_zy.py": 83,
    "tests/program/test_boolean_softcap.py": 82,
    "tests/check/test_global_match.py": 80,
    "tests/properties/test_dsv4_flashmla_capture.py": 78,
    "tests/program/test_fa3_split.py": 77,
    "tests/query/test_query_counterexamples.py": 76,
    "tests/check/test_fold_sampler_construction.py": 74,
    "tests/program/test_boolean_tanh.py": 72,
    "tests/program/test_tp_allreduce.py": 72,
    "tests/program/test_kernel_self_check.py": 71,
    "tests/program/test_lifted_mixed.py": 71,
    "tests/program/test_dsv4_derive.py": 68,
    "tests/check/test_sampled_replay.py": 68,
    "tests/program/test_sparse_mla.py": 67,
    "tests/program/test_derive_stochastic.py": 65,
    "tests/program/test_sampling_boolean.py": 65,
    "tests/program/test_sampling_rows.py": 65,
    "tests/program/test_mla.py": 63,
    "tests/predict/test_node.py": 63,
    "tests/program/test_boolean_sigma.py": 61,
    "tests/program/test_derive.py": 59,
    "tests/program/test_dsv4_compressor.py": 57,
    "tests/program/test_boolean_dense_norm.py": 54,
    "tests/program/test_dsa_derive.py": 48,
    "tests/query/test_query_fixtures.py": 48,
    "tests/program/test_boolean_silu.py": 47,
    "tests/check/test_glm53_rank_rehearsal.py": 47,
    "tests/query/test_word.py": 46,
    "tests/program/test_compact.py": 46,
    "tests/pipeline/test_serving_rows_deepgemm.py": 44,
    "tests/program/test_tp_allreduce_multimem.py": 44,
    "tests/program/test_token_select_chunks.py": 42,
    "tests/program/test_lifted_mixed_padrev.py": 40,
    "tests/properties/test_dsv4_compressor_capture.py": 38,
    "tests/program/test_mhc.py": 37,
    "tests/program/test_sigmoid_moe_fp8_derive.py": 37,
    "tests/program/test_fp8_shared_scale.py": 37,
    "tests/program/test_fa2_softcap.py": 36,
    "tests/properties/test_golden.py": 36,
    "tests/program/test_lifted_tiny.py": 34,
    "tests/lint/test_p13_duplicate_bodies.py": 34,
    "tests/program/test_lifted_workload_padrev.py": 34,
    "tests/program/test_codec.py": 34,
    "tests/lint/test_p01_core_abstractions.py": 33,
    "tests/program/test_topp_splits_operand.py": 33,
    "tests/program/test_flashmla.py": 32,
    "tests/program/test_boolean_gather.py": 32,
    "tests/pipeline/test_tp_build_relabel.py": 32,
    "tests/program/test_boolean_fp8.py": 30,
    "tests/program/test_fp8_moe.py": 29,
    "tests/check/test_sampled_replay_glm53_rows.py": 29,
    "tests/pipeline/test_run_config_dry_run.py": 29,
    "tests/pipeline/test_circuit_db.py": 27,
    "tests/program/test_tp_allreduce_parts.py": 25,
    "tests/lint/test_p09_layering.py": 25,
    "tests/properties/test_dsv4_moe_capture.py": 24,
    "tests/query/test_boolean_manifest.py": 24,
    "tests/properties/test_glm53_fused_rmsnorm_capture.py": 24,
    "tests/program/test_lifted_request.py": 24,
    "tests/properties/test_sigmoid_router_capture.py": 23,
    "tests/program/test_b1_authored.py": 23,
    "tests/program/test_moe_router_sigmoid.py": 23,
    "tests/program/test_boolean_marlin_moe.py": 23,
    "tests/lint/test_p07_declared_inputs.py": 22,
    "tests/program/test_topp_split_geometry.py": 22,
    "tests/pipeline/test_admit_r19_host_working_set.py": 22,
    "tests/regression/test_attempt_resolution.py": 21,
    "tests/program/test_fp8_block_dot.py": 21,
    "tests/check/test_capture_map_dsv4.py": 20,
    "tests/protocol_options/test_pouw_native.py": 20,
}
#: what each part's job runs, from the repository's root; {k}/{n} names the part and {opts} the suites.py options. A part that
#: lists nothing stops the job: suites.py with no modules would run the whole suite
JOB = ("uv sync -q --all-packages --extra torch-cpu && P=$(python3 integrations/vllm/tools/suite_parts.py {k}/{n})"
       " && test -n \"$P\" && uv run tools/verity/check/suites.py verity-vllm"
       " --jobs \"$(nproc)\" --logs \"$OUT/logs\"{opts} -- $P")
#: the memory each part's job asks for: 4 CPUs' workers at the suite's worker_gb
JOB_MEM_GB = 32


def _skipped(name: str) -> bool:
    return any(Path(name).match(p) for p in NORECURSE)


def modules(suite: Path = SUITE) -> list[str]:
    """Every test module pytest collects from the suite's `testpaths`, by its path from the suite's directory."""
    out = []
    for top in TESTPATHS:
        stack = [suite / top]
        while stack:
            d = stack.pop()
            for p in d.iterdir():
                if p.is_dir():
                    if not _skipped(p.name):
                        stack.append(p)
                elif any(p.match(g) for g in PYTHON_FILES):
                    out.append(p.relative_to(suite).as_posix())
    return sorted(out)


def parts(names: list[str], n: int, seconds: dict[str, float] = SECONDS) -> list[list[str]]:
    """``names`` in ``n`` parts: longest first (by ``seconds``, else `DEFAULT_S`; ties by name), each to the part with the fewest
    seconds so far (the lowest-numbered of those tied); each part's names sorted."""
    if n < 1:
        raise ValueError(f"{n} parts: there must be at least one")
    load, out = [0.0] * n, [[] for _ in range(n)]
    for m in sorted(names, key=lambda m: (-seconds.get(m, DEFAULT_S), m)):
        k = min(range(n), key=lambda i: (load[i], i))
        load[k] += seconds.get(m, DEFAULT_S)
        out[k].append(m)
    return [sorted(p) for p in out]


def part(spec: str, names: list[str] | None = None) -> list[str]:
    """Part ``spec`` (``K/N``, K from 1) of `modules`."""
    m = re.fullmatch(r"(\d+)/(\d+)", spec)
    if not m or not 1 <= int(m[1]) <= int(m[2]):
        raise ValueError(f"{spec!r}: a part is K/N with 1 <= K <= N")
    return parts(modules() if names is None else names, int(m[2]))[int(m[1]) - 1]


def table(report: dict, floor_s: float = 20.0) -> dict[str, float]:
    """{module: seconds} from a suites.py report's per-test seconds (``passed``), the modules of at least ``floor_s``."""
    total: dict[str, float] = {}
    for node, s in report.get("passed", {}).items():
        total[node.split("::", 1)[0]] = total.get(node.split("::", 1)[0], 0.0) + s
    return {m: round(s) for m, s in sorted(total.items(), key=lambda x: (-x[1], x[0])) if s >= floor_s}


def jobs(n: int, opts: list[str] = ()) -> list[list[str]]:
    """The `research run` command of each of the ``n`` parts, with suites.py's ``opts``."""
    o = "".join(f" {x}" for x in opts)
    return [["research", "run", "--on", "auto", "--mem-gb", str(JOB_MEM_GB), "--cwd", "source", "--",
             "bash", "-lc", JOB.format(k=k, n=n, opts=o)] for k in range(1, n + 1)]


def launch(n: int, opts: list[str]) -> int:
    """Start the ``n`` parts at once, from the repository's root, and print each one's run id (or why it didn't start)."""
    procs = [subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
             for cmd in jobs(n, opts)]
    rc = 0
    for k, p in enumerate(procs, 1):
        out = p.communicate()[0]
        run = re.search(r"run (r\d{8}-\d{6}-[0-9a-f]+) launched", out)
        print(f"part {k}/{n}: {run[1] if run else 'not started: ' + out.strip()[-500:]}")
        rc = rc or (0 if run and p.returncode == 0 else 1)
    if rc == 0:
        print("each run's suites.py summary: `research fetch RUN` and its stdout.log; the parts add up to the suite")
    return rc


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    opts = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(prog="suite_parts", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", nargs="?", help="K/N (part K of N's test modules), or `launch`")
    ap.add_argument("n", nargs="?", type=int, default=DEFAULT_PARTS,
                    help=f"launch: the number of parts (default {DEFAULT_PARTS})")
    ap.add_argument("--list", type=int, metavar="N", help="each of N parts' modules and estimated seconds")
    ap.add_argument("--seconds", type=Path, nargs="+", metavar="REPORT",
                    help="SECONDS from suites.py reports of a full run, or of every part of one")
    a = ap.parse_args(argv)
    if a.seconds:
        import json
        passed = {k: v for r in a.seconds for k, v in json.loads(r.read_text()).get("passed", {}).items()}
        for m, s in table({"passed": passed}).items():
            print(f'    "{m}": {s},')
        return 0
    if a.list:
        for k, p in enumerate(parts(modules(), a.list), 1):
            print(f"part {k}/{a.list}: {len(p)} modules, about {sum(SECONDS.get(m, DEFAULT_S) for m in p):.0f} s")
        return 0
    if a.what == "launch":
        return launch(a.n, opts)
    if a.what is None:
        ap.error("name a part (K/N), `launch` or --list N")
    try:
        print("\n".join(part(a.what)))
    except ValueError as e:
        ap.error(str(e))
    return 0


if __name__ == "__main__":
    sys.exit(main())
