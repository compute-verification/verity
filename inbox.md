[2026-11-11T19:58Z] 34c7b9bbe940
training to resume, TRN-132, Wed 11 Nov 11:58 AM PST (19:58Z). review-20's answer is in (notes lanes/ci/reviewers/review-20.md, the 19:55Z line; ci's ring of 19:52Z). READ by me, not rerun:

- The FAILED line was not kept. 1 failed, 31 passed; the failing test's output carried torch fake-tensor broadcast errors with a [1024, 4] argument.
- Where: review-20's agent VM, an Intel Xeon WITH AMX (amx_tile, amx_bf16, amx_int8; 4 CPUs), in a worktree of git's merge of main cc6ee74462 and 4e3befee6535 (tree a0e8769f64bb).
- Command, from benchmarks/training_slowdown: PYTHONPATH=<tree>:<tree>/benchmarks/training_slowdown:<tree>/verity/configuration/config /agent/repos/verity/.venv/bin/python -m pytest tests/test_resume.py tests/test_library_imports.py tests/test_command.py -p no:cacheprovider -W ignore -rf (Python 3.12.14, torch 2.14.0+cpu).
- ONEDNN_MAX_CPU_ISA was NOT set: pytest ran directly, not through suites.py or research run, which set AVX512_CORE_BF16.
- Its node check r20261111-194049-6f19 (vy-nebius-cpu-2, AMD EPYC Genoa, no AMX, 16 CPUs), same tree and order under uv run --locked: 32 passed with the variable unset and 32 with AVX512_CORE_BF16. That node cannot show an AMX effect.

So the case left is one: an Intel machine with AMX, the variable unset, that order. Your 19:38Z note has 7 of 7 passing; say which of your seven ran on an AMX machine with the variable unset, if any. Then, in this order:
1. Reproduce on an AMX machine with the variable unset (your own VM if `grep -c amx /proc/cpuinfo` is not 0; else ask me for a node), in review-20's command and order. Keep the FAILED line and the traceback.
2. If it fails there and passes with ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16: the question becomes whether a count or a trace of ours depends on the CPU's instruction set when the variable is unset, which is a finding about the counter, not about #8952. Send me the test, the op and the two shapes before you change anything.
3. If it does not reproduce in five runs: say so with the machine's flags, and TRN-132 closes as not reproduced, with review-20's line quoted.
#8952 is not held by this (ci's word at 11:35 AM PST). Bound: your reading by Thu 12 Nov 10 AM PST.

