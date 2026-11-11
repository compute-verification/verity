[2026-11-11T10:43Z] ecae993e5814
library (TRN-113), Wed 2:44 AM PST. One finding you should know, and my head.

FINDING, run by me on your VM: a bf16 product on a CPU with AMX is wrong now and then when the process is preempted. Note training-20261111T1039Z-report-amx-bf16-products-wrong-under-preemption.
- Muon's Newton-Schulz (routes/_optimizers.py) on one saved input, 3,000 calls: 53 outputs differ from the first on four threads (27 non-finite), 2 on one thread. The op is a bf16 matmul under oneDNN's AMX kernel; a product that differs is wrong, not a rounding away.
- Under four busy processes of mine, 1,500 calls: 64 differ with no switch; 0 with ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16; 0 with torch.backends.mkldnn.enabled = False.
- tests/test_entries.py under that load, four runs each: 23 passed four times with the cap; 6, 4, 1, 1 failed without it. The suite's one-thread fixture was on in both: threads were never the cause.
What it means: the Muon entries' tests at #8840 and #8901 can fail on a loaded AMX machine (I never saw it there: three entries, three steps); any CPU job that trains under Muon can write a NaN. Not touched: the table, the grade's trace, a capture (fake tensors), PORT's check (counts). A GPU job does not pass oneDNN.
What I did: one line in tests/conftest.py sets the cap before the first product (commit d5a0ad794ec, first on my branch). Yours to rule: whether #8840 or #8901 should carry that line before landing, and whether run.py should set it beside TORCHDYNAMO_DISABLE so a CPU job's values are right too. I put it in the tests only.

HEAD: cursor/training-averaged-weights-6e3e at d7a7de49530d, three commits on #8901's 3bd1ff4cf11d: the cap; ema_horizon on route ema (LATER, VERSION stays 1); eight entries (four of the seven structures, each as its weights and as their average), with a test that refuses an average's entry with no entry of the weights beside it. The thirteen earlier entries' digests are as they were. On nodes now: the table r20261111-104246-29a2 and the suites r20261111-104314-5f98. No PR yet: it waits on your answers (a: PORT's check, b: where the search's configs sit, c: names, and the three pilot starts).

