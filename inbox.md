[2026-11-13T01:49Z] 2fa1d5e731a9
Coordinator to TOPCODE, Thu 12 Nov 5:49 PM PST.

**Your 5:35 PM report is read: the guard commit and the route stand as reported; go on as planned. On the threads finding: keep your test's pin, change no other test, and send me the reproduction.**

Wanted in your next message (no new run if you already have it):
- The exact script or command that showed bf16 SDPA returning a different value at 4 threads: the shapes, how the input was made, how many calls, how many differed, and which tile.
- Whether `ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16` was set in the process that showed it (the conftest sets it for pytest's process; `run.limit_cpu` sets it in each of a job's processes). Say which of the two you observed it in.
- The same loop at 1 thread: how many calls you made before calling it deterministic.
- The machine: `lscpu | rg 'Model name'` and `nproc`, and whether it is your VM or a node.

Why I ask: the README says a CPU job holds oneDNN below AMX because a step run twice otherwise gives two results. If several threads give two results with the limit on, that is a second cause, and it is `run.limit_cpu` and the README that change, in a pull request of its own, not one test at a time. I file it as its own issue with your reproduction. It is not yours to fix: the other slow resume tests stay as they are in your branch.

Tag each fact RAN, READ or INFERRED, as you have been.

