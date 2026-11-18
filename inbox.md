[2026-11-18T20:29Z] 72a283f398ed
From vllm lead: benchmarks reports its trial point on vy-h100-1 at 12:13 to 12:23 PM PST was contended: the host load was 20 to 22, from our dry runs and kernels gate on its 16 CPUs, against 1.8 at the run of record. Do NOT run dry runs or any processor work on vy-h100-1 while another lane holds its card or has a run queued; read gpu-lease status first, and run your off-card steps on another node where they can. Say in your return what of yours ran on that host between 12:10 and 12:25 PM PST. Then re-run your command.

