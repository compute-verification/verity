[2026-11-20T15:12Z] 9d0d74e080f5
From vllm lead, 7:14 AM PST Fri. vy-h100-1 IS UP since 7:10 AM PST and answers (infra, Linear TOP-445, 7:11 AM PST): its disk is whole (211 runs; df 72G, 62 percent used; the H100 at 0 MiB); it is ours at 8:00 AM PST, kernels at 9:00; it stays up until each says done or noon PST. Check free disk before launch (the window record wrote 6.13 GB on a processor node; the card run output and the record must fit beside 62 percent used: say the free bytes in your launch report), and say on your launch report which directory the run writes to.

