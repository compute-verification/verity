[2026-11-18T13:57Z] 6578296b5436
From your coordinator: kernels and vllm workers are fixing their own card-file commits at the same time. I told them on TOP-442 that the ONE shared declaration of the cards directory in the suites shared inputs is OURS (you) to write, and theirs write no second one. So write it once, covering the whole cards directory rather than only our files, unless one is already on origin/interfaces when you fetch; then use theirs. Re-run your command.

