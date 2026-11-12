[2026-11-12T16:21Z] 0cfb31af1330
BACKWARD, TRN-140, 8:21 AM PST: the fused backward's capture is in the store, art:6a0d7e798b0d04914013112c84cfde313b0e4ba84dc8fd0ac47491c75b5667d1 (run r20261112-161612-10b7, cc 12.0, torch 2.14.0+cu130, one GPU for 16 s; the kernel Kineto saw is flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel at bf16). No verdict yet; the CPU counts are next. Checkpoint: note:training-20261112T1620Z-checkpoint-trn140-backward-capture-in-store.

