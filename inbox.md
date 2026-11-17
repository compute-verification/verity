[2026-11-17T20:01Z] 62566280a99d
From the circuits lead: #10862 (the Boolean maximum and minimum commit their result; 13 Boolean Definitions' AND counts and digests moved, `Fp8GroupScale_v2` among them) is ON MAIN since 19:27Z, and so is #10845 (the cut norm part is now `RMSNormFusedCudaAdd_v2{N,C,AT,R}`). Merge the current `main` into #10446 before you regenerate its FP8 pins, so you regenerate once.

