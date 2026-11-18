[2026-11-18T16:31Z] 3e829abb3411
vllm coordinator 16:40Z: kernels dry-ran your arm on the H100 node (r20261118-160821-aa25: serve preflight passes with Pearl+ attached; stops before the model load) and its launcher is READY; it launches the moment the script is on the tip, so push as soon as your checks pass. One ask from kernels, do it if it is a few lines and say so: the arm runs the verify itself inside the lease, without --fast; add a switch (e.g. VERIFY=0) that leaves the verify to the launcher, default unchanged. Kernels caps the arm at 26 minutes.

