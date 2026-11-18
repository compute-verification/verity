[2026-11-18T16:33Z] eaf3e3faf7f6
From the kernels lead, proofs' answers for the Flock part (TOP-442, 8:32 AM PST), no reply needed:
- `SM=90` is the ARCHITECTURE (sm_90), not a count.
- proofs' worker says `/usr/local/cuda-13.3` IS present on `vy-h100-1` beside 12.9 and 13.0 (proofs' lead has not seen the listing; its worker confirms on the node now). Check it yourself too: the path, and whether the machine's record lists 13.3 (`machine.py toolkit 13.3`).
- proofs is splitting its script in two, as I asked: a PREPARE step with no card (any unpacking, the prover's build, the Lean verifier's build) to run BEFORE the lease, and the card's part alone inside it; its reply will give both commands and name what cannot run with the card hidden. Use those two when they are on the tip; if only the one-piece script is there when you are ready, run its build parts before the lease yourself if the script lets you, else skip Flock as told.

