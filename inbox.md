[2026-11-18T11:43Z] c35ba0c83bb2
From the kernels lead, for the smoke run, no reply needed:
- The smoke script DID change on the tip after all: KROWS pushed `5d38f0548ac` (one line out of `benchmarks/kernels/smoke_h100.sh` near 328: the match of the card by its printed name is gone; the card-file `sm_90a` check right after it still refuses a wrong card before any step). That diff is known to me and approved: do not stop for it. Read it, and if the repeated dry run at the tip passes with the card check still refusing nothing on this H100, proceed to launch.
- If the dry run at the tip does NOT pass because of that change, stop and tell me; do not fall back to an older commit by yourself.

