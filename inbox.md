[2026-11-11T11:01Z] e558a65e3375
library to training, the AMX probe on a node, Wed 11 Nov 3:02 AM PST. RUN, not read.

Run r20261111-105618-9c08, node vy-nebius-cpu-11 (hostname computeinstance-e05m8fsfqd6p24xpft), at d7a7de49530d, torch 2.14.0+cpu. Load average 61 to 81 through the run on the 45 CPUs the job may use, so it was busy.

Four threads, 3,000 calls of Newton-Schulz on one input each:
- no cap: 0 of 3,000 differ from the first call, 0 non-finite.
- ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16: 0 of 3,000, 0 non-finite.
Again under 45 busy processes of the job's own: 0 of 3,000 without the cap, 0 with it, 0 on one thread without it.

WHY: it cannot happen there. The node's CPU is an AMD EPYC-Genoa: avx512_bf16, and no CPU with amx_tile (the job printed both from /proc/cpuinfo). oneDNN never takes an AMX kernel on it, so capped and uncapped are the same code. Your VM is an Intel Xeon with amx_tile, amx_bf16 and amx_int8: there the same probe gave 7 of 300 (3 non-finite) uncapped and 0 of 300 capped ten minutes earlier, and 64 of 1,500 against 0 in the A/B of the note.

So for infra: it is a fault of machines whose CPU has AMX (the agents' VMs here), not of this node, and a node run of the suites or of the table was never exposed on it. I checked one node only. `grep -c amx_tile /proc/cpuinfo` on a machine says whether it is exposed: 0 on vy-nebius-cpu-11. I did not look at the other nodes or at the GPU pods' CPUs.

One more thing the run shows: the first call's output has the same sha256 on the node and on your VM under the cap (58d68cfaad03), from inputs that differ (the step that makes the input rounds differently under AMX, the same way every run, as the note says).

Ruling 1, done but for the suites: d5a0ad794ec is cherry-picked onto cursor/training-muon-shampoo-whole-6e3e as 15cf966644d, and "draws" is mended (2f45348a363: test_entries.py's lora docstring, lora.py's "drawn", and the conftest's comment says "limit", not the bare "cap"). #8901's head is 2f45348a363b30ab073d6c77f4963ec35c4a1f48, pushed. Its four touched test files pass here (55). Its suites run on a node now: r20261111-110015-e74d (--changed 3bd1ff4cf11d). I send its result and update the body when it ends.

Log: /opt/cursor/artifacts/library-amx-node-probe-r20261111-105618-9c08.log

