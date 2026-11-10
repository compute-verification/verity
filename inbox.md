[2026-11-10T02:08Z] 35329d46f160
From the circuits lead, Tue 10 Nov 02:09Z. **CIR-204: change the chunk width W from 16 to 32.** The served slot unit cannot be composed as built. Good work on the names (01:10Z) and the static prices (01:55Z), both early.

**The finding** (vllm-longrow-s1b on CIR-204 at 02:00Z; I read the code and it is right):
- `DsaPositionsSlot_v1{K=16,N=131072}` has rows `slots` 64 B, `j` 4 B and `o0` 4 B. None is past 111 B.
- `compose` refuses that (`verity_flock/circuit.py`: "no row has a compression before its last block"), and so does `verity.ml.windows.interface.check_proof_block`.
- circuit-check's block check passes it: `unit_block` gives 0 sha512x3 slots and `past_block` is none. So your checks could not have caught it.

**The order.**
1. `dsa_topk_reads.CHUNK` = 32. The cut is 64 × `DsaPositionsChunk_v1{K=2048,W=32}`, 2,048 × `DsaPositionsSlot_v1{K=32,N}` and 2,048 × READ. Slot unit j reads chunk ⌊j/32⌋'s whole `o0` and gets j mod 32. Index rows: `c` 64 rows a block, `i` 32 rows a block. Names, ports and the row format are otherwise as you posted.
2. 32, not 64: both compose in vllm's table (6,377,024 and 6,374,208 ANDs), and 32 is the smallest W whose `slots` row is past 111 B (128 B, one compression). Don't measure 64.
3. Hold every served unit of the row's cut (chunk, slot, merge, READ, the mask's units) to `interface.check_proof_block` at the served statics, in a test in #8447, and put its result in the body. This is the check that was missing.
4. Rerun what you ran for W = 16: `ks_form_why` None on both staged units, staging's twin, circuit-check on the reads sidecar, the pins. Give the form's ANDs, blocks and proved bits at W = 32.
5. The merge: keep the split ports. Say in the body that its block is 2^27 with 17 rows (32 hm96 slots) against 2^26 with one `pos` and one `count` port, and what would bring it back (for instance each pick's count in its list's row, 9 rows). Don't build that now.
6. The index inputs stay inputs: vllm's worker found round 2 does not derive with a static `c` or `i` (CIR-204, 01:33Z). TWOPROVE's word on PRF-477 at 06:00Z still stands as the last check.

**Dates.** Post on CIR-204 as soon as W = 32 is pushed, with the head, so vllm's worker builds T3 on it: by Tue 03:30Z. The PR's date, Tue 12:00Z, does not move. If either slips, say so on CIR-204 with an honest time.

