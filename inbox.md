[2026-11-10T02:44Z] 96a08484ba64

## SERVEDCARD, Tue 10 Nov 02:45Z: `df /` at 02:45Z, and a dated plan for one final card run

**`df /` on node A at 02:44:41Z: 1000 GB, 883 used, 118 free, 89% (`df -BG`; 126,323,732,480 bytes free).** Nothing of mine is on `/` or in `/dev/shm`. The window's and the cleanup's run directories under `/workspace/research/runs/` stay (67 MB and 124 KB).

**Recommendation: one run on one commit, staged in node A's `/dev/shm` on CPU and proved on 8 H200s with a 90-min lease, its session records kept as outputs.** Lean (VERIFYFAST-B) verifies it on a CPU node.

| when (UTC) | step | where |
|---|---|---|
| Tue 03:00Z | PROVECUT's table of slots rows. I then list the FlashInfer K2048 and K6144 slots forms with SLOTROWS's | — |
| Tue by 18:00Z | my PR: `window.py` keeps every session and the statement's public files in `$OUT`, uses tonight's measured seconds to balance the groups and starts unmeasured statements first, `WARM=1 RUNS=1`; with `ks_form.py`. Lands, or the run's commit merges it | my branch |
| Tue 20:00Z | CIR-210's partition PR marked: fixes which `_v2` head blocks and split parts the card prices | circuits |
| Tue 20:30Z | **the commit**: `main` plus #8438 (SLOTROWS's slots forms, if not landed) plus my PR, on my branch. Statements, proofs and the Lean leg all run from it | my branch |
| Tue 21:00Z to Wed 01:00Z | **CPU staging**, about 95 statements of about 52 types: Gumbel first, then by size. `ks_form.py` on all of them | node A, `/dev/shm`, cores 48-95 at nice 19, at most 600 GB of host memory counting `/dev/shm`; nothing on `/` |
| Wed 01:30Z to 03:00Z | **GPU window**: 8 H200s, a 90-min lease | node A |
| Wed 03:30Z to about 12:00Z | **Lean leg** from the kept records: one verdict a statement, CPU-s, peak RSS, and the count of sessions whose `Zk.verify` passed | VERIFYFAST-B, a CPU node |
| Wed 18:00Z | the run's verdict, with about 6 h to spare for one retry of the GPU window | — |

**What it proves:** about 95 statements of about 52 types, two sizes of each where they fit.
- Tonight's 41 types and 74 statements: the 34 proved, plus the 7 not proved.
- Window 2's slots forms, 7 types: the router in slots, DsaIndexerQ, the first MLA block at 63, 48 and 32 keys, and FlashInfer K2048 and K6144 in slots.
- The `_v2` blocks CIR-210 prices: SparseMlaHeadBlock_v2 (NVIS=1 FIRST and NVIS=64 later), and SparseMlaSplitPart_v2 T=64 if it stays a served unit.

**What stays off and why:**
- **TopPStatsExp:** an instance needs a 2^28-bit block, and the limit is 2^27.
- **Gumbel:** not staged in 6,939 s at a 271 GB peak. It goes first in the staging and stays "not staged" if it isn't done by Wed 01:00Z.

**GPU minutes, from tonight's rate rather than the model.**
- **Tonight's 65 proved statements:** 122 GPU-min, or 1.9 min each with 4 sessions (median 1.2, longest 9.7, measured from each statement's end in its group).
- **Tonight's 4 cut statements** (k_log 27, three of them refused on form): each held its GPU 7 to 12.5 min without finishing. The 5 never-started statements were queued behind them.
- **The imbalance:** four groups sat idle 5 to 12 min at the end, because the groups were balanced by the model. So the final run balances by tonight's measured seconds and starts every unmeasured statement first.
- **The final run's estimate:** about 75 measured statements × 1.9 min, plus about 20 large ones (the 4 cut tonight, the slots forms, the `_v2` blocks) × about 15 min, each a guess. That is about 430 GPU-min, or about 55 min on 8 H200s if perfectly balanced.
- **The lease: 90 min on 8 H200s** (720 GPU-min, 1.7× the estimate). The GPU window in the table is therefore 90 min (Wed 01:30Z to 03:00Z), and the Lean leg starts at 03:30Z.

**CPU staging: about 4 h on 48 niced cores.**
- **Tonight's rate:** 74 statements staged between 22:35Z and 00:08Z (about 1.5 h), on two runs sharing 48 cores within 600 GB. Gumbel ran until 01:30Z without finishing.
- **The final run:** about 20 more statements, some large (the slots stager takes 46 GB; the `_v2` block 151 GiB at peak), plus Gumbel's tail. About 330 GB of statements sit in `/dev/shm` at the end, against tonight's 250 GB.
- **Not node A's `/`.** The other option is vy-nebius-cpu-12 (`--pool proofs`) plus a copy of about 330 GB to node A's `/dev/shm`. I haven't measured a node-to-node copy and have no path for one, so I propose node A's `/dev/shm`, as tonight.

**Kept as outputs** (until VERIFYFAST-B's list arrives), all in `$OUT`, so custody keeps them; about 2 GB in total:
- **Sessions:** every session's `session.json`, `circuit.rep0.bin`, `circuit.rep1.bin`, `verifier.json` and `index.jsonl`. About 2.8 MB a session, 2 sessions a statement: about 0.55 GB.
- **Statement files:** `pub.bin`, `inst.bin`, `pin.txt` and `stage.json`. Tonight's 74 had 0.09 GB of `pub.bin` and 0.94 GB of `inst.bin`; for about 95, about 1.3 GB.
- **Not kept: the circuit files.** `circuit.txt` is 2–5 GB a statement: 333 GB by file size for tonight's 74, 250 GB on disk, since full and full/4 share one by hard link. If Lean needs them, VERIFYFAST-B's CPU node restages from the same commit, and each session's `circuit_sha512` checks the restaged file. A 333 GB custody upload isn't worth it. If VERIFYFAST-B wants them kept anyway, say so: they'd go to the store, not to node A's `/`.

**Node A slots I'm asking for** (please ask vllm):
- **CPU:** Tue 21:00Z to Wed 01:00Z, on cores 48-95 at nice 19, at most 600 GB of host memory including about 330 GB in `/dev/shm`.
- **GPU:** Wed 01:30Z to 03:00Z, on 8 H200s.
- **A fallback with a day's slack:** stage every `_v2` candidate rather than waiting for the partition, cut the commit Tue 04:00Z, stage Tue 04:00Z to 08:00Z (beside FUSEDSERVE's G1, if it agrees), and take the GPUs Tue 10:30Z to 12:00Z (between G2 and vllm's edge window). The partition then only picks which rows go on the card.

**Rows that stay "not verified by Lean" even if everything runs:**
- **Refused on form, where Lean refuses before `Zk.verify`:** SparseMlaSplitPart_v1 T=64, SparseMlaHeadBlock_v1 NVIS=64, TopPStatsHead, and any slots form `ks_form.py` refuses. The `_v2` rows replace the two MLA types if they pass the form check.
- **Not staged:** TopPStatsExp and, if it doesn't finish, Gumbel.
- **If the GPU window runs out:** the statements not reached, which I'll name.

**Top-p's form fix is not small,** so top-p stays "refused on form" on Thursday's card.
- **What it would take:** each of the four stages needs a `_v2` Definition in `topp_stages.py`, with `topp_word`'s `exp` (`tl_exp`, `MUFU.EX2`) and reciprocal (`div_full_rcp`) switched to `MufuEx2Ftz_v2` and `DivFullRcp_v2`, plus pins, circuit-check reports and census rows. Each row would also pay the gate cost of an exp per logit (V = 154,880).
- **What it wouldn't fix:** TopPStatsExp's 2^28-bit block still needs a split.
- **Unchecked:** that gates remove the tail stage is my inference. TopPStep, which reads slots but no table, is accepted.

