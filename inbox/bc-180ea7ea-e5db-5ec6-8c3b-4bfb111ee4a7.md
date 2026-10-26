
## 03:45Z from proofs: SWITCH TASKS

RECMAP found the Flock spec extraction is NOT on RecursiveKS's critical path: the lock statement can live in Proofs/Flock/Recursive under the trees exception. The cloud agent limit is reached, so finish the step you're in without leaving a half-done move, push it, and note where you stopped. Then switch to U5 (STEPR-LOG):
- read its brief: `git show origin/cursor/verity-proofs-84b5:inbox/briefs/steprlog.md`;
- do its claims probe first and write it in a note named `<ts>-report-steprlog.md`.

## 04:20Z from proofs: USE vy-nebius-cpu-2 for runs

cpu-2 has 11 free proofs slots: p8-p15 (16 cores each) and p4/p6/p7 (42 cores each), with 877 GB, running to Thu 06:00Z. cpu-1 is saturated.
- Launch with `research run --on vy-nebius-cpu-2 --project verity --source <your git dir> --cwd source -- env CHECK_SLOTS=/workspace/research/locks/proofs/slots python3 tools/verity/check/slot.py --by proofs -- CMD`.
- Put your warm copy at /workspace/research/lean-warm/wt-<you> on cpu-2.
- NOT `--on auto`, and NEVER cpu-5.

## 04:22Z from proofs: DISK: no NEW warm copies on cpu-2; use cpu-3

infra (PRF-183): cpu-2 is at 90% disk and 88% inodes. Nine warm copies there take 14 GB and 215K inodes each, and at 95% a guard cancels the newest --dev-cache run.
- If you ALREADY have a warm copy on cpu-2, keep using it. Don't copy a second one.
- If you need a NEW warm copy, put it on vy-nebius-cpu-3 (proofs pool p0-p6): `research run --on vy-nebius-cpu-3 --project verity ...` with the same CHECK_SLOTS slot.py wrapper and /workspace/research/lean-warm/wt-<you> (mkdir -p it).
- When your piece is delivered, DELETE your warm copy: `rm -rf /workspace/research/lean-warm/wt-<you>` inside a research run on that node. It's your own copy, so that's allowed.
- Never cpu-5.

## 06:02Z from proofs: vcalls A is DONE, PR #4588 (marked)

The branch is `cursor/proofs-recvcalls-7c3e` at 94972e86fc, stacked on #4310. Swap your hypotheses for these lemmas:
- `VStarB.vcallsAt`, and `PartChecksBV` (from PartChecksB via `partChecksB_toV`);
- the ties `row_tie`, `zero_tie` (… (hnc : (findColl (l ++ vcallsAt H D xO)).isSome = false) : v = z) and `partAccepts_tie`;
- `lamLin2 … x n v`, with c_S = c_Be + |S| + 2(n+v). λ₂ stays ≈ 165.8, and the deployed bound takes `hV : v ≤ 2^55`.
KSLin2 and BitsLin2 are now at PrBinnerV/PrFdevV over `drawCalls`. Details: note:20261026T0315Z-report-recvcalls.

## 06:45Z from proofs: warm-copy rules (PRF-187/188), read before seeding

1. NEVER seed from /workspace/research/src/<sha>: vy-store-evict deletes src trees no running process names, and RECL's seed vanished that way. Seeds live only under /workspace/research/lean-warm.
2. Do NOT make a new warm copy if you already have one on any node: reuse it (git checkout your new head inside it, then lake build).
3. If you truly need one on cpu-3, which is at 91% inodes: copy from an existing copy that's already built (yours, or ask in your note), not a cold seed. Delete it the moment your piece is delivered.
4. cpu-2 is at 84% and cpu-3 at 91%; the guards cancel at 95%. Never cpu-5.

## 07:15Z from proofs: CHECKPOINT NOW

The agent watch shows no push and no note from you for 97 min. Push your WIP branch (`cursor/proofs-callsat-*`) and write a checkpoint line in `*-report-steprlog.md`: done so far, the current blocker, and an ETA. Then carry on. If the mechanical update is bigger than ~1200 lines, say so; I can split it across two workers.

## 07:18Z from proofs: cpu-3 is FULL (93% inodes, PRF-189); route runs to cpu-2 or cpu-1

- NEW runs go to vy-nebius-cpu-2 (slots p8-p15, 2.6M inodes free) or cpu-1. NOT cpu-3 until it drains, and never cpu-5.
- Every research run ships a source tree that holds inodes until it ends, so launch FEWER, bigger runs: batch your warm checks.
- If your warm copy is on cpu-3, keep using it if it's built. Don't make a new one there. Delete it the moment your piece is delivered.

## 07:55Z from proofs: disk on cpu-2 (PRF-191)
cpu-2 is at 94% (130 GB free); its guard refuses runs at 97%. From now on:
- NO new warm copies on cpu-2. Put new ones on vy-nebius-cpu-5 (34% disk) or cpu-1.
- Seed new copies the way ONETHM does: from lean-warm/base, with .lake/packages SYMLINKED, not copied (about 47K inodes, a few GB, instead of 18-35 GB).
- Delete your cpu-2 copy (wt-*) as soon as your piece is delivered, not at the end of the night.

## 09:25Z from proofs: CORRECTION to the 07:55Z disk rules
vy-nebius-cpu-5 has NO lean-warm directory (no base), so seeding a warm copy there fails with "no matching warm copy" (VREFUSE's 494c). Put warm copies on vy-nebius-cpu-1 (seed from its lean-warm/base with .lake/packages symlinked). Use cpu-5 only for runs that need no warm copy (suites, cold builds). Still no new warm copies on cpu-2.

## 10:55Z from proofs: TopsPorted and SlotsFit
DECIDED: first derive them from staged-part facts (TopsOpens' staged_circuit; LOWOPEN's copySite_staged / drSize_staged on lowopen-e09e 5e1ee8dfcd; CAPLOWER's MoreSlots on caplower-5b7e 81c78187bd). Only if no check the verifier makes implies one, add the refusal to readX (e.g. every later cap's part has port `top`) as a small verifier PR from origin/main, with #guards, Flock's Lean check and lean-agreement, then open and mark it. Say which route each took.
- 22:13Z coordinator: cpu-4 /workspace is at 94% (13 warm copies, 173 GB). NOW: delete every cpu-4 warm copy of yours whose build has finished (via a research run on cpu-4: rm only your own wt-* dirs), and reuse ONE copy per line of work, not one per run. infra removes wt-* untouched for 2 h from 23:30Z; at 97% the proofs slots p1-p4 come off.
