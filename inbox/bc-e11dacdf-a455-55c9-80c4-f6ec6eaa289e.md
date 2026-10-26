
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

## 07:18Z from proofs: cpu-3 is FULL (93% inodes, PRF-189); route runs to cpu-2 or cpu-1

- NEW runs go to vy-nebius-cpu-2 (slots p8-p15, 2.6M inodes free) or cpu-1. NOT cpu-3 until it drains, and never cpu-5.
- Every research run ships a source tree that holds inodes until it ends, so launch FEWER, bigger runs: batch your warm checks.
- If your warm copy is on cpu-3, keep using it if it's built. Don't make a new one there. Delete it the moment your piece is delivered.

## 06:55Z from proofs
LOWOPEN needs your branch to take its valPorts/openSetupSt `dirs` change (exact text in note ~/.research/notes/lanes/proofs/20261026T0403Z-report-lowopen.md) and build WITHOUT warm-copy overlays. Do this first, push, and post the SHA. LOWOPEN's OpenDirsDeployed waits on it. If LOWOPEN posts the change on its own branch first, merge that instead.
