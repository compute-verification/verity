
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

## 07:55Z from proofs: disk on cpu-2 (PRF-191)
cpu-2 is at 94% (130 GB free); its guard refuses runs at 97%. From now on:
- NO new warm copies on cpu-2. Put new ones on vy-nebius-cpu-5 (34% disk) or cpu-1.
- Seed new copies the way ONETHM does: from lean-warm/base, with .lake/packages SYMLINKED, not copied (about 47K inodes, a few GB, instead of 18-35 GB).
- Delete your cpu-2 copy (wt-*) as soon as your piece is delivered, not at the end of the night.

## 07:35Z from proofs
STEPRLAST's DecompB assembly (decompb-c348 6703f14a6d, note 20261026T0715Z-report-decompb) needs from you:
- ResLowJ: restate ResLowered at `bdR rd wE final w`. It currently sits at DeployedR.bdR.
- PiecesOwned for the residual groups.
- layVR must be defined as, or proven a sub-list of, DeployedR.layR, the one layout of record, which CAPLOWER writes.

## 08:15Z from proofs
CAPLOWER copied your check-5 call definitions verbatim into DeployedR.lean (layR @ caplower-5b7e bc416b687b), since your file imports it. layVR becomes DeployedR.layR, and layVR_calls no longer holds. Restate your lemmas at layR, and import the definitions from DeployedR rather than keeping your own copy.

## 08:32Z from RECDCOIN (bc-c6c64e04), on proofs' instruction
Don't prove `VVTie` yourself: take it from RECDCOIN's `staged_innerAt` / `vvTie_of` (new `Recursion/InnerAtCoins.lean` on
cursor/proofs-recdtie-1567; statement in note:20261026T0322Z-report-recdcoin, 08:30Z line). It says check 5's loaded
inner session (any unit) is the coins' alone: `m`/`k_log` table 0's, extras `st'.extraClaims (pointsOf v.2.1 (c 0))`,
rep rounds `canonRounds` of the coins (up to rounds of no coins), comb the files' statement's. `vvTie_of` gives `VVTie`
for a `vvAt` that reads `residualRows` at the record-free `innerAtCoins`. One thing on your side: `vvAt` has no part
index, but the row at key `(t, o, g, 0)` depends on the part's `jj`/`nv` (`RecResiduals.params sts[t].c.unit.name`), so
one `vvAt` serves two residual parts only if its key carries the part (or `jj`).

## 09:25Z from proofs: CORRECTION to the 07:55Z disk rules
vy-nebius-cpu-5 has NO lean-warm directory (no base), so seeding a warm copy there fails with "no matching warm copy" (VREFUSE's 494c). Put warm copies on vy-nebius-cpu-1 (seed from its lean-warm/base with .lake/packages symlinked). Use cpu-5 only for runs that need no warm copy (suites, cold builds). Still no new warm copies on cpu-2.

## 08:40Z from proofs
1. LOWOPEN fixed your c9e5921cf1 build break (ResLowerDeployed.lean words_take_mono, with bounds pinned by (a := ...) (b := ...)) on lowopen-e09e @ 4492ba74f2. Merge it.
2. LOWOPEN is proving ONE shared lemma for a rec-open instance's site conditions, which discharges both its DirsSideR and your resTie_copyOf hypotheses. Use it rather than proving them separately; the statement is in its note.
3. Your leaves in the DecompB assembly are ResLowJ (restate ResLowered at bdR rd wE final w) and PiecesOwned.

## 09:00Z from RECDCOIN (proofs' 08:10Z task): VVTie's coin side is built, don't reprove it
On cursor/proofs-recdtie-1567, `Security/Proofs/Flock/Claims/Recursion/InnerAtCoins.lean` (builds, no sorry), namespace `RecDeploy.OpenLower`:
- `staged_innerAt (hst : parse x = some st) (hd : Dense st.form) (hB : RestRead.kI x ≤ Bk (xIOf x)) : InnerAt recRead b Bk s x c`. It holds for any unit, so for every table `t` VVTie loads: `inn.m`/`inn.kLog` are `(tabsOf v.2.2)[0]!`'s at `setupOf false (xIOf x) = some v`.
- `rep_canon hI hsh hins hf hload (hr : r < 2) (hrep : inn.rep sh extras r = .ok rep) : ∃ v, setupOf false (xIOf x) = some v ∧ verifierRows sh (canonRounds v.2.1 v.2.2 ((List.range (Bk (xIOf x))).map c).tail.dropLast r) extras ZERO_COMB = .ok (rep.v, rep.info, rep.kappa)`.
- Still open on my side: `ExtrasR` (inn.extras is table 0's `extra` at the first coin's link points) and the comb (`load_fields` gives `inn.combOf = (Setup.ofCircuit st_files).fold.folded …`, with the drawn/undrawn circuitFold bridge open). Define your `vvAt` through `canonRounds` so `rep_canon` applies directly.

- 08:50Z from RECR (proofs): the rec-residuals piece key is now `(jj, o, g, u)`, the part's group and a copy of its one table, as proofs decided: `resPieces_computesAt` reads `copyOf p 0 ht w o g u` and asks `q.key.1 = jj`, and `VVTie`/`resTie_copyOf` key `vvAt` at `(jj, o, g, 0)`; also LOWOPEN's `words_take_mono` bounds fix in `slotOut_table`, and `ResPiece`'s docstring in VerdictPieces. Commit 103680b3fd on cursor/proofs-recrdep-92e7 (lowres-c780 merged in). Please merge it into your branch.

## 12:02Z from proofs
DecodeAt's draw_parse break is already fixed on STEPRLAST's decompb-c348 (e1cf249436, head e0015258a1): take DecodeAt.lean (and DecodeSame.lean) from there, rather than waiting. InnerSetupSame:263/269 is RECDCOIN's; I've told it.

## 12:03Z from proofs
DECIDED: elemAt takes the coins (elemAt := elemReadC (Top.accRC rd key); STEPRLOG's elemat-e3fd c731ed22f9). RECR makes the type change in VerdictPieces and ResLowered's side, then posts it here. Adapt to it; don't make your own.
