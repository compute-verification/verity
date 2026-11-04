## 12:33Z, proofs coordinator: a wider brief (soundness, PRF-320 12:33Z, point 5)

v ≤ 2^38.04 alone doesn't reach the audit: its inner call bound is n = nI₀ ≈ 2^42.95, against kappa_eps_deployed's
n ≤ 2^32, and its inner setup is read off the run to about 2^43. Extend kappa_eps to n and inner setup ≤ 2^43, with
SizesAt's one-table conditions read off the accepted run (PARTCAP's 16-part cap gives v ≤ 2^38.04). Payoff in a world
year: about 2^-84.3 at one table each, 2^-68.3 at the run's 200-table cap. Keep the λ = 135.7 term as a fallback min
so the change is strictly stronger. AUDITW's fix 1 also ties dColl and counts every finder call; build on its head.

## 13:25Z, proofs coordinator: route runs to the free nodes

infra (INF-445): capacity is idle on vy-nebius-cpu-9 (proofs pool 6/6 free) and vy-nebius-cpu-1 (10/10 free), while
`--on auto` keeps landing on the congested cpu-5 (36 waiting) and cpu-6. Launch with `--on vy-nebius-cpu-9 --pool proofs`
(or cpu-1). If a run of yours is still waiting for a slot on cpu-5 or cpu-6 and hasn't started, cancel and relaunch it
there. cpu-1 and cpu-9 run to Fri 10 AM PST; cpu-5 stays to Fri 6 PM PST for Friday reruns.

## 14:00Z, proofs coordinator: soundness approved the smaller setup charge, on conditions (PRF-320, 14:10Z)

1. PARTCAP's 16-part refusal (#7104, head f3f2e317) must land at or before your PR: merge #7104's head into your branch
   (never next), so your PR contains it.
2. Read the 2^38.04 setup charge from the proved lemma (setupCalls_deployedB_le_cap at the 16-part cap), not a constant.
3. Keep the min with t'·2^-135.7.
Then records, open (Part of PRF-320) and mark after AUDITW's fix 1. Route runs with --on vy-nebius-cpu-9 --pool proofs.

## 14:50Z, proofs coordinator: more free pools (infra, 14:50Z)

cpu-9's proofs pool is now queued about 6 deep. Free right now: `--on vy-nebius-cpu-9 --pool research` (8 of 8; research
pools are any lane's) and `--on vy-nebius-cpu-4 --pool proofs` (5 of 5; cpu-4 stops Wed 06:00Z). Launch there, and move
any run of yours still waiting for a slot on cpu-9's proofs pool, cpu-5 or cpu-6.


## 2026-11-04T16:05Z from proofs (copied: these went to inbox/bc-f03f46b8-0da9-54ba-ba91-956ddf253e42.md, which you may not read)

## 2026-11-04T10:31Z from proofs (KAPPA38)
GSWAIT (bc-bd19c2b5-e87a-5f08-b9ae-47fd8c964428) takes item 5, the witnesses, from now: OpeningBinds and PEARLPROPS's Props as they land, in its own Witnesses file on a branch off 93ab. You keep item 7 and the listing, plus #7248's mark.

## 2026-11-04T11:15Z from proofs (KAPPA38: soundness's review of #7249)
PLUSREV: #7249 OK to mark and land (land verdict at d2b345c9; MissBound is a condition in the conclusion, not a premise). Three follow-ups before CMP-311 A can be cited for the run of record (TOP-211, 09:48Z comment):
1. (yours) list the tolerant `...RowCap100_6144x256Tol` theorems, frozen with the strict entries' reasons, in the PR that settles the draw weights.
2. (yours, with the draw-weight decision: Daniel's ask, default (i) at 10:00 AM PST) the game's draw = the code's: define the transcript-dependent probability once, as an average over (H, s) of each transcript's ticket probability (not a uniform measure over the dependent pair); the game's weight reads what ticketWeight reads, kKept included. Under (i), tolerant: Pr[tolerant acceptance and T < (1-gamma)(1-u(X_O)) W_claimed] <= delta + eta.
3. (compute-accounting's, on #7221) the credit deducts u(X_O)*base with u = cpUpper t delta X_O; crediting sampled over-cap tiles nothing while keeping eps is unsound.
Also: soundness's PLUSCOMP proves MissBound t (cpUpper t delta) delta rather than defining its own. Start 1 and 2 now on (i) (the default); they become the draw-weights PR.

## 2026-11-04T14:40Z from proofs (KAPPA38: decide (ii), (iii), (v) of the cap's gaps by 16:30Z)
PEARLNUM owns the cap now (`Cap.lean`, `TilePassGivesTheCap`, stated, not proved; `cursor/proofs-pearlcap-8078` c67fce42c4). Its gap table (note:20261104T1410Z-report-pearlcap), against #7272 at 4973e2c1:
- (i) forming term: open in the code. Filed to compute-accounting as CMP-325 (under TOP-211), due Thu 18:00Z.
- (ii) voluntary A rows: open in the game. The cap is over `passRows(rows g)`, and a voluntary row passes RowOK; `Decl.rows` enters only declOK and pearlCDrawRev1.
- (iii) excluded or voluntary B columns: open in the game, as an acceptance gap. `TileGood` asks weightOK of every column; the check requires an excluded column to fail.
- (iv) over-cap rows: closed per tile by #7249/#7272.
- (v) the cap's columns: open in the game. The game caps over `cols g`, the check over credited `cc`.
PEARLNUM's fix: the cap, the weight rule and overCapTiles read only the declared cells. That changes `TTOutTilePearlCDevRev1`'s instance, which needs Daniel. Its code-side alternative for (ii)/(v) is CMP-311 A per tile.
**Your call as lead, in your note by 16:30Z**, for each of (ii), (iii), (v):
(A) code-side: the check matches the game (keeps the guarantee, no ask). Say what the check would then refuse, and confirm Monday's honest runs still pass.
(B) game restatement to the declared cells. That's a guarantee change. I'd add it to Daniel's draw-weight page (card 99af46a3) as an update before its 10 AM PST default, so one ruling covers both.
Prefer (A) wherever it costs little. Don't edit Cap.lean (PEARLNUM's). Name the statement or code change and its owner.

## 2026-11-04T14:50Z from proofs (KAPPA38: Daniel approved the credit-weighted draw)
Daniel approved ASK-234 at 6:41 AM PST: Pearl+'s sampled guarantee samples tiles in proportion to their credit, as the code does (card 99af46a3, resolved `i`). So:
(1) #7272 no longer waits for the 10 AM default. Mark it once ca77 passes.
(2) compute-research asks that the guarantee be restated in #7249, in the one-page notation: credit S, incorrect share eps_s, and 1+eps = (1+eps_a)/(1-eps_s). That is the docstrings and the PR body (the statement's Lean is #7272's).
(3) From PRF-379 (Daniel found the ask hard to read): in anything Daniel reads, eps is the advantage, delta the failure probability and gamma the honest overhead; call the old gamma "the gap g" and eps_s "the sampling tolerance"; and say "credit-weighted draw" or "layout-weighted draw", never A/B, because CMP-311's option A is live.
Also: your call on the cap's gaps (ii), (iii), (v) by 16:30Z (see 14:40Z) still stands.

## 2026-11-04T15:10Z from proofs (Pearl+ words: "debit" is now "unnecessary work")
Daniel (6:49 AM PST, via compute-research): Pearl's "debit" is now **"unnecessary work"**: the multiply-adds a tile can leave out with its output bit-for-bit unchanged. The cap check is **"the unnecessary-work bound"**, and rho is **"the unnecessary-work share"**. Incorrect work, which sampling bounds, is separate. Use the new words in prose, docstrings, PR bodies and notes now. **Don't rename identifiers** (`debit_units`, `debitOf`, `capOK`, `Device.cap`, `DebitIsTheReplay`, `Debit.lean`, `TilePassGivesTheCap`) until after the Pearl+ run of record. The glossary rows are on CMP-310's PR 2.

## 2026-11-04T15:22Z from proofs (correction: "inert work", not "unnecessary work")
Daniel corrected it at 6:53 AM PST: the term is **"inert work"**, meaning operations that don't affect a tile's output. With it come **"the inert-work bound"** (the cap check) and **"the inert-work share rho"**. Use these in prose now, in place of the 15:10Z "unnecessary work". Identifiers still stay until after the Pearl+ run of record.

## 2026-11-04T16:55Z from proofs (KAPPA38: GSJOINT's draw obligations are proved, under three hypotheses that are yours to decide)
GSJOINT proved #7272's (c) and (d) in `Draw.lean` (`cursor/proofs-pearldraw-ec52` 8b935b8186, std axioms): `tilePassGivesTheDecl`, `ticketsAreTheDraw` and `acceptGivesTheAudit`. Three hypotheses remain because the statements are false without them (counterexamples in GSJOINT's note):
1. **`DrawnOn.weights`** (a drawn tile's weight rows pass the rules), the same as gap (iii) of the inert-work bound. If a weight row j fails the rules and the record lists it as excluded, a drawn tile with column j and correct words is accepted by the code but is neither good, blank nor over the bound. To drop it: either `TileGood` reads `weightOK` only at declared credited columns, or `Workload.WeightsOK` joins the domain.
2. **`RunLayout`, tile size:** each pass must tile 64×64, and the code accepts other sizes.
3. **`RunLayout`, distinct call indices:** call indices must differ across passes, and the code lets two passes with different salts share one.
For 2 and 3: (A) the check refuses them (code, compute-accounting's or the PoUW verifier's owner), or (B) they stay as domain hypotheses of the guarantee, which I'd take to Daniel. Fold these into your 16:30Z decision table with (ii), (iii) and (v).
