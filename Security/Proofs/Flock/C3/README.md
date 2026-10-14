# Candidate 3's recursive guarantee: the gap list

`Flock.SecurityProofs.RecursiveAuditC3 : Flock.Guarantees.RecursiveAuditC3` (`Recursive/AuditC3.lean`,
`Recursive/AuditC3Proof.lean`) is PublicAudit's proof at candidate 3's table (`holoInnerAt`, R from the registration's
`stacked` entry, V*'s statements over the verifier's own list). It has no `sorry`: every stub in the two tables below
is proved. It takes VBridge (`_hbr`) and V*'s parse (`_hob`) as PublicAudit does; pieces (b)–(h) discharge
them. V*'s statements are indexed by the verifier's own list (`vstarCount ish`), which at candidate 3's inner shape is
circuits' gap 2's (#1847, merged here at `100c17932`): eleven statements in the writer's order, 0–3 `rec-open`, 4–5
`rec-residuals`, 6 `rec-sparse`, 7–9 R's levels `rec-open-stacked`, 10 `rec-opening` at META's `stacked` shape.

What stays open is the hypotheses the discharge takes, `hbr_honest_reg`'s (`C3/Compose/Discharge.lean`, `_hbr` itself)
and `srcOK_holo_rs`'s (`C3/Compose/AtHolo.lean`, `SrcOK` at the checked family), and what the event's bridge and the
bound still need: Open now, next.

## Open now (round 21, against Wed 14 Oct EOD PT)

Owners are round 21's workers (H, C, R, K, T, S, L, E), **rec-lean**, **W**, the next free worker, and S's successor. A
size is in lines of proof: "+N" is what a branch has built, a range is an estimate of what is left. spec-2
(`cursor/c3-spec-2-35f2`), 14 Oct 05:00Z, has round 21's T, K, R, L, E, U, S and C heads, #1851's head `cb1d25f98`,
K's round 5 (`f5be751ac`), rec-lean's round 21 and S's round 22 (`b74cd6ec7`); rec-lean's round-21 pieces branch
(`cursor/c3-rl21-pieces-9b2a`) adds R's rounds 22–24 (`90110fa71`), C's rounds 22–23 (`f0d358227`, with K's round 6
`89750b6fd`), L's round 22 (`927241612`) and W's round 2 (`e83cd54c9`).

Closed since round 20, all on spec-2: the bridge (`acceptsC3_of_verify` and its checks, H; r20261013-194011-b18d),
`SparseRangeOk` from check 5 (`sparseRangeOk_of_check`, H, in the bridge), the top finder (`TopColl`,
`prob_regOrTop_le`, K), the top-root collision (`topRoot_collision`, T), R's opening's round reads (`RoundReads`,
`WRoundReads`, `checkMsgNames`, S; `PortsNamed` as `checkMsgNames`' refusal, S round 22), `OwnIs` (S's `OwnIn`, proved
at the verifier's `own` by `ownIn_checked`), the holo streams' list (`holoStreams_eq`, `holoStreams_sparse_eq`), the rep
forms' alignment (`holoSizes_reps`, at `repForms m kLog claims = zcForms m ++ (lcForms kLog ++ (opForms claims ++
ligForms (schedAt m)))`, from `repRounds_sizes`; `heldBefore_rep_forms`), `drawnOf_holoMap`'s `hS` and `hr`
(`drawnOf_holoMap_forms`), and the sparse rounds' forms (`holoForms_sparse_none`, `holoSelWords`' last conjunct) with
its `hA` (`repFormsAt_holoAt`, at `hcl` below) (rec-lean, round 21). On the pieces branch: `hnd` as `checkAcc`'s
`checkNames` refusal (`names_of_checkAcc`), `readsOwn_of_hown` at the drawn label (`AtLabel`; `LabelsSelf` gone, its
step now `RowsHeld`'s label conjunct), `StackedCoins` and `stackedStreams_of_covers` (R, rounds 22–24); the level-0
salt binder (`salt`, `saltFinder`), the verifier's value source (`valueSrcC3`, `Recursive/ValueSrc.lean`) and
`valueFinder` in the guarantee, and `_hbr` as `VBridgeCOff` off `ValueColl` (C, rounds 22–23); `AccHeldO` from
`RecCoins.opening`'s `accHeld` refusal (`accHeldO_of_opening`, L round 22).

| piece | names | where | size | what it gives | owner (due) |
|---|---|---|---|---|---|
| the reps' round count | `coins_of_checkCommitRounds`, `heldRounds_of_check`, `zp_of_rounds`, `repIdx_of_size`, `repIdx_of_check`, `repIdx_atRep`, `streams_of_custody`, `holoSel_first`, `repsDrawn_of_counts`, `repsDrawn_holoMap`, `repsDrawn_holoMapAt`; `hchk`'s drop-ins `checkCommitRounds_of_checkCovers`, `hchk_of_held`, `hchk_of_acceptsC3`; `covers_checked`, `covers_of_acceptsC3`, `repsDrawn_checked`; `hk` is `UnitFrames.whole` (S15/R2), proved in `unitFrames_holo`; wired: `holoMapOne`, `custody_holoMapOne`, `shape_of_statements`, `repHead_zp`, `holoStreams_of_list`, `stageCoins_checked` | `C3/Reps/DrawnRounds.lean`, `C3/Compose/CoinsChecked.lean`; `cursor/c3-h21-drawn-95d4` | +374 (round 22: +139 −49), round 23: +90 −1 (with the fold and `_htbl`, +110 −13 over six files), `whole` +10, round 24: +222 −37 over seven files (`hS` +45) | `RepsDrawn` at `holoMap`'s one-round draws (`holoSelOne`, where `CoinsEncoded` holds, `coinsEncoded_holoMap`; it does not at all of `holoMap`'s, a sparse phase's round of `n > 1` coins) in `AtRep` form (no `DrawnOf` `∀ Lb`), at `holoCTs` and `m := holoMap` (`repsDrawn_holoMapAt`), with no shape facts: the check pins the frames' `z_partial` round to the schedule's. At the checked sessions and the guarantee's hold (`repsDrawn_checked`): `hchk` from `ChecksC3`'s covers conjunct (`covers_checked`; the verifier's run gives it through `AcceptsC3`, `covers_of_acceptsC3`), `hn` the guarantee's `_htbl` (`RecursiveAuditC3At`'s side condition beside `_hIC`; `tbl` stays a binder, discharged where it is instantiated), `hleaf` from `hLeafC3_checked`, `hrs` from `holoStreams_eq`, `hr` from the reps' forms. Wired (round 24): `stageCoins_checked` is `hcoins` of `srcOK_holo_rs` and `srcOK_repsOfVals_pR` at the checked sessions (`zs` with `hzs`, `rfl`) and any `rep` whose rounds are the record's (`hrd`, `fun _ _ _ _ _ => rfl` at `RepMaps.withReads`), `stageCoins_holo` at `holoMapOne` (`holoMap` drawing at `holoSelOne`; custody at `holoMap` restricts to it, `custody_holoMapOne`): `hA` from `repFormsAt_holoAt` at `ClassOk` and the event's `checkInnerStmt` (`AcceptsC3`'s fifth conjunct), `hcl` and the tie from V*'s list (`shape_of_statements`: `RecVStar.statements` refuses `claims != 2 + 2 · regions`, d7c09dd16), `hS` from V*'s list alone (`holoStreams_of_list`: the list's `shape?` and its stacked table's `fast100`, so `repRounds` and `stackedRounds` build, by `decide` over `fast100`'s range; `repHead_zp` puts the `z_partial` round at `zpRound`), `hheld` from `hrd` (`repRoundsHeld_of_rounds`), `hcus` the guarantee's `_hIC`. Hypotheses left: E's `halg`/`hacc` (`withReads_algebraDecodes`), the values' `hpr`/`hsv`/`hf` (the W/K/C value tie), and `stageCoins_holo`'s stage ties `hrt`/`hst`/`hot`/`hlt` and `hlater` (`laterCoins_of_held`). The integrator's line: `hcoins := FlockC3.Compose.stageCoins_checked rfl hS₀ hss hcls hi halg hacc hpr hsv hf _hIC _htbl (fun _ _ _ _ _ => rfl) hrt hst hot hlt hlater` | H, done (round 24) |
| the rep claims' tie | `hcl`: the inner shape's claims are the class's, `ish.claims = 2 · ish.regions + 2`. The verifier refuses it already: `RecVStar.statements` throws at `i.claims != 2 + 2 · i.regions` (d7c09dd16, 9 Oct), so `shape_of_statements` gives it from the event's list (`hss`), and no `checkInnerShape` refusal is needed. `repFormsAt_holoAt` is `hA` from it, `ClassOk` and `checkInnerStmt`, the event's since round 24 (`AcceptsC3`) | `C3/Compose/CoinsChecked.lean`, `C3/Compose/HoloSparseWords.lean` | +13 | `holoForms_sparse_none`'s `hA`, so `holoSelWords` at the sparse streams; `repsDrawn_checked`'s `hA` and `hcl` (`stageCoins_checked`) | H, done (round 24) |
| the dirs value tie, (b) part 2 | `valueColl`, `tie_read`, `value_tie`, `dirs_tie` (merged); open: `hone`, `hpos`, `hseg` (VBridge's `PartPortsOne`, `ReadsOwn`, `ReadsUnsegmented`) at the read | `C3/RegLevels/ValueTie.lean` | | the tie without `hval`, in finder form | C |
| the value collision in the bound | `ValueColl`, `valueFinder`, `prob_valueColl_le`, `value_tie_coll`, `prob_regTopValue_le` (K rounds 5–6), the level-0 salt binder and `saltFinder`, `valueSrcC3`, `_hbr` as `VBridgeCOff` (C rounds 22–23; merged); open: the bound's `prob_or_le` over the value and salt finders | `Recursive/{ValueFinder,ValueSrc}.lean` | unsized | the value tie's and salt's collision disjuncts priced in the bound | K (round 7, on C's any-pub form) |
| OpenKeys, row 5 | row 5's remainder past `stackedStreams_of_covers` and `StackedCoins` (merged, R rounds 22–24; `hnd` and `LabelsSelf` closed) | `C3/RegLevels/{AccReads,OpensRegs,LgCoins,RRounds}.lean` | unsized | OpenKeys at `RepMaps.OpenCols`, R's levels from the checked sessions | R |
| the laid outputs | `OutsLaid` (`C3/RegLevels/OutRegs.lean:35`): a fail-closed `outsOk` refusal in `RecOpen.checkAt`, then its proof | verifier, `C3/RegLevels/OutRegs.lean` | unsized | `haout_of_laid`'s premise | open (after C's round 22) |
| `haout`'s second half | `haout` (`C3/RegLevels/OpensRegs.lean:105`) past `haout_of_laid` | `C3/RegLevels/{OpensRegs,OutRegs}.lean` | 100–150 | `haout` | C (later) |
| the reps' coins | `HVCoins` (`VBridge/Family/Rep.lean:487`) | `VBridge/Family/Rep.lean` | unsized | | L (round 22) |
| the reps' zero rounds | `HZeroL` (`VBridge/Family/Rep.lean:502`) | `VBridge/Family/Rep.lean` | to size | | L |
| the round reads' alignment | at the map's `sel`, `holoSel_opening_at` and `holoSel_parts` (`C3/Reps/RoundReads.lean`, S round 22) give `RoundReads`' `hsel`; open: `PartEnds sparsePart … (tiedSrc stp) (roundAt L .stacked 0)` (`srcOK_holo_rs`' `he`), `holoSel_parts`' `hm` (ports named `m<m>` against `heldBefore` of `msgRounds`) and `hc₀ = heldBefore (.rep 0) rs rs.size`, and its `hS` (`holoStreams_eq`) | `C3/Reps/RoundReads.lean`, `C3/Compose/` | unsized | `srcOK_holo_rs`'s `hr`, through `openMsgsRead_checked` | rec-lean (round 22) |
| the word-level `StageHeld` | `StageHeldW`: a stage's coins are one stream's words `(g q)[w]`, flat index `p ↔ (q, w)`, history round `j + p = i + 1 − (g q).size + w`; its coins lemma, from `coinsAt_of_idx` at `f (q, w) = coinArr fOf #[(g q)[w]!]` with custody's coins half and `CoinWordsEncoded` at `holoSelWords` (whose last conjunct is `holoForms_sparse_none`); `laterCoins_of_held`'s `h2`, `h3` at it | `C3/Compose/LaterHeld.lean` | 120–130 | the sparse phases' `StageCoins`, for `srcOK_holo_rs`'s `hcoins` | W |
| the sparse phases' `StageHeldW` | `h2`, `h3` at `sparseOf`'s stages (`stepsOf sh w cs`, one word a step), from `coinRounds_held` and `holoSelWords`, `coinWordsEncoded_holoMap` | | unsized | `laterCoins_of_held`'s `h2`, `h3` | open |
| the sparse stage's source | `srcSparseTie_of_sessions` (`SparseSessionAt` and `SparseSrcAt` at each accepted unit, into `sparseTie_of_session`); `sparseOf stp w c` (the sparse steps `stepsOf` at the session's words and coins) and `srcSparseTie_sparseOf`: `SrcSparseTie` at `tiedSrc (sparseOf stp w c)` from `SparseSessionAt` and `SparseRepTie` | `C3/Sparse/SrcSessions.lean`; `cursor/c3-w1-held-9b2a` | +89 | `srcOK_holo_rs`'s `hst`, at `stp := sparseOf stp w c` | W, reduction done (round 1) |
| `SparseRepTie`'s pub half | `pubOf sh c = holoPub … (u.rep i)` at the tied source: check 5's `checkSparse` registers `sparseV` at the record's `sparseCoins` and the rep's inner points, custody, the rep play's outputs, and `holoPub`'s `U` against `layoutFactors`/`eqTable` | | 200–300 | `srcSparseTie_sparseOf`'s pub conjunct | W (round 2) |
| `SparseRepTie`'s `h` half | `fOf w[s]! = (u.rep i).2.h s`: `ownPorts (.sparse _ hr)`'s `("h", rec-c{hr})`, read by `checkOwn`, tied as `ValueTie` ties a read | | 200+ | `srcSparseTie_sparseOf`'s `h` conjunct | W (round 2) |
| `SparseSessionAt` from acceptance | `sparseCoins c3 unit i = ok sc` from check 5, the rec-sparse copy's `RegBit`s and its `RegOut` zeros (`SparseCopyAt`), as `OpeningZeroAtL` is from `WReadO` | | 300+ | `srcSparseTie_sparseOf`'s session conjunct | W (round 2) |
| the public rows' premises | `RowsHeld` (`C3/PubRows/Check.lean:296`), with no producer yet; `PortsNamed` is derived (S, round 22): `RecVStar.checkMsgNames` refuses a circuit naming a port twice (`Flock/Test/MsgNames`), so `hPubRows_of_checked` reads it at each accepted session (`names_of_checkStmt`, `C3/PubRows/Shape.lean`) and `hbr_honest_reg` has no `hnames` | `C3/PubRows/Check.lean` | unsized | `hbr_honest_reg`'s `hdrawn` | E (round 4) |

## Start now (independent, rounds 1–4; proved, but a1c, a cleanup, is open)

Sizes are estimates of the proof, in lines. Owners in this table and the next: **W** a parallel worker, **C** circuits'
code, **L** rec-lean.

| # | lemma (stub) | file | size | depends on | owner |
|---|---|---|---|---|---|
| a1 | `FlockC3.Fork.holo_runs_sound`: `holoTableCL` on one prover's runs, `compiled_runs_sound`'s analogue | `C3/Fork/Runs.lean` | proved (round 4) from a1o and a1l, as `compiled_runs_sound` is from `table_value_clashFree` | a1o, a1l | |
| a1o | `FlockC3.Fork.holoTable_sound`: the oracle table (`holoTableOf`) accepts while no table close to `w` is committed with probability at most `holoError`; `table_sound`'s analogue, where a1s and a1r are used | `C3/Fork/OracleTable.lean` | 900–1400 | a1s, a1r; at most one message near `w` (`card_closeL_ud_le`), its cells a table `R`, and claims read only the cells; `holoLincheck` at `h = sparseFold` at `R.bits` is the lincheck of `R.toStatement` (`hunits`), so `rep_sound`'s phases and `link_sound`; each rep with its sparse phase on its own coins, multiplied as `value_interleave_le` does (`(repError + sparseError)^2`) | W |
| a1l | `FlockC3.Fork.holoTable_value_le_oracle`: on a clash-free set, the compiled table is accepted no more often than the oracle table at `w`; `table_prob_clashFree`'s analogue | `C3/Fork/Coupling.lean` | 900–1300 | `CompiledL`'s lockstep (`lockTableL`, `compiled_le_bad_on`, `bad_zero_of_fixed`) with `regPhase` and R's stream as a third table whose level 0 is fixed (`x.w`, by no `RegClash`) | W |
| a1s | `FlockC3.Fork.sparse_sound`: the sparse phase's Schwartz–Zippel count against `sparseError` | `C3/Fork/Sparse.lean` | 300–500 | `degRounds`' round step (the sum rule in characteristic 2), Schwartz–Zippel per `drawFin` coin; nothing in `Runs.lean` | W |
| a1r | `FlockC3.Fork.regOpen_sound`: R's opening at unique decoding against `regOpenError` | `C3/Fork/RegOpen.lean` | 250–400 | `fast100_uniqueDecoding` (proved, beside it); `ligerito_sound`'s argument at the unique-decoding radius, one message and no list; `opening`'s ring-switch and `γ` count; nothing in `Runs.lean` | W (proved, `cursor/c3-fork-regopen-95d4` @ `7c61afbd8`; lands with a1s in a1's assembly) |
| a1c | cleanup, not soundness: generalize level 0's Ligerito lemmas in `Soundness/` over the radius (they hard-code Johnson's) and drop a1r's copies (`udDecode`, `udPhaseA`, `ligerito_sound_ud`, …) | `C3/Fork/RegOpen.lean`, `Soundness/` | −480 | a1r | W |
| a3 | `FlockC3.Fork.holo_inner_sound_compiled` | `C3/Fork/Sound.lean` | proved (round 2), from a1 | | |
| a4 | `holoAt_fast100`, `holoAt_linkLayout`, `holoAt_rsch`, `holoAt_units` | `C3/Fork/Sound.lean` | proved (round 2) | | |
| b1 | `FlockC3.Compose.holoTableCL_plays` | `C3/Compose/Tape.lean` | 100 | `playsT_interleave_first`, `PlaysT` bind lemmas, `regPhase` at two claims | W |
| b2 | `FlockC3.Compose.vbridge_c3` | `C3/Compose/Bridge.lean` | 120 | b1, `HTape`; `vbridge_of_family`'s VCommits/VDecodes/VFull parts | W (after b1) |
| h1 | `FlockC3.VStarList.stacked_of_openingShape`, `claims_of_stackedShape` | `C3/VStarList/Statements.lean` | 25, 25 | `Except` unfolding of `checkOpeningShape`, `checkStackedShape` | W |
| h2 | `FlockC3.VStarList.vstarParsesScoped_of_list` (gap 4(c), proved: the parse and the scope are the event's); with gap 2, `mReg_of_list` (the list's `rec-opening` units are at `m = (holoAt st hp).cls.shape.mReg`, `claims = STACKED_CLAIMS`; no `hpad`, since `mReg` is padded to `PAD_TO`, and `ℓ`'s range is `ellOk_of_sparseRange`'s) | `C3/VStarList/Statements.lean`, `Shape.lean`, `Range.lean` | 400 | `HmRow.parse` of `circuitFile`'s output (a write-then-parse fact), gap 2 for the holo statements | C (gap 2) + W |
| c1 | (c)'s `hpl` generalization: drop `sh.holo = false` / `holoOf c = .ok false` from `lincheck_of_residuals_holo` (`Algebra/Holo.lean`), `zclc_of_residuals`, `prog_of_zclc` (`Runs/ZcLc.lean`), `run_of_algebra`, `run_of_frames` (`Runs/Rep.lean`), `prog_of_algebra` (`Runs/Prog.lean`), `rs_of_residuals` (`Runs/Rs.lean`), `ligerito_of_residuals_at`, `hlg_of_residuals`, `hlg_of_frames` (`Runs/Lg.lean`), `ligerito_of_residuals`, `ringSwitch_of_residuals`, `zerocheck_of_residuals`, `lincheck_of_residuals`, `build_eval`, `residuals_eq_run`, `buildRes_spec`, `buildAll_spec`, `coefOps_spec`, `prodOps_spec` (`Algebra/`), `decode_part`, `ResZeroAt` (`AlgHolds.lean`), `decode_checkR`, `decode_partR`, `ResZeroAtL` (`ResLayer.lean`), `decode_check` (`ResUnit.lean`), with `check_part_holo` beside `check_part` for the hidden units' net | `VBridge/` (in place, under VBridge G's names) | 600 | none; `check_refuses_plain_on_holo`, `shape?_holo` and `atomOp_plain` stay (plain facts) | W |

a1's 500–800 (rounds 1–3) counted only the composition: the plain chain under `compiled_runs_sound` (`table_sound`'s
oracle count and `CompiledL`'s lockstep, about 3,500 lines) has no holographic analogue yet, so a1o and a1l are new.

So these start at once: a1s and a1r (a W each, against their stubs alone), a1's assembly (L), h1 (W), b1 → b2 (W),
c1 (W); h2 builds on #1847's list.

v1, closed (round 20): verify ⇒ `SparseRangeOk`. `acceptsC3_of_verify` (`C3/VStarList/Accepts.lean`) concludes all of
`AcceptsC3` from the verifier's acceptance at both setups: each runs check 5's range check on the inner files first
(`RecSetup.checkInner`, `RecCoins.checkAtC3`), and `sparseRangeOk_of_check` reads its verdict at the event's `st`, set
up from the same circuit and public file (`setupHidden_same`).

## Next: the honest decoder and the stage discharges (stated, round 3)

Pieces (c)–(g) are obligations of a free decoder (`C3Family`'s `run`, `row`), as G's H-props are of `VStarFamily`'s
free fields. n0 fixes the decoder (`C3Family.honest`, `C3/Compose/Honest.lean`): `run` decodes the inner transcript's
steps stage by stage on the prefix each stage's game plays (`decodeRun`), `row` is the verifier's `fixedRow` at its
hold of the inner transcript (`heldRow`). `hbr_honest` (`C3/Compose/Discharge.lean`, proved) is `VBridgeC` at it, its
decoder reading the last round from the layers in full (`skLeaf`, `C3/Compose/Leaf.lean`); `_hob` is h2's `vstarParsesScoped_of_list`. n0's decoder `dec` is free, and every stage reads `dec`'s messages, so a
stage does not follow from the public rows: perturb one message on a stage's rounds and every stage still plays a
prefix of the same length while that stage's verdict fails (Worker S's counterexample, at rep 0's `v₂`). So each stage
is its read fact (its rounds' messages on the tape are its prover's for a witness its verifier row accepts), from which
`hX_of_read` gives the stage over any family, and the tape is its frame (`HFrame`, `hTape_of_frame`): obligations on
`dec` that n6 states at a fixed decoder. The round 3 stubs `hX_honest`, false at some `dec`, are deleted (round 5).

| # | lemma (stub) | file | size | depends on | owner |
|---|---|---|---|---|---|
| t0 | `stackedTop_of_innerCommits`: at the verifier's hold of the inner transcript, `stacked/rep0`'s first top is the binding round's, so where it is `cap`'s under the salts `salt` (`TopOf`, a binder: the salts the prover's workers committed R's level-0 top rows under, `registrant-holds-openings`), `StackedTopAt` at `salt`; the piece for lean's (b), the tie of `tr` to the sessions, and unused until then (the guarantee takes `_hstk`) | `C3/RegLevels/Top.lean`, `TopHold.lean` | proved (round 4; salt a binder, round 22) | | |
| t1 | `salt_tie`: link (d) at R's level 0, a read whose leaf is a node's commit string under the held salts is that node's row (`capRow`), or `saltFinder`'s hm96 pair collides (`SaltColl`, a disjunct of the guarantee; `prob_regTopValueSalt_le` its union with `regFinder`, `topFinder` and `valueFinder`, also a disjunct now, at the verifier's rows and salts `valueSrcC3`; `valueColl` and K's finder defined in `Recursive/ValueSrc.lean`, below the guarantee, since the value tie's proofs import it) | `C3/RegLevels/ValueTie.lean`, `Recursive/SaltFinder.lean`, `Recursive/ValueSrc.lean`, `Recursive/ValueFinder.lean` | proved (round 22) | | |
| t2 | `hT_of_root`: `opensTopAt_of_regs`' `hT` at R's levels `ℓ ≥ 1` (`out`'s 512 columns registered as the cap entry's bits), or `valueColl` collides, from a root over rows whose read row is the entry's (`tie_root`'s `hroot`); `capRow_bit` (row bit `b < 512` is the node's bit `b`) and `capAt_size` (`htop`, from `CapsAt`). The verifier does not tie a level's port root to a tree over its rows, so `hroot` is open. Its close is the leaf route and needs no premise beyond `registrant-holds-openings`: R's level-`ℓ` tops are rows of ports the reads file `own` names (`checkPubRows`, `fixedRow`'s `stacked/rep0` `tops[ℓ][x]`), so `RegistrantCovers` already makes every read of them a registered read under the salt the prover holds; in place of `hroot`, the read's leaf is the tree leaf of the transcript's `b ‖ c` at its row (`checkPubRows` and the table row's opening of the same root, else a tree collision), and `salt_tie` at the read's own salt | `C3/RegLevels/TopTie.lean` | proved at `hroot` (round 22) | `hroot` | |
| t3 | `_hbr` off the verifier's values' collisions: `RecursiveAuditC3At`'s `_hbr` is `Flock.Premises.VBridgeCOff (ValueColl zs (valueSrcC3 c3))` at `c3 = fun t => c3Of ish (innerHeld wr st reg t)`, the `vsrc` binder gone (`valueSrcC3` and its rows defined in `Recursive/ValueSrc.lean`); `zk_sessions_recursive_fork_regV` (on K's `fork_innerLV`) reads it; `hbr_honest_reg` gives it (`vbridgeCOff_of_vbridgeC`). The value tie at a statement check 5 accepted: `value_tie_stacked` (`coef`, `dirs`), `value_tie_opening` (`v`), `coins_of_session` (the statement of a drawn unit's part passes `RecCoins.check`); `value_tie` at any public file | `Recursive/{RegExtract,AuditC3Proof,ValueChecked}.lean`, `C3/Compose/Discharge.lean` | proved (round 23) | the read facts off `ValueColl`, where these ties enter; `rec-open`, `rec-residuals`, `rec-sparse` values (no rows in `valueRows` yet) | |
| n1 | `hPubRows_of_check'` (from `ChecksPubRows`, `RowsHeld`, `PortsNamed`, `SparseAtShape`); `hTape_of_frame` (from `HFrame`, `hTape_iff_frame`) | `C3/PubRows/Honest.lean`, `Check.lean`, `Frame.lean` | proved (round 4) | | |
| n1′ | `sparseAtShapeChecked`: `SparseAtShape` at the units `checkStmt` passes (`SparseAtShapeChecked`), for every family; `hPubRows_of_checked`: `hPubRows_of_check'` with it, where some transcript each accepting session accepts passes the verifier's checks (`ChecksAccepted`; at the checked family, `checksAccepted_checked`) | `C3/PubRows/Shape.lean` | proved (round 5, H; round 6) | | |
| n2 | `hReps_of_read` (with `hRepStages_honest`, so `hReps_honest_of_read`): `HReps` from the reps' read fact (`RepsRead`) | `C3/HoloValues/Read.lean`, `Shape.lean`, `ZcLc.lean` | proved (round 5, C) | | |
| n2a | the transcript-level holo rep program `holoRepProg`, its refinement to `holoRepC` (`holoRep_refines`) and its run from the stages' checks (`holoProg_of_checks`) | `Soundness/Refine/HoloRep.lean`, `HoloRefines.lean`, `HoloRepConverse.lean` | proved (round 4) | | |
| n3 | `hSparse_of_read` (with `hStages_honest`, so `hSparse_honest_of_read`): `HSparse` from `SparseRead` | `C3/Sparse/Read.lean` | proved (round 4) | | |
| n4 | `hOpening_of_decode`, `hOpening_of_read`: `HOpening` from `OpeningRead`, O1–O6 at `mReg`; `openingRead_of_rows`: `OpeningRead` from the `rec-opening` unit's residuals (`residuals_eq_runO`), at its session and coins (L, `openingRead_of_session`), the claim tie (`ClaimRel` from `claimO` to `u.claim i`: `claimRel_claimO`, from the claim's point at `coinClaim`'s padded coin point `coinPt`, `PointTie`, and its value at w's `rec-claim`, `ValueTie`) and the tape prefix (n6) | `C3/Opening/Read.lean`, `Rows.lean`, `SessionCoins.lean`, `ClaimTie.lean` | proved (rounds 4, 5, 7) | | |
| n5 | `hRegLevels_of_read` (with `hLgStages_honest`, so `hRegLevels_honest_of_read`): `HRegLevels` from `RegLevelsRead`, R's tape against `xR.cap` | `C3/RegLevels/Read.lean` | proved (round 4) | | |
| n6 | `tape_c3DecOf`: at the fixed decoder `c3DecOf` (G's `decOf` at candidate 3: `c3Msgs` cuts the source run's steps at their coins, `c3DecOf_spec` proved), the inner transcript is the source run's steps wherever `SrcOK` holds; from it the read facts at the honest family at that decoder are proved (`at_c3DecOf`, `hFrame_c3DecOf`, `repsRead_c3DecOf`, `sparseRead_c3DecOf`, `openingRead_c3DecOf`, `regLevelsRead_c3DecOf`) | `C3/Compose/Decoder.lean` | proved (round 6, H) | | |
| n7 | the source: the run V*'s layers hold, read stage by stage. As first stated (`SrcOK` and each read fact on its own steps) it is false at a free source: `SrcOK` fixes the source's kinds, coins and types, not its values (C's, S's and L's stops, round 6). Restated as one tie per stage beside `SrcOK`, each read fact from its tie and V*'s sessions: R's opening (`openingReadAt_src_ties`, from `OpenTie`, `PointTie`, `ValueTie` and `SessionAt`), the sparse phases' (`srcSparseOK_of_tie`, from `SparseTie`), the reps' (`srcRepsReadAt_of_tie`, from `RepTie`) and R's levels' (`regLevelsRead_of_tie`, from `LevelTie`, for `RegTape`) are proved. The ties from V*'s layers: `openTie_of_layer`, `sparseTie_of_layer` and `openTieL_of_session` (`OpenTieL` from `SessionAt`, through `cover_rs`) are proved; `srcRepTie_of_alg` takes G's items 1–4 at holo and `RepUnitOfAlg`'s four named parts (of which the holo frames and counts are proved on the `LigShift` stub, and the value stubs from `ProofFrom`/`SetupFrom` at a rep), and `levelTie_of_run` takes R's opening's run with `OpenRel` and the `Opened` climb as hypotheses (the climb proved from one `OpensTopAt` per opening, `opened_of_units`). `SrcOK`'s coin clause and `MsgTysC3` are proved stage by stage (`srcOK_of_stages`: the coin clause from `HLeafC3` and each stage's `CoinsAt`, which the ties' discharges add, their coins being existential; `MsgTysC3` from each stage's `TysAt`), and its decode fixed point from each stage playing on its steps to the run's own output (`decodeRun_runSteps`; `srcOK_of_plays` at `RunPlays`). The tied source (`tiedSrc stp`, `Compose/Tied.lean`) sets each stage's output to its play's on `stp`'s steps, so `RunPlays` holds wherever each stage plays to some output (`runPlays_tie`, `StagesPlay`, which the ties give: `stagesPlay_of_ties`), and each stage's `TysAt` follows from its play along its game's spine (`tysAt_of_runPlays` at `ShapeTys`): `srcOK_tied_of_shapes` is `SrcOK` there from `HLeafC3`, `StageCoins`, `MsgsReadC3` and the shapes' types. A tied source meets `MsgsReadC3` at the committed rounds only: the last round's openings are in no held row (H, as G's `MsgsRead`), so the decoder reads that round from every read wire of the layers (`skLeaf`; before it, `skOf`'s mask, `maskOf_skLeaf`) | `C3/Compose/`, `C3/Opening/{SrcOpening,ClaimTie,SrcLayer,SrcCover}.lean`, `C3/Sparse/Src*.lean`, `C3/Reps/{Tie,Vals}.lean`, `C3/HoloValues/{Frames,Runs}.lean`, `C3/RegLevels/Src*.lean` | each read fact from its tie, the coin clause and `MsgTysC3` from the stages (rounds 6–7); the ties' open parts sized in the round 7 done block; VBridge restated at the committed rounds (`VBridgeC`, H, round 8); the last round's `MsgsReadC3`, at the layers' read wires, is open | n6; `RepsRead` (n2) | |

`hCommits_honest` (`VCommits` at the honest family, G's `hCommits_of` after `pubCommits_of_pubRows`) is proved, and `hcm`
is it at the committed rounds (`vCommits_masked`); `hdec` is `vDecodesR_wmsgLeaf`. `hbr_honest` is at the checked family (`C3Family.checked`: the
honest family at `checkedC3`, round 6), so its conclusion is `_hbr` itself, and the verifier's public-row and statement
checks are the kept sessions' (`checksAccepted_checked`, into `hPubRows_of_checked` at `sparseAtShapeChecked`). What it
takes, each with who discharges it:

* `hrread`, `hsread`, `horead`, `hlread` (`RepsRead`, `SparseRead`, `OpeningRead`, `RegLevelsRead`), through
  `hReps_honest_of_read`, `hSparse_honest_of_read`, `hOpening_of_read`, `hRegLevels_honest_of_read`, and `hfr` (`HFrame`): at n6's fixed
  decoder, from `tape_c3DecOf` and the source's facts (n7). `SrcOK`'s `HLeafC3` is G's `HLeaf`: no decoder meets a
  read fact or `HFrame` at a history shorter than the inner rounds (`zip` stops at the last coin; at `t = []` the tape
  is empty), and the checked sessions refuse one (`hLeafC3_checked`, wherever V* has a statement).
* `hdrawn` (`RowsHeld`): G's `RowsDrawn` at every value `fixedRow` knows, not only `rec-c<k>` (W). The drawn statement's
  label of a unit is the unit by the verifier's refusal of any other (`RecVStar.checkLabels`, in `ChecksC3` and
  `PubRowsChecked`; `label_of_checked`), so `RowsHeld` asks nothing of the labels (E).
* `PortsNamed` (G's `PortNamesUnique` at V*'s circuits) is no hypothesis: the statement check refuses a circuit naming a
  port twice (`checkMsgNames`), so it holds at each accepted session (`names_of_checkStmt`, in `hPubRows_of_checked`).
* `hown` (`OwnIn`): each accepted session's reads hold `own`'s ports by name, value and reads. At
  `own := fun j => ownAt ss j`, the verifier's message ports of statement `j` (`ownPorts` and RecSparse's other
  messages, `sparseMsgs`), it is proved (`ownIn_checked`, `C3/PubRows/Shape.lean`, from `checkOwn`'s second and third
  loops). The session may read more; no reader of `own` needs the converse. `RoundReads` and `WRoundReads` at that
  `own` are `C3/Reps/RoundReads.lean`'s (`roundReads_checked`, `wRoundReads_checked`, from U's links in
  `C3/Reps/RoundLinks.lean`), with the stage selector at the rounds as hypotheses. At the map's `sel` R's is proved
  (`roundReads_checked` at `holoSel_opening_at`, at `c₀ := roundAt L .stacked 0`, R's round 0's history round); what is
  left of `srcOK_holo_rs`'s `hr` is that `PartEnds`' `c₀` (`he`) is that round. `WRoundReads`' `hsel` at the map is
  `holoSel_parts`, given its `hm` (each part's `m<m>` a reps' message round, at `roundOf σ m`) and `hc₀`.
* `hheld` (`HeldIs`): n1c, below.
* `hss` and `_htop`: the event's (`AcceptsC3`'s list and the guarantee's `_hstk`); `_htop` is unused by `hbr_honest`
  and is what n6's `RegLevelsRead` reads.

* n1b, the V* session checks (round 5, in the event): VBridge reads V*'s sessions' acceptance as `AcceptsZKR`, which
  does not include the checks the verifier makes of them under `--rec-inner`. The guarantee's event now carries them
  per session (`ChecksVR`, `ChecksC3`: `checkStmt` with `checkOwn`, check 5's `RecCoins.check` at `c3Of`,
  `checkPubRows`, `checkLabels` and, at the held inner commits, `checkCovers`), and VBridge, the forks and the bound are at the checked sessions (`checkedC3`: each typed session
  kept where an accepted transcript passes `ChecksC3`, `freshOwn` elsewhere; `checkedT` its prover), whose games are
  the typed sessions'. `vbridge_checked` gives VBridge at `checkedC3` from VBridge at the typed sessions; `hbr_honest`
  is at the checked family itself (round 6), where the public-row and statement checks are the kept sessions'
  (`ChecksAccepted`: where a session accepts, some transcript it accepts passes them, `checks_of_keepOwn`).
* n1c, the record–history link: `row` reads the records the coin server writes at the history (`wr t`, round 10;
  the guarantee reads them there, `innerHeld`), so `hheld` (`HeldIs`: at each accepted history the held row is `t`'s
  commitment) is custody of those records as the history's, beside `RecordCustody`: `InnerRecordCustody`
  (`Soundness/Assumptions/Inner.lean`, `live-verifier` at the inner session, the guarantee's `_hIC` at the map `im`),
  which gives `HeldIs` (`heldIs_of_custody`, proved, its `hrow` `rfl` at the honest family).
* The guarantee's form: `hbr_honest` is at the fixed word `fun _ _ => xR`; `hbr_honest_reg` is `_hbr` itself, at the
  word the runs open (`regStmt`), whose cap is `cap`, off `ValueColl` at the verifier's rows: candidate 3's table reads
  only the cap (`holoAccepts_of_cap`), so VBridgeC moves between statements of one cap (`vbridgeC_of_cap`), and gives
  VBridgeCOff at every bad event (`vbridgeCOff_of_vbridgeC`). The guarantee's message types are the stages'
  (`holoTys st hp`, round 10), which `ShapeTys` reads.

## Circuits' code

* Gap 2 is #1847 (draft): the eleven statements, a list missing RecSparse, an R level or the opening refused by name, an
  opening at another shape refused. It moves `PublicAudit`'s and this guarantee's read records (`RecVStar.statements`).
* `RecCoins.check` (check 5) covers `rec-sparse/`, `rec-open-stacked/` and `rec-opening/` at the coin server's coins since
  #1847's `9b2799c7e` (merged here at `100c17932`). No statement here takes any kind's coins as a hypothesis.
* Gap 4, the verifier's own `rec-claim`, is circuits' and in #1847: its row `x` is `sparse/rep<x>`'s claim round's `c_k`
  (`RecVStar.claimRow`, through `fixedRow`). The statement takes V*'s bindings (`bind`) as the verifier of record's
  inputs, so the claim it cites is the one the verifier computes.
* The guarantee holds at every `hp : HoloParams`; to state with h2: `hp` is the verifier's own, the inner shape's
  `sparse` entry (`RecVStar.Inner.sparse`, `[NU, ELL, S_LOG, T_LOG]`).
* Private inner sessions: `RecVStar.statements` refuses a holo inner shape (`statements_holo`), so the guarantee covers
  none until V*'s holo-parts writer lands with its proof (#1896, its own PR after C3 closes).

## Decisions

* Sparse-phase order: the recorded one. `holoTableCL` plays each rep's sparse phase after both reps, reading only that
  rep's zerocheck, lincheck and `h`, fixed before its coins, so the bridge replays the record round for round and no
  reordering lemma is needed.
* `Creg`: the inner statement's `x.cap`, tied by `StackedTopAt` at explicit commit strings to the META-bound record's
  structurally checked `stacked` entry; the
  registrant's openings of every level-0 position are `_hreg` (`HoloReg.Opens`). Closeness is in `holds`, not a hypothesis.
* The tie is the verifier's (round 4 follow-up): the event's `AcceptsC3` holds the inner transcript against `reg`
  (`innerCommitsAt`, `RecVStar.innerCommits` at the verifier's own `tr`, `record` and `reg`), whose binding round's top
  must match the `stacked` entry's five fields as well as its canonical form. `_hstk : StackedTopAt … salt bcs'` carries
  the registered strings explicitly, `cap`'s under `salt`, the held salts the guarantee is applied at
  (`registrant-holds-openings`); a different held top is `topFinder`'s one-run collision event, and a read of the top
  other than its node's row `saltFinder`'s. The digest is
  `metaRegistration st`, META's `registration` of the inner circuit the verifier reads off
  `--rec-inner-circuit`, the file `AcceptsPublic` sets `st` up from.
* Who registers R: the inner session's prover. Its staging commits R's table and writes the registration record, and
  the verifier checks only that META binds the record's digest and that the binding round's top makes its `stacked`
  entry; nothing checks that the cap is a tree over some word. `_hreg` takes the registrant's openings of every level-0
  position, so a prover who registers a cap with no preimage is outside the statement (Weakness, below).
* `ℓ` and the slots' units come from check 5 (round 3, ruling 2 and 4): `SparseRangeOk` (in the event) is
  `RecSparse.Shape.valid`, `sparseV`'s shortness check and `layoutFactors` at the inner circuit's unit range, so
  `ellOk_of_sparseRange` gives `hp.EllOk` and `units_of_sparseRange` gives the slots' units at `UnitsOf`. Every
  candidate-3 session's setup evaluates it on the inner statement from its files, whatever the unit, and refuses a
  session without them (`RecCoins.checkAtC3`); `sparseRangeOk_of_check` reads it off that check at `ClassOk`.
* `mReg` is padded (ruling 4): `mReg = max PAD_TO (…)` and R's bits are zero-padded to it, so `7 ≤ mReg` is free
  (`seven_le_mReg`) and `mReg_of_list` has no `hpad`. Workers on a1s and a1r: a proof that unfolds `mReg` meets the
  `max`.
* The padding is the registrant's (round 4): nothing checks that R's padding bits are zero, and no claim reads them
  (`RegClaim.coord` is zero past the cells). So `RegSchedule.Close` compares only the cells of the message near `w`
  with `R.bits`. Comparing every bit, as round 3 left it, made the inner statement false at a word whose padding is not
  zero while the table in its cells is satisfiable, and an honest prover there is accepted, so `holo_runs_sound` (and
  the guarantee's proof) could not hold.
* `_hob` from the event (ruling 5, round 6): `_hob` is `VStarParsesScoped`, `VStarParses` at the files the verifier's
  acceptance parses into scope. The parse is `ScopeZR.parseH`'s and the scope is #1847's refusal in `checkStmt`
  (`scope_of_accepts`), so `loads_of_verifiesV` reads `_hob` only there, and the verifier's list meets it with no
  writer's round trip (`circuitFile_parses` is gone). `PublicAudit` and `RecursiveAudit` keep `VStarParses`, which gives
  it (`vstarParsesScoped_of`).
* Whole histories (G's `HLeaf`, round 6): `checkedC3` keeps a session only at a history with a coin for each inner
  round, the only histories `recGame` plays, so `HLeafC3` holds at the checked family wherever V* has a statement.
* `m` and claims: derived, not hypotheses. The model's R schedule is `fast100` at `mReg` (`holoAt`), and `claims` is the
  constant `STACKED_CLAIMS` (`regPhase` opens two, `regOpenError` counts two). The verifier's link is transitive: each
  `rec-opening` unit matches META's `stacked` (`stacked_of_openingShape`), which opens two claims at `fast100`'s rounds
  (`claims_of_stackedShape`), and the units' names come from the outer program the verifier compiles (gap 2).
* The claim round (#1847's `100c17932`: each rep's sparse stream ends with its claim, a message without coins): the model
  is unchanged. `sparsePhase` computes the claim (`sparseVerdict`'s `batch`, `Σ_p eq(ζ_p, p)·u_p`), and RecSparse's
  `claim` port, which reads `rec-claim`'s row, checks the sent `c_k` against `u` and `ζ_p`. So n3 takes the round's
  value as RecSparse's `claim` check gives it, and n1's decoder folds the coinless round into the sparse steps.
* Unit slots in range (round 2): `HoloClass.ofExec` keeps slot `σ`'s units below `2^(k_log − ℓ)`. The sparse phase's
  `U_σ` reads a unit's low `k_log − ℓ` bits (`holoPub`'s `eqAt`), while `RegTable.side` places a template only at blocks
  the table has, so a unit past them made the two disagree and a3 false. a3 and `holo_runs_sound` take `hunits`, which
  `holoAt_units` gives. The verifier refuses units past the block (check 5's `layoutFactors`; resolved in round 3 by
  `units_of_sparseRange`).
* `sparseError`, recounted (round 2): `18 + (2ℓ+1)ν + (2ℓ+2)·sLog + pLog + tLog` over `qF` (H2's six rounds are of
  degree 2, H3's of `2ℓ + 1` and `2ℓ + 2`), where round 1 had `12 + 2ℓν + (2ℓ+1)·sLog + …` times the level-0 list
  bound. No list factor: the phase's claims are on `R.bits`, which R's opening decodes uniquely. Only
  `RecursiveAuditC3`'s record moves.

## Weakness

R's level 0 at `fast100` is queried at the unique-decoding radius (rate 1/2, 218 queries): `udQueryError` is about
2^-88.4, so `holoError` is at least that, against PublicAudit's list-decoding terms. Bits to cite: 88, not `fast100`'s
100, which is its list-decoding figure; 100 bits at the unique-decoding radius needs about 247 level-0 queries for R.

A prover registers R (Decisions), and the statement covers only a registration whose cap the registrant can open at
every level-0 position (`_hreg`): a prover who registers a cap with no preimage is in no event the guarantee bounds.

## Stopped: gap 4(b)

The universal class (every statement's table one class) stops: candidate 3's guarantee is per statement
(`holoAt st hp`). Its work stays, unmerged, on `cursor/rec-universal-class-95d4`, `cursor/rec-universal-lower-95d4` and
`cursor/rec-universal-stmt-gaplist-b97f`.
