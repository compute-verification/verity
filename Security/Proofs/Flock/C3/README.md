# Candidate 3's recursive guarantee: the gap list

`Flock.SecurityProofs.RecursiveAuditC3 : Flock.Guarantees.RecursiveAuditC3` (`Recursive/AuditC3.lean`,
`Recursive/AuditC3Proof.lean`) is PublicAudit's proof at candidate 3's table (`holoInnerAt`, R from the registration's
`stacked` entry, V*'s statements over the verifier's own list). Its only `sorry`s are the stubs below; the guarantee's
proof itself has none. It takes VBridge (`_hbr`) and V*'s parse (`_hob`) as PublicAudit does; pieces (b)–(h) discharge
them. V*'s statements are indexed by the verifier's own list (`vstarCount ish`), which at candidate 3's inner shape is
circuits' gap 2's (#1847, merged here at `100c17932`): eleven statements in the writer's order, 0–3 `rec-open`, 4–5
`rec-residuals`, 6 `rec-sparse`, 7–9 R's levels `rec-open-stacked`, 10 `rec-opening` at META's `stacked` shape.

Sizes are estimates of the proof, in lines. Owners: **W** a parallel worker, **C** circuits' code, **L** rec-lean.

## Start now (independent)

| # | lemma (stub) | file | size | depends on | owner |
|---|---|---|---|---|---|
| a1 | `holoTableCL`'s model soundness (`holo_runs_sound`, `compiled_runs_sound`'s analogue; not yet stubbed: it is the body of a3) | new `C3/Fork/Runs.lean` | 1,000–1,500 | the sparse phase's Schwartz–Zippel count (`sparseError`), R's opening at unique decoding (`regOpenError`), the reps as `table_sound_compiledL` | W (sparse, R) + L (assembly) |
| a3 | `FlockC3.Fork.holo_inner_sound_compiled` | `C3/Fork/Sound.lean` | 50 | a1, as `flock_inner_sound_compiled` from `compiled_runs_sound` | L |
| a4 | `holoAt_fast100`, `holoAt_linkLayout`, `holoAt_rsch` | `C3/Fork/Sound.lean` | 30, 40, 30 | `compiledForkSound_execClassAt`'s lemmas; `schedAt`'s `UniqueDecoding` | W |
| b1 | `FlockC3.Compose.holoTableCL_plays` | `C3/Compose/Tape.lean` | 100 | `playsT_interleave_first`, `PlaysT` bind lemmas, `regPhase` at two claims | W |
| b2 | `FlockC3.Compose.vbridge_c3` | `C3/Compose/Bridge.lean` | 120 | b1, `HTape`; `vbridge_of_family`'s VCommits/VDecodes/VFull parts | W (after b1) |
| h1 | `FlockC3.VStarList.stacked_of_openingShape`, `claims_of_stackedShape` | `C3/VStarList/Statements.lean` | 25, 25 | `Except` unfolding of `checkOpeningShape`, `checkStackedShape` | W |
| h2 | `FlockC3.VStarList.vstarParses_of_list` (gap 4(c)); with gap 2, `mReg_of_list` (to state: the list's `rec-opening` units are at `m = (holoAt st hp).cls.shape.mReg`, `claims = STACKED_CLAIMS`) | `C3/VStarList/Statements.lean` | 400 | `HmRow.parse` of `circuitFile`'s output (a write-then-parse fact), gap 2 for the holo statements | C (gap 2) + W |
| c1 | (c)'s `hpl` generalization: drop `sh.holo = false` / `holoOf c = .ok false` from `lincheck_of_residuals_holo` (`Algebra/Holo.lean`), `zclc_of_residuals`, `prog_of_zclc` (`Runs/ZcLc.lean`), `run_of_algebra`, `run_of_frames` (`Runs/Rep.lean`), `prog_of_algebra` (`Runs/Prog.lean`), `rs_of_residuals` (`Runs/Rs.lean`), `ligerito_of_residuals_at`, `hlg_of_residuals`, `hlg_of_frames` (`Runs/Lg.lean`), `ligerito_of_residuals`, `ringSwitch_of_residuals`, `zerocheck_of_residuals`, `lincheck_of_residuals`, `build_eval`, `residuals_eq_run`, `buildRes_spec`, `buildAll_spec`, `coefOps_spec`, `prodOps_spec` (`Algebra/`), `decode_part`, `ResZeroAt` (`AlgHolds.lean`), `decode_checkR`, `decode_partR`, `ResZeroAtL` (`ResLayer.lean`), `decode_check` (`ResUnit.lean`), with `check_part_holo` beside `check_part` for the hidden units' net | `VBridge/` (in place, under VBridge G's names) | 600 | none; `check_refuses_plain_on_holo`, `shape?_holo` and `atomOp_plain` stay (plain facts) | W |

So four start at once: a1/a3 (L with a W on the sparse and R counts), a4 + h1 (W), b1 → b2 (W), c1 (W); h2
builds on #1847's list.

## Next: the honest decoder and the stage discharges

Pieces (c)–(g) are stated as obligations of a free decoder (`C3Family`'s `run`, `row`), as G's H-props are of
`VStarFamily`'s free fields; they are only jointly meaningful through `HTape`. Their discharge at the decoder of V*'s
accepted layers is the next layer, stubbed when (b)'s family is fixed:

| # | lemma (to state) | size | depends on | owner |
|---|---|---|---|---|
| n0 | `C3Family.honest`: the family at `custodied zr _hCust h`, `run` the decode of V*'s layers stage by stage, `row` by `fixedRow`'s prefixes | 400 | b2, h2 | L |
| n1 | `hTape_honest`, `hPubRows_of_check` (g), from `checkPubRows`/`fixedRow` per prefix | 300 | n0 | W |
| n2 | `hReps_honest` (c) | 250 | n0, c1 | W |
| n3 | `hSparse_honest` (d), from `sparse_refines`, `sparsePhase_plays`, with the claim round's `c_k` as RecSparse's `claim` check | 250 | n0 | W |
| n4 | `hOpening_honest` (e): RecOpening's algebra at opening-only shape (`rec-opening/m<m>-c2-p<j>`; no zerocheck or lincheck, O1–O6 at `mReg`) | 300 | n0, c1's `ringSwitch_of_residuals` | W |
| n5 | `hRegLevels_honest` (f): R's levels on `stacked/rep0` after both reps, `rec-top-stacked-L<l>` tops, the binding round against `StackedTop`'s cap | 250 | n0, c1's `ligerito_of_residuals_at` | W |

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

## Decisions

* Sparse-phase order: the recorded one. `holoTableCL` plays each rep's sparse phase after both reps, reading only that
  rep's zerocheck, lincheck and `h`, fixed before its coins, so the bridge replays the record round for round and no
  reordering lemma is needed.
* `Creg`: the inner statement's `x.cap`, tied by `StackedTop` (`_hstk`) to the META-bound record's `stacked` entry; the
  registrant's openings of every level-0 position are `_hreg` (`HoloReg.Opens`). Closeness is in `holds`, not a hypothesis.
* `_hstk`'s tie (round 2): the digest is `metaRegistration st`, META's `registration` of the inner statement's circuit,
  which the verifier reads off `--rec-inner-circuit` (`innerMetaRegistration`), the file `AcceptsPublic` sets `st` up
  from; `reg` is the verifier of record's `--rec-inner-registration`, an input as `bind` and `rows` are. The verifier
  holds the tie in `RecVStar.innerCommits` (`registration`), which `VerifiesVStar` does not model, so it stays a
  hypothesis, but one `st` fixes: SHA-512 and the root bind `reg`, its `stacked` entry and so the cap to `st`'s META.
* `m` and claims: derived, not hypotheses. The model's R schedule is `fast100` at `mReg` (`holoAt`), and `claims` is the
  constant `STACKED_CLAIMS` (`regPhase` opens two, `regOpenError` counts two). The verifier's link is transitive: each
  `rec-opening` unit matches META's `stacked` (`stacked_of_openingShape`), which opens two claims at `fast100`'s rounds
  (`claims_of_stackedShape`), and the units' names come from the outer program the verifier compiles (gap 2).
* The claim round (#1847's `100c17932`: each rep's sparse stream ends with its claim, a message without coins): the model
  is unchanged. `sparsePhase` computes the claim (`sparseVerdict`'s `batch`, `Σ_p eq(ζ_p, p)·u_p`), and RecSparse's
  `claim` port, which reads `rec-claim`'s row, checks the sent `c_k` against `u` and `ζ_p`. So n3 takes the round's
  value as RecSparse's `claim` check gives it, and n1's decoder folds the coinless round into the sparse steps.

## Weakness

R's level 0 at `fast100` is queried at the unique-decoding radius (rate 1/2, 218 queries): `udQueryError` is about
2^-88.4, so `holoError` is at least that, against PublicAudit's list-decoding terms.

## Stopped: gap 4(b)

The universal class (every statement's table one class) stops: candidate 3's guarantee is per statement
(`holoAt st hp`). Its work stays, unmerged, on `cursor/rec-universal-class-95d4`, `cursor/rec-universal-lower-95d4` and
`cursor/rec-universal-stmt-gaplist-b97f`.
