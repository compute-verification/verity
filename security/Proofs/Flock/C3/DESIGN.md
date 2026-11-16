# Candidate 3's recursive guarantee: design

The decisions behind candidate 3's model and proof. `README.md` says what the guarantee is, what stays open and its
weakness.

## Decisions

* Sparse-phase order: the recorded one. `holoTableCL` plays each rep's sparse phase after both reps, reading only that
  rep's zerocheck, lincheck and `h`, fixed before its coins, so the bridge replays the record round for round and no
  reordering lemma is needed.
* `Creg`: the inner statement's `x.cap`, tied by `StackedTopAt` at explicit commit strings to the META-bound record's
  structurally checked `stacked` entry; the registrant's openings of every level-0 position are `_hreg`
  (`HoloReg.Opens`). Closeness is in `holds`, not a hypothesis.
* The tie is the verifier's: the event's `AcceptsC3` holds the inner transcript against `reg` (`innerCommitsAt`,
  `RecVStar.innerCommits` at the verifier's own `tr`, `record` and `reg`), whose binding round's top must match the
  `stacked` entry's five fields as well as its canonical form. `_hstk : StackedTopAt … salt bcs'` carries the registered
  strings explicitly, `cap`'s under `salt`, the held salts the guarantee is applied at (`registrant-holds-openings`); a
  different held top is `topFinder`'s one-run collision event, and a read of the top other than its node's row
  `saltFinder`'s. The digest is `metaRegistration st`, META's `registration` of the inner circuit the verifier reads off
  `--rec-inner-circuit`, the file `AcceptsPublic` sets `st` up from.
* Who registers R: the inner session's prover. Its staging commits R's table and writes the registration record, and
  the verifier checks only that META binds the record's digest and that the binding round's top makes its `stacked`
  entry; nothing checks that the cap is a tree over some word. `_hreg` takes the registrant's openings of every level-0
  position, so a prover who registers a cap with no preimage is outside the statement (Weakness, below).
* `ℓ` and the slots' units come from check 5: `SparseRangeOk` (in the event) is `RecSparse.Shape.valid`, `sparseV`'s
  shortness check and `layoutFactors` at the inner circuit's unit range, so `ellOk_of_sparseRange` gives `hp.EllOk` and
  `units_of_sparseRange` gives the slots' units at `UnitsOf`. Every candidate-3 session's setup evaluates it on the inner
  statement from its files, whatever the unit, and refuses a session without them (`RecCoins.checkAtC3`);
  `sparseRangeOk_of_check` reads it off that check at `ClassOk`.
* `mReg` is padded: `mReg = max PAD_TO (…)` and R's bits are zero-padded to it, so `7 ≤ mReg` is free
  (`seven_le_mReg`) and `mReg_of_list` has no `hpad`. A proof that unfolds `mReg` meets the `max`.
* The padding is the registrant's: nothing checks that R's padding bits are zero, and no claim reads them
  (`RegClaim.coord` is zero past the cells). So `RegSchedule.Close` compares only the cells of the message near `w` with
  `R.bits`. Comparing every bit would make the inner statement false at a word whose padding is not zero while the table
  in its cells is satisfiable, and an honest prover there is accepted, so `holo_runs_sound` could not hold.
* `_hob` from the event: `_hob` is `VStarParsesScoped`, `VStarParses` at the files the verifier's acceptance parses into
  scope. The parse is `ScopeZR.parseH`'s and the scope is `checkStmt`'s refusal (`scope_of_accepts`), so
  `loads_of_verifiesV` reads `_hob` only there, and the verifier's list meets it with no writer's round trip.
  `PublicAudit` and `RecursiveAudit` keep `VStarParses`, which gives it (`vstarParsesScoped_of`).
* Whole histories (G's `HLeaf`): `checkedC3` keeps a session only at a history with a coin for each inner round, the
  only histories `recGame` plays, so `HLeafC3` holds at the checked family wherever V* has a statement.
* `m` and claims: derived, not hypotheses. The model's R schedule is `fast100` at `mReg` (`holoAt`), and `claims` is the
  constant `STACKED_CLAIMS` (`regPhase` opens two, `regOpenError` counts two). The verifier's link is transitive: each
  `rec-opening` unit matches META's `stacked` (`stacked_of_openingShape`), which opens two claims at `fast100`'s rounds
  (`claims_of_stackedShape`), and the units' names come from the outer program the verifier compiles.
* The claim round (each rep's sparse stream ends with its claim, a message without coins): the model is unchanged.
  `sparsePhase` computes the claim (`sparseVerdict`'s `batch`, `Σ_p eq(ζ_p, p)·u_p`), and RecSparse's `claim` port, which
  reads `rec-claim`'s row, checks the sent `c_k` against `u` and `ζ_p`. The verifier's own `rec-claim` row `x` is
  `sparse/rep<x>`'s claim round's `c_k` (`RecVStar.claimRow`, through `fixedRow`).
* Unit slots in range: `HoloClass.ofExec` keeps slot `σ`'s units below `2^(k_log − ℓ)`. The sparse phase's `U_σ` reads
  a unit's low `k_log − ℓ` bits (`holoPub`'s `eqAt`), while `RegTable.side` places a template only at blocks the table
  has, so a unit past them would make the two disagree. `holo_runs_sound` takes `hunits`, which `holoAt_units` gives, and
  the verifier refuses units past the block (check 5's `layoutFactors`, `units_of_sparseRange`).
* `sparseError`: `18 + (2ℓ+1)ν + (2ℓ+2)·sLog + pLog + tLog` over `qF` (H2's six rounds are of degree 2, H3's of `2ℓ + 1`
  and `2ℓ + 2`). No list factor: the phase's claims are on `R.bits`, which R's opening decodes uniquely.
* Prover constraint: the verifier refuses a held `sparse/…` stream whose coins by round are not `sparsePhaseRounds`'
  (`RecVStar.checkSparseCoins`, in `checkCovers`), coins at a zero-coin round included, so a prover change to how the
  sparse phases' coins are recorded keeps `rec_live.sparse_rounds` exactly.

