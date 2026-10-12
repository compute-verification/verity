# The end-to-end theorem: the checklist of its gaps

`FlockSoundness/E2E.lean` states the end-to-end theorem as a skeleton:
- `flock_e2e_count`, the integrity profile's count curve;
- `flock_e2e_drawn`, the drawn units;
- `Prog.flock_e2e_count` and `Prog.flock_e2e_drawn`, both on the circuit of the statement's rows (`Prog.circuit`).

**The headline** is `Prog.flock_headline` (`FlockSoundness/Headline.lean`; `ASSUMPTIONS.md`, The headline): the count
curve on `p.circuit outs` with `pp : ProgPlaces`, SHA-512 as the session's hash and the rows' (`flock_e2e_count_hm96`),
at the live draw `Law.execOS` under A3 (`hA3`). It is proved under the assumptions its footprint
lists (`ASSUMPTIONS.md`, The headline), every hypothesis with its kind:
- hardness: `SHA512CRStrict` (`cr/sha-512`, through `hKS` and `hT`) and `SHA512CRExpected` (`ecr/sha-512`, through
  `hCR`);
- platform: A3 (`uniform/io-getrandombytes`, through `hA3`), and A6 (`uniform/flock-coin-server`), the coin server's OS
  randomness, which the model embodies rather than a hypothesis states;
- model: the session's game, against any prover strategy, with A6's round coins;
- intended to prove, in the table below: `pp.placed` (W6), `pp.aliased` (`Layout.Aliased`), `hHm`, `lay` and `rs` (for
  `vb`), `tr` (for `hExec`), `hConst` and `hZero`;
- numeric conditions: `hks`, `hS`, `hk`, `hRw`, `hM`, `hρ`, `hr`, `ht`, and the budgets `qF`, `qS`, `qT`, which Lean
  doesn't check.

It bounds faulty units of the pinned rows circuit; "computes the Definition" additionally needs `RowsCert` per template
and the Definition's Boolean semantics in Lean, which a generic Boolean-semantics lemma replaces once lane proofs-ir
lands. Its round coins are A6's (`uniform/flock-coin-server`), which the model embodies: live uniform OS coins drawn
each round, so it is cited only for runs that use them (`ASSUMPTIONS.md`, its coins).

**What it says.** Suppose the audit accepts a batched session's proofs for the drawn units under the partition. Then,
except with `ε_ks(σ) + δ_link` (plus `miss_L(K)` for the count), the integrity profile holds for the committed circuit
(the circuit the statement's rows make, chosen on the untrusted side; the lowering does no optimizing, Daniel, Sep 30) at
the committed values. Those are public on the constants (1) and on
the statement's zero (0), and the plurality of the prover's openings elsewhere (`Xpub`, `Audit/FlockPublic.lean`).

**What it doesn't say yet.** Two links to the executable are carried by hypotheses, not by the statement:
- "Accept" is `execAccept`, any event on the audit's outcome that implies the model's acceptance. So as pinned, the
  theorems are the model-level ones.
  - `flock-verify`'s verdict on the run's transcript is not an event on that outcome.
  - Discharging `hExec` therefore changes the statement's shape, not only the proof.
- "Each unit's rows derived by the verifier" is not in the generic forms: `unitRows` appears only in `dp`, and `dp.rows`
  says only that a unit's rows don't depend on the draw or the registration. In the `Prog` forms the units' rows are the
  program's (`Prog.rows`), and `ProgPlaces.placed` carries it.
- What a correct unit computes is not a hypothesis: when its rows are `GateRows.rows G L` (`RowsCert`) and `G` computes
  `D`, a unit that is not faulty computes `D` (`IsRowsUnit.computes_of_cert`, `FlockSoundness/Certificate.lean`).

**What it assumes, cryptographically:** `SHA512CRExpected` (A2, `ecr/sha-512`), for the link finders built from the
prover bounded (`hCR : LinkCR … (reg σ) (cont σ) …`, `ASSUMPTIONS.md`), and nothing else: `ε_ks(σ)` is stated at the
prover. The `_hm96` forms state `ε_ks` and the roots' Merkle binding `δ_tree` under `SHA512CRStrict` (`cr/sha-512`), at
their finders' budgets (`hKS`, `hT`; `ASSUMPTIONS.md`, What is restated over the two forms).

**What it composes.** The skeleton rests on theorems already proved on `main`: `flock_batched_count_public` and
`flock_batched_drawn_public`, which carry the compiled audit, the link theorem at the public transcript
(`flock_batched_linkSoundE_pub`, from `flock_batched_linkSoundE`) and knowledge soundness.

**The assumptions we intend to prove.** Each is one of the named hypotheses below, discharged by the PRs listed, and the
skeleton's pins then lose that hypothesis. Each time, Daniel gets a DM showing the rewritten records when they land on
`main`, and that is the check: a statement reviewer is optional, and nothing waits on one. The last three rows are
the hardness and platform assumptions, which stay.

| Hypothesis | What it says | Owner | Discharged by |
|---|---|---|---|
| `hExec` | the audit's acceptance event implies the model's: `∀ o, execAccept o → o.1 = true` | the refinement lane, "Prove verifier refines Lean model" (bc-159ce83b) | its refinement theorem, game-level (R11), which restates the pins with the executable's verdict on the run's transcript |
| `tr : TreeRewind` (in `_hm96`, in place of `hExec`) | a rewinding at the serving roots whose `covers` says the executable accepting while the model at the registered leaves rejects is no likelier than a run opening a serving root off its plurality; it adds `δ_tree = TreeRewind.bound qT` | the refinement lane (R11) | the link from the executable's openings to the model's at the registered roots; `hExec` gives a cover with no positions, at `δ_tree = 0` (`TreeRewind.ofNever`, `ofNever_bound`) |
| `dp : DerivedPlaces` (`pp : ProgPlaces` in the `Prog` forms) | each drawn unit's place in its table (`UnitPlace`), with its rows the verifier's: `(place S R u h).R = unitRows u`. Its input columns on one gate carry one value in every satisfying witness (`UnitPlace.aliased`; `ProgPlaces.aliased`, `Layout.Aliased`). In the `Prog` forms, `ProgPlaces.placed`: the table's statement holds the program's rows, which `placement_of_setupH` gives for the rows `Rows.ofNet` reads off setupH's net; that those are the program's (W6) is open | audit-lean (1d); flock-verifier (1e); flock-soundness (S4) | 1d's `placement_compose` from `setupH`; 1e, the verifier folding `deriveChecked`'s rows; S4, `unitRows u = Rows.compose (deriveChecked …)` for the partition's units. `aliased` from each slot input's copy position and the forced-zero rows as the verifier's Δ writes them, both kinds (the unit's own and the range's first slot's): inputs on one source copy one position, and the zero's inputs copy forced-zero rows. It is sound because Δ has one source per destination, which the verifier's exactly-once checks keep |
| `hConst : ConstCols … ones` | at each drawn unit's place, only the constant's column sits on a gate of `ones`: no unit's input column reads a constant. With it every satisfying message decodes `ones` to 1 (`UnitPlace.decode_one`, `pubDecodes_of_place`), so the committed trace is 1 there | flock-soundness (S4) | for a program of units placed at its tables' classes, `UProg.constCols_of_classes` (proved: each unit's constant column is wired to `one`, `pairs_oneGate`, and an instance shares a gate only between input columns, `constCols_singleton`); for another partition, its places |
| `hZero : ZeroCols … zeros` | at each drawn unit's place, every column on a gate of `zeros` (the statement's zero) carries 0 in every satisfying witness, so the committed trace is 0 there | audit-lean (1d); flock-verifier (1e); flock-soundness (S4) | the forced-zero rows as the verifier's Δ writes them: each slot input that reads the zero copies a row of `TableClass.zeros`, which `TableClass.zero` holds at 0. For a program placed at its tables' classes, `UProg.zeroCols_of_classes` proves it for the program input `zer` (#316's zero) from one more fact of the class data, `hzr`: each slot input that reads `zer` copies a forced-zero row, which 1d and 1e give from Δ. `UProg.flock_e2e_count_classes_zero` and `_drawn_classes_zero` apply it; the `_classes` forms leave the zero at the plurality (`zeros = ∅`) |
| `vb : ValueBinding` | the wires bind to their registered commit strings (already named on `main`) | lean-value-binding; M0's `hm96-sha512` leaf layout | built from the serving rows by `Binding.HmRows.binding` (`FlockSoundness/Binding/`), with `collide_spec` proved (`ExplicitLeaf.hm96`: an explicit SHA-512 collision). `flock_e2e_count_hm96`, `flock_e2e_drawn_hm96` and their `_exec_hm96` forms take, in its place, `HmRowComputes` and the layout's facts (the next row) |
| `hHm : HmRowComputes` (in `_hm96`) | the circuit's row hashing computes `hm96-sha512/row/v1`: in a satisfying witness, the `b ‖ c` a row's `hm96` slot outputs is the commit string of the row and salt its `sha512x3` and `hm96` slots read (`Assumptions.HmRowComputes`, not cryptographic) | the verified-lowering lane (phase 2i) | only with readers of the witness at fixed addresses (`rowAt`, `saltAt`, `bcAt`), a proof from the `sha512x3` and `hm96` sections of the circuit file: each slot's constraints force SHA-512's compression (and hm96's mask and salt digest), and the chain wiring (`HmRow.rowWires`) from the row prefix's midstate gives SHA-512 of the row |
| `lay : Layout`, `rs : HmRows` (in `_hm96`) | the layout's facts: the decoder reads the rows the witness holds (`decode_row`); the `b ‖ c` the witness holds at a row hashes to the registered leaf (`bc_registered`: the statement's regions pin the `hm96` outputs to the public `b ‖ c`, which `HmRow.checkPublic` hashes into the registered frame-v3-sha512 roots); the returned outputs' leaves (`other_registered`) | lean-value-binding, with flock-verifier (S4) | from the placement (`DerivedPlaces`, `UnitPlace`) and the refinement of `HmRow.regions` and `checkPublic` (R11). Discharging `bc_registered` and `other_registered` from registered roots also pays the roots' Merkle binding: `δ_tree`, under `SHA512CRStrict` (`DESIGN.md` §1), which the `_hm96` bounds add (`tr`, `hT`). `decode_row` counts as discharged only with the fixed readers `rowAt`, `saltAt` |
| `hCR` (`SHA512CRExpected`, A2) | SHA-512 expected-time collision resistance for the link finders of the prover bounded, `σ` (`LinkCR … (reg σ) (cont σ) …`), not of every prover at once | the assumption of record | not discharged: a hardness assumption (`ecr/sha-512`). The `FlockSoundness.CROnly` restatements of every form and the headline take `hCRs : LinkCRStrict` (`cr/sha-512`, the same finders stopped at `q`) instead, at `δ_link` with `t'` replaced by `strictRunCost t' q` (`ASSUMPTIONS.md`, What is restated over the two forms) |
| `hKS`, `hT` (`SHA512CRStrict`, in `_hm96`) | SHA-512 strict-time collision resistance for the knowledge term's finders of each table after each draw (`TableCR`, budgets `qF`, `qS`) and for `tr`'s finder (budget `qT`) | the assumption of record | not discharged: a hardness assumption (`cr/sha-512`), the `_hm96` forms' second |
| `hA3` (A3, in the headline) | the verifier's byte source is uniform (`UniformRandomBytes`), so the live draw's count curve is at most `Law.stratified`'s | the assumption of record | not discharged: the headline's platform assumption (`uniform/io-getrandombytes`) |

**Why the constant is public.** The rows multiply outputs by the constant: each output is a copy row `(form) · 1`, and
the constant's row is `1 · 1`. At a trace whose constant is 0, a unit's rows hold with every output 0 while the
Boolean circuit's outputs are the forms. So rows compute their gates only where the constant is 1, and
`computes_of_cert` takes it (`h1`; red-team review of #207).
- **The committed trace is 1 on `ones`.** The plurality of the prover's openings (`Xplur`) can't carry that on its
  own: for a strategy whose trials never read the constant's commit string, `Xval` is an arbitrary argmax. But every
  satisfying message of a drawn unit's table decodes the constant to 1, so the link event never fires on it
  (`linkEvB_of_pub`), and the link bound holds unchanged at `Xpub` for every strategy.
- **One `ones`.** It must hold the gates the units' constants read, for the certificate's `h1` to hold at `Xpub`, and no
  more: a set too large makes `hConst` false (an input column would sit on a gate of it). S4 discharges it for exactly
  the gates the units' constants read.

**Why the zero is different.** The zero multiplies no row, so rows compute their gates at any value of it (#316). But
"faulty at the committed values" judges a unit that reads the zero at the zero's committed value: at a committed zero of
1, a padded attention unit would be judged against `b = 1` (red-team review of #316, C1). At `Xpub` the committed zero
is 0, which `hZero` makes sound: every satisfying message decodes it to 0. So a claim may read the profile as being
about the zero padding when `hZero` is discharged, and not before.

**How to use this list.**
- Each PR that discharges a hypothesis changes the skeleton to prove it: an argument becomes a theorem applied. It also
  deletes that row here. When only `hCR`, `hKS`, `hT` and `hA3` remain, the end-to-end theorem rests on its hardness
  and platform assumptions alone.
- The numeric side conditions (`hk`, `hRw`, `hM`, `hρ`, `hr`, `ht`) are parameters of the bound, not gaps. They are the
  same as in `flock_batched_count_placed`. `ones` and `zeros`, the statement's public gates, are parameters too.
- `ε_ks(σ)` is the knowledge-soundness term (`ksAvgBE`), which `flock_batched_knowledgeSoundE` bounds; under
  `SHA512CRStrict` it is at most `ksAvgStrict` (`StrictCR.ksAvgBE_le_strict`). `δ_link` is
  `linkBoundE`, the link theorem's `Q_s/(1 − ρ)·(2t'(1 + k)/2^256 + 1/(eM) + k/(eR_w))`.

## Before the headline is cited: the legacy SHA-256 and Blake3 paths

The headline is about SHA-512 throughout. These older paths say otherwise (red-team review of the restatement, change
6). The Blake3 row leaf's setup lemmas are retired: `Refine.setup_wf` with its pin, and `blake3_regions_wf`,
`parse_checkLayout`, `checkLayout_slots` and `pin_ok`, which only it used (`Refine/Setup.lean`). Each item below reaches
past its own files, so it is retired by its owner before the headline is cited. Retiring a pinned theorem removes its
pin, a changed record, which Daniel gets as a DM when it lands on `main`.

| Item | What it is | Owner | What it reaches |
|---|---|---|---|
| `Refine/Live.lean` | `liveTable`, one table's live session on the Blake3 unsalted path, and `Game.Sim` (`Sim.prob_le`, pinned) | the refinement lane (bc-159ce83b) | `Refine/LiveSim.lean` (`live_le`, pinned) and `Refine/LiveCompiled.lean` (`live_le_tableC`, pinned) are built on it |
| `Flock/Tags.lean` | the tags with a SHA-256 frame-v3 tree (`frame_v3_tree: sha256`: `circuit19c7269a`, `circuitFd02e847`, `circuit631567f7`) and SHA-256 leaves | flock-verifier | seven agreement sets in `verifier/vectors.json` name them (`lean_statement`), which `check`'s `lean-agreement` replays against the upstream builds `upstream.json` pins and `ci-bundle.sh` builds; `circuitEb90718f` is `circuit631567f7` with changes; `PROTOCOL.md` §16.5 and the verifier's README describe them |
| `verity/flock-tables` | the tables statement in `live/src/tables.rs`, `python/verity_flock/tables.py` and `verity/ml/flock/README.md` | flock-circuit | M0's code, past the Lean package |
| M0's coins | the non-ZK session's default coins come from one seed (`cfg.coin_seed`). Per-round OS coins need `FC_COINS=os`, which the benches set and `flock-verify verify --coins os` accepts; a run on the default stays outside the headline (`ASSUMPTIONS.md`, The headline, its coins) | proofs (the binary's default, `seed_coins()`) | M0's code |
