# Flock interactive soundness in Lean

This directory (namespace `FlockSoundness`) proves Flock's interactive mode sound at the oracle layer, compiled, and as
knowledge soundness, proves zero knowledge for a model of the session's views, and holds the audit layer (`Audit/`). The
verifier it is about is the Lean package `verity/protocols/proofs/flock/verifier/`. The guarantees and the premises each
takes are the lock's (`security/lean-audit.json`); the named assumptions are stated in
`security/Definitions/Assumptions/Flock.lean`, and `ASSUMPTIONS.md` beside this file discusses them.

## The theorems, by stage

| Piece | a16z stage | Lean | Status |
|---|---|---|---|
| Interaction model: live coins, prover strategies, acceptance probability, optimal prover | (framework) | `Game/Basic.lean` | proved |
| A live coin hits a fixed nonzero polynomial's zero with probability at most degree/|F| (Schwartz–Zippel) | (framework) | `Game/Poly.lean` | proved |
| Two reps on independent coins, any interleaving: the error squares | 1a | `Game/Interleave.lean`, `value_interleave_le` | proved |
| The bound as a function of the parameters, and its value for every `m` | 1a | `Accounting/*` | proved |
| The verifier, following the Lean verifier (`Flock.Piop` to `Flock.Statement`) | 1a | `Model/*`, `Definitions/Flock/` | written (definitions) |
| Composition: one table from one rep and the link points | 1a | `table_sound` | proved from `rep_sound` and `link_sound` |
| One rep is sound from a doomed start | 1a | `rep_sound` (`RepSound.lean`) | proved from the four phase lemmas below |
| One sumcheck round, in each of the spec's three encodings | 1a | `Rounds.lean` | proved |
| Zerocheck phase: the eq-reduction (pinned coordinates independent over GF(2)), the skip point, the residual rounds, the pin | 1a | `zerocheck_phase` (`RepSound.lean`), `zerocheck_phase_of_card` (`ZerocheckPhase.lean`) | proved |
| Lincheck phase: `α`, `β`, the rounds, the final check and the skip coin (statements have `k_log ≥ 6`, as `k_skip = 6` requires) | 1a | `lincheck_phase` (`RepSound.lean`), `lincheck_phase_of_card` (`LincheckPhase.lean`) | proved |
| Opening phase: ring switching (seven coins per claim) and the batch | 1a | `opening_phase` (`RepSound.lean`), `opening_phase_of_card` (`OpeningPhase.lean`) | proved |
| The list size: at most `L₀` codewords near the committed table (ArkLib's Johnson bound) | 1a | `closeMsgs_card_le` (`ListSize.lean`) | proved |
| Phase lemma: Ligerito, level by level: the folds (sumcheck and MCA), the next level's OOD binding, the queries and batching coins, the final check | 1a | `ligerito_phase` (`RepSound.lean`), `ligerito_sound` (`LigeritoPhase.lean`) | proved |
| The link points reduce region checks to opened claims | 1a | `link_sound` (`Soundness.lean`), `link_sound_of_card` (`BindSound.lean`) | proved |
| Masking's padded level 0: Ligerito with the padded code and the extra lanes, and the table theorem for any padding `t` and extra lanes `e` | 1a | `ligeritoPad_sound`, `table_sound_pad` (`LigeritoPad*.lean`, `SoundnessPad.lean`) | proved; the inner proof's soundness separately |
| The padded bound at Flock ZK's values: 2^-205 per table at m = 25–35 | 1a | `table_sound_pad_m1` (`Accounting/PaddedNumbers.lean`) | proved |
| Merkle binding: two verified openings of one position with different columns give a SHA-512 collision | 1b | `Merkle.openCollPair_spec` | proved |
| Rewinding: openings off the plurality table cost collisions, `#bad² ≤ N·Q·#conflict` | 1b | `Rewinding.bad_sq_le`, `prBad_le` | proved |
| Compilation step: the oracle bound plus the rewinding term, for a table committed before any coin or after a prefix | 1c | `Compiled.compile_step`, `compile_step_prefix` | proved |
| Session binding: the challenger's live-coin part keeps each round's bytes, so they are fixed before the round's coins | 1c | the `Game` model (`send` before `coin`) | by construction; no hash |
| The compiled table theorem: the compiled game (caps, openings at the end of the proof), the lockstep coupling with the plurality prover, rewinding at every cap | 1c, interactive | `table_sound_compiled` (`Model/Compiled.lean`, `Game/Lock.lean`, `CompiledLock.lean`, `CompiledSound.lean`; the definitions in `Definitions/Flock/Compiled.lean` and `Definitions/Core/Game/Lock.lean`) | proved |
| Knowledge soundness: extract the committed witness, then Halevi–Micali binding to registered data | 1c, 1e | `table_knowledge_sound`, `table_knowledge_sound_joint` (`Knowledge.lean`); `registered_weights` | the table theorems and the link theorem (A2) proved; `registered_weights` proved (`Binding/`) for rows under `HmRowComputes` (named, phase 2i) and the layout's facts; required for sampled audits |
| The block R1CS means the named circuit (the lowering theorem) | 1d | level 3; its generic step `Lowering.lean`, the placement `ExecSetup.lean`, the derived rows' placement `Compose.lean` | the generic step and the placement from `Stmt.setupH` proved; for derived rows, the placement from the block matrices proved; the link from `setupH`'s unit net to the derived rows remains |
| Zero knowledge of the masked model: Lemma B per table and per session, Lemma A at every coin vector (the prover's refusals in the code's order), Lemma C against an adaptive verifier with its own tape | 1e | `table_shvzk_hm96`, `session_shvzk_hm96`, `session_prefinal_indep_cut`, `adaptive_prefinal_hm96_tape` (`ZK/`) | proved for the model `ZK/Table.lean`; Theorem GK over a model of the session's views, `gk_simulate_hm96` (`ZK/GK/`); the link to the code remains |
| Sampled audits: the integrity profile over any partition and sampling design, minimax and tightness | (composition) | `Audit/*` | proved; per-proof-block terms named |
| The Flock audit at the oracle layer: the batched session's projection to one table, `2^-205` (the audit layer) | 1a, composition | `Audit/Flock.lean`, `flock_session_sound` | proved, given the lowering |
| The compiled lowering: the audit's circuit is the rows the verifier parses, and a satisfied table makes every placed unit correct, for every template | 1a | `Lowering.lean`, `lowering_sound`, `Prog.isRowsUnit`; `placement_of_setupH` | proved; the placement from the executable's accept step |
| The certificate: rows that are `GateRows.rows G` compute `G`, so a correct unit computes the definition `G` lowers (`ASSUMPTIONS.md`; no hypothesis, since the circuit is the statement's rows) | (meaning) | `GateRows.rows_sound` (`GateRows.lean`) | proved for the rows; not stated for a unit's committed values; the premises per template: the lowering comparison tests on samples, and the typed layout's proof (`Types.Dag.layout_sound`, `TmplValue.defComputes_of_layout`) once a template's rows are a checked layout's (`assumption-notes/template-rows.md`) |

The proved results depend only on Lean's three standard axioms (`propext`, `Classical.choice`, `Quot.sound`). Run
`python tools/verity/lean/check.py security/Proofs` to check every declaration (`lean-audit.json`).
