# Network warden Lean package (`network-timing`)

Definitions, guarantees and proofs for the bucketed warden's timing channel: how much a covert sender inside a
spatial unit X can carry across its boundary ∂X through **when** frames leave, once the declared Program fixes their
content and order. Conventions follow POUS's package (`protocols/pous/lean/`): the same toolchain and Mathlib pin,
`autoImplicit` off, no `axiom` (an assumption is a named `Prop` hypothesis), and a check file.

~~~text
NetTiming/Protocol/Grid.lean   trusted: rows, grids, occupancy, left-packing, fill, and the FIFO warden (queue, released, fifoGrid)
NetTiming/Protocol/Run.lean    trusted: statuses, observations, the warden, the interaction (trace, obsRun, wireRun), the
                               Program, acceptance (Accepted, AcceptedUpTo) and accepted observation sequences (AccSeqs)
NetTiming/Protocol/Ingress.lean  trusted: the interaction read inward, a link's frames with the timing erased (rowFrames,
                               linkFrames), and acceptance as it bears on ingress (StatusComplete)
NetTiming/Assumptions.lean     trusted: the named hypotheses W2–W6 (W4's allocation among them), J, and Theorem 3's two
                               dataflow conditions
NetTiming/Guarantees.lean      trusted: each guarantee as a `def G : Prop` (Proposition 2, Theorem 1(b), Theorem 3, the
                               ingress bound)
NetTiming/SecurityProofs.lean  proved: `SecurityProofs.G : Guarantees.G` for each guarantee, from the lemmas below
NetTiming/SecurityProofs/Occupancy.lean  Proposition 2 (the occupancy counts, fill, and the FIFO warden's tightness)
NetTiming/SecurityProofs/Counting.lean   Theorem 1(b) (the tree lemma on accepted runs, zero error, W6 on accepted runs)
NetTiming/SecurityProofs/Advice.lean     Theorem 3 (the 2^A cap, one clock-sync value per window, the fixed clock, the
                                         block caps)
NetTiming/SecurityProofs/Ingress.lean    the ingress bound (the occupancy of the ingress grid, read from inside X)
NetTiming/SecurityProofs/Sanity.lean     a satisfiability witness for every named hypothesis, the joint witnesses, and the
                                         toy of the September draft
lean-audit.json                the policy and pin records for tools/lean/audit.py
CheckAxioms.lean               one #print axioms line per pin, then the three proved but unpinned lemmas
~~~

~~~bash
lake exe cache get                                          # Mathlib's prebuilt oleans; never build Mathlib from source
lake build                                                  # NetTiming (the default target)
python3 ../../../tools/lean/audit.py .                      # Verity's Lean audit
lake env lean CheckAxioms.lean                              # the axioms of the 39 pins and the 3 unpinned
~~~

**Trusted without proof:** the statements in `NetTiming/Protocol/`, `NetTiming/Assumptions.lean` and
`NetTiming/Guarantees.lean`; Lean's kernel; the Mathlib pin. Everything else is checked by machine. Docstring paths such
as `docs/…` and `internal/…` name the network-transparency project store, where the covert-capacity note
(`docs/network-transparency/timing-channel.md`) and the reviews live.

**The audit.** `tools/lean/audit.py` audits every declaration under `NetTiming`: allowed axioms (only `propext`,
`Classical.choice` and `Quot.sound`), named assumptions, escapes, orphan files, the import layers (`NetTiming.Protocol`,
`NetTiming.Assumptions` and `NetTiming.Guarantees` import only one another and Mathlib, never a proof), the dependency
record, a kernel replay, and the 39 pins' records, whose meaning it follows through those three (`meaning`).
A changed record needs a named statement reviewer (Verity's `AGENTS.md`).

## The model

**The run.** A step is one bucket of the whole boundary ∂X. At each step the operator's move is the ingress, the sender's
move uses the secret, and the wardens record the observation (the egress rows, one jitter class, and every boundary link's
status) and emit the wire. `Interaction.trace` builds the run by recursion; the operator reacts to everything it has
observed.

**Acceptance is separate from emission.** `Accepted P op k o a` holds when, at every step before `k`, the recorded rows are
the Program's rows for the ingress so far and the declared advice, and every status is complete. It reads only rows and
statuses (W5's record). A violation is recorded and the run rejected, with no halt, so every theorem is about accepted
runs, and no violation policy is modelled.

**Ingress is the mirror image** (`NetTiming.Protocol.Ingress`). The same `Interaction` is read inward: the secret is the
outside's message `m`, held by `snd m` (the operator and colluding endpoints, choosing when ingress frames arrive); `op`
is X's devices; the record's rows are what the ingress links release into X; the wire is everything that enters X. The
outside sees every record and can recompute X's moves from them, so it reacts to X's egress. The ingress frames'
content and order are the Program's input, charged as reads; `Guarantees.EncardIngressObsDecodableLe` bounds what
their timing carries into X for a decoder reading the records, and `Guarantees.EncardIngressDecodableLe` for one
reading the wire. Both are proved: on the runs counted, a record sequence is determined by each step's frame count on
each link and its jitter class, since under W4 a row is its frames in slot order and the content fixes which frames
come next. `Guarantees.EncardIngressObsDecodableLeContents` and `Guarantees.EncardIngressDecodableLeContents` drop the
one prescribed content: messages whose frames follow some content assignment in a set `Cs` number at most `#Cs` times
the timing bound, so content plus timing is a theorem.

## The trusted layer (`NetTiming.Assumptions`)

| Note | Lean | Says |
|---|---|---|
| W1 fixed parameters | structural | the links, window ends, frame size, jitter bound and view are arguments fixed before the run; the warden has no secret input |
| W2 regeneration | `Regenerates w view` | the wire is a function of the step and the observation, faults included |
| W3 padding | `Padded w B` | every released frame has length B |
| W4 packing | `Packed w` | every recorded row is left-packed |
| W4 allocation | `Allocated w r` | link `ℓ`'s every recorded row holds at most `r ℓ` real frames |
| W5 untimed status | `UntimedStatus w winEnd` | a status other than complete appears only at a window's last step |
| W6 fail-closed | `FailClosed w winEnd` | after a fault the window's later rows are empty, and a status fails at the window's end |
| J jitter | `JitterBounded w D` | every step's jitter class is below D |
| Theorem 3, condition 1 | `FramesReadNoTimingAdvice P step occ` | a step's rows read the timing advice only through the predicted occupancy |
| Theorem 3, condition 2 | `OccupancyReadsOwnBlock blk occ` | a cell's predicted occupancy reads only its own block's timing advice |

For the ingress bound the same `Prop`s are read inward: `Regenerates` is W2 for what enters X, `Packed` and
`Allocated` are W4 for the rows the ingress links release into X, and `JitterBounded` is J for their release times.

`NetTiming.SecurityProofs.Sanity` shows every hypothesis is satisfiable: one warden meets W2–W6 and J together, one
Program meets both conditions while its timing advice still changes its rows, `joint_witness` meets every hypothesis
of `Guarantees.EncardDecodableLeBlocks` at once, with a reacting operator, and `ingress_witness` meets every hypothesis
of `Guarantees.EncardIngressDecodableLe` at once, with a bound it meets exactly (`ingress_witness_tight`).
`allocated_witness` meets them again with an allocation below the slot count (`r = 1 < S = 2`): its 8 messages meet
that bound exactly, below the 27 of `r = S` (`allocated_witness_tight`), so `Allocated` has content.

## The pins

The 39 records in `lean-audit.json`. The 26 guarantees are pinned as `NetTiming.SecurityProofs.G`, each proved by the
lemma of the same statement whose name it spells in UpperCamelCase (`SecurityProofs.CardFree` by `card_free`):

- **Proposition 2** (`NetTiming.SecurityProofs.Occupancy`): `CardFree`, `CardLeftPacked`, `FillInjective`,
  `ExistsFillOfLeftPacked`, `CardCountsEqLe`, `CardCountsLeLe`, `SumReleasedLe`, `EncardFifoGridsLeOfAvailable`,
  `ReleasedSelf`, `EncardFifoGrids`.
- **Theorem 1(b)** (`NetTiming.SecurityProofs.Counting`): `EncardLeProdOfBranching`, `EncardAcceptedRunsLe`,
  `EncardObsDecodableLe`, `EncardDecodableLe`, `NoFaultOfAccepted`.
- **Theorem 3** (`NetTiming.SecurityProofs.Advice`): `RecordedRowEqFill`, `EncardAccSeqsLeAdvice`, `EncardAccSeqsLeSync`,
  `EncardAccSeqsLeFixedClock`, `EncardAccSeqsLeBlocks`, `EncardAccSeqsLeBlocksCounts`, `EncardDecodableLeBlocks`.
- **The ingress bound** (`NetTiming.SecurityProofs.Ingress`): `EncardIngressObsDecodableLe`, `EncardIngressDecodableLe`,
  `EncardIngressObsDecodableLeContents`, `EncardIngressDecodableLeContents`.

The 13 witnesses are pinned as lemmas, under their own names (`NetTiming.SecurityProofs.Sanity`):
`witnessWarden_assumptions`, `witnessWarden_jitter_varies`, `witnessProgram_conditions`,
`witnessProgram_timing_matters`, `joint_witness`, `joint_witness_bound`, `zP_accepts_one`, `zP_constOp_accepts_one`,
`ingress_witness`, `ingress_witness_tight`, `allocated_witness`, `allocated_witness_tight`, `toy_leftPacked`. Their witness definitions are outside `meaning`, so the audit records each witness's statement but
does not follow it into the definitions it constructs.

**Proved and audited, not pinned:** `occ_fill`, `leftPacked_occ_fill` and `leftPacked_fifoGrid`
(`NetTiming.SecurityProofs.Occupancy`).
The audit checks them like every declaration, and they use only the three axioms. But they have no record in
`lean-audit.json`, so no citation may call them pinned. Pinning them needs a statement reviewer.

The statements were frozen and reviewed by a red team before their proofs, and a second round's verdict was GO. The
ingress bound's statements were frozen before their proofs as well; the review of the first two is on the pull request
that added them (danielreuter/verity#894), and the review of the content versions is on the one stacked on it. The
reference implementation, `protocols/network_warden` (`verity_network_warden`), mirrors 15 of the pins and the three
unpinned lemmas by name in its tests; its `PROTOCOL.md` lists them per test file. The witnesses have no Python
counterpart.

## Modelling choices a statement reviewer checks

1. **Steps are buckets across all of ∂X.** The note states Theorem 1 per link-window. The tree lemma is granularity-free,
   and a per-window count follows by applying it to window chunks; that chunking is not stated.
2. **The Program reads ingress up to and including step t.** The note's W1 lag allows only ingress before t. This is
   more permissive, so the bounds cover more Programs.
3. **Theorem 3 counts all accepted sequences,** not only those a run produces, which makes it stronger and needs no
   interaction.
4. **Jitter** is a free per-step class below D. The operator reads it, since it reads the whole observation.
5. **The integrity term** (wrong proof units) is not modelled: this is the exact binding.
6. **Statuses** cover every boundary link, ingress included, since an ingress status is an egress channel.
7. **Counts** are `Set.encard` in `ℕ∞`, so a bound also says the set is finite.

The ingress bound adds these:

8. **Ingress reuses the interaction, read inward.** The outside's party is `snd`, X's devices are `op`, and both see
   every record. The outside gets every move of X's (recomputed from the records) rather than only the egress wire,
   which is at least as strong as the real outside. X's egress rows are determined by the records too: the ingress
   `Obs` holds only the ingress rows, but X's enqueues are `op`'s moves (functions of the records), the egress warden
   releases them by W4, and an outside fault on an egress link shows as a status, which `Obs` holds. So the outside's
   feedback is complete. A decoder inside X may read what X's devices read: the record version covers it, and the wire
   version follows under W2.
9. **Conditioned on the content.** The messages counted are those whose frames on each link, in the order released, are
   a prefix of one prescribed list `c ℓ` (the Program's input, charged as reads; order is content, since the Program
   reads frames in order). How many frames have arrived by the horizon is timing, and is in the count. Messages that
   also vary the content are bounded by partitioning them by their content: the `Contents` versions count those
   following some assignment in `Cs`, at most `#Cs` × the bound.
10. **Acceptance is the statuses only** (`StatusComplete`). The verifier also checks the egress rows against the
    Program, which only removes runs, so the bound covers every accepted run and more. W5 and W6 are not needed: W2
    and W4 hold for every action, faults included, and complete statuses carry nothing.
11. **One allocation per link, one bucket width.** Link `ℓ` has allocation `r ℓ` (`Allocated`) within the shared slot
    count `S`, and every link shares the step, a bucket of ∂X. Ingress links with different bucket widths are not
    modelled; the per-link-window count `(r_in + 1)^T` follows by reading the product over steps and links.
12. **No Program, no clock.** The bound reads no Program and no advice: the outside sender chooses when frames arrive,
    so no clock pins ingress timing. Jitter on the inward release adds `D^n` (J, `D = 1` when tested ideal). W3's frame
    length and the FIFO rule are not used; the FIFO warden attains the bound (`EncardFifoGrids`, read for the ingress
    grid).
