# Network attestation Lean (the `Certifier` area)

Definitions, guarantees and proofs for the network attester's timing channel: how much a covert sender inside a
spatial unit X can carry across its boundary ∂X through **when** frames leave, once the declared Program fixes their
content and order. The statements are `../../Definitions/Certifier/`, the proofs this directory, and each guarantee's
record `../../lean-audit.json`. The network attester's program, which the difftests run, is
`verity/protocols/resource_accounting/network/attester/program/lean/`; `Code.lean` proves it equal to the definitions.

~~~bash
lake exe cache get                         # in security/: Mathlib's prebuilt oleans; never build Mathlib from source
lake build                                 # in security/, then security/Properties/ and security/Proofs/ (this area's proofs)
python3 tools/verity/lean/check.py security/   # from the repository root: Verity's Lean check
~~~

**Trusted without proof:** the statements in `../../Definitions/Certifier/`, the guarantees' modules among them;
Lean's kernel; the Mathlib pin.

## The trusted layer (`NetTiming.Assumptions`)

| Note | Lean | Says |
|---|---|---|
| W1 fixed parameters | structural | the links, execution phase ends, frame size, jitter bound and view are arguments fixed before the run; the network attester has no secret input |
| W2 regeneration | `Regenerates w view` | the wire is a function of the step and the observation, faults included |
| W3 padding | `Padded w B` | every released frame has length B |
| W4 packing | `Packed w` | every recorded row is left-packed |
| W4 allocation | `Allocated w r` | link `ℓ`'s every recorded row holds at most `r ℓ` real frames |
| W5 untimed status | `UntimedStatus w winEnd` | a status other than complete appears only at an execution phase's last step |
| W6 fail-closed | `FailClosed w winEnd` | after a fault the execution phase's later rows are empty, and a status fails at its end |
| J jitter | `JitterBounded w D` | every step's jitter class is below D |

For the ingress bound the same `Prop`s are read inward: `Regenerates` is W2 for what enters X, `Packed` and
`Allocated` are W4 for the rows the ingress links release into X, and `JitterBounded` is J for their release times.

## Modelling choices a reader of the statements checks

1. **Steps are buckets across all of ∂X.** The guarantees count steps, and Theorem 1(b)'s tree lemma is stated per
   step. The per-execution-phase ones read a monitoring period of whole blocks, each with its parent's hypotheses at
   that length and no other. The ingress bound is `(∏_ℓ (r ℓ + 1)^T · D^T)^m` per content assignment over the first
   `m` periods of `T` buckets, `n = m·T` (`EncardIngressObsDecodableLePerExecutionPhase`, and its `Contents` version);
   `Packed`, `Allocated` and `JitterBounded` bound every step's branching whatever came before.
2. **The Program reads ingress up to and including step t.** W1's lag allows only ingress before t. This is more
   permissive, so the bounds cover more Programs.
3. **Theorem 3 counts all accepted sequences,** not only those a run produces, which makes it stronger and needs no
   interaction.
4. **Jitter** is a free per-step class below D. The operator reads it, since it reads the whole observation.
5. **The integrity term** (incorrect proof blocks) is not modelled: this is the exact binding.
6. **Statuses** cover every boundary link, ingress included, since an ingress status is an egress channel.
7. **Counts** are `Set.encard` in `ℕ∞`, so a bound also says the set is finite.

The ingress bound adds these:

8. **Ingress reuses the interaction, read inward.** The outside's party is `snd`, X's devices are `op`, and both see
   every record. The outside gets every move of X's (recomputed from the records) rather than only the egress wire,
   which is at least as strong as the real outside. X's egress rows are determined by the records too: the ingress
   `Obs` holds only the ingress rows, but X's enqueues are `op`'s moves (functions of the records), the egress network
   attester releases them by W4, and an outside fault on an egress link shows as a status, which `Obs` holds. So the
   outside's feedback is complete. A decoder inside X may read what X's devices read: the record version covers it, and
   the wire version follows under W2. Read inward, W2 (`Regenerates`) also covers whatever outside-influenced thing X
   can see of its own egress (backpressure, release timing): it is all a function of the record.
9. **Conditioned on the content.** The messages counted are those whose frames on each link, in the order released, are
   a prefix of one prescribed list `c ℓ` (the Program's input, charged as reads; order is content, since the Program
   reads frames in order). How many frames have arrived by the last step counted is timing, and is in the count.
   Messages that also vary the content are bounded by partitioning them by their content: the `Contents` versions count
   those following some assignment in `Cs`, at most `#Cs` × the bound.
10. **Acceptance is the statuses only** (`StatusComplete`). The verifier also checks the egress rows against the
    Program, which only removes runs, so the bound covers every accepted run and more. W5 and W6 are not needed: W2
    and W4 hold for every action, faults included, and complete statuses carry nothing.
11. **One allocation per link, one bucket width.** Link `ℓ` has allocation `r ℓ` (`Allocated`) within the shared slot
    count `S`, and every link shares the step, a bucket of ∂X. Ingress links with different bucket widths are not
    modelled. Over the first `m` whole execution phases each contributes `(r ℓ + 1)^T`
    (`EncardIngressObsDecodableLePerExecutionPhase`).
12. **No Program, no clock.** The bound reads no Program and no advice: the outside sender chooses when frames arrive,
    so no clock pins ingress timing. Jitter on the inward release adds `D^n` (J, `D = 1` when tested ideal). W3's frame
    length and the FIFO rule are not used; the FIFO network attester attains the bound (`encard_fifoGrids`, read for
    the ingress grid).

The constant-rate Program adds these:

13. **Send ticks, one FIFO per execution block.** `NetTiming.Protocol.Schedule` takes each frame's send tick as the
    Program gives it: a frame is handed at `max(send, last)`, so hand ticks only rise in canonical order, and the FIFO
    starts empty at each execution block's first bucket, a frame handed at or after its span's end never enqueued. The
    network attester's program (`NetworkAttester.Schedule`, `schedule.py`) computes its hand ticks from a declared
    clock sync per execution phase, and over its `W` execution phases it is one execution block (`CodeExpectedRow`).
    The audit checks `E` execution blocks, so the counts take `n ≤ E·g·T` (`hn`): every step they count is a step the
    audit checks.
14. **The frames are the Program's computation, left abstract.** `frames ins e v ℓ` gives link `ℓ`'s frames of
    execution block `e` in canonical order, with their content and send ticks, from the ingress and `e`'s advice `v`.
    Every guarantee holds for every such function, so it covers any way the Program fixes send ticks. Advice that moves
    a send tick is counted in its block's `k` bits.
15. **One clock per spatial unit.** Every egress link reads the same execution phase's sync, and the audit enforces it
    (`Audit.syncs`). A clock per link is a different advice type, covered by `EncardAccSeqsLeAdvice` directly.
16. **The allocation is the slot count `S`** on every egress link, as in Python, where a row has `allocation` slots.
17. **In-set advice.** The audit's advice ranges over the values of Σ_sync: a run whose declared sync is outside the
    set is an `advice` violation and rejected, so its advice is never counted. That an earlier execution phase's grid
    doesn't depend on a later one's sync, in the set or not, has no Lean statement; the rest of an execution phase's
    verdict reads only its own records and syncs.
