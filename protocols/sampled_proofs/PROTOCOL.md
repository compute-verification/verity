# Sampled proofs

Spot-checks a committed transcript of a Verity Program, and says what an accepted transcript still admits.

## Lifecycle

1. The prover commits the transcript's boundary (the replay units' interface values) and registers the root.
2. The RU draw, after the root's registration: the verifier's own randomness (`law.OwnRandomness`). Each replay unit (RU) of a
   class is selected with its stage's probability `p` (`select_replay_units`).
3. The prover replays the selected RUs, then commits and registers their interiors.
4. The VU draw, after the interiors' registration: the verifier's own randomness again. In each selected RU,
   exactly `k` of its verification units (VUs) are checked (`select_verification_units`), or all of them when it
   holds fewer.
5. Each checked VU is established on its authenticated boundary values.
6. An accepted audit emits a `verity.proofs.profile.IntegrityProfile` per RU class (`TwoStageLaw.profile`). It
   holds the exhaustively checked facts: the evaluated Programs, the prescribed families, linkage, and the families
   covered on every audit (`VerificationPlan.covered`). It also holds the law as two named levels, `replay`
   (the `n_r` RUs, drawn `Bernoulli(p)`) and `check` (the `n_v` VUs in each, drawn `Subset(k)`), the counts
   drawn at each, and `delta`. Resource accounting, policies and PoUW each read what they need off it, passing their own
   utility to `worst_case`. The application counts `delta` once.

## Randomness

The verifier draws every coin from its own randomness, directly: `law.OwnRandomness` reads the operating system's
generator with the exact samplers of the Lean verifier's `Flock.Draw` (uniform by rejection on big-endian words,
Bernoulli as a uniform below the numerator, subset by partial Fisher-Yates), with no hash and no derivation. There is no
beacon. Each draw comes after the registration it follows: the RU draw after the boundary, the VU draw after the
interiors. Draws are addressed by RU index in the committed order.

`law.ru_key` and `law.vu_key` keep decision 42's `verity.randomness.derive` of a caller's source and the bindings only for
vLLM's LEGACY challenge, until its re-baseline. They are not a draw of record.

## The law

The law is `TwoStageLaw({class: Stage(p, k)})` (decision 43). A Bernoulli law over single VUs is the special case
where every VU is its own RU with `k = 1` (`TwoStageLaw.bernoulli`). A gate-level `plan.VerificationPlan` is that
case: every gate is its own RU.

An RU with `m` incorrect VUs out of `n_v` passes with probability
`1 - p + p * C(n_v - m, k) / C(n_v, k)` (`IntegrityProfile.accept`). `tests/test_law.py` checks the sampler
against this exactly.

## Boundaries

Obligations are built from a call's declared interface (`args`, `returns`). The protocol never reads
`FunctionCall.out`, which rescans the circuit on every call. `tests/test_plan.py` enforces this.

## Not here yet

- **The audit driver:** steps 1 to 6 as one object, with the registration log and the verifier's draws. The prototype's
  version is on `cursor/protocol-prototype-9816`. It lands with the first real consumer of the full lifecycle. Today
  that is vLLM, whose replay runs on one post-commit challenge, with its strata as RUs at `p = 1`.
- **The security function** and the red-team review (decision 41).
- **Composition checks:** for example, PoUW's tiles per VU must match this partition (item 26). The IR can check
  that one level refines another and count members per parent (`verity.ir.partition.validate_refinement`); wiring
  it to PoUW's levels waits for PoUW.
