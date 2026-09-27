# Sampled proofs

Spot-checks a committed transcript of a Verity Program, and says what an accepted transcript still admits.

## Lifecycle

1. The prover commits the transcript's boundary (the replay units' interface values) and registers the root.
2. The RU draw: `law.ru_key(beacon(t), bindings)`, where `beacon(t)` is the first round published after the root's
   registration time `t`, and `bindings` holds the Program digest and the boundary root. Each replay unit (RU) of a
   class is selected with its stage's probability `p` (`select_replay_units`).
3. The prover replays the selected RUs, then commits and registers their interiors.
4. The VU draw: `law.vu_key(beacon(t'), bindings)`, which also binds the interiors root. In each selected RU,
   exactly `k` of its verification units (VUs) are checked (`select_verification_units`), or all of them when it
   holds fewer.
5. Each checked VU is established on its authenticated boundary values.
6. An accepted audit emits a `verity.proofs.profile.IntegrityProfile` per RU class (`TwoStageLaw.profile`). It
   holds the exhaustively checked facts: the evaluated Programs, the prescribed families, linkage, and the families
   covered on every audit (`VerificationPlan.covered`). Its law is one level, `replay` (the `n_r` RUs), drawn
   `Bernoulli(p * k / n_v)`, with the count drawn and `delta`; it counts wrong RUs (see "The law"). Resource
   accounting, policies and PoUW each read what they need off it, passing their own utility to `worst_case` (work in VUs
   charges `n_v` per wrong RU). The application counts `delta` once.

## Randomness

Every coin is drawn through `verity.randomness.derive(source, domain, context)` (decision 42). The source is a beacon
round or a key holder's secret, never the prover's own root. The domains are `verity/sampled-proofs/ru-sample/v1`
and `verity/sampled-proofs/vu-sample/v1`. Draws are addressed by RU index in the committed order, so a shard of the
population draws its own RUs.

## The law

The law is `TwoStageLaw({class: Stage(p, k)})` (decision 43). A Bernoulli law over single VUs is the special case
where every VU is its own RU with `k = 1` (`TwoStageLaw.bernoulli`). A gate-level `plan.VerificationPlan` is that
case: every gate is its own RU.

An RU with `m` incorrect VUs out of `n_v` passes with probability `1 - p + p * C(n_v - m, k) / C(n_v, k)`. That is
not the profile's law: the interiors are committed after the RU draw, so the prover chooses `m` after seeing which RUs
were drawn, and `m = 1` is always enough to leave a skipped RU wrong. A wrong RU therefore escapes with probability
at most `1 - p * k / n_v`, independently across RUs, and the profile is over the RUs with that rate
(`docs/audit-protocols.md` §3.3, pitfall 6). `tests/test_law.py` checks the sampler against the per-RU formula and
reproduces the attack the old two-level profile missed (§2.9's `TOY`: 64 skipped VUs pass with probability 0.01).

## Boundaries

Obligations are built from a call's declared interface (`args`, `returns`). The protocol never reads
`FunctionCall.out`, which rescans the circuit on every call. `tests/test_plan.py` enforces this.

## Not here yet

- **The audit driver:** steps 1 to 6 as one object, with the registration log and the beacon client. The prototype's
  version is on `cursor/protocol-prototype-9816`. It lands with the first real consumer of the full lifecycle. Today
  that is vLLM, whose replay runs on one post-commit challenge, with its strata as RUs at `p = 1`.
- **The security function** and the red-team review (decision 41).
- **Composition checks:** for example, PoUW's tiles per VU must match this partition (item 26). The IR can check
  that one level refines another and count members per parent (`verity.ir.partition.validate_refinement`); wiring
  it to PoUW's levels waits for PoUW.
