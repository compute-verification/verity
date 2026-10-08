# core_composition: onsite and remote Verity as core protocols

Three Lake packages, each on C-Flock's verifier and the core interface (`../interface`) by path, with no Mathlib:

- `core_composition` (here, lib `CoreComposition`, namespace `Core.Composition`): `product id components`, the
  `Core.Protocol` of a composition, with its verdict rule, its formats and the commitment's binding (`Product.lean`); the
  stub components (`Stubs.lean`); and the `#guard`s that pin the rule (`Guards.lean`).
- `core_onsite` (`../onsite`, `Core.Onsite.protocol`, `verity/onsite/v0`): integrity, work, space and communication.
- `core_remote` (`../remote`, `Core.Remote.protocol`, `verity/remote/v0`): integrity and work.

The rule and the formats are written once, over any list of components, and the two products are values
of it that differ in their id and components. Each product's executable is the core CLI (`onsite id | check | serve`,
`remote id | check | serve`), and its `serve` is what the difftest drives.

## The protocol

| input | format |
| --- | --- |
| params | `{"components":[{"id":ID,"params":PARAMS},...]}`: exactly the product's components, in the order of their ids |
| commitment | 64 bytes: SHA-512 of the product's id, a zero byte, and `{"components":[{"commitment":HEX,"id":ID},...]}`, the components' digests in order (`bind`); the product's own digest |
| evidence | `{"components":[{"commitment":HEX,"evidence":HEX,"id":ID,"params":PARAMS},...]}`: each component's case (what its own `serve` reads), under its id, in the same order |

All three are canonical JSON or bytes, and an object with a key missing or a key more is refused. The check runs each
component's `answer` on its case, then decides, reject first:

| the components' answers | onsite / remote |
| --- | --- |
| any rejects, whatever the others answer | reject |
| the cases don't fit (a component unknown, missing, repeated or out of order, or a case under other params) | reject |
| none rejects, and the commitment isn't the binding of their digests | reject |
| none rejects, the commitment binds them, and some is undecided | undecided |
| every one accepts, and the commitment binds them | accept |

On acceptance the product makes public each component's public outputs and keeps each one's profile, both as JSON objects
keyed by the component's id. On rejection or undecided nothing is public.

## Components

Until their ports land, each component is a stub, `Core.Example.protocol` under an id that says so:
`verity/stub/sampled-proofs/v0` (integrity, sampled proofs' one-stage audit), `verity/stub/pouw/v0` (work),
`verity/stub/pous/v0` (space) and `verity/stub/certifier/v0` (communication, the network certifier). A port replaces its
stub in `Core.Onsite.components` and `Core.Remote.components` with its own `Core.Protocol` value, keeping each list in the
order of its ids (`ordered`, which each product's `#guard` checks). NCI and the trial's reconciling are not components yet.

## The difftest

`uv run tools/verity/check/suites.py verity-core-composition` builds `onsite` and `remote` and drives each one's `serve`
through the harness's `ProtocolPort` (`../../difftest`), holding every field of each outcome but its reason to
`tests/reference.py`: the trial's composition rule from #1036 (`cursor/protocol-interfaces-c3b2` at 46820ebcb, vendored,
with its source files named), over the harness's framing of the core interface (`verity_difftest.protocol.answer`) and a
mirror of the stub. Hypothesis draws 1000 cases a product (seed 0): each component's case accepts, rejects, is undecided or
fails to decode, and the composition's inputs or the case itself are often broken. The port follows top's rule where the
trial's differs, and the reference is changed to match (`verdict`; the trial's, reduced to answers, is `trial_verdict`):

- **Reject dominates.** In the trial a missing statement leaves the product with no verdict even when another component's
  statement has failed, so a prover could hide a rejection by leaving another component undecided. Here any rejection
  rejects. `test_the_rule_changes_the_trial_only_where_a_rejection_meets_an_undecided_component` pins the difference,
  and `test_a_reference_without_reject_dominance_is_caught` shows the difftest tells the two apart: a reference with the
  trial's rule disagrees with the Lean, on a case where one component rejects and another is undecided.
- **Every component counts.** Of the components' statements, the trial's NCI fails the epoch on work's reasons alone
  (integrity's reach it through them), and space's or communication's reasons only zero their explained usage in the
  balance. Here each component's rejection rejects; `trial_verdict` already counts every component's.

A bare `uv run` inside this directory stops its workspace search at `verity/core/protocols/pyproject.toml`; run the suite
through `suites.py`, or `pytest` with the workspace's `.venv`.
