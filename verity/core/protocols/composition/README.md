# core_composition: onsite and remote Verity as core protocols

Two Lake packages, each on C-Flock's verifier and the core interface (`../interface`) by path, onsite also on its
subprotocols' packages, with no Mathlib:

- `core_composition` (here, lib `CoreComposition`, namespace `Core.Composition`): `product id subprotocols`, the
  `Core.Protocol` of a composition, with its verdict rule, its formats and the commitment's binding (`Product.lean`);
  the stub subprotocols (`Stubs.lean`); and the `#guard`s that pin the rule (`Guards.lean`).
- `core_onsite` (`../onsite`, `CoreOnsite`): both composites, built by one constructor, `Core.Onsite.restrict id ids`,
  the product over onsite's subprotocols restricted to those `ids` lists:
  - onsite (`Core.Onsite.protocol`, `verity/onsite/v0`): sampled proofs, PoUW, PoUS, the remote bandwidth audit and the
    network certifier;
  - remote (`Core.Onsite.remote`, `verity/remote/v0`): onsite restricted to sampled proofs and PoUW.

The rule and the formats are written once, over any list of subprotocols, and the two composites are values of
`restrict` that differ in their id and subprotocols. These two are the only composites: `Core.Onsite.ofIds` names the
one a list of subprotocols makes and refuses any other set, since protocol soundness covers only these two. Each
composite's executable is the core CLI (`onsite id | check | serve`, `remote id | check | serve`), and its `serve` is
what the difftest drives; `product IDS id | check | serve` is the core CLI of `ofIds IDS`, which `verity run`'s verifier
runs over a configuration's subprotocols.

## The protocol

| input | format |
| --- | --- |
| params | `{"components":[{"id":ID,"params":PARAMS},...]}`: exactly the product's subprotocols, in the order of their ids |
| commitment | 64 bytes: SHA-512 of the product's id, a zero byte, and `{"components":[{"commitment":HEX,"id":ID},...]}`, the subprotocols' digests in order (`bind`); the product's own digest |
| evidence | `{"components":[{"commitment":HEX,"evidence":HEX,"id":ID,"params":PARAMS},...]}`: each subprotocol's case (what its own `serve` reads), under its id, in the same order |

All three are canonical JSON or bytes, and an object with a key missing or a key more is refused. The check runs each
subprotocol's `answer` on its case, then decides, reject first:

| the subprotocols' answers | onsite / remote |
| --- | --- |
| any rejects, whatever the others answer | reject |
| the cases don't fit (a subprotocol unknown, missing, repeated or out of order, or a case under other params) | reject |
| none rejects, and the commitment isn't the binding of their digests | reject |
| none rejects, the commitment binds them, and some is undecided | undecided |
| every one accepts, and the commitment binds them | accept |

On acceptance the composite makes public each subprotocol's public outputs and keeps each one's profile, both as JSON
objects keyed by the subprotocol's id. On rejection or undecided nothing is public.

`Properties.Composition` (`Security/Properties/Composition.lean`, proved in `Security/Proofs/Properties/Composition.lean`)
states the accept row against this code. `answerAccepts` covers every `Core.Protocol`: what `answer` accepts decodes,
and its check accepts it. `acceptMeans` covers `Core.Composition.check`: what it accepts fits the composition, every
subprotocol accepts its own case, and the commitment is `bind` of the subprotocols' digests.

## Subprotocols

| subprotocol | onsite | remote | `Core.Protocol` | id |
| --- | --- | --- | --- | --- |
| the network certifier (over the period: its sampling blocks of transfer blocks certified and proved, and the bound `K` on what left the site) | yes | | `NetworkCheck.Period.protocol` (`network_check`, `../network/lean`) | `verity/certifier/v0` |
| PoUW (its window check) | yes | yes | `PouwCheck.protocol` (`pouw_check`, `../pouw/lean/check`) | `verity/pouw/v1` |
| sampled proofs (the one-stage audit) | yes | yes | `SampledProofs.OneStage.protocol` (`core_sampled_proofs`, `../../service/sampled_proofs`) | `verity/sampled-proofs/one-stage/v1` |
| PoUS (its check of a run) | yes | | `PousCheck.RunProtocol.protocol` (`pous_check`, `../pous/check`) | `verity/pous/p2/v4/run` |
| the remote bandwidth audit (its session) | yes | | `RemoteAudit.Protocol.protocol` (`remote_audit`, `../remote_bandwidth/lean`) | `verity/remote-bandwidth/v0` |

Until its port lands, a subprotocol is a stub, `Core.Example.protocol` under an id that says so (`Stubs.lean`). A port
replaces its stub in `Core.Onsite.subprotocols` with its own `Core.Protocol` value, its package a `[[require]]` of
`core_onsite`, keeping the list in the order of its ids (`ordered`, which `CoreOnsite`'s `#guard` checks), and deletes
the stub. The difftest's reference (`tests/reference.py`) replaces the stub's mirror with the port's
own Python (`Ported`), or its Python's answers on cases made once (`Recorded`), and its cases come from the port's
own difftest or its period hook. Every subprotocol of both composites is ported; NCI and the trial's reconciling are not
subprotocols yet.

## The difftest

`uv run tools/verity/check/suites.py verity-core-composition` builds `onsite` and `remote` and drives each one's `serve`
through the harness's `ProtocolPort` (`../../difftest`), holding every field of each outcome but its reason to
`tests/reference.py`: the trial's composition rule, onsite's and remote's over their subprotocols' statements, over the harness's framing
of the core interface (`verity_difftest.protocol.answer`), and
each subprotocol's own Python (PoUW's `core_check.case`; sampled proofs' one-stage difftest's `answer`, `one_stage` behind
the protocol's decoders; the network certifier's period difftest's `reference`, its Python mirror over the certifier's
certificates and one-stage's answers, on small signed periods from its `build`), which that subprotocol's own difftest
holds its Lean to; for PoUS and the remote bandwidth audit, their Python verifiers' answers on small periods of their hooks
(`tests/periods.py`: `pous.period.run` signed at S1's deployed parameters, honest and cheating, and
`remote_bandwidth.period.run`, honest, cheating and unverified), whose Lean their packages' difftests hold to those
verifiers (`../pous/check/tests/test_run_protocol.py`, `../remote_bandwidth/lean/tests/test_protocol.py`).
Hypothesis draws 1000 cases a product (seed 0): each subprotocol's case accepts, rejects, is undecided or fails to decode
(one draw in six every one aimed at acceptance, one in six all but one, which is aimed at undecided), and the
composition's inputs or the case itself are often broken. The port follows the rule above where the
trial's differs, and the reference is changed to match (`verdict`; the trial's, reduced to answers, is `trial_verdict`):

- **Reject dominates.** In the trial a missing statement leaves the product with no verdict even when another subprotocol's
  statement has failed, so a prover could hide a rejection by leaving another subprotocol undecided. Here any rejection
  rejects. `test_the_rule_changes_the_trial_only_where_a_rejection_meets_an_undecided_component` pins the difference,
  and `test_a_reference_without_reject_dominance_is_caught` shows the difftest tells the two apart: a reference with the
  trial's rule disagrees with the Lean, on a case where one subprotocol rejects and another is undecided.
- **Every subprotocol counts.** Of the subprotocols' statements, the trial's NCI fails the epoch on PoUW's reasons alone
  (sampled proofs' reach it through them), and PoUS's or the network certifier's reasons only zero their explained usage in the
  balance. Here each subprotocol's rejection rejects; `trial_verdict` already counts every subprotocol's.

A bare `uv run` inside this directory stops its workspace search at `verity/core/protocols/pyproject.toml`; run the suite
through `suites.py`, or `pytest` with the workspace's `.venv`.
