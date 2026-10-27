# core_composition: onsite and remote Verity as core protocols

Three Lake packages, each on C-Flock's verifier and the core interface (`../interface`) by path, onsite and remote also on
their ported components' packages, with no Mathlib:

- `core_composition` (here, lib `CoreComposition`, namespace `Core.Composition`): `product id components`, the
  `Core.Protocol` of a composition, with its verdict rule, its formats and the commitment's binding (`Product.lean`);
  `registered id R s components`, the same after a registration check `R`, on the registration the component `s`
  commits to (`Registered.lean`); both over blobs (`Blobs.lean`); the stub components (`Stubs.lean`); and the `#guard`s
  that pin both rules (`Guards.lean`).
- `core_onsite` (`../onsite`, `Core.Onsite.protocol`, `verity/onsite/v0`): integrity, work, space, bandwidth and
  communication.
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

| component | onsite | remote | `Core.Protocol` | id |
| --- | --- | --- | --- | --- |
| communication (the network certifier over the period: its link-windows certified and proved, and the bound `K` on what left the site) | yes | | `NetworkCheck.Period.protocol` (`network_check`, `../network/lean`) | `verity/certifier/v0` |
| work (PoUW's window check) | yes | yes | `PouwCheck.protocol` (`pouw_check`, `../pouw/lean/check`) | `verity/pouw/v1` |
| sampled proofs (the audit on the proofs themselves, which it reads as blobs: one-stage v2 or the recursive audit, as the params' `form` names it) | yes | yes | `VStar.Recursive.Sampled.protocol` (`core_vstar_recursive`, `../../service/vstar/recursive`) | `verity/sampled-proofs/v1` |
| space (PoUS's check of a run) | yes | | `PousCheck.RunProtocol.protocol` (`pous_check`, `../pous/check`) | `verity/pous/p2/v4/run` |
| bandwidth (the remote bandwidth session) | yes | | `RemoteAudit.Protocol.protocol` (`remote_audit`, `../remote_bandwidth/lean`) | `verity/remote-bandwidth/v0` |
| the wipe (the memory challenger's streamed-label wipe of the period) | yes | | `MemoryChallenger.WipeProtocol.protocol` (`core_memory_challenger`, `../../devices/memory_challenger/lean`) | `verity/wipe/v0` |

Until its port lands, a component is a stub, `Core.Example.protocol` under an id that says so (`Stubs.lean`). A port
replaces its stub in `Core.Onsite.components` and `Core.Remote.components` with its own `Core.Protocol` value, its package
a `[[require]]` of each product, keeping each list in the order of its ids (`ordered`, which each product's `#guard`
checks), and deletes the stub. The difftest's reference (`tests/reference.py`) replaces the stub's mirror with the port's
own Python (`Ported`), or its Python's answers on cases made once (`Recorded`), and its cases come from the port's
own difftest or its period hook. Every component of both products is ported; NCI is a registration check, not a component, and
the trial's reconciling is not a component yet.

A component that reads blobs (`Core.BlobProtocol`, the content-addressed files beside its three byte strings: integrity's
proof files) makes the product one too: `productB` (`CoreComposition/Blobs.lean`) hands the caller's blobs to every
component, those that read none ignore them (`Protocol.withoutBlobs`), and on the blobs it is `product` of its
components on them (`productB_on`). `onsite check P C E [BLOB ...]` reads them after the evidence, and a `serve` case
carries them as hex under `blobs`.

## A registration check

`registered id R s components` checks a registration before composing, in its commit phase, and `registeredB` is the
same over blobs. Onsite is one, with NCI's check of record as `R` (`NciCheck.Protocol.protocol`, `verity/nci/v1`):
`registeredB "verity/onsite/v0" NciCheck.Protocol.protocol
VStar.Recursive.Sampled.protocol.id components`, whose registration is the one sampled proofs commits to (in either form,
v1's registration record), one registration digest in both. NCI is a registration check, not a component (Daniel's ruling: onsite has six
subprotocols, and NCI is none of them). Its formats are the product's with the registration's beside the components':

| input | format |
| --- | --- |
| params | `{"components":[...],"registration":{"id":ID,"params":PARAMS}}` |
| commitment | the product's 64 bytes, then the registration's digest; its own digest |
| evidence | `{"components":[...],"registration":{"commitment":HEX,"evidence":HEX,"id":ID,"params":PARAMS}}` |

It rejects when the registration isn't the registration check's or carries other params, when the registration check
doesn't accept it, or when the commitment doesn't end in its digest, whatever the components answer. When the cases fit
it rejects too when the committing component's case (the first under the id `s`) doesn't commit to that digest, again
whatever the components answer (`mismatch`: `registration: R checked D, s committed D'`, `D'` hex or `no digest`).
Otherwise it answers as the product of the components does under the commitment's first 64 bytes, and the registration
adds nothing to what is public or kept.

## The difftest

`uv run tools/verity/check/suites.py verity-core-composition` builds `onsite` and `remote` and drives each one's `serve`
through the harness's `ProtocolPort` (`../../difftest`), holding every field of each outcome but its reason to
`tests/reference.py`: the trial's composition rule from #1036 (`cursor/protocol-interfaces-c3b2` at 46820ebcb, vendored,
with its source files named), over the harness's framing of the core interface (`verity_difftest.protocol.answer`), and
each component's own Python (PoUW's `core_check.case`; for sampled proofs, its form's split (`sampled_proofs`) over v2's framing
over sampled proofs' one-stage difftest's `one_stage` and decoders, `one_stage_zk`, on the cases v2 decides without a
session's proof, the recursive form refused where the case names none of its files; the network certifier's period difftest's `reference`, its Python mirror over the certifier's
certificates and one-stage's answers, on small signed periods from its `build`; the wipe's mirror,
`wipe_protocol_case.answer`, on signed wipes from its `case`), which that component's own difftest holds its Lean to;
for space and bandwidth, their Python verifiers' answers on small periods of their hooks
(`tests/periods.py`: `pous.period.run` signed at S1's deployed parameters, honest and cheating, and
`remote_bandwidth.period.run`, honest, cheating and unverified), whose Lean their packages' difftests hold to those
verifiers (`../pous/check/tests/test_run_protocol.py`, `../remote_bandwidth/lean/tests/test_protocol.py`).
Hypothesis draws 1000 cases a product (seed 0): each component's case accepts, rejects, is undecided or fails to decode
(one draw in six every one aimed at acceptance, one in six all but one, which is aimed at undecided), every case hands
the components the same blobs (integrity's circuit and public file), and the composition's inputs, its blobs or the case
itself are often broken. Integrity accepts only on a session's proof, which no case carries (its cases are audits with no
session, undecided when their draw drew nothing, and audits whose sessions' files are in no blob), so no product
accepts in the difftest until a v2 case with a proof does. The port follows top's rule where the
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
