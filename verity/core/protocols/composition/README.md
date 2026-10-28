# core_composition: the onsite and remote protocols as compositions of subprotocols

Verity's two composite protocols, onsite and remote, are each one verdict over a list of subprotocols. Three Lake packages
hold them, each on C-Flock's verifier and the core interface (`../interface`) by path, onsite and remote also on their
subprotocols' packages, with no Mathlib:

- `core_composition` (here, lib `CoreComposition`, namespace `Core.Composition`): `product id components`, the
  `Core.Protocol` of a composition, with its verdict rule, its formats and the commitment's binding (`Product.lean`);
  `same id ss components`, the product whose subprotocols `ss` must commit to one record (`Same.lean`); both over blobs
  (`Blobs.lean`); the stub subprotocols (`Stubs.lean`); and the `#guard`s that pin the rules (`Guards.lean`).
- `core_onsite` (`../onsite`, `Core.Onsite.protocol`, `verity/onsite/v0`): sampled proofs, the inference-only policy,
  PoUW, PoUS, proof of useful bandwidth (PoUB), network certification and the memory challenger's wipe, the policy and
  sampled proofs on one registration record.
- `core_remote` (`../remote`, `Core.Remote.protocol`, `verity/remote/v0`): sampled proofs, the inference-only policy and
  PoUW, on one window record.

The rule and the formats are written once, over any list of subprotocols, and the two products are values of it that
differ in their id and subprotocols. Each product's executable is the core CLI (`onsite id | check | serve`,
`remote id | check | serve`), and its `serve` is what the difftest drives.

## The protocol

The JSON key `components` below is the wire format's name for the list of subprotocols.

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
| the cases fit, and the subprotocols tied to one record committed to two (one record, below), whatever they answer | reject |
| none rejects, and the commitment isn't the binding of their digests | reject |
| none rejects, the commitment binds them, and some is undecided | undecided |
| every one accepts, and the commitment binds them | accept |

Any rejection rejects, even when another subprotocol is undecided: otherwise a prover could hide a rejection by leaving
another subprotocol undecided. And every subprotocol counts: each one's rejection rejects the product.

On acceptance the product makes public each subprotocol's public outputs and keeps each one's profile, both as JSON objects
keyed by the subprotocol's id. On rejection or undecided nothing is public.

`Properties.Composition` (`Security/Properties/Composition.lean`, proved in `Security/Proofs/Properties/Composition.lean`)
states the accept row against this code. `answerAccepts` covers every `Core.Protocol`: what `answer` accepts decodes,
and its check accepts it. `acceptMeans` covers `Core.Composition.check`: what it accepts fits the composition, every
subprotocol accepts its own case, and the commitment is `bind` of the subprotocols' digests.

## Subprotocols

| subprotocol | onsite | remote | `Core.Protocol` | id |
| --- | --- | --- | --- | --- |
| network certification (over the period: its sampling blocks of transfer blocks certified and proved, and the bound `K` on what left the site) | yes | | `NetworkCheck.Period.protocol` (`network_check`, `../network/lean`) | `verity/certifier/v0` |
| PoUW (its period credit: the period's sampling blocks credited all or nothing) | yes | | `PouwCheck.Period.protocol` (`pouw_check`, `../pouw/lean/check`) | `verity/pouw/period/v1` |
| PoUW (its sampling block check) | | yes | `PouwCheck.protocol` (`pouw_check`, `../pouw/lean/check`) | `verity/pouw/v1` |
| sampled proofs (the one-stage audit on the proofs themselves, which it reads as blobs) | yes | yes | `SampledProofs.OneStageZk.protocol` (`core_sampled_proofs`, `../../service/sampled_proofs`) | `verity/sampled-proofs/one-stage/v2` |
| the inference-only policy (its check: the declared circuit's division into inference blocks within their gate and fresh-input budgets, and the provenance of the registered information) | yes | yes | `NciCheck.Protocol.protocol` (`nci_check`, `../network/nci/lean`) | `verity/inference-only/v0` |
| PoUS (its check of a run) | yes | | `PousCheck.RunProtocol.protocol` (`pous_check`, `../pous/check`) | `verity/pous/p2/v4/run` |
| proof of useful bandwidth, PoUB (its session) | yes | | `RemoteAudit.Protocol.protocol` (`remote_audit`, `../remote_bandwidth/lean`) | `verity/remote-bandwidth/v0` |
| the wipe (the memory challenger's streamed-label wipe of the period) | yes | | `MemoryChallenger.WipeProtocol.protocol` (`core_memory_challenger`, `../../devices/memory_challenger/lean`) | `verity/wipe/v0` |

Every subprotocol of both products is ported to its own `Core.Protocol`.

A subprotocol without a port is a stub, `Core.Example.protocol` under an id that says so (`Stubs.lean`). A port replaces
its stub in `Core.Onsite.components` and `Core.Remote.components` with its own `Core.Protocol` value, its package a
`[[require]]` of each product, keeping each list in the order of its ids (`ordered`, which each product's `#guard`
checks), and deletes the stub. The difftest's reference (`tests/reference.py`) replaces the stub's mirror with the port's
own Python (`Ported`), or its Python's answers on cases made once (`Recorded`), and its cases come from the port's own
difftest or its period hook.

The inference-only policy has no rounds and reads no coins. Its case is the declared circuit, the policy's budgets and
the registration log before the run (params), the run's own registration record (commitment: its digest is the one
sampled proofs' audit commits to) and the declared division with θ and θ's provenance (evidence). It accepts exactly
when the record names the declared circuit (its `program` is the circuit's SHA-512, as sampled proofs' audit asks of its
own record) and the policy's check (`NciCheck.Baseline.evaluate`) finds that the division complies. Each product ties
the policy's record to sampled proofs' (one record, below), so the circuit the policy checked is the one sampled proofs
audited.

A stated exception: the policy's v0 publishes the declared division and each inference block's profile (its gates and
fresh-input bits) in the clear. Both products compose it for the run of record of 28 Oct only, as an exception to
everything-zero-knowledge, by Daniel's ruling: "Let's run it in the clear tonight as a stated exception." v1's
private form (the division committed under hiding leaves, compliance proved, a padded profile) replaces it after
15:00Z on 28 Oct. `tests/test_inference_only_not_composed.py` refuses any protocol that publishes
private data among the products' components unless its `KNOWN` lists it with its reason, and `KNOWN` only shrinks;
`benchmarks/one_stage/composite.py` writes the exception's line into the record it renders.

A subprotocol that reads blobs (`Core.BlobProtocol`, the content-addressed files beside its three byte strings: sampled
proofs' proof files) makes the product one too: `productB` (`CoreComposition/Blobs.lean`) hands the caller's blobs to
every subprotocol, those that read none ignore them (`Protocol.withoutBlobs`), and on the blobs it is `product` of its
subprotocols on them (`productB_on`). `onsite check P C E [BLOB ...]` reads them after the evidence, and a `serve` case
carries them as hex under `blobs`.

## One record

Every subprotocol whose commitment is the run's record must commit to the composition's one. `same id ss components`
(`sameB` over blobs) is the product of the subprotocols, except that once the cases fit, subprotocols `ss` that didn't
all commit to the first one's digest reject it whatever the subprotocols answer (`strays`: `record: s committed D, t
committed D'`); one whose commitment doesn't decode rejects as a subprotocol. Its formats are the product's. Both
products are one:

- remote, `sameB "verity/remote/v0" [verity/inference-only/v0, verity/pouw/v1, verity/sampled-proofs/one-stage/v2]
  components`, whose policy, PoUW window and one-stage audit each commit to the window record;
- onsite, `sameB "verity/onsite/v0" [verity/inference-only/v0, verity/sampled-proofs/one-stage/v2] components`, whose
  policy and one-stage audit each commit to the run's registration record.

Each product's `#guard`s pin that two records reject it with the policy among the subprotocols. Onsite's PoUW period and
network certification name one record a window, in their evidence and their period commitment, so they join the check
when each window's record names the registration log's head (`Core.Registration`, `../interface`), which then is the one
every subprotocol names.

## The difftest

`uv run tools/verity/check/suites.py verity-core-composition` builds `onsite` and `remote` and drives each one's `serve`
through the harness's `ProtocolPort` (`../../difftest`), holding every field of each outcome but its reason to
`tests/reference.py`: the composition rule above, over the harness's framing of the core interface
(`verity_difftest.protocol.answer`), and each subprotocol's own Python, which that subprotocol's own difftest holds its
Lean to:

- PoUW: `core_check.case`, and for its period `period_check.case`;
- sampled proofs: v2's framing over the one-stage difftest's `one_stage` and decoders, `one_stage_zk_case.answer`, on
  the cases v2 decides without a session's proof;
- network certification: its period difftest's `reference`, the Python mirror over the certifier's certificates,
  one-stage's answers on the compute audits and v2 on each sampling block's proof, on small signed periods from its
  `build`;
- the inference-only policy: its protocol difftest's `reference`, the policy's Python check (`check_baseline`) behind
  the protocol's decoders, on θ's provenance difftest's declarations with at most one change, from its `drawn`, or on
  the one-stage audit's own declared circuit and record (`policy_on`);
- the wipe: its mirror, `wipe_protocol_case.answer`, on signed wipes from its `case`;
- PoUS and PoUB: their Python verifiers' answers on small periods of their hooks (`tests/periods.py`: `pous.period.run`
  signed at its deployed parameters, honest and cheating, and `remote_bandwidth.period.run`, honest, cheating and
  unverified), whose Lean their packages' difftests hold to those verifiers (`../pous/check/tests/test_run_protocol.py`,
  `../remote_bandwidth/lean/tests/test_protocol.py`).

Hypothesis draws 1000 cases a product (seed 0): each subprotocol's case accepts, rejects, is undecided or fails to decode
(one draw in six every one aimed at acceptance, one in six all but one, which is aimed at undecided; the inference-only
policy is never undecided), every case hands the subprotocols the same blobs (sampled proofs' circuit and public file),
and the composition's inputs, its blobs or the case itself are often broken. Sampled proofs accept only on a session's
proof, which no case carries (its cases are audits with no session, undecided when their draw drew nothing, and audits
whose sessions' files are in no blob), so no product accepts in the difftest until a v2 case with a proof does. Often
the tied subprotocols are put on one record (`one_record`), and otherwise they mostly commit to two, which the product
refuses. Remote's cases reject: on one record all three are on PoUW's, which the one-stage audit refuses (PoUW's window
reads registered values, outside v2's mirror). Onsite's reach undecided: on one record the policy declares the one-stage
audit's own circuit on its record, which it accepts.
`test_remote_refuses_two_records_and_else_keeps_the_products_verdict` shows that refusal changes no other verdict of
the product's. `test_a_reference_without_reject_dominance_is_caught` shows the difftest tells the rule above from one
without reject dominance: such a reference disagrees with the Lean on a case where one subprotocol rejects and another
is undecided.

A bare `uv run` inside this directory stops its workspace search at `verity/core/protocols/pyproject.toml`; run the suite
through `suites.py`, or `pytest` with the workspace's `.venv`.
