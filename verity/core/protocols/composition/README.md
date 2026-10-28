# core_composition: the onsite and remote protocols as compositions of subprotocols

Verity's two composite protocols, onsite and remote, are each one verdict over a list of subprotocols. Three Lake packages
hold them, each on C-Flock's verifier and the core interface (`../interface`) by path, onsite and remote also on their
subprotocols' packages, with no Mathlib:

- `core_composition` (here, lib `CoreComposition`, namespace `Core.Composition`): `product id components`, the
  `Core.Protocol` of a composition, with its verdict rule, its formats and the commitment's binding (`Product.lean`);
  `registered id R s components`, the same after a registration check `R`, on the registration the subprotocol `s`
  commits to, `tied id components ties`, the product whose subprotocols must name one value for each tie, and `same id
  ss components`, the one whose subprotocols `ss` must commit to one record (`Registered.lean`); all four over blobs
  (`Blobs.lean`); the stub subprotocols (`Stubs.lean`); and the `#guard`s
  that pin the rules (`Guards.lean`).
- `core_onsite` (`../onsite`, `Core.Onsite.protocol`, `verity/onsite/v0`): sampled proofs, PoUW, PoUS, proof of useful
  bandwidth (PoUB-P2, on PoUS's store), network certification and the memory challenger's wipe, a `tiedB` with PoUS
  and PoUB-P2 on one store.
- `core_remote` (`../remote`, `Core.Remote.protocol`, `verity/remote/v0`): sampled proofs and PoUW, on one window
  record.

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
| PoUW (its period credit: the period's windows credited all or nothing) | yes | | `PouwCheck.Period.protocol` (`pouw_check`, `../pouw/lean/check`) | `verity/pouw/period/v1` |
| PoUW (its window check) | | yes | `PouwCheck.protocol` (`pouw_check`, `../pouw/lean/check`) | `verity/pouw/v1` |
| sampled proofs (the one-stage audit on the proofs themselves, which it reads as blobs) | yes | yes | `SampledProofs.OneStageZk.protocol` (`core_sampled_proofs`, `../../service/sampled_proofs`) | `verity/sampled-proofs/one-stage/v2` |
| PoUS (its check of a run) | yes | | `PousCheck.RunProtocol.protocol` (`pous_check`, `../pous/check`) | `verity/pous/p2/v4/run` |
| proof of useful bandwidth, PoUB-P2 (its session over PoUS's store) | yes | | `RemoteAudit.P2Protocol.protocol` (`remote_audit`, `../remote_bandwidth/lean`) | `verity/remote-bandwidth/p2/v0` |
| the wipe (the memory challenger's streamed-label wipe of the period) | yes | | `MemoryChallenger.WipeProtocol.protocol` (`core_memory_challenger`, `../../devices/memory_challenger/lean`) | `verity/wipe/v0` |

Every subprotocol of both products is ported to its own `Core.Protocol`. NCI is neither a subprotocol nor a
registration check of either product: its check reads the division into inference blocks in the clear, which is private,
so onsite doesn't run it. The inference-only policy is not a subprotocol of either product yet.

A subprotocol without a port is a stub, `Core.Example.protocol` under an id that says so (`Stubs.lean`). A port replaces
its stub in `Core.Onsite.components` and `Core.Remote.components` with its own `Core.Protocol` value, its package a
`[[require]]` of each product, keeping each list in the order of its ids (`ordered`, which each product's `#guard`
checks), and deletes the stub. The difftest's reference (`tests/reference.py`) replaces the stub's mirror with the port's
own Python (`Ported`), or its Python's answers on cases made once (`Recorded`), and its cases come from the port's own
difftest or its period hook.

A subprotocol that reads blobs (`Core.BlobProtocol`, the content-addressed files beside its three byte strings: sampled
proofs' proof files) makes the product one too: `productB` (`CoreComposition/Blobs.lean`) hands the caller's blobs to
every subprotocol, those that read none ignore them (`Protocol.withoutBlobs`), and on the blobs it is `product` of its
subprotocols on them (`productB_on`). `onsite check P C E [BLOB ...]` reads them after the evidence, and a `serve` case
carries them as hex under `blobs`.

## A registration check

`registered id R s components` checks a registration before composing, in its commit phase, and `registeredB` is the
same over blobs, the registration being the one the subprotocol `s` commits to, one registration digest in both. Neither
product is one today. Its formats are the product's with the registration's beside the subprotocols':

| input | format |
| --- | --- |
| params | `{"components":[...],"registration":{"id":ID,"params":PARAMS}}` |
| commitment | the product's 64 bytes, then the registration's digest; its own digest |
| evidence | `{"components":[...],"registration":{"commitment":HEX,"evidence":HEX,"id":ID,"params":PARAMS}}` |

It rejects when the registration isn't the registration check's or carries other params, when the registration check
doesn't accept it, or when the commitment doesn't end in its digest, whatever the subprotocols answer. When the cases fit
it rejects too when the committing subprotocol's case (the first under the id `s`) doesn't commit to that digest, again
whatever the subprotocols answer (`mismatch`: `registration: R checked D, s committed D'`, `D'` hex or `no digest`).
Otherwise it answers as the product of the subprotocols does under the commitment's first 64 bytes, and the registration
adds nothing to what is public or kept.

## One record

Every subprotocol whose commitment is a window record must commit to the composition's one. `same id ss components`
(`sameB` over blobs) is the product of the subprotocols, except that once the cases fit, subprotocols `ss` that didn't
all commit to the first one's digest reject it whatever the subprotocols answer (`strays`: `record: s committed D, t
committed D'`); one whose commitment doesn't decode rejects as a subprotocol. Its formats are the product's. Remote is
one: `sameB "verity/remote/v0" [verity/pouw/v1, verity/sampled-proofs/one-stage/v2] components`, whose PoUW window and
one-stage audit each commit to the window record. At onsite the one-stage audit commits to the window record too.
Onsite's PoUW period and network certification name one record a window, in their evidence and their period
commitment, so they join the check when each window's record names the registration log's head (`Core.Registration`,
`../interface`), which then is the one every subprotocol names.

A tie (`Tie`, `Registered.lean`) generalizes this: a value each named subprotocol reads off its own commitment's bytes
with its own decoder, which must be one value once the cases fit (`Tie.strays`: `what: s committed D, t committed D'`),
else the composition rejects whatever the subprotocols answer; a subprotocol whose commitment doesn't decode rejects as
itself. `tied id components ties` (`tiedB` over blobs) is the product with its ties checked once the cases fit, `same`
is `tied` on the tie `records` of its subprotocols' digests, and `registered id R s components ties` checks its ties
after the registration. Onsite is `tiedB` with one tie, `Core.Onsite.store`: PoUS and PoUB-P2 on one store, the root of `C*`
PoUS's setup published (its commitment's `root`) and the root PoUB-P2's registration names (its commitment's `root`,
the case's `store_root`). The generic remote bandwidth audit (`verity/remote-bandwidth/v0`), whose payload root is its
own salted tree's, is not a component. Once the log exists, both read the store's hardware entry.

## The difftest

`uv run tools/verity/check/suites.py verity-core-composition` builds `onsite` and `remote` and drives each one's `serve`
through the harness's `ProtocolPort` (`../../difftest`), holding every field of each outcome but its reason to
`tests/reference.py`: the composition rule above, over the harness's framing of the core interface
(`verity_difftest.protocol.answer`), and each subprotocol's own Python, which that subprotocol's own difftest holds its
Lean to:

- PoUW: `core_check.case`, and for its period `period_check.case`;
- sampled proofs: v2's framing over the one-stage difftest's `one_stage` and decoders, `one_stage_zk`, on the cases v2
  decides without a session's proof;
- network certification: its period difftest's `reference`, the Python mirror over the certifier's certificates and
  one-stage's answers, on small signed periods from its `build`;
- the wipe: its mirror, `wipe_protocol_case.answer`, on signed wipes from its `case`;
- PoUS and PoUB-P2: their Python verifiers' answers on one committed `C*` (`tests/periods.py`: `pous.period.audit`
  signed at its deployed parameters, honest and cheating, and an honest run on another store; PoUB-P2's sessions over
  that `C*`, honest, cheating and unverified, as `p2_case` answers them), whose Lean their packages' difftests hold to
  those verifiers (`../pous/check/tests/test_run_protocol.py`, `../remote_bandwidth/lean/tests/test_p2_case.py`).

Hypothesis draws 1000 cases a product (seed 0): each subprotocol's case accepts, rejects, is undecided or fails to decode
(one draw in six every one aimed at acceptance, one in six all but one, which is aimed at undecided), every case hands
the subprotocols the same blobs (sampled proofs' circuit and public file), and the composition's inputs, its blobs or the
case itself are often broken. Sampled proofs accept only on a session's proof, which no case carries (its cases are
audits with no session, undecided when their draw drew nothing, and audits whose sessions' files are in no blob), so no
product accepts in the difftest until a v2 case with a proof does. Remote's cases reject: a third of them put both
subprotocols on PoUW's record, which the one-stage audit refuses (PoUW's window reads registered values, outside v2's
mirror), and the rest mostly on two records, which remote refuses;
`test_remote_refuses_two_records_and_else_keeps_the_products_verdict` shows that refusal changes no other verdict of
the product's. Onsite's PoUS and PoUB-P2 cases are on one store, but for PoUS's honest run on another
(`periods.OTHER_STORE`, a quarter of its cases aimed at acceptance), which onsite refuses beside PoUB-P2's;
`test_onsite_refuses_two_stores_and_else_keeps_its_verdict` shows that refusal changes no other verdict. A case that
names PoUB-P2's by the generic audit's id is refused, as no component.
`test_a_reference_without_reject_dominance_is_caught`
shows the difftest tells the rule above from one without reject dominance: such a reference disagrees with the Lean on a
case where one subprotocol rejects and another is undecided.

A bare `uv run` inside this directory stops its workspace search at `verity/core/protocols/pyproject.toml`; run the suite
through `suites.py`, or `pytest` with the workspace's `.venv`.
