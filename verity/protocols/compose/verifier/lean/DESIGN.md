# core_composition: design

The packages, the composition's rule and formats, which sets of subprotocols a verifier runs, each subprotocol's port
and the one record they share. `README.md` says what a composition is, lists each composite's subprotocols, and has the
difftest.

## The packages

Compose's two configurations, onsite and remote, are each one verdict over a list of subprotocols. Three Lake packages
hold them, each on the core interface (`../../../interface`) by path, with no Mathlib: `core_composition` also on the base
package (`core_base`: canonical JSON, bytes and SHA-512), onsite and remote also on Flock's verifier and their
subprotocols' packages:

- `core_composition` (here, lib `CoreComposition`, namespace `Core.Composition`): `product id subprotocols`, the
  `Core.Protocol` of a composition, with its verdict rule, its formats and the commitment's binding (`Product.lean`);
  `same id ss components`, the product whose subprotocols `ss` must commit to one record (`Same.lean`), and `tiedSame id
  ss ties components`, `same` whose subprotocols of each tie must name one value (`Tie.lean`); all over blobs
  (`Blobs.lean`); the stub subprotocols (`Stubs.lean`); and the `#guard`s that pin the rules (`Guards.lean`).
- `core_onsite` (`verity/protocols/compose/verifier/onsite`, `Core.Onsite.protocol`, `verity/onsite/v0`): SPOT, the inference-only policy,
  PoUW, PoUS, PoUB-P2 (on PoUS's store), network certification and the memory attester's
  wipe, the policy and SPOT on one registration record and every window of the period on it, as at remote,
  and PoUS and PoUB-P2 on one store (a `tiedSameB`).
- `core_remote` (`..`): the composite of record `Core.Remote.V1.protocol` (`verity/remote/v1`): SPOT,
  the inference-only policy and Pearl+'s run, the policy and SPOT on one registration record and the run's
  commitment naming it (a `tiedSameB`); and H1's development executable `Core.Remote.protocol` (`verity/remote/v0`),
  with PoUW's period in Pearl+'s place and every window of the period on that record, which no configuration names.

The rule and the formats are written once, over any list of subprotocols, and the composites are values of it that
differ in their id and subprotocols. Each composite's executable is the core CLI (`onsite id | check | serve`,
`remote id | check | serve`), and its `serve` is what the difftest drives.

These are the only composites a verifier runs. `product IDS id | check | serve` (`verity/protocols/compose/verifier/onsite/CoreComposites.lean`,
`Core.Composites.ofIds`), which `verity run`'s verifier runs over a configuration's subprotocols, is each composite's
CLI for its subprotocols, in any order, and refuses every other set, since protocol soundness covers only onsite and
`verity/remote/v1`: H1's `verity/remote/v0` is refused, and runs only as its development executable. Remote's plain
sibling (`Core.Site.Registration.Composite.plain`, the inference-only policy's rows check in v0's place) is refused by
name until its soundness is proved (`Core.Composites.plainRefusal`).

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

`Properties.Composition` (`security/Properties/Composition.lean`, proved in `security/Proofs/Properties/Composition.lean`)
states the accept row against this code. `answerAccepts` covers every `Core.Protocol`: what `answer` accepts decodes,
and its check accepts it. `acceptMeans` covers `Core.Composition.check`: what it accepts fits the composition, every
subprotocol accepts its own case, and the commitment is `bind` of the subprotocols' digests.

## Ports

Every subprotocol of both products is ported to its own `Core.Protocol`.

A subprotocol without a port is a stub, `Core.Example.protocol` under an id that says so (`Stubs.lean`). A port replaces
its stub in `Core.Onsite.components` and `Core.Remote.components` with its own `Core.Protocol` value, its package a
`[[require]]` of each product, keeping each list in the order of its ids (`ordered`, which each product's `#guard`
checks), and deletes the stub. The difftest's reference (`tests/reference.py`) replaces the stub's mirror with the port's
own Python (`Ported`), or its Python's answers on cases made once (`Recorded`), and its cases come from the port's own
difftest or its period hook.

The inference-only policy has no rounds and reads no coins. Its case is the registered circuit, the policy's budgets and
the registration log before the run (params), the run's own registration record (commitment: its digest is the one
SPOT's audit commits to) and the division with θ and θ's provenance (evidence). It accepts exactly
when the record names the registered circuit (its `program` is the circuit's SHA-512, as SPOT's audit asks of its
own record) and the policy's check (`InferenceOnlyCheck.Theta.evaluate`) finds that the division complies. Each product ties
the policy's record to SPOT's (one record, below), so the circuit the policy checked is the one SPOT
audited.

A stated exception: the policy's v0 publishes the division and each inference block's profile (its gates and
fresh-input bits) in the clear. Both products compose it for one run of record only, as an exception to
everything-zero-knowledge, and v1's private form (the division committed under hiding leaves, compliance proved, a
padded profile) replaces it. `tests/test_inference_only_not_composed.py` refuses any protocol that publishes
private data among the products' subprotocols unless its `KNOWN` lists it with its reason, and `KNOWN` only shrinks;
`benchmarks/one_stage/composite.py` writes the exception's line into the record it renders.

A subprotocol that reads blobs (`Core.BlobProtocol`, the content-addressed files beside its three byte strings) makes the product one too: `productB` (`CoreComposition/Blobs.lean`) hands the caller's blobs to
every subprotocol, those that read none ignore them (`Protocol.withoutBlobs`), and on the blobs it is `product` of its
subprotocols on them (`productB_on`). `onsite check P C E [BLOB ...]` reads them after the evidence, and a `serve` case
carries them as hex under `blobs`.

## One record

Every subprotocol whose commitment is the run's record must commit to the composition's one. `same id ss components`
(`sameB` over blobs) is the product of the subprotocols, except that once the cases fit, subprotocols `ss` that didn't
all commit to the first one's digest reject it whatever the subprotocols answer (`strays`: `record: s committed D, t
committed D'`); one whose commitment doesn't decode rejects as a subprotocol. Its formats are the product's. Each
product is one:

- remote, `tiedSameB "verity/remote/v1" [verity/inference-only/v0, verity/sampled-proofs/v1] [run] components`
  (and its development composite, the same over `verity/remote/v0` with `[windows]`), whose policy and SPOT
  each commit to the run's registration record (in either form, the one-stage audit's);
- onsite, `tiedSameB "verity/onsite/v0" [verity/inference-only/v0, verity/sampled-proofs/v1] [windows, store]
  components` (`Core.Onsite.ties`), whose policy and SPOT each commit to the run's registration record (in
  either form, the one-stage audit's).

Each product's `#guard`s pin that two records reject it with the policy among the subprotocols. Onsite's PoUW period and
network certification name one record a window, in their evidence and their period commitment, so they join the check
when each window's record names the registration log's head (`Core.Registration`, `../../../interface`), which then is the one
every subprotocol names.

A tie (`Tie`, `Tie.lean`) is the values each named subprotocol reads off its own case (its commitment's bytes, or its
evidence's) with its own decoder, which must all be one value (`Tie.strays`: `what: s and t named different values`,
neither value, which can be witness). `tiedSame id ss ties components`
(`tiedSameB` over blobs) is `same` with its ties held (`holdTies`): a verdict of `same` that isn't a reject becomes a
reject when a tie strays, and a reject stays itself. Onsite's ties are remote's `Core.Remote.windows` (below), read
from `core_remote` rather than restated, then `Core.Onsite.store`, which puts PoUS and PoUB-P2 on one store: the root
of `C*` PoUS's setup published (its commitment's `root`) and the root PoUB-P2's registration names (its commitment's
`root`, the case's `store_root`). PoUB's generic session (`verity/remote-bandwidth/v0`), whose payload
root is its own salted tree's, is not a subprotocol. Once the log exists, both read the store's hardware entry.
Remote's one tie, `Core.Remote.windows`, puts every window of PoUW's period on the policy's record: the digest of the
record the policy's commitment is, and the digest of each window's Commit record as the period's credit reads it
(`PouwCheck.Credit.commitOf?`). The period's own commitment is the auditor's ledger, so it can't join `same`.

