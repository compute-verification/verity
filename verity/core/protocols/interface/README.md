# core_protocol: the interface every core protocol implements

This is a Lake package (`CoreProtocol`, namespace `Core`). It requires C-Flock's verifier by path for canonical JSON,
bytes and hashes, and nothing else: no Mathlib. A protocol's port is its own package. It requires this one by path,
defines one `Core.Protocol` value, and makes its executable `def main (args) := myProtocol.main args`.

## The interface (`CoreProtocol/Protocol.lean`)

| field | type | what it is |
| --- | --- | --- |
| `id` | `String` | the protocol id and version, e.g. `verity/pouw/v1` |
| `Params`, `params` | `Json → V Params` | the public parameters (k, Δ, the sampling design), read from canonical JSON |
| `Commitment`, `commitment` | `ByteArray → V Commitment` | what the prover committed to before the challenge |
| `digest` | `Commitment → ByteArray` | the digest that binds it, which a record or certificate cites |
| `Evidence`, `evidence` | `ByteArray → V Evidence` | everything else the check reads: record, transcript, proofs, certificate |
| `check` | `Params → Commitment → Evidence → Outcome Public Profile` | the protocol, one total function |
| `Public`, `publicJson` | `Public → Json` | what acceptance makes public |
| `Profile`, `profileJson` | `Profile → Json` | what acceptance established (`Core.Integrity` for protocols that sample) |

An `Outcome` is `accept published profile`, `reject reason` or `undecided reason`. `undecided` means the evidence
doesn't settle the question (no certificate yet). It doesn't reject the prover, and nothing is made public. Public
outputs and the profile exist only in the `accept` case.

`Protocol.answer P params commitment evidence` is the only entry point. It requires the parameters to be canonical
JSON (`Core.canonicalJson`, the bytes `Flock.canon` writes), turns every decoding failure into a rejection prefixed
`params: `, `commitment: ` or `evidence: `, and runs `check`. Every input gets a verdict. `Answer.canonical` writes the
outcome's one canonical form:

```json
{"commitment":"<hex digest, or null when the commitment didn't decode>","profile":<JSON or null>,
 "protocol":"<id>","public":<JSON or null>,"reason":"<string or null>","verdict":"accept|reject|undecided"}
```

Canonical JSON has integers only. Write probabilities and δ as `Core.Ratio` strings (`"n/d"` in lowest terms, or `"n"`,
Python's `str(Fraction)`). `publicJson` never writes `null`.

A protocol that consumes another's answer reads it with `Answer.ofJson` and its profile with `Integrity.ofJson`. Each
accepts only what `toJson` writes (re-encoding must give the same canonical JSON) and only the shapes `answer` returns:
an `accept` has no reason and both its public outputs and its profile, while a refusal has a reason and neither.

`Protocol`, `Profile` and `Ratio` are the package's spec (`layers` in `lean-audit.json`): a property's statement may read
them, and they import only Lean's JSON, C-Flock's bytes and canonical JSON, and one another. `Verdict`, `Ratio` and the
profile's types derive `DecidableEq`.

## The command line (`CoreProtocol/Cli.lean`), the same for every protocol

- `<exe> id` prints the id.
- `<exe> check PARAMS COMMITMENT EVIDENCE` reads three files, prints the outcome line, and exits 0, 1 or 2 for
  accept, reject or undecided.
- `<exe> serve` reads one case per stdin line, `{"commitment":"<hex>","evidence":"<hex>","params":<JSON>}` in canonical
  JSON, and writes one flushed outcome line per case. This is the mode the difftest harness drives.
- A usage error exits 64, and an outcome that can't be written canonically exits 70.

## Blobs (`CoreProtocol/Blobs.lean`)

A check that reads bytes too large for canonical JSON (a statement's circuit file, its public file, its sessions'
records and proofs) is a `Core.BlobProtocol`: a `Protocol` with `checkBlobs`, which also takes `Core.Blobs`. The caller
passes the bytes, and the verifier holds each under the SHA-512 it computes itself, so `Blobs.get digest` finds a blob
only by the digest of its own bytes. The check names each file by a digest its parameters or evidence carry, and the
lookup is the check that the file is the committed one. `BlobProtocol.answer` is `Protocol.answer` with `checkBlobs` on
the caller's blobs. `toProtocol` is the protocol on no blobs, so a composition of `Protocol`s takes it unchanged.
`Protocol` and `Protocol.answer` don't change, and a protocol that reads no blobs needs none of this.

`BlobProtocol.main`'s command line is `Protocol.main`'s, with `check PARAMS COMMITMENT EVIDENCE [BLOB ...]` (each blob a
file) and a `serve` case that may carry `"blobs": ["<hex>", ...]` (absent: none). `CoreProtocol/BlobsExample.lean` is
the toy protocol whose guards cover both.

## The profile (`CoreProtocol/Profile.lean`)

`Core.Integrity` is `verity.core.protocols.profile.IntegrityProfile.to_json`: `programs`, `prescribed`,
`linked`, `covered`, the `levels` coarse to fine (name, query, size, and draw: `{"bernoulli":"p"}`, `{"subset":k}` or
`{"law":"stratified","strata":[...]}`), `delta` and `drawn`. The one change is that δ is an exact ratio, not a float. A
protocol whose acceptance isn't a sampling design over proof blocks (a timed storage audit, a certificate check) declares its
own profile type and JSON.

## The registration log (`CoreProtocol/Registration.lean`)

`Core.Registration` is a deployment's one registration log, which every subprotocol and both composites read. It is
append-only: each entry names the head of the log before it (`previous`), the digest of its last entry or of its
composite's genesis, so the head names the whole log. An entry is one of two kinds, each with its own versioned format:

| format | what it registers |
| --- | --- |
| `verity/registration/site/v1` | the Site's own site entry (`Core.Site.Registration.Entry`, `CoreProtocol/Site.lean`): the sheets, locations and links it adds to the Site, with the Site's JSON, digest and check |
| `verity/registration/information/v1` | θ, each registered value by its committed root (`value`, `root`, `leaves`, `words`, `word_bits`, `domain`, a public checkpoint's pin), and the builder `B` (digest, advice format, public advice cap `ℓ`); each once in the log |

Registration registers the site, θ and `B`, not the circuit `B` builds. A subprotocol's Program and parameters, and
the proof system, are that protocol's public verifier parameters, not entries of the log.

An entry's digest is `SHA-512(format ‖ 0 ‖ canon(entry))`, and the genesis is the Site's, so a log of site entries alone
has the Site's head. `check` holds each entry to the log before it and refuses the first that fails by name: a site
entry by the Site's checks (`Core.Site.Registration.Entry.refusal`, `previous` first), an information entry by
`previous` and then `duplicate-information`. A component reads its view of an accepted log by projection (`View`: the
Site, θ's values and the builder, at the head). As a core protocol, `protocol` (`verity/registration/log/v1`, the
`core-registration` executable) takes the auditor's parameters (`{"auditor", "composite"}`, the auditor the Site's
`Core.Site.Registration.Auditor`), the log as commitment (`{"composite", "entries"}`, its head the digest) and `{}` as
evidence; it makes public the log's view and rejects as `entry I: REFUSAL`, and refuses a log or parameters outside
ASCII without DEL as `Core.canonicalJson` does. `verity.core.protocols.registration` is its Python, held to it by the
composition suite's difftest (`../composition/tests/test_registration_difftest.py`).

## Properties

A protocol's properties go in `Security/Properties/<Protocol>.lean`, stated against `P.check` (or `P.Accepts`) with its
actual parameters, every input covered and assumptions as named `Prop`s. They, not the
port, are what makes a verifier proven. All protocols share one Lake package for them, `security_properties`, with the statements in
`Properties.<Protocol>` and the proofs in `Proofs.Properties.<Protocol>`:
- It requires `security` (Definitions and Models) and, by path, the code it states properties of. It never requires
  `security_proofs`, so code requires nothing, `security` requires no code, and nothing pulls in ArkLib.
- A result whose proof needs `security_proofs` keeps its statement in `Properties.<Protocol>`, and its proof goes in
  `security_proofs` as `Proofs.<Protocol>`, which may require `security_properties`. Sampled proofs' `audit_profile`
  (`Proofs/Flock/Soundness/Audit/OneStage.lean`) is one such result.
- Each protocol adds its root and its `[[require]]` to that one lakefile.

`CoreProtocol/Example.lean` is a toy protocol whose `#guard`s pin every path of `answer` and `case`, and the decoders'
round trips and refusals; `core-protocol-example` runs it.
