# core_protocol: the interface every core protocol implements

This is a Lake package (`CoreProtocol`, namespace `Core`). It requires C-Flock's verifier by path for canonical JSON,
bytes and hashes, and nothing else: no Mathlib. A protocol's port is its own package. It requires this one by path,
defines one `Core.Protocol` value, and makes its executable `def main (args) := myProtocol.main args`.

## The interface (`CoreProtocol/Protocol.lean`)

| field | type | what it is |
| --- | --- | --- |
| `id` | `String` | the protocol id and version, e.g. `verity/pouw/v1` |
| `Params`, `params` | `Json → V Params` | the public parameters (k, Δ, the law), read from canonical JSON |
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
Python's `str(Fraction)`).

## The command line (`CoreProtocol/Cli.lean`), the same for every protocol

- `<exe> id` prints the id.
- `<exe> check PARAMS COMMITMENT EVIDENCE` reads three files, prints the outcome line, and exits 0, 1 or 2 for
  accept, reject or undecided.
- `<exe> serve` reads one case per stdin line, `{"commitment":"<hex>","evidence":"<hex>","params":<JSON>}` in canonical
  JSON, and writes one flushed outcome line per case. This is the mode the difftest harness drives.
- A usage error exits 64, and an outcome that can't be written canonically exits 70.

## The profile (`CoreProtocol/Profile.lean`)

`Core.Integrity` is `verity.experimental.protocols.profile.IntegrityProfile.to_json`: `programs`, `prescribed`,
`linked`, `covered`, the `levels` coarse to fine (name, query, size, and draw: `{"bernoulli":"p"}`, `{"subset":k}` or
`{"law":"stratified","strata":[...]}`), `delta` and `drawn`. The one change is that δ is an exact ratio, not a float. A
protocol whose acceptance isn't a sampling law over units (a timed storage audit, a certificate check) declares its own
profile type and JSON.

## Properties

A protocol's properties go in `Security/Properties/<Protocol>.lean`, stated against `P.check` (or `P.Accepts`) with its
actual parameters, every input covered and assumptions as named `Prop`s. `Security/Properties` is a Lake package of
its own (`security_properties`), which requires each program by path, as `security` (Mathlib only) may not. They gate
promotion into `verity/core/`, not the port. `CoreProtocol/Example.lean` is a toy protocol whose `#guard`s pin every path of `answer` and `case`;
`core-protocol-example` runs it.
