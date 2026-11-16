# core_protocol: the interface every core protocol implements

Every core protocol in Verity, whatever it audits, is checked the same way. Before the challenge the prover commits to
something; afterwards it hands over evidence; and the check reads the public parameters, the commitment and the
evidence and gives a verdict. This Lean package states that shape once, as the type `Core.Protocol`, so that each
protocol's check is one value of it and every protocol's executable answers on the same command line.

A protocol's check is its own Lake package. It requires this one by path, defines one `Core.Protocol` value, and makes
its executable `def main (args) := myProtocol.main args`. `CoreProtocol/Example.lean` is a toy protocol written that
way, the one to read first.

## Try it

~~~sh
bash tools/verity/lean/setup.sh verity/protocols/interface
(cd verity/protocols/interface && lake build core-protocol-example && .lake/build/bin/core-protocol-example id)
~~~

It builds the toy protocol and prints its id, `verity/example/v0`; `check` and `serve` answer on the same executable.

## The interface

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

## The command line, the same for every protocol

- `<exe> id` prints the id.
- `<exe> check PARAMS COMMITMENT EVIDENCE` reads three files, prints the outcome line, and exits 0, 1 or 2 for
  accept, reject or undecided.
- `<exe> serve` reads one case per stdin line, `{"commitment":"<hex>","evidence":"<hex>","params":<JSON>}` in canonical
  JSON, and writes one flushed outcome line per case. This is the mode the difftest harness drives.
- A usage error exits 64, and an outcome that can't be written canonically exits 70.
