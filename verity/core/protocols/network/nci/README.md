# The inference-only policy

The inference-only policy asks that the declared circuit D have an inference-block partition 𝓘: blocks with gate
budget c (exactly c gates each) and fresh-input budget k (at most k bits of fresh input each), where fresh input is
every value outside the registered information θ, the models, code and data fixed before the monitoring period. It is
a protocol with no rounds: the developer registers R = (θ, D, 𝓘), and the verifier checks 𝓘 from the declared circuit
and the registered information alone. In Lean it is `Nci.Complies`, in an older form (below), and its check is
`NciCheck`.

The inference-only policy is a subprotocol of both composites, onsite and remote (`verity/core/protocols/composition`):
`NciCheck.Protocol.protocol` (`verity/inference-only/v0`, the executable `inference-only`) is its check behind the core
interface. Its params are the declared circuit, the budgets and the registration log before the run; its commitment is
the run's own registration record, which must name the declared circuit and which each composite requires to be the
record sampled proofs commits to (`Core.Composition.same`); its evidence is the declared division with θ and θ's
provenance. That evidence reaches the verifier in the clear: the check doesn't run inside a zero-knowledge
proof yet.

## Why it separates inference from training

An inference episode (a chat session) reads little fresh input, say 100 KB of prompt tokens, so honest serving
divides into inference blocks with a small fresh-input budget (`Nci.Guarantees.Inference`). A training episode (an RL
rollout) reads an enormous amount, say 1 TB of new parameters, so training that runs through inference blocks gets
only their yields q(c, k): a pricing argument says how much of a block's fresh-input budget each counted object of a
necessary subcomputation needs. The overhead lower bound sums those yields with free compute, at the hardware's rate,
and with the server-block yields that the memory wipe, proof of useful space (PoUS) and network certification bound (`GLOSSARY.md`, "Size, credit and
the overhead lower bound"). Its numbers come from [`benchmarks/rho`](../../../../../benchmarks/rho/README.md), whose
scenarios register a policy as if it were a subprotocol.

## The Lean

- **The definition**, `Security/Definitions/Nci.lean`. `Nci.Complies Γ α` states an older form: D divides into blocks
  of at most Γ gates, each reading at most α bits of fresh input per gate it holds (`Nci.novelReads`), so at most αΓ
  bits a block. Its link to the (c, k) form is proved: `Nci.Guarantees.CompliesCK` pads a complying partition into a
  (Γ, ⌊αΓ⌋)-partition (`Verity.Partition.IsCK`), and `Nci.Guarantees.ExpandComplies` carries compliance through the
  bit-level expansion at Γ·W for gates up to W bits wide.
- **The guarantees**, `Security/Definitions/Nci/Guarantees/`, each proved in `Security/Proofs/Nci/` as
  `Nci.SecurityProofs.G`. `ReadBound` turns a lower bound on fresh input into a lower bound on size, `Inference` shows
  honest inference complies, and the covering, training and threading statements are pricing arguments.
  `Nci.Guarantees.DampenerRemote` and its relatives state the remote composition's overhead in the older form.
- **The registered information's provenance**, `Nci.Assumptions.BaselineRegistered`: every θ entry is registered
  public data, never an earlier run's output.

## The check

`lean/` is the Lake package `nci_check`, code that reads Programs with C-Flock's decoder (`Flock.Program`) and imports
nothing else but Lean core and the core interface (`core_protocol`).

- `nci-check` runs `NciCheck.Baseline.check`: `NciCheck.Desc.check`, which decides `Nci.Complies Γ α` for a declared
  division aligned to the root's Calls at the cost of the descriptor (one summary per Definition), then each θ entry's
  provenance against the deployment's registration log.
- `nci-check --unrolled` runs the same with `NciCheck.check`, the check gate by gate, kept as the reference; its input
  format is in `NciCheck.Check`'s docstring. `nci-check --policy` runs the policy's check alone, without θ's
  provenance, which is what `benchmarks/verifier_cost` measures.
- `inference-only id | check | serve` runs `NciCheck.Baseline.check` as the core protocol `verity/inference-only/v0`
  (`NciCheck.Protocol`), the subprotocol of onsite and remote.
- `nci-difftest`, `nci-desc-difftest`, `nci-desc-liveness`, `nci-desc-fuzz` and `nci-baseline-difftest` hold them to
  each other and to the Python reference `lean/scripts/nci_reference.py`, run by `../tests/test_nci_*.py`;
  `../tests/test_inference_only_protocol.py` holds `inference-only serve` to that reference behind the protocol's
  decoders.

The check's agreement with `Nci.Complies` is proved: its verdict is the definition's (`DescReports`), and a
  compliance verdict witnesses the policy (`DescWitnesses`). What is open is the step from the protocol's accept to
  that verdict as a property (`baseline_desc` is a proof lemma) and from remote's verdict to the policy's accept
  (remote's rule `checkSame` has no property yet). `benchmarks/nci` holds the
literature index and the training algorithms' traits.

## Names

The directory, the namespaces `Nci` and `NciCheck`, the `nci-*` executables, `benchmarks/nci` and the Lean's
docstrings keep the older names until they are renamed: "NCI" and "novel communication intensity" for the policy,
"isolation unit" and "IU" for an inference block, "novel information" and `novelReads` for fresh input, and
"baseline" for the registered information (`GLOSSARY.md`, Older names).
