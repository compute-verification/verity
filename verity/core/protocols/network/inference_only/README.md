# The inference-only policy

The inference-only policy asks that the registered circuit D have an inference-block partition 𝓘: blocks with gate
budget c (at most c gates each) and fresh-input budget k (at most k bits of fresh input each), where fresh input is
every value outside the registered information θ, the models, code and data fixed before the monitoring period. It is
a protocol with no rounds. Setup fixes θ and the registered compiler B. At the start of each verification phase, before
any coin, B's output on the reported advice partitions that execution block into inference blocks, and the developer's
ZK proof that B ran correctly and explains the committed trace can also show that each has at most c gates and reads
at most k bits outside θ (`verity/core/protocols/remote/README.md`, "Lifecycle"). In Lean it is `InferenceOnly.Complies`; its
check, `InferenceOnlyCheck`, decides the policy at a rate, `InferenceOnly.CompliesRate` (below).

The inference-only policy is a subprotocol of both composites, onsite and remote (`verity/core/protocols/composition`):
`InferenceOnlyCheck.Protocol.protocol` (`verity/inference-only/v0`, the executable `inference-only`) is its check behind the core
interface. Its params are the registered circuit, the budgets and the registration log before the run; its commitment is
the run's own registration record, which must name the registered circuit and which each composite requires to be the
record sampled proofs commits to (`Core.Composition.same`); its evidence is the division with θ and θ's
provenance. That evidence reaches the verifier in the clear: the check doesn't run inside a zero-knowledge
proof yet.

## Why it separates inference from training

An inference episode (a chat session) reads little fresh input, say 100 KB of prompt tokens, so honest serving
divides into inference blocks with a small fresh-input budget (`InferenceOnly.Guarantees.Inference`). A training episode (an RL
rollout) reads an enormous amount, say 1 TB of new parameters, so training that runs through inference blocks gets
only their yields q(c, k): a pricing argument says how much of a block's fresh-input budget each counted object of a
necessary subcomputation needs. The slowdown lower bound sums those yields with free compute, at the hardware's rate,
and with the server-block yields that the memory wipe, proof of useful space (PoUS) and network certification bound (`GLOSSARY.md`, "Size, credit and
the slowdown lower bound"). Its numbers come from [`benchmarks/slowdown`](../../../../../benchmarks/slowdown/README.md), whose
scenarios register a policy as if it were a subprotocol.

## The Lean

- **The definition**, `Security/Definitions/InferenceOnly.lean`. `InferenceOnly.Complies c k` says D has a (c, k)-partition
  (`Verity.Partition.IsCK`): every block at most c gates and at most k fresh input bits (`InferenceOnly.novelReads`).
  `InferenceOnly.CompliesRate c α` is the form the check reads: blocks of at most c gates, each reading at most α fresh input bits
  per gate, so at most αc a block. Where the policy's guarantees assume the policy, they assume `InferenceOnly.CompliesRate`, what
  the check checks; `InferenceOnly.Guarantees.Inference` concludes `InferenceOnly.Complies`. A (c, k)-partition whose blocks have exactly c
  gates complies at rate α = k/c (`InferenceOnly.SecurityProofs.compliesRate_of_complies`, which takes that tiling as a
  hypothesis). The way back is proved too: on a Boolean declaration a partition complying at rate α is
  a (c, ⌊αc⌋)-partition (`InferenceOnly.Guarantees.CompliesCK`), and `InferenceOnly.Guarantees.ExpandComplies` carries the rate through the
  bit-level expansion at c·W for gates up to W bits wide. With its constants registered
  (`Verity.RegisteredCircuit.constInputs`) a declaration needs neither Boolean gates nor positive fan-in for those
  partitions (`InferenceOnly.SecurityProofs.constInputs_isCK`, `constInputs_expand_isCK`), which is how the policy's acceptance
  gives them (`Properties.Remote.PolicyMet`).
- **The guarantees**, `Security/Definitions/InferenceOnly/Guarantees/`, each proved in `Security/Proofs/InferenceOnly/` as
  `InferenceOnly.SecurityProofs.G`. `ReadBound` turns a lower bound on fresh input into a lower bound on size, `Inference` shows
  honest inference complies, and the covering, training and threading statements are pricing arguments.
  `InferenceOnly.Guarantees.SlowdownRemote` and its relatives state the remote composition's overhead at (c, k).
- **The count of (c, k)-inference blocks**, `InferenceOnly.CompliesAtMost c k n` (`Security/Definitions/InferenceOnly.lean`): D's
  computed gates divide into at most n proof blocks of a (c, k)-partition (`Verity.Partition.IsCK`,
  `Security/Definitions/Core/Formalism.lean`).
- **The registered information's provenance**, `InferenceOnly.Assumptions.BaselineRegistered`: every θ entry is registered
  public data, never an earlier run's output.

## The check

`lean/` is the Lake package `inference_only_check`, code that reads Programs with Flock's decoder (`Flock.Program`) and imports
nothing else but Lean core and the core interface (`core_protocol`).

- `inference-only-check` runs `InferenceOnlyCheck.Baseline.check`: `InferenceOnlyCheck.Desc.check`, which decides `InferenceOnly.CompliesRate c α` for a declared
  division aligned to the root's Calls at the cost of the descriptor (one summary per Definition), then each θ entry's
  provenance against the deployment's registration log.
- `inference-only-check --count` runs `InferenceOnlyCheck.Count.check`, the count of (c, k)-inference blocks: the declaration names its block
  Definitions by descriptor key, and the check gives each one's Boolean gates (from a per-primitive gate table), its
  fresh input bits outside θ and its blocks, multiplied through batches and scans, and the root's total against n.
  - Public: the judgment against n, `{"complies": true|false}`, or the refusal (`{"error": code}`, with a `message`
    for some codes).
  - Not public: the rows (each block Definition's gates, fresh bits and blocks), the total and which block failed.
    `inference-only-check --count --detail` prints that whole verdict (`InferenceOnlyCheck.Count.checkDetail`) for the private record
    only.

  `InferenceOnlyCheck.Properties.DescCountsCK` (`Security/Properties/InferenceOnly/Count.lean`) states that a published `complies` means
  `InferenceOnly.CompliesAtMost c k n`; it is not yet proved, and the lock lists it as an open piece. `inference-only-count-difftest` holds
  both forms to the Python reference's `check_count` and `check_count_detail` (`lean/scripts/inference_only_reference.py`), run by
  `../tests/test_inference_only_count_difftest.py`.
- `inference-only-check --unrolled` runs the same with `InferenceOnlyCheck.check`, the check gate by gate, kept as the reference; its input
  format is in `InferenceOnlyCheck.Check`'s docstring. `inference-only-check --policy` runs the policy's check alone, without θ's
  provenance, which is what `benchmarks/verifier_cost` measures.
- `inference-only id | check | serve` runs `InferenceOnlyCheck.Baseline.check` as the core protocol `verity/inference-only/v0`
  (`InferenceOnlyCheck.Protocol`), the subprotocol of onsite and remote.
- `inference-only-difftest`, `inference-only-desc-difftest`, `inference-only-desc-liveness`, `inference-only-desc-fuzz` and `inference-only-baseline-difftest` hold them to
  each other and to the Python reference `lean/scripts/inference_only_reference.py`, run by `../tests/test_inference_only_*.py`;
  `../tests/test_inference_only_protocol.py` holds `inference-only serve` to that reference behind the protocol's
  decoders.

The check's agreement with `InferenceOnly.CompliesRate` is proved: its verdict is the definition's (`DescReports`), and a
compliance verdict witnesses the policy at its rate (`DescWitnesses`); on a verdict of the unrolled check that complies,
every block meets the (c, k) budgets at c = `gamma` and k = α·c (`refines`, a lemma of `Reports`). What is open is the step from the check's
accept to that verdict as a property (`baseline_desc` is a proof lemma), and remote doesn't run the check yet. `benchmarks/nci` holds the
literature index and the training algorithms' traits.

## Names

The directory, the namespaces `InferenceOnly` and `InferenceOnlyCheck`, the `inference-only-*` executables, `benchmarks/nci` and the Lean's
docstrings keep the older names until they are renamed: "NCI" and "novel communication intensity" for the policy,
"isolation unit" and "IU" for an inference block, "novel information" and `novelReads` for fresh input, and
"baseline" for the registered information (`GLOSSARY.md`, Older names).
