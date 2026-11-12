# The inference-only policy

The inference-only policy asks that the registered circuit D have an inference-block partition 𝓘: blocks with gate
budget c (at most c gates each) and fresh-input budget k (at most k bits of fresh input each), where fresh input is
every value outside the registered information θ, the models, code and data fixed before the monitoring period. It is
a protocol with no rounds. Setup fixes θ and the registered compiler B. At the start of each verification phase, before
any coin, B's output on the reported advice partitions that execution block into inference blocks, and the developer's
ZK proof that B ran correctly and explains the committed trace can also show that each has at most c gates and reads
at most k bits outside θ (`verity/protocols/remote/verifier/README.md`, "Lifecycle"). In Lean it is `Nci.Complies`; its
check, `NciCheck`, decides the policy at a rate, `Nci.CompliesRate` (below).

The inference-only policy is a subprotocol of both composites, onsite and remote (`verity/protocols/remote/verifier/composition`):
`NciCheck.Protocol.protocol` (`verity/inference-only/v0`, the executable `inference-only`) is its check behind the core
interface. Its params are the registered circuit, the budgets and the registration log before the run; its commitment is
the run's own registration record, which must name the registered circuit and which each composite requires to be the
record sampled proofs commits to (`Core.Composition.same`); its evidence is the division with θ and θ's
provenance. That evidence reaches the verifier in the clear: the check doesn't run inside a zero-knowledge
proof yet.

## Why it separates inference from training

An inference episode (a chat session) reads little fresh input, say 100 KB of prompt tokens, so honest serving
divides into inference blocks with a small fresh-input budget (`Nci.Guarantees.Inference`). A training episode (an RL
rollout) reads an enormous amount, say 1 TB of new parameters, so training that runs through inference blocks gets
only their yields q(c, k): a pricing argument says how much of a block's fresh-input budget each counted object of a
necessary subcomputation needs. The slowdown lower bound sums those yields with free compute, at the hardware's rate,
and with the server-block yields that the memory wipe, proof of useful space (PoUS) and network certification bound (`GLOSSARY.md`, "Size, credit and
the slowdown lower bound"). Its numbers come from [`verity/protocols/inference_only/rho`](../../rho/README.md), whose
scenarios register a policy as if it were a subprotocol.

## The Lean

- **The definition**, `security/Definitions/Nci.lean`. `Nci.Complies c k` says D has a (c, k)-partition
  (`Verity.Partition.IsCK`): every block at most c gates and at most k fresh input bits (`Nci.novelReads`).
  `Nci.CompliesRate c α` is the form the check reads: blocks of at most c gates, each reading at most α fresh input bits
  per gate, so at most αc a block. Where the policy's guarantees assume the policy, they assume `Nci.CompliesRate`, what
  the check checks; `Nci.Guarantees.Inference` concludes `Nci.Complies`. A (c, k)-partition whose blocks have exactly c
  gates complies at rate α = k/c (`Nci.SecurityProofs.compliesRate_of_complies`, which takes that tiling as a
  hypothesis). The way back is proved too: on a Boolean declaration a partition complying at rate α is
  a (c, ⌊αc⌋)-partition (`Nci.Guarantees.CompliesCK`), and `Nci.Guarantees.ExpandComplies` carries the rate through the
  bit-level expansion at c·W for gates up to W bits wide. With its constants registered
  (`Verity.RegisteredCircuit.constInputs`) a declaration needs neither Boolean gates nor positive fan-in for those
  partitions (`Nci.SecurityProofs.constInputs_isCK`, `constInputs_expand_isCK`), which is how the policy's acceptance
  gives them (`Properties.Remote.PolicyMet`).
- **The guarantees**, `security/Definitions/Nci/Guarantees/`, each proved in `security/Proofs/Nci/` as
  `Nci.SecurityProofs.G`. `ReadBound` turns a lower bound on fresh input into a lower bound on size, `Inference` shows
  honest inference complies, and the covering, training and threading statements are pricing arguments.
  `Nci.Guarantees.DampenerRemote` and its relatives state the remote composition's overhead at (c, k).
- **The count of (c, k)-inference blocks**, `Nci.CompliesAtMost c k n` (`security/Definitions/Nci.lean`): D's
  computed gates divide into at most n proof blocks of a (c, k)-partition (`Verity.Partition.IsCK`,
  `security/Definitions/Core/Formalism.lean`). It is the day's bound's only premise on the circuit
  (`Nci.Cut.Guarantees.CountHE`, with (H) and (E) on the work), and `DampenerRemote` and `DampenerRemotePeriod` take
  it for each declared day.
- **The registered information's provenance**, `Nci.Assumptions.BaselineRegistered`: every θ entry is registered
  public data, never an earlier run's output.

## The check

`lean/` is the Lake package `nci_check`, code that reads Programs with Flock's decoder (`Flock.Program`) and imports
nothing else but Lean core and the core interface (`core_protocol`).

- `nci-check` runs `NciCheck.Baseline.check`: `NciCheck.Desc.check`, which decides `Nci.CompliesRate c α` for a declared
  division aligned to the root's Calls at the cost of the descriptor (one summary per Definition), then each θ entry's
  provenance against the deployment's registration log.
- `nci-check --count` runs `NciCheck.Count.check`, the count of (c, k)-inference blocks: the declaration names its block
  Definitions by descriptor key, and the check gives each one's Boolean gates (from a per-primitive gate table), its
  fresh input bits outside θ and its blocks, multiplied through batches and scans, and the root's total against n.
  - Public: the judgment against n, `{"complies": true|false}`, or the refusal (`{"error": code}`, with a `message`
    for some codes).
  - Not public: the rows (each block Definition's gates, fresh bits and blocks), the total and which block failed.
    `nci-check --count --detail` prints that whole verdict (`NciCheck.Count.checkDetail`) for the private record
    only.

  `NciCheck.Properties.DescCountsCK` (`security/Properties/Nci/Count.lean`) states that a published `complies` means
  `Nci.CompliesAtMost c k n`; it is not yet proved, and the lock lists it as an open piece. `nci-count-difftest` holds
  both forms to the Python reference's `check_count` and `check_count_detail` (`lean/scripts/nci_reference.py`), run by
  `../../../certifier/verifier/network/tests/test_nci_count_difftest.py`.
- `nci-check --unrolled` runs the same with `NciCheck.check`, the check gate by gate, kept as the reference; its input
  format is in `NciCheck.Check`'s docstring. `nci-check --policy` runs the policy's check alone, without θ's
  provenance, which is what `benchmarks/verifier_cost` measures.
- `inference-only id | check | serve` runs `NciCheck.Baseline.check` as the core protocol `verity/inference-only/v0`
  (`NciCheck.Protocol`), the subprotocol of onsite and remote.
- `nci-difftest`, `nci-desc-difftest`, `nci-desc-liveness`, `nci-desc-fuzz` and `nci-baseline-difftest` hold them to
  each other and to the Python reference `lean/scripts/nci_reference.py`, run by `../tests/test_nci_*.py`;
  `../../../certifier/verifier/network/tests/test_inference_only_protocol.py` holds `inference-only serve` to that reference behind the protocol's
  decoders.

The check's agreement with `Nci.CompliesRate` is proved: its verdict is the definition's (`DescReports`), and a
compliance verdict witnesses the policy at its rate (`DescWitnesses`); on a verdict of the unrolled check that complies,
every block meets the (c, k) budgets at c = `gamma` and k = α·c (`refines`, a lemma of `Reports`). What is open is the step from the check's
accept to that verdict as a property (`baseline_desc` is a proof lemma), and remote doesn't run the check yet. `verity/protocols/inference_only/nci` holds the
literature index and the training algorithms' traits.

## Names

The directory, the namespaces `Nci` and `NciCheck`, the `nci-*` executables, `verity/protocols/inference_only/nci` and the Lean's
docstrings keep the older names until they are renamed: "NCI" and "novel communication intensity" for the policy,
"isolation unit" and "IU" for an inference block, "novel information" and `novelReads` for fresh input, and
"baseline" for the registered information (`GLOSSARY.md`, Older names).
