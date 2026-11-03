# The inference-only policy

The inference-only policy (c, k, n) asks that the registered circuit D have a (c, k)-partition 𝓘: inference blocks I,
each of at most c gates and reading at most k bits beyond the registered inputs θ, the models, code and data fixed
before the monitoring period (|I| ≤ c and |in(I) ∖ θ| ≤ k), at most n blocks a period (`GLOSSARY.md`, Notation). It
is a protocol with no rounds. Setup fixes θ and the registered compiler B. At the start of each verification phase,
before any coin, B's output on the reported advice partitions that execution block into inference blocks, and the
developer's ZK proof that B ran correctly and explains the committed trace can also show that each meets the policy
(`verity/core/protocols/remote/README.md`, "Lifecycle"). In Lean it is `Nci.Complies`, in an older form (below), and its
check is `NciCheck`.

The inference-only policy is not yet a subprotocol of the composites: neither onsite nor remote
(`verity/core/protocols/composition`) runs its check today.

## Why it separates inference from training

An inference episode (a chat session) reads little beyond θ, say 100 KB of prompt tokens, so honest serving fits blocks
of a small k, one block a session at slowdown ρ = 1 (Fit; `Nci.Guarantees.Inference` in the older form). A training
episode (an RL rollout) computes new weights θ′, say 1 TB of new parameters, and every gate downstream of the update
depends on them: under A1, a block doing a share x of one token's affected work reads at least u(x) update bits, plus
the cut cost a for each token it enters mid-path, so training through blocks of a small k is slow (Slowdown), and under
A1, A1′ and A2 reaching a target loss covertly costs at least the honest cost H(ℓ*) times min ρp (Security). The symbols and
statements are `GLOSSARY.md`'s (Notation, Training). [`benchmarks/rho`](../../../../../benchmarks/rho/README.md) still
computes the earlier statement, which the training formalism replaces; its scenarios register a policy as if it were a
subprotocol.

## The Lean

- **The definition**, `Security/Definitions/Nci.lean`. `Nci.Complies Γ α` states an older form: D divides into blocks
  of at most Γ gates, each reading at most α bits of fresh input per gate it holds (`Nci.novelReads`), so at most αΓ
  bits a block. The (c, k) form, with an exact gate budget, is not yet in Lean.
- **The guarantees**, `Security/Definitions/Nci/Guarantees/`, each proved in `Security/Proofs/Nci/` as
  `Nci.SecurityProofs.G`. `ReadBound` turns a lower bound on fresh input into a lower bound on size, `Inference` shows
  honest inference complies, and the covering, training and threading statements belong to the earlier statement of
  the training slowdown, which the training formalism replaces. `Nci.Guarantees.DampenerRemote` and its relatives state
  the remote composition's slowdown in that earlier form.
- **The registered information's provenance**, `Nci.Assumptions.BaselineRegistered`: every θ entry is registered
  public data, never an earlier run's output.

## The check

`lean/` is the Lake package `nci_check`, code that reads Programs with C-Flock's decoder (`Flock.Program`) and imports
nothing else but Lean core.

- `nci-check` runs `NciCheck.Baseline.check`: `NciCheck.Desc.check`, which decides `Nci.Complies Γ α` for a declared
  division aligned to the root's Calls at the cost of the descriptor (one summary per Definition), then each θ entry's
  provenance against the deployment's registration log.
- `nci-check --unrolled` runs the same with `NciCheck.check`, the check gate by gate, kept as the reference; its input
  format is in `NciCheck.Check`'s docstring. `nci-check --policy` runs the policy's check alone, without θ's
  provenance, which is what `benchmarks/verifier_cost` measures.
- `nci-difftest`, `nci-desc-difftest`, `nci-desc-liveness`, `nci-desc-fuzz` and `nci-baseline-difftest` hold them to
  each other and to the Python reference `lean/scripts/nci_reference.py`, run by `../tests/test_nci_*.py`.

The check stays experimental until its agreement with `Nci.Complies` is proved. `benchmarks/nci` holds the
literature index and the training algorithms' traits.

## Names

The directory, the namespaces `Nci` and `NciCheck`, the `nci-*` executables, `benchmarks/nci` and the Lean's
docstrings keep the older names until they are renamed: "NCI" and "novel communication intensity" for the policy,
"isolation unit" and "IU" for an inference block, "novel information" and `novelReads` for fresh input, and
"baseline" for the registered information (`GLOSSARY.md`, Older names).
