# Security

The Lean statement and proof of what an accepted run of Verity guarantees: for each subprotocol, and then for the composed
protocol, a bound on the probability that the verifier accepts while a claim about the run fails. The terms are the
[Glossary](../GLOSSARY.md)'s. The first section is the whole security argument, from the hardware to the slowdown a
secret training run suffers, with the symbols every layer uses and where each step stands in Lean.

## The security argument

In five sentences:

1. The hardware computes at most Φ work a tick (the device model).
2. When the verifier accepts, then except with probability δ, almost all registered capacity was spent computing the
   declared circuit correctly (sampled proofs and PoUW), and the declared circuit divides into small inference blocks
   that read little fresh input (the inference-only policy).
3. A training run therefore gets only the capacity left over, plus whatever training the declared work does by itself.
4. A token's training work must be fed that token's activations, so a block that reads little fresh input does little
   of it (the token structure).
5. Against a reference run, every successful training run then needs at least 1/(α · the leftover share) times the
   fleet-time: the slowdown ρ.

The numbers come from the slowdown calculator (`benchmarks/rho`), not from this file.

### Objects

- **Time.** A **tick** is one step of the one clock (`OneClock`, a named premise). A **monitoring period** J is a span
  of consecutive ticks, |J| long; every other span is a block's.
- **Hardware.** Leaves L are devices; a location X is a set of leaves. The **site** is the registered hardware, public;
  hardware outside it is unregistered. Φ_X is X's **capacity**, work a tick at its rated peak (the site entry: SKU,
  clock, rate per operand class). Φ_R and Φ_U are the registered and unregistered capacities, Φ = Φ_R + Φ_U.
- **The actual circuit** A: every gate the hardware computed, hidden work included, each gate g at a leaf x(g) and a
  tick t(g); A[X, I] is its gates at X during the ticks I.
- **Registration**, once and append-only: the site, the registered information θ (input gates whose values are
  committed before J: models, code, data) and the circuit builder B, a public program.
- **Execution.** For each declared advice a_j (a string of at most ℓ_a bits, private and committed, ℓ_a public), B
  outputs B(a_j) = (E_j, 𝒫_j, 𝓘_j, loc_j):
  - E_j, the execution block: a Boolean circuit whose inputs are θ, earlier blocks' outputs and fresh input;
  - 𝒫_j, a partition of E_j's gates into proof blocks, each block's outputs within its public size class and their
    number within a public cap;
  - 𝓘_j, a partition of E_j's gates into inference blocks, each of c gates reading at most k bits of fresh input,
    which is anything outside θ;
  - loc_j : gates(E_j) → registered devices × ticks, the placement.

  a_j, E_j, 𝒫_j, 𝓘_j and loc_j are private, their padded counts public. The declared circuit is D = E_1 ‖ … ‖ E_n.
- **Derived quantities**, none assumed:
  - F, the faulty proof blocks (committed outputs that differ from their gates applied to their committed inputs), and
    the **faulty share** φ = |⋃F| / |D|;
  - Φ_cred, the rate PoUW credits as declared work, and the **declared utilization** u = Φ_cred/Φ_R;
  - Φ_gap, credited work a cheaper route could produce; Φ_bk, bookkeeping (certificate slack and verdict lag);
  - Φ_D, the rate of correct credited declared work;
  - q(K), the most counted multiplications a subcircuit reading K fresh bits computes (Layer 4);
  - ζ = M_ref/C_ref, the reference run's counted multiplications per unit of capacity-time (Layer 4).

### The protocol

Remote, for each execution block E_j, in five steps:

1. **Commitment.** The developer commits E_j's values under hiding leaves (every value crossing a proof-block boundary,
   its inputs and outputs), a_j, 𝒫_j and 𝓘_j, before any of E_j's coins.
2. **Challenge.** The challenger, on an auditor host, draws live coins: the sample of proof blocks (by size class) and
   PoUW's noise and check coins.
3. **Proof.** The developer proves each sampled proof block in zero knowledge and answers PoUW's challenges in time.
4. **Validation.** The verifier checks the proofs, the partitions' public rules, 𝓘_j's budgets (c, k) and PoUW's
   credit.
5. **Accounting.** Each subprotocol's account for E_j is tallied: the bound on faulty proof blocks, the credited work,
   the inference blocks.

The monitoring period is accepted when every subprotocol accepts every E_j and all committed to one registration.
Onsite adds four subprotocols on devices at the site: memory wipes cut each registered server's life into server
blocks, each carrying at most k_wipe bits across its boundary; network certification bounds each server's ingress to
r_in bits a tick, besides side channels at r_side; and PoUS and PoUB bound the memory and bandwidth used.

### Layer 1: the device model

What the hardware can do, the feasible set 𝓕:

- compute: |A[X, I]| ≤ Φ_X·|I| for every location X and span I; hosts carry no compute;
- links, memory and causality, which onsite reads;
- the inventory: Φ_U/Φ is at most a stated share. No registration covers unregistered hardware, so this is an
  assumption about the world.

### Layer 2: protocol soundness

When the verifier accepts, the run lies in the admissible set 𝒜 = 𝓕 ∩ ⋂_p 𝒜_p, one 𝒜_p per subprotocol, except with
probability δ = Σ_p δ_p plus the cryptographic terms (`Verity.Soundness`; its subprotocols' guarantees are the table
under "The guarantees to read first"). What the slowdown reads from 𝒜:

- **sampled proofs:** F lies in the confidence set M_δ, so φ ≤ φ_δ ≈ ln(1/δ)/n, n the proof blocks proved in J (set by
  the proof budget, a share of compute, and the prover overhead γ);
- **PoUW:** Φ_D ≥ (1 − φ)·Φ_cred − Φ_gap − Φ_bk;
- **the inference-only policy:** every 𝓘_j is a (c, k)-partition, the bits faulty proof blocks inject, k_F, counted as
  fresh input;
- **one registration** across the subprotocols;
- **onsite:** each server block reads at most k_S = k_wipe + (r_in + r_side)·Δ_wipe fresh bits.

### Layer 3: accounting

In a run in 𝒜, split A's gates into correct credited declared work and the rest. The rest is **free capacity**, at most
(Φ_U + Φ_R − Φ_D)·|J|: unregistered hardware, registered hardware not running credited declared work (1 − u), the
faulty proof blocks, the credit gap and the bookkeeping.

### Layer 4: the task and its assumptions

The **task** 𝒢 is to retrain the latest frontier model from scratch: reach a loss below every registered model's, from
a random initialization and θ, within J, with probability above η. A **counted multiplication** is a product of a weight
of the model being trained and an **activation**, one of a token position's values at one training step (a forward
value or a gradient). The assumptions name no protocol:

- **T0, reference.** A known successful computation does M_ref counted multiplications in capacity-time C_ref, and runs
  on the whole fleet in fleet-time C_ref (it parallelizes to the fleet: C_ref ≥ Φ·ℓ, ℓ its serial depth).
- **T1, work floor.** Every successful computation does at least M_ref/α counted multiplications. This is the one
  assumption about how much work the task needs. It covers every algorithm, including ones that read registered models'
  outputs, such as the declared circuit's served outputs used as training targets.
- **T2, rate ceiling.** No computation does them faster than the reference per unit of capacity: at most ζ·Φ_X a tick
  at X. Free capacity counts at the reference's efficiency, not at peak.
- **T3, token structure.** Every successful computation's counted multiplications have this structure, for every
  subcircuit reading K ≤ K_max fresh bits: each activation feeds at most a_out of them (**fan-out**); each output sums at
  least a_in of them (**fan-in**), so a subcircuit must read a_in of a token position's activations before it finishes
  any of that position's outputs; each weight is used at most reuse times per token position; and weights and
  activations carry at least b_w and β bits that can't be compressed away. Its range K_max is about N*·b_w, the weight
  state of one token's active path, N* the active parameters per token: a subcircuit reading more can hold a whole path,
  and T3 fails for it.

Under T3 a subcircuit reading K fresh bits does at most

$$ q(K) = \frac{K}{\beta}\,\max\!\Big(a_{\mathrm{out}},\ \frac{\mathrm{reuse}\cdot K}{4\,a_{\mathrm{in}}\,b_w}\Big) $$

counted multiplications (`Nci.Guarantees.ThreadedReads`, where a_in and a_out are the threading's d and e).

### Layer 5: slowdown

The **slowdown** ρ(𝒢, η) is T_𝒢(η; 𝒜)/T_𝒢(η; 𝓕), T_𝒢(η; ·) the least fleet-time at which some computation in the set
succeeds with probability above η, and **fleet-time** elapsed time times Φ, registered and unregistered.

**Remote.** Under the device model, T0–T3 with k ≤ K_max, and acceptance, except with probability δ:

$$ \rho \;\ge\; \frac{1}{\alpha}\cdot\frac{\Phi}{\Phi_U + \Phi_R - \Phi_D + \hat Y}, \qquad
   \hat Y = \frac{|D|}{c}\cdot\frac{q(k + k_F)}{\zeta\,|J|}. $$

1. Layer 2 puts the run in 𝒜.
2. Take any successful computation inside A. Its counted multiplications inside correct declared work number at most
   Σ_I q(k + k_F) over the inference blocks, ζ·Ŷ·|J| (T3, the policy, and `Nci.Guarantees.ThreadedThrottledDark`).
3. Those outside it number at most ζ·(Φ − Φ_D)·|J| (Layer 3 and T2).
4. By T1 the total is at least ζ·C_ref/α, so T_𝒢(η; 𝒜) ≥ Φ·C_ref/(α·(Φ − Φ_D + Ŷ)).
5. By T0, T_𝒢(η; 𝓕) ≤ C_ref. Divide.

**Onsite.** As remote, with each registered server s's contribution capped per server block by T3 at K = k_S, which
needs k_S ≤ K_max:

$$ \rho \;\ge\; \frac{1}{\alpha}\cdot\frac{\Phi}{\Phi_U + \sum_s \min\!\big(q(k_S)/(\zeta\,\Delta_{\mathrm{wipe}}),\ 
   \Phi_s - \Phi_{D,s} + \hat Y_s\big)}. $$

When k_S exceeds K_max, a server block can hold a token's whole path, the cap is void and onsite gives remote's bound.
Without T3 the declared yield Ŷ could approach Φ_D and both bounds fall to 1/α, no slowdown at all: the token structure is what makes
declared work not training.

### Where each step stands

| Step | Lean | Status |
|---|---|---|
| The actual circuit | `Accounting.Placed`, `Accounting.Placed.region` | Defined |
| Declared objects | `Verity.Circuit`, `Verity.Partition`, `Verity.Partition.IsCK`, `Verity.Declaration`; the site, B, a_j and loc_j not yet in Lean | Partly defined |
| Device model | `Accounting.Placed.ComputeRate`, `Accounting.Placed.Links`, `Accounting.Placed.Memory`, `Accounting.Placed.Causal`, `Accounting.Overhead.physClass`; hosts carry no compute, `OneClock` and the inventory not yet in Lean | Defined; three premises unstated |
| The composites | `Core.Remote.protocol`, `Core.Onsite.protocol`, `Properties.Composition.AcceptMeans` | Proved; the policy is not yet a subprotocol of either, and neither checks one registration |
| Protocol soundness | `Accounting.Overhead.restricted`; `Verity.Soundness` not yet in Lean | Open |
| Sampled proofs | `Flock.Guarantees.SampledKSLiveV3`, `Claims.Sampling.RelS` | Proved; the step from F ∈ M_δ to φ ≤ φ_δ is open |
| PoUW | `Accounting.Guarantees.Compose.CircuitCredit` | Proved; nothing derives Φ_D from it |
| The inference-only policy | `NciCheck.Properties.SecurityProofs.DescWitnesses`, `Nci.Complies` | Proved; k_F, what faulty blocks inject, is unbounded |
| Onsite's k_S | `Pous.Guarantees.StreamedWipe`, `NetTiming.Guarantees.PeriodLeakAccepted` | Proved per window; nothing composes them into k_S |
| Accounting | the split of a run's counted multiplications not yet in Lean | Open |
| T0–T2 | the inline hypotheses of `Nci.Guarantees.DampenerRemote`; named Props not yet in Lean | Open |
| T3 and q(K) | `Nci.Threading.Threaded`, `Nci.Assumptions.WeightsIncompressible`, `Nci.Guarantees.ThreadedReads`, `Nci.Guarantees.ThreadedThrottledDark` | Proved for the circuit in hand; stated over every successful computation, with its range, not yet in Lean |
| Slowdown | `Accounting.Overhead.overhead`, `Accounting.Overhead.threshold`, `Accounting.Overhead.cost` | Defined, with cost in server-time, not fleet-time; no theorem reads it |
| The bounds | `Nci.Guarantees.DampenerRemote`, `Nci.Guarantees.DampenerThreaded` | Proved as arithmetic over inline hypotheses, not over the definition above |

## Three trees

- `Definitions/` (package `security`, with `Models.lean`): the spec, the part a reader reads. Each area (`Core`, `Pouw`,
  `Pous`, `NetworkCertifier`, `Accounting`, ...) has its vocabulary, its named assumptions (`Assumptions`) and its
  guarantee statements (`Guarantees`), each a `def G : Prop`.
- `Properties/` (package `security_properties`): statements about the code's Lean functions, the checks each component
  runs, one `Properties.<Protocol>` module per component.
- `Proofs/` (package `security_proofs`): every proof, `<Area>.SecurityProofs.G : <Area>.Guarantees.G` for each
  guarantee, and the lemmas behind them.

What may import what (`tests/test_lean_packages.py`): `security` requires only Mathlib; `security_properties` requires
`security` and the code it states properties of, never `security_proofs`; `security_proofs` requires both, the code and
ArkLib; and no code package requires anything here. The lock's `layers` say which spec modules each module prefix may
import (the spec imports only itself and Mathlib), and its `trees` forbid a statement from reading, or a definitions or
properties module from importing, anything in `Proofs/`. C-Flock is the exception until its spec is extracted: its
guarantees and assumptions are still in `Proofs/Flock`, which `trees` lists by module.

## The lock

Each package's `lean-audit.json` is its lock. For each guarantee it records the signature, the named assumptions among
its hypotheses and a SHA-256 hash of its type, and, per module, every definition the statements read with a digest. The
Lean check (`tools/verity/lean/check.py`) builds every file, replays every declaration through Lean's kernel, and fails
when:

- a declaration uses an axiom other than `propext`, `Classical.choice` and `Quot.sound` (so no `sorry`, `axiom` or
  `native_decide`), or an option that skips the kernel;
- a guarantee has a `Prop` hypothesis that is not a named assumption: a `Prop` listed by name in the lock, with an owner
  and its definition's digest;
- a statement reads a definition outside the spec, or breaks `layers` or `trees`;
- a claim definition lacks its witnesses: a theorem that it fails at a toy instance, and one per deployed record that its
  checkable antecedents hold there;
- a record no longer matches the build.

A changed record means a guarantee now says something else. When one lands, the change is shown, the statement before
and after and each changed definition with its source; nothing waits on it, and a guarantee that only moved shows none.

## The guarantees to read first

**Protocol soundness, `Verity.Soundness`: not yet in Lean on `main`.** It will be one theorem over all the subprotocols:
the verifier accepts while a claim of the period fails with probability at most their errors summed, each subprotocol's
guarantee entering as a target until that subprotocol's own proof discharges it. The subprotocols are the table's first
six rows, memory wipes among them; PoUB's row is outside it. Neither composite runs the bandwidth audit over PoUS's store
or memory wipes yet, and onsite composes PoUB instead (below). Its building block on `main` is `Verity.Sound`
(`Definitions/Core/Game.lean`).

The table lists what it rests on, then one guarantee outside it; each is proved under `Proofs/`. A status names the statement's named assumptions; the
conditions it places on its inputs and setup, which the lock also lists, are left out.

| Subprotocol | Guarantee | Status |
|---|---|---|
| Sampled proofs | `Flock.Guarantees.SampledKSLiveV3`, stated in `Proofs/Flock/Bridge/Refused.lean`, proved in `Proofs/Flock/Bridge/KSLive.lean` | Proved under named assumptions: `UniformRandomBytes`, `RecordCustodyZKJ`, and SHA-512's collision resistance for its finders, `TableCRZAt` and `LinkCRZC` |
| PoUW | `Properties.Pouw.Period.H1CheckedCreditPeriod`, the instance at the deployed H1 tile unit of the general `CheckedCreditPeriod` (both in `Properties/lean-audit.json`), `Properties/Pouw/Period.lean` | Proved under named assumptions: `ServiceSound`, `DrawIsCoin`, `SaltIsLiveCoin`, `RegisteredBeforeDraw`, `StagedPins`, `UnitFixed`, `OperandHMu`, `ReceiptLuckAt`, `GuessLuckAt`. The deployed path doesn't meet `SaltIsLiveCoin` yet, and no served path meets `AcceptedLive` (credit only after `flock-verify --zk --registered` accepts) |
| PoUS | `Pous.Period.PeriodHeld`, `Properties/Pous/Period.lean` | Proved under named assumptions: `CoinCommitHides`, `RingDevice`, and `P2RingAuditSecure`, an open conjecture |
| Bandwidth audit | `Bandwidth.Guarantees.AuditRO`, `Definitions/Accounting/Guarantees/Bandwidth.lean` | Proved, in the random-oracle model |
| Network certification | `NetTiming.Guarantees.PeriodLeakAccepted`, `Definitions/NetworkCertifier/Guarantees/Period.lean` | Proved under named assumptions: `TrustedHost`, `Claims.Flock.ListWithin`, and sampled proofs' `Claims.Sampling.KS` and `Claims.Sampling.Commitment.Binds`; each window is bounded alone, with no union over the period yet |
| Memory wipes | `Pous.Guarantees.StreamedWipe`, `Definitions/Pous/Guarantees/StreamedWipe.lean` | Proved; the PRF enters as its advantage term |
| PoUB | `Properties.RemoteAudit.Line.LineHeld`, `Properties/RemoteAudit/Line.lean` | Outside protocol soundness for now, that audit's own guarantee. Proved under named assumptions: `Accounting.Remote.Held.Assumptions`, `Accounting.Remote.Draw.Assumptions` |

PoUW has other guarantees, outside protocol soundness.

## The remote protocol in Lean

Remote (`verity/core/protocols/remote`, `Core.Remote.protocol`) is the composite for an auditor with no devices at the
site. It composes two subprotocols, sampled proofs' one-stage audit and PoUW's window check, and accepts only when both
accept and the commitment binds both (`Properties.Composition`, proved). Onsite adds PoUS, PoUB and network certification; memory wipes are a component of neither yet.

- **Guarantees:** the table's sampled-proofs and PoUW rows, both proved. The composed statement, `Verity.Soundness`, is
  not in Lean yet (above), so the two compose by hand, and nothing ties the deployed verdicts to the theorems: on
  `main`, remote's sampled-proofs component is one-stage v1, which takes the sessions' verdicts as evidence; the
  verifier's run of `flock-verify --zk` produces them, and the link from remote's acceptance to that run's comes with
  v2.
- **Assumptions:** those rows' named assumptions, `UniformRandomBytes`, `RecordCustodyZKJ`, `TableCRZAt`, `LinkCRZC`,
  `ServiceSound`, `DrawIsCoin`, `SaltIsLiveCoin`, `RegisteredBeforeDraw`, `StagedPins`, `UnitFixed`, `OperandHMu`,
  `ReceiptLuckAt` and `GuessLuckAt`. Trust model: nothing the prover runs or hosts is trusted, and every coin is live,
  with no Fiat–Shamir and no retries. The challenger and the verifier both run on auditor hosts. The challenger, which
  draws each coin after the commitment it challenges, is the one auditor-trusted device remote relies on
  (`trusted-host`); the verifier is the auditor's own check. Two premises about where the challenger runs,
  `CoinsConfidential` and `OneClock`, are prose only (`verity/core/protocols/sampled_proofs/README.md`): no `.lean`
  file defines either. Network certifiers and memory challengers are onsite's.
- **Not PoUB.** `Accounting.Remote` and `Properties.RemoteAudit` are PoUB, the proof of useful bandwidth
  (`verity/core/protocols/remote_bandwidth`), which proves the bits a node read from its registered store. It is a
  different protocol from the remote composite, and a component of onsite only.

## Building and checking

With the toolchain `lean-toolchain` pins:

~~~text
lake build                                    # in Security/: the spec
lake build                                    # in Security/Properties/, then Security/Proofs/ (fetches ArkLib)
python3 tools/verity/lean/check.py Security   # from the repository root
~~~

The check audits `Proofs/` with the spec, from one build, and refuses a machine below its memory floor.
