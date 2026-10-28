# The remote protocol

The remote protocol audits a datacenter whose auditor has no devices at the site: the developer commits everything its
hardware computes, and the auditor checks it from outside with live coins and a Lean verifier. It composes three
subprotocols into one verdict over one commitment: sampled proofs (the committed values satisfy the declared circuit,
except on few proof blocks), PoUW (the hardware's time went to the declared work) and the inference-only policy (the
declared circuit is inference, not training).

**The open gap.** Remote's composed soundness theorem, `Verity.Soundness`, is not in Lean yet. It is due after the Lean
dedupe (one vocabulary). Today the pieces below compose by hand: each is proved on its own,
and no Lean theorem states remote's guarantee from them. Nothing yet ties the deployed verifier's verdicts to those
theorems either, because remote reads sampled proofs' one-stage v1, which takes the sessions' verdicts as evidence
instead of computing them.

This file is an outline with signposts. The Lean is the definition. The vocabulary is the Glossary's: `ARCHITECTURE.md`'s
Glossary, and `GLOSSARY.md` for the notation and the older names. The lifecycle's terms (a declaration, declaring, the
declared advice, a declaration block and a declaration period) are in `ARCHITECTURE.md`'s Glossary, under Declaration,
Block and Period. Lean names keep some older names until they are renamed: `Dampener*` means overhead, `*Horizon` a
monitoring period's pooled bound, `Session*` a run of a subprotocol, and `Window*` a period.

**Which "remote".** Six things in the tree are called "remote", and this page means the first:

- the remote protocol, this one: `Core.Remote` (`CoreRemote.lean`, id `verity/remote/v0`);
- PoUB, proof of useful bandwidth, an onsite subprotocol whose code still says "remote bandwidth": its package
  `verity/core/protocols/remote_bandwidth/`, its Lean verifier `RemoteAudit`, its Lean vocabulary `Accounting.Remote`
  (`Security/Definitions/Accounting/Remote.lean`) and `Accounting.Guarantees.Remote*`, its id `verity/remote-bandwidth/v0`,
  its whole-node benchmark `benchmarks/remote_session/`, and serve-openai's `--remote-verifier`, the address of PoUB's
  verifier;
- `RemoteChallenger` (`verity/core/protocols/sampled_proofs/remote.py`), the client of a challenger on another host;
- `RemoteCertifier` (`verity/core/protocols/certifier/active.py`), onsite's network certifier reached over the network;
- the overhead bound's remote regime, `Nci.Guarantees.DampenerRemote` (the inference-only policy's Lean is still named
  `Nci`) and `benchmarks/rho`'s `regime.remote`: a bound over this protocol (part 6 of the security argument below), not a part of it;
- "run remotely": a job run on a cluster node (`research run --on`), which has nothing to do with any protocol.

## Parties

- **The developer** (the prover) runs the datacenter. It registers, declares, commits its values and proves the drawn
  proof blocks behind the firewall. Nothing it runs or hosts is trusted, and neither is the audited hardware or its
  vendor.
- **The auditor** has two parts, both on auditor hosts that the developer doesn't control: its **challenger** (the
  live coins, which draw what must be proved) and its **verifier** (the Lean verifier: `flock-verify` for each drawn
  proof block, and `remote check` for the verdict). The challenger is the one auditor-trusted device remote relies on
  (`trusted-host`); the verifier is the auditor's own check. The verifier may also draw on its own coins
  (`IO.getRandomBytes`) instead of the challenger's.
- **How the coins reach the developer.** The challenger opens each coin only after the developer's commitment is
  recorded, over one TCP stream that the developer's client opens at its own `127.0.0.1` port (`RemoteChallenger`).
  Deployed, on an auditor host, that port is the far end of a coin link the auditor host opens to the
  developer's machine: `ssh -N -R` straight to that machine's sshd, relayed and counted by network certification's
  code running on the auditor host (`benchmarks/live_challenger/coin_link.py`). No coin path of remote's own, with no
  network certifier anywhere, is specified yet.
- **No auditor devices at the site.** There are no network certifiers or memory challengers. So network certification,
  PoUS and PoUB are onsite's, not remote's, and memory wipes are neither composite's yet. For network certification,
  see `Security/Definitions/NetworkCertifier` and its `Guarantees`.

## Subprotocols

| Subprotocol | What it establishes | Lean check | id |
|---|---|---|---|
| Sampled proofs ([README](../sampled_proofs/README.md)) | the committed values satisfy the declared circuit outside the proof blocks the draw misses, on the proofs themselves, which it reads as blobs | `SampledProofs.OneStageZk.protocol` (`verity/core/service/sampled_proofs`) | `verity/sampled-proofs/one-stage/v2` |
| PoUW ([README](../pouw/README.md)) | the period's credited work, in live words, is the declared work's | `PouwCheck.protocol` (`verity/core/protocols/pouw/lean/check`) | `verity/pouw/v1` |
| The inference-only policy ([README](../network/nci/README.md); its Lean is still named `Nci`, `NciCheck`) | the declared circuit divides into inference blocks within their gate and fresh-input budgets, and θ is registered information | `NciCheck.Protocol.protocol` (`verity/core/protocols/network/nci/lean`) | `verity/inference-only/v0` |

The inference-only policy's check reads only the declared circuit and the registered information, so it needs no
hardware at the site. Its commitment is the run's record, which must name the declared circuit, and remote refuses
unless it is the record PoUW's window and sampled proofs commit to (one record, below). Its evidence, the declared
division and θ's provenance, reaches the verifier in the clear, not inside a zero-knowledge proof.

The code is `CoreRemote.lean`, `protocol = sameB "verity/remote/v0" ids components` over `components =
[NciCheck.Protocol.protocol, PouwCheck.protocol, SampledProofs.OneStageZk.protocol]`, `ids` their ids: every one of the
three commits to the run's record, and once the cases fit, two records' digests reject remote whatever the subprotocols
answer (`Core.Composition.same`; the `#guard`s pin a policy on one record and the others on another). Its executable is `remote id | check PARAMS COMMITMENT EVIDENCE | serve`
(`RemoteMain.lean`). The Lake package `core_remote` is code: it has no Mathlib and nothing of `Security/`
(`lean-audit.json`). The product's rule and its difftest are the composition's ([README](../composition/README.md),
`verity/core/protocols/composition/tests/test_composition_difftest.py`).

## Lifecycle

1. **Registration**, once, before any coin. The developer registers R = (θ, D, 𝓘): the registered inputs θ, the declared
   circuit D, and D's (c, k)-partition 𝓘 into inference blocks (`GLOSSARY.md`, `Verity.Declaration`; read by the
   inference-only policy). It also registers each subprotocol's public Program and parameters, by digest.
2. **Declarations**, one or more per declaration period (a span of one or more sampling periods). A declaration is one
   act of declaring: the developer reports its declared advice (a bit string that a registered compiler compiles, its
   length charged), and the compiler appends one declaration block to the declared circuit. The block's values are
   committed under hiding leaves (hm96 rows in SHA-512 trees) before that declaration period's audit coins.
   `examples/lifecycle/README.md` ("Terms") runs these terms on a toy model.
3. **The audit**, in each sampling period. The live coins draw proof blocks. The developer proves each drawn block with
   C-Flock `--zk` behind the firewall, and the Lean verifier `flock-verify` checks each proof. PoUW's check
   (`pouw-check`) reads each PoUW period's commitment and its drawn proofs and receipts, and credits that period's work.
   No proof is retried, and a failed proof is a visible failure.
4. **The verdict**, by the composition rule (`verity/core/protocols/composition/README.md`). The composite rejects if
   any subprotocol rejects. It also rejects if the cases don't fit (unknown, missing, repeated or out of order), or if
   the commitment isn't `bind` of the subprotocols' digests. It is undecided if it binds and some subprotocol is
   undecided, and it accepts if every subprotocol accepts and it binds. On accept, it publishes each subprotocol's public
   outputs.
5. **Accounting**, per monitoring period. The accepted sampling periods' bounds pool over the monitoring period
   (`Flock.Guarantees.IntegrityHorizon` for sampled proofs, `Accounting.Guarantees.ComposePeriod.CircuitCreditPeriod`
   for PoUW), and the overhead calculator turns them into overhead (see Results).

## The security argument, end to end

**The claim.** Suppose the remote verdict accepts every verification phase of a monitoring period. Then, except with a
failure probability that, beyond the sample and SHA-512, rests on undischarged premises (What's open):

- the committed computation satisfies the registered circuit, except on few proof blocks;
- the hardware's credited time went to the registered circuit's work, on part 3's premises, which the deployed path
  doesn't meet;
- every execution block's inference partition divides its gates into inference blocks of c gates, each reading at most
  k fresh bits (the inference-only policy, part 4), with the link to the first two claims open.

Conditionally on the items under What's open below, a third claim follows: training a new frontier model costs more
GPU-hours than it would without the protocol, by the overhead lower bound.

The argument has seven parts. **Each part is proved, stated or open on its own: no Lean theorem yet chains one part's
conclusion into the next part's hypothesis.** The one partial link is `Accounting.Guarantees.TS.SessionSoundAt`, which
consumes part 1's theorem for PoUW's proofs. The links that are missing are listed under What's open. Each part gives
what it establishes, what it assumes, its Lean and that Lean's status:

- **proved**: its proof is in a lock (`<Area>.SecurityProofs.G : <Area>.Guarantees.G` in
  `Security/lean-audit.json`, or a property's proof in `Security/Properties/lean-audit.json`);
- **stated**: its statement is in a lock, with no proof yet;
- **open**: not in Lean.

Flock's guarantee statements (`Flock.Guarantees.*`) are in `Security/Proofs/Flock/`, not `Definitions/`: its spec
isn't extracted yet.

The derivations are in `Security/README.md`'s security argument; every named `Prop` is listed under Assumptions, by
name, below.

### The assumptions, in plain words

These are the argument's assumptions by kind; Assumptions, by name, below, lists every named `Prop`. Everything
else is a theorem or a named gap.

| Kind | Assumption | Lean | How it's tested, or why it's believed |
|---|---|---|---|
| Cryptography | SHA-512 is collision resistant, and the commitments hide their values. | `cr/sha-512`, `ecr/sha-512` (`TableCRZAt`, `LinkCRZC`, `RowsCRZC`); hm96's hiding | Standard; no attack known. |
| The auditor's devices | The challenger runs on a host the developer doesn't control, its coins are uniform, and they reach the developer only after the commitment. | `Verity.Assumptions.TrustedHost`, `RecordCustodyZKJ`, `UniformRandomBytes`; `CoinsConfidential` (prose only) | Inspect the deployment: where the challenger runs, and its coin path. |
| The hardware | Each GPU computes at most at its data sheet's rates, hosts carry no hidden compute, and unregistered hardware is at most a stated share of the fleet. | `Accounting.Assumptions.TS.Sheet*`; the share is a hypothesis of `DampenerRemote` | Measure the cards and inventory the fleet. The share is a claim about the world, not the protocol. |
| Matrix products | Nobody multiplies the protocol's random noised matrices faster than the known way. Only noised calls are credited: the premise is over random operands, never ones the prover chooses. | `OperandHMu`, `FoldOpt`, `FillOpt` | A faster algorithm for random matrix products refutes it. |
| PoUW's receipts | A skipped round passes its receipt, or a guessed fold state passes, only at the stated rate. Conjectured, as their docstrings say. | `ReceiptLuckAt`, `GuessLuckAt` | Believed because each receipt folds committed accumulator words under a key fixed after the commitment; search for a skipped round or a fold-state guess that passes more often than the rate. |
| Enforcement | Sampled proofs catch each faulty proof block with probability at least q, whatever came before, and a detected cheat ends its identity and takes its hardware out of use within Δ. | `Nci.Assumptions.SampledDetection`, `Nci.Assumptions.TerminalDetection` | The first is what part 2's draw gives, not yet derived from it in Lean. The second is the auditor's policy: check that a refusal triggers it. |
| The baseline | The data the registration names was public, not computed off the books before. | `Nci.Assumptions.BaselineRegistered` | Check the registered data's provenance: each named dataset or model is public, by its digest. |
| Learning, 1: fresh weights are hard to hold | After enough training since registration, a short program that runs no longer than a block may run can't reproduce many freshly trained weight values: each costs about p bits, even to a program that knows the registered model and may replay a little training. | `Nci.Assumptions.DescriptionHard` | Compress the update: on adversarially chosen sets of updated weights, find the shortest program that outputs them within the time budget (codecs, low-rank fits, predictors from the registered model, short replays). A program well under p bits a weight refutes the price. |
| Learning, 2: no cheap way around it | Any way of reaching the target, at a cost worth ruling out, repeatedly combines freshly trained weights with fresh per-token values. | `Nci.Assumptions.CoversEfficientMethods` | Try to reach the target cheaply without that: small adapters, independently trained small experts, distillation from the served model, frozen features. A cheap success refutes it at that target. |

That each use of a weight needs its own token's context is a checkable wiring property of the training circuit: every
gate a use requires reaches, inside that use's own upstream computation, the operand the use reads. It is named in the
lock (`Nci.Capability.Consumers.Upstream`) until it is checked on a named circuit. The two learning rows are all the overhead lower bound assumes
about learning, and they are where its scientific content lies; the second is the least certain.

### 1. The proofs of the sampled proof blocks

**Establishes.** One Flock `--zk` audit of the sampled proof blocks, proved through the ZK server, is knowledge sound:
if the verifier accepts, the prover knows values satisfying the sampled proof blocks, except with probability ε_ks + ε_bind: the knowledge-soundness
error, plus the binding error, which here sums the Lean's link, rows and registered-reads terms.
It is zero knowledge against an honest verifier: the verifier's view is simulatable without those values, to within the
commitments' hiding gap.

**Assumes.**

- SHA-512 collision resistance (`cr/sha-512`, `ecr/sha-512`).
- The verifier's coins are uniform. Part 1's statements draw them in their own law; `SampledKSLiveV3` and
  `IntegrityHorizon` take `FlockSoundness.Assumptions.UniformRandomBytes`.
- The record the verifier reads is the live run's (`FlockSoundness.Assumptions.Zk.RecordCustodyZKJ`).
- The hiding commitments' gap (hm96 at its pinned key), which is below 2^-193 for all but 2^-64 of keys but not proved small for the
  pinned one.

**Lean.**

- Knowledge soundness of the one-stage audit at the sampled proof blocks: `Flock.Guarantees.EndToEndRegisteredRows`,
  **proved**.
- Zero knowledge against an honest verifier at each fixed coin vector: `Flock.Guarantees.ZeroKnowledgeHidden`, **proved**.
  It holds only at coins the prover doesn't refuse and on runs its self-check doesn't stop, and the self-check's inner
  relation (`InnerHolds`) is unproved.
- Zero knowledge against a verifier that chooses its coins: `Flock.Guarantees.MaskedZK`, **proved**, with its ε_zk
  through hm96's hiding gap at the pinned key. What is unproved is the live zero-knowledge server
  (`Flock.zk_server_lean_loop`'s hypotheses).

### 2. Sampling

**Establishes.** The challenger flips coins after the commitments are recorded, and they select the sample, the proof
blocks that are proved. If every sampled block is proved, only the blocks the sample missed can hold faults:
Pr[accept ∧ more faulty proof blocks than the sample allows at δ] ≤ δ + ε_ks + ε_bind, ε_bind the binding error of the
links between proof blocks.

**Assumes.** Part 1's assumptions; the challenger is the auditor's device (`Verity.Assumptions.TrustedHost`); its coins
reach the developer only after the commitment (`CoinsConfidential`, in prose only).

**Lean.** `Flock.Guarantees.SampledKSLiveV3`, **proved**. It states its own sampling relation; it doesn't take part 1's
theorem as a hypothesis.

### 3. The audit of each verification phase

**Establishes.** Sampled proofs bound the faulty proof blocks. PoUW establishes that the phase's credited live work is
at least (1 − ε) of the registered circuit's work, crediting only noised calls: an un-noised call has no live coin, so nothing forces
its work. The composition accepts only if every subprotocol accepts and their commitments bind into one: any rejection rejects
the whole, and nothing is retried.

**Assumes.** PoUW's premises:

- the proof system is sound (`Properties.Pouw.Period.ServiceSound`), a named assumption no theorem discharges yet;
- the published sample is the coin's (`DrawIsCoin`), and registration precedes the sample (`RegisteredBeforeDraw`);
- the salt is a live coin (`SaltIsLiveCoin`), and the staged statement pins the receipts' tile (`StagedPins`). The
  deployed path meets neither `SaltIsLiveCoin` nor the roots' binding that `H1CheckedCreditPeriod` reads
  (`MovedIn ≤ δ_bind` in the Lean): its registered roots are tags;
- PoUW's per-step receipts bind (`OperandHMu`, `ReceiptLuckAt`, `GuessLuckAt`, `UnitFixed`).

The time credit adds the data sheet's rates.

**Lean.**

- The verdict rule: `Properties.Composition.AcceptMeans`, **proved**.
- Sampled proofs' accept rule: `Properties.SampledProofs.AcceptMeans`, **proved**.
- PoUW's accept rule and period credit: `Properties.Pouw.Check.AcceptMeans` and
  `Properties.Pouw.Period.H1CheckedCreditPeriod` (the general `CheckedCreditPeriod` at the deployed tile unit),
  **proved** under the premises above.
- The time credit: `Accounting.Guarantees.Compose.CircuitCredit`, **proved**, reading the data sheet.
- The proofs behind PoUW's credit, `Accounting.Guarantees.TS.SessionSoundAt`: **proved**, under `AcceptedLive` (no served path meets
  it yet), `WindowLaw`, `ServedChecksHold` and `ServedRootsLinked`.
- The verifier's verdicts tied to these theorems: **open**. Remote reads sampled proofs' one-stage v1, which takes the
  sessions' verdicts as evidence; v2 runs `flock-verify verify --zk` itself.

### 4. The inference-only policy

**Establishes.** Every execution block's inference partition 𝓘_j divides the block's gates into inference blocks of
exactly c gates (padded), each reading at most k bits of fresh input, anything outside the registered information θ;
and θ is registered information. This is the hypothesis part 6's local bounds take, a (c, k)-partition
(`Verity.Partition.IsCK`), which the older bound `DampenerRemote` instead assumes outright as `Nci.Complies`.

**How it is checked.**

- *By whom.* The verifier, on an auditor host, in each verification phase, as one of remote's three subprotocols
  (`NciCheck.Protocol.protocol`, id `verity/inference-only/v0`).
- *On what.* The execution block's description, its inference partition and θ's provenance, against the registration.
  No hardware at the site is involved.
- *From which coins.* None. The check is deterministic, so it adds no failure probability of its own. What binds it to
  the rest is its commitment: the run's record, the same record sampled proofs and PoUW commit to. Two records reject
  remote whatever the subprotocols answer.
- *In the clear.* In v0 the partition and θ's provenance reach the verifier in the clear, a stated exception to
  "everything is zero-knowledge". The check inside the ZK server, over the committed partition with padded counts, is
  open.

**Assumes.** The data the registration names was public, not computed off the books before
(`Nci.Assumptions.BaselineRegistered`), and the record the verifier reads is the run's, as in part 1. Nothing about
learning: the policy restricts the shape of the registered circuit, and part 6's premises say why that shape trains
little.

**Lean.**

- The check's verdict is the policy's definition on every expansion of the description
  (`NciCheck.Properties.SecurityProofs.DescReports`), and a compliance verdict witnesses the policy on every expansion
  (`NciCheck.Properties.SecurityProofs.DescWitnesses`): **proved**.
- The policy as a subprotocol of remote, under the composition's accept rule (`Properties.Composition.AcceptMeans`):
  **proved**.

**How it discharges `Complies`, link by link.** An outsider's chain from acceptance to the overhead lower bound runs through
these, and the open ones are the gap where `DampenerRemote` assumes `Complies` today:

1. Acceptance means the policy accepted the run's record: `Properties.Composition.AcceptMeans`, **proved**.
2. A compliance verdict witnesses the policy on the expanded registered circuit: `DescWitnesses`, **proved**.
3. That circuit is the one sampled proofs and PoUW bind, one registration across the subprotocols: the code's one-record
   tie exists, and no theorem states it, **open**.
4. The witnessed policy is the (c, k)-partition the local bounds read: `Nci.Complies` still states the older form
   (blocks of at most Γ gates, each reading at most α fresh bits per gate), and its restatement in (c, k) is **open**.
5. A block's effective fresh input is k plus k_F, the bits faulty proof blocks inject into it, plus k_a, the advice
   bits its structure depends on (the registered compiler builds the block from the advice). Sampled proofs bound the
   faulty proof blocks and the advice's length is public, but no statement bounds either per inference block yet:
   **open**.
6. The overhead lower bound reading the result instead of assuming `Complies`: **open** (part 6).

### 5. The monitoring period

**Establishes.** The verification phases' bounds pool over a monitoring period, even when the developer adapts each
phase to the coins before it: the chance that some subprotocol of some phase accepts while its claim fails is at most the
sum of their errors. For sampled proofs,
Pr[every phase accepted ∧ faulty proof blocks ≥ ln(1/δ)/q] ≤ δ + Σ_t (ε_ks,t + ε_bind,t), q the sample's coverage.
PoUW reports ε at δ_s = 2⁻⁴⁰ (`verity/core/protocols/pouw/credit.py`'s `DELTA`), the sample's error alone. Its pooled
failure over n verification phases is δ_s + n·(δ_bind + δ_ks), PoUW's own binding and proof-system errors (not sampled
proofs' ε_bind and ε_ks above). δ_ks rests on the undischarged `ServiceSound`, and δ_bind has no deployed bound, so
neither has a number yet. Over a week, a month and a year, at one verification phase an hour (n = 168, 730 and 8,766), the
total is 2⁻⁴⁰ + n·(δ_bind + δ_ks): for a year's total below 1%, δ_bind + δ_ks must stay below about 1.1·10⁻⁶ a phase.

**Assumes.** Parts 1 to 4's assumptions.

**Lean.** `Accounting.Guarantees.JointHorizonLive` (all subprotocols, one adaptive game), `Flock.Guarantees.IntegrityHorizon`
(sampled proofs) and `Accounting.Guarantees.ComposePeriod.CircuitCreditPeriod` (PoUW): **proved**. `JointHorizonLive`
takes per-period guarantees as hypotheses; no theorem yet instantiates it with parts 2 and 3.

### 6. The overhead lower bound

**Establishes, conditionally.** The overhead lower bound: training a new frontier model on the monitored fleet costs at
least

$$ \mathrm{overhead}(\mathcal G, \mathcal A, \eta) \;\ge\; \frac{\Omega_*}{C_{\mathrm{ref}}\,\big[\bar\zeta\,f + (q/c)\,(1 - f)\big]} $$

times the GPU-hours it costs unmonitored, up to probability η. No Lean states this bound: it is derived in prose
(`Security/README.md`, part 4) from the local bounds, `Verity.Guarantees.Counting` and `Nci.Guarantees.CapabilityFloor`.

- Ω_* is the training work any covered method needs.
- C_ref is a demonstrated reference run's GPU-hours.
- f is the share of capacity outside correct credited work of the registered circuit, and ζ̄ the rate at which work can be done there.
- q/c is the most training work one verified inference block of c gates can do, per gate.

**Conditional on three things not in place yet:**

- the inference-only policy's guarantee doesn't reach the local bounds yet (part 4, links 3 to 6);
- there is no utilization policy, so nothing bounds f uniformly over accepted runs;
- the rate ceiling ζ̄ is a hypothesis: reference efficiency, ζ̄ equal to the reference run's rate, is open.

**The mechanism, in words.** Learning repeatedly combines freshly learned weights with fresh per-token information. The
inference-only policy divides the registered circuit into inference blocks of c gates, each taking in at most k fresh bits. A
block can import information, spending its fresh bits, or reconstruct it, spending its gates.

- Fresh weights are hard to hold, so a block holds few of them.
- Each use of a weight needs its own token's context, so a block serves few uses.
- So verified work does little learning, and learning at scale has to run on the uncredited capacity, which PoUW keeps
  small.

The per-block cap is about h·(k + c/g) weighted weight-uses, h the bits of fresh weights a block can hold and g the gates
a use's context takes to rebuild. When rebuilding that context needs more fresh weights than a block holds, the cap falls
to h·k.

**Assumes.** The learning rows of the table above, the hardware rows, and parts 1 to 5. Further premises can strengthen
the local bound (an activation price, a joint price, a recoverable source), each a new hypothesis to earn;
`Security/README.md`, part 2, lists them.

**Lean.**

- The local bounds' caps `Nci.Guarantees.HeldBudget` (or, statistically, `Nci.Guarantees.StatHeldBudget`),
  `Nci.Guarantees.ImportOrRecompute`, `Nci.Guarantees.RecomputeByGates`, `Nci.Guarantees.RecomputeByState` and
  `Nci.Guarantees.WeightedProductCap`, and the floor `Nci.Guarantees.CapabilityFloor`: **proved**
  (`Security/Proofs/Nci/Capability/`, `Security/Proofs/Accounting/Coverage.lean`), each from its premises. Every premise,
  structure and bound is surveyed in `Security/Definitions/Nci/Capability/README.md`.
- Their premises, `Nci.Assumptions.CoversEfficientMethods`, `Nci.Assumptions.DescriptionHard` (or
  `Nci.Assumptions.MinEntropyHard`) and the wiring property `Nci.Capability.Consumers.Upstream`, are named assumptions in
  the lock, which those proofs take as hypotheses; none is tested yet: no price is measured at a decoder budget, and the
  certificate is not checked on a named training circuit.
- `Nci.Guarantees.DampenerRemote`: **proved**, but it is the threaded dampener, an older and different bound,
  ρ = max(1, 1/(s + …)/MFU₀) over the thread structure, not the bound displayed above. It is arithmetic over inline
  hypotheses (`Nci.Assumptions.PouwSound` among them, which no PoUW guarantee discharges), not over the local bounds.
- Over a monitoring period, `Nci.Guarantees.DampenerRemotePeriod`: **stated** (its lock entry is
  `DampenerRemotePeriodStated`).
- The block-description bound (`Verity.Guarantees.BlockDescription`) and the counting bound
  (`Verity.Guarantees.Counting`): **proved**.

### 7. What is completeness, not security

That the registered circuit computes what the served model computes is completeness: it says an honest developer's run is
accepted. No security claim rests on it. A wrong circuit costs the developer acceptance; it never makes the auditor
accept a false claim. Likewise the overhead's numbers depend on measured and assumed parameters (the reference run, the
unregistered share, the prices), each labelled with its source in the calculator.

### What's open

- **Chaining the parts in Lean.**
  - `Verity.Soundness`, the one theorem over everything the compiled verifier accepts, isn't in Lean.
  - `SampledKSLiveV3` doesn't read part 1's theorem.
  - `ServiceSound` and `PouwSound` are undischarged premises.
  - `JointHorizonLive` isn't instantiated with parts 2 and 3.
  - Today the parts compose by hand.
- **The verifier's verdicts.** Remote reads one-stage v1, which takes the sessions' verdicts as evidence.
- **PoUW's deployed path.** It meets neither `SaltIsLiveCoin`, the roots' binding nor `AcceptedLive`; `DrawIsCoin` and
  `StagedPins` are unproved code facts.
- **The pooled δ.** δ_ks and δ_bind have no number, so PoUW's failure over a monitoring period has none either.
- **The inference-only policy's link to the bound** (part 4, links 3 to 6): one registration across the subprotocols as
  a theorem; the policy restated in (c, k); bounds on k_F and k_a per inference block; and the bound reading the result
  instead of assuming `Complies`.
- **The inference-only policy in zero knowledge.** v0 checks the policy in the clear, publishing the partition into
  inference blocks; its check inside the ZK server is open.
- **The utilization policy.** Nothing bounds the uncredited share f over every accepted run.
- **The rate ceiling.** Reference efficiency, which sets ζ̄, is a hypothesis.
- **Part 6's premises.**
  - No update price has been measured at a decoder budget.
  - The wiring property hasn't been checked on a named training circuit.
  - Coverage is untested.
  - The caps and the floor are proved, but their honest witnesses hold `DescriptionHard`, `SourceHard`, `Recovers` and
    coverage only degenerately (a decoder that compresses nothing, coverage of every computation): that the premises
    hold together at a decoder that compresses, and with coverage that excludes some computation, isn't shown yet.
  - The displayed overhead bound has no Lean statement; `DampenerRemote` is the older threaded dampener.

### Promising directions

- **Measure update prices.** Hardness of fresh weights is the argument's one information premise. Measure it on real
  fine-tunes: bits per updated weight, for adversarially chosen sets, against decoders that may use the registered model,
  low-rank fits and short replays within the block's time.
- **Measure the width of a use.** The minimal premises charge one fresh bit per use, which makes the per-block cap grow
  like k². How many bits of a token's upstream values a block must import before rebuilding the rest becomes cheap is a
  local description length of the operand given what the block holds. If it is near a token's hidden state, the cap
  becomes linear in k and the overhead large. This is the measurement worth most.
- **Choose (c, k) per block type.** The caps bind only when a block's own capacity is well above what its fresh bits
  allow. Decoding reads earlier tokens' cached keys and values as fresh input, so how k counts them decides whether any
  cap bites.
- **Weight the structure.** With per-weight prices and per-use weights fixed, the strongest bound the wiring property
  gives is a linear program over the weights. Solving it on the training circuit replaces "every value is hard" with
  "enough priced, structurally supported work".
- **Test coverage directly.** Compute to the target for compact-state methods (adapters, independent small experts,
  distillation from served outputs) against full training, at matched targets. The ratio caps the overhead from above.

## Assumptions, by name

Each is a named `Prop` in a lock's `assumptions` (`Security/lean-audit.json`, and `Security/Properties/lean-audit.json`
for the `Properties.*` ones), one line in plain words, except `CoinsConfidential` (prose only). The local
bounds' premises and proofs arrive in the lock in the same change as this section.

Sampled proofs (`benchmarks/prove_path/README.md`, "What an accepted audit means"):

- `FlockSoundness.Assumptions.UniformRandomBytes`: the verifier's coins are uniform random bytes.
- `FlockSoundness.Assumptions.Zk.RecordCustodyZKJ` (`live-verifier`): the record the verifier reads is the one the
  challenger wrote for the live run, and its draw file is the verifier's own draw.
- `FlockSoundness.Assumptions.TableCRZAt` (`cr/sha-512`) and `FlockSoundness.Discharge.ZkLink.LinkCRZC`
  (`ecr/sha-512`): SHA-512 is collision resistant for the prover's table and link finders. The lock names the table
  finders' premise `TableCRZAt`; the statement reads it through its abbreviation `TableCRZ`.
- `trusted-host` (`Verity.Assumptions.TrustedHost`, `Security/Definitions/Core/Assumptions.lean`): the challenger is
  the auditor's device on a host the developer doesn't control, and the custody above rests on it; the verifier runs
  on an auditor host too. In Lean, only onsite's guarantees (network certification's) read `TrustedHost` itself.
- `CoinsConfidential`: the challenger's coins reach the developer's machine only over paths the auditor's hosts
  control, and only after the commitment. A premise about where the challenger runs, stated in prose in
  `verity/core/protocols/sampled_proofs/README.md` and not yet in Lean: no `.lean` file defines it, and no guarantee takes
  it as a hypothesis. (Its companion `OneClock` ties the challenger's clock to onsite's network certifiers; remote has
  none.)

PoUW, as `H1CheckedCreditPeriod` takes them (the general `CheckedCreditPeriod` takes all but `SaltIsLiveCoin` and
`StagedPins`):

- `Properties.Pouw.Period.ServiceSound`: the proof system is sound at δ_ks against PoUW's check, which is C-Flock's
  soundness at each drawn unit (`Flock.Guarantees.EndToEndRegisteredRows`, at the drawn proof blocks of one audit). It is a named assumption: no
  theorem discharges it yet.
- `Properties.Pouw.Period.DrawIsCoin`: the units the check publishes as drawn are the model's draw at the period's live
  coin. A protocol-code fact still to prove.
- `Properties.Pouw.Period.RegisteredBeforeDraw`: each period's registration is fixed before its draw coin. It states
  the order only, and the binding it is used for also needs the rows committed by the registration, which the deployed
  path doesn't do yet: its registered roots are tags, with no row committed (`anchors []`).
- `Properties.Pouw.Period.SaltIsLiveCoin`: each period's salt is its round's live coin. The deployed path doesn't meet
  it: the loop is given its salt before it serves, and nothing in the record binds the salt to a live coin (`Security/Properties/Pouw/Period.lean`,
  `SaltIsLiveCoin`'s docstring).
- `Properties.Pouw.Period.StagedPins`: the staged statement the service proves pins the flat tile the receipts are
  claims of. Neither side has a Lean function yet.
- `Accounting.Assumptions.TS.OperandHMu`, the premise of PoUW's per-step receipts: a cheater can't predict its skipped
  words' values better than μ.
- `Accounting.Assumptions.TS.ReceiptLuckAt` and `Accounting.Assumptions.Compose.GuessLuckAt`: a skipped round passes
  its receipt, or a guessed fold state passes, only at the stated rate.
- `Accounting.Assumptions.TS.UnitFixed`: the run's fixed and folded words are at least those of the units it didn't
  skip. It stays an assumption until those words are defined from the committed values.
- `Accounting.Assumptions.TS.AcceptedLive`, a premise of the proofs behind PoUW's credit (`SessionSoundAt`'s), not of the period credit: an
  accepted proof's check words are `1` as `flock-verify --zk --registered` reads them. No served path meets it yet.
- `Accounting.Assumptions.TS.WindowLaw`, `ServedChecksHold` and `ServedRootsLinked`, `SessionSoundAt`'s other premises:
  the draw follows its stated law, the served checks hold, and the served roots are linked to the registered ones.
- The data sheet: `Accounting.Assumptions.TS.SheetClock`, `SheetDispatch`, `SheetTensor`, `SheetPipe`, `SheetSmem`,
  `SheetWords`, `SheetSemantics`, `FoldOpt`, `FoldOptFoldedAt` and `FillOpt`: the GPU's clock, rates and semantics are
  the data sheet's, and the folds are optimal. The live-words credit over a monitoring period (`CircuitCreditPeriod`,
  `H1CheckedCreditPeriod`) doesn't read them. The time credit `Accounting.Guarantees.Compose.CircuitCredit` does, and it
  is what backs the overhead bound's `PouwSound` (below), so the overhead over remote rests on the data sheet.

The overhead, `DampenerRemote`:

- In `Security/Definitions/Nci/Assumptions/Remote.lean`, three:
  - `Nci.Assumptions.PouwSound`: each PoUW period's certificate is wrong with probability at most p. What backs p is
    the time credit `CircuitCredit`, which reads the data sheet; no PoUW guarantee instantiates it yet.
  - `Nci.Assumptions.SampledDetection`: sampled proofs catch each faulty proof block with probability at least q,
    whatever came before.
  - `Nci.Assumptions.TerminalDetection`: a detected cheat ends its identity and takes its hardware out of use within Δ.
- `Nci.Assumptions.BaselineRegistered` (`Security/Definitions/Nci/Assumptions/Baseline.lean`): the data the
  registration names was public, not computed off the books before.
- `Nci.Assumptions.WeightsIncompressible` (`Security/Definitions/Nci/Assumptions/Threads.lean`): no small subcircuit
  rebuilds the weights it uses from fewer bits than their width. It is the older form, which the local bounds below
  replace (`Security/README.md`, part 2).
- `Nci.Complies` (`Security/Definitions/Nci.lean`): the registration meets the inference-only policy, in its older form.
  `DampenerRemote` takes it as a hypothesis; part 4 is what discharges it once its open links close.

The local bounds (part 6):

- `Nci.Assumptions.DescriptionHard` (in the lock): every selection of an item
  family (updated weights, operands, or both) has time-bounded description length at least its price, less a deficit.
- `Nci.Assumptions.CoversEfficientMethods` (in the lock): every sufficiently cheap successful method carries at least
  the covered work Ω_*, except with probability η.

## Results per monitoring period

The overhead calculator is `benchmarks/rho` (`verity bench rho`; it is being renamed to overhead). Its `periods.py`
gives the overhead lower bound for monitoring periods of a week, a month and a year, over one datacenter
(`reference-h200`, 12,500 HGX nodes) and all compute (`world-h200`), at the remote scenario `reference-remote-h200`
(`uv run python benchmarks/rho/periods.py --scenario reference-remote-h200`):

| Over | Week | Month | Year | Range | Worst corner | No protocol |
|---|---|---|---|---|---|---|
| one datacenter | 2.75 | 2.79 | 2.8 | 1 to 21.1 | 1 | 2.5 |
| the world | 2.75 | 2.79 | 2.8 | 1 to 21.1 | 1 | 2.5 |

- **What the figures mean.** Each figure is the overhead lower bound against the fleet's full capacity. On that scale a
  developer no protocol encumbers already scores 2.5 (1/MFU₀), so these figures barely exceed it.
- **Their status.** They are draft and their premises contested. They come from the calculator's partial-sum route, not
  from part 6's local bounds.
- **What sets them.** The developer's undeclared GPU share sets nearly all of each figure (about 89%).
- **Their range.** The range takes each parameter to either end of its plausible range; the worst corner takes them all
  at once.
- **The failure probability.** The δ is pooled over the monitoring period (`IntegrityHorizon`, `CircuitCreditPeriod`),
  not per sampling period.

## The demo

**No example runs the remote verifier yet:** `remote check` has no run of record; its first is due Wed 28 Oct, 11 AM
PDT (D3). Lifecycle is the only example that steps through the whole remote composite, with a stand-in prover; h200
runs remote's sampled-proofs half for real, and the vLLM demo runs both halves on one served run, PoUW's as a
development run:

- `verity example lifecycle` (`examples/lifecycle/README.md`): the remote composite stepped through one monitoring period
  on CPU in about 2 s: hardware registration, registration, declarations, live draws, each declaration's verdict and the
  period's accounting, with every deviation it lists refused by name. Its prover is a labelled stand-in for C-Flock, and
  its verdict is the composition's Python mirror over per-declaration stand-ins for its subprotocols, not `remote
  check`. It runs; nothing about it is proved.

- `verity example h200` (`examples/h200/README.md`): a fixed attention shape committed on one H200. Eight proof blocks
  are drawn by one live coin, each is proved by C-Flock `--zk` and accepted by the Lean verifier, and a flipped bit is
  refused. At an 8-block draw, that allows at most 43.77% faulty proof blocks at δ = 1%. It runs, at proof level
  none, and isn't placed: its Lean verifier ran beside the prover. On a CPU at a small shape, run
  `uv run python examples/h200/run.py --out /tmp/e2e` (`--tamper l1.wup` must exit 3).
- `verity example vllm` (`examples/vllm/README.md`), the vLLM demo: Llama-3.2-1B INT8 served by vLLM on one H200, its
  values committed while serving, a sampling block's proof blocks sampled by live coins, proved by C-Flock `--zk` and
  checked by the Lean verifier, and PoUW's H1 on the served model. PoUW's verdict there is a development run under the
  challenger's development key and credits nothing, and an H1 commitment's served execution blocks aren't yet a
  `verity/pouw/v1` case, so no run of record of the remote verdict has passed: its run is moved, at proof level none,
  and no remote guarantee exists for its verdicts to reach (the open gap, above).

`examples/dither_full_util/README.md` has no runner and no run of record. What of PoUW's own check runs
is its Lean check, through its difftest against the Python (`lake exe pouw-check`, the suite
`verity-pouw-check-difftest`; `verity/core/protocols/pouw/lean/check/README.md`), and the composite's difftest
(`verity/core/protocols/composition/tests`).

## Read next

- `Security/README.md`: the definitions, properties and proofs trees, the locks, and the remote protocol in Lean.
- `benchmarks/prove_path/README.md` and `benchmarks/prove_path/e2e_chains/README.md`: what an accepted audit means, and
  each protocol's end-to-end chain.
- `ARCHITECTURE.md` (with the Glossary) and `GLOSSARY.md`: the architecture, the terms, the notation and the older names.
