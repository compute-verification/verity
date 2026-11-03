# The remote protocol

The remote protocol audits a datacenter whose auditor has no devices at the site: the developer commits everything its
hardware computes, and the auditor checks it from outside with live coins and a Lean verifier. It composes three
subprotocols into one verdict over one commitment: sampled proofs (the committed values satisfy the registered circuit,
except on few proof blocks), PoUW (the hardware's time went to the registered work) and the inference-only policy (the
registered circuit is inference, not training).

**The open gap.** Remote's composed soundness theorem, `Verity.Soundness`, is not in Lean yet. It is due after the Lean
dedupe (one vocabulary). Today the pieces below compose by hand: each is proved on its own,
and no Lean theorem states remote's guarantee from them. Nothing yet ties the deployed verifier's verdicts to those
theorems either.

This file is an outline with signposts. The Lean is the definition. The vocabulary is the Glossary's: `ARCHITECTURE.md`'s
Glossary, and `GLOSSARY.md` for the notation and the older names. The lifecycle's terms (the reported advice, the
registered compiler, an execution block and an execution phase) are in `ARCHITECTURE.md`'s Glossary, under Reported advice,
Block and Period. Lean names keep some older names until they are renamed: `Dampener*` means overhead, `*Horizon` a
monitoring period's pooled bound, `Session*` a run of a subprotocol, and `Window*` a period.

**Which "remote".** Six things in the tree are called "remote", and this page means the first:

- the remote protocol, this one: `Core.Remote` (`CoreRemote.lean`, id `verity/remote/v0`);
- PoUB, an onsite subprotocol whose code names still say "remote" until they are renamed: its package
  `verity/core/protocols/remote_bandwidth/`, its Lean verifier `RemoteAudit`, its Lean vocabulary `Accounting.Remote`
  (`Security/Definitions/Accounting/Remote.lean`) and `Accounting.Guarantees.Remote*`, its id `verity/remote-bandwidth/v0`,
  its whole-node benchmark `benchmarks/remote_session/`, and serve-openai's `--remote-verifier`, the address of PoUB's
  verifier;
- `RemoteChallenger` (`verity/core/protocols/sampled_proofs/remote.py`), the client of a challenger on another host;
- `CertifierClient` (`verity/core/protocols/certifier/active.py`), onsite's network certifier reached over the network;
- the overhead bound's remote regime, `Nci.Guarantees.DampenerRemote` (the inference-only policy's Lean is still named
  `Nci`) and `benchmarks/rho`'s `regime.remote`: a bound over this protocol, not a part of it (the slowdown bound,
  part 6 of the security argument below);
- "run remotely": a job run on a cluster node (`research run --on`), which has nothing to do with any protocol.

## Parties

- **The developer** (the prover) runs the datacenter. It registers, reports its advice, commits its values and proves the sampled
  proof blocks behind the ZK server. Nothing it runs or hosts is trusted, and neither is the audited hardware or its
  vendor.
- **The auditor** has two parts, both on auditor hosts that the developer doesn't control: its **challenger** (the
  live coins, which sample what must be proved) and its **verifier** (the Lean verifier: `flock-verify` for each sampled
  proof block, and `remote check` for the verdict). The challenger is the one auditor-trusted device remote relies on
  (`trusted-host`); the verifier is the auditor's own check. The verifier may also use its own coins
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
| Sampled proofs ([README](../sampled_proofs/README.md)) | the committed values satisfy the registered circuit outside the proof blocks the sample misses | `SampledProofs.OneStage.protocol` (`verity/core/service/sampled_proofs`) | `verity/sampled-proofs/one-stage/v1` |
| PoUW ([README](../pouw/README.md)) | the period's credited work, in live words, is the registered work's: its windows credited all or nothing | `PouwCheck.Period.protocol` (`verity/core/protocols/pouw/lean/check`), onsite's work subprotocol too | `verity/pouw/period/v1` |
| The inference-only policy ([README](../network/nci/README.md)) | the registered circuit divides into inference blocks within their gate and fresh-input budgets, and θ is registered information | `NciCheck.Protocol.protocol` (`verity/core/protocols/network/nci/lean`) | `verity/inference-only/v0` |

The code is `CoreRemote.lean`, `protocol = tiedSameB "verity/remote/v0" tied [windows] components` over `components =
[NciCheck.Protocol.protocol, PouwCheck.Period.protocol, SampledProofs.OneStage.protocol]`. The policy and the one-stage
audit commit to the run's record, which names the registered circuit; once the cases fit, two different records reject
remote whatever the subprotocols answer (`Core.Composition.same`). The period's commitment is the auditor's ledger, not
the record, so it joins through a tie (`Core.Remote.windows`): every window of the period must name the policy's record
as its Commit record (`windowRecords`, which `PouwCheck.Credit.commitOf?` reads), and a window on another record
rejects remote unless it rejects already. Its executable is `remote id | check PARAMS COMMITMENT EVIDENCE | serve`
(`RemoteMain.lean`). The Lake package `core_remote` is code: it has no Mathlib and nothing of `Security/`
(`lean-audit.json`). The product's rule and its difftest are the composition's ([README](../composition/README.md),
`verity/core/protocols/composition/tests/test_composition_difftest.py`).

## Lifecycle

1. **Setup phase**, once, before any coin. It fixes the public parameters (the protocol and its version, the sampling
   design and δ, the size classes, the inference-only policy's c and k, the advice length ℓ_a, and each subprotocol's
   public Program and parameters, by digest), the registered information θ (for example the weights) and the registered
   compiler B. It fixes no circuit. The Lean (`Verity.Declaration`, `Verity.Committed`) still states the older shape, a
   circuit with a partition committed beside it.
2. **Execution phase** j. The hardware runs, and the developer commits every value it computes under hiding commitments
   in SHA-512 trees: the committed trace.
3. **Verification phase** j, one per execution phase (`ARCHITECTURE.md`, "How an audit works"):
   1. The developer reports its advice a_j, at most ℓ_a bits, committed. B(a_j) outputs the execution block E_j,
      partitioned into inference blocks, each of which is partitioned into proof blocks, with its placement. The
      developer proves through the ZK server that it ran B correctly on the committed advice of at most ℓ_a bits, and
      that B's output explains the committed trace: each proof block's inputs and outputs are committed values. The
      same statement can prove that every inference block has c gates and reads at most k bits outside θ, which is the
      inference-only policy's check.
   2. Only then does the challenger flip the coins that select the sample, so the partition is fixed before any coin.
   3. The developer proves each sampled proof block with C-Flock `--zk` behind the ZK server, and the Lean verifier
      `flock-verify` checks each proof. PoUW's check (`pouw-check`) reads each PoUW period's commitment and its sampled
      proofs and receipts, and credits the phase's work. No proof is retried, and a failed proof is a visible failure.
   4. The composition accepts only if every subprotocol accepts and the commitments match (the verdict, below), and
      accounting adds the phase's share (Accounting, below).
4. **The verdict**, by the composition rule (`verity/core/protocols/composition/README.md`). The composite rejects if
   any subprotocol rejects. It also rejects if the cases don't fit (unknown, missing, repeated or out of order), or if
   the commitment isn't `bind` of the subprotocols' digests. It is undecided if it binds and some subprotocol is
   undecided, and it accepts if every subprotocol accepts and it binds. On accept, it publishes each subprotocol's public
   outputs.
5. **Accounting**, per monitoring period. The accepted verification phases' bounds pool over the monitoring period
   (`Flock.Guarantees.IntegrityHorizon` for sampled proofs, `Accounting.Guarantees.ComposePeriod.CircuitCreditPeriod`
   for PoUW), and the overhead calculator turns them into overhead (see Results).

## The security argument, end to end

**The claim.** Suppose the remote verdict accepts every verification phase of a monitoring period. Then, except with a
failure probability that, beyond the sample and SHA-512, rests on undischarged premises (What's open):

- the committed computation satisfies the registered circuit, except on few proof blocks;
- the hardware's credited time went to the registered circuit's work, on part 3's premises, which the deployed path
  doesn't meet;
- once remote runs the inference-only policy's check (part 4), which it doesn't yet, the registered circuit divides
  into inference blocks of at most Γ gates, each reading at most α_f fresh bits per gate, hence padded blocks of c = Γ
  gates and k = ⌊α_f·Γ⌋ fresh bits.

Conditionally on the items under What's open below, a fourth claim follows: training a new frontier model takes more
fleet-time than it would without the protocol, by the slowdown bound (part 6). That bound is the argument's one
endpoint, and part 6 gives its total failure probability over a monitoring period.

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

The derivations are in `Security/README.md`'s security argument, whose own four parts are cited as "`Security/README.md`,
part N". Assumptions, by name, below, lists the named `Prop`s these statements read; each theorem's full hypotheses
are its Lean.

### The assumptions, in plain words

These are the argument's assumptions by kind; Assumptions, by name, below, lists every named `Prop`. Everything
else is a theorem or a named gap.

| Kind | Assumption | Lean | How it's tested, or why it's believed |
|---|---|---|---|
| Cryptography | SHA-512 is collision resistant, and the commitments hide their values. | `cr/sha-512`, `ecr/sha-512` (`TableCRZAt`, `LinkCRZC`, `RowsCRZC`); the hiding commitments' gap | Standard; no attack known. |
| The auditor's devices | The challenger runs on a host the developer doesn't control, its coins are uniform, and they reach the developer only after the commitment. | `Verity.Assumptions.TrustedHost`, `RecordCustodyZKJ`, `UniformRandomBytes`; `CoinsConfidential` (prose only) | Inspect the deployment: where the challenger runs, and its coin path. |
| The hardware | Each GPU computes at most at its data sheet's rates, hosts carry no hidden compute, and unregistered hardware is at most a stated share of the fleet. | `Accounting.Assumptions.TS.Sheet*`; the share is a hypothesis of `DampenerRemote` | Measure the cards and inventory the fleet. The share is a claim about the world, not the protocol. |
| Matrix products | Nobody multiplies the protocol's random noised matrices faster than the known way. Only noised calls are credited: the premise is over random operands, never ones the prover chooses. | `OperandHMu`, `FoldOpt`, `FillOpt` | A faster algorithm for random matrix products refutes it. |
| PoUW's receipts | A skipped round passes its receipt, or a guessed fold state passes, only at the stated rate. Conjectured, as their docstrings say. | `ReceiptLuckAt`, `GuessLuckAt` | Believed because each receipt folds committed accumulator words under a key fixed after the commitment; search for a skipped round or a fold-state guess that passes more often than the rate. |
| Enforcement | Sampled proofs catch each faulty proof block with probability at least q_det, whatever came before, and a detected cheat ends its identity and takes its hardware out of use within Δ. | `Nci.Assumptions.SampledDetection`, `Nci.Assumptions.TerminalDetection` | The first is what part 2's sample gives, not yet derived from it in Lean. The second is the auditor's policy: check that a refusal triggers it. |
| The baseline | The data the registration names was public, not computed off the books before. | `Nci.Assumptions.ThetaRegistered` | Check the registered data's provenance: each named dataset or model is public, by its digest. |
| Learning, 1: fresh weights are hard to hold | After enough training since registration, a short program that runs no longer than a block may run can't reproduce many freshly trained weight values: each costs about p bits, even to a program that knows the registered model and may replay a little training. | `Nci.Assumptions.DescriptionHard` | Compress the update: on adversarially chosen sets of updated weights, find the shortest program that outputs them within the time budget (codecs, low-rank fits, predictors from the registered model, short replays). A program well under p bits a weight refutes the price. |
| Learning, 2: no cheap way around it | A floor, a hypothesis about every efficient method: each way of reaching the target below T_max carries some local bound's covered structure (freshly trained weights combined with fresh per-token values, a weighted count of at least Ω_*), except with probability η. It is the open part. | `Nci.Assumptions.CoversEfficientMethods` | Untested. Try to reach the target cheaply without that: small adapters, independently trained small experts, distillation from the served model, frozen features. A cheap success refutes it at that target. |

For the counted uses with an upstream certificate, a block must import the specified context or perform the certified
upstream computation. The certificate is a checkable wiring property of the training circuit: every gate a use requires
reaches, inside that use's own upstream computation, the operand the use reads. It is named in the lock
(`Nci.Capability.Consumers.Upstream`) until it is checked on a named circuit. The two learning rows are all the slowdown bound assumes
about learning, and they are where its scientific content lies; the second, the floor, is the open part.

### 1. The proofs of the sampled proof blocks

**Establishes.** One Flock `--zk` audit of the sampled proof blocks, proved through the ZK server, is knowledge sound:
if the verifier accepts, the prover knows values satisfying the sampled proof blocks, except with probability ε_ks + ε_bind: the knowledge-soundness
error, plus the binding error, which here sums the Lean's link, rows and registered-reads terms.
It is zero knowledge against an honest verifier: the verifier's view is simulatable without those values, to within the
commitments' hiding gap.

**Assumes.**

- SHA-512 collision resistance (`cr/sha-512`, `ecr/sha-512`).
- The verifier's coins are uniform. Part 1's statements pick them in their own sampling design; `SampledKSLiveV3` and
  `IntegrityHorizon` take `FlockSoundness.Assumptions.UniformRandomBytes`.
- The challenger's session record the verifier reads is the live run's (`FlockSoundness.Assumptions.Zk.RecordCustodyZKJ`).
- The hiding commitments' gap at their pinned key, which is below 2^-193 for all but 2^-64 of keys but not proved small
  for the pinned one.

**Lean.**

- Knowledge soundness of the one-stage audit at the sampled proof blocks: `Flock.Guarantees.EndToEndRegisteredRows`,
  **proved**.
- Zero knowledge against an honest verifier at each fixed coin vector: `Flock.Guarantees.ZeroKnowledgeHidden`, **proved**.
  It holds only at coins the prover doesn't refuse and on runs its self-check doesn't stop, and the self-check's inner
  relation (`InnerHolds`) is unproved.
- Zero knowledge against a verifier that chooses its coins: `Flock.Guarantees.MaskedZK`, **proved**, with its ε_zk
  through the hiding commitments' gap at the pinned key. What is unproved is the live zero-knowledge server
  (`Flock.zk_server_lean_loop`'s hypotheses).

### 2. Sampling

**Establishes.** The challenger flips coins after the commitments are recorded, and they select the sample, the proof
blocks that are proved. If every sampled block is proved, only the blocks the sample missed can hold faults:
Pr[accept ∧ more faulty proof blocks than the sample allows at δ] ≤ δ + ε_ks + ε_bind, ε_bind the binding error of the
links between proof blocks.

**Assumes.** Part 1's assumptions; the challenger is the auditor's device (`Verity.Assumptions.TrustedHost`, which `SampledKSLiveV3`
doesn't read); its coins
reach the developer only after the commitment (`CoinsConfidential`, in prose only).

**Lean.** `Flock.Guarantees.SampledKSLiveV3`, **proved**. It states its own sampling relation; it doesn't take part 1's
theorem as a hypothesis. It takes the session files' forms as preconditions; the theorem v2's refusals are meant to
meet, V3 with those preconditions discharged by what `verify --zk` refuses, is not stated yet.

### 3. The audit of each verification phase

**Establishes.** Sampled proofs bound the faulty proof blocks. PoUW establishes that the phase's credited live work is
at least (1 − ε) of the registered circuit's work, crediting only noised calls: an un-noised call has no live coin, so nothing forces
its work. The composition accepts only if every subprotocol accepts and their commitments bind into one: any rejection rejects
the whole, and nothing is retried.

**Assumes.** PoUW's premises, as its period's guarantee takes them:

- the proof system is sound against the window's check (`Properties.Pouw.PeriodJ10R.TileServiceSound`), a named
  assumption no theorem discharges yet;
- the published sample is the tile stratum's draw at the coin (`TileDrawIsCoin`), and registration precedes the
  sample (`RegisteredBeforeDraw`);
- the salt is a live coin (`SaltIsLiveCoin`), which the deployed path doesn't meet;
- PoUW's per-step receipts bind (`SharedNoiseLucks`: H-μ⁺ with the receipts' and guesses' lucks at J10-R's shared
  noise; `UnitFixed`).

The time credit adds the data sheet's rates.

**Lean.**

- Remote's rule, the composition's product rule `check`: `Properties.Composition.AcceptMeans`, **proved**. What it
  accepts fits its subprotocols, each accepts its own case, and the commitment binds their digests.
- Sampled proofs: `Properties.SampledProofs.AcceptMeans`, **proved**, covers v1's `OneStage.check`, which remote
  composes.
- PoUW's period credit: `Properties.Pouw.PeriodJ10R.H1J10RCheckedCreditPeriod`, `CheckedCreditPeriod`'s conclusion
  at `PouwCheck.Credit.window`'s verdict over J10-R's tile stratum, the guarantee `PouwCheck.Period.protocol` names
  (`GUARANTEE`) and refuses every other window form for, **proved** under the premises above.
- The time credit: `Accounting.Guarantees.Compose.CircuitCredit`, **proved**, reading the data sheet.
- The proofs behind PoUW's credit, `Accounting.Guarantees.TS.SessionSoundAt`: **proved**, under `AcceptedLive` (no
  served path meets it yet), `WindowLaw`, `ServedChecksHold` and `ServedRootsLinked`. It concludes
  `ProofLayer.SoundAt` from part 1's theorem; it doesn't discharge `ServiceSound`.
- The verifier's verdicts tied to these theorems: **open**. Remote reads sampled proofs' one-stage v1, which takes
  the sessions' verdicts as evidence instead of computing them; v2 (`verity/sampled-proofs/one-stage/v2`) runs the
  proof check (`verity verify proof`, today's `flock-verify --zk`) itself, and remote doesn't read v2 yet.

### 4. The inference-only policy

**Establishes.** The registered circuit divides into inference blocks of at most Γ gates, each reading at most α_f
fresh bits per gate, anything outside the registered information θ (`Nci.Complies Γ α_f`); and θ is registered
information. A partition meeting that is, once padded, a (c, k)-partition with c = Γ and k = ⌊α_f·Γ⌋, or
k = ⌊α_f·Γ·W⌋ for the bit-level expansion at Γ·W gates,
(`Verity.Partition.IsCK`), the hypothesis part 6's local bounds take. With gates up to W bits wide, the bit-level
expansion meets it at Γ·W gates. The Lean slowdown bound `DampenerRemote` instead takes `Complies` as a hypothesis.

**How it is checked.**

- *By whom.* Not by remote yet: the policy isn't one of its subprotocols (Subprotocols, above). Its check runs on its
  own, as `nci-check` (`NciCheck.Baseline.check`, `verity/core/protocols/network/nci/lean`); once composed, the
  verifier runs it on an auditor host in each verification phase.
- *On what.* The registered circuit's description, its inference partition and θ's provenance, against the registration
  log. No hardware at the site is involved.
- *From which coins.* None. The check is deterministic, so it adds no failure probability of its own. What is to bind
  it to the other subprotocols is a commitment to the registration PoUW and sampled proofs commit to; no composite rule
  ties them yet.
- *In the clear.* The check reads the inference partition and θ's provenance in the clear. Before a composite may run
  it, it needs a private form: the partition committed under hiding commitments, and compliance proved in zero
  knowledge.

**Assumes.** The data the registration names was public, not computed off the books before
(`Nci.Assumptions.ThetaRegistered`; only part 6 reads its Lean form); the verifier's copy of the registration log
is the deployment's; that the registered roots bind their data (`Nci.Assumptions.RootsBind`, named in
`Security/Properties/Nci/Baseline.lean`, not yet stated); and, once the policy is composed, SHA-512 collision resistance
for its tie to one registration. Nothing about learning: the policy restricts the shape of the registered circuit, and
part 6's premises say why that shape trains little.

**Lean, and how it discharges `Complies`, link by link.**

1. *Remote's verdict means the policy accepted the same registration.* **Open**: remote doesn't run the policy, and no
   composite rule ties the policy's registration to the one sampled proofs and PoUW bind. The composition's proved
   property, `Properties.Composition.AcceptMeans`, makes a composite accept each subprotocol's accept, so this link
   needs the policy among remote's subprotocols, with that tie.
2. *The policy's check accepts ⇒ `Baseline.evaluate` complies ⇒ `Desc.evaluate` complies.* The bridge `baseline_desc`
   is a proof lemma (`Security/Proofs/Nci/Cover/Checked.lean`), not a locked property, and the policy's check has no
   `AcceptMeans`: **open** as a property. θ's provenance: `NciCheck.Properties.SecurityProofs.BaselineProvenance`,
   **proved**.
3. *A compliance verdict of `Desc.evaluate` ⇒ `Nci.Complies Γ α_f` of the laid-out registered circuit (a `Verity.Declaration`), on every
   expansion:*
   `NciCheck.Properties.SecurityProofs.DescWitnesses` (its `Witnessed`), **proved**, with `DescReports` (the verdict
   is the definition's), **proved**. `Witnessed` yields exactly `DampenerRemote`'s hypothesis `Complies Γ α d`, so for
   the Lean bound only links 1 and 2 remain open.
4. *`Complies` ⇒ the padded (c, k)-partition the local bounds read:* `Nci.Guarantees.CompliesCK`, and through the
   bit-level expansion `Nci.Guarantees.ExpandComplies`, **proved**. Joining them to link 3 for the run's registered circuit
   is open with links 1 and 2.
5. *A block's effective fresh input:* k plus k_F, the bits faulty proof blocks inject into it, plus k_a, the advice bits
   its structure depends on. Sampled proofs bound the faulty proof blocks and the advice's length is public, but no
   statement bounds either per inference block yet: **open**.
6. *The displayed bound reading the result instead of assuming `Complies`:* **open** (part 6).

### 5. The monitoring period

**Establishes.** The verification phases' bounds pool over a monitoring period, even when the developer adapts each
phase to the coins before it: the chance that some subprotocol of some phase accepts while its claim fails is at most the
sum of their errors. For sampled proofs,
Pr[every phase accepted ∧ faulty proof blocks ≥ ln(1/δ)/q_s] ≤ δ + Σ_t (ε_ks,t + ε_bind,t), q_s the sample's coverage.
PoUW reports ε at δ_s = 2⁻⁴⁰ (`verity/core/protocols/pouw/credit.py`'s `DELTA`), the sample's error alone. Its pooled
failure over n verification phases is δ_s + n·(δ_bind + δ_ks), PoUW's own binding and proof-system errors (not sampled
proofs' ε_bind and ε_ks above). δ_ks rests on the undischarged `ServiceSound`, and δ_bind has no deployed bound, so
neither has a number yet. Over a week, a month and a year, at one verification phase an hour (n = 168, 730 and 8,766), the
total is 2⁻⁴⁰ + n·(δ_bind + δ_ks): for a year's total below 1%, δ_bind + δ_ks must stay below about 1.1·10⁻⁶ a phase.

**Assumes.** Parts 1 to 4's assumptions.

**Lean.** `Accounting.Guarantees.JointHorizonLive` (onsite's five audits in one adaptive game; the inference-only policy is not among them), `Flock.Guarantees.IntegrityHorizon`
(sampled proofs) and `Accounting.Guarantees.ComposePeriod.CircuitCreditPeriod` (PoUW): **proved**. `JointHorizonLive`
takes per-period guarantees as hypotheses; no theorem yet instantiates it with parts 2 and 3.

### 6. The slowdown bound

**Establishes, conditionally.** The slowdown bound: under the protocols, training a new frontier model takes at least

$$ \rho(\mathcal G, \eta) \;\ge\; \frac{\Omega_*}{C_{\mathrm{ref}}\,\big[\bar\zeta\,f + (q/c)\,(1 - f)\big]} $$

times the fleet-time the reference training algorithm takes with no protocol (fleet-time: elapsed time times the whole
fleet's capacity, registered and unregistered hardware). It holds except with the total failure probability below,
η + δ + …: an accepted run reaches the target in less fleet-time at most that often. **This is the argument's one
endpoint.** No Lean states it: it is derived in prose (`Security/README.md`, part 4) from the local bounds,
`Verity.Guarantees.Counting` and `Nci.Guarantees.CapabilityFloor`. The Lean bound `DampenerRemote` is a different,
older formula (below), and the Results table comes from a third route, the calculator's.

**Its total failure probability** over a monitoring period of n verification phases is at most

$$ \eta + \delta + \delta_s + n\,(\varepsilon_{ks} + \varepsilon_{bind} + \delta_{ks} + \delta_{bind}) $$

plus part 1's cryptographic terms: coverage's η, sampled proofs' δ with their per-phase knowledge-soundness and binding
errors (part 5), and PoUW's δ_s = 2⁻⁴⁰ with its per-phase δ_ks and δ_bind. δ_ks and δ_bind have no number yet, so
neither does the total. With e the per-phase sum, a week, a month and a year (n = 168, 730 and 8,766 at one
verification phase an hour) give η + δ + 2⁻⁴⁰ + 168·e, + 730·e and + 8,766·e.

- Ω_* is the weighted count of analysed computation any covered method needs.
- C_ref is measured: the fleet-time at which the reference training algorithm reaches the target with probability
  above η, which takes a seed sweep, not one run. It stands in for the unrestricted training cost, which it bounds from
  above, so the bound is relative to that reference.
- f is the free share: the part of the whole fleet's capacity, unregistered hardware included, not doing correct
  credited work of the registered circuit; ζ̄ is the rate at which work can be done there.
- q/c is the most of that count one verified inference block of c gates can do, per gate. The canonical form keeps
  the sum of the blocks' caps; replacing it by (q/c)(1 − f) needs the accounting to tie each qualifying block to
  credited actual computation.
- With Ω_ref the reference's count in the same convention, ζ_ref = Ω_ref/C_ref and the floor Ω_* ≥ Ω_ref/α, the bound
  is 1/(α[(ζ̄/ζ_ref)·f + y]), y = (q/(c·ζ_ref))(1 − f) the share of the reference's rate inference blocks yield. α
  here is the most any method gains over the reference, not the policy's α_f. Only under reference efficiency, ζ̄ = ζ_ref, is
  it 1/(α(f + y)). The unregistered share s ≤ f limits what the bound can certify; it doesn't predict the actual
  slowdown.

**Conditional on three things not in place yet:**

- the inference-only policy's guarantee doesn't reach the local bounds yet (part 4, links 1, 2, 5 and 6);
- there is no utilization policy, so nothing bounds f uniformly over accepted runs;
- the rate ceiling ζ̄ is a hypothesis: reference efficiency, ζ̄ equal to the reference's measured rate ζ_ref, is open.

**The mechanism, in words.** Learning repeatedly combines freshly learned weights with fresh per-token information. The
inference-only policy divides the registered circuit into inference blocks of c gates, each taking in at most k fresh bits. A
block can import information, spending its fresh bits, or reconstruct it, spending its gates.

- Fresh weights are hard to hold, so a block holds few of them.
- For certified uses, a block must import each use's context or perform its upstream computation, so a block serves
  few uses.
- So verified work does little learning, and learning at scale has to run on the uncredited capacity, which PoUW keeps
  small.

The per-block cap is about h·(k + c/g) weighted weight-uses, h the price of fresh weights the description premise lets a
block hold (not a storage bound) and g the gates a use's certified upstream computation takes. When reconstructing that
computation would expose updated weights priced above h, the cap falls to h·k.

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
- The block-description bound (`Verity.Guarantees.BlockDescription`) and the counting bound
  (`Verity.Guarantees.Counting`): **proved**.

Sampled proofs' accept rule and theorems, the Lean slowdown bounds, and the composed soundness the bounds would rest on
(the last column is the chain's status in `benchmarks/prove_path/e2e_chains/README.md`):

| Piece | Lean declaration | Lock | End to end |
|---|---|---|---|
| Sampled proofs' accept rule | `Properties.SampledProofs.AcceptMeans` (`Security/Properties/SampledProofs.lean`; proof `acceptMeans`) | **proved** | — |
| A C-Flock `--zk` proof, read in the sampling relation: Pr[accept ∧ incorrect proof blocks outside what the draw catches at δ] ≤ δ + ε_ks + δ_link | `Flock.Guarantees.SampledKSLiveV3` (stated in `Security/Proofs/Flock/Bridge/Refused.lean`, proved in `Security/Proofs/Flock/Bridge/KSLive.lean`) | **proved** | sampled proofs `--zk`: proved |
| Sampled proofs over a monitoring period | `Flock.Guarantees.IntegrityPeriod`, `IntegrityHorizon` (`Security/Proofs/Flock/EndToEnd/Period/Statement.lean`) | **proved** | — |
| The overhead over remote, per period | `Nci.Guarantees.DampenerRemote` (`Security/Definitions/Nci/Guarantees/Remote.lean`) | **proved** | none with a proof layer: it takes `PouwSound`, which no PoUW guarantee instantiates |
| The overhead over a monitoring period | `Nci.Guarantees.DampenerRemotePeriod` (`Security/Definitions/Nci/Guarantees/RemotePeriod.lean`; proof `Nci.SecurityProofs.DampenerRemotePeriod` in `Security/Proofs/Nci/RemotePeriod.lean`) | **proved**, under `TerminalDetection`, with each window's bounds as hypotheses: PoUW's certificate wrong with probability at most `p` (`CircuitCredit`'s conclusion at the window), the draw missing `m` faulty blocks with probability at most `(1 − q)^m` and a drawn faulty block undetected with probability at most `ks t`, and each declared run within `Throttled`'s bound | none with a proof layer: the per-window bounds are hypotheses, and no served path discharges them yet |
| The composed soundness of remote | `Verity.Soundness`: not yet in Lean, and `lean-audit.json` here lists no guarantee (the open gap, above) | **open** | — |

`DampenerRemote` is the older threaded slowdown bound, ρ = max(1, 1/(s + x/MFU₀)) over the thread structure
(`Nci.Dampener.remoteDampener`; MFU₀ divides only x), not the bound displayed above: arithmetic over inline hypotheses
(`Nci.Assumptions.PouwSound` among them), not over the local bounds.

### 7. What is completeness, not security

That the registered circuit computes what the served model computes is completeness: it says an honest developer's run is
accepted. No security claim rests on it. A wrong circuit costs the developer acceptance; it never makes the auditor
accept a false claim. Likewise the slowdown's numbers depend on measured and assumed parameters (the reference's
measured cost, the unregistered share, the prices), each labelled with its source in the calculator.

### What's open

- **Chaining the parts in Lean.**
  - `Verity.Soundness`, the one theorem over everything the compiled verifier accepts, isn't in Lean.
  - `SampledKSLiveV3` doesn't read part 1's theorem.
  - `ServiceSound` and `PouwSound` are undischarged premises.
  - `JointHorizonLive` isn't instantiated with parts 2 and 3.
  - Today the parts compose by hand.
- **The verifier's verdicts.** Remote reads one-stage v1, which takes the sessions' verdicts as evidence; v2 runs
  `flock-verify --zk` itself, but remote doesn't read v2 yet.
- **PoUW's deployed path.** It meets neither `SaltIsLiveCoin`, the roots' binding nor `AcceptedLive`; `TileDrawIsCoin`
  is an unproved code fact.
- **The pooled δ.** δ_ks and δ_bind have no number, so PoUW's failure over a monitoring period has none either.
- **The inference-only policy's link to the bound** (part 4, links 1, 2, 5 and 6): the policy composed into remote,
  with a rule tying its registration to the one sampled proofs and PoUW bind; the policy's check accepting as a
  property (`baseline_desc` is a proof lemma); bounds on k_F and k_a per inference block; and the bound reading the
  result instead of assuming `Complies`.
- **The inference-only policy in zero knowledge.** Its check reads the partition into inference blocks in the clear;
  its check inside the ZK server is open.
- **The utilization policy.** Nothing bounds the free share f over every accepted run.
- **The rate ceiling.** Reference efficiency, which sets ζ̄, is a hypothesis.
- **Part 6's premises.**
  - No update price has been measured at a decoder budget.
  - The wiring property hasn't been checked on a named training circuit.
  - Coverage, the floor every efficient method is assumed to meet, is the open part: it is untested.
  - The caps and the floor are proved, and their witnesses are toys: at a four-gate circuit, `DescriptionHard`,
    `SourceHard` and `Recovers` hold against a decoder that compresses one string, each parameter spent (at 0 it
    fails); at a toy task, coverage holds at η = 1/2 but not at 0, a one-gate guess escaping it. No instance meets the
    premises on a training circuit.
  - The displayed slowdown bound has no Lean statement; `DampenerRemote` is the older threaded slowdown bound.

### Promising directions

The next steps test or strengthen part 6's premises: one instance meeting the local premises at once, attacks on
update prices, the certificate checked on a training circuit, a use's import cost, weights by linear programming, and
coverage tested directly. Each is in `Security/Definitions/Nci/Capability/README.md` ("How to test each premise", "The
frontier"); how the execution's boundaries fix what a block's c and k count is in `Security/README.md`, part 2.

## Assumptions, by name

Each is a named `Prop` in a lock's `assumptions` (`Security/lean-audit.json`, and `Security/Properties/lean-audit.json`
for the `Properties.*` ones), one line in plain words, except `CoinsConfidential` (prose only). These are the named `Prop`s the statements above
read; each theorem's full hypotheses are its Lean, among them `TagsOkZHJ`, `Built`, `StrictCR.Finder.CR`,
`DrawnLayer.JointSound` and `.Pins` (which `CircuitCredit` also reads), and the local bounds' structural
hypotheses `HeldPriceAtMost`, `SlotsRead`, `OverlapWeights` and `RequiresState`.

Sampled proofs (`benchmarks/prove_path/README.md`, "What an accepted audit means"):

- `FlockSoundness.Assumptions.UniformRandomBytes`: the verifier's coins are uniform random bytes.
- `FlockSoundness.Assumptions.Zk.RecordCustodyZKJ` (`live-verifier`): the record the verifier reads is the one the
  challenger wrote for the live run, and its sample file is the verifier's own sample.
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

PoUW, as `H1J10RCheckedCreditPeriod` takes them (`Security/Properties/Pouw/PeriodJ10R.lean`):

- `Properties.Pouw.PeriodJ10R.TileServiceSound`: the proof system is sound against `Credit.window` at each sampled
  proof block, C-Flock's soundness under live coins. It is a named assumption: no theorem discharges it yet.
- `Properties.Pouw.PeriodJ10R.TileDrawIsCoin`: on every accepted window, the proof blocks the check demands proofs of
  are the tile stratum's draw at the window's coin. A protocol-code fact still to prove.
- `Properties.Pouw.PeriodJ10R.RegisteredBeforeDraw`: each window's registration is fixed before its draw coin. It
  states the order only, and the binding it is used for also needs the rows committed by the registration, which the
  deployed path doesn't do yet.
- `Properties.Pouw.Period.SaltIsLiveCoin`: each window's salt is its round's live coin. The deployed path doesn't meet
  it: the loop is given its salt before it serves, and nothing in the challenger's record binds the salt to a live coin
  (`Security/Properties/Pouw/Period.lean`, `SaltIsLiveCoin`'s docstring).
- `Properties.Pouw.PeriodJ10R.SharedNoiseLucks`: the Duo receipts' lucks (H-μ⁺, `OperandHMu`, with `ReceiptLuckAt`
  and `GuessLuckAt`) at J10-R's shared noise, one computational claim over a call's joint coins.
- `Accounting.Assumptions.TS.UnitFixed`: the run's fixed and folded words are at least those of the units it didn't
  skip. It stays an assumption until those words are defined from the committed values.
- `Accounting.Assumptions.TS.AcceptedLive`, a premise of the proofs behind PoUW's credit (`SessionSoundAt`'s), not of the period credit: an
  accepted proof's check words are `1` as `flock-verify --zk --registered` reads them. No served path meets it yet.
- `Accounting.Assumptions.TS.WindowLaw`, `ServedChecksHold` and `ServedRootsLinked`, `SessionSoundAt`'s other premises:
  the sample follows its stated sampling design, the served checks hold, and the served roots are linked to the
  registered ones.
- The data sheet: `Accounting.Assumptions.TS.SheetClock`, `SheetDispatch`, `SheetTensor`, `SheetPipe`, `SheetSmem`,
  `SheetWords`, `SheetSemantics`, `FoldOpt`, `FoldOptFoldedAt` and `FillOpt`: the GPU's clock, rates and semantics are
  the data sheet's, and the folds are optimal. The live-words credit over a monitoring period (`CircuitCreditPeriod`,
  `H1J10RCheckedCreditPeriod`) doesn't read them. The time credit `Accounting.Guarantees.Compose.CircuitCredit` does, and it
  is what backs the slowdown bound's `PouwSound` (below), so the slowdown over remote rests on the data sheet.

The slowdown, `DampenerRemote`:

- In `Security/Definitions/Nci/Assumptions/Remote.lean`, three:
  - `Nci.Assumptions.PouwSound`: each PoUW period's certificate is wrong with probability at most p. What backs p is
    the time credit `CircuitCredit`, which reads the data sheet; no PoUW guarantee instantiates it yet.
  - `Nci.Assumptions.SampledDetection`: sampled proofs catch each incorrect proof block with probability at least q,
    whatever came before.
  - `Nci.Assumptions.TerminalDetection`: a detected cheat ends its identity and takes its hardware out of use within Δ.
- `Nci.Assumptions.ThetaRegistered` (`Security/Definitions/Nci/Assumptions/Baseline.lean`): the data the
  registration names was public, not computed off the books before.
- `Nci.Assumptions.WeightsIncompressible` (`Security/Definitions/Nci/Assumptions/Threads.lean`): no small subcircuit
  rebuilds the weights it uses from fewer bits than their width. It is the older form, which the local bounds below
  replace (`Security/README.md`, part 2).
- `Nci.Complies` (`Security/Definitions/Nci.lean`): the registration meets the inference-only policy, in its older form.
  `DampenerRemote` takes it as a hypothesis; part 4 is what discharges it once its open links close.

The local bounds (part 6):

- `Nci.Assumptions.DescriptionHard` (in the lock): every selection of an item
  family (updated weights, operands, or both) has time-bounded description length at least its price, less a deficit.
- `Nci.Assumptions.CoversEfficientMethods` (in the lock): the floor, a hypothesis about every efficient method. Each
  way of reaching the target below T_max carries some local bound's covered structure, with at least its weighted
  count Ω_*, except with probability η. It is the open part, untested.

## Results per monitoring period

These figures come from a third route, neither part 6's displayed bound nor `DampenerRemote`: the slowdown
calculator's partial-sum route. The calculator is `benchmarks/rho` (`verity bench rho`; it is being renamed to
the slowdown calculator). Its `periods.py` gives the slowdown bound for monitoring periods of a week, a month and a year, over one datacenter
(`reference-h200`, 12,500 HGX nodes) and all compute (`world-h200`), at the remote scenario `reference-remote-h200`
(`uv run python benchmarks/rho/periods.py --scenario reference-remote-h200`):

| Over | Week | Month | Year | Range (week, month, year) | Worst corner |
|---|---|---|---|---|---|
| one datacenter | 2.99 | 3.03 | 3.04 | 1 to 41, 50.4, 53.9 | 1 |
| the world | 2.99 | 3.03 | 3.04 | 1 to 41, 50.4, 53.9 | 1 |

- **What the figures mean.** Each figure is the hardware multiple over the reference training algorithm at its MFU,
  MFU₀ (`Nci.Dampener.hardwareMultiple`): at least 1, and 1 with no protocol.
- **Their status.** They are draft and their premises contested. They come from the calculator's partial-sum route, not
  from part 6's local bounds.
- **What sets them.** The share of GPU time outside credited work (ε_undecl) sets nearly all of each figure (96% to
  98%).
- **Their range.** The range takes each parameter to either end of its plausible range; the worst corner takes them all
  at once.
- **The failure probability.** The δ is pooled over the monitoring period (`IntegrityHorizon`, `CircuitCreditPeriod`),
  not per verification phase.

## The demo

**No run of record of the remote verifier yet:** `remote check` (to be `verity verify remote`) has none. Lifecycle is the only example that steps through the whole remote composite, with a stand-in prover; h200
runs remote's sampled-proofs half for real, and the vLLM demo runs both halves on one served run, PoUW's as a
development run:

- `verity example lifecycle` (`examples/lifecycle/README.md`): the remote composite stepped through one monitoring period
  on CPU in about 2 s: hardware registration, registration, declarations, live samples, each declaration's verdict and the
  period's accounting, with every deviation it lists refused by name. Its prover is a labelled stand-in for C-Flock, and
  its verdict is the composition's Python mirror over per-declaration stand-ins for its subprotocols, not `remote
  check`. It runs; nothing about it is proved.

- `verity example h200` (`examples/h200/README.md`): a fixed attention shape committed on one H200. Eight proof blocks
  are sampled by one live coin, each is proved by C-Flock `--zk` and accepted by the Lean verifier, and a flipped bit is
  refused. At an 8-block sample, that allows at most 43.77% incorrect proof blocks at δ = 1%. It runs, at proof level
  none, and isn't placed: its Lean verifier ran beside the prover. On a CPU at a small shape, run
  `uv run python examples/h200/run.py --out /tmp/e2e` (`--tamper l1.wup` must exit 3).
- `verity example vllm` (`examples/vllm/README.md`), the vLLM demo: Llama-3.2-1B INT8 served by vLLM on one H200, its
  values committed while serving, a verification phase's proof blocks sampled by live coins, proved by Flock `--zk` and
  checked by the Lean verifier, and PoUW's H1 on the served model. PoUW's verdict there is a development run under the
  challenger's development key and credits nothing, and an H1 commitment's served execution blocks aren't yet a
  `verity/pouw/period/v1` case, so no run of record of the remote verdict has passed: its run is moved, at proof level none,
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
