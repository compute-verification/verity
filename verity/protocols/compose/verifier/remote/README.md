# The remote protocol

The remote protocol audits a datacenter whose auditor has no devices at the site: the developer commits everything its
hardware computes, and the auditor checks it from outside with live coins and a Lean verifier. It composes three
subprotocols into one verdict over one commitment: SPOT (the committed values satisfy the registered circuit,
except on few proof blocks), PoUW (the hardware's time went to the registered circuit) and the inference-only policy (the
registered circuit is inference, not training).

**Remote is `verity/remote/v1`.** Its PoUW subprotocol is Pearl+'s run verdict (`verity/pouw/pearl-plus/v1`), and
`Core.Composites.ofIds`, which names the composite a configuration's verifier runs, refuses v0's subprotocols.

**The open gap.** Remote's composed guarantee, `Verity.remote_soundness_of_subprotocols`, is proved at its compiled
verifier from four open premises: SPOT's and PoUW's targets (`Vocab.SpotSound`, `PouwSound`) and, for the policy,
whose clause is proved on the Program its check judges, the two statements that tie it to the committed files
(`Vocab.WiresSound`, `GateARow`).

This file is an outline with signposts; the Lean is the definition. The vocabulary is `GLOSSARY.md`,
the Glossary, with the notation and the older names. In Lean's older
names, `Session*` means a run of a subprotocol.

**Which "remote".** Four things in the tree are called "remote", and this page means the first:

- the remote protocol, this one: `Core.Remote` (`CoreRemote.lean`, id `verity/remote/v1`; v0 is development);
- PoUB, an onsite subprotocol whose code names still say "remote": `verity/protocols/resource_accounting/memory/poub/verifier/`, `Poub`,
  `ResourceAccounting.Remote*`, `verity/remote-bandwidth/v0`, `benchmarks/remote_session/` and serve-openai's `--remote-verifier`;
- `RemoteChallenger` (`verity/protocols/proofs/spot/verifier/challenger/remote.py`), a remote challenger's client;
- the slowdown bound's remote regime (`Slowdown.Guarantees.SlowdownRemote`, the slowdown calculator's `regime.remote`): a bound over this
  protocol, not in it (part 6 below);

## Parties

- **The developer** (the prover) runs the datacenter. It registers, reports its advice, commits its values and proves the sampled
  proof blocks behind the ZK server. Nothing it runs or hosts is trusted, and neither is the audited hardware or its
  vendor.
- **The auditor** has two parts, both on auditor devices that the developer doesn't control: its **challenger** (the
  live coins, which sample what must be proved) and its **verifier** (the Lean verifier: `verity-verify-proof` for each sampled
  proof block, and `remote-v1 check` for the verdict). The challenger is the one auditor-trusted device remote relies on
  (`trusted-device`); the verifier is the auditor's own check. The verifier may also use its own coins
  (`IO.getRandomBytes`) instead of the challenger's.
- **How the coins reach the developer.** The challenger opens each coin only after the developer's commitment is
  recorded, over one TCP stream that the developer's client opens at its own `127.0.0.1` port (`RemoteChallenger`).
  Deployed, on an auditor device, that port is the far end of a coin link the auditor device opens to the
  developer's machine: `ssh -N -R` straight to that machine's sshd, relayed and counted by network certification's
  code running on the auditor device (`verity/protocols/compose/verifier/live_challenger/coin_link.py`). No coin path of remote's own, with no
  network attester anywhere, is specified yet.
- **No auditor devices at the site.** There are no network attesters or memory attesters. So network certification,
  PoUS and PoUB are onsite's, not remote's, and memory wipes are neither composite's yet. For network certification,
  see `security/Definitions/Certifier` and its `Guarantees`.

## Subprotocols

| Subprotocol | What it establishes | Lean check | id |
|---|---|---|---|
| SPOT ([README](../../../proofs/spot/verifier/README.md)) | the committed values satisfy the registered circuit outside the blocks the sample misses | `VStar.Recursive.Sampled.protocol` (`verity/protocols/proofs/spot_flock/verifier/lean`), one-stage, recursive or two-stage by its `form` | `verity/sampled-proofs/v1` |
| PoUW ([README](../../../resource_accounting/compute/pouw/verifier/README.md)) | a run's credited MACs at Pearl's assumption, at δ over a work-weighted sample | `PouwCheck.PearlPlus.Protocol.protocol` (`verity/protocols/resource_accounting/compute/pouw/verifier/lean/pearlplus`) | `verity/pouw/pearl-plus/v1` |
| PoUW, v0 | the period's credited work, execution phases all or nothing | `PouwCheck.Period.protocol` (`verity/protocols/resource_accounting/compute/pouw/verifier/lean/check`) | `verity/pouw/period/v1` |
| The inference-only policy ([README](../../../workload_classification/training_detection/inference_only/verifier/README.md)) | the registered circuit divides into inference blocks within their gate and fresh-input budgets; θ is registered | `InferenceOnlyCheck.Protocol.protocol` (`verity/protocols/workload_classification/training_detection/inference_only/verifier/lean`) | `verity/inference-only/v0` |

The code is `CoreRemote.lean`: `V1.protocol = tiedSameB "verity/remote/v1" tied ties components`
(`ties = [run, executionPhase]`; the executable `remote-v1`, `RemoteV1Main.lean`), which
`Verity.SoundnessOf` reads, and v0, `protocol = tiedSameB "verity/remote/v0" tied [executionPhases] components`, which no
soundness theorem covers and no configuration names. Pearl+'s run names its record's digest, tied to the policy's by `V1.run`; its check doesn't
read that record yet. The policy and SPOT commit to the run's record, which names the registered circuit; once
the cases fit, two different records reject remote whatever the subprotocols answer (`Core.Composition.same`). v0's
period commitment is the auditor's ledger, not the record, so it joins by a tie (`Core.Remote.executionPhases`): every execution phase of
the period must name the policy's record as its Commit record (`executionPhaseRecords`), and an execution phase on another record rejects
remote unless it rejects already. v0's executable is `remote` (`RemoteMain.lean`); each runs as
`id | check PARAMS COMMITMENT EVIDENCE [BLOB ...] | serve`. The Lake package `core_remote` is code: no Mathlib, no `security/` (`lean-audit.json`). Remote's
rule and its difftest are the composition's ([README](../lean/README.md),
`tests/verity/protocols/compose/verifier/lean/test_composition_difftest.py`).

## Lifecycle

1. **Setup phase**, once, before any coin. It fixes the public parameters (the protocol and its version, the sampling
   design and δ, the size classes, the inference-only policy's c and k, the advice length ℓ_a, and each subprotocol's
   public Program and parameters, by digest), the registered information θ (for example the weights) and the registered
   compiler B. It fixes no circuit: in the Lean (`Verity.RegisteredCircuit`, `Compiler.run`), D, θ, 𝓘 and the proof
   blocks are B's output on the log's θ and the advice; the committed trace is `Verity.Committed`.
2. **Execution phase** j. The hardware runs, and the developer commits every value it computes under hiding commitments
   in SHA-512 trees: the committed trace.
3. **Verification phase** j, one per execution phase (`ARCHITECTURE.md`, section 3, "One period, step by step"):
   1. The developer reports its advice a_j, at most ℓ_a bits, committed. B(a_j) outputs the execution block E_j,
      partitioned into inference blocks, each of which is partitioned into proof blocks, with its placement. The
      developer proves through the ZK server that it ran B correctly on the committed advice of at most ℓ_a bits, and
      that B's output explains the committed trace: each proof block's inputs and outputs are committed values. The
      same statement can prove that every inference block has c gates and reads at most k bits outside θ, which is the
      inference-only policy's check.
   2. Only then does the challenger flip the coins that select the sample, so the partition is fixed before any coin.
   3. The developer proves each sampled proof block with Flock `--zk` behind the ZK server, and the Lean verifier
      `verity-verify-proof` checks each proof. PoUW's check (`pouw-check`) reads each PoUW period's commitment, proofs and
      receipts, and credits the phase's work. No proof is retried, and a failed proof is a visible failure.
   4. The composition accepts only if every subprotocol accepts and the commitments match (the verdict, below), and
      accounting adds the phase's share (Accounting, below).
4. **The verdict**, by the composition rule (`verity/protocols/compose/verifier/lean/README.md`). The composite rejects if
   any subprotocol rejects. It also rejects if the cases don't fit (unknown, missing, repeated or out of order), or if
   the commitment isn't `bind` of the subprotocols' digests. It is undecided if it binds and some subprotocol is
   undecided, and it accepts if every subprotocol accepts and it binds. On accept, it publishes each subprotocol's public
   outputs.
5. **Accounting**, per monitoring period. The accepted verification phases' bounds pool over the monitoring period
   (`Flock.Guarantees.IntegrityPooled` for SPOT, `ResourceAccounting.Guarantees.ComposeMonitoringPeriod.CircuitCreditMonitoringPeriod`
   for PoUW), and the slowdown calculator turns them into the slowdown bound (see Results).

## The security argument, end to end

**The claim.** Suppose the remote verdict accepts every verification phase of a monitoring period. Then, except with a
failure probability that, beyond the sample and SHA-512, rests on undischarged premises (What's open):

- the committed computation satisfies the registered circuit, except on few proof blocks;
- the hardware's credited time went to the registered circuit's work, on part 3's premises, which the deployed path
  doesn't meet;
- the registered circuit the policy's case names divides into inference blocks of at most Γ gates, each reading at
  most α_f fresh bits per gate, hence padded blocks of c = Γ gates and k = ⌊α_f·Γ⌋ fresh bits (part 4); that it is
  the committed files' circuit is open.

Conditionally on the items under What's open below, a fourth claim follows: training a new frontier model takes more
fleet-time than it would without the protocol, by the slowdown bound (part 6). That bound is the argument's one
endpoint, and part 6 gives its total failure probability over a monitoring period.

The argument has seven parts. **Each part is proved, stated or open on its own: no Lean theorem yet chains one part's
conclusion into the next part's hypothesis**, apart from the composition, whose targets are premises. The one partial
link is `ResourceAccounting.Guarantees.TS.SessionSoundAt`, which consumes part 1's theorem for PoUW's proofs. The links that are missing are listed under What's open. Each part gives
what it establishes, what it assumes, its Lean and that Lean's status:

- **proved**: its proof is in a lock (`<Area>.SecurityProofs.G : <Area>.Guarantees.G` in
  `security/lean-audit.json`, or a property's proof in `security/Properties/lean-audit.json`);
- **stated**: its statement is in a lock, with no proof yet;
- **open**: not in Lean.

Flock's guarantee statements (`Flock.Guarantees.*`) are in `security/Proofs/Flock/`, not `Definitions/`: its spec
isn't extracted yet.

The derivations are in `security/README.md`'s security argument, whose own four parts are cited as "`security/README.md`,
part N". Assumptions, by name, below, lists the named `Prop`s these statements read; each theorem's full hypotheses
are its Lean.

### The assumptions, in plain words

These are the argument's assumptions by kind; Assumptions, by name, below, lists every named `Prop`. Everything
else is a theorem or a named gap.

| Kind | Assumption | Lean | How it's tested, or why it's believed |
|---|---|---|---|
| Cryptography | SHA-512 is collision resistant, and the commitments hide their values. | `cr/sha-512`, `ecr/sha-512` (`SHA512CRStrict`, `SHA512CRExpected`, at the finders `TableCRZAt`, `LinkCRZCAt`); the hiding commitments' gap | Standard; no attack known. |
| The auditor's devices | The challenger runs on a host the developer doesn't control, its coins are uniform, and they reach the developer only after the commitment. | `Verity.Assumptions.TrustedDevice`, `RecordCustodyZKJPLaw`, `UniformRandomBytes`; `CoinsAfterCommitment` (prose only) | Inspect the deployment: where the challenger runs, and its coin path. |
| The hardware | Each GPU computes at most at its data sheet's rates, hosts carry no hidden compute, and unregistered hardware is at most a stated share of the fleet. | `ResourceAccounting.Assumptions.TS.Sheet*`; the share is a hypothesis of `SlowdownRemote` | Measure the cards and inventory the fleet. The share is a claim about the world, not the protocol. |
| Matrix products | Nobody multiplies the protocol's random noised matrices faster than the known way. Only noised calls are credited: the assumption is over random operands, never ones the prover chooses. | `OperandHMu`, `FoldOpt`, `FillOpt` | A faster algorithm for random matrix products refutes it. |
| PoUW's receipts | A skipped round passes its receipt, or a guessed fold state passes, only at the stated rate. Conjectured, as their docstrings say. | `ReceiptLuckAt`, `GuessLuckAt` | Believed because each receipt folds committed accumulator words under a key fixed after the commitment; search for a skipped round or a fold-state guess that passes more often than the rate. |
| Enforcement | SPOT catches each incorrect proof block with probability at least q_det, whatever came before, and a detected cheat ends its identity and takes its hardware out of use within Δ. | `Slowdown.Assumptions.SampledDetection`, `Slowdown.Assumptions.TerminalDetection` | The first is what part 2's sample gives, not yet derived from it in Lean. The second is the auditor's policy: check that a refusal triggers it. |
| The registered information | The data the registration names was public, not computed off the books before. | `InferenceOnly.Assumptions.ThetaRegistered` | Check the registered data's provenance: each named dataset or model is public, by its digest. |
| Learning, 1: fresh weights are hard to hold | After enough training since registration, a short program that runs no longer than a block may run can't reproduce many freshly trained weight values: each costs about p bits, even to a program that knows the registered model and may replay a little training. | `Slowdown.Assumptions.DescriptionHard` | Compress the update: on adversarially chosen sets of updated weights, find the shortest program that outputs them within the time budget (codecs, low-rank fits, predictors from the registered model, short replays). A program well under p bits a weight refutes the price. |
| Learning, 2: no cheap way around it | A floor, a hypothesis about every efficient method: each way of reaching the target below T_max carries some local bound's covered structure (freshly trained weights combined with fresh per-token values, a weighted count of at least Ω_*), except with probability η. It is the open part. | `Slowdown.Assumptions.CoversEfficientMethods` | Untested. Try to reach the target cheaply without that: small adapters, independently trained small experts, distillation from the served model, frozen features. A cheap success refutes it at that target. |

For the counted uses with an upstream certificate, a block must import the specified context or perform the certified
upstream computation. The certificate is a checkable wiring property of the training circuit: every gate a use requires
reaches, inside that use's own upstream computation, the operand the use reads. It is named in the lock
(`Slowdown.Capability.Consumers.Upstream`) until it is checked on a named circuit. The two learning rows are all the slowdown bound assumes
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
  `IntegrityPooled` take `FlockSoundness.Assumptions.UniformRandomBytes`.
- The challenger's record the verifier reads is the live run's (`Zk.RecordCustodyZKJPLaw`).
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
blocks that are proved. If every sampled proof block is proved, only the proof blocks the sample missed can be incorrect:
Pr[accept ∧ more incorrect proof blocks than the sample allows at δ] ≤ δ + ε_ks + ε_bind, ε_bind the binding error of the
links between proof blocks.

**Assumes.** Part 1's assumptions; the challenger is the auditor's device (`Verity.Assumptions.TrustedDevice`, which `SampledKSLiveV3`
doesn't read); its coins
reach the developer only after the commitment (`CoinsAfterCommitment`, in prose only).

**Lean.** `Flock.Guarantees.SampledKSLiveV3`, **proved**. It states its own sampling relation; it doesn't take part 1's
theorem as a hypothesis. It takes that the session's files parse and the public file has no shared rows; it reads
the statement's scope, its tags and the sample off acceptance, since `verify --zk` refuses a session outside them
(`Flock.KsScope.check`).

### 3. The audit of each verification phase

**Establishes.** SPOT bound the incorrect proof blocks. PoUW establishes that the phase's credited live work is
at least (1 − ε) of the registered circuit's work, crediting only noised calls: an un-noised call has no live coin, so nothing forces
its work. The composition accepts only if every subprotocol accepts and their commitments bind into one: any rejection rejects
the whole, and nothing is retried.

**Assumes.** PoUW's premises, as its period's guarantee takes them:

- the proof system is sound against the execution phase's check (`Properties.Pouw.MonitoringPeriodJ10R.TileServiceSound`), a named
  assumption no theorem discharges yet;
- the published sample is the tile stratum's sample at the coin (`TileSampleIsCoin`), and registration precedes the
  sample (`RegisteredBeforeDraw`);
- the salt is a live coin (`SaltIsLiveCoin`), which the deployed path doesn't meet;
- PoUW's per-step receipts bind (`SharedNoiseLucks`: H-μ⁺ with the receipts' and guesses' lucks at J10-R's shared
  noise; `UnitFixed`).

The time credit adds the data sheet's rates.

**Lean.**

- Remote's rule, the composition's rule `check`: `Properties.Composition.AcceptMeans`, **proved**. What it
  accepts fits its subprotocols, each accepts its own case, and the commitment binds their digests.
- SPOT: `Properties.Spot.AcceptMeans`, **proved**, covers `OneStage.check`, where remote's one-stage
  form (v2) ends, on its own run's verdicts.
- PoUW's period credit: `Properties.Pouw.MonitoringPeriodJ10R.H1J10RCheckedCreditMonitoringPeriod`, `CheckedCreditMonitoringPeriod`'s conclusion
  at `PouwCheck.Credit.executionPhase`'s verdict over J10-R's tile stratum, the guarantee `PouwCheck.Period.protocol` names
  (`GUARANTEE`) and refuses every other form of an execution phase for, **proved** under the premises above.
- The time credit: `ResourceAccounting.Guarantees.Compose.CircuitCredit`, **proved**, reading the data sheet.
- The proofs behind PoUW's credit, `ResourceAccounting.Guarantees.TS.SessionSoundAt`: **proved**, under `AcceptedLive` (no
  served path meets it yet), `ExecutionPhaseDesign`, `ServedChecksHold` and `ServedRootsLinked`. It concludes
  `ProofLayer.SoundAt` from part 1's theorem; it doesn't discharge `ServiceSound`.
- The verifier's verdicts tied to these theorems: **open** (What's open, below).

### 4. The inference-only policy

**Establishes.** The registered circuit divides into inference blocks of at most Γ gates, each reading at most α_f
fresh bits per gate, anything outside the registered information θ (`InferenceOnly.CompliesRate Γ α_f`); and θ is registered
information. On a Boolean circuit such a partition is a (c, k)-partition with c = Γ and k = ⌊α_f·Γ⌋
(`Verity.Partition.IsCK`), the hypothesis part 6's local bounds take. With gates up to W bits wide, the bit-level
expansion meets it at Γ·W gates and k = ⌊α_f·Γ·W⌋. `SlowdownThreaded` takes `InferenceOnly.CompliesRate Γ α_f` of d.

**How it is checked.**

- *By whom.* By remote's verifier, whose subprotocol it is (Subprotocols, above), and on its own as
  `inference-only-check` (`InferenceOnlyCheck.Theta.check`, `verity/protocols/workload_classification/training_detection/inference_only/verifier/lean`).
- *On what.* The registered circuit's description, its inference partition and θ's provenance, against the registration
  log. No hardware at the site is involved.
- *From which coins.* None. The check is deterministic, so it adds no failure probability of its own. The run's record
  binds it to the others: the policy and SPOT commit to one record (`Core.Remote.V1.tied`), which Pearl+'s run names
  (`V1.run`), and an accepted composite meets both (`Properties.Composition.AcceptSame`, `AcceptTied`, **proved**).
- *In the clear.* The check reads the inference partition and θ's provenance in the clear, and v0 publishes the
  division and each block's gates and fresh bits: remote composes it for one run of record only, a stated exception to
  everything-zero-knowledge (`CoreRemote.lean`), until its private form (the division under hiding leaves, compliance
  proved) replaces it.

**Assumes.** The data the registration names was public, not computed off the books before
(`InferenceOnly.Assumptions.ThetaRegistered`; only part 6 reads its Lean form); the verifier's copy of the registration log
is the deployment's; that the registered roots bind their data (`InferenceOnly.Assumptions.RootsBind`, named in
`security/Properties/InferenceOnly/Theta.lean`, not yet stated); and SHA-512 collision resistance for its tie to one
record. Nothing about learning: the policy restricts the shape of the registered circuit, and
part 6's premises say why that shape trains little.

**Lean, and how it discharges `CompliesRate`, link by link.**

1. *Remote's verdict means the policy accepted the same registration.* **Proved** of the case as sent: the policy
   accepted its own case, on SPOT's record (`Properties.Composition.AcceptMeans`, `AcceptSame`, `AcceptTied`), and
   remote's soundness carries its clause, `Properties.Remote.PolicyHeld`, proved by `Vocab.inferenceOnlySound`. That
   the case's circuit is the committed files' is **open** (`Vocab.WiresSound`, `GateARow`: the open gap).
2. *The policy's verifier accepts ⇒ `Theta.evaluate` complies ⇒ `Desc.evaluate` complies.*
   `InferenceOnlyCheck.Properties.ProtocolAccepts`, **proved**: where the policy's verifier accepts, `Theta.evaluate`
   accepts with `complies` true, and `Desc.evaluate` gives that verdict too. θ's provenance:
   `InferenceOnlyCheck.Properties.SecurityProofs.ThetaProvenance`, **proved**.
3. *A compliance verdict of `Desc.evaluate` ⇒ `InferenceOnly.CompliesRate Γ α_f` of the laid-out `Verity.RegisteredCircuit`, on every
   expansion:*
   `InferenceOnlyCheck.Properties.SecurityProofs.DescWitnesses` (its `Witnessed`), **proved**, with `DescReports` (the verdict
   is the definition's), **proved**; `refines` bounds the unrolled check's blocks by (Γ, α_f·Γ). `SlowdownThreaded`
   takes `CompliesRate Γ α_f d`, what this verdict gives, and `PolicyHeld` holds it of the case's circuit
   (`Properties.Remote.PolicyMet`).
4. *`CompliesRate` ⇒ the (c, k)-partition the local bounds read:* `InferenceOnly.Guarantees.CompliesCK`, and through the
   bit-level expansion `InferenceOnly.Guarantees.ExpandComplies`, **proved**. `PolicyMet` holds both of the
   case's circuit, its constants registered; of the committed files' circuit they wait on link 1's two statements.
5. *A block's effective fresh input:* k plus k_F, the bits incorrect proof blocks inject into it, plus k_a, the advice bits
   its structure depends on. SPOT bound the incorrect proof blocks and the advice's length is public, but no
   statement bounds either per inference block yet: **open**.
6. *The displayed bound reading the result instead of assuming `CompliesRate`:* **open** (part 6).
7. *The count's verdict `InferenceOnly.CompliesAtMost c k n` ⇒ the day's bound:* `Slowdown.Cut.Guarantees.CountHE`, **proved** under
   (H) and (E), the verdict its only premise on the circuit; `InferenceOnlyCheck.Properties.DescCountsCK`, from a published
   `complies` to the verdict, and the tie of the deployment's days to the registered circuits the count checked are **open**.

### 5. The monitoring period

**Establishes.** The verification phases' bounds pool over a monitoring period, even when the developer adapts each
phase to the coins before it: the chance that some subprotocol of some phase accepts while its claim fails is at most the
sum of their errors. For SPOT,
Pr[every phase accepted ∧ incorrect proof blocks ≥ ln(1/δ)/q_s] ≤ δ + Σ_t (ε_ks,t + ε_bind,t), q_s the sample's coverage.
PoUW reports ε at δ_s = 2⁻⁴⁰ (`verity/protocols/resource_accounting/compute/pouw/verifier/credit.py`'s `DELTA`), the sample's error alone. Its pooled
failure over n verification phases is δ_s + n·(δ_bind + δ_ks), PoUW's own binding and proof-system errors (not SPOT's ε_bind and ε_ks above). δ_ks rests on the undischarged `ServiceSound`, and δ_bind has no deployed bound, so
neither has a number yet. Over a week, a month and a year, at one verification phase an hour (n = 168, 720 and 8,760,
`verity.protocol.MONITORING_PERIODS`), the total is 2⁻⁴⁰ + n·(δ_bind + δ_ks): a year's total below 1% needs δ_bind + δ_ks below about
1.1·10⁻⁶ a phase. A year's n·(δ_bind + δ_ks) below 2⁻¹⁰⁰ needs 2⁻¹¹³·¹ a phase; the total
also needs δ_s below 2⁻¹⁰⁰.

**Assumes.** Parts 1 to 4's assumptions.

**Lean.** `ResourceAccounting.Guarantees.JointPooledLive` (onsite's five audits in one adaptive game, inference-only not among them), `Flock.Guarantees.IntegrityPooled`
(SPOT) and `ResourceAccounting.Guarantees.ComposeMonitoringPeriod.CircuitCreditMonitoringPeriod` (PoUW): **proved**. `JointPooledLive`
takes per-period guarantees as hypotheses; no theorem yet instantiates it with parts 2 and 3.

### 6. The slowdown bound

**Establishes, conditionally.** The slowdown bound: under the protocols, training a new frontier model takes at least

$$ \mathrm{slowdown}(\mathcal G, \mathcal A, \eta) \;\ge\; \frac{\Omega_*}{C_{\mathrm{ref}}\,\big[\bar\zeta\,f + (q/c)\,(1 - f)\big]} $$

times the fleet-time the reference training algorithm takes with no protocol (fleet-time: elapsed time times the whole
fleet's capacity, registered and unregistered hardware). It holds except with the total failure probability below,
η + δ + …: an accepted run reaches the target in less fleet-time at most that often. **This is the argument's one
endpoint.** No Lean states it: it is derived in prose (`security/README.md`, part 4) from the local bounds,
`Verity.Guarantees.Counting` and `Slowdown.Guarantees.CapabilityFloor`. The Lean bound `SlowdownRemote` is a different,
older formula (below), and the calculator's figures (Results per monitoring period) come from a third route.

**Its total failure probability** over a monitoring period of n verification phases is at most

$$ \eta + \delta + \delta_s + n\,(\varepsilon_{ks} + \varepsilon_{bind} + \delta_{ks} + \delta_{bind}) $$

plus part 1's cryptographic terms: coverage's η, SPOT's δ with their per-phase knowledge-soundness and binding
errors (part 5), and PoUW's δ_s = 2⁻⁴⁰ with its per-phase δ_ks and δ_bind. δ_ks and δ_bind have no number yet, so
neither does the total. With e the per-phase sum, a week, a month and a year at one verification phase an hour give
η + δ + 2⁻⁴⁰ + 168·e, + 720·e and + 8,760·e.

- Ω_* is the weighted count of analysed computation any covered method needs.
- C_ref is measured: the fleet-time at which the reference training algorithm reaches the target with probability
  above η, which takes a seed sweep, not one run. It stands in for the unrestricted training cost, which it bounds from
  above, so the bound is relative to that reference.
- f is the free share: the part of the whole fleet's capacity, unregistered hardware included, not doing correct
  credited work of the registered circuit; ζ̄ is the rate at which work can be done there.
- q/c is the most of that count one verified inference block of c gates can do, per gate. The canonical form keeps
  the sum of the blocks' yields; replacing it by (q/c)(1 − f) needs the accounting to tie each qualifying block to
  credited actual computation.
- With Ω_ref the reference's count in the same convention, ζ_ref = Ω_ref/C_ref and the floor Ω_* ≥ Ω_ref/α, the bound
  is 1/(α[(ζ̄/ζ_ref)·f + y]), y = (q/(c·ζ_ref))(1 − f) the share of the reference's rate inference blocks yield. α
  here is the most any method gains over the reference, not the policy's α_f. Only under reference efficiency, ζ̄ = ζ_ref, is
  it 1/(α(f + y)). The unregistered share s ≤ f limits what the bound can certify; it doesn't predict the actual
  slowdown.

**Conditional on three things not in place yet:**

- the inference-only policy's guarantee doesn't reach the local bounds yet (part 4: link 1's two statements on the
  committed files, and links 5 and 6);
- there is no utilization policy, so nothing bounds f uniformly over accepted runs;
- the rate ceiling ζ̄ is a hypothesis: reference efficiency, ζ̄ equal to the reference's measured rate ζ_ref, is open.

**The mechanism, in words.** Learning repeatedly combines freshly learned weights with fresh per-token information. The
inference-only policy divides the registered circuit into inference blocks of c gates, each taking in at most k fresh bits. A
block can import information, spending its fresh bits, or reconstruct it, spending its gates.

- Fresh weights are hard to hold, so a block holds few of them.
- For certified uses, a block must import each use's context or perform its upstream computation, so a block serves
  few uses.
- So verified work does little learning, and learning at scale has to run on free compute, which PoUW keeps
  small.

The inference-block yield is about k + c/g served uses, g the gates a use's certified upstream computation takes. When
reconstructing that computation would expose updated weights priced above h, the price of fresh weights the description
assumption lets a block hold (not a storage bound), the yield falls to k.

**Assumes.** The learning rows of the table above, the hardware rows, and parts 1 to 5. Further premises can strengthen
the local bound (an activation price, a joint price, a recoverable source), each a new hypothesis to earn;
`security/README.md`, part 2, lists them.

**Lean.**

- The local bounds `Slowdown.Guarantees.HeldBudget` (or, statistically, `Slowdown.Guarantees.StatHeldBudget`),
  `Slowdown.Guarantees.ImportOrRecompute`, `Slowdown.Guarantees.RecomputeByGates` and `Slowdown.Guarantees.RecomputeByState`, and the
  floor `Slowdown.Guarantees.CapabilityFloor`: **proved**
  (`security/Proofs/Slowdown/Capability/`, `security/Proofs/Slowdown/Coverage.lean`), each from its premises. Every premise,
  structure and bound is surveyed in `security/Definitions/Slowdown/Capability/README.md`.
- Their premises, `Slowdown.Assumptions.CoversEfficientMethods`, `Slowdown.Assumptions.DescriptionHard` (or
  `Slowdown.Assumptions.MinEntropyHard`) and the wiring property `Slowdown.Capability.Consumers.Upstream`, are named assumptions in
  the lock, which those proofs take as hypotheses; none is tested yet: no price is measured at a decoder budget, and the
  certificate is not checked on a named training circuit.
- The block-description bound (`Verity.Guarantees.BlockDescription`) and the counting bound
  (`Verity.Guarantees.Counting`): **proved**.

SPOT's accept rule and theorems, the Lean slowdown bounds, and the composed soundness the bounds would rest on
(the last column is the chain's status in `benchmarks/prove_path/e2e_chains/README.md`):

| Piece | Lean declaration | Lock | End to end |
|---|---|---|---|
| SPOT's accept rule | `Properties.Spot.AcceptMeans` (`security/Properties/Spot.lean`; proof `acceptMeans`) | **proved** | — |
| A Flock `--zk` proof, read in the sampling relation: Pr[accept ∧ incorrect proof blocks outside what the sample catches at δ] ≤ δ + ε_ks + ε_bind | `Flock.Guarantees.SampledKSLiveV3` (stated in `security/Proofs/Flock/Bridge/ScopeGate.lean`, proved in `security/Proofs/Flock/Bridge/KSLive.lean`) | **proved** | SPOT `--zk`: proved |
| SPOT over a monitoring period | `Flock.Guarantees.IntegrityMonitoringPeriod`, `IntegrityPooled` (`security/Proofs/Flock/EndToEnd/Period/Statement.lean`) | **proved** | — |
| The slowdown over remote, per period | `Slowdown.Guarantees.SlowdownRemote` (`security/Definitions/Slowdown/Guarantees/Remote.lean`) | **proved** | none with a proof layer: it takes `PouwSound`, which no PoUW guarantee instantiates |
| The slowdown over a monitoring period | `Slowdown.Guarantees.SlowdownRemoteMonitoringPeriod` (`security/Definitions/Slowdown/Guarantees/RemoteMonitoringPeriod.lean`; proof `Slowdown.SecurityProofs.SlowdownRemoteMonitoringPeriod` in `security/Proofs/Slowdown/RemoteMonitoringPeriod.lean`) | **proved**, under `TerminalDetection`, with each verification phase's bounds as hypotheses: PoUW's certificate wrong with probability at most `p` (`CircuitCredit`'s conclusion at the phase), the sample missing `m` incorrect proof blocks with probability at most `(1 − q)^m` and a sampled incorrect proof block undetected with probability at most `ks t`, and each day's count verdict with (H) and (E), pooled over the days | none with a proof layer: the per-phase bounds are hypotheses, and no served path discharges them yet |
| The composed soundness of remote | `Verity.remote_soundness_of_subprotocols : Verity.RemoteSoundness` (`security/Properties/Verity/Soundness.lean`) | **proved**, from two open targets and the two statements on the committed files (the open gap, above) | — |

`SlowdownRemote` is the slowdown bound max(1, 1/(s + x/MFU₀)) (`Slowdown.Slowdown.remoteSlowdown`; MFU₀ divides only
x), not the bound displayed above: arithmetic over inline hypotheses (`Slowdown.Assumptions.PouwSound` among them), with
the registered circuit's work bounded by `CountHE`.

### 7. What is completeness, not security

That the registered circuit computes what the served model computes is completeness: it says an honest developer's run is
accepted. No security claim rests on it. A wrong circuit costs the developer acceptance; it never makes the auditor
accept a false claim. Likewise the slowdown's numbers depend on measured and assumed parameters (the reference's
measured cost, the unregistered share, the prices), each labelled with its source in the calculator.

### What's open

- **Chaining the parts in Lean.**
  - `Vocab.SpotSound` and `Vocab.PouwSound` aren't proved at remote, nor `Vocab.WiresSound` and `GateARow`, which
    tie the policy's clause, proved on the Program its check judges, to the committed files.
  - `SampledKSLiveV3` doesn't read part 1's theorem.
  - `ServiceSound` and `Slowdown.Assumptions.PouwSound` are undischarged premises.
  - `JointPooledLive` isn't instantiated with parts 2 and 3.
- **The verifier's verdicts.** SPOT's one-stage form is v2 (`verity/sampled-proofs/one-stage/v2`), which runs
  the proof check (`verity verify proof verify --zk`) itself; v1, which takes the verdicts as
  evidence, and the recursive form, which SPOT's soundness doesn't cover, are refused, and the two-stage form is read and
  checked, then refused last while its switch is off. Where v2 accepts, its session
  is in V5's event (`Verity.Sampled.checkBlobs_event`), which is coupled to it (`Verity.Sampled.CouplingTarget`).
- **PoUW's deployed path.** It meets neither `SaltIsLiveCoin`, the roots' binding nor `AcceptedLive`; `TileSampleIsCoin`
  is an unproved code fact.
- **The pooled δ.** δ_ks and δ_bind have no number, so PoUW's failure over a monitoring period has none either.
- **The inference-only policy's link to the bound** (part 4, links 1, 5 and 6): the policy's clause on the committed
  files (`Vocab.WiresSound` and `GateARow`, above); bounds on k_F and k_a per inference block; and the bound reading
  a verdict's rate.
- **The inference-only policy in zero knowledge.** Its verifier reads the partition into inference blocks in the clear;
  its check inside the ZK server is open.
- **The utilization policy.** Nothing bounds the free share f over every accepted run.
- **The rate ceiling.** Reference efficiency, which sets ζ̄, is a hypothesis.
- **Part 6's premises.**
  - No update price has been measured at a decoder budget.
  - The wiring property hasn't been checked on a named training circuit.
  - Coverage, the floor every efficient method is assumed to meet, is the open part: it is untested.
  - The local bounds and the floor are proved, and their witnesses are toys: at a four-gate circuit, `DescriptionHard`,
    `SourceHard` and `Recovers` hold against a decoder that compresses one string, each parameter spent (at 0 it
    fails); at a toy task, coverage holds at η = 1/2 but not at 0, a one-gate guess escaping it. No instance meets the
    premises on a training circuit.
  - The displayed slowdown bound has no Lean statement.

### Promising directions

The next steps test or strengthen part 6's premises: one instance meeting the local premises at once, attacks on
update prices, the certificate checked on a training circuit, a use's import cost, weights by linear programming, and
coverage tested directly. Each is in `security/Definitions/Slowdown/Capability/README.md` ("How to test each premise", "The
frontier"); how the execution's boundaries fix what a block's c and k count is in `security/README.md`, part 2.

## Assumptions, by name

Each is a named `Prop` in a lock's `assumptions` (`security/lean-audit.json`, and `security/Properties/lean-audit.json`
for the `Properties.*` ones), one line in plain words, except `CoinsAfterCommitment` (prose only). These are the named `Prop`s the statements above
read; each theorem's full hypotheses are its Lean, among them `TagsOkZHJ`, `Built`, `StrictCR.Finder.CR`,
`SampledLayer.JointSound` and `.Pins` (which `CircuitCredit` also reads), and the local bounds' structural
hypotheses `HeldPriceAtMost`, `OverlapWeights` and `RequiresState`.

SPOT (`benchmarks/prove_path/README.md`, "What an accepted audit means"):

- `FlockSoundness.Assumptions.UniformRandomBytes`: the verifier's coins are uniform random bytes.
- `FlockSoundness.Assumptions.Zk.RecordCustodyZKJPLaw` (`live-verifier`, read as `RecordCustodyZKJ`): the record the
  verifier reads is the challenger's for the live run, and its sample file is the verifier's own.
- `Verity.Recursive.LiveCoins` (`live-verifier`), in the recursive form: each part's record holds the live run's coins.
- `FlockSoundness.Assumptions.SHA512CRStrict` (`cr/sha-512`) and `FlockSoundness.Assumptions.SHA512CRExpected`
  (`ecr/sha-512`), in `security/Definitions/Assumptions/Flock.lean`: SHA-512 is collision resistant, taken at the prover's
  table and link finders as `TableCRZAt` and `LinkCRZCAt`.
- `trusted-device` (`Verity.Assumptions.TrustedDevice`, `security/Definitions/Core/Assumptions.lean`): the challenger is
  the auditor's device on a host the developer doesn't control, and the custody above rests on it; the verifier runs
  on an auditor device too. In Lean, only onsite's guarantees (network certification's) read `TrustedDevice`.
- `CoinsAfterCommitment`: the challenger's coins reach the developer's machine only over paths the auditor's devices
  control, and only after the commitment. Prose only: no guarantee takes it. Onsite's `CoinsConfidential`
  (`Pouw.Assumptions.CoinsConfidential`, which `Pouw.Guarantees.Dither.FleetSound` takes) is another assumption: a
  replica's coins reach no machine outside its network attester's boundary before it has them.

PoUW, as `H1J10RCheckedCreditMonitoringPeriod` takes them (`security/Properties/Pouw/MonitoringPeriodJ10R.lean`):

- `Properties.Pouw.MonitoringPeriodJ10R.TileServiceSound`: the proof system is sound against `Credit.executionPhase` at each sampled
  proof block, Flock's soundness under live coins. It is a named assumption: no theorem discharges it yet.
- `Properties.Pouw.MonitoringPeriodJ10R.TileSampleIsCoin`: on every accepted execution phase, the proof blocks the check demands proofs of
  are the tile stratum's sample at the phase's coin. A protocol-code fact still to prove.
- `Properties.Pouw.MonitoringPeriodJ10R.RegisteredBeforeDraw`: each execution phase's registration is fixed before its sampling coin. It
  states the order only, and the binding it is used for also needs the rows committed by the registration, which the
  deployed path doesn't do yet.
- `Properties.Pouw.MonitoringPeriod.SaltIsLiveCoin`: each execution phase's salt is its round's live coin. The deployed path doesn't meet
  it: the loop is given its salt before it serves, and nothing in the challenger's record binds the salt to a live coin
  (`security/Properties/Pouw/MonitoringPeriod.lean`, `SaltIsLiveCoin`'s docstring).
- `Properties.Pouw.MonitoringPeriodJ10R.SharedNoiseLucks`: the Duo receipts' lucks (H-μ⁺, `OperandHMu`, with `ReceiptLuckAt`
  and `GuessLuckAt`) at J10-R's shared noise, one computational claim over a call's joint coins.
- `ResourceAccounting.Assumptions.TS.UnitFixed`: the run's fixed and folded words are at least those of the proof blocks it didn't
  skip. It stays an assumption until those words are defined from the committed values.
- `ResourceAccounting.Assumptions.TS.AcceptedLive`, a premise of the proofs behind PoUW's credit (`SessionSoundAt`'s), not of the period credit: an
  accepted proof's check words are `1` as `verity-verify-proof --zk --registered` reads them. No served path meets it yet.
- `ResourceAccounting.Assumptions.TS.ExecutionPhaseDesign`, `ServedChecksHold` and `ServedRootsLinked`, `SessionSoundAt`'s other premises:
  the sample follows its stated sampling design, the served checks hold, and the served roots are linked to the
  registered ones.
- The data sheet: `ResourceAccounting.Assumptions.TS.SheetClock`, `SheetDispatch`, `SheetTensor`, `SheetPipe`, `SheetSmem`,
  `SheetWords`, `SheetSemantics`, `FoldOpt`, `FoldOptFoldedAt` and `FillOpt`: the GPU's clock, rates and semantics are
  the data sheet's, and the folds are optimal. The live-words credit over a monitoring period (`CircuitCreditMonitoringPeriod`,
  `H1J10RCheckedCreditMonitoringPeriod`) doesn't read them. The time credit `ResourceAccounting.Guarantees.Compose.CircuitCredit` does, and it
  is what backs the slowdown bound's `PouwSound` (below), so the slowdown over remote rests on the data sheet.

The slowdown, `SlowdownRemote`:

- In `security/Definitions/Slowdown/Assumptions/Remote.lean`, three:
  - `Slowdown.Assumptions.PouwSound`: each PoUW period's certificate is wrong with probability at most p. What backs p is
    the time credit `CircuitCredit`, which reads the data sheet; no PoUW guarantee instantiates it yet.
  - `Slowdown.Assumptions.SampledDetection`: SPOT catches each incorrect proof block with probability at least q,
    whatever came before.
  - `Slowdown.Assumptions.TerminalDetection`: a detected cheat ends its identity and takes its hardware out of use within Δ.
- (H) and (E), `Slowdown.Cut.Assumptions.Hardness` and `Exports` (`security/Definitions/Assumptions.lean`), on the registered
  circuit's work, and the count's verdict `InferenceOnly.CompliesAtMost c k n` (`security/Definitions/InferenceOnly.lean`), a hypothesis until
  `DescCountsCK` is proved (part 4, link 7).

The local bounds (part 6):

- `Slowdown.Assumptions.DescriptionHard` (in the lock): every selection of an item
  family (updated weights, operands, or both) has time-bounded description length at least its price, less a deficit.
- `Slowdown.Assumptions.CoversEfficientMethods` (in the lock): the floor, a hypothesis about every efficient method. Each
  way of reaching the target below T_max carries some local bound's covered structure, with at least its weighted
  count Ω_*, except with probability η. It is the open part, untested.

## Results per monitoring period

The slowdown calculator, `benchmarks/slowdown`, a third route beside part 6's displayed bound and `SlowdownRemote`,
gives the slowdown bound for monitoring periods of a week, a month and a year (`periods.py`), over one datacenter
(`reference-h200`, the census's 12,500 HGX nodes) and all compute (`world-h200`), at the remote scenario
`reference-remote-h200` (`uv run python benchmarks/slowdown/periods.py --scenario reference-remote-h200`). Run it for
the figures: this page keeps none.

- **What a figure means.** The slowdown over the reference training algorithm at its MFU, MFU₀
  (`Slowdown.Slowdown.hardwareMultiple`): at least 1, and 1 with no protocol.
- **Beside each.** Its range (each ranged parameter alone at either end of its plausible range), its worst corner (all
  at once), and the bound that sets it, with each assumption's standing.
- **The failure probability.** The δ is pooled over the monitoring period (`IntegrityPooled`, `CircuitCreditMonitoringPeriod`),
  not per verification phase. The calculator prints its own: `CertificateSlack`'s p/ε′ at ε′ = 10⁻⁴,
  PoUW's per-phase p assumed at 3·2⁻⁴⁰ (δ_bind, δ_ks unnumbered, part 5), with no term growing with the period.

## The demo

**No run of record of the remote verifier yet:** `remote-v1 check` (to be `verity verify remote`) has none. The remote demo
is `demos/remote.toml` (`demos/README.md`), which `verity run` doesn't accept yet: Pearl+ has no piece. The vLLM workload runs both halves on one workload, PoUW's as a development run:

- the vLLM workload (`adapters/vllm/tools/workload/README.md`): Llama-3.2-1B INT8 served by vLLM on one H200, its
  values committed while serving, a verification phase's proof blocks sampled by live coins, proved by Flock `--zk` and
  checked by the Lean verifier, and PoUW's H1 on the served model. PoUW's verdict there is a development run under the
  challenger's development key and credits nothing, and an H1 commitment's served execution blocks aren't yet a
  `verity/pouw/period/v1` case, so no run of record of the remote verdict has passed: its run is moved, at proof level none,
  and no remote guarantee exists for its verdicts to reach (the open gap, above).

What of PoUW's own check runs is its Lean check, through its difftest against the Python (`lake exe pouw-check`, the
suite `verity-pouw-check-difftest`; `verity/protocols/resource_accounting/compute/pouw/verifier/lean/check/README.md`), and the composite's difftest
(`tests/verity/protocols/compose/verifier/lean`).

## Read next

- `security/README.md`: the definitions, properties and proofs trees, the locks, and the remote protocol in Lean.
- `benchmarks/prove_path/README.md` and `benchmarks/prove_path/e2e_chains/README.md`: what an accepted audit means, and
  each protocol's end-to-end chain.
- `ARCHITECTURE.md` and `GLOSSARY.md` (the Glossary): the architecture, the terms, the notation and the older names.
