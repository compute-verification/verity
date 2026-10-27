# The remote protocol

The remote protocol audits a datacenter whose auditor has no devices at the site: the developer commits everything its
hardware computes, and the auditor checks it from outside with live coins and a Lean verifier. It composes two
subprotocols into one verdict over one commitment: sampled proofs (the committed values satisfy the declared circuit,
except on few proof blocks) and PoUW (the hardware's time went to the declared work).

This file is an outline with signposts. The Lean is the definition. The vocabulary is `GLOSSARY.md`'s. Lean names keep
some older names until they are renamed: `Dampener*` means overhead, `*Horizon` a monitoring period's pooled bound,
`Session*` a run of a subprotocol, and `window` a period.

**Not to be confused with.** Five things in the tree are called "remote":

- the remote protocol, this one: `Core.Remote` (`CoreRemote.lean`, id `verity/remote/v0`);
- PoUB, the remote bandwidth audit (`verity/core/protocols/remote_bandwidth/`, its Lean `RemoteAudit`), an onsite
  subprotocol. Its Lean vocabulary is `Accounting.Remote` (`Security/Definitions/Accounting/Remote.lean`) and
  `Accounting.Guarantees.Remote*`: PoUB's, despite the name;
- `RemoteChallenger` (`verity/core/protocols/sampled_proofs/remote.py`), a challenger reached over the network;
- `RemoteCertifier` (`verity/core/protocols/certifier/active.py`), a network certifier reached over the network;
- the inference-only policy's remote regime (`Nci.Guarantees.DampenerRemote`), the overhead bound over this protocol
  (see the guarantee below).

## Parties

- **The developer** (the prover) runs the datacenter. It registers, declares, commits its values and proves the drawn
  proof blocks behind the firewall. Nothing it runs or hosts is trusted, and neither is the audited hardware or its
  vendor.
- **The auditor** has two parts, both on its own hosts: its **live coins** (the challenger, which draws what must be
  proved) and its **verifier** (the Lean verifier: `flock-verify` for each drawn proof block, and `remote
  check` for the verdict).
- **No auditor devices at the site.** There are no network certifiers or memory challengers. So network certification,
  memory wipes, PoUS and PoUB are onsite's, not remote's. For network certification, see
  `Security/Definitions/NetworkCertifier` and its `Guarantees`.

## Subprotocols

| Subprotocol | What it establishes | Lean check | id |
|---|---|---|---|
| Sampled proofs ([README](../sampled_proofs/README.md)) | the committed values satisfy the declared circuit outside the proof blocks the draw misses | `SampledProofs.OneStage.protocol` (`verity/core/service/sampled_proofs`) | `verity/sampled-proofs/one-stage/v1` |
| PoUW ([README](../pouw/README.md)) | the period's credited work, in live words, is the declared work's | `PouwCheck.protocol` (`verity/core/protocols/pouw/lean/check`) | `verity/pouw/v1` |

**Coming, not there yet:** the inference-only policy (its Lean is still named `Nci`, `NciCheck`) is to join as a third
subprotocol. Under it, the declared circuit divides into inference blocks of bounded size, each reading a bounded amount
of fresh input (input that isn't registered information). Its check reads only the declared circuit and the registered
information, so it needs no hardware at the site. On `main`, `Core.Remote.components` holds only the two rows above.

The code is `CoreRemote.lean`, `protocol = product "verity/remote/v0" [PouwCheck.protocol,
SampledProofs.OneStage.protocol]`. Its executable is `remote id | check PARAMS COMMITMENT EVIDENCE | serve`
(`RemoteMain.lean`). The Lake package `core_remote` is code: it has no Mathlib and nothing of `Security/`
(`lean-audit.json`). The product's rule and its difftest are the composition's ([README](../composition/README.md),
`verity/core/protocols/composition/tests/test_composition_difftest.py`).

## Lifecycle

1. **Registration**, once, before any coin. The developer registers R = (θ, D, 𝓘): the registered inputs θ, the declared
   circuit D, and D's (c, k)-partition 𝓘 into inference blocks (`GLOSSARY.md`, `Verity.Declaration`; read by the
   inference-only policy once it joins). It also registers
   each subprotocol's public Program and parameters, by digest.
2. **Declarations**, one or more per declaration period. Each declaration's declared advice (a bit string that a
   registered compiler compiles, its length charged) appends a declaration block to the declared circuit. Its values are
   committed under hiding leaves (hm96 rows in SHA-512 trees) before that declaration period's audit coins.
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

## The guarantee

**In words.** Suppose the remote verdict accepts every sampling period of a monitoring period. Then, except with a
failure probability set by the draw and by SHA-512, two things hold:

- the committed computation satisfies the declared circuit, except on at most the proof blocks a draw of that size
  misses at δ;
- the period's credited live work is at least (1 − ε) of the declared work.

Everything is about the committed trace. That the declared circuit computes what the served model computes is
completeness, never a security claim.

**The pieces.** "Proved" means its proof is in a lock on `main`. For a guarantee, that's `<Area>.SecurityProofs.G :
<Area>.Guarantees.G` in `Security/lean-audit.json`. For a property of the code, it's its proof in
`Security/Properties/lean-audit.json` (`Properties.Composition.acceptMeans : Properties.Composition.AcceptMeans`). The
last column is the end-to-end chain's status in `benchmarks/prove_path/e2e_chains/README.md`, which can be "none" while
each piece is proved.

| Piece | Lean declaration | Lock | End to end |
|---|---|---|---|
| The verdict rule | `Properties.Composition.AcceptMeans`, `AnswerAccepts` (`Security/Properties/Composition.lean`; proofs `acceptMeans`, `answerAccepts`) | **proved** | — |
| Sampled proofs' accept rule | `Properties.SampledProofs.AcceptMeans` (`Security/Properties/SampledProofs.lean`; proof `acceptMeans`) | **proved** | — |
| A C-Flock `--zk` proof, read in the sampling relation: Pr[accept ∧ faulty proof blocks outside what the draw catches at δ] ≤ δ + ε_ks + δ_link | `Flock.Guarantees.SampledKSLiveV3` (`Security/Proofs/Flock/Bridge/KSLive.lean`) | **proved** | sampled proofs `--zk`: proved |
| Sampled proofs over a monitoring period | `Flock.Guarantees.IntegrityPeriod`, `IntegrityHorizon` (`Security/Proofs/Flock/EndToEnd/Period/Statement.lean`) | **proved** | — |
| The verifier's verdicts tied to those theorems | none: remote reads one-stage v1, which takes the verdicts as evidence; v2 (`verity/sampled-proofs/one-stage/v2`) runs `flock-verify verify --zk` itself, and remote doesn't read v2 yet | **open** | — |
| PoUW's accept rule, and its accepted prefix tied to the period credit | `Properties.Pouw.Check.AcceptMeans`, `Properties.Pouw.Period.CheckedCreditPeriod` (`Security/Properties/Pouw/`; proofs `Properties.Pouw.Check.acceptMeans`, `Properties.Pouw.Period.SecurityProofs.checkedCreditPeriod`) | **proved** | — |
| PoUW's credit over a monitoring period | `Accounting.Guarantees.ComposePeriod.CircuitCreditPeriod` | **proved** | — |
| PoUW's proof layer | `Accounting.Guarantees.TS.SessionSoundAt`: proved, but it needs `AcceptedLive`, which no served path meets yet | **open** as a chain | PoUW `--zk`: broken at the proof layer |
| The overhead over remote, per period | `Nci.Guarantees.DampenerRemote` (`Security/Definitions/Nci/Guarantees/Remote.lean`) | **proved** | none with a proof layer: it takes `PouwSound`, which no PoUW guarantee instantiates |
| The overhead over a monitoring period | `Nci.Guarantees.DampenerRemotePeriod` (`RemotePeriod.lean`): only `DampenerRemotePeriodStated` is in the lock | **open** | — |
| The composed soundness of remote | `Verity.Soundness`: not yet in Lean, and `lean-audit.json` here lists no guarantee | **open** | — |

The pieces compose by hand today: no Lean theorem states remote's guarantee from the rows above.

## Assumptions

Each is a named `Prop` in `Security/lean-audit.json`'s `assumptions`, one line in plain words.

Sampled proofs (`benchmarks/prove_path/README.md`, "What an accepted audit means"):

- `FlockSoundness.Assumptions.UniformRandomBytes`: the verifier's coins are uniform random bytes.
- `FlockSoundness.Assumptions.Zk.RecordCustodyZKJ` (`live-verifier`): the record the verifier reads is the one the
  challenger wrote for the live run, and its draw file is the verifier's own draw.
- `FlockSoundness.Assumptions.TableCRZAt` (`cr/sha-512`) and `FlockSoundness.Discharge.ZkLink.LinkCRZC`
  (`ecr/sha-512`): SHA-512 is collision resistant for the prover's table and link finders.
- `trusted-host` (`Verity.Assumptions.TrustedHost`, `Security/Definitions/Core/Assumptions.lean`): the challenger and
  the verifier run on auditor hosts the developer doesn't control, and the custody above rests on them. In Lean, only
  onsite's guarantees (network certification's) read `TrustedHost` itself. Two premises about where the challenger runs
  have no Lean statement on `main` yet; the sampled proofs README states them: `CoinsConfidential` (its coins reach the
  developer's site only over auditor-host paths) and `OneClock` (its release times and the network certifier's ingress
  times share one clock).

PoUW:

- `Accounting.Assumptions.TS.OperandHMu` (H-μ⁺, the Duo receipts' premise): a cheater can't predict its skipped words'
  values better than μ.
- `Accounting.Assumptions.TS.ReceiptLuckAt` and `Accounting.Assumptions.Compose.GuessLuckAt`: a skipped round passes
  its receipt, or a guessed fold state passes, only at the stated rate.
- `Accounting.Assumptions.TS.AcceptedLive`: an accepted proof's check words are `1` as `flock-verify --zk
  --registered` reads them. No served path meets it yet.
- `Properties.Pouw.Period.DrawIsCoin`, `RegisteredBeforeDraw` and `SaltIsLiveCoin`: the draw is the challenger's live
  coin, the rows are committed before it, and the salt is a live coin. Each is a protocol-code fact still to prove.
- The data sheet: `Accounting.Assumptions.TS.SheetClock`, `SheetDispatch`, `SheetTensor`, `SheetPipe`, `SheetSmem`,
  `SheetWords`, `SheetSemantics`, `FoldOpt`, `FoldOptFoldedAt` and `FillOpt`: the GPU's clock, rates and semantics are
  the data sheet's, and the folds are optimal. The time credit `Accounting.Guarantees.Compose.CircuitCredit` reads them,
  and the live-words credit over a monitoring period doesn't.

The overhead (`Security/Definitions/Nci/Assumptions/Remote.lean`):

- `Nci.Assumptions.PouwSound`: each PoUW period's certificate is wrong with probability at most p.
- `Nci.Assumptions.SampledDetection`: sampled proofs catch each faulty proof block with probability at least q, whatever
  came before.
- `Nci.Assumptions.TerminalDetection`: a detected cheat ends its identity and takes its hardware out of use within Δ.
- `Nci.Assumptions.BaselineRegistered`: the data the registration names was public, not computed off the books before.
- `Nci.Assumptions.WeightsIncompressible`: no small subcircuit rebuilds the weights it uses from fewer bits than
  their width.
- `Nci.Complies`: the declaration meets the inference-only policy.

## Results per monitoring period

The overhead calculator is `benchmarks/rho` (`verity bench rho`; it is being renamed to overhead). Its `periods.py`
gives the overhead's lower bound for monitoring periods of a week, a month and a year. It covers one datacenter
(`reference-h200`, 12,500 HGX nodes) and all compute (`world-h200`), at the remote scenario `reference-remote-h200`:

```text
uv run python benchmarks/rho/periods.py --scenario reference-remote-h200
```

Each figure states which of its parameters are measured on one machine and which are scaled. The δ it reports is pooled
over the monitoring period (`IntegrityHorizon`, `CircuitCreditPeriod`), not per sampling period.

## The demo

**No single example runs the composite yet.** No example's `example.toml` declares `remote`
(`verity example --status`), and `remote check` has no run of record. What runs today is the remote protocol's
sampled-proofs half:

- `verity example h200` (`examples/h200/README.md`): a fixed attention shape committed on one H200. Eight proof blocks
  are drawn by one live coin, each is proved by C-Flock `--zk` and accepted by the Lean verifier, and a flipped bit is
  refused. At an 8-block draw, that allows at most 43.77% faulty proof blocks at δ = 1%. It runs, at proof level
  none, and isn't placed: its Lean verifier ran beside the prover. On a CPU at a small shape, run `uv run python examples/h200/run.py --out /tmp/e2e`
  (`--tamper l1.wup` must exit 3).
- `verity example vllm` (`examples/vllm/README.md`): Llama-3.2-1B served by vLLM on one RTX PRO 6000, its values
  committed while serving, and a sampling period's proof blocks drawn by live coins, proved by C-Flock `--zk` and
  checked by the Lean verifier. It runs, at proof level none: no theorem yet ties those verdicts to a guarantee.

PoUW's half has no runner. `examples/dither_full_util/README.md` has no runner and no run of record. What of PoUW runs
is its Lean check, through its difftest against the Python (`lake exe pouw-check`, the suite
`verity-pouw-check-difftest`; `verity/core/protocols/pouw/lean/check/README.md`), and the composite's difftest
(`verity/core/protocols/composition/tests`).

## Read next

- `Security/README.md`: the definitions, properties and proofs trees, the locks, and the remote protocol in Lean.
- `benchmarks/prove_path/README.md` and `benchmarks/prove_path/e2e_chains/README.md`: what an accepted audit means, and
  each protocol's end-to-end chain.
- `ARCHITECTURE.md` and `GLOSSARY.md`: the architecture, the notation and the older names.
