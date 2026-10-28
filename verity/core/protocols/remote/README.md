# The remote protocol

The remote protocol audits a datacenter whose auditor has no devices at the site: the developer commits everything its
hardware computes, and the auditor checks it from outside with live coins and a Lean verifier. It composes two
subprotocols into one verdict over one commitment: sampled proofs (the committed values satisfy the declared circuit,
except on few proof blocks) and PoUW (the hardware's time went to the declared work).

**The open gap.** Remote's composed soundness theorem, `Verity.Soundness`, is not in Lean yet. It is due after the Lean
dedupe (one vocabulary), and its owner is soundness. Today the pieces below compose by hand: each is proved on its own,
and no Lean theorem states remote's guarantee from them. Nothing yet ties the deployed verifier's verdicts to those
theorems either, because remote reads sampled proofs' one-stage v1, which takes the sessions' verdicts as evidence
instead of computing them.

This file is an outline with signposts. The Lean is the definition. The vocabulary is the Glossary's: `ARCHITECTURE.md`'s
Glossary, and `GLOSSARY.md` for the notation and the older names. The lifecycle's terms (a declaration, declaring, the
declared advice, a declaration block and a declaration period) are in `ARCHITECTURE.md`'s Glossary, under Declaration,
Block and Period. Lean names keep some older names until they are renamed: `Dampener*` means overhead, `*Horizon` a
monitoring period's pooled bound, `Session*` a run of a subprotocol, and `window` a period.

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
  `Nci`) and `benchmarks/rho`'s `regime.remote`: a bound over this protocol (see the guarantee below), not a part of it;
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
  Deployed (the auditor host vy-challenger-1), that port is the far end of a coin link the auditor host opens to the
  developer's machine: `ssh -N -R` straight to that machine's sshd, relayed and counted by network certification's
  code running on the auditor host (`benchmarks/live_challenger/coin_link.py`). No coin path of remote's own, with no
  network certifier anywhere, is specified yet.
- **No auditor devices at the site.** There are no network certifiers or memory challengers. So network certification,
  PoUS and PoUB are onsite's, not remote's, and memory wipes are neither composite's yet. For network certification,
  see `Security/Definitions/NetworkCertifier` and its `Guarantees`.

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
2. **Declarations**, one or more per declaration period (a span of one or more sampling periods). A declaration is one
   act of declaring: the developer reports its declared advice (a bit string that a registered compiler compiles, its
   length charged), and the compiler appends one declaration block to the declared circuit. The block's values are
   committed under hiding leaves (hm96 rows in SHA-512 trees) before that declaration period's audit coins.
   `demos/lifecycle/README.md` ("Terms") runs these terms on a toy model.
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
last column is the end-to-end chain's status in `benchmarks/prove_path/e2e_chains/README.md`. That page's row for the
remote composite is "none" (the open gap), so the column gives its subprotocols' rows (Sampled proofs / C-Flock `--zk`,
PoUW `--zk`), and a chain can be "none" while each piece is proved.

| Piece | Lean declaration | Lock | End to end |
|---|---|---|---|
| The verdict rule | `Properties.Composition.AcceptMeans`, `AnswerAccepts` (`Security/Properties/Composition.lean`; proofs `acceptMeans`, `answerAccepts`) | **proved** | — |
| Sampled proofs' accept rule | `Properties.SampledProofs.AcceptMeans` (`Security/Properties/SampledProofs.lean`; proof `acceptMeans`) | **proved** | — |
| A C-Flock `--zk` proof, read in the sampling relation: Pr[accept ∧ faulty proof blocks outside what the draw catches at δ] ≤ δ + ε_ks + δ_link | `Flock.Guarantees.SampledKSLiveV3` (stated in `Security/Proofs/Flock/Bridge/Refused.lean`, proved in `Security/Proofs/Flock/Bridge/KSLive.lean`) | **proved** | sampled proofs `--zk`: proved |
| Sampled proofs over a monitoring period | `Flock.Guarantees.IntegrityPeriod`, `IntegrityHorizon` (`Security/Proofs/Flock/EndToEnd/Period/Statement.lean`) | **proved** | — |
| The verifier's verdicts tied to those theorems | none: remote reads one-stage v1, which takes the verdicts as evidence; v2 (`verity/sampled-proofs/one-stage/v2`) runs `flock-verify verify --zk` itself, and remote doesn't read v2 yet | **open** | — |
| PoUW's accept rule, and its accepted prefix tied to the period credit | `Properties.Pouw.Check.AcceptMeans`, and `Properties.Pouw.Period.H1CheckedCreditPeriod`, the instance at the deployed H1 tile unit of the general `CheckedCreditPeriod` (`Security/Properties/Pouw/Period.lean`; proofs `Properties.Pouw.Check.acceptMeans`, `Properties.Pouw.Period.SecurityProofs.h1CheckedCreditPeriod` and `checkedCreditPeriod`, all in `Security/Properties/lean-audit.json`) | **proved**, under the PoUW assumptions below | — |
| PoUW's credit over a monitoring period | `Accounting.Guarantees.ComposePeriod.CircuitCreditPeriod` | **proved** | — |
| PoUW's proof layer | `Accounting.Guarantees.TS.SessionSoundAt`: proved, but it needs `AcceptedLive`, which no served path meets yet | **open** as a chain | PoUW `--zk`: broken at the proof layer |
| The overhead over remote, per period | `Nci.Guarantees.DampenerRemote` (`Security/Definitions/Nci/Guarantees/Remote.lean`) | **proved** | none with a proof layer: it takes `PouwSound`, which no PoUW guarantee instantiates |
| The overhead over a monitoring period | `Nci.Guarantees.DampenerRemotePeriod` (`RemotePeriod.lean`): only `DampenerRemotePeriodStated` is in the lock | **open** | — |
| The composed soundness of remote | `Verity.Soundness`: not yet in Lean, and `lean-audit.json` here lists no guarantee (the open gap, above) | **open** | — |

## Assumptions

Each is a named `Prop` in a lock's `assumptions` (`Security/lean-audit.json`, and `Security/Properties/lean-audit.json`
for the `Properties.*` ones), one line in plain words, except the two coin premises below, which are prose only.

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
- `CoinsConfidential` (the challenger's coins reach the developer's machine only over auditor-host paths) and
  `OneClock` (its release times and the network certifier's ingress times share one clock): premises about where the
  challenger runs, stated in prose in `verity/core/protocols/sampled_proofs/README.md` and not yet in Lean. No `.lean`
  file defines either, and no guarantee takes either as a hypothesis.

PoUW, as `H1CheckedCreditPeriod` takes them (the general `CheckedCreditPeriod` takes all but `SaltIsLiveCoin` and
`StagedPins`):

- `Properties.Pouw.Period.ServiceSound`: the proof service is sound at δ_ks against PoUW's check, which is C-Flock's
  soundness at each drawn unit (`Flock.Guarantees.EndToEndRegisteredRows` for one session). It is a named assumption: no
  theorem discharges it yet.
- `Properties.Pouw.Period.DrawIsCoin`: the units the check publishes as drawn are the model's draw at the window's live
  coin. A protocol-code fact still to prove.
- `Properties.Pouw.Period.RegisteredBeforeDraw`: each window's registration is fixed before its draw coin. It states
  the order only, and the binding it is used for also needs the rows committed by the registration, which the deployed
  path doesn't do yet: its registered roots are tags, with no row committed (`anchors []`).
- `Properties.Pouw.Period.SaltIsLiveCoin`: each window's salt is its round's live coin. The deployed path doesn't meet
  it: the loop is given its salt before it serves, and nothing in the record binds the salt to a live coin (`Security/Properties/Pouw/Period.lean`,
  `SaltIsLiveCoin`'s docstring).
- `Properties.Pouw.Period.StagedPins`: the staged statement the service proves pins the flat tile the receipts are
  claims of. Neither side has a Lean function yet.
- `Accounting.Assumptions.TS.OperandHMu` (H-μ⁺, the Duo receipts' premise): a cheater can't predict its skipped words'
  values better than μ.
- `Accounting.Assumptions.TS.ReceiptLuckAt` and `Accounting.Assumptions.Compose.GuessLuckAt`: a skipped round passes
  its receipt, or a guessed fold state passes, only at the stated rate.
- `Accounting.Assumptions.TS.UnitFixed`: the run's fixed and folded words are at least those of the units it didn't
  skip. It stays an assumption until those words are defined from the committed values.
- `Accounting.Assumptions.TS.AcceptedLive`, the proof layer's (`SessionSoundAt`'s), not the period credit's: an
  accepted proof's check words are `1` as `flock-verify --zk --registered` reads them. No served path meets it yet.
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
  rebuilds the weights it uses from fewer bits than their width.
- `Nci.Complies` (`Security/Definitions/Nci.lean`): the declaration meets the inference-only policy.

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

**No example runs the remote verifier yet:** `remote check` has no run of record; its first is due Wed 28 Oct, 11 AM
PDT (D3). Lifecycle is the only example that steps through the whole remote composite, with a stand-in prover; h200
runs remote's sampled-proofs half for real, and the vLLM demo runs both halves on one served run, PoUW's as a
development run:

- `verity example lifecycle` (`demos/lifecycle/README.md`): the remote composite stepped through one monitoring period
  on CPU in about 2 s: hardware registration, registration, declarations, live draws, each declaration's verdict and the
  period's accounting, with every deviation it lists refused by name. Its prover is a labelled stand-in for C-Flock, and
  its verdict is the composition's Python mirror over per-declaration stand-ins for the two subprotocols, not `remote
  check`. It runs; nothing about it is proved.

- `verity example h200` (`demos/h200/README.md`): a fixed attention shape committed on one H200. Eight proof blocks
  are drawn by one live coin, each is proved by C-Flock `--zk` and accepted by the Lean verifier, and a flipped bit is
  refused. At an 8-block draw, that allows at most 43.77% faulty proof blocks at δ = 1%. It runs, at proof level
  none, and isn't placed: its Lean verifier ran beside the prover. On a CPU at a small shape, run
  `uv run python demos/h200/run.py --out /tmp/e2e` (`--tamper l1.wup` must exit 3).
- `verity example vllm` (`demos/vllm/README.md`), the vLLM demo: Llama-3.2-1B INT8 served by vLLM on one H200, its
  values committed while serving, a sampling block's proof blocks sampled by live coins, proved by C-Flock `--zk` and
  checked by the Lean verifier, and PoUW's H1 on the served model. PoUW's verdict there is a development run under the
  challenger's development key and credits nothing, and an H1 commitment's served execution blocks aren't yet a
  `verity/pouw/v1` case, so no run of record of the remote verdict has passed: its run is moved, at proof level none,
  and no remote guarantee exists for its verdicts to reach (the open gap, above).

`demos/dither_full_util/README.md` has no runner and no run of record. What of PoUW's own check runs
is its Lean check, through its difftest against the Python (`lake exe pouw-check`, the suite
`verity-pouw-check-difftest`; `verity/core/protocols/pouw/lean/check/README.md`), and the composite's difftest
(`verity/core/protocols/composition/tests`).

## Read next

- `Security/README.md`: the definitions, properties and proofs trees, the locks, and the remote protocol in Lean.
- `benchmarks/prove_path/README.md` and `benchmarks/prove_path/e2e_chains/README.md`: what an accepted audit means, and
  each protocol's end-to-end chain.
- `ARCHITECTURE.md` (with the Glossary) and `GLOSSARY.md`: the architecture, the terms, the notation and the older names.
