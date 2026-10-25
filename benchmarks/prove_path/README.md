# Prove path

Network accounting under arbitrary vLLM usage proves a served run's step units after the fact: a precommitted compiler turns
the run's advice into each step's Programs, the committing vLLM has recorded each step's values, and the proof service draws
units on live coins, proves them with C-Flock `--zk` behind the Lean firewall, and the Lean verifier accepts or refuses them.
`prove_path.py` is that path on a committed run, a TP Commit (`--tp-commit`), a single-GPU one (`--bundle`, TP1) or a closed
Commit window of a `verity-vllm serve-openai` run (`--serve-run`, TP1): the prover commits the advice under hiding rows (a
served window's the server committed) and each rank's window over the compiler's Programs for it, and the verifier audits each
rank through the one-stage audit's per-rank `service.Stream` on its own coins or a challenger's (`--challenger local`,
`benchmarks/one_stage/a5.py`). `run.sh` builds the prover and the Lean verifier and runs it on the CPU TP stand-in, on a
deferred TP Commit (`PP_TP`), on a single-GPU Commit's row dir from the store (`PP_LLAMA`) or on a served window
(`PP_SERVE_RUN`), with one forged run that must be refused. The docstrings say how.

## What changes when Programs come from advice

A declared run's audit (`a5.py --tp-commit`) derives each rank's Programs from the workload the record served. Under arbitrary
usage no workload is declared: serving decides the batches, arrivals and lengths, and the advice (I1,
`verity.core.protocols.certifier.advice`) records those decisions. The advice is private (Daniel, 24 Oct): it is committed under
hiding hm96 rows and never published.

1. **Registration binds the compiler and the advice's hiding root, not a workload.** The rank's window provenance (core's
   `verity/registration/v1` item `window`, which the record's digest covers) carries `compiler: {name, digest}` and
   `advice: {root, bits}` (`pipeline.advice_windows.binding`). The compiler's digest is `Capped`'s: the inner compiler's digest
   and ℓ, the public cap on the advice's length. The root is the advice's hiding commitment (`advice_windows.commit_advice`:
   I1's canonical encoding in 256-bit rows, hm96-sha512 leaves in one frame-v3-sha512 tree, its domain bound to the compiler).
   `bits` is the advice's length, the leak the certifier charges. Every rank binds the same root: the prover commits the advice
   once. The record gains no field.
2. **Each step's Programs are the compiler's output.** The window's `served_programs` are the digests the compiler produced
   for the advice, and the population program and partition the registration is checked against (R1–R7) are over those
   Programs. The Stream's Check names the unit templates' Program digests, as it does for a declared run.
3. **Named gap: the advice is opened to the verifier until the compile unit exists.** Tonight the verifier is handed the
   commitment's opening (the advice and its salts, `advice_windows.opening`), the way the demo opens the wiring. It recomputes
   the root, compiles the advice with its own copy of the registered compiler, derives its own window
   (`advice_windows.window_of_advice`, reading no committed value), and holds the registration to it
   (`tp.serving_rows.source`: `served_programs`, `compiler` and `advice` among its codes, with R2 against the recompiled
   population). The compile unit below replaces the opening and the recompile. Nothing else in the audit changes.
4. **The advice is fixed before the coins, and the verifier checks it** (`advice_before_draw`), as `pouw.credit.draw_root`
   checks that the credit bits came before the draw coin. The digest of the registration the verifier opens the advice under
   covers its `compiler` and `advice` root. That digest must be the one the draw's receipt names: the stream writes the receipt
   when it receives the registration, and its `select` reads the verifier's coins only after that (a5's
   `verifier-coins-before-the-registration`). A registration whose advice root postdates the draw is refused
   (`advice-root-after-the-draw`): the same advice re-committed after the coins, under fresh salts and so a fresh root, passes
   every check before the draw with its new opening, and `advice_before_draw` refuses it with `after-the-draw`.
5. **The compiler is total.** `compile_advice` takes any input and returns either the requests or a `ValueError` naming why.
   The refusal depends only on the input, so two runs give the same one. It refuses, in order:
   - not bytes;
   - over ℓ, before reading any of it;
   - not I1 advice: the codec's first broken rule (`MalformedAdvice`), truncation and nonzero padding among them;
   - not I1's canonical encoding;
   - a request the compiler can't run: a chunked prefill, a prefix-cache hit, a preemption or a gap (COMPILER's v0 rules,
     `advice_windows.refusals`).

   A malformed opening is refused the same way (`open_advice`). The compile unit's statement quantifies over every advice string
   up to ℓ, not just the honest one, and a refused string has no Programs, so it has no accepting proof.
6. **What the advice can and can't do.**
   - Advice that names a request the Commit didn't serve, or a served one at another arrival, commits no window, because the
     committed store holds no such value (`LookupError` at the prover).
   - A unit committed under other advice and spliced in fails R4.
   - Each of these fails `source` before any coin is read (`more_negatives`, as a5's `advice_negatives`): a registration naming
     another compiler, another advice root, another cap, or none; the opening under another salt; another rank's registration;
     the registration under the compiler listing no transfer gates.
   - The advice can only choose among Programs the compiler emits. Every unit's values are still committed under hiding leaves
     and opened only inside the proof.
7. **Transfer gates are proved whole** (`--transfer-gate T`, both sides). The compiler lists the templates whose units hold
   transfer gates (the I/O shell's), and its digest covers the list. The window's `work:K` law draws every unit of theirs beside
   K of the rest, and the stream registers with that work table. a5's `transfer_gate_negatives` refuse a proved set short of one
   of them, a draw trading one away, and a registration drawing them as the rest. Until COMPILER lists the I/O shell's units, a
   stand-in template stands for them; on Llama TP1 that is layer 0's 160 RoPE heads (`llama32_1b_tg_l0_step1.json`).

What doesn't change:
- the draw: the service's rule, the partition file's law, on live coins read after the receipt, with ε surfaced at δ by the
  outcome;
- the per-member `--zk` sessions behind the firewall;
- Lean's verdicts, the audit record and the Lean `Run.run` agreement.

## What an accepted plain audit means

One property covers both of tonight's chains. In 2b, plain sessions run on the CPU TP stand-in. In 2a, the served window
runs under `--zk`, and that is the demo of record. The chains differ only in which links below are proved. Plain is
`flock-verify verify` without `--zk`, whose guarantee is soundness alone (Daniel, 24 Oct, #4016). The math is modular
(Daniel): each layer is proved once over the layer below and instantiated per configuration.

**The property.** The sampled step units are the sampled class of `NetTiming.Guarantees.SampledAuditAccepted` (#3608,
PRF-144; `Security/Definitions/NetworkCertifier/Guarantees/SampledAudit.lean:34`), stated once at any sampling record `D`:
`∀ H D L K S lay k, Claims.Sampling.KS H D → KSAccepted H D L K S lay k`. Under `--zk`, `D.flock = deployZk rel`; plain,
Flock's plain record. `KSAccepted` (`Security/Definitions/NetworkCertifier/SampledAudit.lean:158`) is the bound below. On
the statements in scope (`ScopeAcc`), at every δ ≥ 0, every decoder within Flock's radius at `D.flock` (`ListWithin`) and
every prover, Pr[acc ∧ ¬RAcc δ] ≤ δ + ε_ks + ε_bind. Its terms:
- acc: the sampled verifier `D.V` accepts.
- P_a(B): the design's probability that the draw misses every unit of B (`Design.Pa`,
  `Security/Definitions/Claims/Sampling.lean:334`).
- ε_ks = κ(δ) − δ + √(γ·Pr[B]) (`κ`, `γ`, the same file, `:731`, `:735`):
  - κ(δ) = c²(δ + κ_F) + (1 + c)(2k/K + e^(−k/8));
  - κ_F is Flock's error at the units' sessions;
  - γ = c⁴·16·(√(2kC) + k)², with c = 2^(2^−40), k the verifier's rounds and C the sessions' weight.
- ε_bind = Pr[B_ext] (`Bext`, `:627`). Pr[B] and Pr[B_ext] are the probabilities of finding a SHA-512 collision: Pr[B] in
  the fork of two runs that share a commitment, and Pr[B_ext] in the extractor's openings. These are events in the run, not
  an assumption.

`RAcc δ` (`Security/Definitions/NetworkCertifier/SampledAudit.lean:135`) holds when:
- the faulty set is in M_δ = {B : P_a(B) > δ} (`RelS`);
- no faulty unit is in a class drawn `all`: the transfer-gate units and N2's head checks (`allSlots`);
- some advice (a, free), with `free` holding values only at the faulty units' committed gates (`FaultyGate`), makes NetTiming's
  `Accepted` (`Security/Definitions/NetworkCertifier/Run.lean:103`) hold over every observation sequence the trace's transfer gates carry (`Carries`), with every status complete
  (`Covered`). With no unit faulty, `free` is empty and `Accepted` is exact.

**The leak bound that follows.** `EncardAccSeqsSampledLe`
(`Security/Definitions/NetworkCertifier/Guarantees/SampledCharge.lean:42`) counts the accepted sequences, with the inputs
besides the ingress ranging over `As`, |As| ≤ 2^c, and each faulty set `RAcc δ` allows committing at most β bits: at most
2^c · #faultySets(δ) · 2^β · D^n. So, except with probability δ + ε_ks + ε_bind, an accepted audit leaks at most c + λ_δ + β_δ + n·log2 D bits:
- c, the advice;
- λ_δ, which units are faulty;
- β_δ, the faulty units' committed bits;
- n·log2 D, the timing (J's D classes over n steps).

At δ ≥ miss(1) no unit is faulty, and the bound is c + n·log2 D. Composing the count with the property is prose: no
theorem states it. It needs the run's inputs to be in `As`, which `RAcc` doesn't bound.

**The charge terms the run prints** (`summary.json` of `prove_path.py audit`):
- c is `advice.bits`, the registration's `window.advice.bits`: the I1 advice's length, which `prove_path.py advice` prints as
  `bits`. Over several windows, `BitsCount` charges their sum.
- δ is `delta`. Per rank, `ranks.<r>.audit.wrong_units` is `{delta, of, bound, rule, per_template}` (the
  `wrong_units` function in `verity/core/protocols/sampled_proofs/one_stage/audit.py`). Except with probability δ, at most
  `bound` of the `of` units are faulty, and at most `per_template[t]` of them are in template t (`Stratified.count_bound`,
  `stratum_bounds`).
- `consumers.exfiltration_bound` gives β_δ + λ_δ. β_δ is the profile's `harm_bound`, weighted by each template's committed
  bits (`_template_bits` in `verity/core/protocols/sampled_proofs/one_stage/partition.py`). λ_δ is `consumers.location_bits`:
  log2 of the at most Σ_{j ≤ bound} C(n, j) faulty sets. prove_path doesn't print either yet. TODO(PROVE-PATH): their
  field names.
- The run doesn't print ε_ks or ε_bind (the profile's `epsilon_log2` is null), or n·log2 D, which is the certifier's term.

**No theorem on `main` takes an accepted audit, `--zk` or plain, to `Accepted` or to the leak bound.**
`SampledAuditAccepted` is stated only, and the step from it to the leak count is prose. What is proved stops at each
session, and, under `--zk`, at sampling's KS given the run's facts. The links below are on `main` at 169118abc1, at the
file and line given, unless a branch is named.

**Shared by both chains** (proved or stated once):
- `flock_verify_sound` (`Security/Proofs/Flock/Soundness/Discharge/FailClosed/One.lean:1705`): one theorem with a part
  per configuration, `zk : Bool`.
- The step from Flock's KS at `D.flock` to sampling's KS, at any record: `Claims.Sampling.SessionKS_of`
  (`Security/Proofs/Flock/Claims/Sampling/Session.lean:661`) and `Claims.Sampling.KS_of`
  (`Security/Proofs/Flock/Claims/Sampling/KS.lean:989`).
- The trim bridge (`Security/Proofs/Flock/Claims/KS/TrimBridge.lean`):
  - KS at an untrimmed record gives KS at its trim (`zk_KS_trim_of_KS` :171, `plain_KS_trim_of_KS` :176).
  - The reverse doesn't hold (`Flock.TrimBridge.hole_agree` :120).
  - So the trimmed record is the weaker statement: its `ListWithin` admits fewer decoders.
- `SampledAuditAccepted`, `EncardAccSeqsSampledLe`, and the missing composition of the two.

**Per configuration:**
- Flock's KS;
- Flock's assembled claims;
- sampling's assembled claims;
- the deployed sampling record. `SamplingDeploy.deploy` (`Security/Proofs/Flock/Claims/Sampling/Deployed.lean:175`) fixes
  `flock := deployZk rel` (line 183), and no plain record exists.

The `--zk` chain (2a):

| Link | Lean name, `main` | Status | Open: owner, date |
|---|---|---|---|
| a `--zk` session is sound | `FlockSoundness.Discharge.FailClosed.flock_verify_sound`, `--zk` part (`Security/Proofs/Flock/Soundness/Discharge/FailClosed/One.lean:1705`). At one table, stratified draw: `SoundZ` :587 and `SoundZH` :1068, with `ViewZ` :838. At J tables, subset draw: `SoundZJ` :678 and `SoundZHJ` :1306, with `ViewZHJ` :1370. With `--registered`: `SoundZHJR` :1638, with `ViewZHJR` :1653 | proved, shared theorem. Premises inside: A3 (`hA3`), the record's custody (`RecordsLiveZK`/`RecordsLiveZKJ`, `DrawOkZK`/`DrawOkZKJ`) and `TableCRZ` (`SHA512CRStrict` at the table finders). The link term is ⊤ unless `LinkCRZC` (`SHA512CRExpected` at the link finders). Lock: frozen `hash,world-unlisted` | — |
| a `--zk` session, end to end at J tables | `Flock.SecurityProofs.EndToEnd : Flock.Guarantees.EndToEnd` (`Security/Proofs/Flock/EndToEnd.lean:22`; statement `Security/Proofs/Flock/EndToEnd/Statement.lean:60`) | proved. Its named assumptions are hypotheses: A3, `Zk.RecordCustodyZKJ`, `TableCRZ`, `LinkCRZC`. Lock: `Security/lean-audit.json:5461`, frozen `hash,world-unlisted,scope` | — |
| Flock KS at the masked record | `FlockDeploy.KS.zk_KS_untrim` (`Security/Proofs/Flock/Claims/KS/ZkExtend.lean:452`), untrimmed `deployZk` | proved, no premise. The trimmed form follows by the shared bridge | — |
| Flock's claims at the masked record | `FlockDeploy.Assembled.flock_zk_assembled` (`Security/Proofs/Flock/Claims/Assembled.lean:227`), at `deployZkTrim` | proved. Its hypotheses: KS (`hKS`), completeness (`hComplete`), the ZK models' links (`hL`, `hS`, `hN`, `hI`, `hC`, `hG`) and `Claims.Hm96Hiding` (`hT1`) | — |
| sampling's KS | `Claims.Sampling.sampling_assembled` (`Security/Proofs/Flock/Claims/Sampling/Assembled.lean:92`, at `D.flock = deployZk rel`); `SamplingDeploy.sampling_deployed_sha512` (`Security/Proofs/Flock/Claims/Sampling/DeployedSha512.lean:32`, at `deploy`, with KS from `zk_KS_untrim`) | proved, with the sampled run's facts as hypotheses: `DrawFixed`, `SessionRuns`, `SessionsIn`, `DrawSlots`, `SessionOpens`, `InteriorFixed`, `ProgramScope`, `ChecksScope`, `RevealsLe`, plus `Complete` and `ZK` for those conjuncts | — |
| the audit implies `Accepted` | `NetTiming.Guarantees.SampledAuditAccepted` (`Security/Definitions/NetworkCertifier/Guarantees/SampledAudit.lean:34`) | stated only, shared. The lock freezes it as `unstated` through `NetTiming.SecurityProofs.SampledAuditAcceptedStated` (`Security/Proofs/NetworkCertifier/SampledAudit.lean:7`, `rfl`) | PRF-144: FINTAU's plan by 06:00Z |
| the sampled leak count | `NetTiming.SecurityProofs.EncardAccSeqsSampledLe` (`Security/Proofs/NetworkCertifier.lean:116`, by `NetTiming.encard_accSeqsSampled_le`, `Security/Proofs/NetworkCertifier/SampledCharge.lean:296`) | proved, shared. No named assumption; its premises are inside the statement. It doesn't read `SampledAuditAccepted` | — |
| acceptance, then the leak bound | none | prose only, shared. It needs the run's inputs in `As`, which `RAcc` doesn't bound | network-accounting's LEAK-LEAN, tonight |

The plain chain (2b):

| Link | Lean name, `main` | Status | Open: owner, date |
|---|---|---|---|
| a plain session is sound | `flock_verify_sound`, plain part (`Security/Proofs/Flock/Soundness/Discharge/FailClosed/One.lean:1705`). Public outputs: `SoundExec` :129. Hidden outputs: `SoundExecH` :549. At 2^−205 a table: `flock_verify_sound_205` (`Security/Proofs/Flock/Soundness/Discharge/FailClosed/Numeric.lean:700`) | proved, shared theorem, at one table and the stratified draw. Premises inside: A3, the record's custody (`RecordsLive`, `DrawOk`) and `TableCRL` (`SHA512CRStrict`). The link term is ⊤ unless `LinkCRL` (`SHA512CRExpected`). Per session; it says nothing of `Accepted` | custody named and listed: REROWS, by 12:00Z 25 Oct |
| Flock KS at the trimmed plain record | `FlockDeploy.KS.plain_KS` (`Security/Proofs/Flock/Claims/KS/PlainLeafKS.lean:392`), at `deployPlainTrim` | proved, no hypothesis. `plain_KS_of_liveAcc` (`Security/Proofs/Flock/Claims/KS/PlainAcc.lean:183`, from `PlainLiveAcc`) and `plain_KS_of_w` (`Security/Proofs/Flock/Claims/KS/PlainWrP.lean:252`, from `PlainLiveW`) reach the same trimmed record | — |
| Flock KS at the untrimmed plain record, which sampling reads | `plain_KS_of_acc` (`Security/Proofs/Flock/Claims/KS/TableBound.lean:462`); `plain_KS_sha512` (`Security/Proofs/Flock/Claims/KS/Width.lean:316`) | proved only from the full-width table bound (`TableBoundθ` at `orZero (wide false)`), which is open. `plain_KS_sha512` also takes one table and `ConflictTied`. The `PlainLiveAcc` route reaches only the trimmed record on `main` | DECOMPB, date to come (about 05:00Z); `PlainLiveAcc` stands while it's open |
| Flock's claims at the plain record | `FlockDeploy.Assembled.flock_plain_assembled` (`Security/Proofs/Flock/Claims/Assembled.lean:158`), at `deployPlainTrim`; `_trim` takes `PlainLiveAcc` in place of KS | proved, with KS and completeness as hypotheses | — |
| sampling's KS | none on `main` | not proved; in progress | FINTAU (`cursor/proofs-plainsamp-944c`): a generalization of `sampling_assembled` over any Flock deployment with KS, instantiated at `deployZk` and plain, not a copy. Its head, 98dff30552, is a copy at the trimmed record: `sampling_assembled_plain` at `D.flock = deployPlainTrim rel`, whose sampling KS covers only the decoders the trimmed spec admits. It must meet the trim finding: the untrimmed record needs the row two above |
| the audit implies `Accepted` | `SampledAuditAccepted` | stated only, shared | PRF-144: FINTAU's plan by 06:00Z |
| the sampled leak count | `EncardAccSeqsSampledLe` | proved, shared | — |
| acceptance, then the leak bound | none | prose only, shared | LEAK-LEAN, tonight |

So an accepted audit gives, at each session it proves:
- 2b: `flock_verify_sound`'s plain part;
- 2a: its `--zk` part, and `EndToEnd` at J tables.

Under `--zk`, sampling's KS also holds, given the run's facts. Neither chain says anything yet about the egress.

**The named assumptions.** Lock entries are keys of `assumptions` in `Security/lean-audit.json`.

| Assumption | Lean name | Lock entry | Read by |
|---|---|---|---|
| SHA-512 collision resistance | `FlockSoundness.Assumptions.SHA512CRStrict`, `SHA512CRExpected` (`Security/Proofs/Flock/Soundness/Assumptions.lean:58`, `:83`); `Claims.StrictBirthday`, `Claims.ExpectedCR` (`Security/Definitions/Claims.lean:313`, `:322`) | `Security/lean-audit.json:914`, `:910`, `:785`, `:765` | `flock_verify_sound` in both chains: plain through `TableCRL` (`Security/Proofs/Flock/Soundness/Discharge/Exec/StrictCRL.lean:228`) and `LinkCRL` (`Security/Proofs/Flock/Soundness/Discharge/Exec/FlockLinkL.lean:565`); `--zk` through `TableCRZ` (`Security/Proofs/Flock/Soundness/Discharge/ZkLink/StrictZ.lean:192`) and `LinkCRZC` (`Security/Proofs/Flock/Soundness/Discharge/ZkLink/CompiledLink.lean:554`). Also `EndToEnd`. Sampling's KS and `SampledAuditAccepted` take none: their bound carries the collision events |
| A3, the coins | `FlockSoundness.Assumptions.UniformRandomBytes` (`Security/Proofs/Flock/Soundness/Assumptions.lean:113`) | `Security/lean-audit.json:918` | `flock_verify_sound` in both chains, and `EndToEnd` |
| record custody | `--zk`: `FlockSoundness.Assumptions.Zk.RecordCustodyZKJ` (`Security/Proofs/Flock/Soundness/Assumptions/Zk.lean:48`), `RecordsLiveZKJ` with the draw file's custody. Plain: none named; `RecordsLive` and `DrawOk` (`Security/Proofs/Flock/Soundness/Discharge/DecodeZL/Statements.lean:95`, `:106`) are premises | `--zk` only: `Security/lean-audit.json:926`. Plain: REROWS names and lists it by 12:00Z 25 Oct | `EndToEnd` names it. `flock_verify_sound` takes its parts as premises in both chains |
| `Claims.Hm96Hiding`, privacy, not soundness | `Claims.Hm96Hiding` (`Security/Definitions/Claims/Hm96.lean:33`) | `Security/lean-audit.json:777` | `flock_zk_assembled`'s zero knowledge (`hT1`); no soundness link |
| `trusted-nebius-host` | `Verity.Assumptions.TrustedNebiusHost` (`Security/Models/Core/Assumptions.lean:50`) | `Security/lean-audit.json:1748` | the warden's guarantees (`Security/Models/NetworkCertifier/Guarantees/Warden.lean`). No link above: it is why `Carries`, inside `RAcc`, holds of the real run |
| W2 | `NetTiming.Regenerates` (`Security/Models/NetworkCertifier/Assumptions.lean:28`) | `Security/lean-audit.json:1313` | the wire bounds `EncardDecodableWithinLeCompiled*` (`Security/Models/NetworkCertifier/Guarantees/AdviceString.lean:51`, `:69`, and their V1 forms) and the ingress bounds (`Security/Models/NetworkCertifier/Guarantees.lean:202`–`274`); no link above |
| W3, W5, W6 | `NetTiming.Padded`, `UntimedStatus`, `FailClosed` (same file, `:33`, `:50`, `:59`) | none | no listed guarantee reads them |
| W4 | `NetTiming.Packed`, `Allocated` (same file, `:37`, `:44`) | `Security/lean-audit.json:1309`, `:1277` | the ingress bounds; no link above |
| J | `NetTiming.JitterBounded` (same file, `:66`) | `Security/lean-audit.json:1297` | the wire and ingress bounds; no link above |
| the advice: `CompilerDeterministic`, `CompilerPrecommitted` | `NetTiming.*` (`Security/Models/NetworkCertifier/Assumptions/AdviceString.lean:26`, `:31`) | `Security/lean-audit.json:1285`, `:1289` | the wire bounds, which hold at exact acceptance only; no link above |
| the advice: `BitsCount` | `NetTiming.BitsCount` (same file, `:21`) | `Security/lean-audit.json:1281` | the wire bounds, `EncardAccSeqsWithinLeConstantRate`, the warden's. The sampled count takes \|As\| ≤ 2^c as a premise instead |

The wire bounds would carry the sampled count from what the transfer gates carry to what a decoder reads off the wire.
No sampled form of them exists yet.

**Tonight's interim, and plain's staging.**
- **2b's plain sessions are inside what `flock_verify_sound` covers today.**
  - Each session has one table: `ProvedScope.statementRules`
    (`verity/core/service/flock/Flock/ProvedScope.lean:192`) refuses more without `--zk`.
  - Each draws by the stratified law: `ProvedScope.lawWhy` (line 278) refuses a subset draw without `--zk`, and a
    bernoulli or work draw in either mode. a5 writes each session's draw as `subset` (`benchmarks/one_stage/a5.py:580`), so
    a plain session drawn as a5 draws is refused.
  - Hidden outputs are covered at one table (`SoundExecH`).
- **Registered reads are not checked in plain mode.**
  - Without `--zk`, `flock-verify verify --registered F` accepts exactly what `verify` does: `FlockVerifyAcceptsR false`
    is `Verifies` (`Security/Proofs/Flock/Soundness/Discharge/FailClosed/One.lean:66`).
  - What fails closed is a shared-row public file: `ProvedScope.sharedRowsOk` (line 222) refuses one without `--zk`.
- **The drawn units' committed rows are not opened outside the session in plain mode** (proofs, thread
  1792894842.280569).
  - `keyRowsH_computes` (`Security/Proofs/Flock/Soundness/Discharge/HiddenExec/KeyH.lean:468`) already binds each drawn
    row's b‖c to its input words, up to a SHA-512 collision, and the registered tree's leaves are built over the same b‖c.
  - A check outside `flock_verify_sound` would be a new form ahead of its proof.
- **A plain served window waits for proofs' PRs.** Tonight, the served window runs under `--zk` (2a).

| PR | Worker, branch | Target | What it proves |
|---|---|---|---|
| A | DECOMPCOMP, `cursor/proofs-plainmulti-1ba6` | Mon 26 Oct | plain at 1–64 tables under the subset draw |
| B | PLAINB, `cursor/proofs-plainhidden-*` (not pushed yet; PLAINB just started) | Tue 27 Oct | plain hidden outputs at J tables, beyond `SoundExecH` at one table |
| C | PLAINC, `cursor/proofs-plainreg-7c3e` | Mon 26 Oct at one table, Tue 27 Oct lifted | plain checked registered reads and shared rows. Its widening of `sharedRowsOk` to plain shared-row files is in progress, and lands only with `FailClosed.SoundExecHR`, its proof, which is on neither `main` nor the branch yet |

The plain served window follows, Wed 28 Oct.

**Off the page:** whether the circuit matches the model. That is completeness: an honest run on a circuit that doesn't
match is refused. It is never a soundness gap.

## The compile unit (the target; not built)

This is the spec for whoever builds the unit (circuits' COMPILER, or a new worker). It closes the named gap: the advice is never
opened, and the verifier never recompiles.

- **Selection All.** Each registration has one compile unit, always proved. Its Spec's selection is `service.All`, so it is
  never sampled. It reads no coins, and its proof goes through the firewall like every unit's.
- **Public inputs.** Only kinds AGENTS.md allows (protocol structure and hiding commitments):
  - d_C, the registered compiler's digest (`Capped.digest`), which fixes the compiler's code and ℓ;
  - ρ, the advice's root (`window.advice.root`);
  - b, the advice's length in bits (`window.advice.bits`, which is public today and is the length term NETREV charges);
  - Π, what the window's Programs commit to: either the window's `served_programs` (request id ↦ Program digest) with the
    population Program's and partition's digests that R1–R7 hold the registration to, or, with COMPILER's padded public step
    templates, the root of a hiding commitment to the routing (which committed row of each step feeds which public template
    instance).
- **Witness.** The advice a, with |a| = b ≤ ℓ, the salts y_0 … y_{R−1} of its R = ⌈b / 256⌉ rows, and, for a committed routing,
  the routing's rows and salts.
- **Statement.** The unit outputs one bit, and it is 1 exactly when all three hold:
  1. ρ is the root of a's rows:
     - row i is a[32i … 32i + 31], zero-padded;
     - leaf i is `verity.ml.commitments.advice.leaf(D, 256, i, row_i, y_i)`: the hm96-sha512 commit string of the row's
       sha512/row/v2 digest under y_i, as an `hm96-sha512/row/v2` row leaf at rank i;
     - the tree is the frame-v3-sha512 tree over those leaves, with domain D = CommitmentDomain(`advice_windows.commitment_binding(C)`,
       −1, RangeIndexedDomain(R), sha512).
  2. compile_C(a) is not a refusal. compile_C is `compile_advice` under d_C: ℓ first, then I1's decode, the canonical
     encoding, and the compiler's rules.
  3. Its Programs P meet Π: the digest of each request's Program is `served_programs`' entry for it (and P's population Program
     is the registered one), or P's routing is the committed routing under Π's root.
- **Quantified over every advice string up to ℓ.** For every a ∈ {0,1}^{≤ℓ} and every set of salts: if the unit's proof verifies
  under (d_C, ρ, b, Π), then ρ opens to a (hm96 binding and SHA-512's collision resistance), compile_C(a) is not a refusal, and
  its Programs meet Π. compile_C is total and deterministic, so every ill-formed string has exactly one refusal and no accepting
  proof.
- **Which Π.** A clear `served_programs` gives away each request's shape, which is a function of the advice. So only the committed
  routing over public row programs keeps the advice private. The clear map is a stepping stone.
- **What it retires:** the opening handed to the verifier (`audit --opening`, `advice_windows.of_opening`), the verifier's
  recompile, and, with the routing form, `served_programs` in the clear.
- **What stays:** `advice_before_draw`, which still holds ρ to the registration the draw followed.

## N2 and tie units on the same Stream

N2-ZK changes `n2_zk.py` on its own branch (`cursor/network-e2e-061a`). Today its head-check units are drawn by the Lean's
`flock-verify draw --stratified n` and audited with `verifier.audit_record` (`audit_member`), outside a `service.Stream`; its
`LinkWindows.stream` is the certifier's records' stream, not the proof units'. To draw them on the one proof service, as
`a5.stream_of` and `a5.coins_draw` do for step units, the interface N2 needs from `verity.core.protocols.sampled_proofs.service`
is:

1. Build the Spec:
   `Spec(values, selection, Check(program, query, proof_deadline=…), public, context)`.
   - `values`: one `Value(port, schema, "serve")` per committed port of the member, with its hiding leaf schema.
   - `selection`: `All()`, since every head check is proved, as N2 draws every unit today.
   - `program`: the head-check adapter's Program digest, `partition.program_digest(Program(unit))`.
   - `query`: the member's `Q_template_instance`.
   - `public`: `Public(item, "structure", why)` for the population, the link-window, the leaf layer and each port's leaves and
     domain.
   - `context`: the link-window's registered root, as bytes.
2. Make the stream: `Stream(spec, unit, budget=…, delta=…, challenger=…, coins=VerifierCoins())` (`a5_coins.VerifierCoins`; with
   `All`, `select` reads no coin).
3. Register the member: `reg = stream.register(rec, ports, leaf_layer=…, window_root=…, counts=…)`, where `rec` is the member's
   one-stage registration.
4. Draw: `draw = stream.select(reg)`. `draw.draw` is the firewall's `--draw-file`.
5. Prove: one `--zk` session per member (`a0.session(..., zk=True, draw_file=…)`).
6. Deliver: `stream.deliver(draw, [SHA-512 of each session record])`.
7. Decide: `stream.outcome(draw, verdicts, statement, proved_units=…, instances=…, served=…, session_roots=…)`. Its audit record
   fails `run` unless the Lean's `Run.run` agrees (`a5.verifier_outcome`).

`audit_member`'s Lean draw becomes `select`, and its `audit_record` becomes `outcome`; the registration and the sessions stay
as they are. The I/O tie units take the same seven steps, with their own Values, Program and selection. The tie is a design for
now (the certifier's protocol, on branch `cursor/io-tie-061a`).

## Status

- **The compiler**, with ℓ = `CAP_BITS` (1 MiB) unless `--cap-bits` gives another:
  - on a TP Commit, `Capped` over `advice_windows.Traced`: the Build's Programs of record, refused unless they are of record.
    On a declared run it emits exactly what a correct compiler emits. I2 doesn't compile a TP rank's Programs yet.
  - on a single-GPU Commit, `Capped` over `I2Compiler`: I2's `predict.steps.compile_steps` (v0) on the advice, under φ from the
    row's declared inputs (`score.phi_of`: the row id, the served workload file, refused unless its digest is the record's, and
    the checkpoint's `config.json`), with each request's prompt and sampling from the record. Its digest names I2's code
    (`code_identity`: `verity` and `verity_vllm` as git lists them, docs and ignored build outputs aside), φ, the seed and ℓ.
    A Program's body is read from the Build dir whose `result.json` names I2's digest, refused where none does. On
    Llama-3.2-1B's TP1 Commit r20261024-055630-5377, I2's four Programs equal the Build's `build_request_LP*` digests
    (`commit.json`'s and the audit summary's `i2`).
  - on a served window, the same over φ of the served engine as vllm's advice compiler (#3619) compiles under (`phi_of_serve`:
    `serve.json`'s model, revision, target profile and execution label, the checkpoint's `config.json`), the server's default
    seed, and each request's body as that compiler gives it (`serve_ingress`: the sampling fields its ingress record says it
    brought, with its prompt ids, by its I1 id: its `user`, else its HTTP id). It is named as `verity-vllm advice` registers it
    (`served_identity`: `compile_steps` by module path, its digest over `source_identity.code_identity`, which covers
    `integrations/vllm` and `verity`), so the audit runs on a tree whose two are the server's. `prove_path.py compiler
    --serve-run DIR` prints the binding's compiler. I2's Programs are held to the ones vllm's advice compiler wrote into the
    row dir (`manifest.json`, by I1 id).
- **A served window** binds the root serve-openai published: the advice is committed once, by the server, and the prover
  commits the window under the published binding. Before any commit and before the draw (folded into the source check a5
  gates the draw on), `serve_check` holds the published records to the verifier's own and refuses by name: another window
  (`window`), a compiler whose digest carries no cap (`cap`) or isn't the verifier's (`compiler`), another advice root
  (`advice`), a published root that isn't 64 bytes of a `rows-sha512` store (`sha512`) or isn't the row dir's run root
  (`run_root`), and a registration naming another Commit (`commit`). Its negatives run beside the audit's. `serve-standin`
  lays a single-GPU Commit of record out as a served run, the stand-in until serve-openai seals its windows.
- **A single-GPU window** is a stand-in of the Commit's `rows-sha512` run root (`stand_in1`, held by `source1`), as
  `deferred_rows.from_bundle` writes one. The sealed bundle records the Commit's absolute node paths, and none is read: the
  Programs come from `--row-dir`, and the Build manifest falls back to `<row dir>/manifest.json`, checked against the record's
  sha256.
- **The advice** is I1's canonical encoding. A declared run's comes from its workload (`advice_windows.advice_of_workload`), a
  served run's from ADVICE-SRC's step trace.
- **The opening** is the named gap above. `run.sh` keeps neither the advice nor its opening in `$OUT`.
