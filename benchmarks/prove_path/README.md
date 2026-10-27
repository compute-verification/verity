# Prove path

Network accounting under arbitrary vLLM usage proves a served run's step units after the fact: a precommitted compiler turns
the run's advice into each step's Programs, the committing vLLM has recorded each step's values, and the proof service draws
units on live coins, proves them with C-Flock `--zk` behind the Lean firewall, and the Lean verifier accepts or refuses them.
`prove_path.py` is that path on a committed run, a TP Commit (`--tp-commit`), a single-GPU one (`--bundle`, TP1) or a closed
Commit window of a `verity-vllm serve-openai` run (`--serve-run`, TP1): the prover commits the advice under hiding rows (a
served window's the server committed) and each rank's window over the compiler's Programs for it, and the verifier audits each
rank through the one-stage audit's per-rank `service.Stream` on its own coins or a challenger's (`--challenger local`,
`examples/one-stage/a5.py`). `run.sh` builds the prover and the Lean verifier and runs it on the CPU TP stand-in, on a
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

One property covers both configurations. 2a, the served window under `--zk`, is the demo (top, 25 Oct). 2b, plain on the
CPU TP stand-in, has soundness as its only guarantee (Daniel, 24 Oct, #4016). Each layer is proved once over the layer
below and instantiated per configuration. On main since 25 Oct, the property (#4054) and the leak step (#4055 with
#4044) are proved given sampling's KS and listed in the lock, both at option A since #4090, and sampling's KS is proved
under `--zk` and at plain's trimmed record given the run's facts (#4037, #4075). **No theorem yet connects the
session verdicts the verifier of record runs to that chain** (the broken shared row below). So 2b demonstrates
per-session plain soundness, not an audit guarantee.

**2a's demo audits a stand-in, not the whole window.** The units it draws from are the ones its partition file picks,
not a draw from every unit of the window's computation. At the demo of record they are 320 RoPE heads
(`RoPEHead_v1{D=64}`, the transfer-gate stand-in) and 8 `SiluMul_v2{I=8192}`, all from layer 0. Its circuit is those 328
units, with every input free. NET-68, below, makes the circuit the whole window.
- **Demo of record, through #4061's interface, on live coins:** gate run r20261025-100403-334e on vy-nebius-1, from
  `cursor/demo-2a-tree-061a` at 4e7d29bfed, against vllm's serve run r20261025-092311-15a1 (port 18091, registered
  compiler 56af1ac0). The coins came from vy-challenger-1, session s20261025T102337Z-b051: 104 draw coins, and one open
  of window 60's root adef445e.
  - The honest arm, through `records_audit`, was accepted: Lean draw and verify on both members, 322 of the 328 units
    drawn, and 25 of 25 negatives refused, R4 and `advice-root-after-the-draw` among them.
  - The forged arm, on local coins (r20261025-093852-2ec5), was refused by Lean: M0's R7, no proof for the forged rep.
  - The charge: at most 2 faulty proof blocks of 328 at δ = 0.5, with 1,679 advice bits under a cap of 8,388,608.
  - #`freeInputs` is 2,752,512 bits, or 2,097,152 if the RoPE stand-ins' inputs counted as ingress. 2a registers none of
    them, so c is all of them even under option A, and the leak bound is vacuous until NET-68.
- **First demo of record:** r20261025-041404-a1c5, on the code before #4061's interface. It drew from 8 layer-0
  `SiluMul_v2{I=8192}` units, with 2,097,152 free input bits against 1,687 declared.

**The property.** The sampled step units are the sampled class of `NetTiming.Guarantees.SampledAuditAccepted` (#3608,
PRF-144). It is stated once, at any sampling record `D`:
`∀ H D L K S lay k, Claims.Sampling.KS H D → KSAccepted H D L K S lay k`, with `D.flock = deployZk rel` under `--zk` and
Flock's plain record otherwise. It is proved on main (#4054) as `NetTiming.SecurityProofs.SampledAuditAccepted`, with no
new assumption, and the lock lists it as that guarantee (owner proofs) since #4090, which landed #4083's lock entry.
`SampledAuditAcceptedStated`, which held its text until then, is retired.

On the statements in scope (`ScopeAcc`), at every δ ≥ 0, every decoder within Flock's radius and every prover,
Pr[acc ∧ ¬RAcc δ] ≤ δ + ε_ks + ε_bind, with ε_ks = κ(δ) − δ + √(γ·Pr[B]) and ε_bind = Pr[B_ext]. Pr[B] and Pr[B_ext] are
the chances of finding a SHA-512 collision in the run, not an assumption. `RAcc δ` holds when:
- the faulty set is in M_δ = {B : P_a(B) > δ} (`RelS`);
- no faulty proof block is in a class drawn `all`: the transfer-gate units and N2's head checks (`allSlots`);
- the advice (`traceVal`, free), its inputs the trace's committed values and `free` holding values exactly at the faulty
  proof blocks' committed gates (`FaultyGate`), makes NetTiming's `Accepted` hold over every observation sequence the trace's
  transfer gates carry (`Carries`), with every status complete (`Covered`). With no unit faulty, `free` is empty and
  `Accepted` is exact. Before #4090 (option A) the advice's inputs could be any values.

**The leak bound that follows.** The theorem is `NetTiming.sampledLeak_reg` (proofs' #4090, over #4055's
`NetTiming.sampledLeak`). The lock-listed guarantee is `NetTiming.Guarantees.SampledLeakAccepted`, proved as
`NetTiming.SecurityProofs.SampledLeakAccepted := fun hSA => NetTiming.sampledLeak_reg hSA` (#4044, restated by #4090;
owner network-accounting; `SampledAuditAccepted` and `Claims.Sampling.KS` owned by proofs). Status: proved and listed in
the lock on main, with #4073's `check.py` fix, on `propext`, `Classical.choice` and `Quot.sound`. Its one hypothesis,
`SampledAuditAccepted`, is a listed guarantee since #4090. The lock keeps `NetTiming.Guarantees.SampledAuditAccepted`
under `assumptions` too, as the premise this statement takes.

It composes the property with the count `EncardAccSeqsSampledLe`, over any deployment. Given `SampledAuditAccepted` and
`Claims.Sampling.KS` at the deployment, except with probability δ + ε_ks + ε_bind, an accepted audit's carried
sequences, cut at the horizon, number at most 2^c · #faultySets(δ) · 2^β · D^n, at whatever values the statement's
registered inputs `reg` hold (`AsReg`). So the audit leaks at most c + λ_δ + β_δ + n·log2 D bits:
- c = #(`freeInputs` \ `reg`): the free inputs are the circuit's input gates that no step before the horizon lists as
  ingress. `RAcc` pins the inputs to the trace, so the count leaves out the trace's values at `reg` and charges every
  other free input. With `reg` empty it is #`freeInputs`, the bound before #4090;
- λ_δ, which units are faulty;
- β_δ, the faulty proof blocks' committed bits;
- n·log2 D, the timing (J's D classes over n steps).

At δ ≥ miss(1) no unit is faulty, and the bound is c + n·log2 D. Take 2a's circuit at the demo of record (334e). Its
328 units have 172,032 16-bit inputs, and no step lists any of them as ingress. So #`freeInputs` = 2,752,512 bits,
against the advice's declared 1,679. At a1c5 it was 2,097,152 against 1,687, and 2b's advice declares 169
(r20261025-032237-db01). So the bound is vacuous at 2a's size.

**Option A is on main** (proofs' #4090, NET-68's P1a). It is EVALCOMP's ask on #ask-daniel (card ddf12c5c,
[its page](https://app.notion.com/p/404399515d9e81d99e61f6534b54b291)), decided provisionally as A on 25 Oct: `RAcc` pins
the inputs to the trace's committed values, and `SampledLeakAccepted` counts at the registered inputs pinned, so
c = #(`freeInputs` \ `reg`) and the registered inputs, the weights among them, stop counting. That the trace's values
at `reg` are the registration's, and so public, is P1b: every registered gate is opened under the registration's root,
and binding fixes its value there except at ε_bind. P1b is on proofs' branches, with no PR yet. `--zk` has the
registered-read binding. Plain's `SoundExecHR` (PR C, #4079, at one table) is in the game form, so it gives that
binding only through the broken shared row below.

A doesn't shrink 2a's c (proofs). 2a registers none of its inputs, and its bits are layer-0 values that the layout
doesn't declare as ingress. So 2a's c is still all of its free inputs, non-ingress and non-registered, and its bound is
vacuous until NET-68's whole-window circuit lands. That is a layout and compiler change, not a claim: NET-68, owned by
network-accounting, with vllm's compiler part as VLLM-103 under it.
- The window's circuit becomes the whole window under a step-indexed scan router. `Fits` is structural, and arrival
  steps and prompt lengths stay private (#4087 landed; #4102, #4108 and #4175 are open).
- The start cadence is ASK-122, top's #ask-daniel card for Daniel. If each second's admission count is public, the
  first token comes at 2.1 s. The default publishes only each window's count, once per 60-bucket window, at 7.1 s. A
  first token within 1 s is out under every option.

Until then, vllm's #4061 has `records_audit` label the bound vacuous whenever #`freeInputs` exceeds the declared charge.

**The charge terms the run prints** (`summary.json` of `prove_path.py audit`; with #4047, each rank's
`ranks.<r>.charge` holds `advice_bits`, `delta`, `wrong_units`, `location_bits`, `harm_bound` and `exfiltration_bound`):
- c is #(`freeInputs` \ `reg`) (#4090), which the run doesn't print. For 2a, which registers no inputs, it is all
  2,752,512 free input bits, counted from 334e's `window/program.json` by `freeInputs`'s definition (#4055). Once the
  advice is the only free input outside the registration (for 2a, after NET-68), c is `advice.bits`, the
  registration's `window.advice.bits`: the I1 advice's length, which `prove_path.py advice` prints as `bits`. Over
  several windows, `BitsCount` charges their sum.
- δ is `delta`. Per rank, `ranks.<r>.audit.wrong_units` is `{delta, of, bound, rule, per_template}` (the
  `wrong_units` function in `verity/core/protocols/sampled_proofs/one_stage/audit.py`). Except with probability δ, at most
  `bound` of the `of` units are faulty, and at most `per_template[t]` of them are in template t (`Stratified.count_bound`,
  `stratum_bounds`).
- β_δ is `harm_bound`: the profile's `harm_bound`, each template's units weighted by its committed bits (`_template_bits`
  in `verity/core/protocols/sampled_proofs/one_stage/partition.py`). λ_δ is `location_bits` (`consumers.location_bits`):
  log2 of the at most Σ_{j ≤ bound} C(n, j) faulty sets. `exfiltration_bound` is β_δ + λ_δ.
- The run doesn't print ε_ks or ε_bind (the profile's `epsilon_log2` is null), or n·log2 D, which is the certifier's term.

**The `--zk` chain (2a)**, read at main e743d6d146, with what landed since, up to main 595cce6dda, marked in its row.
"Shared" marks a link the plain chain uses as it is. proofs has the last word on every Lean name and status in both
tables. The Class column is the status lean's dependency walk found (run r20261025-032006-79c2,
note:lean/20261025T0420Z-report-e2e-chains): proved; under a named assumption; open (stated, not proved); or broken (no
theorem connects it). proofs classes the rows lean's table doesn't name (unlocked lemmas, the leak links) in the same four
classes. Where proofs' reading of `main` differs from lean's table, or a later landing changed it, the row says so.

| Link | Lean name | Class | Status, owner, date | File |
|---|---|---|---|---|
| A session's verdict (one theorem for both, `zk : Bool`) | `FlockSoundness.Discharge.FailClosed.flock_verify_sound`, `--zk` part: `SoundZ` and `SoundZH` at one table and the stratified draw, `SoundZJ` and `SoundZHJ` at J tables and the subset draw, `SoundZHJR` with `--registered` | under a named assumption (A3) | proved. Premises inside the statement: A3, the record's custody (`RecordsLiveZK(J)`, `DrawOkZK(J)`) and `TableCRZ`; the link term is ⊤ unless `LinkCRZC`. The custody is named and listed on main as `FlockSoundness.Assumptions.Zk.RecordCustodyZKLaw`, with its `ZKH`, `ZKJ` and `ZKHJ` forms (#4067). `TableCRZ` becomes `Assumptions.TableCRZAt` on #4107 (open) | `Security/Proofs/Flock/Soundness/Discharge/FailClosed/One.lean` |
| A session, end to end at J tables | `Flock.SecurityProofs.EndToEnd` | under a named assumption (A3, `RecordCustodyZKJ`) | proved, with A3, `Zk.RecordCustodyZKJ`, `TableCRZ` and `LinkCRZC` as hypotheses | `Security/Proofs/Flock/EndToEnd.lean` |
| Session verdict ⇒ Flock KS (shared) | none: `flock_verify_sound` and `EndToEnd` are in the game form (`FlockSoundness.Strategy`, `auditLive`, `prob`); Flock KS, sampling and the property are in the claims form (`Claims.Flock.KS`, `Claims.Sampling.KS`) | broken | no theorem connects the two forms, so the session verdicts above don't feed the claims chain below, which starts from Flock KS at the record. No owner named yet: E2EMAP's refactor plan, 22:00Z 25 Oct | — |
| Flock KS at sampling's record | `FlockDeploy.KS.zk_KS_untrim`, at the untrimmed `deployZk` | proved | proved, no premise; the trimmed form follows by the bridge (`zk_KS_trim_of_KS`) | `Security/Proofs/Flock/Claims/KS/ZkExtend.lean` |
| Flock's claims | `FlockDeploy.Assembled.flock_zk_assembled`, at `deployZkTrim` | proved | proved, with KS, completeness, the ZK models' links and `Claims.Hm96Hiding` as hypotheses | `Security/Proofs/Flock/Claims/Assembled.lean` |
| Flock KS ⇒ sampling KS (shared) | `Claims.Sampling.SessionKS_of`, `Claims.Sampling.KS_of` | proved | proved, at any record | `Security/Proofs/Flock/Claims/Sampling/Session.lean`, `Security/Proofs/Flock/Claims/Sampling/KS.lean` |
| Sampling KS | `Claims.Sampling.sampling_assembled`, `SamplingDeploy.sampling_deployed_sha512` | proved (a lemma, not in the lock) | proved at `deployZk` (the deployed record `SamplingDeploy.deploy` fixes it), with the run's facts as hypotheses: `DrawFixed`, `SessionRuns`, `SessionsIn`, `DrawSlots`, `SessionOpens`, `InteriorFixed`, `ProgramScope`, `ChecksScope`, `RevealsLe`. Since #4037, `sampling_assembled` is the `deployZk` instance of `sampling_assembled_of`, which plain shares, and #4075 gives one deployed record over any Flock record (`deployOf`, `sampling_deployed_of`) | `Security/Proofs/Flock/Claims/Sampling/Assembled.lean`, `Security/Proofs/Flock/Claims/Sampling/DeployedSha512.lean` |
| Accepted audit ⇒ `Accepted` (shared) | `NetTiming.Guarantees.SampledAuditAccepted` | open at e743d6d146; proved since #4054; listed since #4090 | proved on main (#4054, proofs, lean's item 10) as `NetTiming.SecurityProofs.SampledAuditAccepted`, with no new assumption: `Claims.Sampling.KS` is its one hypothesis. The lock lists it as that guarantee (owner proofs) since #4090, which landed #4083's lock entry; `SampledAuditAcceptedStated` is retired. Since #4090, option A (EVALCOMP's ask, decided provisionally): `RAcc` pins the advice's inputs to the trace's committed values (`traceVal`), with `free` exactly at the faulty proof blocks' committed gates. At plain, #4079's `SoundExecHR` (one table) is in the game form, so it gives the registered-read binding only through the broken shared row | `Security/Definitions/NetworkCertifier/Guarantees/SampledAudit.lean`, `Security/Definitions/NetworkCertifier/SampledAudit.lean`, `Security/Proofs/NetworkCertifier/SampledAuditAccepted.lean` |
| Leak count (shared) | `NetTiming.SecurityProofs.EncardAccSeqsSampledLe`, by `NetTiming.encard_accSeqsSampled_le` | proved | proved. No named assumption: its premises (\|As\| ≤ 2^c, the layout fits) are inside the statement, and it doesn't read the property | `Security/Proofs/NetworkCertifier.lean`, `Security/Proofs/NetworkCertifier/SampledCharge.lean` |
| Accepted audit ⇒ leak bound (shared) | `NetTiming.sampledLeak_reg` (#4090, over #4055's `NetTiming.sampledLeak`); the guarantee `NetTiming.SecurityProofs.SampledLeakAccepted : NetTiming.Guarantees.SampledLeakAccepted` (#4044, restated by #4090) | broken at e743d6d146; under a named assumption (`SampledAuditAccepted`) with #4055 and #4044; proved since #4090, which landed #4083's lock entry | proved and listed in the lock on main (#4055, #4044, #4090, with #4073's `check.py` fix), over any deployment, on `propext`, `Classical.choice` and `Quot.sound`. Its one hypothesis, `SampledAuditAccepted`, is a listed guarantee since #4090; the lock keeps `NetTiming.Guarantees.SampledAuditAccepted` under `assumptions` too, as the premise this statement takes. Since #4090 it counts at the registered inputs pinned (`AsReg`): at whatever values each statement's `reg` holds, at most 2^#(`freeInputs` \ `reg`) · #faultySets(δ) · 2^β · Dj^k sequences, so c = #(`freeInputs` \ `reg`). With `reg` empty it is the old bound. That the trace's values at `reg` are the registration's is P1b (proofs, no PR yet) | `Security/Definitions/NetworkCertifier/Guarantees/SampledCharge.lean`, `Security/Proofs/NetworkCertifier/SampledCharge.lean`, `Security/Proofs/NetworkCertifier/SampledLeak.lean` |

**The plain chain (2b).** The shared links above hold as they are. Plain's own links:

| Link | Lean name | Class | Status, owner, date | File |
|---|---|---|---|---|
| A session's verdict | `flock_verify_sound`, plain part: `SoundExec` (public outputs), `SoundExecH` (hidden outputs); at 2^−205 a table, `flock_verify_sound_205` | under a named assumption (A3) | proved, at one table and the stratified draw. Premises inside the statement: A3, the record's custody (`RecordsLive`, `DrawOk`) and `TableCRL`; the link term is ⊤ unless `LinkCRL`. The custody is named and listed on main as `FlockSoundness.Assumptions.RecordCustodyExec` and `RecordCustodyExecH` (#4048). `TableCRL` and `LinkCRL` become `Assumptions.TableCRAt` and `LinkCRAt` on #4107 (open) | `Security/Proofs/Flock/Soundness/Discharge/FailClosed/One.lean`, `Security/Proofs/Flock/Soundness/Discharge/FailClosed/Numeric.lean` |
| Flock KS at the trimmed record | `FlockDeploy.KS.plain_KS`, at `deployPlainTrim` | proved | proved, no hypothesis. `plain_KS_of_liveAcc` (from `PlainLiveAcc`) and `plain_KS_of_w` (from `PlainLiveW`) reach the same record. lean's table read only the `PlainLiveAcc` route; `plain_KS` needs no hypothesis | `Security/Proofs/Flock/Claims/KS/PlainLeafKS.lean` |
| Flock KS at the untrimmed record, as `--zk`'s sampling reads | `FlockDeploy.KS.plain_KS_of_acc`, `plain_KS_sha512`, at the untrimmed `deployPlain` | open at e743d6d146; proved (a lemma, not in the lock) since #4081 | at e743d6d146, proved only from the full-width table bound (`TableBoundθ` at `orZero (wide false)`), which is open. The trimmed KS doesn't carry over: the bridge runs only from untrimmed to trimmed (`plain_KS_trim_of_KS`; `hole_agree`). Since #4081, DECOMPB's `FlockDeploy.KS.plain_KS_untrim` proves it with no premise: the full-width bound at the plain code through the extension shared with `--zk` (`plain_extend`) | `Security/Proofs/Flock/Claims/KS/PlainUntrim.lean`, `Security/Proofs/Flock/Claims/KS/TableBound.lean`, `Security/Proofs/Flock/Claims/KS/Width.lean`, `Security/Proofs/Flock/Claims/KS/TrimBridge.lean` |
| Flock's claims | `FlockDeploy.Assembled.flock_plain_assembled`, at `deployPlainTrim` | proved | proved, with KS and completeness as hypotheses. `flock_plain_assembled_trim` takes `PlainLiveAcc` in place of KS: an open hypothesis of a superseded route, since `plain_KS` proves the same KS with none. No locked statement reads it, so listing it would list nothing (REROWS). REROWS recommends retiring `plain_KS_of_liveAcc` and taking `plain_KS` in `flock_plain_assembled_trim`: DECOMPB's call, with no date posted | `Security/Proofs/Flock/Claims/Assembled.lean`, `Security/Proofs/Flock/Claims/KS/PlainAcc.lean` |
| Sampling KS | `Claims.Sampling.sampling_assembled_plain` and `sampling_assembled_plain_sha512`, at `deployPlainTrim` (#4037) | broken at e743d6d146 (lean: every sampling theorem was pinned to `deployZk`); proved (a lemma, not in the lock) since #4037 | proved on main (#4037, FINTAU, lean's item 1): one theorem, `sampling_assembled_of`, at any Flock record with its decoder's facts, with `sampling_assembled` (`deployZk`) and `sampling_assembled_plain` its instances. The plain one takes `sampling_assembled`'s run facts, no ZK hypothesis, and, at SHA-512, Flock's KS from `plain_KS`, with no hypothesis. #4075's deployed plain record is at the trimmed Flock record too (`deployPlain_flock`). At the trimmed record its sampling KS covers fewer decoders. An instance at the untrimmed record, over the row two above, isn't on main | `Security/Proofs/Flock/Claims/Sampling/Assembled.lean`, `Security/Proofs/Flock/Claims/Sampling/AssembledPlain.lean`, `Security/Proofs/Flock/Claims/Sampling/Deployed.lean` |
| Registered reads | `FlockSoundness.Discharge.FailClosed.FlockVerifyAcceptsR` | broken in plain at e743d6d146; under a named assumption (A3) at one table since #4079 | checked by the verifier (`buildStmt` runs `checkRegistered` in both modes). Since PR C (PLAINC, #4079), `flock_verify_sound` states it at plain with hidden outputs: `SoundExecHR`, at one table, with every drawn unit's registered reads checked. It takes A3, the custody (`RecordsLive`, `DrawOkH`), `LinkCRL` and `TableCRL`: the session verdict's premises. It is in the game form, so it gives option A's registered-read binding only through the broken shared row. Option A's statement is on main (#4090). `sharedRowsOk` accepts a plain shared-row public file only at hidden outputs with `--registered` and hm96-sha512 leaves. The lift to J tables follows PRs A and B | `Security/Proofs/Flock/Soundness/Discharge/FailClosed/One.lean`, `verity/core/service/flock/FlockVerify.lean`, `verity/core/service/flock/Flock/ProvedScope.lean` |

So on main, every link of 2a's claims chain is proved and listed in the lock: the property (#4054, listed since #4090)
and the leak step (#4055 with #4044, at the registered inputs pinned since #4090). 2b's chain is proved at the trimmed
record too, with #4037's plain sampling KS. Sampling at the untrimmed
record, whose KS admits more decoders, has Flock's KS there (#4081) but no instance of `sampling_assembled_of` at it
yet. In both chains, the session verdicts the verifier of record runs connect to that chain through no theorem yet (the
broken shared row).

**The named assumptions.** The session links read the first three, in both chains. Sampling's KS and the property read
none: their bounds carry the collision events.
- SHA-512 collision resistance: `FlockSoundness.Assumptions.SHA512CRStrict` and `SHA512CRExpected`
  (`Security/Proofs/Flock/Soundness/Assumptions.lean`), and `Claims.StrictBirthday` and `Claims.ExpectedCR`
  (`Security/Definitions/Claims.lean`).
- A3, the coins: `FlockSoundness.Assumptions.UniformRandomBytes` (the same file).
- Record custody, under `--zk`: `FlockSoundness.Assumptions.Zk.RecordCustodyZKJ`
  (`Security/Proofs/Flock/Soundness/Assumptions/Zk.lean`), which is `RecordsLiveZKJ` with the draw file's custody. Plain's
  are the premises `RecordsLive` and `DrawOk` (`Security/Proofs/Flock/Soundness/Discharge/DecodeZL/Statements.lean`),
  named and listed on main as `FlockSoundness.Assumptions.RecordCustodyExec` and `RecordCustodyExecH` (#4048,
  `Security/Proofs/Flock/Soundness/Assumptions/Custody.lean`). The inline `--zk` custody is
  `Assumptions.Zk.RecordCustodyZKLaw` and its forms (#4067, `Security/Proofs/Flock/Soundness/Assumptions/CustodyZk.lean`).
  `PlainLiveAcc` isn't listed, since no locked statement reads it.
- The rest of what locked statements take from `trees.except`: [e2e_chains/README.md](e2e_chains/README.md).

No link above reads the rest. `TrustedHost` is why `Carries` holds of the real run, and the wire and advice
assumptions enter through the wire bounds, which have no sampled form yet.
- `Verity.Assumptions.TrustedHost` (`Security/Definitions/Core/Assumptions.lean`).
- The certifier's wire (`Security/Definitions/NetworkCertifier/Assumptions.lean`): W2 `NetTiming.Regenerates`, W4
  `NetTiming.Packed` and `NetTiming.Allocated`, and J `NetTiming.JitterBounded`. W3 (`Padded`), W5 (`UntimedStatus`) and W6
  (`FailClosed`) are stated there, but no listed guarantee reads them, so the lock doesn't list them.
- The advice (`Security/Definitions/NetworkCertifier/Assumptions/AdviceString.lean`): `NetTiming.BitsCount`,
  `NetTiming.CompilerDeterministic` and `NetTiming.CompilerPrecommitted`.
- For privacy, not soundness, under `--zk`: `Claims.Hm96Hiding` (`Security/Definitions/Claims/Hm96.lean`).

**CHAINMAP's status per protocol, and the premises in `Proofs/`** (lean's list, REROWS's sort, the form conditions,
and what is moved and listed) are in [e2e_chains/README.md](e2e_chains/README.md).

**Tonight's interim.**
- **2b's plain sessions are inside what `flock_verify_sound` covers today.** Each session has one table: `statementRules`
  refuses more without `--zk`. Each draws by the stratified law: `lawWhy` refuses a subset draw without `--zk`, and a
  bernoulli or work draw in either mode. a5 writes each session's draw as `subset` (`examples/one-stage/a5.py`), so a
  plain session drawn as a5 draws is refused. The outputs are public, and there are no registered reads. One-table plain
  already covers hidden outputs (`SoundExecH`).
- **In plain mode, the drawn units' committed rows are not opened outside the session** (proofs). `keyRowsH_computes`
  (`Security/Proofs/Flock/Soundness/Discharge/HiddenExec/KeyH.lean`) already binds each drawn row's b‖c to its input
  words, up to a SHA-512 collision, and a check outside `flock_verify_sound` would be a new form ahead of its proof. Since
  C (#4079), a plain session's registered reads are inside the plain theorem at one table (`SoundExecHR`).
- **A plain served window waits for proofs' PRs.**
  - A, J tables (1 to 64) and the subset draw: DECOMPCOMP, Mon 26 Oct.
  - C, checked registered reads and shared rows: PLAINC, on main at one table as `SoundExecHR` (#4079), under A3 and
    the session verdict's premises, lifted after B (Tue 27 Oct at the earliest).
  - B, hidden outputs at J tables, beyond `SoundExecH` at one table: PLAINB, Tue 27 Oct.

  A plain one_stage audit follows, Wed 28 Oct at the earliest. Tonight the served window runs under `--zk` (2a), which
  passed end to end on live coins at a1c5, and through #4061's interface at 334e, the demo of record (above).

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
`LinkWindows.stream` is the certifier's records' stream, not the proof blocks'. To draw them on the one proof service, as
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
    A Program's body is read from the Build dir whose `result.json` names I2's digest, refused where none does, and held to
    that digest (`build_of`): recomputed from its `descriptor.json.gz`, with every recorded digest and every
    `instances.json.gz` row held to it, so a Build whose body was swapped under its old `result.json` is refused. No Program
    digest covers the rows' names, the parameters' names and types, `supplied`, the correspondence or `prefix_cache.json`, so
    the verifier (`audit`, and the compile command `circuit`) reads them only from its own Build of each Program: the trace
    `steps.trace_commands` names, run with the pinned vLLM (`--vllm-python`, `own_builds`) into `--builds` (default
    `OUT/builds`, never under the row dir), each held to I2's digest. The prover's Builds are its row dir's. On
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
- **The audit from the records alone** (`records_audit.py audit`, `run.sh` with `PP_RECORDS=1` on a served window) reads only
  what an integration hands Verity (`verity.ml.windows.interface`): the window's `circuit.json`, which its registration binds
  by SHA-512; the registered compiler and the advice's root and length; the registration and the committed values; and the
  advice's opening, the named gap. The verifier's window is the circuit record's, each template's Definition from Verity's
  catalog or the verifier's own input set. The compile check runs the registered compiler as the command the verifier's
  table names for it (`prove_path.py circuit` for a served window, which takes no row dir and compiles over its own trace of
  each Program), on advice that opens the registered root, and holds its output to the circuit record (less the run root and
  the coins' context) and its charged length to the registered one, so a row dir whose names make another circuit under
  unchanged digests is refused (`circuit`). It
  imports no `verity_vllm` code. The exception is a template whose Definition is still vLLM's (`SiluMul_v2`, `RoPEHead_v1`):
  the input set loads it from `verity.ml.definitions.registry` until #3927 moves those Definitions into `verity`.
- **A single-GPU window** is a stand-in of the Commit's `rows-sha512` run root (`stand_in1`, held by `source1`), as
  `deferred_rows.from_bundle` writes one. The sealed bundle records the Commit's absolute node paths, and none is read: the
  prover's Programs come from `--row-dir` and the verifier's from its own trace, and the Build manifest falls back to
  `<row dir>/manifest.json`, checked against the record's sha256.
- **The advice** is I1's canonical encoding. A declared run's comes from its workload (`advice_windows.advice_of_workload`), a
  served run's from ADVICE-SRC's step trace.
- **The opening** is the named gap above. `run.sh` keeps neither the advice nor its opening in `$OUT`.
