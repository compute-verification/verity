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

One property covers both configurations. 2a, the served window under `--zk`, is the demo (top, 25 Oct). 2b, plain on the
CPU TP stand-in, has soundness as its only guarantee (Daniel, 24 Oct, #4016). Each layer is proved once over the layer
below and instantiated per configuration. **No theorem on main yet takes an accepted audit, under `--zk` or plain, to
`Accepted` or to the leak bound.** What's proved stops at each session's verdict, plus, under `--zk`, sampling's KS given
the run's hypotheses. On open PRs, #4054 proves the property and #4055 with #4044 the leak step. So 2b demonstrates
per-session plain soundness, not an audit guarantee.

**The property.** The sampled step units are the sampled class of `NetTiming.Guarantees.SampledAuditAccepted` (#3608,
PRF-144). It is stated once, at any sampling record `D`:
`∀ H D L K S lay k, Claims.Sampling.KS H D → KSAccepted H D L K S lay k`, with `D.flock = deployZk rel` under `--zk` and
Flock's plain record otherwise. proofs proves it on #4054 as `NetTiming.SecurityProofs.SampledAuditAccepted`, with no
new assumption; its Lean check passed at 12387ce99f (r20261025-034643-a5fa), and its head since moves one docstring line.
The lock change that lists it is proofs' follow-up.

On the statements in scope (`ScopeAcc`), at every δ ≥ 0, every decoder within Flock's radius and every prover,
Pr[acc ∧ ¬RAcc δ] ≤ δ + ε_ks + ε_bind, with ε_ks = κ(δ) − δ + √(γ·Pr[B]) and ε_bind = Pr[B_ext]. Pr[B] and Pr[B_ext] are
the chances of finding a SHA-512 collision in the run, not an assumption. `RAcc δ` holds when:
- the faulty set is in M_δ = {B : P_a(B) > δ} (`RelS`);
- no faulty unit is in a class drawn `all`: the transfer-gate units and N2's head checks (`allSlots`);
- some advice (a, free), with `free` holding values only at the faulty units' committed gates (`FaultyGate`), makes NetTiming's
  `Accepted` hold over every observation sequence the trace's transfer gates carry (`Carries`), with every status complete
  (`Covered`). With no unit faulty, `free` is empty and `Accepted` is exact.

**The leak bound that follows.** The theorem is `NetTiming.sampledLeak_free` (proofs' #4055). The lock-listed guarantee
is `NetTiming.Guarantees.SampledLeakAccepted`, proved as
`NetTiming.SecurityProofs.SampledLeakAccepted := fun hSA => NetTiming.sampledLeak_free hSA` (#4044, owner
network-accounting; `SampledAuditAccepted` and `Claims.Sampling.KS` owned by proofs). Status: proved, and it elaborates on
`propext`, `Classical.choice` and `Quot.sound`. Its lock check waits on a `check.py` fix (`claims.stale` for a listed
`Claims.*` premise) in its own PR, #4073 (LEAK-LEAN).

It composes the property with the count `EncardAccSeqsSampledLe`, over any deployment. Given `SampledAuditAccepted` and
`Claims.Sampling.KS` at the deployment, except with probability δ + ε_ks + ε_bind, an accepted audit's carried
sequences, cut at the horizon, number at most 2^c · #faultySets(δ) · 2^β · D^n. So the audit leaks at most
c + λ_δ + β_δ + n·log2 D bits:
- c = #`freeInputs`, the circuit's input gates that no step before the horizon lists as ingress. `RAcc` lets the advice
  take any value there, so the bound charges every one of them;
- λ_δ, which units are faulty;
- β_δ, the faulty units' committed bits;
- n·log2 D, the timing (J's D classes over n steps).

At δ ≥ miss(1) no unit is faulty, and the bound is c + n·log2 D. For 2a's circuit (run r20261025-041404-a1c5: 8
`SiluMul_v2{I=8192}` units, whose 131,072 16-bit inputs no step lists as ingress), #`freeInputs` = 2,097,152 bits
(2^21), against the advice's declared 1,687 bits (2b's advice declares 169, r20261025-032237-db01). So the bound as
stated is vacuous at 2a's size. A smaller c, the advice's declared bits, needs `SampledAuditAccepted` to pin the
registered input gates to the registration. That is option A, EVALCOMP's ask on #ask-daniel (card ddf12c5c,
[its page](https://app.notion.com/p/404399515d9e81d99e61f6534b54b291)), decided provisionally as A on 25 Oct. Under A,
c = #(`freeInputs` \ Reg): the registered-read binding pins the registered gates, which `--zk` has and plain gets with
PR C. Until A is proved, the page states the larger c.

A alone doesn't shrink 2a's c (proofs): 2a has no registered inputs, and its 2^21 bits are layer-0 activations the
layout doesn't declare as ingress, so 2a's bound is vacuous until that changes. The change, as top settled it on 25 Oct:
the window's circuit starts at the request's token ids, declared as ingress and tied by the I/O tie, with the embedding
a registered read and the activations wired rather than free, and the circuit checks the advice charge. It is a layout
and compiler change, not a claim: NET-68, owner network-accounting, with its date by 12:00Z 25 Oct; the compiler part is
vllm's VLLM-103 under it. Until then, vllm's #4061 has `records_audit` label the bound vacuous whenever #`freeInputs`
exceeds the declared charge.

**The charge terms the run prints** (`summary.json` of `prove_path.py audit`; with #4047, each rank's
`ranks.<r>.charge` holds `advice_bits`, `delta`, `wrong_units`, `location_bits`, `harm_bound` and `exfiltration_bound`):
- c, as the statement stands, is #`freeInputs`, which the run doesn't print: 2,097,152 bits (2^21) for 2a, counted from
  a1c5's `rank0/program.json` by `freeInputs`'s definition (#4055). Under option A (decided provisionally, card
  ddf12c5c), once the advice is the only free input outside the registration (for 2a, after NET-68), c is
  `advice.bits`, the registration's `window.advice.bits`: the I1 advice's length, which `prove_path.py advice` prints as
  `bits`. Over several windows, `BitsCount` charges their sum.
- δ is `delta`. Per rank, `ranks.<r>.audit.wrong_units` is `{delta, of, bound, rule, per_template}` (the
  `wrong_units` function in `verity/core/protocols/sampled_proofs/one_stage/audit.py`). Except with probability δ, at most
  `bound` of the `of` units are faulty, and at most `per_template[t]` of them are in template t (`Stratified.count_bound`,
  `stratum_bounds`).
- β_δ is `harm_bound`: the profile's `harm_bound`, each template's units weighted by its committed bits (`_template_bits`
  in `verity/core/protocols/sampled_proofs/one_stage/partition.py`). λ_δ is `location_bits` (`consumers.location_bits`):
  log2 of the at most Σ_{j ≤ bound} C(n, j) faulty sets. `exfiltration_bound` is β_δ + λ_δ.
- The run doesn't print ε_ks or ε_bind (the profile's `epsilon_log2` is null), or n·log2 D, which is the certifier's term.

**The `--zk` chain (2a)**, on main at e743d6d146. "Shared" marks a link the plain chain uses as it is. proofs has the
last word on every Lean name and status in both tables. The Class column is the status lean's dependency walk found (run
r20261025-032006-79c2, note:lean/20261025T0420Z-report-e2e-chains): proved; under a named assumption; open (stated, not
proved); or broken (no theorem connects it). proofs classes the rows lean's table doesn't name (unlocked lemmas, the leak
links) in the same four classes. Where proofs' reading of `main` differs from lean's table, the row says so.

| Link | Lean name | Class | Status, owner, date | File |
|---|---|---|---|---|
| A session's verdict (one theorem for both, `zk : Bool`) | `FlockSoundness.Discharge.FailClosed.flock_verify_sound`, `--zk` part: `SoundZ` and `SoundZH` at one table and the stratified draw, `SoundZJ` and `SoundZHJ` at J tables and the subset draw, `SoundZHJR` with `--registered` | under a named assumption (A3) | proved. Premises inside the statement: A3, the record's custody (`RecordsLiveZK(J)`, `DrawOkZK(J)`) and `TableCRZ`; the link term is ⊤ unless `LinkCRZC`. REROWS names the custody `FlockSoundness.Assumptions.Zk.RecordCustodyZKLaw` (with its `ZKH`, `ZKJ` and `ZKHJ` forms) on #4067, and `TableCRZ` as `Assumptions.TableCRZAt` on its premises branch (`cursor/proofs-premises-5b1e`, no PR yet) | `Security/Proofs/Flock/Soundness/Discharge/FailClosed/One.lean` |
| A session, end to end at J tables | `Flock.SecurityProofs.EndToEnd` | under a named assumption (A3, `RecordCustodyZKJ`) | proved, with A3, `Zk.RecordCustodyZKJ`, `TableCRZ` and `LinkCRZC` as hypotheses | `Security/Proofs/Flock/EndToEnd.lean` |
| Session verdict ⇒ Flock KS (shared) | none: `flock_verify_sound` and `EndToEnd` are in the game form (`FlockSoundness.Strategy`, `auditLive`, `prob`); Flock KS, sampling and the property are in the claims form (`Claims.Flock.KS`, `Claims.Sampling.KS`) | broken | no theorem connects the two forms, so the session verdicts above don't feed the claims chain below, which starts from Flock KS at the record. No owner named yet: E2EMAP's refactor plan, 22:00Z 25 Oct | — |
| Flock KS at sampling's record | `FlockDeploy.KS.zk_KS_untrim`, at the untrimmed `deployZk` | proved | proved, no premise; the trimmed form follows by the bridge (`zk_KS_trim_of_KS`) | `Security/Proofs/Flock/Claims/KS/ZkExtend.lean` |
| Flock's claims | `FlockDeploy.Assembled.flock_zk_assembled`, at `deployZkTrim` | proved | proved, with KS, completeness, the ZK models' links and `Claims.Hm96Hiding` as hypotheses | `Security/Proofs/Flock/Claims/Assembled.lean` |
| Flock KS ⇒ sampling KS (shared) | `Claims.Sampling.SessionKS_of`, `Claims.Sampling.KS_of` | proved | proved, at any record | `Security/Proofs/Flock/Claims/Sampling/Session.lean`, `Security/Proofs/Flock/Claims/Sampling/KS.lean` |
| Sampling KS | `Claims.Sampling.sampling_assembled`, `SamplingDeploy.sampling_deployed_sha512` | proved (a lemma, not in the lock) | proved at `deployZk` (the deployed record `SamplingDeploy.deploy` fixes it), with the run's facts as hypotheses: `DrawFixed`, `SessionRuns`, `SessionsIn`, `DrawSlots`, `SessionOpens`, `InteriorFixed`, `ProgramScope`, `ChecksScope`, `RevealsLe`. On #4037, `sampling_assembled` is the `deployZk` instance of `sampling_assembled_of`, which plain shares | `Security/Proofs/Flock/Claims/Sampling/Assembled.lean`, `Security/Proofs/Flock/Claims/Sampling/DeployedSha512.lean` |
| Accepted audit ⇒ `Accepted` (shared) | `NetTiming.Guarantees.SampledAuditAccepted` | open | stated only on main (PRF-144); the lock holds its text through `SampledAuditAcceptedStated`. Proved on #4054 (proofs, lean's item 10) as `NetTiming.SecurityProofs.SampledAuditAccepted`, at the statement as it stands, with no new assumption; its Lean check passed at 12387ce99f (r20261025-034643-a5fa). Listing it in the lock in place of `SampledAuditAcceptedStated` is proofs' follow-up (branch `cursor/proofs-prf144-lock-4e84`). Option A adds a conjunct (EVALCOMP's ask, decided provisionally); plain needs PR C for it, Tue 27 Oct | `Security/Definitions/NetworkCertifier/Guarantees/SampledAudit.lean`; the proof in `SampledAuditAccepted.lean` on #4054 |
| Leak count (shared) | `NetTiming.SecurityProofs.EncardAccSeqsSampledLe`, by `NetTiming.encard_accSeqsSampled_le` | proved | proved. No named assumption: its premises (\|As\| ≤ 2^c, the layout fits) are inside the statement, and it doesn't read the property | `Security/Proofs/NetworkCertifier.lean`, `Security/Proofs/NetworkCertifier/SampledCharge.lean` |
| Accepted audit ⇒ leak bound (shared) | `NetTiming.sampledLeak_free` (#4055); the guarantee `NetTiming.SecurityProofs.SampledLeakAccepted : NetTiming.Guarantees.SampledLeakAccepted` (#4044) | broken on main | no theorem on main. Proved on #4055 with #4044, over any deployment, with `SampledAuditAccepted` its one open hypothesis, on `propext`, `Classical.choice` and `Quot.sound`. Its lock check waits on #4073's `check.py` fix. c is #`freeInputs` until option A, since `RAcc` doesn't bound the run's inputs to `As` | `Security/Definitions/NetworkCertifier/Guarantees/SampledCharge.lean`, `Security/Proofs/NetworkCertifier/SampledCharge.lean`; the proof in `SampledLeak.lean` on #4055 |

**The plain chain (2b).** The shared links above hold as they are. Plain's own links:

| Link | Lean name | Class | Status, owner, date | File |
|---|---|---|---|---|
| A session's verdict | `flock_verify_sound`, plain part: `SoundExec` (public outputs), `SoundExecH` (hidden outputs); at 2^−205 a table, `flock_verify_sound_205` | under a named assumption (A3) | proved, at one table and the stratified draw. Premises inside the statement: A3, the record's custody (`RecordsLive`, `DrawOk`) and `TableCRL`; the link term is ⊤ unless `LinkCRL`. REROWS names the custody `FlockSoundness.Assumptions.RecordCustodyExec` and `RecordCustodyExecH` on #4048 (queued on `next`), and `TableCRL` and `LinkCRL` as `Assumptions.TableCRAt` and `LinkCRAt` on its premises branch | `Security/Proofs/Flock/Soundness/Discharge/FailClosed/One.lean`, `Security/Proofs/Flock/Soundness/Discharge/FailClosed/Numeric.lean` |
| Flock KS at the trimmed record | `FlockDeploy.KS.plain_KS`, at `deployPlainTrim` | proved | proved, no hypothesis. `plain_KS_of_liveAcc` (from `PlainLiveAcc`) and `plain_KS_of_w` (from `PlainLiveW`) reach the same record. lean's table read only the `PlainLiveAcc` route; `plain_KS` needs no hypothesis | `Security/Proofs/Flock/Claims/KS/PlainLeafKS.lean` |
| Flock KS at the untrimmed record, as `--zk`'s sampling reads | `FlockDeploy.KS.plain_KS_of_acc`, `plain_KS_sha512`, at the untrimmed `deployPlain` | open | open: proved only from the full-width table bound (`TableBoundθ` at `orZero (wide false)`), which is open. The trimmed KS doesn't carry over: the bridge runs only from untrimmed to trimmed (`plain_KS_trim_of_KS`; `hole_agree`). Plain KS at the untrimmed record is DECOMPB's `plain_KS_untrim`, with no premise: the full-width bound at the plain code through the extension shared with `--zk` (`plain_extend`). It is in PlainUntrim.lean on branch `cursor/proofs-plainuntrim-0716` at 8a5d75b84c, with no PR yet; its Lean check passed there (r20261025-025425-dbe1). DECOMPB's date is to come | `Security/Proofs/Flock/Claims/KS/TableBound.lean`, `Security/Proofs/Flock/Claims/KS/Width.lean`, `Security/Proofs/Flock/Claims/KS/TrimBridge.lean` |
| Flock's claims | `FlockDeploy.Assembled.flock_plain_assembled`, at `deployPlainTrim` | proved | proved, with KS and completeness as hypotheses. `flock_plain_assembled_trim` takes `PlainLiveAcc` in place of KS: an open hypothesis of a superseded route, since `plain_KS` proves the same KS with none. No locked statement reads it, so listing it would list nothing (REROWS). REROWS recommends retiring `plain_KS_of_liveAcc` and taking `plain_KS` in `flock_plain_assembled_trim`: DECOMPB's call, with no date posted | `Security/Proofs/Flock/Claims/Assembled.lean`, `Security/Proofs/Flock/Claims/KS/PlainAcc.lean` |
| Sampling KS | `Claims.Sampling.sampling_assembled_plain` and `sampling_assembled_plain_sha512`, at `deployPlainTrim` (#4037) | broken on main | no theorem on main (lean: every sampling theorem is pinned to `deployZk`). FINTAU (lean's item 1): the general theorem with its `--zk` and plain instances, Mon 26 Oct, on #4037 (branch `cursor/proofs-plainsamp-944c`): one theorem, `sampling_assembled_of`, at any Flock record with its decoder's facts, with `sampling_assembled` (`deployZk`) and `sampling_assembled_plain` its instances. The plain one takes `sampling_assembled`'s run facts, no ZK hypothesis, and, at SHA-512, Flock's KS from `plain_KS`, with no hypothesis. Its Lean check passed at its head, 0baa7ff497 (r20261025-025630-ea45). At the trimmed record its sampling KS covers fewer decoders. The untrimmed `deployPlain` needs the row two above | `Security/Proofs/Flock/Claims/Sampling/Assembled.lean`, and `AssembledPlain.lean` beside it on #4037 |
| Registered reads | `FlockSoundness.Discharge.FailClosed.FlockVerifyAcceptsR` | broken in plain | checked by the verifier (`buildStmt` runs `checkRegistered` in both modes), but not stated by the plain theorem: `FlockVerifyAcceptsR false` is `Verifies`. PR C (PLAINC, #4079) adds `SoundExecHR`, Mon 26 Oct at one table. `sharedRowsOk` refuses a shared-row public file without `--zk` | `Security/Proofs/Flock/Soundness/Discharge/FailClosed/One.lean`, `verity/core/service/flock/FlockVerify.lean`, `verity/core/service/flock/Flock/ProvedScope.lean` |

So on main, 2a's claims chain is open at the property and the leak step, and open PRs prove both: the property on
#4054, the leak step on #4055 with #4044. 2b's also needs plain sampling KS, which #4037 proves at the trimmed record.
Sampling at the untrimmed record, whose KS admits more decoders, also needs Flock's KS there (DECOMPB's
`plain_KS_untrim`, on a branch) and an instance of `sampling_assembled_of` at it. In both chains, the session verdicts
the verifier of record runs connect to that chain through no theorem yet (the broken shared row).

**The named assumptions.** The session links read the first three, in both chains. Sampling's KS and the property read
none: their bounds carry the collision events.
- SHA-512 collision resistance: `FlockSoundness.Assumptions.SHA512CRStrict` and `SHA512CRExpected`
  (`Security/Proofs/Flock/Soundness/Assumptions.lean`), and `Claims.StrictBirthday` and `Claims.ExpectedCR`
  (`Security/Definitions/Claims.lean`).
- A3, the coins: `FlockSoundness.Assumptions.UniformRandomBytes` (the same file).
- Record custody, under `--zk`: `FlockSoundness.Assumptions.Zk.RecordCustodyZKJ`
  (`Security/Proofs/Flock/Soundness/Assumptions/Zk.lean`), which is `RecordsLiveZKJ` with the draw file's custody. Plain's
  are the premises `RecordsLive` and `DrawOk` (`Security/Proofs/Flock/Soundness/Discharge/DecodeZL/Statements.lean`),
  which the lock doesn't name on main. REROWS names and lists them as `FlockSoundness.Assumptions.RecordCustodyExec` and
  `RecordCustodyExecH` (#4048), and the inline `--zk` custody as `Assumptions.Zk.RecordCustodyZKLaw` and its forms
  (#4067). `PlainLiveAcc` isn't listed, since no locked statement reads it.
- The rest of what locked statements take from `trees.except`: CHAINMAP's status below.

No link above reads the rest. `TrustedNebiusHost` is why `Carries` holds of the real run, and the wire and advice
assumptions enter through the wire bounds, which have no sampled form yet.
- `Verity.Assumptions.TrustedNebiusHost` (`Security/Models/Core/Assumptions.lean`).
- The certifier's wire (`Security/Models/NetworkCertifier/Assumptions.lean`): W2 `NetTiming.Regenerates`, W4
  `NetTiming.Packed` and `NetTiming.Allocated`, and J `NetTiming.JitterBounded`. W3 (`Padded`), W5 (`UntimedStatus`) and W6
  (`FailClosed`) are stated there, but no listed guarantee reads them, so the lock doesn't list them.
- The advice (`Security/Models/NetworkCertifier/Assumptions/AdviceString.lean`): `NetTiming.BitsCount`,
  `NetTiming.CompilerDeterministic` and `NetTiming.CompilerPrecommitted`.
- For privacy, not soundness, under `--zk`: `Claims.Hm96Hiding` (`Security/Definitions/Claims/Hm96.lean`).

**CHAINMAP's status per protocol.** lean's CHAINMAP maps each protocol's e2e chain layer by layer: the session's KS,
then sampling, then the audit, then the accounting bound. It reads the lock's 899 guarantees and a dependency walk
(note:lean/20261025T0420Z-report-e2e-chains, run r20261025-032006-79c2, at #3627's head 85bd0f9e26). The status column
is CHAINMAP's as of that note. The last column gives the refactor item that changes it, with the owner and date proofs
posted on 25 Oct at 06:17Z, and what open PRs have done since.

| Protocol | Config | Status (CHAINMAP) | Item, owner, date; open PRs |
|---|---|---|---|
| Sampled proofs / C-Flock | `--zk` | Proved. `sampling_assembled` is not a lock guarantee | item 1, FINTAU: the general theorem with its `--zk` and plain instances, Mon 26 Oct. #4037 makes `sampling_assembled` the `deployZk` instance of `sampling_assembled_of`, and #4075 gives one deployed record over any Flock record (`sampling_deployed_of`) |
| | plain | Proved in the game form. No sampling theorem at plain in claims form | item 1, FINTAU, Mon 26 Oct: #4037's `sampling_assembled_plain`, at the trimmed record. The untrimmed record needs DECOMPB's `plain_KS_untrim` (branch, date to come). The game-to-claims link is the broken shared row: E2EMAP's plan, 22:00Z 25 Oct |
| | recursive | Effectively none: proved, but near-vacuous. No KS-to-sampling for recursion | item 12, REC100 and RESLOW on the RecursiveKS track, Fri 30 Oct, after Daniel's vcalls ruling (default A on 30 Oct). Item 1's recursive instance follows it |
| PoUW | `--zk` | Chain broken at KS → A: `SessionSoundAt` needs `AcceptedLive`, which no served path meets yet | item 9, proofs (ZKFIN, after FinishIs): `SessionSoundAt` to `ProofSoundAt` (δs + δ), with the TS headlines its instances, Tue 27 Oct. `AcceptedLive` on a served path needs FinishIs (Mon 26 Oct) and a served window whose lanes all verify (PRF-131's first pass verified 9 of 32): Wed 28 Oct |
| | plain, recursive | none | none posted |
| PoUS | all | none (ideal model only) | item 11, E2EMAP, over item 1 as a hypothesis: Thu 29 Oct |
| Bandwidth (over PoUS's store) | all | none with a session layer | item 11, E2EMAP: Thu 29 Oct |
| Remote bandwidth | all | none with a session layer | item 11, E2EMAP: Thu 29 Oct |
| Network certifier | all | none: `SampledAuditAccepted` stated, unproved | item 10, FINTAU, due by 06:00Z 27 Oct: proved on #4054, listed by proofs' lock follow-up. The leak step is proved on #4055 with #4044 |
| NCI | all | none with a session layer; it rests on the assumption `PouwSound`, which no PoUW guarantee instantiates | item 11's NCI part: `PouwSound` replaced by item 9's e2e, Fri 30 Oct, after item 9 |
| Accounting (top layer) | — | Already modular: the pattern every lower layer should copy | — |

Items 1 and 9–12 are the statement-changing steps of the note's refactor sketch: KS to sampling once (1), PoUW's headline
from `SessionSoundAt` (9), the certifier's property (10), PoUS, bandwidth, remote bandwidth and NCI over sampling (11),
and recursion's knowledge form (12). The rest (2–6, 8 and 13) are E2EMAP's; item 2 is #4043. E2EMAP turns the map into
a refactor plan by 22:00Z 25 Oct, with the first refactor PR marked Mon 26 Oct.

**Premises in `Proofs/`.** `trees.except` (`Security/lean-audit.json`) exempts 33 module prefixes from the tree rule
until C-Flock's spec is extracted (RECKSLOCK, Wed 4 Nov). 31 are C-Flock's; the other two, `Proofs.Accounting.TS.Session`
and `Proofs.Pouw.PearlC.FlockTile`, are on PoUW's chain. None is PoUS's, bandwidth's, remote bandwidth's, the
certifier's, NCI's or accounting's, so this is a C-Flock question, plus one PoUW definition.
- **lean's list** (run r20261025-031300-0fcc, in CHAINMAP's note): 40 `Prop`-valued definitions under `trees.except`
  that a locked statement takes without a lock entry. 39 are C-Flock's, and one, `SoundAtBridge.RowsSession.CanonRead`,
  is PoUW's. CHAINMAP's list was due by 12:00Z 25 Oct and came out at 04:20Z.
- **REROWS's first sort** (note:proofs/20261025T0353Z-report-premise-audit) gives each of the 40 a verdict: 10 premises
  and 30 conclusions.
  - The premises are C-Flock's:
    - the plain custody (`DecodeZL.RecordsLive`, `DecodeZL.DrawOk`, `HiddenDec.DrawOkH`);
    - the `--zk` custody (`ZkLink.RecordsLiveZK`, `ZkLink.DrawOkZK`, `ZkHidden.DrawOkZKH`, `Composed.RecordsLiveZKJ`);
    - plain's finder hash bounds (`Exec.LinkCRL`, `Exec.SessCRL`);
    - recursion's `RecDeploy.PlainLeavesAdmitted`.
  - The conclusions are acceptance events and the verifier's checks, events a bound measures, antecedents inside a
    conclusion, and fields of setups built from the verifier's own. PoUW's `CanonRead` is one of them, a selector.
  - It also found premises the scan doesn't print: recursion's draw custody (`ZkOuterD.hDraw`, `ZkOuterR.hDraw`), and
    `--zk`'s `TableCRZ` and `LinkCRZC`.
- **Form conditions**, top's third class: `StmtWF` and `RegionsWF` (26 guarantees each), `PadOK` (29), and perhaps
  `isRow` and `isRead`. The first sort counts them as conclusions: the first three are fields of setups that
  `drawSetup*_of` builds from flock-verify's setup, and the last two are selectors. The classification is due by
  15:00Z 25 Oct (proofs). REROWS's four-class sort (conclusion, premise, form condition, well-formedness), which says
  whether the compiled verifier checks each form condition, is due by 22:00Z. A form condition the verifier doesn't
  check comes to this page as a gap.
- **Moved and listed:** none of it on main yet. Every premise is due moved and listed by 22:00Z 25 Oct (proofs).
  - #4048 names the plain custody (`RecordCustodyExec`, `…ExecH`), and #4067 the `--zk` custody
    (`Zk.RecordCustodyZKLaw` and its forms). Both are queued on `next`.
  - The premises branch, `cursor/proofs-premises-5b1e` (no PR yet), names the rest: `TableCRAt`, `LinkCRAt`,
    `TableCRZAt`, `Zk.RecordsCustodyZKJ`, `Zk.DrawCustodyZKHJ`, and `PlainLeavesAdmitted` moved. Its record run
    started 05:45Z.

**Tonight's interim.**
- **2b's plain sessions are inside what `flock_verify_sound` covers today.** Each session has one table: `statementRules`
  refuses more without `--zk`. Each draws by the stratified law: `lawWhy` refuses a subset draw without `--zk`, and a
  bernoulli or work draw in either mode. a5 writes each session's draw as `subset` (`benchmarks/one_stage/a5.py`), so a
  plain session drawn as a5 draws is refused. The outputs are public, and there are no registered reads. One-table plain
  already covers hidden outputs (`SoundExecH`).
- **In plain mode, the drawn units' committed rows are not opened outside the session** (proofs). `keyRowsH_computes`
  (`Security/Proofs/Flock/Soundness/Discharge/HiddenExec/KeyH.lean`) already binds each drawn row's b‖c to its input
  words, up to a SHA-512 collision, and a check outside `flock_verify_sound` would be a new form ahead of its proof. A plain
  session's registered reads are checked by the verifier but are outside the plain theorem (the table above), so a plain
  served window waits for C.
- **A plain served window waits for proofs' PRs.**
  - A, J tables (1 to 64) and the subset draw: DECOMPCOMP, Mon 26 Oct.
  - C, checked registered reads and shared rows: PLAINC, Mon 26 Oct at one table, lifted after B (Tue 27 Oct at the
    earliest). Its widening of `sharedRowsOk` to plain shared-row files lands only with its proof, `SoundExecHR`.
  - B, hidden outputs at J tables, beyond `SoundExecH` at one table: PLAINB, Tue 27 Oct.

  A plain one_stage audit follows, Wed 28 Oct at the earliest. Tonight the served window runs under `--zk` (2a), which
  passed end to end on live coins (r20261025-041404-a1c5).

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
