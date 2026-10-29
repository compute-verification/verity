# Prove path

Network accounting under arbitrary vLLM usage proves a served run's step units after the fact: a precommitted compiler turns
the run's advice into each step's Programs, the committing vLLM has recorded each step's values, and the proof system draws
units on live coins, proves them with C-Flock `--zk` behind the Lean firewall, and the Lean verifier accepts or refuses them.
`prove_path.py` is that path on a committed run, a TP Commit (`--tp-commit`), a single-GPU one (`--bundle`, TP1) or a closed
Commit window of a `verity-vllm serve-openai` run (`--serve-run`, TP1): the prover commits the advice under hiding rows (a
served window's the server committed) and each rank's window over the compiler's Programs for it, and the verifier audits each
rank through the one-stage audit's per-rank `service.Stream` on its own coins or a challenger's (`--challenger local`,
`benchmarks/one_stage/a5.py`). `run.sh` builds the prover and the Lean verifier and runs it on the CPU TP stand-in, on a
deferred TP Commit (`PP_TP`), on a single-GPU Commit's row dir on the node (`PP_LLAMA`) or on a served window
(`PP_SERVE_RUN`), with one forged run that must be refused. Row dirs and serve runs are the node's own, never fetched from the
store, since their bundles and requests hold request values. The docstrings say how.

## What changes when Programs come from advice

A declared run's audit (`a5.py --tp-commit`) derives each rank's Programs from the workload the record served. Under arbitrary
usage no workload is declared: serving decides the batches, arrivals and lengths, and the advice (I1,
`verity.core.protocols.certifier.advice`) records those decisions. The advice is private: it is committed under hiding hm96
rows and never published.

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
3. **Named gap: the advice is opened to the verifier until the compile unit exists.** The verifier is handed the
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
   - a request the compiler can't run: a chunked prefill, a prefix-cache hit, a preemption or a gap (the compiler's v0
     rules, `advice_windows.refusals`).

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
   of them, a draw trading one away, and a registration drawing them as the rest. Until the compiler lists the I/O shell's
   units, a stand-in template stands for them; on Llama TP1 that is layer 0's 160 RoPE heads (`llama32_1b_tg_l0_step1.json`).

What doesn't change:
- the draw: the service's rule, the partition file's law, on live coins read after the receipt, with ε surfaced at δ by the
  outcome;
- the per-member `--zk` sessions behind the firewall;
- Lean's verdicts, the audit record and the Lean `Run.run` agreement.

## What an accepted audit means

Each guarantee below is listed in the lock (`Security/lean-audit.json`), which records its statement, the definitions it
reads and its named assumptions.

**A `--zk` session's acceptance, read in the sampling relation.** `Flock.SecurityProofs.SampledKSLiveV3`
(`Security/Proofs/Flock/Bridge/KSLive.lean`): on the event that `flock-verify verify --zk` accepts the transcript and the
files (their ports `sha512/row/v1`, `row/v2` or `row-seg/v1` rows of distinct names, no root registered under two domains)
and the sampled-proofs run passes the registration's population, the committed transcript, read as the sampling witness, is
outside the sampling relation `RelS` at δ with probability at most δ + ε_ks + δ_link. It takes no hypothesis on the files.
It follows from `SampledKSLiveV2`, the same bound at the circuit files the verifier of record reads. Its named assumptions
are the coins (A3, `FlockSoundness.Assumptions.UniformRandomBytes`), the live verifier, and SHA-512's collision resistance.

**The property.** `NetTiming.SecurityProofs.SampledAuditAccepted : NetTiming.Guarantees.SampledAuditAccepted`, at any
sampling deployment `D`: `∀ H D L K S lay k, Claims.Sampling.KS H D → KSAccepted H D L K S lay k`. The sampled step units
are its sampled class. On the statements in scope (`ScopeAcc`), at every δ ≥ 0, every decoder within Flock's radius and
every prover, Pr[acc ∧ ¬RAcc δ] ≤ δ + ε_ks + ε_bind, with ε_ks = κ(δ) − δ + √(γ·Pr[B]) and ε_bind = Pr[B_ext]. Pr[B] and
Pr[B_ext] are the chances of finding a SHA-512 collision in the run, not an assumption. `RAcc δ` holds when:
- the faulty set is in M_δ = {B : P_a(B) > δ} (`RelS`);
- no faulty proof block is in a class drawn `all`: the transfer-gate units and the network certifier's head checks
  (`allSlots`);
- the advice (`traceVal`, free), its inputs pinned to the trace's committed values and `free` holding values exactly at
  the faulty proof blocks' committed gates (`FaultyGate`), makes NetTiming's `Accepted` hold over every observation
  sequence the trace's transfer gates carry (`Carries`), with every status complete (`Covered`). With no unit faulty,
  `free` is empty and `Accepted` is exact.

**The leak bound.** `NetTiming.SecurityProofs.SampledLeakAccepted : NetTiming.Guarantees.SampledLeakAccepted`
(`NetTiming.sampledLeak_reg`) composes the property with the count `EncardAccSeqsSampledLe`, over any deployment. Given
`SampledAuditAccepted` and `Claims.Sampling.KS` at the deployment, except with probability δ + ε_ks + ε_bind, an accepted
audit's carried sequences, cut at the horizon, number at most 2^c · #faultySets(δ) · 2^β · D^n, at whatever values the
statement's registered inputs `reg` hold (`AsReg`). So the audit leaks at most c + λ_δ + β_δ + n·log2 D bits:
- c = #(`freeInputs` \ `reg`): the free inputs are the circuit's input gates that no step before the horizon lists as
  ingress, and the registered inputs, the weights among them, don't count;
- λ_δ, which units are faulty;
- β_δ, the faulty proof blocks' committed bits;
- n·log2 D, the timing (J's D classes over n steps).

At δ ≥ miss(1) no unit is faulty, and the bound is c + n·log2 D. `SampledLeakRegistered` states the same count at the
registered inputs' registration values rather than the trace's, a disagreement being a SHA-512 collision. The bound is
meaningful only once the advice is the circuit's one free input outside the registration: while a window's circuit is a
stand-in partition rather than the whole window, its free inputs are its units' inputs, far more bits than the advice
declares, and `records_audit` labels the bound `vacuous` whenever they exceed the declared charge.

Over a monitoring period, `NetTiming.SecurityProofs.HorizonLeakAccepted` composes the windows. It takes each window's
conditional escape and its leak relation as hypotheses, which `SampledKSLive` and `SampledLeakAccepted` at the window's
history would discharge, and doesn't discharge them yet.

**The charge terms the run prints** (`summary.json` of `prove_path.py audit`; each rank's `ranks.<r>.charge` holds
`advice_bits`, `delta`, `wrong_units`, `location_bits`, `harm_bound` and `exfiltration_bound`):
- c, which the run doesn't print. Once the advice is the only free input outside the registration, c is `advice.bits`,
  the registration's `window.advice.bits`: the I1 advice's length, which `prove_path.py advice` prints as `bits`. Over
  several windows, c is their sum.
- δ is `delta`. Per rank, `ranks.<r>.audit.wrong_units` is `{delta, of, bound, rule, per_template}` (the
  `wrong_units` function in `verity/core/protocols/sampled_proofs/one_stage/audit.py`). Except with probability δ, at most
  `bound` of the `of` units are faulty, and at most `per_template[t]` of them are in template t (`Stratified.count_bound`,
  `stratum_bounds`).
- β_δ is `harm_bound`: the profile's `harm_bound`, each template's units weighted by its committed bits (`_template_bits`
  in `verity/core/protocols/sampled_proofs/one_stage/partition.py`). λ_δ is `location_bits` (`consumers.location_bits`):
  log2 of the at most Σ_{j ≤ bound} C(n, j) faulty sets. `exfiltration_bound` is β_δ + λ_δ.
- The run doesn't print ε_ks or ε_bind (the profile's `epsilon_log2` is null), or n·log2 D, which is the certifier's term.

**A plain audit** (`--plain`, `PP_PLAIN=1`) is one table under a stratified draw, each window one member with no registered
reads or shared rows (`a5.PLAIN_SCOPE`). `flock_verify_sound`'s plain part covers each session at one table (`SoundExec`,
`SoundExecH` at hidden outputs, `SoundExecHR` with registered reads), in the game form
(`Security/Proofs/Flock/Soundness/Discharge/FailClosed/One.lean`). No theorem carries a plain session's acceptance to the
sampling relation, as `SampledKSLiveV3` does under `--zk`, so a plain audit shows each session's soundness, not the audit
guarantee above.

**The named assumptions.**
- SHA-512 collision resistance: `FlockSoundness.Assumptions.SHA512CRStrict` and `SHA512CRExpected`
  (`Security/Proofs/Flock/Soundness/Assumptions.lean`), and `Claims.StrictBirthday` and `Claims.ExpectedCR`
  (`Security/Definitions/Core/Interaction.lean`).
- A3, the coins: `FlockSoundness.Assumptions.UniformRandomBytes` (the same file).
- Record custody: under `--zk`, `FlockSoundness.Assumptions.Zk.RecordCustodyZKJ`
  (`Security/Proofs/Flock/Soundness/Assumptions/Zk.lean`) and `Zk.RecordCustodyZKLaw` and its forms
  (`Security/Proofs/Flock/Soundness/Assumptions/CustodyZk.lean`); in plain, `FlockSoundness.Assumptions.RecordCustodyExec`
  and `RecordCustodyExecH` (`Security/Proofs/Flock/Soundness/Assumptions/Custody.lean`).
- `Verity.Assumptions.TrustedHost` (`Security/Definitions/Core/Assumptions.lean`), which is why `Carries` holds of the
  real run.
- The certifier's wire (`Security/Definitions/NetworkCertifier/Assumptions.lean`): W2 `NetTiming.Regenerates`, W4
  `NetTiming.Packed` and `NetTiming.Allocated`, and J `NetTiming.JitterBounded`. They enter through the wire bounds, which
  have no sampled form yet.
- For privacy, not soundness, under `--zk`: `Claims.Hm96Hiding` (`Security/Definitions/Claims/Hm96.lean`).

**Off the page:** whether the circuit matches the model. That is completeness: an honest run on a circuit that doesn't
match is refused. It is never a soundness gap.

## The compile unit (the target; not built)

The unit closes the named gap: the advice is never opened, and the verifier never recompiles.

- **Selection All.** Each registration has one compile unit, always proved. Its Spec's selection is `service.All`, so it is
  never sampled. It reads no coins, and its proof goes through the firewall like every unit's.
- **Public inputs.** Only kinds AGENTS.md allows (protocol structure and hiding commitments):
  - d_C, the registered compiler's digest (`Capped.digest`), which fixes the compiler's code and ℓ;
  - ρ, the advice's root (`window.advice.root`);
  - b, the advice's length in bits (`window.advice.bits`, which is public and is the length term the certifier charges);
  - Π, what the window's Programs commit to: either the window's `served_programs` (request id ↦ Program digest) with the
    population Program's and partition's digests that R1–R7 hold the registration to, or, with the compiler's padded public
    step templates, the root of a hiding commitment to the routing (which committed row of each step feeds which public template
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

## The pieces

- **The compiler**, with ℓ = `CAP_BITS` (1 MiB) unless `--cap-bits` gives another:
  - on a TP Commit, `Capped` over `advice_windows.Traced`: the Build's Programs of record, refused unless they are of record.
    On a declared run it emits exactly what a correct compiler emits. I2 doesn't compile a TP rank's Programs yet.
  - on a single-GPU Commit, `Capped` over `I2Compiler`: I2's `predict.steps.compile_steps` (v0) on the advice, under φ from the
    workload's declared inputs (`score.phi_of`: its row id, the served workload file, refused unless its digest is the record's, and
    the checkpoint's `config.json`), with each request's prompt and sampling from the record. Its digest names I2's code
    (`code_identity`: `verity` and `verity_vllm` as git lists them, docs and ignored build outputs aside), φ, the seed and ℓ.
    A Program's body is read from the Build dir whose `result.json` names I2's digest, refused where none does, and held to
    that digest (`build_of`): recomputed from its `descriptor.json.gz`, with every recorded digest and every
    `instances.json.gz` row held to it, so a Build whose body was swapped under its old `result.json` is refused. No Program
    digest covers the rows' names, the parameters' names and types, `supplied`, the correspondence or `prefix_cache.json`, so
    the verifier (`audit`, and the compile command `circuit`) reads them only from its own Build of each Program, which its
    own compile writes with the names vLLM's trace of it gives (`compile_steps(named=True)`, `naming.write_build`,
    `own_builds`; no vLLM, and `--vllm-python` is ignored) into `--builds` (default `OUT/builds`, never under the row dir),
    each held to I2's digest. The prover's Builds are its row dir's. The audit summary's `i2` holds I2's Programs beside the
    Build's `build_request_LP*` digests.
  - on a served window, the same over φ of the served engine as vllm's advice compiler compiles under (`phi_of_serve`:
    `serve.json`'s model, revision, target profile and execution label, the checkpoint's `config.json`), the public default
    seed (`registered_seed`: `engine.served_seed.DEFAULT_SEED`; a run whose `serve.json` records another or none is refused,
    as vllm's advice compiler refuses it), and each request's body as that compiler gives it (`serve_ingress`: the sampling
    fields its ingress record says it brought, with its prompt ids, by its I1 id: its `user`, else its HTTP id). It is named
    as `verity-vllm advice` names it and runs the server's code (`served_identity`: `compile_steps` by module path, its
    digest over `source_identity.code_identity`, which covers `integrations/vllm` and `verity`), so the audit runs on a tree
    whose two are the server's. Its digest is the single-GPU one (`I2Compiler.digest`) with that code in place of this
    tree's: it covers φ (`serve.json`'s model and revision, that pin's `config.json`, the target and the execution), the
    default seed, ℓ and the transfer gates, so a `serve.json` whose model, revision, target or execution differs from the
    one the advice was registered under names another compiler, refused before the draw (`serve_check`'s `compiler`), and
    one recording another default seed is refused outright (`registered_seed`). `prove_path.py compiler --serve-run DIR`
    prints the binding's compiler. I2's Programs are held to the ones vllm's advice compiler wrote into the row dir
    (`manifest.json`, by I1 id).
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
  table names for it (`prove_path.py circuit` for a served window, which takes no row dir and compiles over its own Build of
  each Program), on advice that opens the registered root, and holds its output to the circuit record (less the run root and
  the coins' context) and its charged length to the registered one, so a row dir whose names make another circuit under
  unchanged digests is refused (`circuit`). It imports no `verity_vllm` code.
- **A single-GPU window** is a stand-in of the Commit's `rows-sha512` run root (`stand_in1`, held by `source1`), as
  `deferred_rows.from_bundle` writes one. The sealed bundle records the Commit's absolute node paths, and none is read: the
  prover's Programs come from `--row-dir` and the verifier's from its own compile, and the Build manifest falls back to
  `<row dir>/manifest.json`, checked against the record's sha256.
- **The advice** is I1's canonical encoding. A declared run's comes from its workload (`advice_windows.advice_of_workload`), a
  served run's from its step trace.
- **The opening** is the named gap above. `run.sh` keeps neither the advice nor its opening in `$OUT`.

## The class circuit's size and its window's commit

`class_size.py` sizes a class circuit in ANDs and C-Flock blocks, and `class_rows.py` counts its rows by what fills them and
their SHA-512 compressions (`--probe S,NP,O` lowers one Program at a small class). `window_cost.py` prices a
window's commit from that census, or from a probe projected to a class (`--project 8,1024,64`, which reproduces the exact
census unit for unit), against its target of 0.20 to 0.25 of the served window. It reports in ANDs (58,120 a
compression) and in seconds (the window's rows on the GPU's hashing rates, the registration on the CPU's, which
`--measure-cpu` measures). It covers each lever alone and the plan stacked, the pad pass at three traffics, the
registration's bytes and files, and totals over a week, a month and a year. With `--stream SIZING` it also prices serving's
own stream commit (rows-sha512, made while it serves, before the window's commit) and its seal, from serving's stream of
record (`native_host.schedule_stream_bound`), whole and without the attention interiors (`fa*.m1`, lever F), which nothing
reads after the seal:

~~~text
python window_cost.py PROBE.json --target h200 --project 8,1024,64 [--measure-cpu SCRATCH] \
    [--stream ../../integrations/vllm/tests/pipeline/fixtures/admission/b16.json] [--out FILE]
~~~
