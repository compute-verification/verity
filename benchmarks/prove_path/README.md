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

## What an accepted audit means

The sampled step units are the sampled class of proofs' `NetTiming.Guarantees.SampledAuditAccepted` (#3608, a statement
with no proof yet). Its one hypothesis is `Claims.Sampling.KS` at the same deployment. On the statements in scope, at
every δ ≥ 0 and every decoder within Flock's radius, it bounds Pr[acc ∧ ¬RAcc δ] ≤ δ + ε_ks + ε_bind, where `RAcc δ` holds
when:
- the faulty set is in M_δ = {B : P_a(B) > δ} (`RelS`);
- no faulty unit is in a class drawn `all`: N2's head checks and the transfer-gate units, which are exact;
- some advice (a, free) makes NetTiming's `Accepted` hold over the trace's transfer gates, with `free` holding values only at
  the faulty units' committed gates. With no unit faulty, `free` is empty and `Accepted` is exact.

So beyond the advice, an accepted audit leaks the faulty units' committed bits, and the faulty set is in M_δ.

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
- **The audit from the records alone** (`records_audit.py audit`, `run.sh` with `PP_RECORDS=1` on a served window) reads only
  what an integration hands Verity (`verity.ml.windows.interface`): the window's `circuit.json`, which its registration binds
  by SHA-512; the registered compiler and the advice's root and length; the registration and the committed values; and the
  advice's opening, the named gap. The verifier's window is the circuit record's, each template's Definition from Verity's
  catalog or the verifier's own input set. The compile check runs the registered compiler as the command the verifier's
  table names for it (`prove_path.py circuit` for a served window), on advice that opens the registered root, and holds its
  output to the circuit record (less the run root and the coins' context) and its charged length to the registered one. It
  imports no `verity_vllm` code. The exception is a template whose Definition is still vLLM's (`SiluMul_v2`, `RoPEHead_v1`):
  the input set loads it from `verity_vllm.program.registry` until #3927 moves those Definitions into `verity`.
- **A single-GPU window** is a stand-in of the Commit's `rows-sha512` run root (`stand_in1`, held by `source1`), as
  `deferred_rows.from_bundle` writes one. The sealed bundle records the Commit's absolute node paths, and none is read: the
  Programs come from `--row-dir`, and the Build manifest falls back to `<row dir>/manifest.json`, checked against the record's
  sha256.
- **The advice** is I1's canonical encoding. A declared run's comes from its workload (`advice_windows.advice_of_workload`), a
  served run's from ADVICE-SRC's step trace.
- **The opening** is the named gap above. `run.sh` keeps neither the advice nor its opening in `$OUT`.
