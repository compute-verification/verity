# Prove path

Network accounting under arbitrary vLLM usage proves a served run's step units after the fact: a precommitted compiler turns
the run's advice into each step's Programs, the committing vLLM has recorded each step's values, and the proof service draws
units on live coins, proves them with C-Flock `--zk` behind the Lean firewall, and the Lean verifier accepts or refuses them.
`prove_path.py` is that path on a committed TP run: the prover commits the advice under hiding rows and each rank's window over
the compiler's Programs for it, and the verifier audits each rank through the one-stage audit's per-rank `service.Stream` on its
own coins (`benchmarks/one_stage/a5.py`). `run.sh` builds the prover and the Lean verifier and runs it on the CPU TP stand-in or
on a deferred TP Commit (`PP_TP`), with one forged run that must be refused. The docstrings say how.

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
   - Each of these fails `source` before any coin is read (a5's `advice_negatives` and `more_negatives`): a registration naming
     another compiler, another advice root, another cap, or none; the opening under another salt; another rank's registration.
   - The advice can only choose among Programs the compiler emits. Every unit's values are still committed under hiding leaves
     and opened only inside the proof.

What doesn't change:
- the draw: the service's rule, the partition file's law, on live coins read after the receipt, with ε surfaced at δ by the
  outcome;
- the per-member `--zk` sessions behind the firewall;
- Lean's verdicts, the audit record and the Lean `Run.run` agreement.

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

- **The compiler** is `Capped` over `advice_windows.Traced`: the Build's Programs of record, refused unless they are of record,
  with ℓ = `CAP_BITS` (1 MiB) unless `--cap-bits` gives another. On a declared run it emits exactly what a correct compiler emits.
- I2's `predict.steps.compile_steps` (v0) compiles a request's Program from φ at its advised shape, but not yet a TP rank's
  Programs, which a rank's window needs. Once it does, its adapter (`name`, `digest`, `load(rank, world, requests)`) goes under
  `Capped` in place of `Traced`, with no change to the audit.
- **The advice** is I1's canonical encoding. A declared run's comes from its workload (`advice_windows.advice_of_workload`), a
  served run's from ADVICE-SRC's step trace.
- **The opening** is the named gap above. `run.sh` keeps neither the advice nor its opening in `$OUT`.
