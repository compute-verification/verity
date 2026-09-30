# Flock interactive soundness: what the theorems assume

For the expert reviewer and for Table 1. This file states every named assumption and every named hypothesis the Flock
soundness theorems take, each in full; the detail behind them is in `assumptions/`. What is proved is in `README.md`
and, for the audit layer, `FlockSoundness/Audit/README.md`. `DESIGN.md` holds the derivations.

**The end-to-end theorem** is a skeleton: `flock_e2e_count` and `flock_e2e_drawn`, in `FlockSoundness/E2E.lean`. They
state the chain from the audit's acceptance to the integrity profile on the committed circuit. Each unfinished piece is a
named hypothesis, and `assumptions/e2e-checklist.md` lists them with their owners. Two of those carry what the statement
doesn't say yet: that the acceptance is the executable verifier's (`hExec`), and that the units' rows are the ones the
verifier derives (`dp`, `hL1`).

## Named assumptions (taken without proof)

There are three, each a `Prop` taken as a hypothesis:
- A2 (`Assumptions.SHA512ExpectedTimeCR`), which the link theorem (`flock_batched_linkSoundE`) takes as `hCR`;
- A3 (`Assumptions.UniformRandomBytes`), which the live draw's theorems take as `ht`;
- A4 (`Assumptions.KeyedStreamsUniform`), which the PoUW window's keyed draw takes as `hA4`;
- A5 (`Assumptions.UniformSecret`), which the same draw takes as `hsec`.
`README.md` §1.5 names one more, L1, about the lowering, stated below. The development declares no `axiom`. So a
mis-stated assumption can make a theorem vacuous, but it can never make Lean prove something false.

**A2. Expected-time collision resistance of SHA-512** (in CDGSY24's expected-time sense, ePrint 2024/1434), for the
audit's value layer (Daniel, Sep 27; constant corrected the same day).

- **Statement.** A collision finder whose expected number of SHA-512 evaluations is `T` succeeds with probability at
  most `T/2^256`. `SHA512ExpectedTimeCR` states it for one finder: a game with a fixed strategy, whose outcomes
  record the pair output and the evaluations made, `t` for each run of the prover.
- **Per finder.** Some finder outputs a collision after no evaluations, since collisions exist. So the `Prop` is not
  stated for all finders: the link theorem takes it for the finder it builds from the prover.
- **Why it is plausible, and where it is used:** `assumptions/a2-sha512-expected-time-cr.md`.
- **Upstream:** watch entries `collision-resistance`, `keyed-cr` and `rom-cr` (see Upstream status).

**A3. The verifier's operating-system randomness is independent uniform bytes** (`uniform/io-getrandombytes`), for the
unit draw.

- **Statement.** A byte source is a finite probability space and, on each outcome, the first `L i` bytes of each stream
  `i`. `UniformRandomBytes L t` says every event about those bytes has the probability it has for independent uniform
  bytes. What A3 assumes of the operating system is that the running verifier's calls to `IO.getRandomBytes` are such a
  source, each call's bytes independent of every earlier call's.
- **Why it is not stated about `IO.getRandomBytes`.** To Lean's logic an `IO` action is a function of the world and
  `IO.getRandomBytes` an opaque constant, so a `Prop` saying its bytes are random would be refutable, and every theorem
  taking it vacuous. At run time, Lean v4.34.0's runtime implements it as `lean_io_get_random_bytes`
  (`src/runtime/io.cpp`), which reads `/dev/urandom` on Unix and calls `BCryptGenRandom` on Windows. Re-read it at every
  toolchain bump. What ties the statement to the verifier is `drawOS_def`: the live draw `Flock.Draw.drawOS` is
  `Flock.Draw.drawWith` with every stream's bytes from `IO.getRandomBytes`, and the theorems are about `drawWith`. The
  step from `IO` to the model rests on `drawWith` being generic in its monad, together with A3.
- **Where it is used.** `drawOS_subset_escape_le`, `drawOS_stratified_escape_le`, `drawOS_work_escape_le`,
  `execOS_escape_le` and `execOS_subset_escape_le` (`FlockSoundness/ExecDrawOS.lean`). They hold for every number of
  bytes `L` the model reads. The live verifier reads as many as its reruns need, and its escape is the limit as `L`
  grows, which the theorems don't state. `subset_exec_escape_le` and `stratified_exec_escape_le` take uniform streams
  as their probability space, the same assumption.
- **Upstream:** watch entry `A3-uniform-bytes`.

**A4. `verity.randomness`'s keyed streams at distinct indices pass a test as independent uniform streams do, up to `η`**
(`prf/sha-256`), for the PoUW window's keyed draw. The red team would grant it as a named hypothesis
(`lanes/pous/20260929T1724Z-redteam-a4-keyed-streams.md`).

- **Statement.** A secret of `S` uniform bytes (A5) gives the key `Randomness.derive`. For a test `E` of the key's
  streams at indices whose framings (`Randomness.frames`) are pairwise distinct, each read to `L i` bytes
  (`Randomness.stream`), the streams pass `E` with at most the probability that independent uniform streams do, plus
  `η`. `Randomness.lean` is the Lean spec of `verity.randomness` v1: it reproduces
  `packages/verity/tests/randomness/vectors.json`, and Python matches its vectors
  (`backends/flock/tests/test_randomness_spec.py`).
- **Per test, as A2 is per finder.** For every test the statement would be refutable, since `2^(8S)` sources give at
  most that many stream tuples. `keyedWindow_escape_le` takes A4 for one test per set `B` (`windowTest`: the window draw
  misses `B`), and `η` enters its bound.
- **What it assumes of SHA-256.** Its compression function `f` is a pseudorandom function in two ways.
  - Keyed by its chaining value, it makes the cascade a PRF on prefix-free inputs (Bellare, Canetti and Krawczyk 1996).
    The inputs are prefix-free: `frames` is length-framed and the counter is 8 bytes, so length extension doesn't apply.
  - Keyed by the secret bytes in its first message block, it is a dual PRF, as the HMAC analyses use. The secret follows
    a fixed public tag: 28 bytes for `stream`, and 37 for `derive` with the source's frame header.
  - The secret's uniformity is not A4's but A5's.
- **η, stated.** η stays symbolic in Lean, and every record-level statement keeps `+ η` visible, never absorbed.
  - **The claimed value:** for `S = 32`, η ≤ 2⁻¹²⁸ for the chain's tests. The derivation is generic key search over
    `min(256, 8S)` bits, plus a hybrid over the `q` compression calls the test's streams take. With `q ≤ 2⁴⁰` and a
    test of `T ≤ 2⁸⁰` hash evaluations, the bound stays under 2⁻¹²⁸.
  - **One η for the whole family:** every set `B` (the closures of the bad sets, which `closure_escape` evaluates), every
    stream budget `L`, and every receipt context. The audit takes a sup over `B`, not a union, and `E_B` costs the same
    for every `B`, every `L` and every context.
  - **Vacuity:** A4 is plausible only for `S ≥ 32`, and η can never be below about 2⁻⁸ˢ, since keyed probabilities are
    multiples of that. It holds trivially at η ≥ 1, and is claimed only for the chain's efficient tests `E_B`.
- **Upstream:** watch entry `A4-prf-sha256`.

**A5. The PoUW window's secret is `S` uniform bytes, fresh for the window** (`uniform/python-secrets`).

- **Statement.** A source is a finite probability space and, on each outcome, the secret. `UniformSecret sec` says every
  event about the secret has the probability it has for `S` uniform bytes. `keyedWindow` draws over the real secret
  source, and its theorems take A5 as `hsec`. It is satisfiable: the identity source satisfies it
  (`Law.uniformSecret_id`, `KeyedDraw.lean`).
- **What A5 assumes of the verifier.** Its window secret is `secrets.token_bytes(32)` from the operating system's CSPRNG,
  drawn once the window's receipt is complete and used for that window alone (#423's `Ledger.open`, root's C2). A
  secret reused across windows is outside it, since later windows would be committed after earlier draws were seen.
- **The platform: Linux, with CPython 3.12 or later** (the red team's wording,
  `lanes/pous/20260929T1817Z-redteam-423-receipt-key.md`, question 3). Re-read this on any other platform.
  - `secrets.token_bytes` reads `os.urandom`, which is `getrandom(2)`, the kernel's CSPRNG. It blocks until that is
    initialized.
  - It keeps no user-space state, so a forked process doesn't repeat its output.
  - A VM snapshot restored more than once must not repeat the kernel's CSPRNG state. That holds when the kernel reseeds
    on a VM generation change (vmgenid, Linux 5.18 and later, with hypervisor support) or mixes in a hardware RNG.
  - The CSPRNG is idealized as exactly uniform. The true bound carries its negligible distinguishing advantage.
- **Split out of A4** at the red team's request: it is a property of what produces the secret, not of SHA-256.
- **Upstream:** watch entry `A5-uniform-secret`.

- **L1. Each template's rows compute its gates** (named hypothesis). That is, the rows' outputs are the gates' outputs, in
  the program's Boolean circuit, on every input. The audit certifies drawn units against their rows. That says something
  about the program's Boolean circuit only through L1.
  - **Proved for RoPE** (`rope-head/neox-bf16/pair`): `Rope.rope_sound`, for the pinned file's rows.
  - **The evidence for it, the proof for RoPE, and the path for the rest:** `assumptions/l1-template-rows.md`.

A1, Reed–Solomon mutual correlated agreement up to the Johnson bound, is proved: `mcaError_le_bchks25`, from DKT26 in
ArkLib (`README.md` §1.6; watch entries `A1-johnson-mca` and `udr-mca`).

## Named hypotheses of the proof

- **The link points' layout** is a hypothesis (`Statement.LinkLayout`). Each region's free bits must be distinct and
  read link coordinates below `PT_LOCAL`, and the block coordinates `PT_LOCAL …` must fit inside the point. The
  statements of PROTOCOL.md §16 satisfy it.
- **The placement** (`Placement`; `lowering_sound`, `README.md` §1.5):
  - Placed (`Placement`) means two things:
    - the block matrices' rows at the unit's computed columns read exactly its rows' columns;
    - the rows at its constant read the pin, as every slot's constant is wired to it (spec §16.1).
  - A unit's input columns on one gate also carry one value in every satisfying witness (`UnitPlace.aliased`): they
    copy one source's position, or forced-zero rows.
  - **Proved from the executable's accept step** for `hm96-sha512/row/v1` statements (`placement_of_setupH`,
    `unitPlace_of_setupH`; `README.md` §1.5). Open: which instance a drawn unit is (its block and VU), and `aliased`
    from the flat statement's copies, which `unitPlace_of_setupH` takes as a hypothesis (`Layout.Aliased`).

**The audit layer's named hypotheses** (`Prop`s taken as hypotheses; no `axiom`; the theorems are in
`FlockSoundness/Audit/README.md`):

| Hypothesis | Term | Meaning | To be discharged by |
|---|---|---|---|
| `Analysis.KnowledgeSound` | `ε_ks` | the session accepts while extraction of a drawn unit (picked before the session's first coin) fails | `table_knowledge_sound_joint` (`README.md` §1.3), after Jensen over the prover's states |
| `Analysis.LinkSound` | `δ_link` | extraction succeeds but differs from `X` on the unit's committed inputs and outputs | compiled layer: `flock_batched_linkSoundE` (A2, `ValueBinding`) |
| `Partition.AnchorsSound` | `δ_in` | the session accepts while the committed inputs differ from the anchors' values | `registered_weights` (`README.md` §1.3); at the oracle layer, `InputsSound` (`oracle_anchorsSound`) |
| `Partition.AnchorsSound₂` | `δ_in` | two-stage audits: for every strategy, the session accepts while the committed coarse transcript's inputs differ from the anchors' values | as `AnchorsSound`, for the two-stage game |
| `Partition.InputsSound` | `δ_in` | oracle layer: the session accepts inputs other than the anchors' with probability at most `δ_in` | **proved for Flock** (`flock_inputs_sound`, `2^-205`), given `InputsLoweringSound` |
| `Partition.InputsLoweringSound` | — | inputs other than the anchors' leave some session table (the always-proved input statement) unsatisfied | the input unit's lowering (level 3) |
| `Partition.SessionSound` | `ε` | oracle layer only: a session with a drawn wrong unit accepts with probability at most `ε` | **proved for Flock** (`flock_session_sound`, `2^-205`), from `table_value_sound`, `execArith_correct` and the lowering |
| `Partition.LoweringSound` | — | a drawn wrong unit lies in a session table whose statement no committed table satisfies | the lowering theorem (level 3, another lane) |
| `IsHarmBound` | `H` | every wrong-unit set escaping with probability at least `δ` has harm (a consumer's weight per wrong unit) at most `H` | the value core's `Stratified.harm_bound` returns; that implementation is not proved |

Both per-unit hypotheses are averages over the whole audit, for every prover and every rule that picks a drawn unit
before the session's first coin, not bounds state by state.

| Named at the compiled layer | Meaning | Discharged by |
|---|---|---|
| `ExtractionAnalysis.LinkSound` (`analysisB`, `analysisC`, `analysisBE`) | `δ_link`: extraction succeeds, but every close satisfying candidate decodes, on the unit's committed wires, to values other than the committed transcript's | for `analysisBE`: `flock_batched_linkSoundE`, under A2 for each commit string's finder and `ValueBinding` (named; the `hm96` layout discharges it); composed in `flock_batched_count_linked`, `_drawn_linked` (`_placed`: from `UnitPlace`) |
| `Partition.LoweringSoundB`, `Partition.LoweringSoundC` | a message packing a witness that satisfies the statement of a drawn unit's table decodes, for that unit, to values on which it is correct | `loweringSoundB_of_place`, `loweringSoundC_of_place` from each drawn unit's `UnitPlace` (`lowering_sound`, `README.md` §1.5), given the statement builder's `Placement` and `aliased` (level 3) and L1. `unitPlace_of_setupH` gives it from `Stmt.setupH`, given the unit's instance in the audit circuit and `aliased` |

## Cryptographic assumptions and coins

- **Collision resistance of SHA-512** (`cr/sha-512`) for the Merkle openings (spec §14, with SHA-512 leaves and nodes).
  It is the only hash assumption of the compiled theorem: plain collision resistance, for every classical prover.
  The audit's value layer adds A2 (above).
  DESIGN.md §1 explains why SHA-512 rather than SHA-256, and how the bound depends on the hash's width.
- **No assumption for the rounds.** The verifier keeps each round's bytes, then draws that round's coins, and the
  replay compares bytes. This replaces the SHA-256 round digests. Keeping the bytes costs about the proof size, 0.7 MB
  per rep.
- **Statement binding** (the statement digest, Σ and `link_sha256`). These are in round 0's kept bytes, fixed before any
  coin. Their own collision resistance matters to what the statement refers to (registered data, linked proofs), not to
  this theorem.
- **`os` coins** (the default). Each round's coins are uniform and unknown to the prover before it sent the round. They
  come from the verifier's `/dev/urandom` (spec §7.1). The randomness is trusted, not proved. For the unit draw it is
  A3.
- **`seed` coins** (PR #83, `coin-commit/sha256/v1`) add two assumptions: the seed commitment hides the seed, and
  SHA-256 in counter mode is a PRF. They are needed only if the seed mode becomes the verifier of record's source.
- **The random-oracle model is not used.** Its version of the compiled bound is only a cross-check (DESIGN.md §2).
- **Quantum provers** (DESIGN.md §9). The oracle layer holds against them, but the compiled theorem is classical. The
  compiled protocol is post-quantum sound only asymptotically, and only if SHA-512 is collapsing, which is a conjecture
  (Table 1 would need `collapsing/sha-512`). No concrete post-quantum bound is known at these parameters, and none is
  claimed.
- **Table 1** cites Flock's soundness under the `live-verifier` model and the collision resistance of each row's hashes
  (`cr/sha-512` in the C-Flock SHA-512 row); A1's proof took `rs-proximity-johnson` off. An audit's integrity profile
  would add A2 as `ecr/sha-512`, a `verity.claims` id that no row cites yet.

## What is pinned

`lean-audit.json` records each pinned theorem's statement (`pins`) and the definitions it reads (`reads`). The audit
fails when either changes until the record is rewritten, and a rewritten record needs a named statement reviewer. The
pins include the table theorems (`README.md` §1.1 and §1.2, with their `_exec` forms), `execArith_correct` and
`Merkle.opening_binding`; none takes a named assumption. So do the knowledge-soundness theorems of `README.md` §1.3
(`table_knowledge_sound`, `table_knowledge_sound_joint`, `table_knowledge_sound_joint_tight`, `session_knowledge_sound`,
and the batched session's `flock_batched_knowledgeSoundE`), and the link theorem, `flock_batched_linkSoundE`, which takes A2
as `hCR`. The end-to-end skeleton's `flock_e2e_count` and `flock_e2e_drawn` are pinned too; they take A2 as `hCR` and pass
it to the link theorem. `flock_inputs_sound`, `flock_session_sound`, `lowering_sound` and `mcaError_le_bchks25` are not
pinned. A theorem that is not in `pins` is still audited (standard axioms, kernel replay), but a change to its statement
would not fail the audit.

## Upstream status

Whether a dependency proves what these theorems assume, or what they prove by hand, is computed, not written here. The
watch list is the `upstream` section of `lean-audit.json`, one entry per statement:

- `A1-johnson-mca` and `udr-mca`: A1 (proved here from DKT26) and its unique-decoding fallback;
- `collision-resistance`, `keyed-cr` and `rom-cr`: A2, in three forms;
- `list-size` and `list-size-code`: the Johnson list bound (`ListSize.closeMsgs_card_le`);
- `tensor-fold`: the fold union in `LigeritoLevels`;
- `novel-basis`: the novel basis (`Arith.Correct.xhat_poly`, `L ≤ 40`).

`python tools/lean/upstream.py backends/flock/verifier/lean/soundness` prints each entry's status at the pinned
revisions: the upstream theorems about it, which are proved, and which nobody has acknowledged. The audit runs that pass
over ArkLib, with ArkLib's own `sorry` baseline, and fails on a proved theorem no entry acknowledges. `--lean`, the
exact pass over everything imported (VCVio and CompPoly too), is run at a dependency bump. L1 has no entry: no
dependency states this repository's rows.

## Trusted components

Not assumptions of the theorem, though the verdict depends on them: `assumptions/trusted-components.md`.
