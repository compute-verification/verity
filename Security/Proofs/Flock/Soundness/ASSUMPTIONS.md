# Flock interactive soundness: what the theorems assume

For the expert reviewer and for Table 1. This file states every named assumption and every named hypothesis the Flock
soundness theorems take, each in full; the detail behind them is in `assumption-notes/`. What is proved is in `README.md`
and, for the audit layer, `Audit/README.md`. `DESIGN.md` holds the derivations.

**The guarantee of record** for the one-stage `--zk` audit is `Flock.Guarantees.EndToEnd`
(`../EndToEnd/Statement.lean`, proved as `Flock.SecurityProofs.EndToEnd`). It bounds the executable verifier's
acceptance, `flock-verify verify --zk`, with no model between them, under A3, `live-verifier` (with A6), `cr/sha-512`
and `ecr/sha-512`, each a hypothesis (below). The rest of this section and The headline describe the older, model-level
forms.

**The end-to-end skeleton**: `flock_e2e_count` and `flock_e2e_drawn`, in `E2E.lean`, and
`Prog.flock_e2e_count` and `Prog.flock_e2e_drawn` on the circuit of the statement's rows (`Prog.circuit`). They state the
chain from the audit's acceptance to the integrity profile on the committed circuit. Each unfinished piece is a named
hypothesis, and the headline's footprint (below) lists them with their kinds. The circuit is the one the statement's rows
make, chosen on the untrusted side, and the lowering does no optimizing. So no hypothesis relates the
rows to another circuit: what a correct unit computes is its certificate's (The circuit and the certificate, below). One
hypothesis carries what the statement doesn't say yet: that the acceptance is the executable verifier's (`hExec`; in the
`_hm96` forms, the serving roots' rewinding `tr`, which adds `δ_tree`).

**The headline** is `Prog.flock_headline` (`Headline.lean`), the one statement the others specialize into
(The headline, below).

## Soundness's assumptions (taken without proof)

Soundness rests on five. Three are `Prop`s in `Assumptions.lean`, taken as hypotheses: SHA-512's
collision resistance in two forms, and the Lean verifier's randomness for the unit draw. The fourth, A6,
is the coin server's randomness for the round coins, which the model embodies (A6, below). The fifth, `live-verifier`, is
the record's custody (`PROTOCOL.md` §17.2): `--zk`'s soundness (`zk_session_sound_custody`) takes it as
`Assumptions.Zk.RecordCustodyZK`, in `Assumptions/Zk.lean`. The table's last row is zero
knowledge's, not soundness's: a model claim about the honest prover. The development declares no
`axiom`. So a mis-stated assumption can make a theorem vacuous, but it can never make Lean prove something false.

| `Prop` | Claim id | Bound | Taken for |
|---|---|---|---|
| `Assumptions.SHA512CRStrict` | `cr/sha-512` | `q²/2^513`, for at most `q` evaluations on every outcome | the compiled and knowledge terms' collision finders, `δ_tree`, and the registered reads' `δ_reg`, and, in `CROnly`, the link finders stopped at `q` |
| `Assumptions.SHA512CRExpected` (A2) | `ecr/sha-512` | `T/2^256`, for `T` evaluations in expectation | the link finders (`LinkCR`, `hCR`) |
| `Assumptions.UniformRandomBytes` (A3); per test, `UniformRandomBytesTest` | `uniform/io-getrandombytes` | every event has its probability for uniform bytes, the CSPRNG idealized; per test, one event, within `η` | the unit draw (`ht`; `hA3` in the headline); per test, zero knowledge's salts (`Discharge.ZkView.OsSalts`, `recursiveZK_os_gap`) and the ZK server's tape (`Claims.ZKServer.OsTape`) |
| none: the model's coins (A6) | `uniform/flock-coin-server` | each round's coins uniform, drawn after the prover's message; under `--zk`, committed at `Hello` and opened after it, at a cost of `N·2^-256` (A6, tree coins) | the round coins |
| `Assumptions.Zk.RecordCustodyZK` | `live-verifier` | none: the record `verify --zk` reads is the one the verifier's coin server wrote for the live session (`RecordsLiveZK`), and its draw file the one `flock-verify draw` printed for the verifier's own draw (`DrawFileZK`) | `--zk`'s `hRec` and `hDraw` (`zk_session_sound_custody`, `hCust`; `hDraw` at each draw of `Law.execOS` is `drawOkZK_of_exec`); at `J` tables `Assumptions.Zk.RecordCustodyZKJ`, the same at `RecordsLiveZKJ` and the subset sampling design (`zk_session_soundJ_custody`; `hDraw` is `drawOkZKJ_of_file`), and under hidden outputs (`zk_session_soundHJ_custody`; `hDraw` is `drawOkZKHJ_of_file`) |
| `Assumptions.RecordCustodyExec` (`…ExecH`); `Zk.RecordCustodyZKLaw` (`…ZKHLaw`, `…ZKJLaw`, `…ZKHJLaw`) | `live-verifier` | none: as `RecordCustodyZK`, with `DrawOk` (`DrawOkZK`, and forms) at each draw of the law | `SoundExec*`'s and `SoundZ*`'s `hCust` |
| none: the model's Ligerito prover (`LigProver`) | none: a model claim, as A6's | the honest Ligerito prover's level-≥ 1 bytes are a function of the table, the statement, the public coins, the restart and fresh salts; nothing proves it (`LigProver`'s docstring, `Discharge/Composed/Inner.lean`) | zero knowledge's view, not soundness: `Composed.zk_session_view`, `zk_session_composed` and `ZkEveryCoin.zk_session_view_all` hold for every such prover `P`, as do `ZkHidden.zk_session_viewH`, `zk_session_viewHJ` and `zk_session_composedHJ` |

Both collision-resistance `Prop`s are **per finder, and per prover**. Some finder outputs a collision after no
evaluations, since collisions exist, so neither is stated for all finders: each theorem takes it for the finders it builds
from the prover it bounds. Neither is asked of every prover at once, which SHA-512 can't meet: a prover that hard-codes two
openings of one commit string has finders that find a collision. A finder is a game with a
fixed strategy, whose outcomes record the pair it outputs and the SHA-512 evaluations it made, `t` for each run of the
prover. Its success is a collision because Merkle binding is proved (`Merkle.opening_binding`): two openings that verify
under one cap and differ give two distinct inputs with one SHA-512 output.

**Teeth** (`Teeth.lean`). Both forms hold for every finder of an injective hash (`strict_of_injective`,
`expected_of_injective`, the latter for a nonnegative cost), and both fail for a constant hash, at the finder that outputs
two different strings after no evaluations (`not_strict_const`, `not_expected_const`). So neither form is true of every
hash, nor false of every hash.

**`SHA512CRStrict`. Strict-time collision resistance of SHA-512** (`cr/sha-512`).

- **Statement.** A collision finder that makes at most `q` SHA-512 evaluations on every outcome succeeds with probability
  at most `q²/2^513`: the generic birthday bound `q(q − 1)/(2·2^512)` for a budget fixed in advance (`DESIGN.md` §1).
- **The budgets are not checked.** `StrictCR.Finder.CR H F q` takes `SHA512CRStrict` with the cost `q` on every outcome,
  so nothing in Lean ties `q` to the evaluations the finder makes. Each budget must bound its finder's real SHA-512
  evaluations, for `t′` one session run's (as in `LinkCR`) and `v` one opening's verification:
  - `qF ≥ 2t′ + 2v`, for the rewinding finders (two runs after a cap, and the two conflicting openings verified);
  - `qS ≥ t′ + v`, for the self-clash finder (one run, one further verification);
  - `qT ≥ 2t′ + 2v`, for `tr`'s finder (`TreeRewind.finder`: two runs after the registration);
  - `qR ≥ t′ + v`, for the registered-path finder (`Reads.pick`: one run, and the two openings' climbs);
  - `qG ≥ (1 + 5M)·t′ + 2k·v`, for zero knowledge's simulator finder (`ZK.GK.Model.finder`: the GK simulator with cap
    `M`, at most `1 + 5M` runs of `V*` on every outcome by `sim_runs_le`, then T6's extraction, which re-verifies the
    two views' `k` coin openings each and hashes at most `2(|path| + 3 + n)` aligned inputs, `coinPairs_length_le`);
  - `t′` itself, for the link finders stopped at `q` (`CROnly.LinkCRStrict`, `hCRs`). `q` is checked in Lean: the stopped
    finder's cost, `truncCost`, is at most `q` on every outcome. But that is the modeled cost, `t′` per run played
    (`cT`), so under the strict form `t′` must bound every run on every outcome, not on average: the prover's
    evaluations, and the acceptance check `accω` the finder runs on each. The openings' verification and the extraction
    (`rdT`, `cand`) are charged nowhere, and must fit in `t′` or in `q`'s slack.

  A smaller budget makes the hypothesis a claim about SHA-512 that `cr/sha-512` doesn't make, and a bound cited with it
  is unsupported. Tying the five to one per-run count in the signatures would check this in Lean; it is a follow-up.
- **Where it is used.** The compiled term's rewinding finders (`Adv_{r,ℓ}`: two continuations after a rep's level-`ℓ`
  cap) and its self-clash finder (one run that opens a position two ways), the knowledge term's level-0 finder (`Adv₀`:
  two runs after the level-0 cap), the serving roots' Merkle binding (`δ_tree`: two continuations after the
  registration), and the registered reads' path binding (`δ_reg`: one run, whose registered read opens the held root at
  another leaf than the registrant's), at the budgets above. `StrictCR.lean` restates each term but
  `δ_reg` at those finders (What is restated over the two forms, below); `Registered/Binding.lean` states
  `δ_reg` at its own (`Reads.prob_off_le`). Zero knowledge's `⊥_bind` term takes it for the simulator finder at `qG`
  (`ZK.GK.Model.gk_simulate`, `gk_simulate_hm96`, `gk_simulate_avg`): `V*` opening a coin of its committed tree two ways
  is a SHA-512 collision, computed (`bindColl_isColl`, `bindCollKeyed_isColl`).

**`SHA512CRExpected`. Expected-time collision resistance of SHA-512** (A2, `ecr/sha-512`; in CDGSY24's expected-time
sense, ePrint 2024/1434), for the audit's value layer.

- **Statement.** A collision finder whose expected number of SHA-512 evaluations is `T` succeeds with probability at
  most `T/2^256`. The fixed-budget form doesn't extend to finders that stop adaptively, and the link finders do: they rerun
  the prover until a rerun accepts.
- **Where it is used.** The link theorem takes it for the finders it builds from the prover it bounds (`LinkCR … R τ …`).
  `flock_batched_linkSoundE` gives the bound at each prover whose finders satisfy it, and `⊤` at any other
  (`linkBoundCR`); the end-to-end theorems take `hCR` for `σ`'s finders alone.
- **Why it is plausible:** `assumption-notes/a2-sha512-expected-time-cr.md`.
- **Upstream:** watch entries `collision-resistance`, `keyed-cr` and `rom-cr` (see Upstream status).

**A3. The operating system's randomness is independent uniform bytes** (`uniform/io-getrandombytes`), for the
verifier's unit draw and the proof service's epoch salt, and, per test, for zero knowledge: the commitments' salts and
the ZK server's tape.

- **Statement.** A byte source is a finite probability space and, on each outcome, the first `L i` bytes of each stream
  `i`. `UniformRandomBytes L t` says every event about those bytes has the probability it has for independent uniform
  bytes; the unit draw takes it. `UniformRandomBytesTest L t E η` (as A4 is) says one event `E` has it within
  `η`; zero knowledge takes it at the event each theorem bounds, and `η` enters the bound. A3 assumes the operating
  system's reads are such a source, each call independent of the others: the verifier's `IO.getRandomBytes`
  (Lean v4.34.0's runtime reads `/dev/urandom`), the firewall's frame and tap, and the prover's and a registrant's
  salts.
- **Why per test, and the rest.** Any source with fewer outcomes than byte strings refutes the every-event form, as
  Linux's CRNG does past 32 bytes: its zero idealizes the CSPRNG, and `η` is the CSPRNG's advantage against the
  theorem's test. `assumption-notes/a3-uniform-bytes.md` has the argument, why A3 isn't about `IO.getRandomBytes`, its
  tie to the verifier, its scope (the masks included), what is outside it, why no test runs, and where it is used,
  salts included.
- **Upstream:** watch entry `A3-uniform-bytes`.

**A6. The coin server's round coins are independent uniform** (`uniform/flock-coin-server`), for the session's
coins.

- **Statement.** Each round's coins are uniform field elements, independent of every earlier round's and drawn after the
  prover's message of that round. The model's session draws its coins exactly so (the games' coin nodes), so A6 is what
  makes the model's coins the run's.
- **Why it is not a `Prop`.** The session game draws its own coins: no theorem reads a coin source it could be a
  hypothesis about, as A3 is about the unit draw's byte source. Stating it as a `Prop` would need the session played on
  a byte source, which the model doesn't have.
- **What it assumes of the coin server.** The Rust coin server (`live/src/lib.rs`, `Server::coins`) reads each round's
  coins from `/dev/urandom` when it has no seed. A3 is about the byte sources the theorems read (the unit draw's, the
  salts' and the ZK server's tape), so it doesn't cover these bytes, which the model's coin nodes draw.
- **Seed coins are outside it.** By default M0's non-ZK session derives every coin from one OS seed (`cfg.coin_seed`;
  `live/src/coin_seed.rs`, through `verity.ml.randomness.derive`). Those sessions also rest on `prf/sha-512` (A4's claim)
  and on the seed commitment's hiding, and no headline covers them. Under `FC_COINS=os` the server draws every round's
  coins as above (The headline, its coins).
- **Tree coins (`--zk`).** Under `--zk` the coins are committed, not drawn round by round (`coin-tree/hm96-sha512/v2`,
  `live/src/coin_tree.rs`; the live `PROTOCOL.md` §9). A6 for these sessions is the following.
  - *The coins.* At `Hello`, after the prover's nonce and before its first message (`Commit`), the coin server reads
    from `/dev/urandom` a fresh hm96 key (256 bytes, bit 2047 cleared: hm96 §2's key space), then a block of coins and a 192-byte salt for each stream and round
    (`CoinTree::draw_with`, at `Hello` in `live/src/lib.rs`), commits to every block as keyed hm96 leaves under one
    SHA-512 tree, and answers `Hello` with the root and the key. It opens round `r`'s leaves only in
    its answer to the prover's round-`r` message. That ordering takes the place of "drawn after the prover's message"
    in the statement above. As with A6 itself, the record can't establish it: flock-verify checks each opening against
    the root, not when it was drawn or opened.
  - *Why the model's coins are still the run's.* Given the key, each leaf has its own salt, so opening one reveals
    nothing about another. A hybrid swaps the unopened leaves for commitments to independent coins (openings resampled
    consistently); each leaf moves it by at most
    `2^-256`: hm96 `PROTOCOL.md` §3's two-vector case (§3's `2^-257` is the distance to the ideal distribution, which
    doesn't apply, because the coin digests aren't uniform). So for every event `E`
    of the transcript, `Pr_run[E] ≤ Pr_model[E] + N·2^-256`, with the published root and key treated as randomness the
    prover could have drawn itself. The step is statistical only (no hash property, no `hash-derived-key`, no CRS):
    the key is fresh for each session.
  - *N* counts every leaf committed under the session's key, opened or not: `|streams| · 2^rounds_log`
    (`CoinTree::build`; the `SHA-512(PAD)` padding is no leaf). flock-circuit's schedule (`coin_spec_of`) has the
    session's stream plus two reps per table at `rounds_log = 9`: `N = (1 + 2J)·512`, so `N·2^-256 < 2^-245` at J = 1.
  - *What the bound needs.* The salts are uniform and independent, one per leaf, never reused. The key is independent
    of the salts, drawn after the prover's nonce, and never reused across sessions. Both come from `/dev/urandom`, which
    is A6's source, not A3's. A5's snapshot caveat (The platform, under A5) applies to the draw at `Hello`. The
    test-only `seed-injection` hooks (`inject_coin_tape`, `inject_coin_edit`) are outside A6. Unlike the opening order,
    this is checkable from the record: the backend identity carries `seed_injection`, the statement digest pins it
    (`Flock/Tags.lean`), and it is false in every proving build (`check_build.sh`).
  - *Citing condition.* Until a Lean lemma states the hybrid, every citation of a `--zk` headline
    (`zk_session_sound` and its hidden-output, J-table and registered forms) gives δ as the Lean bound plus
    `N·2^-256`, and names that term as a step argued by hand.

## What is restated over the two forms

Each term of the audit's bound is under one form, at the finders built from the prover it bounds
(`StrictCR.lean` for the strict form). A rewinding finder's budget `qF` counts its two runs of the prover
and two verifications; a one-run finder's `qS` one run and one verification; `qT` is the serving roots' rewinding finder's.

| Term | Form | Theorem | Bound |
|---|---|---|---|
| the compiled theorem | `SHA512CRStrict` | `table_sound_compiled_strict` (`CompiledCR`) | `tableError + √(2^{log N₀}·2Q₀·qF²/2^513) + Σ_{r<2} Σ_ℓ √(2^{log N_ℓ}·Q_ℓ·qF²/2^513) + qS²/2^513` |
| `ε_ks`, the knowledge term | `SHA512CRStrict` | `ksBoundAccB_le_strict`, `ksAvgBE_le_strict` (`TableCR`, `hKS`) | `ksAvgStrict`: per table `ksStrict = tableError + Σ_{r<2} Σ_ℓ √(2^{log N_ℓ}·Q_ℓ·qF²/2^513) + qS²/2^513 + √(k·qF²/2^513)`, over `1 − N₀/(ek)`, its largest after each draw averaged over the draw |
| `δ_tree`, the serving roots' binding | `SHA512CRStrict` | `tree_le_strict`, `TreeRewind.le` (`tr`, `hT`) | `TreeRewind.bound qT = √(N_s·Q_s·qT²/2^513)` (`DESIGN.md` §1) |
| `δ_reg`, the registered reads' path binding | `SHA512CRStrict` | `Reads.prob_off_le` (`hR`) | `qR²/2^513` |
| `δ_link`, the link term | `SHA512CRExpected` | `flock_batched_linkSoundE` (`LinkCR`, `hCR`) | `linkBoundE`, finite at the provers whose link finders satisfy it |
| `δ_link`, under `cr/sha-512` alone | `SHA512CRStrict` | `CROnly.linkCR_of_strict` (`LinkCRStrict`, `hCRs`: the same finders stopped at `q` evaluations) | `linkBoundE` at per-run cost `strictRunCost t' q = 2^256·(t'/q + q²/2^513)`; at `q³ = t'·2^512` the hash part is `3(1 + k)·t'^{2/3}/2^{512/3}` (`CROnly.linkBoundE_at_cube`) |

**`cr/sha-512` alone** (`CROnly/`). Every e2e theorem of the non-ZK session and the headline is restated
with `hCRs` in place of `hCR` (`CROnly.flock_e2e_*`, `CROnly.uprog_e2e_*`, `CROnly.prog_e2e_count`, `_drawn`,
`CROnly.flock_headline`), so they rest on `SHA512CRStrict` and no `SHA512CRExpected`. The price is the link term's: for
audit B (`Q_s ≈ 2^24`, `ρ = 1/2`) `δ_link` is about `t^{2/3}·2^-123.5` against the record's `t·2^-209.4`, so `2^-70.2`
instead of `2^-129.4` at `t = 2^80`. `expected_of_strict` (truncation at `q` and Markov) reads the stopped finder's
bound as linear in its expected cost, which loses `(2(1 + k))^{1/3}` against the best budget for the finder's own cost.
**Scope:** these are all theorems about the non-ZK session (`batchedSession`, `Audit/FlockBatched.lean`, whose verifier
checks `final_c` in the clear). `--zk`'s soundness theorems (`Discharge/ZkSession`: `zk_flock_count`, `_execOS`) are
about a model that receives `final_c'` before the inner proof, which the verifier does not. `CROnly` restates
none of them, and calls nothing of main's `--zk` sound.

The four `flock_e2e_*_hm96` theorems (`Binding/E2E.lean`) add the four:
`miss_L(K) + ksAvgStrict + δ_link + δ_tree` for the count curve, the same without `miss_L(K)` for a drawn unit. `δ_tree`
enters through `tr : StrictCR.TreeRewind`, a rewinding at the serving roots (the prover after its registration, with the
unit draw in place of the proof coins) whose `covers` says that the executable accepting while the model at the
registered leaves rejects is no likelier than a run opening a serving root off its plurality. That cover is the link from
the executable's openings to the model's (R11), still open, so `tr` is a named hypothesis; the skeleton's `hExec` (every
executable accept is the model's) gives one with no positions, at `δ_tree = 0` (`TreeRewind.ofNever`, `ofNever_bound`).

`flock_e2e_drawn_hm96_reads` and `flock_e2e_count_hm96_reads` (`Registered/E2E.lean`) are the drawn-unit
and count forms with registered reads (PROTOCOL.md §16.10): they add `δ_reg`, the event that a registered read's leaf
under the prover's registration is not the registrant's, so that off it the binding's commitment at every registered read
is the registrant's leaf. They take the layout's registered reads (`rd : Reads`: the held roots, the positions the
verifier's program reads, and the registrant's openings, intended to prove), `hchk` (on every outcome the executable
accepts, the registered check held at the model's leaves: `Flock.Registered.check_ok` read at `Layout.regLeaf`, the
same link R11 that `tr` names, intended to prove) and `hR` (hardness: `SHA512CRStrict` for the registered-path finder at
budget `qR`).

## The headline

`Prog.flock_headline` (`Headline.lean`) is the count curve with every choice made:
- **the circuit** is the statement's rows' (`p.circuit outs`), each drawn unit's place from its table's statement
  (`pp : ProgPlaces`);
- **the hash** is SHA-512 (`Refine.H512`, `Refine.enc512`), for the session's leaves and nodes and for the rows' commit
  strings, with the rows' value binding (`flock_e2e_count_hm96`);
- **the draw** is the live one (`Law.execOS`: `Flock.Draw.drawOS`'s stratified draw on a byte source `src`), under A3
  (`hA3`).

It is proved under the assumptions its footprint lists. Its bound is `miss(K) + ksAvgStrict + δ_link + δ_tree`.

**Its footprint.** Every hypothesis of its signature is an assumption, listed with its kind: hardness, platform, model,
intended to prove, or a numeric condition on the bound's parameters.

| Hypothesis | Kind | What it is |
|---|---|---|
| `hKS` (`StrictCR.TableCR`), `hT` (`StrictCR.Finder.CR`) | hardness | `SHA512CRStrict` of SHA-512 (`cr/sha-512`), for the knowledge finders of each table after each draw at budgets `qF`, `qS`, and for `tr`'s finder at budget `qT` |
| `hCR` (`LinkCR`) | hardness | `SHA512CRExpected` of SHA-512 (`ecr/sha-512`), for the prover's link finders |
| `hA3` (`UniformRandomBytes`) | platform | A3 (`uniform/io-getrandombytes`), for the unit draw's byte source |
| none: the round coins | platform | A6 (`uniform/flock-coin-server`), the coin server's OS randomness: each round's coins uniform, drawn after the prover's message (under `--zk`: committed at `Hello` and opened after it, plus `N·2^-256`; A6, tree coins). The model embodies it (next row), so no hypothesis states it |
| none: the session's game | model | the protocol as `batchedSession` models it, against any prover `Strategy`, with the round coins A6 gives |
| `pp.placed`, `pp.aliased` | intended to prove | W6 (the table's statement holds the program's rows) and `Layout.Aliased` |
| `hHm`; `lay`, `rs` | intended to prove | `HmRowComputes`, and the layout's facts (in place of `vb`) |
| `tr` | intended to prove | the serving roots' rewinding, in place of `hExec`, which gives one at `δ_tree = 0` (`TreeRewind.ofNever`) |
| `hConst`, `hZero` | intended to prove | the constants' and the zero's columns |
| `hks`, `hS`, `hk`, `hRw`, `hM`, `hρ`, `hr`, `ht` | numeric condition | conditions on the bound's parameters, and the executable's strata matching `Law.stratified`'s |
| `qF`, `qS`, `qT` | numeric condition | conditions Lean doesn't check: `qF ≥ 2t′ + 2v`, `qS ≥ t′ + v`, `qT ≥ 2t′ + 2v` (`SHA512CRStrict`, above) |

Nothing else in it is assumed. The four `flock_e2e_*_hm96` forms have the same footprint, with `dp : DerivedPlaces` (intended
to prove: the derived column placements) in place of `pp`, and without `hA3`: the generic forms hold at every sampling
design, and the `_exec` forms at uniform byte tapes (`Binding/E2E.lean`). `flock_e2e_count_exec_hm96` takes `hks`
and `hS`; the other three don't.

**What it bounds.** The headline bounds faulty proof blocks of the pinned rows circuit; "computes the Definition" also
needs `RowsCert` per template and the Definition's Boolean semantics in Lean (The circuit and the certificate, below).
Once every Definition lowers to a Boolean circuit over AND, XOR and NOT, a generic
Boolean-semantics lemma replaces the per-template `RowsCert`.

**Its coins.** The statement is for live uniform OS coins drawn each round: the model's proof coins are uniform and
drawn after the prover's message, as the coin server's `os` coins are under A6. M0's non-ZK session draws them under
`FC_COINS=os` on both sides (`live/src/bin/flock-circuit.rs`, `seed_coins()`), with no seed and no PRF. Its identity
then names the mode, and `flock-verify verify --coins os` binds that identity (`Tags.withCoins`; the verifier's
`PROTOCOL.md` §7.1). The benches run M0 so by default (`verity/ml/flock/pod/70-class-sweep.sh`, `60-circuit.sh`,
`61-circuit-cell.sh`), and their records name the coins (`coins: live-os`). The binary's default still derives every coin
from one OS seed through a SHA-512 PRF (`config_with` sets `cfg.coin_seed` from `seed_coins()`; `live/src/coin_seed.rs`).
That rests on `prf/sha-512` and the seed commitment's hiding, and the headline takes neither. So the headline is cited
only for runs on per-round OS coins: a session whose identity names `live-os`, or a bench point with `coins: live-os`.
The identity and a bench point's `coins` say which mode ran. flock-verify replays the recorded coins, so that they were
the verifier's own OS draws, each drawn after the prover's message, is the record's custody (`PROTOCOL.md` §17.2) and
A6, which no check of the record establishes. The custody is `live-verifier`, named in every mode (the table above).

## Premises the lock lists by name

The Lean check lists by name the premises that guarantees take inside their statements, and each has its record
in `Security/lean-audit.json`. `assumption-notes/lock-listed-premises.md` gives each one's claim id, what it reduces to and
which guarantees take it.

## Other named assumptions

Not soundness's: the PoUW window's keyed draw takes A4 and A5, and `HmRowComputes` is a fact about the circuit, still to
prove. Completeness's is `JsonSpec` (`Claims.JsonSpec`, `Definitions/Claims/Json.lean`): `Json.parse` reads back what
the honest prover's printer (`printJ`, the bytes `Json.compress` prints) and `canon` print, which no proof can show, as
`parse` and `canon` are `partial`; only the completeness proofs read it (`Proofs/Flock/Claims/Complete/Json/`). Zero knowledge's two are `Hm96Hiding`, stated in the spec (`Claims.Hm96Hiding`, `Definitions/Claims/Hm96.lean`,
where the claims list it as a named assumption) and read in `Assumptions.lean` (`hm96Hiding_apply`), and
`PadNonvanishing`, in `Assumptions.lean`; both are in `docs/zk-proof-public.md`. `Hm96Hiding` is now proved for hm96 at a bound on its gap (`CROnly.hm96Hiding_gap`; for
hm96-sha512 over a uniform key, `CROnly.hm96_sha512_bad_keys`, What is pinned, below), and `PadNonvanishing` is proved at
M1's padding constant (`ZK.padNonvanishing40`: `Ŵ_cols(ω_q)` is 0 or 1, and `PAD_C = x` is neither). So zero knowledge's guarantees
take no premise on the key: they are stated at `hidingGap My saltHash`, the hiding gap of hm96-sha512's pinned key `K*`,
which nothing proves small. The detail, and what still takes `hash-derived-key` (`Assumptions.HashDerivedKey`):
`assumption-notes/hash-derived-key.md`.
Zero knowledge's coverage (closed by the refusal): `Composed.zk_session_view` holds at every coin vector the honest
prover doesn't refuse (`refused_coins` in `flock-circuit.rs`; `CoinedSetup.Admits`: `β ≠ 0`, the mask slot's rank
`128·k`, `ρ₀, ρ₁` independent over F128, and no zerocheck outer coin `1`; the verifier also checks `β` and `ρ`), and
there is no other kind of coin vector. At an outer coin `r_j = 1` (`inv 0 = 0` in the residual round) the honest pads
miss the inner proof's constraint by a witness-dependent amount that the view would show (`Definitions/Flock/ZK/Table.lean`'s inner
proof), so the prover refuses it: `zk_veil::Masking` stops as the coin opens, before anything that depends on it is
sent, reading only the coin. That costs completeness at most `(m − 13)·2^-128` per rep under uniform coins
(`ZkEveryCoin.outer_one_le`), and no statement charges it. `ZkEveryCoin.zk_session_view_all` holds at every coin vector,
refused ones included, with the coins uniform: at a refused coin the prover stops and the simulator outputs the honest
prefix on the dummy witness, within `2·|LeafPos|·2^-193` both ways (`ZK.Table.table_abort_hm96`: Lemma A at every coin
vector and T1 on every pre-final leaf), and elsewhere Lemma B.
Which verifiers: at each coin vector `zk_session_view` is honest-verifier zero knowledge for those fixed coins. It covers
a verifier that chooses its coins only because the coin server commits every coin at `Hello`, before the prover's first
message (`verity/ml/flock/live/src/coin_tree.rs`), and the prover checks each opening against that commitment before it
uses the coin; the binding is collision resistance of `x ↦ SHA-512(ν ‖ x)` (`keyed-cr`, watched in `lean-audit.json`).
No statement models that commitment yet, so this order of messages and the binding assumption are what the
malicious-verifier claim rests on beyond the theorems.
- A4 (`Assumptions.KeyedStreamsUniform`), which the PoUW window's keyed draw takes as `hA4`;
- A5 (`Assumptions.UniformSecret`), which the same draw takes as `hsec`;
- `Assumptions.HmRowComputes`, not cryptographic: the circuit's row hashing, which the rows' value binding
  (`Binding.HmRows.binding`) and `flock_e2e_*_hm96` take as `hHm`.

**A4. `verity.ml.randomness`'s keyed streams at distinct indices pass a test as independent uniform streams do, up to `η`**
(`prf/sha-512`), for the PoUW window's keyed draw.

- **Statement.** A secret of `S` uniform bytes (A5) gives the key `Randomness.derive`. For a test `E` of the key's
  streams at indices whose framings (`Randomness.frames`) are pairwise distinct, each read to `L i` bytes
  (`Randomness.stream`), the streams pass `E` with at most the probability that independent uniform streams do, plus
  `η`. `Randomness.lean` is the Lean spec of `verity.ml.randomness` v2 (on `Flock.Sha512.hash`): it reproduces
  `verity/ml/randomness/tests/vectors.json`, and Python matches its vectors
  (`verity/ml/flock/tests/test_randomness_spec.py`).
- **Per test, as `SHA512CRExpected` is per finder.** For every test the statement would be refutable, since `2^(8S)` sources give at
  most that many stream tuples. `keyedWindow_escape_le` takes A4 for one test per set `B` (`windowTest`: the window draw
  misses `B`), and `η` enters its bound.
- **What it assumes of SHA-512.** Its compression function `f`, on a 1024-bit message block and a 512-bit chaining
  value, is a pseudorandom function in two ways.
  - Keyed by its chaining value, it makes the cascade a PRF on prefix-free inputs (Bellare, Canetti and Krawczyk 1996).
    The inputs are prefix-free: `frames` is length-framed and the counter is 8 bytes, so length extension doesn't apply.
  - Keyed by the secret bytes in its first message block, it is a dual PRF, as the HMAC analyses use. The secret follows
    a fixed public tag: 28 bytes for `stream`, and 37 for `derive` with the source's frame header.
  - The secret's uniformity is not A4's but A5's.
- **η, stated.** η stays symbolic in Lean, and every record-level statement keeps `+ η` visible, never absorbed.
  - **The claimed value:** for `S = 32`, η ≤ 2⁻¹²⁸ for the chain's tests: generic key search over `min(256, 8S)` bits,
    plus a hybrid over the test's `q` compression calls. The key is SHA-512 truncated to 32 bytes, which costs nothing
    at `S = 32`. The derivation, and what truncation costs: `assumption-notes/a4-prf-sha512.md`.
  - **One η for the whole family:** every set `B` (the closures of the bad sets, which `closure_escape` evaluates), every
    stream budget `L`, and every receipt context. The audit takes a sup over `B`, not a union, and `E_B` costs the same
    for every `B`, every `L` and every context.
  - **Vacuity:** A4 is plausible only for `S ≥ 32`, and η can never be below about 2⁻⁸ˢ, since keyed probabilities are
    multiples of that. It holds trivially at η ≥ 1, and is claimed only for the chain's efficient tests `E_B`.
- **Upstream:** watch entry `A4-prf-sha512`.

**A5. The PoUW window's secret is `S` uniform bytes, fresh for the window** (`uniform/python-secrets`).

- **Statement.** A source is a finite probability space and, on each outcome, the secret. `UniformSecret sec` says every
  event about the secret has the probability it has for `S` uniform bytes. `keyedWindow` draws over the real secret
  source, and its theorems take A5 as `hsec`. It is satisfiable: the identity source satisfies it
  (`Law.uniformSecret_id`, `KeyedDraw.lean`).
- **What A5 assumes of the verifier.** Its window secret is `secrets.token_bytes(32)` from the operating system's CSPRNG,
  drawn once the window's receipt is complete and used for that window alone (`Ledger.open`). A
  secret reused across windows is outside it, since later windows would be committed after earlier draws were seen.
- **The platform: Linux, with CPython 3.12 or later.** Re-read this on any other platform.
  - `secrets.token_bytes` reads `os.urandom`, which is `getrandom(2)`, the kernel's CSPRNG. It blocks until that is
    initialized.
  - It keeps no user-space state, so a forked process doesn't repeat its output.
  - A VM snapshot restored more than once must not repeat the kernel's CSPRNG state. That holds when the kernel reseeds
    on a VM generation change (vmgenid, Linux 5.18 and later, with hypervisor support); RDSEED, mixed in at reseeds, doesn't.
  - The CSPRNG is idealized as exactly uniform. The true bound carries its distinguishing advantage.
- **Split out of A4**: it is a property of what produces the secret, not of the hash.
- **Upstream:** watch entry `A5-uniform-secret`.

**`HmRowComputes`. The circuit's row hashing computes the `hm96-sha512/row/v1` or `hm96-sha512/row/v2` commit string**
(`circuit/hm96-sha512-row`; not cryptographic, and to be proved: verified-lowering design, phase 2i).

- **Statement.** At every witness the circuit's constraints accept, the `b ‖ c` a serving row's `hm96` slot outputs is
  `(SHA-512(row) ⊕ M·y) ‖ SHA-512(salt_prefix ‖ y)` for the row's bytes and the salt `y` that its `sha512x3` and `hm96`
  slots read. `HmRows.Computes` instantiates it at the drawn units' rows, with SHA-512 and the executable's key.
- **Bit rows (`sha512/row/v2`).** For a row of `ℓ` bits (`FlockLevel3.RowV2.enc`), it covers the constants the verifier
  derives from `ℓ` alone and Δ pins: the prefix `P(ℓ)`'s midstate as the first `cv` (in the row's first compression, or
  in its `hm96` slot when it has no compression before its last block); every message bit at or past `ℓ` (the last
  byte's zero fill bits and SHA-512's padding, in a block with row bytes or in a block of its own); and a unit's leaf bit
  past `ℓ`, read from its slot's forced-zero row. That these constants are SHA-512's padding of `P(ℓ)` and the row's
  bytes is part of what it assumes. `HmRows.v2` takes `enc_inj` from `FlockLevel3.RowV2.enc_inj`.
- **What it gives, and `δ_tree`:** `assumption-notes/hm-row-computes.md`.
- **Discharged only by fixed readers.** `HmRowComputes` and `decode_row` count as discharged only for readers of the
  witness at fixed addresses, as `Layout.row`, `Layout.salt` (`rowAt`, `saltAt`) and `HmRows.bcAt` are; a reader that
  chooses its values discharges nothing.
- **Proved so far** for `sha512/row/v1` rows: `row_bc_setupH` (untyped, given `hv1` and `ho`), `keyedRowsOuts_computes`
  (e2e-layout's drawn units) and `row_bc_setupH_typed` (typed, no hypothesis); `assumption-notes/hm-row-computes.md`.
- **Upstream:** watch entry `hm-row-computes`.

A1, Reed–Solomon mutual correlated agreement up to the Johnson bound, is proved: `mcaError_le_bchks25`, from DKT26 in
ArkLib (`README.md` §1.6; watch entries `A1-johnson-mca` and `udr-mca`).

## Named hypotheses of the proof

- **The circuit and the certificate** (no hypothesis). The end-to-end theorems are about `C = Prog.circuit`, the circuit
  the statement's rows make, each unit its template's rows; "faulty" is measured there. The headline bounds faulty proof blocks
  of the pinned rows circuit; "computes the Definition" additionally needs `RowsCert` per template and the Definition's
  Boolean semantics in Lean. That is a separate theorem, `IsRowsUnit.computes_of_cert`
  (`Certificate.lean`): when the unit's rows are the rows `GateRows.rows G L` derives from a gate circuit
  `G` (`RowsCert`) and `G` computes a definition `D` (`∀ x, G.eval x = D x`, `G` being `D`'s lowering, which is in
  Python, not Lean), a unit that is not faulty, with its constant 1, carries on each committed output wire `D`'s output
  on its input wires. `RowsCert` is a fact about finite data; `∀ x, G.eval x = D x` is not, and nothing in Lean ties `D`
  to a Definition until its Boolean semantics is stated there. `Types.Dag.layout_sound` (its type's outputs, for every
  checked typed layout) and `GateRows.rows_sound` (for rows derived from any gate circuit and layout) prove
  that rows satisfied with the constant 1 carry the gate circuit's outputs, checked in the kernel. Once every Definition
  lowers to a Boolean circuit over AND, XOR and NOT, a generic Boolean-semantics lemma replaces the
  per-template `RowsCert`.
- **The statement's form** is no hypothesis: `verify` refuses forms the proof omits (PROTOCOL.md §16.15).
- **The link points' layout** is a hypothesis (`Statement.LinkLayout`). Each region's free bits must be distinct and
  read link coordinates below `PT_LOCAL`, and the block coordinates `PT_LOCAL …` must fit inside the point. The
  statements of PROTOCOL.md §16 satisfy it.
- **The placement** (`Placement`; `lowering_sound`, `README.md` §1.5):
  - Placed (`Placement`) means two things:
    - the block matrices' rows at the unit's computed columns read exactly its rows' columns;
    - the rows at its constant read the pin, as every slot's constant is wired to it (spec §16.1).
  - A unit's input columns on one gate also carry one value in every satisfying witness (`UnitPlace.aliased`): they
    copy one source's position, or forced-zero rows.
  - **Proved from the executable's accept step** for `hm96-sha512/row/v1` and `/row/v2` statements
    (`placement_of_setupH`, `unitPlace_of_setupH`; `README.md` §1.5). Open: which instance a drawn unit is (its block
    and VU), and `aliased` from the flat statement's copies, which `unitPlace_of_setupH` takes as a hypothesis
    (`Layout.Aliased`).

**The audit layer's named hypotheses** (`Prop`s taken as hypotheses; no `axiom`; the theorems are in
`Audit/README.md`):

| Hypothesis | Term | Meaning | To be discharged by |
|---|---|---|---|
| `Analysis.KnowledgeSound` | `ε_ks` | the session accepts while extraction of a drawn unit (picked before the session's first coin) fails | `table_knowledge_sound_joint` (`README.md` §1.3), after Jensen over the prover's states |
| `Analysis.LinkSound` | `δ_link` | extraction succeeds but differs from `X` on the unit's committed inputs and outputs | compiled layer: `flock_batched_linkSoundE` (`SHA512CRExpected`, `ValueBinding`) |
| `Partition.AnchorsSound` | `δ_in` | the session accepts while the committed inputs differ from the anchors' values | `registered_weights` (`README.md` §1.3); at the oracle layer, `InputsSound` (`oracle_anchorsSound`) |
| `Partition.AnchorsSound₂` | `δ_in` | two-stage audits: for every strategy, the session accepts while the committed coarse trace's inputs differ from the anchors' values | as `AnchorsSound`, for the two-stage game |
| `Partition.InputsSound` | `δ_in` | oracle layer: the session accepts inputs other than the anchors' with probability at most `δ_in` | **proved for Flock** (`flock_inputs_sound`, `2^-205`), given `InputsLoweringSound` |
| `Partition.InputsLoweringSound` | — | inputs other than the anchors' leave some session table (the always-proved input statement) unsatisfied | the input unit's lowering (level 3) |
| `Partition.SessionSound` | `ε` | oracle layer only: a session with a drawn faulty proof block accepts with probability at most `ε` | **proved for Flock** (`flock_session_sound`, `2^-205`), from `table_value_sound`, `execArith_correct` and the lowering |
| `Partition.LoweringSound` | — | a drawn faulty proof block lies in a session table whose statement no committed table satisfies | the lowering theorem (level 3, another lane) |
| `IsHarmBound` | `H` | every set of faulty proof blocks with acceptance probability at least `δ` has harm (a consumer's weight per faulty proof block) at most `H` | the value core's `Stratified.harm_bound` returns; that implementation is not proved |

Both per-unit hypotheses are averages over the whole audit, for every prover and every rule that picks a drawn unit
before the session's first coin, not bounds state by state.

| Named at the compiled layer | Meaning | Discharged by |
|---|---|---|
| `ExtractionAnalysis.LinkSound` (`analysisB`, `analysisC`, `analysisBE`) | `δ_link`: extraction succeeds, but every close satisfying candidate decodes, on the unit's committed wires, to values other than the committed trace's | for `analysisBE`: `flock_batched_linkSoundE`, under `SHA512CRExpected` for each commit string's finder and `ValueBinding` (named; for the serving rows it is `HmRows.binding`, from `HmRowComputes` and the layout's facts); composed in `flock_batched_count_linked`, `_drawn_linked` (`_placed`: from `UnitPlace`) |
| `Partition.LoweringSoundB`, `Partition.LoweringSoundC` | a message packing a witness that satisfies the statement of a drawn unit's table decodes, for that unit, to values on which it is correct | `loweringSoundB_of_place`, `loweringSoundC_of_place` from each drawn unit's `UnitPlace` (`lowering_sound`, `README.md` §1.5), given the statement builder's `Placement` and `aliased` (level 3). `unitPlace_of_setupH` gives it from `Stmt.setupH`, given the unit's instance in the audit circuit and `aliased` |

## Cryptographic assumptions and coins

- **SHA-512 for the Merkle openings** (spec §14, with SHA-512 leaves and nodes), under the two forms above: the
  compiled theorem needs only `SHA512CRStrict`, for every classical prover, and the audit's value layer adds
  `SHA512CRExpected`. DESIGN.md §1 explains why SHA-512 rather than SHA-256, and how the bound depends on the hash's
  width.
- **No assumption for the rounds.** The verifier keeps each round's bytes, then draws that round's coins, and the
  replay compares bytes. This replaces the SHA-256 round digests. Keeping the bytes costs about the proof size, 0.7 MB
  per rep.
- **Statement binding** (the statement digest, Σ and `link_sha256`). These are in round 0's kept bytes, fixed before any
  coin. Their own collision resistance matters to what the statement refers to (registered data, linked proofs), not to
  this theorem.
- **`os` coins** (the default). Each round's coins are uniform and unknown to the prover before it sent the round. They
  come from the verifier's `/dev/urandom` (spec §7.1). The randomness is trusted, not proved. For the unit draw it is
  A3. The theorems, the headline among them, are for these coins.
- **`seed` coins** (`coin-commit/sha256/v1`; M0's non-ZK session's default, `cfg.coin_seed`, `coin-commit/sha512`)
  add two assumptions: the seed commitment hides the seed, and SHA-256 in counter mode is a PRF. No theorem takes them,
  so a run on seed coins is outside the headline (The headline, its coins).
- **`tree` coins** (`--zk`, `coin-tree/hm96-sha512/v2`): the `--zk` headlines' coins. Soundness pays `N·2^-256` for
  the commitment's statistical hiding (A6, tree coins); zero knowledge uses its binding (`cr/sha-512`).
- **The random-oracle model is not used.** Its version of the compiled bound is only a cross-check (DESIGN.md §2).
- **Quantum provers** (DESIGN.md §9). The oracle layer holds against them, but the compiled theorem is classical. The
  compiled protocol is post-quantum sound only asymptotically, and only if SHA-512 is collapsing, which is a conjecture
  (Table 1 would need `collapsing/sha-512`). No concrete post-quantum bound is known at these parameters, and none is
  claimed.
- **Table 1** cites Flock's soundness under the `live-verifier` model and the collision resistance of each row's hashes
  (`cr/sha-512` in the C-Flock SHA-512 row); A1's proof took `rs-proximity-johnson` off. An audit's integrity profile
  would add `SHA512CRExpected` as `ecr/sha-512`, a `verity.core.claims` id that no row cites yet.

## What is pinned

`lean-audit.json` records each pinned theorem's statement (`pins`) and the definitions it reads (`reads`). The audit
fails when either changes until the record is rewritten. A rewritten record is shown when it lands on `main`; nobody
signs off and nothing waits on it. The
pins include the table theorems (`README.md` §1.1 and §1.2, with their `_exec` forms), `execArith_correct` and
`Merkle.opening_binding`; none takes a named assumption. So do the knowledge-soundness theorems of `README.md` §1.3
(`table_knowledge_sound`, `table_knowledge_sound_joint`, `table_knowledge_sound_joint_tight`, `session_knowledge_sound`,
and the batched session's `flock_batched_knowledgeSoundE`), and the link theorem, `flock_batched_linkSoundE`, whose bound is
`linkBoundCR`: finite exactly at the provers whose link finders satisfy `SHA512CRExpected`. The end-to-end skeleton's `flock_e2e_count` and
`flock_e2e_drawn` are pinned too; they take `SHA512CRExpected` for `σ`'s link finders as `hCR`. So are the four
`flock_e2e_*_hm96`, which also take `SHA512CRStrict` for the knowledge term's finders (`hKS`) and `δ_tree`'s (`hT`).
So are the headline, `Prog.flock_headline` (its footprint is The headline, above), and the theorems the footprints cite
as proved: the `StrictCR` terms (`table_sound_compiled_strict`, `ksBoundAccB_le_strict`, `ksAvgBE_le_strict`,
`tree_le_strict`, `TreeRewind.le`, `TreeRewind.ofNever_bound`), `Teeth`'s four (`strict_of_injective`,
`expected_of_injective`, `not_strict_const`, `not_expected_const`), `Prog.flock_e2e_count` and `Prog.flock_e2e_drawn`,
and the certificate's `IsRowsUnit.computes_of_cert`.
The `cr/sha-512` restatements are pinned beside them (`FlockSoundness.CROnly`: `expected_of_strict`, `linkCR_of_strict`,
`linkBoundE_at_cube`, each `flock_e2e_*`, `uprog_e2e_*`, `prog_e2e_*` and `flock_headline`), and so is zero knowledge's
`Hm96Hiding`, proved: `hm96Hiding_gap` (no hypothesis, at `δ₁` the hiding gap), the leftover hash lemma
(`sum_hidingGap_le`, `hankel_universal`), hm96-sha512's values (`hm96_sha512_sum`: mean `2^-257` over a uniform key;
`hm96_sha512_bad_keys`: `Hm96Hiding` at `2^-193` fails for at most `2^-64` of the keys), and Lemma B and Lemma C with it
discharged (`session_shvzk_gap`, `session_shvzk_le`, `adaptive_prefinal_gap`, `adaptive_prefinal_le`,
`adaptive_prefinal_key`, `adaptive_prefinal_hm96_sha512`). What zero knowledge's guarantees leave open about the key is
that the pinned key's gap is small: they are stated at it, and `hash-derived-key` (`hm96/PROTOCOL.md` §2a) bounds it by
`2^-193` only where a lemma or `RecursiveAudit` states that bound.
Zero knowledge's Theorem GK is pinned (`ZK/GK/`): `Model.gk_simulate` with its coin-tree forms
`Model.gk_simulate_coins` and `Model.gk_simulate_coins_keyed`, `gk_simulate_hm96` and `gk_simulate_avg`, the run bounds
`Model.sim_runs_le`, `Model.sim_runs_ev` and `Model.sim_runs_tail`, the two distances `prefinalClose_hm96` and
`rewindClose_hm96`, and T6's extraction (`coinColl_isColl`, `coinCollKeyed_isColl`, `bindColl_isColl`,
`bindCollKeyed_isColl`, and its cost `coinPairs_length_le`).
`flock_inputs_sound`, `flock_session_sound`, `lowering_sound` and `mcaError_le_bchks25` are not
pinned. A theorem that is not in `pins` is still audited (standard axioms, kernel replay), but a change to its statement
would not fail the audit.

## Upstream status

Whether a dependency proves what these theorems assume is computed from `lean-audit.json`'s `upstream` watch list: `assumption-notes/upstream.md`.

## Trusted components

Not assumptions of the theorem, though the verdict depends on them: `assumption-notes/trusted-components.md`.
