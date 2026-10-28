# Proof of useful work

Certifies that a declared workload of integer or FP8 matmuls was computed with at least a stated fraction of the honest
work, by sampled recomputation of committed checked values. The setting and the cost model (W1 on an RTX 4090) are the
PoUW problem statement's; the guarantee is Lean's `Pouw.SecurityProofs.EndToEnd` instantiated with a scheme's
conjecture. PoUW's Lean is under `Security/` (`Definitions/Pouw`, `Definitions/Pouw`, `Proofs/Pouw`), and every theorem this document
cites is pinned in `Security/lean-audit.json`.

This document describes the scheme while it is experimental; what is proved is the Lean. Each guarantee G is stated in
`Definitions/Pouw` and proved in `Proofs/Pouw`, as `Pouw.SecurityProofs.G : Pouw.Guarantees.G`, and this document points at
those names rather than restating them.

## The interface

A scheme (`protocol.PoUWScheme`) is a named, versioned object, looked up by name (`schemes.get`), with pinned vectors under
`tests/vectors`, like `verity.ml.commitments.scheme.CommitmentScheme`. It turns one declared matmul's committed operand
rows into, for each tile, checked values, a transcript leaf and the useful output. It exposes:

- `name` and `certificate`: what an accepted audit rests on (the conjecture's claim id, γ_0, Ω*, the Lean guarantees,
  the hypotheses, the allowances);
- `check(matmul)`: refuses a declaration outside the domain, before any work;
- `tiles(shape)`: the checked units, a partition of the output cells; a tile is rows of A times rows of B at full k;
- `wref(shape)`: the honest reference's W1 cost, from the layout only;
- `noise(ctx)`: the matmul's hash-derived values, from the salt, the declaration and the committed roots;
- `encode(ctx, noise, a_rows, b_rows, tile)`: the checked product's operands for a tile, from its strips, or nothing when a
  row is malformed;
- `admissible(ctx, ops)`: the input policy (worst-case-input schemes accept every well-formed tile);
- `checked(ctx, ops)`, `digest(ctx, noise, checked)`, `useful(ctx, noise, ops, checked)`.

Operands are committed as rows along k: A is m × k and the weight B is committed as its n rows, so the product is A·Bᵀ.
Everything else is the protocol's, the same for every scheme.

## Lifecycle (`audit`)

Steps 1 to 4 are `audit`'s: what the served path's prover commits and its verifier reads. Step 5 is the served window's
draw. Steps 6 and 7, run in the clear, are a reference tool and not a verifier (`benchmarks/pouw/clear/clear_audit.py`):
the served window proves its drawn tiles in zero knowledge, and its verifier of record is the Lean (`lean/check`).

1. **Weights.** Each weight's rows are committed in a `verity.ml.commitments.merkle` tree bound to the weight id (owner −2);
   `weights_root` binds every weight root.
2. **Salt.** The challenger opens the epoch's salt coin after the verifier recorded `weights_root` ("Epoch coins"), or
   in a live window after the window's open root (below); the salt is
   `derive(coin, "verity/pouw/epoch-salt/v1", {"epoch": e, "coins": coins_root, "weights": weights_root})`,
   32 bytes (`Epoch.start`, `salt_of`). The weights' root is the same in every window of a model, so a salt coin opened
   after it binds no window, and a rule built on it would let one challenger session serve every window. So a live
   window's salt coin opens after its open root, `salt.window_open_root(weights_root, requests_root, S)` =
   SHA-512(b"verity/pouw-window-open/v1" ‖ weights_root ‖ requests_root ‖ SHA-512(S)), every part 64 bytes: the
   weights' root, the root of the window's round-0 request table, committed before the salt, and the SHA-512 of S's
   ASCII, S the id the auditor's gate assigns when it reserves the window's session (1 to 64 of `[A-Za-z0-9._:-]`,
   `salt.SESSION`), which the prover can't relabel. The gate opens the session on that root (`open --session S --root
   R`), and its challenger opens the salt coin only after R (`serve.py --root`). `core_check.salt` (Lean
   `PouwCheck.saltCheck`) takes the salt's coins and roots only through the window's challenger session, as
   `core_check.rounds` does ("Live epochs"), never from the prover: S from the session's signed reserve entry in the
   auditor's ledger, R from its signed open entry, and the coin commitment and the roots the challenger received from
   the session's signed record and log. It refuses a session opened without a reserve (`salt`), an R that isn't
   `window_open_root` of the window's record and S (`salt`: another window's weights or requests, R under another
   session's id, or H1's salt opened after the weights' root), a log of the session that doesn't show R alone as what
   its salt coin followed (`salt`), and a salt coin that isn't the epoch's `pouw/salt` coin 0 opened after R (`coin`).
   Each session has its own R, and R fixes one window's weights and requests, so one session salts one window. Not
   yet: vllm's served window doesn't call `salt`. An epoch without coins
   (`Epoch.unverified`, benchmarks and replays of recorded runs) keeps v1,
   `derive(seed, "verity/pouw/salt/v1", {"weights": weights_root})` over a label it is handed. A certified epoch
   (`Epoch.certified`, `salt.py`) is one PoUW window p of the network certifier's, its certifier windows p·k … p·k + k − 1
   (k the certifier's `pouw_window`; any other run of windows is refused), registered under a stream (its 64-byte
   digest) and a prover (its 64-byte registered id), and its salt is `pouw_salt(reveal, p, stream, prover)`, 64 bytes,
   from the certifier's 512-bit salt for p. The verifier reads that salt with `salt.revealed`: the certifier's
   `reveal(p, registration)`, which fails closed unless the proof service registered p (PoUW's reader, `salt.registered`,
   reads p off the served window's context, `<prefix>/pouw-window/<p>/<manifest SHA-512>`) and p's last ingress record is
   published. `pouw_salt` is SHA-512(b"verity/pouw-salt/v1" ‖ reveal ‖ u64be(p) ‖ stream ‖ prover), every part
   fixed-width, the tag keeping its input apart from `window_salt`'s: two registrations of p under another stream or
   prover are salted apart. That keeps a tile computed once to one of them only when the drawn unit reads the salt,
   and the served-zk window's units read none (below). p counts the certifier's PoUW windows from its origin
   (`Proxy(origin=…)`), each with a salt drawn once from fresh randomness, so no epoch is bound; a proxy started again,
   or a second link, counts from 0 again, so a served-zk window's stream names the certifier's run:
   `served_commit.stream_id` reads its `certifier_run`, (key id, run id, ingress link), beside the deployment and the
   stream's name. Every signed answer of the certifier binds that run, a `Proxy` and its `RemoteCertifier` give it
   alike, the run id is drawn when a proxy starts (`Proxy.run_id`) and the ingress link has no default
   (`Proxy(ingress_link=…)`). That salt binds no weights, and
   the prover holds it from p's first ingress tick, so a certified window passes only weights, a stream and a prover
   recorded before p's salt was drawn: the prover hands the certifier `pouw_record` of them, SHA-512(b"verity/pouw-record/v1"
   ‖ weights_root ‖ stream ‖ prover), one 64-byte record for p, signed by the key the certifier registers for it
   (`Proxy(prover_key=…)`, whose id is `salt.prover_id`, over `weights_message` under the stream), only while p's first
   ingress tick (the draw) is ahead (`Proxy.commit_weights`), and the verifier reads it with `salt.committed`
   (`Proxy.committed`, on `reveal`'s conditions). A window with no record made before its draw, or under another stream
   or prover, is refused, and the challenger checks that first (`core_check.recorded_reason`), asking for the reveal
   only after. So one proxy records one stream and prover for p. The certifier reveals p's salt once, and
   only until its reveal deadline on its own clock after p's end (`Proxy(reveal_deadline=…)`, one PoUW window by
   default), so one registration of p, made after p's work, gets it. `salt.WindowOpens` is a once-only ledger of
   opens: p opens once per stream and prover, when the challenger receives it, before its draw and any proof (its
   tiles are private, so that is the finest public key), and a second open under them is refused
   (`reregistered`). It is not credit: a window's credit is its outcome's verdict, and no period record yet closes a
   stream's credit at its first refused window.
   - **No served-zk certified window is credited today.** A certified window credits only units that read its salt
     (`salt.unit_reason`: a `salt` parameter that is the whole 64-byte salt as a coin port, as the compiled Program's
     units take it, **Seeds and noise** below). The served-zk window's unit, `Pc8TileSide`, reads two committed fp8-pi
     side rows and no salt, so rows formed under any salt, or before any, would pass in every window and under every
     stream. Its challenger refuses every certified window first (`unsalted`), before the stream receives it and before
     the certifier is asked for anything. A certified window credits once its units read its salt and their proofs
     are held to it. The opening of record (`core_check.opening`, Lean `PouwCheck.opening`) does the second: in a
     certified window it opens only a salted template's unit (`TILES`' `Salted`: today the H1 tile unit, `PouwH1Tile`
     at K = 64, R = 4, C = 8), only under the 64-byte salt the verifier holds (`core_check.certified_salt`, the salt
     `pass_epoch` checked), and only when the proof's public row, the port after its R A rows and C B rows, is
     `public_bc` of the window's context, that salt, the unit's position and its entry's side roots (`public_row`:
     salt ‖ u64be(position) ‖ root_a ‖ root_b, bit for bit the leaves `Compiled.inputs` fills for them). It refuses an
     unsalted template's unit in a certified window whatever salt it is passed, a salted template's unit outside one,
     and every unit of a certified window it is passed no salt for. The served challenger passes it the salt its
     pass was checked against (`served_commit.Challenger.check`, from `core_check.certified_salt`). The compiled
     Program is no pinned template, so no window of it opens.
   - **The reveal deadline** (`Proxy(reveal_deadline=…)`, one PoUW window by default) is a parameter: p's registration
     takes the reveal from p's last ingress record until that long past p's end on the certifier's clock (by the end of
     p + 1 at the default), which bounds how long the prover holds p's salt before its work is registered under it. It must be sized against a measured commit pipeline: a deadline shorter than the honest
     pipeline's time past p's end refuses every honest window, and the served H1 window spends 439 s on receipts.
     No Lean statement covers the certified window's timing yet (`EndToEndWindow`'s t_d is the
     uncertified window's).
   - **The certifier–challenger link**. The certifier runs on the link and the challenger on an
     auditor host, so the challenger asks through the certifier's client (`certifier.active.RemoteCertifier`, served by
     `serve_reveals`): each `committed` and `reveal` is a request the challenger signs for the certifier's run and an
     answer the certifier signs over its run, p, the registration's digest and the value, which the client verifies
     under the certifier's key. A `Proxy` in process has no signer, so it names no run, and a certified challenger
     refuses it. The certifier's key and run id are the window's `Deployment` (`served_commit.Deployment`), beside its
     challenger's key, and the challenger refuses a client for any other. Open: the deployment is to come from the
     auditor's deployment record, which nothing writes yet, so the challenger's caller passes it.
3. **Units.** For each matmul (a 64-bit index, a shape, a weight id), the activation rows are committed bound to the salt
   and the index. The scheme derives its noise from (salt, matmul, roots); every tile yields checked values and a digest.
   The useful output is assembled from the tiles.
4. **Transcript.** A tree over every tile's digest, in matmul order then tile order, bound to (salt, weights_root).
   Its root must reach the challenger within the deadline of the salt coin's release: the verifier's rule at the
   deadline in `Security/Definitions/Pouw/Game/Deadline.lean`, which `WorkBetweenSaltAndDeadline` reads. The served
   window states it as `Check.registration_deadline`, which `served_commit.py params` writes; the challenger refuses
   parameters without it, and refuses as `late-registration` a window received after the epoch's draw coin was released
   (each root's receipt comes before the coin's release). The deadline is in seconds on the
   challenger's clock, and the proof service refuses a later registration, or one before the salt, as
   `late-registration`. That ends the epoch, and the window earns no credit (`served_zk`'s challenger,
   `registration_deadline` in its parameters).
5. **Selection.** A served window's draw is one uniform subset of K′ of its N tiles, without replacement, drawn after
   its registration's receipt (`benchmarks/pouw/served_zk/served_commit.py`, `Challenger.select`). A certified window's
   draw coin is a live epoch's `pouw/draw` coin 0 (`window.live_challenger`), under the commitment the challenger reads
   itself from the record its challenger signed (`challenger.trusted_commitment` of `Challenger(coin_record=…,
   deployment=…)`), under the challenger key the window's deployment pins (`Deployment.challenger_key`; `Deployment.pinned`
   reads it from infra's published key record and refuses a key whose SHA-512 isn't the one stated in code). The draw's
   check of record (`core_check.draw`, Lean `drawCheck`) takes that record and the key it names, `{key, record}`, and
   the pinned key (`key`): it refuses a record whose key isn't the pinned one, no pin, or seeded coins under a pin
   (`coin-key`), so a record the prover signs over coins it holds is refused; it verifies the record's LMS signature
   under the pinned key and reads the commitment from it, so a bare commitment is refused: one trusted root per coin.
   - **N and K′.** A window of N_real real tiles registers the level `window.bucket(N_real)`: it is credited g_ℓ tiles,
     the largest grid point at most N_real, and pads with filler tiles to N = g_(ℓ+1) = `window.pad(N_real + 1)`, at
     least one filler. Every tile is the one template's and does w_max = `tile_work`, and filler tiles are drawn like
     any other but credited nothing. `window.sampling` registers the law `{"law": "subset", "k": K′}`, the registration's
     `population` being N, with K′ = `window.k_prime(ε, δ, W, N, w_max)` and W = g_ℓ·w_max the credited work: the
     smallest K >= 1 with (1 − ε·W/(N·w_max))^K <= δ, compared exactly in rationals (`window.miss_at_most`), N if no
     K <= N has it, and 1 when ε·W/(N·w_max) = 1. A wrong set holding a share ε of W has at least ε·W/w_max tiles, so
     the draw misses it with probability at most δ.
   - **The draw.** The challenger opens the epoch's draw coin once, after the registration (its digest), and keys the
     draw `derive(coin, "verity/pouw/served-zk-draw/v1", {"stream", "registration", "receipt"})` (`draw_key`,
     `core_check.draw_key_context`). The units are `core_check.stream_subset(K′, N, ·)`, Lean's `Draw.subset`
     (`flock-verify draw --population N --k K′ --stream`), over the 2·K′·width(N) + 64 bytes of the key's stream
     `"draw"` (`draw_bytes`, width(N) = ⌈(bitlen(N) + 64)/8⌉ bytes): a sparse partial Fisher–Yates whose step i swaps
     position i with i + r, r uniform below N − i by rejection over big-endian width(N)-byte words, and the units are
     the first K′ positions, ascending. So every K′-subset of [0, N) is equally likely. A stream that runs out refuses
     (`sampler`), and nothing redraws. `core_check.draw` holds the units to the coin's opening.
   - **Sessions.** The units are proved in ⌈K′/cap⌉ sessions (`window.sessions`) by ascending rank: session σ's instance
     s is drawn[σ·cap + s], and the filler tile N − 1 after the last (`window.instance_ts`, "The served audit window").
   - `Sampled(t)` (t independent work-weighted draws, Lean `WorkWeightedSampling`) and `Lottery(target)` (Pearl's mining
     rule: the tiles whose digest, read little-endian, is at most target·|rows|·|cols|·k) are the reference tool's
     (`clear_audit`), unwired: no served path selects with either.
6. **Check.** For each selected tile the prover opens A's rows, B's rows and the leaf. The verifier checks the openings,
   recomputes the tile through the scheme (`encode`, `admissible`, `checked`, `digest`) and compares the leaf. It refuses
   a tile any of whose opened rows fails the scheme's rules (`encode`'s parse, then `admissible`); Lean's
   `TileCheck.Accepts` asks the same of the tile's weight rows (the protocol's `weightOK`), and a good tile
   (`TileGood`) has every one passing.
The row and tile trees are framed SHA-256 unless the scheme names another frame hash (`commitment_hash`): `"hm96-sha512"`
(`audit.HM96`, Pearl-C's -h3) is `frame-v3-sha512` with hm96-sha512 leaves, in its section. The BLAKE3 tree modes `frame-b3`
and `frame-b3s` that -h1, -h2 and BLAKE3 -h3 named were deleted with the archive in `2091252fe3` (one hash, SHA-512).

7. **Profile.** When every selected tile passes, `WorkProfile` reports what the draw established, ε at δ, and for a scheme
   with a γ_0 the work certified at that ε, under the certificate's conjecture. PoUW has no policy parameter: accounting
   decides which ε at δ it accepts. The statement is
   `Pouw.Guarantees.EndToEnd` (proof `Pouw.SecurityProofs.EndToEnd`); a scheme with no γ_0 gets only its sampling half,
   `Pouw.Guarantees.WorkWeightedSampling`.

**What is public.** Every protocol is zero-knowledge, and only the minimum is public. |W|, the size of the weights, is
public. A served window's rows serve at a public fixed length, so its useful work is public ("The fixed length" below).
`AGENTS.md` ("Design rules") states them.

**Randomness.** The protocol owns the outer coins, the salt and the draw, and both are the challenger's epoch coins, never
a public beacon and never Fiat-Shamir over the prover's roots. A window's check item says its coin mode, its optional
`coins`, `live` or `seeded` (`window.mode`, Compose's `CoinMode`; an item that names none is seeded). In a seeded window
(`coins.SCHEME`, Pearl-C's served path) a scheme's noise is its own derivation from the salt and the prover's roots,
Fiat-Shamir inside the scheme, whose grinding is inside the conjecture's error term. In a live window
(`window.LIVE_SCHEME`, "Live epochs" below) round r's noise is keyed by round r's coin, opened only after the verifier
recorded round r's root.

## Epoch coins (`coins`)

The challenger is the auditor's service that issues coins. Before epoch e it draws one 32-byte coin per label (`"salt"`,
`"draw"`, in that order) from its entropy, and publishes the coin commitment, a frame-v3-sha512 tree over hiding leaves.
A served challenger's entropy, every coin and hm96 salt, is the Lean verifier of record's read of the operating system
(`verifier.coins`: `pouw-check-difftest`'s `coins`, `IO.getRandomBytes`, A3); `coins.Challenger` and
`window.live_challenger` take it as an argument and have no default, so the Python never reads the operating system for a
coin:

~~~text
x_l     = SHA-512("verity/pouw/epoch-coin/v1" || 0x00 || coin_l)
value_l = hm96-sha512 tree_leaf(key, commit_string(key, x_l, salt_l))     salt_l: 192 fresh bytes, key the pinned default
root    = MerkleTree over (value_salt, value_draw) at positions 0, 1, binding tagged("verity/pouw/epoch-coins/v1", e),
          owner -1, schema "verity/pouw/epoch-coins/v1"
~~~

`tagged(tag, parts…)` is SHA-512(tag ‖ 0x00 ‖ frame(part)…), all 64 bytes (`protocol.tagged`, `frame` being core's
`verity.ml.randomness.frame`); every binding and root the protocol hashes itself is one.

An unopened coin is statistically hidden (`hm96/PROTOCOL.md`); binding rests on SHA-512's collision resistance. A coin is
opened live and once: the salt coin after the verifier recorded `weights_root` (a live window's after its open root,
"Live epochs" below), the draw coin after it recorded what the
draw keys (the transcript's root; Pearl-C's work sampling design adds its records' root, `clear_work.draw_after`). An
opening is (epoch, label, after, coin, hm96 salt, path); the verifier refuses one whose epoch or label is not the one it
asked for, that is malformed, that does not open against the epoch's root, or whose `after` is not what it recorded
(`coin_reason`). A second request for an opened coin is refused, so a retry never redraws. The coins are the designated
verifier's: an outcome convinces the auditor, who trusts its own challenger, and no third party (claim `live-verifier`).
Vectors: `tests/vectors/epoch_coins_v1.json`.

**Live coins** come from the proof service's one challenger (`verity.core.protocols.sampled_proofs.challenger`, Lean of
record `SampledProofs.Challenger`), whose coins are indexed by label and index and named by no protocol, so every client
uses the same form: a layout is a scheme, an epoch and its families `(label, count)` in order, coin `(label, i)` at its
family's offset plus i, at most 2^20 coins. Before the epoch the challenger draws every coin, 64 bytes, SHA-512's whole
width (the seeded coins above stay 32 bytes, so a recorded seeded registration reads as it did), and publishes

~~~text
x_p     = SHA-512(scheme || 0x00 || coin_p)                               coin_p: 64 fresh bytes (`COIN_BYTES`)
value_p = hm96-sha512 tree_leaf(key, commit_string(key, x_p, salt_p))     salt_p: 192 fresh bytes, key the pinned default
root    = frame-v3-sha512 tree over value_0 … value_(K−1) at positions 0 … K−1, leaf schema the scheme, domain binding
          identity_digest_sha512("verity/challenger/coins/v1", {epoch, families, scheme}), owner -1
~~~

and opens coin `(label, i)` once, after the root the verifier recorded for it. `live_reason` checks one opening (its epoch,
label and index, its form, the path against the root, and `after`); `order_reason` checks a family's openings against the
roots the verifier recorded, in order: coin i opened after root i, none twice ("a retry never redraws"), none before its
root is recorded, and every recorded root's coin opened. `LiveRecord` is the verifier's side: it records a root and only
then asks the challenger for that root's coin. The commitment those checks read is the challenger's own: before any coin
opens, the challenger signs it with its LMS key on its host (`LiveChallenger.sign_record`, `write_record`; Lean
`SampledProofs.ChallengerRecord`), and the verifier reads that record from the challenger's file under its own key
(`trusted_commitment`), never from the prover's bundle, and fails closed on a missing, malformed or unsigned record.

Timing is a trust assumption, not a check. `after` binds an opening to the root the verifier recorded, but can't show the
coin was released after it: that rests on the challenger releasing coin i only once it has received root i. The
challenger is auditor-trusted, the auditor's device on a machine the prover doesn't control, under the named assumption
`trusted-host`. The record keeps the order on its own clock: `LiveRecord.log` logs root i's receipt
(`received_at`) and then coin i's release after it (`released_at`).

**Live epochs** (`window.live_challenger`, `rounds_reason`, `round_key`; Lean `PouwWindow.Rounds`). A live PoUW epoch's
coins are the layout `live_families(R)` under `LIVE_SCHEME` (`"verity/pouw/epoch-coins/live/v1"`): its salt coin
(`"pouw/salt"`), its draw coin (`"pouw/draw"`) and R round coins (`"pouw/round"`, rounds numbered from 0 as Compose's
`Declared.round` numbers them), R the most rounds a window of the epoch takes, fixed when the challenger commits. The
salt coin opens once, after the window's open root (`salt.window_open_root`, step 2), and `core_check.salt` (Lean
`PouwCheck.saltCheck`) is the rule on it, read through the same session as the round coins. A Pearl-C pass's session
takes its salt coin alone (`SALT_FAMILIES`, `salt_epoch_reason`; `Properties.Pouw.LiveSalt`), opened after its salt
epoch's root (`salt.epoch_open_root` of the weights' root and the epoch's number k, the pass's `pass_index`, which
the verifier recomputes and never reads from the pass), so it ends complete once that coin is released. Round r's coin
opens after
the verifier recorded round r's root, the root of the values round r commits, and round r's noise key is
`derive(coin_r, "verity/pouw/round-noise/v1", {coins: root, epoch: e, root: root_r, round: r})`: the deferred noised
products, keyed by a coin the prover could not see when it committed. `core_check.rounds` (Lean
`PouwCheck.roundsCheck`) is the rule on a window's round coins. It reads the challenger's coins only through the session
the auditor's signed ledger opens and ends closed for the window under the window's pinned challenger key, run from a
commit the host's pins log records and the verifier's own copy of main's first-parent history holds
(`challenger.trusted_session_of` over `challenger.pinned_session`, Lean `SampledProofs.Challenger.trustedSessionOf`
over `pinnedSession`): the record, on the host the ledger opened the session on, and the close signed under that key,
the release log the close signs and the ledger's end sealed, and the round roots that log shows the challenger
received, in order. Coins a prover's own in-process challenger drew, an unsigned or tampered log, a ledger signed under
another key or cut before the session's end, a second session for the window, and a session the host ran from a tree
its pins log doesn't record, or from a branch commit merged into main but not on main's first-parent line, are
refused; `core_check.salt` reads its session through the same `pinned_session`. A seeded epoch's commitment is unchanged (`SCHEME`, salt
and draw only).

**Pearl-C periods** (`credit.pearl_c_period_credit`). A live Pearl-C pass's salt is the coin of a salt-only
session (`window.SALT_FAMILIES`) the gate opens on its salt epoch's root `salt.epoch_open_root(W, k)`
(`core_check.live_salt`). Under that layout an abandoned session ends complete, like a used one, so a pass earns only
inside a Pearl-C period, and the period earns only when all of these hold: its close's sessions are exactly the presented
passes' epochs, in its order (`credit.period`), so a complete session no pass presents refuses it; each session is a
Pearl-C salt session (``salt-only``; H1's checks refuse such a session, so the two never share a period); each session is
credited toward the period (``circuit``, `challenger.credited_session`); each session is pinned within the period's
freshness floor of its open pin (``pin``, H1's `credit.committed_pin` and `credit.committed_floor`); each pass's context
is a string and the pass passes `core_check.pass_epoch` (``pass``); each W's epochs are numbered on from the lane's earlier credited periods with no gap or repeat (``index``),
so a W whose numbering broke is spent; and no two passes of one epoch hold one call index (`credit.passes_distinct`).
Its Lean port is open.

## Registered weights (`registration`)

One rule for every construction that serves a checkpoint: Pearl-C4 (`PearlC4.check_registration`) and served FP8's Pearl-C
(`PearlC.check_registration`) both call `check_registration`. No Lean statement reads a registration.

- **The keyed 8-block rotation (rung 3) is the rule**, for Pearl-C4 and for served FP8. An unrotated checkpoint
  registers only from the curated list, the fallback.
- **The keyed V/O rotation plus the keyed head interleave are adopted inside it, applied after it.** The V/O rotation
  acts on each KV head's v_proj rows and o_proj columns, and the interleave on o_proj's heads. They are never applied on
  an unrotated checkpoint, curated or not.
- **The keyed latent rotation is adopted after either.** Each MLA latent takes a keyed
  8-block rotation of its own, built as the residual's is (its S unmixed, calibrated blocks): the q latent on q_a's
  rows, with q_a_layernorm folded into q_b's and the DSA indexer's wq_b's columns; the kv latent on kv_a's first
  kv_lora_rank rows (the RoPE key's rows are untouched), with kv_a_layernorm folded into kv_b's columns, and so into the
  W_UK_T and W_UV the absorbed attention derives from kv_b. It is never applied on an unrotated checkpoint.
- A registration declares its transforms in order, and only four stacks pass: the rotation alone, or the rotation, V/O
  and the interleave, each optionally followed by the latent rotation. V/O without the interleave, the interleave
  without V/O, and any other order are refused, because each was measured only so, after the rotation.
- **A checkpoint that needs γ folded registers only with it**, as Llama-3.1-70B does under Pearl-C4: the cast moves its
  layer 0's update by 158% with the RMSNorm scales unfolded, and by under 4% with them folded. The rotation folds γ, so
  its stacks pass as they are. An unrotated, curated checkpoint declares `gamma-fold` (every RMSNorm scale folded into
  the linears it feeds), and takes no V/O or interleave.
- **A linear registers only where the stack's rotations reach it** (`check_linear`). The 8-block rotation mixes the
  residual stream: a linear that reads it takes the rotation on its columns, one that writes it on its rows. MLA's q_b
  and kv_b (with the absorbed W_UK_T and W_UV) and the DSA indexer's wq_b read a latent behind its own RMSNorm, so only
  the latent rotation reaches them: under a stack without it they keep the checkpoint's weights, are not registered and
  earn no credit, though they are still served and bound. A linear off every rotated stream is refused under every
  rotated stack. On GLM-5.3 at TP8, Pearl-C admits q_b and wq_b, so the latent rotation credits both again; kv_b
  (k = 512) stays outside Pearl-C's domain (`benchmarks/pouw/glm53_rotation.py`). A curated unrotated checkpoint
  registers each of its linears.
- **The keyed transforms are keyed after the weights' root is recorded**, by `rotation_key`:
  `derive(salt, "verity/pouw/rotation-key/v1", {"weights": weights_root}).stream("key", 32)`, over the salt of an epoch
  started from the challenger's salt coin (`Epoch.start`). It refuses any other salt: an `Epoch.unverified` one, a
  beacon's or a fixed seed's. The rule reads what a registration declares; applying the transforms is the
  registration's own.
- **Served FP8 applies no rotation yet.** The vLLM option's executor (`pouw.SERVED_TRANSFORMS`) applies none and reads no
  curated list, so its commitment's `registration` (`pouw.weights_claim`) says it claims nothing about which weights were
  served, and names the refusal. Its salt is still the live coin, so what the draw establishes of the work is unchanged.

## The proof's identifier (v1) (`identifier`)

Where a drawn tile is checked by a C-Flock proof rather than recomputed, the proof is bound to the oracle, the salt and the
workload it is for by its `identifier`, a JSON object in the proof's header:

~~~text
identifier  {"format": "verity/pouw/identifier/v1", "scheme": name, "salt": hex(salt), "workload": hex(SHA-512(canonical(W)))}
W           {"matmuls": [{"index": "i", "m": m, "k": k, "n": n, "weight": "w"}, …], "weights_root": hex(weights_root)}
~~~

- **The fields.** `scheme` is the scheme's registered name (`schemes.get`), which names the oracle H, so there is no seed
  digest. A name `schemes.get` doesn't register has no identifier. `salt` is the epoch's 32 bytes, or a certified epoch's 64. Hex is lowercase.
- **The declared workload** W (`workload_json`) is every matmul in declaration order, then the weights root, which binds
  every weight's row root (Lifecycle, step 1). The index and the weight id are 64-bit, so they are decimal strings: RFC 8785
  writes a number as an IEEE 754 double, which is exact only up to 2^53. m, k and n are numbers.
- **The canonical form** is RFC 8785 (`verity.ml.circuits.codec.canonical_json`) on its domain, ASCII strings and integers of
  magnitude at most 2^53 − 1: members in key order, no whitespace, and no escapes, since no string here needs one. A
  dimension past 2^53 − 1 is refused; no scheme's domain holds one. The identifier's bytes are this form of the object
  (`identifier_bytes`), and `workload` hashes this form of W (`workload_bytes`).
- **What binds it.** The proof binds its header's object through `pub.sha`, which seeds C-Flock's Fiat-Shamir transcript:
  Σ = SHA-512(sigmaTag ‖ c.sha ‖ digest ‖ pub.sha), under `cr/sha-512`.
- **The verifier's rule.** The verifier never takes the prover's identifier. It recomputes its own from the scheme and the
  epoch it audits, and hands those bytes to C-Flock's verifier (`--identifier F`). C-Flock's verifier compares them with the
  proof's header byte for byte, and reports the object in its verdict without reading a field. So a proof that names
  another scheme, salt or workload is refused.
- **Vectors:** `tests/vectors/identifier_v1.json` (`tests/identifier_vectors.py`) holds one case or more per named scheme:
  its inputs, W, both canonical byte strings as hex, and their SHA-512. A 64-bit case pins the decimal strings past 2^53.

## The served audit window (`window`)

A served window is audited by drawing tiles of one fixed template from the window's committed calls (the grid, the work
level and K′ are in `window`'s docstring). This section is the window's layout for gates: what its commitment holds, so
that a drawn tile's reads are a function of public inputs alone. `benchmarks/pouw/served_zk/served_commit.py` commits a
window this way.

**The fixed length.** Correct tokens served past the honest run's end-of-sequence are not useful work, so a workload of record
serves at a public fixed length, ignore_eos at each slot's public max_tokens, which the window registers (its `length`
item) and the verifier checks against its own parameters, so W_use is public. PouwCheck refuses
a workload whose stop mode is not `fixed` (`window-form: eos-keyed-row`), a served length or max_tokens other than the
verifier's (`window-form: length-mismatch`, `window-form: max-tokens`), and a window registered before the rule, with no
`length` item (`window-form: no-fixed-length`), which is not of record (`window.length_problem`). A served H1 window's
credit holds it the same way (`credit._length`, `PouwCheck.Credit.Internal.length`): its period's open, which the auditor
signs before window 0's first coin (`--length`), commits each request's public max_tokens and stop `fixed`
(`credit.committed_length`), and each step's served rows m_s, which its manifest rows fix under `calls_root` before the
draw coin (`credit.served_rows`), are that length's schedule, every request admitted in round 0: at least one row a request in round 0, and in round s ≥ 1 one
decode row for each request whose max_tokens is past s. A round serving more ran a request past its public max_tokens and
one serving fewer stopped a request at an EOS-keyed done source (`length-mismatch`); a period with no fixed length
credits nothing (`no-fixed-length`). Each round's live proof (`PouwLiveSlots_v2`, credit's `_live`) proves m_s, and its
public port `cap` binds each request's hidden cap to its public max_tokens: credit opens it at the known `<prefix>caps`
of the period's committed max_tokens, and the unit refuses a live request whose header's cap is another or whose round is
not 0, and a leaf live without a public cap. So a request takes a decode row in round s only when s is below its own
public max_tokens, and no prover can move a decode row from one request to another while keeping each round's count.

**The window root** (port `entries`) is a frame-v3-sha512 tree of P entries (the public bound, 16 for served windows),
registered before the draw. The calls come first, then exactly one filler, then empty entries up to P. Entry e is an
`hm96-sha512/row/v2` row at index e: the leaf is hm96-sha512's tree leaf of `commit_string(key, x, salt_e)`, where x is
the entry row's `sha512/row/v2` digest over its 2,688 bits, salt_e is 192 fresh bytes and key is the pinned default
(`Entry`, `entry_bc`, `entry_leaf`).

**An entry row** is 336 bytes (168 u16 words, 21 128-bit words). Each integer is a u64, little-endian, and each digest
starts on a 128-bit word. The call's place in its pass (tile0, cols, m) follows root_h3, so the fields before it keep their
offsets and only the record moves, and the -h3 root and what a unit computes from it sit together:

~~~text
bytes     0..8    kind      0 none, 1 call, 2 filler
          8..16   lo        the range's first tile index, < 2^40
         16..24   hi        one past its last, < 2^40
         24..32   n_a       A's rows: 1..2^D_A for a call or the filler, 0 for none
         32..40   n_b       B's rows: 1..2^D_B, 0 for none
         40..48   zero      so the digests start on a 128-bit word
         48..112  root_a    A's side root (64 bytes)
        112..176  root_b    B's side root
        176..240  root_h3   a call's pass's -h3 tile-tree root; zero for the filler and for none
        240..248  tile0     the call's first tile in its pass; zero for the filler and for none
        248..256  cols      the call's tile columns ⌈n/64⌉: 1..2^(D_B − 6), 64·cols >= n_b; zero for the filler and none
        256..264  m         the call's rows: a multiple of 16 in n_a..2^D_A, tile0 + ⌈m/64⌉·cols <= 2^40; zero likewise
        264..272  zero      so the record starts on a 128-bit word
        272..336  record    a call's record: SHA-512 of its RFC 8785 canonical JSON; zero for the filler and for none
~~~

An empty entry is the zero row. A call's record is the committer's own description of the call: its `shape` [m, k, n],
its `tile0`, and `keep`, for each side the places in the call of the rows the side tree holds, strictly increasing. The
entry binds the record only by its SHA-512, so no tile reads it; tile0, cols and m are the record's, in the row, so a tile
reads them.

**A side tree** is dense: it holds only the rows its call keeps (a row the scheme refused is no tile's), in order. So the
call's n_a·n_b tiles are the products of A's rows and B's, row-major, and index u of the range [lo, hi) is A row
u div n_b and B row u mod n_b (`tile_in`). The filler's sides are zero rows (index field included): n_b = min(2^D_B, max(⌈√fill⌉, ⌈fill/2^D_A⌉))
and n_a = ⌈fill/n_b⌉. Its range covers N − N_real ≤ n_a·n_b tiles. Every side tree has 2^D leaves, with D_A = 13 (the
served batch is at most 8,192 tokens) and D_B = 17. Leaf i < n is row i's frame-v3 leaf under the row's own schema. Every
leaf from n on is frame-v3's pad leaf of the side domain at its rank: frame-v3 pads any tree to a power of two this way,
so a side root is `tree_levels` over the n row leaves followed by 2^D − n pad leaves (`side_levels`, `side_path`).

**A side row** is the call's operand row, then its place in the call (`side_row`): I for an A row (< 2^13), J for a B row
(< 2^17), a u32 in one 128-bit word. The operand words keep their offsets, and the row's schema is its role's, as before:

~~~text
words     0..k/2          the k E4M3 codes, two a word, low first
          k/2..k/2+64     P's 64 BF16 words
          k/2+64..+66     the index I or J, a u32, low word first
          k/2+66..+72     zero
~~~

That is k/2 + 72 words, 2,120 at k = 4,096 (33,920 bits). A's rows are `hm96-sha512/row-seg/v1` (role X), B's
`hm96-sha512/row/v2` (role W).

**The side domains** are one per window and side, with no entry index and no row count:

~~~text
side_domain(w, s) = frame-v3-sha512 domain, binding identity_digest_sha512("verity/pouw/window/side-domain/v1",
                    {"window": hex(w), "side": s, "depth": D_s}), owner −1, positions [0, 2^D_s)
~~~

Here w is the window root's domain id, which is one-stage's served domain of the window's program, partition and
context. So the side domains, like the window's, are a function of the context and ℓ alone.

**The -h3 tile tree** is the pass's own tile commitment, the one its manifest publishes (`commitment`: root and count),
not a tree the window's committer builds. Under `pearl-c-sm120-v1-h3` (SHA-512 since `b91ebf369`) it is one tree per
pass, over every call's tiles:

~~~text
domain    hm96-sha512 (frame-v3-sha512, hm96-sha512 leaves over 64-byte digests), binding
          `audit.transcript_binding(epoch)` (its salt and weights' root), owner −1, positions [0, count)
          (`audit.transcript_domain`): one per pass, not per window or call
depth     D_H3 = ⌈log2 count⌉, count the pass's tile count, which the domain id binds
leaf p    the pass's tile p, in workload order then each call's 64 x 64 tiles row-major (`audit.tile_refs`); its value
          x ‖ y, x the tile's -h3 leaf (`pearl_c.leaf(·, "h3")`: `sha512/row-seg/v1` over the tile's messages, per tile row
          and 32-column chunk le32 C̃ ‖ le32 U, 256 bytes each) and y its 192-byte salt (the prover's ChaCha20 stream,
          nonce 2^63, blocks 3p to 3p + 2); the frame leaf is frame-v3's at rank and position p, under
          `hm96-sha512/row-seg/v1`, of hm96-sha512's tree leaf of commit_string(key, x, y)
~~~

So a window holds calls of one pass, each call's entry names that pass's root, and the window statement holds every
call's root_h3 to the root the pass published.

**The verdict asserts the served result.** A drawn tile's unit (TileRead, not yet a Program) reads three committed
rows: its A row and its B row through its entry's side roots, and its pass's -h3 leaf through root_h3. It compares, in
gates, the leaf's x with the -h3 leaf its own tile's words give, so the tile it proves is the tile the pass served. The
read is not at position t. Window tile t is one word, (I, J) = (keep_a[i], keep_b[j]) of its call for
(i, j) = `tile_in(entry, t)`. The unit reads I from its A row's index field and J from its B row's, and tile0, cols and m
from its entry row, and the pass's leaf holding the word is (`tile_read`)

~~~text
p     tile0 + (I div 64)·cols + (J div 64)
at    (I mod 64) | (J mod 64) << 6 | (rows/16 − 1) << 12
rows  min(64, m − 64·(I div 64)), the tile's height: 64, or the call's short last tile row (16 | m)
~~~

with I < m and J < 64·cols. The pass's tiles are in workload order, then each call's 64 x 64 tiles row-major, so this
is `audit.tile_refs`'s tile holding (I, J). Up to 4,096 window tiles read one leaf, and the leaf a unit computes is its
word's 64 x 64 tile's (the output of `Pc8TileLeafH3`, not on main yet, from the word and the tile's other messages as
advice), not `Pc8TileHidden`'s 1 x 1 digest. Until TileRead is a Program, a drawn tile's unit is `Pc8TileSide`
(`pc8.side_tile`, `served_zk/stage_tile.py`): `Pc8TileHidden`'s 1 x 1 check and digest over the drawn word's two side
rows, which it reads whole, index field included (`pc8.SIDE_PORTS`, row `fp8-pi`), so its committed inputs are the side
trees' leaves. It reads I and J as values and uses neither yet, and it reads no -h3 leaf.

The side rows' indices are bound to the record in the window's reference only: the window statement in gates reads the P
entry rows, not the up to 2^13 + 2^17 side rows under each entry or the record's preimage. `window_problem`, given the
records and the side rows' indices, holds each call's indices to its record's keep, strictly increasing, and its tile0,
cols and m to its shape, so i -> I and j -> J are injective and no two window tiles read one word.

**The window statement** (L4, `window_problem`) holds on the P entry rows and the pass's published tile root when all of
the following hold:

- each row is an entry row: 336 bytes, a known kind, and zero words at 40..48 and 264..272;
- each entry satisfies the layout (`layout_problem`): its range is 40-bit, an empty entry is the zero row, a call's or
  the filler's sides hold 1..2^D rows, the filler's root_h3, record, tile0, cols and m are zero, and a call's m, cols
  and pass tiles are in the table's bounds;
- the kinds are the calls (at least one), then one filler, then empty entries;
- the calls' and the filler's ranges are nonempty and contiguous from 0;
- a call's range is exactly its n_a·n_b tiles, and the filler's is at most its n_a·n_b;
- every call's root_h3 is the pass's tile root;
- the calls' pass tiles [tile0, tile0 + ⌈m/64⌉·cols) are in order and disjoint;
- the ranges end at N, and the registered level is `bucket` of the calls' tiles N_real.

**The tile instances' public inputs.** The K′ drawn tiles are proved in S = ⌈K′/cap⌉ sessions of `cap` instances each
(`sessions`). Session σ's instance s takes the tile t = drawn[σ·cap + s] while σ·cap + s < K′, and otherwise t = N − 1,
the window's last tile, which is always a filler tile (`instance_ts`). The verifier writes its own `--public-inputs` file
F_σ (C-Flock's `verity/flock-public-inputs/v1`, read by Lean's `HmIn.checkOwn`) from the one-stage registration (its
population N, its sampling design subset:K′, and its window root and domain at port `entries`), its own draw, σ, the cap
and the pass's tile domain (`public_inputs`). F_σ is a key-sorted JSON header line
`{"format", "instances": cap, "ports"}`, then each instance's words, u16 little-endian, instance-major, in port order:

~~~text
port       words   seed           derivation                               value
t          8       drawn          verity/pouw/window/t/v1                  t as a u128
root       32      registration   verity/pouw/window/root/v1               the window root
domain     32      registration   verity/pouw/window/domain/v1             its domain id w
domain_a   32      registration   verity/pouw/window/side-domain-a/v1      side_domain(w, "a")'s id
domain_b   32      registration   verity/pouw/window/side-domain-b/v1      side_domain(w, "b")'s id
domain_h3  32      pass           verity/pouw/window/domain-h3/v1          the pass's tile domain id (its D_H3)
~~~

Each port is a whole number of 128-bit words, so no row leaf shares a word with a public input; an instance is 336
bytes. The window statement is proved as its own direct statement per window, separate from the tiles' sessions.

**Sizing.** Served windows register levels 87 to 143 (N from 9,216 to 1,179,648). A C-Flock session is at most
m = k_log + max(3, ⌈log2 blocks⌉) ≤ 35, so a session of a unit of 2^k_log bits per instance holds at most 2^(35 − k_log)
instances (`session_cap`). The drawn tile's unit in the served runs, `Pc8TileLeafH3{K=4096, sm120}`, is 3,492,322 ANDs
at k_log 24; TileRead's three reads add to it.

~~~text
ε, δ               K′ over levels 87..143                  cap at k_log 24   S     cap at k_log 25   S
0.1%, 2^-40        N up to level 100, at most 31,178        2,048             16    1,024             31
1%, 2^-128         N at level 87, at most 9,937             2,048             5     1,024             10
~~~

The largest K′ is at level 143: 27,713 tiles unpadded, times g_(ℓ+1)/g_ℓ = 9/8 at levels whose point is 8·2^e.

**Vectors:** `tests/vectors/window.json` (`tests/window_vectors.py`) pins a toy window of P = 4 entries over a toy window
domain. Call 0 keeps A rows 0, 1 and 3 of four, so its refused row 2 is no leaf. Both calls name one toy pass's tile
root. The vectors hold each entry's JSON, its row, digest, salt, b ‖ c and leaf; one side tree's leaves, its pad leaf, its
root and a path; the window root; and F_σ, with a toy pass tile domain, for the draw (1, 4, 6) in sessions of 2, where
the second session pads with N − 1 = 8.

## `ncp-v1`: noise-cancelling polarization under TTNCP_U(0.5%), route U at β = 128

For int7 A and B (entries in [−64, 63], one two's-complement byte each):

~~~text
E₁ ∈ [−31, 32]^(m×k)   per unit; row i = low 6 bits − 31 of SHAKE256("verity/pouw/ncp/E1/v1" ‖ 0 ‖ frame(salt, index, root_A, weight, i))
F₁ ∈ [−31, 32]^(k×n)   per (epoch, weight); column j from SHAKE256("verity/pouw/ncp/F1/v1" ‖ 0 ‖ frame(salt, weight, j))
R                      signed permutation with R[π(j)][j] = σ_j:  E₂[i][j] = σ_j·E₁[i][π(j)],  F₂[π(l)][j] = −σ_l·F₁[l][j]
E₃ = E₁ + E₂,  F₃ = F₁ + F₂
X = [A + E₁ | A + E₂ | −(A + E₃)],   Y = [B + F₁ ; B + F₂ ; B + F₃],   X·Y = A·Bᵀ
β        128: X + β is u8 and Y is s8 on the int7 domain (Lean `Pouw.SecurityProofs.NCP.RangeU`,
         `Pouw.SecurityProofs.NCP.BiasU8Iff`)
c₀[j]    −β·Σ_{l<3k} Y[l][j]   (−3β·colsum(B) for an unsigned P), the accumulator's initial value
checked  C^U_t = c₀ + Σ_{l<16t} (X + β)[:, l]·Y[l, :] mod 2^32, t = 1 … 3k/16 (Lean `mmWU 128`); the last is A·Bᵀ
digest   SHAKE256("verity/pouw/ncp/tile/v1" ‖ 0 ‖ words)[:32], words little-endian, step-major, row-major within a step
tiles    16 × 16
domain   16 | m;  64 | k, n;  1,088 ≤ k ≤ 2^16  (γ_0 = 1/200 ≥ 16/(3k));  n₀ = 1,600 ≤ n ≤ 2^16  (Ω* = 1 + f/(3n₀) ≤ 1.005)
wref     3·m·k·n + f·m·k, f = 24   (route U under W1 with sub-word moves priced, Lean `wrefU 24`; the per-weight forming of
         B + F_i is left out)
γ        1 − (1 − γ_0)/Ω* ≈ 0.995%, the certificate's at n₀ ≤ n (`Pouw.Guarantees.NCP.GammaFromTTNCP_U_v1`), so the
         certificate records n₀ with β and f
~~~

- **Checked values.** Route U's accumulator, the u8 × s8 tensor-core schedule the W1 accounting prices. It equals the
  unbiased running sums shifted by public data, C^U_t = C_t − β·Σ_{l ≥ 16t} Y[l][j]
  (`Pouw.SecurityProofs.NCP.CheckedUEq`), so every bound proved modulo V₀ transfers. The unbiased sums
  (`checked_unbiased`) are retired as checked values; their honest program would cost 36 units per position (γ = 1.08%
  at n = 2,048).
- **β = 129** (f = 20, `Pouw.Guarantees.NCP.GammaFromTTNCP_U_C2`) needs
  an unsigned P and a SASS check that the IADD3 with two negated sources is one instruction. The native route-U kernel
  (`benchmarks/pouw/kernels/pouw_gemm.cu`, `ncp_route_u`) reads X + β already formed, so its SASS shows the u8 × s8 `IMMA` but not the
  forming step; `ncp-v1` stays at β = 128.
- **γ is conditional on TTNCP_U(0.5%).** A2, as first written, required the model program to
  reproduce every output, not only the checked words. Restricted to the checked words, A2 makes the Theorem D chain
  equivalent to TT_NCP rather than a reduction to milder assumptions (`Pouw.SecurityProofs.Dimension.A2WordsIff`). So
  TTNCP_U, TT_NCP over the route-U checked values, stays the named conjecture, and every γ here holds only under it.

- **P.** `ncp-v1` uses the word-granular production P v1.1: a SHAKE256 Fisher-Yates over the k/4 aligned words from
  seed `pous-pouw-ncp-Pword-v1`; repair 1 (seed + `/repair`) so that no word lands in its own 16-step; repair 2 (seed +
  `/repair2`): while some word lies on a cycle of the arcs w → p(p(w)) that stay inside w's step, swap p at the least such
  word with p at a drawn word, under repair 1's condition. The order inside each word is kept and every sign is +1. Its
  SHA-256 (entries as 2-byte little-endian integers) is `536757a2…` at k = 1,088, `7f00a9b7…` at 2,048, `24e68f30…` at
  4,096 and `a57ea054…` at 11,008, all reproduced by an independent generator, which certified independence rank
  T − 1, λ₃ = 32 and MinRank 16 at each. The repairs guarantee clause 1 of the rejection rule and the MinRank argument's
  hypotheses (a) to (c), not the rule's other clauses or λ₃ ≥ 16: the rule (Lean `Admissible`) is checked at every k, and
  `tests/test_pouw_word_p.py` certifies all of them exactly at the k the MVP uses (1,088, 2,048, 11,008).
  `ncp-v1-shift24` uses the shift by 24 instead, admissible with independent checkpoints for every k ≥ 1,024 with
  16 ∣ k (Lean `Pouw.SecurityProofs.NCP.ShiftBy24Admissible`, `Pouw.SecurityProofs.NCP.IndependentWitness`).
- **Lean.** The certificate's guarantees are its `theorems`: NCP's in `Security/Definitions/Pouw/Guarantees/NCP.lean`
  (the reduction `Pouw.Guarantees.NCP.GammaFromTTNCP_U_v1`, from the named assumption `Pouw.NCP.Assumptions.TTNCP_U`),
  the barrier `Pouw.Guarantees.Barrier.Generic`, and the game's `Pouw.Guarantees.Theorem1`, `WorkWeightedSampling` and
  `EndToEnd`.

  Assumed: TTNCP_U(0.5%); `Word.NzNCP` of the route-U words (the floors' hypothesis). Not proved: that M_4090 runs H_ref
  within `wref` (`AdmitsRef`), and the byte-level noise expansion here (the Lean takes any expansion of H's answers with
  the stated ranges).

## `pearl-fp8-v4`: Pearl's FP8 scheme (`schemes/pearl_kw.py`)

Pearl's certificate-v4 FP8 scheme, as published: the whitepaper "Pearl Floating Point Scheme Specification" (Sep 2026)
and pearl-research-labs/pearl at `284b147b`, whose verifier code rules where the two differ. The module's
docstring states every parameter.

- **Conformance.** `tests/vectors/pearl_kw.json` holds Pearl's own outputs, computed by its Rust at `284b147b` from
  inputs `tests/pearl_vectors.py` derives; the generator is `tests/pearl_vectors.rs`. For both of Pearl's devices, H100
  and B200, the reference reproduces:
  - the labelled hashes, the noise lines and the BF16 operations;
  - the device matmul and its partials;
  - the census (λ, the grid, the skip);
  - the lane layout;
  - on whole 16 × 16 tiles at k = 1,024: the FP10 norms, the noise, the quantized codes and the per-row scales, the
    accumulator, the extract, J, and the jackpot policy's verdict (one tile approved, one rejected for dead entries).
- **The RTX 4090.** Pearl's whitelist has no Ada device, so `ada` is ours: Verity's pinned `sm89.mma.m16n8k32.e4m3`
  atom (`verity.ml.circuits.silicon`), with H100's structure (a fresh accumulator every 128 products, promoted by one FP32 add), δ = 1
  and W = 13. With plain chained `mma.sync` instead, the 14-bit adder anchored at the running sum fails Pearl's
  unpredictable-summands census on honest inputs.
- **What the protocol replaces.** The chain headers become the salt. The activation and weight roots stand in for
  Pearl's keyed-BLAKE3 Merkle openings. The seed order is kept.
- **The conjecture.** Pearl's Assumption 1 with the jackpot policy. It names no γ_0, so the profile carries only the
  sampling half and the policy's allowances (5% skippable summands, 1/64 dead entries per side, the unchecked peel).
- **Its differences from milestone 1's D1**, the hardened KW transcript: see the module docstring.

- **Vectors.** `ncp-v1`'s checked values equal, bit for bit, Lean's own `#eval` of
  `mmWU 128 k A B E₁ F₁ R (blockEnd 16 (3k) t)` at every checkpoint, and the retired unbiased form equals Lean's
  `mmW (blockEnd 16 (3k) t) (stackX k A E₁ R) (stackY k B R F₁)`, for the word-granular P, the shift by 24 and a random
  signed permutation (`tests/vectors/ncp_lean.json`, from `tests/ncp_lean.py` on the integrated package, with the Lean
  sources' SHA-256). The scheme's own pinned vectors are `tests/vectors/ncp.json`.

## `ncp-v2`: NCP-INT as a circuit under sampled proofs (`verity/core/protocols/pouw/circuit`)

Sampled proofs is PoUW's verifier here: the call is a `verity.ml.circuits` Program, drawn units are proved, never recomputed, and
the audit's integrity profile carries the work bound. This package has NCP-INT at small shapes (16 | m, 16 | n,
64 | k); the certificate's domain is still `ncp-v1`'s.

**The units.** A weight's Y strips and c₀ read only B, the salt, the weight id and the column ids, so they are formed once
per run and weight: `NcpWeight{K,N,PERM}`'s root body is its `NcpFormY` units (`partition.weight_query`). One linear call
is `NcpLinear{M,K,N,PERM}`, which reads its weight's Y strips and c₀ as committed inputs. Each root body holds only
template instances, so each partition is `Q_template_instances` over them (`circuit.partition.query`); every value is
computed in exactly one unit. `NcpWindow{M,K,N,PERM,C}` is C calls on one weight in one root, the Y units once
(`partition.window_cut`): the composition passes the invariant too (`test_calls_on_one_weight_compose_without_recompute`).

~~~text
template            per                          reads                                         returns
NcpFormY{K,PERM}    16-column weight strip       16 weight rows, salt, weight id, column ids   the Y strip (s8), c₀
  (NcpWeight: once per run and weight)
NcpDigest{K}        16-row strip of A            its 16 A rows                                 D_s
NcpNode             node of the call's tree      its two children                              the parent (the root is D_A)
NcpKey              call                         D_A, salt, forward, position, weight id       K
NcpFormX{K,PERM}    16-row strip of A            16 A rows, K, row ids                         the X strip (X + β, u8)
NcpTile{K}          16 × 16 block                its X strip, its Y strip, c₀                  the credited words, and Z
NcpDequant          output entry                 Z, s_i, t_j, bias_j                           y = bf16(fp32(s_i)·fp32(t_j)·Z) + bias_j
~~~

**The formats.** Every hashed field is a 64-bit word: big-endian for SHA-512 (a digest word is its big-endian value) and
little-endian for SHAKE256. Tags are 8 ASCII bytes. `noise(b) = (b mod 64) − 31`.

~~~text
r_i = SHA-512("ncp2-row" ‖ row i's int7 bytes)            D_s = SHA-512("ncp2-stp" ‖ r_16s ‖ … ‖ r_16s+15)
D_A = RFC 6962 tree over D_1 … D_(m/16), node = SHA-512("ncp2-nod" ‖ left ‖ right); one leaf's root is the leaf
K   = SHAKE256(D_A ‖ "ncp2-key" ‖ salt ‖ forward ‖ position ‖ weight id)[:32],   position = layer << 32 | site
E₁ row i    = noise(SHAKE256(ι(i) ×5 ‖ "ncp2-E1\0" ‖ K)[:k])
F₁ column j = noise(SHAKE256(ι(j) ×5 ‖ "ncp2-F1\0" ‖ salt ‖ weight id)[:k]),        ι(i) = i << 32 | i
~~~

X, Y, c₀ and the checked values are `ncp-v1`'s (`schemes.ncp.assemble`, `checked_route_u`) on this noise: the tile returns
checkpoints 1 … 3k/16 − 1, step-major and row-major within a step, and Z, the last (A·Bᵀ exactly, mod 2^32). A and the
weight are committed a leaf per row, so a proved unit opens exactly the rows it reads.

**Why the formats look this way.** Each is forced by the partition, and each is checked by a test.
- **D_s is its own unit.** An IR node is one call, placed after its arguments, so one unit can't both write D_s and read
  D_A, which depends on it. The design note's X unit is two here: the digest unit and the forming unit. Both read the
  strip's rows, which are committed inputs, and they compute different functions.
- **K is per call, and each row's XOF begins with the row's own index in all five of Keccak's columns.** A row's XOF over
  (salt, forward, D_A, …, i) would compute round 1's values of the call-wide fields alone in every X unit: the same
  function of the same values in two units. The recompute check rejects that layout
  (`test_the_index_in_every_column_is_what_keeps_the_xof_recompute_free`). F₁ carries the column index the same way,
  and K's message begins with D_A's eight lanes, so that no two calls' key units compute a common value.
- **c₀ is the Y unit's.** Every tile of a column would otherwise compute −β·Σ Y[:, j] again.
- **The index binding is anchored inputs.** `Q_template_instances` admits no constant node at the root, so the forward
  index, the position, the weight id and the global row and column indices are Program inputs. Each has an anchor the
  verifier owns (`circuit.anchors`):

  ~~~text
  salt        the run's salt
  forward     the verifier's own enumeration of the window's forwards
  position    (layer, site) from the registered layout
  weight id   the registered id of the matrix whose rows B opens under the weights root
  row ids     the call's global row indices, distinct
  column ids  the weight's global column indices, shards included, distinct
  ~~~

  The verifier builds these inputs itself (`Traced.call_args`, `Traced.weight_args`) or refuses a prover's that differ
  (`Anchors.check_call`), and a window's `Ledger` refuses a repeated call index (forward, position), A11's
  `UniqueCallIndices`. `SaltWindows` refuses one across every window of a salt, whose forwards continue from its
  `next_forward`.
- **Every primitive returns at most 32 bits.** SHA-512 words are carried as (hi, lo) halves and Keccak lanes as
  (lo, hi).

**The verifier builds the Program.** `traced_as(scheme, m, k, n)` gives the call's and the weight's
Definitions from the scheme and the shape alone; the scheme pins P (`ncp-v2`: the word-granular P, as `ncp-v1`;
`ncp-v2-shift24`: the shift by 24). The verifier never takes a prover's descriptor: `Traced.check_descriptor` refuses one
whose digest isn't its own Program's. So the width rule is a property of that Definition, checked on it
(`partition.width_rule(shape, kind)` takes no Definition and no served count; a test's variant goes through
`width_rule_of`, which refuses an empty served set).

**The separator** (Z stays in the tile). The protocol pins the separator port: the tile's Z, the
second field of `NcpTile`'s return (`partition.SEPARATOR_PORTS`). The registrant can't choose it.
- `partition.width_rule` removes the pinned ports' values, then width-checks every unit from which a served output (y)
  is still reachable, under `cut.WIDTH_RULE`. Only the dequantization units are, and each returns 16 bits. Every other
  unit is behind the separator.
- A tile whose word leaf reached y would itself be reachable, and would fail, along with every PoUW unit upstream of it
  (`test_a_word_leaf_wired_to_the_served_output_fails_the_width_rule`).
- `partition.separator` checks the topology. Every path from a PoUW unit (digest, node, key, form-X, tile; and form-Y in a
  window) to y passes through a pinned Z. What crosses is m·n values of 32 bits, 8,192 bits per tile, and no credited word is
  delivered.

**The draw** (`plan`). The strata are the verifier's own, as `verify` derives them: one per
template of the call's partition, in the order of its first unit (`Flock/Draw.lean` `strataOf`). The sampling design
draws each stratum, and each drawn tile is proved with its closure: its X unit, strip digest, the node units on that
strip's path and the key, each once, and its Y unit in the weight's Program (`Plan.proved_y`), once per plan. Every
other template is drawn at its floor, so a wholly wrong one is caught directly (`audit_work_whole_stratum`). The
sampling design is a seam:
- `WorkLaw` is Verity's work sampling design, with its floors (`Flock/Draw.lean`): k_s = min(n_s, max(f_s,
  ⌈K·w_s·n_s/W⌉)), with K = 27,713, the draws that establish ε = 0.1% at δ = 2⁻⁴⁰, the default report point and not a
  policy parameter (`plan.record_k`; 27,712 misses). Only the tile template does
  PoUW work, so W is its stratum's, and k_tile = min(n_tile, max(f_tile, K)).
  - `plan.work_table` is the verifier's `--work-table`: every template of the call, `{"work": w, "floor": f}`, the tile's
    W_ref per tile and 0 for `digest`, `node`, `key`, `form_x` and `dequant`; `dequant`'s floor is y's (below), and the
    others' 1 unless POUS sets them. A call's work is in one stratum (0 < w_s·n_s in one only) and its tiles are of one
    template: `work_table`, `WorkLaw.budget` and `window_laws` refuse any other (`plan.check_work_strata`), since the
    window guarantee takes it as its hypothesis `hone`. Where the cap K_c ≤ N_c binds, a second work stratum is
    under-drawn: in one constructed window ten faulty proof blocks have acceptance probability 0.156 against a claimed 2.4·10⁻⁴
    (`test_the_red_team_s_capped_window_is_refused`).
  - `plan.law_object` is the sampling design `verify` derives from it, each stratum `{template, units, work, floor, k}`.
  - K is a **window's** budget: a call draws its share K_c = ⌈K·W_c/W_window⌉, capped at its units
    (`plan.window_laws`). That is the sampling design `verify` enforces when each call is its own statement with its own
    K_c (`--work K_c`); within one Program (`ncp2.window`) the calls share their templates' strata and one K.
  - **y's floors.** No tile's closure holds a dequantization unit, so y is
    drawn at its floor: each call's table gives `dequant` `{"work": 0, "floor": f_c}`, f_c = ⌈K_y·n_c/N⌉, n_c its y
    cells and N the window's, with K_y = K = 27,713 (`plan.window_laws`). So k_c = min(n_c, f_c) meets k_c·N ≥ K_y·n_c
    or proves the call's y, and a window draws at most K_y + C y cells. δ stays 2⁻⁴⁰: a tile-bad and a y-bad wrong set
    are one family of the one window audit, so its sampling term is the larger of (1 − ε)^K and (1 − ε)^K_y, not their
    sum. Per call the floor is a guarantee (`audit_work_floor`); `flock-verify draw --work` derives the same sampling
    design from these tables.
  - **The window guarantee covers the split across calls and y's floors**, `audit_window_split_of_record` in
    `FlockSoundness/Audit/Window.lean` (proved at `f06327bd`): the per-call sampling design with y's floors at
    K = K_y = 27,713 accepts a window with at least 0.1% of its work unsound or 0.1% of its y cells wrong with probability
    at most 2⁻⁴⁰ + ε_ks + ε_bind. Its `callK` and `windowK` are `WorkLaw.budget` and `WorkLaw.sizes`, recomputed from
    their formulas in `test_the_window_law_is_the_window_pin_s`; its hypotheses are `hone` (above), y's floors
    (`hfy`), and a window that does some work and has y cells (`hW`, `hN`: `window_laws` refuses W = 0 or N = 0,
    `plan.py:186`, `plan.py:191`). One Program (`ncp2.window`) is its case C = 1: K_c = min(N, K), the K `verify`
    takes (`1 <= k <= n`), and y's floor K_y. Outside it, the window's use of it rests on four things:
    - *The verifier derives every number itself.* `window_laws` (`plan.py:164`) takes only the verifier's own layouts
      (`Traced.layout`, `__init__.py:144`; any other is refused by `Traced.check_layout`, `__init__.py:136`). It
      computes W from them (`plan.py:185`) and N (`plan.py:190`), and the floors ⌈K_y·n_s/N⌉ (`plan.py:195`);
      `WorkLaw.budget` computes W_c and N_c from the call's own strata and K_c = min(N_c, max(1, ⌈K·W_c/W⌉))
      (`plan.py:153–157`). No prover-stated number enters.
    - *Call indices are distinct and the verifier's.* `Traced.draw` admits the call to the window's `Ledger`
      (`__init__.py:97`), which refuses a repeated (forward, position) before anything is drawn (`anchors.py:215–221`),
      and each stratum's context is ("call", forward, position, "stratum", template) (`plan.py:291`). The index is the
      verifier's anchors' (`Anchors`, `anchors.py:42`, which `Anchors.check_call` holds the call's inputs to,
      `anchors.py:83`). This is the guarantee's M1.
    - *Every call's commitment precedes the window key.* A window's `anchors.Receipt` lists every call with
      its anchors, shape and commitment, and is immutable (`anchors.py:130`); its digest binds each call's full
      anchors (index, salt, weight id, rows and columns), shape and commitment. `Ledger.open` derives the key from it
      alone, under `DRAW_DOMAIN` with {"receipt": digest} as its context, and admits only the calls it lists
      (`anchors.py:199`, `anchors.py:215`). It is the only place a ledger's receipt, key and source are set: a
      `Ledger` refuses every attribute assignment (`anchors.py:187`), and `Traced.draw` takes the key from the
      ledger, refusing one not opened from a receipt (`__init__.py:96`). A call's openings are checked against its
      commitment as the receipt lists it, by call index (`Ledger.commitment`, `Traced.check_openings`,
      `__init__.py:101`), never one supplied with them. So W, N, each K_c and every commitment the openings answer
      to are fixed by the receipt before any draw.
    - *Each window's source is fresh and used once.* `Ledger.open` takes no source: it draws 32 bytes from
      the operating system's CSPRNG (`secrets`, `anchors.py:165`) once the receipt is complete, for that window
      alone. So A4 (`prf/sha-512`) applies as stated, one fresh uniform source per derivation, with the source's
      uniformity its own claim, A5 (`uniform/python-secrets`). Once the window's draw is sent (`Ledger.sent`), it draws
      nothing more, and the audit record keeps its source and receipt digest (`Ledger.record`, `anchors.py:245`),
      from which anyone re-derives the key and the draw (`replay_key`). The limit: a third party can check that the
      draw came from the recorded source, not that the source was fresh. A verifier colluding with a prover could pick a
      source whose draw misses; that is the designated verifier's trust, and a beacon or a commit-and-reveal would be
      needed for a third party to check it. No protocol timing changes: the key is still the verifier's own
      randomness after the receipt.
    - *ε_ks and ε_bind are the window's.* The guarantee's are one window `Analysis`'s (M2). The per-call values are not
      cited as the window's; no statement composes them yet.
- `CountLaw` is `main`'s stratified rule, k_s = max(1, round(K·n_s/N)).

**The draw is per call.** Each stratum's subset is drawn from the verifier's key under the call's index,
(forward, position), with the template: the strata are the call's own, one per template, and the calls of a window drawn
under one key draw independently, so K split across them keeps the window's (1 − ε)^K. The verifier draws through
`Traced.draw(law, anchors, ledger)`, which takes no layout and no key: it draws over `Traced.layout()`. The
window's `anchors.Ledger` is opened from its complete `Receipt` (`Ledger.open`), which derives the window's key from a
fresh source under the receipt's digest; `Traced.draw` draws only under that key, admits the call first, refusing one
the receipt doesn't list or the window already holds before anything is drawn, and then keys each stratum's subset
with ("call", forward, position, "stratum", template). So every call's commitment precedes the key, and each call of a
window draws under a context of its own, the product sampling design the window guarantee takes as M1.

**The call's Y is linked to the weight Program.** The call reads its weight's Y strips and c₀ as committed
inputs, so at each drawn tile's column strip (the strips `Plan.proved_y` proves in the weight Program) the plan opens
the call's Y and c₀ there beside the weight Program's output there, a ("y", strip) leaf, and `Traced.check_link` refuses
any strip where they differ (`test_the_call_s_y_is_linked_to_the_weight_program`: another weight's Y, or one c₀ changed).
A wrong Y input on a strip no drawn tile reads spoils only undrawn tiles, which the sampling design charges like any
unsound tile.

**The layout is the verifier's.** The Layout theorems (`Layout.accountable_compute` and the rest) hold only for the
window's own layout, so the verifier derives each tile's own unit and its closure from the shape alone, through the
Program it builds (`Traced.layout`), and refuses any other (`Traced.check_layout`). `draw` refuses a layout whose tile is
not a tile unit or is not its closure's first unit: a tile mapped to its X strip is C1 again
(`test_a_prover_s_layout_mapping_a_tile_to_its_x_strip_is_refused`).

The profile is `Stratified` over the tile strata, and its `harm_bound`, at W_ref per tile, bounds the unsound work.

## `pearl-c-h100-v0` and `pearl-c-h100-v1`: Pearl-C, Pearl's product with its noisy clean-up checked (`schemes/pearl_c.py`)

The cheap-binding H100 candidate. It takes Pearl's H100 product and output tickets, and adds Pearl's noisy clean-up
pinned bit for bit, so that the useful output U is a checked value.
γ rests on the named conjecture TT_OUT(1/400), not on counting, so the certificate names no γ_0. The module's docstring
states every step. TT_OUT's device forms are stated at the W1 machine
(`Security/Definitions/Pouw/Assumptions/PearlC/TTOutW1.lean`): a strategy's cost is `W1.w1Priced`'s on the record's
device, so an attack on them is a W1 program, and the γ guarantees that read them conclude at that cost model.

~~~text
seeds     seed_B = H_"pearl-c/v0/seed-B"(salt ‖ root_B);  seed_A = H_"pearl-c/v0/seed-A"(salt ‖ root_A ‖ root_B ‖ index u64)
          -h3: every H_ℓ SHA-512 (`s512`); row i's own seed, H_"seed-A"(salt ‖ d_i ‖ root_B ‖ index·2^32 + i u64),
          d_i row i's SHA-512 digest
forming   v0, Pearl's as ported (`pearl_kw`): FP10 rows, exact norms, α, β, N = E·Fᵀ, X̃ = fp8(clamp(fma(α, X, β·N)))
          v1, the rework's: FP32 rows x; s = max|x|, ρ = RMS over every 8th position, α = 448/max(s + 4ρ, 2^−32), σ = ρ,
          β_row = (σ·α)·√r/(256·16), E′ = e4m3(β_row·E), X̃ = e4m3(HOPPER_E4M3_K32.step(C = α·x, E′_row, F line)),
          with F_A and F_B's lines at L2 norm 16 (E's at Pearl's 256)
chain     C̃ = the H100 replay of A′·B̃ᵀ: a +0 accumulator per 128 products chains four HOPPER_E4M3_K32 atoms,
          promoted by one FP32 nearest-even add (−0 → +0)
peel      P_A = [bf16(β_A·E_A) ‖ bf16(A′·F_B)],  P_B = [bf16((β_B·E_B)·G − B̃·F_A) ‖ −bf16(β_B·E_B)],  G = F_Bᵀ·F_A
          (v1: bf16(E′) in place of bf16(β·E));
          U = C̃ + P_A·P_Bᵀ as four HOPPER_BF16_WGMMA_K16 steps into C̃'s word
leaf      TurboSHAKE128 (D = 0x21) per tile row and 32-column chunk over its C̃ then U words, then TurboSHAKE128
          (D = 0x22) over those digests; -h3: `sha512/row-seg/v1` over the tile's messages
trees     A's rows, B's rows, the tile leaves: SHA-256 Merkle; -h3: `audit.HM96`
tickets   Pearl's lottery over C̃ per 16 × 16 subtile
useful    U/(α_A·α_B), glue
~~~

- **Pinned where Pearl isn't.** Pearl's verifier runs G and the B-side factor on a tolerance path.
  - Here G, A′·F_B and B̃·F_A are the same FP8 chain as C̃ (A′·F_B is the product's 32 fused columns).
  - (β_B·E_B)·G is a sequential FP32 FMA chain from +0.
  - The peel is the BF16 wgmma accumulating into C̃'s register.
- **Conformance.**
  - The chain equals Pearl's own Rust H100 replay at `284b147b`: the pinned matmuls, and both 16 × 16 H100 tiles
    after `pearl_kw`'s forming (`tests/vectors/pearl_kw.json`).
  - It equals `pearl_kw.replay` on random codes, partial windows included.
  - The scheme's own vectors are `tests/vectors/pearl_c.json`: for each forming, a 16 × 16 × 128 test instance and one
    64 × 1,024 × 64 tile.
- **Two formings.**
  - v0 is the port, not the design: it keeps Pearl's FP10 decode and exact norms, whose A-only work counts
    as skippable.
  - v1 is the rework, with its roundings pinned in the module docstring: the order of ρ's partial sums, α and β_row's
    FP32 steps, and the 2^−32 floor that keeps an all-zero row finite.
    - Only ρ depends on A alone. The model adopts s, ρ and α as its own FP8 quantizer, so W_ref counts 68 units per
      element of forming, 64 of them credited (`Costs.adopted`).
    - `wref` is Lean's `wrefOfRev1` and `credit_of` its `creditOfRev1` (rev1, `DeviceRev1.lean`; `wrefDevRev1` and
      `creditDevRev1` at another device's prices), pinned on a grid of shapes: `wrefOf` and `creditOf` less the first
      promotion add of every word (`first_adds`, m·n adds), which adds into +0 and so returns its group's word. The
      weight side (B's forming, B̃·F_A, G, P_B) is `job_cost`, once per weight and epoch, in neither.
    - Every W1 price is a device's (`Prices`, the device statement's): `H100_PRICES` for sm_90 (FADD 32, a BF16 MAC 2,
      forming 64 + 4) and `SM120_PRICES` for sm_120 (8, 2, 40 + 1: each step at its cheapest bit-exact form, as measured
      on the card). `wref`, `credit_of`, `job_cost` and the debit read the scheme's record; the registered schemes are the
      H100's, and sm_120's record goes with sm_120's atom.
    - Its domain is TT_OUT's: 128 ∣ k (whole promotion groups), 64 ∣ n with n ≥ 64 (`pearlCDomainDev`, the 64-column
      tile: a 1 × (n mod 64) edge tile's words can be guessed whole), and on every row of both operands the noise floor ρ·α ≥ 1
      (exact product) and liveness on the exact statistic, at most k/64 entries with x² ≥ 64ρ², where
      ρ² = Σ_{p ≡ 0 (8)} x_p² / ⌈k/8⌉ over the words' values (`live_row`).
    - It shares v0's seed labels (`pearl-c/v0/seed-*`).
- **Hashing format -h3** (`pearl-c-sm120-v1-h3`, `serving.HASH_FORMATS["h3"]`: tree hash `audit.HM96`, `seed_a="row"`,
  `row_cap=True`, `row_digest="sha512/row-seg/v1"`) is SHA-512 throughout, with E_A seeded per row. It is the one hash
  format besides h0 (one hash, SHA-512, as the proof service's row commitments), so
  the scheme assumes SHA-512's collision resistance (`cr/sha-512`). The BLAKE3 formats -h1, -h2 and BLAKE3 -h3, with
  their vectors and kernels, were deleted with the archive in `2091252fe3`.
  - Labels: H_ℓ(x) = `pearl_c.s512`(ℓ, x), the first 32 bytes of SHA-512(``pearl-c/h3s/<ℓ>\0`` zero-padded to 128 bytes ‖
    x), for ℓ in seed-A, seed-B, noise-line, line and jackpot; the label's ``h3s`` is the domain tag the format was first
    pinned under. seed_B = H_seed-B(salt ‖ root_B), seed_A = H_seed-A(salt ‖ root_A ‖ root_B ‖ u64 LE index), the roots
    64 bytes.
  - Row i of unit u draws its E_A line from its own seed, `row_seed`(salt, d_i, root_B, u, i) = H_seed-A over salt ‖ d_i ‖
    root_B ‖ u64 LE (u · 2^32 + i), 168 bytes, with u and i each below 2^32 and d_i = `pearl_c.row_digest`(row i),
    `rowleaf.sha512_row_seg_digest` over the row's FP32 words little-endian: 1 KiB segments hashed on their own, then
    their digests (a 4096-wide row is 16 segments, a 14336-wide one 56). `hash_h3.cuh`'s `s512_rows` and `s512_seeds`
    compute them; the pins are in the tests, and the kernels, built for the host, write those bytes.
  - Why per row: a seed that doesn't read A, such as the epoch coins and the index alone, tells the prover each row's
    noise before it chooses the row. It can then form every row onto a few shared A′ rows, compute their C̃ once per
    weight and copy it, while every row passes every rule (Pearl-C4's `noise` has the same shape, and every row folds
    onto one A′ row there; its -h3 is in its section). Keyed on its
    own digest, each candidate row costs an oracle query again. Forming waits on that row's digest alone, so A's tree may
    leave the main lane (`schedule_refusals`).
  - Lines (`pearl_c.sample_line`): Pearl's rule over 32 bytes H_line(H_noise-line(seed) ‖ side ‖ factor ‖ u32 LE line)
    in place of BLAKE3's keyed by Pearl's line key.
  - The tile leaf is `sha512_row_seg_digest` over the tile's messages themselves (le32 C̃ ‖ le32 U per row and 32-column
    chunk, row-major), 64 bytes; its 1 KiB segment digests take the message digests' place. The tickets are
    H_jackpot(seed_A ‖ fold), seed_A the unit's over root_A.
  - Trees: A's rows, B's rows and the tile leaves are `audit.HM96`, `frame-v3-sha512` with hm96-sha512 leaves
    (`audit.Hm96Domain`): x is a row's `sha512/row-seg/v1` digest (the same d_i) or the tile leaf, the salt the prover's
    ChaCha20 stream (`audit.hm96_salt`, nonce the unit index for A's tree, 2^62 + w for weight w's, 2^63 for the
    transcript's), and an opening's value carries its salt.
  - Where the protocol, not the scheme, hashes, it hashes with SHA-512 and keeps all 64 bytes: each domain's binding
    (`tagged`, which every frame of core's takes at 64 bytes) and the records' root. The salt and the draw are the epoch
    coins, and the draw's `derive` is core's, on SHA-512 (`prf/sha-512`).
  - The row cap: the debit cap is checked for every credited row, each row's debit at most ρ times its share of the
    credit, as well as per tile. It is (C2) of `TTOutRowSeed.lean`, under which `ttOutRowSeed_of_ttOut` reduces the
    row-seeded TT_OUT to rev1's per-unit one at error (q + R)/2^128 over the R declared rows. An aggregate cap doesn't
    transfer. The work sampling design checks it on every credited A row of a drawn tile, over its credited columns,
    after the tile's cap (`clear_work.row_cap`), and `volunteer` gives up a row over it first. The hidden cap checks it
    in gates (`circuit.cap`'s v2 templates, `cap.served`).
  - `pearl-c-sm120-v1-h3` is registered (the served default, `pouw_pearl_c_device`), and so is `pearl-c-h100-v1-h3`,
    `PearlC(hashing="h3")` on the H100 record (GLM-5.3's FP8 form on the H200). The `test-16-v1-h3` and
    `decode-32-v1-h3` vectors pin a call end to end, with A's row root and row 0's leaf, digest, seed and E_A line, and a
    hash of every row digest and seed. h0's verifier refuses an -h3 transcript on its activation openings.
- **Row tiles.** m is a multiple of 16, not of 64: the last row tile may be partial, in whole 16 × 16 ticket subtiles.
  Decode's 32 × 64 tile has a leaf over its 64 digests (2 KB), so decode is not padded to 64 rows.
- **The debit** (`schemes/pearl_c_debit.py`) is `Skip.lean`'s, replayed by the verifier on a drawn tile.
  - `word_flags` flags an atom when some set of its promotion group's atoms containing it, skipped jointly, leaves the
    word's output unchanged. The later promotions are tested by intervals found backward by bisection.
  - It equals P1's Lean-checked word vectors: all 552 when they were cut, and the 448 kept in
    `tests/vectors/pearl_c_skip_vectors*.txt.gz` under the 256 KiB blob limit (every word with k ≤ 512, and the
    aligned-spike words at k = 2,048 and 8,192).
  - A tile's chain debit is 40 units per flagged atom-word and, under rev1 (`tileDebitRev1`), an add (32 units) per
    later promotion into a +0 total, the g ≥ 1 whose total after g groups is ±0 (`word_debit`, rev1's
    `laterZeroPromos`), each summed over its words. The forming debit's per-tile rule is pending from P3's owner, so
    `forming_elements` is a stub that returns 0.
  - The census's aligned-spike tile at R = 300 passes the domain with 74.6% of its atom-words flagged, about 15 times the
    cap.
- **The work sampling design** is what the verifier credits, draws and checks. Its records (`Split`, `UnitRecord`,
  `records_root`) are `schemes/pearl_c_work.py`'s, which vLLM's prover writes; its law and clear audit are a reference
  tool, ``benchmarks/pouw/clear/clear_work.py``.
  - A weight's `Split` holds at most ⌊k/100⌋ side channels. The unit reads the kept channels zero-padded to `k_unit`,
    and credit is on the kept width, so padding earns nothing. The audit recomputes S by the pinned rule through
    `split_check`, which the model integration supplies.
  - Before the draws the prover declares a record per matmul: its real rows (the rest are zero filler), its kept width,
    and its excluded rows. `records_root` binds the records and the splits.
  - One exclusion rule, per row: a row that fails the per-row rules stays in the unit, bound as any other, and its cells
    earn no credit. Tiles with credited cells stay drawable, and the audit also opens one tile per excluded row.
    - A known difference from Lean: the work sampling design excludes a failing weight row and accepts its tiles, while
      Lean's good tile (`TileGood`) and correct unit (`Correct`) never exclude one, so `TileCheck.Accepts` refuses the
      tile as `check_tile` does. The work sampling design stays credit-sound there only while an excluded column earns
      nothing (`test_an_excluded_weight_row_earns_nothing_and_its_tiles_are_accepted`).
  - **Voluntary rows (V-EX):** the record may also give up up to ⌈m/64⌉ A rows
    (`voluntary_a`), and each weight up to ⌈n/64⌉ B rows (`voluntary_b`, the same in every unit of the weight).
    - They are committed in `records_root` beside the excluded rows, before the draw coin opens. A record without them
      encodes exactly as before.
    - A voluntary row earns no credit, and its words are still checked on every drawn tile that holds it.
    - It is exempt from the two-sided per-row check and R1, and it opens no exclusion tile.
    - The tile cap and R1 read the credited rows only. Past the count caps the audit rejects.
    - `volunteer` is an honest prover's remedy for a tile over the cap: it gives up the rows whose removal brings each
      tile under, before the draw coin opens.
    - A weight whose n is not whole column tiles is served zero-padded to the next whole tile (`pad_weight`; GLM-5.3's
      lm_head, 19,360 rows a TP8 rank, to 19,392), and the last tile's rows, its real columns with the zero rows, are
      voluntary (`declare`'s `padded_b`, though the zero rows fail the noise floor). That tile has no credited cell, so
      no draw lands on it and it opens no exclusion tile: no tile check sees a zero row. The credit is the full tiles'
      real columns (19,328 of 19,360) at the padded n's cell credit.
  - A cell's draw weight is its row's share of `W_ref` at the kept width, the `W_ref` of the Lean statement the draw
    cites (`draw_wref`). On sm_120 at its cap that is `wrefDevRev1K` at `Prices.sm120Loop` and the honest cast at 8.72
    (`SM120_LOOP_CAST_8P72`), the statement of the served γ and ε. Its ratio to `wref` differs between linears, so a
    served step's mixed layout drawn by `wref` would not be work-weighted in its sense. Elsewhere it
    is `wref`. A tile's draw weight is the sum over its credited cells. Where every cell is credited, that is the cited
    statement's ticket (the tile's `Wref`). The statements weigh every cell, so a tile with filler, excluded or
    voluntary rows gets less than its `Wref`'s ticket, and unless every tile's credited share is the same, the draw is
    not yet the one they state (Not here yet). An accepted unit is credited (1 − ρ)·`credit_of` over its credited
    cells.
  - **The sampling guarantee is per credited cell.** A drawn tile's uncredited cells (filler, excluded or voluntary)
    never count as coverage. Volunteering a tile's other rows lowers that tile's draw weight to its credited cells.
    Lean's statements don't model that yet: their draw weight is the whole tile's `Wref`, whatever is volunteered.
  - On an opened tile the verifier checks that filler rows are zero, that every row is zero past the kept width, that
    credited rows pass the per-row rules and excluded rows fail them, and that the replayed debit over the credited words
    is at most ρ times those cells' credit; under -h3, every credited A row's too. ρ is the device record's cap
    (`Device.cap`): 1/400 on the H100.
  - That credit is the cells' share of the unit's `credit_of` at the kept width, |rows|·|cols|/(m·n) (the per-tile
    statement's `tileShare`). A row's forming and A′·F_B are so charged once across its row band, a |cols|/n share in
    each tile. `credit_of` of the tile's own shape would charge them whole in every tile: at 8,192³ that cap is 2.17×
    the cells' share.
  - The draws are keyed by the epoch's draw coin, opened after the verifier recorded the commitment (the transcript root
    and the records root, `draw_after`), through `derive(coin, DRAW_DOMAIN, {"transcript", "records", "salt"})`
    (`draw_key`). A replay of a recorded run keys them as the run recorded them (`RecordedDraw`, `replay_key`).
- **The TurboSHAKE128 reference** is `verity.ml.crypto.turboshake`, checked against RFC 9861 and `hashlib`'s
  Keccak-f[1600].
- **U's exact order, CPU-pinned** (`schemes/pearl_c_u.py`), for the Lean model of the clean-up.
  - It states the chain with its promotion, the fused columns, the factors' single BF16 roundings, P_B's FMA chain and the
    four K16 steps, and both tensor-core steps as integer grouped sums.
  - It equals `pearl_c` bit for bit.
  - Its vectors (`tests/vectors/pearl_c_u.json`) are cells of the H100 kernel's gate problem, and they agree with the
    gate's expected buffers, whose hashes they record.
  - The H100 capture confirms it on silicon.

## `pearl-c-sm120-v1`: Pearl-C v1 on the RTX PRO 6000 Blackwell (`schemes/pearl_c_device.py`)

The same scheme as `pearl-c-h100-v1` on another device record. The record (`Device`) is a scheme parameter: the FP8 atom
(E4M3, k = 32), the BF16 peel step (k = 16), G (atoms per promotion), the W1 prices (`Prices`) and the domain's k bound.
`pearl_c`, `pearl_c_debit`, `pearl_c_u` and the work law (`clear_work`) read them from it, and the H100's record (`H100`) is every
function's default, so every `pearl-c-h100-*` output and vector above is unchanged.

- **The atom is the card's captured step**, `BLACKWELL_SM120_E4M3_M16N8K32`: `mma.sync` `kind::f8f6f4` m16n8k32 E4M3 as one
  group of all 32 products and the accumulator in a 26-bit adder (Lean's `Pipeline ⟨[32], 26, −133⟩`), where the H100's
  and Ada's FP8 adders are 14 bits wide.
- **The cap is ρ = 1/1,000**, the record's `cap`; the H100's stays 1/400.
  - The verifier rejects a drawn tile whose replayed debit passes 1/1,000 of its credited cells' credit, and an accepted
    unit is credited (1 − 1/1,000)·`credit_of`. `volunteer` reads the same cap. The certificate states TT_OUT(1/1000) and
    names the cap among its parameters.
  - γ at the headline record (FP32 at 8.376, the packed cast at 8.72, rev1's credit), per unit, is the pinned guarantee
    `Pouw.Guarantees.PearlC.GammaSm120v1LoopCast8p72Rev1Cap1000_8192` (8,192³) and
    `Pouw.Guarantees.PearlC.GammaSm120v1LoopCast8p72Rev1Cap1000_16384` (16,384³); per audit tile, its sampled twins
    `Pouw.Guarantees.PearlC.SampledSm120v1LoopCast8p72Rev1Cap1000_8192` and
    `Pouw.Guarantees.PearlC.SampledSm120v1LoopCast8p72Rev1Cap1000_16384`, from TT_OUT at 1/1,000.
  - The per-unit γ doesn't need TT_OUT(1/1,000): the same γ for the cap-1/1,000 protocol follows from rev1 at 1/400 alone
    (`Pouw.Guarantees.PearlC.GammaSm120v1LoopCast8p72Rev1Cap1000_8192OfCap400` and
    `Pouw.Guarantees.PearlC.GammaSm120v1LoopCast8p72Rev1Cap1000_16384OfCap400`, from `TTOutPearlCDevRev1W1` at 1/400).
  - At the issue-bound FP32 add (8.00), the pinned guarantees are
    `Pouw.Guarantees.PearlC.GammaSm120v1Cast8p72Rev1Cap1000_8192` and
    `Pouw.Guarantees.PearlC.GammaSm120v1Cast8p72Rev1Cap1000_16384`, with the sampled
    `Pouw.Guarantees.PearlC.SampledSm120v1Cast8p72Rev1Cap1000_8192` and
    `Pouw.Guarantees.PearlC.SampledSm120v1Cast8p72Rev1Cap1000_16384`. The ledger publishes the larger of the two prices,
    the in-loop add's.
  - The verifier's credit, cap and debit are rev1's, the rule these γ are proved for: `credit_of` is `creditDevRev1`,
    `wref` is `wrefDevRev1` (`wrefDevRev1K` at the cast's offset), and the tile rule is `tileDebitRev1` against
    ρ·`creditDevRev1`·`tileShare`. Python's γ, credit and tile cap equal the Lean's exactly on every pinned price twin
    and on the H100's rev1 pins (`benchmarks/pouw/tests/test_price_twins_accounting.py`).
- **Every other field of `SM120` is a stand-in**, and `standins` names each one; the certificate lists them as hypotheses.
  - The peel is `HOPPER_BF16_M16N8K16`, which matched the RTX 5090's `mma.sync` m16n8k16 BF16.
  - G = 4, the H100's W1 prices and k ≤ 2^16 are the H100's. They are statement parameters; P1 and the debit are not yet re-derived for a 26-bit adder.
- **`SM120_UNPROMOTED`** is the same record with G = None: one accumulator from +0 over the whole of k, whose word is the
  last atom's output, with no promotion adds in `wref` and 32 ∣ k. No scheme is registered on it, and the debit
  raises on it, until Pearl-C is restated without promotion.
- `pearl_c_u` restates a step of several groups as one grouped sum per group, the accumulator joining the first and each
  group's word the next.
- Only forming v1 runs on a non-H100 record (v0 is Pearl's H100 port).
- The work sampling design aligns `k_unit` to 128, so it requires 32·G ∣ 128.
- **The circuit (`circuit.pc8`)** is forming v1's tile check as a Program, cut as `circuit.pc4` cuts Pearl-C4's. Its
  units are three templates, each with the device record as a static (`DEV`: `h100`, `sm120`, `sm120-unpromoted`):

  ~~~text
  template              per                      reads                                   returns
  Pc8RowB{K,DEV}        row j of B               B_j, E_B,j, F_B, F_A, G                 B̃_j's codes, P_B,j, ok
  Pc8RowA{K,DEV}        row i of A               A_i, E_A,i, F_A, F_B                    A′_i's codes, P_A,i, ok
  Pc8Tile{K,TM,TN,DEV}  tile of TM × TN words    its rows' A′, P_A, B̃, P_B               C̃ and U of its words
  ~~~

  - The roots are `pc4`'s: `Pc8Weight{K,N,DEV}`, `Pc8Call{M,K,N,TM,TN,DEV}` and `Pc8Check{M,K,N,TM,TN,DEV}`.
  - The atom is the record's (`HopperE4m3QgmmaDot32` or `BlackwellE4m3QmmaDot32`), and so is its promotion period. The
    peel is `HopperBF16WgmmaDot16` on every record.
  - ``ok`` is the domain on the row: every word finite, ρ finite, the noise floor as the sign of fma(ρ, α, −1), and
    `live_row` on exact squares. `words.PouwSqAcc`, `PouwSqCeil` and `PouwSqGe` scale each square by 2^300 into a
    576-bit integer, which never leaves the row unit. No other rejection of the scheme's is reachable on such a row (the
    module docstring).
  - The noise lines and G are verifier-derived public inputs, as in `pc4` (`anchors.TileAnchors`, G on the record's
    chain), and each is a public input region in a unit's proof (`pc8.DERIVATIONS`; one u16 word per E4M3 code).
  - The Programs evaluate to `pearl_c.json`'s pinned H100 test instance (and its decode-32 case's B̃, P_B, C̃ and U), and
    to the scheme on sm_120's records, a short last tile row included. Every partition computes each value in exactly
    one unit.
  - **A short last tile row.** `Shape.check` is the scheme's domain down the rows (`Row tiles` above: whole 16-row
    ticket subtiles), so where 64 ∤ m the last tile row is r = m mod 64 rows, its own batch of `Pc8Tile` (or
    `Pc8TileHidden`) at r × 64, whose digest is the scheme's leaf over its real rows. So m = 80 is a 64-row tile row and
    a 16-row one, and decode's m = 32 is one 32-row tile row at the default tile.
    - `plan.tile_layout` proves a short tile with its r `Pc8RowA` units, in its own stratum at W_ref r/64 of a whole
      tile's.
    - A call with both whole and short tile rows is therefore two tile strata, which the work sampling design refuses
      (`plan.check_work_strata`, the window pin's `hone`). Under the work sampling design such a call is not drawn yet;
      a call that is all one tile row height is.
  - The hiding Programs are `pc4`'s (``Shape.hidden``; the rows and their vectors are under `circuit.pc4` below):
    `Pc8TileHidden` returns its tile digest in place of C̃ and U, the weight's Program is `Pc8Weight`, and the rows are
    C-Flock's committed ports (`pc8.PORTS`), with `pouw/row-fp32/v1` input rows, `pouw/row-fp8-p/v1` rows and
    `pouw/row-ok/v1` rows. A 64 × 64 tile's digest is 281 Keccak-p[1600, 12] in gates. `plan.tile_layout` draws a tile
    with its rows, as for `pc4`.
  - **The cap** (`circuit.cap`; opt-in, `pc8.Shape.cap`, off by default, on the hiding Programs only) decides rev1's
    `capOK` (`pearlCTilesDevRev1`: `tileDebitRev1 ≤ ρ·creditDevRev1·tileShare` over a tile's passing rows and all its
    columns, at the record's `cap`) in gates, the debit with it. The call is `Pc8CallCap` (the check `Pc8CheckCap`), cut
    into a unit per word, since a cap word is about 7.8M ANDs at K = 8,192 and a whole cap tile does not stage, and its
    tile digest and tally cut per tile row, since C-Flock gives each row a unit reads or writes a 2^18-bit hm96 slot in
    a block of at most 2^27 bits (at most 256 rows) and a 64 × 64 tile has 4,096 words:

    ~~~text
    template                        per           reads                                 returns
    Pc8RowACap{K,DEV}               row i of A    `Pc8RowA`'s inputs                    `Pc8RowA`'s outputs, sd_i
    Pc8CapStrip{K,TM,TN,N,RHO,DEV}  TM rows of A  their ok words                        their admission bits, the bound
    Pc8CapWordUnit{K,DEV}           word (i, j)   A′_i ‖ P_A,i, B̃_j ‖ P_B,j, its cert  C̃, U, flags, their count, later
                                                                                        +0 promotions, good
    Pc8CapRowDigest{TN}             tile row      its TN words' C̃ and U                 its chunk digests
    Pc8CapRowTally{K,TN}            tile row      its TN words' flags, counts,          its flag count, later +0
                                                  promotions and good, its A row's sd   promotions, forming elements,
                                                                                        good
    Pc8CapTile{TM,TN}               tile          its TM rows' chunk digests            the tile digest
    Pc8CapTally{K,TM,TN,N,RHO,DEV}  tile          its TM rows' sums, its strip's bits   the cap's verdict (an ok word)
                                                  and bound
    ~~~

    - sd_i is row i's salt-dead count per block of 32 elements (P3's `saltDeadP`, decided at its support's two ends:
      `saltDeadP_iff_ends`). The certificate is Freeze's intervals for the word's groups, prover advice in no row or
      region (`cap.ADVICE`); only the exact intervals pass, and any other certificate fails good.
    - **A drawn tile is accepted only when its tile check passes and its tally's verdict is ok**: its tile unit's digest
      is the miner's h0, and its tally's ok word, the tally's public output beside the digest, is ok. The verdict is
      `Pc8TileCap`'s (the cap in one tile unit, the split's reference): every passing row's words are good, and the
      cap's inequality holds in its integer form on 31-bit words (`cap._cap_constants`).
    - Every cap word is proved, never sampled: a drawn tile's closure (`plan.tile_layout`) holds its TM A rows, its
      strip, all TM·TN of its word units, its TM row digests and row tallies, and its tally (4,291 units at 64 × 64, none
      with more than 66 rows). Every partition still computes each value in exactly one unit: the strip computes its
      rows' admission bits once for all the tiles in it.
    - The cap's rows are `sd`, `cap-word`, `cap-flags`, `cap-chunks`, `cap-sums` and `cap-rows` (under `circuit.pc4`
      below).
    - The cap takes whole tiles only: `Shape.check` refuses a short last tile row under the cap.
    - **The per-row cap** (`pc8.Shape.row_rho`, under the cap only) is v2 of `Pc8TileCap`, `Pc8CapTally`,
      `Pc8CallCap` and `Pc8CheckCap`, each with the static RHOR (v1 stays byte for byte): the verdict also holds each
      passing row's own sums to the cap's inequality on that one row at ρr = 1/RHOR, the clear path's row cap
      (`clear_work.cap_failure`, which `test_pearlc_cap_agreement` holds it to tile by tile). `cap.served(scheme,
      shape)` is a scheme's hiding cap shape: under -h3, ρr is `row_cap_at(k)`, 4ρ (RHOR = 100) at
      `pearl-c-h100-v1-h3`'s census k = 256 and ρ at every other k; under -h0 it is v1, the tile's cap alone.
  - The row cut along k (``Shape.row_k``, `circuit.rowk`) is `pc4`'s, below, on each of the three records. It takes
    whole tiles only: `Shape.check` refuses a short last tile row under the cut.
  - Not in it yet: the cap in a run. Until it is, Pearl-C's runs are checked by its reference audit, a development
    shortcut that opens rows.

## `pearl-c-nvfp4-v0`: Pearl-C on sm_120's block-scaled FP4, NVFP4 only (`schemes/pearl_c4.py`)

The RTX PRO 6000 design. It is Pearl-C with the FP8 product replaced by `mma.sync kind::mxf4nvf4`, on NVFP4
(UE4M3 scales per 16).
- **MXFP4 is a contrast line, not in the claimed domain.** The module keeps it (UE8M0 scales per 32,
  `pearl-c-mxfp4-v0`), unregistered in `schemes.get`, with a certificate that claims nothing.
- **Why it is excluded:** its int8 closure holds under F2, but F1′ below puts 9.5–57% of honest MXFP4 credit over the
  cap on every model, so its line is D on honest coverage. Its UE8M0 scales don't move with the salt, and its round-up
  scales make honest zero codes reliable.
The conjecture it rests on, TT_OUT-FP4 (`tt-out/fp4-sm120`), is proposed, not proved, so
the certificate names no γ_0. The module's docstring states every step and rounding. TT_OUT-FP4 is stated at the W1
machine on `W1Device.fp4Sm120` (`Security/Definitions/Pouw/Assumptions/PearlC/TTOutFp4W1.lean`), and the headline γ
guarantees conclude at that cost model. Its domain rule D-SK reads `B̃`'s lines from the oracle, which a W1 program
can't call, so it says nothing about the deployment until D-SK is stated over SHA-512.

~~~text
rows      k BF16 words per row of either operand
stats     s, ρ as pearl-c-h100-v1; σ = ρ/4; α = 2688/max(s + 4σ, 2^−32) (mxfp4: floored to a power of two);
          β_row = (σ·α)·f32(16/√2346.125), a_E = ue4m3(β_row)
seeds     seed_A = H_"seed-A"(salt ‖ root_A ‖ root_B ‖ index u64 LE);  seed_B = H_"seed-B"(salt ‖ root_B);
          H_ℓ = SHA-512 of pearl-c4/v1/ℓ ‖ \0 (zero-padded to one 128-byte block) ‖ x, its first 32 bytes (`labelled`)
lines     32 uniform E2M1 codes, 16 bytes low nibble first: quarter index mod 4 of SHA-512(pearl-c4/v2/line ‖ \0
          zero-padded to one 128-byte block ‖ seed ‖ role ‖ ⌊index/4⌋ u32 LE), all 64 bytes (`line_block`)
forming   D_t = NVF4.step(C = α·x_t, E ‖ 0 at a_E, F_t ‖ 0 at 2^−4); per scale block s_b from amax|D|/6 (nvfp4: UE4M3
          nearest even, at least 2^−9; mxfp4: UE8M0 rounded up); code = e2m1(D·(1/s_b))
chain     C̃ = the scheme's atom chained from +0 over k/64 steps, no promotion
peel      Ẽ = E·a_E·2^−4 (exact in BF16); P_A = [Ẽ_A ‖ bf16(A′·F_B)], P_B = [bf16(Ẽ_B·G − B̃·F_A) ‖ −Ẽ_B], G = F_Bᵀ·F_A;
          U = C̃ + P_A·P_Bᵀ as four HOPPER_BF16_M16N8K16 steps (sm_120 mma.sync) into C̃'s word
leaf      pearl-c-h100's; tickets: Pearl's lottery over C̃, -h3's SHA-512 H_jackpot(seed_A ‖ fold) at every hashing
useful    U/(α_A·α_B), glue
~~~

- **The chain is exact on benign families:** there each C̃ word is the exact dot product rounded once toward zero (a
  test), so rounding binds no word to its order. D₄ doesn't guarantee it (`Fp4Freeze.noLoss_gap`, an in-domain step
  that drops a bit), so no claim rests on exactness: the debit's terms are replayed per run. The design's defense
  against a salt-free precomputation is that every code and every block scale depends on the salt.
- **B̃ is keyed by this job's salt (D-SK):** seed_B = H_"seed-B"(salt ‖ root_B) (`seed_b`), and the verifier forms B̃
  itself from it on every drawn tile. So a transcript whose B̃ was formed under another salt, or under none, fails its
  checked values (a test). This is load-bearing: B̃ exists only once the salt does, so the salt-free base split is
  two-sided (A′·B̃ᵀ = A0·B0ᵀ + ΔA·B̃ᵀ + A0·ΔBᵀ, the cheater paying both corrections), and that is what ties it on
  crafted stride rows. A variant that fixes B̃ before the salt needs a per-salt split debit instead.
- **seed_A doesn't read A's rows, so Pearl-C's adaptive-A attack applies here too.** `noise` keys every row's E_A by the
  unit's seed_A alone, so the prover knows each row's noise before it chooses the row and can fold every row onto one A′
  row, whose C̃ it computes once per weight row, while every row passes every rule. The fix is Pearl-C's -h3 (above):
  - `PearlC4(hashing="h3")` (`pearl-c-nvfp4-v0-h3`, not registered) draws row i's E_A as line 0 (role E_A) of its own
    seed, `pearl_c4.row_seed`: `pearl_c.row_seed`'s layout under Pearl-C4's label, H_"seed-A"(salt ‖ d_i ‖ root_B ‖
    u·2^32 + i u64 LE) over the row's SHA-512 digest d_i (`pearl_c.row_digest`), the same SHA-512 seed-A derivation as
    the unit's.
  - seed_B and what it keys (E_B, F_A, F_B) are unchanged, and the tickets keep the unit's seed_A over root_A. A's and
    B's trees are `audit.HM96`, the tile leaf is Pearl-C's -h3 (SHA-512); the work sampling design checks the row cap as
    well as the tile cap.
  - The `test-16-nvfp4-h3` vector pins a call end to end, its roots 64 bytes. -h0 keeps its names, circuit
    (`circuit.pc4`) and kernels, and no device path forms -h3 yet.
  - At every hashing the seeds (`pearl-c4/v1`) and the noise lines (`pearl-c4/v2/line`) are SHA-512 and the tickets
    -h3's, so Pearl-C4 claims `cr/sha-512` and reaches no BLAKE3. The device's seeds (`pearl_c4.cu`'s `commit_rows`) and
    lines (`line_block`) hash with `pearl_c/sha512.cuh`.
- **The domain (`check`):** 1,024 ≤ k ≤ 2^16 with 128 | k, 64 × 64 tiles, m ≤ 2^24, and 128 ≤ n ≤ 2^18. The n ≥ 128
  floor is B-OVF's (below): an MQA unit of one 64-wide KV head (n = 64) would need a 5.7% discount and is not claimed.
- **B-OVF, a measured credit discount per n bucket (`B_OVF`, `b_ovf`, `credit_of`):** creditOf(m, k, n) = mnk + 32·mk + 256·mn + f_s·mk − ⌈β(n_b)·mnk⌉, where n_b is the largest tabled n ≤ n.
  - β = 4.14 / 2.23 / 1.19 / 0.50 / 0.18% at n_b = 128 / 256 / 512 / 1,024 / 2,048, and 0 from 4,096 up.
  - It covers a layout for B̃'s residual chosen after the salt and shared by all n weight rows. At small n such a layout
    fits the realised change pattern, and it can pay even where F1′ is near zero, so no factor on F1′ closes it.
  - β is measured, not derived. It is the worst net saving that annealing plus exact block optimisation found over the
    families R1 admits at 10×, after A0's gather and ρ, with a 1.5× margin on the gain, made non-increasing in n.
  - It is measured on each bucket's own R1 admission boundary at k = 1,024, where the cap is largest, and at 0.005,
    0.01 and 0.015 inside it, since the worst family can sit there (at n = 256 and 2,048 it does). The need is
    1.5× the mean over 8 draws of the best search. Each bucket takes the largest of that need, the
    boundary check (2.23% at n = 256), the earlier grid's cells under the same cap (0.18% at n = 2,048, from its
    L = 0, p = 0.40 cell) and the grid's table.
  - Exact optimisation stays within 1.02× of annealing at n = 256 and 512.
  - R1's per-row bound and the tile cap read `credit_of`, so both tighten with it. The fork (`tt-out-aw`) takes the same
    discount.
  - Honest cost, from shapes alone: 0.54% of credit on Qwen2.5-0.5B, 0.24% on 1.5B, about 0.1% on 3B, 7B and
    Llama-3.2-1B/3B, and at most 0.019% from 8B up.
- **Admission, per row and salt-free (`row_passes`):** every word a finite widened BF16 word (its low 16 bits zero, the
  only words `parse_row` writes; Lean's `Fp4Sem.RowOK` and `OperandOK`), and D-NF, a_E is a normal UE4M3 (one a_E per
  row, so the noise is strictly row-relative). It reads (`dnf`) 0 ≤ β and v in 0x08–0x7E, where v is β's E4M3 byte
  with its sign; these are the bytes Lean accepts. The pinned stats never make β negative or NaN, but the rule states both cases:
  - A negative β would cast to a byte with bit 7 set. At β = −1/4096 the byte 0x80 passes 8 ≤ v alone, and sm_120
    reads it from its low 7 bits as a zero scale, so the row would carry no noise.
  - A NaN β with its sign clear would cast to 0x7F.
- **Screened blocks are charged, not rejected (`dead_blocks`):** a block is screened when (clean amax_b)² > 90·ρ²
  (9.5ρ), and the screen covers every salt-dead element (Lean `saltDead4_screened`). Items 3 and 4 below charge the
  reference's witness (`salt_dead`) per element, not every element of a screened block, so a row with many screened blocks is rejected only when its tile's
  debit passes the cap, and neither item charges more than its block form. F1′ charges a screened block's near-certain
  codes at their own p_t. This replaces D-SS's one-in-64 row cap, which rejected about half of real activation rows.
- **The debit, replayed per drawn tile over its credited words (`tile_debit`),** in FP4 units. Items 1 and 1b and R1
  are the base-split and flatness fix.
  1. **F1′, the base split's expected saving,** per row and aligned 128-window of A′ and of B̃, on every word the row
     feeds:
     - p_t = P(code ≠ its modal code) at the block's modal scale byte, under the exact law of the row's noise given
       this salt's E line (`noise_law`);
     - the charge is 128·min(1, E[max(0, R_24 − κ)] + E[max(0, R_skip − κ)]), where R_24 is one 2:4 pass plus a
       CUDA-core patch of the excess, and R_skip is a skip or patch per k64 atom;
     - it is taken at the cheater's sorted layout inside each block, with κ = min(c·M_w, 0.125) inside the expectation,
       where c = 16.92/128 and M_w counts the window's blocks whose realised byte isn't their modal byte;
     - B̃'s rows are charged `F1_B_OVERFIT` = 10 times it per window, capped at the window's MACs. That factor is
       measured, not derived; at n ≤ 1,024 a stronger search pays past it, which B-OVF covers.

  1b. **F2, the masked int8 route's saving,** per tile on both operands: max(0, 1 − c_L − 4·(2 − f_A − f_B)) per chain
      MAC.
      - f is read at the tile's modal byte, per row and per block-column class, taking the largest reading.
      - c_L is the fast-matmul catalogue's minimum at its lower-bound add count, one add per new form. It is pinned per shape (`pearl_c4_c_L`), and a shape between table points reads the next
        point up in m, k and n, whose c_L is the lower.
        - The routes: Perminov's catalogue at `ac13ca88`, every orientation, rational levels included, nested with
          Strassen–Winograd levels; W1's int8 leaf 2.0; 4.25 FP4 slots per pre-add element-op; A′'s pre-adds amortised
          over n and B's over m.
        - The table is never above the Strassen formula pinned before (min over 1 ≤ L ≤ log₂(k/16) of
          2·(7/8)^L + 4.25·(1/m + 1/n)·((7/4)^L − 1)/0.75).
        - Values: 0.661, 0.579, 0.507 and 0.408 at 8,192³ to 65,536³, where the formula gave 0.808 to 0.542.
        - Honest tiles are unaffected: F2 fires only past f_A + f_B = 2 − (1 − c_L)/4, 1.915 at 8,192³, against their
          0.13–0.30.

  R1. **A row whose own F1′ per word passes ρ·(its cell credit) is uncredited.**
      - The prover declares it (`clear_work.declare` with the transcript).
      - The verifier checks the declaration on every drawn tile (`check_opened`): a credited row over the bound, or an
        excluded row under it that passes the per-row rules, fails the tile.
      - The row is still formed, checked and hashed.
  2. per identity atom (its lone skip leaves the word unchanged), 64 per atom-word;
  3. per word and position whose A code and B code are both salt-dead (`salt_dead`, replayed per element on both
     sides from the drawn tile's own forming), one MAC; a fully dead atom is charged its 64;
  4. per salt-dead A element (the reference's witness, `salt_dead`) or unneeded one (every credited B code at its
     position is zero), f_s·|cols|/n;
  5. per (word, block) whose largest possible scaled dot, 2304·m_a·m_b·2^(e_a+e_b−2), lies below its step's grid (it
     adds exactly 0), one block of MACs;
  - and D-24: per row and aligned 128 of k whose every aligned 8-chunk has at most 2 of its 4 pairs (2j, 2j + 1)
    holding a nonzero code ((code & 7) ≠ 0), 0.5 per MAC on every word it feeds. That is the pair rule of sm_120's NVFP4
    sparse MMA (`mma.sp…kind::mxf4nvf4…m16n8k128`, 2 of each 8-chunk's 4 pairs kept); `kind::f8f6f4`'s element-wise
    2:4 has no UE4M3 16-block scales and runs at the FP8 rate, so it saves nothing. It is charged per row window rather
    than per 16-row span, since a cheaper computation may gather any 16 such rows; the literal spans are counted.
  - A code is salt-dead unless two draws in the noise support give it two values (`salt_dead`: E ≡ +6, the element's
    F_t in {+6, 0, −6}, its block's other F lines 0). Uncharged codes are those two draws prove live, so the charge errs
    high.
  - D-SS's spread bound (2^15.8, implied by D-NF) is counted, not enforced: items 2 and 5 charge what it would exclude.
  - D-SB is enforced on every formed side before hashing (`check_scale_bytes`): every block scale in 0x01–0x7E (UE8M0:
    short of 0xFF) and every a_E at most 0x7E, so no UE4M3 byte has bit 7 set, which sm_120 reads from its low 7 bits (a
    second encoding of the same C̃), none is 0x7F (NaN), and no block scale is zero. Every committed word is finite
    (`parse_row`). An a_E of 0x00 is D-NF's exclusion, not a rejection, since a zero row (decode's filler) forms it. The
    verifier forms every scale itself from the committed rows, and every replayed step reads a UE4M3 byte as the hardware
    does (`hw_scales`).
    The decode is the Lean `scaleDyadic` proposal's.
    - Its `nvf4Specials` table has 28 rows.
    - Four of those rows are NaN accumulators whose scale bytes set bit 7. They are valid scales under the hardware
      decode, and outside the model through the accumulator (a test); the card writes 0x7FFFFFFF.
    - The pinned model (`verity.ml.circuits.silicon.models._scale_dyadic`) reads the low 7 bits too, and all 17 bit-7
      vectors replay on the RTX PRO 6000 word for word.
  - A drawn tile is rejected when its debit exceeds ρ = 1/400 of its credited cells' share of creditOf.
- **Accounting** in FP4 units (placeholder W1 prices: a BF16 MAC 4, an FP32 instruction 16):
  - W_ref = mnk + 32·mk + 256·mn + 117·mk;
  - creditOf = mnk + 32·mk + 256·mn + f_s·mk, f_s = 98 (the steps that read the salt at their cheapest bit-exact price:
    the noise atom 64, the block scale about 1, the cast 32.8), so W_ref − creditOf = 19·mk (the noisy amax, 17, and
    ρ's accumulate, 2);
  - the weight side is `job_cost`.
- **Registered weights** are the shared rule ("Registered weights (`registration`)"), through
  `PearlC4.check_registration`.
- **The work sampling design** is `clear_work`'s. The scheme brings its own row parser, per-row rules and tile debit.
- **The reference replay (`pearl_c4_replay`)** checks a device run's own transcript under that sampling design's rules,
  which every measured row needs:
  - the device's dump: A's and B's rows, its roots and its tile leaves, and the scale bytes its forming wrote;
  - `prove` turns the dump into transcript v0: the transcript commitment, the records with the rows that fail the
    per-row rules or R1 (`r1_row`, row by row over its workers), the draws and their openings, and a count of device
    scale bytes outside D-SB's range;
  - `verify` recomputes the salt, the draws and every opened tile, including its debit against the cap, and prints the
    accept line.

  It takes about 2 CPU-minutes per 64 × 64 tile at k = 8,192, and F1′ adds about 0.35 CPU-s per formed row, evaluated
  once per row and shared by R1 and the debit (about 0.75 CPU-minutes per tile; measured on a cloud-agent VM before the
  sharing, at twice that). The protocol keys the draws and the salt by the epoch's live coins, with no Fiat-Shamir in its
  draw. The replay's shortcuts are in every report: its draws are keyed by Fiat-Shamir over the commitment in place of
  the draw coin, the epoch's salt is from the benchmark's label, and the transcript tree and the records are built on
  the host after timing.

  Its ACCEPT is replay-verified (development shortcut; the protocol's check is a ZK proof). The replay opens rows and
  words and recomputes them in the clear. The protocol keeps every value hidden: it checks a drawn tile by a ZK proof of
  its units under sampled proofs, which opens nothing. Its units are the hiding Programs of `circuit.pc4` (below), whose
  every row is a committed hm96-sha512 row hashed inside the proof and whose roots return only admission bits and tile
  digests. So no report cites the replay's ACCEPT as the protocol's verdict.
- **The circuit (`circuit.pc4`)** is the tile check as a Program, the first step of moving it into sampled proofs. Its
  units are three templates:

  ~~~text
  template          per                      reads                                   returns
  Pc4RowB{K}        row j of B               B_j, E_B,j, F_B, F_A, G                 B̃_j's codes and scale bytes, P_B,j, ok
  Pc4RowA{K}        row i of A               A_i, E_A,i, F_A, F_B                    A′_i's codes and scale bytes, P_A,i, ok
  Pc4Tile{K,TM,TN}  tile of TM × TN words    its rows' A′, P_A, B̃, P_B               C̃ and U of its words
  ~~~

  - `Pc4Weight{K,N}` holds the B rows, once per job and weight. `Pc4Call{M,K,N,TM,TN}` reads their outputs as
    committed inputs. `Pc4Check{M,K,N,TM,TN}` is both in one root. The scheme's tile is 64 × 64.
  - A unit is a tile, not a word, because a unit reads its rows through their commitments. Once rows are hidden it
    hashes them, and a word unit would hash about 4.9 KiB of rows for each word.
  - ``ok`` is the row's admission (`parse_row`'s finiteness, then `row_passes`). A row unit returns it as its ok row's
    one word, `SelectBf16` of the bit between the constants 1 and 0 (`pc4.ok_word`), so bits 1–15 are zero in gates.
    No row check sees a hidden row's bits, C-Flock's and Lean's included, so this is what gives one admission value
    one leaf; the reference refuses any other word (`leaves.ok_of`).
  - The noise lines and G are verifier-derived public inputs, built as `ncp-v2`'s anchors are (`anchors.TileAnchors`,
    from the scheme's `noise(ctx)`): the lines are SHA-512 of seeds that read public values alone (seed_A the
    salt, A's root, B's registered root and the matmul index; seed_B the salt and B's root), and G = F_Bᵀ·F_A (`gram`)
    is a function of them. `check_inputs` refuses any other. Both Programs read F_A and F_B; E_A is the call's, E_B and
    G the weight's.
  - In a unit's proof each anchored input is a C-Flock public input region (`In`), costing no gate: `pc4.PORTS` names
    each row unit's regions, and each region names its recomputation (`pc4.DERIVATIONS`: the rule, the seed, the domain
    and the derivation function; as C-Flock's circuit names them, `anchors.public_ports`: the seed by the scheme's label,
    `pc4.SEEDS`, and the rule by its versioned id, `verity/pouw/anchor/pearl-c4/<anchor>/v2`, never the function's
    Python name, since both strings are in C-Flock's circuit pin). The verifier values them from its own `--public-inputs` file
    (`TileAnchors.public_inputs`, C-Flock's `verity/flock-public-inputs/v1`: a header line naming the ports, then each
    instance's regions), whose content for one unit is `TileAnchors.regions`: every value unpacked into u16 words (one
    per E2M1 code, two per FP32 word of G, low half first), each region whole 128-bit words, its bytes the words
    little-endian. `tests/vectors/pouw_regions.json` pins both. Each region is in the unit's parameter leaf order, not
    the order its gates first read it (F_B is read transposed), and the order is part of what the verifier checks: a file
    with the right values in another order, or with two instances swapped, is refused (`TileAnchors.check_public_inputs`,
    as C-Flock's Lean verifier refuses it).
  - The hiding Programs (``Shape.hidden``: `Pc4CallHidden`, `Pc4CheckHidden`) differ from the plain ones only in the
    tile: `Pc4TileHidden` is `Pc4Tile` and its tile digest (`circuit.leaves.PearlTileDigest`, `pearl_c.leaf` of C̃ and
    U, on `hashes.KeccakP1600R12`), which it returns in place of C̃ and U. The weight's Program is `Pc4Weight` under
    hiding too. A root returns only admission bits and tile digests, and every partition still computes each value in
    exactly one unit.
  - The hiding is C-Flock's: every row a unit reads or writes is a committed port, which C-Flock commits as an
    hm96-sha512 row and hashes in the proof (`--zk`), so the Programs hold no leaf and no salt. `pc4.PORTS` names each
    unit's committed rows: a row unit's input row (bf16), the nvfp4-p row it writes and its ``ok`` row (every output is
    committed), and a tile's TM + TN nvfp4-p rows and its tile digest. A tile reads each row as its row unit wrote
    it (the same words, schema and salt), so the units link by equal commit strings b ‖ c.
  - The rows are rows of 16-bit words, each committed under the schema of its row and its tree's role
    (`leaves.ROW_SCHEMA`): an `hm96-sha512/row-seg/v1` row (`sha512/row-seg/v1`: the words' bit row in 1 KiB segments that
    hash in parallel, frame-v3's PROTOCOL.md) or an `hm96-sha512/row/v2` row (`sha512/row/v2`: the words' bit row,
    serial). No digest names the role or the schema, so the registration domain of the tree a row sits in binds the schema
    id, the word count and the role, and each tree holds exactly one (schema id, word count, role). Every narrower element
    is packed low bits first, so a row's bytes are the tensor's memory (`rowleaf.bit_row`'s of its elements):

    ~~~text
    schema id            words                                                     count             role               row (X / W)
    pouw/row-bf16/v1     k BF16 words                                              k                 X (A_i), W (B_j)   seg / v2
    pouw/row-fp32/v1     k FP32 words, low half first                              2k                X, W               seg / v2
    pouw/row-nvfp4-p/v1  k E2M1 codes two per byte (low nibble first), the k/16    k/4 + k/32 + 64   X (A′_i ‖ P_A,i),  seg / v2
                         UE4M3 scale bytes, then P's 64 BF16 words                                   W (B̃_j ‖ P_B,j)
    pouw/row-fp8-p/v1    k E4M3 codes, then P's 64 BF16 words                      k/2 + 64          X, W               seg / v2
    pouw/row-fp8-pi/v1   an fp8-p row, then its index field (a window's side row,  k/2 + 72          X, W               seg / v2
                         `window.side_row`): I or J, a u32 low word first, 6 zero
    pouw/row-ok/v1       the row's admission bit, bit 0 of one word, bits 1–15 0   1                 X, W               v2 / v2
    pouw/tile-h0/v1      the tile's 32-byte h0 digest (`pearl_c.leaf` of C̃, U)    16                X                  v2
    pouw/row-sd/v1       a row of A's k/32 salt-dead counts, each 32-bit word as   k/16              X                  seg
                         two, low half first (Pearl-C's cap)
    pouw/cap-word/v1     a cap word's C̃ and U, as the FP32 row's words            4                 X                  v2
    pouw/cap-flags/v1    a cap word's k/32 flag bits 16 a word, low bit first      ⌈k/512⌉ + 5       X                  v2
                         (the last word's high bits 0), its flag count and later
                         +0 promotions as two words each, then good's ok word
    pouw/cap-chunks/v1   a tile row's chunk digests, TurboSHAKE128 (D = 0x21) of   64                X                  v2
                         each 32 columns' C̃ and U (`pearl_c.message_digest`), two
                         bytes a word, low byte first (TN ≤ 64; the second 0 at
                         TN ≤ 32), then 32 zero words
    pouw/cap-sums/v1     a tile row's flag count, later +0 promotions and forming  7                 X                  v2
                         elements as two words each, then its words' good as an ok
                         word
    pouw/cap-rows/v1     a strip's admission bits 16 a word, 64 bits (TM ≤ 64,     64                X                  v2
                         the rest 0), then the cap's bound as two words, then 58
                         zero words
    ~~~

    Every per-call row (role X) but ok, the tile digest, the cap's rows and the row cut's rows (below) is a row-seg row,
    so decode's serial hash chain is a segment's, not the row's: A_i (bf16, 8 KiB at k
    = 4,096, 8 segments), A′_i ‖ P_A,i (3 segments), and Pearl-C's FP32 A_i (16 KiB, 16 segments), fp8-p row (5
    segments) and salt-dead counts (1 KiB at k = 8,192, 1 segment). The cap's per-word and per-row-sum rows are at most
    one SHA-512 block at k ≤ 2^14, so a segment's chain is the row's, and v2 hashes them in fewer blocks. The chunk
    digests and the strip row are v2 rows padded with zero words to 64, one whole block: C-Flock's composer refuses a
    unit none of whose rows hashes a block before its last, and without a whole block the strip, a row digest, the tile
    and the tally would read and write only one-block v2 rows. They are not row-seg rows: padded, each costs two
    compressions, as its short row-seg row did, where a row-seg row padded to a block would cost three. A row digest and
    the strip write the zero words as constants, and the tile and the tally leave them unread. Every other row is a v2 row, the registered rows (role W) included, in the registered tree and in every
    statement that reads or writes them, so each (row, role) has one leaf schema wherever it appears and no row is
    `sha512/row/v1`. A v2 row costs what a v1 row of the same words would, a
    128-byte prefix block and the row, and is any length; the role byte v2 drops is bound by the registration domain.
    nvfp4-p's codes and scales are core's `nvfp4_row_bytes`, as in flock-fp's packed rows, with P appended: every unit
    that reads such a row reads all of it, since a tile needs A′_i and P_A,i both. The salt is 192 bytes; the verifier
    computes the tree leaf, H(leaf_prefix(key) ‖ b ‖ c), from b ‖ c.
  - A unit's C-Flock statement names its rows in META (`leaves.statement_rows` of its ports): ``row_schemas``, each
    row's [schema, bits] (the input rows, one per row a tile reads, then the output rows), and ``out_row_ports``, each
    output row's run of output ports, which META names when an output row is row-seg or a run of several. Every output
    row is a row of exactly its words, as every input row is. A row unit writes A′_i ‖ P_A,i (or the fp8-p row) as three
    (two) output ports, the codes, the scales and P, which C-Flock commits as one row-seg row of exactly their words,
    and B̃_j ‖ P_B,j (or the registered fp8-p row) as one v2 row of them; the ok word and the tile digest are each
    one port's v2 row, not padded to whole blocks. So the writer's leaf is the leaf of the
    unit that reads the row as one input port.
  - In gates the hiding adds only the tile digest: 281 Keccak-p[1600, 12] a 64 × 64 tile.
  - Vectors: `tests/vectors/pouw_rows.json` (`tests/pouw_rows_vectors.py`) pins each schema's row bytes, row digest,
    commit string and tree leaf in each role at k = 64 and 4,096 (a row-seg row's with each segment's length and digest),
    the cap's rows at the k where their words change (sd in one segment and two, cap-flags with and without padding,
    cap-chunks with one digest and two), and
    the digest of two tiles, which the gates compute. `test_circuit_leaves.py` rebuilds every row-seg vector from the
    spec by hand and reads core's own row-seg vectors as PoUW's rows. C-Flock's ports commit every row byte for byte:
    `verity/ml/flock/tests/test_pouw_rows.py` recomputes each one's vector through C-Flock's port for the row, and two v2
    rows and a tile through the Lean verifier's row constants; on C-Flock's writer/reader pairs (`ROW_SEG_PAIRS`: A′_i ‖
    P_A,i at k = 4,096, the fp8-p row at k = 1,024; `ROW_V2_PAIRS`: an ok row, a tile digest, B̃_j ‖ P_B,j and the
    registered fp8-p row at k = 1,024) PoUW's commit string is the writer's and the reader's. Each of the six units'
    statement rows compose (a toy unit with the unit's ports): META is `statement_rows`', and every staged row's b ‖ c
    is PoUW's commit string of its words. No C-Flock composer stages a PoUW unit's own gates with these rows yet
    (`class_statement.stage` commits one output port for all of a unit's outputs), so neither does a `--zk` proof.
  - The six PoUW primitives the row units call (`PouwF32SqrtRn`, `PouwETilde`, `PouwBf16Neg`, `PouwSqAcc`, `PouwSqCeil`,
    `PouwSqGe`) have Boolean versions, the same functions on bits (`circuit.boolean`, `_v2`, traced from `forms` builders;
    their AND counts are pinned in circuit-check's `pins.json`). C-Flock's lowering maps a primitive id to a piece of its own
    (`ir_lower.PIECES`); for these six the piece is their Boolean version's builder (`ir_lower._boolean_build`), the one that
    lowers an activation of the Boolean version itself.
  - So do the four Keccak words the hiding tile's digest calls (`PouwXor32`, `PouwChi32`, `PouwRotl64Lo`, `PouwRotl64Hi`,
    on (lo, hi) half-lanes in `hashes`): xor is free, χ one AND a bit, and a rotation a barrel rotator (218 ANDs a half)
    that a constant amount, every one `hashes` builds, folds to wiring, so a Keccak-p[1600, 12] is 19,200 ANDs, all χ's.
    A lowering that keeps a call's outputs as XOR forms, as `ir_lower.lower_gates` keeps a piece's, lets χ's XOR terms
    compound round over round: a 12-round TurboSHAKE128 then has AND operands of 3,518 bits on average (7,970 at most),
    and lowering one permutation runs out of memory. So `PouwChi32_v2` commits its outputs (`forms.Circuit.commit`, a
    row each and no gate), and a Keccak-p[1600, 12] lowered through the four is 38,400 rows, no AND operand over 12 bits.
    C-Flock's piece for each of the four is its builder too, commits included.
  - The draw is `plan.tile_layout` (beside `ncp-v2`'s `plan.layout`, which is unchanged): one stratum per template,
    each tile at the W_ref it is given, and a drawn tile proved with its closure, its TM `Pc4RowA` units, and with its
    TN `Pc4RowB` units in the weight's Program, each once per plan (`Plan.proved_b`). It opens nothing.
  - **The row cut (`circuit.rowk`, ``Shape.row_k``) is opt-in**, for both checks. C-Flock proves one instance in at most
    a 2^27-bit block (`verity_flock.circuit.K_MAX`), and a whole row unit outgrows it before the served k (Pearl-C's row
    of A on sm_120 past k = 2,560; Pearl-C4's fills it at 4,096). A shape with ``row_k`` (2,048 deployed, `rowk.ROW_K`)
    cuts each row unit along k into a head (the row's statistics and E′, or a_E), one form per segment of row_k positions
    (its codes, and the clean-up's 32 chains over it started from the previous form's carry) and a tail (P, from the
    chains and, on B, E′·G). Pearl-C's ok is the tail's (the dead count is carried), Pearl-C4's the head's.
    - A segment is whole device windows and whole 32-position blocks, so each form adds the same atoms in the same order
      as the whole row, and the carry is exact: the pieces' outputs are the whole row's, and every value is computed in
      exactly one unit.
    - The tail writes the fp8-p or nvfp4-p row today's row unit writes, so a tile reads its rows unchanged. The pieces
      pass their values in six more rows, v2 in either role (`leaves`' pc8-head, pc8-carry, fp8-codes, pc4-head,
      pc4-carry and nvfp4-codes; `rowk.PORTS8`, `PORTS4`). At row_k = 2,048 each is short (at most 55 words) or whole
      SHA-512 blocks, the v2 rows C-Flock's widened scope takes; pc8-carry ends in 62 zero words for that.
    - A form reads its segment of F_A and F_B as a public input region under the anchor's slice rule,
      `verity/pouw/anchor/<scheme>/<anchor>/slice/v1` (`rowk.DERIVATIONS8`, `DERIVATIONS4`,
      `anchors.Derivation.per_segment`), which only the cut's shapes name.
    - A shape without ``row_k`` builds today's Programs, their digests unchanged. `pouw_rows.json` pins the six rows and
      `pouw_regions.json` the slice regions (`row_k_cases`). Every form still opens the row of A whole, so a form's SHA
      work grows with k, and past k = 18,432 (Pearl-C's row of A) a form no longer fits.
  - Not in it yet: the debit and the cap, R1, D-24, and the plan's link checks and paths. Until those land, runs are
    checked by the replay above, a development shortcut that opens rows, and every report says so.
- **Stand-ins:** the RTX 5090 pins of both atoms and of the BF16 step, and the E2M1/UE4M3/UE8M0 casts modelled from the
  PTX ISA.
- **Vectors:** `tests/vectors/pearl_c4.json` (`tests/pearl_c4_vectors.py`) holds a 16 × 128 × 16 test instance per format
  and one 64 × 1,024 × 64 NVFP4 tile, each with its replayed debit and cap, and A row 0's F1′ inputs (the noise law's
  digest, the modal bytes, p_t) and charge.

### A row's statistics, scales and forming (`pc8`, `pc4`)

A row of `K` words `x_0, …, x_{K−1}` (`pc8`: FP32; `pc4`: BF16, widened by `Bf16ToF32_v1` wherever read) gives its
scales and codes by the following primitive calls, in this order. Each result is a word of its primitive's return width;
`|·|` is `F32Fabs_v1`.

1. **Running max.** `s_0 = |x_0|`, and for `t = 1, …, K − 1`, `s_t = SelectI32_v1(I32Le_v1(s_{t−1}, |x_t|), |x_t|, s_{t−1})`.
   `s = s_{K−1}`.
2. **Mean square and `ρ`.** With `n = K/8`, partial `i < n` is `a_i = F32Fma_v2(x_{8i}, x_{8i}, c_i)`, where `c_i = +0`
   for `i < 32` and `c_i = a_{i−32}` otherwise (accumulator `i mod 32`); accumulator `j` is its last partial. The butterfly
   runs rounds `h = 16, 8, 4, 2, 1`, each mapping the sums `v` to `v'_j = F32Add_v2(v_j, v_{j+h})` for `j < min(h, n)`,
   with `+0` for `v_{j+h}` when `j + h ≥ n`. `Σ` is the last round's `v'_0`, and
   `ρ = PouwF32SqrtRn_v1(F32Div_v2(Σ, f32(n)))`.
3. **`pc8`'s scales and floor.** `α = F32Div_v2(448, F32Fmaxf_v1(F32Add_v2(s, F32Mul_v2(4, ρ)), 2⁻³²))` and
   `β = F32Mul_v2(F32Mul_v2(ρ, α), 0x3AB504F3)` (`2^(−19/2)`). The floor is `φ = F32Fma_v2(ρ, α, −1)`: the row is above
   the noise floor where `I32Le_v1(0, φ)`.
4. **`pc8`'s forming.** Noise code `e_l` (`l < 32`) becomes `ẽ_l = F32ToE4m3Sat_v1(F32Mul_v2(β, E4m3ToF32_v1(e_l)))`.
   Position `t`'s formed word `D_t` is one E4M3 k32 atom of the device (`BlackwellE4m3QmmaDot32_v1` on sm_120,
   `HopperE4m3QgmmaDot32_v1` on the H100), with accumulator `F32Mul_v2(α, x_t)`, over the pairs `(ẽ_l, F_{t,l})`, where
   `F_t` is the form basis' row `t`. The code is `F32ToE4m3Sat_v1(D_t)`.
5. **`pc4`'s scales.** `δρ = F32Mul_v2(ρ, 1/4)`, then `α = F32Div_v2(2688, F32Fmaxf_v1(F32Add_v2(s, F32Mul_v2(4, δρ)), 2⁻³²))`
   and `β = F32Mul_v2(F32Mul_v2(δρ, α), 0x3EA920A7)` (`k_β`).

The Lean names are `rowMax`, `rowRho`, `alpha8`, `beta8`, `floor8`, `eTilde8`, `code8W`, `alpha4` and `beta4`
(`Security/Definitions/Pouw/PearlC/RowSem.lean`, namespace `PearlC.RowCircuit`). Pearl-C's and Pearl-C4's semantics at them are `pc8Sem` and
`pc4Sem`, on the row's words re-encoded (`f32Word`; `pc4`: `bf16WordOf`). Their `α` and `β` are those words read back as
values. `pc4`'s rules ask finite widened BF16 words, so `bf16WordOf` of a word's value is the word's high half with `−0`
written `+0`. The `pc4` table registers the high half itself (`rows4`), and a row unit reads `−0` as `+0`
(`formRow_table`). `pc8Sem`'s floor is `φ`'s value plus 1 where `ρ`'s word is finite and `0` where it is not, so the rule
`1 ≤ floor` is `ρ` finite and `φ ≥ 0`, the circuit's flag. `pc4Sem`'s `β` is `0` where `β`'s word is not finite, so
`RowAdmit4` refuses the row, as the circuit's flag (`β ≤ F32_MAX`) does. `pc8Sem`'s forming value is `D_t`'s,
and its `e4m3` is the code wherever `F32ToE4m3Sat_v1` agrees with `e4m3Word` and `D_t` is not NaN.

## `compile`: PoUW as one pass over the served Program (`circuit/compile.py`)

Decision 102's shape: the served Program is the model's own, and PoUW is one compile pass over it. `compile(program, scheme,
groups)` returns a `Compiled` (the compiled `Program`, the served one, the scheme, its `Site`s and `Unit`s, and each site's
seed group); `rebuild(served, scheme, descriptor, groups)` is the verifier's side, which compiles itself and refuses any other
descriptor (`traced_as` at Program level).

**Find.** `MATMULS` registers each family's A and B ports, (m, k, n) per Call and the arm it is registered at:

~~~text
family                  A     B      per Call     arm
S8Dot_v1, S8Dot_v2      a     b      1 × K × 1    the plain arm: NOISE "none", T0 "zero", COMMIT "final", no tail, LIMBS "none"
S8Matmul_v1             a     b      1 × K × N    the same; each of b's N rows is a dot
GemmAccumulatorS8_v1    x     wrow   1 × K × 1    any (the W8A8 accumulator, ScaledMmS8Coordinate_v1's node 0)
GemmAccumulatorS8_v2    x     wrow   1 × K × 1    any (on bits)
DotBf16_v1 .. v3        x     w      1 × K × 1    any (bf16 linears, and attention's QKᵀ)
~~~

A Call nested in a Definition is found through it when its operands are that Definition's parameters, at any depth; a
Program whose root is the matmul is served as one Call of it (`PouwCall_v1{FN}`). Attention is to be credited like every A·B:
its QKᵀ is a `DotBf16` kind like any other, but the one registered scheme, `eps1-sm120-h1fx1`, has no unit
for it, so its dots are refused (`scheme`) until one is plugged. P·V as `AttnBlock_v3` serves it, tensor-core steps on a P
the block computes, is no Call of a registered family, so it is refused (`uncovered`) until it is one.

**Tile and replace.** A site is a root node's dots of one kind, and its distinct A rows and B rows, first use first, are its
operands. Its A rows go in blocks of the scheme's r, each block's B rows in chunks of c (r × c is a scheme parameter),
and a tile is the site's dots in one block and chunk. Each tile is one self-contained unit of its kind
(`Scheme.units`: per kind a factory, which carries that kind's noise form; `Scheme.plug` sets one, as a plugged unit will for
P·V): `(a: its A rows, b: its B rows, salt, position, roots) -> {z, receipt}`. z, its cells' exact products, feeds the Call's
consumers in the Call's place, so served values don't change; the receipt is a sink only the Program's outputs read. The units
sit at the root, one batch per site and tile shape, ahead of the node that read them; that node's Definition F becomes
`PouwRest_v1{FN=F}`, F without its dots, reading their z. The compiled Program is `PouwCompiled_v1{SERVED, SCHEME, GROUPS}`:
the served parameters, `pouw.salt`, `pouw.positions` and `pouw.roots` to `{served: the served return, receipts}`.

**Seeds and noise** (`circuit/seeds.py`). The salt is the Program's one coin (`In(kind="coin")`), the
epoch's 64 bytes, a certified epoch's (`Epoch.certified`); `seeds.seed`, `seeds.receipt_seed` and `Compiled.inputs` refuse
any other width. A unit derives its seed in its own proof, and its noise from the seed:

~~~text
seed_j  = SHA-512(u64(j) ‖ (SEED_TAG ‖ salt ‖ root(A_j) ‖ root(B_j)) ⊕ u64(j)*)        PouwSeed
σ_R     = SHA-512(u64(j) ‖ (RECEIPT_TAG ‖ salt ‖ root(A_j) ‖ root(B_j)) ⊕ u64(j)*)     PouwUnitSeeds, with seed_j
block_t = SHA-512(NOISE_TAG ‖ seed_j ‖ u64(t))                                          PouwNoiseBlock
~~~

j is the unit's position and (root(A_j), root(B_j)) its seed group's roots. ``⊕ u64(j)*`` XORs every message word after the
first with the position, an injective encoding that makes every gate of a seed read its own position, so no two units compute
a common value. Unit u reads position slot u (`Compiled.positions(base)`) and its seed group's roots.

The seed inputs `pouw.roots`, `pouw.positions` and `pouw.salt` are registered values the one-stage verifier fills itself,
so the prover chooses none of them (`sampled_proofs.one_stage.registered`'s `supplied_rows` and `supplied_root`; the
sampled_proofs README's "Supplied values"; `Compiled.inputs` gives the leaves it fills):

- `pouw.roots`: each seed group's (root(A), root(B)) from the roots the record names for its operands: a served port's
  root (A's committed tap, an AROW row table included) and a registered value's (B's weights, which the verifier holds to
  the root it pinned before the salt, `one_stage.registration.check`'s `pinned`).
- `pouw.positions`: unit u's position is u after every unit of the log's earlier records (`registered.log_base`), so no
  two records of one log share a position and each (salt, position) seeds one unit.
- `pouw.salt`: per window, filled by the verifier.

R7 recomputes each one's root and refuses a record that registers any other (`R7-supplied`).

A unit's receipt keys and chain starts come from σ_R, a hash of its own tag, never from the noise seed. A unit
whose folds take keys calls `PouwUnitSeeds` once for both, [seed_j, σ_R] with the masked salt and roots shared, since two
seeds at one position are one seed (the coin rules' `E_COIN_REUSE`, `ports.py`).

**Seed groups.** `groups` names the calls committed together: sites of one key share a slot of
`pouw.roots`, each unit still at its own position. One commitment fixes the noise of every unit it covers, so the grinding
premium is a constant number of units per group (`Compiled.cover`), and the scheme's `cover` is the fewest a group may
cover. By default each site is a group, and a site is one call: a layer's QKᵀ batched over its heads and the step's batch
is one site, so one seed covers the call, not one head.

The attention kinds' seed groups are checked by GRIND's bound itself (`Scheme.rows`, `Compiled.grind`). A
kind's `RowNoise` gives the noise bits of one A row and one B row at contraction k: QKᵀ's Q row k + 320 and K row k
(`QK_ROWS`, on `DotBf16_v1 .. v3`), P·V's P row 224 and V column k (`PV_ROWS`, for the kind a plugged unit takes). Over the
group's distinct rows of the kind, each with its bits and its readers among the group's dots, the grind term is
`grind.visible_premium` (the formula `benchmarks/pouw/grind_price.py`'s table reads) over the group's dots, at
T = 2^128 and δ = 2^-40. Over 0.1% (`grind.FLAG`) the pass refuses the group (`cover`): its noise is the live per-round coin
("Live epochs"), not a seed's. One rule covers both regimes:

- at decode the Q rows bind, so it counts every (query head, request) pair, whatever KV heads they share: one call is 8B's
  decode at batch ≥ 13 for QKᵀ and ≥ 25 for P·V (70B: 7 and 13), from a context of 64 keys. Over fewer keys a KV head's K
  rows (k bits) or V columns (L bits), each read by its query heads, bind instead, and the check asks for a larger batch
  (70B's QKᵀ over 16 keys: 12);
- at prefill a KV head's K rows and V columns, read by all its query heads, bind, so it counts (KV head, request)
  components, far fewer the longer the prompt: a 16-token causal prompt needs 166 for QKᵀ (GRIND's table says 171, counting
  every Q row's readers as the prompt's L keys).

A bf16 linear on `DotBf16` carries QKᵀ's form under a scheme that keeps it for its family. A seed group never spans calls run
one after another (`span`): a seed binds its calls' operand roots, and layer ℓ + 1's operands exist only after layer ℓ has
run.

**The floor** (`Compiled.floor`). Per unit, by slot, the work its noise forces: its cells' dots where its kind's form
(`Scheme.form`: `Scheme.rows`, or `Scheme.noise` for a kind the grind term doesn't read) puts noise on both its rows at
its k, and k multiply-adds each; a kind with a clean side, or with no form, none. H1's s8 kinds carry `h1tile.H1_ROWS`:
an X row its k dither bits, a W column its k coin bits and U's 256. It is read off the plan, with no gate walked, with
each unit's batch node in the compiled Program's root for placement. It is the floor
`Accounting.Guarantees.PouwBlockCredit` attributes to the block a unit is placed at, `Wb · Nmin`
with `Wb` the least of the block's units' and `Nmin` the right units the draw established, and not every gate the salt
reaches: the seed's and the noise's SHA-512 and the receipt fold read it too, and are outside the floor.

**Check** (`CompileError.code`):

- `coin-read`: the served Program reads a coin (a coin parameter, a seed of its own, or it is already compiled);
- `edited-arm`, `arm`: a Call isn't the registered arm; `scan`, `operand-inside`: a matmul inside a scan, or on an operand
  computed inside its Definition;
- `uncovered`: a registered Call or a matmul step left outside every unit;
- `cover`: a seed group covers fewer units than the scheme's `cover`, or its dots of a kind with a `RowNoise` carry a grind
  term over 0.1%;
- `span`: a seed group holds a call and a call that reads its output;
- `salt-flow`: the salt reaches anything but a unit's seed;
- `position`: a unit's seed reads a position or roots other than its own parameters, or at the root a unit batch reads slots
  other than its own: no unit chooses its position, and no two share one;
- `seed`: a unit derives no seed, or more than one; `noise-in-z`: its z reads a node the seed reaches;
- `separator`: a unit read past its z; `width`: a node outside the units that commits more than the width rule allows;
- `name`, `scheme`: a parameter already named `pouw.*`, or a scheme or kind that isn't registered.

`Compiled.check_gates` adds the gate-level cut (no value computed in two units), the separator and the width rule by
reachability, for a Program small enough to flatten. Where the coin rules have their seed role (`ports.py`), the seed, each unit and
the compiled Program declare their position `In(kind="position")` beside the salt's coin port, and its Build check refuses
some units before compile's own checks do: two seeds at one position (`E_COIN_REUSE`), a position the unit computes
(`E_PORT_KIND`), a seed's noise reaching z or past the unit's sinks (`E_COIN_FLOW`).

**Schemes.** The pass's own `compile.SCHEMES`, separate from `circuit.SCHEMES`; `BUILTIN` loads this package's at the first
name not registered.

- `eps1-sm120-h1fx1` (`circuit/h1tile.py`, where `circuit/segment.py` is on the tree): sm_120's H1 + fx1 on a warp's 32 × 32
  tile, for the s8 kinds on bits, under CP-W (the `PouwWarpCheck_v2` body). Each lane folds H1's sets over
  X = A + b and W' = B + c + U with fx1 from its keys, and receipt words 2 s and 2 s + 1 are the warp's XOR of y and of z
  after set s: 2 (KS + m) words, 8,320 bits at K = 4096. Its noise is counter mode (`h1tile.Layout`): each row's and
  column's from blocks of seed_j's own, and each lane's fold keys, its chain's start, a block of σ_R's (`h1tile.fold_keys`).
  The edited GEMM kernel still starts its chains from the noise seed (`benchmarks/pouw/edited_gemm/cgemm.cu`'s
  ``ch[e].y = mix32(a.seed ^ …)``), which the σ_R rule above retires: a kernel follow-up keys them from σ_R. Whether a
  tile is a legal proof block (a unit's outputs fit a public size class up to 262,144 bits, so 8,320 fit), or the warp
  check must be a unit kind of its own aggregating coordinates, is still open.

The seed, the unit's two seeds and the noise block (`_v1` on words, `_v2` on bits), `PouwU129` and `PouwH1Tile` are
registered Definitions, each checked by circuit-check at its sidecar's binding (`seeds.circuit-check.py`,
`h1tile.circuit-check.py`); `PouwH1Tile` only where `circuit/segment.py` is on the tree, at one lane (4 × 8, K = 64).

Not here yet: P·V as a registered Call, and a staged split for a dot whose operands its holder computes; a padded tile for a
site that fills none (decode's few tokens); the vLLM integration serving the compiled Program.

## Dither (v3.2)

Dither, PoUW's no-fold protocol, credit for a served model's int8 matmuls from committed noised prefix sums, is specified
in `verity/core/protocols/pouw/dither/PROTOCOL.md` while it is experimental. The served PoUW moves to Dither's no-fold
receipt: its H1 and J10-R units fold their words into keyed receipts, whose detection rests on
`SharedNoiseLucks` (`Properties.Pouw.PeriodJ10R`), and that premise has no place in Dither.

## Not here yet

- The fast implementations: the RTX 4090 kernels and the vLLM adapter (`integrations/vllm/verity_vllm/protocols/pouw/__init__.py`)
  are gated by these vectors when they land. Pearl-C's sm_120 kernel is `benchmarks/pouw/pearl_c_sm120/`; its H100
  kernel (-h1, BLAKE3) was deleted with the archive in `2091252fe3`.
- The per-weight share of the honest reference's work in `wref`, and a `TargetProfile` Definition of the NCP GEMM.
- For Pearl-C's draw: `GγSampled`, `HonestTileCapRate` and `CompleteSampled` take tickets proportional to the whole
  tile's `Wref`, while the audit weighs a tile by its credited cells. The two agree on fully credited tiles. Closing it
  means Lean statements over credited-cell tickets, or tickets over every cell.
- C-Flock's side of the proof's identifier: the header's `identifier`, its binding through `pub.sha`, and the verifier's
  `--identifier`.
- For `ncp-v2`:
  - the lead domain's shapes (k ≥ 1,088) as a recorded run rather than a test;
  - C-Flock pieces for the PoUW primitives;
  - the floors POUS sets for the zero-work templates other than `dequant` (1 until then);
  - the window guarantee, `audit_window_split_of_record`, in this tree's Lean;
  - the audit for a draw that depends on the registration: the key's context is the receipt digest, which the `audit`
    game the guarantees state (the registration before the coin, `L.draw ω`) can't express. Either apply the guarantees
    strategy by strategy, at the context digest(reg σ), or state the audit for a draw `L.draw R ω` with A4 uniform over
    contexts;
  - a statement composing the calls' ε_ks and ε_bind into the window's (M2);
  - the served-output designation and the pinned ports in `Q_nested_instances` v0's spec, evaluator and Lean port;
  - the vLLM adapter's `traced_as`;
  - the reuse of a weight's Y proofs across a run's windows, which needs its commitments linked across windows;
  - NCP-FP8 and H-1T.
