# The remote bandwidth audit v0 (`verity/remote-bandwidth/v0`)

A verifier with no hardware at the site certifies, pass by pass, how many reads of a node's registered blocks the node
made. The node's memory is its GPUs' HBM and its host's DRAM. Every coin is a live coin from the challenger, committed
before the session; no challenge is derived from the prover's messages (no Fiat–Shamir). No prover hardware is trusted:
the host, its CPUs and its NICs are all part of the prover. The code is `audit.py` (the lifecycle), `coins.py` (the coin
stream), `receipts.py` (seeds, plans, receipts and trees), `codec.py` (the store's encoding) and `timing.py` (the
honest schedule). Every byte format below is pinned by `tests/vectors.json`.

This is a reference implementation: pure Python, with openings checked in the clear. The clear checks are the
specification of what each zero-knowledge proof unit proves on the deployed path (§8).

## 1. What an accepted session certifies

Let \(N\) be the store's blocks, \(b = 4096\) bytes each. A session is a stream of \(K = WP\) passes, grouped into \(W\)
windows of \(P\) for the openings. Pass \(k\)'s coin left the verifier at \(r_k\), and its root arrived by
\(D_k = r_k + A\). Pass \(k\) is a plan of \(L_k\) distinct blocks and their \(L_k\) receipts, each made after the coin
could reach the node and before the root left it: inside \(I_k = [r_k + d^{\downarrow}_{\text{floor}}, D_k -
d^{\uparrow}_{\text{floor}}]\), which is \(A - \text{RTT}_{\text{floor}}\) long. Except with probability \(\delta\) per
window, at most \(m^*_t\) of window \(t\)'s \(U_t = \sum_k L_k\) receipts are wrong (`IntegrityProfile.bound`).

`ledger` turns that into γ, given the node's measured capacity \(R\) (HBM plus host DRAM read bandwidth), its on-chip
storage \(S_{\text{chip}}\) and the trusted round-trip floor \(\text{RTT}_{\text{floor}}\). Over the session's span
\(\Sigma = D_{K-1} - r_0\), the reads of registered data are at least \((1 - \gamma) R \Sigma\), with

\[
\gamma = \mu_{\text{timing}} + \mu_{\text{overlap}} + \mu_{\text{sample}} + \mu_{\text{chip}} + \lambda + \mu_{\text{rebuild}} + \mu_{\text{host}} + \mu_{\text{link}} .
\]

| term | what it is | where it comes from |
| --- | --- | --- |
| \(\mu_{\text{timing}}\) | \(1 - \sum_t U_t b / (R\Sigma)\): the span the receipts don't fill (the coin interval a pass doesn't fill, the round trip at the session's ends, blocks a plan leaves out) | the record |
| \(\mu_{\text{overlap}}\) | \((\sum_k \lvert I_k\rvert - \lvert\bigcup_k I_k\rvert)/\Sigma\): the time in which one read may serve two passes' receipts (§5) | the record and \(\text{RTT}_{\text{floor}}\) |
| \(\mu_{\text{sample}}\) | \(\sum_t m^*_t\, b / (R\Sigma)\) | the record's profiles, at δ |
| \(\mu_{\text{chip}}\) | \(K S_{\text{chip}} / (R\Sigma)\): each pass, what is held on chip needn't be read again | \(S_{\text{chip}}\), measured |
| \(\lambda\) | AuditRO's slack: `queries_log2` bits per certified receipt and `slack_log2` per pass; 0.195% at 64 bits a 4 KiB block, more than \(\mu_{\text{sample}}\) | proved (`Bandwidth.Guarantees.AuditRO`, applied per pass) |
| \(\mu_{\text{rebuild}}\) | re-encoding blocks from their payloads with leftover compute | a model, on `WorkHardness(p, k)` |
| \(\mu_{\text{host}}\) | the host tier's own shortfall, and host DRAM bandwidth PCIe can't carry | a model, measured capacities |
| \(\mu_{\text{link}}\) | relays that meet \(z\) and a copy of the block over a link (§4) | a model |

The protocol bounds the first five and passes the last three through: the record can't bound them (a protocol carries
no policy parameter). The failure probability is \(W\delta\) for a session of \(W\) windows, plus \(2^{-\text{slack}}\)
per pass, plus \(s/2^{512}\) per window for guessing a receipt.

What the record says is conservation: correct receipts, at least \((1-\gamma)\) of the declared capacity. The
node-level statement, that this node's bandwidth went to registered data, also needs the identity assumption: no
undeclared tensor-class hardware takes part.

**The round-trip floor is an assumption.** \(\text{RTT}_{\text{floor}}\) is the round trip between the challenger and the
node's memory that no message can beat: the great-circle distance at the speed of light, or a named network assumption.
A measured round trip is not a floor, because the node controls its side of every measurement and can slow it. With no
trusted floor (`rtt_floor_ns = 0`, the default), \(\mu_{\text{overlap}}\) charges the whole round trip.

## 2. Roles and parameters

- **The prover** is the node. It registers a store, runs a pass per coin (inference reads every weight block and the
  live KV pages once a decode step, topped up with filler reads), sends each pass's plan with the root of its
  receipts, and opens what the verifier draws.
- **The challenger** is the auditor's online service. It draws every coin of a session from its own generator, commits
  to them before the session, and releases each once, on schedule (`coins.Challenger`).
- **The verifier** is remote. It holds a secret `source` (for the codec seed) and a clock, keeps the record, and checks.
  In the reference, `audit.Session` is the verifier and drives the challenger.

`Params`:

| field | meaning |
| --- | --- |
| `blocks` | \(N\), the registered store's blocks |
| `passes` | \(P\), the passes a window |
| `interval_ns` | \(\Delta\): pass \(k\)'s coin is due at \(r_0 + k\Delta\) |
| `deadline_ns` | \(A\): pass \(k\)'s root is due by \(r_k + A\) |
| `windows` | \(W\), the windows in a session |
| `samples` | \(s\), the openings per window |
| `delta` | δ of each window's profile |
| `rounds` | the codec's \(k \ge 2\) |

## 3. Registration

1. The prover commits to its payload blocks (`payload_tree`): the weights, the KV pages and filler, 4 KiB each. The
   commitment is a frame-v3-sha512 tree with one block per leaf, each an `hm96-sha512/row/v2` leaf:
   \(x_j\) = `sha512/row/v2`\((\text{payload}_j)\) (`SHA512(P(32768) ‖ payload_j)`), value
   `hm96.SHA512.tree_leaf(key, commit_string(key, x_j, y_j))` with \(y_j\) a fresh 192-byte salt and hm96-sha512's
   pinned key (`receipts.hidden`), in
   `CommitmentDomain(SHA512("verity/remote-bandwidth/payload/v0\0" ‖ frame(node)), 0, positions 0 … N − 1, "sha512")`.
   `node` is the node's id, 1 to 64 bytes. This is the leaf PoUS's answer circuit proves over a block.
2. Only after the payload root has arrived does the verifier draw the codec seed from its own secret and a fresh
   32-byte nonce: `derive(source, "verity/remote-bandwidth/codec/v0", {node, blocks, payload, nonce}).stream("codec", 32)`.
3. The prover encodes every block under that seed (`codec.py`): \(E_j = \text{encode}(\text{payload}_j)\), Sloth over
   \(p = 2^{512} - 569\) at \(k\) rounds, word \(w\) of block \(j\) at store index \(64j + w\). This is byte-exact with
   `verity/word-codec/kv/v0` at `p512`, the KV kernel's codec. For \(j = \text{slot} \cdot \text{LAYERS} + \text{layer}\),
   it is that codec's token-layer layout.
4. `Registration.digest = SHA512("verity/remote-bandwidth/registration/v0\0" ‖ frame(node) ‖ u64be N ‖ payload root ‖
   codec seed ‖ u8 k)`. A record cites this digest, and so does each session's coin commitment.

Three properties of registration matter:

- **The order is the point.** If the codec seed came before the payloads, the prover would register
  \(\text{payload}_j = \text{decode}(0)\) and hold a store of zeros. Drawn after, each \(E_j\) costs \(k\) square roots
  a word to produce, whatever the payload, all-zero weights included. That is the only rebuild bound: no
  incompressibility is assumed, and every bound is decode-free.
- **No commitment to \(E\) is needed.** ρ permutes the canonical words, so `decode` is a bijection from canonical
  blocks to payloads, and the payload root plus the codec seed fix \(E\). An opened block that isn't canonical fails
  (`codec`).
- **The node is in the codec seed.** Two nodes registering the same weights hold different stores, and the tweak
  \(\tau i\) separates every word of one store.

## 4. The coin stream and the passes

Before a session the challenger draws its \(K' = W(P + 1)\) coins and commits to them (`coins.py`), and the prover
receives the commitment before the first coin:

~~~text
c_k         32 bytes from the challenger's generator; pass k's coin at k < W·P, window t's openings coin at W·P + t
x_k         SHA512("verity/remote-bandwidth/coin/v0\0" ‖ c_k)
value_k     hm96-sha512 tree_leaf(key, commit_string(key, x_k, y_k)), y_k a fresh 192-byte salt, key hm96-sha512's pinned key
root        frame-v3-sha512 over value_0 … value_(K'−1): CommitmentDomain(SHA512("verity/remote-bandwidth/coins/v0\0" ‖
            registration digest ‖ u64be session), −1, positions 0 … K' − 1, "sha512"), schema "verity/remote-bandwidth/coins/v0"
opening     (k, c_k, y_k, path): checked by the prover against the root (`coin_reason`)
~~~

The challenger can't choose a coin after seeing a root, and hm96 keeps every coin hidden until it is released. This
is the construction of PoUW's epoch coins over a stream.

The verifier releases pass \(k\)'s coin at \(r_k \ge r_0 + k\Delta\) (`Session.release` refuses earlier), one-way: no
round trip waits on it. Pass \(k\) (`receipts.py`):

~~~text
seed_k      SHA512("verity/remote-bandwidth/seed/v0\0" ‖ c_k ‖ u64be k)
plan_k      R runs (first, count) in read order: count ≥ 1, inside the store, pairwise disjoint, no run starting where
            the one before it ends.  Position p is the p-th block of the runs concatenated; L_k positions in all.
            digest SHA512("verity/remote-bandwidth/plan/v0\0" ‖ u32be R ‖ (u64be first ‖ u64be count) × R)
y_(k,p)     SHA512(Z("verity/remote-bandwidth/receipt/v0\0" ‖ frame(seed_k)) ‖ u64be j ‖ (E_j ⊕ spread(z_(k,j)))),
            j = plan_k[p]
spread(z)   z over 4096 bytes: byte u of z XORed into byte u mod 4096 (folded), or a shorter z repeated
leaf c      v_c = y_(k,16c) ‖ … ‖ y_(k,16c+15), positions in plan order, the last leaf short; committed as
            hm96-sha512/row/v2: x_c = sha512/row/v2(v_c), value tree_leaf(key, commit_string(key, x_c, salt_c)),
            salt_c a fresh 192-byte salt
root_k      frame-v3-sha512, CommitmentDomain(SHA512("verity/remote-bandwidth/receipt-tree/v0\0" ‖ seed_k ‖ plan
            digest), 0, positions 0 … ⌈L_k/16⌉ − 1, "sha512"), schema "hm96-sha512/row/v2"
~~~

`Z` zero-pads to 128 bytes, so the seed fills the first SHA-512 block. No state over a block's bytes exists before the
seed, and the seed exists only once its coin has been released. No seed reads a root: the schedule alone orders the
passes (§5).

The prover sends each root with its pass's plan as the pass ends (`Session.receive`). A root for a coin not yet
released rejects with `early`, a second root for a pass with `duplicate`, a plan that breaks its format with `plan`.

**The plan** is the prover's choice and public. A decode step reads every weight block and only the live KV pages, and
a block being written in the pass can't be receipted as registered, so a pass need not cover the store; the record
credits each pass with its plan's \(L_k\) blocks, no more. Three rules make that sound:

- **Distinct blocks.** Two positions on one block would let one read serve two receipts. The verifier checks it on
  receipt, in \(O(R \log R)\) for \(R\) runs, so the plan must be in the clear. (Committing the block indices in
  increasing order beside the receipts and checking sampled neighbours doesn't enforce it: \(0, \dots, N-1, 0, \dots,
  N-1\) credits the store twice with one descent, a neighbour pair the draws find with probability about
  \(s/2N\).)
- **Bound to the root.** The tree's binding is the seed and the plan's digest, and its size is \(\lceil L_k/16\rceil\)
  leaves, so a root opens only under the plan sent with it.
- **Canonical.** No two consecutive runs continue each other, so a block sequence has one plan and one digest
  (`Plan.of` builds it from block indices, the prover-side registry's form).

The plan's order is the tree's, not necessarily the order of the reads: nothing in a receipt depends on its position,
so the prover may list a pass's blocks in store order whatever order it reads them in, which makes the plan one run
per contiguous range read (§8). It may choose the plan after reading, since it is credited only with what it receipts
and every receipt is sampled; what it can't do is credit one read twice.

**\(z_{(k,j)}\)** is PoUW's noised output of the tile the prover computes before reading block \(j\) in pass \(k\)
(`Tiles`). Three choices in it are load-bearing:

- **PoUW's noise for a pass is derived from the pass's seed,** so \(z\) differs every pass even when the weights and
  activations don't. Otherwise a pass's \(z\) would be known before its coin.
- **Each block of a pass has its own tile,** so distinct receipts bind independent \(z\) (a pass reads a block once).
  If one \(z\) served \(n\) receipts, shipping it across a link would unlock \(n b\) bytes of receipts, and
  \(\mu_{\text{link}}\)'s \(b/|z|\) would not hold.
- **\(z\) enters elementwise, before any compression.** The block and \(z\) must then meet in one place for each
  receipt, so a relay between them carries \(\min(|z|, b)\) bytes per block. Hashing a digest of \(z\) instead would let
  a 64-byte midstate cross.

`NoiseTiles` is the stub: SHA-512 in counter mode over the seed and \(j\), with empty evidence. Deployed, \(z\) comes
from Pearl-C's k-slice accumulators, inside Pearl-C's domain, and `recompute` is Pearl-C's check of a sampled tile (§8).

## 5. Deadlines, and what the schedule forces

Pass \(k\)'s root must arrive by \(D_k = r_k + A\), where \(r_k\) is the verifier's clock when \(c_k\) left. A root after
its deadline rejects with `late`.

**The tight deadline** is the coin's trip down, the pass, the root's reduction and its trip up, each trip at its longest
(`timing.deadline`):

\[
A = d^{\downarrow}_{\max} + t_{\text{pass}} + g + d^{\uparrow}_{\max} = \text{RTT} + 2J + t_{\text{pass}} + g ,
\]

with \(J\) each direction's jitter and \(g\) the root's reduction (the tree's last levels, an all-gather of the GPUs'
subtree roots, the send). It is a whole round trip, not half: the coin's one-way trip comes before the pass can start,
and the root's comes after it ends. The honest prover keeps up whenever \(\Delta \ge t_{\text{pass}}\): it starts pass
\(k\) once \(c_k\) is there and pass \(k-1\)'s reads are done, and reduces each root beside the next pass's reads, so it
is never late (`timing.honest`, replayed against `Session` in the tests).

**What the schedule forces.** A receipt of pass \(k\) is seed-first and \(\text{seed}_k\) needs \(c_k\), which can't reach
the node's memory before \(r_k + d^{\downarrow}_{\text{floor}}\). Every receipt of pass \(k\) is bound by its root, which
must leave the node by \(D_k - d^{\uparrow}_{\text{floor}}\). So the read of \(E_j\) behind receipt \((k, j)\) happens in

\[
I_k = [\, r_k + d^{\downarrow}_{\text{floor}},\; D_k - d^{\uparrow}_{\text{floor}} \,], \qquad \lvert I_k\rvert = A - \text{RTT}_{\text{floor}} ,
\]

or the block was held on chip since an earlier read (\(\mu_{\text{chip}}\)). A plan's blocks are distinct, so one read
serves at most one receipt of a pass. A read at time \(\tau\) therefore serves at most \(n(\tau)\) receipts, one for
each pass whose \(I_k\) contains \(\tau\), and the receipts' bytes exceed the reads by at most

\[
R \int (n(\tau) - 1)^+ \, d\tau = R\Big(\sum_k \lvert I_k\rvert - \Big\lvert \bigcup_k I_k \Big\rvert\Big) ,
\]

which is \(\mu_{\text{overlap}}\) (`audit.overlap_ns`). When \(\Delta \ge A - \text{RTT}_{\text{floor}}\), consecutive
passes' windows don't overlap, and each coin certifies at most one read of each planned block: \(\mu_{\text{overlap}} =
0\). A shorter \(\Delta\) is allowed and charged.

**The cost.** A pass credits \(t_{\text{pass}} R\) bytes every \(\Delta\), so in steady state
(`timing.mu`), with \(U = \text{RTT} + 2J - \text{RTT}_{\text{floor}}\) the round trip's uncertainty:

\[
\mu_{\text{timing}} + \mu_{\text{overlap}} = 1 - \frac{t_{\text{pass}}}{\Delta} + \frac{\max(0,\, A - \text{RTT}_{\text{floor}} - \Delta)}{\Delta}
= \begin{cases} (g + U)/\Delta & t_{\text{pass}} \le \Delta \le t_{\text{pass}} + g + U \\ 1 - t_{\text{pass}}/\Delta & \text{beyond,} \end{cases}
\]

least at \(\Delta^* = t_{\text{pass}} + g + U\), where it is \((g + U)/(t_{\text{pass}} + g + U)\). Every microsecond of
the round trip the verifier can't rule out is charged once a pass. §9 has the grid.

**No gap between windows.** Windows only group passes for the openings: the coin stream runs on across a window's end,
so the end costs nothing, and only the session's two ends do (\((A - \Delta)/\Sigma\)). Two sessions over one store
must not overlap: a session's first coin is due no earlier than the previous session's last deadline less
\(\text{RTT}_{\text{floor}}\), or their records would share reads that neither ledger sees.

## 6. Openings and the accept rule

Window \(t\) is passes \(tP, \dots, tP + P - 1\). Once their roots are all in, or the last one's deadline has passed (a
missing root then rejects the window, `missing`), the challenger releases the window's openings coin \(c_{WP + t}\),
and the verifier draws \(s\) distinct receipts uniformly from the window's \(U_t\) with it (`Session.close`):
`derive(c_(WP+t), "verity/remote-bandwidth/openings/v0", {registration, window: t}).subset(min(s, U_t), U_t,
"openings")`, with unit \(u\) as (pass, position): pass \(tP + i\) for the first \(i\) whose prefix sum
\(L_{tP} + \dots + L_{tP+i}\) exceeds \(u\), and the position \(u\) less the passes before it. The coin is released only
after every root it draws from is fixed, so the draw is independent of the roots.

The prover opens each draw (`Prover.open`, `Opening`): the block \(E_j\) at \(j = \text{plan}_k[p]\), its receipt leaf's
receipts \(v_c\) with the leaf's salt and path to \(\text{root}_k\), the payload's salt and path to the payload root,
and PoUW's evidence for \(z_{(k,j)}\). Openings aren't timed, because the roots already bind every receipt.

The verifier checks each opening in order (`Session.verify`):

1. the indices are the draw's and every field has its size (`malformed`);
2. the leaf, under \(\text{seed}_k\) from the released \(c_k\), the plan received with the root and the leaf's salt, has
   its path to \(\text{root}_k\) (`receipt-path`);
3. the block's words are canonical (`codec`), and their decoding's leaf under the payload salt has its path to the
   payload root at \(j\) (`payload-path`);
4. PoUW's check recomputes \(z_{(k,j)}\) from the evidence (`tile`);
5. \(y_{(k,p)}\), recomputed from \(j\), the block and \(z\), is the leaf's receipt at \(p\) (`receipt`).

A window is accepted only if every opening checks. A rejected window ends its session: every later call is refused,
and nothing of the session is recorded.

## 7. The record

`Session.record()` returns a `Record` once every window is accepted. `public()` holds the registration digest, the coin
commitment's root, \(N\), \(b\), each window's passes and receipts \(U_t\), and the session's span. Each `WindowRecord`
carries its passes' lengths \(L_k\), each coin's release \(r_k\) and deadline \(D_k\), its openings coin's release, and
an `IntegrityProfile`.

**How `IntegrityProfile` fits.** It fits the openings exactly. The one level is `receipt`, \(U_t\) members, with
`Subset(s)` drawn. That is valid because every pass's root and plan are committed before the openings coin is released,
and `bound()` is \(m^*_t\). A level of passes above it would only group (the subset is uniform over all \(U_t\)), and a
profile's hierarchy has one fan-out per level, which passes of different lengths don't have.

It doesn't fit the rest of γ: the timing, overlap, chip, rebuild, host and link terms aren't a sampling design over
units. So, as the core protocol interface provides for a protocol whose acceptance isn't only a sampling design, the
protocol's profile is its own: each window's `IntegrityProfile` with its passes' schedule. `ledger` computes every γ
term from it plus the node's capacities and the trusted floor.

## 8. What deployment changes, and what is open

- **Zero knowledge.** Every protocol is zero-knowledge, with no clear mode, and an opened block decodes to a KV page,
  which is user data. Deployed, each opening is a proof unit the proof service proves in zero knowledge, and the
  verifier sees only its result. The unit's statement is §6's checks 2–5, with the opening's fields as witness and
  public inputs the pass's root, its seed, its plan's digest, \(p\), \(j\), the payload root and the registration. The
  clear checks here are the specification of what each unit proves. The formats already hide: payload blocks and
  receipt leaves are `hm96-sha512/row/v2` leaves under fresh salts, and the coins are hm96 leaves until released, so the
  formats don't change when the openings move into proofs.
- **Where \(z\) comes from.** \(z\) comes from Pearl-C's k-slice accumulators, inside Pearl-C's domain, with Pearl-C's
  noise for the pass derived from the pass's seed. That gives \(|z|/|E| = 4m/k_s\), at least 1 for batch
  \(\ge k_s/4\), where the tile's final output would give \(4m/k\) (1/16 at batch 64). All-zero weights needn't be
  supported; where Pearl-C's accumulators don't apply, NCP's middle checkpoints are the fallback. `Tiles` stays generic:
  the reference takes \(z\) as bytes of any length.
- **The schedule.** Which tile precedes position \(p\) of a pass is the prover-side registry's (ra-vllm-hooks).
  `Tiles` takes it as given.
- **The plan on the wire.** The reference takes \(4 + 16R\) bytes a pass, which is too many for KV. Listed in store
  order, the weights are a run per region and a full pass is one run, but live KV pages are scattered through the
  pool. At worst (every other page live) a 60 GB pool of 64 KiB layer-pages (16 tokens, 8 KV heads of 128, bf16 K and
  V) is 458k runs, 7.3 MB a pass: 2.9 Gb/s per GPU at 20 ms passes, twice that for a plane-major cache. Deployed, a
  plan is sent as its change from the previous pass's plan. A decode step at batch 64 allocates or frees about 4
  pages a layer, 128 runs over 32 layers, about 4 KB a step, and the verifier updates its sorted copy and checks
  distinctness in \(O(\text{changes} \cdot \log R)\).
- **What the plan shows.** v0's plans are in the clear, so the verifier sees which KV pages were live in each pass: the
  batch's shape and context lengths over time, not their contents. Hiding them goes with the zero-knowledge openings:
  the plan committed, its distinctness and each opening's \(j = \text{plan}[p]\) proved, only \(L_k\) public (the draws
  need it).
- **KV registered mid-session.** A KV page's payload is chosen after the codec seed, so it can't be bound by
  registration order the way weights and filler are. It is bound only by integrity: sampled proofs prove it is the
  model's actual computation. The reference registers a fixed store. Appending pages needs their payload roots
  committed before the coins of the passes that read them, and fresh store indices. Until then, a block being rewritten
  stays out of every plan, as the prover-side registry already does with blocks a write has touched.
- **Multi-GPU trees.** Each GPU's blocks should be an aligned power-of-two range of leaves, so that a pass's root is
  its GPUs' subtree roots combined. That keeps \(g\) to an all-gather and a few hashes; the coin's broadcast to the
  GPUs is part of its trip down.
- **Sketch receipts** (a ±1 projection hashed, `RateAuditSketch`) are a cheaper receipt. Two rules apply to them:
  - The sketch's matrix must be derived from each pass's seed, never once per window. A per-window matrix makes each
    block's contribution a fixed digest: read once and kept, it serves every later pass's receipt (red team F29,
    art:61fc2e4c). With live coins, the pass's coin keys both the sketch and the hash.
  - \(z\) must enter before the sketch, because a linear sketch of \(E \oplus Z\) splits.

  The reference has only the full SHA-512 receipt, which the pass's seed keys from its first block.
- **The challenger's state.** The challenger keeps every coin of a session until it releases it and never releases one
  twice. Persisting that across challenger restarts, and committing a stream that outlives one session, is out of
  scope.
- **Keeping receipts for the draws.** Any receipt of a window may be drawn once its roots are in, so the prover keeps
  them, and the leaves' salts: 64 bytes a 4 KiB block and 192 a leaf of 16, about 1/60 of what it read in the window.
  Keeping each tree above its leaves and re-reading an opened leaf's 16 blocks instead costs a small fraction of the
  reads again (ra-openings puts it at 0.06–0.1%). The reference keeps every pass whole.
- **Lean.** The statements are ra-remote-theorem's (`Accounting.Remote`: `Conservation`, `NodeLevel` on the
  actual-circuit model), with the per-pass routes ra-remote-lean and ra-bound-lean prove. Nothing here is proved yet.
  What this code supplies is the schedule (each pass's coin released at \(r_k\), its root due at \(D_k\), the windows
  \(I_k\) and their overlap), the plan's distinctness, one tile a receipt, and the profile at δ.

## 9. The timing

`python -m verity.core.protocols.remote_bandwidth.timing` prints the grid: passes of 20 and 50 ms, RTT 0.2–50 ms,
jitter 0–1 ms each way, a trusted floor of 0, 0.5 or 0.9 of the RTT, and \(g\) = 50 µs. In steady state:

| pass | RTT | floor | at \(\Delta = t_{\text{pass}}\) | at \(\Delta^*\) |
| --- | --- | --- | --- | --- |
| 20 ms | 0.2 ms | none / 0.5 | 1.75% / 1.25% | 1.72% / 1.23% |
| 20 ms | 1 ms | none / 0.5 / 0.9 | 5.75% / 3.25% / 1.25% | 5.44% / 3.15% / 1.23% |
| 20 ms | 5 ms | none / 0.5 / 0.9 | 25.8% / 13.3% / 3.25% | 20.5% / 11.7% / 3.15% |
| 20 ms | 20 ms | none / 0.5 / 0.9 | vacuous / 50.8% / 10.8% | 50.2% / 33.7% / 9.71% |
| 50 ms | 1 ms | none / 0.5 / 0.9 | 2.30% / 1.30% / 0.50% | 2.25% / 1.28% / 0.50% |
| 50 ms | 5 ms | none / 0.5 / 0.9 | 10.3% / 5.30% / 1.30% | 9.34% / 5.03% / 1.28% |
| 50 ms | 20 ms | none / 0.5 / 0.9 | 40.3% / 20.3% / 4.30% | 28.7% / 16.9% / 4.12% |

with \(J\) = 0.05 ms each way. At 20 ms passes, 1 ms RTT and a floor of half of it, μ falls from 3.25% at
\(\Delta = t_{\text{pass}}\) to 3.15% at \(\Delta^*\) = 20.65 ms and rises again beyond (4.8% at 21 ms, 20% at 25 ms).
Jitter counts twice: 2.7%, 3.2%, 4.5%, 7.2% and 11.3% at \(J\) = 0, 0.05, 0.2, 0.5 and 1 ms. A pass reads distinct
blocks, so it is at most a read of the store: about 52 ms for a GPU's 94 GB at 1.8 TB/s.

**Sessions** of \(W\) windows of \(T\) = 1 s, simulated at \(\Delta^*\): 3.20% (\(W\) = 1), 3.15% (10) and 3.15% (100) at
20 ms passes, 1 ms RTT and floor 0.5. The window boundary has no cost of its own: the previous design paid
\(\text{RTT} + \text{tail}\) for every isolated window (2.1–6.6% at \(T\) = 1 s) and its sessions only their ends.

**For comparison, the chained seed.** Had each seed also read the previous pass's root (the coin still the only
randomness, the root there for order alone), pass \(k+1\)'s receipts couldn't start before pass \(k\)'s root existed,
however the passes' windows overlapped. Only \(g\) would go uncovered, \(g/(t_{\text{pass}} + g)\): 0.25% at 20 ms passes
and 0.10% at 50 ms, whatever the RTT and the floor. Under live coins alone, the order comes from the clock, and the
verifier pays for every microsecond of the round trip it can't rule out.

**Openings** at \(\varepsilon_s = 0.1\%\), δ = \(2^{-40}\): \(s = 27{,}726\) per window, about 8.5–9.1 KB each (block,
leaf, both salts, both paths). That is 235–253 MB a window per node: 1.9–2.0 Gb/s at \(T\) = 1 s, 0.19–0.20 Gb/s at
\(T\) = 10 s. PoUW's tile evidence comes on top unless PoUW's own sampled check covers the same tiles.

That is the per-window reading, each window's profile at its own δ. Read over a horizon instead, the total is what
counts: with every window drawing \(s\) after its own roots from live coins, the mean of the windows' wrong fractions is
below \(\bar d\) except with probability \((1-\bar d)^{\sum_t s}\), however the prover adapts each window to the draws
before it (ra-openings' causal horizon bound; a paper proof, adopted once it is in Lean). So 27,726 openings serve a
whole horizon: 47 a window over 10 minutes at \(T\) = 1 s, about 400 kB/s (3.2 Mb/s) whatever \(T\), and 68 kB/s over an
hour. The wrong receipts are then at most \(W \bar d \max_t U_t\). The reference records each window's profile, which
both readings start from; `ledger` takes the per-window one.
