# The remote bandwidth audit v0 (`verity/remote-bandwidth/v0`)

A verifier with no hardware at the site certifies, window by window, how many reads of a node's registered blocks the
node made. The node's memory is its GPUs' HBM and its host's DRAM. The verifier's draws come from its own secret
(live coins, no Fiat–Shamir). No prover hardware is trusted: the host, its CPUs and its NICs are all part of the
prover. The code is `audit.py` (the lifecycle), `receipts.py` (the seed chain, receipts and trees), `codec.py` (the
store's encoding) and `timing.py` (the honest schedule). Every byte format below is pinned by `tests/vectors.json`.

This is a reference implementation: pure Python, with openings checked in the clear. §8 lists what deployment changes.

## 1. What an accepted session certifies

Let \(N\) be the store's blocks, \(b = 4096\) bytes each. An accepted window \(t\) certifies \(P_t\) passes. Pass \(i\)
is a plan of \(L_i\) distinct blocks and their \(L_i\) receipts, made after the previous pass's root existed and before
this pass's root was sent; a pass over the whole store has \(L_i = N\). All \(P_t\) roots arrived in \([r_t, D_t]\) on
the verifier's clock, where \(r_t\) is when the window's challenge left. Except with probability \(\delta\) per
window, at most \(m^*_t\) of the window's \(U_t = \sum_i L_i\) receipts are wrong (`IntegrityProfile.bound`).

`ledger` turns that into γ, given the node's measured capacity \(R\) (HBM plus host DRAM read bandwidth) and on-chip
storage \(S_{\text{chip}}\). Over the session's span \(\Sigma = D_{n-1} - r_0\), the reads of registered data are at
least \((1 - \gamma) R \Sigma\), with

\[
\gamma = \mu_{\text{timing}} + \mu_{\text{sample}} + \mu_{\text{chip}} + \lambda + \mu_{\text{rebuild}} + \mu_{\text{host}} + \mu_{\text{link}} .
\]

| term | what it is | where it comes from |
| --- | --- | --- |
| \(\mu_{\text{timing}}\) | \(1 - \sum_t U_t b / (R\Sigma)\): the span the receipts don't fill (round trip, the tail, chaining gaps, blocks a plan leaves out) | the record |
| \(\mu_{\text{sample}}\) | \(\sum_t m^*_t\, b / (R\Sigma)\) | the record's profiles, at δ |
| \(\mu_{\text{chip}}\) | \(\sum_t P_t S_{\text{chip}} / (R\Sigma)\): each pass, what is held on chip needn't be read again | \(S_{\text{chip}}\), measured |
| \(\lambda\) | AuditRO's slack: `queries_log2` bits per certified receipt and `slack_log2` per pass | proved (`Bandwidth.Guarantees.AuditRO`, applied per pass) |
| \(\mu_{\text{rebuild}}\) | re-encoding blocks from their payloads with leftover compute | a model, on `WorkHardness(p, k)` |
| \(\mu_{\text{host}}\) | the host tier's own shortfall, and host DRAM bandwidth PCIe can't carry | a model, measured capacities |
| \(\mu_{\text{link}}\) | relays that meet \(z\) and a copy of the block over a link (§5) | a model |

The protocol bounds the first four and passes the last three through: the record can't bound them (a protocol carries
no policy parameter). The failure probability is \(n\delta\) for a session of \(n\) windows, plus \(2^{-\text{slack}}\)
per pass, plus \(s/2^{512}\) per window for guessing a receipt.

What the record says is conservation: correct receipts, at least \((1-\gamma)\) of the declared capacity. The
node-level statement, that this node's bandwidth went to registered data, also needs the identity assumption: no
undeclared tensor-class hardware takes part.

## 2. Roles and parameters

- **The prover** is the node. It registers a store, runs passes over it (inference reads every weight block and the
  live KV pages once a decode step, topped up with filler reads), sends each pass's plan, the blocks it read, with the
  root of its receipts, and opens what the verifier draws.
- **The verifier** is remote. It holds a secret `source` and a clock, draws every challenge, nonce, codec seed and
  opening from them, and checks.

`Params`:

| field | meaning |
| --- | --- |
| `blocks` | \(N\), the registered store's blocks |
| `window_ns` | \(T\): the verifier sends a boundary message every \(T\) |
| `rtt_allowance_ns` | the round-trip allowance in each deadline |
| `tail_ns` | the allowance for the pass in flight at a boundary (one pass plus the chaining gap) |
| `windows` | \(n\), the windows in a session |
| `samples` | \(s\), the openings per window |
| `delta` | δ of each window's profile |
| `min_passes` | the fewest passes a window may certify |
| `rounds` | the codec's \(k \ge 2\) |

## 3. Registration

1. The prover commits to its payload blocks (`payload_tree`): the weights, the KV pages and filler, 4 KiB each. The
   commitment is a frame-v3-sha512 tree with one block per leaf:
   `CommitmentDomain(SHA512("verity/remote-bandwidth/payload/v0\0" ‖ frame(node)), 0, positions 0 … N − 1, "sha512")`,
   with schema `verity/remote-bandwidth/payload/v0`. `node` is the node's id, 1 to 64 bytes.
2. Only after the payload root has arrived does the verifier draw the codec seed from its own secret and a fresh
   32-byte nonce: `derive(source, "verity/remote-bandwidth/codec/v0", {node, blocks, payload, nonce}).stream("codec", 32)`.
3. The prover encodes every block under that seed (`codec.py`): \(E_j = \text{encode}(\text{payload}_j)\), Sloth over
   \(p = 2^{512} - 569\) at \(k\) rounds, word \(w\) of block \(j\) at store index \(64j + w\). This is byte-exact with
   `verity/word-codec/kv/v0` at `p512`, the KV kernel's codec. For \(j = \text{slot} \cdot \text{LAYERS} + \text{layer}\),
   it is that codec's token-layer layout.
4. `Registration.digest = SHA512("verity/remote-bandwidth/registration/v0\0" ‖ frame(node) ‖ u64be N ‖ payload root ‖
   codec seed ‖ u8 k)`. A record cites this digest, and a session's first pass chains from it.

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

## 4. Sessions, windows and chained passes

A session is \(n\) windows. The verifier sends window 0's challenge \(\sigma_0\) (`Session.start`), and at each
window's end the next window's challenge, or the close after the last (`Session.boundary`).
\(\sigma_t = K_t.\text{stream}(\text{"challenge"}, 32)\) comes from the window key
\(K_t = \text{derive}(\text{source}, \text{"verity/remote-bandwidth/window/v0"}, \{\text{registration}, \text{window}: t, \text{nonce}\})\),
with a fresh 32-byte nonce per window. The challenge is fresh, not secret: the prover sees it. The key stays with the
verifier, and the openings are drawn from it.

Pass \(i\) of window \(t\) (`receipts.py`):

~~~text
seed_(t,i)  SHA512("verity/remote-bandwidth/seed/v0\0" ‖ σ_t ‖ u32be i ‖ root_(t,i−1))
            root_(t,−1) = root_(t−1,last); root_(0,−1) = Registration.digest
plan_(t,i)  R runs (first, count) in read order: count ≥ 1, inside the store, pairwise disjoint, no run starting where
            the one before it ends.  Position p is the p-th block of the runs concatenated; L_i positions in all.
            digest SHA512("verity/remote-bandwidth/plan/v0\0" ‖ u32be R ‖ (u64be first ‖ u64be count) × R)
y_(i,p)     SHA512(Z("verity/remote-bandwidth/receipt/v0\0" ‖ frame(seed_(t,i))) ‖ u64be j ‖ (E_j ⊕ spread(z_(i,p)))),
            j = plan_(t,i)[p]
spread(z)   z over 4096 bytes: byte u of z XORed into byte u mod 4096 (folded), or a shorter z repeated
leaf c      y_(16c) ‖ … ‖ y_(16c+15), positions in plan order, the last leaf short
root_(t,i)  frame-v3-sha512, CommitmentDomain(SHA512("verity/remote-bandwidth/receipt-tree/v0\0" ‖ seed_(t,i) ‖ plan
            digest), 0, positions 0 … ⌈L_i/16⌉ − 1, "sha512"), schema "verity/remote-bandwidth/receipts/v0"
~~~

`Z` zero-pads to 128 bytes, so the seed fills the first SHA-512 block. No state over a block's bytes exists before the
seed, and the seed exists only after the previous pass's root.

The prover sends each root with its pass's plan as the pass ends (`Session.receive`). On a boundary message, it
finishes the pass in flight, which is its window's last, and chains the next window's first pass from that pass's
root. A root out of sequence (a skipped index, a root for a window whose challenge hasn't been sent, a root for a
window already final) rejects with `order`; a plan that breaks its format rejects with `plan`.

**The plan** is the prover's choice and public. A decode step reads every weight block and only the live KV pages, and
a block being written in the pass can't be receipted as registered, so a pass need not cover the store; the record
credits each pass with its plan's \(L_i\) blocks, no more. Three rules make that sound:

- **Distinct blocks.** Two positions on one block would let one read serve two receipts. The verifier checks it on
  receipt, in \(O(R \log R)\) for \(R\) runs, so the plan must be in the clear. (Committing the block indices in
  increasing order beside the receipts and checking sampled neighbours doesn't enforce it: \(0, \dots, N-1, 0, \dots,
  N-1\) credits the store twice with one descent, a neighbour pair the draws find with probability about
  \(s/2N\).)
- **Bound to the root.** The tree's binding is the seed and the plan's digest, and its size is \(\lceil L_i/16\rceil\)
  leaves, so a root opens only under the plan sent with it, and the next pass's seed chains from both.
- **Canonical.** No two consecutive runs continue each other, so a read sequence has one plan and one digest
  (`Plan.of` builds it from block indices in read order, the prover-side registry's form).

The prover may choose the plan after reading, since it is credited only with what it receipts and every receipt is
sampled; what it can't do is credit one read twice.

**\(z_{(i,p)}\)** is PoUW's noised output of the tile the prover computes before the pass's \(p\)-th read (`Tiles`).
Three choices in it are load-bearing:

- **PoUW's noise for a pass is derived from the pass's seed,** so \(z\) differs every pass even when the weights and
  activations don't (all-zero weights). Otherwise a pass's \(z\) would be known before its seed.
- **Each position has its own tile,** so distinct receipts bind independent \(z\). If one \(z\) served \(n\)
  receipts, shipping it across a link would unlock \(n b\) bytes of receipts, and \(\mu_{\text{link}}\)'s \(b/|z|\)
  would not hold.
- **\(z\) enters elementwise, before any compression.** The block and \(z\) must then meet in one place for each
  receipt, so a relay between them carries \(\min(|z|, b)\) bytes per block. Hashing a digest of \(z\) instead would let
  a 64-byte midstate cross.

`NoiseTiles` is the stub: SHA-512 in counter mode over the seed and \(p\), with empty evidence. Deployed, `output` is
the GPU's tile output and `recompute` is PoUW's check of a sampled tile.

## 5. Deadlines

Window \(t\)'s roots must all arrive by \(D_t = b_t + \text{RTT} + \text{tail}\), where \(b_t\) is the verifier's clock
when it sent the window's boundary message. A root after its window's deadline rejects with `late`.

Two things make the deadline mode-independent:

- **The tail allowance covers the pass in flight,** so the honest prover never stops early. A boundary costs it
  nothing, and the session pays \(\text{RTT} + \text{tail}\) once, at its end.
- **The chain makes overlap harmless.** Window \(t+1\)'s challenge leaves at \(b_t\), before \(D_t\), but its first
  pass needs window \(t\)'s last root. So consecutive windows' passes can't share reads.

An isolated window is a session of one window. The next isolated window can start only at the previous deadline: two
unchained windows open at once would let one read of a block serve both windows' receipts.

## 6. Openings and the accept rule

Once window \(t\)'s roots are final (the next window's first root has arrived, or \(D_t\) has passed), the verifier
draws \(s\) distinct receipts uniformly from the window's \(U_t\) (`Session.close`):
`K_t.subset(min(s, U_t), U_t, "openings")`, with unit \(u\) as (pass, position): pass \(i\) is the first whose prefix
sum \(L_0 + \dots + L_i\) exceeds \(u\), and the position is \(u\) less the passes before it. A window with fewer than
`min_passes` roots is rejected (`few`).

The prover opens each draw (`Prover.open`, `Opening`): the block \(E_j\) at \(j = \text{plan}_{(t,i)}[p]\), its receipt
leaf with the leaf's path to \(\text{root}_{(t,i)}\), the payload's path to the payload root, and PoUW's evidence for
\(z_{(i,p)}\). Openings aren't timed, because the roots already bind every receipt.

The verifier checks each opening in order (`Session.verify`):

1. the indices are the draw's and every field has its size (`malformed`);
2. the leaf, under \(\text{seed}_{(t,i)}\), which it computes from \(\sigma_t\) and the received roots, and the plan
   received with the root, has its path to \(\text{root}_{(t,i)}\) (`receipt-path`);
3. the block's words are canonical (`codec`), and their decoding has its path to the payload root at \(j\)
   (`payload-path`);
4. PoUW's check recomputes \(z_{(i,p)}\) from the evidence (`tile`);
5. \(y_{(i,p)}\), recomputed from \(j\), the block and \(z\), is the leaf's receipt at \(p\) (`receipt`).

A window is accepted only if every opening checks. A rejected window ends its session: every later call is refused,
and nothing of the session is recorded.

## 7. The record

`Session.record()` returns a `Record` once every window is accepted. `public()` holds the registration digest, \(N\),
\(b\), each window's passes and receipts \(U_t\), and the session's span. Each `WindowRecord` carries its passes'
lengths \(L_i\), \(r_t\), \(D_t\) and an `IntegrityProfile`.

**How `IntegrityProfile` fits.** It fits the openings exactly. The one level is `receipt`, \(U_t\) members, with
`Subset(s)` drawn. That is valid because every pass's root and plan are committed before the draw, and `bound()` is
\(m^*_t\). A level of passes above it would only group (the subset is uniform over all \(U_t\)), and a profile's
hierarchy has one fan-out per level, which passes of different lengths don't have.

It doesn't fit the rest of γ: the timing, chip, rebuild, host and link terms aren't a sampling design over units. So,
as the core protocol interface provides for a protocol whose acceptance isn't only a sampling design, the protocol's
profile is its own: each window's `IntegrityProfile` with its passes and span. `ledger` computes every γ term from it
plus the node's capacities.

## 8. What deployment changes, and what is open

- **Zero knowledge.** Every protocol is zero-knowledge, and an opened block decodes to a KV page, which is user data.
  Deployed, each opening check (§6, items 2–5) is a statement the proof service proves in zero knowledge, with hm96
  leaves in place of the clear payload leaves, and the verifier sees only the statement's result.
- **Where \(z\) enters.** Binding the tile's output gives \(|z|/|E| = 4m/k\): 1/16 at batch 64 and 1/128 at batch 8,
  so at small-batch decode a relay saves far more than it ships. Binding \(k\)-slice accumulators gives \(4m/k_s\),
  at least 1 for batch \(\ge k_s/4\). That needs PoUW's conjecture for partial accumulators. The reference takes
  \(z\) as bytes of any length.
- **The schedule.** Which tile precedes position \(p\) of a pass is the prover-side registry's (ra-vllm-hooks).
  `Tiles` takes it as given.
- **The plan on the wire.** \(4 + 16R\) bytes a pass. A decode step's plan is one run for the weights (if they are
  registered in read order), a run or a few per KV layer's live pages, and the filler's: hundreds of bytes per pass,
  so tens of kB/s at 20–150 ms passes, beside the openings' Gb/s. The verifier's check is \(O(R \log R)\) a pass.
- **What the plan shows.** The plan is in the clear, so the verifier sees which KV pages were live in each pass: the
  batch's shape and context lengths over time, not their contents. Hiding it means committing the plan and proving its
  distinctness and each opening's \(j = \text{plan}[p]\) in zero knowledge, with only \(L_i\) public (the draws need
  it). That is open.
- **KV registered mid-session.** A KV page's payload is chosen after the codec seed, so it can't be bound by
  registration order the way weights and filler are. It is bound only by integrity: sampled proofs prove it is the
  model's actual computation. The reference registers a fixed store. Appending pages needs their payload roots in the
  chain and fresh store indices. Until then, a block being rewritten stays out of every plan, as the prover-side
  registry already does with blocks a write has touched.
- **Multi-GPU trees.** Each GPU's blocks should be an aligned power-of-two range of leaves, so that a pass's root is
  its GPUs' subtree roots combined. That keeps the chaining gap to an all-gather and a few hashes.
- **Sketch receipts** (a ±1 projection hashed, `RateAuditSketch`) are a cheaper receipt. Two rules apply to them:
  - The sketch's matrix must be derived from each pass's chained seed, never once per window. A per-window matrix
    makes each block's contribution a fixed digest: read once and kept, it serves every later pass's receipt (red team
    F29, art:61fc2e4c).
  - \(z\) must enter before the sketch, because a linear sketch of \(E \oplus Z\) splits.

  The reference has only the full SHA-512 receipt, which the pass's seed keys from its first block.
- **The challenger's state.** v0's on-site audit keeps a challenger state on disk so that no nonce repeats across
  restarts. Here every nonce is drawn fresh from the verifier's randomness, and one session over a registration is
  open at a time. Persisting that across verifier restarts is out of scope.
- **Lean.** Nothing here is stated in Lean yet. The guarantee to state is §1 over chained passes: AuditRO per pass, the
  chain's ordering, and the profile at δ.

## 9. What chaining forces, and the timing

Without the chain, one challenge serves the whole window: every pass's seed is known at once, and one read of
\(E_j\) serves every pass's receipt. A window then forces one read of the store, whatever \(P\) is. With the chain,
pass \(i+1\)'s receipts need \(\text{root}_i\), which needs every receipt of pass \(i\). So a window forces \(P_t\)
passes, one after another, each over its plan's distinct blocks. On-chip storage carries at most \(S_{\text{chip}}\)
across each pass boundary (\(\mu_{\text{chip}}\)), so pass \(i\) reads at least \(L_i b - S_{\text{chip}}\) bytes; a
prover gains nothing from short passes, since each costs another \(S_{\text{chip}}\).

The chain costs a gap \(g\) per pass: the root's reduction, an all-gather across GPUs, the next seed and its broadcast,
about 25–100 µs. Prefetching the next pass's first blocks on chip during the gap hides up to \(S_{\text{chip}}/R\)
(about 146 µs on the RTX PRO 6000), which \(\mu_{\text{chip}}\) already charges.

`python -m verity.core.protocols.remote_bandwidth.timing` prints the grid over RTT 1–50 ms, \(T\) 0.1–10 s and passes
of 20–150 ms. With a 100 µs gap:

- **Passes per window:** \(P = \lceil T / (t_{\text{pass}} + g) \rceil\), from 2 (150 ms passes, \(T\) = 0.3 s) to 498
  (20 ms, 10 s).
- **Sessions:** \(\mu_{\text{timing}} \approx (\text{RTT} + t_{\text{pass}})/(nT) + g/(t_{\text{pass}} + g)\), for
  example 0.50–0.55% (20 ms passes) and 0.20–0.25% (50 ms passes) at \(T\) = 1 s, \(n\) = 100.
- **Isolated windows:** they pay \(\text{RTT} + \text{tail}\) every window: 2.1–6.6% at \(T\) = 1 s with 20 ms
  passes, and 0.6–1.5% only at \(T\) = 10 s.

**Openings** at \(\varepsilon_s = 0.1\%\), δ = \(2^{-40}\): \(s = 27{,}726\) per window, about 8.1–8.7 KB each (block,
leaf, both paths). That is 224–242 MB a window per node: 1.8–1.9 Gb/s at \(T\) = 1 s, 0.19 Gb/s at \(T\) = 10 s.
PoUW's tile evidence comes on top unless PoUW's own sampled check covers the same tiles. Drawing once per session over
all its passes divides the bandwidth by \(n\), at the price of one profile per session instead of per window.
