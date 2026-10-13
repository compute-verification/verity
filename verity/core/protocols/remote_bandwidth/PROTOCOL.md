# The remote bandwidth audit v0 (`verity/remote-bandwidth/v0`)

A verifier with no hardware at the site certifies, window by window, how many reads of a node's registered blocks the
node made. The node's memory is its GPUs' HBM and its host's DRAM. No prover hardware is trusted: the host, its CPUs
and its NICs are all part of the prover. Each window's challenge and its openings coin are live coins from the
challenger, committed before the session. Inside a window the passes' seeds are Fiat–Shamir: each is derived from the
window's challenge and the prover's previous root, so the passes are chained. Daniel allowed that for this protocol on
12 Oct, because a live coin a pass is impractical at its rate (the friction skill's rulings). Appendix A is the
live-coin design and what it would cost.

The code is `audit.py` (the lifecycle), `coins.py` (the coins), `receipts.py` (seeds, plans, receipts and trees),
`codec.py` (the store's encoding) and `timing.py` (the honest schedule, and where the sketch's bound is read). Every byte format below is pinned by
`tests/vectors.json`. This is a reference implementation: pure Python, with openings checked in the clear. The clear
checks are the specification of what each zero-knowledge proof unit proves on the deployed path (§8).

## 1. What an accepted session certifies

Let \(N\) be the store's blocks, \(b = 4096\) bytes each. A session has \(W\) windows. Window \(t\)'s challenge
\(\sigma_t\) left the verifier at \(b_{t-1}\) (\(b_{-1}\) is the session's start). Its boundary message left at
\(b_t\), and its roots arrived by \(D_t = b_t + A\). Window \(t\) is \(P_t\) chained passes. Pass \((t, i)\) is a plan
of \(L_{t,i}\) distinct blocks and their receipts. Every receipt was made after the root before it existed and before
its own root left the node. So every read serves at most one receipt of the session (§5). Except with probability
\(\delta\) per window, at most \(m^*_t\) of window \(t\)'s \(U_t = \sum_i L_{t,i}\) receipts are wrong
(`IntegrityProfile.bound`). Pooled over the session's windows instead, the openings bound the wrong receipts of all
of them at once, and each window's share, except with \(\delta\) for the session (§10).

`ledger` turns that into γ, given the node's measured capacity \(R\) (HBM plus host DRAM read bandwidth) and its
on-chip storage \(S_{\text{chip}}\). Over the session's span \(\Sigma = D_{W-1} - b_{-1}\), the reads of registered
data are at least \((1 - \gamma) R \Sigma\), with

\[
\gamma = \mu_{\text{timing}} + \mu_{\text{sample}} + \mu_{\text{chip}} + \lambda + \mu_{\text{rebuild}} + \mu_{\text{host}} + \mu_{\text{link}} .
\]

| term | what it is | where it comes from |
| --- | --- | --- |
| \(\mu_{\text{timing}}\) | \(1 - \sum_t U_t b / (R\Sigma)\): the span the receipts don't fill (the gap \(g\) after each pass, the session's two ends, blocks a plan leaves out) | the record |
| \(\mu_{\text{sample}}\) | \(\sum_t m^*_t\, b / (R\Sigma)\) read per window (`Ledger.sample`), or the pooled bound's wrong receipts \(\cdot\, b / (R\Sigma)\) read over the horizon (`Ledger.horizon`, §10) | the record's profiles, at δ |
| \(\mu_{\text{chip}}\) | \(\sum_t P_t S_{\text{chip}} / (R\Sigma)\): each pass, what is held on chip needn't be read again | \(S_{\text{chip}}\), measured |
| \(\lambda\) | AuditRO's slack: `queries_log2` bits per receipt (every certified one, in either reading) and `slack_log2` per pass; 0.195% at 64 bits a 4 KiB block, more than \(\mu_{\text{sample}}\) | proved (`Bandwidth.Guarantees.AuditRO`, applied per pass) |
| \(\mu_{\text{rebuild}}\) | re-encoding blocks from their payloads with leftover compute | a model, on `WorkHardness(p, k)` |
| \(\mu_{\text{host}}\) | the host tier's own shortfall, and host DRAM bandwidth PCIe can't carry | a model, measured capacities |
| \(\mu_{\text{link}}\) | under a registration that binds \(z\) only (0 otherwise): relays that meet \(z\) and a copy of the block over a link (§4) | a model |

The protocol bounds the first four and passes the last three through: the record can't bound them (a protocol carries
no policy parameter). The failure probability is \(W\delta\) for a session of \(W\) windows read per window, or
\(\delta\) read over the horizon, plus \(2^{-\text{slack}}\) per pass, plus \(s/2^{512}\) per window for guessing a
receipt. Under the default receipt kind, sketch-sha512, a receipt also rests on `RateAuditSketch`, which adds at most
\(C \cdot 2^{-95.12}\) a session for a prover of \(C\) SHA-512 compressions (`timing.sketch_failure`, §4): no more
than \(2^{-40}\) up to \(C = 2^{55.12}\). block-sha512, kept for comparison, adds nothing. Neither adds to γ.

**Each window alone.** The same terms over window \(t\)'s own span \(\Sigma_t = D_t - b_{t-1}\) (`WindowRecord.span_ns`),
with its \(U_t\), \(P_t\) and \(m^*_t\) (or the pooled bound's \(f_t U_t\)) in place of the session's sums, give
\(\gamma_t\): window \(t\)'s reads are at least \((1 - \gamma_t) R \Sigma_t\) (`Ledger.windows`, a `WindowLedger`
each). Every read behind window \(t\)'s receipts falls in \(\Sigma_t\): its chain starts from \(\sigma_t\), which left
at \(b_{t-1}\), and its roots arrived by \(D_t\). Consecutive spans overlap by \(A\), but no read serves two windows'
receipts (§5). The every-window claim is \(\max_t \gamma_t\) (`Ledger.worst`, and `Ledger.worst_horizon` read over
the horizon), which the ledger reports beside the session's γ. A window the prover leaves idle is its own
\(\gamma_t\) there, where the session's γ sees about \(1/W\) of it. Read per window, the failure probability is the
same \(W\delta\); read over the horizon, every window's \(f_t\) holds on the session's one event (§10). A window is
charged \(A\) and the rest of the pass in flight at its challenge, where the session is charged them once, so its
\(\mu_{\text{timing}}\) is up to about \((A + t_{\text{pass}})/(T + A)\) (§10).

What the record says is conservation: correct receipts, at least \((1-\gamma)\) of the declared capacity. The prover
is one undifferentiated resource of capacity \(R\) (Daniel, 12 Oct, #1884): the guarantee is over whatever reads the
node made, with no identity assumption about which hardware made them.

No round-trip floor is assumed. The chain orders a session's passes by itself, so the round trip appears only at the
session's two ends (§5). The live-coin design needs a trusted floor to be affordable at all (Appendix A).

## 2. Roles and parameters

- **The prover** is the node. It registers a store and runs chained passes (inference reads every weight block and the
  live KV pages once a decode step, topped up with filler reads). It sends each pass's plan with the root of its
  receipts, and opens what the verifier draws.
- **The challenger** is the auditor's online service. It draws every coin of a session from its own generator, commits
  to them before the session, and releases each once (`coins.Challenger`).
- **The verifier** is remote. It holds a secret `source` (for the codec seed) and a clock, keeps the record, and checks.
  In the reference, `audit.Session` is the verifier and drives the challenger.

`Params`:

| field | meaning |
| --- | --- |
| `blocks` | \(N\), the registered store's blocks |
| `deadline_ns` | \(A\): window \(t\)'s roots are due by \(b_t + A\) |
| `windows` | \(W\), the windows in a session |
| `samples` | \(s\), the openings per window |
| `delta` | δ of each window's profile |
| `min_passes` | the fewest passes a window may certify |
| `codec` | the store's codec, the one its registration names (§3): `p1024/k3/masked` unless another is named |
| `receipt` | the receipt kind, the one its registration names (§4): `sketch-sha512` unless `block-sha512` is named |
| `bind` | whether receipts bind PoUW's tile output \(z\), as its registration says (§4): no unless it binds them |

The window's length \(T = b_t - b_{t-1}\) is the verifier's to choose, window by window. It is not a parameter.

## 3. Registration

1. The prover commits to its payload blocks (`payload_tree`): the weights, the KV pages and filler, 4 KiB each. The
   commitment is a frame-v3-sha512 tree with one block per leaf, each an `hm96-sha512/row/v2` leaf:
   \(x_j\) = `sha512/row/v2`\((\text{payload}_j)\) (`SHA512(P(32768) ‖ payload_j)`), value
   `hm96.SHA512.tree_leaf(key, commit_string(key, x_j, y_j))` with \(y_j\) a fresh 192-byte salt and hm96-sha512's
   pinned key (`receipts.hidden`), in
   `CommitmentDomain(SHA512("verity/remote-bandwidth/payload/v0\0" ‖ frame(node)), 0, positions 0 … N − 1, "sha512")`.
   `node` is the node's id, 1 to 64 bytes. This is the leaf PoUS's answer circuit proves over a block.
2. Only after the payload root has arrived does the verifier draw the codec seed from its own secret and a fresh
   32-byte nonce: `derive(source, "verity/remote-bandwidth/codec/v0", {node, blocks, payload, codec, nonce}).stream("codec",
   32)`, `codec` the name of the codec the registration names. The registration also names its receipt kind and
   whether its receipts bind \(z\) (§4), neither in this context: the store depends on neither.
3. The prover encodes every block under that seed with that codec (`codec.py`, below):
   \(E_j = \text{encode}(\text{payload}_j)\).
4. `Registration.digest = SHA512("verity/remote-bandwidth/registration/v0\0" ‖ frame(node) ‖ u64be N ‖ payload root ‖
   codec seed ‖ frame(codec name) ‖ frame(receipt kind) ‖ u8 bind)`, `frame` being `verity.ml.randomness.frame` ("s" ‖
   u64be length ‖ the name in UTF-8) and bind 1 if the receipts bind \(z\), else 0. A record cites this digest, each
   session's coin commitment binds it, and each session's chain starts from it, so a session's coins and seeds are
   bound to its kind and to whether it binds \(z\).

**The codec** is Sloth over \(p = 2^{1024} - 105\), with the mask \(\sigma_K\) between rounds, at \(k = 3\): `p1024/k3/masked`,
`codec.DEFAULT` (Daniel, 12 Oct, #1884). The fallback is \(k = 2\) at the same width, `p1024/k2/masked` (`codec.FALLBACK`).
The prime is kv/v0's at 1,024 bits and the one ra-squaring-ceiling measured. \(p = 2^{512} - 569\) unmasked (`p512/k2`)
stays available as the upgrade path. A registration may name any of them (or more rounds), and the session's `Params`
must name the same. Each is byte-exact with `verity/word-codec/kv/v0`, the KV kernel's codec (ra-codec-mix's `kv.py`,
#1894, `key(..., mask=True)`), and the tests pin `kv.py`'s vectors at k = 2 and k = 3 masked and at `p512/k2`:

~~~text
word        128 bytes, little-endian, canonical (< p); a 4 KiB block is 32 words, word w of block j at store index
            i = 32j + w, j below 2^59 so every i fits a u64 (at p512: 64 words of 64 bytes, i = 64j + w, j below 2^58)
ρ(a)        s = a^((p+1)/4); if s² = a the even one of {s, p − s}, else (s² = −a) the odd one
ρ⁻¹(y)      y² if y is even, p − y² if y is odd
XOF(m)      SHA512(m ‖ u32be 0) ‖ SHA512(m ‖ u32be 1) ‖ …
c_r         LE(XOF(D ‖ u8 5 ‖ "p1024" ‖ seed ‖ u32be r)[0 : 136]) mod p, r = 0 … k − 1, D = "verity/word-codec/kv/v0\0"
τ           the same at r = 2^32 − 1, and 1 in place of 0;  t(i) = τ · i mod p
K           LE(XOF(D ‖ u8 5 ‖ "p1024" ‖ seed ‖ u32be 2^32 − 2)[0 : 128]): 1,024 bits, not reduced mod p
σ_K(x)      x ⊕ K if that is below p, else x: an involution on [0, p)
encode      from the payload word, for r = 0 … k − 1: x ↦ σ_K(x) if r ≥ 1, then x ↦ ρ(x + c_r + t(i))
decode      for r = k − 1 … 0: x ↦ ρ⁻¹(x) − c_r − t(i), then x ↦ σ_K(x) if r ≥ 1
~~~

For \(j = \text{slot} \cdot \text{LAYERS} + \text{layer}\) a block is kv/v0's token-layer: a token's keys and values in
one layer, 4,096 bytes, are 32 words at 1,024 bits. Encoding a block costs \(32k\) square roots of 1,021 squarings each,
96 at the default, and decoding it \(32k\) squarings and \(32(k-1)\) masks. A payload word at or above \(p\) is refused:
its top 16 bits are all ones, a bf16 NaN (and its top byte 0xFF, an E4M3 NaN), so every block of finite values encodes.
\(k = 1\) is refused, because a word can then be partly kept and completed for less than an encode (red team F27).

Three properties of registration matter:

- **The order is the point.** If the codec seed came before the payloads, the prover would register
  \(\text{payload}_j = \text{decode}(0)\) and hold a store of zeros. Drawn after, each \(E_j\) costs \(k\) square roots
  a word to produce, whatever the payload, all-zero weights included. That is the only rebuild bound: no
  incompressibility is assumed, and every bound is decode-free.
- **No commitment to \(E\) is needed.** ρ and \(\sigma_K\) permute the canonical words, so `decode` is a bijection from
  canonical blocks to payloads, and the payload root plus the codec seed fix \(E\). An opened block that isn't canonical fails
  (`codec`).
- **The node is in the codec seed.** Two nodes registering the same weights hold different stores, and the tweak
  \(\tau i\) separates every word of one store.

## 4. The coins and the chained passes

Before a session the challenger draws its \(2W\) coins and commits to them (`coins.py`), and the prover receives the
commitment before the first coin:

~~~text
c_k         32 bytes from the challenger's generator; window t's challenge σ_t = c_t, its openings coin c_(W+t)
x_k         SHA512("verity/remote-bandwidth/coin/v0\0" ‖ c_k)
value_k     hm96-sha512 tree_leaf(key, commit_string(key, x_k, y_k)), y_k a fresh 192-byte salt, key hm96-sha512's pinned key
root        frame-v3-sha512 over value_0 … value_(2W−1): CommitmentDomain(SHA512("verity/remote-bandwidth/coins/v0\0" ‖
            registration digest ‖ u64be session), −1, positions 0 … 2W − 1, "sha512"), schema "verity/remote-bandwidth/coins/v0"
opening     (k, c_k, y_k, path): checked by the prover against the root (`coin_reason`)
~~~

The challenger can't choose a coin after seeing a root, and hm96 keeps every coin hidden until it is released. This
is the construction of PoUW's epoch coins.

The verifier releases \(\sigma_0\) at the session's start (`Session.start`). At window \(t\)'s end it sends the boundary
message, which is \(\sigma_{t+1}\), or the close after the last window (`Session.boundary`). Each is one-way: no round
trip waits on it. The prover refuses an openings coin as a challenge (`Prover.run`). Pass \(i\) of window \(t\)
(`receipts.py`):

~~~text
seed_(t,i)  SHA512("verity/remote-bandwidth/seed/v0\0" ‖ σ_t ‖ u32be i ‖ root_(t,i−1)),
            root_(t,−1) = root_(t−1,last) in a session, and the registration's digest before its first window
plan_(t,i)  R runs (first, count) in read order: count ≥ 1, inside the store, pairwise disjoint, no run starting where
            the one before it ends.  Position p is the p-th block of the runs concatenated; L positions in all.
            digest SHA512("verity/remote-bandwidth/plan/v0\0" ‖ u32be R ‖ (u64be first ‖ u64be count) × R)
e_(i,j)     E_j, the block alone; under a registration that binds z, E_j ⊕ spread(z_(i,j))
y_(i,p)     sketch-sha512, the default: SHA512(Z("verity/remote-bandwidth/sketch-receipt/v0\0" ‖ frame(seed_(t,i))) ‖
            u64be j ‖ σ_(i,j)),  j = plan_(t,i)[p]
            block-sha512, for comparison: SHA512(Z("verity/remote-bandwidth/receipt/v0\0" ‖ frame(seed_(t,i))) ‖ u64be j ‖
            e_(i,j))
S_(t,i)     sketch-sha512's 16 × 4096 signs: K = derive(seed_(t,i), "verity/remote-bandwidth/sketch-signs/v0",
            {"tag": "sketch-sha512", "k": 16, "n": 4096}), row c the first 512 bytes of K.stream(("row", c)), bit u
            (byte u // 8, LSB first) 1 → −1 and 0 → +1
σ_(i,j)     S_(t,i) · e, e the 4096 bytes of e_(i,j) as int8, over the integers (|σ[c]| ≤ 2^19): 16 int32
            little-endian, 64 bytes
spread(z)   z over 4096 bytes: byte u of z XORed into byte u mod 4096 (folded), or a shorter z repeated
leaf c      v_c = y_(i,16c) ‖ … ‖ y_(i,16c+15), positions in plan order, the last leaf short; committed as
            hm96-sha512/row/v2: x_c = sha512/row/v2(v_c), value tree_leaf(key, commit_string(key, x_c, salt_c)),
            salt_c a fresh 192-byte salt
root_(t,i)  frame-v3-sha512, CommitmentDomain(SHA512("verity/remote-bandwidth/receipt-tree/v0\0" ‖ seed_(t,i) ‖ plan
            digest), 0, positions 0 … ⌈L/16⌉ − 1, "sha512"), schema "hm96-sha512/row/v2"
~~~

`Z` zero-pads to 128 bytes, so the seed fills the first SHA-512 block. No state over a block's bytes exists before the
seed, and the seed exists only once the root before it does. The pass index stays in the seed, so two passes never
share one even if their previous roots collide.

The prover sends each root with its pass's plan as the pass ends (`Session.receive`). A root out of turn rejects with
`order`: a window's roots come in index order, and a window's first root makes the window before it final. A plan
that breaks its format rejects with `plan`.

**Fiat–Shamir, and what grinding buys.** A seed after the first depends on the prover's own root, so the prover can
grind: re-salting a leaf gives a new root and a new next seed, at the cost of a leaf and its path and no read. A
chosen seed buys nothing against the block-sha512 receipt, because no seed makes a receipt cheaper to compute without
the block's bytes after the seed. The queries a grinding prover makes are counted by AuditRO's `queries_log2` bits a
certified receipt (λ). Other users of the seed must hold for a seed the prover chose: sketch-sha512's signs (below),
and under a registration that binds \(z\), PoUW's noise for the pass (§8).

**The two receipt kinds.** The registration names one (`receipts.KINDS`), and `Params.receipt` must name the same.
sketch-sha512 is the default (`receipts.DEFAULT`) and the stated receipt: the receipt the guarantee is stated for
(Daniel, 12 Oct, #1884). block-sha512 is kept as a second kind, for comparison only; it is not the stated receipt.
sketch-sha512 is ra-codec-tensor's `sketch_receipt.py` (#1916, at 5b8decfc5), byte for byte: the suite pins its numpy
outputs, and `receipts.SketchReceipter` computes the same in pure Python. Each kind's receipt is 64 bytes, so leaves,
salts and the tree don't change. The difference is what is hashed after the seed's block: sketch-sha512 hashes \(j\)
and the 64-byte sketch, one compression; block-sha512 hashes \(j\) and the 4,096 bytes, 33 SHA-512 compressions. The
sketch's signs come from the pass's seed, so like the seed they exist only once the root before it does.

What each kind rests on:

- **block-sha512** needs only the random oracle. A correct receipt is a query at the block's bytes after the seed,
  so it bounds the reads by AuditRO (`Bandwidth.Guarantees.AuditRO`, applied per pass), with λ's slack. It assumes
  nothing about the device: the prover may hold any function of the store.
- **sketch-sha512** also rests on `RateAuditSketch` (ra-rate-audit-lean, #1849, open). The random oracle takes a
  correct receipt to a correct \(\sigma_j\). `RateAuditSketch` takes a correct \(\sigma_j\) to the block's 128
  sectors of 32 bytes having been obtained, except for a guess. A query made without one of them is right with
  chance at most \(g/2^{256}\) per query, \(g\) the largest fiber of one sector under the signs. The signs bind
  that far except with \(\Pr_S[g < \text{fib}_S]\). Four conditions come with it:
  - **The verbatim class.** The device obtains a sector only whole and as it is (`rate-sketch-verbatim`). A device
    that holds blocks' sketches, or a host that computes sketches once the signs are out and relays them, is outside
    the class. Window isolation must keep such a host off the link.
  - **Signs fresh per pass.** \(S_{(t,i)}\) comes from each pass's chained seed, never once per window (F29, §8).
    The statement's window is this protocol's pass.
  - **Grinding.** The statement takes the signs uniform. Here they are derived from a seed the prover can choose
    (above), so the bound counts every key the prover sees, ground or not (below).
  - **The 16 rows.** At \(k_s = 16\) no \(16 \times 32\) block of signs is injective (Siegel's lemma). So
    `SketchBinds` doesn't apply, and the bound is read at a rank \(r\) of every sector's block of signs: \(r^* = 12\),
    where it costs \(2^{-95.12}\) a SHA-512 compression of the prover's (below). At \(k_s = 64\) the signs bind except
    for a column dependency (`RateAuditTensorSketch`), at 256 bytes a block, and \(r^* = 27\).

  Under the default the sketch is of \(E_j\) alone, which is #1849's digest of the tile, \(\text{dig}_S(e_t)\), so
  the statement applies as it is. Binding \(z\) is the registration's opt-in (`bind`), PoUW's join (Daniel, 12 Oct:
  the guarantee is conservation over one undifferentiated resource, with no identity to bind). Under it \(z\) enters
  before the sketch, \(e = E_j \oplus \text{spread}(z)\), as §8's third rule requires, and XOR with a fixed \(z\)
  permutes each sector, so each sector's fiber is the same. But #1849's digest is of the tile alone, so that form
  needs #1849 restated over the block and \(z\) together, and it is not part of the stated guarantee.

**Where sketch-sha512's bound is read** (`timing.sketch_reading`, `timing.sketch_failure`, pinned by
`tests/test_sketch_reading.py`). The field is \(\mathbb{Q}\). The sketch's sums never wrap (\(|\sigma[c]| \le
2^{19}\)), so \(\sigma\) is \(S e\) over the integers. If sector \(u\)'s \(k_s \times 32\) block of signs \(S_u\) has
rank \(r\) over \(\mathbb{Q}\), then \(r\) of the sector's bytes are fixed by the others and by \(S_u e_u\), so each
of its fibers has at most \(256^{32-r}\) members, and a guess of \(S_u e_u\) is right with chance at most
\(2^{-8r}\).

The rank's law is bounded through \(\mathbb{F}_2\). Multiply each column of a uniform \(\pm 1\) matrix by its first
row's sign, and replace every other row \(v\) by \((\text{first} - v)/2\). Both keep the rank over \(\mathbb{Q}\), and
they leave the all-ones row over a uniform 0/1 matrix \(B\). That matrix's rank over \(\mathbb{Q}\) is at least its
rank over \(\mathbb{F}_2\), which is \(1 + \text{rank}\,U\), where \(U\) is \(B\) with its first column added to the
others and then dropped: a uniform \((k_s - 1) \times 31\) matrix over \(\mathbb{F}_2\). Its law is exact: an
\(m \times n\) matrix over \(\mathbb{F}_2\) has rank \(s\) in \(\prod_{i<s} (2^m - 2^i)(2^n - 2^i)/(2^s - 2^i)\) of
the \(2^{mn}\) cases. So

\[
\Pr[\text{rank}_{\mathbb{Q}} S_u \le r] \;\le\; \Pr[\text{rank}_{\mathbb{F}_2} U \le r - 1] .
\]

| \(r\), 16 rows | 15 | 14 | 13 | 12 | 11 | 10 |
| --- | --- | --- | --- | --- | --- | --- |
| \(\log_2\) of the bound on \(\Pr[\text{rank} \le r]\) | −16.00 | −34.59 | −55.39 | −78.30 | −103.25 | −130.23 |

| \(r\), 64 rows | 31 | 30 | 29 | 28 | 27 | 26 |
| --- | --- | --- | --- | --- | --- | --- |
| \(\log_2\) of the bound on \(\Pr[\text{rank} \le r]\) | −32.00 | −66.58 | −103.39 | −142.30 | −183.25 | −226.23 |

The bound is exact at 2 rows, \(2^{1-n}\), and the suite checks it against every \(\pm 1\) matrix up to \(4 \times 4\)
and \(3 \times 5\). At 16 rows it is loose at full rank: \(2^{-16}\), where two rows equal up to sign already give
about \(\binom{16}{2} 2^{-31} = 2^{-24.1}\). Where it is read, it costs under one bit (next).

**Grinding, and \(r^*\).** Every key is a fresh oracle output, the derive of a fresh seed, so its signs are uniform
whatever the prover did to reach it. A prover of \(C\) SHA-512 compressions over a session sees at most \(C + P\)
keys, \(P\) the session's passes, and makes at most \(C\) guesses. Read at rank \(r\), by the union bound over keys,
guesses and the block's 128 sectors,

\[
F(r) \;\le\; C \cdot 2^{-8r} + (C + P) \cdot 128 \cdot \Pr[\text{rank} \le r - 1] ,
\]

so its cost a compression is \(\rho(r) = 2^{-8r} + 128 \Pr[\text{rank} \le r - 1]\), and \(r^*\) minimizes it. At 16
rows \(r^* = 12\): the guess's \(2^{-96}\) beside \(128 \cdot 2^{-103.25} = 2^{-96.25}\), so \(\rho(r^*) =
2^{-95.12}\). At \(r = 11\), \(\rho = 2^{-88}\); at 13, \(2^{-71.30}\). At 64 rows \(r^* = 27\) and \(\rho(r^*) =
2^{-215.85}\).

| rows | \(r^*\) | \(\rho(r^*)\) | \(C\) at \(F = 2^{-40}\) | \(C\) at \(F = 2^{-64}\) |
| --- | --- | --- | --- | --- |
| 16 | 12 | \(2^{-95.12}\) | \(2^{55.12}\): 8.1 days of 8 GPUs | \(2^{31.12}\): 42 ms of 8 GPUs |
| 64 | 27 | \(2^{-215.85}\) | \(2^{175.85}\) | \(2^{151.85}\) |

The GPUs are RTX PRO 6000s at \(7.0 \cdot 10^9\) SHA-512 compressions a second each, block-sha512's measured rate
(#1916, below). \(C\) counts every compression the prover makes, on any hardware: the grinding needs no access to the
store. So at 16 rows a session stays below \(2^{-40}\) up to \(2^{55.12}\) compressions: \(2^{-50.19}\) for one node's
10-minute session, \(2^{-43.02}\) for a node-day, and \(2^{-40}\) for about 9,400 such GPUs grinding through a
10-minute session. \(2^{-64}\) is out of reach at 16 rows (42 ms of one node's hashing) and needs 64 rows, whose cost
on the GPU isn't measured.

A grinding try is not a pass's commitment. It reads no block: re-salting one receipt leaf changes the root without
changing any receipt. As the reference computes it, a try is 15 + 3⌈log₂ n⌉ compressions over \(n\) leaves
(`timing.grind_compressions`): the leaf's salt digest, the hm96 key's digest and its tree leaf (9), the leaf's hash
(2), its path (3 a level), the next seed (2) and its signs' key (2). That is 66 compressions, 9.4 ns on one GPU, for
an 8 GiB pass of 131,072 leaves, so a window of \(T\) = 1 s affords a node of 8 GPUs about \(2^{29.7}\) tries. A
prover that keeps the constant prefixes' midstates needs at least 5 compressions a try (the root's last, and the
seed's and the key's 2 each, whose first blocks hold the root and the seed): at most \(2^{33.4}\) tries a second on
that node. Seeing one sector's \(16 \times 32\) signs takes 16 more, one stream block a row. The bound doesn't rest on
any of these counts: it charges a key to every compression.

None of this moves γ: \(F\) adds to the failure probability, not to the reads certified. Daniel's 1% target (§10:
short by under 1% in every window, at \(2^{-40}\)) holds. For one node's 10-minute horizon, \(F = 2^{-50.19}\), so
keeping the total at \(2^{-40}\) takes the openings' δ to \(2^{-40} - 2^{-50.19}\), and their count up by about
0.003%. It breaks only past \(2^{55.12}\) compressions in a session, where \(F\) alone reaches \(2^{-40}\). The
ledger doesn't compute \(F\): it is a failure term, not a term of γ (`ledger`).

What each costs on the GPU (#1916): one RTX PRO 6000 (sm_120, 2,092 MHz), an 8 GiB pass of 2,097,152 blocks, best of
3, run r20261012-224926-17ce at 5b8decfc5 (art:dea3a309). τ is a pass's time over one plain read of the same
bytes, 5.60 ms.

| kind | receipts | leaves and tree | τ a pass | τ, binding a 4 KiB \(z\) on every block |
| --- | --- | --- | --- | --- |
| sketch-sha512 | 5.82 ms for the sketch (1,475 GB/s, memory-bound) and 0.33 ms for the hash | 0.90 ms | 1.26 | 2.27 |
| block-sha512 | 9.87 ms, ALU-bound (33 compressions a block, \(7.0 \cdot 10^9\)/s) | 0.90 ms | 1.93 | 2.26 |

Fused into the weight GEMM, sketch-sha512 is estimated at about 0.27 reads a pass, or 0.11 if the commit hides under
the GEMM's memory time; that is estimated, not measured. The GEMM's sketch warps add 0.053 at a 70B model, batch 8, and
the hash, leaves and tree were measured at 0.22. The reference's pure-Python sketch takes about 0.25 ms a block, against
0.013 ms for block-sha512's hash. That cost is the reference's, and the GPU's is the table's.

**The plan** is the prover's choice and public. A decode step reads every weight block and only the live KV pages, and
a block being written in the pass can't be receipted as registered, so a pass need not cover the store; the record
credits each pass with its plan's \(L\) blocks, no more. Three rules make that sound:

- **Distinct blocks.** Two positions on one block would let one read serve two receipts. The verifier checks it on
  receipt, in \(O(R \log R)\) for \(R\) runs, so the plan must be in the clear. (Committing the block indices in
  increasing order beside the receipts and checking sampled neighbours doesn't enforce it: \(0, \dots, N-1, 0, \dots,
  N-1\) credits the store twice with one descent, a neighbour pair the draws find with probability about
  \(s/2N\).)
- **Bound to the root.** The tree's binding is the seed and the plan's digest, and its size is \(\lceil L/16\rceil\)
  leaves, so a root opens only under the plan sent with it.
- **Canonical.** No two consecutive runs continue each other, so a block sequence has one plan and one digest
  (`Plan.of` builds it from block indices, the prover-side registry's form).

The plan's order is the tree's, not necessarily the order of the reads: nothing in a receipt depends on its position,
so the prover may list a pass's blocks in store order whatever order it reads them in, which makes the plan one run
per contiguous range read (§8). It may choose the plan after reading, since it is credited only with what it receipts
and every receipt is sampled; what it can't do is credit one read twice.

**\(z_{(i,j)}\)**, under a registration that binds \(z\) only, is PoUW's noised output of the tile the prover computes
before reading block \(j\) in pass \(i\) (`Tiles`). A registration that doesn't bind it takes no tiles: its receipts
are of the block alone, and its openings carry no tile evidence. Three choices in the binding form are load-bearing:

- **PoUW's noise for a pass is derived from the pass's seed,** so \(z\) differs every pass even when the weights and
  activations don't. Otherwise a pass's \(z\) would be known before its seed.
- **Each block of a pass has its own tile,** so distinct receipts bind independent \(z\) (a pass reads a block once).
  If one \(z\) served \(n\) receipts, shipping it across a link would unlock \(n b\) bytes of receipts, and
  \(\mu_{\text{link}}\)'s \(b/|z|\) would not hold.
- **\(z\) enters elementwise, before any compression.** The block and \(z\) must then meet in one place for each
  receipt, so a relay between them carries \(\min(|z|, b)\) bytes per block. Hashing a digest of \(z\) instead would let
  a 64-byte midstate cross.

`NoiseTiles` is the stub: SHA-512 in counter mode over the seed and \(j\), with empty evidence. Deployed, \(z\) comes
from Pearl-C's k-slice accumulators, inside Pearl-C's domain, and `recompute` is Pearl-C's check of a sampled tile (§8).

## 5. Deadlines, and what the chain forces

**What the chain forces.** A receipt of pass \((t, i)\) is seed-first, and \(\text{seed}_{(t,i)}\) is unpredictable
until \(\text{root}_{(t,i-1)}\) is fixed (in the random-oracle model, to a prover within λ's queries). The receipts
under \(\text{root}_{(t,i-1)}\) were all computed before it was fixed. So the reads behind pass \((t, i)\)'s correct
receipts happen after every read behind pass \((t, i-1)\)'s, and the chain runs on across windows. A plan's blocks are
distinct, so one read serves at most one receipt of a pass, and therefore at most one receipt of the session, unless
the block was held on chip since an earlier read (\(\mu_{\text{chip}}\)). There is no overlap term inside a session:
the order comes from the hashes, not the clock.

**The deadline.** Window \(t\)'s roots must all arrive by \(D_t = b_t + A\), where \(b_t\) is the verifier's clock when
the boundary message left. A root after its window's deadline rejects with `late`. The tight allowance is the
boundary message's trip down, the pass in flight, its root's reduction and its trip up, each trip at its longest
(`timing.deadline`):

\[
A = d^{\downarrow}_{\max} + t_{\text{pass}} + g + d^{\uparrow}_{\max} = \text{RTT} + 2J + t_{\text{pass}} + g ,
\]

with \(J\) each direction's jitter and \(g\) the gap a pass waits on: the root's reduction (the tree's last levels, an
all-gather of the GPUs' subtree roots, the send), the next seed and its broadcast to every GPU. It is a whole round
trip, not half. The boundary message's one-way trip comes before the prover knows the window has ended, the pass it
has just started must finish, and its root's trip comes after. The honest prover runs passes back to back,
\(t_{\text{pass}} + g\) apart. On the boundary message it finishes the pass in flight, which is its window's last, and
starts the next window's chain under \(\sigma_{t+1}\) from that pass's root. It is never late: in integer nanoseconds
its last root arrives at most \(A - 1\) after the boundary, which happens when the pass in flight starts the
nanosecond before the message arrives (`timing.honest`, replayed against `Session` in the tests).

The deadlines don't make the chain sound. What they do is place window \(t\)'s reads in
\([b_{t-1} + d^{\downarrow}_{\text{floor}}, D_t - d^{\uparrow}_{\text{floor}}]\), so each window certifies a rate, and
fix the window's roots before its openings are drawn.

**The cost.** Only \(g\) goes uncovered between passes, so in steady state
\(\mu_{\text{timing}} = g/(t_{\text{pass}} + g)\) (`timing.chained`): 0.25% at 20 ms passes and 0.10% at 50 ms, whatever
the RTT. A window's end costs nothing, because the chain runs on through the boundary. The round trip appears only at
the session's two ends: \(\sigma_0\)'s trip down before the first pass, and after the last boundary the slack of the
pass in flight and its root's trip up, together at most \(A + J\). So
\(g/(t_{\text{pass}} + g) \le \mu_{\text{timing}} \le g/(t_{\text{pass}} + g) + (A + J)/\Sigma\). §10 has the grid.

**Between sessions.** Sessions over one store are not chained to each other: each session's chain starts from the
registration's digest. So a session's first coin leaves no earlier than the previous session's last deadline,
`Record.end_ns`, less the trusted round-trip floor if there is one, and the store's first session's no earlier than
the verifier's clock at registration. Every `Session` names that time (`not_before_ns`, required: a session built
without it is refused) and refuses an earlier `start`. Otherwise two sessions could share reads that neither ledger
sees: one read a pass serving a receipt of each chain, both accepted, twice the reads credited. Two records over one
registration can be checked for it: the later's first release (its first window's `released_ns`) is at or after the
earlier's last deadline, less the floor. (Starting a session's chain from the
previous session's last root would remove the wait, at the cost of tying each session's verification to the one
before it.)

## 6. Openings and the accept rule

Window \(t\)'s roots are final once the next window's first root has arrived, or \(D_t\) has passed. A window with
fewer than `min_passes` roots then rejects with `few`. Otherwise the challenger releases the window's openings coin
\(c_{W+t}\), and the verifier draws \(s\) distinct receipts uniformly from the window's \(U_t\) with it
(`Session.close`): `derive(c_(W+t), "verity/remote-bandwidth/openings/v0", {registration, window: t}).subset(min(s,
U_t), U_t, "openings")`. Unit \(u\) is (pass, position): pass \(i\) for the first \(i\) whose prefix sum
\(L_{t,0} + \dots + L_{t,i}\) exceeds \(u\), and the position \(u\) less the passes before it. Windows are drawn in
order. The coin is released only after every root it draws from is fixed, so the draw is independent of the roots.

**Why a coin, not Fiat–Shamir, for the openings.** Drawing from the window's last root instead (`derive` keyed by
\(\text{root}_{(t,\text{last})}\)) is also sound in the random-oracle model, but the prover can grind it. Re-salting the
last leaf re-rolls the draws for the price of a leaf and its path, so a prover with wrong receipts tries \(Q\) roots
and keeps one whose draws miss them. By the union bound the window's δ becomes \(Q\delta\), so \(s\) must rise to the
count at \(\delta/Q\) (`timing.samples_for`). A try reads no block and costs a few to a few tens of compressions (§4
counts a seed's), so a window of \(T\) = 1 s affords one node of 8 GPUs about \(2^{30}\) to \(2^{34}\) of them. At
\(\varepsilon_s = 0.1\%\) and δ = \(2^{-40}\):

| grinding \(Q\) | \(s\) | × the coin's |
| --- | --- | --- |
| none (the coin) | 27,726 | 1.00 |
| \(2^{20}\) | 41,589 | 1.50 |
| \(2^{40}\) | 55,452 | 2.00 |
| \(2^{64}\) (λ's budget) | 72,088 | 2.60 |

It would also void the horizon reading (§10), whose proof takes each window's draw to be independent of what the
prover did in that window, unless that is re-proved for a grinding prover. The coin costs one message a window, sent
after the roots, which the verifier is waiting for anyway. So the protocol keeps it.

The prover opens each draw (`Prover.open`, `Opening`): the block \(E_j\) at \(j = \text{plan}_{(t,i)}[p]\), its receipt
leaf's receipts \(v_c\) with the leaf's salt and path to \(\text{root}_{(t,i)}\), the payload's salt and path to the
payload root, and under a registration that binds \(z\), PoUW's evidence for \(z_{(i,j)}\). Openings aren't timed,
because the roots already bind every receipt, and the next window's passes run while they are checked.

The verifier checks each opening in order (`Session.verify`):

1. the indices are the draw's and every field has its size, the tile evidence empty unless the registration binds
   \(z\) (`malformed`);
2. the leaf, under \(\text{seed}_{(t,i)}\) (from \(\sigma_t\), \(i\) and the root received before
   \(\text{root}_{(t,i)}\)), the plan received with the root and the leaf's salt, has its path to
   \(\text{root}_{(t,i)}\) (`receipt-path`);
3. the block, read as the registered codec's words, has every word below \(p\) (`codec`), and its decoding's leaf under
   the payload salt has its path to the payload root at \(j\) (`payload-path`);
4. under a registration that binds \(z\), PoUW's check recomputes \(z_{(i,j)}\) from the evidence (`tile`);
5. \(y_{(i,p)}\), recomputed from \(j\) and \(e_{(i,j)}\) (the block, with \(z\) if the registration binds it) under
   the registration's kind, is the leaf's receipt at \(p\) (`receipt`). Under sketch-sha512 the verifier derives the
   pass's signs \(S_{(t,i)}\) from \(\text{seed}_{(t,i)}\), once for each pass it draws from, and recomputes
   \(\sigma_j\) from \(e_{(i,j)}\) before hashing it. It takes no sketch from the prover: the opening is the same under
   either kind.

**What the codec changes in check 3.** Under `p512/k2` a block was 64 words of 64 bytes, each below \(2^{512} - 569\),
and each word's decode two squarings. Under the default it is 32 words of 128 bytes, each below \(2^{1024} - 105\): a
word is refused if it is one of the 105 values from \(p\) to \(2^{1024} - 1\). Word \(w\)'s index is \(32j + w\), and
its decode is three inverted rounds, each a squaring with the sign rule less \(c_r + t(i)\), with \(\sigma_K\) after the
rounds \(r = 2\) and \(r = 1\) (XOR with \(K\), kept only if the result is below \(p\)). A block is then 96 squarings mod
a 1,024-bit prime and 64 masks, where it was 128 mod a 512-bit prime. In the proof unit (§8), where a product's cost
grows about as the square of its width, that is about three times the decode check's multiplications; the fallback is
64 squarings and 32 masks, about twice. The opening is the same size: a block is 4,096 bytes under every codec.

A window is accepted only if every opening checks. A rejected window ends its session: every later call is refused,
and nothing of the session is recorded. A pass whose seed is not the chain's (passes of a window run at once, a window
restarted from the registration, a window's passes under the previous window's challenge) fails check 2 at its first
opening.

## 7. The record

`Session.record()` returns a `Record` once every window is accepted. `public()` holds the registration digest, the coin
commitment's root, \(N\), \(b\), the receipt kind, whether its receipts bind \(z\), the codec's name and parameters
(`codec`, `codec_params`: the field \(p = 2^n - c\), \(k\) and the mask), each window's passes and receipts \(U_t\), and
the session's span. The digest binds the kind, `bind` and the codec already, but the record names them as well: what
its receipts rest on depends on the first two (§4), and `ledger` takes a link term only from a record that binds
\(z\); \(\mu_{\text{rebuild}}\)'s model is priced under the codec (§3), so a record is re-checked against the model's
codec by name. Each
`WindowRecord` carries its passes' lengths \(L_{t,i}\), its challenge's release \(b_{t-1}\), its deadline \(D_t\) (so
its span \(\Sigma_t\), `span_ns`), its openings coin's release, and an `IntegrityProfile`. That is every input of
\(\gamma_t\) but the node's measured \(R\) and \(S_{\text{chip}}\) and the models, so the every-window claim is
checkable from the record (§1). `Record.end_ns` is the last deadline, which the next session over the store waits
out.

**How `IntegrityProfile` fits.** It fits the openings exactly. The one level is `receipt`, \(U_t\) members, with
`Subset(s)` drawn. That is valid because every pass's root and plan are committed before the openings coin is released,
and `bound()` is \(m^*_t\). A level of passes above it would only group (the subset is uniform over all \(U_t\)), and a
profile's hierarchy has one fan-out per level, which passes of different lengths don't have.

It doesn't fit the rest of γ: the timing, chip, rebuild, host and link terms aren't a sampling design over units. So,
as the core protocol interface provides for a protocol whose acceptance isn't only a sampling design, the protocol's
profile is its own: each window's `IntegrityProfile` with its schedule. `ledger` computes every γ term, the session's
and each window's, from it plus the node's capacities.

## 8. What deployment changes, and what is open

- **Zero knowledge.** Every protocol is zero-knowledge, with no clear mode, and an opened block decodes to a KV page,
  which is user data. Deployed, each opening is a proof unit the proof service proves in zero knowledge, and the
  verifier sees only its result. The unit's statement is §6's checks 2, 3 and 5 (and 4 under a registration that binds
  \(z\)), with the opening's fields as witness and
  public inputs the pass's root, its seed, its plan's digest, \(p\), \(j\), the payload root and the registration. The
  clear checks here are the specification of what each unit proves. The formats already hide: payload blocks and
  receipt leaves are `hm96-sha512/row/v2` leaves under fresh salts, and the coins are hm96 leaves until released, so the
  formats don't change when the openings move into proofs.
- **Where \(z\) comes from,** under a registration that binds it. \(z\) comes from Pearl-C's k-slice accumulators,
  inside Pearl-C's domain, with Pearl-C's noise for the pass derived from the pass's seed. That gives \(|z|/|E| = 4m/k_s\), at least 1 for batch
  \(\ge k_s/4\), where the tile's final output would give \(4m/k\) (1/16 at batch 64). All-zero weights needn't be
  supported; where Pearl-C's accumulators don't apply, NCP's middle checkpoints are the fallback. `Tiles` stays generic:
  the reference takes \(z\) as bytes of any length. The seed is Fiat–Shamir, so Pearl-C's check must hold for noise
  from a seed the prover chose among \(Q\).
- **The schedule.** Under a registration that binds \(z\), which tile precedes position \(p\) of a pass is the
  prover-side registry's (ra-vllm-hooks).
  `Tiles` takes it as given. A window ends on the verifier's boundary message, not after a count of passes: a prover
  that stops at a count idles until the message, which \(\mu_{\text{timing}}\) charges, and one whose count runs past
  the message is late.
- **The plan on the wire.** The reference takes \(4 + 16R\) bytes a pass, which is too many for KV. Listed in store
  order, the weights are a run per region and a full pass is one run, but live KV pages are scattered through the
  pool. At worst (every other page live) a 60 GB pool of 64 KiB layer-pages (16 tokens, 8 KV heads of 128, bf16 K and
  V) is 458k runs, 7.3 MB a pass: 2.9 Gb/s per GPU at 20 ms passes, twice that for a plane-major cache. Deployed, a
  plan is sent as its change from the previous pass's plan. A decode step at batch 64 allocates or frees about 4
  pages a layer, 128 runs over 32 layers, about 4 KB a step, and the verifier updates its sorted copy and checks
  distinctness in \(O(\text{changes} \cdot \log R)\). Checked whole, a plan of 245k runs takes about 300 ms as
  `Plan` on tuples, more than a 7–120 ms pass, and about 11 ms as `Plan.from_array` on the store's (R, 2) array,
  about 5 ms of it the digest. Its checks and encoding are numpy in `verity.ml.kernels.plans`, since a protocol
  imports no third-party package, and the suite holds it to `Plan`'s digests and refusals on the same rows.
  `Session.receive` takes either.
- **What the plan shows.** v0's plans are in the clear, so the verifier sees which KV pages were live in each pass: the
  batch's shape and context lengths over time, not their contents. Hiding them goes with the zero-knowledge openings:
  the plan committed, its distinctness and each opening's \(j = \text{plan}[p]\) proved, only \(L\) public (the draws
  need it).
- **KV registered mid-session.** A KV page's payload is chosen after the codec seed, so it can't be bound by
  registration order the way weights and filler are. It is bound only by integrity: sampled proofs prove it is the
  model's actual computation. The reference registers a fixed store. Appending pages needs their payload roots
  committed before the passes that read them, and fresh store indices. Until then, a block being rewritten stays out
  of every plan, as the prover-side registry already does with blocks a write has touched.
- **Multi-GPU trees.** Each GPU's blocks should be an aligned power-of-two range of leaves, so that a pass's root is
  its GPUs' subtree roots combined. That keeps \(g\) to an all-gather, a few hashes and the next seed's broadcast.
- **Sketch receipts** (a ±1 projection hashed, `RateAuditSketch`) are the stated receipt, sketch-sha512 (§4). Three
  rules apply to them:
  - The sketch's matrix must be derived from each pass's seed, never once per window. A per-window matrix makes each
    block's contribution a fixed digest: read once and kept, it serves every later pass's receipt (red team F29,
    art:61fc2e4c).
  - The matrix must resist a seed the prover chose, since it can grind the seed through its previous root.
  - Under a registration that binds \(z\), \(z\) must enter before the sketch, because a linear sketch of
    \(E \oplus Z\) splits.

  The reference's sketch-sha512 meets all three. Its signs come from each pass's seed. For the second it counts every
  key the prover sees, and reads the bound at \(r^* = 12\) (§4). Under a registration that binds \(z\), \(z\) is
  XORed into the block before the sketch. In the proof unit, check 5 under sketch-sha512 is 16 signed sums over the
  block's 4,096 bytes and one compression, in place of block-sha512's 33 compressions.
- **The challenger's state.** The challenger keeps every coin of a session until it releases it and never releases one
  twice. Persisting that across challenger restarts, and committing coins that outlive one session, is out of scope.
- **Keeping receipts for the draws.** Any receipt of a window may be drawn once its roots are final, so the prover
  keeps them, and the leaves' salts: 64 bytes a 4 KiB block and 192 a leaf of 16, about 1/60 of what it read in the
  window. Keeping each tree above its leaves and re-reading an opened leaf's 16 blocks instead costs a small fraction of
  the reads again (ra-openings puts it at 0.06–0.1%). The reference keeps every pass whole.
- **Lean.** The statement is ra-remote-theorem's `Accounting.Remote.Conservation`, with the per-pass routes
  ra-remote-lean and ra-bound-lean prove. The openings are ra-openings' (#1900, open):
  `Accounting.Guarantees.RateWindowDraw`, a window's pooled draw over its passes' receipts, and `RateSessionDraw`,
  the horizon and every window at once, both from `HorizonSampling`, the causal horizon bound. They take windows of
  one size, \(M = P \cdot N\), and draws with replacement. The record's windows differ in \(U_t\), and its draws are
  distinct, which only lowers the chance that they all pass. `pooled` applies the same argument at each window's own
  size, which the Lean doesn't state yet. What this code supplies is the schedule (each window's challenge released
  at \(b_{t-1}\), its roots due by \(D_t\)), the chain, the plan's distinctness, one tile a receipt under a
  registration that binds \(z\), the sketch's bound read at \(r^*\) (`timing.sketch_reading`), and the profile at δ.
  The chain's ordering is a random-oracle argument over a prover of bounded queries, which the statements must say.

## 9. Why the passes are chained

A pass's receipts must not be computable before the pass, or one read could serve many passes. Either the clock or
the hashes can order the passes. Live coins order them by the clock. Each pass's coin is released on a schedule, so
consecutive passes' read windows overlap by the round trip the verifier can't rule out. That overlap is charged every
pass and needs a trusted round-trip floor to be affordable (Appendix A). The chain orders them by the hashes, so the
round trip is charged once a session. At 20 ms passes and 1 ms RTT, a live coin a pass costs 5.44% with no trusted
floor and 3.15% with a floor of half the RTT. The chain costs 0.25% with no floor, and also saves the 50 coins a
second per node the stream would send.

## 10. The timing

`python -m verity.core.protocols.remote_bandwidth.timing` prints the grid: passes of 20 and 50 ms, windows of
\(T\) = 0.3, 1 and 10 s, RTT 1–50 ms (half each way), \(J\) = 0.05 ms each way and \(g\) = 50 µs, with the least
allowance, and after them §4's rank bound, grinding tries and sessions' failure. \(\mu_{\text{timing}}\) by session, at
\(T\) = 1 s, and the largest of a \(W\) = 100 session's windows' own (§1):

| pass | RTT | passes a window | one window | \(W\) = 10 | \(W\) = 100 | steady state | worst window |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 20 ms | 1 ms | 50 | 2.07% | 0.41% | 0.26% | 0.25% | 4.03% |
| 20 ms | 5 ms | 50 | 2.45% | 0.45% | 0.27% | 0.25% | 4.40% |
| 20 ms | 20 ms | 50 | 3.86% | 0.60% | 0.28% | 0.25% | 5.78% |
| 20 ms | 50 ms | 50 | 6.56% | 0.90% | 0.31% | 0.25% | 8.42% |
| 50 ms | 1 ms | 20 | 4.87% | 0.51% | 0.10% | 0.10% | 9.62% |
| 50 ms | 5 ms | 20 | 5.23% | 0.55% | 0.11% | 0.10% | 9.97% |
| 50 ms | 20 ms | 20 | 6.56% | 0.70% | 0.12% | 0.10% | 11.23% |
| 50 ms | 50 ms | 20 | 9.10% | 0.99% | 0.15% | 0.10% | 13.65% |

A session's ends are its only round-trip cost, so a long session reaches the steady state: at \(T\) = 10 s and
\(W\) = 100 it is 0.25% at 20 ms passes and 0.10–0.11% at 50 ms over the whole RTT range. A short window costs nothing
by itself: at \(T\) = 0.3 s and \(W\) = 100 it is 0.27–0.43% at 20 ms passes. A pass reads distinct blocks, so it is at
most a read of the store: about 52 ms for a GPU's 94 GB at 1.8 TB/s.

A window by itself does pay its ends. Its span is \(T + A\), the pass in flight at its challenge is the window
before's, and its own chain starts only when that pass ends, so a window is charged \(A\) and up to a pass whatever
\(W\) is: about \((A + t_{\text{pass}})/(T + A)\) at worst, the last column. That is under 1% only at \(T\) ≥ 5 s for
20 ms passes and RTT up to 5 ms (0.82–0.90%), \(T\) ≥ 10 s up to 20 ms RTT (0.61–0.80%), and at \(T\) = 15 s for
50 ms passes (0.67–0.99% over the RTT range; 1.01–1.49% at 10 s).

**Openings** at \(\varepsilon_s = 0.1\%\), δ = \(2^{-40}\): \(s = 27{,}726\) per window, about 8.5–9.1 KB each (block,
leaf, both salts, both paths). That is 235–253 MB a window per node: 1.9–2.0 Gb/s at \(T\) = 1 s, 0.19–0.20 Gb/s at
\(T\) = 10 s. Under a registration that binds \(z\), PoUW's tile evidence comes on top unless PoUW's own sampled check
covers the same tiles. A Fiat–Shamir
draw would multiply all of this by 1.5–2.6 (§6).

That is the per-window reading, each window's profile at its own δ, \(W\delta\) in all. Read over the horizon, the
windows' draws pool (`RateSessionDraw`, §8). Each window draws its \(s_t\) after its own roots from its own openings
coin, so however the prover adapts each window to the draws before it, every draw passes while
\(\sum_t s_t g_t \le (1-\bar d) \sum_t s_t\) with probability at most \((1-\bar d)^{\sum_t s_t}\), where \(g_t\) is
window \(t\)'s share of correct receipts. Set that to δ: except with δ, the wrong shares \(f_t = 1 - g_t\) have
\(\sum_t s_t f_t < \bar d \sum_t s_t\), and each \(f_t \le 1 - s_t/U_t\), since \(s_t\) distinct draws found \(s_t\)
correct receipts. The same event bounds every window's share by itself: \(f_t < \bar d \sum_u s_u / s_t\), about
\(\ln(1/\delta)/s_t\), with no union over the windows.

Daniel's claim is that hybrid (#1884): short by under 0.15% over a 10-minute horizon and by under 1% in every window,
at \(2^{-40}\). That needs \(d \cdot n \ge 18{,}470\) openings over the horizon's \(n\) windows and \(d \ge 2{,}773\) a
window. With 7 or more windows in 10 minutes (\(T\) up to 85 s) the window floor is what binds: 2,773 openings a
window, about 24–25 MB, which is 0.19–0.20 Gb/s at \(T\) = 1 s and 19–20 Mb/s at \(T\) = 10 s, a tenth of the
per-window reading at 0.1%. The horizon is then short by under \(\ln(2^{40})/(d\,n)\): 0.0017% at \(T\) = 1 s.

`ledger` reports μ_sample both ways: `sample`, the per-window reading, and `horizon`, the pooled one (`pooled`: the
most \(\sum_t f_t U_t\) the bounds above allow), with `floor`, the largest \(f_t\) they allow, and γ for each
(`gamma`, `gamma_horizon`). It reports each window's the same way over its own span (`Ledger.windows`: each
\(f_t\), from `floors`, is its `floor`, and \(f_t U_t\) its `horizon`), and the worst window's γ either way beside the
session's (`worst`, `worst_horizon`). All of it starts from each window's profile and span, which the record keeps.

The 1% in every window bounds μ_sample; a window's γ adds its own μ_timing above, λ's 0.195%, and μ_chip. At 2,773
openings and \(T\) = 1 s the worst window's γ is about 1% + 0.195% + 4–14%. For the window's whole γ under 1%, its
openings and \(T\) both rise: at \(T\) = 20 s and RTT up to 20 ms, 20 ms passes leave 0.41–0.50% to μ_timing, and a
floor of 0.25% takes \(\ln(2^{40})/0.0025 \approx 11{,}090\) openings a window (94–101 MB, 38–40 Mb/s); with λ that is
0.86–0.95% before μ_chip.

## Appendix A. Live coins, and why the protocol doesn't use them

The alternative: a live coin a pass and no chain. The challenger commits \(K\) pass coins and \(W\) openings coins.
Pass \(k\)'s coin \(c_k\) leaves at \(r_k \ge r_0 + k\Delta\), its seed is \(\text{SHA512}(D \,‖\, c_k \,‖\,
\text{u64be } k)\), reading no root, and its root is due by \(D_k = r_k + A\). It is what 5 Oct's rule, live coins
only, would require. `timing.live_mu` and `timing.live_best_interval` keep its closed forms, and the tests check them.

**What the schedule forces.** A receipt of pass \(k\) needs \(c_k\), which can't reach the node's memory before
\(r_k + d^{\downarrow}_{\text{floor}}\), and its root must leave the node by \(D_k - d^{\uparrow}_{\text{floor}}\). So
the read behind it happens in

\[
I_k = [\, r_k + d^{\downarrow}_{\text{floor}},\; D_k - d^{\uparrow}_{\text{floor}} \,], \qquad \lvert I_k\rvert = A - \text{RTT}_{\text{floor}} .
\]

A read at time \(\tau\) serves at most one receipt for each pass whose \(I_k\) contains it, \(n(\tau)\) in all, so the
receipts' bytes exceed the reads by at most

\[
R \int (n(\tau) - 1)^+ \, d\tau = R\Big(\sum_k \lvert I_k\rvert - \Big\lvert \bigcup_k I_k \Big\rvert\Big) ,
\]

the overlap term \(\mu_{\text{overlap}}\). The tight allowance is the same whole round trip,
\(A = \text{RTT} + 2J + t_{\text{pass}} + g\): the coin's trip down comes before the pass, and the root's trip up
after it.

**The cost.** A pass credits \(t_{\text{pass}} R\) bytes every \(\Delta\). With
\(U = \text{RTT} + 2J - \text{RTT}_{\text{floor}}\), the round trip's uncertainty, the steady state is

\[
\mu_{\text{timing}} + \mu_{\text{overlap}} = 1 - \frac{t_{\text{pass}}}{\Delta} + \frac{\max(0,\, A - \text{RTT}_{\text{floor}} - \Delta)}{\Delta}
= \begin{cases} (g + U)/\Delta & t_{\text{pass}} \le \Delta \le t_{\text{pass}} + g + U \\ 1 - t_{\text{pass}}/\Delta & \text{beyond,} \end{cases}
\]

least at \(\Delta^* = t_{\text{pass}} + g + U\), where it is \((g + U)/(t_{\text{pass}} + g + U)\). Every microsecond of
the round trip the verifier can't rule out is charged once a pass, where the chain charges it once a session. With
\(J\) = 0.05 ms each way and \(g\) = 50 µs:

| pass | RTT | floor (fraction of RTT) | at \(\Delta = t_{\text{pass}}\) | at \(\Delta^*\) | chained |
| --- | --- | --- | --- | --- | --- |
| 20 ms | 0.2 ms | none / 0.5 / 0.9 | 1.75% / 1.25% / 0.85% | 1.72% / 1.23% / 0.84% | 0.25% |
| 20 ms | 1 ms | none / 0.5 / 0.9 | 5.75% / 3.25% / 1.25% | 5.44% / 3.15% / 1.23% | 0.25% |
| 20 ms | 5 ms | none / 0.5 / 0.9 | 25.8% / 13.3% / 3.25% | 20.5% / 11.7% / 3.15% | 0.25% |
| 20 ms | 20 ms | none / 0.5 / 0.9 | vacuous / 50.8% / 10.8% | 50.2% / 33.7% / 9.71% | 0.25% |
| 50 ms | 1 ms | none / 0.5 / 0.9 | 2.30% / 1.30% / 0.50% | 2.25% / 1.28% / 0.50% | 0.10% |
| 50 ms | 5 ms | none / 0.5 / 0.9 | 10.3% / 5.30% / 1.30% | 9.34% / 5.03% / 1.28% | 0.10% |
| 50 ms | 20 ms | none / 0.5 / 0.9 | 40.3% / 20.3% / 4.30% | 28.7% / 16.9% / 4.12% | 0.10% |

Jitter counts twice: at 20 ms passes, 1 ms RTT and a floor of half of it, μ at \(\Delta^*\) is 2.68%, 3.15%, 4.53%,
7.19% and 11.31% at \(J\) = 0, 0.05, 0.2, 0.5 and 1 ms.

**The floor is an assumption.** \(\text{RTT}_{\text{floor}}\) is the round trip between the challenger and the node's
memory that no message can beat: the great-circle distance at the speed of light, or a named network assumption. A
measured round trip is not a floor, because the node controls its side of every measurement and can slow it. With no
trusted floor, the overlap charges the whole round trip, and live coins cost 5.4% at 1 ms RTT where the chain costs
0.25%.

**The rate.** A coin a pass is a committed, released message every 20–50 ms per node, each with its hm96 opening,
for every node audited. That is the rate Daniel called impractical. The chain needs one challenge and one openings
coin a window.
