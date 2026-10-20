# The remote bandwidth audit v0 (`verity/remote-bandwidth/v0`)

A verifier with no hardware at the site certifies, window by window, how many reads of a node's registered blocks the
node made. The node's memory is its GPUs' HBM and its host's DRAM. No prover hardware is trusted: the host, its CPUs
and its NICs are all part of the prover. Each window's challenge and its openings coin are live coins from the
challenger, committed before the session. Inside a window the passes' seeds are Fiat–Shamir: each is derived from the
window's challenge and the prover's previous root, so the passes are chained. Daniel allowed that for this protocol on
12 Oct, because a live coin a pass is impractical at its rate (the friction skill's rulings). Appendix A is the
live-coin design and what it would cost.

The code is `audit.py` (the lifecycle), `coins.py` (the coins), `receipts.py` (seeds, plans, receipts and trees),
`codec.py` (the store's encoding), `memory_map.py` (serving's record of the KV blocks it wrote and freed, the audit's
input from serving) and `timing.py` (the honest schedule, and where the sketch's bound is read). Every byte format below is pinned by
`tests/vectors.json`. This is a reference implementation: pure Python, with openings checked in the clear. The clear
checks are the specification of what each zero-knowledge proof unit proves on the deployed path (§8). The verifier's
definition is Lean code, `lean/` (`RemoteAudit`: `verifyWindow`, a window's draw and its openings' checks, and
`RemoteAudit.Session`, the session's bookkeeping call for call), which `audit.py` mirrors and is held to by
`lean/tests/`' difftest, comparing every event's result, the outcome and the record; `audit.py` is still the verifier
of record (`README.md`). The Lean refuses a registration that binds PoUW's tiles.

## 1. What an accepted session certifies

Let \(N\) be the store's blocks, \(b = 4096\) bytes each. A session has \(W\) windows. Window \(t\)'s challenge
\(\sigma_t\) left the verifier at \(b_{t-1}\) (\(b_{-1}\) is the session's start). Its boundary message left at
\(b_t\), and its roots arrived by \(D_t = b_t + A\). Window \(t\) is \(P_t\) chained passes. Pass \((t, i)\) is a plan
of \(L_{t,i}\) distinct blocks and their receipts. Every receipt was made after the root before it existed and before
its own root left the node. So every read serves at most one receipt of the session (§5). Except with probability
\(\delta\) per window, at most \(m^*_t\) of window \(t\)'s \(U_t = \sum_i L_{t,i}\) receipts are wrong
(`IntegrityProfile.bound`). Pooled over the session's windows instead, the openings bound the wrong receipts of all
of them at once, and each window's share, except with \(\delta\) for the session (§10).

`ledger` turns that into γ, given the node's capacity \(R\) (its read bandwidth over every registered tier, its GPUs'
HBM plus its host's DRAM; specified or measured, §1 "The node, and its one \(R\)" and "Which \(R\) is the headline")
and its on-chip storage \(S_{\text{chip}}\). Over the session's span \(\Sigma = D_{W-1} - b_{-1}\), the credited reads
are at least \((1 - \gamma) R \Sigma\), with

\[
\gamma = \mu_{\text{timing}} + \mu_{\text{sample}} + \mu_{\text{chip}} + \lambda + \mu_{\text{rebuild}} + \mu_{\text{host}} + \mu_{\text{link}} + \mu_{\text{carrier}} .
\]

A receipt of a registered block is credited. One of a block appended after the codec seed (§4) is credited only on
the property's verdict on the circuits that wrote it (`credit_post_seed`, §3); without one it is certified, drawn and
opened like any other, but not credited.

| term | what it is | where it comes from |
| --- | --- | --- |
| \(\mu_{\text{timing}}\) | \(1 - \sum_t U^{c}_t b / (R\Sigma)\), \(U^{c}_t\) window \(t\)'s credited receipts: the span the credited receipts don't fill (the gap \(g\) after each pass, the session's two ends, blocks a plan leaves out, post-seed blocks without a verdict) | the record |
| \(\mu_{\text{sample}}\) | \(\sum_t m^*_t\, b / (R\Sigma)\) read per window (`Ledger.sample`), or the pooled bound's wrong receipts \(\cdot\, b / (R\Sigma)\) read over the horizon (`Ledger.horizon`, §10) | the record's profiles, at δ |
| \(\mu_{\text{chip}}\) | \(\sum_t P_t S_{\text{chip}} / (R\Sigma)\): each pass, what is held on chip needn't be read again | \(S_{\text{chip}}\), measured |
| \(\lambda\) | AuditRO's slack: per pass of \(k\) receipts (every certified one, in either reading), `named_bits(k)`, a whole number of bits at least \(\log_2(q^k/k!) \ge \log_2 C(q, k)\) at \(q = 2^{64}\) (`queries_log2`), and `slack_log2`; 0.116% at the node's 192.6M 4 KiB blocks a pass over every tier (37.9 bits a block), more than \(\mu_{\text{sample}}\) | proved (`Bandwidth.Guarantees.AuditRO`, applied per pass) |
| \(\mu_{\text{rebuild}}\) | \(C_{\text{sq}} b / (E_{\text{sq}} R)\): blocks re-encoded from their payloads in place of reads, each at least the record's codec's \(E_{\text{sq}}\) squarings (`Codec.encode_squarings`: 98,016 at p1024/k3/masked, 65,344 at k = 2, 65,152 at p512/k2) at the node's stated ceiling of \(C_{\text{sq}}\) squarings a second at that width (`Rebuild`) | the record's codec, and assumption 5 stated per width (`WorkHardness(p, k)`) |
| \(\mu_{\text{host}}\) | the host tier's own shortfall, and host DRAM bandwidth PCIe can't carry | a model, measured capacities |
| \(\mu_{\text{link}}\) | under a registration that binds \(z\) only (0 otherwise): relays that meet \(z\) and a copy of the block over a link (§4) | a model |
| \(\mu_{\text{carrier}}\) | only when post-seed receipts are credited (0 otherwise): \(\sum_{t,i} \min(C_t, a_{t,i} b) / (R\Sigma)\), \(a_{t,i}\) pass \((t, i)\)'s post-seed blocks and \(C_t = R_{\text{in}} L + (c_\rho \cdot U_t b \cdot L / E) b\) the verdict's carrier bytes: a credited post-seed block may hold another block's encoding, and one read of it then serves two credited receipts of the pass (§3) | the property's verdict (I2's `P(c, R_in, L)`) |

The protocol bounds the first four, prices \(\mu_{\text{rebuild}}\) from the record's codec (its name and parameters)
at a stated squaring ceiling, passes \(\mu_{\text{host}}\) and \(\mu_{\text{link}}\) through (the record can't bound
them; a protocol carries no policy parameter), and reads \(\mu_{\text{carrier}}\) off the property's verdict. `ledger` takes the rebuild model as a `Rebuild(codec, squarings_per_s)` and refuses one of another
codec than the record's, or a bare fraction. The stated ceiling is the node's, the sum over its devices: 1.67 ×
`kv.cu`'s 1.499·10^10 squarings a second at 1,024 bits for each RTX PRO 6000 with its share of the host
(ra-squaring-ceiling's rule, #1887, on `kv.cu`'s FOLD 6, Daniel's card e2a13d4a; 1.567 × 5.13·10^10 at 512 bits),
2.003·10^11 for the node's eight. There, and at the node's measured \(R\) of 12.92 TB/s, \(\mu_{\text{rebuild}}\) is
0.065% under the default codec, 0.097% at k = 2 and 0.31% under p512/k2 (`timing.SQUARINGS_PER_S`,
`timing.READ_BYTES_PER_S`). The failure probability is \(W\delta\) for a session of \(W\) windows read per window, or
\(\delta\) read over the horizon, plus \(2^{-\text{slack}}\) per pass, plus \(s/2^{512}\) per window for guessing a
receipt. Under the default receipt kind, sketch-sha512, a receipt also rests on `RateAuditSketch`, which adds at most
\(C \cdot 2^{-95.12}\) a session at 16 rows for a prover of \(C\) SHA-512 compressions (`timing.sketch_failure`, §4):
no more than \(2^{-40}\) up to \(C = 2^{55.12}\). At 64 rows, a registration's other row count, it is
\(C \cdot 2^{-215.85}\). block-sha512, kept for comparison, adds nothing. Neither adds to γ.

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

The window is stated: \(T\) = 60 s (`Params.window_ns`, §2), with 33,273 openings a window (`Params.samples`), 555 a
second. There, on one node whose store spans every tier (94 GB of each GPU's HBM and host DRAM's share, 789 GB,
`timing.spanning`), every window's whole \(\gamma_t\) is under 1% at the measured \(R\): 0.888–0.920% in the worst
window at RTT 1–20 ms. The horizon's 0.15% over 10 minutes is stated for its pooled \(\mu_{\text{sample}}\)
(0.0083%), with its whole γ (0.66%) reported beside it: \(\mu_{\text{chip}}\) exceeds 0.15% by itself, and λ with
\(\mu_{\text{rebuild}}\) (§10, `timing.stated`). Below about 373 GB over every tier (44 GB a GPU) no \(T\) meets 1%
(§10, `timing.store_floor`). 20 s and 1 s are options with their own readings: at 20 s and 11,091 openings every
window's \(\mu_{\text{sample}}\) is 0.25% and its whole \(\gamma_t\) 1.26–1.35%, and at 1 s and 2,773 its
\(\mu_{\text{sample}}\) is under 1% and its whole \(\gamma_t\) 5–11%.

**The node, and its one \(R\)** (MEM-26). The prover is the whole node (below), so every capacity a ledger takes is
the node's, the sum over its devices, whatever share of the node the store is on:

- \(R\) is the node's read capacity over every registered tier: its 8 GPUs' HBM plus its host's DRAM. Measured, that
  is 8 × 1,539 + 606.8 = 12,918.8 GB/s (`timing.READ_BYTES_PER_S`); specified, 8 × 1,597 + 819.2 = 13,595.2 GB/s
  (`timing.SPEC_BYTES_PER_S`).
- \(S_{\text{chip}}\) is every GPU's on-chip storage and the host's caches: 8 × 263,163,904 + 940,435,456 =
  3,045,746,688 B (`timing.CHIP_BYTES`; the host's part, two Xeon 6776P sockets' caches, is an estimate).
- \(C_{\text{sq}}\) is the sum of the devices' ceilings, 8 × one GPU's share (`timing.SQUARINGS_PER_S`).

One device's own values are `timing`'s `GPU_` and `HOST_` constants. A device the store isn't on can still hold store
bytes on chip and re-encode blocks, which is why \(S_{\text{chip}}\) and \(C_{\text{sq}}\) are the node's in every
session (the red team's R6, F1). A session over part of the node, as on one GPU, establishes its certified bytes,
\((1 - \gamma) R \Sigma\), in which \(R\) cancels, so they are the same at either \(R\). Its γ is the node's only once
the store spans the node. A store on one GPU leaves the other GPUs and host DRAM idle, about 88% of \(R\), in
\(\mu_{\text{timing}}\). A store in all 8 GPUs' HBM and none of host DRAM leaves host DRAM's 4.7% of \(R\) idle: 752 GB
of HBM reads 5.56–5.59% in the worst stated window (§10). Every window under 1% needs the store over every tier, host
DRAM holding its share.

**Which \(R\) is the headline** (Daniel, 13 Oct). γ is reported twice: against the node's specified read bandwidth,
which is the headline, and against the measured one beside it. The figures above are at the measured \(R\) = 12.92
TB/s. The node is specified at 13.60 TB/s (the RTX PRO 6000 Server Edition's 1.597 TB/s a GPU and 16 channels of
DDR5-6400), which no measured kernel reaches, so at spec every reading also carries that gap:
\(\gamma_{\text{spec}} = 1 - (1 - \gamma_{\text{meas}})\,R_{\text{meas}}/R_{\text{spec}}\), 5.82–5.85% for the worst
stated window. The 1% per window holds at the measured \(R\) only. The two-machine sessions so far are one GPU's, over
an 85 GiB store: both honest sessions accepted and a prover dropping 1/16 of its blocks rejected in its first window
(#2005, #2028; verifier r20261014-193042-3d4f, prover r20261014-193057-8fec). As the verifier recorded them, at one
GPU's constants and the 1.597× ceiling then stated, they read 0.0997 and 0.0999 at spec and 0.0659 and 0.0661
measured. At the node's \(S_{\text{chip}}\) and \(C_{\text{sq}}\) they certify 415.99 and 415.90 TB of the 433.9 TB
their receipts claim over 300 s (431.4 TB at one GPU's as recorded), and their γ
against the node's \(R\) is 0.898 at spec and 0.893 measured, nearly all of it the idle node. Either reading's \(R\)
is the verifier's, never the prover's: a node's own calibration can only raise it, because every term but the timing
term is a fixed byte count over \(R\Sigma\), and a prover reporting a slower read would shrink them all with no more
bytes read (MEM-24: 0.066 read as 0.0057 at the slowest calibration still accepted). The measured reading takes each
tier's fastest measured read: 1,539 GB/s a GPU (the honest session of 14 Oct, `timing.GPU_READ_BYTES_PER_S`) and
606.8 GB/s of host DRAM (r20261015-081037-5ecf, `timing.HOST_READ_BYTES_PER_S`), which
`benchmarks/remote_session/verifier.py`'s ledgers take as `timing.READ_BYTES_PER_S`. \(R\) is the rate of plain, incompressible reads,
which the store's encoded blocks are: the GPU's compression engine reads a compressible buffer faster
(7.55 TB/s against 1.536 TB/s plain, r20261015-043430-0a05), but the encoded store reads at 1.515 TB/s under a
compressible allocation, since the hardware compresses 128-byte units, one codec word, and no encoded word is constant
(the red team's runs r20261015-062054-6945 and r20261015-062750-5541). Other work's reads are bounded in bus time, not in logical bytes (MEM-22).

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
| `samples` | \(s\), the openings per window: 33,273 (`SAMPLES`, 555 a second of the stated window) unless another count is named |
| `delta` | δ of each window's profile: \(2^{-40}\) (`DELTA`) unless another is named |
| `min_passes` | the fewest passes a window may certify |
| `window_ns` | \(T\), the stated window: no boundary message leaves sooner than this after its window's challenge (`Session.boundary` refuses one). 60 s (`WINDOW_NS`) unless another is named; 20 s at 11,091 openings and 1 s at 2,773 are options, read in §10 |
| `codec` | the store's codec, the one its registration names (§3): `p1024/k3/masked` unless another is named |
| `receipt` | the receipt kind, the one its registration names (§4): `sketch-sha512` unless `block-sha512` is named |
| `bind` | whether receipts bind PoUW's tile output \(z\), as its registration says (§4): no unless it binds them |
| `layout` | how a kernel's sketch vectors map to registered blocks, the one its registration names (§4): `store-order/v0` |
| `rows` | sketch-sha512's \(k_s\), the one its registration names (§4): 16 or 64, 16 unless 64 is named (None is the kind's default); 0 under block-sha512 |

A window's length \(b_t - b_{t-1}\) is at least `window_ns`. The verifier may end a window later but never sooner.
The record keeps each window's span, so its \(\gamma_t\) is read over the span it actually had, and the record doesn't
carry `window_ns`.

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
   32)`, `codec` the name of the codec the registration names. The registration also names its receipt kind, whether
   its receipts bind \(z\), its block layout and sketch-sha512's rows (§4), none of them in this context: the store
   depends on none of them.
3. The prover encodes every block under that seed with that codec (`codec.py`, below):
   \(E_j = \text{encode}(\text{payload}_j)\).
4. `Registration.digest = SHA512("verity/remote-bandwidth/registration/v0\0" ‖ frame(node) ‖ u64be N ‖ payload root ‖
   codec seed ‖ frame(codec name) ‖ frame(receipt kind) ‖ u8 bind ‖ frame(block layout) ‖ u32be rows)`, `frame` being
   `verity.ml.randomness.frame` ("s" ‖ u64be length ‖ the name in UTF-8), bind 1 if the receipts bind \(z\), else 0,
   and rows sketch-sha512's \(k_s\) (0 under block-sha512). A record cites this digest, each session's coin commitment
   binds it, and each session's chain starts from it, so a session's coins and seeds are bound to its kind, to whether
   it binds \(z\), to the block cut its kernels' sketch vectors follow and to the sketch's rows.

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
d_j         a bound block's digest: sha512/row/v2(payload_j) = SHA512(P(32768) ‖ payload_j), its payload leaf's x_j
b_j         a bound block's offset: LE(XOF(D ‖ u8 5 ‖ "p1024" ‖ seed ‖ u32be 2^32 − 3 ‖ u64be j ‖ d_j)[0 : 64]) mod p;
            0 under the seed's key
c_a         an append's coin: 32 bytes the verifier draws when the append's root arrives, its digests committed
b_j(c_a)    an appended block's offset: LE(XOF(D ‖ u8 5 ‖ "p1024" ‖ seed ‖ u32be 2^32 − 4 ‖ u64be j ‖ d_j ‖ c_a)
            [0 : 64]) mod p, the b_j that encode and decode take for it
encode      from the payload word, for r = 0 … k − 1: x ↦ σ_K(x) if r ≥ 1, then x ↦ ρ(x + c_r + t(i) + b_j)
decode      for r = k − 1 … 0: x ↦ ρ⁻¹(x) − c_r − t(i) − b_j, then x ↦ σ_K(x) if r ≥ 1; under a bound key, refused
            unless the payload's digest is d_j
~~~

For \(j = \text{slot} \cdot \text{LAYERS} + \text{layer}\) a block is kv/v0's token-layer: a token's keys and values in
one layer, 4,096 bytes, are 32 words at 1,024 bits. Encoding a block costs \(32k\) square roots of 1,021 squarings each,
96 at the default, and decoding it \(32k\) squarings and \(32(k-1)\) masks. A payload word at or above \(p\) is refused:
its top 16 bits are all ones, a bf16 NaN (and its top byte 0xFF, an E4M3 NaN), so every block of finite values encodes.
\(k = 1\) is refused, because a word can then be partly kept and completed for less than an encode (red team F27).

**A block's key** is the seed's or its own (MEM-29). A block whose payload the payload root committed before the codec
seed was drawn (every block of a registration) is encoded under the seed's key, \(b_j = 0\), as above. A block whose
payload is written after the seed (a KV page, or any block appended to a registered store) must be encoded under its
bound key, whose offset is derived from the seed, \(j\), its payload's digest \(d_j\) and its append's coin \(c_a\),
\(b_j(c_a)\) (`codec.encode(..., bound=True, coin=c_a)`, `codec.decode(..., d, c_a)`): only bound blocks may be
registered after the seed. Under the
seed's key alone, such a block's encoding is fixed by the seed and its address, so its writer, a circuit that can learn
the key, picks the encoding \(E\) (all zeros, or another block's) and writes the payload \(\text{decode}(E)\). The store
then holds the picked encoding, the prover holds nothing for it, and each of its receipts is computable without a read
(`tests/test_key_binding.py` runs it against the reference: a prover that holds only the weights' block passes a window
that draws the KV block's receipts). Under the bound key, a picked \(E\) decodes under digest \(d\)'s key to a payload
whose digest is \(d\) only if \(d\) is a fixed point of that map, which a try hits with chance about \(2^{-512}\). So
the writer chooses only the payload, and its encoding costs an encode. Binding is necessary, not sufficient: it stops
the writer picking an encoding, not stacking. A writer that can learn the key can compute another registered block's
encoding in the circuit, or take it as input, and write it (or any cheap invertible function of it) as a bound block's
payload. Decoding that carrier then yields the other block's encoding, and one read serves both blocks' receipts. A
carrier costs one encode, what the honest prover pays a write, so binding doesn't price it. So a block written after
the seed is credited only once the share of the store its writers can program is bounded (the stacking campaign,
note:20261017T0723Z-draft-stacking-model): on the property's verdict on the circuits that wrote it (I2, property/v1,
\(P(c, R_{\text{in}}, L)\), note:20261017T1752Z-draft-program-credited-kv), which `ledger` takes as `credit_post_seed`
and which charges what its carriers can hold (\(\mu_{\text{carrier}}\), §1). Without a verdict (None) such a block may
still be appended and read (§4), and its receipts are certified but not credited: one read of a carrier then serves
one credited receipt, its passenger's, so the credited reads are the registration's, whose payloads no writer chose
after the seed. A verdict that the property fails is refused. \(d_j\) is the payload leaf's \(x_j\), so the
leaf hashes the payload once for both (34 compressions), and \(b_j\) is one SHA-512 output, 2 compressions. The prover
keeps \(d_j\), or \(b_j\) in its place, beside the block: 64 bytes a 4 KiB block, 1.6%. Blocks under the seed's key are
unchanged, byte for byte.

**The append coin.** The digest alone leaves the writer advice: everything it chooses before committing \(d_j\) that the
key doesn't read (padding and dead-value bits, run inputs, re-key times, schedules) can still be chosen with the
encoding in view, and only an encode budget bounded that, a bound the stacked sketch analysis loses past about
\(41 - \log_2(\text{passes})\) bits (note:20261017T1855Z-report-stacking-lean). The coin makes that advice 0: the
verifier draws \(c_a\) only once the append's root, which commits every \(d_j\) of the append, has arrived (§4), so the
prover can compute none of the append's encodings before that commit, and no choice made before it can depend on one.
Its derivation index, \(2^{32} - 4\), is not the pre-coin form's \(2^{32} - 3\), so a key of one form is never one of
the other's. The pre-coin form is `kv.py`'s `encode_token_layer(..., bound=True)` for \(j = \text{slot} \cdot
\text{LAYERS} + \text{layer}\), byte for byte, and the tests pin it (`kv_vectors.json`'s "bound", and `vectors.json`'s
`codec.bound`); no session decodes an appended block under it. The coin form is pinned by `vectors.json`'s
`codec.coin`, blocks 0 and 20 under a fixed coin. `kv.cu` and `paged.cu` take the bound key as a per-block offset
added to the tweak, \(b_j + \tau W (j - j')\) for a block written at another block's indices \(j'\) (`kv_encode_bound`,
`paged.cu`'s offset table; gated against both pre-coin pins by `kv_bench.py` and `paged_gate.py`). Both forms enter the
words only as that offset, so they encode the coin form unchanged, given \(b_j(c_a)\) in place of \(b_j\); `kv_twin.py`
and vLLM's encoded store take the seed's key only (§8).

Three properties of registration matter:

- **The order is the point.** If the codec seed came before the payloads, the prover would register
  \(\text{payload}_j = \text{decode}(0)\) and hold a store of zeros. Drawn after, each \(E_j\) costs \(k\) square roots
  a word to produce, whatever the payload, all-zero weights included. That is the only rebuild bound: no
  incompressibility is assumed, and every bound is decode-free. A block written after the seed can't be ordered this
  way, so its bound key does the ordering: its key follows its payload.
- **No commitment to \(E\) is needed.** ρ and \(\sigma_K\) permute the canonical words, so `decode` under the seed's key
  is a bijection from canonical blocks to payloads, and the payload root plus the codec seed fix \(E\). Under a bound
  key the append's root fixes \(d_j = x_j\) too, so with its coin it still fixes \(E\). An opened block that isn't
  canonical fails
  (`codec`).
- **The node is in the codec seed.** Two nodes registering the same weights hold different stores, and the tweak
  \(\tau i\) separates every word of one store.

**One live registration per node.** Each registration draws a fresh codec seed, so a prover that registered \(R\)
times and kept the registration whose key suits it would choose \(\log_2 R\) bits of the seed: a retry that redraws,
which live coins rule out (the red team's KB-4, note:20261017T1859Z-report-property-redteam §R4.2). The verifier draws
through its `Registry`, which holds at most one live registration per node:

- a second `register` for a node while it has a live registration is refused (`RegistryError`), counted (`refused`),
  and draws nothing;
- each session over the registration is the registry's (`Registry.session`, its coins' session number its count so
  far);
- the registration ends only when the verifier closes it (`Registry.close`), never on the prover's word. The close
  counts each of its sessions as it stands: accepted, the failure that rejected it, or unended, which counts as failed.

So a failed session stays on its registration's record: no fresh draw can replace that registration while it is
live, and closing it to draw again records the failure. Every codec seed a node has been given stays in the registry
(`Registry.drawn`, \(R\)), which an advice form can charge as \(\log_2 R\). Refusal rather than supersession: a second
registration that superseded the first would cut its session in flight short, so the record would show the supersession
in place of that session's own verdict, and the prover could have a fresh draw whenever it asked. The remote session
verifier keeps one registry for its process, or across processes in a file (`--registry`). It answers a refused hello
with `bye` and the reason, closes its registration after its last session, and leaves a registration live when its
process ends early, until `--close`. The registry's record holds digests and outcomes, never the verifier's secret.

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
seed_(t,i)  SHA512("verity/remote-bandwidth/seed/v0\0" ‖ σ_t ‖ u32be i ‖ prev_(t,i)),
            root_(t,−1) = root_(t−1,last) in a session, and the registration's digest before its first window
prev_(t,i)  root_(t,i−1) while no append has been received before root_(t,i); otherwise
            SHA512("verity/remote-bandwidth/chained/v0\0" ‖ root_(t,i−1) ‖ a_(t,i)) (`audit.chained`)
a_(t,i)     the chain over every append received before root_(t,i), in order: a = 64 zero bytes, then for each append
            a ← SHA512("verity/remote-bandwidth/appends/v0\0" ‖ a ‖ u64be first ‖ u64be count ‖ its root)
            (`audit.append_head`)
plan_(t,i)  R runs (first, count) in read order: count ≥ 1, inside the store, pairwise disjoint, no run starting where
            the one before it ends.  Position p is the p-th block of the runs concatenated; L positions in all.
            digest SHA512("verity/remote-bandwidth/plan/v0\0" ‖ u32be R ‖ (u64be first ‖ u64be count) × R)
e_(i,j)     E_j, the block alone; under a registration that binds z, E_j ⊕ spread(z_(i,j))
y_(i,p)     sketch-sha512, the default: SHA512(Z("verity/remote-bandwidth/sketch-receipt/v0\0" ‖ frame(seed_(t,i))) ‖
            u64be j ‖ σ_(i,j)),  j = plan_(t,i)[p]
            block-sha512, for comparison: SHA512(Z("verity/remote-bandwidth/receipt/v0\0" ‖ frame(seed_(t,i))) ‖ u64be j ‖
            e_(i,j))
S_(t,i)     sketch-sha512's k_s × 4096 signs, k_s the registration's rows (16 or 64): K = derive(seed_(t,i),
            "verity/remote-bandwidth/sketch-signs/v0", {"tag": "sketch-sha512", "k": k_s, "n": 4096}), row c the first
            512 bytes of K.stream(("row", c)), bit u (byte u // 8, LSB first) 1 → −1 and 0 → +1
σ_(i,j)     S_(t,i) · e, e the 4096 bytes of e_(i,j) as int8, over the integers (|σ[c]| ≤ 2^19): k_s int32
            little-endian, 64 bytes at 16 rows and 256 at 64
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

The prover sends each root with its pass's plan and its step as the pass ends (`Session.receive`), the plan whole or
as its change from the session's previous pass's plan (`Plan.changed`: the blocks it adds and removes, the result in
store order; before the first pass, the registered store's). A pass covers one step \(s\) of the served circuit's
execution, the engine step, and its plan is the blocks the circuit reads at \(s\) (§8, "The pass and its step"). The
step is public, an integer below \(2^{64}\), and the steps strictly increase over the session's passes, across its
windows, so no two passes cover one step. A root out of turn rejects with `order`: a window's roots come in index
order, and a window's first root makes the window before it final. A root with no step, or with a step not above the
previous pass's, rejects with `step`. A plan that breaks its format, or a change that removes a block its plan doesn't
hold or adds one it does, rejects with `plan`.

**Appends and releases.** Blocks written after the codec seed (KV pages) join the store mid-session. The prover commits
their payloads in an append tree (`append_tree`: the payload tree's leaves, \(x_j = d_j\), in a frame-v3-sha512 domain
bound to the registration's digest over the indices it takes) and sends its root and size (`Session.append`). The
verifier assigns the store's next fresh indices, from \(N\) up, draws the append's coin \(c_a\) from the challenger's
generator (`Challenger.append_coin`), and replies with the first index and the coin:

~~~text
prover → verifier   {"type": "append", "root": hex(append tree's root), "count": n}
verifier → prover   {"type": "appended", "first": j, "coin": hex(c_a)}      c_a 32 bytes, drawn when the root arrives
~~~

So each block's index, and with \(d_j\) and \(c_a\) its bound key and its encoding (§3), is fixed before any pass whose
plan holds it: a plan received before the append that commits one of its blocks rejects with `plan`. Only then does
the prover encode the append's blocks, each under its bound key \(b_j(c_a)\), keeping \(d_j\) beside it
(`Store.append` commits, `Store.encode` encodes under the coin). The timing follows the free convention (§8, "The
pass and its step"), with residency deferred to the encode, so the coin's round trip is off serving's critical path.
A block completed at step \(s\) is appended once it is complete, and stays plaintext in its slot, which serving reads
in the clear, until its coin arrives and the device encodes it in place. Its write is recorded at the step before the
first step \(t\) whose reads see it encoded, at \(t - 1\) (at \(s\) itself when the coin is back before step
\(s + 1\)'s reads), so it is resident, and planned, from pass \(t\) on. Until then it is in no record and uncredited,
as a partly written block is (memory-map/v1, `memory_map.py`). The append therefore goes out before the record writes
the block, the root of every pass whose plan holds the block follows the append, and that pass's seed binds it
(below). The record's step follows from the encode, and moving it gains nothing: recorded later, the block's encoded
reads go uncredited; recorded earlier, a plan holds bytes that are not its encoding, and their receipts fail. The
convention is unchanged: step \(t - 1\)'s record takes effect after that step's reads. Each coin is live and drawn
once: the challenger commits none ahead, a retry never redraws one, and an append whose coin is never used is a
failure in the record. A coin is used once a root follows its append, the append then joining that root's window
(the lifetime cap, below). The rejections are `coin`: an append whose root was sent before (a retry; no coin is
drawn), an append after the last window's roots are final (no root can follow it; no coin is drawn), and, at the last
window's close, an append no root followed (`Session.close`). A block the prover stops holding is released
(`Session.release`, runs of the store), and a plan received after its release that holds it rejects with `plan`; its
receipts from earlier passes can still be drawn, so the prover keeps it until its windows are drawn. A block freed
before its encode lands is *withdrawn*: its values' last read is at or before the step its write would be recorded
at, so it gets no write, no free and no credit. It was never resident encoded, and a block no step reads while it's
resident is in no record, so the line its write would be in lists its index in `withdrawn` instead (memory-map/v1's
per-step list, optional and empty by default, so a map without withdrawals reads as before). A line's writes and
withdrawals take the store's next fresh indices together, and a withdrawn index is never written. The prover's
release covers the index and names it withdrawn (`Session.release(runs, withdrawn)`), and the verifier checks it was
never written: an appended block, inside the release's runs, that no plan received before held. An append that is
malformed or would take the store past its codec's `max_blocks` rejects with `append`, and a release of a block
outside the store or released already, or one naming withdrawn a block that is registered, outside its runs or held
by a plan, with `release`. Appends are a session's: the next session over the store starts again from \(N\) (§8).

**No block appended after a pass's seed is in that pass's plan.** A pass's seed binds every append received before
its root (`prev_(t,i)` above), so every block its plan can hold, which must be appended before its root, was
committed before its seed. A block appended after the seed is in no plan of the pass, and its receipts are credited
in no pass whose seed came before its append. Without this the seed would depend only on \(\sigma_t\) and the root
before, which the prover knows before it sends that root, so a page appended mid-window could hold the pass's
sketches of other blocks (64 bytes a block at 16 rows, 64 blocks a page), and their receipts would follow from the
page's bytes without reading them: the red team's KB-1, in note:20261017T1859Z-report-property-redteam §R4.2, whose
test is `test_append_order.py`. A page that relayed any receipt of a pass whose seed binds it would be a fixed point
of SHA-512. The prover therefore sends a pass's appends before its root and computes the seed after they're sent
(`Store.head`, `Prover.run(..., head=...)`). A seed computed without an append the verifier received before the root
fails every receipt of the pass. While a session has no appends the seeds are unchanged, so the GPU passes, which
append nothing, derive them as before.

**The lifetime cap.** \(L\) here is the lifetime cap of §1's carrier term, not a pass's length \(L_{t,i}\). An
appended block joins the window of the next root received after its append, and no plan of a window \(L\) or more
after that one may hold it (`Params.lifetime`, `LIFETIME`: 60 windows, an hour at the stated window). A plan that does
rejects with `lifetime`. So every post-seed block leaves the credited store within \(L\) windows, and the stacked
stock the carrier term charges is the input rate times \(L\) (§1, §8). A block the prover keeps longer, a prefix
cache's for instance, is re-keyed: released, and its payload appended again at the store's next fresh index, which
draws a fresh coin and so gives it a fresh bound key (§3), so its encoding is computed anew. A re-key's free is the
one exception to a free at its values' last read (§8, "The pass and its step"). The re-key's step frees the old
index, and the new index is written when its encode lands (deferred residency, above), so the value is uncredited in
between, and withdrawn if its last read comes first. A re-key is only at the cap (§8, "The memory map's choices"),
so its time is the map's, not a choice made after the coin. Under deferred residency a block appended in window
\(w\), the one its append joined, is resident from \(w\) or \(w + 1\). Its re-key, recorded at a step before the
first pass of window \(w + L\), therefore falls at age \(L - 1\) at the latest. At \(L = 1\) that is the window it was
appended in, before it may be resident, and a block whose encode lands after the boundary is never credited. So a
session that defers residency has \(L \ge 2\) (`Params.deferred`; the verifier refuses one at \(L = 1\)). Counting
the lifetime from residency instead of from the append would allow \(L = 1\); that is a later decision, and it would
change the Lean port's lifetime check and the property `LifetimeCapped`. The record names \(L\) (`lifetime`), and
the remote session sends it with the session, so the prover re-keys in time.

**Fiat–Shamir, and what grinding buys.** A seed after the first depends on the prover's own root, so the prover can
grind: re-salting a leaf gives a new root and a new next seed, at the cost of a leaf and its path and no read. A
chosen seed buys nothing against the block-sha512 receipt, because no seed makes a receipt cheaper to compute without
the block's bytes after the seed. The queries a grinding prover makes are counted by AuditRO's \(q\), at most
\(2^{64}\) a pass (`queries_log2`), whose \(C(q, k)\) λ charges. Other users of the seed must hold for a seed the prover chose: sketch-sha512's signs (below),
and under a registration that binds \(z\), PoUW's noise for the pass (§8).

**The two receipt kinds.** The registration names one (`receipts.KINDS`), and `Params.receipt` must name the same.
Every receipt of its sessions is of that kind, the weights' and the KV pages' alike (§8). sketch-sha512 is the default (`receipts.DEFAULT`) and the stated receipt: the receipt the guarantee is stated for
(Daniel, 12 Oct, #1884). block-sha512 is kept as a second kind, for comparison only; it is not the stated receipt.
sketch-sha512 is ra-codec-tensor's `sketch_receipt.py` (#1916, at 5b8decfc5; at 64 rows, f666d887b), byte for byte:
the suite pins its numpy outputs at both row counts, and `receipts.SketchReceipter` computes the same in pure Python.
Each kind's receipt is 64 bytes, so leaves, salts and the tree don't change. The difference is what is hashed after
the seed's block: sketch-sha512 hashes \(j\) and the sketch, one compression at 16 rows (64 bytes) and three at 64
(256 bytes); block-sha512 hashes \(j\) and the 4,096 bytes, 33 SHA-512 compressions. The sketch's signs come from the
pass's seed, so like the seed they exist only once the root before it does.

**The rows.** sketch-sha512's \(k_s\) is a registration parameter (`receipts.ROWS`: 16 or 64), bound in the digest
(§3) and in the signs' key (`"k"`), so 64-row signs are not 16-row signs and 48 more, and a session's `Params.rows`
must be the registration's. 16 is the default. 64 rows read the bound at \(r^* = 27\), \(2^{-215.85}\) a compression
(below), so no session is long enough to need a cap. On the device reader they cost τ = 1.226 a pass in one kernel
with an L2 hint, below 16 rows' 1.258 (ra-codec-tensor at 47b703a14, run r20261013-071237-9303, art:dd7bbe89). In
the fused GEMM they cost more: at 70B and B = 8, #1916's step cost d is 3.49 / 4.00 at k = 2 / 3 against 16 rows'
2.40 / 3.18 (#1916 at cf4f823c4, run r20261013-122226-8001), since the two sketch warps bind there. So 16 stays the
default, and a registration names 64 to use it.

What each kind rests on:

- **block-sha512** needs only the random oracle. A correct receipt is a query at the block's bytes after the seed,
  so it bounds the reads by AuditRO (`Bandwidth.Guarantees.AuditRO`, applied per pass), with λ's slack. It assumes
  nothing about the device: the prover may hold any function of the store.
- **sketch-sha512** also rests on `RateAuditSketch` (ra-rate-audit-lean, #1849, open). The random oracle takes a
  correct receipt to a correct \(\sigma_j\). `RateAuditSketch` takes a correct \(\sigma_j\) to the block's 128
  sectors of 32 bytes having been obtained, except for a guess. A query made without one of them is right with
  chance at most \(g/2^{256}\) per query, \(g\) the largest fiber of one sector under the signs. The signs bind
  that far except with \(\Pr_S[g < \text{fib}_S]\). Four conditions come with it:
  - **The store is held as its blocks** (`rate-sketch-verbatim`, restated by Daniel on 13 Oct). In every tier the
    prover holds the store only as its blocks, any number of copies anywhere, and any device may compute from the
    blocks it reads, every such read counted. A layout from which sketches cost less to read than the blocks, such as
    a table of precomputed sketches, is outside it. That exclusion is needed: Lean shows such a table passes every
    check while reading 1/64 of each block at 16 rows (`RateSketchTable`), though it needs about \(2^{65536}\) times
    the store in memory. What Lean proves today is the narrower form, where each device obtains a sector only whole
    and as it is.
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
10-minute session. \(2^{-64}\) is out of reach at 16 rows (42 ms of one node's hashing) and needs 64 rows, a
registration's other row count, at τ 1.226 a pass on the reader; the fused GEMM's cost at 64 rows isn't measured yet.

A grinding try is not a pass's commitment. It reads no block: re-salting one receipt leaf changes the root without
changing any receipt. As the reference computes it, a try is 15 + 3⌈log₂ n⌉ compressions over \(n\) leaves
(`timing.grind_compressions`): the leaf's salt digest, the hm96 key's digest and its tree leaf (9), the leaf's hash
(2), its path (3 a level), the next seed (2) and its signs' key (2). That is 66 compressions, 9.4 ns on one GPU, for
an 8 GiB pass of 131,072 leaves, so a node of 8 GPUs affords about \(2^{29.7}\) tries a second, \(2^{35.6}\) in a
stated 60 s window. A prover that keeps the constant prefixes' midstates needs at least 5 compressions a try (the
root's last, and the seed's and the key's 2 each, whose first blocks hold the root and the seed): at most
\(2^{33.4}\) tries a second on that node, \(2^{39.3}\) in a window. Seeing one sector's \(16 \times 32\) signs takes 16 more, one stream block a row. The bound doesn't rest on
any of these counts: it charges a key to every compression.

None of this moves γ: \(F\) adds to the failure probability, not to the reads certified, so Daniel's claim at
\(2^{-40}\) (§10) reads as before. For one node's 10-minute horizon, \(F = 2^{-50.19}\), so
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

**The block layout.** The registration fixes the store's block cut: block \(j\) is the encoded store's bytes \(4096j\)
to \(4096j + 4095\), whole codec words (the 32 words \(32j, \dots, 32j + 31\) at p1024, 64 at p512), the cut the payload
tree's leaves and the opening check (§6, check 3) use. A kernel need not read a block as one vector: a tensor-core
GEMM reads its operand as fragments, 32 bytes of each 512-byte step a lane. So the registration also names a block
layout, how a kernel's sketch vectors map to registered blocks (`receipts.LAYOUTS`; `Params.layout` names the same),
and its digest binds the name. The layout is the registration's, never the launch's: the prover's launch parameters
(the GEMM's `groups`, its tile) can't move the cut. store-order/v0, the default and so far the only one, makes a
vector one registered block's 4,096 bytes in store order. Its check is `receipts.plan_of`: from a kernel's byte map
(for each byte of each vector, the store byte it holds, or −1 where it holds none), the plan the vectors are, refusing
a vector with padding, a byte past the store, a codec word split with another vector, bytes of two blocks, or a block
twice. A layout the verifier can't map to whole registered blocks is refused, at registration and by `Params`.

Only registered blocks' bytes are credited. A pass's plan is `plan_of` of its kernel's vectors under the registered
layout, and the record credits \(b\) bytes a position, so padding (a CTA's last tile zero-filled past its tasks) is
never a position, and a vector of padding alone, whose receipt the seed alone determines, is never credited. The
fragment cut that `gemm_blocks` builds splits every 128-byte word across 4 vectors (RB-D1, #1916), so `plan_of` refuses
it. A kernel that keeps it needs a layout of its own, one whose every vector is still one whole registered block (a
fixed permutation of the block's bytes, which the verifier applies to the opened block before its sketch), named in
`LAYOUTS` and here before any of its receipts is credited. ra-codec-tensor proposes the permutation its fused GEMM
can produce, and it is added as a named layout with its check and vectors (Daniel, 13 Oct). Until then the fused
GEMM's vectors are refused: store-order/v0 is the only layout.

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
counts a seed's), so the stated 60 s window affords one node of 8 GPUs about \(2^{35.6}\) to \(2^{39.3}\) of them.
At the stated window's 33,273 openings (\(\varepsilon_s = \ln(2^{40})/s\) = 0.083%) and δ = \(2^{-40}\), the count at
\(\delta/Q\) is \(\lceil s\,(40 + \log_2 Q)/40 \rceil\):

| grinding \(Q\) | \(s\) | × the coin's |
| --- | --- | --- |
| none (the coin) | 33,273 | 1.00 |
| \(2^{20}\) | 49,910 | 1.50 |
| \(2^{40}\) | 66,546 | 2.00 |
| \(2^{64}\) (λ's budget) | 86,510 | 2.60 |

It would also void the horizon reading (§10), whose proof takes each window's draw to be independent of what the
prover did in that window, unless that is re-proved for a grinding prover. The coin costs one message a window, sent
after the roots, which the verifier is waiting for anyway. So the protocol keeps it.

The prover opens each draw (`Prover.open`, `Opening`): the block \(E_j\) at \(j = \text{plan}_{(t,i)}[p]\), its receipt
leaf's receipts \(v_c\) with the leaf's salt and path to \(\text{root}_{(t,i)}\), the payload's salt and path to the
payload root (to its append's root, for a block appended after the seed, which also carries its payload's digest
\(d_j\)), and under a registration that binds \(z\), PoUW's evidence for \(z_{(i,j)}\). Openings aren't timed,
because the roots already bind every receipt, and the next window's passes run while they are checked.

The verifier checks each opening in order (`Session.verify`):

1. the indices are the draw's and every field has its size, the tile evidence empty unless the registration binds
   \(z\), and the digest 64 bytes for an appended block (\(j \ge N\)) and empty for a registered one (`malformed`);
2. the leaf, under \(\text{seed}_{(t,i)}\) (from \(\sigma_t\), \(i\) and the root received before
   \(\text{root}_{(t,i)}\)), the plan received with the root and the leaf's salt, has its path to
   \(\text{root}_{(t,i)}\) (`receipt-path`);
3. the block, read as the registered codec's words, has every word below \(p\) (`codec`), and its decoding's leaf under
   the payload salt has its path to the payload root at \(j\) (`payload-path`). An appended block is decoded under the
   bound key of the \(d_j\) it carries and the coin the verifier drew for the append that holds \(j\), \(b_j(c_a)\),
   refused (`codec`) unless its decoding's digest is \(d_j\), and its leaf's path goes to that append's root. A block
   encoded before its coin (under the pre-coin form, a guessed coin or another append's) decodes under no key the
   verifier holds for it;
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
64 squarings and 32 masks, about twice. The opening is the same size: a block is 4,096 bytes under every codec. Every
block of a v0 registration is under the seed's key; a bound block (§3, appended as §4 says) is decoded less
\(c_r + t(i) + b_j\), with \(b_j = b_j(c_a)\) 2 compressions from \(d_j = x_j\), which the payload leaf's check hashes
anyway, and the append's coin, and
refused (`codec`) unless its decoding's digest is \(d_j\). Its opening is 64 bytes longer, the digest.

A window is accepted only if every opening checks. A rejected window ends its session: every later call is refused,
and nothing of the session is recorded. A pass whose seed is not the chain's (passes of a window run at once, a window
restarted from the registration, a window's passes under the previous window's challenge) fails check 2 at its first
opening.

## 7. The record

`Session.record()` returns a `Record` once every window is accepted. `public()` holds the registration digest, the coin
commitment's root, \(N\), \(b\), the receipt kind, whether its receipts bind \(z\), the codec's name and parameters
(`codec`, `codec_params`: the field \(p = 2^n - c\), \(k\) and the mask), the block layout, the sketch's rows
(`rows`), each window's passes and receipts \(U_t\) and how many of them are of blocks appended after the seed
(`post_seed`), the blocks the session appended and released (`appended`, `released`), the lifetime cap \(L\) it held
the appended blocks to (`lifetime`, §4), the chain \(a\) over its appends in the order received (`appends`, §4's
`a_(t,i)` after the last append, null without one: a hash over hiding roots and their indices, which the memory-map
unit's verdict is on, §8), and the session's span. The
digest binds the kind, `bind`, the
codec, the layout and the rows already, but the record names them as well: what its receipts rest on depends on the
kind, `bind` and the rows (§4), and `ledger` takes a link term only from a record that binds \(z\);
\(\mu_{\text{rebuild}}\) is priced from the record's codec (§1, §3), and `ledger` refuses a rebuild model of any
other; and the layout says which bytes a credited block is (§4). Each
`WindowRecord` carries its passes' lengths \(L_{t,i}\), their post-seed reads \(a_{t,i}\) (`post_seed`, which
`ledger` credits only on a verdict, §1), their steps (`steps`, strictly increasing, §4), their heads (`heads`, the
chain over the appends received before each pass's root, which its seed binds, null before the first), their stale
bounds (`stale`, `Session._stale_below`), its challenge's release \(b_{t-1}\), its deadline \(D_t\) (so its span
\(\Sigma_t\), `span_ns`), its openings coin's release, and an `IntegrityProfile`. The stale bound of a pass of window
\(t\) is \(N + \sum \{\text{count}_a : w_a \le t - L\}\) over the session's appends \(a\), \(w_a\) the window \(a\)
joined (that of the first root received after it, the first pass whose head chains it): the store index below which
every appended block is past the cap, the same for every pass of a window. `public()` lists the heads and stale
bounds per window too (`heads`, `stale`), which the memory-map unit's verdict is on (§8): each head is a hiding
commitment, a hash over hiding append roots and their public indices like `appends`, and each stale bound is
protocol structure, a function of \(N\) (`blocks`), \(L\) (`lifetime`), the heads, and the appends' extents (first,
count), which the verifier sees in the clear with each append and `appends` commits to. So `stale` adds nothing beyond
the appends' extents and the windows they joined; the record lists the extents only as their chain and total
(`appended`), so read from it alone `stale` shows the blocks appended \(L\) windows back, window by window. Whether the
extents should be public at all is a separate question about `appends` (the counts per window depend on the
workload). That is every input of
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

### Declared outputs

A session's outputs are these, and only these (decided 18 Oct, 17:10Z, under proofs' rule that a protocol publishes
its structure, its outputs, the verifier's coins and draws, and hiding commitments):

- **The verdict**, per window and for the session: accepted, or the failure that ended it (one of `FAILURES`, at the
  window it ended in).
- **γ**, per window (\(\gamma_t\)) and over the session, each against both \(R\): the node's specified read bandwidth,
  the headline, and its measured one (§1, "Which \(R\) is the headline"). The session's is read both ways, per window at
  \(W\delta\) and over the horizon at \(\delta\) (§10), each with its failure probability. γ is over the registered
  store only: a receipt of a block appended after the codec seed is drawn and opened like any other but not credited
  (`credit_post_seed` is None, so \(\mu_{\text{carrier}}\) is 0), and γ doesn't depend on what the workload appends.
  A paced session's windows, for γ, are its stretches. A paced session is one whose windows are each a group of at
  most \(k\) passes under a live coin of its own, every pass reading every registered block, with the groups taken a
  stretch of about \(T\) at a time and the last stretch closing the session: its reads back the credited span's end,
  and it is not credited.
- **The memory line**, a session total only, in three forms, each with its ε and δ: the registered bytes the session
  shows held in node memory are at least \((1 - \varepsilon) N b\), except with probability δ.
  - *Snapshot*: the registered blocks in node memory at the start of each credited pass, averaged over those passes.
  - *Time-held*: the registered blocks in node memory at each instant of the credited span, averaged over the span.
    Only a paced session has one.
  - *Link-charged*: the time-held line with the node's boundary links charged as carrying blocks back into node
    memory at their ceiling, which drops Identity.

  Every form charges the blocks the node could re-encode from their payloads (its squaring ceiling at the record's
  codec) and read from its own disks (their measured read ceiling), and its on-chip storage \(S_{\text{chip}}\). δ is
  the horizon's error plus the sketch's (`timing.sketch_failure`).

\(R\), \(S_{\text{chip}}\), the squaring ceiling and the disks' and links' ceilings are the verifier's: the node's
stated or measured capacities, protocol structure, never the prover's report (§1).

Everything else that follows the workload is a value, not public: the blocks the session appended and released and
the chain over its appends, each pass's plan, its post-seed reads \(a_{t,i}\), its step and its root's arrival, and any
memory line per window or by kind (weights, KV, host). Each window's receipts \(U_t\) stay public, since its draws are
over them. Until `public()` is trimmed to match (after #2519, which keeps `stale` out of it), it still carries
`appended`, `released` and each window's `post_seed`.

## 8. What deployment changes, and what is open

- **Zero knowledge.** Every protocol is zero-knowledge, with no clear mode, and an opened block decodes to a KV page,
  which is user data. §6's openings reveal it, so the clear checks are only the specification of what each unit
  proves. Deployed, each drawn receipt is one proof unit that the proof service proves in zero knowledge, and the
  verifier sees only its verdict. The unit is §6's checks 1–3 and 5 on one opening (and 4 under a registration that
  binds \(z\)), plus the link from the block's payload to the circuit's committed values:
  - *Public*, each of the four kinds the rules allow (protocol structure, outputs, the verifier's coins and draws,
    hiding commitments): the registration (its digest, \(N\), \(b\), the codec and the receipt kind); the draw
    (window \(t\), pass \(i\), the position in the pass, and \(L_{t,i}\)); the pass's root \(\text{root}_{(t,i)}\)
    and the root received before it; the pass's seed \(\text{seed}_{(t,i)}\), from \(\sigma_t\), \(i\) and that
    root; the plan's commitment (v0's plan digest); the payload root; for an appended block, its append's root and
    coin \(c_a\); and the roots of the circuit's value commitments.
  - *Witness*: the block \(E_j\), its payload \(\text{payload}_j\) and digest \(d_j\), the index \(j\), the receipt
    leaf's receipts, and every salt and path: the receipt leaf's, the payload leaf's, the plan's at the position, and
    the value leaves'.
  - *What it proves.* Check 1 is the unit's shape: each witness fills its port's fixed size, and the position is the
    draw's. Check 2: the receipt leaf, under the seed, the plan and the leaf's salt, has its path to the pass's root,
    and \(j\) is the plan's entry at the position. Check 3: every word of \(E_j\) is below \(p\), and \(E_j\) decodes
    to \(\text{payload}_j\), under the seed's key for a registered block and under \(b_j(c_a)\) for an appended one,
    with \(d_j = \text{sha512/row/v2}(\text{payload}_j)\) and the payload leaf on its path, at \(j\), to the payload
    root or to the append's root. Check 5: the receipt at the position, recomputed from \(j\) and \(E_j\) (and \(z\)
    if bound) under the registration's kind, is the leaf's. And for an appended block, \(\text{payload}_j\) is the
    committed KV values the circuit wrote to block \(j\), under the layout: each of those values opens under its
    value commitment's root (the vLLM Commit's hiding leaves). So the receipted block is the circuit's KV, not bytes
    the prover chose.
  - *Output*: the unit's verdict, accepted or §6's failure name, within the 16-bit output limit. The record and
    `ledger` read only verdicts, and nothing an opening carries leaves the unit.

  The formats already hide: payload blocks and receipt leaves are `hm96-sha512/row/v2` leaves under fresh salts, and
  the coins are hm96 leaves until released, so the formats don't change when the openings move into proofs. A public
  \(c_a\) and append root say which append the drawn block belongs to, but not its index or its contents. Which value
  leaves a block's payload bytes are is the memory-map unit's public row's (`map_value_leaves` in
  `verity.ml.circuits.requirement_units`: a present value \(k\) is leaf \(\text{pos}[k]\) of the value commitments,
  and payload \(c\) is store index \(\text{first} + c\)). Open: the units' circuits; and the plan's commitment in
  place of v0's clear plan ("What the plan shows", below).
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
  V) is 458k runs, 7.3 MB a pass: 2.9 Gb/s per GPU at 20 ms passes, twice that for a plane-major cache. So a plan
  may be sent as its change from the previous pass's plan (§4, `Plan.changed`). A decode step at batch 64 allocates
  or frees about 4 pages a layer, 128 runs over 32 layers, about 4 KB a step. The reference applies a change in
  \(O(R + \text{changes})\), in Python; deployed, the verifier updates its sorted copy and checks distinctness in
  \(O(\text{changes} \cdot \log R)\). Checked whole, a plan of 245k runs takes about 300 ms as
  `Plan` on tuples, more than a 7–120 ms pass, and about 11 ms as `Plan.from_array` on the store's (R, 2) array,
  about 5 ms of it the digest. Its checks and encoding are numpy in `verity.ml.kernels.plans`, since a protocol
  imports no third-party package, and the suite holds it to `Plan`'s digests and refusals on the same rows.
  `Session.receive` takes any of the three.
- **What the plan shows.** v0's plans are in the clear, so the verifier sees which KV pages were live in each pass: the
  batch's shape and context lengths over time, not their contents. Hiding them goes with the zero-knowledge openings:
  the plan committed, its distinctness and each opening's \(j = \text{plan}[p]\) proved, only \(L\) public (the draws
  need it).
- **The pass and its step.** A pass covers one step \(s\) of the served circuit's execution, which it names with its
  root (§4; for vLLM, the engine step its Store backend tags each pass with). The step is the requirement's (the Lean
  module `Definitions.Accounting.Requirement` in `Security/`), and the pass's plan is to be the
  blocks the circuit reads at \(s\), `readBlocks s` (`PlanMatches`). For a store that holds only what the circuit
  keeps live, that is the store resident when \(s\) begins (`Store.resident s`, under `StoreMatches`): the blocks
  registered or written at a step before \(s\), less those freed or overwritten at one. So a step's memory-map record
  takes effect after the step's reads: its writes are of blocks the step completed, or an earlier step did and the
  device has since encoded (deferred residency, §4), and its frees are of blocks whose values the step read last. A
  writer therefore records a free at the step of its values' last read, not at the next write's. The prover's audit
  side gives each pass its plan's change at its step (`Replay.take(s)`). It refuses a pass at a step whose record it
  has applied, or before the last pass's, and a record of a step before the last pass's (`memory_map.py`). The step
  is public structure, and the map and the plan are witness. Because the steps strictly increase, a step has at most
  one pass, so its reads are credited at most once; a step with no pass is credited nothing. Open:
  - *The plan's check.* That a pass's plan is the circuit's reads at its step, block for block, is the plan's proof
    unit's to check, inside zero knowledge, against the prover's memory map and the circuit's requirement (I4). The
    verifier here checks only that each pass names a step above the last.
  - *The timing.* The step says which reads a pass stands for, not when serving made them: nothing yet shows that the
    pass at \(s\) ran while serving executed \(s\). A certifier that stamps each step's record and each pass's root
    on one clock would. A prover may run serving ahead of its passes (`Ahead` in `benchmarks/remote_session/prover.py`),
    at most \(k\) steps past the pass in flight, \(k\) its own choice: each pass covers the newest step serving has
    handed, the steps serving ran past in between have no pass and are credited nothing, and the pass at \(s\) reads
    the store resident at \(s\) while serving executes later steps. That store must still be in place as the pass
    reads it, so serving writes into memory whose block was released at step \(r\) only once no pass yet to read is
    at a step up to \(r\) (the free fence). A write that skips the fence lands over a block a pass has yet to read,
    and a drawn receipt of it, of other bytes than the opening's block, rejects (§6). The record is memory-map/v1's
    as before, each pass still starts after the root before it and sends its root once its receipts are made (§5),
    and the verifier sees the same messages, so what it accepts is unchanged.
- **The requirement's link.** What the audit credits is the node's reads of its store. Counting them as the served
  circuit's reads needs A1's link (`StoreMatches`, `PlanMatches`, note:20261017T1847Z-draft-a1-requirement), and three
  of its gaps are on the audit's side. `ledger` takes the link as `requirement`, a `RequirementLink`: whether its
  proofs hold, the density \(D\), the circuit's read bits at each step, and the memory-map unit's verdict on the
  session's appends and its passes' heads and stale bounds (`mapped`, below). On it, a pass's credit is its credited
  receipts at \(D\) bits each, at most the circuit's read bits at the pass's step, and the ledger says it is linked
  (`Ledger.linked`); without it, the credit is the store's, flagged as such. The checks behind the link belong to the
  units named below, and the ledger takes them as given. Those units are sampled, so the link carries what their draw
  established: \(\varepsilon\), the most share of the linked credit the units it leaves unproved can carry (the
  passes whose plan unit is wrong and those holding a block a wrong memory-map unit wrote), except with
  probability \(\delta\). The session is credited \((1 - \varepsilon)\sum_\text{passes} \min(nD, r_s)\), a window
  its own sum less \(\varepsilon\) of the session's (the unproved share may all be one window's), and \(\delta\) adds
  to the ledger's error (`Ledger.error`, `error_horizon`). That \(\varepsilon\) bounds the unproved credit, not only
  the share of units drawn wrong, is the link's to establish: a wrong unit at a heavy step carries more than its
  share.
  - *Reads the circuit doesn't make.* The plan is the resident set, and the circuit's reads at a step can be fewer: a
    gather's unselected rows, requests absent from the step's batch, unrouted experts. Their receipts are capped at
    the circuit's reads at the pass's step (`read_bits`, `requirement.Requirement.read_bits`). That each receipted
    block is one the circuit reads at that step (`plan = readBlocks s`, so a block of an unread row is in no plan)
    is the plan unit's to prove (I4), against the memory map and the requirement, inside zero knowledge.
  - *Partly filled blocks.* A block holding fewer than \(B\) bits of the circuit's values (a request's last block,
    values of different lifetimes sharing a block) would overstate the credit by \(B/D\) if it counted whole. On the
    link, a receipt counts \(D\) bits, the fewest any planned block carries. \(D\) is the census's to prove from its
    layout rule (`Packs`), and \(D = B\) is the rule "full blocks only". The sampling term still charges a whole
    block for a wrong receipt.
  - *Registered blocks past their last read.* memory-map/v1 frees a registered block like a written one: a free of a
    slot below \(N\) releases it, and no later plan may hold it (`Session.release`). The writer records the free at
    the step that reads the block's values for the last time. That each free is at its values' last read is the
    memory-map unit's to prove (I4). Until it does, a dense decoder's weights, which every step reads, need no free,
    and the census must show it.
  - *The memory map's choices* (the red team's KB-2, note:20261017T1859Z-report-property-redteam §R4.2). Between a
    pass's seed and its reads, the writers could choose which pages to free, when to re-key a block, the order of a
    batch of appends, and dummy appends that shift every later \(j\). Each choice moves the bound keys or the live
    set, and none of them costs an encode beyond the writes. Freeing 100 of \(10^4\) live pages is 803 bits of
    choice, past what the advice bound tolerates. Even with the map fixed, a writer keeps choices the map doesn't
    read (padding and dead-value bits, run inputs, schedules), and the stacked sketch bound goes vacuous past about
    \(41 - \log_2(\text{passes})\) bits of them (note:20261017T1855Z-report-stacking-lean). The append coin (§3, §4)
    makes that advice 0: an appended block's key is \(b_j(c_a) = H(\text{seed}, j, d_j, c_a)\), and \(c_a\) is drawn
    after the append's root commits every \(d_j\) and its place, so no choice made before the commit can depend on an
    encoding. After the coin the prover's remaining choice is to append again, which draws a fresh coin: a retry
    that redraws unless the map makes it, which check 3 below covers. The audit charges no advice for these
    choices because W's memory map fixes them: memory-map/v1's frees and writes are functions of the served
    steps, fixed before the coins. That the session's appends, releases and plans are that map's is the memory-map
    unit's to prove (proofs', I4), inside zero knowledge, and the audit relies on it for exactly this:
    1. every append is a write of the map, of a block completed at the write's step or before it (§4's deferred
       residency records the write at the step before the block's first encoded read), appended once complete and
       before any plan holds it (§4 checks the latter), and sent after the passes of the steps up to its production
       step and before the rest: each pass's head is the chain through the appends of the steps below its own, so no
       append is bound into a seed at or before its step's;
    2. the appends are in the map's order, each at the index the verifier gave it (the store's next fresh one), and
       the map has no write the session doesn't append and no appended block it neither writes nor withdraws, a
       withdrawn one being a block it completed whose values' last read came at or before its encode landed (§4), so
       there are no dummy appends;
    3. a re-key (a release and the payload appended again, §4's lifetime cap) is the map's, at the cap, of the same
       payload, so \(d_j\) is unchanged and only its index, its coin and its key are fresh (KB-6), its free at the
       re-key's step and its new index's write when that encode lands; at the cap means its old index is below the
       stale bound of the first pass after its step (`WindowRecord.stale`), so the old block may be in no plan from
       that pass on, and no earlier re-key is taken: which blocks are re-keyed or freed stays witness;
    4. every free but a re-key's is at its values' last read, of a registered block or a written one (above);
    5. the map is the one the plan unit's plans are of (`PlanMatches` reads the same map).

    The unit's public input is the record's `appends`, the chain \(a\) over the session's appends (§4, §7), each
    pass's step and head (`WindowRecord.heads`, the chain its seed binds), and each pass's stale bound
    (`WindowRecord.stale`). The units chain their own appends step by step, and the link holds only if every pass's
    head is that chain through the appends of the steps below the pass's step (item 1) and the whole chain is
    `appends`, so the appends' places among the passes are bound; each map instance takes the stale bound of the pass
    after its step and refuses moving a value whose old index isn't below it (item 3). A prover that runs serving ahead
    (`Ahead`, "The pass and its step" above) still meets item 1: its steps without a pass have their appends sent
    between the passes around them, since it hands serving's records over, and sends their appends, only before the
    next pass's root. Its witness is the map, the appends' digests \(d_j\) and salts, and the plans. The releases stay
    witness, held to the map through the plans; the verifier here sees them in the clear and keeps released blocks
    out of later plans, a check deployment moves into the plan unit. `ledger` takes the unit's verdict as
    `RequirementLink.mapped(record)` and refuses a link whose verdict isn't true of the record's chain, heads and stale
    bounds: without it the credit is the store's, as without a link. The proof is proofs'; the audit side is the
    chain, the heads, the stale bounds and the hook.
- **KV registered mid-session.** A KV page's payload is chosen after the codec seed, so registration order can't bind
  its encoding the way it binds the weights' and filler's. Its bound key stops its writer picking its encoding (§3),
  but not stacking: sampled proofs show a payload is the circuit's actual computation, and an arbitrary circuit that
  can learn the key can compute another registered block's encoding as that payload. So a KV page is credited only
  on the property's verdict on its writers (§3, `credit_post_seed`), and then \(\mu_{\text{carrier}}\) charges what
  its carriers can hold (§1). Only bound blocks may be registered after the seed. The reference appends them (§4):
  payload roots committed before the passes that read them, fresh store indices, each page encoded under its bound
  key and its append's coin with \(d_j\) kept beside it, an opening of one carrying \(d_j\), which check 3 decodes
  under with the coin, and releases that keep a block out of every later plan. Open:
  - *The verdict's binding.* `ledger` takes the verdict as given (`PostSeedVerdict`: whether it holds, and its carrier
    bytes for a window's certified reads). That it is of the circuits that wrote this session's appends, and that
    their payloads are those circuits' values (sampled proofs), are I2's and I4's to check. That no appended block is
    credited past \(L\) the session checks itself (§4, `lifetime`); that the verdict was computed at the record's
    \(L\) is the verdict's to say.
  - *Appends across sessions.* A session's appends end with it, and the next session starts again from \(N\), so a
    page that outlives its session is appended again in the next, its encoding recomputed under its new index.
    Carrying appends over needs the next session to take the previous one's append roots and releases as its
    starting store.
  - *The device side.* A KV page's index is the store's next fresh one, not \(\text{slot} \cdot \text{LAYERS} +
    \text{layer}\): serving records each block it completes, once the device has encoded it (§4's deferred
    residency), with its slot, the index it gave it and its \(d_j\), each slot it frees, and each appended block it
    withdrew, freed before its encode landed, step by step
    (memory-map/v1, the program's I3, `memory_map.py`; a write releases the block its slot held, and vLLM's `kv_free`
    is a free, recorded at its values' last read, so the store's filler plan leaves the released blocks out), and the
    prover's audit side turns the record into appends, releases and plan changes (`Replay`; deferred, a step's
    appended blocks are those of appends already sent). The record is the prover's: \(d_j\) is witness, and the
    verifier sees only the append roots, the releases and the plans; `benchmarks/remote_session` carries them over
    the wire: each append its own message, answered with its coin (§4), sent after the root before the pass that
    first reads it and before that pass's reads, its blocks encoded once the coin arrives; then each root's message
    its releases, its plan as a change, and the pass's step. A KV block under the seed's key, as vLLM's encoded store
    (whose `Codec` is keyed by the store index alone) encodes them today, is not to be credited: its writer may have
    picked its encoding, and then its receipts need no read. `kv.cu` and `paged.cu` take the bound key as a
    per-block offset, the coin form unchanged since the offset is the key's output \(b_j(c_a)\), which serving does
    not pass yet.
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
- **KV receipts are the driver's receipts.** A KV page's receipt is the registration's one receipt kind (sketch-sha512:
  `SketchReceipter` under the pass's seed, its signs under `SIGN_DOMAIN`, hashed under `SKETCH_RECEIPT_DOMAIN`) of a
  registered KV block under the registration's block layout, committed in the pass's one tree and listed in its one
  plan with the weights'. A KV block is a registered block like any other: 4,096 bytes of the encoded store, 32 whole
  codec words, \(j = \text{slot} \cdot \text{LAYERS} + \text{layer}\) (§3), and its kernel's vectors pass `plan_of`
  (§4). The device's KV receipt as it stands (word_codec's `kv_receipt.py` and `kv.cu`'s `rc_sha8_kernel`, #1916)
  is a third kind:
  its own SHA-512 domain (`verity/word-codec/kv-receipt/sha512/v0`), ro-sketch-blake3 signs, and half-block vectors
  that take their 4,096 bytes from 16 tokens' head shares, zero-padded to whole blocks (RB-D2). The verifier can open
  none of it, so none of it is credited: no pass's plan holds a KV block until its receipts are the registration's kind
  over the registered KV layout, and a step that commits KV receipts of another kind keeps them out of its plan and
  its credited bytes.
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

`python -m verity.core.protocols.remote_bandwidth.timing` prints the grid. It covers passes of 20 and 50 ms, windows
of \(T\) = 1, 20 and 60 s (the stated window), RTT 1–50 ms (half each way), \(J\) = 0.05 ms each way and \(g\) =
50 µs, at the least allowance. After that come the openings, the stated claim through `ledger` and the store floor
(below), and §4's rank bound, grinding tries and sessions' failure. \(\mu_{\text{timing}}\) by session at the stated
\(T\) = 60 s, and the largest of a \(W\) = 100 session's windows' own (§1):

| pass | RTT | passes a window | one window | \(W\) = 10 | \(W\) = 100 | steady state | worst window |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 20 ms | 1 ms | 2,993 | 0.27% | 0.25% | 0.25% | 0.25% | 0.30% |
| 20 ms | 5 ms | 2,993 | 0.28% | 0.25% | 0.25% | 0.25% | 0.31% |
| 20 ms | 20 ms | 2,993 | 0.30% | 0.25% | 0.25% | 0.25% | 0.33% |
| 20 ms | 50 ms | 2,993 | 0.35% | 0.26% | 0.25% | 0.25% | 0.38% |
| 50 ms | 1 ms | 1,199 | 0.17% | 0.10% | 0.10% | 0.10% | 0.25% |
| 50 ms | 5 ms | 1,199 | 0.18% | 0.10% | 0.10% | 0.10% | 0.26% |
| 50 ms | 20 ms | 1,199 | 0.20% | 0.10% | 0.10% | 0.10% | 0.28% |
| 50 ms | 50 ms | 1,199 | 0.25% | 0.11% | 0.10% | 0.10% | 0.33% |

A session's ends are its only round-trip cost, so a long session reaches the steady state \(g/(t_{\text{pass}} + g)\)
whatever \(T\) is. At \(T\) = 1 s and \(W\) = 100 it is still 0.26–0.31% at 20 ms passes and 0.10–0.15% at 50 ms. A
pass reads distinct blocks, so it is at most a read of the store: about 61 ms for the node's 789 GB over every tier
at its measured 12.92 TB/s, as for each GPU's 94 GB of it at 1,539 GB/s.

A window by itself does pay its ends. Its span is \(T + A\), the pass in flight at its challenge belongs to the window
before, and its own chain starts only when that pass ends. So a window is charged \(A\) and up to a pass whatever
\(W\) is: about \((A + t_{\text{pass}})/(T + A)\) at worst, the last column. At the stated 60 s that is 0.30–0.38%
(20 ms passes) and 0.25–0.33% (50 ms). At the 20 s option it is 0.41–0.65% (20 ms) and 0.50–0.75% (50 ms), and at
the 1 s option 4.03–8.42% (20 ms) and 9.62–13.65% (50 ms).

**Openings** at the stated \(T\) = 60 s and δ = \(2^{-40}\): \(s = 33{,}273\) per window (555 a second,
\(\varepsilon_s = \ln(2^{40})/s\) = 0.083%), each about 8.5–9.1 KB (block, leaf, both salts, both paths). That is
282–303 MB a window per node, or 38–40 Mb/s. The 20 s option's \(\varepsilon_s\) = 0.25% takes 11,091 a window at the
same rate: 94–101 MB, the same 38–40 Mb/s. The 1 s option's 1% takes 2,773: 23–25 MB, or 0.19–0.20 Gb/s. Under a
registration that binds \(z\), PoUW's tile evidence comes on top unless PoUW's own sampled check covers the same
tiles. A Fiat–Shamir draw would multiply all of this by 1.5–2.6 (§6).

That is the per-window reading, each window's profile at its own δ, \(W\delta\) in all. Read over the horizon, the
windows' draws pool (`RateSessionDraw`, §8). Each window draws its \(s_t\) after its own roots from its own openings
coin, so however the prover adapts each window to the draws before it, every draw passes while
\(\sum_t s_t g_t \le (1-\bar d) \sum_t s_t\) with probability at most \((1-\bar d)^{\sum_t s_t}\), where \(g_t\) is
window \(t\)'s share of correct receipts. Set that to δ: except with δ, the wrong shares \(f_t = 1 - g_t\) have
\(\sum_t s_t f_t < \bar d \sum_t s_t\), and each \(f_t \le 1 - s_t/U_t\), since \(s_t\) distinct draws found \(s_t\)
correct receipts. The same event bounds every window's share by itself: \(f_t < \bar d \sum_u s_u / s_t\), about
\(\ln(1/\delta)/s_t\), with no union over the windows.

Daniel's claim is that hybrid (#1884): short by under 0.15% over a 10-minute horizon (for \(\mu_{\text{sample}}\),
below) and by under 1% in every window (for the whole \(\gamma_t\)), at \(2^{-40}\), with the window stated at
\(T\) = 60 s and 33,273 openings (13 Oct). For \(\mu_{\text{sample}}\) it
needs \(d \ge 2{,}773\) openings a window and \(d \cdot n \ge 18{,}470\) over the horizon's \(n\) windows. The stated
window's 33,273 give each window a floor of 0.083%, and the horizon's 10 windows \(\ln(2^{40})/(d\,n)\) = 0.0083%.
The 20 s option's 11,091 give each window 0.25%, and over its 30 windows the same 0.0083%. The 1 s option's 2,773
give each window 1%, and over its 600 windows 0.0017%.

`ledger` reports μ_sample both ways: `sample`, the per-window reading, and `horizon`, the pooled one (`pooled`: the
most \(\sum_t f_t U_t\) the bounds above allow), with `floor`, the largest \(f_t\) they allow, and γ for each
(`gamma`, `gamma_horizon`). It reports each window's the same way over its own span (`Ledger.windows`: each
\(f_t\), from `floors`, is its `floor`, and \(f_t U_t\) its `horizon`), and the worst window's γ either way beside the
session's (`worst`, `worst_horizon`). All of it starts from each window's profile and span, which the record keeps.

**The whole γ.** `timing.stated` reads the claim through `ledger`: an honest 10-minute session on one node of 8 RTX
PRO 6000s and their host, at the node's measured \(R\) = 12.92 TB/s (and at spec, 13.60 TB/s) and \(S_{\text{chip}}\)
= 3,045,746,688 B, under the default codec at the node's stated squaring ceiling. Over every tier, the store is the
same GB of each GPU's HBM and host DRAM's share (`timing.spanning`), and a pass is its read at \(R\). In HBM alone, it
is the same GB a GPU and nothing on host DRAM, and a pass is its read at the GPUs' 12.31 TB/s.
\(\mu_{\text{chip}}\) is about \(S_{\text{chip}}\) over the store, and a window's \(\mu_{\text{timing}}\) about
\((\text{RTT} + 2 t_{\text{pass}})/T\), so the store's size trades one against the other. Among the grid's stores 94 GB
a GPU, 789 GB over every tier with 61 ms passes, is best except at \(T\) = 1 s. The worst window's terms and its
\(\gamma_t\), at the measured \(R\) and at spec, and over the horizon's \(W\) windows the pooled
\(\mu_{\text{sample}}\) and the whole γ, at RTT 1 and 20 ms:

| \(T\) | openings a window | store, GB a GPU | \(\mu_{\text{timing}}\) | \(\mu_{\text{sample}}\) | \(\mu_{\text{chip}}\) | λ | \(\mu_{\text{rebuild}}\) | worst \(\gamma_t\) | at spec | \(W\) | pooled \(\mu_{\text{sample}}\) | horizon γ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 60 s | 33,273 | 789 GB, 94, every tier | 0.24–0.27% | 0.083% | 0.39% | 0.115% | 0.065% | 0.888–0.920% | 5.819–5.849% | 10 | 0.0083% | 0.660–0.663% |
| 60 s | 33,273 | 520 GB, 62, every tier | 0.23–0.26% | 0.083% | 0.58% | 0.117% | 0.065% | 1.076–1.107% | 5.998–6.028% | 10 | 0.0083% | 0.900–0.903% |
| 60 s | 33,273 | 252 GB, 30, every tier | 0.29–0.33% | 0.083% | 1.21% | 0.120% | 0.065% | 1.768–1.799% | 6.656–6.685% | 10 | 0.0083% | 1.657–1.660% |
| 60 s | 33,273 | 752 GB, 94, HBM | 4.93–4.96% | 0.079% | 0.39% | 0.110% | 0.065% | 5.565–5.595% | 10.263–10.292% | 10 | 0.0079% | 5.348–5.351% |
| 20 s | 11,091 | 789 GB, 94, every tier | 0.45–0.54% | 0.25% | 0.38% | 0.115% | 0.065% | 1.259–1.353% | 6.172–6.261% | 30 | 0.0083% | 0.660–0.663% |
| 20 s | 11,091 | 520 GB, 62, every tier | 0.50–0.59% | 0.25% | 0.58% | 0.117% | 0.065% | 1.511–1.604% | 6.411–6.500% | 30 | 0.0083% | 0.900–0.903% |
| 20 s | 11,091 | 252 GB, 30, every tier | 0.40–0.49% | 0.25% | 1.20% | 0.120% | 0.065% | 2.034–2.127% | 6.908–6.996% | 30 | 0.0083% | 1.657–1.660% |
| 1 s | 2,773 | 252 GB, 30, every tier | 2.60–4.38% | 0.95–0.97% | 1.16–1.18% | 0.115–0.118% | 0.065% | 4.925–6.664% | 9.655–11.308% | 600 | 0.0017% | 1.650–1.653% |
| 1 s | 2,773 | 789 GB, 94, every tier | 8.00–9.62% | 0.90–0.92% | 0.35–0.36% | 0.105–0.106% | 0.065% | 9.441–11.033% | 13.947–15.460% | 600 | 0.0017% | 0.654–0.657% |
| 90 s | 11,091 | 789 GB, 94, every tier | 0.17–0.19% | 0.25% | 0.39% | 0.116% | 0.065% | 0.986–1.007% | 5.913–5.932% | 6 | 0.042% | 0.699–0.703% |

At the stated \(T\) = 60 s, then, every window is under 1% for the whole \(\gamma_t\) at the measured \(R\):
0.888–0.920% with 94 GB a GPU over every tier, and 1.08–1.11% at 62 GB a GPU. At spec it is 5.8% at best, the
measured kernels' shortfall from the datasheets. The same 94 GB a GPU in HBM alone is 5.56–5.59%, host DRAM's idle
4.7% of \(R\) in \(\mu_{\text{timing}}\). At the 20 s option, every window under 1% holds for \(\mu_{\text{sample}}\)
(0.25%) and not for the whole \(\gamma_t\), which is 1.26% at best. At \(T\) = 90 s with 11,091 a window it is
0.99–1.01%. The 1 s option's per-window claim is \(\mu_{\text{sample}}\) alone. Its whole \(\gamma_t\) is 5–11%,
nearly all \(\mu_{\text{timing}}\): the window's idle ends, which the record states exactly.

**The store floor.** No \(T\) brings every window under 1% on too small a store. As \(T\) grows, a window's
\(\mu_{\text{sample}}\) (at 555 openings a second) and its ends vanish, and its \(\gamma_t\) tends to
\((g + (S_{\text{chip}} + \Lambda)/R + \mu_{\text{rebuild}}(t_{\text{pass}} + g))/(t_{\text{pass}} + g)\),
where \(t_{\text{pass}}\) is the store's read at \(R\) and \(\Lambda\) a pass's `named_bits` and `slack_log2` in bytes,
whose share a block depends on the store's size (`timing.store_floor` solves for it). Below 373.2 GB over every tier
(44.5 GB a GPU), \(\mu_{\text{chip}}\), λ and \(\mu_{\text{rebuild}}\) alone exceed 1%. With the chain's steady state
at \(g\) = 50 µs, no \(T\) meets 1% below 451.1 GB (53.7 GB a GPU). The suite checks the closed form against `ledger`
on a 20-minute window, 2% either side of it.

**The horizon's 0.15%** is stated for its pooled \(\mu_{\text{sample}}\), with the whole γ reported beside it (Daniel,
13 Oct: the default put to him, which applied with no objection by its deadline). The pooled
\(\mu_{\text{sample}}\) is 0.0083% at every stated and optional \(T\), and the whole γ 0.66% at best.
\(\mu_{\text{chip}}\) (the node's \(S_{\text{chip}}\) over its store, 0.39% at 94 GB a GPU over every tier) exceeds
0.15% by itself, and λ (37.9 bits a 4 KiB block, 0.116%) does with \(\mu_{\text{rebuild}}\). Neither λ nor
\(\mu_{\text{chip}}\) shrinks with more openings or longer windows. The whole γ meets 0.15% at no
\(T\) and no count of openings today. Even with λ and \(\mu_{\text{chip}}\) at 0 it would be 0.159–0.162%: the
chain's steady state (0.082% at \(g\) = 50 µs and 61 ms passes, 0.086–0.089% with the session's ends),
\(\mu_{\text{rebuild}}\) (0.065%) and the pooled \(\mu_{\text{sample}}\).

**What λ and \(\mu_{\text{chip}}\) would take** (the grid's last sections, at 789 GB over every tier and 61 ms
passes). λ is AuditRO's \(C(q, k)\) for a pass's \(k\) receipts, bounded by \(q^k/k!\), which takes no assumption:
about \(\log_2(e q/k)\) a receipt, 37.9 bits at 4 KiB (0.116%). Under 0.05% it may be 16.4 bits a 4 KiB receipt. Two
ways get there:

- **Wider receipts.** A receipt of \(m\) blocks gives 0.059%, 0.040% and 0.030% at 8, 12 and 16 KiB (bounded by
  \(q^k\) instead, 64 bits a receipt, 0.098%, 0.065% and 0.049%). On the GPU a receipt's sketch is the same work a byte, and
  its hash and share of the tree are \(1/m\). #1916's τ of 1.26 a pass scales to about 1.09 at 16 KiB (from its
  measured parts; the whole isn't measured). The openings pay instead. Each carries \(m\) blocks: 20.9 KB at
  16 KiB with a payload leaf a receipt, or 27.0 KB with PoUS's 4 KiB leaf. That is 2.4–3.0 times today's, or
  0.70–0.90 GB a window and 93–120 Mb/s per node. Each opening's proof unit also decodes \(m\) blocks (check 3).
  sketch-sha512's rate a compression goes from \(2^{-95.12}\) to \(2^{-93.88}\) at 16 KiB's 512 sectors.
  `RateAuditSketch` would need restating for a receipt of \(m\) blocks, and the registration would name \(m\).
- **Fewer queries.** At 4 KiB, 0.05% needs \(q \le 2^{42.5}\) queries to a pass's oracle:
  874 s of one GPU's SHA-512, 109 s of the node's. No prover hardware is trusted, so the protocol can't state such a
  bound.

\(\mu_{\text{chip}}\) is \(S_{\text{chip}}\) over a pass's reads, and a pass reads each block at most once. So it
falls only with more store read at \(R\): 6.09 TB over every tier for 0.05%, 7.9 times the node's 768 GB of HBM. At
94 GB a GPU and host DRAM's share it is 0.386%. Host DRAM beyond its share doesn't substitute: at its 606.8 GB/s, a
pass longer than the HBM's 61 ms read leaves the HBM idle. At 200 ms, \(\mu_{\text{chip}}\) is 0.118% and 66% of the
capacity is idle.

What would lower \(\mu_{\text{chip}}\) instead is a flush. Each pass's root would key a computation that needs the
chip's whole storage, and its output would go into the next seed, so the chip's state carries nothing of the store
into the next pass. That needs three things:

- **An assumption.** A bandwidth-hard function on \(S_{\text{chip}}\): evaluating it while holding \(X\) bytes of
  the store on chip costs at least \(X\) bytes more of device-memory traffic, with \(R\) bounding reads and writes
  together. Spilling the held bytes instead costs \(2X\).
- **The verifier's work.** It recomputes the flush for every pass whose seed it uses, which at 33,273 openings is
  nearly every pass.
- **The GPU's time.** At least one write and one read of \(S_{\text{chip}}\): 26–105 µs at 5–20 TB/s on chip
  (unmeasured). That adds 0.04–0.17% to the steady state.

At the stated \(T\) = 60 s with 789 GB over every tier and RTT 1 ms, the horizon's whole γ is 0.66% (0.74% bounded
by \(q^k\)). 16 KiB receipts make it 0.58%, and a 26 µs flush with them 0.23%. The worst window's \(\gamma_t\) goes
from 0.888% to 0.80% and 0.46%. With λ and \(\mu_{\text{chip}}\) at 0 the horizon's γ is 0.16%, because the chain's
steady state and \(\mu_{\text{rebuild}}\) alone exceed 0.15%.

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
