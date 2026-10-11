# The bandwidth audit, v0

Status: v0, ready for a red team. The Python reference and verifier are `audit.py` and `tags.py`, pinned by
`tests/vectors.json`; the GPU tagger is `benchmarks/bandwidth`, checked bit for bit against them and timed on the
RTX PRO 6000 (§6). The guarantee's Lean statement and proof are lane `ra-bandwidth-lean`'s, under `Security/`'s
`Accounting` area; until it lands, §5 is the statement that lane proves, not a proved one.

## 1. What a window certifies

A window certifies that a device moved at least `I` bytes of a fixed store `E` from its memory into the chip within
`T + RTT` of receiving a fresh seed: its memory delivered at least `I / (T + RTT)` bytes per second to the audit. With
probability at least `1 − δ_s − δ_γ` over the verifier's coins, an accepted window has

~~~text
I  ≥  ((1 − ε_s)·B − k_γ) · info  −  S_chip  −  cap_side · (T + RTT)
~~~

where `B` is the store's block count, `info` the bytes of a block the tag binds (§4), `ε_s = 1 − δ_s^(1/s)` for `s`
samples, `k_γ` the most unread blocks whose tags come out right by luck (§5), `S_chip` the bytes the chip can hold
when the window starts, and `cap_side` the rate of every path into the chip other than device memory. Each block
whose tag is right was read from device memory in the window, held on chip, brought over a side path, or guessed; none
is rebuilt from `W`, because `T + RTT` is shorter than one P2 root (§5).

## 2. The store and its setup

`E` is PoUS's P2 store `C*` (`verity.core.protocols.pous.schemes.p2`): `B = 2^19` blocks of 2,056 bytes (1.08 GB),
encoded from the useful data `W` under PoUS's setup (`pous.setup`: the commitment to `W`, then the verifier's salt,
then `vk`). The audit reuses that setup and `vk` unchanged: a block's identity is its leaf under `vk.root`
(`pous.verifier.leaf_hash`, RFC 6962 SHA-512 tree), and the verifier holds only `vk`.

Each block of `C*` is two slow roots of its keyed payload, `c = ρ(σ_K(ρ(m + t)))`. Under P2's assumptions a block is
incompressible except through `W`, and rebuilding it from `W` takes two roots in sequence.

## 3. A window

1. **Seed.** The verifier derives the window's key from its own secret,
   `key = derive(source, "verity/bandwidth/window/v1", {root: vk.root, salt: vk.salt, window: n})`
   (`verity.ml.randomness`), sends `σ = key.stream("seed", 32)` and starts its clock (`Window.start`). A window number
   is used once.
2. **Tags.** The prover computes `y_j = Tag(σ, j, E_j)` for every `j < B` (§4), packs them into leaves, and returns the
   RFC 6962 SHA-512 root of the leaves (§4). The verifier stops its clock on the root (`Window.commit`); the root is on
   time iff the elapsed time is at most `T + RTT` (`Params.limit_ns`).
3. **Samples.** Only for an on-time root, the verifier draws `s` indices, `key.uniform(B, ("sample", t))` for `t < s`,
   uniform with replacement, and sends them. A late or malformed root closes the window rejected and draws nothing.
4. **Openings.** For each sample `i`, in order, the prover sends `E_i` with its audit path to `vk.root`, and the tags
   of `i`'s leaf with the leaf's path to the committed root (`TagCommitment.open`). The verifier checks both paths and
   recomputes `y_i` from `σ`, `i` and `E_i` (`Window.verdict`). Openings are not timed: the root binds every tag.

The verifier accepts iff the root is on time and every opening checks. It refuses a second start, a second root, or a
window number used before, and never redraws. The verdict publishes one bit (`Verdict.public`); which check failed,
which sample and the elapsed time stay with the verifier. Openings reveal blocks of `C*` to the verifier, so the
verifier that runs this is PoUS's memory challenger, which already holds `vk` and publishes only verdicts.

## 4. Tags

~~~text
Z(x)           x zero-padded to whole 128-byte blocks
prefix(D, σ)   Z(D ‖ frame(name) ‖ frame(σ))              one SHA-512 block: a device starts each block from its midstate
               D = "verity/bandwidth/tag/v1\0" for tags, "verity/bandwidth/tag-leaf/v1\0" for leaves
ro-sha512/v1   y_j = SHA512(prefix ‖ u64be j ‖ E_j)                                       64 bytes
gf31-poly/v1   y_j = Σ_i (w_i mod p) · g_j^(W−1−i) mod p,  p = 2^31 − 1                    4 bytes, u32be
               w_0 … w_(W−1): E_j's little-endian u32 words (W = 514), by Horner
               g_j = 2 + x, x the first value below p − 2 among the sixteen u32be words, each shifted right by
                     one, of SHA512(prefix ‖ u64be j ‖ u32be c), for c = 0, 1, …
leaf_c         SHA512(prefix(leaf) ‖ u64be c ‖ y_(cL) ‖ … ‖ y_(cL+L−1)),  L = 64 / tag bytes; the last leaf may be short
root           RFC 6962 SHA-512 tree over the leaves: node = SHA512(0x01 ‖ left ‖ right)
~~~

`frame` is `verity.ml.randomness.frame`. `σ` comes before the block, so no state over a block exists before `σ`.
The leaves of `gf31-poly/v1` hold 16 tags (32,768 leaves at `B = 2^19`), those of `ro-sha512/v1` one.

| | `ro-sha512/v1` | `gf31-poly/v1` |
|---|---|---|
| role | the tag the Lean lane proves first | the deployed candidate: `benchmarks/pous/hbm_audit`'s hash, a fresh point per block |
| `γ`, a tag right without the block it binds | `2^-512` (random oracle) | `(W − 1)/(p − 2) ≈ 2^-22` (two blocks that differ mod p collide) |
| `info`, bytes of a block it binds | 2,056 | 1,991.75 (`⌈W·log2 p⌉` = 15,934 bits) |
| honest work per block after the seed | 17 SHA-512 compressions | one compression for `g_j` and 514 multiply-adds mod p |

`gf31-poly/v1`'s word map is not injective: `w`, `w + p` and `w + 2p` agree mod p, so its tag binds a block's
words mod p, 96.9% of the block (§7, finding 1).

## 5. The bound, and what it rests on

`bound(params, B, block_bytes, chip_bytes=…, side_bytes_per_s=…, delta_samples=…, delta_guess=…)` computes §1's `I`.

- **Wrong tags.** A root whose tags are wrong on a fraction `ε` of blocks passes `s` independent samples with
  probability `(1 − ε)^s`, at most `δ_s` once `ε ≥ ε_s`. At `δ_s = 1%`, `s = 459` gives `ε_s ≤ 1%` (`samples_for`).
- **Lucky tags.** An unread block's committed tag is right with probability at most `γ`, independently across blocks
  (the seed's tags and points are independent per block under the random oracle). More than `k_γ` are right with
  probability at most `C(B, k_γ + 1)·γ^(k_γ + 1) ≤ δ_γ` (`guess_slack`): `k_γ = 0` for `ro-sha512/v1`, and 8 for
  `gf31-poly/v1` at `B = 2^19`, `δ_γ = 2^-40`. A prover that holds part of a block and guesses the rest is the
  compression argument the Lean lane makes; `bound` counts only all-or-nothing reads.
- **On chip.** Whatever the chip holds when `σ` arrives, at most `S_chip` bytes: on the RTX PRO 6000, 128 MiB of L2
  and 188 SMs × (256 KiB of registers + 128 KiB of L1 and shared memory), 208,142,336 bytes, 19.3% of `E`, from the
  device's own properties (`device.json` of the run in §6).
- **Side paths.** Every other path into the chip during `T + RTT`, at most `cap_side` bytes per second. Host RAM is
  not part of the allowed hardware (Daniel, 3 Oct 2026), so the side paths are the GPU's PCIe Gen 5 x16 link,
  shared by host, NVMe, NIC and peer GPUs; the RTX PRO 6000 Server has no NVLink. `cap_side` is the link's raw rate,
  32 GT/s × 16 lanes at 128b/130b = 63.0 GB/s; the measured rates come close (§6).
- **No rebuild.** `Params` refuses `T + RTT ≥ regen_floor_ns`, by default the fastest cooperating P2 root,
  `2 · ROOT_CALL_NS = 1,708,058 ns`; the fastest root measured on the RTX PRO 6000 takes 18.53 ms (floor 2.26 ms,
  art:f26f917e0605e2974f44a9b0f8ae7f4fe4a78b95181deaf8d8c457ce98c64ad3). One root, not two, so a prover holding a
  block's inner root `ρ(m + t)` (as large as the block) still rebuilds nothing.

Assumptions the statement names: SHA-512 as a random oracle (the seed, `g_j`, `ro-sha512/v1`) and its collision
resistance (`vk`, the tag tree); P2's (its store is incompressible except through `W`, and SeqRoot); the side-path cap
and `S_chip` as measured for the device; a live verifier that keeps its secret until the window starts.

## 6. Parameters

`T` is set from the honest tag pass measured on the device, with headroom, and must leave `T + RTT` under the rebuild
floor. `T` and `s` are fixed here once a red team has read this; the proposal is `gf31-poly/v1`, `s = 459` and
`T = 1.1 ms`, which leaves up to 0.6 ms for `RTT`.

The GPU tagger (`benchmarks/bandwidth/run.sh`) on one RTX PRO 6000 Blackwell Server Edition of vy-nebius-2, under a
timed lease: run `r20261011-225829-61f9`, art:eb242aa584c8b60b7f36b63f6077d96893fd50049bc7d950bbb9c0be2e839d28. Its
gates passed before anything was timed: both tags' roots over the full store (`B = 2^19`) and over the pinned vectors'
37 blocks equal the reference's, with the output buffers poisoned, and a pass without the tag kernel does not match.
Times are medians of 100 passes, each after a 512 MiB write that clears L2 (p99 in brackets).

| | time | rate over `E` (1.078 GB) |
|---|---|---|
| raw device read of `E` | 0.730 ms | 1,476 GB/s |
| `gf31-poly/v1` pass: tags and leaves, then the tree | 0.910 ms (0.921) = 0.767 + 0.141 | 1,185 GB/s |
| `gf31-poly/v1` pass, host clock to the root on the host | 0.924 ms (0.938) | |
| `gf31-poly/v1` whole window through `audit.Verifier`, 20 windows | 0.998 ms (max 1.030), all accepted | |
| `ro-sha512/v1` pass: tags and leaves, then the tree | 1.728 ms (1.734) = 1.415 + 0.313 | 624 GB/s |
| `ro-sha512/v1` whole window, 20 windows | 1.737 ms (max 1.780), all late | |
| PCIe host to device, pinned copy / a kernel reading pinned host memory | | 56.6 / 51.4 GB/s |

`gf31-poly/v1`'s tag kernel runs at 95% of the raw read rate; the rest of its pass is the tree, whose 15 levels of
two dependent compressions each set its length. A prover that tags only a fraction `f` of the blocks saves about that
fraction of the tag kernel: 0.906, 0.871, 0.835 and 0.552 ms at `f = 0, 5, 10, 50%`. With `s = 459`, a window that
skips 1% is rejected with probability 99%, and one that skips 5% with probability 1 − 6·10^-11. Both skip windows
run on the device were rejected.

`bound` at `s = 459`, `δ_s = 1%`, `δ_γ = 2^-40`, `S_chip = 208,142,336` bytes and `cap_side = 63.0 GB/s`:

| tag | `T`, `RTT` | `I` | share of `E` | `I / (T + RTT)` |
|---|---|---|---|---|
| `gf31-poly/v1` | 1.1 ms, 0 | 756,352,602 bytes | 70.2% | 688 GB/s |
| `gf31-poly/v1` | 1.1 ms, 0.354 ms | 734,045,156 bytes | 68.1% | 505 GB/s |
| `gf31-poly/v1` | 1.708 ms (the largest `Params` allows), 0 | 718,035,657 bytes | 66.6% | 420 GB/s |

At `T = 1.1 ms` the certified rate is 47% of the raw read rate. The rest goes to the chip's storage (19.3% of `E`),
the side path (6.4%), `gf31-poly/v1`'s binding (3.1%), the sampling error (1%), and the time the honest pass takes
beyond a raw read.

## 7. Findings and open questions for the red team

1. **`gf31-poly/v1` binds 96.9% of a block.** A prover can keep `E'` with each word reduced mod p (15,934 of 16,448
   bits per block) and read 3.1% less. `bound` already counts `info = 1,991.75` bytes, so the statement holds, but the
   deployed tag should bind the whole block: options are GF(2^61 − 1) over 32-bit words (injective, `γ ≈ 2^-52`) or
   31-bit limbs of the block's bit string. Either is a new tag name, not an edit of this one.
2. **The store's leaves are PoUS's, not P2 spec §4's.** `vk`'s leaf is the 99-byte little-endian leaf of
   `pous.verifier`; the P2 spec's is 107 bytes, big-endian (`benchmarks/pous/p2_v1/wire.py`). The audit uses what PoUS
   uses on `main` and inherits its resolution.
3. **`S_chip` and `cap_side` are device numbers, not proofs.** They enter the bound as parameters; a device whose chip
   holds more, or with another path in, changes them.
4. **Per-block independence of lucky tags** holds in the random-oracle model for both constructions; the GF31 tag's
   `γ` is over `g_j` alone, which is uniform on `[2, p − 1]` given everything but `σ`.
5. **Openings disclose sampled blocks** of `C*` (an encoding of `W`) to the verifier; the memory challenger already
   holds `vk` and publishes only verdicts, so nothing leaves it.
6. **`ro-sha512/v1` does not fit on the RTX PRO 6000.** Its honest pass takes 1.73 ms, past the 1.708 ms rebuild
   floor, so no `T` that `Params` allows admits it (§6). It is SHA-512-bound (18 compressions per block), not
   memory-bound. It remains the tag the proof treats first; deployment needs `gf31-poly/v1` or its injective
   successor.
7. **`S_chip` counts L2, the register files and L1/shared memory.** Smaller on-chip stores (instruction and
   constant caches, copy-engine buffers) are left out. A red team should confirm that none of them can hold a useful
   share of `E` across a window's start.

## 8. Code

- `tags.py`: `get(name)`, each construction's `keyed(σ)(j, block)`, `guess`, `info_bits`; the tag tree (`leaves`).
- `audit.py`: the prover's reference (`Store`, `tag_pass(store, tag, σ) -> TagCommitment` with `root` and
  `open(j) -> Opening`), the verifier (`Verifier(vk, params, source).window(n)`: `start() -> σ`, `commit(root) ->
  samples`, `verdict(openings) -> Verdict`), and `bound`.
- `tests/bandwidth_vectors.py` writes `tests/vectors.json`: tags at 2,056 bytes for both constructions, window seeds and
  samples, and a whole window over 37 blocks.
- `benchmarks/bandwidth`: the GPU tagger (`tagger.cu`), its harness API (`gpu.py`: `Tagger.tag_pass(tag, store, blocks,
  σ, workspace)` returns the root; `open(j, store_tree)` gives the `Opening` this verifier checks), and the run of §6.
