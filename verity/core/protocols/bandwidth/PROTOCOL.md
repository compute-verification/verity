# The bandwidth audit, v0

Status: v0 with the certified tag `ro-blake3/v1`, ready for the red team's re-review. The Python reference and verifier
are `audit.py` and `tags.py`, pinned by `tests/vectors.json`; the GPU tagger is `benchmarks/bandwidth`, checked bit for
bit against them and timed on the RTX PRO 6000 (§6). The guarantee is `Bandwidth.Guarantees.AuditRO`, in the
`Accounting` area of `Security/`'s guarantees, proved as `Bandwidth.SecurityProofs.AuditRO` with no named assumption
by lane `ra-bandwidth-lean` (PR #1806); its completeness is `HonestPasses`. Until #1806 is on `main`, the statement is
that PR's. `T` and `s` are proposed in §6 and fixed once the red team has read this.

## 1. What a window certifies

A window certifies that a device moved at least `I` bits of a fixed store `E` from its memory into the chip within
`T + RTT` of receiving a fresh seed: its memory delivered at least `I / (T + RTT)` to the audit. For a certified tag
(`ro-blake3/v1`, `ro-sha512/v1`), an accepted window has

~~~text
I  ≥  k·b − S − c − G·b − λ   bits,     k = ⌈(1 − ε_s)·N⌉
~~~

except with probability at most

~~~text
((k − 1)/N)^s  +  s/2^h  +  C(q, k)/2^λ
~~~

over the store, the oracle and the verifier's samples. `N` is the store's block count and `b` a block's bits; `S` the
bits the chip holds when `σ` arrives; `c` the bits every side source can deliver in `T + RTT`; `G` the blocks the
device can rebuild from setup within the window; `q` its oracle queries; `h` a tag's bits; `s` the samples; `λ` the
slack the last term charges. The three terms are the three ways a sampled block passes without its contents on chip
at a query: the samples miss the blocks the device never queried; a tag is guessed; a device queried `k` blocks with
less information than they hold, which would describe the store in fewer bits than it has.

At the deployed point, `ε_s = 1/64`, `s = 1,849`, `h = 256` (`ro-blake3/v1`), `q ≤ 2^40` and `λ = 40·k + 41`:

| | value |
|---|---|
| `N`, `b` | `2^19` blocks of 16,448 bits (2,056 bytes); `E` is 1,077,936,128 bytes |
| `k` | 516,096 blocks, `E` less 1/64 (16.8 MB) |
| `λ` | 20,643,881 bits = 2,580,485 bytes, 0.24% of `E` |
| error | `2^-42.01` (samples) + `2^-245.1` (guesses) + `2^-41` (compression) ≤ `2^-40.42` |

`G = 0`, because `Params` refuses `T + RTT` at or past one rebuild (§5). `S` and `c` are the device's (§5); at
`T = 1.2 ms` and `RTT = 0`, `I` is 774,752,092 bytes, 71.9% of `E`, 646 GB/s (§6). `audit.bound` computes `I` and the
error; `test_bound_is_the_guarantee_at_the_proofs_point` checks it against the formula.

`gf31-poly/v1` is an uncertified fast check: nothing bounds a device's ingress through it (§4), and `bound` refuses it.

## 2. The store and its setup

`E` is PoUS's P2 store `C*` (`verity.core.protocols.pous.schemes.p2`): `N = 2^19` blocks of 2,056 bytes (1.08 GB),
encoded from the useful data `W` under PoUS's setup (`pous.setup`: the commitment to `W`, then the verifier's salt,
then `vk`). The audit reuses that setup and `vk`, with one change of input: a block's identity is its leaf under
`vk.root` (`pous.verifier.leaf_hash`, RFC 6962 SHA-512 tree), and the verifier holds only `vk`.

**One store per device.** The verifier draws each device's salt with the device's identity bound in,

~~~text
salt = derive(source, "verity/bandwidth/salt/v1", {w: commitment, device: id}).stream("salt", 24)
~~~

(`audit.draw_salt`), where `id` is the identity the device attests (its GPU's UUID, with its SKU recorded beside it),
and runs PoUS's setup with that salt in place of `pous.setup.draw_salt`'s. PoUS's setup takes the salt as an input and
needs only that it is drawn after the commitment to `W`, from the verifier's secret or a later beacon round, which
this one is, so PoUS is unchanged. Devices that hold one `W` under one secret then hold different stores, so their
`vk`, seeds and tags differ, and one device's commitment fails another's window
(`test_one_devices_commitment_fails_anothers_window`). Without it, two such devices hold byte-identical stores and get
the same `σ` for the same window number, and one device's root passes both windows.

The salt is the place for the device, not the window key. With the device only in the window key, two devices holding
byte-identical stores get different seeds, but one device that reads each block once can tag it under both seeds and
send the other device's tags (16.8 MB a window, under `c`) over its side link. Distinct stores are what make the
ingress of devices audited together add up (§5), and the window key, which binds `vk.root` and `vk.salt`, carries the
device without a change.

Each block of `C*` is two slow roots of its keyed payload, `c = ρ(σ_K(ρ(m + t)))`. Under P2's assumptions a block is
incompressible except through `W`, and rebuilding it from `W` takes two roots in sequence.

## 3. A window

1. **Seed.** The verifier derives the window's key from its own secret,
   `key = derive(source, "verity/bandwidth/window/v1", {root: vk.root, salt: vk.salt, window: n})`
   (`verity.ml.randomness`), sends `σ = key.stream("seed", 32)` and starts its clock (`Window.start`). A window number
   is used once.
2. **Tags.** The prover computes `y_j = Tag(σ, j, E_j)` for every `j < N` (§4), packs them into leaves, and returns the
   RFC 6962 SHA-512 root of the leaves. The verifier stops its clock on the root (`Window.commit`); the root is on time
   iff the elapsed time is at most `T + RTT` (`Params.limit_ns`).
3. **Samples.** Only for an on-time root, the verifier draws `s` indices, `key.uniform(N, ("sample", t))` for `t < s`,
   uniform with replacement, and sends them. A late or malformed root closes the window rejected and draws nothing.
4. **Openings.** For each sample `i`, in order, the prover sends `E_i` with its audit path to `vk.root`, and the tags
   of `i`'s leaf with the leaf's path to the committed root (`TagCommitment.open`). The verifier checks both paths and
   recomputes `y_i` from `σ`, `i` and `E_i` (`Window.verdict`). Openings are not timed: the root binds every tag.

The verifier accepts iff the root is on time and every opening checks. It refuses a second start, a second root, or a
window number used before, and never redraws. One window over a store is open at a time: from `σ` until its root
arrives, or until `T + RTT` has passed without one, the verifier refuses to start another window over the same store,
which stays unstarted and may start once the first has closed. Two windows open at once could both be answered from
one read of each block, so their ingress would not add; with one open at a time, each window's ingress is read within
its own interval. A deployment keeps one `Verifier` per store, which enforces this
(`test_one_window_over_a_store_is_open_at_a_time`). The verdict publishes one bit (`Verdict.public`); which check failed,
which sample and the elapsed time stay with the verifier. Openings reveal blocks of `C*` to the verifier, so the
verifier that runs this is PoUS's memory challenger, which already holds `vk` and publishes only verdicts.

A deployment may open a window when the prover says it is ready: the bound holds for every device state laid out
before `σ`, so the prover's preparation costs the verifier nothing. The honest prover's preparation on the GPU is one
untimed pass under a throwaway seed, then a read of 512 MiB of another buffer, so the window starts with its launch
path warm and L2 holding clean lines and no part of `E` (§6).

## 4. Tags

~~~text
Z(x)           x zero-padded to whole 128-byte blocks
prefix(D, σ)   Z(D ‖ frame(name) ‖ frame(σ))        86 bytes padded to 128: exactly SHA-512's first message block
               D = "verity/bandwidth/tag/v1\0" for tags, "verity/bandwidth/tag-leaf/v1\0" for leaves
ro-blake3/v1   y_j = BLAKE3-keyed(K, u64be j ‖ E_j),   K = SHA512(prefix(tag, σ))[0:32]      32 bytes
ro-sha512/v1   y_j = SHA512(prefix(tag, σ) ‖ u64be j ‖ E_j)                                64 bytes
gf31-poly/v1   y_j = Σ_i (w_i mod p) · g_j^(W−1−i) mod p,  p = 2^31 − 1                      4 bytes, u32be
               w_0 … w_(W−1): E_j's little-endian u32 words (W = 514), by Horner
               g_j = 2 + x, x the first value below p − 2 among the sixteen u32be words, each shifted right by
                     one, of SHA512(prefix(tag, σ) ‖ u64be j ‖ u32be c), for c = 0, 1, …
leaf_c         SHA512(prefix(leaf, σ) ‖ u64be c ‖ y_(cL) ‖ … ‖ y_(cL+L−1)),  L = 16 (1 for ro-sha512/v1);
               the last leaf may be short
root           RFC 6962 SHA-512 tree over the leaves: node = SHA512(0x01 ‖ left ‖ right)
~~~

`frame` is `verity.ml.randomness.frame`. `ro-blake3/v1`'s message is 2,064 bytes, three BLAKE3 chunks (1,024, 1,024
and 16 bytes): 33 chunk compressions and 2 parent compressions, each starting from `K`, which keyed mode puts in every
chunk's and every parent's chaining value. Its leaf of 16 tags takes 5 SHA-512 compressions after the prefix.

**Every tag is seed-first, `H(σ, j, E_j)`.** No state over any byte of `j` or `E_j` can be computed before `σ`:

- `ro-sha512/v1`: the prefix is 86 bytes (a 24-byte domain, the 21-byte framed name, the 41-byte framed `σ`) padded to
  128, so `σ` lies wholly in SHA-512's first block and `u64be j` starts the second. Every chaining value over `j` or
  `E_j` descends from a compression of `σ`.
- `ro-blake3/v1`: `K` is a SHA-512 of the same prefix, and every BLAKE3 compression of the message starts from `K`.
- `gf31-poly/v1`: the point `g_j` is a SHA-512 from the prefix's midstate.

This is what lets a random oracle fresh at `σ` model the tag. A tag with the block before the seed, `H(E_j, σ)`, has a
chaining state after `E_j` (64 bytes for SHA-512, 32 for a BLAKE3 chunk under a public key) that a device computes
before the window and keeps instead of the block: it then answers every tag reading 1/32 of `E` or less, and the
window certifies nothing. `test_every_tag_is_seed_first` checks, for every tag, that the framed `σ` lies within the
first 128 bytes it hashes and that the tag of a block changes with `σ` alone.

| | `ro-blake3/v1` | `ro-sha512/v1` | `gf31-poly/v1` |
|---|---|---|---|
| role | certified, the deployed tag | certified, the proof's first instance | uncertified fast check |
| `h`, a tag's bits | 256 | 512 | not a random oracle |
| honest work per block after `σ` | 35 BLAKE3 compressions | 17 SHA-512 compressions, its leaf another | one SHA-512 compression for `g_j`, 514 multiply-adds mod p |
| honest pass on the RTX PRO 6000 (§6) | 0.98 ms | 1.73 ms, past the rebuild floor | 0.88 ms |

**`gf31-poly/v1` certifies nothing about bandwidth.** A polynomial evaluated at a point is not a random oracle: a
Kedlaya–Umans evaluation structure for each block's polynomial answers any `g_j` from a few of its words, and the
structures for all of `E` fit in the RTX PRO 6000's spare device memory (about 95 GiB beside a 1.08 GB store). A device
holding them answers every tag without reading `E_j`'s words, so the compression argument has nothing to count, and
`bound` refuses the tag. It stays as a fast check that a device holds `E` (or something equivalent): its pass is the
fastest of the three, and a device missing blocks fails it exactly as it fails the certified tags. Its word map binds
only the words mod p (`w`, `w + p` and `w + 2p` agree), 15,934 of a block's 16,448 bits (`info_bits`).

## 5. The bound's parameters, and the model it rests on

| parameter | value | source |
|---|---|---|
| `ε_s`, `s` | 1/64, 1,849 | `audit.EPSILON`, `audit.SAMPLES` (`samples_for(N, -42)`) |
| `q` | at most `2^40` oracle queries in a window | `audit.QUERIES_LOG2`; the RTX PRO 6000 makes about `2^26` BLAKE3 compressions in 1.7 ms |
| `λ` | `40·k + 41` bits | `audit.SLACK_LOG2`: `C(q, k)/2^λ ≤ q^k/2^λ = 2^-41` |
| `S` | 208,142,336 bytes (19.3% of `E`) | `S_chip`: 128 MiB of L2 and 188 SMs × (256 KiB of registers + 128 KiB of L1 and shared memory), from the device's properties (`device.json`) |
| `c` | `cap_side · (T + RTT)`, `cap_side` = 63.0 GB/s | the GPU's PCIe Gen 5 x16 link at its raw rate, 32 GT/s × 16 lanes at 128b/130b; measured 56.6 GB/s host to device, 51.4 GB/s zero-copy (§6) |
| `G` | 0 | `Params` refuses `T + RTT ≥ 2 · ROOT_CALL_NS = 1,708,058 ns`, the fastest cooperating P2 root; one root, not two, so a device holding a block's inner root rebuilds nothing |

The claims registry (`verity.core.claims`, on #1806's branch) names what a deployment rests on:

- **`random-oracle`** (assumption): the tag is a random oracle fresh at `σ`. For `ro-blake3/v1` the instance is keyed
  BLAKE3 under a key derived from `σ` by SHA-512; for `ro-sha512/v1`, SHA-512 after a prefix that holds `σ`. The
  seed, the samples and both trees use SHA-512 and its collision resistance as before.
- **`bandwidth-instantiation`** (assumption, conjectured): the blocks of `E` are no easier to hold information about
  than uniform blocks, except by rebuilding them from setup. This is where P2's incompressibility enters.
- **`bandwidth-s-chip`**, **`bandwidth-side-cap`**, **`bandwidth-regen-time`** (assumptions, measured): `S`, `c` and the
  rebuild time on the stated device, each falsified by a measurement (a larger on-chip working set, a faster path in,
  a faster root).
- **`bandwidth-device`** (model): the device is a chip with device memory and side sources, all laid out before `σ`; it
  reads device memory only through counted ingress, computes every tag on chip, and commits before the samples.
  A deployment is an instance when two conditions hold, which the registry's text should state:
  1. *Side sources carry data only.* A side source (NVMe, NIC, a peer GPU) delivers bits, which `c` counts, but no tag
     and nothing else computed from `E` elsewhere. A helper with a copy of `E` that sends 32-byte tags delivers a
     block's share of the audit in 32 of its 2,056 bytes, which `c` undercounts 64 times.
  2. *Every device that could hold a copy of `E` is audited in the same window, or the device is isolated*: its side
     links carry nothing from such a device during the window.

  The residual of condition 2 is a helper's slack. A device audited in the same window that finishes its own pass at
  `t < T + RTT` has the rest of the window free. The side link does not limit what it could send (63 GB/s × 1.2 ms
  is 2.4 million tags, more than `N`); its reads do, at most `(T + RTT − t)` × its read rate / 2,056 bytes blocks:
  about 144,000 blocks (27% of `N`) at `T = 1.2 ms`, a 1.0 ms pass and 1,476 GB/s. Condition 1 excludes it; a
  deployment that cannot enforce condition 1 isolates the device, or charges that many blocks against `k`.

Host RAM is not part of the allowed hardware (Daniel, 3 Oct 2026); the RTX PRO 6000 Server has no NVLink, so its side
paths are the PCIe link, shared by host, NVMe, NIC and peer GPUs.

## 6. Parameters and measurements

`T` must cover the honest prover's elapsed time on the verifier's clock with margin, and leave `T + RTT` under the
rebuild floor. The proposal: **`ro-blake3/v1`, `s = 1,849`, `T = 1.2 ms`**, which leaves up to 0.5 ms for `RTT`.

The GPU tagger (`benchmarks/bandwidth/run.sh`) on one RTX PRO 6000 Blackwell Server Edition of vy-nebius-2, under a
timed lease, SM clock 2,092 MHz throughout. Every run's gates passed before anything was timed: SHA-512's and keyed
BLAKE3's known answers on host and device; every tag's root over the full store (`N = 2^19`) and over the pinned
vectors' 37 blocks equal to the reference's, with the output buffers poisoned; and a pass without the tag kernel that
does not match. A failed gate times nothing.

This section reports run `r20261012-001813-b2be`
(art:bde66bd3aff8f3598ce7bddacf4d8a6630c0200ea32ee8d1d16aba5349d37603). Runs `r20261011-235743-0c42`
(art:8735b1084479fca8cf64bd225c58fb85a2afe6599a5b6c864759a421699a2e8e) and `r20261012-000847-217b` ran the same kernels
and reproduced every clean pass time to within 0.01 ms.

Times are medians of 100 passes (20 for *idle*; p99 in brackets), in ms, under four L2 conditions set before the pass:
*dirty*, after a 512 MiB write; *warm*, whatever the last pass left; *clean*, after a 512 MiB read of another buffer;
*idle*, clean after 250 ms with the GPU idle.

| | dirty | warm | clean | idle |
|---|---|---|---|---|
| raw device read of `E` | 0.730, 1,476 GB/s | | | |
| `ro-blake3/v1` pass: tags, then leaves and tree | 1.577 (1.590) | 1.130 (1.367) | **0.976 (0.981)** = 0.833 + 0.143 | 0.990 (0.994) |
| `ro-sha512/v1` pass | 1.727 (1.734) | 1.722 (1.727) | 1.720 (1.727) = 1.409 + 0.311 | 1.733 (1.738) |
| `gf31-poly/v1` pass | 0.911 (0.922) | 0.881 (0.887) | 0.885 (0.893) = 0.742 + 0.143 | 0.893 (0.901) |

Whole windows through `audit.Verifier` at `T = 1.2 ms` and `s = 1,849`, 50 per tag, each after the honest prover's
preparation (§3), with the verifier's clock beside the device's pass time:

| | verifier's clock, p50 (max) | device's pass, p50 (max) | accepted |
|---|---|---|---|
| `ro-blake3/v1` | 1.012 ms (1.025) | 0.981 ms (0.992) | 50 of 50 |
| `gf31-poly/v1` | 0.911 ms (0.942) | 0.886 ms (0.909) | 50 of 50 |
| `ro-sha512/v1` | 1.750 ms (1.763) | 1.726 ms (1.729) | none: all late |

The slowest honest `ro-blake3/v1` window took 1.025 ms, 175 µs (17%) inside `T = 1.2 ms`, and `T + RTT` leaves
0.508 ms for the round trip under the rebuild floor. One window per tag at a 100 ms deadline with the floor lifted,
which certifies nothing, was accepted for all three: `ro-sha512/v1`'s openings pass even though its timed windows are
late. PCIe into the GPU measured 56.6 GB/s (pinned copy) and 51.4 GB/s (a kernel reading pinned host memory).

`ro-blake3/v1`'s pass depends on L2's state when it starts. Its loads fetch 256 bytes into L2 per miss
(`ld.global.nc.L1::no_allocate.L2::256B`), which helps when L2's lines are clean and costs 0.6 ms when they are dirty
and must be written back first. The honest prover controls that state before `σ`, so it reads 512 MiB of another
buffer before each window, and `T` is set from the clean condition. The pass is neither memory- nor compute-bound:
its tag kernel takes 0.83 ms against a raw read of 0.73 ms and a compute floor of about 0.51 ms (35 compressions per
block at BLAKE3's 5.88 bytes per SM per clock on sm_120, notes `kb/sm120/tried-techniques.md`); the rest of the pass is
the leaves and the tree, 0.14 ms.

The windows run through `audit.Verifier` with the honest prover's preparation (§3), and record the device's pass time
beside the verifier's. Without the warm pass, the last window's openings and verdict (3.3 s of Python BLAKE3 per window,
on the same host) left about 0.18 ms of host time inside each later `ro-blake3/v1` window while the device's pass stayed
at 1.01 ms (run `r20261012-000847-217b`, art:3a657af4fa9d2c009caff8382b05bcb97e5bdb3b0804a0705c77830ad87b1324). A
device that skips 1% or 5% of the blocks was rejected in every run, for every tag; at `s = 1,849` it passes with
probability `0.99^1849 ≈ 2^-26.8` or `0.95^1849 ≈ 2^-136.8`.

`bound` at `s = 1,849`, `S = 208,142,336` bytes and `cap_side = 63.0 GB/s` (error `2^-40.42` throughout):

| `T` | `RTT` | `I` | share of `E` | `I / (T + RTT)` |
|---|---|---|---|---|
| 1.0 ms | 0 | 787,355,169 bytes | 73.0% | 787 GB/s |
| **1.2 ms** | **0** | **774,752,092 bytes** | **71.9%** | **646 GB/s** |
| 1.2 ms | 0.3 ms | 755,847,477 bytes | 70.1% | 504 GB/s |
| 1.4 ms | 0 | 762,149,015 bytes | 70.7% | 544 GB/s |
| 1.708 ms (the largest `Params` allows) | 0 | 742,736,685 bytes | 68.9% | 435 GB/s |

At `T = 1.2 ms` the certified rate is 44% of the raw read rate (1,476 GB/s). Of `E`'s 1,078 MB, the chip's storage
takes 208 MB (19.3%), the side link 76 MB (7.0%), the sample 17 MB (1.6%) and `λ` 2.6 MB (0.24%); the rest of the gap
from the raw rate is the time the honest pass and its margin take beyond a raw read.

## 7. Findings and open questions for the red team

1. **`ro-sha512/v1` does not fit on the RTX PRO 6000.** Its honest pass takes 1.73 ms, past the 1.708 ms rebuild floor,
   so no `T` that `Params` allows admits it; its windows are late. It is SHA-512-bound (17 compressions per block and
   one for its leaf), and it stays as the proof's first instance and the reference's.
2. **`ro-blake3/v1` is the deployed tag.** Keyed BLAKE3 is the fastest random-oracle candidate on sm_120, whose
   integer pipe runs every integer operation at about 64 thread-operations per SM per clock. TurboSHAKE128, the second
   candidate, was not built: on the same device it hashes 4.95 bytes per SM per clock against BLAKE3's 5.88 (notes
   `kb/sm120/tried-techniques.md`, run `r20260930-061717-3e84`), a compute floor of 0.56 ms on this store against
   0.51 ms.
3. **The helper-in-slack residual** of `bandwidth-device` (§5) is not in the Lean statement; the model claim excludes
   it. A red team should say whether a deployment should charge it against `k` instead.
4. **The store's leaves are PoUS's, not P2 spec §4's.** `vk`'s leaf is the 99-byte little-endian leaf of
   `pous.verifier`; the P2 spec's is 107 bytes, big-endian (`benchmarks/pous/p2_v1/wire.py`). The audit uses what PoUS
   uses on `main` and inherits its resolution.
5. **`S_chip` counts L2, the register files and L1 and shared memory.** Smaller on-chip stores (instruction and
   constant caches, copy-engine buffers) are left out. A red team should confirm that none of them can hold a useful
   share of `E` across a window's start.
6. **Openings disclose sampled blocks** of `C*` (an encoding of `W`) to the verifier; the memory challenger already
   holds `vk` and publishes only verdicts, so nothing leaves it.
7. **The audit imports PoUS** (`pous.codec`, `pous.schemes.p2`, `pous.verifier`) for its store and setup. The import
   boundary for protocols is Daniel's to rule on; until then `tests/test_boundaries.py` carries the edge.

## 8. Code

- `tags.py`: `get(name)`, each construction's `keyed(σ)(j, block)`, `certified`, `guess`, `info_bits`; the tag tree
  (`leaves`).
- `audit.py`: a device's salt (`draw_salt`), the prover's reference (`Store`, `tag_pass(store, tag, σ) -> TagCommitment` with `root` and
  `open(j) -> Opening`), the verifier (`Verifier(vk, params, source).window(n)`: `start() -> σ`, `commit(root) ->
  samples`, `verdict(openings) -> Verdict`), the proof's point (`EPSILON`, `SAMPLES`, `QUERIES_LOG2`, `SLACK_LOG2`,
  `certified_blocks`, `samples_for`) and `bound`.
- `tests/bandwidth_vectors.py` writes `tests/vectors.json`: tags at 2,056 bytes for every construction, two devices'
  salts, window seeds and samples, and a whole window over 37 blocks.
- `benchmarks/bandwidth`: the GPU tagger (`tagger.cu`, one kernel per tag, the leaves and the tree), its harness API
  (`gpu.py`: `Tagger.tag_pass(tag, store, blocks, σ, workspace)` returns the root; `open(j, store_tree)` gives the
  `Opening` this verifier checks; `windows` runs the honest prover's preparation and the windows), its Python twins
  (`benchmarks/bandwidth/tests/test_tagger_twin.py`), and the runs of §6.
