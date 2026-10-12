# The bandwidth audit, v0

Status: v0 with the certified tag `ro-blake3/v1` at `s = 1,849` and `T = 1.2 ms`. The red team granted head
`f18c1854a` with conditions (art:847a28cdb35482186f67e1aec1bcab52c9bf1ad73cac87c2a7141b366be12fa6), fixing `T` and
`s`, then objected at `5c62196c1` (art:dbbce953a1916411304d1f4558010b6c18cbcb4b69c2c514e32ca83a2bc6b9c5): windows over
one store could share a key across verifiers, tags and restarts, and the verifier accepted a store not salted for its
device. It granted `5761861fd`, with those fixes, with two conditions
(art:c67e2351ef1718c239e20fca8ddc6d92fc3ded80941fb0f1e50812cd6892cb18): one challenger state per store (§3) and a
device id the challenger reads itself (§2). Both are fixed and await its check; §6 quotes the rate granted at
`5761861fd`. The Python reference and verifier are `audit.py` and `tags.py`, pinned by `tests/vectors.json`; the GPU
tagger is `benchmarks/bandwidth`, checked bit for bit against them and timed on the RTX PRO 6000 (§6). The guarantee
is `Bandwidth.Guarantees.AuditRO`, in the `Accounting` area of `Security/`'s guarantees, proved as
`Bandwidth.SecurityProofs.AuditRO` with no named assumption by lane `ra-bandwidth-lean` (PR #1806); its completeness
is `HonestPasses`. Until #1806 is on `main`, the statement is that PR's.

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

At the deployed point, `ε_s = 1/64`, `s = 1,849`, `h = 256` (`ro-blake3/v1`), `q ≤ 2^64` and `λ = 64·k + 41`:

| | value |
|---|---|
| `N`, `b` | `2^19` blocks of 16,448 bits (2,056 bytes); `E` is 1,077,936,128 bytes |
| `k` | 516,096 blocks, `E` less 1/64 (16.8 MB) |
| `λ` | 33,030,185 bits = 4,128,774 bytes, 0.38% of `E` |
| error | `2^-42.01` (samples) + `2^-245.1` (guesses) + `2^-41` (compression) ≤ `2^-40.42` |

`G = 0`, because `Params` refuses `T + RTT` at or past one rebuild (§5). `S` and `c` are the device's (§5).
`audit.bound` computes `I` and the error, and §6 evaluates it at the measured `S` and `c` and quotes the rate the red
team granted at `5761861fd`.
`test_bound_is_the_guarantee_at_the_proofs_point` checks `bound` against the formula.

`gf31-poly/v1` is an uncertified fast check: nothing bounds a device's ingress through it (§4), and `bound` refuses it.

## 2. The store and its setup

`E` is PoUS's P2 store `C*` (`verity.core.protocols.pous.schemes.p2`): `N = 2^19` blocks of 2,056 bytes (1.08 GB),
encoded from the useful data `W` under PoUS's setup (`pous.setup`: the commitment to `W`, then the verifier's salt,
then `vk`). The audit reuses that setup and `vk`, with one change of input: a block's identity is its leaf under
`vk.root` (`pous.verifier.leaf_hash`, RFC 6962 SHA-512 tree), and the verifier holds only `vk`.

**One store per device.** The verifier draws each device's salt with the device's id bound in,

~~~text
salt = derive(source, "verity/bandwidth/salt/v2", {w: commitment, device: id}).stream("salt", 24)
~~~

(`audit.draw_salt`), and runs PoUS's setup with that salt in place of `pous.setup.draw_salt`'s. PoUS's setup takes the
salt as an input and needs only that it is drawn after the commitment to `W`, from the verifier's secret or a later
beacon round, which this one is, so PoUS is unchanged. Devices that hold one `W` under one secret then hold different
stores, so their `vk`, seeds and tags differ, and one device's commitment fails another's window
(`test_one_devices_commitment_fails_anothers_window`). Without it, two such devices hold byte-identical stores, and one
device's root passes the other's windows. So the verifier refuses any other store: `audit.Verifier` takes the device's
id, the commitment to `W` and the source the salt was drawn from, and refuses `vk` unless `vk.salt` is `draw_salt`'s
for them, before any window. A store set up under PoUS's own salt, another device's store, or one salted under another
commitment is never audited as this device's (`test_a_store_not_salted_for_its_device_is_refused`).

**The device's id** is the 16 bytes of its GPU's UUID, nothing else: `draw_salt` refuses any other value, and
`audit.device_id` reads it only from the one spelling nvidia-smi prints, `GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx` in
lowercase hex, refusing uppercase, missing dashes, braces, a MIG instance's id and anything else. So one GPU has one id
and one salt. The challenger reads the id itself, on the device's node, from the NVIDIA driver: `audit.read_device(gpu)`
asks nvidia-smi (NVML) for the GPU's UUID and MIG mode, reads the `GPU-…` string through `device_id`, and refuses a MIG
slice, whether the selector names one or the GPU is in MIG mode (its CUDA devices are then slices, each with a UUID of
its own). The prover's word is only checked against it: `benchmarks/bandwidth` refuses a tagger whose CUDA device
(`cudaDeviceProp.uuid`, `Tagger.device()`) has another UUID (`gpu.challenger_device`;
`test_the_challenger_reads_a_devices_id_from_the_driver`, `test_the_challenger_reads_the_device_it_audits`). Nothing on
the chip signs it: the verifier trusts the id because it trusts the host the challenger runs on (`trusted-nebius-host`,
§5) to report the UUID of the GPU it audits. A host that reported another GPU's id would get that GPU's store, and two
GPUs would answer from one store. The GPU's confidential-computing attestation could sign the UUID with a key in the
chip, which would remove the host from this step; the audit does not use it yet. The SKU the device reports is
recorded beside the id.

The salt is the place for the device, not the window key. With the device only in the window key, two devices holding
byte-identical stores get different seeds, but one device that reads each block once can tag it under both seeds and
send the other device's tags (16.8 MB a window, under `c`) over its side link. Distinct stores are what make the
ingress of devices audited together add up (§5), and the window key, which binds `vk.root` and `vk.salt`, carries the
device without a change.

Each block of `C*` is two slow roots of its keyed payload, `c = ρ(σ_K(ρ(m + t)))`. Under P2's assumptions a block is
incompressible except through `W`, and rebuilding it from `W` takes two roots in sequence.

## 3. A window

1. **Seed.** The verifier draws a fresh 32-byte nonce from its own randomness and records it in the store's state
   (`Window.open`, below). Then it derives the window's key from its own secret,
   `key = derive(source, "verity/bandwidth/window/v2", {root: vk.root, salt: vk.salt, tag: name, nonce: nonce})`
   (`verity.ml.randomness`), sends `σ = key.stream("seed", 32)` and starts its clock (`Window.start`). The nonce is
   never a counter and never drawn twice for a store, so no two windows over a store share a key, whichever verifier,
   tag or restart opens them. A key fixed by a window number repeats in every verifier that counts from the same
   place: a window of the fast check reveals, after its commitment, the samples of the certified tag's window of the
   same number, and a restarted verifier's window accepts a root replayed from before the restart. A second verifier
   of the store, another tag's window and a restart each draw new samples, and a replayed root is refused
   (`test_a_second_verifier_of_the_store_repeats_no_window`, `test_another_tags_window_reveals_no_samples`,
   `test_a_restarted_challenger_repeats_no_window_and_refuses_a_replayed_root`).
2. **Tags.** The prover computes `y_j = Tag(σ, j, E_j)` for every `j < N` (§4), packs them into leaves, and returns the
   RFC 6962 SHA-512 root of the leaves. The verifier stops its clock on the root (`Window.commit`); the root is on time
   iff the elapsed time is at most `T + RTT` (`Params.limit_ns`).
3. **Samples.** Only for an on-time root, the verifier draws `s` indices, `key.uniform(N, ("sample", t))` for `t < s`,
   uniform with replacement, and sends them. A late or malformed root closes the window rejected and draws nothing.
4. **Openings.** For each sample `i`, in order, the prover sends `E_i` with its audit path to `vk.root`, and the tags
   of `i`'s leaf with the leaf's path to the committed root (`TagCommitment.open`). The verifier checks both paths and
   recomputes `y_i` from `σ`, `i` and `E_i` (`Window.verdict`). Openings are not timed: the root binds every tag.

The verifier accepts iff the root is on time and every opening checks. It refuses a second start or a second root,
and never redraws. One window over a store is open at a time: from its record until its root arrives, or until
`T + RTT` after `σ` without one, the verifier refuses to open another window over the same store, which stays closed
and may open once the first has closed. Two windows open at once could both be answered from
one read of each block, so their ingress would not add; with one open at a time, each window's ingress is read within
its own interval (`test_one_window_over_a_store_is_open_at_a_time`).

**The challenger's state.** Both rules hold across every verifier of a store, whatever its tag, and across restarts:
every verifier of a store takes the store's one challenger state (`audit.ChallengerState`). It is a file in the
challenger's state directory on its host, passed explicitly and never the LMS key directory, and the store names it:
`<vk.root>-<vk.salt>.state` in hex (`audit.state_name`). The directory holds nothing else: `ChallengerState.create` and
opening refuse a directory holding any file that is not a state named by the store its first line names, so a copy or
hard link of a state under another name, a symbolic link, a subdirectory or a stray file is refused for every store,
and `create` refuses a store that has a state. A store therefore has one state in the directory, and the directory's
listing shows which stores have one (`test_a_store_has_one_challenger_state`). The file's first line names the format
and the store (`vk.root`, `vk.salt`); then each window appends `open NONCE TAG`, fsynced before its `σ` leaves, and
`close NONCE` when it closes (its root, or `T + RTT` without one). The file is append-only: nothing rewrites or
truncates it, and a store's state is never made again, so it is never reset. A state that is missing, a link, unreadable
(a record cut short, a nonce opened twice, a close that is not the open window's), another store's, or changed since this
challenger last wrote it opens no window. The challenger holds the file's lock while it runs, so one challenger holds a
store's state at a time. A nonce in the file is never drawn again, and one window over the store is open at a time,
whichever verifier opens it. A window the file shows open when the state is reopened belonged to a challenger that
stopped; reopening records it closed, and no verifier can accept it. A file cannot detect being restored from a
snapshot, nor a directory a copy of a state kept in another directory or on another host. Both rest on
`trusted-nebius-host`, which network-accounting widens under the same id to say that the device runs its stated code
on its Nebius host, "…, and its host never restores an earlier copy of the state the device keeps there", pending
network-accounting's widening (Daniel, decision 2). The challenger's stated code keeps one state directory and never
copies a state, and its host never brings an earlier copy back. PoUS's memory challenger cites the same assumption for
its session-id log. The audit needs it for the two properties below.

**What the no-rollback presumption is for.** A single window's bound does not need it. AuditRO's bound for one
accepted window (§5) needs only that the prover could not know `σ` or the samples before `σ` left, and both come from
the window's key, which binds the verifier's secret and a 32-byte nonce from the challenger's own randomness. Two
windows over a store share a key only if their nonces collide, with probability below `n²/2^257` over `n` windows, so
keys are unique given fresh randomness alone; the state's refusal of a nonce it holds only catches a source that
repeats. A state rolled back, restored or forked therefore replays no window unless the randomness repeats too. The
presumption is still needed for two properties, both about windows that overlap in time:

- **One open window per store.** The state enforces it through its record of the open window and its lock. A second
  copy of the state, or one restored to before an open window's record, would let a second challenger open a window
  over the store while that one is open, and two such windows can be answered from one read of each block.
- **Additivity across windows.** The ingress of a store's windows over a period, or of a co-audited set's (§5), adds up
  only because no two windows over a store are open at once. Without the presumption each accepted window's bound
  still holds, but their sum does not bound the total.

The verdict publishes one bit (`Verdict.public`); which check failed,
which sample and the elapsed time stay with the verifier. Openings reveal blocks of `C*` to the verifier, so the
verifier that runs this is PoUS's memory challenger, which already holds `vk` and publishes only verdicts.

A deployment may open a window when the prover says it is ready: the bound holds for every device state laid out
before `σ`, so the prover's preparation costs the verifier nothing. The honest prover's preparation on the GPU is one
untimed pass under a throwaway seed, then a read of 512 MiB of another buffer, so the window starts with its launch
path warm and L2 holding clean lines and no part of `E` (§6). The honest client always prepares before it asks for
`σ` (`gpu.Prover`, checked by `benchmarks/bandwidth/tests/test_prover.py`): a window run without it took 1.195 ms.
It opens the window first, so the state's fsync falls before the preparation, not between it and `σ`: with the fsync
after the scrub, the GPU and the host idle for it, and honest windows took 1.095 ms at the median and one of 50 was
late (§6).

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
| `q` | at most `2^64` oracle queries in a window | `audit.QUERIES_LOG2`; the RTX PRO 6000 makes about `2^26` BLAKE3 compressions in 1.7 ms, so `2^64` covers `2^38` such devices helping, and the bound needs no assumption on `q` |
| `λ` | `64·k + 41` bits | `audit.SLACK_LOG2`: `C(q, k)/2^λ ≤ q^k/2^λ = 2^-41`; raising `q` from `2^40` cost 1.55 MB (0.14 points) |
| `S` | 239,992,832 bytes (22.3% of `E`) | `S_chip` (`summarize.s_chip_bytes`, from the device's properties in `device.json`): 128 MiB of L2; for each of 188 SMs, 256 KiB of registers, 128 KiB of L1 and shared memory, and 160 KiB of other stores; then 1 MiB at the chip level (see below) |
| `c` | `cap_side · (T + RTT)`, `cap_side` = 63.0 GB/s | the GPU's PCIe Gen 5 x16 link at its raw rate, 32 GT/s × 16 lanes at 128b/130b; measured 56.3 GB/s host to device, 51.3 GB/s zero-copy (§6) |
| `G` | 0 | `Params` refuses `T + RTT ≥ 2 · ROOT_CALL_NS = 1,708,058 ns`, the fastest cooperating P2 root; one root, not two, so a device holding a block's inner root rebuilds nothing |

`S` counts every on-chip store a resident kernel can hold data in across `σ`, at its raw capacity. Beside L2, the
register files and L1 and shared memory, each SM has instruction caches (about 4 KB of L0 per SMSP and 64 KB of L1),
constant caches, uniform registers (256 bytes a warp) and a TLB; a kernel can load constants before `σ`, or hold data
as instruction immediates that it runs before `σ` and again, hitting in the cache, after. The chip has copy-engine,
PCIe and memory-controller buffers and video engines. The instruction caches' sizes are measured on an RTX PRO 5000
(GB202, sm_120, the RTX PRO 6000's die) in [Dissecting SM_120](https://zartbot.github.io/micro_arch/nvidia/sm_120/),
chapter 7; the allowance of 160 KiB a SM and 1 MiB for the chip is the red team's
(art:847a28cdb35482186f67e1aec1bcab52c9bf1ad73cac87c2a7141b366be12fa6), and it costs 31.9 MB (3.0% of `E`). sm_120
has no Tensor Memory (the same reference, chapter 14), so there is no term for it. Measuring the caches on the
RTX PRO 6000 (code- and constant-footprint latency sweeps) could tighten the allowance.

The claims registry (`verity.core.claims`, on #1806's branch) names what a deployment rests on:

- **`random-oracle`** (assumption): the tag is a random oracle fresh at `σ`. For `ro-blake3/v1` the instance is keyed
  BLAKE3 under a key derived from `σ` by SHA-512; for `ro-sha512/v1`, SHA-512 after a prefix that holds `σ`. The
  seed, the samples and both trees use SHA-512 and its collision resistance as before. The proof's oracle is one call,
  `H(σ, j, E_j)`, while BLAKE3 is a tree of compressions; indifferentiability does not carry a bound like this one over
  to the tree (Ristenpart, Shacham and Shrimpton), so that step rests on `bandwidth-instantiation`.
- **`bandwidth-instantiation`** (assumption, conjectured): the blocks of `E` are no easier to hold information about
  than uniform blocks, except by rebuilding them from setup. This is where P2's incompressibility enters. It also
  covers a property of `ro-blake3/v1`'s tree: its message's chunks 1 and 2 (`E_j[1016:2040]` and the 16-byte tail)
  hold no byte of `j`, so blocks with equal aligned chunks share that chunk's chaining value within a window, and a
  device tags them with less work. P2's blocks don't repeat (near-uniform values mod `p` under a fresh key per block;
  the 37-block vector store has no repeated chunk or tail), so the saving on `E` is nil. A store with repeated
  aligned chunks is not an instance.
- **`bandwidth-s-chip`**, **`bandwidth-side-cap`**, **`bandwidth-regen-time`** (assumptions, measured): `S`, `c` and the
  rebuild time on the stated device, each falsified by a measurement (a larger on-chip working set, a faster path in,
  a faster root). The device for all three is a whole GPU, not a MIG slice: slices share the chip's L2 and crossbar,
  so a co-resident slice's path in is on chip, not over PCIe. The RTX PRO 6000 Server's only link is its PCIe Gen 5
  x16 (no NVLink, no chip-to-chip link). A deployment records the SKU the device attests (it supports confidential
  computing) beside its id (§2).
- **`trusted-nebius-host`** (assumption, core's): the challenger runs its stated code on its Nebius host, which offers
  no attesting hardware, "…, and its host never restores an earlier copy of the state the device keeps there", the
  wording network-accounting adds under the same id, pending network-accounting's widening (Daniel, decision 2). The
  audit cites it for two properties: the device id the challenger reads from the driver is the UUID of the GPU it
  audits (§2), and the challenger keeps one state directory, never copies a state and never gets an earlier copy back
  (§3), which one open window per store and additivity across windows need and a single window's bound does not.
- **`bandwidth-device`** (model): the device is a chip with device memory and side sources, all laid out before `σ`; it
  reads device memory only through counted ingress, computes every tag on chip, and commits before the samples.

  A tag computed elsewhere from a copy of a store and sent in is outside the model: `c` counts bits, and a helper's
  32-byte tags deliver a block's share of the audit in 1/64 of its bytes. What a helper can deliver is limited by its
  reads, not its link (63 GB/s × 1.2 ms is 2.4 million tags, more than `N`). A helper that interleaves reads of a copy
  of another device's store with its own pass, at its measured read rate (1,476 GB/s), covers
  `(T + RTT)` × 1,476 GB/s less its own `I`: about 500,000 blocks, 96% of `N`, at `T = 1.2 ms` (the red team counted
  92% at the earlier `I`). Charging that against `k` leaves nothing, so a per-device figure means something only for
  a device no helper reaches. **`bandwidth-device` is therefore stated for a co-audited set** (Daniel, decision 4,
  11 Oct 2026, 8:54 PM PDT):

  - **Which devices count.** Whole GPUs whose only link is PCIe: no MIG slice (§2) and no NVLink. Every device
    that could hold a copy of a set's stores is in the set or isolated. An NVLink domain is certified as one set
    later.
  - **What a set certifies.** The devices of a set run their windows at the same time, with one start and one
    deadline (an offset is a helper's time), over pairwise-distinct stores (§2). The set's chips are then one chip,
    and the same compression argument gives `Σ_i I_i ≥ Σ_i k·b − Σ_i S_i − c_ext − Σ_i λ` bits, except with
    probability the sum of the devices' errors, where `c_ext` counts only links from outside the set. Traffic between
    the set's devices stays inside it, so a helper's slack is counted and the residual disappears. For `m`
    RTX PRO 6000s, each on its own PCIe link, `c_ext ≤ m · cap_side · (T + RTT)`, so the set's total is at least `m`
    times the per-device figure; that bounds the set's total, not any one device. This is AuditRO for `m` stores
    sampled separately, which lane `ra-bandwidth-lean` is proving; the set's figure is certified once it lands.
  - **What is certified now.** Per device, for an isolated device: `I ≥ k·b − S − c − λ` for one device whose side
    links carry nothing computed from any copy of its store during the window, the statement AuditRO makes. A peer
    that relays tags in over the side link is charged inside the side budget by `AuditRelay` (#1819), at `m·b` bits
    beyond `c` for `m` relayed blocks with `m·(h − log₂ N) ≥ c + μ`; at the full PCIe cap that is more blocks than
    `N`, so a device a peer can reach gets its figure only in a co-audited set. A device that is neither isolated nor
    in a co-audited set gets no figure.

Host RAM is not part of the allowed hardware (Daniel, 3 Oct 2026); the RTX PRO 6000 Server has no NVLink, so its side
paths are the PCIe link, shared by host, NVMe, NIC and peer GPUs.

## 6. Parameters and measurements

`T` must cover the honest prover's elapsed time on the verifier's clock with margin, and leave `T + RTT` under the
rebuild floor. The parameters: **`ro-blake3/v1`, `s = 1,849`, `T = 1.2 ms`**, which leaves up to 0.5 ms for `RTT`,
fixed by the red team at `f18c1854a`.

**The certified rate, granted at `5761861fd`** (art:c67e2351ef1718c239e20fca8ddc6d92fc3ded80941fb0f1e50812cd6892cb18).
An accepted `ro-blake3/v1` window at `T = 1.2 ms` and `s = 1,849` certifies that the device's memory delivered at least
741,353,308 bytes of `E` (68.8%) within `T + RTT`: 618 GB/s with `RTT = 0`, and 722,448,693 bytes (67.0%), 482 GB/s,
with a 0.3 ms `RTT` allowance. The error is `2^-40.42`. The last table of this section is `audit.bound` at these
parameters. Under Daniel's reading of `bandwidth-device` (§5), the rate is certified now for an isolated device, and
for a co-audited set it is each device's share of the set's total once AuditRO for `m` stores lands.

The GPU tagger (`benchmarks/bandwidth/run.sh`) on one RTX PRO 6000 Blackwell Server Edition of vy-nebius-2, under a
timed lease, SM clock 2,085 to 2,092 MHz throughout. Every run's gates passed before anything was timed: SHA-512's and
keyed BLAKE3's known answers on host and device; every tag's root over the full store (`N = 2^19`) and over the pinned
vectors' 37 blocks equal to the reference's, with the output buffers poisoned; and a pass without the tag kernel that
does not match. A failed gate times nothing.

This section reports run `r20261012-024259-a1d6`
(art:34582b732f2c2e97ecd0621bad0bd17688684365b24c51538f8e9146153b3a14) at `7eb0467af`, the objection's fixes in place.
Runs `r20261012-011339-b99d` (art:ebd5d428c8dcf6f0807b54454f3fb3107b53fcaa4635ebf3c300c83a3d211f5d, at `84fc4199d`),
`r20261012-001813-b2be` (art:bde66bd3aff8f3598ce7bddacf4d8a6630c0200ea32ee8d1d16aba5349d37603, at `f18c1854a`),
`r20261011-235743-0c42` (art:8735b1084479fca8cf64bd225c58fb85a2afe6599a5b6c864759a421699a2e8e) and
`r20261012-000847-217b` ran the same kernels and reproduced every clean pass time to within 0.01 ms.

Times are medians of 100 passes (20 for *idle*; p99 in brackets), in ms, under four L2 conditions set before the pass:
*dirty*, after a 512 MiB write; *warm*, whatever the last pass left; *clean*, after a 512 MiB read of another buffer;
*idle*, clean after 250 ms with the GPU idle.

| | dirty | warm | clean | idle |
|---|---|---|---|---|
| raw device read of `E` | 0.730, 1,476 GB/s | | | |
| `ro-blake3/v1` pass: tags, then leaves and tree | 1.574 (1.595) | 1.129 (1.346) | **0.976 (0.983)** = 0.832 + 0.143 | 0.994 (1.042) |
| `ro-sha512/v1` pass | 1.729 (1.742) | 1.725 (1.734) | 1.726 (1.737) = 1.412 + 0.313 | 1.744 (1.748) |
| `gf31-poly/v1` pass | 0.909 (0.920) | 0.878 (0.885) | 0.883 (0.893) = 0.742 + 0.141 | 0.899 (0.913) |

Whole windows through `audit.Verifier` at `T = 1.2 ms` and `s = 1,849`, 50 per tag, each after the honest prover's
preparation (§3), with the verifier's clock beside the device's pass time:

| | verifier's clock, p50 (max) | device's pass, p50 (max) | accepted |
|---|---|---|---|
| `ro-blake3/v1` | 1.024 ms (1.042) | 0.985 ms (0.995) | 50 of 50 |
| `gf31-poly/v1` | 0.920 ms (1.000) | 0.889 ms (0.923) | 50 of 50 |
| `ro-sha512/v1` | 1.773 ms (1.792) | 1.735 ms (1.750) | none: all late |

The slowest honest `ro-blake3/v1` window took 1.042 ms, 158 µs (13%) inside `T = 1.2 ms`, and `T + RTT` leaves
0.508 ms for the round trip under the rebuild floor. One window per tag at a 100 ms deadline with the floor lifted,
which certifies nothing, was accepted for all three: `ro-sha512/v1`'s openings pass even though its timed windows are
late. Prepared like the others, that window's `ro-blake3/v1` pass took 1.025 ms on the verifier's clock (1.195 ms
unprepared at `f18c1854a`). PCIe into the GPU measured 56.3 GB/s (pinned copy) and 51.3 GB/s (a kernel reading
pinned host memory).

Every window of the run went through the store's one challenger state: 53 per tag (50 timed, two with skipped blocks,
one at the 100 ms deadline), 159 nonces in all, each opened and closed once and none repeated. Run
`r20261012-022850-f3ea` (art:a2154aeefb5c7c1e6c0f99218828d44ddd764514489ca00da3688ce8d74395f8) at `4ae302654`, before
`Window.open`, fsynced the state between the prover's scrub and `σ` (§3). Its gates passed and every skipping device
was rejected, but honest `ro-blake3/v1` windows took 1.095 ms at the median on the verifier's clock (1.221 ms at most)
with the device's pass at 1.007 ms, and one of 50 was late; three of 50 `gf31-poly/v1` windows were late. It is the
finding behind §3's ordering, not a measurement of the audit. Run `r20261012-034853-e233`
(art:6efb3ebf776e10ced6fe6ad128185373189cfc0cedeae3e5fbb67a4f63856737) at `a677eaf3c` checked the grant's two
conditions on the node and timed nothing: nvidia-smi gave the leased GPU's `GPU-…` UUID with MIG disabled,
`challenger_device` accepted the tagger running on it, and 15 windows over the pinned vectors' 37-block store went
through the one state file the store names.

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

`bound` at `s = 1,849`, `q ≤ 2^64`, `S = 239,992,832` bytes and `cap_side = 63.0 GB/s` (error `2^-40.42`
throughout); the rows at `T = 1.2 ms` are the rate granted at `5761861fd`, and the others are the formula's arithmetic:

| `T` | `RTT` | `I` | share of `E` | `I / (T + RTT)` |
|---|---|---|---|---|
| 1.0 ms | 0 | 753,956,385 bytes | 69.9% | 754 GB/s |
| 1.2 ms | 0 | 741,353,308 bytes | 68.8% | 618 GB/s |
| 1.2 ms | 0.3 ms | 722,448,693 bytes | 67.0% | 482 GB/s |
| 1.4 ms | 0 | 728,750,231 bytes | 67.6% | 521 GB/s |
| 1.708 ms (the largest `Params` allows) | 0 | 709,337,901 bytes | 65.8% | 415 GB/s |

At `T = 1.2 ms`, `bound`'s `I / T` is 42% of the raw read rate (1,476 GB/s). Of `E`'s 1,078 MB, the chip's storage
takes 240 MB (22.3%), the side link 76 MB (7.0%), the sample 17 MB (1.6%) and `λ` 4.1 MB (0.38%); the rest of the gap
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
3. **The helper residual** (§5): an interleaving helper covers most of another device's blocks, so Daniel stated
   `bandwidth-device` for a co-audited set of whole PCIe-only GPUs (decision 4). The set's figure needs AuditRO for `m`
   stores (lane `ra-bandwidth-lean`); until then the per-device bound covers an isolated device.
4. **The store's leaves are PoUS's, not P2 spec §4's.** `vk`'s leaf is the 99-byte little-endian leaf of
   `pous.verifier`; the P2 spec's is 107 bytes, big-endian (`benchmarks/pous/p2_v1/wire.py`). The audit uses what PoUS
   uses on `main` and inherits its resolution.
5. **`S_chip` counts every on-chip store** (§5): L2, the register files, L1 and shared memory, and an allowance of
   160 KiB a SM and 1 MiB for the chip for the instruction and constant caches, uniform registers, TLBs and chip-level
   buffers, the red team's. The instruction caches are a third party's measurements on another GB202 card; measuring
   them on the RTX PRO 6000 is open.
6. **Openings disclose sampled blocks** of `C*` (an encoding of `W`) to the verifier; the memory challenger already
   holds `vk` and publishes only verdicts, so nothing leaves it.
7. **The audit imports PoUS** (`pous.codec`, `pous.schemes.p2`, `pous.verifier`) for its store and setup. Daniel
   allowed this import edge (decision 7, 11 Oct 2026), and `tests/test_boundaries.py` enforces it.

## 8. Code

- `tags.py`: `get(name)`, each construction's `keyed(σ)(j, block)`, `certified`, `guess`, `info_bits`; the tag tree
  (`leaves`).
- `audit.py`: a device's id and salt (`device_id(uuid)`, `read_device(gpu)` from nvidia-smi,
  `draw_salt(source, commitment, device)`), the prover's
  reference (`Store`, `tag_pass(store, tag, σ) -> TagCommitment` with `root` and
  `open(j) -> Opening`), the store's challenger state (`ChallengerState.create(state_dir, vk)` once, then
  `ChallengerState(state_dir, vk)`, the file `state_name(vk)` in the challenger's state directory, shared by every
  verifier of the store), the verifier
  (`Verifier(vk, params, source, state, device=, commitment=, salt_source=).window()`: `open()`, `start() -> σ`,
  `commit(root) -> samples`, `verdict(openings) -> Verdict`; `window_key(source, vk, tag, nonce)`), the proof's point
  (`EPSILON`, `SAMPLES`, `QUERIES_LOG2`, `SLACK_LOG2`, `certified_blocks`, `samples_for`) and `bound`.
- `tests/bandwidth_vectors.py` writes `tests/vectors.json`: tags at 2,056 bytes for every construction, two devices'
  ids and salts, window seeds and samples over the first device's store, and a whole window over 37 blocks.
- `benchmarks/bandwidth`: the GPU tagger (`tagger.cu`, one kernel per tag, the leaves and the tree), its harness API
  (`gpu.py`: `challenger_device(tagger, gpu)` is the id the challenger reads, refused unless `Tagger.device()`, the
  tagger's CUDA device, is that GPU; `Tagger.tag_pass(tag, store, blocks, σ, workspace)` returns the root; `open(j, store_tree)` gives the
  `Opening` this verifier checks; `Prover(tagger, store, blocks, scrub)` is a store's one prover for every tag, and
  `commit(window)` opens the window, prepares, then takes `σ` and commits; `windows` runs every window of every tag through it and
  through the store's one challenger state), its Python twins
  (`benchmarks/bandwidth/tests/test_tagger_twin.py`), and the runs of §6.
