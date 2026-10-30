# The bandwidth audit

A run of the bandwidth audit certifies how much of a fixed store a device read from its device memory into the chip
within a deadline of a fresh seed. This is the audit over PoUS's store (v0), one of the subprotocols
(`ARCHITECTURE.md`, Glossary), and no composite protocol runs it: its tag `ro-blake3/v1` is BLAKE3, which the one-hash
rule (`AGENTS.md`, "Design rules") keeps off the served path. The bandwidth subprotocol on the served path is
PoUB (`verity/core/protocols/poub/`), a different protocol, which applies this audit's guarantee per pass.
This package stays for the benchmarks that measure its tags (`benchmarks/bandwidth`).

## 1. What a run certifies, and on what

An accepted run certifies that the device moved at least `I` bits of the store `E` from its device memory into the chip
within `T + RTT` of receiving the seed `σ`, on the verifier's clock: its memory delivered at least `I / (T + RTT)` to the
audit. For a certified tag (`ro-blake3/v1`, `ro-sha512/v1`; §4),

~~~text
I  ≥  k·b − S − c − G·b − λ   bits,     k = ⌈(1 − ε_s)·N⌉
~~~

except with probability at most

~~~text
((k − 1)/N)^s  +  s/2^h  +  C(q, k)/2^λ
~~~

over the store, the oracle and the verifier's samples. `N` is the store's label count and `b` a label's bits; `S` the
bits the chip holds when `σ` arrives; `c` the bits every side source can deliver in `T + RTT`; `G` the labels the device
can rebuild from setup within the run; `q` its oracle queries; `h` a tag's bits; `s` the samples; `λ` the slack the last
term charges. The three terms are the three ways a sampled label passes without its contents on chip at a query: the
samples miss the labels the device never queried; a tag is guessed; or the device queried `k` labels with less
information than they hold, which would describe the store in fewer bits than it has.

This is `Bandwidth.Guarantees.AuditRO` (`Security/Definitions/Accounting/Guarantees/Bandwidth.lean`, over the game in
`Security/Definitions/Accounting/Bandwidth/Game.lean`), proved as `Bandwidth.SecurityProofs.AuditRO` with no named
assumption: the device's bounds `S`, `c`, `G` and `q` define the class of devices it quantifies over. Its completeness
is `HonestPasses`: an honest device is accepted on every store, oracle and sample, with ingress exactly `N·b`.
`AuditRelay` covers side sources that relay tags (below). `audit.bound` computes `I` and the error for a certified tag
and refuses any other (`test_bound_is_the_guarantee_at_the_proofs_point`, `test_the_fast_check_has_no_bound`).

**The operating point.**

| parameter | value | source |
|---|---|---|
| `N`, `b` | `2^19` labels of 16,448 bits (2,056 bytes); `E` is 1,077,936,128 bytes | PoUS's P2 store (§3) |
| `ε_s`, `k` | 1/64; 516,096 labels, `E` less 16.8 MB | `audit.EPSILON`, `certified_blocks` |
| `s` | 1,849, the fewest with `((k − 1)/N)^s ≤ 2^-42` | `audit.SAMPLES`, `samples_for(N, -42)` |
| `h` | 256 (`ro-blake3/v1`) | §4 |
| `q` | at most `2^64` oracle queries in a run, past what any fleet computes in one, so the bound needs no assumption on `q` | `audit.QUERIES_LOG2` |
| `λ` | `64·k + 41` = 33,030,185 bits (4,128,774 bytes, 0.38% of `E`): `C(q, k)/2^λ ≤ q^k/2^λ = 2^-41` | `audit.SLACK_LOG2` |
| `T` | 1.2 ms, which leaves up to 0.508 ms for `RTT` under the rebuild floor | `GRANTED_T_NS` (`benchmarks/resource_accounting/certified.py`) |
| `S` | 239,992,832 bytes (22.3% of `E`), below | `summarize.s_chip_bytes` (`benchmarks/bandwidth`) |
| `c` | `cap_side · (T + RTT)`, `cap_side` = 63.0 GB/s: the GPU's PCIe Gen 5 x16 link at its raw rate, 32 GT/s × 16 lanes at 128b/130b | `summarize.PCIE_GEN5_X16` |
| `G` | 0: `Params` refuses `T + RTT` at or past one P2 root (§3) | `audit.REGEN_FLOOR_NS` |

At this point the error is `2^-42.01` (samples) + `2^-245.1` (guesses) + `2^-41` (compression) ≤ `2^-40.42`, and with
`RTT = 0` an accepted `ro-blake3/v1` run certifies at least 741,353,308 bytes of `E` (68.8%), 618 GB/s over `T`. Over a
monitoring period, the ingress of a store's accepted runs adds up, at their errors summed, because no two runs over a
store are open at once (§4, §5).

`S` counts every on-chip store a resident kernel can hold data in across `σ`, at its raw capacity: 128 MiB of L2; for
each of 188 SMs, 256 KiB of registers, 128 KiB of L1 and shared memory, and 160 KiB for its other stores; and 1 MiB at
the chip level. An SM's other stores are its instruction caches (about 4 KB of L0 per SMSP and 64 KB of L1, measured on
the same die in [Dissecting SM_120](https://zartbot.github.io/micro_arch/nvidia/sm_120/), chapter 7), constant caches,
uniform registers (256 bytes a warp) and TLB, where a kernel can keep constants, or data as instruction immediates,
across `σ`. The chip's are its copy-engine, PCIe and memory-controller buffers and its video engines. sm_120 has no
Tensor Memory (the same reference, chapter 14).

**The premises** that carry the bound to a real device are claims of `verity.core.claims`:

- **`random-oracle`** (assumption): the tag is a random oracle fresh at `σ`: for `ro-blake3/v1`, keyed BLAKE3 under a
  key derived from `σ` by SHA-512; for `ro-sha512/v1`, SHA-512 after a prefix that holds `σ`. The proof's oracle is one
  call, `H(σ, j, E_j)`, while BLAKE3 is a tree of compressions; indifferentiability doesn't carry a bound like this one
  over to the tree (Ristenpart, Shacham and Shrimpton), so that step rests on `bandwidth-instantiation`.
- **`bandwidth-instantiation`** (assumption, conjectured): the labels of `E` are no easier to hold information about
  than uniform ones, except by rebuilding them from setup; this is where P2's incompressibility enters. It also covers
  `ro-blake3/v1`'s tree: chunks 1 and 2 of its message (`E_j[1016:2040]` and the 16-byte tail) hold no byte of `j`, so
  labels with equal aligned chunks share that chunk's chaining value within a run, and a device tags them with fewer
  compressions. P2's labels don't repeat (near-uniform values mod `p` under a fresh key per label), so the saving on `E`
  is nil; a store with repeated aligned chunks is outside the assumption.
- **`bandwidth-s-chip`**, **`bandwidth-side-cap`**, **`bandwidth-regen-time`** (assumptions, measured): `S`, `c` and
  the rebuild time on the stated device, each falsified by a measurement (a larger on-chip working set, a faster path
  in, a faster root). The device is a whole GPU, not a MIG slice: slices share the chip's L2 and crossbar, so a
  co-resident slice's path in is on chip, not over PCIe. The RTX PRO 6000 Server's only link is its PCIe Gen 5 x16 (no
  NVLink, no chip-to-chip link), shared by the host, NVMe, NIC and peer GPUs; host RAM reaches the chip only over it,
  inside `c`.
- **`bandwidth-device`** (model): the device is a chip with device memory and side sources, all laid out before `σ`; it
  reads device memory only through counted ingress, computes every tag on chip, and commits before the samples. A tag
  computed elsewhere from a copy of the store and sent in is outside the model: `c` counts bits, a helper's 32-byte tag
  delivers a label's share of the audit in 1/64 of its bytes, and `c` carries more tags than `N`. What limits a helper
  is its reads, and one that reads a copy of the store between its own reads covers nearly all of it within `T + RTT`,
  so charging it against `k` leaves nothing. `bandwidth-device` is therefore stated for a co-audited set:
  - **Which devices count.** Whole GPUs whose only link is PCIe: no MIG slice and no NVLink. Every device that could
    hold a copy of a set's stores is in the set or isolated.
  - **What a set would certify.** The set's devices run at once, with one start and one deadline (an offset is a
    helper's time), over pairwise-distinct stores (§3). The set's chips are then one chip, and the same compression
    argument gives `Σ_i I_i ≥ Σ_i k·b − Σ_i S_i − c_ext − Σ_i λ` bits, except with probability the sum of the devices'
    errors, where `c_ext` counts only links from outside the set; for `m` RTX PRO 6000s, each on its own PCIe link,
    `c_ext ≤ m · cap_side · (T + RTT)`. That is `AuditRO` for `m` stores sampled separately, which has no Lean statement
    yet, so a set gets no figure.
  - **What is certified.** `I ≥ k·b − S − c − λ` for an isolated device, one whose side links carry nothing computed
    from any copy of its store during the run, as `AuditRO` states. A peer that relays tags over the side link is
    charged inside the side budget by `AuditRelay`, at `m·b` bits beyond `c` for `m` relayed labels with
    `m·(h − log₂ N) ≥ c + μ`; at the full PCIe cap that is more labels than `N`, so a device a peer can reach gets no
    figure. A device that is neither isolated nor in a co-audited set gets none either.
- **`trusted-host`** (assumption): "The auditor's device runs its stated code, timing included, on a host the prover
  doesn't control, with its CPU isolated from other tenants and a monotonic clock; the host never restores its earlier
  state." The audit cites it only for the verifier's state (§5). It covers no machine the prover controls: not the GPU's
  node, and not the device id the node reads (§2).

## 2. The parties

No guarantee assumes trusted prover hardware.

- **The verifier** is the auditor's side, run on its memory challenger: the auditor's hardware, on a machine the
  prover doesn't control, never a process on the GPU's node (`audit.Verifier`, `Window`, `ChallengerState`). It holds
  the secret `source`, draws each run's nonce, derives its key, sends `σ`, runs the clock, draws the samples, checks the
  openings and keeps the store's state (§5).
- **The prover** is everything on the GPU's node: the GPU, and the code that reads its id and drives the tagger
  (`audit.read_device`, and `benchmarks/bandwidth`'s `gpu.challenger_device` and `gpu.Prover`), carrying `σ`, the root,
  the samples and the openings between the GPU and the verifier. The guarantees hold if it is adversarial: `σ` and the
  samples come from the verifier's key, which the node never holds, and leave the verifier only when it sends them, the
  samples after the root; the deadline runs on the verifier's clock, so `RTT` includes the hop to the node; and every
  opening is checked against `vk` and the committed root.

One thing rests on the prover: **the device's identity.** The node reads the GPU's UUID and SKU from its own driver
(§3), so an adversarial node can report another GPU's. A run's bound holds for the store the verifier salted, whichever
chip answered, but which GPU that was is the node's word, and with it a per-GPU figure, the pairwise-distinct stores of
a co-audited set and the device class `S` and `c` are stated for. The GPU's confidential-computing attestation, checked
by the verifier, would move this to the chip; the audit doesn't use it.

## 3. The store and its leaves

`E` is PoUS's P2 store `C*` (`verity.core.protocols.pous.schemes.p2`): `N = 2^19` labels of 2,056 bytes (1.08 GB),
encoded from the useful data `W` under PoUS's setup (`pous.setup`: the commitment to `W`, then the verifier's salt, then
`vk`). Each label is two slow roots of its keyed payload, `c = ρ(σ_K(ρ(m + t)))`: under P2's assumptions a label is
incompressible except through `W`, and rebuilding it from `W` takes two roots in sequence. `Params` refuses `T + RTT`
at or past `REGEN_FLOOR_NS` = `2 · ROOT_CALL_NS` = 1,708,058 ns, the fastest cooperating P2 root: one root, not two, so
even a device holding a label's inner root rebuilds nothing, and `G = 0` (`test_params_refuse_a_window_that_fits_a_root`).

The audit reuses PoUS's setup and `vk`, with the salt as its one change of input (below). A label's identity is its
leaf under `vk.root`, and the verifier holds only `vk`. The leaf is PoUS's (`pous.verifier.leaf_hash`): SHA-512 of 99
bytes, `0x00`, `salt ‖ u64(segment)` as 32 little-endian bytes, the label's index in its segment as 2 little-endian
bytes, and the label's `sha512/row/v2` digest. The tree is RFC 6962's in SHA-512 (`pous.verifier.MerkleTree`).

**One store per device.** The verifier draws each device's salt with the device's id bound in,

~~~text
salt = derive(source, "verity/bandwidth/salt/v2", {w: commitment, device: id}).stream("salt", 24)
~~~

(`audit.draw_salt`), and PoUS's setup runs with it in place of `pous.setup.draw_salt`'s. PoUS's setup takes the salt as
an input and needs only that it is drawn after the commitment to `W`, from the verifier's secret or a later beacon
round, which this one is, so PoUS is unchanged. Devices that hold one `W` under one secret then hold different stores,
so their `vk`, seeds and tags differ, and one device's commitment fails another's run
(`test_one_devices_commitment_fails_anothers_window`). `audit.Verifier` takes the device's id, the commitment to `W` and
the salt's source, and refuses `vk` before any run unless `vk.salt` is `draw_salt`'s for them: a store set up under
PoUS's own salt, another device's store, or one salted under another commitment is never audited as this device's
(`test_a_store_not_salted_for_its_device_is_refused`).

The salt carries the device, not the run's key. With the device only in the key, two devices holding byte-identical
stores would get different seeds, but one device that reads each label once could tag it under both seeds and send the
other's tags over its side link. Distinct stores are what make the ingress of co-audited devices add up (§1).

**The device's id** is the 16 bytes of its GPU's UUID (`audit.device_id`), read only from the one spelling nvidia-smi
prints, `GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx` in lowercase hex; uppercase, missing dashes, braces, a MIG slice's
id and anything else are refused, and `draw_salt` refuses any value but 16 bytes, so one GPU has one id and one salt
(`test_a_device_has_one_id_its_gpus_uuid`). `audit.read_device(gpu)` reads it on the GPU's node: it asks
nvidia-smi (NVML) for the GPU's UUID and MIG mode and refuses a MIG slice, whether the selector names one or the GPU is
in MIG mode, whose CUDA devices are its slices, each with a UUID of its own
(`test_the_challenger_reads_a_devices_id_from_the_driver`). `benchmarks/bandwidth` refuses a tagger whose CUDA device
(`cudaDeviceProp.uuid`, `Tagger.device()`) has another UUID (`gpu.challenger_device`,
`test_the_challenger_reads_the_device_it_audits`), which keeps an honest node consistent and binds an adversarial one to
nothing (§2). The SKU the node reports is recorded beside the id, and is the node's word too.

## 4. A run and its timing

A run is one execution of the audit over one store (`Verifier.window()`, a `Window`):

1. **Seed.** The verifier draws a fresh 32-byte nonce from its own randomness and records it in the store's state
   (`Window.open`, §5). It derives the run's key from its own secret,
   `key = derive(source, "verity/bandwidth/window/v2", {root: vk.root, salt: vk.salt, tag: name, nonce: nonce})`
   (`audit.window_key`, `verity.ml.randomness`), sends `σ = key.stream("seed", 32)` and starts its clock
   (`Window.start`). The nonce is never a counter and never drawn twice for a store, so no two runs over a store share
   a key, whichever verifier, tag or restart opens them. A key fixed by a run number would repeat in every verifier
   counting from the same place: a fast-check run would reveal, after its commitment, the samples of the certified
   tag's run of the same number, and a restarted verifier would accept a root replayed from before the restart
   (`test_a_second_verifier_of_the_store_repeats_no_window`, `test_another_tags_window_reveals_no_samples`,
   `test_a_restarted_challenger_repeats_no_window_and_refuses_a_replayed_root`).
2. **Tags.** The prover computes `y_j = Tag(σ, j, E_j)` for every `j < N`, packs them into leaves and returns the
   leaves' RFC 6962 SHA-512 root (below). The verifier stops its clock on the root (`Window.commit`); the root is on
   time iff the elapsed time is at most `T + RTT` (`Params.limit_ns`).
3. **Samples.** Only for an on-time root, the verifier draws `s` indices, `key.uniform(N, ("sample", t))` for `t < s`,
   uniform with replacement, and sends them. A late or malformed root closes the run rejected and draws nothing.
4. **Openings.** For each sample `i`, in order, the prover sends `E_i` with its path to `vk.root`, and the tags of
   `i`'s leaf with the leaf's path to the committed root (`TagCommitment.open`). The verifier checks both paths and
   recomputes `y_i` from `σ`, `i` and `E_i` (`Window.verdict`). Openings are not timed: the root binds every tag.

The verifier accepts iff the root is on time and every opening checks. It refuses a second start or a second root, and
never redraws. One run over a store is open at a time: from its record until its root arrives, or until `T + RTT` after
`σ` without one, the verifier refuses to open another over the store, which stays closed and may open once the first has
closed (`test_one_window_over_a_store_is_open_at_a_time`). Two runs open at once could both be answered from one read of
each label, so their ingress wouldn't add.

**Timing.** `T` covers the honest prover's elapsed time on the verifier's clock with margin, and `T + RTT` stays under
the rebuild floor (§3). The bound holds for every device state laid out before `σ`, so the verifier may open a run when
the prover says it is ready, and the prover's preparation costs the verifier nothing. The honest prover (`gpu.Prover`,
held to this by `benchmarks/bandwidth/tests/test_prover.py`) opens the run first, so the state's fsync falls before its
preparation and not between the preparation and `σ`; then prepares, with one untimed pass under a throwaway seed and a
read of 512 MiB of another buffer, so the run starts with its launch path warm and L2 holding clean lines and no part of
`E`; and only then asks for `σ`. `ro-blake3/v1`'s loads fetch 256 bytes into L2 per miss
(`ld.global.nc.L1::no_allocate.L2::256B`), which helps when L2's lines are clean and costs a write-back first when they
are dirty, so `T` is set for clean lines.

### Tags

~~~text
Z(x)           x zero-padded to whole 128-byte chunks
prefix(D, σ)   Z(D ‖ frame(name) ‖ frame(σ))        86 bytes for tags, 91 for leaves, padded to 128: SHA-512's first chunk
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

`frame` is `verity.ml.randomness.frame`. `ro-blake3/v1`'s message is 2,064 bytes, three BLAKE3 chunks (1,024, 1,024 and
16 bytes): 33 chunk compressions and 2 parent compressions, each starting from `K`, which keyed mode puts in every
chunk's and every parent's chaining value. Its leaf of 16 tags takes 5 SHA-512 compressions after the prefix.

**Every tag is seed-first, `H(σ, j, E_j)`.** No state over any byte of `j` or `E_j` can be computed before `σ`:

- `ro-sha512/v1`: the prefix is 86 bytes (a 24-byte domain, the 21-byte framed name, the 41-byte framed `σ`) padded to
  128, so `σ` lies wholly in SHA-512's first chunk and `u64be j` starts the second. Every chaining value over `j` or
  `E_j` descends from a compression of `σ`.
- `ro-blake3/v1`: `K` is a SHA-512 of the same prefix, and every BLAKE3 compression of the message starts from `K`.
- `gf31-poly/v1`: the point `g_j` is a SHA-512 from the prefix's midstate.

That is what lets a random oracle fresh at `σ` model the tag. A tag with the label before the seed, `H(E_j, σ)`, has a
chaining state after `E_j` (64 bytes for SHA-512, 32 for a BLAKE3 chunk under a public key) that a device computes
before the run and keeps instead of the label: it then answers every tag reading 1/32 of `E` or less, and the run
certifies nothing. `test_every_tag_is_seed_first` checks, for every tag, that the framed `σ` lies within the first 128
bytes it hashes and that a label's tag changes with `σ` alone.

| | `ro-blake3/v1` | `ro-sha512/v1` | `gf31-poly/v1` |
|---|---|---|---|
| role | certified, the operating point's tag | certified | uncertified fast check |
| `h`, a tag's bits | 256 | 512 | not a random oracle |
| the honest prover's cost per label after `σ` | 35 BLAKE3 compressions | 17 SHA-512 compressions, and one for its leaf | one SHA-512 compression for `g_j`, then 514 multiply-adds mod p |

`ro-blake3/v1` is the certified tag that fits the deadline on the RTX PRO 6000. `ro-sha512/v1`'s honest pass there is
past the rebuild floor, so no `T` that `Params` allows admits it; it stays as the all-SHA-512 tag.

**`gf31-poly/v1` certifies nothing about bandwidth.** A polynomial evaluated at a point is not a random oracle: a
Kedlaya–Umans evaluation structure for each label's polynomial answers any `g_j` from a few of its words, and the
structures for all of `E` fit in the RTX PRO 6000's device memory beside the store. A device holding them answers every
tag without reading `E_j`'s words, so the compression argument has nothing to count, and `bound` refuses the tag. It
stays as a fast check that a device holds `E` (or something equivalent): a device missing labels fails it as it fails
the certified tags. Its word map binds only the words mod p (`w`, `w + p` and `w + 2p` agree), 15,934 of a label's
16,448 bits (`info_bits`).

## 5. The verdict and the record

The verdict publishes one bit, whether the run was accepted (`Verdict.public`); which check failed (`late`,
`malformed`, `store-path`, `tag-path`, `tag-mismatch`), at which sample, and the elapsed time stay with the verifier.
Openings reveal labels of `C*`, an encoding of `W`, to the verifier, so it runs on PoUS's memory challenger, which
already holds `vk` and publishes only verdicts.

**The store's state.** §4's two rules, no repeated key and one open run per store, hold across every verifier of a
store, whatever its tag, and across restarts, because every verifier of a store takes the store's one state
(`audit.ChallengerState`), kept on the verifier's machine and never on the GPU's node.

- It is a file in the verifier's state directory, passed explicitly and never a signer's key directory, named by the
  store: `<vk.root>-<vk.salt>.state` in hex (`audit.state_name`). The directory holds nothing else:
  `ChallengerState.create` and opening refuse a directory holding any entry that isn't a state named by the store its
  first line names, so a copy or hard link of a state under another name, a symbolic link, a subdirectory or a stray
  file is refused for every store, and `create` refuses a store that has a state. A store therefore has one state in
  the directory, never made again, and the directory's listing shows which stores have one
  (`test_a_store_has_one_challenger_state`).
- Its first line is `verity/bandwidth/challenger-state/v1 ROOT SALT`, the store's root and salt in hex. Each run then
  appends `open NONCE TAG`, fsynced before its `σ` leaves (`test_the_state_records_a_window_before_its_seed`), and
  `close NONCE` when it closes: at its root, at `T + RTT` without one (recorded when the next run opens), or when its
  verifier stops first. Nothing rewrites or truncates the file.
- The verifier holds the file's lock while it runs, so one verifier holds a store's state at a time. A state that is
  missing, a link, unreadable (a first line that isn't the format's, a record that is neither, a nonce opened twice, a
  close that isn't the open run's, a last line cut short), another store's, held by another verifier, or changed since
  this verifier last wrote it opens no run (`test_the_state_fails_closed`). A run the file shows open belonged to a
  verifier that stopped: reopening records it closed, and no verifier can accept it
  (`test_a_window_open_when_its_challenger_stops_never_passes`).

**What `trusted-host` is for.** A file can't detect being restored from a snapshot, nor a directory a copy of a state
kept in another directory or on another host: that the verifier keeps one state directory, never copies a state and
never gets an earlier copy back rests on `trusted-host` for its machine. A single run's bound doesn't need it: `AuditRO`
needs only that the prover couldn't know `σ` or the samples before `σ` left, and both come from the run's key, which
binds the verifier's secret and a nonce from its own randomness. Two runs over a store share a key only if their nonces
collide, with probability below `n²/2^257` over `n` runs, so the state's refusal of a nonce it holds only catches a
source that repeats, and a state rolled back, restored or forked replays no run unless the randomness repeats too. The
assumption is needed for the two properties of runs that overlap in time:

- **One open run per store.** A second copy of the state, or one restored to before an open run's record, would let a
  second verifier open a run over the store while that one is open, and both could be answered from one read of each
  label.
- **Additivity.** The ingress of a store's runs over a monitoring period, or of a co-audited set's, adds up only
  because no two runs over a store are open at once. Without it each accepted run's bound still holds, but their sum
  doesn't bound the total.

## 6. The code and its tests

- `tags.py`: the three constructions (`get(name)`, each one's `keyed(σ)(j, label)`, `certified`, `guess`, `info_bits`)
  and the tag tree (`leaves`).
- `audit.py`: the device's id and salt (`device_id(uuid)`, `read_device(gpu)` from nvidia-smi,
  `draw_salt(source, commitment, device)`); the prover's reference (`Store`, `tag_pass(store, tag, σ) -> TagCommitment`
  with `root` and `open(j) -> Opening`); the store's state (`ChallengerState.create(state_dir, vk)` once, then
  `ChallengerState(state_dir, vk)`, the file `state_name(vk)`); the verifier
  (`Verifier(vk, params, source, state, device=, commitment=, salt_source=).window()`: `open()`, `start() -> σ`,
  `commit(root) -> samples`, `verdict(openings) -> Verdict`; `window_key(source, vk, tag, nonce)`); `Params` (the tag,
  `s`, `T`, the `RTT` allowance and the rebuild floor); the operating point (`EPSILON`, `SAMPLES`, `QUERIES_LOG2`,
  `SLACK_LOG2`, `certified_blocks`, `samples_for`) and `bound`.
- `openings.py`: the openings of PoUB's chained passes, `P` passes over the store with one pooled draw of
  `s` openings after the last root: the draw (`draw`), its sampling error alone and over a monitoring period's draws
  (`window_error`, `horizon_error_log2`), `λ` in three forms (`slack_bits`: `crude`, this audit's `64·k + 41`; `exact`;
  and `named`, a proposal with no Lean statement), and the expected bytes and verifier operations of a draw's openings
  sent as multiproofs (`opening_cost`, `retention`, `point`, `cheapest`, `achievable`).
- `tests/test_bandwidth.py`: each tag against its formula, the seed-first property, the tag tree and its short last
  leaf, honest runs over a P2 store and at P2's width, every rejection (late, malformed, a wrong label or tag path,
  guessed and skipped labels, openings out of order), a run's once-only and one-open rules, the device's id and salt,
  the state's rules, the operating point and `bound` against the formula.
- `tests/bandwidth_vectors.py` writes `tests/vectors.json`, and `tests/test_bandwidth_vectors.py` holds the code to it:
  tags at 2,056 bytes for every construction, two devices' ids and salts, run seeds and samples over the first device's
  store, every tag's root over 37 generated labels, and a whole run over them (37 is not a power of two, and the
  16-tag leaves end short). Label `j` is the first bytes of the SHA-512s of
  `"verity/bandwidth/test-block/v1\0" ‖ u64be j ‖ u32be c` for `c = 0, 1, …`, concatenated.
- `tests/test_openings.py`: `openings.py` against this audit's sampling term and slack, and its multiproof and
  retention expectations by enumeration against `verity.ml.commitments.multiproof`.
- `benchmarks/bandwidth`: the GPU tagger (`tagger.cu`, one kernel per tag, the leaves and the tree), whose roots over
  the full store and the vectors' 37 labels must equal `tag_pass`'s before anything is timed; its harness (`gpu.py`:
  `challenger_device(tagger, gpu)`, `Tagger.tag_pass`, `open(j, store_tree)`, `Prover(tagger, store, blocks, scrub)`
  and its `commit(window)`, and `windows`, which runs every tag's runs through the store's one state); its Python twins
  (`benchmarks/bandwidth/tests/test_tagger_twin.py`); and `summarize.py`, which evaluates `bound` at the device's `S`
  and `c`.
