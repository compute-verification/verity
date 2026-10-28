# The remote bandwidth audit v0 (`verity/remote-bandwidth/v0`)

A verifier with no hardware at the site certifies, window by window, how many reads of a node's registered blocks the
node made. The node's memory is its GPUs' HBM and its host's DRAM. No prover hardware is trusted: the host, its CPUs
and its NICs are all part of the prover. Each window's challenge and its openings coin are live coins from the
challenger, committed before the session. Inside a window the passes' seeds are Fiat–Shamir: each is derived from the
window's challenge and the prover's previous root, so the passes are chained. That is allowed for this protocol,
because a live coin a pass is impractical at its rate (`AGENTS.md`, "Live coins only"). Appendix A is the
live-coin design and what it would cost.

The code is `audit.py` (the lifecycle), `coins.py` (the coins), `receipts.py` (seeds, plans, receipts and trees),
`codec.py` (the store's encoding), `memory_map.py` (serving's record of the KV blocks it wrote and freed, the audit's
input from serving), `store.py` (a registered store bound to a pinned model, §3), `timing.py` (the honest schedule,
and where the sketch's bound is read) and `p2.py` (the audit over PoUS's P2 store, §11). Every byte format below is
pinned by
`tests/vectors.json`. This is a reference implementation: pure Python, with openings checked in the clear. The clear
checks are the specification of what each zero-knowledge proof block proves on the deployed path (§8). The verifier's
definition is Lean code, `lean/` (`RemoteAudit`: `verifyWindow`, a window's draw and its openings' checks, and
`RemoteAudit.Session`, the session's bookkeeping call for call), which `audit.py` mirrors and is held to by
`lean/tests/`' difftest, comparing every event's result, the outcome and the record. The Lean is the verifier of
record, and `audit.py` is the reference that runs beside it (`README.md`). Both refuse a registration that binds PoUW's
tiles.

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
| \(\lambda\) | AuditRO's slack: per pass of \(k\) receipts (every certified one, in either reading), `named_bits(k)`, a whole number of bits at least \(\log_2(q^k/k!) \ge \log_2 C(q, k)\) at \(q = 2^{64}\) (`queries_log2`), and `slack_log2` (`SLACK_LOG2`, 67: the failure probability's \(2^{-\text{slack}}\) a pass, below, adds at most \(2^{-50}\) up to \(2^{17}\) passes, more than an hour of the node's 41.2 ms passes over gate runs 8 and 9's 532.6 GB store); 0.116% at the node's 192.6M 4 KiB blocks a pass over every tier (37.9 bits a block), more than \(\mu_{\text{sample}}\) | proved (`Bandwidth.Guarantees.AuditRO`, applied per pass) |
| \(\mu_{\text{rebuild}}\) | \(C_{\text{sq}} b / (E_{\text{sq}} R)\): blocks re-encoded from their payloads in place of reads, each at least the record's codec's \(E_{\text{sq}}\) squarings (`Codec.encode_squarings`: 98,016 at p1024/k3/masked, 65,344 at k = 2, 65,152 at p512/k2) at the node's stated ceiling of \(C_{\text{sq}}\) squarings a second at that width (`Rebuild`) | the record's codec, and assumption 5 stated per width (`WorkHardness(p, k)`) |
| \(\mu_{\text{host}}\) | the host tier's own shortfall, and host DRAM bandwidth PCIe can't carry | a model, measured capacities |
| \(\mu_{\text{link}}\) | under a registration that binds \(z\) only (0 otherwise): relays that meet \(z\) and a copy of the block over a link (§4) | a model |
| \(\mu_{\text{carrier}}\) | only when post-seed receipts are credited (0 otherwise): \(\sum_{t,i} \min(C_t, a_{t,i} b) / (R\Sigma)\), \(a_{t,i}\) pass \((t, i)\)'s post-seed blocks and \(C_t = R_{\text{in}} L + (c_\rho \cdot U_t b \cdot L / E) b\) the verdict's carrier bytes: a credited post-seed block may hold another block's encoding, and one read of it then serves two credited receipts of the pass (§3) | the property's verdict (I2's `P(c, R_in, L)`) |

The protocol bounds the first four, prices \(\mu_{\text{rebuild}}\) from the record's codec (its name and parameters)
at a stated squaring ceiling, passes \(\mu_{\text{host}}\) and \(\mu_{\text{link}}\) through (the record can't bound
them; a protocol carries no policy parameter), and reads \(\mu_{\text{carrier}}\) off the property's verdict. `ledger` takes the rebuild model as a `Rebuild(codec, squarings_per_s)` and refuses one of another
codec than the record's, or a bare fraction. The stated ceiling is the node's, the sum over its devices: 1.67 ×
`kv.cu`'s 1.499·10^10 squarings a second at 1,024 bits for each RTX PRO 6000 with its share of the host
(`kv.cu`'s FOLD 6; 1.567 × 5.13·10^10 at 512 bits),
2.003·10^11 for the node's eight. There, and at the node's measured \(R\) of 12.92 TB/s, \(\mu_{\text{rebuild}}\) is
0.065% under the default codec, 0.097% at k = 2 and 0.31% under p512/k2 (`timing.SQUARINGS_PER_S`,
`timing.READ_BYTES_PER_S`). The failure probability is \(W\delta\) for a session of \(W\) windows read per window, or
\(\delta\) read over the horizon, plus \(2^{-\text{slack}}\) per pass, plus \(s/2^{512}\) per window for guessing a
receipt, plus the deployed draw's terms beyond \(\delta\) (`ledger.draw`, `PooledDraw`'s): \(\sum_t s'_t/2^{64}\) for
a rejected first word (`ledger.rejection`), the coin commitment's hiding terms (\(n \cdot 2^{-191}\) over \(n\)
windows), and the receipts' binding term, \(\sum_t \sqrt{|L_t| s_t} \cdot q_t/2^{256.5}\) under `cr/sha-512` for
window \(t\)'s fork finder of \(q_t = 2FC + 2 \cdot 69 \, s_t\) SHA-512 calls, which runs the prover twice from the
cut before the window's coin (`ledger.binding`). It is charged at the session's bounds, not at one record's \(U_t\)
and \(s'_t\), which follow the prover's play: \(|L_t|\), the most receipts a window of the session can have, is its
pass cap (the pace's \(k\), unpaced `STEP_LIMIT` − 1) times its extent (\(N\), or the codec's `max_blocks` when the
session reserves append coins), and \(s_t = \min(s, |L_t|)\). Drawn \(g > 1\) windows a coin (`groups_per_coin`), its
\(t\) is a draw: \(\lceil W/g \rceil\) of them, each with at most \(g\) times a window's \(|L_t|\) receipts and the
cap \(s\) over them together. It takes the chance that the two runs' roots differ as
0, which the deployed protocol doesn't yet ensure (the gap `binding-split`, §8). The windows' PRF advantages are the draw's fourth term: named with their claim
(`prf/sha-512`, `ledger.PRF_CLAIM`), with no value. The memory line's \(\delta\) carries the hiding and binding terms
beside its \(2^{-\text{slack}}\) a pass. Under the default receipt kind, sketch-sha512, a receipt also rests on the receipts' sketch (`SketchBound`,
derived by `Accounting.Guarantees.Remote.SketchBoundDeployed`), which adds at most \(Q \cdot 2^{-95.12}\) a session
at 16 rows for a prover of \(Q = F \cdot C\) SHA-512 compressions, \(C\) the node's own over the span and \(F\) the
outside-compute factor (\(2^{20}\), provisional), and \((Q + P) N \cdot 2^{-512}\) for a receipt right without its
hash, and \((Q + P)^2 \cdot 2^{-257}\) for the keys' distance from uniform (two seeds whose 256-bit sign keys collide
share their signs; `timing.sketch_failure`, §4): no more than \(2^{-40}\) up to \(Q = 2^{55.12}\), and
\(2^{-30.19}\) for one node's 10-minute session at \(F = 2^{20}\). At 64 rows, a registration's other row count, it
is \(Q \cdot 2^{-215.85} + (Q + P)^2 \cdot 2^{-257}\). block-sha512, kept for comparison, adds nothing. Neither adds
to γ.

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

The window is stated: \(T\) = 60 s (`Params.window_ns`, §2), with `SAMPLES` openings a window (`Params.samples`).
There, on one node whose store spans every tier (94 GB of each GPU's HBM and host DRAM's share, 789 GB,
`timing.spanning`), every window's whole \(\gamma_t\) is under 1% at the measured \(R\): 0.896–0.927% in the worst
window at RTT 1–20 ms. The horizon's 0.15% over 10 minutes is stated for its pooled \(\mu_{\text{sample}}\)
(0.0083%), with its whole γ (0.67%) reported beside it: \(\mu_{\text{chip}}\) exceeds 0.15% by itself, and λ with
\(\mu_{\text{rebuild}}\) (§10, `timing.stated`). Below about 380 GB over every tier (45 GB a GPU) no \(T\) meets 1%
(§10, `timing.store_floor`). 20 s and 1 s are options with their own readings: at 20 s and 11,091 openings every
window's \(\mu_{\text{sample}}\) is 0.25% and its whole \(\gamma_t\) 1.27–1.36%, and at 1 s and 2,773 its
\(\mu_{\text{sample}}\) is under 1% and its whole \(\gamma_t\) 5–11%.

**The node, and its one \(R\)**. The prover is the whole node (below), so every capacity a ledger takes is
the node's, the sum over its devices, whatever share of the node the store is on:

- \(R\) is the node's read capacity over every registered tier: its 8 GPUs' HBM plus its host's DRAM. Measured, that
  is 8 × 1,539 + 606.8 = 12,918.8 GB/s (`timing.READ_BYTES_PER_S`); specified, 8 × 1,597 + 819.2 = 13,595.2 GB/s
  (`timing.SPEC_BYTES_PER_S`).
- \(S_{\text{chip}}\) is every GPU's on-chip storage and the host's: 8 × 263,163,904 + 1,000,341,504 =
  3,105,652,736 B (`timing.CHIP_BYTES`). The host's part bounds the whole host, two Xeon 6776P sockets of 64 cores,
  whatever share its VM gets: their caches at the guest's CPUID sizes, with allowances for registers and buffers.
- \(C_{\text{sq}}\) is the sum of the devices' ceilings, 8 × one GPU's share (`timing.SQUARINGS_PER_S`).

One device's own values are `timing`'s `GPU_` and `HOST_` constants. A device the store isn't on can still hold store
bytes on chip and re-encode blocks, which is why \(S_{\text{chip}}\) and \(C_{\text{sq}}\) are the node's in every
session (as the red team found). A session over part of the node, as on one GPU, establishes its certified bytes,
\((1 - \gamma) R \Sigma\), in which \(R\) cancels, so they are the same at either \(R\). Its γ is the node's only once
the store spans the node. A store on one GPU leaves the other GPUs and host DRAM idle, about 88% of \(R\), in
\(\mu_{\text{timing}}\). A store in all 8 GPUs' HBM and none of host DRAM leaves host DRAM's 4.7% of \(R\) idle: 752 GB
of HBM reads 5.57–5.60% in the worst stated window (§10). Every window under 1% needs the store over every tier, host
DRAM holding its share.

**Which \(R\) is the headline.** γ is reported twice: against the node's specified read bandwidth,
which is the headline, and against the measured one beside it. The figures above are at the measured \(R\) = 12.92
TB/s. The node is specified at 13.60 TB/s (the RTX PRO 6000 Server Edition's 1.597 TB/s a GPU and 16 channels of
DDR5-6400), which no measured kernel reaches, so at spec every reading also carries that gap:
\(\gamma_{\text{spec}} = 1 - (1 - \gamma_{\text{meas}})\,R_{\text{meas}}/R_{\text{spec}}\), 5.82–5.85% for the worst
stated window. The 1% per window holds at the measured \(R\) only. The two-machine sessions so far are one GPU's, over
an 85 GiB store: both honest sessions accepted and a prover dropping 1/16 of its blocks rejected in its first window.
As the verifier recorded them, at one
GPU's constants and the 1.597× ceiling then stated, they read 0.0997 and 0.0999 at spec and 0.0659 and 0.0661
measured. At the node's \(S_{\text{chip}}\) and \(C_{\text{sq}}\) they certify 415.99 and 415.90 TB of the 433.9 TB
their receipts claim over 300 s (431.4 TB at one GPU's as recorded), and their γ
against the node's \(R\) is 0.898 at spec and 0.893 measured, nearly all of it the idle node. Either reading's \(R\)
is the verifier's, never the prover's: a node's own calibration can only raise it, because every term but the timing
term is a fixed byte count over \(R\Sigma\), and a prover reporting a slower read would shrink them all with no more
bytes read (0.066 read as 0.0057 at the slowest calibration still accepted). The measured reading takes each
tier's fastest measured read: 1,539 GB/s a GPU (an honest session's, `timing.GPU_READ_BYTES_PER_S`) and
606.8 GB/s of host DRAM (`timing.HOST_READ_BYTES_PER_S`), which
`benchmarks/remote_session/verifier.py`'s ledgers take as `timing.READ_BYTES_PER_S`. \(R\) is the rate of plain, incompressible reads,
which the store's encoded blocks are: the GPU's compression engine reads a compressible buffer faster
(7.55 TB/s against 1.536 TB/s plain), but the encoded store reads at 1.515 TB/s under a
compressible allocation, since the hardware compresses 128-byte units, one codec word, and no encoded word is constant
(the red team's runs). Other work's reads are bounded in bus time, not in logical bytes.

What the record says is conservation: correct receipts, at least \((1-\gamma)\) of the declared capacity. The prover
is one undifferentiated resource of capacity \(R\): the guarantee is over whatever reads the
node made, with no identity assumption about which hardware made them.

No round-trip floor is assumed. The chain orders a session's passes by itself, so the round trip appears only at the
session's two ends (§5). The live-coin design needs a trusted floor to be affordable at all (Appendix A).

## 2. Roles and parameters

- **The prover** is the node. It registers a store and runs chained passes (inference reads every weight block and the
  live KV pages once a decode step, topped up with filler reads). It sends each pass's plan with the root of its
  receipts, and opens what the verifier draws.
- **The challenger** is the proof service's one challenger (`sampled_proofs.challenger`, on the auditor's host through
  `sampled_proofs.remote` when it is remote). It draws every coin of a session, commits to them before the session,
  and releases each once, after the root the verifier recorded for it, in a release log it signs (`coins.Coins`).
- **The verifier** is remote. It holds a secret `source` (for the codec seed) and a clock, keeps the record, and checks.
  In the reference, `audit.Session` is the verifier and drives the challenger.

`Params`:

| field | meaning |
| --- | --- |
| `blocks` | \(N\), the registered store's blocks |
| `deadline_ns` | \(A\): window \(t\)'s roots are due by \(b_t + A\) |
| `windows` | \(W\), the windows in a session |
| `samples` | \(s\), the openings per window: `SAMPLES` unless another count is named |
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

**A store bound to a pinned model** (`store.py`). The payload root is the prover's choice unless the verifier holds it
to a store it can recompute. Under a binding (`verifier.py --model ROLE`), the verifier reads the model's pin, its files'
SHA-512s and the SHA-512 of their canonical JSON (`read_pin`), and holds every pinned file to its digest before reading
any (`check`). It then lays out the store the checkpoint alone fixes: \(E\) arenas of \(A\) blocks, one an engine at
one GPU, each holding the checkpoint's 2-D BF16 weights in vLLM's module order, each from a block boundary, zeros
between and after, and then host filler of zeros (`arena`, `Store`). Before step 1 the verifier sends a fresh 32-byte
salt key, and the prover takes block \(j\)'s 192-byte payload salt as `SHAKE256("verity/store/salt/v0\0" ‖ key ‖
SHA512("verity/remote-session/payload-salts/v0") ‖ u64be j)` (`store_salt`), so the payload root follows from the pin,
the store's shape and the key (`expected_root`). A hello is refused before the registry draws when its store is shorter
than the arenas, its layout isn't the store's regions, or its payload root isn't that root (`Binding.hold`). The
verifier's record keeps the binding: the model's name and `checkpoint_sha512`, the store's shape, both roots and
`bound`. A period's circuit is this tree's git hash, so the circuit a period signs commits to this code. The binding's
decision is also Lean (`lean/RemoteAudit/Store.lean`), and reading the files is Python only (§8).

**The codec** is Sloth over \(p = 2^{1024} - 105\), with the mask \(\sigma_K\) between rounds, at \(k = 3\): `p1024/k3/masked`,
`codec.DEFAULT`. The fallback is \(k = 2\) at the same width, `p1024/k2/masked` (`codec.FALLBACK`).
The prime is kv/v0's at 1,024 bits and the one the squaring ceiling was measured at. \(p = 2^{512} - 569\) unmasked
(`p512/k2`) stays available as the upgrade path. A registration may name any of them (or more rounds), and the session's
`Params` must name the same. Each is byte-exact with `verity/word-codec/kv/v0`, the KV kernel's codec (`kv.py`'s
`key(..., mask=True)`), and the tests pin `kv.py`'s vectors at k = 2 and k = 3 masked and at `p512/k2`:

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
\(k = 1\) is refused, because a word can then be partly kept and completed for less than an encode (red team).

**A block's key** is the seed's or its own. A block whose payload the payload root committed before the codec
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
the seed is credited only once the share of the store its writers can program is bounded (the stacking model): on the
property's verdict on the circuits that wrote it (I2, property/v1, \(P(c, R_{\text{in}}, L)\)), which `ledger` takes as
`credit_post_seed` and which charges what its carriers can hold (\(\mu_{\text{carrier}}\), §1). Without a verdict (None)
such a block may still be appended and read (§4), and its receipts are certified but not credited: one read of a carrier
then serves one credited receipt, its passenger's, so the credited reads are the registration's, whose payloads no
writer chose after the seed. A verdict that the property fails is refused. \(d_j\) is the payload leaf's \(x_j\), so the
leaf hashes the payload once for both (34 compressions), and \(b_j\) is one SHA-512 output, 2 compressions. The prover
keeps \(d_j\), or \(b_j\) in its place, beside the block: 64 bytes a 4 KiB block, 1.6%. Blocks under the seed's key are
unchanged, byte for byte.

**The append coin.** The digest alone leaves the writer advice: everything it chooses before committing \(d_j\) that the
key doesn't read (padding and dead-value bits, run inputs, re-key times, schedules) can still be chosen with the
encoding in view, and only an encode budget bounded that, a bound the stacked sketch analysis loses past about
\(41 - \log_2(\text{passes})\) bits. The coin makes that advice 0: the
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
which live coins rule out (a red-team finding). The verifier draws
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

Before a session the challenger draws its coins and commits to them, one epoch of the proof service's live coins
(`coins.py`, `sampled_proofs.challenger`), and the prover receives the commitment before the first coin:

~~~text
layout      scheme verity/remote-bandwidth/coins/v1, epoch k (the session's index over its registration), families
            challenge W, openings W and append A, the appends the session may take (none, and no append family, for
            a session that takes none: a session reserves exactly the coins it will use)
c           64 bytes a coin from the challenger's generator, committed whole (`challenger.live_leaf`, hm96-sha512 under
            a fresh salt, frame-v3-sha512 over the layout); the audit's coin is its first 32 bytes (`coins.coin`):
            window t's challenge σ_t is challenge coin t, its openings coin openings coin t, append i's c_a append coin i
R           SHA512("verity/remote-bandwidth/window/v1\0" ‖ registration digest ‖ u64be k): the session's window
after       σ_0 after R; σ_t (t ≥ 1) after the last root received when it leaves; window t's openings coin after window
            t's last root; append i's coin after append i's root
opening     (epoch, label, index, after, coin, salt, path): checked by the prover against the commitment (`coin_reason`)
~~~

The challenger can't choose a coin after seeing a root, and hm96 keeps every coin hidden until it is released. This
is the construction of PoUW's epoch coins, from the same challenger. Each coin leaves once, after the root the verifier
recorded for it, and the challenger logs both before the coin leaves its host and signs the log when the session ends.
The verifier holds that log to its own record (`Session.released_reason`): the commitment it used, the same roots in
the same order, σ_0 after \(R\), and each window's openings coin after that window's last root, so a coin opened before
its window's root is refused. On the auditor's host the session is the one window \(\{R\}\) of the auditor's signed
ledger (`remote.trusted_session`), which opens one challenger session a root, ever, and the window one of the audit's
period's (`coins.trusted`'s `period`): a session whose coins came from a second challenger session for the same
(period, registration), or whose window the ledger holds in another period, is refused, so a prover can't run two and
keep the coins it likes. `coins.trusted` reads that session only through `challenger.pinned_session`: the
host's pins log (`pins`, as its gate serves it) is the one the session's open sealed and records the open's pin, and
that commit is on the verifier's own `git rev-list --first-parent origin/main` (`main`), so the challenger ran a tree
main's check passed. A session with no pins log or no copy of main, an open that sealed none, or a pin off main's
first-parent line is refused.

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
execution, the engine step, and its plan holds every complete block the circuit reads at \(s\), filler beside them
allowed (§8, "The pass and its step"). The
step is public, an integer below \(2^{64}\), and the steps strictly increase over the session's passes, across its
windows, so no two passes cover one step. A root out of turn rejects with `order`: a window's roots come in index
order, and a window's first root makes the window before it final. A root with no step, or with a step not above the
previous pass's, rejects with `step`. A plan that breaks its format, or a change that removes a block its plan doesn't
hold or adds one it does, rejects with `plan`, and so does one that would take its window's receipts
\(U_t = \sum_i L_{t,i}\) to \(2^{64}\) (`RECEIPT_LIMIT`), so each of the window's draws reads its sampler word from its
stream's first SHA-512 block.

**Appends and releases.** Blocks written after the codec seed (KV pages) join the store mid-session. The prover commits
their payloads in an append tree (`append_tree`: the payload tree's leaves, \(x_j = d_j\), in a frame-v3-sha512 domain
bound to the registration's digest over the indices it takes) and sends its root and size (`Session.append`). The
verifier assigns the store's next fresh indices, from \(N\) up, asks the challenger for the append's coin \(c_a\), its
next append coin, opened after the append's root (`Coins.append`), and replies with the first index and the coin:

~~~text
prover → verifier   {"type": "append", "root": hex(append tree's root), "count": n}
verifier → prover   {"type": "appended", "first": j, "coin": hex(c_a)}      c_a 32 bytes, released after the root arrives
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
before its encode lands is *withdrawn*: its values' last structural reader (§8, "The pass and its step") is at or
before the step its write would be recorded at, so it gets no write, no free and no credit. It was never resident
encoded, and a block no step reads while it's resident is in no record, so the line its write would be in lists its
index in `withdrawn` instead (memory-map/v1's per-step list, optional and empty by default, so a map without withdrawals
reads as before). A line's writes and withdrawals take the store's next fresh indices together, and a withdrawn index is
never written. The prover's release covers the index and names it withdrawn (`Session.release(runs, withdrawn)`), and
the verifier checks it was never written: an appended block, inside the release's runs, that no plan received before
held. An append that is malformed or would take the store past its codec's `max_blocks` rejects with `append`, and a
release of a block outside the store or released already, or one naming withdrawn a block that is registered, outside
its runs or held by a plan, with `release`. Appends are a session's: the next session over the store starts again from
\(N\) (§8).

**No block appended after a pass's seed is in that pass's plan.** A pass's seed binds every append received before
its root (`prev_(t,i)` above), so every block its plan can hold, which must be appended before its root, was
committed before its seed. A block appended after the seed is in no plan of the pass, and its receipts are credited
in no pass whose seed came before its append. Without this the seed would depend only on \(\sigma_t\) and the root
before, which the prover knows before it sends that root, so a page appended mid-window could hold the pass's
sketches of other blocks (64 bytes a block at 16 rows, 64 blocks a page), and their receipts would follow from the
page's bytes without reading them, a red-team finding whose
test is `test_append_order.py`. A page that relayed any receipt of a pass whose seed binds it would be a fixed point
of SHA-512. The prover therefore sends a pass's appends before its root and computes the seed after they're sent
(`Store.head`, `Prover.run(..., head=...)`). A seed computed without an append the verifier received before the root
fails every receipt of the pass. While a session has no appends the seeds are unchanged, so the GPU passes, which
append nothing, derive them as before.

**The lifetime cap.** \(L\) here is the lifetime cap of §1's carrier term, not a pass's length \(L_{t,i}\). An
appended block joins the window of the next root received after its append, and no plan of a window \(L\) or more
after that one may hold it (`Params.lifetime`, a duration's windows, below). A plan that does rejects with `lifetime`. So every post-seed block leaves the credited store within \(L\) windows, and the stacked
stock the carrier term charges is the input rate times \(L\) (§1, §8). A block the prover keeps longer, a prefix
cache's for instance, is re-keyed: released, and its payload appended again at the store's next fresh index, which
draws a fresh coin and so gives it a fresh bound key (§3), so its encoding is computed anew. A re-key's free is the
one exception to a free at its values' last structural reader (§8, "The pass and its step"). The re-key's step frees
the old index, and the new index is written when its encode lands (deferred residency, above), so the value is
uncredited in between, and withdrawn if its last structural reader comes first. A re-key is only at the cap (§8,
"The memory map's choices"), so its time is the map's, not a choice made after the coin. Under deferred residency a
block appended in window \(w\), the one its append joined, is resident from \(w\) or \(w + 1\). Its re-key, recorded at
a step before the first pass of window \(w + L\), therefore falls at age \(L - 1\) at the latest. At \(L = 1\) that is
the window it was appended in, before it may be resident, and a block whose encode lands after the boundary is never
credited. So a session that defers residency has \(L \ge 2\) (`Params.deferred`; the verifier refuses one at \(L = 1\)).
The record names \(L\) (`lifetime`), and the remote session sends it with the session, so the prover re-keys in time.

\(L\) is a duration's whole windows: \(L = \max(L_{\min}, \lfloor D/T \rfloor)\) at \(D = 30\) s (`LIFETIME_NS`,
`lifetime_windows`), \(L_{\min}\) 2 for a session that defers residency and 1 otherwise. That is 1 window at the stated
60 s (2 deferred), 2 at 20 s deferred, and 30 at 1 s. The vLLM serving store defers residency and its hello says so
(`StoreBackend.deferred`), and its map re-keys a window ahead, so a serving session refuses \(L < 2\) (`Serving`): at
\(L = 1\) every line would re-key every live block. A served session at the stated window therefore has \(L = 2\). A
paced session's windows are its groups, and its \(L\) counts them at their nominal length, the stretch over its groups
(the verifier's; `Params` alone counts a group as the pace's stated window, the fewest the life allows): 353 for k = 1
at 85 ms over a 60 s stretch (706 groups), and 6 for k = 82 at 61 ms (12 groups). The duration is what the carrier term
needs. A carrier of a declared circuit's may work on a block from its append's coin until the block leaves, so what the
carriers may hold grows with the passes in \(L\) windows: `Accounting.Guarantees.DeclaredCircuitRequirement` charges
\(\lceil 1024 \cdot L c P N \cdot 4096 / E \rceil\) bits a pass for it, \(128\, L P c / E\) of the store, at the
property's cap \(c\) a certified byte. A count of 60 windows was an hour at the stated window, which put that share at
916% for 8B (\(c\) = 0.442, eight engines) and 190% for 70B (TP8, 0.0918), so the guarantee said nothing. Over 30 s of
61 ms passes the share is 7.6% and 1.6%, and the guarantee's ε a session 8.6% and 2.6%,
at a session whose passes a window are fixed (paced, \(P = k\)) and whose
windows draw at least 2,773 openings. No deployed KV-credited session is such a session yet: the gate's k = 1 groups
draw fewer. A window longer than \(D\) still holds a block for \(L_{\min}\) windows: at the stated 60 s, 8B's ε is 16%,
and 31.5% for a deferring prover such as the serving store (70B: 4.2% and 7.3%), so 10% at 8B needs windows or groups
of at most about 35 s (17 s deferred). The cost of the short life is re-keys at the cap, about 5 GPU-µs a block at k = 3. At the served models'
page lives, 30 s re-keys about 0.004 GPU a GPU at 8B and 0.05 at 70B.
Each pass still reads every live block, and the memory line, which credits registered blocks, doesn't read \(L\).
Counting the life from residency instead of from the append would allow \(L = 1\) under deferred residency, but it
would buy the carrier term nothing: a carrier's work starts at the append's coin, so the term charges the windows from
the append whenever the count starts, and since the prover chooses when a block becomes resident, a count from
residency would let it lengthen the life the term has to charge.

**Fiat–Shamir, and what grinding buys.** A seed after the first depends on the prover's own root, so the prover can
grind: re-salting a leaf gives a new root and a new next seed, at the cost of a leaf and its path and no read. A
chosen seed buys nothing against the block-sha512 receipt, because no seed makes a receipt cheaper to compute without
the block's bytes after the seed. The queries a grinding prover makes are counted by AuditRO's \(q\), at most
\(2^{64}\) a pass (`queries_log2`), whose \(C(q, k)\) λ charges. Other users of the seed must hold for a seed the prover chose: sketch-sha512's signs (below),
and under a registration that binds \(z\), PoUW's noise for the pass (§8).

**The two receipt kinds.** The registration names one (`receipts.KINDS`), and `Params.receipt` must name the same.
Every receipt of its sessions is of that kind, the weights' and the KV pages' alike (§8). sketch-sha512 is the default (`receipts.DEFAULT`) and the stated receipt: the receipt the guarantee is stated for.
block-sha512 is kept as a second kind, for comparison only; it is not the stated receipt.
The GPU kernels' sketch-sha512 receipts are held byte for byte to `receipts.SketchReceipter`
(`benchmarks/codec_tensor/sketch_receipt.py` lays its signs out for them), and the suite pins its outputs at both row counts.
Each kind's receipt is 64 bytes, so leaves, salts and the tree don't change. The difference is what is hashed after
the seed's block: sketch-sha512 hashes \(j\) and the sketch, one compression at 16 rows (64 bytes) and three at 64
(256 bytes); block-sha512 hashes \(j\) and the 4,096 bytes, 33 SHA-512 compressions. The sketch's signs come from the
pass's seed, so like the seed they exist only once the root before it does.

**The rows.** sketch-sha512's \(k_s\) is a registration parameter (`receipts.ROWS`: 16 or 64), bound in the digest
(§3) and in the signs' key (`"k"`), so 64-row signs are not 16-row signs and 48 more, and a session's `Params.rows`
must be the registration's. 16 is the default. 64 rows read the bound at \(r^* = 27\), \(2^{-215.85}\) a compression
(below), so no session is long enough to need a cap. On the device reader they cost τ = 1.226 a pass in one kernel
with an L2 hint, below 16 rows' 1.258. In
the fused GEMM they cost more: at 70B and B = 8, its step cost d is 3.49 / 4.00 at k = 2 / 3 against 16 rows'
2.40 / 3.18, since the two sketch warps bind there. So 16 stays the
default, and a registration names 64 to use it.

What each kind rests on:

- **block-sha512** needs only the random oracle. A correct receipt is a query at the block's bytes after the seed,
  so it bounds the reads by AuditRO (`Bandwidth.Guarantees.AuditRO`, applied per pass), with λ's slack. It assumes
  nothing about the device: the prover may hold any function of the store.
- **sketch-sha512** also rests on `RateAuditSketch` (proved). The random oracle takes a
  correct receipt to a correct \(\sigma_j\). `RateAuditSketch` takes a correct \(\sigma_j\) to the block's 128
  sectors of 32 bytes having been obtained, except for a guess. A query made without one of them is right with
  chance at most \(g/2^{256}\) per query, \(g\) the largest fiber of one sector under the signs. The signs bind
  that far except with \(\Pr_S[g < \text{fib}_S]\). Four conditions come with it:
  - **The store is held as its blocks** (`rate-sketch-verbatim`). In every tier the
    prover holds the store only as its blocks, any number of copies anywhere, and any device may compute from the
    blocks it reads, every such read counted. A layout from which sketches cost less to read than the blocks, such as
    a table of precomputed sketches, is outside it. That exclusion is needed: Lean shows such a table passes every
    check while reading 1/64 of each block at 16 rows (`RateSketchTable`), though it needs about \(2^{65536}\) times
    the store in memory. What Lean proves today is the narrower form, where each device obtains a sector only whole
    and as it is.
  - **Signs fresh per pass.** \(S_{(t,i)}\) comes from each pass's chained seed, never once per window (§8).
    The statement's window is this protocol's pass.
  - **Grinding.** The statement takes the signs uniform. Here they are derived from a seed the prover can choose
    (above), so the bound counts every key the prover sees, ground or not (below).
  - **The 16 rows.** At \(k_s = 16\) no \(16 \times 32\) block of signs is injective (Siegel's lemma). So
    `SketchBinds` doesn't apply, and the bound is read at a rank \(r\) of every sector's block of signs: \(r^* = 12\),
    where it costs \(2^{-95.12}\) a SHA-512 compression of the prover's (below). At \(k_s = 64\) the signs bind except
    for a column dependency (`RateAuditTensorSketch`), at 256 bytes a block, and \(r^* = 27\).

  Under the default the sketch is of \(E_j\) alone, which is `RateAuditSketch`'s digest of the tile,
  \(\text{dig}_S(e_t)\), so the statement applies as it is. Binding \(z\) is the registration's opt-in (`bind`), PoUW's
  join (the guarantee is conservation over one undifferentiated resource, with no identity to bind). Under it \(z\)
  enters before the sketch, \(e = E_j \oplus \text{spread}(z)\), as §8's third rule requires, and XOR with a fixed \(z\)
  permutes each sector, so each sector's fiber is the same. But `RateAuditSketch`'s digest is of the tile alone, so that
  form needs it restated over the block and \(z\) together, and it is not part of the stated guarantee. The verifier's
  registry (`RemoteAudit.Registry.register`, and `audit.Registry.register` beside it) refuses such a registration until
  its definition covers it.

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
whatever the prover did to reach it, unless two seeds' 256-bit sign keys collide, which shares their signs. A prover of
\(Q\) SHA-512 compressions over a session sees at most \(Q + P\) keys, \(P\) the session's passes, and makes at most
\(Q\) guesses. Read at rank \(r\), by the union bound over keys, guesses and the block's 128 sectors, and the keys'
birthday (`SketchHashRandom`'s ε), the session's failure is

\[
\delta_s(r) \;\le\; Q \cdot 2^{-8r} + (Q + P) \cdot 128 \cdot \Pr[\text{rank} \le r - 1] + (Q + P)^2 \cdot 2^{-257} ,
\]

(the receipt guessed without its hash, \((Q + P) N \cdot 2^{-512}\), aside), so its cost a compression is \(\rho(r) = 2^{-8r} + 128 \Pr[\text{rank} \le r - 1]\), and \(r^*\) minimizes it. At 16
rows \(r^* = 12\): the guess's \(2^{-96}\) beside \(128 \cdot 2^{-103.25} = 2^{-96.25}\), so \(\rho(r^*) =
2^{-95.12}\). At \(r = 11\), \(\rho = 2^{-88}\); at 13, \(2^{-71.30}\). At 64 rows \(r^* = 27\) and \(\rho(r^*) =
2^{-215.85}\). The birthday term doesn't move 16 rows' thresholds; at 64 rows it is what binds them.

| rows | \(r^*\) | \(\rho(r^*)\) | \(Q\) at \(\delta_s = 2^{-40}\) | \(Q\) at \(\delta_s = 2^{-64}\) |
| --- | --- | --- | --- | --- |
| 16 | 12 | \(2^{-95.12}\) | \(2^{55.12}\): 8.1 days of 8 GPUs | \(2^{31.12}\): 42 ms of 8 GPUs |
| 64 | 27 | \(2^{-215.85}\) | \(2^{108.5}\) | \(2^{96.5}\) |

The GPUs are RTX PRO 6000s at \(7.0 \cdot 10^9\) SHA-512 compressions a second each, block-sha512's measured rate
(below). \(Q\) counts every compression the prover makes, on any hardware: the grinding needs no access to the
store, so it can run off the node. The ledger charges \(Q = F \cdot C\): \(C\) the node's own compressions over the
session's span, at its GPUs' rate, and \(F\) the outside-compute factor the node states (`Node.sketch_outside`,
`timing.SKETCH_OUTSIDE` = \(2^{20}\) by default: \(2^{20}\) nodes' SHA-512, about 8.4M RTX PRO 6000s), which the
verifier records with δs (`sketch_failure`'s `outside`) and the auditor's line reads back (`verifier.sketch_at`).
\(F = 2^{20}\) is provisional, under the sketch's random-oracle ask. Each factor of 2 in \(F\) costs a bit: at 16 rows
a session stays below \(2^{-40}\) up to \(Q = 2^{55.12}\) compressions, 0.67 s of one node's hashing at \(F = 2^{20}\)
(8.1 days at \(F = 1\)). One node's 10-minute session reads \(2^{-30.19}\) (\(2^{-50.19}\) at \(F = 1\)), a node-day
\(2^{-23.02}\), and gate run 8's 227 s session \(2^{-31.59}\) (\(2^{-51.59}\) at \(F = 1\), \(2^{-41.59}\) at
\(2^{10}\)). The memory line barely feels it, since AuditRO's \(2^{-41}\) a pass dominates its δ: gate run 8's 2,118
passes put it at \(2^{-29.95}\) at \(F = 1\) and \(2^{-29.55}\) at \(2^{20}\), 0.4 bits. 64 rows brings δs under
\(2^{-50}\) at \(F = 2^{20}\) (\(2^{-127.13}\) for 10 minutes, the keys' birthday) at τ 1.226 a pass on the reader, the option if that
slack is ever cut; \(2^{-64}\) is out of reach at 16 rows and needs it too. The fused GEMM's cost at 64 rows isn't
measured yet.

A grinding try is not a pass's commitment. It reads no block: re-salting one receipt leaf changes the root without
changing any receipt. As the reference computes it, a try is 15 + 3⌈log₂ n⌉ compressions over \(n\) leaves
(`timing.grind_compressions`): the leaf's salt digest, the hm96 key's digest and its tree leaf (9), the leaf's hash
(2), its path (3 a level), the next seed (2) and its signs' key (2). That is 66 compressions, 9.4 ns on one GPU, for
an 8 GiB pass of 131,072 leaves, so a node of 8 GPUs affords about \(2^{29.7}\) tries a second, \(2^{35.6}\) in a
stated 60 s window. A prover that keeps the constant prefixes' midstates needs at least 5 compressions a try (the
root's last, and the seed's and the key's 2 each, whose first blocks hold the root and the seed): at most
\(2^{33.4}\) tries a second on that node, \(2^{39.3}\) in a window. Seeing one sector's \(16 \times 32\) signs takes 16 more, one stream block a row. The bound doesn't rest on
any of these counts: it charges a key to every compression.

None of this moves γ: \(\delta_s\) adds to the failure probability, not to the reads certified. The ledger adds
\(\delta_s\) to the memory line's δ and to γ's error (`ledger.sketch`), not to a term of γ. That δ is the profile's
δ, AuditRO's \(2^{-41}\) a pass and \(\delta_s\): over one node's 10-minute horizon at 50 passes a second,
\(2^{-40} + 2^{-26.13} + 2^{-30.19} = 2^{-26.04}\) (\(2^{-26.13}\) at \(F = 1\)). The stated claim at \(2^{-40}\) (§10) is
the openings' δ; the line's total is above it on the slack alone, whatever \(F\).

What each costs on the GPU: one RTX PRO 6000 (sm_120, 2,092 MHz), an 8 GiB pass of 2,097,152 blocks, best of
3. τ is a pass's time over one plain read of the same
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
fragment cut that `gemm_blocks` builds splits every 128-byte word across 4 vectors, so `plan_of` refuses
it. A kernel that keeps it needs a layout of its own, one whose every vector is still one whole registered block (a
fixed permutation of the block's bytes, which the verifier applies to the opened block before its sketch), named in
`LAYOUTS` and here before any of its receipts is credited. A permutation the fused GEMM
can produce is added as a named layout, with its check and vectors, once it is proposed. Until then the fused
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

Window \(t\)'s roots are final once the next window's first root has arrived, or \(D_t\) has passed, or, in a paced
session (§7, "Pacing"), once it holds its \(k\) roots, past which it takes none, and its boundary has left (so \(D_t\) is
set): so the challenger's signed log, where
the openings coin's request names the window's last root, fixes a paced window's end at that root. A window with
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
At the stated window's `SAMPLES` openings (\(\varepsilon_s = \ln(2^{40})/s\) = 0.083%) and δ = \(2^{-40}\), the count at
\(\delta/Q\) is \(\lceil s\,(40 + \log_2 Q)/40 \rceil\):

| grinding \(Q\) | \(s\) | × the coin's |
| --- | --- | --- |
| none (the coin) | `SAMPLES` | 1.00 |
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

The verifier checks each opening in order (`Session.verify`), after rejecting as `malformed` a window with no draws or
openings that aren't one a draw:

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
a 1,024-bit prime and 64 masks, where it was 128 mod a 512-bit prime. In the proof block (§8), where a product's cost
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
unit's verdict is on, §8), the receipts each draw requested (`samples`, the session's \(s\): a draw of \(U\) receipts
drew \(\min(s, U)\), and the memory line's δ charges the draw's rejection at \(s\) a draw, one draw a window unless
`groups_per_coin` draws several from one coin), and the session's span. The
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
every appended block is past the cap, the same for every pass of a window. `public()` lists the heads per window too
(`heads`), which the memory-map unit's verdict is on (§8), each a hiding commitment, a hash over hiding append roots
and their indices like `appends`. It lists no stale bound: a bound sums the counts of the appends each window
joined, which follow the workload and are neither protocol structure nor an output, so it is a value. The verifier
keeps it in its own record (`WindowRecord.stale`, `profile()`) and never publishes it, and the memory-map unit takes
it inside its proof, recomputing it from the committed appends (§8). The verifier receives each append's count in
the clear on the wire, and that count follows the workload: this is a known open item,
fixed before any KV credit by fixed-size padded appends on a
public cadence. The record holds every input of
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

### Pacing

A paced session (`Params.pace`, `Pace(k, window_ns)`) releases a live coin every \(k\) passes: each window is a group
of at most \(k\) passes under a challenge of its own, from the indexed challenger like every window's, with its own
openings coin, and the last, the closing window, follows at least one other. The stated window doesn't hold between a
paced session's windows (`Params.window_ns` is then the least time from a challenge to the next, 1 ns in the drivers):
`Pace.window_ns` is the stated \(T\) its γ is declared over ("Declared outputs"), and the lifetime cap \(L\) counts
groups (§4, "The lifetime cap"). Window \(t\)'s boundary message is
due once \(t\) has \(k - 1\) roots, so that the pass in flight is its last
(at \(k = 1\), once window \(t - 1\) has its root, and window 0's at once), and leaves at the latest at
\(\max(b_{t-1} + (k - 1) A, D_{t-1})\) (`Session.next_boundary_ns`). So the honest prover never waits on a coin, and a
window's roots are due within about \(k\) allowances of its challenge. A window's \((k+1)\)-th root is out of turn
(`order`).

That is two windows in flight, `Pace.in_flight` \(f = 2\), the default. At \(f = 1\) (`Pace(k, window_ns, 1)`, its
pace naming `in_flight`) window \(t\)'s boundary message is due once \(t\) has its \(k\) roots, at the latest at
\(\max(b_{t-1} + k A, D_{t-1})\), and is refused before any of them arrived; a root of \(t\) after it is out of turn
(`order`). So every coin leaves after the root before it, which the challenger's signed log shows
(`coins.released_reason`) and a record naming \(f = 1\) holds (`Record`), and the honest prover waits a round trip on
each coin. The Lean is `RemoteAudit.Flight`, which is the session at \(f = 2\) (`Flight.run_two`).

Each root's arrival on the verifier's clock is in the record (`WindowRecord.arrived_ns`), so the record bounds when
each group's reads were: after its challenge left at \(s_g = b_{g-1}\), and before
\(e_g = \max(\text{its last root's arrival}, s_{g+1})\) (the closing window's: its last arrival). The time-held line
reads that (`ledger.memory`, `Accounting.Guarantees.Remote.HeldOverTime`). A registered block absent from node memory
between two correct reads of it is absent for at most the groups between them, each of whose passes read it wrong, or
it came back by a rebuild, a disk read or a link. The credited span runs from \(e_0\) (before the first group's reads a
block may never have been there) to the closing window's challenge \(s_C\), whose reads back it. A block a pass leaves
out could be absent across that pass with neither a wrong read nor a rebuild, so the time-held forms need every pass to
read every registered block (`Covers`). A session one of whose passes leaves one out has the snapshot only.

### Declared outputs

A session's outputs are these, and only these (under the rule that a protocol publishes only
its structure, its outputs, the verifier's coins and draws, and hiding commitments). An accepted session's are
`ledger.declared`'s, from its record and the node's capacities (`ledger.Node`):

- **The verdict**, per window and for the session: accepted, or the failure that ended it (one of `FAILURES`, at the
  window it ended in).
- **γ**, per window (\(\gamma_t\)) and over the session, each against both \(R\): the node's specified read bandwidth,
  the headline, and its measured one (§1, "Which \(R\) is the headline"). The session's is read both ways, per window at
  \(W\delta\) and over the horizon at \(\delta\) (§10), each with its failure probability. γ is over the registered
  store only: a receipt of a block appended after the codec seed is drawn and opened like any other but not credited
  (`credit_post_seed` is None, so \(\mu_{\text{carrier}}\) is 0), and γ doesn't depend on what the workload appends.
  A paced session's coins leave as its roots arrive, so when they left is a value, and its γ per window is over the
  fixed windows of the stated \(T\) (`Pace.window_ns`) from its first challenge, the last ending at its last deadline
  (`ledger.windows`), protocol structure. A pass is in the fixed window that holds it whole, from its challenge's
  release to its root's arrival, and one that straddles a boundary is in neither (at \(k = 1\), whose coins leave a
  pass early, about two passes a boundary); the session's γ credits every pass. A window's post-seed reads
  \(\sum_i a_{t,i}\) follow from its γ: its timing term is \(1 - (U_t - \sum_i a_{t,i})\, b / (R \Sigma_t)\), and the
  others are functions of public counts (a fixed window's, of the passes it holds). Live KV is
  uncredited and doesn't enter the store, so no append is credited and they carry no workload. Revisit this when
  appends are credited.
- **The memory line**, a session total only, in three forms, each with its ε and δ: the registered bytes the session
  shows held in node memory are at least \((1 - \varepsilon) N b\), except with probability δ.
  - *Snapshot*: the registered blocks in node memory at the start of each credited pass, averaged over those passes.
  - *Time-held*: the registered blocks in node memory at each instant of the credited span, averaged over the span.
    Only a paced session whose every pass reads every registered block has one ("Pacing").
  - *Link-charged*: the time-held line with the node's boundary links charged as carrying blocks back into node
    memory at their ceiling, which drops Identity.

  Every form charges the blocks the node could re-encode from their payloads (its squaring ceiling at the record's
  codec) and read from its own disks (their measured read ceiling), each at its rate over any interval plus one burst
  (`ledger.Node`: a maximal rebuild batch, \(S_{\text{chip}}\), and a maximal disk read, a second at the ceiling;
  link-charged, one link window as well), and its on-chip storage \(S_{\text{chip}}\). δ is the horizon's error plus
  the coin commitment's hiding terms \(\sum_t \mathrm{hide}_t\), at most \(n \cdot 2^{-191}\) over \(n\) windows
  (`ledger.hiding`), plus the receipts' binding term \(\sum_t \sqrt{|L_t| s_t} \cdot q_t/2^{256.5}\) at the session's
  bounds (`ledger.binding`, §1; it takes the gap `binding-split` as closed),
  plus the sketch's (`timing.sketch_failure`) plus the draw's rejection \(\sum_t k_t / 2^{64}\) (`ledger.rejection`),
  at the receipts each draw requested, \(k_t\) the record's `samples`, not the \(\min(k_t, U_t)\) it drew: a draw
  that opens all its receipts is charged \(k_t\) too, as `LineHeld`'s error is. These are the ledger's parts
  (`ledger.DELTA_PARTS`), the horizon's profile and slack and then these four. `LineHeld`'s error also carries the
  windows' PRF advantages, \(\sum_t \mathrm{Adv}(\mathrm{prf}\,t)\), which have no numeric value: the ledger names
  them by their claim (prf/sha-512, `ledger.PRF_CLAIM`) and puts them in no δ.

  Every form, the link-charged one included, is conditional on `NoOutsideCompute`, statement C in full: no device
  outside the node computes on a registered block or a receipt during the session. Identity is only its links clause:
  a relay prover, a thin node forwarding each coin to a helper that holds the store and computes the receipts, sends no
  block over the links, so neither Identity nor the links' charge sees it, and only C excludes it. The line's statement
  takes C as a premise (`LineHeld`, `LineHeldIdentity`, beside `SketchBoundAt`, the receipts' sketch wherever
  computed, which `SketchBoundDeployed` gives at the sketch's δs on the line's trace read on the sketch outcomes and
  `Accounting.Guarantees.Remote.SketchBoundLine` carries to the line's coins), and the line of record says it is
  conditional on it (`auditor_line.json`'s `conditional_on`, `credit.md`; the `memory-line-of-record` label names the
  field's names, and `research data label` holds it to the file) until certifiers cover the gate node's boundary
  (`certifier-coverage`) and certify that nothing but the verifier's messages entered the node over the credited span.
  `auditor_line.py --boundary`, off by default, refuses a session's credit when the certifiers' records of the boundary
  links over its span show other ingress. `auditor_line.py --lockdown`, infra's record of the provider's lockdown of
  the gate node, grounds C for a session whose signed span lies inside an applied interval: the line is then
  conditional on `ProviderLockdown` (`trusted-host`), which names what the lockdown doesn't close, in C's place.

\(R\), \(S_{\text{chip}}\), the squaring ceiling, the disks' and links' ceilings and the bursts are the verifier's: the
node's stated or measured capacities, protocol structure, never the prover's report (§1).

Everything else that follows the workload is a value, not public: the blocks the session appended and released and
the chain over its appends, each pass's plan, its post-seed reads \(a_{t,i}\) (a window's sum follows from its γ,
above), its step and its root's arrival, and any
memory line per window or by kind (weights, KV, host). Each window's receipts \(U_t\) stay public, since its draws are
over them. Until `public()` is trimmed to match (it already keeps `stale` out), it still carries
`appended`, `released` and each window's `post_seed`.

## 8. What deployment changes, and what is open

- **Zero knowledge.** Every protocol is zero-knowledge, and an opened block decodes to a KV page,
  which is user data. §6's openings reveal it, so the clear checks are only the specification of what each unit
  proves. Deployed, each drawn receipt is one proof block that the proof service proves in zero knowledge, and the
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
  - *Output*: the unit's verdict, accepted or §6's failure name, within the 16-bit size class, the least public one. The record and
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
  prover-side registry's.
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
  \(O(\text{changes} \cdot \log R)\). Checked whole on one core, a plan of 245k runs takes about 230–340 ms as
  `Plan` on tuples, more than a 7–120 ms pass, and as `Plan.from_array` on the store's (R, 2) array about 55 ms in
  store order and 150 ms shuffled, with about 5 ms more for the digest. `Plan.from_array`
  reads the array through the buffer protocol and checks it column by column in the standard library, since a
  protocol imports no third-party package and a verifier no `verity.ml` code, and the suite holds it to `Plan`'s
  digests and refusals on the same rows. `Session.receive` takes any of the three.
- **What the plan shows.** v0's plans are in the clear, so the verifier sees which KV pages were live in each pass: the
  batch's shape and context lengths over time, not their contents. Hiding them goes with the zero-knowledge openings:
  the plan committed, its distinctness and each opening's \(j = \text{plan}[p]\) proved, only \(L\) public (the draws
  need it).
- **The pass and its step.** A pass covers one step \(s\) of the served circuit's execution, which it names with its
  root (§4; for vLLM, the engine step its Store backend tags each pass with). The step is the requirement's (the Lean
  module `Definitions.Accounting.Requirement` in `Security/`), and the pass's plan is to hold every complete block
  the circuit reads at \(s\), `blocks (creditedReads s) ⊆ plan` (`PlanMatches`), and may hold others (filler, a
  gathered table's unselected rows, a paused request's pages). The honest plan is the store resident when \(s\)
  begins (`Store.resident s`, which holds them under `StoreMatches`): the blocks
  registered or written at a step before \(s\), less those freed or overwritten at one. So a step's memory-map record
  takes effect after the step's reads: its writes are of blocks the step completed, or an earlier step did and the
  device has since encoded (deferred residency, §4), and its frees are of blocks whose values' last structural reader
  the step is. A writer therefore records a free there, not at the next write's.

  A value's *last structural reader* is the last step of the schedule that runs a gate with the value as an operand, a
  gather that may select it included, whichever request the gate is in, so a page requests share is freed once, after
  the last of them reads it. That is where the Lean module's `live` ends, so the resident set holds every block
  `StoreMatches` asks for (`requirement.Requirement.last` over the census, in Python). It reads the circuit, its
  schedule and the registration, never a committed value, so residency comes from the public schedule. A free at a
  value's last *use*, the last step whose gather selected it, would read residency off committed values.
  `StoreMatches` would fail for every candidate no later step selects, and one faulty proof block upstream of the selection
  would move the free. In a red-team counterexample,
  one flipped bit of a length unit keeps \(L - 1\) dead pages of \(L\) resident to the last
  step, \((L - 1)(S - 1) \cdot 32\) bits more, while the structural residency is the same in both runs. The
  structural free holds more instead: each candidate page stays resident to the last step that may select it, \(L\)
  times the honest run's use there with no faulty proof block. The prover does hold those pages, so the audit's own claim is
  sound, but they are memory the circuit may not need. That is why registered weights, which `W` must hold whatever
  the router or the tokens select, are creditable in every format, and KV only where no selection reads it. Each
  request's arrival step and executed extent are part of the schedule. They are host- or value-derived until the
  padded static circuit makes them public, so until then the extent rests on one unit (a red-team finding).

  The prover's audit side gives each pass its plan's change at its step (`Replay.take(s)`). It refuses a pass at a
  step whose record it has applied, or before the last pass's, and a record of a step before the last pass's
  (`memory_map.py`). The step
  is public structure, and the map and the plan are witness. Because the steps strictly increase, a step has at most
  one pass, so its reads are credited at most once; a step with no pass is credited nothing. Open:
  - *The plan's check.* That a pass's plan holds every complete block the circuit reads at its step is the plan's
    proof block's to check, inside zero knowledge, against the prover's memory map and the circuit's requirement
    (I4). The verifier here checks only that each pass names a step above the last.
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
- **The schedule check.** Credit after the seed waits on it (`schedule.py`, of record in
  `lean/RemoteAudit/Schedule.lean`, held by the difftest). Before the session's first coin the prover commits a cap
  \(a\) on its schedule's bits, with the session's parameters (`Params.schedule_cap`; none, the default, and nothing
  after the seed is credited). After the session it publishes the schedule: when each appended block is written and
  freed, which registered blocks are freed and when, and each pass's step. The verifier refuses a schedule over the
  cap, computes each pass's plan from it (`Schedule.plan`, the blocks resident when the step begins, in store order)
  and the session's appends (`Schedule.appendRuns`), and refuses the session if a pass's plan or an append differs
  (`Session.scheduled`, a `Scheduled` verdict carrying the cap; `Ledger.steering` is the cap). `ledger` credits
  nothing after the seed without a `Scheduled` of the same session. So a credited plan is the verifier's own function
  of a schedule of at most \(a\) bits, and a plan that follows a coin at that schedule is refused: the `a` of the
  stacked guarantee's `SteeredLayout` premise is the cap plus one (`Properties.RemoteAudit.steered_of_checked`, under
  `Properties.RemoteAudit.ScheduleLayoutGap`), never the coins' bits for free. The cap, not the published schedule's
  own bits, is what is credited: `SteeredLayout` takes one \(a\) at every coin, and a schedule's length can follow
  the coins. A prover that publishes a 200-bit schedule on most coins and a 300-bit one where it steers takes up to
  \(2^{200} + 2^{300}\) layouts, so only \(a = 301\) holds of it, and its 200-bit sessions credited at 200 would
  understate `DeclaredCircuitRequirement`'s \(\delta\), which grows as \(2^a\), by \(2^{100}\) (red team).
  Under a cap of 300 both are credited at 300; under one of 200 its 300-bit schedules are refused, and those
  sessions credit nothing after the seed. Open: the schedule is witness (a
  request's arrivals and lengths), in the clear here as v0's plans are, so deployed it is committed and the check is
  a proof block whose output is the bits; the bits are the whole session's schedule, an upper bound on what the prover
  chose after a pass's coins; the cap travels with the session's parameters, and `benchmarks/remote_session`'s hello
  carries none, so its sessions credit nothing after the seed; and A7's sessions send their circuit's read sets as
  plans, not the resident sets, so
  the check refuses them and they are credited nothing after the seed until they send the replay's plans.
- **The requirement's link.** What the audit credits is the node's reads of its store. Counting them as the served
  circuit's reads needs A1's link (`StoreMatches`, `PlanMatches`), and three
  of its gaps are on the audit's side. `ledger` takes the link as `requirement`, a `RequirementLink`: whether its
  proofs hold, the circuit's read bits from complete blocks at each step \(r_s\), and the memory-map unit's verdict
  on the session's appends and its passes' heads (`mapped`, below). On it, a pass's credit is the circuit's read bits
  at its step, less a block for each planned block it credits no receipt of (`linked_bits`, by `RequirementMoved`
  below), and the ledger says it is linked (`Ledger.linked`); without it, the credit is the store's, flagged as
  such. The checks behind the link belong to the
  units named below, and the ledger takes them as given. Those units are sampled, so the link carries what their draw
  established: \(\varepsilon\), the most share of the linked credit the units it leaves unproved can carry (the
  passes whose plan unit is wrong and those holding a block a wrong memory-map unit wrote), except with
  probability \(\delta\). The session is credited \((1 - \varepsilon)\sum_\text{passes} \max(0, r_s - u_s B)\),
  \(u_s\) the pass's planned blocks it credits no receipt of (0 when its receipts cover its plan), a window
  its own sum less \(\varepsilon\) of the session's (the unproved share may all be one window's), and \(\delta\) adds
  to the ledger's error (`Ledger.error`, `error_horizon`). That \(\varepsilon\) bounds the unproved credit, not only
  the share of units drawn wrong, is the link's to establish: a faulty proof block at a heavy step carries more than its
  share.
  - *Reads the circuit doesn't make.* The plan is the resident set, and the circuit's reads at a step can be fewer: a
    gather's unselected rows, requests absent from the step's batch, unrouted experts, filler. `PlanMatches` is
    one-sided: the plan holds every complete block the circuit reads at the step, and the blocks beyond those carry
    none of its reads, so a pass is credited the circuit's reads at its step (`read_bits`, its `creditedReadBits`:
    the bits of `requirement.credited_reads`), never its receipts. That the plan holds every complete block read at
    its step is the plan unit's to prove (I4: PlanMatch, which proves the plan is exactly the blocks the step reads,
    or PlanShard, which proves the inclusion over a registered store), against the memory map and the requirement,
    inside zero knowledge. PlanShard at GAPS > 0 lets a shard's plan leave blocks out and makes each shard's count
    public, so how many of a shard's blocks each plan holds; at GAPS = 0 every count is its shard's size, and a plan
    is the whole store.
  - *Partly filled blocks.* A block holding fewer than \(B\) bits of the circuit's reads (a request's last block,
    values of different lifetimes sharing a block, a table block with one row selected, filler) would overstate the
    credit if its receipt counted whole, so the credit counts reads, not receipts (`linked_bits`).
    `Accounting.Guarantees.RequirementMoved`: under `PlanMatches`, a pass that moved the planned blocks \(M\) moved
    every bit the circuit reads from complete blocks at its step but at most \(B\) for each planned block outside
    \(M\). So a pass whose credited receipts cover its plan is credited \(r_s\), and one that credits no receipt of
    \(u_s\) of its planned blocks (post-seed ones, without the property's verdict) \(\max(0, r_s - u_s B)\). No
    planned block need carry any of the circuit's bits, and a wrong receipt, a planned block outside \(M\), is what
    the sampling term charges a whole block for. `PlanMatches` packs the reads into the planned blocks (`Packs`), so
    \(r_s\) is at most \(L B\), and the ledger refuses a link reading more.
  - *Registered blocks past their last structural reader.* memory-map/v1 frees a registered block like a written one:
    a free of a slot below \(N\) releases it, and no later plan may hold it (`Session.release`). The writer records
    the free at its values' last structural reader ("The pass and its step"), so a gathered table, an embedding or
    the experts a router indexes, stays resident until the last step whose gather may select any of its rows,
    whichever rows the run selected, and its residency reads no token or routing decision. That each free is at its
    values' last structural reader is the memory-map unit's to prove (I4: `MemoryMapStep` holds each registered
    block's free to `requirement_units.registered_frees`, from the public schedule and the registered layout). Until
    it does, a dense decoder's weights, which every step reads, need no free, and the census must show it.
  - *The memory map's choices* (a red-team finding). Between a
    pass's seed and its reads, the writers could choose which pages to free, when to re-key a block, the order of a
    batch of appends, and dummy appends that shift every later \(j\). Each choice moves the bound keys or the live
    set, and none of them costs an encode beyond the writes. Freeing 100 of \(10^4\) live pages is 803 bits of
    choice, past what the advice bound tolerates. Even with the map fixed, a writer keeps choices the map doesn't
    read (padding and dead-value bits, run inputs, schedules), and the stacked sketch bound goes vacuous past about
    \(41 - \log_2(\text{passes})\) bits of them. The append coin (§3, §4)
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
       withdrawn one being a block it completed whose values' last structural reader came at or before its encode
       landed (§4), so there are no dummy appends;
    3. a re-key (a release and the payload appended again, §4's lifetime cap) is the map's, at the cap, of the same
       payload, so \(d_j\) is unchanged and only its index, its coin and its key are fresh, its free at the
       re-key's step and its new index's write when that encode lands; at the cap means its old index is below the
       stale bound of the first pass after its step (`WindowRecord.stale`, which the unit recomputes from the
       committed appends, §7), so the old block may be in no plan from that pass on, and no earlier re-key is taken:
       which blocks are re-keyed or freed stays witness;
    4. every free but a re-key's is at its values' last structural reader, of a registered block or a written one
       ("The pass and its step"), a step of the public schedule that the unit's public row names (its `last`, which
       is `requirement.Requirement.last`), never a step read off the circuit's values;
    5. the map is the one the plan unit's plans are of (`PlanMatches` reads the same map).

    The unit's public input is the record's `appends`, the chain \(a\) over the session's appends (§4, §7), and each
    pass's step and head (`WindowRecord.heads`, the chain its seed binds). The units chain their own appends step by
    step, and the link holds only if every pass's head is that chain through the appends of the steps below the pass's
    step (item 1) and the whole chain is `appends`, so the appends' places among the passes are bound; each map
    instance recomputes the stale bound of the pass after its step from those committed appends, \(N\) and \(L\) (a
    value, never public: §7), and refuses moving a value whose old index isn't below it (item 3). A prover that runs
    serving ahead (`Ahead`, "The pass and its step" above) still meets item 1: its steps without a pass have their
    appends sent between the passes around them, since it hands serving's records over, and sends their appends, only
    before the next pass's root. Its witness is the map, the appends' extents, digests \(d_j\) and salts, and the
    plans. The releases stay
    witness, held to the map through the plans; the verifier here sees them in the clear and keeps released blocks
    out of later plans, a check deployment moves into the plan unit. `ledger` takes the unit's verdict as
    `RequirementLink.mapped(record)` and refuses a link whose verdict isn't true of the record's chain and heads:
    without it the credit is the store's, as without a link. The proof is proofs'; the audit side is the chain, the
    heads, the stale bounds it keeps and the hook.
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
    is a free, recorded in the line of the step whose forward ran last, which bounds its values' last structural
    reader from above and is that reader for a request that ran at that step and whose attention reads every row it
    holds, so the store's filler plan leaves the released blocks out), and the
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
    block's contribution a fixed digest: read once and kept, it serves every later pass's receipt (red team).
  - The matrix must resist a seed the prover chose, since it can grind the seed through its previous root.
  - Under a registration that binds \(z\), \(z\) must enter before the sketch, because a linear sketch of
    \(E \oplus Z\) splits.

  The reference's sketch-sha512 meets all three. Its signs come from each pass's seed. For the second it counts every
  key the prover sees, and reads the bound at \(r^* = 12\) (§4). Under a registration that binds \(z\), \(z\) is
  XORed into the block before the sketch. In the proof block, check 5 under sketch-sha512 is 16 signed sums over the
  block's 4,096 bytes and one compression, in place of block-sha512's 33 compressions.
- **KV receipts are the driver's receipts.** A KV page's receipt is the registration's one receipt kind (sketch-sha512:
  `SketchReceipter` under the pass's seed, its signs under `SIGN_DOMAIN`, hashed under `SKETCH_RECEIPT_DOMAIN`) of a
  registered KV block under the registration's block layout, committed in the pass's one tree and listed in its one
  plan with the weights'. A KV block is a registered block like any other: 4,096 bytes of the encoded store, 32 whole
  codec words, \(j = \text{slot} \cdot \text{LAYERS} + \text{layer}\) (§3), and its kernel's vectors pass `plan_of`
  (§4). The device's KV receipt as it stands (word_codec's `kv_receipt.py` and `kv.cu`'s `rc_sha8_kernel`)
  is a third kind:
  its own SHA-512 domain (`verity/word-codec/kv-receipt/sha512/v0`), ro-sketch-blake3 signs, and half-block vectors
  that take their 4,096 bytes from 16 tokens' head shares, zero-padded to whole blocks. The verifier can open
  none of it, so none of it is credited: no pass's plan holds a KV block until its receipts are the registration's kind
  over the registered KV layout, and a step that commits KV receipts of another kind keeps them out of its plan and
  its credited bytes.
- **The challenger's state.** The challenger keeps every coin of a session until it releases it and never releases one
  twice. Persisting that across challenger restarts, and committing coins that outlive one session, is out of scope.
- **Keeping receipts for the draws.** Any receipt of a window may be drawn once its roots are final, so the prover
  keeps them, and the leaves' salts: 64 bytes a 4 KiB block and 192 a leaf of 16, about 1/60 of what it read in the
  window. Keeping each tree above its leaves and re-reading an opened leaf's 16 blocks instead costs a small fraction of
  the reads again (about 0.06–0.1%). The reference keeps every pass whole.
- **Lean.** The statement is `Accounting.Guarantees.RemoteSessionRunConservation` (and `RemoteWindowRunConservation`,
  each with a `Relay` form): `Accounting.SecurityProofs.Remote.Conservation` at the per-pass routes proved for
  it. The openings are
  `Accounting.Guarantees.RateWindowDraw`, a window's pooled draw over its passes' receipts, and `RateSessionDraw`,
  the horizon and every window at once, both from `HorizonSampling`, the causal horizon bound. They take windows of
  one size, \(M = P \cdot N\), and draws with replacement. The record's windows differ in \(U_t\), and its draws are
  distinct, which only lowers the chance that they all pass. `pooled` applies the same argument at each window's own
  size: `Accounting.Guarantees.Remote.PooledDraw` states it on the deployed draw (`Accounting.Remote.Draw`: `Key.subset`
  on the first SHA-512 blocks a window's coin gives, released after its roots, the sampler exact on each draw's first
  word and its fallbacks free), for windows the prover sizes after the earlier windows' coins, each below
  \(2^{448}\) receipts (the verifier's `RECEIPT_LIMIT`, \(2^{64}\), keeps them there), at δ plus
  \(\sum_t s_t/2^{64}\) (a rejected first word) plus each window's PRF advantage, C3's term, against one
  distinguisher a window that the session fixes (`Session.distinguisher`: it runs the session through window \(t\)
  at the coin commitment with leaf \(t\) committing zeros and reads at most \(s_t\) first blocks of the stream it
  is given), plus the commitment's hiding terms, hm96's, at most \(n \cdot 2^{-191}\) over \(n\) windows
  (`Committed.hide`, `ledger.hiding`): the prover holds the root of every coin from the start, so a window is
  fixed before its coin only once that root is (`Live` of `Committed.view`). Plus the receipts' binding term
  (`Receipts.binding`, `ledger.binding`): `Refused` lets an opened receipt through only correct or moved off the row
  its root holds (`Receipts.held`, the plurality of the accepted rows it shows over its window's continuations from
  the cut before its coin, the same at every continuation: `Binding.held_cont`, `tally_le_held`), and a move is two
  accepted openings of one root and rank with different rows, a SHA-512 collision among their checks
  (`Hash.Guarantees.RowBinding`, proved), charged under `cr/sha-512` through the fork form: `ReceiptsBindingOfCR`
  bounds it by \(\sum_t \sqrt{|L_t| s_t (q_t^2/2^{513} + \mathrm{split}_t)}\), two runs from window \(t\)'s cut on
  independent openings coins, with \(L_t\) a set holding every receipt window \(t\) opens at every outcome, \(s_t\) the
  session's draw cap and \(\mathrm{split}_t\) the chance that the two runs' roots differ. `ledger.binding` charges it
  at the session's bounds (§1) and at \(\mathrm{split}_t = 0\) (`Binding.binding_le_of_fixed`), which holds where the
  cut fixes the roots; the deployed prover may compute them from the coin commitment, so that is the gap
  `binding-split` (below).
  `pooled` is computed in exact rationals in the verifier's Lean
  (`lean/RemoteAudit/Pooled.lean`, `pooled.py` held to it by a difftest), the budget \(\bar d \cdot S\) from above by a
  certified rational root, and `Properties.RemoteAudit.PooledCovers` proves its W at least the bound `PooledDraw`
  reads (the greedy by \(U_t/s_t\) is the fractional knapsack's optimum); the memory line reads that W from the
  record's windows (`Properties.RemoteAudit.LineCoversSession`). What this code supplies is the schedule (each
  window's challenge released at \(b_{t-1}\), its roots due by \(D_t\)), the chain, the plan's distinctness, one tile
  a receipt under a registration that binds \(z\), the sketch's bound read at \(r^*\) (`timing.sketch_reading`), and
  the profile at δ.
  The chain's ordering is a random-oracle argument over a prover of bounded queries, which the statements must say.
- **The gap `binding-split`.** The verifier sends the prover the coin commitment in its `session` message, before the
  first coin, and each challenge coin with its leaf and path (`coins.coin_reason`, `prover.py`). The commitment's root
  is a function of every coin, window \(t\)'s openings coin among them, so the prover may compute window \(t\)'s roots
  from it, and two runs from the cut on independent openings coins may root the window differently:
  \(\mathrm{split}_t\) has no bound, and hiding doesn't give one (the two runs share the salts, and the move is read
  after the coin is released). `ledger.binding` takes \(\mathrm{split}_t = 0\), and every line says it is conditional
  on it (`auditor_line.py`'s `CONDITIONAL_ON`). The change that closes it: the verifier sends the prover no commitment
  and no coin's leaf or path until the session ends, only the challenge and append coins' values and each window's
  draws, and at
  `end` it sends the commitment and every released coin's opening, which the prover checks then (`layout_reason`,
  `coin_reason`) and refuses the session's record over. The auditor's check of the coins against the record's
  commitment is unchanged; what moves is the prover's own check that no coin was chosen after a root, which becomes
  after the fact. Window \(t\)'s roots, ranks and the receipts' verifier then read only the cut before its openings
  coin (the challenge coins, the earlier windows' openings, the append coins and the prover's own randomness), so the
  two runs root it alike, \(\mathrm{split}_t = 0\), and `Binding.binding_le_of_fixed`'s hypothesis holds. It touches
  `verifier.py`'s `session` and `coin` messages, `prover.py`'s coin check, `wire.py`, the Lean model of the play
  (`Committed.play`, which gives the prover the commitment from the start) and a statement that the play fixes each
  window's receipts by its cut, which is what closes the gap. The same view may also let `PooledDraw` drop the
  hiding terms, which is for its statement to show.
- **The binding's Lean.** The store binding's decision (`store.py`, §3) is ported to `lean/RemoteAudit/Store.lean`
  and held to the Python by a difftest (`lean/tests/test_binding.py`), refusals compared word for word: the pin from
  its file's bytes (`pinOf`, `modelPin`), `checkpoint_sha512`, the payload salts, the expected payload root from the
  weights blocks' digests (`expectedRoot`) and `Binding.hold`'s decision (`hold`). Reading the files stays Python only:
  the pin's bytes, every pinned file's SHA-512 (`check`), the safetensors headers and config.json, the arena's layout
  from them (`tensors`, `arena`) and the weights blocks' digests (`weight_digests`). The verifier's Lean process
  decides `bound`: it takes the pin's bytes and role and the weights digests at startup (`bind`: `modelPin`, then the
  store's shape) and decides each hello (`hold`), `Binding` beside it as the reference, a disagreement refusing the
  hello as the registry's does; the gate line credits `bound` only where the Lean held the root. The Lean's root is
  per subtree of 2^16 leaves, the subtrees on `--layout-workers` threads, as `expected_root`'s are over processes.

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

**Openings** at the stated \(T\) = 60 s and δ = \(2^{-40}\): \(s\) = `SAMPLES` per window (555 a second,
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

The stated claim is that hybrid: short by under 0.15% over a 10-minute horizon (for \(\mu_{\text{sample}}\),
below) and by under 1% in every window (for the whole \(\gamma_t\)), at \(2^{-40}\), with the window stated at
\(T\) = 60 s and `SAMPLES` openings. For \(\mu_{\text{sample}}\) it
needs \(d \ge 2{,}773\) openings a window and \(d \cdot n \ge 18{,}470\) over the horizon's \(n\) windows. The stated
window's `SAMPLES` give each window a floor of 0.083%, and the horizon's 10 windows \(\ln(2^{40})/(d\,n)\) = 0.0083%.
The 20 s option's 11,091 give each window 0.25%, and over its 30 windows the same 0.0083%. The 1 s option's 2,773
give each window 1%, and over its 600 windows 0.0017%.

`ledger` reports μ_sample both ways: `sample`, the per-window reading, and `horizon`, the pooled one (`pooled`: the
most \(\sum_t f_t U_t\) the bounds above allow), with `floor`, the largest \(f_t\) they allow, and γ for each
(`gamma`, `gamma_horizon`). It reports each window's the same way over its own span (`Ledger.windows`: each
\(f_t\), from `floors`, is its `floor`, and \(f_t U_t\) its `horizon`), and the worst window's γ either way beside the
session's (`worst`, `worst_horizon`). All of it starts from each window's profile and span, which the record keeps.

**The whole γ.** `timing.stated` reads the claim through `ledger`: an honest 10-minute session on one node of 8 RTX
PRO 6000s and their host, at the node's measured \(R\) = 12.92 TB/s (and at spec, 13.60 TB/s) and \(S_{\text{chip}}\)
= 3,105,652,736 B, under the default codec at the node's stated squaring ceiling. Over every tier, the store is the
same GB of each GPU's HBM and host DRAM's share (`timing.spanning`), and a pass is its read at \(R\). In HBM alone, it
is the same GB a GPU and nothing on host DRAM, and a pass is its read at the GPUs' 12.31 TB/s.
\(\mu_{\text{chip}}\) is about \(S_{\text{chip}}\) over the store, and a window's \(\mu_{\text{timing}}\) about
\((\text{RTT} + 2 t_{\text{pass}})/T\), so the store's size trades one against the other. Among the grid's stores 94 GB
a GPU, 789 GB over every tier with 61 ms passes, is best except at \(T\) = 1 s. The worst window's terms and its
\(\gamma_t\), at the measured \(R\) and at spec, and over the horizon's \(W\) windows the pooled
\(\mu_{\text{sample}}\) and the whole γ, at RTT 1 and 20 ms:

| \(T\) | openings a window | store, GB a GPU | \(\mu_{\text{timing}}\) | \(\mu_{\text{sample}}\) | \(\mu_{\text{chip}}\) | λ | \(\mu_{\text{rebuild}}\) | worst \(\gamma_t\) | at spec | \(W\) | pooled \(\mu_{\text{sample}}\) | horizon γ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 60 s | `SAMPLES` | 789 GB, 94, every tier | 0.24–0.27% | 0.083% | 0.39% | 0.115% | 0.065% | 0.896–0.927% | 5.827–5.856% | 10 | 0.0083% | 0.668–0.671% |
| 60 s | `SAMPLES` | 520 GB, 62, every tier | 0.23–0.26% | 0.083% | 0.595% | 0.117% | 0.065% | 1.088–1.119% | 6.009–6.038% | 10 | 0.0083% | 0.911–0.914% |
| 60 s | `SAMPLES` | 252 GB, 30, every tier | 0.29–0.33% | 0.083% | 1.23% | 0.120% | 0.065% | 1.792–1.823% | 6.678–6.708% | 10 | 0.0083% | 1.680–1.684% |
| 60 s | `SAMPLES` | 752 GB, 94, HBM | 4.93–4.96% | 0.079% | 0.39% | 0.110% | 0.065% | 5.573–5.602% | 10.271–10.299% | 10 | 0.0079% | 5.355–5.358% |
| 20 s | 11,091 | 789 GB, 94, every tier | 0.45–0.54% | 0.25% | 0.39% | 0.115% | 0.065% | 1.267–1.360% | 6.179–6.268% | 30 | 0.0083% | 0.668–0.671% |
| 20 s | 11,091 | 520 GB, 62, every tier | 0.50–0.59% | 0.25% | 0.59% | 0.117% | 0.065% | 1.522–1.616% | 6.422–6.511% | 30 | 0.0083% | 0.911–0.914% |
| 20 s | 11,091 | 252 GB, 30, every tier | 0.40–0.49% | 0.25% | 1.23% | 0.120% | 0.065% | 2.057–2.150% | 6.930–7.019% | 30 | 0.0083% | 1.680–1.684% |
| 1 s | 2,773 | 252 GB, 30, every tier | 2.60–4.38% | 0.95–0.97% | 1.18–1.20% | 0.115–0.118% | 0.065% | 4.948–6.687% | 9.677–11.329% | 600 | 0.0017% | 1.674–1.677% |
| 1 s | 2,773 | 789 GB, 94, every tier | 8.00–9.62% | 0.90–0.92% | 0.356–0.362% | 0.105–0.106% | 0.065% | 9.448–11.040% | 13.953–15.466% | 600 | 0.0017% | 0.661–0.664% |
| 90 s | 11,091 | 789 GB, 94, every tier | 0.17–0.19% | 0.25% | 0.39% | 0.116% | 0.065% | 0.994–1.015% | 5.920–5.940% | 6 | 0.042% | 0.707–0.710% |

At the stated \(T\) = 60 s, then, every window is under 1% for the whole \(\gamma_t\) at the measured \(R\):
0.896–0.927% with 94 GB a GPU over every tier, and 1.09–1.12% at 62 GB a GPU. At spec it is 5.8% at best, the
measured kernels' shortfall from the datasheets. The same 94 GB a GPU in HBM alone is 5.57–5.60%, host DRAM's idle
4.7% of \(R\) in \(\mu_{\text{timing}}\). At the 20 s option, every window under 1% holds for \(\mu_{\text{sample}}\)
(0.25%) and not for the whole \(\gamma_t\), which is 1.27% at best. At \(T\) = 90 s with 11,091 a window it is
0.994–1.015%. The 1 s option's per-window claim is \(\mu_{\text{sample}}\) alone. Its whole \(\gamma_t\) is 5–11%,
nearly all \(\mu_{\text{timing}}\): the window's idle ends, which the record states exactly.

**The store floor.** No \(T\) brings every window under 1% on too small a store. As \(T\) grows, a window's
\(\mu_{\text{sample}}\) (at 555 openings a second) and its ends vanish, and its \(\gamma_t\) tends to
\((g + (S_{\text{chip}} + \Lambda)/R + \mu_{\text{rebuild}}(t_{\text{pass}} + g))/(t_{\text{pass}} + g)\),
where \(t_{\text{pass}}\) is the store's read at \(R\) and \(\Lambda\) a pass's `named_bits` and `slack_log2` in bytes,
whose share a block depends on the store's size (`timing.store_floor` solves for it). Below 380.5 GB over every tier
(45.3 GB a GPU), \(\mu_{\text{chip}}\), λ and \(\mu_{\text{rebuild}}\) alone exceed 1%. With the chain's steady state
at \(g\) = 50 µs, no \(T\) meets 1% below 458.4 GB (54.6 GB a GPU). The suite checks the closed form against `ledger`
on a 20-minute window, 2% either side of it.

**The horizon's 0.15%** is stated for its pooled \(\mu_{\text{sample}}\), with the whole γ reported beside it. The pooled
\(\mu_{\text{sample}}\) is 0.0083% at every stated and optional \(T\), and the whole γ 0.67% at best.
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
  its hash and share of the tree are \(1/m\). The device reader's τ of 1.26 a pass scales to about 1.09 at 16 KiB
  (from its measured parts; the whole isn't measured). The openings pay instead. Each carries \(m\) blocks: 20.9 KB at
  16 KiB with a payload leaf a receipt, or 27.0 KB with PoUS's 4 KiB leaf. That is 2.4–3.0 times today's, or
  0.70–0.90 GB a window and 93–120 Mb/s per node. Each opening's proof block also decodes \(m\) blocks (check 3).
  sketch-sha512's rate a compression goes from \(2^{-95.12}\) to \(2^{-93.88}\) at 16 KiB's 512 sectors.
  `RateAuditSketch` would need restating for a receipt of \(m\) blocks, and the registration would name \(m\).
- **Fewer queries.** At 4 KiB, 0.05% needs \(q \le 2^{42.5}\) queries to a pass's oracle:
  874 s of one GPU's SHA-512, 109 s of the node's. No prover hardware is trusted, so the protocol can't state such a
  bound.

\(\mu_{\text{chip}}\) is \(S_{\text{chip}}\) over a pass's reads, and a pass reads each block at most once. So it
falls only with more store read at \(R\): 6.21 TB over every tier for 0.05%, 8.1 times the node's 768 GB of HBM. At
94 GB a GPU and host DRAM's share it is 0.394%. Host DRAM beyond its share doesn't substitute: at its 606.8 GB/s, a
pass longer than the HBM's 61 ms read leaves the HBM idle. At 200 ms, \(\mu_{\text{chip}}\) is 0.120% and 66% of the
capacity is idle.

What would lower \(\mu_{\text{chip}}\) instead is a flush. Each pass's root would key a computation that needs the
chip's whole storage, and its output would go into the next seed, so the chip's state carries nothing of the store
into the next pass. That needs three things:

- **An assumption.** A bandwidth-hard function on \(S_{\text{chip}}\): evaluating it while holding \(X\) bytes of
  the store on chip costs at least \(X\) bytes more of device-memory traffic, with \(R\) bounding reads and writes
  together. Spilling the held bytes instead costs \(2X\).
- **The verifier's work.** It recomputes the flush for every pass whose seed it uses, which at `SAMPLES` openings is
  nearly every pass.
- **The GPU's time.** At least one write and one read of \(S_{\text{chip}}\): 26–105 µs at 5–20 TB/s on chip
  (unmeasured). That adds 0.04–0.17% to the steady state.

At the stated \(T\) = 60 s with 789 GB over every tier and RTT 1 ms, the horizon's whole γ is 0.67% (0.75% bounded
by \(q^k\)). 16 KiB receipts make it 0.58%, and a 26 µs flush with them 0.23%. The worst window's \(\gamma_t\) goes
from 0.896% to 0.81% and 0.46%. With λ and \(\mu_{\text{chip}}\) at 0 the horizon's γ is 0.16%, because the chain's
steady state and \(\mu_{\text{rebuild}}\) alone exceed 0.15%.

## 11. The P2 store

`p2.py` runs the audit over PoUS's P2-encoded weights store: the store is PoUS's `C*`, `W` encoded under P2
(`pous.schemes.p2`, `p2-16448/v4`) at PoUS's setup, and the receipts are made by the GPU's own decode of it while it
serves. This section is what differs from §1–§8; the rest (seeds, plans, leaves, trees, windows, deadlines, coins,
draws) is unchanged, over units where those sections say blocks. Its byte formats are pinned by `tests/test_p2.py`
and held to the Lean by `lean/tests/test_p2.py`.

**The store.** Block \(j\) is 2,056 bytes, a big-endian integer below \(p = 2^{16448} - 21065\), at byte \(2056j\);
\(j = \text{blocks\_per\_segment} \cdot s + v\). A unit is four consecutive blocks, unit \(g\) blocks \(4g \ldots 4g +
3\): 8,224 bytes, 257 whole 32-byte sectors. The store's \(U = \lfloor N/4 \rfloor\) units are what plans range over,
and the last \(N - 4U\) blocks are in no unit and no receipt certifies them (none in PoUS's store of \(2^{16}\)-block
segments). The receipt's vector is a unit, not a block, because block \(j\) starts at byte \(8j \bmod 32\): a
receipt of one block would leave sectors half read, and `SketchBoundSession`'s guess term, over sectors, would grow
from \(Q/2^{96}\) to \(Q/2^{64}\).

**Registration** (`p2.Registration`, `p2.register`). PoUS's setup: the prover commits to `W` (`setup.commit`), the
verifier draws the salt from its own secret after the commitment (`setup.draw_salt`), and builds `vk` from the `C*` it
decodes to the committed `W` (`setup.verifier_key_from_encoding`). The registration is the node, P2's parameters, the
commitment, the salt, `vk`'s root and \(k_s\); its digest is `p2.py`'s table, and it is the session's
\(\text{root}_{(0,-1)}\) as in §4. The setup of record is commitment `9aaaf192…` and its salt.

**The decode kernel's sketch.** This is the interface the kernel computes, and `p2.Sketcher` is its CPU reference.
Per pass \(i\), after root \(i-1\):

~~~text
seed_i     the audit's (receipts.seed): SHA512(SEED_DOMAIN ‖ σ_t ‖ u32be i ‖ root_(t,i−1)), 64 bytes
S_i        k_s rows (16) of 1,028 bytes, from seed_i on the host: K = derive(seed_i,
           "verity/remote-bandwidth/sketch-signs/v0", {"tag": "sketch-sha512", "k": k_s, "n": 8224}), row c =
           K.stream(("row", c), 1028) (verity.ml.randomness); bit x of a row (byte x // 8, bit x % 8, LSB first) is
           the sign of the unit's byte x: 1 → −1, 0 → +1
~~~

Per block \(j\) the kernel decodes, from the 2,056 bytes it read of `C*`, before decoding them:

~~~text
b          j mod 4, the block's place in its unit
signs      bytes 257·b … 257·b + 256 of each row: bit x of that slice is block byte x's sign
π_j[c]     Σ_(x < 2056) sign_c(x) · int8(c_j[x]), c_j's bytes as stored (byte 0 the integer's most significant),
           int8 two's complement; k_s int32, |π_j[c]| ≤ 128·2056
~~~

Per unit \(g\), once its four blocks are decoded (in any order):

~~~text
σ_g[c]     π_(4g)[c] + π_(4g+1)[c] + π_(4g+2)[c] + π_(4g+3)[c], exact in int32 (|σ_g[c]| ≤ 128·8224 < 2^21)
y_g        SHA512(Z(SKETCH_RECEIPT_DOMAIN ‖ frame(seed_i)) ‖ u64be g ‖ σ_g as k_s int32 little-endian):
           one compression after the pass's 128-byte prefix, whose midstate every unit shares
~~~

The host commits the \(y_g\) in plan order as §4's tree (`p2.Prover.commit`). The partial is a function of the bytes
the decode read, so the receipt binds the read the decode made; computing it from the decoded `W`, or from a second
read, is another protocol. Per block it is \(16 \times 2056\) sign-and-add steps beside the decode's two 16,448-bit
squarings. The fixture kernels build against comes from the serving loop over the toy `C*` (`serve.py fixture`,
pinned there by its SHA-512):
a pinned seed, the sign rows, every block's \(\pi\), every unit's \(\sigma\) and \(y\).

**Openings** (`p2.Opening`, `p2.Session.verify`). An opening of unit \(g\) is its four blocks, each with its path to
`vk`'s root, and the receipt leaf with its salt and path. The verifier checks the leaf on the pass's root, each block on
`vk` under PoUS's leaf (`pous.verifier.leaf_hash`, `payload-path` if one fails; in the Lean, PoUS's own leaf and
path check, `PousCheck.Merkle`), and \(y_g\) recomputed from the four blocks (`p2.check`). It decodes nothing: `vk`'s leaves are of the `C*` the verifier decoded to the committed `W`, so a block on its
path is `C*`'s block, by SHA-512's collision resistance. A P2 store is fixed at setup, so a session over it appends
nothing (`append`).

**What it rests on, and what is open.**

- The record (`Session.record`) is §7's, with `block_bytes` 8,224, `codec` `p2-16448/v4` and layout `p2-unit4/v0`.
  `audit.Rebuild` refuses a P2 codec, so `audit.ledger` refuses a P2 record: \(\mu_{\text{rebuild}}\) at P2's encode
  waits on `WorkHardnessP2` and `SquaringCeilingP2`, drafted (not decided) in Accounting's assumptions, which need the
  H200's squaring rate at 16,448 bits measured.
- The sketch bound at units is `SketchBoundSession` at \(m = 257\) sectors, drafted as `SketchBoundP2`:
  \(\delta_s = Q/2^{96} + (Q + P)(\varepsilon + N/2^{512}) + (Q + P)^2/2^{257}\) at `SketchRank 16 257 12 ε`, with
  \(\varepsilon = 257 \cdot\) `timing.sign_rank_at_most(16, 11)` \(= 2^{-95.2491}\) (\(2^{-96.2548}\) at 4,096-byte
  blocks' 128 sectors). The memory line's `SketchBoundAt` follows from it by `SketchBoundLine`, which holds at every
  \(m\).
- Not yet built: the Lean session over a P2 store (the Lean ports here are the registration digest, the partials,
  the unit receipt and the opening check, `lean/RemoteAudit/P2.lean`), the registry's one live registration a node for
  P2, and the zero-knowledge opening unit of a P2 unit (§8).

## Appendix A. Live coins, and why the protocol doesn't use them

The alternative: a live coin a pass and no chain. The challenger commits \(K\) pass coins and \(W\) openings coins.
Pass \(k\)'s coin \(c_k\) leaves at \(r_k \ge r_0 + k\Delta\), its seed is \(\text{SHA512}(D \,‖\, c_k \,‖\,
\text{u64be } k)\), reading no root, and its root is due by \(D_k = r_k + A\). It is what the rule of live coins
only would require. `timing.live_mu` and `timing.live_best_interval` keep its closed forms, and the tests check them.

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
for every node audited. That rate is impractical. The chain needs one challenge and one openings
coin a window.
