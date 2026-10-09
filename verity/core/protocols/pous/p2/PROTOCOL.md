# P2 specification v4

> v4 is v1.1 with one hash: SHA-512, with its whole 64-byte output, everywhere (§13), as Daniel ruled on 2026-10-06 at
> 15:07Z. v1.1 is frozen v1 (`art:a7a29e84eef9d2584482b7f3e9722af6b579e2984a9633352564c562ce9e2021`, relayed in
> `note:20261004T1836Z-report-relay-p2-spec-v1-frozen` from `store:pous/docs/efficient-crypto/p2-spec-draft.md`,
> sha256 `3f49c34bb065af9c11b92935d1460e6f3caf7b12caf6d89560d5c8a7e2725267`) with the RTT allowance capped at
> 354 µs, as Daniel ruled on 2026-10-04 (§12). Relative and `/cursor/stores/…` links point into that store, which is
> also in `art:8bd64630bc06c23d5095996d2f198ce4bd557413722ad94bd3ca3c1a949e42e9`. v4 supersedes v1.1 and the package's
> `p2-16448/v3`; nothing in it is optional, and no hash in it is a parameter.

Status: specification v4, 2026-10-06. It keeps v1.1's mathematics, sizes, k, Δ, RTT cap and framing, and changes only
the hash (§13). This is not deployment approval: P2 remains experimental because the P2–M1-SGI instantiation assumption is
not accepted (default, Daniel deferred, 2026-09-29). Its challenge derivation is `verity.ml.randomness` v2
(Appendix A), and the vectors in §7 are generated under it. The errata of 2026-10-07 (§14) clarify v4 as they do v1.1, without changing meaning.

## 1. Parameter set

The frozen deployment parameter set is:

| Item | Value |
|---|---|
| Protocol version | `p2/v4` |
| Width | w = 16,448 bits = 2,056 bytes |
| Prime identifier | `p2-p16448-c21065` |
| Modulus | p = 2^16448 − 21065 |
| Prime congruence | p mod 8 = 7, hence p mod 4 = 3 |
| Primality certificate | ECPP, 339 steps; three independent verifications passed |
| Payload profile | `p2-2054B`: 2,054 bytes = 16,432 bits |
| Deployment store | B = 2^19 blocks, 8 segments of 65,536 blocks |
| Exact-domain abstract family | 2^8 through 2^23 blocks; pinned in the store's `lean/pous` trusted layer |
| Hash | SHA-512 (FIPS 180-4), its whole 64-byte output, for every digest; §4's table |

The congruence certificate is direct integer arithmetic: 2^16448 is 0 mod 8 and 21065 is 1 mod 8, so p is 7 mod 8.

The prime is certified by a 339-step Atkin–Morain/FastECPP certificate produced by Andreas Enge's CM 0.4.4 `ecpp-mpi`, completed 28 Sep 2026 at 22:01Z. The PARI/GP certificate is `internal/efficient-crypto/p2-prime/cm-cert-p16448-c21065`, SHA-256 `6a3f6f1170295adcf73a49733da02e42ff6d96e73bacce666855cc9a5ef1d3c5`. Its Primo-format copy is `cm-cert-p16448-c21065.primo`, SHA-256 `a759215b2ecb294b8516b4069c72bc1f031cd8409c1ceed4833b6dcd9c762244`.

Three independent verifications pass: PARI/GP 2.15.4 `primecertisvalid` returned 1 and confirmed the certificate's first N is exactly `2^16448 − 21065`; CM's `ecpp-check` accepted it; and the campaign's standard-Python-plus-gmpy2 `ecpp_verify.py`, sharing no PARI or CM code, accepted all 339 steps against that target. Logs are `pari-verify-cm.log`, `cm-ecpp.log` and `mine-verify-cm.log` under `internal/efficient-crypto/p2-prime/`. The maintained certificate record is `docs/efficient-crypto/p2-prime-certificate.md`. V1 keeps p without requiring a Lean-checked primality proof; the alternative p′ is unnecessary.

Reduced-width bounty parameters use the same deterministic rule: choose the least positive c0 congruent to 1 mod 4 for which `2^w − c0` passes the stated primality procedure. The draft ladder gives c0 values 189, 17, 173, 237, 189, 317, 569 and 105 at widths 64, 96, 128, 192, 256, 384, 512 and 1024.

## 2. Canonical values, packing and accounting

All field values are canonical integers in `[0,p)`. A w-bit integer is encoded as exactly `w/8` bytes, unsigned, big-endian, including leading zero bytes. Bit number 0 is the least significant bit of that integer. XOR acts on the w-bit binary representations, bit by bit; it is not byte-string concatenation or field addition.

The sole normative deployment profile is `p2-2054B`, with b = 16,432 = 2,054 bytes (default, Daniel deferred, 2026-09-29). Profile names are case-sensitive. The former `p2-16447b` continuous-bit profile is non-normative research material and is not accepted by a deployment manifest.

For `p2-2054B`, `0 ≤ m < 2^16432 < 2^16447 < p`. No sentinel is embedded in m. For a manifest declaring B blocks, W is the concatenation of the exactly 2,054-byte, leading-zero-padded big-endian encodings of `m_0,…,m_(B−1)`, in segment-major then local-block order. Thus W has exactly `2,054·B` bytes. This is both continuous most-significant-bit-first packing and one byte-aligned payload per block; there are no w-bit codeword slots in W.

The manifest commits the profile and B, so the W byte length is determined without a sentinel or padding convention. The field/codeword wire encoding remains the fixed-width format above; implementations reject noncanonical field encodings, wrong widths and decoded values outside the declared payload width.

The exact-domain Lean model `m1p B w p b` treats b as a parameter. The store's `lean/pous` trusted layer pins `P2MeetsM1pB19Uncond`, `P2MeetsM1pFamilyUncond` and `P2MeetsM1pW8192Uncond` for every `20·16448 ≤ 21·b` and `b ≤ 16447`, including b = 16,432. Their repository sync follows PRs #162/#183.

The retained-state threshold is measured against the code, not the payload:

```text
S = floor((18/19) · |C|) = floor((18/19) · B · 16448) bits.
```

It does not change with b. At `p2-2054B`, the 16-bit code expansion is still part of `|C|`.

## 3. Maps

For canonical z in `[0,p)`, define:

```text
d(z) = z² mod p                 when z is even
       (−z²) mod p              when z is odd
```

Since p is 3 mod 4, exactly one of y and −y is a quadratic residue for nonzero y. Define `ρ = d⁻¹`:

1. For y = 0, return 0.
2. Compute `s = y^((p+1)/4) mod p` once.
3. If `s² = y`, return the even member of `{s,p−s}`. If `s² = −y mod p`, return the odd member. Any other result is an error.

For canonical r and full-width mask K in `[0,2^w)`, define the involution:

```text
x = r XOR K
σ_K(r) = x    if x < p
         r    otherwise
```

The rejection branch is mandatory. It must not reduce x modulo p.

For block j with public tweak `t_j ∈ [0,p)` and public mask `K_j ∈ [0,2^w)`:

```text
E_j(m) = ρ(σ_Kj(ρ((m + t_j) mod p)))
D_j(c) = (d(σ_Kj(d(c))) − t_j) mod p
```

`E_j` performs two roots and `D_j` performs two signed squarings. Both consume and return canonical values. `D_j(E_j(m)) = m`.

## 4. Setup and key derivation

### Hashes and domain tags

Every digest in v4 is SHA-512 (FIPS 180-4) with its whole 64-byte output; nothing is truncated except a derived key,
which `verity.ml.randomness` makes 32 bytes (Appendix A). `\0` is one zero byte. `frame` is
`verity.ml.randomness.frame`: `tag || u64be(len(body)) || body`, with tag `b` and the bytes for a byte string, `s`
and its UTF-8 for a string, and `i` and the minimal big-endian bytes (at least one) for a nonnegative integer.

| Use | Bytes hashed (or derivation) | Output |
|---|---|---:|
| W commitment | `SHA-512("verity/tagged-sha512/v1\0" ‖ u32be(27) ‖ "verity-pous/w-commitment/v2" ‖ u64be(len(payload)) ‖ payload)` | 64 |
| Setup salt | `derive(source, "verity/pous/p2-setup/v4", {model, prime, protocol}).value` | 32 |
| Key chunk | `SHA-512(base ‖ P ‖ u32be(i))`, base starting `"verity/pous/p2/keys/v4\0"`, P ∈ {`K`, `t`} | 64 |
| Row digest R | `SHA-512(P(8n) ‖ block)`, `P(b) = ("verity/sha512-row/v2\0" ‖ u64be(b))` zero-padded to 128 bytes | 64 |
| Leaf | `SHA-512(0x00 ‖ setup_salt[32] ‖ u64be(s) ‖ u16be(v) ‖ R[64])` | 64 |
| Node | `SHA-512(0x01 ‖ left[64] ‖ right[64])` | 64 |
| Epoch id | `SHA-512("verity/pous/p2/epoch/v4\0" ‖ u32be(len(manifest)) ‖ manifest ‖ w_commitment ‖ setup_salt ‖ u64be(B) ‖ root)` | 64 |
| Audit id | `derive(source, "verity/pous/p2-audit-id/v4", {counter, epoch}).stream("id", 32)` | 32 |
| Challenge key (mode 0) | `derive(source, "verity/pous/p2-audit-challenge/v4", {audit, epoch, root, salt})` | key |
| Round key (mode 1) | `derive(beacon_j, "verity/pous/p2-audit-round/v4", {audit, epoch, root, round})` | key |
| Transcript hash | `SHA-512("verity/pous/p2/transcript/v4\0" ‖ frames)` | 64 |
| Record frame hashes | `SHA-512(exact frame)` | 64 |
| `derive`, `stream`, `uniform` | `verity.ml.randomness` v2 (Appendix A) | 32-byte key; 64-byte stream blocks |
| Wire envelope | magic `"P2AU"`, version byte `0x04` (`p2-audit-wire/v4`) | — |
| Vector entry hash (§7) | `SHA-512(canonical_json_bytes(entry without "sha512"))` | 64 |

The four derivation domains name P2 (`p2-…`) so that they never collide with the generic protocol's
`verity/pous/audit-challenge/v1`. §13 lists each v1 name and its v4 replacement.

### Setup

The normative setup assumption is **P2-EXP-IO** (default, Daniel deferred, 2026-09-29): after W is committed, setup supplies fresh XOF streams independently for every `(segment, block, purpose)`, independently of the complete pre-setup transcript. Purpose is K or t. Under this idealization, masks are mutually independent uniform w-bit strings and rejection-sampled tweaks are mutually independent uniform elements of `Z_p`; its expander error is zero by definition.

Concrete counter-mode SHA-512 replacing P2-EXP-IO is a separate named assumption: **SHA-512-CTR/P2-EXP-IO**. A deterministic globally fixed SHA-512 function does not become a fresh independent-output oracle merely because its input contains a post-commitment salt.

A standard secure PRG is provably insufficient here. Proposition 1 of `internal/efficient-crypto/attacks/p2-seed-expander.md` constructs an ordinary secure locally seekable PRG whose public seed describes every P2 ciphertext for W = 0. Ordinary PRG security therefore cannot replace P2-EXP-IO.

Setup order is load-bearing:

1. Fix W and its exact model manifest. Commit to the bytes, payload bit length, model identifier, protocol version and prime identifier.
2. Only after that commitment is immutable, obtain the first valid unpredictable and unbiasable verifier/beacon source. Bind it; neither party may grind alternatives.
3. Draw one 256-bit setup salt through `verity.ml.randomness` v2 (Appendix A). The formula below shows the deployment prime id; at any reduced width it is replaced by that parameter set's `p2-p{w}-c{c0}`:

```text
setup_salt_256 =
  derive(
    source,
    "verity/pous/p2-setup/v4",
    {
      "model": model_commitment_bytes,      64 bytes
      "prime": "p2-p16448-c21065",
      "protocol": "p2/v4"
    }
  ).value                                   32 bytes
```

`source` is the verifier secret or beacon round selected after the W commitment, never a prover-selected commitment.

4. For each block, form the injective base below. Both integers are unsigned 64-bit big-endian. If global index `i = s·blocks_per_segment + v`, `segment = s` and `local_block = v`: key derivation uses the local within-segment index, never global i.

```text
prefix_body =
     ASCII("verity/pous/p2/keys/v4\0")      23 bytes
  || frame("p2/v4")                         14 bytes
  || frame("p2-p16448-c21065")              25 bytes
  || frame(model_commitment_bytes)          73 bytes  (a 64-byte digest)
  || frame(setup_salt_256)                  41 bytes  (176 bytes in all at the deployment prime)
prefix = prefix_body || 0x00 * (−len(prefix_body) mod 128)            256 bytes: whole 128-byte blocks
base   = prefix || u64be(segment) || u64be(local_block)              272 bytes

chunk(P, i) = SHA-512(base || ASCII(P) || u32be(i))                   P ∈ {"K", "t"}, i = 0, 1, 2, …
stream_P    = chunk(P, 0) || chunk(P, 1) || chunk(P, 2) || …
```

Let n = w/8. K is the first n bytes of `stream_K`, interpreted big-endian; n = 2,056 at deployment, so K is the first 2,056 of the 33 chunks' 2,112 bytes. For t, read consecutive n-byte candidates of `stream_t` (candidate a is bytes `[a·n, (a+1)·n)`) and accept the first integer below p. This is exact rejection sampling; modulo reduction and “subtract p once” are forbidden. The derivation of t is v1's; only the stream under it changed.

The base is injective in `(protocol, prime, model, salt, segment, block)`: the tag is fixed, each frame carries its own length, the zero padding is determined by the frames' total length, and the two integers have fixed width. The purpose byte and the counter follow at fixed offsets. Because the prefix is whole 128-byte blocks, SHA-512's chaining value after it is one midstate per setup, shared by every block and purpose. Each chunk's message is `272 + 1 + 4 = 277` bytes; past the prefix it is the 21 bytes `u64be(segment) ‖ u64be(local_block) ‖ P ‖ u32be(i)`, SHA-512's `0x80` byte, zero bytes and the 16-byte length: one 128-byte block. So each 64-byte chunk is exactly one compression after the midstate, and a deployment block's K and first t candidate cost 66 compressions. At a reduced width the prime id is shorter and the prefix is still padded to the next multiple of 128 bytes (256 at every width in §7).

Every `(segment,block,purpose)` is distinct. Masks must be full-width uniform outputs. Sparse, repeated, truncated, density-conditioned, shared or linear-schedule masks are nonconforming. Setup salts cannot be reused across model commitments. Keys are public, but deployment must expand them where used rather than store a key-sized table that violates the space bound.

5. Encode C and bind the audit verification root. Derive audit challenges only after that root is fixed.

Counter-mode SHA-512 with this v4 framing is the conforming realization of the separate **SHA-512-CTR/P2-EXP-IO** assumption (F4 as amended by v4, §13). Its cost on the deployment device is measured in §11; v1's SHAKE256 measurement does not carry over. ChaCha8 would add a weaker reduced-round assumption and is nonconforming.

## Audit protocol and wire formats

This section defines `p2-audit-wire/v4`. Multi-byte integers are unsigned big-endian. Byte strings have exactly the stated length. Receivers reject unknown versions, unknown types, reserved flag bits, noncanonical integers, length mismatches and trailing bytes. Every digest field is 64 bytes; the setup salt, the audit id and the round nonce are 32.

### Parties and transport

The **verifier** commits setup coins, issues each challenge, authenticates responses and decides the audit. The **prover**, acting through its **responder**, stores C and returns challenged encoded blocks.

The transport product is not prescribed. It must provide an ordered, loss-detecting, integrity-protected sequence of delimited frames with one authenticated peer in each role. Each delivery event supplies exactly one frame and its byte boundary; the wire verifier does not infer boundaries by reading the body's declared length from an undelimited stream. The transport must not reorder, duplicate or silently alter frames. The verifier must be able to measure the transport round trip used for the audit's RTT allowance. Confidentiality is not required: setup data, challenges, blocks and paths are public.

Every frame is:

```text
offset  bytes  field
0       4      ASCII "P2AU"
4       1      wire version = 0x04
5       1      message type
6       4      body length, u32be
10      n      body
```

The body length excludes the 10-byte header. `p2-audit-wire/v4` accepts only named profiles with w ≤ 16,448, canonical manifests of at most 512 bytes, `blocks_per_segment ≤ 2^16`, `segments < 2^64`, and k ≤ 105. Deployment setup accepts only `p2-2054B`; reduced-width conformance vectors may use a named `p2-nb` or `p2-nB` test profile. A registered scheme's profile is `p2-{payload_bytes}B` at every width; `p2-nb` profiles exist only in reduced-width conformance vectors, and no registered scheme produces them. Let n = w/8, B be the manifest block count, h(i,B) be the exact RFC 9162 sibling count, and r be the number of FINAL records. Accepted body lengths are:

| Type | Code | Accepted body length |
|---|---|---:|
| `SETUP_COMMIT` | `0x01` | exactly `68 + manifest_length`, with `manifest_length ≤ 512` |
| `SETUP_SALT` | `0x02` | 96 |
| `SETUP_BLOCK` | `0x03` | `108 + n` |
| `SETUP_ROOT` | `0x04` | 169 |
| `AUDIT_OPEN` | `0x10` | 124 |
| `CHALLENGE` | `0x11` | 138 |
| `RESPONSE` | `0x12` | exactly `112 + n + 64·h(i,B)` |
| `FINAL` | `0x13` | exactly `182 + 147·r`, with `0 ≤ r ≤ k ≤ 105` |

Thus every accepted profile has an absolute body cap of 15,617 bytes, attained by a 105-record `FINAL` (a RESPONSE is at most `112 + 2,056 + 64·80 = 7,288` bytes, since B < 2^80). For the deployment profile at B = 2^19, a RESPONSE body is `112 + 2,056 + 19·64 = 3,384` bytes, 3,394 framed, of which the authentication path is 1,216. A declared or delivered body outside its type-specific length is malformed code 1, even when below 15,617 bytes.

Fields at their frame offsets (body offset + 10):

| Frame | Fields, `offset:length` |
|---|---|
| `SETUP_COMMIT` | `manifest_length 10:4`, `manifest 14:ml`, `w_commitment (14+ml):64` |
| `SETUP_SALT` | `w_commitment 10:64`, `setup_salt 74:32` |
| `SETUP_BLOCK` | `w_commitment 10:64`, `setup_salt 74:32`, `global_index 106:8`, `block_length 114:4`, `block 118:n` |
| `SETUP_ROOT` | `w_commitment 10:64`, `setup_salt 74:32`, `block_count 106:8`, `verification_root 114:64`, `setup_mode 178:1` |
| `AUDIT_OPEN` | `epoch_id 10:64`, `audit_id 74:32`, `challenge_mode 106:1`, `k 107:2`, `delta_ns 109:8`, `rtt_allowance_ns 117:8`, `verification_flags 125:1`, `audit_counter 126:8` |
| `CHALLENGE` | `epoch_id 10:64`, `audit_id 74:32`, `round 106:2`, `index 108:8`, `round_nonce 116:32` |
| `RESPONSE` | `epoch_id 10:64`, `audit_id 74:32`, `round 106:2`, `index 108:8`, `block_length 116:4`, `block 120:n`, `path_count (120+n):2`, `siblings (122+n):64·path_count` |
| `FINAL` | `epoch_id 10:64`, `audit_id 74:32`, `verdict 106:1`, `failure_code 107:1`, `rounds_completed 108:2`, `delta_ns 110:8`, `rtt_allowance_ns 118:8`, `transcript_hash 126:64`, `record_count 190:2`, `records 192:147·r` |

### Setup commitment, salt and verification root

The setup manifest is canonical UTF-8 JSON using `verity.ml.commitments.identity.canonical_json_bytes`: sorted object keys, no insignificant whitespace, direct Unicode, no floats. It contains exactly `scheme`, `w`, `c0`, `profile`, `blocks_per_segment` and `segments`; `scheme` is exactly the JSON string `"p2"`, `profile` is the case-sensitive profile string, and the other four values are JSON integers. The W commitment is `verity.ml.commitments.identity.tagged_sha512` under the tag `verity-pous/w-commitment/v2`:

```text
manifest = canonical_json_bytes(params.manifest())
payload  = u64be(len(manifest)) || manifest || W
w_commitment =
  SHA-512(
    ASCII("verity/tagged-sha512/v1\0")
    || u32be(27) || ASCII("verity-pous/w-commitment/v2")
    || u64be(len(payload))
    || payload
  )                                                    64 bytes
```

W uses exactly the `p2-2054B` packing in §2: segment-major then local-block-major, with each payload in one 2,054-byte leading-zero-padded big-endian encoding. Reduced-width conformance profiles use continuous fixed-b-bit packing, most-significant bit first, with only zero right-padding in the final byte. `SETUP_COMMIT` (`type=0x01`, prover to verifier) has:

```text
u32be(manifest_length) || manifest || w_commitment[64]
```

Only after accepting this frame does the verifier derive the 32-byte setup salt as specified in §4. `SETUP_SALT` (`0x02`, verifier to prover) is:

```text
w_commitment[64] || setup_salt_256[32]
```

For `p2-audit-wire/v4`, `model_commitment_bytes` in §4 is exactly `w_commitment`.

In public-encoder setup, the prover then sends every encoded block in strictly increasing global index using `SETUP_BLOCK` (`0x03`):

```text
w_commitment[64] || setup_salt_256[32] || u64be(global_index)
|| u32be(block_length) || block[block_length]
```

At w = 16,448, `block_length` is 2,056. The verifier rejects gaps, duplicates or out-of-order indices, decodes the stream, checks the reconstructed W commitment, and constructs the verification root. In private-encoder setup the verifier already has W and constructs the same C and root itself; no `SETUP_BLOCK` frames are required.

For global block index `i`, let:

```text
s = floor(i / blocks_per_segment)
v = i mod blocks_per_segment
P(b) = ASCII("verity/sha512-row/v2\0") || u64be(b), zero-padded to 128 bytes
R(block_i) = SHA-512(P(8·n) || block_i[n])                                     64 bytes
leaf(i) =
  SHA-512(
    0x00 || setup_salt_256[32] || u64be(s) || u16be(v)
    || R(block_i)[64]
  )                                                                           107 bytes hashed
node(left,right) = SHA-512(0x01 || left[64] || right[64])
```

R is core's `sha512/row/v2` digest of the block as a row of 8n bits, `verity.ml.commitments.rowleaf.sha512_row_v2_digest(block, 8·n)`; at n = 2,056 it is 18 compressions, the first of them constant. The tree is the RFC 6962 shape: for n > 1 leaves, split at the largest power of two strictly below n; preserve global block order. One leaf's root is the leaf itself. The verification root is exactly 64 bytes.

**Why this leaf.** The leaf hashes the block's row digest, inside a wrapper that binds the setup, the segment and the index. R is the digest of the block that core's row rule already defines and that `answer_circuit`'s `PousRowDigest` proves, so a circuit that opens or decodes a block, and a verifier that recomputes a leaf, hash each block once, as one row, and the leaf adds one compression to R's 18: the wrapper is 107 bytes. The wrapper keeps what a bare row digest would lose. Its `0x00`, against the nodes' `0x01`, keeps a leaf from ever being a node (RFC 6962 §2.1). The setup salt makes a leaf of one setup useless in another, even for equal blocks. `(s, v)` lets a verifier that keeps leaves check one without its path, although an RFC 9162 path already fixes the index. Binding needs only `cr/sha-512`: two blocks with one leaf give a SHA-512 collision, in the wrapper or in R.

`SETUP_ROOT` (`0x04`, prover to verifier) is:

```text
w_commitment[64] || setup_salt_256[32] || u64be(block_count)
|| verification_root[64] || setup_mode[1]
```

`setup_mode` is 0 for private encoder and 1 for public encoder. The verifier accepts the root only if it equals the root it constructed through the selected MVP setup path.

Both parties compute:

```text
epoch_id =
  SHA-512(
    ASCII("verity/pous/p2/epoch/v4\0")
    || u32be(len(manifest)) || manifest
    || w_commitment[64] || setup_salt_256[32]
    || u64be(block_count) || verification_root[64]
  )                                                    64 bytes
```

An epoch id is never reused with a different tuple.

Given a recoverable `AUDIT_OPEN`, a setup failure emits a zero-record `FINAL` whose transcript contains only that open frame. Setup failure codes are:

- code 1 for malformed envelopes or manifests, wrong fixed lengths or setup mode, a noncanonical setup block `≥ p`, nonzero W padding, or another structural/canonicality error;
- code 2 for a `SETUP_SALT`, `SETUP_BLOCK` or `SETUP_ROOT` commitment/salt binding mismatch, a `SETUP_ROOT.block_count` that disagrees with `segments·blocks_per_segment`, or a missing, duplicated, reordered or wrong-index setup block;
- code 3 when decoded blocks reconstruct a different W commitment, or the correctly framed leaves reconstruct a different verification root;
- code 5 when a setup block decodes outside the selected payload profile;
- code 6 only when the transport fails before delivering a required complete setup frame.

The same 1, 2, 3, 5 precedence applies if one setup transcript has multiple faults. A setup transcript that ends before `SETUP_ROOT` is transport code 6 unless a frame already read gave code 1 or 2, which it reports instead. When a recorded transcript contains a valid fixed-size `AUDIT_OPEN`, setup transport failure code 6 is recorded as a zero-record `FINAL`: verdict 0, `rounds_completed = record_count = 0`, epoch id, audit id, Δ and RTT copied from `AUDIT_OPEN`, and `transcript_hash = SHA-512("verity/pous/p2/transcript/v4\0" || exact AUDIT_OPEN frame)`. No partial setup bytes enter that hash. If no valid fixed-size `AUDIT_OPEN` is available, no interoperable `FINAL` identity exists and the session fails without a `FINAL`; the transport/evidence layer records the setup failure outside this wire format.

### Audit and challenge derivation

`AUDIT_OPEN` (`0x10`, verifier to prover) is:

```text
epoch_id[64] || audit_id[32] || challenge_mode[1] || u16be(k)
|| u64be(delta_ns) || u64be(rtt_allowance_ns)
|| verification_flags[1] || u64be(audit_counter)
```

Wire conformance permits `1 ≤ k ≤ 105`; k = 0 and k > 105 are malformed code 1 and produce a zero-record `FINAL`. A deployment audit sets exactly k = 105; smaller positive values exist only for bounded test vectors. Challenge mode 0 is a designated-verifier secret key; mode 1 is a roundwise beacon. Verification flag bit 0 requests canonical decode checking and bit 1 requests comparison with verifier-held payload. Bit 1 requires bit 0; setting bit 1 alone is malformed code 1. All other bits are reserved and zero.

For mode 0, `source` is a verifier secret never sent to the prover:

```text
id_key = derive(
  source, "verity/pous/p2-audit-id/v4",
  {"counter": audit_counter, "epoch": epoch_id}
)
audit_id = id_key.stream("id", 32)
audit_key = derive(
  source, "verity/pous/p2-audit-challenge/v4",
  {
    "audit": audit_id,
    "epoch": epoch_id,
    "root": verification_root,
    "salt": setup_salt_256
  }
)
i_j = audit_key.uniform(block_count, j)
round_nonce_j = audit_key.stream(("nonce", j), 32)
```

Thus every index is uniform in `[0,B)`, with replacement; j is the integer round index passed to the pinned `verity.ml.randomness` sampler. A fresh audit counter gives a fresh audit id and challenge coordinate. The verifier persists used `(epoch_id,audit_id)` pairs across audits and process restarts and rejects reuse before releasing round 0.

The `verity.ml.randomness` argument types are normative: `counter`, `round` and sampler index j are nonnegative integer parts, `"id"` is a string stream index, and `("nonce",j)` is a tuple containing a string then an integer. They are not pre-encoded `u64be` byte strings. Appendix A gives every byte they become.

Mode 1 is valid only with a beacon value `beacon_j` that was unavailable before round j. For each round:

```text
round_key_j = derive(
  beacon_j, "verity/pous/p2-audit-round/v4",
  {"audit": audit_id, "epoch": epoch_id, "root": verification_root, "round": j}
)
i_j = round_key_j.uniform(block_count, j)
round_nonce_j = beacon_j
```

Each `beacon_j` is exactly 32 bytes. In mode 1, the verifier derives `audit_id` with the same `verity/pous/p2-audit-id/v4` formula from a separate verifier secret or opening beacon that is not reused as a round beacon. The verifier records every beacon identifier and verifies its authenticity outside this wire format. A round beacon known at audit opening is not sufficient for sequential unpredictability.

### Challenge and response

`CHALLENGE` (`0x11`, verifier to prover) is:

```text
epoch_id[64] || audit_id[32] || u16be(round_j) || u64be(index_i)
|| round_nonce[32]
```

Deployment rounds are numbered 0 through 104. The receiver checks epoch, audit, round, nonce and the deterministically derived index.

`RESPONSE` (`0x12`, prover to verifier) always carries the block and a root authentication path:

```text
epoch_id[64] || audit_id[32] || u16be(round_j) || u64be(index_i)
|| u32be(block_length) || block[block_length]
|| u16be(path_count) || sibling_0[64] || ... || sibling_(path_count-1)[64]
```

The block is the spec's fixed-width unsigned big-endian encoding; at the deployment point it is exactly 2,056 bytes and must be below p. Siblings are ordered from the leaf toward the root. Direction bits are omitted: RFC 9162 path verification derives left/right from `index_i` and `block_count`. `path_count` must be exactly the count consumed by that verification; for B = 2^19 it is 19, 1,216 bytes of siblings. A one-block tree has `path_count=0`. Always carrying a path gives one interoperable root-only format; a verifier that also retains every leaf may still verify the path.

### Sequential reveal and deadlines

The verifier releases round j+1 only after the complete response frame for round j has been received. It stops immediately after any failed round.

For round j, the verifier starts its monotonic clock immediately before handing the first byte of the challenge frame to the integrity-protected transport. It stops when the last byte of the corresponding response frame is received. Parsing and cryptographic verification follow the stop event but malformed and wrong responses still fail.

The on-time condition is:

```text
elapsed_ns ≤ delta_ns + rtt_allowance_ns
```

Equality is on time; one nanosecond above the sum is late. The verifier measures the RTT allowance on the same authenticated transport and records the method and samples. The frozen deployment sets Δ = 0.5 ms with an on-node verifier and permits a measured RTT allowance of at most 354 µs (354,000 ns), so `Δ + RTT ≤ 854 µs`, under the hardware/cooperation class in §6 (v1.1, Daniel's ruling, §12; v1: at most 1 ms). Δ and `rtt_allowance_ns` in `AUDIT_OPEN` are the verifier's own: a verifier never opens an audit with Δ ≠ 500,000 ns or an RTT allowance above 354,000 ns. The wire format carries them and assigns no failure code to them. `elapsed_ns` is the verifier's own measurement, recorded as u64; a measurement outside `[0, 2^64)` is the verifier's fault, and the session ends without a `FINAL`. A late, malformed, unauthenticated, replayed, wrong-index, wrong-block or wrong-path response fails the whole audit. There is no partial acceptance.

Only a frame reported as delivered at a transport frame boundary enters the exact frame sequence. A delivered frame whose header declares more or fewer bytes than that delivered frame contains is malformed code 1; its exact delivered bytes are hashed, including the `truncated-response` vector. If the transport ends or reports loss after a challenge without reporting any complete response-frame delivery event, that round fails with transport code 6; unframed partial bytes are not hashed.

Perfect isolation, the `floor((18/19)·|C|)` state cap, accounting of transient/cross-answer state and exclusion of outside help are procedural assumptions. The wire format does not enforce them.

### Response verification and final record

For each response the verifier:

1. validates the envelope, ids, round, index, fixed block length and canonical block `< p`;
2. recomputes the leaf and RFC 9162 path to the epoch's verification root;
3. enforces the strict deadline;
4. when flag bit 0 is set, runs P2 decode and requires a canonical payload for the selected profile;
5. when flag bit 1 is set, also compares that payload with verifier-held expected bytes.

Authentication is mandatory. Decode checks are additional diagnostics once setup has bound the root to committed W.

When one input has multiple faults, the verifier selects exactly one failure in this order: malformed envelope, length or canonicality (code 1); identity, round, index, domain or replay (code 2); authentication (code 3); deadline (code 4); optional decode or payload comparison (code 5). An index outside `[0,block_count)` is code 2. A path byte-count inconsistent with `path_count` is code 1. A RESPONSE can be envelope- and length-consistent yet carry the wrong number of complete siblings; if `path_count` differs from the exact RFC 9162 count for the challenged `(index,block_count)`, it is also malformed code 1, before hashing the path. With the correct count, a path that does not reach the bound root is authentication code 3. The one setup exception is a root that disagrees with verifier-reconstructed setup blocks: this is code 3 before checking `AUDIT_OPEN.epoch_id`, because the declared root itself is unauthenticated.

A malformed challenge emits one failing record with the expected round and derived index, elapsed zero, flags zero, the hash of the exact challenge frame and a zero response hash. A complete malformed response emits one record with its measured elapsed time and exact response hash. Failure before a complete response emits the transport record specified below. A fixed-size `AUDIT_OPEN` with invalid mode, k or reserved verification bits emits a zero-record `FINAL` by reading its identity and timing fields at their fixed offsets; an open frame too malformed to recover those fields is rejected at the transport/session layer and cannot produce a `FINAL`.

The verifier records the exact `AUDIT_OPEN`, challenge and response frames; start, stop and elapsed nanoseconds; RTT evidence; software/image identity; hardware identity; per-round checks; and the final result. This may be attached to a `research` Attempt or another evidence record without changing the bytes below.

Define:

```text
transcript_hash =
  SHA-512(
    ASCII("verity/pous/p2/transcript/v4\0")
    || exact AUDIT_OPEN frame
    || exact CHALLENGE_0 frame || exact RESPONSE_0 frame
    || ...
  )                                                    64 bytes
```

`FINAL` (`0x13`, verifier to prover and evidence record) is:

```text
epoch_id[64] || audit_id[32] || verdict[1] || failure_code[1]
|| u16be(rounds_completed) || u64be(delta_ns) || u64be(rtt_allowance_ns)
|| transcript_hash[64] || u16be(record_count) || records
```

Each record is 147 bytes:

```text
u16be(round_j) || u64be(index_i) || u64be(elapsed_ns) || flags[1]
|| SHA-512(exact challenge frame)[64] || SHA-512(exact response frame)[64]
```

A zero response hash is 64 zero bytes.

Record flags are bit 0 authenticated, bit 1 on time, bit 2 decode performed, bit 3 decode valid; other bits are zero. Bit 1 is evaluated and set only after authentication succeeds, so a complete on-time response that fails parsing, identity or authentication has flags zero. `rounds_completed` and `record_count` are both the number of emitted round records, including a final failed attempt; a failure before round 0 or in `AUDIT_OPEN` has zero. Verdict is 1 only when all k records authenticate and are on time and every requested optional check succeeds. Failure codes are 0 success, 1 malformed, 2 id/round/index/replay/domain, 3 authentication, 4 late, 5 decode or payload comparison and 6 transport. A complete but invalid response is hashed and included as the failing record. If transport fails before a complete response frame arrives, the failing record uses an all-zero response hash, the transcript ends after that round's challenge, and its flags are zero. Any frame delivered after the expected k exchanges changes the verdict to failure code 2 at stopping round k; `rounds_completed = record_count = k`, and the extra frame is not appended to the transcript hash.

### State isolation and state accounting

This subsection maps the Lean sequential audit game to deployment. It is normative for claims that cite `Pous.Meets` or the P2 exact-domain pins. Let:

```text
|C| = block_count · w
S = floor((18/19) · |C|) bits.
```

The bound is a condition on the adversarial prover quantified by the security guarantee, not a storage rule for the honest prover. The formal conditional claim is: every perfectly isolated prover whose retained state is at most S bits and whose round program is `Prog.Bounded D Q` passes with probability at most `δ + ε`. The honest prover retains all of C and is expected to pass. Its wall-clock interpretation additionally depends on the external calibration and SeqRoot assumption below.

#### Game-to-deployment map

| Lean object | Deployment resource |
|---|---|
| W | the exact committed plaintext model bytes; available to every response round for free |
| pp | the scheme's public parameters; free under the state cap, but `|pp|` is charged by `SpaceBound` |
| public setup | fixed scheme constants are free; a per-setup value is free only if it is part of W or formal pp, is exposed through the modeled public-oracle interface, or is justified by the external concrete-to-ideal reduction |
| `A₁(W,pp,C,ω)` | an arbitrary function of W, pp, all of C and the whole ideal primitive ω; encoder coins are not an input |
| `σ : Bits S` | all non-free information that can survive a round boundary, jointly |
| `step W pp j σ i` | one responder invocation given the round number and current challenge, subject to that answer's formal `(D,Q)` oracle bound |
| returned `Bits S` | the only non-free state allowed to survive for the next round |

For the exact-domain `m1p` pins, pp has width zero. The pins therefore do not themselves make the public verification root, setup salt, concrete K/t representation or authentication metadata free. Fixed constants may be hard-coded; per-setup values require the P2-to-M1p instantiation argument or must be charged. After preprocessing, direct access to C and ω is gone; ω is available online only through the modeled forward-query interface.

The current challenge means `(audit_id, j, i_j, round_nonce)` and the associated public verifier fields. Earlier challenges, responses and revealed C blocks are not fresh free inputs in the Lean game: retaining them for a later round counts unless they are recomputed solely from the true free inputs identified above.

#### What is charged at a boundary

At a state boundary, concatenate every non-free bit available to, controlled by, or queryable by the responder. The resulting retained representation must have at most S bits. Deployment accounting conservatively charges the bytes of every accessible copy; duplicate caches are not deduplicated. It includes:

- GPU HBM and SRAM that survive the round, CPU and pinned host RAM, registers or runtime objects that survive, swap, disk/NVMe, files, object stores and database rows;
- driver, framework, allocator, kernel, filesystem and page caches when their contents depend on C or preprocessing;
- state held by another process, container, VM, device or service that the responder can read or query;
- compressed, coded, encrypted, sparse or indirect representations of C, their indices and metadata, and any non-public decompressor/table;
- prior responses, decoded blocks or decoded weight tiles retained across answers.

A region proven to be a fixed constant or a deterministic function only of the true free inputs adds no information, but any extra annotations or hidden payload in it are charged. A local copy of the exact committed W is free because W is an explicit input to every Lean `step`; an altered copy carrying hints is not. This does not make C free: the aggregate-Q calibration and SeqRoot assumption must separately exclude regenerating a challenged C block from W before its answer deadline.

Reading the exact committed W from local disk or over the verifier's authenticated channel is free. That source must return only W and fixed framing; it cannot be used as a general storage or advice channel. Reading any other disk or network object that encodes C or preprocessing output either charges all information obtainable from it to σ or violates perfect isolation. A public-setup value may be stored or recomputed without charge only when it meets the formal or externally justified free-input rule above.

#### Boundary times and transient work

The state boundary is enforced:

1. after preprocessing and erasure of C, encoder coins and setup secrets, immediately before round 0's challenge; and
2. after the complete RESPONSE for round j has been produced and before CHALLENGE j+1 is released.

There is no unmetered computation gap between rounds. The responder's next-state choice is part of round j. Working memory and local computation within a round are unrestricted by S, matching `Prog`. Before the next boundary, every transient byte must be erased or represented in the counted next state. Temporary files, child processes, GPU allocations, swap, caches and asynchronous work do not become free merely because the response was already sent.

In `Prog.Bounded D Q`, D is the number of adaptive oracle rounds and Q is the total number of modeled forward-oracle queries in one answer, across all cooperating hardware; neither bounds local computation. The P2 pins hold for every D. A deployment claim must separately freeze the hardware/cooperation class, calibrate aggregate Q for the `Δ + RTT` response window, and assume SeqRoot excludes an inverse or equivalent target computation in that same window. The Lean game contains no wall clock: every `(D,Q)`-bounded program is treated as on time.

Cross-answer decoded-block, root, lattice, table or key caches count whenever they contain information not determined by the true free inputs. Reloading a charged object after the boundary does not evade the cap: if it remains reachable, its retained information was part of σ at that boundary.

#### Perfect isolation and evidence

During an audit the responder may communicate only with the authenticated verifier transport, including an exact-W input channel if W is not already local. The deployment must isolate the responder host or VM from other networks, storage services, processes and devices; prevent DMA or peer-device access; bind the responder to an attested image/configuration; and account jointly for every cooperating process and device in the selected hardware class.

The attested image and configuration must be frozen before preprocessing. Any setup-dependent executable, library or configuration payload counts as state. At each boundary, all cooperating processes and devices must be suspended, asynchronous work terminated or joined, transient state erased, and the retained state atomically measured and committed before the next challenge is released.

For each audit the verifier's evidence record must include the attested image and executable hashes; host, device and isolation identity; process/cgroup and device membership; mounted storage and allowed-file policy; network namespace/firewall policy and counters; memory and persistent-file measurements at preprocessing and every boundary; state byte/bit counts and hashes; cache/temporary-state wipe results; child exit status; and per-round timing already required above.

Perfect isolation remains an assumption, not a property of `p2-audit-wire/v4`. The wire format cannot establish that native code did not use DMA, a covert channel, shared kernel/page cache, an unmeasured process, remote storage or false attestation. Production enforcement and independent attestation of those controls remain procedural.

#### MVP and reference harness

In the vLLM/MVP deployment, C is the P2-encoded weight blocks retained for serving, principally in GPU HBM with any host or disk backing also part of the retained deployment state. W is the committed plaintext model and is free in the game. The honest responder retains all of C and answers by lookup; it is outside the S-bit adversarial class. For a capped adversarial run, every encoded or decoded weight cache that survives an answer—including framework cross-invocation caches—counts. The MVP's ordinary in-process objects do not by themselves enforce perfect isolation.

`internal/efficient-crypto/p2-bounty-kit/isolation_harness.py` is the executable small-width reference. Preprocessing runs in one process and may emit only an S-capped state file. State files are byte-granular, so the harness conservatively permits at most `floor(S/8)` bytes. Every round runs in a fresh process with only W, pp, public setup, the prior state and challenge; only the response and next-state files may be written. The harness uses a fresh user/network namespace when available, a deny-network Python audit policy, a cleared environment, per-process address-space/CPU/wall-time limits, fresh temporary directories, an allowed-file scan, and state-size checks after preprocessing and every round.

This harness makes the file/process boundary executable for the included toy Python responders but is not a hostile-native-code sandbox, perfect-isolation implementation or attestation system. Its fallback without user namespaces is only a Python-level diagnostic. It does not isolate native syscalls or the mount/PID namespaces, mediate `/proc`, DMA or peer devices, measure remote state, count oracle depth or queries, enforce the audit wire deadline, or test SeqRoot. Its address-space, CPU and wall-time limits are diagnostic guards, not the Lean `(D,Q)` bound.

`test_isolation_harness.py` has finite scope: it checks only the built-in full-C, raw-cap, side-file, environment and Python-socket strategies. The raw responder stores 18 blocks exactly at the cap. Across 80 fixed-seed toy audits, the test checks that it succeeds exactly when every sampled index is among those 18; the probability `(18/19)^k` follows from the independent with-replacement challenge rule, not from that finite test.

### Audit wire test vectors

`benchmarks/pous/p2_v1/vectors.py` generates every vector file of this specification in one command,
`cd benchmarks/pous && uv run python -m p2_v1.vectors`, and rewrites each pin in §7 that changed. `audit-vectors.json` holds a complete two-round w=64 setup/audit and one full-width 2,056-byte response. It includes W, setup sources, audit sources and counters, elapsed times and the intermediate values needed to recompute every field. Both small and full-width audit ids use the mode-0 `derive(source,"verity/pous/p2-audit-id/v4",{"counter":counter,"epoch":epoch_id}).stream("id",32)` map above. `audit-negative-vectors.json` holds 31 complete negative and boundary transcripts, each self-checked against the reference verifier `benchmarks/pous/p2_v1/wire.py` when generated and checked again by `benchmarks/pous/tests/test_pous_p2_wire.py`; it includes the setup source for each setup family. The wrong-domain case derives its challenge under v1's `verity/pous/audit-challenge/v2`. `expected_stop_round` is the failing round, k on success or on an extra post-k frame, and zero for a failure before round 0. The deployment value is k=105; k=2 exists only to keep the conformance vectors small. File and message hashes are listed in §7.

## 5. Audit

- B = 2^19 blocks, arranged as 8 segments of 65,536 blocks.
- k = 105 challenges, revealed and answered sequentially.
- Each answer has Δ = 0.5 ms plus a measured RTT allowance of at most 354 µs (Δ + RTT ≤ 854 µs; v1.1, §12), with the verifier on-node.
- Q = 2^20 complete forward decodes per answer in the abstract model.
- The prover's retained state is at most `floor((18/19)·|C|)` bits, where `|C| = B·w`; it is not `(18/19)` of the payload. The state cap applies between answers.
- `P2SlackFamilyUncond` certifies that k = 105 absorbs 128 bits per block of bounded, reusable structural leakage for every B from 2^8 through 2^23.
- Partial, batched, cross-block and expander work is in scope. “No inverse” means no modular root, nor another computation with the same effect on the challenged target, finishes within the answer window.

These values are the frozen F2 and F6 defaults (default, Daniel deferred, 2026-09-29), with F2's RTT allowance as Daniel ruled for v1.1 (2026-10-04, §12). The audit remains conditional on P2–M1-SGI, SeqRoot, perfect isolation, the exact `[0,p)` model and the allowed hardware.

### Normative error total

The frozen F3 error allocation is (default, Daniel deferred, 2026-09-29):

> A prover that retains at most ⌊(18/19)·|C|⌋ bits beyond W and the public setup between challenges passes an audit
> with probability at most 1% + 2^-128. The 1% is the audit's sampling term, exact in the model. The 2^-128 is a sum of
> four terms:
>
> - (a) **the ideal-model term** ε_ideal ≤ 2^-129, proved in Lean for P2's exact-domain model
>   (`P2ErrorSplitB19Uncond`; the proved bound is far smaller than 2^-129);
> - (b) **the instantiation term** ε_inst ≤ 2^-130, assumed: that square–mask–square behaves like independent ideal
>   permutations in this storage game (P2–M1-SGI);
> - (c) **the challenge-derivation term** ε_chal ≤ 2^-131, assumed: that `verity.ml.randomness` outputs are
>   indistinguishable from uniform;
> - (d) **the expander term** ε_exp = 0 under the setup assumption P2-EXP-IO (F4); its concrete realization by
>   counter-mode SHA-512 is the separate named assumption SHA-512-CTR/P2-EXP-IO (§4).
>
> The terms combine additively by a reduction outside Lean, so the total is at most 2^-129 + 2^-130 + 2^-131 < 2^-128.

`P2ErrorSplitB19Uncond` proves this split at k = 104 for B = 2^19. The k = 105 deployment inherits it because one additional sequential challenge can only lower acceptance; this monotonicity step is a paper argument outside Lean.

### Trusted-layer pins

The frozen F5 decision pins `P2SlackB19Uncond`, `P2SlackFamilyUncond` and `P2ErrorSplitB19Uncond` in the store's `lean/pous` trusted layer (default, Daniel deferred, 2026-09-29), together with the exact-domain pins `P2MeetsM1pB19Uncond`, `P2MeetsM1pFamilyUncond` and `P2MeetsM1pW8192Uncond`. Repository synchronization follows PRs #162/#183.

## 6. Operating point and allowed hardware

The frozen operating point is B = 2^19 blocks, Q = 2^20, w = 16,448, Δ = 0.5 ms with an on-node verifier, and a measured RTT allowance of at most 354 µs, so Δ + RTT ≤ 854 µs (default, Daniel deferred, 2026-09-29; the 354 µs cap is v1.1's, Daniel's ruling of 2026-10-04, §12). The adversary may use a CPU or GPU core with cooperation within one chiplet. ASICs and FPGAs are excluded. The verifier and responder are pinned on the same chiplet. The verifier records the CPU/GPU model, clocks, topology, cooperating cores and measured RTT.

SeqRoot evidence is artifact `art:b92b6d56fa12cc8e04b3f6843991c2e76f49366adc89140a96c33842f70b41ed`, run `r20260929-060834-0ec4`. The fastest measured 16,448-bit root took 12.8 ms with eight cooperating cores. The on-node RTT p99.9 was 10–22 µs. Against the architectural floor with eight cores of one chiplet cooperating, 5,960.5 µs ÷ 3.490 = 1,708.1 µs (*est.*, §12), the 2× margin holds while `Δ + RTT` stays under 854,029 ns, which v1.1's RTT allowance of at most 354 µs keeps. v1's "through approximately 1.9 ms" used a floor about 2.2× too slow (§12). These estimates do not admit cross-chiplet, multi-GPU, FPGA or ASIC cooperation.

## 7. Test vectors

Canonical vectors are `benchmarks/pous/p2_v1/vectors.json` (schema `p2-test-vectors/v4`), checked by
`reference.conformance()` (`benchmarks/pous/tests/test_pous_p2v1.py`), and the two audit files beside it. Every file
records the `verity.ml.randomness` derivation it was generated under (`"randomness"`), here
**`verity/randomness/derive/v2`** (Appendix A). `cd benchmarks/pous && uv run python -m p2_v1.vectors` regenerates all of
them, and the package's copy of the keys (`verity/core/protocols/pous/tests/fixtures/p2-v4-keys.json`), and
rewrites these pins; a change to the derivation moves every setup salt, and with it every K, t and c of `vectors.json`,
every audit id, index and nonce, and so every frame and FINAL of both audit files, while m, the primes and the σ
rejection cases do not move.

Each vector's source is `SHA-512("p2-vector-source/{w}/0")` (64 bytes), its model commitment
`SHA-512("p2-vector-model/{w}")`, its segment 2 and local block 7, and its m the first n bytes of
`SHA-512("p2-vector-m/{w}/0" ‖ u32be(0)) ‖ SHA-512(… ‖ u32be(1)) ‖ …` reduced mod `2^b`, b the payload bits
(`8·(n − 1)` at a reduced width, 16,432 at deployment). For every map entry, its `sha512` is
`SHA-512(canonical_json_bytes(entry_without_sha512))`.

- `vectors.json` SHA-512: `959911fd371111fe20c4cee6fc532ff205bbf377962c73ac539a4d2de9b22121287e6e996d9ae4bfbef9f26fcd67ac78620873702c8b2899bc41a21bdd76e853`
- w=64 entry: `c34350fa0e0b8ae7c521cd7df833c160df1498adc938a99feb6166c2f37b3e6a2ca07239b9ab71fdd979b02efc8c8951d68f32d108abbde5cc44bac3137241e1`
- w=96 entry: `df8941e52d73848baa7862cd3f20ea6f7adc1da79149db154b0be9e0d28be8905b95aac079cd98e14b56f8077ba6cfb173e0472df7e83b40138c6f16a871ff1f`
- w=128 entry: `adf7cd300e2d4ea93643e94d563b02c80e29d88d65a34a55269f9e37da6b71bde6dce1c9e4c04c561dd103a8dc4c3955ee83ecabef20ad717de4672e8014fcae`
- w=16,448 entry: `2a227a512683d67624bca51ba087ec05a4dda85c5e93b67c50c03349a7823f50f84b98f4f2f9e92d9c87ddc41ab0e5fd8d2211472709a05093e7c0121a86811c`

The file includes explicit sigma rejection-branch cases at w=64 and w=16,448, and carries v1's GMP primality screen of
each point's p (BPSW plus 64 Miller–Rabin repetitions), which v4 does not change.

Audit wire vectors:

- `audit-vectors.json` SHA-512: `5f8ed7f279ccd2f2d5bb60607fb5d7bec82bf982ac85da268f76cb8fbb16d36424e7b99eaa386508840aa2bd3c3b501512bde86b930871a769cd470a14ab176e`
- `audit-negative-vectors.json` SHA-512: `f57340a0965628051bd0c938c4c7e2db5a7133982ae1a523a1221a3fbb273bfbcb1ebc1d29680b61b47f86d45e7c98aa4a5ac6311545c19c78728d857596a687`
- w=64 audit-open frame: `190450d36926cd0c6c66ec0161d9815cc3f1f3bd3d9c864cdfe87aa1ece95a06e8f63ce3261d258ff22425ef852e4434df50c141c265e79ca0493ac3f0c81887`
- w=64 round-0 challenge: `227ee32fbec8715102a4e6585af430ab88a22bfca2ed65d8ef372c408439e4fdcf07a6747e22f6501f0e005eab499ad4b82c167fadbc1ddac8bd9827e8b271c7`
- w=64 round-0 response: `36ac81b1ef2d7499e96563c4c53e7ac8b7b08c8d2c2907217d43196c12f82268851db7f5ce59bc73c56babb0ca20a38e071db002febf8b4fc84c7741471f073d`
- w=64 final frame: `30f813852726925a93d3a1a90985a434d3bcd98377acd54d36cef43c2f78ab5f28b6c07338b827b80c3e888682e74cb8d97527978882c2d838dbd407c3d83adf`
- w=16,448 response frame: `48f7b363b38b914f157567d6ce3398e5d40a2b25c0db73e811e4b0a7a62154a04cd71dee111ab852a5ae228a7f6593224a4eefcb0415d795469677e75103579c`

## 8. Conformance of the repository

`benchmarks/pous/p2_v1/reference.py` is this specification (§§3–4 and the derivations), `wire.py` is
`p2-audit-wire/v4` and its reference verifier, and `memory_challenger.py` audits with v4's leaves and mode-0 derivation
over the harness's own responder wire, not these frames. The registered scheme `p2-16448/v4`
(`verity.core.protocols.pous.schemes.p2`) follows the maps, the values, the profile, the tweak rejection, k and
the hash, and its keys are §4's streams, but it runs the generic protocol's setup, so four things differ:

| Topic | v4 | `p2-16448/v4` |
|---|---|---|
| Setup salt | 32 bytes, `verity/pous/p2-setup/v4` over the model commitment | 24 bytes, the generic `setup.draw_salt` |
| Key prefix | frames the model commitment and the 32-byte salt (176 bytes, padded to 256) | frames the 24-byte salt (95 bytes, padded to 128), so its base is 144 bytes; the chunks are the same function |
| Leaf | `0x00 ‖ salt[32] ‖ u64be(s) ‖ u16be(v) ‖ R` | `0x00 ‖ tag_s[32] ‖ u16le(v) ‖ R`, `tag_s` the generic protocol's per-segment tag |
| Challenges | `verity/pous/p2-audit-challenge/v4`, bound to audit, epoch, root and salt | the generic `verity/pous/audit-challenge/v1`, bound to the root and salt |

Both commit W with tagged SHA-512 under `verity-pous/w-commitment/v2`, build the RFC 6962 tree with SHA-512 nodes, and
hash the same R. `p2-16448/v2` (ChaCha8 keys) is nonconforming to F4.

## 9. Freeze checklist

Every item below is normative. A policy choice is not a substitute for the corresponding pinned artifact or executable check.

- [x] Archive and independently verify the ECPP certificate for the exact p: both certificate hashes and all three passing verification records are pinned in §1 and `docs/efficient-crypto/p2-prime-certificate.md`.
- [x] Freeze payload profile `p2-2054B` and the corresponding exact-domain `m1p` pins.
- [x] Freeze B, k, Q, Δ, round-trip accounting and the allowed hardware/cooperation class together; the SeqRoot run and margin pins are cited in §§5–6.
- [x] Allocate ideal-model, expander and concrete-instantiation errors with the normative `1% + 2^-128` total in §5.
- [x] Freeze P2-EXP-IO, the concrete SHA-512-CTR/P2-EXP-IO assumption, v4 expander framing and vectors.
- [x] Specify challenge derivation after vk through `verity.ml.randomness`, including versioned domains, audit/epoch binding, round indices and sequential beacon requirements (`p2-audit-wire/v4`).
- [x] Specify root authentication and RFC 9162 inclusion paths, with SHA-512 nodes over 0x00-tagged leaves of each block's row digest.
- [x] Specify complete setup, audit-open, challenge, response and final-record frames, including failure behavior.
- [x] Specify replay prevention and domain separation across models, epochs, audits, segments, blocks and rounds.
- [x] Specify and test state isolation: the normative subsection above maps the Lean state and boundaries to deployment, and `internal/efficient-crypto/p2-bounty-kit/isolation_harness.py` with `test_isolation_harness.py` passes its full-C, 18/19 raw-storage, over-cap, side-file, environment and network controls. Native-code/OS isolation, GPU/DMA and cache wiping, aggregate resource measurement, attestation and verifier evidence collection remain procedural deployment controls.
- [x] Specify deadline clock boundaries, strict failure behavior, RTT allowance input and transcript evidence, including the frozen numerical Δ/RTT/hardware choice and SeqRoot evidence.
- [x] Generate §7's vectors under `verity.ml.randomness` v2 (one command, §7).
- [x] Obtain byte-for-byte interoperability results from two independent implementations of `p2-audit-wire/v4`: the reference verifier `benchmarks/pous/p2_v1/wire.py` and the Lean `pous_check` (`verity/core/protocols/pous/check`, `PousCheck.Wire` and `PousCheck.Audit.check`) agree byte for byte on every vector of §7, `uv run python verity/core/protocols/pous/check/interop.py --all`: 40 transcripts (the 31 of `audit-negative-vectors.json`, `audit-vectors.json`'s end-to-end audit and its full-width response under a public and a private setup, and `vectors.json`'s 4 vectors and 2 σ cases, the verifier holding its own root of the private setup), 0 mismatches in each setup's outputs (its failure code among them), `FINAL` frame, verdict, code, record count and transcript hash, or fault code and message (run `r20261008-174759-2f4d`, `art:6e51b5bb3d65ea9bf034a704c5c810717db5fc24dbff20222f34c4b8dd818bf2`). The Lean is a verifier only: the setup and response frames both read are the vectors', which Python's prover wrote. v1's results (`internal/efficient-crypto/p2-audit-wire-interop.md`) do not carry over.
- [x] Add negative conformance cases for malformed encodings, setup failures, replayed challenges, wrong domains, failed authentication paths and late answers: 31 complete transcripts in `audit-negative-vectors.json`; its hash is listed in §7.
- [ ] Migrate the MVP implementation and evidence path to every requirement in §8's table.

## 10. Decisions (frozen)

1. F1: `p2-2054B` is the sole deployment payload profile (default, Daniel deferred, 2026-09-29).
2. F2: B = 2^19, Q = 2^20, w = 16,448, Δ = 0.5 ms, RTT allowance at most 354 µs, and same-chiplet cooperation and pinning as §6 (default, Daniel deferred, 2026-09-29; the RTT cap, at most 1 ms in v1, is v1.1's, Daniel's ruling of 2026-10-04, §12).
3. F3: the normative error total and allocation in §5 (default, Daniel deferred, 2026-09-29).
4. F4: accept P2-EXP-IO, realized by counter-mode SHA-512 with v4 framing; ChaCha8 is nonconforming (default, Daniel deferred, 2026-09-29; SHA-512 in place of SHAKE256 by Daniel's ruling of 2026-10-06, §13).
5. F5: pin `P2SlackB19Uncond`, `P2SlackFamilyUncond` and `P2ErrorSplitB19Uncond` (default, Daniel deferred, 2026-09-29).
6. F6: k = 105, absorbing 128 bits per block of structural leakage over B = 2^8 through 2^23 (default, Daniel deferred, 2026-09-29).
7. Keep p = 2^16448 − 21065 under its ECPP certificate, without requiring a Lean-checked primality proof (default, Daniel deferred, 2026-09-29).
8. Do not accept P2–M1-SGI; P2 remains experimental despite this frozen specification (default, Daniel deferred, 2026-09-29).
9. One hash: SHA-512 with its whole output for every digest, and SHA-256 retired from P2 (Daniel's ruling of 2026-10-06, 15:07Z, §13).

## 11. Remaining before deployment

- Migrate the MVP to v4 as listed in §8.
- Counter-mode SHA-512 key derivation on the deployment device, measured: on an NVIDIA RTX PRO 6000 Blackwell Server Edition (sm_120, vy-nebius-1 GPU 3), `p2-2054B` (n = 2,056) at B = 2^19, with `p2keys4.cuh` (`-DP2_FAST -DP2_KEYS4 -DP2SQ_KARA`), every block's `K` and `t` are derived at 1.136 × 10^8 blocks/s (4.62 ms per store, the median of 10), and the setup's first derivation for the whole store takes 5.39 ms of wall time. All 2^19 blocks' `K` and `t` equal hashlib's, as do the 59 reference blocks and the pinned `p2-v4-keys.json` entry (`p2-v4/w16448/0`). Run r20261008-152508-ff34, art:07d65cea4534caaa8f92e364455fd4b998632b4e020302ce6f38525a50617dfa (`decode_bench.py --keys-all`).
- Accept P2–M1-SGI.
- Synchronize the trusted-layer pins into the repository after PRs #162/#183.

## 12. Amendment v1.1 (2026-10-04)

*As written for v1.1. v4 (§13) changed the hashes, the byte formats and the vectors; it kept the cap.*

**v1.1 is Daniel's ruling of 4 Oct 2026, 2:21 PM PDT (21:21Z),** relayed as ruling 3 of the migration-kickoff
announcement (Slack thread 1791150333.889129): "*P2:* cap the RTT allowance near 0.35 ms (Δ + RTT ≤ about 0.85 ms) and
keep w = 16,448. @memory-accounting, please write it into the spec and `ROOT_CALL_NS`, with the exact cap that keeps 2×
on the Zen 5 floor."

**What changed.** F2's RTT allowance, from at most 1 ms to at most **354 µs (354,000 ns)**, so `Δ + RTT ≤ 854 µs`
("Sequential reveal and deadlines" and §§5, 6, 8 and 10). Nothing else changed: w, p, B, Q, k, the hardware class, the
protocol version `p2/v1`, the byte formats and every vector. Every `AUDIT_OPEN` in `audit-vectors.json` and
`audit-negative-vectors.json` carries `delta_ns = 500,000` and `rtt_allowance_ns = 0`, which is inside the cap, so the
hashes in §7 still hold.

**Why.** v1's "2× through about 1.9 ms" (§6) divided a one-core floor of about 13.2 ms by the measured 3.5×, then halved
it. 13.2 ms appears to be the measured Zen 4 root (44.68 ms) divided by 3.38, the ratio of measured to best-plan floor
at 5,504 bits (this source is inferred: 44.68 ÷ 3.38 = 13.2, ÷ 3.49 = 3.79 ms, ÷ 2 = 1.89 ms). At 16,448 bits, the
timing red team's own floor is 5.96 ms, 7.5× below the measured kernel rather than 3.38×. Against it, v1's
`Δ + RTT = 1.5 ms` kept only 1.14× (1,708.1 ÷ 1,500). The P2 assumption review found the frozen cap unsafe on the same
grounds (`art:f33b450f2cd48f6cec2057915ae08b2221a4a6060dc2f9fa38055e8e4d74b98c`, §4).

**The cap, derived.**
- The single-core floor is the timing red team's best case for the adversary (`regate.py`): a Zen 5 desktop core at
  5.7 GHz with the full 512-bit datapath. It issues 2 IFMA and 4 vector operations per cycle at 100% efficiency, with
  the best plan of Toom-4, Karatsuba and Karatsuba over 317 limbs of 52 bits, including normalisation, Toom evaluation
  and interpolation, the fold by c0 and the carry passes, plus 25 cycles of latency. That is 2,060.8 cycles, so
  **τ_F = 361.55 ns per squaring** at w = 16,448.
- One root is (w − 2) squarings plus about 20 multiplications, counted as 40 squaring-equivalents: 16,486 × 361.55 ns =
  **5,960.5 µs**. The red team's 5.97 ms rounds τ_F up to 362 ns first.
- The cooperation speedup is the measured one on one 8-core Zen 4 chiplet at 5.475 GHz: 2,716.7 ns per squaring on one
  core against 778.5 ns on eight, **3.490×** (`results.jsonl`). The roots agree: 44.68 ms ÷ 12.80 ms = 3.49×.
- The fastest cooperating root is 5,960.5 ÷ 3.490 = **1,708.1 µs**. Exactly, from the sources' own numbers:
  16,486 × (2,060.845 cycles ÷ 5.7 GHz) × (778.5 ÷ 2,716.7) = 1,708,058.35 ns, and half of it is 854,029.18 ns.
- `ROOT_CALL_NS` is that half in ns, rounded down: **854,029**. The 2× margin needs `Δ + RTT` under it. With
  `rounds = 0`, the certificate admits `Δ + RTT ≤ 854,028 ns` and refuses 854,029 ns and above.
- The cap is the largest whole µs with `Δ + cap < ROOT_CALL_NS`: 500,000 + 354,000 = 854,000 < 854,029, and
  500,000 + 355,000 = 855,000 is not. So **`RTT_ALLOWANCE_CAP_NS = 354,000`**, and `Δ + RTT ≤ 854 µs`.
- The cap and the certificate agree. The verifier accepts the cap and refuses every whole µs past it: 355 µs, and v1's
  1 ms. The certificate's own edge, an RTT of 354,028 ns, is 28 ns past the cap, under the µs the cap is stated in.
- The margin at the cap is 1,708,058.35 ÷ 854,000 = **2.0001×**; one µs more would leave 1.998×.
- The implementation: `verity.core.protocols.pous.schemes.p2.ROOT_CALL_NS = 854_029` and
  `RTT_ALLOWANCE_CAP_NS = 354_000`, and `benchmarks/pous/p2_v1/reference.py`'s `MAX_RTT_ALLOWANCE_NS = 354_000`.

**What the cap rests on.**
- **The floor is a model, not a measurement.** No kernel reaches it: the campaign's best measured kernel is 3.38× slower
  at 5,504 bits. A core with twice the IFMA throughput halves the root, and so the cap.
- **The cap has no slack above 3.490×.** Rounding the speedup up to 3.5× leaves 1.994× at the cap; at 3.5× the cap would
  be 351 µs.
- **The margin mixes a measurement with a model.** The 3.490× was measured on an untuned Zen 4 kernel, at 2,717 ns per
  squaring, and is applied to the modelled Zen 5 floor of 361.55 ns. If the exchange between cores costs a fixed number
  of ns, a kernel at the floor gains less from cooperation, so this composition favours the adversary; but it is not a
  measurement of cooperation at the floor.
- **The red team's exchange model gives more cooperation than was measured.** It assumes 20–40 ns of synchronisation per
  squaring (`coop.log`, `regate.py`) and gives 4.24× (40 ns) and 5.55× (20 ns) at 8 cores and 16,448 bits. Under it the
  margin at the cap is 1.64× and 1.26×, short of 2×, and the cap that kept 2× would be 202 µs or 37 µs. The measured
  exchange inside one chiplet is a 33 ns core-to-core round trip. The cap rests on the measured speedup, not on this
  model.
- **Only 8-core cooperation is measured.** A 16-core Zen 5c CCD, or a larger die counted as "one chiplet", is not.
- **Honest answers fit.** The measured on-node RTT is 6.8–16 µs at the median and 10–22 µs at p99.9. Honest timed
  answers reached at most 161–256 µs (review §4). Both sit inside `Δ + RTT = 854 µs`.

**Alternatives not taken.** Margins are the cooperating root ÷ `(Δ + RTT)` at the measured 3.490×.
- **An RTT cap of 0.1 ms** (the review's suggestion, `art:f33b450f…`, "What must change", item 4a), `Δ + RTT ≤ 0.6 ms`:
  2.85×, and 2.34× and 1.79× under the modelled 4.24× and 5.55×.
- **v1's 1 ms kept, with w widened to 20,992,** the smallest multiple of 64 that keeps 2× at `Δ + RTT = 1.5 ms`: root
  10,476.7 µs, 2.001×, and 1.43× and 1.15× under the same model at that width (4.87× and 6.06×). It would cost about
  1.38× in encode and 1.08–1.27× in decode per payload byte, a new prime (`p = 2^20992 − 27177`, by `find_primes.py`)
  with its own ECPP certificate, a revised wire format (`p2-audit-wire/v1` accepts only `w ≤ 16,448`), and new Lean
  pins: every P2 pin and its row lemmas are stated at w = 16,448, and k = 105 would have to be re-derived.

**Sources.**
- The timing red team: `note:20261004T2102Z-report-relay-p2-redteam-timing`, with its scripts and logs (`regate.py`,
  `regate.log`, `coop.log`, `decode.py`, `find_primes.py`) in
  `art:23a323f313d2193873c59ee8ab59a0f1197167a0b4c6d008346d771c3e72de57`.
- The SeqRoot and RTT measurement: run `r20260929-060834-0ec4`,
  `art:b92b6d56fa12cc8e04b3f6843991c2e76f49366adc89140a96c33842f70b41ed`.
- F2's 1.9 ms derivation: `note:20261004T2102Z-report-relay-p2-deployment-decisions`.
- The review: `art:f33b450f2cd48f6cec2057915ae08b2221a4a6060dc2f9fa38055e8e4d74b98c`.
- Frozen v1: `art:a7a29e84eef9d2584482b7f3e9722af6b579e2984a9633352564c562ce9e2021`.

## 13. Amendment v4 (2026-10-06)

**v4 is Daniel's ruling of 6 Oct 2026, 15:07Z,** relayed by memory-accounting: one hash, SHA-512, with its whole output,
everywhere; SHA-256 retires.

**What changed.** Only the hash, and what its output size moves:

- Every SHA-256 is SHA-512 with all 64 bytes: the W commitment, the verification tree's leaves and nodes, the epoch id,
  the transcript hash and the frame hashes in a record. Every digest field on the wire grows from 32 to 64 bytes, and
  every frame size and offset with it (§4's two tables). The setup salt, the audit id and the nonce stay 32 bytes.
- The leaf hashes the block's `sha512/row/v2` digest R, not the block (§4, "Why this leaf").
- The keys are counter-mode SHA-512 over a prefix padded to whole 128-byte blocks, where v1 read two SHAKE256 streams;
  t's rejection sampling is unchanged. The assumption that names the concrete expander is now SHA-512-CTR/P2-EXP-IO,
  where v1's was "SHAKE256/P2-EXP-IO storage-game idealization".
- The challenge derivation is `verity.ml.randomness` v2 (Appendix A), SHA-512 where v1 was SHA-256.
- The claims P2 cites name `cr/sha-512` where they named `cr/sha-256`.
- Every name with a version moved:

| v1 | v4 |
|---|---|
| protocol `p2/v1` | `p2/v4` |
| `p2-audit-wire/v1`, version byte `0x01` | `p2-audit-wire/v4`, version byte `0x04` |
| `veritor/tagged-sha256/v1\0`, tag `verity-pous/w-commitment/v1` | `verity/tagged-sha512/v1\0`, tag `verity-pous/w-commitment/v2` |
| `verity/pous/p2-setup/v1` | `verity/pous/p2-setup/v4` |
| `frame("verity/pous/p2/expand/v1")` opening the SHAKE256 base | `"verity/pous/p2/keys/v4\0"` opening the key prefix |
| `verity/pous/p2/epoch/v1\0` | `verity/pous/p2/epoch/v4\0` |
| `verity/pous/p2/transcript/v1\0` | `verity/pous/p2/transcript/v4\0` |
| `verity/pous/audit-id/v1` | `verity/pous/p2-audit-id/v4` |
| `verity/pous/audit-challenge/v2` | `verity/pous/p2-audit-challenge/v4` |
| `verity/pous/audit-round/v1` | `verity/pous/p2-audit-round/v4` |
| `p2-test-vectors/v1`, `p2-audit-wire-v1/…` | `p2-test-vectors/v4`, `p2-audit-wire-v4/…` |

**What did not change.** w, p, the profile, B, Q, k, Δ, the RTT cap, the hardware class, the maps, the error allocation,
the frame types and their field order, the failure codes and their precedence, and the state-isolation rules.

**What v4 assumes in place of v1.** `cr/sha-512` (collision resistance of SHA-512) for the commitment and the tree, and
SHA-512-CTR/P2-EXP-IO for the keys. Separately, the memory challenger's VM runs on Nebius, which offers no attesting
hardware, so the deployment cites core's `trusted-nebius-host` assumption (`verity.core.claims`; Lean
`Verity.Assumptions.TrustedNebiusHost`, Daniel's answer of 2026-10-06). Perfect isolation (§4) is about the prover; this
is about the verifier's host.

## 14. Errata (2026-10-07)

Daniel accepted four clarifications on 7 Oct 2026 (21:39Z). They change no rule, value, vector or pin; the reference verifier already behaved this way. They hold for v1.1 and v4 alike. They came from the Lean port of the timed-audit check, which found the prose silent where the code had to choose.

1. Δ and the RTT allowance in `AUDIT_OPEN` are the verifier's own, and no failure code covers them (§ Sequential reveal and deadlines).
2. A setup transcript that ends early is transport code 6 unless a frame already read gave code 1 or 2 (§ Setup commitment, salt and verification root).
3. A registered scheme's profile is `p2-{payload_bytes}B`; `p2-nb` exists only in reduced-width conformance vectors (§ Parties and transport).
4. An `elapsed_ns` outside u64 is the verifier's fault, and the session ends without a `FINAL` (§ Sequential reveal and deadlines).

## Appendix A. `verity.ml.randomness` v2

The derivation the challenges use, byte for byte. `frames(index)` is `frame(p_1) || … || frame(p_m)` for the parts of
an index (one part when the index is not a tuple).

```text
derive(source, domain, ctx) =
  Key(first 32 bytes of SHA-512(
    ASCII("verity/randomness/derive/v2\0")
    || frame(source) || frame(domain)
    || frame(frame(k_1) || frame(ctx[k_1]) || … || frame(k_m) || frame(ctx[k_m]))   keys k_1 < … < k_m, sorted
  ))
key.stream(index, N) = the first N bytes of block_0 || block_1 || …,
  block_c = SHA-512(ASCII("verity/randomness/stream/v2\0") || key[32] || frame(frames(index)) || u64be(c))
key.uniform(n, j): width = ⌈(bitlen(n) + 64) / 8⌉ bytes; read consecutive width-byte big-endian words from
  key.stream(("u", j), …) and return the first word x below 2^(8·width) − (2^(8·width) mod n), as x mod n
```

`source` is nonempty bytes; `domain` matches `verity/[a-z0-9-]+/[a-z0-9-]+/v[0-9]+`; context keys are strings and values
bytes, strings or nonnegative integers. v1 is the same with `…/v1\0` tags and SHA-256 (all 32 bytes as the key, 32-byte
stream blocks).
