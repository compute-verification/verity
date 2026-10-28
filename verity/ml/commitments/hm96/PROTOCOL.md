# Halevi–Micali hiding leaves: `hm96-sha256/v1` and `hm96-sha512/v1`

An unsalted leaf digest such as vllm-v1's position leaf or frame-v3's `sha256/row/v1` is a deterministic function of the value. Anyone
holding the digest can search a low-entropy value, for example a token id, however zero-knowledge the proof over it is. The hm96 leaf
rules wrap a SHA-256 or SHA-512 leaf digest so that the published data hides the value, statistically given uniform salts (section 3;
with the pinned key the bound holds except for a 2^-64 fraction of keys). Binding rests on the collision resistance of the one hash
alone. `hm96-sha512/v1` is the rule for SHA-512 data commitments (SHA-512 on every commitment path); both
instances have the same hiding bound.

- **Reference.** `verity.ml.commitments.hm96`, stdlib only: the instances `SHA256` and `SHA512`, and `HidingLayout`, one leaf
  over its inner layout. The module-level names are `hm96-sha256/v1`'s.
  `verity.ml.commitments.hm96.batch` (numpy) computes the masks and tree leaves of many rows at once, byte-identical to it.
- **Vectors.** `vectors.json` (`hm96-sha256/v1`) and `vectors_sha512.json` (`hm96-sha512/v1`), written by
  `python -m verity.ml.commitments.hm96.vectors`. `--check` exits nonzero if either is stale.
- **Conformance.** `verity/ml/commitments/tests/test_hm96.py` checks the reference against the vectors and a bit-level matrix
  product; `test_hm96_batch.py` checks `batch` against the reference; `integrations/vllm/tests/commit/test_hiding.py` checks the
  host committer's batch path against both.
- **Source.** Halevi and Micali, "Practical and Provably-Secure Commitment Schemes from Collision-Free Hashing", CRYPTO 1996 (HM96):
  their first scheme, with the affine universal family below, applied to a digest of the value (their O(k) variant). The hiding bound
  uses the generalized leftover hash lemma of Dodis, Ostrovsky, Reyzin and Smith, "Fuzzy Extractors", SIAM J. Computing 38(1), 2008
  (DORS08), Lemmas 2.2(b) and 2.4.

## 1. The leaf

With H the instance's hash, n its output bits and l = 2n + 512 the salt bits:

~~~text
x    = H( inner.prefix ‖ value ‖ inner.suffix )             the port's unsalted leaf digest (H)       private
y    = l/8 bytes from the operating system's generator     the salt, fresh per leaf, stored with it  private
c    = H( salt_prefix ‖ y )
b    = x ⊕ M·y                                             M = M(key), an n × l matrix over GF(2)
leaf = H( leaf_prefix(key) ‖ b ‖ c )

salt_prefix      = salt tag                  zero-padded to one block of H
leaf_prefix(key) = leaf tag ‖ H(key)         zero-padded to one block of H
~~~

| | `hm96-sha256/v1` | `hm96-sha512/v1` |
|---|---|---|
| H, block | SHA-256, 64 bytes | SHA-512, 128 bytes |
| n: bits of x, b, c and the leaf | 256 | 512 |
| l: salt bits (bytes) | 1,024 (128) | 1,536 (192) |
| key bits (bytes); the top bit is zero | 1,279 (160) | 2,047 (256) |
| commit string `b ‖ c` | 64 bytes | 128 bytes |
| salt tag | `verity/hm96-sha256/salt/v1\0` | `verity/hm96-sha512/salt/v1\0` |
| leaf tag | `verity/hm96-sha256/leaf/v1\0` | `verity/hm96-sha512/leaf/v1\0` |
| key label | `verity/hm96-sha256/key/v1\0` | `verity/hm96-sha512/key/v1\0` |

- **The inner layout** is any `LeafLayout` with hash H that the verifier derives for the position: a fixed prefix, suffix and value
  length. The vLLM integration's SHA-256 host committer's opt-in uses its position leaf,
  `H("verity/pos-leaf/v0" ‖ u64be(n) ‖ value)`. A circuit prefers a block-aligned prefix such as `sha256/row/v1`'s,
  `sha512/row/v1`'s or a bit row's (`sha512/row/v2`), since its midstate is a constant. Under frame-v3 the tree leaf's schema names the inner one: `hm96-sha512/row/v1`, or `hm96-sha512/row/v2` over a bit
  row (`frame_v3/PROTOCOL.md` §6).
- **`b ‖ c` is the commit string.** A backend proves it and publishes it, and the verifier computes the tree leaf natively.
- **An opening** is `(value, y)`. The verifier recomputes `x`, `c`, `b` and the leaf, and checks the path with the base scheme's tree
  and root bindings (frame-v3-sha512's for `hm96-sha512/row/v2`).
- **Bits.** Bit t of a byte string is bit `t mod 8` of byte `⌊t/8⌋`. `y_j`, `x_i` and `b_i` are read this way, and `M·y` is written
  this way into n/8 bytes.

## 2. The key and the matrix

- **The key** has `n + l − 1` bits `k_0 …`, and the bit above them is zero, so every matrix has one encoding.
- **The matrix** is `M[i][j] = k_{i+j}` for `i < n` and `j < l`, a Hankel matrix (a Toeplitz matrix with its columns reversed). So
  `(M·y)_i = ⊕_j k_{i+j} y_j`.
- **The family is universal.** For a nonzero d, the map `key ↦ M·d` is linear, with rows `d ≪ i` for `i < n`. Those rows are
  independent, so `M·d` is uniform over n bits when the key is. For `y ≠ y'`, `Pr[M·y = M·y'] = 2^-n`.
- **The key is fixed and public.** Each instance's configured key is its pinned key (section 2a), a circuit
  constant that fixes a gadget's XOR network. `fresh_key` draws a uniform key, which is the theorem's setting; it exists for tests.
- **The key is bound.** Every leaf binds `H(key)` through its prefix, and `scheme_digest(key)` names the configured rule for a root
  binding or statement. The verifier takes the key from its own statement, never from an opening. A key chosen after commitment
  would let a committer fit an opening to any value.

## 2a. How the pinned keys are derived: the setup claim

~~~text
key = H( key label ‖ u32be(0) ) ‖ H( key label ‖ u32be(1) ) ‖ …   truncated to the key's bytes, the top bit cleared

hm96-sha256/v1: SHA-256, i = 0 … 4 (160 bytes)        hm96-sha512/v1: SHA-512, i = 0 … 3 (256 bytes)
~~~

- **The setup is transparent.** Nothing is chosen and nobody holds a trapdoor. The label is a fixed public string, and anyone
  re-derives the key (`Instance.key_from_label`; the vectors spell out both keys).
- **What the hiding claim assumes about the key.** The pinned-key bound of section 3 holds for all but a `2^-64` fraction of keys.
  Using a derived key assumes it isn't in that fraction: a counter-mode hash output over a public label is a typical draw of the
  family.
  - This is the assumption `hash-derived-key` in `verity.core.claims`, a standard nothing-up-my-sleeve assumption.
  - It is used in the `common-reference-string` model.
  - It is not a collision-resistance assumption, and binding doesn't use it.
- **One key per instance, for every leaf, run and deployment.** N in section 3 counts every leaf ever committed under that key.

## 3. Hiding

**Theorem.** Let the key be uniform, and the salts `y_1 … y_N` independent, uniform and independent of the key. For any two vectors of
inner digests `x⃗` and `x⃗'`, the published data `(key, (b_i, c_i)_i)`, and therefore every leaf, tree and root computed from it, are
within statistical distance `N · 2^((2n − l)/2) = N · 2^-256` for both instances.

**Proof.**
1. `c = H(…‖ y)` takes at most 2^n values, so the average min-entropy of y given c is at least `l − n` bits (DORS08 Lemma 2.2(b)).
2. By the generalized leftover hash lemma (DORS08 Lemma 2.4) over the universal family of section 2, for one leaf,
   `SD((key, M·y, c), (key, U, c)) ≤ ½ · √(2^(n − (l − n))) = 2^((2n − l)/2 − 1) = 2^-257`, with U uniform over n bits.
3. Given the key, the leaves are independent, so the distance of the product is at most the sum. For N leaves under one key,
   `SD((key, (M·y_i, c_i)_i), (key, (U_i, c_i)_i)) ≤ N · 2^-257`.
4. `b_i = x_i ⊕ M·y_i`, and XORing `x_i` maps the ideal distribution to itself. So the data for `x⃗` is within `N · 2^-257` of an
   ideal distribution that doesn't depend on the digests, and two such vectors are within `N · 2^-256`. ∎

**With the pinned key.** Write `δ(key)` for the distance in step 3 at a fixed key. It doesn't depend on the digests, and its mean over
keys is at most `N · 2^-257`. By Markov's inequality, for all but a `2^-t` fraction of keys, every pair of digest vectors is within
`N · 2^(t−256)`. At t = 64 that is `N · 2^-192`. Treating the pinned key as a uniform draw is the one step beyond the theorem: a
common-reference-string step, justified by the derivation of section 2a.

**Sizes.** `l = 2n + 512` bits gives the same bound for both instances. With the pinned key at t = 64, `N = 2^64` leaves under one
key stay within 2^-128. `Instance.statistical_distance_log2(N, key_failure_log2=t)` computes the bound.

**Against HM96's generic theorem.** HM96 Theorem 1 covers any universal family, by choosing h with `h(y) = x`: `|y| = 4k + 2n + 4`
bits gives `2^-k`. So 1024 bits give k = 127 for SHA-256, and 1536 bits give k = 127 for SHA-512. With the affine family the offset b
is a one-time pad on `M·y`, which is why the direct bound above is tighter.

**What the bound needs:**
- uniform, independent salts, one per leaf and never reused: two leaves with one salt reveal `x ⊕ x'`;
- a key independent of the salts.

**Randomness.** Hiding is statistical given uniform salts. Real salts come from a cryptographic generator: the operating system's
(`os.urandom`, itself a ChaCha20-based generator on Linux), or a stream seeded from it, such as a prover's ChaCha20 stream. The bound
treats either as uniform, the modelling convention this repository uses everywhere; neither is a different assumption. What differs is
custody: salts drawn per leaf are stored beside the values, and salts expanded from one seed are exposed together if that seed is.

**Public:** the key, the commit string `b ‖ c` of every row a proof touches, the leaf digests and the tree. **Private:** the value,
x and y. Opening a leaf reveals its value and salt. Salts are independent, so the bound still covers the unopened leaves.

## 4. Binding

For a fixed key and inner layout, two different openings `(v, y) ≠ (v', y')` of one leaf give a collision of H:
- if `(b, c) ≠ (b', c')`, the leaf hashes collide, since the prefix is the same;
- if `(b, c) = (b', c')` and `y ≠ y'`, the salt hashes collide;
- if `(b, c) = (b', c')` and `y = y'`, then `x = b ⊕ M·y = x'` with `v ≠ v'`, and the inner digests collide (same prefix, suffix and
  length).

This holds for every key, including one the committer chose. A key substituted after commitment is excluded by the leaf's key digest
(a collision again). The only assumption is collision resistance of H: `cr/sha-256` for `hm96-sha256/v1`, `cr/sha-512` for
`hm96-sha512/v1`. There is no random oracle anywhere.

## 5. What a backend proves

Per leaf, the gadget takes the private value and salt:
- it computes x exactly as for the unsalted leaf;
- it computes c from the constant midstate of the salt prefix;
- it computes `b = x ⊕ M·y`;
- it publishes `b ‖ c`.

The verifier computes `tree_leaf(key, b ‖ c)` natively and checks the tree with the base scheme. `HidingLayout(inner, key, instance)`
is what the gadget and the statement check read: `inner`, `salt_layout` (H, the salt prefix, l/8 bytes) and `instance.rows_of(key)`
for the XOR network.

**The key is the statement's, never a witness.** The gadget takes the key as a circuit constant, or as a public input the verifier
checks against `scheme_digest`. A prover free to choose the key could solve `M_k'·y = b ⊕ x'` for a key k' (n linear equations in
`n + l − 1` unknowns) and open a committed `b ‖ c` to any `x'`. The leaf's key digest stops that only where the verifier recomputes the
leaf with the key the circuit used.

**Circuit cost beyond the unsalted inner digest:**

| | `hm96-sha256/v1` | `hm96-sha512/v1` |
|---|---|---|
| c: private compressions (y, then the padding) | 3 (y fills two 64-byte blocks, and padding takes a third) | 2 (192 + 17 bytes fit two 128-byte blocks) |
| c: ANDs | 3 × 22,696 = 68,088 | 2 × 58,120 = 116,240 |
| b: two-input XORs for the pinned key, `Σ_i weight(row i)` | 135,803 | 400,193 |
| the leaf hash | native at the verifier | native at the verifier |

**AND model (ripple carry).**
- A w-bit addition costs w − 1 ANDs, and Ch and Maj cost w each.
- A SHA-256 compression is 64 × (7 additions + Ch + Maj) + 48 × 3 additions + 8 additions at w = 32: 22,696 ANDs per 64-byte block.
- A SHA-512 compression is 80 × (7 additions + Ch + Maj) + 64 × 3 additions + 8 additions at w = 64: 58,120 ANDs per 128-byte block,
  which is 1.28× SHA-256 per byte.
- A BLAKE3 compression is 7 × 8 × 6 additions at w = 32: 10,416 ANDs.
- A public prefix block is a constant midstate; every other compression is private.

## 6. Cost per row

These are analytic, one row of R bytes per leaf, against the archived keyed-BLAKE3 row leaf (`blake3-keyed/row/v2`), for comparison:
- *stored* is the salt bytes kept beside the row;
- the *ANDs* are in-circuit, over the block-aligned row digest (`sha256/row/v1` or `sha512/row/v1`; a bit row of R bytes costs the
  same).

| R | stored, sha256 | stored, sha512 | ANDs, keyed-BLAKE3 row | ANDs, hm96-sha256 | ANDs, hm96-sha512 | sha512 ÷ BLAKE3 row | sha512 ÷ sha256 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 128 | 192 | 10,416 | 90,784 | 174,360 | 16.7× | 1.92× |
| 256 | 128 (50%) | 192 (75%) | 41,664 | 181,568 | 290,600 | 6.97× | 1.60× |
| 1,536 | 128 (8.3%) | 192 (12.5%) | 260,400 | 635,488 | 871,800 | 3.35× | 1.37× |
| 3,072 | 128 (4.2%) | 192 (6.3%) | 520,800 | 1,180,192 | 1,569,240 | 3.01× | 1.33× |
| 4,096 | 128 (3.1%) | 192 (4.7%) | 697,872 | 1,543,328 | 2,034,200 | 2.91× | 1.32× |
| 8,192 | 128 (1.6%) | 192 (2.3%) | 1,406,160 | 2,995,872 | 3,894,040 | 2.77× | 1.30× |

- **Where the cost goes.** Most of the gap to the BLAKE3 row is the collision-resistant hash hashing the row itself. The hiding part
  is fixed per row: 68,088 ANDs for SHA-256, and 116,240 for SHA-512, plus the XORs. That is +6.1% over an unsalted SHA-256 row digest
  at R = 3,072, and +8.0% over an unsalted SHA-512 row digest.
- **Proofs publish** `b ‖ c` per row: 64 bytes for SHA-256, 128 for SHA-512. An opening adds the salt.
- **Serving** adds per row, natively:
  - 5 SHA-256 compressions (3 for the salt, 2 for the leaf), or 4 SHA-512 compressions (2 and 2);
  - one n × l GF(2) product, which costs l/8 table lookups and n/8-byte XORs with `byte_tables(key)`.
- **Measured costs.** `benchmarks/commitments/hiding_leaf_cost.py` measures host time per row and prints the full table, with
  compressions per rule.

## 7. Proof-internal leaves

A proof's own Merkle trees, such as Flock's Ligerito column trees, hide their unopened leaves with the same function:
- the inner digest is the backend's column digest under H;
- each leaf gets its own salt;
- the key is the instance's pinned key, fixed before the root;
- the leaf is `tree_leaf`.

Opening a column reveals the column and its salt. `flock-leaf/hm96-sha256` reserves exactly `hm96-sha256/v1`'s
sizes: a 128-byte salt and a 160-byte key. `hm96-sha512/v1` needs 192 and 256. Salts from the prover's seeded stream, like the
operating system's, are modelled as uniform, so the leaves hide statistically given uniform salts (section 3). The key is fixed
before the root and is never a witness (section 5).

## 8. Hashes and claims

Table 1 lists one hash per instance, of class standard, under collision resistance: SHA-256 for `hm96-sha256/v1` (`cr/sha-256`), and
SHA-512 for `hm96-sha512/v1` (`cr/sha-512`). It covers the inner digest, the salt digest and the leaf.

Hiding rests on no hash property. It is statistical given uniform salts, and its bound depends on the key:
- a uniform key: `N · 2^-256` for N leaves;
- the pinned key, which every configuration uses: `N · 2^-192`, except for a `2^-64` fraction of keys. That condition is the
  common-reference-string step of section 3, justified by the derivation of section 2a.

`verity.core.claims` names these:
- the guarantee `statistical-hiding`;
- the model `common-reference-string`;
- the setup assumption `hash-derived-key`.

A Table 1 row that cites hm96 hiding under the pinned key cites all three.
