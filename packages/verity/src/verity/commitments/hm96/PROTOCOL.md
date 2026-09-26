# `hm96-sha256/v1`: Halevi–Micali hiding leaves on SHA-256

An unsalted leaf digest such as vllm-v1's position leaf or frame-v3's `sha256/row/v1` is a deterministic function of the value. Anyone
holding the digest can search a low-entropy value, for example a token id, however zero-knowledge the proof over it is. `hm96-sha256/v1`
is a leaf rule that wraps any SHA-256 leaf digest so that the published data hides the value statistically. Binding rests on the
collision resistance of SHA-256 alone.

- **Reference.** `verity.commitments.hm96`, stdlib only. `Hm96Sha256(base, key)` configures it over a `vllm-v1` configuration.
- **Vectors.** `vectors.json`, written by `python -m verity.commitments.hm96.vectors`. `--check` exits nonzero if the file is stale.
- **Conformance.** `packages/verity/tests/commitments/test_hm96.py` checks the reference against the vectors and a bit-level matrix
  product; `integrations/vllm/tests/commit/test_hiding.py` checks the host committer's batch path against both.
- **Source.** Halevi and Micali, "Practical and Provably-Secure Commitment Schemes from Collision-Free Hashing", CRYPTO 1996 (HM96):
  their first scheme, with the affine universal family below, applied to a digest of the value (their O(k) variant). The hiding bound
  uses the generalized leftover hash lemma of Dodis, Ostrovsky, Reyzin and Smith, "Fuzzy Extractors", SIAM J. Computing 38(1), 2008
  (DORS08), Lemmas 2.2(b) and 2.4.

## 1. The leaf

~~~text
x    = SHA-256( inner.prefix ‖ value ‖ inner.suffix )      the port's unsalted SHA-256 leaf digest       private
y    = 128 bytes from the operating system's generator     the salt, fresh per leaf, stored with it     private
c    = SHA-256( SALT_PREFIX ‖ y )
b    = x ⊕ M·y                                             M = M(key), a 256 × 1024 matrix over GF(2)
leaf = SHA-256( leaf_prefix(key) ‖ b ‖ c )

SALT_PREFIX      = "verity/hm96-sha256/salt/v1\0"                  zero-padded to 64 bytes
leaf_prefix(key) = "verity/hm96-sha256/leaf/v1\0" ‖ SHA-256(key)   zero-padded to 64 bytes
~~~

- **The inner layout** is any SHA-256 `LeafLayout` the verifier derives for the position: a fixed prefix, suffix and value length. The
  host committer's opt-in uses vllm-v1's position leaf (`pos_leaf_layout`); a circuit prefers a block-aligned prefix such as
  `sha256/row/v1`'s, since its midstate is a constant.
- **`b ‖ c` is the commit string**, 64 bytes. A backend proves it and publishes it, and the verifier computes the tree leaf natively.
- **An opening** is `(value, y)`. The verifier recomputes `x`, `c`, `b` and the leaf, and checks the path with the base scheme's tree
  (vllm-v1's node and lift rule and root bindings).
- **Bits.** Bit t of a byte string is bit `t mod 8` of byte `⌊t/8⌋`. `y_j`, `x_i` and `b_i` are read this way, and `M·y` is written
  this way into 32 bytes.

## 2. The key and the matrix

- **The key** is 160 bytes: bits `k_0 … k_1278`, with bit 1279 zero, so every matrix has one encoding.
- **The matrix** is `M[i][j] = k_{i+j}` for `i < 256` and `j < 1024`, a Hankel matrix (a Toeplitz matrix with its columns reversed).
  So `(M·y)_i = ⊕_j k_{i+j} y_j`.
- **The family is universal.** For a nonzero d, the map `key ↦ M·d` is linear, with rows `d ≪ i` for `i < 256`. Those rows are
  independent, so `M·d` is uniform over 256 bits when the key is. For `y ≠ y'`, `Pr[M·y = M·y'] = 2^-256`.
- **The pinned key**, `DEFAULT_KEY`, is `SHA-256(KEY_LABEL ‖ u32be(i))` for i = 0…4, concatenated, with bit 1279 cleared, where
  `KEY_LABEL = "verity/hm96-sha256/key/v1\0"`. It is a circuit constant: a gadget's XOR network is fixed by it.
- **A per-run key.** A committer may draw its own key per run from the operating system's generator (`fresh_key`). Hiding then needs
  no step beyond the theorem in section 3. A circuit takes that key as a public parameter.
- **The key is public and bound.** Every leaf binds `SHA-256(key)` through its prefix, and `scheme_digest(key)` names the configured
  rule for a root binding or statement. The verifier takes the key from its own statement, never from an opening. A key chosen after
  commitment would let a committer fit an opening to any value.

## 3. Hiding

**Theorem.** Let the key be uniform, and the salts `y_1 … y_N` independent, uniform and independent of the key. For any two vectors of
inner digests `x⃗` and `x⃗'`, the published data `(key, (b_i, c_i)_i)`, and therefore every leaf, tree and root computed from it, are
within statistical distance `N · 2^-256`.

**Proof.**
1. `c = SHA-256(…‖ y)` takes at most 2^256 values, so the average min-entropy of y given c is at least `1024 − 256 = 768` bits
   (DORS08 Lemma 2.2(b)).
2. By the generalized leftover hash lemma (DORS08 Lemma 2.4) over the universal family of section 2, for one leaf,
   `SD((key, M·y, c), (key, U, c)) ≤ ½ · √(2^(256 − 768)) = 2^-257`, with U uniform over 256 bits.
3. Given the key, the leaves are independent, so the distance of the product is at most the sum. For N leaves under one key,
   `SD((key, (M·y_i, c_i)_i), (key, (U_i, c_i)_i)) ≤ N · 2^-257`.
4. `b_i = x_i ⊕ M·y_i`, and XORing `x_i` maps the ideal distribution to itself. So the data for `x⃗` is within `N · 2^-257` of an
   ideal distribution that doesn't depend on the digests, and two such vectors are within `N · 2^-256`. ∎

**With the pinned key.** Write `δ(key)` for the distance in step 3 at a fixed key. It doesn't depend on the digests, and its mean over
keys is at most `N · 2^-257`. By Markov's inequality, for all but a `2^-t` fraction of keys, every pair of digest vectors is within
`N · 2^(t−256)`. At t = 64 that is `N · 2^-192`. The pinned key is derived from a public label, so it wasn't chosen to be bad. Treating
it as a uniform draw is the one step beyond the theorem, a common-reference-string step; a per-run key removes it.

**Sizes.** `|y| = 1024 = |x| + |c| + 512` bits. With the pinned key at t = 64, `N = 2^64` leaves under one key stay within 2^-128;
with a per-run key they stay within 2^-192. `statistical_distance_log2(N, key_failure_log2=t)` computes the bound.

**Against HM96's generic theorem.** HM96 Theorem 1 covers any universal family, by choosing h with `h(y) = x`: `|y| = 4k + 2·256 + 4`
bits gives `2^-k`, so 1024 bits give k = 127 per commitment. With the affine family the offset b is a one-time pad on `M·y`, which is
why the direct bound above is tighter.

**What the bound needs:**
- a salt per leaf from the operating system's generator, never reused: two leaves with one salt reveal `x ⊕ x'`;
- salts that aren't expanded from a seed, since that makes hiding computational, resting on the generator;
- a key independent of the salts.

As everywhere in this repository, the operating system's generator is modelled as uniform.

**Public:** the key, the commit string `b ‖ c` of every row a proof touches, the leaf digests and the tree. **Private:** the value,
x and y. Opening a leaf reveals its value and salt. Salts are independent, so the bound still covers the unopened leaves.

## 4. Binding

For a fixed key and inner layout, two different openings `(v, y) ≠ (v', y')` of one leaf give a SHA-256 collision:
- if `(b, c) ≠ (b', c')`, the leaf hashes collide, since the prefix is the same;
- if `(b, c) = (b', c')` and `y ≠ y'`, the salt hashes collide;
- if `(b, c) = (b', c')` and `y = y'`, then `x = b ⊕ M·y = x'` with `v ≠ v'`, and the inner digests collide (same prefix, suffix and
  length).

This holds for every key, including one the committer chose. A key substituted after commitment is excluded by the leaf's key digest
(a collision again). The only assumption is collision resistance of SHA-256, `cr/sha-256`; there is no random oracle anywhere.

## 5. What a backend proves

Per leaf, the gadget takes the private value and salt:
- it computes x exactly as for the unsalted leaf;
- it computes c from the constant midstate of `SALT_PREFIX`;
- it computes `b = x ⊕ M·y`;
- it publishes `b ‖ c`.

The verifier computes `tree_leaf(key, b ‖ c)` natively and checks the tree with the base scheme. `HidingLayout(inner, key)` is what the
gadget and the statement check read: `inner`, `salt_layout` (SHA-256, `SALT_PREFIX`, 128 bytes) and `rows(key)` for the XOR network.

**Circuit cost beyond the unsalted inner digest:**
- **c:** 3 private SHA-256 compressions. y fills two blocks, and the padding block follows. That is 3 × 22,696 = 68,088 ANDs.
- **b:** XOR only, `Σ_i weight(row i)` two-input XORs: 135,803 for `DEFAULT_KEY` (about 256 · 1024 / 2), and no AND.
- **The leaf hash** is native at the verifier, so it costs the circuit nothing.

**AND model.** A 32-bit addition costs 31 ANDs (ripple carry), and Ch and Maj cost 32 each. A SHA-256 compression is 64 × (7 additions
+ Ch + Maj) + 48 × 3 additions + 8 additions = 22,696 ANDs. A BLAKE3 compression is 7 × 8 × 6 additions = 10,416 ANDs. A public
64-byte prefix block is a constant midstate; every other compression is private.

## 6. Cost per row

These are analytic, one row of R bytes per leaf, against today's keyed-BLAKE3 row leaf (`blake3-keyed/row/v2`) and today's serving
leaf (vllm-v1's position leaf):
- *stored* is the bytes kept beside the row;
- *hashed* is the payload bytes through the hash, excluding constant prefix blocks;
- *compressions* are native;
- *ANDs* are in-circuit.

| R | stored | hashed (hm96 over pos-leaf) | SHA-256 compressions (pos-leaf → hm96) | ANDs, keyed-BLAKE3 row | ANDs, hm96 | hm96 ÷ BLAKE3 row |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 128 (3,200%) | 222 | 1 → 6 | 10,416 | 90,784 | 8.72× |
| 256 | 128 (50%) | 474 | 5 → 10 | 41,664 | 181,568 | 4.36× |
| 1,536 | 128 (8.3%) | 1,754 | 25 → 30 | 260,400 | 635,488 | 2.44× |
| 3,072 | 128 (4.2%) | 3,290 | 49 → 54 | 520,800 | 1,180,192 | 2.27× |
| 4,096 | 128 (3.1%) | 4,314 | 65 → 70 | 697,872 | 1,543,328 | 2.21× |
| 8,192 | 128 (1.6%) | 8,410 | 129 → 134 | 1,406,160 | 2,995,872 | 2.13× |

- **Where the cost goes.** Most of the gap to the BLAKE3 row is SHA-256 hashing the row itself, since a compression costs 2.18× a
  BLAKE3 compression. The hiding part is a fixed 3 compressions (68,088 ANDs) and 135,803 XORs per row. That is +6.1% over an unsalted
  SHA-256 row digest at R = 3,072, and +2.3% at R = 8,192.
- **Proofs publish 64 bytes per row** (`b ‖ c`), not 32. An opening adds the 128-byte salt.
- **Serving** adds 5 SHA-256 compressions per row natively (3 for the salt, 2 for the leaf) and one 256 × 1024 GF(2) product. The
  product costs 128 table lookups and 32-byte XORs with `byte_tables(key)`.
- **Measured costs.** `benchmarks/commitments/hiding_leaf_cost.py` measures host time per row and prints this table.

## 7. Proof-internal leaves

A proof's own Merkle trees, such as Flock's Ligerito column trees, hide their unopened leaves with the same function:
- the inner digest is the backend's SHA-256 column digest;
- each leaf gets its own salt;
- the key is the pinned one, or a per-proof key fixed in the transcript before the root;
- the leaf is `tree_leaf`.

Opening a column reveals the column and its salt. Lane flock-circuit's `flock-leaf/hm96-sha256` reserves exactly these sizes: a
128-byte salt and a 160-byte key. Salts drawn from a seeded stream cipher, rather than from the operating system per salt, make the
hiding computational in that generator (section 3).

## 8. Hashes

Table 1 lists one hash, of class standard: SHA-256, for the inner digest, the salt digest and the leaf, under collision resistance.
Hiding is statistical and rests on no hash property.
