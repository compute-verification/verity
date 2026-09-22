# BabyBear transcript vectors (for `lane/a-gpu`)

`transcript_vu4.json` and `transcript_vu64.json` are the complete Fiat–Shamir transcripts of the Candidate A prover
built with `--features babybear` (PROTOCOL.md §13) on the first 4 and 64 VUs of `bench-instances/v1 vu-k1536`
(PCS mode, `NON_ZK_PROOF_DIAGNOSTIC`), written by

~~~sh
verity-gkr dump-transcript --dir EXPORT_DIR --vus 4 --out transcript_vu4.json    # EXPORT_DIR from
uv run --package verity-numerical python -m verity_numerical.gkr_export.vu export --field babybear --tier vu-k1536 --lo 0 --hi 64 --out EXPORT_DIR
~~~

(`FIELD=babybear TRANSCRIPTS=1 bash backends/gkr/run_measure_vu.sh` does both; run r20260922-070527-a263, commit
1906a36.) `python3 replay.py` re-derives every state and challenge of both files from the absorbed bytes with `hashlib`
alone — the conventions below, executable. B = 4: 1,079 absorbs, 1,426 challenges, 16,393 proof messages; B = 64:
1,604 / 1,991 / 18,493. A GPU prover
that produces the same proof messages in the same order must reproduce every `state` and every `challenge` below
bit for bit; the first divergence names the phase (the `label` events).

## Schema (`verity-gkr/transcript/v1`)

~~~text
field          "babybear", modulus 2013265921, extension "babybear^6 (X^6-22)", degree 6, base_bytes 4, hash "sha256"
domain         "verity-gkr/candidate-a/babybear6/v1"
mode           "pcs"; ligero {k: 4096, n: 16384, t: 192}
vus, steps     B and 96; segments [{name, circuit_sha256, units, columns}] (units = 96·B rows, epilogue = B rows)
public_y16     the B recorded output words (base-field integers)
root           Ligero Merkle root (hex, 32 bytes)
events         the transcript, in order (below)
proof_msgs     every extension element the prover sends, in order ([c0..c5] as integers)
proof_bytes    the Ligero opening (hex): per opened column, rows × 4-byte LE values, then the Merkle path (depth 14 × 32 bytes)
~~~

Events are one of

~~~text
{"label": "..."}                          the phase the following events belong to (informational, not absorbed)
{"absorb": HEX, "state": HEX}            state <- SHA-256(state || bytes); `state` is the value after the absorb
{"challenge": [c0,...,c5], "ctr": N}      one extension challenge; the squeeze counter before it (6 squeezes)
~~~

The first event is the domain absorb: `state_0 = SHA-256(domain)` (the `absorb` field shows the domain bytes).

## Byte-exact conventions

* **Base field element**: canonical integer in `[0, p)`, `p = 2^31 − 2^27 + 1`, serialised as **4 bytes little-endian**.
* **Extension element** `x = Σ c_i X^i` over `F_p[X]/(X^6 − 22)`: 24 bytes, `c_0` first, each coordinate 4 bytes LE.
  A vector of extension elements is the concatenation in vector order (`absorb_exts`). This is the polynomial of
  `backends/gkr/tensor/reference.py` (`W = 22`); coordinate order matches its `[c0, ..., c5]`.
* **Absorb**: `state <- SHA-256(state || bytes)` (32-byte state, no length prefix, no padding beyond SHA-256's own).
* **Challenge**: each coordinate is `SHA-256(state || "chal" || ctr_le64)` with `ctr` a `u64` LE counter that
  starts at 0 for the transcript and increments per digest; the 32-byte digest is read as eight 4-byte LE words and
  the **first word `< p`** is the coordinate (rejection sampling: if all eight are `>= p`, the next counter value is
  tried). Six coordinates = six digests = one challenge; `slots` (`rounds.protocol`) counts challenges. Absorbing
  does *not* reset the counter; the state is not modified by squeezing.
* **Preamble** (all absorbs before the first challenge): for each segment, `sha256(circuit.txt bytes)` then
  `units` as `u64` LE; then the Ligero root (32 bytes); then `steps` (96) as `u64` LE and the `B` public `y16`
  words concatenated (4 bytes LE each).
* **Table challenges**: for each table in circuit order (declared: POW, ALIGN, LEAD, NORM; then the implicit range
  tables narrowest first: R7, R8, R9, R14, R16): `z`, and `beta` for multi-column tables (beta = 1 for the R tables,
  no squeeze).
* **Segment points**: per segment `z0` (`s_out` challenges of the output layer) then `gamma` (`ceil(log2 units)`).
* **logUp per table** (PROTOCOL.md §4.2): absorb `q_root`; challenge `lambda`; then for each level `k = 1..n_T`:
  `k` rounds of (absorb the degree-3 round polynomial as its 4 evaluations `h(0), h(1), h(2), h(3)`; challenge
  `r_k`), then absorb `(p0, p1, q0, q1)` (4 elements), challenge `mu`, challenge `lambda`.
* **Checker GKR per segment, per layer** (§4.1): phase 1 — `b` rounds of (absorb `h(0..3)`, challenge); phase 2a —
  `s_in` rounds of (absorb `s(0), s(1), s(2)`, challenge), absorb `(V(l*), P(l*))`; phase 2b — `s_in` rounds, absorb
  `V(r*)`; then challenge `alpha` (except after the last layer).
* **Opening**: challenges `rho_1..rho_m` (`m = 2·segments + tables + 1`), challenge `geo`; absorb the Ligero row
  combination challenge `r` (one challenge; coefficients `r^i`), absorb `w` (k = 4096 elements) and `q` (2k − 1
  coefficients), then `t = 192` column indices: repeat (challenge `c`; `j = c_0 mod n`; skip if already chosen).
* **Merkle tree**: leaf `j` = `SHA-256(column_j)` with the column's `rows` base values, 4 bytes LE each, top row
  first; node = `SHA-256(left || right)`; `n = 16384` leaves, depth 14; `root` = the single top node. Paths list the
  sibling at each level from the leaf up.
* **Reed–Solomon** (`ligero.rs::encode`): a row of `k = 4096` base values is read as the *evaluations* of a
  polynomial of degree `< k` on `H = <omega_k>` in natural order (`omega_k = omega_n^4`); the codeword is its
  `n = 16384` evaluations on `D = <omega_n>` in natural order (inverse NTT over `H`, zero-pad the coefficients,
  NTT over `D`). `omega_n = g^((p-1)/n)` with generator `g = 31` (`Fp::root_of_unity`): `omega_{2^27} = 31^15`,
  and `omega_{2^k}` is its repeated square. Bit reversal is internal to the butterflies; inputs and outputs are in
  natural order. The committed vector `X` (segment rows row-major, then every table's multiplicities) is cut into
  rows of `k`; the last row is zero-padded.
* **Proof messages** (`proof_msgs`) are the values the prover *sends*; every one of them is also absorbed, so the
  `events` list is self-contained. `proof_bytes` (opened columns + paths) is *not* absorbed: the verifier checks it
  against `root`.
