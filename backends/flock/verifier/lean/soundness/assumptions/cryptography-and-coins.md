# Cryptographic assumptions and coins

- **Collision resistance of SHA-512** (`cr/sha-512`) for the Merkle openings (spec §14, with SHA-512 leaves and nodes).
  It is the only hash assumption of the compiled theorem: plain collision resistance, for every classical prover.
  The audit's value layer adds A2 (`a2-sha512-expected-time-cr.md`).
  `../DESIGN.md` §1 explains why SHA-512 rather than SHA-256, and how the bound depends on the hash's width.
- **No assumption for the rounds.** The coin server keeps each round's bytes, then draws that round's coins, and the
  replay compares bytes. This replaces the SHA-256 round digests. Keeping the bytes costs about the proof size, 0.7 MB
  per rep.
- **Statement binding** (the statement digest, Σ and `link_sha256`). These are in round 0's kept bytes, fixed before any
  coin. Their own collision resistance matters to what the statement refers to (registered data, linked proofs), not to
  this theorem.
- **`os` coins** (the default). Each round's coins are uniform and unknown to the prover before it sent the round. They
  come from the coin server's `/dev/urandom` (spec §7.1). The randomness is trusted, not proved.
- **`seed` coins** (PR #83, `coin-commit/sha256/v1`) add two assumptions: the seed commitment hides the seed, and
  SHA-256 in counter mode is a PRF. They are needed only if the seed mode becomes the verifier of record's source.
- **The random-oracle model is not used.** Its version of the compiled bound is only a cross-check (`../DESIGN.md` §2).
- **Quantum provers** (`../DESIGN.md` §9). The oracle layer holds against them, but the compiled theorem is classical. The
  compiled protocol is post-quantum sound only asymptotically, and only if SHA-512 is collapsing, which is a conjecture
  (Table 1 would need `collapsing/sha-512`). No concrete post-quantum bound is known at these parameters, and none is
  claimed.
- **Table 1** cites `computational-soundness` under `cr/sha-512` and `live-verifier`; A1's proof took
  `rs-proximity-johnson` off. An audit's integrity profile would add A2, a new claim once a render cites it.
