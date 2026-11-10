# hash-derived-key: zero knowledge at the pinned key's hiding gap

The statement is in `../ASSUMPTIONS.md`.

- **Which guarantees take no premise on the key.** `Flock.Guarantees.ZeroKnowledgeHidden`, `Flock.Guarantees.RecursiveZK`,
  the recursive audit's circuit privacy `RecursiveCircuitPrivate`, `RecursiveCircuitIndist` and
  `RecursiveCircuitPrivateLeaves`, and `FailClosed.flock_verify_sound`'s views `ViewZ`, `ViewZHJ` and `ViewZHJR`. They are
  stated at the hiding gap of hm96-sha512's pinned key `K*`, `hidingGap My saltHash`.
- **The gap's tail over keys.** Nothing proves that gap small for `K*`: it is below `2^-193` for all but `2^-64` of keys
  (`CROnly.hm96_sha512_tail`), and that `K*` is not one of the others is unproved.
- **What still takes `hash-derived-key`.** `hash-derived-key` is `Assumptions.HashDerivedKey`: `Hm96Hiding` at `2^-193`,
  instantiated at hm96-sha512's `ZkView.My` and `ZkView.saltHash`. The lemmas that state `2^-193` still take it
  (`Composed.zk_session_composed`, `ZkHidden.zk_session_composedHJ`), and so does the recursive audit's property
  (`RecursiveAudit`, whose (ii) is `recursiveCircuitPrivateLeaves_le` at `2^-193`).
- **Hiding needs no property of SHA-512.** Salted SHA-512's hiding is statistical, not computational: `hm96Hiding_gap`
  reads nothing of SHA-512, and the leftover hash lemma only that `c(y) = SHA-512(sp ‖ y)` takes at most `2^512` values,
  so no reduction to a property of SHA-512 is needed, and `cr/sha-512` enters binding, not hiding. The salts' uniformity
  is A3's (`a3-uniform-bytes.md`, the salts). `hash-derived-key` is the common-reference-string step for one public
  constant, which no reduction can take to a property of SHA-512 (there is no secret and no adversary); bounds stated at
  the gap itself take no premise (`CROnly.hm96Hiding_gap`).
