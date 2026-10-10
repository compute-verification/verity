# A3: where the operating system's bytes are read, the salts, and why hiding needs no SHA-512 property

The statement is in `../ASSUMPTIONS.md`.

- **Where it is used.** `drawOS_subset_escape_le`, `drawOS_stratified_escape_le`, `drawOS_work_escape_le`,
  `execOS_escape_le` and `execOS_subset_escape_le` (`ExecDrawOS.lean`). They hold for every number of
  bytes `L` the model reads. The live verifier reads as many as its reruns need, and its acceptance probability is the
  limit as `L` grows, which the theorems don't state. `subset_exec_escape_le` and `stratified_exec_escape_le` take
  uniform streams as their probability space, the same assumption.
- **The salts** (Daniel, 9 Oct, decisions 4(a) and 4(c)). C-Flock's prover reads every hm96 leaf salt, 192 bytes a leaf,
  from the operating system (`getrandom`, the kernel's `getrandom(2)`, which `/dev/urandom` also reads), and the
  recursive audit's firewall and a registrant read theirs the same way (`os.urandom`). A3 is that those reads, too, are
  independent uniform bytes. Zero knowledge's hiding theorems are stated at uniform salts; `Discharge/ZkView/OsSalts.lean`
  reads 192 bytes as one salt, one for one (`saltEquiv`; `leaf_os`: the executable's leaf at those bytes is the model's),
  and takes them to the OS's salts: `hm96Hiding_os` and `hm96Sha512_os` (one leaf), `ideal_leaves_swap_os` (T1 for `n`
  leaves), `hides_os` and `hides_leaves_os` (a registration's `PrivateCircuit.Hides`), and, for the recursive audit,
  `Flock.SecurityProofs.recursiveZK_os` (`Recursive/OsSalts.lean`). So no premise on a stream generator is left on the
  salts. The masks and the Reed–Solomon padding are still a ChaCha20 expansion of an OS seed (`ProverRng`), which the
  zero-knowledge theorems draw uniformly; that step is a PRG's, not A3's, and no `Prop` states it.
- **Hiding needs no property of SHA-512.** Salted SHA-512's hiding is statistical, not computational: `hm96Hiding_gap`
  reads nothing of SHA-512, and the leftover hash lemma only that `c(y) = SHA-512(sp ‖ y)` takes at most `2^512` values,
  so no reduction to a property of SHA-512 is needed, and `cr/sha-512` enters binding, not hiding. The salts' uniformity
  is A3's (the salts, above). `hash-derived-key` is the common-reference-string step for one public constant, which no
  reduction can take to a property of SHA-512 (there is no secret and no adversary); bounds stated at the gap itself take
  no premise (`CROnly.hm96Hiding_gap`, `Flock.SecurityProofs.recursiveZK_gap`).
