# A3: why per test, what ties it to the verifier, its scope, and where it is used

The statement is in `../ASSUMPTIONS.md`: `UniformRandomBytes L t` (every event), which the unit sample takes, and
`UniformRandomBytesTest L t E η` (A3 per test, one event within `η`, either way), which zero knowledge takes.

- **Why per test.** The every-event form is refutable at the deployed source. Linux's CRNG (5.18 and later,
  `drivers/char/random.c`) answers one read past 32 bytes with a ChaCha20 stream under a 256-bit key derived for that
  call, so the event that the bytes are among the source's outputs has probability 1 there and at most `2^(256 - 8n)` at
  `n` uniform bytes: any source with fewer outcomes than byte strings fails it
  (note:proofs/20261018T1925Z-report-redteam-zk-server-os, P4). Its zero is an idealization, the CSPRNG taken as exactly
  uniform; the true bound carries the CSPRNG's distinguishing advantage against the theorem's test, which is the per-test
  form's `η`. A salt is 192 bytes and a rep's samples are `drawBytes` bytes, so zero knowledge takes the per-test form.
  The unit sample still takes the every-event form, under the same idealization, since its reads are past 32 bytes too.
- **Why it is not stated about `IO.getRandomBytes`.** To Lean's logic an `IO` action is a function of the world and
  `IO.getRandomBytes` an opaque constant, so a `Prop` saying its bytes are random would be refutable, and every theorem
  taking it vacuous. At run time, Lean v4.34.0's runtime implements it as `lean_io_get_random_bytes`
  (`src/runtime/io.cpp`), which reads `/dev/urandom` on Unix and calls `BCryptGenRandom` on Windows. Re-read it at every
  toolchain bump.
- **What ties the statement to the verifier** is `drawOS_def`: the live sample `Flock.Draw.drawOS` is
  `Flock.Draw.drawWith` with every stream's bytes from `IO.getRandomBytes`, and the theorems are about `drawWith`. The
  step from `IO` to the model rests on `drawWith` being generic in its monad, together with A3.
- **Where it is used.** `drawOS_subset_escape_le`, `drawOS_stratified_escape_le`, `drawOS_work_escape_le`,
  `execOS_escape_le` and `execOS_subset_escape_le` (`ExecDrawOS.lean`). They hold for every number of
  bytes `L` the model reads. The live verifier reads as many as its reruns need, and its acceptance probability is the
  limit as `L` grows, which the theorems don't state. `subset_exec_escape_le` and `stratified_exec_escape_le` take
  uniform streams as their probability space, the same assumption. The recursive audit's sessions of V* need none of
  it: their law is the full sample, which samples every unit at every byte string (`execOS_subset_full_mem`,
  `execOS_subset_full_escape`), so `RecursiveRanRegistered`, `PublicAudit` and `RecursiveAuditC3` are proved from
  their forms with no A3 premise (`…NoA3`). `RecursiveSound` is proved the same way but is no longer a guarantee: its
  form admits a possibilistic break, and the recursion's guarantee is `Flock.Guarantees.RecursiveKS`. Those are fork
  forms and state no bits; the recursion's knowledge form, `Claims.Recursion.KS`, is at its level
  `Claims.Recursion.lam`, about 65 bits per SHA-512 call at the
  deployed shape, below the 100-bit floor and the 2^-128 target under every weight. Route 2C's bound λ₂
  (`lamLin2`) evaluates to at least 100 at the deployed records (`RecDeploy.lamLin2_deployed_ge_dep`, about 165.8), but
  `KSLin2` at the deployment's dense statements (`KSLin2On` at `RecDeploy.DenseAt`) only follows from a premise still
  open. At the coin server's record (`recTr`) read by `recRead` (`RecDeploy.ksLin2_deployed_partAccepts`), `PartOpensB`
  is proved, `PartChecksB` is no premise of its own (`VStarB.exists_chk_iff`), `recRead` reads the record back
  (`readsBack_recTr`) and the readers' calls are true (`readersTrue_layoutBTops`), so what stays open there is
  `DecompB` at what each part's circuit accepts (`PartAccepts`).
  Holo statements are candidate 3's own model, and `BitsLin2` at the dense
  statements follows from it (`bitsLin2On_of_ksLin2On`).
- **The salts.** Flock's prover reads every hm96 leaf salt, 192 bytes a leaf,
  from the operating system (`getrandom`, the kernel's `getrandom(2)`, which `/dev/urandom` also reads; under the
  ZK server, from its tap), and the recursive audit's ZK server and a registrant read theirs the same way (`os.urandom`).
  A3 per test (`UniformRandomBytesTest`) is that those reads pass the theorem's test as independent uniform bytes do, up
  to `η`. Zero knowledge's hiding theorems are stated at uniform salts; `Discharge/ZkView/OsSalts.lean` reads 192 bytes
  as one salt, one for one (`saltEquiv`; `leaf_os`: the executable's leaf at those bytes is the model's), and takes them
  to the OS's salts, each bound plus `η`: `hm96Hiding_os` and `hm96Sha512_os` (one leaf), `ideal_leaves_swap_os` (T1 for
  `n` leaves), `hides_os` and `hides_leaves_os` (a registration, at one event), and, for the recursive audit,
  `Flock.SecurityProofs.recursiveZK_os_gap` (`Recursive/OsSalts.lean`).
- **Its scope: the masks and the Reed–Solomon padding** are the operating system's bytes too (`live/src/zk_hooks.rs`),
  read by the ZK server's Lean processes. Each rep's frame reads its samples in one call at its start (`flock-firewall
  frame`; `Frame.cmd` is `Frame.cmdWith IO.getRandomBytes`, and `Frame.drawsWith` refuses a read of any other length):
  the pads `h`, the dot-product mask `μ`, the inner rows' padding, `τ`'s salt and the pads tree's salts. The tap
  (`flock-firewall tap`) serves the prover's every other read: the mask slot's words, level 0's hiding, the other trees'
  salts and the coin tree's nonce. What ties them to the model is `Claims.ZKServer.frame_cmd` and `frame_draws` (the
  frame's read, on rep `i`'s stream of the tape, is the release's samples), as `drawOS_def` is for the unit sample;
  `Claims/ZK/Os.lean` takes the ZK server's three theorems to those bytes (`zkServerLeanOs`). So no premise on a stream
  generator is left on the salts or the masks.
- **What is outside it.** A VM snapshot restored more than once repeats the kernel's CRNG state unless the kernel
  reseeds on a VM generation change (vmgenid, Linux 5.18 and later, with hypervisor support). For one-time pads a
  repeated state is total: two sessions under one pad reveal the XOR of their values.
  - A hardware RNG doesn't close it. From 5.18 the kernel mixes RDSEED in only when it reseeds (`extract_entropy`),
    about once a minute, and a read is the per-CPU key's ChaCha20 stream (`crng_make_state`).
  - Our VMs (Linux 6.11 under KVM) expose no VM generation counter: there is no ACPI device for it, and
    `vmgenid`, built as a module, isn't loaded.
    There, restoring a VM's memory more than once is outside A3. A boot from a disk image is a fresh kernel, seeded
    anew.
  - A read before the kernel's CRNG is seeded is outside it. Lean's runtime opens `/dev/urandom` and reads it to the
    full count, retrying on `EINTR`, with no user-space buffer. Unlike `getrandom(2)`, that doesn't wait for the seeding
    (the kernel logs `uninitialized urandom read`). The same nodes report `entropy_avail` 256 of 256: seeded.
  - A `seed-injection` build (the selftest's) expands the prover's values from an injected `ProverSeed` with ChaCha20:
    no record from such a build is evidence for a theorem that takes A3.
- **Falsification.** No test runs: A3 is an assumption about the world, scoped by platform, and the
  lock's `shape.conjectures`, the one place a falsification test is recorded, holds mathematical conjectures only. What
  checks it is the citation of `io.cpp` and `random.c`, re-read at every toolchain bump and on any platform but Linux.
