# A3: why per test, what ties it to the verifier, its scope, and where it is used

The statement is in `../ASSUMPTIONS.md`: `UniformRandomBytes L t` (every event), which the unit draw takes, and
`UniformRandomBytesTest L t E η` (A3 per test, one event within `η`, either way), which zero knowledge takes.

- **Why per test.** The every-event form is refutable at the deployed source. Linux's CRNG (5.18 and later,
  `drivers/char/random.c`) answers one read past 32 bytes with a ChaCha20 stream under a 256-bit key derived for that
  call, so the event that the bytes are among the source's outputs has probability 1 there and at most `2^(256 - 8n)` at
  `n` uniform bytes: any source with fewer outcomes than byte strings fails it
  (note:proofs/20261018T1925Z-report-redteam-zk-server-os, P4). Its zero is an idealization, the CSPRNG taken as exactly
  uniform; the true bound carries the CSPRNG's distinguishing advantage against the theorem's test, which is the per-test
  form's `η`. A salt is 192 bytes and a rep's draws are `drawBytes` bytes, so zero knowledge takes the per-test form.
  The unit draw still takes the every-event form, under the same idealization, since its reads are past 32 bytes too.
- **Why it is not stated about `IO.getRandomBytes`.** To Lean's logic an `IO` action is a function of the world and
  `IO.getRandomBytes` an opaque constant, so a `Prop` saying its bytes are random would be refutable, and every theorem
  taking it vacuous. At run time, Lean v4.34.0's runtime implements it as `lean_io_get_random_bytes`
  (`src/runtime/io.cpp`), which reads `/dev/urandom` on Unix and calls `BCryptGenRandom` on Windows. Re-read it at every
  toolchain bump.
- **What ties the statement to the verifier** is `drawOS_def`: the live draw `Flock.Draw.drawOS` is
  `Flock.Draw.drawWith` with every stream's bytes from `IO.getRandomBytes`, and the theorems are about `drawWith`. The
  step from `IO` to the model rests on `drawWith` being generic in its monad, together with A3.
- **Where it is used.** `drawOS_subset_escape_le`, `drawOS_stratified_escape_le`, `drawOS_work_escape_le`,
  `execOS_escape_le` and `execOS_subset_escape_le` (`ExecDrawOS.lean`). They hold for every number of
  bytes `L` the model reads. The live verifier reads as many as its reruns need, and its acceptance probability is the
  limit as `L` grows, which the theorems don't state. `subset_exec_escape_le` and `stratified_exec_escape_le` take
  uniform streams as their probability space, the same assumption.
- **The salts** (Daniel, 9 Oct, decisions 4(a) and 4(c)). C-Flock's prover reads every hm96 leaf salt, 192 bytes a leaf,
  from the operating system (`getrandom`, the kernel's `getrandom(2)`, which `/dev/urandom` also reads; under the
  firewall, from its tap), and the recursive audit's firewall and a registrant read theirs the same way (`os.urandom`).
  A3 per test (`UniformRandomBytesTest`) is that those reads pass the theorem's test as independent uniform bytes do, up
  to `η`. Zero knowledge's hiding theorems are stated at uniform salts; `Discharge/ZkView/OsSalts.lean` reads 192 bytes
  as one salt, one for one (`saltEquiv`; `leaf_os`: the executable's leaf at those bytes is the model's), and takes them
  to the OS's salts, each bound plus `η`: `hm96Hiding_os` and `hm96Sha512_os` (one leaf), `ideal_leaves_swap_os` (T1 for
  `n` leaves), `hides_os` and `hides_leaves_os` (a registration, at one event), and, for the recursive audit,
  `Flock.SecurityProofs.recursiveZK_os_gap` (`Recursive/OsSalts.lean`).
- **Its scope: the masks and the Reed–Solomon padding** are the operating system's bytes too (`live/src/zk_hooks.rs`),
  read by the firewall's Lean processes. Each rep's frame reads its draws in one call at its start (`flock-firewall
  frame`; `Frame.cmd` is `Frame.cmdWith IO.getRandomBytes`, and `Frame.drawsWith` refuses a read of any other length):
  the pads `h`, the dot-product mask `μ`, the inner rows' padding, `τ`'s salt and the pads tree's salts. The tap
  (`flock-firewall tap`) serves the prover's every other read: the mask slot's words, level 0's hiding, the other trees'
  salts and the coin tree's nonce. What ties them to the model is `Claims.ZKServer.frame_cmd` and `frame_draws` (the
  frame's read, on rep `i`'s stream of the tape, is the release's draws), as `drawOS_def` is for the unit draw;
  `Claims/ZK/Os.lean` takes the ZK server's three theorems to those bytes (`zkServerLeanOs`). So no premise on a stream
  generator is left on the salts or the masks.
- **What is outside it.** A VM snapshot restored more than once repeats the kernel's CRNG state unless the kernel
  reseeds on a VM generation change (vmgenid, Linux 5.18 and later, with hypervisor support) or mixes in a hardware RNG.
  For one-time pads a repeated state is total: two sessions under one pad reveal the XOR of their values. A
  `seed-injection` build (the selftest's) expands the prover's values from an injected `ProverSeed` with ChaCha20: no
  record from such a build is evidence for a theorem that takes A3.
- **Falsification.** No test runs: A3 is an assumption about the world, scoped by platform (8 Oct ruling), and the
  lock's `shape.conjectures`, the one place a falsification test is recorded, holds mathematical conjectures only. What
  checks it is the citation of `io.cpp` and `random.c`, re-read at every toolchain bump and on any platform but Linux.
