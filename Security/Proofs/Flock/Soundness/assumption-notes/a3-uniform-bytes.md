# A3: what ties it to the verifier, where it is used, and the salts and masks

The statement is in `../ASSUMPTIONS.md`.

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
- **The masks and the Reed–Solomon padding** are the operating system's bytes too (`live/src/zk_hooks.rs`), read by
  the firewall's Lean processes: each rep's frame reads its draws (the pads, `μ`, the rows' padding, `τ`'s salt and the
  pads tree's salts) in one call at its start, and the tap serves the prover's every other read. What ties them to the
  model is `Claims.ZKServer.frame_cmd` (`Frame.cmd` is `Frame.cmdWith IO.getRandomBytes`) and `frame_draws` (the
  frame's read, on rep `i`'s stream of the tape, is the release's draws), as `drawOS_def` is for the unit draw;
  `Claims/ZK/Os.lean` takes the ZK server's three theorems to those bytes (`zkServerLeanOs`). So no premise on a stream
  generator is left on the salts or the masks: ChaCha20 expands a seed only in a `seed-injection` build, whose records
  are no evidence for a theorem that takes A3.
