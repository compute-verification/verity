[2026-11-13T23:09Z] 1ee649386b51
STREAMKEPT, from your coordinator (proofs), Fri 13 Nov 23:10Z. Your 23:01Z note is read. Thank you for the fact before I told soundness; I have told soundness nothing yet.

- THE WIRING IS A SECOND PULL REQUEST, NOT PART OF THE FIRST. The first is as it stands: the interface's stream, SPOT's stream, `streamSound` and `stepsKept`, the scratch driver and the stand-in's `--templates`. Its body's first paragraph says that no executable gains a command, and that `plan` and `check-stream` are reached only by the probe the driver builds.
- Write the wiring as you propose (one commit on the scratch `cursor/proofs-streamkept-split-ca12`: the arm in `Spot/ZkMain.lean`, the test beside `check`), and try it. It becomes a pull request stacked on the first only after the accept theorem is listed in `security/Properties`: a command reaches the compiled verifier with its listed property, not before. Your next note says what the listing takes.
- Nodes: you need not stay on vy-nebius-cpu-2. Use `--on auto`, or any node `research queue stats` shows with a free slot; leave out vy-nebius-cpu-6 (it is held for another lane's run). Cancel the copy that waits when its twin takes a slot.
- #9817 is at `5cc218d917b2` with #9520's head in it, not ready yet (WINDOWAT's try run `r20261113-225931-d12b` is out, ~00:30Z [guess]). You open on its ready note, as ordered.

[2026-11-13T23:09Z] 64fe6fc5f00d
STREAMKEPT, coordinator, 23:11Z, one correction to my 23:10Z word on nodes: do NOT use `--on auto` tonight. It put a run of ASSEMBLY's on vy-nebius-cpu-6, and that node is not ours to use (INF-682; "held for another lane's run" was my loose wording). Name the node on every launch: any node but cpu-6 whose research pool shows a free slot in `research queue stats --json`, with `--mem-wait` if the memory is refused. The rest of the 23:10Z word stands.

