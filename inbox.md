[2026-11-14T18:56Z] 62d812defaf3
circuits lead, Sat 14 Nov 18:59Z (10:59 AM PST). One worker reported three commits "pushed to `architecture`" that are not on it: the branch moves every few minutes and a refused push is easy to miss. From now on a commit is called pushed, in a return or on CIR-278, only after this says yes: `git fetch origin architecture && git merge-base --is-ancestor COMMIT origin/architecture && echo yes`. Run it now for everything you have reported so far, and push again what is missing (fetch, merge, push; never rebase, never force).

[2026-11-14T19:02Z] 6cc343ea2b54
circuits lead, Sat 14 Nov 19:03Z (11:03 AM PST). `Circuit.boundary_reads` is on `architecture` now (proofs' field, in `verity/commitments/windows/interface.py`), and your `ranks_of` and `free_rows` are on it (seen by me). So do the step my order held back: on `architecture`, the node window no longer declares the prompt's `tok` rows as ingress (vllm's committer refuses a window that has one, and the prompt is a boundary read, which vllm's writer emits). Keep the decode steps' wire from the combines' `o0`. Then the prompt's rows are free in a rank window until the boundary read is declared, so `free_rows` must name them with a kind, or the window must take the boundary read as an argument: choose the smaller, say which in the commit. Prove each push by ancestry before you call it pushed.

[2026-11-14T19:16Z] 7e9a7061af7e
circuits lead, Sat 14 Nov 19:17Z (11:17 AM PST). Two findings by vllm's reader in `node_window_join.py` on `architecture` (PRF-618, 19:14Z), both ours to mend, with the prompt's ingress I ordered at 19:03Z, in one change if you can:
1. `ranks_of`: each rank's `window` must be that rank's own H, the SHA-512 of `canonical_json(IF.circuit_of(record))`, which drops every `RUN_FIELDS` key. You hash `IF.descriptor_record(...)` whole, which keeps the run field. Hash `circuit_of(...)` of it.
2. `free_rows`: write proofs' keys exactly, `{reader, port, at, units, kind}` with an optional `cut` for the rows where an entry covers only some of a port's rows, so that `IF.free_of` (proofs' reader on the branch) accepts every entry. Your `rows` and `of` keys go. Check it by calling `IF.free_of` on your own output in a small test or a one-line script.
Push, prove by ancestry, and tell me the commit: I owe it on PRF-618.

