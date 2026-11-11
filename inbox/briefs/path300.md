# PATH300: the ordered list of steps from what runs today to 300 GB, each with its owner, its pull request and its state (reading; no code)

**Why.** I staff from this list tonight and give it to Daniel. Every step that has nobody on it gets a worker the hour your list is in.

**The figures, all the compute audit's part of the exfiltration capacity, a year over the world's datacenters, modeled at utilization 0.4** (sources: `inbox/proofs-coordinator.md` on `cursor/verity-proofs-84b5`, entries `## BREAKEVEN, Wed 11 Nov` 14:12Z, 16:58Z and 17:37Z, and the note `~/.research/notes/lanes/proofs/20261111T1738Z-report-breakeven-row-i.md`):
- **What the served engine runs today** (the first stage only, on the partition it serves), at measured seconds: 29,319 GB. The same with the verifier not reading the program at every statement: 6,867 GB.
- **The two-stage sampling design on the partition `main` defines:** 19,642 GB.
- **The best designed configuration:** 5,445 GB; with a request's output at half its bits (unjustified): 3,735 GB; the same with no read of the program: about 1,067 GB.
- At the vLLM lane's measured saturated serving rate each figure is about 0.056 of itself; which rate the goal is read at is Daniel's, and a brief on it is being written.

**Your table: one row a step, in the order that takes the served figure down fastest.** Steps I know of [check each; add what I miss]: the verifier reads the program once a call (#8949, marked), once a statement's parse (branch `cursor/proofs-progjson-89e3`), once a window (branch `cursor/proofs-keptverify-604f`); the Lean package split it waits on (#8789); the sampling design by width of block (#8950 is its figure; is it served?); the sparse-attention head's key block as scores and values units (#8917 is on `main`, Definitions only: who writes its Program, who serves it?); the head block's host-side terms (measured on branch `cursor/proofs-headprof-2b18`; kernels' design is due Thu 12 Nov 6 PM PST); the fused per-expert blocks served (goal `proofs/spot-fused-cut-served`; Linear VLLM-334, PRF-558); the second stage on a served engine (#8875, the vLLM lane's); the wide Merkle leaf (PRF-561, HASH-119), which moves the binding error and the prover's seconds.
For each row: what it is in plain words; the figure before and after [from a run you name, or marked as your arithmetic]; its state (runs on the served engine / on `main`, not run / on a branch / designed / not started); the pull request or branch and its head; its owner (our worker by its pull request's branch, or the lane: read `research msg owners` and the Linear issue); its date; and **who is on it now, or "nobody"**.

**Then three lines:** the steps with nobody on them, largest first; the steps that wait on another lane and what exactly they wait for; and what the figure would be with every staffed step landed, at utilization 0.4 and at the measured rate.

**Deadline (UTC):** 19:00Z a first entry with the rows you have; 20:15Z the table whole, as a note (`kind: draft`) and in your entry. The standing rules' words apply: this table goes to Daniel.
