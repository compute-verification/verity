[2026-11-12T23:07Z] 95b789b5b1ef
Coordinator, Thu 12 Nov 23:07Z. To every worker of the proofs lane: THE SECOND NODE IS OPEN NOW. MOVE YOUR WAITING RUNS TO IT.

- vy-nebius-2 is this lane's alone until Sat 14 Nov 02:00Z: eight proofs places, four Lean build slots, 1,716 GiB of memory, 120 cores, nothing else on it (infra's ready line, 22:56Z).
- What is established: infra's try of a file from `main` passed there in 31 seconds (`r20261112-225225-6d46`). My try of `SampledTwo/Projection.lean` at TWOLEG's `4b34ea95bb` (`r20261112-230345-2d09`) took a Lean slot there within a minute of launch at 23:04Z and was building at 23:06Z; its verdict is not in yet, and I say in your mailbox if it fails for a reason of the node.
- THE WORDS. A try from your VM: `verity lean try FILE --on vy-nebius-2 --pool proofs --as proofs`. Any other job: `research run --on vy-nebius-2 --pool proofs --as proofs --mem-gb N --timeout S ... -- CMD`, the same wrapper as on cpu-12 with the machine's name changed; inside it, `python3 tools/verity/lean/lean_changed.py --try FILE` (or `--build FILE...`, `--records ...`) as before.
- DO NOW: every run of yours that is WAITING in a line on another node and has not started: cancel it (`research cancel RUN`, after reading its state once more in the same minute) and launch it again on vy-nebius-2. A run that has started stays where it is. The lines on cpu-8, cpu-10 and cpu-12 were 7, 9 and 10 runs long at 22:58Z, the oldest 92 minutes; a run in line there gains nothing by staying.
- ASK THE MEMORY YOU USE. Tries of this lane peaked at 2 to 7 GB tonight: ask 12 GB for a try, not 32 or 64. A records run: what it peaked at last time plus a quarter.
- FOUR LEAN BUILDS RUN AT ONCE (one per copied tree). Put several files in one run. If a Lean slot there makes you wait more than ten minutes, say so in your entry's first line and I ask infra for more trees (two more fit on the disk).
- THE RULE "end your turn when you would only wait on a run" now bites less: a try there should come back in minutes. Compile, read the errors, fix, compile again. Nothing of the two-round leg had compiled at 23:00Z; that is what the next hours are for.
- It stops by itself at Sat 02:00Z. `--timeout` on every run.

