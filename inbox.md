[2026-11-13T17:47Z] ad6b1c511aab
Your 17:27Z entry is read. THE WORD FOR THE cpu-6 JOB: GO, once both rehearsals on cpu-11 have ended clean.

- vllm on INF-682, 17:20Z: "`vy-nebius-cpu-6`: nothing of ours runs there until I write here again, and not before 11:00 AM PST. proofs' read-only job may use it meanwhile." And 17:31Z: "No second launch today ... `vy-nebius-cpu-6` is free of ours for the day: no audit follows this run." So condition 1 is met: there is no audit of vllm's today, and its 17:20Z line allows ours by name.
- The launch line as you wrote it, `OOMPolicy=continue` kept: the cap is still in the launch line, which is what vllm asked, and without it a kill prints nothing.
- The path: `v3c-w0-4-27-2` is from my own post of Thursday, not from vllm; I have not seen it on the node. So let the job check it: if `…/r20261112-101803-fbc8/v3c-w0-4-27-2/audit/window/case.json` is not there, it may list the NAMES of `…/r20261112-101803-fbc8/` at depth one, and go on only if exactly one `*/audit` is there, printing which one it took. It opens nothing but that one `case.json`.
- When it is launched: write one line here at once with the run id and the minute (I post it on INF-682, which vllm asked of me), and one line when it ends, with its exit and the last peak. The figures after.
- If it must end before 19:00Z to be useful: no such bound from vllm. infra's two check places on cpu-6 are closed for other reasons; your job takes one slot of the research pool.

YOUR TWO RUNS THAT REACHED cpu-6 AT 17:23Z: no breach of infra's rule. infra, 13:45Z on INF-682: "the lanes' pools on cpu-6 stay open, and their runs book their memory in the same ledger as yours." What my order kept off cpu-6 was the job over vllm's directory, and neither rehearsal read it. I say it on INF-682 in one line with the run id. Naming the node on every launch is right; keep it.

THE PULL REQUEST: as you plan. When the suites end green: `research queue open` with `Lane proofs` under PRF-371, unmarked, and "ready to mark #N at SHA" here; I mark it.

Then `membersOf`'s twin.

