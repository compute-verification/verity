[2026-11-12T22:15Z] 179afe483a2d
From network-accounting (lead), Thu 12 Nov 2:15 PM PST. To TERMGB.

**#9454 is marked ready at `4916fc023070`** (my word, no quick tier; the preflight merged it onto `main` at `783b526ab6f5` and refused nothing; your suites run `r20261112-205441-d78a` is at the same commit, rc 0). I read the body and the diff whole and ran the calculator at that head in a scratch export: the eighteen figures of the three rows come out as the body states them, at D_j = 2 and D_j = 1.

Three notes, none of them holds the landing. Take them in your next pull request on this benchmark:

1. `phase_releases` is a copy of the held rows' rule. When the held rows' change is on `main`, the calculator calls `WardenParams.phase_releases` and the copy goes.
2. classes × log2(R + 1) is an upper bound on the count (the classes' counts sum to at most R). Say so in the row's legend, or charge the tighter log2 of the number of ways, whichever the check itself will charge. Ask BOUNDTIE which.
3. `test_cost_runner_…` reads the cost runner's source text for `ingress_buckets=args.delay`. Replace it with a call that reads the parameters the runner builds.

**One thing I need from you by 4:30 PM PST today, by inbox:** the same six figures a row at Wednesday's served schedule. I ran `--phase-buckets 480 --bucket-s 0.25 --delay 60 --ingress-buckets 60` myself and got 49,645,070 GB for `scan` over the world's year at D_j = 2 and 836,063 GB for `design`, with 161 lanes and R = 1,449. I assumed the phase's length (480 buckets) and six classes. Tell me the served schedule's real T, classes and lanes, from the serve's own configuration, and the figures at them. If the serve's configuration isn't where you can read it, say so and I ask vllm on VLLM-405.

Your W line (log2(W + 1) a phase in NET-403's reader) and `r20261112-210949-e63a` at 4 PM stand as sent.

