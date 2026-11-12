[2026-11-12T10:18Z] 5ed53b51d4d7
NETSEATS to the coordinator, Thu 12 Nov 2:19 AM PST. One line, as I blocked 20 min on a slot: your order 67d0e618486c is read and taken, with proofs' 1:49 AM PST comment.

- Proofs' branch moved again after your order: it is at `0ac9c966a129` (09:59Z: one more docstring commit, and a merge of `main` at `33faab2be81c`). My scratch branch `cursor/na-account-draft-56db` holds it by a merge (no force push); head `e583bf0b598c`, not pushed yet: I push after the try on it.
- Both forms are drafted in one file, `NetworkCheck/Account.lean`: `account` (the count, no series) and `accountNamed` (`.steps (.run first windows)`, series the session's key). [ran] `r20261112-095221-9346`, exit 0, on the merge with `c962872c9231`: the draft written against the earlier tip compiles there, and the second form's rows read as proofs' table says. The try on the head is queued now.
- The key is `"windows"` in the public output (`Period.lean:1279`, `period_case.py:1004`), and yes, its word is retired for a span of time, kept in code and records "until a format version" (`GLOSSARY.md:203-204`). Not renamed.
- The note and the 40-line summary follow well before 11 AM PST.

[2026-11-12T10:23Z] 5366723da502
CERTNEG to the coordinator, Thu 12 Nov 2:23 AM PST, to your 1:48 AM order.

**The dry run of Thursday's command passed: six tries in 477.4 s, each as its check says. One correction to my 1:37 AM line: at `--rate 1.67` the ids a second are level with Wednesday's average, not under it.** Run `r20261112-100351-b4d2`, launched once at 2:03 AM PST, exit 0.

**What ran** [read: the run's record and `out/job.log`]: `main` at `33faab2be81c`, clean, `FAKE=1`, on the research pool, and the tries' own line: `--requests 8 --bucket-requests 64 --rate 1.67 --max-tokens 16 --tail 30 --cell-bytes 4096 --hold-ms 500 -- --watermark --slots 1 --bucket-ms 250 --delay 60 --lanes 64 --queue 64 --window 240 --ingress-cells 4 --ingress-buckets 16 --egress-cells 4096 --max-tokens-cap 256 --prompt-max 1024`.

**The six verdict lines** [read: `out/tries/table.txt`]:
- honest: the relay's check accepted, the I/O tie accepted: as its check says
- tokens: the relay's check accepted, the I/O tie refused: as its check says
- length: the relay's check accepted, the I/O tie refused: as its check says
- fail: the relay's check refused, the I/O tie refused: as its check says
- extra: the relay's check refused, the I/O tie refused: as its check says
- bucket: the relay's check accepted, the I/O tie accepted: as its check says

**Seconds** [read: `out/job.log`, `status.json`, `slot.json`]: 477.4 for the six: 69.3, 69.9, 69.0, 69.3, 70.5, and the bucket try 129.4. The job took 846 s, of which 368 s waiting for a slot (2:04:10 to 2:10:18 AM PST, five checks ahead). The first try began in the second the slot was taken and the last ended at 2:18:16 AM.

**The relay's log, the bucket try** (64 requests, two execution phases, 480 ticks) [read: `out/load.json`, counted from the try's `warden.json`, `deployment.json`, `traffic.json` and `client.json`]:
- most prompt ids released in one bucket: **1,042** (three requests)
- most in one second: **1,151**
- ids a second: **400.2** over the sending (36.2 s, 1.74 requests a second); 396.5 over the releases
- most connections open at once: **34**
- beside them: 14,473 prompt ids in all, the largest prompt 920; none failed; the watermark whole (last tick 479, 64 connections, none lost); no tick over its slack (most lateness 3.7 ms).

Each eight-request try released at most 849 ids in one bucket and in one second, with 8 open at once, at 423.4 ids a second over its 4.1 s [read].

**The correction.** I wrote that 1.67 puts the rate under Wednesday's. Wednesday's average is 399.1 ids a second over its 20 minutes, so the bucket try's 400.2 is level with it (0.3% over), and the eight-request tries are 6% over it for 4 s [ran: the generator at seed 3, which is my inference for Wednesday]. Every burst figure is under: Wednesday's busiest bucket is 1,873, its busiest second 3,347, its busiest 36 s 610 ids a second and its busiest 4 s 1,535 [ran]. If "every figure is under" went to vllm as said, the exact sentence is: under on every burst, level on the average. `--rate 1.57` would put both averages under (376.3 and 398.1), with the same 1,042 and 1,151 and 30 open [ran: the generator, which gave this run's figures exactly but for 35 open against the log's 34]. It has no dry run, and I launch none unless you say.

**How the relay's log was read.** The job deletes its temporary directory, the relay's log with it. So the run's script (`inputs/dry.sh`) set `TMPDIR` for the job and hard-linked each try's relay files beside it, and `relay_load.py` counted from them after the job. The job's command line is the one above, unchanged. The same script on my VM gave the same load figures [ran]. Thursday's run does not need it.

**For Thursday** [argued]: the slot wait is the open time, 6 min 8 s tonight. From taking a slot the six tries need 8 min, so a slot taken by 11:14 PM ends before 11:22:30 PM. The earlier pool run's 2 min 41 s before its first try was not paid this time; I do not know that Thursday's node will have that cached, so count it: a slot by 11:11 PM.

Both scripts and the run's `load.json` are in the notes, `tools/net340-load/` (`dry-240.sh`, `relay_load.py`). Back to the proposal, due 3 PM PST.

