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

[2026-11-12T10:28Z] 136cfc068006
NETSEATS to the coordinator, Thu 12 Nov 2:28 AM PST. **Network's account value is drafted in both forms and compiles against proofs' tip. One key of the answer gates K, and four things do not fit the form.** Marks: [ran], [read], [argued]. No figures.

- **Note:** `note:network-accounting/20261112T1027Z-report-network-account-value-draft`.
- **Branch:** `cursor/na-account-draft-56db`, head `29447a3e255ac5cfca3a994efe565a2c30cf6f08`, pushed, no pull request, nothing marked. Proofs' file is untouched.
- **Proofs' tip read:** `e28f52c10a7d` (10:16Z). It moved four times while I worked; the last two commits are text only. My branch holds it by merges. [ran]
- **Try** [ran]: `r20261112-102546-3bff`, exit 0, on the head: both forms and 21 guards elaborate against proofs' `Account.lean`. It builds the file's imports and writes no record. A value written at `c45a492ab640` compiled unchanged at `c962872c9231` (`r20261112-095221-9346`, exit 0).

**The value** (`NetworkCheck/Account.lean`, ours)
- **Form 1, the value:** K on a link, free, in bits; `.span (.published "windows") seconds`; error `eps`, `pool := .sum`; `unpriced`: ε_ks, the commitment's binding, the certifier's openings.
- **Form 2, beside it:** `.steps (.run (.published "first") (.published "windows")) seconds`, `series := some (.published "session")`.
- **Keys that exist** [read]: `K`, `eps`, `windows` (`Period.lean:1278-1279`, `period_case.py:1004`) and `link` in the profile (`:1284`, `:1044`).
- **The count's key is `"windows"`.** Its word is retired, kept in records "until a format version" (`GLOSSARY.md:203-204`). Not renamed.
- **Keys missing:** the head (the change that first passes the check a site), `first`, `session`, and `certified`, the gate.

**Where it lives** [ran]: a private file beside `Period.lean`, imported by the private root, with one line in `public_boundary.toml`. Left public, the boundary check fails with two edges: the export does not copy `CoreProtocol/Account.lean`.

**K on answers read as given**
- Today's answer makes no row, but by accident: it names no head, and `link` is not a registered id. [ran]
- With both, and no flag, `rows` prints the K's summed against the link's capacity, standing "certified". [ran]
- **The gate: `uncertified := .flagged (.published "certified")`, a key no answer holds.** Absent gives no reading, so the link reads "not audited". A guard breaks the build in the change that first writes the key. [ran]
- The label for "read as given" must not be this key as false: false makes a row that still reads "certified". [ran]

**What does not fit**
1. **The form has no tie.** After the switch K is certified only inside an accepted composite; `rows` reads answers one by one. To ask of proofs. [read; argued]
2. **`egress_mediated`:** `.refused` and `.safe` do not fit. `.flagged` fits only by omission: the check writes `certified` only when answers are verified and egress is mediated. No standing carries "rehearsal: egress not mediated, no bound". [ran]
3. **The seconds of one step** are an execution phase's, per schedule and not per protocol: `window × bucket_ms`, published nowhere (`Period.lean:287-300`). The draft takes them as a parameter. Our fix: publish the span in milliseconds. [read]
4. **Neither form adds cover across sessions:** the certifier counts execution phases from its own start (`warden.py:473`, `:706`). That needs a clock its sessions share, a change of the device. [ran for two sessions]

**Form 2 needs of us**
- `first` and `session` in the public output. Both are parameters today (`Period.lean:340`, `:345`), signed in every certificate (`certificate.py:44-51`) and held in the chain's record; neither is in the public output, the profile or the registration. [read]
- **A week on one link** [argued]: form 1 covers the longest single certification period, form 2 the longest session. With S sessions of n periods each, form 1 charges 1 − 1/(S·n) of the week at capacity and form 2 charges 1 − 1/S. Today's driver makes each period one execution phase.

**Our other amounts:** none is a row of ours. The certifier's marks are ours to publish and memory's wipe to account for; the one-stage facts are sampled proofs'. [read]

**Slips, in my own shell only, nothing sent:** one machine name in an unmasked comment excerpt and one in a commit subject; twice a run's waiting output listed other runs' command lines. Section 6 of the note.

