[2026-11-11T03:01Z] 32de9fc57665
From CROSS (NET-311 job 3), Tue 10 Nov 7:01 PM PST. Job 3 is as far as it goes without your word; I end my turn now. Three lines.

- **The note:** `lanes/network-accounting/20261111T0300Z-report-net311-cross-job3.md` (synced, 4ea2d4223c), with the scripts, each head's JSON and the four messages in `tools/net311cross/job3/`.
- **One more line of evidence that each result is the same change** [run]: file by file, the added and removed line counts of `CANDIDATE...RESULT` are those of the pull request's own diff: #8459 3 of 3 files (+819 −0), #8690 18 of 18 (+1823 −90), #8702 25 of 25 (+2888 −98), `security/lean-audit.json` among them.
- **The runs at 7:01 PM PST**, all running, none with a failed suite so far in what the log shows (it lags): `r20261111-025229-857d` (#8459) 12 suites passed, `r20261111-025310-9a68` (#8690) 17, `r20261111-025310-0d44` (#8702) 17. Open with you: finding 1 (the one path in `tools/verity/circuit_check/pyproject.toml` of #8702's result) and finding 2 (#8702's result wants one merge of #8690's, its own side of `connection_rows.py` kept). I made neither.

[2026-11-11T03:03Z] 532f3d430789
TIMEPAIR, NET-296, Tue 10 Nov 7:03 PM PST. **A correction to my 6:35 PM PST line, and my answer on the look. The second burst case under the same hold reads at chance: the shift comes and goes under the hold, it is not the hold's age. The plan runs as fixed.**

- **Case 2** [run: same two runs; 6:28:27 to 6:56:54 PM PST; 8,312 scored; `halted` 0]: process stamp 0.0003 [0.0000, 0.0012], kernel stamp 0.0005 [0.0000, 0.0016]; the pacer's shift -0.4 µs. Case 1 was 0.0182 and 0.0189.
- **The shift has two values and flips between them** [run: the pacer's write ends in 17-second segments]. It is about 0 or about -13 µs. Case 1: 0 for its first 52 s, -16 for 35 s, mixed for six minutes, -13 from 6:07:53 PM PST to its end but for two segments. Case 2: 0 throughout but for 6:38:21 to 6:40:22 PM PST (-13) and half of 6:47:53 to 6:48:27 PM PST. What moves is bit 1's write end (170 µs after its time in place of 183); bit 0's is 183 in both states.
- **So I withdraw "it came in a quarter of an hour into the hold and stayed"**: that was case 1 alone. This morning's sets may each have met either state; the awake set need not be the cause, and neither is the hold's minute.
- **The hold itself slows every release** [run, two runs that were not designed as a pair]: the pacer's write ends 181 to 187 µs after its time under the hold, 56 to 85 µs in the 4:08 PM PST rehearsal without it (same head, same hosts). That fits infra's second guess, from the hold's first minute. Block 2 gives tonight's own figure without the hold.

**The look: I would take (1), a fresh hold from 9:30 to 10:05 PM PST, but to read what goes with a flip, not minutes 0 to 30.** One burst case of 29 minutes as in block 1, its guard before it, and beside it on cpu-10 one probe of mine:
- which core each of the serving side's three processes (server, certifier, pacer) is on, every 20 ms, read by PID from the kernel's per-process record (no command line), and which cores are siblings;
- infra's (a): a fixed loop of about 2 ms, run three times each second on the next core in turn, so each core is timed once every 64 s, with that core's steal ticks.
- **Yes, (a) can run beside the pacer.** The probe takes about 2% of one core. It lands on the pacer's core during a release once or twice in 17,400 intervals, whichever the bit [argued]; that shows as a late release.
- It can show whether a flip goes with a process moving to another core, with a core running slower, or with steal. It shows nothing if the case never flips: case 2 was at 0 for 26 of its 28 minutes.
- **No planned row of block 1 falls after 9:00:10 PM PST** (the last ends 8:50:42 PM PST). All of block 2 does (guard 9:03:00, cases 9:05:30 to 9:21:50 PM PST), without the hold, as designed.
- **I need from you:** when the look's three runs may be on the nodes (may they be launched at 9:24 PM PST and sleep until 9:31, or only after 9:30), and by when they are off.

One slip: at about 6:53 PM PST I printed a case's whole config to my own terminal and cpu-10's private address was in it. It is in no message, note, label or Linear text.

