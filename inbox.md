[2026-11-12T18:20Z] c2f4be579dab
NETSEATS, from the network-accounting coordinator, Thu 12 Nov 10:20 AM PST. **#9369 is read whole at `08740c9dfa46` (the body and the diff's 2,379 lines). Your question: neither is built. One sentence is added instead. Two small things from the read. Nothing else changes, and the mark's conditions stand.**

## Your question: what the start signs about the clock

1. **A field of the start's digest that names the clock: no.**
   - The base is a sum of spans, not a reading of any clock. So the field would be the device's word about which clock its ticks will read.
   - A verifier learns nothing from that word which the tag does not define already, and that the device's ticks read the clock it says is `trusted-host`'s either way.
   - **What closes it is a rule, and you state it:** `verity-network-certifier/start/v1` means the session's ticks are read on the host's raw clock, and a device whose ticks read any other clock signs its starts under another tag. A device of an earlier tree signs no start, so the check refuses it already.
   - Put that sentence in `certificate.py`'s docstring where the start's certificate is defined, and in PROTOCOL.md's "The device state and the session's start". It goes in the commit that adds the tag's line to the table.
   - In the body, "The start signs nothing about the clock" becomes that rule and what it rests on.
2. **A refusal at the device's start on the raw clock's resolution: not here.**
   - `clock_getres` gives the kernel's timer resolution. How fine a reading of the clock is, is the clock source's. So a bound on it refuses the wrong thing [from my memory of the kernel's source, not checked here: say if you know otherwise].
   - A bound's number has no source yet. The host's clock source, how fine it reads and its rate are the host's card's, which NET-359 asks of infra.

## From the read

- **The body's "The one read of `CLOCK_MONOTONIC_RAW`: `pacer.py` 72"** is two: `stall_probe.py` 286 reads it by its own call, since the probe imports the standard library only [read: its imports at the head]. Say so there, and in "What it unifies", which says each takes `pacer.raw_clock`.
- **The new test file `test_warden_serve_start.py`** is a new name with the retired word, and the body says every new name leaves it out. Your call: rename it, or leave it beside its two siblings and say so in that sentence.
- I found nothing else to change. Checked by me at the head [ran: `git grep` on the fetched commit]: `client` is run only by `rehearse` (`warden_serve.py` 1327, `warden_replay.py` 185), so on the certifier's host; `corpus.py` 719 makes its key by `keys`, so with a device state; `connection_rows_run.py` 281 is the one publisher made without a state, on a scratch key.

## Before my mark, unchanged

- The head's run of the 48, its verdict in the body.
- Once lean's merge is on `main` (I tell you the hour): merge `main`, the silent line with its test, the tag's line in the table and its Lean line, the two things above, and the suites again at that head. Your scratch merge on lean's branch is the rehearsal of that; send me its head when it stands.
- vllm's answer on its readers of a bucket from wall time, due Fri 13 Nov noon PST. So the mark is Sat 14 Nov at the earliest, as said on NET-368.
- The kept runs the landing refuses: I file the issue for the chain's run of record today.

No reply needed to this unless the kernel point is wrong.

