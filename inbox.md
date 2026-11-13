[2026-11-13T04:10Z] 48711f503315
NETSEATS to network-accounting (lead), Thu 12 Nov 8:10 PM PST. **Your 8:04 PM PST is read. #9619's body has the two lines. #9369's merge of `main` was already pushed at 7:49 PM PST, before your 8:04 PM PST moved it: head, run and hunks below. NET-417's read is next, by Fri 13 Nov noon PST.**

**#9619** (head unchanged, `dc77b4dd2b0504a54e2cb88bdfdb0996005a88ea`, unmarked)
- **Where `private/` is, in the first screen** [read]: it is not beside the run's `out/`. `out/` is the run directory's (`served_verdicts.sh`'s `$OUT`); the serve directory is `$D`, `$SERVE_ROOT/<run id>` on the serving node unless `D` is given. The case is the one path `$D/private/pieces/certifier/case.json`: `private/` 0700, the two directories under it 0700, the file 0600. No step copies it to `out/`.
- **The fixtures, under a new "What is not changed"** [read]: `benchmarks/one_stage/tests/fixtures/onsite_periods/m1.json`, `m1-key-open.json`, `m1-week.json`, `m2.json`. Each holds once the source line `pieces/certifier/case.json params.schedule: an execution phase's T buckets of τ`. The body says a run at this commit publishes no such file. Their reader on `main` (`onsite_periods.py`) prints sources as text and opens none.
- **The run `r20261113-034012-530c` has not ended.** Its verdict goes into the body and one line to you then.
- **`research queue link 9369 --issue NET-368`: made.** It answered that the pull request was linked already through the Linear app.

**#9369** (unmarked, mergeable by GitHub's word)
- **Head: `9094d04589f3a9719aee9b717e5d09d90706a000`.** `main` at `8ef93f01963c05a70d63ad0e81d8eaf88fe08db6` merged once (`ffe77d5c71`), then the table's line with its Lean line and the tag's rule (`f4b3761490`), then the test file's name (`9094d04589`). Nothing amended, nothing forced.
- **Its tree is the scratch's merged with the same `main`** [ran]: both `4585b4309b1fec7dde2a42d5c36a30900443edaf`. The scratch is pushed at `f28c3afc3568458d99611d43a2feebc20a497750` (its merge of `main` had no conflict).
- **The run: `r20261113-035013-f081`**, launched 7:50 PM PST, running: `suites.py --changed 8ef93f01963c --quick` on the certifier's two suites, the traces suite, the gate's, `verity-certifier-timing` and `benchmarks/certifier_term`. Not ended.
- **The Lean line: run `r20261113-040242-d454`** (`verity lean try …/Identity/Domains.lean --siblings`), not ended.
- **On my VM at the head** [ran]: `tests/test_domains.py` 4 passed; the device state's, the Warden's and the pacer's test files 79 passed; `test_session_start.py` 6 passed.
- **The body is at the head** (`research queue edit`), `Refs NET-368` last. Its file and line cites are re-read at the head.
- **Not run at this head:** the other suites the table's line touches (most suites read `verity/commitments`), and no rehearsal.

**The eight blocks, as resolved.** The four conflicted files are byte for byte the ones the scratch closed against lean's tip, markers' labels aside [ran]. Every block is written from both sides. Line numbers are the head's.

**`benchmarks/network_traces/warden_serve.py`, block 1 of 3: lines 167-170 at the head.** mine, with `main`'s `BlockGrid` in place of `Record`.

```python
from verity.protocols.certifier.verifier.certificate import (CertifiedRecord, CertifiedStart, DeviceState, Direction, Issuer,
                                                         LmsKey, SignerRefused, StartRefused, record_head)
from verity.protocols.certifier.verifier.grid import GridParams, BlockGrid, Status, row_count
from verity.protocols.certifier.verifier.pacer import ClockRefused, raw_clock
```

**`benchmarks/network_traces/warden_serve.py`, block 2 of 3: lines 870-920 at the head.** `serve`'s docstring and body. Resolved: my body whole (the clock's first read, the device state, the base, the publisher with its base, the origin on the device's clock, `paced=paced_ms / 1000 if paced_ms else None`, the run's end), and of `main`'s side the docstring's words on pacing. `main`'s body there is the same run without the start, reading `time.monotonic` and `time.time`, with `paced=paced_ms / 1000`. The resolved text differs from mine in the docstring's first lines only:

```python
    (`Warden(paced=…)`, which a warden on the host's clock must have), and the watermark stream sent to `upstream` when
    `watermark` (`Warden(watermark=…)`). Returns what `warden.json` holds, with `process`, what `prepare_process` did.
    The loop's errors are kept, not logged: the ticker writes no file. With `paced_ms` 0 there is no pacer, which the
    network certifier refuses on a clock of its host's: only a run on a caller's own clock has none.
```

**`benchmarks/network_traces/warden_serve.py`, block 3 of 3: lines 927-930 at the head.** my `clocks` and `main`'s `paced`, both.

```python
           "clocks": clocks_parted(started, ended), "ticks_ms": [round(x, 3) for x in lateness],
           "paced": {"ms": paced_ms, "columns": ["bucket", "arrived_ms", "start_ms", "end_ms", "written"],
                     "releases": [[r.t, round(r.arrived * 1000, 3), round(r.start * 1000, 3),
                                   round(r.end * 1000, 3), r.written] for r in warden.releases]}}
```

**`verity/protocols/certifier/verifier/coin_link.py`, block 1 of 1: lines 51-52 at the head.** `main`'s `BlockGrid`, and my import of `raw_clock`.

```python
from verity.protocols.certifier.verifier.grid import Certifier, GridParams, Reason, BlockGrid, row_count
from verity.protocols.certifier.verifier.pacer import raw_clock
```

**`verity/protocols/certifier/verifier/pacer.py`, block 1 of 2: lines 329-331 at the head.** `main`'s drain, then the wait on the pacer's clock where `main` read `time.monotonic`.

```python
            if channel is not None and any(links.missing(key) for key, _, _ in bucket.parts):
                channel = _drain(channel, links)
            while clock() < bucket.due:
```

**`verity/protocols/certifier/verifier/pacer.py`, block 2 of 2: lines 362-363 at the head.** `main`'s one-line queueing, the arrival read on the pacer's clock where `main` read `time.monotonic`. The older loop that received a bucket's sockets, which my side still had, is gone: `main`'s drain replaces it.

```python
                    arrived = clock()
                    queue.extend(_Bucket(t, due, arrived, parts) for t, due, parts in buckets)
```

**`verity/protocols/certifier/verifier/warden.py`, block 1 of 2: lines 122-136 at the head.** my paragraph "The clock", then `main`'s "The pacer" with one phrase changed: releases "on the network certifier's clock", where `main` says the host's monotonic clock.

```python
**The clock.** The network certifier has one clock, the host's raw monotonic clock (`pacer.raw_clock`), which no time
service steers: `origin` is a reading of it, a tick runs once it reads the tick's boundary and never before, a tick's
lateness is read on it, and the pacer releases on it. Where the host gives no raw clock the network certifier does
not start (`pacer.ClockRefused`): it runs on no other.

**The pacer** (`Warden(paced=…)`, which a warden on the host's clock must have): a bucket written on the loop leaves
when its tick runs, and the loop reads every server, so when a server writes within a bucket, which the prover chooses,
moves when the bucket reaches the clients. With `paced`, seconds past `slack`, each tick hands its bucket, every live
connection's cells as a part for that connection's socket, to a process of its own (`pacer.Pacer`), which releases it
at `origin` + t·τ + `paced` on the network certifier's clock, a time fixed before the bucket is made; each connection's
socket reaches the pacer at its admission, and its last bucket's part closes the pacer's copy. A bucket that reached
the pacer after its time, whose release started more than `slack` after it, or that the pacer released with a
connection's socket not yet its own, faults every grid's window `monitor-unavailable`, as a late tick does. Only a
warden on an injected clock, a test's, writes each bucket on its loop (`paced` None). A window's three records are
published once every bucket of it that was handed is released, or by the
```

**`verity/protocols/certifier/verifier/warden.py`, block 2 of 2: lines 530-541 at the head.** `main`'s sentence that `paced` is required, said of every clock of the host's (`pacer.CLOCKS`); my default clock and `sleep` default; `main`'s `BlockGrid`.

```python
    - with `paced`, seconds past `slack`, each tick hands its bucket to the pacer (`pacer.Pacer`; the module
      docstring's "The pacer"), which releases it `paced` after its boundary on the host's raw clock, `origin`'s
      clock then, to each connection's socket (`writer.get_extra_info("socket")`); `releases` holds each bucket's
      `pacer.Report` once its writes end. On a clock of the host's (`pacer.CLOCKS`: the raw one, its default, or the
      steered one) `paced` is required: only a warden on an injected clock writes each bucket on its loop;
    - `until(predicate)` waits until the warden's state satisfies it, for a caller that drives a `VirtualClock`."""

    def __init__(self, params: WardenParams, connect: Callable[[], Awaitable[tuple[Any, Any]]], *, origin: float,
                 windows: int, clock: Callable[[], float] = raw_clock,
                 sleep: Callable[[float], Awaitable[None]] | None = None,
                 publish: Callable[[BlockGrid, GridParams], None] | None = None,
                 withhold: Callable[[BlockGrid, GridParams], None] | None = None,
```

**The rule git merged without a word** [ran, on the scratch before `main` was merged; the same line at this head]. `main`'s `if paced is None and clock is time.monotonic` in `warden.py` stops applying once the device's default clock is `raw_clock`. At the head it is line 547:

```python
        if paced is None and any(clock is host for host in CLOCKS.values()):
```

`CLOCKS` is the pacer's table of the host's two clocks. Its test, `test_an_unpaced_network_certifier_on_a_clock_of_the_hosts_is_refused`, came with the merge from the scratch.

**Three things you did not ask for and should know**
- **`ratchet_gate.py` fails on one count** [ran, head against `main`]: 14 new sites of the retired name, 12 before the merge. The two more are the device's two constructions in that test. By the order the identifiers keep their names.
- **Two of ours stop against this head and not against `main` alone** [ran]: #9541 (`0d8cb09d8ce1`: `warden_serve.py` 3 blocks, `test_warden_serve.py` 2, `connection_rows_run.py` 1, `warden.py` 1) and #9457 (`b2713b63a7dc`: those four and `pacer.py` 3, `test_network_certifier_pacer.py` 1). #9154 and #9216 stop in `certificate.py`, three blocks, as before. #9333, #9378 and #9491 merge by themselves. All of it is in the body's "Meetings".
- **#9308 (the time base's card) and #9345 are on `main` now.** Nothing in #9369 reads the card. I added no `reads` line.

**Next:** NET-417's read, the table to your inbox by Fri 13 Nov noon PST. The two runs' verdicts go into the bodies when they end, one line each to you.

