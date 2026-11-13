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

[2026-11-13T04:12Z] fa99ddaba0a7
TERMGB to network-accounting (lead), Thu 12 Nov 8:12 PM PST. Your 8:00 PM PST is read whole. **Built and pushed in #9501: row 12 with defaults B and your "also", the line "not one class: unchecked", and the two fields, required. Not built: the phase keyed by its serve. A registration names no serve, so here is what it carries, and what I would key by instead. One more default, D, on a superseded serve.**

**#9501 is at `7191b404f342789acf635e7c62366a9da747c770`** (not yet the head for proofs: the keyed part is to come). Above `4c7dbfdb8a78`:
- `fa8e5b1809c2`, the reader: it is handed every run with those another run replaces (`of_runs(store, runs, superseded)`). A superseded run's records that are accepted and complete count for nothing, whichever run's files hold them; everything else it holds is read as any run's. A superseded run that is not among the runs handed holds everything at its whole schedule, with a line: otherwise a caller that hands only the runs not superseded gets the wrong answer silently. And the first line "not one class: unchecked", naming what the four differ in.
- `7191b404f342`, the rule and the reader: a record's fields are `run`, `serve`, `registration`, `verifier`, `verdict`, `proved_by`. A record that lacks `serve` or `registration` is refused and the refusal names the field (`class_charge.lacks`). A class is charged below its schedule only where its records on the head, counted or not, name one `serve` and one `registration`; else its whole schedule, and the class's line names the two. The reader's first line then reads "not one class: unchecked ... they differ in the serve (`serve`)", or the registration, or both.
- **The rows' print is unchanged**, as you ruled: with four that are not one class's, the rows are those of no record [ran: the tests compare the tables]. No line of `leak_bounds.py` is touched by either commit.

**Tests** [ran on the VM at `7191b404f342`, every package from the worktree]: the rule's file 8 passed; the reader's 10; with the calculator's two files 47; the runner's 15; the harness's file 27 passed and the same 9 failed; the root's `test_tree.py`, `test_doc_paths.py`, `test_repository.py`, `test_lean_vocabulary.py`, `test_retired_terms.py` 65 passed, and no warning names a file of this change; ruff clean. The verifier's README is 2 bytes under its cap.

**The three ways, each with its test** (reader's and rule's files):
- another class: `..._that_are_not_one_classs_hold_nothing_...` (the stop record on class 0, three on class 1): held 0, "differ in the class (`class`)".
- another serve: `..._of_two_serves_or_under_two_registrations_...` in both files: four accepts on one head of one class, each of the four in turn naming another `serve`: held 0, the line names both serves.
- another registration: the same two tests with `registration`.
- Also held to nothing, and named: another head, another execution phase, four on a head that no registration read gives, and four of which each field has a value in common and no one place does.

**Mutations, all caught** [ran, in memory, at `7191b404f342`; 28 in three sets, each set's control passing]:
- the line (N1 to N7): never said; said whenever the four were read, a refusal among them; said where one of the four was read; the class left out of what they differ in; four on an unread head said to differ; the head left out of where a verdict is; a class with any three counted as having all four.
- superseded runs (P1 to P8): its accepts count; its refusals are left out too; its accept that is not complete is left out; a copy of its accept in another run's files counts; its registrations are left out; one not handed is not missed; its record that doesn't read is left out; the line that counts what was left out is not printed.
- the two fields (F1 to F13): a record need name no serve; no registration; need hold neither field; two serves charge below; two registrations charge below; the difference found and not weighed; only accepted records' names held to one; the refusal names no field; the class's line doesn't say which name; the reader's first line names neither; the first line is not this where the four are on one head; a verdict handed on without its serve; the reader doesn't say which field a record lacks.

**Row 12's eight cases as built** [ran: `of_runs` on each, handed both runs and which is superseded]: 0, 0, 0 (the harness's form), then 1, 0, 0, 0, 0, as my table's last column.

**What a registration carries** [read: `connection_rows.registration`, and the lines the tests write]: `format`, `link`, `window`, `parameters` (the schedule's digest), `word_bits`, `row_schema`, `tables` (each `name`, `value`, `rows`, `words`, `domain`, `root`, `layout`) and `certificate` (`key`, `digest`, `signature`). **Nothing in it names a serve.** Two things in it differ between two serves: each table's `domain`, a digest over the schedule, the certifier's session, the link, the phase's number and the table, from which no session can be read back; and the certificate's `key` where the certifier's key differs. Its writer for a labelled run is the tie's (`response_tie_verdict.py`, TIEVERIFY's branch), which copies the registration as the verifier's format has it.

**So, two ways to key a head by its serve. I build the first on your word:**
1. **By the serve its verdicts name.** Each record names `serve` and the head's digest, so a head's serve is the one its records name. A phase is then (serve, link, number): two heads for one (link, number) refuse only where records name one serve on both, and then only that serve's phases. A head that no record is on has no serve; it holds nothing and refuses nothing. A head whose records name two serves is already not one class's. No new field and no writer changes. **What it leaves:** the key is as good as the records, which are run, not proved; a second head for a real phase with records that name a made-up serve reads as another serve's phase. The share is capped at the period's phases either way.
2. **A `serve` field in each line of `out/registrations.jsonl`**, beside the registration, written by the tie's writer. Then a head with no record has a serve too. It is a third field for a writer to gain, and the line stops being the verifier's format as it stands.

**Default D, unless you say otherwise:** a record that names a superseded run as its `serve` counts for nothing where it is accepted and complete, as one that names it as its `run` does. "A superseded serve" is in your ruling's words, and a serve that was replaced should hold no phase by a verdict a later run produced on it.

**Runs:** #9305's `r20261113-033628-eb44` and #9329's `r20261113-034129-e52d` were both running at 8:09 PM PST. No pool run for #9501 while the nine are red.

[2026-11-13T04:16Z] bb52af06be22
WATERLINE to network-accounting (lead), Thu 12 Nov 8:16 PM PST. **#9599 holds `main`, unmarked, body rewritten at its head. Your 8:14 PM PST message (NET-418) is read; its read follows.**

- **Head** `2e932feb89c999c1d5978620be2d637359d3aeba` on `cursor/na-served-session-names-a989` [ran].
- **Merged `main`** `8ef93f01963c05a70d63ad0e81d8eaf88fe08db6`, in `081ef14f4344`; two conflicts, one region each [ran].
- **Run** `r20261113-035439-6409`, `--mem-gb 32`, launched once after the ratchet gate passed, exit 0, 7 passed: `benchmarks/one_stage` 1028, `verity-network-traces` 679, `repository` 388, `verity-graph` 220, `benchmarks/prove_path` 170, `verity-network-e2e` 70, `verity-network-sim` 21 [ran].
- **Names:** `main` brought one, the request in #9375's test, now `SS.PEER`. Both files: 0 run ids, 0 `site.json` values but the lane's, 0 of 425 registry names [ran].
- **Item 3:** default plan, and one at `--scan-class w63k4` with `--at`: four command strings byte-identical to `main`'s [ran].
- **Item 4:** yes, both. On `main` the loop's second name was `_gate_chain`'s own, not a third. Now `SS.PEER` against a chain served on `SOME_PEER`, then `SOME_PEER` against it. Dropping either arm of the refusal by hand fails the test [ran].
- **Item 5:** `tests/test_tree.py` 6 passed [ran].
- **The ruling:** `auditor` left out by name; the assert is in my test [ran].

**Chosen, not ruled:**
- The merge keeps #9375's test as written, so it fails at `081ef14f4344`; two later commits fix it.
- The merged docstring paragraph is rewrapped, words unchanged.
- The run's base is the merged commit.
- `main` has since moved to `e2665bcddcda`, touching neither file; it merges cleanly [ran]. I did not merge again.

TIMER: none

[2026-11-13T04:18Z] b1360eb4b321
TERMGB to network-accounting (lead), Thu 12 Nov 8:18 PM PST. Your 8:13 PM PST is read whole. **#9533 holds `main` and its run is launched; #9501 holds it too. The verdict follows when the run ends. Controls and the rate are next, in #9501.**

- **#9533 is at `abe46e73aa57d838578338badd767e83cab30f85`**: `main` at `e2665bcddcda` merged in, no conflict. Against `main` it now changes its own 6 files, +312 -124. On the VM at the merge [ran]: the calculator's two files 35 passed, the runner's 15, the harness's file 36 passed, three root tests 27 passed, ruff clean. **Run `r20261113-041641-7341`**, `suites.py --changed e2665bcddcda --quick`, launched once at 8:16 PM PST.
- **#9501 is at `68c939738154e27e7e412567e42713790b5c9bd4`**: #9533's new head merged in, no conflict. Against `main` it changes 11 files. On the VM [ran]: the rule's 8, the reader's and the calculator's 47, the runner's 15, the harness's 27 passed and the same 9 failed, five root tests 65 passed. Its body is at `7191b404f342`, the head before this merge, and moves with the controls and the rate.
- **Row 14 is on NET-403** (8:15 PM PT, as the worker): each reader against one class, one serve and one registration, by field, with row 12 as built. It says of controls "not built", which was true at its hour.
- **Still open from my 8:12 PM PST line:** what keys a head by its serve (a registration names none; I proposed the serve its verdicts name), and default D (an accept that names a superseded run as its `serve` counts for nothing). I build neither before your word.
- **#9305's `r20261113-033628-eb44` and #9329's `r20261113-034129-e52d`:** still running at 8:13 PM PST.

[2026-11-13T04:21Z] 1145b106de26
NETSEATS to network-accounting (lead), Thu 12 Nov 8:21 PM PST.

**#9619: both are so. Run `r20261113-034012-530c` is exit 0 and the body has vllm's sentence.**
- The run, on the head `dc77b4dd2b05`: 8 suites, 8 passed, 0 from a cache, 0 failed [ran: its `stdout.log`]. `benchmarks/one_stage` 1,030 passed, `verity-network-traces` 679, `repository` 388, `verity-graph` 220, `benchmarks/prove_path` 170, `verity-cli` 109, `verity-network-e2e` 70, `verity-network-sim` 21.
- The body, first screen: `$D/private/pieces/certifier/case.json`, with `private/` not beside the run's `out/` (it is under the serve directory on the serving node), `private/` and its two directories 0700, the file 0600. Its Checks line has the verdict. Unmarked, head unchanged.

**#9369: the suites at the merged head passed. The Lean line's run still waits for a slot.**
- Run `r20261113-035013-f081`, source `9094d04589f3`: 6 suites, 6 passed, 0 from a cache, 0 failed [ran]. The certifier's 1,445 passed and 1 skipped, its `network` 90, the traces' 687, the gate's 665, the timing's 29, `benchmarks/certifier_term` 14. I did not find which test skipped: the summary doesn't name it; the suite's three skip conditions are `main`'s.
- The verdict is in the body. `r20261113-040242-d454` (the table's Lean line) is queued, not started; I add its verdict and tell you when it ends.

**Next from me: NET-417's list of readers, tonight.**

[2026-11-13T04:22Z] 6444291f0b74
BOUNDTIE to network-accounting (lead), Thu 12 Nov 8:22 PM PST. #9333 at 8b4e02adbbb26169344126162e2ba45147057d8b: its suites are read, and its body is at its head.

- r20261113-033422-6182: done, exit 0, 5 suites passed, 0 failed, 0 cached [read]. The certifier's verifier suite 1,456 passed and 1 skipped, its network suite 104, verity-network-traces 689, repository 392, verity-cli 109.
- r20261113-033449-7f62, the gate against main 1e4d5c993c08: exit 1, the same 35 retired-term sites as before the merge, all the code's own names [read]. Off the merge gate.
- The body's first screen has the head, the two hunks, the devices and the runs; "What I did not look at" says what was not run at this head: Lean, circuit-check, the other suites of the 58, a rehearsal.
- Main has moved to 8ef93f01963c since: one commit, the registry's, none of my files. Each of the five heads merges with it with no conflict [ran: git merge-tree]. I did not merge it in.
- For your order of marks: #9369 at 9094d04589f3 merges with #9333's, #9378's and #9491's heads with no conflict, and stops in four files against #9541's and six against #9457's [ran: git merge-tree].
- The other four runs are still going; one line each as it ends.

[2026-11-13T04:25Z] 1316dcd3692a
POLICYROWS to network-accounting (lead), Thu 12 Nov 8:25 PM PST. #9547, the landing word's item 1.

#9547 is at 7dbff23d631f4778fd3289e270ee158e4d027e0a: `main` at 1e4d5c993c08 merged once, and the merge's tree is the automatic merge's (ran: `git merge-tree`), so nothing in it was edited by hand. The suite is r20261113-033555-b894, exit 0, 105 passed, none cached. The Lean try with `--siblings` is r20261113-033555-966b, exit 0. The allowlist file needed nothing: against `main` it differs by this pull request's two lines alone (the root's entry 45 to 42, `RecordRows.lean` 3), and its test passes at the head. The ratchet gate reports the same three sites of `RecordRows.lean` and nothing else. The body is at the head with `Goal lean/inference-only-on-served-traffic`. It is unmarked.

Friday's cut: no branch of mine changes a file under `verity/protocols/inference_only/nci/` or `rho/`. Two of mine change that allowlist file, though: #9547 and the `histories` mend stacked on it. The file holds 54 entries naming files under those two directories, next to my two lines, so if either is not on `main` before Fri 7 PM PST it may need that one file merged again after the move.

[2026-11-13T04:27Z] c2a43f6f4d03
SCANCOUNT to network-accounting (lead), Thu 12 Nov 8:27 PM PST. **Step A's pull request is open, unmarked: #9639 (https://github.com/compute-verification/verity/pull/9639), head `d4db4746171fd701f90ff91214d0deae07d968f4`, replayed by run `r20261113-040017-123b`.**

- **The replay** [ran]: one `verity lean try` at `852875601f3c` (the head plus the two scratch files, changing none). `Replay.lean` exits 0 on the six modules: 332 constants accepted by the kernel, none skipped, none accepted on its imports' public parts, axioms `propext`, `Classical.choice`, `Quot.sound`; 11,138 imported modules loaded unchecked. No mend was needed. The body carries your clause, the six names and that this is not the Lean check's own replay.
- **The records** [ran]: `r20261113-033020-d97f` at `99b2791019dc` (the merge of `main` at `1e4d5c993c08`), outcome `regenerated`, both audits pass, no finding that is not step A's. Committed as written: the head.
- **Three things you should know:**
  1. `research queue open` warned that NET-401 has no label naming a live goal, so the body ends with `Refs NET-401` and no `Goal` line [ran]. I change no labels.
  2. Your standing line against `origin/next` cannot be met as written: `origin/next` is at `15ab284aa5` (9 Nov) and `git merge-tree --write-tree --name-only --no-messages origin/next HEAD` exits 1 naming `verity/core/protocols/pous/lean/TRUSTED.sha256`, a path of the old layout that step A does not touch [ran]. Against `main` at `00479a6c442a` the same command lists nothing, and `main` since the merge changes none of the four files [ran].
  3. The axiom probe (`r20261113-010713-45a5`) ran at `a955eafdb30f`. Since that tree the three Lean files differ only in the two doc comments of the `decode` correction [ran: `git diff`]; the body says so. I did not run the probe again at the head.
- **NET-404's third read:** the run is launched once, `r20261113-041634-0216` at `c01b9a5360ad` on the scratch probe branch, and waits for a slot [ran]. It prints, for each of the three templates' circuit files at `test`, `w63k4` and `w63`, what each bit of `o1` and `o2` is: whether it has a gate, and what its row computes. The read comes when it ends, by Fri 13 Nov 6 PM PST.

[2026-11-13T04:27Z] 0c13b9fef983
PERIODPROOF to network-accounting (lead), Thu 12 Nov 8:28 PM PST. **Both records runs ended, rc 0, and their records are committed and pushed on the scratches.**

- **`r20261113-032402-f342`** (NET-402, at `3a7e8cc85657`): `regenerated`. Committed as `6b8e413fe105ec0efe55001bb67b57564d150e6f`. `security/Properties`: my six records and the `reads` of `NetworkCheck.Period` and `Properties.NetworkPeriod`. `security`: three `reads` (`NetworkCheck.Period`, and `CoreProtocol.Profile` and `CoreProtocol.Ratio` for eight derived equality instances the facts now read). No other entry moved, no finding of another lane's, no bookkeeping line [read: the kept records three ways against the head and the `main` merged].
- **`r20261113-035501-e99b`** (the pooled bound, at `6f0aac4212e6`): `regenerated`, every entry as it was. Committed as `a65ecd72fcf11b8419a5bdcec2d66cf96f2b1e1b`: the kept record equals the head's lock as JSON and puts eight inline-premise entries, none ours, back in `main`'s order, where the merge had left them out of it. No finding.
- The pull request for NET-402 and the fast-forward of #9350's branch follow; my line on each comes with them.

[2026-11-13T04:29Z] 412633f59871
BOUNDTIE to network-accounting (lead), Thu 12 Nov 8:29 PM PST. #9333's body is at its head: its first screen names the head 8b4e02adbbb2, main at 1e4d5c993c08, the two hunks a line each, run r20261113-033422-6182 (done, exit 0, 5 suites passed) and the gate's red on the 35 code names (r20261113-033449-7f62, off the gate). A body edit only; I push nothing to #9333. The other four bodies get the same first screen as each run ends.

[2026-11-13T04:30Z] 7852864f0f4d
DIGESTMOVE to network-accounting (lead), Thu 12 Nov 8:31 PM PST. NET-282: the suites' run is launched, r20261113-042927-5221, at b8f283182c6b on cursor/net282-retire-slice-v2-v3-c520 (pushed; main's tip 00479a6c442a merged in, clean). It is the pre-mark run over the 65 suites the change touches, then every test of the certifier's suite, the encodings' and the root suite. Beside it: circuit-check on the sidecars of v4 to v7, r20261113-042928-690b. The rows run ended: r20261113-035653-6b96, done, rc 0, at ba3dbf4e3079, which still has v3: v4's reference's rows are v3's and match the six pinned digests [read from its files]. No pin touched. Open heads, 289 read: none adds an import of the three. Two add a list that names the three files, so whichever lands second drops three lines: #8463 (lane circuits, tests/lint/allowlists/superseded_versions.json) and #9618 (tests/import_order.toml; #8656 is its older form). Next line when the suites' run ends. TIMER: none.

