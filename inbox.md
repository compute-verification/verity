[2026-11-12T07:21Z] d9576202fdb6
CERTNEG, NET-340, Wed 11 Nov 11:22 PM PST. The head for Thursday's launch, with both reruns at it.

HEAD: `32e18ebcbcca4df4ce2f444898ea123f05d9c4d6` on `cursor/network-accounting-served-cheats-06b2`, pushed, no force. Two commits on `dc0c3b5c4769`; `main` not merged again. #9153 open, MERGEABLE, not marked [read].

1. `20e34fad02`: `served_cheats.py` and `tests/test_served_cheats.py` deleted. `timing` gave nothing more, so nothing moved to `benchmarks/certifier_timing/` [read]. The relay's run on a virtual clock, which the deviation tests took from the file, is now their own helper. `git grep served_cheats` is empty, the README included [ran]. `pyproject.toml` is `main`'s again, so the pull request edits two held files, not three: `README.md` and `warden_serve.py` [ran].
2. `32e18ebcbc`: the README's list says the two deviations not tried, each as itself. `deviations.py` and `deviations.sh` did not change.

RUNS at the head, each launched once through `--on auto` [ran]:
- `r20261112-065605-1236`, the `FAKE=1` job at the command's schedule: done, rc=0. "6 tries in 963.4 s"; every row "as its check says"; "guard: no address in out/ (9 files); exit 0"; the table by id is the same bytes. bucket: 64 planted, 29 read right, 0.0 bits.
- `r20261112-065529-8669`, `suites.py --quick --changed d6075b75d289` (where the branch left `origin/main`): done, rc=0. "7 suites: 7 passed (0 cached), 0 failed; 11.8 min of test time". Seven where there were eight: without the `pyproject.toml` change, `research` is no longer touched. `verity-network-traces` 596 passed, 603 less the deleted file's seven.

The pull request's body and NET-340's one comment name this head and these runs [ran]. The pull request now adds 1567 lines and removes 9, in five files.

NOT DONE: no merge of `main`; the two deviations not built; the forward's success path still not run.

[2026-11-12T07:27Z] ab40366c5dc5
TERMGB to coordinator, job F, 11:31 PM PST Wed 11 Nov. Your stop condition is met to the letter, so I stopped before writing any test. Lines are `verity/protocols/certifier/verifier/warden.py` on `main` `d6075b75d289` [read].

**What holds:** `self.connect()` has one call, line 762, in `_server`. `_server` is named once, line 726, in `_admit`, after `c.admitted = t` (line 721). `_admit` is called once, line 621, in `_tick`. So `connect` runs only on an admission.

**The finding:** with `Warden(watermark=…)` (off by default) the network certifier opens one more connection to the server with no admission.
- Line 584, in `run()`: `opening = asyncio.create_task(self._open_watermark()) if self.watermark is not None else None`: at the run's start, before any tick.
- Line 886, in `_open_watermark`: `_, writer = await self.watermark()`.
- `benchmarks/network_traces/warden_serve.py:780` gives it the same address as `connect`: `watermark=(lambda: asyncio.open_connection(*upstream)) if watermark else None`.
- The code reads no byte from it: the reader half is dropped at line 886, and `_watermark_writer` is only written and closed (lines 893, 928-940, 946-950). The module docstring says so (lines 122-131).
- One thing under that [argued, from asyncio's source, not run]: the writer keeps the reader, so bytes the server sends on that connection sit unread in asyncio's buffer, up to about twice its 64 KiB limit, until the connection closes. None reaches a cell, a record or a client.

So "the server's connection is opened only on the client's admission" is true of `connect` and false of the device's connections to the server when the watermark is on. "These cells are all the server's bytes the device holds" is true of what the code reads; with the watermark on, the process also holds that unread buffer.

**The ruling I need:** (a) job F's test as you wrote it, on `connect` alone, named for `connect`, plus a fourth test that the watermark stream is opened with no admission and never read (a reader that fails the test if read); or (b) wait for soundness. I write nothing until you say. Nothing is pushed or opened.

