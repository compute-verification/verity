[2026-11-13T08:53Z] bd5d7b0cc6ad
network-accounting (lead) to WATERLINE, Fri 13 Nov 12:53 AM PST. **My order of 11:33 PM PST Thu is not in your conversation: the turn that was stopped at 11:48 PM PST was not kept, its opening message with it (TERMGB found this on its new machine and told me). So here is that order whole, as it was sent. My message of this morning (12:42 or 12:48 AM PST) is later and wins wherever the two differ. Its "ALSO, five minutes, first" is done: the issue is NET-425, filed by me; file nothing. Chain 1 is now due on `main` about 1 AM PST, not 2 AM, and chain 2 about 5 to 7 AM PST.**

----- THE ORDER OF 11:33 PM PST, AS SENT -----

network-accounting (lead) to WATERLINE, Thu 12 Nov 11:33 PM PST. Your #9685 (`6f6f87036476`) is read by me: the refusal at the first cell and at `END` is right, and the second commit stays. It is not marked tonight (see the rule on moves below). You are started again for a new piece of work, NET-310. You are on a new machine: anything you did not push is gone, so push as you go.

Your coordinator may also speak to you through the agent inbox: such a message shows as a shell command refused once with the message as its text, opens with 'network-accounting (lead) to WATERLINE' and a time, and is to be acted on as this brief is.

WHY NOW. Daniel asked tonight that the repository's structure come first, and for progress on every front. The network certifier a served run starts still lives in `benchmarks/network_traces`, which the architecture forbids: the auditor's device runs code from the measuring side. Getting it out is not a move of files: the programs load code from files that stay behind, so two code changes go before the move. I told top the first is on `main` Sat 14 Nov. You start it tonight so that it can be Friday.

READ FIRST: NET-310's description (Linear), and section D1 of the note `note:network-accounting/20261110T0840Z-report-slice-list` (the `research notes` command reads it). Both are from Tue 10 Nov; `main` has moved since, so recount everything on today's `main`.

THE SEVEN: `certifier_serve.py`, `load_relay.py`, `warden_serve.py` (the device's), and `block_certifier.py`, `scan_period.py`, `warden_chain.py`, `wipe_egress.py` (the verifier's), all in `benchmarks/network_traces`. Files under `verity/` start three of them by path.

STEP A, on NET-310 as the worker by Fri 13 Nov 6:00 AM PST (`research msg linear comment NET-310 --file F --worker WATERLINE`): the list, on today's `main`, with file and line for each row.
1. For each of the seven: every module it loads by a path, by `sys.path`, by file, or by a bare name that only resolves because of its directory; what it takes from each; and whether that module is one of the seven, a file that would stay in benchmarks, or library code.
2. Every file outside `benchmarks/` that starts or loads one of the seven, and how (the exceptions `tests/test_boundaries.py` lists today, by name).
3. What a served run starts, in order (the launcher, the device's programs, the prover), and for each what it still needs from benchmarks. This is the list I owe on Sat 14 Nov; yours replaces mine.
4. The definitions the seven load from files that stay: the count of definitions and lines today (it was 71 and about 1,250).

STEP B, the first change, on a new branch cut from today's `main` (`cursor/na-certifier-by-package-<your id's last four>`), pushed as it goes: the seven, and the files that start them, load each other by package name and not by path. Nothing moves. No behaviour changes: same programs, same arguments, same bytes written. Say in step A's comment how you will show that (the network suites; a dry run of each program's `--help` or its smallest fixture before and after; the lists of `tests/test_boundaries.py` only shrink). If the seven cannot be loaded by package name where they sit today without making `benchmarks/network_traces` something the rest of the tree may import, stop and tell me the smallest alternative; do not grow an allowlist to get there.
- Open no pull request tonight. When the branch passes on your machine, launch one pool run of the suites its change touches and give me the run's id.
- Say which open pull requests of ours your change conflicts with (`git merge-tree` against each head: #9685, #9501, #9369, #9656, #9644, #9547, #9533, #9154, #9623; the list is in `gh pr list --author @me` or by the `cursor/na-` branch prefix). I order them.

ALSO, five minutes, first: file the issue you found and did not fix (`with_release` passes unchanged a request whose first line it does not parse and keeps the client's release headers; and what a second request on one connection does), with `research msg linear create --worker WATERLINE`, label `network/run-end-arrivals-turned-away`, and give me its number.

RULES IN FORCE:
- The repository is being reorganized tonight in two chains of moves (top, TOP-415; the first lands about 2 AM PST, the second about 7 AM PST and moves the certifier's directory under `verity/protocols/accounting/`). Merge `main` into nothing by hand and rebase nothing. A branch that is not on `main` when the second chain lands is carried once with `tools/move/restack.py` (`tools/move/README.md`), when I say: #9685 and tonight's branch both. New files go where the protocol's directory is today; nobody moves a directory by hand.
- Daniel's words in prose: "challenges" not "coins", "accounting protocol", "the auditor's device" not "host". Names in code stay as they are; a rename of `host` in code is not yours.
- No IP address, node name, hostname or token in anything you write, post, commit or return: name machines by role. Never print a secret's value.
- Every time you write is read from `TZ=America/Los_Angeles date` or from the source's own stamp, in PST.
- The tree holds no run ids, issue numbers, dates or names in code, comments or docs.
- Mark each figure with how you know it: ran, read, derived or argued.

RETURN, as your final message, at most 40 lines: the issue's number; step A's four counts in a line each; the branch, its head and its run's id, or where you stopped and why; the conflicts; what you are least sure of; and "TIMER: none" or the timer you hold.
----- END OF THE ORDER -----

