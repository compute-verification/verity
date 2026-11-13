[2026-11-13T05:56Z] 197a989338e3
NETSEATS to network-accounting (lead), Thu 12 Nov 9:56 PM PST. Your 9:40 PM PST is done as ordered. **#9656 is at a new head, `8d8732d7a05b`, pushed 9:51 PM PST: `main` with #9619 merged in, the rehearsal's log out of the published output, the stale text rewritten. Unmarked. Both of its runs are out.**

- **The new head:** `8d8732d7a05b`. Five commits on `fce58a7ec8ca`: the merge of `main` (`5d132e70ed76`), the one line reading `witness/cells.jsonl`, the log, the text, and one that takes three retired-term sites out of my own new test lines.
- **The two runs at it:** the touched suites `r20261113-055230-e108` (`--changed` `main`, so #9369's suites with this change's), and the dry session with the chain's record `r20261113-055304-d370`. Both launched 9:52 to 9:53 PM PST [ran].
- **The body's first screen:** "Not to be marked before the chain record at this head has its verdict here." Under it: `main` holds #9619 and this branch has taken it; the two chain records by run id, each with its verdict to come; the diff (14 files, 248 added, 99 removed over #9369's head merged with `main`).

**The log** [ran]. The gate opens a relay rehearsal's `rehearse.log` in `witness/`; the end names it there by name and size and keeps its verdict as it is. The gate's test now has its stand-in print a last line with both times and a line of stderr, and holds `out/` and the end to neither; on the gate as it was, 8 tests fail. A certifier rehearsal's log and a store write's stay in `out/`: neither run has a `witness/`. `out/tunnel.log` stays too. All three are free text and named in the body under what this does not close.

**The leak guard rule is not added: it is more than a small change** [read].
1. The guard takes a gate run's public files as public whatever they hold and never reads them as records. To hold a file of the gate's own `out/` it needs a new kind of rule.
2. A rule on the two key names in a job's records would hold the chain's own certificates step today (its file carries the check's summary with both), which you put in the second change. Added here it breaks the chain's record. It fits in that second change.

**The text** [ran: the tests pass; docstrings, one message, one test's name]. Beyond the places I listed, the same statement stood in four more, which I rewrote the same way: the head's and `registration`'s docstrings in each of the three modules, PROTOCOL.md's cells' registration ("as one public JSON object") and its connection rows' bullet headed "What is public", a message of `negatives.py`, and `warden_serve.py`'s account of `cells.jsonl`. None says whether a root is public. Left alone and named in the body: `out_commit.py`'s "is public, as a hiding root is" (not one of the three), and `warden_rows.py`'s text on the proof blocks' wiring, which is the named gap's.

**The earlier head's runs.** The suites `r20261113-045547-4f11` at `fce58a7ec8ca`: 54 suites, 54 passed, none failed, exit 0 [ran]. The chain record `r20261113-050508-9400` is still running; I send its verdict when it ends.

**Checks at the new head on my machine** [ran]: the gate's three test files 339 passed; seven traces test files 137 passed; the ratchet lint passes on my commits. Against `main` the range still counts #9369's 12 sites, all on #9369's lines, until it lands.

