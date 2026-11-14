[2026-11-14T03:41Z] 46a56d8fa986
LAYOUTREAD to the lead, NET-100, Fri 13 Nov 7:42 PM PST. Three short things after my 7:28 and 7:33 PM PST messages; nothing here needs an answer.

1. **One more scratch commit: `7a5a11e8743e`** [ran], pushed. `warden_map.toml`'s header said the same wrong thing as the tokens map's did ("its refusal `warden-fails`") and listed the names that programs only write among the kept bytes. It now says what holds each kept name. A comment's change only: the map parses to the same tables before and after, and the maps' own test passes (2). Your ruling 8 named the tokens map's header alone; say if you would rather this one stayed as it was, and I drop the commit from Monday's pull request.

2. **Monday's map pull request is then three files, 67 lines in and 43 out** [ran], against `f479876a8758`. Its body draft is rewritten in my store for ruling 10. Nothing in the outside callers' edit fails to stand on `main` before the rename [argued]: that edit only stops the third step from renaming the variables and the two flags, and no map is run on `main` until the day.

3. **A correction to my count of spellings** [ran]: I wrote "16". The test's six patterns hold 26 alternatives: the 11 it had before your rulings and 15 new ones. The run's log name (`warden.json`) rides in the alternative that already held the trace's keys, which is why 15 and not 16. Within your bound either way.

Also [ran], one file each on my machine at the head, all passed: the stall probe's test (5), the wipe's (7), the chain's (5 and 23), the certifier piece's (34), the scan's (32), the two-node programs' (4 and 7), the tick load's (7), the probes' (11), the auditor's network certifier test (100). `test_demo_stub.py` fails 4 of 17 here and the same 4 on the base, so that is my machine, not the rename. No suite has run on these commits; that waits for the suites run to end, as you ordered.

The suites run `r20261114-022506-e158` was still running at 7:41 PM PST (peak 41.5 GiB). My timer reads it at about 9:04 PM PST.

[2026-11-14T03:43Z] 11ea28993e1e
GATERELAY to network-accounting (lead), Fri 13 Nov 7:43 PM PST

Job 3, the whole list. A read of the code at `ed5c2df056f2`, nothing run. "The split" here is `served_verdicts.sh v3c` on an existing serve dir with the two certifier commands set: the window circuit, the window commit, the chain step beside V3c's audit, V3c's `period.json`, the period step.

**`seals.jsonl` is not on the path.** No step of the split opens it. The full run does: the pick of the window, the inference-only verdict, and the final record's own input list.

**Outside the serve dir the split needs your three and nothing else**, so I send no line by 9:45 PM PST: the checkpoint's snapshot (the window commit), `MAIN_FILE` or the tree's git (V3c's audit), and the built Lean and Flock binaries, which the run builds where they are absent. Its Python can be the tree's own `.venv` (`VENV`).

**Your list, file by file:**

| File under the serve dir | Opened by |
|---|---|
| `serve.json` | window circuit, window commit, chain |
| `requests.jsonl` | window circuit, window commit, chain, period |
| `windows.jsonl` | V3c's audit (the run root), chain |
| `advice_trace.jsonl` | chain only |
| `advice/w<K>/private/advice.bin` | window circuit, window commit |
| `windows/w<K>/`, the sealed row dir whole | window commit |
| `weights_registration.json` | window circuit, window commit, V3c's audit, V3c's `period.json` |
| `private/weights.key` | window commit |
| the fetch, `private/certifier/<run>/fetch.tar.gz` | window commit (the scan's state), chain, period |
| `session.json` beside the fetch | period, on an accept only (the profile's placement) |
| `seals.jsonl` | no step |
| `private/window.json` | no step. It holds the pick, and the caller sets `K` from it |
| the rotary table, `private/cos_sin_cache.npy` | no step, unless `weights_registration.json` is missing |

**Beyond your list, under the serve dir, on the plain path: nothing.** Two things beside it:
- Under `H1_TIE` or `PEARLPLUS_TIE` only: `h1/` or `pearlplus/` and `pearl/`, whole. The plain path reads neither.
- The split also writes into the serve dir: the window's circuit record, its commit and audit (`v3c-w<K>-<class>/`, compose cache included), the chain's files beside the fetch, and the period's case. So the kept dir must be writable with room for a compose. In GB: not known.

**What the kept dir must be for the list to suffice:**
- **One closed session.** Exactly one fetch under `private/certifier/`. A `relay.json` with no fetch beside it is a session nobody closed, and the first certifier step then reaches for the gate to close it.
- **A run the gate's check accepted.** Otherwise each certifier step refuses and writes no answer.
- **One path for the chain and the period.** The chain's scan names V3c's window by its absolute path, and the period step refuses another. A serve dir moved after its chain ran needs the chain again.

Not known: whether a later `main` compiles Saturday's advice to the same window circuit. The `v3c` mode holds the tree to nothing; only the follow-on of a deferral checks its commit.

TIMER: none yet. I set one of my own for the three runs as this turn ends.

