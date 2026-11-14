[2026-11-14T02:58Z] 374e3fabeeed
ONSITEMAIN, Fri 13 Nov 6:58 PM PST.

**The kept dir of serve `r20261109-064213-20a2` is listed. `served_verdicts.sh v3c` from `main`'s layout cannot run on it as it stands, and it must not run in place. The front did run on Monday, and the audited window is the front's group with its advice written, but the group is not four requests.**

**What I did** [ran]: one listing and two reads of the files it showed, through `research pods ssh`, each a Python program on stdin that opens files for reading only. Under ten seconds together, no pool slot, nothing written, no run launched. The node the serve ran on did not answer on ssh's port; the kept dir is on the CPU node the follow-on and my rehearsal ran on, and that is what I listed.

**1. Your four unknowns**

| Asked | Answer | Mark |
|---|---|---|
| Did the serve set `CERTIFIER_FRONT`? | Yes. Exactly one front's fetch is kept, with its `session.json` beside it: gate not local, exit 0, "the rehearsal ended", verdict accepted. Only the front writes it. | read |
| Which `LOAD`? | No kept file says, and the run's launch record on my VM does not name it. The files fit `LOAD=front` alone: every request the run recorded is in the audited window, and they are as many as the front's group. | argued |
| Where is the front's fetch? | Under `private/certifier/<run>/fetch.tar.gz`. `private/certifier-front` and its log are not in the kept dir. | read |
| Is the audited window the front's four requests, with its advice written? | It is the front's group and nothing else, and its advice is written (`advice.bin`, non-empty, with its registration). **It is not four requests.** The count is the kept `session.json`'s `prompts.count`; I leave it out of this line. The class is `8,32,4,4,16`, not `main`'s default. | read |

**2. What the kept dir is** [read]
- **It is the follow-on's copy, not the serve dir.** It holds `private/deferred/deferred.env`, the deferral's list and sums, and the follow-on's outputs: the window commit, V3c's audit and `period.json`, the chain's and the period check's logs, the record.
- **It lacks `serve.json` and `advice_trace.jsonl`,** and `private/window.json`. The deferral's list never carried them.
- **It holds more than the deferral's list:** the window's advice, its row dir, the weights key and the rotary table are there, copied later.
- It is kept until 15 Dec at priority `keep`, and the login can write it.
- **On the onsite line's tree the split did run whole on it, Monday:** the chain exit 0 and accepted, the period check exit 0, the record composed `accept` [read: the kept verdict lines' first words]. From `main`'s layout nothing has.

**3. What stops `main`'s `v3c` on it, by the lines** [read: `main` at `6953e38e55a4`; nothing run]

Because the dir holds a `deferred.env`, `main`'s `v3c` takes the follow-on's path. There:
- **The product.** The file says `PRODUCT=remote-network-no-pouw`. `main`'s preflight takes `onsite` or `remote` and exits 3 on any other. Setting another in the environment is refused as a clash with the file.
- **The certifier's two commands.** The file carries the onsite line's, which pass `--certifier-key`. `main`'s piece takes no such flag.
- **The tree.** The file names the onsite line's commit `9068336929e0`; any other tree needs `FOLLOW_ANY_TREE=1`.

On a copy without `private/deferred`, `main`'s plain `v3c` path is taken. There:
- **`serve.json` is read first and is not there.** `window-circuit --serve-run D` and `window-commit --serve-run D` both load it. The step ends at the window circuit, exit 12.
- **The chain's endpoint step** copies `advice_trace.jsonl`, `requests.jsonl`, `windows.jsonl` and `serve.json` from the serve dir, and refuses to run where one is missing. Two are missing.

**In place it would write over the run of record's outputs:** the V3c directory, the window commit, the window-circuit record, and the certifier's directories beside the fetch. So any run is on a copy.

**4. What would have to be moved or named**
- **Moved, to a new directory:** `requests.jsonl`, `windows.jsonl`, `seals.jsonl`, `weights_registration.json`, the window's advice and row dir, the weights key, the rotary table, and the front's `fetch.tar.gz` with its `session.json`. Not `private/deferred`, and none of the follow-on's outputs.
- **Fetched from the served node, if it still holds its serve dir:** `serve.json` and `advice_trace.jsonl`. I can't say it does: that node did not answer. Without these two there is no run from `main`'s layout, whatever else is named.
- **Named:** `D` the copy, `K` the window, `V3C_CLASS=8,32,4,4,16`, `PRODUCT=onsite`, the two certifier commands in `main`'s form, `V3C_CHALLENGER`, and the checkpoint's snapshot on the node, which the window commit reads from outside the serve dir.

**5. Not read**
- Whether `main`'s code reads the onsite line's row dir, advice and fetch as that tree wrote them. This is the larger risk: the window was served from another tree.
- Whether `main`'s window circuit builds the five-field class. `main` names it in two test files only.
- Whether the served node's serve dir still exists.

So the cheaper way is not this dir: a `v3c` step from `main` needs a serve dir that `main`'s own serve wrote, with its `serve.json` and `advice_trace.jsonl`.

Least sure of: the `LOAD`. I argue it from the requests; no file names it.

