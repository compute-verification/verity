[2026-11-13T23:38Z] 6ba40e602532
WALKER, Fri 13 Nov, 3:38 PM PST.

**s-t20's full step ended with the fix, exit status 0: 70 claims, none refused, N 104,421,392,384. Its highest resident memory was 65.99 GiB, under the 82.8 GiB stop, and it took 5 h 58 min.**

RAN r20261113-172658-0505 (30a0a4935c: cfc639f925 with #9676 and #9730 merged, a local commit; a node of 23 cores and 503.7 GiB; ask 92 GB), untraced:

| | s-t20, with the fix |
|---|---|
| build | 874.5 s, highest 57.52 GiB |
| resident after the build | 27.63 GiB, and 14.16 GiB after one collection |
| reading before the pass | 5,551.0 s, highest 59.55 GiB |
| resident at the pass's start | 57.51 GiB |
| the pass | 14,634.9 s: 1 point, 656,700,577,858 words, 1,506 products |
| highest resident in the pass | 65.99 GiB, the run's highest; 58.35 GiB at the pass's end |
| seconds building indices | 1,408.5 s, 9.6% of the pass: 1,254,890 builds over 204,200,116,532 entries, the largest 65,536,000 |
| the reading, whole | 20,196.3 s, the pass 72% of it |
| the run | 21,494 s |
| claims, refused | 70, none |
| N | 104,421,392,384 |
| origins, nonfinite nodes, NaN nodes | 0, 24, 0 |
| the `Work` as JSON | 245,790 bytes, sha256 3c8efa45257c6ee9e94adca6b0b53a5144ece77f05a9a26bf708dcda68962896 |

- **Peak.** RAN: the pass's highest is reached around an iteration's 247th product, 65.3 GiB in iteration 1 and 65.99 GiB in iteration 2, 8.5 GiB above the pass's start. RAN b24e, before the fix: 65.3 GiB at the same products. INFERRED: the pass's highest is no longer the takes'.
- **The embedding table's gradient.** RAN, read every 300 s: 60.9 to 61.5 GiB resident in iteration 1's and 61.9 to 62.5 GiB in iteration 2's. RAN b24e, before the fix: 134.4 GiB there, then its stop at 153.0 GiB.
- **Resident after the build and at the pass's start.** RAN: 14.16 GiB after the build and one collection, 57.51 GiB at the pass's start. The reading before the pass adds 43.35 GiB at ctx 1,024, against about 1.0 GiB at s(2,64,512) (RAN d990 and 9ed5). It is not the pass's.
- **Time.** RAN: the five products after the peak, an iteration's 248th to 252nd, took about 2,400 s in iteration 1 and about 1,800 s in iteration 2, each read to within 300 s. b24e took about 1,500 s over the same five before the fix (RAN, its heartbeats), on another node.
- **The index builds.** INFERRED from the counts: 12,290 of the builds are the table's takes, 2,048 at an iteration's start and 4,097 in its gradient, each of 16,384,000 entries, 98.6% of the entries built. RAN ec35: the builds were 1.7% of the pass at s(2,128,2048).
- **The ask.** INFERRED before the run, a peak of about 69 GiB; RAN, 65.99 GiB. The 92 GB ask left 16.8 GiB under its stop.

The reader's caches, no run:

- **`Reader._leaves`'s entry count** (READ `work.py`, the same lines in #9558's tree and the run's): one entry for each reference object `read` is asked about, keyed by `id(refs)` with the object kept. `read` is asked for every argument of every non-input node of a body being summarized, and an argument's `refs` is the body's own object. So the count is the arguments of the Program's distinct bodies, each once, however many contexts, depths or iterations summarize a body. INFERRED: it follows the descriptor's size, and ctx only through the nodes the frontend's bodies hold. RAN 9ed5, the trace: 7,552 entries at s(2,64,512), reaching 8,907,364 objects.
- **`_leaf_runs`'s `limit` does not bound an entry** (READ). It bounds one `Strided` grid: at most 32,768 runs, past which the grid is one hull. A `Concat`'s entry is the sum over its parts with no cap, and an `Explicit` list gives one run for each break in contiguity, up to its count. A `Concat` cut from a `PartLog` walks the whole prefix `log.parts[:j]` again at each cut, where `_parts` shares it through `Reader.logs`. INFERRED from that loop: n cuts of one log hold order n² runs between them.

What I did with it:

- RAN `research queue edit 9558`: the body's s-t20 section now has this table, the peak, the time and the figures, and "Not checked, and open" says s-t20 was read once, on the local merge. The live body equals my file (`cmp`). #9558's head is still cfc639f925, and it is unmarked.
- No commit this turn. The two docstring nits are a patch on the VM, held for the next commit on #9558.
- Nothing crossed #9558, #9676 or #9730, and no merge of `main`.
- Nothing of mine is running. The run's `out/` holds `st20.json`, `st20.work.json` and `st20.log`.

