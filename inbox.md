[2026-11-13T16:56Z] bd0f64b5e378
WALKER, Fri 13 Nov, 8:56 AM PST.

**The trace at s(2,64,512) ended, and its five largest sites differ between two tables: the first table's five are all the Boolean trace's, and the other four tables hold one set of five, four of them the reader's.** This is the case you named for the larger shape; I have not launched it. The fix for #9558 is pushed (cfc639f925) and its before/after readings at A, B, C and D are running; that is a separate message.

RAN r20261113-154950-9ed5: #9730's head 5efe513c14, s(2,64,512), 12 GB ask, rc 0 in 3,674 s. The build took 49.8 s, the reading 2,838.7 s under the tracer, N 4,462,739,456.

## The five largest sites at each table

Each row is a site (file:line, function), its live bytes, and the table's line gives the share of all traced bytes the five cover. `trace.py` is `verity/circuits/boolean/trace.py`; `work.py` is the reader. The harness's own allocations are 0.0 GiB in every table. All RAN.

| table | traced bytes, blocks | the five cover | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|---|
| 81.6 s, RSS 1.33 GiB | 129,055,001 in 1,760,867 | 0.7406 | `trace.py:125 _br` 56,528,448 | `trace.py:242 TL.__xor__` 11,803,848 | `trace.py:633 _prune` 10,483,944 | `trace.py:144 _tog` 8,399,240 | `trace.py:245 TL.__xor__` 8,367,168 |
| 734.0 s, RSS 2.24 GiB | 392,579,428 in 7,142,325 | 0.7956 | `work.py:624 _leaf_runs.grid` 101,689,808 | `work.py:603 _parts.walk` 77,004,288 | `work.py:656 _leaf_runs.walk` 65,850,880 | `work.py:1344 Reader.summary` 47,337,944 | `trace.py:633 _prune` 20,450,640 |
| 1,700.7 s, RSS 3.62 GiB | 699,000,549 in 13,210,165 | 0.8994 | `work.py:624` 308,035,336 | `work.py:656` 131,420,904 | `work.py:1344` 116,019,448 | `work.py:603` 52,741,712 | `trace.py:633` 20,450,640 |
| 2,731.6 s, RSS 4.69 GiB | 862,330,865 in 16,200,884 | 0.9017 | `work.py:624` 308,035,336 | `work.py:1344` 189,884,200 | `work.py:656` 131,420,904 | `work.py:603` 127,770,704 | `trace.py:633` 20,450,640 |
| the reading's end, RSS 5.08 GiB | 794,866,740 in 14,897,014 | 0.9102 | `work.py:624` 308,035,336 | `work.py:1344` 189,885,360 | `work.py:656` 131,420,904 | `work.py:603` 73,728,000 | `trace.py:633` 20,450,640 |

- **Which tables differ.** RAN: table 1 against each of the other four. Tables 2 to 5 hold the same five sites; only their order changes. Table 1 is 31 s into the reading, where the Boolean lowering's sets are what is live. `trace.py:633 _prune` is the one site in all five tables.
- **Two sites stop growing.** RAN: `work.py:624` holds 308,035,336 bytes from 1,700.7 s to the end, and `work.py:656` 131,420,904. `work.py:1344` grows to the end, and `work.py:603` goes up and down (the column reads).

## Who holds it at the reading's end

RAN, a census of the objects each of the reader's holders reaches after the reading: together 0.652 GiB in 13,897,015 objects, of 0.741 GiB traced. 0.009 GiB in 100,284 objects is reached by none of them.

| holder | entries | reaches | in objects | of it |
|---|---|---|---|---|
| `Reader._leaves` | 7,552 | 0.400 GiB | 8,907,364 | tuples 0.230, ints 0.142, lists 0.027 |
| `Reader.memo` (the summaries) | 77 | 0.252 GiB | 4,983,773 | tuples 0.186, ints 0.064 |
| `Reader._match`, `Reader.logs`, the `Work`, the reasons | 67, 0, -, - | 0.000 GiB | under 5,000 each | |

- **The largest holder is a cache with no bound.** READ `work.py`: `Reader.leaves` keeps `_leaf_runs(refs)` for every reference object it is asked about, by `id(refs)`, for the reader's life. INFERRED from the sizes: its 0.400 GiB is sites `work.py:624` and `work.py:656` (0.287 + 0.122 GiB), and `memo`'s 0.252 GiB is `work.py:1344` and `work.py:603` (0.177 + 0.069 GiB), each site's parts kept in its `Site`.
- **Whose code.** RAN `git log -S`: `_leaf_runs`, `Reader._leaves` and `_parts` are on `main` (36d4203423 and 7d89f231d4), and #9676 and #9730 did not add them. Nothing in this trace is #9558's: this tree has no pass.

## What the trace covers and what it does not

- **It covers the reading before the pass, on #9676 and #9730, at one shape.** How any site grows with ctx or vocab is not in it. That needs a second shape.
- **It does not cover the built Program.** RAN: resident was 1.25 GiB when tracing started, after the build.
- **Its resident figures are the tracer's, not the reading's.** RAN: at the reading's end 5.08 GiB resident, of which 0.741 GiB is traced live and 0.945 GiB is the tracer's own tables. RAN r20261113-160433-d990, the same shape untraced: 2.234 GiB resident at the pass's start, so the reading adds about 1.0 GiB to the Program's 1.25. The traced 0.741 GiB is about three quarters of that.
- **It names no cause of s-t20's stop.** That was `Walk._strided`'s index, in the pass, which this tree does not have.

**Next:** the before/after readings for the fix (four runs, old and new in each), then #9558's body, then s-t20's full step. I launch the larger shape of this trace only if you say so.

