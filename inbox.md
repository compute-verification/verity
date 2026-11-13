[2026-11-13T09:13Z] dcff7dd6d9e3
WALKER, Fri 13 Nov, 1:20 AM PST. TRN-145.
**What grows with the context is the reader valuing the causal mask word by word. Every softmax operand's words are put through `unselected`, and every softmax row through `row_of` → `value`, and each word valued is memoized in `value_of`. That is time and memory in ctx² × heads × layers, not in the description. A smaller part is a Reader-wide cache of leaf runs that is never freed. I'm sending this before 72e0 ends because its traces already agree with d4d5's, and tracing has slowed both to hours. I'll change nothing until you rule.**

RAN, tracing allocations with tracemalloc from the reading's start (sites read oldest to newest; lines are #9676's `work.py`):
- r20261113-072936-d4d5, s(2,256,512): more than 70 min on nodes 1330–1336 of TrainStep's 5639. The largest live sites are 0.64 GiB in 6,199,426 objects at `value_of[j_, leaf_] = "g"` (851), reached from `masked_rows` (966) → `row_of` (909) → `value` (934), and 0.18 GiB in 6,201,505 objects of `member_words` slices from `row_of` (933).
- r20261113-080221-72e0, tiny(2,1024,64): at node 5158 of 10845 it holds 1.33 GiB traced (RSS 7.1 GiB, peak 8.9 GiB).
  - 0.53 GiB in 12,582,397 objects is `_leaf_runs`' exact run tuples (578), kept in the Reader's `self._leaves` for the whole reading.
  - 0.30 GiB in 2,715,087 objects is `value_of` entries from `softmax_rows`' `got = [unselected(*w) for w in a.refs]` (955) → `value` of each select's condition (873 → 861).
- RAN r20261113-055829-e320 earlier: tiny(2,1024,32000) under #9676 read in 2,043 s at a 33.49 GiB peak, while s(2,64,512) read in 146 s at 2.96 GiB.
- RAN r20261113-061108-54ec, s-t20 under #9676, still running:
  - The first TrainStep summary ended near 120 GiB.
  - In the second, node 5170 again took about 10 min and raised the peak to 137.6 GiB (the stop is at 144). RSS then fell back to 95.1 GiB, and is 100.1 GiB at node 8471 of 16,403.
  - INFERRED: most of the RSS is transient per-word lists and the arenas they leave behind, since traced live memory is a small part of RSS in both traces.

What I would change, in order of size:
1. `value` stops memoizing "g" for a word whose node is an input or not a primitive. That is an O(1) test, and it is most of what `row_of` stores. Same answers, less memory, no time saved.
2. `softmax_rows` streams its words instead of building `got`, `cells` and `slices` as lists of every word. `self._leaves` keeps its runs as numpy arrays and is freed with the summary rather than with the Reader. That gives a flat memory profile, but the time is unchanged.
3. The time and the rest: "the mask makes these rows differ" is decided by valuing each select's condition word, one per score word. Making it O(description) means reading the mask from how it is built: a batch of selects whose condition compares two index ranges, decided per Definition and multiplied through the batch. That changes a reading rule, not a cache, and it is your call whether it belongs in #9676 or its own PR.

#9676 as it stands removes the 365M-word cache and keeps every `Work` equal. My proposal: land it as is, and do 1 and 2 in it or after it on your word, with 3 as its own piece.

