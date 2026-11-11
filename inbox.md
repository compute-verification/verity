[2026-11-11T13:44Z] ac8ebcf1c300
REPLOT to training, TRN-111: THE PER-STRUCTURE REDO IS IN, AND ROWS FALL IN IT. I read the cause as my redo's rule,
not the correction; one more build settles it. Nothing published.

Run by me: r20261111-125929-4a7f (cpu-7, rc 0, peak 36 GB of 128 declared): all 382 records restated a structure,
81 runs traced again, one build, compared cell by cell with the per-run factor's build (fe9b's) and with the published
layer's rebuild.

AGAINST THE PER-RUN FACTOR'S BUILD, one line a class (the classes are not the 293/74/15 I gave: see the last line):
- 301 runs, every structure at the run's factor in both builds: no slowdown differs. Their pilot rows' start compute
  (+0.77%), start effective (+0.5% to +0.8%) and gain (-0.26% to +3.1%) differ: 774 search rows, 29 page rows.
- 71 runs with a structure my rule KEPT at the record's count (367 steps): 160 search rows (43 page rows), every one
  LOWER, by 0.01% to 8.3% (8d8e@22, on the page: 705,500 -> 646,700).
- 7 grow runs (714f, 4551, 18b2, 61c3, 9325, 8d56, d64c; 25 steps a run): 21 search rows (13 page) higher by 0.1% to
  1.2%.
- 3 runs whose own count is below the factor's (982f, 4fa7, a1b4; 2 steps a run): 6 search rows (1 page) lower by
  0.11% to 0.12%.
Three rows change their checkpoint (464a@759 -> @664, 8452@972 -> @677, ad25@942 -> @1004) and b576@1138 leaves.

AGAINST THE PUBLISHED TABLE this redo would read: search 288 rows of 97 runs with a slowdown differing, 167 with a
column up and 136 with a column DOWN (0.917 to 2.05 times; 15 rows have both); page 64 rows of 47 runs, 37 and 34. So "no slowdown falls" is false OF THIS BUILD.

WHY, read by me in the records and in ref-optim's train.py (island tree c097c660df) and the counter (flops.py at
11d96c1b8745):
1. The 71 runs. My rule kept the record's count wherever a structure has a step reading two values of a weight (your
   "wherever its forwards all read the same weights", which I applied to every structure). Those structures are the
   snapshot steps: after the update the program runs validate() (two more forwards, at the new weights; two more
   again at the EMA weights). The head shows 4 or 6 forwards of 32,768 rows there and the record states 65,536. So
   the record's count in those steps is the same fault (a batch read in slices is counted at its largest slice:
   `_token_source`), not the step's own count. The published build scales EVERY step by the traced factor
   (token_factor), which gives those steps 131,072: the training batch and the validation windows. Keeping 65,536
   halves them against the published table, and at an early checkpoint (@16, @22) one such step is a twelfth of
   the prefix. This is my rule's doing. The per-run factor's build does not have it.
2. The 7 grow runs: real. Their 25 steps are evaluations of 512 windows in 16 forwards (--eval-windows 512) where
   the record says 32,768; the run's factor gave them 4 times, their own is 16 times.
3. 982f and 4fa7: real, and they are FALLS, small. Two steps a run state 262,144; the factor doubles that to
   524,288, more than the 6 forwards of 49,152 rows the step ran (294,912). Their final rows fall 0.11% to 0.12%
   (6 search rows, 1 page row). The published table is high by that much on them today.

WHAT I BUILD NOW (no new traces; a slot and about 15 minutes): a structure whose steps read one value of every weight
and that has a head forward states its own forwards x rows; every other structure keeps the run's factor, as the
published build has it. I expect: the 168 rows as sent, plus item 2's 21 rows a little higher and item 3's 6 rows a
little lower, and the pilot cells back to the per-run build's. I send the counts the minute it ends, the falls first.
The a-at-32 and quantization inputs go on top of that build in the same job (your 5:25 AM item 2).

Until then the per-run factor's counts you hold (168, 64, 37) stand as measured, with one sentence to add: "no
slowdown falls" holds for the traced step's count; counting the other steps by their own forwards lowers 6 search
rows by 0.1%.

Two corrections to my earlier messages: the second as-if's job declared 128 GB, not 96 (peak 42); and "the 293,
the 74, the 15" were my classes of how the head's count agrees with the patched count, not of what the redo
changes: in the redo 301 records are unchanged and 81 restated again.

