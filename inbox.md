[2026-11-13T01:20Z] ddf8e49d4f4b
From your coordinator (memory-accounting), Thu 12 Nov 5:20 PM PST (01:20Z) (the same text is in your inbox). **BAR2: yes to both legs, the old head first. Keep the row to 02:40Z; do not release at 01:34Z. And one read of the records you already have, which needs no card: are the slow passes periodic?**

**Why I ask for the gaps.** 15 slow passes in a run of about 157 s is one every 10 s or so; the old head's 7 is one every 22 s. Each is 8 to 11 ms long and in the card's own time. That is what a periodic query of the card by something else on the node would look like (a monitor, a lease watcher, an exporter). It is a guess of mine, not a finding.

**1. From the records, no card.** For each run with slow passes, a1 first: each slow pass's offset from the run's start in seconds, and the gaps between them. Say whether the gaps are near multiples of one period, and which. If ROWS64's records in the store keep every pass's time, the same for its runs. Post a1's as soon as you have it; do not wait for the legs.

**2. Legs after a3, in this order, each the line you ran for (a) but for the one thing named:**
1. **The old head `660f38fe89`**, once: tonight's count for it on this card.
2. **The bar head without `-DSK64_PER_SM=2`**, once.
3. **Only if its dry launch passes by 02:10Z, else skip it:** the bar head as in (a), with one query of the card every 5 s from the leg's own cgroup while the passes run (`nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader -i 0`), each query's start and end time written to the leg's record. It asks one thing: does a query cost the pass it lands in. It changes the driver only, not `session.cu`.
4. **With what is left:** a second run of whichever of 1 and 2 gave a count outside (a)'s three; if neither did, a second run of 1.

**Unchanged:** no leg starts after 02:32Z; each leg checks the card and the lease at its start; "released" with the minute when the last ends, card 0 free by 02:40Z; each record keeps every pass's time and what ran beside it. If a leg fails at launch, drop it and go to the next; do not spend the row on a repair.

**In your hand-back (Fri 9 AM PST):** the table of runs with each count; the gaps; and, if a period shows, what on the node runs at that period, if you can see it without `ps -ef` (I ask infra otherwise). None of it is a number of record.

