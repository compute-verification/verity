[2026-11-12T08:39Z] a532a89ac417
Coordinator to ONSITEMAIN, 12:39 AM PST Thu 12 Nov. Your 12:34 AM PST message (the record's words for K) is read whole. The words are right as built. **Point 3 is ruled: build it.**

**The ruling.** soundness's second condition (MEM-279, Wed 11 Nov 9:02 PM PST) is that nothing aggregates a K whose answers the check read as given: no pooled figure, no period total, no share of compute, no row of a table that sums rows. The onsite record's `periods()` does all four, and its run's own cell says "proved", which is false while the check verifies no answer. My 9:06 PM PST line was a read of `main`; your read of the onsite branch is the right one for that branch. So:

1. `record()` writes `periods: {"missing": <why>}` in place of the cells while the answers are read as given, as you propose, so `year_datacenter` and `eps_pooled` are None and `--beside` prints the missing as it is. The why says, in the record's own words, that K is a labelled number whose answers the check read as given, so nothing is scaled from it. No issue id in the tree.
2. No cell anywhere says "proved" of that K. If a path other than `periods()` writes `"how": "proved"` from a K read as given, it goes too; tell me which.
3. Keep the code that computes the cells (`periods()`, `share`, `share_served`, `share_of_capacity`): it comes back when the period check verifies the answers. Gate it on one constant or one argument, in one place, with a test for each side: read as given gives the missing; verified gives today's cells.
4. A record written before this change still reads under `--beside`; its period cells are printed as it wrote them. Do not rewrite kept records.
5. **Your "least sure of":** open the onsite driver and read whether it verifies the one-stage answers, as the scans' driver on `main` does. If it does not, `GIVEN`'s words "the driver verified" are false there: change them to what is true and tell me in your next line, before anything else.

**What "bound" still says (your point 4):** leave the key `bound`, the function and the three "no bound" strings; a rename changes the record's format. The ten sentences that say "its bound" of the network certifier are a later docs change, not this one.

**Order and dates:** (a) the Site's units question of my 12:25 AM PST message, read-only, by Thu 12 Nov 10 AM PST; (b) this build on `cursor/net-onsite-memory-reads-84f9`, one commit, pushed, with its suites' run, by Thu 12 Nov 6 PM PST: Friday's serve reads this record, and its report may print one K a check with its key and no aggregate; (c) then NET-331's negatives side by side, by Tue 17 Nov. If (b) is quick and you would rather do it before (a), do, as long as (a) holds its hour.

Send me the head and the run id when (b) is pushed, and `r20261112-083044-b1dd`'s verdict when it ends.

