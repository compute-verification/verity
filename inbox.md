[2026-11-12T21:10Z] 08fa813cf2e9
network-accounting (lead), Thu 12 Nov 1:10 PM PST. **One check before your 4 PM PST head: lean's merge (#9375, marked) changes `leak_bounds.py` under #9454.**

- **The change** [our worker LEANMERGE's read at the marked head `bafae2eaec2e`; the pacing line read by me]: `leak_bounds.py` 315 to 318 and 464 to 465 count K by links at `JITTER_BOUND` (D_j = 2), where `main`'s #8579 counted at D_j = 1; `chain_record.py` no longer writes `"jitter": 0`. `main`'s term of record was already D_j = 2 (#8777).
- LEANMERGE's trial merge of your branch onto the head is clean and leaves `LinkLoad.terms` as `main`'s, so the merged result takes D_j = 2 there.
- **Run #9454's tests on a trial merge with `bafae2eaec2e`,** on your machine, and tell me with the pushed head: which pinned figures move, and whether Friday's two (91,778 GB a world-year, 24.48 GB a datacenter-year) are among them. Push no merge of lean's head to #9454's branch; it merges `main` when #9375 has landed.

