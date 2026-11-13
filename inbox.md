[2026-11-13T01:49Z] 2cf9b2ece910
TOPCODE, Thu 5:48 PM PST

**The real entry breaks the whole-number guard: two of its backward excesses aren't a whole multiple of its 111 excess normalisations. I haven't bent the guard. Without it, the norm rule covers all 10 remaining lines and the entry is equal; with it, those 10 lines stay as differences.**

RAN r20261113-014310-d364: of_record on `m-t20-2x-wildcard-topcode-coded` at head abedd475dc, counted step 3. One fix came first (abedd475dc): the code rounds each token as one group, `[tokens, 1, d]`, the view the record's `quantize` reshapes to (READ, the record's `train.py`, `quantize`: `y.reshape(-1, shape[-1] // group, group)`). That cleared 11 layout differences. Of the 17 left, 7 are R's construction, which I now name in `record_only` (dea3ccc1a1). READ, the record's `train.py`, `step_of`: `FRESH.clear()` makes R again in every step, so its counter sees it in the step. The other 10 are the norm.

The norm rule's inputs on this entry, config against record (RAN, `of_record`'s own functions on that run's counted record and the record, on my VM):

| key | config | record | excess |
|---|---|---|---|
| fwd/act mul [x, s] (the normalisation, the rule's `scale`) | 114 | 3 | **111** |
| fwd/updated mul [x, g] against the record's `_fused_rms_norm` | 111 | 111 (fused) | 0 |
| bwd/grad div [x] | 111 | 3 | 108 |
| bwd/grad mul [s, s] | 111 | 3 | 108 |
| bwd/grad mul [s] | 111 | 3 | 108 |
| bwd/grad mul [x, s] | 111 | 3 | 108 |
| bwd/grad mul [x] | 111 | 3 | 108 |
| bwd/grad mul [x, x] | 333 | 6 | 327 |
| bwd/grad mul [x, g] (not in the record) | 108 | 0 | 108 |

x is [1, 1024, 768], s [1, 1024, 1] and g [768]. The 3 plain normalisations on each side are the code's `rms_unit` at the cut, one per forward.

The guard takes `(cfg[k][0] − n) % excess[scale][0]` over the held backward keys: 108 % 111 = 108 and 327 % 111 = 105, so it refuses. INFERRED, why: there are 111 = 3 × 37 normalisations a step (18 blocks × 2 + the final norm, in 3 forwards). Block 0's attention norm reads the embedding, which is frozen, so its input gradient is never computed: 3 fewer of each backward product, so 108 = 111 − 3 and 327 = 3 × 111 − 6.

Without the guard (RAN, the same code with those two lines removed, on the same records), the rule fires: "111 multiplies by an updated gain, the same on both sides, and on the config's alone 0 by θ's gains, 111 normalisations and 975 products of the backward's … beside 3 plain normalisations on both sides". With the 7 named lines, nothing differs.

Your call. Either the guard goes and the docstring states its limit (my reading of your fallback), or a guard that admits a frozen input's missing backward. I see no simple whole-number form of the second: the fused norm's backward has 3 [x, x] products a normalisation and the plain one 2, so no single ratio fits both. Until you answer, the entry stands as "differs, 10" under the guard.

Counted step (condition 9): the counted-4 half of the same run is still finishing. RAN, the same run: the record's modal structure is held by 5,385 of its 5,389 steps, and its peak RSS was 4.8 GB, so the every-entry job asks 7 GB a process. Next I launch the every-entry comparison at GROW's head b16502a23d and at mine. Its outcome for my entry depends on your call on the guard.

