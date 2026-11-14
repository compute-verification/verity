[2026-11-14T08:59Z] 6dc21b09b333
ROUTES, Sat 14 Nov 1:01 AM PST.

**#9964's new head is 5f3af34937fc926ebc4b232b7547236edc118f9d (delta bebb06fe..5f3af349, two commits, one push). Both must-fixes are done as ruled. The quick tier is r20261114-085920-6af0.**

- Must-fix 1 (RAN): every route's `DIALS` and `FIXED` key is in `synthetic.KIND_DIALS` (`kind_of` reads its branch) or `OTHER_DIALS` (rates, schedules, shares, sizes, `store`), and the two sets are disjoint. The mapping test names each route's unclassified dial. In a scratch run, the reader's `grad_noise` added to subset's `DIALS` failed naming it under subset and ema; the scratch is reverted.
- Must-fix 2 (READ): `kind_of` keeps the group count. Its docstring says: "An optimizer's number of parameter groups is: one outside torch's fused kernel keeps a group a rate, so a config that adds a rate is a new kind until traced." The body will say the same.
- RAN at the head: the agreement, either-kernel and mapping tests, 65/65; ruff clean.
- Slow: 8 cases, as before. fullmuon and fullmuon-fe stay unmarked unless the quick run times them over 5 s.
- The run r20261114-081222-d0cf at bebb06fe is superseded and won't be used.
- Events: a broadcast about naming models (no action; I start no workers).

