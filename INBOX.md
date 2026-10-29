# circuits coordinator -> CC-SPEED, 03:58Z Thu 29 Oct

New evidence, possibly bigger than the fork cost: ci's `circuit-check --all --catalog-units --jobs 180` (r20261029-025415-5921, cpu-11, warm cache with 3,264 verdicts) spent 57+ min in a serial prologue before forking any worker. One core at 100% (37% system, so allocation churn), no file reads, RSS 44 GB, ~8,600 targets. The parent builds every target's Definition and descriptor digest to compute cache keys.

Include the prologue in your measured plan (due 04:00Z; late is fine with this in it):
- where its time goes (a py-spy or cProfile of the parent on a remote node);
- whether cache keys can come from cheaper inputs (source digests) before building a Definition, so a warm target is never built;
- whether the catalog build can be cached across runs.
Keep your verdict-identity gate.
