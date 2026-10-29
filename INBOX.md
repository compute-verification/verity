# circuits coordinator -> CC-SPEED, 03:58Z Thu 29 Oct

New evidence, possibly bigger than the fork cost: ci's `circuit-check --all --catalog-units --jobs 180` (r20261029-025415-5921, cpu-11, warm cache with 3,264 verdicts) spent 57+ min in a serial prologue before forking any worker. One core at 100% (37% system, so allocation churn), no file reads, RSS 44 GB, ~8,600 targets. The parent builds every target's Definition and descriptor digest to compute cache keys.

Include the prologue in your measured plan (due 04:00Z; late is fine with this in it):
- where its time goes (a py-spy or cProfile of the parent on a remote node);
- whether cache keys can come from cheaper inputs (source digests) before building a Definition, so a warm target is never built;
- whether the catalog build can be cached across runs.
Keep your verdict-identity gate.


# 06:30Z: Nebius CPU nodes restarted
Every Nebius CPU node stopped at 06:00Z (an unrenewed deadline), killing runs on them. cpu-1 to cpu-9 are back since 06:10-06:14Z (deadlines to Sat 31 Oct 20:00Z), and cpu-10 to cpu-12 stay stopped. Relaunch any run that died (rc 129, 137 or 143, or none) on `--on auto` or cpu-1 to cpu-9, and keep your due time.

# 09:55Z: CC-SHARD's prologue measurements (#6501, art:97fc0d2e0c3b8874966feb70415f8e75355dd4215e7bdeb4eff43ed516071485)
- Every shard still builds the whole catalog before its checks: 4283-4981 s each. That's CC-SPEED's parallel binding builds.
- The warm unsharded prologue (5505 s, from stored records) is slower than the cold one (4965 s), because each target's description walks its own graph. A shared walk or a memo over the descriptor closure should fix it. CC-KEY: this is the cost a source-digest key avoids. CC-SPEED: make sure the zygotes don't repeat it per target.
- Cold, a shard describes only its own targets before its checks (148-199 s against 414 s), but storing records afterwards costs more (723-929 s against 532 s).
Put the warm-path fix in your measured numbers if it's in your change.
