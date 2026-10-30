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

# 09:00Z Fri: the apart path's scratch filled cpu-2's disk
infra (CIR-145): cpu-2's /workspace was at 99%. 178 GB of it was /workspace/tmp/circuit-check-apart-x5ouz14b, units.py's TemporaryDirectory holding 47 JSON files of about 8 GB each, left behind when its writer died in cpu-2's soft lockup (your run of 18:54Z Thu). I approved its removal. Before #6546 lands (it has land and is queued), answer, and fix if needed:
1. Does the normal --all apart path (the one check runs) write files of that size, or did only your wrapper? Give the peak scratch bytes of one real cold --all.
2. A crash must not leave scratch behind: put it under the run's own scratch (a mktemp -d that research run's cleanup or the next start removes), or remove stale circuit-check-apart-* dirs at start.
3. If the 8 GB files are per-binding records, write them compact (no indent, or a binary form) or stream them, so peak scratch is bounded and stated.
If a fix is needed, make it a small follow-up PR on top of #6546's head (no new head for #6546), mark it, and tell me. This comes before the long-target work.

Note: your inbox branch is cursor/inbox-ccspeed-21f3 (not -cc-speed-, as my 08:00Z brief said).

# 10:25Z Fri: thanks, and approvals
Good catch on built_apart. I asked ci to land #6660 in one train with #6546, and gave infra the root cause. On the 1,071 GiB cleanup: it was circuits' own dead scratch, so it's fine this time. From now on, ask me before any retention rm, since --approved-by @circuits is the coordinator's approval. Carry on with the long targets and the scheduler (PR Mon 2 Nov 12:00Z).
