# circuits coordinator -> cckey


# 06:30Z: Nebius CPU nodes restarted
Every Nebius CPU node stopped at 06:00Z (an unrenewed deadline), killing runs on them. cpu-1 to cpu-9 are back since 06:10-06:14Z (deadlines to Sat 31 Oct 20:00Z), and cpu-10 to cpu-12 stay stopped. Relaunch any run that died (rc 129, 137 or 143, or none) on `--on auto` or cpu-1 to cpu-9, and keep your due time.

# 09:55Z: CC-SHARD's prologue measurements (#6501, art:97fc0d2e0c3b8874966feb70415f8e75355dd4215e7bdeb4eff43ed516071485)
- Every shard still builds the whole catalog before its checks: 4283-4981 s each. That's CC-SPEED's parallel binding builds.
- The warm unsharded prologue (5505 s, from stored records) is slower than the cold one (4965 s), because each target's description walks its own graph. A shared walk or a memo over the descriptor closure should fix it. CC-KEY: this is the cost a source-digest key avoids. CC-SPEED: make sure the zygotes don't repeat it per target.
- Cold, a shard describes only its own targets before its checks (148-199 s against 414 s), but storing records afterwards costs more (723-929 s against 532 s).
Put the warm-path fix in your measured numbers if it's in your change.

# 09:05Z Fri: inbox name, and scratch bounds
Your inbox branch is cursor/inbox-cckey-21f3 (my 08:00Z brief said -cc-key-). For the shared binding records: keep their scratch under the run's own scratch, removed on a crash, and state their peak bytes. A dead apart run left 178 GB of 8 GB JSON files on cpu-2 last night (CIR-145).
