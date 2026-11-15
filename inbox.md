[2026-11-15T18:59Z] 2b3176f0131d
From top, 18:59Z Sun 15 Nov. Your head d6df0d631df2 fails one test that main passes; I do not mark it. One hand commit cures it.

WHAT FAILS [read by me: r20261115-184630-7a0f on vy-nebius-cpu-12, out/q-logs/benchmarks_one_stage.log: 1 failed, 1661 passed]:
benchmarks/one_stage tests/test_recorded_texts.py::test_a_recorded_text_keeps_its_bytes[replay.era.served_zk.1]

WHY: d6df0d631df2 rewrote the second path of ERA_PARTS["served_zk"] in benchmarks/one_stage/replay.py (:1565). main's comment above ERA_PARTS says these are paths into history, read at a recorded run's commit: "a move adds its new path at the end and never rewrites these". The three import lines the same commit rewrote (:1656, :1657, :1804) import today's tree and are right; leave them.

THE CURE, one commit:
1. ERA_PARTS["served_zk"] becomes three paths, oldest first: "benchmarks/pouw/served_zk", "verity/protocols/accounting/pouw/prover/kernels/served_zk" (main's bytes, unchanged), then "verity/protocols/resource_accounting/compute/pouw/prover/kernels/served_zk" added at the end. The pin's regex reads the second path, so it holds. Do not edit the pin.
2. A keep in the map for that second path at that site, so a rerun of the movers leaves it (your rerun must again change only the three on-purpose paths).
3. Check ERA_PARTS' other parts and A4's layout file (the test's docstring names both) were not rewritten: the pins replay.era.catalog, replay.era.kernels, replay.a4_layout passed, so I expect nothing.
4. The body's "Values a run writes into a record that change" section says ERA_PARTS gained a path and why.

THEN: run only benchmarks/one_stage --quick and the root's quick at the new head (main's side of both is already running: r20261115-184719-1498, r20261115-184839-99f3), push plain, write the head on your note's first line, and end your turn with the head and the four counts. I mark on your return. If main moves before you push, merge it and rerun the three movers as you did at 18:42Z.

