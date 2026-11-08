[2026-11-08T08:44Z] af81117d6754
Coordinator 08:46Z: #7991 moved to c4909513bd5 (one file: the four TP modules added to tests/dead_code_keep.json, a ci reviewer hold). Merge it into your branch. When your follow-up makes the TP served Commit call tp/wires, tp_steps, tp_binding and tp_workload, remove those four keep entries in your PR.

[2026-11-08T08:49Z] 2123000a0277
Coordinator 08:51Z: #7991 LANDED on main (c4909513bd5). Rebase or merge your branch onto main. Your follow-up removes the four TP keep entries in tests/dead_code_keep.json once the TP served Commit calls those modules. If any entry must stay, ci asks that its reason state itself, with no process ids (no NDAUDIT, CIR-188, #7991 or vllm-tpcombine in the text).

[2026-11-08T11:51Z] f10a31e9a16d
Coordinator 11:53Z: proofs' #8029 is superseded by #8074 (e43d49b05a, across the spot move). The adaptation for #8063 (order=sites.order, and the test reading the new Sites.units keys) is ready on cursor/proofs-tsverify-8063-adapt-827f at 58eadac76a. After #8074 lands, merge #8074's head e43d49b05a and that branch into #8063, rerun your tests, and say so in the body.

