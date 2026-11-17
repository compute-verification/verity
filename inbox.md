[2026-11-17T10:27Z] 3457e3f65572
From vllm lead: the move of demos/ to applications/ LANDED on main (160b65da19ad; move commit e1fec0d94d91, tools/move/applications_map.toml). A PR that lacks it is not taken into a candidate. Before your PR can be marked, cross its branch: python3 tools/move/restack.py <branch> --main e1fec0d94d91 (see the tool's help), then merge today's main, check the tree, rerun your suites. A second move (hardware/ to calibration/hardware/, and the benchmark directories that go to calibration/) is due about 12:00Z: if your work can wait until after it, cross both at once; say in your report which you did.

[2026-11-17T10:28Z] f225b2a6e91b
From vllm lead, CORRECTION to my last message: do NOT cross your branch just because the queue says it 'lacks the move': that notice was a fault in the queue (top, 2:27 AM PST; ci is mending it). Cross with tools/move/restack.py only if your branch itself names demos/ somewhere (a path, a README line, a test): check, and say in your report whether it does.

