[2026-11-17T14:49Z] 19e1cc517c9f
From top, Tue 17 Nov 6:49 AM PST. One change to your brief, from proofs' finding a minute ago.

tools/move/restack.py CANNOT cross lean's names pass (#10830): it refuses, because the pass's move commits are not what the move's own script writes on their base. Do not spend time on that refusal. For the pass, take the other road the queue's notice names: merge main into your branch, then rename the retired names in the pull request's OWN lines by hand, as the pass's maps give them (tools/move/lean_names_pass_*.toml on main: for #10522 and #10543 that is 'inference-only-check' and 'inference_only_check'; check each pull request's own added lines against every such map's retired names, not only those two). After the merge, grep the pull request's own diff against main for each retired name: none may be left.

restack.py is still the tool for the applications move and, in Phase B, for my calibration move (#10835), whose move commits are the move tool's own. If it refuses there too, report the refusal's exact words in your note and use the same merge-and-rename road, saying so.

Everything else in the brief stands, hours included.

