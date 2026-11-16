[2026-11-16T08:16Z] 38f7e05590cc
Coordinator 12:20 AM PST, infra rule on TOP-402 (12:15 AM): node A has 65 GB free; NO new fresh-clone run starts under 60 GB free. Put `df -h /` in your launch step: if under 60 GB, do not start; wait for a run to end. If yours is already running, make sure its job removes the clone and WORK at exit (also on failure), and report its disk peak. The same check goes in the 4:45 AM run.

