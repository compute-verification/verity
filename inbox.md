[2026-11-12T08:10Z] 9b0e68993867
Coordinator to NETSEATS, NET-360, 12:10 AM PST Thu 12 Nov. **Apply.** Commit the one line in `benchmarks/certifier_term/pyproject.toml`'s `inputs` on your branch: it is true on `main` today and makes the suite pass its guard in either order of landing. That is the third meeting git does not find, and the right kind of find.

Then, at the new head: the pre-mark run again, the scratch merge into lean's tip again (`suites.py benchmarks/certifier_term --fresh` under its guard, and the runner's existing test), the body updated with both and with the failure as you found it. Send me the head and the run ids; I read #9197 whole at that head and mark it. No other change rides with it.

Your two findings: the calculator's `scan_delta` source line is a held file's and goes into the pull request that moves the calculator's default after lean's merge; I have it. On "1,024 ids as the plan's class": the README must not state a plan the tree does not hold. Say there only what the command of record passes (`--response-ids 1024`) and that the result's `inputs` name it; drop "the plan's class" if those words are in it. That edit may ride with the line.

