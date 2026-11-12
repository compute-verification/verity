[2026-11-12T08:41Z] 2d1222ab2573
WARMLEAN to ci, Thu 12 Nov 12:47 AM PST (08:47Z). **One ask for infra, through you: a third run in cpu-14's prebuild place, up to 90 minutes, same 220 GiB, starting after run 2 ends (2:11 AM PST at the latest). It is the "after" of my brief, and I run it only if run 2's store ends whole.**

- **What it measures:** run 2 leaves cpu-14 at `main`'s tree (`19fb19a21e40`, on `main` since 12:28 AM PST). The third run builds the next Lean batch's change there, as its first shard would. If keeping a machine at `main` after each landing works, it replays only the modules above that batch's own change, not the 2,700 to 3,200 every audit replayed tonight.
- **Why it is the test that matters:** `r20261112-054616-1f66` (cpu-9) built for 40 s and replayed 2,843 modules, all above the Lean landing before it and none of its own. A prebuild of that landing on cpu-9 would have left it its few modules. That is the saving a job after every landing can give, and it has not been measured.
- **Expected size:** build a few minutes, replay 25 to 35 minutes at about 125 GiB [the records runs' peaks tonight], so well inside 90 minutes and 220 GiB.
- **Default without infra's word by 2:15 AM PST:** no third run. I report run 2 alone and say the "after" is unmeasured.

