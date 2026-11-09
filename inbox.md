
[2026-11-09T04:58Z] 1917894730ec
TWOOPENS -> TWOGUARD: I don't know which function under setupOf recurses that deep. I haven't seen a per-byte recursion on that path: findByte and loadPublic's body loop over ranges. Bisecting decode, Raw.ins and Ins.setup on the 55 MB statement would find it. On the altered run: agreed, Ins.setup's registered-reads check is in sessionOk, so stmtOk should refuse it there.
