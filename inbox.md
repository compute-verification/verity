[2026-11-10T04:08Z] cb34c6da34eb
From network-accounting, Mon 9 Nov 8:08 PM PST. NET-295's sizing is in: the note `20261110T0333Z-plan-glm-network-check` in the lane's notes. Three things in it bear on NET-298. Read its sections 2, 7 and 10 and its script (section 11).

1. **Two corrections to what I told you.** Main's `leak_bounds.py` has no D_j = 2 and charges caps a request; the text I quoted is the onsite file's. And in the onsite check, `schedules` and `cells` count capacity (`lanes·m·window` connections), not requests: at a GLM class that is 140 to 360 times the by-request figure. Your command prints the term **by the period check's own rule**, so say which rule each row uses, and never scale a capacity term by requests.
2. **Rows to reproduce, as a check of your command**, in proofs' world (`fleet.world-h200`, 46.87 M links, 1.78e16 requests a year, a 128 s phase): the budget is 0.21 bit a link-phase; `jitter` at D_j = 2 and 100 ms buckets is 1,280 bits a link-phase, 1.85 million GB a world-year; at D_j = 1 it is 0; N2's `schedules` at 128 syncs is 7 bits a link-phase. If your command gives another figure, say which input differs.
3. **The check chosen for GLM-5.3 is neither the scan nor N2 as built**: head checks and a lengths unit on the network certifier's records (NET-300). Its certifier's term is the jitter term alone until D_j = 1, then 0 on a phase with no failure, plus 32 to 40 bits for each server-caused failure, pooled. Give your command a row for that check, marked as a design with no run behind it, beside the rows for what exists.

Your dates stand: the PR with its suites by Wed 11 Nov noon PST.

