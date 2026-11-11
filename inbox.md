[2026-11-11T03:41Z] df91cad4348c
From the infra lead, 03:43Z, after my 03:39Z message. The queue's own states, read by me now: #8570 and #8560 are marked ready: ci crosses them at about 05:00Z; push nothing to them, and once each branch's head has moved run its suites on the crossed head. #8603, #8602, #8598 are closed and not in the queue: yours to reopen and cross by hand with top's restack command (#8603 before #8602).

