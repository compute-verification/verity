[2026-11-20T12:52Z] 38074d54f786
Your confirming run r20261120-121429-540c ended rc 0 (33 min, cpu-4). Read it and report: the suites' counts and whether any red is left.

[2026-11-20T13:10Z] 3430f40eb90d
URGENT from the coordinator (ci, TOP-447, 5:07 AM PST): preflight-fixtures is RED on the tip because two fixtures registered in fixtures/artifacts.json by 7087b817736 (the Flock suite's re-record, 3:24 AM PST) are in neither the machine's store nor the public export: art:7fa434603c3b932b8c021cbc11fad9c5502e3f95f2ac19707ab1b44c780b952f and art:bd47bdb13a0b1566dc57da0d3a84edc8bc9f7f40fe74fe2a. If you recorded them: put both in the evidence store NOW from where the bytes are (research data put --kind ... --file|--tree P --preserve, so the art: id matches what artifacts.json names; check with research data get on another machine or a run), then report to me the two ids, where the bytes were, and the command; if the bytes are gone, regenerate and fix the ids in artifacts.json on a branch. Do this before anything else; ci recounts at 6:30 AM PST.

