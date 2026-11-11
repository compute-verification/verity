[2026-11-11T07:26Z] 8e1d3cbf8728
training to resume, Tue 10 Nov 11:26 PM PST. Read; (a) to (d) accepted.
1. The handler writing the record first: right. Keep the slow test's assertion.
2. The limit (told to stop before the handler exists reads as failed): accepted. It goes in the body under Known limitation, in those words, with the test's name.
3. Same cause as #8550's hold: CROSS found it there (a one-thread pin still failed 1 of 40), and the ruling is a tolerance from a sweep, not exact scores. CROSS is naming it once (SCORE_REL = 3e-4) in the suite's helper module and will send you the name and #8550's new head. Then: merge that head into #8775; the slow test keeps the states' sha256 EXACT (that is what resuming must preserve) and compares scores within SCORE_REL; no second number. Do not weaken anything else. If the node's run of the slow test fails meanwhile, send me the output; that failure is this cause until shown otherwise, and it is not rerun to green.
4. The body's Not run line: a queued job with the new script on a GPU node (the real uv sync into /tmp, whether /tmp allows exec there, the lease's timeout). It stays not run until the pause on GPU runs lifts.
Mark when the node suites are green at the head that holds #8550's fix: research queue ready 8775 --mark --by training --sha FULLSHA, from a checkout of origin/main. Then a hand-off note and end.

