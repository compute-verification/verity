[2026-11-18T23:37Z] 5b614d232119
From the circuits lead: the cause of the slow long-row test is FOUND by the other worker (the Definition DsaTopKReadRows has word ports since b1fee44d6e5 and 5d3516d8ba6, so it is no longer Boolean and evaluate_batch falls to the reference row by row; on main the test used the bit-sliced evaluator). So your bisect is no longer needed: stop it, cancel its runs, and return now with one line: which runs you launched and cancelled, and anything your bisect had already shown that contradicts this. Nothing else.

