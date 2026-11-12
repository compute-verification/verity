[2026-11-12T18:25Z] c903ec3278bd
From circuits' lead, Thu 12 Nov 10:26 AM PST. Card 1 of node A, where your hours are now (to 21:00Z), is hot: memory-accounting measured its clock held at 42% of full or lower about ten seconds into full work (INF-757). Their results on it were right, only slow. Nothing in your capture changes. Three things:
- Quote no time measured on card 1, in the pull request or on CIR-262.
- If a step runs several times slower than you expect or times out, that is the likely cause: lengthen the wait before you debug it.
- Keep `nvidia-smi -q -i 1` from before and after the capture in the run's outputs (its thermal slowdown counters), and say in your return what they read.
Do not move to another card: the calendar's other rows are other lanes'.

