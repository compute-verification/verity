[2026-11-13T03:36Z] c3d897039a3f
WALKER, Thu 12 Nov, about 7:40 PM PST (by the run clock). TRN-145.
**s-t20 at 2^-3 has read past the depth that killed 7c4e and 84cf, but its memory is climbing about 25 GiB every 10 minutes and has passed the 160 GB I asked for.**

- READ, r20261113-005600-8f5f heartbeats: RSS held at 57.6 GiB from the build until 6,615 s. Then it rose to 90.1 GiB at 7,215 s, 112.3 at 7,815, 138.5 at 8,415 and 159.0 at 9,015, all still in the reading, with the pass not started. Its session scope shows 179.1 GiB, growing 0.7 GiB/min, against the scope's cap of 604.7 GiB.
- INFERRED: the climb started where the earlier runs died, so this is the deep part of `value`'s walk. Each of its more than 1,000 nested frames holds `words(nd.args[i])`, a list of every word of an argument, and the cache `words_of` holds them too. If the climb keeps its rate, it reaches the scope's cap in about 10 h, which is before the 14 h lease ends. Other runs share the node and the cap, so it could be killed sooner.
- A 160 GB reservation sets no cap of its own, so nothing kills it at 160. I am not touching the run without your word. The choices I see:
  - (a) Let it run and watch it.
  - (b) Cancel it and relaunch at 102ad1d702 plus #9578's reader, either on a branch for the measurement or after #9578 lands. The new form keeps no frame per node, but `words_of` still grows.
  - (c) Relaunch with a larger memory ask. That changes nothing about the climb.

#9578, done as you asked, in words only: the test is now `test_the_suites_programs_keep_their_readings`, its docstring and the comment above `CASES` state what each holds, and the body carries the history (recorded with the recursive reader at 4e7a25c6a766, rewritten by you to the same bytes). Its head is 3e4bdf6f12; on my VM, 100 passed and ruff is clean. The node run of `suites.py benchmarks/circuit_properties` at that head is r20261113-033438-1e24, in flight. I send its last line when it ends.

