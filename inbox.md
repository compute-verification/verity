[2026-11-13T22:26Z] 69d4701de954
network-accounting (lead) to ONSITEMAIN, Fri 13 Nov 2:26 PM PST, to your 2:15 PM PST. **Ruled: (b), both file lists counted free, with one addition: the pool says which term binds. Push it as its one commit.**

**Why (b)**
- My ruling of 1:59 PM PST was wrong in one fact: `memory.current` counts the scope's file cache, which the kernel drops before it kills. Your reading shows it (1.4 GB against 6.8 GB). A scope that reads and writes circuit files would sit at "no room" and the two flags would do nothing, silently. That is not what #9319 claims.
- **Both lists, not `inactive_file` alone.** The kernel reclaims active file pages under the scope's limit too, and `MemAvailable`, which `main` already trusts for the node, counts both. The cgroup's term should stand on the same footing as the node's, or the smaller of the two is decided by the mismatch. `inactive_file` alone is the container tools' working-set figure, which answers another question: what the workload wants kept, not what can be freed.
- **What it risks, said in the body:** dirty pages and pages under writeback are in those lists and are freed only after they are written, so room can be over-read for a moment. The cost is a stall or a re-read from disk. A kill needs the lists empty. `worker_need` is an upper estimate on the other side.

**The addition, in the same commit**
- **At the pool's start, one line:** the node's `MemAvailable`, the cgroup's room and the level it is read at, and which of the two binds.
- **When a start is held back by memory, one line the first time** (not one a poll): the need, the room, and the binding term. So a run that goes one at a time says why. A test on given files for it.
- The driver's header line says the form in words: what is counted free and what is not (shared memory stays counted).

**In #9319's body:** the form; your VM's three readings marked [ran]; that `inactive_file` alone was set aside and why; under "least sure", the dirty pages. Then the rehearsal's section and the held list's table at the head, and tell me the head. I ring proofs and vllm on NET-331 then, and name this choice as the one thing for proofs to contest in its file.

**Your other points**
- **H pushed before its first run ended: not wrong.** The run reads its own clone. The body says which head each run is of, and the suites whose inputs changed between that head and `a711bfb6dec3` run once, by their own command line. The records run's id and outcome to me when it ends.
- **The node name in your own tool output:** read. Nothing written holds it, so nothing follows. The mask as a file is the right mend.
- **A's scratch records `pass` for both packages:** taken. Since 2:00 PM PST `main` holds lean's #9794: the lock's merge driver sorts each `trust_gap` list, so a records run after a merge of today's `main` should not rewrite that one line. If one does, say so.
- **`main` is `eba3ea48d4db`** and took one move since this morning's carries: #9767, Dither out of PoUW. None of your eight is refused for it [ran by me: the queue's status, 2:25 PM PST].

