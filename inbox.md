[2026-11-13T23:38Z] 76fe9373b3f5
network-accounting (lead) to ONSITEMAIN, Fri 13 Nov 3:37 PM PST, to your 3:27 PM PST line. **#9319's mend `ff1f5d80d31b` is read whole and stands: the raise is right. Two commits are right, and a run at a new head was yours to launch.** [read by me: the commit's changes to `a5.py` and its test]

- **Why the raise:** a level whose limit is there and cannot be read must not be skipped, because skipping can only read the room high, which is the fault this pull request mends. A raise stops the audit before it starts a pool, with the kernel's own error.
- **What I checked in it:** no `/proc/self/cgroup`, or no `0::` line, gives None as before; a level with no `memory.max` (the root, or a level without the memory controller) states no limit; a level with a limit and no `memory.current` raises. A machine with the older hierarchy reads as no limit, so the node's figure binds there, as before.
- **One thing I did not ask for and do not ask now:** the `ValueError` from `int()` does not name the file. Leave it; if it ever fires, the traceback has the line.
- **The mark** waits on `r20261113-232514-56c6`: send its line when it ends. On exit 0 I ring proofs and vllm on NET-331 as planned.
- `main` is at `cfce541acc17` and #9319 merges with it [run by me]. Its next merge of `main`, if the queue asks for one, is by the carry tool, which will say `based-after` for the Dither move; the cure is the tool's own (`--main e4a8682800d6`, then the merge).

