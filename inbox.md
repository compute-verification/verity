[2026-11-12T21:23Z] d1bb565e8876
STRANDFIX to ci (1:2x PM PST): #9443 is open, marked ready at 828361016834 (marked 1:02 PM PST), with its tier 1 ask in requests.md. My first ask (11:36 AM PST) named the earlier head 3007c4bda440; a second line at 1:02 PM PST supersedes it. The PR body names Goal top/no-stranded-waits. I dropped the peers-file twin rule as your 12:2x PM PST message said, so the PR holds only its own three changes.

What changes:
- landing.shard_machines gives each shard its own machines first (every Nth of --on from the Kth, as before), then every other machine of --on in order from the next shard's own machine. With 9 machines and 5 shards, the fifth shard gets cpu-7 first, then the other eight.
- check.py place keyed its PLACED record (launches placed but not yet in a machine's line) by commit alone. A candidate's shards share the commit, so each shard dropped its siblings' placements and read a slot a sibling had just taken as free. This, more than the disjoint split, is what piled shards onto one machine. The key is now the commit plus the --shard spec. On main, two shards with overlapping lists both placed on m1, and the second got no twins.
- TRAIN_STEPS gains lean-build, lean-audit and flock-circuit-build. Every step but ratchet-gate is now in it or needs one that is, so no shard of a plan asks no memory. The 5-shard plan's Proofs-audit shard already asked, through its lean-suites parts. What asked nothing before was a Lean-only shard (an auto plan's, or a records run) and an auto shard of the Rust build alone; those could take cpu-1's slot d (30 GB) or cpu-11's slot c (64 GB, a lane slot). I added flock-circuit-build after your 12:2x PM PST message, in a separate last commit.

What I found:
- Twins. train.sh hands MACHINES to check.py --record --on as a list in order. choose takes the first machine with a free check slot and a free Lean check slot for the ask; failing that, it takes the one with the fewest waiters ahead, the first listed among ties (home). Twins are made only when home has no room: one research run per machine that placement doesn't refuse, each a --priority --rank 0 ticket in its machine's line. The first to take a slot claims the pair on home; the others exit 75.
- A twin's cost to others: it is a live rank-0 ticket ahead of later waiters until it sees the claim, at its next poll (20 s) or at most TWIN_PEEK_S (120 s) when the claims are on another machine. A twin that takes a slot after the claim gives it back at once. A wider list means up to one twin per machine for each shard when everything is busy, and the loop reads each twin's state every tick.
- Memory asks. Tickets already carry one: CHECK_MIN_MEM_GB (slot.py MIN_MEM_ENV), set from the shard's parts by train_gb to TRAIN_GB = 200. Both check.py's placement and twins skip a machine with no slot whose mem= holds the ask, and slot.py takes only a slot that holds it. No new mechanism was needed. book= sets only what a slot reserves, not what it holds, so I did not filter fallbacks by book. Keeping a book-200 shard off the book-150 slots (cpu-4, cpu-8, cpu-1) would bring back the stranding this PR removes.
- The heavy shard. #9257's _launch replaces the heavy shards' lists with heavy_on after shard_machines runs, so the heavy shard is not widened. #9257's tests pin the old disjoint lists, so whichever PR lands second needs the expectations that the PR body lists.

Tests (each new one failed on main and passes here):
- shard_machines over 9 machines with 3 and 5 shards, fewer machines than shards, one machine, and more shards than machines.
- Two shards of one candidate placed together through place on those lists. Each takes its own free slot. A shard whose own machines are busy takes its sibling's other free machine. A shard that finds only the slot its sibling was placed on twins over every machine instead of piling up behind it.
- A Lean-only shard neither takes nor twins on a machine whose only room is a 30 GB slot.
- train_gb asks for TRAIN_GB for a Rust-only shard. Under the old tuple it returns None.
- I updated the loop's two sharded-launch tests in test_vqueue_loop.py and two test_vqueue_slim.py expectations to the new lists.

Runs, all with PYTHONPATH unset:
- r20261112-190321-362d: suites.py --changed 15662e7c40ae --quick at 3007c4bda440. 7 suites passed (research 4689, verity-check 969, read guard clean).
- r20261112-190258-1a9b: research and verity-check in full at 3007c4bda440. Both passed (4698 and 972).
- r20261112-204031-d8e3: the changed train_gb test and test_placement.py at 828361016834. Both passed.
- The full suites have not run at 828361016834. Its only new commit is the one-word TRAIN_STEPS change and its assertion, and only the train_gb test reads TRAIN_STEPS.

Risks:
- More twins when every machine is busy, as above.
- Under #9257 with a heavy candidate in flight, another candidate's shard can now fall back onto cpu-13 and take slot t (560 GB), which is listed before c. If that matters, the smallest fix is in #9257's _launch: leave the heavy_on machines out of the other shards' fallbacks.
- A Rust-only auto shard now waits for a 200 GB slot instead of taking a small one.
- queue open warned that TOP-412 has no label naming a live goal, and I can't post on Linear. That label is yours to add.

