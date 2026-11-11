[2026-11-11T20:19Z] d505d949fa15
REPLOT to training, TRN-111 (a), Wed 11 Nov 12:20 PM PST (20:20Z). WHAT STOPS THE THREE COUNTS FOR NOW: the builds wait for a slot. No count yet.

1. RAN by me: r20261111-195654-9c12 (the two builds at te_core 7de182a661 with size-scaling) stopped itself at 20:12:56Z. Its processes held 31 GB, its declared MEM_GB, at the builds' stage (four te_core.algorithm_policy processes at once), before size-scaling's step. Nothing went over the declaration. The `both` run's measured peak at that stage was 25 GB, and 32 was declared from it; the sampler sums resident sizes, so forked workers count shared pages each.

2. RELAUNCHED at 20:15Z, declared 64 GB (guard at 63), same inputs:
   - r20261111-201517-d96c: the builds at 7de182a661 (commit 26d6c24708's code), for the three counts.
   - r20261111-201517-c982: your ruling 4's build: the same job with the put branch's code (cursor/training-put-55a1 at e6c0507f2a, te_core tree 11f297ba3e, cut from 225e85e26e).
   - r20261111-201712-071f: standard-workloads built on the put branch (13 GB, from the earlier run's measured 10.4 GB).
   READ by me at 20:18Z: all 12 slots of the node's pool are held (checks of auto, proofs, kernel, training) and 12 runs are ahead of mine in line. From a slot, d96c needs about 35 minutes. If it has no slot by 1:45 PM PST I tell you, since 2:30 PM PST is then at risk.

3. ONE INPUT of ruling 4's build differs from the staged build's, and must: the price check's copy of main's gate table. RAN by me on the VM (python -m te_core.gates_main DIR, in the put branch's checkout): against the stored copy (4dafee69, the TF32 price's branch before it landed) the put tree's table is not the same statement for statement (DEFINITIONS and recompute name the moved modules); against origin/main's copy now (61f13c2275, training_slowdown 4.13.5) it is: same true, rc 0. So c982's replot-inputs.tar holds main's copy at 61f13c2275. Every other input has the staged build's sha256.

4. The put's commands exist, on the put branch and on #9010's branch alike (the scorer and the two builders are byte-equal on both; nothing under te_core changed). RAN by me on the VM from the put branch's checkout, on stored bodies, each printing sha256, bytes, rows, note length and "console.put would refuse: nothing":
   python scorer/plot2_final.py put DIR --note FILE --dry
   python scorer/plot2_final.py put-fit BODY --dry
   python islands/standard-workloads/standard_workloads.py --put BODY --served FILE --dry
   python islands/ladder-v3/size_scaling.py put FILE --dry
   Your draft note in FILE, with {art} where the id goes: 482 characters with a 68-character id (the dry form puts 68 zeros' worth of id there). The staged paths come with (b).

5. A CORRECTION of my 11:55 AM message: of the modules size_scaling.py imports, te_core.gates is not the only one that differs between 4.13.3 and 4.13.6: console.py differs too, in its BRANCH line alone.

6. TWO THINGS FOR YOU TO RULE WHEN YOU READ (b), neither stops the work:
   - size-scaling's source names the art of the tables it read. Staged, it names art:7b1ab651... (the tables you read); I check that each build's own search tables equal that art's tables/corr/search in every cell before I call the citation true.
   - standard-workloads' source reads "(te_core 4.13.6, on cursor/covert-witness-4207 at 26d6c24708)". READ by me: that branch is gone from origin, and 26d6c24708 is the --base I gave (the published builder's commit), as in the build you read. The served panel's reads "at abb0941fcf".

