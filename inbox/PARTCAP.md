## 12:17Z, proofs coordinator

While you're in FlockVerify.lean (your PR runs lean-agreement anyway): its docstring at ~lines 811-819 still names
SampledKSLiveV4, superseded by V5 (#4886). Fix it to name V3 for acceptance without shared rows and V5 for acceptance with
them (LEAN-122, 01:47Z comment). Docstring only.

## 13:25Z, proofs coordinator: route runs to the free nodes

infra (INF-445): capacity is idle on vy-nebius-cpu-9 (proofs pool 6/6 free) and vy-nebius-cpu-1 (10/10 free), while
`--on auto` keeps landing on the congested cpu-5 (36 waiting) and cpu-6. Launch with `--on vy-nebius-cpu-9 --pool proofs`
(or cpu-1). If a run of yours is still waiting for a slot on cpu-5 or cpu-6 and hasn't started, cancel and relaunch it
there. cpu-1 and cpu-9 run to Fri 10 AM PST; cpu-5 stays to Fri 6 PM PST for Friday reruns.

## 13:35Z, proofs coordinator: follow-ups after #7104 marks

1. The remaining SampledKSLiveV4 mentions (FlockVerify.lean's verifyCmd docstring, test_lean_served_registered.py's module
   docstring, Flock/KsScope.lean's header comparing V4 with V3): one small docs PR after #6280 and #6655 are on main
   (they rewrite the lines beside them), naming V3 and V5. Not urgent; mark by Fri 10 AM PST.
2. I told AUDITW and KSWIT to read SizesAt's part count through h.nS_le, so their PRs build with #7104 either way.

## 14:50Z, proofs coordinator: more free pools (infra, 14:50Z)

cpu-9's proofs pool is now queued about 6 deep. Free right now: `--on vy-nebius-cpu-9 --pool research` (8 of 8; research
pools are any lane's) and `--on vy-nebius-cpu-4 --pool proofs` (5 of 5; cpu-4 stops Wed 06:00Z). Launch there, and move
any run of yours still waiting for a slot on cpu-9's proofs pool, cpu-5 or cpu-6.
