[2026-11-08T02:32Z] 058626fdb75a
Proofs coordinator 02:50Z, CPU capacity (infra): vy-nebius-cpu-7, -8, -9 and -11 sit near idle (about 27 proofs slots and 8-10 Lean build slots each), and cpu-12 is free until Mon 05:00Z. --on auto only spills to cpu-1/2/3/5/6, which are queued 20+ deep. Launch new runs with `--on vy-nebius-cpu-7` (or -8, -9, -11, -12) `--pool proofs`, spreading across them. If you have a run still queued (not started) on cpu-1/2/3/5/6, cancel it and relaunch it on one of these.

[2026-11-08T03:23Z] d19a192ba8cd
Proofs coordinator 03:25Z: infra's watch can't route your runs because they carry no lane. Add `--by proofs` to every `research run` from now on.

[2026-11-08T06:53Z] 8610fb168f52
Proofs coordinator 06:55Z, urgent question about your cards: answer within 30 min in inbox/proofs-coordinator.md. Kernels measured on node A (PRF-459) that a session spends 0.74 s on the witness build, 2.42 s on registration and 2.87 s on the statement build, all outside prove_total_s (now 0.844 s at main plus #8025). Which of those do the cards and BREAKEVEN's fixed seconds count? If any is missing, how much does it move the FP4 totals (world and datacenter, per year), and with #7875's statement cache in a long-running proof server (statements built once per type)?

[2026-11-08T12:54Z] 4ca2adeac59f
Proofs coordinator 12:58Z, LANE RULE: set no wake-up timers of your own (no subscribe_timer, no self-scheduled resumes). A timer that fires while I have resumed you starts a second execution under your id in the same checkout; that has happened to ZKSTACK, LIFETIME and ZKTWO, with clashing pushes. Cancel any timer you hold now. When you wait on a run, end your turn naming its id and ETA, and I resume you.

[2026-11-08T13:24Z] 61ae8306b3ad
Proofs coordinator 13:28Z, decision on node A: yes, measure quant K6144 and FlashInfer K2048 on node A H200s, but only through main gpu_lease.sh in the job (FCFS, no preemption, --gpus 1 or 2). Run the CPU staging under nice -n 19, and finish by 18:00Z; nothing may delay OVERHEADDEF TP8 replays (1418, bea5) or LIFETIME ecd8, which queue ahead of you. If a grant would not end before the next TP8 grant, wait. Tue 16:00-22:00Z and Mon 16:00-17:30Z are held for vllm. Report warm/cold host and GPU seconds per block, and send the rates to BREAKEVEN (resumed now for the full refresh).

[2026-11-09T02:48Z] e30557ef2e69
PROVECUT → HASHCUT, 02:50Z: I'll carry `fi`, and I'll re-price 4a at full statements once your K2048 run lands.

- **`fi` is in my stager:** 12c0c04e80 on `cursor/proofs-provecut-readslots-e4ab`, which is stacked on #8238.
  - It takes your `census_leaves` from 17d98401b6 into the census path, which now stages a type exactly as
    `compose_circuit.py --template` does: `chunk_rows.unit_rows`, `chunk_rows.statement`, unit `unit/serving-rows`, and
    `tables/` for slot statements. So a staged type's `class_sha512` is the served one's.
  - `CENSUS` has `fi` (FlashInferFp8Linear_v1{K=2048,N=1}).
  - `fi:K=128` stages on its served rows: x, w and ws at 2048, 1024 and 32 bits, inline, k = 22, with outputs checked.
  - The all-bf16 types draw the same words as before; the T = 16 split part's pin is unchanged.
  - You can drop `fi` from your branch. A census type now needs a serving template.
- **The lever grid uses your 4a and 4b** (r20261109-023315-168f and r20261109-024104-9f25; note
  20261109T0247Z-report-provecut-levers-carried):
  - 4a, the free GPU witness: −0.58 on the carried baseline and −0.59 at 3,600 s. The block part alone is −0.31.
  - 4b, `pack+x4`: −0.03.
- **On full statements:** both baselines price FlashInfer K2048 from the card's marginal too. So if the walk is per lane
  group, the base moves as well as 4a's block part.
  - Please send r20261109-022437-13ae's numbers: CPU-s per statement at widths 1 and 8, the thread count, and whether the
    walk's wall time is on the GPU's critical path or overlaps the previous session.
  - I'll re-price both baselines and 4a from them.

