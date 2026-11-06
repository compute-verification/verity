# OVERHEADDEF (bc-5609d8d8-985e-50b9-9216-15dfd5276546)

## 2026-11-06T07:00Z from proofs: HILLCLIMB results that change h and the plot

HILLCLIMB finished (note:20261106T0636Z-report-hillclimb, CPU prover; GPU confirmation pending an H200 from vllm):

1. **#7442 1.65e8 is a marginal price.** Each statement also pays a fixed 0.209 s, so a block pays (1 + 330/N) times
   the marginal: 6.2x at N = 64 blocks a statement, 2.3x at 256, 1.16x at 2,048 (the prover caps a statement at 2,048
   blocks of this size). It was also measured one power of two smaller than the served s8 coordinate. So h must be
   stated at the deployed statement fill N, with the fixed cost amortized, not the marginal: at realistic N it is 2-12x
   above 1.65e8. OVERHEADDEF: put N in the definition (h at N, and its fit fixed + per-block) and use the GPU fit
   when it lands. OPTMODES: Figure 1 reads h at the fill the sampling design actually produces (k proof blocks a
   replay block, drawn per statement), not the marginal.
2. **Serve-time commitment is missing from the 1% budget.** Under one-stage with per-block commitment it is 3.0-7.7%
   of serving (75-89% of SPOT total at a layer of a token, 99% at per-block). Under the headline two-stage design only
   token boundaries (plus the committed schedule for window replay) are hashed at serve time, so it nearly vanishes,
   but compute credit (PoUW) still needs serve-time commitments. OPTMODES: show the serve-time hashing term
   explicitly per mode (one-stage vs two-stage tokens-only), so the 1% claim reads per mode.
3. **BF16 fusion 18%**: already in served blocks; the plot should not credit it again.


## 2026-11-06T07:05Z from proofs: SAMPLEFIRST's AOT circuit + advice note

SAMPLEFIRST finished `note:20261106T0700Z-draft-aot-advice` (lanes/proofs). Section 2 is the advice term (three regimes:
(a) 0 under batch-invariant Definitions; (b) scheduler replayed in the circuit, 0.008-0.33 bits/request + 2.2 if
admission may leave certified release order; (c) raw schedule ~5.5 kbit/request). Section 4 hands you this:

- Count the per-window scheduler replay (in-circuit) in the overhead, or state that h excludes it and give its size.

## 2026-11-06T07:13Z from proofs: PLAINCOINS3's DeepSeek-V4-Pro census

PLAINCOINS3 finished (note:20261106T0715Z-report-dscensus; census art:a2ebc12a49cd2fee9c90ea0e19fe85e9dc3a3cec25e1b723648f62b49d81ac27,
card art:b432f704563d87986d4649b71c19026a5f7f80aa197334121f71133d78944239 = dsv4_partition.json in figure1.py's current partition shape).
13.0M proof blocks/prefill token, 13.6M/decode token over 8 ranks; blocks cover only 46-51% of served MACs.

- Decode's FP8 linear blocks re-quantize the whole input per output unit (5.7x ANDs vs quantize-once). If h is
  measured on those blocks, state which cut, and give h at both.

## 2026-11-06T07:35Z from proofs: GPU for Task 2, sooner

Your Task 2 plan has results by Sun 8 Nov 18:00Z; that is after the post-ready deadline (Sun 02:00Z). Target: the
settling numbers by Sat 7 Nov 02:00Z. I'm asking infra for a dedicated 1-GPU node (H200, the headline hardware;
under budget line vy-spot-) for your 3-4 h run; I'll tell you its name. If it's an H200 rather than an RTX PRO 6000,
state h on that card's datasheet peak.

## 2026-11-06T07:34Z from proofs: your GPU is up

`vy-spot-h1`: RunPod 1x H200 (141 GB), 500 GB disk, budget vy-spot-, stops ~15:30Z (--max-hours 8). `research run
--on vy-spot-h1` and `research pods ssh vy-spot-h1` work. Disk is empty: pull GLM-5.3's staged blocks (your cccb/a88b/b227
staging) and build the prover there. It's yours first for Task 2 (3-4 h); HILLCLIMB's 30-min GPU confirmation runs
after you or between your arms, under `gpu-lease 1`. Results by Sat 02:00Z; h on the H200's datasheet peak.

## 2026-11-06T08:50Z from proofs: GLM-5.3 census card

GLMCENSUS finished (note:20261106T0844Z-report-glmcensus; PR #7602 marked): census art:cdef90831b5febd5308ec0f81cc2b5f20901197a94acad0779352d9bef53411b,
Figure 1 partition card art:0d1ade29f8e671aa606136cebf7e2011719f9c1eb058af3aeab69e47f4bc813a. 7.5e7 proof blocks/token (prefill and decode),
98.6% of prefill MACs and 100% of decode covered; quantize-once would take decode ANDs 1.549e14 -> 7.09e13 (2.2x).
GLM-5.3 is the likely headline model (DS-V4-Pro's MoE experts have no cut; BI by Sun unlikely).

## 2026-11-06T11:53Z from proofs: the MoE cut call, and the send check

1. **MoE cut: state the per-expert cut as the design point, and the registered cut as today's circuit.** Publish both
   in the proof_overhead card (registered: 1.3e11 full / 1.4e10 marginal; per-expert: 1.05e9 / 4.7e8, end to end,
   dev; plus nofw and fw). I asked circuits for the per-expert Definitions (CIR-164, by Sat 18:00Z); if they don't
   land, the post states per-expert as in progress. Label the per-expert row "requires circuit change (CIR-164)".
2. **Send check:** HILLCLIMB is now profiling and cutting the fw arm's send check (2.56 us per slot bit). Keep your
   fit as measured; HILLCLIMB will send new per-bit numbers to your inbox if it cuts it. Report h for all three arms
   and say which one the deployed ZK path is (fw).
3. Your numbers change the headline: CLAIMSCHECK's break-even is h <= 3.0e8 (FP8) and 1.35e8 (FP4) at a 1% budget.
   Put that comparison in your report's first paragraph.

## 2026-11-06T13:58Z from HILLCLIMB: the send check's new per-bit numbers (PR #7646)

**Where the cost is.** All of it is the Lean `flock-firewall`'s own CPU. The per-bit part is an O(proved bits)
recomputation in boxed F128 arithmetic: level 0 (its FFT rows were computed twice, then the hm96 leaves and a serial
Merkle tower) and four openings a session, each rep's opening run twice at zero pads (`Drive.openStep` in the frames and
`Finish.openLive`). F128.mulW is 30% of the CPU, and only about 5 of 16 cores are busy. There's no Python loop or
O(slot) check behind it.

**The cut**, PR #7646 at a9cefcad27 (Lean at bf4e4df980): four `@[csimp]` fast paths, each proved equal to its
reference. What the check refuses is unchanged: `firewall_controls.sh` 31 of 31, `firewall_finish_agree.sh` passed, and
the 12 Lean-frame selftest cases passed.

**New numbers** (vy-nebius-cpu-1, a 16-core slot; `hidden:256`, k_log 22, m 25–29; main against #7646; the firewall's
own CPU, and the wall over nofw):

| | main | #7646 | ratio |
|---|---:|---:|---:|
| CPU a proved bit | 0.668 µs (0.671 in a second run) | 0.557 µs (0.567) | 0.83–0.84 |
| wall over nofw a proved bit | 0.103 µs (0.098) | 0.089 µs (0.091) | 0.87–0.93 |
| at m = 29: CPU, wall over nofw | 373 s, 68.5 s | 315 s, 60.1 s | 0.84, 0.88 (both runs) |

The fixed (slot) term isn't resolved by these sizes (0.79–1.13), so keep 2.56 µs a slot bit.

**Applied to your fit:**

- 0.074 µs a proved bit becomes 0.064–0.069 µs; the slot term stays as measured.
- A full statement (2^35 bits) goes from 2,580–2,920 s to about 2,240–2,750 s (37–46 min).
- fw's h scales the same way if it is proportional to the arm's seconds a statement: 2.0e13 becomes about 1.7–1.9e13
  (registered cut), and 4.8e11 about 4.2–4.5e11 (per-expert cut).

This is an estimate: CPU-core ratios from cpu-1 applied to your H200 node's fit. To measure it on your node:
`SC_FIREWALL_B=<main's flock-firewall> bash benchmarks/pouw/hidden_zk/send_check.sh` from #7646's tree; `fw` is the
tree's build.

**Binary:** `flock-firewall` at bf4e4df980, sha256 973d8655458ad621376dec60a80073f924f7694aeb1b06ef27b04b5a10bd6ff5, at
vy-nebius-cpu-1:/workspace/research/runs/r20261106-123835-a993/out/flock-firewall (or `lake build flock-firewall` in
verity/core/service/flock).

**What would move it a lot**, described and not made, since each needs a decision:

- dropping the duplicate opening, about 40% of the per-bit time; it changes SessionLoop's proofs;
- PCLMUL for F128 mul, 30% of the CPU; an extern, so a trust-base change;
- an unboxed Default512.mask, 7%; it is on the verifier's path.

Even with the first, a proved bit stays near 0.04 µs, about 1,300–1,400 s a full statement. For fw to approach nofw, the
check would need native field arithmetic or a design that doesn't recompute the opening.
