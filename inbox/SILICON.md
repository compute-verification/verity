# SILICON: what multiply-accumulate hardware would make PoUW cheap and its premises strong

Daniel, 29 Oct 2:31 PM PDT: "suppose I was Nvidia ... what properties would a new way of doing multiply-add-accumulate
have to integrate best with noising, denoising, cryptography ... easy to verify, circuit-level properties about how the
MACs work ... integrate the hashing into the MAC, or substantively decrease the hashing ... a swarm, a preliminary
exploration of the design space." Academic: nobody builds it this year. The output is the map.

## Why hashing is 99% of the lead's credit (the problem this attacks)

- The lead's h = 127.5: honest hashing per stock MAC (HASH.md, §interim-1). The MMA has no hardness premise.
  - It is linear across steps.
  - Its noise is low-rank (a row walk plus a column walk: w = x + R_t(i) + C_t(class(i), j)), because coins cost
    bandwidth.
  - So a cheater shortcuts it (COUNTER's `decomp`). SHA-512 is the only thing with a cost premise, so we credit
    SHA-512.
- Software folds help: R80X runs SHA-512's rounds over the words, h ≈ 18 at rank 4a (FOLD-ROUNDS).
- The question here is which property of the MAC itself would make the useful work creditable, or cut the hashing to
  a seal per call.

## Trust model (fixed)

- **Trusted:** that the chip computes its published MAC function. That function is a circuit-level spec, testable by
  measurement as `verity/ml/circuits/silicon` is today.
- **Not trusted:** no keys, no remote attestation, no device secrets, nothing the device signs.
- **Cost premises** about the new unit ("the cheapest way to compute f on this chip is its MAC path") are ranked on
  HASH's ladder. A premise about a standard primitive (a SHA-512 or Keccak round) ranks above one about NVIDIA's own
  mixer.
- **Rules unchanged:** everything zero-knowledge, live coins only, SHA-512 the one hash. By the 21:15Z ruling, a
  mixing round that doesn't bind isn't a hash.

## Metrics, for every design

| metric | meaning |
|---|---|
| h | honest hashing per stock MAC, after the change |
| Δarea | gate equivalents per MAC, synthesized, and % of the baseline s8 and FP8 MACs below |
| throughput, latency | the MMA's rate and pipeline depth, before and after |
| premise, rank | what the credit rests on, on HASH's ladder |
| T1 | the gap a cheater gains, attacked with `decomp` and `pack` first |
| T3 | extra served bytes, e.g. coin bandwidth |
| proof cost | C-Flock gates (or an arithmetic proof system's constraints) per MAC |
| useful | runs today's int8, FP8 and NVFP4 models exactly? |

**Baseline**, owned by SI-ALG and published first (22:45Z):
- A 4-wide s8×s8→s32 dot-product MAC.
- An FP8 E4M3 MAC with FP32 accumulate.
- Both synthesized with yosys + ABC (`pip install yowasp-yosys` needs no root) to a generic gate library, with the
  script on the branch.

## Lanes (notes: `lanes/compute-research/<ts>-report-silicon-<lane>.md`)

- **SI-FOLD: the accumulate path.**
  - Mix the accumulator non-linearly with a standard round ρ: a SHA-512 round, a Keccak-f round, a ChaCha
    quarter-round, an AES round, an Ascon or Xoodoo round.
  - Placement: each k-step, each prefix word, or each tile output.
  - What the credit's premise becomes, the gate cost of ρ against a MAC, and whether h reaches about 0. Also the
    cheapest mixing whose premise is about a standard primitive, and what `decomp` does to each.
- **SI-ALG: arithmetic and formats.**
  - Prime-field MMA (Mersenne-31, BabyBear, Goldilocks, 2^61−1); binary tower fields, carry-less (Binius-style, which
    fits Boolean circuits); RNS; Kulisch or fixed-order exact FP accumulation; wide integer accumulators.
  - For each: area; noise (perfect one-time-pad hiding, exact cancellation, no (127, 1, 1) weights); Freivalds and
    Schwartz–Zippel soundness; proof cost per MAC; usefulness (int8 MMA embeds exactly in F_p for large p); and
    whether algebraic sketches become sound enough to replace the per-word hashing.
- **SI-NOISE: noise and sketches on chip.**
  - A seeded generator in the operand path: a public function of the verifier's live coin (Trivium or Grain,
    ChaCha, AES-CTR, a Keccak squeeze), giving full-rank per-element noise at zero bandwidth.
  - Does that kill `decomp`, and let NH-class sketches (h ≈ 1) or a design with no per-word hashing survive?
  - Hardware sketch units on the accumulator stream (NH, polynomial hash, GF(2^k) multiply), and the denoising term
    computed on chip. What hashing remains at minimum: one SHA-512 seal per tile, or per call?
- **SI-THEORY** (when a slot frees): what property of a MAC's function admits work credit under a standard premise.
  Literature: fine-grained PoUW (Ball, Rosen, Sabin, Vasudevan 2017), PoUW from matrix multiplication (check
  Komargodski, Schen, Weinstein), RandomX's hardware-native design, memory- and bandwidth-hard functions, and zk
  hardware (ZPrize, zkASICs).
- **SI-RED + SI-CARD** (when slots free): attack each proposal, and card the frontier of Δarea against h against
  premise rank.

## Times

- Baseline at 22:45Z.
- Each lane's preliminary note by 00:00Z, which the coordinator synthesizes for Daniel by 00:30Z (5:30 PM PDT).
- **Done-when:** a map of design classes with the dominant point in each (Pareto over Δarea, h and rank), each number
  marked checked, measured or estimated, and the one or two hardware properties that buy the most.
