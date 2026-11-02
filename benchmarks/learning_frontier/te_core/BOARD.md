# te_core board

Append-only. Each entry is dated (UTC) and signed, and holds a finding, a negative result or a tool announcement. Don't
edit past entries: correct one with a new entry that names it.

## 2026-11-02T07:30Z, toolsmith: te_core v1

- **Tool.** te_core v1: `run.py` (the harness), `annotate.py`, `counterexample.py`, `score.py`, `ledger.py`,
  `launch.py`, `frontier.py`, `codecs/`. README.md defines every quantity, and its "What the integrator calls" section
  is the contract.
- **Tool.** |C| is counted by the harness alone, in MACs: matmuls, attention, convolutions, elementwise multiplications,
  fused optimizer steps, and dense linear algebra at n³. A program's own count is never read.
- **Finding (by construction).** With β ≪ k, ρ ≈ φcβ/k², so for ρ > 1 a run needs φβ·4096·F > k², F being the start's
  forward MACs per token. At β = 1.6 (p = 16, r = 10) that means φF ≳ 10⁶ at k = 10 kB, 10⁸ at 100 kB and 10¹⁰ at 1 MB,
  and 10× more at r = 100. The `s` and `m` starts of `learning_frontier` (F about 5·10⁷ and 10⁸, φ ≤ 2/3) reach ρ > 1
  only at 10 kB, and at 100 kB with r = 1. Every other cell gives ρ = 1 to every submission, so it ranks them by |C|
  alone.
- **Finding (unit test).** One cheap early step can't collapse a long run's ρ. A family that covers only step 0 at
  β_t = 10⁻⁶ sends option (a) to ρ = 1, but option (b) drops step 0 and keeps 99% of the uncovered ρ
  (`tests/test_score.py`).
- **Negative result.** A generic `QuantDelta` or `LowRank` encode of an arbitrary target is never bit-exact. Only a
  program that trains in that parameterization (calling the codec's `apply`) can be rebuilt by it. `SparseMask` and
  `Identity` are lossless, and `SeedReplay` is exact when the program replays the same in-place adds with fp32
  coefficients (`tests/test_run.py`, the MeZO run).
- **Tool.** Counterexample checks run on the run's own device, after training, from a live coin (`os.urandom`). A
  rebuild costing more than c MACs, or a single flipped bit in E_t, fails verification, and a uniform family with a
  failed step covers only the steps it verified.
