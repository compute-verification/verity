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

## 2026-11-02T07:50Z, toolsmith: te_core v2 (covert_witness script mode)

- **Tool.** te_core moved to `benchmarks/covert_witness/te_core/`. `python -m te_core.run --script islands/CLASS/train.py
  --out OUT -- --start NAME --out SUB ...` runs a covert_witness island unmodified (`cw.py`). A step is one data-stream
  batch to the next. Checkpoints are the E's the island encodes (the first in each 5% of the run, and the one it
  submits), each scored as decode(θ, E) on the `search` set by `prepare.evaluate_bpb`. The record says whether the
  submitted model is decode(θ, E) bit for bit. `te_core.hooks` is optional.
- **Tool.** The counter recognizes θ's tensors by value: a persistent tensor covering its storage that equals one of θ's
  exactly in both dtypes (an island's `p.data = p.data.float()`) is θ until it is written; the start's buffers (RoPE
  tables) are θ's. v1 counted such a recast as an updated weight from step 0.
- **Tool.** Harness work (loading, encoding, decoding, evaluation) is neither counted nor timed, and `totals` seconds
  are the steps' (`process_*` is the whole process). The screen matches names, not substrings (`time.perf_counter` and
  `torch.compile` pass: dynamo is disabled before torch loads).
- **Finding (CPU smoke, tiny start).** Under the harness, the reference island counts 1.06× its analytic MACs (the
  elementwise products and Newton–Schulz work it logs as `extra`). Step 0 is all `fwd/theta`, and every later step is
  `fwd/updated` across the blocks and the tied head. For e100kb, U is the adapters and norm gains only. For seed (MeZO),
  U is the perturbed blocks and the unperturbed tied head stays θ.

## 2026-11-02T08:00Z, toolsmith: GPU smoke of v2 on vy-nebius-2 (RTX PRO 6000)

- **Finding.** On `s100`, at 2% of the budget, the reference island ran 93 steps under the harness. It counted
  9.70·10¹⁴ MACs, 1.086× its analytic count. It ran at 0.30 s per step, 6.3·10¹³ FLOP/s, against the 1.49·10¹⁴ the
  budget assumes for compiled training. A harness run therefore takes about 2.4× a compiled run's wall time, which the
  tier timeouts (3× the tier minutes plus 15 min) allow. It raised no flags, and its model is decode(θ, E) bit for bit.
- **Finding.** On `s100`, e1mb (LoRA, E 959,754 bytes) has U = the adapters and norm gains, 1% of the forward. The seed
  island's log replays bit for bit on CUDA. `cpu_parallel` fired on the LoRA run (43 CPU-s in 25 s): autograd's device
  thread and the main thread both drive the dispatch mode, so read that flag as informational for adapter runs.
- **Tool.** vy-nebius-2 takes runs only through its slot pools (`--gpus` is refused). `te_core.launch` sends
  `NODE pool=gpu` lines as `--pool gpu` with `gpu-lease 1` inside. `research run`'s node-local pools need main's
  `research`.
