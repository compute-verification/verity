# Aggregated PoUW words

Research analysis (GD-AGG, golden demo, 28 Oct 2026): committing one 16-bit word for a group of g output cells,
W = Σ f(z) mod 2^16 over the group, instead of one word a cell. Nothing in `verity` imports it, and its Definitions
(`AggMix`, `AggM32a`, ...) are prototypes for counting ANDs, not registered ones.

- `agg_f.py`: the candidate maps f, as numpy references over all 2^16 inputs.
- `agg_spec.py`: each f's additive spectrum (ρ_lin, the residue biases), the bound it gives on what linear statistics of
  a group's cells leak, and r_max.
- `agg_ands.py`: f, the sum and Dither's block on bits, in the repo's Boolean lowering, each checked on all inputs.
- `agg_attacks.py`: linear collapse, low-bit structure, separation and truncation, exact on `dither.py`'s coins.
- `agg_entropy.py`, `agg_partial.py`, `agg_partial2.py`: guessing with nothing computed, with each cell's dither
  product skipped, and with that product's group sum computed by the collapse.
- `agg_line.py`: the cells known up to one direction (g − 1 linear statistics), exact over the 2^16 values left.
- `agg_spec8.py`: the same spectra and bounds for 8-bit cell words (GD-NOISE's two-GEMM commits), f on a byte.
- `agg_cost.py`: the proof block each g implies, its ANDs, proving GPU-s a window and the serve-time bytes priced at
  GD-HASH's measured cost a byte, for Llama-3.2-1B at T 8,192: J's two terms by g and window length, the best g for
  each, and the sensitivity to the opened rows' sizes.
- `agg_reframe.py`: the same blocks at ε 10^-4 and δ 2^-40 a period, n → ∞: proving's share of a 1,000-GPU year, the
  serve terms with f's epilogue hidden or exposed, the best g under each cap and premise, and a K = 7,168 linear.

`uv run python benchmarks/pouw/agg_words/run_all.py OUT [step ...]` writes each step's JSON under OUT. The findings are
`note:compute-research/20261028T1945Z-report-gd-agg` in the notes store.
