# The inference-only policy's literature index and slowdown board: design

## How estimates merge findings

An estimate answers one question in one context, such as "bits per activation crossing a training pipeline boundary at
retained quality". Its three values follow one convention:

- `low` is what a malicious developer could reach on the best published evidence, however thin.
- `central` is what the strongest measurement at the largest scale supports.
- `high` is what standard practice uses.

## How we measure slowdown

**The yardstick.** Suppose an actor's last pretrain reached held-out loss \(L^*\) with compute \(C^*\). After the deal,
how much compute must a compliant circuit spend to reach \(L^*\) again? The slowdown, net of better recipes and of
progress, is

\[
R \;=\; \frac{1}{g}\,\min_A \frac{\text{slowdown}_A}{m_A}.
\]

- \(\text{slowdown}_A\) is the policy's slowdown for training strategy \(A\): the board and the library compute it from
  \(A\)'s traits.
- \(m_A(C) = C_{\text{ref}}(L_A(C))/C\) is \(A\)'s compute multiplier: the compute the reference recipe needs to reach the
  held-out loss \(A\) reaches with compute \(C\), divided by \(C\). It is the compute-equivalent gain (`davidson-2023-ceg/1`),
  with the reference and the test set made explicit. The reference is a compute-optimal dense AdamW Transformer (`hoffmann-2022-chinchilla/1`), with
  \(C_{\text{ref}}(L)\) from a Chinchilla-form sampling design (`besiroglu-2024-chinchilla-replication/2`).
- \(g\) is algorithmic progress in the base recipe (architecture, data, optimizer, allocation) since the last pretrain. It
  applies with or without the policy, so the talk shows it and then brackets it. Loss-based estimates put it near 2.5x
  per year, range 1.1-5x (`ho-2024-algorithmic-progress/3`, `gundlach-2025-origin/14`, `whitfill-2025-selection-bias/1`).
  Capability-based measures, which include post-training and data, run faster: 3.5-10x per year
  (`whitfill-2025-time-horizon/1`, `ho-2025-rosetta-stone/5`, `ho-2026-least-understood/1`). Most of it is

**The loss.** Bits per UTF-8 byte, which does not depend on the tokenizer (`gao-2020-pile/1`,
`huang-2024-compression-represents-intelligence/5`), on a broad multi-domain corpus (`magnusson-2023-paloma/1`) written
after every compared model's training cutoff (`li-2024-compression-time-split/1`), fixed and published in advance. The
corpus matters: a different validation set can reverse which recipe wins (`lourie-2025-unreliable/2`), and
quality-filtered data can lose on broad web text while winning benchmarks (`penedo-2024-fineweb/3`). Loss is preferred
to benchmarks because it scales smoothly and extrapolates within a recipe (`gadre-2024-overtraining/2`,
`brandfonbrener-2024-loss-to-loss/1`), and equal loss means equal downstream scores when corpus and architecture are
fixed (`du-2024-emergent-loss/1`). Benchmark-based multipliers appear in the library only flagged.
