[2026-11-12T23:23Z] df4cc2878d09
ROWS, Thu, 3:23 PM PST. **Done: all seven parts are in the report `training-20261112T2321Z-report-ROWS-figures-in-rows`, with the handoff `training-20261112T2321Z-handoff-ROWS`. Both are synced, and the scratch branch is pushed.**

The LEAN-238 answer, 93 words, as it can be posted:

> training (ROWS). Training's calculator does not count in rows: it prices a step's MACs by kind at each Definition's AND + XOR + NOT, and holds no templates. In AND gates every c and the run's gates at S fall exactly 3.008×, |C| 2.88–2.92×, ρ and S 3.1–4.1%. In rows of each Definition lowered alone they fall 2.94×, 2.76–2.81× and 4.5–6.0%. k doesn't move; rho moves only on a measured design. Γ's m per template needs the adapter's cut and `UnitStager`'s lowering per shape; served `Gemm_v2` is within 1.4% of its calls' pins.

- **Part 2's factors** are as in my 3:20 PM message [RAN r20261112-223324-5495, ended, 99 of 99]. In AND gates, then rows alone:
  - c and the run's gates at S: ÷ 3.0084, then ÷ 2.9364.
  - |C|: ÷ 2.884 to 2.916, then ÷ 2.759 to 2.805.
  - ρ: × 0.9587 to 0.9694, then × 0.9397 to 0.9551.
  - S: × 0.9587, then × 0.9397.
  - Most moved: `s-t20-1.1x-fullmuon-fe-wsd(-ema)` and `s-t20-1.1x-ref-optim-961`. Least moved: xl.
- **Item 6** has its own heading. The gap is 3.06–4.13%, largest at `s-t20-1.1x-fullmuon-fe-wsd-ema`.
- **Part 7** is sized for UNITS's 2:54 PM answer. Three switches:
  - A swaps the column to AND.
  - B pins the eight rows-alone m.
  - C reads m per template, and needs a map from ops to templates and the adapter's cut, which training doesn't hold [INFERRED].
- **Part 5** is sized, not run: 298 templates at vllm's `69655a037c6a`, through `class_statement.unit_lowering`. Γ in rows is about 3.8·10¹² a block, before table reads [INFERRED].
- **Every rows figure names its writer.** The rows-alone column is both writers'; the template m is given at both.

Runs:
- Ended with success: c272, 3b38, 939e, ee4b, 5495, bfaf.
- Failed: 57c0 and e4c7, both understood.
- Cancelled: ba47, 80c8 and 399a.
- None is running.

