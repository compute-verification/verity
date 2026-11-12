[2026-11-12T17:42Z] c21f9e4af7de
TIE to training, Thu 12 Nov 9:45 AM PST. TRN-86 items 1, 3 and 4: plan, what moved, my date. Nothing built yet.

**My date: the PR ready to mark Thu 12 Nov 8 PM PST.** The one risk to it is the Lean part of item 4 (below); if its records run is still out then, Fri 13 Nov 9 AM PST.

**What moved on main under me** [READ, `origin/main` at `9ddae97ba2f7`, fetched 9:29 AM PST]
- The four landed: #8589 `589249ad95`, #8591 `d21813f29c`, #8703 `e64ebb84e0`, #8700 `eb0b925dfb`.
- Since #8703 landed: `benchmarks/inference_only` has not changed. The calculator (`verity/protocols/inference_only/rho`) changed only in its README (#8606) and in PoUW's per-phase failure as a parameter (#8813: `data/envelopes.toml`, `rho/params.py`, two tests); `rho/min_cut.py` and its test are as #8703 left them. `security/Proofs/Nci/CutCharged.lean` is as #8703 left it. #8700 renamed lean's count over a set of blocks to budgets (`bound_on_budgets`, `bound_on_budgets_he`; `bound_on_rows` beside them).
- `tied.toml` is not on main: it is mine to add. `benchmarks/compute` has the give-up by call (`--per-call`, #9021, #9072, #9085).

**Plan, item 1 and 3** (one branch from `origin/main`, `cursor/trn86-period-print-20af`)
- `benchmarks/inference_only/tied.toml`, beside `served.toml`, in the same keys: the served class with every tied value right. `c` and the work are the served policy's; `k` = 146,080 in its two terms (the scan state row, 145,568, and a 512-bit salt); a table for what the tie leaves: a node's bound of 671,369 untied words for each 4 blocks of one 600 s execution phase, 2,048 MACs a word at most, 32 bits a word. Its header says nothing serves it and what it rests on.
- `verity bench inference_only --uncredited-share S`, no default. Without it the command prints what it prints today (the served policy's 1 first), then one line: the tied figures print only at a stated share, and no PoUW verdict from this tree gives one. With it, after the served print, a second section under its own name:
  - the label first: the share was typed on the command line, so every figure of the section rests on no PoUW verdict;
  - which work the figures are for (item 3): bf16 values, every weight and the embedding table outside the registered weights, and that (H) is false of a work that reads a registered one;
  - one block: the cut number with every word tied, never alone, beside its limit `1 / (s + 1/ρ)` at the share;
  - a week, a month and a year, a datacenter and the world (the calculator's fleets and periods, as the served table): the blocks, the untied words a block, then each reading under `Charged.reading`'s name (one β_x with the best threshold; one β_x at the top budget, no block charged; a β_x a block) with its threshold and the blocks charged, its figure, its limit at the share, and the pooled total in gates (`Charged.gates`, and the same at the limit);
  - a fourth reading, the one compute-accounting took: the uncredited words counted as free input (`k + 32 u`, `u` at most `s N / 2,048` words a block) and their MACs charged whole;
  - proved, premise, scaled, as the served table has them; the section's rows in the JSON document.
- Tests in the benchmark's suite: the file against the calculator's own constants, no share prints no tied figure, a typed share prints the label and the function's own figures. README of the benchmark.

**Plan, item 4.** With today's lint the ratchet gate reads #8703's landing as 45 added sites, not 47 [RAN, `ratchet_gate.py e64ebb84e0^ e64ebb84e0`]: "cap" 38, "Nci" 5, "sampled proofs" 2.
- 41 go: `rho/min_cut.py` 18 to 0, its test 13 to 0, the calculator's README 1, `Definitions/Nci/Guarantees/Cut.lean` 3 (docstrings), `Proofs/Nci/CutCharged.lean` 6 (docstrings).
- 4 stay, each a name Lean reads: `CutCharged.lean`'s import line, `namespace` and `end` (it ends at 3, not 0), and the import of it in `Proofs/Nci.lean` (that file is below its base count already).
- Words: what a block that isn't charged reads at most becomes its budget (the Glossary's fresh-input budget, and lean's own `bound_on_budgets`), so `Charged.cap` is `Charged.budget` and the reading is "one β_x at the top budget, no block charged"; `capped`'s figure is the limit at the share. `Charged.cap` is read by `min_cut.py` and its test alone [RAN, `rg`]. No Lean name changes.

**Two things to say now**
1. The Lean prose (9 words in two Lean files) makes this PR's check rebuild `security`'s packages, since `Guarantees/Cut.lean` is imported widely. I include it, as the issue says, with `verity lean try` on both files and a records run to show no record moves. COUNTS edits `CutCharged.lean` for item 2; I tell it which six docstring lines I touch. If you would rather the Lean prose ride COUNTS's PR, say so and I drop it from mine: the Python and README sites (32) do not depend on it.
2. I send COUNTS the item 2 pointers and you the paragraph on `verity bench compute --per-call` next, both before I build.

