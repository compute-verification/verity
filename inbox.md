[2026-11-15T01:16Z] 077abd7e55f5
model: claude-opus-5-5-high; subscriptions removed: none held.

PREMISES, Sat 14 Nov 5:16 PM PST.

**The calculator's draw sites are settled on `architecture` at c6973b6561e7: the lint's draw sites in `benchmarks/slowdown` go from 12 to 0.** (RAN) By sense:
- The sample becomes "sampled" in four places: "every ingress unit is sampled" (`run.py` and the test that pins it), "first sampled incorrect step" (`data/reference.toml`), and "sparse samples today" and "steps in a class sampled" (`data/envelopes.toml`'s sources, which follow tiers.py's new text). The local variable `run.py`'s `drawn` becomes `sampled`.
- The coins: "(the coins' draws)" becomes "(the coins' flips)", the seat's convention for a draw of coins.
- Plotted: "∞ drawn at the right edge" becomes "∞ plotted" (`figures.py`), "drew nothing" becomes "plotted nothing" (`test_figures`, which the lint doesn't count), and `test_analysis`'s list of rendered figures `drawn` becomes `plotted`.

Outputs, against the scratch lock: 42 lines change their words (the δ_coin line and the certified-ingress line in each scenario of `run.py`), `periods.py`'s output doesn't change, and 0 lines have a changed number. `test_analysis` and `test_figures`: 193 passed. I didn't use the seat's piece: every site was prose or a local variable no other file reads, so I edited by hand.

[2026-11-15T01:18Z] a80d8ee05c71
model: claude-opus-5-5-high; subscriptions removed: none held.

PREMISES, Sat 14 Nov 5:18 PM PST.

**`benchmarks/training_slowdown` is pushed: `ba8debf6d97` (merged as `ba8debf6d971`). Its draw sites go from 2 to 0 by the lint.** (RAN) In `test_resume.py`, the local `drawn` held three values the restored random generators produce, a sense the Glossary row doesn't retire. The lint flags it anyway, so it is now `sampled`. `test_the_generators_a_step_may_read_come_back_as_they_were_kept` passes. Nothing else reads the name.

[2026-11-15T01:18Z] b774ddd8e21f
model: claude-opus-5-5-high; subscriptions removed: none held.

PREMISES, Sat 14 Nov 5:18 PM PST.

**The policy's directory is pushed: `419dc4ed1b6`. Its draw sites go from 4 to 0 by the lint.** (RAN) In `verity/protocols/policy/inference_only/verifier/lean/scripts/desc_fuzz_programs.py`, the function `draw(rng, seed)`, which picks a fuzz declaration's parameters at random, is now `pick`, and its two docstring mentions say "picked at random (`pick`)". It is called only inside the file; the certifier's test calls `fuzz.shared`. `draw_words_map.toml` keeps this file's `draw` as another sense, but the lint counts it. The script's output, one Programs file of 7.2 MB, is byte-identical before and after (`diff -r`). `tests/lint/allowlists/repo_retired_terms.json` still lists a count for the file; it is a shared list, so I left it.

