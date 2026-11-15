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

[2026-11-15T01:18Z] 2f136b29fd53
model: claude-opus-5-5-high; subscriptions removed: none held.

PREMISES, Sat 14 Nov 5:18 PM PST.

**Every "draw" site in the five directories is settled: by the lint, 18 draw sites go to 0. My three commits are `09279360404` (`benchmarks/slowdown`), `ba8debf6d97` (`benchmarks/training_slowdown`) and `419dc4ed1b6` (the policy). `benchmarks/circuit_properties` and `benchmarks/nci` needed nothing from me, because `draw_words_map.toml` had already run over them. No printed number changed.**

Sites by sense, and what each became (RAN):
- The sample (the verifier's coins choosing blocks) became "sampled":
  - `run.py`'s "every ingress unit is sampled" and the test that pins it;
  - `data/reference.toml`'s "first sampled incorrect step";
  - `data/envelopes.toml`'s sources "sparse samples today" and "steps in a class sampled at a floor of one", which follow tiers.py's new text;
  - `run.py`'s local `drawn`, now `sampled`.
- A draw of coins became a flip: `run.py`'s "(the coins' draws)" is now "(the coins' flips)", the seat's convention.
- Plotted:
  - `figures.py`'s "∞ drawn at the right edge" is now "∞ plotted";
  - `test_figures`' "drew nothing" is now "plotted nothing" (the lint doesn't count it);
  - `test_analysis`'s list of rendered figures, `drawn`, is now `plotted`.
- A random value from a generator or a fuzz: this sense isn't retired, but the lint flags it.
  - `test_resume.py`'s generator values `drawn` are now `sampled`.
  - `desc_fuzz_programs.py`'s function `draw` and its two mentions are now `pick`, "picked at random".
- `benchmarks/circuit_properties` (`points.py`, `work.py`, README, `test_work.py`) and `benchmarks/nci` (`exposure.py`, `statement.py`, `tiers.py`): `draw_words_map.toml` renamed these at `9bf4f27fa80`, with "sample" and "sampled" for the points' random words. The lint shows 0 draw sites in both. CALLERS's 13 listed sites are among them; in `benchmarks/training_slowdown` only `test_resume`'s 2 were left, and I took those.
- `benchmarks/inference_only`: no draw sites.

Every identifier I renamed was a local variable or a function read only in its own file, so no word map was needed. I kept no stored key, seed tag or record key, because none held the word.

Checks (RAN):
- Calculator, against the scratch lock: 42 lines change their words (the δ_coin and certified-ingress lines), `periods.py`'s output doesn't change, and 0 lines have a changed number. `test_analysis` and `test_figures`: 193 passed.
- `test_resume`'s test passes.
- The fuzz script's output is byte-identical before and after.

Listed and left:
- `benchmarks/nci`'s 19 "law" sites (`README.md` 1, `fit.py` 3, `statement.py` 9, `tests/test_statement.py` 6). The law map ran at `632a1cb4ec9` and left them, rightly: every one is a loss law (scaling law), L(N, D) = E + A/N^α + B/D^β in `fit.py`, and the local `law` in `statement.py` and its test is the name of the fit a target reads (`FIT_TARGET`, `fit.FITS[law]`). None of them is a sampling design. They need a lint exception or a word of their own, such as `loss_law` or `fit`, from that map's owner or from me if you say so.
- `benchmarks/nci/algorithms.json` ("64 draws", "16k draws": the random directions of a zeroth-order optimizer) and `parameters.json` ("one noise draw"): data files, which the lint doesn't count. They are another sense, a generator's random draw, so I left them.
- `tests/lint/allowlists/repo_retired_terms.json`'s counts for these files: a shared list, left for the regeneration.

