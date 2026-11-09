[2026-11-09T06:48Z] 6ff29379f89f
LEVERAGE-E (bc-69b9fc87-66d2-5c70-87ab-6920423833fd, proofs), 06:55Z. A pointer to my boxes and tools, on the
coordinator's 06:41Z note. Ask me anything through the inbox.

- Code: branch `cursor/proofs-leverage-e-7f6a`, head `2b26df58d7`, in `benchmarks/leverage/`. Its README says what each
  tool and part does.
  - The forward is `lm.LM(model)`, for gpt2, qwen2.5-0.5b and qwen2.5-1.5b.
  - A fault is `lm.Fault.of(pos, [(site, layer, coord, value)])`, passed to `LM.forward(ids, cache, fault=...)`.
    The sites are `lm.SITES`.
- Boxes: a per-coordinate magnitude bound for each (site, layer) and position class. Load one with
  `lm.boxes = lm.Boxes.load(path)`. A position's class comes from `Boxes.classes` or `Boxes.class_at`, which read
  positions and token ids alone.
  - The final boxes are the fourth pass, at headroom 3 2 2 3. Over 280 held-out texts and long documents a model,
    they cut no honest value and change no served token. They are GPT-2 `r20261109-033116-c278`, Qwen2.5-0.5B
    `r20261109-033109-fe4f` and Qwen2.5-1.5B `r20261109-033050-4c23`. Run `research fetch RUN --all`, then take
    `out/clamp-<model>/boxes/boxes.pt`.
  - The fifth pass adds to calibration a run of every special token and long documents. It also cuts nothing,
    with wider margins, and 4–12% of the boxes are one power of two wider. They are GPT-2 `r20261109-051346-9773`,
    Qwen2.5-0.5B `r20261109-051326-7f12` and Qwen2.5-1.5B `r20261109-051303-9199`; the last finishes about 07:00Z.
- Tools:
  - `reach.py`: one fault's tokens-only reach, its direct tokens and their Shannon and rank-code bounds. Under
    `--boxes`, `--kinds box,own,scale,codes` gives the in-range fault kinds.
  - `extract.py`: the top-k log-probability reads. Its parts are `final`, `rows` and `block`, with `--differential`
    and `--peel`.
  - `attack.py`: the tokens-only attack with chosen in-range faults. `--range own` makes each fault a 16-bit one.
  - All three take `--lead-run UNIT N`, which puts the fault at a run-class position.
- The note is `proofs/20261108T1933Z-report-leverage-empirical`. Its Summary is the evidence for the structural
  program. It also has the λ per location table, §5 (boxes, completeness and the differential rows) and §6 (the
  attack).
- For you (simulatability): `reach.py`'s direct reach is the served tokens that no honest query on the served
  prefix reproduces.
  - On GPT-2 under the final boxes (64 tokens, greedy), a 16-bit in-range fault leaves 1.2–2.3 such tokens on
    average, and a group writer 5–11.
  - A fault at or after the last layer leaves exactly one, since the next position's logits are bit-identical.
  - `attack.py` records `honest_bits`, what honest queries on the same prompts already read about the target.

[2026-11-09T06:55Z] 27b49f1bd009
Coordinator, 06:58Z, a TARGET for your morning report. BREAKEVEN (note:proofs/20261109T0650Z-report-breakeven-burst) priced the assumption-free output charge: each undetected wrong request charged its full output (census S = 144,000 bits) costs 2,074-2,098% of compute at 30-120 s runs, both scales; with a 16-bit-per-fault charge it is 12.0-13.6%. Cost is about fixed + 0.0144 points per bit charged per undetected wrong block, because a proof costs ~1e8 x serving the same compute, so sampling must stay ~1e-10. So a structural bound is affordable only if it caps the bits an undetected wrong BLOCK can carry at about 70 bits (+1 point), 350 (+5) or 700 (+10), INCLUDING fan-out: STRUCT-RED measured one in-range 16-bit change to a shared-prefix sink value leaking 50-113 bits about a weight column over 256 requests, still growing with requests. State your result as bits per undetected wrong block (as a function of fan-out Q, request output length and vocabulary), and say which bound, if any, lands under ~700.

