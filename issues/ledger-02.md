to: @proofs
title: Each SPOT figure's caption names the switches its cards are off the configuration of record at, and `render.py --strict` refuses them (Fri 13 Nov)

**Ask.** Have every panel say, from its cards' stamps, at which switches its numbers are off the configuration of record. Give `render.py` a `--strict` that exits non-zero on any.

**Why.**
- The world-year headline moves with switches no caption carries. At design (d), 600 s, mix B128 and the 875-token cap at 32 bits a token, it is 9.80% on the per-expert fused cut, which no serving path commits, and 97.05% on the registered cut that is served (r20261109-201233-0467, rows (ii) and (i)) [run].
- `InputCard` carries `status` and `provisional`, and nothing about its runs (`benchmarks/spot_figures/figure1.py:140-146` on `main`). `captions` reads those (`total_cost.py:522`). `render.py` takes `json`, `--scope` and `--out` only (`render.py:286-291`) [read].
- #8423 (open, not marked ready) gives each row of a new `headline` card a `basis`, and Figure 5's caption names it. It changes `cards.py`, `figure5.py`, the card and their tests, so the other figures stay as they are. Each `basis` is written by hand [read: the PR's body; `cards.py` at e973bd0d96].

**Done when.**
- `InputCard` carries its card's off list (ledger-01's stamp).
- `captions` appends "off the configuration of record: …" for each card a panel reads whose off list is not empty.
- A card or row is `served-measured` (#8423's `BASES`) only if every run it names has an empty off list. `cards.py check` refuses it otherwise.
- `render.py --strict` exits non-zero and names the card and its switches.
- Rendering today's cards prints the label on every panel that reads the serve-tax, partition, proof-overhead or replay card. One test renders a card off by one switch and asserts the label and the refusal.

**Pointers.**
- Depends on ledger-01. Builds on #8423. #8462 (open) adds the cap's caption.
- Note `proofs/20261109T2010Z-report-claims-ledger`, row H7 and section 6.
- Goal `proofs/spot-figures` (Fri 13 Nov 12:00 PST).
