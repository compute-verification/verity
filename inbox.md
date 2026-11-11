[2026-11-11T01:38Z] 6e8c438ee95a
TRN-93 PAIR, early word (Tue 10 Nov 5:38 PM PST). Nothing needed from you; the full note and comment come before Wed noon PST.

1. HeldRate, at the prices the 103 uses, is false. The 103 charges a held weight 2.5 bits and a token's entry 6,236 bits (0.87 bits a value times the residual stream's 7,168 values). A block need not enter a token at full width. Construction: hold one column of wq_b (65,536 weights a step) and read one bf16 coordinate a token, 16 raw bits, no coder assumed. Filled to the serving block's 3.19e11 bits it works on 2.0e10 tokens; the premise allows 5.1e7. That is 390 times over. Every raw entry narrower than 390 of the 7,168 coordinates undercuts it; the attention latents (2,048 values a token) do if they code under 3.04 bits a value, 3.5 times over at 0.87.

2. The bound's number is not broken by it. Those blocks do few products. Forward products a serving block, mine, from main's shapes: column tile at most 2.6e15, attention latents 5.1e16, held layers 4.7e16, the last two layers with the head 7.5e16 (the most I could build), against the pair limit 1.63e18: 22 times under. Replay from data does not fit: the reference model's state is 12 times the block's bits even at 2.5 bits a weight.

3. Today (ran at main 2b869ed35d): 102.8 a week, a month and a year, one datacenter and the world alike. #5173's own tree gives the same number to 17 digits: no input the class reads has changed since 27 Oct. Hidden training gets 0.389% of a period's gates; reference training runs that fit: 0.048, 0.206, 2.51 for one datacenter, 180, 772, 9,396 for the world. #5173's own premise row called HeldRate contested past 1.1e9 bits.

4. Landing: main lost every Nci.Ldl file on 6 Nov (923bdcc7e4). Core now has Lemma 1 and Theorem 3 (Verity.Guarantees.BlockDescription, Counting), and Nci.Formalism's token-work bound has the pair bound's shape. None of the five open deletions touches either.

Leaning: leave #5173 closed. The 103 comes from a limit 22 times above the best block known, not from a true premise. Command: cd /agent/repos/verity && uv run python /tmp/trn93/pair_scratch.py --tornado (the script's text goes in the note). Checkpoint: note training-20261111T0120Z-checkpoint-trn93-pair-first-figures.

