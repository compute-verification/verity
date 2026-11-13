circuits' worker FP8READ, Thu 12 Nov 4:50 PM PST (00:50Z Fri), third comment.

**The pull request is open, unmarked: [#9567](https://github.com/compute-verification/verity/pull/9567), head `70a27d97ec60`. Every one of the capture's 1,472 tokens now equals `FlashMlaV32Decode_v1`'s own evaluator: 6,029,312 words, 0 differ. No name, static or port moved.**

- **The evaluator over every token** is run `r20261112-185922-3b8c`, off the card, from the first card run's saved capture. It replaces the cancelled run my second comment named.
- **At the PR's head** (run `r20261112-231418-ada0`): circuit-check 17 targets, 0 failures, and the whole token 0 failures on edge vectors only; 57 suites passed and `repository` whole passed; the pins update pass changed no file.
- **The captures ran one commit earlier** (`1f6f0969a9`). Since then the script lost seven catch-all handlers, for the adapter's lint, and `main` was merged in. The saved capture compared again at the head gives the card run's record byte for byte. The launch stage at the head has not run on a card.
- **I run nothing on a card tonight.**
