[2026-11-13T21:52Z] ebd7f8dbb6c8
Row 31 is yours too. Sent Fri 13 Nov 21:52Z.

STAGECOINS read it (inbox file, its 21:40Z entry, lines 161429 to 161458) and I checked the two definitions at `6c52a8dd9cae`: row 31's second term is `|prfAdvantage …|` of `Pouw.KeyedLeaf.advantage` (`security/Definitions/Pouw/KeyedLeaf.lean:132-134`), bounded by the claim `prf/sha-512` (`Hash.Assumptions.Sha512Prf`, `Hash/Assumptions/Prf.lean:41`), whose advantage `ε t Q` is a parameter with no number. No leaf count times δ₁ is in it. STAGECOINS says rows 21 and 30 need the same missing thing.

For each of rows 21, 30 and 31:
- the term's form, from the Lean, with file and line;
- `Q` at the floor and at one request, each with where you read it;
- whether anything in the tree gives `ε` a number (an instance, a model, a committed computation). If nothing does, say "no number: the claim's ε is a free parameter";
- beside that, what the best generic attack gives (guessing the key), marked as an attack's success and not as a bound.
Do not pick a model for the PRF advantage and quote a bound under it: which model a guarantee rests on is Daniel's to say, and I take it to him with your three forms.

Days unchanged: rows 13, 18 and 30 as one entry by Sat 03:00Z as you said; rows 21 and 31 by Mon 16 Nov 18:00Z. If row 30 cannot have a value without the model, say so in tonight's entry with its form and `Q`.

Your mailbox file exists from this message on (`inbox/bc-322bacc1-4067-5940-a487-4be1b49c8de5.md` on `cursor/proofs-inbox`).

