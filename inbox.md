[2026-11-18T08:22Z] da21c0e33e78
From the kernels lead, for the card-by-short-name reader, no reply needed:
- CIRCUITS will post on TOP-441 "what our probes and the target need from the one reader of a card by its short name" when its worker returns it (said at 12:21 AM PST). Read TOP-441's newest comments for that line before you fix the function's signature, and again before you push; if the probes need something your reader does not give (e.g. the card from a driver's name AND from a short name through one record type, or the architecture with it), give it in the same function's result, not in a second reader.

