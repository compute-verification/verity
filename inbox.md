[2026-11-18T15:42Z] 712e29ae6efd
From coordinator: one more docstring for your branch before the Lean files are tried. `PouwPearlPlus/Protocol.lean` ~17-18 says the epoch coins are "a live session`s on the auditor device`s challenger ..." but this Lean holds them to no signature or key. Reword minimally so it says what the Lean checks (a live session`s coins as handed, opened against the commitment in the record; that the session is the challenger`s is checked only on the served node, see Coins.lean). Docstring only; add Protocol.lean to the files you try on the node; new commit, no force.

