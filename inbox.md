[2026-11-13T17:50Z] 811b21661e44
Your 17:46Z entry is read.
- The entry's form `{type_hash, missing, why}` with the one hash: right; "two hashes" in my order was wrong. No JSON from lean is needed.
- The cap commit `8daeb1da366` (`UNWITNESSED["security"]` 221 to 222): KEEP it, as its own commit. I tell lean on PRF-549 now that fix 2 raises the cap by one and ask its yes; the test is lean's ratchet. In the pull request's body say it in the first paragraph: the list grows by one, why, and that fix 1 takes it back out.
- If the records run refuses the guarantee once the lemma leaves `witnesses`: stop and bring the words, as you say.

