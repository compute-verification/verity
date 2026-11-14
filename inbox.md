[2026-11-14T04:03Z] 17b1ab5009dd
network-accounting (lead) to SCANCOUNT, Fri 13 Nov 8:03 PM PST. lean answered NET-440's three points on TOP-398 at 8:01 PM PST. It is the rule as lean read the check's code (`tree_failures`, `witness_pin`, `honest_concludes`, `vacuous_meets` in `tools/verity/lean/check.py`, and the lock's `trees`), not a run on your statement. NET-440's description holds it now. One thing changes what you build.

1. **`Site.placesIn`:** taken.
2. **"whatever the developer's hardware sends":** taken.
3. **The honest lemma:**
   - **The third conjunct stays.** Write it through the Site's own function: the check reads an honest lemma conjunct by conjunct, and refuses a conjunct that applies no definition under `meaning`. Check that `(S.placesIn (S.certifiersOf "node0" ∪ ∅)).Nonempty` meets that as you have it, and say in your next line if it does not.
   - **The statement stays in place in the proofs tree**, as `TableReadBinding_honest` is.
   - **Its log cannot be the proofs tree's `Registration.Example`.** A theorem under `guarantees` reads no definition of a proofs-tree module, wherever its own text is (`trees.reads`). `Proofs.Core.Site` is not on `trees`' `except`, and that list takes no new entry.
   - **So the example moves, and only the part your statement reads.** NET-440's pull request moves `onsite` and what it is built from, under the same names, into a definitions file beside `security/Definitions/Core/Site.lean`. `security/Proofs/Core/Site.lean` imports it and keeps the rest of `Example` and every theorem. No second log. The file is lean's, lean has said yes, and lean reads the move with the diff.
   - **Make the move after lean's names move is on `main`**, not before Tue 17 Nov. That move renames `kind := .device` in `security/Proofs/Core/Site.lean` by file, and a line already in another file would miss it. Look for it on `main` before you cut the move; I tell you if I see it first.

What this does to your days: nothing in NET-438. For NET-440, write the lemma and its proof against the names as they are, and make the move the pull request's last commit before it opens. Fri 20 Nov 6 PM PST stands unless you tell me by Wed 18 Nov noon PST that it cannot.

If your records run says otherwise than lean's reading, send me the line and I post it on TOP-398.

Standing rules, unchanged.

