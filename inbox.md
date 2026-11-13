[2026-11-13T01:44Z] 1196fd65ba7d
From the proofs coordinator, on your 01:34Z entry. Good work: `heldOf`, `HeldOfFiles` and `HeldOfCheck` compiled, and the finding on the size bound is the kind I need early.

**FIRST, one short entry by 02:15Z that settles the finding with numbers, before you change anything:**
1. Which is it: is `two h` itself `none` at a real honest run (the `scopeOk` conjunct of `two` fails), or does only the cheap route fail (`descOk`'s `memberOk` is too strong, while `scopeOk` on the unrolled programs would still hold)? Your entry says both in different places. If `two h` is `none`, the proved verifier refuses Friday's own run and the two-stage theorem says nothing about it; I tell Daniel that at 7 PM PST, so I need it exact.
2. Friday's numbers: one class of 8 requests, 64 tokens each, 20 templates (19 chunks and the combine), so n = 512 inner blocks a template if n counts every inner block of the run; say what n counts. `B` and `R` for a chunk and for the combine at V = 154,880, from the second try's records (GATHER's and HONEST2's entries cite them, run `92eb`), and `2 + n·B + 3·n·R` against 262,144 for each.
3. Whether HONEST2's instance passes `two` only because it is small.

**RULINGS:**
- **The per-block bound (your NEEDS 1) is yours.** You found it and you have read the files; FRAME is full. Restate `MemberFits`' size bound per inner block and carry it through `partitions_keyProg`, `scopeOk_members` and `memberOk`. Work on a branch of your own cut from FRAME's L1 head (`cursor/proofs-frame-l1-9c4e`, which moved the wiring's definitions into `Code.lean`), touching `ScopeOf.lean`, `Bridge/Members.lean` and `ScopeDesc.lean`. FRAME is told not to edit those definitions.
- **If a definition that `two` reads changes** (so the model's verifier accepts more), write the before and after in your entry as a STOP RULE table: the lock records of `SpotKSTwo` and `SpotExfilTwo` move and it joins FRAME's notice. Daniel has said yes to such changes; I still list each one for him.
- **Order:** the entry above; the per-block bound; `Built` from the parse (your NEEDS 2, yours); then (c), the verifier-side per-template check and its lemma, with TWOLEG's line `if !heldOkDesc p bs i then throw HELD_WHY` after it. `heldOf` deciding `descOk` and `templateOk` in place of `two`: yes.
- **`RefReadsWf h` at `heldOf`** (the one premise of OWNREADS's `ownReads_of`) is OWNREADS's, on your compiled files. Do not write it.
- **The date:** Daniel's rule is nothing assumed by Fri 6 PM PST (Sat 02:00Z), not noon. Say in the same entry your hour for each of the three with that as the limit, and what stays a named premise at that hour if any does.

Machines: vy-nebius-cpu-12 as you have been. An entry by 02:15Z for the numbers, then by 03:00Z whose first line says what has compiled.

