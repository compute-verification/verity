# ARCHFILE: inbox (read at each hourly checkpoint)

`git fetch -q origin cursor/verity-top-4db4 && git show origin/cursor/verity-top-4db4:inbox/ARCHFILE.md`

## Thu 12 Nov 10:57 PM PST: Daniel changed the page himself; the export is replaced, and the tree of record changes with it

Daniel edited the page with his own hand between 10:25 and 10:50 PM PST and told top at 10:51 PM PST ("I did make a
few changes btw"). `inbox/ARCHFILE-page.md` on this branch is now sections 1 to 7 as the page stood at 10:50:33 PM PST
(06:50:33Z). It is the text of record. Your first copy (commit `0a2136a1304`, the 10:28 PM PST export) already held his
earlier edits ("challenges" where the page said "coins"; "registers its hardware, its circuit compiler and other
information (e.g. the model's weights)"). It lacks six places, which
`git diff 0a2136a1304 origin/cursor/verity-top-4db4 -- inbox/ARCHFILE-page.md` shows whole:

1. Section 2, idea 3: "Protocols come in two layers, and both call the proofs." The accounting layer (PoUW, PoUS and
   PoUB, the certifier), the policy layer on it (inference-only), the proofs module beside both (Flock, with SPOT and
   recursion on it), compose on top.
2. Section 4's table of parts: the paths are `protocols/proofs/flock`, `protocols/proofs/spot`,
   `protocols/proofs/recursion`, `protocols/accounting/pouw`, `protocols/accounting/pous` and `poub`,
   `protocols/accounting/certifier`, `protocols/policy/inference_only`.
3. Section 5's diagram of the protocols (a new `flowchart LR` with the subgraphs accounting, policy and proofs).
4. Section 6's tree: under `protocols/` the lines `proofs/`, `accounting/`, `policy/`, `compose/`, `experimental/`.
5. Section 6's rule 7: "Inside `protocols/`: two layers ... Each group is a directory of its own: `proofs/`,
   `accounting/` and `policy/`."
6. Section 7's rows: `protocols/accounting/pouw/prover/`, `protocols/accounting/pouw/`, and "Each accounting protocol"
   where the row said "Each resource protocol".

What this changes in your task:

- **A. The file.** `ARCHITECTURE.md` is written from the new export. Apply the six places to what you have; nothing else
  in sections 1 to 7 changed. Do not write "coin" where the page says "challenge", and do not write "resource protocol".
- **B. The tree of record, in the same PR.** `tests/tree.toml` (on main since #9539) says `verity/protocols/` is "one
  directory per protocol, the list flat". Daniel's page now says otherwise, and your `END_STATE_DOCS` check needs the new
  paths in the tree file, so the PR that brings the file also changes the tree. top asks for it (the header's "not
  changed here without asking top" is met). What the tree file and `tests/test_tree.py` are to say:
  - `verity/protocols/` means the protocols in three groups, with `compose/` and `experimental/` beside them.
  - Three new `[[tree]]` entries, each still to come (`owner = "top"`, `day = "2026-11-15"`):
    `verity/protocols/proofs/` (flock, spot and recursion: what the other protocols call to prove),
    `verity/protocols/accounting/` (pouw, pous, poub and certifier: compute, memory and network),
    `verity/protocols/policy/` (inference_only, on accounting/).
  - The eight protocols' entries name the grouped place (`verity/protocols/proofs/flock/` and so on) and keep
    `protocol = true`. `verity/protocols/compose/` and `verity/protocols/experimental/` stay where they are.
  - Today's eight flat directories (`verity/protocols/flock/` ... `verity/protocols/inference_only/`) become
    `[[not_yet]]` lines, each with `to` its grouped place, `owner = "top"`, `day = "2026-11-15"`. `NOT_YET_MAX` rises by
    exactly those eight. The list otherwise only shrinks; say in the PR body that this one rise is Daniel's change of
    the shape, by top's word.
  - The test keeps checking the inside of each protocol's directory at the place main holds it: flat today, grouped
    once the move lands. A stray inside either place still fails, and nothing is checked twice. The lines already on the
    not-yet list for paths inside a flat directory (`verity/protocols/pouw/clear/` and the like) keep their `path`, and
    a `to` of theirs that names a protocol's directory names the grouped place.
  - The header comment and the test's docstring name the levels as they now are. How the file spells a protocol's
    inside (`verity/protocols/NAME/...` today) is yours to choose; keep the file readable for a lead who opens it once.
  - Run `tests/test_tree.py` and `tests/test_doc_paths.py` locally (each is under a minute), `python3 tests/test_tree.py
    HEAD` on a merge with `origin/main`, and the remote suites as your brief says. Main is `7f5205fe75e4` now.
- **C. Not yours.** No file moves (top's map worker moves the directories on Sunday's cut), no `GLOSSARY.md` row, no
  `AGENTS.md` change, no import-rule test. If the lane-text lint or another test refuses a sentence of the page, bring
  the sentence to top in your report instead of rewording Daniel's text.
- **D. The hour.** The PR is open, unmarked, by **Fri 13 Nov 3:00 AM PST** (it was 1:30 AM). If the tree change is not
  done by 2:30 AM PST, open the PR with what stands and say in the body exactly what is missing.

## Thu 12 Nov 11:14 PM PST: two more places, both in section 7 (the export is replaced again)

`git diff bd4e3d7a0d1 origin/cursor/verity-top-4db4 -- inbox/ARCHFILE-page.md` shows them whole:

7. Daniel asked for one term for the auditor's hardware, and it is "device": the certifier's row of section 7 now reads
   "the auditor's device runs them from there" where it said "the auditor's host". Use "device", never "host", for the
   auditor's hardware anywhere in the file. "Host" for a developer's server stays as the page has it.
8. A new second row in the table "Files that only change place": "The eight protocols in their three groups", flat today,
   at the end `verity/protocols/proofs/`, `accounting/` and `policy/`. In the file it loses its Who and Day cells like
   the other rows.

Daniel also made the structure tonight's top priority at 11:03 PM PST. Nothing else changes for you: your PR is still
open, unmarked, by Fri 13 Nov 3:00 AM PST, and sooner is better.

## Thu 12 Nov 11:21 PM PST: place 9, `registration` is a part of its own (the export is replaced a third time)

Daniel, 11:15 PM PST, on top's proposal of a part that checks what was registered: "makes sense". The page has it now.
`git diff 7f752e97b8d origin/cursor/verity-top-4db4 -- inbox/ARCHFILE-page.md` shows the seven changed places whole:

9. Section 4 has a new row "Registration `verity/registration`" above Compose, and Compose's row no longer holds the
   registration check (its "takes" cell reads "Registration's outcome and each protocol's."). Section 5's first diagram
   has a `registration/` node inside `verity/` with three arrows (protocols to registration; registration to
   protocol.py and to commitments). Section 6's tree has a `registration/` line under `runtime/`, and rule 4 gains a
   sentence on what `registration/` uses. Section 7's row "The registration check" ends at `verity/registration/`.

What it adds to your tasks:

- The file follows the export, as before.
- `tests/tree.toml`: one more `[[tree]]` entry, `verity/registration/`, with the page's meaning, `owner = "proofs"`
  and `day = "2026-11-20"`. Say in the PR body that the day is top's until proofs gives its own on TOP-415 (the page
  says "not dated", and the tree file needs a day for an entry `main` doesn't hold). If a `[[not_yet]]` line today
  sends the registration check to `verity/protocols/compose/verifier/`, its `to` becomes `verity/registration/`.
- If the import-rule sentence of rule 4 is checked by a test that reads the file, leave that test alone and say so.

The hour does not move: PR open, unmarked, by Fri 13 Nov 3:00 AM PST.
