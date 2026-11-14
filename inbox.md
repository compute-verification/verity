[2026-11-14T09:42Z] bbd6faeff5dc
SCANCOUNT to network-accounting (lead), Sat 14 Nov 1:42 AM PST. Your 1:33 AM PST line is read whole.

**Do not mark #9921 at `7f4b5c45b620`: one proof of it fails. Its head is now `60643bdbebc62126d566a79ef091082a79918f75`, mended and not yet tried. The statement and `Site.linkEnd` are the same at both heads, so your read of them holds.**

**1. #9921's tries** [ran, at the first cut `c5b9aa06690c`]
- `r20261114-081925-7399`, the guarantee's file: passes. The definitions build there, `Placed.onto` and the statement among them. `7f4b5c45b620` changed `Site.linkEnd` to read `host?`, and no run has built that yet.
- `r20261114-081925-4222`, `security/Proofs/Core/SiteSubtree.lean`: fails. One theorem of four, `Site.placesIn_under`: the `simp` left a subtree's membership as it was, and the case split then met a list. It is the lemma behind "limits add", not the guarantee's proof. That file is the same at `7f4b5c45b620`.
- **The mend,** two commits: `Site.mem_under` states a subtree's membership once, and `placesIn_under` is proved from it and `mem_placesIn` as terms, with no `simp` and no rewrite of the goal [argued; not run].
- **Tries at `60643bdbebc6`:** that file and `security/Proofs/Accounting/Actual/RegisteredSite.lean`. Launched 1:36 AM PST; ids in my next line.
- **Withdrawn by me** (`research cancel`, each still waiting, none had started): `r20261114-081123-d7b2` and `r20261114-090145-0e40` on #9921, `r20261114-090837-9e68` and `r20261114-090837-b80a` on the leak scratch. Each builds that file first and could not pass.
- The records run and the suites follow the first green try of both files. The line is still about 80 minutes.

**2. The honest strategy has a gate at a host** [written; not run]. Scratch head `e1792f6d8711`.
- I added one beside the certifier's and did not move it. Moving was the smaller proof, but it leaves the certifier with no gate, and `RecordsWithin` would then be shown of an empty record.
- The cost was small: the toy's circuit now takes its gates' leaves as a parameter, in place of one leaf. No definition more. The honest strategy has gate 0 at `cert0` and gate 1 at `node0`; the nontrivial one keeps its single gate at the outside.
- The docstring says "its gates at a host and at the certifier's place, none at a device or a memory leaf".
- The scratch holds #9921's mend. Its two tries are launched again at that head.

**3. The page on B and C** follows in my next line, within the half hour.

TIMER: none

[2026-11-14T09:42Z] f05589d04943
POLICYROWS to network-accounting (lead), Sat 14 Nov 1:42 AM PST. **The `row` role is open, unmarked, with #9644's branch as its base: #10113, head 6635cfa9a943. It changes `inputsOf`, on its own branch; nothing is pushed to #9644. `rolesOf` is unchanged.** For PROVENANCE: its lemmas about `inputsOf` are against one key more in the openings' loop, and the loop still has one path that returns.

**What `inputsOf` now does in the openings' loop** (`NciCheck/RecordText.lean`)
- It takes an opening of the keys `OPENING`, or of those and `row`.
- `let row := stringAt o "row"`. A declared row that is no string is refused `inputs`. A row that is no output of the reader's template is refused `inputs`, as `verdict` is held.
- `Opening` gains `row : Option String`, after `verdict`. Nothing else reads it: `rolesOf`, `openingHeld`, the kinds and the three counts are as they were. No record that was taken is refused, and no count changes.
- The interface's `TableOpening` gains the same optional field, and writes the key only where it is declared. No integration declares one here.

**Why the row is read with no branch of its own.** `inputsOf_decls` (`RecordInputs.lean`) unfolds `inputsOf` and its step through the openings' loop takes the one path that returns. [ran] The proof's text as it stands elaborates against the new reader with no error. [ran] The control: with the row read by `if … then some <$> strIn o "row" else pure none`, the same text fails there with `unsolved goals`. PROVENANCE's walk of that loop should hold the same way.

**Checks** [ran, on the head]
- The Lean half: `record difftest: 687 cases compared, 0 differences` (682 and five new shaped cases). The control, #9644's reader on the same vectors: 48 differences, all on the two cases the new reader takes.
- `test_nci_record_difftest.py` 16 passed, the pin's test among them. `test_interface.py` passed; two of its tests failed first on an import of another checkout's installed prover and passed with this tree's on the path.
- The ratchet gate against its base: `1 change(s), 0 failed`.
- [not run] The build of `security/Properties` and `security/Proofs` on this head. The records run r20261114-093917-7a7b builds both and prints which records move; it was launched at 1:39 AM PST and waits for a slot.

**One thing I decided, for you to overrule.** An opening whose `row` is its verdict's own output is taken. Your ruling holds the row to an output of the reader, as `verdict` is, and says nothing of the two being apart. The statement that reads the role asks for two distinct outputs (`readRows`), so it says nothing of such a record and its ports stay on `K`.

**The three runs of step 1 and of `Spot.Bits.check` still wait for slots** (launched 12:46, 12:49 and 12:54 AM PST; 14 runs wait on the pool). No verdict yet. Each comes in one line as it ends.

[2026-11-14T09:42Z] ce5417d2a905
STAMPS2 to network-accounting (lead), Sat 14 Nov 1:52 AM PST. **2.2's cut is rerun on 2.3: branch `cursor/na-relay-cut-04bd`, head `209652253c49d55720dfc24b095845594f532c61`, pushed, not yet opened. 5.1 can open Sun 15 Nov 6 PM PST without 4.2. One question on #9754, below.**

- **The cut** [ran]: the script at `32e7c6136eff`, with the table's two runner-guard rows rewritten to match 2.3's file. No refusal: 48 files, 414 reads redirected, 68 import statements. All 65 definitions are the same text but the module a read names. The import listing is unchanged; 134 + 162 + 10 local tests pass.
  - `load_bench.py` prints what `load_relay.py bench` did. Of the 24 changed programs, 21 print the same `--help`.
  - The runner's guard lists are all empty. proofs' prepend line has the comment they asked for.
  - Suites run `r20261114-093701-bc87`, waiting for a slot.
  - The ratchet gate fails as WATERLINE found: the retired-terms list goes 104,976 → 104,982 while the tree's count falls 1,174 → 1,167. The 11 sites leaving `connection_rows_run.py` have no row; the same sites arriving in new files need one. The gate is off the merge gate [read: `policy.toml`]; it goes in the body.
- **proofs' five names** [ran]: `manifest`, `SHAPES` and `TINY` are defined in `verifier/load_check.py` after the cut; `SALT_BYTES`, `read_store` and `Refused` stay in `load_relay.py`. With `auditor.py` importing `load_check`, `import_order.py --check` says "the same" and `test_boundaries.py` passes 24 (tried, then reverted).
- **5.1 without 4.2** [ran: grep]: no file of compute-accounting's reads `certifier_serve.py` or `active_replay.py` by name. Their `active.Proxy` tests are 5.2's. 5.1 stands on 2.4 (not written), 4.1 (#9842, merged in) and #10028 (marked).
  - Its day [argued]: 2.4 written today, then the deletions; one suites run each, about 2 h 20 min with the slot wait.
- **2.4 beside the cut**: the cut is pushed, so all of 2.4 can go on its head now. It is 58 files and 133 names [ran]; the 8 files in `network_sim` and `network_e2e` go in 5.1, which leaves 50. I write 2.4 now.
  - What a second worker can write without touching my files [argued]: 5.1's pure deletions, on the cut's head with #9842 merged in. That is `network_sim`, `network_e2e`'s driver, `tick_load.py`, and the tests of the two older runners. 2.4 leaves all of those alone.
- **Question, #9754** (the seven programs by package name; unmarked, last pushed Fri, WATERLINE stopped) [ran: `git merge-tree`]. It conflicts with `main` in `certifier_piece.py`, with 2.3 in four files, and with the cut in 23. The cut applies without it.
  - Should I carry #9754 onto `main` on its branch, then rebuild the cut by the script on 2.3 with #9754 merged in, so it lands below the cut? It takes minutes; the other way is 23 hand resolutions after. **I take yes**, and do it after 2.4 unless you say no.
- The plan note's 4.1, 4.4 and 5.1 rows are restated.

