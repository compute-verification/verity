[2026-11-10T20:27Z] 7f294ef0d4df
CROSS to network-accounting, NET-311, #8477's mend: nine tenths done, one Tap site needs your word. 12:27 PM PST.

Head now: `cursor/net311-main-8477-a9af` 6b4c70fc5b39e68d7e88fd41108728525040e367, three commits on 831156e972 (= main 5482988e3496f12fe6683ccf355583a486cf4246 merged into 2986a0cdb3). Pushed. Only `warden_term_map.toml` and `warden_tokens_map.toml` differ from 96de5d04f2's maps [run].

1. WARDEN: 2a fails, 2b taken. With the row in `warden_map.toml`'s `[words]` the names step's `move_check.py` exits 1: "28 paths are not part of a pure move", every one "edit beyond moved paths and module names" (19 .py, 6 .sh, 3 .md) [run]. 2b: 32 rows, each a constant whole, and 14 rows for the name `WARDEN` by its written forms. Staged on 831156e972, the tokens step's diff is byte for byte the one the bare row gave: 362,360 bytes, 271 sites, 88 files; rerun changes 0 files [run].
2. The 14 form rows, for you to judge (I do not call them a trick, each is a form the name is written in, but 10 are keyed on syntax, not a name): `corpus.WARDEN`, `chain_record.WARDEN`, `CR.WARDEN`, `cr.WARDEN`; then `\`WARDEN\``, `WARDEN =`, `WARDEN.get`, `in WARDEN`, `== WARDEN`, `A5-WARDEN`; and four for a shell variable of the same spelling in `chain_record.sh` (19 sites, not the constant): `WARDEN=`, `$WARDEN`, `${WARDEN`, `export WARDEN` [run: the staged counts].
3. Tap: nine of the ten sites have a longer phrase (5 rows: "the warden and", "the warden sent", "its warden sent", "the warden's cells", "the warden's code"). The term step's diff of the two Tap files is identical before and after, and the step still certifies [run].
4. BLOCKED, the tenth site: `Security/Definitions/NetworkCertifier/Tap.lean` line 8, "Definitions only. The warden (\`Definitions.NetworkCertifier.Warden\`) and the network certifier are one device". No longer phrase is prose there. Left of it the sentence ends, and `prose_word` refuses a phrase holding "." [read, and run: "only. The warden" renames nothing]. Right of it is "(" and then a name in backticks, and `prose_word` refuses a match whose right neighbour is "(" or a backtick [read, and run: "The warden (" renames nothing]. Line 7 is blank, so no phrase reaches it from the heading [read].
5. So the head keeps that one bare row, `"The warden" = "The relay"`: the ten sites read "relay" exactly as before, and `test_maps.py` is red on 2 second names, down from 7, both 'The warden' against the tokens map's "the warden" and "The warden" [run]. Dropping the row gives 0 second names [run] but the site then reads "The network certifier (…) and the network certifier are one device", which is false, so I did not.

Ways out I see, none of which I take unasked:
 (a) the test's owner allows this one file-scoped phrase;
 (b) `rename.py`'s `prose_word` lets a phrase hold a sentence's end (". " between two words), and the row becomes `"only. The warden"`: a tool change in tools/move/ with its tests, not a map;
 (c) the Lean docstring's sentence is reworded so the word has a prose neighbour: a file outside tools/move/;
 (d) it stays red on this key.

The two suites are running at 6b4c70fc5b; I expect `tools/move` red on that one test and report it as it comes. A ruling that reaches me by 1:45 PM PST I apply and rerun; else I send what stands by 2:30.

