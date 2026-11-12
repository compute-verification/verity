[2026-11-12T02:38Z] b5a329d30628
FORM to training, TRN-103, Wed 11 Nov 6:38 PM PST (02:38Z). THE RECORDS FAILED AT 7bc9188630: TWO MODULES OF MINE DO NOT BUILD OVER MAIN. I AM MENDING THEM IN MY FILES. No one line yet.

r20261112-015300-82dc [RAN: research fetch --all, read to its last line]: "AUDIT: FAIL ... security/Proofs", build exit 1 in 1528 s; "Some required targets logged failures: Proofs.Properties.Nci.RecordUnits, Proofs.Properties.Nci.DecodeKeys". 18 of our modules were not attempted because they import one of the two (RecordComplies among them). The other 60 modules of the proofs package's Nci area built, lean's `Record.lean` among them.

1. `DecodeKeys.lean:33-34`, a moved import [READ]: `open private getDef ... from Flock.Program`. On main `Flock.Program` is a five-line re-export and the decoder's private names are in the module `Circuits.Program`. Mend: `from Circuits.Program`, as main's own `BodiesOk/` files have it. One line.

2. `RecordUnits.lean:88` and `:93`, NOT a moved import [READ]: main's commit 9725922ca6 ("Template query: an input node is a primitive with an input id, one definition") changed `Flock.TemplateQuery.evaluate` itself. It skipped a root node where `isInput tid`; it now skips where `d.isInputNode x`, which is `isInput id && (d.def? id).any (·.prim)`. So `evaluate_spec` as I state it (the ranges are the root's nodes whose callee's id is no input id) is false on main for a composite keyed as an input id. `Spot.Slots.inputGates` changed the same way; none of our files reads it [RAN: rg].
   What I do, unless you say otherwise: `evaluate_spec` states the query's own rule; `ranges_spec` keeps its conclusion and takes one more hypothesis, that a root node whose callee's key is an input id is one the query reads as an input node, proved for a decoded Program from `DecodeKeys.decode_keys` (keys apart) and the check's own refusal `keyedAsInput P = false`, both of which `ReadProg` already holds. `RecordTemplates.lean:96-122` unfolds `evaluate` too and gets the same mend. `form_complies` and the four statements keep their text. No change to the check or to its difftest.
   The check's refusal of a root composite keyed as an input id (`keyedAsInput`) is now stricter than the query needs. I leave it: dropping it is a change to the check, not this merge.

r20261112-015300-0faa at 7bc9188630 [RAN, read to its last line "PACKAGES-RC 0"]: `check.py --build`, kernel replay included, "AUDIT: PASS" for each of six packages: the check's (1730 declarations in 17 modules), SPOT's (5630 in 50), the certifier's verifier (1502 in 10, 4 guarantees), its device (1101 in 11), remote (67 in 1), onsite (33 in 2). So #8686's `DefinitionDigest.lean` builds at its head, and `Flock.TemplateQuery` and `Flock.Sigma` resolve from `NciCheck/Record.lean` and `RecordText.lean`.

COUNTS and WIDTH read `ranges_spec` and `ReadProg` [INFERRED from their branches' imports, not re-read]: `ranges_spec` gains one hypothesis, `ReadProg` is unchanged.

WHEN: each try is a node run of minutes (`lean_changed.py --try FILE`), then the records again (25 min of build today). The one line is not before about 04:30Z (8:30 PM PST) [INFERRED]. The 7 AM PST push still holds if the mend builds in two or three tries. My VM was replaced again at 02:30Z; nothing unpushed was lost.

NOT CHECKED: whether the 18 modules behind the two build once these are mended; the suites; the proofs package's kernel replay.

