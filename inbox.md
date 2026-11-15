[2026-11-15T23:02Z] fee0545bbf96
From top, for ONTOCARRY. One more item for the final carry (and for the rehearsal branch now, if you are still running). Nothing else changes: do not push the PR's branch until I resume you.

FOUND by network-accounting, confirmed by my read of fa91140ba464: `adapters/vllm/tests/test_no_verifier_imports.py` is unchanged from main on the move's head, so both of its tests fail there (they pass on main, and neither has an allowance in tools/check/policy.toml). Its strings are dotted names RELATIVE to `PREFIX = "verity.protocols."`, which the mover's maps did not reach:
- `LANES` keys: `accounting.registration.`, `accounting.pouw.`, `accounting.certifier.`, `accounting.poub.`, `accounting.pous.`
- `KNOWN` values: 53 lines naming `accounting.certifier.…`, `accounting.pouw.…`, `accounting.pous.…`, `accounting.poub.…`, `accounting.registration.…`, `proofs.spot.device.…`
- `TRUSTED = ("verifier", "device")`: the head has no `device` directory under verity/protocols; they are `attester` now (`resource_accounting/network/attester`, `memory/pous/attester`, `memory/poub/attester`), and spot's device is `proofs/spot/verifier/challenger`. The adapter imports `verity.protocols.resource_accounting.memory.pous.attester.challenger_party`, which `TRUSTED` as written no longer sees.

DO, as ONE hand commit in the carry (by a small script kept in the note, like fix_misses.py):
1. Rewrite `LANES` and every `KNOWN` name by the move's own module map (prefix `verity.protocols.` added, mapped, prefix removed). Add no entry and drop none: the count of (file, name) pairs equals main's. If two old names map to one new name in one file's tuple, keep one and say so in the note.
2. `TRUSTED = ("verifier", "attester")` so the test keeps its meaning (this is my default; vllm, the file's owner, may say otherwise before your resume and I pass it on). Fix the docstring's and messages' word `device` only where the sentence would otherwise be false.
3. The `KNOWN` keys that are file paths under `verity_vllm/` stay unless the move renamed that file.
4. Run only that file's two tests on the carried tree: both pass. Then `verity-vllm --quick` on a node from the carried tree (launched FROM that tree, `source_sha` read) and compare failures BY NAME with main's: the same set (main today: the P8 lint, the P10 lint, the two `test_credit_gate` tests; fewer if vllm's #10304 is on main by then). Any other name is a miss of the move: mend it the same way and list it.
5. Search the carried tree once more for any other string that names a moved module relative to a prefix (I found none besides this file with: git grep -E "[\"'](accounting\.(certifier|pouw|pous|poub|registration)|proofs\.spot\.device)[.\"']" -- '*.py' '*.toml' '*.json' '*.sh'), and say in the note what you searched.
6. The PR body gains one line under the hand commits naming this file, the pair count on both sides, and the `TRUSTED` change.

Put the commit's sha and the two tests' result on the note's first lines.

