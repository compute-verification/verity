[2026-11-13T02:39Z] 2d229035402e
FIELDS, Thu 18:39 PST

**I had already merged `main` into #9565 before your 6:25 PM message arrived: #9038 had landed, and your earlier order said to merge then and redo `field_leaf`. Its head is `fe31b503bb`. I'll merge `main` once more, and regenerate once, only after #9502 lands. Say if you want the merge undone instead; that needs a force push or a reverting commit.**

- READ (git): #9038's change to `field_leaf` was inside the branch for an empty return, which #9565 deletes, so this side's lines stand. No file on `main` reads either deleted definition (`rg`). The other three auto-merged proof files keep only this PR's edits.
- RAN r20261113-021149-636b: I cancelled it 3 minutes in, for the merge. It has no suite result and no `verity-vllm` test to name; its last line is the cancel. The suites now run at `fe31b503bb` as r20261113-021552-e198 (`--changed 49d12bdc36 --quick`, three-hour limit). If `verity-vllm` passes 55 minutes there, I'll name its last test.
- RAN r20261113-021149-517d (still running, at `66a3543f46`): `verity/protocols/spot/verifier/lean` PASS with records unchanged (0 guarantees), then `certifier/verifier/lean` PASS. It is now on `security` and `security/Properties`. The last lines are so far: "AUDIT: PASS; reports in .../requirer-verity_protocols_certifier_verifier_lean" and "check.py exit 0 in 33 s".
- RAN r20261113-021552-c5aa (running, at `fe31b503bb`): records for `verity/circuits/lean` and its requirers. It checks that the merged proofs build. I won't commit its records; they get regenerated once after #9502.
- #9565's body now lists each guarantee whose reads changed beside the definition: `ServedComplies` beside the six definitions (fieldsOf new and gone, `tmplOf`, `TemplateOf`, `outRow`, `OutLeaf`), and the 59 other guarantees reading `Circuits.Program` beside `Flock.Program.fieldsOf` (new), which is the one lock entry `reads/Circuits.Program`.
- READ: noted that #9561 is marked and admitted.

