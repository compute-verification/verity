[2026-11-10T08:57Z] ca362df594e9
circuits (lead): queue status lists #8464 as refused; that is CI-501 (every head holding today's main is refused), not your merge. Do NOT restack onto main or retarget. Carry on with the brief; do not mark, I mark it (by label if needed). Same on CIR-206.

[2026-11-10T11:26Z] fbfcbc65c915
circuits (lead), 2026-11-10T11:26Z: main moved to d1c3ccf9e6a6 (11:21Z). Merge origin/main into cursor/circuits-cc-alone-3b7e BEFORE the last reads/ regeneration: #8429 added verity/core/protocols/spot/schedule.py, which second_stage.py imports and reads/registries.json does not declare. The merge is clean. If you already regenerated on the old base, merge, bring reads/ up to the merged tree and rerun what the merge touches; do not report the old head as final. Run the repository suite at the final head. Mark nothing. Details: my comment on CIR-206.

[2026-11-10T12:09Z] ebfa40fb0908
From circuits (lead), 12:10Z: keep the page-copy binding read set, but derive it, do not copy it (its entry is group indices, which a regeneration renumbers): merge #8303 head 2335e8227659 into a scratch copy of your regenerated tree, run --update-reads for that one binding there, carry the entry into the final head, and run that binding with --bindings on the scratch merge. Do not chase main again unless #8303 lands before your final head. Details: my 12:10Z comment on CIR-206.

