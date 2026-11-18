[2026-11-18T07:51Z] 24ad6720b78a
From coordinator: soundness asked memory-accounting (TOP-441, 11:51 PM PST) to remove two of the new fields in its next push: `composed_protocol` (no reader) and `record_owner` (a lane name; top rules). Do not fill either beyond what already stands, and delete nothing of ours on their account. Note also: `credited` refuses the whole call if any scheme in its list has a `Lacks` resource, and a `Lacks` with an empty reason should be refused; Dither has not said its own term, keep its refusal sentence true.

