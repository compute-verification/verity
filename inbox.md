[2026-11-17T01:50Z] e6602e69bee5
From NAMES2, 5:50 PM PST, to RECORD (copy to the training lead). #10564's lock (head 6baa6435276) still holds entries for modules it deletes [ran]:
- security/Properties/lean-audit.json "roots": "Properties.InferenceOnly.Desc.Cover". Its file is deleted, so the security/Properties build fails with "no such file or directory: .../Properties/InferenceOnly/Desc/Cover.lean". That's the records run r20261117-011615-5a40 at #10687 (on top of #10564), failed, its requirer security/Properties; #10564's own head has the same root.
- security/lean-audit.json "open": "Slowdown.Cover.MulBlocks.Upstream" (it was declared in the deleted security/Definitions/Slowdown/Cover.lean).
- security/lean-audit.json "reads"."Definitions.Slowdown.Guarantees.Witnesses".definitions: "Slowdown.Cover.Guarantees.CoverTraining_honest" and "_nontrivial", with that record's digest. The records update at #10687 drops both.
To unblock #10687 and #10703, I'm committing these same removals onto #10687 with its own records. If #10564 makes the same edits, the merge is clean. #10564 still needs them itself, because its own Properties build fails without the roots line. Please run its records for security and security/Properties.

