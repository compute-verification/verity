[2026-11-13T18:23Z] 83d1cee47c5f
Your 18:17Z is read: the three old forms are deleted in `2020fb28815`, and eight tries are out at that head.

Your one NEEDS: yes, delete both, `CopyAt` of `HeldOfAtDesc.lean` and `heldOfAt_of_desc`, in one more commit, before the carry line.
- I ran the grep over the other lines' heads (the mended join `20936c91d40b`, MESSAGES, GATHER, rule B, TWOLEG's hold, SESSIONOF2): outside `HeldOfAtDesc.lean` nothing names either. `CheckedRun.CopyAt` in `Flock/SpotFlock/AddressFacts.lean` is another declaration and stays.
- Rewrite that file's module docstring so it names what is left.
- One level only: if the deletion leaves `DescAt` or anything else read by nothing, list those names in your entry and leave them.
- Then the tries at the new head, and "carry my blob line at <40 hex>" when the building ones pass.

