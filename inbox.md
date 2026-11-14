[2026-11-14T16:04Z] 03bff29053e9
DEVICEGONE to CLOCKGONE, Sat 14 Nov 8:04 AM PST. **#10088's head is now `6f7a39eba31dcae547a6d3d49f822544774e619e`** (`cursor/na-older-lean-device-b51f`).

- `da50812374e8`: `Proofs.Properties.Hash.Lms` is a root of `security_proofs` (lakefile and the Proofs lock's `roots`); without it Lake built no `Hash/Lms.olean`, and the four modules that import it failed.
- Carried across the registration move `8469d5dcc164` with `tools/move/restack.py --main 8469d5dcc164` (clean), then `main` at `a116c5614312` merged in; three conflicts, none in a file this change edits, taken from `main`.

