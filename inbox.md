[2026-11-18T03:23Z] 51f13bf98742
Two additions for the `interfaces` branch. (1) Rule 11: a commit that deletes an old path carries two lines in its message, `Deletes: WHAT` and `Replaced-by: WHAT`. (2) `verity/protocols/resource_accounting/scheme.py` is on `origin/interfaces` since merge `d9daaf2bfd8`: merge `origin/interfaces` and build against that copy. Its hook names (`linear`, `kv`, ...) will be renamed in verity words tonight ("a block is about to be computed"): use the names as they are on the branch when you push, and say in your return where you name a hook so a rename finds it.

