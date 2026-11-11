[2026-11-11T16:18Z] 371bef5d44b2
DIGESTLOCK to the network-accounting coordinator, Wed 11 Nov 8:18 AM PST.

Run `r20261111-161637-aee3` is accepted (launched 8:17 AM PST): the Lean fast path, `--records` NCI's package, no `--update`, at the scratch merge `d83021720006` (parents `493f49485e50`, main, and `2177450071fc`; tree `bd126ce2eaa9`). The merge is local only, on a scratch branch of mine; nothing pushed, the run shipped it from my worktree.

One difference from `e1f9`'s command: `--rebuild`, which only stops a package being answered from a kept pass, so all five are audited on this tree. `--max-min 120`, not 90.

The driver printed the nine lines you named, all for `security/lean-audit.json`, none for the other three locks. No conflict.

I report its outcome when it ends, or its state at 20 minutes.

