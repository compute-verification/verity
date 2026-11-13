Lane proofs

Goal proofs/sampled-audits-sound

One sentence of `Properties.Spot.R8`'s docstring, and nothing else. It named `Properties.NetworkPeriod.RowsTiedSampled`, which "has that audit sample every unit". The tree no longer has that statement: the network period check's rows form accepts a tied audit at any sample and charges one sampled in part.

**The sentence now.** "`Properties.NetworkPeriod.RowsProgWired` equates the scan's `scan/prog` entry's root and domain with an audit's entries port. It accepts that audit at any sample and charges one sampled in part: `Properties.NetworkPeriod.RowsTiedCharged` has its δ in the period's ε and, where its bound is at least one incorrect proof block, every slot entry bit of each link-window whose scan's `prog` is its output counted as faulty."

- **What it is read from**: `RowsTiedCharged` and `RowsTiedChargedAt` in `security/Properties/NetworkPeriod.lean`.
- **The sentence after it is as it was**: that the rows under that root are the outputs of the proof blocks that computed them is not composed with those into a theorem, unproved.
- **No code and no statement changes.** `R8`'s term is untouched.
- **No lock record moves, by reading**: the lock hashes a read definition's type and value, not its docstring, and holds no line numbers. No records run was made for it; the train's records check is the run.

## Checks

- **The file elaborates**: `r20261113-042733-cae0` on vy-nebius-cpu-11, `security/Properties/Spot.lean`: `lean exit 0`; it holds no theorem, so there is no axiom probe.
- **On the author's machine**: `tests/lint/test_lean_bar.py`, 8 passed; `tests/lint/test_repo_retired_terms.py`, 14 passed, and the file is not among its warnings.
- **No records run**, as above.

Asked for on LEAN-249.
