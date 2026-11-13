Lane proofs

Goal proofs/sampled-audits-sound

One sentence of `Properties.Spot.R8`'s docstring, and nothing else. It named `Properties.NetworkPeriod.RowsTiedSampled`, which "has that audit sample every unit". The tree no longer has that statement: the network period check's rows form accepts a tied audit at any sample and charges one sampled in part.

**The sentence now.** "`Properties.NetworkPeriod.RowsProgWired` equates the scan's `scan/prog` entry's root and domain with an audit's entries port. It accepts that audit at any sample and charges one sampled in part: `Properties.NetworkPeriod.RowsTiedCharged` has its δ in the period's ε and, where its bound is at least one incorrect proof block, every slot entry bit of each link-window whose scan's `prog` is its output counted as faulty."

- **What it is read from**: `RowsTiedCharged` and `RowsTiedChargedAt` in `security/Properties/NetworkPeriod.lean`.
- **The sentence after it is as it was**: that the rows under that root are the outputs of the proof blocks that computed them is not composed with those into a theorem, unproved.
- **No code, no statement and no lock record changes.** `R8`'s term is untouched. The lock records a definition it reads by the hash of its type and value terms, not its docstring, and holds no line numbers. That is read from the lock's code, not shown by a records run.

## Checks

CHECKS

Asked for on LEAN-249.
