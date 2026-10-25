import Mathlib.Logic.Embedding.Basic

/-!
# A deterministic schedule: the vocabulary (trusted)

`Accounting.Assumptions.SteeredLayout layout steer` says a pass's layout reads the codec and the append coins `y` only
through an `a`-bit summary. It holds of every layout at `a = |Y|` (the summary is `y`), so it bounds nothing until
something counts `a`. `ScheduleDeterministic` is what counts it: the layout at each prefix of the session's coins is
the verifier's own plan (`canon`) of a schedule whose encoding (`enc`, the verifier's too) has at most `a` bits, and
the encoding is one-to-one. The prover may choose the schedule after the coins, but every layout it can then steer to
is the plan of one of fewer than `2^(a+1)` schedules. `a` is one number for every coin: the cap on the schedule's bits
the prover commits before the coins, which the verifier holds the schedule to, and never `|Y|` for free. The bits of
the one schedule published after the coins are not such an `a`, since their length can follow the coins.

The code's check is `RemoteAudit.Schedule.check`
(`verity/core/protocols/remote_bandwidth/lean/RemoteAudit/Schedule.lean`): it refuses a session whose passes' plans
are not its published schedule's (`Schedule.plan`) or whose schedule is over the committed cap, and returns the
schedule's bits.
`Properties.RemoteAudit.scheduleDeterministic_of_checked` derives `ScheduleDeterministic` from its acceptance, and
`Properties.RemoteAudit.steered_of_scheduleDeterministic` derives `SteeredLayout` at `a + 1` bits from it
(`Security/Proofs/Properties/RemoteAudit/Schedule.lean`, which also holds a layout that is `SteeredLayout` at one bit
and not `ScheduleDeterministic` at any `a`).
-/

namespace Accounting

/-- **Each pass's layout is the verifier's plan of a schedule of at most `a` bits.** `S` is the schedules, `canon c s`
the layout the verifier's code computes from schedule `s` for the pass at prefix `c` of the session's coins, and
`enc s` the schedule's encoding, as bits. Both are the verifier's, fixed before the coins: the prover chooses only
the schedule. The encoding is one-to-one, and at every prefix `c` and every codec and append coins `y` the layout
`layout c y` is `canon c s` for a schedule `s` whose encoding has at most `a` bits. -/
def ScheduleDeterministic {Pfx Y S : Type} {N T M : ℕ} (canon : Pfx → S → Fin N ↪ Fin T × Fin M)
    (enc : S → List Bool) (a : ℕ) (layout : Pfx → Y → Fin N ↪ Fin T × Fin M) : Prop :=
  Function.Injective enc ∧ ∀ c y, ∃ s, (enc s).length ≤ a ∧ layout c y = canon c s

end Accounting
