import FlockSoundness.Protocol.Discharge.ZkLink.TablesAt

/-!
# The coin server's draw at every draw of the live law, at `--zk`

`zk_session_sound` takes `DrawOkZK I dj mPts` at each of `Law.execOS`'s draws: the draw file `dj S` proves `S`'s units,
and setupH, under the `--zk` tags, sets up at `mPts` link points. The first half is what the record carries, so it is
custody; the second is flock-verify's own arithmetic. This file splits them.

* `drawOf law pop S`: the verifier's own draw of `S`, as `UnitDraw.ofJson` decodes it: its law and population, `S`'s
  units ascending, no closure.
* `DrawFileZK dj law pop`: every draw file `dj S` that `UnitDraw.ofJson` decodes decodes to `drawOf law pop S`. It is a
  property of the file the coin server records, not of flock-verify (DrawWords' `hdj`, with the law).
* `drawOkZK_of_exec`: from `DrawFileZK` at the live law's own stratified law, `DrawOkZK` at every draw of `Law.execOS`,
  at `mPtsOf c.g (∑ s, ks s)`. At a run that succeeds the draw has `∑ s, ks s` units (`execOS_draw_card`); at a run that
  reads past the source's bytes the draw is every unit (`execOS_draw_card_or`), and a draw file of every unit that
  setupH accepts passes `UnitDraw.ofJson`'s per-stratum count, so every stratum draws all its units and
  `∑ s, ks s = n`.
-/

namespace FlockSoundness.Discharge.ZkLink

open Flock Exec ZkExec Finset FlockSoundness.Refine


variable {n : ℕ}

/-- **The verifier's own draw of `S`** under `law` over `[0, pop)`, as `UnitDraw.ofJson` decodes it: `S`'s units,
ascending, and no closure. -/
def drawOf (law : Draw.Law) (pop : ℕ) (S : Finset (Fin n)) : Draw.UnitDraw :=
  { law, population := pop, units := ((S.image Fin.val).sort (· ≤ ·)).toArray }

/-- **The record's draw file is the verifier's draw** (custody of the record's `unit_draw`): at every draw `S`, a draw
file `dj S` that `UnitDraw.ofJson` decodes decodes to the verifier's own draw of `S` under `law` over `[0, pop)`. -/
def DrawFileZK (dj : Finset (Fin n) → Lean.Json) (law : Draw.Law) (pop : ℕ) : Prop :=
  ∀ S D, Draw.UnitDraw.ofJson (dj S) = .ok D → D = drawOf law pop S

end FlockSoundness.Discharge.ZkLink
