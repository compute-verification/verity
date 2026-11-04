module

public import Definitions.Flock.ZK.RealView
public import Definitions.Flock.ZK.AnyValues

/-!
# The ZK server's release with real hm96 leaves, at any values

`Table.release` and `Session.releaseAt` have every leaf nobody opens ideal. Here every leaf is real, on both branches of
the server's check: where it passes, the view's hidden leaves are hm96 leaves under fresh salts (`Hm96.viewR`); where it
fails, the same run's pre-final messages leave (`Hm96.preR`), and every one of their leaves is an hm96 leaf of what it
commits, `τ`'s commitment included, since nothing is opened.

* `Table.PreLeaves`: every leaf of the pre-final messages, the ones the final message opens first.
* `Table.preOf`: a view's pre-final messages. `Hm96.preR`: the real prover's, `preOf` of its run with real leaves.
* `Hm96.TauLeaf`: `τ`'s commitment is an hm96 leaf.
* `Hm96.releaseR`: one table's release behind the check, with real leaves.
* `Session.releaseR`, `Session.simR`: a session's release at an outcome, and its simulator, with real leaves;
  `Session.unopened`: how many leaves the release at an outcome leaves unopened.
* `Session.WitnessesFirst`: the witnesses read none of the server's draws for the session.

`Flock.Guarantees.TableReleaseHm96` and `Flock.Guarantees.SessionReleaseHm96` (`Definitions/Flock/Guarantees/ZkAny.lean`)
bound these releases by their simulators within `2·δ₁` a leaf left unopened.
-/

@[expose] public section

namespace FlockSoundness.ZK

namespace Table

variable {F K W Salt B D Hi VL : Type} [Field F] [Field K] [AddCommGroup B] [CharP F 2]
  (T : Table F K W Salt (B × Salt) D Hi VL)

/-- **Every leaf of the pre-final messages**: those the final message opens (level 0's at the opened positions, each
rep's pads tree's at its opened positions, and each rep's commitment to `τ`), then those nobody opens (`Hid`). -/
abbrev PreLeaves := (Fin T.nS ⊕ (Fin 2 × Fin T.nQ) ⊕ Fin 2) ⊕ T.Hid

/-- **A view's pre-final messages**: everything before the final message, with the leaves the view has. -/
def preOf (v : T.View) : T.PreView :=
  { sHat := v.sHat, sReg := v.sReg, masked := v.masked, y1 := v.y1, ybar := v.ybar, lig := v.lig, inner := v.inner
    Y1 := v.Y1, Y2 := v.Y2, leaves0 := v.leaves0, tauCom := v.tauCom, leavesP := v.leavesP }

variable {T} {C : Type}

/-- **The real prover's pre-final messages**: the pre-final part of its run with real leaves (`viewR`). Every leaf is an
hm96 leaf of what it commits: level 0's and the pads trees' at the opened positions under their salts, the others under
fresh salts, and `τ`'s commitment under `y_τ`. -/
noncomputable def Hm96.preR (M : T.Hm96 C) (w : W) (ω : T.Rand) (ys : T.Hid → Salt) : T.PreView :=
  T.preOf (M.viewR w ω ys)

/-- **`τ`'s commitment is an hm96 leaf** of `τ`'s digest `digT τ`, as level 0's and the pads trees' leaves are. -/
def Hm96.TauLeaf (M : T.Hm96 C) (digT : F → B) : Prop :=
  ∀ t y, T.leafT t y = M.H (digT t + M.My y, M.cs y)

/-- **What the ZK server releases behind the check `go`, with real leaves**: the view where `go` passes, the same run's
pre-final messages where it doesn't. -/
noncomputable def Hm96.releaseR (M : T.Hm96 C) (go : T.Rand → Bool) (w : W) (ω : T.Rand) (ys : T.Hid → Salt) :
    T.View ⊕ T.PreView :=
  if go ω then .inl (M.viewR w ω ys) else .inr (M.preR w ω ys)

end Table

namespace Session

variable {F K W Salt B D Hi VL : Type} [Field F] [Field K] [AddCommGroup B] [CharP F 2] {J : ℕ}
  (T : Fin J → Table F K W Salt (B × Salt) D Hi VL) {C : Type} (M : ∀ j, (T j).Hm96 C) {St : Type}

/-- **What the ZK server releases from a session at the outcome `s`, with real leaves**: table `j`'s view where its file
left (`full s j`), the same run's pre-final messages elsewhere, each on its table's randomness and fresh salts. -/
noncomputable def releaseR (full : St → Fin J → Bool) (w : Fin J → W) (s : St)
    (p : ∀ j, (T j).Rand × ((T j).Hid → Salt)) : ∀ j, (T j).View ⊕ (T j).PreView :=
  fun j => if full s j then .inl ((M j).viewR (w j) (p j).1 (p j).2) else .inr ((M j).preR (w j) (p j).1 (p j).2)

/-- **The simulator at the outcome `s`, with real leaves**: `S_shvzk(x, e)` on the dummy with real leaves where table
`j`'s file left, the dummy's real pre-final messages elsewhere, each on its own randomness. -/
noncomputable def simR (full : St → Fin J → Bool) (w₀ : Fin J → W) (s : St)
    (σ : ∀ j, ((T j).SimRand × ((T j).Hid → Salt)) × ((T j).Rand × ((T j).Hid → Salt))) :
    ∀ j, (T j).View ⊕ (T j).PreView :=
  fun j => if full s j then .inl ((M j).simR (w₀ j) (σ j).1.1 (σ j).1.2)
    else .inr ((M j).preR (w₀ j) (σ j).2.1 (σ j).2.2)

/-- **The leaves the release at `s` leaves unopened**: a table's hidden leaves where its file left, every leaf of its
pre-final messages elsewhere. -/
def unopened (full : St → Fin J → Bool) (s : St) : ℕ :=
  ∑ j, if full s j then Fintype.card (T j).Hid else Fintype.card (T j).PreLeaves

/-- **The witnesses are fixed before the server's randomness**: the session's witnesses, as a function `wit` of every
draw the server makes for it (each table's level-0 randomness, pads and salts), read none of those draws. A server that
answers `level0` only once every table's witness is in gives it. At `Table.onVals` the witness carries the worker's
values, so it fixes them too. -/
def WitnessesFirst {Ω : Type} (wit : Ω → Fin J → W) : Prop :=
  ∀ p p', wit p = wit p'

end Session

end FlockSoundness.ZK
