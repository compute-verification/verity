import Proofs.Flock.Claims.Recursion.NonVacuous
import Proofs.Flock.Claims.Recursion.VerdictRowEDeployed

/-!
# Row E's target at the decision

`Top.TargetE` (`NonVacuous`) asks, at every decision where the rows are asked, for each rep's accumulator chain and
position with `UnitSums` and `ReadsEnds`, at accumulators `acc x ms cs` that can't depend on the decision. This file
fixes that `acc` and proves `TargetE` from the `rec-open` pieces' running sums.

* **The accumulators** (`accR`): `DecAt.accE` (`VerdictRowEDeployed`), the chain's rows at each rep's ends, at the
  decision. A decision is unique where it exists (`DecAt.eq`: its reading, setup and session are all equations), so
  `accR` is `D.accE` at every decision `D` (`accR_eq`). Every rep's `accR` has four elements (`accR_size`).
* **The target** (`targetE`): at `accR`, the chain `chainAt x ms i` from `r · N`. `ReadsEnds` is `readsEnds_accE`
  (row G from the dense inner statement, row C's sizes from the reading and the `shaped` base check, and row C's
  points `DecAt.points_eq`); `UnitSums` is `unitSums_of_pieces` at pieces that hold where `P` does.
-/

namespace RecDeploy

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP pointsOf)
open FlockVBridge.AccChain (UnitSums ReadsEnds)
open FlockSoundness.Discharge.Recursion (HClass)
open RecDeploy.Rows (DecAt nOpen accHolds)

namespace Rows.DecAt

variable {rd : Msg → Option FinalP} {x : Msg} {ms : List Msg} {cs : List Coin}

/-- **A decision is unique**: its reading (`dec`), its setup (`setup`) and its session (`decode`) are equations. -/
theorem eq (D D' : DecAt rd x ms cs) : D = D' := by
  obtain ⟨d, hd, spec, sts, hs, sess, hdec⟩ := D
  obtain ⟨d', hd', spec', sts', hs', sess', hdec'⟩ := D'
  obtain rfl : d = d' := Option.some.inj (hd.symm.trans hd')
  have e : (spec, sts) = (spec', sts') := Except.ok.inj (hs.symm.trans hs')
  simp only [Prod.mk.injEq] at e
  obtain ⟨rfl, rfl⟩ := e
  obtain rfl : sess = sess' := Except.ok.inj (hdec.symm.trans hdec')
  rfl

end Rows.DecAt

open Classical in
/-- **The reps' accumulators** at `x`, `ms`, `cs`: `DecAt.accE` at the decision, where there is one. -/
noncomputable def accR (rd : Msg → Option FinalP) (chainAt : Msg → List Msg → ℕ → ℕ → Array F128) (x : Msg)
    (ms : List Msg) (cs : List Coin) (k : ℕ) : Array F128 :=
  if h : Nonempty (DecAt rd x ms cs) then (Classical.choice h).accE chainAt k else #[0, 0, 0, 0]

theorem accR_eq {rd : Msg → Option FinalP} (chainAt : Msg → List Msg → ℕ → ℕ → Array F128) {x : Msg}
    {ms : List Msg} {cs : List Coin} (D : DecAt rd x ms cs) : accR rd chainAt x ms cs = D.accE chainAt := by
  funext k
  unfold accR
  rw [dif_pos ⟨D⟩, Rows.DecAt.eq (Classical.choice ⟨D⟩) D]

theorem accR_size (rd : Msg → Option FinalP) (chainAt : Msg → List Msg → ℕ → ℕ → Array F128) (x : Msg)
    (ms : List Msg) (cs : List Coin) (k : ℕ) : (accR rd chainAt x ms cs k).size = 4 := by
  unfold accR
  split
  · exact Rows.DecAt.accE_size _ chainAt k
  · rfl

namespace Top

/-- **Row E's target at `accR`**: each table's chain `chainAt x ms i`, rep `r` from `r · N`. `pieces x` are the
`rec-open` pieces, which hold their running sums (`accHolds`) where `P` does (`hholds`), read the decision's side
(`hside`) and the verifier's `coef` (`hcoef`, the coin tie), and cover every query of every level of each rep
(`hcover`). Row C's points are the first coin's (`DecAt.points_eq`). -/
theorem targetE (rd : Msg → Option FinalP) (P : Msg → List Msg → List Coin → Prop)
    (chainAt : Msg → List Msg → ℕ → ℕ → Array F128) (pieces : Msg → List OpenPiece)
    (sideAt : Msg → List Msg → OpenPiece → OpenSide) (coefAt : Msg → List Coin → OpenPiece → Array F128)
    (hholds : ∀ (x : Msg) (ms : List Msg) (cs : List Coin), P x ms cs →
      ∀ q ∈ pieces x, accHolds sideAt coefAt chainAt q x ms cs)
    (hside : ∀ (x : Msg) (ms : List Msg) (cs : List Coin), At rd P x ms cs →
      ∀ (D : DecAt rd x ms cs) (cls : HClass) (i : ℕ), i < D.sts.size → ∀ r : Fin 2,
        ∀ q ∈ pieces x, q.table = i → q.rep = r.val → sideAt x ms q = D.side cls i r)
    (hcoef : ∀ (x : Msg) (ms : List Msg) (cs : List Coin), At rd P x ms cs →
      ∀ (D : DecAt rd x ms cs) (cls : HClass) (i : ℕ), i < D.sts.size → ∀ r : Fin 2,
        ∀ q ∈ pieces x, q.table = i → q.rep = r.val →
          coefAt x cs q = D.coefAt cls (accR rd chainAt x ms cs) i r q.level q.query)
    (hcover : ∀ (x : Msg) (ms : List Msg) (cs : List Coin), At rd P x ms cs →
      ∀ (D : DecAt rd x ms cs) (cls : HClass) (i : ℕ), i < D.sts.size → ∀ r : Fin 2,
        ∀ ℓ < (D.files cls i).sch.levels.size, ∀ j < ((D.files cls i).sch.levels[ℓ]!).q,
          ∃ q ∈ pieces x, q.table = i ∧ q.rep = r.val ∧ q.level = ℓ ∧ q.query = j) :
    TargetE rd P (accR rd chainAt) := by
  intro x ms cs hA D cls i hi
  obtain ⟨⟨xo, sto, hst, hf, hx⟩, hr, hb, hP⟩ := id hA
  have hco := hcoef x ms cs hA D cls i hi
  rw [accR_eq chainAt D] at hco ⊢
  have hG := D.rowG cls (D.accE chainAt) hst hf hx hi
  refine ⟨fun _ => chainAt x ms i, fun r => r * nOpen (D.files cls i).sch, fun r => ?_, fun r => ?_⟩
  · exact D.unitSums_of_pieces cls (D.accE chainAt) i r (pieces x) sideAt coefAt chainAt (hholds x ms cs hP)
      (hside x ms cs hA D cls i hi r) (hco r) (hcover x ms cs hA D cls i hi r)
  · exact D.readsEnds_accE cls (D.accE chainAt) i r chainAt hG D.points_eq (idx_lt D hi r)
      (shaped_of_base hr hb D _ (idx_lt D hi r))

end Top

end RecDeploy
