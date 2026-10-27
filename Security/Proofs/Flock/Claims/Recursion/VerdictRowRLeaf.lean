import Proofs.Flock.Claims.Recursion.ElemRead

/-!
# Row R's leaf at the concrete readers

`Top.TargetRDep rd (PD rd resP key (elemReadC (Top.accRC rd key)) resVRead side posAt) (Top.accRC rd key)`, the row R
leaf of `recKSDecompL_of_leaves` with `elemAt := Rows.elemReadC (Top.accRC rd key)` and `vvAt := Rows.resVRead`
(`targetRDep_readers`). `ElemCover`'s elements half is `DecAt.elemReadC_eq` and its verifier values half
`DecAt.resVRead_eq`, at every decision `At` admits. What it still takes is that `resP x` has the pieces
(`ResPiecesAt`): for each rep `r` and group `p` of S, one of part `p`'s `S` keyed by group `p` at a copy `(o, g)` with
`o + g = r`.
-/

namespace RecDeploy.DecompBR

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP)
open Flock.RecAlgebra (structureOf)
open Flock.RecResiduals (portSizes groups)
open FlockSoundness.Discharge.Recursion (HClass)
open RecDeploy.Rows (DecAt elemReadC resVRead)

/-- **The pieces `resP` has** at every decision `At` admits at the concrete readers: for each rep `r` and group `p` of
S, a piece of part `p`'s `S`, keyed by group `p` at a copy `(o, g)` with `o + g = r`. -/
def ResPiecesAt (rd : Msg → Option FinalP) (resP : Msg → List ResPiece) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ)
    (side : Msg → List Msg → OpenPiece → OpenSide) (posAt : Msg → List Coin → OpenPiece → ℕ) : Prop :=
  ∀ (x : Msg) (ms : List Msg) (cs : List Coin),
    Top.At rd (PD rd resP key (elemReadC (Top.accRC rd key)) resVRead side posAt) x ms cs →
    ∀ (D : DecAt rd x ms cs) (cls : HClass) (i : ℕ), i < D.sts.size →
    ∀ (S : RecAlgebra.GfS) (gs : Array (Array ℕ × Array ℕ × Array ℕ)),
      structureOf (D.sh cls i) = .ok S → groups S (portSizes (D.sh cls i)) = .ok gs →
      ∀ (r : Fin 2) p (_ : p < gs.size), ∃ q ∈ resP x,
        (∃ ns nv, RecResiduals.part (D.sh cls i) p = .ok (q.S, ns, nv)) ∧
        q.key.1 = p ∧ q.key.2.1 + q.key.2.2.1 = r.val

/-- **Row R's leaf at the concrete readers**: `TargetRDep` at `elemAt := elemReadC (Top.accRC rd key)`,
`vvAt := resVRead` and the accumulators `Top.accRC rd key`, from the pieces `resP` has (`ResPiecesAt`). -/
theorem targetRDep_readers (rd : Msg → Option FinalP) {resP : Msg → List ResPiece}
    {key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ} {side : Msg → List Msg → OpenPiece → OpenSide}
    {posAt : Msg → List Coin → OpenPiece → ℕ} (hp : ResPiecesAt rd resP key side posAt) :
    Top.TargetRDep rd (PD rd resP key (elemReadC (Top.accRC rd key)) resVRead side posAt) (Top.accRC rd key) :=
  targetRDep_accRC rd fun x ms cs hA D cls i hi S gs hS hg r p hpp => by
    obtain ⟨⟨xo, sto, hst, hf, hx⟩, -⟩ := id hA
    obtain ⟨q, hq, hpart, h1, hog⟩ := hp x ms cs hA D cls i hi S gs hS hg r p hpp
    exact ⟨q, hq, hpart, h1, hog, fun k hk => D.elemReadC_eq cls (Top.accRC rd key) hst hf hx hi hS hg r hpp q.key h1
      hog k hk⟩

end RecDeploy.DecompBR
