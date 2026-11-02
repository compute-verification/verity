import Proofs.Flock.Claims.Recursion.VerdictRowG
import Proofs.Flock.Claims.Recursion.RecordCanon
import Proofs.Flock.Claims.Recursion.DecodeTies
import Proofs.Flock.Claims.Recursion.VerdictRowDPieces
import Proofs.Flock.Claims.Recursion.VerdictRowR
import Proofs.Flock.Claims.Recursion.VerdictRowE

/-!
# The verdict's rows, assembled at a table

`DecAt.runs_assembled`: table `i`'s reps run (`TableFiles.Runs`, `runs_of_rows`) at a decision of an inner statement
the reader reads at the dense form, from the rows as their owners proved them:

* **G** outright (`rowG`).
* **B**: S11's stream, S12's rounds and their coins (`finds`, `roundsWF`, `noEmpty`); the frames (`hframed`) stay a
  hypothesis.
* **C** from G, the session's points and each rep's shaped proof (`rowC_of`).
* **L** stays a hypothesis (`hL`).
* **D** from G, C's decode and the `rec-open` pieces (`rowD_of_pieces`), at the pieces' holding, layout (`hside`),
  coin (`hpos`) and cover (`hcover`) ties.
* **R** from S, its groups, the rounds' sizes, `w`'s size and the parts' residuals (`rowR_of_parts`).
* **E** from G and, at each rep, the units' sums (`UnitSums`) and the algebra's reading of their ends (`ReadsEnds`)
  (`rowE_of_units`), where `verifierRows` returns by R's sizes (`verifierRows_rep`).
-/

namespace RecDeploy.Rows

open Flock
open FlockDeploy (Coin FinalP tabsOf pointsOf shapeAt)
open FlockVBridge.Family (FramedAt ProofShaped roundSizes)
open FlockVBridge.AccChain (UnitSums ReadsEnds)
open Flock.RecAlgebra (verifierRows valOf NoEmpty structureOf)
open Flock.RecResiduals (portSizes groups)
open FlockSoundness.Discharge.Exec (repFrames)
open FlockSoundness.Discharge.Recursion (HClass)

namespace DecAt

variable {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {ms : List Claims.Msg} {cs : List Coin}
  (D : DecAt rd x ms cs) (cls : HClass) (acc : ℕ → Array F128)

/-- **Table `i`'s reps run at the decision**, from rows G, B, C, D, R and E as proved, at the hypotheses they still
take: B's frames, row L, D's pieces and their ties, R's structure, sizes and residuals, and E's unit sums and ends. -/
theorem runs_assembled {xo : Claims.Msg} {sto : Claims.Recursion.CompiledStatement} (hst : parse xo = some sto)
    (hf : Claims.Recursion.Form.Dense sto.form) (hx : x = xIOf xo) {i : ℕ} (hi : i < D.sts.size)
    (hti : i < D.spec.tables.size)
    (hframed : ∀ r : Fin 2, FramedAt ((D.files cls i).stream r).rounds 0 (repFrames (D.files cls i).st
      (D.files cls i).sch ((D.files cls i).spec.domain (D.files cls i).spec.tables[(D.files cls i).i]! r).toUTF8
      (2 + (D.files cls i).extras.size) ((D.files cls i).proof r)
      (valOf (D.sh cls i) (D.rep cls acc i r).w (D.rep cls acc i r).comb (D.rep cls acc i r).cons .T)))
    (hpts : D.sess.points = pointsOf D.spec D.d.c₁)
    (hk : ∀ r : Fin 2, 2 * i + r < D.d.f.ps.size)
    (hsh : ∀ r : Fin 2,
      ProofShaped (shapeAt (tabsOf D.sts) (pointsOf D.spec D.d.c₁) (2 * i + r)) (D.d.f.ps[2 * i + r]'(hk r)))
    (hL : ∀ r, D.RowL cls acc i r)
    (pieces : List OpenPiece) (sideAt : Claims.Msg → List Claims.Msg → OpenPiece → OpenSide)
    (posAt : Claims.Msg → List Coin → OpenPiece → ℕ)
    (hholds : ∀ q ∈ pieces, openHolds sideAt posAt q x ms cs)
    (hside : ∀ r : Fin 2, ∀ q ∈ pieces, q.table = i → q.rep = r.val → sideAt x ms q = D.side cls i r)
    (hpos : ∀ r : Fin 2, ∀ v info kappa,
      verifierRows (D.sh cls i) (D.rep cls acc i r).rounds (D.rep cls acc i r).extras (D.rep cls acc i r).comb =
        .ok (v, info, kappa) →
      ∀ q ∈ pieces, q.table = i → q.rep = r.val → posAt x cs q = (info.lg.conss[q.level]!).pos[q.query]!)
    (hcover : ∀ r : Fin 2, ∀ ℓ < (D.files cls i).sch.levels.size, ∀ j < ((D.files cls i).sch.levels[ℓ]!).q,
      ∃ q ∈ pieces, q.table = i ∧ q.rep = r.val ∧ q.level = ℓ ∧ q.query = j)
    {S : RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)}
    (hS : structureOf (D.sh cls i) = .ok S) (hg : groups S (portSizes (D.sh cls i)) = .ok gs)
    (hsz : ∀ r : Fin 2, (D.rep cls acc i r).rounds.toList.map Array.size = roundSizes (D.sh cls i))
    (hw : ∀ r : Fin 2, (D.rep cls acc i r).w.size = S.nw) (hparts : ∀ r : Fin 2, D.PartsZero cls acc i r gs)
    (accC : Fin 2 → ℕ → Array F128) (p0 : Fin 2 → ℕ)
    (hu : ∀ r : Fin 2, UnitSums (D.sh cls i) (accC r) (D.opens cls acc i r (p0 r)))
    (hrd : ∀ r : Fin 2, ReadsEnds (D.sh cls i) (D.rep cls acc i r) (accC r) (D.opens cls acc i r (p0 r))) :
    (D.files cls i).Runs := by
  have hG := D.rowG cls acc hst hf hx hi
  have hC : ∀ r, D.RowC cls acc i r := fun r => D.rowC_of cls acc i r hG hpts (hk r) (hsh r)
  have hne : ∀ r : Fin 2, NoEmpty (D.rep cls acc i r).rounds := fun r => D.noEmpty cls acc hti r
  exact D.runs_of_rows cls acc i hG
    (fun r => ⟨D.finds cls hti r, hne r, D.roundsWF cls hti r, hframed r⟩) hC hL
    (fun r => D.rowD_of_pieces cls acc i r hG.merkle hG.fast (hC r).decode pieces sideAt posAt hholds (hside r)
      (hpos r) (hcover r))
    (fun r => D.rowR_of_parts cls acc i r hS hg (hne r) (hsz r) (hw r) (hparts r))
    (fun r => D.rowE_of_units cls acc i r hG (D.verifierRows_rep cls acc i r hS (hne r) (hsz r)) (hu r) (hrd r))

end DecAt

end RecDeploy.Rows
