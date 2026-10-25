import Proofs.Flock.Claims.Recursion.VerdictRowD
import Proofs.Flock.Claims.Recursion.OpenPieces

/-!
# Row D of the verdict from the `rec-open` pieces

The `rec-open` pieces (`OpenPieces`) hold, each, where its query opens at the side and position it reads
(`openHolds`). This file is the half of `hpieces` that takes them to row D (`DecAt.rowD_of_queries`).

* `DecAt.side`: rep `r` of table `i`'s side at the decision: the table's schedule, and the rep's cap and Ligerito proof.
* `DecAt.rowD_of_pieces`: `RowD` from row G's `merkle` and `fast`, row C's `decode` and the pieces, where the pieces
  cover every query of every level of the rep (`hcover`), read the rep's side off the messages (`hside`) and, at the
  verifier rows, the position the rows draw for their query off the coins (`hpos`, the coin tie).
-/

namespace RecDeploy.Rows

open Flock
open Flock.RecAlgebra (verifierRows)
open FlockSoundness.Refine (shapeOf)
open FlockSoundness.Discharge.Recursion (HClass)

namespace DecAt

variable {rd : Claims.Msg → Option FlockDeploy.FinalP} {x : Claims.Msg} {ms : List Claims.Msg}
  {cs : List FlockDeploy.Coin} (D : DecAt rd x ms cs) (cls : HClass) (acc : ℕ → Array F128) (i : ℕ) (r : Fin 2)

/-- **Rep `r` of table `i`'s side at the decision**: the table's schedule, and the rep's decided cap and Ligerito
proof. -/
def side : OpenSide :=
  ⟨(D.files cls i).sch, ((D.files cls i).proof r).cap, ((D.files cls i).proof r).ligerito⟩

/-- **Row D from the `rec-open` pieces**: at table `i`'s plain scheme and `fast100` schedule and rep `r`'s decoded
file, where every query `j` of every level `ℓ` is some piece's (`hcover`), the pieces of rep `r` read its side
(`hside`) and, at the verifier rows, the position the rows draw for their query (`hpos`), and every piece holds,
every level's queries pass (`rowD_of_queries`). -/
theorem rowD_of_pieces
    (hmerkle : (D.files cls i).st.merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted")
    (hfast : Flock.fast100 (D.files cls i).st.m = some (D.files cls i).sch)
    (hdec : decodeProof (shapeOf (D.files cls i).st (D.files cls i).sch (D.files cls i).extras.size)
      ((D.files cls i).bytes r) = .ok ((D.files cls i).proof r))
    (pieces : List OpenPiece) (sideAt : Claims.Msg → List Claims.Msg → OpenPiece → OpenSide)
    (posAt : Claims.Msg → List FlockDeploy.Coin → OpenPiece → ℕ)
    (hholds : ∀ q ∈ pieces, openHolds sideAt posAt q x ms cs)
    (hside : ∀ q ∈ pieces, q.table = i → q.rep = r.val → sideAt x ms q = D.side cls i r)
    (hpos : ∀ v info kappa,
      verifierRows (D.sh cls i) (D.rep cls acc i r).rounds (D.rep cls acc i r).extras (D.rep cls acc i r).comb =
        .ok (v, info, kappa) →
      ∀ q ∈ pieces, q.table = i → q.rep = r.val → posAt x cs q = (info.lg.conss[q.level]!).pos[q.query]!)
    (hcover : ∀ ℓ < (D.files cls i).sch.levels.size, ∀ j < ((D.files cls i).sch.levels[ℓ]!).q,
      ∃ q ∈ pieces, q.table = i ∧ q.rep = r.val ∧ q.level = ℓ ∧ q.query = j) :
    D.RowD cls acc i r :=
  D.rowD_of_queries cls acc i r hmerkle hfast hdec fun v info kappa hv ℓ hℓ j hj => by
    obtain ⟨q, hq, hi, hr, rfl, rfl⟩ := hcover ℓ hℓ j hj
    have h := hholds q hq
    unfold openHolds at h
    rw [hside q hq hi hr, hpos v info kappa hv q hq hi hr] at h
    exact h

end DecAt

end RecDeploy.Rows
