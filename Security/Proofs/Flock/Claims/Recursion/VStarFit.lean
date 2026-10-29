import Proofs.Flock.Claims.Recursion.VStarDeployed
import Proofs.Flock.Claims.Recursion.SeqPrefix

/-!
# The inner prover's rows fitted to the compiled statement (`InnerRec.fit`)

Each inner round before the last carries one commit string a slot of V\*'s layout of record (`DeployedR.layR`), in the
layout's order (`slotsAt`, a part, a port and an instance each):

* round 0: RecCap's `N` instances at port `top` (`capHold`);
* stream round `j` (round `j + 1`): where it commits tops, each top's instances at its cap (`capPieceAt`), then the row it
  absorbs at the first part holding it (`rowPiece`); where it commits none, that row alone where it absorbs one, else
  nothing.

`InnerRec.fit x` makes the inner records commit exactly those rows: as many as the round's slots, each sized to its
port's words (`fitSlot`), a top's as `Firewall.topRows` lays a node (its 32 words, then zero words), every word 16 bits.
So the deployed prover's rows are counted and sized from `x`'s schedule and port widths by construction; at records
already of that shape `fit` changes nothing. `strs` is the round's commit strings, and the inner prover's message is
their concatenation (`P_eval`).
-/

namespace RecDeploy.VStarBuild

open _root_.Flock
open Claims (Msg)

/-- Part `s`'s circuit at the compiled statement `x`; the default circuit where it has none (as `topPiece` reads it). -/
def circAt (x : Msg) (s : ℕ) : Circuit :=
  ((readX x).bind fun r => r.w.parts[s]?.bind partCircuit).getD default

/-- **The absorbed row's slot of stream round `j`**, as `rowPiece` reads it: the first part holding the row
(`heldRows`), the port's index and the instance. -/
def rowSlot (r : XR) (j : ℕ) : List (ℕ × ℕ × ℕ) :=
  ((List.range r.w.parts.length).findSome? fun s => ((heldRows r s).find? (·.1 == j)).bind fun h =>
    (r.w.parts[s]?.bind partCircuit).map fun _ => (s, h.2.2.1, h.2.2.2)).toList

/-- **The `e`-th top's slots of stream round `j`**, as `capPieceAt` reads them: its cap's part, port `top`, and rep
`j / R`'s `n` instances. -/
def capSlots (r : XR) (rs : Array RecVStar.SchedRound) (j e : ℕ) : List (ℕ × ℕ × ℕ) :=
  let s := capsFrom r rs + topsBefore rs (j % rs.size) + e
  let c := (r.w.parts[s]?.bind partCircuit).getD default
  let n := ((rs.getD (j % rs.size) default).2.1).getD e 0
  (List.range n).map fun t => (s, (capPort c).getD 0, j / rs.size * n + t)

/-- **Inner round `i`'s slots** at the compiled statement `x`, in the order its message lays their commit strings. -/
def slotsAt (x : Msg) (i : ℕ) : List (ℕ × ℕ × ℕ) :=
  match readX x, i with
  | none, _ => []
  | some r, 0 => ((capAt r).bind fun p => (r.w.parts[p.1]?.bind partCircuit).bind fun c => (capPort c).map fun q =>
      (List.range p.2).map fun n => (p.1, q, n)).getD []
  | some r, j + 1 => ((repSched r.shape).map fun rs =>
      let sr := rs.getD (j % rs.size) default
      if sr.2.1.isEmpty then (if decide (j < 2 * rs.size) && sr.2.2.2 then rowSlot r j else [])
      else (List.range sr.2.1.size).flatMap (capSlots r rs j) ++ rowSlot r j).getD []

/-- A slot's port's word count. -/
def widthAt (x : Msg) (sl : ℕ × ℕ × ℕ) : ℕ := (((circAt x sl.1).ports[sl.2.1]?).map (·.words)).getD 0

/-- A slot's port is a cap's `top`. -/
def isTop (x : Msg) (sl : ℕ × ℕ × ℕ) : Bool := (((circAt x sl.1).ports[sl.2.1]?).map (·.name == "top")).getD false

/-- `k` words of `ws`, each reduced to 16 bits, zero words past its end. -/
def fitRow (k : ℕ) (ws : Array ℕ) : Array ℕ := ((ws.toList.map (· % 65536) ++ List.replicate k 0).take k).toArray

/-- **A row fitted to its slot**: at a cap's `top`, a node's 32 words then zero words (`Firewall.topRows`); at any other
port, the port's words. -/
def fitSlot (x : Msg) (sl : ℕ × ℕ × ℕ) (ws : Array ℕ) : Array ℕ :=
  if isTop x sl then fitRow 32 ws ++ Array.replicate (widthAt x sl - 32) 0 else fitRow (widthAt x sl) ws

/-- **The inner records fitted to the compiled statement `x`**: at the history before round `i`, one row a slot of
round `i` (`slotsAt`), the `j`-th the records' `j`-th fitted to its slot (`fitSlot`). -/
def InnerRec.fit (I : InnerRec) (x : Msg) : InnerRec where
  R := I.R
  ne := I.ne
  rows w r h := (slotsAt x h.length).mapIdx fun j sl => fitSlot x sl ((I.rows w r h).getD j #[])
  rest := I.rest

/-- **A round's commit strings**: hm96's `b ‖ c` of each row's digest under its salt, in order. -/
def strs (I : InnerRec) (w : Claims.Flock.Wit) (ρ : I.Tape) (h : List (Msg × Msg)) : List Msg :=
  ((I.rows w ρ.1 h).mapIdx fun j ws => (digestOf ws, ρ.2 h.length j)).map fun p =>
    hmLeaf (asX p.1 + maskX p.2, saltH FlockDeploy.sha512 p.2)

/-- **The inner prover's message is its round's commit strings, concatenated.** -/
theorem P_eval (I : InnerRec) (w : Claims.Flock.Wit) (xI : Msg) (h : List (Msg × Msg)) (ρ : I.Tape) :
    (((I.P w).next xI h ρ).eval FlockDeploy.sha512).1 = (strs I w ρ h).flatten := by
  simp [InnerRec.P, commitThen_eval, strs, Claims.OC.eval]

theorem fit_rows_length (I : InnerRec) (x : Msg) (w : Claims.Flock.Wit) (r : I.R) (h : List (Msg × Msg)) :
    ((I.fit x).rows w r h).length = (slotsAt x h.length).length := by
  simp [InnerRec.fit]

theorem fit_rows_getElem? (I : InnerRec) (x : Msg) (w : Claims.Flock.Wit) (r : I.R) (h : List (Msg × Msg)) (j : ℕ) :
    ((I.fit x).rows w r h)[j]? =
      ((slotsAt x h.length)[j]?).map fun sl => fitSlot x sl ((I.rows w r h).getD j #[]) := by
  simp [InnerRec.fit, List.getElem?_mapIdx]

theorem fitRow_size (k : ℕ) (ws : Array ℕ) : (fitRow k ws).size = k := by
  simp [fitRow]

theorem fitSlot_size (x : Msg) (sl : ℕ × ℕ × ℕ) (ws : Array ℕ) (h32 : isTop x sl = true → 32 ≤ widthAt x sl) :
    (fitSlot x sl ws).size = widthAt x sl := by
  unfold fitSlot
  split
  · rename_i ht
    simp [fitRow_size]
    have := h32 ht
    omega
  · exact fitRow_size _ _

theorem strs_length (I : InnerRec) (x : Msg) (w : Claims.Flock.Wit) (ρ : (I.fit x).Tape) (h : List (Msg × Msg)) :
    (strs (I.fit x) w ρ h).length = (slotsAt x h.length).length := by
  simp [strs, InnerRec.fit]

end RecDeploy.VStarBuild
