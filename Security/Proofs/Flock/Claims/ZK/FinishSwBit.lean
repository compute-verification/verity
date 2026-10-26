import Proofs.Flock.Claims.ZK.FinishSwFw
import Proofs.Flock.VBridge.Algebra.ToVStar
import Proofs.Flock.VBridge.Residuals
import Proofs.Flock.Soundness.Refine.Final

/-!
# The firewall's switched vectors at the packed table

`FinishSrc`'s `repCut_fw` asks its source's switched vectors to be the firewall's (`Finish.switchVec`), read off the
packed F-table `g` by the bits of its elements; the sources give the prover's (`FlockComplete.switchVec`), read off the
trace's bits. At the packed table they agree, wherever the claim's point has the `m − 6` coordinates the PIOP's claims
have (`switchVec_packed`): `eqTable` of its tail has `2^(m−7)` entries, each a word of `packed`, whose bit `v` is the
trace's bit `128 y + v` (`packed_bit`).
-/

namespace FlockComplete

open Flock

/-- A fold that adds `f v` where `b v` holds is the sum of the terms `if b v then f v else 0`. -/
theorem foldl_ite_add (b : ℕ → Bool) (f : ℕ → F128) : ∀ (L : List ℕ) (a : F128),
    L.foldl (fun acc v => if b v then acc + f v else acc) a =
      (L.map fun v => if b v then f v else 0).foldl (· + ·) a
  | [], _ => rfl
  | v :: L, a => by
    simp only [List.foldl_cons, List.map_cons]
    rw [foldl_ite_add b f L]
    congr 1
    by_cases hb : b v = true
    · rw [ite_eq_left hb, ite_eq_left hb]
    · rw [ite_eq_right hb, ite_eq_right hb, Flock.RecAlgebra.add_zero_f128]

/-- **Bit `v` of the packed table's word `y` is the trace's bit `128 y + v`.** -/
theorem packed_bit (tr : Trace) (m : ℕ) {y v : ℕ} (hy : y < 2 ^ (m - 7)) (hv : v < 128) :
    ((packed tr m)[y]!).bit v = tr.z[128 * y + v]! := by
  unfold packed
  rw [FlockSoundness.Refine.getElem!_range_map _ hy]
  have e : (List.range 128).foldl (fun acc v => if tr.z[128 * y + v]! then acc + F128.monomial v else acc) 0 =
      FlockVBridge.ofBits (fun v => tr.z[128 * y + v]!) := by
    unfold FlockVBridge.ofBits
    exact foldl_ite_add (fun v => tr.z[128 * y + v]!) F128.monomial (List.range 128) 0
  rw [e, Flock.RecAlgebra.bit_eq_bitOf _ hv, FlockVBridge.bitOf_ofBits _ hv]

/-- **The firewall's switched vector at the packed table is the prover's**, at a claim with `m − 6` coordinates. -/
theorem switchVec_packed {m : ℕ} (hm : 7 ≤ m) (tr : Trace) (c : Claim) (hx : c.x.size = m - 6) :
    Flock.Firewall.Finish.switchVec (packed tr m) c = FlockComplete.switchVec tr c := by
  have hsz : (eqTable (c.x.extract 1 c.x.size)).size = 2 ^ (m - 7) := by
    rw [FlockLevel3.eqTable_size, Array.size_extract]
    congr 1
    omega
  unfold Flock.Firewall.Finish.switchVec FlockComplete.switchVec
  dsimp only
  apply Array.ext
  · simp
  · intro v h1 _
    have hv : v < 128 := by simpa using h1
    simp only [Array.getElem_map, Array.getElem_range]
    unfold Flock.Firewall.Finish.sumF FlockComplete.sumF
    refine Flock.RecAlgebra.foldl_mem_congr _ fun y hy a => ?_
    rw [List.mem_range, hsz] at hy
    dsimp only
    rw [packed_bit tr m hy hv]

end FlockComplete
