module

public import Proofs.Flock.C3.Compose.HoloStreamSizes
public import Proofs.Flock.VBridge.Sparse.Defs
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Proofs.Flock.C3.Compose.HoloEncoded

/-! The sparse phases' rounds at the map read. -/

@[expose] public section

namespace FlockC3.Compose

open Flock Flock.RecVStar Flock.Guarantees

theorem heldBefore_size (s : HoloStream) (rs : Array SchedRound) :
    heldBefore s rs rs.size = (rs.toList.map fun q => s.group q.1).sum := by
  unfold heldBefore
  rw [Nat.min_self]
  congr 1
  refine List.ext_getElem (by simp) fun n _ h₂ => ?_
  have hn : n < rs.size := by simpa using h₂
  simp [hn]

theorem heldBefore_sparse_size (x : Fin 2) (sp : RecSparse.Shape) :
    heldBefore (.sparse x) (sparseRounds sp) (sparseRounds sp).size = FlockVBridge.Sparse.nc (sparseOf sp) := by
  have hs : holoSizes [(.sparse x, sparseRounds sp)] =
      (sparseRounds sp).toList.flatMap fun q => List.replicate q.1 1 := by
    simp [holoSizes]
  have h : ((sparseRounds sp).toList.map (·.1)).sum = (sparseForms (sparseOf sp)).length := by
    have e := congrArg List.length (holoSizes_sparse x sp)
    rwa [hs, flatMap_replicate_fst, List.length_replicate, List.length_map] at e
  rw [heldBefore_size]
  show ((sparseRounds sp).toList.map fun q => q.1).sum = _
  rw [h, sparseForms, List.length_replicate]
  unfold FlockVBridge.Sparse.nc FlockVBridge.Sparse.nR
  omega

theorem roundAt_sparse (rs srs : Array SchedRound) (R : List (HoloStream × Array SchedRound)) (x : Fin 2) (q : ℕ) :
    roundAt ((.rep 0, rs) :: (.rep 1, rs) :: (.sparse 0, srs) :: (.sparse 1, srs) :: R) (.sparse x) q =
      1 + 2 * heldBefore (.rep 0) rs rs.size + x.val * heldBefore (.sparse 0) srs srs.size +
        heldBefore (.sparse x) srs q := by
  have h1 : heldBefore (.rep 1) rs rs.size = heldBefore (.rep 0) rs rs.size := rfl
  rcases x with ⟨_ | _ | n, hn⟩
  · show 1 + roundIn _ (.sparse 0) q = 1 + 2 * heldBefore (.rep 0) rs rs.size +
      0 * heldBefore (.sparse 0) srs srs.size + heldBefore (.sparse 0) srs q
    rw [roundIn_cons_ne (s' := .rep 0) (s := .sparse 0) (by decide),
      roundIn_cons_ne (s' := .rep 1) (s := .sparse 0) (by decide), roundIn_cons_self, h1]
    omega
  · show 1 + roundIn _ (.sparse 1) q = 1 + 2 * heldBefore (.rep 0) rs rs.size +
      1 * heldBefore (.sparse 0) srs srs.size + heldBefore (.sparse 1) srs q
    rw [roundIn_cons_ne (s' := .rep 0) (s := .sparse 1) (by decide),
      roundIn_cons_ne (s' := .rep 1) (s := .sparse 1) (by decide),
      roundIn_cons_ne (s' := .sparse 0) (s := .sparse 1) (by decide), roundIn_cons_self, h1]
    omega
  · exact absurd hn (by omega)

theorem roundAt_sparseRounds (rs : Array SchedRound) (sp : RecSparse.Shape) (R : List (HoloStream × Array SchedRound))
    (x : Fin 2) (q : ℕ) :
    roundAt ((.rep 0, rs) :: (.rep 1, rs) :: (.sparse 0, sparseRounds sp) :: (.sparse 1, sparseRounds sp) :: R)
        (.sparse x) q =
      1 + 2 * heldBefore (.rep 0) rs rs.size + x.val * FlockVBridge.Sparse.nc (sparseOf sp) +
        heldBefore (.sparse x) (sparseRounds sp) q := by
  rw [roundAt_sparse, heldBefore_sparse_size]

theorem filter_range_sublist (P : ℕ → Bool) (m n : ℕ) (h : ∀ q ∈ (List.range n).filter P, q < m) :
    List.Sublist ((List.range n).filter P) (List.range m) := by
  induction n with
  | zero => simp
  | succ n ih =>
    have ih' := ih fun q hq => h q (((List.range_sublist.2 (Nat.le_succ n)).filter P).subset hq)
    have hn : P n = true → n < m := fun hp => h n (by simp [hp])
    rw [List.range_succ, List.filter_append, List.filter_cons, List.filter_nil]
    cases hp : P n
    · simpa using ih'
    · show List.Sublist (List.filter P (List.range n) ++ [n]) (List.range m)
      exact (List.filter_sublist.append (List.Sublist.refl [n])).trans
        (by rw [← List.range_succ]; exact List.range_sublist.2 (hn hp))

theorem map_getD_eq_filter {H : Array (Array F128)} {ks : List ℕ} (hks : List.Sublist ks (List.range H.size))
    (hne : ∀ q ∈ ks, (H.getD q #[]).size ≠ 0) (hlen : ks.length = (H.filter (·.size != 0)).size) :
    ks.map (H.getD · #[]) = (H.filter (·.size != 0)).toList := by
  have hH : (List.range H.size).map (H.getD · #[]) = H.toList := by
    refine List.ext_getElem (by simp) fun n _ h₂ => ?_
    have hn : n < H.size := by simpa using h₂
    first
    | simp [hn]
    | simp only [List.getElem_map, List.getElem_range, Array.getElem_toList, Array.getD_eq_getD_getElem?,
        Array.getElem?_eq_getElem hn, Option.getD_some]
  have hsub : List.Sublist (ks.map (H.getD · #[])) H.toList := by
    rw [← hH]
    exact hks.map _
  have hf : (ks.map (H.getD · #[])).filter (·.size != 0) = ks.map (H.getD · #[]) := by
    rw [List.filter_eq_self]
    intro a ha
    obtain ⟨q, hq, rfl⟩ := List.mem_map.1 ha
    simpa using hne q hq
  have h2 := hsub.filter (fun a : Array F128 => a.size != 0)
  rw [hf, ← Array.toList_filter] at h2
  refine h2.eq_of_length ?_
  rw [List.length_map, Array.length_toList]
  exact hlen

theorem coinRounds_held {H : Array (Array F128)} {rs : Array SchedRound}
    (hlt : ∀ q ∈ coinRounds rs, q < H.size) (hne : ∀ q ∈ coinRounds rs, (H.getD q #[]).size ≠ 0)
    (hlen : (coinRounds rs).length = (H.filter (·.size != 0)).size) :
    (coinRounds rs).map (H.getD · #[]) = (H.filter (·.size != 0)).toList :=
  map_getD_eq_filter (filter_range_sublist (fun k => 0 < (rs.getD k default).1) H.size rs.size hlt) hne hlen

end FlockC3.Compose
