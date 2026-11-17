module

public import Definitions.Pouw.H1Zy.Lane
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Data.Rat.Cast.Order
import Mathlib.Tactic.Linarith.Frontend
import Mathlib.Tactic.NormNum.Abs
import Mathlib.Tactic.NormNum.DivMod
import Mathlib.Tactic.NormNum.OfScientific
import Mathlib.Tactic.Ring.RingNF
import Proofs.Pouw.H1Zy.Combine

/-! What a lane's `ok = 1` gives, on words. -/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Pouw.H1Zy

theorem length_pairs {α : Type} (f : α → α → α) : ∀ l : List α, (pairs f l).length = (l.length + 1) / 2
  | [] => by simp [pairs]
  | [_] => by simp [pairs]
  | _ :: _ :: rest => by
      simp only [pairs, List.length_cons, length_pairs f rest]
      omega

theorem pairs_mux_getElem? {α : Type} (s : Bool) : ∀ (l : List α) (i : ℕ), 2 * i + 1 < l.length →
    (pairs (mux s) l)[i]? = l[2 * i + (if s then 1 else 0)]?
  | [], _, h => by simp at h
  | [_], _, h => by simp at h
  | a :: b :: rest, 0, _ => by cases s <;> simp [pairs, mux]
  | a :: b :: rest, i + 1, h => by
      simp only [List.length_cons] at h
      have := pairs_mux_getElem? s rest i (by omega)
      simp only [pairs, List.getElem?_cons_succ]
      rw [this]
      have e : 2 * (i + 1) + (if s then 1 else 0) = (2 * i + (if s then 1 else 0)) + 1 + 1 := by split <;> omega
      rw [e]
      simp

theorem foldl_pick_getElem? {α : Type} : ∀ (sel : List Bool) (cands : List α) (i : ℕ),
    (i + 1) * 2 ^ sel.length ≤ cands.length →
    (sel.foldl (fun cs s => pairs (mux s) cs) cands)[i]? = cands[i * 2 ^ sel.length + bitsVal sel]?
  | [], cands, i, _ => by simp [bitsVal]
  | s :: ss, cands, i, h => by
      simp only [List.foldl_cons, List.length_cons] at h ⊢
      have hv := bitsVal_lt ss
      have h2 : 2 * ((i + 1) * 2 ^ ss.length) ≤ cands.length := by
        rw [pow_succ] at h
        linarith [show (i + 1) * (2 ^ ss.length * 2) = 2 * ((i + 1) * 2 ^ ss.length) by ring]
      have hp : (i + 1) * 2 ^ ss.length ≤ (pairs (mux s) cands).length := by
        rw [length_pairs]
        omega
      rw [foldl_pick_getElem? ss (pairs (mux s) cands) i hp]
      rw [pairs_mux_getElem? s cands _ (by nlinarith)]
      congr 1
      simp only [bitsVal, pow_succ]
      split <;> ring

theorem pick_ofFn {α : Type} (d : α) (sel : List Bool) (hsel : sel.length = 5) (f : Fin 32 → α) :
    pick d sel (List.ofFn f) = f ⟨bitsVal sel, by have := bitsVal_lt sel; rw [hsel] at this; omega⟩ := by
  have hlt : bitsVal sel < 32 := by have := bitsVal_lt sel; rw [hsel] at this; omega
  have h := foldl_pick_getElem? sel (List.ofFn f) 0 (by simp [hsel])
  simp only [zero_mul, zero_add, List.getElem?_ofFn] at h
  unfold pick
  rw [List.headD_eq_head?_getD, List.head?_eq_getElem?, h]
  simp [hlt]

theorem tree_orBit (l : List Bool) : tree orBit false l = false ↔ ∀ b ∈ l, b = false := by
  rw [tree_eq_foldl orBit false (fun a b c => Bool.or_assoc a b c) (fun a => Bool.false_or a) l]
  have key : ∀ (l : List Bool) (x : Bool), l.foldl orBit x = (x || l.any id) := by
    intro l
    induction l with
    | nil => intro x; simp
    | cons b l ih => intro x; simp [List.foldl_cons, ih, orBit, Bool.or_assoc]
  simp [key]

theorem zyCheck_iff {R C : ℕ} (dqW : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 16) (zr : Fin R → Fin C → BitVec 32)
    (yr : Fin R → Fin C → BitVec 16) (z : Fin R → Fin C → BitVec 32) (sa : Fin R → BitVec 32)
    (sw : Fin C → BitVec 32) :
    zyCheck dqW zr yr z sa sw = true ↔ ∀ i j, zr i j = z i j ∧ yr i j = dqW (z i j) (sa i) (sw j) := by
  unfold zyCheck
  rw [tree_andBit]
  constructor
  · intro h i j
    have := h _ (List.mem_flatten.2 ⟨_, List.mem_ofFn.2 ⟨i, rfl⟩, List.mem_ofFn.2 ⟨j, rfl⟩⟩)
    simpa [zyCell] using this
  · intro h b hb
    obtain ⟨l, hl, hbl⟩ := List.mem_flatten.1 hb
    obtain ⟨i, rfl⟩ := List.mem_ofFn.1 hl
    obtain ⟨j, rfl⟩ := List.mem_ofFn.1 hbl
    simp [zyCell, (h i j).1, (h i j).2]

theorem toInt_mul_bounds (a b : BitVec 8) : -16384 ≤ a.toInt * b.toInt ∧ a.toInt * b.toInt ≤ 16384 := by
  have h1 := BitVec.le_toInt a
  have h2 := BitVec.toInt_lt (x := a)
  have h3 := BitVec.le_toInt b
  have h4 := BitVec.toInt_lt (x := b)
  norm_num at h1 h2 h3 h4
  constructor <;> nlinarith

theorem toInt_ofInt_of_range (v : ℤ) (h1 : -2 ^ 31 ≤ v) (h2 : v < 2 ^ 31) : (BitVec.ofInt 32 v).toInt = v := by
  rw [BitVec.toInt_ofInt]
  exact Int.bmod_eq_of_le (by norm_num at h1 ⊢; omega) (by norm_num at h2 ⊢; omega)

private def prodAt {K : ℕ} (x w : Fin K → BitVec 8) (k : ℕ) : ℤ := (codeAt x k).toInt * (codeAt w k).toInt

private theorem sum_prodAt_bounds {K : ℕ} (x w : Fin K → BitVec 8) : ∀ m : ℕ,
    -(16384 * (m : ℤ)) ≤ ∑ k ∈ Finset.range m, prodAt x w k ∧ ∑ k ∈ Finset.range m, prodAt x w k ≤ 16384 * m
  | 0 => by simp
  | m + 1 => by
      have ih := sum_prodAt_bounds x w m
      have hb : -16384 ≤ prodAt x w m ∧ prodAt x w m ≤ 16384 := toInt_mul_bounds _ _
      rw [Finset.sum_range_succ]
      push_cast
      constructor <;> linarith [ih.1, ih.2, hb.1, hb.2]

theorem gemmAccSteps_toInt {K : ℕ} (x w : Fin K → BitVec 8) (hK : K < 2 ^ 17) : ∀ n, 32 * n ≤ K →
    (gemmAccSteps x w n).toInt = ∑ k ∈ Finset.range (32 * n), prodAt x w k
  | 0, _ => by simp [gemmAccSteps]
  | n + 1, h => by
      have ih := gemmAccSteps_toInt x w hK n (by omega)
      have e : (gemmAccSteps x w n).toInt +
          ∑ j : Fin 32, (codeAt x (32 * n + j.val)).toInt * (codeAt w (32 * n + j.val)).toInt =
            ∑ k ∈ Finset.range (32 * (n + 1)), prodAt x w k := by
        rw [ih, show 32 * (n + 1) = 32 * n + 32 by ring, Finset.sum_range_add]
        congr 1
      have hb := sum_prodAt_bounds x w (32 * (n + 1))
      have hm : ((32 * (n + 1) : ℕ) : ℤ) ≤ 131071 := by exact_mod_cast (show 32 * (n + 1) ≤ 131071 by omega)
      have lo : -2 ^ 31 ≤ ∑ k ∈ Finset.range (32 * (n + 1)), prodAt x w k := by linarith [hb.1]
      have hi : ∑ k ∈ Finset.range (32 * (n + 1)), prodAt x w k ≤ 2 ^ 31 - 1 := by linarith [hb.2]
      simp only [gemmAccSteps, immaS8]
      rw [e]
      unfold satS32
      rw [min_eq_left hi, max_eq_right lo]
      exact toInt_ofInt_of_range _ lo (by linarith)

theorem gemmAccS8_toInt {K : ℕ} (x w : Fin K → BitVec 8) (hK32 : 32 ∣ K) (hK : K < 2 ^ 17) :
    (gemmAccS8 x w).toInt = ∑ k, (x k).toInt * (w k).toInt := by
  have h := gemmAccSteps_toInt x w hK (K / 32) (Nat.mul_div_le K 32 |>.trans_eq' (by ring))
  rw [Nat.mul_div_cancel' hK32] at h
  unfold gemmAccS8
  rw [h, ← Fin.sum_univ_eq_sum_range]
  refine Finset.sum_congr rfl fun k _ => ?_
  simp [prodAt, codeAt, k.isLt]

theorem gemmAccS8_eq {K : ℕ} (x w : Fin K → BitVec 8) (hK32 : 32 ∣ K) (hK : K < 2 ^ 17) :
    gemmAccS8 x w = BitVec.ofInt 32 (∑ k, (x k).toInt * (w k).toInt) := by
  rw [← gemmAccS8_toInt x w hK32 hK, BitVec.ofInt_toInt]

theorem toInt_ofInt_dot {K : ℕ} (x w : Fin K → BitVec 8) (hK32 : 32 ∣ K) (hK : K < 2 ^ 17) :
    (BitVec.ofInt 32 (∑ k, (x k).toInt * (w k).toInt)).toInt = ∑ k, (x k).toInt * (w k).toInt := by
  rw [← gemmAccS8_eq x w hK32 hK, gemmAccS8_toInt x w hK32 hK]

theorem LaneIn.sel_length {K nw : ℕ} {P : Type} (x : LaneIn K nw P) : x.sel.length = 5 := by simp [LaneIn.sel]

theorem laneOk_spec {K nw : ℕ} {P : Type}
    (chain : P → Fin 32 → (Fin 4 → Fin K → BitVec 8) → (Fin 8 → Fin K → BitVec 8) → Fin nw → BitVec 32)
    (dqW : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 16) (x : LaneIn K nw P) (hK32 : 32 ∣ K) (hK : K < 2 ^ 17)
    (hok : laneOk chain dqW x = true) :
    (∀ t : Fin 11, x.lane ⟨5 + t.val, by omega⟩ = false) ∧
    (∀ s t, x.zr (laneRow x.l s) (laneCol x.l t) = BitVec.ofInt 32 (∑ k, (x.a s k).toInt * (x.b t k).toInt) ∧
      x.yr (laneRow x.l s) (laneCol x.l t) =
        dqW (BitVec.ofInt 32 (∑ k, (x.a s k).toInt * (x.b t k).toInt)) (x.sa s) (x.sw t)) ∧
    ∀ w, x.partials w = mask32 (chain x.pub x.l x.a x.b w) (tree orBit false (List.ofFn x.admit)) := by
  unfold laneOk at hok
  simp only at hok
  set high := tree orBit false (List.ofFn fun t : Fin 11 => x.lane ⟨5 + t.val, by omega⟩) with hhigh
  cases hh : high
  · rw [hh] at hok
    simp only [mux, Bool.false_eq_true, ↓reduceIte] at hok
    rw [tree_andBit] at hok
    have hz := hok _ List.mem_cons_self
    rw [zyCheck_iff] at hz
    refine ⟨?_, ?_, ?_⟩
    · have := (tree_orBit _).1 (hhigh ▸ hh)
      intro t
      exact this _ (List.mem_ofFn.2 ⟨t, rfl⟩)
    · intro s t
      have hc := hz s t
      rw [pick_ofFn _ _ x.sel_length] at hc
      rw [gemmAccS8_eq _ _ hK32 hK] at hc
      exact hc
    · intro w
      have := hok _ (List.mem_cons_of_mem _ (List.mem_ofFn.2 ⟨w, rfl⟩))
      simpa [eq32] using this
  · rw [hh] at hok
    simp [mux] at hok

end Pouw.H1Zy
