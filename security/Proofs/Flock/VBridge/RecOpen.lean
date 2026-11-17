module

public import Proofs.Flock.VBridge.Open
public import Proofs.Flock.VBridge.Residuals
public import Proofs.Flock.Soundness.Refine.Paths
import Mathlib.Algebra.Order.Floor.Semifield
import Mathlib.Algebra.Order.Ring.Star
import Mathlib.Data.List.GetD
import Mathlib.MeasureTheory.Integral.Bochner.Basic
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
import Mathlib.Tactic.ReduceModChar
import VCVio.Prelude.Core

/-! V*'s opening, `RecOpen_v3` or `RecOpen_v4`, computes the verifier's climb and the opening's consistency term (`sound_recOpen`). -/

@[expose] public section

namespace FlockVBridge

open Flock.HmNets hiding and
open Flock FlockSoundness.Discharge.Hm

def consStructure (LANES : ℕ) : GfS where
  ops := (List.range LANES).map (fun l => [Term.w l]) ++ (List.range (2 * LANES)).map fun i => [Term.v i]
  res := (List.range 2).map fun r =>
    { mul := (List.range LANES).map fun l => (LANES + r * LANES + l, l), add := [Term.w (LANES + r)] }

def recOpen (LANES H : ℕ) (row aux acc coef dirs : List Lin) : M (List Lin × List Lin) := do
  let t ← residualForms (consStructure LANES) (row ++ acc.take 256) coef
  let out ← climbRow H row aux (dirs.take 32)
  pure (out, t ++ acc.drop 256)

def consTerm (LANES : ℕ) (r a c : Array F128) (k : ℕ) : F128 :=
  ((List.range LANES).map fun l => c[k * LANES + l]! * r[l]!).foldl (· + ·) a[k]!

theorem bitL_toBytes (x : F128) {t : ℕ} (ht : t < 128) : bitL x.toBytes.data.toList t = bitOf x t := by
  rw [bitL, F128.toBytes, Array.toList_ofFn, List.getD_eq_getElem?_getD, List.getElem?_ofFn, dite_eq_left (by omega),
    Option.getD_some, bitOf, FlockLevel3.h_bit]
  by_cases h : t < 64
  · simp only [show t / 8 < 8 by omega, h, ↓reduceIte, UInt64.toNat_toUInt8, UInt64.toNat_shiftRight,
      FlockLevel3.toNat_toUInt64 (8 * (t / 8)) (by omega), Nat.mod_eq_of_lt (by omega : 8 * (t / 8) < 64)]
    rw [Nat.testBit_mod_two_pow, Nat.testBit_shiftRight, show 8 * (t / 8) + t % 8 = t by omega]
    simp [show t % 8 < 8 by omega]
  · simp only [show ¬ t / 8 < 8 by omega, h, ↓reduceIte, UInt64.toNat_toUInt8, UInt64.toNat_shiftRight,
      FlockLevel3.toNat_toUInt64 (8 * (t / 8 - 8)) (by omega), Nat.mod_eq_of_lt (by omega : 8 * (t / 8 - 8) < 64)]
    rw [Nat.testBit_mod_two_pow, Nat.testBit_shiftRight, show 8 * (t / 8 - 8) + t % 8 = t - 64 by omega]
    simp [show t % 8 < 8 by omega]

theorem bitL_colBytes : ∀ (L : List F128) (t : ℕ),
    bitL (FlockSoundness.Refine.colBytes L) t = FlockVBridge.wbit (fun i => L.getD i 0) t
  | [], t => by
    simp only [FlockSoundness.Refine.colBytes, List.flatMap_nil, bitL, List.getD_nil, FlockVBridge.wbit, bitOf,
      show (0 : F128) = ⟨0, 0⟩ from rfl, FlockLevel3.w128, UInt64.toNat_zero, mul_zero, add_zero]
    simp
  | x :: L, t => by
    have hx : x.toBytes.data.toList.length = 16 := FlockSoundness.Refine.toBytes_length x
    rw [FlockSoundness.Refine.colBytes, List.flatMap_cons, ← FlockSoundness.Refine.colBytes]
    by_cases ht : t < 128
    · rw [bitL_append_left _ (by omega), bitL_toBytes x ht, FlockVBridge.wbit, Nat.div_eq_of_lt ht, Nat.mod_eq_of_lt ht]
      rfl
    · rw [bitL_append_right _ (by omega), hx, bitL_colBytes L, FlockVBridge.wbit, FlockVBridge.wbit,
        show t / 128 = (t - 8 * 16) / 128 + 1 by omega, show t % 128 = (t - 8 * 16) % 128 by omega]
      rfl

theorem bitL_rowBytes (r : Array F128) (t : ℕ) :
    bitL (MerkleScheme.rowBytes r).data.toList t = FlockVBridge.wbit (fun i => r.toList.getD i 0) t := by
  rw [FlockSoundness.Refine.rowBytes_data, bitL_colBytes]

theorem bitL_rowBytes_append (r a : Array F128) (t : ℕ) :
    bitL (MerkleScheme.rowBytes r ++ MerkleScheme.rowBytes a).data.toList t =
      FlockVBridge.wbit (fun i => (r.toList ++ a.toList).getD i 0) t := by
  rw [ByteArray.toList_data_append, FlockSoundness.Refine.rowBytes_data, FlockSoundness.Refine.rowBytes_data,
    show FlockSoundness.Refine.colBytes r.toList ++ FlockSoundness.Refine.colBytes a.toList =
      FlockSoundness.Refine.colBytes (r.toList ++ a.toList) by simp [FlockSoundness.Refine.colBytes, List.flatMap_append],
    bitL_colBytes]

theorem size_rowBytes (r : Array F128) : (MerkleScheme.rowBytes r).size = 16 * r.size := by
  have h : ∀ L : List F128, (FlockSoundness.Refine.colBytes L).length = 16 * L.length := by
    intro L
    induction L with
    | nil => rfl
    | cons x L ih =>
      rw [FlockSoundness.Refine.colBytes, List.flatMap_cons, ← FlockSoundness.Refine.colBytes, List.length_append,
        FlockSoundness.Refine.toBytes_length, ih, List.length_cons]
      ring
  rw [← Array.length_toList, ← ByteArray.size_data, ← Array.length_toList, FlockSoundness.Refine.rowBytes_data, h]

theorem getD_toList (a : Array F128) (i : ℕ) : a.toList.getD i 0 = a[i]! := by
  by_cases h : i < a.size
  · rw [List.getD_eq_getElem _ _ (by simpa using h), Array.getElem_toList, getElem!_pos a i h]
  · rw [List.getD_eq_default _ _ (by simp; omega), getElem!_neg a i h]
    rfl

theorem bit_eq_wbit {z : Var → Bool} {W : List Lin} {b : ByteArray} {w : ℕ → F128} (h : Carries z W b)
    (hb : ∀ t, bitL b.data.toList t = FlockVBridge.wbit w t) (q : ℕ) : bit z W q = FlockVBridge.wbit w q := by
  by_cases hq : q < W.length
  · rw [h.2 q hq, hb]
  · have hl : b.data.toList.length = b.size := Array.length_toList
    rw [← hb, bit, List.getElem?_eq_none (by omega), Option.getD_none, ev_zero, bitL,
      List.getD_eq_default _ _ (by have := h.1; omega)]
    simp

theorem carries_of_wbit {z : Var → Bool} {W : List Lin} {b : ByteArray} {w : ℕ → F128} (hl : W.length = 8 * b.size)
    (h : ∀ q < W.length, bit z W q = FlockVBridge.wbit w q) (hb : ∀ t, bitL b.data.toList t = FlockVBridge.wbit w t) :
    Carries z W b :=
  ⟨hl, fun q hq => by rw [h q hq, hb]⟩

theorem carries_split {z : Var → Bool} {w : List Lin} {b b' : ByteArray} (h : Carries z w (b ++ b')) :
    Carries z (w.take (8 * b.size)) b ∧ Carries z (w.drop (8 * b.size)) b' := by
  obtain ⟨hl, hb⟩ := h
  rw [ByteArray.size_append] at hl
  have hbl : b.data.toList.length = b.size := Array.length_toList
  refine ⟨⟨by simp [hl], fun t ht => ?_⟩, ⟨by simp [hl]; omega, fun t ht => ?_⟩⟩
  · simp only [List.length_take] at ht
    have := hb t (by omega)
    rw [ByteArray.toList_data_append, bitL_append_left _ (by omega)] at this
    rw [bit, List.getElem?_take, ite_eq_left_iff.2 (fun h' => absurd (by omega) h'), ← bit, this]
  · simp only [List.length_drop] at ht
    have := hb (8 * b.size + t) (by omega)
    rw [ByteArray.toList_data_append, bitL_append_right _ (by omega), hbl, Nat.add_sub_cancel_left] at this
    rw [bit, List.getElem?_drop, ← bit, this]

theorem zero_add_f128 (x : F128) : 0 + x = x := by
  cases x
  show F128.add ⟨0, 0⟩ _ = _
  simp [F128.add]

theorem length_consStructure_res (LANES : ℕ) : (consStructure LANES).res.length = 2 := by
  rw [consStructure, List.length_map, List.length_range]

theorem consStructure_res (LANES : ℕ) {k : ℕ} (hk : k < (consStructure LANES).res.length) :
    (consStructure LANES).res[k] =
      { mul := (List.range LANES).map fun l => (LANES + k * LANES + l, l), add := [Term.w (LANES + k)] } := by
  simp only [consStructure, List.getElem_map, List.getElem_range]

theorem consStructure_ops_w (LANES : ℕ) {l : ℕ} (hl : l < LANES) : (consStructure LANES).ops.getD l [] = [Term.w l] := by
  rw [consStructure, List.getD_eq_getElem _ _ (by simp; omega), List.getElem_append_left (by simpa using hl),
    List.getElem_map, List.getElem_range]

theorem consStructure_ops_v (LANES : ℕ) {i : ℕ} (hi : i < 2 * LANES) :
    (consStructure LANES).ops.getD (LANES + i) [] = [Term.v i] := by
  rw [consStructure, List.getD_eq_getElem _ _ (by simp; omega), List.getElem_append_right (by simp)]
  simp only [List.length_map, List.length_range, Nat.add_sub_cancel_left, List.getElem_map, List.getElem_range]

theorem foldl_add_single (x : F128) : [x].foldl (· + ·) 0 = x := zero_add_f128 x

theorem opVal_cons_w (LANES : ℕ) (w v : ℕ → F128) {l : ℕ} (hl : l < LANES) :
    opVal (consStructure LANES) (termVal w v) l = w l := by
  rw [opVal, consStructure_ops_w LANES hl]
  simp only [List.map_cons, List.map_nil, termVal]
  exact foldl_add_single _

theorem opVal_cons_v (LANES : ℕ) (w v : ℕ → F128) {i : ℕ} (hi : i < 2 * LANES) :
    opVal (consStructure LANES) (termVal w v) (LANES + i) = v i := by
  rw [opVal, consStructure_ops_v LANES hi]
  simp only [List.map_cons, List.map_nil, termVal]
  exact foldl_add_single _

/-- `F` stays opaque, so no step unfolds a product. -/
theorem map_mul_congr {α : Type} (g : α → ℕ × ℕ) (F : ℕ → F128) (u v : α → F128) :
    ∀ L : List α, (∀ x ∈ L, F (g x).1 = u x) → (∀ x ∈ L, F (g x).2 = v x) →
      (L.map g).map (fun ab => F ab.1 * F ab.2) = L.map fun x => u x * v x
  | [], _, _ => rfl
  | x :: L, hu, hv => by
    rw [List.map_cons, List.map_cons, List.map_cons, hu x List.mem_cons_self, hv x List.mem_cons_self,
      map_mul_congr g F u v L (fun y hy => hu y (List.mem_cons_of_mem _ hy)) fun y hy => hv y (List.mem_cons_of_mem _ hy)]

theorem resVal_consStructure (LANES : ℕ) (r a c : Array F128) (hr : r.size = LANES) (k : ℕ)
    (hk : k < (consStructure LANES).res.length) :
    resVal (consStructure LANES) (termVal (fun i => (r.toList ++ a.toList).getD i 0) (fun i => c.toList.getD i 0))
      (consStructure LANES).res[k] = consTerm LANES r a c k := by
  have hk2 : k < 2 := by rwa [length_consStructure_res] at hk
  have hrl : r.toList.length = LANES := by simpa using hr
  have hacc : (r.toList ++ a.toList).getD (LANES + k) 0 = a[k]! := by
    rw [List.getD_eq_getElem?_getD, List.getElem?_append_right (by omega), hrl, Nat.add_sub_cancel_left,
      ← List.getD_eq_getElem?_getD, getD_toList]
  have hu : ∀ l ∈ List.range LANES, opVal (consStructure LANES)
      (termVal (fun i => (r.toList ++ a.toList).getD i 0) (fun i => c.toList.getD i 0))
      ((fun l => (LANES + k * LANES + l, l)) l).1 = c[k * LANES + l]! := fun l hl => by
    rw [List.mem_range] at hl
    have hkl : k * LANES + l < 2 * LANES := by
      have : k * LANES ≤ 1 * LANES := Nat.mul_le_mul_right _ (by omega)
      omega
    show opVal _ _ (LANES + k * LANES + l) = _
    rw [show LANES + k * LANES + l = LANES + (k * LANES + l) by omega]
    exact (opVal_cons_v LANES _ _ hkl).trans (getD_toList c _)
  have hv : ∀ l ∈ List.range LANES, opVal (consStructure LANES)
      (termVal (fun i => (r.toList ++ a.toList).getD i 0) (fun i => c.toList.getD i 0))
      ((fun l => (LANES + k * LANES + l, l)) l).2 = r[l]! := fun l hl => by
    rw [List.mem_range] at hl
    show opVal _ _ l = _
    refine (opVal_cons_w LANES _ _ hl).trans ?_
    show (r.toList ++ a.toList).getD l 0 = r[l]!
    rw [List.getD_eq_getElem?_getD, List.getElem?_append_left (by omega), ← List.getD_eq_getElem?_getD, getD_toList]
  rw [consStructure_res LANES hk, resVal, consTerm]
  simp only [List.map_cons, List.map_nil, termVal]
  rw [foldl_add_single, hacc, map_mul_congr _ _ (fun l => c[k * LANES + l]!) (fun l => r[l]!) _ hu hv]

theorem sound_recOpen (ms : MerkleScheme) (hms : ∀ l r, ms.node l r = Sha512.hash (l ++ r)) (LANES H : ℕ) (hH : H ≤ 32)
    (row aux acc coef dirs : List Lin) :
    Sound (recOpen LANES H row aux acc coef dirs) fun z o =>
      ∀ (r a c : Array F128) (rest : ByteArray) (idx : ℕ) (ss : List ByteArray),
        r.size = LANES → a.size = 2 →
        Carries z row (MerkleScheme.rowBytes r) → Carries z acc (MerkleScheme.rowBytes a ++ rest) →
        Carries z coef (MerkleScheme.rowBytes c) →
        List.Forall₂ (fun w b => b.size = 64 ∧ Carries z w b) (sibForms aux H) ss →
        (∀ h < H, ev z (dirs.getD h Lin.zero) = idx.testBit h) →
        Carries z o.1 (Flock.climb ms (Sha512.hash (MerkleScheme.rowBytes r)) idx ss ++ ⟨Array.replicate 64 0⟩) ∧
        Carries z o.2 (MerkleScheme.rowBytes #[consTerm LANES r a c 0, consTerm LANES r a c 1] ++ rest) := by
  unfold recOpen
  refine Sound.bind' (sound_residualForms _ _ _) fun t => Sound.bind' (sound_climbRow ms hms H row aux _) fun out =>
    Sound.pure' fun z hout ht r a c rest idx ss hr ha hrow hacc hcoef hss hdir => ⟨?_, ?_⟩
  · exact hout _ idx ss hrow hss fun h hh => by rw [getD_take_of_lt (by omega)]; exact hdir h hh
  · obtain ⟨htl, htv⟩ := ht
    obtain ⟨hat, had⟩ := carries_split hacc
    rw [size_rowBytes, ha, show 8 * (16 * 2) = 256 from rfl] at hat had
    have hres := htv _ _ (bit_eq_wbit (carries_append hrow hat) (bitL_rowBytes_append r a))
      (bit_eq_wbit hcoef (bitL_rowBytes c))
    rw [length_consStructure_res] at htl
    refine carries_append (carries_of_wbit ?_ (fun q hq => ?_) (bitL_rowBytes _)) had
    · rw [htl, size_rowBytes]
      rfl
    · rw [htl] at hq
      have h := hres (q / 128) (by rw [length_consStructure_res]; omega) (q % 128) (by omega)
      rw [show 128 * (q / 128) + q % 128 = q by omega, resVal_consStructure LANES r a c hr] at h
      rw [h]
      simp only [FlockVBridge.wbit]
      rcases (by omega : q / 128 = 0 ∨ q / 128 = 1) with h0 | h1
      · rw [h0, List.getD_cons_zero]
      · rw [h1, List.getD_cons_succ, List.getD_cons_zero]

end FlockVBridge
