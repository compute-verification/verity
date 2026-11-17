module

public import Definitions.Pouw.H1Zy.LaneChunk
public import Mathlib.Algebra.GroupWithZero.Defs
import Mathlib.Data.Rat.Cast.Order
import Mathlib.Tactic.Linarith.Frontend
import Mathlib.Tactic.NormNum.Abs
import Mathlib.Tactic.NormNum.DivMod
import Mathlib.Tactic.NormNum.OfScientific
import Mathlib.Tactic.Ring.RingNF
import Proofs.Pouw.H1Zy.Combine
import Proofs.Pouw.H1Zy.Lane

/-! A lane's chunks, accepted at one carry chain, check the whole lane (`laneOk_of_chunks`). -/

@[expose] public section

namespace Pouw.H1Zy

def joinRows {n Kc : ℕ} (x : Fin n → Fin Kc → BitVec 8) : Fin (n * Kc) → BitVec 8 := fun k =>
  have hK : 0 < Kc := Nat.pos_of_ne_zero fun h => by have := k.isLt; simp [h] at this
  x ⟨k.val / Kc, (Nat.div_lt_iff_lt_mul hK).2 k.isLt⟩ ⟨k.val % Kc, Nat.mod_lt _ hK⟩

def lastFin (n : ℕ) (hn : 0 < n) : Fin n := ⟨n - 1, by omega⟩

def joinIn {n Kc nw : ℕ} {P : Type} (xs : Fin n → ChunkIn Kc nw P) (hn : 0 < n) : LaneIn (n * Kc) nw P where
  a := fun s => joinRows fun c => (xs c).a s
  sa := (xs ⟨0, hn⟩).sa
  b := fun t => joinRows fun c => (xs c).b t
  sw := (xs ⟨0, hn⟩).sw
  admit := (xs (lastFin n hn)).admit
  zr := (xs (lastFin n hn)).zr
  yr := (xs (lastFin n hn)).yr
  partials := (xs (lastFin n hn)).partials
  pub := (xs (lastFin n hn)).pub
  lane := (xs (lastFin n hn)).lane

theorem segOf_joinRows {n Kc : ℕ} (x : Fin n → Fin Kc → BitVec 8) (hKc : 32 ∣ Kc) (c : Fin n) {j : ℕ}
    (hj : j < Kc / 32) : segOf (joinRows x) (c.val * (Kc / 32) + j) = segOf (x c) j := by
  obtain ⟨q, rfl⟩ := hKc
  have hq : 32 * q / 32 = q := by omega
  rw [hq] at hj ⊢
  funext i
  have hi := i.isLt
  have hc := c.isLt
  have hlt' : 32 * j + i.val < 32 * q := by omega
  have key : 32 * (c.val * q + j) + i.val = (32 * j + i.val) + 32 * q * c.val := by ring
  have hlt : 32 * (c.val * q + j) + i.val < n * (32 * q) := by
    rw [key]
    have : c.val + 1 ≤ n := hc
    nlinarith
  have hpos : 0 < 32 * q := by omega
  simp only [segOf, codeAt, hlt, hlt', ↓reduceDIte, joinRows]
  congr 1
  · ext
    simp only [key, Nat.add_mul_div_left _ _ hpos, Nat.div_eq_of_lt hlt', zero_add]
  · ext
    simp only [key, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hlt']

theorem run_join {n Kc nw : ℕ} {P : Type} (M : ChainSteps P nw) (p : P) (l : Fin 32) (hKc : 32 ∣ Kc)
    (a : Fin n → Fin 4 → Fin Kc → BitVec 8) (b : Fin n → Fin 8 → Fin Kc → BitVec 8) (σ : ChainSt) (c : Fin n) :
    ∀ j, j ≤ Kc / 32 →
      M.run p l (c.val * (Kc / 32)) (a c) (b c)
          (M.run p l 0 (fun s => joinRows fun d => a d s) (fun t => joinRows fun d => b d t) σ (c.val * (Kc / 32))) j =
        M.run p l 0 (fun s => joinRows fun d => a d s) (fun t => joinRows fun d => b d t) σ (c.val * (Kc / 32) + j)
  | 0, _ => rfl
  | j + 1, hj => by
    rw [← Nat.add_assoc]
    simp only [ChainSteps.run]
    rw [run_join M p l hKc a b σ c j (by omega), zero_add]
    simp only [segOf_joinRows _ hKc c (by omega : j < Kc / 32)]

theorem zRun_join {n Kc : ℕ} (x w : Fin n → Fin Kc → BitVec 8) (hKc : 32 ∣ Kc) (z : BitVec 32) (c : Fin n) :
    ∀ j, j ≤ Kc / 32 →
      zRun (x c) (w c) (zRun (joinRows x) (joinRows w) z (c.val * (Kc / 32))) j =
        zRun (joinRows x) (joinRows w) z (c.val * (Kc / 32) + j)
  | 0, _ => rfl
  | j + 1, hj => by
    rw [← Nat.add_assoc]
    simp only [zRun]
    rw [zRun_join x w hKc z c j (by omega), segOf_joinRows x hKc c (by omega), segOf_joinRows w hKc c (by omega)]

theorem zRun_zero {K : ℕ} (x w : Fin K → BitVec 8) : ∀ m, zRun x w 0 m = gemmAccSteps x w m
  | 0 => rfl
  | m + 1 => by simp only [zRun, gemmAccSteps, zRun_zero x w m]; rfl

theorem mux_false_eq_true {b t : Bool} : mux b t false = true ↔ b = false ∧ t = true := by
  cases b <;> simp [mux]

theorem ChunkIn.l_eq {Kc nw : ℕ} {P : Type} {x y : ChunkIn Kc nw P} (h : x.lane = y.lane) : x.l = y.l :=
  Fin.ext (by simp only [ChunkIn.l, ChunkIn.sel, h])

theorem chunkOk_parts {Kc nw : ℕ} {P : Type} (M : ChainSteps P nw) (dqW : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 16)
    {n c : ℕ} {x : ChunkIn Kc nw P} (h : chunkOk M dqW n c x = true) :
    tree orBit false (List.ofFn fun t : Fin 11 => x.lane ⟨5 + t.val, by omega⟩) = false ∧
      (c ≠ 0 → x.sa = x.carryIn.sa ∧ x.sw = x.carryIn.sw) ∧
      (c + 1 ≠ n → x.carryOut = chunkCarry M c x) := by
  simp only [chunkOk] at h
  obtain ⟨hh, ht⟩ := mux_false_eq_true.1 h
  rw [tree_andBit] at ht
  refine ⟨hh, fun hc => ?_, fun hc => ?_⟩
  · simp only [hc, ↓reduceIte] at ht
    constructor
    · funext s
      have := ht (eq32 (x.sa s) (x.carryIn.sa s)) (List.mem_append_left _ (List.mem_append_left _ (List.mem_ofFn.2 ⟨s, rfl⟩)))
      simpa [eq32] using this
    · funext t
      have := ht (eq32 (x.sw t) (x.carryIn.sw t)) (List.mem_append_left _ (List.mem_append_right _ (List.mem_ofFn.2 ⟨t, rfl⟩)))
      simpa [eq32] using this
  · simp only [hc, ↓reduceIte] at ht
    simpa using ht _ (List.mem_append_right _ (List.mem_singleton_self _))

theorem chunkCarry_eq {n Kc nw : ℕ} {P : Type} (M : ChainSteps P nw)
    (dqW : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 16) (hn : 0 < n) (hKc : 32 ∣ Kc)
    (xs : Fin n → ChunkIn Kc nw P)
    (hpub : ∀ c, (xs c).pub = (xs (lastFin n hn)).pub) (hlane : ∀ c, (xs c).lane = (xs (lastFin n hn)).lane)
    (hcarry : ∀ (c : ℕ) (h : c + 1 < n), (xs ⟨c + 1, h⟩).carryIn = (xs ⟨c, by omega⟩).carryOut)
    (hok : ∀ c : Fin n, chunkOk M dqW n c.val (xs c) = true) :
    ∀ (c : ℕ) (hc : c < n), chunkCarry M c (xs ⟨c, hc⟩) =
      { chain := M.run (xs (lastFin n hn)).pub (xs (lastFin n hn)).l 0 (joinIn xs hn).a (joinIn xs hn).b
          (M.init (xs (lastFin n hn)).pub (xs (lastFin n hn)).l) ((c + 1) * (Kc / 32))
        z := fun s t => zRun ((joinIn xs hn).a s) ((joinIn xs hn).b t) 0 ((c + 1) * (Kc / 32))
        sa := (xs ⟨0, hn⟩).sa
        sw := (xs ⟨0, hn⟩).sw } := by
  intro c
  induction c with
  | zero =>
    intro hc
    have hp := hpub ⟨0, hc⟩
    have hl := ChunkIn.l_eq (hlane ⟨0, hc⟩)
    have hr := run_join M (xs (lastFin n hn)).pub (xs (lastFin n hn)).l hKc (fun d => (xs d).a) (fun d => (xs d).b)
      (M.init (xs (lastFin n hn)).pub (xs (lastFin n hn)).l) ⟨0, hc⟩ (Kc / 32) le_rfl
    simp only [zero_mul, ChainSteps.run, zero_add] at hr
    simp only [chunkCarry, ↓reduceIte, zero_mul, zero_add, one_mul, hp, hl, Carry.mk.injEq, and_true]
    refine ⟨hr, ?_⟩
    funext s t
    have hz := zRun_join (fun d => (xs d).a s) (fun d => (xs d).b t) hKc 0 ⟨0, hc⟩ (Kc / 32) le_rfl
    simp only [zero_mul, zRun, zero_add] at hz
    exact hz
  | succ c ih =>
    intro hc
    have hc' : c < n := by omega
    have ihc := ih hc'
    have hprev := (chunkOk_parts M dqW (hok ⟨c, hc'⟩)).2.2 (show c + 1 ≠ n by omega)
    have hin : (xs ⟨c + 1, hc⟩).carryIn = _ := (hcarry c hc).trans (hprev.trans ihc)
    have hsc := (chunkOk_parts M dqW (hok ⟨c + 1, hc⟩)).2.1 (show c + 1 ≠ 0 by omega)
    have hp := hpub ⟨c + 1, hc⟩
    have hl := ChunkIn.l_eq (hlane ⟨c + 1, hc⟩)
    have hr := run_join M (xs (lastFin n hn)).pub (xs (lastFin n hn)).l hKc (fun d => (xs d).a) (fun d => (xs d).b)
      (M.init (xs (lastFin n hn)).pub (xs (lastFin n hn)).l) ⟨c + 1, hc⟩ (Kc / 32) le_rfl
    have hk : (c + 1) * (Kc / 32) + Kc / 32 = (c + 1 + 1) * (Kc / 32) := by ring
    rw [hk] at hr
    simp only [chunkCarry, Nat.add_one_ne_zero, ↓reduceIte, hp, hl, hin, Carry.mk.injEq]
    refine ⟨hr, ?_, ?_, ?_⟩
    · funext s t
      have hz := zRun_join (fun d => (xs d).a s) (fun d => (xs d).b t) hKc 0 ⟨c + 1, hc⟩ (Kc / 32) le_rfl
      rw [hk] at hz
      exact hz
    · rw [hsc.1, hin]
    · rw [hsc.2, hin]

theorem laneOk_of_chunks {n Kc nw : ℕ} {P : Type} (M : ChainSteps P nw)
    (dqW : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 16) (hn : 0 < n) (hKc : 32 ∣ Kc)
    (xs : Fin n → ChunkIn Kc nw P)
    (hpub : ∀ c, (xs c).pub = (xs (lastFin n hn)).pub) (hlane : ∀ c, (xs c).lane = (xs (lastFin n hn)).lane)
    (hcarry : ∀ (c : ℕ) (h : c + 1 < n), (xs ⟨c + 1, h⟩).carryIn = (xs ⟨c, by omega⟩).carryOut)
    (hok : ∀ c : Fin n, chunkOk M dqW n c.val (xs c) = true) :
    laneOk (chainOf M) dqW (joinIn xs hn) = true := by
  have hlast := chunkCarry_eq M dqW hn hKc xs hpub hlane hcarry hok (n - 1) (by omega)
  have hn1 : n - 1 + 1 = n := by omega
  have hks : (n * Kc) / 32 = n * (Kc / 32) := Nat.mul_div_assoc n hKc
  have h := hok (lastFin n hn)
  simp only [lastFin] at h hlast
  simp only [chunkOk, hn1, ↓reduceIte] at h
  obtain ⟨hh, ht⟩ := mux_false_eq_true.1 h
  rw [tree_andBit] at ht
  rw [hlast] at ht
  simp only [hn1] at ht
  have hsa : (xs ⟨n - 1, by omega⟩).sa = (xs ⟨0, hn⟩).sa := by
    have := congrArg Carry.sa (chunkCarry_eq M dqW hn hKc xs hpub hlane hcarry hok (n - 1) (by omega))
    simpa [chunkCarry] using this
  have hsw : (xs ⟨n - 1, by omega⟩).sw = (xs ⟨0, hn⟩).sw := by
    have := congrArg Carry.sw (chunkCarry_eq M dqW hn hKc xs hpub hlane hcarry hok (n - 1) (by omega))
    simpa [chunkCarry] using this
  rw [hsa, hsw] at ht
  simp only [laneOk, chainOf, hks]
  refine mux_false_eq_true.2 ⟨hh, ?_⟩
  rw [tree_andBit]
  intro b hb
  apply ht
  refine List.mem_append_right _ ?_
  simpa only [zRun_zero, gemmAccS8, hks, joinIn, lastFin, LaneIn.sel, ChunkIn.sel, LaneIn.l, ChunkIn.l] using hb

theorem chunk_lt {n Kc : ℕ} (hn : 0 < n) (hK : n * Kc < 2 ^ 17) : Kc < 2 ^ 17 :=
  lt_of_le_of_lt (Nat.le_mul_of_pos_left Kc hn) hK

theorem chunks_spec {n Kc nw : ℕ} {P : Type} (M : ChainSteps P nw)
    (dqW : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 16) (hn : 0 < n) (hKc : 32 ∣ Kc) (hK : n * Kc < 2 ^ 17)
    (xs : Fin n → ChunkIn Kc nw P)
    (hpub : ∀ c, (xs c).pub = (xs (lastFin n hn)).pub) (hlane : ∀ c, (xs c).lane = (xs (lastFin n hn)).lane)
    (hcarry : ∀ (c : ℕ) (h : c + 1 < n), (xs ⟨c + 1, h⟩).carryIn = (xs ⟨c, by omega⟩).carryOut)
    (hok : ∀ c : Fin n, chunkOk M dqW n c.val (xs c) = true) :
    let y := joinIn xs hn
    (∀ t : Fin 11, y.lane ⟨5 + t.val, by omega⟩ = false) ∧
    (∀ s t, y.zr (laneRow y.l s) (laneCol y.l t) = BitVec.ofInt 32 (∑ k, (y.a s k).toInt * (y.b t k).toInt) ∧
      y.yr (laneRow y.l s) (laneCol y.l t) =
        dqW (BitVec.ofInt 32 (∑ k, (y.a s k).toInt * (y.b t k).toInt)) (y.sa s) (y.sw t)) ∧
    ∀ w, y.partials w = mask32 (chainOf M y.pub y.l y.a y.b w) (tree orBit false (List.ofFn y.admit)) :=
  laneOk_spec (chainOf M) dqW (joinIn xs hn) (Dvd.dvd.mul_left hKc n) hK
    (laneOk_of_chunks M dqW hn hKc xs hpub hlane hcarry hok)

end Pouw.H1Zy
