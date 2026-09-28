/-
The executable's mask `Hm96.Default512.mask` (per-byte tables of `M`'s columns, built by loops) is `maskV`, bit for bit:
bit `t < 512` of `M·y` is `⊕_{j < 1536} key_{t+j} · y_j`.
-/
import Proofs.Flock.Soundness.Discharge.Hm.Slot

namespace FlockSoundness.Discharge.Hm

open Flock.HmNets hiding and

open Flock

/-! ## Sums over ℤ/2 as Boolean folds -/

/-- `⊕_{j ∈ l} f j`. -/
def xsum (l : List ℕ) (f : ℕ → Bool) : Bool := l.foldl (fun acc j => acc ^^ f j) false

theorem xsum_nil (f : ℕ → Bool) : xsum [] f = false := rfl

theorem xsum_cons (a : ℕ) (l : List ℕ) (f : ℕ → Bool) : xsum (a :: l) f = (f a ^^ xsum l f) := by
  unfold xsum
  rw [List.foldl_cons, Bool.false_xor, foldl_xor_shift]

theorem xsum_append (l l' : List ℕ) (f : ℕ → Bool) : xsum (l ++ l') f = (xsum l f ^^ xsum l' f) := by
  unfold xsum
  rw [List.foldl_append, foldl_xor_shift _ l']

theorem xsum_congr {l : List ℕ} {f g : ℕ → Bool} (h : ∀ j ∈ l, f j = g j) : xsum l f = xsum l g :=
  List.foldl_ext _ _ _ fun a j hj => by rw [h j hj]

theorem xsum_xor (l : List ℕ) (f g : ℕ → Bool) : xsum l (fun j => f j ^^ g j) = (xsum l f ^^ xsum l g) := by
  induction l with
  | nil => rfl
  | cons a l ih =>
    rw [xsum_cons, xsum_cons, xsum_cons, ih]
    cases f a <;> cases g a <;> cases xsum l f <;> cases xsum l g <;> rfl

theorem xsum_false (l : List ℕ) : xsum l (fun _ => false) = false := by
  induction l with
  | nil => rfl
  | cons a l ih => rw [xsum_cons, ih]; rfl

theorem xsum_single (x : Bool) : ∀ (n : ℕ) {L : ℕ}, L < n → xsum (List.range n) (fun s => decide (s = L) && x) = x
  | 0, _, h => absurd h (Nat.not_lt_zero _)
  | n + 1, L, h => by
    rw [List.range_succ, xsum_append, xsum_cons, xsum_nil, Bool.xor_false]
    by_cases hL : L < n
    · rw [xsum_single x n hL, show decide (n = L) = false from decide_eq_false (by omega)]
      simp
    · have hL' : L = n := by omega
      subst hL'
      rw [xsum_congr (g := fun _ => false) fun s hs => by
        simp only [List.mem_range] at hs
        simp [show s ≠ L by omega], xsum_false]
      simp

theorem xsum_map (l : List ℕ) (h : ℕ → ℕ) (f : ℕ → Bool) : xsum (l.map h) f = xsum l fun j => f (h j) := by
  unfold xsum
  rw [List.foldl_map]

theorem xsum_range_mul (f : ℕ → Bool) (m : ℕ) : ∀ n : ℕ,
    xsum (List.range (m * n)) f = xsum (List.range n) fun p => xsum (List.range m) fun s => f (m * p + s)
  | 0 => rfl
  | n + 1 => by
    rw [Nat.mul_succ, List.range_add, xsum_append, xsum_range_mul f m n, List.range_succ, xsum_append, xsum_cons,
      xsum_nil, Bool.xor_false, xsum_map]

/-! ## Arrays and words, bit by bit -/

theorem get!_set! {a : Array UInt64} {i j : ℕ} {x : UInt64} (hj : j < a.size) :
    (a.set! i x)[j]! = if i = j then x else a[j]! := by
  rw [Array.set!_eq_setIfInBounds, getElem!_pos (a.setIfInBounds i x) j (by simpa using hj),
    Array.getElem_setIfInBounds, getElem!_pos a j hj]

theorem size_set! (a : Array UInt64) (i : ℕ) (x : UInt64) : (a.set! i x).size = a.size := by
  rw [Array.set!_eq_setIfInBounds, Array.size_setIfInBounds]

theorem get!_replicate {n j : ℕ} (hj : j < n) : (Array.replicate n (0 : UInt64))[j]! = 0 := by
  rw [getElem!_pos _ j (by simpa using hj), Array.getElem_replicate]

theorem testBit_or (x y : UInt64) (b : ℕ) : (x ||| y).toNat.testBit b = (x.toNat.testBit b || y.toNat.testBit b) := by
  rw [UInt64.toNat_or, Nat.testBit_or]

theorem testBit_xor (x y : UInt64) (b : ℕ) : (x ^^^ y).toNat.testBit b = (x.toNat.testBit b ^^ y.toNat.testBit b) := by
  rw [UInt64.toNat_xor, Nat.testBit_xor]

theorem testBit_shl (x : UInt64) {s b : ℕ} (hs : s < 64) (hb : b < 64) :
    (x <<< s.toUInt64).toNat.testBit b = (decide (s ≤ b) && x.toNat.testBit (b - s)) := by
  rw [UInt64.toNat_shiftLeft, Nat.testBit_mod_two_pow, Nat.testBit_shiftLeft]
  have : s.toUInt64.toNat % 64 = s := by
    rw [Nat.toUInt64_eq, UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega)]
  rw [this]
  simp [hb]

/-! ## The key's bits -/

/-- Key bit `t` (`Hm96.bit`). -/
abbrev kb (t : ℕ) : Bool := Hm96.bit Hm96.Default512.key t

theorem kb_eq (t : ℕ) : kb t = (Hm96.Default512.key.get! (t / 8)).toNat.testBit (t % 8) := by
  simp only [kb, Hm96.bit, Nat.testBit_eq_decide_div_mod_eq, Nat.shiftRight_eq_div_pow]
  generalize (Hm96.Default512.key.get! (t / 8)).toNat / 2 ^ (t % 8) % 2 = r
  by_cases h : r = 1 <;> simp [h]

/-- `Hm96.Default512.keyBit`. -/
def keyBitU (t : ℕ) : UInt64 := ((Hm96.Default512.key.get! (t / 8)).toUInt64 >>> (t % 8).toUInt64) &&& 1

theorem testBit_keyBitU (t b : ℕ) : (keyBitU t).toNat.testBit b = (decide (b = 0) && kb t) := by
  rw [keyBitU, UInt64.toNat_and, Nat.testBit_and, UInt64.toNat_shiftRight, Nat.testBit_shiftRight, kb_eq]
  have h1 : (t % 8).toUInt64.toNat % 64 = t % 8 := by
    rw [Nat.toUInt64_eq, UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega)]
  rw [h1, UInt8.toNat_toUInt64]
  by_cases hb : b = 0
  · subst hb
    simp
  · have : Nat.testBit 1 b = false := by
      rw [show (1 : ℕ) = 2 ^ 0 from rfl, Nat.testBit_two_pow]
      simp [Ne.symm hb]
    rw [show (1 : UInt64).toNat = 1 from rfl, this]
    simp [hb]

/-! ## Loops as folds -/

theorem foldl_range'_induct {α : Type} (P : ℕ → α → Prop) (f : α → ℕ → α) (s : ℕ) :
    ∀ (n : ℕ) (a : α), P s a → (∀ i a, s ≤ i → i < s + n → P i a → P (i + 1) (f a i)) →
      P (s + n) ((List.range' s n).foldl f a)
  | 0, a, h, _ => h
  | n + 1, a, h, hstep => by
    rw [List.range'_concat, List.foldl_append, List.foldl_cons, List.foldl_nil, Nat.one_mul, ← Nat.add_assoc]
    exact hstep (s + n) _ (by omega) (by omega)
      (foldl_range'_induct P f s n a h fun i a h1 h2 h3 => hstep i a h1 (by omega) h3)

def colsStep (j : ℕ) (out : Array UInt64) (i : ℕ) : Array UInt64 :=
  out.set! (8 * j + i / 64) (out[8 * j + i / 64]! ||| keyBitU (i + j) <<< (i % 64).toUInt64)

theorem cols_eq : Hm96.Default512.cols =
    (List.range' 0 1536).foldl (fun out j => (List.range' 0 512).foldl (colsStep j) out)
      (Array.replicate (8 * 1536) 0) := by
  simp only [Hm96.Default512.cols, Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl,
    bind_pure_comp, map_pure, pure_bind, Id.run_pure, Std.Legacy.Range.size, Nat.reduceSub, Nat.reduceAdd, Nat.sub_zero,
    Nat.div_one]
  rfl

def tabStep (p v : ℕ) (t : Array UInt64) (k : ℕ) : Array UInt64 :=
  t.set! (8 * (256 * p + v) + k)
    (t[8 * (256 * p + (v &&& v - 1)) + k]! ^^^ Hm96.Default512.cols[8 * (8 * p + (v &&& 256 - v).log2) + k]!)

theorem tables_eq : Hm96.Default512.tables =
    (List.range' 0 192).foldl (fun t p => (List.range' 1 255).foldl (fun t v => (List.range' 0 8).foldl (tabStep p v) t) t)
      (Array.replicate (8 * 256 * 192) 0) := by
  simp only [Hm96.Default512.tables, Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl,
    bind_pure_comp, map_pure, pure_bind, Id.run_pure, Std.Legacy.Range.size, Nat.reduceSub, Nat.reduceAdd, Nat.sub_zero,
    Nat.div_one]
  rfl

def accStep (salt : ByteArray) (p : ℕ) (acc : Array UInt64) (k : ℕ) : Array UInt64 :=
  acc.set! k (acc[k]! ^^^ Hm96.Default512.tables[8 * (256 * p + (salt.get! p).toNat) + k]!)

def accOf (salt : ByteArray) : Array UInt64 :=
  (List.range' 0 192).foldl (fun acc p => (List.range' 0 8).foldl (accStep salt p) acc) (Array.replicate 8 0)

theorem mask_eq (salt : ByteArray) : Hm96.Default512.mask salt =
    (List.range' 0 8).foldl
      (fun out k => (List.range' 0 8).foldl (fun out j => out.push ((accOf salt)[k]! >>> (8 * j).toUInt64 &&& 255).toUInt8) out)
      (ByteArray.emptyWithCapacity 64) := by
  simp only [Hm96.Default512.mask, Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl,
    bind_pure_comp, map_pure, pure_bind, Id.run_pure, Std.Legacy.Range.size, Nat.reduceSub, Nat.reduceAdd, Nat.sub_zero,
    Nat.div_one]
  rfl

/-! ## `cols`: column `j` of `M` -/

/-- After the first `N` of `cols`' writes (in loop order, write `512 j + i` sets bit `i % 64` of limb `i / 64` of column
`j`), bit `b` of entry `c` is key bit `64 (c % 8) + b + c / 8` if written, else 0. -/
def ColsInv (N : ℕ) (out : Array UInt64) : Prop :=
  out.size = 8 * 1536 ∧ ∀ c < 8 * 1536, ∀ b < 64,
    (out[c]!).toNat.testBit b = (decide (512 * (c / 8) + 64 * (c % 8) + b < N) && kb (64 * (c % 8) + b + c / 8))

theorem colsInv_step {j i : ℕ} (hi : i < 512) {out : Array UInt64} (h : ColsInv (512 * j + i) out) :
    ColsInv (512 * j + i + 1) (colsStep j out i) := by
  obtain ⟨hs, hb⟩ := h
  refine ⟨by rw [colsStep, size_set!, hs], fun c hc b hb' => ?_⟩
  rw [colsStep, get!_set! (by rw [hs]; exact hc)]
  split_ifs with hcc
  · subst hcc
    rw [testBit_or, hb _ hc b hb', testBit_shl _ (Nat.mod_lt _ (by norm_num)) hb', testBit_keyBitU,
      show (8 * j + i / 64) / 8 = j by omega, show (8 * j + i / 64) % 8 = i / 64 by omega]
    by_cases h1 : b < i % 64
    · simp [show 512 * j + 64 * (i / 64) + b < 512 * j + i by omega,
        show 512 * j + 64 * (i / 64) + b < 512 * j + i + 1 by omega, show ¬ (i % 64 ≤ b) by omega]
    · by_cases h2 : b = i % 64
      · subst h2
        simp [show ¬ (512 * j + 64 * (i / 64) + i % 64 < 512 * j + i) by omega,
          show 512 * j + 64 * (i / 64) + i % 64 < 512 * j + i + 1 by omega, show 64 * (i / 64) + i % 64 = i by omega]
      · simp [show ¬ (512 * j + 64 * (i / 64) + b < 512 * j + i) by omega,
          show ¬ (512 * j + 64 * (i / 64) + b < 512 * j + i + 1) by omega, show b - i % 64 ≠ 0 by omega]
  · rw [hb c hc b hb']
    have : 512 * (c / 8) + 64 * (c % 8) + b ≠ 512 * j + i := by omega
    by_cases h3 : 512 * (c / 8) + 64 * (c % 8) + b < 512 * j + i
    · simp [h3, show 512 * (c / 8) + 64 * (c % 8) + b < 512 * j + i + 1 by omega]
    · simp [h3, show ¬ (512 * (c / 8) + 64 * (c % 8) + b < 512 * j + i + 1) by omega]

theorem colsInv_final : ColsInv (512 * 1536) Hm96.Default512.cols := by
  rw [cols_eq]
  have h0 : ColsInv (512 * 0) (Array.replicate (8 * 1536) 0) := by
    refine ⟨Array.size_replicate .., fun c hc b _ => ?_⟩
    rw [get!_replicate hc]
    simp
  have := foldl_range'_induct (fun j out => ColsInv (512 * j) out)
    (fun out j => (List.range' 0 512).foldl (colsStep j) out) 0 1536 _ h0 fun j out _ hj h => by
      have hi := foldl_range'_induct (fun i o => ColsInv (512 * j + i) o) (colsStep j) 0 512 out
        (by simpa using h) fun i o _ hi h => colsInv_step (by omega) h
      rw [Nat.zero_add] at hi
      rwa [Nat.mul_succ]
  simpa using this

/-- **`cols`**: bit `b` of limb `l` of column `j` is key bit `64 l + b + j`. -/
theorem cols_bit {j l b : ℕ} (hj : j < 1536) (hl : l < 8) (hb : b < 64) :
    (Hm96.Default512.cols[8 * j + l]!).toNat.testBit b = kb (64 * l + b + j) := by
  rw [colsInv_final.2 _ (by omega) b hb, show (8 * j + l) / 8 = j by omega, show (8 * j + l) % 8 = l by omega]
  simp [show 512 * j + 64 * l + b < 512 * 1536 by omega]

/-! ## `tables`: the XOR of `M`'s columns over a byte's set bits -/

/-- Bit `b` of limb `k` of table entry `256 p + v`: `⊕_{s < 8} v_s · key_{64k + b + 8p + s}`. -/
def tabB (p v k b : ℕ) : Bool := xsum (List.range 8) fun s => v.testBit s && kb (64 * k + b + (8 * p + s))

theorem tabB_zero (p k b : ℕ) : tabB p 0 k b = false := by
  rw [tabB, xsum_congr (g := fun _ => false) fun s _ => by simp, xsum_false]

/-- A byte's lowest set bit: `v &&& (256 - v)` is `2^L`, and `v &&& (v - 1)` is `v` without bit `L`. -/
theorem lowBit_facts : ∀ v < 256, 0 < v → ∃ L < 8, v &&& 256 - v = 2 ^ L ∧
    ∀ s < 8, v.testBit s = ((v &&& v - 1).testBit s ^^ decide (s = L)) := by
  decide +kernel

theorem tabB_rec {p v k b L : ℕ} (hL : L < 8) (h : ∀ s < 8, v.testBit s = ((v &&& v - 1).testBit s ^^ decide (s = L))) :
    tabB p v k b = (tabB p (v &&& v - 1) k b ^^ kb (64 * k + b + (8 * p + L))) := by
  unfold tabB
  rw [xsum_congr (g := fun s => ((v &&& v - 1).testBit s && kb (64 * k + b + (8 * p + s))) ^^
      (decide (s = L) && kb (64 * k + b + (8 * p + L)))) fun s hs => by
    simp only [List.mem_range] at hs
    rw [h s hs]
    by_cases hsL : s = L
    · subst hsL
      cases (v &&& v - 1).testBit s <;> simp
    · simp [hsL], xsum_xor, xsum_single _ 8 hL]

/-- After the writes before `N` (write `8 (256 p + v) + k` sets limb `k` of entry `256 p + v`), bit `b` of limb `k` of
entry `256 p + v` is `tabB p v k b` if written, else 0. -/
def TabInv (N : ℕ) (t : Array UInt64) : Prop :=
  t.size = 8 * 256 * 192 ∧ ∀ p < 192, ∀ v < 256, ∀ k < 8, ∀ b < 64,
    (t[8 * (256 * p + v) + k]!).toNat.testBit b = (decide (8 * (256 * p + v) + k < N) && tabB p v k b)

theorem tabInv_step {p v k : ℕ} (hp : p < 192) (hv1 : 1 ≤ v) (hv : v < 256) (hk : k < 8) {t : Array UInt64}
    (h : TabInv (8 * (256 * p + v) + k) t) : TabInv (8 * (256 * p + v) + k + 1) (tabStep p v t k) := by
  obtain ⟨hs, hb⟩ := h
  refine ⟨by rw [tabStep, size_set!, hs], fun p' hp' v' hv' k' hk' b hb' => ?_⟩
  rw [tabStep, get!_set! (by rw [hs]; omega)]
  split_ifs with he
  · have e1 : p' = p := by omega
    have e2 : v' = v := by omega
    have e3 : k' = k := by omega
    subst e1 e2 e3
    obtain ⟨L, hL, hL1, hL2⟩ := lowBit_facts v' hv (by omega)
    have hlt : v' &&& v' - 1 < v' := lt_of_le_of_lt Nat.and_le_right (by omega)
    rw [testBit_xor, hb p' hp' _ (by omega) k' hk' b hb', hL1, Nat.log2_two_pow, cols_bit (by omega) hk' hb',
      tabB_rec hL hL2]
    simp [show 8 * (256 * p' + (v' &&& v' - 1)) + k' < 8 * (256 * p' + v') + k' by omega]
  · rw [hb p' hp' v' hv' k' hk' b hb']
    by_cases h3 : 8 * (256 * p' + v') + k' < 8 * (256 * p + v) + k
    · simp [h3, show 8 * (256 * p' + v') + k' < 8 * (256 * p + v) + k + 1 by omega]
    · simp [h3, show ¬ (8 * (256 * p' + v') + k' < 8 * (256 * p + v) + k + 1) by omega]

theorem tabInv_skip0 {p : ℕ} {t : Array UInt64} (h : TabInv (8 * (256 * p)) t) : TabInv (8 * (256 * p + 1)) t := by
  obtain ⟨hs, hb⟩ := h
  refine ⟨hs, fun p' hp' v' hv' k' hk' b hb' => ?_⟩
  rw [hb p' hp' v' hv' k' hk' b hb']
  by_cases hv0 : v' = 0
  · subst hv0
    rw [tabB_zero]
    simp
  · by_cases h3 : 8 * (256 * p' + v') + k' < 8 * (256 * p)
    · simp [h3, show 8 * (256 * p' + v') + k' < 8 * (256 * p + 1) by omega]
    · simp [h3, show ¬ (8 * (256 * p' + v') + k' < 8 * (256 * p + 1)) by omega]

theorem tabInv_final : TabInv (8 * 256 * 192) Hm96.Default512.tables := by
  rw [tables_eq]
  have h0 : TabInv (8 * (256 * 0)) (Array.replicate (8 * 256 * 192) 0) := by
    refine ⟨Array.size_replicate .., fun p hp v hv k hk b _ => ?_⟩
    rw [get!_replicate (by omega)]
    simp
  have := foldl_range'_induct (fun p t => TabInv (8 * (256 * p)) t)
    (fun t p => (List.range' 1 255).foldl (fun t v => (List.range' 0 8).foldl (tabStep p v) t) t) 0 192 _ h0
    fun p t _ hp h => by
      have hv := foldl_range'_induct (fun v t => TabInv (8 * (256 * p + v)) t)
        (fun t v => (List.range' 0 8).foldl (tabStep p v) t) 1 255 t (tabInv_skip0 h) fun v t hv1 hv h => by
          have hk := foldl_range'_induct (fun k t => TabInv (8 * (256 * p + v) + k) t) (tabStep p v) 0 8 t
            (by simpa using h) fun k t _ hk h => tabInv_step (by omega) hv1 (by omega) (by omega) h
          rw [Nat.zero_add] at hk
          rwa [show 8 * (256 * p + (v + 1)) = 8 * (256 * p + v) + 8 by ring]
      rwa [show 8 * (256 * (p + 1)) = 8 * (256 * p + (1 + 255)) by ring]
  simpa using this

/-- **`tables`**: bit `b` of limb `k` of entry `256 p + v` is `⊕_{s < 8} v_s · key_{64k + b + 8p + s}`. -/
theorem tables_bit {p v k b : ℕ} (hp : p < 192) (hv : v < 256) (hk : k < 8) (hb : b < 64) :
    (Hm96.Default512.tables[8 * (256 * p + v) + k]!).toNat.testBit b = tabB p v k b := by
  rw [tabInv_final.2 p hp v hv k hk b hb]
  simp [show 8 * (256 * p + v) + k < 8 * 256 * 192 by omega]

/-! ## `mask` -/

/-- After the salt's first `P` bytes, bit `b` of limb `k` of the accumulator is `⊕_{p < P} tabB p y_p k b`. -/
def AccInv (salt : ByteArray) (P : ℕ) (acc : Array UInt64) : Prop :=
  acc.size = 8 ∧ ∀ k < 8, ∀ b < 64,
    (acc[k]!).toNat.testBit b = xsum (List.range P) fun p => tabB p (salt.get! p).toNat k b

theorem accInv_step (salt : ByteArray) {P : ℕ} (hP : P < 192) {acc : Array UInt64} (h : AccInv salt P acc) :
    AccInv salt (P + 1) ((List.range' 0 8).foldl (accStep salt P) acc) := by
  obtain ⟨hs, hb⟩ := h
  have hJ := foldl_range'_induct (fun k0 a => a.size = 8 ∧ ∀ k < 8, a[k]! = if k < k0 then
      acc[k]! ^^^ Hm96.Default512.tables[8 * (256 * P + (salt.get! P).toNat) + k]! else acc[k]!)
    (accStep salt P) 0 8 acc ⟨hs, fun k _ => by simp⟩ fun k0 a _ hk0 ⟨ha, hk⟩ => by
      refine ⟨by rw [accStep, size_set!, ha], fun k hk' => ?_⟩
      rw [accStep, get!_set! (by omega)]
      split_ifs with h1 h2 h2
      · subst h1
        rw [hk k0 hk', ite_eq_right (by omega)]
      · omega
      · rw [hk k hk', ite_eq_left (by omega)]
      · rw [hk k hk', ite_eq_right (by omega)]
  rw [Nat.zero_add] at hJ
  refine ⟨hJ.1, fun k hk b hb' => ?_⟩
  rw [hJ.2 k hk, ite_eq_left hk, testBit_xor, hb k hk b hb',
    tables_bit hP (UInt8.toNat_lt _) hk hb', List.range_succ, xsum_append, xsum_cons, xsum_nil, Bool.xor_false]

theorem accInv_final (salt : ByteArray) : AccInv salt 192 (accOf salt) := by
  have := foldl_range'_induct (fun P acc => AccInv salt P acc)
    (fun acc p => (List.range' 0 8).foldl (accStep salt p) acc) 0 192 (Array.replicate 8 0)
    ⟨Array.size_replicate .., fun k hk b _ => by rw [get!_replicate hk]; simp [xsum_nil]⟩
    fun P acc _ hP h => accInv_step salt (by omega) h
  rw [Nat.zero_add] at this
  exact this

theorem foldl_push2 (g : ℕ → ℕ → UInt8) (m : ℕ) : ∀ (l : List ℕ) (b : ByteArray),
    (l.foldl (fun out k => (List.range' 0 m).foldl (fun out j => out.push (g k j)) out) b).data.toList =
      b.data.toList ++ (l.map fun k => (List.range' 0 m).map (g k)).flatten
  | [], b => by simp
  | k :: l, b => by
    rw [List.foldl_cons, foldl_push2 g m l, foldl_push_data (g k), List.map_cons, List.flatten_cons, List.append_assoc]

theorem getD_flatten_range' (g : ℕ → ℕ → UInt8) (m : ℕ) : ∀ (n : ℕ),
    ((List.range' 0 n).map fun k => (List.range' 0 m).map (g k)).flatten.length = m * n ∧
      ∀ k < n, ∀ j < m, ((List.range' 0 n).map fun k => (List.range' 0 m).map (g k)).flatten.getD (m * k + j) 0 = g k j
  | 0 => ⟨rfl, fun k hk => absurd hk (Nat.not_lt_zero _)⟩
  | n + 1 => by
    obtain ⟨hl, hg⟩ := getD_flatten_range' g m n
    rw [List.range'_concat, Nat.one_mul, Nat.zero_add, List.map_append, List.flatten_append, List.map_cons,
      List.map_nil, List.flatten_cons, List.flatten_nil, List.append_nil]
    refine ⟨by rw [List.length_append, hl, List.length_map, List.length_range', Nat.mul_succ], fun k hk j hj => ?_⟩
    by_cases hkn : k < n
    · rw [List.getD_append _ _ _ _ (by rw [hl]; nlinarith), hg k hkn j hj]
    · have hk' : k = n := by omega
      subst hk'
      rw [List.getD_append_right _ _ _ _ (by rw [hl]; omega), hl, Nat.add_sub_cancel_left,
        List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range' (by omega), Option.map_some,
        Option.getD_some, Nat.zero_add, Nat.one_mul]

/-- **The executable's mask is `maskV`**: bit `t < 512` of `Hm96.Default512.mask y` (byte-string order) is
`⊕_{j < 1536} key_{t+j} · y_j` (the bytes of `y` past 192 are not read). -/
theorem bitL_mask (s : List UInt8) {t : ℕ} (ht : t < 512) :
    bitL (Hm96.Default512.mask ⟨s.toArray⟩).data.toList t = maskV Hm96.Default512.key (bitL s) t := by
  rw [mask_eq, foldl_push2, show (ByteArray.emptyWithCapacity 64).data.toList = [] from rfl, List.nil_append, bitL,
    show t / 8 = 8 * (t / 64) + t / 8 % 8 by omega,
    (getD_flatten_range' _ 8 8).2 _ (by omega) _ (Nat.mod_lt _ (by norm_num))]
  rw [UInt64.toNat_toUInt8, Nat.testBit_mod_two_pow, UInt64.toNat_and, Nat.testBit_and, UInt64.toNat_shiftRight,
    Nat.testBit_shiftRight]
  have e1 : (8 * (t / 8 % 8)).toUInt64.toNat % 64 = 8 * (t / 8 % 8) := by
    rw [Nat.toUInt64_eq, UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega)]
  have e2 : (255 : UInt64).toNat.testBit (t % 8) = true := by
    rw [show (255 : UInt64).toNat = 2 ^ 8 - 1 from rfl, Nat.testBit_two_pow_sub_one]
    simp [show t % 8 < 8 by omega]
  rw [e1, e2, show 8 * (t / 8 % 8) + t % 8 = t % 64 by omega, (accInv_final _).2 _ (by omega) _ (by omega)]
  simp only [show t % 8 < 8 by omega, decide_true, Bool.and_true, Bool.true_and]
  rw [maskV, show 1536 = 8 * 192 from rfl]
  change _ = xsum (List.range (8 * 192)) fun j => kb (t + j) && bitL s j
  rw [xsum_range_mul]
  refine xsum_congr fun p hp => ?_
  simp only [List.mem_range] at hp
  rw [tabB]
  refine xsum_congr fun s' hs' => ?_
  simp only [List.mem_range] at hs'
  rw [get!_eq_getD, bitL, show (8 * p + s') / 8 = p by omega, show (8 * p + s') % 8 = s' by omega,
    show 64 * (t / 64) + t % 64 + (8 * p + s') = t + (8 * p + s') by omega, Bool.and_comm]

end FlockSoundness.Discharge.Hm
