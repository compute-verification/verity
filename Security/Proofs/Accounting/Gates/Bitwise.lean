import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.BigOperators.Group.List
import Mathlib.Data.Fintype.Pi
import Mathlib.Data.List.GetD
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum.Basic
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring
import Definitions.Accounting.Guarantees.GatesBitwise

/-!
# Bits for free, proved

`EncodeChainRefuted` by construction. Gates `0` and `1` are the constants `0` and `1`; each coin byte of the codec's
draw (its round constants', its mask's and word 1's tweak's, `slotOf`) is a coin input, and each of its bits a table
read (`bitG`, a gather of the byte's low 8 bits over a constant `0`/`1` column) whose derived interval is `[0, 1]`.
Over those bits any Boolean formula (`BF`) compiles to free gates (`compile`): NOT is the affine map `1 − u` and AND
the u8 s8-family step `0 + u·v`, each derived `[0, 1]`. An output byte is its 8 bits' formulas summed by affine maps
(`buildSum`, derived `[0, 2^t − 1]` after `t` bits). Every function of the draw's bits has a formula (`shannon`), so
every target byte of every word does, and the circuit's only units are its table reads' (`ubG`, at most `9` a bit).
-/

namespace Accounting.SecurityProofs.GatesBitwise

open Accounting.Gates

variable {ι : Type}

/-! ## Values and intervals of a circuit extended at its end -/

lemma foldl_snoc {α β : Type} (f : List α → β → α) :
    ∀ (L : List β) (vs : List α), ∃ t, L.foldl (fun vs g => vs ++ [f vs g]) vs = vs ++ t
  | [], vs => ⟨[], by simp⟩
  | g :: L, vs => by
    obtain ⟨t, ht⟩ := foldl_snoc f L (vs ++ [f vs g])
    exact ⟨[f vs g] ++ t, by rw [List.foldl_cons, ht, List.append_assoc]⟩

lemma foldl_snoc_length {α β : Type} (f : List α → β → α) :
    ∀ (L : List β) (vs : List α), (L.foldl (fun vs g => vs ++ [f vs g]) vs).length = vs.length + L.length
  | [], vs => by simp
  | g :: L, vs => by
    rw [List.foldl_cons, foldl_snoc_length f L]
    simp only [List.length_append, List.length_cons, List.length_nil]
    omega

lemma values_length (C : Dag ι) (x : ι → ℤ) : (values C x).length = C.length :=
  (foldl_snoc_length (fun vs (g : Gate ι) => g.value x vs) C []).trans (by simp)

lemma intervals_length (C : Dag ι) : (intervals C).length = C.length :=
  (foldl_snoc_length (fun ivs (g : Gate ι) => g.derive ivs) C []).trans (by simp)

/-- Gate `j`'s derived interval. -/
def ivOf (C : Dag ι) (j : ℕ) : Iv := (intervals C).getD j (0, 0)

lemma val_append (C E : Dag ι) (x : ι → ℤ) (j : ℕ) (h : j < C.length) : val (C ++ E) x j = val C x j := by
  obtain ⟨t, ht⟩ : ∃ t, values (C ++ E) x = values C x ++ t := by
    unfold values; rw [List.foldl_append]; exact foldl_snoc _ _ _
  unfold val; rw [ht, List.getD_append _ _ _ _ (by rw [values_length]; exact h)]

lemma ivOf_append (C E : Dag ι) (j : ℕ) (h : j < C.length) : ivOf (C ++ E) j = ivOf C j := by
  obtain ⟨t, ht⟩ : ∃ t, intervals (C ++ E) = intervals C ++ t := by
    unfold intervals; rw [List.foldl_append]; exact foldl_snoc _ _ _
  unfold ivOf; rw [ht, List.getD_append _ _ _ _ (by rw [intervals_length]; exact h)]

lemma values_snoc (C : Dag ι) (g : Gate ι) (x : ι → ℤ) :
    values (C ++ [g]) x = values C x ++ [g.value x (values C x)] := by
  simp only [values, List.foldl_append, List.foldl_cons, List.foldl_nil]

lemma intervals_snoc (C : Dag ι) (g : Gate ι) : intervals (C ++ [g]) = intervals C ++ [g.derive (intervals C)] := by
  simp only [intervals, List.foldl_append, List.foldl_cons, List.foldl_nil]

lemma val_snoc (C : Dag ι) (g : Gate ι) (x : ι → ℤ) : val (C ++ [g]) x C.length = g.value x (values C x) := by
  unfold val
  rw [values_snoc, List.getD_append_right _ _ _ _ (by rw [values_length]), values_length, Nat.sub_self]
  simp

lemma ivOf_snoc (C : Dag ι) (g : Gate ι) : ivOf (C ++ [g]) C.length = g.derive (intervals C) := by
  unfold ivOf
  rw [intervals_snoc, List.getD_append_right _ _ _ _ (by rw [intervals_length]), intervals_length, Nat.sub_self]
  simp

lemma val_take (C : Dag ι) (x : ι → ℤ) (n j : ℕ) (hj : j < n) (hn : n ≤ C.length) :
    (values (C.take n) x).getD j 0 = val C x j := by
  have h := val_append (C.take n) (C.drop n) x j (by rw [List.length_take]; omega)
  rw [List.take_append_drop] at h
  exact h.symm

lemma iv_take (C : Dag ι) (n j : ℕ) (hj : j < n) (hn : n ≤ C.length) :
    (intervals (C.take n)).getD j (0, 0) = ivOf C j := by
  have h := ivOf_append (C.take n) (C.drop n) j (by rw [List.length_take]; omega)
  rw [List.take_append_drop] at h
  exact h.symm

lemma val_gate (C : Dag ι) (x : ι → ℤ) (i : ℕ) (h : i < C.length) :
    val C x i = C[i].value x (values (C.take i) x) := by
  rw [← val_take C x (i + 1) i (by omega) h, List.take_add_one, List.getElem?_eq_getElem h, Option.toList_some]
  have := val_snoc (C.take i) C[i] x
  rw [List.length_take, min_eq_left h.le] at this
  exact this

lemma iv_gate (C : Dag ι) (i : ℕ) (h : i < C.length) : ivOf C i = C[i].derive (intervals (C.take i)) := by
  rw [← iv_take C (i + 1) i (by omega) h, List.take_add_one, List.getElem?_eq_getElem h, Option.toList_some]
  have := ivOf_snoc (C.take i) C[i]
  rw [List.length_take, min_eq_left h.le] at this
  exact this

/-! ## The proven window, gate by gate -/

/-- What `Proven` asks of one gate, from the derived intervals `ivs` of the gates before it. -/
def GateOk (ivs : List Iv) (g : Gate ι) : Prop :=
  (g.op.IsDerived → (g.derive ivs).Within g.lo g.hi) ∧
    (∀ c₀ cs, g.op = .affine c₀ cs → s32Min ≤ g.lo ∧ g.hi ≤ s32Max) ∧
    ∀ fa fb k sat, g.op = .mma fa fb k sat →
      g.args.length = 2 * k + 1 ∧ s32Min ≤ g.lo ∧ g.hi ≤ s32Max ∧
        (∀ j ∈ g.args.tail.take k, (ivs.getD j (0, 0)).Within fa.lo fa.hi) ∧
        ∀ j ∈ g.args.tail.drop k, (ivs.getD j (0, 0)).Within fb.lo fb.hi

lemma proven_iff (C : Dag ι) : Proven C ↔ ∀ i (h : i < C.length), GateOk (intervals (C.take i)) C[i] := Iff.rfl

lemma proven_snoc (D : Dag ι) (g : Gate ι) (hD : Proven D) (hg : GateOk (intervals D) g) : Proven (D ++ [g]) := by
  rw [proven_iff] at hD ⊢
  intro i hi
  rw [List.length_append, List.length_singleton] at hi
  rcases Nat.lt_or_ge i D.length with h | h
  · rw [List.getElem_append_left h, List.take_append_of_le_length h.le]
    exact hD i h
  · obtain rfl : i = D.length := by omega
    rw [List.getElem_append_right le_rfl, List.take_left]
    simpa using hg

/-! ## Free gates, and extending a circuit by them -/

/-- A free operation: a constant, an affine map or an s8-family step. -/
def FreeOp : Op ι → Prop
  | .const _ | .affine .. | .mma .. => True
  | _ => False

/-- `D` is `C` followed by free gates. -/
def Ext (C D : Dag ι) : Prop := ∃ E, D = C ++ E ∧ ∀ g ∈ E, FreeOp g.op

lemma Ext.refl (C : Dag ι) : Ext C C := ⟨[], by simp, by simp⟩

lemma Ext.trans {C D D' : Dag ι} (h1 : Ext C D) (h2 : Ext D D') : Ext C D' := by
  obtain ⟨E1, rfl, h1⟩ := h1
  obtain ⟨E2, rfl, h2⟩ := h2
  exact ⟨E1 ++ E2, by simp, fun g hg => (List.mem_append.1 hg).elim (h1 g) (h2 g)⟩

lemma Ext.snoc (D : Dag ι) (g : Gate ι) (hg : FreeOp g.op) : Ext D (D ++ [g]) := ⟨[g], rfl, by simpa using hg⟩

lemma Ext.len {C D : Dag ι} (h : Ext C D) : C.length ≤ D.length := by
  obtain ⟨E, rfl, -⟩ := h
  simp

lemma Ext.val {C D : Dag ι} (h : Ext C D) (x : ι → ℤ) (j : ℕ) (hj : j < C.length) : val D x j = val C x j := by
  obtain ⟨E, rfl, -⟩ := h
  exact val_append C E x j hj

lemma Ext.iv {C D : Dag ι} (h : Ext C D) (j : ℕ) (hj : j < C.length) : ivOf D j = ivOf C j := by
  obtain ⟨E, rfl, -⟩ := h
  exact ivOf_append C E j hj

/-! ## Intervals of bits and of partial sums -/

/-- Both ends in `[0, 1]`. -/
def BitIv (a : Iv) : Prop := 0 ≤ a.1 ∧ a.1 ≤ 1 ∧ 0 ≤ a.2 ∧ a.2 ≤ 1

lemma bit00 : BitIv (0, 0) := by norm_num [BitIv]

lemma bit11 : BitIv (1, 1) := by norm_num [BitIv]

lemma BitIv.within {a : Iv} (h : BitIv a) : a.Within 0 1 := ⟨h.1, h.2.2.2⟩

lemma BitIv.within8 {a : Iv} (h : BitIv a) : a.Within 0 255 := ⟨h.1, by have := h.2.2.2; omega⟩

lemma bitIv_join {a b : Iv} (ha : BitIv a) (hb : BitIv b) : BitIv (Iv.join a b) := by
  obtain ⟨a1, a2, a3, a4⟩ := ha
  obtain ⟨b1, b2, b3, b4⟩ := hb
  exact ⟨le_min a1 b1, min_le_of_left_le a2, le_max_of_le_left a3, max_le a4 b4⟩

lemma bitIv_foldl_join : ∀ (l : List Iv) (s : Iv), BitIv s → (∀ b ∈ l, BitIv b) → BitIv (l.foldl Iv.join s)
  | [], _, hs, _ => hs
  | b :: l, s, hs, hl =>
    bitIv_foldl_join l _ (bitIv_join hs (hl b (by simp))) (fun c hc => hl c (by simp [hc]))

lemma bitIv_mul {a b : Iv} (ha : BitIv a) (hb : BitIv b) : BitIv (Iv.mul a b) := by
  obtain ⟨a1, a2, a3, a4⟩ := ha
  obtain ⟨b1, b2, b3, b4⟩ := hb
  have p : ∀ u v : ℤ, 0 ≤ u → u ≤ 1 → 0 ≤ v → v ≤ 1 → 0 ≤ u * v ∧ u * v ≤ 1 :=
    fun u v h1 h2 h3 h4 => ⟨mul_nonneg h1 h3, by nlinarith⟩
  obtain ⟨p1, q1⟩ := p _ _ a1 a2 b1 b2
  obtain ⟨p2, q2⟩ := p _ _ a1 a2 b3 b4
  obtain ⟨p3, q3⟩ := p _ _ a3 a4 b1 b2
  obtain ⟨p4, q4⟩ := p _ _ a3 a4 b3 b4
  exact ⟨le_min (le_min p1 p2) (le_min p3 p4), min_le_of_left_le (min_le_of_left_le q1),
    le_max_of_le_left (le_max_of_le_left p1), max_le (max_le q1 q2) (max_le q3 q4)⟩

/-- A partial sum of `t` bits: in `[0, 2^t − 1]`. -/
def AccIv (t : ℕ) (a : Iv) : Prop := 0 ≤ a.1 ∧ a.2 ≤ 2 ^ t - 1

lemma accStep {t : ℕ} {ia ir : Iv} (ha : AccIv t ia) (hr : BitIv ir) :
    AccIv (t + 1) (Iv.add (Iv.add (0, 0) (Iv.scale 1 ia)) (Iv.scale (2 ^ t) ir)) := by
  obtain ⟨a1, a2⟩ := ha
  obtain ⟨r1, r2, r3, r4⟩ := hr
  have h2t : (0 : ℤ) < 2 ^ t := by positivity
  unfold Iv.add Iv.scale AccIv
  rw [ite_eq_left (by norm_num), ite_eq_left h2t.le]
  dsimp only
  constructor
  · nlinarith
  · rw [pow_succ]; nlinarith

/-! ## The gates -/

/-- A gate of no step, primitive or site. -/
def mk (op : Op ι) (args : List ℕ) (lo hi : ℤ) : Gate ι := ⟨op, args, lo, hi, 0, "", .other, false⟩

/-- The constant `c`. -/
def cG (c : ℤ) : Gate ι := mk (.const c) [] c c

/-- A coin byte. -/
def inG (s : ι) : Gate ι := mk (.input .coin s) [] 0 255

/-- Bit `t`'s column over the gates `0` and `1` (the constants `0` and `1`): entry `u` is `u`'s bit `t`. -/
def col (t : ℕ) : List ℕ := (List.range 256).map fun u => if u / 2 ^ t % 2 = 1 then 1 else 0

/-- Bit `t` of gate `j`'s value: a table read of its low 8 bits over `col t`. -/
def bitG (j t : ℕ) : Gate ι := mk (.gather (some 8) 0) (j :: col t) 0 1

/-- NOT: `1 − u`. -/
def notG (a : ℕ) : Gate ι := mk (.affine 1 [-1]) [a] 0 1

/-- AND: the u8 step `0 + u·v`, its accumulator gate `0`. -/
def andG (a b : ℕ) : Gate ι := mk (.mma .u8 .u8 1 true) [0, a, b] 0 1

/-- A partial sum's next bit: `a + 2^t · r`. -/
def accG (t a r : ℕ) : Gate ι := mk (.affine 0 [1, 2 ^ t]) [a, r] 0 255

lemma notG_derive (ivs : List Iv) (a : ℕ) :
    (notG (ι := ι) a).derive ivs = Iv.add (1, 1) (Iv.scale (-1) (ivs.getD a (0, 0))) := rfl

lemma andG_derive (ivs : List Iv) (a b : ℕ) :
    (andG (ι := ι) a b).derive ivs = Iv.add (ivs.getD 0 (0, 0)) (Iv.mul (ivs.getD a (0, 0)) (ivs.getD b (0, 0))) :=
  rfl

lemma accG_derive (ivs : List Iv) (t a r : ℕ) :
    (accG (ι := ι) t a r).derive ivs =
      Iv.add (Iv.add (0, 0) (Iv.scale 1 (ivs.getD a (0, 0)))) (Iv.scale (2 ^ t) (ivs.getD r (0, 0))) := rfl

lemma bitG_derive (ivs : List Iv) (j t : ℕ) :
    (bitG (ι := ι) j t).derive ivs = ((col t).map fun j => ivs.getD j (0, 0)).foldl Iv.join (0, 0) := rfl

lemma notG_bit {u : Iv} (h : BitIv u) : BitIv (Iv.add (1, 1) (Iv.scale (-1) u)) := by
  obtain ⟨u1, u2, u3, u4⟩ := h
  unfold Iv.add Iv.scale BitIv
  rw [ite_eq_right (by norm_num)]
  dsimp only
  omega

lemma andG_bit {u v : Iv} (hu : BitIv u) (hv : BitIv v) : BitIv (Iv.add (0, 0) (Iv.mul u v)) := by
  obtain ⟨m1, m2, m3, m4⟩ := bitIv_mul hu hv
  unfold Iv.add BitIv
  dsimp only
  omega

lemma bitG_bit (ivs : List Iv) (j t : ℕ) (h0 : ivs.getD 0 (0, 0) = (0, 0)) (h1 : ivs.getD 1 (0, 0) = (1, 1)) :
    BitIv ((bitG (ι := ι) j t).derive ivs) := by
  rw [bitG_derive]
  refine bitIv_foldl_join _ _ bit00 fun b hb => ?_
  obtain ⟨j', hj', rfl⟩ := List.mem_map.1 hb
  obtain ⟨u, -, rfl⟩ := List.mem_map.1 hj'
  split_ifs
  · rw [h1]; exact bit11
  · rw [h0]; exact bit00

lemma cG_ok (ivs : List Iv) (c : ℤ) : GateOk ivs (cG (ι := ι) c) :=
  ⟨fun _ => ⟨le_rfl, le_rfl⟩, fun _ _ h => (by cases h), fun _ _ _ _ h => (by cases h)⟩

lemma inG_ok (ivs : List Iv) (s : ι) : GateOk ivs (inG s) :=
  ⟨fun h => (h : False).elim, fun _ _ h => (by cases h), fun _ _ _ _ h => (by cases h)⟩

lemma bitG_ok (ivs : List Iv) (j t : ℕ) (h0 : ivs.getD 0 (0, 0) = (0, 0)) (h1 : ivs.getD 1 (0, 0) = (1, 1)) :
    GateOk ivs (bitG (ι := ι) j t) :=
  ⟨fun _ => (bitG_bit ivs j t h0 h1).within, fun _ _ h => (by cases h), fun _ _ _ _ h => (by cases h)⟩

lemma notG_ok (ivs : List Iv) (a : ℕ) (ha : BitIv (ivs.getD a (0, 0))) : GateOk ivs (notG (ι := ι) a) := by
  refine ⟨fun _ => ?_, fun _ _ _ => ⟨by norm_num [notG, mk, s32Min], by norm_num [notG, mk, s32Max]⟩,
    fun _ _ _ _ h => by cases h⟩
  rw [notG_derive]
  exact (notG_bit ha).within

lemma andG_ok (ivs : List Iv) (a b : ℕ) (h0 : ivs.getD 0 (0, 0) = (0, 0)) (ha : BitIv (ivs.getD a (0, 0)))
    (hb : BitIv (ivs.getD b (0, 0))) : GateOk ivs (andG (ι := ι) a b) := by
  refine ⟨fun _ => ?_, fun _ _ h => (by cases h), fun fa fb k sat h => ?_⟩
  · rw [andG_derive, h0]
    exact (andG_bit ha hb).within
  · simp only [andG, mk, Op.mma.injEq] at h
    obtain ⟨rfl, rfl, rfl, rfl⟩ := h
    refine ⟨rfl, by norm_num [andG, mk, s32Min], by norm_num [andG, mk, s32Max], ?_, ?_⟩
    · intro j hj
      simp only [andG, mk, List.tail_cons, List.take_succ_cons, List.take_zero, List.mem_singleton] at hj
      subst hj
      exact ha.within8
    · intro j hj
      simp only [andG, mk, List.tail_cons, List.drop_succ_cons, List.drop_zero, List.mem_singleton] at hj
      subst hj
      exact hb.within8

lemma accG_ok (ivs : List Iv) (t a r : ℕ) (ht : t + 1 ≤ 8) (ha : AccIv t (ivs.getD a (0, 0)))
    (hr : BitIv (ivs.getD r (0, 0))) : GateOk ivs (accG (ι := ι) t a r) := by
  refine ⟨fun _ => ?_, fun _ _ _ => ⟨by norm_num [accG, mk, s32Min], by norm_num [accG, mk, s32Max]⟩,
    fun _ _ _ _ h => by cases h⟩
  rw [accG_derive]
  obtain ⟨h1, h2⟩ := accStep ha hr
  have : (2 : ℤ) ^ (t + 1) ≤ 2 ^ 8 := pow_le_pow_right₀ (by norm_num) ht
  exact ⟨h1, by norm_num at this; show _ ≤ (255 : ℤ); linarith⟩

lemma clampS32_of {z : ℤ} (h1 : s32Min ≤ z) (h2 : z ≤ s32Max) : clampS32 z = z := by
  unfold clampS32; rw [min_eq_right h2, max_eq_right h1]

lemma notG_value (a : ℕ) (x : ι → ℤ) (vs : List ℤ) : (notG a).value x vs = 1 - vs.getD a 0 := by
  simp [Gate.value, Gate.raw, Gate.operands, Op.raw, Op.out, notG, mk]
  ring

lemma accG_value (t a r : ℕ) (x : ι → ℤ) (vs : List ℤ) :
    (accG t a r).value x vs = vs.getD a 0 + 2 ^ t * vs.getD r 0 := by
  simp [Gate.value, Gate.raw, Gate.operands, Op.raw, Op.out, accG, mk]

lemma andG_value (a b : ℕ) (x : ι → ℤ) (vs : List ℤ) (h0 : vs.getD 0 0 = 0)
    (ha : vs.getD a 0 = 0 ∨ vs.getD a 0 = 1) (hb : vs.getD b 0 = 0 ∨ vs.getD b 0 = 1) :
    (andG a b).value x vs = vs.getD a 0 * vs.getD b 0 := by
  have : (andG (ι := ι) a b).value x vs = clampS32 (vs.getD 0 0 + vs.getD a 0 * vs.getD b 0) := by
    simp [Gate.value, Gate.raw, Gate.operands, Op.raw, Op.out, andG, mk, dot]
  rw [this, h0, zero_add]
  apply clampS32_of
  · rcases ha with ha | ha <;> rcases hb with hb | hb <;> rw [ha, hb] <;> norm_num [s32Min]
  · rcases ha with ha | ha <;> rcases hb with hb | hb <;> rw [ha, hb] <;> norm_num [s32Max]

/-! ## Boolean formulas, compiled to free gates -/

/-- A Boolean formula over variables `ℕ`. -/
inductive BF
  | var (v : ℕ)
  | tt
  | ff
  | not (f : BF)
  | and (f g : BF)

/-- A formula's value under an assignment. -/
def BF.eval (β : ℕ → Bool) : BF → Bool
  | .var v => β v
  | .tt => true
  | .ff => false
  | .not f => !f.eval β
  | .and f g => f.eval β && g.eval β

/-- A bit as an integer. -/
def bitv (b : Bool) : ℤ := if b then 1 else 0

lemma bitv01 (b : Bool) : bitv b = 0 ∨ bitv b = 1 := by cases b <;> simp [bitv]

lemma bitv_eq (b : Bool) : bitv b = (b.toNat : ℤ) := by cases b <;> rfl

/-- `f` compiled onto `C` over variable wires `w`: the extended circuit and the gate holding `f`'s value. -/
def compile (w : ℕ → ℕ) : BF → Dag ι → Dag ι × ℕ
  | .var v, C => (C, w v)
  | .tt, C => (C, 1)
  | .ff, C => (C, 0)
  | .not f, C =>
    let p := compile w f C
    (p.1 ++ [notG p.2], p.1.length)
  | .and f g, C =>
    let p := compile w f C
    let q := compile w g p.1
    (q.1 ++ [andG p.2 q.2], q.1.length)

/-- The assignment the base circuit's wires hold on input `x`. -/
def asg (C0 : Dag ι) (w : ℕ → ℕ) (x : ι → ℤ) (v : ℕ) : Bool := decide (val C0 x (w v) = 1)

/-- What compiling asks of the base circuit: gates `0` and `1` the constants `0` and `1`, and every wire a bit. -/
structure BaseOk (C0 : Dag ι) (w : ℕ → ℕ) : Prop where
  len : 2 ≤ C0.length
  iv0 : ivOf C0 0 = (0, 0)
  iv1 : ivOf C0 1 = (1, 1)
  val0 : ∀ x, val C0 x 0 = 0
  val1 : ∀ x, val C0 x 1 = 1
  wlt : ∀ v, w v < C0.length
  wiv : ∀ v, BitIv (ivOf C0 (w v))
  wval : ∀ x v, val C0 x (w v) = 0 ∨ val C0 x (w v) = 1

/-- `D` extends `C` by free gates, is proven, and its gate `r` holds `f`'s value, a bit. -/
def Spec (C0 : Dag ι) (w : ℕ → ℕ) (C : Dag ι) (f : BF) (D : Dag ι) (r : ℕ) : Prop :=
  Ext C D ∧ Proven D ∧ r < D.length ∧ BitIv (ivOf D r) ∧ ∀ x, val D x r = bitv (f.eval (asg C0 w x))

lemma compile_spec (C0 : Dag ι) (w : ℕ → ℕ) (hB : BaseOk C0 w) :
    ∀ (f : BF) (C : Dag ι), Ext C0 C → Proven C → Spec C0 w C f (compile w f C).1 (compile w f C).2
  | .var v, C, hC, hP => by
    refine ⟨Ext.refl C, hP, lt_of_lt_of_le (hB.wlt v) hC.len, ?_, fun x => ?_⟩
    · show BitIv (ivOf C (w v))
      rw [hC.iv (w v) (hB.wlt v)]
      exact hB.wiv v
    · show val C x (w v) = bitv (asg C0 w x v)
      rw [hC.val x (w v) (hB.wlt v)]
      unfold asg bitv
      rcases hB.wval x v with h | h <;> rw [h] <;> simp
  | .tt, C, hC, hP => by
    have h2 := hB.len
    refine ⟨Ext.refl C, hP, (show 1 < C.length by have := hC.len; omega), ?_, fun x => ?_⟩
    · show BitIv (ivOf C 1)
      rw [hC.iv 1 (by omega), hB.iv1]
      exact bit11
    · show val C x 1 = bitv true
      rw [hC.val x 1 (by omega), hB.val1]
      rfl
  | .ff, C, hC, hP => by
    have h2 := hB.len
    refine ⟨Ext.refl C, hP, (show 0 < C.length by have := hC.len; omega), ?_, fun x => ?_⟩
    · show BitIv (ivOf C 0)
      rw [hC.iv 0 (by omega), hB.iv0]
      exact bit00
    · show val C x 0 = bitv false
      rw [hC.val x 0 (by omega), hB.val0]
      rfl
  | .not f, C, hC, hP => by
    have ih := compile_spec C0 w hB f C hC hP
    show Spec C0 w C f.not ((compile w f C).1 ++ [notG (compile w f C).2]) (compile w f C).1.length
    rcases hq : compile w f C with ⟨D, r⟩
    rw [hq] at ih
    obtain ⟨hE, hPD, hr, hbit, hv⟩ := ih
    dsimp only
    refine ⟨hE.trans (Ext.snoc D _ trivial), proven_snoc D _ hPD (notG_ok _ r hbit), by simp, ?_, fun x => ?_⟩
    · rw [ivOf_snoc, notG_derive]
      exact notG_bit hbit
    · rw [val_snoc, notG_value]
      show 1 - val D x r = _
      rw [hv x]
      simp only [BF.eval]
      cases f.eval (asg C0 w x) <;> simp [bitv]
  | .and f g, C, hC, hP => by
    have h2 := hB.len
    have ih1 := compile_spec C0 w hB f C hC hP
    show Spec C0 w C (f.and g) ((compile w g (compile w f C).1).1 ++ [andG (compile w f C).2
      (compile w g (compile w f C).1).2]) (compile w g (compile w f C).1).1.length
    rcases hq : compile w f C with ⟨D, r⟩
    rw [hq] at ih1
    obtain ⟨hE, hPD, hr, hbit, hv⟩ := ih1
    have ih2 := compile_spec C0 w hB g D (hC.trans hE) hPD
    dsimp only
    rcases hq2 : compile w g D with ⟨D2, r2⟩
    rw [hq2] at ih2
    obtain ⟨hE2, hPD2, hr2, hbit2, hv2⟩ := ih2
    dsimp only
    have hC2 : Ext C0 D2 := hC.trans (hE.trans hE2)
    have h0 : ivOf D2 0 = (0, 0) := by rw [hC2.iv 0 (by omega), hB.iv0]
    have hbit' : BitIv (ivOf D2 r) := by rw [hE2.iv r hr]; exact hbit
    refine ⟨hE.trans (hE2.trans (Ext.snoc D2 _ trivial)), proven_snoc D2 _ hPD2 (andG_ok _ r r2 h0 hbit' hbit2),
      by simp, ?_, fun x => ?_⟩
    · rw [ivOf_snoc, andG_derive]
      show BitIv (Iv.add (ivOf D2 0) (Iv.mul (ivOf D2 r) (ivOf D2 r2)))
      rw [h0]
      exact andG_bit hbit' hbit2
    · have hv0 : val D2 x 0 = 0 := by rw [hC2.val x 0 (by omega), hB.val0]
      have hvr : val D2 x r = bitv (f.eval (asg C0 w x)) := by rw [hE2.val x r hr, hv x]
      have hvr2 := hv2 x
      have ha01 : (values D2 x).getD r 0 = 0 ∨ (values D2 x).getD r 0 = 1 := by
        show val D2 x r = 0 ∨ val D2 x r = 1
        rw [hvr]
        exact bitv01 _
      have hb01 : (values D2 x).getD r2 0 = 0 ∨ (values D2 x).getD r2 0 = 1 := by
        show val D2 x r2 = 0 ∨ val D2 x r2 = 1
        rw [hvr2]
        exact bitv01 _
      rw [val_snoc, andG_value r r2 x (values D2 x) hv0 ha01 hb01]
      show val D2 x r * val D2 x r2 = _
      rw [hvr, hvr2]
      simp only [BF.eval]
      cases f.eval (asg C0 w x) <;> cases g.eval (asg C0 w x) <;> simp [bitv]

/-! ## Bytes from bits -/

/-- `Σ b_j · 2^(t + j)`. -/
def bitsVal : List Bool → ℕ → ℕ
  | [], _ => 0
  | b :: bs, t => b.toNat * 2 ^ t + bitsVal bs (t + 1)

/-- Formulas `fs` compiled onto `C` and summed, bit `j` weighted `2^(t + j)`, onto the partial sum at gate `a`. -/
def buildSum (w : ℕ → ℕ) : List BF → ℕ → ℕ → Dag ι → Dag ι × ℕ
  | [], _, a, C => (C, a)
  | f :: fs, t, a, C =>
    let p := compile w f C
    buildSum w fs (t + 1) p.1.length (p.1 ++ [accG t a p.2])

lemma buildSum_cons (w : ℕ → ℕ) (f : BF) (fs : List BF) (t a : ℕ) (C : Dag ι) :
    buildSum w (f :: fs) t a C =
      buildSum w fs (t + 1) (compile w f C).1.length ((compile w f C).1 ++ [accG t a (compile w f C).2]) := rfl

lemma buildSum_spec (C0 : Dag ι) (w : ℕ → ℕ) (hB : BaseOk C0 w) :
    ∀ (fs : List BF) (t a : ℕ) (C : Dag ι) (A : (ι → ℤ) → ℕ), t + fs.length ≤ 8 → Ext C0 C → Proven C →
      a < C.length → AccIv t (ivOf C a) → (∀ x, val C x a = (A x : ℤ)) →
      Ext C (buildSum w fs t a C).1 ∧ Proven (buildSum w fs t a C).1 ∧
        (buildSum w fs t a C).2 < (buildSum w fs t a C).1.length ∧
        ∀ x, val (buildSum w fs t a C).1 x (buildSum w fs t a C).2 =
          ((A x + bitsVal (fs.map fun f => f.eval (asg C0 w x)) t : ℕ) : ℤ)
  | [], t, a, C, A, _, _, hP, ha, _, hv => ⟨Ext.refl C, hP, ha, fun x => by simp [buildSum, bitsVal, hv x]⟩
  | f :: fs, t, a, C, A, ht, hC, hP, ha, hacc, hv => by
    have ih1 := compile_spec C0 w hB f C hC hP
    rw [buildSum_cons]
    rcases hq : compile w f C with ⟨D, r⟩
    rw [hq] at ih1
    obtain ⟨hE, hPD, hr, hbit, hvr⟩ := ih1
    dsimp only
    have hla : ivOf D a = ivOf C a := hE.iv a ha
    have hlen : (List.length (f :: fs)) = fs.length + 1 := rfl
    have hgo : GateOk (intervals D) (accG (ι := ι) t a r) :=
      accG_ok _ t a r (by omega) (by show AccIv t (ivOf D a); rw [hla]; exact hacc) hbit
    have ih2 := buildSum_spec C0 w hB fs (t + 1) D.length (D ++ [accG t a r])
      (fun x => A x + (f.eval (asg C0 w x)).toNat * 2 ^ t) (by omega)
      ((hC.trans hE).trans (Ext.snoc D (accG t a r) trivial)) (proven_snoc D _ hPD hgo) (by simp)
      (by rw [ivOf_snoc, accG_derive]; exact accStep (by show AccIv t (ivOf D a); rw [hla]; exact hacc) hbit)
      (fun x => by
        rw [val_snoc, accG_value]
        show val D x a + 2 ^ t * val D x r = _
        rw [hE.val x a ha, hv x, hvr x, bitv_eq]
        push_cast
        ring)
    obtain ⟨a1, a2, a3, a4⟩ := ih2
    refine ⟨hE.trans ((Ext.snoc D (accG t a r) trivial).trans a1), a2, a3, fun x => ?_⟩
    rw [a4 x]
    simp only [List.map_cons, bitsVal]
    push_cast
    ring

/-- Each list of formulas summed into one gate, on circuits extending one another. -/
def buildAll (w : ℕ → ℕ) : List (List BF) → Dag ι → Dag ι × List ℕ
  | [], C => (C, [])
  | fs :: Φ, C =>
    let p := buildSum w fs 0 0 C
    let q := buildAll w Φ p.1
    (q.1, p.2 :: q.2)

lemma buildAll_cons (w : ℕ → ℕ) (fs : List BF) (Φ : List (List BF)) (C : Dag ι) :
    buildAll w (fs :: Φ) C =
      ((buildAll w Φ (buildSum w fs 0 0 C).1).1, (buildSum w fs 0 0 C).2 :: (buildAll w Φ (buildSum w fs 0 0 C).1).2) :=
  rfl

lemma buildAll_spec (C0 : Dag ι) (w : ℕ → ℕ) (hB : BaseOk C0 w) :
    ∀ (Φ : List (List BF)) (C : Dag ι), (∀ fs ∈ Φ, fs.length ≤ 8) → Ext C0 C → Proven C →
      Ext C (buildAll w Φ C).1 ∧ Proven (buildAll w Φ C).1 ∧
        ∀ o < Φ.length, ∀ x, val (buildAll w Φ C).1 x ((buildAll w Φ C).2.getD o 0) =
          (bitsVal ((Φ.getD o []).map fun f => f.eval (asg C0 w x)) 0 : ℤ)
  | [], C, _, _, hP => ⟨Ext.refl C, hP, fun o ho => by simp at ho⟩
  | fs :: Φ, C, hl, hC, hP => by
    have h2 := hB.len
    have hs := buildSum_spec C0 w hB fs 0 0 C (fun _ => 0) (by simpa using hl fs (by simp)) hC hP
      (by have := hC.len; omega)
      (by rw [hC.iv 0 (by omega), hB.iv0]; exact ⟨le_rfl, by norm_num⟩)
      (fun x => by rw [hC.val x 0 (by omega), hB.val0]; simp)
    rw [buildAll_cons]
    rcases hq : buildSum w fs 0 0 C with ⟨D, r⟩
    rw [hq] at hs
    obtain ⟨hE, hPD, hr, hv⟩ := hs
    obtain ⟨hE2, hP2, hv2⟩ := buildAll_spec C0 w hB Φ D (fun fs' h => hl fs' (by simp [h])) (hC.trans hE) hPD
    dsimp only
    refine ⟨hE.trans hE2, hP2, fun o ho x => ?_⟩
    cases o with
    | zero =>
      simp only [List.getD_cons_zero]
      rw [hE2.val x r hr, hv x]
      simp
    | succ o =>
      simp only [List.getD_cons_succ]
      exact hv2 o (by simp at ho; omega) x

/-! ## Every function of finitely many bits has a formula -/

/-- `β` on the variables below `M`, `false` above. -/
def trunc (M : ℕ) (β : ℕ → Bool) (v : ℕ) : Bool := if v < M then β v else false

lemma trunc_succ (M : ℕ) (β : ℕ → Bool) : (fun v => if v = M then β M else trunc M β v) = trunc (M + 1) β := by
  funext v
  unfold trunc
  by_cases h1 : v = M
  · subst h1; simp
  · by_cases h2 : v < M
    · rw [ite_eq_right h1, ite_eq_left h2, ite_eq_left (by omega)]
    · rw [ite_eq_right h1, ite_eq_right h2, ite_eq_right (by omega)]

theorem shannon : ∀ (M : ℕ) (ψ : (ℕ → Bool) → Bool), ∃ f : BF, ∀ β, f.eval β = ψ (trunc M β)
  | 0, ψ => by
    refine ⟨if ψ (fun _ => false) then .tt else .ff, fun β => ?_⟩
    have : trunc 0 β = fun _ => false := funext fun v => by simp [trunc]
    rw [this]
    split <;> simp_all [BF.eval]
  | M + 1, ψ => by
    obtain ⟨f1, h1⟩ := shannon M (fun γ => ψ (fun v => if v = M then true else γ v))
    obtain ⟨f0, h0⟩ := shannon M (fun γ => ψ (fun v => if v = M then false else γ v))
    refine ⟨.not (.and (.not (.and (.var M) f1)) (.not (.and (.not (.var M)) f0))), fun β => ?_⟩
    rw [← trunc_succ M β]
    cases hb : β M
    · have h0' : f0.eval β = ψ (fun v => if v = M then false else trunc M β v) := h0 β
      rw [← h0']
      simp [BF.eval, hb]
    · have h1' : f1.eval β = ψ (fun v => if v = M then true else trunc M β v) := h1 β
      rw [← h1']
      simp [BF.eval, hb]

/-! ## The base circuit: the constants, the coin bytes and their bits -/

/-- Gate `i` of the base circuit over `N` coin bytes in slots `sl`: `0` and `1`, the bytes, then their bits. -/
def baseGate (N : ℕ) (sl : ℕ → ι) (i : ℕ) : Gate ι :=
  if i = 0 then cG 0 else if i = 1 then cG 1 else if i < 2 + N then inG (sl (i - 2))
  else bitG (2 + (i - 2 - N) / 8) ((i - 2 - N) % 8)

/-- The base circuit. -/
def C0 (N : ℕ) (sl : ℕ → ι) : Dag ι := (List.range (2 + N + 8 * N)).map (baseGate N sl)

/-- Variable `v`'s wire: bit `v % 8` of byte `v / 8`, and the constant `0` past the bytes. -/
def wire (N v : ℕ) : ℕ := if v < 8 * N then 2 + N + v else 0

section Base

variable (N : ℕ) (sl : ℕ → ι)

lemma C0_length : (C0 N sl).length = 2 + N + 8 * N := by simp [C0]

lemma C0_get (i : ℕ) (h : i < (C0 N sl).length) : (C0 N sl)[i] = baseGate N sl i := by simp [C0]

lemma C0_get_bit (v : ℕ) (h : 2 + N + v < (C0 N sl).length) :
    (C0 N sl)[2 + N + v] = bitG (2 + v / 8) (v % 8) := by
  rw [C0_get]
  unfold baseGate
  rw [ite_eq_right (show 2 + N + v ≠ 0 by omega), ite_eq_right (show 2 + N + v ≠ 1 by omega),
    ite_eq_right (show ¬ 2 + N + v < 2 + N by omega), show 2 + N + v - 2 - N = v by omega]

lemma C0_val0 (x : ι → ℤ) : val (C0 N sl) x 0 = 0 := by
  rw [val_gate _ x 0 (by rw [C0_length]; omega), C0_get]
  rfl

lemma C0_val1 (x : ι → ℤ) : val (C0 N sl) x 1 = 1 := by
  rw [val_gate _ x 1 (by rw [C0_length]; omega), C0_get]
  rfl

lemma C0_iv0 : ivOf (C0 N sl) 0 = (0, 0) := by
  rw [iv_gate _ 0 (by rw [C0_length]; omega), C0_get]
  rfl

lemma C0_iv1 : ivOf (C0 N sl) 1 = (1, 1) := by
  rw [iv_gate _ 1 (by rw [C0_length]; omega), C0_get]
  rfl

lemma C0_val_in (x : ι → ℤ) (j : ℕ) (hj : j < N) : val (C0 N sl) x (2 + j) = x (sl j) := by
  rw [val_gate _ x _ (by rw [C0_length]; omega), C0_get]
  unfold baseGate
  rw [ite_eq_right (show 2 + j ≠ 0 by omega), ite_eq_right (show 2 + j ≠ 1 by omega), ite_eq_left (show 2 + j < 2 + N by omega),
    show 2 + j - 2 = j by omega]
  rfl

lemma col_getD (f : ℕ → ℤ) (t z : ℕ) (h0 : f 0 = 0) (h1 : f 1 = 1) :
    ((col t).map f).getD z 0 = if z < 256 ∧ z / 2 ^ t % 2 = 1 then 1 else 0 := by
  unfold col
  rw [List.map_map, List.getD_eq_getElem?_getD, List.getElem?_map]
  by_cases hz : z < 256
  · rw [List.getElem?_range hz]
    simp only [Option.map_some, Option.getD_some, Function.comp]
    split_ifs <;> simp_all
  · rw [List.getElem?_eq_none (by simp; omega)]
    simp [hz]

lemma C0_val_bit (x : ι → ℤ) (v : ℕ) (hv : v < 8 * N) :
    val (C0 N sl) x (2 + N + v) =
      if (x (sl (v / 8)) % 2 ^ 8).toNat < 256 ∧ (x (sl (v / 8)) % 2 ^ 8).toNat / 2 ^ (v % 8) % 2 = 1 then 1
      else 0 := by
  have hl : 2 + N + v < (C0 N sl).length := by rw [C0_length]; omega
  rw [val_gate _ x _ hl, C0_get_bit N sl v]
  have hvs0 : (values ((C0 N sl).take (2 + N + v)) x).getD 0 0 = 0 := by
    rw [val_take _ x _ 0 (by omega) hl.le, C0_val0]
  have hvs1 : (values ((C0 N sl).take (2 + N + v)) x).getD 1 0 = 1 := by
    rw [val_take _ x _ 1 (by omega) hl.le, C0_val1]
  have hvsj : (values ((C0 N sl).take (2 + N + v)) x).getD (2 + v / 8) 0 = x (sl (v / 8)) := by
    rw [val_take _ x _ _ (by omega) hl.le, C0_val_in N sl x (v / 8) (by omega)]
  generalize values ((C0 N sl).take (2 + N + v)) x = vs at hvs0 hvs1 hvsj
  show ((col (v % 8)).map fun j => vs.getD j 0).getD
    (Op.gatherAt (some 8) ((col (v % 8)).map fun j => vs.getD j 0).length (vs.getD (2 + v / 8) 0)) 0 = _
  rw [col_getD _ _ _ hvs0 hvs1, hvsj]
  rfl

lemma C0_iv_bit (v : ℕ) (hv : v < 8 * N) : BitIv (ivOf (C0 N sl) (2 + N + v)) := by
  have hl : 2 + N + v < (C0 N sl).length := by rw [C0_length]; omega
  rw [iv_gate _ _ hl, C0_get_bit N sl v]
  exact bitG_bit _ _ _ (by rw [iv_take _ _ 0 (by omega) hl.le, C0_iv0])
    (by rw [iv_take _ _ 1 (by omega) hl.le, C0_iv1])

lemma C0_proven : Proven (C0 N sl) := by
  rw [proven_iff]
  intro i hi
  rw [C0_get]
  have hl := hi
  rw [C0_length] at hl
  unfold baseGate
  split_ifs with h0 h1 h2
  · exact cG_ok _ 0
  · exact cG_ok _ 1
  · exact inG_ok _ _
  · exact bitG_ok _ _ _ (by rw [iv_take _ _ 0 (by omega) hi.le, C0_iv0]) (by rw [iv_take _ _ 1 (by omega) hi.le, C0_iv1])

lemma C0_base : BaseOk (C0 N sl) (wire N) where
  len := by rw [C0_length]; omega
  iv0 := C0_iv0 N sl
  iv1 := C0_iv1 N sl
  val0 := C0_val0 N sl
  val1 := C0_val1 N sl
  wlt v := by unfold wire; split_ifs <;> rw [C0_length] <;> omega
  wiv v := by
    unfold wire
    split_ifs with h
    · exact C0_iv_bit N sl v h
    · rw [C0_iv0]; exact bit00
  wval x v := by
    unfold wire
    split_ifs with h
    · rw [C0_val_bit N sl x v h]; split_ifs <;> simp
    · left; exact C0_val0 N sl x

lemma trunc_asg (x : ι → ℤ) : trunc (8 * N) (asg (C0 N sl) (wire N) x) = asg (C0 N sl) (wire N) x := by
  funext v
  unfold trunc
  split_ifs with h
  · rfl
  · unfold asg wire
    rw [ite_eq_right h, C0_val0]
    rfl

end Base

/-! ## Units -/

/-- A gate's most units: one a budgeted gate, its index bits and one a gather, none any other. -/
def ubG (g : Gate ι) : ℕ :=
  match g.op with
  | .budgeted _ _ => 1
  | .gather _ _ => g.indexBits + 1
  | _ => 0

lemma units_le (C : Dag ι) (i : ℕ) (h : i < C.length) : units C i ≤ ubG C[i] := by
  unfold units ubG
  rw [List.getElem?_eq_getElem h]
  dsimp only
  cases hop : C[i].op with
  | budgeted cls f => dsimp only; split_ifs <;> omega
  | gather low fill =>
    cases hargs : C[i].args with
    | nil => exact Nat.zero_le _
    | cons j col => dsimp only; split_ifs <;> omega
  | _ => exact Nat.zero_le _

lemma sum_getD_map (f : Gate ι → ℕ) :
    ∀ C : Dag ι, ∑ i ∈ Finset.range C.length, ((C[i]?).map f).getD 0 = (C.map f).sum
  | [] => by simp
  | g :: C => by
    rw [List.length_cons, Finset.sum_range_succ', List.map_cons, List.sum_cons, ← sum_getD_map f C]
    simp only [List.getElem?_cons_succ, List.getElem?_cons_zero, Option.map_some, Option.getD_some]
    ring

lemma budgeted_le (C : Dag ι) : budgeted C ≤ (C.map ubG).sum := by
  unfold budgeted
  rw [← sum_getD_map ubG C]
  refine Finset.sum_le_sum fun i hi => ?_
  rw [Finset.mem_range] at hi
  rw [List.getElem?_eq_getElem hi]
  exact units_le C i hi

lemma sum_ubG_free (E : Dag ι) (hE : ∀ g ∈ E, FreeOp g.op) : (E.map ubG).sum = 0 := by
  refine List.sum_eq_zero fun y hy => ?_
  obtain ⟨g, hg, rfl⟩ := List.mem_map.1 hy
  have := hE g hg
  unfold ubG
  unfold FreeOp at this
  split <;> simp_all

lemma C0_ub (N : ℕ) (sl : ℕ → ι) : ((C0 N sl).map ubG).sum ≤ 72 * N := by
  unfold C0
  rw [List.map_map, List.range_add, List.map_append, List.sum_append, List.map_map]
  have h1 : ((List.range (2 + N)).map (ubG ∘ baseGate N sl)).sum = 0 := by
    refine List.sum_eq_zero fun y hy => ?_
    obtain ⟨i, hi, rfl⟩ := List.mem_map.1 hy
    rw [List.mem_range] at hi
    simp only [Function.comp]
    unfold baseGate
    split_ifs <;> rfl
  have h2 : ((List.range (8 * N)).map ((ubG ∘ baseGate N sl) ∘ fun x => 2 + N + x)).sum ≤ (8 * N) * 9 := by
    have := List.sum_le_length_nsmul ((List.range (8 * N)).map ((ubG ∘ baseGate N sl) ∘ fun x => 2 + N + x)) 9
      (by
        intro y hy
        obtain ⟨v, -, rfl⟩ := List.mem_map.1 hy
        simp only [Function.comp]
        unfold baseGate
        split_ifs <;> simp [ubG, cG, inG, bitG, mk, Gate.indexBits])
    simpa using this
  omega

/-! ## Which gates the circuit has -/

/-- A gate of the construction: a coin byte declared `[0, 255]`, or no input and no budgeted gate. -/
def GoodG (g : Gate ι) : Prop :=
  match g.op with
  | .input b _ => b = .coin ∧ g.lo = 0 ∧ g.hi = 255
  | .budgeted _ _ => False
  | _ => True

lemma good_of_free (g : Gate ι) (h : FreeOp g.op) : GoodG g := by
  unfold GoodG
  unfold FreeOp at h
  split <;> simp_all

lemma C0_good (N : ℕ) (sl : ℕ → ι) (g : Gate ι) (hg : g ∈ C0 N sl) : GoodG g := by
  unfold C0 at hg
  obtain ⟨i, -, rfl⟩ := List.mem_map.1 hg
  unfold baseGate
  split_ifs <;> simp [GoodG, cG, inG, bitG, mk]

lemma declared_of_good (C : Dag ι) (x : ι → ℤ) (hC : ∀ g ∈ C, GoodG g) (hx : ∀ s, 0 ≤ x s ∧ x s ≤ 255) :
    DeclaredHold C x := by
  intro i hi hnd
  have hg := hC C[i] (List.getElem_mem hi)
  generalize C[i] = g at hg hnd ⊢
  unfold GoodG at hg
  cases hop : g.op with
  | input b s =>
    rw [hop] at hg
    obtain ⟨-, hlo, hhi⟩ := hg
    simp only [Gate.raw, hop, Op.raw]
    rw [hlo, hhi]
    exact hx s
  | budgeted cls f => rw [hop] at hg; exact hg.elim
  | _ => rw [hop] at hnd; exact absurd trivial hnd

/-! ## The codec's draw, read back from the coin bytes' bits -/

/-- The coin bytes: the `k` round constants', the mask's and word 1's tweak's, `n/8` each. -/
def NN (F : Field) (k : ℕ) : ℕ := (k + 2) * F.bytes

/-- Coin byte `j`'s slot. -/
def slotOf (F : Field) (k j : ℕ) : Slot :=
  if j < k * F.bytes then .const (j / F.bytes) (j % F.bytes)
  else if j < (k + 1) * F.bytes then .mask (j - k * F.bytes)
  else .tweak 1 (j - (k + 1) * F.bytes)

/-- The coin inputs on draw `d`: each slot's byte. -/
def xd (F : Field) (k : ℕ) (d : Draw k) (s : Slot) : ℤ := s.value F d (fun _ => 0)

lemma byte_le (N l : ℕ) : byte N l ≤ 255 := by
  have := Nat.mod_lt (N / 256 ^ l) (show 256 > 0 by norm_num)
  unfold byte
  omega

lemma xd_range (F : Field) (k : ℕ) (d : Draw k) (s : Slot) : 0 ≤ xd F k d s ∧ xd F k d s ≤ 255 := by
  have hb : ∀ N l, ((byte N l : ℕ) : ℤ) ≤ 255 := fun N l => by exact_mod_cast byte_le N l
  cases s with
  | payload i l => dsimp only [xd, Slot.value]; exact ⟨by positivity, hb _ _⟩
  | const r l =>
    dsimp only [xd, Slot.value]
    split_ifs
    · exact ⟨by positivity, hb _ _⟩
    · exact ⟨le_rfl, by norm_num⟩
  | mask l => dsimp only [xd, Slot.value]; exact ⟨by positivity, hb _ _⟩
  | tweak i l => dsimp only [xd, Slot.value]; exact ⟨by positivity, hb _ _⟩

lemma toNat_bit (T t : ℕ) : (decide (T / 2 ^ t % 2 = 1)).toNat = T / 2 ^ t % 2 := by
  rcases Nat.mod_two_eq_zero_or_one (T / 2 ^ t) with h | h <;> simp [h]

lemma bits_byte (T : ℕ) (hT : T < 256) :
    bitsVal ((List.range 8).map fun t => decide (T / 2 ^ t % 2 = 1)) 0 = T := by
  have hr : List.range 8 = [0, 1, 2, 3, 4, 5, 6, 7] := rfl
  rw [hr]
  simp only [List.map_cons, List.map_nil, bitsVal, toNat_bit]
  norm_num
  omega

/-- Byte `j`'s value, read back from its 8 bits. -/
def svN (β : ℕ → Bool) (j : ℕ) : ℕ := bitsVal ((List.range 8).map fun t => β (8 * j + t)) 0

/-- `n/8` bytes from `base`, read back as a number. -/
def decN (F : Field) (β : ℕ → Bool) (base : ℕ) : ℕ := ∑ l ∈ Finset.range F.bytes, 256 ^ l * svN β (base + l)

/-- The key the bits hold. -/
def keyOf (F : Field) (k : ℕ) (β : ℕ → Bool) : Key :=
  ⟨List.ofFn fun r : Fin k => decN F β (r.1 * F.bytes), decN F β (k * F.bytes)⟩

/-- The tweak multiplier the bits hold. -/
def tauOf (F : Field) (k : ℕ) (β : ℕ → Bool) : ℕ := decN F β ((k + 1) * F.bytes)

lemma asg_bit (N : ℕ) (sl : ℕ → ι) (x : ι → ℤ) (hx : ∀ s, 0 ≤ x s ∧ x s ≤ 255) (j t : ℕ) (hj : j < N)
    (ht : t < 8) : asg (C0 N sl) (wire N) x (8 * j + t) = decide ((x (sl j)).toNat / 2 ^ t % 2 = 1) := by
  have hv : 8 * j + t < 8 * N := by omega
  unfold asg wire
  rw [ite_eq_left hv, C0_val_bit N sl x _ hv, show (8 * j + t) / 8 = j by omega, show (8 * j + t) % 8 = t by omega]
  obtain ⟨h0, h255⟩ := hx (sl j)
  have h3 : x (sl j) % 2 ^ 8 = x (sl j) := Int.emod_eq_of_lt h0 (by norm_num; omega)
  have h4 : (x (sl j)).toNat < 256 := by omega
  rw [h3]
  by_cases h5 : (x (sl j)).toNat / 2 ^ t % 2 = 1 <;> simp [h4, h5]

lemma svN_eq (N : ℕ) (sl : ℕ → ι) (x : ι → ℤ) (hx : ∀ s, 0 ≤ x s ∧ x s ≤ 255) (j : ℕ) (hj : j < N) :
    svN (asg (C0 N sl) (wire N) x) j = (x (sl j)).toNat := by
  unfold svN
  rw [List.map_congr_left (fun t ht => asg_bit N sl x hx j t hj (List.mem_range.1 ht))]
  exact bits_byte _ (by have := hx (sl j); omega)

lemma sum_bytes (N : ℕ) : ∀ m, ∑ l ∈ Finset.range m, 256 ^ l * byte N l = N % 256 ^ m
  | 0 => by simp [Nat.mod_one]
  | m + 1 => by
    rw [Finset.sum_range_succ, sum_bytes N m, Nat.mod_pow_succ]
    unfold byte
    ring

lemma decN_eq (F : Field) (k : ℕ) (d : Draw k) (base : ℕ) (hb : base + F.bytes ≤ NN F k) :
    decN F (asg (C0 (NN F k) (slotOf F k)) (wire (NN F k)) (xd F k d)) base =
      ∑ l ∈ Finset.range F.bytes, 256 ^ l * (xd F k d (slotOf F k (base + l))).toNat := by
  unfold decN
  refine Finset.sum_congr rfl fun l hl => ?_
  rw [Finset.mem_range] at hl
  rw [svN_eq _ _ _ (xd_range F k d) _ (by omega)]

lemma slotOf_const (F : Field) (k r l : ℕ) (hr : r < k) (hl : l < F.bytes) :
    slotOf F k (r * F.bytes + l) = .const r l := by
  have hnb : 0 < F.bytes := by omega
  have h1 : (r + 1) * F.bytes ≤ k * F.bytes := Nat.mul_le_mul_right _ (Nat.succ_le_of_lt hr)
  have e : (r + 1) * F.bytes = r * F.bytes + F.bytes := by ring
  unfold slotOf
  rw [ite_eq_left (show r * F.bytes + l < k * F.bytes by omega)]
  rw [show r * F.bytes + l = l + r * F.bytes by ring, Nat.add_mul_div_right _ _ hnb, Nat.div_eq_of_lt hl, zero_add,
    Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hl]

lemma slotOf_mask (F : Field) (k l : ℕ) (hl : l < F.bytes) : slotOf F k (k * F.bytes + l) = .mask l := by
  have e : (k + 1) * F.bytes = k * F.bytes + F.bytes := by ring
  unfold slotOf
  rw [ite_eq_right (show ¬ k * F.bytes + l < k * F.bytes by omega),
    ite_eq_left (show k * F.bytes + l < (k + 1) * F.bytes by omega), show k * F.bytes + l - k * F.bytes = l by omega]

lemma slotOf_tweak (F : Field) (k l : ℕ) (hl : l < F.bytes) :
    slotOf F k ((k + 1) * F.bytes + l) = .tweak 1 l := by
  have e : (k + 1) * F.bytes = k * F.bytes + F.bytes := by ring
  unfold slotOf
  rw [ite_eq_right (show ¬ (k + 1) * F.bytes + l < k * F.bytes by omega),
    ite_eq_right (show ¬ (k + 1) * F.bytes + l < (k + 1) * F.bytes by omega),
    show (k + 1) * F.bytes + l - (k + 1) * F.bytes = l by omega]

lemma key_decode (F : Field) (k : ℕ) (d : Draw k) (hd : d ∈ draws F k) (hn : F.n = 8 * F.bytes) :
    keyOf F k (asg (C0 (NN F k) (slotOf F k)) (wire (NN F k)) (xd F k d)) = d.key ∧
      tauOf F k (asg (C0 (NN F k) (slotOf F k)) (wire (NN F k)) (xd F k d)) = d.2.2 := by
  unfold draws at hd
  rw [Finset.mem_product, Fintype.mem_piFinset, Finset.mem_product, Finset.mem_range, Finset.mem_Ico] at hd
  obtain ⟨hc, hK, hτ1, hτ⟩ := hd
  have hpn : F.p ≤ 2 ^ F.n := Nat.sub_le _ _
  have h256 : 2 ^ F.n = 256 ^ F.bytes := by rw [hn, pow_mul]; norm_num
  have hNN : NN F k = k * F.bytes + F.bytes + F.bytes := by unfold NN; ring
  have hsum : ∀ N, N < 256 ^ F.bytes → ∑ l ∈ Finset.range F.bytes, 256 ^ l * byte N l = N := fun N hN => by
    rw [sum_bytes, Nat.mod_eq_of_lt hN]
  refine ⟨?_, ?_⟩
  · unfold keyOf Draw.key
    congr 1
    · congr 1
      funext r
      have hr := r.2
      have h1 : (r.1 + 1) * F.bytes ≤ k * F.bytes := Nat.mul_le_mul_right _ (Nat.succ_le_of_lt hr)
      have e : (r.1 + 1) * F.bytes = r.1 * F.bytes + F.bytes := by ring
      rw [decN_eq F k d _ (by omega)]
      have hrc := Finset.mem_range.1 (hc r)
      rw [← hsum (d.1 r) (by omega)]
      refine Finset.sum_congr rfl fun l hl => ?_
      rw [Finset.mem_range] at hl
      rw [slotOf_const F k r.1 l hr hl]
      dsimp only [xd, Slot.value]
      rw [dite_eq_left hr]
      simp
    · rw [decN_eq F k d _ (by omega), ← hsum d.2.1 (by omega)]
      refine Finset.sum_congr rfl fun l hl => ?_
      rw [Finset.mem_range] at hl
      rw [slotOf_mask F k l hl]
      rfl
  · unfold tauOf
    rw [decN_eq F k d _ (by rw [add_mul, one_mul]; omega), ← hsum d.2.2 (by omega)]
    refine Finset.sum_congr rfl fun l hl => ?_
    rw [Finset.mem_range] at hl
    rw [slotOf_tweak F k l hl]
    dsimp only [xd, Slot.value, Draw.tweak]
    rw [mul_one, Nat.mod_eq_of_lt hτ]
    simp

/-! ## The targets' formulas -/

/-- Bit `t` of byte `l` of word `i`'s target, as a function of the bits. -/
def target (F : Field) (k : ℕ) (β : ℕ → Bool) (i l t : ℕ) : Bool :=
  decide (byte (lessParity (encodeWord F.p (keyOf F k β) (tauOf F k β * i % F.p) 0)) l / 2 ^ t % 2 = 1)

/-- Its formula over the coin bytes' bits. -/
noncomputable def fml (F : Field) (k i l t : ℕ) : BF :=
  Classical.choose (shannon (8 * NN F k) fun β => target F k β i l t)

lemma fml_eval (F : Field) (k i l t : ℕ) (β : ℕ → Bool) :
    (fml F k i l t).eval β = target F k (trunc (8 * NN F k) β) i l t :=
  Classical.choose_spec (shannon (8 * NN F k) fun β => target F k β i l t) β

/-- Output `o = i·(n/8) + l`'s 8 bits' formulas: byte `l` of word `i`'s target, for `q` words. -/
noncomputable def phis (F : Field) (k q : ℕ) : List (List BF) :=
  (List.range (q * F.bytes)).map fun o => (List.range 8).map fun t => fml F k (o / F.bytes) (o % F.bytes) t

lemma phis_length (F : Field) (k q : ℕ) : (phis F k q).length = q * F.bytes := by simp [phis]

lemma phis_getD (F : Field) (k q o : ℕ) (h : o < q * F.bytes) :
    (phis F k q).getD o [] = (List.range 8).map fun t => fml F k (o / F.bytes) (o % F.bytes) t := by
  rw [phis, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range h]
  rfl

lemma phis_len8 (F : Field) (k q : ℕ) : ∀ fs ∈ phis F k q, fs.length ≤ 8 := by
  intro fs hfs
  obtain ⟨o, -, rfl⟩ := List.mem_map.1 hfs
  simp

end Accounting.SecurityProofs.GatesBitwise

namespace Accounting.SecurityProofs.Gates

open Accounting.Gates Accounting.SecurityProofs.GatesBitwise

theorem EncodeChainRefuted : Accounting.Guarantees.Gates.EncodeChainRefuted := by
  intro F k B E εnum εden hε hεlt hk hp hp4 hn8 hroom hE H
  have hpn : F.p ≤ 2 ^ F.n := Nat.sub_le _ _
  have hn66 : 66 ≤ F.n := by
    by_contra h
    have : 2 ^ F.n < 2 ^ 66 := Nat.pow_lt_pow_right (by norm_num) (by omega)
    omega
  have hn : F.n = 8 * F.bytes := by unfold Field.bytes; omega
  have hnb : 0 < F.bytes := by omega
  have hNN : NN F k = (k + 2) * F.bytes := rfl
  have hqp : 72 * NN F k + 1 ≤ F.p := by rw [hNN]; omega
  obtain ⟨hext, hPr, hout⟩ := buildAll_spec (C0 (NN F k) (slotOf F k)) (wire (NN F k))
    (C0_base (NN F k) (slotOf F k)) (phis F k (72 * NN F k + 1)) (C0 (NN F k) (slotOf F k))
    (phis_len8 F k _) (Ext.refl _) (C0_proven (NN F k) (slotOf F k))
  generalize buildAll (wire (NN F k)) (phis F k (72 * NN F k + 1)) (C0 (NN F k) (slotOf F k)) = R at hext hPr hout
  obtain ⟨Cf, rs⟩ := R
  dsimp only at hext hPr hout
  obtain ⟨Et, rfl, hEt⟩ := hext
  have hgood : ∀ g ∈ C0 (NN F k) (slotOf F k) ++ Et, GoodG g := fun g hg =>
    (List.mem_append.1 hg).elim (C0_good _ _ g) (fun h => good_of_free g (hEt g h))
  have hob : OperandBits B (C0 (NN F k) (slotOf F k) ++ Et) := by
    intro g hg cls f hop
    have := hgood g hg
    unfold GoodG at this
    rw [hop] at this
    exact this.elim
  have hci : CodecInputs F (C0 (NN F k) (slotOf F k) ++ Et) (draws F k) (fun _ => 0) (xd F k) := by
    refine ⟨fun i ⟨g, hg, hop⟩ => ?_, fun i _ => Or.inl ⟨i, fun d _ => rfl⟩⟩
    have := hgood g hg
    unfold GoodG at this
    rw [hop] at this
    exact absurd this.1 (by decide)
  have hd0 : ((fun _ => 0, 0, 1) : Draw k) ∈ draws F k := by
    unfold draws
    rw [Finset.mem_product, Fintype.mem_piFinset, Finset.mem_product]
    exact ⟨fun _ => Finset.mem_range.2 (show 0 < F.p by omega), Finset.mem_range.2 (Nat.two_pow_pos _),
      Finset.mem_Ico.2 ⟨le_rfl, show 1 < F.p by omega⟩⟩
  have hcard : 0 < (draws F k).card := Finset.card_pos.mpr ⟨_, hd0⟩
  have hcw : ComputesWords F k (C0 (NN F k) (slotOf F k) ++ Et) (xd F k) (fun _ => 0)
      (List.range (72 * NN F k + 1)) (fun i l => rs.getD (i * F.bytes + l) 0) εnum εden := by
    refine ⟨draws F k, Finset.Subset.refl _, Nat.mul_lt_mul_of_pos_right hεlt hcard, fun d hd => ⟨?_, ?_⟩⟩
    · exact declared_of_good _ _ hgood (xd_range F k d)
    · intro i hi l hl
      rw [List.mem_range] at hi
      have hlt : i * F.bytes + l < (72 * NN F k + 1) * F.bytes := by
        have : (i + 1) * F.bytes ≤ (72 * NN F k + 1) * F.bytes := Nat.mul_le_mul_right _ hi
        rw [add_mul, one_mul] at this
        omega
      have h1 : (i * F.bytes + l) / F.bytes = i := by
        rw [add_comm, Nat.add_mul_div_right _ _ hnb, Nat.div_eq_of_lt hl, zero_add]
      have h2 : (i * F.bytes + l) % F.bytes = l := by
        rw [add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hl]
      obtain ⟨hkey, htau⟩ := key_decode F k d hd hn
      have hfun : ∀ t, (fml F k i l t).eval (asg (C0 (NN F k) (slotOf F k)) (wire (NN F k)) (xd F k d)) =
          decide (byte (lessParity (encodeWord F.p d.key (d.tweak F i) 0)) l / 2 ^ t % 2 = 1) := by
        intro t
        rw [fml_eval, trunc_asg]
        unfold target
        rw [hkey, htau]
        rfl
      have hmap : ((List.range 8).map fun t => fml F k i l t).map
          (fun f => f.eval (asg (C0 (NN F k) (slotOf F k)) (wire (NN F k)) (xd F k d))) =
          (List.range 8).map fun t =>
            decide (byte (lessParity (encodeWord F.p d.key (d.tweak F i) 0)) l / 2 ^ t % 2 = 1) := by
        rw [List.map_map]
        exact List.map_congr_left fun t _ => hfun t
      rw [hout _ (by rw [phis_length]; exact hlt) (xd F k d), phis_getD F k _ _ hlt, h1, h2, hmap,
        bits_byte _ (Nat.lt_succ_of_le (byte_le _ _))]
  have key := H hε hk hp hp4 (C0 (NN F k) (slotOf F k) ++ Et) (fun _ => 0) (xd F k)
    (List.range (72 * NN F k + 1)) (fun i l => rs.getD (i * F.bytes + l) 0) ⟨hPr, hob⟩ hci List.nodup_range
    (fun i hi => by rw [List.mem_range] at hi; omega) hcw
  have hrun : runBits (C0 (NN F k) (slotOf F k) ++ Et) = 0 := by
    refine List.sum_eq_zero fun y hy => ?_
    obtain ⟨g, hg, rfl⟩ := List.mem_map.1 hy
    have := hgood g hg
    unfold GoodG at this
    unfold Gate.inputBits
    split
    · rename_i b' s heq
      rw [heq] at this
      rw [ite_eq_right (by rw [this.1]; decide)]
    · rfl
  have hbud : budgeted (C0 (NN F k) (slotOf F k) ++ Et) ≤ 72 * NN F k := by
    refine (budgeted_le _).trans ?_
    rw [List.map_append, List.sum_append, sum_ubG_free Et hEt, add_zero]
    exact C0_ub _ _
  rw [List.length_range, hrun, zero_mul, add_zero] at key
  have hn1 : 0 < F.n - 1 := by omega
  have h1 : (72 * NN F k + 1) * E ≤ budgeted (C0 (NN F k) (slotOf F k) ++ Et) := Nat.le_of_mul_le_mul_right key hn1
  have h2 : 72 * NN F k + 1 ≤ (72 * NN F k + 1) * E := Nat.le_mul_of_pos_right _ hE
  omega

end Accounting.SecurityProofs.Gates
