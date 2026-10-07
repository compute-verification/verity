/-
Level 3: a lookup slot's rows compute `table[index]` (`Lookup.build`, PROTOCOL.md §16.5). Over any field of
characteristic 2, a witness satisfying every row of the slot (each row `z_r = (A_r·z)(B_r·z)`, the product rows' B side
the table's, as `foldB` folds it), with the constant 1, has bit `j` of the output word equal to bit `j` of the table word
the index bits name. The rows force it: the index bits are bits (`z = z·z`), each decoder is one-hot on its bits, each
product row picks one table bit, and each output bit sums them.
-/
import Mathlib.Algebra.CharP.Two
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.BigOperators.Group.List.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Tactic.Ring
import Mathlib.Tactic.IntervalCases
import Mathlib.Tactic.Linarith
import Flock.Lookup

open Flock Lookup

set_option linter.unusedSectionVars false

namespace FlockLevel3

section
variable {K : Type*} [Field K] [CharP K 2] [DecidableEq K]

/-- A column's value: `KONST` (the constant, before `build` resolves it) is 1. -/
def colVal (z : ℕ → K) (c : ℕ) : K := if c = KONST then 1 else z c

/-- A row's linear form over `z`. -/
def rowVal (z : ℕ → K) (cs : Array ℕ) : K := (cs.toList.map (colVal z)).sum

/-- A form's value: its columns, plus 1 for the constant. -/
def formVal (z : ℕ → K) (f : Form) : K := rowVal z (cols f)

/-- Row `r` of the builder holds under `z`. -/
def RowOk (z : ℕ → K) (bd : Builder) (r : ℕ) : Prop := z r = rowVal z (bd.a.getD r #[]) * rowVal z (bd.b.getD r #[])

/-- The value of the bits `bs` (low bit first), each `1` when `z` is 1 there. -/
def bitsVal (z : ℕ → K) : List ℕ → ℕ
  | [] => 0
  | b :: bs => (if z b = 1 then 1 else 0) + 2 * bitsVal z bs

theorem bitsVal_append (z : ℕ → K) : ∀ l₁ l₂ : List ℕ, bitsVal z (l₁ ++ l₂) = bitsVal z l₁ + 2 ^ l₁.length * bitsVal z l₂
  | [], l₂ => by simp [bitsVal]
  | b :: l₁, l₂ => by
    simp only [List.cons_append, bitsVal, bitsVal_append z l₁ l₂, List.length_cons, pow_succ]
    ring

theorem bitsVal_lt (z : ℕ → K) : ∀ l : List ℕ, bitsVal z l < 2 ^ l.length
  | [] => by simp [bitsVal]
  | b :: l => by
    have := bitsVal_lt z l
    simp only [bitsVal, List.length_cons, pow_succ]
    split <;> omega

/-- In characteristic 2, a value that is 0 or 1, plus 1, is the other. -/
theorem one_add_of_bit {x : K} (hx : x = 0 ∨ x = 1) : 1 + x = if x = 1 then 0 else 1 := by
  rcases hx with rfl | rfl
  · simp
  · simp [CharTwo.add_self_eq_zero]

/-- In a field, `x = x·x` makes `x` a bit. -/
theorem bit_of_idem {x : K} (h : x = x * x) : x = 0 ∨ x = 1 := by
  by_cases h0 : x = 0
  · exact Or.inl h0
  · right
    have h1 : x * 1 = x * x := by rw [mul_one]; exact h
    exact (mul_left_cancel₀ h0 h1).symm

/-! ### The builder -/

/-- `decode`'s combining step, folded over `List.range N` from `(bd, fs)`: row `i` has `A = cols (f i).1` and
`B = cols (f i).2`, pushed at `bd.a.size + i`, and its form is that single row. -/
def combine (f : ℕ → Form × Form) (acc : Builder × Array Form) (i : ℕ) : Builder × Array Form :=
  let (bd, r) := row acc.1 (f i).1 (f i).2
  (bd, acc.2.push (#[r], false))

theorem combine_spec (f : ℕ → Form × Form) : ∀ (N : ℕ) (bd : Builder) (fs : Array Form),
    ((List.range N).foldl (combine f) (bd, fs)).1.a = bd.a ++ ((List.range N).map fun i => cols (f i).1).toArray ∧
    ((List.range N).foldl (combine f) (bd, fs)).1.b = bd.b ++ ((List.range N).map fun i => cols (f i).2).toArray ∧
    ((List.range N).foldl (combine f) (bd, fs)).2 = fs ++ ((List.range N).map fun i => ((#[bd.a.size + i], false) : Form)).toArray
  | 0, bd, fs => by simp
  | N + 1, bd, fs => by
    obtain ⟨ha, hb, hf⟩ := combine_spec f N bd fs
    rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
    simp only [combine, row, ha, hb, hf]
    refine ⟨?_, ?_, ?_⟩ <;> (apply Array.ext' ; simp)

theorem decode_combine (bd0 : Builder) (bits : List ℕ) (fuel : ℕ) (hs : ∀ b, bits ≠ [b]) :
    decode bd0 bits (fuel + 1) =
      let h := bits.length / 2
      let d1 := decode bd0 (bits.take h) fuel
      let d2 := decode d1.1 (bits.drop h) fuel
      let c := (List.range (2 ^ bits.length)).foldl
        (combine fun i => (d1.2.1.getD (i % 2 ^ h) default, d2.2.1.getD (i / 2 ^ h) default)) (d2.1, #[])
      (c.1, c.2, d2.1.a.size) := by
  rw [decode.eq_3 _ _ _ (fun b hb => hs b hb)]
  rfl

/-- The rows appended after `bd`: `A` and `B` together. -/
def Appends (bd bd' : Builder) : Prop :=
  ∃ ra rb : List (Array ℕ), bd'.a = bd.a ++ ra.toArray ∧ bd'.b = bd.b ++ rb.toArray ∧ ra.length = rb.length

theorem Appends.refl (bd : Builder) : Appends bd bd := ⟨[], [], by simp, by simp, rfl⟩

theorem Appends.trans {x y w : Builder} (h₁ : Appends x y) (h₂ : Appends y w) : Appends x w := by
  obtain ⟨a₁, b₁, ha₁, hb₁, hl₁⟩ := h₁
  obtain ⟨a₂, b₂, ha₂, hb₂, hl₂⟩ := h₂
  exact ⟨a₁ ++ a₂, b₁ ++ b₂, by rw [ha₂, ha₁]; simp, by rw [hb₂, hb₁]; simp, by simp [hl₁, hl₂]⟩

theorem Appends.size {x y : Builder} (h : Appends x y) : x.a.size ≤ y.a.size ∧ (x.a.size = x.b.size → y.a.size = y.b.size) := by
  obtain ⟨a, b, ha, hb, hl⟩ := h
  rw [ha, hb]; simp only [Array.size_append, List.size_toArray]; omega

theorem Appends.getD {x y : Builder} (h : Appends x y) (hxs : x.a.size = x.b.size) {r : ℕ} (hr : r < x.a.size) :
    y.a.getD r #[] = x.a.getD r #[] ∧ y.b.getD r #[] = x.b.getD r #[] := by
  obtain ⟨a, b, ha, hb, hl⟩ := h
  rw [ha, hb]
  simp only [Array.getD_eq_getD_getElem?, Array.getElem?_append_left hr, Array.getElem?_append_left (hxs ▸ hr)]
  exact ⟨trivial, trivial⟩

theorem combine_appends (f : ℕ → Form × Form) (N : ℕ) (bd : Builder) (fs : Array Form) :
    Appends bd ((List.range N).foldl (combine f) (bd, fs)).1 := by
  obtain ⟨ha, hb, -⟩ := combine_spec f N bd fs
  exact ⟨_, _, ha, hb, by simp⟩

theorem decode_single (bd0 : Builder) (b fuel : ℕ) : decode bd0 [b] fuel = (bd0, #[(#[b], true), (#[b], false)], 0) := by
  cases fuel <;> rfl

theorem colVal_of_ne (z : ℕ → K) {c : ℕ} (h : c ≠ KONST) : colVal z c = z c := by simp [colVal, h]

theorem formVal_row (z : ℕ → K) {r : ℕ} (h : r ≠ KONST) : formVal z ((#[r], false) : Form) = z r := by
  simp [formVal, cols, rowVal, colVal_of_ne z h]

theorem formVal_bit (z : ℕ → K) {b : ℕ} (hb : b ≠ KONST) (hz : z b = 0 ∨ z b = 1) (i : ℕ) (hi : i < 2) :
    formVal z ((#[(#[b], true), (#[b], false)] : Array Form)[i]!) = if bitsVal z [b] = i then 1 else 0 := by
  have hk : colVal z KONST = 1 := by simp [colVal]
  interval_cases i
  · simp only [formVal, cols, rowVal, bitsVal]
    simp [colVal_of_ne z hb, hk]
    rcases hz with h | h <;> simp [h, CharTwo.add_self_eq_zero]
  · simp only [formVal, cols, rowVal, bitsVal]
    simp [colVal_of_ne z hb]
    rcases hz with h | h <;> simp [h]

theorem split_eq {v₁ v₂ i p : ℕ} (hp : 0 < p) (hv : v₁ < p) : (v₁ = i % p ∧ v₂ = i / p) ↔ v₁ + p * v₂ = i := by
  constructor
  · rintro ⟨rfl, rfl⟩; exact Nat.mod_add_div i p
  · rintro rfl
    refine ⟨?_, ?_⟩
    · rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hv]
    · rw [Nat.add_mul_div_left _ _ hp, Nat.div_eq_of_lt hv, zero_add]

/-- **The decoders.** `decode` appends rows to `A` and `B` together and returns `2^|bits|` forms; for two bits or more,
form `i` is the single row `first + i`, the block ending the rows. When the new rows hold and `bits` are bits, form `i` is
1 exactly when the bits' value is `i`: each decoder is one-hot. -/
theorem decode_spec (z : ℕ → K) : ∀ (fuel : ℕ) (bd : Builder) (bits : List ℕ), bits ≠ [] → bits.length ≤ fuel + 1 →
    Appends bd (decode bd bits fuel).1 ∧ (decode bd bits fuel).2.1.size = 2 ^ bits.length ∧
    (2 ≤ bits.length → (decode bd bits fuel).2.2 + 2 ^ bits.length = (decode bd bits fuel).1.a.size ∧
      ∀ i < 2 ^ bits.length, (decode bd bits fuel).2.1.getD i default = (#[(decode bd bits fuel).2.2 + i], false)) ∧
    (bd.a.size = bd.b.size → (∀ b ∈ bits, b ≠ KONST ∧ (z b = 0 ∨ z b = 1)) → (decode bd bits fuel).1.a.size ≤ KONST →
      (∀ r, bd.a.size ≤ r → r < (decode bd bits fuel).1.a.size → RowOk z (decode bd bits fuel).1 r) →
      ∀ i < 2 ^ bits.length, formVal z ((decode bd bits fuel).2.1.getD i default) = if bitsVal z bits = i then 1 else 0) := by
  intro fuel
  induction fuel with
  | zero =>
    intro bd bits hne hlen
    obtain ⟨b, rfl⟩ : ∃ b, bits = [b] := by
      match bits, hne, hlen with
      | [b], _, _ => exact ⟨b, rfl⟩
    rw [decode_single]
    refine ⟨Appends.refl bd, rfl, fun h => by simp at h, fun _ hb _ _ i hi => ?_⟩
    have := formVal_bit z (hb b (by simp)).1 (hb b (by simp)).2 i (by simpa using hi)
    simpa [Array.getD_eq_getD_getElem?, Array.getElem!_eq_getD] using this
  | succ f ih =>
    intro bd bits hne hlen
    by_cases hs : ∃ b, bits = [b]
    · obtain ⟨b, rfl⟩ := hs
      rw [decode_single]
      refine ⟨Appends.refl bd, rfl, fun h => by simp at h, fun _ hb _ _ i hi => ?_⟩
      have := formVal_bit z (hb b (by simp)).1 (hb b (by simp)).2 i (by simpa using hi)
      simpa [Array.getD_eq_getD_getElem?, Array.getElem!_eq_getD] using this
    · push Not at hs
      have h2 : 2 ≤ bits.length := by
        rcases bits with _ | ⟨x, _ | ⟨y, t⟩⟩
        · exact absurd rfl hne
        · exact absurd rfl (hs x)
        · simp
      rw [decode_combine _ _ _ hs]
      dsimp only
      set h := bits.length / 2 with hh
      have hb4 : 1 ≤ h ∧ h < bits.length ∧ bits.length - h ≤ f + 1 ∧ h ≤ f + 1 := by omega
      have htl : (bits.take h).length = h := by simp; omega
      have hdl : (bits.drop h).length = bits.length - h := by simp
      obtain ⟨A1, S1, -, V1⟩ := ih bd (bits.take h)
        (by intro e; have := congrArg List.length e; rw [htl] at this; simp at this; omega) (by rw [htl]; omega)
      set d1 := decode bd (bits.take h) f
      obtain ⟨A2, S2, -, V2⟩ := ih d1.1 (bits.drop h)
        (by intro e; have := congrArg List.length e; rw [hdl] at this; simp at this; omega) (by rw [hdl]; omega)
      set d2 := decode d1.1 (bits.drop h) f
      set g : ℕ → Form × Form := fun i => (d1.2.1.getD (i % 2 ^ h) default, d2.2.1.getD (i / 2 ^ h) default)
      obtain ⟨ca, cb, cf⟩ := combine_spec g (2 ^ bits.length) d2.1 #[]
      have A3 := combine_appends g (2 ^ bits.length) d2.1 #[]
      set c := (List.range (2 ^ bits.length)).foldl (combine g) (d2.1, #[])
      have hcs : c.1.a.size = d2.1.a.size + 2 ^ bits.length := by rw [ca]; simp
      refine ⟨A1.trans (A2.trans A3), by rw [cf]; simp, fun _ => ⟨hcs.symm, fun i hi => ?_⟩, ?_⟩
      · rw [cf]; simp [Array.getD_eq_getD_getElem?, hi]
      intro hsz hb hK hok i hi
      have hs1 := (A1.size).2 hsz
      have hs2 := (A2.size).2 hs1
      have hle1 := (A1.size).1
      have hle2 := (A2.size).1
      -- the sub-decoders' rows hold: they are rows of `c.1` too
      have ok1 : ∀ r, bd.a.size ≤ r → r < d1.1.a.size → RowOk z d1.1 r := by
        intro r h1 h2'
        have e := (A2.trans A3).getD hs1 h2'
        have := hok r h1 (by omega)
        simpa [RowOk, e.1, e.2] using this
      have ok2 : ∀ r, d1.1.a.size ≤ r → r < d2.1.a.size → RowOk z d2.1 r := by
        intro r h1 h2'
        have e := A3.getD hs2 h2'
        have := hok r (by omega) (by omega)
        simpa [RowOk, e.1, e.2] using this
      have v1 := V1 hsz (fun b hm => hb b (List.mem_of_mem_take hm)) (by omega) ok1
      have v2 := V2 hs1 (fun b hm => hb b (List.mem_of_mem_drop hm)) (by omega) ok2
      -- row `first + i` is the product of the two halves' forms
      have hr := hok (d2.1.a.size + i) (by omega) (by omega)
      have ga : c.1.a.getD (d2.1.a.size + i) #[] = cols (g i).1 := by rw [ca]; simp [Array.getD_eq_getD_getElem?, hi]
      have gb : c.1.b.getD (d2.1.a.size + i) #[] = cols (g i).2 := by rw [cb, hs2]; simp [Array.getD_eq_getD_getElem?, hi]
      have gf : c.2.getD i default = (#[d2.1.a.size + i], false) := by rw [cf]; simp [Array.getD_eq_getD_getElem?, hi]
      rw [gf, formVal_row z (by omega)]
      rw [RowOk, ga, gb] at hr
      rw [hr]
      change formVal z (g i).1 * formVal z (g i).2 = _
      simp only [g]
      have hp : 0 < 2 ^ h := Nat.two_pow_pos h
      rw [v1 (i % 2 ^ h) (by rw [htl]; exact Nat.mod_lt _ hp),
        v2 (i / 2 ^ h) (by rw [hdl]; exact (Nat.div_lt_iff_lt_mul hp).2 (by rw [← pow_add]; convert hi using 2; omega))]
      have hsplit : bitsVal z bits = bitsVal z (bits.take h) + 2 ^ h * bitsVal z (bits.drop h) := by
        conv_lhs => rw [← List.take_append_drop h bits]
        rw [bitsVal_append, htl]
      rw [hsplit]
      have key := split_eq (v₂ := bitsVal z (bits.drop h)) (i := i) hp
        (by have := bitsVal_lt z (bits.take h); rwa [htl] at this)
      by_cases e : bitsVal z (List.take h bits) + 2 ^ h * bitsVal z (List.drop h bits) = i
      · obtain ⟨e1, e2⟩ := key.2 e
        simp [e1, e2, Nat.mod_add_div]
      · by_cases e1 : bitsVal z (List.take h bits) = i % 2 ^ h
        · have e2 : ¬ bitsVal z (List.drop h bits) = i / 2 ^ h := fun e2 => e (key.1 ⟨e1, e2⟩)
          simp only [e1, e2, ↓reduceIte, mul_zero]
          rw [e1] at e
          simp [e]
        · simp only [e1, ↓reduceIte, zero_mul]
          rw [ite_eq_right_of_eq_false _ _ (eq_false fun e' => e1 (key.2 e').1)]

/-! ### `build`'s loops -/

theorem forIn_range_foldl {σ : Type} (k : ℕ) (s : σ) (g : ℕ → σ → σ) :
    (forIn [:k] s fun i r => pure (ForInStep.yield (g i r)) : V σ) = .ok ((List.range k).foldl (fun r i => g i r) s) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl]
  simp [Std.Legacy.Range.size, List.range_eq_range']
  rfl

/-- A `while` loop that returned: `Q` holds of its result when every step from a `P`-state either stops in a `Q`-state or
continues in a `P`-state of lower measure. -/
theorem loop_spec {σ ε : Type} (P Q : σ → Prop) (μ : σ → ℕ) {f : Unit → σ → Except ε (ForInStep σ)}
    (hf : ∀ s v, P s → f () s = .ok v → (∀ s', v = .done s' → Q s') ∧ ∀ s', v = .yield s' → P s' ∧ μ s' < μ s) :
    ∀ (s r : σ), P s → Lean.Loop.forIn Lean.Loop.mk s f = .ok r → Q r := by
  intro s
  induction hμ : μ s using Nat.strong_induction_on generalizing s with
  | _ N ih =>
    intro r hs h
    rw [Lean.Loop.forIn_eq_of_monadTail] at h
    rcases hv : f () s with e | v
    · rw [hv] at h; cases h
    · rw [hv] at h
      obtain ⟨hd, hy⟩ := hf s v hs hv
      cases v with
      | done val => cases h; exact hd _ rfl
      | yield val =>
        obtain ⟨hP, hlt⟩ := hy val rfl
        exact ih (μ val) (hμ ▸ hlt) val rfl r hP h

/-- `build`'s padding to a word: empty rows pushed to `A` and `B` until `A` is a whole number of words. -/
theorem while_pad_spec {s s' : Builder}
    (h : (forIn Lean.Loop.mk s fun _ r =>
      if (r.a.size % 128 != 0) = true then
        pure (ForInStep.yield ({ a := r.a.push #[], b := r.b.push #[] } : Builder))
      else pure (ForInStep.done r) : V Builder) = .ok s') :
    ∃ k, s'.a = s.a ++ (List.replicate k (#[] : Array ℕ)).toArray ∧ s'.b = s.b ++ (List.replicate k (#[] : Array ℕ)).toArray ∧
      s'.a.size % 128 = 0 := by
  refine loop_spec (fun r : Builder => ∃ k, r.a = s.a ++ (List.replicate k (#[] : Array ℕ)).toArray ∧
      r.b = s.b ++ (List.replicate k (#[] : Array ℕ)).toArray)
    (fun r => ∃ k, r.a = s.a ++ (List.replicate k (#[] : Array ℕ)).toArray ∧
      r.b = s.b ++ (List.replicate k (#[] : Array ℕ)).toArray ∧ r.a.size % 128 = 0)
    (fun r => (128 - r.a.size % 128) % 128) (fun r v hr hv => ?_) s s' ⟨0, by simp, by simp⟩ h
  obtain ⟨k, ha, hb⟩ := hr
  by_cases hc : (r.a.size % 128 != 0) = true
  · simp only [hc, ↓reduceIte] at hv
    cases hv
    refine ⟨fun s' e => (by cases e), fun s' e => ?_⟩
    cases e
    simp only [bne_iff_ne, ne_eq] at hc
    refine ⟨⟨k + 1, ?_, ?_⟩, ?_⟩
    · rw [ha]; apply Array.ext'; simp [List.replicate_succ']
    · rw [hb]; apply Array.ext'; simp [List.replicate_succ']
    · simp only [Array.size_push]; omega
  · simp only [hc, Bool.false_eq_true, ↓reduceIte] at hv
    cases hv
    simp only [bne_iff_ne, ne_eq, not_not] at hc
    exact ⟨fun s' e => (by cases e; exact ⟨k, ha, hb, hc⟩), fun s' e => (by cases e)⟩

/-! ### What `build` builds -/

/-- The input rows: row `i < 128` reads column `i` in `A` and in `B` when `i < n`, and nothing otherwise. -/
def inputRows (n : ℕ) : Builder :=
  List.foldl (fun (r : Builder) i => if i < n then { a := r.a.push #[i], b := r.b.push #[i] }
    else { a := r.a.push #[], b := r.b.push #[] }) { } (List.range 128)

/-- The product rows: for each output bit, one row per high minterm `h`, reading it in `A` (the `B` side is the table's). -/
def productRows (hiM : Array Form) (bits : ℕ) (s : Builder) : Builder :=
  List.foldl (fun (r : Builder) _ => List.foldl (fun (r : Builder) i =>
    { a := r.a.push (cols (hiM.getD i default)), b := r.b.push #[] }) r (List.range hiM.size)) s (List.range bits)

/-- The output rows: bit `j < bits` sums its product rows in `A`, times the constant in `B`. -/
def outputRows (prod0 nh bits : ℕ) (s : Builder) : Builder :=
  List.foldl (fun (r : Builder) i => if i < bits then
      { a := r.a.push (Array.map (fun x => prod0 + i * nh + x) (Array.range nh)), b := r.b.push #[KONST] }
    else { a := r.a.push #[], b := r.b.push #[] }) s (List.range 128)

/-- `KONST` resolved to the constant's column. -/
def fixKonst (k : ℕ) (rs : Array (Array ℕ)) : Array (Array ℕ) :=
  Array.map (fun r => Array.map (fun c => if (c == KONST) = true then k else c) r) rs

/-- **`build`, step by step.** An accepted slot passed its checks, and its net is the input rows, the two decoders, the
product rows, padding to a word, the output rows and the constant, with `KONST` resolved and encoded sparse. -/
theorem build_spec {name : String} {table : ByteArray} {n lo bits : ℕ} {net : Net} (h : build name table n lo bits = .ok net) :
    2 ≤ lo ∧ lo < n ∧ bits ≤ 32 ∧ n ≤ 128 ∧ table.size = 4 * 2 ^ n ∧
    ∃ (d1 d2 : Builder × Array Form × ℕ) (r3 : Builder),
      d1 = decode (inputRows n) (List.range lo) lo ∧
      d2 = decode d1.1 ((List.range (n - lo)).map (lo + ·)) (n - lo) ∧
      d1.2.1.size = 2 ^ lo ∧ d2.2.1.size = 2 ^ (n - lo) ∧
      (∃ k, r3.a = (productRows d2.2.1 bits d2.1).a ++ (List.replicate k (#[] : Array ℕ)).toArray ∧
        r3.b = (productRows d2.2.1 bits d2.1).b ++ (List.replicate k (#[] : Array ℕ)).toArray ∧ r3.a.size % 128 = 0) ∧
      Sparse.ofRows (fixKonst (outputRows d2.1.a.size d2.2.1.size bits r3).a.size
          ((outputRows d2.1.a.size d2.2.1.size bits r3).a.push #[(outputRows d2.1.a.size d2.2.1.size bits r3).a.size]))
          (2 ^ net.unitLog) = .ok net.a ∧
      Sparse.ofRows (fixKonst (outputRows d2.1.a.size d2.2.1.size bits r3).a.size
          ((outputRows d2.1.a.size d2.2.1.size bits r3).b.push #[(outputRows d2.1.a.size d2.2.1.size bits r3).a.size]))
          (2 ^ net.unitLog) = .ok net.b ∧
      net.useful = (outputRows d2.1.a.size d2.2.1.size bits r3).a.size + 1 ∧
      net.constPos = (outputRows d2.1.a.size d2.2.1.size bits r3).a.size ∧
      net.outGroups = #[⟨r3.a.size / 128, 1, #[bits]⟩] ∧
      net.lookup = some ⟨table, n, lo, bits, d2.1.a.size, d1.2.2⟩ := by
  simp only [build] at h
  split at h
  · simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h
  rename_i hchk
  simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, Bool.not_eq_true', Bool.and_eq_false_iff, decide_eq_false_iff_not,
    decide_eq_true_eq, not_or, not_lt, not_not, not_le] at hchk
  obtain ⟨⟨⟨hts, hlo, hln⟩, hb32⟩, hn⟩ := hchk
  refine ⟨hlo, hln, hb32, hn, hts, ?_⟩
  simp only [forIn_range_foldl, bind, Except.bind] at h
  set d1 := decode (List.foldl (fun (r : Builder) i => if i < n then { a := r.a.push #[i], b := r.b.push #[i] }
    else { a := r.a.push #[], b := r.b.push #[] }) { } (List.range 128)) (List.range lo) lo with hd1
  set d2 := decode d1.1 (List.map (fun x => lo + x) (List.range (n - lo))) (n - lo) with hd2
  rcases hW : (forIn Lean.Loop.mk (productRows d2.2.1 bits d2.1) fun _ s => if (s.a.size % 128 != 0) = true then
      pure (ForInStep.yield ({ a := s.a.push #[], b := s.b.push #[] } : Builder)) else pure (ForInStep.done s) :
      V Builder) with e | r3
  · rw [productRows] at hW; rw [hW] at h; cases h
  have hpad := while_pad_spec hW
  rw [productRows] at hW
  rw [hW] at h
  simp only at h
  split at h
  · simp [throw, throwThe, MonadExceptOf.throw] at h
  rename_i hshape
  simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, not_not] at hshape
  set r4 := List.foldl (fun (r : Builder) i => if i < bits then
      { a := r.a.push (Array.map (fun x => d2.1.a.size + i * d2.2.1.size + x) (Array.range d2.2.1.size)),
        b := r.b.push #[KONST] }
    else { a := r.a.push #[], b := r.b.push #[] }) r3 (List.range 128) with hr4
  rcases hO : Net.checkOrder (toString "lookup " ++ toString name) 128 r4.a.size
      (Array.map (fun r => Array.map (fun c => if (c == KONST) = true then r4.a.size else c) r) (r4.a.push #[r4.a.size]))
      (Array.map (fun r => Array.map (fun c => if (c == KONST) = true then r4.a.size else c) r) (r4.b.push #[r4.a.size]))
    with e | u
  · rw [hO] at h; cases h
  rw [hO] at h
  simp only at h
  rcases hA : Sparse.ofRows (Array.map (fun r => Array.map (fun c => if (c == KONST) = true then r4.a.size else c) r)
      (r4.a.push #[r4.a.size])) (2 ^ max 7 (log2ceil (r4.a.push #[r4.a.size]).size)) with e | A
  · rw [hA] at h; cases h
  rw [hA] at h
  simp only at h
  rcases hB : Sparse.ofRows (Array.map (fun r => Array.map (fun c => if (c == KONST) = true then r4.a.size else c) r)
      (r4.b.push #[r4.a.size])) (2 ^ max 7 (log2ceil (r4.a.push #[r4.a.size]).size)) with e | B
  · rw [hB] at h; cases h
  rw [hB] at h
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  exact ⟨d1, d2, r3, rfl, rfl, hshape.1, hshape.2, hpad, hA, hB, by simp [r4, outputRows], rfl, rfl, rfl⟩

/-! ### The rows of each stage -/

theorem foldl_push (f g : ℕ → Array ℕ) : ∀ (N : ℕ) (s : Builder),
    (List.range N).foldl (fun (r : Builder) i => ({ a := r.a.push (f i), b := r.b.push (g i) } : Builder)) s =
      { a := s.a ++ ((List.range N).map f).toArray, b := s.b ++ ((List.range N).map g).toArray }
  | 0, s => by simp
  | N + 1, s => by
    rw [List.range_succ, List.foldl_append, foldl_push f g N s]
    simp only [List.foldl_cons, List.foldl_nil, List.map_append, List.map_cons, List.map_nil]
    congr 1 <;> (apply Array.ext'; simp)

theorem inputRows_eq_gen (N n : ℕ) :
    List.foldl (fun (r : Builder) i => if i < n then { a := r.a.push #[i], b := r.b.push #[i] }
      else { a := r.a.push #[], b := r.b.push #[] }) { } (List.range N) =
    { a := ((List.range N).map fun i => if i < n then #[i] else #[]).toArray,
      b := ((List.range N).map fun i => if i < n then #[i] else #[]).toArray } := by
  have : (fun (r : Builder) i => if i < n then ({ a := r.a.push #[i], b := r.b.push #[i] } : Builder)
      else { a := r.a.push #[], b := r.b.push #[] }) =
      fun (r : Builder) i => ({ a := r.a.push (if i < n then #[i] else #[]), b := r.b.push (if i < n then #[i] else #[]) } : Builder) := by
    funext r i; split <;> rfl
  rw [this, foldl_push]
  congr 1 <;> exact Array.empty_append

theorem inputRows_eq (n : ℕ) : inputRows n =
    { a := ((List.range 128).map fun i => if i < n then #[i] else #[]).toArray,
      b := ((List.range 128).map fun i => if i < n then #[i] else #[]).toArray } :=
  inputRows_eq_gen 128 n

theorem productRows_eq (hiM : Array Form) (bits : ℕ) (s : Builder) : productRows hiM bits s =
    { a := s.a ++ ((List.range (bits * hiM.size)).map fun t => cols (hiM.getD (t % hiM.size) default)).toArray,
      b := s.b ++ (List.replicate (bits * hiM.size) (#[] : Array ℕ)).toArray } := by
  unfold productRows
  induction bits with
  | zero => simp
  | succ J ih =>
    rw [List.range_succ, List.foldl_append, ih, List.foldl_cons, List.foldl_nil,
      foldl_push (fun i => cols (hiM.getD i default)) (fun _ => #[])]
    simp only [Nat.succ_mul, List.range_add, List.map_append, List.map_map]
    congr 1
    · apply Array.ext'
      simp
      intro a ha
      rw [Nat.mod_eq_of_lt ha]
    · rw [Array.append_assoc]
      congr 1
      apply Array.ext'
      simp only [Array.toList_append, List.map_const', List.length_range, List.replicate_add]

/-- The output rows' `A` and `B` columns: bit `j < bits` sums its product rows, times the constant. -/
def outA (p nh bits j : ℕ) : Array ℕ := if j < bits then Array.map (fun x => p + j * nh + x) (Array.range nh) else #[]
def outB (bits j : ℕ) : Array ℕ := if j < bits then #[KONST] else #[]

theorem outputRows_eq (p nh bits : ℕ) (s : Builder) : outputRows p nh bits s =
    { a := s.a ++ ((List.range 128).map (outA p nh bits)).toArray,
      b := s.b ++ ((List.range 128).map (outB bits)).toArray } := by
  have : (fun (r : Builder) i => if i < bits then
      ({ a := r.a.push (Array.map (fun x => p + i * nh + x) (Array.range nh)), b := r.b.push #[KONST] } : Builder)
      else { a := r.a.push #[], b := r.b.push #[] }) =
      fun (r : Builder) i => ({ a := r.a.push (outA p nh bits i), b := r.b.push (outB bits i) } : Builder) := by
    funext r i
    unfold outA outB
    split <;> rfl
  rw [outputRows, this, foldl_push]

/-! ### The theorem -/

/-- A row's linear form over the net's columns (after `KONST` is resolved). -/
def rowSum (z : ℕ → K) (cs : Array ℕ) : K := (cs.toList.map z).sum

/-- **The product rows' B side, as `Lookup.foldB` folds it** (`foldB_get`): row `prod0 + j·nh + h` (`j < bits`,
`h < nh = 2^(n − lo)`) reads low minterm `l`'s row `loTop + l` exactly when bit `j` of table word `h·2^lo + l` is set. -/
def tableB (lk : LookupTable) (z : ℕ → K) (r : ℕ) : K :=
  if lk.prod0 ≤ r ∧ r < lk.prod0 + lk.bits * 2 ^ (lk.n - lk.lo) then
    ∑ l ∈ Finset.range (2 ^ lk.lo),
      if (word lk (2 ^ lk.lo) (2 ^ lk.bits - 1) ((r - lk.prod0) % 2 ^ (lk.n - lk.lo)) l).testBit
          ((r - lk.prod0) / 2 ^ (lk.n - lk.lo)) then z (lk.loTop + l) else 0
  else 0

theorem Appends.of_eq {x y : Builder} (ra rb : List (Array ℕ)) (ha : y.a = x.a ++ ra.toArray) (hb : y.b = x.b ++ rb.toArray)
    (hl : ra.length = rb.length) : Appends x y := ⟨ra, rb, ha, hb, hl⟩

theorem getD_append_map {α : Type} (x : Array α) (f : ℕ → α) (N i : ℕ) (d : α) (hi : i < N) :
    (x ++ ((List.range N).map f).toArray).getD (x.size + i) d = f i := by
  simp [Array.getD_eq_getD_getElem?, hi]

theorem getD_append_left' {α : Type} (x y : Array α) (i : ℕ) (d : α) (hi : i < x.size) : (x ++ y).getD i d = x.getD i d := by
  simp [Array.getD_eq_getD_getElem?, Array.getElem?_append_left hi]

theorem rowSum_fix (z : ℕ → K) {k : ℕ} (hk : z k = 1) (row : Array ℕ) :
    rowSum z (row.map fun c => if (c == KONST) = true then k else c) = rowVal z row := by
  simp only [rowSum, rowVal, Array.toList_map, List.map_map]
  congr 1
  apply List.map_congr_left
  intro c _
  by_cases h : c = KONST <;> simp [colVal, h, hk]

theorem list_sum_range (g : ℕ → K) : ∀ N : ℕ, ((List.range N).map g).sum = ∑ i ∈ Finset.range N, g i
  | 0 => by simp
  | N + 1 => by rw [List.range_succ, List.map_append, List.sum_append, list_sum_range g N, Finset.sum_range_succ]; simp

theorem getD_append_replicate {α : Type} (x : Array α) (y d : α) (N i : ℕ) (hi : i < N) :
    (x ++ (List.replicate N y).toArray).getD (x.size + i) d = y := by
  simp [Array.getD_eq_getD_getElem?, hi]

theorem testBit_mask (w bits j : ℕ) (hj : j < bits) : (w &&& (2 ^ bits - 1)).testBit j = w.testBit j := by
  rw [Nat.testBit_and, Nat.testBit_two_pow_sub_one]; simp [hj]

theorem lt_mul_of {j x nh bits : ℕ} (hx : x < nh) (hj : j < bits) : j * nh + x < bits * nh :=
  Nat.lt_of_lt_of_le (Nat.add_lt_add_left hx _) (by rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right _ hj)

theorem fixKonst_getD (k : ℕ) (X : Array (Array ℕ)) (y : Array ℕ) {r : ℕ} (hr : r < X.size) :
    (fixKonst k (X.push y)).getD r #[] = (X.getD r #[]).map fun c => if (c == KONST) = true then k else c := by
  simp [fixKonst, Array.getD_eq_getD_getElem?, Array.getElem?_push, hr, Nat.ne_of_lt hr]

/-- **A lookup slot's rows compute `table[index]`.** `build` encodes rows `RA`, `RB` into its net's `A` and `B`. Over any
field of characteristic 2, let `z` satisfy every row, `z_r = (RA_r · z)(RB_r · z + tableB_r)`, where `tableB` is the
product rows' B side, the table's, as `Lookup.foldB` folds it, and let the constant be 1. Then the index bits are bits,
and bit `j < bits` of the output word is bit `j` of the table word the index bits name, for every table and index. -/
theorem build_computes {name : String} {table : ByteArray} {n lo bits : ℕ} {net : Net}
    (hb : build name table n lo bits = .ok net) (hK : net.useful ≤ KONST) :
    ∃ (RA RB : Array (Array ℕ)) (lk : LookupTable) (col : ℕ),
      Sparse.ofRows RA (2 ^ net.unitLog) = .ok net.a ∧ Sparse.ofRows RB (2 ^ net.unitLog) = .ok net.b ∧
      RA.size = net.useful ∧ RB.size = net.useful ∧ net.lookup = some lk ∧
      lk.table = table ∧ lk.n = n ∧ lk.lo = lo ∧ lk.bits = bits ∧ net.outGroups = #[⟨col, 1, #[bits]⟩] ∧
      ∀ z : ℕ → K, (∀ r < net.useful, z r = rowSum z (RA.getD r #[]) * (rowSum z (RB.getD r #[]) + tableB lk z r)) →
        z net.constPos = 1 →
        (∀ i < n, z i = 0 ∨ z i = 1) ∧
        ∀ j < bits, z (128 * col + j) = if (rd32 table (4 * bitsVal z (List.range n))).testBit j then 1 else 0 := by
  obtain ⟨hlo, hln, hb32, hn, hts, d1, d2, r3, hd1, hd2, hs1, hs2, ⟨k, hr3a, hr3b, hr3m⟩, hA, hB, huse, hconst, hout, hlk⟩ :=
    build_spec hb
  -- the stages append rows to A and B together
  have I0 := inputRows_eq n
  have hI0a : (inputRows n).a.size = 128 := by rw [I0]; simp
  have hI0b : (inputRows n).b.size = 128 := by rw [I0]; simp
  have hlo0 : List.range lo ≠ [] := by simp; omega
  have hhi0 : (List.range (n - lo)).map (lo + ·) ≠ [] := by simp; omega
  obtain ⟨A1, S1, F1, -⟩ := decode_spec (K := K) (fun _ => 0) lo (inputRows n) (List.range lo) hlo0 (by simp)
  rw [← hd1] at A1 S1 F1
  simp only [List.length_range] at S1 F1
  obtain ⟨A2, S2, -, -⟩ := decode_spec (K := K) (fun _ => 0) (n - lo) d1.1 ((List.range (n - lo)).map (lo + ·)) hhi0
    (by simp)
  rw [← hd2] at A2 S2
  have PE := productRows_eq d2.2.1 bits d2.1
  set P := productRows d2.2.1 bits d2.1 with hP
  have AP : Appends d2.1 P := Appends.of_eq _ _ (by rw [PE]) (by rw [PE]) (by simp)
  have A3 : Appends P r3 := Appends.of_eq _ _ hr3a hr3b rfl
  set F := outputRows d2.1.a.size d2.2.1.size bits r3 with hF
  have FE := outputRows_eq d2.1.a.size d2.2.1.size bits r3
  rw [← hF] at FE
  have AF : Appends r3 F := Appends.of_eq _ _ (by rw [FE]) (by rw [FE]) (by simp)
  have e0 : (inputRows n).a.size = (inputRows n).b.size := by rw [hI0a, hI0b]
  have e1 := (A1.size).2 e0
  have e2 := (A2.size).2 e1
  have eP := (AP.size).2 e2
  have e3 := (A3.size).2 eP
  have eF := (AF.size).2 e3
  have hPs : P.a.size = d2.1.a.size + bits * d2.2.1.size := by rw [PE]; simp
  have hFs : F.a.size = r3.a.size + 128 := by rw [FE]; simp
  have hr3s : P.a.size ≤ r3.a.size := (A3.size).1
  have h1s := (A1.size).1
  have h2s := (A2.size).1
  refine ⟨_, _, _, r3.a.size / 128, hA, hB, by simp [fixKonst, huse], by simp [fixKonst, huse, eF], hlk, rfl, rfl, rfl,
    rfl, hout, ?_⟩
  intro z hrow hk
  set lk : LookupTable := ⟨table, n, lo, bits, d2.1.a.size, d1.2.2⟩ with hlkd
  have hKs : F.a.size < KONST := by rw [huse] at hK; omega
  have hk' : z F.a.size = 1 := by rw [← hconst]; exact hk
  -- every row of `F`, `KONST` read as 1
  have hrowF : ∀ r < F.a.size,
      z r = rowVal z (F.a.getD r #[]) * (rowVal z (F.b.getD r #[]) + tableB lk z r) := by
    intro r hr
    have := hrow r (by rw [huse]; omega)
    rw [fixKonst_getD _ _ _ hr, fixKonst_getD _ _ _ (by rw [← eF]; exact hr), rowSum_fix z hk', rowSum_fix z hk'] at this
    exact this
  have tB0 : ∀ r, (r < d2.1.a.size ∨ P.a.size ≤ r) → tableB lk z r = 0 := by
    intro r hr
    simp only [tableB, hlkd]
    split
    · rename_i hc; rw [← hs2, ← hPs] at hc; omega
    · rfl
  -- a row of an earlier stage is that row of `F`
  have AF1 := (A2.trans (AP.trans (A3.trans AF)))
  have AF0 := A1.trans AF1
  -- the index bits are bits
  have hbits : ∀ i < n, z i = 0 ∨ z i = 1 := by
    intro i hi
    have hi128 : i < 128 := by omega
    have e := AF0.getD e0 (by rw [hI0a]; exact hi128)
    have h := hrowF i (by omega)
    rw [e.1, e.2, tB0 i (Or.inl (by omega)), I0] at h
    simp only at h
    have hK' : i ≠ KONST := by unfold KONST; omega
    simp [Array.getD_eq_getD_getElem?, hi128, hi, rowVal, colVal, hK'] at h
    exact bit_of_idem h
  refine ⟨hbits, ?_⟩
  -- the decoders' rows hold: they are rows of `F`, before the product rows
  have hbitsLo : ∀ b ∈ List.range lo, b ≠ KONST ∧ (z b = 0 ∨ z b = 1) := by
    intro b hb; simp at hb; exact ⟨by unfold KONST; omega, hbits b (by omega)⟩
  have hbitsHi : ∀ b ∈ (List.range (n - lo)).map (lo + ·), b ≠ KONST ∧ (z b = 0 ∨ z b = 1) := by
    intro b hb; simp at hb; obtain ⟨x, hx, rfl⟩ := hb; exact ⟨by unfold KONST; omega, hbits _ (by omega)⟩
  have ok1 : ∀ r, (inputRows n).a.size ≤ r → r < d1.1.a.size → RowOk z d1.1 r := by
    intro r h1 h2
    have e := AF1.getD e1 h2
    have h := hrowF r (by omega)
    rw [e.1, e.2, tB0 r (Or.inl (by omega)), add_zero] at h
    exact h
  have ok2 : ∀ r, d1.1.a.size ≤ r → r < d2.1.a.size → RowOk z d2.1 r := by
    intro r h1 h2
    have e := (AP.trans (A3.trans AF)).getD e2 h2
    have h := hrowF r (by omega)
    rw [e.1, e.2, tB0 r (Or.inl (by omega)), add_zero] at h
    exact h
  have V1 := (decode_spec z lo (inputRows n) (List.range lo) hlo0 (by simp)).2.2.2
  rw [← hd1] at V1
  have hLo := V1 e0 hbitsLo (by omega) ok1
  have V2 := (decode_spec z (n - lo) d1.1 ((List.range (n - lo)).map (lo + ·)) hhi0 (by simp)).2.2.2
  rw [← hd2] at V2
  have hHi := V2 e1 hbitsHi (by omega) ok2
  simp only [List.length_range, List.length_map] at hLo hHi
  set low := bitsVal z (List.range lo) with hlow
  set high := bitsVal z ((List.range (n - lo)).map (lo + ·)) with hhigh
  -- low minterm `l` is the single row `loTop + l`
  have hLoRow : ∀ l < 2 ^ lo, z (d1.2.2 + l) = if low = l then 1 else 0 := by
    intro l hl
    have := hLo l hl
    rw [(F1 hlo).2 l hl, formVal_row z (by have := (F1 hlo).1; unfold KONST at hKs ⊢; omega)] at this
    exact this
  have hlowlt : low < 2 ^ lo := by have := bitsVal_lt z (List.range lo); simpa using this
  have hhighlt : high < 2 ^ (n - lo) := by have := bitsVal_lt z ((List.range (n - lo)).map (lo + ·)); simpa using this
  -- the sum over the low minterms picks the low index
  have hsumLo : ∀ (Q : ℕ → Prop) [DecidablePred Q],
      (∑ l ∈ Finset.range (2 ^ lo), if Q l then z (d1.2.2 + l) else 0) = if Q low then 1 else 0 := by
    intro Q _
    rw [Finset.sum_eq_single low]
    · rw [hLoRow low hlowlt]; simp
    · intro l hl hne; rw [hLoRow l (Finset.mem_range.1 hl)]; simp [Ne.symm hne]
    · intro h; exact absurd (Finset.mem_range.2 hlowlt) h
  -- product row `prod0 + t`: the high minterm `t % nh` times bit `t / nh` of its table word at the low index
  have hProd : ∀ t < bits * d2.2.1.size, z (d2.1.a.size + t) =
      (if high = t % d2.2.1.size then 1 else 0) *
        (if (word lk (2 ^ lo) (2 ^ bits - 1) (t % d2.2.1.size) low).testBit (t / d2.2.1.size) then 1 else 0) := by
    intro t ht
    have hr : d2.1.a.size + t < P.a.size := by omega
    have e := (A3.trans AF).getD eP hr
    have h := hrowF (d2.1.a.size + t) (by omega)
    rw [e.1, e.2, PE] at h
    simp only at h
    rw [getD_append_map _ _ _ _ _ ht, e2, getD_append_replicate _ _ _ _ _ ht] at h
    have hnh0 : 0 < d2.2.1.size := by rw [hs2]; exact Nat.two_pow_pos _
    have htb : tableB lk z (d2.1.a.size + t) =
        if (word lk (2 ^ lo) (2 ^ bits - 1) (t % d2.2.1.size) low).testBit (t / d2.2.1.size) then 1 else 0 := by
      simp only [tableB, hlkd]
      rw [ite_eq_left_of_eq_true _ _ (eq_true (by rw [← hs2]; omega)), Nat.add_sub_cancel_left, ← hs2]
      exact hsumLo _
    rw [← e2, htb] at h
    rw [h, ← formVal, hHi (t % d2.2.1.size) (by rw [← hs2]; exact Nat.mod_lt _ hnh0)]
    simp [rowVal]
  -- output bit `j`: its product rows summed, the one at the high index left
  intro j hj
  have hj128 : j < 128 := by omega
  have hr : r3.a.size + j < F.a.size := by omega
  have h := hrowF (r3.a.size + j) hr
  rw [FE] at h
  simp only at h
  rw [getD_append_map _ _ _ _ _ hj128, e3, getD_append_map _ _ _ _ _ hj128, ← e3,
    tB0 _ (Or.inr (by omega)), add_zero] at h
  simp only [outA, outB, hj, ↓reduceIte] at h
  have hKr : rowVal z #[KONST] = 1 := by simp [rowVal, colVal]
  rw [hKr, mul_one] at h
  have hsum : rowVal z (Array.map (fun x => d2.1.a.size + j * d2.2.1.size + x) (Array.range d2.2.1.size)) =
      ∑ x ∈ Finset.range d2.2.1.size, z (d2.1.a.size + (j * d2.2.1.size + x)) := by
    simp only [rowVal, Array.toList_map, Array.toList_range, List.map_map]
    rw [list_sum_range]
    apply Finset.sum_congr rfl
    intro x hx
    simp only [Function.comp, add_assoc]
    rw [colVal_of_ne]
    have := Finset.mem_range.1 hx
    have : j * d2.2.1.size + x < bits * d2.2.1.size := lt_mul_of this hj
    unfold KONST at hKs ⊢; omega
  have hnh0 : 0 < d2.2.1.size := by rw [hs2]; exact Nat.two_pow_pos _
  rw [hsum] at h
  have hterm : ∀ x ∈ Finset.range d2.2.1.size, z (d2.1.a.size + (j * d2.2.1.size + x)) =
      (if high = x then 1 else 0) *
        (if (word lk (2 ^ lo) (2 ^ bits - 1) x low).testBit j then 1 else 0) := by
    intro x hx
    have hx' := Finset.mem_range.1 hx
    rw [hProd _ (lt_mul_of hx' hj)]
    rw [show (j * d2.2.1.size + x) % d2.2.1.size = x by rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hx'],
      show (j * d2.2.1.size + x) / d2.2.1.size = j by
        rw [Nat.add_comm, Nat.add_mul_div_right _ _ hnh0, Nat.div_eq_of_lt hx', zero_add]]
  rw [Finset.sum_congr rfl hterm, Finset.sum_eq_single high] at h
  · rw [show 128 * (r3.a.size / 128) + j = r3.a.size + j by rw [Nat.mul_div_cancel' (Nat.dvd_of_mod_eq_zero hr3m)], h]
    simp only [↓reduceIte, one_mul, word, hlkd]
    rw [testBit_mask _ _ _ hj]
    have hidx : bitsVal z (List.range n) = high * 2 ^ lo + low := by
      rw [show n = lo + (n - lo) by omega, List.range_add, bitsVal_append]
      simp only [List.length_range]
      rw [← hlow, ← hhigh]
      ring
    rw [hidx]
  · intro x _ hne; simp [Ne.symm hne]
  · intro hh; exact absurd (Finset.mem_range.2 (by rw [hs2]; exact hhighlt)) hh

end

end FlockLevel3
