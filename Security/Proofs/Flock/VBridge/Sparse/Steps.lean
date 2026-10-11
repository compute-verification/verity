import Proofs.Flock.VBridge.Sparse.Values

/-!
# The sparse-phase unit's steps

`Flock.RecSparse.steps` is sound against values computed from the builder's own residuals (`sound_steps`): whenever its
rows hold, if the message ports' first elements carry `w`, the `v` port carries `v` and the `tau` port the slots' types
`τ` (`StepsIn`), its output's 128-bit elements carry `stepVals` and, with `T > 1`, its tail is each type with its low
`T_LOG` bits cleared. The residuals are `RecResiduals.residualFormsA`'s, through `RecResiduals.residualForms_eq` and the
bridge's `residualForms_eq` to `sound_residualForms` (`rval`); the two claim chains are loops over a range
(`sound_chain`); Φ's product, the type-selected sum and H4 are the gadgets' (`Gadgets.lean`, `Values.lean`).
-/

namespace FlockVBridge.Sparse

open Flock.HmNets hiding and
open Flock (F128)
open FlockSoundness.Discharge.Hm
open Flock.RecSparse (messagePorts vParts offset elem)

/-! ## Forms and bits -/

theorem holds_iff {z : Var → Bool} {f : List Lin} {x : F128} :
    Holds z f x ↔ f.length = 128 ∧ ∀ k < 128, bit z f k = bitOf x k := by
  refine ⟨fun h => ⟨h.1, fun k _ => ?_⟩, fun h => ⟨h.1, (polyOf_eq_poly_iff z f x).2 fun k => ?_⟩⟩
  · rw [bit_eq_getD]
    exact (polyOf_eq_poly_iff z f x).1 h.2 k
  · by_cases hk : k < 128
    · rw [← bit_eq_getD]
      exact h.2 k hk
    · rw [List.getD_eq_default _ _ (by omega), ev_zero, bitOf_ge x (by omega)]

theorem bit_append_left (z : Var → Bool) {a : List Lin} (b : List Lin) {i : ℕ} (h : i < a.length) :
    bit z (a ++ b) i = bit z a i := by
  simp only [bit, List.getElem?_append_left h]

theorem bit_append_right (z : Var → Bool) (a b : List Lin) (i : ℕ) : bit z (a ++ b) (a.length + i) = bit z b i := by
  simp only [bit, List.getElem?_append_right (Nat.le_add_right _ _), Nat.add_sub_cancel_left]

theorem bit_append_len (z : Var → Bool) (a b : List Lin) {n : ℕ} (h : a.length = n) (i : ℕ) :
    bit z (a ++ b) (n + i) = bit z b i := by
  subst h; exact bit_append_right z a b i

/-- Element `i` of `bits` carries `x` when its 128 bits are `x`'s. -/
theorem holds_elem {z : Var → Bool} {bits : List Lin} {i : ℕ} {x : F128} (hl : 128 * i + 128 ≤ bits.length)
    (h : ∀ k < 128, bit z bits (128 * i + k) = bitOf x k) : Holds z (elem bits.toArray i) x := by
  refine holds_iff.2 ⟨by simp [elem]; omega, fun k hk => ?_⟩
  rw [← h k hk]
  simp [bit, elem, List.getElem?_drop, hk]

theorem bit_flatten {z : Var → Bool} {fs : List (List Lin)} {xs : List F128} (h : List.Forall₂ (Holds z) fs xs) :
    ∀ (rest : List Lin) r (hr : r < xs.length), ∀ k < 128, bit z (fs.flatten ++ rest) (128 * r + k) = bitOf xs[r] k := by
  induction h with
  | nil => exact fun _ r hr => absurd hr (Nat.not_lt_zero _)
  | @cons f x fs xs hf _ ih =>
    intro rest r hr k hk
    rw [List.flatten_cons, List.append_assoc]
    cases r with
    | zero =>
      rw [Nat.mul_zero, Nat.zero_add, bit_append_left z _ (by rw [hf.1]; exact hk)]
      exact (holds_iff.1 hf).2 k hk
    | succ r =>
      rw [show 128 * (r + 1) + k = f.length + (128 * r + k) by rw [hf.1]; omega, bit_append_right]
      exact ih rest r (by simpa using hr) k hk

theorem length_flatten_holds {z : Var → Bool} {fs : List (List Lin)} {xs : List F128}
    (h : List.Forall₂ (Holds z) fs xs) : fs.flatten.length = 128 * xs.length := by
  induction h with
  | nil => rfl
  | cons hf _ ih => rw [List.flatten_cons, List.length_append, hf.1, ih, List.length_cons]; omega

/-- The output's elements: whatever follows them. -/
theorem post_of {z : Var → Bool} {L : List (List Lin)} {xs : List F128} (hL : List.Forall₂ (Holds z) L xs)
    (tail : List Lin) (n : ℕ) :
    ∀ r (hr : r < xs.length), ∀ k < 128, bit z (L.flatten ++ tail ++ List.replicate n Lin.zero) (128 * r + k) =
      bitOf xs[r] k := by
  intro r hr k hk
  rw [List.append_assoc]
  exact bit_flatten hL _ r hr k hk

/-- The output's tail, past four elements. -/
theorem bit_out_tail {z : Var → Bool} {L : List (List Lin)} {xs : List F128} (hL : List.Forall₂ (Holds z) L xs)
    (h4 : xs.length = 4) (tail : List Lin) (n : ℕ) {i : ℕ} (hi : i < tail.length) :
    bit z (L.flatten ++ tail ++ List.replicate n Lin.zero) (512 + i) = bit z tail i := by
  have hl := length_flatten_holds hL
  rw [h4] at hl
  rw [bit_append_left z _ (by rw [List.length_append, hl]; omega), show 512 + i = L.flatten.length + i by omega,
    bit_append_right]

/-- The output's length: the zeros fill it to `n` bits. -/
theorem length_out {z : Var → Bool} {L : List (List Lin)} {xs : List F128} (hL : List.Forall₂ (Holds z) L xs)
    (tail : List Lin) (n : ℕ) (h : 128 * xs.length + tail.length ≤ n) :
    (L.flatten ++ tail ++ List.replicate (n - (L.flatten ++ tail).length) Lin.zero).length = n := by
  rw [List.length_append, List.length_replicate, List.length_append, length_flatten_holds hL]
  omega

theorem outWords_ge (sh : Flock.RecSparse.Shape) : 128 * (Flock.RecSparse.residualNames sh).length +
    16 * (if sh.stacked then sh.S else 0) ≤ 16 * Flock.RecSparse.outWords sh := by
  unfold Flock.RecSparse.outWords Flock.RecSparse.F_WORDS Flock.RecSparse.ROW_BLOCK
  omega

/-! ## Lists -/

theorem forall₂_range {α β : Type} {R : α → β → Prop} (f : ℕ → α) (g : ℕ → β) :
    ∀ n, (∀ t < n, R (f t) (g t)) → List.Forall₂ R ((List.range n).map f) ((List.range n).map g)
  | 0, _ => by simp
  | n + 1, h => by
      rw [List.range_succ, List.map_append, List.map_append]
      exact List.rel_append (forall₂_range f g n fun t ht => h t (by omega)) (.cons (h n (by omega)) .nil)

theorem forall₂_flatMap_range {α β : Type} {R : α → β → Prop} (f : ℕ → List α) (g : ℕ → List β) :
    ∀ n, (∀ t < n, List.Forall₂ R (f t) (g t)) → List.Forall₂ R ((List.range n).flatMap f) ((List.range n).flatMap g)
  | 0, _ => by simp
  | n + 1, h => by
      rw [List.range_succ, List.flatMap_append, List.flatMap_append, List.flatMap_cons, List.flatMap_cons,
        List.flatMap_nil, List.flatMap_nil, List.append_nil, List.append_nil]
      exact List.rel_append (forall₂_flatMap_range f g n fun t ht => h t (by omega)) (h n (by omega))

theorem forall₂_getD {α β : Type} {R : α → β → Prop} {d : α} {e : β} :
    ∀ {l : List α} {m : List β}, List.Forall₂ R l m → ∀ {t}, t < m.length → R (l.getD t d) (m.getD t e)
  | _, _, .nil, _, h => absurd h (Nat.not_lt_zero _)
  | _, _, .cons hab _, 0, _ => hab
  | _, _, .cons _ hs, _ + 1, h => forall₂_getD hs (Nat.lt_of_succ_lt_succ h)

theorem forall₂_zip {α β γ δ : Type} {R : α → β → Prop} {Q : γ → δ → Prop} :
    ∀ {l : List α} {m : List β} {l' : List γ} {m' : List δ}, List.Forall₂ R l m → List.Forall₂ Q l' m' →
      List.Forall₂ (fun p q => R p.1 q.1 ∧ Q p.2 q.2) (l.zip l') (m.zip m')
  | _, _, _, _, .nil, _ => by simp
  | _, _, _, _, .cons _ _, .nil => by simp
  | _, _, _, _, .cons h hs, .cons h' hs' => .cons ⟨h, h'⟩ (forall₂_zip hs hs')

theorem forall₂_map_getD {z : Var → Bool} {fs : List (List (List Lin))} {xs : List (List F128)}
    (h : List.Forall₂ (List.Forall₂ (Holds z)) fs xs) {t : ℕ} (ht : ∀ x ∈ xs, t < x.length) :
    List.Forall₂ (Holds z) (fs.map (·.getD t [])) (xs.map (·.getD t 0)) := by
  induction h with
  | nil => exact .nil
  | cons hf _ ih =>
    exact .cons (forall₂_getD hf (ht _ List.mem_cons_self)) (ih fun x hx => ht x (List.mem_cons_of_mem _ hx))

theorem foldl_add (l : List ℕ) (a : ℕ) : l.foldl (· + ·) a = a + l.foldl (· + ·) 0 := by
  induction l generalizing a with
  | nil => simp
  | cons x l ih => rw [List.foldl_cons, List.foldl_cons, ih, ih (0 + x)]; omega

theorem foldl_take_lt : ∀ (l : List ℕ) (j : ℕ), j < l.length → (∀ x ∈ l, 0 < x) →
    (l.take j).foldl (· + ·) 0 < l.foldl (· + ·) 0
  | [], _, h, _ => absurd h (Nat.not_lt_zero _)
  | x :: l, 0, _, hp => by
      have := hp x List.mem_cons_self
      rw [List.take_zero, List.foldl_nil, List.foldl_cons, foldl_add]
      omega
  | x :: l, j + 1, h, hp => by
      have ih := foldl_take_lt l j (by simpa using h) fun y hy => hp y (List.mem_cons_of_mem _ hy)
      rw [List.take_succ_cons, List.foldl_cons, List.foldl_cons, foldl_add, foldl_add l]
      omega

/-! ## The loops -/

theorem sound_forIn_range' {σ : Type} (f : ℕ → σ → M (ForInStep σ)) (I : (Var → Bool) → ℕ → σ → Prop) :
    ∀ (n a : ℕ), (∀ j s, a ≤ j → j < a + n →
      Sound (f j s) fun z y => ∃ s', y = .yield s' ∧ (I z j s → I z (j + 1) s')) →
    ∀ s, Sound (forIn (List.range' a n) s f) fun z r => I z a s → I z (a + n) r
  | 0, a, _, s => by
      show Sound (forIn ([] : List ℕ) s f) _
      rw [List.forIn_nil]
      exact Sound.pure' fun z h => h
  | n + 1, a, hf, s => by
      rw [List.range'_succ, List.forIn_cons]
      refine Sound.bind' (hf a s le_rfl (by omega)) fun y => ?_
      cases y with
      | done b => exact Sound.pure' fun z hy => by obtain ⟨s', h, -⟩ := hy; cases h
      | yield b =>
        refine Sound.mono (sound_forIn_range' f I n (a + 1) (fun j s h1 h2 => hf j s (by omega) (by omega)) b) ?_
        intro z r ih hy hs
        obtain ⟨s', h, hI⟩ := hy
        cases h
        rw [show a + (n + 1) = a + 1 + n by omega]
        exact ih (hI hs)

/-- **A claim chain is its fold**: `c ← A j + c · B j` over `[0:n]`, each product by `mulF`. -/
theorem sound_chain (A B : ℕ → List Lin) (n : ℕ) (init : List Lin) (f : ℕ → List Lin → M (ForInStep (List Lin)))
    (hf : ∀ j r, Sound (f j r) fun z y => ∃ s', y = .yield s' ∧
      ∀ x bj aj, Holds z r x → Holds z (B j) bj → Holds z (A j) aj → Holds z s' (aj + x * bj)) :
    Sound (forIn [0:n] init f) fun z out => ∀ (a b : ℕ → F128) (c0 : F128), Holds z init c0 →
      (∀ i < n, Holds z (A i) (a i)) → (∀ i < n, Holds z (B i) (b i)) →
      Holds z out ((List.range n).foldl (fun c i => a i + c * b i) c0) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  have e : List.range' [0:n].start [0:n].size [0:n].step = List.range' 0 n := by simp [Std.Legacy.Range.size]
  rw [e]
  refine Sound.mono (sound_forIn_range' f (fun z j c => ∀ (a b : ℕ → F128) (c0 : F128), Holds z init c0 →
      (∀ i < n, Holds z (A i) (a i)) → (∀ i < n, Holds z (B i) (b i)) →
      Holds z c ((List.range j).foldl (fun c i => a i + c * b i) c0)) n 0 (fun j s _ hj => ?_) init) ?_
  · refine Sound.mono (hf j s) fun z y ⟨s', hy, hstep⟩ => ⟨s', hy, fun hI a b c0 h0 hA hB => ?_⟩
    rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
    exact hstep _ _ _ (hI a b c0 h0 hA hB) (hB j (by omega)) (hA j (by omega))
  · intro z r h a b c0 h0 hA hB
    simp only [Nat.zero_add] at h
    exact h (fun a b c0 h0 _ _ => by simpa using h0) a b c0 h0 hA hB

/-! ## The tail -/

theorem testBit_clear (τ t b : ℕ) : (τ - τ % 2 ^ t).testBit b = (decide (t ≤ b) && τ.testBit b) := by
  rw [Nat.sub_eq_of_eq_add (Nat.div_add_mod τ (2 ^ t)).symm, Nat.testBit_two_pow_mul, Nat.testBit_div_two_pow]
  by_cases hb : t ≤ b
  · simp [hb, Nat.sub_add_cancel hb]
  · simp [hb]

theorem chunk_bits (z : Var → Bool) (tau : List Lin) (t s : ℕ) (ht : t ≤ 16) (hl : 16 * s + 16 ≤ tau.length) :
    (List.replicate t Lin.zero ++ (tau.toArray.extract (16 * s + t) (16 * s + 16)).toList).length = 16 ∧
    ∀ b < 16, bit z (List.replicate t Lin.zero ++ (tau.toArray.extract (16 * s + t) (16 * s + 16)).toList) b =
      (decide (t ≤ b) && bit z tau (16 * s + b)) := by
  refine ⟨by simp; omega, fun b hb => ?_⟩
  by_cases hb' : b < t
  · rw [bit_append_left z _ (by simpa using hb')]
    simp [bit, hb', Nat.not_le.2 hb']
  · obtain ⟨c, rfl⟩ : ∃ c, b = t + c := ⟨b - t, by omega⟩
    rw [bit_append_len z _ _ (by simp : (List.replicate t Lin.zero).length = t) c]
    have hc' : c < 16 * s + 16 - (16 * s + t) := by omega
    simp [bit, List.getElem?_drop, hc', Nat.add_assoc]

/-- **The tail is each slot's type with its low `t` bits cleared**: a slot a 16-bit word, `t` zeros then the type's
other bits. -/
theorem tail_bits (z : Var → Bool) (tau : List Lin) (t : ℕ) (ht : t ≤ 16) : ∀ S, 16 * S ≤ tau.length →
    ((List.range S).flatMap fun s => List.replicate t Lin.zero ++
      (tau.toArray.extract (16 * s + t) (16 * s + 16)).toList).length = 16 * S ∧
    ∀ s < S, ∀ b < 16, bit z ((List.range S).flatMap fun s => List.replicate t Lin.zero ++
      (tau.toArray.extract (16 * s + t) (16 * s + 16)).toList) (16 * s + b) = (decide (t ≤ b) && bit z tau (16 * s + b))
  | 0, _ => ⟨by simp, fun s hs => absurd hs (Nat.not_lt_zero _)⟩
  | S + 1, hl => by
      obtain ⟨ihl, ihb⟩ := tail_bits z tau t ht S (by omega)
      obtain ⟨hcl, hcb⟩ := chunk_bits z tau t S ht (by omega)
      rw [List.range_succ, List.flatMap_append, List.flatMap_cons, List.flatMap_nil, List.append_nil]
      refine ⟨by rw [List.length_append, ihl, hcl]; omega, fun s hs b hb => ?_⟩
      rcases Nat.lt_succ_iff_lt_or_eq.1 hs with hs | rfl
      · rw [bit_append_left z _ (by rw [ihl]; omega)]
        exact ihb s hs b hb
      · rw [bit_append_len z _ _ ihl]
        exact hcb b hb

/-! ## The demultiplexer's values -/

theorem demuxVal_range (tb : ℕ → ℕ → Bool) (m S : ℕ) (X : ℕ → F128) :
    demuxVal tb (List.range m).reverse ((List.range S).map fun s => [X s]) =
      (List.range S).map fun s => (List.range (2 ^ m)).map fun t => if low (tb s) m = t then X s else 0 := by
  rw [demuxVal_eq]
  refine List.ext_getElem (by simp) fun i h1 h2 => ?_
  simp [List.getElem_zipIdx, splits_range]

theorem demux_column (tb : ℕ → ℕ → Bool) (τ : ℕ → ℕ) (m S : ℕ) (X : ℕ → F128)
    (h : ∀ s < S, ∀ j < m, tb s j = (τ s).testBit j) {t : ℕ} (ht : t < 2 ^ m) :
    ((List.range S).map fun s => (List.range (2 ^ m)).map fun t => if low (tb s) m = t then X s else 0).map
      (·.getD t 0) = (List.range S).map fun s => if τ s % 2 ^ m = t then X s else 0 := by
  rw [List.map_map]
  refine List.map_congr_left fun s hs => ?_
  rw [List.mem_range] at hs
  simp only [Function.comp_apply, low_congr (h s hs), low_testBit]
  rw [List.getD_eq_getElem _ _ (by simpa using ht), List.getElem_map, List.getElem_range]

/-! ## The ports -/

section Steps

variable (sh : Flock.RecSparse.Shape) (P : String → List Lin)

theorem structureOf_length : (Flock.RecSparse.structureOf sh).2.length = (Flock.RecSparse.structureOf sh).1.res.length := by
  unfold Flock.RecSparse.structureOf
  generalize (Flock.RecSparse.resList sh).run #[] = p
  obtain ⟨rs, ops⟩ := p
  simp

theorem nv_eq : Flock.RecSparse.nv sh = 79 + sh.degrees.foldl (· + ·) 0 + 2 * sh.ell + sh.PP +
    (if sh.stacked then sh.S + sh.T else 0) := by
  cases h : sh.stacked <;>
    simp [Flock.RecSparse.nv, vParts, Flock.RecSparse.SKIP, Flock.RecSparse.PRE_ROUNDS, h] <;> omega

theorem vp_offsets : offset (vParts sh) "pre" = 64 ∧ offset (vParts sh) "sc" = 77 ∧
    offset (vParts sh) "ur" = 78 + sh.degrees.foldl (· + ·) 0 + 2 * sh.ell ∧
    offset (vParts sh) "eqs" = 79 + sh.degrees.foldl (· + ·) 0 + 2 * sh.ell + sh.PP := by
  cases h : sh.stacked <;>
    simp [vParts, offset, Flock.RecSparse.SKIP, Flock.RecSparse.PRE_ROUNDS, h] <;> omega

theorem degrees_pos : ∀ x ∈ sh.degrees, 0 < x := by
  intro x hx
  simp only [Flock.RecSparse.Shape.degrees, List.mem_append, List.mem_replicate, Flock.RecSparse.Shape.d] at hx
  omega

/-- The `v` indices `steps` reads are `v`'s. -/
theorem v_bounds :
    (∀ j < 6, offset (vParts sh) "pre" + 2 * j < Flock.RecSparse.nv sh) ∧
    (∀ j < sh.degrees.length, offset (vParts sh) "sc" + (sh.degrees.take j).foldl (· + ·) 0 < Flock.RecSparse.nv sh) ∧
    offset (vParts sh) "ur" + 0 < Flock.RecSparse.nv sh ∧
    (sh.stacked = true → ∀ s < sh.S, offset (vParts sh) "eqs" + s < Flock.RecSparse.nv sh) := by
  obtain ⟨h1, h2, h3, h4⟩ := vp_offsets sh
  have hn := nv_eq sh
  refine ⟨fun j hj => by omega, fun j hj => ?_, by omega, fun hs s hss => ?_⟩
  · have := foldl_take_lt _ j hj (degrees_pos sh)
    omega
  · simp only [hs, ↓reduceIte] at hn
    omega

/-- The residuals `steps` reads by name. -/
def Used (nm : String) : Prop :=
  nm = "h1" ∨ nm = "wv" ∨ nm = "f0" ∨ nm = "bp" ∨ (sh.stacked = true ∧ nm = "claim") ∨ (∃ j < 6, nm = s!"pre{j}") ∨
    (∃ j < sh.degrees.length, nm = s!"sc{j}") ∨ (∃ j < sh.ell, nm = s!"fx{j}" ∨ nm = s!"fy{j}")

/-- The value of the builder's residual `nm` at `w`, `v` (`resVal`). -/
def rval (w v : ℕ → F128) (nm : String) : F128 :=
  resVal (gfsOf (Flock.RecSparse.structureOf sh).1) (termVal w v)
    ((gfsOf (Flock.RecSparse.structureOf sh).1).res.getD ((Flock.RecSparse.structureOf sh).2.idxOf nm) ⟨[], []⟩)

/-- The values `steps` computes from residual values `R`, `w`, `v` and the slots' types `τ`: H2's and H3's claims, then
with `T > 1` H4's (the plane batch against the type-selected sum) and the registered claim's residual, else `bp`. -/
def stepVals (R : String → F128) (w v : ℕ → F128) (τ : ℕ → ℕ) : List F128 :=
  let mp := messagePorts sh
  let vp := vParts sh
  let c₂ := (List.range 6).foldl (fun c j => R s!"pre{j}" + c * v (offset vp "pre" + 2 * j)) (R "h1")
  let c₄ := (List.range sh.degrees.length).foldl
    (fun c j => R s!"sc{j}" + c * v (offset vp "sc" + (sh.degrees.take j).foldl (· + ·) 0)) (w (offset mp "v2"))
  let fac := [R "f0"] ++ (List.range sh.ell).flatMap (fun j => [R s!"fx{j}", R s!"fy{j}"]) ++ [v (offset vp "ur" + 0)]
  let h2 := c₂ + R "wv"
  let h3 := c₄ + treeVal fac.length fac
  if sh.stacked then
    let wt := fun t =>
      sumVal ((List.range sh.S).map fun s => if τ s % 2 ^ sh.tLog = t then v (offset vp "eqs" + s) else 0)
    let sel := (((List.range sh.T).map wt).zip ((List.range sh.T).map fun t => w (offset mp "vt" + t))).map
      fun e => e.1 * e.2
    [h2, h3, sumVal (R "bp" :: sel), R "claim"]
  else [h2, h3, R "bp"]

/-- What `steps` reads: the message ports' first elements carry `w` (in `steps`' order), `v`'s port `v`, `v2`'s and
`vt`'s elements `w` at their offsets, and with `T > 1` each slot's 16 bits of `tau` its type. -/
structure StepsIn (z : Var → Bool) (w v : ℕ → F128) (τ : ℕ → ℕ) : Prop where
  hw : ∀ q, bit z ((messagePorts sh).flatMap fun p => (P p.1).take (128 * p.2)) q = wbit w q
  hv : ∀ q, bit z ((P "v").take (128 * Flock.RecSparse.nv sh)) q = wbit v q
  hv2 : Holds z (elem (P "v2").toArray 0) (w (offset (messagePorts sh) "v2"))
  hvt : sh.stacked = true → ∀ t < sh.T, Holds z (elem (P "vt").toArray t) (w (offset (messagePorts sh) "vt" + t))
  htau : sh.stacked = true → ∀ s < sh.S, ∀ b < 16, bit z (P "tau") (16 * s + b) = (τ s).testBit b

/-- **`steps` computes `stepVals`**: whenever its rows hold, if its ports carry `w`, `v` and `τ` (`StepsIn`), its output's
128-bit elements carry `stepVals` at the builder's residuals (`rval`), and with `T > 1` its tail is each slot's type with
its low `T_LOG` bits cleared. At a `v` port of `nv` elements, a `tau` port of a word a slot, `T_LOG ≤ 16`, and residual
names the builder's structure has. -/
theorem sound_steps (hPv : 128 * Flock.RecSparse.nv sh ≤ (P "v").length)
    (hτ : sh.stacked = true → 16 * sh.S ≤ (P "tau").length) (ht16 : sh.tLog ≤ 16)
    (hn : ∀ nm, Used sh nm → nm ∈ (Flock.RecSparse.structureOf sh).2) :
    Sound (Flock.RecSparse.steps sh P) fun z out => ∀ w v τ, StepsIn sh P z w v τ →
      out.length = 16 * Flock.RecSparse.outWords sh ∧
      (∀ r (hr : r < (stepVals sh (rval sh w v) w v τ).length), ∀ k < 128,
        bit z out (128 * r + k) = bitOf (stepVals sh (rval sh w v) w v τ)[r] k) ∧
      (sh.stacked = true → ∀ s < sh.S, ∀ b < 16,
        bit z out (512 + (16 * s + b)) = (τ s - τ s % 2 ^ sh.tLog).testBit b) := by
  have hlen := structureOf_length sh
  obtain ⟨bPre, bSc, bUr, bEqs⟩ := v_bounds sh
  unfold Flock.RecSparse.steps
  rcases hs : Flock.RecSparse.structureOf sh with ⟨S, names⟩
  simp only [hs] at hlen hn
  dsimp only
  rw [Flock.RecResiduals.residualForms_eq, FlockVBridge.residualForms_eq]
  refine Sound.bind' (sound_residualForms _ _ _) fun out => ?_
  refine Sound.bind' (sound_chain (fun j => elem out.toArray (names.idxOf s!"pre{j}"))
    (fun j => elem ((P "v").take (128 * Flock.RecSparse.nv sh)).toArray (offset (vParts sh) "pre" + 2 * j)) _ _ _
    fun j r => Sound.bind' (sound_mulF _ _) fun p => Sound.pure' fun z hp =>
      ⟨_, rfl, fun x bj aj hx hb ha => holds_xorL ha (hp x bj hx hb)⟩) fun c2 => ?_
  refine Sound.bind' (sound_chain (fun j => elem out.toArray (names.idxOf s!"sc{j}"))
    (fun j => elem ((P "v").take (128 * Flock.RecSparse.nv sh)).toArray
      (offset (vParts sh) "sc" + (sh.degrees.take j).foldl (· + ·) 0)) _ _ _
    fun j r => Sound.bind' (sound_mulF _ _) fun p => Sound.pure' fun z hp =>
      ⟨_, rfl, fun x bj aj hx hb ha => holds_xorL ha (hp x bj hx hb)⟩) fun c4 => ?_
  refine Sound.bind' (sound_tree _ _) fun phi => ?_
  -- The facts both branches read: the residuals, `v`'s elements, H2's and H3's claims.
  have core : ∀ z w v τ, StepsIn sh P z w v τ →
      (out.length = 128 * (gfsOf S).res.length ∧ ∀ w v : ℕ → F128,
        (∀ q, bit z ((messagePorts sh).flatMap fun p => (P p.1).take (128 * p.2)) q = wbit w q) →
        (∀ q, bit z ((P "v").take (128 * Flock.RecSparse.nv sh)) q = wbit v q) →
        ∀ r (hr : r < (gfsOf S).res.length), ∀ k < 128,
          bit z out (128 * r + k) = bitOf (resVal (gfsOf S) (termVal w v) (gfsOf S).res[r]) k) →
      (∀ nm, Used sh nm → Holds z (elem out.toArray (names.idxOf nm)) (rval sh w v nm)) ∧
      ∀ i < Flock.RecSparse.nv sh, Holds z (elem ((P "v").take (128 * Flock.RecSparse.nv sh)).toArray i) (v i) := by
    intro z w v τ hin hres
    obtain ⟨hlenO, hval⟩ := hres
    have hgl : (gfsOf S).res.length = names.length := by simp [gfsOf, hlen]
    refine ⟨fun nm hu => ?_, fun i hi => ?_⟩
    · have hi : names.idxOf nm < names.length := List.idxOf_lt_length_of_mem (hn nm hu)
      refine holds_elem (by rw [hlenO, hgl]; omega) fun k hk => ?_
      rw [hval w v hin.hw hin.hv _ (by rw [hgl]; exact hi) k hk]
      simp only [rval, hs, List.getD_eq_getElem _ _ (show names.idxOf nm < (gfsOf S).res.length by rw [hgl]; exact hi)]
    · refine holds_elem (by rw [List.length_take]; omega) fun k hk => ?_
      rw [hin.hv, wbit_elem v i hk]
  split
  · rename_i hst
    simp only [pure_bind]
    refine Sound.bind' (sound_demux _ _ _) fun lists => ?_
    refine Sound.bind' (sound_mapM_rel _ (fun z (p : List Lin × List Lin) (e : F128 × F128) =>
      Holds z p.1 e.1 ∧ Holds z p.2 e.2) (fun _ (e : F128 × F128) => e.1 * e.2) (fun z r y => Holds z r y)
      (fun x => ?_) _) fun sel => ?_
    · obtain ⟨a, b⟩ := x
      exact Sound.mono (sound_mulF a b) fun z c h e he => h _ _ he.1 he.2
    refine Sound.pure' fun z hsel hlists hphi hc4 hc2 hres w v τ hin => ?_
    obtain ⟨HR, HV⟩ := core z w v τ hin hres
    have H2 := holds_xorL (hc2 (fun j => rval sh w v s!"pre{j}") (fun j => v (offset (vParts sh) "pre" + 2 * j)) _
      (HR _ (Or.inl rfl)) (fun i hi => HR _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl ⟨i, hi, rfl⟩)))))))
      (fun i hi => HV _ (bPre i hi))) (HR "wv" (Or.inr (Or.inl rfl)))
    have hF := List.rel_append (List.rel_append (.cons (HR "f0" (Or.inr (Or.inr (Or.inl rfl)))) .nil)
      (forall₂_flatMap_range _ (fun j => [rval sh w v s!"fx{j}", rval sh w v s!"fy{j}"]) sh.ell fun j hj =>
        .cons (HR _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr ⟨j, hj, Or.inl rfl⟩))))))))
          (.cons (HR _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr ⟨j, hj, Or.inr rfl⟩))))))))
            .nil)))
      (.cons (HV _ bUr) .nil)
    have Hphi := hphi _ (by simp) hF
    rw [hF.length_eq] at Hphi
    have H3 := holds_xorL (hc4 (fun j => rval sh w v s!"sc{j}")
      (fun j => v (offset (vParts sh) "sc" + (sh.degrees.take j).foldl (· + ·) 0)) _ hin.hv2
      (fun i hi => HR _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl ⟨i, hi, rfl⟩))))))))
      (fun i hi => HV _ (bSc i hi))) Hphi
    have Hτ := hin.htau hst
    have hS0 : 0 < sh.S := Nat.two_pow_pos _
    have hL := hlists ((List.range sh.S).map fun s => [v (offset (vParts sh) "eqs" + s)])
      (forall₂_range _ _ _ fun s hs => .cons (HV _ (bEqs hst s hs)) .nil)
    rw [demuxVal_range] at hL
    have hne : lists ≠ [] := by
      rintro rfl
      have := hL.length_eq
      simp at this
      omega
    have htb : ∀ s < sh.S, ∀ j < sh.tLog,
        (fun s j => ev z ((P "tau").toArray.getD (16 * s + j) Lin.zero)) s j = (τ s).testBit j :=
      fun s hs j hj => by
        rw [← Hτ s hs j (by omega)]
        simp [bit]
    have HWT : ∀ t < sh.T, Holds z (Flock.RecSparse.xorAllL (lists.map (·.getD t [])))
        (sumVal ((List.range sh.S).map fun s =>
          if τ s % 2 ^ sh.tLog = t then v (offset (vParts sh) "eqs" + s) else 0)) := by
      intro t ht
      have := holds_xorAllL (forall₂_map_getD hL (t := t) fun x hx => by
          simp only [List.mem_map, List.mem_range] at hx
          obtain ⟨s, -, rfl⟩ := hx
          rw [List.length_map, List.length_range]
          exact ht)
        (by simpa using hne)
      rwa [demux_column _ τ sh.tLog sh.S _ htb ht] at this
    have H4 := holds_xorAllL (.cons (HR "bp" (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))
      (hsel _ (forall₂_zip (forall₂_range _ _ sh.T HWT) (forall₂_range _ _ sh.T fun t ht => hin.hvt hst t ht))))
      (List.cons_ne_nil _ _)
    have HL : List.Forall₂ (Holds z) [Flock.RecOpen.xorL c2 (elem out.toArray (names.idxOf "wv")),
        Flock.RecOpen.xorL c4 phi, Flock.RecSparse.xorAllL (elem out.toArray (names.idxOf "bp") :: sel),
        elem out.toArray (names.idxOf "claim")] (stepVals sh (rval sh w v) w v τ) := by
      simp only [stepVals, hst, ↓reduceIte]
      exact .cons H2 (.cons H3 (.cons H4 (.cons (HR "claim" (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl ⟨hst, rfl⟩))))))
        .nil)))
    obtain ⟨htl, htb'⟩ := tail_bits z (P "tau") sh.tLog ht16 sh.S (hτ hst)
    have h4l : (stepVals sh (rval sh w v) w v τ).length = 4 := by simp [stepVals, hst]
    have how := outWords_ge sh
    have hrn : (Flock.RecSparse.residualNames sh).length = 4 := by simp [Flock.RecSparse.residualNames, hst]
    simp only [hst, ↓reduceIte] at how
    refine ⟨length_out HL _ _ (by rw [h4l, htl]; omega), post_of HL _ _, fun _ s hs b hb => ?_⟩
    rw [bit_out_tail HL h4l _ _ (by rw [htl]; omega), htb' s hs b hb, Hτ s hs b hb, testBit_clear]
  · rename_i hst
    simp only [pure_bind]
    refine Sound.pure' fun z hphi hc4 hc2 hres w v τ hin => ?_
    obtain ⟨HR, HV⟩ := core z w v τ hin hres
    have H2 := holds_xorL (hc2 (fun j => rval sh w v s!"pre{j}") (fun j => v (offset (vParts sh) "pre" + 2 * j)) _
      (HR _ (Or.inl rfl)) (fun i hi => HR _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl ⟨i, hi, rfl⟩)))))))
      (fun i hi => HV _ (bPre i hi))) (HR "wv" (Or.inr (Or.inl rfl)))
    have hF := List.rel_append (List.rel_append (.cons (HR "f0" (Or.inr (Or.inr (Or.inl rfl)))) .nil)
      (forall₂_flatMap_range _ (fun j => [rval sh w v s!"fx{j}", rval sh w v s!"fy{j}"]) sh.ell fun j hj =>
        .cons (HR _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr ⟨j, hj, Or.inl rfl⟩))))))))
          (.cons (HR _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr ⟨j, hj, Or.inr rfl⟩))))))))
            .nil)))
      (.cons (HV _ bUr) .nil)
    have Hphi := hphi _ (by simp) hF
    rw [hF.length_eq] at Hphi
    have H3 := holds_xorL (hc4 (fun j => rval sh w v s!"sc{j}")
      (fun j => v (offset (vParts sh) "sc" + (sh.degrees.take j).foldl (· + ·) 0)) _ hin.hv2
      (fun i hi => HR _ (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl ⟨i, hi, rfl⟩))))))))
      (fun i hi => HV _ (bSc i hi))) Hphi
    have HL : List.Forall₂ (Holds z) [Flock.RecOpen.xorL c2 (elem out.toArray (names.idxOf "wv")),
        Flock.RecOpen.xorL c4 phi, elem out.toArray (names.idxOf "bp")] (stepVals sh (rval sh w v) w v τ) := by
      simp only [stepVals, hst, Bool.false_eq_true, ↓reduceIte]
      exact .cons H2 (.cons H3 (.cons (HR "bp" (Or.inr (Or.inr (Or.inr (Or.inl rfl))))) .nil))
    have h3l : (stepVals sh (rval sh w v) w v τ).length = 3 := by simp [stepVals, hst]
    have how := outWords_ge sh
    have hrn : (Flock.RecSparse.residualNames sh).length = 3 := by simp [Flock.RecSparse.residualNames, hst]
    simp only [hst, Bool.false_eq_true, ↓reduceIte] at how
    exact ⟨length_out HL _ _ (by rw [h3l, List.length_nil]; omega), post_of HL _ _, fun h => absurd h hst⟩

end Steps

end FlockVBridge.Sparse
