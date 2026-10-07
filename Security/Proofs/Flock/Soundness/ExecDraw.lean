import Proofs.Flock.Soundness.Audit.Law
import Flock.Draw

/-!
# The executable draw escapes no more often than the model's

`Flock.Draw.subset k n src` is the verifier's executable sampler (`backends/flock/verifier/lean/Flock/Draw.lean`): a
partial Fisher–Yates over a `Std.HashMap` of swaps, each index drawn by `Flock.Draw.uniform`, an exact rejection sampler
on big-endian words, and the `k` values sorted by `Array.qsort`. `Law.subset_exec_escape_le`: over a uniform stream of
bytes, it returns a draw that misses `B` with probability at most `(Law.subset n k hk).escape B`, the model's uniform
`k`-subset. A stream that runs out gives `none`, which only lowers the executable's escape.

* `mem_qsort`: `Array.qsort` keeps its elements, through core's private `qsort.sort` and `qpartition.loop`.
* `subset_eq`, `subset_misses`: the loop restated as a list recursion, and its `HashMap` read through a permutation `π`
  with `sw.getD t t = π t`: the draw misses `B` only if `π` does on `[0, k)`.
* `uniformL_le`: the rejection sampler. `uniform_streamOf` ties `Flock.Draw.uniform` and `beNat` at a position to it.
* `loopL_le`: step `i` draws uniformly over the pool `π '' [i, n)`, which holds all of `B` while the earlier values miss
  it, so the loop escapes `B` with probability at most `C(n − i − |B|, c)/C(n − i, c)`.
-/

open private Array.qsort.sort Array.qpartition.loop from Init.Data.Array.QSort.Basic

namespace FlockSoundness.Audit.Law

/-- The byte stream of `L` bytes `f`, as `Flock.Draw` reads it. -/
def streamOf {L : ℕ} (f : Fin L → Fin 256) : ByteArray := ⟨Array.ofFn fun i => UInt8.ofFin (f i)⟩

end FlockSoundness.Audit.Law


namespace FlockSoundness.Audit.DrawExec

theorem loop_perm {α : Type} {n : ℕ} (lt : α → α → Bool) (lo hi : ℕ) (hhi : hi < n) (pivot : α) :
    ∀ (d : ℕ) (as : Vector α n) (i k : ℕ) (ilo : lo ≤ i) (ik : i ≤ k) (w : k ≤ hi), hi - k = d →
      (Array.qpartition.loop lt lo hi hhi pivot as i k ilo ik w).2.toArray.Perm as.toArray := by
  intro d
  induction d with
  | zero =>
    intro as i k ilo ik w hd
    rw [Array.qpartition.loop.eq_def]
    have : ¬ k < hi := by omega
    simp only [this, dite_false]
    exact (Vector.swap_perm _ _).toArray
  | succ d ih =>
    intro as i k ilo ik w hd
    rw [Array.qpartition.loop.eq_def]
    have hk : k < hi := by omega
    simp only [hk, dite_true]
    split
    · exact (ih _ _ _ _ _ _ (by omega)).trans (Vector.swap_perm _ _).toArray
    · exact ih _ _ _ _ _ _ (by omega)

theorem qpartition_perm {α : Type} {n : ℕ} (as : Vector α n) (lt : α → α → Bool) (lo hi : ℕ) (w : lo ≤ hi)
    (hlo : lo < n) (hhi : hi < n) : (Array.qpartition as lt lo hi w hlo hhi).2.toArray.Perm as.toArray := by
  unfold Array.qpartition
  simp only
  refine (loop_perm lt lo hi hhi _ _ _ _ _ _ _ _ rfl).trans ?_
  have h1 : ∀ (b : Vector α n) (c : Bool) (i j : ℕ) (hi' : i < n) (hj : j < n),
      (if c = true then b.swap i j hi' hj else b).toArray.Perm b.toArray := by
    intro b c i j hi' hj
    split
    · exact (Vector.swap_perm _ _).toArray
    · exact Array.Perm.refl _
  exact (h1 _ _ _ _ _ _).trans ((h1 _ _ _ _ _ _).trans (h1 _ _ _ _ _ _))

theorem sort_perm {α : Type} (lt : α → α → Bool) {n : ℕ} :
    ∀ (d : ℕ) (as : Vector α n) (lo hi : ℕ) (w : lo ≤ hi) (hlo : lo < n) (hhi : hi < n), hi - lo = d →
      (Array.qsort.sort lt as lo hi w hlo hhi).toArray.Perm as.toArray := by
  intro d
  induction d using Nat.strong_induction_on with
  | _ d ih =>
    intro as lo hi w hlo hhi hd
    rw [Array.qsort.sort.eq_def]
    split
    · rename_i hlt
      generalize hq : Array.qpartition as lt lo hi w hlo hhi = q
      obtain ⟨⟨mid, hmid⟩, as'⟩ := q
      have hp : as'.toArray.Perm as.toArray := by
        have := qpartition_perm as lt lo hi w hlo hhi
        rw [hq] at this
        exact this
      simp only
      split
      · exact hp
      · rename_i hge
        refine ((ih (hi - (mid + 1)) (by omega) _ _ _ _ _ _ rfl).trans
          ((ih (mid - lo) (by omega) _ _ _ _ _ _ rfl))).trans hp
    · exact Array.Perm.refl _

theorem mem_qsort {α : Type} (as : Array α) (lt : α → α → Bool) (x : α) : x ∈ as.qsort lt ↔ x ∈ as := by
  unfold Array.qsort
  split
  · exact Iff.rfl
  · exact (sort_perm lt _ _ _ _ _ _ _ rfl).mem_iff


open Flock.Draw

abbrev St := Option (Option (Array ℕ)) × Std.HashMap ℕ ℕ × ℕ

/-- One swap of `subset`'s Fisher–Yates: positions `i` and `i + r`. -/
def swapAt (sw : Std.HashMap ℕ ℕ) (i r : ℕ) : Std.HashMap ℕ ℕ :=
  (sw.insert i (sw.getD (i + r) (i + r))).insert (i + r) (sw.getD i i)

/-- `subset`'s loop as a list recursion. -/
def loopSw (n : ℕ) (src : ByteArray) : List ℕ → Std.HashMap ℕ ℕ → ℕ → Option (Std.HashMap ℕ ℕ)
  | [], sw, _ => some sw
  | i :: is, sw, pos => match uniform (n - i) src (src.size + 1) pos with
    | none => none
    | some (r, p) => loopSw n src is (swapAt sw i r) p

/-- A `forIn` whose body draws with `uniform` and swaps, followed by a continuation that returns the early exit or
finishes: the loop computes `loopSw`. -/
theorem forIn_char (n : ℕ) (src : ByteArray) (G : Std.HashMap ℕ ℕ → Array ℕ)
    (F : ℕ → St → Id (ForInStep St)) (POST : St → Id (Option (Array ℕ)))
    (hFn : ∀ i s, uniform (n - i) src (src.size + 1) s.2.2 = none → F i s = pure (.done (some none, s.2.1, s.2.2)))
    (hFs : ∀ i s r p, uniform (n - i) src (src.size + 1) s.2.2 = some (r, p) →
      F i s = pure (.yield (none, swapAt s.2.1 i r, p)))
    (hPs : ∀ s r, s.1 = some r → POST s = pure r) (hPn : ∀ s, s.1 = none → POST s = pure (some (G s.2.1))) :
    ∀ (l : List ℕ) (sw : Std.HashMap ℕ ℕ) (pos : ℕ),
      (forIn (m := Id) l ((none : Option (Option (Array ℕ))), sw, pos) F >>= POST).run =
        (loopSw n src l sw pos).map G := by
  intro l
  induction l with
  | nil =>
    intro sw pos
    simp only [List.forIn_nil, loopSw, Option.map_some]
    rw [pure_bind, hPn _ rfl]
    rfl
  | cons i is ih =>
    intro sw pos
    rw [List.forIn_cons]
    cases h : uniform (n - i) src (src.size + 1) pos with
    | none =>
      rw [hFn i _ h]
      simp only [loopSw, h, Option.map_none, pure_bind]
      rw [hPs _ none rfl]
      rfl
    | some rp =>
      obtain ⟨r, p⟩ := rp
      rw [hFs i _ r p h]
      simp only [loopSw, h, pure_bind]
      exact ih _ _

theorem subset_eq (k n : ℕ) (src : ByteArray) :
    subset k n src = (loopSw n src (List.range' 0 k) ∅ 0).map
      (fun sw => ((Array.range k).map fun i => sw.getD i i).qsort (· < ·)) := by
  unfold subset
  dsimp only
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  have hsize : ([:k] : Std.Legacy.Range).size = k := by
    show (k - 0 + 1 - 1) / 1 = k
    omega
  rw [hsize]
  refine forIn_char n src _ _ _ ?_ ?_ ?_ ?_ (List.range' 0 k) ∅ 0
  · intro i s h; simp only [h]
  · intro i s r p h; simp only [h]; rfl
  · intro s r h; simp only [h]
  · intro s h; simp only [h]

/-- The loop on the permutation `t ↦ sw.getD t t`. -/
def loopP (n : ℕ) (src : ByteArray) : List ℕ → Equiv.Perm ℕ → ℕ → Option (Equiv.Perm ℕ)
  | [], π, _ => some π
  | i :: is, π, pos => match uniform (n - i) src (src.size + 1) pos with
    | none => none
    | some (r, p) => loopP n src is ((Equiv.swap i (i + r)).trans π) p

theorem getD_swapAt (sw : Std.HashMap ℕ ℕ) (π : Equiv.Perm ℕ) (h : ∀ t, sw.getD t t = π t) (i r t : ℕ) :
    (swapAt sw i r).getD t t = ((Equiv.swap i (i + r)).trans π) t := by
  simp only [swapAt, Std.HashMap.getD_insert, beq_iff_eq, h, Equiv.trans_apply, Equiv.swap_apply_def]
  by_cases h1 : i + r = t
  · subst h1
    by_cases h2 : i + r = i
    · simp [h2]
    · simp
  · by_cases h2 : i = t
    · subst h2; simp
    · simp [h1, h2, Ne.symm h1, Ne.symm h2]

theorem loopSw_loopP (n : ℕ) (src : ByteArray) : ∀ (l : List ℕ) (sw : Std.HashMap ℕ ℕ) (π : Equiv.Perm ℕ) (pos : ℕ),
    (∀ t, sw.getD t t = π t) →
      ∀ sw', loopSw n src l sw pos = some sw' → ∃ π', loopP n src l π pos = some π' ∧ ∀ t, sw'.getD t t = π' t := by
  intro l
  induction l with
  | nil => intro sw π pos h sw' hs; simp only [loopSw, Option.some.injEq] at hs; subst hs; exact ⟨π, rfl, h⟩
  | cons i is ih =>
    intro sw π pos h sw' hs
    simp only [loopSw] at hs
    simp only [loopP]
    cases hu : uniform (n - i) src (src.size + 1) pos with
    | none => rw [hu] at hs; exact absurd hs (by simp)
    | some rp =>
      obtain ⟨r, p⟩ := rp
      rw [hu] at hs
      exact ih _ _ _ (getD_swapAt sw π h i r) sw' hs

/-- **The executable's output misses `B` only if the loop's permutation does on `[0, k)`.** -/
theorem subset_misses (k n : ℕ) (src : ByteArray) (B : Finset ℕ) (D : Array ℕ) (hD : subset k n src = some D)
    (hB : ∀ u ∈ B, u ∉ D) : ∃ π, loopP n src (List.range' 0 k) (Equiv.refl ℕ) 0 = some π ∧ ∀ i < k, π i ∉ B := by
  rw [subset_eq] at hD
  cases hl : loopSw n src (List.range' 0 k) ∅ 0 with
  | none => rw [hl] at hD; exact absurd hD (by simp)
  | some sw =>
    rw [hl, Option.map_some, Option.some.injEq] at hD
    obtain ⟨π, hπ, hsw⟩ := loopSw_loopP n src _ ∅ (Equiv.refl ℕ) 0 (fun t => by simp) sw hl
    refine ⟨π, hπ, fun i hi hmem => hB _ hmem ?_⟩
    rw [← hD, mem_qsort, Array.mem_map]
    exact ⟨i, Array.mem_range.2 hi, hsw i⟩

open Game Finset
open scoped ENNReal

/-! ## Uniform byte streams -/

/-- The bytes of `g`. -/
def bsOf {ℓ : ℕ} (g : Fin ℓ → Fin 256) : List UInt8 := List.ofFn fun i => UInt8.ofFin (g i)

/-- A big-endian word, as `Flock.Draw.beNat` reads one. -/
def beL (bs : List UInt8) : ℕ := bs.foldl (fun acc b => acc * 256 + b.toNat) 0

theorem length_bsOf {ℓ : ℕ} (g : Fin ℓ → Fin 256) : (bsOf g).length = ℓ := by simp [bsOf]

theorem avg_equiv {C D : Type} [Fintype C] [Fintype D] (e : C ≃ D) (f : D → ℝ≥0∞) :
    avg (fun c => f (e c)) = avg f := by
  unfold avg
  rw [Fintype.card_congr e]
  exact Fintype.sum_equiv e _ _ (fun _ => rfl)

theorem avg_prod {C D : Type} [Fintype C] [Fintype D] (f : C × D → ℝ≥0∞) :
    avg f = avg fun c => avg fun d => f (c, d) := by
  unfold avg
  rw [Fintype.sum_prod_type, Fintype.card_prod, Nat.cast_mul,
    ENNReal.mul_inv (Or.inr (ENNReal.natCast_ne_top _)) (Or.inl (ENNReal.natCast_ne_top _))]
  refine Finset.sum_congr rfl fun c _ => ?_
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun d _ => ?_
  ring

theorem avg_comm {C D : Type} [Fintype C] [Fintype D] (f : C → D → ℝ≥0∞) :
    (avg fun c => avg fun d => f c d) = avg fun d => avg fun c => f c d := by
  rw [← avg_prod (fun p : C × D => f p.1 p.2), ← avg_prod (fun p : D × C => f p.2 p.1),
    ← avg_equiv (Equiv.prodComm C D) (fun p : D × C => f p.2 p.1)]
  rfl

/-- Splitting a stream into its first `w` bytes and the rest. -/
def appendEquiv (w ℓ : ℕ) : (Fin w → Fin 256) × (Fin ℓ → Fin 256) ≃ (Fin (w + ℓ) → Fin 256) where
  toFun p := Fin.append p.1 p.2
  invFun g := (fun i => g (Fin.castAdd ℓ i), fun j => g (Fin.natAdd w j))
  left_inv p := by
    obtain ⟨a, b⟩ := p
    simp only [Prod.mk.injEq]
    exact ⟨funext fun i => Fin.append_left a b i, funext fun j => Fin.append_right a b j⟩
  right_inv g := by
    funext i
    refine Fin.addCases (fun a => ?_) (fun b => ?_) i
    · simp [Fin.append_left]
    · simp [Fin.append_right]

theorem bsOf_append {w ℓ : ℕ} (a : Fin w → Fin 256) (b : Fin ℓ → Fin 256) :
    bsOf (Fin.append a b) = bsOf a ++ bsOf b := by
  unfold bsOf
  rw [← List.ofFn_fin_append]
  congr 1
  funext i
  refine Fin.addCases (fun x => ?_) (fun y => ?_) i
  · simp [Fin.append_left]
  · simp [Fin.append_right]

theorem avg_split {w ℓ : ℕ} (H : List UInt8 → ℝ≥0∞) :
    avg (fun g : Fin (w + ℓ) → Fin 256 => H (bsOf g)) =
      avg fun a : Fin w → Fin 256 => avg fun b : Fin ℓ → Fin 256 => H (bsOf a ++ bsOf b) := by
  rw [← avg_equiv (appendEquiv w ℓ), avg_prod]
  simp only [appendEquiv, Equiv.coe_fn_mk, bsOf_append]

theorem beL_snoc (l : List UInt8) (x : UInt8) : beL (l ++ [x]) = beL l * 256 + x.toNat := by
  simp [beL, List.foldl_append]

theorem sum_range_mul_add (F : ℕ → ℝ≥0∞) : ∀ N : ℕ,
    ∑ y ∈ range N, ∑ x ∈ range 256, F (y * 256 + x) = ∑ z ∈ range (N * 256), F z := by
  intro N
  induction N with
  | zero => simp
  | succ N ih =>
    rw [Finset.sum_range_succ, ih, Nat.succ_mul, Finset.sum_range_add]

/-- **A uniform `w`-byte word is uniform on `[0, 256^w)`.** -/
theorem avg_beL (F : ℕ → ℝ≥0∞) : ∀ w : ℕ,
    avg (fun g : Fin w → Fin 256 => F (beL (bsOf g))) = (∑ z ∈ range (256 ^ w), F z) / ((256 ^ w : ℕ) : ℝ≥0∞) := by
  intro w
  induction w generalizing F with
  | zero =>
    have : ∀ g : Fin 0 → Fin 256, bsOf g = [] := fun g => by simp [bsOf]
    simp only [this, pow_zero, Finset.range_one, Finset.sum_singleton, Nat.cast_one, div_one]
    rw [avg_const]
    rfl
  | succ w ih =>
    rw [← avg_equiv (Fin.snocEquiv fun _ => Fin 256), avg_prod]
    have hs : ∀ (x : Fin 256) (g : Fin w → Fin 256),
        bsOf ((Fin.snocEquiv fun _ => Fin 256) (x, g)) = bsOf g ++ [UInt8.ofFin x] := by
      intro x g
      have hsn : (Fin.snocEquiv fun _ => Fin 256) (x, g) = Fin.snoc g x := rfl
      rw [hsn]
      simp only [bsOf, List.ofFn_succ', Fin.snoc_castSucc, Fin.snoc_last, List.concat_eq_append]
    simp only [hs, beL_snoc, UInt8.toNat_ofFin]
    rw [avg_comm]
    have h256 : ∀ y, avg (fun x : Fin 256 => F (y * 256 + ↑x)) = (∑ x ∈ range 256, F (y * 256 + x)) / (256 : ℝ≥0∞) := by
      intro y
      unfold avg
      rw [Fintype.card_fin, ← Finset.mul_sum, Fin.sum_univ_eq_sum_range (fun x => F (y * 256 + x)) 256, mul_comm,
        ENNReal.div_eq_inv_mul, mul_comm]
      norm_num
    simp only [h256]
    rw [ih (fun y => (∑ x ∈ range 256, F (y * 256 + x)) / (256 : ℝ≥0∞)), ← ENNReal.sum_div, sum_range_mul_add,
      ENNReal.div_eq_inv_mul, ENNReal.div_eq_inv_mul, ENNReal.div_eq_inv_mul, ← mul_assoc,
      ← ENNReal.mul_inv (Or.inr (by norm_num)) (Or.inr (by norm_num)), pow_succ, Nat.cast_mul]
    norm_num

/-! ## The rejection sampler -/

/-- `Flock.Draw.uniform m` on the rest of the stream: the value and the stream after it. -/
def uniformL (m : ℕ) : ℕ → List UInt8 → Option (ℕ × List UInt8)
  | 0, _ => none
  | fuel + 1, bs =>
    if bs.length < Flock.Draw.width m then none
    else if beL (bs.take (Flock.Draw.width m)) < 2 ^ (8 * Flock.Draw.width m) - 2 ^ (8 * Flock.Draw.width m) % m then
      some (beL (bs.take (Flock.Draw.width m)) % m, bs.drop (Flock.Draw.width m))
    else uniformL m fuel (bs.drop (Flock.Draw.width m))

theorem sum_range_mod (m : ℕ) (q : ℕ → ℝ≥0∞) : ∀ t : ℕ,
    ∑ z ∈ range (m * t), q (z % m) = t * ∑ r ∈ range m, q r := by
  intro t
  induction t with
  | zero => simp
  | succ t ih =>
    rw [Nat.mul_succ, Finset.sum_range_add, ih, Nat.cast_succ, add_mul, one_mul]
    congr 1
    refine Finset.sum_congr rfl fun x hx => ?_
    rw [Nat.mul_add_mod_self_left, Nat.mod_eq_of_lt (Finset.mem_range.1 hx)]

theorem reject_arith (m S : ℕ) (hm : 0 < m) (hS : 0 < S) (q : ℕ → ℝ≥0∞) :
    (∑ z ∈ range S, if z < S - S % m then q (z % m) else (∑ r ∈ range m, q r) / m) / (S : ℝ≥0∞) =
      (∑ r ∈ range m, q r) / m := by
  set Q := (∑ r ∈ range m, q r) / (m : ℝ≥0∞) with hQ
  have hA : S - S % m = m * (S / m) := by have := Nat.div_add_mod S m; omega
  have hsplit : S = m * (S / m) + (S - m * (S / m)) := by have := Nat.div_add_mod S m; omega
  have hmQ : (m : ℝ≥0∞) * Q = ∑ r ∈ range m, q r := by
    rw [hQ, ENNReal.mul_div_cancel (by exact_mod_cast hm.ne') (ENNReal.natCast_ne_top _)]
  have hsum : (∑ z ∈ range S, if z < S - S % m then q (z % m) else Q) = (S : ℝ≥0∞) * Q := by
    rw [hA]
    obtain ⟨t, ht⟩ : ∃ t, t = S / m := ⟨_, rfl⟩
    obtain ⟨R, hR⟩ : ∃ R, R = S - m * t := ⟨_, rfl⟩
    have hSR : S = m * t + R := by rw [hR, ht]; exact hsplit
    rw [← ht, hSR, Finset.sum_range_add]
    have h1 : (∑ z ∈ range (m * t), if z < m * t then q (z % m) else Q) = ∑ z ∈ range (m * t), q (z % m) :=
      Finset.sum_congr rfl fun z hz => by rw [ite_eq_left (Finset.mem_range.1 hz)]
    have h2 : (∑ x ∈ range R, if m * t + x < m * t then q ((m * t + x) % m) else Q) = ∑ x ∈ range R, Q :=
      Finset.sum_congr rfl fun x _ => by rw [ite_eq_right (by omega)]
    rw [h1, h2, sum_range_mod m, Finset.sum_const, Finset.card_range, nsmul_eq_mul, ← hmQ, ← mul_assoc,
      ← add_mul, mul_comm (t : ℝ≥0∞), ← Nat.cast_mul, ← Nat.cast_add]
  rw [hsum, mul_comm, ENNReal.mul_div_cancel_right (by exact_mod_cast hS.ne') (ENNReal.natCast_ne_top _)]

theorem width_pos (m : ℕ) : 0 < Flock.Draw.width m := by
  unfold Flock.Draw.width; omega

open Classical in
/-- **The rejection sampler**: if every value `r < m` it can return is followed by an event of probability at most `q r`
on the rest of the stream, the draw returns some `r` and then that event with probability at most the average of `q`. -/
theorem uniformL_le (m : ℕ) (hm : 0 < m) (Q : ℕ → List UInt8 → Prop) (q : ℕ → ℝ≥0∞)
    (hQ : ∀ r < m, ∀ ℓ, prCoin (fun h : Fin ℓ → Fin 256 => Q r (bsOf h)) ≤ q r) :
    ∀ fuel ℓ, prCoin (fun g : Fin ℓ → Fin 256 => ∃ r rest, uniformL m fuel (bsOf g) = some (r, rest) ∧ Q r rest) ≤
      (∑ r ∈ range m, q r) / m := by
  intro fuel
  induction fuel with
  | zero => intro ℓ; simp [uniformL, prCoin_false]
  | succ fuel ih =>
    intro ℓ
    by_cases hℓ : ℓ < Flock.Draw.width m
    · refine le_of_eq_of_le (b := 0) ?_ (zero_le)
      convert prCoin_false (C := Fin ℓ → Fin 256) using 2 with g
      simp [uniformL, length_bsOf, hℓ]
    · obtain ⟨ℓ', rfl⟩ : ∃ ℓ', ℓ = Flock.Draw.width m + ℓ' := ⟨ℓ - Flock.Draw.width m, by omega⟩
      set w := Flock.Draw.width m with hw
      set S := 2 ^ (8 * w) with hS
      unfold prCoin
      rw [avg_split (H := fun bs => if ∃ r rest, uniformL m (fuel + 1) bs = some (r, rest) ∧ Q r rest then 1 else 0)]
      have hstep : ∀ (a : Fin w → Fin 256) (b : Fin ℓ' → Fin 256),
          uniformL m (fuel + 1) (bsOf a ++ bsOf b) =
            if beL (bsOf a) < S - S % m then some (beL (bsOf a) % m, bsOf b) else uniformL m fuel (bsOf b) := by
        intro a b
        have hl : (bsOf a).length = w := length_bsOf a
        simp only [uniformL, List.length_append, hl, length_bsOf, List.take_left' hl, List.drop_left' hl, ← hw, ← hS]
        simp
      simp only [hstep]
      calc (avg fun a : Fin w → Fin 256 => avg fun b : Fin ℓ' → Fin 256 =>
            if ∃ r rest, (if beL (bsOf a) < S - S % m then some (beL (bsOf a) % m, bsOf b)
              else uniformL m fuel (bsOf b)) = some (r, rest) ∧ Q r rest then (1 : ℝ≥0∞) else 0)
          ≤ avg fun a : Fin w → Fin 256 =>
            (fun z => if z < S - S % m then q (z % m) else (∑ r ∈ range m, q r) / m) (beL (bsOf a)) := by
            refine avg_mono fun a => ?_
            simp only
            split
            · rename_i hlt
              have hr : beL (bsOf a) % m < m := Nat.mod_lt _ hm
              refine le_trans (le_of_eq ?_) (hQ _ hr ℓ')
              unfold prCoin
              congr 1
              funext b
              congr 1
              apply propext
              constructor
              · rintro ⟨r, rest, h, hq⟩
                simp only [Option.some.injEq, Prod.mk.injEq] at h
                obtain ⟨rfl, rfl⟩ := h
                exact hq
              · intro hq
                exact ⟨_, _, rfl, hq⟩
            · exact ih ℓ'
        _ = (∑ z ∈ range S, if z < S - S % m then q (z % m) else (∑ r ∈ range m, q r) / m) / (S : ℝ≥0∞) := by
            have hSw : S = 256 ^ w := by rw [hS, pow_mul]; norm_num
            refine (avg_beL (fun z => if z < S - S % m then q (z % m) else (∑ r ∈ range m, q r) / m) w).trans ?_
            rw [← hSw]
        _ = (∑ r ∈ range m, q r) / m := reject_arith m S hm (by positivity) q

/-! ## The executable's reads, on the rest of the stream -/

theorem size_streamOf {L : ℕ} (f : Fin L → Fin 256) : (Law.streamOf f).size = L := by
  simp [Law.streamOf, ByteArray.size]

theorem get!_streamOf {L : ℕ} (f : Fin L → Fin 256) (j : ℕ) (hj : j < L) :
    (Law.streamOf f).get! j = UInt8.ofFin (f ⟨j, hj⟩) := by
  show (Array.ofFn fun i => UInt8.ofFin (f i))[j]! = _
  simp [hj]

theorem take_drop_bsOf {L : ℕ} (f : Fin L → Fin 256) (pos w : ℕ) (h : pos + w ≤ L) :
    ((bsOf f).drop pos).take w = (List.range w).map fun i => (Law.streamOf f).get! (pos + i) := by
  apply List.ext_getElem
  · simp [length_bsOf]; omega
  · intro i h1 h2
    simp only [List.getElem_take, List.getElem_drop, List.getElem_map, List.getElem_range, bsOf, List.getElem_ofFn]
    rw [get!_streamOf f (pos + i) (by simp at h2; omega)]

theorem beNat_streamOf {L : ℕ} (f : Fin L → Fin 256) (pos w : ℕ) (h : pos + w ≤ L) :
    Flock.Draw.beNat (Law.streamOf f) pos w = beL (((bsOf f).drop pos).take w) := by
  rw [take_drop_bsOf f pos w h, beL, List.foldl_map]
  rfl

/-- **`Flock.Draw.uniform` at position `pos` is `uniformL` on the stream from `pos`.** -/
theorem uniform_streamOf {L : ℕ} (f : Fin L → Fin 256) (m : ℕ) : ∀ fuel pos, pos ≤ L →
    match uniformL m fuel ((bsOf f).drop pos) with
    | none => Flock.Draw.uniform m (Law.streamOf f) fuel pos = none
    | some (r, rest) => ∃ p, p ≤ L ∧ rest = (bsOf f).drop p ∧ Flock.Draw.uniform m (Law.streamOf f) fuel pos = some (r, p) := by
  intro fuel
  induction fuel with
  | zero => intro pos _; simp [uniformL, Flock.Draw.uniform]
  | succ fuel ih =>
    intro pos hpos
    have hlen : ((bsOf f).drop pos).length = L - pos := by simp [length_bsOf]
    by_cases hw : L - pos < Flock.Draw.width m
    · have hu : uniformL m (fuel + 1) ((bsOf f).drop pos) = none := by simp [uniformL, hlen, hw]
      rw [hu]
      simp only [Flock.Draw.uniform, size_streamOf]
      rw [ite_eq_left (by omega)]
    · have hw' : pos + Flock.Draw.width m ≤ L := by omega
      have hx0 := beNat_streamOf f pos _ hw'
      have hdd : ((bsOf f).drop pos).drop (Flock.Draw.width m) = (bsOf f).drop (pos + Flock.Draw.width m) := by
        rw [List.drop_drop]
      by_cases hx : Flock.Draw.beNat (Law.streamOf f) pos (Flock.Draw.width m) <
          2 ^ (8 * Flock.Draw.width m) - 2 ^ (8 * Flock.Draw.width m) % m
      · have hu : uniformL m (fuel + 1) ((bsOf f).drop pos) =
            some (Flock.Draw.beNat (Law.streamOf f) pos (Flock.Draw.width m) % m, (bsOf f).drop (pos + Flock.Draw.width m)) := by
          simp only [uniformL, hlen, hw, ite_false, ← hx0, hx, ite_true, hdd]
        rw [hu]
        refine ⟨pos + Flock.Draw.width m, hw', rfl, ?_⟩
        simp only [Flock.Draw.uniform, size_streamOf]
        rw [ite_eq_right (by omega)]
        simp only [hx, ite_true]
      · have hu : uniformL m (fuel + 1) ((bsOf f).drop pos) =
            uniformL m fuel ((bsOf f).drop (pos + Flock.Draw.width m)) := by
          simp only [uniformL, hlen, hw, ite_false, ← hx0, hx, hdd]
        rw [hu]
        have he : Flock.Draw.uniform m (Law.streamOf f) (fuel + 1) pos =
            Flock.Draw.uniform m (Law.streamOf f) fuel (pos + Flock.Draw.width m) := by
          simp only [Flock.Draw.uniform, size_streamOf]
          rw [ite_eq_right (by omega)]
          simp only [hx, ite_false]
        rw [he]
        exact ih (pos + Flock.Draw.width m) hw'

/-! ## The permutation loop, on the rest of the stream -/

/-- `loopP` on the rest of the stream, with `uniformL`'s fuel `F`. -/
def loopL (n F : ℕ) : List ℕ → Equiv.Perm ℕ → List UInt8 → Option (Equiv.Perm ℕ)
  | [], π, _ => some π
  | i :: is, π, bs => match uniformL (n - i) F bs with
    | none => none
    | some (r, rest) => loopL n F is ((Equiv.swap i (i + r)).trans π) rest

theorem loopP_streamOf {L : ℕ} (f : Fin L → Fin 256) (n : ℕ) : ∀ (l : List ℕ) (π : Equiv.Perm ℕ) (pos : ℕ), pos ≤ L →
    loopP n (Law.streamOf f) l π pos = loopL n (L + 1) l π ((bsOf f).drop pos) := by
  intro l
  induction l with
  | nil => intro π pos _; rfl
  | cons i is ih =>
    intro π pos hpos
    have h := uniform_streamOf f (n - i) (L + 1) pos hpos
    simp only [loopP, loopL, size_streamOf]
    revert h
    cases hu : uniformL (n - i) (L + 1) ((bsOf f).drop pos) with
    | none => intro h; simp only at h; rw [h]
    | some rr =>
      obtain ⟨r, rest⟩ := rr
      intro h
      simp only at h
      obtain ⟨p, hp, hrest, hU⟩ := h
      rw [hU]
      simp only
      rw [ih _ p hp, hrest]

/-- The loop never changes the values below its first index. -/
theorem loopL_prefix (n F : ℕ) : ∀ (c j : ℕ) (π π' : Equiv.Perm ℕ) (bs : List UInt8),
    loopL n F (List.range' j c) π bs = some π' → ∀ t < j, π' t = π t := by
  intro c
  induction c with
  | zero => intro j π π' bs h t _; simp only [List.range'_zero, loopL, Option.some.injEq] at h; rw [h]
  | succ c ih =>
    intro j π π' bs h t ht
    rw [List.range'_succ] at h
    simp only [loopL] at h
    split at h
    · exact absurd h (by simp)
    · rename_i r rest _
      rw [ih (j + 1) _ π' rest h t (by omega), Equiv.trans_apply,
        Equiv.swap_apply_of_ne_of_ne (by omega) (by omega)]

theorem prCoin_le_one {C : Type} [Fintype C] (p : C → Prop) : prCoin p ≤ 1 := by
  classical
  unfold prCoin avg
  calc (∑ c : C, (Fintype.card C : ℝ≥0∞)⁻¹ * (if p c then 1 else 0)) ≤ ∑ c : C, (Fintype.card C : ℝ≥0∞)⁻¹ * 1 :=
        Finset.sum_le_sum fun c _ => by gcongr; split <;> simp
    _ ≤ 1 := by
        rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, mul_one]
        exact ENNReal.mul_inv_le_one _

/-- A map of `[0, n)` into itself by a permutation is onto. -/
theorem perm_onto {n : ℕ} (π : Equiv.Perm ℕ) (hπ : ∀ t < n, π t < n) (u : ℕ) (hu : u < n) : ∃ t < n, π t = u := by
  classical
  have hsub : (range n).image π ⊆ range n := fun x hx => by
    obtain ⟨t, ht, rfl⟩ := mem_image.1 hx
    exact mem_range.2 (hπ t (mem_range.1 ht))
  have hcard : (range n).card ≤ ((range n).image π).card := by
    rw [card_image_of_injective _ π.injective]
  have heq := Finset.eq_of_subset_of_card_le hsub hcard
  have : u ∈ (range n).image π := by rw [heq]; exact mem_range.2 hu
  obtain ⟨t, ht, h⟩ := mem_image.1 this
  exact ⟨t, mem_range.1 ht, h⟩

/-- **The pool**: of the `n − i` values step `i` can take, exactly `|B|` are in `B`, while no earlier value is. -/
theorem card_pool_mem (n i : ℕ) (hi : i ≤ n) (B : Finset ℕ) (hB : ∀ u ∈ B, u < n) (π : Equiv.Perm ℕ)
    (hπ : ∀ t < n, π t < n) (hpre : ∀ t < i, π t ∉ B) :
    ((range (n - i)).filter fun r => π (i + r) ∈ B).card = B.card := by
  classical
  have hinj : Set.InjOn (fun r => π (i + r)) ((range (n - i)).filter fun r => π (i + r) ∈ B) :=
    fun a _ b _ h => by simpa using h
  rw [← card_image_of_injOn hinj]
  congr 1
  ext u
  simp only [mem_image, mem_filter, mem_range]
  constructor
  · rintro ⟨r, ⟨_, hr⟩, rfl⟩; exact hr
  · intro hu
    obtain ⟨t, ht, rfl⟩ := perm_onto π hπ u (hB u hu)
    have hti : i ≤ t := by
      by_contra h
      exact hpre t (by omega) hu
    exact ⟨t - i, ⟨by omega, by rwa [Nat.add_sub_cancel' hti]⟩, by rw [Nat.add_sub_cancel' hti]⟩

/-- One step of the hypergeometric: `(m − b)/m · C(m − 1 − b, c)/C(m − 1, c) = C(m − b, c + 1)/C(m, c + 1)`. -/
theorem choose_step_ratio (m b c : ℕ) (hc : c + 1 ≤ m) :
    ((m - b : ℕ) : ℝ≥0∞) * (((m - 1 - b).choose c : ℝ≥0∞) / ((m - 1).choose c : ℝ≥0∞)) / (m : ℝ≥0∞) =
      ((m - b).choose (c + 1) : ℝ≥0∞) / (m.choose (c + 1) : ℝ≥0∞) := by
  have hP : m * (m - 1).choose c = m.choose (c + 1) * (c + 1) := by
    have := Nat.add_one_mul_choose_eq (m - 1) c
    rwa [Nat.sub_add_cancel (by omega)] at this
  have hX : (m - b) * (m - 1 - b).choose c = (m - b).choose (c + 1) * (c + 1) := by
    rcases Nat.eq_zero_or_pos (m - b) with h0 | hpos
    · rw [h0, zero_mul, Nat.choose_zero_succ, zero_mul]
    · have := Nat.add_one_mul_choose_eq (m - b - 1) c
      rw [Nat.sub_add_cancel hpos] at this
      rw [show m - 1 - b = m - b - 1 by omega]
      exact this
  have ht : ∀ x : ℕ, (x : ℝ≥0∞) ≠ ⊤ := ENNReal.natCast_ne_top
  have hL : ((m - b : ℕ) : ℝ≥0∞) * (((m - 1 - b).choose c : ℝ≥0∞) / ((m - 1).choose c : ℝ≥0∞)) / (m : ℝ≥0∞) =
      (((m - b) * (m - 1 - b).choose c : ℕ) : ℝ≥0∞) / ((m * (m - 1).choose c : ℕ) : ℝ≥0∞) := by
    rw [Nat.cast_mul, Nat.cast_mul, ENNReal.div_eq_inv_mul, ENNReal.div_eq_inv_mul, ENNReal.div_eq_inv_mul,
      ENNReal.mul_inv (Or.inr (ht _)) (Or.inl (ht _))]
    ring
  rw [hL, hX, hP, Nat.cast_mul, Nat.cast_mul,
    ENNReal.mul_div_mul_right _ _ (by exact_mod_cast (show c + 1 ≠ 0 by omega)) (ht _)]

theorem swap_trans_lt {n i r : ℕ} (hr : i + r < n) (π : Equiv.Perm ℕ) (hπ : ∀ t < n, π t < n) :
    ∀ t < n, ((Equiv.swap i (i + r)).trans π) t < n := by
  intro t ht
  rw [Equiv.trans_apply]
  refine hπ _ ?_
  rw [Equiv.swap_apply_def]
  split_ifs <;> omega

open Classical in
/-- **The loop's escape**: from step `i`, with `π` mapping `[0, n)` into itself and no earlier value in `B`, the rest of
the loop returns a permutation with no value below `k` in `B` with probability at most `C(n − i − |B|, c)/C(n − i, c)`. -/
theorem loopL_le (n k F : ℕ) (B : Finset ℕ) (hB : ∀ u ∈ B, u < n) : ∀ (c i : ℕ) (π : Equiv.Perm ℕ),
    i + c = k → k ≤ n → (∀ t < n, π t < n) → (∀ t < i, π t ∉ B) → ∀ ℓ,
      prCoin (fun h : Fin ℓ → Fin 256 => ∃ π', loopL n F (List.range' i c) π (bsOf h) = some π' ∧ ∀ t < k, π' t ∉ B) ≤
        ((n - i - B.card).choose c : ℝ≥0∞) / ((n - i).choose c : ℝ≥0∞) := by
  intro c
  induction c with
  | zero =>
    intro i π _ _ _ _ ℓ
    simp only [Nat.choose_zero_right, Nat.cast_one, div_one]
    exact prCoin_le_one _
  | succ c ih =>
    intro i π hk hkn hπ hpre ℓ
    have hm : 0 < n - i := by omega
    set E' := ((n - (i + 1) - B.card).choose c : ℝ≥0∞) / ((n - (i + 1)).choose c : ℝ≥0∞) with hE'
    let Q : ℕ → List UInt8 → Prop := fun r rest =>
      ∃ π', loopL n F (List.range' (i + 1) c) ((Equiv.swap i (i + r)).trans π) rest = some π' ∧ ∀ t < k, π' t ∉ B
    let q : ℕ → ℝ≥0∞ := fun r => if π (i + r) ∈ B then 0 else E'
    have hev : ∀ bs, (∃ π', loopL n F (List.range' i (c + 1)) π bs = some π' ∧ ∀ t < k, π' t ∉ B) ↔
        ∃ r rest, uniformL (n - i) F bs = some (r, rest) ∧ Q r rest := by
      intro bs
      rw [List.range'_succ]
      simp only [loopL]
      constructor
      · rintro ⟨π', h, hB'⟩
        split at h
        · exact absurd h (by simp)
        · rename_i r rest hu
          exact ⟨r, rest, hu, π', h, hB'⟩
      · rintro ⟨r, rest, hu, π', h, hB'⟩
        refine ⟨π', ?_, hB'⟩
        rw [hu]
        exact h
    have hQ : ∀ r < n - i, ∀ ℓ', prCoin (fun h : Fin ℓ' → Fin 256 => Q r (bsOf h)) ≤ q r := by
      intro r hr ℓ'
      by_cases hin : π (i + r) ∈ B
      · simp only [q, hin, ite_true]
        refine (prCoin_mono fun h hQ' => ?_).trans (prCoin_false).le
        obtain ⟨π', h', hB'⟩ := hQ'
        have := loopL_prefix n F c (i + 1) _ π' _ h' i (by omega)
        rw [Equiv.trans_apply, Equiv.swap_apply_left] at this
        exact hB' i (by omega) (this ▸ hin)
      · simp only [q, hin, ite_false]
        refine ih (i + 1) _ (by omega) hkn (swap_trans_lt (by omega) π hπ) (fun t ht => ?_) ℓ'
        rw [Equiv.trans_apply]
        rcases Nat.lt_or_ge t i with hti | hti
        · rw [Equiv.swap_apply_of_ne_of_ne (by omega) (by omega)]
          exact hpre t hti
        · rw [show t = i by omega, Equiv.swap_apply_left]
          exact hin
    have hsum : (∑ r ∈ range (n - i), q r) = ((n - i - B.card : ℕ) : ℝ≥0∞) * E' := by
      simp only [q]
      rw [Finset.sum_ite, Finset.sum_const_zero, zero_add, Finset.sum_const, nsmul_eq_mul]
      congr 1
      have hc := Finset.card_filter_add_card_filter_not (s := range (n - i)) (fun r => π (i + r) ∈ B)
      rw [card_pool_mem n i (by omega) B hB π hπ hpre, Finset.card_range] at hc
      exact_mod_cast (show ((range (n - i)).filter fun r => ¬ π (i + r) ∈ B).card = n - i - B.card by omega)
    calc prCoin (fun h : Fin ℓ → Fin 256 =>
            ∃ π', loopL n F (List.range' i (c + 1)) π (bsOf h) = some π' ∧ ∀ t < k, π' t ∉ B)
        = prCoin (fun h : Fin ℓ → Fin 256 => ∃ r rest, uniformL (n - i) F (bsOf h) = some (r, rest) ∧ Q r rest) := by
          congr 1; funext h; exact propext (hev _)
      _ ≤ (∑ r ∈ range (n - i), q r) / ((n - i : ℕ) : ℝ≥0∞) := uniformL_le (n - i) hm Q q hQ F ℓ
      _ = ((n - i - B.card : ℕ) : ℝ≥0∞) * (((n - i - 1 - B.card).choose c : ℝ≥0∞) /
            ((n - i - 1).choose c : ℝ≥0∞)) / ((n - i : ℕ) : ℝ≥0∞) := by
          rw [hsum, hE', Nat.sub_sub n i 1]
      _ = ((n - i - B.card).choose (c + 1) : ℝ≥0∞) / ((n - i).choose (c + 1) : ℝ≥0∞) :=
          choose_step_ratio (n - i) B.card c (by omega)

end FlockSoundness.Audit.DrawExec

namespace FlockSoundness.Audit.Law

open Game Finset
open scoped ENNReal

/-- **The executable draw escapes no more often than the model's.** Its probability space is a uniform stream: that
the verifier's bytes are uniform is A3 (`Assumptions.UniformRandomBytes`, `uniform/io-getrandombytes`), which
`drawOS_subset_escape_le` (`ExecDrawOS.lean`) takes for the live draw, reruns included. -/
theorem subset_exec_escape_le (n k L : ℕ) (hk : k ≤ n) (B : Finset (Fin n)) :
    prCoin (fun f : Fin L → Fin 256 =>
        ∃ D, Flock.Draw.subset k n (Law.streamOf f) = some D ∧ ∀ u ∈ B, u.val ∉ D) ≤
      (subset n k hk).escape B := by
  classical
  set B' : Finset ℕ := B.image Fin.val with hB'
  have hB'n : ∀ u ∈ B', u < n := fun u hu => by obtain ⟨v, _, rfl⟩ := mem_image.1 hu; exact v.isLt
  have hcard : B'.card = B.card := card_image_of_injective _ Fin.val_injective
  calc prCoin (fun f : Fin L → Fin 256 =>
          ∃ D, Flock.Draw.subset k n (Law.streamOf f) = some D ∧ ∀ u ∈ B, u.val ∉ D)
      ≤ prCoin (fun f : Fin L → Fin 256 => ∃ π', DrawExec.loopL n (L + 1) (List.range' 0 k) (Equiv.refl ℕ)
          (DrawExec.bsOf f) = some π' ∧ ∀ t < k, π' t ∉ B') := by
        refine prCoin_mono fun f ⟨D, hD, hmiss⟩ => ?_
        obtain ⟨π, hπ, hπB⟩ := DrawExec.subset_misses k n (Law.streamOf f) B' D hD (fun u hu hmem => by
          obtain ⟨v, hv, rfl⟩ := mem_image.1 hu
          exact hmiss v hv hmem)
        rw [DrawExec.loopP_streamOf f n _ _ 0 (Nat.zero_le _), List.drop_zero] at hπ
        exact ⟨π, hπ, hπB⟩
    _ ≤ ((n - 0 - B'.card).choose k : ℝ≥0∞) / ((n - 0).choose k : ℝ≥0∞) :=
        DrawExec.loopL_le n k (L + 1) B' hB'n k 0 (Equiv.refl ℕ) (by omega) hk (fun t ht => ht)
          (fun t ht => absurd ht (Nat.not_lt_zero t)) L
    _ = (subset n k hk).escape B := by rw [subset_escape, hcard, Nat.sub_zero]

end FlockSoundness.Audit.Law

