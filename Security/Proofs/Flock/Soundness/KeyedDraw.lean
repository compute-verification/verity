import Proofs.Flock.Soundness.ExecDrawOS
import Randomness
import Proofs.Flock.Soundness.PlanDraw
import Proofs.Flock.Soundness.Audit.Window
import Proofs.Flock.Soundness.Audit.RegDraw

/-!
# The PoUW circuit's keyed draw escapes no more often than the model's

`Key.subset` (`Randomness.subset`) draws each swap `i` of a `k`-subset from its own stream, index `("s", *index, i)`,
where `Flock.Draw.subset` reads one stream. So its escape needs its own proof, on the same pieces as #408's: the swap map
read as a permutation (`getD_swapAt`), the rejection sampler (`uniformL_le`) and the pool (`card_pool_mem`).

* `subsetOn_eq`, `subsetOn_misses`: the per-swap loop as a list recursion, and its `HashMap` read through a permutation.
* `loopK_le`: swap `i` on its own uniform stream draws uniformly over the pool `π '' [i, n)`, so the loop escapes `B` with
  at most `C(n − i − |B|, c)/C(n − i, c)`.
* `Law.subsetOn_escape_le`: over independent uniform streams, one per swap, `Key.subset` misses `B` with at most the
  uniform `k`-subset's escape.

The window (`PlanDraw.window`, #364's `plan.draw` for each call under one key, admission refusing a repeated call index):

* `drawAll_eq`, `drawK_eq_drawOn`: the calls' draws are one draw over the window's flat strata (`flat`), and on the key
  it is `drawOn` on the key's streams, stratum `p`'s swap `i` at index `("s", *context, i)`.
* `Law.drawOn_escape_le`: **C per-call draws form one stratified law**. Over independent uniform streams, one per stratum
  and swap, the window draw misses `B` with at most `(Law.stratified σ k hk).escape B`, for strata that describe `σ`'s
  (`PlanStrata`).
* `Law.keyedWindow`, `Law.keyedWindow_escape_le`: **tier 3**. The live window draw, on the key derived from a verifier
  secret of `S` bytes, escapes `B` with at most `(Law.stratified σ k hk).escape B + η`, under A4
  (`Assumptions.KeyedStreamsUniform`) for the test `windowTest` and swap indices with distinct framings. As in #416, it
  holds for every number of bytes `L` read per swap, and the unbounded Python stream is the limit.
* `Randomness.frames_inj'`, `Law.windowIdx_inj`, `Law.keyedWindow_escape_le_of_names`: a frame parses uniquely, so the
  swap indices' framings are distinct when the calls are admitted and each call's stratum names are distinct. So tier 3
  holds from the draw's own checks, with every swap index framed in under `2^64` bytes.
* `Law.keyedWindow_audit_of_record`, `Law.keyedWindow_extraction_audit_of_record`: **step 4**, the window pins on the live
  keyed draw at the sizing of record: `2^-40 + η + ε_ks + δ_link`, on a fresh uniform window secret (A5).
* `Law.keyedWindowReg_extraction_audit_of_record` (the claim of record), `Law.keyedWindowReg_audit_of_record`: the same on the
  real game, whose key context is the receipt digest (#423's C1), strategy by strategy (`prob_auditReg`).
-/

namespace FlockSoundness.Audit.KeyedDraw

open Game Finset DrawExec
open scoped ENNReal

/-- `Key.subset`'s loop as a list recursion, swap `i` drawing from `S i`. -/
def loopSwK (n : ℕ) (S : ℕ → ByteArray) : List ℕ → Std.HashMap ℕ ℕ → Option (Std.HashMap ℕ ℕ)
  | [], sw => some sw
  | i :: is, sw => match Randomness.uniformOn (n - i) (S i) with
    | none => none
    | some r => loopSwK n S is (swapAt sw i r)

abbrev StK := Option (Option (Array ℕ)) × Std.HashMap ℕ ℕ

theorem forIn_charK (n : ℕ) (S : ℕ → ByteArray) (G : Std.HashMap ℕ ℕ → Array ℕ)
    (F : ℕ → StK → Id (ForInStep StK)) (POST : StK → Id (Option (Array ℕ)))
    (hFn : ∀ i s, Randomness.uniformOn (n - i) (S i) = none → F i s = pure (.done (some none, s.2)))
    (hFs : ∀ i s r, Randomness.uniformOn (n - i) (S i) = some r → F i s = pure (.yield (none, swapAt s.2 i r)))
    (hPs : ∀ s r, s.1 = some r → POST s = pure r) (hPn : ∀ s, s.1 = none → POST s = pure (some (G s.2))) :
    ∀ (l : List ℕ) (sw : Std.HashMap ℕ ℕ),
      (forIn (m := Id) l ((none : Option (Option (Array ℕ))), sw) F >>= POST).run = (loopSwK n S l sw).map G := by
  intro l
  induction l with
  | nil =>
    intro sw
    simp only [List.forIn_nil, loopSwK, Option.map_some]
    rw [pure_bind, hPn _ rfl]
    rfl
  | cons i is ih =>
    intro sw
    rw [List.forIn_cons]
    cases h : Randomness.uniformOn (n - i) (S i) with
    | none =>
      rw [hFn i _ h]
      simp only [loopSwK, h, Option.map_none, pure_bind]
      rw [hPs _ none rfl]
      rfl
    | some r =>
      rw [hFs i _ r h]
      simp only [loopSwK, h, pure_bind]
      exact ih _

theorem subsetOn_eq (k n : ℕ) (S : ℕ → ByteArray) :
    Randomness.subsetOn k n S = (loopSwK n S (List.range' 0 k) ∅).map
      (fun sw => ((Array.range k).map fun i => sw.getD i i).qsort (· < ·)) := by
  unfold Randomness.subsetOn
  dsimp only
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  have hsize : ([:k] : Std.Legacy.Range).size = k := by
    show (k - 0 + 1 - 1) / 1 = k
    omega
  rw [hsize]
  refine forIn_charK n S _ _ _ ?_ ?_ ?_ ?_ (List.range' 0 k) ∅
  · intro i s h; simp only [h]
  · intro i s r h; simp only [h]; rfl
  · intro s r h; simp only [h]
  · intro s h; simp only [h]

/-- The per-swap loop on the permutation `t ↦ sw.getD t t`. -/
def loopPK (n : ℕ) (S : ℕ → ByteArray) : List ℕ → Equiv.Perm ℕ → Option (Equiv.Perm ℕ)
  | [], π => some π
  | i :: is, π => match Randomness.uniformOn (n - i) (S i) with
    | none => none
    | some r => loopPK n S is ((Equiv.swap i (i + r)).trans π)

theorem loopSwK_loopPK (n : ℕ) (S : ℕ → ByteArray) : ∀ (l : List ℕ) (sw : Std.HashMap ℕ ℕ) (π : Equiv.Perm ℕ),
    (∀ t, sw.getD t t = π t) →
      ∀ sw', loopSwK n S l sw = some sw' → ∃ π', loopPK n S l π = some π' ∧ ∀ t, sw'.getD t t = π' t := by
  intro l
  induction l with
  | nil => intro sw π h sw' hs; simp only [loopSwK, Option.some.injEq] at hs; subst hs; exact ⟨π, rfl, h⟩
  | cons i is ih =>
    intro sw π h sw' hs
    simp only [loopSwK] at hs
    simp only [loopPK]
    cases hu : Randomness.uniformOn (n - i) (S i) with
    | none => rw [hu] at hs; exact absurd hs (by simp)
    | some r =>
      rw [hu] at hs
      exact ih _ _ (getD_swapAt sw π h i r) sw' hs

/-- **The per-swap sampler's output misses `B` only if its permutation does on `[0, k)`.** -/
theorem subsetOn_misses (k n : ℕ) (S : ℕ → ByteArray) (B : Finset ℕ) (D : Array ℕ)
    (hD : Randomness.subsetOn k n S = some D) (hB : ∀ u ∈ B, u ∉ D) :
    ∃ π, loopPK n S (List.range' 0 k) (Equiv.refl ℕ) = some π ∧ ∀ i < k, π i ∉ B := by
  rw [subsetOn_eq] at hD
  cases hl : loopSwK n S (List.range' 0 k) ∅ with
  | none => rw [hl] at hD; exact absurd hD (by simp)
  | some sw =>
    rw [hl, Option.map_some, Option.some.injEq] at hD
    obtain ⟨π, hπ, hsw⟩ := loopSwK_loopPK n S _ ∅ (Equiv.refl ℕ) (fun t => by simp) sw hl
    refine ⟨π, hπ, fun i hi hmem => hB _ hmem ?_⟩
    rw [← hD, mem_qsort, Array.mem_map]
    exact ⟨i, Array.mem_range.2 hi, hsw i⟩

/-- `uniformOn` on a stream of `L` bytes is `uniformL` on its bytes, with `L + 1` fuel. -/
theorem uniformOn_streamOf {L : ℕ} (g : Fin L → Fin 256) (m : ℕ) :
    Randomness.uniformOn m (Law.streamOf g) = (uniformL m (L + 1) (bsOf g)).map Prod.fst := by
  have h := uniform_streamOf g m (L + 1) 0 (Nat.zero_le _)
  rw [List.drop_zero] at h
  unfold Randomness.uniformOn
  rw [size_streamOf]
  revert h
  cases uniformL m (L + 1) (bsOf g) with
  | none => intro h; simp only at h; rw [h]; rfl
  | some rr =>
    obtain ⟨r, rest⟩ := rr
    intro h
    obtain ⟨p, _, _, hU⟩ := h
    rw [hU]; rfl

/-- The per-swap loop on byte lists, swap `i` reading `S i` from its start. -/
def loopK (n F : ℕ) (S : ℕ → List UInt8) : List ℕ → Equiv.Perm ℕ → Option (Equiv.Perm ℕ)
  | [], π => some π
  | i :: is, π => match uniformL (n - i) F (S i) with
    | none => none
    | some (r, _) => loopK n F S is ((Equiv.swap i (i + r)).trans π)

theorem loopK_congr (n F : ℕ) (S S' : ℕ → List UInt8) : ∀ (l : List ℕ) (π : Equiv.Perm ℕ),
    (∀ j ∈ l, S j = S' j) → loopK n F S l π = loopK n F S' l π := by
  intro l
  induction l with
  | nil => intro π _; rfl
  | cons i is ih =>
    intro π h
    simp only [loopK, h i List.mem_cons_self]
    cases uniformL (n - i) F (S' i) with
    | none => rfl
    | some rr => exact ih _ fun j hj => h j (List.mem_cons_of_mem _ hj)

theorem loopK_prefix (n F : ℕ) (S : ℕ → List UInt8) : ∀ (c j : ℕ) (π π' : Equiv.Perm ℕ),
    loopK n F S (List.range' j c) π = some π' → ∀ t < j, π' t = π t := by
  intro c
  induction c with
  | zero => intro j π π' h t _; simp only [List.range'_zero, loopK, Option.some.injEq] at h; rw [h]
  | succ c ih =>
    intro j π π' h t ht
    rw [List.range'_succ] at h
    simp only [loopK] at h
    split at h
    · exact absurd h (by simp)
    · rename_i r rest _
      rw [ih (j + 1) _ π' h t (by omega), Equiv.trans_apply,
        Equiv.swap_apply_of_ne_of_ne (by omega) (by omega)]

/-- The swaps' streams from swap `i` on: swap `i + j` reads `G j`. -/
def swapsOf {c L : ℕ} (i : ℕ) (G : Fin c → Fin L → Fin 256) : ℕ → List UInt8 := fun j =>
  if h : i ≤ j ∧ j - i < c then bsOf (G ⟨j - i, h.2⟩) else []

/-- Splitting off the first of `c + 1` streams. -/
def consEquiv (c : ℕ) (α : Type) : α × (Fin c → α) ≃ (Fin (c + 1) → α) where
  toFun p := Fin.cons p.1 p.2
  invFun G := (G 0, fun j => G j.succ)
  left_inv p := by obtain ⟨x, G⟩ := p; simp
  right_inv G := by funext j; exact Fin.cons_self_tail G ▸ rfl

theorem avg_div_sum {C : Type} [Fintype C] (R : Finset ℕ) (f : ℕ → C → ℝ≥0∞) (m : ℝ≥0∞) :
    avg (fun c => (∑ r ∈ R, f r c) / m) = (∑ r ∈ R, avg (f r)) / m := by
  unfold avg
  simp only [div_eq_mul_inv, Finset.sum_mul, Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun r _ => Finset.sum_congr rfl fun c _ => ?_
  ring

open Classical in
/-- `1` if `p` holds, else `0`: `prCoin`'s indicator. -/
noncomputable def ind (p : Prop) : ℝ≥0∞ := if p then 1 else 0

theorem prCoin_eq_avg_ind {C : Type} [Fintype C] (p : C → Prop) : prCoin p = avg fun c => ind (p c) := rfl

theorem prCoin_const_le {C : Type} [Fintype C] (p : Prop) : prCoin (fun _ : C => p) ≤ ind p := by
  classical
  by_cases hp : p
  · simp only [ind, hp, ite_true]; exact prCoin_le_one _
  · simp only [ind, hp, ite_false]; exact (prCoin_false).le

open Classical in
/-- **The per-swap loop's escape**: from swap `i`, each swap on its own uniform stream, the rest of the loop returns a
permutation with no value below `k` in `B` with probability at most `C(n − i − |B|, c)/C(n − i, c)`. -/
theorem loopK_le (n k L : ℕ) (B : Finset ℕ) (hB : ∀ u ∈ B, u < n) : ∀ (c i : ℕ) (π : Equiv.Perm ℕ),
    i + c = k → k ≤ n → (∀ t < n, π t < n) → (∀ t < i, π t ∉ B) →
      prCoin (fun G : Fin c → Fin L → Fin 256 =>
          ∃ π', loopK n (L + 1) (swapsOf i G) (List.range' i c) π = some π' ∧ ∀ t < k, π' t ∉ B) ≤
        ((n - i - B.card).choose c : ℝ≥0∞) / ((n - i).choose c : ℝ≥0∞) := by
  intro c
  induction c with
  | zero =>
    intro i π _ _ _ _
    simp only [Nat.choose_zero_right, Nat.cast_one, div_one]
    exact prCoin_le_one _
  | succ c ih =>
    intro i π hk hkn hπ hpre
    have hm : 0 < n - i := by omega
    set E' := ((n - (i + 1) - B.card).choose c : ℝ≥0∞) / ((n - (i + 1)).choose c : ℝ≥0∞) with hE'
    let P : ℕ → (Fin c → Fin L → Fin 256) → Prop := fun r G' =>
      ∃ π', loopK n (L + 1) (swapsOf (i + 1) G') (List.range' (i + 1) c) ((Equiv.swap i (i + r)).trans π) = some π' ∧
        ∀ t < k, π' t ∉ B
    let q : ℕ → ℝ≥0∞ := fun r => if π (i + r) ∈ B then 0 else E'
    have hev : ∀ (x : Fin L → Fin 256) (G' : Fin c → Fin L → Fin 256),
        (∃ π', loopK n (L + 1) (swapsOf i (Fin.cons x G')) (List.range' i (c + 1)) π = some π' ∧ ∀ t < k, π' t ∉ B) ↔
          ∃ r rest, uniformL (n - i) (L + 1) (bsOf x) = some (r, rest) ∧ P r G' := by
      intro x G'
      have h0 : swapsOf i (Fin.cons x G' : Fin (c + 1) → Fin L → Fin 256) i = bsOf x := by
        simp [swapsOf]
      have hrest : ∀ π₀, loopK n (L + 1) (swapsOf i (Fin.cons x G' : Fin (c + 1) → Fin L → Fin 256))
          (List.range' (i + 1) c) π₀ = loopK n (L + 1) (swapsOf (i + 1) G') (List.range' (i + 1) c) π₀ := by
        intro π₀
        refine loopK_congr _ _ _ _ _ _ fun j hj => ?_
        rw [List.mem_range'_1] at hj
        have h1 : i ≤ j ∧ j - i < c + 1 := ⟨by omega, by omega⟩
        have h2 : i + 1 ≤ j ∧ j - (i + 1) < c := ⟨by omega, by omega⟩
        simp only [swapsOf, h1, h2, and_self, dite_true]
        congr 1
        have : (⟨j - i, h1.2⟩ : Fin (c + 1)) = (⟨j - (i + 1), h2.2⟩ : Fin c).succ := by
          ext; simp; omega
        rw [this, Fin.cons_succ]
      rw [List.range'_succ]
      simp only [loopK, h0]
      constructor
      · rintro ⟨π', h, hB'⟩
        split at h
        · exact absurd h (by simp)
        · rename_i r rest hu
          exact ⟨r, rest, hu, π', (hrest _).symm.trans h, hB'⟩
      · rintro ⟨r, rest, hu, π', h, hB'⟩
        refine ⟨π', ?_, hB'⟩
        rw [hu]
        exact (hrest _).trans h
    have hP : ∀ r < n - i, prCoin (P r) ≤ q r := by
      intro r hr
      by_cases hin : π (i + r) ∈ B
      · simp only [q, hin, ite_true]
        refine (prCoin_mono fun G' hP' => ?_).trans (prCoin_false).le
        obtain ⟨π', h', hB'⟩ := hP'
        have := loopK_prefix n (L + 1) _ c (i + 1) _ π' h' i (by omega)
        rw [Equiv.trans_apply, Equiv.swap_apply_left] at this
        exact hB' i (by omega) (this ▸ hin)
      · simp only [q, hin, ite_false]
        refine ih (i + 1) _ (by omega) hkn (swap_trans_lt (by omega) π hπ) (fun t ht => ?_)
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
    calc prCoin (fun G : Fin (c + 1) → Fin L → Fin 256 =>
            ∃ π', loopK n (L + 1) (swapsOf i G) (List.range' i (c + 1)) π = some π' ∧ ∀ t < k, π' t ∉ B)
        = avg fun G' : Fin c → Fin L → Fin 256 => prCoin fun x : Fin L → Fin 256 =>
            ∃ r rest, uniformL (n - i) (L + 1) (bsOf x) = some (r, rest) ∧ P r G' := by
          rw [prCoin_eq_avg_ind, ← avg_equiv (consEquiv c (Fin L → Fin 256)), avg_prod, avg_comm]
          refine congrArg avg (funext fun G' => ?_)
          rw [prCoin_eq_avg_ind]
          refine congrArg avg (funext fun x => ?_)
          simp only [consEquiv, Equiv.coe_fn_mk]
          exact congrArg ind (propext (hev x G'))
      _ ≤ avg fun G' : Fin c → Fin L → Fin 256 =>
            (∑ r ∈ range (n - i), (fun r G' => ind (P r G')) r G') / ((n - i : ℕ) : ℝ≥0∞) := by
          refine avg_mono fun G' => ?_
          exact uniformL_le (n - i) hm (fun r _ => P r G') (fun r => ind (P r G'))
            (fun r _ ℓ' => prCoin_const_le _) (L + 1) L
      _ = (∑ r ∈ range (n - i), prCoin (P r)) / ((n - i : ℕ) : ℝ≥0∞) := by
          rw [avg_div_sum]
          rfl
      _ ≤ (∑ r ∈ range (n - i), q r) / ((n - i : ℕ) : ℝ≥0∞) := by
          gcongr with r hr
          exact hP r (Finset.mem_range.1 hr)
      _ = ((n - i - B.card : ℕ) : ℝ≥0∞) * (((n - i - 1 - B.card).choose c : ℝ≥0∞) /
            ((n - i - 1).choose c : ℝ≥0∞)) / ((n - i : ℕ) : ℝ≥0∞) := by
          rw [hsum, hE', Nat.sub_sub n i 1]
      _ = ((n - i - B.card).choose (c + 1) : ℝ≥0∞) / ((n - i).choose (c + 1) : ℝ≥0∞) :=
          choose_step_ratio (n - i) B.card c (by omega)


/-- Per-swap streams, swap `i` reading `G i`. -/
def streamsOf {k L : ℕ} (G : Fin k → Fin L → Fin 256) : ℕ → ByteArray := fun i =>
  if h : i < k then Law.streamOf (G ⟨i, h⟩) else ByteArray.empty

theorem loopPK_streamsOf {k L : ℕ} (G : Fin k → Fin L → Fin 256) (n : ℕ) : ∀ (l : List ℕ) (π : Equiv.Perm ℕ),
    (∀ j ∈ l, j < k) → loopPK n (streamsOf G) l π = loopK n (L + 1) (swapsOf 0 G) l π := by
  intro l
  induction l with
  | nil => intro π _; rfl
  | cons i is ih =>
    intro π hl
    have hi : i < k := hl i List.mem_cons_self
    have hS : streamsOf G i = Law.streamOf (G ⟨i, hi⟩) := by simp [streamsOf, hi]
    have hW : swapsOf 0 G i = bsOf (G ⟨i, hi⟩) := by simp [swapsOf, hi]
    simp only [loopPK, loopK, hS, hW, uniformOn_streamOf]
    cases uniformL (n - i) (L + 1) (bsOf (G ⟨i, hi⟩)) with
    | none => rfl
    | some rr =>
      obtain ⟨r, rest⟩ := rr
      exact ih _ fun j hj => hl j (List.mem_cons_of_mem _ hj)

end FlockSoundness.Audit.KeyedDraw

namespace FlockSoundness.Audit.Law

open Game Finset
open scoped ENNReal

/-- **`Key.subset` escapes no more often than the uniform `k`-subset**: over independent uniform streams, one per swap,
the spec's per-swap sampler returns a draw missing `B` with probability at most `C(n − |B|, k)/C(n, k)`. -/
theorem subsetOn_escape_le (n k L : ℕ) (hk : k ≤ n) (B : Finset (Fin n)) :
    prCoin (fun G : Fin k → Fin L → Fin 256 =>
        ∃ D, Randomness.subsetOn k n (KeyedDraw.streamsOf G) = some D ∧ ∀ u ∈ B, u.val ∉ D) ≤
      (subset n k hk).escape B := by
  classical
  set B' : Finset ℕ := B.image Fin.val with hB'
  have hB'n : ∀ u ∈ B', u < n := fun u hu => by obtain ⟨v, _, rfl⟩ := mem_image.1 hu; exact v.isLt
  have hcard : B'.card = B.card := card_image_of_injective _ Fin.val_injective
  calc prCoin (fun G : Fin k → Fin L → Fin 256 =>
          ∃ D, Randomness.subsetOn k n (KeyedDraw.streamsOf G) = some D ∧ ∀ u ∈ B, u.val ∉ D)
      ≤ prCoin (fun G : Fin k → Fin L → Fin 256 => ∃ π', KeyedDraw.loopK n (L + 1) (KeyedDraw.swapsOf 0 G)
          (List.range' 0 k) (Equiv.refl ℕ) = some π' ∧ ∀ t < k, π' t ∉ B') := by
        refine prCoin_mono fun G ⟨D, hD, hmiss⟩ => ?_
        obtain ⟨π, hπ, hπB⟩ := KeyedDraw.subsetOn_misses k n _ B' D hD (fun u hu hmem => by
          obtain ⟨v, hv, rfl⟩ := mem_image.1 hu
          exact hmiss v hv hmem)
        rw [KeyedDraw.loopPK_streamsOf G n _ _ (fun j hj => by rw [List.mem_range'_1] at hj; omega)] at hπ
        exact ⟨π, hπ, hπB⟩
    _ ≤ ((n - 0 - B'.card).choose k : ℝ≥0∞) / ((n - 0).choose k : ℝ≥0∞) :=
        KeyedDraw.loopK_le n k L B' hB'n k 0 (Equiv.refl ℕ) (by omega) hk (fun t ht => ht)
          (fun t ht => absurd ht (Nat.not_lt_zero t))
    _ = (subset n k hk).escape B := by rw [subset_escape, hcard, Nat.sub_zero]

end FlockSoundness.Audit.Law

namespace FlockSoundness.Audit.KeyedDraw

open Game Finset DrawExec Randomness
open scoped ENNReal

/-- The window's strata, in draw order: each call's layout strata, with the call's index. -/
def flat (calls : List PlanDraw.Call) : List (List Part × PlanDraw.Stratum) :=
  calls.flatMap fun c => c.strata.map fun s => (c.index, s)

/-- The window draw on the key, over the flat strata. -/
def drawK (key : ByteArray) (L : ℕ) : List (List Part × PlanDraw.Stratum) → Option (List ℕ)
  | [] => some []
  | (call, s) :: rest => do
    let loc ← subset key s.k s.units.length (PlanDraw.context call s.name) L
    let r ← drawK key L rest
    pure (loc.toList.map (fun i => s.units.getD i 0) ++ r)

theorem drawn_eq (key : ByteArray) (L : ℕ) (call : List Part) : ∀ (ss : List PlanDraw.Stratum)
    (rest : List (List Part × PlanDraw.Stratum)),
    (do let d ← PlanDraw.drawn key call ss L; let r ← drawK key L rest; pure (d ++ r)) =
      drawK key L (ss.map (fun s => (call, s)) ++ rest) := by
  intro ss
  induction ss with
  | nil => intro rest; simp [PlanDraw.drawn]
  | cons s ss ih =>
    intro rest
    simp only [PlanDraw.drawn, List.map_cons, List.cons_append, drawK]
    cases subset key s.k s.units.length (PlanDraw.context call s.name) L with
    | none => rfl
    | some loc =>
      simp only [Option.bind_eq_bind, Option.bind_some, ← ih rest]
      cases PlanDraw.drawn key call ss L with
      | none => rfl
      | some d =>
        simp only [Option.bind_some]
        cases drawK key L rest with
        | none => rfl
        | some r => simp [List.append_assoc]

/-- **The calls' draws are one draw over the window's strata.** -/
theorem drawAll_eq (key : ByteArray) (L : ℕ) : ∀ calls : List PlanDraw.Call,
    PlanDraw.drawAll key calls L = drawK key L (flat calls) := by
  intro calls
  induction calls with
  | nil => rfl
  | cons c cs ih =>
    simp only [PlanDraw.drawAll, flat, List.flatMap_cons]
    rw [← drawn_eq key L c.index c.strata]
    rw [ih]
    rfl

/-- The window draw on given streams, stratum at position `p` drawing swap `i` from `T p i`. -/
def drawOn (T : ℕ → ℕ → ByteArray) : List (List Part × PlanDraw.Stratum) → ℕ → Option (List ℕ)
  | [], _ => some []
  | (_, s) :: rest, p => do
    let loc ← subsetOn s.k s.units.length (T p)
    let r ← drawOn T rest (p + 1)
    pure (loc.toList.map (fun i => s.units.getD i 0) ++ r)

theorem subsetOn_congr (k n : ℕ) (S S' : ℕ → ByteArray) (h : ∀ i < k, S i = S' i) :
    subsetOn k n S = subsetOn k n S' := by
  rw [subsetOn_eq, subsetOn_eq]
  congr 1
  suffices ∀ (l : List ℕ) (sw : Std.HashMap ℕ ℕ), (∀ i ∈ l, i < k) → loopSwK n S l sw = loopSwK n S' l sw from
    this _ _ fun i hi => by rw [List.mem_range'_1] at hi; omega
  intro l
  induction l with
  | nil => intro sw _; rfl
  | cons i is ih =>
    intro sw hl
    simp only [loopSwK, h i (hl i List.mem_cons_self)]
    cases uniformOn (n - i) (S' i) with
    | none => rfl
    | some r => exact ih _ fun j hj => hl j (List.mem_cons_of_mem _ hj)

theorem drawK_eq_drawOn (key : ByteArray) (L : ℕ) (T : ℕ → ℕ → ByteArray) :
    ∀ (WS : List (List Part × PlanDraw.Stratum)) (p : ℕ),
      (∀ j (h : j < WS.length), ∀ i < WS[j].2.k,
        T (p + j) i = stream key (swapIndex (PlanDraw.context WS[j].1 WS[j].2.name) i) L) →
      drawK key L WS = drawOn T WS p := by
  intro WS
  induction WS with
  | nil => intro p _; rfl
  | cons x rest ih =>
    intro p hT
    obtain ⟨call, s⟩ := x
    simp only [drawK, drawOn]
    have h0 : subset key s.k s.units.length (PlanDraw.context call s.name) L = subsetOn s.k s.units.length (T p) := by
      unfold subset
      exact subsetOn_congr _ _ _ _ fun i hi => (hT 0 (by simp) i hi).symm
    rw [h0, ih (p + 1) fun j hj i hi => by
      have := hT (j + 1) (by simp; omega) i hi
      rw [show p + (j + 1) = p + 1 + j by omega] at this
      simpa using this]

/-- **What a successful window draw holds**: every stratum's `Key.subset` succeeded, and each of its local indices'
units is in the draw. -/
theorem drawOn_units (T : ℕ → ℕ → ByteArray) : ∀ (WS : List (List Part × PlanDraw.Stratum)) (p : ℕ) (D : List ℕ),
    drawOn T WS p = some D → ∀ j (h : j < WS.length), ∃ loc, subsetOn WS[j].2.k WS[j].2.units.length (T (p + j)) = some loc ∧
      ∀ i ∈ loc, WS[j].2.units.getD i 0 ∈ D := by
  intro WS
  induction WS with
  | nil => intro p D _ j h; simp at h
  | cons x rest ih =>
    intro p D hD j hj
    obtain ⟨call, s⟩ := x
    simp only [drawOn] at hD
    cases hl : subsetOn s.k s.units.length (T p) with
    | none => rw [hl] at hD; exact absurd hD (by simp)
    | some loc =>
      rw [hl] at hD
      simp only [Option.bind_eq_bind, Option.bind_some] at hD
      cases hr : drawOn T rest (p + 1) with
      | none => rw [hr] at hD; exact absurd hD (by simp)
      | some r =>
        rw [hr] at hD
        simp only [Option.bind_some, Option.pure_def, Option.some.injEq] at hD
        subst hD
        rcases j with _ | j
        · refine ⟨loc, by simpa using hl, fun i hi => List.mem_append_left _ ?_⟩
          exact List.mem_map.2 ⟨i, Array.mem_toList_iff.2 hi, rfl⟩
        · obtain ⟨loc', h1, h2⟩ := ih (p + 1) r hr j (by simp at hj; omega)
          refine ⟨loc', by rw [show p + (j + 1) = p + 1 + j by omega]; simpa using h1, fun i hi =>
            List.mem_append_right _ (by simpa using h2 i hi)⟩


end FlockSoundness.Audit.KeyedDraw

namespace FlockSoundness.Audit.Law

open Game Finset DrawExec Randomness
open scoped ENNReal

variable {n M : ℕ}

/-- The window's layout strata describe `σ`'s: stratum `p` draws `k p` of its units, has `σ`'s stratum's size, and its
units, in order, run over `σ`'s stratum `p`. -/
def PlanStrata (σ : Fin n → Fin M) (k : Fin M → ℕ) (WS : List (List Part × PlanDraw.Stratum)) : Prop :=
  WS.length = M ∧ ∀ (p : Fin M) (h : p.val < WS.length), WS[p.val].2.k = k p ∧
    WS[p.val].2.units.length = (stratum σ p).card ∧
    (range WS[p.val].2.units.length).image (fun j => WS[p.val].2.units.getD j 0) = (stratum σ p).map Fin.valEmbedding

/-- Per-stratum, per-swap streams `F p i`, as the window draw reads them. -/
def windowStreams {k : Fin M → ℕ} {L : ℕ} (F : (p : Fin M) → Fin (k p) → Fin L → Fin 256) : ℕ → ℕ → ByteArray :=
  fun p => if h : p < M then KeyedDraw.streamsOf (F ⟨p, h⟩) else fun _ => ByteArray.empty

/-- **The window draw escapes no more often than `Law.stratified`**: over independent uniform streams, one per stratum
and swap, it returns a draw missing `B` with at most the stratified law's escape. -/
theorem drawOn_escape_le (σ : Fin n → Fin M) (k : Fin M → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card)
    (WS : List (List Part × PlanDraw.Stratum)) (hWS : PlanStrata σ k WS) (L : ℕ) (B : Finset (Fin n)) :
    prCoin (fun F : (p : Fin M) → Fin (k p) → Fin L → Fin 256 =>
        ∃ D, KeyedDraw.drawOn (windowStreams F) WS 0 = some D ∧ ∀ u ∈ B, u.val ∉ D) ≤
      (stratified σ k hk).escape B := by
  classical
  obtain ⟨hlen, hW⟩ := hWS
  have hp : ∀ p : Fin M, p.val < WS.length := fun p => hlen ▸ p.isLt
  let U : Fin M → List ℕ := fun p => WS[p.val]'(hp p) |>.2.units
  let Bl : (p : Fin M) → Finset (Fin (stratum σ p).card) := fun p =>
    univ.filter fun j => ∃ u ∈ B, (U p).getD j.val 0 = u.val
  calc prCoin (fun F : (p : Fin M) → Fin (k p) → Fin L → Fin 256 =>
          ∃ D, KeyedDraw.drawOn (windowStreams F) WS 0 = some D ∧ ∀ u ∈ B, u.val ∉ D)
      ≤ prCoin (fun F : (p : Fin M) → Fin (k p) → Fin L → Fin 256 => ∀ p ∈ (univ : Finset (Fin M)),
          (fun p (G : Fin (k p) → Fin L → Fin 256) => ∃ D, subsetOn (k p) (stratum σ p).card (KeyedDraw.streamsOf G) = some D ∧
            ∀ j ∈ Bl p, j.val ∉ D) p (F p)) := by
        refine prCoin_mono fun F ⟨D, hD, hmiss⟩ p _ => ?_
        obtain ⟨loc, hloc, hmem⟩ := KeyedDraw.drawOn_units _ WS 0 D hD p.val (hp p)
        obtain ⟨hk', hn', _⟩ := hW p (hp p)
        have hT : windowStreams F p.val = KeyedDraw.streamsOf (F p) := by
          simp [windowStreams, p.isLt]
        rw [Nat.zero_add, hT, hk', hn'] at hloc
        refine ⟨loc, hloc, fun j hj hjl => ?_⟩
        obtain ⟨u, hu, hue⟩ := (Finset.mem_filter.1 hj).2
        exact hmiss u hu (hue ▸ hmem _ hjl)
    _ = ∏ p, prCoin (fun G : Fin (k p) → Fin L → Fin 256 =>
          ∃ D, subsetOn (k p) (stratum σ p).card (KeyedDraw.streamsOf G) = some D ∧ ∀ j ∈ Bl p, j.val ∉ D) :=
        prCoin_pi_forall (Ω := fun p => Fin (k p) → Fin L → Fin 256) univ (fun p (G : Fin (k p) → Fin L → Fin 256) =>
          ∃ D, subsetOn (k p) (stratum σ p).card (KeyedDraw.streamsOf G) = some D ∧ ∀ j ∈ Bl p, j.val ∉ D)
    _ ≤ ∏ p, (subset (stratum σ p).card (k p) (hk p)).escape (Bl p) :=
        Finset.prod_le_prod fun p _ => subsetOn_escape_le _ _ _ (hk p) (Bl p)
    _ ≤ (stratified σ k hk).escape B := by
        rw [stratified_escape]
        refine Finset.prod_le_prod fun p _ => ?_
        rw [subset_escape]
        have hcard : (B.filter (σ · = p)).card ≤ (Bl p).card := by
          obtain ⟨_, hn', himg⟩ := hW p (hp p)
          have hsub : (B.filter (σ · = p)).map Fin.valEmbedding ⊆ (Bl p).image fun j => (U p).getD j.val 0 := by
            intro x hx
            obtain ⟨u, hu, rfl⟩ := Finset.mem_map.1 hx
            have hus : u.val ∈ (stratum σ p).map Fin.valEmbedding :=
              Finset.mem_map_of_mem _ (mem_stratum.2 (Finset.mem_filter.1 hu).2)
            rw [← himg] at hus
            obtain ⟨j, hj, hje⟩ := Finset.mem_image.1 hus
            have hj' : j < (stratum σ p).card := by rw [← hn']; exact Finset.mem_range.1 hj
            refine Finset.mem_image.2 ⟨⟨j, hj'⟩, Finset.mem_filter.2 ⟨Finset.mem_univ _,
              u, (Finset.mem_filter.1 hu).1, hje⟩, hje⟩
          calc (B.filter (σ · = p)).card = ((B.filter (σ · = p)).map Fin.valEmbedding).card := (Finset.card_map _).symm
            _ ≤ ((Bl p).image fun j => (U p).getD j.val 0).card := Finset.card_le_card hsub
            _ ≤ (Bl p).card := Finset.card_image_le
        have hnum : ((stratum σ p).card - (Bl p).card).choose (k p) ≤
            ((stratum σ p).card - (B.filter (σ · = p)).card).choose (k p) :=
          Nat.choose_le_choose _ (Nat.sub_le_sub_left hcard _)
        exact ENNReal.div_le_div_right (by exact_mod_cast hnum) _


theorem drawOn_congr (T T' : ℕ → ℕ → ByteArray) : ∀ (WS : List (List Part × PlanDraw.Stratum)) (p : ℕ),
    (∀ j (h : j < WS.length), ∀ i < WS[j].2.k, T (p + j) i = T' (p + j) i) →
      KeyedDraw.drawOn T WS p = KeyedDraw.drawOn T' WS p := by
  intro WS
  induction WS with
  | nil => intro p _; rfl
  | cons x rest ih =>
    intro p hT
    obtain ⟨call, s⟩ := x
    simp only [KeyedDraw.drawOn]
    have h0 : subsetOn s.k s.units.length (T p) = subsetOn s.k s.units.length (T' p) :=
      KeyedDraw.subsetOn_congr _ _ _ _ fun i hi => by simpa using hT 0 (by simp) i hi
    rw [h0, ih (p + 1) fun j hj i hi => by
      have := hT (j + 1) (by simp; omega) i hi
      rw [show p + (j + 1) = p + 1 + j by omega] at this
      simpa using this]

/-- The flat strata's default, for positions past the end. -/
def noStratum : List Part × PlanDraw.Stratum := ([], ⟨"", [], 0⟩)

/-- **The window's streams**: swap `i` of stratum `p`, numbered `finSigmaFinEquiv ⟨p, i⟩`, is at index
`("s", *context, i)`, the stratum's `plan.draw` context. -/
def windowIdx (WS : List (List Part × PlanDraw.Stratum)) (k : Fin M → ℕ) (j : Fin (∑ p, k p)) : List Part :=
  let x := (finSigmaFinEquiv (n := k)).symm j
  swapIndex (PlanDraw.context (WS.getD x.1.val noStratum).1 (WS.getD x.1.val noStratum).2.name) x.2.val

/-- The window draw on given streams, numbered as `windowIdx` numbers them. -/
def windowOf (k : Fin M → ℕ) (streams : Fin (∑ p, k p) → ByteArray) : ℕ → ℕ → ByteArray := fun p i =>
  if h : p < M then (if hi : i < k ⟨p, h⟩ then streams (finSigmaFinEquiv ⟨⟨p, h⟩, ⟨i, hi⟩⟩) else ByteArray.empty)
  else ByteArray.empty

/-- **The test A4 is taken for**: the window draw computed from the streams misses `B`. -/
def windowTest (WS : List (List Part × PlanDraw.Stratum)) (k : Fin M → ℕ) (B : Finset (Fin n))
    (streams : Fin (∑ p, k p) → ByteArray) : Prop :=
  ∃ D, KeyedDraw.drawOn (windowOf k streams) WS 0 = some D ∧ ∀ u ∈ B, u.val ∉ D

/-- **The live PoUW window draw as a law**: `PlanDraw.window` on the key derived from the window's secret `sec ω` of
`S` bytes, each swap's stream read to `L` bytes; a refused or short draw draws every unit. -/
def keyedWindow {Ω : Type} [Fintype Ω] [Nonempty Ω] {S : ℕ} (sec : Ω → Fin S → Fin 256) (domain : String)
    (context : List (String × Part)) (calls : List PlanDraw.Call) (L : ℕ) : Law n where
  Ω := Ω
  draw ω := match PlanDraw.window (derive ⟨Array.ofFn fun b => UInt8.ofFin (sec ω b)⟩ domain context) calls L with
    | some D => univ.filter fun u => u.val ∈ D
    | none => univ

/-- A5 is satisfiable: the identity source, `S` uniform bytes themselves. -/
theorem uniformSecret_id (S : ℕ) : Assumptions.UniformSecret (fun ω : Fin S → Fin 256 => ω) := fun _ => rfl

/-- **Tier 3 for the PoUW window draw**: on a uniform window secret (A5), under A4 for the draw's escape test, the live
window draw escapes `B` with at most `Law.stratified`'s escape plus `η`, for every number of bytes `L` read per swap. -/
theorem keyedWindow_escape_le (σ : Fin n → Fin M) (k : Fin M → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card)
    (calls : List PlanDraw.Call) (hWS : PlanStrata σ k (KeyedDraw.flat calls)) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    {S : ℕ} (sec : Ω → Fin S → Fin 256) (hsec : Assumptions.UniformSecret sec) (L : ℕ) (domain : String)
    (context : List (String × Part)) (η : ℝ≥0∞)
    (hinj : ∀ j j', frames (windowIdx (KeyedDraw.flat calls) k j) = frames (windowIdx (KeyedDraw.flat calls) k j') → j = j')
    (hA4 : ∀ B : Finset (Fin n), Assumptions.KeyedStreamsUniform S domain context (windowIdx (KeyedDraw.flat calls) k) (fun _ => L)
      (windowTest (KeyedDraw.flat calls) k B) η)
    (B : Finset (Fin n)) :
    (keyedWindow (n := n) sec domain context calls L).escape B ≤ (stratified σ k hk).escape B + η := by
  classical
  set WS := KeyedDraw.flat calls with hWSdef
  obtain ⟨hlen, hW⟩ := hWS
  rcases B.eq_empty_or_nonempty with rfl | ⟨b, hb⟩
  · rw [stratified_escape_empty]
    exact (DrawExec.prCoin_le_one _).trans le_self_add
  · -- the escape event is the test on the key's streams
    have hkey : ∀ ω : Ω, Disjoint ((keyedWindow (n := n) sec domain context calls L).draw ω) B →
        windowTest WS k B fun j => stream (derive ⟨Array.ofFn fun b => UInt8.ofFin (sec ω b)⟩ domain context)
          (windowIdx WS k j) L := by
      intro ω hd
      set key := derive ⟨Array.ofFn fun b => UInt8.ofFin (sec ω b)⟩ domain context
      change Disjoint (match PlanDraw.window key calls L with
        | some D => univ.filter fun u => u.val ∈ D
        | none => univ) B at hd
      split at hd
      · rename_i D hD
        refine ⟨D, ?_, fun u hu huD => Finset.disjoint_left.1 hd (Finset.mem_filter.2 ⟨Finset.mem_univ _, huD⟩) hu⟩
        unfold PlanDraw.window at hD
        split at hD
        · rw [KeyedDraw.drawAll_eq, KeyedDraw.drawK_eq_drawOn key L (windowOf k fun j => stream key (windowIdx WS k j) L)
            WS 0 ?_] at hD
          · exact hD
          · intro j hj i hi
            have hjM : j < M := hlen ▸ hj
            have hk' := (hW ⟨j, hjM⟩ hj).1
            have hiK : i < k ⟨j, hjM⟩ := hk' ▸ hi
            rw [Nat.zero_add]
            simp only [windowOf, hjM, hiK, dite_true]
            unfold windowIdx
            rw [Equiv.symm_apply_apply]
            simp only [List.getD_eq_getElem _ _ hj]
        · exact absurd hD (by simp)
      · exact absurd hb (Finset.disjoint_left.1 hd (Finset.mem_univ b))
    -- the uniform side is the window draw on per-stratum, per-swap streams
    let e' : ((p : Fin M) → Fin (k p) → Fin L → Fin 256) ≃ (Fin (∑ p, k p) → Fin L → Fin 256) :=
      (Equiv.piCurry fun p (_ : Fin (k p)) => Fin L → Fin 256).symm.trans
        ((finSigmaFinEquiv (n := k)).arrowCongr (Equiv.refl _))
    have hunif : prCoin (fun f : (j : Fin (∑ p, k p)) → Fin ((fun _ => L) j) → Fin 256 =>
        windowTest WS k B fun j => ⟨Array.ofFn fun b => UInt8.ofFin (f j b)⟩) ≤ (stratified σ k hk).escape B := by
      rw [KeyedDraw.prCoin_eq_avg_ind, ← DrawExec.avg_equiv e', ← KeyedDraw.prCoin_eq_avg_ind]
      refine le_trans (prCoin_mono fun F hF => ?_) (drawOn_escape_le σ k hk WS ⟨hlen, hW⟩ L B)
      obtain ⟨D, hD, hmiss⟩ := hF
      refine ⟨D, ?_, hmiss⟩
      rw [← hD]
      refine drawOn_congr _ _ WS 0 fun j hj i hi => ?_
      have hjM : j < M := hlen ▸ hj
      have hiK : i < k ⟨j, hjM⟩ := (hW ⟨j, hjM⟩ hj).1 ▸ hi
      rw [Nat.zero_add]
      simp only [windowOf, windowStreams, KeyedDraw.streamsOf, hjM, hiK, dite_true]
      simp only [e', Equiv.trans_apply, Equiv.arrowCongr_apply, Equiv.coe_refl, Function.comp_apply,
        Equiv.symm_apply_apply]
      rfl
    calc (keyedWindow (n := n) sec domain context calls L).escape B
        ≤ prCoin (fun ω : Ω => windowTest WS k B fun j =>
            stream (derive ⟨Array.ofFn fun b => UInt8.ofFin (sec ω b)⟩ domain context) (windowIdx WS k j) L) :=
          prCoin_mono hkey
      _ = prCoin (fun src : Fin S → Fin 256 => windowTest WS k B fun j =>
            stream (derive ⟨Array.ofFn fun b => UInt8.ofFin (src b)⟩ domain context) (windowIdx WS k j) L) :=
          hsec fun src => windowTest WS k B fun j =>
            stream (derive ⟨Array.ofFn fun b => UInt8.ofFin (src b)⟩ domain context) (windowIdx WS k j) L
      _ ≤ prCoin (fun f : (j : Fin (∑ p, k p)) → Fin ((fun _ => L) j) → Fin 256 =>
            windowTest WS k B fun j => ⟨Array.ofFn fun b => UInt8.ofFin (f j b)⟩) + η := hA4 B hinj
      _ ≤ (stratified σ k hk).escape B + η := by gcongr

end FlockSoundness.Audit.Law

namespace FlockSoundness.Randomness

/-- Two numbers below `2^(8ℓ)` with the same `ℓ` big-endian bytes are equal. -/
theorem eq_of_bytes {x y ℓ : ℕ} (hx : x < 2 ^ (8 * ℓ)) (hy : y < 2 ^ (8 * ℓ))
    (h : ∀ q < ℓ, (x >>> (8 * q)) % 256 = (y >>> (8 * q)) % 256) : x = y := by
  refine Nat.eq_of_testBit_eq fun j => ?_
  by_cases hj : j < 8 * ℓ
  · have hq : j / 8 < ℓ := by omega
    have hr : j % 8 < 8 := Nat.mod_lt _ (by omega)
    have key : ∀ z : ℕ, z.testBit j = ((z >>> (8 * (j / 8))) % 256).testBit (j % 8) := by
      intro z
      rw [show (256 : ℕ) = 2 ^ 8 by norm_num, Nat.testBit_mod_two_pow, Nat.testBit_shiftRight]
      simp only [hr, decide_true, Bool.true_and]
      congr 1
      omega
    rw [key x, key y, h _ hq]
  · rw [Nat.testBit_lt_two_pow (lt_of_lt_of_le hx (Nat.pow_le_pow_right (by norm_num) (by omega))),
      Nat.testBit_lt_two_pow (lt_of_lt_of_le hy (Nat.pow_le_pow_right (by norm_num) (by omega)))]

theorem toUInt8_toNat (z : ℕ) : z.toUInt8.toNat = z % 256 := by
  simp [Nat.toUInt8]

theorem be8_toList (x : ℕ) : (be8 x).data.toList = (List.range 8).map fun i => (x >>> (8 * (7 - i))).toUInt8 := by
  simp [be8, Array.toList_map]

theorem be8_inj {x y : ℕ} (hx : x < 2 ^ 64) (hy : y < 2 ^ 64) (h : (be8 x).data.toList = (be8 y).data.toList) :
    x = y := by
  rw [be8_toList, be8_toList] at h
  refine eq_of_bytes (ℓ := 8) hx hy fun q hq => ?_
  have hi := congrArg (fun l : List UInt8 => (l[7 - q]?).map UInt8.toNat) h
  simp only [List.getElem?_map, List.getElem?_range (show 7 - q < 8 by omega), Option.map_some,
    toUInt8_toNat] at hi
  have : 7 - (7 - q) = q := by omega
  rw [this] at hi
  exact Option.some.inj hi

/-- `natBytes n`'s length. -/
def natLen (n : ℕ) : ℕ := max 1 (((if n = 0 then 0 else Nat.log2 n + 1) + 7) / 8)

theorem natBytes_toList (n : ℕ) :
    (natBytes n).data.toList = (List.range (natLen n)).map fun i => (n >>> (8 * (natLen n - 1 - i))).toUInt8 := by
  simp [natBytes, natLen, Array.toList_map]

theorem lt_natLen (n : ℕ) : n < 2 ^ (8 * natLen n) := by
  unfold natLen
  by_cases h0 : n = 0
  · subst h0; positivity
  · simp only [h0, ite_false]
    refine lt_of_lt_of_le (Nat.lt_log2_self (n := n)) (Nat.pow_le_pow_right (by norm_num) ?_)
    omega

theorem natBytes_inj {n m : ℕ} (h : (natBytes n).data.toList = (natBytes m).data.toList) : n = m := by
  rw [natBytes_toList, natBytes_toList] at h
  have hlen : natLen n = natLen m := by simpa using congrArg List.length h
  refine eq_of_bytes (ℓ := natLen n) (lt_natLen n) (hlen ▸ lt_natLen m) fun q hq => ?_
  have hi := congrArg (fun l : List UInt8 => (l[natLen n - 1 - q]?).map UInt8.toNat) h
  simp only [List.getElem?_map, List.getElem?_range (show natLen n - 1 - q < natLen n by omega),
    List.getElem?_range (show natLen n - 1 - q < natLen m by omega), Option.map_some, toUInt8_toNat] at hi
  rw [show natLen n - 1 - (natLen n - 1 - q) = q by omega, ← hlen,
    show natLen n - 1 - (natLen n - 1 - q) = q by omega] at hi
  exact Option.some.inj hi

/-- A part's body, the bytes its frame carries after the tag and the length. -/
def body : Part → ByteArray
  | .bytes b => b
  | .str s => s.toUTF8
  | .int n => natBytes n

/-- A part's tag. -/
def tag : Part → UInt8
  | .bytes _ => 98
  | .str _ => 115
  | .int _ => 105

theorem toList_append (a b : ByteArray) : (a ++ b).data.toList = a.data.toList ++ b.data.toList := by
  rw [ByteArray.data_append, Array.toList_append]

theorem frame_toList (p : Part) :
    (frame p).data.toList = tag p :: ((be8 (body p).size).data.toList ++ (body p).data.toList) := by
  cases p <;> simp [frame, body, tag]

theorem length_be8 (x : ℕ) : (be8 x).data.toList.length = 8 := by
  rw [be8_toList, List.length_map, List.length_range]

theorem length_toList (a : ByteArray) : a.data.toList.length = a.size := by
  cases a; exact Array.length_toList

theorem eq_of_toList {a b : ByteArray} (h : a.data.toList = b.data.toList) : a = b := by
  cases a; cases b; simp only [ByteArray.mk.injEq]; exact Array.toList_inj.1 h

/-- **A frame parses uniquely**: a part's frame followed by anything determines the part and the rest, for bodies
under `2^64` bytes. -/
theorem frame_append_inj {p q : Part} {r r' : List UInt8} (hp : (body p).size < 2 ^ 64) (hq : (body q).size < 2 ^ 64)
    (h : (frame p).data.toList ++ r = (frame q).data.toList ++ r') : p = q ∧ r = r' := by
  rw [frame_toList, frame_toList] at h
  simp only [List.cons_append, List.cons.injEq, List.append_assoc] at h
  obtain ⟨htag, h⟩ := h
  obtain ⟨h8, h⟩ := List.append_inj h (by rw [length_be8, length_be8])
  have hsize := be8_inj hp hq h8
  obtain ⟨hb, hr⟩ := List.append_inj h (by rw [length_toList, length_toList, hsize])
  refine ⟨?_, hr⟩
  have hbody := eq_of_toList hb
  cases p <;> cases q <;> simp only [tag] at htag <;> first | exact absurd htag (by decide) | skip
  · exact congrArg _ hbody
  · exact congrArg _ (String.toByteArray_inj.1 hbody)
  · exact congrArg _ (natBytes_inj hb)

theorem foldl_frames_toList (l : List Part) (acc : ByteArray) :
    (l.foldl (fun acc p => acc ++ frame p) acc).data.toList =
      acc.data.toList ++ (l.map fun p => (frame p).data.toList).flatten := by
  induction l generalizing acc with
  | nil => simp
  | cons p ps ih => rw [List.foldl_cons, ih, toList_append]; simp

/-- **A sequence of frames parses uniquely**, for bodies under `2^64` bytes. -/
theorem flatten_frames_inj : ∀ (l l' : List Part), (∀ p ∈ l, (body p).size < 2 ^ 64) →
    (∀ p ∈ l', (body p).size < 2 ^ 64) →
    (l.map fun p => (frame p).data.toList).flatten = (l'.map fun p => (frame p).data.toList).flatten → l = l' := by
  intro l
  induction l with
  | nil =>
    intro l' _ _ h
    cases l' with
    | nil => rfl
    | cons q qs => rw [List.map_cons, List.flatten_cons, frame_toList] at h; simp at h
  | cons p ps ih =>
    intro l' hs hs' h
    cases l' with
    | nil => rw [List.map_cons, List.flatten_cons, frame_toList] at h; simp at h
    | cons q qs =>
      rw [List.map_cons, List.map_cons, List.flatten_cons, List.flatten_cons] at h
      obtain ⟨rfl, hr⟩ := frame_append_inj (hs p List.mem_cons_self) (hs' q List.mem_cons_self) h
      rw [ih qs (fun x hx => hs x (List.mem_cons_of_mem _ hx)) (fun x hx => hs' x (List.mem_cons_of_mem _ hx)) hr]

/-- **`frames` is injective** on indices whose parts, and whose framed parts together, are under `2^64` bytes. -/
theorem frames_inj {l l' : List Part} (hs : ∀ p ∈ l, (body p).size < 2 ^ 64) (hs' : ∀ p ∈ l', (body p).size < 2 ^ 64)
    (hb : (l.foldl (fun acc p => acc ++ frame p) ByteArray.empty).size < 2 ^ 64)
    (hb' : (l'.foldl (fun acc p => acc ++ frame p) ByteArray.empty).size < 2 ^ 64)
    (h : frames l = frames l') : l = l' := by
  unfold frames at h
  have h' := congrArg (fun a : ByteArray => a.data.toList ++ ([] : List UInt8)) h
  obtain ⟨hp, _⟩ := frame_append_inj (p := .bytes _) (q := .bytes _) hb hb' h'
  simp only [Part.bytes.injEq] at hp
  have hl := congrArg (fun a : ByteArray => a.data.toList) hp
  simp only [foldl_frames_toList] at hl
  exact flatten_frames_inj l l' hs hs' (by simpa using hl)

theorem length_frame (p : Part) : (frame p).data.toList.length = 9 + (body p).size := by
  rw [frame_toList, List.length_cons, List.length_append, length_be8, length_toList]; omega

/-- An index framed in under `2^64` bytes has every part, and its parts' frames together, under `2^64` bytes. -/
theorem small_of_frames {l : List Part} (h : (frames l).size < 2 ^ 64) :
    (∀ p ∈ l, (body p).size < 2 ^ 64) ∧ (l.foldl (fun acc p => acc ++ frame p) ByteArray.empty).size < 2 ^ 64 := by
  have hlen := length_frame (.bytes (l.foldl (fun acc p => acc ++ frame p) ByteArray.empty))
  rw [← length_toList] at h
  unfold frames at h
  simp only [body] at hlen
  refine ⟨fun p hp => ?_, by omega⟩
  have hb : (body p).size ≤ (l.foldl (fun acc p => acc ++ frame p) ByteArray.empty).size := by
    have hL : (l.foldl (fun acc p => acc ++ frame p) ByteArray.empty).size =
        ((l.map fun p => (frame p).data.toList).flatten).length := by
      rw [← length_toList (l.foldl (fun acc p => acc ++ frame p) ByteArray.empty), foldl_frames_toList]
      simp
    rw [hL, List.length_flatten, List.map_map]
    have := List.le_sum_of_mem (List.mem_map_of_mem (f := List.length ∘ fun p => (frame p).data.toList) hp)
    simp only [Function.comp_apply, length_frame] at this
    omega
  omega

theorem frames_inj' {l l' : List Part} (hs : (frames l).size < 2 ^ 64) (h : frames l = frames l') : l = l' := by
  obtain ⟨h1, h2⟩ := small_of_frames hs
  obtain ⟨h1', h2'⟩ := small_of_frames (h ▸ hs)
  exact frames_inj h1 h1' h2 h2' h

end FlockSoundness.Randomness

namespace FlockSoundness.Audit.Law

open Game Finset Randomness
open scoped ENNReal

variable {n M : ℕ}

theorem swapIndex_inj {c c' : List Part} {i i' : ℕ} (h : swapIndex c i = swapIndex c' i') : c = c' ∧ i = i' := by
  unfold swapIndex at h
  simp only [List.cons_append, List.cons.injEq, true_and] at h
  obtain ⟨h1, h2⟩ := List.append_inj' h rfl
  exact ⟨h1, by simpa using h2⟩

theorem context_inj {c c' : List Part} {s s' : String} (h : PlanDraw.context c s = PlanDraw.context c' s') :
    c = c' ∧ s = s' := by
  unfold PlanDraw.context at h
  obtain ⟨h1, h2⟩ := List.append_inj' h rfl
  simp only [List.cons.injEq, Part.str.injEq, and_true, true_and] at h2
  refine ⟨?_, h2⟩
  cases c with
  | nil => cases c' with
    | nil => rfl
    | cons _ _ => simp at h1
  | cons x xs => cases c' with
    | nil => simp at h1
    | cons y ys => simpa using h1

/-- Admitted calls with distinct stratum names each give the window's strata distinct (call, name) pairs. -/
theorem flat_pairs_nodup (calls : List PlanDraw.Call) (hadm : (calls.map fun c => frames c.index).Nodup)
    (hnames : ∀ c ∈ calls, (c.strata.map (·.name)).Nodup) :
    ((KeyedDraw.flat calls).map fun x => (x.1, x.2.name)).Nodup := by
  unfold KeyedDraw.flat
  rw [List.map_flatMap]
  refine List.nodup_flatMap.2 ⟨fun c hc => ?_, ?_⟩
  · rw [List.map_map]
    have := (hnames c hc).map (f := fun s => (c.index, s)) (fun a b h => by simpa using h)
    rw [List.map_map] at this
    exact this
  · have hp : List.Pairwise (fun a b : PlanDraw.Call => frames a.index ≠ frames b.index) calls :=
      List.pairwise_map.1 hadm
    refine hp.imp fun {a b} hab => ?_
    intro x hxa hxb
    simp only [List.map_map, List.mem_map, Function.comp_apply] at hxa hxb
    obtain ⟨_, _, rfl⟩ := hxa
    obtain ⟨_, _, hx⟩ := hxb
    exact hab (congrArg frames (Prod.mk.inj hx).1).symm

/-- **The window's swap streams have pairwise distinct framings**: admitted calls, distinct stratum names within each
call, and every swap index framed in under `2^64` bytes. -/
theorem windowIdx_inj (calls : List PlanDraw.Call) (k : Fin M → ℕ) (hlen : (KeyedDraw.flat calls).length = M)
    (hadm : (calls.map fun c => frames c.index).Nodup) (hnames : ∀ c ∈ calls, (c.strata.map (·.name)).Nodup)
    (hsmall : ∀ j, (frames (windowIdx (KeyedDraw.flat calls) k j)).size < 2 ^ 64) :
    ∀ j j', frames (windowIdx (KeyedDraw.flat calls) k j) = frames (windowIdx (KeyedDraw.flat calls) k j') → j = j' := by
  intro j j' h
  have hl := frames_inj' (hsmall j) h
  set WS := KeyedDraw.flat calls
  have h1 : ((finSigmaFinEquiv (n := k)).symm j).1.val < WS.length := hlen ▸ ((finSigmaFinEquiv (n := k)).symm j).1.isLt
  have h1' : ((finSigmaFinEquiv (n := k)).symm j').1.val < WS.length :=
    hlen ▸ ((finSigmaFinEquiv (n := k)).symm j').1.isLt
  simp only [windowIdx] at hl
  rw [List.getD_eq_getElem _ _ h1, List.getD_eq_getElem _ _ h1'] at hl
  set x := (finSigmaFinEquiv (n := k)).symm j with hx
  set x' := (finSigmaFinEquiv (n := k)).symm j' with hx'
  obtain ⟨hc, hi⟩ := swapIndex_inj hl
  obtain ⟨hc1, hc2⟩ := context_inj hc
  have hpair : (WS.map fun y => (y.1, y.2.name))[x.1.val]'(by simpa using h1) =
      (WS.map fun y => (y.1, y.2.name))[x'.1.val]'(by simpa using h1') := by
    simp only [List.getElem_map]
    exact Prod.ext hc1 hc2
  have hp := (List.Nodup.getElem_inj_iff (flat_pairs_nodup calls hadm hnames)).1 hpair
  have hxx : x = x' := by
    have hfst : x.1 = x'.1 := Fin.ext hp
    refine Sigma.ext hfst ?_
    exact (Fin.heq_ext_iff (congrArg k hfst)).2 hi
  exact (finSigmaFinEquiv (n := k)).symm.injective hxx

/-- **Tier 3 for the PoUW window draw, from the draw's own checks**: `keyedWindow_escape_le` with its distinct framings
from admission, distinct stratum names within each call, and swap indices framed in under `2^64` bytes. A window that
admission refuses draws every unit. -/
theorem keyedWindow_escape_le_of_names (σ : Fin n → Fin M) (k : Fin M → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card)
    (calls : List PlanDraw.Call) (hWS : PlanStrata σ k (KeyedDraw.flat calls)) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    {S : ℕ} (sec : Ω → Fin S → Fin 256) (hsec : Assumptions.UniformSecret sec) (L : ℕ) (domain : String)
    (context : List (String × Part)) (η : ℝ≥0∞) (hnames : ∀ c ∈ calls, (c.strata.map (·.name)).Nodup)
    (hsmall : ∀ j, (frames (windowIdx (KeyedDraw.flat calls) k j)).size < 2 ^ 64)
    (hA4 : ∀ B : Finset (Fin n), Assumptions.KeyedStreamsUniform S domain context (windowIdx (KeyedDraw.flat calls) k)
      (fun _ => L) (windowTest (KeyedDraw.flat calls) k B) η)
    (B : Finset (Fin n)) :
    (keyedWindow (n := n) sec domain context calls L).escape B ≤ (stratified σ k hk).escape B + η := by
  classical
  by_cases hadm : (calls.map fun c => frames c.index).Nodup
  · exact keyedWindow_escape_le σ k hk calls hWS sec hsec L domain context η
      (windowIdx_inj calls k hWS.1 hadm hnames hsmall) hA4 B
  · rcases B.eq_empty_or_nonempty with rfl | ⟨b, hb⟩
    · rw [stratified_escape_empty]
      exact (DrawExec.prCoin_le_one _).trans le_self_add
    · refine le_trans (le_of_eq_of_le (b := 0) ?_ zero_le) zero_le
      refine le_antisymm ((prCoin_mono (q := fun _ => False) fun ω hd => ?_).trans prCoin_false.le) zero_le
      change Disjoint (match PlanDraw.window _ calls L with
        | some D => univ.filter fun u => u.val ∈ D
        | none => univ) B at hd
      unfold PlanDraw.window at hd
      simp only [hadm, ↓reduceIte] at hd
      exact absurd hb (Finset.disjoint_left.1 hd (Finset.mem_univ b))

end FlockSoundness.Audit.Law

namespace FlockSoundness.Audit.Law

open Game Finset Randomness
open scoped ENNReal

section Audits

variable {n M : ℕ} {cl : Fin n → Finset (Fin n)} {Reg : Type} {session : Finset (Fin n) → Reg → Game Bool}

/-- **The PoUW window audit on the live keyed draw, at the sizing of record** (step 4): the audit accepts while the unsound
tiles do at least 0.1% of the work, or the unsound y cells are at least 0.1% of them, with probability at most
`2^-40 + η + ε_ks + δ_link`. It is `audit_window_split_of_record_of_le_slack` at `L := keyedWindow sec domain context calls L`,
whose escape `keyedWindow_escape_le_of_names` bounds by the product law's plus `η`: one `η` for every `B`, A4's for the
draw's escape tests (`windowTest`), which cost the same for every `B`.

The key's context is the receipt digest, so the draw depends on the registration: it holds per strategy, for the context
`context` that the strategy's registration fixes (C1, #423: the key is derived from the complete receipt, and a call the
receipt doesn't list is refused), and for every context, A4 and the `Analysis` taken uniformly over them. The window's
secret is fresh and uniform (A5, C2). -/
theorem keyedWindow_audit_of_record {C : ℕ} (σ : Fin n → Fin M) (call : Fin M → Fin C) (w v f : Fin M → ℕ)
    (calls : List PlanDraw.Call) (hWS : PlanStrata σ (windowK σ call w f 27713) (KeyedDraw.flat calls))
    {Ω : Type} [Fintype Ω] [Nonempty Ω] {S : ℕ} (sec : Ω → Fin S → Fin 256) (hsec : Assumptions.UniformSecret sec)
    (L : ℕ) (domain : String) (context : List (String × Part)) (η : ℝ≥0∞)
    (hnames : ∀ c ∈ calls, (c.strata.map (·.name)).Nodup)
    (hsmall : ∀ j, (frames (windowIdx (KeyedDraw.flat calls) (windowK σ call w f 27713) j)).size < 2 ^ 64)
    (hA4 : ∀ B : Finset (Fin n), Assumptions.KeyedStreamsUniform S domain context
      (windowIdx (KeyedDraw.flat calls) (windowK σ call w f 27713)) (fun _ => L)
      (windowTest (KeyedDraw.flat calls) (windowK σ call w f 27713) B) η)
    (A : Analysis ((keyedWindow (n := n) sec domain context calls L).closure cl) Reg session) {εks δlink : ℝ≥0∞}
    (hks : A.KnowledgeSound εks) (hlink : A.LinkSound δlink) (hW : 0 < totalWork σ w)
    (hone : ∀ s, 0 < w s → callWork σ call w (call s) = w s * (stratum σ s).card)
    (hN : 0 < totalWork σ v) (hfy : ∀ s, 27713 * v s * (stratum σ s).card ≤ f s * totalWork σ v)
    (σ' : Strategy (audit ((keyedWindow (n := n) sec domain context calls L).closure cl) Reg session)) :
    prob (fun o => o.1 = true ∧ (totalWork σ w ≤ 1000 * unsoundWork σ w cl (A.wrong (A.committedOf σ')) ∨
        totalWork σ v ≤ 1000 * unsoundWork σ v cl (A.wrong (A.committedOf σ'))))
        (audit ((keyedWindow (n := n) sec domain context calls L).closure cl) Reg session) σ' ≤
      ((2 : ℝ≥0∞) ^ 40)⁻¹ + η + εks + δlink :=
  audit_window_split_of_record_of_le_slack
    (fun B => keyedWindow_escape_le_of_names σ _ (windowK_le σ call w f 27713) calls hWS sec hsec L domain context η
      hnames hsmall hA4 B) A hks hlink hW hone hN hfy σ'

/-- **The claim of record: the PoUW window audit on the live keyed draw at the compiled layer** (randomized extractor):
`keyedWindow_audit_of_record`'s bound with `ε_ks(σ)` and `δ_link(σ)`, through `extraction_audit_window_split_of_record_of_le_slack`. -/
theorem keyedWindow_extraction_audit_of_record {C : ℕ} (σ : Fin n → Fin M) (call : Fin M → Fin C) (w v f : Fin M → ℕ)
    (calls : List PlanDraw.Call) (hWS : PlanStrata σ (windowK σ call w f 27713) (KeyedDraw.flat calls))
    {Ω : Type} [Fintype Ω] [Nonempty Ω] {S : ℕ} (sec : Ω → Fin S → Fin 256) (hsec : Assumptions.UniformSecret sec)
    (L : ℕ) (domain : String) (context : List (String × Part)) (η : ℝ≥0∞)
    (hnames : ∀ c ∈ calls, (c.strata.map (·.name)).Nodup)
    (hsmall : ∀ j, (frames (windowIdx (KeyedDraw.flat calls) (windowK σ call w f 27713) j)).size < 2 ^ 64)
    (hA4 : ∀ B : Finset (Fin n), Assumptions.KeyedStreamsUniform S domain context
      (windowIdx (KeyedDraw.flat calls) (windowK σ call w f 27713)) (fun _ => L)
      (windowTest (KeyedDraw.flat calls) (windowK σ call w f 27713) B) η)
    (A : ExtractionAnalysis ((keyedWindow (n := n) sec domain context calls L).closure cl) Reg session)
    {εks δlink : (R : Reg) → Cont ((keyedWindow (n := n) sec domain context calls L).closure cl) Reg session R → ℝ≥0∞}
    (hks : A.KnowledgeSound εks) (hlink : A.LinkSound δlink) (hW : 0 < totalWork σ w)
    (hone : ∀ s, 0 < w s → callWork σ call w (call s) = w s * (stratum σ s).card)
    (hN : 0 < totalWork σ v) (hfy : ∀ s, 27713 * v s * (stratum σ s).card ≤ f s * totalWork σ v)
    (σ' : Strategy (audit ((keyedWindow (n := n) sec domain context calls L).closure cl) Reg session)) :
    prob (fun o => o.1 = true ∧ (totalWork σ w ≤ 1000 * unsoundWork σ w cl (A.wrong (A.committedOf σ')) ∨
        totalWork σ v ≤ 1000 * unsoundWork σ v cl (A.wrong (A.committedOf σ'))))
        (audit ((keyedWindow (n := n) sec domain context calls L).closure cl) Reg session) σ' ≤
      ((2 : ℝ≥0∞) ^ 40)⁻¹ + η + εks (reg σ') (cont σ') + δlink (reg σ') (cont σ') :=
  extraction_audit_window_split_of_record_of_le_slack
    (fun B => keyedWindow_escape_le_of_names σ _ (windowK_le σ call w f 27713) calls hWS sec hsec L domain context η
      hnames hsmall hA4 B) A hks hlink hW hone hN hfy σ'

/-- **The claim of record on the real, receipt-keyed game** (#423's C1): the key's context `ctx R` is the digest of the
registration's receipt, so the draw is `auditReg`'s. Strategy by strategy (`prob_auditReg`), it is
`keyedWindow_extraction_audit_of_record` at `ctx (reg σ)`, with A4 and the analysis taken for every registration and one `η`
for all of them. -/
theorem keyedWindowReg_extraction_audit_of_record {C : ℕ} (σ : Fin n → Fin M) (call : Fin M → Fin C)
    (w v f : Fin M → ℕ) (calls : List PlanDraw.Call)
    (hWS : PlanStrata σ (windowK σ call w f 27713) (KeyedDraw.flat calls))
    {Ω : Type} [Fintype Ω] [Nonempty Ω] {S : ℕ} (sec : Ω → Fin S → Fin 256) (hsec : Assumptions.UniformSecret sec)
    (L : ℕ) (domain : String) (ctx : Reg → List (String × Part)) (η : ℝ≥0∞)
    (hnames : ∀ c ∈ calls, (c.strata.map (·.name)).Nodup)
    (hsmall : ∀ j, (frames (windowIdx (KeyedDraw.flat calls) (windowK σ call w f 27713) j)).size < 2 ^ 64)
    (hA4 : ∀ (R : Reg) (B : Finset (Fin n)), Assumptions.KeyedStreamsUniform S domain (ctx R)
      (windowIdx (KeyedDraw.flat calls) (windowK σ call w f 27713)) (fun _ => L)
      (windowTest (KeyedDraw.flat calls) (windowK σ call w f 27713) B) η)
    (A : ∀ R, ExtractionAnalysis ((keyedWindow (n := n) sec domain (ctx R) calls L).closure cl) Reg session)
    {εks δlink : ∀ R, (R' : Reg) → Cont ((keyedWindow (n := n) sec domain (ctx R) calls L).closure cl) Reg session R' →
      ℝ≥0∞}
    (hks : ∀ R, (A R).KnowledgeSound (εks R)) (hlink : ∀ R, (A R).LinkSound (δlink R)) (hW : 0 < totalWork σ w)
    (hone : ∀ s, 0 < w s → callWork σ call w (call s) = w s * (stratum σ s).card)
    (hN : 0 < totalWork σ v) (hfy : ∀ s, 27713 * v s * (stratum σ s).card ≤ f s * totalWork σ v)
    (σ' : Strategy (auditReg (fun R => (keyedWindow (n := n) sec domain (ctx R) calls L).closure cl) session)) :
    prob (fun o => o.1 = true ∧
        (totalWork σ w ≤ 1000 * unsoundWork σ w cl ((A σ'.1).wrong ((A σ'.1).committedOf (atReg σ'))) ∨
          totalWork σ v ≤ 1000 * unsoundWork σ v cl ((A σ'.1).wrong ((A σ'.1).committedOf (atReg σ')))))
        (auditReg (fun R => (keyedWindow (n := n) sec domain (ctx R) calls L).closure cl) session) σ' ≤
      ((2 : ℝ≥0∞) ^ 40)⁻¹ + η + εks σ'.1 (reg (atReg σ')) (cont (atReg σ')) +
        δlink σ'.1 (reg (atReg σ')) (cont (atReg σ')) := by
  rw [prob_auditReg]
  exact keyedWindow_extraction_audit_of_record σ call w v f calls hWS sec hsec L domain (ctx σ'.1) η hnames hsmall
    (hA4 σ'.1) (A σ'.1) (hks σ'.1) (hlink σ'.1) hW hone hN hfy (atReg σ')

/-- The oracle-layer form on the real, receipt-keyed game: at most `2^-40 + η + ε_ks + δ_link`, one `ε_ks` and one
`δ_link` for every registration. -/
theorem keyedWindowReg_audit_of_record {C : ℕ} (σ : Fin n → Fin M) (call : Fin M → Fin C) (w v f : Fin M → ℕ)
    (calls : List PlanDraw.Call) (hWS : PlanStrata σ (windowK σ call w f 27713) (KeyedDraw.flat calls))
    {Ω : Type} [Fintype Ω] [Nonempty Ω] {S : ℕ} (sec : Ω → Fin S → Fin 256) (hsec : Assumptions.UniformSecret sec)
    (L : ℕ) (domain : String) (ctx : Reg → List (String × Part)) (η : ℝ≥0∞)
    (hnames : ∀ c ∈ calls, (c.strata.map (·.name)).Nodup)
    (hsmall : ∀ j, (frames (windowIdx (KeyedDraw.flat calls) (windowK σ call w f 27713) j)).size < 2 ^ 64)
    (hA4 : ∀ (R : Reg) (B : Finset (Fin n)), Assumptions.KeyedStreamsUniform S domain (ctx R)
      (windowIdx (KeyedDraw.flat calls) (windowK σ call w f 27713)) (fun _ => L)
      (windowTest (KeyedDraw.flat calls) (windowK σ call w f 27713) B) η)
    (A : ∀ R, Analysis ((keyedWindow (n := n) sec domain (ctx R) calls L).closure cl) Reg session) {εks δlink : ℝ≥0∞}
    (hks : ∀ R, (A R).KnowledgeSound εks) (hlink : ∀ R, (A R).LinkSound δlink) (hW : 0 < totalWork σ w)
    (hone : ∀ s, 0 < w s → callWork σ call w (call s) = w s * (stratum σ s).card)
    (hN : 0 < totalWork σ v) (hfy : ∀ s, 27713 * v s * (stratum σ s).card ≤ f s * totalWork σ v)
    (σ' : Strategy (auditReg (fun R => (keyedWindow (n := n) sec domain (ctx R) calls L).closure cl) session)) :
    prob (fun o => o.1 = true ∧
        (totalWork σ w ≤ 1000 * unsoundWork σ w cl ((A σ'.1).wrong ((A σ'.1).committedOf (atReg σ'))) ∨
          totalWork σ v ≤ 1000 * unsoundWork σ v cl ((A σ'.1).wrong ((A σ'.1).committedOf (atReg σ')))))
        (auditReg (fun R => (keyedWindow (n := n) sec domain (ctx R) calls L).closure cl) session) σ' ≤
      ((2 : ℝ≥0∞) ^ 40)⁻¹ + η + εks + δlink := by
  rw [prob_auditReg]
  exact keyedWindow_audit_of_record σ call w v f calls hWS sec hsec L domain (ctx σ'.1) η hnames hsmall (hA4 σ'.1)
    (A σ'.1) (hks σ'.1) (hlink σ'.1) hW hone hN hfy (atReg σ')

end Audits

end FlockSoundness.Audit.Law
