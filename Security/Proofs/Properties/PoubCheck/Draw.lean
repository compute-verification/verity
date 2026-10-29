import Properties.PoubCheck
import Proofs.Properties.PoubCheck.Bisect
import Proofs.Properties.PoubCheck.Sort

/-!
# The window's draws (`DrawIsSubset`)

`Key.subset`'s loop is a partial Fisher–Yates over a `HashMap` of swaps: read through `t ↦ sw.getD t t` it is a
function that keeps `[0, n)` and is injective there, and swap `i` composes it with the transposition of `i` and
`i + r`, `r < n − i` (`loopSwK_perm`). So a subset that succeeds with `k ≤ n` is `k` distinct values below `n`
(`subsetOn_spec`). `subsetExact` returns the subset at the first read length that succeeds (`subsetExact_some`), and
`draws` maps each value to its receipt by the passes' running ends (`locate`).
-/

open PoubCheck

namespace Properties.PoubCheck

theorem uniform_lt {m : Nat} {src : ByteArray} : ∀ {fuel pos r p : Nat},
    Flock.Draw.uniform m src fuel pos = some (r, p) → r < m := by
  intro fuel
  induction fuel with
  | zero => intro pos r p h; simp [Flock.Draw.uniform] at h
  | succ fuel ih =>
    intro pos r p h
    unfold Flock.Draw.uniform at h
    dsimp only at h
    split at h
    · cases h
    · split at h
      · rename_i hx
        obtain ⟨rfl, -⟩ := Prod.mk.inj (Option.some.inj h)
        rcases Nat.eq_zero_or_pos m with rfl | hm
        · simp at hx
        · exact Nat.mod_lt _ hm
      · exact ih h

theorem uniformOn_lt {m : Nat} {s : ByteArray} {r : Nat} (h : FlockSoundness.Randomness.uniformOn m s = some r) :
    r < m := by
  unfold FlockSoundness.Randomness.uniformOn at h
  cases hu : Flock.Draw.uniform m s (s.size + 1) 0 with
  | none => rw [hu] at h; cases h
  | some rp =>
    rw [hu] at h
    obtain ⟨r', p⟩ := rp
    simp only [Option.map_some, Option.some.injEq] at h
    subst h
    exact uniform_lt hu

/-! ## The sampler -/

/-- One swap of `Key.subset`'s Fisher–Yates: positions `i` and `i + r`. -/
def swapAt (sw : Std.HashMap Nat Nat) (i r : Nat) : Std.HashMap Nat Nat :=
  (sw.insert i (sw.getD (i + r) (i + r))).insert (i + r) (sw.getD i i)

/-- `Key.subset`'s loop as a list recursion, swap `i` drawing from `S i`. -/
def loopSwK (n : Nat) (S : Nat → ByteArray) : List Nat → Std.HashMap Nat Nat → Option (Std.HashMap Nat Nat)
  | [], sw => some sw
  | i :: is, sw => match FlockSoundness.Randomness.uniformOn (n - i) (S i) with
    | none => none
    | some r => loopSwK n S is (swapAt sw i r)

abbrev StK := Option (Option (Array Nat)) × Std.HashMap Nat Nat

theorem forIn_charK (n : Nat) (S : Nat → ByteArray) (G : Std.HashMap Nat Nat → Array Nat)
    (F : Nat → StK → Id (ForInStep StK)) (POST : StK → Id (Option (Array Nat)))
    (hFn : ∀ i s, FlockSoundness.Randomness.uniformOn (n - i) (S i) = none → F i s = pure (.done (some none, s.2)))
    (hFs : ∀ i s r, FlockSoundness.Randomness.uniformOn (n - i) (S i) = some r →
      F i s = pure (.yield (none, swapAt s.2 i r)))
    (hPs : ∀ s r, s.1 = some r → POST s = pure r) (hPn : ∀ s, s.1 = none → POST s = pure (some (G s.2))) :
    ∀ (l : List Nat) (sw : Std.HashMap Nat Nat),
      (forIn (m := Id) l ((none : Option (Option (Array Nat))), sw) F >>= POST).run = (loopSwK n S l sw).map G := by
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
    cases h : FlockSoundness.Randomness.uniformOn (n - i) (S i) with
    | none =>
      rw [hFn i _ h]
      simp only [loopSwK, h, Option.map_none, pure_bind]
      rw [hPs _ none rfl]
      rfl
    | some r =>
      rw [hFs i _ r h]
      simp only [loopSwK, h, pure_bind]
      exact ih _

theorem subsetOn_eq (k n : Nat) (S : Nat → ByteArray) :
    FlockSoundness.Randomness.subsetOn k n S = (loopSwK n S (List.range' 0 k) ∅).map
      (fun sw => ((Array.range k).map fun i => sw.getD i i).qsort (· < ·)) := by
  unfold FlockSoundness.Randomness.subsetOn
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

/-- The transposition of `i` and `j`. -/
def swapFn (i j t : Nat) : Nat := if t = i then j else if t = j then i else t

theorem swapFn_swapFn (i j t : Nat) : swapFn i j (swapFn i j t) = t := by
  unfold swapFn
  by_cases h1 : t = i <;> by_cases h2 : t = j <;> simp_all

theorem getD_swapAt (sw : Std.HashMap Nat Nat) (f : Nat → Nat) (h : ∀ t, sw.getD t t = f t) (i r t : Nat) :
    (swapAt sw i r).getD t t = f (swapFn i (i + r) t) := by
  simp only [swapAt, Std.HashMap.getD_insert, beq_iff_eq, h, swapFn]
  by_cases h1 : i + r = t
  · rw [ite_eq_left h1]
    by_cases h3 : t = i
    · rw [ite_eq_left h3]; congr 1; omega
    · rw [ite_eq_right h3, ite_eq_left h1.symm]
  · rw [ite_eq_right h1]
    by_cases h2 : i = t
    · rw [ite_eq_left h2, ite_eq_left h2.symm]
    · rw [ite_eq_right h2, ite_eq_right (Ne.symm h2), ite_eq_right (Ne.symm h1)]

/-- `f` keeps `[0, n)` and is injective there. -/
def PermOn (n : Nat) (f : Nat → Nat) : Prop :=
  (∀ t, t < n → f t < n) ∧ ∀ a b, a < n → b < n → f a = f b → a = b

theorem permOn_swap {n : Nat} {f : Nat → Nat} (hf : PermOn n f) {i j : Nat} (hi : i < n) (hj : j < n) :
    PermOn n (fun t => f (swapFn i j t)) := by
  have hlt : ∀ t, t < n → swapFn i j t < n := by
    intro t ht; unfold swapFn; split
    · exact hj
    · split
      · exact hi
      · exact ht
  refine ⟨fun t ht => hf.1 _ (hlt t ht), fun a b ha hb hab => ?_⟩
  have := hf.2 _ _ (hlt a ha) (hlt b hb) hab
  rw [← swapFn_swapFn i j a, this, swapFn_swapFn]

theorem loopSwK_perm (n : Nat) (S : Nat → ByteArray) : ∀ (l : List Nat) (sw : Std.HashMap Nat Nat) (f : Nat → Nat),
    (∀ i ∈ l, i < n) → (∀ t, sw.getD t t = f t) → PermOn n f →
      ∀ sw', loopSwK n S l sw = some sw' → ∃ f', (∀ t, sw'.getD t t = f' t) ∧ PermOn n f' := by
  intro l
  induction l with
  | nil =>
    intro sw f _ h hf sw' hs
    simp only [loopSwK, Option.some.injEq] at hs
    subst hs
    exact ⟨f, h, hf⟩
  | cons i is ih =>
    intro sw f hl h hf sw' hs
    simp only [loopSwK] at hs
    cases hu : FlockSoundness.Randomness.uniformOn (n - i) (S i) with
    | none => rw [hu] at hs; cases hs
    | some r =>
      rw [hu] at hs
      have hi := hl i (List.mem_cons_self ..)
      have hr := uniformOn_lt hu
      exact ih _ _ (fun x hx => hl x (List.mem_cons_of_mem _ hx)) (getD_swapAt sw f h i r)
        (permOn_swap hf hi (by omega)) sw' hs

/-- **A `k`-subset of `[0, n)` that succeeds, `k ≤ n`, is `k` distinct values below `n`.** -/
theorem subsetOn_spec (k n : Nat) (S : Nat → ByteArray) (hk : k ≤ n) (us : Array Nat)
    (h : FlockSoundness.Randomness.subsetOn k n S = some us) :
    us.size = k ∧ (∀ x ∈ us.toList, x < n) ∧ us.toList.Nodup := by
  rw [subsetOn_eq] at h
  cases hl : loopSwK n S (List.range' 0 k) ∅ with
  | none => rw [hl] at h; cases h
  | some sw =>
    rw [hl, Option.map_some, Option.some.injEq] at h
    obtain ⟨f, hsw, hf⟩ := loopSwK_perm n S _ ∅ id (fun i hi => by simp at hi; omega) (fun t => by simp)
      ⟨fun t ht => ht, fun a b _ _ hab => hab⟩ sw hl
    have hp := qsort_toList_perm ((Array.range k).map fun i => sw.getD i i) (· < ·)
    rw [h] at hp
    simp only [Array.toList_map, Array.toList_range] at hp
    refine ⟨?_, fun x hx => ?_, ?_⟩
    · have := hp.length_eq
      simpa using this
    · obtain ⟨i, hi, rfl⟩ := List.mem_map.1 (hp.mem_iff.1 hx)
      rw [hsw]
      exact hf.1 _ (by rw [List.mem_range] at hi; omega)
    · rw [hp.nodup_iff, List.Nodup, List.pairwise_map]
      refine List.Pairwise.imp_of_mem (fun {a b} ha hb hab => ?_) List.pairwise_lt_range
      rw [List.mem_range] at ha hb
      rw [hsw, hsw]
      intro heq
      have := hf.2 a b (by omega) (by omega) heq
      omega

abbrev StE := Option (Option (Array Nat)) × Nat

theorem forIn_exact (P : Nat → Option (Array Nat)) (F : Nat → StE → Id (ForInStep StE))
    (POST : StE → Id (Option (Array Nat)))
    (hFs : ∀ x s out, P s.2 = some out → F x s = pure (.done (some (some out), s.2)))
    (hFn : ∀ x s, P s.2 = none → F x s = pure (.yield (none, 2 * s.2)))
    (hPs : ∀ s r, s.1 = some r → POST s = pure r) (hPn : ∀ s, s.1 = none → POST s = pure none) :
    ∀ (l : List Nat) (len : Nat) (out : Array Nat),
      (forIn (m := Id) l ((none : Option (Option (Array Nat))), len) F >>= POST).run = some out → ∃ L, P L = some out := by
  intro l
  induction l with
  | nil =>
    intro len out h
    simp only [List.forIn_nil] at h
    rw [pure_bind, hPn _ rfl] at h
    cases h
  | cons x l ih =>
    intro len out h
    rw [List.forIn_cons] at h
    cases hp : P len with
    | some o =>
      rw [hFs x _ o hp, pure_bind] at h
      dsimp only at h
      rw [pure_bind, hPs (some (some o), len) (some o) rfl] at h
      cases h
      exact ⟨len, hp⟩
    | none =>
      rw [hFn x _ hp, pure_bind] at h
      exact ih _ _ h

theorem subsetExact_some (key : ByteArray) (k n : Nat) (index : List FlockSoundness.Randomness.Part) (us : Array Nat)
    (h : subsetExact key k n index = some us) : ∃ L, FlockSoundness.Randomness.subset key k n index L = some us := by
  unfold subsetExact at h
  simp only [Std.Legacy.Range.forIn_eq_forIn_range'] at h
  refine forIn_exact (fun L => FlockSoundness.Randomness.subset key k n index L) _ _ ?_ ?_ ?_ ?_ _ 64 us h
  · intro x s out hs; simp only [hs]
  · intro x s hs; simp only [hs]
  · intro s r hs; simp only [hs]
  · intro s hs; simp only [hs]

/-! ## The draws -/

theorem drawIsSubset : DrawIsSubset := by
  intro registration t coin samples lengths drawn h
  unfold draws at h
  dsimp only at h
  have hP := prefixSums_fold (fun n => n) lengths
  simp only [List.map_id'] at hP
  generalize (lengths.foldl (fun (acc : Array Nat × Nat) n => (acc.1.push (acc.2 + n), acc.2 + n)) (#[], 0)).1 = ends
    at h hP
  have htotal := back_getD_of_prefix hP
  rw [htotal] at h
  cases hu : subsetExact (openingsKey registration t coin) (min samples lengths.toList.sum) lengths.toList.sum
    [.str "openings"] with
  | none =>
    simp only [openingsKey] at hu
    rw [hu] at h
    cases h
  | some us =>
    simp only [openingsKey] at hu
    rw [hu] at h
    change some _ = some drawn at h
    replace h := (Option.some.inj h).symm
    obtain ⟨L, hL⟩ := subsetExact_some _ _ _ _ _ hu
    obtain ⟨hsize, hlt, hnodup⟩ := subsetOn_spec _ _ _ (Nat.min_le_right _ _) us hL
    have hdsize : drawn.size = us.size := by rw [h]; simp
    have hk : ∀ k (hk : k < drawn.size), drawn[k].1 < lengths.size ∧ drawn[k].2 < lengths[drawn[k].1]! ∧
        receiptIndex lengths drawn[k].1 drawn[k].2 = us[k]! := by
      intro k hk
      have huk : us[k]! < lengths.toList.sum := by
        rw [getElem!_pos us k (by omega)]
        exact hlt _ (Array.getElem_mem_toList _)
      obtain ⟨hi, hlo, hhi, hoff⟩ := locate hP _ huk
      have hd : drawn[k] = (bisectRight ends us[k]!,
          us[k]! - (if bisectRight ends us[k]! = 0 then 0 else ends[bisectRight ends us[k]! - 1]!)) := by
        subst h
        simp [getElem!_pos us k (by simpa using hk)]
      rw [hd]
      simp only [Array.length_toList] at hi
      refine ⟨hi, ?_, ?_⟩
      · rw [hoff]
        have : lengths[bisectRight ends us[k]!]! = lengths.toList[bisectRight ends us[k]!]! := by
          rw [getElem!_pos lengths _ hi, getElem!_pos lengths.toList _ (by simpa using hi)]
          simp
        rw [this]
        omega
      · simp only [receiptIndex]
        rw [hoff]
        omega
    refine ⟨L, us, hL, hdsize, hsize, hk, fun k₁ k₂ h₁ h₂ heq => ?_⟩
    have e₁ := (hk k₁ h₁).2.2
    have e₂ := (hk k₂ h₂).2.2
    rw [heq] at e₁
    rw [e₁, getElem!_pos us k₁ (by omega), getElem!_pos us k₂ (by omega)] at e₂
    have : us.toList[k₁]'(by simp; omega) = us.toList[k₂]'(by simp; omega) := by simpa using e₂
    exact (List.Nodup.getElem_inj hnodup).1 this

end Properties.PoubCheck
