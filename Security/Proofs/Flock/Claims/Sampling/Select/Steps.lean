import Proofs.Flock.Claims.Sampling.Select.Layout
import Properties.SampledProofs.Draw

open SampledProofs

namespace Properties.SampledProofs.SecurityProofs

open Lean (Json)

/-- The law object `Select.select` reads: the expansion's, else the named law's. -/
def lawObj (named : Json) (derived : Option Flock.Draw.Law) : Json := (derived.map (·.toJson)).getD named

/-- **`Select.select`'s steps**: an accepted draw read the population `n`, the rule's law over it and its expansion, and
proves every unit it drew: all of `[0, n)` under `All`, else the units `drawUnits` draws from the coins under the law
`flockLaw` accepts. -/
theorem select_ok (rule : Select.Rule) (d : Program) (query : Json) (table : Option Json)
    (cm : Option (Array (Int × Array Int))) (src : ByteArray) (draw : Json) (proved : Array ℕ) (read : ℕ)
    (h : Select.select rule d query table cm src = .ok (draw, proved, read)) :
    ∃ n named derived, Partition.units d query = .ok n ∧ rule.lawOver n = .ok named ∧
      Draw.deriveLaw d query named table = .ok derived ∧ ∃ units : Array ℕ, (∀ x ∈ units, x ∈ proved) ∧
        ((rule = .all ∧ units = Array.range n) ∨
          (rule ≠ .all ∧ ∃ parsed read', Draw.flockLaw (lawObj named derived) n = .ok parsed ∧
            Draw.drawUnits (derived.getD parsed) n src = .ok (units, read'))) := by
  unfold Select.select at h
  obtain ⟨n, hn, h⟩ := bind_eq_ok.1 h
  obtain ⟨named, hnamed, h⟩ := bind_eq_ok.1 h
  obtain ⟨derived, hderived, h⟩ := bind_eq_ok.1 h
  refine ⟨n, named, derived, hn, hnamed, hderived, ?_⟩
  dsimp only at h
  have tail : ∀ (units : Array ℕ) (c : Bool) (e : String) (X : Json) (C : Array ℕ) (r : ℕ),
      (if c = true then ((throw e : Flock.V Unit) >>= fun _ => (pure (X, units ++ C, r) : Flock.V (Json × Array ℕ × ℕ)))
        else pure (X, units ++ C, r)) = .ok (draw, proved, read) → ∀ x ∈ units, x ∈ proved := by
    intro units c e X C r ht x hx
    split at ht
    · exact absurd ht (by simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind])
    · simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at ht
      obtain ⟨-, rfl, -⟩ := ht
      exact Array.mem_append_left _ hx
  split at h
  · split at h
    · exact absurd h (by simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind])
    · split at h
      · rw [pure_bind] at h
        exact ⟨_, tail _ _ _ _ _ _ h, Or.inl ⟨rfl, rfl⟩⟩
      · rename_i hne
        obtain ⟨parsed, hp, h⟩ := bind_eq_ok.1 h
        obtain ⟨⟨units, read'⟩, hu, h⟩ := bind_eq_ok.1 h
        exact ⟨units, tail _ _ _ _ _ _ h, Or.inr ⟨fun e => hne e, parsed, read', hp, hu⟩⟩
  · split at h
    · rw [pure_bind] at h
      exact ⟨_, tail _ _ _ _ _ _ h, Or.inl ⟨rfl, rfl⟩⟩
    · rename_i hne
      obtain ⟨parsed, hp, h⟩ := bind_eq_ok.1 h
      obtain ⟨⟨units, read'⟩, hu, h⟩ := bind_eq_ok.1 h
      exact ⟨units, tail _ _ _ _ _ _ h, Or.inr ⟨fun e => hne e, parsed, read', hp, hu⟩⟩

/-- A law object's kind, as `flockLaw` and `deriveLaw` read it. -/
def kindOf (law : Json) : String := match getD law "law" with | .str s => s | _ => ""

/-- A probability `NUM/DEN` `probability?` accepts has `0 < DEN` and `NUM ≤ DEN`. -/
theorem probability_ok (p : String) (a b : ℕ) (h : Draw.probability? p = some (a, b)) : 0 < b ∧ a ≤ b := by
  unfold Draw.probability? at h
  split at h
  · split at h
    · simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      rename_i hc
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
      exact ⟨hc.1.2, hc.2⟩
    · exact absurd h (by simp)
  · exact absurd h (by simp)

theorem throw_bind_ne {ε α β : Type} (e : ε) (f : α → Except ε β) (b : β) :
    ((throw e : Except ε α) >>= f) ≠ .ok b := by
  simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind]

theorem throw_ne {ε α : Type} (e : ε) (b : α) : (throw e : Except ε α) ≠ .ok b := by
  simp [throw, throwThe, MonadExceptOf.throw]

/-- **The law `flockLaw` accepts**: a subset of the law object's `k` units, no more than the population, a Bernoulli law
at a probability, or a law of kind `stratified` or `work`. -/
theorem flockLaw_cases (law : Json) (n : ℕ) (parsed : Flock.Draw.Law) (h : Draw.flockLaw law n = .ok parsed) :
    (∃ k, parsed = .subset k ∧ k ≤ n ∧ nat? (getD law "k") = some k) ∨
      (∃ a b, parsed = .bernoulli a b ∧ 0 < b ∧ a ≤ b) ∨ kindOf law = "stratified" ∨ kindOf law = "work" := by
  unfold Draw.flockLaw at h
  split at h
  next s hs =>
    have hk : kindOf law = s := by simp [kindOf, hs]
    dsimp only at h
    split at h
    · split at h
      · exact absurd h (throw_bind_ne _ _ _)
      split at h
      · rename_i k hnat
        split at h
        · exact absurd h (throw_bind_ne _ _ _)
        · obtain rfl := (Except.ok.inj h).symm
          exact Or.inl ⟨_, rfl, by omega, hnat⟩
      · exact absurd h (throw_ne _ _)
    · split at h
      · exact absurd h (throw_bind_ne _ _ _)
      split at h
      · split at h
        · obtain rfl := (Except.ok.inj h).symm
          exact Or.inr (Or.inl ⟨_, _, rfl, probability_ok _ _ _ (by assumption)⟩)
        · exact absurd h (throw_ne _ _)
      · exact absurd h (throw_ne _ _)
    · exact Or.inr (Or.inr (Or.inl (by rwa [hk])))
    · exact Or.inr (Or.inr (Or.inr (by rwa [hk])))
    · exact absurd h (throw_ne _ _)
  next => exact absurd h (throw_ne _ _)

/-- **`deriveLaw`'s expansion**: none for a law of neither kind `stratified` nor `work`, else the stratified or work law
`Flock.Draw.derive` or `Flock.Draw.deriveWork` expands over the query's strata. -/
theorem deriveLaw_cases (d : Program) (query law : Json) (table : Option Json)
    (derived : Option Flock.Draw.Law) (h : Draw.deriveLaw d query law table = .ok derived) :
    (derived = none ∧ kindOf law ≠ "stratified" ∧ kindOf law ≠ "work") ∨
      ∃ S k law', Partition.strata d query = .ok S ∧ derived = some law' ∧
        (Flock.Draw.derive k S = .ok law' ∨
          ∃ t tab, Flock.Draw.workTableOfJson t = .ok tab ∧ Flock.Draw.deriveWork k S tab = .ok law') := by
  unfold Draw.deriveLaw at h
  split at h
  next s hs =>
    have hk : kindOf law = s := by simp [kindOf, hs]
    dsimp only at h
    split at h
    · rename_i hc
      obtain rfl := (Except.ok.inj h).symm
      simp only [Bool.not_eq_true', Bool.or_eq_false_iff, beq_eq_false_iff_ne, ne_eq] at hc
      exact Or.inl ⟨rfl, hk ▸ hc.1, hk ▸ hc.2⟩
    · obtain ⟨S, hS, h⟩ := bind_eq_ok.1 h
      split at h
      · split at h
        · obtain ⟨law', hd, h⟩ := bind_eq_ok.1 h
          obtain rfl := (Except.ok.inj h).symm
          exact Or.inr ⟨S, _, law', hS, rfl, Or.inl hd⟩
        · split at h
          · obtain ⟨tab, ht, h⟩ := bind_eq_ok.1 h
            obtain ⟨law', hd, h⟩ := bind_eq_ok.1 h
            obtain rfl := (Except.ok.inj h).symm
            exact Or.inr ⟨S, _, law', hS, rfl, Or.inr ⟨_, tab, ht, hd⟩⟩
          · exact absurd h (throw_ne _ _)
      · exact absurd h (throw_ne _ _)
  next hne =>
    have hk : kindOf law = "" := by
      unfold kindOf; split
      · rename_i s hs; exact absurd hs (hne s)
      · rfl
    dsimp only at h
    split at h
    swap
    · rename_i hc; exact absurd (by decide) hc
    obtain rfl := (Except.ok.inj h).symm
    exact Or.inl ⟨rfl, by rw [hk]; decide, by rw [hk]; decide⟩

/-- The units of ascending ranges `[lo, hi)`. -/
def rsSize (rs : Array (ℕ × ℕ)) : ℕ := (expand rs.toList).length

/-- The units of strata. -/
def strataSize (S : Array (String × Array (ℕ × ℕ))) : ℕ := (S.toList.map fun p => rsSize p.2).sum

theorem foldl_rs (l : List (ℕ × ℕ)) (a : ℕ) :
    l.foldl (fun a (x : ℕ × ℕ) => a + (x.2 - x.1)) a = a + (expand l).length := by
  induction l generalizing a with
  | nil => simp [expand]
  | cons p l ih => simp only [List.foldl_cons, ih, expand, List.flatMap_cons, List.length_append, List.length_range']; omega

theorem size_rs (rs : Array (ℕ × ℕ)) : rs.foldl (fun a (lo, hi) => a + (hi - lo)) 0 = rsSize rs := by
  rw [← Array.foldl_toList]
  exact (foldl_rs rs.toList 0).trans (Nat.zero_add _)

theorem stratum_size (s : Flock.Draw.Stratum) : s.size = rsSize s.units := size_rs s.units

theorem foldl_strata (l : List (String × Array (ℕ × ℕ))) (a : ℕ) :
    l.foldl (fun acc (x : String × Array (ℕ × ℕ)) => acc + rsSize x.2) a = a + (l.map fun p => rsSize p.2).sum := by
  induction l generalizing a with
  | nil => simp
  | cons p l ih => simp only [List.foldl_cons, ih, List.map_cons, List.sum_cons]; omega

theorem size_strata (S : Array (String × Array (ℕ × ℕ))) :
    S.foldl (fun acc (x : String × Array (ℕ × ℕ)) => acc + rsSize x.2) 0 = strataSize S := by
  rw [← Array.foldl_toList]
  exact (foldl_strata S.toList 0).trans (Nat.zero_add _)

/-- **`Flock.Draw.derive`'s law**: each stratum its ranges, drawing `stratumK` of its units. -/
theorem derive_ok (k : ℕ) (S : Array (String × Array (ℕ × ℕ))) (law' : Flock.Draw.Law)
    (h : Flock.Draw.derive k S = .ok law') :
    law' = .stratified k (S.map fun p => { template := p.1, units := p.2, k := Flock.Draw.stratumK k (rsSize p.2) (strataSize S) }) := by
  unfold Flock.Draw.derive at h
  dsimp only at h
  split at h
  · exact absurd h (throw_bind_ne _ _ _)
  · obtain rfl := (Except.ok.inj h).symm
    simp only [size_rs, size_strata]

/-- **`Flock.Draw.deriveWork`'s law**: each stratum its ranges, drawing `workK` of its units by its work and floor. -/
theorem deriveWork_ok (k : ℕ) (S : Array (String × Array (ℕ × ℕ))) (tab : Array (String × ℕ × ℕ))
    (law' : Flock.Draw.Law) (h : Flock.Draw.deriveWork k S tab = .ok law') :
    ∃ ws : Array Flock.Draw.WorkStratum, List.Forall₂ (fun p (w : Flock.Draw.WorkStratum) => w.units = p.2) S.toList ws.toList ∧
      law' = .work k (ws.map fun s => { s with k := Flock.Draw.workK k s.work s.size (Flock.Draw.totalWork ws) s.floor }) := by
  unfold Flock.Draw.deriveWork at h
  dsimp only at h
  split at h
  · exact absurd h (throw_bind_ne _ _ _)
  · obtain ⟨ws, hws, h⟩ := bind_eq_ok.1 h
    refine ⟨ws, (mapM_ok _ _ _ hws).imp fun {p w} hp => ?_, ?_⟩
    · obtain ⟨t, rs⟩ := p
      split at hp
      · obtain rfl := (Except.ok.inj hp).symm; rfl
      · exact absurd hp (throw_ne _ _)
    · split at h
      · exact absurd h (throw_bind_ne _ _ _)
      · exact (Except.ok.inj h).symm

/-- `Slots.select`'s steps: the layout, and the checked proof units of the units `Select.select` proves. -/
theorem slots_select_ok (rule : Select.Rule) (d : Program) (query : Json) (table : Option Json)
    (cm : Option (Array (Int × Array Int))) (src : ByteArray) (l : Slots.Layout) (c : Array ℕ)
    (h : Slots.select rule d query table cm src = .ok (l, c)) :
    Slots.layout d query = .ok l ∧ ∃ draw proved read, Select.select rule d query table cm src = .ok (draw, proved, read) ∧
      c = Slots.checked l proved := by
  unfold Slots.select at h
  obtain ⟨l', hl, h⟩ := bind_eq_ok.1 h
  obtain ⟨⟨draw, proved, read⟩, hs, h⟩ := bind_eq_ok.1 h
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj h)
  exact ⟨hl, draw, proved, read, hs, rfl⟩

/-- The design under the `All` rule draws both classes `all`. -/
theorem slotDesign_all (d : Program) (query : Json) (table : Option Json) (n : ℕ) (named : Json)
    (derived : Option Flock.Draw.Law) (hn : Partition.units d query = .ok n) (hnamed : Select.Rule.all.lawOver n = .ok named)
    (hd : Draw.deriveLaw d query named table = .ok derived) :
    Properties.SampledProofs.slotDesign .all d query table = .ok ⟨[.all, .all]⟩ := by
  unfold Properties.SampledProofs.slotDesign
  rw [hn, bind_ok, hnamed, bind_ok, hd, bind_ok]
  rfl

/-- The design under a law draws the population as the law `Select.select` samples and the inputs `all`. -/
theorem slotDesign_law (rule : Select.Rule) (d : Program) (query : Json) (table : Option Json) (n : ℕ)
    (named : Json) (derived : Option Flock.Draw.Law) (parsed : Flock.Draw.Law) (hne : rule ≠ .all)
    (hn : Partition.units d query = .ok n) (hnamed : rule.lawOver n = .ok named)
    (hd : Draw.deriveLaw d query named table = .ok derived) (hp : Draw.flockLaw (lawObj named derived) n = .ok parsed) :
    Properties.SampledProofs.slotDesign rule d query table =
      .ok ⟨[Properties.SampledProofs.oneStageOf (derived.getD parsed), .all]⟩ := by
  unfold Properties.SampledProofs.slotDesign
  rw [hn, bind_ok, hnamed, bind_ok, hd, bind_ok]
  split
  · exact absurd rfl hne
  · unfold lawObj at hp
    rw [hp, bind_ok]
    rfl

end Properties.SampledProofs.SecurityProofs
