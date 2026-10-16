import Proofs.Flock.Claims.Finder
import Proofs.Flock.Claims.Sampling.Extract

/-!
# `B_ext`: the commit strings opened inside the regions' Flock proofs bind the boundary values

Each region's Flock proof opens the commit strings of its boundary values inside the proof: its witness holds a value
`u` and a salt `s` with `commit(u; s) = c`. Two regions that share a boundary open one commit string, and the sampled
extractor recovers both openings, one from each region's reruns. If they open it to different values of one length,
`H(u ‖ s) = H(u′ ‖ s′)` with `u ‖ s ≠ u′ ‖ s′`: a collision (`commit_collision`). So the extraction finds a collision
whenever the values it assembled disagree, and that event is `KSLin`'s linear term at `β = 1`.

* `Opening`, `Opening.Valid`, `Disagree`: a commit string, value and salt; `commit(u; s) = c`; two openings of one
  commit string to values of one length that differ.
* `findColl_isSome`, `disagree_collision`: `findColl` over the openings' evaluations `(u ‖ s, c)` finds a collision
  whenever valid openings disagree.
* `bindFinder`, `Pr_disagree_le`: the binding finder over any coins, outputting `findColl` of the openings it collects,
  with its SHA-512 calls plus one per opening; valid openings disagree with probability at most its `Pr[collision]`.
  `Pr_disagree_le_ecr`: under `ExpectedCR` at that finder, at most its expected calls over `2^256`.
* `KSLin_of_binding`: **the reduction**. If an accepted run whose extraction fails either falls in an event bounded by
  `κ + √(γ · Pr[B])` or has valid openings that disagree, `KSLin` holds at `β = 1` with the binding finder over the
  judged run's coins and the extractor's (`β > 0`).
* `bindFinder_cost_le`, `Pr_disagree_le_bits`: with an extractor making at most `t` SHA-512 calls per rerun (as
  `SExt_calls_le` gives) and `runs(x)` reruns in expectation (`SExt_runs_exp`, `SERep_cost`), the finder's expected
  calls are at most `t · runs(x) + 𝔼[#openings]`, so under `ExpectedCR` the term is at most that over `2^256`.
-/

namespace Claims.Sampling

open Claims

/-- **Two openings of one commitment to different values of one length are a collision** of `H`. -/
theorem commit_collision {H : Msg → Msg} {u s u' s' : Msg} (hc : commit H u s = commit H u' s')
    (hl : u.length = u'.length) (hne : u ≠ u') : Collision H (u ++ s) (u' ++ s') :=
  ⟨fun h => hne (List.append_inj h hl).1, hc⟩

/-- An opening: a commit string `c`, a value `u` and a salt `s`. -/
abbrev Opening := Msg × Msg × Msg

/-- The opening is valid: `commit(u; s) = c`. -/
def Opening.Valid (H : Msg → Msg) (o : Opening) : Prop := commit H o.2.1 o.2.2 = o.1

/-- Two of the openings open one commit string to values of one length that differ. -/
def Disagree (os : List Opening) : Prop :=
  ∃ o ∈ os, ∃ o' ∈ os, o.1 = o'.1 ∧ o.2.1.length = o'.2.1.length ∧ o.2.1 ≠ o'.2.1

/-- The openings as evaluations of `H`: input `u ‖ s`, output `c`. -/
def pairsOf (os : List Opening) : List (Msg × Msg) := os.map fun o => (o.2.1 ++ o.2.2, o.1)

/-- `findColl` finds a collision whenever the list holds two different inputs with one output. -/
theorem findColl_isSome :
    ∀ l : List (Msg × Msg), (∃ p ∈ l, ∃ p' ∈ l, p.1 ≠ p'.1 ∧ p.2 = p'.2) → (findColl l).isSome := by
  intro l
  induction l with
  | nil => rintro ⟨p, hp, _⟩; simp at hp
  | cons hd tl ih =>
    rintro ⟨p, hp, p', hp', hne, heq⟩
    obtain ⟨q, a⟩ := hd
    simp only [findColl]
    cases hf : tl.find? (fun p => decide (p.1 ≠ q) && decide (p.2 = a)) with
    | some p'' => rfl
    | none =>
      apply ih
      have hno : ∀ y ∈ tl, ¬ (y.1 ≠ q ∧ y.2 = a) := by
        intro y hy hy'
        have := List.find?_eq_none.1 hf y hy
        simp [hy'.1, hy'.2] at this
      rcases List.mem_cons.1 hp with rfl | hp <;> rcases List.mem_cons.1 hp' with rfl | hp'
      · exact absurd rfl hne
      · exact absurd ⟨fun h => hne h.symm, heq.symm⟩ (hno _ hp')
      · exact absurd ⟨hne, heq⟩ (hno _ hp)
      · exact ⟨p, hp, p', hp', hne, heq⟩

/-- **Valid openings that disagree give a collision**, the first `findColl` finds among their evaluations. -/
theorem disagree_collision (H : Msg → Msg) (os : List Opening) (hv : ∀ o ∈ os, o.Valid H) (hd : Disagree os) :
    ∃ u v, findColl (pairsOf os) = some (u, v) ∧ Collision H u v := by
  have hl : ∀ p ∈ pairsOf os, H p.1 = p.2 := by
    intro p hp
    obtain ⟨o, ho, rfl⟩ := List.mem_map.1 hp
    exact hv o ho
  obtain ⟨o, ho, o', ho', hc, hlen, hne⟩ := hd
  have hs : (findColl (pairsOf os)).isSome := findColl_isSome _
    ⟨(o.2.1 ++ o.2.2, o.1), List.mem_map.2 ⟨o, ho, rfl⟩, (o'.2.1 ++ o'.2.2, o'.1), List.mem_map.2 ⟨o', ho', rfl⟩,
      fun h => hne (List.append_inj h hlen).1, hc⟩
  obtain ⟨⟨u, v⟩, huv⟩ := Option.isSome_iff_exists.1 hs
  exact ⟨u, v, huv, findColl_spec H _ hl u v huv⟩

/-- **The binding finder**: on coins `S`, collect the openings `os s` and output the first collision among their
evaluations. Its cost is the SHA-512 calls `cost s` that produce the openings, plus one per opening to evaluate it. -/
def bindFinder (S : Type) [Fintype S] [Nonempty S] (os : S → List Opening) (cost : S → ℕ) : Finder where
  S := S
  out s := findColl (pairsOf (os s))
  cost s := cost s + (os s).length

/-- **Valid openings disagree no more often than the binding finder finds a collision.** -/
theorem Pr_disagree_le (H : Msg → Msg) (S : Type) [Fintype S] [Nonempty S] (os : S → List Opening)
    (cost : S → ℕ) :
    Pr (fun s : S => (∀ o ∈ os s, o.Valid H) ∧ Disagree (os s)) ≤ (bindFinder S os cost).Pr H :=
  Pr_mono fun s ⟨hv, hd⟩ => disagree_collision H (os s) hv hd

/-- **Under `ExpectedCR` at the binding finder**, valid openings disagree with probability at most its expected
SHA-512 calls over `2^256`. -/
theorem Pr_disagree_le_ecr (H : Msg → Msg) (S : Type) [Fintype S] [Nonempty S] (os : S → List Opening)
    (cost : S → ℕ) (hecr : ExpectedCR H (bindFinder S os cost)) :
    Pr (fun s : S => (∀ o ∈ os s, o.Valid H) ∧ Disagree (os s)) ≤
      Ex (fun s : S => ((cost s + (os s).length : ℕ) : ℝ)) / 2 ^ (256 : ℝ) :=
  (Pr_disagree_le H S os cost).trans hecr

theorem Pr_union_le {α : Type} [Fintype α] (p q : α → Prop) : Pr (fun a => p a ∨ q a) ≤ Pr p + Pr q := by
  classical
  unfold Pr
  rw [← add_div]
  refine div_le_div_of_nonneg_right ?_ (Nat.cast_nonneg _)
  exact_mod_cast (Finset.card_le_card fun a ha => by
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_union] at ha ⊢; exact ha).trans
    (Finset.card_union_le _ _)

/-- **`B_ext` is `KSLin`'s linear term at `β = 1`.** If, on every prover, an accepted run whose extraction fails
either falls in an event `Q` that `κ + √(γ · Pr[B])` bounds or has valid openings `os` that disagree, then `KSLin`
holds with `β = 1` and the binding finder over the judged run's tape and coins and the extractor's. -/
theorem KSLin_of_binding {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg)
    (V : Verifier Coin) (R : Rel W) (E : Extractor Coin W V) (κ γ : Msg → ℝ)
    (os : ∀ (x : Msg) (Ω : Type), Prover Ω → Ω × (Fin (V.k x) → Coin) × E.C x → List Opening)
    (cost : ∀ (x : Msg) (Ω : Type), Prover Ω → Ω × (Fin (V.k x) → Coin) × E.C x → ℕ)
    (Q : ∀ (x : Msg) (Ω : Type), Prover Ω → Ω × (Fin (V.k x) → Coin) × E.C x → Prop)
    (hsplit : ∀ x (Ω : Type) (P : Prover Ω) (s : Ω × (Fin (V.k x) → Coin) × E.C x),
      (run H P V x s.1 s.2.1).out = .acc → ¬ R.ext x (E.eval H P x s.1 s.2.1 s.2.2).1 →
        Q x Ω P s ∨ ((∀ o ∈ os x Ω P s, o.Valid H) ∧ Disagree (os x Ω P s)))
    (hQ : ∀ x (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
      Pr (Q x Ω P) ≤ κ x + Real.sqrt (γ x * PrB H V P x)) :
    KSLin H V R E κ γ (fun _ => 1)
      (fun Ω _ _ P x => bindFinder (Ω × (Fin (V.k x) → Coin) × E.C x) (os x Ω P) (cost x Ω P)) := by
  intro x Ω _ _ P
  unfold PrBad
  calc Pr (fun s : Ω × (Fin (V.k x) → Coin) × E.C x =>
        (run H P V x s.1 s.2.1).out = .acc ∧ ¬ R.ext x (E.eval H P x s.1 s.2.1 s.2.2).1)
      ≤ Pr (fun s => Q x Ω P s ∨ ((∀ o ∈ os x Ω P s, o.Valid H) ∧ Disagree (os x Ω P s))) :=
        Pr_mono fun s ⟨h1, h2⟩ => hsplit x Ω P s h1 h2
    _ ≤ Pr (Q x Ω P) + Pr (fun s => (∀ o ∈ os x Ω P s, o.Valid H) ∧ Disagree (os x Ω P s)) := Pr_union_le _ _
    _ ≤ κ x + Real.sqrt (γ x * PrB H V P x) +
          1 * (bindFinder (Ω × (Fin (V.k x) → Coin) × E.C x) (os x Ω P) (cost x Ω P)).Pr H := by
        rw [one_mul]
        exact add_le_add (hQ x Ω P) (Pr_disagree_le H _ (os x Ω P) (cost x Ω P))

/-- **The binding finder's expected calls**: with the extractor's SHA-512 calls as its cost, at most `t` per rerun and
`runs(x)` reruns in expectation, the finder makes at most `t · runs(x) + 𝔼[#openings]` calls in expectation. -/
theorem bindFinder_cost_le {Ω Coin W : Type} [Fintype Ω] [Nonempty Ω] [Fintype Coin] [Nonempty Coin]
    [Inhabited Coin] (H : Msg → Msg) (V : Verifier Coin) (E : Extractor Coin W V) (P : Prover Ω) (x : Msg)
    (os : Ω × (Fin (V.k x) → Coin) × E.C x → List Opening) (t : ℕ) (runs : Msg → ℝ)
    (hcalls : ∀ ω r₀ c, (E.eval H P x ω r₀ c).2.1 ≤ t * (E.eval H P x ω r₀ c).2.2)
    (hruns : ExtractRunsExpLe H E runs) :
    Ex (fun s : Ω × (Fin (V.k x) → Coin) × E.C x =>
        (((E.eval H P x s.1 s.2.1 s.2.2).2.1 + (os s).length : ℕ) : ℝ)) ≤
      t * runs x + Ex (fun s => ((os s).length : ℝ)) := by
  push_cast
  rw [Ex_add]
  refine add_le_add ?_ le_rfl
  calc Ex (fun s : Ω × (Fin (V.k x) → Coin) × E.C x => ((E.eval H P x s.1 s.2.1 s.2.2).2.1 : ℝ))
      ≤ Ex (fun s : Ω × (Fin (V.k x) → Coin) × E.C x => (t : ℝ) * ((E.eval H P x s.1 s.2.1 s.2.2).2.2 : ℝ)) :=
        Ex_mono fun s => by exact_mod_cast hcalls s.1 s.2.1 s.2.2
    _ = t * Ex (fun ω : Ω => Ex (fun s : (Fin (V.k x) → Coin) × E.C x => ((E.eval H P x ω s.1 s.2).2.2 : ℝ))) := by
        rw [Ex_mul_left, Ex_prod]
    _ ≤ t * Ex (fun _ : Ω => runs x) :=
        mul_le_mul_of_nonneg_left (Ex_mono fun ω => hruns x Ω P ω) (Nat.cast_nonneg _)
    _ = t * runs x := by rw [Ex_const]

/-- **`B_ext` in bits**: under `ExpectedCR` at the binding finder whose cost is the extractor's SHA-512 calls, valid
openings disagree with probability at most `(t · runs(x) + 𝔼[#openings]) / 2^256`. -/
theorem Pr_disagree_le_bits {Ω Coin W : Type} [Fintype Ω] [Nonempty Ω] [Fintype Coin] [Nonempty Coin]
    [Inhabited Coin] (H : Msg → Msg) (V : Verifier Coin) (E : Extractor Coin W V) (P : Prover Ω) (x : Msg)
    (os : Ω × (Fin (V.k x) → Coin) × E.C x → List Opening) (t : ℕ) (runs : Msg → ℝ)
    (hcalls : ∀ ω r₀ c, (E.eval H P x ω r₀ c).2.1 ≤ t * (E.eval H P x ω r₀ c).2.2)
    (hruns : ExtractRunsExpLe H E runs)
    (hecr : ExpectedCR H (bindFinder (Ω × (Fin (V.k x) → Coin) × E.C x) os
      fun s => (E.eval H P x s.1 s.2.1 s.2.2).2.1)) :
    Pr (fun s : Ω × (Fin (V.k x) → Coin) × E.C x => (∀ o ∈ os s, o.Valid H) ∧ Disagree (os s)) ≤
      (t * runs x + Ex (fun s => ((os s).length : ℝ))) / 2 ^ (256 : ℝ) :=
  (Pr_disagree_le_ecr H _ os _ hecr).trans
    (div_le_div_of_nonneg_right (bindFinder_cost_le H V E P x os t runs hcalls hruns) (by positivity))

end Claims.Sampling
