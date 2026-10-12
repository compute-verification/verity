import Models.Accounting.Guarantees.Bridges
import Proofs.Accounting.Actual.Physical
import Proofs.Accounting.Actual.Localization
import Models.Pouw.Assumptions.Deadline
import Proofs.Pouw.Deadline

/-!
# B4 and PoUW's row, proved, and the honest clock at `CM_A`

`CM_A`'s programs carry their oracle tree (`costModel_queries`), and the honest clock holds at it with the adversary
itself as the program, at every deadline from `t_d` on, with equality (`honestClock`): its cost, the work in
`[tₛ + ℓ, t_d − ℓ)`, is what the run spends after the salt (`timedRun_spentIn`). So the deadline theorem at `CM_A`, at
the deadline `t_d`, is PoUW's row from B1 (`pouwRow`). Causality is what puts the work that matters in that window: a
gate off `V` that depends on the salt and that `V` reads before `t_d` lies in it (`cost_window`,
`prover_cost_window`).

What the outside `O = (X ∪ cert)ᶜ` does during `[a, b)` is part of what it does before `b` (`acts_restrict`), which
causality's global form makes a function of what it reads from outside before `b`, its inputs being fixed. That is
what it read before `a` (fixed), what `X` sent it during `[a, b)` and what the certifiers sent it then
(`crossing_split`). `X`'s wires to `O` run on the uncertified links across `X`'s boundary (`Links`), so they are the
signals those links deliver at their ends outside `X` (`crossing_signal`). The certifiers' wires to `O` read during
`[a, b)` left them before `b` (`crossing_sent`, by causality): before `a`, a function of the certifiers' inputs and of
what they read before `a`, both fixed, or during `[a, b)`, a function of their record before `b` and of their jitter
word on `[a, b)`'s buckets (`TrustedUnits`), the record before `a` being fixed (`seen_filter_lt`). So what `O` does is
a function of the record placed during `[a, b)`, of the jitter word on those buckets and of the uncertified links'
signals during `[a, b)`, which take at most (the number of records placed in `[a, b)`) × `χ^{buckets}` ×
`2^{β°_X (b − a)}` values (`Alphabet`, `encard_range_le`).
-/

namespace Accounting.SecurityProofs

open Finset

namespace Actual

/-- A map that factors through another takes at most as many values. -/
theorem encard_range_le {Ω α β : Type*} {f : Ω → α} {g : Ω → β} (h : ∀ r r', g r = g r' → f r = f r') :
    (Set.range f).encard ≤ (Set.range g).encard := by
  rcases isEmpty_or_nonempty Ω with hΩ | ⟨⟨r₀⟩⟩
  · simp [Set.range_eq_empty]
  · have hf : Function.FactorsThrough f g := h
    refine le_trans (Set.encard_le_encard ?_) (Set.encard_image_le (Function.extend g f fun _ => f r₀) _)
    rintro _ ⟨r, rfl⟩
    exact ⟨g r, ⟨r, rfl⟩, hf.extend_apply _ r⟩

/-- What a location does during `J ⊆ I` is what it does during `I`, cut to `J`. -/
theorem acts_restrict {L : Type*} (R : Actual.Run L) {X : Finset L} {I J : Set ℕ} (hJ : J ⊆ I) :
    R.acts X J = ((R.acts X I).1 ∩ {p | p.1 ∈ J}, (R.acts X I).2.1 ∩ {p | p.2.2.1 ∈ J},
      (R.acts X I).2.2 ∩ {p | p.1 ∈ J}) := by
  simp only [Actual.Run.acts, Actual.Run.sent, Prod.mk.injEq]
  refine ⟨?_, ?_, ?_⟩ <;> ext p <;> simp only [Set.mem_inter_iff, Set.mem_ofPred_eq] <;> constructor
  · rintro ⟨g, hx, ht, rfl⟩
    exact ⟨⟨g, hx, hJ ht, rfl⟩, ht⟩
  · rintro ⟨⟨g, hx, -, rfl⟩, ht⟩
    exact ⟨g, hx, ht, rfl⟩
  · rintro ⟨g, a, hr, ha, hg, ht, rfl⟩
    exact ⟨⟨g, a, hr, ha, hg, hJ ht, rfl⟩, ht⟩
  · rintro ⟨⟨g, a, hr, ha, hg, -, rfl⟩, ht⟩
    exact ⟨g, a, hr, ha, hg, ht, rfl⟩
  · rintro ⟨v, a, hv, ha, rfl⟩
    exact ⟨⟨v, a, hJ hv, ha, rfl⟩, hv⟩
  · rintro ⟨⟨v, a, -, ha, rfl⟩, hv⟩
    exact ⟨v, a, hv, ha, rfl⟩

/-- Under causality, what crosses from `X` to `Y` during `[a, b)` is what `X` sent before `b` that arrives in `Y`
then. -/
theorem crossing_sent {L : Type*} (R : Actual.Run L) {ℓ : L → L → ℕ} (hc : R.A.Causal ℓ) {X Y : Set L}
    (hXY : Disjoint X Y) (a b : ℕ) : R.crossing X Y (Set.Ico a b) =
      (R.sent X (Set.Iio a) ∪ R.sent X (Set.Ico a b)) ∩ {p | p.1 ∈ Set.Ico a b ∧ p.2.1 ∈ Y} := by
  ext p
  simp only [Actual.Run.crossing, Actual.Run.sent, Set.mem_inter_iff, Set.mem_union, Set.mem_ofPred_eq]
  constructor
  · rintro ⟨g, s, hs, hx, hy, ht, rfl⟩
    have hgX : R.A.x g ∉ X := fun h => Set.disjoint_left.1 hXY h hy
    have hst : R.A.t s ≤ R.A.t g := (Nat.le_add_right _ _).trans (hc g s hs)
    refine ⟨?_, ht, hy⟩
    by_cases h : R.A.t s < a
    · exact Or.inl ⟨g, s, hs, hx, hgX, h, rfl⟩
    · exact Or.inr ⟨g, s, hs, hx, hgX, ⟨not_lt.1 h, hst.trans_lt ht.2⟩, rfl⟩
  · rintro ⟨⟨g, s, hs, hx, -, -, rfl⟩ | ⟨g, s, hs, hx, -, -, rfl⟩, ht, hy⟩ <;> exact ⟨g, s, hs, hx, hy, ht, rfl⟩

open Classical in
/-- The uncertified links across `X`'s boundary, the ones `β°_X` counts. -/
noncomputable def uncertified {L E : Type*} [Fintype E] (ends : E → L × L) (cert X : Finset L) : Finset E :=
  (boundary ends X).filter fun e => (ends e).1 ∉ cert ∧ (ends e).2 ∉ cert

open Classical in
theorem mem_uncertified {L E : Type*} [Fintype E] {ends : E → L × L} {cert X : Finset L} {e : E} :
    e ∈ uncertified ends cert X ↔ e ∈ boundary ends X ∧ (ends e).1 ∉ cert ∧ (ends e).2 ∉ cert :=
  mem_filter

theorem uncertifiedCapacity_eq {L E : Type*} [Fintype E] (ends : E → L × L) (β : E → ℕ) (cert X : Finset L) :
    uncertifiedCapacity ends β cert X = ∑ e ∈ uncertified ends cert X, β e := rfl

/-- Under `Links`, what crosses from `X` to the leaves in neither `X` nor `cert` is the signals the uncertified links
across `X`'s boundary deliver at their ends outside `X`: runs alike in those signals are alike in it. -/
theorem crossing_signal {L E : Type*} [Fintype E] {R R' : Actual.Run L} {ends : E → L × L} {β β' : E → ℕ}
    {link : Fin R.A.N → Fin R.A.N → E} {link' : Fin R'.A.N → Fin R'.A.N → E} (hL : R.A.Links ends β link)
    (hL' : R'.A.Links ends β' link') {cert X : Finset L} (hXc : Disjoint X cert) {I : Set ℕ}
    (h : ∀ e ∈ uncertified ends cert X, ∀ u ∈ I,
      R.signal link e (outer ends X e) u = R'.signal link' e (outer ends X e) u) :
    R.crossing ↑X (↑X ∪ ↑cert)ᶜ I ⊆ R'.crossing ↑X (↑X ∪ ↑cert)ᶜ I := by
  rintro _ ⟨g, s, hs, hx, hy, ht, rfl⟩
  have hgX : R.A.x g ∉ X := fun h' => hy (Or.inl h')
  have hgc : R.A.x g ∉ cert := fun h' => hy (Or.inr h')
  have hne : R.A.x s ≠ R.A.x g := fun h' => hgX (h' ▸ hx)
  have he := (hL.1 g s hs hne).2
  have ho : outer ends X (link g s) = R.A.x g := (inner_outer hgX hx he.symm).2
  have hu : link g s ∈ uncertified ends cert X := by
    refine mem_uncertified.2 ⟨mem_boundary hgX hx he.symm, ?_⟩
    have hsc : R.A.x s ∉ cert := Finset.disjoint_left.1 hXc hx
    rcases he with he | he <;> rw [he] <;> exact ⟨‹_›, ‹_›⟩
  have hsig : R.seen s ∈ R.signal link (link g s) (outer ends X (link g s)) (R.A.t g) :=
    ⟨g, s, hs, hne, rfl, ho.symm, rfl, rfl⟩
  rw [h _ hu _ ht] at hsig
  obtain ⟨g', s', hs', hne', hl', hy', ht', hp⟩ := hsig
  have key : R'.A.x s' = R.A.x s := by
    have he' := (hL'.1 g' s' hs' hne').2
    rw [hl'] at he'
    rcases he with he | he <;> rcases he' with he' | he' <;> rw [he] at he' <;>
      simp only [Prod.mk.injEq] at he'
    · exact he'.1.symm
    · exact absurd (he'.1.trans (hy'.trans ho)) hne
    · exact absurd (he'.2.trans (hy'.trans ho)) hne
    · exact he'.2.symm
  refine ⟨g', s', hs', ?_, ?_, ?_, ?_⟩
  · rw [key]
    exact hx
  · rw [hy', ho]
    exact hy
  · rw [ht']
    exact ht
  · rw [ht', hy', ho, hp]

/-- The gates of `F` placed before `b`, as seen, are those placed before `a` or during `[a, b)`, cut to ticks
before `b`. -/
theorem seen_filter_lt {L : Type*} (ρ : Actual.Run L) (F : Finset (Fin ρ.A.N)) (a b : ℕ) :
    ρ.seen '' ↑(F.filter fun g => ρ.A.t g < b) =
      (ρ.seen '' ↑(F.filter fun g => ρ.A.t g < a) ∪ ρ.seen '' ↑(F.filter fun g => a ≤ ρ.A.t g ∧ ρ.A.t g < b)) ∩
        {p | p.1 < b} := by
  ext p
  simp only [Set.mem_image, Finset.coe_filter, Set.mem_ofPred_eq, Set.mem_union, Set.mem_inter_iff]
  constructor
  · rintro ⟨g, ⟨hg, hb⟩, rfl⟩
    refine ⟨?_, hb⟩
    by_cases ha : ρ.A.t g < a
    · exact Or.inl ⟨g, ⟨hg, ha⟩, rfl⟩
    · exact Or.inr ⟨g, ⟨hg, not_lt.1 ha, hb⟩, rfl⟩
  · rintro ⟨⟨g, ⟨hg, -⟩, rfl⟩ | ⟨g, ⟨hg, -, -⟩, rfl⟩, hb⟩ <;> exact ⟨g, ⟨hg, hb⟩, rfl⟩

/-- No work is placed in an empty window. -/
theorem workIn_empty {L : Type*} (ρ : Actual.Run L) (π : Fin ρ.A.N → ℕ) (V : Finset L) : ρ.workIn π V ∅ = 0 := by
  simp [Actual.Run.workIn, Placed.work]

/-- `CM_A`'s programs reach the oracle only through counted queries, the hash's gates: a prover carries it. -/
theorem costModel_queries (Q R S : Type) {L : Type} [LinearOrder L] [Fintype L] (𝓕 : Set (Actual.Run L))
    (V : Finset L) (tₛ t_d ℓ : ℕ) (read : Set (ℕ × L × ℕ × L × ℕ × ℕ × ℕ) → Pouw.Transcript)
    (hash : (ρ : Actual.Run L) → Finset (Fin ρ.A.N)) (π : (ρ : Actual.Run L) → Fin ρ.A.N → ℕ) :
    (Actual.costModel Q R S 𝓕 V tₛ t_d ℓ read hash π).Queries :=
  fun P U act σ s => P.queries U act σ s

/-- **The cost window** (v2.4, B1): if what the outside of `V` can tell at `tₛ + ℓ` is the same in `ρ` and `ρ'`, and
latencies into `V` are at least `ℓ`, a gate off `V` that no gate of `ρ'` looks like (it depends on what tells the two
apart) and that `V` reads before `t_d` lies in `[tₛ + ℓ, t_d − ℓ)`. -/
theorem cost_window {L : Type*} [DecidableEq L] [Fintype L] {ρ ρ' : Actual.Run L} {V : Finset L} {tₛ t_d ℓ : ℕ}
    (ℓ' : L → L → ℕ) (hfresh : ρ.history Vᶜ (tₛ + ℓ) = ρ'.history Vᶜ (tₛ + ℓ)) (hc : ρ.A.Causal ℓ')
    (hℓ : ∀ y v, y ∉ V → v ∈ V → ℓ ≤ ℓ' y v) {g v : Fin ρ.A.N} (hg : ρ.A.x g ∉ V)
    (hdep : ∀ h, ρ'.seen h ≠ ρ.seen g) (hv : g ∈ ρ.A.reads v) (hvV : ρ.A.x v ∈ V) (hvt : ρ.A.t v < t_d) :
    ρ.A.t g ∈ Set.Ico (tₛ + ℓ) (t_d - ℓ) := by
  refine ⟨not_lt.1 fun ht => ?_, ?_⟩
  · have hb : ρ.before Vᶜ (tₛ + ℓ) = ρ'.before Vᶜ (tₛ + ℓ) := congrArg (fun p => p.2.1) hfresh
    have hm : ρ.seen g ∈ ρ.before Vᶜ (tₛ + ℓ) := ⟨g, Finset.mem_compl.2 hg, ht, rfl⟩
    rw [hb] at hm
    obtain ⟨h, -, -, hh⟩ := hm
    exact hdep h hh.symm
  · have h₁ := hc v g hv
    have h₂ := hℓ _ _ hg hvV
    omega

/-- At `CM_A`: in a prover's run, a gate off `V` that depends on the salt and that `V` reads before `t_d` lies in the
cost window, the prover's history at `tₛ + ℓ` not depending on the salt (`fresh`). -/
theorem prover_cost_window {Q R S : Type} {L : Type} [LinearOrder L] [Fintype L] {𝓕 : Set (Actual.Run L)}
    {V : Finset L} {tₛ t_d ℓ : ℕ} {read hash} (P : Actual.Prover Q R S 𝓕 V tₛ t_d ℓ read hash) (H U act σ)
    (s s' : S) (ℓ' : L → L → ℕ) (hc : (P.strategy (H, U, act, σ, s)).A.Causal ℓ')
    (hℓ : ∀ y v, y ∉ V → v ∈ V → ℓ ≤ ℓ' y v) {g v : Fin (P.strategy (H, U, act, σ, s)).A.N}
    (hg : (P.strategy (H, U, act, σ, s)).A.x g ∉ V)
    (hdep : ∀ h, (P.strategy (H, U, act, σ, s')).seen h ≠ (P.strategy (H, U, act, σ, s)).seen g)
    (hv : g ∈ (P.strategy (H, U, act, σ, s)).A.reads v) (hvV : (P.strategy (H, U, act, σ, s)).A.x v ∈ V)
    (hvt : (P.strategy (H, U, act, σ, s)).A.t v < t_d) :
    (P.strategy (H, U, act, σ, s)).A.t g ∈ Set.Ico (tₛ + ℓ) (t_d - ℓ) :=
  cost_window ℓ' (P.fresh H U act σ s s') hc hℓ hg hdep hv hvV hvt

/-- What an adversary's timed run spends from the salt's release to a deadline `td` at or after `t_d` is its cost at
`CM_A`, the prover's work in `[tₛ + ℓ, t_d − ℓ)`. -/
theorem timedRun_spentIn {Q R S : Type} {L : Type} [LinearOrder L] [Fintype L] {𝓕 : Set (Actual.Run L)}
    {V : Finset L} {tₛ t_d ℓ : ℕ} {read : Set (ℕ × L × ℕ × L × ℕ × ℕ × ℕ) → Pouw.Transcript}
    {hash : (ρ : Actual.Run L) → Finset (Fin ρ.A.N)} {π : (ρ : Actual.Run L) → Fin ρ.A.N → ℕ} (U : Pouw.Workload)
    (A : Pouw.Adversary (Actual.costModel Q R S 𝓕 V tₛ t_d ℓ read hash π)) (H : Q → R) (s : S) {td : ℝ}
    (htd : (t_d : ℝ) ≤ td) :
    (Actual.timedRun A U).spentIn H s tₛ td =
      (Actual.costModel Q R S 𝓕 V tₛ t_d ℓ read hash π).cost A.online H U A.act (A.pre.eval H) s := by
  have h₁ : {u : ℕ | (u : ℝ) < td} ∩ Set.Ico (tₛ + ℓ) (t_d - ℓ) = Set.Ico (tₛ + ℓ) (t_d - ℓ) :=
    Set.inter_eq_right.2 fun u hu => lt_of_lt_of_le (Nat.cast_lt.2 (hu.2.trans_le (Nat.sub_le _ _))) htd
  have h₀ : {u : ℕ | (u : ℝ) < tₛ} ∩ Set.Ico (tₛ + ℓ) (t_d - ℓ) = ∅ :=
    Set.subset_empty_iff.1 fun u hu =>
      absurd (Nat.cast_lt.1 (show (u : ℝ) < tₛ from hu.1)) (not_lt.2 ((Nat.le_add_right _ _).trans hu.2.1))
  simp only [Actual.costModel, Actual.timedRun, Pouw.TimedRun.spentIn, h₁, h₀, workIn_empty, Nat.cast_zero,
    sub_zero]

/-- **The honest clock at `CM_A`**: for an adversary's timed run, at a deadline `td` at or after `t_d`, the adversary
itself is the program `HonestClock` asks for, with equality: its cost, the prover's work in `[tₛ + ℓ, t_d − ℓ)`, is
what the run spends in `(tₛ, td]` (`timedRun_spentIn`). -/
theorem honestClock {Q R S : Type} {L : Type} [LinearOrder L] [Fintype L] {𝓕 : Set (Actual.Run L)} {V : Finset L}
    {tₛ t_d ℓ : ℕ} {read : Set (ℕ × L × ℕ × L × ℕ × ℕ × ℕ) → Pouw.Transcript}
    {hash : (ρ : Actual.Run L) → Finset (Fin ρ.A.N)} {π : (ρ : Actual.Run L) → Fin ρ.A.N → ℕ}
    (U : Pouw.Workload) (A : Pouw.Adversary (Actual.costModel Q R S 𝓕 V tₛ t_d ℓ read hash π)) {td : ℝ}
    (htd : (t_d : ℝ) ≤ td) :
    Pouw.Assumptions.HonestClock (Actual.costModel Q R S 𝓕 V tₛ t_d ℓ read hash π) U A (Actual.timedRun A U) td :=
  ⟨A, rfl, fun H s => ⟨fun _ => rfl, (timedRun_spentIn U A H s htd).ge, le_rfl⟩⟩

end Actual

/-- **PoUW's row, from B1**: the deadline theorem at `CM_A` with its honest clock at the deadline `t_d`, where what the
timed run spends after the salt is the cost and the transcript, received at `t_d`, is in by then. -/
theorem pouwRow : Guarantees.pouwRow := by
  intro Q R S _ _ _ _ L _ _ 𝓕 V tₛ t_d ℓ read hash π P D γ₀ Ωmax ε hγ hΩ hW hB1 U hU A T hT q hq hA t εs hεs
  have h := Pouw.SecurityProofs.WorkBetweenSaltAndDeadline Q R S _ P D γ₀ Ωmax ε hγ hΩ hW hB1 U hU A
    (Actual.timedRun A U) t_d (Actual.honestClock U A le_rfl) T hT q hq
    (fun H s => ⟨(Actual.timedRun_spentIn U A H s le_rfl).trans_le (hA H s).1, (hA H s).2⟩) t εs hεs
  simpa only [Pouw.AcceptsBy, Actual.timedRun, le_refl, true_and] using h

theorem B4 : Guarantees.B4 := by
  intro L E Ω _ _ _ S ℓ ends β link 𝒱 cert X spec record bucket jitter a b hc hα hT hXc hh hr
  obtain ⟨-, -, J, hJ⟩ := hT
  set O : Finset L := (X ∪ cert)ᶜ with hO
  have hOc : (↑O : Set L) = (↑X ∪ ↑cert)ᶜ := by rw [hO, coe_compl, coe_union]
  have hXO : Disjoint X O := Finset.disjoint_left.2 fun y hy hy' => (mem_compl.1 hy') (mem_union_left _ hy)
  have hcO : Disjoint (↑cert : Set L) ↑O := by
    rw [hOc]
    exact disjoint_compl_right.mono_left Set.subset_union_right
  have hrest : ((↑X : Set L) ∪ ↑O)ᶜ = ↑cert := by
    ext y
    by_cases hy : y ∈ X
    · simp [hO, hy, Finset.disjoint_left.1 hXc hy]
    · simp [hO, hy]
  let K := (Finset.Ico a b).image bucket
  let B := Actual.uncertified ends cert X
  let sig : ↥B × ↥(Finset.Ico a b) → Ω → Set (ℕ × L × ℕ × ℕ × ℕ) := fun i r =>
    (S r).signal (link r) i.1 (Actual.outer ends X i.1) i.2
  let rec_ : Ω → Set (ℕ × L × ℕ × ℕ × ℕ) := fun r =>
    (S r).seen '' ↑((record r).filter fun g => a ≤ (S r).A.t g ∧ (S r).A.t g < b)
  have hfac : ∀ r r', (rec_ r, (fun k : ↥K => J r k), fun i => sig i r) =
      (rec_ r', (fun k : ↥K => J r' k), fun i => sig i r') →
      (S r).acts O (Set.Ico a b) = (S r').acts O (Set.Ico a b) := by
    intro r r' hΦ
    simp only [Prod.mk.injEq] at hΦ
    obtain ⟨hrec, hj, hsig⟩ := hΦ
    have hcin : ∀ r r', (S r).inputs ↑cert = (S r').inputs ↑cert := fun r r' => congrArg Prod.fst (hh r r').1
    have hcpre : (S r).crossing (↑cert)ᶜ ↑cert (Set.Iio a) = (S r').crossing (↑cert)ᶜ ↑cert (Set.Iio a) :=
      congrArg (fun p => p.2.2) (hh r r').1
    have hs0 : (S r).sent ↑cert (Set.Iio a) = (S r').sent ↑cert (Set.Iio a) :=
      congrArg (fun p => p.2.1) (Actual.global_of_causal hc cert hcin a r r' hcpre)
    have hs1 : (S r).sent ↑cert (Set.Ico a b) = (S r').sent ↑cert (Set.Ico a b) :=
      hJ a b r r' (by
          rw [Actual.seen_filter_lt (S r) _ a b, Actual.seen_filter_lt (S r') _ a b]
          exact congrArg (· ∩ {p | p.1 < b}) (congrArg₂ (· ∪ ·) (hh r r').2.2 hrec))
        fun t ht => congrFun hj ⟨bucket t, mem_image_of_mem _ (Finset.mem_Ico.2 ht)⟩
    have hcert : (S r).crossing ↑cert ↑O (Set.Ico a b) = (S r').crossing ↑cert ↑O (Set.Ico a b) := by
      rw [Actual.crossing_sent (S r) (hr r).1 hcO, Actual.crossing_sent (S r') (hr r').1 hcO, hs0, hs1]
    have hX : (S r).crossing ↑X ↑O (Set.Ico a b) = (S r').crossing ↑X ↑O (Set.Ico a b) := by
      rw [hOc]
      exact (Actual.crossing_signal (hr r).2 (hr r').2 hXc fun e he u hu =>
          congrFun hsig ⟨⟨e, he⟩, ⟨u, Finset.mem_Ico.2 hu⟩⟩).antisymm
        (Actual.crossing_signal (hr r').2 (hr r).2 hXc fun e he u hu =>
          (congrFun hsig ⟨⟨e, he⟩, ⟨u, Finset.mem_Ico.2 hu⟩⟩).symm)
    have hOin : ∀ r r', (S r).inputs ↑O = (S r').inputs ↑O := fun r r' => congrArg Prod.fst (hh r r').2.1
    have hOpre : (S r).crossing (↑O)ᶜ ↑O (Set.Iio a) = (S r').crossing (↑O)ᶜ ↑O (Set.Iio a) :=
      congrArg (fun p => p.2.2) (hh r r').2.1
    have hall : (S r).acts O (Set.Iio b) = (S r').acts O (Set.Iio b) := by
      refine Actual.global_of_causal hc O hOin b r r' ?_
      rw [Actual.crossing_split (S r) hXO (t₀ := a) (le_refl b),
        Actual.crossing_split (S r') hXO (t₀ := a) (le_refl b), hrest, hOpre, hX, hcert]
    rw [Actual.acts_restrict (S r) (I := Set.Iio b) Set.Ico_subset_Iio_self,
      Actual.acts_restrict (S r') (I := Set.Iio b) Set.Ico_subset_Iio_self, hall]
  refine (Actual.encard_range_le hfac).trans ?_
  have hsub : (Set.range fun r => (rec_ r, (fun k : ↥K => J r k), fun i => sig i r)) ⊆
      (Set.range rec_) ×ˢ
        ((Set.univ : Set (↥K → Fin (2 ^ jitter))) ×ˢ Set.univ.pi fun i => Set.range (sig i)) := by
    rintro _ ⟨r, rfl⟩
    exact ⟨⟨r, rfl⟩, Set.mem_univ _, fun i _ => ⟨r, rfl⟩⟩
  refine (Set.encard_le_encard hsub).trans ?_
  rw [Set.encard_prod, Set.encard_prod, Set.encard_pi_eq_prod_encard, Set.encard_univ,
    ENat.card_eq_coe_fintype_card, Fintype.card_fun, Fintype.card_fin, Fintype.card_coe]
  have hsig : ∏ i, (Set.range (sig i)).encard ≤
      ((2 ^ (uncertifiedCapacity ends β cert X * (b - a)) : ℕ) : ℕ∞) := by
    refine (Finset.prod_le_prod fun i _ => hα i.1.1 (Actual.outer ends X i.1.1) i.2.1).trans (le_of_eq ?_)
    rw [← Nat.cast_prod, Finset.prod_pow_eq_pow_sum, Fintype.sum_prod_type, Actual.uncertifiedCapacity_eq,
      ← Finset.sum_coe_sort B, Finset.sum_mul]
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_coe, Nat.card_Ico, smul_eq_mul, mul_comm]
  calc _ ≤ (Set.range rec_).encard *
        ((((2 ^ jitter) ^ K.card : ℕ) : ℕ∞) * ((2 ^ (uncertifiedCapacity ends β cert X * (b - a)) : ℕ) : ℕ∞)) := by
        gcongr
    _ = _ := by rw [← Nat.cast_mul, ← pow_mul, ← pow_add]

end Accounting.SecurityProofs
