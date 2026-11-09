module

public import Proofs.WeightSecurity.Lemmas
public import Proofs.Pouw.Prob

/-! `WeightSecurity.SecurityProofs.Theft : WeightSecurity.Guarantees.Theft`, the leakage lemma. For each covert string
`s`, the outsider with `s` written in (`Outsider.withString`) rebuilds, from honest answers, exactly the view the
outsider has against the datacenter deviating through the channel with `s` (`rebuild_view`), so it outputs the same
model (`withString_model`). Outside `LeaksThrough`'s `ε`, the real view is one of those, so the outsider steals only
where some honest-use party `O.withString ch s` does: at most `|S| · p` by the union bound and `HonestUseHard`.

The witnesses (`TheftHonest`, `TheftNontrivial`) and the tightness lemma (`TheftTight`) are proved at the toy instance:
honest use reveals the first weight bit and nothing of the second, so any output built from the first bit is the weights
on at most half the coins (`Toy.guess_le`). -/

@[expose] public section

namespace WeightSecurity.SecurityProofs

open Finset WeightSecurity WeightSecurity.Assumptions

variable {R X Y M Θ Θ' : Type}

/-- **The rebuilt view is the channel's view**: the outsider with `s` written in, run against honest use, rebuilds after
`n` rounds exactly the outsider's view against the datacenter that deviates through `ch` with `s`. -/
theorem rebuild_view (O : Outsider R X M Θ') (ch : Channel R X Y M) (C : Θ → X → Y) (s : ch.S) (θ : Θ) (r : R) :
    ∀ n, O.rebuild ch s r ((O.withString ch s).view r (honest C θ) n) = O.view r (ch.resp C s θ r) n
  | 0 => rfl
  | n + 1 => by
    have query : ∀ h, (O.withString ch s).query r h = ch.ask s r (O.rebuild ch s r h) (O.query r (O.rebuild ch s r h)) :=
      fun _ => rfl
    simp only [Outsider.view, Outsider.rebuild, honest, Channel.resp, query, rebuild_view O ch C s θ r n]

/-- **The outsider with `s` written in outputs the outsider's model** against the channel's datacenter with `s`. -/
theorem withString_model (O : Outsider R X M Θ') (ch : Channel R X Y M) (C : Θ → X → Y) (s : ch.S) (θ : Θ) (r : R)
    (q : ℕ) : (O.withString ch s).model r (honest C θ) q = O.model r (ch.resp C s θ r) q :=
  congrArg (O.out r) (rebuild_view O ch C s θ r q)

/-- **Model weight security** (`WeightSecurity.Guarantees.Theft`). -/
theorem Theft : Guarantees.Theft := by
  intro Ω R Θ X Y M Θ' _ _ C gen Stolen q A p real O ch ε hhon hleak hsim
  classical
  have split : ∀ z : Ω × R, Stolen (gen z.1) (O.model z.2 (real (gen z.1) z.2) q) →
      (¬ ∃ s, O.view z.2 (real (gen z.1) z.2) q = O.view z.2 (ch.resp C s (gen z.1) z.2) q) ∨
        ∃ s ∈ (univ : Finset ch.S), Stolen (gen z.1) ((O.withString ch s).model z.2 (honest C (gen z.1)) q) := by
    intro z hz
    by_cases h : ∃ s, O.view z.2 (real (gen z.1) z.2) q = O.view z.2 (ch.resp C s (gen z.1) z.2) q
    · obtain ⟨s, hs⟩ := h
      refine Or.inr ⟨s, mem_univ s, ?_⟩
      rw [withString_model, Outsider.model, ← hs]
      exact hz
    · exact Or.inl h
  calc Claims.Pr (fun z : Ω × R => Stolen (gen z.1) (O.model z.2 (real (gen z.1) z.2) q))
      ≤ Pouw.pr (fun z : Ω × R =>
          (¬ ∃ s, O.view z.2 (real (gen z.1) z.2) q = O.view z.2 (ch.resp C s (gen z.1) z.2) q) ∨
            ∃ s ∈ (univ : Finset ch.S),
              Stolen (gen z.1) ((O.withString ch s).model z.2 (honest C (gen z.1)) q)) := Pouw.pr_mono split
    _ ≤ ε + ∑ s : ch.S, Pouw.pr (fun z : Ω × R =>
          Stolen (gen z.1) ((O.withString ch s).model z.2 (honest C (gen z.1)) q)) :=
        (Pouw.pr_or_le _ _).trans (add_le_add hleak (Pouw.Proofs.pr_exists_le_sum _ _))
    _ ≤ ε + ∑ _s : ch.S, p := add_le_add le_rfl (sum_le_sum fun s _ => hhon _ (hsim s))
    _ = Fintype.card ch.S * p + ε := by rw [sum_const, card_univ, nsmul_eq_mul]; ring

end WeightSecurity.SecurityProofs

namespace WeightSecurity.Toy

open Finset WeightSecurity WeightSecurity.Assumptions

/-- A coin of the toy has probability `1/4`. -/
theorem pr_point (θ : Bool × Bool) : Pouw.pr (fun z : (Bool × Bool) × Unit => z.1 = θ) = 4⁻¹ := by
  rw [Pouw.pr_eq_card _ {(θ, ())} fun z => by simp [Prod.ext_iff]]
  simp

/-- **Honest use hides the second bit**: a model built from the first weight bit alone is the weights on at most half
the coins. -/
theorem guess_le (f : Bool → Bool × Bool) :
    Claims.Pr (fun z : (Bool × Bool) × Unit => Stolen (gen z.1) (f z.1.1)) ≤ 2⁻¹ := by
  calc Claims.Pr (fun z : (Bool × Bool) × Unit => Stolen (gen z.1) (f z.1.1))
      ≤ Pouw.pr (fun z : (Bool × Bool) × Unit => z.1 = f false ∨ z.1 = f true) := Pouw.pr_mono fun z h => by
        obtain ⟨⟨a, b⟩, ⟨⟩⟩ := z
        cases a
        · exact Or.inl h.symm
        · exact Or.inr h.symm
    _ ≤ _ := Pouw.pr_or_le _ _
    _ = 2⁻¹ := by rw [pr_point, pr_point]; norm_num

/-- `HonestUseHard` at the toy: one query, every party, `p = 1/2`. -/
theorem hard : HonestUseHard C gen Stolen 1 anyone 2⁻¹ := fun P _ =>
  guess_le fun b => P.out () [(P.query () [], b)]

/-- The thief steals the weights from the cheating datacenter on every coin. -/
theorem thief_steals : Claims.Pr (fun z : (Bool × Bool) × Unit =>
    Stolen (gen z.1) (thief.model z.2 (cheatDC (gen z.1) z.2) 1)) = 1 :=
  Pouw.pr_eq_one fun _ => rfl

end WeightSecurity.Toy

namespace WeightSecurity.SecurityProofs

open WeightSecurity WeightSecurity.Assumptions

/-- `Theft`'s premises hold together at the honest toy datacenter (`WeightSecurity.Guarantees.TheftHonest`). -/
theorem TheftHonest : Guarantees.TheftHonest :=
  ⟨Toy.hard, le_of_eq (Pouw.pr_eq_zero fun _ h => h ⟨(), rfl⟩), fun _ => Set.mem_univ _⟩

/-- `Theft`'s bound fails once `LeaksThrough` does (`WeightSecurity.Guarantees.TheftNontrivial`). -/
theorem TheftNontrivial : Guarantees.TheftNontrivial := by
  refine ⟨Toy.hard, fun _ => Set.mem_univ _, fun h => ?_, ?_⟩
  · refine absurd (le_trans (Pouw.pr_mono fun z (hz : z.1 = (false, true)) => ?_) h) ?_
    · obtain ⟨θ, ⟨⟩⟩ := z
      obtain rfl : θ = (false, true) := hz
      rintro ⟨⟨⟩, h⟩
      exact absurd h (by decide)
    · rw [Toy.pr_point]
      norm_num
  · rw [Toy.thief_steals]
    norm_num [show Fintype.card Toy.silent.S = 1 from rfl]

/-- `Theft`'s factor `|S|` is attained (`WeightSecurity.Guarantees.TheftTight`). -/
theorem TheftTight : Guarantees.TheftTight := by
  refine ⟨le_of_eq (Pouw.pr_eq_zero fun z h => h ⟨z.1.2, rfl⟩), ?_⟩
  rw [Toy.thief_steals]
  norm_num [show Fintype.card Toy.oneBit.S = 2 from rfl]

end WeightSecurity.SecurityProofs
