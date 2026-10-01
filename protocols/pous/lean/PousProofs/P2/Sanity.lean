import PousProofs.P2.Security

/-!
# P2's bridges can hold, and SGI has content (satisfiability witnesses)

* `idealAudit_bridges`: in `idealAudit sch S Q`, the ideal model's own sequential game taken as the deployed audit,
  P2-EXP-IO, SeqRoot and P2–M1-SGI hold at once, with `ε = 0` and `S' = S`.
* `not_sgi_freeRoots_b19`: in `freeRoots`, a deployment where a strategy that stores nothing passes every audit (as a
  prover that computes roots for free would), P2–M1-SGI fails at the MVP's parameters. So SGI is not true of every
  audit: it is what rules such a deployment out.
-/

namespace Pous.P2

open scoped ENNReal

variable {M : IdealModel}

/-- The ideal model's sequential game as a deployed audit: a strategy is an `S`-bit preprocessing and a sequential
responder with `Q` queries per answer, at some depth, and the deployed and uniform keys are the ideal primitive. -/
noncomputable def idealAudit (sch : Scheme M) (S Q : ℕ) : RealAudit sch where
  Strategy := Σ D : ℕ, Preprocessor sch S × SeqResponder sch S D Q
  Bounded S' _ := S' = S
  Forward Q' _ := Q' = Q
  accept k W s := pr (seqAccepts sch k W s.2.1 s.2.2)
  acceptIdeal k W s := pr (seqAccepts sch k W s.2.1 s.2.2)

theorem idealAudit_bridges (sch : Scheme M) (S Q : ℕ) :
    ExpIO (idealAudit sch S Q) S 0 ∧ SeqRoot (idealAudit sch S Q) S Q ∧ SGI (idealAudit sch S Q) S S Q 0 := by
  refine ⟨fun k W s _ => by rw [add_zero]; exact le_rfl, fun _ _ => rfl, fun k W s _ _ => ?_⟩
  obtain ⟨D, A₁, A₂⟩ := s
  exact ⟨D, A₁, A₂, by rw [add_zero]; exact le_rfl⟩

/-- A deployment where every strategy is within the bounds and forward-only, and passes every audit: what a prover
that computes roots for free would see. -/
def freeRoots (sch : Scheme M) : RealAudit sch where
  Strategy := Unit
  Bounded _ _ := True
  Forward _ _ := True
  accept _ _ _ := 1
  acceptIdeal _ _ _ := 1

/-- Where the ideal model is secure with room to spare (`δ + ε + ε' < 1`), SGI fails for `freeRoots`. -/
theorem not_sgi_freeRoots {sch : Scheme M} {S S' Q k : ℕ} {δ ε ε' : ℝ≥0∞}
    (hsec : ∀ D, AuditSecure sch .sequential S' D Q k δ ε) (hlt : δ + ε + ε' < 1) :
    ¬ SGI (freeRoots sch) S S' Q ε' := by
  intro h
  obtain ⟨D, A₁, A₂, h1⟩ := h k (fun _ => false) () trivial trivial
  have h2 : (1 : ℝ≥0∞) ≤ δ + ε + ε' := h1.trans (by gcongr; exact hsec D _ A₁ A₂)
  exact absurd (h2.trans_lt hlt) (lt_irrefl 1)

theorem delta_add_eps_add_eps_lt_one : ENNReal.ofReal Params.δ + Params.εMax + Params.εMax < 1 := by
  have h2 : (2⁻¹ : ℝ≥0∞) = ENNReal.ofReal (1 / 2) := by
    rw [ENNReal.ofReal_div_of_pos two_pos, ENNReal.ofReal_one, ENNReal.ofReal_ofNat, one_div]
  have hε : Params.εMax = ENNReal.ofReal ((1 / 2) ^ 128) := by
    rw [Params.εMax, Params.secParam, h2, ENNReal.ofReal_pow (by norm_num)]
  rw [hε, ← ENNReal.ofReal_add (by norm_num [Params.δ]) (by positivity),
    ← ENNReal.ofReal_add (by norm_num [Params.δ]) (by positivity), ENNReal.ofReal_lt_one]
  norm_num [Params.δ]

/-- **SGI has content at the MVP's parameters.** For every payload width in range, P2–M1-SGI onto `⌊ρ·|C|⌋` bits
with `2^20` forward evaluations per answer and `ε_SGI = 2^-128` fails for `freeRoots`. -/
theorem not_sgi_freeRoots_b19 (b : ℕ) (hsp : 20 * 16448 ≤ 21 * b) (hb : b ≤ 16447) (S : ℕ) :
    ¬ SGI (freeRoots (m1p (2 ^ 19) 16448 (2 ^ 16448 - 21065) b)) S
      ((m1p (2 ^ 19) 16448 (2 ^ 16448 - 21065) b).stateBound Params.ρ) (2 ^ 20) Params.εMax :=
  not_sgi_freeRoots (k := 88) (fun D => (PousP2p.p2_meets_m1p_b19 b hsp hb D).2.2.2.2)
    delta_add_eps_add_eps_lt_one

end Pous.P2
