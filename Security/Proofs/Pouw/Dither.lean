import Models.Pouw.Guarantees.Dither

/-!
# Dither's work-soundness: the freed-compute bound, the map to the floors of record, and the honest witness

* `FreedOfFloor`: the guarantee `Models.Pouw.Guarantees.Dither.FreedOfFloor`, weak duality of the period's linear
  program.
* `workFloor_iff_serviceFloorPerProver`: where the machine's cost of a run is its priced cost at `λ`, the priced floor
  `WorkFloor` is the service floor per prover of the floors of record (`ServiceFloorPerProver`), so the conclusions
  of `StagedFloorKOneSidedPerProver` and `DitherWorkSound` are one form.
* `honest_witness` (rule I): an honest prover of today's Dither round (`L = 32`, U in `±1, …, ±64`, placement
  `t mod 32`, on a `1 × 32 × 1` instance) over a one-round period meets every hypothesis of `DitherWorkSound` at `c = 1`
  and is accepted on every coin, at priced cost `k H`. The model (`flat`) prices every program at least one honest
  round, so Theorem O holds in it at `c = 1`; it witnesses that the premises hold together with an honest run, not that
  Theorem O holds in a real model (O-CHARGE, O-BITS).
-/

namespace Pouw.SecurityProofs.Dither

open Pouw.RelativeSecurity Pouw.Dither Pouw.Assumptions Finset

/-! ## Freed compute -/

/-- `Models.Pouw.Guarantees.Dither.FreedOfFloor`. -/
theorem FreedOfFloor : Pouw.Guarantees.Dither.FreedOfFloor := by
  intro G Pd W ε X μ z z' hμ hneeds hval hW hε hfit' hslack hfit hfloor
  refine ⟨W, z', hW, hfit', ?_⟩
  have h1 : ∑ j, Pd.val j * z j ≤ ∑ r, μ r * (Pd.cap r - X r) := by
    calc ∑ j, Pd.val j * z j ≤ ∑ j, (∑ r, μ r * Pd.needs j r) * z j :=
          Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_right (hval j) (hfit.1 j)
      _ = ∑ r, μ r * ∑ j, z j * Pd.needs j r := by
          simp_rw [Finset.sum_mul, Finset.mul_sum]
          rw [Finset.sum_comm]
          exact Finset.sum_congr rfl fun r _ => Finset.sum_congr rfl fun j _ => by ring
      _ ≤ ∑ r, μ r * (Pd.cap r - X r) :=
          Finset.sum_le_sum fun r _ => mul_le_mul_of_nonneg_left (by linarith [hfit.2 r]) (hμ r)
  have h2 : ∑ r, μ r * (W * Pd.hon r) ≤ ∑ r, μ r * Pd.cap r := by
    refine Finset.sum_le_sum fun r _ => mul_le_mul_of_nonneg_left ?_ (hμ r)
    have hs : 0 ≤ ∑ j, z' j * Pd.needs j r :=
      Finset.sum_nonneg fun j _ => mul_nonneg (hfit'.1 j) (hneeds j r)
    linarith [hfit'.2 r]
  have h3 : ∑ r, μ r * (Pd.cap r - X r) =
      ∑ r, μ r * (Pd.cap r - W * Pd.hon r) + (∑ r, μ r * (W * Pd.hon r) - ∑ r, μ r * X r) := by
    rw [← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun r _ => by ring
  have h4 : ε * ∑ r, μ r * (W * Pd.hon r) ≤ ε * ∑ r, μ r * Pd.cap r := mul_le_mul_of_nonneg_left h2 hε
  unfold Period.value
  nlinarith [h1, h3, h4, hfloor, hslack]

/-! ## The map to the floors of record -/

/-- Where the machine's cost of every run is its priced cost at `λ`, the priced floor is the service floor per
prover, the conclusion form of `StagedFloorKOneSidedPerProver`. -/
theorem workFloor_iff_serviceFloorPerProver (G : Accel) (S : Service) (p : G.Res → ℝ) (F : ℝ)
    (κ : ∀ (Ω : Type) [Fintype Ω], G.Mc.Alg (S.Seed × Ω) (S.Com × S.Ans) → ENNReal)
    (hc : ∀ {I O : Type} (f : G.Mc.Alg I O) (x : I), G.Mc.cost f x = G.priced p f x) :
    WorkFloor G S p F κ ↔ ServiceFloorPerProver G.Mc S F κ := by
  unfold WorkFloor ServiceFloorPerProver Machine.charge
  simp only [hc]

/-! ## The honest witness (rule I) -/

/-- Programs as functions carrying a work tag of at least one round; the machine's own costs are `0`. -/
def tagged : Machine where
  Alg I O := {g : (I → O) × ℕ // 1 ≤ g.2}
  eval f := f.1.1
  time _ := 0
  cost _ _ := 0
  size _ := 0
  seq f g := ⟨(g.1.1 ∘ f.1.1, f.1.2 + g.1.2), by have := f.2; omega⟩
  pair f g := ⟨(fun x => (f.1.1 x, g.1.1 x), f.1.2 + g.1.2), by have := f.2; omega⟩
  fst := ⟨(Prod.fst, 1), le_rfl⟩
  snd := ⟨(Prod.snd, 1), le_rfl⟩
  eval_seq _ _ _ := rfl
  eval_pair _ _ _ := rfl
  eval_fst _ := rfl
  eval_snd _ := rfl
  time_nonneg _ := le_rfl
  cost_le_time _ _ := le_rfl
  size_le_time _ := le_rfl
  size_nonneg _ := le_rfl
  time_seq_le _ _ := by norm_num
  le_time_seq _ _ := le_rfl
  time_pair_le _ _ := by norm_num
  le_time_pair_left _ _ := le_rfl
  le_time_pair_right _ _ := le_rfl
  time_fst := rfl
  time_snd := rfl

/-- One resource at rate 1; a program tagged `t` uses `t H` of it over `t H` seconds, holds nothing, imports nothing. -/
noncomputable def flat (H : ℝ) : Accel where
  Mc := tagged
  Res := Unit
  rate _ := 1
  mem := 0
  use f _ _ := H * (f.1.2 : ℝ)
  held _ _ := 0
  wall f _ := H * (f.1.2 : ℝ)
  ext _ _ := 0
  P1 := True
  P2 := True

/-- Today's Dither knobs (`L = 32`, U in `±1, …, ±64`, placement `t mod 32`) on a `1 × 32 × 1` instance. -/
abbrev K1 : Knobs where
  n := 1
  n' := 1
  nb := 1
  L := 32
  u := 64
  place t := ⟨t.val % 32, Nat.mod_lt _ (by norm_num)⟩

/-- Today's Dither round at `K1`. -/
abbrev P1 : RoundProto := ditherRound K1

/-- The committed operands: zero. -/
def a0 : P1.Inst := (fun _ _ => 0, fun _ _ => 0)

/-- The honest round program: the honest transcript, tagged one round. -/
def honRound (H : ℝ) (a : P1.Inst) : (flat H).Mc.Alg P1.Draw P1.Tr :=
  (⟨(ditherHonest K1 a, 1), le_rfl⟩ : {g : (P1.Draw → P1.Tr) × ℕ // 1 ≤ g.2})

/-- The witness service: a one-round period's live coins, the transcript as the answer, and the check on it. -/
noncomputable def S1 : Service where
  Seed := Paths (P1.Draw × P1.Chk) 1
  lam := 256
  card_seed := by
    refine (Fintype.card_congr (Equiv.prodPUnit (P1.Draw × P1.Chk))).trans ?_
    simp [ditherRound, Fintype.card_prod, Fintype.card_fin]
  Com := Unit
  Ans := P1.Tr
  Coins := Unit
  verify _ σ T := @decide (P1.ok a0 σ.1.1 σ.1.2 T) (Classical.propDecidable _)

/-- The witness analysis map: every adversary's period is cut into the honest round. -/
noncomputable def run1 (H : ℝ) : PeriodMap (flat H) S1 P1 1 :=
  fun _ _ _ _ => { play := .round a0 (honRound H a0) fun _ => .done, Bind := fun _ _ => False, ZK := fun _ _ => False }

/-- The honest prover: it commits nothing before the coins and answers with the honest transcript, tagged one round. -/
def honest (H : ℝ) : (flat H).Mc.Alg (S1.Seed × Unit) (S1.Com × S1.Ans) :=
  (⟨(fun ω => ((), ditherHonest K1 a0 ω.1.1.1), 1), le_rfl⟩ :
    {g : (S1.Seed × Unit → S1.Com × S1.Ans) × ℕ // 1 ≤ g.2})

/-- The witness price: the one resource. -/
def p1 : Unit → ℝ := fun _ => 1

theorem prob_false {Ω : Type} [Fintype Ω] : prob (fun _ : Ω => False) = 0 := by
  unfold prob
  simp

theorem avg_le_one {Ω : Type} [Fintype Ω] (f : Ω → ℝ) (h : ∀ x, f x ≤ 1) : avg f ≤ 1 := by
  unfold avg
  rcases Nat.eq_zero_or_pos (Fintype.card Ω) with h0 | hpos
  · simp [h0]
  · rw [div_le_one (by exact_mod_cast hpos)]
    calc ∑ ω, f ω ≤ ∑ _ω : Ω, (1 : ℝ) := Finset.sum_le_sum fun ω _ => h ω
      _ = Fintype.card Ω := by simp

theorem priced_flat (H : ℝ) {I O : Type} (f : (flat H).Mc.Alg I O) (x : I) :
    (flat H).priced p1 f x = H * (f.1.2 : ℝ) := by
  change ∑ _r : Unit, (1 : ℝ) * (H * (f.1.2 : ℝ) / 1) = H * (f.1.2 : ℝ)
  simp

theorem simplex_p1 (H : ℝ) : Simplex (G := flat H) p1 := by
  refine ⟨fun _ => zero_le_one, ?_⟩
  change ∑ _r : Unit, (1 : ℝ) = 1
  simp

theorem g1_flat (H : ℝ) : (flat H).G1 := by
  intro I O f x
  simp [flat, tagged]

theorem g2_flat (H : ℝ) (hH : 0 ≤ H) : (flat H).G2 := by
  refine ⟨fun _ => by simp [flat], fun f x r => ⟨?_, ?_⟩⟩
  · exact mul_nonneg hH (Nat.cast_nonneg _)
  · simp [flat]

/-- **Theorem O holds in the witness model at `c = 1`**: every round program costs at least one honest round. -/
theorem theoremO_flat (H η : ℝ) (hH : 0 < H) : TheoremO (flat H) P1 p1 1 η 0 H := by
  intro _ _ _ _ a M
  unfold roundMoment
  refine avg_le_one _ fun x => ?_
  have ht : (1 : ℝ) ≤ (M.1.2 : ℝ) := by exact_mod_cast M.2
  have hs : roundShortfall 1 0 H ((flat H).priced p1 M x.1) ((flat H).ext M x.1) = 0 := by
    rw [priced_flat]
    unfold roundShortfall
    rw [zero_mul, add_zero, mul_div_cancel_left₀ _ hH.ne']
    exact max_eq_left (by linarith)
  split_ifs <;> simp [hs]

/-- **The service is sound in the witness**, at no break and for any finders. -/
theorem serviceSound_flat (H : ℝ) (hH : 0 ≤ H) (sha : Claims.Msg → Claims.Msg) (fb fz : Finders (flat H).Mc S1 1) :
    ServiceSoundDither (flat H) S1 P1 (run1 H) (Equiv.refl _) sha fb fz 0 0 0 0 := by
  intro Ω' _ M _ r
  refine ⟨fun w => ?_, fun w => ?_, fun x _ => ⟨Or.inl ⟨rfl, trivial⟩, fun q => ?_, ?_⟩⟩
  · show prob (fun _ => False) ≤ 0 + 0 * _
    rw [prob_false]
    simp
  · show prob (fun _ => False) ≤ 0 + 0 * _
    rw [prob_false]
    simp
  · have ht : (1 : ℝ) ≤ (M.1.2 : ℝ) := by exact_mod_cast M.2
    show H * ((1 : ℕ) : ℝ) + 0 ≤ H * (M.1.2 : ℝ)
    simp only [Nat.cast_one, mul_one, add_zero]
    nlinarith
  · show (0 : ℝ) + 0 ≤ 0
    norm_num

theorem import_flat (H : ℝ) : ImportWithinCertifier (flat H) S1 0 := fun _ _ _ _ _ => le_rfl

/-- **The honest prover is accepted on every coin**, and commits first. -/
theorem honest_accepted (H : ℝ) (σ : S1.Seed) :
    S1.verify ((flat H).Mc.eval (honest H) (σ, ())).1 σ ((flat H).Mc.eval (honest H) (σ, ())).2 = true := by
  show @decide (P1.ok a0 σ.1.1 σ.1.2 (ditherHonest K1 a0 σ.1.1)) (Classical.propDecidable _) = true
  exact @decide_eq_true _ (Classical.propDecidable _) rfl

/-- **The honest witness** (rule I): at `k = 1`, `c = 1`, `β = K = 0` and no break, the honest prover of today's
Dither round meets every hypothesis of `DitherWorkSound`, commits first, is accepted on every coin, and is priced
`k H`. -/
theorem honest_witness (H η : ℝ) (hH : 0 < H) (hη : 0 ≤ η) (sha : Claims.Msg → Claims.Msg)
    (fb fz : Finders (flat H).Mc S1 1) :
    Simplex (G := flat H) p1 ∧ 0 < H ∧ 0 ≤ η ∧ (0 : ℝ) ≤ 0 ∧
      (flat H).G1 ∧ (flat H).G2 ∧ (flat H).P1 ∧ (flat H).P2 ∧ TheoremO (flat H) P1 p1 1 η 0 H ∧
      ServiceSoundDither (flat H) S1 P1 (run1 H) (Equiv.refl _) sha fb fz 0 0 0 0 ∧
      ImportWithinCertifier (flat H) S1 0 ∧
      (∀ σ σ' r, ((flat H).Mc.eval (honest H) (σ, r)).1 = ((flat H).Mc.eval (honest H) (σ', r)).1) ∧
      (∀ σ, S1.verify ((flat H).Mc.eval (honest H) (σ, ())).1 σ ((flat H).Mc.eval (honest H) (σ, ())).2 = true) ∧
      ∀ ω, (flat H).priced p1 (honest H) ω = (1 : ℕ) * H := by
  refine ⟨simplex_p1 H, hH, hη, le_rfl, g1_flat H, g2_flat H hH.le, trivial, trivial, theoremO_flat H η hH,
    serviceSound_flat H hH.le sha fb fz, import_flat H, fun _ _ _ => rfl, honest_accepted H, fun ω => ?_⟩
  rw [priced_flat]
  show H * ((1 : ℕ) : ℝ) = ((1 : ℕ) : ℝ) * H
  ring

end Pouw.SecurityProofs.Dither
