module

public import Proofs.Flock.SpotFlock.Honest
public import Proofs.Flock.SpotFlock.KS
import Mathlib.Analysis.Complex.ExponentialBounds

/-! SPOT[Flock] meets `SampledKSTwo`'s premises at a knowledge error below `1`, and accepts where Flock does
(`accepting`). -/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling

variable (h : Held)

theorem ref_numbers (t : ℕ) : 1 ≤ (shapeOf (h.xS₀ t)).length ∧ (shapeOf (h.xS₀ t)).length ≤ 200 ∧
    (2 : ℝ) ^ 10 ≤ n₀ h t ∧ n₀ h t ≤ ((shapeOf (h.xS₀ t)).length : ℝ) * 2 ^ 23 := by
  obtain ⟨s, spec, sts, hset, h1, h2, -, h4⟩ := zkOk_sound (h.xS₀_ok t)
  have hlen : (shapeOf (h.xS₀ t)).length = sts.size := by simp [shapeOf, hset]
  have hN : FlockDeploy.N₀ true (h.xS₀ t) = n₀ h t := by
    unfold FlockDeploy.N₀ n₀ shapeOf
    rw [FlockDeploy.tables_eq hset, hset]
    simp [FlockDeploy.tableLen, List.map_map, Function.comp_def]
  obtain ⟨hlo, hhi⟩ := FlockDeploy.N₀_le hset h1 fun st hst =>
    FlockDeploy.tableLen_zk_bounds st (h4 st hst).1 (h4 st hst).2.1
  rw [← hN, hlen]
  exact ⟨Nat.succ_le_of_lt h1, h2, hlo, hhi⟩

theorem tables_le (x : Msg) : (code h).tables x ≤ 200 := by
  have hfold : ∀ l : List ℕ, (∀ a ∈ l, a ≤ 200) → l.foldr max 0 ≤ 200 := by
    intro l hl
    induction l with
    | nil => simp
    | cons a l ih =>
      simp only [List.foldr_cons]
      exact max_le (hl a List.mem_cons_self) (ih fun b hb => hl b (List.mem_cons_of_mem a hb))
  refine hfold ((List.range (h.ms.length + 1)).map fun t => (shapeOf (h.xS₀ t)).length) fun a ha => ?_
  obtain ⟨t, -, rfl⟩ := List.mem_map.1 ha
  exact (ref_numbers h t).2.1

/-- `ref_numbers`'s `200` tables are Flock's masked scope's (`zkOk`): the two-stage verifier refuses every run of a
template with more than one. -/
theorem ref_one_table {s : AuditStatement} {d : Design.Two} (hd : two h s = some d) {t : ℕ}
    (ht : t < h.ms.length) :
    (shapeOf (h.xS₀ t)).length = 1 ∧ n₀ h t ≤ 2 ^ 23 ∧ ⌈4 * n₀ h t / Real.exp 1⌉₊ ≤ 12343986 := by
  have h1 : (shapeOf (h.xS₀ t)).length = 1 := by
    have hok := templateOk_of_two h hd ht
    unfold templateOk at hok
    cases hr : FlockDeploy.readOf true (h.xS₀ t) with
    | none => rw [hr] at hok; cases hok
    | some r =>
      rw [hr] at hok
      simp only [Bool.and_eq_true, beq_iff_eq] at hok
      exact hok.1.2
  have hN : n₀ h t ≤ 2 ^ 23 := by
    have hle := (ref_numbers h t).2.2.2
    rw [h1] at hle
    simpa using hle
  refine ⟨h1, hN, ?_⟩
  have he0 : 0 < Real.exp 1 := Real.exp_pos 1
  have hlo := Real.exp_one_gt_d9
  rw [Nat.ceil_le, div_le_iff₀ he0]
  push_cast
  linarith

theorem inv_two_pow_eq (k : ℕ) : (2 : ℝ) ^ (-(k : ℝ)) = 2⁻¹ ^ k := by
  rw [Real.rpow_neg (by norm_num), Real.rpow_natCast, inv_pow]

section Numbers

variable {T : Type} (F : Claims.Flock.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) T)

theorem κF_le (x : Msg) (t : ℕ) :
    Claims.SamplingGeneric.κFAt (SamplingDeploy.deployOf F (code h)) x (n₀ h t) ≤ 2⁻¹ ^ 190 := by
  obtain ⟨hn1, hn200, hN, hN'⟩ := ref_numbers h t
  have hn : ((shapeOf (h.xS₀ t)).length : ℝ) ≤ 200 := by exact_mod_cast hn200
  have hk1 : 1507 ≤ ⌈4 * n₀ h t / Real.exp 1⌉₊ := Claims.Flock.Numbers.k_ge hN
  have hk2 : (⌈4 * n₀ h t / Real.exp 1⌉₊ : ℝ) ≤ 200 * 2 ^ 24 :=
    (Claims.Flock.Numbers.k_le_mul (by exact_mod_cast hn1) (le_trans (by positivity) hN) hN').trans (by nlinarith)
  have he : Real.exp (-(⌈4 * n₀ h t / Real.exp 1⌉₊ : ℝ) / 8) ≤ 2⁻¹ ^ 250 := by
    have := Claims.Flock.Numbers.exp_neg_k_le hk1
    rwa [show (-(250 : ℝ)) = -((250 : ℕ) : ℝ) by norm_num, inv_two_pow_eq] at this
  have e205 : (2 : ℝ) ^ (-(205 : ℝ)) = 2⁻¹ ^ 205 := by
    rw [show (-(205 : ℝ)) = -((205 : ℕ) : ℝ) by norm_num, inv_two_pow_eq]
  have hε : (SamplingDeploy.deployOf F (code h)).εtable x = ((code h).tables x : ℝ) * 2 ^ (-(205 : ℝ)) := rfl
  have hw : (2 : ℝ) ^ (-((SamplingDeploy.deployOf F (code h)).innerWaits x : ℝ)) = 2⁻¹ ^ 210 :=
    inv_two_pow_eq 210
  have hcap : ((SamplingDeploy.deployOf F (code h)).innerCap x : ℝ) = 2 ^ 256 := by
    show ((Claims.Flock.Numbers.cap : ℕ) : ℝ) = 2 ^ 256
    norm_num [Claims.Flock.Numbers.cap]
  have h1 : 4 * (((code h).tables x : ℝ) * 2⁻¹ ^ 205) ≤ 4 * (200 * 2⁻¹ ^ 205) := by
    have : ((code h).tables x : ℝ) ≤ 200 := by exact_mod_cast tables_le h x
    gcongr
  have h3 : 2 * (⌈4 * n₀ h t / Real.exp 1⌉₊ : ℝ) / 2 ^ 256 ≤ 2 * (200 * 2 ^ 24) / 2 ^ 256 := by
    gcongr
  have hsum : 4 * (200 * (2 : ℝ)⁻¹ ^ 205) + 2⁻¹ ^ 210 + 2 * (200 * 2 ^ 24) / 2 ^ 256 + 2⁻¹ ^ 250 ≤ 2⁻¹ ^ 190 := by
    norm_num
  rw [Claims.SamplingGeneric.κFAt_eq, hε, e205, hw, hcap]
  linarith

end Numbers

theorem slack_le_one_add : Claims.Sampling.slack ≤ 1 + 2⁻¹ ^ slackExp := by
  have ej : (2 : ℝ) ^ (-(slackExp : ℝ)) = 2⁻¹ ^ slackExp := inv_two_pow_eq slackExp
  have h := rpow_one_add_le_one_add_mul_self (s := 1) (by norm_num) (p := (2 : ℝ) ^ (-(slackExp : ℝ)))
    (by positivity) (by rw [ej]; exact pow_le_one₀ (by norm_num) (by norm_num))
  calc Claims.Sampling.slack = (1 + 1 : ℝ) ^ ((2 : ℝ) ^ (-(slackExp : ℝ))) := by
        simp only [Claims.Sampling.slack, one_add_one_eq_two]
    _ ≤ 1 + (2 : ℝ) ^ (-(slackExp : ℝ)) * 1 := h
    _ = 1 + 2⁻¹ ^ slackExp := by rw [ej, mul_one]

/-- Squaring `slack_le_one_add` instead leaves `2^{−2j}` over. -/
theorem slack_sq_le_one_add : Claims.Sampling.slack ^ 2 ≤ 1 + 2 * 2⁻¹ ^ slackExp := by
  have ej : (2 : ℝ) ^ (-(slackExp : ℝ)) = 2⁻¹ ^ slackExp := inv_two_pow_eq slackExp
  have hp : 2 * (2 : ℝ) ^ (-(slackExp : ℝ)) ≤ 1 := by rw [ej]; norm_num [slackExp]
  have hsq : Claims.Sampling.slack ^ 2 = (2 : ℝ) ^ (2 * (2 : ℝ) ^ (-(slackExp : ℝ))) := by
    unfold Claims.Sampling.slack
    rw [← Real.rpow_natCast, ← Real.rpow_mul (by norm_num)]
    congr 1
    push_cast
    ring
  have h := rpow_one_add_le_one_add_mul_self (s := 1) (by norm_num) (p := 2 * (2 : ℝ) ^ (-(slackExp : ℝ)))
    (by positivity) hp
  rw [hsq, ← ej]
  simpa only [one_add_one_eq_two, mul_one] using h

theorem exp_neg_le {k : ℕ} (hk : 2 ^ 10 ≤ k) : Real.exp (-(k : ℝ) / 8) ≤ 2⁻¹ ^ 184 := by
  norm_num at hk
  have hk' : (1024 : ℝ) ≤ k := by exact_mod_cast hk
  have h1 : Real.exp (-(k : ℝ) / 8) ≤ Real.exp ((128 : ℕ) * (-1)) := by
    rw [Real.exp_le_exp]
    push_cast
    linarith
  have h2 : Real.exp (-1) ≤ (2718 / 1000)⁻¹ := by
    rw [Real.exp_neg]
    exact inv_anti₀ (by norm_num) (by linarith [Real.exp_one_gt_d9])
  rw [Real.exp_nat_mul] at h1
  exact h1.trans ((pow_le_pow_left₀ (Real.exp_pos _).le h2 128).trans (by norm_num))

theorem κ_le_of {Coin : Type} {A : Claims.SamplingGeneric.Arg Coin} (L : Claims.SamplingGeneric.Compiled Coin A)
    (x : Msg) {δ : ℝ} (hδ0 : 0 ≤ δ) (hs : L.κs x ≤ 2⁻¹ ^ 190) : L.κ δ x ≤ (1 + 2⁻¹ ^ 9) * δ + 2⁻¹ ^ 182 := by
  have hj : (2 : ℝ)⁻¹ ^ slackExp = 2⁻¹ ^ 10 := by rw [slackExp]
  have hc := slack_le_one_add
  have hc2 := slack_sq_le_one_add
  rw [hj] at hc hc2
  have hc2' : Claims.Sampling.slack ^ 2 ≤ 1 + 2⁻¹ ^ 9 := by
    have : (2 : ℝ) * 2⁻¹ ^ 10 = 2⁻¹ ^ 9 := by norm_num
    linarith
  have hs0 := L.κs_nonneg x
  have hk : ∀ n : ℕ, 2 ^ 10 ≤ 2 ^ slackExp * (2 * n + 1) := fun _ =>
    Nat.le_mul_of_pos_right _ (by omega)
  have hkO : 2 ^ 10 ≤ L.kOut x := hk _
  have hkI : 2 ^ 10 ≤ L.kIn x := hk _
  have hO : 2 * (L.kOut x : ℝ) / L.capOut x = 2⁻¹ ^ 191 := by
    have h0 : (0 : ℝ) < L.kOut x := by exact_mod_cast lt_of_lt_of_le (by norm_num) hkO
    rw [Claims.SamplingGeneric.Compiled.capOut]
    push_cast
    rw [div_eq_iff (mul_pos (by positivity) h0).ne']
    ring
  have hI : 2 * (L.kIn x : ℝ) / L.capIn x = 2⁻¹ ^ 191 := by
    have h0 : (0 : ℝ) < L.kIn x := by exact_mod_cast lt_of_lt_of_le (by norm_num) hkI
    rw [Claims.SamplingGeneric.Compiled.capIn]
    push_cast
    rw [div_eq_iff (mul_pos (by positivity) h0).ne']
    ring
  have heO := exp_neg_le hkO
  have heI := exp_neg_le hkI
  unfold Claims.SamplingGeneric.Compiled.κ
  rw [hO, hI]
  have hA : Claims.Sampling.slack ^ 2 * (δ + L.κs x) ≤ (1 + 2⁻¹ ^ 9) * (δ + 2⁻¹ ^ 190) :=
    mul_le_mul hc2' (by linarith) (by linarith) (by positivity)
  have hB : Claims.Sampling.slack * (2⁻¹ ^ 191 + Real.exp (-(L.kIn x : ℝ) / 8)) ≤
      (1 + 2⁻¹ ^ 10) * (2⁻¹ ^ 191 + 2⁻¹ ^ 184) :=
    mul_le_mul hc (by linarith) (by positivity) (by positivity)
  have hC : (1 + (2 : ℝ)⁻¹ ^ 9) * (δ + 2⁻¹ ^ 190) = (1 + 2⁻¹ ^ 9) * δ + (1 + 2⁻¹ ^ 9) * 2⁻¹ ^ 190 := by ring
  have hD : (1 + (2 : ℝ)⁻¹ ^ 9) * 2⁻¹ ^ 190 + (2⁻¹ ^ 191 + 2⁻¹ ^ 184) + (1 + 2⁻¹ ^ 10) * (2⁻¹ ^ 191 + 2⁻¹ ^ 184) ≤
      2⁻¹ ^ 182 := by
    norm_num
  linarith

theorem κ_le (x : Msg) {δ : ℝ} (hδ0 : 0 ≤ δ) :
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots (two h)
      (rs h) (cap h)).κ δ x ≤ (1 + 2⁻¹ ^ 9) * δ + 2⁻¹ ^ 182 :=
  κ_le_of _ x hδ0 (Claims.SamplingGeneric.maxBelow_le (by positivity) fun u _ => κF_le h _ x (templateOf h.ms u))

theorem κ_lt_one (x : Msg) {δ : ℝ} (hδ0 : 0 ≤ δ) (hδ : δ ≤ 1 / 2) :
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots (two h)
      (rs h) (cap h)).κ δ x < 1 := by
  have := κ_le h x hδ0
  have h1 : (1 + (2 : ℝ)⁻¹ ^ 9) * δ ≤ (1 + 2⁻¹ ^ 9) * (1 / 2) := mul_le_mul_of_nonneg_left hδ (by positivity)
  have h2 : (1 + (2 : ℝ)⁻¹ ^ 9) * (1 / 2) + 2⁻¹ ^ 182 < 1 := by norm_num
  linarith

/-- A witness that builds the instance needs a `Held`, whose `xS₀_ok` needs each template's masked Flock statement's
bytes decided in Flock's masked scope, and for a run accepted with positive probability, an accepting run of Flock's
masked verifier at each template's statement of a session statements' message the check passes. -/
theorem accepting
    (hopens : Claims.SamplingGeneric.SessionOpensTwo FlockDeploy.sha512
      (SamplingDeploy.deploy _root_.Flock.Readable.relation (code h)) (two h) iroots) :
    SamplingDeploy.CallsUniform (fun _ => L) (fun _ => coinByte) ∧
      SamplingDeploy.RegMatchesTwo ((base h).atOrder (order h)) (reg h) (two h) ∧
      ((base h).atOrder (order h)).RefusesTwo FlockDeploy.sha512 (two h) ∧
      ((base h).atOrder (order h)).ReplayCap FlockDeploy.sha512 (two h) (rs h) (cap h) ∧
      ((base h).atOrder (order h)).InteriorFixed FlockDeploy.sha512 iroots ∧
      SessionRunsTo FlockDeploy.sha512 (SamplingDeploy.deploy _root_.Flock.Readable.relation (code h)) ∧
      SamplingDeploy.SessionsIn (code h) ∧
      (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots (two h)
        (rs h) (cap h)).KS FlockDeploy.sha512 ∧
      (∀ x δ, 0 ≤ δ →
        (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots (two h)
          (rs h) (cap h)).κ δ x ≤ (1 + 2⁻¹ ^ 9) * δ + 2⁻¹ ^ 182) ∧
      ∀ x m₀ m₁ m₂ c₀ c₁, stmtOk h x m₁ m₂ c₀ c₁ = true →
        ∀ (Ω : Type) (P' : ℕ → Prover Ω) (ω : ℕ → Ω)
          (r' : ∀ t, Fin ((FlockDeploy.V true).k (stmtAt h m₂ t)) → FlockDeploy.Coin),
          (∀ t < h.ms.length,
            (run FlockDeploy.sha512 (P' t) (FlockDeploy.V true) (stmtAt h m₂ t) (ω t) (r' t)).out = .acc) →
          0 < Pr fun r : Fin ((code h).V.k x) → FlockDeploy.Coin =>
            (run FlockDeploy.sha512 (proverSeq h m₀ m₁ m₂ P') (code h).V x ω r).out = .acc :=
  ⟨callsUniform, regMatchesTwo_code h, refusesTwo_code h _, replayCap h _, interiorFixed_code h _,
    sessionRunsTo h _root_.Flock.Readable.relation FlockDeploy.sha512, sessionsIn_code h,
    ksTwo_spot h hopens, fun x _ hδ0 => κ_le h x hδ0,
    fun x m₀ m₁ m₂ c₀ c₁ hok _ P' ω r' hacc => accepts_seq h FlockDeploy.sha512 x m₀ m₁ m₂ c₀ c₁ hok P' ω r' hacc⟩

end SpotFlock
