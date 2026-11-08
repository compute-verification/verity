module

public import Proofs.Flock.SpotFlock.Honest
public import Proofs.Flock.SpotFlock.KS

/-!
# SPOT[Flock] meets `SampledKSTwo`'s premises, at a knowledge error below `1`, and accepts where Flock does

`Flock.SecurityProofs.SampledKSTwo` (`Proofs.Flock.SamplingGeneric.Guarantees`) bounds the two-stage sampled verifier's
knowledge error at Flock's masked record and SHA-512 from eight premises. At SPOT[Flock]'s code (`SpotFlock.code h`),
at every set of the auditor's copies `h`:

* `ref_numbers`: template `t`'s reference session statement `xS₀ t` has `1` to `200` tables, their level-0 lengths
  summed `2^10` to `n·2^23` (`zkOk_sound`, `FlockDeploy.N₀_le`); so the code's tables, the most any template's
  reference statement has, are at most `200` (`tables_le`);
* `κF_le`: so the sessions' knowledge error `κ_F = 4·ε_table + 2^{−m′} + 2k′/K′ + e^{−k′/8}` is at most `2^{−190}`;
* `slack_le_one_add`: the waits' slack factor `c = 2^{2^{−40}}` is at most `1 + 2^{−40}` (`2^t ≤ 1 + t` on `[0, 1]`);
* `κ_le`: the compiled two-stage verifier's `κ_δ = c²(δ + κ_s) + (1 + c)(2k/K + e^{−k/8})` is at most `δ + 2^{−37}` at
  every statement and every `δ ∈ [0, 1]`, with `K = 2^128·k` and `k ≥ 2^40` (`κ_le_of`); so `κ_δ < 1` at every
  `δ ≤ 1/2` (`κ_lt_one`);
* `accepting`: A3 at the coins' bytes (`callsUniform`), the code's facts, the sessions' runs one way
  (`sessionRunsTo`) and each unit's session in its template's scope (`sessionsIn_at`) hold; with the sessions' openings
  (`SessionOpensTwo`), their runs both ways and their scope at one `N₀` per statement (`SessionsIn`), `CompiledTwo.KS`
  holds there, at `κ_δ ≤ δ + 2^{−37}`; and a prover that plays a Flock prover at each template's session statement is
  accepted with positive probability wherever Flock's masked verifier accepts every one at a round-2 message the check
  passes (`accepts_seq`).

So the premises hold together at a verifier whose acceptance is Flock's at its session statements. A witness that builds
the instance needs a `Held`, whose `xS₀_ok` needs each template's masked Flock statement's bytes decided in Flock's
masked scope, and for a run accepted with positive probability, an accepting run of Flock's masked verifier at each
template's statement of a round-2 message the check passes.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling

variable (h : Held)

/-! ## The knowledge error -/

/-- **Template `t`'s reference statement's numbers**: `1` to `200` tables, their masked level-0 lengths summed `2^10`
to `n·2^23` at `n` tables. -/
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

/-- **The code's tables are at most `200`**: the most any template's reference statement has (`ref_numbers`). -/
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

theorem inv_two_pow_eq (k : ℕ) : (2 : ℝ) ^ (-(k : ℝ)) = 2⁻¹ ^ k := by
  rw [Real.rpow_neg (by norm_num), Real.rpow_natCast, inv_pow]

section Numbers

variable {T : Type} (F : Claims.Flock.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) T)

/-- **The sessions' knowledge error is at most `2^{−190}`**: `κ_F` at the code's at most `200` tables (`tables_le`),
`m′ = 210`, `K′ = 2^256` and `k′ = ⌈4N₀/e⌉` from `1507` to `200·2^24` at template `0`'s `N₀` (`ref_numbers`), at
every statement and every Flock record. -/
theorem κF_le (x : Msg) : Claims.Sampling.κF (SamplingDeploy.deployOf F (code h)) x ≤ 2⁻¹ ^ 190 := by
  obtain ⟨hn1, hn200, hN, hN'⟩ := ref_numbers h 0
  have hn : ((shapeOf (h.xS₀ 0)).length : ℝ) ≤ 200 := by exact_mod_cast hn200
  have hk1 : 1507 ≤ (SamplingDeploy.deployOf F (code h)).innerK x := Claims.Flock.Numbers.k_ge hN
  have hk2 : ((SamplingDeploy.deployOf F (code h)).innerK x : ℝ) ≤ 200 * 2 ^ 24 :=
    (Claims.Flock.Numbers.k_le_mul (by exact_mod_cast hn1) (le_trans (by positivity) hN) hN').trans (by nlinarith)
  have he : Real.exp (-((SamplingDeploy.deployOf F (code h)).innerK x : ℝ) / 8) ≤ 2⁻¹ ^ 250 := by
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
  have h3 : 2 * ((SamplingDeploy.deployOf F (code h)).innerK x : ℝ) / 2 ^ 256 ≤ 2 * (200 * 2 ^ 24) / 2 ^ 256 := by
    gcongr
  have hsum : 4 * (200 * (2 : ℝ)⁻¹ ^ 205) + 2⁻¹ ^ 210 + 2 * (200 * 2 ^ 24) / 2 ^ 256 + 2⁻¹ ^ 250 ≤ 2⁻¹ ^ 190 := by
    norm_num
  unfold Claims.Sampling.κF
  rw [hε, e205, hw, hcap]
  linarith

end Numbers

/-- **The waits' slack factor is at most `1 + 2^{−40}`**: `2^t ≤ 1 + t` at `t = 2^{−40} ∈ [0, 1]`. -/
theorem slack_le_one_add : Claims.Sampling.slack ≤ 1 + 2⁻¹ ^ 40 := by
  have e40 : (2 : ℝ) ^ (-40 : ℝ) = 2⁻¹ ^ 40 := by
    rw [show (-40 : ℝ) = -((40 : ℕ) : ℝ) by norm_num, inv_two_pow_eq]
  have h := rpow_one_add_le_one_add_mul_self (s := 1) (by norm_num) (p := (2 : ℝ) ^ (-40 : ℝ)) (by positivity)
    (by rw [e40]; norm_num)
  calc Claims.Sampling.slack = (1 + 1 : ℝ) ^ ((2 : ℝ) ^ (-40 : ℝ)) := by
        simp only [Claims.Sampling.slack, one_add_one_eq_two]
    _ ≤ 1 + (2 : ℝ) ^ (-40 : ℝ) * 1 := h
    _ = 1 + 2⁻¹ ^ 40 := by rw [e40, mul_one]

/-- **`κ_δ ≤ δ + 2^{−37}`** at a sampled record whose sessions' knowledge error is at most `2^{−190}` and whose waits
wait for at least `2^40` accepted runs, at every `δ ∈ [0, 1]`: `c²(δ + κ_F) + (1 + c)(2k/K + e^{−k/8})` at
`c ≤ 1 + 2^{−40}`, `2k/K = 2^{−127}` and `e^{−k/8} ≤ 2^{−250}`. -/
theorem κ_le_of {Coin Col T : Type} (D : Claims.Sampling.Deploy Coin Col T) (x : Msg) {δ : ℝ} (hδ0 : 0 ≤ δ)
    (hδ1 : δ ≤ 1) (hF0 : 0 ≤ Claims.Sampling.κF D x) (hF : Claims.Sampling.κF D x ≤ 2⁻¹ ^ 190) (hk : 2 ^ 40 ≤ D.k x) :
    Claims.Sampling.κ D δ x ≤ δ + 2⁻¹ ^ 37 := by
  have hs := slack_le_one_add
  have hs0 : 0 ≤ Claims.Sampling.slack := Claims.Sampling.slack_pos.le
  have hkr : (2 : ℝ) ^ 40 ≤ (D.k x : ℝ) := by exact_mod_cast hk
  have hk0 : (0 : ℝ) < D.k x := lt_of_lt_of_le (by positivity) hkr
  have hcap : 2 * (D.k x : ℝ) / D.cap x = 2⁻¹ ^ 127 := by
    rw [Claims.Sampling.Deploy.cap]
    push_cast
    rw [div_eq_iff (mul_pos (by positivity) hk0).ne']
    ring
  have he : Real.exp (-(D.k x : ℝ) / 8) ≤ 2⁻¹ ^ 250 := by
    have := Claims.Flock.Numbers.exp_neg_k_le (le_trans (by norm_num) hk)
    rwa [show (-(250 : ℝ)) = -((250 : ℕ) : ℝ) by norm_num, inv_two_pow_eq] at this
  have hs2 : Claims.Sampling.slack ^ 2 ≤ 1 + 2⁻¹ ^ 38 := by
    have h2 : Claims.Sampling.slack ^ 2 ≤ (1 + 2⁻¹ ^ 40) ^ 2 := pow_le_pow_left₀ hs0 hs 2
    have h3 : (1 + (2 : ℝ)⁻¹ ^ 40) ^ 2 ≤ 1 + 2⁻¹ ^ 38 := by norm_num
    linarith
  unfold Claims.Sampling.κ
  rw [hcap]
  have hA : Claims.Sampling.slack ^ 2 * (δ + Claims.Sampling.κF D x) ≤ (1 + 2⁻¹ ^ 38) * (δ + 2⁻¹ ^ 190) :=
    mul_le_mul hs2 (by linarith) (by linarith) (by positivity)
  have hB : (1 + Claims.Sampling.slack) * (2⁻¹ ^ 127 + Real.exp (-(D.k x : ℝ) / 8)) ≤
      (2 + 2⁻¹ ^ 40) * (2⁻¹ ^ 127 + 2⁻¹ ^ 250) :=
    mul_le_mul (by linarith) (by linarith) (by positivity) (by positivity)
  have hC1 : (2 : ℝ)⁻¹ ^ 38 * δ ≤ 2⁻¹ ^ 38 := by
    calc (2 : ℝ)⁻¹ ^ 38 * δ ≤ 2⁻¹ ^ 38 * 1 := by gcongr
      _ = 2⁻¹ ^ 38 := mul_one _
  have hC : (1 + (2 : ℝ)⁻¹ ^ 38) * (δ + 2⁻¹ ^ 190) ≤ δ + 2⁻¹ ^ 38 + (1 + 2⁻¹ ^ 38) * 2⁻¹ ^ 190 := by
    have e : (1 + (2 : ℝ)⁻¹ ^ 38) * (δ + 2⁻¹ ^ 190) = δ + 2⁻¹ ^ 38 * δ + (1 + 2⁻¹ ^ 38) * 2⁻¹ ^ 190 := by ring
    rw [e]
    linarith
  have hD : (1 + (2 : ℝ)⁻¹ ^ 38) * 2⁻¹ ^ 190 + (2 + 2⁻¹ ^ 40) * (2⁻¹ ^ 127 + 2⁻¹ ^ 250) + 2⁻¹ ^ 38 ≤ 2⁻¹ ^ 37 := by
    norm_num
  linarith

/-- **`κ_δ ≤ δ + 2^{−37}`** at SPOT[Flock]'s compiled two-stage verifier, at Flock's masked record, every statement and
every `δ ∈ [0, 1]`: its `κ_δ` is at most its sampled record's (`κ_compiled_le`), whose sessions' error is at most
`2^{−190}` (`κF_le`) and whose waits wait for `k = 2^40·(2n_PU + 1) ≥ 2^40` accepted runs. -/
theorem κ_le (x : Msg) {δ : ℝ} (hδ0 : 0 ≤ δ) (hδ1 : δ ≤ 1) :
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots (two h)
      (rs h)).κ δ x ≤ δ + 2⁻¹ ^ 37 :=
  (SamplingDeploy.κ_compiled_le (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots δ x).trans
    (κ_le_of _ x hδ0 hδ1 (Claims.Sampling.κF_nonneg _ x (SamplingDeploy.deployOf_εtable_nonneg _ (code h) x))
      (κF_le h _ x) (by unfold Claims.Sampling.Deploy.k; exact Nat.le_mul_of_pos_right _ (by omega)))

/-- **`κ_δ < 1` at every `δ ≤ 1/2`**: the bound `CompiledTwo.KS` gives SPOT[Flock] is below `1`. -/
theorem κ_lt_one (x : Msg) {δ : ℝ} (hδ0 : 0 ≤ δ) (hδ : δ ≤ 1 / 2) :
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots (two h)
      (rs h)).κ δ x < 1 := by
  have := κ_le h x hδ0 (by linarith)
  have h37 : (2 : ℝ)⁻¹ ^ 37 < 1 / 2 := by norm_num
  linarith

/-! ## The premises together -/

/-- **SPOT[Flock] meets `SampledKSTwo`'s premises one way, its bound says something, and it accepts where Flock does**,
at every set of the auditor's copies `h`: A3 at the coins' bytes (`callsUniform`, at both stages), the code's facts
(`RegMatchesTwo`, `Code.RefusesTwo`, `Code.InteriorFixed`), the sessions' runs one way (an accepted run at coins that
check a unit is Flock's masked verifier accepting the unit's template's session at its round-2 statement) and each
unit's session in its template's scope; with the sessions' openings (`SessionOpensTwo`), their runs both ways and
their scope (`SessionsIn`), `CompiledTwo.KS` at Flock's masked record and SHA-512, at `κ_δ ≤ δ + 2^{−37}` at every
statement and `δ ∈ [0, 1]`; and a prover is accepted on a positive share of the coins wherever Flock's masked verifier
accepts one at each template's session statement of a round-2 message the check passes (`accepts_seq`). -/
theorem accepting
    (hruns : SessionRuns FlockDeploy.sha512 (SamplingDeploy.deploy _root_.Flock.Readable.relation (code h)))
    (hin : SamplingDeploy.SessionsIn (code h))
    (hopens : Claims.SamplingGeneric.SessionOpensTwo FlockDeploy.sha512
      (SamplingDeploy.deploy _root_.Flock.Readable.relation (code h)) (two h) iroots) :
    SamplingDeploy.CallsUniform (fun _ => L) (fun _ => coinByte) ∧
      SamplingDeploy.RegMatchesTwo ((base h).atOrder (order h)) (reg h) (two h) ∧
      ((base h).atOrder (order h)).RefusesTwo FlockDeploy.sha512 (two h) ∧
      ((base h).atOrder (order h)).InteriorFixed FlockDeploy.sha512 iroots ∧
      SessionRunsTo FlockDeploy.sha512 (SamplingDeploy.deploy _root_.Flock.Readable.relation (code h)) ∧
      (∀ x u v xS vS, (code h).session x u v = some (xS, vS) →
        (∃ s spec sts, FlockDeploy.ZkIn xS s spec sts ∧ sts.size ≤ (code h).tables x) ∧
          FlockDeploy.N₀ true xS = n₀ h (templateOf h.ms u)) ∧
      (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots (two h)
        (rs h)).KS FlockDeploy.sha512 ∧
      (∀ x δ, 0 ≤ δ → δ ≤ 1 →
        (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots (two h)
          (rs h)).κ δ x ≤ δ + 2⁻¹ ^ 37) ∧
      ∀ x m₀ m₁ m₂ c₀ c₁, stmtOk h x m₁ m₂ c₀ c₁ = true →
        ∀ (Ω : Type) (P' : ℕ → Prover Ω) (ω : ℕ → Ω)
          (r' : ∀ t, Fin ((FlockDeploy.V true).k (stmtAt h m₂ t)) → FlockDeploy.Coin),
          (∀ t < h.ms.length,
            (run FlockDeploy.sha512 (P' t) (FlockDeploy.V true) (stmtAt h m₂ t) (ω t) (r' t)).out = .acc) →
          0 < Pr fun r : Fin ((code h).V.k x) → FlockDeploy.Coin =>
            (run FlockDeploy.sha512 (proverSeq h m₀ m₁ m₂ P') (code h).V x ω r).out = .acc :=
  ⟨callsUniform, regMatchesTwo_code h, refusesTwo_code h _, interiorFixed_code h _,
    sessionRunsTo h _root_.Flock.Readable.relation FlockDeploy.sha512, sessionsIn_at h,
    ksTwo_spot h hruns hin hopens, fun x _ hδ0 hδ1 => κ_le h x hδ0 hδ1,
    fun x m₀ m₁ m₂ c₀ c₁ hok _ P' ω r' hacc => accepts_seq h FlockDeploy.sha512 x m₀ m₁ m₂ c₀ c₁ hok P' ω r' hacc⟩

end SpotFlock
