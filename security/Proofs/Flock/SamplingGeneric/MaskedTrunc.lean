module

public import Proofs.Flock.Claims.KS.ZkExtend
public import Proofs.Flock.SamplingGeneric.Masked

/-!
# The deployed sampled verifier over Flock's masked record, its sessions' `E^rep` truncated at `K₀`

At the deployed masked record each wait of a session's `E^rep` gives up only after `2^256` reruns, so the sessions'
strict reruns `ρ_s`, and with them `B_conf`'s cost on every coin, are above `2^256`, where the strict birthday bound
(`KSRegAtSB`) says nothing. With each wait giving up after `K₀ > 0` reruns (`FlockDeploy.deployZkAt`), Flock's masked
KS holds with no other premise (`FlockDeploy.KS.zk_KS_untrim_at`), each session's `κ` is at most
`t·2^{−203} + 2^{−210} + 2k/K₀ + 2^{−250}` at `t` tables (`κs_compiledTrunc_le`), and `ρ_s ≤ 1 + 210·K₀`
(`ρs_compiledTrunc_le`). The sampled record whose sessions run that record (`deployTrunc`) meets the
contract wherever the deployed one does, since its sessions' runs, scope and openings don't read the truncation
(`sessions_compiledTrunc`), so its compiled verifier is knowledge sound in the committed form at the strict birthday
bound (`ksRegAtSB_maskedTrunc`).
-/

@[expose] public section

namespace Claims.SamplingGeneric

theorem maxBelowN_le {n M : ℕ} {f : ℕ → ℕ} (h : ∀ u < n, f u ≤ M) : maxBelowN n f ≤ M := by
  unfold maxBelowN
  suffices hl : ∀ l : List ℕ, (∀ a ∈ l, a ≤ M) → l.foldr max 0 ≤ M from
    hl _ fun a ha => by
      obtain ⟨u, hu, rfl⟩ := List.mem_map.1 ha
      exact h u (List.mem_range.1 hu)
  intro l hl
  induction l with
  | nil => exact Nat.zero_le _
  | cons a l ih =>
    simp only [List.foldr_cons]
    exact max_le (hl a (List.mem_cons.2 (Or.inl rfl))) (ih fun b hb => hl b (List.mem_cons.2 (Or.inr hb)))

end Claims.SamplingGeneric

namespace SamplingDeploy

open Claims.Sampling (InnerNumbers SessionRuns SessionOpens)

variable (K₀ : ℕ) (c : Code) (iroots : Claims.Msg → Claims.Seen → ℕ → List Claims.Msg)

/-- **The deployed sampled record over Flock's masked record with `E^rep` truncated at `K₀`**: `deployOf` at
`FlockDeploy.deployZkAt K₀`, its sessions' truncation `K₀`. -/
noncomputable def deployTrunc :
    Claims.Sampling.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) Claims.Flock.ZkTrace :=
  { deployOf (FlockDeploy.deployZkAt K₀ _root_.Flock.Readable.relation) c with innerCap := fun _ => K₀ }

/-- **The deployed sampled verifier over it**, as `compiled`. -/
noncomputable abbrev compiledTrunc :
    Claims.SamplingGeneric.Compiled FlockDeploy.Coin
      (Claims.SamplingGeneric.Arg.ofFlock (FlockDeploy.deployZkAt K₀ _root_.Flock.Readable.relation)) :=
  Claims.SamplingGeneric.ofDeploy (deployTrunc K₀ c) iroots

/-- **The sessions rerun at most `1 + 210·K₀` times**: every proof block's shape has `Numbers.waits` waits, each
truncated at `K₀`. -/
theorem ρs_compiledTrunc_le (x : Claims.Msg) :
    (compiledTrunc K₀ c iroots).ρs x ≤ 1 + Claims.Flock.Numbers.waits * K₀ :=
  Claims.SamplingGeneric.maxBelowN_le fun _ _ => le_rfl

/-- **Each session's knowledge error at the truncation**: `κ_s ≤ t·2^{−203} + 2^{−210} + 2k/K₀ + 2^{−250}` where the
statement's sessions set up at most `t` tables at `N₀ ≥ 2^10`, `k = ⌈4N₀/e⌉`: `κ_F`'s terms at the record, as
Flock's `κ` at a truncation `K₀` is bounded, at `ε_table = t·2^{−205}`. -/
theorem κs_compiledTrunc_le (x : Claims.Msg) (hN : 2 ^ 10 ≤ c.innerN₀ x) :
    (compiledTrunc K₀ c iroots).κs x ≤ (c.tables x : ℝ) * 2 ^ (-(203 : ℝ)) + 2 ^ (-(210 : ℝ)) +
      2 * (⌈4 * c.innerN₀ x / Real.exp 1⌉₊ : ℝ) / K₀ + 2 ^ (-(250 : ℝ)) := by
  have hk := Claims.Flock.Numbers.exp_neg_k_le (Claims.Flock.Numbers.k_ge hN)
  have h4 : 4 * ((c.tables x : ℝ) * 2 ^ (-(205 : ℝ))) = (c.tables x : ℝ) * 2 ^ (-(203 : ℝ)) := by
    have : (2 : ℝ) ^ (-(203 : ℝ)) = 4 * 2 ^ (-(205 : ℝ)) := by
      rw [show (-(203 : ℝ)) = 2 + -(205 : ℝ) by norm_num, Real.rpow_add (by norm_num)]; norm_num
    rw [this]; ring
  have hw : (2 : ℝ) ^ (-(Claims.Flock.Numbers.waits : ℝ)) = 2 ^ (-(210 : ℝ)) := by
    unfold Claims.Flock.Numbers.waits; norm_num
  refine Claims.SamplingGeneric.maxBelow_le (by positivity) fun u _ => ?_
  have hb : (compiledTrunc K₀ c iroots).κb x u = 4 * ((c.tables x : ℝ) * 2 ^ (-(205 : ℝ))) +
      2 ^ (-(Claims.Flock.Numbers.waits : ℝ)) + 2 * (⌈4 * c.innerN₀ x / Real.exp 1⌉₊ : ℝ) / K₀ +
      Real.exp (-(⌈4 * c.innerN₀ x / Real.exp 1⌉₊ : ℝ) / 8) := rfl
  rw [hb, h4, hw]
  linarith

/-- **The inner numbers are Flock's** at the truncation: the waits, `N₀`, `ε_table` and weight as at the deployed
record (`innerNumbers_deploy`), the truncation by definition. -/
theorem innerNumbers_deployTrunc (h : SessionsIn c.atInnerN₀) : InnerNumbers (deployTrunc K₀ c) := by
  intro x u v xS vS hs
  obtain ⟨hw, -, hN, he, hwt⟩ := innerNumbers_deploy _root_.Flock.Readable.relation c h x u v xS vS hs
  exact ⟨hw, rfl, hN, he, hwt⟩

/-- **The sessions' runs don't read the truncation.** -/
theorem sessionRuns_deployTrunc (H : Claims.Msg → Claims.Msg)
    (h : SessionRuns H (deploy _root_.Flock.Readable.relation c)) : SessionRuns H (deployTrunc K₀ c) :=
  h

/-- **The sessions' openings don't read the truncation.** -/
theorem sessionOpens_deployTrunc (H : Claims.Msg → Claims.Msg)
    (h : SessionOpens H (deploy _root_.Flock.Readable.relation c) iroots) :
    SessionOpens H (deployTrunc K₀ c) iroots :=
  h

/-- **The contract at the truncation**: the code's four facts and the deployed record's sessions' runs, scope and
openings (`sessions_ofDeploy`). -/
theorem sessions_compiledTrunc (hd : c.DrawLaw) (hr : c.Refuses FlockDeploy.sha512)
    (hi : c.InteriorFixed FlockDeploy.sha512 iroots) (hf : c.DrawFixed)
    (hruns : SessionRuns FlockDeploy.sha512 (deploy _root_.Flock.Readable.relation c))
    (hnum : SessionsIn c.atInnerN₀)
    (hopens : SessionOpens FlockDeploy.sha512 (deploy _root_.Flock.Readable.relation c) iroots) :
    (compiledTrunc K₀ c iroots).Sessions FlockDeploy.sha512 :=
  Claims.SamplingGeneric.sessions_ofDeploy (deployTrunc K₀ c) iroots FlockDeploy.sha512 hd hr hi hf
    (sessionRuns_deployTrunc K₀ c _ hruns) (innerNumbers_deployTrunc K₀ c hnum)
    (sessionOpens_deployTrunc K₀ c iroots _ hopens)

/-- **The deployed sampled verifier over Flock's masked record with `E^rep` truncated at `K₀ > 0`, in the committed
form at the strict birthday bound**: `KSReg` by `flockKSReg` (Flock's KS by `zk_KS_untrim_at`, the list bound by
`listWithin_deployZk`, binding by `commit_binds`, the contract by `sessions_compiledTrunc`), read at `StrictBirthday`
by `ksRegAtSB_of_ksReg` with the opening checks' count from the reveal count. It takes what `ksReg_masked` takes, and
the reveal count; its `q_T` reads `ρ_s ≤ 1 + 210·K₀` (`ρs_compiledTrunc_le`). -/
theorem ksRegAtSB_maskedTrunc (hK₀ : 0 < K₀) (hd : c.DrawLaw) (hr : c.Refuses FlockDeploy.sha512)
    (hi : c.InteriorFixed FlockDeploy.sha512 iroots) (hf : c.DrawFixed)
    (hruns : SessionRuns FlockDeploy.sha512 (deploy _root_.Flock.Readable.relation c))
    (hnum : SessionsIn c.atInnerN₀)
    (hopens : SessionOpens FlockDeploy.sha512 (deploy _root_.Flock.Readable.relation c) iroots) (hrev : RevealsLe c) :
    (compiledTrunc K₀ c iroots).KSRegAtSB FlockDeploy.sha512 :=
  have hS := sessions_compiledTrunc K₀ c iroots hd hr hi hf hruns hnum hopens
  Claims.SamplingGeneric.Compiled.ksRegAtSB_of_ksReg hS.toCostContract
    (Claims.SamplingGeneric.openCallsLe_ofDeploy (deployTrunc K₀ c) iroots
      (openCallsLe_deployOf (FlockDeploy.deployZkAt K₀ _root_.Flock.Readable.relation) c FlockDeploy.sha512 hrev))
    (Claims.SamplingGeneric.flockKSReg FlockDeploy.sha512 _ _ (FlockDeploy.KS.zk_KS_untrim_at hK₀)
      (Claims.Sampling.listWithin_deployZk _root_.Flock.Readable.relation) (commit_binds c _) hS)

end SamplingDeploy
