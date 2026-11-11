module

public import Proofs.Flock.Claims.KS.ZkExtend
public import Proofs.Flock.SamplingGeneric.Masked

/-!
# The deployed sampled verifier over Flock's masked record, its sessions' `E^rep` truncated at `K₀`

At the deployed masked record each wait of a session's `E^rep` gives up only after `2^256` reruns, so the sessions'
strict reruns `ρ_s`, and with them `B_conf`'s cost on every coin, are above `2^256`, where the strict birthday bound
(`KSRegAtSB`) says nothing. With each wait giving up after `K₀ > 0` reruns (`FlockDeploy.deployZkAt`), Flock's masked
KS holds with no other premise (`FlockDeploy.KS.zk_KS_untrim_at`), `κ`'s term `2k/K` is `2k/K₀`, and
`ρ_s ≤ 1 + 210·K₀` (`ρs_compiledTrunc_le`). The sampled record whose sessions run that record (`deployTrunc`) meets the
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
