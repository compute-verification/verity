import Proofs.Flock.Claims.Complete.Shaped.Walk

/-!
# The honest PIOP's lengths

The honest rep's first three phases return messages of the lengths `decodeProof` checks, whatever their coins:
`zerocheckP_shape` (two 64-entry first-round messages, and `m − 6` rounds and challenges), `lincheckP_shape` (its
`kLog − 6` rounds and a 64-entry partial table, at a circuit whose fold returns at every challenge; the loop halves the
folded table `2^kLog` once a round, `halve_size`) and `ringSwitchP_shape` (one 128-entry switch vector per claim).
-/

namespace FlockComplete.JsonRT

open Flock Transcript FlockComplete FlockDeploy FlockSoundness.Refine

theorem zerocheckP_shape (resp : ℕ → Claims.Msg) (m : ℕ) (digest cap : ByteArray) (tr : Trace) :
    totT.T (zerocheckP (asksS resp) m digest cap tr) fun p =>
      p.1.round1ab.size = 64 ∧ p.1.round1c.size = 64 ∧ p.1.rounds.size = m - 6 ∧ p.2.rho.size = m - 6 := by
  unfold zerocheckP
  dsimp only
  repeat (first
    | (refine Total.forIn_bind_upto _ _ _
        (fun i (s : Array F128 × Array F128 × Array (F128 × F128) × Array F128) =>
          s.2.2.1.size = i ∧ s.2.2.2.size = i) ⟨rfl, rfl⟩
        (fun i _ s hs => ?_) (fun s hs => ?_))
    | (with_reducible apply Total.bind')
    | (with_reducible intro _)
    | (with_reducible apply sampleP_tot)
    | (with_reducible apply sampleVecP_tot)
    | (with_reducible apply Total.liftTM; intro _; exact ⟨_, rfl⟩)
    | (exact totT.pure _ ⟨_, rfl, by simp [*]⟩)
    | (exact totT.pure _ ⟨by simp, by simp, ‹_ ∧ _›.1, ‹_ ∧ _›.2⟩))

theorem halve_size (t : F128) (a : Array F128) : (halve t a).size = a.size / 2 := by simp [halve]

theorem lincheckP_shape (resp : ℕ → Claims.Msg) (c : CircuitFold) (m : ℕ) (tr : Trace) (zc : ZcOut)
    (hfold : ∀ α, ∃ v, c.fold α (skipWeights zc.z) (zc.rho.extract 0 (c.kLog - 6)) = .ok v) (hk : 6 ≤ c.kLog) :
    totT.T (lincheckP (asksS resp) c m tr zc) fun p => p.1.rounds.size = c.kLog - 6 ∧ p.1.zPartial.size = 64 := by
  unfold lincheckP
  dsimp only
  refine Total.bind' _ (Total.liftTM _ _ fun _ => ⟨_, rfl⟩) fun _ => ?_
  refine Total.bind' _ (sampleP_tot resp) fun alpha => ?_
  refine Total.bind' _ (sampleP_tot resp) fun beta => ?_
  split
  · refine Total.bind' _ (Total.pure' _ _) fun full => ?_
    refine Total.forIn_bind_upto _ _ _
      (fun i (s : Array F128 × Array F128 × Array (F128 × F128) × Array F128) =>
        s.2.1.size = 2 ^ (c.kLog - i) ∧ s.2.2.1.size = i) ⟨by simp, rfl⟩ (fun i hi s hs => ?_) (fun s hs => ?_)
    · refine Total.bind' _ (Total.liftTM _ _ fun _ => ⟨_, rfl⟩) fun _ => ?_
      refine Total.bind' _ (Total.liftTM _ _ fun _ => ⟨_, rfl⟩) fun _ => ?_
      refine Total.bind' _ (sampleP_tot resp) fun t => ?_
      refine totT.pure _ ⟨_, rfl, ?_, by simp [hs.2]⟩
      dsimp only
      rw [halve_size, hs.1, pow_half (by omega), Nat.sub_sub]
    · refine Total.bind' _ (Total.liftTM _ _ fun _ => ⟨_, rfl⟩) fun _ => ?_
      refine Total.bind' _ (sampleP_tot resp) fun rPrime => ?_
      refine totT.pure _ ⟨hs.2, ?_⟩
      dsimp only
      rw [hs.1, show c.kLog - (c.kLog - 6) = 6 by omega]
      rfl
  · exfalso
    exact fold_ne_error hfold ‹_›

theorem ringSwitchP_shape (resp : ℕ → Claims.Msg) (tr : Trace) (claims : Array Claim) :
    totT.T (ringSwitchP (asksS resp) tr claims) fun p => p.1.size = claims.size ∧ ∀ v ∈ p.1, v.size = 128 := by
  unfold ringSwitchP
  dsimp only
  refine Total.bind' _ (Total.liftTM _ _ fun _ => ⟨_, rfl⟩) fun _ => ?_
  refine Total.forIn_bind_upto _ _ _
    (fun i (s : Array (Array F128) × Array F128 × Array (Array F128 × Array F128)) =>
      s.1.size = i ∧ ∀ v ∈ s.1, v.size = 128) ⟨rfl, by simp⟩ (fun i hi s hs => ?_) (fun s hs => ?_)
  · refine Total.bind' _ (Total.liftTM _ _ fun _ => ⟨_, rfl⟩) fun _ => ?_
    refine Total.bind' _ (Total.liftTM _ _ fun _ => ⟨_, rfl⟩) fun _ => ?_
    refine Total.bind' _ (sampleVecP_tot resp 7) fun rpp => ?_
    refine totT.pure _ ⟨_, rfl, by simp [hs.1], fun v hv => ?_⟩
    rcases Array.mem_push.1 hv with hv | rfl
    · exact hs.2 v hv
    · simp [switchVec]
  · refine Total.bind' _ (sampleVecP_tot resp _) fun gamma => ?_
    exact totT.pure _ hs

end FlockComplete.JsonRT
