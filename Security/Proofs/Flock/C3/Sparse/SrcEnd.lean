import Proofs.Flock.C3.Sparse.SrcTie
import Proofs.Flock.C3.Opening.ClaimTie
import Proofs.Flock.C3.Compose.DecodeFix
import Proofs.Flock.Recursive.ForkRuns
import Proofs.Flock.Soundness.Discharge.ZkExec.VerdictInner

/-!
# The sparse end of the claim tie

`claim_of_tie`: a tied rep (`SparseTie`) that plays its sparse phase to the run's claim ends at its unit's claim
(`sparseClaim`), since the unit's verifier row runs the phase to it (`sparse_refines`) and a game plays on one tape to
one outcome (`PlaysT.det`). `coord_of_coins`: at the record's sparse coins (`sparseCoins`, whose `want` check fixes
their counts), `coord`'s three branches are `coinPt`'s three blocks and its zero padding. `ties_of_ends`: the point
and value ties from the reps' ends.
-/

namespace FlockC3.Sparse

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.Compose
open FlockVBridge.Sparse (SCoins nw nc nR wClaim verifierRow outOf_eq tail_iff shOf out_eq laid_sparse check_sparse
  sparseClaim sparse_refines)
open FlockVBridge.OfVStar (bitOf_zero)
open FlockVBridge.AccChain (eq_of_bitOf)

/-- `playsT_of_unit`, keeping the claim. -/
theorem playsT_claim_of_unit {sh : SparseShape} {pub : SparsePub GF128 sh} {τ' : Fin (2 ^ sh.sLog) → ℕ}
    {h : Fin 64 → GF128} {w : Array Flock.F128} {τ : Array ℕ} {c : SCoins} (hu : SparseUnit sh pub τ' h w τ c) :
    PlaysT (sparsePhase execArith sh pub τ' h) (stepsOf sh w c.cs) (some (sparseClaim sh w c)) := by
  obtain ⟨hw, hc, rfl, hτ, hh, o, ho, hacc⟩ := hu
  obtain rfl : (fun σ : Fin (2 ^ sh.sLog) => τ[σ.val]!) = τ' := funext hτ
  obtain rfl : (fun s : Fin 64 => fOf w[s.val]!) = h := funext hh
  obtain ⟨o', ho', hpl, hiff⟩ := sparse_refines sh w c τ hw hc
  have e : Except.ok o = (Except.ok o' : Except String _) := ho.symm.trans ho'
  cases e
  rw [(hiff.1 hacc).2] at hpl
  exact playsT_stepsOf hpl

/-- **A tied rep's claim is its unit's**: its stage is the unit's steps, and the claim it plays to is `sparseClaim`. -/
theorem claim_of_tie {e : ExecHolo} {u : C3Run e} {i : Fin 2} (h : SparseTie e u i)
    (hp : PlaysT (sparseAt e u i) (u.sp i) (some (u.claim i))) :
    ∃ (w : Array Flock.F128) (c : SCoins), w.size = nw e.cls.shape ∧ nc e.cls.shape ≤ c.cs.size ∧
      u.sp i = stepsOf e.cls.shape w c.cs ∧ u.claim i = sparseClaim e.cls.shape w c := by
  obtain ⟨w, c, cc, v, hw, hc, hpub, hh, hs, hck, hpar, hL, hin, h0⟩ := h
  have hv := (check_sparse hck hpar).1
  simp only [Flock.RecSparse.Shape.valid, Bool.and_eq_true, decide_eq_true_eq] at hv
  obtain ⟨hbits, -⟩ := laid_sparse (unitShape e) hv.2 v hL w (verifierRow (shOf (unitShape e)) c) (tauArr e) hin
  have hq : PlaysT (sparseAt e u i) (stepsOf e.cls.shape w c.cs) (some (sparseClaim e.cls.shape w c)) := by
    refine playsT_claim_of_unit (τ := tauArr e) ⟨hw, hc, hpub,
      fun σ => arr_ofFn_get (fun σ => (e.τ σ).val) σ.isLt, hh, _, out_eq (shOf (unitShape e)) w (tauArr e) c hw,
      ⟨fun x hx => ?_, fun y hy => ?_⟩⟩
    · obtain ⟨r, hr, rfl⟩ := Array.mem_iff_getElem.1 hx
      refine eq_of_bitOf fun k hk => (hbits r hr k hk).symm.trans ((h0 _ ?_).trans (bitOf_zero k).symm)
      have := Nat.lt_of_lt_of_eq hr (outOf_size (unitShape e) _ _ _ _)
      omega
    · rw [outOf_eq] at hy
      split at hy
      · simp at hy
      · exact (tail_iff (shOf (unitShape e)) (tauArr e)).2
          (fun s hs => (arr_ofFn_get (fun σ => (e.τ σ).val) hs).trans_lt (e.τ _).isLt) y hy
  rw [hs] at hp
  exact ⟨w, c, hw, hc, hs, Option.some.inj (FlockSoundness.Game.PlaysT.det hp hq)⟩

theorem size_of_sizes {cs : Array (Array Flock.F128)} {want : Array ℕ} (h : cs.map (·.size) = want) {i : ℕ}
    (hi : i < want.size) : (cs[i]!).size = want[i]! := by
  subst h
  rw [map_get! cs _ (by simpa using hi)]

/-- **The record's sparse coins have the phase's counts** (`sparseCoins`' `want` check). -/
theorem sparseCoins_sizes {c3 : Flock.RecCoins.C3} {unit : String} {k : ℕ} {sc : Flock.RecCoins.SparseCoins}
    (h : Flock.RecCoins.sparseCoins c3 unit k = .ok sc) :
    sc.rounds.size = c3.sparse.degrees.length ∧ sc.zetaP.size = c3.sparse.pLog ∧ sc.rho.size = c3.sparse.tLog := by
  unfold Flock.RecCoins.sparseCoins at h
  dsimp only at h
  obtain ⟨cs, -, h⟩ := FlockVBridge.PartS.bind_ok h
  try dsimp only at h
  by_cases hst : c3.sparse.stacked = true
  all_goals first | rw [ite_eq_left hst, ite_eq_left hst] at h | rw [ite_eq_right hst, ite_eq_right hst] at h
  all_goals
    split at h
    · cases h
    rename_i hb
    cases h
    simp only [bne_iff_ne, ne_eq, not_not] at hb
    dsimp only
    refine ⟨by simp, ?_, ?_⟩
    · rw [size_of_sizes hb (by simp <;> omega), append_get!_left _ _ (by simp <;> omega),
        append_get!_right _ _ (j := 0) (by simp <;> omega) (by simp)]
      all_goals first | rfl | simp
  · rw [size_of_sizes hb (by simp <;> omega), append_get!_right _ _ (j := 0) (by simp <;> omega) (by simp)]
    all_goals first | rfl | simp
  · simp only [Flock.RecSparse.Shape.stacked, decide_eq_true_eq] at hst
    show (0 : ℕ) = _
    omega

/-- **The claim's point is the coin point** where its coins at `r_k`, `ζ_p` and `ρ_t` are the record's. -/
theorem coord_of_coins {c3 : Flock.RecCoins.C3} {unit : String} {sh : Flock.Shape} {ssh : SparseShape}
    {w : Array Flock.F128} {c : SCoins} {k : ℕ} {sc : Flock.RecCoins.SparseCoins}
    (hsc : Flock.RecCoins.sparseCoins c3 unit k = .ok sc) (hc3 : shOf c3.sparse = ssh) (hm : sh.m = ssh.mReg)
    (hrk : ∀ j < ssh.nu, c.cs[12 + j]! = sc.rounds[j]!)
    (hzp : ∀ j < ssh.pLog, c.cs[12 + nR ssh + j]! = sc.zetaP[j]!)
    (hrho : ∀ j < ssh.tLog, c.cs[12 + nR ssh + ssh.pLog + j]! = sc.rho[j]!) :
    ∀ j < ssh.mReg, (sparseClaim ssh w c).coord j = fOf (FlockC3.Opening.coinPt c3 sh sc)[j]! := by
  obtain ⟨hr, hz, ht⟩ := sparseCoins_sizes hsc
  have hnu : c3.sparse.nu = ssh.nu := by subst hc3; rfl
  have hr' : ssh.nu ≤ sc.rounds.size := by
    subst hc3; rw [hr, FlockVBridge.Sparse.degs_length]; exact Nat.le_add_right _ _
  have hz' : sc.zetaP.size = ssh.pLog := by subst hc3; rw [hz, FlockVBridge.Sparse.pLog_shOf]
  have ht' : sc.rho.size = ssh.tLog := by subst hc3; exact ht
  have hA : (sc.rounds.extract 0 c3.sparse.nu).size = ssh.nu := by rw [Array.size_extract]; omega
  have hAZ : (sc.rounds.extract 0 c3.sparse.nu ++ sc.zetaP).size = ssh.nu + ssh.pLog := by
    rw [Array.size_append, hA, hz']
  have hAZR : (sc.rounds.extract 0 c3.sparse.nu ++ sc.zetaP ++ sc.rho).size = ssh.nu + ssh.pLog + ssh.tLog := by
    rw [Array.size_append, hAZ, ht']
  intro j hj
  have hcp : (FlockC3.Opening.coinPt c3 sh sc)[j]! = (sc.rounds.extract 0 c3.sparse.nu ++ sc.zetaP ++ sc.rho ++
      Array.replicate (sh.m - (sc.rounds.extract 0 c3.sparse.nu ++ sc.zetaP ++ sc.rho).size) 0)[j]! := rfl
  rw [hcp]
  simp only [RegClaim.coord, vget]
  by_cases h1 : j < ssh.nu
  · rw [ite_eq_left h1, dite_eq_left h1]
    show fOf c.cs[12 + j]! = _
    rw [hrk j h1, append_get!_left _ _ (by omega), append_get!_left _ _ (by omega), append_get!_left _ _ (by omega),
      extract_get! _ 0 _ j (by omega) (by omega), Nat.zero_add]
  by_cases h2 : j < ssh.nu + ssh.pLog
  · rw [ite_eq_right h1, ite_eq_left h2, dite_eq_left (show j - ssh.nu < ssh.pLog by omega)]
    show fOf c.cs[12 + nR ssh + (j - ssh.nu)]! = _
    rw [hzp (j - ssh.nu) (by omega), append_get!_left _ _ (by omega), append_get!_left _ _ (by omega),
      append_get!_right _ _ (j := j - ssh.nu) (by omega) (by omega)]
  by_cases h3 : j < ssh.nu + ssh.pLog + ssh.tLog
  · rw [ite_eq_right h1, ite_eq_right h2, dite_eq_left (show j - ssh.nu - ssh.pLog < ssh.tLog by omega)]
    show fOf c.cs[12 + nR ssh + ssh.pLog + (j - ssh.nu - ssh.pLog)]! = _
    rw [hrho (j - ssh.nu - ssh.pLog) (by omega), append_get!_left _ _ (by omega),
      append_get!_right _ _ (j := j - ssh.nu - ssh.pLog) (by omega) (by omega)]
  rw [ite_eq_right h1, ite_eq_right h2, dite_eq_right (show ¬ j - ssh.nu - ssh.pLog < ssh.tLog by omega),
    append_get!_right _ _ (j := j - (ssh.nu + ssh.pLog + ssh.tLog)) (by omega) (by rw [Array.size_replicate]; omega),
    repl_get _ (by omega), fOf_zero]

/-- **The point and value ties from the reps' ends**: each rep's claim is its unit's (`claim_of_tie`), whose coins at
`r_k`, `ζ_p` and `ρ_t` are the record's sparse coins and whose claim port is R's `rec-claim` row. -/
theorem ties_of_ends {c3 : Flock.RecCoins.C3} {unit : String} {sh : Flock.Shape} {e : ExecHolo} {u : C3Run e}
    {wS : Fin 2 → Array Flock.F128} {cS : Fin 2 → SCoins} {wR : Array Flock.F128}
    (hend : ∀ i, u.claim i = sparseClaim e.cls.shape (wS i) (cS i)) (hc3 : c3.sparse = unitShape e)
    (hm : sh.m = e.cls.shape.mReg)
    (hcoins : ∀ (i : Fin 2) sc, Flock.RecCoins.sparseCoins c3 unit i.val = .ok sc →
      (∀ j < e.cls.shape.nu, (cS i).cs[12 + j]! = sc.rounds[j]!) ∧
      (∀ j < e.cls.shape.pLog, (cS i).cs[12 + nR e.cls.shape + j]! = sc.zetaP[j]!) ∧
      ∀ j < e.cls.shape.tLog, (cS i).cs[12 + nR e.cls.shape + e.cls.shape.pLog + j]! = sc.rho[j]!)
    (hval : ∀ i : Fin 2, fOf (wS i)[wClaim e.cls.shape]! = fOf wR[Flock.RecOpening.claimW sh i]!) :
    FlockC3.Opening.PointTie c3 unit sh u.claim ∧ FlockC3.Opening.ValueTie sh wR u.claim := by
  have hc3' : shOf c3.sparse = e.cls.shape := (congrArg shOf hc3).trans rfl
  refine ⟨fun i sc hsc j hj => ?_, fun i => ?_⟩
  · obtain ⟨hrk, hzp, hrho⟩ := hcoins i sc hsc
    rw [hend i]
    exact coord_of_coins hsc hc3' hm hrk hzp hrho j hj
  · rw [hend i]
    exact hval i

end

end FlockC3.Sparse
