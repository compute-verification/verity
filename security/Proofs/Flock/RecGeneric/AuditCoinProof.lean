module

public import Proofs.Flock.RecGeneric.AuditCoin
public import Proofs.Flock.RecGeneric.SampledProof
public import Proofs.Core.Game.Prob
public import Proofs.Flock.RecGeneric.StatementProof

/-!
# The recursive audit with the coin in the outer statement: its proof

At the registration `R` the strategy sent, the game is the sampled audit at `L R`'s coin-tagged design with the one
registration `Unit`: a sample of `tagged (L R)` names its coin (`coinOf_tagged`), so the session there is the compiled
verifier at `stmt R ω`. `Flock.SecurityProofs.RecursiveSampledKS` holds there unchanged. Its sampling term is `L R`'s,
since a coin's own unit is correct in every trace, and its knowledge term is
`Comp.err` at the coin, since every sample of `tagged (L R)` holds its coin's unit.
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims FlockSoundness FlockSoundness.Game Spot.Sampling FlockSoundness.SampledGeneric
open scoped ENNReal

variable {n : ℕ}

theorem untag_equivFin (L : SamplingDesign n) (a : L.Ω × Option (Fin n)) : untag L (Fintype.equivFin _ a) = a :=
  Equiv.symm_apply_apply _ a

theorem mem_tagged {L : SamplingDesign n} {ω : L.Ω} {u : Fin (Fintype.card (L.Ω × Option (Fin n)))} :
    u ∈ (tagged L).draw ω ↔ untag L u = (ω, none) ∨ ∃ v ∈ L.draw ω, untag L u = (ω, some v) := by
  classical
  constructor
  · intro h
    obtain ⟨a, ha, rfl⟩ := Finset.mem_map.1 h
    rw [Equiv.coe_toEmbedding, untag_equivFin]
    rcases Finset.mem_insert.1 ha with rfl | ha
    · exact Or.inl rfl
    · obtain ⟨v, hv, rfl⟩ := Finset.mem_image.1 ha
      exact Or.inr ⟨v, hv, rfl⟩
  · intro h
    refine Finset.mem_map.2 ⟨untag L u, ?_, (Fintype.equivFin _).apply_symm_apply u⟩
    rcases h with h | ⟨v, hv, h⟩
    · rw [h]; exact Finset.mem_insert_self _ _
    · rw [h]; exact Finset.mem_insert_of_mem (Finset.mem_image.2 ⟨v, hv, rfl⟩)

theorem tag_mem_tagged (L : SamplingDesign n) (ω : L.Ω) : Fintype.equivFin _ (ω, none) ∈ (tagged L).draw ω :=
  mem_tagged.2 (Or.inl (untag_equivFin L _))

/-- **A sample of the coin-tagged design names its coin.** -/
theorem coinOf_tagged (L : SamplingDesign n) (ω : L.Ω) : coinOf L ((tagged L).draw ω) = ω := by
  have hne : ((tagged L).draw ω).Nonempty := ⟨_, tag_mem_tagged L ω⟩
  rw [coinOf, dif_pos hne]
  rcases mem_tagged.1 (Finset.min'_mem _ hne) with h | ⟨v, _, h⟩ <;> rw [h]

theorem stmtAt_tagged (L : SamplingDesign n) (stmt : L.Ω → Msg) (x₀ : Msg) (ω : L.Ω) :
    stmtAt L stmt x₀ ((tagged L).draw ω) = stmt ω := by
  rw [stmtAt, coinOf_tagged]
  exact if_pos rfl

/-- **The compiled verifier's acceptance reads the statement and the claims prover only.** -/
theorem prob_sessionOf {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg)
    (V : Verifier Coin) (p : Bool → Prop) {x x' : Msg} (hx : x = x') (s : Strategy (sessionOf H V x))
    (s' : Strategy (sessionOf H V x')) (hs : s.1 = s'.1) :
    prob p (sessionOf H V x) s = prob p (sessionOf H V x') s' := by
  subst hx
  obtain ⟨P, f⟩ := s
  obtain ⟨P', f'⟩ := s'
  obtain rfl : P = P' := hs
  rfl

end Claims.RecGeneric

open Claims Claims.RecGeneric FlockSoundness FlockSoundness.Game Spot.Sampling FlockSoundness.SampledGeneric in
/-- **The recursive audit with the coin in the outer statement** (`Flock.Guarantees.RecursiveAuditBoundCoin`):
`Flock.SecurityProofs.RecursiveSampledKS` at the coin-tagged design of the registered `L σ.1`, with the one
registration `Unit`, in `Verity.AuditBound`'s form. -/
theorem Flock.SecurityProofs.RecursiveAuditBoundCoin : Flock.Guarantees.RecursiveAuditBoundCoin := by
  classical
  intro Coin W _ _ _ H I Sv K mI Mo n Reg Dh Hc L B Correct stmt reads opens M x₀ hWF hKS hLC hSv hO hO₀ hx₀ hJ hM σ hCR δ
  set R₀ := σ.1
  set L₀ := L R₀
  set T := tagged L₀
  set stmt' : Finset (Fin (Fintype.card (L₀.Ω × Option (Fin n)))) → Unit → Hash.Msg :=
    fun S _ => stmtAt L₀ (stmt R₀) x₀ S
  set B' : Binding Unit Hc := { B with cs := fun _ => B.cs R₀ }
  set Correct' : Unit → B'.Trace → Fin (Fintype.card (L₀.Ω × Option (Fin n))) → Prop :=
    fun _ tr u => ∀ v, (untag L₀ u).2 = some v → Correct R₀ tr v
  set reads' : Fin (Fintype.card (L₀.Ω × Option (Fin n))) → Finset B'.Pos := fun u => ((untag L₀ u).2.map reads).getD ∅
  set opens' : Finset (Fin (Fintype.card (L₀.Ω × Option (Fin n)))) → Unit →
      Fin (Fintype.card (L₀.Ω × Option (Fin n))) → W → B'.Pos → B'.Opening :=
    fun S _ u w c => ((untag L₀ u).2.map fun u' => opens R₀ (coinOf L₀ S) u' w c).getD default
  set τ' : Cont T Unit (proofSystemOf H K.V stmt').session () := fun ω => ⟨(σ.2 ω).1, fun _ => PUnit.unit⟩
  set σ' : Strategy (audit T Unit (proofSystemOf H K.V stmt').session) :=
    ⟨(), fun ω => Verity.Game.Strategy.toMap _ _ (τ' ω)⟩
  have hcont : cont σ' = τ' := rfl
  have hO' : ∀ S R', K.OuterKSAt H (stmt' S R') ∧ 0 ≤ K.κO (stmt' S R') ∧ 0 ≤ K.γO (stmt' S R') ∧
      K.Contract H (Sv.IVStar I.V I.ans) (fun v => I.opens (Sv.openView v)) (stmt' S R') :=
    fun S _ => by
      change K.OuterKSAt H (stmtAt L₀ (stmt R₀) x₀ S) ∧ 0 ≤ K.κO (stmtAt L₀ (stmt R₀) x₀ S) ∧
        0 ≤ K.γO (stmtAt L₀ (stmt R₀) x₀ S) ∧
        K.Contract H (Sv.IVStar I.V I.ans) (fun v => I.opens (Sv.openView v)) (stmtAt L₀ (stmt R₀) x₀ S)
      rw [stmtAt]
      split
      · exact hO R₀ _
      · exact hO₀
  have hJ' : (jointOf H K.V (K.R I) (K.E I Sv) stmt' B' reads' opens').Sound Correct' := by
    rintro S ⟨⟩ u ⟨w, k⟩ huS hrel
    obtain ⟨w', rfl, hw'⟩ := (rel_ext_iff _ _ _).1 hrel
    change K.R I (stmtAt L₀ (stmt R₀) x₀ S) w' at hw'
    rw [stmtAt] at hw'
    by_cases hS : S = (tagged L₀).draw (coinOf L₀ S)
    swap
    · rw [if_neg hS] at hw'
      exact absurd hw' (hx₀ w')
    rw [if_pos hS] at hw'
    have hmem : ∀ v, (untag L₀ u).2 = some v → v ∈ L₀.draw (coinOf L₀ S) := by
      intro v hv
      have huS' := huS
      rw [hS] at huS'
      rcases mem_tagged.1 huS' with h | ⟨v', hv', h⟩
      · rw [h] at hv
        simp at hv
      · rw [h] at hv
        obtain rfl : v' = v := by simpa using hv
        exact hv'
    rcases hu : (untag L₀ u).2 with _ | v
    · refine ⟨fun c hc => ?_, fun tr _ v hv => ?_⟩
      · change c ∈ reads' u at hc
        simp [reads', hu] at hc
      · exact absurd (hu.symm.trans hv) (by simp)
    · obtain ⟨hcom, hcor⟩ := hJ R₀ (coinOf L₀ S) v w' (hmem v hu) hw'
      refine ⟨fun c hc => ?_, fun tr htr v' hv' => ?_⟩
      · simp only [jointOf, reads', opens', hu, Option.map_some, Option.getD_some] at hc ⊢
        exact hcom c hc
      · obtain rfl : v = v' := Option.some.inj (hu.symm.trans hv')
        refine hcor tr fun c hc => ?_
        have := htr c (by simp only [jointOf, reads', hu, Option.map_some, Option.getD_some]; exact hc)
        simpa only [jointOf, opens', hu, Option.map_some, Option.getD_some] using this
  have hb := Flock.SecurityProofs.RecursiveSampledKS Coin W H I Sv K mI Mo _ Unit Dh Hc T B' Correct' stmt' reads'
    opens' M hWF hKS hLC hSv hO' hJ' hM σ' hCR
  have hprob : prob (fun o => o = true) (.send Reg fun R => .coin (L R).Ω fun ω => sessionOf H K.V (stmt R ω)) σ =
      prob (fun o => o.1 = true) (audit T Unit (proofSystemOf H K.V stmt').session) σ' := by
    show avg (fun ω => prob _ (sessionOf H K.V (stmt R₀ ω)) (σ.2 ω)) =
      avg (fun ω => prob _ ((sessionOf H K.V (stmt' (T.draw ω) ())).map fun ok => (ok, T.draw ω))
        (Verity.Game.Strategy.toMap _ _ (τ' ω)))
    refine congrArg avg (funext fun ω => ?_)
    rw [prob_map]
    exact prob_sessionOf H K.V _ (stmtAt_tagged L₀ (stmt R₀) x₀ ω).symm _ _ rfl
  have hmiss : ∀ tr : B.Trace, T.missProb (incorrect B' Correct' () tr) = L₀.missProb (incorrect B Correct R₀ tr) := by
    intro tr
    refine congrArg Game.prCoin (funext fun ω => propext ?_)
    simp only [Finset.disjoint_left, mem_incorrect, Correct', not_forall, exists_prop, not_not]
    constructor
    · intro h v hv
      by_contra hc
      exact h (mem_tagged.2 (Or.inr ⟨v, hv, untag_equivFin L₀ _⟩)) ⟨v, by rw [untag_equivFin], hc⟩
    · rintro h u hu ⟨x, hx, hc⟩
      rcases mem_tagged.1 hu with hu' | ⟨v, hv, hu'⟩
      · rw [hu'] at hx
        simp at hx
      · rw [hu'] at hx
        obtain rfl : v = x := by simpa using hx
        exact hc (h hv)
  have herr : avg (fun ω => ⨆ u ∈ T.draw ω,
      ENNReal.ofReal (K.err I Sv H (proverOf (x := stmt' (T.draw ω) ()) (cont σ' ω)) (stmt' (T.draw ω) ()) mI Mo)) ≤
      avg (fun ω => ENNReal.ofReal (K.err I Sv H (σ.2 ω).1 (stmt R₀ ω) mI Mo)) := by
    refine avg_mono fun ω => iSup₂_le fun _ _ => le_of_eq ?_
    exact congrArg (fun x => ENNReal.ofReal (K.err I Sv H (σ.2 ω).1 x mI Mo)) (stmtAt_tagged L₀ (stmt R₀) x₀ ω)
  by_cases hE : L₀.missProb (incorrect B Correct R₀
      (((jointOf H K.V (K.R I) (K.E I Sv) stmt' B' reads' opens').toTrials T).plur () τ')) ≤ δ
  · refine (Verity.Game.prob_mono (fun o h => h.1) _ _).trans ?_
    rw [hprob]
    refine hb.trans ?_
    rw [add_assoc]
    exact add_le_add ((hmiss _).trans_le hE) (add_le_add herr le_rfl)
  · exact ((Verity.Game.prob_mono (q := fun _ => False) (fun o h => hE h.2) _ _).trans
      (Verity.Game.prob_false _ _).le).trans (by simp)
