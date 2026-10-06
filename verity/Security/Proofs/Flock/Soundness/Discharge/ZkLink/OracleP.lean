import Proofs.Flock.Soundness.Discharge.ZkLink.Push

/-!
# The `--zk` table with every committed object an oracle

The interactive-oracle `--zk` table (`tableAfterZK`) receives each rep's pads as two words and `τ` in the clear. The
primed table (`tabOrcP`) receives both as oracles (`Oracle F`), the pads on lanes `0` and `1` (`padWords`) and `τ` at
`(0, 0)`, so that every table node of the compiled table (level caps, pads roots, `τ`'s commitments) faces an oracle.
Each oracle the primed prover sends gives words or a `τ` the interactive-oracle prover can send, so the primed table is
no easier (`value_repZP_le`) and keeps `TabZK.after_sound`'s bound (`tabOrcP_sound`). Inside a session, the primed
table at its position accepts uncommitted no more often for an optimal prover (`orcSessZ_value_le`).
-/

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting ZkSession
open scoped ENNReal NNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
variable {D Sl : Type}

/-- The pads words an oracle carries: `C_h` on lane `0`, `C_μ` on lane `1`. -/
def padWords {N : ℕ} (O : Oracle F) : (Fin N → F) × (Fin N → F) :=
  (fun p => O 0 p.val, fun p => O 1 p.val)

/-- **The inner proof with `τ` an oracle**, read at `(0, 0)`. -/
def innerGameP {N : ℕ} (dom : Fin N ↪ F) (s nP q : ℕ) (pos : F → Fin N) {nA : ℕ} (rows : Fin nA → Lin F s)
    (tr : Triple s) (Ch Cμ : Fin N → F) : Game Bool := do
  let ε ← draw F
  let ρσ ← recv (F × F)
  let γ ← draw (Fin (nA + 3) → F)
  let Oτ ← recv (Oracle F)
  let β ← draw F
  let y ← recv (Fin (s + nP) → F)
  let w ← draw (Fin q → F)
  pure (decide (innerCheck dom pos rows tr Ch Cμ ε ρσ γ (Oτ 0 0) β y w))

/-- **A `--zk` rep with its pads an oracle** (`repZK` with `padWords` and `innerGameP`). -/
def repZP {N : ℕ} (dom : Fin N ↪ F) (s nP q : ℕ) (pos : F → Fin N) {nA : ℕ} (tr : Triple s)
    (phase : Game ((Fin nA → Lin F s) × Bool)) : Game Bool := do
  let O ← recv (Oracle F)
  let o ← phase
  let b ← innerGameP dom s nP q pos o.1 tr (padWords O).1 (padWords O).2
  pure (o.2 && b)

theorem value_innerGameP_le {N : ℕ} (dom : Fin N ↪ F) (s nP q : ℕ) (pos : F → Fin N) {nA : ℕ}
    (rows : Fin nA → Lin F s) (tr : Triple s) (Ch Cμ : Fin N → F) {γ' : Type} (k : Bool → Game γ')
    (p : γ' → Prop) :
    value p ((innerGameP dom s nP q pos rows tr Ch Cμ).bind k) ≤
      value p ((innerGame dom s nP q pos rows tr Ch Cμ).bind k) := by
  simp only [innerGameP, innerGame, bind_eq, pure_eq, draw, recv, bind_coin, bind_send, bind_ret, value_coin,
    value_send]
  refine Finset.sum_le_sum fun ε _ => mul_le_mul_right ?_ _
  refine iSup_mono fun ρσ => Finset.sum_le_sum fun γ _ => mul_le_mul_right ?_ _
  exact iSup_le fun Oτ => le_iSup_of_le (Oτ 0 0) le_rfl

/-- **The primed rep is no easier** than the interactive-oracle rep. -/
theorem value_repZP_le {N : ℕ} (dom : Fin N ↪ F) (s nP q : ℕ) (pos : F → Fin N) {nA : ℕ} (tr : Triple s)
    (phase : Game ((Fin nA → Lin F s) × Bool)) (p : Bool → Prop) :
    value p (repZP dom s nP q pos tr phase) ≤ value p (repZK dom s nP q pos tr phase) := by
  simp only [repZP, repZK, bind_eq, pure_eq, recv, bind_send, bind_ret, value_send]
  refine iSup_le fun O => le_iSup_of_le (padWords O) ?_
  exact value_bind_mono phase fun o => value_innerGameP_le dom s nP q pos o.1 tr _ _ _ p

/-! ## Reordered rows -/

/-- The batch's rows reordered by `π`, its last three (the triple's) kept. -/
def appendPerm {nA : ℕ} (π : Equiv.Perm (Fin nA)) : Equiv.Perm (Fin (nA + 3)) :=
  finSumFinEquiv.symm.trans ((Equiv.sumCongr π (Equiv.refl _)).trans finSumFinEquiv)

omit [Field F] [Fintype F] [DecidableEq F] in
theorem append_comp_perm {s nA : ℕ} (π : Equiv.Perm (Fin nA)) (rows : Fin nA → Lin F s) (T : Fin 3 → Lin F s) :
    Fin.append (rows ∘ π) T = Fin.append rows T ∘ appendPerm π := by
  funext i
  refine Fin.addCases (fun i => ?_) (fun i => ?_) i <;> simp [appendPerm, Fin.append_left, Fin.append_right]

omit [Fintype F] [DecidableEq F] in
theorem innerCheck_perm {N : ℕ} (dom : Fin N ↪ F) {s nP q : ℕ} (pos : F → Fin N) {nA : ℕ} (π : Equiv.Perm (Fin nA))
    (rows : Fin nA → Lin F s) (tr : Triple s) (Ch Cμ : Fin N → F) (ε : F) (ρσ : F × F) (γ : Fin (nA + 3) → F)
    (τ β : F) (y : Fin (s + nP) → F) (w : Fin q → F) :
    innerCheck dom pos (rows ∘ π) tr Ch Cμ ε ρσ γ τ β y w ↔
      innerCheck dom pos rows tr Ch Cμ ε ρσ (γ ∘ (appendPerm π).symm) τ β y w := by
  have hc : ∀ (R : Fin (nA + 3) → Lin F s) i,
      batchCoef γ (R ∘ appendPerm π) i = batchCoef (γ ∘ (appendPerm π).symm) R i := fun R i => by
    simp only [batchCoef, Function.comp]
    exact Fintype.sum_equiv (appendPerm π) _ _ fun k => by simp
  have ht : ∀ R : Fin (nA + 3) → Lin F s,
      batchT γ (R ∘ appendPerm π) = batchT (γ ∘ (appendPerm π).symm) R := fun R => by
    simp only [batchT, Function.comp]
    exact congrArg Neg.neg (Fintype.sum_equiv (appendPerm π) _ _ fun k => by simp)
  simp only [innerCheck, append_comp_perm, hc, ht]

/-- **The inner proof does not see its rows' order**: `γ` is uniform, so reordering the rows reorders its coefficients. -/
theorem value_innerGameP_perm {N : ℕ} (dom : Fin N ↪ F) (s nP q : ℕ) (pos : F → Fin N) {nA : ℕ}
    (π : Equiv.Perm (Fin nA)) (rows : Fin nA → Lin F s) (tr : Triple s) (Ch Cμ : Fin N → F) {γ' : Type}
    (k : Bool → Game γ') (p : γ' → Prop) :
    value p ((innerGameP dom s nP q pos (rows ∘ π) tr Ch Cμ).bind k) =
      value p ((innerGameP dom s nP q pos rows tr Ch Cμ).bind k) := by
  simp only [innerGameP, bind_eq, pure_eq, draw, recv, bind_coin, bind_send, bind_ret, value_coin, value_send]
  refine Finset.sum_congr rfl fun ε _ => congrArg (_ * ·) (iSup_congr fun ρσ =>
    Fintype.sum_equiv (Equiv.arrowCongr (appendPerm π) (Equiv.refl F)) _ _ fun γ => ?_)
  simp only [innerCheck_perm]
  rfl

/-- **The primed rep does not see its rows' order.** -/
theorem value_repZP_perm {N : ℕ} (dom : Fin N ↪ F) (s nP q : ℕ) (pos : F → Fin N) {nA : ℕ} (tr : Triple s)
    (π : Equiv.Perm (Fin nA)) (phase : Game ((Fin nA → Lin F s) × Bool)) (p : Bool → Prop) :
    value p (repZP dom s nP q pos tr (phase.map fun o => (o.1 ∘ π, o.2))) = value p (repZP dom s nP q pos tr phase) := by
  simp only [repZP, map, bind_eq, pure_eq, recv, bind_send, bind_assoc, bind_ret, value_send]
  refine iSup_congr fun O => ?_
  exact le_antisymm (value_bind_mono phase fun o => (value_innerGameP_perm dom s nP q pos π o.1 tr _ _ _ p).le)
    (value_bind_mono phase fun o => (value_innerGameP_perm dom s nP q pos π o.1 tr _ _ _ p).ge)

/-! ## The primed table -/

/-- **Rep `r` of the primed table** (`repZP` over the masked rep, its rows in flock-verify's order, as
`repMaskedZC`'s). -/
noncomputable def repZPAt (A : Arith F K) (T : TabZK F mPts) (C₀ : Oracle F) (r : ℕ)
    (extra : List (Claim F T.S.m)) : Game Bool :=
  repZP T.dom T.np T.nP T.q T.pos T.tr
    ((repMaskedZK A T.pads T.padsA T.tr T.S T.sch T.hm T.t T.c₀ r C₀ extra).map fun o => (o.1 ∘ rowOrder, o.2))

/-- **The primed table after its link points**, with the dummy last message (the compiled table's openings). -/
noncomputable def tabOrcP (A : Arith F K) (T : TabZK F mPts) (C₀ : Oracle F) (pts : Fin 2 → Fin mPts → F) :
    Game (Bool × Oracle F) :=
  (interleave (repZPAt A T C₀ 0 (extraClaims T.S T.ptLocal mPts pts))
    (repZPAt A T C₀ 1 (extraClaims T.S T.ptLocal mPts pts))).bind fun acc =>
      Game.send (OpensZ F D Sl) fun _ => Game.ret (acc.1 && acc.2, C₀)

theorem value_bind_dummy {α β : Type} (O : Type) [Nonempty O] (p : β → Prop) (f : α → β) (g : Game α) :
    value p (g.bind fun a => Game.send O fun _ => Game.ret (f a)) = value (fun a => p (f a)) g := by
  have h : (g.bind fun a => Game.send O fun _ => Game.ret (f a)) = Lock.withDummy O (g.map f) := by
    simp only [Lock.withDummy, map, bind_assoc, bind_ret]
  rw [h, Lock.value_withDummy, value_map]

/-- **The primed table keeps the table's bound**: whatever level-0 oracle `C₀`, it accepts with `C₀` uncommitted with
probability at most `T.bound`. -/
theorem tabOrcP_sound [Inhabited D] [Inhabited Sl] (A : Arith F K) (hA : A.Correct) (T : TabZK F mPts)
    (C₀ : Oracle F) :
    value (fun out : Bool × Oracle F => out.1 = true ∧ ¬ T.Committed A out.2)
      (.coin (Fin 2 → Fin mPts → F) fun pts => tabOrcP (D := D) (Sl := Sl) A T C₀ pts) ≤ T.bound := by
  have := T.hnP
  set ε := ENNReal.ofReal (padRepError T.sch (T.S.shape mPts) T.t T.e) + εInner T.N F T.q T.δ with hε
  unfold TabZK.bound
  rw [← hε]
  simp only [tabOrcP]
  by_cases hC : CommittedPad A T.S T.sch T.t T.c₀ C₀
  · rw [value_coin]
    refine le_of_eq_of_le (Finset.sum_eq_zero fun pts _ => ?_) zero_le
    rw [value_bind_dummy, value_congr (q := fun _ => False) (fun acc => by simp [TabZK.Committed, hC]),
      value_false, mul_zero]
  · refine (value_coin_le _ _ _
      (fun pts => ¬ RepDoomedW A T.S T.sch (padWits A T.sch T.t T.c₀ C₀) (extraClaims T.S T.ptLocal mPts pts))
      (ENNReal.ofReal (padLinkError T.sch (T.S.shape mPts) T.t)) (ε * ε)
      (link_sound_pad A hA T.S T.sch T.t T.hpad T.c₀ T.ptLocal mPts C₀ hC T.hlay) fun pts hpts => ?_).trans ?_
    · rw [not_not] at hpts
      rw [value_bind_dummy]
      have hrep := fun r => (value_repZP_perm T.dom T.np T.nP T.q T.pos T.tr rowOrder _ (· = true)).le.trans <|
        (value_repZP_le T.dom T.np T.nP T.q T.pos T.tr _ (· = true)).trans
        (rep_sound_pad_zk A hA T.S T.sch T.hshape T.hmsch T.hm T.t T.hpad T.c₀ T.e r mPts C₀ _
          (extraClaims_length T.S T.ptLocal mPts pts) hpts T.pads T.padsA T.tr T.dom T.nP T.q T.pos T.hpos T.δ T.hL
          T.hδ)
      calc value (fun acc : Bool × Bool =>
              (acc.1 && acc.2, C₀).1 = true ∧ ¬ T.Committed A (acc.1 && acc.2, C₀).2)
              (interleave (repZPAt A T C₀ 0 (extraClaims T.S T.ptLocal mPts pts))
                (repZPAt A T C₀ 1 (extraClaims T.S T.ptLocal mPts pts)))
          = value (fun acc : Bool × Bool => acc.1 = true ∧ acc.2 = true)
              (interleave (repZPAt A T C₀ 0 (extraClaims T.S T.ptLocal mPts pts))
                (repZPAt A T C₀ 1 (extraClaims T.S T.ptLocal mPts pts))) :=
            value_congr (fun acc => by simp [TabZK.Committed, hC]) _
        _ ≤ _ := value_interleave_le (fun b : Bool => b = true) (fun b : Bool => b = true) _ _
        _ ≤ ε * ε := mul_le_mul' (hrep 0) (hrep 1)
    · rw [sq]

/-! ## The primed table in a session -/

open Classical in
/-- The primed table's output as a compiled output: accepted when both reps accept and the table is not committed. -/
noncomputable def orcOutZ [Inhabited D] [Inhabited Sl] (A : Arith F K) (T : TabZK F mPts) (cap₀ : Cap D)
    (a : Bool × Oracle F) : OutZC F D Sl :=
  ⟨a.1 && decide (¬ T.Committed A a.2), cap₀, (PendZ.reject, PendZ.reject), default⟩

/-- **The oracle session**: the primed table in the compiled table's place. -/
noncomputable def orcSessZ [Inhabited D] [Inhabited Sl] (A : Arith F K) (T : TabZK F mPts)
    (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutZC F D Sl))) (C₀ : Oracle F) (cap₀ : Cap D) :
    Game (List (OutZC F D Sl)) :=
  Game.coin (Fin 2 → Fin mPts → F) fun pts =>
    batch (pre.map (· pts) ++ (tabOrcP (D := D) (Sl := Sl) A T C₀ pts).map (orcOutZ A T cap₀) :: post.map (· pts))

open Classical in
/-- **The oracle session is bounded by the table**: whatever the other games, the primed table in the session accepts
uncommitted with probability at most `T.bound`, for an optimal prover. -/
theorem orcSessZ_value_le [Inhabited D] [Inhabited Sl] (A : Arith F K) (hA : A.Correct) (T : TabZK F mPts)
    (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutZC F D Sl))) (C₀ : Oracle F) (cap₀ : Cap D) :
    value (fun outs : List (OutZC F D Sl) => (outs.getD pre.length dfltOutZC).accepted = true)
      (orcSessZ A T pre post C₀ cap₀) ≤ T.bound := by
  calc value (fun outs : List (OutZC F D Sl) => (outs.getD pre.length dfltOutZC).accepted = true)
        (orcSessZ A T pre post C₀ cap₀)
      ≤ value (fun o : OutZC F D Sl => o.accepted = true) (Game.coin (Fin 2 → Fin mPts → F) fun pts =>
          (tabOrcP (D := D) (Sl := Sl) A T C₀ pts).map (orcOutZ A T cap₀)) := by
        simp only [orcSessZ, value_coin]
        refine Finset.sum_le_sum fun pts _ => mul_le_mul_right ?_ _
        have := value_batch_getD_le (fun o : OutZC F D Sl => o.accepted = true) dfltOutZC
          ((tabOrcP (D := D) (Sl := Sl) A T C₀ pts).map (orcOutZ A T cap₀)) (post.map (· pts)) (pre.map (· pts))
        simpa only [List.length_map] using this
    _ = value (fun out : Bool × Oracle F => out.1 = true ∧ ¬ T.Committed A out.2)
          (.coin (Fin 2 → Fin mPts → F) fun pts => tabOrcP (D := D) (Sl := Sl) A T C₀ pts) := by
        simp only [value_coin, value_map]
        refine Finset.sum_congr rfl fun pts _ => ?_
        congr 1
        exact value_congr (fun a => by simp [orcOutZ]) _
    _ ≤ T.bound := tabOrcP_sound A hA T C₀

end FlockSoundness.Discharge.ZkLink
