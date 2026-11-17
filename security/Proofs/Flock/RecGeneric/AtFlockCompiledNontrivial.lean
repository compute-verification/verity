module

public import Proofs.Flock.RecGeneric.AtFlockNontrivial
public import Proofs.Flock.RecGeneric.AtFlockCompiledContract
public import Proofs.Flock.RecGeneric.AtFlockServer

/-! `RecursiveKSCompiledEvery`'s conclusion is false at the compiled record whose verifier accepts everything, from the one named assumption `RecDeploy.KSFixture`. -/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion
open Claims.SamplingGeneric (Arg)

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

omit [Inhabited Coin] in
theorem RunFinder.PrAt_log_nil {Ω C : Type} [Fintype Ω] [Nonempty Ω] [Fintype C] [Nonempty C] {n : ℕ}
    (H : Msg → Msg) (F : RunFinder Coin) (m : RunModel Coin Ω n C) (x : Msg) (hlog : ∀ ω u, m.log ω u = []) :
    F.PrAt H m x = 0 := by
  have hfm : ∀ l : List RunExp, (l.flatMap fun _ => ([] : List (Msg × Msg))) = [] :=
    fun _ => List.flatMap_eq_nil_iff.2 fun _ _ => rfl
  unfold RunFinder.PrAt
  split_ifs with h
  · unfold Finder.Pr
    refine (Pr_congr fun t => iff_false_intro ?_).trans Pr_false
    rintro ⟨u, v, hs, -⟩
    simp only [RunFinder.finder, hlog, hfm, findColl, reduceCtorEq] at hs
  · rfl

omit [Nonempty Coin] in
theorem Comp.PrOut_of_wits_none (K : Comp Coin) (H : Msg → Msg) {Ω : Type} [Fintype Ω] (P : Prover Ω) (x : Msg)
    (hW : ∀ ω u, K.wits H P x ω u = none) : K.PrOut H P x = 0 := by
  unfold Comp.PrOut
  refine (Pr_congr fun t => iff_false_intro ?_).trans Pr_false
  rintro ⟨w, hw, -⟩
  cases (hW t.1 t.2).symm.trans hw

omit [Fintype Coin] [Nonempty Coin] in
theorem Comp.rerunCalls_of_wits_none (K : Comp Coin) (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg)
    (hW : ∀ ω u, K.wits H P x ω u = none) (ω : Ω) (u : (Fin (K.V.k x) → Coin) × K.EO.C x) :
    K.rerunCalls H P x ω u = [] :=
  congrArg (fun o : Option K.WO => (o.map (K.calls H x u.1)).getD []) (hW ω u)

variable {Col T Col' T' : Type}

theorem Compiled.EO_none_of_stmts_nil (DO : Claims.Flock.Deploy Coin Col' T') (L : Compiled Coin (Arg.ofFlock DO))
    (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (hs : ∀ ms, L.stmts x ms = []) (hS : 0 < L.nS x)
    (hP : DO.parse [] = none) (ω : Ω) (r : Fin (L.V.k x) → Coin) (c : L.EO.C x) :
    (L.EO.eval H P x ω r c).1 = none := by
  rcases h : (L.EO.eval H P x ω r c).1 with _ | ws
  · rfl
  obtain ⟨-, hlen, hws⟩ := Compiled.eo_some_staged L H P x ω r c h
  have h0 : 0 < ws.length := by
    rw [hlen]
    exact hS
  obtain ⟨h1, h2⟩ := hws 0 h0
  rw [h1, hs] at h2
  have hc : (Arg.ofFlock DO).check [] (ws[0]'h0).2 = false := by
    show Claims.Flock.check DO.parse [] _ = false
    unfold Claims.Flock.check
    rw [hP]
  exact absurd (hc.symm.trans h2) Bool.false_ne_true

theorem Compiled.prBad_silent (DO : Claims.Flock.Deploy Coin Col' T') (L : Compiled Coin (Arg.ofFlock DO))
    {W : Type} (I : Proto Coin W) (S : Server) (H : Msg → Msg) (x : Msg)
    (hv : ∀ as cs, L.V.next x as cs = .ret .acc) (hk : 0 < L.V.k x) (hs : ∀ ms, L.stmts x ms = [])
    (hS : 0 < L.nS x) (hP : DO.parse [] = none) :
    PrBad H L.V (L.toComp.R I) (L.toComp.E I S) x Claims.Witness.silentP = 1 :=
  have hO : ∀ (ω : Unit) (d : Fin (L.toComp.V.k x) → Coin) (c : L.toComp.EO.C x),
      (L.toComp.EO.eval H Claims.Witness.silentP x ω d c).1 = none :=
    fun ω d c => Compiled.EO_none_of_stmts_nil DO L H Claims.Witness.silentP x hs hS hP ω d c
  prBad_acc H L.V (L.toComp.R I) (L.toComp.E I S) x Claims.Witness.silentP hv hk fun ω r₀ c =>
    Comp.E_none_of_EO_none L.toComp I S H Claims.Witness.silentP x ω r₀ c (hO ω)

theorem Comp.err_ofFlock_quiet (K : Comp Coin) (Bd : Bound Coin Col T) (decI : List (ℕ × Col) → List T)
    (Cp : Caps Coin) (S : Server) (H : Msg → Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg)
    (M R : ℕ) (hB : PrB H K.V P x = 0) (hPO : K.PrOut H P x = 0)
    (hIn : ∀ F : RunFinder Coin, F.PrAt H (K.innerModel (S.IVStar Bd.DE.V Bd.ans) H P x) (K.xI x) = 0)
    (hVer : ∀ F : RunFinder Coin, F.PrAt H (K.verModel (S.IVStar Bd.DE.V Bd.ans) H P x) (K.xI x) = 0) :
    K.err (Proto.ofFlock Bd decI Cp) S H P x (![0, M, R] : Fin 3 → ℕ) M =
      K.κ (Proto.ofFlock Bd decI Cp) x +
        (4 * (Cp.width (K.xI x) : ℝ) + K.β (Proto.ofFlock Bd decI Cp) x) * (1 / ((M : ℝ) + 1)) +
        4 * (((Bd.DE.k (K.xI x) : ℝ) + 1) / ((R : ℝ) + 1)) := by
  show K.κ (Proto.ofFlock Bd decI Cp) x + Real.sqrt (K.γ (Proto.ofFlock Bd decI Cp) x * PrB H K.V P x) +
      ∑ i : Fin 3, (![termB Bd.DE.V.k, termFC Cp, termF0 Bd.DE.k] i).w (K.xI x) *
        (((![termB Bd.DE.V.k, termFC Cp, termF0 Bd.DE.k] i).F (![0, M, R] i)).PrAt H
            (K.innerModel (S.IVStar Bd.DE.V Bd.ans) H P x) (K.xI x) +
          (![termB Bd.DE.V.k, termFC Cp, termF0 Bd.DE.k] i).tail (K.xI x) (![0, M, R] i)) +
      K.reads (Proto.ofFlock Bd decI Cp) x * K.PrOut H P x +
      (forkPlan (Coin := Coin) Bd.DE.V.k).PrAt H (K.verModel (S.IVStar Bd.DE.V Bd.ans) H P x) (K.xI x) +
      K.β (Proto.ofFlock Bd decI Cp) x *
        ((linkPlan (Coin := Coin) Bd.DE.V.k M).PrAt H (K.verModel (S.IVStar Bd.DE.V Bd.ans) H P x) (K.xI x) +
          1 / ((M : ℝ) + 1) + K.PrOut H P x) = _
  rw [Fin.sum_univ_three]
  show K.κ (Proto.ofFlock Bd decI Cp) x + Real.sqrt (K.γ (Proto.ofFlock Bd decI Cp) x * PrB H K.V P x) +
      (4 * (Bd.DE.V.k (K.xI x) : ℝ) *
          ((forkPlan (Coin := Coin) Bd.DE.V.k).PrAt H (K.innerModel (S.IVStar Bd.DE.V Bd.ans) H P x) (K.xI x) + 0) +
        4 * (Cp.width (K.xI x) : ℝ) *
          ((capPlan Cp M).PrAt H (K.innerModel (S.IVStar Bd.DE.V Bd.ans) H P x) (K.xI x) + 1 / ((M : ℝ) + 1)) +
        4 * ((conflictPlan (Coin := Coin) Bd.DE.k R).PrAt H (K.innerModel (S.IVStar Bd.DE.V Bd.ans) H P x)
            (K.xI x) + ((Bd.DE.k (K.xI x) : ℝ) + 1) / ((R : ℝ) + 1))) +
      K.reads (Proto.ofFlock Bd decI Cp) x * K.PrOut H P x +
      (forkPlan (Coin := Coin) Bd.DE.V.k).PrAt H (K.verModel (S.IVStar Bd.DE.V Bd.ans) H P x) (K.xI x) +
      K.β (Proto.ofFlock Bd decI Cp) x *
        ((linkPlan (Coin := Coin) Bd.DE.V.k M).PrAt H (K.verModel (S.IVStar Bd.DE.V Bd.ans) H P x) (K.xI x) +
          1 / ((M : ℝ) + 1) + K.PrOut H P x) = _
  rw [hB, hPO, hIn, hIn, hIn, hVer, hVer]
  simp only [mul_zero, Real.sqrt_zero]
  ring

theorem Comp.err_ofFlock_silent (K : Comp Coin) (Bd : Bound Coin Col T) (decI : List (ℕ × Col) → List T)
    (Cp : Caps Coin) (S : Server) (H : Msg → Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg)
    (M R : ℕ) (hB : PrB H K.V P x = 0) (hW : ∀ ω u, K.wits H P x ω u = none)
    (hown : ∀ ω u, K.ownAt H P x ω u = []) :
    K.err (Proto.ofFlock Bd decI Cp) S H P x (![0, M, R] : Fin 3 → ℕ) M =
      K.κ (Proto.ofFlock Bd decI Cp) x +
        (4 * (Cp.width (K.xI x) : ℝ) + K.β (Proto.ofFlock Bd decI Cp) x) * (1 / ((M : ℝ) + 1)) +
        4 * (((Bd.DE.k (K.xI x) : ℝ) + 1) / ((R : ℝ) + 1)) :=
  Comp.err_ofFlock_quiet K Bd decI Cp S H P x M R hB (Comp.PrOut_of_wits_none K H P x hW)
    (fun F => RunFinder.PrAt_log_nil H F _ _ (Comp.rerunCalls_of_wits_none K H P x hW))
    fun F => RunFinder.PrAt_log_nil H F _ _ fun ω u => by
      show K.rerunCalls H P x ω u ++ K.ownAt H P x ω u = []
      rw [Comp.rerunCalls_of_wits_none K H P x hW ω u, hown ω u, List.append_nil]

theorem Compiled.not_bound_silent (DO : Claims.Flock.Deploy Coin Col' T') (L : Compiled Coin (Arg.ofFlock DO))
    (Bd : Bound Coin Col T) (decI : List (ℕ × Col) → List T) (Cp : Caps Coin) (S : Server) (H : Msg → Msg)
    (x : Msg) (hv : ∀ as cs, L.V.next x as cs = .ret .acc) (hk : 0 < L.V.k x) (hs : ∀ ms, L.stmts x ms = [])
    (hS : 0 < L.nS x) (hP : DO.parse [] = none) (hown : ∀ ms, L.own H x ms = [])
    (hκ : L.toComp.κ (Proto.ofFlock Bd decI Cp) x < 1 / 2)
    (h : ∀ (m : Fin (Proto.ofFlock Bd decI Cp).nT → ℕ) (M : ℕ),
      PrBad H L.V (L.toComp.R (Proto.ofFlock Bd decI Cp)) (L.toComp.E (Proto.ofFlock Bd decI Cp) S) x
          Claims.Witness.silentP ≤
        L.toComp.err (Proto.ofFlock Bd decI Cp) S H Claims.Witness.silentP x m M) :
    False := by
  have hE := Comp.err_ofFlock_silent L.toComp Bd decI Cp S H Claims.Witness.silentP x
    ⌈4 * (4 * (Cp.width (L.toComp.xI x) : ℝ) + L.toComp.β (Proto.ofFlock Bd decI Cp) x)⌉₊
    (16 * (Bd.DE.k (L.toComp.xI x) + 1)) (prB_acc H _ _ x hv)
    (fun ω u => Compiled.EO_none_of_stmts_nil DO L H Claims.Witness.silentP x hs hS hP ω u.1 u.2)
    fun ω u => hown (run H Claims.Witness.silentP L.V x ω u.1).msgs
  have h1 := (Compiled.prBad_silent DO L (Proto.ofFlock Bd decI Cp) S H x hv hk hs hS hP).symm.trans_le
    ((h _ _).trans_eq hE)
  have h2 := add_lt_add (add_lt_add hκ (Claims.Witness.Recursion.tail_lt_quarter
    (4 * (Cp.width (L.toComp.xI x) : ℝ) + L.toComp.β (Proto.ofFlock Bd decI Cp) x)))
    (Claims.Witness.Recursion.f0_tail_lt (Bd.DE.k (L.toComp.xI x)))
  exact absurd (h1.trans_lt h2) (by norm_num)

omit [Fintype Coin] [Nonempty Coin] in
theorem Bound.deploy_κ (Bd : Bound Coin Col T) (y : Msg) : Claims.Flock.κ Bd.deploy y = Claims.Flock.κ Bd.DE y := by
  simp only [Claims.Flock.κ, Claims.Flock.Deploy.k, Bound.deploy]

omit [Fintype Coin] [Nonempty Coin] in
theorem Bound.deploy_waits (Bd : Bound Coin Col T) (y : Msg) : Bd.deploy.waits y = Bd.DE.waits y := by
  simp only [Bound.deploy]

omit [Fintype Coin] [Nonempty Coin] in
theorem Bound.deploy_k (Bd : Bound Coin Col T) (y : Msg) : Bd.deploy.k y = Bd.DE.k y := by
  simp only [Claims.Flock.Deploy.k, Bound.deploy]

theorem κ_ofFlock_deploy (D : RecAt Coin) (IV : Verifier Coin) (Bd : Bound Coin Col T)
    (decI : List (ℕ × Col) → List T) (x : Msg) :
    Claims.Recursion.κ (D.record IV) (Inner.ofFlock Bd.deploy decI) x =
      Claims.Flock.κ Bd.DE (xI (D.record IV) x) +
        ((1 + Bd.DE.waits (xI (D.record IV) x) * Bd.DE.k (xI (D.record IV) x) : ℕ) : ℝ) *
          ∑ s : Fin ((D.record IV).nS x), ((D.record IV).nums x s).κ := by
  simp only [Claims.Recursion.κ, Claims.Recursion.NI, Inner.ofFlock, Bound.deploy_κ, Bound.deploy_waits,
    Bound.deploy_k]

theorem Compiled.κ_ofFlock {A : Arg Coin} (L : Compiled Coin A) (Bd : Bound Coin Col T)
    (decI : List (ℕ × Col) → List T) (Cp : Caps Coin) (x : Msg) :
    L.toComp.κ (Proto.ofFlock Bd decI Cp) x =
      Claims.Flock.κ Bd.DE (L.xI x) + ((1 + Bd.DE.waits (L.xI x) * Bd.DE.k (L.xI x) : ℕ) : ℝ) *
        ∑ s : Fin (L.nS x), L.κb x s :=
  rfl

section AcceptAll

variable (D : RecAt Coin) (Bd : Bound Coin Col T) (DO : Claims.Flock.Deploy Coin Col' T') (Cp : Caps Coin)
  (x : Msg)

omit [Fintype Coin] [Nonempty Coin] in
theorem compiledAt_acceptAll_next : ∀ as cs, (compiledAt D.acceptAll Bd DO Cp).V.next x as cs = .ret .acc :=
  fun _ _ => rfl

omit [Fintype Coin] [Nonempty Coin] in
theorem compiledAt_acceptAll_stmts : ∀ ms, (compiledAt D.acceptAll Bd DO Cp).stmts x ms = [] :=
  fun _ => rfl

omit [Fintype Coin] [Nonempty Coin] in
theorem compiledAt_acceptAll_own (H : Msg → Msg) : ∀ ms, (compiledAt D.acceptAll Bd DO Cp).own H x ms = [] :=
  fun _ => rfl

omit [Fintype Coin] [Nonempty Coin] in
theorem compiledAt_acceptAll_k (hIV : IVStarOfBound Bd) :
    (compiledAt D.acceptAll Bd DO Cp).V.k x = (D.record Bd.IVStar).V.k x := by
  show (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)).V.k x = _
  rw [hIV]

omit [Fintype Coin] [Nonempty Coin] in
theorem compiledAt_acceptAll_nS (hIV : IVStarOfBound Bd) :
    (compiledAt D.acceptAll Bd DO Cp).nS x = (D.record Bd.IVStar).nS x := by
  show (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)).nS x = _
  rw [hIV]

theorem compiledAt_acceptAll_κ (hIV : IVStarOfBound Bd) :
    (compiledAt D.acceptAll Bd DO Cp).toComp.κ (Proto.ofFlock Bd Bd.deploy.decode Cp) x =
      Claims.Recursion.κ (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy Bd.deploy.decode) x := by
  rw [Compiled.κ_ofFlock, κ_ofFlock_deploy]
  show Claims.Flock.κ Bd.DE (xI (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)) x) +
      ((1 + Bd.DE.waits (xI (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)) x) *
        Bd.DE.k (xI (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)) x) : ℕ) : ℝ) *
      ∑ s : Fin ((D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)).nS x),
        ((D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)).nums x s).κ = _
  rw [hIV]

end AcceptAll

theorem not_ksCompiledOn_acceptAll (H : Msg → Msg) (D : RecAt Coin) (Bd : Bound Coin Col T)
    (DO : Claims.Flock.Deploy Coin Col' T') (Cp : Caps Coin) (G : Msg → Prop) (x : Msg) (hx : G x)
    (hIV : IVStarOfBound Bd) (hk : 0 < (D.record Bd.IVStar).V.k x) (hS : 0 < (D.record Bd.IVStar).nS x)
    (hP : DO.parse [] = none)
    (hκ : Claims.Recursion.κ (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy Bd.deploy.decode) x < 1 / 2) :
    ¬ KSCompiledOn H D.acceptAll Bd DO Cp G := fun h =>
  Compiled.not_bound_silent DO (compiledAt D.acceptAll Bd DO Cp) Bd Bd.deploy.decode Cp (Server.ofBound Bd) H x
    (compiledAt_acceptAll_next D Bd DO Cp x) (hk.trans_eq (compiledAt_acceptAll_k D Bd DO Cp x hIV).symm)
    (compiledAt_acceptAll_stmts D Bd DO Cp x) (hS.trans_eq (compiledAt_acceptAll_nS D Bd DO Cp x hIV).symm) hP
    (compiledAt_acceptAll_own D Bd DO Cp x H) ((compiledAt_acceptAll_κ D Bd DO Cp x hIV).trans_lt hκ)
    fun m M => h x hx Unit Claims.Witness.silentP m M

end Claims.RecGeneric

namespace Flock.Guarantees

open Claims (Msg Out)
open FlockDeploy (Coin FinalP)

/-- **`RecursiveKSCompiledEvery`'s nontrivial lemma**: `RecursiveKSCompiledEvery`'s conclusion,
`RecursiveKSCompiledAtRec` at the deployed values, is false at the verifier that accepts everything
(`RecAt.acceptAll` of the deployment's compiled record), at every brings, from the named assumption
`RecDeploy.KSFixture` (`recursiveKSEvery_nontrivial`'s): some bytes the deployed readers read as a dense compiled
statement with a V\* session whose inner statement is in the plain scope, with a live-coin record at whose transcript
its V\* parts are in scope. The silent prover is accepted on every run and `E*` extracts nothing, so
`Pr[accept ∧ ¬extract] = 1`; every finder term, the gated link's included, is `0`; the tails are below `1/4` each; and
`λ₂ ≥ 100` at the deployed decoders gives `κ < 1/2`. -/
theorem recursiveKSCompiledEvery_nontrivial (h : RecDeploy.KSFixture) :
    ∃ (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ) (hd : Msg → Flock.RecRecord.Record)
      (recRest : Msg → List (Msg × Out) → List UInt8), ∀ b : RecDeploy.Brings,
      ¬ RecursiveKSCompiledAtRec
        (RecDeploy.recBoundAt _root_.Flock.Readable.relation
          { RecDeploy.deployedB b with tr := RecDeploy.recTr aux nOf hd recRest }
          (RecDeploy.DeployedR.layR RecDeploy.OpenRows.elemR)).acceptAll
        RecDeploy.DeployedR.rdR RecDeploy.DeployedR.finalR RecDeploy.DeployedR.wER RecDeploy.DeployedR.wR := by
  obtain ⟨x, aux, nOf, hd, recRest, hP, hO, hn, hS⟩ := h
  obtain ⟨s, spec, sts, hin⟩ := plainIn_of_parses hP
  refine ⟨aux, nOf, hd, recRest, fun b => ?_⟩
  have hO' : ∀ j < RecDeploy.nS x, ∃ i spec' sts', RecDeploy.PartIn
      (RecDeploy.staged ({ RecDeploy.deployedB b with tr := RecDeploy.recTr aux nOf hd recRest } : RecDeploy.Brings).st
        x (RecDeploy.pick ({ RecDeploy.deployedB b with tr := RecDeploy.recTr aux nOf hd recRest } :
          RecDeploy.Brings).st x j) j) i spec' sts' := hO
  let Bd := RecDeploy.bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR RecDeploy.DeployedR.rdR)
    (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation RecDeploy.DeployedR.rdR).cols RecDeploy.DeployedR.wER
    FlockDeploy.joinR RecDeploy.rowInDeployed RecDeploy.DeployedR.finalR RecDeploy.DeployedR.wR
  let D := RecDeploy.recBoundAt _root_.Flock.Readable.relation
    { RecDeploy.deployedB b with tr := RecDeploy.recTr aux nOf hd recRest }
    (RecDeploy.DeployedR.layR RecDeploy.OpenRows.elemR)
  exact Claims.RecGeneric.not_ksCompiledOn_acceptAll FlockDeploy.sha512 D Bd
    (RecDeploy.deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep (fun _ => True) x trivial
    (Claims.RecGeneric.ivStarOfBound Bd) (RecDeploy.recBound_k_pos _ _ _ _ _ _ _ _ _ _ x) hP.2.1
    (RecDeploy.zkRec_parse_nil _)
    (Claims.Recursion.κ_lt_half_of_lamLin2
      (RecDeploy.lamLin2_deployed_ge_dep _root_.Flock.Readable.relation
        { RecDeploy.deployedB b with tr := RecDeploy.recTr aux nOf hd recRest }
        (RecDeploy.DeployedR.layR RecDeploy.OpenRows.elemR) (FlockDeploy.nextR RecDeploy.DeployedR.rdR)
        (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation RecDeploy.DeployedR.rdR).cols
        RecDeploy.DeployedR.wER FlockDeploy.joinR RecDeploy.rowInDeployed RecDeploy.DeployedR.finalR
        RecDeploy.DeployedR.wR Bd.deploy.decode hin hO' hn hS (n := 0) (by norm_num) (v := 0) (by norm_num)))

end Flock.Guarantees
