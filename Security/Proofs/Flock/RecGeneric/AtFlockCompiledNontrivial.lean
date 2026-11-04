module

public import Proofs.Flock.RecGeneric.AtFlockNontrivial
public import Proofs.Flock.RecGeneric.AtFlockCompiledContract
public import Proofs.Flock.RecGeneric.AtFlockServer

/-!
# `RecursiveKSCompiledEvery` at the verifier that accepts everything (the Lean check's `nontrivial`)

`RecursiveKSCompiledEvery`'s nontrivial lemma, as `recursiveKSEvery_nontrivial` is `RecursiveKSEvery`'s: its
conclusion, `RecursiveKSCompiledAtRec` at the deployed values, is false at the compiled record whose verifier keeps its
rounds and accepts at its first round with no SHA-512 call (`RecAt.acceptAll`), from the one named assumption
`RecDeploy.KSFixture`. The silent prover is accepted on every run. The accepting record stages no V\* statement, so
`Compiled.EO`'s first session's witness would have to pass the outer Flock's check at the empty statement, which its
parser refuses: `E_O` gives no witnesses (`Compiled.EO_none_of_stmts_nil`), `E*` extracts nothing
(`Comp.E_none_of_EO_none`), `Pr[accept ∧ ¬extract] = 1`, and no rerun's witnesses fall outside `R_O`. No rerun logs a
call, so every run finder of `Comp.err` finds nothing (`RunFinder.PrAt_log_nil`), which leaves `κ` and the tails
(`Comp.err_ofFlock_quiet`); the tails are below `1/4` each at a large enough `M` and `R`, and `λ₂ ≥ 100` at the deployed
decoders gives `κ < 1/2`.
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion
open Claims.SamplingGeneric (Arg)

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

omit [Inhabited Coin] in
/-- **A run finder finds nothing where no run logs a call**: the evaluations it searches are empty. -/
theorem RunFinder.PrAt_log_nil {Ω C : Type} [Fintype Ω] [Nonempty Ω] [Fintype C] [Nonempty C] {n : ℕ}
    (H : Msg → Msg) (F : RunFinder Coin) (m : RunModel Coin Ω n C) (x : Msg) (hlog : ∀ ω u, m.log ω u = []) :
    F.PrAt H m x = 0 := by
  have hfm : ∀ l : List RunExp, (l.flatMap fun _ => ([] : List (Msg × Msg))) = [] :=
    fun _ => List.flatMap_eq_nil_iff.2 fun _ _ => rfl
  unfold RunFinder.PrAt
  split_ifs with h
  · unfold Finder.Pr
    refine (Pr_congr fun s => iff_false_intro ?_).trans Pr_false
    rintro ⟨u, v, hs, -⟩
    simp only [RunFinder.finder, hlog, hfm, findColl, reduceCtorEq] at hs
  · rfl

variable {Col T Col' T' : Type} [DecidableEq Col] [DecidableEq Col']

/-- **`Compiled.EO` over the outer Flock gives no witnesses where the run stages no statement**, at a statement with a
session whose outer parser refuses the empty statement: witnesses are one per session, the first paired with the first
session's staged statement, the empty one, and accepted by the outer check there (`Compiled.eo_some_staged`), which
refuses every witness at a statement that doesn't parse. -/
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
    rw [hP] <;> rfl
  exact absurd (hc.symm.trans h2) Bool.false_ne_true

/-- **`Comp.err` at the element-level Flock where nothing is found**: where `B` finds nothing, no rerun's witnesses
fall outside `R_O` and no run finder finds anything in the composite's reruns, `Comp.err` at the loop bounds
`(·, M, R)` is the composite's `κ` and the tails: `FC`'s and the link's at `M`, `F0`'s at `R`. -/
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

/-- **`KSCompiledOn` fails at the verifier that accepts everything** at any statement of `G` where the compiled verifier
takes a round, the statement has a V\* session, the outer Flock's parser refuses the empty statement and `κ < 1/2` at
the deployed decoders, with the server's `IVStar` the bound record's (`IVStarOfBound`): `Pr[accept ∧ ¬extract] = 1`
for the silent prover, every finder term is `0`, and at `M = ⌈4·(4W + β)⌉` and `R = 16·(k_I + 1)` each tail is below
`1/4`. -/
theorem not_ksCompiledOn_acceptAll (H : Msg → Msg) (D : RecAt Coin) (Bd : Bound Coin Col T)
    (DO : Claims.Flock.Deploy Coin Col' T') (Cp : Caps Coin) (G : Msg → Prop) (x : Msg) (hx : G x)
    (hIV : IVStarOfBound Bd) (hk : 0 < (D.record Bd.IVStar).V.k x) (hS : 0 < (D.record Bd.IVStar).nS x)
    (hP : DO.parse [] = none)
    (hκ : Claims.Recursion.κ (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy Bd.deploy.decode) x < 1 / 2) :
    ¬ KSCompiledOn H D.acceptAll Bd DO Cp G := by
  intro h
  have hk' : 0 < (compiledAt D.acceptAll Bd DO Cp).V.k x := by
    show 0 < (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)).V.k x
    rw [hIV]
    exact hk
  have hS' : 0 < (compiledAt D.acceptAll Bd DO Cp).nS x := by
    show 0 < (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)).nS x
    rw [hIV]
    exact hS
  have hκK : (compiledAt D.acceptAll Bd DO Cp).toComp.κ (Proto.ofFlock Bd Bd.deploy.decode Cp) x < 1 / 2 := by
    show Claims.Recursion.κ (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans))
      (Inner.ofFlock Bd.deploy Bd.deploy.decode) x < 1 / 2
    rw [hIV]
    exact hκ
  have hEO := Compiled.EO_none_of_stmts_nil DO (compiledAt D.acceptAll Bd DO Cp) H Claims.Witness.silentP x
    (fun _ => rfl) hS' hP
  have hW : ∀ ω u, (compiledAt D.acceptAll Bd DO Cp).toComp.wits H Claims.Witness.silentP x ω u = none :=
    fun ω u => hEO ω u.1 u.2
  have hBad : PrBad H (compiledAt D.acceptAll Bd DO Cp).V
      ((compiledAt D.acceptAll Bd DO Cp).toComp.R (Proto.ofFlock Bd Bd.deploy.decode Cp))
      ((compiledAt D.acceptAll Bd DO Cp).toComp.E (Proto.ofFlock Bd Bd.deploy.decode Cp) (Server.ofBound Bd)) x
      Claims.Witness.silentP = 1 :=
    prBad_acc H _ _ _ x _ (fun _ _ => rfl) hk' fun ω r₀ c =>
      Comp.E_none_of_EO_none _ _ _ H _ x ω r₀ c fun d c' => hEO ω d c'
  have hB : PrB H (compiledAt D.acceptAll Bd DO Cp).toComp.V Claims.Witness.silentP x = 0 :=
    prB_acc H _ _ x fun _ _ => rfl
  have hPO : (compiledAt D.acceptAll Bd DO Cp).toComp.PrOut H Claims.Witness.silentP x = 0 := by
    unfold Comp.PrOut
    refine (Pr_congr fun t => iff_false_intro ?_).trans Pr_false
    rintro ⟨w, hw, -⟩
    cases (hW t.1 t.2).symm.trans hw
  have hrer : ∀ ω u,
      (compiledAt D.acceptAll Bd DO Cp).toComp.rerunCalls H Claims.Witness.silentP x ω u = [] := by
    intro ω u
    unfold Comp.rerunCalls
    rw [hW] <;> rfl
  have hown : ∀ ω u, (compiledAt D.acceptAll Bd DO Cp).toComp.ownAt H Claims.Witness.silentP x ω u = [] :=
    fun _ _ => rfl
  have hIn : ∀ F : RunFinder Coin, F.PrAt H ((compiledAt D.acceptAll Bd DO Cp).toComp.innerModel
      ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans) H Claims.Witness.silentP x)
      ((compiledAt D.acceptAll Bd DO Cp).toComp.xI x) = 0 :=
    fun F => RunFinder.PrAt_log_nil H F _ _ hrer
  have hVer : ∀ F : RunFinder Coin, F.PrAt H ((compiledAt D.acceptAll Bd DO Cp).toComp.verModel
      ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans) H Claims.Witness.silentP x)
      ((compiledAt D.acceptAll Bd DO Cp).toComp.xI x) = 0 :=
    fun F => RunFinder.PrAt_log_nil H F _ _ fun ω u => by
      show (compiledAt D.acceptAll Bd DO Cp).toComp.rerunCalls H Claims.Witness.silentP x ω u ++
        (compiledAt D.acceptAll Bd DO Cp).toComp.ownAt H Claims.Witness.silentP x ω u = []
      rw [hrer, hown, List.append_nil]
  have hE := Comp.err_ofFlock_quiet (compiledAt D.acceptAll Bd DO Cp).toComp Bd Bd.deploy.decode Cp
    (Server.ofBound Bd) H Claims.Witness.silentP x
    ⌈4 * (4 * (Cp.width ((compiledAt D.acceptAll Bd DO Cp).toComp.xI x) : ℝ) +
      (compiledAt D.acceptAll Bd DO Cp).toComp.β (Proto.ofFlock Bd Bd.deploy.decode Cp) x)⌉₊
    (16 * (Bd.DE.k ((compiledAt D.acceptAll Bd DO Cp).toComp.xI x) + 1)) hB hPO hIn hVer
  have h1 := hBad.symm.trans_le ((h x hx Unit Claims.Witness.silentP _ _).trans_eq hE)
  have hT := Claims.Witness.Recursion.tail_lt_quarter
    (4 * (Cp.width ((compiledAt D.acceptAll Bd DO Cp).toComp.xI x) : ℝ) +
      (compiledAt D.acceptAll Bd DO Cp).toComp.β (Proto.ofFlock Bd Bd.deploy.decode Cp) x)
  have hF := Claims.Witness.Recursion.f0_tail_lt (Bd.DE.k ((compiledAt D.acceptAll Bd DO Cp).toComp.xI x))
  linarith

end Claims.RecGeneric

namespace Flock.Guarantees

open Claims (Msg Out)
open FlockDeploy (Coin FinalP)

/-- **`RecursiveKSCompiledEvery`'s nontrivial lemma**: `RecursiveKSCompiledEvery`'s conclusion,
`RecursiveKSCompiledAtRec` at the deployed values, is false at the verifier that accepts everything
(`RecAt.acceptAll` of the deployment's compiled record), at every brings, from the named assumption
`RecDeploy.KSFixture` (`recursiveKSEvery_nontrivial`'s): some bytes the deployed readers read as a dense compiled
statement with a V\* session whose inner statement is in the plain scope, with a coin-server record at whose transcript
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
