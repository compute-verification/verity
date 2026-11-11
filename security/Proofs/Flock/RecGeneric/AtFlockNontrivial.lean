module

public import Proofs.Flock.RecGeneric.AtFlock
public import Proofs.Flock.Recursive.KSNontrivial

/-!
# `RecursiveKSEvery` at the verifier that accepts everything (the Lean check's `nontrivial`)

`RecursiveKSEvery`'s nontrivial lemma, as `recursiveKS_nontrivial` is `RecursiveKS`'s: its conclusion,
`RecursiveKSEveryAtRec` at the deployed values, is false at the compiled record whose verifier keeps its rounds and
accepts at its first round with no SHA-512 call (`RecAt.acceptAll`), from the one named assumption
`RecDeploy.KSFixture`. The silent prover is accepted on every run; the accepting record stages no V\* statement, which
the outer Flock's parser refuses, so `E_O` gives no witnesses (`EO_none_of_vstar_nil`) and the composite's `E*` extracts
nothing (`Comp.E_none_of_EO_none`): `Pr[accept ∧ ¬extract] = 1`. The record calls `H` nowhere, so `B`, `B_I`, `FCI` and
`F0I` find nothing, and the gated link, which searches only once a rerun gives witnesses, finds nothing
(`linkPlan_PrAt_noAcc`); the tails are below `1/4` each at a large enough `M` and `R`, and `λ₂ ≥ 100` at the deployed
decoders gives `κ < 1/2`.
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

omit [Inhabited Coin] in
/-- **The gated link finds nothing where no run is accepted**: its plan's first question is answered no, so it searches
no run. -/
theorem linkPlan_PrAt_noAcc {Ω C : Type} [Fintype Ω] [Nonempty Ω] [Fintype C] [Nonempty C] {n : ℕ}
    (H : Msg → Msg) (k : Msg → ℕ) (M : ℕ) (m : RunModel Coin Ω n C) (x : Msg) (hacc : ∀ ω u, ¬ m.acc ω u) :
    (linkPlan (Coin := Coin) k M).PrAt H m x = 0 := by
  unfold RunFinder.PrAt
  split_ifs with h
  · unfold Finder.Pr
    refine (Pr_congr fun s => iff_false_intro ?_).trans Pr_false
    rintro ⟨u, v, hs, -⟩
    simp [RunFinder.finder, linkPlan, Plan.exec, hacc, findColl] at hs
  · rfl

/-- **`E*` extracts nothing where `E_O` gives no witnesses**: the judged sample's inner view is then rejected
(`decProg`), so the inner `E^rep` stops. -/
theorem Comp.E_none_of_EO_none {W : Type} (K : Comp Coin) (I : Proto Coin W) (S : Server) (H : Msg → Msg) {Ω : Type}
    (P : Prover Ω) (x : Msg) (ω : Ω) (r₀ : Fin (K.V.k x) → Coin) (c : (K.E I S).C x)
    (hO : ∀ d c', ((K.EO.ext x d c').eval H (run H P K.V x ω)).1 = none) :
    ((K.E I S).eval H P x ω r₀ c).1 = none := by
  unfold Extractor.eval
  simp only [Comp.E, erep, serve, Comp.stage, eval_bind_fst, hO, Comp.decProg, RP.eval, reduceCtorEq, ↓reduceIte]

variable {Col T Col' T' : Type} [DecidableEq Col] [DecidableEq Col']

omit [DecidableEq Col] in
/-- **`KSLinkOn` fails at the verifier that accepts everything** at any statement of `G` where the compiled verifier
takes a round, the statement has a V\* session, the outer Flock's parser refuses the empty statement and `κ < 1/2` at
the deployed decoders: `Pr[accept ∧ ¬extract] = 1` for the silent prover, every finder term is `0`, and at
`M = ⌈4·(4W + β₂)⌉` and `R = 16·(k_I + 1)` each tail is below `1/4`. -/
theorem not_ksLinkOn_acceptAll [Zero Col] [Zero Col'] (H : Msg → Msg) (D : RecAt Coin) (Bd : Bound Coin Col T)
    (DO : Claims.Flock.Deploy Coin Col' T') (Cp : Caps Coin) (G : Msg → Prop) (x : Msg) (hx : G x)
    (hk : 0 < (D.record Bd.IVStar).V.k x) (hS : 0 < (D.record Bd.IVStar).nS x) (hP : DO.parse [] = none)
    (hκ : Claims.Recursion.κ (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy Bd.deploy.decode) x < 1 / 2) :
    ¬ KSLinkOn H D.acceptAll Bd DO Cp G := by
  intro h
  have hc : ∀ y (r : Fin ((D.acceptAll.record Bd.IVStar).V.k y) → Coin) ws,
      (D.acceptAll.record Bd.IVStar).callsAt H y r ws = [] := fun _ _ _ => rfl
  have hs : ∀ y ps, (D.acceptAll.record Bd.IVStar).vstar y ps = [] := fun _ _ => rfl
  have hv : ∀ as cs, (D.acceptAll.record Bd.IVStar).V.next x as cs = .ret .acc := fun _ _ => rfl
  have hk' : 0 < (D.acceptAll.record Bd.IVStar).V.k x := hk
  have hS' : 0 < (D.acceptAll.record Bd.IVStar).nS x := hS
  have hκ' : Claims.Recursion.κ (D.acceptAll.record Bd.IVStar) (Inner.ofFlock Bd.deploy Bd.deploy.decode) x < 1 / 2 := hκ
  have hO : ∀ (ω : Unit) d c, (((compDep D.acceptAll Bd DO).EO.ext x d c).eval H
      (run H Claims.Witness.silentP (compDep D.acceptAll Bd DO).V x ω)).1 = none := fun _ d c =>
    EO_none_of_vstar_nil H _ (Inner.ofFlock Bd.deploy Bd.deploy.decode) DO DO.decode x (hs x) hS' hP _ d c
  have hL : ∀ M : ℕ, (linkPlan (Coin := Coin) Bd.DE.V.k M).PrAt H
      ((compDep D.acceptAll Bd DO).verModel ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans) H Claims.Witness.silentP x)
      ((compDep D.acceptAll Bd DO).xI x) = 0 := fun M =>
    linkPlan_PrAt_noAcc H _ M _ _ fun ω u hu => by
      simp [Comp.verModel, Comp.wits, Extractor.eval, hO] at hu
  have h1 := h x hx Unit Claims.Witness.silentP
    ⌈4 * (4 * (Cp.width (Claims.Recursion.xI (D.acceptAll.record Bd.IVStar) x) : ℝ) +
      βdev2 (D.acceptAll.record Bd.IVStar) (Inner.ofFlock Bd.deploy Bd.deploy.decode) Cp x)⌉₊
    (16 * ((Inner.ofFlock Bd.deploy Bd.deploy.decode).k (Claims.Recursion.xI (D.acceptAll.record Bd.IVStar) x) + 1))
  have hBad : PrBad H (compDep D.acceptAll Bd DO).V
      ((compDep D.acceptAll Bd DO).R (Proto.ofFlock Bd Bd.deploy.decode Cp))
      ((compDep D.acceptAll Bd DO).E (Proto.ofFlock Bd Bd.deploy.decode Cp) (Server.ofBound Bd)) x
      Claims.Witness.silentP = 1 :=
    prBad_acc H _ _ _ x _ hv hk' fun ω r₀ c => Comp.E_none_of_EO_none _ _ _ H _ x ω r₀ c (hO ω)
  have hB : PrB H (D.acceptAll.record Bd.IVStar).V Claims.Witness.silentP x = 0 := prB_acc H _ _ x hv
  rw [hBad, hB, Claims.Witness.Recursion.prBinnerV_noCalls H (D.acceptAll.record Bd.IVStar) hc hs,
    prFCI_noCalls H (D.acceptAll.record Bd.IVStar) hc,
    Claims.Witness.Recursion.prF0I_noCalls H (D.acceptAll.record Bd.IVStar) hc, hL] at h1
  simp only [mul_zero, Real.sqrt_zero, zero_add, add_zero] at h1
  have hT := Claims.Witness.Recursion.tail_lt_quarter
    (4 * (Cp.width (Claims.Recursion.xI (D.acceptAll.record Bd.IVStar) x) : ℝ) +
      βdev2 (D.acceptAll.record Bd.IVStar) (Inner.ofFlock Bd.deploy Bd.deploy.decode) Cp x)
  have hF := Claims.Witness.Recursion.f0_tail_lt
    ((Inner.ofFlock Bd.deploy Bd.deploy.decode).k (Claims.Recursion.xI (D.acceptAll.record Bd.IVStar) x))
  linarith

end Claims.RecGeneric

namespace Flock.Guarantees

open Claims (Msg Out)
open FlockDeploy (Coin FinalP)

/-- **`RecursiveKSEvery`'s nontrivial lemma**: `RecursiveKSEvery`'s conclusion, `RecursiveKSEveryAtRec` at the deployed
values, is false at the verifier that accepts everything (`RecAt.acceptAll` of the deployment's compiled record), at
every brings, from the named assumption `RecDeploy.KSFixture` (`recursiveKS_nontrivial`'s): some bytes the deployed
readers read as a dense compiled statement with a V\* session whose inner statement is in the plain scope, with a
live-coin record at whose transcript its V\* parts are in scope. The silent prover is accepted on every run and `E*`
extracts nothing, so `Pr[accept ∧ ¬extract] = 1`; every finder term, the gated link's included, is `0`; the tails are
below `1/4` each; and `λ₂ ≥ 100` at the deployed decoders gives `κ < 1/2`. -/
theorem recursiveKSEvery_nontrivial (h : RecDeploy.KSFixture) :
    ∃ (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ) (hd : Msg → Flock.RecRecord.Record)
      (recRest : Msg → List (Msg × Out) → List UInt8), ∀ b : RecDeploy.Brings,
      ¬ RecursiveKSEveryAtRec
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
  exact Claims.RecGeneric.not_ksLinkOn_acceptAll FlockDeploy.sha512 D Bd
    (RecDeploy.deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep (fun _ => True) x trivial
    (RecDeploy.recBound_k_pos _ _ _ _ _ _ _ _ _ _ x) hP.2.1 (RecDeploy.zkRec_parse_nil _)
    (Claims.Recursion.κ_lt_half_of_lamLin2
      (RecDeploy.lamLin2_deployed_ge_dep _root_.Flock.Readable.relation
        { RecDeploy.deployedB b with tr := RecDeploy.recTr aux nOf hd recRest }
        (RecDeploy.DeployedR.layR RecDeploy.OpenRows.elemR) (FlockDeploy.nextR RecDeploy.DeployedR.rdR)
        (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation RecDeploy.DeployedR.rdR).cols
        RecDeploy.DeployedR.wER FlockDeploy.joinR RecDeploy.rowInDeployed RecDeploy.DeployedR.finalR
        RecDeploy.DeployedR.wR Bd.deploy.decode hin hO' hn hS (n := 0) (by norm_num) (v := 0) (by norm_num)))

end Flock.Guarantees
