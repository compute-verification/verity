module

public import Proofs.Flock.RecGeneric.ToyProof
public import Proofs.Flock.RecGeneric.OuterProof
public import Proofs.Flock.RecGeneric.AtFlockCompiledNontrivial
public import Proofs.Flock.Claims.Witnesses.Shared
public import Proofs.Flock.Claims.Deploy

/-! `CompositionKS`'s and `CompilerKS`'s witnesses at coins `Bool` and SHA-512: honest at `twoRounds`, `eqServer` and `Compiled.echo`, nontrivial at `noWitness` and `silentL`. -/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion
open Claims.SamplingGeneric (Arg)

/-- A coin's bytes: `[1]` for `true`, `[]` for `false`. -/
def ansBit (b : Bool) : Msg := if b then [1] else []

/-- **Π_in's verifier**: two rounds; it answers the first with its coin's bytes and accepts at the second. -/
def twoV : Verifier Bool :=
  ⟨fun _ => 2, fun _ as cs => .ret (if as.length < 2 then .msg (ansBit (cs.getLastD default)) else .acc)⟩

/-- **Π_in with two rounds**: `twoV`, a relation that holds at every statement, and `E^rep` at one wait that needs no
accepted rerun, which gives `()` wherever the judged run accepted; no error terms. -/
noncomputable def twoRounds : Proto Bool Unit where
  V := twoV
  R _ _ := True
  Scope _ := True
  Op := Unit
  opens _ := []
  assemble _ _ := some ()
  check _ _ := true
  waits _ := 1
  k _ := 0
  cap _ := 0
  κ _ := 0
  γ _ := 0
  ans := ansBit
  nT := 0
  terms i := i.elim0

/-- **A server whose commit string is its row**: one row per earlier round, under the one salt `()`; a commit string
opens to a row exactly where it is that row. -/
def eqServer : Server where
  Salt := Unit
  width := 0
  nAt _ _ := 1
  join rows := rows.headD []
  final m := some ([[([], ())]], m)
  commit row _ := .ret row
  opens a row _ := .ret (decide (a = row))

theorem twoRounds_wf : twoRounds.WF :=
  ⟨⟨fun m => !m.isEmpty, fun c => by cases c <;> rfl⟩, fun _ => le_rfl, fun i => Fin.elim0 i, fun i => Fin.elim0 i,
    fun i => Fin.elim0 i⟩

theorem erep_one_acc {D W Op : Type} (H : Msg → Msg) (run : D → Transcript) (opens : Seen → List Op) (w : W) (d₀ : D)
    (ρ : Fin 1 → Fin 0 → D) (h : (run d₀).out = .acc) :
    ((erep 1 0 0 opens (fun _ => some w) (fun _ => true) d₀ ρ).eval H run).1 = some w := by
  have hl : ((List.finRange 1).map fun i => (List.finRange 0).map (ρ i)) = [[]] := by rfl
  have hs : (run d₀).seen.out = .acc := h
  show (RP.eval H run (if (run d₀).seen.out = .acc then
    firstWait 0 opens (fun _ => some w) (fun _ => true) ((List.finRange 1).map fun i => (List.finRange 0).map (ρ i))
    else .ret none)).1 = some w
  rw [if_pos hs, hl]
  rfl

theorem twoRounds_ksAbortLin (H : Msg → Msg) : twoRounds.KSAbortLin H := by
  intro x Ω _ _ P A _ _ keep m
  have hb : twoRounds.toInner.PrBadAbort H x P keep = 0 := by
    unfold Inner.PrBadAbort
    refine (Pr_congr fun s => iff_false_intro ?_).trans Pr_false
    rintro ⟨hacc, hR⟩
    exact hR ((congrArg (Rel.ext twoRounds.R x)
      (erep_one_acc H (abortRun H P twoRounds.V x keep s.1) twoRounds.opens () s.2.1 s.2.2 hacc)).mpr trivial)
  exact (le_of_eq hb).trans (add_nonneg le_rfl (Finset.sum_nonneg fun i _ => Fin.elim0 i))

theorem twoRounds_lastCoinFree (H : Msg → Msg) : twoRounds.LastCoinFree H := by
  intro x as cs cs' ha _ _ _
  have h : ¬ as.length < 2 := by rw [ha]; exact lt_irrefl _
  have e : ∀ c : List Bool, twoRounds.V.next x as c = OC.ret Out.acc := fun c => by
    show OC.ret (if as.length < 2 then Out.msg (ansBit (c.getLastD default)) else Out.acc) = _
    rw [if_neg h]
  rw [e cs, e cs']

theorem eqServer_binds : eqServer.Binds := by
  intro H a row row' y y' hne h1 h2
  have e1 : decide (a = row) = true := h1
  have e2 : decide (a = row') = true := h2
  exact absurd ((of_decide_eq_true e1).symm.trans (of_decide_eq_true e2)) hne

theorem echo_accAt : AccAt FlockDeploy.sha512 (compile Compiled.echo twoRounds eqServer).V [] :=
  ⟨Unit, Claims.Witness.silentP, (), fun _ => false, by decide⟩

/-- **`CompilerKS`'s premises hold together at two sessions**: `twoRounds` is well formed, knowledge sound and reads no
last coin, `eqServer` binds, `echoArg` is knowledge sound, and the echo compiler at `Open(twoRounds, eqServer)` meets
Π_out's sessions' contract and the composition's at the empty statement, where it runs two sessions and accepts the
silent prover. -/
theorem compilerKS_honest :
    twoRounds.WF ∧ twoRounds.KSAbortLin FlockDeploy.sha512 ∧ twoRounds.LastCoinFree FlockDeploy.sha512 ∧
      eqServer.Binds ∧ (echoArg Bool).KS FlockDeploy.sha512 ∧
      (compile Compiled.echo twoRounds eqServer).Sessions FlockDeploy.sha512 ∧
      (compile Compiled.echo twoRounds eqServer).toComp.Contract FlockDeploy.sha512
        (eqServer.IVStar twoRounds.V twoRounds.ans) (fun v => twoRounds.opens (eqServer.openView v)) [] ∧
      (compile Compiled.echo twoRounds eqServer).nS [] = 2 ∧
      AccAt FlockDeploy.sha512 (compile Compiled.echo twoRounds eqServer).V [] :=
  ⟨twoRounds_wf, twoRounds_ksAbortLin _, twoRounds_lastCoinFree _, eqServer_binds, echoArg_KS _,
    echo_sessions (eqServer.IVStar twoRounds.V twoRounds.ans) _,
    echo_contract _ (eqServer.IVStar twoRounds.V twoRounds.ans) _ [], rfl, echo_accAt⟩

/-- **`CompositionKS`'s premises hold together at two sessions**: as `compilerKS_honest`, with `E_O` the echo
compiler's, knowledge sound at the empty statement by O1 (`outerSessionsKS`) at nonnegative `κ_O` and `γ_O`. -/
theorem compositionKS_honest :
    twoRounds.WF ∧ twoRounds.KSAbortLin FlockDeploy.sha512 ∧ twoRounds.LastCoinFree FlockDeploy.sha512 ∧
      eqServer.Binds ∧ (compile Compiled.echo twoRounds eqServer).toComp.OuterKSAt FlockDeploy.sha512 [] ∧
      0 ≤ (compile Compiled.echo twoRounds eqServer).toComp.κO [] ∧
      0 ≤ (compile Compiled.echo twoRounds eqServer).toComp.γO [] ∧
      (compile Compiled.echo twoRounds eqServer).toComp.Contract FlockDeploy.sha512
        (eqServer.IVStar twoRounds.V twoRounds.ans) (fun v => twoRounds.opens (eqServer.openView v)) [] ∧
      (compile Compiled.echo twoRounds eqServer).nS [] = 2 ∧
      AccAt FlockDeploy.sha512 (compile Compiled.echo twoRounds eqServer).toComp.V [] := by
  have hS : (compile Compiled.echo twoRounds eqServer).Sessions FlockDeploy.sha512 :=
    echo_sessions (eqServer.IVStar twoRounds.V twoRounds.ans) _
  exact ⟨twoRounds_wf, twoRounds_ksAbortLin _, twoRounds_lastCoinFree _, eqServer_binds,
    fun Ω _ _ P => outerSessionsKS Bool FlockDeploy.sha512 (echoArg Bool) (compile Compiled.echo twoRounds eqServer)
      (echoArg_KS _) hS [] Ω P,
    Compiled.κO_nonneg hS [], Compiled.γO_nonneg _ [],
    echo_contract _ (eqServer.IVStar twoRounds.V twoRounds.ans) _ [], rfl, echo_accAt⟩

/-- **Π_in with an empty relation**: no rounds, and `E^rep` gives nothing. -/
noncomputable def noWitness : Proto Bool Unit where
  V := ⟨fun _ => 0, fun _ _ _ => .ret .rej⟩
  R _ _ := False
  Scope _ := True
  Op := Unit
  opens _ := []
  assemble _ _ := none
  check _ _ := true
  waits _ := 0
  k _ := 0
  cap _ := 0
  κ _ := 0
  γ _ := 0
  ans := ansBit
  nT := 0
  terms i := i.elim0

/-- **A compiled verifier that accepts after one round** (`Claims.Witness.accV`) and runs no session. -/
noncomputable def silentL : Compiled Bool (echoArg Bool) where
  V := Claims.Witness.accV Bool
  parses _ := True
  xI := id
  pre _ := 0
  nS _ := 0
  sh _ _ := ()
  κb _ _ := 0
  γb _ _ := 0
  stmts _ _ := []
  session _ _ v := v
  msgs _ _ := []
  calls _ _ _ _ := []
  own _ _ _ := []

/-- The compiler that gives `silentL` at every verifier. -/
noncomputable def silentC : Compiler Bool (echoArg Bool) := fun _ => silentL

/-- `silentL` at `Open(noWitness, eqServer)`, with its `E_O`. -/
noncomputable def silentK : Comp Bool := (compile silentC noWitness eqServer).toComp

theorem silentK_prBad (H : Msg → Msg) :
    PrBad H silentK.V (silentK.R noWitness) (silentK.E noWitness eqServer) [] Claims.Witness.silentP = 1 :=
  Claims.Witness.prBad_accV (Coin := Bool) H (silentK.R noWitness) (silentK.E noWitness eqServer) []
    Claims.Witness.silentP fun _ hw => hw.2

theorem silentK_err (H : Msg → Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (m : Fin noWitness.nT → ℕ)
    (M : ℕ) : silentK.err noWitness eqServer H P [] m M = 0 := by
  haveI : IsEmpty (Fin noWitness.nT) := ⟨fun i => Fin.elim0 i⟩
  have hκO : silentK.κO [] = 0 := by
    show ∑ _s : Fin 0, (0 : ℝ) = 0
    simp
  have hγO : silentK.γO [] = 0 := by
    show (∑ _s : Fin 0, _) ^ 2 = (0 : ℝ)
    simp
  have hκ : silentK.κ noWitness [] = 0 := by
    have h0 : noWitness.κ (silentK.xI []) = 0 := rfl
    unfold Comp.κ
    rw [hκO, mul_zero, add_zero, h0]
  have hγ : silentK.γ noWitness [] = 0 := by
    unfold Comp.γ
    rw [hγO, mul_zero]
  have hr : silentK.reads noWitness [] = 0 := by
    unfold Comp.reads
    simp
  have hβ : silentK.β noWitness [] = 0 := by
    have h0 : noWitness.V.k (silentK.xI []) = 0 := rfl
    unfold Comp.β
    rw [h0, Nat.cast_zero, mul_zero]
  have hlog : ∀ ω u, (silentK.verModel (eqServer.IVStar noWitness.V noWitness.ans) H P []).log ω u = [] := by
    intro ω u
    show silentK.rerunCalls H P [] ω u ++ silentK.ownAt H P [] ω u = []
    unfold Comp.rerunCalls
    cases silentK.wits H P [] ω u <;> rfl
  have hF : (forkPlan (Coin := Bool) noWitness.V.k).PrAt H
      (silentK.verModel (eqServer.IVStar noWitness.V noWitness.ans) H P []) (silentK.xI []) = 0 :=
    RunFinder.PrAt_log_nil H _ _ _ hlog
  unfold Comp.err
  rw [hκ, hγ, hr, hF, hβ]
  simp

/-- **`CompositionKS`'s conclusion fails** at `silentK`, which accepts the silent prover with an empty relation: the
bad event has probability `1`, the error `0`. -/
theorem compositionKS_nontrivial :
    ¬ PrBad FlockDeploy.sha512 silentK.V (silentK.R noWitness) (silentK.E noWitness eqServer) []
        Claims.Witness.silentP ≤
      silentK.err noWitness eqServer FlockDeploy.sha512 Claims.Witness.silentP [] (fun _ => 0) 0 := by
  rw [silentK_prBad, silentK_err]
  norm_num

/-- **`CompilerKS`'s conclusion fails** at the compiler `silentC` over `noWitness` and `eqServer`, whose composition
is `silentK`. -/
theorem compilerKS_nontrivial :
    ¬ PrBad FlockDeploy.sha512 (compile silentC noWitness eqServer).V
        ((compile silentC noWitness eqServer).toComp.R noWitness)
        ((compile silentC noWitness eqServer).toComp.E noWitness eqServer) [] Claims.Witness.silentP ≤
      (compile silentC noWitness eqServer).toComp.err noWitness eqServer FlockDeploy.sha512 Claims.Witness.silentP []
        (fun _ => 0) 0 :=
  compositionKS_nontrivial

end Claims.RecGeneric
