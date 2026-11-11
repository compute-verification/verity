module

public import Proofs.Flock.Claims.Witnesses.SamplingGeneric
public import Proofs.Flock.Claims.CompleteFails
public import Proofs.Flock.Claims.Deploy

/-!
# The sampled compiler's completeness says something

`CompilerComplete` (`Definitions.Claims.SamplingGeneric`) is a guarantee that the lock's `claims.generic` names, held to
its honest and nontrivial witnesses (the lock's `witnesses`), and `Compiled.Complete`, what it concludes, is a claim held
to its `fails` and `meets` witnesses (the lock's `claims.witnesses`). Each builds its instance on the toys of
`Witnesses.SamplingGeneric`.

* `twoL` over `twoA` meets the completeness contract at `K = 2`, with two sampled sessions: each is the silent prover's
  run of `sesV` on a statement `twoA` serves, the run accepts when both accept, and `twoA`'s silent prover is complete at
  error `0`. The witness `twoWit` opens both inputs as `true`, so no proof block is faulty: `twoComplete_meets`, and at
  SHA-512 `compilerComplete_honest`.
* `rejL`: a compiled verifier over `twoA` that rejects every run, on `twoL`'s statement, partition and commitment, at
  error `0`, so `twoWit` is in its honest relation and it is not complete: `compilerComplete_nontrivial`,
  `Compiled.Complete_fails`.
-/

@[expose] public section

namespace Claims.Witness.SamplingGeneric

open Claims.SamplingGeneric
open Claims.Witness (silentP)
open Claims.CompleteFails (rejV not_complete_rejV zeroHash out64_zeroHash)

/-- The witness that opens both of `twoStmt`'s inputs as `true`. -/
def twoWit : Claims.Sampling.Wit := ⟨[⟨0, true, []⟩, ⟨1, true, []⟩]⟩

theorem twoWit_val (H : Msg → Msg) {v : ℕ} (hv : v < 2) : twoWit.val H trueCm twoStmt v = some true := by
  interval_cases v <;> simp [Claims.Sampling.Wit.val, twoWit, Claims.Sampling.CommitmentScheme.Opens, trueCm, OC.eval]

/-- Under `trueCm`, `twoWit` leaves no proof block of `twoPart` faulty. -/
theorem two_rhon (H : Msg → Msg) (x : Msg) :
    Claims.Sampling.Rhon H trueCm (fun _ => some twoStmt) (fun _ => some twoPart) x twoWit := by
  refine ⟨twoStmt, twoPart, rfl, rfl, Finset.eq_empty_of_forall_notMem fun v hv => ?_⟩
  simp only [Claims.Sampling.faulty, Finset.mem_filter, Finset.mem_range] at hv
  obtain ⟨hv2, hf⟩ := hv
  replace hv2 : v < 2 := by simpa [twoPart_blocks] using hv2
  unfold Claims.Sampling.Faulty at hf
  refine hf ⟨[v], ?_, fun _ => true,
    two_correct hv2 (twoWit.val H trueCm twoStmt) (fun _ => true) (twoWit_val H hv2)⟩
  rw [twoPart_blocks]
  interval_cases v <;> rfl

/-- The compiled verifier over `twoA` that rejects every run, on `twoL`'s statement, partition and commitment, with no
sessions, at error `0`. -/
def rejL (Coin : Type) : Compiled Coin (twoA Coin) where
  V := rejV Coin
  parse _ := some twoStmt
  scopeCheck _ := true
  partition _ := some twoPart
  commit := trueCm
  stage2 _ := ∅
  checked _ _ := []
  sh _ _ := ()
  κb _ _ := 0
  γb _ _ := 0
  rounds _ _ := ∅
  session _ _ _ := none
  iroots _ _ _ := []
  reveal _ _ _ _ := []
  openCalls _ := 0
  nOut _ := 2
  nIn _ := 2
  Tape := Unit
  P _ := silentP
  δc _ _ := 0

section Complete

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- `twoA`'s silent prover is complete at error `0`: `sesV` accepts its one empty message. -/
theorem twoA_complete (H : Msg → Msg) :
    Claims.Complete H (twoA Coin).V (fun _ => silentP) (twoA Coin).R (fun _ => True) (fun _ => 0) := by
  intro _ x _ _ _ ω
  have hacc : ∀ r : Fin ((twoA Coin).V.k x) → Coin, (run H silentP (twoA Coin).V x ω r).out = .acc := by
    intro r
    have h := (twoA_run H silentP [] () 0 x r).1
    have h0 : twoMsg H silentP [] () 0 = [] := rfl
    rw [h0] at h
    change (run H (twoP H silentP [] () 0) (twoA Coin).V x () r).seen.out = .acc
    rw [h]
    simp [sesOut]
  calc 1 - (0 : ℝ) ≤ 1 := by norm_num
    _ = Pr fun r : Fin ((twoA Coin).V.k x) → Coin => (run H silentP (twoA Coin).V x ω r).out = .acc :=
      ((Pr_congr fun r => iff_true_intro (hacc r)).trans Claims.ZKMal.Pr_true).symm

omit [Fintype Coin] [Nonempty Coin] in
/-- `twoL` meets the completeness contract over `twoA`'s silent prover at `K = 2`: block `u`'s session is the silent
prover's run of `sesV` on its own round, the run accepts when both sessions do, each sampled block is at its own slot,
and the errors are `0`. -/
theorem twoL_completeSessions (H : Msg → Msg) :
    (twoL Coin).CompleteSessions H (fun _ => silentP) (fun _ => True) (fun _ => 0) (fun _ => 2) where
  honest := by
    intro x w _ _ ω
    change Unit at ω
    cases ω
    intro r u hu
    have hu2 : u < 2 := by
      change u ∈ [0, 1] at hu
      simp at hu
      omega
    refine ⟨[], (), (), trivial, trivial, ?_, fun r' _ => ?_⟩
    · show 1 = (if h : u < 2 then ({⟨u, h⟩} : Finset (Fin 2)) else ∅).card
      simp [hu2]
    · have h1 : (run H silentP (twoA Coin).V [] () ((twoL Coin).coinsOf x u [] r')).seen =
          ⟨[(twoMsg H silentP x () u, sesOut [twoMsg H silentP x () u])], sesOut [twoMsg H silentP x () u]⟩ :=
        (twoA_run H silentP x () u [] _).1
      change twoSession u (run H silentP (twoL Coin).V x () r').seen =
        some ([], (run H silentP (twoA Coin).V [] () ((twoL Coin).coinsOf x u [] r')).seen)
      rw [twoL_run, h1]
      interval_cases u <;> rfl
  accepts := by
    intro x w _ _ ω
    change Unit at ω
    cases ω
    intro r _
    have h0 : twoMsg H silentP x () 0 = [] := rfl
    have h1 : twoMsg H silentP x () 1 = [] := rfl
    change (run H silentP (twoL Coin).V x () r).seen.out = .acc
    rw [twoL_run, h0, h1]
    simp [twoOut]
  among := by
    intro x w _ _ ω r u hu
    have hu2 : u < 2 := by
      change u ∈ [0, 1] at hu
      simp at hu
      omega
    refine ⟨u, hu2, u, ?_, rfl⟩
    change ([0, 1] : List ℕ)[u]? = some u
    interval_cases u <;> rfl
  err x := ⟨le_rfl, fun _ _ _ _ _ _ _ _ _ => by
    change ((2 : ℕ) : ℝ) * min 1 (0 : ℝ) ≤ 0
    norm_num [min_def]⟩

/-- **Two sampled sessions meet the completeness contract** (`twoL`, at every hash): `twoA`'s silent prover is complete,
`twoL` meets the contract and the completeness contract over it, the empty statement is in scope, and `twoWit` leaves
no proof block faulty. So `CompilerComplete` reads antecedents that hold together with more than one sampled session, at
a witness in the honest relation. -/
theorem twoComplete_meets (H : Msg → Msg) :
    Claims.Complete H (twoA Coin).V (fun _ => silentP) (twoA Coin).R (fun _ => True) (fun _ => 0) ∧
      (twoL Coin).Sessions H ∧
      (twoL Coin).CompleteSessions H (fun _ => silentP) (fun _ => True) (fun _ => 0) (fun _ => 2) ∧
      (twoL Coin).Scope [] ∧
      Claims.Sampling.Rhon H (twoL Coin).commit (twoL Coin).parse (twoL Coin).partition [] twoWit :=
  ⟨twoA_complete H, twoL_sessions H, twoL_completeSessions H, twoL_scope [], two_rhon H []⟩

/-- **`rejL` is not complete**: `twoWit` is in its honest relation on the empty statement, in scope, and every run is
rejected, at error `0`. -/
theorem rejL_not_complete : ¬ (rejL Coin).Complete zeroHash := fun h =>
  not_complete_rejV (Coin := Coin) out64_zeroHash (rejL Coin).P
    (R := Claims.Sampling.Rhon zeroHash (rejL Coin).commit (rejL Coin).parse (rejL Coin).partition)
    (Scope := (rejL Coin).Scope) (δc := (rejL Coin).δc zeroHash) (x := []) (w := twoWit)
    ⟨twoStmt, twoPart, rfl, rfl, inScope_two⟩ (two_rhon zeroHash []) (show (0 : ℝ) < 1 by norm_num) h

end Complete

end Claims.Witness.SamplingGeneric

/-! ## Stated over the spec -/

namespace Claims.SamplingGeneric

open Claims.Witness (silentP)
open Claims.Witness.SamplingGeneric

/-- **`CompilerComplete`'s premises hold together at two sessions**: `twoA`'s silent prover is complete, `twoL` meets
the contract and the completeness contract over it, the empty statement is in scope, and `twoWit` is in its honest
relation. -/
theorem compilerComplete_honest :
    Claims.Complete FlockDeploy.sha512 (twoA Bool).V (fun _ => silentP) (twoA Bool).R (fun _ => True) (fun _ => 0) ∧
      (twoL Bool).Sessions FlockDeploy.sha512 ∧
      (twoL Bool).CompleteSessions FlockDeploy.sha512 (fun _ => silentP) (fun _ => True) (fun _ => 0) (fun _ => 2) ∧
      (twoL Bool).Scope [] ∧
      Claims.Sampling.Rhon FlockDeploy.sha512 (twoL Bool).commit (twoL Bool).parse (twoL Bool).partition [] twoWit :=
  twoComplete_meets _

/-- **`CompilerComplete`'s conclusion fails** at `rejL`, which rejects every run. -/
theorem compilerComplete_nontrivial : ¬ (rejL Bool).Complete Claims.CompleteFails.zeroHash := rejL_not_complete

/-- **`Compiled.Complete` fails** at some compiled verifier. -/
theorem Compiled.Complete_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] :
    ∃ (H : Msg → Msg) (A : Arg Coin) (L : Compiled Coin A), ¬ L.Complete H :=
  ⟨_, _, _, rejL_not_complete⟩

end Claims.SamplingGeneric
