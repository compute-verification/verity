import Definitions.Claims.SamplingGeneric
import Proofs.Flock.Claims.Witnesses.Sampling

/-!
# Refutation witnesses: the generic sampled compiler's claims

Each claim of the generic sampled compiler (`Definitions.Claims.SamplingGeneric`) fails at a toy instance, on the toy
statement of `Witnesses.Sampling` (`toyStmt`, one input gate, returned, its partition `toyPart` one proof block `{0}`,
in scope).

* `toyA`: a proof system whose verifier accepts after one round and whose relation is empty, at errors `0`, so
  `Claims.KS` fails at its extractor (`Witness.KS_fails`): `Arg.KS_fails`.
* `sksL`: a compiled verifier over `toyA` that accepts after one round, samples nothing, has no sessions and whose
  commitment opens nothing (`rejCm`), so every proof block is incorrect under any witness, and none is in the relation
  at `δ = 0` or at the count `t = 1`. Its extractor and `B_conf` make no SHA-512 call of their own, so `B_ext` and
  `B_conf` find nothing, `B` finds nothing, and `κ < 1/3` at `δ = 0`: `Compiled.KS_fails`, `Compiled.KSCount_fails`.
  Its committed trace opens nothing, so its incorrect set is `{0}`, whose acceptance probability is `0`, while the
  committed form's error is `0`: `Compiled.KSReg_fails`, at `δ = 0`.

The lock's `meets` witnesses of these claims are Flock's decoder bounds at its deployed records
(`Claims.Flock.decoder_meets_plain`, `_zk`), the checkable antecedent `FlockKS` and `FlockKSReg` read at Flock.
-/

namespace Claims.Witness.SamplingGeneric

open Claims.SamplingGeneric
open Claims.Sampling (AuditStatement forkOn)
open Claims.Witness.Sampling (NoHash ownLog_noHash noHash_bind noHash_waitAccC noHash_flatMapRP rejCm toyProg toyStmt
  toyPart toyPart_blocks toy_boundary toy_Pa inScope_toy sksD sksD_noR sksD_noRCount sksD_outer sksD_missAt slack_pos
  slack_le_two)

/-! ## The toy instances -/

/-- The proof system whose verifier accepts after one round, whose relation is empty and whose extractor gives up, at
errors `0`. -/
noncomputable def toyA (Coin : Type) : Arg Coin where
  Sh := Unit
  shape _ := ()
  V := accV Coin
  rounds _ := 1
  W := Unit
  R _ _ := False
  EC _ := Unit
  ext _ _ _ := .ret none
  κ _ := 0
  γ _ := 0

/-- The compiled verifier over `toyA` that accepts after one round, samples nothing, has no sessions, and whose
commitment opens nothing. -/
noncomputable def sksL (Coin : Type) : Compiled Coin (toyA Coin) where
  V := accV Coin
  parse _ := some toyStmt
  scopeCheck _ := true
  partition _ := some toyPart
  commit := rejCm
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
  Tape := Unit
  P _ := silentP
  δc _ _ := 0

/-! ## A compiled verifier with no sessions makes no call of its own -/

section NoSessions

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- At a compiled verifier with no sessions, a run's extractions make no call of their own. -/
theorem noHash_runExt (L : Compiled Coin A) (hs : ∀ x u v, L.session x u v = none) (x : Msg) (s : AuditStatement)
    (r : Fin (L.V.k x) → Coin) (v : Seen) (c : L.SlotEC x) : NoHash (L.runExt x s r v c) := by
  unfold Compiled.runExt
  refine noHash_flatMapRP _ (fun iu => ?_) _
  split
  · simp only [Compiled.unitExt, hs]
    trivial
  · trivial

/-- At a compiled verifier with no sessions, the sampling extractor makes no call of its own: its waits rerun, and its
extractions return nothing. -/
theorem noHash_compiledE (L : Compiled Coin A) (hs : ∀ x u v, L.session x u v = none) (x : Msg)
    (r₀ : Fin (L.V.k x) → Coin) (c : L.E.C x) : NoHash (L.E.ext x r₀ c) := by
  intro v₀
  dsimp only [Compiled.E]
  split
  · trivial
  · split_ifs
    -- The waits' coin lists are generalized first: unifying against them unfolds `List.finRange` at the cap.
    · generalize List.map c.1 (List.finRange (L.cap x)) = rs
      refine noHash_bind _ _ (noHash_waitAccC _ _ _) fun outer => ?_
      generalize (List.finRange (L.k x)).zip outer = jos
      refine noHash_bind _ _ (noHash_flatMapRP _ (fun jo => ?_) _) fun _ => trivial
      generalize List.map (fun l => forkOn (L.stage2 x) jo.2.1 (c.2.1 jo.1 l)) (List.finRange (L.cap x)) = rs'
      refine noHash_bind _ _ (noHash_waitAccC _ _ _) fun inner => ?_
      generalize (List.finRange (L.k x)).zip inner = lis
      exact noHash_flatMapRP _ (fun _ => noHash_runExt L hs _ _ _ _ _) _
    · trivial

/-- At a compiled verifier with no sessions, `B_ext` finds nothing. -/
theorem bext_pr (H : Msg → Msg) (L : Compiled Coin A) (hs : ∀ x u v, L.session x u v = none) (hk : ∀ x, 0 < L.V.k x)
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg) :
    (Claims.Sampling.Bext H L.V L.E Ω P x).Pr H = 0 := by
  simp only [Claims.Sampling.Bext]
  rw [dite_eq_left (hk x)]
  unfold Finder.Pr
  rw [Pr_congr fun s => iff_false_intro (by
    simp [ownLog_noHash _ _ _ (noHash_compiledE L hs _ _ _), findColl])]
  exact Pr_false

/-- At a compiled verifier with no sessions, `B_conf` finds nothing. -/
theorem bconf_pr (H : Msg → Msg) (L : Compiled Coin A) (hs : ∀ x u v, L.session x u v = none) (Ω : Type)
    [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg) : (Compiled.Bconf H L Ω P x).Pr H = 0 := by
  rcases hx : L.parse x with _ | s
  · simp [Compiled.Bconf, hx, Finder.Pr, Pr_false]
  · simp only [Compiled.Bconf, hx]
    unfold Finder.Pr
    rw [Pr_congr fun q => iff_false_intro (by
      simp [ownLog_noHash _ _ _ (noHash_runExt L hs _ _ _ _ _), findColl])]
    exact Pr_false

end NoSessions

/-! ## `sksL`'s numbers -/

section Numbers

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- The proof block is correct only where its input's value is committed. -/
theorem toy_correct_val (val : ℕ → Option Bool) (ι : ℕ → Bool) (h : toyProg.Correct (toyProg.refIn [0] val ι) [0]) :
    ∃ b, val 0 = some b := by
  obtain ⟨b, hb, _⟩ := (h 0 (List.mem_singleton.2 rfl)).2 toy_boundary
  have hr : toyProg.refIn [0] val ι 0 = val 0 := by simp [Claims.Sampling.Prog.refIn, toy_boundary]
  exact ⟨b, hr.symm.trans hb⟩

theorem sksL_scope (x : Msg) : (sksL Coin).Scope x := ⟨toyStmt, toyPart, rfl, rfl, inScope_toy⟩

theorem sksL_κs (x : Msg) : (sksL Coin).κs x = 0 := by
  show ((List.range 1).map fun _ => (0 : ℝ)).foldr max 0 = 0
  simp

theorem sksL_κ (x : Msg) : (sksL Coin).κ 0 x < 1 / 3 := by
  have hs := slack_le_two
  have hs0 := slack_pos
  have ho := sksD_outer (Coin := Coin) x
  have hk : (sksL Coin).k x = (sksD Coin).k x := rfl
  have hc : (sksL Coin).cap x = (sksD Coin).cap x := rfl
  have ho0 : 0 ≤ 2 * ((sksD Coin).k x : ℝ) / ((sksD Coin).cap x : ℝ) + Real.exp (-((sksD Coin).k x : ℝ) / 8) := by
    positivity
  have he : (sksL Coin).κ 0 x = (1 + slack) * (2 * ((sksD Coin).k x : ℝ) / ((sksD Coin).cap x : ℝ) +
      Real.exp (-((sksD Coin).k x : ℝ) / 8)) := by
    unfold Compiled.κ
    rw [sksL_κs, hk, hc]
    ring
  rw [he]
  nlinarith

theorem sksL_missAt (x : Msg) : (sksL Coin).missAt 1 x = 0 :=
  (show (sksL Coin).missAt 1 x = Claims.Sampling.missAt (sksD Coin) 1 x from rfl).trans (sksD_missAt x)

theorem sksL_bext (H : Msg → Msg) (x : Msg) :
    (Claims.Sampling.Bext H (sksL Coin).V (sksL Coin).E Unit silentP x).Pr H = 0 :=
  bext_pr H (sksL Coin) (fun _ _ _ => rfl) (fun _ => Nat.one_pos) Unit silentP x

/-- `KSLinOn`'s bound fails at `sksL`'s silent prover, at a relation with no witness at `x` and a bound `κ' < 1`. -/
theorem sksL_bound_fails (H : Msg → Msg) (Rx : Rel Claims.Sampling.Wit) (κ' : ℝ) (x : Msg) (hR : ∀ w, ¬ Rx x w)
    (hκ : κ' < 1) :
    ¬ PrBad H (sksL Coin).V Rx (sksL Coin).E x silentP ≤
      κ' + Real.sqrt ((sksL Coin).γ x * PrB H (sksL Coin).V silentP x) +
        1 * (Claims.Sampling.Bext H (sksL Coin).V (sksL Coin).E Unit silentP x).Pr H := by
  intro h
  have hp : PrBad H (sksL Coin).V Rx (sksL Coin).E x silentP = 1 := prBad_accV H Rx _ x silentP hR
  have hb : PrB H (sksL Coin).V silentP x = 0 := prB_accV H silentP x
  rw [hp, hb, mul_zero, Real.sqrt_zero, add_zero, sksL_bext, mul_zero, add_zero] at h
  linarith

theorem sksL_opensAt (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (r : Fin ((sksL Coin).V.k x) → Coin)
    (c : (sksL Coin).SlotEC x) (g : ℕ) : (sksL Coin).opensAt H P x ω r c g = none := by
  have hp : (sksL Coin).parse x = some toyStmt := rfl
  have hc : (sksL Coin).commit = rejCm := rfl
  simp only [Compiled.opensAt, hp]
  simp [Claims.Sampling.Wit.val, Claims.Sampling.CommitmentScheme.Opens, hc, rejCm, OC.eval]

/-- `sksL`'s committed trace opens nothing. -/
theorem sksL_regVal (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (g : ℕ) :
    (sksL Coin).regVal H P x ω g = none := by
  simp [Compiled.regVal, sksL_opensAt, Pr_false]

/-- So its incorrect set holds the proof block. -/
theorem sksL_Freg (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) : 0 ∈ (sksL Coin).Freg H P x ω := by
  have hp : (sksL Coin).parse x = some toyStmt := rfl
  simp only [Compiled.Freg, hp]
  rw [Finset.mem_filter]
  refine ⟨Finset.mem_range.2 Nat.one_pos, ?_⟩
  rintro ⟨b, hb, val, ι, hv, hc⟩
  change toyPart.blocks[0]? = some b at hb
  rw [toyPart_blocks] at hb
  simp only [List.getElem?_cons_zero, Option.some.injEq] at hb
  subst hb
  obtain ⟨c, hc⟩ := toy_correct_val val ι hc
  rw [hv 0 Nat.one_pos, sksL_regVal] at hc
  cases hc

/-- Whose acceptance probability is `0`: the incorrect set is outside `M_0`. -/
theorem sksL_notM (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) :
    (sksL Coin).Freg H P x ω ∉ toyStmt.design.M ((sksL Coin).part toyStmt) 0 := by
  have hw : Claims.Sampling.window 0 1 ((sksL Coin).Freg H P x ω) ≠ ∅ := by
    intro hw
    have h0 : 0 ∈ Claims.Sampling.window 0 1 ((sksL Coin).Freg H P x ω) := by
      simp [Claims.Sampling.window, sksL_Freg H P x ω]
    rw [hw] at h0
    simp at h0
  intro hM
  have hpa : (0 : ℝ) < toyStmt.design.Pa toyPart ((sksL Coin).Freg H P x ω) := hM
  rw [toy_Pa, if_neg hw] at hpa
  exact lt_irrefl 0 hpa

/-- `sksL`'s committed form's error is `0`. -/
theorem sksL_εreg (H : Msg → Msg) (x : Msg) : (sksL Coin).εreg H x silentP = 0 := by
  have hb : PrB H (sksL Coin).V silentP x = 0 := prB_accV H silentP x
  unfold Compiled.εreg
  rw [sksL_κs, hb, bconf_pr H (sksL Coin) (fun _ _ _ => rfl) Unit silentP x]
  simp

end Numbers

/-! ## The claims -/

section Claims

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **`Arg.KS` fails** at `toyA`. -/
theorem argKS_fails (H : Msg → Msg) : ¬ (toyA Coin).KS H :=
  Witness.KS_fails H (toyA Coin).R (toyA Coin).E (toyA Coin).κ (toyA Coin).γ [] (fun _ => id)
    (show (0 : ℝ) < 1 by norm_num)

/-- **`Compiled.KS` fails** at `sksL`, at `δ = 0`. -/
theorem compiledKS_fails (H : Msg → Msg) : ¬ (sksL Coin).KS H := by
  intro h
  have h1 := h 0 le_rfl [] (sksL_scope []) Unit silentP
  have hb := sksL_bound_fails (Coin := Coin) H _ ((sksL Coin).κ 0 []) []
    (sksD_noR (Coin := Coin) H 0 le_rfl []) (by linarith [sksL_κ (Coin := Coin) []])
  exact hb h1

/-- **`Compiled.KSCount` fails** at `sksL`, at `t = 1`, whose miss is `0`. -/
theorem compiledKSCount_fails (H : Msg → Msg) : ¬ (sksL Coin).KSCount H := by
  intro h
  have h1 := h 1 [] (sksL_scope []) Unit silentP
  have hb := sksL_bound_fails (Coin := Coin) H _ ((sksL Coin).κ ((sksL Coin).missAt 1 []) []) []
    (sksD_noRCount (Coin := Coin) H []) (by rw [sksL_missAt]; linarith [sksL_κ (Coin := Coin) []])
  exact hb h1

/-- **`Compiled.KSReg` fails** at `sksL`'s silent prover, at `δ = 0`: it accepts always, with its committed trace's
incorrect set outside `M_0`, at an error of `0`. -/
theorem compiledKSReg_fails (H : Msg → Msg) : ¬ (sksL Coin).KSReg H := by
  intro h
  have h1 := (h [] (sksL_scope []) Unit silentP).2 toyStmt rfl 0 le_rfl
  have hp : Pr (fun p : Unit × (Fin ((sksL Coin).V.k []) → Coin) =>
      (run H silentP (sksL Coin).V [] p.1 p.2).out = .acc ∧
        (sksL Coin).Freg H silentP [] p.1 ∉ toyStmt.design.M ((sksL Coin).part toyStmt) 0) = 1 :=
    Eq.trans (Pr_congr fun p : Unit × (Fin ((sksL Coin).V.k []) → Coin) =>
      iff_true_intro ⟨run_accV_out H silentP [] p.1 p.2, sksL_notM H silentP [] p.1⟩) Claims.ZKMal.Pr_true
  have h2 : (1 : ℝ) ≤ 0 + (sksL Coin).εreg H [] silentP := hp.symm.trans_le h1
  rw [sksL_εreg] at h2
  norm_num at h2

end Claims

end Claims.Witness.SamplingGeneric

/-! ## Stated over the spec -/

namespace Claims.SamplingGeneric

/-- **`Arg.KS` fails** at some proof system. -/
theorem Arg.KS_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ∃ A : Arg Coin, ¬ A.KS H :=
  ⟨_, Witness.SamplingGeneric.argKS_fails H⟩

/-- **`Compiled.KS` fails** at some compiled verifier. -/
theorem Compiled.KS_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ∃ (A : Arg Coin) (L : Compiled Coin A), ¬ L.KS H :=
  ⟨_, _, Witness.SamplingGeneric.compiledKS_fails H⟩

/-- **`Compiled.KSCount` fails** at some compiled verifier. -/
theorem Compiled.KSCount_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ∃ (A : Arg Coin) (L : Compiled Coin A), ¬ L.KSCount H :=
  ⟨_, _, Witness.SamplingGeneric.compiledKSCount_fails H⟩

/-- **`Compiled.KSReg` fails** at some compiled verifier. -/
theorem Compiled.KSReg_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ∃ (A : Arg Coin) (L : Compiled Coin A), ¬ L.KSReg H :=
  ⟨_, _, Witness.SamplingGeneric.compiledKSReg_fails H⟩

end Claims.SamplingGeneric
