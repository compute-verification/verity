import Definitions.Claims.SamplingGeneric
import Proofs.Flock.Claims.Witnesses.Sampling
import Proofs.Flock.SamplingGeneric.Counter

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
  committed form's error is `0`: `Compiled.KSReg_fails`, at `δ = 0`. At no reruns and no own calls of `toyA`'s its
  `γ` is `0` and its work `1 + k + k²`, so `λ > 0`: `Compiled.Bits_fails`.
* `Counter.L₀ 1 0` (`SamplingGeneric/Counter.lean`): checking its one reveal costs one SHA-512 call at `openCalls = 0`,
  so its extractor's own calls exceed `Efficient`'s bound: `Compiled.Efficient_fails`.

The lock's `meets` witnesses of these claims are Flock's decoder bounds at its deployed records
(`Claims.Flock.decoder_meets_plain`, `_zk`), the checkable antecedent `FlockKS` and `FlockKSReg` read at Flock, and
`twoSessions_meets`: the contract `Compiled.Sessions` holds with two sampled sessions; for `Bits` and `Efficient` also
`twoCosts_meets`: the proof system's reruns and own calls and the opening checks' count are bounded there too.

* `twoA`: a proof system whose verifier accepts the one empty message, every witness in its relation, at errors `0`.
* `twoL`: a compiled verifier over `twoA` that samples both proof blocks of a two-input statement (`twoStmt`, partition
  `twoPart`), block `u`'s session at round `u`, under a commitment whose openings of `true` pass (`trueCm`). It meets
  `Compiled.Sessions` at every hash, its commitment binds, and the silent prover is accepted: `twoSessions_meets`.
-/

namespace Claims.Witness.SamplingGeneric

open Claims.SamplingGeneric
open Claims.Sampling (AuditStatement forkOn slack)
open Claims.Witness.Sampling (NoHash ownLog_noHash noHash_bind noHash_waitAccC noHash_flatMapRP rejCm toyProg toyStmt
  toyPart toyPart_blocks toy_boundary toy_Pa inScope_toy sksD sksD_noR sksD_noRCount sksD_outer sksD_missAt slack_pos
  slack_le_two)

/-! ## The toy instances -/

/-- The proof system whose verifier accepts after one round, whose relation is empty and whose extractor makes no wait,
at errors `0`. -/
def toyA (Coin : Type) : Arg Coin where
  Sh := Unit
  shape _ := ()
  V := accV Coin
  rounds _ := 1
  wf _ := rfl
  W := Unit
  R _ _ := False
  Op := Unit
  opens _ := []
  assemble _ _ := (none, 0)
  check _ _ := false
  waits _ := 0
  k _ := 0
  cap _ := 0
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

/-- A finder that outputs nothing finds nothing. -/
theorem finder_pr_none (H : Msg → Msg) (F : Finder) (h : ∀ s, F.out s = none) : F.Pr H = 0 := by
  unfold Finder.Pr
  rw [Pr_congr fun s => iff_false_intro (by simp [h s])]
  exact Pr_false

/-- At a compiled verifier with no sessions, `B_conf` finds nothing. -/
theorem bconf_pr (H : Msg → Msg) (L : Compiled Coin A) (hs : ∀ x u v, L.session x u v = none) (Ω : Type)
    [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg) : (Compiled.Bconf H L Ω P x).Pr H = 0 := by
  simp only [Compiled.Bconf]
  -- The finder's coins are the match's, so the parse is generalized everywhere at once.
  rcases L.parse x with _ | s
  · exact finder_pr_none H _ fun _ => rfl
  · exact finder_pr_none H _ fun q => by
      simp [ownLog_noHash _ _ _ (noHash_runExt L hs _ _ _ _ _), findColl]

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

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem sksL_scope (x : Msg) : (sksL Coin).Scope x := ⟨toyStmt, toyPart, rfl, rfl, inScope_toy⟩

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem sksL_κs (x : Msg) : (sksL Coin).κs x = 0 := by
  show ((List.range 1).map fun _ => (0 : ℝ)).foldr max 0 = 0
  simp

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
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
  generalize 2 * ((sksD Coin).k x : ℝ) / ((sksD Coin).cap x : ℝ) + Real.exp (-((sksD Coin).k x : ℝ) / 8) = X at ho ho0 ⊢
  calc (1 + slack) * X ≤ 3 * X := by gcongr; linarith
    _ ≤ 3 * (1 / 50) := by gcongr
    _ < 1 / 3 := by norm_num

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
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

open Classical in
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
  rw [toy_Pa] at hpa
  simp [hw] at hpa

/-- `sksL`'s committed form's error is `0`. -/
theorem sksL_εreg (H : Msg → Msg) (x : Msg) : (sksL Coin).εreg H x silentP = 0 := by
  have hb : PrB H (sksL Coin).V silentP x = 0 := prB_accV H silentP x
  unfold Compiled.εreg
  rw [sksL_κs, hb, bconf_pr H (sksL Coin) (fun _ _ _ => rfl) Unit silentP x]
  simp

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem sksL_κ_pos (x : Msg) : 0 < (sksL Coin).κ 0 x := by
  have := slack_pos
  unfold Compiled.κ
  rw [sksL_κs]
  positivity

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem sksL_γ (x : Msg) : (sksL Coin).γ x = 0 := by
  have h : (sksL Coin).γs x = 0 := by
    show ((List.range 1).map fun _ => (0 : ℝ) / (((toyA Coin).rounds () : ℕ) : ℝ)).foldr max 0 = 0
    simp
  unfold Compiled.γ
  rw [h, mul_zero, mul_zero]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- At no reruns and no own calls of `toyA`'s, `sksL`'s work is `1 + k + k²`. -/
theorem sksL_work (x : Msg) :
    (sksL Coin).work (fun _ => 0) (fun _ => 0) x = 1 + 2 ^ 40 * 3 + (2 ^ 40 * 3) ^ 2 := by
  have hk : ((sksL Coin).k x : ℝ) = 2 ^ 40 * 3 := by
    rw [show (sksL Coin).k x = 2 ^ 40 * 3 from rfl]; push_cast; ring
  have hn : ((sksL Coin).nPU x : ℝ) = 1 := by
    rw [show (sksL Coin).nPU x = 1 from rfl]; exact Nat.cast_one
  have he : (sksL Coin).es (fun _ => 0) x = 0 := by
    show ((List.range 1).map fun _ => (0 : ℝ)).foldr max 0 = 0
    simp
  have hh : (sksL Coin).hs (fun _ => 0) x = 0 := by
    show ((List.range 1).map fun _ => (0 : ℕ)).foldr max 0 = 0
    simp
  have ho : (sksL Coin).openCalls x = 0 := rfl
  unfold Compiled.work
  rw [hk, hn, he, hh, ho]
  push_cast
  ring

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem sksL_lam_pos (x : Msg) :
    0 < Claims.Sampling.lamLin ((sksL Coin).κ 0 x) ((sksL Coin).γ x) 1
      ((sksL Coin).work (fun _ => 0) (fun _ => 0) x) := by
  have hκ := sksL_κ (Coin := Coin) x
  have hκ0 := sksL_κ_pos (Coin := Coin) x
  have h3 : Real.logb 2 3 < 2 := by
    rw [Real.logb_lt_iff_lt_rpow (by norm_num) (by norm_num)]; norm_num
  have h1 : Real.logb 2 3 < Real.logb 2 (1 / (sksL Coin).κ 0 x) := by
    refine Real.logb_lt_logb (by norm_num) (by norm_num) ?_
    rw [lt_div_iff₀ hκ0]; linarith
  have hW : Real.logb 2 (2 * 1 * (sksL Coin).work (fun _ => 0) (fun _ => 0) x) ≤ 101 := by
    rw [sksL_work, Real.logb_le_iff_le_rpow (by norm_num) (by positivity)]
    norm_num
  unfold Claims.Sampling.lamLin
  rw [sksL_γ, Real.logb_zero]
  have := lt_min (lt_min h1 (show Real.logb 2 3 < 255.5 - 0 / 2 by linarith))
    (show Real.logb 2 3 < 256 - Real.logb 2 (2 * 1 * (sksL Coin).work (fun _ => 0) (fun _ => 0) x) by linarith)
  linarith

theorem sksL_expectedCR (H : Msg → Msg) (x : Msg) :
    ExpectedCR H (Claims.Sampling.Bext H (sksL Coin).V (sksL Coin).E Unit silentP x) := by
  unfold ExpectedCR
  rw [sksL_bext]
  exact div_nonneg (Ex_nonneg fun _ => Nat.cast_nonneg _) (by positivity)

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

/-- **`Compiled.Bits` fails** at `sksL`, at `δ = 0` and no reruns and no own calls of `toyA`'s: its silent prover makes
one call, is accepted, and no witness is in the relation, while `λ > 0`. -/
theorem compiledBits_fails (H : Msg → Msg) : ¬ (sksL Coin).Bits H (fun _ => 0) (fun _ => 0) := by
  intro h
  have hlam := sksL_lam_pos (Coin := Coin) []
  have hc := calls_silentP_accV (Coin := Coin) H []
  have hB := strictBirthday_B_accV (Coin := Coin) H silentP []
  have hE := sksL_expectedCR (Coin := Coin) H []
  have hp : PrBad H (sksL Coin).V (Claims.Sampling.R H (sksL Coin).commit (sksL Coin).parse (sksL Coin).partition 0)
      (sksL Coin).E [] silentP = 1 :=
    prBad_accV H _ _ [] silentP (sksD_noR (Coin := Coin) H 0 le_rfl [])
  have h2 : PrBad H (sksL Coin).V (Claims.Sampling.R H (sksL Coin).commit (sksL Coin).parse (sksL Coin).partition 0)
      (sksL Coin).E [] silentP ≤
        (1 : ℕ) * (2 : ℝ) ^ (-Claims.Sampling.lamLin ((sksL Coin).κ 0 []) ((sksL Coin).γ []) 1
          ((sksL Coin).work (fun _ => 0) (fun _ => 0) [])) :=
    h 0 le_rfl [] (sksL_scope []) Unit silentP 1 le_rfl hc hB hE
  rw [hp, Nat.cast_one, one_mul] at h2
  have : (2 : ℝ) ^ (-Claims.Sampling.lamLin ((sksL Coin).κ 0 []) ((sksL Coin).γ []) 1
      ((sksL Coin).work (fun _ => 0) (fun _ => 0) [])) < 1 :=
    Real.rpow_lt_one_of_one_lt_of_neg (by norm_num) (by linarith)
  linarith

end Claims

/-! ## Two sampled sessions meet the contract -/

/-- Two input gates, both returned. -/
def twoProg : Claims.Sampling.Prog := ⟨[.input, .input], [0, 1]⟩

/-- The program, drawn `all`. -/
def twoStmt : AuditStatement := ⟨[], twoProg, ⟨"", 0, []⟩, ⟨[.all]⟩, []⟩

/-- One class of one stratum of two proof blocks, `{0}` and `{1}`. -/
def twoPart : Claims.Sampling.Partition := ⟨[[[[0], [1]]]]⟩

/-- The commitment whose openings of `true` pass and of `false` fail, with no call: two openings that pass open one
value, so it binds at every hash. -/
def trueCm : Claims.Sampling.CommitmentScheme Bool := ⟨fun _ o => .ret o.val, fun _ _ _ => []⟩

/-- A session's decision on the prover's messages: accept the one empty message. -/
def sesOut (as : List Msg) : Out := if as = [[]] then .acc else .rej

/-- The sessions' verifier: one round, accepting the empty message, with no call. -/
def sesV (Coin : Type) : Verifier Coin := ⟨fun _ => 1, fun _ as _ => .ret (sesOut as)⟩

/-- The compiled verifier: two rounds, session `u`'s at round `u`, accepting when both messages are empty. -/
def twoV (Coin : Type) : Verifier Coin :=
  ⟨fun _ => 2, fun _ as _ => .ret (if as.length < 2 then .msg [] else if as = [[], []] then .acc else .rej)⟩

/-- The proof system the sessions run: `sesV`, every witness in the relation, at errors `0`; its extractor waits for
no accepted rerun and assembles the witness. -/
def twoA (Coin : Type) : Arg Coin where
  Sh := Unit
  shape _ := ()
  V := sesV Coin
  rounds _ := 1
  wf _ := rfl
  W := Unit
  R _ _ := True
  Op := Unit
  opens _ := []
  assemble _ _ := (some (), 0)
  check _ _ := true
  waits _ := 1
  k _ := 0
  cap _ := 1
  κ _ := 0
  γ _ := 0

/-- Session `u` of a run of `twoV`: the run's `u`-th message, decided as `sesV` decides it. -/
def twoSession (u : ℕ) (v : Seen) : Option (Msg × Seen) :=
  v.msgs[u]?.map fun m => ([], ⟨[(m.1, sesOut [m.1])], sesOut [m.1]⟩)

/-- The compiled verifier over `twoA` that samples both proof blocks, block `u`'s session at round `u`, under
`trueCm`, a session's witness revealing its block's input as `true`. -/
def twoL (Coin : Type) : Compiled Coin (twoA Coin) where
  V := twoV Coin
  parse _ := some twoStmt
  scopeCheck _ := true
  partition _ := some twoPart
  commit := trueCm
  stage2 _ := Finset.univ
  checked _ _ := [0, 1]
  sh _ _ := ()
  κb _ _ := 0
  γb _ _ := 0
  rounds _ u := if h : u < 2 then {⟨u, h⟩} else ∅
  session _ u v := twoSession u v
  iroots _ _ _ := []
  reveal s u _ _ := [(s.roots, ⟨u, true, []⟩)]
  openCalls _ := 0
  Tape := Unit
  P _ := silentP
  δc _ _ := 0

section Two

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- A prover's history against `twoV` before round `u`: none before round `0`, its first message before round `1`. -/
def twoHist (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) : ℕ → List (Msg × Msg)
  | 0 => []
  | _ + 1 => [(((P.next x [] ω).eval H).1, [])]

/-- Its message at round `u`. -/
def twoMsg (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (u : ℕ) : Msg :=
  ((P.next x (twoHist H P x ω u) ω).eval H).1

/-- Session `u`'s prover: the prover at its history before round `u`. -/
def twoP (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (u : ℕ) : Prover Unit :=
  ⟨fun _ _ _ => P.next x (twoHist H P x ω u) ω⟩

/-- `twoV`'s decision on two messages. -/
def twoOut (a b : Msg) : Out := if [a, b] = [[], []] then .acc else .rej

omit [Fintype Coin] [Nonempty Coin] in
theorem twoL_run (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (r : Fin ((twoL Coin).V.k x) → Coin) :
    (run H P (twoL Coin).V x ω r).seen =
      ⟨[(twoMsg H P x ω 0, .msg []), (twoMsg H P x ω 1, twoOut (twoMsg H P x ω 0) (twoMsg H P x ω 1))],
        twoOut (twoMsg H P x ω 0) (twoMsg H P x ω 1)⟩ := by
  by_cases h : [twoMsg H P x ω 0, twoMsg H P x ω 1] = [[], []]
  · simp only [twoMsg, twoHist] at h ⊢
    simp [run, twoL, twoV, stepWith, OC.eval, Trace.seen, twoOut, h]
  · simp only [twoMsg, twoHist] at h ⊢
    simp [run, twoL, twoV, stepWith, OC.eval, Trace.seen, twoOut, h]

omit [Fintype Coin] [Nonempty Coin] in
theorem twoA_run (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (u : ℕ) (xS : Msg)
    (ρ : Fin ((twoA Coin).V.k xS) → Coin) :
    (run H (twoP H P x ω u) (twoA Coin).V xS () ρ).seen =
        ⟨[(twoMsg H P x ω u, sesOut [twoMsg H P x ω u])], sesOut [twoMsg H P x ω u]⟩ ∧
      (run H (twoP H P x ω u) (twoA Coin).V xS () ρ).vLog = [] := by
  by_cases h : [twoMsg H P x ω u] = [[]]
  · simp only [twoMsg] at h ⊢
    simp [run, twoA, sesV, twoP, stepWith, OC.eval, Trace.seen, sesOut, h]
  · simp only [twoMsg] at h ⊢
    simp [run, twoA, sesV, twoP, stepWith, OC.eval, Trace.seen, sesOut, h]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem twoL_parse {x : Msg} {s : AuditStatement} (h : (twoL Coin).parse x = some s) : s = twoStmt :=
  (Option.some.inj h).symm

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem twoL_partition {s : AuditStatement} {P : Claims.Sampling.Partition} (h : (twoL Coin).partition s = some P) : P = twoPart :=
  (Option.some.inj h).symm

theorem twoPart_blocks : twoPart.blocks = [[0], [1]] := rfl

/-- The toy is in the sampling scope. -/
theorem inScope_two : Claims.Sampling.InScope twoPart twoStmt := by
  refine ⟨⟨fun g x hx i hi => ?_, by decide⟩,
    ⟨by decide, by decide, by decide, by decide, by decide⟩, ?_⟩
  · rcases g with _ | _ | g
    · simp [twoStmt, twoProg] at hx
      subst hx
      simp [Claims.Sampling.BGate.reads] at hi
    · simp [twoStmt, twoProg] at hx
      subst hx
      simp [Claims.Sampling.BGate.reads] at hi
    · simp [twoStmt, twoProg] at hx
  · refine ⟨rfl, ?_⟩
    simp [twoStmt, twoPart, Claims.Sampling.OneStage.WellFormed]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem twoL_scope (x : Msg) : (twoL Coin).Scope x := ⟨twoStmt, twoPart, rfl, rfl, inScope_two⟩

theorem two_Pa (B : Finset ℕ) :
    twoStmt.design.Pa twoPart B = if Claims.Sampling.window 0 2 B = ∅ then 1 else 0 := by
  simp [Claims.Sampling.Design.Pa, twoStmt, twoPart, Claims.Sampling.classesPa, Claims.Sampling.OneStage.Pa]

theorem two_Pa_nonneg (B : Finset ℕ) : 0 ≤ twoStmt.design.Pa twoPart B := by
  rw [two_Pa]
  split_ifs <;> norm_num

/-- Block `u` is correct where its input's value is committed. -/
theorem two_correct {u : ℕ} (hu : u < 2) (val : ℕ → Option Bool) (ι : ℕ → Bool) (hv : val u = some true) :
    twoProg.Correct (twoProg.refIn [u] val ι) [u] := by
  intro g hg
  rw [List.mem_singleton] at hg
  subst hg
  have hb : twoProg.Boundary [g] g = true := by
    interval_cases g <;> decide
  have hg : twoProg.gates[g]? = some .input := by
    interval_cases g <;> rfl
  have hr : twoProg.refIn [g] val ι g = some true := by
    simp [Claims.Sampling.Prog.refIn, hb, hv]
  refine ⟨fun x hx i hi => ?_, fun _ => ⟨true, hr, ?_⟩⟩
  · rw [hg] at hx
    cases hx
    simp [Claims.Sampling.BGate.reads] at hi
  · rw [Claims.Sampling.Prog.runIn, hg]
    simpa [Claims.Sampling.BGate.val] using hr

/-- `twoA` is knowledge sound at errors `0`: at an accepted run its extractor assembles the one witness, which is in
the relation. -/
theorem twoA_KS (H : Msg → Msg) : (twoA Coin).KS H := by
  intro x Ω _ _ P
  have hf : ∀ s : Ω × (Fin ((twoA Coin).V.k x) → Coin) × (twoA Coin).E.C x,
      ¬ ((run H P (twoA Coin).V x s.1 s.2.1).out = .acc ∧
        ¬ (twoA Coin).R.ext x ((twoA Coin).E.eval H P x s.1 s.2.1 s.2.2).1) := by
    rintro s ⟨hacc, hR⟩
    apply hR
    have hacc' : (run H P (twoA Coin).V x s.1 s.2.1).seen.out = .acc := hacc
    have hfin : List.finRange 1 = [(0 : Fin 1)] := rfl
    simp only [Extractor.eval, Arg.E, Arg.ext, ERep, RP.eval, hacc', ite_true]
    simp [twoA, hfin, firstWait, waitAcc, RP.bind, RP.eval, Rel.ext]
  have h0 : PrBad H (twoA Coin).V (twoA Coin).R (twoA Coin).E x P = 0 := by
    unfold PrBad
    rw [Pr_congr fun s => iff_false_intro (hf s)]
    exact Pr_false
  rw [h0]
  simp [twoA]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem trueCm_binds (H : Msg → Msg) : (twoL Coin).commit.Binds H := by
  intro roots o o' _ hne ho ho'
  have h1 : o.val = true := ho
  have h2 : o'.val = true := ho'
  exact absurd (h1.trans h2.symm) hne

/-- `twoL` meets the contract at every hash. -/
theorem twoL_sessions (H : Msg → Msg) : (twoL Coin).Sessions H where
  draws := by
    intro x s P hs hP _ B hB
    obtain rfl := twoL_parse hs
    obtain rfl := twoL_partition hP
    rw [twoPart_blocks] at hB
    refine ⟨fun _ => twoStmt.design.Pa twoPart B, fun _ => two_Pa_nonneg B, fun r => ?_, (Ex_const _).le⟩
    by_cases h0 : B = ∅
    · subst h0
      have h1 : twoStmt.design.Pa twoPart ∅ = 1 := by rw [two_Pa]; simp [Claims.Sampling.window]
      rw [h1]
      exact Pr_le_one _
    · obtain ⟨v, hv⟩ := Finset.nonempty_iff_ne_empty.2 h0
      have hv2 : v < 2 := Finset.mem_range.1 (hB hv)
      have hn : ∀ d : Fin ((twoL Coin).V.k x) → Coin,
          ¬ ∀ u ∈ (twoL Coin).checked x (forkOn ((twoL Coin).stage2 x) r d), u ∉ B := by
        intro d hall
        refine hall v ?_ hv
        show v ∈ [0, 1]
        interval_cases v <;> simp
      rw [Pr_congr fun d => iff_false_intro (hn d), Pr_false]
      exact two_Pa_nonneg B
  slots _ _ := le_rfl
  refuses x _ _ _ _ _ := twoL_scope x
  interior _ _ _ _ _ _ _ := rfl
  fixed _ _ _ _ := ⟨Finset.subset_univ _, fun _ => rfl⟩
  runs := by
    intro x Ω P ω r u hu
    have hu2 : u < 2 := by
      change u ∈ [0, 1] at hu
      simp at hu
      omega
    refine ⟨[], twoP H P x ω u, rfl, le_rfl, le_rfl, ?_, fun r' _ hacc => ?_, fun r' _ => ⟨?_, ?_⟩⟩
    · show 1 = (if h : u < 2 then ({⟨u, h⟩} : Finset (Fin 2)) else ∅).card
      simp [hu2]
    · have ho : twoOut (twoMsg H P x ω 0) (twoMsg H P x ω 1) = .acc :=
        (congrArg Seen.out (twoL_run H P x ω r')).symm.trans hacc
      have hm : twoMsg H P x ω 0 = [] ∧ twoMsg H P x ω 1 = [] := by simpa [twoOut] using ho
      have hs : sesOut [twoMsg H P x ω u] = .acc := by
        interval_cases u <;> simp [sesOut, hm.1, hm.2]
      exact (congrArg Seen.out (twoA_run H P x ω u [] ((twoL Coin).coinsOf x u [] r')).1).trans hs
    · change twoSession u (run H P (twoL Coin).V x ω r').seen = _
      rw [twoL_run, (twoA_run H P x ω u [] _).1]
      interval_cases u <;> rfl
    · intro q hq
      rw [(twoA_run H P x ω u [] _).2] at hq
      cases hq
  opens := by
    intro x s P Ω Pv ω r u xS vS w hs hP _ _ hu _ _
    obtain rfl := twoL_parse hs
    obtain rfl := twoL_partition hP
    have hu2 : u < 2 := by
      change u ∈ [0, 1] at hu
      simp at hu
      omega
    refine ⟨[u], by rw [twoPart_blocks]; interval_cases u <;> rfl, fun val ι hq => ?_⟩
    have hv := (hq (twoStmt.roots, ⟨u, true, []⟩) (List.mem_singleton_self _) rfl).1 rfl
    exact two_correct hu2 val ι hv

omit [Fintype Coin] [Nonempty Coin] in
/-- The silent prover is accepted. -/
theorem twoL_silent (H : Msg → Msg) (r : Fin ((twoL Coin).V.k []) → Coin) :
    (run H silentP (twoL Coin).V [] () r).out = .acc := by
  have h0 : twoMsg H silentP [] () 0 = [] := rfl
  have h1 : twoMsg H silentP [] () 1 = [] := rfl
  have ho : (run H silentP (twoL Coin).V [] () r).seen.out = .acc := by
    rw [twoL_run, h0, h1]
    simp [twoOut]
  exact ho

omit [Inhabited Coin] in
/-- `twoA`'s extractor: the replay, then its witness if the replay accepted. -/
theorem twoA_ext (x : Msg) (r₀ : Fin ((twoA Coin).V.k x) → Coin) (c : (twoA Coin).E.C x) :
    (twoA Coin).E.ext x r₀ c = .rerun r₀ fun v₀ => if v₀.out = .acc then .ret (some ()) else .ret none := rfl

omit [Inhabited Coin] in
theorem twoA_eval_runs (H : Msg → Msg) (x : Msg) {Ω : Type} (P : Prover Ω) (ω : Ω)
    (r₀ : Fin ((twoA Coin).V.k x) → Coin) (c : (twoA Coin).E.C x) : ((twoA Coin).E.eval H P x ω r₀ c).2.2 = 1 := by
  show (((twoA Coin).E.ext x r₀ c).eval H (run H P (twoA Coin).V x ω)).2.2 = 1
  rw [twoA_ext]
  by_cases h : (run H P (twoA Coin).V x ω r₀).seen.out = .acc <;> simp [RP.eval, h]

omit [Inhabited Coin] in
/-- `twoA`'s extractor reruns once, its replay. -/
theorem twoA_runs (H : Msg → Msg) : (twoA Coin).RerunsLe H (fun _ => 1) := fun x _ P ω r₀ c =>
  (twoA_eval_runs H x P ω r₀ c).le

omit [Inhabited Coin] in
theorem twoA_runsExp (H : Msg → Msg) : (twoA Coin).RerunsExpLe H (fun _ => 1) := by
  intro x Ω P ω
  have h1 : ∀ s : (Fin ((twoA Coin).V.k x) → Coin) × (twoA Coin).E.C x,
      (((twoA Coin).E.eval H P x ω s.1 s.2).2.2 : ℝ) = 1 := fun s => by
    rw [twoA_eval_runs]; exact Nat.cast_one
  simp only [h1, Ex_const, le_refl]

omit [Inhabited Coin] in
/-- `twoA`'s extractor makes no SHA-512 call of its own. -/
theorem twoA_own (H : Msg → Msg) : (twoA Coin).OwnLe H (fun _ => 0) := fun x _ P ω r₀ c => by
  show (ownLog H (run H P (twoA Coin).V x ω) ((twoA Coin).E.ext x r₀ c)).length ≤ 0
  rw [twoA_ext]
  by_cases h : (run H P (twoA Coin).V x ω r₀).seen.out = .acc <;> simp [ownLog, h]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- Checking a session's reveal under `trueCm` makes no call, at `openCalls = 0`. -/
theorem twoL_openCalls (H : Msg → Msg) : (twoL Coin).OpenCallsLe H := fun _ _ _ _ _ _ => Nat.le_refl 0

end Two

/-- **Two sampled sessions meet the contract** (`twoL`, at every hash): over a knowledge-sound proof system (`twoA`)
with a binding commitment, a compiled verifier that samples both proof blocks of a statement in scope, each block's
session at its own round, meets `Compiled.Sessions`, and the silent prover is accepted. So `CompilerKS`,
`CompilerKSReg` and `CompilerKSCount` read antecedents that hold with more than one sampled session. -/
theorem twoSessions_meets {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    (twoA Coin).KS H ∧ (twoL Coin).commit.Binds H ∧ (twoL Coin).Sessions H ∧ (twoL Coin).Scope [] ∧
      ∀ r, (twoL Coin).checked [] r = [0, 1] ∧ (run H silentP (twoL Coin).V [] () r).out = .acc :=
  ⟨twoA_KS H, trueCm_binds H, twoL_sessions H, twoL_scope [], fun r => ⟨rfl, twoL_silent H r⟩⟩

/-- **The costs at the two sessions are bounded** (`twoA`, `twoL`, at every hash): the proof system's extractor reruns
once, its replay, strictly and in expectation, and makes no SHA-512 call of its own, and checking a session's reveal
makes none, at `openCalls = 0`. So `CompilerBits` and `CompilerEfficient` read antecedents that hold, with
`twoSessions_meets`', with more than one sampled session. -/
theorem twoCosts_meets {Coin : Type} [Fintype Coin] [Nonempty Coin] (H : Msg → Msg) :
    (twoA Coin).RerunsLe H (fun _ => 1) ∧ (twoA Coin).RerunsExpLe H (fun _ => 1) ∧ (twoA Coin).OwnLe H (fun _ => 0) ∧
      (twoL Coin).OpenCallsLe H :=
  ⟨twoA_runs H, twoA_runsExp H, twoA_own H, twoL_openCalls H⟩

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

/-- **`Compiled.Bits` fails** at some compiled verifier and some costs of its proof system. -/
theorem Compiled.Bits_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ∃ (A : Arg Coin) (L : Compiled Coin A) (e : A.Sh → ℝ) (h : A.Sh → ℕ), ¬ L.Bits H e h :=
  ⟨_, _, _, _, Witness.SamplingGeneric.compiledBits_fails H⟩

/-- **`Compiled.Efficient` fails** at some compiled verifier and some costs of its proof system. -/
theorem Compiled.Efficient_fails (H : Msg → Msg) :
    ∃ (A : Arg Unit) (L : Compiled Unit A) (ρ : A.Sh → ℕ) (e : A.Sh → ℝ) (h : A.Sh → ℕ), ¬ L.Efficient H ρ e h :=
  ⟨_, _, _, _, _, Counter.not_efficient_openCalls H⟩

end Claims.SamplingGeneric
