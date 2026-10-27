import Proofs.Flock.Claims.Witnesses.Sampling
import Proofs.NetworkCertifier.RecordRows
import Definitions.NetworkCertifier.HorizonLeak
import Definitions.NetworkCertifier.TapRows
import Definitions.NetworkCertifier.SampledAudit
import Definitions.NetworkCertifier.SampledReg
import Definitions.NetworkCertifier.Schedule
import Definitions.NetworkCertifier.Assumptions

/-!
# The instances the network certifier's witnesses are stated at

Each built by hand, with the facts about it that read no deployed record (`Witnesses`, `PeriodWitnesses` state the
witnesses):

* `notStmt`: one input gate and its negation, drawn `all`, its one proof unit `{0, 1}` (`notPart`), at a layout whose
  ingress is the input and whose egress slot's presence gate is the negation (`notLay`). With nothing held and the
  input `false`, the circuit computes `true` there, so the step `obs0`, which carries no frame, is accepted under no
  possible faulty set (`obs0_notAcc`): the one unit is drawn `all`, so the only one is empty.
* `leakD`: `sksD` at `notStmt`, a verifier that accepts everything, where no opening passes, so no witness is in
  `RLeak` (`leakD_noRLeak`). `tapD`: `leakD` with every opening passing (`accCm`), where the network certifier's
  openings `tapOps` register `obs0` (`tapOps_opens`), so no witness is in `RLeakTap` (`tapD_noRLeakTap`) and the
  per-window leak bound of `PeriodLeakAccepted` fails at its silent prover (`tapD_bound_fails`).
* `horizon_bound_fails`: `HorizonLeakAccepted`'s bound fails at an audit that accepts everything, at one window whose
  faulty set holds the one unit `Fmax = 1` allows none of.
* `fxRows`: a rows format of the Python network certifier's fixture `fxQueue` (three connections, `fixtureAgrees`
  checks its egress record), one window of ten buckets of four slots, which `fxQueueSched` fits (`fxQueue_fits`).
  `Fits` reads the format's connections and its buckets a window, not its frames' encoding.
* `fxRate`: the constant-rate Program at one bucket (one window of one bucket, one link of one slot, the one sync value
  of `syncOne`), and `fxFixed`, it with its clock fixed. Both accept the honest step `fxHonest`
  (`fxHonest_accepted`). `AccSeqsAt acc` is `AccSeqs` with `acc` in place of the Program's acceptance, and at the
  verifier that accepts everything, `acceptAll`, one step's sequences are infinitely many
  (`accSeqsAt_acceptAll_infinite`).
* `fxHonestI`: the honest certifier at `fxRate`, whose first step is `fxHonest` (`fxHonestI_obsRun`), and `fxLeakI`, a
  sender whose secret is its one frame's length, which `fxLeakDec` reads back from the wire. Both meet W2 with the wire
  their record and J at `D = 1` (`fx_wire`), and at `acceptAll` every secret of `fxLeakI` is decoded (`fxLeak_not_le`).
-/

namespace NetTiming.Witness

open Claims Claims.Sampling Claims.Witness Claims.Witness.Sampling

/-! ## The negation's statement -/

/-- A layout of one link of one slot, with no gate. -/
def trivLay : Layout 1 1 := ⟨fun _ _ _ => (0, []), fun _ => []⟩

/-- One input gate and its negation, the negation returned. -/
def notProg : Claims.Sampling.Prog := ⟨[.input, .not 0], [1]⟩

/-- The program, drawn `all`. -/
def notStmt : Claims.Sampling.Stmt := ⟨[], notProg, ⟨"", 0, []⟩, ⟨[⟨.all, none⟩]⟩, []⟩

/-- One class of one stratum of one replay unit, the proof unit `{0, 1}`. -/
def notPart : Claims.Sampling.Partition := ⟨[[[[[0, 1]]]]]⟩

/-- Step `t`'s ingress is the input, and its egress slot's presence gate the negation. -/
def notLay : Layout 1 1 := ⟨fun _ _ _ => (1, []), fun _ => [0]⟩

/-- One step: no frame on the link, every status complete. -/
def obs0 : Obs 1 1 1 := ⟨fun _ _ => none, 0, fun _ => .complete⟩

theorem inScope_not : Claims.Sampling.InScope notPart notStmt := by
  refine ⟨⟨fun g x hx i hi => ?_, by decide⟩,
    ⟨by decide, by decide, by decide, by decide, by decide, by decide⟩, rfl, ?_⟩
  · rcases g with _ | _ | g
    · simp [notStmt, notProg] at hx
      subst hx
      simp [BGate.reads] at hi
    · simp [notStmt, notProg] at hx
      subst hx
      simp [BGate.reads] at hi
      omega
    · simp [notStmt, notProg] at hx
  · simp [notStmt, notPart, Claims.Sampling.Rule.WellFormed, Claims.Sampling.OneStage.WellFormed]

theorem notLay_fits : notLay.Fits notProg 1 := by
  refine ⟨fun t _ ℓ σ g hg => ?_, fun t _ g hg => ?_, fun t _ t' _ htt' => absurd htt' (by omega)⟩
  · simp [notLay, slotGates] at hg
    subst hg
    decide
  · simp [notLay] at hg
    subst hg
    rfl

theorem rejCm_val_none (H : Msg → Msg) (s : Claims.Sampling.Stmt) (w : Claims.Sampling.Wit) (g : ℕ) :
    w.val H rejCm s g = none := by
  simp [Wit.val, Commitment.Opens, rejCm, OC.eval]

/-- With nothing held and the input `false`, the circuit computes the negation `true`. -/
theorem notProg_val (inp : ℕ → Bool) (h0 : inp 0 = false) : valOver notProg inp (fun _ => none) 1 = true := by
  have hg0 : notProg.gates[0]? = some .input := rfl
  have hg1 : notProg.gates[1]? = some (.not 0) := rfl
  have hv0 : valOver notProg inp (fun _ => none) 0 = false := by
    rw [valOver]
    simp [hg0, BGate.val, h0]
  rw [valOver]
  simp [hg1, BGate.val, hv0]

theorem obs0_covered : Covered 1 [obs0] := by
  intro t ht _ κ
  obtain rfl : t = 0 := by omega
  rfl

/-- **`obs0` is accepted under no possible faulty set**, whatever the inputs `As` admits, at one step at `δ = 0`: the
only possible faulty set is empty, since the one unit is drawn `all`, so no gate is held, and the circuit, the negation
of the ingress `false`, holds a frame where `obs0` has none. -/
theorem obs0_notAcc (As : Set (ℕ → Bool)) :
    [obs0] ∉ AccSeqsSampled notProg notPart notStmt.design notLay 1 As 0 (fun _ _ => [false]) 1 := by
  intro h
  obtain ⟨-, -, F, hF, a, -, free, hfree, hacc⟩ := h
  have hF0 : F = ∅ := by
    classical
    simp only [faultySets, Finset.mem_filter, Finset.mem_powerset] at hF
    obtain ⟨hsub, -, hdis⟩ := hF
    refine Finset.eq_empty_of_forall_notMem fun v hv => ?_
    have hv1 : v < 1 := Finset.mem_range.1 (hsub hv)
    obtain rfl : v = 0 := by omega
    exact Finset.disjoint_left.1 hdis hv (by decide)
  subst hF0
  have hnone : free = fun _ => none := funext fun g => by
    cases hg : free g with
    | none => rfl
    | some b =>
      obtain ⟨v, hv, -⟩ := hfree g b hg
      simp at hv
  subst hnone
  obtain ⟨_, hrows, -⟩ := hacc 0 Nat.one_pos
  have hin : notLay.inputOf 1 [[false]] a 0 = false := rfl
  have h1 : (none : Option (List Bool)) =
      rowOf (valOver notProg (notLay.inputOf 1 [[false]] a) (fun _ => none)) (notLay.egress 0 0) 0 :=
    congrFun (congrFun hrows 0) 0
  have hrow : rowOf (valOver notProg (notLay.inputOf 1 [[false]] a) (fun _ => none)) (notLay.egress 0 0) 0 =
      if valOver notProg (notLay.inputOf 1 [[false]] a) (fun _ => none) 1 then
        some (([] : List ℕ).map (valOver notProg (notLay.inputOf 1 [[false]] a) (fun _ => none)))
      else none := rfl
  rw [hrow, notProg_val _ hin] at h1
  simp at h1

/-! ## The accepting verifiers at the negation -/

section Toy

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- `sksD` at `notStmt`: the verifier accepts, no opening passes, and no unit has a session. -/
noncomputable def leakD (Coin : Type) : Claims.Sampling.Deploy Coin Unit Unit :=
  { sksD Coin with parse := fun _ => some notStmt, partition := fun _ => some notPart }

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem leakD_scopeAcc (x : Msg) : ScopeAcc (leakD Coin) (fun _ => notLay) (fun _ => 1) x :=
  ⟨⟨notStmt, notPart, rfl, rfl, inScope_not⟩, notStmt, rfl, notLay_fits⟩

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem leakD_κ (x : Msg) : Claims.Sampling.κ (leakD Coin) 0 x < 1 / 3 :=
  sksD_κ (Coin := Coin) x

theorem leakD_bext (H : Msg → Msg) (decode : List (ℕ × Unit) → List Unit) (x : Msg) :
    (Bext H (leakD Coin).V (Claims.Sampling.E (leakD Coin) decode) Unit silentP x).Pr H = 0 := by
  simp only [Bext]
  rw [dite_eq_left (show 0 < (leakD Coin).V.k x from Nat.one_pos)]
  unfold Finder.Pr
  rw [Pr_congr fun s => iff_false_intro (by
    simp [ownLog_noHash _ _ _ (noHash_E (leakD Coin) (fun _ _ _ => rfl) _ _ _ _), findColl])]
  exact Pr_false

theorem leakD_breg (H : Msg → Msg) (decode : List (ℕ × Unit) → List Unit) (x : Msg) :
    (Breg H (leakD Coin).V (Claims.Sampling.E (leakD Coin) decode) (leakD Coin).commit (leakD Coin).parse
      (fun _ => []) Unit silentP x).Pr H = 0 := by
  unfold Finder.Pr
  rw [Pr_congr fun s => iff_false_intro (by
    simp only [Breg, ownLog_noHash _ _ _ (noHash_E (leakD Coin) (fun _ _ _ => rfl) _ _ _ _)]
    simp [regLog, leakD, findColl])]
  exact Pr_false

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- **No witness is in `RLeak`** at `leakD`, at one step at `δ = 0`, whatever the inputs `As` admits: the step the
trace carries has no frame, where the circuit, the negation of the ingress the trace carries, holds one, and the only
possible faulty set is empty, since the one unit is drawn `all` (`obs0_notAcc`). -/
theorem leakD_noRLeak (H : Msg → Msg) (As : Claims.Sampling.Stmt → (ℕ → Bool) → Set (ℕ → Bool)) (x : Msg)
    (w : Claims.Sampling.Wit) :
    ¬ RLeak H (leakD Coin).commit (leakD Coin).parse (leakD Coin).partition 1 1 1 (fun _ => notLay) (fun _ => 1) 0 As 1
      x w := by
  rintro ⟨s, P, hs, hP, h⟩
  simp only [leakD, Option.some.injEq] at hs hP
  subst hs hP
  have hval := rejCm_val_none H notStmt w
  have hcar : Carries notLay notProg (w.val H rejCm notStmt) (fun _ _ => [false]) 1 [obs0] := by
    intro t ht
    obtain rfl : t = 0 := by omega
    refine ⟨Nat.one_pos, fun ℓ => ?_, ?_⟩
    · funext σ
      simp [obs0, rowOf, notLay, traceVal, Prog.read, notProg, hval]
    · simp [notLay, traceVal, Prog.read, notProg, hval]
  exact obs0_notAcc _ (h (fun _ _ => [false]) [obs0] hcar obs0_covered rfl (by simp [obs0]))

/-- `KSLin`'s bound fails at `leakD`'s silent prover, at a relation with no witness at `x` and a bound `κ' < 1`. -/
theorem leakD_bound_fails (H : Msg → Msg) (decode : List (ℕ × Unit) → List Unit) (Rx : Rel Claims.Sampling.Wit)
    (κ' : ℝ) (x : Msg) (hR : ∀ w, ¬ Rx x w) (hκ : κ' < 1) :
    ¬ PrBad H (leakD Coin).V Rx (Claims.Sampling.E (leakD Coin) decode) x silentP ≤
      κ' + Real.sqrt (Claims.Sampling.γ (leakD Coin) x * PrB H (leakD Coin).V silentP x) +
        1 * (Bext H (leakD Coin).V (Claims.Sampling.E (leakD Coin) decode) Unit silentP x).Pr H := by
  intro h
  have hp : PrBad H (leakD Coin).V Rx (Claims.Sampling.E (leakD Coin) decode) x silentP = 1 :=
    prBad_accV H Rx _ x silentP hR
  have hb : PrB H (leakD Coin).V silentP x = 0 := prB_accV H silentP x
  simp only [hp, hb, mul_zero, Real.sqrt_zero, add_zero, leakD_bext] at h
  linarith

/-- `SampledLeakRegistered`'s bound fails at `leakD`'s silent prover, at a relation with no witness at `x`, a bound
`κ' < 1` and no registration. -/
theorem leakD_reg_bound_fails (H : Msg → Msg) (decode : List (ℕ × Unit) → List Unit) (Rx : Rel Claims.Sampling.Wit)
    (κ' : ℝ) (x : Msg) (hR : ∀ w, ¬ Rx x w) (hκ : κ' < 1) :
    ¬ PrBad H (leakD Coin).V Rx (Claims.Sampling.E (leakD Coin) decode) x silentP ≤
      κ' + Real.sqrt (Claims.Sampling.γ (leakD Coin) x * PrB H (leakD Coin).V silentP x) +
        (Bext H (leakD Coin).V (Claims.Sampling.E (leakD Coin) decode) Unit silentP x).Pr H +
        (Breg H (leakD Coin).V (Claims.Sampling.E (leakD Coin) decode) (leakD Coin).commit (leakD Coin).parse
          (fun _ => []) Unit silentP x).Pr H := by
  intro h
  have hp : PrBad H (leakD Coin).V Rx (Claims.Sampling.E (leakD Coin) decode) x silentP = 1 :=
    prBad_accV H Rx _ x silentP hR
  have hb : PrB H (leakD Coin).V silentP x = 0 := prB_accV H silentP x
  simp only [hp, hb, mul_zero, Real.sqrt_zero, add_zero, leakD_bext, leakD_breg] at h
  linarith

/-- The network certifier's openings at `notStmt`: the input `false` and the negation's presence gate `false`. -/
def tapOps : List (Opening Bool) := [⟨0, false, []⟩, ⟨1, false, []⟩]

/-- `leakD` with every opening passing. -/
noncomputable def tapD (Coin : Type) : Claims.Sampling.Deploy Coin Unit Unit :=
  { leakD Coin with commit := accCm }

/-- **The network certifier's openings register `obs0`**, under every opening passing: they open the input and the
egress slot's presence gate, both `false`, so the step's row holds no frame and its ingress is `false`. -/
theorem tapOps_opens (H : Msg → Msg) :
    TapOpens H accCm notLay 1 (fun _ _ => [false]) notStmt tapOps [obs0] := by
  refine ⟨fun q _ => rfl, fun t ht => ?_⟩
  obtain rfl : t = 0 := by omega
  refine ⟨fun ℓ σ g hg => ?_, fun g hg => ?_, Nat.one_pos, fun ℓ => ?_, rfl⟩
  · simp [notLay, slotGates] at hg
    subst hg
    exact ⟨fun b hb => by simp [notStmt, notProg] at hb, ⟨1, false, []⟩, by simp [tapOps], rfl⟩
  · simp [notLay] at hg
    subst hg
    exact ⟨⟨0, false, []⟩, by simp [tapOps], rfl⟩
  · funext σ
    rfl

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- **No witness is in `RLeakTap`** at `tapD`, at one step at `δ = 0`, whatever the inputs `As` admits: the network
certifier registers `obs0` (`tapOps_opens`), which no possible faulty set accepts (`obs0_notAcc`). -/
theorem tapD_noRLeakTap (H : Msg → Msg) (As : Claims.Sampling.Stmt → (ℕ → Bool) → Set (ℕ → Bool)) (x : Msg)
    (w : Claims.Sampling.Wit) :
    ¬ RLeakTap H (tapD Coin).commit (tapD Coin).parse (tapD Coin).partition 1 1 1 (fun _ => notLay) (fun _ => 1) 0
      As 1 (fun _ => tapOps) x w := by
  rintro ⟨s, P, hs, hP, h⟩
  obtain rfl : notStmt = s := Option.some.inj hs
  obtain rfl : notPart = P := Option.some.inj hP
  exact obs0_notAcc _ (h (fun _ _ => [false]) [obs0] (tapOps_opens H) obs0_covered rfl (by simp [obs0]))

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem tapD_κ (x : Msg) : Claims.Sampling.κ (tapD Coin) 0 x < 1 / 3 :=
  sksD_κ (Coin := Coin) x

theorem tapD_bext (H : Msg → Msg) (decode : List (ℕ × Unit) → List Unit) (x : Msg) :
    (Bext H (tapD Coin).V (Claims.Sampling.E (tapD Coin) decode) Unit silentP x).Pr H = 0 := by
  simp only [Bext]
  rw [dite_eq_left (show 0 < (tapD Coin).V.k x from Nat.one_pos)]
  unfold Finder.Pr
  rw [Pr_congr fun s => iff_false_intro (by
    simp [ownLog_noHash _ _ _ (noHash_E (tapD Coin) (fun _ _ _ => rfl) _ _ _ _), findColl])]
  exact Pr_false

/-- With every opening passing at no call, `B_tap` checks the network certifier's openings at no call either. -/
theorem tapD_btap (H : Msg → Msg) (decode : List (ℕ × Unit) → List Unit) (x : Msg) :
    (Btap H (tapD Coin).V (Claims.Sampling.E (tapD Coin) decode) (tapD Coin).commit (tapD Coin).parse
      (fun _ => tapOps) Unit silentP x).Pr H = 0 := by
  unfold Finder.Pr
  rw [Pr_congr fun s => iff_false_intro (by
    simp only [Btap, ownLog_noHash _ _ _ (noHash_E (tapD Coin) (fun _ _ _ => rfl) _ _ _ _)]
    simp [tapLog, tapD, leakD, accCm, OC.eval, tapOps, findColl])]
  exact Pr_false

end Toy

/-- **`PeriodLeakAccepted`'s per-window bound fails at the accepting verifier** `tapD`, at every hash, at one step at
`δ = 0` with no input registered and the statement `[]` in scope (`leakD_scopeAcc`): no witness is in `RLeakTap`
(`tapD_noRLeakTap`), so the silent prover's run fails it with probability `1`, while `κ < 1/3` and the finders find
nothing. -/
theorem tapD_bound_fails (H : Msg → Msg) :
    ¬ PrBad H (tapD Unit).V
        (RLeakTap H (tapD Unit).commit (tapD Unit).parse (tapD Unit).partition 1 1 1 (fun _ => notLay) (fun _ => 1) 0
          (fun s => AsReg s.prog notLay 1 ∅) 1 (fun _ => tapOps))
        (Claims.Sampling.E (tapD Unit) fun _ => [()]) [] silentP ≤
      Claims.Sampling.κ (tapD Unit) 0 [] +
          Real.sqrt (Claims.Sampling.γ (tapD Unit) [] * PrB H (tapD Unit).V silentP []) +
        (Bext H (tapD Unit).V (Claims.Sampling.E (tapD Unit) fun _ => [()]) Unit silentP []).Pr H +
        (Btap H (tapD Unit).V (Claims.Sampling.E (tapD Unit) fun _ => [()]) (tapD Unit).commit (tapD Unit).parse
          (fun _ => tapOps) Unit silentP []).Pr H := by
  intro h
  have hp : PrBad H (tapD Unit).V
      (RLeakTap H (tapD Unit).commit (tapD Unit).parse (tapD Unit).partition 1 1 1 (fun _ => notLay) (fun _ => 1) 0
        (fun s => AsReg s.prog notLay 1 ∅) 1 (fun _ => tapOps))
      (Claims.Sampling.E (tapD Unit) fun _ => [()]) [] silentP = 1 :=
    prBad_accV H _ _ [] silentP (tapD_noRLeakTap H _ [])
  have hb : PrB H (tapD Unit).V silentP [] = 0 := prB_accV H silentP []
  simp only [hp, hb, mul_zero, Real.sqrt_zero, add_zero, tapD_bext, tapD_btap] at h
  linarith [tapD_κ (Coin := Unit) []]

/-! ## The horizon -/

/-- **`HorizonLeakAccepted`'s bound fails at an audit that always accepts**, at `δ = 1/2` and `e = e' = 0`: one window
whose faulty set holds its one unit, which `Fmax = 1` allows none of, so the good event fails on every coin. -/
theorem horizon_bound_fails :
    ¬ Pr (fun _ : Fin 1 → Unit => (∀ _ : Fin 1, True) ∧
        ¬ HorizonGood (fun _ => trivLay) (fun _ => 0) (fun _ => ∅) 1 (fun _ : Fin 1 => notStmt) (fun _ => notPart)
          (fun _ _ => false) (fun _ _ _ => ([] : List Bool)) 1 (fun _ => {0}) (fun _ => ([] : List (Obs 1 1 1)))) ≤
      (1 / 2 : ℝ) + ∑ _ : Fin 1, (0 : ℝ) + ∑ _ : Fin 1, (0 : ℝ) := by
  simp [Pr, HorizonGood]
  norm_num

/-! ## The Python network certifier's fixture -/

/-- A frame of the fixture's three connections, as a number below `9`: its connection's ordinal times three, plus its
kind. -/
def fxCode : Fin 3 × CellView Unit → ℕ
  | (c, .data _) => 3 * c.val
  | (c, .pad) => 3 * c.val + 1
  | (c, .fail) => 3 * c.val + 2

/-- The frame a number below `9` is. -/
def fxFrame (n : ℕ) : Option (Fin 3 × CellView Unit) :=
  if h : n < 9 then
    some (⟨n / 3, by omega⟩, if n % 3 = 0 then .data () else if n % 3 = 1 then .pad else .fail)
  else none

/-- **A rows format of the fixture `fxQueue`**: its three connections in order, a frame encoded as that many `true`
bits, one egress link and one status, ten buckets a window. -/
def fxRows : RowsFormat (Fin 3) Unit 1 1 where
  conns := List.finRange 3
  enc x := List.replicate (fxCode x) true
  dec l := fxFrame l.length
  link := 0
  T := 10
  jitter _ _ _ := 0
  status _ _ _ _ := .complete

theorem fxRows_wf : fxRows.Wf := by
  refine ⟨List.nodup_finRange 3, List.mem_finRange, ?_⟩
  rintro ⟨c, ⟨⟨⟩⟩ | _ | _⟩ <;> fin_cases c <;> rfl

theorem fxQueue_bucket (c : Fin 3) (j : ℕ) (hj : j < fxQueueSched.cells c) : fxQueueSched.bucket c j < 10 := by
  fin_cases c <;> simp [fxQueueSched, WardenSchedule.bucket] at hj ⊢ <;> omega

/-- **`fxQueue`'s schedule fits the format** over one window of four slots: its cells fall in buckets `4 … 9`, and no
bucket holds more than four frames. -/
theorem fxQueue_fits : (RowsFormat.tapRecord fxRows fxRows_wf 4 1).fits fxQueueSched := by
  refine ⟨fun c j hj => by simpa [fxRows] using fxQueue_bucket c j hj, fun b => ?_⟩
  rcases Nat.lt_or_ge b 10 with hb | hb
  · interval_cases b <;> decide
  · have h0 : ∀ c, laneLoad fxQueueSched c b = 0 := by
      intro c
      have hadm : ¬ (fxQueueSched.adm c ≤ b ∧ b < fxQueueSched.adm c + fxQueueSched.D) := by
        fin_cases c <;> simp [fxQueueSched] <;> omega
      simp only [laneLoad, hadm, ↓reduceIte, zero_add, List.length_eq_zero_iff, List.filter_eq_nil_iff,
        List.mem_range, beq_iff_eq]
      intro j hj
      have := fxQueue_bucket c j hj
      omega
    simp [fxRows, h0]

/-! ## The shipped schedule at one bucket -/

/-- One clock-sync value, `δ = 0` and `ρ = 1`. -/
def syncOne : SyncSet where
  deltas := {0}
  rhos := {1}
  deltas_nonempty := ⟨0, Finset.mem_singleton_self 0⟩
  rhos_nonempty := ⟨1, Finset.mem_singleton_self 1⟩
  rhos_pos := fun ρ h => by rw [Finset.mem_singleton.1 h]; decide

/-- `syncOne`'s one value. -/
def syncOne0 : syncOne.Elem := (⟨0, Finset.mem_singleton_self 0⟩, ⟨1, Finset.mem_singleton_self 1⟩)

/-- **The constant-rate Program at one bucket**: one window of one bucket, one link of one slot, the sync set
`syncOne`, and one frame `[true]` anchored at bucket 0. -/
def fxRate : Program 1 1 Unit (Unit × (Fin 1 → syncOne.Elem)) :=
  constantRate 1 1 syncOne fun _ _ _ => [⟨[true], 0, 0⟩]

/-- `fxRate` with its clock fixed at `syncOne0`: a Program whose clock-sync advice is `Unit`. -/
def fxFixed : Program 1 1 Unit (Unit × (Fin 1 → Unit)) where
  rows t ins a ℓ := fxRate.rows t ins (a.1, fun _ => syncOne0) ℓ

/-- The operator, which sends nothing. -/
def fxOp : ℕ → List (Obs 1 1 1) → Unit := fun _ _ => ()

/-- **The honest certifier's one step at `fxRate`**: the Program's row, jitter class 0, its one status complete. -/
def fxHonest : List (Obs 1 1 1) :=
  [⟨fxRate.rows 0 [()] ((), fun _ => syncOne0), 0, fun _ => .complete⟩]

/-- The observation sequences of length `n` with jitter below `D` that the acceptance `acc` passes: `AccSeqs` with
`acc` in place of the Program's acceptance (`accSeqs_eq_at`). -/
def AccSeqsAt {L K S : ℕ} (acc : List (Obs L K S) → Prop) (n D : ℕ) : Set (List (Obs L K S)) :=
  {o | o.length = n ∧ (∀ x ∈ o, x.jitter < D) ∧ acc o}

/-- The verifier that accepts everything. -/
def acceptAll {L K S : ℕ} (_ : List (Obs L K S)) : Prop := True

theorem accSeqs_eq_at {L K S : ℕ} {In Adv : Type} (P : Program L S In Adv) (op : ℕ → List (Obs L K S) → In)
    (n D : ℕ) : AccSeqs P op n D = AccSeqsAt (AcceptedUpTo P op n) n D :=
  rfl

/-- **`fxHonest` is accepted** at `fxRate` and at `fxFixed`, under the advice `((), syncOne0)`. -/
theorem fxHonest_accepted : fxHonest ∈ AccSeqs fxRate fxOp 1 1 ∧ fxHonest ∈ AccSeqs fxFixed fxOp 1 1 := by
  have hj : ∀ x ∈ fxHonest, x.jitter < 1 := by
    intro x hx
    simp [fxHonest] at hx
    simp [hx]
  have hl : 0 < fxHonest.length := by simp [fxHonest]
  refine ⟨⟨rfl, hj, ((), fun _ => syncOne0), fun t ht => ?_⟩, ⟨rfl, hj, ((), fun _ => ()), fun t ht => ?_⟩⟩ <;>
  · obtain rfl : t = 0 := by omega
    exact ⟨hl, rfl, fun _ => rfl⟩

/-- **At the verifier that accepts everything, one step's sequences are infinitely many**: one link of one slot
carries any payload. -/
theorem accSeqsAt_acceptAll_infinite : (AccSeqsAt (acceptAll (L := 1) (K := 1) (S := 1)) 1 1).Infinite := by
  refine Set.infinite_of_injective_forall_mem
    (f := fun k : ℕ => [(⟨fun _ _ => some (List.replicate k true), 0, fun _ => .complete⟩ : Obs 1 1 1)]) ?_ ?_
  · intro a b hab
    have := congrArg (fun o : List (Obs 1 1 1) => o.head?.map fun x => x.rows 0 0) hab
    simpa using this
  · intro k
    simp [AccSeqsAt, acceptAll]

/-- No finite count bounds `AccSeqsAt acceptAll` at one step. -/
theorem accSeqsAt_acceptAll_not_le (N : ℕ) : ¬ (AccSeqsAt (acceptAll (L := 1) (K := 1) (S := 1)) 1 1).encard ≤ N := by
  rw [accSeqsAt_acceptAll_infinite.encard_eq]
  simp

/-- **The honest certifier at `fxRate`**: at each step it records and puts on the wire `fxRate`'s row under the
advice `((), syncOne0)`, jitter class 0 and its one status complete, whatever the sender does. -/
def fxHonestI : Interaction 1 1 1 Unit Unit Unit (Obs 1 1 1) where
  certifier :=
    { obs := fun k xs _ => ⟨fxRate.rows k xs ((), fun _ => syncOne0), 0, fun _ => .complete⟩
      wire := fun k xs _ => ⟨fxRate.rows k xs ((), fun _ => syncOne0), 0, fun _ => .complete⟩
      fault := fun _ => False }
  op := fxOp
  snd := fun _ _ _ => ()

/-- **A sender whose secret shows**: its secret `z` is its action, and the certifier records and puts on the wire one
frame of `z` bits, jitter class 0, its one status complete. -/
def fxLeakI : Interaction 1 1 1 ℕ Unit ℕ (Obs 1 1 1) where
  certifier :=
    { obs := fun _ _ as => ⟨fun _ _ => some (List.replicate (as.getLast?.getD 0) true), 0, fun _ => .complete⟩
      wire := fun _ _ as => ⟨fun _ _ => some (List.replicate (as.getLast?.getD 0) true), 0, fun _ => .complete⟩
      fault := fun _ => False }
  op := fxOp
  snd := fun z _ _ => z

/-- `fxLeakI`'s decoder: the first step's frame's length. -/
def fxLeakDec (w : List (Obs 1 1 1)) : ℕ := ((w.head?.bind fun o => o.rows 0 0).map List.length).getD 0

/-- Both instances meet W2, the wire a function of the record (the identity), and J at `D = 1`. -/
theorem fx_wire : Regenerates fxHonestI.certifier (fun _ o => o) ∧ JitterBounded fxHonestI.certifier 1 ∧
    Regenerates fxLeakI.certifier (fun _ o => o) ∧ JitterBounded fxLeakI.certifier 1 :=
  ⟨fun _ _ _ => rfl, fun _ _ _ => Nat.one_pos, fun _ _ _ => rfl, fun _ _ _ => Nat.one_pos⟩

/-- `fxHonestI`'s first step is `fxHonest`. -/
theorem fxHonestI_obsRun : fxHonestI.obsRun () 1 = fxHonest := rfl

/-- **At the verifier that accepts everything, every secret of `fxLeakI` is accepted and decoded.** -/
theorem fxLeak_decodes (z : ℕ) : acceptAll (fxLeakI.obsRun z 1) ∧ fxLeakDec (fxLeakI.wireRun z 1) = z := by
  refine ⟨trivial, ?_⟩
  simp [fxLeakDec, fxLeakI, Interaction.wireRun, Interaction.trace]

/-- No finite count bounds the secrets of `fxLeakI` that the verifier that accepts everything passes and its decoder
recovers. -/
theorem fxLeak_not_le (N : ℕ) :
    ¬ {z : ℕ | acceptAll (fxLeakI.obsRun z 1) ∧ fxLeakDec (fxLeakI.wireRun z 1) = z}.encard ≤ N := by
  have : {z : ℕ | acceptAll (fxLeakI.obsRun z 1) ∧ fxLeakDec (fxLeakI.wireRun z 1) = z} = Set.univ :=
    Set.eq_univ_of_forall fun z => fxLeak_decodes z
  rw [this, Set.encard_univ, ENat.card_eq_top_of_infinite]
  simp

end NetTiming.Witness
