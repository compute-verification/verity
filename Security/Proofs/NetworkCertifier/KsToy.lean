import Proofs.NetworkCertifier.WitnessToys
import Proofs.NetworkCertifier.KSWired
import Proofs.Flock.Claims.Sampling.KS
import Proofs.Flock.Claims.ERep
import Definitions.NetworkCertifier.Period

/-!
# A deployment where `Claims.Sampling.KS` holds, and why `PeriodLeakRegisteredWf` reads `RegOpensAt` and `RowsFormat.Wf`

`ksNotD` is `leakD` (the negation `notStmt`, its verifier accepting) with a commitment that passes one value at each
position, `false` at gate `0` and `true` at gate `1` (`notCm`), one proof block checked on every run, and a session whose
Flock witness reveals both gates' values. `Claims.Sampling.KS` holds there (`ksNotD_KS`, by `Claims.Sampling.KS_of`):
the commitment binds with no call, every decoder within Flock's radius lists the one trace, which Flock's relation at
`Claims.Witness.Flock.inStmt` holds, so every session's extraction finds a witness, whose openings make the block
correct; the block is checked on every run, so the only possible faulty set is empty.

Without `RegOpensAt`, the first part of `PeriodLeakRegisteredWf` is false (`not_windowLeakAtAnyOps`): at the layout
`freeLay`, where the input is free and the negation is the egress slot's presence gate, a registration `badReg` that
opens the input `true`, which `notCm` refuses, pins the input at `true`, so the circuit holds no frame, while the
network certifier's openings `frameTops`, which pass, register a frame (`obs1`). No witness is then in `RLeakTap`, the
silent prover is accepted, and its bound fails, at every hash, with `KSBy` at the wired check (`freeLay` has no
wires), the decoder's radius, binding and the scope all holding. The registration `goodReg`, which opens the input
`false`, meets `RegOpensAt` (`ksNotD_regOpens`).

`WitnessToys`' rows format `blindRows` encodes every frame as the empty list (`blindRows_not_wf`): on the schedule
`oneSched`, which fits it, the record of every serving `z` is `obs1` (`blindRun_obs`), which the circuit accepts at
`goodReg` (`frameTops_acc`), while the client's cell tells the servings apart (`cellDec_run`).
-/

namespace NetTiming.Witness

open Claims Claims.Witness Claims.Witness.Sampling
open Claims.Sampling hiding NoHash

/-! ## Rerun programs with no call of their own -/

theorem noHash_waitAcc {R : Type} (k : ℕ) : ∀ (rs : List R) (acc : List Seen), NoHash (waitAcc k rs acc)
  | [], _ => trivial
  | r :: rs, acc => by
    unfold waitAcc
    split_ifs
    · trivial
    · exact fun v => noHash_waitAcc k rs _

theorem noHash_firstWait {R W Op : Type} (k : ℕ) (opens : Seen → List Op) (asm : List Op → Option W)
    (chk : W → Bool) : ∀ rss : List (List R), NoHash (firstWait k opens asm chk rss)
  | [] => trivial
  | rs :: rss => by
    unfold firstWait
    refine noHash_bind _ _ (noHash_waitAcc k rs []) fun vs => ?_
    split
    · split_ifs
      · trivial
      · exact noHash_firstWait k opens asm chk rss
    · exact noHash_firstWait k opens asm chk rss

/-- **A wait list that assembles one witness everywhere finds it**: when every candidate assembles to `w` and `w`
passes the check, the first wait returns `w`. -/
theorem firstWait_eval {R W Op : Type} (H : Msg → Msg) (run : R → Trace) (k : ℕ) (opens : Seen → List Op)
    (asm : List Op → Option W) (chk : W → Bool) {w : W} (ha : ∀ l, asm l = some w) (hc : chk w = true) :
    ∀ {rss : List (List R)}, rss ≠ [] → ((firstWait k opens asm chk rss).eval H run).1 = some w
  | [], h => absurd rfl h
  | rs :: rss, _ => by
    simp only [firstWait]
    rw [(RP.eval_bind H run _ _).1]
    simp [ha, hc, RP.eval]

/-! ## The deployment -/

/-- **A commitment that passes one value at each position**, with no call: `false` at gate `0`, `true` past it. -/
def notCm : CommitmentScheme Bool := ⟨fun _ o => .ret (decide (o.val = decide (0 < o.pos))), fun _ _ _ => []⟩

/-- `notCm` binds: two openings that pass at one position have one value. -/
theorem notCm_binds (H : Msg → Msg) : notCm.Binds H := by
  intro roots o o' hp hv h h'
  simp only [CommitmentScheme.Opens, notCm, OC.eval, decide_eq_true_eq] at h h'
  exact absurd (by rw [h, h', hp] : o.val = o'.val) hv

/-- What the session's Flock witness reveals: the input `false` and its negation `true`, under the statement's roots. -/
def notReveal : List (List Msg × Opening Bool) := [([], ⟨0, false, []⟩), ([], ⟨1, true, []⟩)]

/-- Step `t`'s egress slot's presence gate is the negation, and no step has an ingress, so the input is free. -/
def freeLay : BoundaryGates 1 1 := { egress := fun _ _ _ => (1, []), ingress := fun _ => [] }

theorem freeLay_fits : freeLay.Fits notProg 1 := by
  refine ⟨fun t _ ℓ σ g hg => ?_, fun t _ g hg => ?_, fun t _ t' _ htt' => absurd htt' (by omega),
    wiresFit_none rfl _ _⟩
  · simp [freeLay, slotGates] at hg
    subst hg
    decide
  · simp [freeLay] at hg

/-- **`leakD` with a session that finds a witness**: the commitment `notCm`, the one proof block checked on every run,
its session at Flock's toy record `compD`, and its witness revealing `notReveal`. -/
noncomputable def ksNotD (Coin : Type) : Claims.Sampling.Deploy Coin Unit Unit :=
  { leakD Coin with
    commit := notCm
    flock := Claims.Witness.Flock.compD Coin
    checked := fun _ _ => [0]
    session := fun _ _ v => some ([], v)
    reveal := fun _ _ _ _ => notReveal }

/-- The network certifier's openings: the negation `true`, a frame on the link. -/
def frameTops : List (Opening Bool) := [⟨1, true, []⟩]

/-- A registration of the input at `true`, which `notCm` refuses. -/
def badReg : List (Opening Bool) := [⟨0, true, []⟩]

/-- A registration of the input at `false`, which `notCm` passes. -/
def goodReg : List (Opening Bool) := [⟨0, false, []⟩]

/-! ## Flock's toy record finds a witness -/

/-- At a decoder that lists the one trace on every word, `compD` assembles the empty witness from any openings. -/
theorem compD_assemble {Coin : Type} (decode : List (ℕ × Unit) → List Unit) (hdec : ∀ os, () ∈ decode os) (xS : Msg)
    (l : List (ℕ → List (ℕ × Unit))) :
    Claims.Flock.assemble (Claims.Witness.Flock.compD Coin).parse (Claims.Witness.Flock.compD Coin).table decode
      (Claims.Witness.Flock.compD Coin).read xS l = some ⟨[]⟩ := by
  have hrel : ∀ w, decide (Claims.Flock.RelS Claims.Witness.Flock.inStmt w) = true := fun w =>
    decide_eq_true (Claims.Witness.Flock.inStmt_relS w)
  have hf : ∀ os, (decode os).find? (fun M =>
      decide (Claims.Flock.RelS Claims.Witness.Flock.inStmt ((Claims.Witness.Flock.compD Coin).read [M]))) = some () :=
    fun os => by
      cases h : (decode os).find? (fun M =>
          decide (Claims.Flock.RelS Claims.Witness.Flock.inStmt ((Claims.Witness.Flock.compD Coin).read [M]))) with
      | none => exact absurd (hrel _) (List.find?_eq_none.1 h () (hdec os))
      | some M => rfl
  show (((List.range 1).mapM fun j => (decode (l.flatMap fun o => o j)).find? fun M =>
      decide (Claims.Flock.RelS Claims.Witness.Flock.inStmt ((Claims.Witness.Flock.compD Coin).read [M]))).map
      (Claims.Witness.Flock.compD Coin).read) = some ⟨[]⟩
  simp only [hf]
  rfl

theorem compD_check {Coin : Type} (xS : Msg) (w : Claims.Flock.Wit) :
    Claims.Flock.check (Claims.Witness.Flock.compD Coin).parse xS w = true := by
  simp only [Claims.Flock.check, Claims.Witness.Flock.compD, decide_eq_true_eq]
  exact Claims.Witness.Flock.inStmt_relS w

/-! ## `KS` at `ksNotD` -/

section Toy

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- A decoder within Flock's radius at `compD` lists the one trace on every word: its codeword length is `0`. -/
theorem mem_of_listWithin {decode : List (ℕ × Unit) → List Unit}
    (hdec : Claims.Flock.ListWithin (ksNotD Coin).flock.len (ksNotD Coin).flock.enc (ksNotD Coin).flock.θ decode)
    (os : List (ℕ × Unit)) : () ∈ decode os :=
  hdec os () (by
    show (1 - (0 : ℝ)) * ((0 : ℕ) : ℝ) ≤ _
    rw [Nat.cast_zero, mul_zero]
    exact Nat.cast_nonneg _)

/-- **Every checked unit's extraction finds the empty witness**: its session is shown on every run, and Flock's toy
record assembles the empty witness, which its relation holds, from the first wait. -/
theorem ksNotD_unitWit (H : Msg → Msg) (decode : List (ℕ × Unit) → List Unit) (hdec : ∀ os, () ∈ decode os)
    (x : Msg) (run' : (Fin ((ksNotD Coin).V.k x) → Coin) → Trace) (u : ℕ) (r : Fin ((ksNotD Coin).V.k x) → Coin)
    (ρ : (ksNotD Coin).UnitCoins x) :
    Claims.Sampling.unitWit H (ksNotD Coin) decode x run' u r ρ = some ⟨[]⟩ := by
  have hs : ∀ v, (ksNotD Coin).session x u v = some ([], v) := fun _ => rfl
  simp only [Claims.Sampling.unitWit, hs]
  exact firstWait_eval H run' _ _ _ _ (compD_assemble (Coin := Coin) decode hdec []) (compD_check (Coin := Coin) [] _)
    (List.ne_nil_of_length_pos (by rw [List.length_map, List.length_finRange]; exact Nat.succ_pos 7))

theorem ksNotD_sessionKS (H : Msg → Msg) : Claims.Sampling.SessionKS H (ksNotD Coin) := by
  intro decode hdec x Ω _ _ P
  refine ⟨fun _ _ => 0, fun _ _ => le_rfl, ?_, fun ω r sel => ?_⟩
  · show Ex (fun _ => (0 : ℝ)) ≤ _
    rw [Ex_const]
    exact ge_of_eq (prB_accV (Coin := Coin) H P x)
  · rw [Pr_congr fun q => iff_false_intro (by
      rintro ⟨-, i, -, u, -, hw⟩
      exact absurd ((ksNotD_unitWit H decode (mem_of_listWithin hdec) x _ u _ _).symm.trans hw)
        (Option.some_ne_none _)), Pr_false]
    exact add_nonneg (sksD_κF_nonneg (Coin := Coin) x) (Real.sqrt_nonneg _)

private theorem ksRefIn_boundary {C : Prog} {u : List ℕ} {g : ℕ} (hg : C.Boundary u g = true)
    (val : ℕ → Option Bool) (ι : ℕ → Bool) : C.refIn u val ι g = val g := by
  simp [Prog.refIn, hg]

private theorem ksRunIn_input {C : Prog} {i : ℕ} (hi : C.gates[i]? = some .input) (ref : ℕ → Option Bool)
    (u : List ℕ) : C.runIn ref u i = ref i := by
  rw [Prog.runIn, hi]
  rfl

private theorem ksRunIn_not {C : Prog} {i j : ℕ} (hi : C.gates[i]? = some .input) (hj : C.gates[j]? = some (.not i))
    (hij : i < j) (ref : ℕ → Option Bool) : C.runIn ref [i, j] j = (ref i).map (!·) := by
  rw [Prog.runIn, hj]
  simp [BGate.val, hij, ksRunIn_input hi]

/-- The negation's block `{0, 1}` is correct on the values the session's witness reveals, `false` and `true`. -/
theorem notProg_correct (val : ℕ → Option Bool) (ι : ℕ → Bool) (h0 : val 0 = some false) (h1 : val 1 = some true) :
    notProg.Correct (notProg.refIn [0, 1] val ι) [0, 1] := by
  have hi : notProg.gates[0]? = some .input := rfl
  have hj : notProg.gates[1]? = some (.not 0) := rfl
  have r0 : notProg.refIn [0, 1] val ι 0 = some false := (ksRefIn_boundary (by decide) _ _).trans h0
  have r1 : notProg.refIn [0, 1] val ι 1 = some true := (ksRefIn_boundary (by decide) _ _).trans h1
  intro g hg
  rcases List.mem_cons.1 hg with rfl | hg
  · refine ⟨fun x hx k hk _ => ?_, fun _ => ⟨false, r0, by rw [ksRunIn_input hi, r0]⟩⟩
    rw [hi] at hx
    obtain rfl : BGate.input = x := Option.some.inj hx
    simp [BGate.reads] at hk
  · obtain rfl : g = 1 := List.mem_singleton.1 hg
    refine ⟨fun x hx k hk hku => ?_, fun _ => ⟨true, r1, by simp [ksRunIn_not hi hj Nat.one_pos, r0]⟩⟩
    rw [hj] at hx
    obtain rfl : BGate.not 0 = x := Option.some.inj hx
    simp only [BGate.reads, List.mem_singleton] at hk
    subst hk
    exact absurd List.mem_cons_self hku

/-- **The session's witness makes its block correct**: it reveals the input `false` and the negation `true`, both of
which `notCm` passes under the statement's roots. -/
theorem ksNotD_sessionOpens (H : Msg → Msg) :
    Claims.Sampling.SessionOpens H (ksNotD Coin) (fun _ _ _ => []) := by
  intro x s Pt Ω P ω r u xS vS wF hs hP _ _ hu _ _
  obtain rfl : notStmt = s := Option.some.inj hs
  obtain rfl : notPart = Pt := Option.some.inj hP
  obtain ⟨p, hp⟩ := hu
  have hp0 : p.val = 0 := Nat.lt_one_iff.1 p.isLt
  rw [hp0] at hp
  have hu0 : u = 0 := (Option.some.inj hp).symm
  subst hu0
  refine ⟨[0, 1], rfl, fun val ι hval => ?_⟩
  have h0 := (hval ([], ⟨0, false, []⟩) (by simp [ksNotD, notReveal]) rfl).1 rfl
  have h1 := (hval ([], ⟨1, true, []⟩) (by simp [ksNotD, notReveal]) rfl).1 rfl
  exact notProg_correct val ι h0 h1

theorem notPa (B : Finset ℕ) :
    notStmt.design.Pa notPart B = if Claims.Sampling.window 0 1 B = ∅ then 1 else 0 := by
  simp [Design.Pa, notStmt, notPart, classesPa, OneStage.Pa]

/-- **The draw law**: the one block is checked on every run, and the design draws it `all`. -/
theorem ksNotD_drawLaw : Claims.Sampling.DrawLaw (ksNotD Coin) := by
  intro x s Pt hs hP _ B hB
  obtain rfl : notStmt = s := Option.some.inj hs
  obtain rfl : notPart = Pt := Option.some.inj hP
  have hPa : 0 ≤ notStmt.design.Pa notPart B := by
    rw [notPa]
    split_ifs <;> norm_num
  refine ⟨fun _ => notStmt.design.Pa notPart B, fun _ => hPa, fun r => ?_, (Ex_const _).le⟩
  show Pr _ ≤ notStmt.design.Pa notPart B
  by_cases h0 : 0 ∈ B
  · rw [Pr_congr fun d => iff_false_intro (by intro h; exact h 0 h0 ⟨0, Nat.one_pos⟩ rfl), Pr_false]
    exact hPa
  · have hB0 : B = ∅ := Finset.eq_empty_of_forall_notMem fun v hv => by
      have hv1 : v < 1 := Finset.mem_range.1 (hB hv)
      obtain rfl : v = 0 := by omega
      exact h0 hv
    subst hB0
    rw [notPa, if_pos (by simp [Claims.Sampling.window])]
    exact Pr_le_one _

theorem ksNotD_refuses (H : Msg → Msg) : Claims.Sampling.Refuses H (ksNotD Coin) :=
  fun _ _ _ _ _ _ => ⟨notStmt, notPart, rfl, rfl, inScope_not⟩

/-- **`Claims.Sampling.KS` holds at `ksNotD`**, at every hash (`Claims.Sampling.KS_of`). -/
theorem ksNotD_KS (H : Msg → Msg) : Claims.Sampling.KS H (ksNotD Coin) :=
  Claims.Sampling.KS_of (fun _ _ _ => []) (notCm_binds H) (ksNotD_sessionKS H) (ksNotD_sessionOpens H)
    (fun _ _ _ _ _ _ _ => rfl) ksNotD_drawLaw (ksNotD_refuses H) (fun _ => le_rfl)

/-! ## The registration -/

theorem ksNotD_κ (x : Msg) : Claims.Sampling.κ (ksNotD Coin) 0 x < 1 / 3 :=
  sksD_κ (Coin := Coin) x

theorem ksNotD_scopeAcc (x : Msg) : ScopeAcc (ksNotD Coin) (fun _ => freeLay) (fun _ => 1) x :=
  ⟨⟨notStmt, notPart, rfl, rfl, inScope_not⟩, notStmt, rfl, freeLay_fits⟩

/-- `badReg` doesn't open the registered input under `notCm`. -/
theorem ksNotD_not_regOpens (H : Msg → Msg) (x : Msg) :
    ¬ RegOpensAt H (ksNotD Coin) (fun _ => {0}) (fun _ => badReg) x :=
  fun h => Bool.false_ne_true ((h notStmt rfl).1 ⟨0, true, []⟩ (List.mem_singleton.2 rfl))

/-- `goodReg` opens the registered input under `notCm`. -/
theorem ksNotD_regOpens (H : Msg → Msg) (x : Msg) :
    RegOpensAt H (ksNotD Coin) (fun _ => {0}) (fun _ => goodReg) x := fun s hs => by
  obtain rfl : notStmt = s := Option.some.inj hs
  refine ⟨fun o ho => ?_, fun g hg => ?_⟩
  · simp only [goodReg, List.mem_singleton] at ho
    subst ho
    rfl
  · obtain rfl := Finset.mem_singleton.1 hg
    exact ⟨⟨0, false, []⟩, List.mem_singleton.2 rfl, rfl⟩

/-! ## The network certifier's openings register a frame -/

theorem frameTops_opens (H : Msg → Msg) :
    TapOpens H notCm freeLay 1 (fun _ _ => []) notStmt frameTops [obs1] := by
  refine ⟨fun q hq => ?_, fun t ht => ?_⟩
  · simp only [frameTops, List.mem_singleton] at hq
    subst hq
    rfl
  · obtain rfl : t = 0 := by omega
    refine ⟨fun ℓ σ g hg => ?_, fun g hg => by simp [freeLay] at hg, Nat.one_pos, fun ℓ => ?_, rfl⟩
    · simp [freeLay, slotGates] at hg
      subst hg
      exact ⟨fun b hb => by simp [notStmt, notProg] at hb, ⟨1, true, []⟩, by simp [frameTops], rfl⟩
    · funext σ
      rfl

theorem obs1_covered : Covered 1 [obs1] := by
  intro t ht _ κ
  obtain rfl : t = 0 := by omega
  rfl

theorem zero_free : 0 ∈ freeInputs notProg freeLay 1 := by
  decide

/-- With nothing held and the input `true`, the circuit computes the negation `false`. -/
theorem notProg_val_true (inp : ℕ → Bool) (h0 : inp 0 = true) : valOver notProg inp (fun _ => none) 1 = false := by
  have hg0 : notProg.gates[0]? = some .input := rfl
  have hg1 : notProg.gates[1]? = some (.not 0) := rfl
  have hv0 : valOver notProg inp (fun _ => none) 0 = true := by
    rw [valOver]
    simp [hg0, BGate.val, h0]
  rw [valOver]
  simp [hg1, BGate.val, hv0]

/-- **`obs1` is accepted under no possible faulty set with the input pinned at `true`**: no gate is held, so the
negation of `true` holds no frame, where `obs1` has one. -/
theorem obs1_notAcc_bad (δ : ℝ) :
    [obs1] ∉ AccSeqsSampled notProg notPart notStmt.design freeLay 1 (AsReg notProg freeLay 1 {0} (regVal badReg)) δ
      (fun _ _ => []) 1 := by
  intro h
  obtain ⟨-, -, F, hF, a, ha, free, hfree, hacc⟩ := h
  obtain rfl := notFaulty_empty hF
  have hnone : free = fun _ => none := funext fun g => by
    cases hg : free g with
    | none => rfl
    | some b =>
      obtain ⟨v, hv, -⟩ := hfree g b hg
      simp at hv
  subst hnone
  obtain ⟨ha1, -⟩ := ha
  have ha0 : a 0 = true := ha1 0 (Finset.mem_singleton_self 0) zero_free
  obtain ⟨_, hrows, -⟩ := hacc 0 Nat.one_pos
  have hin : freeLay.inputOf 1 [[]] a 0 = true := ha0
  have h1 : (some [] : Option (List Bool)) =
      rowOf (valOver notProg (freeLay.inputOf 1 [[]] a) (fun _ => none)) (freeLay.egress 0 0) 0 := by
    refine (congrFun (congrFun hrows 0) 0).trans ?_
    show rowOf (valOver notProg (resolveWires notProg freeLay.wires _ (fun _ => none) notProg.gates.length)
      (fun _ => none)) (freeLay.egress 0 0) 0 = _
    rw [resolveWires_nil (wires := freeLay.wires) rfl]
    rfl
  have hrow : rowOf (valOver notProg (freeLay.inputOf 1 [[]] a) (fun _ => none)) (freeLay.egress 0 0) 0 =
      if valOver notProg (freeLay.inputOf 1 [[]] a) (fun _ => none) 1 then
        some (([] : List ℕ).map (valOver notProg (freeLay.inputOf 1 [[]] a) (fun _ => none)))
      else none := rfl
  rw [hrow, notProg_val_true _ hin] at h1
  simp at h1

/-- **No witness is in `RLeakTap`** at `ksNotD` with the input pinned at `badReg`'s value, at every `δ`: the network
certifier registers `obs1` (`frameTops_opens`), which no possible faulty set accepts (`obs1_notAcc_bad`). -/
theorem ksNotD_noRLeakTap (H : Msg → Msg) (δ : ℝ) (x : Msg) (w : Claims.Sampling.Wit) :
    ¬ RLeakTap H (ksNotD Coin).commit (ksNotD Coin).parse (ksNotD Coin).partition 1 1 1 (fun _ => freeLay)
      (fun _ => 1) δ (fun s _ => AsReg s.prog freeLay 1 {0} (regVal badReg)) 1 (fun _ => frameTops) x w := by
  rintro ⟨s, P, hs, hP, h⟩
  obtain rfl : notStmt = s := Option.some.inj hs
  obtain rfl : notPart = P := Option.some.inj hP
  exact obs1_notAcc_bad δ (h (fun _ _ => []) [obs1] (frameTops_opens H) obs1_covered rfl (by simp [obs1]))

/-! ## The finders find nothing -/

/-- At a commitment whose openings make no call, checking what a witness reveals makes no call. -/
theorem noHash_checkRevealed {R : Type} (Cm : CommitmentScheme Bool)
    (hCm : ∀ roots o, ∃ b, Cm.opens roots o = .ret b) (l : List (List Msg × Opening Bool)) :
    NoHash (checkRevealed (R := R) Cm l) := by
  refine noHash_flatMapRP _ (fun p => ?_) l
  obtain ⟨b, hb⟩ := hCm p.1 p.2
  show NoHash ((ofOC (Cm.opens p.1 p.2)).bind _)
  rw [hb]
  exact noHash_bind _ _ trivial fun _ => trivial

/-- At a commitment whose openings make no call, a checked unit's extraction makes no call of its own. -/
theorem noHash_unitExt {Col T : Type} [DecidableEq Col] [Zero Col] (D : Claims.Sampling.Deploy Coin Col T)
    (hCm : ∀ roots o, ∃ b, D.commit.opens roots o = .ret b) (decode : List (ℕ × Col) → List T) (x : Msg)
    (s : Claims.Sampling.AuditStatement) (u : ℕ) (r : Fin (D.V.k x) → Coin) (v : Seen) (ρ : D.UnitCoins x) :
    NoHash (unitExt D decode x s u r v ρ) := by
  unfold unitExt
  split
  · trivial
  · -- The wait's coin lists are generalized first: unifying against them unfolds `List.finRange`.
    generalize ((List.finRange (D.innerWaits x)).map fun i =>
      (List.finRange (D.innerCap x)).map fun j => forkOn (D.rounds x u) r (ρ i j)) = rss
    refine noHash_bind _ _ (noHash_firstWait _ _ _ _ rss) fun o => ?_
    cases o with
    | none => trivial
    | some wF => exact noHash_checkRevealed _ hCm _

/-- As `noHash_E`, at a deployment whose unit extractions make no call of their own. -/
theorem noHash_E_of {Col T : Type} [DecidableEq Col] [Zero Col] (D : Claims.Sampling.Deploy Coin Col T)
    (decode : List (ℕ × Col) → List T)
    (hu : ∀ x s u r v ρ, NoHash (unitExt D decode x s u r v ρ)) (x : Msg) (r₀ : Fin (D.V.k x) → Coin)
    (c : (Claims.Sampling.E D decode).C x) : NoHash ((Claims.Sampling.E D decode).ext x r₀ c) := by
  intro v₀
  dsimp only [Claims.Sampling.E]
  split
  · trivial
  · split_ifs
    · generalize List.map c.1 (List.finRange (D.cap x)) = rs
      refine noHash_bind _ _ (noHash_waitAccC _ _ _) fun outer => ?_
      generalize (List.finRange (D.k x)).zip outer = jos
      refine noHash_bind _ _ (noHash_flatMapRP _ (fun jo => ?_) _) fun _ => trivial
      generalize List.map (fun l => forkOn (D.stage2 x) jo.2.1 (c.2.1 jo.1 l)) (List.finRange (D.cap x)) = rs'
      refine noHash_bind _ _ (noHash_waitAccC _ _ _) fun inner => ?_
      generalize (List.finRange (D.k x)).zip inner = lis
      exact noHash_flatMapRP _ (fun _ => noHash_flatMapRP _ (fun _ => hu _ _ _ _ _ _) _) _
    · trivial

/-- At a commitment whose openings make no call, the sampling extractor logs no call of its own. -/
theorem ownLog_E {Col T : Type} [DecidableEq Col] [Zero Col] (D : Claims.Sampling.Deploy Coin Col T)
    (hCm : ∀ roots o, ∃ b, D.commit.opens roots o = .ret b) (decode : List (ℕ × Col) → List T) (H : Msg → Msg)
    (x : Msg) (run : (Fin (D.V.k x) → Coin) → Trace) (r₀ : Fin (D.V.k x) → Coin)
    (c : (Claims.Sampling.E D decode).C x) :
    ownLog H run ((Claims.Sampling.E D decode).ext x r₀ c) = [] :=
  ownLog_noHash H run _ (noHash_E_of D decode (noHash_unitExt D hCm decode) x r₀ c)

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem ksNotD_opens (roots : List Msg) (o : Opening Bool) :
    ∃ b, (ksNotD Coin).commit.opens roots o = .ret b :=
  ⟨decide (o.val = decide (0 < o.pos)), rfl⟩

theorem ksNotD_bext (H : Msg → Msg) (decode : List (ℕ × Unit) → List Unit) (x : Msg) :
    (Bext H (ksNotD Coin).V (Claims.Sampling.E (ksNotD Coin) decode) Unit silentP x).Pr H = 0 := by
  simp only [Bext]
  rw [dite_eq_left (show 0 < (ksNotD Coin).V.k x from Nat.one_pos)]
  unfold Finder.Pr
  rw [Pr_congr fun s => iff_false_intro (by
    simp [ownLog_E (ksNotD Coin) ksNotD_opens, findColl])]
  exact Pr_false

theorem ksNotD_btap (H : Msg → Msg) (decode : List (ℕ × Unit) → List Unit) (x : Msg) :
    (Btap H (ksNotD Coin).V (Claims.Sampling.E (ksNotD Coin) decode) (ksNotD Coin).commit (ksNotD Coin).parse
      (fun _ => frameTops) Unit silentP x).Pr H = 0 := by
  unfold Finder.Pr
  rw [Pr_congr fun s => iff_false_intro (by
    simp only [Btap, ownLog_E (ksNotD Coin) ksNotD_opens]
    simp [tapLog, ksNotD, leakD, notCm, OC.eval, frameTops, findColl])]
  exact Pr_false

theorem ksNotD_breg (H : Msg → Msg) (decode : List (ℕ × Unit) → List Unit) (x : Msg) :
    (Breg H (ksNotD Coin).V (Claims.Sampling.E (ksNotD Coin) decode) (ksNotD Coin).commit (ksNotD Coin).parse
      (fun _ => badReg) Unit silentP x).Pr H = 0 := by
  unfold Finder.Pr
  rw [Pr_congr fun s => iff_false_intro (by
    simp only [Breg, ownLog_E (ksNotD Coin) ksNotD_opens]
    simp [regLog, ksNotD, leakD, notCm, OC.eval, badReg, findColl])]
  exact Pr_false

end Toy

/-- **The per-window bound fails at `ksNotD` with the registration `badReg`**, at every hash, at one step at `δ = 0`
and the statement `[]`: no witness is in `RLeakTap` (`ksNotD_noRLeakTap`), so the silent prover's run fails it with
probability `1`, while `κ < 1/3` and the finders find nothing. -/
theorem ksNotD_reg_bound_fails (H : Msg → Msg) :
    ¬ PrBad H (ksNotD Unit).V
        (RLeakTap H (ksNotD Unit).commit (ksNotD Unit).parse (ksNotD Unit).partition 1 1 1 (fun _ => freeLay)
          (fun _ => 1) 0 (fun s _ => AsReg s.prog freeLay 1 {0} (regVal badReg)) 1 (fun _ => frameTops))
        (Claims.Sampling.E (ksNotD Unit) fun _ => [()]) [] silentP ≤
      Claims.Sampling.κ (ksNotD Unit) 0 [] +
          Real.sqrt (Claims.Sampling.γ (ksNotD Unit) [] * PrB H (ksNotD Unit).V silentP []) +
        (Bext H (ksNotD Unit).V (Claims.Sampling.E (ksNotD Unit) fun _ => [()]) Unit silentP []).Pr H +
        (Btap H (ksNotD Unit).V (Claims.Sampling.E (ksNotD Unit) fun _ => [()]) (ksNotD Unit).commit
          (ksNotD Unit).parse (fun _ => frameTops) Unit silentP []).Pr H +
        (Breg H (ksNotD Unit).V (Claims.Sampling.E (ksNotD Unit) fun _ => [()]) (ksNotD Unit).commit
          (ksNotD Unit).parse (fun _ => badReg) Unit silentP []).Pr H := by
  intro h
  have hp : PrBad H (ksNotD Unit).V
      (RLeakTap H (ksNotD Unit).commit (ksNotD Unit).parse (ksNotD Unit).partition 1 1 1 (fun _ => freeLay)
        (fun _ => 1) 0 (fun s _ => AsReg s.prog freeLay 1 {0} (regVal badReg)) 1 (fun _ => frameTops))
      (Claims.Sampling.E (ksNotD Unit) fun _ => [()]) [] silentP = 1 :=
    prBad_accV H _ _ [] silentP (ksNotD_noRLeakTap H _ [])
  have hb : PrB H (ksNotD Unit).V silentP [] = 0 := prB_accV H silentP []
  simp only [hp, hb, mul_zero, Real.sqrt_zero, add_zero, ksNotD_bext, ksNotD_btap, ksNotD_breg] at h
  linarith [ksNotD_κ (Coin := Unit) []]

/-- **`PeriodLeakRegistered`'s per-window part, at every registration**: its first conjunct without the antecedent
`RegOpensAt`, every other hypothesis kept. -/
def WindowLeakAtAnyOps : Prop :=
  ∀ {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
    (H : Msg → Msg) (D : Claims.Sampling.Deploy Coin Col T) (L K S : ℕ)
    (lay : Claims.Sampling.AuditStatement → BoundaryGates L S), KSBy H D (wiredOk fun s => (lay s).wires) →
    ∀ (k : Claims.Sampling.AuditStatement → ℕ) (δ : ℝ) (decode : List (ℕ × Col) → List T), 0 ≤ δ →
    Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ decode →
    ∀ (reg : Claims.Sampling.AuditStatement → Finset ℕ) (ops : Claims.Sampling.AuditStatement → List (Opening Bool))
      (Dj : ℕ) (tops : Claims.Sampling.AuditStatement → List (Opening Bool)), D.commit.Binds H →
    ∀ m, ScopeAcc D lay k m → ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
      PrBad H D.V
          (RLeakTap H D.commit D.parse D.partition L K S lay k δ
            (fun s _ => AsReg s.prog (lay s) (k s) (reg s) (regVal (ops s))) Dj tops)
          (Claims.Sampling.E D decode) m P ≤
        Claims.Sampling.κ D δ m + Real.sqrt (Claims.Sampling.γ D m * PrB H D.V P m) +
          (Bext H D.V (Claims.Sampling.E D decode) Ω P m).Pr H +
          (Btap H D.V (Claims.Sampling.E D decode) D.commit D.parse tops Ω P m).Pr H +
          (Breg H D.V (Claims.Sampling.E D decode) D.commit D.parse ops Ω P m).Pr H

/-- **Without `RegOpensAt`, `PeriodLeakRegistered`'s per-window part is false**: at SHA-512, at `ksNotD`, where `KS`
holds (`ksNotD_KS`) and so `KSBy` at the wired check (`freeLay` has no wires), the decoder that lists the one trace
is within Flock's radius, `notCm` binds and the statement is in scope, the registration `badReg`, which doesn't open
(`ksNotD_not_regOpens`), breaks the bound (`ksNotD_reg_bound_fails`). -/
theorem not_windowLeakAtAnyOps : ¬ WindowLeakAtAnyOps := fun h =>
  ksNotD_reg_bound_fails FlockDeploy.sha512
    (h FlockDeploy.sha512 (ksNotD Unit) 1 1 1 (fun _ => freeLay) (ksWired_none (fun _ => rfl) (ksNotD_KS _))
      (fun _ => 1) 0 (fun _ => [()]) le_rfl (Claims.Witness.Flock.listWithin_unit _ _ _) (fun _ => {0})
      (fun _ => badReg) 1 (fun _ => frameTops) (notCm_binds _) [] (ksNotD_scopeAcc []) Unit silentP)

/-! ## `WitnessToys`' rows format that doesn't read back, at `freeLay` -/

theorem obs_eq_obs1 {o : Obs 1 1 1} (hr : ∀ ℓ σ, o.rows ℓ σ = some []) (hj : o.jitter = 0)
    (hs : ∀ κ, o.status κ = .complete) : o = obs1 := by
  obtain ⟨rows, jitter, status⟩ := o
  have h1 : rows = fun _ _ => some [] := funext fun ℓ => funext (hr ℓ)
  have h3 : status = fun _ => .complete := funext hs
  obtain rfl : jitter = 0 := hj
  subst h1 h3
  rfl

/-- **The record doesn't tell the servings apart**: on every serving `z`, its one window is `obs1`, the frame encoded
as the empty list. -/
theorem blindRun_obs (z : ℕ) (i : Fin 1) : (blindRun () (oneSched, z)).2 i = [obs1] := by
  fin_cases i
  show [_] = [obs1]
  congr 1
  exact obs_eq_obs1 (fun ℓ σ => by
    fin_cases ℓ
    fin_cases σ
    simp [blindRows, bucketFrames, laneFrames, oneSched, cellArr, wardenCode, wardenLane, WardenSchedule.bucket,
      streamCell]) rfl fun _ => rfl

/-- **`frameTops` registers only what the circuit accepts at `goodReg`**: a sequence the network certifier registers
at `frameTops` is accepted with the empty faulty set and the input `false`, whose negation `true` holds the frame. -/
theorem frameTops_acc (H : Msg → Msg) (op : ℕ → List (Obs 1 1 1) → List Bool) (o : List (Obs 1 1 1))
    (hto : TapOpens H notCm freeLay 1 op notStmt frameTops o) (hcov : Covered 1 o) (hlen : o.length = 1)
    (hj : ∀ y ∈ o, y.jitter < 1) :
    o ∈ AccSeqsSampled notProg notPart notStmt.design freeLay 1 (AsReg notProg freeLay 1 {0} (regVal goodReg)) 0 op
      1 := by
  obtain ⟨-, hstep⟩ := hto
  obtain ⟨-, -, h0, hrows, -⟩ := hstep 0 Nat.one_pos
  refine ⟨hlen, hj, ∅, empty_mem_notFaulty, fun _ => false,
    ⟨fun g hg _ => by obtain rfl := Finset.mem_singleton.1 hg; rfl, fun _ _ => rfl⟩, fun _ => none,
    fun _ _ h => by simp at h, fun t ht => ?_⟩
  obtain rfl : t = 0 := by omega
  refine ⟨h0, ?_, hcov 0 Nat.one_pos h0⟩
  funext ℓ σ
  rw [hrows ℓ]
  show rowOf (tapVal frameTops) (freeLay.egress 0 ℓ) σ = rowOf (valOver notProg (resolveWires notProg freeLay.wires
    (freeLay.inputOf 1 (ingressUpTo op o 0) fun _ => false) (fun _ => none) notProg.gates.length) fun _ => none)
    (freeLay.egress 0 ℓ) σ
  rw [resolveWires_nil (wires := freeLay.wires) rfl]
  have hv : valOver notProg (freeLay.inputOf 1 (ingressUpTo op o 0) fun _ => false) (fun _ => none) 1 = true :=
    notProg_val _ rfl
  show (if tapVal frameTops 1 then some (([] : List ℕ).map (tapVal frameTops)) else none) =
    if valOver notProg (freeLay.inputOf 1 (ingressUpTo op o 0) fun _ => false) (fun _ => none) 1 then
      some (([] : List ℕ).map (valOver notProg (freeLay.inputOf 1 (ingressUpTo op o 0) fun _ => false) fun _ => none))
    else none
  rw [hv]
  rfl

end NetTiming.Witness
