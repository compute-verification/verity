import Proofs.Flock.Claims.Witnesses.Flock
import Proofs.Flock.Claims.ZK.SamplingNonVacuous

/-!
# Refutation witnesses: the sampling compiler's claims

Each of the sampling compiler's claims (`Definitions.Claims.Sampling`) fails at a toy deployment. The statement is
`toyStmt`: one input gate, returned, drawn `all`, its partition `toyPart` one proof unit `{0}`, in scope
(`inScope_toy`), so the witnesses hold whether or not `KS`, `KSCount` and `Bits` are read on the scope only.

* `sksD`: the verifier accepts after one round, and no opening passes (`rejCm`), so every unit is faulty and no
  witness is in the relation at any `δ ≥ 0` or count `t ≤ 1`. The extractor makes no SHA-512 call of its own
  (`noHash_E`), so `B_ext` finds nothing, `B` finds nothing, and the numbers give `κ < 1/3` at `δ = 0`.
  `KS_fails`, `KSCount_fails` (at `t = 1`, whose miss is `0`), `Bits_fails` (there `λ > 0`).
* `compSD`: the verifier rejects after one round, and every opening passes (`accCm`), so the opening of the input
  (`toyWit`) leaves no unit faulty. `Complete_fails`, `Nontrivial_fails`, `ZK_fails`
  (`Sampling.ZKMal.not_samplingZK_collSim`, at `Sampling.ZKMal.quietZD`, its masked deployment whose reference prover
  is quiet and whose simulator collides), `Binding_fails` (two openings of different values at one position pass
  with no call), `Universal_fails` (its universal unit returns nothing).
* `effSD`: its sessions' decoder counts one step, and nothing parses while the check takes everything.
  `Efficient_fails` at `p = 0`; `ScopeMatches_fails`.
* `ExtractOwnLe_fails`: an extractor that hashes once, at a bound of `0`.

`FaultyBoundary` is a theorem (`Claims.Sampling.faultyBoundary`), so it has no refutation. `BitsLin`, `KSLin` and the
scoped `KSLinOn` are refuted through `Bits_fails`, `KS_fails` and `KSCount_fails`, whose proofs read each.

`ZK` reads the masked deployment's own reference prover and simulator, as `Complete` reads the deployment's
(`Sampling.Deploy.P`), so it is not met at every masked deployment (`ZK_vacuous_refuted`, at `zkSD`, `compSD` with a
masked inner Flock).
-/

namespace Claims.Witness.Sampling

open Claims Claims.Sampling

/-! ## Rerun programs with no call of their own -/

/-- A rerun program that makes no SHA-512 call of its own. -/
def NoHash {R α : Type} : RP R α → Prop
  | .ret _ => True
  | .hash _ _ => False
  | .rerun _ k => ∀ v, NoHash (k v)

section NoHash

variable {R α β : Type}

theorem ownLog_noHash (H : Msg → Msg) (run : R → Trace) : ∀ p : RP R α, NoHash p → ownLog H run p = []
  | .ret _, _ => rfl
  | .hash _ _, h => h.elim
  | .rerun r k, h => ownLog_noHash H run (k (run r).seen) (h _)

theorem noHash_bind : ∀ (p : RP R α) (f : α → RP R β), NoHash p → (∀ a, NoHash (f a)) → NoHash (p.bind f)
  | .ret a, _, _, hf => hf a
  | .hash _ _, _, h, _ => h.elim
  | .rerun _ k, f, h, hf => fun v => noHash_bind (k v) f (h v) hf

theorem noHash_waitAccC (k : ℕ) : ∀ (rs : List R) (acc : List (R × Seen)), NoHash (waitAccC k rs acc)
  | [], _ => trivial
  | r :: rs, acc => by
    unfold waitAccC
    split_ifs
    · trivial
    · exact fun v => noHash_waitAccC k rs _

theorem noHash_flatMapRP {γ : Type} (f : β → RP R (List γ)) (hf : ∀ b, NoHash (f b)) :
    ∀ bs : List β, NoHash (flatMapRP f bs)
  | [] => trivial
  | b :: bs => noHash_bind _ _ (hf b) fun _ => noHash_bind _ _ (noHash_flatMapRP f hf bs) fun _ => trivial

end NoHash

/-! ## The toy statement -/

/-- A commitment whose every opening passes, with no call. -/
def accCm : Commitment Bool := ⟨fun _ _ => .ret true, fun _ _ _ => []⟩

/-- A commitment whose every opening fails, with no call. -/
def rejCm : Commitment Bool := ⟨fun _ _ => .ret false, fun _ _ _ => []⟩

/-- One input gate, returned. -/
def toyProg : Prog := ⟨[.input], [0]⟩

/-- The program, drawn `all`. -/
def toyStmt : AuditStatement := ⟨[], toyProg, ⟨"", 0, []⟩, ⟨[.all]⟩, []⟩

/-- One class of one stratum of one proof block, `{0}`. -/
def toyPart : Partition := ⟨[[[[0]]]]⟩

/-- The opening of the input's value. -/
def toyWit : Wit := ⟨[⟨0, true, []⟩]⟩

theorem toyPart_blocks : toyPart.blocks = [[0]] := rfl

theorem toy_boundary : toyProg.Boundary [0] 0 = true := by decide

theorem toy_ref (H : Msg → Msg) (Cm : Commitment Bool) (w : Wit) (ι : ℕ → Bool) :
    toyProg.refIn [0] (w.val H Cm toyStmt) ι 0 = w.val H Cm toyStmt 0 := by
  simp [Prog.refIn, toy_boundary]

theorem toy_runIn (ref : ℕ → Option Bool) : toyProg.runIn ref [0] 0 = ref 0 := by
  rw [Prog.runIn]
  rfl

/-- The unit is correct exactly when the input's value is opened. -/
theorem toy_correct_iff (H : Msg → Msg) (Cm : Commitment Bool) (w : Wit) (ι : ℕ → Bool) :
    toyProg.Correct (toyProg.refIn [0] (w.val H Cm toyStmt) ι) [0] ↔ ∃ b, w.val H Cm toyStmt 0 = some b := by
  constructor
  · intro h
    obtain ⟨b, hb, _⟩ := (h 0 (List.mem_singleton.2 rfl)).2 toy_boundary
    exact ⟨b, (toy_ref H Cm w ι).symm.trans hb⟩
  · rintro ⟨b, hb⟩ g hg
    obtain rfl := List.mem_singleton.1 hg
    refine ⟨fun x hx i hi _ => ?_, fun _ => ⟨b, ?_, ?_⟩⟩
    · simp only [toyProg, List.getElem?_cons_zero, Option.some.injEq] at hx
      subst hx
      simp [BGate.reads] at hi
    · rw [toy_ref, hb]
    · rw [toy_runIn, toy_ref, hb]

theorem toy_faulty_iff (H : Msg → Msg) (Cm : Commitment Bool) (w : Wit) :
    Faulty H Cm toyStmt toyPart w 0 ↔ w.val H Cm toyStmt 0 = none := by
  unfold Faulty
  rw [toyPart_blocks]
  simp only [List.getElem?_cons_zero, Option.some.injEq, exists_eq_left']
  rw [show toyStmt.prog = toyProg from rfl]
  simp only [toy_correct_iff, exists_const]
  generalize w.val H Cm toyStmt 0 = o
  cases o <;> simp

theorem toy_faulty (H : Msg → Msg) (Cm : Commitment Bool) (w : Wit) :
    faulty H Cm toyStmt toyPart w = if w.val H Cm toyStmt 0 = none then {0} else ∅ := by
  ext v
  simp only [faulty, Finset.mem_filter, Finset.mem_range, toyPart_blocks, List.length_singleton]
  constructor
  · rintro ⟨hv, hf⟩
    obtain rfl : v = 0 := by omega
    simp [(toy_faulty_iff H Cm w).1 hf]
  · intro hv
    split_ifs at hv with h
    · obtain rfl := Finset.mem_singleton.1 hv
      exact ⟨Nat.one_pos, (toy_faulty_iff H Cm w).2 h⟩
    · simp at hv

theorem toy_Pa (B : Finset ℕ) : toyStmt.design.Pa toyPart B = if window 0 1 B = ∅ then 1 else 0 := by
  simp [Design.Pa, toyStmt, toyPart, classesPa, OneStage.Pa]

theorem toyWit_val (H : Msg → Msg) : toyWit.val H accCm toyStmt 0 = some true := by
  simp [Wit.val, toyWit, Commitment.Opens, accCm, OC.eval]

theorem rejCm_val (H : Msg → Msg) (w : Wit) : w.val H rejCm toyStmt 0 = none := by
  simp [Wit.val, Commitment.Opens, rejCm, OC.eval]

theorem rejCm_faulty (H : Msg → Msg) (w : Wit) : faulty H rejCm toyStmt toyPart w = {0} := by
  rw [toy_faulty, rejCm_val]; rfl

/-- The toy is in the sampling scope. -/
theorem inScope_toy : InScope toyPart toyStmt := by
  refine ⟨⟨fun g x hx i hi => ?_, by decide⟩,
    ⟨by decide, by decide, by decide, by decide, by decide⟩, ?_⟩
  · rcases g with _ | g
    · simp [toyStmt, toyProg] at hx
      subst hx
      simp [BGate.reads] at hi
    · simp [toyStmt, toyProg] at hx
  · refine ⟨rfl, ?_⟩
    simp [toyStmt, toyPart, OneStage.WellFormed]

/-! ## The toy deployments -/

/-- The toy deployment of `KS`, `KSCount` and `Bits`: the verifier accepts, no opening passes, and no unit has a
session. -/
noncomputable def sksD (Coin : Type) : Deploy Coin Unit Unit where
  V := accV Coin
  parse _ := some toyStmt
  scopeCheck _ := true
  partition _ := some toyPart
  commit := rejCm
  universal _ := ⟨[], []⟩
  flock := Flock.ksD Coin
  stage2 _ := ∅
  checked _ _ := []
  rounds _ _ := ∅
  session _ _ _ := none
  reveal _ _ _ _ := []
  innerWaits _ := 8
  innerN₀ _ := 64 * Real.exp 1
  innerCap _ := 65536
  εtable _ := 0
  weight _ := 0
  openCalls _ := 0
  Tape := Unit
  P _ := silentP
  δc _ _ := 0

/-- The toy deployment of `Complete`, `Nontrivial`, `ZK`, `Binding` and `Universal`: the verifier rejects, every
opening passes, and the universal unit returns nothing. -/
noncomputable def compSD (Coin : Type) : Deploy Coin Unit Unit where
  V := rejV Coin
  parse _ := some toyStmt
  scopeCheck _ := true
  partition _ := some toyPart
  commit := accCm
  universal _ := ⟨[], []⟩
  flock := Flock.ksD Coin
  stage2 _ := ∅
  checked _ _ := []
  rounds _ _ := ∅
  session _ _ _ := none
  reveal _ _ _ _ := []
  innerWaits _ := 0
  innerN₀ _ := 0
  innerCap _ := 0
  εtable _ := 0
  weight _ := 0
  openCalls _ := 0
  Tape := Unit
  P _ := silentP
  δc _ _ := 0

/-- The empty column, the zero of a word of field-element lists. -/
local instance zeroCol : Zero (List Bool) := ⟨[]⟩

/-- The toy deployment of `Efficient` and `ScopeMatches`: its sessions' decoder counts one step, nothing parses, and
the check takes everything. -/
def effSD (Coin : Type) : Deploy Coin (List Bool) Unit where
  V := accV Coin
  parse _ := none
  scopeCheck _ := true
  partition _ := none
  commit := accCm
  universal _ := ⟨[], []⟩
  flock := Flock.effD Coin
  stage2 _ := ∅
  checked _ _ := []
  rounds _ _ := ∅
  session _ _ _ := none
  reveal _ _ _ _ := []
  innerWaits _ := 0
  innerN₀ _ := 0
  innerCap _ := 0
  εtable _ := 0
  weight _ := 0
  openCalls _ := 0
  Tape := Unit
  P _ := silentP
  δc _ _ := 0

/-! ## `sksD`'s numbers -/

section Numbers

variable {Coin : Type}

theorem sksD_scope (x : Msg) : Scope (sksD Coin) x := ⟨toyStmt, toyPart, rfl, rfl, inScope_toy⟩

theorem sksD_nPU (x : Msg) : nPU (sksD Coin) x = 1 := rfl

theorem sksD_k (x : Msg) : (sksD Coin).k x = 2 ^ 40 * 3 := rfl

theorem sksD_innerK (x : Msg) : (sksD Coin).innerK x = 256 := by
  have h : 4 * (64 * Real.exp 1) / Real.exp 1 = ((256 : ℕ) : ℝ) := by
    rw [div_eq_iff (Real.exp_pos 1).ne']; push_cast; ring
  simp only [Deploy.innerK, sksD, h, Nat.ceil_natCast]

theorem slack_pos : 0 < slack := by unfold slack; positivity

theorem slack_le_two : slack ≤ 2 := by
  unfold slack
  calc (2 : ℝ) ^ ((2 : ℝ) ^ (-40 : ℝ)) ≤ 2 ^ (1 : ℝ) :=
        Real.rpow_le_rpow_of_exponent_le (by norm_num) (Real.rpow_le_one_of_one_le_of_nonpos (by norm_num) (by norm_num))
    _ = 2 := Real.rpow_one 2

theorem exp_neg_le_inv {y : ℝ} (hy : 0 ≤ y) : Real.exp (-y) ≤ (y + 1)⁻¹ := by
  rw [Real.exp_neg]
  exact inv_anti₀ (by linarith) (Real.add_one_le_exp y)

theorem sksD_κF (x : Msg) : κF (sksD Coin) x ≤ 1 / 20 := by
  have hw : (2 : ℝ) ^ (-((8 : ℕ) : ℝ)) = 1 / 256 := by
    rw [Real.rpow_neg (by norm_num), Real.rpow_natCast]; norm_num
  have he := exp_neg_le_inv (show (0 : ℝ) ≤ 32 by norm_num)
  unfold κF
  rw [sksD_innerK]
  simp only [sksD] at hw ⊢
  rw [hw, show -((256 : ℕ) : ℝ) / 8 = -32 by norm_num]
  have : (32 + 1 : ℝ)⁻¹ = 1 / 33 := by norm_num
  push_cast
  linarith

theorem sksD_κF_nonneg (x : Msg) : 0 ≤ κF (sksD Coin) x := by
  unfold κF
  simp only [sksD]
  positivity

theorem sksD_outer (x : Msg) :
    2 * ((sksD Coin).k x : ℝ) / ((sksD Coin).cap x : ℝ) + Real.exp (-((sksD Coin).k x : ℝ) / 8) ≤ 1 / 50 := by
  have hk : ((sksD Coin).k x : ℝ) = 2 ^ 40 * 3 := by rw [sksD_k]; push_cast; ring
  have hc : ((sksD Coin).cap x : ℝ) = 2 ^ 128 * (2 ^ 40 * 3) := by
    simp only [Deploy.cap, sksD_k]; push_cast; ring
  rw [hk, hc, neg_div]
  have he := exp_neg_le_inv (show (0 : ℝ) ≤ 2 ^ 40 * 3 / 8 by positivity)
  have h1 : (2 ^ 40 * 3 / 8 + 1 : ℝ)⁻¹ ≤ 1 / 100 := by
    rw [inv_le_comm₀ (by positivity) (by norm_num)]; norm_num
  have h2 : 2 * (2 ^ 40 * 3 : ℝ) / (2 ^ 128 * (2 ^ 40 * 3)) ≤ 1 / 100 := by
    rw [div_le_iff₀ (by positivity)]; norm_num
  linarith

theorem sksD_κ (x : Msg) : κ (sksD Coin) 0 x < 1 / 3 := by
  have hs := slack_le_two
  have hs0 := slack_pos
  have hF := sksD_κF (Coin := Coin) x
  have hF0 := sksD_κF_nonneg (Coin := Coin) x
  have ho := sksD_outer (Coin := Coin) x
  have ho0 : 0 ≤ 2 * ((sksD Coin).k x : ℝ) / ((sksD Coin).cap x : ℝ) + Real.exp (-((sksD Coin).k x : ℝ) / 8) := by
    positivity
  unfold κ
  rw [zero_add]
  have h1 : slack ^ 2 * κF (sksD Coin) x ≤ 4 * (1 / 20) := by
    have : slack ^ 2 ≤ 4 := by nlinarith
    calc slack ^ 2 * κF (sksD Coin) x ≤ 4 * κF (sksD Coin) x := by gcongr
      _ ≤ 4 * (1 / 20) := by gcongr
  have h2 : (1 + slack) * (2 * ((sksD Coin).k x : ℝ) / ((sksD Coin).cap x : ℝ) +
      Real.exp (-((sksD Coin).k x : ℝ) / 8)) ≤ 3 * (1 / 50) := by
    calc _ ≤ 3 * (2 * ((sksD Coin).k x : ℝ) / ((sksD Coin).cap x : ℝ) + Real.exp (-((sksD Coin).k x : ℝ) / 8)) := by
          gcongr; linarith
      _ ≤ 3 * (1 / 50) := by gcongr
  linarith

theorem sksD_κ_pos (x : Msg) : 0 < κ (sksD Coin) 0 x := by
  have := sksD_κF_nonneg (Coin := Coin) x
  unfold κ
  have := slack_pos
  positivity

theorem sksD_γ (x : Msg) : γ (sksD Coin) x = slack ^ 4 * 16 := by
  simp [γ, sksD, accV]

theorem sksD_work (x : Msg) : work (sksD Coin) x ≤ 2 ^ 100 := by
  unfold work
  rw [sksD_k, sksD_nPU, sksD_innerK]
  simp only [sksD]
  norm_num

theorem sksD_work_pos (x : Msg) : 0 < work (sksD Coin) x := by
  unfold work
  positivity

theorem sksD_lam_pos (x : Msg) : 0 < lamLin (κ (sksD Coin) 0 x) (γ (sksD Coin) x) 1 (work (sksD Coin) x) := by
  have hκ := sksD_κ (Coin := Coin) x
  have hκ0 := sksD_κ_pos (Coin := Coin) x
  have hw := sksD_work (Coin := Coin) x
  have hw0 := sksD_work_pos (Coin := Coin) x
  have hs := slack_le_two
  have hs0 := slack_pos
  have h3 : Real.logb 2 3 < 2 := by
    rw [Real.logb_lt_iff_lt_rpow (by norm_num) (by norm_num)]; norm_num
  have h1 : Real.logb 2 3 < Real.logb 2 (1 / κ (sksD Coin) 0 x) := by
    refine Real.logb_lt_logb (by norm_num) (by norm_num) ?_
    rw [lt_div_iff₀ hκ0]; linarith
  have hγ : Real.logb 2 (γ (sksD Coin) x) ≤ 8 := by
    rw [sksD_γ, Real.logb_le_iff_le_rpow (by norm_num) (by positivity)]
    have : slack ^ 4 ≤ 16 := by
      calc slack ^ 4 ≤ 2 ^ 4 := by gcongr
        _ = 16 := by norm_num
    norm_num; nlinarith
  have hW : Real.logb 2 (2 * 1 * work (sksD Coin) x) ≤ 101 := by
    rw [Real.logb_le_iff_le_rpow (by norm_num) (by positivity)]
    norm_num; linarith
  unfold lamLin
  have := lt_min (lt_min h1 (show Real.logb 2 3 < 255.5 - Real.logb 2 (γ (sksD Coin) x) / 2 by linarith))
    (show Real.logb 2 3 < 256 - Real.logb 2 (2 * 1 * work (sksD Coin) x) by linarith)
  linarith

theorem sksD_noR (H : Msg → Msg) (δ : ℝ) (hδ : 0 ≤ δ) (x : Msg) (w : Wit) :
    ¬ R H (sksD Coin).commit (sksD Coin).parse (sksD Coin).partition δ x w := by
  rintro ⟨s, P, hs, hP, h⟩
  simp only [sksD, Option.some.injEq] at hs hP
  subst hs hP
  change δ < toyStmt.design.Pa toyPart (faulty H rejCm toyStmt toyPart w) at h
  rw [rejCm_faulty, toy_Pa, ite_eq_right (by decide)] at h
  linarith

theorem sksD_noRCount (H : Msg → Msg) (x : Msg) (w : Wit) :
    ¬ RCount H (sksD Coin).commit (sksD Coin).parse (sksD Coin).partition 1 x w := by
  rintro ⟨s, P, hs, hP, h⟩
  simp only [sksD, Option.some.injEq] at hs hP
  subst hs hP
  change (faulty H rejCm toyStmt toyPart w).card < 1 at h
  rw [rejCm_faulty] at h
  simp at h

theorem sksD_missAt (x : Msg) : missAt (sksD Coin) 1 x = 0 := by
  have hle : toyStmt.design.miss toyPart 1 ≤ 0 := by
    refine Real.iSup_le (fun B => ?_) le_rfl
    obtain ⟨hB, hc⟩ := B.2
    have h0 : 0 ∈ B.1 := by
      obtain ⟨v, hv⟩ := Finset.card_pos.1 hc
      have := Finset.mem_range.1 (hB hv)
      rw [toyPart_blocks, List.length_singleton] at this
      obtain rfl : v = 0 := by omega
      exact hv
    rw [toy_Pa, ite_eq_right]
    intro hw
    have : 0 ∈ window 0 1 B.1 := by simp [window, h0]
    rw [hw] at this
    simp at this
  show max 0 (toyStmt.design.miss ((sksD Coin).part toyStmt) 1) = 0
  exact max_eq_left hle

end Numbers

/-! ## `sksD`'s extractor makes no call of its own -/

section Extract

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

omit [Inhabited Coin] in
/-- At a deployment whose units have no session, the sampling extractor makes no call of its own: its waits rerun, and
its unit extractions return nothing. -/
theorem noHash_E {Col T : Type} [DecidableEq Col] [Zero Col] (D : Deploy Coin Col T) (hs : ∀ x u v, D.session x u v = none)
    (decode : List (ℕ × Col) → List T) (x : Msg) (r₀ : Fin (D.V.k x) → Coin) (c : (E D decode).C x) :
    NoHash ((E D decode).ext x r₀ c) := by
  intro v₀
  dsimp only [E]
  split
  · trivial
  · split_ifs
    -- The waits' coin lists are generalized first: unifying against them unfolds `List.finRange` at the cap.
    · generalize List.map c.1 (List.finRange (D.cap x)) = rs
      refine noHash_bind _ _ (noHash_waitAccC _ _ _) fun outer => ?_
      generalize (List.finRange (D.k x)).zip outer = jos
      refine noHash_bind _ _ (noHash_flatMapRP _ (fun jo => ?_) _) fun _ => trivial
      generalize List.map (fun l => forkOn (D.stage2 x) jo.2.1 (c.2.1 jo.1 l)) (List.finRange (D.cap x)) = rs'
      refine noHash_bind _ _ (noHash_waitAccC _ _ _) fun inner => ?_
      generalize (List.finRange (D.k x)).zip inner = lis
      refine noHash_flatMapRP _ (fun _ => noHash_flatMapRP _ (fun _ => ?_) _) _
      simp only [unitExt, hs]
      trivial
    · trivial

theorem sksD_bext (H : Msg → Msg) (decode : List (ℕ × Unit) → List Unit) (x : Msg) :
    (Bext H (sksD Coin).V (E (sksD Coin) decode) Unit silentP x).Pr H = 0 := by
  simp only [Bext]
  rw [dite_eq_left (show 0 < (sksD Coin).V.k x from Nat.one_pos)]
  unfold Finder.Pr
  rw [Pr_congr fun s => iff_false_intro (by
    simp [ownLog_noHash _ _ _ (noHash_E (sksD Coin) (fun _ _ _ => rfl) _ _ _ _), findColl])]
  exact Pr_false

theorem sksD_expectedCR (H : Msg → Msg) (decode : List (ℕ × Unit) → List Unit) (x : Msg) :
    ExpectedCR H (Bext H (sksD Coin).V (E (sksD Coin) decode) Unit silentP x) := by
  unfold ExpectedCR
  rw [sksD_bext]
  exact div_nonneg (Ex_nonneg fun _ => Nat.cast_nonneg _) (by positivity)

/-- `KSLin`'s bound fails at `sksD`'s silent prover, at a relation with no witness at `x` and a bound `κ' < 1`. -/
theorem sksD_bound_fails (H : Msg → Msg) (decode : List (ℕ × Unit) → List Unit) (Rx : Rel Wit) (κ' : ℝ) (x : Msg)
    (hR : ∀ w, ¬ Rx x w) (hκ : κ' < 1) :
    ¬ PrBad H (sksD Coin).V Rx (E (sksD Coin) decode) x silentP ≤
      κ' + Real.sqrt (γ (sksD Coin) x * PrB H (sksD Coin).V silentP x) +
        1 * (Bext H (sksD Coin).V (E (sksD Coin) decode) Unit silentP x).Pr H := by
  intro h
  have hp : PrBad H (sksD Coin).V Rx (E (sksD Coin) decode) x silentP = 1 := prBad_accV H Rx _ x silentP hR
  have hb : PrB H (sksD Coin).V silentP x = 0 := prB_accV H silentP x
  rw [hp, hb, mul_zero, Real.sqrt_zero, add_zero, sksD_bext, mul_zero, add_zero] at h
  linarith

end Extract

/-! ## The claims -/

section Claims

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

-- The `first` alternatives cover `KS`, `KSCount` and `Bits` read at every statement and read on the scope only, so
-- one proof elaborates on both.
set_option linter.unreachableTactic false in
set_option linter.unusedTactic false in
/-- **The sampling `KS` fails** at `sksD`, at `δ = 0`. -/
theorem KS_fails (H : Msg → Msg) : ¬ Claims.Sampling.KS H (sksD Coin) := by
  intro h
  have h1 := h 0 (fun _ => [()]) le_rfl (Flock.listWithin_unit _ _ _)
  have hb := sksD_bound_fails (Coin := Coin) H (fun _ => [()]) _ _ [] (sksD_noR (Coin := Coin) H 0 le_rfl []) (by linarith [sksD_κ (Coin := Coin) []])
  first
  | exact hb (h1 [] Unit silentP)
  | exact hb (h1 [] (sksD_scope []) Unit silentP)

set_option linter.unreachableTactic false in
set_option linter.unusedTactic false in
/-- **The sampling `KSCount` fails** at `sksD`, at `t = 1`. -/
theorem KSCount_fails (H : Msg → Msg) : ¬ KSCount H (sksD Coin) := by
  intro h
  have h1 := h 1 (fun _ => [()]) (Flock.listWithin_unit _ _ _)
  have hb := sksD_bound_fails (Coin := Coin) H (fun _ => [()]) _ (κ (sksD Coin) (missAt (sksD Coin) 1 []) []) []
    (sksD_noRCount (Coin := Coin) H []) (by rw [sksD_missAt]; linarith [sksD_κ (Coin := Coin) []])
  first
  | exact hb (h1 [] Unit silentP)
  | exact hb (h1 [] (sksD_scope []) Unit silentP)

set_option linter.unreachableTactic false in
set_option linter.unusedTactic false in
/-- **The sampling `Bits` fails** at `sksD`, at `δ = 0`. -/
theorem Bits_fails (H : Msg → Msg) : ¬ Claims.Sampling.Bits H (sksD Coin) := by
  intro h
  have h1 := h 0 (fun _ => [()]) le_rfl (Flock.listWithin_unit _ _ _)
  have hlam := sksD_lam_pos (Coin := Coin) []
  have hc := calls_silentP_accV (Coin := Coin) H []
  have hB := strictBirthday_B_accV (Coin := Coin) H silentP []
  have hE := sksD_expectedCR (Coin := Coin) H (fun _ => [()]) []
  have hp : PrBad H (sksD Coin).V (R H (sksD Coin).commit (sksD Coin).parse (sksD Coin).partition 0)
      (E (sksD Coin) fun _ => [()]) [] silentP = 1 :=
    prBad_accV H _ _ [] silentP (sksD_noR (Coin := Coin) H 0 le_rfl [])
  have h2 : PrBad H (sksD Coin).V (R H (sksD Coin).commit (sksD Coin).parse (sksD Coin).partition 0)
      (E (sksD Coin) fun _ => [()]) [] silentP ≤
        (1 : ℕ) * (2 : ℝ) ^ (-lamLin (κ (sksD Coin) 0 []) (γ (sksD Coin) []) 1 (work (sksD Coin) [])) := by
    first
    | exact h1 [] Unit silentP 1 le_rfl hc hB hE
    | exact h1 [] (sksD_scope []) Unit silentP 1 le_rfl hc hB hE
  rw [hp, Nat.cast_one, one_mul] at h2
  have : (2 : ℝ) ^ (-lamLin (κ (sksD Coin) 0 []) (γ (sksD Coin) []) 1 (work (sksD Coin) [])) < 1 :=
    Real.rpow_lt_one_of_one_lt_of_neg (by norm_num) (by linarith)
  linarith

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- **The sampling `Binding` fails** at `compSD`: two openings of different values at one position pass with no call. -/
theorem Binding_fails : ¬ Binding (compSD Coin) := fun h => by
  obtain ⟨p, hp, -⟩ := h constH [] ⟨0, true, []⟩ ⟨0, false, []⟩ rfl (by decide) rfl rfl
  simp [compSD, accCm, OC.eval] at hp

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- **The sampling `Universal` fails** at `compSD`, whose universal unit returns nothing, at the class `(0, 0, 1)`. -/
theorem Universal_fails : ¬ Universal (compSD Coin) := fun h => by
  have := h ⟨0, 0, 1⟩ ⟨0, [], [0]⟩ [] ⟨rfl, rfl, rfl, by simp, by simp [SizeClass.indexBits]⟩ rfl
  simp [compSD, Prog.eval, PrivateCircuit.eval] at this

omit [Nonempty Coin] in
/-- **The sampling `Complete` fails** at `compSD`, whose reference prover is silent, at its error `δ_c = 0`. -/
theorem Complete_fails : ¬ Claims.Sampling.Complete constH (compSD Coin) :=
  Witness.Complete_fails (compSD Coin).P _ _ ((compSD Coin).δc constH) (x := []) (w := toyWit)
    ⟨toyStmt, toyPart, rfl, rfl, inScope_toy⟩
    ⟨toyStmt, toyPart, rfl, rfl, by show faulty constH accCm toyStmt toyPart toyWit = ∅; rw [toy_faulty, toyWit_val]; rfl⟩
    (show (0 : ℝ) < 1 from zero_lt_one)

omit [Fintype Coin] [Nonempty Coin] in
/-- **The sampling `Nontrivial` fails** at `compSD`, which rejects the recorded run. -/
theorem Nontrivial_fails (H : Msg → Msg) :
    ¬ Claims.Sampling.Nontrivial H (compSD Coin) [⟨[], [], fun _ => default⟩] :=
  fun h => Witness.Nontrivial_fails H h.1

/-- **The sampling `ZK` fails** at the colliding simulator, as the simulator of `compSD`'s masked deployment whose
reference prover is quiet, at every `ε_zk < 1` and every `γ`. -/
theorem ZK_fails (H : Msg → Msg) (u v : Msg) (εzk γ : Msg → ℝ) (x : Msg) (hε : εzk x < 1) :
    ¬ Claims.Sampling.ZK H (ZKMal.quietZD (compSD Coin) u v) εzk γ :=
  ZKMal.not_samplingZK_collSim H _ u v εzk γ x toyWit (fun _ => rfl) rfl
    ⟨toyStmt, toyPart, rfl, rfl, by show faulty constH accCm toyStmt toyPart toyWit = ∅; rw [toy_faulty, toyWit_val]; rfl⟩ Nat.one_pos hε

/-- `compSD` with a masked inner Flock (`ZKMal.toyD`, so one coin). -/
noncomputable def zkSD : Deploy Unit Unit Unit := { compSD Unit with flock := ZKMal.toyD }

/-- **The sampling `ZK` is not met at every masked deployment**, at `ε_zk = 0` and `γ = 0`. -/
theorem ZK_vacuous_refuted (H : Msg → Msg) :
    ¬ ∀ D : ZkDeploy Unit Unit Unit, D.flock.zk = true → Claims.Sampling.ZK H D 0 0 :=
  ZKMal.not_forall_samplingZK H zkSD rfl [] toyWit
    ⟨toyStmt, toyPart, rfl, rfl, by show faulty H accCm toyStmt toyPart toyWit = ∅; rw [toy_faulty, toyWit_val]; rfl⟩
    Nat.one_pos

/-- **The sampling `Efficient` fails** at `effSD` and `p = 0`: its sessions' decoder counts one step. -/
theorem Efficient_fails (H : Msg → Msg) : ¬ Claims.Sampling.Efficient H (effSD Coin) 0 := fun h => by
  have := h.2.2.2.2 []
  simp [effSD, Flock.effD] at this

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- **The sampling `ScopeMatches` fails** at `effSD`, whose scope is empty and whose check takes everything. -/
theorem ScopeMatches_fails : ¬ Claims.Sampling.ScopeMatches (effSD Coin) := fun h => by
  obtain ⟨s, P, hs, -⟩ := (h []).2 rfl
  simp [effSD] at hs

end Claims

/-- The extractor that hashes the empty message once and gives up. -/
def hashE {Coin W : Type} (V : Verifier Coin) : Extractor Coin W V where
  C _ := Unit
  ext _ _ _ := .hash [] fun _ => .ret none

/-- **`ExtractOwnLe` fails** at `hashE`, which hashes once, at a bound of `0`. -/
theorem ExtractOwnLe_fails {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg)
    (V : Verifier Coin) : ¬ ExtractOwnLe H (hashE V : Extractor Coin W V) fun _ => 0 := by
  intro h
  have := h [] Unit silentP () (fun _ => default) ()
  simp [hashE, ownLog] at this

end Claims.Witness.Sampling
