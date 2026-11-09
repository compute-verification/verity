module

public import Proofs.Flock.Claims.Witnesses.Flock
public import Proofs.Flock.Claims.ZK.SamplingNonVacuous
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Algebra.Order.Ring.Star
import Mathlib.Algebra.Order.Star.Real
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.RingTheory.WittVector.IsPoly
import Mathlib.Tactic.ENatToNat
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Tactic.Polynomial.Basic
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.Prob
import Std.Tactic.BVDecide.Normalize.Prop

/-!
# Refutation witnesses: the sampling compiler's claims

Each of the sampling compiler's claims (`Definitions.Claims.Sampling`) fails at a toy deployment. The statement is
`toyStmt`: one input gate, returned, sampled `all`, its partition `toyPart` one proof block `{0}`, in scope
(`inScope_toy`).

* `sksD`: the verifier accepts after one round, and no opening passes (`rejCm`), so every unit is faulty and no
  witness is in the relation at any `δ ≥ 0` or count `t ≤ 1` (`sksD_noR`, `sksD_noRCount`), and its miss at `t = 1` is
  `0` (`sksD_missAt`). The compiled sampled verifier's `KS`, `KSCount` and `Bits` fail at its compiled record
  (`Witnesses.SamplingGeneric`'s `sksL`), which reads these.
* `compSD`: the verifier rejects after one round, and every opening passes (`accCm`), so the opening of the input
  (`toyWit`) leaves no unit faulty. `Complete_fails`, `Nontrivial_fails`, `ZK_fails`
  (`Sampling.ZKMal.not_samplingZK_collSim`, at `Sampling.ZKMal.quietZD`, its masked deployment whose reference prover
  is quiet and whose simulator collides), `Binding_fails` (two openings of different values at one position pass
  with no call), `Universal_fails` (its universal unit returns nothing).
* `effSD`: nothing parses while the check takes everything. `ScopeMatches_fails`.
* `ExtractOwnLe_fails`: an extractor that hashes once, at a bound of `0`.

`FaultyBoundary` is a theorem (`Claims.Sampling.faultyBoundary`), so it has no refutation. `BitsLin` and the scoped
`KSLinOn` are refuted through the compiled sampled verifier's `Bits` and `KS` failing at `sksL`
(`Witnesses.Stated`).

`ZK` reads the masked deployment's own reference prover and simulator, as `Complete` reads the deployment's
(`Sampling.Deploy.P`), so it is not met at every masked deployment (`ZK_vacuous_refuted`, at `zkSD`, `compSD` with a
masked inner Flock).
-/

@[expose] public section

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

theorem ownLog_noHash (H : Msg → Msg) (run : R → Transcript) : ∀ p : RP R α, NoHash p → ownLog H run p = []
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
def accCm : CommitmentScheme Bool := ⟨fun _ _ => .ret true, fun _ _ _ => []⟩

/-- A commitment whose every opening fails, with no call. -/
def rejCm : CommitmentScheme Bool := ⟨fun _ _ => .ret false, fun _ _ _ => []⟩

/-- One input gate, returned. -/
def toyProg : Prog := ⟨[.input], [0]⟩

/-- The program, sampled `all`. -/
def toyStmt : AuditStatement := ⟨[], toyProg, ⟨"", 0, []⟩, ⟨[.all]⟩, []⟩

/-- One class of one stratum of one proof block, `{0}`. -/
def toyPart : Partition := ⟨[[[[0]]]]⟩

/-- The opening of the input's value. -/
def toyWit : Wit := ⟨[⟨0, true, []⟩]⟩

theorem toyPart_blocks : toyPart.blocks = [[0]] := rfl

theorem toy_boundary : toyProg.Boundary [0] 0 = true := by decide

theorem toy_ref (H : Msg → Msg) (Cm : CommitmentScheme Bool) (w : Wit) (ι : ℕ → Bool) :
    toyProg.refIn [0] (w.val H Cm toyStmt) ι 0 = w.val H Cm toyStmt 0 := by
  simp [Prog.refIn, toy_boundary]

theorem toy_runIn (ref : ℕ → Option Bool) : toyProg.runIn ref [0] 0 = ref 0 := by
  rw [Prog.runIn]
  rfl

/-- The unit is correct exactly when the input's value is opened. -/
theorem toy_correct_iff (H : Msg → Msg) (Cm : CommitmentScheme Bool) (w : Wit) (ι : ℕ → Bool) :
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

theorem toy_faulty_iff (H : Msg → Msg) (Cm : CommitmentScheme Bool) (w : Wit) :
    Faulty H Cm toyStmt toyPart w 0 ↔ w.val H Cm toyStmt 0 = none := by
  unfold Faulty
  rw [toyPart_blocks]
  simp only [List.getElem?_cons_zero, Option.some.injEq, exists_eq_left']
  rw [show toyStmt.prog = toyProg from rfl]
  simp only [toy_correct_iff, exists_const]
  generalize w.val H Cm toyStmt 0 = o
  cases o <;> simp

theorem toy_faulty (H : Msg → Msg) (Cm : CommitmentScheme Bool) (w : Wit) :
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
  simp [Wit.val, toyWit, CommitmentScheme.Opens, accCm, OC.eval]

theorem rejCm_val (H : Msg → Msg) (w : Wit) : w.val H rejCm toyStmt 0 = none := by
  simp [Wit.val, CommitmentScheme.Opens, rejCm, OC.eval]

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

theorem sksD_k (x : Msg) : (sksD Coin).k x = 2 ^ 10 * 3 := rfl

theorem slack_pos : 0 < slack := by unfold slack; positivity

theorem slack_le_two : slack ≤ 2 := by
  unfold slack
  calc (2 : ℝ) ^ ((2 : ℝ) ^ (-(slackExp : ℝ))) ≤ 2 ^ (1 : ℝ) :=
        Real.rpow_le_rpow_of_exponent_le (by norm_num)
          (Real.rpow_le_one_of_one_le_of_nonpos (by norm_num) (neg_nonpos.2 (Nat.cast_nonneg _)))
    _ = 2 := Real.rpow_one 2

theorem exp_neg_le_inv {y : ℝ} (hy : 0 ≤ y) : Real.exp (-y) ≤ (y + 1)⁻¹ := by
  rw [Real.exp_neg]
  exact inv_anti₀ (by linarith) (Real.add_one_le_exp y)

theorem sksD_outer (x : Msg) :
    2 * ((sksD Coin).k x : ℝ) / ((sksD Coin).cap x : ℝ) + Real.exp (-((sksD Coin).k x : ℝ) / 8) ≤ 1 / 50 := by
  have hk : ((sksD Coin).k x : ℝ) = 2 ^ 10 * 3 := by rw [sksD_k]; push_cast; ring
  have hc : ((sksD Coin).cap x : ℝ) = 2 ^ 128 * (2 ^ 10 * 3) := by
    simp only [Deploy.cap, sksD_k]; push_cast; ring
  rw [hk, hc, neg_div]
  have he := exp_neg_le_inv (show (0 : ℝ) ≤ 2 ^ 10 * 3 / 8 by positivity)
  have h1 : (2 ^ 10 * 3 / 8 + 1 : ℝ)⁻¹ ≤ 1 / 100 := by
    rw [inv_le_comm₀ (by positivity) (by norm_num)]; norm_num
  have h2 : 2 * (2 ^ 10 * 3 : ℝ) / (2 ^ 128 * (2 ^ 10 * 3)) ≤ 1 / 100 := by
    rw [div_le_iff₀ (by positivity)]; norm_num
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

/-! ## The claims -/

section Claims

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

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
