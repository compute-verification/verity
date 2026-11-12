module

public import Proofs.Flock.Claims.Witnesses.Sampling
public import Proofs.Flock.Claims.Witnesses.SamplingGeneric
public import Proofs.Flock.SamplingGeneric.Deployed
public import Proofs.Flock.SamplingGeneric.TwoAt
public import Definitions.NetworkCertifier.TapRows
public import Definitions.NetworkCertifier.Witness
public import Definitions.NetworkCertifier.Guarantees.SampledAudit
public import Definitions.NetworkCertifier.Guarantees.PooledLeak
public import Definitions.NetworkCertifier.PooledLeak
public import Definitions.NetworkCertifier.SampledCharge
public import Definitions.NetworkCertifier.SampledReg
public import Proofs.Flock.Claims.DeployCode
public import Proofs.Flock.Claims.Sampling.Leaf
public import Proofs.Flock.Claims.Sampling.Select.Scope
public import Proofs.Flock.Claims.Sampling.Session
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
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
import Proofs.Flock.Claims.KS.ZkExtend
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.Sampling.Assembled
import Proofs.Flock.SamplingGeneric
import Proofs.NetworkCertifier.SampledAuditAccepted
import Proofs.NetworkCertifier.SampledLeakRegPlain
import Std.Tactic.BVDecide.Normalize.Bool

/-!
# The honest and nontrivial lemmas of the network certifier's acceptance guarantees

Each guarantee that says what an accepted run establishes has two lemmas (the two-lemma rule):
`G_honest`, its premises at an honest instance, and `G_nontrivial`, its conclusion false at an instance whose verifier
accepts everything. Each builds its instance and takes no hypothesis.

* `SampledAuditAccepted`, `SampledLeakAccepted`, `SampledLeakRegistered`, `PeriodLeakAccepted` (and
  `Accounting.Guarantees.Bridge.CommOfPeriodLeakAccepted`). Their premise `Compiled.KS` holds at SHA-512 at the
  deployed compiled form over the masked Flock record at the deployed relation, at a toy code that parses no statement,
  accepts no run and opens no session (`SamplingDeploy.toyCode`): `ks_toy`, from the compiler at Flock's KS there
  (`SamplingDeploy.ks_compiled`, `FlockDeploy.KS.zk_KS_untrim`) and the code's facts, which hold there. So does the
  code's commitment's binding (`binds_deployOf`). Their acceptance conclusion fails at a compiled verifier that accepts
  everything: `SampledAuditAccepted` at `sksL`, where no opening passes, so the one unit is faulty and no witness is in
  `RelS`; the leak bounds at `leakL`, `sksL` at a statement whose program negates its input. There, with nothing
  opened, the trace holds `false` at the negation, which the circuit computes `true` from the ingress the trace
  carries; the one unit is sampled `all`, so no possible incorrect set holds it, and no witness is in `RLeak`.
  `PeriodLeakAccepted`'s at `certL`, `leakL` under a commitment whose openings of `true` pass (`trueCm`): the network
  certifier's openings (`certOps`) register a frame at the input `true`, where the circuit's negation holds none, and no
  witness is in `RLeakCertifier`. These statements read the toy verifiers of `Witnesses.SamplingGeneric` (`sksL`,
  `leakL`, `certL`) and the layouts and openings of `Definitions.NetworkCertifier.Witness` (`trivLay`, `notLay`,
  `certOps`), and no definition of this module: a statement the lock lists reads no proof-only definition outside the
  excepted toys.
* `PooledLeakAccepted`. Its two inputs, each window's escape bound and its leak relation's error, hold with no error at
  one window audited by the compiled form at the toy code, at `q = 1` with the window's one proof block incorrect: the
  code accepts no run, so the audit accepts with probability `0`. Its other hypotheses hold there as well, the network
  certifier running its code on no connection, so its bound reads there. At a code that accepts a run the two inputs
  stay hypotheses, which the lemma does not discharge. At one window whose audit always accepts, `δ = 1/2`, `Fmax = 1`
  and no error, its observations empty where the horizon is one step, no window meets its leak relation, and the
  escape bound reads `1 ≤ 1/2`.
* `PooledIncorrectBlocksLe`. Its sample hypotheses hold at the uniform `k`-subset sample (`uniform_sample`), and its soundness
  hypothesis at `ε = 0` when no unit is wrong. At one window of one unit, that unit wrong, an audit that always accepts,
  `ε = 0` and `q = F = 1`, its bound reads `1 ≤ 0`.

The count conjuncts of the leak bounds read no verifier: they hold at every deployment, the accepting one included.
-/

@[expose] public section

namespace NetTiming.Witness

open Claims Claims.Sampling Claims.Witness Claims.Witness.Sampling
open Claims.SamplingGeneric (Compiled)
open Claims.Witness.SamplingGeneric (toyA sksL trueCm sksL_scope sksL_κ sksL_bound_fails noHash_compiledE bext_pr
  notProg notStmt notPart leakL certL)

local notation "relD" => _root_.Flock.Readable.relation
local notation "shaD" => FlockDeploy.sha512
local notation "toyL" =>
  SamplingDeploy.compiled (FlockDeploy.deployZk _root_.Flock.Readable.relation) SamplingDeploy.toyCode
    (fun _ _ _ => [])

/-! ## At the toy code -/

section Honest

set_option synthInstance.maxSize 1024 in
/-- **`Compiled.KS` at the toy code** (`SamplingDeploy.toyCode`), the deployed compiled form over the masked Flock
record at the deployed relation, at SHA-512: the compiler at Flock's KS there (`FlockDeploy.KS.zk_KS_untrim`), its
decoder within the radius (`listWithin_deployZk`), the code's commitment binding (`binds_deployOf`), and the code's
facts, which hold since it parses no statement, accepts no run and opens no session. -/
theorem ks_toy : (toyL).KS shaD :=
  SamplingDeploy.ks_compiled (FlockDeploy.deployZk relD) SamplingDeploy.toyCode (fun _ _ _ => []) shaD
    FlockDeploy.KS.zk_KS_untrim (listWithin_deployZk relD)
    (binds_deployOf (FlockDeploy.deployZk relD) SamplingDeploy.toyCode shaD)
    (fun _ _ _ hx => by
      change (none : Option AuditStatement) = some _ at hx
      simp at hx)
    (fun x _ P ω r hacc => absurd ((SamplingDeploy.run_rejV2_out shaD P x ω r).symm.trans hacc) (by simp))
    (fun _ _ _ _ _ _ _ => rfl)
    (fun _ _ _ ⟨p, hp⟩ => by
      change ([] : List ℕ)[p.val]? = some _ at hp
      simp at hp)
    (fun _ _ _ _ _ _ ⟨p, hp⟩ => by
      change ([] : List ℕ)[p.val]? = some _ at hp
      simp at hp)
    (fun _ _ _ _ _ hs => by
      change (none : Option (Msg × Seen)) = some _ at hs
      simp at hs)
    (fun _ _ _ _ _ _ _ _ _ _ _ hx => by
      change (none : Option AuditStatement) = some _ at hx
      simp at hx)

set_option synthInstance.maxSize 1024 in
/-- **`SampledAuditAccepted`'s premise at the toy code**, which is `SampledLeakAccepted`'s too: `Compiled.KS`
(`ks_toy`). The code accepts no run (`SamplingDeploy.run_rejV2_out`) and parses no statement, so the facts behind
`Compiled.KS` hold there with nothing to check. At a code that accepts a run `Compiled.KS` stays the guarantee's
hypothesis, which this lemma does not discharge. -/
theorem SampledAuditAccepted_honest : (toyL).KS shaD := ks_toy

set_option synthInstance.maxSize 1024 in
/-- **`PeriodLeakAccepted`'s premises at the toy code**, which are `SampledLeakRegistered`'s and
`CommOfPeriodLeakAccepted`'s too: `Compiled.KS` (`ks_toy`), and the code's commitment binds (`binds_deployOf`). The
code accepts no run (`SamplingDeploy.run_rejV2_out`) and parses no statement, so the facts behind `Compiled.KS` hold
there with nothing to check. At a code that accepts a run `Compiled.KS` stays the guarantee's hypothesis, which this
lemma does not discharge; the binding holds at every code. -/
theorem PeriodLeakAccepted_honest : (toyL).KS shaD ∧ (toyL).commit.Binds shaD :=
  ⟨ks_toy, binds_deployOf (FlockDeploy.deployZk relD) SamplingDeploy.toyCode shaD⟩

/-- **`PooledLeakAccepted`'s two inputs at the toy code**: over one window, audited by the compiled form at the toy
code against a prover that sends nothing, with its one proof block incorrect and its observations empty where the
horizon is one step, at `q = 1`. The code accepts no run (`SamplingDeploy.run_rejV2_out`), so the audit escapes with
probability `0 = (1 - q)^1`, and accepts outside its leak relation with probability `0`: each input holds with no
error. At a code that accepts a run the two inputs stay the guarantee's hypotheses, which this lemma does not
discharge. -/
theorem PooledLeakAccepted_honest :
    (∀ z ∈ (Set.univ : Set Unit), ∀ (_ : Fin 1) (_ : Fin 1 → Fin ((toyL).V.k []) → FlockDeploy.Coin),
        Pr (fun r : Fin ((toyL).V.k []) → FlockDeploy.Coin =>
          (Claims.run shaD silentP (toyL).V [] () r).out = .acc) ≤ (1 - 1) ^ ({0} : Finset ℕ).card + 0) ∧
      ∀ z ∈ (Set.univ : Set Unit), ∀ i : Fin 1,
        Pr (fun ω : Fin 1 → Fin ((toyL).V.k []) → FlockDeploy.Coin =>
          (Claims.run shaD silentP (toyL).V [] () (ω i)).out = .acc ∧
            ¬LeakAt (K := 1) (fun _ => notLay) (fun _ => 1) (fun _ => ∅) 1 notStmt notPart (fun _ => false)
              (fun _ _ => [false]) {0} []) ≤ 0 := by
  have hrej : ∀ r : Fin ((toyL).V.k []) → FlockDeploy.Coin,
      ¬(Claims.run shaD silentP (toyL).V [] () r).out = .acc := fun r hacc =>
    absurd ((SamplingDeploy.run_rejV2_out shaD silentP [] () r).symm.trans hacc) (by simp)
  refine ⟨fun _ _ _ _ => le_of_eq ?_, fun _ _ _ => le_of_eq ?_⟩
  · rw [Pr_congr fun r => iff_false_intro (hrej r), Pr_false]
    norm_num
  · rw [Pr_congr fun ω => iff_false_intro (by exact fun h => hrej _ h.1)]
    exact Pr_false

/-- At that instance `PooledLeakAccepted`'s other hypotheses hold as well, the network certifier running its code on no
connection, so its bound reads there: at `δ = 1/2` and `Fmax = 1`, with no error. -/
example (h : Guarantees.PooledLeakAccepted) :
    Pr (fun ω : Fin 1 → Fin ((toyL).V.k []) → FlockDeploy.Coin =>
        (∀ i : Fin 1, (Claims.run shaD silentP (toyL).V [] () (ω i)).out = .acc) ∧
          ¬PooledGood (K := 1) (fun _ => notLay) (fun _ => 1) (fun _ => ∅) 1 (fun _ : Fin 1 => notStmt)
            (fun _ => notPart) (fun _ _ => false) (fun _ _ _ _ => [false]) 1 (fun _ => {0}) (fun _ => [])) ≤
      1 / 2 + ∑ _ : Fin 1, (0 : ℝ) + ∑ _ : Fin 1, (0 : ℝ) := by
  let f : RecordFormat (Fin 0) Unit (Fin 1 → List (Obs 1 1 1)) :=
    ⟨fun _ _ _ _ => [], fun _ _ c => c.elim0, fun _ => True, fun _ _ _ => funext fun c => c.elim0⟩
  have hF : Real.log (1 / (1 / 2 : ℝ)) ≤ 1 * ((1 : ℕ) : ℝ) :=
    (Real.log_le_sub_one_of_pos (by norm_num : (0 : ℝ) < 1 / (1 / 2))).trans (by norm_num)
  exact (h (K := 1) (fun _ => notLay) (fun _ => 1) (fun _ => ∅) 1 (fun _ : Fin 1 => notStmt) (fun _ => notPart)
    (fun _ _ => false) (fun _ _ _ _ => [false]) (fun _ => 1) (faultyBits notStmt.prog notPart {0})
    (fun _ v hv => by obtain rfl : v = 0 := Nat.lt_one_iff.1 hv; exact le_rfl)
    (Ω := Fin ((toyL).V.k []) → FlockDeploy.Coin)
    (fun (_ : Unit) (c : Fin 0) => c.elim0) f () (fun _ p => recordingCode f p.1 fun c => c.elim0) (fun _ => rfl)
    (fun c => c.elim0) Set.univ (fun _ _ c => c.elim0) (fun _ _ => trivial)
    (Set.univ : Set Unit) (fun _ _ => ⟨1, 0, fun c => c.elim0, fun c => c.elim0⟩) (fun _ _ => ())
    (fun _ _ _ => Set.mem_univ _) (fun _ _ => ()) (fun _ _ _ => rfl)
    (fun _ _ _ r => (Claims.run shaD silentP (toyL).V [] () r).out = .acc) (fun _ _ _ _ _ => rfl)
    (fun _ _ _ => {0}) (fun _ _ _ _ _ => rfl)
    (fun _ _ _ => Finset.singleton_subset_iff.2 (Finset.mem_range.2 Nat.one_pos))
    (fun _ => 0) (fun _ => 0) (fun _ => le_rfl) 1 zero_le_one le_rfl
    PooledLeakAccepted_honest.1 PooledLeakAccepted_honest.2 (1 / 2) (by norm_num) 1 hF).1 () (Set.mem_univ _)

end Honest

/-! ## At the accepting verifier -/

/-- One step: no frame on the link, every status complete. -/
def obs0 : Obs 1 1 1 := ⟨fun _ _ => none, 0, fun _ => .complete⟩

/-- One step: a frame on the link, every status complete. -/
def obs1 : Obs 1 1 1 := ⟨fun _ _ => some [], 0, fun _ => .complete⟩

theorem inScope_not : Claims.Sampling.InScope notPart notStmt := by
  refine ⟨⟨fun g x hx i hi => ?_, by decide⟩,
    ⟨by decide, by decide, by decide, by decide, by decide⟩, rfl, ?_⟩
  · rcases g with _ | _ | g
    · simp [notStmt, notProg] at hx
      subst hx
      simp [BGate.reads] at hi
    · simp [notStmt, notProg] at hx
      subst hx
      simp [BGate.reads] at hi
      omega
    · simp [notStmt, notProg] at hx
  · simp [notStmt, notPart, Claims.Sampling.OneStage.WellFormed]

theorem notLay_fits : notLay.Fits notProg 1 := by
  refine ⟨fun t _ ℓ σ g hg => ?_, fun t _ g hg => ?_, fun t _ t' _ htt' => absurd htt' (by omega)⟩
  · simp [notLay, slotGates] at hg
    subst hg
    decide
  · simp [notLay] at hg
    subst hg
    rfl

theorem rejCm_val_none (H : Msg → Msg) (s : Claims.Sampling.AuditStatement) (w : Claims.Sampling.Wit) (g : ℕ) :
    w.val H rejCm s g = none := by
  simp [Wit.val, CommitmentScheme.Opens, rejCm, OC.eval]

/-- With nothing held and the input `false`, the circuit computes the negation `true`. -/
theorem notProg_val (inp : ℕ → Bool) (h0 : inp 0 = false) : valOver notProg inp (fun _ => none) 1 = true := by
  have hg0 : notProg.gates[0]? = some .input := rfl
  have hg1 : notProg.gates[1]? = some (.not 0) := rfl
  have hv0 : valOver notProg inp (fun _ => none) 0 = false := by
    rw [valOver]
    simp [hg0, BGate.val, h0]
  rw [valOver]
  simp [hg1, BGate.val, hv0]

/-- With nothing held and the input `true`, the circuit computes the negation `false`. -/
theorem notProg_val_true (inp : ℕ → Bool) (h0 : inp 0 = true) : valOver notProg inp (fun _ => none) 1 = false := by
  have hg0 : notProg.gates[0]? = some .input := rfl
  have hg1 : notProg.gates[1]? = some (.not 0) := rfl
  have hv0 : valOver notProg inp (fun _ => none) 0 = true := by
    rw [valOver]
    simp [hg0, BGate.val, h0]
  rw [valOver]
  simp [hg1, BGate.val, hv0]

section Toy

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem leakL_scopeAcc (x : Msg) : ScopeAcc (leakL Coin) (fun _ => notLay) (fun _ => 1) x :=
  ⟨⟨notStmt, notPart, rfl, rfl, inScope_not⟩, notStmt, rfl, notLay_fits⟩

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem certL_scopeAcc (x : Msg) : ScopeAcc (certL Coin) (fun _ => notLay) (fun _ => 1) x :=
  ⟨⟨notStmt, notPart, rfl, rfl, inScope_not⟩, notStmt, rfl, notLay_fits⟩

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem leakL_κ (x : Msg) : (leakL Coin).κ 0 x < 1 / 3 :=
  sksL_κ (Coin := Coin) x

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem certL_κ (x : Msg) : (certL Coin).κ 0 x < 1 / 3 :=
  sksL_κ (Coin := Coin) x

theorem leakL_bext (H : Msg → Msg) (x : Msg) : (Bext H (leakL Coin).V (leakL Coin).E Unit silentP x).Pr H = 0 :=
  bext_pr H (leakL Coin) (fun _ _ _ => rfl) (fun _ => Nat.one_pos) Unit silentP x

theorem certL_bext (H : Msg → Msg) (x : Msg) : (Bext H (certL Coin).V (certL Coin).E Unit silentP x).Pr H = 0 :=
  bext_pr H (certL Coin) (fun _ _ _ => rfl) (fun _ => Nat.one_pos) Unit silentP x

theorem leakL_breg (H : Msg → Msg) (x : Msg) :
    (Breg H (leakL Coin).V (leakL Coin).E (leakL Coin).commit (leakL Coin).parse (fun _ => []) Unit silentP x).Pr H =
      0 := by
  unfold Finder.Pr
  rw [Pr_congr fun s => iff_false_intro (by
    simp only [Breg, ownLog_noHash _ _ _ (noHash_compiledE (leakL Coin) (fun _ _ _ => rfl) _ _ _)]
    simp [regLog, leakL, findColl])]
  exact Pr_false

theorem certL_btap (H : Msg → Msg) (x : Msg) :
    (Btap H (certL Coin).V (certL Coin).E (certL Coin).commit (certL Coin).parse (fun _ => certOps) Unit silentP
      x).Pr H = 0 := by
  unfold Finder.Pr
  rw [Pr_congr fun s => iff_false_intro (by
    simp only [Btap, ownLog_noHash _ _ _ (noHash_compiledE (certL Coin) (fun _ _ _ => rfl) _ _ _)]
    simp [certifierLog, certL, leakL, certOps, trueCm, OC.eval, findColl])]
  exact Pr_false

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- **No witness is in `RLeak`** at `leakL`, at one step at `δ = 0`, whatever the inputs `As` admits: the step the
trace carries has no frame, where the circuit, the negation of the ingress the trace carries, holds one, and the only
possible incorrect set is empty, since the one unit is sampled `all`. -/
theorem leakL_noRLeak (H : Msg → Msg) (As : Claims.Sampling.AuditStatement → (ℕ → Bool) → Set (ℕ → Bool)) (x : Msg)
    (w : Claims.Sampling.Wit) :
    ¬ RLeak H (leakL Coin).commit (leakL Coin).parse (leakL Coin).partition 1 1 1 (fun _ => notLay) (fun _ => 1) 0 As 1
      x w := by
  rintro ⟨s, P, hs, hP, h⟩
  simp only [leakL, Option.some.injEq] at hs hP
  subst hs hP
  have hval := rejCm_val_none H notStmt w
  have hcar : Carries notLay notProg (w.val H rejCm notStmt) (fun _ _ => [false]) 1 [obs0] := by
    intro t ht
    obtain rfl : t = 0 := by omega
    refine ⟨Nat.one_pos, fun ℓ => ?_, ?_⟩
    · funext σ
      simp [obs0, rowOf, notLay, traceVal, Prog.read, notProg, hval]
    · simp [notLay, traceVal, Prog.read, notProg, hval]
  have hcov : Covered 1 [obs0] := by
    intro t ht _ κ
    obtain rfl : t = 0 := by omega
    rfl
  obtain ⟨-, -, F, hF, a, -, free, hfree, hacc⟩ := h (fun _ _ => [false]) [obs0] hcar hcov rfl (by simp [obs0])
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

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- **No witness is in `RLeakCertifier`** at `certL`, at one step at `δ = 0`, whatever the inputs `As` admits: the
network certifier's openings (`certOps`) register a frame at the input `true`, where the circuit, the negation of that
ingress, holds none, and the only possible incorrect set is empty, since the one unit is sampled `all`. -/
theorem certL_noWit (H : Msg → Msg) (As : Claims.Sampling.AuditStatement → (ℕ → Bool) → Set (ℕ → Bool))
    (x : Msg) (w : Claims.Sampling.Wit) :
    ¬ RLeakCertifier H (certL Coin).commit (certL Coin).parse (certL Coin).partition 1 1 1 (fun _ => notLay)
      (fun _ => 1) 0 As 1 (fun _ => certOps) x w := by
  rintro ⟨s, P, hs, hP, h⟩
  simp only [certL, leakL, Option.some.injEq] at hs hP
  subst hs hP
  have hop : CertifierRegisters H trueCm notLay 1 (fun _ _ => [true]) notStmt certOps [obs1] := by
    refine ⟨fun q hq => ?_, fun t ht => ?_⟩
    · simp only [certOps, List.mem_cons, List.not_mem_nil, or_false] at hq
      rcases hq with rfl | rfl <;> rfl
    · obtain rfl : t = 0 := by omega
      refine ⟨fun ℓ σ g hg => ?_, fun g hg => ?_, Nat.one_pos, fun ℓ => ?_, ?_⟩
      · simp [notLay, slotGates] at hg
        subst hg
        exact ⟨fun b hb => by simp [notStmt, notProg] at hb, ⟨1, true, []⟩, by simp [certOps], rfl⟩
      · simp [notLay] at hg
        subst hg
        exact ⟨⟨0, true, []⟩, by simp [certOps], rfl⟩
      · funext σ
        simp [obs1, rowOf, notLay, certifierVal, certOps]
      · simp [notLay, certifierVal, certOps]
  have hcov : Covered 1 [obs1] := by
    intro t ht _ κ
    obtain rfl : t = 0 := by omega
    rfl
  obtain ⟨-, -, F, hF, a, -, free, hfree, hacc⟩ := h (fun _ _ => [true]) [obs1] hop hcov rfl (by simp [obs1])
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
  have hin : notLay.inputOf 1 [[true]] a 0 = true := rfl
  have h1 : (some [] : Option (List Bool)) =
      rowOf (valOver notProg (notLay.inputOf 1 [[true]] a) (fun _ => none)) (notLay.egress 0 0) 0 :=
    congrFun (congrFun hrows 0) 0
  have hrow : rowOf (valOver notProg (notLay.inputOf 1 [[true]] a) (fun _ => none)) (notLay.egress 0 0) 0 =
      if valOver notProg (notLay.inputOf 1 [[true]] a) (fun _ => none) 1 then
        some (([] : List ℕ).map (valOver notProg (notLay.inputOf 1 [[true]] a) (fun _ => none)))
      else none := rfl
  rw [hrow, notProg_val_true _ hin] at h1
  simp at h1

/-- `KSLin`'s bound fails at `leakL`'s silent prover, at a relation with no witness at `x` and a bound `κ' < 1`. -/
theorem leakL_bound_fails (H : Msg → Msg) (Rx : Rel Claims.Sampling.Wit) (κ' : ℝ) (x : Msg) (hR : ∀ w, ¬ Rx x w)
    (hκ : κ' < 1) :
    ¬ PrBad H (leakL Coin).V Rx (leakL Coin).E x silentP ≤
      κ' + Real.sqrt ((leakL Coin).γ x * PrB H (leakL Coin).V silentP x) +
        1 * (Bext H (leakL Coin).V (leakL Coin).E Unit silentP x).Pr H := by
  intro h
  have hp : PrBad H (leakL Coin).V Rx (leakL Coin).E x silentP = 1 := prBad_accV H Rx _ x silentP hR
  have hb : PrB H (leakL Coin).V silentP x = 0 := prB_accV H silentP x
  simp only [hp, hb, mul_zero, Real.sqrt_zero, add_zero, leakL_bext] at h
  linarith

/-- `SampledLeakRegistered`'s bound fails at `leakL`'s silent prover, at a relation with no witness at `x`, a bound
`κ' < 1` and no registration. -/
theorem leakL_reg_bound_fails (H : Msg → Msg) (Rx : Rel Claims.Sampling.Wit) (κ' : ℝ) (x : Msg) (hR : ∀ w, ¬ Rx x w)
    (hκ : κ' < 1) :
    ¬ PrBad H (leakL Coin).V Rx (leakL Coin).E x silentP ≤
      κ' + Real.sqrt ((leakL Coin).γ x * PrB H (leakL Coin).V silentP x) +
        (Bext H (leakL Coin).V (leakL Coin).E Unit silentP x).Pr H +
        (Breg H (leakL Coin).V (leakL Coin).E (leakL Coin).commit (leakL Coin).parse (fun _ => []) Unit silentP
          x).Pr H := by
  intro h
  have hp : PrBad H (leakL Coin).V Rx (leakL Coin).E x silentP = 1 := prBad_accV H Rx _ x silentP hR
  have hb : PrB H (leakL Coin).V silentP x = 0 := prB_accV H silentP x
  simp only [hp, hb, mul_zero, Real.sqrt_zero, add_zero, leakL_bext, leakL_breg] at h
  linarith

/-- `PeriodLeakAccepted`'s bound fails at `certL`'s silent prover, at a relation with no witness at `x` and a bound
`κ' < 1`. -/
theorem certL_bound_fails (H : Msg → Msg) (Rx : Rel Claims.Sampling.Wit) (κ' : ℝ) (x : Msg) (hR : ∀ w, ¬ Rx x w)
    (hκ : κ' < 1) :
    ¬ PrBad H (certL Coin).V Rx (certL Coin).E x silentP ≤
      κ' + Real.sqrt ((certL Coin).γ x * PrB H (certL Coin).V silentP x) +
        (Bext H (certL Coin).V (certL Coin).E Unit silentP x).Pr H +
        (Btap H (certL Coin).V (certL Coin).E (certL Coin).commit (certL Coin).parse (fun _ => certOps) Unit silentP
          x).Pr H := by
  intro h
  have hp : PrBad H (certL Coin).V Rx (certL Coin).E x silentP = 1 := prBad_accV H Rx _ x silentP hR
  have hb : PrB H (certL Coin).V silentP x = 0 := prB_accV H silentP x
  simp only [hp, hb, mul_zero, Real.sqrt_zero, add_zero, certL_bext, certL_btap] at h
  linarith

end Toy

/-- **`SampledAuditAccepted`'s conclusion fails at the accepting verifier** `sksL`, at a horizon of no step: no opening
passes, so no witness is in `RelS`, and `RAcc` reads `RelS`. -/
theorem SampledAuditAccepted_nontrivial : ¬ KSAccepted shaD (sksL Bool) 1 1 1 (fun _ => trivLay) (fun _ => 0) := by
  intro h
  have hscope : ScopeAcc (sksL Bool) (fun _ => trivLay) (fun _ => 0) [] :=
    ⟨sksL_scope [], toyStmt, rfl, fun t ht => absurd ht (Nat.not_lt_zero t), fun t ht => absurd ht (Nat.not_lt_zero t),
      fun t ht => absurd ht (Nat.not_lt_zero t)⟩
  exact sksL_bound_fails (Coin := Bool) shaD _ _ []
    (fun w ⟨s, P, hs, hP, hrel, _⟩ => sksD_noR (Coin := Bool) shaD 0 le_rfl [] w ⟨s, P, hs, hP, hrel⟩)
    (by linarith [sksL_κ (Coin := Bool) []]) (h 0 le_rfl [] hscope Unit silentP)

/-- **`SampledLeakAccepted`'s acceptance conclusion fails at the accepting verifier** `leakL`, at one step at `δ = 0`,
with no input registered (`leakL_noRLeak`). -/
theorem SampledLeakAccepted_nontrivial :
    ¬ KSLinOn shaD (leakL Bool).V (ScopeAcc (leakL Bool) (fun _ => notLay) (fun _ => 1))
      (RLeak shaD (leakL Bool).commit (leakL Bool).parse (leakL Bool).partition 1 1 1 (fun _ => notLay) (fun _ => 1) 0
        (fun s => AsReg s.prog notLay 1 ∅) 1)
      (leakL Bool).E ((leakL Bool).κ 0) (leakL Bool).γ (fun _ => 1) (Bext shaD (leakL Bool).V (leakL Bool).E) :=
  fun h => leakL_bound_fails shaD _ _ [] (leakL_noRLeak shaD _ []) (by linarith [leakL_κ (Coin := Bool) []])
    (h [] (leakL_scopeAcc []) Unit silentP)

/-- **`SampledLeakRegistered`'s acceptance conclusion fails at the accepting verifier** `leakL`, at one step at
`δ = 0`, with no input registered: at the statement `[]`, its silent prover's bound fails. -/
theorem SampledLeakRegistered_nontrivial :
    ¬ PrBad shaD (leakL Bool).V
        (RLeak shaD (leakL Bool).commit (leakL Bool).parse (leakL Bool).partition 1 1 1 (fun _ => notLay)
          (fun _ => 1) 0 (fun s _ => AsReg s.prog notLay 1 ∅ (regVal [])) 1)
        (leakL Bool).E [] silentP ≤
      (leakL Bool).κ 0 [] + Real.sqrt ((leakL Bool).γ [] * PrB shaD (leakL Bool).V silentP []) +
        (Bext shaD (leakL Bool).V (leakL Bool).E Unit silentP []).Pr shaD +
        (Breg shaD (leakL Bool).V (leakL Bool).E (leakL Bool).commit (leakL Bool).parse (fun _ => []) Unit silentP
          []).Pr shaD :=
  leakL_reg_bound_fails shaD _ _ [] (leakL_noRLeak shaD _ []) (by linarith [leakL_κ (Coin := Bool) []])

/-- **`PeriodLeakAccepted`'s acceptance conclusion fails at the accepting verifier** `certL`, at one step at `δ = 0`,
with no input registered: at the statement `[]`, its silent prover's bound fails (`certL_noWit`). -/
theorem PeriodLeakAccepted_nontrivial :
    ¬ PrBad shaD (certL Bool).V
        (RLeakCertifier shaD (certL Bool).commit (certL Bool).parse (certL Bool).partition 1 1 1 (fun _ => notLay)
          (fun _ => 1) 0 (fun s => AsReg s.prog notLay 1 ∅) 1 (fun _ => certOps))
        (certL Bool).E [] silentP ≤
      (certL Bool).κ 0 [] + Real.sqrt ((certL Bool).γ [] * PrB shaD (certL Bool).V silentP []) +
        (Bext shaD (certL Bool).V (certL Bool).E Unit silentP []).Pr shaD +
        (Btap shaD (certL Bool).V (certL Bool).E (certL Bool).commit (certL Bool).parse (fun _ => certOps) Unit silentP
          []).Pr shaD :=
  certL_bound_fails shaD _ _ [] (certL_noWit shaD _ []) (by linarith [certL_κ (Coin := Bool) []])

/-! ## The monitoring period -/

/-- **`PooledLeakAccepted`'s escape bound fails at an audit that always accepts**: one window, `δ = 1/2`, `Fmax = 1`, no
error, and the window's observations empty where its horizon is one step, so its good event fails at every sample. -/
theorem PooledLeakAccepted_nontrivial :
    ¬ Pr (fun _ : Fin 1 → Unit => (∀ _ : Fin 1, True) ∧
        ¬PooledGood (K := 1) (fun _ => notLay) (fun _ => 1) (fun _ => ∅) 1
          (fun _ : Fin 1 => notStmt) (fun _ => notPart) (fun _ _ => false) (fun _ _ _ _ => [false]) 1 (fun _ => ∅)
          (fun _ => [])) ≤
      (1 / 2 : ℝ) + ∑ _ : Fin 1, (0 : ℝ) + ∑ _ : Fin 1, (0 : ℝ) := by
  have hp : (fun _ : Fin 1 → Unit => (∀ _ : Fin 1, True) ∧
      ¬PooledGood (K := 1) (fun _ => notLay) (fun _ => 1) (fun _ => ∅) 1
        (fun _ : Fin 1 => notStmt) (fun _ => notPart) (fun _ _ => false) (fun _ _ _ _ => [false]) 1 (fun _ => ∅)
        (fun _ => [])) = fun _ => True :=
    funext fun _ => propext (iff_true_intro ⟨fun _ => trivial, fun h => absurd (h.2 0).2.1 (by simp)⟩)
  rw [hp]
  norm_num [Pr, Claims.prUniform]

/-- **`PooledIncorrectBlocksLe`'s sample hypotheses at the honest instance**: the uniform `k`-subset sample samples `k` units
and each `k`-subset with share `1 / C(n, k)`. -/
theorem uniform_sample (n k : ℕ) :
    (∀ x : {S : Finset (Fin n) // S.card = k}, x.1.card = k) ∧
      ∀ S : Finset (Fin n), S.card = k →
        (Finset.univ.filter fun x : {S : Finset (Fin n) // S.card = k} => x.1 = S).card * n.choose k =
          Fintype.card {S : Finset (Fin n) // S.card = k} := by
  refine ⟨fun x => x.2, fun S hS => ?_⟩
  have h1 : (Finset.univ.filter fun x : {S : Finset (Fin n) // S.card = k} => x.1 = S) = {⟨S, hS⟩} := by
    ext x
    simp [Subtype.ext_iff]
  rw [h1, Finset.card_singleton, one_mul, Fintype.card_finset_len, Fintype.card_fin]

/-- **`PooledIncorrectBlocksLe`'s soundness hypothesis at the honest instance**: at the uniform `k`-subset sample
(`uniform_sample`) with no unit wrong, every acceptance event meets an incorrect proof block with probability `0`, so `ε = 0`. -/
theorem PooledIncorrectBlocksLe_honest (n k : ℕ) :
    ∀ (m : ℕ) (acc : Fin m → (Fin m → {S : Finset (Fin n) // S.card = k}) → Prop) (i : Fin m),
      Pr (fun ω => acc i ω ∧ ¬ Disjoint (ω i).1 (∅ : Finset (Fin n))) ≤ 0 := by
  intro m acc i
  simp [Pr, Claims.prUniform]

/-- **`PooledIncorrectBlocksLe`'s conclusion fails at an audit that always accepts**, at `ε = 0`: one window of one unit,
that unit wrong, `q = F = 1`. -/
theorem PooledIncorrectBlocksLe_nontrivial :
    ¬ Pr (fun _ : Fin 1 → Unit => (∀ _ : Fin 1, True) ∧
        1 ≤ ∑ _ : Fin 1, (Finset.univ : Finset (Fin 1)).card) ≤
      (1 - 1 : ℝ) ^ 1 + ∑ _ : Fin 1, (0 : ℝ) := by
  simp [Pr, Claims.prUniform]

end NetTiming.Witness
