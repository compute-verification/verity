module

public import Definitions.NetworkCertifier.Guarantees.PeriodChecked
public import Proofs.NetworkCertifier.Witnesses
public import Proofs.NetworkCertifier.WitnessesChecked
public import Proofs.NetworkCertifier.SampledAuditAccepted
public import Proofs.Flock.SamplingGeneric
public import Proofs.Flock.Claims.Prob
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
import Std.Tactic.BVDecide.Normalize.Bool

/-!
# Checks of the period statement with checked inputs: which run the check is held to

`CheckedWitness` bounds a check's miss on the runs it is given. `SampledLeakChecked` reads it at the trace's run
(`traceRun`): the circuit on the trace's committed values, an incorrect proof block's committed gates held at them,
`false` where the trace opens nothing. `PeriodLeakChecked` reads it at the run at the network certifier's rows
(`certifierRun`), which holds those gates at the trace filled in with the network certifier's openings. The two differ
at a gate the network certifier opens and the trace leaves unopened, and these checks show that the difference decides
the statement: held to the trace's run, `PeriodLeakChecked`'s first part is false.

**The toy** (`freeL`): one input gate and its exclusive or with itself, the egress presence gate, in one proof block
sampled at rate `0`; the verifier accepts every run and has no session, so the extracted witness opens nothing; the
commitment passes an opening of `true` and no other, so it binds. The compiled verifier meets the compiler's contract
(`freeL_sessions`), so it is knowledge sound by the compiler's theorem (`freeL_ks`). The check is `r 1 = false`: no
frame at the slot, which every run of the circuit that holds no gate meets. The network certifier's openings
(`certOps`) register a frame there (`selfXor_registers`).

* `freeL_traceRun_checked`: the check's miss on the trace's run is `0`.
* `freeL_noWit`: no witness is in `RLeakCertifierChecked` at the network certifier's openings.
* `freeL_traceRun_insufficient`: so the first part's premises hold at the trace's run, and its bound fails.
* `periodLeakChecked_traceRun_fails`: the first part of `PeriodLeakChecked`, with `traceRun` in place of
  `certifierRun` and nothing else changed, is false.
* `freeL_certifierRun_miss`: on the run at the network certifier's rows the check misses with probability `1` at the
  toy, so the toy meets `PeriodLeakChecked`'s premise at no miss below `1`, and does not refute it.

**The statement's nontrivial lemma.** `PeriodLeakChecked_nontrivial`: its acceptance conclusion fails at the accepting
verifier `certL`, as `PeriodLeakAccepted`'s does, at the check every run passes, which misses with probability `0` on
any runs (`checkedWitness_true`): a witness in `RLeakCertifierChecked` is in `RLeakCertifier`
(`rLeakCertifier_of_rLeakCertifierChecked`), and none is (`certL_noWit`).

**The read-back.** `RecordFormat.readsBy_read`: every record format reads back under its own schedule as the
parameter, so `RecordFormat.ReadsBy` asks of a format what `RecordFormat.read_record` gives, at fewer parameters.
-/

@[expose] public section

namespace NetTiming

open Claims Claims.Sampling

/-- **The leak relation at the network certifier's rows with a valid run gives the leak relation there**: a sequence
accepted with a valid run is accepted. -/
theorem rLeakCertifier_of_rLeakCertifierChecked {H : Msg → Msg} {Cm : CommitmentScheme Bool}
    {parse : Msg → Option AuditStatement} {partition : AuditStatement → Option Partition} {L K S : ℕ}
    {lay : AuditStatement → BoundaryGates L S} {k : AuditStatement → ℕ} {δ : ℝ}
    {As : AuditStatement → (ℕ → Bool) → Set (ℕ → Bool)} {valid : AuditStatement → (ℕ → Bool) → Prop} {Dj : ℕ}
    {tops : AuditStatement → List (Opening Bool)} {x : Msg} {w : Wit}
    (h : RLeakCertifierChecked H Cm parse partition L K S lay k δ As valid Dj tops x w) :
    RLeakCertifier H Cm parse partition L K S lay k δ As Dj tops x w := by
  obtain ⟨s, P, hs, hP, h⟩ := h
  refine ⟨s, P, hs, hP, fun op o hC hCov hl hD => ?_⟩
  obtain ⟨hl', hD', F, hF, a, ha, free, hfree, -, hacc⟩ := h op o hC hCov hl hD
  exact ⟨hl', hD', F, hF, a, ha, free, hfree, hacc⟩

/-- **Every record format reads back under its own schedule as the parameter** (`RecordFormat.read_record`). -/
theorem RecordFormat.readsBy_read {κ Cell Ob : Type} (f : RecordFormat κ Cell Ob) :
    f.ReadsBy f.read fun θ s => θ = s := by
  rintro θ s rfl hfit a
  exact f.read_record _ hfit a

end NetTiming

namespace NetTiming.Witness

open Claims Claims.Sampling Claims.Witness Claims.Witness.Sampling
open Claims.SamplingGeneric (Arg Compiled)
open Claims.Witness.SamplingGeneric (twoA twoA_KS trueCm sksL sksL_κ noHash_compiledE bext_pr notPart certL)
open NetTiming.SampledAuditProof (valOver_held valOver_gate heldAdv_of faultyGate_of_heldAdv)

/-! ## What a rerun program can return -/

/-- Every value the rerun program returns meets `p`, whatever its calls and its reruns answer. -/
def RetAll {R α : Type} (p : α → Prop) : RP R α → Prop
  | .ret a => p a
  | .hash _ k => ∀ a, RetAll p (k a)
  | .rerun _ k => ∀ v, RetAll p (k v)

section RetAll

variable {R α β : Type}

theorem retAll_eval {p : α → Prop} (H : Msg → Msg) (rn : R → Transcript) :
    ∀ m : RP R α, RetAll p m → p (m.eval H rn).1
  | .ret _, h => h
  | .hash q k, h => retAll_eval H rn (k (H q)) (h (H q))
  | .rerun r k, h => retAll_eval H rn (k (rn r).seen) (h (rn r).seen)

theorem retAll_bind {p : α → Prop} {q : β → Prop} :
    ∀ (m : RP R α) (f : α → RP R β), RetAll p m → (∀ a, p a → RetAll q (f a)) → RetAll q (m.bind f)
  | .ret a, _, h, hf => hf a h
  | .hash _ k, f, h, hf => fun a => retAll_bind (k a) f (h a) hf
  | .rerun _ k, f, h, hf => fun v => retAll_bind (k v) f (h v) hf

theorem retAll_true : ∀ m : RP R α, RetAll (fun _ => True) m
  | .ret _ => trivial
  | .hash _ k => fun a => retAll_true (k a)
  | .rerun _ k => fun v => retAll_true (k v)

theorem retAll_flatMapRP_nil {γ : Type} (f : β → RP R (List γ))
    (hf : ∀ b, RetAll (fun l : List γ => l = []) (f b)) :
    ∀ bs : List β, RetAll (fun l : List γ => l = []) (flatMapRP f bs)
  | [] => by
    show ([] : List γ) = []
    rfl
  | b :: bs =>
    retAll_bind (p := fun l : List γ => l = []) (q := fun l : List γ => l = []) _ _ (hf b) fun cs hcs =>
      retAll_bind (p := fun l : List γ => l = []) (q := fun l : List γ => l = []) _ _
        (retAll_flatMapRP_nil f hf bs) fun ds hds => by
          have h1 : cs = [] := hcs
          have h2 : ds = [] := hds
          show cs ++ ds = []
          simp [h1, h2]

end RetAll

section Extract

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- At a compiled verifier that samples no proof block, a run's extractions return nothing. -/
theorem retAll_runExt_nil (L : Compiled Coin A) (hck : ∀ x r, L.checked x r = []) (x : Msg)
    (s : Claims.Sampling.AuditStatement) (r : Fin (L.V.k x) → Coin) (v : Seen) (c : L.SlotEC x) :
    RetAll (fun l : List (List Msg × Claims.Sampling.Opening Bool) => l = []) (L.runExt x s r v c) := by
  unfold Claims.SamplingGeneric.Compiled.runExt
  have hz : (List.finRange (L.nPU x)).zip (L.checked x r) = [] := by simp [hck x r]
  rw [hz]
  exact rfl

/-- At a compiled verifier that samples no proof block, the witness the sampling extractor returns opens nothing. -/
theorem retAll_E_nil (L : Compiled Coin A) (hck : ∀ x r, L.checked x r = []) (x : Msg)
    (r₀ : Fin (L.V.k x) → Coin) (c : L.E.C x) :
    RetAll (fun o : Option Claims.Sampling.Wit => ∀ w, o = some w → w.vals = []) (L.E.ext x r₀ c) := by
  intro v₀
  dsimp only [Claims.SamplingGeneric.Compiled.E]
  split
  · intro w hw
    cases hw
  · split_ifs
    · generalize List.map c.1 (List.finRange (L.capOut x)) = rs
      refine retAll_bind (p := fun _ => True) _ _ (retAll_true _) fun outer _ => ?_
      generalize (List.finRange (L.kOut x)).zip outer = jos
      refine retAll_bind (p := fun l : List (List Msg × Claims.Sampling.Opening Bool) => l = []) _ _
        (retAll_flatMapRP_nil _ (fun jo => ?_) _) fun found hfound => ?_
      · generalize List.map (fun l => forkOn (L.stage2 x) jo.2.1 (c.2.1 jo.1 l)) (List.finRange (L.capIn x)) = rs'
        refine retAll_bind (p := fun _ => True) _ _ (retAll_true _) fun inner _ => ?_
        generalize (List.finRange (L.kIn x)).zip inner = lis
        exact retAll_flatMapRP_nil _ (fun li => retAll_runExt_nil L hck _ _ _ _ _) _
      · intro w hw
        have hf : found = [] := hfound
        subst hf
        obtain rfl := Option.some.inj hw
        rfl
    · intro w hw
      cases hw

end Extract

/-! ## The toy -/

/-- One input gate and its exclusive or with itself, returned: gate `1` computes `false` at every input. -/
def selfXorProg : Claims.Sampling.Prog := ⟨[.input, .xor [0, 0]], [1]⟩

/-- The program, each proof block sampled independently at rate `0`. -/
def selfXorStmt : Claims.Sampling.AuditStatement := ⟨[], selfXorProg, ⟨"", 0, []⟩, ⟨[.bernoulli 0 1]⟩, []⟩

/-- The compiled verifier over `twoA` at `selfXorStmt`, its two gates one proof block (`notPart`): it accepts after one
round, samples nothing and has no session, and its commitment passes the openings of `true` (`trueCm`). -/
noncomputable def freeL (Coin : Type) : Compiled Coin (twoA Coin) where
  V := accV Coin
  parse _ := some selfXorStmt
  scopeCheck _ := true
  partition _ := some notPart
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
  nOut _ := 1
  nIn _ := 1
  Tape := Unit
  P _ := silentP
  δc _ _ := 0

theorem inScope_selfXor : Claims.Sampling.InScope notPart selfXorStmt := by
  refine ⟨⟨fun g x hx i hi => ?_, by decide⟩,
    ⟨by decide, by decide, by decide, by decide, by decide⟩, rfl, ?_⟩
  · rcases g with _ | _ | g
    · simp [selfXorStmt, selfXorProg] at hx
      subst hx
      simp [BGate.reads] at hi
    · simp [selfXorStmt, selfXorProg] at hx
      subst hx
      simp [BGate.reads] at hi
      omega
    · simp [selfXorStmt, selfXorProg] at hx
  · simp [selfXorStmt, notPart, Claims.Sampling.OneStage.WellFormed]

theorem notLay_fits_selfXor : notLay.Fits selfXorProg 1 := by
  refine ⟨fun t _ ℓ σ g hg => ?_, fun t _ g hg => ?_, fun t ht t' ht' htt' => absurd htt' (by omega)⟩
  · simp [notLay, slotGates] at hg
    subst hg
    decide
  · simp [notLay] at hg
    subst hg
    rfl

/-- The design accepts every set of proof blocks with probability `1`: it samples none. -/
theorem selfXor_Pa (B : Finset ℕ) : selfXorStmt.design.Pa notPart B = 1 := by
  simp [Claims.Sampling.Design.Pa, selfXorStmt, notPart, Claims.Sampling.classesPa, Claims.Sampling.OneStage.Pa]

/-- The exclusive or of gate `0` with itself is `false`. -/
theorem xorSelf_getD (f : ℕ → Bool) (own : Option Bool) :
    ((BGate.xor [0, 0]).val (fun i => if _h : i < 1 then some (f i) else some false) own).getD false = false := by
  have h : (BGate.xor [0, 0]).val (fun i => if _h : i < 1 then some (f i) else some false) own =
      some (Bool.xor (Bool.xor false (f 0)) (f 0)) := rfl
  rw [h]
  cases f 0 <;> rfl

/-- With every held gate held at `false`, gate `1` is `false`, whatever the inputs. -/
theorem selfXor_val (inp : ℕ → Bool) (free : ℕ → Option Bool) (hfree : ∀ g b, free g = some b → b = false) :
    valOver selfXorProg inp free 1 = false := by
  cases h1 : free 1 with
  | some b =>
    rw [valOver_held h1]
    exact hfree 1 b h1
  | none =>
    rw [valOver_gate h1 (show selfXorProg.gates[1]? = some (.xor [0, 0]) from rfl)]
    exact xorSelf_getD (valOver selfXorProg inp free) (some (inp 1))

/-- A trace that opens nothing holds `false` at every gate. -/
theorem selfXor_traceVal (ref : ℕ → Option Bool) (href : ∀ g, ref g = none) (g : ℕ) :
    traceVal selfXorProg ref g = false := by
  rcases g with _ | _ | g <;> simp [traceVal, Prog.read, selfXorProg, href]

/-- A trace that leaves gate `1` unopened, filled in with the network certifier's openings, holds `true` there. -/
theorem selfXor_traceVal_fill (ref : ℕ → Option Bool) (href : ref 1 = none) :
    traceVal selfXorProg (fillTrace ref (certifierVal certOps)) 1 = true := by
  simp [traceVal, Prog.read, selfXorProg, fillTrace, href, certifierVal, certOps]

/-- **The network certifier registers a frame at the slot**: at `selfXorStmt`, one step and the operator that sends
`true`, its openings `certOps` pass under `trueCm`, open both gates, and hold the record `obs1`, a frame at the slot. -/
theorem selfXor_registers (H : Msg → Msg) :
    CertifierRegisters H trueCm notLay 1 (fun _ _ => [true]) selfXorStmt certOps [obs1] := by
  refine ⟨fun q hq => ?_, fun t ht => ?_⟩
  · simp only [certOps, List.mem_cons, List.not_mem_nil, or_false] at hq
    rcases hq with rfl | rfl <;> rfl
  · obtain rfl : t = 0 := by omega
    refine ⟨fun ℓ σ g hg => ?_, fun g hg => ?_, Nat.one_pos, fun ℓ => ?_, ?_⟩
    · simp [notLay, slotGates] at hg
      subst hg
      exact ⟨fun b hb => by simp [selfXorStmt, selfXorProg] at hb, ⟨1, true, []⟩, by simp [certOps], rfl⟩
    · simp [notLay] at hg
      subst hg
      exact ⟨⟨0, true, []⟩, by simp [certOps], rfl⟩
    · funext σ
      simp [obs1, rowOf, notLay, certifierVal, certOps]
    · simp [notLay, certifierVal, certOps]

section Toy

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem freeL_scope (x : Msg) : (freeL Coin).Scope x := ⟨selfXorStmt, notPart, rfl, rfl, inScope_selfXor⟩

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem freeL_scopeAcc (x : Msg) : ScopeAcc (freeL Coin) (fun _ => notLay) (fun _ => 1) x :=
  ⟨freeL_scope x, selfXorStmt, rfl, notLay_fits_selfXor⟩

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem freeL_κ (x : Msg) : (freeL Coin).κ 0 x < 1 / 3 :=
  sksL_κ (Coin := Coin) x

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem freeL_binds (H : Msg → Msg) : (freeL Coin).commit.Binds H := by
  intro roots o o' _ hne ho ho'
  have h1 : o.val = true := ho
  have h2 : o'.val = true := ho'
  exact absurd (h1.trans h2.symm) hne

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- `freeL` samples no proof block. -/
theorem freeL_unchecked {x : Msg} {r : Fin ((freeL Coin).V.k x) → Coin} {u : ℕ}
    (hu : u ∈ (freeL Coin).checked x r) : False := by
  have h : (freeL Coin).checked x r = [] := rfl
  rw [h] at hu
  simp at hu

/-- `freeL` meets the compiler's contract at every hash: it samples nothing, at a design whose rate is `0`. -/
theorem freeL_sessions (H : Msg → Msg) : (freeL Coin).Sessions H where
  draws := by
    intro x s P hs hP _ B _
    obtain rfl : s = selfXorStmt := (Option.some.inj hs).symm
    obtain rfl : P = notPart := (Option.some.inj hP).symm
    exact ⟨fun _ => 1, fun _ => zero_le_one, fun _ => Pr_le_one _, ((Ex_const 1).trans (selfXor_Pa B).symm).le⟩
  slots _ _ := Nat.zero_le _
  shared _ _ _ _ _ _ _ _ _ _ _ _ _ := rfl
  keys x r := by
    have h : (freeL Coin).checked x r = [] := rfl
    rw [h]
    simp
  unions _ := le_rfl
  refuses x _ _ _ _ _ := freeL_scope x
  interior _ _ _ _ _ _ _ := rfl
  fixed _ _ _ hu := (freeL_unchecked hu).elim
  runs _ _ _ _ _ _ hu := (freeL_unchecked hu).elim
  opens _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ hu _ _ := (freeL_unchecked hu).elim

/-- **`freeL` is knowledge sound** at every hash: the compiler's theorem at `twoA`'s knowledge soundness, the
commitment's binding and the contract. -/
theorem freeL_ks (H : Msg → Msg) : (freeL Coin).KS H :=
  Claims.SamplingGeneric.compilerKS Coin (twoA Coin) (freeL Coin) H (twoA_KS H) (freeL_binds H) (freeL_sessions H)

theorem freeL_bext (H : Msg → Msg) (x : Msg) : (Bext H (freeL Coin).V (freeL Coin).E Unit silentP x).Pr H = 0 :=
  bext_pr H (freeL Coin) (fun _ _ _ => rfl) (fun _ => Nat.one_pos) Unit silentP x

theorem freeL_btap (H : Msg → Msg) (x : Msg) :
    (Btap H (freeL Coin).V (freeL Coin).E (freeL Coin).commit (freeL Coin).parse (fun _ => certOps) Unit silentP
      x).Pr H = 0 := by
  unfold Finder.Pr
  rw [Pr_congr fun s => iff_false_intro (by
    simp only [Btap, ownLog_noHash _ _ _ (noHash_compiledE (freeL Coin) (fun _ _ _ => rfl) _ _ _)]
    simp [certifierLog, freeL, certOps, trueCm, OC.eval, findColl])]
  exact Pr_false

/-- **The check `r 1 = false` misses with probability `0` on the trace's run** at `freeL`, at every statement and
prover: the extracted witness opens nothing, so the trace holds `false` at every gate, and the run, with the incorrect
proof block's committed gates held there, holds `false` at gate `1`. -/
theorem freeL_traceRun_checked (H : Msg → Msg) (x : Msg) :
    CheckedWitness H (freeL Coin) (traceRun H (freeL Coin).commit) (fun _ r => r 1 = false) (fun _ => 0) x := by
  intro Ω _ _ P
  unfold PrCheckMiss
  refine le_of_eq ((Pr_congr fun r => iff_false_intro ?_).trans Pr_false)
  rintro ⟨-, w, s, Pt, hw, hs, hPt, -, ρ, hρ, hnv⟩
  obtain rfl : s = selfXorStmt := (Option.some.inj hs).symm
  obtain rfl : Pt = notPart := (Option.some.inj hPt).symm
  rw [traceRun, Set.mem_singleton_iff] at hρ
  subst hρ
  have hvals : w.vals = [] :=
    retAll_eval H (run H P (freeL Coin).V x r.1) ((freeL Coin).E.ext x r.2.1 r.2.2)
      (retAll_E_nil (freeL Coin) (fun _ _ => rfl) x r.2.1 r.2.2) w hw
  have hval : ∀ g, w.val H (freeL Coin).commit selfXorStmt g = none := fun g => by simp [Wit.val, hvals]
  refine hnv (selfXor_val _ _ fun g b hb => ?_)
  have hg := faultyGate_of_heldAdv hb
  have hb' := (heldAdv_of hg).symm.trans hb
  exact (Option.some.inj hb').symm.trans (selfXor_traceVal _ hval g)

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- **No witness is in `RLeakCertifierChecked`** at `freeL` with the check `r 1 = false`, at one step at `δ = 0`,
whatever the inputs `As` admits and whichever proof blocks are incorrect: the network certifier's openings (`certOps`)
register a frame at the slot, and a run that the check passes holds none there. -/
theorem freeL_noWit (H : Msg → Msg) (As : Claims.Sampling.AuditStatement → (ℕ → Bool) → Set (ℕ → Bool)) (x : Msg)
    (w : Claims.Sampling.Wit) :
    ¬ RLeakCertifierChecked H (freeL Coin).commit (freeL Coin).parse (freeL Coin).partition 1 1 1 (fun _ => notLay)
      (fun _ => 1) 0 As (fun _ r => r 1 = false ∧ True) 1 (fun _ => certOps) x w := by
  rintro ⟨s, P, hs, hP, h⟩
  obtain rfl : s = selfXorStmt := (Option.some.inj hs).symm
  obtain rfl : P = notPart := (Option.some.inj hP).symm
  have hcov : Covered 1 [obs1] := by
    intro t ht _ κ
    obtain rfl : t = 0 := by omega
    rfl
  obtain ⟨-, -, F, -, a, -, free, -, hvalid, hacc⟩ :=
    h (fun _ _ => [true]) [obs1] (selfXor_registers H) hcov rfl (by simp [obs1])
  obtain ⟨_, hrows, -⟩ := hacc 0 Nat.one_pos
  have hv : valOver selfXorProg (notLay.inputOf 1 [[true]] a) free 1 = false := hvalid.1
  have h1 : (some [] : Option (List Bool)) =
      rowOf (valOver selfXorProg (notLay.inputOf 1 [[true]] a) free) (notLay.egress 0 0) 0 :=
    congrFun (congrFun hrows 0) 0
  have hrow : rowOf (valOver selfXorProg (notLay.inputOf 1 [[true]] a) free) (notLay.egress 0 0) 0 =
      if valOver selfXorProg (notLay.inputOf 1 [[true]] a) free 1 then
        some (([] : List ℕ).map (valOver selfXorProg (notLay.inputOf 1 [[true]] a) free))
      else none := rfl
  rw [hrow, hv] at h1
  simp at h1

/-- The first part's bound fails at `freeL`'s silent prover, at a relation with no witness at `x` and a bound
`κ' < 1`. -/
theorem freeL_bound_fails (H : Msg → Msg) (Rx : Rel Claims.Sampling.Wit) (κ' : ℝ) (x : Msg) (hR : ∀ w, ¬ Rx x w)
    (hκ : κ' < 1) :
    ¬ PrBad H (freeL Coin).V Rx (freeL Coin).E x silentP ≤
      κ' + Real.sqrt ((freeL Coin).γ x * PrB H (freeL Coin).V silentP x) +
        (Bext H (freeL Coin).V (freeL Coin).E Unit silentP x).Pr H +
        (Btap H (freeL Coin).V (freeL Coin).E (freeL Coin).commit (freeL Coin).parse (fun _ => certOps) Unit silentP
          x).Pr H := by
  intro h
  have hp : PrBad H (freeL Coin).V Rx (freeL Coin).E x silentP = 1 := prBad_accV H Rx _ x silentP hR
  have hb : PrB H (freeL Coin).V silentP x = 0 := prB_accV H silentP x
  simp only [hp, hb, mul_zero, Real.sqrt_zero, add_zero, freeL_bext, freeL_btap] at h
  linarith

/-- **At `freeL` the check `r 1 = false` misses on the run at the network certifier's rows with probability `1`**, at
the silent prover: the network certifier registers a frame (`selfXor_registers`), the trace opens nothing, so it
disagrees with no opening, and that run holds gate `1` at the network certifier's opening, `true`. So the toy meets
`CheckedWitness` at `certifierRun` at no miss below `1`. -/
theorem freeL_certifierRun_miss (H : Msg → Msg) :
    ¬ CheckedWitness H (freeL Coin)
      (certifierRun H (freeL Coin).commit 1 1 1 (fun _ => notLay) (fun _ => 1) (fun _ => certOps))
      (fun _ r => r 1 = false) (fun _ => 0) [] := by
  intro h
  have hone : PrCheckMiss H (freeL Coin)
      (certifierRun H (freeL Coin).commit 1 1 1 (fun _ => notLay) (fun _ => 1) (fun _ => certOps))
      (fun _ r => r 1 = false) [] silentP = 1 := by
    unfold PrCheckMiss
    refine (Pr_congr fun r => iff_true_intro ⟨run_accV_out H silentP [] r.1 r.2.1, ?_⟩).trans Claims.ZKMal.Pr_true
    have hacc := run_accV_out (Coin := Coin) H silentP [] r.1 r.2.1
    obtain ⟨w, hw⟩ : ∃ w, ((freeL Coin).E.eval H silentP [] r.1 r.2.1 r.2.2).1 = some w :=
      ⟨_, Claims.SamplingGeneric.Compiled.E_eval H (freeL Coin) silentP [] r.1 r.2.1 r.2.2 (s := selfXorStmt) rfl
        hacc⟩
    have hvals : w.vals = [] :=
      retAll_eval H (run H silentP (freeL Coin).V [] r.1) ((freeL Coin).E.ext [] r.2.1 r.2.2)
        (retAll_E_nil (freeL Coin) (fun _ _ => rfl) [] r.2.1 r.2.2) w hw
    have hval : ∀ g, w.val H (freeL Coin).commit selfXorStmt g = none := fun g => by simp [Wit.val, hvals]
    have hF : (0 : ℕ) ∈ faulty H (freeL Coin).commit selfXorStmt notPart w := by
      simp only [Claims.Sampling.faulty, Finset.mem_filter, Finset.mem_range]
      refine ⟨by decide, ?_⟩
      rintro ⟨u, hu, ι, hc⟩
      obtain rfl : u = [0, 1] := (Option.some.inj hu).symm
      obtain ⟨b, hb, -⟩ := (hc 0 (by decide)).2 (by decide)
      have hr : selfXorStmt.prog.refIn [0, 1] (w.val H (freeL Coin).commit selfXorStmt) ι 0 = none := by
        have hbd : selfXorStmt.prog.Boundary [0, 1] 0 = true := by decide
        simp [Prog.refIn, hbd, hval]
      cases hr.symm.trans hb
    have hfg : FaultyGate selfXorStmt.prog notPart (faulty H (freeL Coin).commit selfXorStmt notPart w) 1 :=
      ⟨0, hF, [0, 1], rfl, by decide, by decide⟩
    have hrun : ∀ inp : ℕ → Bool, valOver selfXorProg inp
        (heldAdv selfXorProg notPart (faulty H (freeL Coin).commit selfXorStmt notPart w)
          (fillTrace (w.val H (freeL Coin).commit selfXorStmt) (certifierVal certOps))) 1 = true :=
      fun _ => (valOver_held (heldAdv_of hfg)).trans (selfXor_traceVal_fill _ (hval 1))
    refine ⟨w, selfXorStmt, notPart, hw, rfl, rfl, ?_, _,
      ⟨fun g _ b hb => ?_, fun _ _ => [true], [obs1], selfXor_registers H, rfl⟩, fun hv => ?_⟩
    · have he : allSlots selfXorStmt.design notPart = ∅ := by
        simp [allSlots, allSlotsFrom, selfXorStmt, notPart]
      rw [he]
      exact Finset.disjoint_empty_right _
    · cases (hval g).symm.trans hb
    · cases (hrun _).symm.trans hv
  have h1 : PrCheckMiss H (freeL Coin)
      (certifierRun H (freeL Coin).commit 1 1 1 (fun _ => notLay) (fun _ => 1) (fun _ => certOps))
      (fun _ r => r 1 = false) [] silentP ≤ 0 := h Unit silentP
  rw [hone] at h1
  norm_num at h1

end Toy

/-! ## The check held to the trace's run does not carry the bound -/

/-- **`CheckedWitness` at the trace's run does not carry the first part's bound.** At `freeL Bool`, SHA-512 and the
statement `[]`: the compiled verifier is knowledge sound, its commitment binds, the statement is in the audit's scope,
and the checks `r 1 = false` and `True` each miss with probability `0` on the trace's run (`CheckedWitness` at
`traceRun`). Yet the bound of `PeriodLeakChecked`'s first part fails at the silent prover, at one step at `δ = 0` with
no input registered: no witness is in `RLeakCertifierChecked` (`freeL_noWit`). -/
theorem freeL_traceRun_insufficient :
    (freeL Bool).KS FlockDeploy.sha512 ∧ (freeL Bool).commit.Binds FlockDeploy.sha512 ∧
      ScopeAcc (freeL Bool) (fun _ => notLay) (fun _ => 1) [] ∧
      CheckedWitness FlockDeploy.sha512 (freeL Bool) (traceRun FlockDeploy.sha512 (freeL Bool).commit)
        (fun _ r => r 1 = false) (fun _ => 0) [] ∧
      CheckedWitness FlockDeploy.sha512 (freeL Bool) (traceRun FlockDeploy.sha512 (freeL Bool).commit)
        (fun _ _ => True) (fun _ => 0) [] ∧
      ¬ PrBad FlockDeploy.sha512 (freeL Bool).V
          (RLeakCertifierChecked FlockDeploy.sha512 (freeL Bool).commit (freeL Bool).parse (freeL Bool).partition 1 1 1
            (fun _ => notLay) (fun _ => 1) 0 (fun s => AsReg s.prog notLay 1 ∅) (fun _ r => r 1 = false ∧ True) 1
            (fun _ => certOps))
          (freeL Bool).E [] silentP ≤
        (freeL Bool).κ 0 [] + 0 + 0 +
            Real.sqrt ((freeL Bool).γ [] * PrB FlockDeploy.sha512 (freeL Bool).V silentP []) +
          (Bext FlockDeploy.sha512 (freeL Bool).V (freeL Bool).E Unit silentP []).Pr FlockDeploy.sha512 +
          (Btap FlockDeploy.sha512 (freeL Bool).V (freeL Bool).E (freeL Bool).commit (freeL Bool).parse
            (fun _ => certOps) Unit silentP []).Pr FlockDeploy.sha512 :=
  ⟨freeL_ks _, freeL_binds _, freeL_scopeAcc [], freeL_traceRun_checked _ [], checkedWitness_true _ _ _ [],
    freeL_bound_fails FlockDeploy.sha512 _ ((freeL Bool).κ 0 [] + 0 + 0) [] (freeL_noWit FlockDeploy.sha512 _ [])
      (by linarith [freeL_κ (Coin := Bool) []])⟩

/-- **`PeriodLeakChecked`'s first part with the check held to the trace's run is false**: the first part as
`Guarantees.PeriodLeakChecked` states it, with `traceRun` in place of `certifierRun` in its two premises and nothing
else changed, fails at `freeL Bool` (`freeL_traceRun_insufficient`). The first part reads none of the monitoring
period's parameters, so the statement with that premise fails with it at every period. -/
theorem periodLeakChecked_traceRun_fails :
    ¬ ∀ {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}
      (H : Msg → Msg) (D : Compiled Coin A) (_ : D.KS H) (L K S : ℕ)
      (lay : Claims.Sampling.AuditStatement → BoundaryGates L S) (k : Claims.Sampling.AuditStatement → ℕ) (δ : ℝ)
      (_ : 0 ≤ δ) (reg : Claims.Sampling.AuditStatement → Finset ℕ)
      (valid₁ valid₂ : Claims.Sampling.AuditStatement → (ℕ → Bool) → Prop) (ε₁ ε₂ : Msg → ℝ) (Dj : ℕ)
      (tops : Claims.Sampling.AuditStatement → List (Claims.Sampling.Opening Bool)) (_ : D.commit.Binds H),
      ∀ m, ScopeAcc D lay k m → CheckedWitness H D (traceRun H D.commit) valid₁ ε₁ m →
        CheckedWitness H D (traceRun H D.commit) valid₂ ε₂ m →
        ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
          PrBad H D.V
              (RLeakCertifierChecked H D.commit D.parse D.partition L K S lay k δ
                (fun s => AsReg s.prog (lay s) (k s) (reg s)) (fun s r => valid₁ s r ∧ valid₂ s r) Dj tops)
              D.E m P ≤
            D.κ δ m + ε₁ m + ε₂ m + Real.sqrt (D.γ m * PrB H D.V P m) + (Bext H D.V D.E Ω P m).Pr H +
              (Btap H D.V D.E D.commit D.parse tops Ω P m).Pr H := by
  intro h
  obtain ⟨hks, hbinds, hscope, hchk₁, hchk₂, hfail⟩ := freeL_traceRun_insufficient
  exact hfail (h FlockDeploy.sha512 (freeL Bool) hks 1 1 1 (fun _ => notLay) (fun _ => 1) 0 le_rfl (fun _ => ∅)
    (fun _ r => r 1 = false) (fun _ _ => True) (fun _ => 0) (fun _ => 0) 1 (fun _ => certOps) hbinds [] hscope hchk₁
    hchk₂ Unit silentP)

/-! ## The statement's nontrivial lemma -/

/-- **`PeriodLeakChecked`'s acceptance conclusion fails at the accepting verifier** `certL`, at one step at `δ = 0`,
with no input registered, at the checks every run passes and the misses `0`: at the statement `[]`, its silent
prover's bound fails, since no witness is in `RLeakCertifierChecked` (`rLeakCertifier_of_rLeakCertifierChecked`,
`certL_noWit`). -/
theorem PeriodLeakChecked_nontrivial :
    ¬ PrBad FlockDeploy.sha512 (certL Bool).V
        (RLeakCertifierChecked FlockDeploy.sha512 (certL Bool).commit (certL Bool).parse (certL Bool).partition 1 1 1
          (fun _ => notLay) (fun _ => 1) 0 (fun s => AsReg s.prog notLay 1 ∅) (fun _ _ => True ∧ True) 1
          (fun _ => certOps))
        (certL Bool).E [] silentP ≤
      (certL Bool).κ 0 [] + 0 + 0 +
          Real.sqrt ((certL Bool).γ [] * PrB FlockDeploy.sha512 (certL Bool).V silentP []) +
        (Bext FlockDeploy.sha512 (certL Bool).V (certL Bool).E Unit silentP []).Pr FlockDeploy.sha512 +
        (Btap FlockDeploy.sha512 (certL Bool).V (certL Bool).E (certL Bool).commit (certL Bool).parse
          (fun _ => certOps) Unit silentP []).Pr FlockDeploy.sha512 :=
  certL_bound_fails FlockDeploy.sha512 _ ((certL Bool).κ 0 [] + 0 + 0) []
    (fun w hw => certL_noWit FlockDeploy.sha512 _ [] w (rLeakCertifier_of_rLeakCertifierChecked hw))
    (by linarith [certL_κ (Coin := Bool) []])

end NetTiming.Witness
