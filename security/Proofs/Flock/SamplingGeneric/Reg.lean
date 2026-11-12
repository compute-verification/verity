module

public import Proofs.Flock.SamplingGeneric.Split
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
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
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.Aggregate
import Proofs.Flock.Claims.ERepCost
import Proofs.Flock.Claims.Finder
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.Sampling.Binding
import Proofs.Flock.Claims.Sampling.Extract
import Proofs.Flock.SamplingGeneric.Assemble
import Proofs.Flock.Soundness.Rewinding

/-!
# The sampled compiler's committed form

`CompilerKSReg` from the sessions' bound (`Compiled.SessionKS`): the knowledge soundness of a compiled sampled verifier
against the committed trace's incorrect set (`Compiled.Freg` at the plurality `Compiled.regVal`, which the prover's tape
fixes before the window's coins), so its sampling term is the acceptance probability `P_a(Freg)` of a set fixed at
registration.

* `plurAt`, `pr_off_le`: **the rewinding lemma at a table's plurality** (`FlockSoundness.Rewinding.prBad_le`), over
  any finite space: a point opens a position off a trace that is the table's plurality wherever a point opens, with
  probability at most `N·√Pr[conflict]`. The committed trace reads it over the runs at one tape; a trace read at the
  statement reads it over the tapes and the runs together (`SamplingGeneric/RegX.lean`).
* `kept`, `opensAt_eq`: what a run's extractions keep (`runExt`, session by session of the sample); `opensAt` is the
  assembled witness's value there.
* `InConf`, `OffReg`, `Conflict`: a run's extractions keep two openings of one position that disagree; they open a gate
  of the program off the committed trace; two runs' extractions open one gate two ways.
* `selOf`, `selOf_some`: the first slot of a sample whose block is in a set; `selReg` is it at `Freg`.
* `trace_split`: **the deterministic half**, at any trace. At an accepted run that samples a proof block no values
  agreeing with the trace make correct, the block's session is not extracted at its first slot, or the run's
  extractions disagree with themselves or open a gate off the trace: otherwise the trace on the gates, the kept
  openings elsewhere, makes the block correct (`Sessions.opens`). `reg_split` is it at the committed trace and `Freg`.
* `bconfX_of_disagree`, `bconf_of_disagree`: two kept openings of one position under one roots that disagree are a
  collision among the two runs' extractions' own calls, at two tapes and at one.
* `pr_inConf_le`, `pr_conflict_le`: a disagreement within one run, or between two runs at one tape, is a collision
  `B_conf` finds (`CommitmentScheme.Binds`).
* `pr_offReg_le`: the rewinding lemma at one tape's runs' openings of the program's gates, the committed trace being
  their plurality (`plur`, `regVal_eq_plur`): a run opens a gate off it with probability at most `N·√Pr[conflict]`.
* `pr_unfound_le`: the sessions' bound at the selection `leadSel selReg` (the first slot of the session of the first
  slot of the sample whose block is in `Freg`), averaged by Jensen's inequality.
* `pr_miss_le`, `miss_of_event`: a set of proof blocks refused at `δ` is missed with probability at most `P_a ≤ δ`
  (`Sessions.draws`), so the event's bound at a family of sets gives the bound on an accepted run whose set is outside
  `M_δ`.
* `ksReg_of_sessionKS`, `compilerKSReg_of`: **`KSReg`**: the union of the three, and on the sample, `miss_of_event` at
  `Freg`.
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims Finset
open Claims.Sampling (AuditStatement CommitmentScheme Opening InScope forkOn assemble ownLog flatMapRP flatMapRP_eval
  val_assemble ιOf ιOf_eq)

theorem Pr_eq_card_of_mem {α : Type} [Fintype α] (p : α → Prop) (S : Finset α) (h : ∀ a, a ∈ S ↔ p a) :
    Pr p = (#S : ℝ) / Fintype.card α := by
  classical
  unfold Pr Claims.prUniform
  rw [show univ.filter p = S from Finset.ext fun a => by rw [mem_filter, h]; exact and_iff_right (mem_univ _)]

theorem Pr_pos_of {α : Type} [Fintype α] {p : α → Prop} (a : α) (h : p a) : 0 < Pr p := by
  classical
  have : Nonempty α := ⟨a⟩
  unfold Pr Claims.prUniform
  have hc : 0 < #(univ.filter p) := card_pos.2 ⟨a, mem_filter.2 ⟨mem_univ _, h⟩⟩
  exact div_pos (Nat.cast_pos.2 hc) (Nat.cast_pos.2 Fintype.card_pos)

/-- Coins fresh on `I`, an event's probability as the average over the coins off `I`. -/
theorem Pr_forkOn {k : ℕ} {Coin : Type} [Fintype Coin] [Nonempty Coin] (I : Finset (Fin k))
    (p : (Fin k → Coin) → Prop) : Pr p = Ex fun r => Pr fun d : Fin k → Coin => p (forkOn I r d) := by
  classical
  calc Pr p = Ex fun R => if p R then (1 : ℝ) else 0 := (Ex_ind p).symm
    _ = Ex fun rd : (Fin k → Coin) × (Fin k → Coin) => if p (forkOn I rd.1 rd.2) then (1 : ℝ) else 0 :=
        (Sampling.Ex_forkOn I fun R => if p R then (1 : ℝ) else 0).symm
    _ = Ex fun r => Ex fun d : Fin k → Coin => if p (forkOn I r d) then (1 : ℝ) else 0 := Ex_prod _
    _ = Ex fun r => Pr fun d : Fin k → Coin => p (forkOn I r d) := by
        congr 1
        funext r
        exact Ex_ind _

theorem Pr_forkOn_prod {k : ℕ} {Coin C : Type} [Fintype Coin] [Nonempty Coin] [Fintype C] (I : Finset (Fin k))
    (p : (Fin k → Coin) → C → Prop) :
    Pr (fun q : (Fin k → Coin) × C => p q.1 q.2) =
      Ex fun r => Pr fun q : (Fin k → Coin) × C => p (forkOn I r q.1) q.2 := by
  calc Pr (fun q : (Fin k → Coin) × C => p q.1 q.2) = Ex fun R => Pr fun c : C => p R c :=
        Sampling.Pr_prod_eq_Ex (fun q : (Fin k → Coin) × C => p q.1 q.2)
    _ = Ex fun rd : (Fin k → Coin) × (Fin k → Coin) => Pr fun c : C => p (forkOn I rd.1 rd.2) c :=
        (Sampling.Ex_forkOn I fun R => Pr fun c : C => p R c).symm
    _ = Ex fun r => Ex fun d : Fin k → Coin => Pr fun c : C => p (forkOn I r d) c := Ex_prod _
    _ = Ex fun r => Pr fun q : (Fin k → Coin) × C => p (forkOn I r q.1) q.2 := by
        congr 1
        funext r
        exact (Sampling.Pr_prod_eq_Ex fun q : (Fin k → Coin) × C => p (forkOn I r q.1) q.2).symm

/-- What the assembled witness opens is a kept opening under the statement's roots that passes. -/
theorem val_some_mem {H : Msg → Msg} {Cm : CommitmentScheme Bool} {s : AuditStatement}
    {found : List (List Msg × Opening Bool)} {g : ℕ} {b : Bool} (h : (assemble s found).val H Cm s g = some b) :
    ∃ q ∈ found, q.1 = s.roots ∧ q.2.pos = g ∧ q.2.val = b ∧ Cm.Opens H s.roots q.2 = true := by
  unfold Sampling.Wit.val at h
  obtain ⟨o, ho, rfl⟩ := Option.map_eq_some_iff.1 h
  have hpo := List.find?_some ho
  simp only [Bool.and_eq_true, beq_iff_eq] at hpo
  obtain ⟨a, ha, hao⟩ := List.mem_filterMap.1 (List.mem_of_find?_eq_some ho)
  by_cases har : a.1 = s.roots
  · simp only [har, ite_true, Option.some.injEq] at hao
    subst hao
    exact ⟨a, ha, har, hpo.1, rfl, hpo.2⟩
  · simp [har] at hao

/-- The value of a two-valued table with the larger weight (`true` on a tie) has the largest weight. -/
theorem le_plur (f : Bool → ℝ) (v : Bool) : f v ≤ f (decide (f false ≤ f true)) := by
  by_cases h : f false ≤ f true
  · rw [decide_eq_true h]
    cases v
    · exact h
    · exact le_rfl
  · rw [decide_eq_false h]
    cases v
    · exact le_rfl
    · exact (not_le.1 h).le

/-- The value most points of a table of optional bits open at a position (`true` on a tie). -/
noncomputable def plurAt {S : Type} [Fintype S] (op : S → ℕ → Option Bool) (g : ℕ) : Bool :=
  decide (Pr (fun a : S => op a g = some false) ≤ Pr fun a : S => op a g = some true)

/-- **The rewinding lemma at a table's plurality** (`FlockSoundness.Rewinding.prBad_le`): for a table `op` of optional
bits over a finite space and a trace `tr` that is, at each position a point opens, the value most points open there
(`plurAt`), a point opens a position below `N` off `tr` with probability at most `N·√Pr[conflict]`, a conflict being
two points that open one position below `N` two ways. The space is the reader's: the runs at one tape
(`Compiled.pr_offReg_le`), or the tapes and the runs together. -/
theorem pr_off_le {S : Type} [Fintype S] [Nonempty S] (N : ℕ) (op : S → ℕ → Option Bool) (tr : ℕ → Option Bool)
    (htr : ∀ a g b, op a g = some b → tr g = some (plurAt op g)) :
    Pr (fun a : S => ∃ g < N, ∃ b, op a g = some b ∧ tr g ≠ some b) ≤
      N * Real.sqrt (Pr fun aa : S × S =>
        ∃ g : Fin N, (op aa.1 g).isSome ∧ (op aa.2 g).isSome ∧ op aa.1 g ≠ op aa.2 g) := by
  classical
  have hcard : (0 : ℝ) < Fintype.card S := by exact_mod_cast Fintype.card_pos
  have hcnt : ∀ (g : Fin N) (b : Bool),
      (#(FlockSoundness.Rewinding.openedTo (fun (a : S) (g : Fin N) => op a g) g b) : ℝ) =
        Pr (fun a : S => op a g = some b) * Fintype.card S := by
    intro g b
    rw [Pr_eq_card_of_mem (fun a : S => op a g = some b)
      (FlockSoundness.Rewinding.openedTo (fun (a : S) (g : Fin N) => op a g) g b)
      fun a => mem_filter.trans (and_iff_right (mem_univ _)), div_mul_cancel₀ _ hcard.ne']
  have hpl : FlockSoundness.Rewinding.IsPlurality (fun (a : S) (g : Fin N) => op a g) fun g => plurAt op g := by
    intro g v
    have h := le_plur (fun b => Pr fun a : S => op a g = some b) v
    have h' : (#(FlockSoundness.Rewinding.openedTo (fun (a : S) (g : Fin N) => op a g) g v) : ℝ) ≤
        #(FlockSoundness.Rewinding.openedTo (fun (a : S) (g : Fin N) => op a g) g (plurAt op g)) := by
      rw [hcnt, hcnt]
      exact mul_le_mul_of_nonneg_right h hcard.le
    exact_mod_cast h'
  have hbad := FlockSoundness.Rewinding.prBad_le (op := fun (a : S) (g : Fin N) => op a g) (fun g => plurAt op g)
    hpl N fun a => (card_filter_le _ _).trans (card_univ.trans (Fintype.card_fin _)).le
  have hoff : Pr (fun a : S => ∃ g < N, ∃ b, op a g = some b ∧ tr g ≠ some b) ≤
      (#(FlockSoundness.Rewinding.bad (fun (a : S) (g : Fin N) => op a g) fun g => plurAt op g) : ℝ) /
        Fintype.card S := by
    calc Pr (fun a : S => ∃ g < N, ∃ b, op a g = some b ∧ tr g ≠ some b)
        ≤ Pr (fun a : S => ∃ g,
            FlockSoundness.Rewinding.Off (fun (a : S) (g : Fin N) => op a g) (fun g => plurAt op g) a g) :=
          Pr_mono fun a ha => by
            obtain ⟨g, hg, b, hb, hne⟩ := ha
            refine ⟨⟨g, hg⟩, Option.isSome_iff_exists.2 ⟨b, hb⟩, fun h => hne ?_⟩
            rw [htr a g b hb]
            exact (hb.symm.trans h).symm
      _ = _ := Pr_eq_card_of_mem _
          (FlockSoundness.Rewinding.bad (fun (a : S) (g : Fin N) => op a g) fun g => plurAt op g)
          fun a => by simp only [FlockSoundness.Rewinding.bad, mem_filter, mem_univ, true_and]
  have hmem : ∀ aa : S × S, aa ∈ FlockSoundness.Rewinding.conflict (fun (a : S) (g : Fin N) => op a g) ↔
      ∃ g, FlockSoundness.Rewinding.ConflictAt (fun (a : S) (g : Fin N) => op a g) aa g :=
    fun aa => by simp only [FlockSoundness.Rewinding.conflict, mem_filter, mem_univ, true_and]
  have hconf : (#(FlockSoundness.Rewinding.conflict (fun (a : S) (g : Fin N) => op a g)) : ℝ) /
      (Fintype.card S : ℝ) ^ 2 =
      Pr fun aa : S × S => ∃ g : Fin N, (op aa.1 g).isSome ∧ (op aa.2 g).isSome ∧ op aa.1 g ≠ op aa.2 g := by
    rw [Pr_eq_card_of_mem
        (fun aa : S × S => ∃ g : Fin N, (op aa.1 g).isSome ∧ (op aa.2 g).isSome ∧ op aa.1 g ≠ op aa.2 g)
        (FlockSoundness.Rewinding.conflict (fun (a : S) (g : Fin N) => op a g)) hmem,
      Fintype.card_prod S S, Nat.cast_mul, sq]
  calc Pr (fun a : S => ∃ g < N, ∃ b, op a g = some b ∧ tr g ≠ some b)
      ≤ _ := hoff
    _ ≤ _ := hbad
    _ = N * Real.sqrt (Pr fun aa : S × S =>
          ∃ g : Fin N, (op aa.1 g).isSome ∧ (op aa.2 g).isSome ∧ op aa.1 g ≠ op aa.2 g) := by
      rw [Fintype.card_fin, hconf, Real.sqrt_mul (mul_self_nonneg (N : ℝ)),
        Real.sqrt_mul_self (Nat.cast_nonneg _)]

namespace Compiled

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-! ## A run's extractions -/

/-- **What a run's extractions keep**, at the prover's run `(ω, r)` and `A`'s extractor's coins `c` (`runExt`). -/
noncomputable def kept (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg)
    (s : AuditStatement) (ω : Ω) (r : Fin (L.V.k x) → Coin) (c : L.SlotEC x) : List (List Msg × Opening Bool) :=
  ((L.runExt x s r (run H P L.V x ω r).seen c).eval H (run H P L.V x ω)).1

theorem opensAt_eq (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    {s : AuditStatement} (hs : L.parse x = some s) (r : Fin (L.V.k x) → Coin) (c : L.SlotEC x) (g : ℕ) :
    L.opensAt H P x ω r c g = (assemble s (L.kept H P x s ω r c)).val H L.commit s g := by
  simp only [opensAt, kept, hs]

/-! ## The three events -/

/-- A run's extractions keep two openings under one roots at one position, of different values. -/
def InConf (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (s : AuditStatement) (ω : Ω)
    (r : Fin (L.V.k x) → Coin) (c : L.SlotEC x) : Prop :=
  ∃ q ∈ L.kept H P x s ω r c, ∃ q' ∈ L.kept H P x s ω r c, q.1 = q'.1 ∧ q.2.pos = q'.2.pos ∧ q.2.val ≠ q'.2.val

/-- A run's extractions open a gate of the program off the committed trace. -/
def OffReg (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (r : Fin (L.V.k x) → Coin) (c : L.SlotEC x) : Prop :=
  ∃ g < L.nG x, ∃ b, L.opensAt H P x ω r c g = some b ∧ L.regVal H P x ω g ≠ some b

/-- Two runs' extractions, at one tape, open a gate of the program two ways (`FlockSoundness.Rewinding.ConflictAt`). -/
def Conflict (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (rr : ((Fin (L.V.k x) → Coin) × L.SlotEC x) × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) : Prop :=
  ∃ g : Fin (L.nG x), (L.opensAt H P x ω rr.1.1 rr.1.2 g).isSome ∧ (L.opensAt H P x ω rr.2.1 rr.2.2 g).isSome ∧
    L.opensAt H P x ω rr.1.1 rr.1.2 g ≠ L.opensAt H P x ω rr.2.1 rr.2.2 g

/-- The first slot of a sample whose proof block is in `F`. -/
def selOf (L : Compiled Coin A) (x : Msg) (F : Finset ℕ) (l : List ℕ) : Option (Fin (L.nPU x)) :=
  (List.finRange (L.nPU x)).find? fun i => (l[i.val]?).any fun v => decide (v ∈ F)

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- A sample of at most `n_PU` slots with a block in `F` has a first such slot, its block below `n_PU`. -/
theorem selOf_some (L : Compiled Coin A) (x : Msg) {F : Finset ℕ} {l : List ℕ} (hlen : l.length ≤ L.nPU x)
    (hFlt : ∀ v ∈ F, v < L.nPU x) {v : ℕ} (hvl : v ∈ l) (hvF : v ∈ F) :
    ∃ i, L.selOf x F l = some i ∧ ∃ u : Fin (L.nPU x), l[i.val]? = some u.val ∧ u.val ∈ F := by
  obtain ⟨n, hn⟩ := List.mem_iff_getElem?.1 hvl
  have hnl : n < l.length := (List.getElem?_eq_some_iff.1 hn).1
  obtain ⟨i, hi⟩ := Option.isSome_iff_exists.1
    ((List.find?_isSome (p := fun i : Fin (L.nPU x) => (l[i.val]?).any fun v => decide (v ∈ F))).2
      ⟨⟨n, lt_of_lt_of_le hnl hlen⟩, List.mem_finRange _, by
        show (l[n]?).any (fun v => decide (v ∈ F)) = true
        rw [hn, Option.any_some]
        exact decide_eq_true hvF⟩)
  have hpi := List.find?_some hi
  refine ⟨i, hi, ?_⟩
  cases h : l[i.val]? with
  | none =>
    rw [h, Option.any_none] at hpi
    exact absurd hpi Bool.false_ne_true
  | some w =>
    rw [h, Option.any_some] at hpi
    have hwF := of_decide_eq_true hpi
    exact ⟨⟨w, hFlt w hwF⟩, rfl, hwF⟩

/-- The first slot of the sample whose proof block is in the committed trace's incorrect set (`selOf` at `Freg`). -/
noncomputable def selReg (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (l : List ℕ) : Option (Fin (L.nPU x)) :=
  L.selOf x (L.Freg H P x ω) l

/-- A sample of at most `n_PU` slots with a block in `Freg` has a first such slot, its block below `n_PU`. -/
theorem selReg_some (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    {l : List ℕ} (hlen : l.length ≤ L.nPU x) (hFlt : ∀ v ∈ L.Freg H P x ω, v < L.nPU x) {v : ℕ} (hvl : v ∈ l)
    (hvF : v ∈ L.Freg H P x ω) :
    ∃ i, L.selReg H P x ω l = some i ∧ ∃ u : Fin (L.nPU x), l[i.val]? = some u.val ∧ u.val ∈ L.Freg H P x ω :=
  L.selOf_some x hlen hFlt hvl hvF

/-! ## The deterministic half -/

set_option synthInstance.maxSize 1024 in
/-- **The deterministic half, at a trace**: at an accepted run whose sample holds, at slot `i`, a proof block `u` that
no values agreeing with the trace `tr` on the program's gates make correct, the session of `i` is not extracted at its
first slot `i₀`'s coins, or the run's extractions disagree with themselves or open a gate off `tr`. Otherwise the values
that are `tr` on the program's gates and the kept openings elsewhere, with the interior values the kept openings give,
agree with every opening the session's witness reveals at `u` that passes (`u`'s slot shows `i₀`'s session,
`Sessions.shared`), so they make `u` correct (`Sessions.opens`). The trace is the reader's: the committed trace at the
run's tape (`reg_split`), or one read at the statement. -/
theorem trace_split {H : Msg → Msg} {L : Compiled Coin A} (hS : L.Sessions H) {x : Msg} {s : AuditStatement}
    {Pt : Sampling.Partition} (hs : L.parse x = some s) (hPt : L.partition s = some Pt) (hin : InScope Pt s)
    {Ω : Type} (P : Prover Ω) (ω : Ω) (r : Fin (L.V.k x) → Coin) (c : L.SlotEC x)
    (hacc : (run H P L.V x ω r).out = .acc) (i u : Fin (L.nPU x)) (hiu : (L.checked x r)[i.val]? = some u.val)
    (tr : ℕ → Option Bool)
    (hF : ¬ ∃ b, Pt.blocks[u.val]? = some b ∧ ∃ (val : ℕ → Option Bool) (ι : ℕ → Bool),
      (∀ g < s.prog.gates.length, val g = tr g) ∧ s.prog.Correct (s.prog.refIn b val ι) b)
    (i₀ u₀ : Fin (L.nPU x)) (hlead : L.leadOf x (L.checked x r) i = some i₀)
    (hu₀ : (L.checked x r)[i₀.val]? = some u₀.val) :
    ¬ L.Found H x (run H P L.V x ω) u₀ r (c i₀ u₀) ∨ L.InConf H P x s ω r c ∨
      ∃ g < L.nG x, ∃ b, L.opensAt H P x ω r c g = some b ∧ tr g ≠ some b := by
  have hu := List.mem_of_getElem? hiu
  by_cases hfd : L.Found H x (run H P L.V x ω) u₀ r (c i₀ u₀)
  swap
  · exact Or.inl hfd
  by_cases hconf : L.InConf H P x s ω r c
  · exact Or.inr (Or.inl hconf)
  by_cases hoff : ∃ g < L.nG x, ∃ b, L.opensAt H P x ω r c g = some b ∧ tr g ≠ some b
  · exact Or.inr (Or.inr hoff)
  exfalso
  have hagree : ∀ q ∈ L.kept H P x s ω r c, ∀ q' ∈ L.kept H P x s ω r c, q.1 = q'.1 → q.2.pos = q'.2.pos →
      q.2.val = q'.2.val :=
    fun q hq q' hq' h1 h2 => Classical.byContradiction fun hne => hconf ⟨q, hq, q', hq', h1, h2, hne⟩
  have hreg : ∀ g < L.nG x, ∀ b, L.opensAt H P x ω r c g = some b → tr g = some b :=
    fun g hg b hb => Classical.byContradiction fun hne => hoff ⟨g, hg, b, hb, hne⟩
  have hnG : L.nG x = s.prog.gates.length := by simp only [Compiled.nG, hs]
  obtain ⟨xS, vS, hsess, hsh, w, hw, hR⟩ := hfd
  obtain ⟨hk, hc, hfirst⟩ := of_leadOf hlead
  have hsu : L.session x u (run H P L.V x ω r).seen = some (xS, vS) := by
    rw [← hS.shared x Ω P ω r _ _ _ _ hacc hu₀ hiu hk]
    exact hsess
  have hub : u.val ∈ L.sessBlocks x (L.checked x r) i₀ := by
    rw [sessBlocks, hc]
    exact List.mem_filterMap.2 ⟨i.val, self_mem_sessSlots hiu u.isLt, hiu⟩
  obtain ⟨b, hb, hcor⟩ := hS.opens x s Pt Ω P ω r u xS vS w hs hPt hin hacc hu hsu hR
  refine hF ⟨b, hb, fun g => if g < s.prog.gates.length then tr g else L.opensAt H P x ω r c g,
    ιOf (L.kept H P x s ω r c) (L.iroots x (run H P L.V x ω r).seen u), fun g hg => ?_,
    hcor _ _ fun q hq hok => ?_⟩
  · show (if g < s.prog.gates.length then tr g else L.opensAt H P x ω r c g) = tr g
    exact ite_eq_left hg
  · have hqk : q ∈ L.kept H P x s ω r c :=
      L.mem_runExt H x (run H P L.V x ω) s r _ c hu₀ u₀.isLt hfirst
        (L.sessExt_mem H x (run H P L.V x ω) s _ u₀ r (c i₀ u₀) hsess hsh hw hub hq hok)
    refine ⟨fun hr => ?_, fun hi => ?_⟩
    · have hv : L.opensAt H P x ω r c q.2.pos = some q.2.val := by
        rw [L.opensAt_eq H P x ω hs]
        rw [hr] at hok
        exact val_assemble hagree hqk hr hok
      show (if q.2.pos < s.prog.gates.length then tr q.2.pos
        else L.opensAt H P x ω r c q.2.pos) = some q.2.val
      by_cases hg : q.2.pos < s.prog.gates.length
      · rw [ite_eq_left hg]
        exact hreg _ (by rw [hnG]; exact hg) _ hv
      · rw [ite_eq_right hg]
        exact hv
    · rw [← hi]
      exact ιOf_eq hagree hqk

set_option synthInstance.maxSize 1024 in
/-- **The deterministic half**: at an accepted run whose sample holds proof block `u` of the committed trace's
incorrect set at slot `i`, the session of `i` is not extracted at its first slot `i₀`'s coins, or the run's extractions
disagree with themselves or open a gate off the committed trace: `u ∈ Freg` says no values agreeing with the committed
trace make `u` correct (`trace_split` at `regVal`). -/
theorem reg_split {H : Msg → Msg} {L : Compiled Coin A} (hS : L.Sessions H) {x : Msg} {s : AuditStatement}
    {Pt : Sampling.Partition} (hs : L.parse x = some s) (hPt : L.partition s = some Pt) (hin : InScope Pt s)
    {Ω : Type} (P : Prover Ω) (ω : Ω) (r : Fin (L.V.k x) → Coin) (c : L.SlotEC x)
    (hacc : (run H P L.V x ω r).out = .acc) (i u : Fin (L.nPU x)) (hiu : (L.checked x r)[i.val]? = some u.val)
    (hF : u.val ∈ L.Freg H P x ω) (i₀ u₀ : Fin (L.nPU x)) (hlead : L.leadOf x (L.checked x r) i = some i₀)
    (hu₀ : (L.checked x r)[i₀.val]? = some u₀.val) :
    ¬ L.Found H x (run H P L.V x ω) u₀ r (c i₀ u₀) ∨ L.InConf H P x s ω r c ∨ L.OffReg H P x ω r c := by
  have hpart : L.part s = Pt := by simp only [Compiled.part, hPt, Option.getD_some]
  simp only [Freg, hs, mem_filter, mem_range] at hF
  rw [hpart] at hF
  exact trace_split hS hs hPt hin P ω r c hacc i u hiu (L.regVal H P x ω) hF.2 i₀ u₀ hlead hu₀

/-! ## Binding: disagreements are `B_conf`'s collisions -/

/-- Two openings that two runs' extractions keep, at two tapes, under one roots at one position, of different values,
are a collision among the two runs' extractions' own calls, which `findColl` finds. The tapes need not be one: an
opening binds under its roots, whichever tape opened it. -/
theorem bconfX_of_disagree {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) {x : Msg}
    (s : AuditStatement) {Ω : Type} (P : Prover Ω) (ω ω' : Ω)
    (rc rc' : (Fin (L.V.k x) → Coin) × L.SlotEC x) {q q' : List Msg × Opening Bool}
    (hq : q ∈ L.kept H P x s ω rc.1 rc.2 ∨ q ∈ L.kept H P x s ω' rc'.1 rc'.2)
    (hq' : q' ∈ L.kept H P x s ω rc.1 rc.2 ∨ q' ∈ L.kept H P x s ω' rc'.1 rc'.2)
    (h1 : q.1 = q'.1) (h2 : q.2.pos = q'.2.pos) (h3 : q.2.val ≠ q'.2.val) :
    ∃ u v, findColl (ownLog H (run H P L.V x ω) (L.runExt x s rc.1 (run H P L.V x ω rc.1).seen rc.2) ++
      ownLog H (run H P L.V x ω') (L.runExt x s rc'.1 (run H P L.V x ω' rc'.1).seen rc'.2)) = some (u, v) ∧
      Collision H u v := by
  have hpass : ∀ t, t ∈ L.kept H P x s ω rc.1 rc.2 ∨ t ∈ L.kept H P x s ω' rc'.1 rc'.2 →
      L.commit.Opens H t.1 t.2 = true ∧ ∀ e ∈ ((L.commit.opens t.1 t.2).eval H).2,
        e ∈ ownLog H (run H P L.V x ω) (L.runExt x s rc.1 (run H P L.V x ω rc.1).seen rc.2) ++
          ownLog H (run H P L.V x ω') (L.runExt x s rc'.1 (run H P L.V x ω' rc'.1).seen rc'.2) := by
    rintro t (ht | ht)
    · obtain ⟨ho, hl⟩ := L.passes_runExt H x (run H P L.V x ω) s rc.1 (run H P L.V x ω rc.1).seen rc.2 t ht
      exact ⟨ho, fun e he => List.mem_append_left _ (hl e he)⟩
    · obtain ⟨ho, hl⟩ := L.passes_runExt H x (run H P L.V x ω') s rc'.1 (run H P L.V x ω' rc'.1).seen rc'.2 t ht
      exact ⟨ho, fun e he => List.mem_append_right _ (hl e he)⟩
  have hvalid : ∀ p ∈ ownLog H (run H P L.V x ω) (L.runExt x s rc.1 (run H P L.V x ω rc.1).seen rc.2) ++
      ownLog H (run H P L.V x ω') (L.runExt x s rc'.1 (run H P L.V x ω' rc'.1).seen rc'.2), H p.1 = p.2 :=
    fun p hp => (List.mem_append.1 hp).elim (Sampling.ownLog_valid H _ _ p) (Sampling.ownLog_valid H _ _ p)
  obtain ⟨rt, o⟩ := q
  obtain ⟨rt', o'⟩ := q'
  dsimp only at h1 h2 h3
  subst h1
  have hc := Sampling.disagree_collision hbind h2 h3 (hpass _ hq).1 (hpass _ hq').1 (hpass _ hq).2 (hpass _ hq').2
  obtain ⟨⟨u, v⟩, huv⟩ := Option.isSome_iff_exists.1 (Sampling.findColl_isSome _ hc)
  exact ⟨u, v, huv, findColl_spec H _ hvalid u v huv⟩

/-- Two openings that two runs' extractions keep at one tape, under one roots at one position, of different values, are
a collision among the two runs' extractions' own calls, which `findColl` finds (`bconfX_of_disagree` at one tape). -/
theorem bconf_of_disagree {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) {x : Msg}
    (s : AuditStatement) {Ω : Type} (P : Prover Ω) (ω : Ω)
    (rc rc' : (Fin (L.V.k x) → Coin) × L.SlotEC x) {q q' : List Msg × Opening Bool}
    (hq : q ∈ L.kept H P x s ω rc.1 rc.2 ∨ q ∈ L.kept H P x s ω rc'.1 rc'.2)
    (hq' : q' ∈ L.kept H P x s ω rc.1 rc.2 ∨ q' ∈ L.kept H P x s ω rc'.1 rc'.2)
    (h1 : q.1 = q'.1) (h2 : q.2.pos = q'.2.pos) (h3 : q.2.val ≠ q'.2.val) :
    ∃ u v, findColl (ownLog H (run H P L.V x ω) (L.runExt x s rc.1 (run H P L.V x ω rc.1).seen rc.2) ++
      ownLog H (run H P L.V x ω) (L.runExt x s rc'.1 (run H P L.V x ω rc'.1).seen rc'.2)) = some (u, v) ∧
      Collision H u v :=
  bconfX_of_disagree hbind s P ω ω rc rc' hq hq' h1 h2 h3

/-- An event of a tape and two runs at it that gives a collision `B_conf`'s output is at most `Pr[B_conf]`. -/
theorem le_bconf (H : Msg → Msg) (L : Compiled Coin A) {x : Msg} {s : AuditStatement} (hs : L.parse x = some s)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    {Q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) × ((Fin (L.V.k x) → Coin) × L.SlotEC x) → Prop}
    (hQ : ∀ q, Q q → ∃ u v, findColl (ownLog H (run H P L.V x q.1)
        (L.runExt x s q.2.1.1 (run H P L.V x q.1 q.2.1.1).seen q.2.1.2) ++
      ownLog H (run H P L.V x q.1) (L.runExt x s q.2.2.1 (run H P L.V x q.1 q.2.2.1).seen q.2.2.2)) = some (u, v) ∧
        Collision H u v) :
    Pr Q ≤ (Bconf H L Ω P x).Pr H := by
  simp only [Bconf, hs]
  exact Pr_mono hQ

/-- **One run's disagreement**: at most `Pr[B_conf]`, its first run. -/
theorem pr_inConf_le {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) {x : Msg}
    {s : AuditStatement} (hs : L.parse x = some s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) :
    Ex (fun ω : Ω => Pr fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.InConf H P x s ω rc.1 rc.2) ≤
      (Bconf H L Ω P x).Pr H := by
  calc Ex (fun ω : Ω => Pr fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.InConf H P x s ω rc.1 rc.2)
      = Pr (fun p : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.InConf H P x s p.1 p.2.1 p.2.2) :=
        (Sampling.Pr_prod_eq_Ex (fun p : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
          L.InConf H P x s p.1 p.2.1 p.2.2)).symm
    _ = Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
          L.InConf H P x s q.1 q.2.1.1 q.2.1.2) :=
        (Pr_drop (fun (ω : Ω) (rc : (Fin (L.V.k x) → Coin) × L.SlotEC x) => L.InConf H P x s ω rc.1 rc.2)).symm
    _ ≤ (Bconf H L Ω P x).Pr H :=
        le_bconf H L hs P fun q hq => by
          obtain ⟨a, ha, b, hb, h1, h2, h3⟩ := hq
          exact bconf_of_disagree hbind s P q.1 q.2.1 q.2.2 (Or.inl ha) (Or.inl hb) h1 h2 h3

/-- **Two runs' conflict**: at most `Pr[B_conf]`. -/
theorem pr_conflict_le {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) {x : Msg}
    {s : AuditStatement} (hs : L.parse x = some s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) :
    Ex (fun ω : Ω => Pr fun rr : ((Fin (L.V.k x) → Coin) × L.SlotEC x) × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      L.Conflict H P x ω rr) ≤ (Bconf H L Ω P x).Pr H := by
  refine (Sampling.Pr_prod_eq_Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) ×
      ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.Conflict H P x q.1 q.2)).symm.trans_le
    (le_bconf H L hs P fun q hq => ?_)
  obtain ⟨g, h1, h2, hne⟩ := hq
  obtain ⟨b1, hb1⟩ := Option.isSome_iff_exists.1 h1
  obtain ⟨b2, hb2⟩ := Option.isSome_iff_exists.1 h2
  rw [hb1, hb2] at hne
  rw [L.opensAt_eq H P x q.1 hs] at hb1
  rw [L.opensAt_eq H P x q.1 hs] at hb2
  obtain ⟨a, ha, ha1, ha2, ha3, -⟩ := val_some_mem hb1
  obtain ⟨a', ha', ha1', ha2', ha3', -⟩ := val_some_mem hb2
  exact bconf_of_disagree hbind s P q.1 q.2.1 q.2.2 (Or.inl ha) (Or.inr ha') (ha1.trans ha1'.symm)
    (ha2.trans ha2'.symm) (by rw [ha3, ha3']; exact fun h => hne (congrArg some h))

/-! ## The rewinding lemma at the committed trace -/

/-- The committed trace's value at a gate some run opens. -/
noncomputable def plur (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (g : ℕ) : Bool :=
  decide (Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.opensAt H P x ω rc.1 rc.2 g = some false) ≤
    Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.opensAt H P x ω rc.1 rc.2 g = some true))

/-- At a gate some run's extractions open, the committed trace is the plurality's value. -/
theorem regVal_eq_plur (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    {r : Fin (L.V.k x) → Coin} {c : L.SlotEC x} {g : ℕ} {b : Bool} (h : L.opensAt H P x ω r c g = some b) :
    L.regVal H P x ω g = some (L.plur H P x ω g) := by
  have hpos := Pr_pos_of (p := fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
    L.opensAt H P x ω rc.1 rc.2 g = some b) (r, c) h
  simp only [regVal, plur]
  split_ifs with h0
  · cases b
    · exact absurd h0.2 hpos.ne'
    · exact absurd h0.1 hpos.ne'
  · exact congrArg some (decide_eq_decide.2 Iff.rfl)

/-- **The rewinding lemma at the committed trace**: at each tape, a run's extractions open a gate of the program off
the committed trace with probability at most `N·√Pr[conflict]`, `N` the program's gates (`pr_off_le` at the tape's
runs' openings, the committed trace being their plurality). -/
theorem pr_offReg_le (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) :
    Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.OffReg H P x ω rc.1 rc.2) ≤
      L.nG x * Real.sqrt (Pr fun rr : ((Fin (L.V.k x) → Coin) × L.SlotEC x) × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
        L.Conflict H P x ω rr) :=
  pr_off_le (L.nG x) (fun (rc : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g : ℕ) => L.opensAt H P x ω rc.1 rc.2 g)
    (L.regVal H P x ω) fun _ _ _ h => L.regVal_eq_plur H P x ω h

/-! ## The sessions' bound at the selection -/

/-- **The sessions' bound at a selection**, averaged over the tape: the run accepts and the proof block at the picked
slot is not found at the slot's coins with probability at most `κ_s + √(k(x)·γ_s·Pr[B])` (Jensen's inequality). -/
theorem pr_unfound_le (H : Msg → Msg) (L : Compiled Coin A) {x : Msg} {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (β : Ω → (Fin (L.V.k x) → Coin) → ℝ) (hβ0 : ∀ ω r, 0 ≤ β ω r)
    (hβ : Ex (fun p : Ω × (Fin (L.V.k x) → Coin) => β p.1 p.2) ≤ PrB H L.V P x)
    (sel : Ω → List ℕ → Option (Fin (L.nPU x)))
    (hκ : ∀ ω r, Pr (fun p : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
      (run H P L.V x ω (forkOn (L.stage2 x) r p.1)).out = .acc ∧
        ∃ i, sel ω (L.checked x (forkOn (L.stage2 x) r p.1)) = some i ∧
          ∃ u : Fin (L.nPU x), (L.checked x (forkOn (L.stage2 x) r p.1))[i.val]? = some u.val ∧
            ¬ L.Found H x (run H P L.V x ω) u (forkOn (L.stage2 x) r p.1) (p.2 i u)) ≤
        L.κs x + Real.sqrt (L.V.k x * L.γs x * β ω r)) :
    Ex (fun ω : Ω => Pr fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
      (run H P L.V x ω rc.1).out = .acc ∧ ∃ i, sel ω (L.checked x rc.1) = some i ∧
        ∃ u : Fin (L.nPU x), (L.checked x rc.1)[i.val]? = some u.val ∧
          ¬ L.Found H x (run H P L.V x ω) u rc.1 (rc.2 i u)) ≤
      L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) := by
  have hγ0 : 0 ≤ (L.V.k x : ℝ) * L.γs x := mul_nonneg (Nat.cast_nonneg _) (L.γs_nonneg x)
  calc Ex (fun ω : Ω => Pr fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
        (run H P L.V x ω rc.1).out = .acc ∧ ∃ i, sel ω (L.checked x rc.1) = some i ∧
          ∃ u : Fin (L.nPU x), (L.checked x rc.1)[i.val]? = some u.val ∧
            ¬ L.Found H x (run H P L.V x ω) u rc.1 (rc.2 i u))
      ≤ Ex (fun ω : Ω => L.κs x + Ex fun r => Real.sqrt (L.V.k x * L.γs x * β ω r)) :=
        Ex_mono fun ω => (Pr_forkOn_prod (L.stage2 x) (fun (r : Fin (L.V.k x) → Coin) (c : L.SlotEC x) =>
            (run H P L.V x ω r).out = .acc ∧ ∃ i, sel ω (L.checked x r) = some i ∧
              ∃ u : Fin (L.nPU x), (L.checked x r)[i.val]? = some u.val ∧
                ¬ L.Found H x (run H P L.V x ω) u r (c i u))).trans_le
          ((Ex_mono (hκ ω)).trans_eq (by rw [Ex_add, Ex_const]))
    _ = L.κs x + Ex (fun p : Ω × (Fin (L.V.k x) → Coin) => Real.sqrt (L.V.k x * L.γs x * β p.1 p.2)) := by
        rw [Ex_add, Ex_const,
          Ex_prod (fun p : Ω × (Fin (L.V.k x) → Coin) => Real.sqrt (L.V.k x * L.γs x * β p.1 p.2))]
    _ ≤ L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) := by
        refine add_le_add le_rfl ((Ex_sqrt_le _ fun p => mul_nonneg hγ0 (hβ0 _ _)).trans (Real.sqrt_le_sqrt ?_))
        rw [Ex_mul_left]
        exact mul_le_mul_of_nonneg_left hβ hγ0

/-! ## The sample's miss -/

/-- **A set refused at `δ` is missed with probability at most `δ`**: for a set `F` of proof blocks below `n_PU`, the
sample holds no block of `F` while `F` is outside `M_δ` with probability at most `P_a(F) ≤ δ` (`Sessions.draws`). -/
theorem pr_miss_le {H : Msg → Msg} {L : Compiled Coin A} (hS : L.Sessions H) {x : Msg} {s : AuditStatement}
    {Pt : Sampling.Partition} (hs : L.parse x = some s) (hPt : L.partition s = some Pt) (hin : InScope Pt s)
    (F : Finset ℕ) (hFlt : ∀ v ∈ F, v < L.nPU x) {δ : ℝ} (hδ : 0 ≤ δ) :
    Pr (fun r : Fin (L.V.k x) → Coin => F ∉ s.design.M (L.part s) δ ∧ ∀ u ∈ L.checked x r, u ∉ F) ≤ δ := by
  have hpart : L.part s = Pt := by simp only [Compiled.part, hPt, Option.getD_some]
  by_cases hM : F ∈ s.design.M (L.part s) δ
  · exact (Pr_mono (q := fun _ => False) fun _ h => h.1 hM).trans (Pr_false.trans_le hδ)
  · have hsub : F ⊆ range Pt.blocks.length := fun v hv =>
      mem_range.2 (by rw [← nPU_eq hs hPt]; exact hFlt v hv)
    obtain ⟨esc, -, hesc, hEesc⟩ := hS.draws x s Pt hs hPt hin _ hsub
    have hM' : ¬ δ < s.design.Pa (L.part s) F := hM
    rw [hpart] at hM'
    calc Pr (fun r : Fin (L.V.k x) → Coin => F ∉ s.design.M (L.part s) δ ∧ ∀ u ∈ L.checked x r, u ∉ F)
        ≤ Pr (fun r : Fin (L.V.k x) → Coin => ∀ u ∈ L.checked x r, u ∉ F) := Pr_mono fun _ h => h.2
      _ = Ex fun r => Pr fun d : Fin (L.V.k x) → Coin =>
            ∀ u ∈ L.checked x (forkOn (L.stage2 x) r d), u ∉ F := Pr_forkOn (L.stage2 x) _
      _ ≤ Ex esc := Ex_mono hesc
      _ ≤ δ := hEesc.trans (not_lt.1 hM')

/-- **The second conjunct from the first**, at any family `F` of sets of proof blocks below `n_PU`, one a tape: if the
run accepts with a sampled block in its tape's set with probability at most `ε`, it accepts with that set outside `M_δ`
with probability at most `δ + ε` (`pr_miss_le` at each tape). -/
theorem miss_of_event {H : Msg → Msg} {L : Compiled Coin A} (hS : L.Sessions H) {x : Msg} {s : AuditStatement}
    {Pt : Sampling.Partition} (hs : L.parse x = some s) (hPt : L.partition s = some Pt) (hin : InScope Pt s)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (F : Ω → Finset ℕ) (hFlt : ∀ ω, ∀ v ∈ F ω, v < L.nPU x)
    {ε : ℝ} (h1 : Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
      ∃ u ∈ L.checked x p.2, u ∈ F p.1) ≤ ε) {δ : ℝ} (hδ : 0 ≤ δ) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
      F p.1 ∉ s.design.M (L.part s) δ) ≤ δ + ε := by
  have hmiss : ∀ ω, Pr (fun r : Fin (L.V.k x) → Coin => F ω ∉ s.design.M (L.part s) δ ∧
      ∀ u ∈ L.checked x r, u ∉ F ω) ≤ δ := fun ω => pr_miss_le hS hs hPt hin (F ω) (hFlt ω) hδ
  have hmiss' : Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => F p.1 ∉ s.design.M (L.part s) δ ∧
      ∀ u ∈ L.checked x p.2, u ∉ F p.1) ≤ δ :=
    (Sampling.Pr_prod_eq_Ex _).trans_le ((Ex_mono hmiss).trans (Ex_const δ).le)
  refine (Sampling.Pr_or_le (fun p hp => ?_) h1 hmiss').trans_eq (add_comm _ _)
  by_cases hm : ∃ u ∈ L.checked x p.2, u ∈ F p.1
  · exact Or.inl ⟨hp.1, hm⟩
  · exact Or.inr ⟨hp.2, fun u hu hF => hm ⟨u, hu, hF⟩⟩

/-! ## The committed form -/

set_option synthInstance.maxSize 1024 in
/-- **`KSReg` from the sessions' bound**: at each tape, an accepted run that samples a proof block of `Freg` has the
first such block's session unfound at its first slot (the sessions' bound at `leadSel selReg`), or its extractions
disagree with themselves (`B_conf`) or open a gate off the committed trace (the rewinding lemma); and a set refused at
`δ` is missed with probability at most `P_a ≤ δ` (`miss_of_event`). -/
theorem ksReg_of_sessionKS {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) (hS : L.Sessions H)
    (hSK : L.SessionKS H) : L.KSReg H := by
  intro x hsc Ω _ _ P
  obtain ⟨s, Pt, hs, hPt, hin⟩ := hsc
  obtain ⟨β, hβ0, hβ, hκ⟩ := hSK x Ω P
  have hFlt : ∀ ω, ∀ v ∈ L.Freg H P x ω, v < L.nPU x := by
    intro ω v hv
    simp only [Freg, hs, mem_filter, mem_range] at hv
    simp only [Compiled.nPU, hs]
    exact hv.1
  have hper : ∀ ω, Pr (fun r : Fin (L.V.k x) → Coin => (run H P L.V x ω r).out = .acc ∧
      ∃ u ∈ L.checked x r, u ∈ L.Freg H P x ω) ≤
      Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => (run H P L.V x ω rc.1).out = .acc ∧
        ∃ i, L.leadSel x (L.selReg H P x ω) (L.checked x rc.1) = some i ∧
          ∃ u : Fin (L.nPU x), (L.checked x rc.1)[i.val]? = some u.val ∧
            ¬ L.Found H x (run H P L.V x ω) u rc.1 (rc.2 i u)) +
      (Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.InConf H P x s ω rc.1 rc.2) +
        L.nG x * Real.sqrt (Pr fun rr : ((Fin (L.V.k x) → Coin) × L.SlotEC x) ×
          ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.Conflict H P x ω rr)) := by
    intro ω
    have hIO := (Sampling.Pr_union_le
      (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.InConf H P x s ω rc.1 rc.2)
      (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.OffReg H P x ω rc.1 rc.2)).trans
        (add_le_add le_rfl (L.pr_offReg_le H P x ω))
    refine (Pr_fst (β := L.SlotEC x) (fun r : Fin (L.V.k x) → Coin => (run H P L.V x ω r).out = .acc ∧
      ∃ u ∈ L.checked x r, u ∈ L.Freg H P x ω)).symm.trans_le (Sampling.Pr_or_le (fun rc hrc => ?_) le_rfl hIO)
    obtain ⟨hacc, v, hvl, hvF⟩ := hrc
    obtain ⟨i, hi, u, hiu, huF⟩ := L.selReg_some H P x ω (hS.slots x rc.1) (hFlt ω) hvl hvF
    obtain ⟨i₀, u₀, hlead, hu₀⟩ := leadOf_some hiu u.isLt
    rcases reg_split hS hs hPt hin P ω rc.1 rc.2 hacc i u hiu huF i₀ u₀ hlead hu₀ with h | h | h
    · exact Or.inl ⟨hacc, i₀, by rw [leadSel, hi, Option.bind_some, hlead], u₀, hu₀, h⟩
    · exact Or.inr (Or.inl h)
    · exact Or.inr (Or.inr h)
  have hT := L.pr_unfound_le H P β hβ0 hβ (fun ω => L.leadSel x (L.selReg H P x ω))
    fun ω r => hκ ω r (L.leadSel x (L.selReg H P x ω))
  have hI := pr_inConf_le hbind hs P
  have hC : Ex (fun ω : Ω => Real.sqrt (Pr fun rr : ((Fin (L.V.k x) → Coin) × L.SlotEC x) ×
      ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.Conflict H P x ω rr)) ≤ Real.sqrt ((Bconf H L Ω P x).Pr H) :=
    (Ex_sqrt_le _ fun _ => Pr_nonneg _).trans (Real.sqrt_le_sqrt (pr_conflict_le hbind hs P))
  have hn : (0 : ℝ) ≤ L.nG x := Nat.cast_nonneg _
  have h1 : Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
      ∃ u ∈ L.checked x p.2, u ∈ L.Freg H P x p.1) ≤ L.εreg H x P := by
    refine (Sampling.Pr_prod_eq_Ex _).trans_le ((Ex_mono hper).trans ?_)
    rw [Ex_add, Ex_add, Ex_mul_left]
    unfold Compiled.εreg
    linarith [mul_le_mul_of_nonneg_left hC hn]
  refine ⟨h1, fun s' hs' δ hδ => ?_⟩
  rw [show s' = s from Option.some.inj (hs'.symm.trans hs)]
  exact miss_of_event hS hs hPt hin P (L.Freg H P x) hFlt h1 hδ

end Compiled

/-- **The committed form** from the sessions' bound. -/
theorem compilerKSReg_of (hsess : SessionKSOf) : CompilerKSReg := by
  intro Coin _ _ _ A L H hA hB hS
  exact Compiled.ksReg_of_sessionKS hB hS (hsess Coin A L H hA hS)

end Claims.SamplingGeneric
