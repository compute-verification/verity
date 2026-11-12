module

public import Definitions.NetworkCertifier.SampledReg
public import Proofs.Flock.Claims.Sampling.KS
public import Proofs.Flock.SamplingGeneric.Assemble
public import Definitions.NetworkCertifier.Guarantees.SampledAudit
public import Definitions.NetworkCertifier.SampledCharge
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
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.Finder
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.Sampling.Binding
import Proofs.NetworkCertifier.SampledAuditAccepted
import Proofs.NetworkCertifier.SampledCharge
import Proofs.NetworkCertifier.SampledLeak

/-!
# The leak bound at the registration's values

`RAcc` pins the circuit's inputs to the trace's committed values, and holds every incorrect proof block's committed gate as
advice. At a registered input that is a free input:

* in an incorrect proof block, the gate is held (`FaultyGate`: an input is on its proof block's boundary), so its input
  value is never read;
* in any other proof block, the proof block is correct on its committed values, so the trace opens the gate
  (`val_some_of_input`) and its value there is the opened one (`traceVal_of_val`);
* in no proof block, it is an unread input (`Prog.Unread`, which `Partitions` may leave out): the run's value at every
  gate but an unread input is the same whatever the unread inputs' values (`valOver_congr_held`), and the layout fits
  the program, so its egress gates are returned and none is an unread input.

So when the trace agrees with the registration at the registered inputs it opens, the accepted advice can be moved to
the inputs pinned at the registration's values (`rLeak_reg_of_rAcc`), a set the statement and the registration fix. An
extracted trace that disagrees opens a registered position to a value the registration doesn't. The compiled extractor
checked that opening with its own calls (`Compiled.passes_found`, `Compiled.ownLog_E`), so under
`CommitmentScheme.Binds` its calls and those that check
the registration's opening collide, and `B_reg` finds the collision (`regColl_of_disagree`, `Pr_regColl_le_Breg`).
`sampledLeak_registered` is the bound: `Pr[acc ∧ ¬RLeak] ≤ δ + ε_ks + ε_bind + Pr[B_reg]`, with the leak relation at the
registration's values, and the count `2^#(freeInputs \ reg) · #faultySets · 2^β · Dj^k` there.
-/

@[expose] public section

namespace NetTiming

open Claims Claims.Sampling NetTiming.SampledAuditProof

/-! ### A registered input of a correct proof block is opened at the trace's value -/

theorem val_some_of_input {H : Msg → Msg} {Cm : CommitmentScheme Bool} {s : AuditStatement} {P : Partition} {w : Wit}
    (hP : P.Partitions s.prog) {g : ℕ} (hg : s.prog.gates[g]? = some .input) (hr : ¬ s.prog.Unread g)
    (hnf : ¬ FaultyGate s.prog P (faulty H Cm s P w) g) : ∃ b, w.val H Cm s g = some b := by
  obtain ⟨ι, hι⟩ := correct_of_not_faulty H Cm s P w
  have hlt : g < s.prog.gates.length := (List.getElem?_eq_some_iff.1 hg).1
  have hmem : g ∈ P.gates := hP.2.2.1 g hlt (fun b h => by rw [hg] at h; cases h) hr
  obtain ⟨v, u, hu, hgu⟩ := exists_of_mem_gates hmem
  have hv : v ∉ faulty H Cm s P w := fun hv => hnf ⟨v, hv, u, hu, hgu, boundary_of_input hg⟩
  obtain ⟨b, hb, -⟩ := (hι v u hu hv g hgu).2 (boundary_of_input hg)
  rw [ref_of_boundary (boundary_of_input hg)] at hb
  exact ⟨b, hb⟩

theorem traceVal_of_val {C : Prog} {ref : ℕ → Option Bool} {g : ℕ} {b : Bool} (hg : C.gates[g]? = some .input)
    (h : ref g = some b) : traceVal C ref g = b := by
  simp [traceVal, Prog.read, hg, h]

/-! ### Moving the advice to the registration's values -/

theorem inputOf_congr_at {L S : ℕ} (lay : BoundaryGates L S) (k : ℕ) (ins : List (List Bool)) {a a' : ℕ → Bool} {g : ℕ}
    (h : a g = a' g) : lay.inputOf k ins a g = lay.inputOf k ins a' g := by
  unfold BoundaryGates.inputOf
  split
  · rfl
  · rw [h]

/-- **Inputs held or unread don't move `C`'s run**: two inputs that agree at every input `free` doesn't hold, but the
unread ones, give one value at every gate but an unread input. -/
theorem valOver_congr_held (C : Prog) {inp inp' : ℕ → Bool} (free : ℕ → Option Bool)
    (h : ∀ i, C.gates[i]? = some .input → free i = none → ¬ C.Unread i → inp i = inp' i) :
    ∀ g, ¬ C.Unread g → valOver C inp free g = valOver C inp' free g := by
  intro g
  induction g using Nat.strong_induction_on with
  | _ g ih =>
    intro hg
    rcases hf : free g with _ | b
    · rcases hx : C.gates[g]? with _ | x
      · rw [valOver_of_none_none C inp hf hx, valOver_of_none_none C inp' hf hx]
      · rw [valOver_of_none_some C inp hf hx, valOver_of_none_some C inp' hf hx]
        have hrd : ∀ i ∈ x.reads, (if _h : i < g then some (valOver C inp free i) else some false) =
            if _h : i < g then some (valOver C inp' free i) else some false := by
          intro i hi
          split_ifs with hig
          · rw [ih i hig (not_unread_of_read hx hi)]
          · rfl
        rw [bgate_val_congr x _ hrd]
        cases x <;> first | rfl | rw [h g hx hf hg]
    · rw [valOver_of_some C inp hf, valOver_of_some C inp' hf]

/-- **At the registration's values**: on a partitioned program that its layout fits, `RAcc δ` at a trace that opens each
registered input, where it opens one, to the registration's value `ρ` there gives the leak relation at the inputs pinned
at `ρ`. -/
theorem rLeak_reg_of_rAcc {H : Msg → Msg} {Cm : CommitmentScheme Bool} {parse : Msg → Option AuditStatement}
    {partition : AuditStatement → Option Partition} {L K S : ℕ} {lay : AuditStatement → BoundaryGates L S} {k : AuditStatement → ℕ} {δ : ℝ} {Dj : ℕ}
    (reg : AuditStatement → Finset ℕ) (ρ : AuditStatement → ℕ → Bool) {x : Msg} {w : Wit}
    (hpart : ∀ s P, parse x = some s → partition s = some P → P.Partitions s.prog)
    (hfit : ∀ s, parse x = some s → (lay s).Fits s.prog (k s))
    (hagree : ∀ s, parse x = some s → ∀ g ∈ reg s, ∀ b, w.val H Cm s g = some b → b = ρ s g)
    (h : RAcc H Cm parse partition L K S lay k δ x w) :
    RLeak H Cm parse partition L K S lay k δ (fun s _ => AsReg s.prog (lay s) (k s) (reg s) (ρ s)) Dj x w := by
  obtain ⟨s, P, hs, hP, hRel, hdisj, free, hfree, hheld, hacc⟩ := h
  have hF := faulty_mem_faultySets hRel hdisj
  have hPart := hpart s P hs hP
  set tv := traceVal s.prog (w.val H Cm s) with htv
  set G := freeInputs s.prog (lay s) (k s) with hG
  let a' : ℕ → Bool := fun g => if g ∈ G then (if g ∈ reg s then ρ s g else tv g) else false
  have hpt : ∀ i, s.prog.gates[i]? = some .input → free i = none → ¬ s.prog.Unread i →
      (fun g => if g ∈ G then tv g else false) i = a' i := by
    intro i hi hfi hri
    show (if i ∈ G then tv i else false) = if i ∈ G then (if i ∈ reg s then ρ s i else tv i) else false
    by_cases hiG : i ∈ G
    · rw [ite_eq_left hiG, ite_eq_left hiG]
      by_cases hir : i ∈ reg s
      · rw [ite_eq_left hir]
        have hnf : ¬ FaultyGate s.prog P (faulty H Cm s P w) i := fun hfg => hheld i hfg hfi
        obtain ⟨b, hb⟩ := val_some_of_input hPart hi hri hnf
        rw [htv, traceVal_of_val hi hb]
        exact hagree s hs i hir b hb
      · rw [ite_eq_right hir]
    · rw [ite_eq_right hiG, ite_eq_right hiG]
  refine ⟨s, P, hs, hP, fun op o hC hCov hl hD => ⟨hl, hD, _, hF, a',
    ⟨fun g hg hgf => ?_, fun g hg => ite_eq_right hg⟩, free, hfree, ?_⟩⟩
  · show (if g ∈ G then (if g ∈ reg s then ρ s g else tv g) else false) = ρ s g
    rw [ite_eq_left hgf, ite_eq_left hg]
  · intro t ht
    obtain ⟨h, hr, hst⟩ := hacc op o hC hCov t ht
    refine ⟨h, ?_, hst⟩
    have hv : ∀ g, ¬ s.prog.Unread g →
        valOver s.prog ((lay s).inputOf (k s) (ingressUpTo op o t) tv) free g =
          valOver s.prog ((lay s).inputOf (k s) (ingressUpTo op o t) a') free g :=
      valOver_congr_held s.prog free fun i hi hfi hri =>
        (inputOf_restrict s.prog (lay s) (k s) _ tv hi).trans
          (inputOf_congr_at (lay s) (k s) _ (hpt i hi hfi hri))
    rw [hr]
    simp only [circuitProgram]
    funext ℓ
    exact rowOf_congr fun σ g hg => hv g (not_unread_of_out ((hfit s hs).1 t ht ℓ σ g hg).1)

/-! ### A disagreement with the registration is a collision -/

theorem mem_assemble_of_val {H : Msg → Msg} {Cm : CommitmentScheme Bool} {s : AuditStatement} {found : List (List Msg × Opening Bool)}
    {g : ℕ} {b : Bool} (h : (assemble s found).val H Cm s g = some b) :
    ∃ q ∈ found, q.1 = s.roots ∧ q.2.pos = g ∧ q.2.val = b := by
  unfold Wit.val at h
  obtain ⟨o, ho, rfl⟩ := Option.map_eq_some_iff.1 h
  have hpo := List.find?_some ho
  simp only [Bool.and_eq_true, beq_iff_eq] at hpo
  obtain ⟨q, hq, hqo⟩ := List.mem_filterMap.1 (List.mem_of_find?_eq_some ho)
  by_cases hr : q.1 = s.roots
  · simp only [hr, ite_true, Option.some.injEq] at hqo
    subst hqo
    exact ⟨q, hq, hr, hpo.1, rfl⟩
  · simp [hr] at hqo

theorem regVal_mem {ops : List (Opening Bool)} {g : ℕ} (h : ∃ o ∈ ops, o.pos = g) :
    ∃ o ∈ ops, o.pos = g ∧ o.val = regVal ops g := by
  obtain ⟨o, ho⟩ : ∃ o, ops.find? (fun o => o.pos == g) = some o :=
    Option.isSome_iff_exists.1 (List.find?_isSome.2 (by obtain ⟨o, hm, hp⟩ := h; exact ⟨o, hm, by simp [hp]⟩))
  refine ⟨o, List.mem_of_find?_eq_some ho, by simpa using List.find?_some ho, ?_⟩
  simp [regVal, ho]

theorem regLog_valid (H : Msg → Msg) (Cm : CommitmentScheme Bool) (roots : List Msg) (ops : List (Opening Bool)) :
    ∀ p ∈ regLog H Cm roots ops, H p.1 = p.2 := by
  intro p hp
  obtain ⟨o, -, hpo⟩ := List.mem_flatMap.1 hp
  exact OC.eval_log H _ p hpo

section Compiled

open Claims.SamplingGeneric (Arg Compiled)

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- The registration's collision: the extraction's own calls and the registration's checks hold two different inputs
with one output. -/
def RegColl (H : Msg → Msg) (D : Compiled Coin A) (ops : AuditStatement → List (Opening Bool))
    {Ω : Type} (P : Prover Ω) (x : Msg) (s : Ω × (Fin (D.V.k x) → Coin) × D.E.C x) : Prop :=
  ∃ p ∈ ownLog H (run H P D.V x s.1) (D.E.ext x s.2.1 s.2.2) ++
      ((D.parse x).map fun st => regLog H D.commit st.roots (ops st)).getD [],
    ∃ p' ∈ ownLog H (run H P D.V x s.1) (D.E.ext x s.2.1 s.2.2) ++
      ((D.parse x).map fun st => regLog H D.commit st.roots (ops st)).getD [],
    p.1 ≠ p'.1 ∧ p.2 = p'.2

/-- A kept opening of an accepted run passes, and its check's calls are the extraction's own. -/
theorem passes_of_mem_found (H : Msg → Msg) (D : Compiled Coin A) {Ω : Type}
    (P : Prover Ω) (x : Msg) (ω : Ω) (r₀ : Fin (D.V.k x) → Coin) (c : D.ECoins x) {st : AuditStatement}
    (hst : D.parse x = some st) (hacc : (run H P D.V x ω r₀).out = .acc) {q : List Msg × Opening Bool}
    (hq : q ∈ D.found H x st (run H P D.V x ω) c (D.outerList H x (run H P D.V x ω) c)) :
    D.commit.Opens H q.1 q.2 = true ∧
      ∀ t ∈ ((D.commit.opens q.1 q.2).eval H).2, t ∈ ownLog H (run H P D.V x ω) (D.E.ext x r₀ c) := by
  rw [Claims.SamplingGeneric.Compiled.found_eq] at hq
  obtain ⟨h1, h2⟩ := D.passes_found H x (run H P D.V x ω) st c _ q hq
  exact ⟨h1, fun t ht => D.ownLog_E H P x ω r₀ c hst hacc t (h2 t ht)⟩

/-- **A disagreement with the registration is a collision**: at an accepted run, an extracted trace that opens a
registered position to a value other than the registration's makes, with the registration's opening there, calls that
collide among the extraction's own and the registration's checks. -/
theorem regColl_of_disagree (H : Msg → Msg) (D : Compiled Coin A) (hb : D.commit.Binds H)
    (reg : AuditStatement → Finset ℕ) (ops : AuditStatement → List (Opening Bool)) {Ω : Type}
    (P : Prover Ω) (x : Msg) (s : Ω × (Fin (D.V.k x) → Coin) × D.E.C x) {st : AuditStatement}
    (hst : D.parse x = some st) (hreg : RegOpens H D.commit st (reg st) (ops st))
    (hacc : (run H P D.V x s.1 s.2.1).out = .acc) {w : Wit} (hw : (D.E.eval H P x s.1 s.2.1 s.2.2).1 = some w)
    {g : ℕ} (hg : g ∈ reg st) {b : Bool} (hwb : w.val H D.commit st g = some b) (hne : b ≠ regVal (ops st) g) :
    RegColl H D ops P x s := by
  rw [D.E_eval H P x s.1 s.2.1 s.2.2 hst hacc, Option.some.injEq] at hw
  subst hw
  obtain ⟨q, hq, hqr, hqp, hqv⟩ := mem_assemble_of_val hwb
  obtain ⟨hqo, hql⟩ := passes_of_mem_found H D P x s.1 s.2.1 s.2.2 hst hacc hq
  obtain ⟨o', ho', ho'p, ho'v⟩ := regVal_mem (hreg.2 g hg)
  have hqo' : D.commit.Opens H st.roots q.2 = true := by rw [← hqr]; exact hqo
  have hql' : ∀ t ∈ ((D.commit.opens st.roots q.2).eval H).2,
      t ∈ ownLog H (run H P D.V x s.1) (D.E.ext x s.2.1 s.2.2) := by
    intro t ht
    rw [← hqr] at ht
    exact hql t ht
  have hrl : ((D.parse x).map fun st => regLog H D.commit st.roots (ops st)).getD [] =
      regLog H D.commit st.roots (ops st) := by
    rw [hst]
    rfl
  unfold RegColl
  rw [hrl]
  exact disagree_collision hb (hqp.trans ho'p.symm) (by rw [hqv, ho'v]; exact hne) hqo' (hreg.1 o' ho')
    (fun t ht => List.mem_append_left _ (hql' t ht))
    (fun t ht => List.mem_append_right _ (List.mem_flatMap.2 ⟨o', ho', ht⟩))

/-- **The reduction's probability step**: the registration's collisions have probability at most `B_reg`'s. -/
theorem Pr_regColl_le_Breg (H : Msg → Msg) (D : Compiled Coin A)
    (ops : AuditStatement → List (Opening Bool)) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg) :
    Pr (fun s : Ω × (Fin (D.V.k x) → Coin) × D.E.C x => RegColl H D ops P x s) ≤
      (Breg H D.V D.E D.commit D.parse ops Ω P x).Pr H := by
  unfold Breg Finder.Pr
  refine Pr_mono fun s h => ?_
  obtain ⟨⟨u, v⟩, huv⟩ := Option.isSome_iff_exists.1 (findColl_isSome _ h)
  refine ⟨u, v, huv, findColl_spec H _ (fun p hp => ?_) u v huv⟩
  rcases List.mem_append.1 hp with hp | hp
  · exact ownLog_valid H _ _ p hp
  · cases hx : D.parse x with
    | none => rw [hx] at hp; exact absurd hp (by simp)
    | some st => rw [hx] at hp; exact regLog_valid H D.commit st.roots (ops st) p hp

/-- **The leak bound at the registration's values**, at any compiled sampled verifier `D` where `Compiled.KS` holds and
the trace's commitment binds, given `SampledAuditAccepted`. Take a public layout `lay` and horizon `k`, `δ ≥ 0`, each
statement's registered inputs `reg` and the registration's openings `ops` of them. On the statements in scope whose
registration opens their registered inputs:

`Pr[acc ∧ ¬RLeak] ≤ δ + ε_ks + ε_bind + Pr[B_reg]`, with the leak relation at the inputs pinned at the registration's
values (`AsReg` at `regVal`), a set the statement and the registration fix. -/
theorem sampledLeak_registered (hSA : Guarantees.SampledAuditAccepted) (H : Msg → Msg) (D : Compiled Coin A)
    (hks : D.KS H) (hb : D.commit.Binds H) (L K S : ℕ) (lay : AuditStatement → BoundaryGates L S)
    (k : AuditStatement → ℕ) (δ : ℝ) (hδ : 0 ≤ δ) (reg : AuditStatement → Finset ℕ)
    (ops : AuditStatement → List (Opening Bool)) (Dj : ℕ) (x : Msg) (hx : ScopeAcc D lay k x)
    (hreg : ∀ st, D.parse x = some st → RegOpens H D.commit st (reg st) (ops st)) (Ω : Type) [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) :
    PrBad H D.V
        (RLeak H D.commit D.parse D.partition L K S lay k δ
          (fun st _ => AsReg st.prog (lay st) (k st) (reg st) (regVal (ops st))) Dj) D.E x P ≤
      D.κ δ x + Real.sqrt (D.γ x * PrB H D.V P x) + (Bext H D.V D.E Ω P x).Pr H +
        (Breg H D.V D.E D.commit D.parse ops Ω P x).Pr H := by
  have hacc := hSA H D L K S lay k hks δ hδ x hx Ω P
  rw [one_mul] at hacc
  have hpart : ∀ st P', D.parse x = some st → D.partition st = some P' → P'.Partitions st.prog := by
    intro st P' hs hP'
    obtain ⟨s₁, P₁, hs₁, hP₁, hin⟩ := hx.1
    rw [hs] at hs₁
    obtain rfl := Option.some.inj hs₁
    rw [hP'] at hP₁
    obtain rfl := Option.some.inj hP₁
    exact hin.2.1
  have hfit : ∀ st, D.parse x = some st → (lay st).Fits st.prog (k st) := by
    intro st hs
    obtain ⟨s₂, hs₂, hf⟩ := hx.2
    rw [hs] at hs₂
    obtain rfl := Option.some.inj hs₂
    exact hf
  unfold PrBad at hacc ⊢
  calc Pr (fun s : Ω × (Fin (D.V.k x) → Coin) × D.E.C x =>
        (run H P D.V x s.1 s.2.1).out = .acc ∧
          ¬ (RLeak H D.commit D.parse D.partition L K S lay k δ
            (fun st _ => AsReg st.prog (lay st) (k st) (reg st) (regVal (ops st))) Dj).ext x
            (D.E.eval H P x s.1 s.2.1 s.2.2).1)
      ≤ Pr (fun s : Ω × (Fin (D.V.k x) → Coin) × D.E.C x =>
          ((run H P D.V x s.1 s.2.1).out = .acc ∧
            ¬ (RAcc H D.commit D.parse D.partition L K S lay k δ).ext x
              (D.E.eval H P x s.1 s.2.1 s.2.2).1) ∨ RegColl H D ops P x s) := by
        refine Pr_mono fun s ⟨hac, hR⟩ => ?_
        rcases hw : (D.E.eval H P x s.1 s.2.1 s.2.2).1 with _ | w
        · exact Or.inl ⟨hac, by simp [Rel.ext]⟩
        · rw [hw] at hR
          by_cases hA : RAcc H D.commit D.parse D.partition L K S lay k δ x w
          · refine Or.inr ?_
            by_contra hnc
            refine hR (rLeak_reg_of_rAcc reg (fun st => regVal (ops st)) hpart hfit (fun st hst g hg b hwb => ?_) hA)
            by_contra hne
            exact hnc (regColl_of_disagree H D hb reg ops P x s hst (hreg st hst) hac hw hg hwb hne)
          · exact Or.inl ⟨hac, hA⟩
    _ ≤ _ := Pr_union_le _ _
    _ ≤ _ := add_le_add hacc (Pr_regColl_le_Breg H D ops P x)

end Compiled

end NetTiming
