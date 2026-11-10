module

public import Definitions.NetworkCertifier.TapRows
public import Proofs.NetworkCertifier.SampledLeakReg
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
import Proofs.Flock.Claims.Sampling.ProgReads
import Proofs.NetworkCertifier.SampledAuditAccepted
import Proofs.NetworkCertifier.SampledCharge
import Proofs.NetworkCertifier.SampledLeak

/-!
# At the network certifier's rows: the filled trace, and a disagreement as a collision

The network certifier registers its records at the transfer gates (`CertifierRegisters`). On the sampling relation
(`R δ`), every proof block outside the incorrect set is correct on the trace's committed values, so the trace opens
every transfer gate there (each is committed: an output or an input). Where the trace opens a transfer gate it agrees
with the network certifier, unless the two openings collide (below). Fill in the network certifier's value at every
gate the trace leaves unopened (`fillTrace`): a correct proof block stays correct (`Prog.correct_mono`, it reads only
opened values), and the filled trace carries the network certifier's records at every transfer gate
(`carries_fillTrace`), the incorrect proof blocks' included, since the trace's own value there, where it has one, is
the network certifier's. `PeriodChecked` takes the leak relation at the network certifier's rows from it
(`rLeakCertifierChecked_of`).

An extracted trace that disagrees with the network certifier opens a transfer gate to a value the network certifier's
opening there doesn't. The compiled extractor checked that opening with its own calls (`Compiled.passes_found`,
`Compiled.ownLog_E`), so under `CommitmentScheme.Binds` its calls and those that check the network certifier's opening
collide, and `B_tap` finds the collision (`certifierColl_of_disagree`, `Pr_certifierColl_le_Btap`).

The collision step follows the registration's (proofs' P1b), at the network certifier's openings.
-/

@[expose] public section

namespace NetTiming.TapProof

open Claims Claims.Sampling NetTiming NetTiming.SampledAuditProof

/-! ### The trace filled in with the network certifier's values -/

theorem fillTrace_ext {ref : ℕ → Option Bool} {tv : ℕ → Bool} : ∀ g b, ref g = some b → fillTrace ref tv g = some b :=
  fun g b h => by simp [fillTrace, h]

theorem traceVal_fillTrace {C : Prog} {ref : ℕ → Option Bool} {tv : ℕ → Bool} {g : ℕ}
    (hnc : ∀ b, C.gates[g]? ≠ some (.const b)) (hag : ∀ b, ref g = some b → b = tv g) :
    traceVal C (fillTrace ref tv) g = tv g := by
  rw [traceVal, read_of_not_const hnc]
  rcases hr : ref g with _ | b
  · simp [fillTrace, hr]
  · simp [fillTrace, hr, hag b hr]

theorem traceVal_fillTrace_of {C : Prog} {ref : ℕ → Option Bool} {tv : ℕ → Bool} {g : ℕ}
    (hc : ∀ b, C.gates[g]? = some (.const b) → tv g = b) (hag : ∀ b, ref g = some b → b = tv g) :
    traceVal C (fillTrace ref tv) g = tv g := by
  by_cases hk : ∃ b, C.gates[g]? = some (.const b)
  · obtain ⟨b, hb⟩ := hk
    rw [traceVal, read_const hb, hc b hb]
    rfl
  · exact traceVal_fillTrace (fun b h => hk ⟨b, h⟩) hag

/-- **The filled trace carries the network certifier's records**: where the trace opens a transfer gate the network
certifier opens, at the network certifier's value, the trace filled in with the network certifier's values carries every
record the network certifier registers. -/
theorem carries_fillTrace {L K S : ℕ} {H : Msg → Msg} {Cm : CommitmentScheme Bool} {lay : BoundaryGates L S} {k : ℕ}
    {op : ℕ → List (Obs L K S) → List Bool} {s : AuditStatement} {ops : List (Opening Bool)} {o : List (Obs L K S)}
    {ref : ℕ → Option Bool} (hfit : lay.Fits s.prog k) (htap : CertifierRegisters H Cm lay k op s ops o)
    (hag : ∀ g, (∃ q ∈ ops, q.pos = g) → ∀ b, ref g = some b → b = certifierVal ops g) :
    Carries lay s.prog (fillTrace ref (certifierVal ops)) op k o := by
  intro t ht
  obtain ⟨hE, hI, ho, hrows, hop⟩ := htap.2 t ht
  refine ⟨ho, fun ℓ => ?_, ?_⟩
  · rw [hrows ℓ]
    exact rowOf_congr fun σ g hg => (traceVal_fillTrace_of (hE ℓ σ g hg).1 (hag g (hE ℓ σ g hg).2)).symm
  · rw [hop]
    refine List.map_congr_left fun g hg => (traceVal_fillTrace (fun b h => ?_) (hag g (hI g hg))).symm
    rw [hfit.2.1 t ht g hg] at h
    cases h

/-! ### A disagreement with the network certifier is a collision -/

section Compiled

open Claims.SamplingGeneric (Arg Compiled)

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

theorem certifierVal_mem {ops : List (Opening Bool)} {g : ℕ} (h : ∃ o ∈ ops, o.pos = g) :
    ∃ o ∈ ops, o.pos = g ∧ o.val = certifierVal ops g := by
  obtain ⟨o, ho⟩ : ∃ o, ops.find? (fun o => o.pos == g) = some o :=
    Option.isSome_iff_exists.1 (List.find?_isSome.2 (by obtain ⟨o, hm, hp⟩ := h; exact ⟨o, hm, by simp [hp]⟩))
  refine ⟨o, List.mem_of_find?_eq_some ho, by simpa using List.find?_some ho, ?_⟩
  simp [certifierVal, ho]

theorem certifierLog_valid (H : Msg → Msg) (Cm : CommitmentScheme Bool) (roots : List Msg) (ops : List (Opening Bool)) :
    ∀ p ∈ certifierLog H Cm roots ops, H p.1 = p.2 := by
  intro p hp
  obtain ⟨o, -, hpo⟩ := List.mem_flatMap.1 hp
  exact OC.eval_log H _ p hpo

/-- A collision against the network certifier's openings: the extraction's own calls and the checks of the network
certifier's openings hold two different inputs with one output. -/
def CertifierColl (H : Msg → Msg) (D : Compiled Coin A)
    (tops : AuditStatement → List (Opening Bool)) {Ω : Type} (P : Prover Ω) (x : Msg)
    (s : Ω × (Fin (D.V.k x) → Coin) × D.E.C x) : Prop :=
  LogColl (ownLog H (run H P D.V x s.1) (D.E.ext x s.2.1 s.2.2) ++
    ((D.parse x).map fun st => certifierLog H D.commit st.roots (tops st)).getD [])

/-- **A disagreement with the network certifier is a collision**: at an accepted run, an extracted trace that opens a
gate the network certifier opens to a value other than the network certifier's makes, with the network certifier's
opening there, calls that collide among the extraction's own and the checks of the network certifier's openings. -/
theorem certifierColl_of_disagree (H : Msg → Msg) (D : Compiled Coin A) (hb : D.commit.Binds H)
    (tops : AuditStatement → List (Opening Bool)) {Ω : Type} (P : Prover Ω) (x : Msg)
    (s : Ω × (Fin (D.V.k x) → Coin) × D.E.C x) {st : AuditStatement} (hst : D.parse x = some st)
    (hpass : ∀ q ∈ tops st, D.commit.Opens H st.roots q = true) (hacc : (run H P D.V x s.1 s.2.1).out = .acc)
    {w : Wit} (hw : (D.E.eval H P x s.1 s.2.1 s.2.2).1 = some w) {g : ℕ} (hq : ∃ q ∈ tops st, q.pos = g)
    {b : Bool} (hwb : w.val H D.commit st g = some b) (hne : b ≠ certifierVal (tops st) g) :
    CertifierColl H D tops P x s := by
  rw [D.E_eval H P x s.1 s.2.1 s.2.2 hst hacc, Option.some.injEq] at hw
  subst hw
  obtain ⟨q, hq', hqr, hqp, hqv⟩ := mem_assemble_of_val hwb
  obtain ⟨hqo, hql⟩ := passes_of_mem_found H D P x s.1 s.2.1 s.2.2 hst hacc hq'
  obtain ⟨o', ho', ho'p, ho'v⟩ := certifierVal_mem hq
  have hqo' : D.commit.Opens H st.roots q.2 = true := by rw [← hqr]; exact hqo
  have hql' : ∀ t ∈ ((D.commit.opens st.roots q.2).eval H).2,
      t ∈ ownLog H (run H P D.V x s.1) (D.E.ext x s.2.1 s.2.2) := by
    intro t ht
    rw [← hqr] at ht
    exact hql t ht
  have htl : ((D.parse x).map fun st => certifierLog H D.commit st.roots (tops st)).getD [] =
      certifierLog H D.commit st.roots (tops st) := by
    rw [hst]
    rfl
  unfold CertifierColl
  rw [htl]
  exact disagree_collision hb (hqp.trans ho'p.symm) (by rw [hqv, ho'v]; exact hne) hqo' (hpass o' ho')
    (fun t ht => List.mem_append_left _ (hql' t ht))
    (fun t ht => List.mem_append_right _ (List.mem_flatMap.2 ⟨o', ho', ht⟩))

/-- **The reduction's probability step**: the collisions against the network certifier's openings have probability at
most `B_tap`'s. -/
theorem Pr_certifierColl_le_Btap (H : Msg → Msg) (D : Compiled Coin A)
    (tops : AuditStatement → List (Opening Bool)) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg) :
    Pr (fun s : Ω × (Fin (D.V.k x) → Coin) × D.E.C x => CertifierColl H D tops P x s) ≤
      (Btap H D.V D.E D.commit D.parse tops Ω P x).Pr H := by
  unfold Btap Finder.Pr
  refine Pr_mono fun s h => ?_
  obtain ⟨⟨u, v⟩, huv⟩ := Option.isSome_iff_exists.1 (findColl_isSome _ h)
  refine ⟨u, v, huv, findColl_spec H _ (fun p hp => ?_) u v huv⟩
  rcases List.mem_append.1 hp with hp | hp
  · exact ownLog_valid H _ _ p hp
  · cases hx : D.parse x with
    | none => rw [hx] at hp; exact absurd hp (by simp)
    | some st => rw [hx] at hp; exact certifierLog_valid H D.commit st.roots (tops st) p hp

end Compiled

end NetTiming.TapProof
