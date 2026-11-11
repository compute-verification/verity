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
# The leak bound at the network certifier's rows

The network certifier registers its records at the transfer gates (`CertifierRegisters`). On the sampling relation (`R δ`), every
proof block outside the incorrect set is correct on the trace's committed values, so the trace opens every transfer gate
there (each is committed: an output or an input). Where the trace opens a transfer gate it agrees with the network
certifier, unless the two openings collide (below). Fill in the network certifier's value at every gate the trace leaves
unopened (`fill`): a correct proof block stays correct (`Prog.correct_mono`, it reads only opened values), and the filled
trace carries the network certifier's records at every transfer gate (`carries_fill`), the incorrect proof blocks'
included, since the trace's own value there, where it has one, is the network certifier's. So `accepted_of` gives
`Accepted` with the incorrect proof blocks' committed gates held at the filled values, the advice the count charges (`2^β`
over `faultySets`). A held gate's input value is never read, any other input in a proof block is opened, and no egress
gate depends on an input in none (an unread input, `Prog.Unread`), so the inputs move back to the trace's own values
(`valOver_congr_held`): the network certifier's records are among the sequences `AccSeqsSampled` counts at the trace's
values (`rLeakCertifier_of_r`).

An extracted trace that disagrees with the network certifier opens a transfer gate to a value the network certifier's
opening there doesn't. The compiled extractor checked that opening with its own calls (`Compiled.passes_found`,
`Compiled.ownLog_E`), so under
`CommitmentScheme.Binds` its calls and those that check the network certifier's opening collide, and `B_tap` finds the
collision (`certifierColl_of_disagree`, `Pr_certifierColl_le_Btap`). `sampledLeak_certifier` is the bound:
`Pr[acc ∧ ¬RLeakCertifier] ≤ δ + ε_ks + ε_bind + Pr[B_tap]`.

The registered-input and collision steps follow the registration's (proofs' P1b), at the network certifier's openings.
-/

@[expose] public section

namespace NetTiming.TapProof

open Claims Claims.Sampling NetTiming NetTiming.SampledAuditProof

/-! ### The trace filled in with the network certifier's values -/

/-- The committed values `ref`, with `tv`'s value at every gate `ref` leaves unopened. -/
def fill (ref : ℕ → Option Bool) (tv : ℕ → Bool) (g : ℕ) : Option Bool := some ((ref g).getD (tv g))

theorem fill_ext {ref : ℕ → Option Bool} {tv : ℕ → Bool} : ∀ g b, ref g = some b → fill ref tv g = some b :=
  fun g b h => by simp [fill, h]

theorem traceVal_fill {C : Prog} {ref : ℕ → Option Bool} {tv : ℕ → Bool} {g : ℕ}
    (hnc : ∀ b, C.gates[g]? ≠ some (.const b)) (hag : ∀ b, ref g = some b → b = tv g) :
    traceVal C (fill ref tv) g = tv g := by
  rw [traceVal, read_of_not_const hnc]
  rcases hr : ref g with _ | b
  · simp [fill, hr]
  · simp [fill, hr, hag b hr]

theorem traceVal_fill_of {C : Prog} {ref : ℕ → Option Bool} {tv : ℕ → Bool} {g : ℕ}
    (hc : ∀ b, C.gates[g]? = some (.const b) → tv g = b) (hag : ∀ b, ref g = some b → b = tv g) :
    traceVal C (fill ref tv) g = tv g := by
  by_cases hk : ∃ b, C.gates[g]? = some (.const b)
  · obtain ⟨b, hb⟩ := hk
    rw [traceVal, read_const hb, hc b hb]
    rfl
  · exact traceVal_fill (fun b h => hk ⟨b, h⟩) hag

/-- **The filled trace carries the network certifier's records**: where the trace opens a transfer gate the network
certifier opens, at the network certifier's value, the trace filled in with the network certifier's values carries every
record the network certifier registers. -/
theorem carries_fill {L K S : ℕ} {H : Msg → Msg} {Cm : CommitmentScheme Bool} {lay : BoundaryGates L S} {k : ℕ}
    {op : ℕ → List (Obs L K S) → List Bool} {s : AuditStatement} {ops : List (Opening Bool)} {o : List (Obs L K S)}
    {ref : ℕ → Option Bool} (hfit : lay.Fits s.prog k) (htap : CertifierRegisters H Cm lay k op s ops o)
    (hag : ∀ g, (∃ q ∈ ops, q.pos = g) → ∀ b, ref g = some b → b = certifierVal ops g) :
    Carries lay s.prog (fill ref (certifierVal ops)) op k o := by
  intro t ht
  obtain ⟨hE, hI, ho, hrows, hop⟩ := htap.2 t ht
  refine ⟨ho, fun ℓ => ?_, ?_⟩
  · rw [hrows ℓ]
    exact rowOf_congr fun σ g hg => (traceVal_fill_of (hE ℓ σ g hg).1 (hag g (hE ℓ σ g hg).2)).symm
  · rw [hop]
    refine List.map_congr_left fun g hg => (traceVal_fill (fun b h => ?_) (hag g (hI g hg))).symm
    rw [hfit.2.1 t ht g hg] at h
    cases h

/-! ### The sampling relation gives the leak relation at the network certifier's rows -/

section Compiled

open Claims.SamplingGeneric (Arg Compiled)

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- **At the network certifier's rows**: on the audit's scope at `δ ≥ 0`, the sampling relation at a trace that agrees
with the network certifier wherever both open a gate gives the leak relation at the network certifier's rows, at any
inputs `As` that suffice (`AdviceIn`). -/
theorem rLeakCertifier_of_r (H : Msg → Msg) (D : Compiled Coin A) {L K S : ℕ} (lay : AuditStatement → BoundaryGates L S)
    (k : AuditStatement → ℕ) {δ : ℝ} (hδ : 0 ≤ δ) {As : AuditStatement → (ℕ → Bool) → Set (ℕ → Bool)} {Dj : ℕ}
    (tops : AuditStatement → List (Opening Bool))
    (hin : ∀ s P, D.partition s = some P → ∀ ρ, AdviceIn K s.prog P s.design (lay s) (k s) ρ (As s ρ) δ) {x : Msg}
    (hx : ScopeAcc D lay k x) {w : Wit} (hR : R H D.commit D.parse D.partition δ x w)
    (hagree : ∀ s, D.parse x = some s → (∀ q ∈ tops s, D.commit.Opens H s.roots q = true) →
      ∀ g, (∃ q ∈ tops s, q.pos = g) → ∀ b, w.val H D.commit s g = some b → b = certifierVal (tops s) g) :
    RLeakCertifier H D.commit D.parse D.partition L K S lay k δ As Dj tops x w := by
  obtain ⟨s, P, hs, hP, hM⟩ := hR
  obtain ⟨⟨s₁, P₁, hs₁, hP₁, hsc⟩, s₂, hs₂, hfit⟩ := hx
  rw [hs] at hs₁ hs₂
  obtain rfl := Option.some.inj hs₁
  obtain rfl := Option.some.inj hs₂
  rw [hP] at hP₁
  obtain rfl := Option.some.inj hP₁
  have hF := faulty_mem_faultySets hM (disjoint_allSlots hδ hM)
  obtain ⟨ι, hι⟩ := correct_of_not_faulty H D.commit s P w
  refine ⟨s, P, hs, hP, fun op o htap hcov hl hD => ?_⟩
  have hext : ∀ g b, w.val H D.commit s g = some b → fill (w.val H D.commit s) (certifierVal (tops s)) g = some b :=
    fill_ext
  have hcar := carries_fill hfit htap (hagree s hs htap.1)
  have hacc := accepted_of hsc.1 hsc.2.1 hfit ι
    (fun v u hu hv => Prog.correct_mono (Prog.refIn_mono hext) (hι v u hu hv)) hcar hcov
  have hacc' : Accepted (circuitProgram s.prog (lay s) (k s)) op (k s) o
      (traceVal s.prog (w.val H D.commit s),
        heldAdv s.prog P (faulty H D.commit s P w) (fill (w.val H D.commit s) (certifierVal (tops s)))) := by
    intro t ht
    obtain ⟨h, hr, hst⟩ := hacc t ht
    refine ⟨h, ?_, hst⟩
    have hv : ∀ g, ¬ s.prog.Unread g →
        valOver s.prog ((lay s).inputOf (k s) (ingressUpTo op o t)
          (traceVal s.prog (fill (w.val H D.commit s) (certifierVal (tops s)))))
          (heldAdv s.prog P (faulty H D.commit s P w) (fill (w.val H D.commit s) (certifierVal (tops s)))) g =
        valOver s.prog ((lay s).inputOf (k s) (ingressUpTo op o t) (traceVal s.prog (w.val H D.commit s)))
          (heldAdv s.prog P (faulty H D.commit s P w) (fill (w.val H D.commit s) (certifierVal (tops s)))) g :=
      valOver_congr_held s.prog _ fun i hi hfi hri => inputOf_congr_at (lay s) (k s) _ (by
        have hnf : ¬ FaultyGate s.prog P (faulty H D.commit s P w) i := fun hfg => by
          rw [heldAdv_of hfg] at hfi
          cases hfi
        obtain ⟨b, hb⟩ := val_some_of_input hsc.2.1 hi hri hnf
        rw [traceVal_of_val hi hb, traceVal_of_val hi (hext i b hb)])
    rw [hr]
    simp only [circuitProgram]
    funext ℓ
    exact rowOf_congr fun σ g hg => hv g (not_unread_of_out (hfit.1 t ht ℓ σ g hg).1)
  exact ⟨hl, hD, _, hF, hin s P hP _ _ hF op o
    ⟨_, Set.mem_singleton _, _, fun g b h => faultyGate_of_heldAdv h, hacc'⟩⟩

/-! ### A disagreement with the network certifier is a collision -/

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

/-- **The leak bound at the network certifier's rows**, at any compiled sampled verifier `D` where `Compiled.KS` holds
and the trace's commitment binds. Take a public layout `lay` and horizon `k`, `δ ≥ 0`, inputs `As` that suffice
(`AdviceIn`), and the network certifier's openings `tops s` at each statement. On the statements in the audit's scope:
`Pr[acc ∧ ¬RLeakCertifier] ≤ δ + ε_ks + ε_bind + Pr[B_tap]`. -/
theorem sampledLeak_certifier (H : Msg → Msg) (D : Compiled Coin A) (hks : D.KS H) (hb : D.commit.Binds H) (L K S : ℕ)
    (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) (δ : ℝ) (hδ : 0 ≤ δ)
    (As : AuditStatement → (ℕ → Bool) → Set (ℕ → Bool)) (Dj : ℕ)
    (hin : ∀ s P, D.partition s = some P → ∀ ρ, AdviceIn K s.prog P s.design (lay s) (k s) ρ (As s ρ) δ)
    (tops : AuditStatement → List (Opening Bool)) (x : Msg) (hx : ScopeAcc D lay k x) (Ω : Type) [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) :
    PrBad H D.V (RLeakCertifier H D.commit D.parse D.partition L K S lay k δ As Dj tops) D.E x P ≤
      D.κ δ x + Real.sqrt (D.γ x * PrB H D.V P x) + (Bext H D.V D.E Ω P x).Pr H +
        (Btap H D.V D.E D.commit D.parse tops Ω P x).Pr H := by
  have hks' := hks δ hδ x hx.1 Ω P
  rw [one_mul] at hks'
  unfold PrBad at hks' ⊢
  calc Pr (fun s : Ω × (Fin (D.V.k x) → Coin) × D.E.C x =>
        (run H P D.V x s.1 s.2.1).out = .acc ∧
          ¬ (RLeakCertifier H D.commit D.parse D.partition L K S lay k δ As Dj tops).ext x
            (D.E.eval H P x s.1 s.2.1 s.2.2).1)
      ≤ Pr (fun s : Ω × (Fin (D.V.k x) → Coin) × D.E.C x =>
          ((run H P D.V x s.1 s.2.1).out = .acc ∧
            ¬ (R H D.commit D.parse D.partition δ).ext x (D.E.eval H P x s.1 s.2.1 s.2.2).1) ∨
            CertifierColl H D tops P x s) := by
        refine Pr_mono fun s ⟨hac, hR⟩ => ?_
        rcases hw : (D.E.eval H P x s.1 s.2.1 s.2.2).1 with _ | w
        · exact Or.inl ⟨hac, by simp [Rel.ext]⟩
        · rw [hw] at hR
          by_cases hA : R H D.commit D.parse D.partition δ x w
          · refine Or.inr ?_
            by_contra hnc
            refine hR (rLeakCertifier_of_r H D lay k hδ tops hin hx hA fun st hst hpass g hq b hwb => ?_)
            by_contra hne
            exact hnc (certifierColl_of_disagree H D hb tops P x s hst hpass hac hw hq hwb hne)
          · exact Or.inl ⟨hac, hA⟩
    _ ≤ _ := Pr_union_le _ _
    _ ≤ _ := add_le_add hks' (Pr_certifierColl_le_Btap H D tops P x)

end Compiled

end NetTiming.TapProof
