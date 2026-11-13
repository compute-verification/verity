module

public import Definitions.NetworkCertifier.Guarantees.PeriodChecked
public import Proofs.NetworkCertifier.Carried
public import Proofs.NetworkCertifier.Checked
public import Proofs.NetworkCertifier.SampledAuditAccepted
public import Proofs.NetworkCertifier.SampledCharge
public import Proofs.NetworkCertifier.SampledLeak
public import Proofs.NetworkCertifier.SampledLeakReg
public import Proofs.NetworkCertifier.TapRows
public import Proofs.Flock.Claims.Finder
public import Proofs.Flock.Claims.Prob
public import Proofs.Flock.Claims.Sampling.Binding
public import Proofs.Flock.Claims.Sampling.ProgReads
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Data.NNRat.Floor
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.Order.CompletePartialOrder
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.RingTheory.WittVector.IsPoly
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.ENatToNat
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

/-!
# The leak through the network certifier over a monitoring period, with the checked inputs left out

`periodLeakChecked` is `Guarantees.PeriodLeakChecked`, in two steps, each over the checked form of what it reads.

**Each window's bound.** On the audit's scope, the sampling relation at a trace that agrees with the network certifier
wherever both open a gate gives the leak relation at the network certifier's rows with a valid run, where the run at
those rows (`certifierRun`) is `valid` (`rLeakCertifierChecked_of`). The trace filled in with the network certifier's
values carries its records (`TapProof.carries_fillTrace`), so `accepted_of` gives `Accepted` with the incorrect proof
blocks' committed gates held at the filled values, the advice the count charges. A held gate's input value is never
read, any other input in a proof block is opened, and no egress gate depends on an input in none (an unread input,
`Prog.Unread`), so the inputs move back to the trace's own values (`valOver_congr_held`), pinned at the free inputs:
the run the relation accepts at is the run at the network certifier's rows. An accepted audit escapes the relation
only where `Compiled.KS` fails, the trace disagrees with a passing opening of the network certifier, which is a
collision (`certifierColl_of_disagree`, `Pr_certifierColl_le_Btap`), or one of the two checks misses on that run,
which `CheckedWitness` at `certifierRun` bounds (`periodChecked_first`).

**The count.** A window that meets `WindowGoodChecked`, whose observations the network certifier registers, has them
among the sequences `AccSeqsChecked` counts at its registration (`mem_accSeqsChecked_of_windowGoodChecked`). A period's
records whose every window is such a sequence, each window's operator reading the windows before it, are, at the
incorrect sets they are accepted with, runs over the empty pool that restart at no window, which `encard_carried_le`
counts with the checked inputs left out; over the windows' possible incorrect sets they number at most `windowsCount`
at `reg ∪ chk` (`encard_periodChecked_le`). The client's view reads back from the records under a registered parameter
(`RecordFormat.ReadsBy`), so the views, and the secrets they decode to, number at most `#Θ` times that
(`periodChecked_second`).

`rLeakCertifier_of_rLeakCertifierChecked` and `rLeakCertifierChecked_of_rLeakCertifier` move between the leak relation
at the network certifier's rows and its checked form, the second at a check every run passes;
`RecordFormat.readsBy_read` reads every record format back under its own schedule as the parameter. They carry
`PeriodLeakAccepted` to its instance of this statement.
-/

@[expose] public section

namespace NetTiming

open Claims Claims.Sampling Finset

theorem asReg_congr {L S : ℕ} (C : Prog) (lay : BoundaryGates L S) (k : ℕ) (R : Finset ℕ) {ρ ρ' : ℕ → Bool}
    (h : ∀ g ∈ R, ρ g = ρ' g) : AsReg C lay k R ρ = AsReg C lay k R ρ' := by
  ext a
  constructor <;> rintro ⟨h1, h2⟩ <;> refine ⟨fun g hg hf => ?_, h2⟩
  · rw [h1 g hg hf, h g hg]
  · rw [h1 g hg hf, ← h g hg]

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

/-- **At a check every run passes, the leak relation at the network certifier's rows is its checked form.** -/
theorem rLeakCertifierChecked_of_rLeakCertifier {H : Msg → Msg} {Cm : CommitmentScheme Bool}
    {parse : Msg → Option AuditStatement} {partition : AuditStatement → Option Partition} {L K S : ℕ}
    {lay : AuditStatement → BoundaryGates L S} {k : AuditStatement → ℕ} {δ : ℝ}
    {As : AuditStatement → (ℕ → Bool) → Set (ℕ → Bool)} {valid : AuditStatement → (ℕ → Bool) → Prop}
    (hv : ∀ s r, valid s r) {Dj : ℕ} {tops : AuditStatement → List (Opening Bool)} {x : Msg} {w : Wit}
    (h : RLeakCertifier H Cm parse partition L K S lay k δ As Dj tops x w) :
    RLeakCertifierChecked H Cm parse partition L K S lay k δ As valid Dj tops x w := by
  obtain ⟨s, P, hs, hP, h⟩ := h
  refine ⟨s, P, hs, hP, fun op o hC hCov hl hD => ?_⟩
  obtain ⟨hl', hD', F, hF, a, ha, free, hfree, hacc⟩ := h op o hC hCov hl hD
  exact ⟨hl', hD', F, hF, a, ha, free, hfree, hv s _, hacc⟩

/-- **Every record format reads back under its own schedule as the parameter** (`RecordFormat.read_record`). -/
theorem RecordFormat.readsBy_read {κ Cell Ob : Type} (f : RecordFormat κ Cell Ob) :
    f.ReadsBy f.read fun θ s => θ = s := by
  rintro θ s rfl hfit a
  exact f.read_record _ hfit a

/-- **The count of a period's records, with the checked inputs left out.** The records whose every window is a
sequence its registered Program accepts with a valid run (`AccSeqsChecked`), window `i`'s operator reading the windows
before it, number at most `windowsCount` at `reg ∪ chk`. At the incorrect sets `F i` its windows are accepted with,
such a record is a run over the empty pool that restarts at no window (`leakAtCarried_of_leakAtChecked`), and
`encard_carried_le` counts those at each family `F`; the families are the product of the windows' possible incorrect
sets, and the sum over them of the per-family products is the product of the per-window sums
(`Finset.prod_univ_sum`), each at most `#faultySets_i · 2^β_i` times the window's other factors. -/
theorem encard_periodChecked_le {L K S n : ℕ} (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ)
    (δ : ℝ) (reg chk : AuditStatement → Finset ℕ) (valid : AuditStatement → (ℕ → Bool) → Prop) (Dj : ℕ)
    (sw : Fin n → AuditStatement) (Pw : Fin n → Partition) (β : Fin n → ℕ)
    (hβ : ∀ i, ∀ F ∈ faultySets (sw i).design (Pw i) δ, faultyBits (sw i).prog (Pw i) F ≤ β i)
    (hfits : ∀ i, (lay (sw i)).Fits (sw i).prog (k (sw i)))
    (hchk : ∀ i, CheckedUnique (sw i).prog (Pw i) (sw i).design (lay (sw i)) (k (sw i)) (chk (sw i)) (valid (sw i)))
    (ρ : Fin n → ℕ → Bool) (op : Fin n → List (List (Obs L K S)) → ℕ → List (Obs L K S) → List Bool) :
    {o : Fin n → List (Obs L K S) | ∀ i, o i ∈ AccSeqsChecked (sw i).prog (Pw i) (sw i).design (lay (sw i))
        (k (sw i)) (AsReg (sw i).prog (lay (sw i)) (k (sw i)) (reg (sw i)) (ρ i)) (valid (sw i)) δ
        (op i (runsBefore o i)) Dj}.encard ≤
      ((windowsCount lay k δ (fun s => reg s ∪ chk s) Dj sw Pw β : ℕ) : ℕ∞) := by
  classical
  have hcls : ∀ i, CheckedClosing (sw i).prog (Pw i) (sw i).design (lay (sw i)) (k (sw i)) (chk (sw i))
      (valid (sw i)) [] := fun _ => by unfold CheckedClosing; intros; rfl
  let c : Fin n → ℕ := fun i =>
    2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \ (reg (sw i) ∪ chk (sw i))).card
  let Sq : (Fin n → Finset ℕ) → Set (Fin n → List (Obs L K S)) := fun F =>
    {o | ∃ rin rout : Fin n → Hash.Row, PoolChained (fun _ => false) (fun _ => []) rin rout ∧
      ∀ i, LeakAtCarried lay k reg valid (fun _ => []) (fun _ => []) Dj (sw i) (Pw i) (ρ i)
        (op i (runsBefore o i)) (rin i) (rout i) (F i) (o i)}
  have hSq : ∀ F, (Sq F).encard ≤
      ((∏ i, c i * (2 ^ faultyBits (sw i).prog (Pw i) (F i) * Dj ^ k (sw i)) : ℕ) : ℕ∞) := fun F => by
    have h := encard_carried_le lay k reg chk valid (fun _ => []) (fun _ => []) Dj sw Pw ρ op hchk hcls
      (fun _ => false) (fun _ => []) F
    simp only [List.toFinset_nil, Finset.union_empty] at h
    exact h
  have hsub : {o : Fin n → List (Obs L K S) | ∀ i, o i ∈ AccSeqsChecked (sw i).prog (Pw i) (sw i).design
        (lay (sw i)) (k (sw i)) (AsReg (sw i).prog (lay (sw i)) (k (sw i)) (reg (sw i)) (ρ i)) (valid (sw i)) δ
        (op i (runsBefore o i)) Dj} ⊆
      ⋃ F ∈ Fintype.piFinset (fun i => faultySets (sw i).design (Pw i) δ), Sq F := by
    intro o ho
    have ho' : ∀ i, (o i).length = k (sw i) ∧ (∀ y ∈ o i, y.jitter < Dj) ∧
        ∃ F ∈ faultySets (sw i).design (Pw i) δ, AcceptedChecked (sw i).prog (Pw i) (lay (sw i)) (k (sw i))
          (AsReg (sw i).prog (lay (sw i)) (k (sw i)) (reg (sw i)) (ρ i)) (valid (sw i)) F
          (op i (runsBefore o i)) (o i) := fun i => ho i
    choose hl hj F hF hacc using ho'
    exact Set.mem_iUnion₂.2 ⟨F, Fintype.mem_piFinset.2 hF, fun _ => [], fun _ => [], fun _ => by simp [poolOpen],
      fun i => leakAtCarried_of_leakAtChecked
        (show LeakAtChecked lay k reg valid Dj (sw i) (Pw i) (ρ i) (op i (runsBefore o i)) (F i) (o i) from
          ⟨hfits i, disjoint_of_mem_faultySets (hF i), hl i, hj i, hacc i⟩)⟩
  have hi : ∀ i, ∑ F ∈ faultySets (sw i).design (Pw i) δ,
        c i * (2 ^ faultyBits (sw i).prog (Pw i) F * Dj ^ k (sw i)) ≤
      c i * (faultySets (sw i).design (Pw i) δ).card * 2 ^ β i * Dj ^ k (sw i) := fun i =>
    (Finset.sum_le_card_nsmul _ _ (c i * (2 ^ β i * Dj ^ k (sw i))) fun F hF =>
      Nat.mul_le_mul_left _ (Nat.mul_le_mul_right _ (Nat.pow_le_pow_right (by norm_num) (hβ i F hF)))).trans
      (le_of_eq (by rw [smul_eq_mul]; ring))
  have hps : ∏ i, ∑ F ∈ faultySets (sw i).design (Pw i) δ,
        c i * (2 ^ faultyBits (sw i).prog (Pw i) F * Dj ^ k (sw i)) =
      ∑ F ∈ Fintype.piFinset (fun i => faultySets (sw i).design (Pw i) δ),
        ∏ i, c i * (2 ^ faultyBits (sw i).prog (Pw i) (F i) * Dj ^ k (sw i)) := Finset.prod_univ_sum ..
  have hnat : ∑ F ∈ Fintype.piFinset (fun i => faultySets (sw i).design (Pw i) δ),
        ∏ i, c i * (2 ^ faultyBits (sw i).prog (Pw i) (F i) * Dj ^ k (sw i)) ≤
      windowsCount lay k δ (fun s => reg s ∪ chk s) Dj sw Pw β := by
    refine hps.symm.le.trans ?_
    unfold windowsCount
    exact Finset.prod_le_prod fun i _ => hi i
  calc _ ≤ (⋃ F ∈ Fintype.piFinset (fun i => faultySets (sw i).design (Pw i) δ), Sq F).encard :=
        Set.encard_le_encard hsub
    _ ≤ ∑ F ∈ Fintype.piFinset (fun i => faultySets (sw i).design (Pw i) δ), (Sq F).encard :=
        encard_biUnion_le _ _
    _ ≤ ∑ F ∈ Fintype.piFinset (fun i => faultySets (sw i).design (Pw i) δ),
          ((∏ i, c i * (2 ^ faultyBits (sw i).prog (Pw i) (F i) * Dj ^ k (sw i)) : ℕ) : ℕ∞) :=
        Finset.sum_le_sum fun F _ => hSq F
    _ ≤ _ := by exact_mod_cast hnat

section Compiled

open Claims.SamplingGeneric (Arg Compiled)

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- **A window that meets its good event holds an accepted sequence**: where the network certifier registers the
window's observations at the operator `op`, the layout fits the window's registration and the observations are among
the sequences `AccSeqsChecked` counts there, at the registration's values. -/
theorem mem_accSeqsChecked_of_windowGoodChecked {H : Msg → Msg} {D : Compiled Coin A} {L K S : ℕ}
    {lay : AuditStatement → BoundaryGates L S} {k : AuditStatement → ℕ} {δ : ℝ} {reg : AuditStatement → Finset ℕ}
    {valid : AuditStatement → (ℕ → Bool) → Prop} {Dj : ℕ} {s₀ : AuditStatement} {P₀ : Partition}
    (hP₀ : D.partition s₀ = some P₀) {ρ : ℕ → Bool} {tops : AuditStatement → List (Opening Bool)}
    {op : ℕ → List (Obs L K S) → List Bool} {x : Msg} {w : Wit} {o : List (Obs L K S)}
    (htap : ∀ s, D.parse x = some s → CertifierRegisters H D.commit (lay s) (k s) op s (tops s) o)
    (h : WindowGoodChecked H D L K S lay k δ reg valid Dj s₀ ρ tops x w o) :
    (lay s₀).Fits s₀.prog (k s₀) ∧
      o ∈ AccSeqsChecked s₀.prog P₀ s₀.design (lay s₀) (k s₀) (AsReg s₀.prog (lay s₀) (k s₀) (reg s₀) ρ)
        (valid s₀) δ op Dj := by
  obtain ⟨⟨s₁, P₁, hs₁, hP₁, hl⟩, ⟨-, s₂, hs₂, hfits⟩, s, hs, ⟨hprog, hdes, hlay, hk, hreg, hpart⟩, hval, hρ, hcov,
    hlen, hj⟩ := h
  rw [Option.some_injective _ (hs₁.symm.trans hs)] at hP₁ hl
  rw [Option.some_injective _ (hs₂.symm.trans hs)] at hfits
  rw [Option.some_injective _ (hP₁.symm.trans (hpart.trans hP₀))] at hl
  have hρ' : ∀ g ∈ reg s, traceVal s.prog (w.val H D.commit s) g = ρ g := hρ
  have hmem := hl op o (htap s hs) hcov hlen hj
  beta_reduce at hmem
  rw [asReg_congr _ _ _ _ hρ'] at hmem
  rw [hval, hprog, hdes, hlay, hk, hreg] at hmem
  rw [hprog, hlay, hk] at hfits
  exact ⟨hfits, hmem⟩

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- **The held and counted parts of `Guarantees.PeriodLeakChecked`**, at one check `valid`: a good secret's view is read
from the windows' records under a registered parameter, each record a sequence its registered Program accepts with a
valid run (`mem_accSeqsChecked_of_windowGoodChecked`), so the good secrets are decoded from the views `rd` reads off
`Θ` and the records `encard_periodChecked_le` counts. -/
theorem periodChecked_second (H : Msg → Msg) (D : Compiled Coin A) (L K S : ℕ)
    (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) (δ : ℝ)
    (reg chk : AuditStatement → Finset ℕ) (valid : AuditStatement → (ℕ → Bool) → Prop) (Dj : ℕ)
    {n : ℕ} (sw : Fin n → AuditStatement) (Pw : Fin n → Partition) (β : Fin n → ℕ)
    (hplaced : WindowsPlaced D.partition δ sw Pw β)
    (hchk : ∀ i, CheckedUnique (sw i).prog (Pw i) (sw i).design (lay (sw i)) (k (sw i)) (chk (sw i)) (valid (sw i)))
    (ρ : Fin n → ℕ → Bool) (op : Fin n → List (List (Obs L K S)) → ℕ → List (Obs L K S) → List Bool)
    {Conn Cell Srv Dev Zt Par : Type} (arr : Srv → Conn → Arrivals Cell)
    (f : RecordFormat Conn Cell (Fin n → List (Obs L K S))) (tap : Dev)
    (run : Dev → CertifierSchedule Conn × Srv → (Conn → List (ℕ × CellView Cell)) × (Fin n → List (Obs L K S)))
    (htrust : Verity.Assumptions.TrustedHost tap run fun p => recordingCode f p.1 (arr p.2))
    (rd : Par → (Fin n → List (Obs L K S)) → Conn → List (ℕ × CellView Cell))
    (allows : Par → CertifierSchedule Conn → Prop) (hread : f.ReadsBy rd allows)
    (Θ : Set Par) (Zs : Set Zt) (schedOf : Zt → CertifierSchedule Conn) (srvOf : Zt → Srv)
    (dec : (Conn → List (ℕ × CellView Cell)) → Zt)
    (hch : PeriodChannelReg f allows Θ Zs schedOf srvOf dec fun p => (run tap p).1)
    (x : Zt → Fin n → Msg) (wit : Zt → Fin n → Wit) (tops : AuditStatement → List (Opening Bool))
    (htap : ∀ z ∈ Zs, ∀ i s, D.parse (x z i) = some s → CertifierRegisters H D.commit (lay s) (k s)
      (op i (runsBefore (run tap (schedOf z, srvOf z)).2 i)) s (tops s) ((run tap (schedOf z, srvOf z)).2 i)) :
    (∀ z ∈ Zs, (∀ i, WindowGoodChecked H D L K S lay k δ reg valid Dj (sw i) (ρ i) tops (x z i) (wit z i)
          ((run tap (schedOf z, srvOf z)).2 i)) →
        ∃ θ ∈ Θ, (run tap (schedOf z, srvOf z)).1 = rd θ (run tap (schedOf z, srvOf z)).2 ∧
          ∀ i, (run tap (schedOf z, srvOf z)).2 i ∈
            AccSeqsChecked (sw i).prog (Pw i) (sw i).design (lay (sw i)) (k (sw i))
              (AsReg (sw i).prog (lay (sw i)) (k (sw i)) (reg (sw i)) (ρ i)) (valid (sw i)) δ
              (op i (runsBefore (run tap (schedOf z, srvOf z)).2 i)) Dj) ∧
      {z | z ∈ Zs ∧ ∀ i, WindowGoodChecked H D L K S lay k δ reg valid Dj (sw i) (ρ i) tops (x z i) (wit z i)
          ((run tap (schedOf z, srvOf z)).2 i)}.encard ≤
        Θ.encard * ((windowsCount lay k δ (fun s => reg s ∪ chk s) Dj sw Pw β : ℕ) : ℕ∞) := by
  have hheld : ∀ z ∈ Zs, (∀ i, WindowGoodChecked H D L K S lay k δ reg valid Dj (sw i) (ρ i) tops (x z i) (wit z i)
        ((run tap (schedOf z, srvOf z)).2 i)) →
      ∃ θ ∈ Θ, (run tap (schedOf z, srvOf z)).1 = rd θ (run tap (schedOf z, srvOf z)).2 ∧
        ∀ i, (run tap (schedOf z, srvOf z)).2 i ∈
          AccSeqsChecked (sw i).prog (Pw i) (sw i).design (lay (sw i)) (k (sw i))
            (AsReg (sw i).prog (lay (sw i)) (k (sw i)) (reg (sw i)) (ρ i)) (valid (sw i)) δ
            (op i (runsBefore (run tap (schedOf z, srvOf z)).2 i)) Dj := by
    intro z hz hg
    obtain ⟨hfit, θ, hθ, hal⟩ := hch.1 z hz
    refine ⟨θ, hθ, ?_, fun i => (mem_accSeqsChecked_of_windowGoodChecked (hplaced.1 i) (htap z hz i) (hg i)).2⟩
    rw [htrust (schedOf z, srvOf z)]
    exact (hread θ _ hal hfit (arr (srvOf z))).symm
  refine ⟨hheld, ?_⟩
  set G := {z | z ∈ Zs ∧ ∀ i, WindowGoodChecked H D L K S lay k δ reg valid Dj (sw i) (ρ i) tops (x z i) (wit z i)
    ((run tap (schedOf z, srvOf z)).2 i)} with hG
  rcases G.eq_empty_or_nonempty with he | ⟨z₀, hz₀⟩
  · rw [he, Set.encard_empty]
    exact zero_le
  have hfits : ∀ i, (lay (sw i)).Fits (sw i).prog (k (sw i)) := fun i =>
    (mem_accSeqsChecked_of_windowGoodChecked (hplaced.1 i) (htap z₀ hz₀.1 i) (hz₀.2 i)).1
  let Os : Set (Fin n → List (Obs L K S)) := {o | ∀ i, o i ∈ AccSeqsChecked (sw i).prog (Pw i) (sw i).design
    (lay (sw i)) (k (sw i)) (AsReg (sw i).prog (lay (sw i)) (k (sw i)) (reg (sw i)) (ρ i)) (valid (sw i)) δ
    (op i (runsBefore o i)) Dj}
  let Vs : Set (Conn → List (ℕ × CellView Cell)) :=
    (fun p : Par × (Fin n → List (Obs L K S)) => rd p.1 p.2) '' (Θ ×ˢ Os)
  have hsub : G ⊆ dec '' Vs := fun z hz => by
    obtain ⟨θ, hθ, hview, hmem⟩ := hheld z hz.1 hz.2
    exact ⟨(run tap (schedOf z, srvOf z)).1, ⟨(θ, (run tap (schedOf z, srvOf z)).2), ⟨hθ, hmem⟩, hview.symm⟩,
      hch.2 z hz.1⟩
  have hOs : Os.encard ≤ ((windowsCount lay k δ (fun s => reg s ∪ chk s) Dj sw Pw β : ℕ) : ℕ∞) :=
    encard_periodChecked_le lay k δ reg chk valid Dj sw Pw β hplaced.2 hfits hchk ρ op
  calc G.encard ≤ (dec '' Vs).encard := Set.encard_le_encard hsub
    _ ≤ Vs.encard := Set.encard_image_le _ _
    _ ≤ (Θ ×ˢ Os).encard := Set.encard_image_le _ _
    _ = Θ.encard * Os.encard := Set.encard_prod
    _ ≤ _ := mul_le_mul' le_rfl hOs

end Compiled

end NetTiming

namespace NetTiming.TapProof

open Claims Claims.Sampling NetTiming NetTiming.SampledAuditProof

theorem Pr_or4_le {α : Type} [Fintype α] {p q₁ q₂ q₃ q₄ : α → Prop} {a b c d : ℝ}
    (h : ∀ s, p s → q₁ s ∨ q₂ s ∨ q₃ s ∨ q₄ s) (h₁ : Pr q₁ ≤ a) (h₂ : Pr q₂ ≤ b) (h₃ : Pr q₃ ≤ c)
    (h₄ : Pr q₄ ≤ d) : Pr p ≤ a + b + c + d :=
  (Pr_mono h).trans (((Pr_union_le _ _).trans (add_le_add h₁ ((Pr_union_le _ _).trans
    (add_le_add h₂ ((Pr_union_le _ _).trans (add_le_add h₃ h₄)))))).trans (le_of_eq (by ring)))

section Compiled

open Claims.SamplingGeneric (Arg Compiled)

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- **At the network certifier's rows, with a valid run**: on the audit's scope at `δ ≥ 0`, the sampling relation at a
trace that agrees with the network certifier wherever both open a gate gives the leak relation at the network
certifier's rows with a valid run, where every run at those rows (`certifierRun`) is `valid`. The run the relation
accepts at, on the inputs pinned to the trace's, is that run. -/
theorem rLeakCertifierChecked_of (H : Msg → Msg) (D : Compiled Coin A) {L K S : ℕ}
    (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) {δ : ℝ} (hδ : 0 ≤ δ)
    (reg : AuditStatement → Finset ℕ) (valid : AuditStatement → (ℕ → Bool) → Prop) (Dj : ℕ)
    (tops : AuditStatement → List (Opening Bool)) {x : Msg} (hx : ScopeAcc D lay k x) {w : Wit}
    (hR : R H D.commit D.parse D.partition δ x w)
    (hagree : ∀ s, D.parse x = some s → (∀ q ∈ tops s, D.commit.Opens H s.roots q = true) →
      ∀ g, (∃ q ∈ tops s, q.pos = g) → ∀ b, w.val H D.commit s g = some b → b = certifierVal (tops s) g)
    (hv : ∀ s Pt, D.parse x = some s → D.partition s = some Pt →
      ∀ r ∈ certifierRun H D.commit L K S lay k tops s Pt w, valid s r) :
    RLeakCertifierChecked H D.commit D.parse D.partition L K S lay k δ
      (fun s => AsReg s.prog (lay s) (k s) (reg s)) valid Dj tops x w := by
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
  have hag := hagree s hs htap.1
  have hext : ∀ g b, w.val H D.commit s g = some b →
      fillTrace (w.val H D.commit s) (certifierVal (tops s)) g = some b := fillTrace_ext
  have hcar := carries_fillTrace hfit htap hag
  have hacc := accepted_of hsc.1 hsc.2.1 hfit ι
    (fun v u hu hv => Prog.correct_mono (Prog.refIn_mono hext) (hι v u hu hv)) hcar hcov
  let a : ℕ → Bool := fun g =>
    if g ∈ freeInputs s.prog (lay s) (k s) then traceVal s.prog (w.val H D.commit s) g else false
  refine ⟨hl, hD, _, hF, a, ⟨fun g _ hg => ite_eq_left hg, fun g hg => ite_eq_right hg⟩,
    heldAdv s.prog P (faulty H D.commit s P w) (fillTrace (w.val H D.commit s) (certifierVal (tops s))),
    fun g b h => faultyGate_of_heldAdv h, ?_, ?_⟩
  · have hinp : valOver s.prog ((lay s).inputOf (k s) ((List.range (k s)).map fun i => op i (o.take i))
          (traceVal s.prog (w.val H D.commit s)))
        (heldAdv s.prog P (faulty H D.commit s P w) (fillTrace (w.val H D.commit s) (certifierVal (tops s)))) =
        valOver s.prog ((lay s).inputOf (k s) ((List.range (k s)).map fun i => op i (o.take i)) a)
          (heldAdv s.prog P (faulty H D.commit s P w) (fillTrace (w.val H D.commit s) (certifierVal (tops s)))) :=
      funext (valOver_congr_input s.prog _ fun i hi => inputOf_restrict s.prog (lay s) (k s) _ _ hi)
    rw [← hinp]
    exact hv s P hs hP _ ⟨hag, op, o, htap, rfl⟩
  · intro t ht
    obtain ⟨h, hr, hst⟩ := hacc t ht
    refine ⟨h, ?_, hst⟩
    have hv1 : ∀ g, ¬ s.prog.Unread g →
        valOver s.prog ((lay s).inputOf (k s) (ingressUpTo op o t)
          (traceVal s.prog (fillTrace (w.val H D.commit s) (certifierVal (tops s)))))
          (heldAdv s.prog P (faulty H D.commit s P w) (fillTrace (w.val H D.commit s) (certifierVal (tops s)))) g =
        valOver s.prog ((lay s).inputOf (k s) (ingressUpTo op o t) (traceVal s.prog (w.val H D.commit s)))
          (heldAdv s.prog P (faulty H D.commit s P w) (fillTrace (w.val H D.commit s) (certifierVal (tops s)))) g :=
      valOver_congr_held s.prog _ fun i hi hfi hri => inputOf_congr_at (lay s) (k s) _ (by
        have hnf : ¬ FaultyGate s.prog P (faulty H D.commit s P w) i := fun hfg => by
          rw [heldAdv_of hfg] at hfi
          cases hfi
        obtain ⟨b, hb⟩ := val_some_of_input hsc.2.1 hi hri hnf
        rw [traceVal_of_val hi hb, traceVal_of_val hi (hext i b hb)])
    have hv2 : ∀ g,
        valOver s.prog ((lay s).inputOf (k s) (ingressUpTo op o t) (traceVal s.prog (w.val H D.commit s)))
          (heldAdv s.prog P (faulty H D.commit s P w) (fillTrace (w.val H D.commit s) (certifierVal (tops s)))) g =
        valOver s.prog ((lay s).inputOf (k s) (ingressUpTo op o t) a)
          (heldAdv s.prog P (faulty H D.commit s P w) (fillTrace (w.val H D.commit s) (certifierVal (tops s)))) g :=
      valOver_congr_input s.prog _ fun i hi => inputOf_restrict s.prog (lay s) (k s) _ _ hi
    rw [hr]
    simp only [circuitProgram]
    funext ℓ
    exact rowOf_congr fun σ g hg => (hv1 g (not_unread_of_out (hfit.1 t ht ℓ σ g hg).1)).trans (hv2 g)

/-- **Each window's bound**: the first part of `Guarantees.PeriodLeakChecked`. An accepted run whose extracted witness
is not in the leak relation at the network certifier's rows with a valid run is outside the sampling relation, which
`Compiled.KS` bounds, or disagrees with a passing opening of the network certifier, a collision
(`certifierColl_of_disagree`, `Pr_certifierColl_le_Btap`), or has a run at those rows that one of the two checks
refuses, which `CheckedWitness` at `certifierRun` bounds (`rLeakCertifierChecked_of`). -/
theorem periodChecked_first (H : Msg → Msg) (D : Compiled Coin A) (hks : D.KS H) (hb : D.commit.Binds H) (L K S : ℕ)
    (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) (δ : ℝ) (hδ : 0 ≤ δ)
    (reg : AuditStatement → Finset ℕ) (valid₁ valid₂ : AuditStatement → (ℕ → Bool) → Prop) (ε₁ ε₂ : Msg → ℝ)
    (Dj : ℕ) (tops : AuditStatement → List (Opening Bool)) (x : Msg) (hx : ScopeAcc D lay k x)
    (h₁ : CheckedWitness H D (certifierRun H D.commit L K S lay k tops) valid₁ ε₁ x)
    (h₂ : CheckedWitness H D (certifierRun H D.commit L K S lay k tops) valid₂ ε₂ x)
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) :
    PrBad H D.V (RLeakCertifierChecked H D.commit D.parse D.partition L K S lay k δ
        (fun s => AsReg s.prog (lay s) (k s) (reg s)) (fun s r => valid₁ s r ∧ valid₂ s r) Dj tops) D.E x P ≤
      D.κ δ x + ε₁ x + ε₂ x + Real.sqrt (D.γ x * PrB H D.V P x) + (Bext H D.V D.E Ω P x).Pr H +
        (Btap H D.V D.E D.commit D.parse tops Ω P x).Pr H := by
  have hks' := hks δ hδ x hx.1 Ω P
  rw [one_mul] at hks'
  have hm₁ := h₁ Ω P
  have hm₂ := h₂ Ω P
  have hc := Pr_certifierColl_le_Btap H D tops P x
  unfold PrBad at hks' ⊢
  unfold PrCheckMiss at hm₁ hm₂
  refine (Pr_or4_le (fun s hs => ?_) hks' hc hm₁ hm₂).trans (le_of_eq (by ring))
  beta_reduce at hs ⊢
  obtain ⟨hacc, hng⟩ := hs
  rcases hw : (D.E.eval H P x s.1 s.2.1 s.2.2).1 with _ | w
  · exact Or.inl ⟨hacc, by simp [Rel.ext]⟩
  · rw [hw] at hng
    by_cases hA : R H D.commit D.parse D.partition δ x w
    · by_contra hcon
      simp only [not_or] at hcon
      obtain ⟨-, hnc, hn₁, hn₂⟩ := hcon
      refine hng (rLeakCertifierChecked_of H D lay k hδ reg _ Dj tops hx hA
        (fun st hst hpass g hq b hwb => ?_) (fun st Pt hst hPt r hr => ?_))
      · by_contra hne
        exact hnc (certifierColl_of_disagree H D hb tops P x s hst hpass hacc hw hq hwb hne)
      · have hdisj : Disjoint (faulty H D.commit st Pt w) (allSlots st.design Pt) := by
          obtain ⟨s', P', hs', hP', hM⟩ := hA
          rw [hst] at hs'
          obtain rfl := Option.some.inj hs'
          rw [hPt] at hP'
          obtain rfl := Option.some.inj hP'
          exact disjoint_allSlots hδ hM
        exact ⟨by_contra fun h => hn₁ ⟨hacc, w, st, Pt, rfl, hst, hPt, hdisj, r, hr, h⟩,
          by_contra fun h => hn₂ ⟨hacc, w, st, Pt, rfl, hst, hPt, hdisj, r, hr, h⟩⟩
    · exact Or.inl ⟨hacc, hA⟩

end Compiled

end NetTiming.TapProof

namespace NetTiming

open Claims Claims.Sampling

/-- **The leak through the network certifier over a monitoring period, with the checked inputs left out**:
`Guarantees.PeriodLeakChecked`, from each window's bound at the network certifier's rows with a valid run
(`TapProof.periodChecked_first`) and the held and counted parts at the two checks together
(`periodChecked_second`). -/
theorem periodLeakChecked : Guarantees.PeriodLeakChecked := by
  intro Coin _ _ _ A H D hks L K S lay k δ hδ reg chk valid₁ valid₂ ε₁ ε₂ Dj n sw Pw β hplaced hchk ρ op
    Conn Cell Srv Dev Zt Par arr f tap run htrust rd allows hread Θ Zs schedOf srvOf dec hch x wit tops hb
  exact ⟨fun m hm h₁ h₂ Ω _ _ P =>
      TapProof.periodChecked_first H D hks hb L K S lay k δ hδ reg valid₁ valid₂ ε₁ ε₂ Dj tops m hm h₁ h₂ Ω P,
    fun htap => periodChecked_second H D L K S lay k δ reg chk (fun s r => valid₁ s r ∧ valid₂ s r) Dj sw Pw β
      hplaced hchk ρ op arr f tap run htrust rd allows hread Θ Zs schedOf srvOf dec hch x wit tops htap⟩

end NetTiming
