module

public import Definitions.NetworkCertifier.Guarantees.PooledLeak
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Algebra.Order.Ring.Star
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
import Mathlib.Order.CompletePartialOrder
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
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.Sampling.Draws
import Proofs.Flock.Soundness.Audit.Periods
import Proofs.Flock.Soundness.Game
import Proofs.NetworkCertifier.PeriodSum
import Proofs.NetworkCertifier.Pooled
import Proofs.NetworkCertifier.SampledCharge
import Proofs.NetworkCertifier.SampledLeak
import Proofs.NetworkCertifier.Tap
import Std.Tactic.BVDecide.Normalize.Prop

/-!
# The leak through the network certifier over a monitoring period of sampled audits

The probability half: each execution phase's audit, given the earlier phases' coins, accepts with probability at
most `(1 - q)^|W_i| + e_i` plus the average over the phase's own coins of a charge that is not negative, and the
escapes multiply over the phases with the errors adding and the charges paid once, at the average over the period's
coins of their sum (`pr_adaptive_add_ex_le`, the area's one step over a period's phases); a union bound adds the leak
relations' errors `e' i` (`pr_pooled_ex_le`). At the charge zero that is `pr_pooled_le`, which `pooledLeak` reads.

The count half: a family of incorrect sets with fewer than `Fmax` proof blocks in total is the proof blocks listed by some
`Fin Fmax → Option (Σ i, Fin (n i))` (`exists_famOf_eq`), so there are at most `(N + 1)^Fmax` of them. At one family,
each window's observations number at most `2^#(freeInputs \ reg) · 2^(faultyBits) · Dj^k` at any operator
(`encard_leakAt_le`, from `encard_acceptedFaulty_le` at `AsReg`), and the window's operator is the one the earlier
windows' observations fix (`runsBefore`), so the periods number at most the product of the windows' bounds, by
induction over the windows (`encard_chain_le`); the faulty bits sum to at most `b · Fmax`. Under `trusted-host` the
cells the client receives read back from the network certifier's record under the schedule, one its record format
holds (`certifier_cells_read`), so they
number at most `#Ss` times those records, which bounds the secrets they decode to (`encard_pooledGood_le`);
`∏_c (cap_c + 1)` is room, as in the period's bound.

At operators that read no earlier window the statement is the one with each window's operator fixed before the
period (`pooledLeak_fixed`).
-/

@[expose] public section

namespace NetTiming

open Claims Claims.Sampling Finset

/-! ### The families of incorrect sets -/

/-- The proof blocks that `g` lists in window `i`. -/
def famOf {m : ℕ} {n : Fin m → ℕ} {F : ℕ} (g : Fin F → Option (Σ i : Fin m, Fin (n i))) (i : Fin m) : Finset ℕ :=
  (univ.filter fun v : Fin (n i) => ∃ t, g t = some ⟨i, v⟩).image Fin.val

theorem famOf_subset {m : ℕ} {n : Fin m → ℕ} {F : ℕ} (g : Fin F → Option (Σ i : Fin m, Fin (n i))) (i : Fin m) :
    famOf g i ⊆ range (n i) := by
  intro x hx
  simp only [famOf, mem_image] at hx
  obtain ⟨v, -, rfl⟩ := hx
  exact mem_range.2 v.2

theorem exists_famOf_eq {m : ℕ} (n : Fin m → ℕ) (Fmax : ℕ) (W : Fin m → Finset ℕ) (hW : ∀ i, W i ⊆ range (n i))
    (hF : ∑ i, (W i).card ≤ Fmax) : ∃ g : Fin Fmax → Option (Σ i : Fin m, Fin (n i)), famOf g = W := by
  classical
  let T : Finset (Σ i : Fin m, Fin (n i)) := univ.sigma fun i => univ.filter fun v : Fin (n i) => v.val ∈ W i
  have hfib : ∀ i, (univ.filter fun v : Fin (n i) => v.val ∈ W i).card = (W i).card := by
    intro i
    rw [← Finset.card_map Fin.valEmbedding]
    congr 1
    ext x
    simp only [Finset.mem_map, Finset.mem_filter, Finset.mem_univ, true_and, Fin.valEmbedding_apply]
    constructor
    · rintro ⟨v, hv, rfl⟩
      exact hv
    · intro hx
      exact ⟨⟨x, Finset.mem_range.1 (hW i hx)⟩, hx, rfl⟩
  have hTc : T.card ≤ Fmax := by
    rw [Finset.card_sigma]
    simp only [hfib]
    exact hF
  let g : Fin Fmax → Option (Σ i : Fin m, Fin (n i)) := fun t =>
    if h : t.val < T.card then some (T.equivFin.symm ⟨t.val, h⟩).val else none
  have hg : ∀ p, (∃ t, g t = some p) ↔ p ∈ T := by
    intro p
    constructor
    · rintro ⟨t, ht⟩
      by_cases h : t.val < T.card
      · simp only [g, h, dite_true, Option.some.injEq] at ht
        rw [← ht]
        exact (T.equivFin.symm _).2
      · simp only [g, h, dite_false, reduceCtorEq] at ht
    · intro hp
      refine ⟨⟨(T.equivFin ⟨p, hp⟩).val, lt_of_lt_of_le (T.equivFin ⟨p, hp⟩).2 hTc⟩, ?_⟩
      simp only [g, (T.equivFin ⟨p, hp⟩).2, dite_true, Fin.eta, Equiv.symm_apply_apply]
  refine ⟨g, funext fun i => ?_⟩
  ext x
  simp only [famOf, mem_image, mem_filter, mem_univ, true_and, hg]
  constructor
  · rintro ⟨v, hv, rfl⟩
    simpa [T] using hv
  · intro hx
    exact ⟨⟨x, mem_range.1 (hW i hx)⟩, by simpa [T] using hx, rfl⟩

theorem faultyBits_le_card_mul (C : Prog) (P : Partition) {n b : ℕ} {F : Finset ℕ} (hF : F ⊆ range n)
    (hb : ∀ v < n, faultyBits C P {v} ≤ b) : faultyBits C P F ≤ F.card * b := by
  have h : faultyBits C P F = ∑ v ∈ F, faultyBits C P {v} := by simp only [faultyBits, Finset.sum_singleton]
  rw [h]
  calc ∑ v ∈ F, faultyBits C P {v} ≤ ∑ _v ∈ F, b := sum_le_sum fun v hv => hb v (mem_range.1 (hF hv))
    _ = F.card * b := by rw [sum_const, smul_eq_mul]

/-! ### The count -/

theorem encard_leakAt_le {L K S : ℕ} (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) (reg : AuditStatement → Finset ℕ) (Dj : ℕ)
    (s : AuditStatement) (P : Partition) (ρ : ℕ → Bool) (op : ℕ → List (Obs L K S) → List Bool) (F : Finset ℕ) :
    {o | LeakAt lay k reg Dj s P ρ op F o}.encard ≤
      ((2 ^ (freeInputs s.prog (lay s) (k s) \ reg s).card * (2 ^ faultyBits s.prog P F * Dj ^ k s) : ℕ) : ℕ∞) := by
  by_cases hfits : (lay s).Fits s.prog (k s)
  · calc _ ≤ {o : List (Obs L K S) | o.length = k s ∧ (∀ x ∈ o, x.jitter < Dj) ∧
            AcceptedFaulty s.prog P (lay s) (k s) (AsReg s.prog (lay s) (k s) (reg s) ρ) F op o}.encard :=
          Set.encard_le_encard fun o h => h.2
      _ ≤ (AsReg s.prog (lay s) (k s) (reg s) ρ).encard * ((2 ^ faultyBits s.prog P F * Dj ^ k s : ℕ) : ℕ∞) :=
          encard_acceptedFaulty_le _ _ _ _ hfits _ _ _ _
      _ ≤ ((2 ^ (freeInputs s.prog (lay s) (k s) \ reg s).card : ℕ) : ℕ∞) *
            ((2 ^ faultyBits s.prog P F * Dj ^ k s : ℕ) : ℕ∞) :=
          mul_le_mul' (encard_asReg_le _ _ _ _ _) le_rfl
      _ = _ := (Nat.cast_mul _ _).symm
  · have : {o | LeakAt lay k reg Dj s P ρ op F o} = ∅ :=
      Set.eq_empty_iff_forall_notMem.2 fun o h => hfits h.1
    rw [this, Set.encard_empty]
    exact zero_le

/-- **The count at any relation on the whole period**: the secrets whose windows' observations together meet `Good` at
incorrect sets holding fewer than `Fmax` proof blocks in total number at most
`#Ss · ∏_c (cap_c + 1) · (N + 1)^Fmax · 2^(b · Fmax) · ∏_i 2^(cw i) · D i`, when the period's observations meeting
`Good` at incorrect sets `F i` of each window's proof blocks number at most `∏_i 2^(cw i) · 2^(bits i (F i)) · D i` and
`bits i F ≤ |F| · b`. -/
theorem encard_pooledJoint_le {L K S m : ℕ} (Good : (Fin m → Finset ℕ) → (Fin m → List (Obs L K S)) → Prop)
    (n : Fin m → ℕ) (b : ℕ) (cw D : Fin m → ℕ) (bits : Fin m → Finset ℕ → ℕ)
    (hbits : ∀ i F, F ⊆ range (n i) → bits i F ≤ F.card * b)
    (hcnt : ∀ F : Fin m → Finset ℕ, (∀ i, F i ⊆ range (n i)) →
      {o | Good F o}.encard ≤ ((∏ i, (2 ^ cw i * (2 ^ bits i (F i) * D i)) : ℕ) : ℕ∞))
    {Conn Cell Srv Dev Z : Type} [Fintype Conn] (arr : Srv → Conn → Arrivals Cell)
    (f : RecordFormat Conn Cell (Fin m → List (Obs L K S))) (tap : Dev)
    (run : Dev → CertifierSchedule Conn × Srv → (Conn → List (ℕ × CellView Cell)) × (Fin m → List (Obs L K S)))
    (htrust : Verity.Assumptions.TrustedHost tap run fun p => recordingCode f p.1 (arr p.2))
    (cap : Conn → ℕ) (Ss : Set (CertifierSchedule Conn)) (hfit : ∀ s ∈ Ss, f.fits s)
    (Zs : Set Z) (sched : Z → CertifierSchedule Conn) (srv : Z → Srv) (hs : ∀ z ∈ Zs, sched z ∈ Ss)
    (dec : (Conn → List (ℕ × CellView Cell)) → Z) (hdz : ∀ z ∈ Zs, dec (run tap (sched z, srv z)).1 = z)
    (W : Z → Fin m → Finset ℕ) (hWn : ∀ z i, W z i ⊆ range (n i)) (Fmax : ℕ) :
    {z | z ∈ Zs ∧ (∑ i, (W z i).card < Fmax ∧ Good (W z) ((run tap (sched z, srv z)).2))}.encard ≤
      Ss.encard * (((∏ c, (cap c + 1)) * ((∑ i, n i) + 1) ^ Fmax * 2 ^ (b * Fmax) * ∏ i, (2 ^ cw i * D i) : ℕ) : ℕ∞) := by
  classical
  set G := {z | z ∈ Zs ∧ (∑ i, (W z i).card < Fmax ∧ Good (W z) ((run tap (sched z, srv z)).2))} with hG
  set B : ℕ := 2 ^ (b * Fmax) * ∏ i, (2 ^ cw i * D i) with hB
  let s : Finset (Fin Fmax → Option (Σ i : Fin m, Fin (n i))) := univ.filter fun g => ∑ i, (famOf g i).card < Fmax
  let X : (Fin Fmax → Option (Σ i : Fin m, Fin (n i))) → Set (Fin m → List (Obs L K S)) := fun g =>
    {o | Good (famOf g) o}
  let Y : Set (Fin m → List (Obs L K S)) := ⋃ g ∈ s, X g
  let V : Set (Conn → List (ℕ × CellView Cell)) :=
    (fun p : CertifierSchedule Conn × (Fin m → List (Obs L K S)) => f.read p.1 p.2) '' (Ss ×ˢ Y)
  have hgood : ∀ z ∈ G, (run tap (sched z, srv z)).2 ∈ Y := by
    rintro z ⟨-, hlt, hleak⟩
    obtain ⟨g, hg⟩ := exists_famOf_eq n Fmax (W z) (hWn z) hlt.le
    refine Set.mem_biUnion (x := g) ?_ ?_
    · simp only [s, Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_ofPred_eq, hg]
      exact hlt
    · show Good (famOf g) _
      rw [hg]
      exact hleak
  have hsub : G ⊆ dec '' V := fun z hz =>
    ⟨(run tap (sched z, srv z)).1, ⟨(sched z, (run tap (sched z, srv z)).2), ⟨hs z hz.1, hgood z hz⟩,
      (certifier_cells_read f arr tap run htrust _ (hfit _ (hs z hz.1)) _).symm⟩, hdz z hz.1⟩
  have hX : ∀ g ∈ s, (X g).encard ≤ ((B : ℕ) : ℕ∞) := by
    intro g hg
    have hlt : ∑ i, (famOf g i).card < Fmax := (Finset.mem_filter.1 hg).2
    have hbit : ∑ i, bits i (famOf g i) ≤ b * Fmax := by
      calc ∑ i, bits i (famOf g i) ≤ ∑ i, (famOf g i).card * b :=
            sum_le_sum fun i _ => hbits i _ (famOf_subset g i)
        _ = (∑ i, (famOf g i).card) * b := (sum_mul _ _ _).symm
        _ ≤ Fmax * b := Nat.mul_le_mul_right _ hlt.le
        _ = b * Fmax := mul_comm _ _
    have hnat : ∏ i, (2 ^ cw i * (2 ^ bits i (famOf g i) * D i)) ≤ B := by
      calc ∏ i, (2 ^ cw i * (2 ^ bits i (famOf g i) * D i))
          = (∏ i, 2 ^ bits i (famOf g i)) * ∏ i, (2 ^ cw i * D i) := by
            rw [← Finset.prod_mul_distrib]
            exact Finset.prod_congr rfl fun i _ => by ring
        _ = 2 ^ (∑ i, bits i (famOf g i)) * ∏ i, (2 ^ cw i * D i) := by
            rw [Finset.prod_pow_eq_pow_sum]
        _ ≤ B := Nat.mul_le_mul_right _ (Nat.pow_le_pow_right (by norm_num) hbit)
    calc (X g).encard ≤ ((∏ i, (2 ^ cw i * (2 ^ bits i (famOf g i) * D i)) : ℕ) : ℕ∞) :=
          hcnt (famOf g) (famOf_subset g)
      _ ≤ ((B : ℕ) : ℕ∞) := by exact_mod_cast hnat
  have hY : Y.encard ≤ ((((∑ i, n i) + 1) ^ Fmax * B : ℕ) : ℕ∞) := by
    calc Y.encard ≤ ∑ g ∈ s, (X g).encard := encard_biUnion_le s X
      _ ≤ ∑ _g ∈ s, ((B : ℕ) : ℕ∞) := sum_le_sum hX
      _ = ((s.card * B : ℕ) : ℕ∞) := by rw [sum_const, nsmul_eq_mul]; push_cast; ring
      _ ≤ ((((∑ i, n i) + 1) ^ Fmax * B : ℕ) : ℕ∞) := by
          have : s.card ≤ ((∑ i, n i) + 1) ^ Fmax := by
            refine (Finset.card_le_univ s).trans_eq ?_
            rw [Fintype.card_fun, Fintype.card_option, Fintype.card_sigma, Fintype.card_fin]
            simp only [Fintype.card_fin]
          exact_mod_cast Nat.mul_le_mul_right B this
  have hcap1 : 0 < ∏ c, (cap c + 1) := Finset.prod_pos fun c _ => Nat.succ_pos _
  have hroom : ((∑ i, n i) + 1) ^ Fmax * B ≤ (∏ c, (cap c + 1)) * ((∑ i, n i) + 1) ^ Fmax * 2 ^ (b * Fmax) *
      ∏ i, (2 ^ cw i * D i) := by
    calc ((∑ i, n i) + 1) ^ Fmax * B ≤ (∏ c, (cap c + 1)) * (((∑ i, n i) + 1) ^ Fmax * B) :=
          Nat.le_mul_of_pos_left _ hcap1
      _ = _ := by rw [hB]; ring
  calc G.encard ≤ (dec '' V).encard := Set.encard_le_encard hsub
    _ ≤ V.encard := Set.encard_image_le _ _
    _ ≤ (Ss ×ˢ Y).encard := Set.encard_image_le _ _
    _ = Ss.encard * Y.encard := Set.encard_prod
    _ ≤ _ := by
        gcongr
        exact hY.trans (by exact_mod_cast hroom)

/-- **The count at any per-window relation**: the secrets whose windows' observations each meet `Good i` at incorrect
sets holding fewer than `Fmax` proof blocks in total number at most
`#Ss · ∏_c (cap_c + 1) · (N + 1)^Fmax · 2^(b · Fmax) · ∏_i 2^(cw i) · D i`, when window `i`'s observations at an
incorrect set `F` of its proof blocks number at most `2^(cw i) · 2^(bits i F) · D i` and `bits i F ≤ |F| · b`. -/
theorem encard_pooledFam_le {L K S m : ℕ} (Good : Fin m → Finset ℕ → List (Obs L K S) → Prop) (n : Fin m → ℕ) (b : ℕ)
    (cw D : Fin m → ℕ) (bits : Fin m → Finset ℕ → ℕ) (hbits : ∀ i F, F ⊆ range (n i) → bits i F ≤ F.card * b)
    (hcnt : ∀ i F, F ⊆ range (n i) → {o | Good i F o}.encard ≤ ((2 ^ cw i * (2 ^ bits i F * D i) : ℕ) : ℕ∞))
    {Conn Cell Srv Dev Z : Type} [Fintype Conn] (arr : Srv → Conn → Arrivals Cell)
    (f : RecordFormat Conn Cell (Fin m → List (Obs L K S))) (tap : Dev)
    (run : Dev → CertifierSchedule Conn × Srv → (Conn → List (ℕ × CellView Cell)) × (Fin m → List (Obs L K S)))
    (htrust : Verity.Assumptions.TrustedHost tap run fun p => recordingCode f p.1 (arr p.2))
    (cap : Conn → ℕ) (Ss : Set (CertifierSchedule Conn)) (hfit : ∀ s ∈ Ss, f.fits s)
    (Zs : Set Z) (sched : Z → CertifierSchedule Conn) (srv : Z → Srv) (hs : ∀ z ∈ Zs, sched z ∈ Ss)
    (dec : (Conn → List (ℕ × CellView Cell)) → Z) (hdz : ∀ z ∈ Zs, dec (run tap (sched z, srv z)).1 = z)
    (W : Z → Fin m → Finset ℕ) (hWn : ∀ z i, W z i ⊆ range (n i)) (Fmax : ℕ) :
    {z | z ∈ Zs ∧ (∑ i, (W z i).card < Fmax ∧ ∀ i, Good i (W z i) ((run tap (sched z, srv z)).2 i))}.encard ≤
      Ss.encard * (((∏ c, (cap c + 1)) * ((∑ i, n i) + 1) ^ Fmax * 2 ^ (b * Fmax) * ∏ i, (2 ^ cw i * D i) : ℕ) : ℕ∞) :=
  encard_pooledJoint_le (fun F o => ∀ i, Good i (F i) (o i)) n b cw D bits hbits
    (fun F hF => by
      calc {o : Fin m → List (Obs L K S) | ∀ i, Good i (F i) (o i)}.encard
          = (Set.pi Set.univ fun i => {o | Good i (F i) o}).encard := by
            congr 1
            ext o
            simp
        _ = ∏ i, {o | Good i (F i) o}.encard := Set.encard_pi_eq_prod_encard
        _ ≤ ∏ i, ((2 ^ cw i * (2 ^ bits i (F i) * D i) : ℕ) : ℕ∞) :=
            Finset.prod_le_prod fun i _ => hcnt i _ (hF i)
        _ = ((∏ i, (2 ^ cw i * (2 ^ bits i (F i) * D i)) : ℕ) : ℕ∞) := (Nat.cast_prod _ _).symm)
    arr f tap run htrust cap Ss hfit Zs sched srv hs dec hdz W hWn Fmax

theorem encard_pooledGood_le {L K S : ℕ} (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) (reg : AuditStatement → Finset ℕ) (Dj : ℕ)
    {m : ℕ} (sw : Fin m → AuditStatement) (Pw : Fin m → Partition) (ρ : Fin m → ℕ → Bool)
    (op : Fin m → List (List (Obs L K S)) → ℕ → List (Obs L K S) → List Bool) (n : Fin m → ℕ) (b : ℕ)
    (hb : ∀ i, ∀ v < n i, faultyBits (sw i).prog (Pw i) {v} ≤ b)
    {Conn Cell Srv Dev Z : Type} [Fintype Conn] (arr : Srv → Conn → Arrivals Cell)
    (f : RecordFormat Conn Cell (Fin m → List (Obs L K S))) (tap : Dev)
    (run : Dev → CertifierSchedule Conn × Srv → (Conn → List (ℕ × CellView Cell)) × (Fin m → List (Obs L K S)))
    (htrust : Verity.Assumptions.TrustedHost tap run fun p => recordingCode f p.1 (arr p.2))
    (cap : Conn → ℕ) (Ss : Set (CertifierSchedule Conn)) (hfit : ∀ s ∈ Ss, f.fits s)
    (Zs : Set Z) (sched : Z → CertifierSchedule Conn) (srv : Z → Srv) (hs : ∀ z ∈ Zs, sched z ∈ Ss)
    (dec : (Conn → List (ℕ × CellView Cell)) → Z) (hdz : ∀ z ∈ Zs, dec (run tap (sched z, srv z)).1 = z)
    (W : Z → Fin m → Finset ℕ) (hWn : ∀ z i, W z i ⊆ range (n i)) (Fmax : ℕ) :
    {z | z ∈ Zs ∧ PooledGood lay k reg Dj sw Pw ρ op Fmax (W z) (run tap (sched z, srv z)).2}.encard ≤
      Ss.encard * (((∏ c, (cap c + 1)) * ((∑ i, n i) + 1) ^ Fmax * 2 ^ (b * Fmax) *
        ∏ i, (2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \ reg (sw i)).card * Dj ^ k (sw i)) : ℕ) : ℕ∞) :=
  encard_pooledJoint_le
    (fun F o => ∀ i, LeakAt lay k reg Dj (sw i) (Pw i) (ρ i) (op i (runsBefore o i)) (F i) (o i)) n b
    (fun i => (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \ reg (sw i)).card) (fun i => Dj ^ k (sw i))
    (fun i F => faultyBits (sw i).prog (Pw i) F) (fun i _ hF => faultyBits_le_card_mul _ _ hF (hb i))
    (fun F _ => by
      have h := encard_chain_le (fun i l => {x | LeakAt lay k reg Dj (sw i) (Pw i) (ρ i) (op i l) (F i) x})
        (fun i => 2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \ reg (sw i)).card *
          (2 ^ faultyBits (sw i).prog (Pw i) (F i) * Dj ^ k (sw i)))
        fun i l => encard_leakAt_le lay k reg Dj (sw i) (Pw i) (ρ i) (op i l) (F i)
      exact h)
    arr f tap run htrust cap Ss hfit Zs sched srv hs dec hdz W hWn Fmax

/-! ### The monitoring period -/

/-- **The monitoring period's bound at any good event of an execution phase, with a charge paid once**: every audit
accepts while the incorrect sets hold `Fmax` proof blocks or more, or some phase's good event fails, with probability
at most `(1 - q)^Fmax + Σ e + Σ e'` plus the average, over the period's coins, of the charges' sum, where phase `i`'s
conditional escape holds, beside `e i`, the average over the phase's own coins of its charge `cl i ω`:
`pr_adaptive_add_ex_le` at the conditional escape, and a union bound over the good events' errors. -/
theorem pr_pooled_ex_le {Ω : Type} [Fintype Ω] [Nonempty Ω] {m : ℕ} (acc : Fin m → (Fin m → Ω) → Ω → Prop)
    (hacc : ∀ i ω ω', (∀ j, j < i → ω j = ω' j) → acc i ω = acc i ω') (W : (Fin m → Ω) → Fin m → Finset ℕ)
    (hW : ∀ i ω ω', (∀ j, j < i → ω j = ω' j) → W ω i = W ω' i) (e e' : Fin m → ℝ) (he : ∀ i, 0 ≤ e i)
    (cl : Fin m → (Fin m → Ω) → Ω → ℝ) (hcl : ∀ i ω ω', (∀ j, j < i → ω j = ω' j) → cl i ω = cl i ω')
    (hcl0 : ∀ i ω x, 0 ≤ cl i ω x) (q : ℝ) (hq0 : 0 ≤ q) (hq1 : q ≤ 1)
    (hesc : ∀ i ω, Pr (acc i ω) ≤ (1 - q) ^ (W ω i).card + e i + Ex (cl i ω))
    (Good : Fin m → (Fin m → Ω) → Prop) (hleak : ∀ i, Pr (fun ω => acc i ω (ω i) ∧ ¬Good i ω) ≤ e' i) (Fmax : ℕ) :
    Pr (fun ω => (∀ i, acc i ω (ω i)) ∧ ¬(∑ i, (W ω i).card < Fmax ∧ ∀ i, Good i ω)) ≤
      (1 - q) ^ Fmax + ∑ i, e i + (Ex fun ω => ∑ i, cl i ω (ω i)) + ∑ i, e' i := by
  classical
  have hmiss := pr_adaptive_add_ex_le (Ω := Ω) (r := 1 - q) (by linarith) (by linarith) m acc
    (fun i ω => (W ω i).card) e cl hacc (fun i ω ω' h => by simp only [hW i ω ω' h]) hcl he hcl0 hesc Fmax
  calc Pr (fun ω => (∀ i, acc i ω (ω i)) ∧ ¬(∑ i, (W ω i).card < Fmax ∧ ∀ i, Good i ω))
      ≤ Pr (fun ω => ((∀ i, acc i ω (ω i)) ∧ Fmax ≤ ∑ i, (W ω i).card) ∨
          ∃ i ∈ (univ : Finset (Fin m)), acc i ω (ω i) ∧ ¬Good i ω) := by
        refine Pr_mono fun ω h => ?_
        obtain ⟨hall, hng⟩ := h
        by_cases hlt : ∑ i, (W ω i).card < Fmax
        · have : ¬∀ i, Good i ω := fun hl => hng ⟨hlt, hl⟩
          push Not at this
          obtain ⟨i, hi⟩ := this
          exact Or.inr ⟨i, mem_univ i, hall i, hi⟩
        · exact Or.inl ⟨hall, not_lt.1 hlt⟩
    _ ≤ _ := Pr_or_le _ _
    _ ≤ ((1 - q) ^ Fmax + ∑ i, e i + Ex fun ω => ∑ i, cl i ω (ω i)) + ∑ i, e' i :=
        add_le_add hmiss ((Pr_exists_mem_le_sum univ _).trans (sum_le_sum fun i _ => hleak i))

/-- **The monitoring period's bound at any good event of an execution phase**: every audit accepts while the incorrect
sets hold `Fmax` proof blocks or more, or some phase's good event fails, with probability at most
`(1 - q)^Fmax + Σ e + Σ e'`: `pr_pooled_ex_le` at the charge zero. -/
theorem pr_pooled_le {Ω : Type} [Fintype Ω] [Nonempty Ω] {m : ℕ} (acc : Fin m → (Fin m → Ω) → Ω → Prop)
    (hacc : ∀ i ω ω', (∀ j, j < i → ω j = ω' j) → acc i ω = acc i ω') (W : (Fin m → Ω) → Fin m → Finset ℕ)
    (hW : ∀ i ω ω', (∀ j, j < i → ω j = ω' j) → W ω i = W ω' i) (e e' : Fin m → ℝ) (he : ∀ i, 0 ≤ e i) (q : ℝ)
    (hq0 : 0 ≤ q) (hq1 : q ≤ 1) (hesc : ∀ i ω, Pr (acc i ω) ≤ (1 - q) ^ (W ω i).card + e i)
    (Good : Fin m → (Fin m → Ω) → Prop) (hleak : ∀ i, Pr (fun ω => acc i ω (ω i) ∧ ¬Good i ω) ≤ e' i) (Fmax : ℕ) :
    Pr (fun ω => (∀ i, acc i ω (ω i)) ∧ ¬(∑ i, (W ω i).card < Fmax ∧ ∀ i, Good i ω)) ≤
      (1 - q) ^ Fmax + ∑ i, e i + ∑ i, e' i := by
  have h := pr_pooled_ex_le acc hacc W hW e e' he (fun _ _ _ => 0) (fun _ _ _ _ => rfl) (fun _ _ _ => le_rfl) q hq0
    hq1
    (fun i ω => by
      simp only [Ex_const, add_zero]
      exact hesc i ω)
    Good hleak Fmax
  simp only [Finset.sum_const_zero, Ex_const, add_zero] at h
  exact h

/-- **The leak through the network certifier over a monitoring period**: `Guarantees.PooledLeakAccepted`, from
`pr_pooled_le` at the leak relations, `(1 - q)^Fmax ≤ δ` at `Fmax ≥ ln(1/δ)/q`, and `encard_pooledGood_le` at each sample
of the coins. -/
theorem pooledLeak : Guarantees.PooledLeakAccepted := by
  intro L K S lay k reg Dj m sw Pw ρ op n b hb Ω _ _ Conn Cell Srv Dev Z _ arr f tap run htrust cap Ss hcap hfit Zs
    schedOf srvOf hs dec hdz acc hacc W hW hWn e e' he q hq0 hq1 hesc hleak δ hδ Fmax hF
  refine ⟨fun z hz => ?_, fun ω => encard_pooledGood_le lay k reg Dj sw Pw ρ op n b hb arr f tap run htrust cap Ss
    hfit Zs (fun z => schedOf z ω) (fun z => srvOf z ω) (fun z hz => hs z hz ω) (dec ω) (fun z hz => hdz z hz ω)
    (fun z => W z ω) (fun z i => hWn z ω i) Fmax⟩
  have hpow : (1 - q) ^ Fmax ≤ δ := by
    have h1 : 1 - q ≤ Real.exp (-q) := by linarith [Real.add_one_le_exp (-q)]
    calc (1 - q) ^ Fmax ≤ Real.exp (-q) ^ Fmax := pow_le_pow_left₀ (by linarith) h1 Fmax
      _ = Real.exp (Fmax * (-q)) := (Real.exp_nat_mul _ _).symm
      _ ≤ Real.exp (-Real.log (1 / δ)) := Real.exp_le_exp.2 (by nlinarith)
      _ = δ := by rw [one_div, Real.log_inv, neg_neg, Real.exp_log hδ]
  have h := pr_pooled_le (acc z) (hacc z) (W z) (hW z) e e' he q hq0 hq1 (hesc z hz)
    (fun i ω => LeakAt lay k reg Dj (sw i) (Pw i) (ρ i) (op i (runsBefore (run tap (schedOf z ω, srvOf z ω)).2 i))
      (W z ω i) ((run tap (schedOf z ω, srvOf z ω)).2 i))
    (hleak z hz) Fmax
  exact h.trans (by linarith)

set_option linter.unusedVariables false in
/-- **`PooledLeakAccepted` at operators fixed before the period**: its statement with window `i`'s operator `op i` one
that reads no earlier window, `PooledGood` written out, which is `PooledLeakAccepted` at the operator
`fun i _ => op i`, one that ignores the windows before it. -/
theorem pooledLeak_fixed (h : Guarantees.PooledLeakAccepted) :
    ∀ {L K S : ℕ} (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ)
      (reg : AuditStatement → Finset ℕ) (Dj : ℕ) {m : ℕ} (sw : Fin m → AuditStatement) (Pw : Fin m → Partition)
      (ρ : Fin m → ℕ → Bool) (op : Fin m → ℕ → List (Obs L K S) → List Bool)
      (n : Fin m → ℕ) (b : ℕ) (hb : ∀ i, ∀ v < n i, faultyBits (sw i).prog (Pw i) {v} ≤ b)
      {Ω : Type} [Fintype Ω] [Nonempty Ω] {Conn Cell Srv Dev Z : Type} [Fintype Conn]
      (arr : Srv → Conn → Arrivals Cell) (f : RecordFormat Conn Cell (Fin m → List (Obs L K S))) (tap : Dev)
      (run : Dev → CertifierSchedule Conn × Srv → (Conn → List (ℕ × CellView Cell)) × (Fin m → List (Obs L K S)))
      (htrust : Verity.Assumptions.TrustedHost tap run fun p => recordingCode f p.1 (arr p.2))
      (cap : Conn → ℕ) (Ss : Set (CertifierSchedule Conn)) (hcap : ∀ s ∈ Ss, ∀ c, s.cells c ≤ cap c)
      (hfit : ∀ s ∈ Ss, f.fits s)
      (Zs : Set Z) (schedOf : Z → (Fin m → Ω) → CertifierSchedule Conn) (srvOf : Z → (Fin m → Ω) → Srv)
      (hs : ∀ z ∈ Zs, ∀ ω, schedOf z ω ∈ Ss) (dec : (Fin m → Ω) → (Conn → List (ℕ × CellView Cell)) → Z)
      (hdz : ∀ z ∈ Zs, ∀ ω, dec ω (run tap (schedOf z ω, srvOf z ω)).1 = z)
      (acc : Z → Fin m → (Fin m → Ω) → Ω → Prop)
      (hacc : ∀ z i ω ω', (∀ j, j < i → ω j = ω' j) → acc z i ω = acc z i ω')
      (W : Z → (Fin m → Ω) → Fin m → Finset ℕ)
      (hW : ∀ z i ω ω', (∀ j, j < i → ω j = ω' j) → W z ω i = W z ω' i)
      (hWn : ∀ z ω i, W z ω i ⊆ range (n i)) (e e' : Fin m → ℝ) (he : ∀ i, 0 ≤ e i)
      (q : ℝ) (hq0 : 0 ≤ q) (hq1 : q ≤ 1)
      (hesc : ∀ z ∈ Zs, ∀ i ω, Pr (acc z i ω) ≤ (1 - q) ^ (W z ω i).card + e i)
      (hleak : ∀ z ∈ Zs, ∀ i, Pr (fun ω => acc z i ω (ω i) ∧
        ¬LeakAt lay k reg Dj (sw i) (Pw i) (ρ i) (op i) (W z ω i) ((run tap (schedOf z ω, srvOf z ω)).2 i)) ≤ e' i)
      (δ : ℝ) (hδ : 0 < δ) (Fmax : ℕ) (hF : Real.log (1 / δ) ≤ q * Fmax),
      (∀ z ∈ Zs, Pr (fun ω => (∀ i, acc z i ω (ω i)) ∧
          ¬(∑ i, (W z ω i).card < Fmax ∧ ∀ i, LeakAt lay k reg Dj (sw i) (Pw i) (ρ i) (op i) (W z ω i)
            ((run tap (schedOf z ω, srvOf z ω)).2 i))) ≤
          δ + ∑ i, e i + ∑ i, e' i) ∧
        ∀ ω, {z | z ∈ Zs ∧ (∑ i, (W z ω i).card < Fmax ∧ ∀ i, LeakAt lay k reg Dj (sw i) (Pw i) (ρ i) (op i)
            (W z ω i) ((run tap (schedOf z ω, srvOf z ω)).2 i))}.encard ≤
          Ss.encard * (((∏ c, (cap c + 1)) * ((∑ i, n i) + 1) ^ Fmax * 2 ^ (b * Fmax) *
            ∏ i, (2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \ reg (sw i)).card * Dj ^ k (sw i)) : ℕ) :
              ℕ∞) := by
  intro L K S lay k reg Dj m sw Pw ρ op n b hb Ω _ _ Conn Cell Srv Dev Z _ arr f tap run htrust cap Ss hcap hfit Zs
    schedOf srvOf hs dec hdz acc hacc W hW hWn e e' he q hq0 hq1 hesc hleak δ hδ Fmax hF
  exact h lay k reg Dj sw Pw ρ (fun i _ => op i) n b hb arr f tap run htrust cap Ss hcap hfit Zs
    schedOf srvOf hs dec hdz acc hacc W hW hWn e e' he q hq0 hq1 hesc hleak δ hδ Fmax hF

end NetTiming
