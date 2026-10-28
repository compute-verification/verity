import Proofs.Pous.Statements
import Proofs.Pous.Thm1.Theorem1iTimed
import Proofs.Pous.Thm1.Theorem1iiSeq
import Proofs.Pous.Thm1.Theorem1iiiTimed
import Proofs.Pous.Graded.Theorem
import Proofs.Pous.Continuous.MissCount
import Proofs.Pous.Continuous.Verifier

/-!
# The continuous audit from `TimedINC`: `Pous.Guarantees.PerAnswer` and `Pous.Guarantees.ContinuousAudit`

The proofs of both guarantees, pinned as `Pous.SecurityProofs.PerAnswer` and `Pous.SecurityProofs.ContinuousAudit`,
over the memory challenger of `Security/Definitions/Pous/Model/Continuous.lean`. Neither uses a hypothesis about
the responder's timing. The memory challenger's rules enter through `Continuous/Verifier.lean`: each round's record
depends only on the indices up to its own (`contRecords_ofFn`), and a hit returned the right block within `(D, Q)`
(`hit_on_time`).

1. **One history** (`pr_bad_le`). Fix the indices `h` sent before a round. The responder's memory at that round's send
   time is a function of `(W, pp, C, ω)` (`heldAt`), and the round's program depends only on `W`, `pp`, `h`, that
   memory and the index. Stopped at `(D, Q)` (`PousGraded.cut`) it is a `(D, Q)`-bounded program that returns the block
   of every hit (`hit_pred`). For each length `s ≤ β` of the memory, Theorem 1(ii)'s patch compressor (`pr_fits_le`)
   bounds by `π` the event that the memory and the non-hits fit in `β` bits. So `Pr[bad h] ≤ (β + 1)·π`.
2. **Step (a)** (`per_answer`). Off `bad h`, at most `p₀(β, |m_h|) = 1 − ε + ε_proof` of the indices are hits
   (`good_le_p₀`).
3. **Stopping** (`continuous_audit`). An accepting run has at most `f` non-hits. Take `q_h = max(ε_h − ε_proof, 0)`
   off `bad h` and `0` on it. Then `q_h` lower-bounds the conditional miss probability at every setup and history, and
   the counting step (`pr_misses_gt_le`) bounds the misses against `Σ q` by `βe` at each setup. Off the event that some
   visited history is bad, `Σ q` is `Σ max(ε_j − ε_proof, 0)`. That event has probability at most `n·(β + 1)·π`,
   because the indices are independent of the setup and so each round's history is bad with probability at most
   `(β + 1)·π`.
-/

namespace PousContinuous

open Pous PousThm1
open scoped ENNReal

variable {M : IdealModel}

/-- `l` as an `s`-bit string, padded with `false` or truncated. -/
def toBits (s : ℕ) (l : List Bool) : Bits s := fun t => l.getD t false

theorem ofFn_toBits (s : ℕ) (l : List Bool) (h : l.length = s) : List.ofFn (toBits s l) = l := by
  subst h
  refine List.ext_getElem (by simp) fun t h1 h2 => ?_
  simp [toBits, List.getElem?_eq_getElem h2]

/-- **Averaging over the first coordinate.** -/
theorem pr_prod_le_of_forall {α β : Type} [Fintype α] [Fintype β] [Nonempty α] [Nonempty β]
    (P : α → β → Prop) (c : ℝ≥0∞) (h : ∀ a, pr (P a) ≤ c) : pr (fun x : α × β => P x.1 x.2) ≤ c := by
  have := pr_prod_le P (fun _ => False) c fun a _ => h a
  rwa [pr_false', zero_add] at this

theorem length_filter_ofFn {α : Type} {n : ℕ} (R : Fin n → α) (p : α → Prop) [DecidablePred p] :
    ((List.ofFn R).filter fun r => decide (p r)).length = (Finset.univ.filter fun j => p (R j)).card := by
  have h := congrArg Multiset.card (Multiset.filter_map p R (Finset.univ : Finset (Fin n)).val)
  simp only [Fin.univ_val_map, Multiset.filter_coe, Multiset.coe_card, Multiset.card_map] at h
  rw [h, Finset.card_def, Finset.filter_val]
  rfl

section

variable (sch : Scheme M) (D Q D_late Q_late : ℕ) (W : Bits sch.n) (mach : ContMachine sch) (arr : ℕ → ℕ)

/-- The responder's memory at the send time of the round after `h`, at setup `a`. -/
def heldAt (a : M.Ω × sch.Coins) (h : List (Fin sch.B)) : List Bool :=
  contMem mach W (sch.enc W a.1 a.2).1 (sch.enc W a.1 a.2).2 a.1 arr
    (contAfter mach D_late W (sch.enc W a.1 a.2).1 (sch.enc W a.1 a.2).2 a.1 arr h)

/-- At setup `a`, the round after `h` is a hit on index `i`. -/
def hitAt (a : M.Ω × sch.Coins) (h : List (Fin sch.B)) (i : Fin sch.B) : Prop :=
  (contRecord mach D Q D_late Q_late W (sch.enc W a.1 a.2).1 (sch.enc W a.1 a.2).2 a.1 arr
    (contAfter mach D_late W (sch.enc W a.1 a.2).1 (sch.enc W a.1 a.2).2 a.1 arr h) i).verdict = .hit

/-- The round's program, stopped at `(D, Q)`, from a memory given as `τ`. -/
def stopped (W : Bits sch.n) (pp : Bits sch.t) (h : List (Fin sch.B)) {s : ℕ} (τ : Bits s) (i : Fin sch.B) :
    OracleAlgorithm M (Bits sch.ℓ) :=
  progMap (fun r : Option (Bits sch.ℓ) × List Bool => r.1.getD fun _ => false)
    (PousGraded.cut ((none, []) : Option (Bits sch.ℓ) × List Bool) D Q
      (mach.round W pp h D_late (List.ofFn τ) (some i)))

theorem stopped_bounded (W : Bits sch.n) (pp : Bits sch.t) (h : List (Fin sch.B)) {s : ℕ} (τ : Bits s)
    (i : Fin sch.B) : (stopped sch D Q D_late mach W pp h τ i).Bounded D Q :=
  bounded_progMap _ (PousGraded.cut_bounded _ _ _ _)

/-- The stopped program returns the block of every hit. -/
theorem hit_pred (s : ℕ) (a : M.Ω × sch.Coins) (h : List (Fin sch.B)) (i : Fin sch.B)
    (hs : (heldAt sch D_late W mach arr a h).length = s) (hh : hitAt sch D Q D_late Q_late W mach arr a h i) :
    (stopped sch D Q D_late mach W (sch.enc W a.1 a.2).1 h (toBits s (heldAt sch D_late W mach arr a h)) i).run
      (M.answer a.1) = (sch.enc W a.1 a.2).2 i := by
  unfold hitAt contRecord at hh
  obtain ⟨h1, h2, h3⟩ := hit_on_time _ _ _ _ _ _ hh
  unfold contProg at h1 h2 h3
  rw [contAfter_sent] at h1 h2 h3
  unfold stopped
  rw [ofFn_toBits s _ hs, run_progMap]
  unfold heldAt
  rw [PousGraded.cut_run _ _ _ D Q h2 h3, h1]
  rfl

open Classical in
/-- After `h`, the memory and the non-hits fit in `β` bits. -/
def bad (β : ℕ) (a : M.Ω × sch.Coins) (h : List (Fin sch.B)) : Prop :=
  (heldAt sch D_late W mach arr a h).length + sch.B +
    (Finset.univ.filter fun i => ¬ hitAt sch D Q D_late Q_late W mach arr a h i).card * sch.ℓ ≤ β

/-- **One history.** `Pr[bad h] ≤ (β + 1)·π`, by one patch compressor per length of the memory. -/
theorem pr_bad_le (β : ℕ) (π : ℝ≥0∞) (hinc : TimedINC sch D Q β π) (h : List (Fin sch.B)) :
    pr (fun a => bad sch D Q D_late Q_late W mach arr β a h) ≤ (β + 1) * π := by
  classical
  calc pr (fun a => bad sch D Q D_late Q_late W mach arr β a h)
      ≤ pr (fun a => ∃ s ∈ Finset.range (β + 1),
          (heldAt sch D_late W mach arr a h).length = s ∧ bad sch D Q D_late Q_late W mach arr β a h) :=
        pr_mono fun a ha => ⟨_, Finset.mem_range.2 (by unfold bad at ha; omega), rfl, ha⟩
    _ ≤ ∑ s ∈ Finset.range (β + 1),
          pr (fun a => (heldAt sch D_late W mach arr a h).length = s ∧ bad sch D Q D_late Q_late W mach arr β a h) :=
        pr_exists_le _ _
    _ ≤ ∑ _s ∈ Finset.range (β + 1), π := by
        refine Finset.sum_le_sum fun s _ => le_trans (pr_mono fun a ha => ?_)
          (pr_fits_le sch D Q β s π hinc W
            (fun W pp C ω => toBits s (contMem mach W pp C ω arr (contAfter mach D_late W pp C ω arr h)))
            (fun W pp τ i => stopped sch D Q D_late mach W pp h τ i)
            fun W pp τ i => stopped_bounded sch D Q D_late mach W pp h τ i)
        obtain ⟨hs, hb⟩ := ha
        have hsub : (Finset.univ.filter fun i => ¬ (stopped sch D Q D_late mach W (sch.enc W a.1 a.2).1 h
              (toBits s (heldAt sch D_late W mach arr a h)) i).run (M.answer a.1) = (sch.enc W a.1 a.2).2 i) ⊆
            Finset.univ.filter fun i => ¬ hitAt sch D Q D_late Q_late W mach arr a h i := by
          intro i
          simp only [Finset.mem_filter, Finset.mem_univ, true_and]
          exact fun hn hh => hn (hit_pred sch D Q D_late Q_late W mach arr s a h i hs hh)
        unfold bad at hb
        rw [hs] at hb
        exact le_trans (Nat.add_le_add_left (Nat.mul_le_mul_right _ (Finset.card_le_card hsub)) _) hb
    _ = (β + 1) * π := by rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul, Nat.cast_add, Nat.cast_one]

/-- `p₀` at the memory's size is `1 − ε + ε_proof`. -/
theorem p₀_eq (β : ℕ) (a : M.Ω × sch.Coins) (h : List (Fin sch.B)) :
    p₀ β (heldAt sch D_late W mach arr a h).length sch.B sch.ℓ =
      1 - contFreed sch (heldAt sch D_late W mach arr a h).length + epsProof sch β := by
  unfold p₀ contFreed epsProof
  ring

/-- **Step (a) at one setup.** Off `bad h`, the round after `h` is a hit with probability at most
`1 − ε + ε_proof`. -/
theorem pr_hit_le (β : ℕ) (a : M.Ω × sch.Coins) (h : List (Fin sch.B))
    (hb : ¬ bad sch D Q D_late Q_late W mach arr β a h) :
    pr (hitAt sch D Q D_late Q_late W mach arr a h) ≤
      ENNReal.ofReal (1 - contFreed sch (heldAt sch D_late W mach arr a h).length + epsProof sch β) := by
  rw [← p₀_eq]
  exact pr_good_le_p₀ sch.B sch.ℓ β _ sch.B_pos _ hb

/-- Off `bad h`, the round after `h` misses with probability at least `ε − ε_proof`. -/
theorem miss_ge (β : ℕ) (a : M.Ω × sch.Coins) (h : List (Fin sch.B))
    (hb : ¬ bad sch D Q D_late Q_late W mach arr β a h) :
    contFreed sch (heldAt sch D_late W mach arr a h).length - epsProof sch β ≤
      (pr fun i => ¬ hitAt sch D Q D_late Q_late W mach arr a h i).toReal := by
  classical
  have hgb := Finset.card_filter_add_card_filter_not (s := Finset.univ) (hitAt sch D Q D_late Q_late W mach arr a h)
  rw [Finset.card_univ, Fintype.card_fin] at hgb
  have hb' : β < (heldAt sch D_late W mach arr a h).length + sch.B +
      (Finset.univ.filter fun i => ¬ hitAt sch D Q D_late Q_late W mach arr a h i).card * sch.ℓ := by
    unfold bad at hb
    convert Nat.lt_of_not_le hb using 4
  have hr := good_le_p₀ sch.B sch.ℓ β _ _ _ sch.B_pos hgb hb'
  rw [p₀_eq] at hr
  have hB : (0 : ℝ) < sch.B := by exact_mod_cast sch.B_pos
  have hgb' : ((Finset.univ.filter (hitAt sch D Q D_late Q_late W mach arr a h)).card : ℝ) +
      (Finset.univ.filter fun i => ¬ hitAt sch D Q D_late Q_late W mach arr a h i).card = sch.B := by
    exact_mod_cast hgb
  rw [pr_toReal, Fintype.card_fin, le_div_iff₀ hB]
  nlinarith

end

/-- **Step (a)**, `Pous.Guarantees.PerAnswer`, from `TimedINC`. -/
theorem per_answer : Pous.Guarantees.PerAnswer := by
  intro M sch D Q D_late Q_late β π hinc W mach arr h
  exact le_trans (pr_mono fun a ha => by_contra fun hb =>
      ha (pr_hit_le sch D Q D_late Q_late W mach arr β a h hb))
    (pr_bad_le sch D Q D_late Q_late W mach arr β π hinc h)

open Classical in
/-- **The continuous audit's guarantee**, `Pous.Guarantees.ContinuousAudit`, from `TimedINC`. -/
theorem continuous_audit : Pous.Guarantees.ContinuousAudit := by
  intro M sch D Q D_late Q_late β n π hinc W mach arr f βe hβe
  have : Nonempty (Fin sch.B) := ⟨⟨0, sch.B_pos⟩⟩
  set G := ⨅ θ : Set.Ioi (0 : ℝ), ((θ : ℝ) * f + Real.log (1 / βe)) / (1 - Real.exp (-(θ : ℝ))) with hG
  let hist : (Fin n → Fin sch.B) → Fin n → List (Fin sch.B) := fun I j => (List.ofFn I).take j
  let hit := hitAt sch D Q D_late Q_late W mach arr
  let held : M.Ω × sch.Coins → List (Fin sch.B) → ℕ := fun a h => (heldAt sch D_late W mach arr a h).length
  let ev : M.Ω × sch.Coins → (Fin n → Fin sch.B) → Prop := fun a I =>
    ((Finset.univ.filter fun j : Fin n => ¬ hit a (hist I j) (I j)).card : ℝ) ≤ f ∧
      G < ∑ j : Fin n, max (contFreed sch (held a (hist I j)) - epsProof sch β) 0
  let q : M.Ω × sch.Coins → List (Fin sch.B) → ℝ := fun a h =>
    if bad sch D Q D_late Q_late W mach arr β a h then 0 else max (contFreed sch (held a h) - epsProof sch β) 0
  let ev' : M.Ω × sch.Coins → (Fin n → Fin sch.B) → Prop := fun a I =>
    ((Finset.univ.filter fun j : Fin n => ¬ hit a (hist I j) (I j)).card : ℝ) ≤ f ∧
      G < ∑ j : Fin n, q a (hist I j)
  let stop : M.Ω × sch.Coins → (Fin n → Fin sch.B) → Prop := fun a I =>
    ∃ j ∈ (Finset.univ : Finset (Fin n)), bad sch D Q D_late Q_late W mach arr β a (hist I j)
  have himp : ∀ x : Outcome sch n,
      (let pp := (sch.enc W x.1 x.2.1).1
       let C := (sch.enc W x.1 x.2.1).2
       let rs := contRecords mach D Q D_late Q_late W pp C x.1 arr .start (List.ofFn x.2.2)
       contAccepts f rs ∧ G < (rs.map fun r => max (contFreed sch r.held - epsProof sch β) 0).sum) →
      ev (x.1, x.2.1) x.2.2 := by
    rintro ⟨ω, r, I⟩ ⟨hcount, hsum⟩
    simp only [contAccepts, contRecords_ofFn] at hcount hsum
    refine ⟨le_trans ?_ hcount, hsum.trans_eq ?_⟩
    · rw [length_filter_ofFn]
      refine Nat.cast_le.2 (Finset.card_le_card fun j => ?_)
      simp only [Finset.mem_filter, Finset.mem_univ, true_and]
      exact id
    · rw [List.map_ofFn, List.sum_ofFn]
      rfl
  have hre : pr (fun x : Outcome sch n => ev (x.1, x.2.1) x.2.2) =
      pr (fun y : (M.Ω × sch.Coins) × (Fin n → Fin sch.B) => ev y.1 y.2) :=
    pr_equiv (Equiv.prodAssoc _ _ _).symm (fun y : (M.Ω × sch.Coins) × (Fin n → Fin sch.B) => ev y.1 y.2)
  refine le_trans (pr_mono himp) ?_
  rw [hre]
  have hsplit : ∀ y : (M.Ω × sch.Coins) × (Fin n → Fin sch.B), ev y.1 y.2 → stop y.1 y.2 ∨ ev' y.1 y.2 := by
    rintro ⟨a, I⟩ ⟨h1, h2⟩
    by_cases hs : stop a I
    · exact Or.inl hs
    · refine Or.inr ⟨h1, h2.trans_eq (Finset.sum_congr rfl fun j _ => ?_)⟩
      have : ¬ bad sch D Q D_late Q_late W mach arr β a (hist I j) := fun hb => hs ⟨j, Finset.mem_univ _, hb⟩
      simp only [q, this, ↓reduceIte]
  have hstop : pr (fun y : (M.Ω × sch.Coins) × (Fin n → Fin sch.B) => stop y.1 y.2) ≤ n * (β + 1) * π := by
    refine (pr_exists_le _ _).trans ?_
    calc ∑ j : Fin n, pr (fun y : (M.Ω × sch.Coins) × (Fin n → Fin sch.B) =>
            bad sch D Q D_late Q_late W mach arr β y.1 (hist y.2 j))
        ≤ ∑ _j : Fin n, ((β : ℝ≥0∞) + 1) * π := by
          refine Finset.sum_le_sum fun j _ => ?_
          rw [← pr_equiv (Equiv.prodComm _ _)]
          simp only [Equiv.prodComm_apply, Prod.fst_swap, Prod.snd_swap]
          exact pr_prod_le_of_forall (fun I a => bad sch D Q D_late Q_late W mach arr β a (hist I j)) _
            fun I => pr_bad_le sch D Q D_late Q_late W mach arr β π hinc (hist I j)
      _ = n * (β + 1) * π := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_assoc]
  have hmain : pr (fun y : (M.Ω × sch.Coins) × (Fin n → Fin sch.B) => ev' y.1 y.2) ≤ ENNReal.ofReal βe := by
    refine pr_prod_le_of_forall ev' _ fun a => ?_
    have hq : ∀ h : List (Fin sch.B), h.length < n → q a h ≤ (pr fun i => ¬ hit a h i).toReal := by
      intro h _
      by_cases hb : bad sch D Q D_late Q_late W mach arr β a h
      · simp only [q, hb, ↓reduceIte]
        exact ENNReal.toReal_nonneg
      · simp only [q, hb, ↓reduceIte]
        exact max_le (miss_ge sch D Q D_late Q_late W mach arr β a h hb) ENNReal.toReal_nonneg
    convert pr_misses_gt_le n (fun h i => ¬ hit a h i) (q a) hq f βe hβe
  calc pr (fun y : (M.Ω × sch.Coins) × (Fin n → Fin sch.B) => ev y.1 y.2)
      ≤ pr (fun y : (M.Ω × sch.Coins) × (Fin n → Fin sch.B) => stop y.1 y.2 ∨ ev' y.1 y.2) := pr_mono hsplit
    _ ≤ pr (fun y : (M.Ω × sch.Coins) × (Fin n → Fin sch.B) => stop y.1 y.2) +
          pr (fun y : (M.Ω × sch.Coins) × (Fin n → Fin sch.B) => ev' y.1 y.2) := pr_or_le _ _
    _ ≤ n * (β + 1) * π + ENNReal.ofReal βe := add_le_add hstop hmain
    _ = ENNReal.ofReal βe + n * (β + 1) * π := add_comm _ _

end PousContinuous
