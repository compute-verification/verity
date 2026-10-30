import Proofs.Flock.Soundness.CompiledSound
import Proofs.Flock.Soundness.CompiledAlign

/-!
# The compiled bound for any level-0 table

`table_sound_compiled` is stated for `T*(P*)`, the plurality of the level-0 openings (`C0star`).
The coupling works for any table `T` the oracle prover sends at the level-0 cap
(`compiled_le_bad`), and only the level-0 term of the bad runs depends on `T`: the probability that
a run opens a level-0 position off `T`. The reps' cap terms do not depend on the level-0 table
(`advRT_eq`, from the alignment of the locksteps). So

> `Pr[accepted ∧ T uncommitted] ≤ ε_c⁻ + Pr[a run opens a level-0 position off T]`
> (`table_sound_compiled_of_table`),

with `ε_c⁻ = tableError + Σ √(N_ℓ·Q_ℓ·Adv_{r,ℓ}) + Pr[self-clash]` (`epsCminus`), one number for
every `T`. This is the form the knowledge extractor uses (`Knowledge.lean`).
-/

namespace FlockSoundness

open Game Model Accounting

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
variable {D : Type}

/-- `Adv_{r,ℓ}` computed through the lockstep for the level-0 table `T`. -/
noncomputable def advRT [Inhabited D] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level)
    (ext : ℕ) (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (T : Oracle F) (r ℓ : ℕ)
    (σ : Strategy (tableC A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts)) : ℝ :=
  Lock.forkE (plurC H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (slotOf r ℓ)
    (fun d b b' => ind (fun x : OutC F D × OutC F D =>
      ∃ p, Rewinding.ConflictAt (opR H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r ℓ d) x p) (b, b'))
    (lockTable A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts T σ.1) σ.2

theorem advRT_C0star [Inhabited D] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level)
    (ext : ℕ) (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (r ℓ : ℕ)
    (σ : Strategy (tableC A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts)) :
    advRT A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts
        (C0star A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1 σ.2) r ℓ σ =
      advR A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts r ℓ σ :=
  rfl

open Classical in
/-- **The bad runs, cap by cap, for any level-0 table `C₀`**, given a bound `x₀` on the level-0
rewinding term. -/
theorem bad_le_gen [Inhabited D] (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (C₀ : Oracle F)
    (σ : Strategy (Game.send (Cap D)
      (tableCmp A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts))) (x₀ : ℝ)
    (h0 : expect (ind fun b => ∃ p, Rewinding.Off (op₀ H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1)
        (rowT _ _ C₀) b p)
      (tableCmp A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1) σ.2 ≤ x₀) :
    Lock.leafE (plurC H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
        (fun (i : IRep F D × IRep F D) (out : OutC F D) => ind (fun out : OutC F D =>
          out.accepted = true ∧ ¬ (AgreeRep 0 C₀ i.1 out.pend.1.reqs out.opens ∧
            AgreeRep 1 C₀ i.2 out.pend.2.reqs out.opens)) out)
        (lockTable A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts C₀ σ.1) σ.2 ≤
      expect (ind (SelfClash H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩))
          (tableC A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts) σ + x₀ +
        ∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 (l₀ :: l₁ :: rest).length,
          Real.sqrt (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).logLen *
            (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).queries *
            advRT A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts C₀ r ℓ σ) := by
  have hsplit := Lock.leafE_mono_rel (plurC H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (fun i a b hR => bad_pointwise H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ _ σ.1 i a b hR)
    (lockTable A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts C₀ σ.1) σ.2
  rw [Lock.leafE_add, Lock.leafE_add, Lock.leafE_const, Lock.leafE_const] at hsplit
  simp only [Lock.leafE_sum] at hsplit
  have eS : expect (ind (SelfClash H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩))
      (tableC A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts) σ =
    expect (ind (SelfClash H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩))
      (tableCmp A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1) σ.2 := rfl
  have hR : ∀ r ∈ Finset.range 2, ∀ ℓ ∈ Finset.Icc 1 (l₀ :: l₁ :: rest).length,
      Lock.leafE (plurC H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
        (fun (i : IRep F D × IRep F D) b => ind (fun b => ∃ t d,
          Lock.getPair (getRep 0) (getRep 1) i (slotOf r ℓ) = some (t, d) ∧
            ∃ p, Rewinding.Off (opR H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r ℓ d) (rowT _ _ t) b p) b)
        (lockTable A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts C₀ σ.1) σ.2 ≤
      Real.sqrt (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).logLen *
        (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).queries *
        advRT A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts C₀ r ℓ σ) := by
    intro r hr ℓ _
    have := Lock.leafE_off_le (opR H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r ℓ)
      (rowT (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).lanes (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).logLen)
      (slotOf r ℓ) (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).queries
      (fun d b => (opOf_count _ _ _ _ _ _ _).trans (slotOpens_length _ _ _ _))
      (plurC H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
      (fun d g σ' => plurC_plurality H E _ r ℓ (Finset.mem_range.1 hr) d g σ')
      (lockTable A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts C₀ σ.1) σ.2
      (getPair_init_none _)
    rw [Fintype.card_fin] at this
    unfold advRT
    simpa [Nat.cast_pow] using this
  have hsum := Finset.sum_le_sum fun r hr => Finset.sum_le_sum fun ℓ hℓ => hR r hr ℓ hℓ
  rw [eS]
  linarith

/-- **The reps' fork probabilities do not depend on the level-0 table.** -/
theorem advRT_eq [Inhabited D] (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (T : Oracle F) (r ℓ : ℕ)
    (σ : Strategy (tableC A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts)) :
    advRT A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts T r ℓ σ =
      advR A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts r ℓ σ :=
  Lock.forkE_congr _ _ _ (aligned_lockTable A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts T
    (C0star A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1 σ.2) σ.1) σ.2

open Classical in
/-- **`ε_c⁻`**: `tableError`, the reps' cap terms and the self-clash probability; the compiled
error without its level-0 rewinding term. -/
noncomputable def epsCminus [Inhabited D] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level)
    (ext : ℕ) (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (σ : Strategy (tableC A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts)) : ℝ :=
  tableError ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (S.shape mPts) +
    ∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 (l₀ :: l₁ :: rest).length,
      Real.sqrt (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).logLen *
        (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).queries *
        advR A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts r ℓ σ) +
    expect (ind (SelfClash H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩))
      (tableC A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts) σ

open Classical in
/-- A run opens a level-0 position, with a verifying path, to a row other than `T`'s. -/
def Off₀ (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (sch : Schedule) (cap₀ : Cap D)
    (T : Oracle F) (out : OutC F D) : Prop :=
  ∃ p, Rewinding.Off (op₀ H E sch cap₀) (rowT (lvOf sch 0).lanes (lvOf sch 0).logLen T) out p

open Classical in
/-- **Compiled soundness for any level-0 table** (the review's (A′)). For every compiled prover
`P*` and every table `T`: `Pr[accepted ∧ no codeword near T packs a satisfying witness]` is at most
`ε_c⁻` plus the probability that a run opens a level-0 position off `T`. -/
theorem table_sound_compiled_of_table (A : Arith F K)
    (hA : A.Correct) [Inhabited D] (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (S : Statement)
    (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ)
    (hsch : fast100 S.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (hlay : S.LinkLayout ptLocal mPts) (T : Oracle F)
    (σ : Strategy (tableC A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts)) :
    expect (ind fun out : OutC F D => out.accepted = true ∧
        ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ T)
      (tableC A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts) σ ≤
    epsCminus A H E S m k0 l₀ l₁ rest ext (fast100_shape S.m _ hsch).2.1 hm ptLocal mPts σ +
      expect (ind (Off₀ H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1 T))
        (tableCmp A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1) σ.2 := by
  revert σ
  show ∀ σ : Strategy (Game.send (Cap D)
    (tableCmp A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts)), _
  intro σ
  have h1 := compiled_le_bad A hA H E S m k0 l₀ l₁ rest ext (fast100_shape S.m _ hsch).2.1 hsch
    hm ptLocal mPts hlay T σ.1 σ.2
  have h2 := bad_le_gen A H E S m k0 l₀ l₁ rest ext (fast100_shape S.m _ hsch).2.1 hm ptLocal mPts T
    σ _ le_rfl
  have h3 : (∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 (l₀ :: l₁ :: rest).length,
      Real.sqrt (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).logLen *
        (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).queries *
        advRT A H E S m k0 l₀ l₁ rest ext (fast100_shape S.m _ hsch).2.1 hm ptLocal mPts T r ℓ σ)) =
    ∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 (l₀ :: l₁ :: rest).length,
      Real.sqrt (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).logLen *
        (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).queries *
        advR A H E S m k0 l₀ l₁ rest ext (fast100_shape S.m _ hsch).2.1 hm ptLocal mPts r ℓ σ) :=
    Finset.sum_congr rfl fun r _ => Finset.sum_congr rfl fun ℓ _ => by
      congr 1
      congr 1
      exact advRT_eq A H E S m k0 l₀ l₁ rest ext _ hm ptLocal mPts T r ℓ σ
  have e : expect (ind fun out : OutC F D => out.accepted = true ∧
        ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ T)
      (tableC A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts) σ =
    expect (ind fun out : OutC F D => out.accepted = true ∧
        ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ T)
      (tableCmp A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1) σ.2 := rfl
  rw [e]
  unfold epsCminus Off₀
  linarith

end FlockSoundness
