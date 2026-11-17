module

public import Definitions.Flock.Extract
public import Proofs.Flock.Soundness.CompiledTable
public import Proofs.Flock.Soundness.Extract
public import Proofs.Flock.Soundness.Game

/-! Knowledge soundness of one table: the extractor reruns the compiled prover after its level-0 cap `K` times and
decodes the table it fills (`table_knowledge_sound`, `table_knowledge_sound_joint_tight`). -/

@[expose] public section

namespace FlockSoundness

open Game Model Accounting

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
variable {D : Type}

noncomputable def extractTable (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (sch : Schedule)
    (cap₀ : Cap D) (bs : List (OutC F D)) : Oracle F :=
  tabOf (lvOf sch 0).lanes (lvOf sch 0).logLen
    (Extract.firstTable (op₀ H E sch cap₀) (fun _ _ => 0) bs)

theorem off₀_extractTable (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (sch : Schedule)
    (cap₀ : Cap D) (bs : List (OutC F D)) (out : OutC F D) :
    Off₀ H E sch cap₀ (extractTable H E sch cap₀ bs) out ↔
      ∃ p, Rewinding.Off (op₀ H E sch cap₀)
        (Extract.firstTable (op₀ H E sch cap₀) (fun _ _ => 0) bs) out p := by
  unfold Off₀ extractTable
  rw [rowT_tabOf]

theorem epsCminus_nonneg [Inhabited D] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level)
    (ext : ℕ) (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (σ : Strategy (tableC A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts)) :
    0 ≤ epsCminus A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts σ := by
  unfold epsCminus
  have h1 := tableError_nonneg ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (S.shape mPts)
  have h2 := expect_nonneg (ind_nonneg (SelfClash H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩))
    (tableC A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts) σ
  have h3 : 0 ≤ ∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 (l₀ :: l₁ :: rest).length,
      Real.sqrt (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).logLen *
        (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).queries *
        advR A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts r ℓ σ) :=
    Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => Real.sqrt_nonneg _
  linarith

open Classical in
theorem uncommitted_costs (A : Arith F K) (hA : A.Correct)
    [Inhabited D] (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (S : Statement)
    (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ)
    (hsch : fast100 S.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (hlay : S.LinkLayout ptLocal mPts)
    (σ : Strategy (tableC A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts))
    (bs : List (OutC F D))
    (hbs : ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
      (extractTable H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1 bs)) :
    expect (ind fun out : OutC F D => out.accepted = true)
        (tableC A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts) σ ≤
      epsCminus A H E S m k0 l₀ l₁ rest ext (fast100_shape S.m _ hsch).2.1 hm ptLocal mPts σ +
        expect (ind fun ω => ∃ p, Rewinding.Off (op₀ H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1)
            (Extract.firstTable (op₀ H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1) (fun _ _ => 0) bs)
            ω p)
          (tableCmp A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1) σ.2 := by
  have h := table_sound_compiled_of_table A hA H E S m k0 l₀ l₁ rest ext hsch hm ptLocal mPts
    hlay (extractTable H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1 bs) σ
  have e1 : (ind fun out : OutC F D => out.accepted = true ∧
      ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
        (extractTable H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1 bs)) =
      ind fun out : OutC F D => out.accepted = true := by
    funext out
    unfold ind
    by_cases h : out.accepted = true
    · rw [if_pos ⟨h, hbs⟩, if_pos h]
    · rw [if_neg (fun h' => h h'.1), if_neg h]
  have e2 : (ind (Off₀ H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1
      (extractTable H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1 bs))) =
      ind fun ω => ∃ p, Rewinding.Off (op₀ H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1)
        (Extract.firstTable (op₀ H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1) (fun _ _ => 0) bs) ω p := by
    funext out
    unfold ind
    by_cases h : Off₀ H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1
        (extractTable H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1 bs) out
    · rw [if_pos h, if_pos ((off₀_extractTable _ _ _ _ _ _).1 h)]
    · rw [if_neg h, if_neg (fun h' => h ((off₀_extractTable _ _ _ _ _ _).2 h'))]
  rw [e1, e2] at h
  exact h

open Classical in
theorem table_knowledge_sound (A : Arith F K) (hA : A.Correct)
    [Inhabited D] (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (S : Statement)
    (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ)
    (hsch : fast100 S.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (hlay : S.LinkLayout ptLocal mPts)
    (σ : Strategy (tableC A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts))
    (Kr : ℕ) (hK : 1 ≤ Kr)
    (hε : epsCminus A H E S m k0 l₀ l₁ rest ext (fast100_shape S.m _ hsch).2.1 hm ptLocal mPts σ <
      expect (ind fun out : OutC F D => out.accepted = true)
        (tableC A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts) σ) :
    expectN (tableCmp A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1) σ.2 Kr
        (ind fun bs => ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
          (extractTable H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1 bs)) ≤
      (Kr * adv₀ A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ +
          2 ^ l₀.logLen / (Real.exp 1 * Kr)) /
        (expect (ind fun out : OutC F D => out.accepted = true)
            (tableC A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts) σ -
          epsCminus A H E S m k0 l₀ l₁ rest ext (fast100_shape S.m _ hsch).2.1 hm ptLocal mPts σ) := by
  have htail := Extract.tail_of_fixed
    (tableCmp A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1) σ.2
    (fun bs => expect (ind fun ω => ∃ p, Rewinding.Off (op₀ H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1)
        (Extract.firstTable (op₀ H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1) (fun _ _ => 0) bs) ω p)
      (tableCmp A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1) σ.2)
    (fun bs => Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
      (extractTable H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1 bs))
    (fun bs => expect_nonneg (ind_nonneg _) _ _) _ _ hε
    (fun bs hbs => uncommitted_costs A hA H E S m k0 l₀ l₁ rest ext hsch hm ptLocal mPts hlay σ bs
      hbs) Kr
  have hmass := Extract.expect_off_firstTable_le' (op₀ H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1)
    (fun _ _ => 0) (tableCmp A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1) σ.2 Kr hK
  rw [Fintype.card_fin] at hmass
  have hpos : 0 < expect (ind fun out : OutC F D => out.accepted = true)
        (tableC A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts) σ -
      epsCminus A H E S m k0 l₀ l₁ rest ext (fast100_shape S.m _ hsch).2.1 hm ptLocal mPts σ := by
    linarith
  refine htail.trans (div_le_div_of_nonneg_right ?_ hpos.le)
  convert hmass using 2
  all_goals first
    | exact rfl
    | simp only [lvOf, List.getD_cons_zero, Nat.cast_pow, Nat.cast_ofNat]

open Classical in
theorem table_knowledge_sound_joint_tight (A : Arith F K)
    (hA : A.Correct) [Inhabited D] (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (S : Statement)
    (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ)
    (hsch : fast100 S.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (hlay : S.LinkLayout ptLocal mPts)
    (σ : Strategy (tableC A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts))
    (Kr : ℕ) (hK : 1 ≤ Kr) :
    expectN (tableCmp A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1) σ.2 Kr
        (fun bs => expect (ind fun out : OutC F D => out.accepted = true ∧
            ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
              (extractTable H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1 bs))
          (tableC A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts) σ) ≤
      epsCminus A H E S m k0 l₀ l₁ rest ext (fast100_shape S.m _ hsch).2.1 hm ptLocal mPts σ +
        (Kr * adv₀ A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ +
          2 ^ l₀.logLen / (Real.exp 1 * Kr)) := by
  revert σ
  show ∀ σ : Strategy (Game.send (Cap D)
    (tableCmp A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts)), _
  intro σ
  change expectN (tableCmp A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1) σ.2 Kr
      (fun bs => expect (ind fun out : OutC F D => out.accepted = true ∧
          ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
            (extractTable H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1 bs))
        (tableCmp A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1) σ.2) ≤ _
  set ε := expect (ind fun out : OutC F D => out.accepted = true)
    (tableCmp A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1) σ.2 with hεdef
  have hprod : ∀ bs : List (OutC F D),
      expect (ind fun out : OutC F D => out.accepted = true ∧
          ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
            (extractTable H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1 bs))
        (tableCmp A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1) σ.2 =
      ε * ind (fun bs => ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
        (extractTable H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1 bs)) bs := by
    intro bs
    by_cases hc : Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
      (extractTable H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1 bs)
    · have : (ind fun out : OutC F D => out.accepted = true ∧
          ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
            (extractTable H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1 bs)) = fun _ => 0 := by
        funext out; simp [ind, hc]
      rw [this, expect_const]
      simp [ind, hc]
    · have : (ind fun out : OutC F D => out.accepted = true ∧
          ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
            (extractTable H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1 bs)) =
          ind fun out : OutC F D => out.accepted = true := by
        funext out
        unfold ind
        by_cases h : out.accepted = true
        · rw [if_pos ⟨h, hc⟩, if_pos h]
        · rw [if_neg (fun h' => h h'.1), if_neg h]
      rw [this, show ind (fun bs => ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
        (extractTable H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1 bs)) bs = 1 from if_pos hc, mul_one]
  rw [show (fun bs : List (OutC F D) => expect (ind fun out : OutC F D => out.accepted = true ∧
      ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
        (extractTable H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1 bs))
      (tableCmp A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1) σ.2) =
    fun bs => ε * ind (fun bs => ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
      (extractTable H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1 bs)) bs from funext hprod,
    expectN_const_mul]
  have hjoint := Extract.joint_of_fixed
    (tableCmp A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1) σ.2
    (fun bs => expect (ind fun ω => ∃ p, Rewinding.Off (op₀ H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1)
        (Extract.firstTable (op₀ H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1) (fun _ _ => 0) bs) ω p)
      (tableCmp A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1) σ.2)
    (fun bs => Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
      (extractTable H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1 bs))
    (fun bs => expect_nonneg (ind_nonneg _) _ _) ε _
    (epsCminus_nonneg A H E S m k0 l₀ l₁ rest ext _ hm ptLocal mPts σ)
    (fun bs hbs => uncommitted_costs A hA H E S m k0 l₀ l₁ rest ext hsch hm ptLocal mPts hlay σ bs
      hbs) Kr
  have hmass := Extract.expect_off_firstTable_le' (op₀ H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1)
    (fun _ _ => 0) (tableCmp A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1) σ.2 Kr hK
  rw [Fintype.card_fin] at hmass
  have hmass' : expectN (tableCmp A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1) σ.2 Kr
      (fun bs => expect (ind fun ω => ∃ p,
          Rewinding.Off (op₀ H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1)
            (Extract.firstTable (op₀ H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1) (fun _ _ => 0) bs) ω p)
        (tableCmp A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1) σ.2) ≤
      Kr * adv₀ A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ +
        2 ^ l₀.logLen / (Real.exp 1 * Kr) := by
    convert hmass using 2
    all_goals first
      | exact rfl
      | simp only [lvOf, List.getD_cons_zero, Nat.cast_pow, Nat.cast_ofNat]
  linarith

open Classical in
theorem table_knowledge_sound_joint (A : Arith F K)
    (hA : A.Correct) [Inhabited D] (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (S : Statement)
    (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ)
    (hsch : fast100 S.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (hlay : S.LinkLayout ptLocal mPts)
    (σ : Strategy (tableC A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts))
    (Kr : ℕ) (hK : 1 ≤ Kr) :
    expectN (tableCmp A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1) σ.2 Kr
        (fun bs => expect (ind fun out : OutC F D => out.accepted = true ∧
            ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
              (extractTable H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1 bs))
          (tableC A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts) σ) ≤
      2 * epsCminus A H E S m k0 l₀ l₁ rest ext (fast100_shape S.m _ hsch).2.1 hm ptLocal mPts σ +
        2 * (Kr * adv₀ A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ +
          2 ^ l₀.logLen / (Real.exp 1 * Kr)) := by
  have h := table_knowledge_sound_joint_tight A hA H E S m k0 l₀ l₁ rest ext hsch hm ptLocal
    mPts hlay σ Kr hK
  have h1 := epsCminus_nonneg A H E S m k0 l₀ l₁ rest ext (fast100_shape S.m _ hsch).2.1 hm ptLocal
    mPts σ
  have h2 : 0 ≤ adv₀ A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ := by
    unfold adv₀ expect₂
    exact expect_nonneg (fun _ => expect_nonneg (fun _ => ind_nonneg _ _) _ _) _ _
  have h3 : 0 ≤ (2 : ℝ) ^ l₀.logLen / (Real.exp 1 * Kr) := by positivity
  have h4 : 0 ≤ (Kr : ℝ) * adv₀ A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ :=
    mul_nonneg (Nat.cast_nonneg _) h2
  linarith

end FlockSoundness
