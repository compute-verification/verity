import Proofs.Flock.Soundness.Discharge.Exec.CompiledTableL
import Proofs.Flock.Soundness.Discharge.Exec.SessionL
import Proofs.Flock.Soundness.Session

/-!
# A salted table inside a batched session

`Session.lean` over a leaf scheme `Lf`: the table after the link points is `tableAfterL`, whose openings carry salts
and verify from their rows' leaves under them (`OpensOKL`). The arguments are `Session.lean`'s, with the salted
lockstep (`lockAfterL`, `lockTableL`'s body) and the salted openings (`op₀L`, `opRL`, `SelfClashL`); the oracle side
and the reps' locksteps are unchanged.

* `session_sound_of_table_accL`: for an event `Acc` of the session's outputs that implies the table accepts, and any
  level-0 table `T`, `Pr[Acc ∧ T uncommitted] ≤ ε_c⁻ + Pr[Acc ∧ a run opens a level-0 position off T]` (`epsSL`).
* `session_knowledge_soundL`: the table accepts while the extraction from `K ≥ 1` reruns after `Commit`
  (`extractTableSL`) fails with probability at most `ε_c⁻ + K·adv₀ + N₀/(eK)` (`adv₀SL`).
-/

namespace FlockSoundness.Discharge.Exec

open Game Model Accounting

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
variable {D Sl : Type}

/-! ## The session after `Commit` -/

/-- The oracle table's run after the link points, with the dummy last message. -/
def tableOrcAfterL (A : Arith F K) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m)
    (ptLocal mPts : ℕ) (C₀ : Oracle F) (pts : Fin 2 → Fin mPts → F) : Game (Bool × Oracle F) :=
  (interleave (rep A S sch hm C₀ (extraClaims S ptLocal mPts pts))
    (rep A S sch hm C₀ (extraClaims S ptLocal mPts pts))).bind
    fun acc => Game.send (Opens F D × Salts Sl) fun _ => Game.ret (acc.1 && acc.2, C₀)

omit [Fintype K] in
theorem tableOrcL_eq (A : Arith F K) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m)
    (ptLocal mPts : ℕ) (C₀ : Oracle F) :
    tableOrcL (D := D) (Sl := Sl) A S sch hm ptLocal mPts C₀ =
      Game.coin (Fin 2 → Fin mPts → F) fun pts =>
        tableOrcAfterL (D := D) (Sl := Sl) A S sch hm ptLocal mPts C₀ pts :=
  rfl

/-- **A batched session after `Commit`** (PROTOCOL.md §5): one link-point draw, then the games
`pre`, the table with level-0 cap `cap₀`, and the games `post`, concurrently. -/
noncomputable def sessAfterL (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutCL F D Sl))) (cap₀ : Cap D) :
    Game (List (OutCL F D Sl)) :=
  Game.coin (Fin 2 → Fin mPts → F) fun pts =>
    batch (pre.map (· pts) ++ tableAfterL A H E Lf S sch hm ptLocal mPts cap₀ pts :: post.map (· pts))

open Classical in
/-- The oracle table's output as a compiled output: accepted when both reps accept and the table is
not committed. -/
noncomputable def orcOutL [Inhabited Sl] (A : Arith F K) (S : Statement) (sch : Schedule) (cap₀ : Cap D)
    (a : Bool × Oracle F) : OutCL F D Sl :=
  ⟨a.1 && decide (¬ Committed A S sch a.2), cap₀, (PendL.reject, PendL.reject),
    fun _ _ _ => ([], []), fun _ _ _ => default⟩

/-- The oracle session: the table's oracle run in its place. -/
noncomputable def orcSessL [Inhabited Sl] (A : Arith F K) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m)
    (ptLocal mPts : ℕ) (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutCL F D Sl)))
    (C₀ : Oracle F) (cap₀ : Cap D) : Game (List (OutCL F D Sl)) :=
  Game.coin (Fin 2 → Fin mPts → F) fun pts =>
    batch (pre.map (· pts) ++
      (tableOrcAfterL (D := D) (Sl := Sl) A S sch hm ptLocal mPts C₀ pts).map (orcOutL A S sch cap₀) ::
        post.map (· pts))

/-- The session's leaf relation: the table's, at its position. -/
def SessRelL [Inhabited D] [Inhabited Sl] (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (S : Statement) (sch : Schedule) (C₀ : Oracle F) (cap₀ : Cap D) (n : ℕ)
    (i : IRep F D × IRep F D) (a b : List (OutCL F D Sl)) : Prop :=
  ∃ x, a.getD n dfltOutL = orcOutL A S sch cap₀ x ∧ TopRelL H E Lf sch C₀ cap₀ i x (b.getD n dfltOutL)

/-- The table's lockstep after the link points (`lockTable` below its coin). -/
noncomputable def lockAfterL [Inhabited D] [Inhabited Sl] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level)
    (ext : ℕ) (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (C₀ : Oracle F)
    (cap₀ : Cap D) (pts : Fin 2 → Fin mPts → F) :
    Lock (Oracle F) (Cap D) (Lock.getPair (getRep 0) (getRep 1))
      (TopRelL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ C₀ cap₀) ((fun _ => none), (fun _ => none))
      (tableOrcAfterL (D := D) (Sl := Sl) A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts C₀ pts)
      (tableAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts cap₀ pts) :=
  Lock.bind (R := Lock.RPair (RecRel 0 C₀ ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
      (RecRel 1 C₀ ⟨m, k0, l₀ :: l₁ :: rest, ext⟩))
    (fun i acc pend h => .send (Opens F D × Salts Sl) fun os => .ret (by
      obtain ⟨⟨hv₀, hs₀⟩, ⟨hv₁, hs₁⟩⟩ := h
      refine ⟨rfl, rfl, fun hacc ha₀ ha₁ => ?_, fun hacc => ?_, hs₀, hs₁⟩
      · simp only [Bool.and_eq_true, decide_eq_true_eq] at hacc
        rw [← hv₀ os.1 ha₀, ← hv₁ os.1 ha₁]
        simp [hacc.1.1, hacc.1.2]
      · simp only [Bool.and_eq_true, decide_eq_true_eq] at hacc
        exact hacc.2))
    (Lock.interleaveLock disjoint_reps
      (lockRep A 0 (by norm_num) S m k0 l₀ l₁ rest ext hk0 hm C₀ (extraClaims S ptLocal mPts pts))
      (lockRep A 1 (by norm_num) S m k0 l₀ l₁ rest ext hk0 hm C₀ (extraClaims S ptLocal mPts pts)))

theorem aligned_lockAfterL [Inhabited D] [Inhabited Sl] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level)
    (ext : ℕ) (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (C₁ C₂ : Oracle F)
    (cap₀ : Cap D) (pts : Fin 2 → Fin mPts → F) :
    Lock.Aligned (lockAfterL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts C₁ cap₀ pts)
      (lockAfterL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts C₂ cap₀ pts) := by
  refine Lock.Aligned.bind ?_ (Lock.Aligned.interleaveLock disjoint_reps
    (aligned_lockRep A 0 (by norm_num) S m k0 l₀ l₁ rest ext hk0 hm C₁ C₂ _)
    (aligned_lockRep A 1 (by norm_num) S m k0 l₀ l₁ rest ext hk0 hm C₁ C₂ _))
  intro _ _ _ _ _ _ _
  exact .send _ fun _ => .ret _ _

/-- **The table in lockstep inside the session**: its lockstep at position `pre.length`, the same
game on both sides everywhere else. -/
noncomputable def lockSessL [Inhabited D] [Inhabited Sl] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level)
    (ext : ℕ) (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutCL F D Sl))) (C₀ : Oracle F) (cap₀ : Cap D) :
    Lock (Oracle F) (Cap D) (Lock.getPair (getRep 0) (getRep 1))
      (SessRelL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ C₀ cap₀ pre.length)
      ((fun _ => none), (fun _ => none))
      (orcSessL (D := D) (Sl := Sl) A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post C₀ cap₀)
      (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) :=
  .coin (Fin 2 → Fin mPts → F) fun pts =>
    Lock.mono (fun _ a b h => by simpa only [SessRelL, List.length_map] using h)
      (Lock.batchLock dfltOutL
        (Lock.mapL (R₂ := fun i o b => ∃ x, o = orcOutL A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ x ∧
            TopRelL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ C₀ cap₀ i x b)
          (orcOutL A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀) (fun _ a _ hr => ⟨a, rfl, hr⟩)
          (lockAfterL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts C₀ cap₀ pts))
        (post.map (· pts)) (pre.map (· pts)))

theorem aligned_lockSessL [Inhabited D] [Inhabited Sl] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level)
    (ext : ℕ) (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutCL F D Sl))) (C₁ C₂ : Oracle F)
    (cap₀ : Cap D) :
    Lock.Aligned (lockSessL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post C₁ cap₀)
      (lockSessL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post C₂ cap₀) :=
  .coin _ fun pts => Lock.Aligned.mono (Lock.Aligned.batchLock dfltOutL
    (Lock.Aligned.mapL (aligned_lockAfterL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts C₁ C₂ cap₀
      pts)) _ _)

/-! ## The oracle side -/

open Classical in
/-- **The oracle session is bounded by the table.** Whatever the other games, the oracle table in
the session accepts uncommitted with probability at most `tableError`, for an optimal prover. -/
theorem orcSess_value_leL (A : Arith F K) (hA : A.Correct)
    [Inhabited D] [Inhabited Sl] (S : Statement) (sch : Schedule) (hsch : fast100 S.m = some sch) (hm : 13 ≤ S.m)
    (ptLocal mPts : ℕ) (hlay : S.LinkLayout ptLocal mPts)
    (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutCL F D Sl))) (C₀ : Oracle F) (cap₀ : Cap D) :
    value (fun outs : List (OutCL F D Sl) => (outs.getD pre.length dfltOutL).accepted = true)
        (orcSessL (D := D) (Sl := Sl) A S sch hm ptLocal mPts pre post C₀ cap₀) ≤
      ENNReal.ofReal (tableError sch (S.shape mPts)) := by
  calc value (fun outs : List (OutCL F D Sl) => (outs.getD pre.length dfltOutL).accepted = true)
        (orcSessL (D := D) (Sl := Sl) A S sch hm ptLocal mPts pre post C₀ cap₀)
      ≤ value (fun o : OutCL F D Sl => o.accepted = true) (Game.coin (Fin 2 → Fin mPts → F) fun pts =>
          (tableOrcAfterL (D := D) (Sl := Sl) A S sch hm ptLocal mPts C₀ pts).map (orcOutL A S sch cap₀)) := by
        simp only [orcSessL, value_coin]
        refine Finset.sum_le_sum fun pts _ => mul_le_mul_right ?_ _
        have := value_batch_getD_le (fun o : OutCL F D Sl => o.accepted = true) dfltOutL
          ((tableOrcAfterL (D := D) (Sl := Sl) A S sch hm ptLocal mPts C₀ pts).map (orcOutL A S sch cap₀))
          (post.map (· pts)) (pre.map (· pts))
        simpa only [List.length_map] using this
    _ = value (fun a : Bool × Oracle F => a.1 = true ∧ ¬ Committed A S sch a.2)
          (tableOrcL (D := D) (Sl := Sl) A S sch hm ptLocal mPts C₀) := by
        rw [tableOrcL_eq]
        simp only [value_coin, value_map]
        refine Finset.sum_congr rfl fun pts _ => ?_
        congr 1
        exact value_congr (fun a => by simp [orcOutL]) _
    _ ≤ value (fun a : Bool × Oracle F => a.1 = true ∧ ¬ Committed A S sch a.2)
          (Game.send (Oracle F) (tableOrcL (D := D) (Sl := Sl) A S sch hm ptLocal mPts)) := by
        rw [value_send]
        exact le_iSup (fun C => value (fun a : Bool × Oracle F => a.1 = true ∧
          ¬ Committed A S sch a.2) (tableOrcL (D := D) (Sl := Sl) A S sch hm ptLocal mPts C)) C₀
    _ = value (fun a : Bool × Oracle F => a.1 = true ∧ ¬ Committed A S sch a.2)
          (table A S sch hm ptLocal mPts) := by
        rw [← withDummy_tableL, Lock.value_withDummy]
    _ ≤ _ := table_value_sound A hA S sch hsch hm ptLocal mPts hlay

/-! ## Rewinding in the session -/

open Classical in
/-- The table `P̃` sends at a rep's cap in the session: the plurality of the table's openings there. -/
noncomputable def plurSL [Inhabited D] [Inhabited Sl] (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (sch : Schedule) (n : ℕ) : Lock.Chooser (Oracle F) (Cap D) (List (OutCL F D Sl)) :=
  fun s d g σ => tabOf (lvOf sch (s / 2)).lanes (lvOf sch (s / 2)).logLen
    (Classical.choose (Rewinding.exists_pluralityE
      (fun outs : List (OutCL F D Sl) => opRL H E Lf sch (s % 2) (s / 2) d (outs.getD n dfltOutL)) g σ))

theorem plurS_pluralityL [Inhabited D] [Inhabited Sl] (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (sch : Schedule) (n r ℓ : ℕ) (hr : r < 2) (d : Cap D) (g : Game (List (OutCL F D Sl)))
    (σ : Strategy g) :
    Rewinding.IsPluralityE (fun outs : List (OutCL F D Sl) => opRL H E Lf sch r ℓ d (outs.getD n dfltOutL))
      g σ (rowT (lvOf sch ℓ).lanes (lvOf sch ℓ).logLen (plurSL H E Lf sch n (slotOf r ℓ) d g σ)) := by
  have key : ∀ s, Rewinding.IsPluralityE
      (fun outs : List (OutCL F D Sl) => opRL H E Lf sch (s % 2) (s / 2) d (outs.getD n dfltOutL)) g σ
      (rowT (lvOf sch (s / 2)).lanes (lvOf sch (s / 2)).logLen (plurSL H E Lf sch n s d g σ)) := by
    intro s
    unfold plurSL
    rw [rowT_tabOf]
    exact Classical.choose_spec (Rewinding.exists_pluralityE _ g σ)
  have h1 : slotOf r ℓ % 2 = r := by unfold slotOf; omega
  have h2 : slotOf r ℓ / 2 = ℓ := by unfold slotOf; omega
  have := key (slotOf r ℓ)
  rw [h1, h2] at this
  exact this

/-- **`Adv_{r,ℓ}` in the session**: the collision finder at the table's rep `r`'s level-`ℓ` cap. -/
noncomputable def advRSL [Inhabited D] [Inhabited Sl] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level)
    (ext : ℕ) (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutCL F D Sl))) (cap₀ : Cap D) (r ℓ : ℕ)
    (σ : Strategy (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀)) :
    ℝ :=
  Lock.forkE (plurSL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length) (slotOf r ℓ)
    (fun d b b' => ind (fun x : List (OutCL F D Sl) × List (OutCL F D Sl) => ∃ p, Rewinding.ConflictAt
      (fun outs : List (OutCL F D Sl) =>
        opRL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r ℓ d (outs.getD pre.length dfltOutL)) x p) (b, b'))
    (lockSessL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post default cap₀) σ

/-- **`Adv₀` in the session**: two runs after `Commit` open a level-0 position of the table to
different rows. -/
noncomputable def adv₀SL [Inhabited D] [Inhabited Sl] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutCL F D Sl))) (cap₀ : Cap D)
    (σ : Strategy (sessAfterL A H E Lf S sch hm ptLocal mPts pre post cap₀)) : ℝ :=
  expect₂ (fun b b' => ind (fun x : List (OutCL F D Sl) × List (OutCL F D Sl) => ∃ p,
      Rewinding.ConflictAt (fun outs : List (OutCL F D Sl) => op₀L H E Lf sch cap₀ (outs.getD pre.length
        dfltOutL)) x p) (b, b'))
    (sessAfterL A H E Lf S sch hm ptLocal mPts pre post cap₀) σ

open Classical in
/-- **`ε_c⁻` in the session**: `tableError`, the table's rep caps' fork terms and its self-clash. -/
noncomputable def epsSL [Inhabited D] [Inhabited Sl] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level)
    (ext : ℕ) (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutCL F D Sl))) (cap₀ : Cap D)
    (σ : Strategy (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀)) :
    ℝ :=
  tableError ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (S.shape mPts) +
    ∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 (l₀ :: l₁ :: rest).length,
      Real.sqrt (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).logLen *
        (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).queries *
        advRSL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post cap₀ r ℓ σ) +
    expect (ind fun outs : List (OutCL F D Sl) =>
        SelfClashL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (outs.getD pre.length dfltOutL))
      (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ

/-! ## The compiled bound for any table, in the session -/

open Classical in
/-- `bad_pointwise_accL` for an event of the whole session's outputs that implies the table at
position `n` accepts, such as "the session accepts". -/
theorem bad_pointwise_ofL [Inhabited D] [Inhabited Sl] (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (sch : Schedule)
    (C₀ : Oracle F) (cap₀ : Cap D) (i : IRep F D × IRep F D) (a : Bool × Oracle F) (n : ℕ)
    (Acc : List (OutCL F D Sl) → Prop) (hAcc : ∀ outs, Acc outs → (outs.getD n dfltOutL).accepted = true)
    (b : List (OutCL F D Sl)) (hR : TopRelL H E Lf sch C₀ cap₀ i a (b.getD n dfltOutL)) :
    ind (fun outs : List (OutCL F D Sl) => Acc outs ∧
      ¬ (AgreeRep 0 C₀ i.1 (outs.getD n dfltOutL).pend.1.reqs (outs.getD n dfltOutL).opens ∧
        AgreeRep 1 C₀ i.2 (outs.getD n dfltOutL).pend.2.reqs (outs.getD n dfltOutL).opens)) b ≤
    ind (SelfClashL H E Lf sch) (b.getD n dfltOutL) +
      ind (fun outs : List (OutCL F D Sl) => Acc outs ∧
        ∃ p, Rewinding.Off (op₀L H E Lf sch cap₀) (rowT _ _ C₀) (outs.getD n dfltOutL) p) b +
      ∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 sch.levels.length,
        ind (fun b => ∃ t d, Lock.getPair (getRep 0) (getRep 1) i (slotOf r ℓ) = some (t, d) ∧
          ∃ p, Rewinding.Off (opRL H E Lf sch r ℓ d) (rowT _ _ t) b p) (b.getD n dfltOutL) := by
  have h := bad_pointwise_accL H E Lf sch C₀ cap₀ i a (b.getD n dfltOutL) hR
  by_cases hA : Acc b
  · have e1 : ind (fun outs : List (OutCL F D Sl) => Acc outs ∧
        ¬ (AgreeRep 0 C₀ i.1 (outs.getD n dfltOutL).pend.1.reqs (outs.getD n dfltOutL).opens ∧
          AgreeRep 1 C₀ i.2 (outs.getD n dfltOutL).pend.2.reqs (outs.getD n dfltOutL).opens)) b =
        ind (fun out : OutCL F D Sl => out.accepted = true ∧ ¬ (AgreeRep 0 C₀ i.1 out.pend.1.reqs
          out.opens ∧ AgreeRep 1 C₀ i.2 out.pend.2.reqs out.opens)) (b.getD n dfltOutL) := by
      unfold ind
      simp only [hA, hAcc b hA, true_and]
    have e2 : ind (fun outs : List (OutCL F D Sl) => Acc outs ∧
        ∃ p, Rewinding.Off (op₀L H E Lf sch cap₀) (rowT _ _ C₀) (outs.getD n dfltOutL) p) b =
        ind (fun out : OutCL F D Sl => out.accepted = true ∧
          ∃ p, Rewinding.Off (op₀L H E Lf sch cap₀) (rowT _ _ C₀) out p) (b.getD n dfltOutL) := by
      unfold ind
      simp only [hA, hAcc b hA, true_and]
    rw [e1, e2]
    exact h
  · have h0 : ind (fun outs : List (OutCL F D Sl) => Acc outs ∧
        ¬ (AgreeRep 0 C₀ i.1 (outs.getD n dfltOutL).pend.1.reqs (outs.getD n dfltOutL).opens ∧
          AgreeRep 1 C₀ i.2 (outs.getD n dfltOutL).pend.2.reqs (outs.getD n dfltOutL).opens)) b = 0 := by
      simp [ind, hA]
    rw [h0]
    have h1 := ind_nonneg (SelfClashL H E Lf sch) (b.getD n dfltOutL)
    have h2 := ind_nonneg (fun outs : List (OutCL F D Sl) => Acc outs ∧
      ∃ p, Rewinding.Off (op₀L H E Lf sch cap₀) (rowT _ _ C₀) (outs.getD n dfltOutL) p) b
    have h3 : 0 ≤ ∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 sch.levels.length,
        ind (fun b => ∃ t d, Lock.getPair (getRep 0) (getRep 1) i (slotOf r ℓ) = some (t, d) ∧
          ∃ p, Rewinding.Off (opRL H E Lf sch r ℓ d) (rowT _ _ t) b p) (b.getD n dfltOutL) :=
      Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => ind_nonneg _ _
    linarith

open Classical in
/-- **Compiled soundness for any level-0 table, in a batched session, for an acceptance event.**
For every event `Acc` of the session's outputs that implies the table accepts (the table's own
acceptance, or the session's), every session strategy after `Commit` and every table `T`: `Acc`
holds with `T` uncommitted with probability at most `ε_c⁻` plus the probability that `Acc` holds
in a run that opens a level-0 position off `T`. The coupling's bad event needs only the table's
acceptance (`bad_pointwise_ofL`). -/
theorem session_sound_of_table_accL (A : Arith F K)
    (hA : A.Correct) [Inhabited D] [Inhabited Sl] (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement)
    (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ)
    (hsch : fast100 S.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (hlay : S.LinkLayout ptLocal mPts)
    (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutCL F D Sl))) (cap₀ : Cap D)
    (Acc : List (OutCL F D Sl) → Prop)
    (hAcc : ∀ outs, Acc outs → (outs.getD pre.length dfltOutL).accepted = true) (T : Oracle F)
    (σ : Strategy (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀)) :
    expect (ind fun outs : List (OutCL F D Sl) => Acc outs ∧
        ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ T)
      (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ ≤
    epsSL A H E Lf S m k0 l₀ l₁ rest ext (fast100_shape S.m _ hsch).2.1 hm ptLocal mPts pre post cap₀ σ +
      expect (ind fun outs : List (OutCL F D Sl) => Acc outs ∧
          Off₀L H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ T (outs.getD pre.length dfltOutL))
        (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ := by
  have hk0 := (fast100_shape S.m _ hsch).2.1
  -- the coupling
  have hcpl := Lock.expect_le_coupled (plurSL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length)
    (fun outs : List (OutCL F D Sl) => Acc outs ∧ ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ T)
    (fun outs : List (OutCL F D Sl) => (outs.getD pre.length dfltOutL).accepted = true)
    (fun (i : IRep F D × IRep F D) (outs : List (OutCL F D Sl)) =>
      Acc outs ∧
        ¬ (AgreeRep 0 T i.1 (outs.getD pre.length dfltOutL).pend.1.reqs (outs.getD pre.length dfltOutL).opens ∧
          AgreeRep 1 T i.2 (outs.getD pre.length dfltOutL).pend.2.reqs (outs.getD pre.length dfltOutL).opens))
    (fun i a b hR hE hB => by
      obtain ⟨x, hx, ha2, -, hacc, -, -, -⟩ := hR
      rw [hx]
      have h1 : x.1 = true := by
        by_contra hne
        exact hB ⟨hE.1, fun h => hne (hacc (hAcc _ hE.1) h.1 h.2)⟩
      simp only [orcOutL, h1, Bool.true_and, decide_eq_true_eq]
      rw [ha2]
      exact hE.2)
    (lockSessL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post T cap₀) σ
  -- the oracle side
  have horc : expect (ind fun outs : List (OutCL F D Sl) => (outs.getD pre.length dfltOutL).accepted = true)
      (orcSessL (D := D) (Sl := Sl) A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post T cap₀) (Lock.oracleStrat (plurSL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length) (lockSessL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post T cap₀) σ) ≤
      tableError ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (S.shape mPts) := by
    rw [← prob_toReal]
    exact ENNReal.toReal_le_of_le_ofReal (tableError_nonneg _ _)
      ((prob_le_value _ _ _).trans (orcSess_value_leL A hA S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hsch hm ptLocal mPts hlay
        pre post T cap₀))
  -- the bad runs, cap by cap
  have hsplit := Lock.leafE_mono_rel (plurSL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length)
    (F := fun (i : IRep F D × IRep F D) (outs : List (OutCL F D Sl)) => ind (fun outs : List (OutCL F D Sl) =>
      Acc outs ∧
        ¬ (AgreeRep 0 T i.1 (outs.getD pre.length dfltOutL).pend.1.reqs
            (outs.getD pre.length dfltOutL).opens ∧
          AgreeRep 1 T i.2 (outs.getD pre.length dfltOutL).pend.2.reqs
            (outs.getD pre.length dfltOutL).opens)) outs)
    (fun i a b hR => by
      obtain ⟨x, -, hrel⟩ := hR
      exact bad_pointwise_ofL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ T cap₀ i x pre.length Acc hAcc b hrel) (lockSessL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post T cap₀) σ
  rw [Lock.leafE_add, Lock.leafE_add, Lock.leafE_const, Lock.leafE_const] at hsplit
  simp only [Lock.leafE_sum] at hsplit
  have hR : ∀ r ∈ Finset.range 2, ∀ ℓ ∈ Finset.Icc 1 (l₀ :: l₁ :: rest).length,
      Lock.leafE (plurSL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length)
        (fun (i : IRep F D × IRep F D) (outs : List (OutCL F D Sl)) => ind (fun b => ∃ t d,
          Lock.getPair (getRep 0) (getRep 1) i (slotOf r ℓ) = some (t, d) ∧
            ∃ p, Rewinding.Off (opRL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r ℓ d) (rowT _ _ t) b p) (outs.getD pre.length dfltOutL)) (lockSessL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post T cap₀) σ ≤
      Real.sqrt (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).logLen * (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).queries *
        advRSL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post cap₀ r ℓ σ) := by
    intro r hr ℓ _
    have := Lock.leafE_off_le
      (fun d (outs : List (OutCL F D Sl)) => opRL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r ℓ d (outs.getD pre.length dfltOutL))
      (rowT (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).lanes (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).logLen) (slotOf r ℓ) (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).queries
      (fun d b => (opOfL_count _ _ _ _ _ _ _ _).trans (slotOpensL_length _ _ _ _))
      (plurSL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length)
      (fun d g σ' => plurS_pluralityL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length r ℓ (Finset.mem_range.1 hr) d g σ') (lockSessL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post T cap₀) σ
      (getPair_init_none _)
    rw [Fintype.card_fin] at this
    have heq : advRSL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post cap₀ r ℓ σ =
        Lock.forkE (plurSL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length) (slotOf r ℓ)
          (fun d b b' => ind (fun x : List (OutCL F D Sl) × List (OutCL F D Sl) => ∃ p,
            Rewinding.ConflictAt (fun outs : List (OutCL F D Sl) => opRL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r ℓ d
              (outs.getD pre.length dfltOutL)) x p) (b, b')) (lockSessL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post T cap₀) σ :=
      Lock.forkE_congr _ _ _ (aligned_lockSessL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre
        post default T cap₀) σ
    rw [heq]
    show Lock.leafE (plurSL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length)
        (fun (i : IRep F D × IRep F D) (b : List (OutCL F D Sl)) => ind (fun b => ∃ t d,
          Lock.getPair (getRep 0) (getRep 1) i (slotOf r ℓ) = some (t, d) ∧
            ∃ p, Rewinding.Off (fun outs : List (OutCL F D Sl) =>
              opRL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r ℓ d (outs.getD pre.length dfltOutL))
              (rowT _ _ t) b p) b)
        (lockSessL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post T cap₀) σ ≤ _
    simpa [Nat.cast_pow] using this
  have hsum := Finset.sum_le_sum fun r hr => Finset.sum_le_sum fun ℓ hℓ => hR r hr ℓ hℓ
  have e0 : expect (ind fun outs : List (OutCL F D Sl) => Acc outs ∧
        Off₀L H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ T (outs.getD pre.length dfltOutL))
      (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ =
    expect (fun outs : List (OutCL F D Sl) =>
        ind (fun outs : List (OutCL F D Sl) => Acc outs ∧ ∃ p, Rewinding.Off (op₀L H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀) (rowT _ _ T) (outs.getD pre.length dfltOutL) p) outs)
      (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ := rfl
  have eS : expect (ind fun outs : List (OutCL F D Sl) => SelfClashL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (outs.getD pre.length dfltOutL))
      (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ =
    expect (fun outs : List (OutCL F D Sl) => ind (SelfClashL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (outs.getD pre.length dfltOutL))
      (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ := rfl
  unfold epsSL
  rw [e0, eS]
  linarith

open Classical in
/-- **Compiled soundness for any level-0 table, in a batched session.** For every session strategy
after `Commit` and every table `T`: the table accepts with `T` uncommitted with probability at most
`ε_c⁻` plus the probability that a run opens a level-0 position of the table off `T`. -/
theorem session_sound_of_tableL (A : Arith F K) (hA : A.Correct)
    [Inhabited D] [Inhabited Sl] (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement)
    (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ)
    (hsch : fast100 S.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (hlay : S.LinkLayout ptLocal mPts)
    (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutCL F D Sl))) (cap₀ : Cap D) (T : Oracle F)
    (σ : Strategy (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀)) :
    expect (ind fun outs : List (OutCL F D Sl) => (outs.getD pre.length dfltOutL).accepted = true ∧
        ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ T)
      (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ ≤
    epsSL A H E Lf S m k0 l₀ l₁ rest ext (fast100_shape S.m _ hsch).2.1 hm ptLocal mPts pre post cap₀ σ +
      expect (ind fun outs : List (OutCL F D Sl) =>
          Off₀L H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ T (outs.getD pre.length dfltOutL))
        (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ :=
  (session_sound_of_table_accL A hA H E Lf S m k0 l₀ l₁ rest ext hsch hm ptLocal mPts hlay pre post
    cap₀ (fun outs => (outs.getD pre.length dfltOutL).accepted = true) (fun _ h => h) T σ).trans
    (add_le_add le_rfl (expect_mono (fun outs => ind_mono (fun _ (h : _ ∧ _) => h.2) outs) _ _))

/-! ## Knowledge soundness in the session -/

/-- **The extractor's table in the session**: at each level-0 position of the table, the first
verifying opening one of the runs `bs` after `Commit` made there, and zero elsewhere. -/
noncomputable def extractTableSL [Inhabited D] [Inhabited Sl] (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (sch : Schedule) (n : ℕ) (cap₀ : Cap D) (bs : List (List (OutCL F D Sl))) : Oracle F :=
  tabOf (lvOf sch 0).lanes (lvOf sch 0).logLen
    (Extract.firstTable (fun outs : List (OutCL F D Sl) => op₀L H E Lf sch cap₀ (outs.getD n dfltOutL))
      (fun _ _ => 0) bs)

theorem epsS_nonnegL [Inhabited D] [Inhabited Sl] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level)
    (ext : ℕ) (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutCL F D Sl))) (cap₀ : Cap D)
    (σ : Strategy (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀)) :
    0 ≤ epsSL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post cap₀ σ := by
  unfold epsSL
  have h1 := tableError_nonneg ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (S.shape mPts)
  have h2 := expect_nonneg (ind_nonneg fun outs : List (OutCL F D Sl) =>
      SelfClashL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (outs.getD pre.length dfltOutL))
    (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ
  have h3 : 0 ≤ ∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 (l₀ :: l₁ :: rest).length,
      Real.sqrt (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).logLen *
        (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).queries *
        advRSL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post cap₀ r ℓ σ) :=
    Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => Real.sqrt_nonneg _
  linarith

open Classical in
/-- **Knowledge soundness of a table in a batched session, the audit's form.** The probability that
the table accepts in a fresh run of the session after `Commit`, while the extraction from `K ≥ 1`
independent reruns of everything after `Commit` fails, is at most `ε_c⁻ + K·adv₀ + N₀/(eK)`, with
no condition on the acceptance probability. -/
theorem session_knowledge_soundL (A : Arith F K) (hA : A.Correct)
    [Inhabited D] [Inhabited Sl] (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement)
    (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ)
    (hsch : fast100 S.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (hlay : S.LinkLayout ptLocal mPts)
    (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutCL F D Sl))) (cap₀ : Cap D)
    (σ : Strategy (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀))
    (Kr : ℕ) (hK : 1 ≤ Kr) :
    expectN (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ Kr
        (fun bs => expect (ind fun outs : List (OutCL F D Sl) =>
            (outs.getD pre.length dfltOutL).accepted = true ∧
              ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
                (extractTableSL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length cap₀ bs))
          (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ) ≤
      epsSL A H E Lf S m k0 l₀ l₁ rest ext (fast100_shape S.m _ hsch).2.1 hm ptLocal mPts pre post cap₀ σ +
        (Kr * adv₀SL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀ σ +
          2 ^ l₀.logLen / (Real.exp 1 * Kr)) := by
  set ε := expect (ind fun outs : List (OutCL F D Sl) => (outs.getD pre.length dfltOutL).accepted = true) (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ
    with hεdef
  -- a fresh run is independent of the extractor's runs: the joint probability is a product
  have hprod : ∀ bs : List (List (OutCL F D Sl)),
      expect (ind fun outs : List (OutCL F D Sl) => (outs.getD pre.length dfltOutL).accepted = true ∧
          ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (extractTableSL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length cap₀ bs)) (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ =
        ε * ind (fun bs => ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (extractTableSL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length cap₀ bs)) bs := by
    intro bs
    by_cases hc : Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (extractTableSL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length cap₀ bs)
    · have : (ind fun outs : List (OutCL F D Sl) => (outs.getD pre.length dfltOutL).accepted = true ∧
          ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (extractTableSL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length cap₀ bs)) = fun _ => 0 := by
        funext outs; simp [ind, hc]
      rw [this, expect_const]
      simp [ind, hc]
    · have : (ind fun outs : List (OutCL F D Sl) => (outs.getD pre.length dfltOutL).accepted = true ∧
          ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (extractTableSL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length cap₀ bs)) =
          ind fun outs : List (OutCL F D Sl) => (outs.getD pre.length dfltOutL).accepted = true := by
        funext outs
        unfold ind
        by_cases h : (outs.getD pre.length dfltOutL).accepted = true
        · rw [if_pos ⟨h, hc⟩, if_pos h]
        · rw [if_neg (fun h' => h h'.1), if_neg h]
      rw [this, show ind (fun bs => ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (extractTableSL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length cap₀ bs)) bs = 1
        from if_pos hc, mul_one]
  rw [show (fun bs : List (List (OutCL F D Sl)) => expect (ind fun outs : List (OutCL F D Sl) =>
      (outs.getD pre.length dfltOutL).accepted = true ∧ ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (extractTableSL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length cap₀ bs))
      (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ) = fun bs => ε * ind (fun bs => ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (extractTableSL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length cap₀ bs)) bs
    from funext hprod, expectN_const_mul]
  -- the fixed-table fact: an uncommitted extraction costs off-table mass
  have hcost : ∀ bs : List (List (OutCL F D Sl)),
      ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (extractTableSL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length cap₀ bs) →
      ε ≤ epsSL A H E Lf S m k0 l₀ l₁ rest ext (fast100_shape S.m _ hsch).2.1 hm ptLocal mPts pre post
          cap₀ σ +
        expect (ind fun ω => ∃ p, Rewinding.Off (fun outs : List (OutCL F D Sl) => op₀L H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (outs.getD pre.length dfltOutL)) (Extract.firstTable (fun outs : List (OutCL F D Sl) => op₀L H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (outs.getD pre.length dfltOutL)) (fun _ _ => 0) bs) ω p) (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ := by
    intro bs hbs
    have h := session_sound_of_tableL A hA H E Lf S m k0 l₀ l₁ rest ext hsch hm ptLocal mPts hlay
      pre post cap₀ (extractTableSL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length cap₀ bs) σ
    have e1 : (ind fun outs : List (OutCL F D Sl) => (outs.getD pre.length dfltOutL).accepted = true ∧
        ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (extractTableSL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length cap₀ bs)) =
        ind fun outs : List (OutCL F D Sl) => (outs.getD pre.length dfltOutL).accepted = true := by
      funext outs
      unfold ind
      by_cases h : (outs.getD pre.length dfltOutL).accepted = true
      · rw [if_pos ⟨h, hbs⟩, if_pos h]
      · rw [if_neg (fun h' => h h'.1), if_neg h]
    have e2 : (ind fun outs : List (OutCL F D Sl) =>
        Off₀L H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (extractTableSL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length cap₀ bs) (outs.getD pre.length dfltOutL)) =
        ind fun ω => ∃ p, Rewinding.Off (fun outs : List (OutCL F D Sl) => op₀L H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (outs.getD pre.length dfltOutL)) (Extract.firstTable (fun outs : List (OutCL F D Sl) => op₀L H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (outs.getD pre.length dfltOutL)) (fun _ _ => 0) bs) ω p := by
      funext outs
      unfold ind Off₀L extractTableSL
      rw [rowT_tabOf]
    rw [e1, e2] at h
    exact h
  have hjoint := Extract.joint_of_fixed (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ
    (fun bs => expect (ind fun ω => ∃ p, Rewinding.Off (fun outs : List (OutCL F D Sl) => op₀L H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (outs.getD pre.length dfltOutL)) (Extract.firstTable (fun outs : List (OutCL F D Sl) => op₀L H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (outs.getD pre.length dfltOutL)) (fun _ _ => 0) bs)
      ω p) (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ)
    (fun bs => Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (extractTableSL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length cap₀ bs))
    (fun bs => expect_nonneg (ind_nonneg _) _ _) ε _
    (epsS_nonnegL A H E Lf S m k0 l₀ l₁ rest ext _ hm ptLocal mPts pre post cap₀ σ) hcost Kr
  have hmass := Extract.expect_off_firstTable_le' (fun outs : List (OutCL F D Sl) => op₀L H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (outs.getD pre.length dfltOutL)) (fun _ _ => 0) (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ Kr hK
  rw [Fintype.card_fin] at hmass
  have hmass' : expectN (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ Kr (fun bs => expect (ind fun ω => ∃ p,
        Rewinding.Off (fun outs : List (OutCL F D Sl) => op₀L H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (outs.getD pre.length dfltOutL)) (Extract.firstTable (fun outs : List (OutCL F D Sl) => op₀L H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (outs.getD pre.length dfltOutL)) (fun _ _ => 0) bs) ω p) (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ) ≤
      Kr * adv₀SL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀ σ + 2 ^ l₀.logLen / (Real.exp 1 * Kr) := by
    convert hmass using 2
    all_goals first
      | exact rfl
      | simp only [lvOf, List.getD_cons_zero, Nat.cast_pow, Nat.cast_ofNat]
  linarith

end FlockSoundness.Discharge.Exec
