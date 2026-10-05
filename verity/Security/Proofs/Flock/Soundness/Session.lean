import Proofs.Flock.Soundness.Knowledge
import Proofs.Flock.Soundness.Game.Batch

/-!
# A table inside a batched session (README.md §1.3)

**The session** follows PROTOCOL.md §5. `Commit` carries every table's level-0 cap before any coin
(S3, S13, S16). The link points are one draw for all tables. Then every table's reps and openings
run concurrently, in the order the prover chooses. So after `Commit`, a table sits in a batch with
the rest of the session, whatever that is (`sessAfter`): the games `pre` and `post` may be other
tables or anything else that reads the link points.

**The compiled bound for any table**, in the session (`session_sound_of_table`). The table's
lockstep runs below the link points, beside the same game on both sides everywhere else
(`Lock.batchLock`). The oracle prover `P̃` then plays a whole session, and the table's value bounds
it (`value_batch_getD_le`). Rewinding at the table's rep caps works as for one table. `ε_c⁻`
(`epsS`) is the table's own: `tableError`, its rep caps' fork terms and its self-clash, all
computed in the session.

**Knowledge soundness in the session** (`session_knowledge_sound`). The extractor reruns
everything after `Commit` `K` times, and fills the table from the table's openings in those runs
(`extractTableS`). The session accepts while that extraction fails with probability at most
`ε_c⁻ + K·adv₀ + N₀/(eK)`, the table theorem's form, with `adv₀` the level-0 finder in the session.
-/

namespace FlockSoundness

open Game Model Accounting

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
variable {D : Type}

/-! ## The session after `Commit` -/

open Classical in
/-- A table's run after the link points: both reps, then its openings. -/
noncomputable def tableAfter (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (cap₀ : Cap D)
    (pts : Fin 2 → Fin mPts → F) : Game (OutC F D) :=
  (interleave (repC A 0 S sch hm (extraClaims S ptLocal mPts pts))
    (repC A 1 S sch hm (extraClaims S ptLocal mPts pts))).bind
    fun pend => Game.send (Opens F D) fun o => Game.ret
      ⟨pend.1.verdict o && pend.2.verdict o &&
        decide (OpensOK H E cap₀ 0 pend.1 o ∧ OpensOK H E cap₀ 1 pend.2 o), cap₀, pend, o⟩

/-- The oracle table's run after the link points, with the dummy last message. -/
def tableOrcAfter (A : Arith F K) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m)
    (ptLocal mPts : ℕ) (C₀ : Oracle F) (pts : Fin 2 → Fin mPts → F) : Game (Bool × Oracle F) :=
  (interleave (rep A S sch hm C₀ (extraClaims S ptLocal mPts pts))
    (rep A S sch hm C₀ (extraClaims S ptLocal mPts pts))).bind
    fun acc => Game.send (Opens F D) fun _ => Game.ret (acc.1 && acc.2, C₀)

theorem tableOrc_eq (A : Arith F K) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m)
    (ptLocal mPts : ℕ) (C₀ : Oracle F) :
    tableOrc (D := D) A S sch hm ptLocal mPts C₀ =
      Game.coin (Fin 2 → Fin mPts → F) fun pts =>
        tableOrcAfter (D := D) A S sch hm ptLocal mPts C₀ pts :=
  rfl

/-- A default output, for positions outside a batch. -/
def dfltOut [Inhabited D] : OutC F D :=
  ⟨false, fun _ => default, (PendL.reject, PendL.reject), fun _ _ _ => ([], [])⟩

/-- **A batched session after `Commit`** (PROTOCOL.md §5): one link-point draw, then the games
`pre`, the table with level-0 cap `cap₀`, and the games `post`, concurrently. -/
noncomputable def sessAfter (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutC F D))) (cap₀ : Cap D) :
    Game (List (OutC F D)) :=
  Game.coin (Fin 2 → Fin mPts → F) fun pts =>
    batch (pre.map (· pts) ++ tableAfter A H E S sch hm ptLocal mPts cap₀ pts :: post.map (· pts))

open Classical in
/-- The oracle table's output as a compiled output: accepted when both reps accept and the table is
not committed. -/
noncomputable def orcOut (A : Arith F K) (S : Statement) (sch : Schedule) (cap₀ : Cap D)
    (a : Bool × Oracle F) : OutC F D :=
  ⟨a.1 && decide (¬ Committed A S sch a.2), cap₀, (PendL.reject, PendL.reject),
    fun _ _ _ => ([], [])⟩

/-- The oracle session: the table's oracle run in its place. -/
noncomputable def orcSess (A : Arith F K) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m)
    (ptLocal mPts : ℕ) (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutC F D)))
    (C₀ : Oracle F) (cap₀ : Cap D) : Game (List (OutC F D)) :=
  Game.coin (Fin 2 → Fin mPts → F) fun pts =>
    batch (pre.map (· pts) ++
      (tableOrcAfter (D := D) A S sch hm ptLocal mPts C₀ pts).map (orcOut A S sch cap₀) ::
        post.map (· pts))

/-- The session's leaf relation: the table's, at its position. -/
def SessRel [Inhabited D] (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (S : Statement) (sch : Schedule) (C₀ : Oracle F) (cap₀ : Cap D) (n : ℕ)
    (i : IRep F D × IRep F D) (a b : List (OutC F D)) : Prop :=
  ∃ x, a.getD n dfltOut = orcOut A S sch cap₀ x ∧ TopRel H E sch C₀ cap₀ i x (b.getD n dfltOut)

/-- The table's lockstep after the link points (`lockTable` below its coin). -/
noncomputable def lockAfter [Inhabited D] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level)
    (ext : ℕ) (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (C₀ : Oracle F)
    (cap₀ : Cap D) (pts : Fin 2 → Fin mPts → F) :
    Lock (Oracle F) (Cap D) (Lock.getPair (getRep 0) (getRep 1))
      (TopRel H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ C₀ cap₀) ((fun _ => none), (fun _ => none))
      (tableOrcAfter (D := D) A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts C₀ pts)
      (tableAfter A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts cap₀ pts) :=
  Lock.bind (R := Lock.RPair (RecRel 0 C₀ ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
      (RecRel 1 C₀ ⟨m, k0, l₀ :: l₁ :: rest, ext⟩))
    (fun i acc pend h => .send (Opens F D) fun o => .ret (by
      obtain ⟨⟨hv₀, hs₀⟩, ⟨hv₁, hs₁⟩⟩ := h
      refine ⟨rfl, rfl, fun hacc ha₀ ha₁ => ?_, fun hacc => ?_, hs₀, hs₁⟩
      · simp only [Bool.and_eq_true, decide_eq_true_eq] at hacc
        rw [← hv₀ o ha₀, ← hv₁ o ha₁]
        simp [hacc.1.1, hacc.1.2]
      · simp only [Bool.and_eq_true, decide_eq_true_eq] at hacc
        exact hacc.2))
    (Lock.interleaveLock disjoint_reps
      (lockRep A 0 (by norm_num) S m k0 l₀ l₁ rest ext hk0 hm C₀ (extraClaims S ptLocal mPts pts))
      (lockRep A 1 (by norm_num) S m k0 l₀ l₁ rest ext hk0 hm C₀ (extraClaims S ptLocal mPts pts)))

theorem aligned_lockAfter [Inhabited D] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level)
    (ext : ℕ) (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (C₁ C₂ : Oracle F)
    (cap₀ : Cap D) (pts : Fin 2 → Fin mPts → F) :
    Lock.Aligned (lockAfter A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts C₁ cap₀ pts)
      (lockAfter A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts C₂ cap₀ pts) := by
  refine Lock.Aligned.bind ?_ (Lock.Aligned.interleaveLock disjoint_reps
    (aligned_lockRep A 0 (by norm_num) S m k0 l₀ l₁ rest ext hk0 hm C₁ C₂ _)
    (aligned_lockRep A 1 (by norm_num) S m k0 l₀ l₁ rest ext hk0 hm C₁ C₂ _))
  intro _ _ _ _ _ _ _
  exact .send _ fun _ => .ret _ _

/-- **The table in lockstep inside the session**: its lockstep at position `pre.length`, the same
game on both sides everywhere else. -/
noncomputable def lockSess [Inhabited D] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level)
    (ext : ℕ) (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutC F D))) (C₀ : Oracle F) (cap₀ : Cap D) :
    Lock (Oracle F) (Cap D) (Lock.getPair (getRep 0) (getRep 1))
      (SessRel A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ C₀ cap₀ pre.length)
      ((fun _ => none), (fun _ => none))
      (orcSess (D := D) A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post C₀ cap₀)
      (sessAfter A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) :=
  .coin (Fin 2 → Fin mPts → F) fun pts =>
    Lock.mono (fun _ a b h => by simpa only [SessRel, List.length_map] using h)
      (Lock.batchLock dfltOut
        (Lock.mapL (R₂ := fun i o b => ∃ x, o = orcOut A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ x ∧
            TopRel H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ C₀ cap₀ i x b)
          (orcOut A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀) (fun _ a _ hr => ⟨a, rfl, hr⟩)
          (lockAfter A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts C₀ cap₀ pts))
        (post.map (· pts)) (pre.map (· pts)))

theorem aligned_lockSess [Inhabited D] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level)
    (ext : ℕ) (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutC F D))) (C₁ C₂ : Oracle F)
    (cap₀ : Cap D) :
    Lock.Aligned (lockSess A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post C₁ cap₀)
      (lockSess A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post C₂ cap₀) :=
  .coin _ fun pts => Lock.Aligned.mono (Lock.Aligned.batchLock dfltOut
    (Lock.Aligned.mapL (aligned_lockAfter A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts C₁ C₂ cap₀
      pts)) _ _)

/-! ## The oracle side -/

open Classical in
/-- **The oracle session is bounded by the table.** Whatever the other games, the oracle table in
the session accepts uncommitted with probability at most `tableError`, for an optimal prover. -/
theorem orcSess_value_le (A : Arith F K) (hA : A.Correct)
    [Inhabited D] (S : Statement) (sch : Schedule) (hsch : fast100 S.m = some sch) (hm : 13 ≤ S.m)
    (ptLocal mPts : ℕ) (hlay : S.LinkLayout ptLocal mPts)
    (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutC F D))) (C₀ : Oracle F) (cap₀ : Cap D) :
    value (fun outs : List (OutC F D) => (outs.getD pre.length dfltOut).accepted = true)
        (orcSess (D := D) A S sch hm ptLocal mPts pre post C₀ cap₀) ≤
      ENNReal.ofReal (tableError sch (S.shape mPts)) := by
  calc value (fun outs : List (OutC F D) => (outs.getD pre.length dfltOut).accepted = true)
        (orcSess (D := D) A S sch hm ptLocal mPts pre post C₀ cap₀)
      ≤ value (fun o : OutC F D => o.accepted = true) (Game.coin (Fin 2 → Fin mPts → F) fun pts =>
          (tableOrcAfter (D := D) A S sch hm ptLocal mPts C₀ pts).map (orcOut A S sch cap₀)) := by
        simp only [orcSess, value_coin]
        refine Finset.sum_le_sum fun pts _ => mul_le_mul_right ?_ _
        have := value_batch_getD_le (fun o : OutC F D => o.accepted = true) dfltOut
          ((tableOrcAfter (D := D) A S sch hm ptLocal mPts C₀ pts).map (orcOut A S sch cap₀))
          (post.map (· pts)) (pre.map (· pts))
        simpa only [List.length_map] using this
    _ = value (fun a : Bool × Oracle F => a.1 = true ∧ ¬ Committed A S sch a.2)
          (tableOrc (D := D) A S sch hm ptLocal mPts C₀) := by
        rw [tableOrc_eq]
        simp only [value_coin, value_map]
        refine Finset.sum_congr rfl fun pts _ => ?_
        congr 1
        exact value_congr (fun a => by simp [orcOut]) _
    _ ≤ value (fun a : Bool × Oracle F => a.1 = true ∧ ¬ Committed A S sch a.2)
          (Game.send (Oracle F) (tableOrc (D := D) A S sch hm ptLocal mPts)) := by
        rw [value_send]
        exact le_iSup (fun C => value (fun a : Bool × Oracle F => a.1 = true ∧
          ¬ Committed A S sch a.2) (tableOrc (D := D) A S sch hm ptLocal mPts C)) C₀
    _ = value (fun a : Bool × Oracle F => a.1 = true ∧ ¬ Committed A S sch a.2)
          (table A S sch hm ptLocal mPts) := by
        rw [← withDummy_table, Lock.value_withDummy]
    _ ≤ _ := table_value_sound A hA S sch hsch hm ptLocal mPts hlay

/-! ## Rewinding in the session -/

open Classical in
/-- The table `P̃` sends at a rep's cap in the session: the plurality of the table's openings there. -/
noncomputable def plurS [Inhabited D] (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (sch : Schedule) (n : ℕ) : Lock.Chooser (Oracle F) (Cap D) (List (OutC F D)) :=
  fun s d g σ => tabOf (lvOf sch (s / 2)).lanes (lvOf sch (s / 2)).logLen
    (Classical.choose (Rewinding.exists_pluralityE
      (fun outs : List (OutC F D) => opR H E sch (s % 2) (s / 2) d (outs.getD n dfltOut)) g σ))

theorem plurS_plurality [Inhabited D] (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (sch : Schedule) (n r ℓ : ℕ) (hr : r < 2) (d : Cap D) (g : Game (List (OutC F D)))
    (σ : Strategy g) :
    Rewinding.IsPluralityE (fun outs : List (OutC F D) => opR H E sch r ℓ d (outs.getD n dfltOut))
      g σ (rowT (lvOf sch ℓ).lanes (lvOf sch ℓ).logLen (plurS H E sch n (slotOf r ℓ) d g σ)) := by
  have key : ∀ s, Rewinding.IsPluralityE
      (fun outs : List (OutC F D) => opR H E sch (s % 2) (s / 2) d (outs.getD n dfltOut)) g σ
      (rowT (lvOf sch (s / 2)).lanes (lvOf sch (s / 2)).logLen (plurS H E sch n s d g σ)) := by
    intro s
    unfold plurS
    rw [rowT_tabOf]
    exact Classical.choose_spec (Rewinding.exists_pluralityE _ g σ)
  have h1 : slotOf r ℓ % 2 = r := by unfold slotOf; omega
  have h2 : slotOf r ℓ / 2 = ℓ := by unfold slotOf; omega
  have := key (slotOf r ℓ)
  rw [h1, h2] at this
  exact this

/-- **`Adv_{r,ℓ}` in the session**: the collision finder at the table's rep `r`'s level-`ℓ` cap. -/
noncomputable def advRS [Inhabited D] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level)
    (ext : ℕ) (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutC F D))) (cap₀ : Cap D) (r ℓ : ℕ)
    (σ : Strategy (sessAfter A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀)) :
    ℝ :=
  Lock.forkE (plurS H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length) (slotOf r ℓ)
    (fun d b b' => ind (fun x : List (OutC F D) × List (OutC F D) => ∃ p, Rewinding.ConflictAt
      (fun outs : List (OutC F D) =>
        opR H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r ℓ d (outs.getD pre.length dfltOut)) x p) (b, b'))
    (lockSess A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post default cap₀) σ

/-- **`Adv₀` in the session**: two runs after `Commit` open a level-0 position of the table to
different rows. -/
noncomputable def adv₀S [Inhabited D] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutC F D))) (cap₀ : Cap D)
    (σ : Strategy (sessAfter A H E S sch hm ptLocal mPts pre post cap₀)) : ℝ :=
  expect₂ (fun b b' => ind (fun x : List (OutC F D) × List (OutC F D) => ∃ p,
      Rewinding.ConflictAt (fun outs : List (OutC F D) => op₀ H E sch cap₀ (outs.getD pre.length
        dfltOut)) x p) (b, b'))
    (sessAfter A H E S sch hm ptLocal mPts pre post cap₀) σ

open Classical in
/-- **`ε_c⁻` in the session**: `tableError`, the table's rep caps' fork terms and its self-clash. -/
noncomputable def epsS [Inhabited D] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level)
    (ext : ℕ) (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutC F D))) (cap₀ : Cap D)
    (σ : Strategy (sessAfter A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀)) :
    ℝ :=
  tableError ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (S.shape mPts) +
    ∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 (l₀ :: l₁ :: rest).length,
      Real.sqrt (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).logLen *
        (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).queries *
        advRS A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post cap₀ r ℓ σ) +
    expect (ind fun outs : List (OutC F D) =>
        SelfClash H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (outs.getD pre.length dfltOut))
      (sessAfter A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ

/-! ## The compiled bound for any table, in the session -/

open Classical in
/-- `bad_pointwise_acc` for an event of the whole session's outputs that implies the table at
position `n` accepts, such as "the session accepts". -/
theorem bad_pointwise_of [Inhabited D] (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (sch : Schedule)
    (C₀ : Oracle F) (cap₀ : Cap D) (i : IRep F D × IRep F D) (a : Bool × Oracle F) (n : ℕ)
    (Acc : List (OutC F D) → Prop) (hAcc : ∀ outs, Acc outs → (outs.getD n dfltOut).accepted = true)
    (b : List (OutC F D)) (hR : TopRel H E sch C₀ cap₀ i a (b.getD n dfltOut)) :
    ind (fun outs : List (OutC F D) => Acc outs ∧
      ¬ (AgreeRep 0 C₀ i.1 (outs.getD n dfltOut).pend.1.reqs (outs.getD n dfltOut).opens ∧
        AgreeRep 1 C₀ i.2 (outs.getD n dfltOut).pend.2.reqs (outs.getD n dfltOut).opens)) b ≤
    ind (SelfClash H E sch) (b.getD n dfltOut) +
      ind (fun outs : List (OutC F D) => Acc outs ∧
        ∃ p, Rewinding.Off (op₀ H E sch cap₀) (rowT _ _ C₀) (outs.getD n dfltOut) p) b +
      ∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 sch.levels.length,
        ind (fun b => ∃ t d, Lock.getPair (getRep 0) (getRep 1) i (slotOf r ℓ) = some (t, d) ∧
          ∃ p, Rewinding.Off (opR H E sch r ℓ d) (rowT _ _ t) b p) (b.getD n dfltOut) := by
  have h := bad_pointwise_acc H E sch C₀ cap₀ i a (b.getD n dfltOut) hR
  by_cases hA : Acc b
  · have e1 : ind (fun outs : List (OutC F D) => Acc outs ∧
        ¬ (AgreeRep 0 C₀ i.1 (outs.getD n dfltOut).pend.1.reqs (outs.getD n dfltOut).opens ∧
          AgreeRep 1 C₀ i.2 (outs.getD n dfltOut).pend.2.reqs (outs.getD n dfltOut).opens)) b =
        ind (fun out : OutC F D => out.accepted = true ∧ ¬ (AgreeRep 0 C₀ i.1 out.pend.1.reqs
          out.opens ∧ AgreeRep 1 C₀ i.2 out.pend.2.reqs out.opens)) (b.getD n dfltOut) := by
      unfold ind
      simp only [hA, hAcc b hA, true_and]
    have e2 : ind (fun outs : List (OutC F D) => Acc outs ∧
        ∃ p, Rewinding.Off (op₀ H E sch cap₀) (rowT _ _ C₀) (outs.getD n dfltOut) p) b =
        ind (fun out : OutC F D => out.accepted = true ∧
          ∃ p, Rewinding.Off (op₀ H E sch cap₀) (rowT _ _ C₀) out p) (b.getD n dfltOut) := by
      unfold ind
      simp only [hA, hAcc b hA, true_and]
    rw [e1, e2]
    exact h
  · have h0 : ind (fun outs : List (OutC F D) => Acc outs ∧
        ¬ (AgreeRep 0 C₀ i.1 (outs.getD n dfltOut).pend.1.reqs (outs.getD n dfltOut).opens ∧
          AgreeRep 1 C₀ i.2 (outs.getD n dfltOut).pend.2.reqs (outs.getD n dfltOut).opens)) b = 0 := by
      simp [ind, hA]
    rw [h0]
    have h1 := ind_nonneg (SelfClash H E sch) (b.getD n dfltOut)
    have h2 := ind_nonneg (fun outs : List (OutC F D) => Acc outs ∧
      ∃ p, Rewinding.Off (op₀ H E sch cap₀) (rowT _ _ C₀) (outs.getD n dfltOut) p) b
    have h3 : 0 ≤ ∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 sch.levels.length,
        ind (fun b => ∃ t d, Lock.getPair (getRep 0) (getRep 1) i (slotOf r ℓ) = some (t, d) ∧
          ∃ p, Rewinding.Off (opR H E sch r ℓ d) (rowT _ _ t) b p) (b.getD n dfltOut) :=
      Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => ind_nonneg _ _
    linarith

open Classical in
/-- **Compiled soundness for any level-0 table, in a batched session, for an acceptance event.**
For every event `Acc` of the session's outputs that implies the table accepts (the table's own
acceptance, or the session's), every session strategy after `Commit` and every table `T`: `Acc`
holds with `T` uncommitted with probability at most `ε_c⁻` plus the probability that `Acc` holds
in a run that opens a level-0 position off `T`. The coupling's bad event needs only the table's
acceptance (`bad_pointwise_of`). -/
theorem session_sound_of_table_acc (A : Arith F K)
    (hA : A.Correct) [Inhabited D] (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (S : Statement)
    (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ)
    (hsch : fast100 S.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (hlay : S.LinkLayout ptLocal mPts)
    (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutC F D))) (cap₀ : Cap D)
    (Acc : List (OutC F D) → Prop)
    (hAcc : ∀ outs, Acc outs → (outs.getD pre.length dfltOut).accepted = true) (T : Oracle F)
    (σ : Strategy (sessAfter A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀)) :
    expect (ind fun outs : List (OutC F D) => Acc outs ∧
        ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ T)
      (sessAfter A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ ≤
    epsS A H E S m k0 l₀ l₁ rest ext (fast100_shape S.m _ hsch).2.1 hm ptLocal mPts pre post cap₀ σ +
      expect (ind fun outs : List (OutC F D) => Acc outs ∧
          Off₀ H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ T (outs.getD pre.length dfltOut))
        (sessAfter A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ := by
  have hk0 := (fast100_shape S.m _ hsch).2.1
  -- the coupling
  have hcpl := Lock.expect_le_coupled (plurS H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length)
    (fun outs : List (OutC F D) => Acc outs ∧ ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ T)
    (fun outs : List (OutC F D) => (outs.getD pre.length dfltOut).accepted = true)
    (fun (i : IRep F D × IRep F D) (outs : List (OutC F D)) =>
      Acc outs ∧
        ¬ (AgreeRep 0 T i.1 (outs.getD pre.length dfltOut).pend.1.reqs (outs.getD pre.length dfltOut).opens ∧
          AgreeRep 1 T i.2 (outs.getD pre.length dfltOut).pend.2.reqs (outs.getD pre.length dfltOut).opens))
    (fun i a b hR hE hB => by
      obtain ⟨x, hx, ha2, -, hacc, -, -, -⟩ := hR
      rw [hx]
      have h1 : x.1 = true := by
        by_contra hne
        exact hB ⟨hE.1, fun h => hne (hacc (hAcc _ hE.1) h.1 h.2)⟩
      simp only [orcOut, h1, Bool.true_and, decide_eq_true_eq]
      rw [ha2]
      exact hE.2)
    (lockSess A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post T cap₀) σ
  -- the oracle side
  have horc : expect (ind fun outs : List (OutC F D) => (outs.getD pre.length dfltOut).accepted = true)
      (orcSess (D := D) A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post T cap₀) (Lock.oracleStrat (plurS H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length) (lockSess A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post T cap₀) σ) ≤
      tableError ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (S.shape mPts) := by
    rw [← prob_toReal]
    exact ENNReal.toReal_le_of_le_ofReal (tableError_nonneg _ _)
      ((prob_le_value _ _ _).trans (orcSess_value_le A hA S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hsch hm ptLocal mPts hlay
        pre post T cap₀))
  -- the bad runs, cap by cap
  have hsplit := Lock.leafE_mono_rel (plurS H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length)
    (F := fun (i : IRep F D × IRep F D) (outs : List (OutC F D)) => ind (fun outs : List (OutC F D) =>
      Acc outs ∧
        ¬ (AgreeRep 0 T i.1 (outs.getD pre.length dfltOut).pend.1.reqs
            (outs.getD pre.length dfltOut).opens ∧
          AgreeRep 1 T i.2 (outs.getD pre.length dfltOut).pend.2.reqs
            (outs.getD pre.length dfltOut).opens)) outs)
    (fun i a b hR => by
      obtain ⟨x, -, hrel⟩ := hR
      exact bad_pointwise_of H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ T cap₀ i x pre.length Acc hAcc b hrel) (lockSess A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post T cap₀) σ
  rw [Lock.leafE_add, Lock.leafE_add, Lock.leafE_const, Lock.leafE_const] at hsplit
  simp only [Lock.leafE_sum] at hsplit
  have hR : ∀ r ∈ Finset.range 2, ∀ ℓ ∈ Finset.Icc 1 (l₀ :: l₁ :: rest).length,
      Lock.leafE (plurS H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length)
        (fun (i : IRep F D × IRep F D) (outs : List (OutC F D)) => ind (fun b => ∃ t d,
          Lock.getPair (getRep 0) (getRep 1) i (slotOf r ℓ) = some (t, d) ∧
            ∃ p, Rewinding.Off (opR H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r ℓ d) (rowT _ _ t) b p) (outs.getD pre.length dfltOut)) (lockSess A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post T cap₀) σ ≤
      Real.sqrt (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).logLen * (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).queries *
        advRS A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post cap₀ r ℓ σ) := by
    intro r hr ℓ _
    have := Lock.leafE_off_le
      (fun d (outs : List (OutC F D)) => opR H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r ℓ d (outs.getD pre.length dfltOut))
      (rowT (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).lanes (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).logLen) (slotOf r ℓ) (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).queries
      (fun d b => (opOf_count _ _ _ _ _ _ _).trans (slotOpens_length _ _ _ _))
      (plurS H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length)
      (fun d g σ' => plurS_plurality H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length r ℓ (Finset.mem_range.1 hr) d g σ') (lockSess A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post T cap₀) σ
      (getPair_init_none _)
    rw [Fintype.card_fin] at this
    have heq : advRS A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post cap₀ r ℓ σ =
        Lock.forkE (plurS H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length) (slotOf r ℓ)
          (fun d b b' => ind (fun x : List (OutC F D) × List (OutC F D) => ∃ p,
            Rewinding.ConflictAt (fun outs : List (OutC F D) => opR H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r ℓ d
              (outs.getD pre.length dfltOut)) x p) (b, b')) (lockSess A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post T cap₀) σ :=
      Lock.forkE_congr _ _ _ (aligned_lockSess A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre
        post default T cap₀) σ
    rw [heq]
    show Lock.leafE (plurS H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length)
        (fun (i : IRep F D × IRep F D) (b : List (OutC F D)) => ind (fun b => ∃ t d,
          Lock.getPair (getRep 0) (getRep 1) i (slotOf r ℓ) = some (t, d) ∧
            ∃ p, Rewinding.Off (fun outs : List (OutC F D) =>
              opR H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r ℓ d (outs.getD pre.length dfltOut))
              (rowT _ _ t) b p) b)
        (lockSess A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post T cap₀) σ ≤ _
    simpa [Nat.cast_pow] using this
  have hsum := Finset.sum_le_sum fun r hr => Finset.sum_le_sum fun ℓ hℓ => hR r hr ℓ hℓ
  have e0 : expect (ind fun outs : List (OutC F D) => Acc outs ∧
        Off₀ H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ T (outs.getD pre.length dfltOut))
      (sessAfter A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ =
    expect (fun outs : List (OutC F D) =>
        ind (fun outs : List (OutC F D) => Acc outs ∧ ∃ p, Rewinding.Off (op₀ H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀) (rowT _ _ T) (outs.getD pre.length dfltOut) p) outs)
      (sessAfter A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ := rfl
  have eS : expect (ind fun outs : List (OutC F D) => SelfClash H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (outs.getD pre.length dfltOut))
      (sessAfter A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ =
    expect (fun outs : List (OutC F D) => ind (SelfClash H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (outs.getD pre.length dfltOut))
      (sessAfter A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ := rfl
  unfold epsS
  rw [e0, eS]
  linarith

open Classical in
/-- **Compiled soundness for any level-0 table, in a batched session.** For every session strategy
after `Commit` and every table `T`: the table accepts with `T` uncommitted with probability at most
`ε_c⁻` plus the probability that a run opens a level-0 position of the table off `T`. -/
theorem session_sound_of_table (A : Arith F K) (hA : A.Correct)
    [Inhabited D] (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (S : Statement)
    (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ)
    (hsch : fast100 S.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (hlay : S.LinkLayout ptLocal mPts)
    (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutC F D))) (cap₀ : Cap D) (T : Oracle F)
    (σ : Strategy (sessAfter A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀)) :
    expect (ind fun outs : List (OutC F D) => (outs.getD pre.length dfltOut).accepted = true ∧
        ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ T)
      (sessAfter A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ ≤
    epsS A H E S m k0 l₀ l₁ rest ext (fast100_shape S.m _ hsch).2.1 hm ptLocal mPts pre post cap₀ σ +
      expect (ind fun outs : List (OutC F D) =>
          Off₀ H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ T (outs.getD pre.length dfltOut))
        (sessAfter A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ :=
  (session_sound_of_table_acc A hA H E S m k0 l₀ l₁ rest ext hsch hm ptLocal mPts hlay pre post
    cap₀ (fun outs => (outs.getD pre.length dfltOut).accepted = true) (fun _ h => h) T σ).trans
    (add_le_add le_rfl (expect_mono (fun outs => ind_mono (fun _ (h : _ ∧ _) => h.2) outs) _ _))

/-! ## Knowledge soundness in the session -/

/-- **The extractor's table in the session**: at each level-0 position of the table, the first
verifying opening one of the runs `bs` after `Commit` made there, and zero elsewhere. -/
noncomputable def extractTableS [Inhabited D] (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (sch : Schedule) (n : ℕ) (cap₀ : Cap D) (bs : List (List (OutC F D))) : Oracle F :=
  tabOf (lvOf sch 0).lanes (lvOf sch 0).logLen
    (Extract.firstTable (fun outs : List (OutC F D) => op₀ H E sch cap₀ (outs.getD n dfltOut))
      (fun _ _ => 0) bs)

theorem epsS_nonneg [Inhabited D] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level)
    (ext : ℕ) (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutC F D))) (cap₀ : Cap D)
    (σ : Strategy (sessAfter A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀)) :
    0 ≤ epsS A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post cap₀ σ := by
  unfold epsS
  have h1 := tableError_nonneg ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (S.shape mPts)
  have h2 := expect_nonneg (ind_nonneg fun outs : List (OutC F D) =>
      SelfClash H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (outs.getD pre.length dfltOut))
    (sessAfter A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ
  have h3 : 0 ≤ ∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 (l₀ :: l₁ :: rest).length,
      Real.sqrt (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).logLen *
        (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).queries *
        advRS A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post cap₀ r ℓ σ) :=
    Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => Real.sqrt_nonneg _
  linarith

open Classical in
/-- **Knowledge soundness of a table in a batched session, the audit's form.** The probability that
the table accepts in a fresh run of the session after `Commit`, while the extraction from `K ≥ 1`
independent reruns of everything after `Commit` fails, is at most `ε_c⁻ + K·adv₀ + N₀/(eK)`, with
no condition on the acceptance probability. -/
theorem session_knowledge_sound (A : Arith F K) (hA : A.Correct)
    [Inhabited D] (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (S : Statement)
    (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ)
    (hsch : fast100 S.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (hlay : S.LinkLayout ptLocal mPts)
    (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutC F D))) (cap₀ : Cap D)
    (σ : Strategy (sessAfter A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀))
    (Kr : ℕ) (hK : 1 ≤ Kr) :
    expectN (sessAfter A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ Kr
        (fun bs => expect (ind fun outs : List (OutC F D) =>
            (outs.getD pre.length dfltOut).accepted = true ∧
              ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
                (extractTableS H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length cap₀ bs))
          (sessAfter A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ) ≤
      epsS A H E S m k0 l₀ l₁ rest ext (fast100_shape S.m _ hsch).2.1 hm ptLocal mPts pre post cap₀ σ +
        (Kr * adv₀S A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀ σ +
          2 ^ l₀.logLen / (Real.exp 1 * Kr)) := by
  set ε := expect (ind fun outs : List (OutC F D) => (outs.getD pre.length dfltOut).accepted = true) (sessAfter A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ
    with hεdef
  -- a fresh run is independent of the extractor's runs: the joint probability is a product
  have hprod : ∀ bs : List (List (OutC F D)),
      expect (ind fun outs : List (OutC F D) => (outs.getD pre.length dfltOut).accepted = true ∧
          ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (extractTableS H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length cap₀ bs)) (sessAfter A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ =
        ε * ind (fun bs => ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (extractTableS H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length cap₀ bs)) bs := by
    intro bs
    by_cases hc : Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (extractTableS H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length cap₀ bs)
    · have : (ind fun outs : List (OutC F D) => (outs.getD pre.length dfltOut).accepted = true ∧
          ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (extractTableS H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length cap₀ bs)) = fun _ => 0 := by
        funext outs; simp [ind, hc]
      rw [this, expect_const]
      simp [ind, hc]
    · have : (ind fun outs : List (OutC F D) => (outs.getD pre.length dfltOut).accepted = true ∧
          ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (extractTableS H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length cap₀ bs)) =
          ind fun outs : List (OutC F D) => (outs.getD pre.length dfltOut).accepted = true := by
        funext outs
        unfold ind
        by_cases h : (outs.getD pre.length dfltOut).accepted = true
        · rw [if_pos ⟨h, hc⟩, if_pos h]
        · rw [if_neg (fun h' => h h'.1), if_neg h]
      rw [this, show ind (fun bs => ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (extractTableS H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length cap₀ bs)) bs = 1
        from if_pos hc, mul_one]
  rw [show (fun bs : List (List (OutC F D)) => expect (ind fun outs : List (OutC F D) =>
      (outs.getD pre.length dfltOut).accepted = true ∧ ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (extractTableS H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length cap₀ bs))
      (sessAfter A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ) = fun bs => ε * ind (fun bs => ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (extractTableS H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length cap₀ bs)) bs
    from funext hprod, expectN_const_mul]
  -- the fixed-table fact: an uncommitted extraction costs off-table mass
  have hcost : ∀ bs : List (List (OutC F D)),
      ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (extractTableS H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length cap₀ bs) →
      ε ≤ epsS A H E S m k0 l₀ l₁ rest ext (fast100_shape S.m _ hsch).2.1 hm ptLocal mPts pre post
          cap₀ σ +
        expect (ind fun ω => ∃ p, Rewinding.Off (fun outs : List (OutC F D) => op₀ H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (outs.getD pre.length dfltOut)) (Extract.firstTable (fun outs : List (OutC F D) => op₀ H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (outs.getD pre.length dfltOut)) (fun _ _ => 0) bs) ω p) (sessAfter A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ := by
    intro bs hbs
    have h := session_sound_of_table A hA H E S m k0 l₀ l₁ rest ext hsch hm ptLocal mPts hlay
      pre post cap₀ (extractTableS H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length cap₀ bs) σ
    have e1 : (ind fun outs : List (OutC F D) => (outs.getD pre.length dfltOut).accepted = true ∧
        ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (extractTableS H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length cap₀ bs)) =
        ind fun outs : List (OutC F D) => (outs.getD pre.length dfltOut).accepted = true := by
      funext outs
      unfold ind
      by_cases h : (outs.getD pre.length dfltOut).accepted = true
      · rw [if_pos ⟨h, hbs⟩, if_pos h]
      · rw [if_neg (fun h' => h h'.1), if_neg h]
    have e2 : (ind fun outs : List (OutC F D) =>
        Off₀ H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (extractTableS H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length cap₀ bs) (outs.getD pre.length dfltOut)) =
        ind fun ω => ∃ p, Rewinding.Off (fun outs : List (OutC F D) => op₀ H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (outs.getD pre.length dfltOut)) (Extract.firstTable (fun outs : List (OutC F D) => op₀ H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (outs.getD pre.length dfltOut)) (fun _ _ => 0) bs) ω p := by
      funext outs
      unfold ind Off₀ extractTableS
      rw [rowT_tabOf]
    rw [e1, e2] at h
    exact h
  have hjoint := Extract.joint_of_fixed (sessAfter A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ
    (fun bs => expect (ind fun ω => ∃ p, Rewinding.Off (fun outs : List (OutC F D) => op₀ H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (outs.getD pre.length dfltOut)) (Extract.firstTable (fun outs : List (OutC F D) => op₀ H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (outs.getD pre.length dfltOut)) (fun _ _ => 0) bs)
      ω p) (sessAfter A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ)
    (fun bs => Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (extractTableS H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length cap₀ bs))
    (fun bs => expect_nonneg (ind_nonneg _) _ _) ε _
    (epsS_nonneg A H E S m k0 l₀ l₁ rest ext _ hm ptLocal mPts pre post cap₀ σ) hcost Kr
  have hmass := Extract.expect_off_firstTable_le' (fun outs : List (OutC F D) => op₀ H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (outs.getD pre.length dfltOut)) (fun _ _ => 0) (sessAfter A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ Kr hK
  rw [Fintype.card_fin] at hmass
  have hmass' : expectN (sessAfter A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ Kr (fun bs => expect (ind fun ω => ∃ p,
        Rewinding.Off (fun outs : List (OutC F D) => op₀ H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (outs.getD pre.length dfltOut)) (Extract.firstTable (fun outs : List (OutC F D) => op₀ H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (outs.getD pre.length dfltOut)) (fun _ _ => 0) bs) ω p) (sessAfter A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ) ≤
      Kr * adv₀S A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀ σ + 2 ^ l₀.logLen / (Real.exp 1 * Kr) := by
    convert hmass using 2
    all_goals first
      | exact rfl
      | simp only [lvOf, List.getD_cons_zero, Nat.cast_pow, Nat.cast_ofNat]
  linarith

end FlockSoundness
