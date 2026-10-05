import FlockSoundness.Protocol.Discharge.Exec.CompiledL
import FlockSoundness.Protocol.Discharge.Exec.SessionL

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

/-! ## The oracle side -/

/-! ## Rewinding in the session -/

open Classical in
/-- The table `P̃` sends at a rep's cap in the session: the plurality of the table's openings there. -/
noncomputable def plurSL [Inhabited D] [Inhabited Sl] (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (sch : Schedule) (n : ℕ) : Lock.Chooser (Oracle F) (Cap D) (List (OutCL F D Sl)) :=
  fun s d g σ => tabOf (lvOf sch (s / 2)).lanes (lvOf sch (s / 2)).logLen
    (Classical.choose (Rewinding.exists_pluralityE
      (fun outs : List (OutCL F D Sl) => opRL H E Lf sch (s % 2) (s / 2) d (outs.getD n dfltOutL)) g σ))

/-! ## The compiled bound for any table, in the session -/

/-! ## Knowledge soundness in the session -/

/-- **The extractor's table in the session**: at each level-0 position of the table, the first
verifying opening one of the runs `bs` after `Commit` made there, and zero elsewhere. -/
noncomputable def extractTableSL [Inhabited D] [Inhabited Sl] (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (sch : Schedule) (n : ℕ) (cap₀ : Cap D) (bs : List (List (OutCL F D Sl))) : Oracle F :=
  tabOf (lvOf sch 0).lanes (lvOf sch 0).logLen
    (Extract.firstTable (fun outs : List (OutCL F D Sl) => op₀L H E Lf sch cap₀ (outs.getD n dfltOutL))
      (fun _ _ => 0) bs)

end FlockSoundness.Discharge.Exec
