import FlockSoundness.Model.Compiled
import FlockSoundness.Game.Lock
import FlockSoundness.LigeritoMid
import FlockSoundness.Protocol.CompiledLock

/-!
# The compiled Flock verifier runs in lockstep with the oracle verifier

`Game.Lock` between `Model.table` (with a dummy last message) and `Model.tableC`: the same coins
and messages, caps for tables. Along the lockstep, the compiled claim evaluated at any openings
that agree with the fixed tables at every requested position equals the oracle claim
(`StRel`), so at the end the compiled verdict equals the oracle verdict on such openings.

Slots: `0` is the level-0 cap; rep `r`'s level-`ℓ` cap (`ℓ ≥ 1`) is `slotOf r ℓ = 2ℓ + r`.
-/

namespace FlockSoundness

open Game Model Accounting

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
variable {D : Type}

/-! ## The recursive rounds -/

/-! ## Ligerito, a rep, a table -/

/-- The oracle table's continuation after the level-0 table `C₀`, with the dummy last message. -/
def tableOrc (A : Arith F K) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (C₀ : Oracle F) : Game (Bool × Oracle F) :=
  Game.coin (Fin 2 → Fin mPts → F) fun pts =>
    (interleave (rep A S sch hm C₀ (extraClaims S ptLocal mPts pts))
      (rep A S sch hm C₀ (extraClaims S ptLocal mPts pts))).bind
      fun acc => Game.send (Opens F D) fun _ => Game.ret (acc.1 && acc.2, C₀)

theorem withDummy_table (A : Arith F K) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m)
    (ptLocal mPts : ℕ) :
    Lock.withDummy (Opens F D) (table A S sch hm ptLocal mPts) =
      Game.send (Oracle F) (tableOrc (D := D) A S sch hm ptLocal mPts) := by
  refine congrArg (Game.send (Oracle F)) (funext fun C₀ =>
    congrArg (@Game.coin _ (Fin 2 → Fin mPts → F) _ _) (funext fun pts => ?_))
  simp only [Game.bind_ret, Game.bind_eq, Game.pure_eq, Game.bind_assoc]

open Classical in
/-- The compiled table's continuation after the level-0 cap. -/
noncomputable def tableCmp (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (cap₀ : Cap D) :
    Game (OutC F D) :=
  Game.coin (Fin 2 → Fin mPts → F) fun pts =>
    (interleave (repC A 0 S sch hm (extraClaims S ptLocal mPts pts))
      (repC A 1 S sch hm (extraClaims S ptLocal mPts pts))).bind
      fun pend => Game.send (Opens F D) fun o => Game.ret
        ⟨pend.1.verdict o && pend.2.verdict o &&
          decide (OpensOK H E cap₀ 0 pend.1 o ∧ OpensOK H E cap₀ 1 pend.2 o), cap₀, pend, o⟩

theorem tableC_eq (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) :
    tableC A H E S sch hm ptLocal mPts = Game.send (Cap D) (tableCmp A H E S sch hm ptLocal mPts) :=
  rfl

/-- **The table's leaf relation** below the level-0 cap. -/
def TopRel (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (sch : Schedule) (C₀ : Oracle F)
    (cap₀ : Cap D) (i : IRep F D × IRep F D) (a : Bool × Oracle F) (out : OutC F D) : Prop :=
  a.2 = C₀ ∧ out.cap₀ = cap₀ ∧
    (out.accepted = true → AgreeRep 0 C₀ i.1 out.pend.1.reqs out.opens →
      AgreeRep 1 C₀ i.2 out.pend.2.reqs out.opens → a.1 = true) ∧
    (out.accepted = true →
      OpensOK H E cap₀ 0 out.pend.1 out.opens ∧ OpensOK H E cap₀ 1 out.pend.2 out.opens) ∧
    RecStruct sch i.1 out.pend.1 ∧ RecStruct sch i.2 out.pend.2

/-- **The table in lockstep** below the level-0 cap: the link points, both reps interleaved, the
openings. -/
noncomputable def lockTable [Inhabited D] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level)
    (ext : ℕ) (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (C₀ : Oracle F)
    (cap₀ : Cap D) :
    Lock (Oracle F) (Cap D) (Lock.getPair (getRep 0) (getRep 1))
      (TopRel H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ C₀ cap₀) ((fun _ => none), (fun _ => none))
      (tableOrc (D := D) A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts C₀)
      (tableCmp A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts cap₀) :=
  .coin (Fin 2 → Fin mPts → F) fun pts =>
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

end FlockSoundness
