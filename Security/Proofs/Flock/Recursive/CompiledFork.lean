import Proofs.Flock.Recursive.Target
import Proofs.Flock.Soundness.Assumptions.Recursive

/-!
# Recursion: the compiled inner protocol's forks

What the compile lemma (`flock_inner_sound_compiled`, `Compile`) reads of two runs of C-Flock's compiled inner protocol
(`flockInnerCL`), and the extraction it is proved from.

* `ClashCL`, `clashPickCL`: two runs of the compiled table that clash (either run self-clashes, `SelfClashL`, or the two
  open a position of the level-0 cap or of a rep's recorded cap to different rows), and the inner fork's output: on two
  inner transcripts whose steps play the compiled table to outcomes that clash, a collision of `H` (`collOf`), else
  nothing.
* `compiled_fork_or_value` (open): on every hidden template, for every `G` and every prover of `flockInnerCL`, a fork of
  the prover's own runs on two transcripts satisfying `G` that clash, or the compiled table accepts a transcript
  satisfying `G` no more often than the interactive table's `value`, which `table_value_sound` bounds.
-/

namespace FlockSoundness.Discharge.Recursion

section Compiled

open Game Model Accounting Exec StrictCR
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {D Sl : Type}

/-- **Two runs of the compiled table clash**: either run's outcome self-clashes (`SelfClashL`), or the two open a
position of the first run's level-0 cap, or of a rep's level-`ℓ` cap the first run records, to different rows. -/
def ClashCL (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (sch : Schedule)
    (o o' : OutCL F D Sl) : Prop :=
  SelfClashL H E Lf sch o ∨ SelfClashL H E Lf sch o' ∨
    (∃ p, Rewinding.ConflictAt (op₀L H E Lf sch o.cap₀) (o, o') p) ∨
    ∃ i < 2, ∃ ℓ, 1 ≤ ℓ ∧ ∃ cap, (if i = 0 then o.pend.1 else o.pend.2).caps[ℓ - 1]? = some cap ∧
      ∃ p, Rewinding.ConflictAt (opRL H E Lf sch i ℓ cap) (o, o') p

omit [Fintype F] [DecidableEq F] in
/-- **A clash is a collision.** -/
theorem ClashCL.collision {H : List UInt8 → D} {E : Merkle.Enc (List F) D} {Lf : Merkle.Leaf (List F) Sl D H}
    {sch : Schedule} {o o' : OutCL F D Sl} (h : ClashCL H E Lf sch o o') : Merkle.Collision H := by
  rcases h with h | h | ⟨_, h⟩ | ⟨_, _, _, _, _, _, _, h⟩
  · exact h.collision
  · exact h.collision
  · exact op₀L_conflict_collision h
  · exact opRL_conflict_collision h

variable {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]

open Classical in
/-- **The compiled table's outcome on an inner transcript**: the outcome the transcript's steps on the schedule play
`Model.tableCL` to on the hidden statement `M.toStatement regs`, when they play it to one. -/
noncomputable def outCL (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (cls : HClass) (regs : List (Region cls.kLog (cls.m - cls.kLog)))
    (sch : Schedule) (M : Hidden cls) (t : List (List (SMsg T) × ((k : Fin r) → CT k))) : Option (OutCL F D Sl) :=
  if h : ∃ out, PlaysT (tableCL A H E Lf (M.toStatement regs) sch cls.hm cls.ptLocal cls.mPts)
      (schedSteps T CT 0 t) out then some h.choose else none

/-- **The inner fork's output at the compiled table**: on two inner transcripts whose steps play the compiled table to
outcomes that clash (`ClashCL`), a collision of `H` (`collOf`), else nothing. -/
noncomputable def clashPickCL (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (cls : HClass) (regs : List (Region cls.kLog (cls.m - cls.kLog)))
    (sch : Schedule) (M : Hidden cls) (t t' : List (List (SMsg T) × ((k : Fin r) → CT k))) : Pair :=
  collOf H (∃ o o', outCL T CT A H E Lf cls regs sch M t = some o ∧ outCL T CT A H E Lf cls regs sch M t' = some o' ∧
      ClashCL H E Lf sch o o')
    fun ⟨_, _, _, _, hc⟩ => hc.collision

/-- **The compiled table's runs satisfying `G` are a run of the interactive table, or two of them clash.** On every
hidden template `M` (true or false), for every `G` and every prover `s` of `flockInnerCL`: two continuations of `s` after
some round's message, both of whose transcripts satisfy `G`, play the compiled table to outcomes that clash
(`clashPickCL`, a collision of `H`), or the compiled table accepts a transcript satisfying `G` with probability at most the
interactive table's `value` (`Model.table`: the oracles sent whole, the game drawing its own coins).

Its proof is open (about 500–800 lines), in two parts:
* the extraction (about 300–500): with no such fork, at every node of `s` where a round's message carries a commitment
  (the level-0 cap, a rep's level cap), every two runs through it that satisfy `G` open each position of it to the same
  row (else they, or the run that opens it twice, are a `ClashCL`: `op₀L_conflict_collision`,
  `opRL_conflict_collision`, `SelfClashL.collision`), so the rows the runs satisfying `G` open, and a default elsewhere,
  are an oracle; the interactive table's prover that sends those oracles whole, and `s`'s other messages, accepts at
  every coin at which a run satisfying `G` accepts, its queries reading the rows the compiled run opens. This is
  `table_sound_compiledL`'s plurality extraction (`CompiledL`) with every off-plurality opening a clash of two runs
  satisfying `G` in place of its square-root terms; `G` is a predicate on the schedule's transcripts, not on
  `OutCL`, so the restriction is made at each node of `schedInner`.
* the schedule's simulation (about 200–300): `leafAvg_plays_le` (`sched_sound`) for that prover, with the coin
  coordinates a round does not read, and the rounds whose messages are ill-typed, averaged out first (a mixture of
  deterministic provers of `Model.table`, each at most its `value`, `prob_le_value`). -/
theorem compiled_fork_or_value (A : Arith F K) [Inhabited D] [Inhabited Sl] (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (cls : HClass)
    (regs : List (Region cls.kLog (cls.m - cls.kLog))) (sch : Schedule) (hsch : fast100 cls.m = some sch)
    (M : Hidden cls)
    (G : List ((flockInnerCL T CT A H E Lf cls regs sch).Msg × (flockInnerCL T CT A H E Lf cls regs sch).Coin) → Prop)
    (s : Strategy (flockInnerCL T CT A H E Lf cls regs sch).game) :
    (flockInnerCL T CT A H E Lf cls regs sch).Fork H (pickOn G (clashPickCL T CT A H E Lf cls regs sch M)) s ∨
      prob (fun t => (flockInnerCL T CT A H E Lf cls regs sch).accepts M t ∧ G t)
          (flockInnerCL T CT A H E Lf cls regs sch).game s ≤
        value (fun out => out.1 = true) (table A (M.toStatement regs) sch cls.hm cls.ptLocal cls.mPts) := by
  sorry

end Compiled

end FlockSoundness.Discharge.Recursion
