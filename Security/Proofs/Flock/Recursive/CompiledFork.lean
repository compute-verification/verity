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
* `Inner.fork_of_outcomes`: the fork at round `0` is two independent runs of the prover from the root
  (`forkG_zero_outcome`), so any two transcripts its runs reach whose `pick` is a collision are a fork;
  `fork_of_clashCL` at the compiled table.
* `compiled_clash_or_value` (open): on every hidden template, for every `G` and every prover of `flockInnerCL` with at
  least one round, two transcripts its runs reach that satisfy `G` and clash, or the compiled table accepts a transcript
  satisfying `G` no more often than the interactive table's `value`, which `table_value_sound` bounds.
* `compiled_fork_or_value`: the same with a fork in place of the two transcripts, at every number of rounds, what the
  compile lemma reads.
-/

namespace FlockSoundness.Game

variable {α β : Type}

/-- **A run of `g`, then of `f a` at the outcome `a` it reached, reaches what the second run reaches.** -/
theorem outcome_bind {f : α → Game β} {t : (a : α) → Strategy (f a)} {c : β} :
    ∀ (g : Game α) (s : Strategy g) (a : α), Outcome g s a → Outcome (f a) (t a) c →
      Outcome (g.bind f) (Strategy.bind g f s t) c
  | .ret b, _, a, h, hc => by
      have h' : b = a := h
      subst h'
      exact hc
  | .send _ k, s, a, h, hc => outcome_bind (k s.1) s.2 a h hc
  | @Game.coin _ _ _ _ k, s, a, h, hc => by
      obtain ⟨x, h⟩ := h
      exact ⟨x, outcome_bind (k x) (s x) a h hc⟩

theorem outcome_map (φ : α → β) (g : Game α) (s : Strategy g) (a : α) (h : Outcome g s a) :
    Outcome (g.map φ) (Strategy.map φ g s) (φ a) :=
  outcome_bind (f := fun a => .ret (φ a)) (t := fun _ => PUnit.unit) g s a h rfl

end FlockSoundness.Game

namespace FlockSoundness.StrictCR

open Game

/-- **Two independent runs reach every pair of outcomes of one run.** -/
theorem twoRuns_outcome {β γ : Type} {g : Game β} {s : Strategy g} (pick : β → β → γ) {b b' : β}
    (h : Outcome g s b) (h' : Outcome g s b') : Outcome (twoRuns g s pick).g (twoRuns g s pick).s (pick b b') :=
  outcome_bind (f := fun b => g.map (pick b)) (t := fun b => Strategy.map (pick b) g s) g s b h
    (outcome_map (pick b) g s b' h')

end FlockSoundness.StrictCR

namespace FlockSoundness.Discharge.Recursion

open Game

section Runs

variable {M C β γ : Type} [Fintype C] [Nonempty C]

/-- **The rest of the run from the root reaches each leaf's outcomes** (`contF`): an outcome of the rounds is one of
the continuation `f t` at a leaf `t`, and whatever the trial `T` reaches at that leaf the rest of the run reaches. -/
theorem outcome_contF : ∀ (r : ℕ) (f : List (M × C) → Game β)
    (T : (t : List (M × C)) → Strategy (f t) → StrictCR.Finder γ) (s : Strategy (rounds M C r f)) (b : β),
    Outcome (rounds M C r f) s b → ∃ (t : List (M × C)) (s' : Strategy (f t)), Outcome (f t) s' b ∧
      ∀ c, Outcome (T t s').g (T t s').s c → Outcome (contF r f T s).g (contF r f T s).s c
  | 0, _, _, s, _, h => ⟨[], s, h, fun _ hc => hc⟩
  | r + 1, f, T, s, b, h => by
      obtain ⟨ρ, h⟩ := h
      obtain ⟨t, s', hb, hc⟩ :=
        outcome_contF r (fun t => f ((s.1, ρ) :: t)) (fun t => T ((s.1, ρ) :: t)) (s.2 ρ) b h
      exact ⟨(s.1, ρ) :: t, s', hb, fun c h' => ⟨ρ, hc c h'⟩⟩

/-- `outcome_contF` for the inner game: the rest of the run that returns the transcript reaches every transcript the
prover's runs reach. -/
theorem outcome_contF_ret (r : ℕ) (s : Strategy (rounds M C r .ret)) (t : List (M × C))
    (h : Outcome (rounds M C r .ret) s t) :
    Outcome (contF r .ret (fun t _ => (⟨.ret t, PUnit.unit⟩ : StrictCR.Finder (List (M × C)))) s).g
      (contF r .ret (fun t _ => (⟨.ret t, PUnit.unit⟩ : StrictCR.Finder (List (M × C)))) s).s t := by
  obtain ⟨_, _, hb, hc⟩ :=
    outcome_contF r .ret (fun t _ => (⟨.ret t, PUnit.unit⟩ : StrictCR.Finder (List (M × C)))) s t h
  exact hc t hb

/-- **The fork at round `0` is two independent runs of the prover from the root**, so it reaches `pick` of every two
transcripts the prover's runs reach. -/
theorem forkG_zero_outcome (pick : List (M × C) → List (M × C) → StrictCR.Pair) {r : ℕ}
    (s : Strategy (rounds M C r .ret)) (hr : 0 < r) {t t' : List (M × C)}
    (ht : Outcome (rounds M C r .ret) s t) (ht' : Outcome (rounds M C r .ret) s t') :
    Outcome (forkG pick 0 r .ret (fun t _ => ⟨.ret t, PUnit.unit⟩) s).g
      (forkG pick 0 r .ret (fun t _ => ⟨.ret t, PUnit.unit⟩) s).s (pick t t') := by
  obtain ⟨n, rfl⟩ : ∃ n, r = n + 1 := ⟨r - 1, by omega⟩
  exact StrictCR.twoRuns_outcome pick (outcome_contF_ret (n + 1) s t ht) (outcome_contF_ret (n + 1) s t' ht')

/-- **Two runs of an inner prover are a fork** (`Inner.Fork`, at round `0`) when `pick` of their transcripts is a
collision. -/
theorem Inner.fork_of_outcomes {D : Type} {H : List UInt8 → D} {I : Inner}
    {pick : List (I.Msg × I.Coin) → List (I.Msg × I.Coin) → StrictCR.Pair} {s : Strategy I.game} (hr : 0 < I.r)
    {t t' : List (I.Msg × I.Coin)} (ht : Outcome I.game s t) (ht' : Outcome I.game s t')
    (hc : StrictCR.IsColl H (pick t t')) : I.Fork H pick s :=
  ⟨0, hr, pick t t', forkG_zero_outcome pick s hr ht ht', hc⟩

/-- With no round, the inner game's one run is the empty transcript. -/
theorem prob_rounds_zero (p : List (M × C) → Prop) (s : Strategy (rounds M C 0 .ret)) (h : ¬ p []) :
    prob p (rounds M C 0 .ret) s = 0 := by
  show prob p (.ret []) PUnit.unit = 0
  simp [prob, h]

end Runs

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

/-- **Two runs of the compiled inner prover satisfying `G` that clash are a fork** (`Inner.fork_of_outcomes`, at round
`0`), which `pickOn G (clashPickCL …)` reads as a collision of `H`. -/
theorem fork_of_clashCL (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (cls : HClass) (regs : List (Region cls.kLog (cls.m - cls.kLog)))
    (sch : Schedule) (M : Hidden cls)
    (G : List ((flockInnerCL T CT A H E Lf cls regs sch).Msg × (flockInnerCL T CT A H E Lf cls regs sch).Coin) → Prop)
    (s : Strategy (flockInnerCL T CT A H E Lf cls regs sch).game) (hr : 0 < r)
    {t t' : List ((flockInnerCL T CT A H E Lf cls regs sch).Msg × (flockInnerCL T CT A H E Lf cls regs sch).Coin)}
    (ht : Game.Outcome (flockInnerCL T CT A H E Lf cls regs sch).game s t)
    (ht' : Game.Outcome (flockInnerCL T CT A H E Lf cls regs sch).game s t') (hG : G t) (hG' : G t')
    {o o' : OutCL F D Sl} (ho : outCL T CT A H E Lf cls regs sch M t = some o)
    (ho' : outCL T CT A H E Lf cls regs sch M t' = some o') (hc : ClashCL H E Lf sch o o') :
    (flockInnerCL T CT A H E Lf cls regs sch).Fork H (pickOn G (clashPickCL T CT A H E Lf cls regs sch M)) s := by
  refine Inner.fork_of_outcomes (I := flockInnerCL T CT A H E Lf cls regs sch) hr ht ht' ?_
  simp only [pickOn, hG, hG', and_self, ↓reduceIte]
  exact collOf_isColl H _ ⟨o, o', ho, ho', hc⟩

/-- **The compiled table's runs satisfying `G` are a run of the interactive table, or two of them clash.** On every
hidden template `M` (true or false), for every `G` and every prover `s` of `flockInnerCL` with at least one round: two
transcripts the runs of `s` reach, both satisfying `G`, on which the compiled table's outcomes clash (`ClashCL`), or the
compiled table accepts a transcript satisfying `G` with probability at most the interactive table's `value`
(`Model.table`: the oracles sent whole, the game drawing its own coins).

Its proof is open (about 500–800 lines), in two parts:
* the extraction (about 300–500): with no such pair, at every node of `s` where a round's message carries a commitment
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
theorem compiled_clash_or_value (A : Arith F K) [Inhabited D] [Inhabited Sl] (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (cls : HClass)
    (regs : List (Region cls.kLog (cls.m - cls.kLog))) (sch : Schedule) (hsch : fast100 cls.m = some sch)
    (M : Hidden cls)
    (G : List ((flockInnerCL T CT A H E Lf cls regs sch).Msg × (flockInnerCL T CT A H E Lf cls regs sch).Coin) → Prop)
    (s : Strategy (flockInnerCL T CT A H E Lf cls regs sch).game) (hr : 0 < r) :
    (∃ t t', Game.Outcome (flockInnerCL T CT A H E Lf cls regs sch).game s t ∧
        Game.Outcome (flockInnerCL T CT A H E Lf cls regs sch).game s t' ∧ G t ∧ G t' ∧
        ∃ o o', outCL T CT A H E Lf cls regs sch M t = some o ∧ outCL T CT A H E Lf cls regs sch M t' = some o' ∧
          ClashCL H E Lf sch o o') ∨
      prob (fun t => (flockInnerCL T CT A H E Lf cls regs sch).accepts M t ∧ G t)
          (flockInnerCL T CT A H E Lf cls regs sch).game s ≤
        value (fun out => out.1 = true) (table A (M.toStatement regs) sch cls.hm cls.ptLocal cls.mPts) := by
  sorry

/-- **The compiled table's runs satisfying `G` are a run of the interactive table, or a fork of them clashes.** On every
hidden template `M`, for every `G` and every prover `s` of `flockInnerCL`: a fork of `s` (two continuations after some
round's message, both of whose transcripts satisfy `G`) that `clashPickCL` reads as a collision of `H`, or the compiled
table accepts a transcript satisfying `G` with probability at most the interactive table's `value`. With no round the
compiled table accepts nothing (`flockInnerCL_accepts`); otherwise from `compiled_clash_or_value` and
`fork_of_clashCL`. -/
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
  rcases Nat.eq_zero_or_pos r with rfl | hr
  · refine Or.inr (le_of_eq_of_le (prob_rounds_zero _ s fun hp => ?_) zero_le)
    obtain ⟨hp, -⟩ := hp
    obtain ⟨_, _, _, _, h, _⟩ := (flockInnerCL_accepts T CT A H E Lf cls regs sch M []).1 hp
    simp [schedSteps] at h
  · rcases compiled_clash_or_value T CT A H E Lf cls regs sch hsch M G s hr with
      ⟨t, t', ht, ht', hG, hG', o, o', ho, ho', hc⟩ | h
    · exact Or.inl (fork_of_clashCL T CT A H E Lf cls regs sch M G s hr ht ht' hG hG' ho ho' hc)
    · exact Or.inr h

end Compiled

end FlockSoundness.Discharge.Recursion
