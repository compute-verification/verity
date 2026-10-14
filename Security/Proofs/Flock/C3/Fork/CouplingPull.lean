import Proofs.Flock.Soundness.CompiledLock
import Proofs.Flock.C3.Fork.CouplingGame

/-!
# a1l, step 2: R's compiled stream is rep 0's, read at index 2

R's compiled Ligerito reads its openings at rep index `2` (`ligeritoC A 2`), and `lockLig` locksteps only reps `0` and
`1`. Rep `r`'s stream reads the openings only through `o r`, so it is rep 0's stream on the openings pulled from index
`r` (`pullO r o = fun _ => o r`): `ligeritoC_pull`. On those, `RecRel 0` is `RecRel 2` (`RecRel.pull`).
-/

namespace FlockC3.Fork

open FlockSoundness FlockSoundness.Game FlockSoundness.Model FlockSoundness.Accounting

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {D : Type}

/-- The openings at index `r`, read as index 0's (and every other index's). -/
abbrev pullO (r : ℕ) (o : Opens F D) : Opens F D := fun _ => o r

/-- A compiled Ligerito state with its claims read on the openings pulled from `r`. -/
abbrev pullS (r : ℕ) (st : LigStateC F K D) : LigStateC F K D :=
  { st with s := fun o => st.s (pullO r o), q := fun o => st.q (pullO r o) }

/-- A pending verdict read on the openings pulled from `r`. -/
abbrev pullP (r : ℕ) (p : PendL F D) : PendL F D := ⟨fun o => p.verdict (pullO r o), p.caps, p.reqs⟩

theorem foldRoundsC_pull (A : Arith F K) (r : ℕ) : ∀ (k : ℕ) (s : Opens F D → K) (q : Opens F D → Quad K),
    foldRoundsC A k (fun o => s (pullO r o)) (fun o => q (pullO r o)) =
      (foldRoundsC A k s q).map fun x => (x.1, fun o => x.2.1 (pullO r o), fun o => x.2.2 (pullO r o))
  | 0, s, q => rfl
  | k + 1, s, q => by
      simp only [foldRoundsC, Game.bind_eq, Game.pure_eq, Game.map, Game.bind_assoc]
      refine congrArg (Game.bind _) (funext fun c => ?_)
      refine congrArg (Game.bind _) (funext fun msg => ?_)
      rw [foldRoundsC_pull A r k (fun o => (q o).eval c) (fun o => fromMsg msg ((q o).eval c))]
      simp only [Game.map, Game.bind_assoc, Game.bind_ret]

theorem twoOodC_pull (A : Arith F K) (r dim split : ℕ) (s : Opens F D → K) (q : Opens F D → Quad K) :
    twoOodC A dim split (fun o => s (pullO r o)) (fun o => q (pullO r o)) =
      (twoOodC A dim split s q).map fun x => (fun o => x.1 (pullO r o), fun o => x.2.1 (pullO r o), x.2.2) :=
  rfl

theorem queryAndBatchC_pull (A : Arith F K) (r : ℕ) (lv : Level) (ch : List K) (start : ℕ)
    (st : LigStateC F K D) (wm : Bool) :
    queryAndBatchC A r lv ch start (pullS r st) wm = (queryAndBatchC A 0 lv ch start st wm).map (pullS r) := by
  cases wm <;> rfl

theorem recursiveRoundsC_pull (A : Arith F K) (r n : ℕ) (b : ℕ → F) (initial : List K) (ext : ℕ) :
    ∀ (levs : List Level) (st : LigStateC F K D),
      recursiveRoundsC A r n b initial ext levs (pullS r st) =
        (recursiveRoundsC A 0 n b initial ext levs st).map (pullP r)
  | [], _ => rfl
  | [lv], st => by
      simp only [recursiveRoundsC, Game.bind_eq, Game.pure_eq, Game.map, Game.bind_assoc]
      rw [foldRoundsC_pull A r lv.lanes st.s st.q]
      simp only [Game.map, Game.bind_assoc, Game.bind_ret]
      refine congrArg (Game.bind _) (funext fun x => ?_)
      refine congrArg (Game.bind _) (funext fun yr => ?_)
      rw [queryAndBatchC_pull A r lv x.1 _ { st with s := x.2.1, q := x.2.2, levels := st.levels ++ [x.1] } false]
      simp only [Game.map, Game.bind_assoc, Game.bind_ret]
  | lv :: lv' :: rest, st => by
      simp only [recursiveRoundsC, Game.bind_eq, Game.map, Game.bind_assoc]
      rw [foldRoundsC_pull A r lv.lanes st.s st.q]
      simp only [Game.map, Game.bind_assoc, Game.bind_ret]
      refine congrArg (Game.bind _) (funext fun x => ?_)
      refine congrArg (Game.bind _) (funext fun C' => ?_)
      rw [twoOodC_pull A r _ _ x.2.1 x.2.2]
      simp only [Game.map, Game.bind_assoc, Game.bind_ret]
      refine congrArg (Game.bind _) (funext fun y => ?_)
      rw [queryAndBatchC_pull A r lv x.1 _
        { st with s := y.1, q := y.2.1, levels := st.levels ++ [x.1], oods := st.oods ++ y.2.2 } true]
      simp only [Game.map, Game.bind_assoc, Game.bind_ret]
      refine congrArg (Game.bind _) (funext fun st3 => ?_)
      exact recursiveRoundsC_pull A r n b initial ext (lv' :: rest)
        { st3 with level := st3.level + 1, caps := st3.caps ++ [C'], prevLevel := lv' }

/-- **Rep `r`'s compiled Ligerito is rep 0's**, read on the openings pulled from `r`. -/
theorem ligeritoC_pull (A : Arith F K) (r m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ) (T : F)
    (b : ℕ → F) :
    ligeritoC (D := D) A r ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ T b =
      (ligeritoC A 0 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ T b).map (pullP r) := by
  simp only [ligeritoC, Game.bind_eq, Game.map, Game.bind_assoc]
  refine congrArg (Game.bind _) (funext fun z₀ => ?_)
  refine congrArg (Game.bind _) (funext fun y₀ => ?_)
  refine congrArg (Game.bind _) (funext fun β₀ => ?_)
  refine congrArg (Game.bind _) (funext fun msg => ?_)
  conv => lhs; rw [foldRoundsC_pull A r k0 (fun _ => A.embed T + A.embed β₀ * A.embed y₀)
    (fun _ => fromMsg msg (A.embed T + A.embed β₀ * A.embed y₀))]
  simp only [Game.map, Game.bind_assoc, Game.bind_ret]
  refine congrArg (Game.bind _) (funext fun x => ?_)
  refine congrArg (Game.bind _) (funext fun C₁ => ?_)
  rw [twoOodC_pull A r _ _ x.2.1 x.2.2]
  simp only [Game.map, Game.bind_assoc, Game.bind_ret]
  refine congrArg (Game.bind _) (funext fun y => ?_)
  rw [queryAndBatchC_pull A r l₀ x.1 0
    ⟨y.1, y.2.1, m - 7 - k0 + 1, [], ⟨List.ofFn z₀, β₀, none⟩ :: y.2.2, [], 0, [], [], l₀⟩ true]
  simp only [Game.map, Game.bind_assoc, Game.bind_ret]
  refine congrArg (Game.bind _) (funext fun st3 => ?_)
  exact recursiveRoundsC_pull A r (m - 7) b x.1 ext (l₁ :: rest)
    { st3 with level := 1, caps := [C₁], prevLevel := l₁ }

/-- On the openings pulled from `2`, rep 0's leaf relation is rep 2's. -/
theorem RecRel.pull {C : Oracle F} {sch : Schedule} {i : IRep F D} {v : Bool} {p : PendL F D}
    (h : RecRel 0 C sch i v p) : RecRel 2 C sch i v (pullP 2 p) :=
  ⟨fun o ho => h.1 (pullO 2 o) ho, h.2⟩

end FlockC3.Fork
