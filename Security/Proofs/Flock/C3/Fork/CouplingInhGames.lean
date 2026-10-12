import Proofs.Flock.Soundness.Refine.LiveCompiled
import Proofs.Flock.Soundness.Model.HoloCompiled

/-!
# a1l, step 6's pieces: every message of the holographic tables' sub-games has an inhabited type

`AllInh` of each game `holoTable` and `holoTableCL` are built from: oracle Ligerito's (`foldRounds` to `ligerito`, as
`LiveCompiled` proves the compiled ones'), the holographic lincheck's, the sparse phase's (`drawFin`, `degRounds`), the
registered phase's at any `fin` whose games all have it, and each rep's (`holoRep`, `holoRepC`).
-/

namespace FlockC3.Fork

open FlockSoundness FlockSoundness.Game FlockSoundness.Model FlockSoundness.Accounting

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {D : Type}

theorem allInh_foldRounds (A : Arith F K) : ∀ (k : ℕ) (s : K) (q : Quad K), AllInh (foldRounds A k s q)
  | 0, _, _ => .ret _
  | k + 1, _, _ => by
      simp only [foldRounds]
      refine AllInh.bind' (allInh_drawK A) fun _ => ?_
      repeat' (first
        | exact AllInh.ret _ | exact AllInh.pure' _ | exact AllInh.recv' _ | exact AllInh.draw' _
        | refine AllInh.bind' ?_ fun _ => ?_ | intro _ | split | dsimp only)
      all_goals exact allInh_foldRounds A k _ _

theorem allInh_twoOod (A : Arith F K) (dim split : ℕ) (s : K) (q : Quad K) : AllInh (twoOod A dim split s q) := by
  unfold twoOod
  repeat' (first
    | exact AllInh.ret _ | exact AllInh.pure' _ | exact AllInh.recv' _ | exact AllInh.draw' _
    | refine AllInh.bind' ?_ fun _ => ?_)

theorem allInh_queryAndBatch (A : Arith F K) (lv : Level) (C : Oracle F) (ch : List K) (start : ℕ) (st : LigState F K)
    (withMessage : Bool) : AllInh (queryAndBatch A lv C ch start st withMessage) := by
  unfold queryAndBatch
  repeat' (first
    | exact AllInh.ret _ | exact AllInh.pure' _ | exact AllInh.recv' _ | exact AllInh.draw' _
    | refine AllInh.bind' ?_ fun _ => ?_ | intro _ | split)

theorem allInh_recursiveRounds (A : Arith F K) (n : ℕ) (b : ℕ → F) (initial : List K) (ext : ℕ) :
    ∀ (lvs : List Level) (st : LigState F K), AllInh (recursiveRounds A n b initial ext lvs st)
  | [], _ => .ret _
  | [lv], st => by
      simp only [recursiveRounds]
      refine AllInh.bind' (allInh_foldRounds A _ _ _) fun _ => ?_
      repeat' (first
        | exact allInh_queryAndBatch A _ _ _ _ _ _ | exact AllInh.ret _ | exact AllInh.pure' _
        | exact AllInh.recv' _ | exact AllInh.draw' _ | refine AllInh.bind' ?_ fun _ => ?_ | intro _ | split | dsimp only)
  | lv :: lv' :: rest, st => by
      simp only [recursiveRounds]
      refine AllInh.bind' (allInh_foldRounds A _ _ _) fun _ => ?_
      repeat' (first
        | exact AllInh.ret _ | exact AllInh.pure' _ | exact AllInh.recv' _ | exact AllInh.draw' _
        | refine AllInh.bind' ?_ fun _ => ?_ | intro _ | split | dsimp only)
      all_goals first
        | exact allInh_twoOod A _ _ _ _
        | exact allInh_queryAndBatch A _ _ _ _ _ _
        | exact allInh_recursiveRounds A n b initial ext _ _

theorem allInh_ligerito (A : Arith F K) (sch : Schedule) (T : F) (b : ℕ → F) (C₀ : Oracle F) :
    AllInh (ligerito A sch T b C₀) := by
  unfold ligerito
  repeat' (first
    | exact AllInh.ret _ | exact AllInh.pure' _ | exact AllInh.recv' _ | exact AllInh.draw' _
    | refine AllInh.bind' ?_ fun _ => ?_ | intro _ | split | dsimp only)
  all_goals first
    | exact allInh_foldRounds A _ _ _
    | exact allInh_twoOod A _ _ _ _
    | exact allInh_queryAndBatch A _ _ _ _ _ _
    | exact allInh_recursiveRounds A _ _ _ _ _ _

theorem allInh_holoLincheck (A : Arith F K) (S : Statement) (zc : ZcOut F S.m) : AllInh (holoLincheck A S zc) := by
  unfold holoLincheck
  repeat' (first
    | exact AllInh.ret _ | exact AllInh.pure' _ | exact AllInh.recv' _ | exact AllInh.draw' _
    | refine AllInh.bind' ?_ fun _ => ?_ | intro _ | split | dsimp only)
  all_goals exact allInh_lincheckRounds _ _

theorem allInh_drawFin : ∀ n : ℕ, AllInh (drawFin (F := F) n)
  | 0 => .ret _
  | n + 1 => by
      simp only [drawFin]
      exact AllInh.bind' (AllInh.draw' F) fun _ => AllInh.bind' (allInh_drawFin n) fun _ => AllInh.pure' _

theorem allInh_degRounds (k : ℕ) : ∀ (n : ℕ) (claim : F), AllInh (degRounds k n claim)
  | 0, _ => .ret _
  | n + 1, _ => by
      simp only [degRounds]
      repeat' (first
        | exact AllInh.ret _ | exact AllInh.pure' _ | exact AllInh.recv' _ | exact AllInh.draw' _
        | refine AllInh.bind' ?_ fun _ => ?_ | intro _ | split | dsimp only)
      all_goals exact allInh_degRounds k n _

theorem allInh_sparsePhase (A : Arith F K) (sh : SparseShape) (pub : SparsePub F sh) (τ : Fin (2 ^ sh.sLog) → ℕ)
    (h : Fin 64 → F) : AllInh (sparsePhase A sh pub τ h) := by
  unfold sparsePhase
  repeat' (first
    | exact allInh_drawFin _ | exact allInh_degRounds _ _ _ | exact AllInh.ret _ | exact AllInh.pure' _
    | exact AllInh.recv' _ | exact AllInh.draw' _ | refine AllInh.bind' ?_ fun _ => ?_ | intro _ | split | dsimp only)

theorem allInh_regPhase {α : Type} (A : Arith F K) (S : Statement) (sh : SparseShape)
    (units : Fin (2 ^ sh.sLog) → Finset ℕ) (τ : Fin (2 ^ sh.sLog) → ℕ) (no : α) {fin : OpenOut F → Game α}
    (hfin : ∀ op, AllInh (fin op))
    (reps : Option (ZcOut F S.m × HoloLcOut F S.m (S.kLog - 6)) × Option (ZcOut F S.m × HoloLcOut F S.m (S.kLog - 6))) :
    AllInh (regPhase A S sh units τ no fin reps) := by
  unfold regPhase
  split
  · refine AllInh.bind' (allInh_sparsePhase A sh _ τ _) fun _ => ?_
    split
    · exact AllInh.pure' _
    refine AllInh.bind' (allInh_sparsePhase A sh _ τ _) fun _ => ?_
    split
    · exact AllInh.pure' _
    split
    · refine AllInh.bind' (allInh_opening A _ _) fun _ => ?_
      split
      · exact AllInh.pure' _
      exact hfin _
    · exact AllInh.pure' _
  · exact AllInh.pure' _

theorem allInh_holoRep (A : Arith F K) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (C₀ : Oracle F)
    (extra : List (Claim F S.m)) : AllInh (holoRep A S sch hm C₀ extra) := by
  unfold holoRep
  refine AllInh.bind' (allInh_zerocheck A _) fun _ => ?_
  split
  · exact AllInh.pure' _
  refine AllInh.bind' (allInh_holoLincheck A S _) fun _ => ?_
  split
  · exact AllInh.pure' _
  refine AllInh.bind' (allInh_opening A _ _) fun _ => ?_
  split
  · exact AllInh.pure' _
  exact AllInh.bind' (allInh_ligerito A _ _ _ _) fun _ => AllInh.pure' _

theorem allInh_holoRepC [Nonempty D] (A : Arith F K) (r : ℕ) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m)
    (extra : List (Claim F S.m)) : AllInh (holoRepC (D := D) A r S sch hm extra) := by
  unfold holoRepC
  refine AllInh.bind' (allInh_zerocheck A _) fun _ => ?_
  split
  · exact AllInh.pure' _
  refine AllInh.bind' (allInh_holoLincheck A S _) fun _ => ?_
  split
  · exact AllInh.pure' _
  refine AllInh.bind' (allInh_opening A _ _) fun _ => ?_
  split
  · exact AllInh.pure' _
  exact AllInh.bind' (allInh_ligeritoC (D := D) A _ _ _ _) fun _ => AllInh.pure' _

end FlockC3.Fork
