import FlockSoundness.Refine.LiveSim
import FlockSoundness.Game.Union
import FlockSoundness.Protocol.Discharge.Exec.Shape

/-!
# The static shape of a rep

`DecodesAll` (`Live.lean`) asks for each rep's run on a tape cut into rounds: the messages a round's bytes decode to,
then the round's coin. Where a run's coins fall among its messages is the game's, not the run's: every game of a rep
takes its steps in an order fixed by the statement and the schedule, and no step's kind depends on the values before
it. A rep may only stop early, on a reject.

* `Kind`, `kindOf`: a tape step's kind, a message or a coin, with its type.
* `Spine g σ`: every run of `g` takes steps of the kinds of `σ`, in order, until it returns. `Exact g σ`: and it
  returns exactly at the end of `σ`.
* `Spine.prefix`, `Exact.kinds`: a run's kinds are a prefix of `σ` (all of `σ`, for `Exact`).
* `zerocheck_exact`, `lincheck_exact`, `opening_exact`: the PIOP's phases have exact shapes; `ligeritoC_spine` and
  `repC_spine`: compiled Ligerito and a whole rep have spines, `repShape`, which read only the dimensions, the number
  of claims and the schedule.
-/

namespace FlockSoundness.Discharge.Exec

open FlockSoundness.Game FlockSoundness.Accounting FlockSoundness.Model

variable {α β : Type}

/-- Messages and coin steps interleaved along kinds: a message where the kind is a message's, a coin where it is a
coin's, until a list runs out. -/
def weave : List Kind → List Msg → List Msg → List Step
  | .msg _ :: σ, m :: ms, cs => .inl m :: weave σ ms cs
  | .coin _ :: σ, ms, c :: cs => .inr c :: weave σ ms cs
  | _, _, _ => []

theorem weave_self : ∀ ts : List Step,
    weave (ts.map kindOf) (ts.filterMap Sum.getLeft?) (ts.filterMap Sum.getRight?) = ts
  | [] => rfl
  | .inl x :: ts => by
      rw [List.map_cons, List.filterMap_cons, List.filterMap_cons, Sum.getLeft?_inl, Sum.getRight?_inl]
      exact congrArg (_ :: ·) (weave_self ts)
  | .inr x :: ts => by
      rw [List.map_cons, List.filterMap_cons, List.filterMap_cons, Sum.getLeft?_inr, Sum.getRight?_inr]
      exact congrArg (_ :: ·) (weave_self ts)

/-- **A run's tape is its messages and coins woven along the spine.** Where the coins fall among the messages is the
spine's, not the run's. -/
theorem Spine.tape {g : Game α} {ts : List Step} {a : α} {σ : List Kind} (hp : PlaysT g ts a) (hs : Spine g σ) :
    ts = weave (σ.take ts.length) (ts.filterMap Sum.getLeft?) (ts.filterMap Sum.getRight?) := by
  have h := List.prefix_iff_eq_take.mp (Spine.prefix hp hs)
  rw [List.length_map] at h
  rw [← h]
  exact (weave_self ts).symm

/-! ## The PIOP's phases -/

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {D : Type}

/-! ## Compiled Ligerito -/

/-! ## A rep -/

omit [Fintype K] in
/-- **A rep's run follows `repShape`**: on any tape, a run of rep `r` takes steps whose kinds, in order, are a prefix
of `repShape`. -/
theorem repC_kinds (A : Arith F K) (r : ℕ) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m)
    (extra : List (Claim F S.m)) {ts : List Step} {p : PendL F D} (h : PlaysT (repC A r S sch hm extra) ts p) :
    ts.map kindOf <+: repShape F K D S.m S.kLog extra.length sch :=
  Spine.prefix h (repC_spine A r S sch hm extra)

omit [Fintype K] in
/-- **A rep's tape is determined by its messages and coins.** Two runs of rep `r`, with the same messages and coin
steps in order, are runs on the same tape: `repShape` places the coins among the messages. -/
theorem repC_tape (A : Arith F K) (r : ℕ) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m)
    (extra : List (Claim F S.m)) {ts ts' : List Step} {p p' : PendL F D} (h : PlaysT (repC A r S sch hm extra) ts p)
    (h' : PlaysT (repC A r S sch hm extra) ts' p') (hms : ts.filterMap Sum.getLeft? = ts'.filterMap Sum.getLeft?)
    (hcs : ts.filterMap Sum.getRight? = ts'.filterMap Sum.getRight?) : ts = ts' := by
  have hlen : ts.length = ts'.length := by
    have e : ∀ l : List Step, l.length = (l.filterMap Sum.getLeft?).length + (l.filterMap Sum.getRight?).length := by
      intro l
      induction l with
      | nil => rfl
      | cons s l ih =>
          cases s with
          | inl x =>
              simp only [List.filterMap_cons, Sum.getLeft?_inl, Sum.getRight?_inl, List.length_cons, ih]
              omega
          | inr x =>
              simp only [List.filterMap_cons, Sum.getLeft?_inr, Sum.getRight?_inr, List.length_cons, ih]
              omega
    rw [e ts, e ts', hms, hcs]
  rw [Spine.tape h (repC_spine A r S sch hm extra), Spine.tape h' (repC_spine A r S sch hm extra), hlen, hms, hcs]

end FlockSoundness.Discharge.Exec
