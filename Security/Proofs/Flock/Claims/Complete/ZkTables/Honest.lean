import Proofs.Flock.Claims.Complete.Zk.Honest
import Proofs.Flock.Claims.Scope.Split

/-!
# The honest masked prover at several tables

The deployed relation reads `J` tables' hidden values interleaved: table `c`'s bit `a` is hidden value `a·J + c`
(`Readable.zOf`, `Readable.Sys.bit`). `tracesOf` reads table `i`'s bit `a` at `i·2^m + a`, which is the relation's only
at one table, so on a statement with several tables `honestPZk` runs each table on bits that are not its share of the
witness. Here each table's trace is the one-table trace of its share (`Readable.tableWit`, `traces`), and the honest
masked prover runs the masked session on those traces (`prog`, `honestP`).

* `traces_one`: at one table the traces are `tracesOf`'s, so on a statement whose setup has one table the program is
  `progZk`'s (`prog_one`) and the prover's next message `honestPZk`'s (`honestP_one`).
-/

namespace FlockComplete.ZkTables

open Flock FlockDeploy

/-- The session's traces: table `i`'s from its share of the witness's bits (`Readable.tableWit`). -/
def traces (sts : Array Flock.Stmt) (w : Claims.Flock.Wit) : Array Trace :=
  (Array.range sts.size).map fun i => traceOf sts[i]! (Readable.tableWit sts.size w i).hid.toArray 0

/-- **The honest masked prover's program** at several tables: `progZk`'s, on each table's share of the witness. -/
def prog (x : Claims.Msg) (w : Claims.Flock.Wit) (ω : TapeZk) : PI Claims.Msg :=
  match setupOf true x with
  | none => .ret []
  | some (_, spec, sts) => do
    match ← (sessionPZk asks Hashes.pi spec (tabsOf sts) (traces sts w) ω).run with
    | .ok (record, proofs) => pure (Final.encode ⟨x, record, proofs⟩)
    | .error _ => pure []

/-- **Flock's honest prover** at the masked verifier, at any number of tables: at each history, its program fed the
verifier's replies. -/
def honestP (w : Claims.Flock.Wit) : Claims.Prover TapeZk :=
  ⟨fun x hist ω => feed (prog x w ω) (hist.map Prod.snd)⟩

/-- **One table's share is the whole witness.** -/
theorem tableWit_one (w : Claims.Flock.Wit) : Readable.tableWit 1 w 0 = w := by
  obtain ⟨l⟩ := w
  show (⟨(List.range l.length).map fun a => l.getD (a * 1 + 0) false⟩ : Claims.Flock.Wit) = ⟨l⟩
  congr 1
  apply List.ext_getElem (by simp)
  intro n h1 h2
  simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h2]

/-- **At one table the traces are `tracesOf`'s.** -/
theorem traces_one {sts : Array Flock.Stmt} (h : sts.size = 1) (w : Claims.Flock.Wit) :
    traces sts w = tracesOf sts w := by
  unfold traces tracesOf
  apply Array.ext
  · simp
  · intro i h1 _
    simp only [Array.size_map, Array.size_range] at h1
    obtain rfl : i = 0 := by omega
    simp only [Array.getElem_map, Array.getElem_range, h, zero_mul, tableWit_one]

/-- **On a statement whose setup has one table the program is `progZk`'s.** -/
theorem prog_one {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array Flock.Stmt}
    (hs : setupOf true x = some (s, spec, sts)) (h1 : sts.size = 1) (w : Claims.Flock.Wit) (ω : TapeZk) :
    prog x w ω = progZk x w ω := by
  unfold prog progZk
  rw [hs]
  dsimp only
  rw [traces_one h1]
  rfl

/-- **On a statement whose setup has one table the prover's next message is `honestPZk`'s.** -/
theorem honestP_one {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array Flock.Stmt}
    (hs : setupOf true x = some (s, spec, sts)) (h1 : sts.size = 1) (w : Claims.Flock.Wit)
    (hist : List (Claims.Msg × Claims.Msg)) (ω : TapeZk) : (honestP w).next x hist ω = (honestPZk w).next x hist ω := by
  show feed (prog x w ω) _ = feed (progZk x w ω) _
  rw [prog_one hs h1]

end FlockComplete.ZkTables
