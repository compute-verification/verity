import Proofs.Flock.Recursive.Flock

/-!
# Recursion: the registered reads of V*'s sessions, read off an outcome

What `flock-verify verify --zk --registered F` reads from a registered value at an outcome of the recursive audit, as
functions of the outcome and of the registration's root `rt` (an instance's registered commitment, a `Port` of the
verifier's `--registered` file): the rows it reads (`readsZ`), the commit string `b ‖ c` it read at each (`rowZ`), and
the opening of `rt` it carries there (`opZ`).

* `StmtRead rt st i bc path`: a drawn instance of the statement `st` reads, by its ref into the table of `st`'s
  circuit's port named `rt`'s name, the table row whose commit string is `bc`; `i` is the position the verifier's
  program reads in the unit `st`'s header names for the instance (`rt.reads`), and `path` the path `st`'s header gives
  that table row (`Registered.entry`).
* `ReadV zs rt o i bc path`: at V*'s outer record `o`, a session whose verifier holds `rt` (`rt ∈ own`) sets up, from
  its transcript's record, a statement that reads `bc` at `i` through `path`.
* `readsZ`, `rowZ`, `opZ`: at the recursive audit's outcome, some session reads row `i` of `rt`; the commit string and
  path of one such read (`Classical.epsilon`), and the opening `(rt, bc, path)`.
* `opens_of_accepts`: when every session accepts (`AcceptsV`), each such read's commit string opens `rt` at its row
  through its path: the session's registered check (`checkRegistered`, `check_tables`, `checkPort_at`) opened it, at
  the position its header gives, which is the one `rt` gives for the instance's unit.

`Proofs.Flock.Recursive.ReadsOpen` states it in the form the recursive audit's property takes (`zk_reads_open`).
-/

namespace FlockSoundness.Discharge.Recursion

open FlockSoundness.Discharge.Exec (setupOf)
open FlockSoundness.Discharge.ZkExec (zkInputs)

/-- **A registered read of the statement `st` against the port `rt`**: a drawn instance `u` of `st` reads, by its ref
into the table of `st`'s circuit's port named `rt`'s name, the table row whose commit string is `bc`; `i` is the
position the verifier's program reads in the unit `st`'s header names for `u` (`rt.reads`), and `path` the path `st`'s
header gives that table row. -/
def StmtRead (rt : Flock.Registered.Port) (st : Flock.Stmt) (i : ℕ) (bc path : ByteArray) : Prop :=
  ∃ t idx p positions paths u, st.pub.tables = some t ∧ Flock.Registered.unitIndices st.pub = .ok idx ∧
    (List.range st.c.ports.size).find? (fun q => st.c.ports[q]!.name == rt.name) = some p ∧
    Flock.Registered.entry st.pub rt = .ok (positions, paths) ∧ u < st.pub.n ∧
    rt.reads[idx.getD u 0]? = some (some i) ∧
    bc = Flock.Registered.tableRow st.pub t p (Flock.Registered.ref t st.c.ports.size p u) ∧
    path = paths.getD (Flock.Registered.ref t st.c.ports.size p u) ByteArray.empty

/-- **The registered check opens each read against a port the verifier holds**: when `checkRegistered` accepts `st`
against `own` and `rt ∈ own`, a read's commit string opens `rt` at its row through its path. -/
theorem StmtRead.opens {own : Array Flock.Registered.Port} {rt : Flock.Registered.Port} {st : Flock.Stmt} {i : ℕ}
    {bc path : ByteArray} (hck : ZkReg.checkRegistered st (some own) = .ok ()) (hrt : rt ∈ own)
    (h : StmtRead rt st i bc path) :
    Flock.Registered.opens rt bc i path = true := by
  obtain ⟨t, idx, p, positions, paths, u, ht, hidx, hp, hent, hu, hi, rfl, rfl⟩ := h
  obtain ⟨t', idx', ht', hidx', hports⟩ := ZkReg.check_tables (ZkReg.checkRegistered_ok hck)
  rw [ht] at ht'
  cases ht'
  rw [hidx] at hidx'
  cases hidx'
  obtain ⟨p', positions', paths', hp', hent', hall⟩ := ZkReg.checkPort_at (hports rt hrt)
  rw [hp] at hp'
  cases hp'
  rw [hent] at hent'
  cases hent'
  obtain ⟨-, hreads, hopens⟩ := hall u hu
  rw [hi] at hreads
  have hpos : i = positions.getD (Flock.Registered.ref t st.c.ports.size p u) 0 :=
    Option.some.inj (Option.some.inj hreads)
  rw [hpos]
  exact hopens

variable {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type} {dg : ByteArray → ByteArray} {N : ℕ}

/-- **A registered read against `rt` at V*'s outer record `o`**: a session `j` whose verifier holds `rt` among its
registered ports sets up the statements `sts` from its transcript's record, and one of them reads `bc` at `i` through
`path` (`StmtRead`). -/
def ReadV (zs : Reg₂ → (j : Fin S) → (Vs j).Outer Reg') (rt : Flock.Registered.Port)
    (o : Reg₂ × VRec Vs Reg' dg N) (i : ℕ) (bc path : ByteArray) : Prop :=
  ∃ (j : Fin S) (spec : Flock.SessionSpec) (sts : Array Flock.Stmt) (k : ℕ) (hk : k < sts.size),
    rt ∈ (zs o.1 j).own ∧ setupOf (zkInputs (zs o.1 j).I) (o.2 j).2.2.2.record = .ok (spec, sts) ∧
    StmtRead rt sts[k] i bc path

/-- **When every session accepts, each read at V*'s outer record opens `rt` at its row**: the session's registered
check accepted the statement that reads it. -/
theorem ReadV.opens {zs : Reg₂ → (j : Fin S) → (Vs j).Outer Reg'} {rt : Flock.Registered.Port}
    {o : Reg₂ × VRec Vs Reg' dg N} {i : ℕ} {bc path : ByteArray} (hacc : AcceptsV zs o)
    (h : ReadV zs rt o i bc path) : Flock.Registered.opens rt bc i path = true := by
  obtain ⟨j, spec, sts, k, hk, hrt, hs, hst⟩ := h
  obtain ⟨spec', sts', hs', hreg, -⟩ := (show Accepts (zs o.1) o.2 from hacc) j
  rw [hs] at hs'
  cases hs'
  exact hst.opens (ZkExec.array_forM_ok hreg k hk) hrt

variable {Reg Ω H : Type} (zs : Reg → Ω → H → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
  (rt : Flock.Registered.Port)

/-- **The rows a run read**: at the recursive audit's outcome `o` (the registration, the draw, the rounds and V*'s
outer record), some session reads row `i` of `rt`. -/
def readsZ (o : Reg × Ω × H × (Reg₂ × VRec Vs Reg' dg N)) (i : ℕ) : Prop :=
  ∃ x : ByteArray × ByteArray, ReadV (zs o.1 o.2.1 o.2.2.1) rt o.2.2.2 i x.1 x.2

/-- One read of row `i` of `rt` at `o`: its commit string and path (any pair when there is none). -/
noncomputable def readZ (o : Reg × Ω × H × (Reg₂ × VRec Vs Reg' dg N)) (i : ℕ) : ByteArray × ByteArray :=
  Classical.epsilon fun x : ByteArray × ByteArray => ReadV (zs o.1 o.2.1 o.2.2.1) rt o.2.2.2 i x.1 x.2

/-- **The commit string a run read at row `i`.** -/
noncomputable def rowZ (o : Reg × Ω × H × (Reg₂ × VRec Vs Reg' dg N)) (i : ℕ) : ByteArray :=
  (readZ zs rt o i).1

/-- **The opening of `rt` a run carries at row `i`**: `rt`, the commit string and its path. -/
noncomputable def opZ (o : Reg × Ω × H × (Reg₂ × VRec Vs Reg' dg N)) (i : ℕ) :
    Flock.Registered.Port × ByteArray × ByteArray :=
  (rt, rowZ zs rt o i, (readZ zs rt o i).2)

/-- **At an outcome whose sessions all accept, each read's commit string opens `rt` at its row.** -/
theorem opens_of_accepts (o : Reg × Ω × H × (Reg₂ × VRec Vs Reg' dg N))
    (hacc : AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2) (i : ℕ) (hi : readsZ zs rt o i) :
    (opZ zs rt o i).1 = rt ∧ (opZ zs rt o i).2.1 = rowZ zs rt o i ∧
      Flock.Registered.opens rt (rowZ zs rt o i) i (opZ zs rt o i).2.2 = true :=
  ⟨rfl, rfl, ReadV.opens hacc (Classical.epsilon_spec hi)⟩

end FlockSoundness.Discharge.Recursion
