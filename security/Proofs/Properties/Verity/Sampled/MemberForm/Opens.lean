module

public import Proofs.Properties.Verity.Sampled.Window.Unbound
public import Proofs.Properties.Verity.Sampled.Place

/-!
# The openings at a window's places, a member at a time

The claim at a window reads each member at its own gates of the members' circuits side by side
(`Flock.Guarantees.ReadsWith`), and asks of each member that the openings at its gates hold wherever its deployed
commitment opens (`MemberV5.ReadAt.opens`, `MemberV5.OpensUnder`). The vocabulary's openings at a gate no file fixes
are `Hash.RowLeaf` openings at the gate's registered place (`Vocab.Opens`), and a window's files register for a gate
its member's own place (`placeOfFiles`, `placeOfWith`). This module carries the one to the other:

* **A gate opens at a place** (`OpensPlace`): a `Hash.RowLeaf` opening under SHA-512 at the place, whose row holds
  the value's bits from the place's offset. It is `Vocab.Opens`'s clause at a gate no file fixes.
* **A window's places are its members'** (`PlacesWith`, `placesWith_of`, `placesWith_read`): openings that hold
  wherever a gate opens at the window's place for it hold, at each member's own gates, wherever the gate opens at
  that member's files' place (`placeOfRead`); so do the vocabulary's, where it places each gate at the committed
  files' place.
* **Member by member** (`OpensWith`, `OpensOwn`, `opensWith_of`, `opensWith_read`): `ReadsWith`'s recursion
  over the members, keeping of each only its openings' clause. It holds of the vocabulary's openings at members
  whose deployed commitments open only at their own files' places (`OpensOwn`), whose files are the
  window's in order.

Nothing here reads a play, a verdict or a session.
-/

@[expose] public section

namespace Verity.Sampled.MemberForm

open Flock.Guarantees (MemberV5)

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **A value of `w` bits opens at the place `pl`**: `pl` is a place, at which the row-leaf verifier accepts an
opening under SHA-512 whose row holds the value's bits from the place's offset. -/
def OpensPlace (pl : Option Place) (w : ℕ) (v : BitVec w) : Prop :=
  ∃ p, pl = some p ∧ ∃ o : Hash.RowLeaf.Opening,
    Hash.RowLeaf.accepts Verity.sha512 Verity.sha512 p.verifier p.root p.rank o ∧
      ∀ i : Fin w, o.row[p.offset + i.val]? = some (v.getLsbD i.val)

/-- A value that opens at a place opens there at an equal place and an equal width. -/
theorem OpensPlace.cast {pl pl' : Option Place} {w w' : ℕ} (hp : pl' = pl) (h : w = w') {v : BitVec w}
    (ho : OpensPlace pl w v) : OpensPlace pl' w' (v.cast h) := by
  obtain ⟨p, hpl, o, ha, hrow⟩ := ho
  exact ⟨p, hp.trans hpl, o, ha, fun i =>
    (hrow ⟨i.val, lt_of_lt_of_eq i.isLt h.symm⟩).trans (congrArg some (BitVec.getLsbD_cast h v).symm)⟩

/-- **The openings `op` hold wherever a gate opens at its member's place**, a first member's files and the rest's
side by side: at each gate the first member's files don't fix, a value that opens at those files' place for the gate
(`placeOfRead`) is opened by `op` at the gate's place among the circuits side by side; and the same of the rest, at
their gates. -/
def PlacesWith : (r : Verity.FileRead) → (rs : List Verity.FileRead) →
    ((g : Fin (Verity.FileRead.circuitWith r rs).N) → BitVec ((Verity.FileRead.circuitWith r rs).w g) → Prop) → Prop
  | r, [], op => ∀ (g : Fin r.circuit.N) (v : BitVec (r.circuit.w g)), r.consts g = none →
      OpensPlace (placeOfRead r g) (r.circuit.w g) v → op g v
  | r, r' :: rs, op =>
    (∀ (g : Fin r.circuit.N) (v : BitVec (r.circuit.w g)), r.consts g = none →
        OpensPlace (placeOfRead r g) (r.circuit.w g) v →
          op (Fin.castAdd (Verity.FileRead.circuitWith r' rs).N g)
            (v.cast (Fin.append_left r.circuit.w (Verity.FileRead.circuitWith r' rs).w g).symm)) ∧
      PlacesWith r' rs (fun (j : Fin (Verity.FileRead.circuitWith r' rs).N)
          (v : BitVec ((Verity.FileRead.circuitWith r' rs).w j)) =>
        op (Fin.natAdd r.circuit.N j)
          (v.cast (Fin.append_right r.circuit.w (Verity.FileRead.circuitWith r' rs).w j).symm))

/-- **A window's places are its members'**: openings that hold wherever a gate the files don't fix opens at the
place the files side by side register for it (`placeOfWith`) hold, member by member, wherever a gate opens at its
member's own place. -/
theorem placesWith_of : ∀ (r : Verity.FileRead) (rs : List Verity.FileRead)
    (op : (g : Fin (Verity.FileRead.circuitWith r rs).N) → BitVec ((Verity.FileRead.circuitWith r rs).w g) → Prop),
    (∀ (g : Fin (Verity.FileRead.circuitWith r rs).N) (v : BitVec ((Verity.FileRead.circuitWith r rs).w g)),
      Verity.FileRead.constsWith r rs g = none →
        OpensPlace (placeOfWith r rs g) ((Verity.FileRead.circuitWith r rs).w g) v → op g v) →
      PlacesWith r rs op
  | _, [], _, h => h
  | r, r' :: rs, _, h =>
    ⟨fun g _ hc ho =>
      h (Fin.castAdd (Verity.FileRead.circuitWith r' rs).N g) _
        ((Fin.addCases_left (m := r.circuit.N) (n := (Verity.FileRead.circuitWith r' rs).N)
          (motive := fun _ => Option Bool) (left := r.consts) (right := Verity.FileRead.constsWith r' rs) g).trans hc)
        (ho.cast (Fin.addCases_left (m := r.circuit.N) (n := (Verity.FileRead.circuitWith r' rs).N)
          (motive := fun _ => Option Place) (left := placeOfRead r) (right := placeOfWith r' rs) g) _),
      placesWith_of r' rs _ fun g _ hc ho =>
        h (Fin.natAdd r.circuit.N g) _
          ((Fin.addCases_right (m := r.circuit.N) (n := (Verity.FileRead.circuitWith r' rs).N)
            (motive := fun _ => Option Bool) (left := r.consts) (right := Verity.FileRead.constsWith r' rs) g).trans
            hc)
          (ho.cast (Fin.addCases_right (m := r.circuit.N) (n := (Verity.FileRead.circuitWith r' rs).N)
            (motive := fun _ => Option Place) (left := placeOfRead r) (right := placeOfWith r' rs) g) _)⟩

/-- **The vocabulary's openings hold wherever a gate opens at its member's place**, where the vocabulary places
each gate at the committed files' place (`placeOfFiles`): at a gate no file fixes, `Vocab.Opens` is an opening at the
gate's registered place. -/
theorem placesWith_read (V : Vocab d α) (θ c : ByteArray) (F : Array ByteArray)
    (hplace : ∀ g, V.place θ F c g = placeOfFiles (V.read θ F) g) :
    PlacesWith (V.read θ F).first (V.read θ F).rest (V.Opens θ F c) :=
  placesWith_of _ _ _ fun g v hc ho => by
    obtain ⟨p, hp, o, ha, hrow⟩ := ho
    have hc' : (V.read θ F).consts g = none := hc
    unfold Vocab.Opens
    rw [hc']
    exact ⟨p, (hplace g).trans hp, o, ha, hrow⟩

/-- **The member's deployed commitment opens only at its own files' places**: wherever it opens a gate its files
don't fix, under the member's roots, every value whose first bit it opens opens at the place the member's committed
files register for the gate (`placeOfRead`). -/
def OpensOwn {Reg : Type} {nJ : ℕ} (m : MemberV5 Reg nJ) : Prop :=
  m.OpensUnder m.fileRead fun g v => OpensPlace (placeOfRead m.fileRead g) (m.fileRead.circuit.w g) v

/-- Openings that hold wherever a member's deployed commitment opens hold under any openings they give at the gates
the files don't fix. -/
theorem opensUnder_mono {Reg : Type} {nJ : ℕ} {m : MemberV5 Reg nJ} {r : Verity.FileRead}
    {op op' : (g : Fin r.circuit.N) → BitVec (r.circuit.w g) → Prop} (h : m.OpensUnder r op)
    (hop : ∀ g v, r.consts g = none → op g v → op' g v) : m.OpensUnder r op' :=
  fun hN g b dg hc ho v hv => hop g v hc (h hN g b dg hc ho v hv)

/-- **The openings `op` hold wherever each member's deployed commitment opens**, a first member's files and the
rest's side by side and the members `ms` in order: `ReadsWith`'s recursion, keeping of each member its openings'
clause (`MemberV5.ReadAt.opens`). -/
def OpensWith {Reg : Type} {nJ : ℕ} : (r : Verity.FileRead) → (rs : List Verity.FileRead) →
    ((g : Fin (Verity.FileRead.circuitWith r rs).N) → BitVec ((Verity.FileRead.circuitWith r rs).w g) → Prop) →
    List (MemberV5 Reg nJ) → Prop
  | r, [], op, ms => ∃ m, ms = [m] ∧ m.OpensUnder r op
  | r, r' :: rs, op, ms => ∃ m ms', ms = m :: ms' ∧
      m.OpensUnder r
        (fun (i : Fin r.circuit.N) (v : BitVec (r.circuit.w i)) =>
          op (Fin.castAdd (Verity.FileRead.circuitWith r' rs).N i)
            (v.cast (Fin.append_left r.circuit.w (Verity.FileRead.circuitWith r' rs).w i).symm)) ∧
      OpensWith r' rs
        (fun (j : Fin (Verity.FileRead.circuitWith r' rs).N)
            (v : BitVec ((Verity.FileRead.circuitWith r' rs).w j)) =>
          op (Fin.natAdd r.circuit.N j)
            (v.cast (Fin.append_right r.circuit.w (Verity.FileRead.circuitWith r' rs).w j).symm)) ms'

/-- **Member by member**: openings that hold wherever a gate opens at its member's place hold wherever each member's
deployed commitment opens, at members that open only at their own files' places and whose files are the ones side by
side, in order. -/
theorem opensWith_of {Reg : Type} {nJ : ℕ} : ∀ (r : Verity.FileRead) (rs : List Verity.FileRead)
    (op : (g : Fin (Verity.FileRead.circuitWith r rs).N) → BitVec ((Verity.FileRead.circuitWith r rs).w g) → Prop)
    (ms : List (MemberV5 Reg nJ)), PlacesWith r rs op → ms.map (·.fileRead) = r :: rs → (∀ m ∈ ms, OpensOwn m) →
      OpensWith r rs op ms
  | r, [], op, ms, hP, hf, hown => by
    cases ms with
    | nil => simp at hf
    | cons m ms' =>
      cases ms' with
      | cons m' ms'' => simp at hf
      | nil =>
        obtain rfl : m.fileRead = r := List.head_eq_of_cons_eq hf
        exact ⟨m, rfl, opensUnder_mono (hown m (List.mem_singleton.2 rfl)) hP⟩
  | r, r' :: rs, op, ms, hP, hf, hown => by
    cases ms with
    | nil => simp at hf
    | cons m ms' =>
      obtain rfl : m.fileRead = r := List.head_eq_of_cons_eq hf
      obtain ⟨hP1, hP2⟩ := hP
      exact ⟨m, ms', rfl, opensUnder_mono (hown m List.mem_cons_self) hP1,
        opensWith_of r' rs _ ms' hP2 (List.tail_eq_of_cons_eq hf) fun m' hm' => hown m' (List.mem_cons_of_mem _ hm')⟩

/-- **The vocabulary's openings hold wherever each member's deployed commitment opens**, at members whose files are
the window's in order and whose deployed commitments open only at their own files' places, where the vocabulary
places each gate at the committed files' place. -/
theorem opensWith_read {Reg : Type} {nJ : ℕ} (V : Vocab d α) (θ c : ByteArray) (F : Array ByteArray)
    (hplace : ∀ g, V.place θ F c g = placeOfFiles (V.read θ F) g) (ms : List (MemberV5 Reg nJ))
    (hread : ms.map (·.fileRead) = (V.read θ F).first :: (V.read θ F).rest) (hown : ∀ m ∈ ms, OpensOwn m) :
    OpensWith (V.read θ F).first (V.read θ F).rest (V.Opens θ F c) ms :=
  opensWith_of _ _ _ ms (placesWith_read V θ c F hplace) hread hown

end Verity.Sampled.MemberForm
