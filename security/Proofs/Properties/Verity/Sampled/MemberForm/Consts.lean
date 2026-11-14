module

public import Proofs.Properties.Verity.Sampled.Window.Unbound

/-!
# The openings at a window's constants, a member at a time

The claim at a window reads each member at its own gates of the members' circuits side by side
(`Flock.Guarantees.ReadsWith`), and asks of each member that the openings at its gates hold at its files' constants
(`MemberV5.ReadAt.consts`, `Verity.Sampled.ConstsOpen` at that member's files alone). The vocabulary's openings hold at
the constants of the whole window's files, by their definition (`Vocab.Opens` at a registered constant). This module
carries the one to the other:

* **The vocabulary's openings hold at the window's constants** (`constsOpen_read`): `Vocab.opens_const`, at every
  gate the window's files fix.
* **A window's constants are its members'** (`constsOpen_left`, `constsOpen_right`): openings that hold at the
  constants of a first member's files and the rest's, side by side (`Verity.FileRead.constsWith`), hold at the first
  member's constants at its own gates, and at the rest's at theirs.
* **Member by member** (`ConstsWith`, `constsWith_of`, `constsWith_read`): `ReadsWith`'s recursion over the members,
  keeping of each only its constants' clause, which so holds of the vocabulary's openings at every window.

Nothing here reads a play, a verdict or a session: the constants are public circuit structure.
-/

@[expose] public section

namespace Verity.Sampled.MemberForm

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **The vocabulary's openings hold at the committed files' constants**: a gate the files fix to `b` opens to every
value all of whose bits are `b` (`Vocab.opens_const`). -/
theorem constsOpen_read (V : Vocab d α) (θ c : ByteArray) (F : Array ByteArray) :
    ConstsOpen (V.read θ F) (V.Opens θ F c) :=
  fun _ _ hg v hv => (V.opens_const hg v).2 hv

/-- **Openings that hold at the constants of a first member's files and the rest's hold at the first member's**, at
its own gates of the circuits side by side. -/
theorem constsOpen_left {r r' : Verity.FileRead} {rs : List Verity.FileRead}
    {op : (g : Fin (Verity.FileRead.circuitWith r (r' :: rs)).N) →
      BitVec ((Verity.FileRead.circuitWith r (r' :: rs)).w g) → Prop}
    (h : ConstsOpen ⟨r, r' :: rs⟩ op) :
    ConstsOpen (.one r) fun (i : Fin r.circuit.N) (v : BitVec (r.circuit.w i)) =>
      op (Fin.castAdd (Verity.FileRead.circuitWith r' rs).N i)
        (v.cast (Fin.append_left r.circuit.w (Verity.FileRead.circuitWith r' rs).w i).symm) := by
  intro g b hg v hv
  have hw : (Verity.FileRead.circuitWith r (r' :: rs)).w (Fin.castAdd (Verity.FileRead.circuitWith r' rs).N g) =
      r.circuit.w g := Fin.append_left r.circuit.w (Verity.FileRead.circuitWith r' rs).w g
  refine h (Fin.castAdd (Verity.FileRead.circuitWith r' rs).N g) b ?_ _ ?_
  · exact (Fin.addCases_left (m := r.circuit.N) (n := (Verity.FileRead.circuitWith r' rs).N)
      (motive := fun _ => Option Bool) (left := r.consts) (right := Verity.FileRead.constsWith r' rs) g).trans hg
  · exact fun i => (BitVec.getLsbD_cast hw.symm v).trans (hv ⟨i.val, lt_of_lt_of_eq i.isLt hw⟩)

/-- **Openings that hold at the constants of a first member's files and the rest's hold at the rest's**, at their own
gates of the circuits side by side. -/
theorem constsOpen_right {r r' : Verity.FileRead} {rs : List Verity.FileRead}
    {op : (g : Fin (Verity.FileRead.circuitWith r (r' :: rs)).N) →
      BitVec ((Verity.FileRead.circuitWith r (r' :: rs)).w g) → Prop}
    (h : ConstsOpen ⟨r, r' :: rs⟩ op) :
    ConstsOpen ⟨r', rs⟩ fun (j : Fin (Verity.FileRead.circuitWith r' rs).N)
        (v : BitVec ((Verity.FileRead.circuitWith r' rs).w j)) =>
      op (Fin.natAdd r.circuit.N j)
        (v.cast (Fin.append_right r.circuit.w (Verity.FileRead.circuitWith r' rs).w j).symm) := by
  intro g b hg v hv
  have hw : (Verity.FileRead.circuitWith r (r' :: rs)).w (Fin.natAdd r.circuit.N g) =
      (Verity.FileRead.circuitWith r' rs).w g := Fin.append_right r.circuit.w (Verity.FileRead.circuitWith r' rs).w g
  refine h (Fin.natAdd r.circuit.N g) b ?_ _ ?_
  · exact (Fin.addCases_right (m := r.circuit.N) (n := (Verity.FileRead.circuitWith r' rs).N)
      (motive := fun _ => Option Bool) (left := r.consts) (right := Verity.FileRead.constsWith r' rs) g).trans hg
  · exact fun i => (BitVec.getLsbD_cast hw.symm v).trans (hv ⟨i.val, lt_of_lt_of_eq i.isLt hw⟩)

/-- **The openings `op` hold at each member's constants, a member at a time**: at the first member's files alone at
its own gates of the circuits side by side, and at each later member's at theirs, by `Flock.Guarantees.ReadsWith`'s
recursion over the members. -/
def ConstsWith : (r : Verity.FileRead) → (rs : List Verity.FileRead) →
    ((g : Fin (Verity.FileRead.circuitWith r rs).N) → BitVec ((Verity.FileRead.circuitWith r rs).w g) → Prop) → Prop
  | r, [], op => ConstsOpen (.one r) op
  | r, r' :: rs, op =>
    ConstsOpen (.one r) (fun (i : Fin r.circuit.N) (v : BitVec (r.circuit.w i)) =>
        op (Fin.castAdd (Verity.FileRead.circuitWith r' rs).N i)
          (v.cast (Fin.append_left r.circuit.w (Verity.FileRead.circuitWith r' rs).w i).symm)) ∧
      ConstsWith r' rs (fun (j : Fin (Verity.FileRead.circuitWith r' rs).N)
          (v : BitVec ((Verity.FileRead.circuitWith r' rs).w j)) =>
        op (Fin.natAdd r.circuit.N j)
          (v.cast (Fin.append_right r.circuit.w (Verity.FileRead.circuitWith r' rs).w j).symm))

/-- **Openings that hold at a window's constants hold at each member's**, a member at a time (`constsOpen_left`,
`constsOpen_right`). -/
theorem constsWith_of : ∀ (r : Verity.FileRead) (rs : List Verity.FileRead)
    (op : (g : Fin (Verity.FileRead.circuitWith r rs).N) → BitVec ((Verity.FileRead.circuitWith r rs).w g) → Prop),
    ConstsOpen ⟨r, rs⟩ op → ConstsWith r rs op
  | _, [], _, h => h
  | _, _ :: _, _, h => ⟨constsOpen_left h, constsWith_of _ _ _ (constsOpen_right h)⟩

/-- **The vocabulary's openings hold at each member's constants at every window**: of the committed files as the
vocabulary reads them, a member at a time. -/
theorem constsWith_read (V : Vocab d α) (θ c : ByteArray) (F : Array ByteArray) :
    ConstsWith (V.read θ F).first (V.read θ F).rest (V.Opens θ F c) :=
  constsWith_of _ _ _ (constsOpen_read V θ c F)

end Verity.Sampled.MemberForm
