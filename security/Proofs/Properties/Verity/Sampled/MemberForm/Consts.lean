module

public import Proofs.Properties.Verity.Sampled.Window.Unbound

/-! The openings at a window's constants, a member at a time. -/

@[expose] public section

namespace Verity.Sampled.MemberForm

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

theorem constsOpen_read (V : Vocab d α) (θ c : ByteArray) (F : Array ByteArray) :
    ConstsOpen (V.read θ F) (V.Opens θ F c) :=
  fun _ _ hg v hv => (V.opens_const hg v).2 hv

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

theorem constsWith_of : ∀ (r : Verity.FileRead) (rs : List Verity.FileRead)
    (op : (g : Fin (Verity.FileRead.circuitWith r rs).N) → BitVec ((Verity.FileRead.circuitWith r rs).w g) → Prop),
    ConstsOpen ⟨r, rs⟩ op → ConstsWith r rs op
  | _, [], _, h => h
  | _, _ :: _, _, h => ⟨constsOpen_left h, constsWith_of _ _ _ (constsOpen_right h)⟩

theorem constsWith_read (V : Vocab d α) (θ c : ByteArray) (F : Array ByteArray) :
    ConstsWith (V.read θ F).first (V.read θ F).rest (V.Opens θ F c) :=
  constsWith_of _ _ _ (constsOpen_read V θ c F)

end Verity.Sampled.MemberForm
