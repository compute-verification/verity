module

public import Proofs.Flock.Soundness.E2E
public import Proofs.Flock.Soundness.Types.ProgramPlaces
public import Proofs.Flock.Soundness.Discharge.Const.Inst

/-! `hConst`: only a unit's constant column sits on the constant's gate, when each unit's instance wires its constant where the program does, which an instance chosen from `Prog.isRowsUnit` need not. -/

@[expose] public section

namespace FlockSoundness.Discharge.Const

open FlockSoundness Audit

section places

variable {N n : ℕ} {p : FlockSoundness.Prog N n} {outs : Finset (Fin N)} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}

theorem constCols_of_oneWire (pp : ProgPlaces p outs plan tab) (one : Fin N)
    (h : ∀ u, (pp.inst u).wire (p.rows u).one = one) :
    (p.partition outs).ConstCols plan tab pp.derived.place {one} :=
  (p.partition outs).constCols_singleton plan tab _ one fun _ _ u _ => h u

theorem constCols_of_oneGate (pp : ProgPlaces p outs plan tab) {one : Fin N} (hp : Prog.OneGate p one)
    (hi : ∀ u, (pp.inst u).wire (p.rows u).one = Prog.oneAt p u) :
    (p.partition outs).ConstCols plan tab pp.derived.place {one} :=
  constCols_of_oneWire pp one fun u => (hi u).trans (hp u)

theorem constCols_of_instC (pp : ProgPlaces p outs plan tab) {one : Fin N} (hp : Prog.OneGate p one)
    (hinst : ∀ u, pp.inst u = Prog.instC p outs u) :
    (p.partition outs).ConstCols plan tab pp.derived.place {one} :=
  constCols_of_oneGate pp hp fun u => by rw [hinst u]; exact Prog.instC_wire_one p outs u

/-- `{one}` is exactly the set of gates the units' constants read, as the certificate's `h1` needs; a set with another gate a unit's column reads makes `ConstCols` false. -/
theorem image_oneWire (pp : ProgPlaces p outs plan tab) {one : Fin N}
    (h : ∀ u, (pp.inst u).wire (p.rows u).one = one) (hn : 0 < n) :
    Finset.univ.image (fun u => (pp.inst u).wire (p.rows u).one) = {one} := by
  have : (fun u => (pp.inst u).wire (p.rows u).one) = fun _ => one := funext h
  rw [this]
  apply Finset.image_const
  exact ⟨⟨0, hn⟩, Finset.mem_univ _⟩

end places

namespace UProg

variable {k one : ℕ} (hk : one < k)

theorem toRows_oneAt : ∀ {outs : List ℕ} (up : FlockSoundness.UProg k one outs) (u : Fin up.n),
    Prog.oneAt (up.toRows hk) u = up.oneOf hk
  | _, .inputs, u => u.elim0
  | _, .snoc up s src valid, u => by
    induction u using Fin.lastCases with
    | last =>
      change Prog.oneAt (FlockSoundness.Prog.snoc (up.toRows hk) s.R (up.insOf s src valid) (up.oneOf hk)
        (up.insOf_ne hk s src valid)) (Fin.last up.n) = _
      rw [Prog.oneAt_last]
      rfl
    | cast v =>
      change Prog.oneAt (FlockSoundness.Prog.snoc (up.toRows hk) s.R (up.insOf s src valid) (up.oneOf hk)
        (up.insOf_ne hk s src valid)) v.castSucc = _
      rw [Prog.oneAt_castSucc, toRows_oneAt up v]
      rfl

theorem toRows_oneGate {outs : List ℕ} (up : FlockSoundness.UProg k one outs) :
    Prog.OneGate (up.toRows hk) (up.oneOf hk) :=
  toRows_oneAt hk up

variable {outs : List ℕ} (up : FlockSoundness.UProg k one outs) {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin up.n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin up.n)) → (R : Reg) → Fin up.n → Fin (plan S R).length}

end UProg

end FlockSoundness.Discharge.Const
