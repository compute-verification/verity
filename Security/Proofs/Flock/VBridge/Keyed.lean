import Proofs.Flock.VBridge.Unit
import Proofs.Flock.Soundness.Discharge.Placed.Keyed

/-!
# V*'s opening in the circuit's program on Δ's keys

`keyInst`: unit `u` of `Placed.keyProg c hU n`, the program a keyed audit layer is read against, as an instance of
`c.upv` stacked copies of the unit net's rows: the program's own wiring (`Const.Prog.rowsWireC`) carried along
`rows_circuitProg`. Its constant is on the gate the program wires it to (`keyInst_one`).

`keyProg_correct`: a transcript with no wrong unit makes every unit correct, so each copy's committed columns are the
transcript's (`copyVal_committed`). `keyProg_recOpen`: when the verifier built `c`'s unit net from V*'s laid-out opening
(`netOf`), copy `j` of unit `u` computes `laid_recOpen`'s outputs at any transcript carrying 1 on the unit's constant
gate (`unit_recOpen`).
-/

namespace FlockVBridge

open FlockSoundness FlockSoundness.Audit FlockSoundness.Discharge FlockSoundness.Discharge.Hm

variable {c : Flock.Circuit} (hU : Built c.unit) (n : ℕ)

/-- The gates of `keyProg c hU n`. -/
noncomputable abbrev keyGates : ℕ :=
  Placed.gates (Placed.inGates n (Placed.circuitRows c hU).nIn) (Placed.circuitRows c hU) n

/-- **Unit `u` of the circuit's program on Δ's keys**, as an instance of `c.upv` stacked copies of the unit net's
rows. -/
noncomputable def keyInst (outs : Finset (Fin (keyGates hU n))) (u : Fin n) :
    ((Placed.keyProg c hU n).partition outs).IsRowsUnit u ((Rows.ofNet hU).stack c.upv) :=
  (Const.Prog.castW (Placed.rows_circuitProg c hU n _ _ u)
    (Const.Prog.rowsWireC (Placed.keyProg c hU n) u)).toIsRowsUnit outs

/-- `keyInst` puts the unit's constant on the gate the program wires it to. -/
theorem keyInst_one (outs : Finset (Fin (keyGates hU n))) (u : Fin n) :
    (keyInst hU n outs u).wire ((Rows.ofNet hU).stack c.upv).one = Const.Prog.oneAt (Placed.keyProg c hU n) u :=
  (Const.Prog.castW_wire_one _ _).trans (Const.Prog.rowsWireC_wire_one _ u)

/-- **No wrong unit, every unit correct.** -/
theorem keyProg_correct {outs : Finset (Fin (keyGates hU n))} {X : Fin (keyGates hU n) → Bool}
    (hw : ((Placed.keyProg c hU n).partition outs).wrong X = ∅) (u : Fin n) :
    ((Placed.keyProg c hU n).partition outs).Correct X u := by
  by_contra h
  have hu := (Partition.mem_wrong _).2 h
  rw [hw] at hu
  simp at hu

/-- **A copy of V*'s laid-out opening in the circuit's program on Δ's keys computes the climb and the consistency
term**: `unit_recOpen` at `keyInst`, when the verifier built `c`'s unit net from `recOpenLaid` (`netOf`) and the
transcript carries 1 on unit `u`'s constant gate. -/
theorem keyProg_recOpen (outs : Finset (Fin (keyGates hU n))) (X : Fin (keyGates hU n) → Bool) (u : Fin n) {j : ℕ}
    (hj : j < c.upv) (hone : X (Const.Prog.oneAt (Placed.keyProg c hU n) u) = true)
    (ms : Flock.MerkleScheme) (hms : ∀ l r, ms.node l r = Flock.Sha512.hash (l ++ r)) (LANES H : ℕ) (hH : H ≤ 32)
    (p : Ports) (outBits : Array ℕ) {name : String}
    (hl : Flock.HmNets.netOf name (recOpenLaid LANES H p outBits) = .ok c.unit)
    (hu : c.unit.unitLog ≤ 32) (hout : (recOpenOuts LANES H p).length ≤ outBits.foldl (· + ·) 0)
    (r a cf : Array Flock.F128) (rest : ByteArray) (idx : ℕ) (ss : List ByteArray)
    (hr : r.size = LANES) (ha : a.size = 2) (hss : ss.length = H)
    (hrowN : p.row = 8 * (Flock.MerkleScheme.rowBytes r).size) (hauxN : 512 * H ≤ p.aux)
    (haccN : p.acc = 8 * (Flock.MerkleScheme.rowBytes a ++ rest).size)
    (hcoefN : p.coef = 8 * (Flock.MerkleScheme.rowBytes cf).size) (hdirsN : H ≤ p.dirs)
    (hrow : ColsHold (copyVal (keyInst hU n outs u) X hj) 0 (Flock.MerkleScheme.rowBytes r))
    (haux : ∀ i (hi : i < ss.length), ss[i].size = 64 ∧
      ColsHold (copyVal (keyInst hU n outs u) X hj) (p.row + 512 * i) ss[i])
    (hacc : ColsHold (copyVal (keyInst hU n outs u) X hj) (p.row + p.aux) (Flock.MerkleScheme.rowBytes a ++ rest))
    (hcoef : ColsHold (copyVal (keyInst hU n outs u) X hj) (p.row + p.aux + p.acc) (Flock.MerkleScheme.rowBytes cf))
    (hdir : ∀ h < H, copyVal (keyInst hU n outs u) X hj (p.row + p.aux + p.acc + p.coef + h) = idx.testBit h) :
    ColsHold (copyVal (keyInst hU n outs u) X hj)
      (outStart ((recOpenNet LANES H p).run Flock.HmNets.St.empty).2 p.bits outBits)
      ((Flock.climb ms (Flock.Sha512.hash (Flock.MerkleScheme.rowBytes r)) idx ss ++ ⟨Array.replicate 64 0⟩) ++
        (Flock.MerkleScheme.rowBytes #[consTerm LANES r a cf 0, consTerm LANES r a cf 1] ++ rest)) :=
  unit_recOpen hU (keyInst hU n outs u) X hj ((congrArg X (keyInst_one hU n outs u)).trans hone) ms hms LANES H hH p
    outBits hl hu hout r a cf rest idx ss hr ha hss hrowN hauxN haccN hcoefN hdirsN hrow haux hacc hcoef hdir

end FlockVBridge
