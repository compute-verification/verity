import Proofs.Flock.Bridge.FilePart.Srcs
import Proofs.Flock.Bridge.FilePartTarget
import Proofs.Flock.C3.Reps.ReadWhole

/-!
# The circuit file's partition, computed, at `unitLog ≤ 32`

`filePart_of_log`: `FilePartTarget`'s equation at every descriptor whose unit net has `unitLog ≤ 32`. Unit `u`'s block of
`SampledProofs.FilePart.ofCircuit` is `blockOf` of the key program's partition at `constsOf` (`blockOf_keyProg`): its
rows' lowered gates (`FilePart.blockOf_uniform`), then the sources it owns (`FilePart.mem_srcs_iff`), because
`RecResiduals.readIns` marks exactly the template inputs a computed row of `Rows.ofNet` reads (`readIns_iff`).

The bound is the one `Rows.ofNet_a` and `Rows.ofNet_b` take: `readIns` reads the net's sparse rows, which hold each
column modulo `2^32`, while `Rows.ofNet` holds rows whose columns run to `2^unitLog`. Past `unitLog = 32` a computed row
of `Rows.ofNet` can read a column `readIns` doesn't mark, and the equation is then neither provable nor refutable.
-/

namespace Flock.Bridge

/-- **`readIns` marks exactly the template inputs a computed row reads**, at `unitLog ≤ 32`. -/
theorem readIns_iff {net : Flock.Net} (hU : FlockSoundness.Built net) (hlog : net.unitLog ≤ 32) (y : ℕ)
    (hy : y < (FlockSoundness.Rows.ofNet hU).nIn) :
    (Flock.RecResiduals.readIns net).getD y false = true ↔
      ∃ j : Fin (FlockSoundness.Rows.ofNet hU).nComp,
        ∃ c₀ ∈ (FlockSoundness.Rows.ofNet hU).a j ++ (FlockSoundness.Rows.ofNet hU).b j, c₀.val = y := by
  constructor
  · intro h
    obtain ⟨-, r, hr, hr', hc⟩ := (FlockC3.Reps.ReadWhole.getD_readIns net y).1 h
    exact FlockC3.Reps.ReadWhole.readCol_of_row hU hlog hr hr' hc
  · intro h
    obtain ⟨r, hr, hr', hc⟩ := FlockC3.Reps.ReadWhole.row_of_readCol (col := y) hU hlog h
    exact (FlockC3.Reps.ReadWhole.getD_readIns net y).2
      ⟨lt_of_lt_of_eq hy (FlockSoundness.Rows.ofNet_nIn hU), r, hr, hr', hc⟩

/-- **The circuit's constants fix exactly the constant's and the zero's gates**, `0` and `1`. -/
theorem constsOf_eq_none (c : Flock.Circuit) (hU : FlockSoundness.Built c.unit) (n : ℕ)
    (g : Fin (FlockSoundness.Discharge.Layout.keyN c hU n)) : constsOf c hU n g = none ↔ 2 ≤ g.val := by
  have h1 : (FlockSoundness.Discharge.Layout.oneG : Fin (FlockSoundness.Discharge.Layout.keyN c hU n)).val = 0 := rfl
  have h0 : (FlockSoundness.Discharge.Layout.zeroG : Fin (FlockSoundness.Discharge.Layout.keyN c hU n)).val = 1 := rfl
  unfold constsOf
  by_cases ho : g = FlockSoundness.Discharge.Layout.oneG
  · rw [if_pos ho]
    have : g.val = 0 := by rw [ho]; exact h1
    constructor
    · intro h; cases h
    · intro h; omega
  · rw [if_neg ho]
    by_cases hz : g = FlockSoundness.Discharge.Layout.zeroG
    · rw [if_pos hz]
      have : g.val = 1 := by rw [hz]; exact h0
      constructor
      · intro h; cases h
      · intro h; omega
    · rw [if_neg hz]
      have h1' : g.val ≠ 0 := fun h => ho (Fin.ext (h.trans h1.symm))
      have h0' : g.val ≠ 1 := fun h => hz (Fin.ext (h.trans h0.symm))
      exact ⟨fun _ => by omega, fun _ => rfl⟩

/-- **The code's block is `blockOf`'s**: at `unitLog ≤ 32`, any outputs `outs` and unit `u`, `ofCircuit`'s block `u` is
`blockOf` of the key program's partition at the circuit's constants. -/
theorem blockOf_keyProg (c : Flock.Circuit) (hU : FlockSoundness.Built c.unit) (hlog : c.unit.unitLog ≤ 32) (n : ℕ)
    (outs : Finset (Fin (FlockSoundness.Discharge.Layout.keyN c hU n))) (u : Fin n) :
    ((SampledProofs.FilePart.ofCircuit c n).block u.val).toList =
      blockOf ((FlockSoundness.Discharge.Placed.keyProg c hU n).partition outs) (fun g => constsOf c hU n g) u := by
  have hSm := FilePart.mem_srcs_iff c (FlockSoundness.Rows.ofNet hU) (Flock.RecResiduals.readIns c.unit)
    (readIns_iff hU hlog) _ (SampledProofs.FilePart.mem_ofCircuit_srcs c n)
  rw [SampledProofs.FilePart.block_toList]
  exact (FilePart.blockOf_uniform
    (k := FlockSoundness.Discharge.Placed.inGates n (FlockSoundness.Discharge.Layout.keyB c hU))
    (R := FlockSoundness.Discharge.Placed.circuitRows c hU)
    (ins := FlockSoundness.Discharge.Placed.insOf n (FlockSoundness.Discharge.Layout.keyB c hU)
      (FlockSoundness.Discharge.Aliased.keyIdx c (FlockSoundness.Discharge.Layout.keyB c hU)))
    (one := FlockSoundness.Discharge.Placed.oneGate n _)
    (hone := FlockSoundness.Discharge.Placed.insOf_ne_one n _ _) outs (fun g => constsOf c hU n g)
    (B := FlockSoundness.Discharge.Layout.keyB c hU) rfl (fun g _ => constsOf_eq_none c hU n g)
    (Rd := fun b => ∃ i : Fin (FlockSoundness.Discharge.Placed.circuitRows c hU).nComp,
      ∃ col ∈ (FlockSoundness.Discharge.Placed.circuitRows c hU).a i ++
        (FlockSoundness.Discharge.Placed.circuitRows c hU).b i,
      ∃ hc : col.val < (FlockSoundness.Discharge.Placed.circuitRows c hU).nIn,
        (FlockSoundness.Discharge.Aliased.keyIdx c (FlockSoundness.Discharge.Layout.keyB c hU) ⟨col.val, hc⟩).map
          Fin.val = some b)
    (fun g v hg2 hgk => FilePart.reads_uniform outs g v hg2 hgk) _
    (SampledProofs.FilePart.ofCircuit_srcs_sorted c n) hSm u).symm

/-- **`FilePartTarget` at `unitLog ≤ 32`**: at every descriptor `D` whose unit net has `unitLog ≤ 32`, unit rows `hU`,
population `n` and unit `u`, `FilePart.ofCircuit`'s block `u` is `blockOf` of the key program's partition at the
circuit's constants. -/
theorem filePart_of_log (D : Flock.Desc) (hU : FlockSoundness.Built D.unroll.unit) (hlog : D.unroll.unit.unitLog ≤ 32)
    (n : ℕ) (u : Fin n) :
    let P := (FlockSoundness.Discharge.Placed.keyProg D.unroll hU n).partition
      (FlockSoundness.Discharge.Layout.outsOf D.unroll hU n)
    ((SampledProofs.FilePart.ofCircuit D.unroll n).block u.val).toList =
      blockOf P (fun g => constsOf D.unroll hU n g) u :=
  blockOf_keyProg D.unroll hU hlog n (FlockSoundness.Discharge.Layout.outsOf D.unroll hU n) u

end Flock.Bridge
