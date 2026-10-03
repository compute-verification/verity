import FlockSoundness.Discharge.PrivateCircuit.UnitEval
import FlockSoundness.Discharge.PrivateCircuit.Interp
import FlockSoundness.Discharge.Layout.Outs

/-!
# The universal unit's partition, read at its own gates

`UnitWrong P progOf xOf yOf` (`InstanceSound.lean`) leaves the reads `progOf`, `xOf`, `yOf` free, and so holds of every
partition for some `yOf` (`unitWrong_trivial`, `UnitWrongReal.lean`): it says something about the circuit only once the
reads are the circuit's own. Here they are, at the session's partition `(keyProg c hU n).partition (outsOf c hU n)` of
the parsed circuit `c`, whose units are `c.upv` stacked copies of the parsed net's rows `Rows.ofNet hU`:

* `UIns`: where the universal unit's arguments (`op`, `src`, `out`, `x`) sit among the net's input columns;
* `UniversalComputes`: **the parsed net computes `UniversalUnit_v1`**: on any input bits, with the constant 1, bit `k`
  of the unit's returned outputs (column `outCol (k / 16) (k % 16)`, `OutColAt`) is `interp`'s output `k`, and `0` past
  the `nOut` outputs. A fact about the net alone, with no partition, session or assignment in it;
* `uProg`, `uX`, `uY`: unit `u`'s program, data and claimed outputs at an assignment `v`, read at the gates its input
  columns and its returned outputs' columns are wired to. `uY` is the first `nOut` returned bits when the rest are
  zero, and all of them otherwise (a claim no unit meets: it is longer than `nOut`);
* `UnitWrongOn P good …`: `UnitWrong` at the assignments `good` holds of. A unit's local evaluation reads the constant's
  gate `oneG` from the assignment, so the partition is the interpreter's only where `v oneG = true`.
-/

namespace FlockSoundness.Discharge.PrivateCircuit

open Flock Audit

/-- **The universal unit's arguments among `nIn` input columns**: `op` bit `t` of gate `g`, bit `j` of gate `g`'s source
`s`, bit `j` of output `i`'s wire index, and data bit `j`, each on one input column. -/
structure UIns (cls : Universal.Cls) (nIn : ℕ) where
  op : Fin cls.G → Fin 2 → Fin nIn
  src : Fin cls.G → Fin 2 → Fin (Universal.idxBits cls.G cls.nIn) → Fin nIn
  out : Fin cls.nOut → Fin (Universal.idxBits cls.G cls.nIn) → Fin nIn
  x : Fin cls.nIn → Fin nIn

/-- The `K`-bit vector of the bits `f`, LSB first. -/
def bitsLE (K : ℕ) (f : Fin K → Bool) : BitVec K := (BitVec.ofBoolListLE (List.ofFn f)).cast (List.length_ofFn)

namespace UIns

variable {cls : Universal.Cls} {nIn : ℕ} (L : UIns cls nIn)

/-- **The program on the input bits `z`**: gate `g`'s op and sources, output `i`'s wire index, read at `L`'s columns. -/
def prog (z : Fin nIn → Bool) : cls.Unit where
  gates g := ⟨bitsLE 2 fun t => z (L.op g t), bitsLE _ fun j => z (L.src g 0 j), bitsLE _ fun j => z (L.src g 1 j)⟩
  out i := bitsLE _ fun j => z (L.out i j)

/-- **The data on the input bits `z`**, read at `L`'s columns. -/
def data (z : Fin nIn → Bool) : Fin cls.nIn → Bool := fun j => z (L.x j)

end UIns

section Circuit

variable (c : Flock.Circuit) (hU : Built c.unit)

/-- **The unit's returned-output bits**: 16 per output word of the first copy's leaf (`OutColAt`'s `j`, `t`). -/
def outBits : ℕ := 16 * (c.leavesOut.getD 0 #[]).size

/-- **The parsed net computes `UniversalUnit_v1{G, nIn, nOut}`** at the arguments' columns `L`: the `nOut` outputs fit
in the returned bits, each returned bit's column `outCol (k / 16) (k % 16)` is a computed row, and on every input bits
`z`, with the constant 1, returned bit `k` is `interp`'s output `k` on the program and data at `L`, and `0` past the
outputs. -/
structure UniversalComputes (cls : Universal.Cls) (L : UIns cls (FlockSoundness.Rows.ofNet hU).nIn) : Prop where
  out_le : cls.nOut ≤ outBits c
  out_comp : ∀ k < outBits c, (FlockSoundness.Rows.ofNet hU).nIn ≤ Layout.outCol c (k / 16) (k % 16) ∧
    Layout.outCol c (k / 16) (k % 16) < (FlockSoundness.Rows.ofNet hU).nIn + (FlockSoundness.Rows.ofNet hU).nComp
  computes : ∀ (z : Fin (FlockSoundness.Rows.ofNet hU).nIn → Bool) (k : ℕ) (hk : k < outBits c),
    rowsEval (FlockSoundness.Rows.ofNet hU) z true ⟨Layout.outCol c (k / 16) (k % 16), by
      have := (out_comp k hk).2; show _ < _ + _ + 1; omega⟩ =
      if h : k < cls.nOut then Universal.interp (L.prog z) (L.data z) ⟨k, h⟩ else false

variable (n : ℕ)

/-- **The gate of unit `u`'s column `x`** in the session's program (`instC`'s wire), the constant's past the columns. -/
noncomputable def colGate (u : Fin n) (x : ℕ) : Fin (Layout.keyN c hU n) :=
  if h : x < ((Placed.keyProg c hU n).rows u).w then
    (Const.Prog.instC (Placed.keyProg c hU n) (Layout.outsOf c hU n) u).wire ⟨x, h⟩
  else Layout.oneG

/-- **Returned bit `k`'s column in a unit's rows**: the first copy's computed column `outCol (k / 16) (k % 16)`, stacked
(`OutColAt` at copy 0). -/
noncomputable def outStack (k : ℕ) : ℕ :=
  c.upv * (FlockSoundness.Rows.ofNet hU).nIn +
    (0 * ((FlockSoundness.Rows.ofNet hU).nComp + 1) +
      (1 + (Layout.outCol c (k / 16) (k % 16) - (FlockSoundness.Rows.ofNet hU).nIn)))

variable {c hU n}

/-- **Unit `u`'s input bits at `v`**: the values at the gates its input columns are wired to (`insAt`). -/
noncomputable def uIn (v : Fin (Layout.keyN c hU n) → Bool) (u : Fin n) : Fin (FlockSoundness.Rows.ofNet hU).nIn → Bool :=
  fun j => v (Const.Prog.insAt (Placed.keyProg c hU n) u j.val)

variable {cls : Universal.Cls}

/-- **The program `v` holds**: each unit's program at its input bits. -/
noncomputable def uProg (L : UIns cls (FlockSoundness.Rows.ofNet hU).nIn) (v : Fin (Layout.keyN c hU (Universal.units cls)) → Bool) :
    cls.Prog :=
  fun u => L.prog (uIn v u)

/-- **Unit `i`'s data at `v`**, its data bits at its input bits (`false` past the units and the data). -/
noncomputable def uX (L : UIns cls (FlockSoundness.Rows.ofNet hU).nIn) (v : Fin (Layout.keyN c hU (Universal.units cls)) → Bool)
    (i j : ℕ) : Bool :=
  if h : i < Universal.units cls then
    if hj : j < cls.nIn then L.data (uIn v ⟨i, h⟩) ⟨j, hj⟩ else false
  else false

/-- **Unit `i`'s returned bits at `v`**: the values at its returned outputs' gates, in order. -/
noncomputable def uRet (v : Fin (Layout.keyN c hU (Universal.units cls)) → Bool) (i : Fin (Universal.units cls)) :
    List Bool :=
  List.ofFn fun k : Fin (outBits c) => v (colGate c hU (Universal.units cls) i (outStack c hU k.val))

variable (cls) in
/-- **Unit `i`'s claimed outputs at `v`**: its first `nOut` returned bits when the others are `0`, and all of its returned
bits otherwise (`[]` past the units). -/
noncomputable def uY (v : Fin (Layout.keyN c hU (Universal.units cls)) → Bool) (i : ℕ) : List Bool :=
  if h : i < Universal.units cls then
    if ((uRet v ⟨i, h⟩).drop cls.nOut).all (fun b => !b) then (uRet v ⟨i, h⟩).take cls.nOut else uRet v ⟨i, h⟩
  else []

end Circuit

/-- **`UnitWrong` at the assignments `good` holds of.** -/
def UnitWrongOn {C : Audit.Circuit} {c : Universal.Cls} (P : Audit.Partition C (Universal.units c))
    (good : (Fin C.N → Bool) → Prop) (progOf : (Fin C.N → Bool) → c.Prog) (xOf : (Fin C.N → Bool) → ℕ → ℕ → Bool)
    (yOf : (Fin C.N → Bool) → ℕ → List Bool) : Prop :=
  ∀ v, good v → ∀ (i : Fin (Universal.units c)), i ∈ P.wrong v ↔ Universal.run c (progOf v) i (xOf v i) ≠ yOf v i

end FlockSoundness.Discharge.PrivateCircuit
