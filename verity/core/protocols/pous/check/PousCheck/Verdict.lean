/-!
# The PoUS verdict unit as code

`PousVerdict_v1{K, M, LIMIT}` (`verity/ml/protocols/pous/verdict_circuit.py`), the proof unit whose one output is a
memory challenger's session verdict, in executable Lean:

* `prog K M LIMIT`: its gates in layout order and its output wire, builder for builder `verdict_circuit.gates`, on arrays;
* `Prog.eval`: a program run gate by gate, as `Claims.Sampling.Prog.eval` runs it;
* `roundVerdict`, `verdicts`, `accepts`: what its output is to be, from the openings `encode` lays out.

`Proofs.Pous.Continuous.VerdictCode` proves each is the security analysis's (`PousVerdict.verdictProg`, `Prog.eval`,
`PousVerdict.verdicts`, `contAccepts`), so `PousVerdict.VerdictIs` holds of `prog`, and `tests/test_verdict_difftest.py`
holds `prog` to the Definition's compiled circuit gate for gate (`PousCheck.VerdictDifftest`). Lean core only.

The builders read an array's size, and the parts of a pair they still need, before they push to the array, so that it is
not shared and each push is in place: at `K = 132` the unit has 318,154 gates.
-/

namespace PousCheck.Verdict

/-- A gate of the Boolean profile: an input, a constant bit, `Not_v1` of a wire, `And<n>_v1` or `Xor<n>_v1` of wires. -/
inductive Gate where
  | input
  | const (b : Bool)
  | not (i : Nat)
  | and (ws : List Nat)
  | xor (ws : List Nat)
  deriving Inhabited, BEq, Repr

/-- A program: its gates in layout order (wire `g` is gate `g`'s value), and the gates it returns. -/
structure Prog where
  gates : Array Gate
  outs : List Nat

/-! ## The gates (`verdict_circuit.gates`) -/

/-- A round record's bits: `u32 j ‖ u64 index ‖ u64 send_ns ‖ u64 receipt_ns ‖ H(block)`, 92 bytes. -/
def recordBits : Nat := 736

def digestBits : Nat := 512

/-- The unit's input bits: every record, then every expected digest. -/
def inputs (K : Nat) : Nat := (recordBits + digestBits) * K

/-- The positions of the big-endian integer at bytes `pos … pos + nbytes − 1` of a byte string from position `base`, least
significant bit first (`verdict_circuit.be_wires`). -/
def beWires (base pos nbytes : Nat) : List Nat :=
  (List.range (8 * nbytes)).map fun k => base + 8 * (pos + nbytes - 1 - k / 8) + k % 8

/-- `width` constant wires of `value`, least significant first. -/
def constWires (w0 w1 width value : Nat) : List Nat := (List.range width).map fun i => if value.testBit i then w1 else w0

/-- The carries of `a + ¬s + c` after `gs`, given the bits `(a_i, p_i)`: per bit `q = a ⊕ c`, `t = p ∧ q`, `c' = a ⊕ t`.
Returns the gates and the carries `c_0 = c, …, c_n`. -/
def carries : Array Gate → List (Nat × Nat) → Nat → Array Gate × List Nat
  | gs, [], c => (gs, [c])
  | gs, (x, p) :: r, c =>
    let q := gs.size
    let t := carries (((gs.push (.xor [x, c])).push (.and [p, q])).push (.xor [x, q + 1])) r (q + 2)
    (t.1, c :: t.2)

/-- **`a + ¬s + c0`** after `gs` (`verdict_circuit.chain`): the gates `p_i = a_i ⊕ s_i ⊕ 1`, then the carries. Returns the
gates, the `p` wires and the carries. -/
def chain (w1 : Nat) (gs : Array Gate) (a s : List Nat) (c0 : Nat) : Array Gate × List Nat × List Nat :=
  let ps := (List.range (a.zip s).length).map (gs.size + ·)
  let t := carries (gs ++ (a.zip s).map fun xy => Gate.xor [xy.1, xy.2, w1]) (a.zip ps) c0
  (t.1, ps, t.2)

/-- The sum bits `d_i = p_i ⊕ c_i` after `gs`. -/
def sums (gs : Array Gate) (ps cs : List Nat) : Array Gate × List Nat :=
  let ds := (List.range (ps.zip cs).length).map (gs.size + ·)
  (gs ++ (ps.zip cs).map (fun pc => Gate.xor [pc.1, pc.2]), ds)

/-- The equalities of round `j`: each bit of its record's `j` against `j`'s, and each bit of its `H(block)` against the
expected digest's, as `Xor3(a, b, 1)`. -/
def litGates (K w0 w1 j : Nat) : List Gate :=
  ((beWires (recordBits * j) 0 4).zip (constWires w0 w1 32 j)).map (fun xc => Gate.xor [xc.1, xc.2, w1]) ++
    (List.range digestBits).map fun t => Gate.xor [recordBits * j + 224 + t, recordBits * K + digestBits * j + t, w1]

/-- **Round `j`'s gates** after `gs`, and its hit wire: the equalities, `receipt + ¬send + 1` and its sum `receipt − send`,
`LIMIT + ¬(receipt − send) + 1`, and one AND over the equalities and the two last carries. -/
def roundGates (K LIMIT w0 w1 j : Nat) (gs : Array Gate) : Array Gate × Nat :=
  let lits := litGates K w0 w1 j
  let ls := (List.range lits.length).map (gs.size + ·)
  let c1 := chain w1 (gs ++ lits) (beWires (recordBits * j) 20 8) (beWires (recordBits * j) 12 8) w1
  let c := c1.2.2.getLastD 0
  let d := sums c1.1 c1.2.1 c1.2.2
  let c2 := chain w1 d.1 (constWires w0 w1 64 (min LIMIT (2 ^ 64 - 1))) d.2 w1
  let h := c2.1.size
  (c2.1.push (.and (ls ++ [c, c2.2.2.getLastD 0])), h)

/-- The inputs, the constants `0` and `1`. -/
def initGates (K : Nat) : Array Gate := ((Array.replicate (inputs K) Gate.input).push (.const false)).push (.const true)

/-- Round `j` after the gates and hit wires so far. -/
def hitStep (K LIMIT : Nat) (st : Array Gate × List Nat) (j : Nat) : Array Gate × List Nat :=
  let hs := st.2
  let r := roundGates K LIMIT (inputs K) (inputs K + 1) j st.1
  (r.1, hs ++ [r.2])

/-- Every round's gates, and their hit wires. -/
def hitGates (K LIMIT : Nat) : Array Gate × List Nat := (List.range K).foldl (hitStep K LIMIT) (initGates K, [])

/-- The bits of `n`: `max(K, M).bit_length()`. -/
def bitLength (n : Nat) : Nat := if n = 0 then 0 else n.log2 + 1

/-- The miss counter's width (`verdict_circuit.counter_width`). -/
def counterWidth (K M : Nat) : Nat := bitLength (max K M)

/-- One round of the miss counter: `cnt + 0 + ¬hit`. -/
def countStep (w1 w : Nat) (st : Array Gate × List Nat) (h : Nat) : Array Gate × List Nat :=
  let n := st.1.size
  let cnt := st.2
  let c := chain w1 (st.1.push (.not h)) cnt (List.replicate w w1) n
  sums c.1 c.2.1 c.2.2

/-- The counter over every round. -/
def countGates (K M LIMIT : Nat) : Array Gate × List Nat :=
  let hg := hitGates K LIMIT
  hg.2.foldl (countStep (inputs K + 1) (counterWidth K M)) (hg.1, List.replicate (counterWidth K M) (inputs K))

/-- **The verdict unit `PousVerdict_v1{K, M, LIMIT}`** (`verdict_circuit.gates`): its one output, the last carry of
`M + ¬cnt + 1`, is `cnt ≤ M`. -/
def prog (K M LIMIT : Nat) : Prog :=
  let cg := countGates K M LIMIT
  let c := chain (inputs K + 1) cg.1 (constWires (inputs K) (inputs K + 1) (counterWidth K M) M) cg.2 (inputs K + 1)
  ⟨c.1, [c.2.2.getLastD 0]⟩

/-! ## Running a program -/

/-- A gate's value from its operands' values `f`. -/
def gval (f : Nat → Bool) : Gate → Bool
  | .input => false
  | .const b => b
  | .not i => !f i
  | .and ws => (ws.map f).all id
  | .xor ws => (ws.map f).foldl Bool.xor false

/-- One gate run after the wires so far, `st.2` inputs read: an input reads the next of `x`, and a wire at or past the
reading gate's own reads `0`. -/
def wireStep (x : Array Bool) (st : Array Bool × Nat) : Gate → Array Bool × Nat
  | .input => let k := st.2; (st.1.push (x.getD k false), k + 1)
  | g => let k := st.2; (st.1.push (gval (fun i => st.1.getD i false) g), k)

/-- `C` run on inputs `x`, gate by gate (`x` read as an array: an input reads its position in constant time). -/
def Prog.wires (C : Prog) (x : List Bool) : Array Bool := (C.gates.foldl (wireStep x.toArray) (#[], 0)).1

/-- `C`'s returned values on inputs `x`. -/
def Prog.eval (C : Prog) (x : List Bool) : List Bool := C.outs.map fun o => (C.wires x).getD o false

/-! ## The openings and their verdicts -/

/-- Bits, least significant first, as a number. -/
def bitsVal : List Bool → Nat
  | [] => 0
  | b :: bs => b.toNat + 2 * bitsVal bs

/-- The big-endian integer at bytes `pos … pos + nbytes − 1` of `row`. -/
def beField (row : List Bool) (pos nbytes : Nat) : Nat := bitsVal ((beWires 0 pos nbytes).map fun i => row.getD i false)

/-- Round `j`'s opening: its record as the memory challenger writes it (`memory_challenger.Round.value`), and the expected
digest of the block it challenged. A byte string is bits in byte order, bit `t` being bit `t mod 8` of byte `t / 8`. -/
structure Opening where
  row : List Bool
  expected : List Bool

def Opening.j (o : Opening) : Nat := beField o.row 0 4
def Opening.send (o : Opening) : Nat := beField o.row 12 8
def Opening.receipt (o : Opening) : Nat := beField o.row 20 8

/-- The record's `H(block)`. -/
def Opening.digest (o : Opening) : List Bool := (List.range digestBits).map fun t => o.row.getD (224 + t) false

/-- The memory challenger's verdict on one round. -/
inductive RoundVerdict where
  | hit
  | wrong
  | late
  | reject
  deriving DecidableEq, Repr

/-- **Round `j`'s verdict** (`verdict_circuit.round_verdict`): rejected when its record names another round or its receipt
is before its send, late past `LIMIT`, wrong when its `H(block)` is not the expected digest, else a hit. -/
def roundVerdict (LIMIT j : Nat) (o : Opening) : RoundVerdict :=
  if o.j = j ∧ o.send ≤ o.receipt then
    if o.receipt - o.send ≤ LIMIT then (if o.digest = o.expected then .hit else .wrong) else .late
  else .reject

/-- The rounds' verdicts, round `j` the `j`-th opening's. -/
def verdicts (LIMIT : Nat) (os : List Opening) : List RoundVerdict := os.mapIdx fun j o => roundVerdict LIMIT j o

/-- The rounds that are not hits. -/
def misses (vs : List RoundVerdict) : Nat := (vs.filter (· ≠ .hit)).length

/-- Rule 5 at tolerance `M`: at most `M` rounds are not hits. -/
def accepts (M : Nat) (vs : List RoundVerdict) : Bool := decide (misses vs ≤ M)

/-- The unit's inputs in leaf order: every record, then every expected digest. -/
def encode (os : List Opening) : List Bool := os.flatMap (·.row) ++ os.flatMap (·.expected)

end PousCheck.Verdict
