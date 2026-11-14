module

public import Properties.Nci.Record
public import Properties.Flock.Circuit.CircuitOf
public import Definitions.Accounting.ReadUnit
public import Spot.Bits

/-!
# A reader's ports as gates of its circuit file

A check holds a row as an opening (`Held.opened`) under a pin, by the roles of its reader's Definition (`Roles`): which
parameter is the key, which the root, the domain id and the opened rows, and which output row is the verdict. The
record names a unit by its template and its number, and a port by its parameter's name. The circuit a session checks
is the one its circuit file describes (`Flock.Bridge.circuitOf`), and it names gates. This module gives the gates from
the names, by the one map from a Definition's parameters to a file's gates, `Spot.Bits`:

* `rowGate`, `outRowGate`: the file's gate on a bit of a parameter, and of an output row, of one of its units.
  `Spot.Bits.fieldBit` finds the bit's instance, port and bit among the unit's ports, where the file's rows are the
  Definition's parameters in order, each on its packed 16-bit words, and `Spot.Bits.inGate` and `outGate` give the
  key program's gate there. The key program has the circuit's gates, gate for gate.
* `laidOutAs`: whether the file is laid out as that Definition's template, as `Spot.Bits.check` decides it, and its
  unit is the record's.
* `readRows`: whether a Definition's rows are a table read's: six parameters, the key's two words, the root's and the
  domain id's 32 each, then the witness row, its salt and its path, a digest a level, each named once; and two output
  rows, the row returned and then the verdict's one word.
* `portsOf`: a read unit's gates (`Accounting.ReadUnit.Ports`) at the roles, where the Definition's rows are a table
  read's: the gates of the key, the root, the domain id and the three opened parameters, of the output row the unit
  returns and of the first bit of its verdict's row. None where a bit of one of them has no gate.

**What they do not say.** `rowGate` reads no row of the file: that an input port's gate holds the bit the registered
row commits is the proof block's. `portsOf` reads `rowGate` and `outRowGate` alone, so its gates are a unit's ports
only where `laidOutAs` holds. Which output row a reader returns is no role of `Roles`: `portsOf` takes its name, and
the table's count and its row's words, which the reader's template fixes. A table's leaves sit each at its own rank.
-/

@[expose] public section

namespace NciCheck.Properties.Record

open Flock.Program (Prog Defn)
open Accounting

/-- **A Definition's parameters with their words**, in order: each parameter's name and its packed 16-bit words
(`Spot.Bits.wordsOf`), as a template instance's input rows hold them. -/
def paramWords (d : Defn) : List (String × ℕ) := d.params.toList.map fun p => (p.1, Spot.Bits.wordsOf p.2)

/-- **A Definition's output rows with their words**, in order: return field `f` is the row `o<f>`, as a record names
it, on its packed 16-bit words. -/
def outWords (d : Defn) : List (String × ℕ) :=
  (Spot.Bits.fieldsOf d.ret).zipIdx.map fun tf => (s!"o{tf.2}", Spot.Bits.wordsOf tf.1)

/-- **The gate on bit `b` of parameter `port` of unit `un`**, in the circuit of `n` units of the file `c`, laid out as
the Definition `d`'s template: the bit's instance, port and bit among the unit's input ports (`Spot.Bits.fieldBit`),
and the key program's gate there (`Spot.Bits.inGate`). None where `d` has no such parameter, the bit is past it, or
the port's bit is the zero. -/
def rowGate (c : Flock.Circuit) (d : Defn) (n un : ℕ) (port : String) (b : ℕ) : Option ℕ :=
  (Spot.Bits.fieldBit c (paramWords d) port b).bind fun kpt => Spot.Bits.inGate c n un kpt.1 kpt.2.1 kpt.2.2

/-- **The gate on bit `b` of output row `out` of unit `un`**, in the circuit of `n` units of the file `c`, laid out as
the Definition `d`'s template: the bit's instance, leaf and bit among the unit's returned outputs
(`Spot.Bits.fieldBit`), and the key program's gate there (`Spot.Bits.outGate`), a computed row's. -/
def outRowGate (c : Flock.Circuit) (d : Defn) (n un : ℕ) (out : String) (b : ℕ) : Option ℕ :=
  (Spot.Bits.fieldBit c (outWords d) out b).bind fun kjt => Spot.Bits.outGate c n un kjt.1 kjt.2.1 kjt.2.2

/-- **Whether the templated file `ct` is laid out as template `tid` of the Program `P`, and its unit `u` is the
record's unit `un`.** `Spot.Bits.check` gives no reason against the circuit the file unrolls to: its rows are the
Definition's parameters in order and its unit's ports read them. A unit of the file is the record's unit of the same
number. -/
def laidOutAs (ct : Flock.Circuit) (P : Prog) (tid : String) (u un : ℕ) : Bool :=
  (Spot.Bits.check (Flock.Desc.tmpl ct).unroll P tid).isNone && u == un

open Classical in
/-- **The first `k` gates of `gate`, as gates of a circuit of `N` gates**: none where one of them is none or no gate of
the circuit. -/
noncomputable def gatesOf (N k : ℕ) (gate : ℕ → Option ℕ) : Option (Fin k → Fin N) :=
  if h : ∀ i : Fin k, ∃ g : Fin N, gate i.val = some g.val then some fun i => Classical.choose (h i) else none

/-- **A read unit's gates at the roles `ro`**, in a circuit `C` whose gate on bit `b` of parameter `p` is `inG p b`
and on bit `b` of output row `o` is `outG o b`, for a table of `count` rows of `words` 16-bit words, each leaf at its
own rank. The index is the key parameter's first 32 bits, the root and the domain id are the root and domain
parameters' 512; the opened parameters are three, in order the witness row, its salt and its path, a digest a level;
the row returned is the output row `out`'s; and the verdict is the first bit of the verdict's row. None where the
opened parameters are not three, or a bit of one of these has no gate. -/
noncomputable def portsAtGates (C : Verity.Circuit) (inG outG : String → ℕ → Option ℕ) (ro : Roles) (out : String)
    (count words : ℕ) : Option (ReadUnit.Ports C) :=
  if ro.opened.length = 3 then
    (gatesOf C.N 32 (inG ro.key)).bind fun (tok : Fin 32 → Fin C.N) =>
    (gatesOf C.N 512 (inG ro.root)).bind fun (root : Fin 512 → Fin C.N) =>
    (gatesOf C.N 512 (inG ro.domain)).bind fun (domain : Fin 512 → Fin C.N) =>
    (gatesOf C.N (16 * words) (inG (ro.opened.getD 0 ""))).bind fun (rowG : Fin (16 * words) → Fin C.N) =>
    (gatesOf C.N (8 * Hash.Hm96.SALT_BYTES) (inG (ro.opened.getD 1 ""))).bind
      fun (saltG : Fin (8 * Hash.Hm96.SALT_BYTES) → Fin C.N) =>
    (gatesOf C.N (Hash.Tree.depth count * 512) (inG (ro.opened.getD 2 ""))).bind
      fun (pathG : Fin (Hash.Tree.depth count * 512) → Fin C.N) =>
    (gatesOf C.N (16 * words) (outG out)).bind fun (hG : Fin (16 * words) → Fin C.N) =>
    (gatesOf C.N 1 (outG ro.verdict)).map fun (okG : Fin 1 → Fin C.N) =>
      ({ count := count, words := words, position := fun r => r, tok := tok, root := root, domain := domain,
         row := rowG, salt := saltG,
         path := fun (l : Fin (Hash.Tree.depth count)) (i : Fin 512) =>
           pathG ⟨512 * l.val + i.val, by have := l.isLt; have := i.isLt; omega⟩,
         h := hG, ok := okG ⟨0, Nat.one_pos⟩ } : ReadUnit.Ports C)
  else none

/-- **Whether the Definition `d`'s rows are a table read's**, for a table of `count` rows of `words` 16-bit words, at
the roles `ro` and the output row `out`. Its parameters are six, in order: the key on two words, the root and the
domain id on 32 each, then the opened three, the witness row on `words`, its salt on half its bytes and its path on
32 a level. Its output rows are two: `out` on `words`, then the verdict's on one. No two parameters share a name, and
`out` is not the verdict's row. -/
def readRows (d : Defn) (ro : Roles) (out : String) (count words : ℕ) : Bool :=
  decide (paramWords d = [(ro.key, 2), (ro.root, 32), (ro.domain, 32), (ro.opened.getD 0 "", words),
    (ro.opened.getD 1 "", Hash.Hm96.SALT_BYTES / 2), (ro.opened.getD 2 "", 32 * Hash.Tree.depth count)]) &&
  decide (outWords d = [(out, words), (ro.verdict, 1)]) &&
  decide ([ro.key, ro.root, ro.domain, ro.opened.getD 0 "", ro.opened.getD 1 "", ro.opened.getD 2 ""].Nodup) &&
  decide ([out, ro.verdict].Nodup)

/-- **The read unit's gates of unit `un` of the templated file `ct`**, at the roles `ro` of the Definition `d` and the
output row `out`, in the circuit of `n` units the file describes: where `d`'s rows are a table read's (`readRows`),
`portsAtGates` at `rowGate` and `outRowGate` of the circuit the file unrolls to; none otherwise. -/
noncomputable def portsOf (ct : Flock.Circuit) (hB : FlockSoundness.Built (Flock.Desc.tmpl ct).unroll.unit) (n : ℕ)
    (d : Defn) (un : ℕ) (ro : Roles) (out : String) (count words : ℕ) :
    Option (ReadUnit.Ports (Flock.Bridge.circuitOf (.tmpl ct) hB n)) :=
  if readRows d ro out count words then
    portsAtGates (Flock.Bridge.circuitOf (.tmpl ct) hB n) (rowGate (Flock.Desc.tmpl ct).unroll d n un)
      (outRowGate (Flock.Desc.tmpl ct).unroll d n un) ro out count words
  else none

end NciCheck.Properties.Record
