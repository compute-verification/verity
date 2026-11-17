module

public import Properties.InferenceOnly.Record
public import Properties.Flock.Circuit.CircuitOf
public import Definitions.ResourceAccounting.ReadUnit
public import Spot.Bits

/-! A reader's ports as gates of its circuit file. -/

@[expose] public section

namespace InferenceOnlyCheck.Properties.Record

open Flock.Program (Prog Defn)
open ResourceAccounting

def paramWords (d : Defn) : List (String × ℕ) := d.params.toList.map fun p => (p.1, Spot.Bits.wordsOf p.2)

def outWords (d : Defn) : List (String × ℕ) :=
  (Flock.Program.fieldsOf d.ret).zipIdx.map fun tf => (s!"o{tf.2}", Spot.Bits.wordsOf tf.1)

def rowGate (c : Flock.Circuit) (d : Defn) (n un : ℕ) (port : String) (b : ℕ) : Option ℕ :=
  (Spot.Bits.fieldBit c (paramWords d) port b).bind fun kpt => Spot.Bits.inGate c n un kpt.1 kpt.2.1 kpt.2.2

def outRowGate (c : Flock.Circuit) (d : Defn) (n un : ℕ) (out : String) (b : ℕ) : Option ℕ :=
  (Spot.Bits.fieldBit c (outWords d) out b).bind fun kjt => Spot.Bits.outGate c n un kjt.1 kjt.2.1 kjt.2.2

def laidOutAs (ct : Flock.Circuit) (P : Prog) (tid : String) (u un : ℕ) : Bool :=
  (Spot.Bits.check (Flock.Desc.tmpl ct).unroll P tid).isNone && u == un

open Classical in
noncomputable def gatesOf (N k : ℕ) (gate : ℕ → Option ℕ) : Option (Fin k → Fin N) :=
  if h : ∀ i : Fin k, ∃ g : Fin N, gate i.val = some g.val then some fun i => Classical.choose (h i) else none

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

def readRows (d : Defn) (ro : Roles) (out : String) (count words : ℕ) : Bool :=
  decide (paramWords d = [(ro.key, 2), (ro.root, 32), (ro.domain, 32), (ro.opened.getD 0 "", words),
    (ro.opened.getD 1 "", Hash.Hm96.SALT_BYTES / 2), (ro.opened.getD 2 "", 32 * Hash.Tree.depth count)]) &&
  decide (outWords d = [(out, words), (ro.verdict, 1)]) &&
  decide ([ro.key, ro.root, ro.domain, ro.opened.getD 0 "", ro.opened.getD 1 "", ro.opened.getD 2 ""].Nodup) &&
  decide ([out, ro.verdict].Nodup)

noncomputable def portsOf (ct : Flock.Circuit) (hB : FlockSoundness.Built (Flock.Desc.tmpl ct).unroll.unit) (n : ℕ)
    (d : Defn) (un : ℕ) (ro : Roles) (out : String) (count words : ℕ) :
    Option (ReadUnit.Ports (Flock.Bridge.circuitOf (.tmpl ct) hB n)) :=
  if readRows d ro out count words then
    portsAtGates (Flock.Bridge.circuitOf (.tmpl ct) hB n) (rowGate (Flock.Desc.tmpl ct).unroll d n un)
      (outRowGate (Flock.Desc.tmpl ct).unroll d n un) ro out count words
  else none

end InferenceOnlyCheck.Properties.Record
