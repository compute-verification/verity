module

public import Properties.Verity.Game
public import Proofs.Flock.Bridge.Spec
public import Proofs.Flock.Bridge.LeafMap
public import Proofs.Flock.Soundness.Discharge.HiddenExec.KeyHP

/-!
# The place the committed files register for a gate

A base's `Vocab.place` read off the committed files alone, so a deployed base can define its place by it and
`Instance.OpensAt` reads it (`Sampled/OpensAt.lean`):

* `placeFrom`: **the place a commitment's layout registers for a claims position** (`Flock.Bridge.commitDepL`'s): the
  registered port whose root is the position's row's root, as the deployed path check finds it
  (`Claims.Sampling.regPath`), read as `Hash.RowLeaf`'s verifier (the port's frame-v3 domain, the executable's hm96
  key, the format of the port's registered schema), the port's root, the row's leaf, and the bit's place in its row.
* `placeOfRead`: **the place the committed files register for a gate**: the files parsed as `verify --zk` parses them,
  and `placeFrom` at the deployed commitment's layout there, at the gate's claims position.
-/

@[expose] public section

namespace Verity.Sampled

open Properties.NetworkCertifier.Bytes (bs)

/-- **The place a commitment's layout registers for claims position `i`** (`Flock.Bridge.commitDepL`'s `C`, `rows`,
`c`, `L` and `roots`): the registered port whose root is the position's row's root, as the deployed path check finds it
(`Claims.Sampling.regPath`), read as `Hash.RowLeaf`'s verifier at the port's frame-v3 domain
(`Flock.Bridge.Spec.domainOf`), the executable's hm96 key and its registered schema's format
(`Flock.Bridge.Spec.formatOf`); the port's root; the row's leaf; and the bit's place in its row. -/
noncomputable def placeFrom (C : Verity.Boolean.Circuit) (rows : Fin C.N → Option Flock.Bridge.RowAt)
    (c : Flock.Circuit) (L : Flock.Bridge.LeafMap) (roots : List Claims.Msg) (i : ℕ) : Option Place :=
  (Flock.Bridge.atOf C rows roots i).bind fun ri =>
    ((Flock.Bridge.portsOfL c L roots).find? fun p => decide (bs p.root = ri.1)).map fun reg =>
      { verifier := { domain := Flock.Bridge.Spec.domainOf reg, key := bs Flock.Hm96.Default512.key,
                      format := Flock.Bridge.Spec.formatOf reg.schema }
        root := Flock.Bridge.Spec.dig reg.root, rank := ri.2, offset := (Flock.Bridge.placeOf C rows i).2.val }

open Classical in
/-- **The place the committed files register for gate `g`**: the files parsed as `verify --zk` parses them (the
circuit file and its tables by `Flock.Desc.parse`, the public file by `HmIn.loadPublic` with no public input port),
the deployed commitment's layout at that parse (`Live.Cd`'s), and `placeFrom` there at the gate's claims position;
`none` where the files don't parse. -/
noncomputable def placeOfRead (r : Verity.FileRead) (g : Fin r.circuit.N) : Option Place :=
  match Flock.Desc.parse r.tags r.circuitFile r.tables with
  | .error _ => none
  | .ok D =>
    match Flock.HmIn.loadPublic r.tags D.unroll {} r.publicFile with
    | .error _ => none
    | .ok pub =>
      if hU : FlockSoundness.Built D.unroll.unit then
        let outs := FlockSoundness.Discharge.Layout.outsOfP D.unroll hU pub.n {}
        let C := (FlockSoundness.Discharge.Placed.keyProgP D.unroll {} hU pub.n).circuit outs
        let L := Flock.Bridge.LeafMap.ofPublic D.unroll pub pub.n
        let rows : Fin C.N → Option Flock.Bridge.RowAt := fun x =>
          (FlockSoundness.Discharge.HiddenExec.rowOfHP D.unroll hU pub.n {} outs x).map fun w =>
            ⟨w.2.1.val, L.ref w.2.1.val w.1.val, FlockSoundness.Discharge.Layout.portRowW D.unroll w.2.1, w.2.2.val⟩
        if h : g.val < C.N then
          placeFrom C rows D.unroll L (Flock.Bridge.rootsOfL D.unroll L) (Flock.Bridge.posOf C ⟨g.val, h⟩)
        else none
      else none

end Verity.Sampled
