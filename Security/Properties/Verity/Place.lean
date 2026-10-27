import Properties.Verity.Game

/-!
# Where the committed files open each gate (`Verity.FileRead.place`)

The vocabulary's positions (`Vocab.place`, ONE-7) at the committed files as v2 reads them. Gate `g` of the circuit the
files describe (`FileRead.circuit`) is bit `k` of unit `u`'s row of port `q` (`HiddenExec.rowOfH`, the game's own row
map, read off the circuit file), and its place (`placeAt`) is that row's leaf as the spec's row-leaf verifier opens it:

* the verifier: the port's name as the frame-v3 domain (`Bridge.domOf`), with the population's `n` leaves, the default
  hm96 key (`Hm96.Default512.key`) and the port's format (`Bridge.Openings.formatOf`);
* the root: the port's registered frame-v3 root, over the public file's `b ‖ c` at each unit (`Bridge.leavesOf`);
* the rank `u` and the offset `k`.

A gate no row holds has no place, and neither does any gate of files that don't read. The place is a function of the
files alone: the registration's roots are the public file's, so neither the parameters nor the commitment move it.
`Flock.Bridge.Openings.place_opens` is the opening that a live session's deployed commitment gives at it.
-/

namespace Flock.Bridge.Openings

/-- **A port's format**: `sha512/row-seg/v1` at a segmented port, `sha512/row/v2` at a bit row, `sha512/row/v1`
otherwise, as C-Flock frames the port's rows (`Registered.portSchema`) and hashes them (`rowMsg`). -/
def formatOf (p : Flock.Port) : Hash.RowLeaf.Format :=
  if p.segs.isSome then .seg else if p.v2 then .v2 else .v1

end Flock.Bridge.Openings

namespace Verity

open FlockSoundness.Discharge in
/-- **The place of gate `g`** of the circuit file `D` (its unit built, `hU`) with `n` units, under the public file
`pub`: the leaf of its row (`HiddenExec.rowOfH`) under its port's verifier and registered root, at the row's unit and
the gate's bit in the row. -/
noncomputable def placeAt (D : Flock.Desc) (hU : FlockSoundness.Built D.unroll.unit) (n : ℕ) (pub : Flock.Public)
    (g : Fin (Layout.keyN D.unroll hU n)) : Option Place :=
  (HiddenExec.rowOfH D.unroll hU n (Layout.outsOf D.unroll hU n) g).map fun x =>
    { verifier := ⟨⟨(Flock.Bridge.domOf D.unroll x.2.1).data.toList, n, fun k => k⟩,
        Flock.Hm96.Default512.key.data.toList, Flock.Bridge.Openings.formatOf D.unroll.ports[x.2.1]⟩
      root := fun i => (Flock.FV3.root (Flock.Bridge.domOf D.unroll x.2.1) (Flock.Bridge.leavesOf D.unroll pub n x.2.1)
        Flock.Sha512.hash).data.getD i.val 0
      rank := x.1.val
      offset := x.2.2.val }

/-- **A parse's places**, at the public file each descriptor reads (`pubOf`); none at files that don't parse. -/
noncomputable def parsedPlace (pubOf : Flock.Desc → Option Flock.Public) :
    (o : Option Flock.Bridge.Parsed) → Fin (Flock.Bridge.Parsed.circuit o).N → Option Place
  | some r, g => (pubOf r.D).bind fun pub => placeAt r.D r.hU r.n pub g
  | none, _ => none

/-- **The committed files' places** (`Vocab.place` at them): `placeAt` at what the circuit file parses to and the public
file it reads, as `Flock.Bridge.parsedOf` reads them. -/
noncomputable def FileRead.place (r : FileRead) : Fin r.circuit.N → Option Place :=
  parsedPlace (fun D => (Flock.HmOut.loadPublic r.tags D.unroll r.publicFile).toOption)
    (Flock.Bridge.parsedOf r.tags r.circuitFile r.tables r.publicFile)

end Verity
