import Proofs.Accounting.TS.Session.Checked.Layer
import Proofs.Accounting.TS.Session.Held.Known
import Definitions.Accounting.Served

/-!
# A checked unit's proof layer with its served checks, through its C-Flock sessions

`Accounting.TS.Checked.checkedLayer`'s relation asks only that a port named `ok` exist. This layer, at the same
sessions, takes two steps further:

* (a) **The `ok` rows are the verifier's.** flock-verify `--registered F` registers the `ok` port at known content
  (`Registered.KNOWN_KEY`: h1_prove's rows, each the one word `1` under the public `ok_salt`), and refuses a file whose
  `ok` root is not the root of those rows (`Registered.checkKnownAll`). The layer's acceptance is flock-verify's,
  that check included. On an accepted session the registrant's row at every read of a port named `ok` has the
  `sha512/row/v2` bytes of the word `1` (`OkVal`), or the registrant's openings exhibit a SHA-512 collision
  (`OkColl`, a fact of the session's commit-time openings, with both preimages in hand: `FlockC3.RegLevels.knownColl`);
* (b) **The served checks hold.** The relation (`HeldRel`) is `CheckedRel` and, unless `OkColl`, an assignment the
  circuit's evaluation agrees with on every drawn unit's wires at which every drawn unit's checks hold in the declared
  circuit (`Accounting.Served.Held` of `Cd`, at the unit `unitOf u` and the values `dec vals`).

What ties the C-Flock circuit to `Cd` is the premise `Accounting.Assumptions.TS.OkRealizes`: on an assignment the key
program's evaluation agrees with at unit `u`, whose `ok` wires carry the `ok` row, `Cd`'s checks of `unitOf u` hold.
Unit `u` of the session is one unit of `Cd`, and its partials are proof-layer values.

These definitions read `RowsSession`, in the proofs package, so they sit here under `reads_exempt`'s
`Proofs.Accounting.TS.Session`, as `Accounting.TS.Checked`'s do.
-/

namespace Accounting.TS.Held

open Accounting.TS Accounting.TS.Checked Flock.SoundAtBridge FlockSoundness.Discharge.ZkReg
open Pouw.RelativeSecurity (Paths)
open FlockSoundness.Refine (H512)

/-- The `ok` port's row: the one u16 word `1` (`h1unit.register_ok`). -/
def okRow : Array ℕ := #[1]

/-- The session's row encodings (`rsShHJS`), the ones its registrants' openings are of. -/
noncomputable abbrev rsOf (s : RowsSession) :=
  rsShHJS (Reg := s.Reg) s.dj s.y₀ s.ht s.hc s.hscope s.hU s.hpub

/-- **Wire `g` of unit `u` is an `ok` wire**: a wire of `u` (`P.io`) at a registered read of `rdU u` whose registered
port is named `ok`. -/
def OkWire (s : RowsSession) (ok : String) (u : Fin s.n) (g : Fin s.Circ.N) : Prop :=
  g ∈ s.P.io u ∧ (s.rdU u).isRead (s.vb.pos g) ∧ ((s.rdU u).port (s.vb.pos g)).name = ok

/-- **`v` is the `ok` row at the row position `p`**: its row bytes there (`rsOf`'s encoding, `sha512/row/v2` at a v2
row port) are the `ok` row's (`Registered.rowBytesV2 okRow`). -/
def OkVal (s : RowsSession) (p : s.vb.Pos) (v : s.vb.Val) : Prop :=
  (rsOf s).enc p v = (Flock.Registered.rowBytesV2 okRow).data.toList

/-- **The `ok` tie collides at the session**: at some registered read `p` of some unit's registrant, at a port named
`ok`, the known-content extractor against the `ok` port's known rows `ks` (`knownColl`) outputs two different SHA-512
inputs with one digest. -/
def OkColl (s : RowsSession) (ok : String) (ks : Array (Array ℕ × ByteArray)) : Prop :=
  ∃ (u : Fin s.n) (p : s.vb.Pos), (s.rdU u).isRead p ∧ ((s.rdU u).port p).name = ok ∧ ∃ a b,
    FlockC3.RegLevels.knownColl ((s.rdU u).port p) ks ((s.rdU u).readAt p) ((rsOf s).enc p ((s.rdU u).wv p))
      ((s.rdU u).ws p) ((s.rdU u).wpath p) = some (a, b) ∧ a ≠ b ∧ H512 a = H512 b

/-- **The checked unit's relation with its served checks**: `CheckedRel`, and the layer value `v` is the registered
reads of one assignment the circuit's evaluation agrees with on every wire of every drawn unit of `o`, at which every
drawn unit `u`'s checks hold in the declared circuit `Cd` (`Accounting.Served.Held` at `unitOf u` and `dec vals`); or
the `ok` tie collides (`OkColl`) at the `ok` port's known rows in `known`. -/
def HeldRel {ι : Type} (s : RowsSession) (o : s.Out) (ok : String) (Cd : Accounting.Compose.Declared ι)
    (cls : ℕ → Accounting.Served.Cls) (unitOf : Fin s.n → ℕ) (dec : (Fin s.Circ.N → Bool) → ι → ℤ)
    (known : Array (String × Array (Array ℕ × ByteArray))) (v : ℕ → ℕ → Option Bool) : Prop :=
  CheckedRel s o ok v ∧
    ((∃ vals : Fin s.Circ.N → Bool, v = rowsReg s vals ∧
        (∀ u ∈ o.2.1, ∀ g ∈ s.P.io u, s.Circ.eval vals g = vals g) ∧
        ∀ u ∈ o.2.1, Accounting.Served.Held Cd cls (unitOf u) (dec vals)) ∨
      ∃ ks, (ok, ks) ∈ known ∧ OkColl s ok ks)

/-- **The held layer through its C-Flock sessions**: proof `i` of round `w` is accepted where flock-verify
`--zk --registered` accepts its session at the registered file's known content `known x w i` (`LiveAcceptsZKR`, and
`Registered.checkKnownAll`, which `flock-verify` runs on `--registered F`), opens the committed rows at the registered
reads (`openedReg`), and its relation is `HeldRel` at round `w`'s declared circuit `Cd w`. -/
noncomputable def heldLayer {ι Ω : Type} {S nπ : ℕ} (sess : Paths Ω S → Fin S → Fin nπ → RowsSession)
    (out : ∀ x w i, (sess x w i).Out) (ok : String)
    (known : Paths Ω S → Fin S → Fin nπ → Array (String × Array (Array ℕ × ByteArray)))
    (Cd : Fin S → Accounting.Compose.Declared ι) (cls : Fin S → ℕ → Accounting.Served.Cls)
    (unitOf : ∀ x w i, Fin (sess x w i).n → ℕ) (dec : ∀ x w i, (Fin (sess x w i).Circ.N → Bool) → ι → ℤ)
    (right : Paths Ω S → (Fin S → Fin nπ → ℕ → ℕ → Option Bool) → Prop) :
    ProofLayer Ω S nπ (ℕ → ℕ → Option Bool) where
  accepted x w i := LiveAcceptsZKR (sess x w i).I (sess x w i).own (out x w i) ∧
    Flock.Registered.checkKnownAll (sess x w i).own (known x w i) = .ok ()
  opened x w i := openedReg (sess x w i)
  rel x w i v := HeldRel (sess x w i) (out x w i) ok (Cd w) (cls w) (unitOf x w i) (dec x w i) (known x w i) v
  right := right

end Accounting.TS.Held
