import Proofs.Accounting.TS.Session.Assumptions

/-!
# A checked unit's proof layer through its C-Flock sessions

The proof layer a checked record (`Accounting.Served`'s `Held`, at `PouwH1TileZy_v1`) is realized by, when proof `i` of
round `w` on the path `x` is the C-Flock session `sess x w i` (a `RowsSession`) with outcome `out x w i`:

* `accepted`: flock-verify `--zk --registered` accepts the session (`LiveAcceptsZKR`), so the proof system's rule
  (`Accounting.Assumptions.TS.AcceptedLive`) holds of it by definition;
* `opened`: the registered rows, the committed trees' own: at each unit `u` and each wire of `u` at a registered read
  of `rdU u` (the unit's `zy` and A rows, its registered B, and its `ok` row), the bit of the registrant's opening
  there; nothing elsewhere (`openedReg`);
* `rel`: the unit's Definition with its check word fixed (`CheckedRel`): the circuit's port named `ok` is a port of
  `own`, and the opened rows are, at every registered read, one wire assignment that the circuit's own evaluation
  agrees with on every wire each drawn unit reads or computes (`Circuit.eval`, the whole key program on that
  assignment's inputs). The `ok` port's registrant is the verifier, whose rows are the one word `1` under the public
  `ok_salt`, so on an accepted proof the relation holds the `ok` wires the circuit computes from the opened `zy`, A
  and B rows to `1`;
* `right`: the caller's.

`rowsReg s vals` reads an assignment `vals` at the same registered reads as `openedReg`, so `rowsReg s s.Xv` is the
session's value layer (`XplurZC`) at them, the `rows` `ServedChecksHold` and `ServedRootsLinked` read.

These definitions read `RowsSession`, in the proofs package, so they sit here under `reads_exempt`'s
`Proofs.Accounting.TS.Session`, as `Accounting.Assumptions.TS`'s do.
-/

namespace Accounting.TS.Checked

open Accounting.TS Flock.SoundAtBridge FlockSoundness.Discharge.ZkReg
open Pouw.RelativeSecurity (Paths)

open Classical in
/-- **A session's registered reads, as a layer value**: at unit `a` and wire `b`, `f` of them where `b` is a wire of
unit `a` (`P.io`) at a registered read of `rdU a`; `none` elsewhere. -/
noncomputable def regRead (s : RowsSession) (f : Fin s.n → Fin s.Circ.N → Bool) : ℕ → ℕ → Option Bool :=
  fun a b =>
    if h : a < s.n ∧ b < s.Circ.N then
      if (⟨b, h.2⟩ : Fin s.Circ.N) ∈ s.P.io ⟨a, h.1⟩ ∧ (s.rdU ⟨a, h.1⟩).isRead (s.vb.pos ⟨b, h.2⟩) then
        some (f ⟨a, h.1⟩ ⟨b, h.2⟩)
      else none
    else none

/-- **An assignment's registered reads**: `vals` at every registered read of every unit. -/
noncomputable def rowsReg (s : RowsSession) (vals : Fin s.Circ.N → Bool) : ℕ → ℕ → Option Bool :=
  regRead s fun _ g => vals g

/-- **The committed rows at the registered reads**: the bit of the registrant's opening of unit `u`'s row at each of
its wires there (`RowsSession.wo`, the commit-time opening the service registers with the root). -/
noncomputable def openedReg (s : RowsSession) : ℕ → ℕ → Option Bool :=
  regRead s fun u g => s.vb.bit (s.vb.val (s.wo u (s.vb.pos g))) g

/-- **The checked unit's relation at `ok`**: the circuit has a port named `ok`, `own` registers a port of that name,
and the layer value `v` is the registered reads of one wire assignment the circuit's evaluation agrees with on every
wire of every drawn unit of the outcome `o`. -/
def CheckedRel (s : RowsSession) (o : s.Out) (ok : String) (v : ℕ → ℕ → Option Bool) : Prop :=
  ((∃ q : Fin s.c.ports.size, s.c.ports[q].name = ok) ∧ ∃ pt ∈ s.own, pt.name = ok) ∧
    ∃ vals : Fin s.Circ.N → Bool, v = rowsReg s vals ∧ ∀ u ∈ o.2.1, ∀ g ∈ s.P.io u, s.Circ.eval vals g = vals g

/-- **The checked layer through its C-Flock sessions**: proof `i` of round `w` is accepted where flock-verify
`--zk --registered` accepts its session, opens the committed rows at the registered reads (`openedReg`), and its
relation is the unit's Definition with its check word `ok` registered (`CheckedRel`). -/
noncomputable def checkedLayer {Ω : Type} {S nπ : ℕ} (sess : Paths Ω S → Fin S → Fin nπ → RowsSession)
    (out : ∀ x w i, (sess x w i).Out) (ok : String)
    (right : Paths Ω S → (Fin S → Fin nπ → ℕ → ℕ → Option Bool) → Prop) :
    ProofLayer Ω S nπ (ℕ → ℕ → Option Bool) where
  accepted x w i := LiveAcceptsZKR (sess x w i).I (sess x w i).own (out x w i)
  opened x w i := openedReg (sess x w i)
  rel x w i v := CheckedRel (sess x w i) (out x w i) ok v
  right := right

end Accounting.TS.Checked
