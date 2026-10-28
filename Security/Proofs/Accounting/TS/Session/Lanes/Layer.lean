import Proofs.Accounting.TS.Session.Checked.Layer
import Proofs.Accounting.TS.Session.Held.Layer
import Proofs.Flock.Recursive.Registrants
import Flock.KnownRead
import PouwWindow.OwnPorts
import PouwWindow.FullDraw

/-!
# A leaf's proof layer through its 33 linked C-Flock sessions (option A)

Under option A a drawn 32 × 32 H1 leaf is proved as 33 statements (`h1zy`, SERVE-REG): its 32 lanes
`PouwH1LaneZy_v1{TILE}`, lane `l` reading its 4 A rows by row slot, its 8 B rows by column slot, the leaf's admit bits,
its whole z/y record, row `l` of the leaf's partials, the salt, position and roots, and the index `l`, and returning
`ok`; and their combine `PouwH1Combine_v1{TILE}`, reading the 32 partials rows and the leaf's receipt and returning `ok`.

The leaf is one unit of the declared circuit: the 33 statements' `ok` words are that unit's
`check` words, and its partials are proof-layer values, no values of the declared circuit. They are committed once,
after the draw, under hiding leaves, read only by the unit's own 33 statements, which all open that one commitment,
and they are what the unit's checks are proved through, not inputs it is credited on. So the unit's served checks
(`UnitServed`) read its A and B rows, its record and its receipt, and no partials row.

This file states, on words, what the 33 statements check and what one unit's served checks are:

* `rowOf`, `colOf`: lane `l = 4g + q`'s row slot `s` is row `g + 8s` and its column slot `t` column
  `2q + 8⌊t/2⌋ + (t mod 2)` (`h1tile.TileShape.rows`, `cols` at `R = C = 32`), so its cells are `h1zy.lane_cells`;
* `LaneView`, `LaneOk`: what lane `l`'s statement reads, and its `ok = 1`: its partials row is the honest one
  (`part`, `h1zy.lane_partial`'s words, each u32, at its rows, the public inputs and `l`) and each of its 32 cells has
  `z = Σ_k a_k b_k` and `y = dq [z, s_a, s_w]` at its row's and column's scales (`PouwZyCheck_v1{4, 8}`);
* `CombineView`, `CombineOk`: what the combine reads, and its `ok = 1`: the XOR, in lane order, of the 32 rows is the
  receipt word by word (`xorAll`);
* `PortRow`, `portRead`: a wire assignment at every bit of every unit's port rows, registered or not, which the
  circuits' premise reads (`Accounting.Assumptions.TS.LeafChecks`);
* `UnitServed`: the leaf's served checks, `Accounting.Served`'s `ProductOf` and `DequantOf` at every cell (`CellOk`,
  with `dq` the Program's dequant on its words, `PouwZyDequant_v2` today) and its receipt the XOR of its lanes' honest
  partials (in the order `lane_partials` splits the tile unit's receipt);
* `SameTree`, `RowsAgree`, `LeafColl`: the 33 sessions' registrants' rows (`RowsSession.rdU`, each the commit-time
  opening its session registers) are one row's bytes wherever two of them read one registered tree (root, leaf count,
  domain and schema, under any port name) at one position; or two of the
  registrants' records clash (`ReadsX.Clash`: their openings, or hm96's leaves over their rows, are two different
  SHA-512 inputs with one digest, both in hand). Each opening opens its root there (`ReadsX.committed`), so one of the
  two holds (`Accounting.TS.Lanes.rowsAgree_or_leafColl`);
* `KnownHeld`, `LeafKnownColl`: the known-content tie (at every port the verifier registers at known content:
  each lane's `admit` and `pub`, every statement's `o0`). Where flock-verify's known-content check
  (`Registered.checkKnownAll`) accepts a session's `own` at its known content, the registrant's row at every read of
  such a port is the bytes of the known row at its position (`KnownHeld`), or the registrant's openings exhibit a
  SHA-512 collision (`Accounting.TS.Held.OkColl` at that port's name and rows; `LeafKnownColl` at one of the 33);
* `leafLayer`: the unit's proof layer. Proof `i` of round `w` is accepted where flock-verify `--zk --registered`
  accepts every one of its 33 sessions (`sess x w i (some l)` lane `l`'s, `sess x w i none` the combine's), its
  known-content check (`known x w i k`, the session's registered file's known content), its complete check (the
  file sets `complete`, `Registered.checkComplete`) and its check that each known port, and each port the file marks
  `read_in_full` (`full x w i k`), is a row of its words whose every bit the circuit reads or returns
  (`KnownRead.check`) included, and the leaf verifier's checks of its registered file and of its session's draw
  (`pouw-leaf-own`: `PouwWindow.Lanes.checkOwn`, and `PouwWindow.Lanes.checkFull`, every unit drawn) pass, opens the
  leaf's values committed before σ_w (`Leaf`: its A and B rows with their scales and its z/y record), and its relation is
  `UnitServed` at the path's public inputs and receipt, which are no leaves of the round (`ProofLayer`), or `LeafColl`
  or `LeafKnownColl` at the proof's sessions.

Words are integers as `Accounting.Served` reads them (codes s8, z s32, scales and y their encodings); partial and
receipt words are `BitVec 32`. These definitions sit beside the checked layer's, under `reads_exempt`'s
`Proofs.Accounting.TS.Session`.
-/

namespace Accounting.TS.Lanes

open Accounting.TS Accounting.TS.Checked Flock.SoundAtBridge FlockSoundness.Discharge.ZkReg
open Pouw.RelativeSecurity (Paths)

/-- Lane `l`'s row at row slot `s < 4`: `g + 8s` with `g = ⌊l/4⌋`. -/
def rowOf (l : Fin 32) (s : Fin 4) : Fin 32 :=
  ⟨l.val / 4 + 8 * s.val, by have := l.isLt; have := s.isLt; omega⟩

/-- Lane `l`'s column at column slot `t < 8`: `2q + 8⌊t/2⌋ + (t mod 2)` with `q = l mod 4`. -/
def colOf (l : Fin 32) (t : Fin 8) : Fin 32 :=
  ⟨2 * (l.val % 4) + 8 * (t.val / 2) + t.val % 2, by have := t.isLt; omega⟩

/-- An A or B row: its `K` codes, then its scale. -/
structure Row (K : ℕ) where
  codes : Fin K → ℤ
  scale : ℤ

/-- **A leaf's values committed before σ_w**: its 32 A rows, its 32 B rows and its z/y record, row-major. -/
structure Leaf (K : ℕ) where
  a : Fin 32 → Row K
  b : Fin 32 → Row K
  z : Fin 32 → Fin 32 → ℤ
  y : Fin 32 → Fin 32 → ℤ

/-- Lane `l`'s A rows by row slot. -/
def laneA {K : ℕ} (L : Leaf K) (l : Fin 32) : Fin 4 → Row K := fun s => L.a (rowOf l s)

/-- Lane `l`'s B rows by column slot. -/
def laneB {K : ℕ} (L : Leaf K) (l : Fin 32) : Fin 8 → Row K := fun t => L.b (colOf l t)

/-- The XOR, in lane order, of 32 rows of `nw` words, word by word. -/
def xorAll {nw : ℕ} (p : Fin 32 → Fin nw → BitVec 32) : Fin nw → BitVec 32 :=
  fun w => (List.ofFn fun l => p l w).foldl (· ^^^ ·) 0

/-- **Cell `(i, j)` of a leaf is served right**: its `z` is the dot of its A row's and B row's codes, and its `y` is
`dq` of its `z` at the A row's and the B row's scales. -/
def CellOk {K : ℕ} (dq : List ℤ → ℤ) (L : Leaf K) (i j : Fin 32) : Prop :=
  L.z i j = ∑ k, (L.a i).codes k * (L.b j).codes k ∧ L.y i j = dq [L.z i j, (L.a i).scale, (L.b j).scale]

/-- **The unit's served checks**: every cell of the leaf is right (`CellOk`), and its receipt is the XOR, in lane order,
of its lanes' honest partials (`part` at the public inputs `pub`, each lane's index and its rows). -/
def UnitServed {K nw : ℕ} {Pub : Type} (dq : List ℤ → ℤ)
    (part : Pub → Fin 32 → (Fin 4 → Row K) → (Fin 8 → Row K) → Fin nw → BitVec 32) (pub : Pub)
    (receipt : Fin nw → BitVec 32) (L : Leaf K) : Prop :=
  (∀ i j, CellOk dq L i j) ∧ receipt = xorAll fun l => part pub l (laneA L l) (laneB L l)

/-- **What lane `l`'s statement reads**: its A rows by row slot, its B rows by column slot, the leaf's record, its
partials row, the public inputs (admit, salt, position, roots), its index and its `ok` word. -/
structure LaneView (K nw : ℕ) (Pub : Type) where
  a : Fin 4 → Row K
  b : Fin 8 → Row K
  z : Fin 32 → Fin 32 → ℤ
  y : Fin 32 → Fin 32 → ℤ
  part : Fin nw → BitVec 32
  pub : Pub
  lane : Fin 32
  ok : Bool

/-- **Lane `ok = 1`**: its partials row is the honest one at its rows, the public inputs and its index, and each of its
32 cells, `(rowOf lane s, colOf lane t)`, has `z` the dot of its rows' codes and `y` its `dq`. -/
def LaneOk {K nw : ℕ} {Pub : Type} (dq : List ℤ → ℤ)
    (part : Pub → Fin 32 → (Fin 4 → Row K) → (Fin 8 → Row K) → Fin nw → BitVec 32) (v : LaneView K nw Pub) : Prop :=
  v.part = part v.pub v.lane v.a v.b ∧
    ∀ s t, v.z (rowOf v.lane s) (colOf v.lane t) = ∑ k, (v.a s).codes k * (v.b t).codes k ∧
      v.y (rowOf v.lane s) (colOf v.lane t) = dq [v.z (rowOf v.lane s) (colOf v.lane t), (v.a s).scale, (v.b t).scale]

/-- Lane `l`'s view of a leaf `L`, its public inputs `pub` and its partials row `p`, with its `ok` word `1`. -/
def laneView {K nw : ℕ} {Pub : Type} (L : Leaf K) (pub : Pub) (p : Fin nw → BitVec 32) (l : Fin 32) :
    LaneView K nw Pub :=
  ⟨laneA L l, laneB L l, L.z, L.y, p, pub, l, true⟩

/-- **What the combine reads**: the 32 partials rows, lane-major, the leaf's receipt and its `ok` word. -/
structure CombineView (nw : ℕ) where
  parts : Fin 32 → Fin nw → BitVec 32
  receipt : Fin nw → BitVec 32
  ok : Bool

/-- **Combine `ok = 1`**: the rows' XOR, in lane order, is the receipt word by word. -/
def CombineOk {nw : ℕ} (v : CombineView nw) : Prop := xorAll v.parts = v.receipt

/-- **Wire `g` is a bit of one of unit `u`'s port rows**: its commit string is `u`'s row of some port `q` of the
circuit, an input port or an output row, not a single gate's own string (`Readers.ofRows`). -/
def PortRow (s : RowsSession) (u : Fin s.n) (g : Fin s.Circ.N) : Prop :=
  ∃ q : Fin s.c.ports.size, s.vb.pos g = Sum.inl (u, q)

open Classical in
/-- **An assignment's port rows**: `vals` at every wire of unit `a` that is a bit of one of `a`'s port rows (`PortRow`),
registered or not; `none` elsewhere. Where every port is registered it is `rowsReg`
(`RowsSession.rowsReg_eq_portRead`). -/
noncomputable def portRead (s : RowsSession) (vals : Fin s.Circ.N → Bool) : ℕ → ℕ → Option Bool :=
  fun a b =>
    if h : a < s.n ∧ b < s.Circ.N then
      if (⟨b, h.2⟩ : Fin s.Circ.N) ∈ s.P.io ⟨a, h.1⟩ ∧ PortRow s ⟨a, h.1⟩ ⟨b, h.2⟩ then some (vals ⟨b, h.2⟩)
      else none
    else none

/-- The session's row encodings (`rsShHJS`), the ones its registrants' openings are of. -/
noncomputable abbrev rsOf (s : RowsSession) :=
  rsShHJS (Reg := s.Reg) s.dj s.y₀ s.ht s.hc s.hscope s.hU s.hpub

/-- **Two registered ports name one tree**: one root, leaf count, domain and leaf schema, all that
`Flock.Registered.opensLeaf` reads of a port. Two statements' ports of one value (lane `l`'s `partials` and the
combine's `p<l>`) have different names and one tree. -/
def SameTree (a b : Flock.Registered.Port) : Prop :=
  a.root = b.root ∧ a.leaves = b.leaves ∧ a.domain = b.domain ∧ a.schema = b.schema

/-- **A leaf's sessions' registrants' rows agree at one tree and position**: wherever registered reads `p` of unit
`u`'s registrant in session `k` and `p'` of unit `u'`'s in session `k'` name one registered tree (`SameTree`) and one
position, the two registrants' rows there are one row's bytes, each at its session's encoding. -/
def RowsAgree (sess : Option (Fin 32) → RowsSession) : Prop :=
  ∀ (k k' : Option (Fin 32)) (u : Fin (sess k).n) (u' : Fin (sess k').n) (p : (sess k).vb.Pos)
    (p' : (sess k').vb.Pos), ((sess k).rdU u).isRead p → ((sess k').rdU u').isRead p' →
    SameTree (((sess k).rdU u).port p) (((sess k').rdU u').port p') →
    ((sess k).rdU u).readAt p = ((sess k').rdU u').readAt p' →
    (rsOf (sess k)).enc p (((sess k).rdU u).wv p) = (rsOf (sess k')).enc p' (((sess k').rdU u').wv p')

/-- **Two of a leaf's sessions' registrants' records clash**: some unit's registrant in session `k` and some unit's in
session `k'` (the same session or not) have registered reads whose openings, or hm96's leaves over their rows, are two
different SHA-512 inputs with one digest (`ReadsX.Clash`). -/
def LeafColl (sess : Option (Fin 32) → RowsSession) : Prop :=
  ∃ (k k' : Option (Fin 32)) (u : Fin (sess k).n) (u' : Fin (sess k').n), ((sess k).rdU u).Clash ((sess k').rdU u')

/-- A registered file's known content (`Registered.KNOWN_KEY`): each port it names, with its rows' u16 words and salts. -/
abbrev Known := Array (String × Array (Array ℕ × ByteArray))

/-- **The known rows hold at a session**: at every registered read `p` of every unit's registrant whose registered port
is one the known content `known` names, `p`'s position is one of the known rows' and the registrant's row bytes there
(`rsOf`'s encoding) are the `sha512/row/v2` bytes of the known row at that position. -/
def KnownHeld (s : RowsSession) (known : Known) : Prop :=
  ∀ nm ks, (nm, ks) ∈ known → ∀ (u : Fin s.n) (p : s.vb.Pos), (s.rdU u).isRead p → ((s.rdU u).port p).name = nm →
    (s.rdU u).readAt p < ks.size ∧
      (rsOf s).enc p ((s.rdU u).wv p) = (Flock.Registered.rowBytesV2 ks[(s.rdU u).readAt p]!.1).data.toList

/-- **A leaf's known-content tie collides**: at one of its 33 sessions `k`, at a port `nm` its known content names with
rows `ks`, the known-content extractor outputs two different SHA-512 inputs with one digest
(`Accounting.TS.Held.OkColl` at `nm`, the `ok` tie at any known port). -/
def LeafKnownColl (sess : Option (Fin 32) → RowsSession) (known : Option (Fin 32) → Known) : Prop :=
  ∃ k nm ks, (nm, ks) ∈ known k ∧ Accounting.TS.Held.OkColl (sess k) nm ks

/-- **The leaf's proof layer through its 33 sessions**: proof `i` of round `w` is accepted where flock-verify
`--zk --registered` accepts every one of its sessions (`sess x w i (some l)` lane `l`'s, `sess x w i none` the
combine's), its known-content check at the session's known content `known x w i k` included
(`Registered.checkKnownAll`, which `flock-verify` runs on `--registered F`), its complete check
(`Registered.checkComplete`, which it runs where `F` sets `complete`: every port of the circuit is a port of `own`) and
its known-read check (`KnownRead.check`, which it runs on every statement at `F`'s known content, then at each port
`full x w i k` names, the ports `F` marks `read_in_full`, as one row of its registered row's words
(`KnownRead.fullRows`): each such port is a `sha512/row/v2` row of those words, and a computed row of the unit reads each
bit of such an input row, a copy returns each bit of such an output row) and the leaf verifier's check of its file
(`PouwWindow.Lanes.checkOwn`, which `pouw-leaf-own` runs: the file holds the ports `want x w i k`, the statement's data
ports at the accepted registration's and produced record's roots) and of its session's draw
(`PouwWindow.Lanes.checkFull`, which `pouw-leaf-own` runs on the session's record: the draw is `subset` of every unit of
a non-empty population, the verifier's draw policy for a leaf's statements) and of its file's reads
(`PouwWindow.Lanes.checkOneRead`, which `pouw-leaf-own` runs too: every port of the file has one read) and of the
leaf's tile (`PouwWindow.Lanes.Tile.check` at 32 × 32 and `K`, which `pouw-leaf-own` runs first, in `lane?` and
`combine?`: among others `32 ∣ K` and `K < 2^17`, where the lanes' s8 accumulator binds),
opens the leaf's values committed before σ_w (`leaf`), and its relation is the unit's served checks (`UnitServed`) at
the path's public inputs and receipt, or
two of its sessions' registrants' records clash (`LeafColl`), or a session's known-content tie collides
(`LeafKnownColl`). -/
noncomputable def leafLayer {Ω : Type} {S nπ K nw : ℕ} {Pub : Type}
    (sess : Paths Ω S → Fin S → Fin nπ → Option (Fin 32) → RowsSession) (out : ∀ x w i k, (sess x w i k).Out)
    (known : Paths Ω S → Fin S → Fin nπ → Option (Fin 32) → Known)
    (full : Paths Ω S → Fin S → Fin nπ → Option (Fin 32) → Array String)
    (want : Paths Ω S → Fin S → Fin nπ → Option (Fin 32) → List Flock.Registered.Port)
    (dq : List ℤ → ℤ) (part : Pub → Fin 32 → (Fin 4 → Row K) → (Fin 8 → Row K) → Fin nw → BitVec 32)
    (pub : Paths Ω S → Fin S → Fin nπ → Pub) (receipt : Paths Ω S → Fin S → Fin nπ → Fin nw → BitVec 32)
    (leaf : Paths Ω S → Fin S → Fin nπ → Leaf K)
    (right : Paths Ω S → (Fin S → Fin nπ → Leaf K) → Prop) : ProofLayer Ω S nπ (Leaf K) where
  accepted x w i := ∀ k, LiveAcceptsZKR (sess x w i k).I (sess x w i k).own (out x w i k) ∧
    Flock.Registered.checkKnownAll (sess x w i k).own (known x w i k) = .ok () ∧
    Flock.Registered.checkComplete (sess x w i k).c (sess x w i k).own = .ok () ∧
    Flock.KnownRead.check (sess x w i k).c
      (known x w i k ++ Flock.KnownRead.fullRows (sess x w i k).own (full x w i k)) = .ok () ∧
    PouwWindow.Lanes.checkOwn (want x w i k) (sess x w i k).own = .ok () ∧
    PouwWindow.Lanes.checkFull (out x w i k).2.2.record = .ok () ∧
    PouwWindow.Lanes.checkOneRead (sess x w i k).own = .ok () ∧
    PouwWindow.Lanes.Tile.check ⟨32, 32, K⟩ = .ok ()
  opened := leaf
  rel x w i v := UnitServed dq part (pub x w i) (receipt x w i) v ∨ LeafColl (sess x w i) ∨
    LeafKnownColl (sess x w i) (known x w i)
  right := right

end Accounting.TS.Lanes
