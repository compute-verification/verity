import Proofs.Flock.Claims.Recursion.VStarBDeploy
import Proofs.Flock.Claims.Recursion.BoundReaders
import Flock.HmRow

/-!
# V*'s bound layout off the registered ports, at affine forms (`layoutBDeployed`)

`VStarB.LayoutB` reads messages and calls at affine forms over the hidden values (`VStarB.readForms`; a position is the
one-wire form, `readForms_pos`, and `readForms` is `VStar.readLin`, `readForms_eq_readLin`). This file reads the bound
layout off the compiled statement at V*'s registered message-round ports, as `layoutOf` (`DeployLayout.lean`) reads
the plain one, but at the representation `Bound.IVStar` binds:

* **Rounds.** The deployed plain verifier's (`layoutOf.rounds`): round 0, the link message; then the two reps' stream
  rounds; then the last message.
* **A stream round's commit strings** (`streamRound`, `noTops`). At `IVStar`, round `j + 1` (rep stream round `j`) is
  the concatenation of its commit strings (`Bound.nAt`): its tops' nodes and its copies of level 0's top, then one for
  its free elements `F_k` where it absorbs a message (`RecVStar.SchedRound`). A round with none carries the empty
  message, which the statement fixes (`fixedAt`, `[]`, not `layoutOf`'s framed `[0]`).
* **Held rounds** (`msgAt`). A stream round whose only string is `F_k`'s is held whole by the part whose port under
  `rec-c<k>` reads that rep's row (`heldRows`, as `partHolds`): the commit string `b ‖ c` of that row, the 1024 output
  bits of the port's hm96 slot (`hmCommit`), slot `q` of the instance's VU in the hm96 range (`VStar.rowWires` wires
  port `q`'s chain into hm96 slot `q`), each at its output column's hidden value (the slot's output rows are copy rows
  `form · 1`, `HmNets.layout`).
* **Calls** (`shaAt`, `hmCalls`). Each held port's two hm96 evaluations: the salt's, `SHA-512(p_salt ‖ y) = c`, the
  salt `y` at the slot's input columns and `c` the commit string's last 64 bytes; and the row's,
  `SHA-512(rowIn row) = d`, the row at its unit input columns (`rowHidden`) after its `sha512/row/v1` prefix, and `d`
  an XOR form, `b ⊕ Φy` (`hmDigest`: `HmNets.maskBit`'s mask, read backwards). The digest `d` is no column of the
  circuit, which is why the layout reads at forms.
* **Pieces.** None. The last round's pieces are its rows and salts, then the element-level last message, at the byte
  offsets the bound record's `final` fixes; `KSBoundRoute` leaves `final` free, so they are a layout at a chosen
  `final`. The held ports' rows and salts are here (`rowHidden`, `hmSalt`).

What holds of the route's layout premises here:

* `HeldB` is false at `layoutBDeployed` wherever a compiled statement of the shape parses
  (`heldB_layoutBDeployed_iff`): round 0, the level-0 cap of `root_B` (128 nodes at `fast100`), is neither fixed nor
  held nor assembled. The parts read it only at the nodes the queries reach (`rec-top-L0` at `pos[j] >>> H`), which
  depend on the coins, so no coin-independent layout holds it in the shipped V*. The same holds at each stream round
  that commits tops (a Ligerito level's first round, `cap (i + 1)` nodes) or copies level 0's top, and at the last
  round until a `final` fixes its pieces.
* `LayoutB.WholeBeforeLast` holds (`wholeBeforeLast_layoutBDeployed`), so `HeldWhole` does and READERS1's route
  (`ksBoundRoute_heldWhole`) applies at this layout.
* `PartChecksB`, `PartOpensB`, `DecompB`, `ReadersTrue` and `ReadersLogColl` are statements about these forms now.
  `ReadersTrue` asks the hm96 slots' calls to be true evaluations, which is the hm96 slot's and the row chain's
  semantics at `Flock.check` (`Discharge/Hm/{Compress,Hm96,Row}`): it no longer needs a change of representation.
  `ReadersLogColl` also asks for `nextE`'s calls (the level copies' climbs, `VStar.levelCalls_read`, which read at
  `copyForm` over the V* statement's parse, not off `x`), and is not reached by this layout.
-/

namespace Claims.Recursion.VStarB

/-- `readForms` is `VStar.readLin`, the reading `levelCalls_read` gives a level copy's calls at. -/
theorem readForms_eq_readLin (w : Flock.Wit) (fs : List _root_.Flock.HmNets.Lin) :
    readForms w fs = VStar.readLin w fs := rfl

end Claims.Recursion.VStarB

namespace RecDeploy

open Flock
open Claims (Msg)
open FlockDeploy (Coin)

/-! ## An hm96 slot's forms -/

/-- The hidden value of column `col` of the slot at in-block column `base`, at instance `n` of a part's one table (block
`n / g`, `Readable.copyCol`'s index for one table, as `rowHidden`). -/
def slotHid (c : Circuit) (n base col : ℕ) : ℕ := n / c.g * 2 ^ c.kLog + (base + col)

/-- The in-block column of port `q`'s hm96 slot at instance `n`: slot `q` of VU `n % g` in the hm96 range. -/
def hmBase (c : Circuit) (q n : ℕ) : ℕ := c.slotCol c.hmNet (n % c.g) q

/-- The hm96 net's `k`-th input port's first column in its slot: `cv`, `pad`, then the salt `y`. -/
def hmIn (c : Circuit) (k : ℕ) : ℕ := ((portCols (c.nets.getD c.hmNet default).2.inGroups)[k]?.map (·.1)).getD 0

/-- The hm96 net's output port's first column in its slot: `b ‖ c`. -/
def hmOutCol (c : Circuit) : ℕ := ((portCols (c.nets.getD c.hmNet default).2.outGroups)[0]?.map (·.1)).getD 0

/-- **Port `q`'s commit string at instance `n`**, `b ‖ c`: its hm96 slot's 1024 output bits, in byte-string order. -/
def hmCommit (c : Circuit) (q n : ℕ) : List HmNets.Lin :=
  (List.range 1024).map fun t => HmNets.Lin.var (slotHid c n (hmBase c q n) (hmOutCol c + t))

/-- **Port `q`'s salt at instance `n`**: its hm96 slot's 1536 input bits `y`, in byte-string order. -/
def hmSalt (c : Circuit) (q n : ℕ) : List HmNets.Lin :=
  (List.range 1536).map fun j => HmNets.Lin.var (slotHid c n (hmBase c q n) (hmIn c 2 + j))

/-- **Port `q`'s row digest at instance `n`**, an XOR form: hm96's `b = d ⊕ Φy`, where `Φ[i][j]` is key bit `i + j`
(`HmNets.maskBit`), so bit `i` of `d` is `b`'s bit `i` plus the salt bits `j` at which key bit `i + j` is set. -/
def hmDigest (c : Circuit) (q n : ℕ) : List HmNets.Lin :=
  (List.range 512).map fun i =>
    ⟨slotHid c n (hmBase c q n) (hmOutCol c + i) ::
      ((List.range 1536).filter fun j => Hm96.bit Hm96.Default512.key (i + j)).map fun j =>
        slotHid c n (hmBase c q n) (hmIn c 2 + j), false⟩

/-- **Port `q`'s hm96 calls at instance `n`** (port `nm`): the salt's, `SHA-512(p_salt ‖ y) = c`, `c` the commit
string's last 64 bytes; then the row's, `SHA-512(rowIn row) = d`, the row's `sha512/row/v1` prefix, then its bits at
their unit input columns (`rowHidden`). -/
def hmCalls (c : Circuit) (nm : String) (q n : ℕ) : List (Msg × List HmNets.Lin × List HmNets.Lin) :=
  [(Hm96.Default512.saltPrefix.data.toList, hmSalt c q n, (hmCommit c q n).drop 512),
    ((c.ports[q]?.map fun p => (HmRow.rowPrefix p.words).data.toList).getD [],
      ((rowHidden c nm n).getD []).map HmNets.Lin.var, hmDigest c q n)]

/-! ## The layout -/

/-- **What part `s` holds**, off its registered message-round ports, as `partHolds`: for each port under `rec-c<k>` and
each row `rep` it reads, rep `rep`'s `k`-th message round as a round of the rep streams (`msgRound`), the port's name,
its index among the part's circuit's ports, and the instance `rep`. -/
def heldRows (r : XR) (s : ℕ) : List (ℕ × String × ℕ × ℕ) :=
  match (RecVStar.statements r.shape).toOption.bind (fun vs => vs[s]?), r.w.parts[s]?.bind partCircuit,
      repSched r.shape with
  | some st, some c, some rs =>
    (RecVStar.ownPorts st).flatMap fun q => match RecVStar.cIndex q.2.1 with
      | some k => q.2.2.toList.filterMap fun row => do
        let rep ← row
        let j ← msgRound rs rep k
        let qi ← c.ports.findIdx? (·.name == q.1)
        pure (j, q.1, qi, rep)
      | none => []
  | _, _, _ => []

/-- Round `j` of the two reps' streams at the verifier's inner shape (rep `j / R`'s round `j % R`, of `R`). -/
def streamRound (x : Msg) (j : ℕ) : Option RecVStar.SchedRound :=
  (readX x).bind fun r => (repSched r.shape).bind fun rs =>
    if j < 2 * rs.size then some (rs.getD (j % rs.size) default) else none

/-- A stream round commits no cap node: no tops and no copy of level 0's top. Its commit strings are then `F_k`'s
alone, one where it absorbs a message and none where it absorbs none. -/
def noTops (sr : RecVStar.SchedRound) : Bool := sr.2.1.isEmpty && sr.2.2.1 == 0

/-- **V\*'s bound layout off the registered ports**, at forms: the deployed plain verifier's inner rounds; round `0`
neither held nor fixed; stream round `j` at round `j + 1`, fixed to the empty message where it carries no commit
string, held where its one string is `F_k`'s by the part whose port reads that rep's row, as that port's hm96 slot's
output `b ‖ c` (`hmCommit`); each held port's two hm96 calls (`hmCalls`); no pieces. -/
def layoutBDeployed : Claims.Recursion.VStarB.LayoutB where
  rounds := layoutOf.rounds
  msgAt x s i := match i with
    | 0 => none
    | j + 1 => (streamRound x j).bind fun sr =>
      if noTops sr && sr.2.2.2 then
        (readX x).bind fun r => (r.w.parts[s]?.bind partCircuit).bind fun c =>
          ((heldRows r s).find? (·.1 == j)).map fun h => [(([] : Msg), hmCommit c h.2.2.1 h.2.2.2)]
      else none
  shaAt x s := ((readX x).bind fun r => (r.w.parts[s]?.bind partCircuit).map fun c =>
    (heldRows r s).flatMap fun h => hmCalls c h.2.1 h.2.2.1 h.2.2.2).getD []
  fixedAt x i := match i with
    | 0 => none
    | j + 1 => (streamRound x j).bind fun sr => if noTops sr && !sr.2.2.2 then some [] else none

theorem layoutBDeployed_msgAt_zero (x : Msg) (s : ℕ) : layoutBDeployed.msgAt x s 0 = none := rfl

theorem layoutBDeployed_fixedAt_zero (x : Msg) : layoutBDeployed.fixedAt x 0 = none := rfl

theorem layoutBDeployed_pieceAt (x : Msg) (i : ℕ) : layoutBDeployed.pieceAt x i = [] := rfl

/-- **Only the last round could be assembled** (`WholeBeforeLast`): the layout has no pieces. -/
theorem wholeBeforeLast_layoutBDeployed : layoutBDeployed.WholeBeforeLast := fun _ _ _ => rfl

/-- **A held round reads the hm96 slot's output columns**: the forms `hmCommit` reads are one wire each, so the held
commit string is the bytes at those hidden values. -/
theorem readForms_hmCommit (w : Claims.Flock.Wit) (c : Circuit) (q n : ℕ) :
    Claims.Recursion.VStarB.readForms w (hmCommit c q n) =
      Claims.Recursion.VStar.readAt w ((List.range 1024).map fun t => slotHid c n (hmBase c q n) (hmOutCol c + t)) := by
  have h := Claims.Recursion.VStarB.readForms_pos w
    ((List.range 1024).map fun t => slotHid c n (hmBase c q n) (hmOutCol c + t))
  rw [List.map_map] at h
  exact h

/-- **`HeldB` at the deployed bound layout holds only where nothing of the shape parses**: the layout spans at least one
round (`layoutOf_rounds_pos`), and round 0, the level-0 cap of `root_B`, is neither fixed nor held by any part, and
there are no pieces. -/
theorem heldB_layoutBDeployed_iff {W : Type} (D : Claims.Recursion.Deploy Coin) (I : Claims.Recursion.Inner Coin W)
    (shape : Claims.Recursion.Form → Prop) :
    Claims.Recursion.VStarB.HeldB D I layoutBDeployed shape ↔
      ∀ (x : Msg) (st : Claims.Recursion.Stmt), D.parse x = some st → ¬ shape st.form := by
  constructor
  · intro h x st hst hf
    obtain ⟨hK, hcl, -⟩ := h x st hst hf
    have hpos : 0 < I.V.k (Claims.Recursion.xI D x) := by
      rw [hK]
      exact layoutOf_rounds_pos x
    rcases (hcl 0 hpos).1 with h0 | ⟨s, -, h0⟩ | ⟨h0, -⟩
    · rw [layoutBDeployed_fixedAt_zero] at h0
      simp at h0
    · rw [layoutBDeployed_msgAt_zero] at h0
      simp at h0
    · exact h0 (layoutBDeployed_pieceAt x 0)
  · intro h x st hst hf
    exact absurd hf (h x st hst)

end RecDeploy
