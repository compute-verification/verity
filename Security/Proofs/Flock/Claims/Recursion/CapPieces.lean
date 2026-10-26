import Proofs.Flock.Claims.Recursion.VerdictPieces
import Proofs.Flock.VBridge.Family.Openings

/-!
# The `rec-cap` pieces of the verdict, as the parts compute them

V\*'s RecCap statements (`rec-cap/n<N>` at level 0, `rec-cap/n<N>-l<l>` later, `VStar.capUnit`) hold every node of
each Ligerito level's committed tops at a fixed row: instance `n` reads node `n` of `rec-top-L<l>` as its one input
row `top`, 64 u16 words, the node's 64 bytes then 64 zero bytes (`rec_live.top_rows`). Its one output `out` is the
row's padding, words 32 to 63, then 32 zero words (`VStar.capForms`), and V\* registers `out` as zero rows
(`rec-cap-zero`, `RecCoins.checkCap`), so the statement holds exactly where every `top` row is a node's row.

The verdict's checks S16 (each rep's cap is the committed `root_B`) and S18 (the two reps' caps agree) need no piece
at the reading: `decR` reads both caps and `root_B` off round 0's rows' nodes (`FlockDeploy.nodeOf`, `linkR`,
`FitsRep.cap`), so S16 holds by construction (`DecAt.rowL_cap`) and S18 follows from it (`verifyJPWith_ok`). What the
cap parts give the verdict is that each row the opened messages carry at a cap position is a node's row
(`isNodeRow`), so its node determines it (`nodeRow_nodeOf`, `nodeRows_of_all`). This file is that share of the parts' lowering fact, in the
shape of the `rec-open` and `rec-residuals` pieces' (`openPieces_computes`, `resPieces_computes`):

* **The share** (`CapPiece`, `capHolds`): a piece is one node of one level's tops, at the RecCap copy (`key`: table,
  block, VU, copy) that holds it, and it says the row the messages carry there is a node's row.
* **The lowering fact** (`capPieces_computes`): at a part whose V\* statement is `.cap l N`, `PartAccepts` gives the
  share, where each piece's copy reads as `top` the row the opened messages carry at the piece (`hcopy`, the layout's
  tie) and every copy's `top` there is a node's row (`hpad`, the circuit's: `out` is `top`'s padding and V\*'s zero
  rows, which, like check 5's tie at `rec-residuals`, goes from "same digest" to "same row" at the vcalls ask).
-/

namespace RecDeploy

open Claims (Msg)
open FlockDeploy (Coin nodeRow nodeOf isNodeRow nodeRow_nodeOf)
open Claims.Recursion.VStarB

/-! ## Node rows -/

/-- **Rows that are all node rows are their nodes' rows**: round 0 read as nodes (`nodeOf`) loses nothing. -/
theorem nodeRows_of_all {rs : List ByteArray} (h : ∀ r ∈ rs, isNodeRow r = true) :
    (rs.map nodeOf).map nodeRow = rs := by
  rw [List.map_map]
  exact (List.map_congr_left fun r hr => nodeRow_nodeOf (h r hr)).trans (List.map_id rs)

/-! ## The share -/

/-- **A `rec-cap` piece**: node `node` of Ligerito level `level`'s committed tops (`root_B` at level 0), held at
RecCap's copy `key` (table, block, VU, copy). -/
structure CapPiece where
  level : ℕ
  node : ℕ
  key : ℕ × ℕ × ℕ × ℕ

/-- **What a `rec-cap` piece says** at the inner statement `x`, the opened messages `ms` and the coins `cs`: the row
`rowAt` reads off the messages at the piece is a node's row. -/
def capHolds (rowAt : Msg → List Msg → CapPiece → ByteArray) (q : CapPiece) (x : Msg) (ms : List Msg)
    (_cs : List Coin) : Prop :=
  isNodeRow (rowAt x ms q) = true

/-- **The row a RecCap copy holds**: the 128 bytes its `top` input columns read (`VStar.capUnit`'s one input, columns
`0` to `8 · 128 − 1`). -/
def capTop (v : ℕ → Bool) : ByteArray := FlockVBridge.Family.bytesAt v 0 128

/-! ## The lowering fact -/

/-- **The `rec-cap` pieces' share at a `.cap` part**: at the deployed outer record and the staging `showsPart`, where
part `s`'s V\* statement at `x`'s inner shape is `.cap l N`, every piece `ownQ` gives the part is held at one of its
statement's copies, which reads as `top` the row `rowAt` reads off the opened messages (`hcopy`), and every copy of an
accepted witness reads a node's row as `top` (`hpad`). -/
theorem capPieces_computes {H : Msg → Msg} {D : Claims.Recursion.Deploy Coin} {L : LayoutB}
    {shape : Claims.Recursion.Form → Prop} {untr : TrReader} {b : Brings} {cmt : Msg → ℕ → Msg → Msg} {s : ℕ}
    {x : Msg} {c : ℕ → Coin} {ms : List Msg}
    (h : PartAccepts H D (deployZkRec _root_.Flock.Readable.relation) L shape (showsPart untr b) cmt s x c ms)
    {r : XR} (hr : readX x = some r) {ss : Array VStar.Stmt} (hss : _root_.Flock.RecVStar.statements r.shape = .ok ss)
    {l N : ℕ} (hsv : ss[s]? = some (.cap l N))
    (Bd : Claims.Recursion.Bound Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace)
    (pieces : Msg → List CapPiece) (ownQ : Msg → CapPiece → Option ℕ)
    (rowAt : Msg → List Msg → CapPiece → ByteArray)
    (hpad : ∀ (xO : Msg) (w : Claims.Flock.Wit) (p : _root_.Flock.Readable.Parsed),
      _root_.Flock.Readable.parseBytes readOfO xO = some p → L.Agrees cmt x s w xO ms →
      (∀ e ∈ L.calls x s w, H e.1 = e.2) → (Claims.findColl (L.calls x s w ++ vcallsAt H D xO)).isSome = false →
      showsTr untr s x xO c →
      ∀ t (ht : t < p.sts.size) (o g u : ℕ), p.sts[t].c.unit.name = VStar.capName l N →
        o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog) → g < p.sts[t].c.g → u < p.sts[t].c.upv →
        isNodeRow (capTop (_root_.Flock.Readable.copyOf p t ht w o g u)) = true)
    (hcopy : ∀ (xO : Msg) (w : Claims.Flock.Wit) (p : _root_.Flock.Readable.Parsed),
      _root_.Flock.Readable.parseBytes readOfO xO = some p → L.RowsRead x s w ms →
      ∀ q ∈ pieces (Claims.Recursion.xI D x), ownQ x q = some s →
        ∃ ht : q.key.1 < p.sts.size, q.key.2.1 < 2 ^ (p.sts[q.key.1].m - p.sts[q.key.1].c.kLog) ∧
          q.key.2.2.1 < p.sts[q.key.1].c.g ∧ q.key.2.2.2 < p.sts[q.key.1].c.upv ∧
          ∀ os last, Bd.final (ms.getD (Bd.DE.V.k (Claims.Recursion.xI D x) - 1) []) = some (os, last) →
            os.length + 1 = Bd.DE.V.k (Claims.Recursion.xI D x) →
            rowAt (Claims.Recursion.xI D x) (Bd.opened os last) q =
              capTop (_root_.Flock.Readable.copyOf p q.key.1 ht w q.key.2.1 q.key.2.2.1 q.key.2.2.2)) :
    PiecesComputes Bd D pieces (capHolds rowAt) ownQ s x c ms := by
  obtain ⟨xO, w, p, r', sv, hx, -, -, hr', ⟨ss', hss', hsv'⟩, -, hcp, hA, hT, hc, hs⟩ := partAccepts_staged h
  rw [hr] at hr'
  cases hr'
  rw [hss] at hss'
  cases hss'
  rw [hsv] at hsv'
  cases hsv'
  have hRR : L.RowsRead x s w ms := by
    obtain ⟨-, -, -, hRR⟩ := hA
    exact hRR
  intro q hq ho os last hf hl
  obtain ⟨ht, hob, hgb, hub, hrow⟩ := hcopy xO w p hx hRR q hq ho
  unfold capHolds
  rw [hrow os last hf hl]
  exact hpad xO w p hx hA hT hc hs q.key.1 ht q.key.2.1 q.key.2.2.1 q.key.2.2.2 (hcp q.key.1 ht).1 hob hgb hub

end RecDeploy
