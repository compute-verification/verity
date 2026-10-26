import Proofs.Flock.Claims.Recursion.VerdictPieces
import Proofs.Flock.Claims.Recursion.ResAccepts
import Proofs.Flock.Claims.Recursion.OpenedPlain

/-!
# The `rec-open` pieces of the verdict, as the parts compute them

Row D of the verdict (`RunsFrom.opened`) asks every query of every level of each rep's decided proof to open to the
level's cap. `OpenedPlain.opened_plain_of` reduces that to one fact per query (`QueryOpens`), and V\*'s level parts
(`rec-open/l<L>-h<H>`) compute it: each copy climbs from the leaf of the row it reads through the siblings it reads
(`LevelOuts`). This file is that share of the parts' lowering fact, in the shape of the `rec-residuals` pieces'
(`resPieces_computes`).

* `OpenPiece`, `openHolds`: a piece is a query `(table, rep, level, query)` and the copy `(t, o, g, u)` of a level
  table that computes it; it holds where its query opens (`QueryOpens`) at the schedule, cap and proof `sideAt` reads
  off the opened messages and the position `posAt` reads off the coins.
* `climb_of_node`, `openOuts_plain`: a level copy's climb is at hm96-sha512 (`LevelOuts`), whose node is `plain`'s, so
  it is the climb at `plain`.
* `openPieces_computes`: at a part whose V\* statement is a level (`.recOpen`, or R's `.recOpenStacked`),
  `PartAccepts` gives every piece `ownQ` gives the part, from two hypotheses: the layout fact `hcopy` (`OpenCopy`: the
  piece's copy is a copy of the statement at the level's lanes and depth, reading the piece's opened row and siblings)
  and the registration ties `htie` (`OpenTie`: the position bits the copy reads are the low bits of `posAt`'s position,
  `RecCoins.checkOpen`'s `dirs`, and its climb's top is the cap's entry there, `checkOpenTops`).
* `bits_of_dirsRow`: the `dirs` tie at the registered `dirsRow` words gives `OpenTie`'s bits.
-/

namespace RecDeploy

open Claims (Msg)
open FlockDeploy (Coin)
open Claims.Recursion.VStarB
open FlockSoundness.Refine (openAt)
open FlockVBridge.Family (bytesAt outCol plain)
open RecDeploy.OpenedPlain (QueryOpens queryOpens_of_outs dirsRow_testBit)

/-! ## A level copy's climb at `plain` -/

/-- **A climb reads its scheme's node only.** -/
theorem climb_of_node {ms ms' : _root_.Flock.MerkleScheme} (h : ms.node = ms'.node) :
    ∀ (acc : ByteArray) (idx : ℕ) (ss : List ByteArray),
      _root_.Flock.climb ms acc idx ss = _root_.Flock.climb ms' acc idx ss
  | _, _, [] => rfl
  | acc, idx, s :: rest => by
    simp only [_root_.Flock.climb, h]
    exact climb_of_node h _ _ rest

/-- **A copy's outputs at hm96-sha512 are its outputs at `plain`**: the two schemes' nodes are both SHA-512 of the
concatenation, and the outputs read the scheme only through the climb. -/
theorem openOuts_plain {v : ℕ → Bool} {LANES H : ℕ} {r a cf : Array _root_.Flock.F128} {rest : ByteArray} {idx : ℕ}
    {ss : List ByteArray}
    (h : _root_.Flock.Readable.OpenOuts v _root_.Flock.MerkleScheme.hm96Sha512 LANES H r a cf rest idx ss) :
    _root_.Flock.Readable.OpenOuts v plain LANES H r a cf rest idx ss := by
  unfold _root_.Flock.Readable.OpenOuts at h ⊢
  rwa [climb_of_node (ms := _root_.Flock.MerkleScheme.hm96Sha512) (ms' := plain) rfl] at h

/-! ## The `rec-open` pieces -/

/-- **A `rec-open` piece**: query `query` of level `level` of rep `rep` of table `table`'s decided proof, and the copy
`(t, o, g, u)` of a level table, as `Readable.copyOf` indexes it, that computes its climb. -/
structure OpenPiece where
  table : ℕ
  rep : ℕ
  level : ℕ
  query : ℕ
  key : ℕ × ℕ × ℕ × ℕ

/-- **What a `rec-open` piece opens against**: the table's schedule, and the rep's cap and Ligerito proof. -/
structure OpenSide where
  sch : _root_.Flock.Schedule
  cap : Array ByteArray
  lig : _root_.Flock.LigeritoProof

/-- **What a `rec-open` piece says** at the inner statement `x`, the opened messages `ms` and the coins `cs`: its query
opens (`QueryOpens`) at the schedule, cap and proof `sideAt` reads off the messages, at the position `posAt` reads off
the coins. -/
def openHolds (sideAt : Msg → List Msg → OpenPiece → OpenSide) (posAt : Msg → List Coin → OpenPiece → ℕ)
    (q : OpenPiece) (x : Msg) (ms : List Msg) (cs : List Coin) : Prop :=
  QueryOpens (sideAt x ms q).sch (sideAt x ms q).cap (sideAt x ms q).lig q.level q.query (posAt x cs q)

/-- **The layout fact at a `rec-open` piece's copy**: the copy's columns `v` are a level table's at the piece's level's
lanes (`LANES = 2^lanes`) and depth (`H = d − c`), and read the piece's opened row and its siblings. -/
def OpenCopy (sd : OpenSide) (q : OpenPiece) (LANES H : ℕ) (v : ℕ → Bool) : Prop :=
  LANES = 2 ^ (sd.sch.levels[q.level]!).lanes ∧
    H = (sd.sch.levels[q.level]!).d - (sd.sch.levels[q.level]!).cap ∧
    _root_.Flock.Readable.openRow v LANES = (openAt sd.lig (sd.sch.levels.size - 1) q.level).1[q.query]! ∧
    _root_.Flock.sibsOf (openAt sd.lig (sd.sch.levels.size - 1) q.level).2.1
        ((sd.sch.levels[q.level]!).d - (sd.sch.levels[q.level]!).cap) q.query =
      _root_.Flock.Readable.openSibs v LANES H

/-- **The registration ties at a `rec-open` piece's copy**: the position bits the copy's columns `v` read are the low
`d − c` bits of `pos` (the verifier's `dirs` row; `bits_of_dirsRow`), and the first 64 bytes of its outputs, its
climb's top, are the cap's entry `pos >>> (d − c)` (`checkOpenTops`). -/
def OpenTie (sd : OpenSide) (q : OpenPiece) (LANES H pos : ℕ) (v : ℕ → Bool) : Prop :=
  (∀ h < (sd.sch.levels[q.level]!).d - (sd.sch.levels[q.level]!).cap,
      (_root_.Flock.Readable.openIdx v LANES H).testBit h = pos.testBit h) ∧
    bytesAt v (outCol LANES H) 64 =
      (FlockSoundness.Refine.capAt sd.cap sd.lig q.level).getD
        (pos >>> ((sd.sch.levels[q.level]!).d - (sd.sch.levels[q.level]!).cap)) ByteArray.empty

/-- **The `dirs` tie at the registered row gives the position's bits**: where the bits a copy reads are those of the
verifier's `dirsRow pos H` words (16 a word), they are `pos`'s low `H` bits (`dirsRow_testBit`, `H ≤ 32`). -/
theorem bits_of_dirsRow {idx pos H : ℕ} (hH : H ≤ 32)
    (h : ∀ k < H, idx.testBit k = ((_root_.Flock.RecCoins.dirsRow pos H)[k / 16]!).testBit (k % 16)) :
    ∀ k < H, idx.testBit k = pos.testBit k :=
  fun k hk => (h k hk).trans (dirsRow_testBit hH hk)

/-- **The `rec-open` pieces' share at a level part**: where part `s`'s V\* statement at `x`'s inner shape is a level
(`.recOpen`, or one of R's, `.recOpenStacked`), `PartAccepts` gives, through every copy's climb (`partAccepts_staged`,
`levelCopies`, `stackedCopies`), every piece `ownQ` gives the part, at the position the coins give (`posAt`), wherever
the layout places the piece at one of the statement's copies reading its opened row and siblings (`hcopy`,
`OpenCopy`) and the copy's position bits and top are the registered ones (`htie`, `OpenTie`). -/
theorem openPieces_computes {H : Msg → Msg} {D : Claims.Recursion.Deploy Coin} {L : LayoutB}
    {shape : Claims.Recursion.Form → Prop} {untr : TrReader} {b : Brings} {cmt : Msg → ℕ → Msg → Msg} {s : ℕ}
    {x : Msg} {c : ℕ → Coin} {ms : List Msg}
    (h : PartAccepts H D (deployZkRec _root_.Flock.Readable.relation) L shape (showsPart untr b) cmt s x c ms)
    {r : XR} (hr : readX x = some r) {ss : Array VStar.Stmt} (hss : _root_.Flock.RecVStar.statements r.shape = .ok ss)
    {lv LANES HH n : ℕ}
    (hsv : ss[s]? = some (.recOpen lv LANES HH n) ∨ ss[s]? = some (.recOpenStacked lv LANES HH n))
    (Bd : Claims.Recursion.Bound Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace)
    (pieces : Msg → List OpenPiece) (ownQ : Msg → OpenPiece → Option ℕ)
    (sideAt : Msg → List Msg → OpenPiece → OpenSide) (posAt : Msg → List Coin → OpenPiece → ℕ)
    (hcopy : ∀ (xO : Msg) (w : Claims.Flock.Wit) (p : _root_.Flock.Readable.Parsed),
      _root_.Flock.Readable.parseBytes readOfO xO = some p → L.RowsRead x s w ms →
      ∀ q ∈ pieces (Claims.Recursion.xI D x), ownQ x q = some s →
        ∃ ht : q.key.1 < p.sts.size, q.key.2.1 < 2 ^ (p.sts[q.key.1].m - p.sts[q.key.1].c.kLog) ∧
          q.key.2.2.1 < p.sts[q.key.1].c.g ∧ q.key.2.2.2 < p.sts[q.key.1].c.upv ∧
          ∀ os last, Bd.final (ms.getD (Bd.DE.V.k (Claims.Recursion.xI D x) - 1) []) = some (os, last) →
            os.length + 1 = Bd.DE.V.k (Claims.Recursion.xI D x) →
            OpenCopy (sideAt (Claims.Recursion.xI D x) (Bd.opened os last) q) q LANES HH
              (_root_.Flock.Readable.copyOf p q.key.1 ht w q.key.2.1 q.key.2.2.1 q.key.2.2.2))
    (htie : ∀ (xO : Msg) (w : Claims.Flock.Wit) (p : _root_.Flock.Readable.Parsed),
      _root_.Flock.Readable.parseBytes readOfO xO = some p → L.Agrees cmt x s w xO ms →
      (∀ e ∈ L.calls x s w, H e.1 = e.2) → (Claims.findColl (L.calls x s w ++ vcallsAt H D xO)).isSome = false →
      showsTr untr s x xO c →
      ∀ q ∈ pieces (Claims.Recursion.xI D x), ownQ x q = some s → ∀ ht : q.key.1 < p.sts.size,
        ∀ os last, Bd.final (ms.getD (Bd.DE.V.k (Claims.Recursion.xI D x) - 1) []) = some (os, last) →
          os.length + 1 = Bd.DE.V.k (Claims.Recursion.xI D x) →
          OpenTie (sideAt (Claims.Recursion.xI D x) (Bd.opened os last) q) q LANES HH
            (posAt (Claims.Recursion.xI D x) ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) q)
            (_root_.Flock.Readable.copyOf p q.key.1 ht w q.key.2.1 q.key.2.2.1 q.key.2.2.2)) :
    PiecesComputes Bd D pieces (openHolds sideAt posAt) ownQ s x c ms := by
  obtain ⟨xO, w, p, r', sv, hx, -, -, hr', ⟨ss', hss', hsv'⟩, -, hcp, hA, hT, hc, hs⟩ := partAccepts_staged h
  rw [hr] at hr'
  cases hr'
  rw [hss] at hss'
  cases hss'
  have hlev : ∀ t (ht : t < p.sts.size) (o g u : ℕ), o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog) →
      g < p.sts[t].c.g → u < p.sts[t].c.upv →
      _root_.Flock.Readable.LevelOuts (_root_.Flock.Readable.copyOf p t ht w o g u) LANES HH := by
    rcases hsv with hsv | hsv <;> rw [hsv] at hsv' <;> cases hsv'
    · exact fun t ht o g u ho hg hu => _root_.Flock.Readable.levelCopies ht ho hg hu hcp
    · exact fun t ht o g u ho hg hu => _root_.Flock.Readable.stackedCopies ht ho hg hu hcp
  have hRR : L.RowsRead x s w ms := by
    obtain ⟨-, -, -, hRR⟩ := hA
    exact hRR
  intro q hq ho os last hf hl
  obtain ⟨ht, hob, hgb, hub, hcq⟩ := hcopy xO w p hx hRR q hq ho
  obtain ⟨hLA, hHH, hrow, hsib⟩ := hcq os last hf hl
  obtain ⟨hbits, htop⟩ := htie xO w p hx hA hT hc hs q hq ho ht os last hf hl
  have hout := (hlev q.key.1 ht q.key.2.1 q.key.2.2.1 q.key.2.2.2 hob hgb hub).1
  obtain ⟨hrs, -, -, -, hsl, -⟩ := _root_.Flock.Readable.openIns_reads
    (_root_.Flock.Readable.copyOf p q.key.1 ht w q.key.2.1 q.key.2.2.1 q.key.2.2.2) LANES HH
  exact queryOpens_of_outs (openOuts_plain hout) hrow.symm hsib (by rw [hrs, hLA]) (by rw [hsl, hHH]) hbits htop

end RecDeploy
