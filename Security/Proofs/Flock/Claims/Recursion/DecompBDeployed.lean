import Proofs.Flock.Claims.Recursion.NonVacuousRows
import Proofs.Flock.Claims.Recursion.VPiecesDeployed
import Proofs.Flock.Claims.Recursion.OpenLowerDeployed
import Proofs.Flock.Claims.Recursion.CapLower
import Proofs.Flock.Claims.Recursion.RestRead
import Proofs.Flock.Claims.Recursion.RestLens
import Proofs.Flock.Claims.Recursion.RestReadsAt
import Proofs.Flock.Claims.Recursion.RestVerdictBases
import Proofs.Flock.Claims.Recursion.VerdictRowEJoint
import Proofs.Flock.Claims.Recursion.RowDAccept

/-!
# `RecKSDecompR` assembled from chain B's pieces, its open leaves named

`RecKSDecompR` (`NonVacuous`) is `DecompB` at `PartAccepts` in its dense form at `nextR`, at V\*'s bound layout
`layoutBTops (finOpen fin)`. This file assembles it, at any layout `L` (`RecKSDecompL`, which is `RecKSDecompR` at
`layR fin`: `recKSDecompR_eq`), and lands it at the V\* statements (`RecKSDecompLV`, `DecompBV` at `acceptsLV`), from
the composition at the parts' own messages under `JointV` (`decompBV_lastSplitP_sharesJV`), the
row targets (`Top.targets_of_rows`), the V\* pieces and owners (`VPieces`), the `rec-open` lowering (`OpenLower`) and
the base checks (`restBase_holds`). Every fact still open is one hypothesis, a `def … : Prop` here or a target that
a lane already states, so that its owner proves it against a fixed statement.

**The choices.** The parts' pieces are row D's `rec-open` queries (`Rows.rowDPieces key`, owned by their level's
statement, `ownQR`) and the `rec-residuals` groups `resP` (owned by `ownR`); no `rec-cap` piece (the cap rows leave the
verdict's pieces, `nodeOf`). No part owns a base check (`noB`). A `rec-open` piece also carries its running sum at
the chain row D's pieces carry together (`openE`: `openHolds ∧ Top.accPiece`), and row E's accumulators are that
chain's (`Top.accRC rd key`), so `TargetE` is proved (`targetE_accRC`, `rowEHolds_PD`).

**The leaves** (`recKSDecompL_of_leaves`):

* the row targets: `TargetRDep` and `TargetDMerkle` (`TargetDecode` is `decode_at`, with no JSON premise);
* `hrest`, the checks no part holds: `BasesHold` (the base checks), `FinalReadsL`, `RoundLensL`, `ReadsAtL` (the last
  message's reading, the rounds' lengths, the record's reading; RESTREAD's three at `L`), `SaltRowRest` (a round's
  commit strings where no part holds it) and `PiecesOwned` (every piece has an owner, so `hrest` is asked of none);
* `hlowR`, each part's share of the checks it holds that are no pieces: `RestLowR`;
* `hlowP`, each part's share of the pieces at the parts' own messages: `dense` (proved), the residuals (`ResLowJ`),
  the openings (`openPieces_deployed`, from `SideSchR`, `CopyOpensR`, `CopyDirsTieR` and `CopyTopR`; the copy's range,
  `copyFitsR_rowD`, and its position bits, `copyBitsR_of`, are proved) and their running sums (`Top.RowEJoint` at
  `acceptsLV`, RECE's, through `piecesComputes_accPiece`). The leaves stated under `Joint` read `JointV` through
  `joint_of_jointP (jointP_of_jointV _)`.

`recKSDecompD_of_leaves` is the same at the layout of record `DeployedR.layR elem`, the side `sideAtR rd` and the
positions `Rows.posAt`: `TargetDMerkle` proved (`targetDMerkle_PD`, RECL's), RESTREAD's facts as it states them, `RoundLens` proved (`roundLens`), `SideSchR` proved (`sideSchR_sideAtR`) and `CopyTopR` from CAPLOWER's
`OutBindR` and `SideCapR` (`capTop_of`).
-/

namespace RecDeploy.DecompBR

open Flock
open Claims (Msg Out)
open FlockDeploy (Coin FinalP nextR nextR_coin ReadOkR VerdictR)
open Claims.Recursion.VStarB
open RecDeploy.VPieces (VPiece vrows vholds noB vown)
open RecDeploy.OpenLower (ownQR Joint)
open RecDeploy.RestRead (DenseX kI)

noncomputable section

/-! ## The statement at a layout -/

section Statement

variable (rd : Msg → Option FinalP) (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
  (hd : Msg → Flock.RecRecord.Record) (rest : Msg → List (Msg × Out) → List UInt8) (wE : Msg → ℝ)
  (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (L : LayoutB)

/-- The brings whose transcript is the coin server's record. -/
abbrev bB : Brings := { b with tr := recTr aux nOf hd rest }

/-- The compiled record at layout `L`, whose inner view runs `IVStar` at `nextR rd`. -/
abbrev recL : Claims.Recursion.Deploy Coin :=
  recBound _root_.Flock.Readable.relation (bB b aux nOf hd rest) L (bdR rd wE final w).IVStar

/-- What each part's circuit accepts at layout `L`. -/
abbrev acceptsL :=
  PartAccepts FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w L) (deployZkRec _root_.Flock.Readable.relation)
    L Claims.Recursion.Form.Dense (showsPart recRead (bB b aux nOf hd rest)) (cmtTr recRead)

/-- **`RecKSDecompR` at layout `L`**: wherever every part's circuit accepts at a dense compiled statement, `IVStar` at
`nextR rd` accepts or two of the parts' recorded calls collide. -/
def RecKSDecompL : Prop :=
  DecompB FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w L) (bdR rd wE final w).IVStar L
    Claims.Recursion.Form.Dense (acceptsL rd b aux nOf hd rest wE final w L)

/-- What each part's circuit accepts at layout `L`, at its V\* statement and witness (`PartAcceptsV`), the check
`Top.RowEJoint` reads. -/
abbrev acceptsLV :=
  PartAcceptsV FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w L) (deployZkRec _root_.Flock.Readable.relation)
    L Claims.Recursion.Form.Dense (showsPart recRead (bB b aux nOf hd rest)) (cmtTr recRead)

/-- **`RecKSDecompL` at the V\* statements**: `DecompBV` at `acceptsLV`, the form `Flock.SecurityProofs.recursiveKSAt_ofV`
takes, its escape the recorded calls and the compiled verifier's evaluations at the V\* statements. -/
def RecKSDecompLV : Prop :=
  DecompBV FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w L) (bdR rd wE final w).IVStar L
    Claims.Recursion.Form.Dense (acceptsLV rd b aux nOf hd rest wE final w L)

theorem recKSDecompR_eq (fin : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) :
    RecKSDecompR rd b aux nOf hd rest fin wE final w = RecKSDecompL rd b aux nOf hd rest wE final w (layR fin) :=
  rfl

end Statement

/-! ## The pieces and their owners -/

/-- **The parts' pieces**: the `rec-residuals` groups `resP`, then row D's `rec-open` queries; no `rec-cap` node. -/
abbrev rowsD (resP : Msg → List ResPiece) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) : Msg → List VPiece :=
  vrows resP (Rows.rowDPieces key) fun _ => []

/-- What a `rec-open` piece says: its opening (`openHolds`) and its running sum at the chain row D's pieces carry
together (`Top.accPiece`, row E's). -/
abbrev openE (rd : Msg → Option FinalP) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ)
    (side : Msg → List Msg → OpenPiece → OpenSide) (posAt : Msg → List Coin → OpenPiece → ℕ) :
    OpenPiece → Msg → List Msg → List Coin → Prop :=
  fun q x ms cs => openHolds side posAt q x ms cs ∧ Top.accPiece rd key q x ms cs

/-- What each piece says: `resHolds` and `openE`. -/
abbrev holdsD (rd : Msg → Option FinalP) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ)
    (elemAt : Msg → List Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
    (vvAt : Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
    (side : Msg → List Msg → OpenPiece → OpenSide) (posAt : Msg → List Coin → OpenPiece → ℕ) :
    VPiece → Msg → List Msg → List Coin → Prop :=
  vholds (resHolds elemAt vvAt) (openE rd key side posAt) fun _ _ _ _ => True

/-- **The pieces' owners**: `dense` to part 0, a residual group to `ownR`'s part, a query to its level's (`ownQR`). -/
abbrev ownD (ownR : Msg → ResPiece → Option ℕ) : Msg → Top.Piece VPiece → Option ℕ :=
  vown ownR ownQR fun _ _ => none

/-- Where the rows are asked: the parts' pieces hold. -/
abbrev PD (rd : Msg → Option FinalP) (resP : Msg → List ResPiece) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ)
    (elemAt : Msg → List Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
    (vvAt : Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
    (side : Msg → List Msg → OpenPiece → OpenSide) (posAt : Msg → List Coin → OpenPiece → ℕ) :
    Msg → List Msg → List Coin → Prop :=
  Top.RowsHold (rowsD resP key) (holdsD rd key elemAt vvAt side posAt)

/-- **Row E's pieces hold their running sums where the parts' pieces hold**: `accPiece` is a conjunct of `openE`. -/
theorem rowEHolds_PD (rd : Msg → Option FinalP) (resP : Msg → List ResPiece) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ)
    (elemAt : Msg → List Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
    (vvAt : Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
    (side : Msg → List Msg → OpenPiece → OpenSide) (posAt : Msg → List Coin → OpenPiece → ℕ) :
    ∀ (x : Msg) (ms : List Msg) (cs : List Coin), PD rd resP key elemAt vvAt side posAt x ms cs →
      ∀ q ∈ Rows.rowDPieces key x, Top.accPiece rd key q x ms cs := fun _ _ _ h q hq =>
  (show openE rd key side posAt q _ _ _ from
    h (.opn q) (List.mem_append.2 (Or.inl (List.mem_append.2 (Or.inr (List.mem_map.2 ⟨q, hq, rfl⟩)))))).2

/-- **`TargetDMerkle` at the parts' pieces, at `nextR`'s side and the decision's positions, with nothing open**:
RECL's `targetDMerkle_of_accept_sideAtR`, whose pieces say `openHolds` alone, which `openE` gives. -/
theorem targetDMerkle_PD (rd : Msg → Option FinalP) (resP : Msg → List ResPiece)
    (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) (elemAt : Msg → List Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
    (vvAt : Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
    (acc : Msg → List Msg → List Coin → ℕ → Array _root_.Flock.F128) :
    Top.TargetDMerkle rd (PD rd resP key elemAt vvAt (OpenLower.sideAtR rd) Rows.posAt) acc := by
  intro x ms cs hA
  refine Top.targetDMerkle_of_accept_sideAtR (rd := rd) (resP := resP) (key := key) (capP := fun _ => [])
    (resH := resHolds elemAt vvAt) (capH := fun _ _ _ _ => True) (acc := acc) x ms cs
    ⟨hA.1, hA.2.1, hA.2.2.1, fun p hp => ?_⟩
  have h := hA.2.2.2 p hp
  cases p with
  | res q => exact h
  | opn q => exact (show openE rd key (OpenLower.sideAtR rd) Rows.posAt q x ms cs from h).1
  | cap q => trivial

/-! ## The leaves -/

section Leaves

variable (rd : Msg → Option FinalP) (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
  (hd : Msg → Flock.RecRecord.Record) (rest : Msg → List (Msg × Out) → List UInt8) (wE : Msg → ℝ)
  (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (L : LayoutB)
  (resP : Msg → List ResPiece) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ)
  (elemAt : Msg → List Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
  (vvAt : Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
  (side : Msg → List Msg → OpenPiece → OpenSide) (posAt : Msg → List Coin → OpenPiece → ℕ)
  (ownR : Msg → ResPiece → Option ℕ)

/-- **`final` reads the last message at every witnesses** (RESTREAD's `FinalReads`, at layout `L`). -/
def FinalReadsL : Prop :=
  ∀ x, DenseX x → ∀ ws : List Claims.Flock.Wit, ws.length = nS x →
    ∃ os last, final ((L.msgs x ws).getD (kI x - 1) []) = some (os, last) ∧ os.length + 1 = kI x ∧
      ∀ j < os.length, (os.getD j []).length = nAtSched (xIOf x) j

/-- **Each round before the last is 128 bytes a commit string** (RESTREAD's `RoundLens`, at layout `L`). -/
def RoundLensL : Prop :=
  ∀ x, DenseX x → ∀ ws : List Claims.Flock.Wit, ws.length = nS x → ∀ j < kI x - 1,
    ((L.msgs x ws).getD j []).length = 128 * nAtSched (xIOf x) j

/-- **The last step reads at every witnesses whose calls are true** (RESTREAD's `ReadsAt`, at layout `L`). -/
def ReadsAtL : Prop :=
  ∀ x, DenseX x → ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit), ws.length = nS x →
    (∀ e ∈ L.allCalls x ws, FlockDeploy.sha512 e.1 = e.2) →
    ∀ os last, final ((L.msgs x ws).getD (kI x - 1) []) = some (os, last) → os.length + 1 = kI x →
      ReadOkR rd (xIOf x) (os.map (fun o => FlockDeploy.joinR (o.map Prod.fst)) ++ [last])
        ((List.range (kI x)).map c)

/-- **A round no part holds has its commit strings' salts and rows** at every witnesses whose calls are true: round
`j < k - 1` where `firstHolder` is none. Where every such round has a holder, this holds outright. -/
def SaltRowRest : Prop :=
  ∀ x, DenseX x → ∀ j < kI x - 1, firstHolder (recL rd b aux nOf hd rest wE final w L) L x j = none →
    ∀ i < nAtSched (xIOf x) j, ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit), ws.length = nS x →
      (∀ e ∈ L.allCalls x ws, FlockDeploy.sha512 e.1 = e.2) →
      ∀ {Q : Type} (S : ElemSplit FlockDeploy.sha512 (bdR rd wE final w) Q),
        (IVCheck.salt j i : IVCheck Q).Holds FlockDeploy.sha512 (bdR rd wE final w) S (xIOf x) c (L.msgs x ws) ∧
        (IVCheck.row j i : IVCheck Q).Holds FlockDeploy.sha512 (bdR rd wE final w) S (xIOf x) c (L.msgs x ws)

/-- **Every piece has an owner** at a dense compiled statement: each residual group (`ownR`) and each `rec-open`
query (`ownQR`) is given one of the statement's parts. -/
def PiecesOwned : Prop :=
  ∀ x, DenseX x → (∀ q ∈ resP (xIOf x), ∃ s < nS x, ownR x q = some s) ∧
    ∀ q ∈ Rows.rowDPieces key (xIOf x), ∃ s < nS x, ownQR x q = some s

/-- **Each part's share of the checks it holds that are no pieces** (`hlowR`): where part `s`'s circuit accepts, the
salts and rows of the rounds it holds first, and the record's reading where it holds the last round first. -/
def RestLowR : Prop :=
  ∀ s x c ms, acceptsL rd b aux nOf hd rest wE final w L s x c ms →
    (∀ j < kI x - 1, firstHolder (recL rd b aux nOf hd rest wE final w L) L x j = some s →
      ∀ i < nAtSched (xIOf x) j, ∀ {Q : Type} (S : ElemSplit FlockDeploy.sha512 (bdR rd wE final w) Q),
        (IVCheck.salt j i : IVCheck Q).Holds FlockDeploy.sha512 (bdR rd wE final w) S (xIOf x) c ms ∧
        (IVCheck.row j i : IVCheck Q).Holds FlockDeploy.sha512 (bdR rd wE final w) S (xIOf x) c ms) ∧
    (firstHolder (recL rd b aux nOf hd rest wE final w L) L x (kI x - 1) = some s →
      ∀ os last, (bdR rd wE final w).final (ms.getD (kI x - 1) []) = some (os, last) → os.length + 1 = kI x →
        ReadOkR rd (xIOf x) ((bdR rd wE final w).opened os last) ((List.range (kI x)).map c))

/-- **The residual groups' share at the parts' own messages** (`hlowP`'s `rec-residuals` kind): where every part
accepts at the parts' own messages, each group `ownR` gives part `s` holds. -/
def ResLowJ : Prop :=
  ∀ (x : Msg) (st : Claims.Recursion.Stmt), parse x = some st → Claims.Recursion.Form.Dense st.form →
    ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit),
      Joint FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w L) L
        (acceptsL rd b aux nOf hd rest wE final w L) x c ws →
      ∀ s < nS x, PiecesComputes (bdR rd wE final w) (recL rd b aux nOf hd rest wE final w L) resP
        (resHolds elemAt vvAt) ownR s x c (L.msgs x ws)

end Leaves

/-! ## The assembly -/

theorem mem_last_checks {H : Msg → Msg} {rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation}
    {nextE : Msg → List Msg → List Coin → Claims.OC Out}
    {colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)} {wE : Msg → ℝ} {join : List Msg → Msg}
    {rowIn : Msg → Msg} {final : Msg → Option (List (List (Msg × Salt)) × Msg)} {w : Msg → ℝ}
    {hcoin : ∀ x as cs c, FlockDeploy.act false x as cs = .coin c → ((nextE x as cs).eval H).1 = .msg c.bytes}
    {R : Type} {LS : LastSplit H nextE R} {x : Msg} {r : R}
    (h : DeployCheck.last r ∈ (deploySplit H rel nextE colsE wE join rowIn final w hcoin LS).checks x) :
    r ∈ LS.checks x := by
  rcases List.mem_cons.1 h with h | h
  · cases h
  rcases List.mem_append.1 h with h | h
  · obtain ⟨_, _, h⟩ := List.mem_map.1 h
    cases h
  · obtain ⟨r', hr', h⟩ := List.mem_map.1 h
    cases h
    exact hr'

section Assembly

variable (rd : Msg → Option FinalP) (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
  (hd : Msg → Flock.RecRecord.Record) (rest : Msg → List (Msg × Out) → List UInt8) (wE : Msg → ℝ)
  (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (L : LayoutB)
  (resP : Msg → List ResPiece) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ)
  (elemAt : Msg → List Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
  (vvAt : Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
  (side : Msg → List Msg → OpenPiece → OpenSide) (posAt : Msg → List Coin → OpenPiece → ℕ)
  (ownR : Msg → ResPiece → Option ℕ)

/-- **`RecKSDecompR` at layout `L`, from its leaves.** -/
theorem recKSDecompL_of_leaves
    (hRdep : Top.TargetRDep rd (PD rd resP key elemAt vvAt side posAt) (Top.accRC rd key))
    (hDm : Top.TargetDMerkle rd (PD rd resP key elemAt vvAt side posAt) (Top.accRC rd key))
    (hEJ : Top.RowEJoint rd key FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w L) (bdR rd wE final w) L
      Claims.Recursion.Form.Dense (acceptsLV rd b aux nOf hd rest wE final w L))
    (hB : BasesHold rd) (hF : FinalReadsL final L) (hLn : RoundLensL L) (hRd : ReadsAtL rd final L)
    (hSR : SaltRowRest rd b aux nOf hd rest wE final w L) (hOwn : PiecesOwned resP key ownR)
    (hlowR : RestLowR rd b aux nOf hd rest wE final w L)
    (hres : ResLowJ rd b aux nOf hd rest wE final w L resP elemAt vvAt ownR)
    (hSch : OpenLower.SideSchR FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w L)
      Claims.Recursion.Form.Dense recRead (bB b aux nOf hd rest) (cmtTr recRead) L (bdR rd wE final w) side
      (Rows.rowDPieces key))
    (hO : OpenLower.CopyOpensR FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w L)
      Claims.Recursion.Form.Dense recRead (bB b aux nOf hd rest) (cmtTr recRead) L (bdR rd wE final w) side
      (Rows.rowDPieces key))
    (hDirs : OpenLower.CopyDirsTieR FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w L)
      Claims.Recursion.Form.Dense recRead (bB b aux nOf hd rest) (cmtTr recRead) L (bdR rd wE final w)
      (Rows.rowDPieces key) posAt)
    (hTop : OpenLower.CopyTopR FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w L)
      Claims.Recursion.Form.Dense recRead (bB b aux nOf hd rest) (cmtTr recRead) L (bdR rd wE final w) side
      (Rows.rowDPieces key) posAt) :
    RecKSDecompLV rd b aux nOf hd rest wE final w L := by
  have h : Top.Targets rd (rowsD resP key) (holdsD rd key elemAt vvAt side posAt) (Top.accRC rd key) :=
    Top.targets_of_open rd _ _ _ hRdep hDm
      (Top.targetE_accRC rd key (fun P' ch h => Top.targetE_rowD rd P' ch key h) _
        (rowEHolds_PD rd resP key elemAt vvAt side posAt))
      (DecodeAt.decode_at rd _)
  have hP : ∀ x, (recL rd b aux nOf hd rest wE final w L).parse x = parse x := fun _ => rfl
  have hS : ∀ f, Claims.Recursion.Form.Dense f → Claims.Recursion.Form.Dense f := fun _ h => h
  have hC := OpenLower.openCopyR_of FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w L)
    Claims.Recursion.Form.Dense recRead (bB b aux nOf hd rest) (cmtTr recRead) L (bdR rd wE final w) side
    (Rows.rowDPieces key) hP hS hSch
    (OpenLower.copyFitsR_rowD FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w L)
      Claims.Recursion.Form.Dense recRead (bB b aux nOf hd rest) (cmtTr recRead) L key hP hS) hO
  have hT := OpenLower.openTieR_of FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w L)
    Claims.Recursion.Form.Dense recRead (bB b aux nOf hd rest) (cmtTr recRead) L (bdR rd wE final w) side
    (Rows.rowDPieces key) posAt
    (OpenLower.copyBitsR_of FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w L)
      Claims.Recursion.Form.Dense recRead (bB b aux nOf hd rest) (cmtTr recRead) L (bdR rd wE final w) side
      (Rows.rowDPieces key) posAt hP hS hSch
      (OpenLower.copyDirsR_of FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w L)
        Claims.Recursion.Form.Dense recRead (bB b aux nOf hd rest) (cmtTr recRead) L (bdR rd wE final w)
        (Rows.rowDPieces key) posAt hDirs)) hTop
  refine OpenLower.decompBV_lastSplitP_sharesJV FlockDeploy.sha512 rd (Top.pieces (rowsD resP key))
    (Top.pholds (holdsD rd key elemAt vvAt side posAt))
    (Top.hpieces_of_targets rd (rowsD resP key) (holdsD rd key elemAt vvAt side posAt) (Top.accRC rd key) h)
    _root_.Flock.Readable.relation (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE
    rowInDeployed final w noB (ownD ownR) _root_.Flock.Readable.relation (bB b aux nOf hd rest) L
    (bdR rd wE final w).IVStar Claims.Recursion.Form.Dense (deployZkRec _root_.Flock.Readable.relation)
    (showsPart recRead (bB b aux nOf hd rest)) (cmtTr recRead) ?_ ?_ ?_ ?_
  · -- hR
    intro x st hx hf r hr s hs
    exact VPieces.ownP_lt ownR ownQR (fun _ _ => none) hs
  · -- hrest
    intro x st hx hf q hq hown hlast c ws hl hT'
    have hX : DenseX x := ⟨st, hx, hf⟩
    rcases mem_ivChecks _ _ _ hq with rfl | ⟨j, hj, rfl⟩ | ⟨j, hj, i, hi, rfl | rfl⟩ | ⟨q', hq', rfl⟩
    · obtain ⟨os, last, hfin, hos, hcnt⟩ := hF x hX ws hl
      exact ⟨RestRead.kI_pos x, os, last, hfin, hos, hcnt⟩
    · have hj' : j < kI x - 1 := hj
      obtain ⟨os, last, hfin, hos, hcnt⟩ := hF x hX ws hl
      intro os' last' hf'
      have hf'' : final ((L.msgs x ws).getD (kI x - 1) []) = some (os', last') := hf'
      rw [hfin] at hf''
      obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj hf'')
      rw [hLn x hX ws hl j hj']
      exact congrArg (128 * ·) (hcnt j (by omega)).symm
    · exact (hSR x hX j hj hown i hi c ws hl hT' _).1
    · exact (hSR x hX j hj hown i hi c ws hl hT' _).2
    · obtain ⟨r', rfl⟩ := hlast q' rfl
      have hr' := mem_last_checks hq'
      rcases List.mem_cons.1 hr' with rfl | hr'
      · intro os last hfin hos
        exact hRd x hX c ws hl hT' os last hfin hos
      · obtain ⟨v, hv, rfl⟩ := List.mem_map.1 hr'
        rcases List.mem_append.1 hv with hv | hv
        · obtain ⟨b', hb', rfl⟩ := List.mem_map.1 hv
          exact restBase_holds rd hB _ _ _ _ _ wE _ rowInDeployed final w _ _ L _ hx hb' c _
        · obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hv
          exfalso
          obtain ⟨r, hr, -, -⟩ := parse_read hx
          obtain ⟨s, hs⟩ := VPieces.vown_some ownR ownQR (fun _ _ => none) resP (Rows.rowDPieces key)
            (fun _ => []) (VPieces.nS_pos hr) (hOwn x hX).1 (hOwn x hX).2 (fun q hq => by simp at hq) p hp
          have hown' : vown ownR ownQR (fun _ _ => none) x p = none := hown
          rw [hs] at hown'
          cases hown'
  · -- hlowR
    intro s x c ms hacc q hq hown hnp
    obtain ⟨hSRl, hRdl⟩ := hlowR s x c ms hacc
    rcases mem_ivChecks _ _ _ hq with rfl | ⟨j, hj, rfl⟩ | ⟨j, hj, i, hi, rfl | rfl⟩ | ⟨q', hq', rfl⟩
    · cases (hown : (none : Option ℕ) = some s)
    · cases (hown : (none : Option ℕ) = some s)
    · exact (hSRl j hj hown i hi _).1
    · exact (hSRl j hj hown i hi _).2
    · cases q' with
      | setup => cases (hown : (none : Option ℕ) = some s)
      | tag _ => cases (hown : (none : Option ℕ) = some s)
      | last r' =>
        cases r' with
        | read =>
          intro os last hfin hos
          exact hRdl hown os last hfin hos
        | verdict v =>
          cases v with
          | base _ => cases (hown : (none : Option ℕ) = some s)
          | piece p => exact (hnp p rfl).elim
  · -- hlowP
    intro x st hx hf c xs ws hJ s hs
    have hJ' := OpenLower.joint_of_jointP (OpenLower.jointP_of_jointV hJ)
    refine VPieces.piecesComputes_vown ownR ownQR (fun _ _ => none) (bdR rd wE final w)
      (recL rd b aux nOf hd rest wE final w L) resP (Rows.rowDPieces key) (fun _ => []) (resHolds elemAt vvAt)
      (openE rd key side posAt) (fun _ _ _ _ => True) ⟨x, st, hx, hf, rfl⟩ (hres x st hx hf c ws hJ' s hs) ?_ ?_
    · have hop := OpenLower.openPieces_deployed FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w L)
        Claims.Recursion.Form.Dense recRead (bB b aux nOf hd rest) (cmtTr recRead) L (bdR rd wE final w) side
        (Rows.rowDPieces key) posAt hC hT hx hf hJ' hs
      have hac := Top.piecesComputes_accPiece rd key FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w L)
        (bdR rd wE final w) L Claims.Recursion.Form.Dense (acceptsLV rd b aux nOf hd rest wE final w L) hEJ ownQR
        x st hx hf c xs ws hJ s
      exact fun q hq ho os last hfin hk => ⟨hop q hq ho os last hfin hk, hac q hq ho os last hfin hk⟩
    · intro q hq
      simp at hq

end Assembly

/-! ## At the layout of record and `nextR`'s side -/

/-- **The layout of record records every call `layoutBTops` does**: `DeployedR.layR elem`'s calls are
`layoutBTops (finOpen (restR elem))`'s and then the output rows' (`layR_calls`). -/
theorem allCalls_layR (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (x : Msg)
    (ws : List Claims.Flock.Wit) {e : Msg × Msg}
    (h : e ∈ (layoutBTops (finOpen (DeployedR.restR elem))).allCalls x ws) :
    e ∈ (DeployedR.layR elem).allCalls x ws := by
  unfold LayoutB.allCalls at h ⊢
  obtain ⟨l, hl, he⟩ := List.mem_flatten.1 h
  obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 hl
  have hi' : i < ws.length := by simpa using hi
  refine List.mem_flatten.2 ⟨(DeployedR.layR elem).calls x i ws[i], List.mem_iff_getElem.2 ⟨i, by simpa using hi',
    by simp⟩, ?_⟩
  rw [DeployedR.layR_calls]
  exact List.mem_append_left _ (by simpa using he)

section AtNext

variable (rd : Msg → Option FinalP) (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
  (hd : Msg → Flock.RecRecord.Record) (rest : Msg → List (Msg × Out) → List UInt8)
  (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
  (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
  (resP : Msg → List ResPiece) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ)
  (elemAt : Msg → List Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
  (vvAt : Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
  (ownR : Msg → ResPiece → Option ℕ)

/-- **`RecKSDecompR`'s premise at the layout of record, from its leaves**: `recKSDecompL_of_leaves` at
`DeployedR.layR elem`, the side `sideAtR rd` and the decision's positions `Rows.posAt`. `TargetDMerkle` is proved
there (`targetDMerkle_PD`). RESTREAD's facts are as it states them at `restR elem`
(`finalReads` proves the first at `finalR`, and `readsAt` the second at `rdR`, `finalR`), the rounds' lengths are
proved (`roundLens`), and so is the side's schedule (`sideSchR_sideAtR`). The copy's top comes from CAPLOWER's
binding and reader (`capTop_of`). -/
theorem recKSDecompD_of_leaves
    (hRdep : Top.TargetRDep rd (PD rd resP key elemAt vvAt (OpenLower.sideAtR rd) Rows.posAt) (Top.accRC rd key))
    (hEJ : Top.RowEJoint rd key FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w (DeployedR.layR elem))
      (bdR rd wE final w) (DeployedR.layR elem) Claims.Recursion.Form.Dense
      (acceptsLV rd b aux nOf hd rest wE final w (DeployedR.layR elem)))
    (hB : BasesHold rd) (hF : RestRead.FinalReads final (DeployedR.restR elem))
    (hRd : RestRead.ReadsAt rd final (DeployedR.restR elem))
    (hSR : SaltRowRest rd b aux nOf hd rest wE final w (DeployedR.layR elem)) (hOwn : PiecesOwned resP key ownR)
    (hlowR : RestLowR rd b aux nOf hd rest wE final w (DeployedR.layR elem))
    (hres : ResLowJ rd b aux nOf hd rest wE final w (DeployedR.layR elem) resP elemAt vvAt ownR)
    (hO : OpenLower.CopyOpensR FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w (DeployedR.layR elem))
      Claims.Recursion.Form.Dense recRead (bB b aux nOf hd rest) (cmtTr recRead) (DeployedR.layR elem)
      (bdR rd wE final w) (OpenLower.sideAtR rd) (Rows.rowDPieces key))
    (hDirs : OpenLower.CopyDirsTieR FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w (DeployedR.layR elem))
      Claims.Recursion.Form.Dense recRead (bB b aux nOf hd rest) (cmtTr recRead) (DeployedR.layR elem)
      (bdR rd wE final w) (Rows.rowDPieces key) Rows.posAt)
    (hOut : CapLower.OutBindR FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w (DeployedR.layR elem))
      Claims.Recursion.Form.Dense recRead (bB b aux nOf hd rest) (cmtTr recRead) (DeployedR.layR elem)
      (bdR rd wE final w) (Rows.rowDPieces key) Rows.posAt)
    (hCap : CapLower.SideCapR FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w (DeployedR.layR elem))
      Claims.Recursion.Form.Dense recRead (bB b aux nOf hd rest) (cmtTr recRead) (DeployedR.layR elem)
      (bdR rd wE final w) rd (Rows.rowDPieces key) Rows.posAt) :
    RecKSDecompLV rd b aux nOf hd rest wE final w (DeployedR.layR elem) :=
  recKSDecompL_of_leaves rd b aux nOf hd rest wE final w (DeployedR.layR elem) resP key elemAt vvAt
    (OpenLower.sideAtR rd) Rows.posAt ownR hRdep (targetDMerkle_PD rd resP key elemAt vvAt _) hEJ hB hF
    (RestRead.roundLens (DeployedR.restR elem))
    (fun x hx c ws hl hT os last hf hos =>
      hRd x hx c ws hl (fun e he => hT e (allCalls_layR elem x ws he)) os last hf hos)
    hSR hOwn hlowR hres
    (OpenLower.sideSchR_sideAtR FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w (DeployedR.layR elem))
      Claims.Recursion.Form.Dense recRead (bB b aux nOf hd rest) (cmtTr recRead) (DeployedR.layR elem)
      (bdR rd wE final w) rd (Rows.rowDPieces key))
    hO hDirs
    (CapLower.capTop_of FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w (DeployedR.layR elem))
      Claims.Recursion.Form.Dense recRead (bB b aux nOf hd rest) (cmtTr recRead) (DeployedR.layR elem)
      (bdR rd wE final w) rd (Rows.rowDPieces key) Rows.posAt (fun _ => rfl) (fun _ h => h) hOut hCap)

end AtNext

end

end RecDeploy.DecompBR
