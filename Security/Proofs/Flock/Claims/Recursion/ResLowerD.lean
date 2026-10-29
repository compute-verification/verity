import Proofs.Flock.Claims.Recursion.ResLowJD
import Proofs.Flock.Claims.Recursion.ResLowJP
import Proofs.Flock.Claims.Recursion.ResAccOff
import Proofs.Flock.Claims.Recursion.ResTieVV
import Proofs.Flock.Claims.Recursion.ElemRead
import Proofs.Flock.Claims.Recursion.VerdictRowEJoint

/-!
# The port facts `ResLowered` reads at the deployed constants

`resPieces_computesAt` asks, at each piece a part owns, that the rows its copy reads (`rowsRead … (keyCopy …)`) be what
`elemAt` reads off the opened messages and the coins. At `elemAt := Rows.elemReadC (Top.accRC rd key)` that holds on
the frame ports only. On the accumulator ports (`acc0`, `acc1`, `portSizes`' last two, `w`'s last four elements past
`accBase`) `accRC` reads `Top.chainC`, a chain chosen among those that carry row D's running sums (`ChainHolds`).
`accHolds` steps each row from the one before it, so the chain's row 0 is free, and the copy's accumulators (the outer
witness's own chain) are `accRC`'s up to one offset `δ c` per component `c`, shared by `acc0 c` and `acc1 c` (the
chain step is additive, `consTerm`). The final check's residuals read the two as `acc1 c + acc0 c`
(`RecAlgebra.accTerms`), so they vanish at the one where they vanish at the other.

* `accBase X`, `elemPort X p e`: where `w`'s accumulators start, and the `w` index of group `p`'s row element `e`
  (`Rows.portAt`), at the inner statement's setup.
* `accOff X p δ e`: the offset at row element `e`: `δ ((j - accBase X) % 2)` at an accumulator `j`, else `0`.
* `PartDec`, `showsDec`: part `s`'s own decision at its staged statement, and `showsPart` with it.
* `PortsAt`: a piece's context, at a joint acceptance with each part at its own witness (`JointP`).
* `PortsAtV`: `PortsAt` with the parts' staged statements `xs`: each part's check and decision there (`AccV`), and no
  collision with the verifier's evaluations at `xs` (what `JointV` over `showsDec` gives).
* `FramePorts`: on every frame port, the copy's row is `elemReadC acc`'s, at every `acc` (P2).
* `AccPorts`: on the accumulator ports, in `PortsAtV`, the copy's row is `elemReadC (Top.accRC rd key)`'s plus one
  `δ` (P3).
* `ResLoweredP`: `ResLowered` at `JointP`, which KS holds where it reads `hres`; `resLowJP_of_resLoweredP` gives
  `DecompBR.ResLowJP`.
`ResLowerDD`'s `resLowered_D` is `ResLoweredPD` (at `JointV` over `showsDec`, part `s`'s own decision beside its
staged statement) at the deployed constants from P2 and P3.

Nothing downstream of `ResLoweredP` reads `accRC`'s values themselves: it concludes `resHolds`, the vanishing of each
piece's residuals (`ResZero`), and that is all `ResLowJP` and KS's row R take from it.
-/

namespace RecDeploy.ResLower

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP setupOf)
open Claims.Recursion.VStarB

noncomputable section

/-- Where the accumulators start in the inner statement's `w`: past table 0's rounds' elements (`portSizes`). -/
def accBase (xi : Msg) : ℕ :=
  match setupOf false xi with
  | some v => (_root_.Flock.RecAlgebra.rounds (Rows.shapeNoCoin v.2.1 v.2.2)).flatten.size
  | none => 0

/-- Row element `e` of group `p`, as an index of the inner statement's `w` (`Rows.portAt`). -/
def elemPort (xi : Msg) (p e : ℕ) : Option ℕ :=
  match setupOf false xi with
  | some v => Rows.portAt (Rows.shapeNoCoin v.2.1 v.2.2) p e
  | none => none

/-- **The accumulators' offset** at group `p`'s row element `e`: `δ c` at `acc0 c` and `acc1 c` (`w`'s elements
`accBase X + c` and `accBase X + 2 + c`), `0` elsewhere. -/
def accOff (xi : Msg) (p : ℕ) (δ : ℕ → _root_.Flock.F128) (e : ℕ) : _root_.Flock.F128 :=
  match elemPort xi p e with
  | some j => if accBase xi ≤ j then δ ((j - accBase xi) % 2) else 0
  | none => 0

/-- The compiled record at the deployed layout. -/
abbrev DRL (rd : Msg → Option FinalP) (wE : Msg → ℝ) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) (b : Brings) (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) :
    Claims.Recursion.Deploy Coin :=
  recBound _root_.Flock.Readable.relation b (DeployedR.layR elem) (_root_.RecDeploy.bdR rd wE final w).IVStar

/-- What each part's circuit accepts at the deployed layout, at its own witness. -/
abbrev AccP (rd : Msg → Option FinalP) (wE : Msg → ℝ) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) (b : Brings) (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (shape : Claims.Recursion.Form → Prop) (untr : TrReader) (cmt : Msg → ℕ → Msg → Msg) :=
  PartAcceptsP FlockDeploy.sha512 (DRL rd wE final w b elem) (deployZkRec _root_.Flock.Readable.relation)
    (DeployedR.layR elem) shape (showsPart untr b) cmt

/-- **Part `s`'s decision**: the session at its staged statement `xO` decided, and its checks (`drawLaw`, `drawEq`) and
last refusals (`fullDrawO`) passed. -/
def PartDec (xO : Msg) (i : RecIns) : Prop :=
  ∃ (spec' : SessionSpec) (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Msg × Coin))
    (f : FlockDeploy.Final), FlockDeploy.checks true i.base spec' xO link c₁ mids f = .ok () ∧ fullDrawO xO f = true

/-- **What part `s` shows, with its decision**: `showsPart`, and part `s`'s own decision (`PartDec`) wherever its staged
statement sets up. -/
def showsDec (untr : TrReader) (b : Brings) (s : ℕ) (x xO : Msg) (c : ℕ → Coin) : Prop :=
  showsPart untr b s x xO c ∧ ∀ (i : RecIns) (spec : SessionSpec) (sts : Array Stmt),
    setupO xO = some (i, spec, sts) → PartDec xO i

/-- What each part accepts at the deployed layout, at its staged statement `xO` and its own witness, showing its
decision. -/
abbrev AccV (rd : Msg → Option FinalP) (wE : Msg → ℝ) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) (b : Brings) (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (shape : Claims.Recursion.Form → Prop) (untr : TrReader) (cmt : Msg → ℕ → Msg → Msg) :=
  PartAcceptsV FlockDeploy.sha512 (DRL rd wE final w b elem) (deployZkRec _root_.Flock.Readable.relation)
    (DeployedR.layR elem) shape (showsDec untr b) cmt

/-- **The parts' statements `xs` staged at one prefix `ms` of messages**: the first `n` of them are `staged st x ms`'s,
as an accepted run's V\* statements are (`RecDeploy.okRuns_deployRec`). The commit ties between two parts read both
parts' rows at one `ms` (`accCommit_eq`). -/
abbrev XsStaged (st : Stage) (x : Msg) (xs : List Msg) (n : ℕ) : Prop :=
  ∃ ms : List (Msg × Claims.Out), ∀ s < n, xs.getD s [] = staged st x ms s

/-- Past `xs`'s end `getD` reads `[]`, which doesn't set up (`insO_nil`): a part's staged statement that sets up is one
of `xs`. -/
theorem getD_mem_of_setup {xs : List Msg} {s : ℕ} {v : RecIns × SessionSpec × Array Stmt}
    (h : setupO (xs.getD s []) = some v) : xs.getD s [] ∈ xs := by
  rw [List.getD_eq_getElem?_getD] at h ⊢
  by_cases hs : s < xs.length
  · rw [List.getElem?_eq_getElem hs, Option.getD_some]
    exact List.getElem_mem hs
  · rw [List.getElem?_eq_none (by omega)] at h
    simp [setupO, insO_nil] at h

/-- **A piece's port context**: at a compiled statement `x` of the shape, every part accepts at its own witness and the
parts' own messages `ms = (layR elem).msgs x ws` (`JointP`: the calls of all parts true and free of collisions
together), part `s`'s V\* statement is an algebra part, and at its witness `wt = ws[s]` and a staged statement `xO`
that passes `Flock.check`, agrees with the messages and records true calls with no collision, `q` is a
`rec-residuals` piece it owns (`resPD`, `ownRD`), at table 0's part `(S, ns, nv)`, and the last message reads as
`(os, last)`. `P` is what holds there.

Per-part acceptance is not enough: one part's `PartAccepts` leaves free the rounds it doesn't hold first (and with
them the last message's trailer, which `elemReadC` reads), and the accumulator rows other parts' chain steps tie. -/
def PortsAt (rd : Msg → Option FinalP) (wE : Msg → ℝ) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) (b : Brings) (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (shape : Claims.Recursion.Form → Prop) (untr : TrReader) (cmt : Msg → ℕ → Msg → Msg)
    (P : (x : Msg) → (cs : List Coin) → (ms : List Msg) → (p : _root_.Flock.Readable.Parsed) → (ht : 0 < p.sts.size) →
      Claims.Flock.Wit → ResPiece → List ℕ → ℕ → Prop) : Prop :=
  ∀ (x : Msg) (st : Claims.Recursion.Stmt), (DRL rd wE final w b elem).parse x = some st → shape st.form →
    ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit),
    RecDeploy.OpenLower.JointP FlockDeploy.sha512 (DRL rd wE final w b elem) (DeployedR.layR elem)
      (AccP rd wE final w b elem shape untr cmt) x c ws →
    ∀ s (wt : Claims.Flock.Wit), ws[s]? = some wt →
    ∀ (r : XR), readX x = some r → ∀ (ss : Array VStar.Stmt), RecVStar.statements r.shape = .ok ss →
    ∀ (j : ℕ) (nm : String) (pt : VStar.Part), ss[s]? = some (.inner j nm pt) →
    ∀ (xO : Msg) (p : _root_.Flock.Readable.Parsed),
      _root_.Flock.Readable.parseBytes readOfO xO = some p →
      Claims.Flock.check (_root_.Flock.Readable.parseOf _root_.Flock.Readable.relation readOfO) xO wt = true →
      showsPart untr b s x xO c → (DeployedR.layR elem).Agrees cmt x s wt xO ((DeployedR.layR elem).msgs x ws) →
      (∀ e ∈ (DeployedR.layR elem).calls x s wt, FlockDeploy.sha512 e.1 = e.2) →
      (Claims.findColl ((DeployedR.layR elem).calls x s wt ++
        vcallsAt FlockDeploy.sha512 (DRL rd wE final w b elem) xO)).isSome = false →
    ∀ q ∈ DecompBR.resPD (Claims.Recursion.xI (DRL rd wE final w b elem) x), DecompBR.ownRD x q = some s →
    ∀ (ht : 0 < p.sts.size) (m k cl jj : ℕ) (sh : _root_.Flock.Shape) (holo : Bool) (S : _root_.Flock.RecOpen.GfS)
      (ns : List ℕ) (nv : ℕ), RecResiduals.params p.sts[0].c.unit.name = some (m, k, cl, jj) →
      RecResiduals.shape? m k cl = some sh → RecResiduals.holoOf p.sts[0].c = .ok holo →
      RecResiduals.part { sh with holo } jj = .ok (S, ns, nv) →
    ∀ os last, (_root_.RecDeploy.bdR rd wE final w).final
        (((DeployedR.layR elem).msgs x ws).getD
          ((_root_.RecDeploy.bdR rd wE final w).DE.V.k (Claims.Recursion.xI (DRL rd wE final w b elem) x) - 1)
          []) = some (os, last) →
      os.length + 1 = (_root_.RecDeploy.bdR rd wE final w).DE.V.k (Claims.Recursion.xI (DRL rd wE final w b elem) x) →
      P (Claims.Recursion.xI (DRL rd wE final w b elem) x)
        ((List.range ((_root_.RecDeploy.bdR rd wE final w).DE.V.k
          (Claims.Recursion.xI (DRL rd wE final w b elem) x))).map c)
        ((_root_.RecDeploy.bdR rd wE final w).opened os last) p ht wt q ns nv

/-- **`PortsAt` at the parts' staged statements `xs`**: `PortsAt`'s context, with every part's check at
`xs.getD s' []` showing its decision (`AccV`), no collision among the recorded calls and the verifier's own
evaluations at `xs` (`setupCalls`), `xs` staged at one prefix of messages (`XsStaged`), and part `s`'s statement `xO`
the one at `xs.getD s []`. This is what `JointV` over `showsDec` gives at an accepted run; a commit tie between two parts (`accCommit_eq`) reads both parts' setups and decisions, and the
collision, there (`getD_mem_of_setup` puts each staged statement in `xs`). -/
def PortsAtV (rd : Msg → Option FinalP) (wE : Msg → ℝ) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) (b : Brings) (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (shape : Claims.Recursion.Form → Prop) (untr : TrReader) (cmt : Msg → ℕ → Msg → Msg)
    (P : (x : Msg) → (cs : List Coin) → (ms : List Msg) → (p : _root_.Flock.Readable.Parsed) → (ht : 0 < p.sts.size) →
      Claims.Flock.Wit → ResPiece → List ℕ → ℕ → Prop) : Prop :=
  ∀ (x : Msg) (st : Claims.Recursion.Stmt), (DRL rd wE final w b elem).parse x = some st → shape st.form →
    ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit),
    RecDeploy.OpenLower.JointP FlockDeploy.sha512 (DRL rd wE final w b elem) (DeployedR.layR elem)
      (AccP rd wE final w b elem shape untr cmt) x c ws →
    ∀ (xs : List Msg), (∀ s' (hs' : s' < ws.length),
        AccV rd wE final w b elem shape untr cmt s' x c (xs.getD s' []) ws[s'] ((DeployedR.layR elem).msgs x ws)) →
      (Claims.findColl ((DeployedR.layR elem).allCalls x ws ++
        Claims.Recursion.setupCalls FlockDeploy.sha512 (DRL rd wE final w b elem) xs)).isSome = false →
      XsStaged b.st x xs ws.length →
    ∀ s (wt : Claims.Flock.Wit), ws[s]? = some wt →
    ∀ (r : XR), readX x = some r → ∀ (ss : Array VStar.Stmt), RecVStar.statements r.shape = .ok ss →
    ∀ (j : ℕ) (nm : String) (pt : VStar.Part), ss[s]? = some (.inner j nm pt) →
    ∀ (xO : Msg) (p : _root_.Flock.Readable.Parsed), xs.getD s [] = xO →
      _root_.Flock.Readable.parseBytes readOfO xO = some p →
      Claims.Flock.check (_root_.Flock.Readable.parseOf _root_.Flock.Readable.relation readOfO) xO wt = true →
      showsPart untr b s x xO c → (DeployedR.layR elem).Agrees cmt x s wt xO ((DeployedR.layR elem).msgs x ws) →
      (∀ e ∈ (DeployedR.layR elem).calls x s wt, FlockDeploy.sha512 e.1 = e.2) →
      (Claims.findColl ((DeployedR.layR elem).calls x s wt ++
        vcallsAt FlockDeploy.sha512 (DRL rd wE final w b elem) xO)).isSome = false →
    ∀ q ∈ DecompBR.resPD (Claims.Recursion.xI (DRL rd wE final w b elem) x), DecompBR.ownRD x q = some s →
    ∀ (ht : 0 < p.sts.size) (m k cl jj : ℕ) (sh : _root_.Flock.Shape) (holo : Bool) (S : _root_.Flock.RecOpen.GfS)
      (ns : List ℕ) (nv : ℕ), RecResiduals.params p.sts[0].c.unit.name = some (m, k, cl, jj) →
      RecResiduals.shape? m k cl = some sh → RecResiduals.holoOf p.sts[0].c = .ok holo →
      RecResiduals.part { sh with holo } jj = .ok (S, ns, nv) →
    ∀ os last, (_root_.RecDeploy.bdR rd wE final w).final
        (((DeployedR.layR elem).msgs x ws).getD
          ((_root_.RecDeploy.bdR rd wE final w).DE.V.k (Claims.Recursion.xI (DRL rd wE final w b elem) x) - 1)
          []) = some (os, last) →
      os.length + 1 = (_root_.RecDeploy.bdR rd wE final w).DE.V.k (Claims.Recursion.xI (DRL rd wE final w b elem) x) →
      P (Claims.Recursion.xI (DRL rd wE final w b elem) x)
        ((List.range ((_root_.RecDeploy.bdR rd wE final w).DE.V.k
          (Claims.Recursion.xI (DRL rd wE final w b elem) x))).map c)
        ((_root_.RecDeploy.bdR rd wE final w).opened os last) p ht wt q ns nv

/-- **P2, the frame ports**: in a piece's port context, on every row element of its group whose `w` index is a frame
value (below `accBase`), the copy's row is what `elemReadC acc` reads, at every `acc`, where the opened messages decode
(`Rows.DecAt`, which `ReadsAt` and `BasesHold` give at a joint acceptance; at one, `shI_alg` (ResLowerDA) ties `shI` to
table 0's `shapeNoCoin`). -/
def FramePorts (rd : Msg → Option FinalP) (wE : Msg → ℝ) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) (b : Brings) (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (shape : Claims.Recursion.Form → Prop) (untr : TrReader) (cmt : Msg → ℕ → Msg → Msg) : Prop :=
  PortsAt rd wE final w b elem shape untr cmt fun xi cs ms' p ht wt q ns nv =>
    Nonempty (Rows.DecAt rd xi ms' cs) →
    ∀ (acc : Msg → List Msg → List Coin → ℕ → Array _root_.Flock.F128) (e j : ℕ), elemPort xi q.key.1 e = some j →
      j < accBase xi →
      _root_.Flock.Readable.rowsRead ns nv (keyCopy p ht wt q.key) e = Rows.elemReadC acc xi ms' cs q.key e

/-- **P3, the accumulator ports**: in a piece's port context at the parts' staged statements (`PortsAtV`: every part's
setup and decision, and the collision with the verifier's evaluations, which the chain's commit ties read), at a
decision of the opened messages (`Rows.DecAt`, which `resLowered_D` derives from `ReadsAt` and the base checks), one
offset `δ` such that on every row element of its group whose `w` index is an accumulator (`accBase X + i`, `i < 4`:
`acc0 0, acc0 1, acc1 0, acc1 1`), the copy's row is what `elemReadC (Top.accRC rd key)` reads plus `δ (i % 2)`. -/
def AccPorts (rd : Msg → Option FinalP) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) (wE : Msg → ℝ)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) (b : Brings) (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (shape : Claims.Recursion.Form → Prop) (untr : TrReader) (cmt : Msg → ℕ → Msg → Msg) : Prop :=
  PortsAtV rd wE final w b elem shape untr cmt fun xi cs ms' p ht wt q ns nv =>
    Nonempty (Rows.DecAt rd xi ms' cs) →
    ∃ δ : ℕ → _root_.Flock.F128, ∀ e j, elemPort xi q.key.1 e = some j → accBase xi ≤ j →
      _root_.Flock.Readable.rowsRead ns nv (keyCopy p ht wt q.key) e =
        Rows.elemReadC (Top.accRC rd key) xi ms' cs q.key e + δ ((j - accBase xi) % 2)

/-- **`resPieces_computesAt` at a pinned witness and statement** (`PartAcceptsV`): the same share, asked only at the
part's own witness `w` and its staged statement `xO`, at any `shows` that implies `showsPart` (which `hpc` then reads
whole). -/
theorem resPieces_computesV {H : Msg → Msg} {D : Claims.Recursion.Deploy Coin} {L : LayoutB}
    {shape : Claims.Recursion.Form → Prop} {untr : TrReader} {b : Brings} {cmt : Msg → ℕ → Msg → Msg} {s : ℕ}
    {x : Msg} {c : ℕ → Coin} {xO : Msg} {w : Claims.Flock.Wit} {ms : List Msg}
    {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    (hsP : ∀ {s x xO c}, shows s x xO c → showsPart untr b s x xO c)
    (h : PartAcceptsV H D (deployZkRec _root_.Flock.Readable.relation) L shape shows cmt s x c xO w ms)
    {r : XR} (hr : readX x = some r) {ss : Array VStar.Stmt} (hss : _root_.Flock.RecVStar.statements r.shape = .ok ss)
    {j : ℕ} {nm : String} {pt : VStar.Part} (hsv : ss[s]? = some (.inner j nm pt))
    (Bd : Claims.Recursion.Bound Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace)
    (pieces : Msg → List ResPiece) (ownQ : Msg → ResPiece → Option ℕ)
    (elemAt : Msg → List Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
    (vvAt : Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
    (hpc : ∀ (p : _root_.Flock.Readable.Parsed),
      _root_.Flock.Readable.parseBytes readOfO xO = some p →
      Claims.Flock.check (_root_.Flock.Readable.parseOf _root_.Flock.Readable.relation readOfO) xO w = true →
      shows s x xO c → L.Agrees cmt x s w xO ms → (∀ e ∈ L.calls x s w, H e.1 = e.2) →
      (Claims.findColl (L.calls x s w ++ vcallsAt H D xO)).isSome = false →
      ∀ q ∈ pieces (Claims.Recursion.xI D x), ownQ x q = some s →
        ∃ ht : 0 < p.sts.size, 0 < p.sts[0].c.g ∧
          (q.key.2.1 + q.key.2.2.1) / p.sts[0].c.g < 2 ^ (p.sts[0].m - p.sts[0].c.kLog) ∧
          q.key.2.2.2 < p.sts[0].c.upv ∧
          ∀ (m k cl jj : ℕ) (sh : _root_.Flock.Shape) (holo : Bool) (S : _root_.Flock.RecOpen.GfS) (ns : List ℕ)
            (nv : ℕ), _root_.Flock.RecResiduals.params p.sts[0].c.unit.name = some (m, k, cl, jj) →
            _root_.Flock.RecResiduals.shape? m k cl = some sh →
            _root_.Flock.RecResiduals.holoOf p.sts[0].c = .ok holo →
            _root_.Flock.RecResiduals.part { sh with holo } jj = .ok (S, ns, nv) →
            S = q.S ∧ q.key.1 = jj ∧
            ResTie S ns nv
              (vvAt (Claims.Recursion.xI D x) ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) q.key)
              (keyCopy p ht w q.key) ∧
            ∀ os last, Bd.final (ms.getD (Bd.DE.V.k (Claims.Recursion.xI D x) - 1) []) = some (os, last) →
              os.length + 1 = Bd.DE.V.k (Claims.Recursion.xI D x) →
              ResZero S (_root_.Flock.Readable.rowsRead ns nv (keyCopy p ht w q.key))
                  (vvAt (Claims.Recursion.xI D x) ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) q.key) →
                ResZero S (elemAt (Claims.Recursion.xI D x) (Bd.opened os last)
                    ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) q.key)
                  (vvAt (Claims.Recursion.xI D x) ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) q.key)) :
    PiecesComputes Bd D pieces (resHolds elemAt vvAt) ownQ s x c ms := by
  obtain ⟨-, hck, hT, hc, hs, hh, hp, hm, hrw⟩ := h
  obtain ⟨p, r', sv, hx, ht, hpi, hr', ⟨ss', hss', hsv'⟩, -, htab⟩ := stagedReads (hsP hs)
  have hck' : Claims.Flock.check (_root_.Flock.Readable.parseOf _root_.Flock.Readable.relation readOfO) xO w =
      true := hck
  have hcp := _root_.Flock.Readable.stmtCopies_of_allOuts htab
    (_root_.Flock.Readable.allOuts_of_check hck' hx ht hpi)
  rw [hr] at hr'
  cases hr'
  rw [hss] at hss'
  cases hss'
  rw [hsv] at hsv'
  cases hsv'
  intro q hq ho os last hf hl
  obtain ⟨ht', hG, ho', hu, hch⟩ := hpc p hx hck' hs ⟨hh, hp, hm, hrw⟩ hT hc q hq ho
  obtain ⟨m, k, cl, jj, sh, holo, S, ns, nv, h1, h2, h3, h4, hro⟩ :=
    _root_.Flock.Readable.innerCopies ht' ho' (Nat.mod_lt _ hG) hu hcp
  obtain ⟨rfl, -, htie, hE⟩ := hch m k cl jj sh holo S ns nv h1 h2 h3 h4
  exact hE os last hf hl (resVal_zero_of_tie hro htie)

/-- **`resPieces_computesAt` at a pinned witness** (`PartAcceptsP`): `resPieces_computesV` at the statement the part's
acceptance names. -/
theorem resPieces_computesP {H : Msg → Msg} {D : Claims.Recursion.Deploy Coin} {L : LayoutB}
    {shape : Claims.Recursion.Form → Prop} {untr : TrReader} {b : Brings} {cmt : Msg → ℕ → Msg → Msg} {s : ℕ}
    {x : Msg} {c : ℕ → Coin} {w : Claims.Flock.Wit} {ms : List Msg} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    (hsP : ∀ {s x xO c}, shows s x xO c → showsPart untr b s x xO c)
    (h : PartAcceptsP H D (deployZkRec _root_.Flock.Readable.relation) L shape shows cmt s x c w ms)
    {r : XR} (hr : readX x = some r) {ss : Array VStar.Stmt} (hss : _root_.Flock.RecVStar.statements r.shape = .ok ss)
    {j : ℕ} {nm : String} {pt : VStar.Part} (hsv : ss[s]? = some (.inner j nm pt))
    (Bd : Claims.Recursion.Bound Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace)
    (pieces : Msg → List ResPiece) (ownQ : Msg → ResPiece → Option ℕ)
    (elemAt : Msg → List Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
    (vvAt : Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
    (hpc : ∀ (xO : Msg) (p : _root_.Flock.Readable.Parsed),
      _root_.Flock.Readable.parseBytes readOfO xO = some p →
      Claims.Flock.check (_root_.Flock.Readable.parseOf _root_.Flock.Readable.relation readOfO) xO w = true →
      shows s x xO c → L.Agrees cmt x s w xO ms → (∀ e ∈ L.calls x s w, H e.1 = e.2) →
      (Claims.findColl (L.calls x s w ++ vcallsAt H D xO)).isSome = false →
      ∀ q ∈ pieces (Claims.Recursion.xI D x), ownQ x q = some s →
        ∃ ht : 0 < p.sts.size, 0 < p.sts[0].c.g ∧
          (q.key.2.1 + q.key.2.2.1) / p.sts[0].c.g < 2 ^ (p.sts[0].m - p.sts[0].c.kLog) ∧
          q.key.2.2.2 < p.sts[0].c.upv ∧
          ∀ (m k cl jj : ℕ) (sh : _root_.Flock.Shape) (holo : Bool) (S : _root_.Flock.RecOpen.GfS) (ns : List ℕ)
            (nv : ℕ), _root_.Flock.RecResiduals.params p.sts[0].c.unit.name = some (m, k, cl, jj) →
            _root_.Flock.RecResiduals.shape? m k cl = some sh →
            _root_.Flock.RecResiduals.holoOf p.sts[0].c = .ok holo →
            _root_.Flock.RecResiduals.part { sh with holo } jj = .ok (S, ns, nv) →
            S = q.S ∧ q.key.1 = jj ∧
            ResTie S ns nv
              (vvAt (Claims.Recursion.xI D x) ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) q.key)
              (keyCopy p ht w q.key) ∧
            ∀ os last, Bd.final (ms.getD (Bd.DE.V.k (Claims.Recursion.xI D x) - 1) []) = some (os, last) →
              os.length + 1 = Bd.DE.V.k (Claims.Recursion.xI D x) →
              ResZero S (_root_.Flock.Readable.rowsRead ns nv (keyCopy p ht w q.key))
                  (vvAt (Claims.Recursion.xI D x) ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) q.key) →
                ResZero S (elemAt (Claims.Recursion.xI D x) (Bd.opened os last)
                    ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) q.key)
                  (vvAt (Claims.Recursion.xI D x) ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) q.key)) :
    PiecesComputes Bd D pieces (resHolds elemAt vvAt) ownQ s x c ms := by
  obtain ⟨hst, xO, hck, hT, hc, hs, hh, hp, hm, hrw⟩ := h
  exact resPieces_computesV hsP ⟨hst, hck, hT, hc, hs, hh, hp, hm, hrw⟩ hr hss hsv Bd pieces ownQ elemAt vvAt
    (hpc xO)

/-- **`ResLowered` at the parts' own witnesses**: at a compiled statement of the shape where every part accepts at its
own witness and the parts' own messages (`JointP`), each algebra part's `rec-residuals` pieces hold
(`resHolds elemAt vvAt`). This is the acceptance KS holds where it reads `hres` (`RestLowR`'s `hJ`). -/
def ResLoweredP (rd : Msg → Option FinalP) (wE : Msg → ℝ) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) (b : Brings) (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (shape : Claims.Recursion.Form → Prop) (untr : TrReader) (cmt : Msg → ℕ → Msg → Msg)
    (pieces : Msg → List ResPiece) (ownQ : Msg → ResPiece → Option ℕ)
    (elemAt : Msg → List Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
    (vvAt : Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128) : Prop :=
  ∀ (x : Msg) (st : Claims.Recursion.Stmt), (DRL rd wE final w b elem).parse x = some st → shape st.form →
    ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit),
    RecDeploy.OpenLower.JointP FlockDeploy.sha512 (DRL rd wE final w b elem) (DeployedR.layR elem)
      (AccP rd wE final w b elem shape untr cmt) x c ws →
    ∀ s < (DRL rd wE final w b elem).nS x,
      PiecesComputes (_root_.RecDeploy.bdR rd wE final w) (DRL rd wE final w b elem) pieces (resHolds elemAt vvAt) ownQ
        s x c ((DeployedR.layR elem).msgs x ws)

end

end RecDeploy.ResLower

namespace RecDeploy.DecompBR

open Flock
open Claims (Msg Out)
open FlockDeploy (Coin FinalP)
open Claims.Recursion.VStarB

/-- **`ResLowJP` at the deployed layout from `ResLoweredP`**: the two are one statement at the brings `bB`, `Dense`,
`recRead` and `cmtTr recRead`. -/
theorem resLowJP_of_resLoweredP (rd : Msg → Option FinalP) (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round)
    (nOf : Msg → ℕ → ℕ) (hd : Msg → Flock.RecRecord.Record) (rest : Msg → List (Msg × Out) → List UInt8)
    (wE : Msg → ℝ) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (pieces : Msg → List ResPiece) (ownQ : Msg → ResPiece → Option ℕ)
    (elemAt : Msg → List Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
    (vvAt : Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
    (h : ResLower.ResLoweredP rd wE final w (bB b aux nOf hd rest) elem Claims.Recursion.Form.Dense recRead
      (cmtTr recRead) pieces ownQ elemAt vvAt) :
    ResLowJP rd b aux nOf hd rest wE final w (DeployedR.layR elem) pieces elemAt vvAt ownQ :=
  fun x st hx hf c ws hJ s hs => h x st hx hf c ws hJ s hs

end RecDeploy.DecompBR
