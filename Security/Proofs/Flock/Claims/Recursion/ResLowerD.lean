import Proofs.Flock.Claims.Recursion.ResLowJD
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
* `FramePorts`: on every frame port, the copy's row is `elemReadC acc`'s, at every `acc` (P2, CAPPIECES).
* `AccPorts`: on the accumulator ports, the copy's row is `elemReadC (Top.accRC rd key)`'s plus one `δ` (P3,
  RECDCOIN).
* `resLowered_D`: `ResLowered` at the deployed constants from the two (P1, LOWRES).

Nothing downstream of `ResLowered` reads `accRC`'s values themselves: `ResLowered` concludes `resHolds`, the vanishing
of each piece's residuals (`ResZero`), and that is all `resLowJ_of_resLowered` and KS's row R take from it.
-/

namespace RecDeploy.ResLower

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP setupOf)
open Claims.Recursion.VStarB

noncomputable section

/-- Where the accumulators start in the inner statement's `w`: past table 0's rounds' elements (`portSizes`). -/
def accBase (X : Msg) : ℕ :=
  match setupOf false X with
  | some v => (_root_.Flock.RecAlgebra.rounds (Rows.shapeNoCoin v.2.1 v.2.2)).flatten.size
  | none => 0

/-- Row element `e` of group `p`, as an index of the inner statement's `w` (`Rows.portAt`). -/
def elemPort (X : Msg) (p e : ℕ) : Option ℕ :=
  match setupOf false X with
  | some v => Rows.portAt (Rows.shapeNoCoin v.2.1 v.2.2) p e
  | none => none

/-- **The accumulators' offset** at group `p`'s row element `e`: `δ c` at `acc0 c` and `acc1 c` (`w`'s elements
`accBase X + c` and `accBase X + 2 + c`), `0` elsewhere. -/
def accOff (X : Msg) (p : ℕ) (δ : ℕ → _root_.Flock.F128) (e : ℕ) : _root_.Flock.F128 :=
  match elemPort X p e with
  | some j => if accBase X ≤ j then δ ((j - accBase X) % 2) else 0
  | none => 0

/-- The compiled record at the deployed layout. -/
abbrev DRL (rd : Msg → Option FinalP) (wE : Msg → ℝ) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) (b : Brings) (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) :
    Claims.Recursion.Deploy Coin :=
  recBound _root_.Flock.Readable.relation b (DeployedR.layR elem) (_root_.RecDeploy.bdR rd wE final w).IVStar

/-- **A piece's port context**: part `s` accepts at `x` (`PartAccepts`), its V\* statement is an algebra part, and at a
staged statement `xO` and witness `wt` that pass `Flock.check`, agree with the messages, record true calls and no
collision, `q` is a `rec-residuals` piece it owns (`resPD`, `ownRD`), at table 0's part `(S, ns, nv)`, and the last
message reads as `(os, last)`. `P` is what holds there. -/
def PortsAt (rd : Msg → Option FinalP) (wE : Msg → ℝ) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) (b : Brings) (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (shape : Claims.Recursion.Form → Prop) (untr : TrReader) (cmt : Msg → ℕ → Msg → Msg)
    (P : (x : Msg) → (cs : List Coin) → (ms : List Msg) → (p : _root_.Flock.Readable.Parsed) → (ht : 0 < p.sts.size) →
      Claims.Flock.Wit → ResPiece → List ℕ → ℕ → Prop) : Prop :=
  ∀ s x c ms, PartAccepts FlockDeploy.sha512 (DRL rd wE final w b elem) (deployZkRec _root_.Flock.Readable.relation)
      (DeployedR.layR elem) shape (showsPart untr b) cmt s x c ms →
    ∀ (r : XR), readX x = some r → ∀ (ss : Array VStar.Stmt), RecVStar.statements r.shape = .ok ss →
    ∀ (j : ℕ) (nm : String) (pt : VStar.Part), ss[s]? = some (.inner j nm pt) →
    ∀ (xO : Msg) (wt : Claims.Flock.Wit) (p : _root_.Flock.Readable.Parsed),
      _root_.Flock.Readable.parseBytes readOfO xO = some p →
      Claims.Flock.check (_root_.Flock.Readable.parseOf _root_.Flock.Readable.relation readOfO) xO wt = true →
      showsPart untr b s x xO c → (DeployedR.layR elem).Agrees cmt x s wt xO ms →
      (∀ e ∈ (DeployedR.layR elem).calls x s wt, FlockDeploy.sha512 e.1 = e.2) →
      (Claims.findColl ((DeployedR.layR elem).calls x s wt ++
        vcallsAt FlockDeploy.sha512 (DRL rd wE final w b elem) xO)).isSome = false →
    ∀ q ∈ DecompBR.resPD (Claims.Recursion.xI (DRL rd wE final w b elem) x), DecompBR.ownRD x q = some s →
    ∀ (ht : 0 < p.sts.size) (m k cl jj : ℕ) (sh : _root_.Flock.Shape) (holo : Bool) (S : _root_.Flock.RecOpen.GfS)
      (ns : List ℕ) (nv : ℕ), RecResiduals.params p.sts[0].c.unit.name = some (m, k, cl, jj) →
      RecResiduals.shape? m k cl = some sh → RecResiduals.holoOf p.sts[0].c = .ok holo →
      RecResiduals.part { sh with holo } jj = .ok (S, ns, nv) →
    ∀ os last, (_root_.RecDeploy.bdR rd wE final w).final
        (ms.getD ((_root_.RecDeploy.bdR rd wE final w).DE.V.k (Claims.Recursion.xI (DRL rd wE final w b elem) x) - 1)
          []) = some (os, last) →
      os.length + 1 = (_root_.RecDeploy.bdR rd wE final w).DE.V.k (Claims.Recursion.xI (DRL rd wE final w b elem) x) →
      P (Claims.Recursion.xI (DRL rd wE final w b elem) x)
        ((List.range ((_root_.RecDeploy.bdR rd wE final w).DE.V.k
          (Claims.Recursion.xI (DRL rd wE final w b elem) x))).map c)
        ((_root_.RecDeploy.bdR rd wE final w).opened os last) p ht wt q ns nv

/-- **P2, the frame ports**: in a piece's port context, on every row element of its group whose `w` index is a frame
value (below `accBase`), the copy's row is what `elemReadC acc` reads, at every `acc`. -/
def FramePorts (rd : Msg → Option FinalP) (wE : Msg → ℝ) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) (b : Brings) (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (shape : Claims.Recursion.Form → Prop) (untr : TrReader) (cmt : Msg → ℕ → Msg → Msg) : Prop :=
  PortsAt rd wE final w b elem shape untr cmt fun X cs ms' p ht wt q ns nv =>
    ∀ (acc : Msg → List Msg → List Coin → ℕ → Array _root_.Flock.F128) (e j : ℕ), elemPort X q.key.1 e = some j →
      j < accBase X →
      _root_.Flock.Readable.rowsRead ns nv (keyCopy p ht wt q.key) e = Rows.elemReadC acc X ms' cs q.key e

/-- **P3, the accumulator ports**: in a piece's port context, one offset `δ` such that on every row element of its group
whose `w` index is an accumulator (`accBase X + i`, `i < 4`: `acc0 0, acc0 1, acc1 0, acc1 1`), the copy's row is what
`elemReadC (Top.accRC rd key)` reads plus `δ (i % 2)`. -/
def AccPorts (rd : Msg → Option FinalP) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) (wE : Msg → ℝ)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) (b : Brings) (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (shape : Claims.Recursion.Form → Prop) (untr : TrReader) (cmt : Msg → ℕ → Msg → Msg) : Prop :=
  PortsAt rd wE final w b elem shape untr cmt fun X cs ms' p ht wt q ns nv =>
    ∃ δ : ℕ → _root_.Flock.F128, ∀ e j, elemPort X q.key.1 e = some j → accBase X ≤ j →
      _root_.Flock.Readable.rowsRead ns nv (keyCopy p ht wt q.key) e =
        Rows.elemReadC (Top.accRC rd key) X ms' cs q.key e + δ ((j - accBase X) % 2)

end

end RecDeploy.ResLower
