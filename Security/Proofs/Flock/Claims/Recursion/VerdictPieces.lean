import Proofs.Flock.Claims.Recursion.VerdictParts
import Proofs.Flock.Claims.Recursion.FinalPiece

/-!
# The verdict's pieces as the parts compute them

`decompB_lastSplitP` asks one lowering fact per part (`hlow`): what the part's circuit accepts implies every check of
`IVStar` the inventory gives it (`PartComputes`), the verdict's pieces among them. This file splits that fact in two and
states the `rec-residuals` pieces' half.

* **The two shares** (`PiecesComputes`, `RestComputes`, `partComputes_of_shares`): the pieces of the verdict that
  `ownQ` gives the part, each at the messages the last message opens and the coins (as `IVStar`'s `elem` checks read
  them), and every other check it owns. Together they are `PartComputes` (`mem_pieces_of_check` finds a piece's check
  among the split's).
* **The composition at the shares** (`decompB_lastSplitP_shares`): `decompB_lastSplitP` with `hlow` given as the two
  shares, so the pieces' share can be proved apart from the reading, the salts and the rows.
* **The `rec-residuals` pieces' share** (`ResPiece`, `ResZero`, `resHolds`, `resPieces_computes`): a piece keyed by its
  group and a copy of its part's residuals table, `(jj, o, g, u)`, holds where its group's residuals vanish at the
  values the opened messages give (`elemAt`) and the verifier's values at the coins (`vvAt`). At an `.inner` part,
  `partAccepts_resZero` gives it from two hypotheses: check 5's tie at those verifier values (`ResTie`, waiting on the
  vcalls ask, `note:20261024T2340Z-draft-vcalls-ask`), and the layout fact `hcopy`: each piece `ownQ` gives the part is one of its
  statement's copies, with that copy's group, and the elements the copy reads are those `elemAt` reads off the opened
  messages. `rowsRead_msgs` gives the second half of `hcopy` at the raw messages wherever a port is a carried row; the
  step to the opened messages reads `final`, the concrete statement's.

**What stays open for `hpieces`.** `decompB_lastSplitP` also takes `hpieces`: wherever the reading and the four base
checks hold and every piece does, `verifyJPWith` accepts. The `rec-residuals` pieces give the residuals of
`GfResiduals`' groups, which is `AlgebraHolds` per rep (`FlockVBridge.algebraHolds_of_parts`). A rep's run
(`verifyRepPWith`) needs more: the accumulator rows (`AccHolds`), the decoded proof's values being the parts'
(`RunsFrom`'s decode fields) and the Ligerito openings (`Opened`), each a piece of its own owner. The rounds' frames
(`FramedAt`) are custody's (`.after`: each recorded round retains its message, and `decR` reads the messages as the
proof's framed bytes). VBridge's chain (`FlockVBridge.Family.runs_of_files`) assembles them at `sh.holo = false`, the
only form the route covers (`SchedMatches`; `not_schedMatches_holo`).
-/

namespace RecDeploy

open Claims (Msg)
open FlockDeploy (Coin FinalP nextR nextR_coin ReadOkR VerdictR holdsR)
open Claims.Recursion.VStarB

noncomputable section

/-! ## The two shares of a part's checks -/

/-- **Part `s`'s share of the verdict's pieces**: at a compiled statement `x`, each piece at the inner statement that
`ownQ` gives part `s` holds at the messages the last message opens and the coins, wherever the last message reads with
one row list per earlier round (as `IVStar`'s `elem` checks read it). -/
def PiecesComputes (Bd : Claims.Recursion.Bound Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace)
    (D : Claims.Recursion.Deploy Coin) {Q : Type} (pieces : Msg → List Q)
    (pholds : Q → Msg → List Msg → List Coin → Prop) (ownQ : Msg → Q → Option ℕ) (s : ℕ) (x : Msg) (c : ℕ → Coin)
    (ms : List Msg) : Prop :=
  ∀ q ∈ pieces (Claims.Recursion.xI D x), ownQ x q = some s →
    ∀ os last, Bd.final (ms.getD (Bd.DE.V.k (Claims.Recursion.xI D x) - 1) []) = some (os, last) →
      os.length + 1 = Bd.DE.V.k (Claims.Recursion.xI D x) →
      pholds q (Claims.Recursion.xI D x) (Bd.opened os last)
        ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c)

/-- **Part `s`'s share of the other checks**: every check of `IVStar` that the inventory gives part `s` and that is no
piece of the verdict holds at the coins and messages. -/
def RestComputes {Q : Type} (H : Msg → Msg) (D : Claims.Recursion.Deploy Coin)
    (Bd : Claims.Recursion.Bound Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace)
    (S : ElemSplit H Bd (DeployCheck (ReadCheck (VerdictP Q))))
    (own : Msg → IVCheck (DeployCheck (ReadCheck (VerdictP Q))) → Option ℕ) (s : ℕ) (x : Msg) (c : ℕ → Coin)
    (ms : List Msg) : Prop :=
  ∀ q ∈ ivChecks Bd S (Claims.Recursion.xI D x), own x q = some s → (∀ p, q ≠ .elem (.last (.verdict (.piece p)))) →
    q.Holds H Bd S (Claims.Recursion.xI D x) c ms

section Shares

variable (H : Msg → Msg) (rd : Msg → Option FinalP) {Q : Type} (pieces : Msg → List Q)
  (pholds : Q → Msg → List Msg → List Coin → Prop)
  (hpieces : ∀ (x : Msg) (ms : List Msg) (cs : List Coin), ReadOkR rd x ms cs →
    (∀ b ∈ VerdictP.bases, holdsR H rd b x ms cs) → (∀ q ∈ pieces x, pholds q x ms cs) →
      holdsR H rd .verdict x ms cs)
  (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
  (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ)
  (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)

/-- **A piece's check is among the split's only as a piece of the statement**: where the element-level check of piece
`p` is among `IVStar`'s checks at `x`, `p` is among the pieces at `x`. -/
theorem mem_pieces_of_check {x : Msg} {p : Q}
    (h : IVCheck.elem (DeployCheck.last (ReadCheck.verdict (VerdictP.piece p))) ∈
      ivChecks (bdDeployed rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w)
        (deploySplit H rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w (nextR_coin rd H)
          (lastSplitP H rd pieces pholds hpieces).toLast) x) :
    p ∈ pieces x := by
  have h1 := mem_ivChecks_elem _ _ x h
  change DeployCheck.last _ ∈ DeployCheck.setup ::
    ((List.range ((FlockDeploy.V false).k x - 1)).map DeployCheck.tag ++
      ((lastSplitP H rd pieces pholds hpieces).toLast.checks x).map DeployCheck.last) at h1
  rcases List.mem_cons.1 h1 with h2 | h2
  · cases h2
  rcases List.mem_append.1 h2 with h3 | h3
  · obtain ⟨_, _, h4⟩ := List.mem_map.1 h3
    cases h4
  obtain ⟨r, hr, h4⟩ := List.mem_map.1 h3
  cases h4
  change ReadCheck.verdict _ ∈ ReadCheck.read ::
    ((lastSplitP H rd pieces pholds hpieces).checks x).map ReadCheck.verdict at hr
  rcases List.mem_cons.1 hr with h5 | h5
  · cases h5
  obtain ⟨r', hr', h6⟩ := List.mem_map.1 h5
  cases h6
  change VerdictP.piece p ∈ VerdictP.bases.map VerdictP.base ++ (pieces x).map VerdictP.piece at hr'
  rcases List.mem_append.1 hr' with h7 | h7
  · obtain ⟨_, _, h8⟩ := List.mem_map.1 h7
    cases h8
  obtain ⟨p', hp', h8⟩ := List.mem_map.1 h7
  cases h8
  exact hp'

/-- **The two shares are the part's `PartComputes`** at `lastSplitP`'s split and its inventory. -/
theorem partComputes_of_shares (ownB : Msg → VerdictR → Option ℕ) (ownQ : Msg → Q → Option ℕ)
    (D : Claims.Recursion.Deploy Coin) (L : LayoutB) {s : ℕ} {x : Msg} {c : ℕ → Coin} {ms : List Msg}
    (hrest : RestComputes H D (bdDeployed rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w)
      (deploySplit H rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w (nextR_coin rd H)
        (lastSplitP H rd pieces pholds hpieces).toLast)
      (ownHolders D L (ownDeploy (ownJ D L (ownP ownB ownQ)))) s x c ms)
    (hp : PiecesComputes (bdDeployed rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w) D pieces pholds ownQ s
      x c ms) :
    PartComputes H D (bdDeployed rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w)
      (deploySplit H rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w (nextR_coin rd H)
        (lastSplitP H rd pieces pholds hpieces).toLast)
      (ownHolders D L (ownDeploy (ownJ D L (ownP ownB ownQ)))) s x c ms := by
  intro q hq ho
  by_cases hpc : ∃ p, q = .elem (.last (.verdict (.piece p)))
  · obtain ⟨p, rfl⟩ := hpc
    have hp' := mem_pieces_of_check H rd pieces pholds hpieces rel colsE wE rowIn final w hq
    intro os last hf hl
    exact hp p hp' ho os last hf hl
  · exact hrest q hq ho fun p hqp => hpc ⟨p, hqp⟩

/-- **The composition at the shares**: `decompB_lastSplitP` with each part's lowering fact given as its share of the
verdict's pieces (`hlowP`) and its share of the other checks (`hlowR`). -/
theorem decompB_lastSplitP_shares (ownB : Msg → VerdictR → Option ℕ) (ownQ : Msg → Q → Option ℕ)
    (rel' : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings) (L : LayoutB)
    (IV : Claims.Verifier Coin) (shape : Claims.Recursion.Form → Prop) {Col' T' : Type}
    (DO : Claims.Flock.Deploy Coin Col' T') (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (cmt : Msg → ℕ → Msg → Msg)
    (hR : ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement), (recBound rel' b L IV).parse x = some st → shape st.form →
      ∀ r ∈ (lastSplitP H rd pieces pholds hpieces).checks (Claims.Recursion.xI (recBound rel' b L IV) x), ∀ s,
        ownP ownB ownQ x r = some s → s < (recBound rel' b L IV).nS x)
    (hrest : ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement), (recBound rel' b L IV).parse x = some st → shape st.form →
      ∀ q ∈ ivChecks (bdDeployed rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w)
          (deploySplit H rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w (nextR_coin rd H)
            (lastSplitP H rd pieces pholds hpieces).toLast)
          (Claims.Recursion.xI (recBound rel' b L IV) x),
        ownHolders (recBound rel' b L IV) L (ownDeploy (ownJ (recBound rel' b L IV) L (ownP ownB ownQ))) x q = none →
        (∀ r, q = .elem r → ∃ r', r = .last r') →
        ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit), ws.length = (recBound rel' b L IV).nS x →
          (∀ e ∈ L.allCalls x ws, H e.1 = e.2) →
          q.Holds H (bdDeployed rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w)
            (deploySplit H rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w (nextR_coin rd H)
              (lastSplitP H rd pieces pholds hpieces).toLast)
            (Claims.Recursion.xI (recBound rel' b L IV) x) c (L.msgs x ws))
    (hlowR : ∀ s x c ms, PartAccepts H (recBound rel' b L IV) DO L shape shows cmt s x c ms →
      RestComputes H (recBound rel' b L IV) (bdDeployed rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w)
        (deploySplit H rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w (nextR_coin rd H)
          (lastSplitP H rd pieces pholds hpieces).toLast)
        (ownHolders (recBound rel' b L IV) L (ownDeploy (ownJ (recBound rel' b L IV) L (ownP ownB ownQ)))) s x c ms)
    (hlowP : ∀ s x c ms, PartAccepts H (recBound rel' b L IV) DO L shape shows cmt s x c ms →
      PiecesComputes (bdDeployed rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w) (recBound rel' b L IV)
        pieces pholds ownQ s x c ms) :
    DecompB H (recBound rel' b L IV) (bdDeployed rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w).IVStar L
      shape (PartAccepts H (recBound rel' b L IV) DO L shape shows cmt) :=
  decompB_lastSplitP H rd pieces pholds hpieces rel colsE wE rowIn final w ownB ownQ rel' b L IV shape DO shows cmt hR
    hrest fun s x c ms h =>
      partComputes_of_shares H rd pieces pholds hpieces rel colsE wE rowIn final w ownB ownQ _ L (hlowR s x c ms h)
        (hlowP s x c ms h)

end Shares

/-! ## The `rec-residuals` pieces -/

/-- **A `rec-residuals` piece**: its key `(jj, o, g, u)`, the verifier's group `jj` its part computes and instance
`o + g`'s unit `u` of that part's one table (`ResLower.keyCopy`, block `(o + g) / c.g` and VU `(o + g) % c.g` at the
part's own `c.g`; `ResLower.resPieces_computesAt`; `resPieces_computes` reads the first component as the table
instead), and the group of residuals it computes. -/
structure ResPiece where
  key : ℕ × ℕ × ℕ × ℕ
  S : _root_.Flock.RecOpen.GfS

/-- **A group's residuals vanish** at the elements `E` and the verifier's values `vv`. -/
def ResZero (S : _root_.Flock.RecOpen.GfS) (E vv : ℕ → _root_.Flock.F128) : Prop :=
  ∀ q (hq : q < S.res.length),
    FlockVBridge.resVal (FlockVBridge.gfsOf S) (FlockVBridge.termVal E vv) (FlockVBridge.resOf S.res[q]) = 0

/-- **What a `rec-residuals` piece says** at the inner statement `x`, the opened messages `ms` and the coins `cs`: its
group's residuals vanish at the elements `elemAt` reads off the messages and the coins at its copy (the accumulators'
ports are a chain the coins step) and the verifier's values `vvAt` at the coins there. -/
def resHolds (elemAt : Msg → List Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
    (vvAt : Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128) (q : ResPiece) (x : Msg) (ms : List Msg)
    (cs : List Coin) : Prop :=
  ResZero q.S (elemAt x ms cs q.key) (vvAt x cs q.key)

/-- **The `rec-residuals` pieces' share at an `.inner` part**: `partAccepts_resZero` at the verifier's values the coins
give (`vvAt`), where check 5's tie holds there (`htie`) and the layout places each piece `ownQ` gives the part at one
of its statement's copies, of that copy's group, reading what `elemAt` reads off the opened messages (`hcopy`). -/
theorem resPieces_computes {H : Msg → Msg} {D : Claims.Recursion.Deploy Coin} {L : LayoutB}
    {shape : Claims.Recursion.Form → Prop} {untr : TrReader} {b : Brings} {cmt : Msg → ℕ → Msg → Msg} {s : ℕ}
    {x : Msg} {c : ℕ → Coin} {ms : List Msg}
    (h : PartAccepts H D (deployZkRec _root_.Flock.Readable.relation) L shape (showsPart untr b) cmt s x c ms)
    {r : XR} (hr : readX x = some r) {ss : Array VStar.Stmt} (hss : _root_.Flock.RecVStar.statements r.shape = .ok ss)
    {j : ℕ} {nm : String} {pt : VStar.Part} (hsv : ss[s]? = some (.inner j nm pt))
    (Bd : Claims.Recursion.Bound Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace)
    (pieces : Msg → List ResPiece) (ownQ : Msg → ResPiece → Option ℕ)
    (elemAt : Msg → List Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
    (vvAt : Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
    (htie : ∀ (xO : Msg) (w : Claims.Flock.Wit) (p : _root_.Flock.Readable.Parsed),
      _root_.Flock.Readable.parseBytes readOfO xO = some p → L.Agrees cmt x s w xO ms →
      (∀ e ∈ L.calls x s w, H e.1 = e.2) → (Claims.findColl (L.calls x s w ++ vcallsAt H D xO)).isSome = false →
      showsTr untr s x xO c →
      ∀ t (ht : t < p.sts.size) (o g u : ℕ), o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog) → g < p.sts[t].c.g →
        u < p.sts[t].c.upv →
        ∀ (m k cl jj : ℕ) (sh : _root_.Flock.Shape) (holo : Bool) (S : _root_.Flock.RecOpen.GfS) (ns : List ℕ)
          (nv : ℕ), _root_.Flock.RecResiduals.params p.sts[t].c.unit.name = some (m, k, cl, jj) →
          _root_.Flock.RecResiduals.shape? m k cl = some sh → _root_.Flock.RecResiduals.holoOf p.sts[t].c = .ok holo →
          _root_.Flock.RecResiduals.part { sh with holo } jj = .ok (S, ns, nv) →
          ResTie S ns nv
            (vvAt (Claims.Recursion.xI D x) ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) (t, o, g, u))
            (_root_.Flock.Readable.copyOf p t ht w o g u))
    (hcopy : ∀ (xO : Msg) (w : Claims.Flock.Wit) (p : _root_.Flock.Readable.Parsed),
      _root_.Flock.Readable.parseBytes readOfO xO = some p → L.RowsRead x s w ms →
      ∀ q ∈ pieces (Claims.Recursion.xI D x), ownQ x q = some s →
        ∃ ht : q.key.1 < p.sts.size, q.key.2.1 < 2 ^ (p.sts[q.key.1].m - p.sts[q.key.1].c.kLog) ∧
          q.key.2.2.1 < p.sts[q.key.1].c.g ∧ q.key.2.2.2 < p.sts[q.key.1].c.upv ∧
          ∀ (m k cl jj : ℕ) (sh : _root_.Flock.Shape) (holo : Bool) (S : _root_.Flock.RecOpen.GfS) (ns : List ℕ)
            (nv : ℕ), _root_.Flock.RecResiduals.params p.sts[q.key.1].c.unit.name = some (m, k, cl, jj) →
            _root_.Flock.RecResiduals.shape? m k cl = some sh →
            _root_.Flock.RecResiduals.holoOf p.sts[q.key.1].c = .ok holo →
            _root_.Flock.RecResiduals.part { sh with holo } jj = .ok (S, ns, nv) →
            S = q.S ∧ ∀ os last, Bd.final (ms.getD (Bd.DE.V.k (Claims.Recursion.xI D x) - 1) []) = some (os, last) →
              os.length + 1 = Bd.DE.V.k (Claims.Recursion.xI D x) →
              _root_.Flock.Readable.rowsRead ns nv
                  (_root_.Flock.Readable.copyOf p q.key.1 ht w q.key.2.1 q.key.2.2.1 q.key.2.2.2) =
                elemAt (Claims.Recursion.xI D x) (Bd.opened os last)
                  ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) q.key) :
    PiecesComputes Bd D pieces (resHolds elemAt vvAt) ownQ s x c ms := by
  obtain ⟨xO, w, p, hx, hRR, hall⟩ := partAccepts_resZero h hr hss hsv
    (fun t o g u => vvAt (Claims.Recursion.xI D x)
      ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) (t, o, g, u)) htie
  intro q hq ho os last hf hl
  obtain ⟨ht, ho', hg, hu, hch⟩ := hcopy xO w p hx hRR q hq ho
  obtain ⟨m, k, cl, jj, sh, holo, S, ns, nv, h1, h2, h3, h4, hz, -⟩ :=
    hall q.key.1 ht q.key.2.1 q.key.2.2.1 q.key.2.2.2 ho' hg hu
  obtain ⟨rfl, hE⟩ := hch m k cl jj sh holo S ns nv h1 h2 h3 h4
  have hz' : ResZero q.S (_root_.Flock.Readable.rowsRead ns nv
      (_root_.Flock.Readable.copyOf p q.key.1 ht w q.key.2.1 q.key.2.2.1 q.key.2.2.2))
      (vvAt (Claims.Recursion.xI D x) ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) q.key) := hz
  rw [hE os last hf hl] at hz'
  exact hz'

end

end RecDeploy
