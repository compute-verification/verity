import Proofs.Flock.Claims.Recursion.OpenPieces
import Proofs.Flock.Claims.Recursion.SchedDense
import Proofs.Flock.Claims.Recursion.VerdictRowDCover
import Proofs.Flock.Claims.Recursion.SlotsCheck

/-!
# The `rec-open` pieces' lowering at the deployed layout (`OpenCopy`, `OpenTie`)

`openPieces_computes` reads a part's `rec-open` pieces off `PartAccepts`, from the layout fact `hcopy` (`OpenCopy`) and
the registration ties `htie` (`OpenTie`). At the deployed layout neither holds in the form it takes them:

* `hcopy` is handed only `L.RowsRead`, which says nothing at a layout whose messages carry no shared row (every
  deployed one: `rowIn = none`). The copy's row and siblings are tied to the last message only by the part's pieces of
  it (`LayoutB.Agrees`' second conjunct), which `hcopy` isn't given.
* Both are asked at every `ms` agreeing with part `s` alone, but the side a piece opens against is read off the whole
  last message (`finalR`'s trailer, `rdR`'s framing) and every opened round (`readRep`'s fit), and its top is the
  RecCap part's. Other parts' bytes are free in such an `ms`. `DecompB` reads the parts' checks only at the parts' own
  messages `L.msgs x ws`, every part accepting there (`decompB_of_joint`), where every piece is in place.

So this file states the lowering in the form the deployed layout meets:

* `Joint`, `decompB_of_joint`: `DecompB` at `chk'` from `DecompB` at `chk`, wherever every part's `chk'` at the parts'
  own messages gives every part's `chk` there. `decompB_lastSplitP_sharesJ` is `decompB_lastSplitP_shares` with the
  pieces' share asked there.
* `openPieces_computesA`: `openPieces_computes` with one hypothesis `hopen`, given everything `partAccepts_staged`
  names (the part's whole agreement `LayoutB.Agrees` among it) and naming the piece's copy, at which `OpenCopy` and
  `OpenTie` hold.
* The deployed choices: `ownQR` (a piece of table 0 at level `ℓ` to V\*'s statement `.recOpen ℓ …`), `copyAtR` (rep
  `r`'s opening `j`, instance `r q + j`, at block `n / g`, VU `n % g`, copy 0 of the part's one table), and `sideAtR`
  (the side `RecDeploy.Rows.sideAt` reads, without `readRep`'s choice: the table's schedule `schAt`, the cap of round
  0's rows and the last message's proof with the recursive caps the rep's rounds commit, which is what `readRep` keeps
  wherever it fits).
* `OpenCopyR`, `OpenTieR`: the two facts at a layout `L`, bound record `Bd` and side `side` (`sideAtR rd`, or
  `RecDeploy.Rows.sideAt`), at the parts' own messages with every part accepting. `openPieces_deployed` is
  `PiecesComputes` at them.
* `OpenCopy`'s level shape is proved (`levelMatch`: the level statement's `LANES`, `H` and `n` are `schAt`'s level's,
  at a dense statement), so `OpenCopyR` follows from the side's schedule (`SideSchR`, `sideAtR`'s outright), the copy's
  range (`CopyFitsR`) and its opened row and siblings (`CopyOpensR`) alone (`openCopyR_of`). `CopyFitsR` is the pieces'
  range (`PiecesInRangeR`) and the staged circuit's capacity (`StagedFitsR`) (`copyFitsR_of`). `OpenTieR` is the
  position bits (`CopyBitsR`, from check 5's `dirs` tie `CopyDirsR`: `copyBitsR_of`) and the top (`CopyTopR`).
-/

namespace RecDeploy.OpenLower

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP setupOf tabsOf repCtx repLens cutLens rowsOf topGroups nextR nextR_coin ReadOkR
  VerdictR holdsR)
open Claims.Recursion.VStarB
open FlockSoundness.Refine (openAt)
open FlockVBridge.Family (bytesAt outCol plain)
open RecDeploy.OpenedPlain (QueryOpens queryOpens_of_outs)

/-! ## `DecompB` reads the parts' checks only at their own messages -/

/-- **Every part's `chk` at the parts' own messages**: `nS` witnesses, their recorded calls true and free of
collisions, and every part's check at the messages the layout reads off them. -/
def Joint (H : Msg → Msg) (D : Claims.Recursion.Deploy Coin) (L : LayoutB)
    (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (x : Msg) (c : ℕ → Coin) (ws : List Claims.Flock.Wit) : Prop :=
  ws.length = D.nS x ∧ (∀ e ∈ L.allCalls x ws, H e.1 = e.2) ∧ (Claims.findColl (L.allCalls x ws)).isSome = false ∧
    ∀ s < D.nS x, chk s x c (L.msgs x ws)

/-- **`DecompB` is antitone in the parts' checks at their own messages**: it holds at `chk'` wherever it holds at
`chk` and every part's `chk'` at the parts' own messages gives every part's `chk` there. A collision among the
recorded calls is `DecompB`'s own second outcome, so `Joint` may assume none. -/
theorem decompB_of_joint {H : Msg → Msg} {D : Claims.Recursion.Deploy Coin} {IV : Claims.Verifier Coin}
    {L : LayoutB} {shape : Claims.Recursion.Form → Prop} {chk chk' : ℕ → Msg → (ℕ → Coin) → List Msg → Prop}
    (hle : ∀ (x : Msg) (st : Claims.Recursion.Stmt), D.parse x = some st → shape st.form →
      ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit), Joint H D L chk' x c ws →
        ∀ s < D.nS x, chk s x c (L.msgs x ws))
    (hD : DecompB H D IV L shape chk) : DecompB H D IV L shape chk' := by
  intro x st hx hf c ws hl hT hC
  by_cases hcol : (Claims.findColl (L.allCalls x ws)).isSome = true
  · exact Or.inr hcol
  · exact hD x st hx hf c ws hl hT (hle x st hx hf c ws ⟨hl, hT, by simpa using hcol, hC⟩)

section Compose

variable (H : Msg → Msg) (rd : Msg → Option FinalP) {Q : Type} (pieces : Msg → List Q)
  (pholds : Q → Msg → List Msg → List Coin → Prop)
  (hpieces : ∀ (x : Msg) (ms : List Msg) (cs : List Coin), ReadOkR rd x ms cs →
    (∀ b ∈ VerdictP.bases, holdsR H rd b x ms cs) → (∀ q ∈ pieces x, pholds q x ms cs) →
      holdsR H rd .verdict x ms cs)
  (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
  (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ)
  (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)

/-- **The composition with the pieces' share at the parts' own messages**: `decompB_lastSplitP_shares` with `hlowP`
asked only where every part accepts at the parts' own messages (`Joint`), which is all `DecompB` reads. -/
theorem decompB_lastSplitP_sharesJ (ownB : Msg → VerdictR → Option ℕ) (ownQ : Msg → Q → Option ℕ)
    (rel' : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings) (L : LayoutB)
    (IV : Claims.Verifier Coin) (shape : Claims.Recursion.Form → Prop) {Col' T' : Type}
    (DO : Claims.Flock.Deploy Coin Col' T') (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (cmt : Msg → ℕ → Msg → Msg)
    (hR : ∀ (x : Msg) (st : Claims.Recursion.Stmt), (recBound rel' b L IV).parse x = some st → shape st.form →
      ∀ r ∈ (lastSplitP H rd pieces pholds hpieces).checks (Claims.Recursion.xI (recBound rel' b L IV) x), ∀ s,
        ownP ownB ownQ x r = some s → s < (recBound rel' b L IV).nS x)
    (hrest : ∀ (x : Msg) (st : Claims.Recursion.Stmt), (recBound rel' b L IV).parse x = some st → shape st.form →
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
    (hlowP : ∀ (x : Msg) (st : Claims.Recursion.Stmt), (recBound rel' b L IV).parse x = some st → shape st.form →
      ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit),
        Joint H (recBound rel' b L IV) L (PartAccepts H (recBound rel' b L IV) DO L shape shows cmt) x c ws →
        ∀ s < (recBound rel' b L IV).nS x,
          PiecesComputes (bdDeployed rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w) (recBound rel' b L IV)
            pieces pholds ownQ s x c (L.msgs x ws)) :
    DecompB H (recBound rel' b L IV) (bdDeployed rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w).IVStar L
      shape (PartAccepts H (recBound rel' b L IV) DO L shape shows cmt) := by
  have hD := ivstar_of_parts H (recBound rel' b L IV) _ L shape
    (deploySplit H rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w (nextR_coin rd H)
      (lastSplitP H rd pieces pholds hpieces).toLast) _ _
    (ownedBy_deploy H rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w (nextR_coin rd H)
      (lastSplitP H rd pieces pholds hpieces).toLast (ownJ (recBound rel' b L IV) L (ownP ownB ownQ)) _ L shape
      (ownR_lt (lastSplitP H rd pieces pholds hpieces) (ownP ownB ownQ) _ L shape hR))
    (stmtFixes_deploy H rel (nextR rd) colsE wE FlockDeploy.joinR_head rowIn final w (nextR_coin rd H)
      (lastSplitP H rd pieces pholds hpieces).toLast (ownJ (recBound rel' b L IV) L (ownP ownB ownQ)) rel' b L IV
      shape hrest)
  refine decompB_of_joint ?_ hD
  intro x st hx hf c ws hJ s hs
  exact partComputes_of_shares H rd pieces pholds hpieces rel colsE wE rowIn final w ownB ownQ _ L
    (hlowR s x c _ (hJ.2.2.2 s hs)) (hlowP x st hx hf c ws hJ s hs)

end Compose

/-! ## `openPieces_computes` at everything `PartAccepts` names -/

/-- **The `rec-open` pieces' share at a level part, from everything `PartAccepts` names**: `openPieces_computes` with
its two hypotheses as one, `hopen`, which is given what `partAccepts_staged` names at the part (the parsed statement,
its tables and their copies, the agreement `LayoutB.Agrees`, the true and collision-free calls, the shown coins) and
names, for each piece `ownQ` gives the part, the copy at which the piece's `OpenCopy` and `OpenTie` hold. -/
theorem openPieces_computesA {H : Msg → Msg} {D : Claims.Recursion.Deploy Coin} {L : LayoutB}
    {shape : Claims.Recursion.Form → Prop} {untr : TrReader} {b : Brings} {cmt : Msg → ℕ → Msg → Msg} {s : ℕ}
    {x : Msg} {c : ℕ → Coin} {ms : List Msg}
    (h : PartAccepts H D (deployZkRec _root_.Flock.Readable.relation) L shape (showsPart untr b) cmt s x c ms)
    {r : XR} (hr : readX x = some r) {ss : Array VStar.Stmt} (hss : _root_.Flock.RecVStar.statements r.shape = .ok ss)
    {lv LANES HH n : ℕ}
    (hsv : ss[s]? = some (.recOpen lv LANES HH n) ∨ ss[s]? = some (.recOpenStacked lv LANES HH n))
    (Bd : Claims.Recursion.Bound Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace)
    (pieces : Msg → List OpenPiece) (ownQ : Msg → OpenPiece → Option ℕ)
    (sideAt : Msg → List Msg → OpenPiece → OpenSide) (posAt : Msg → List Coin → OpenPiece → ℕ)
    (hopen : ∀ (xO : Msg) (w : Claims.Flock.Wit) (p : _root_.Flock.Readable.Parsed),
      _root_.Flock.Readable.parseBytes readOfO xO = some p → 0 < p.sts.size →
      (∀ sv, ss[s]? = some sv → _root_.Flock.Readable.StmtCopies sv p w) → L.Agrees cmt x s w xO ms →
      (∀ e ∈ L.calls x s w, H e.1 = e.2) →
      (Claims.findColl (L.calls x s w ++ vcallsAt H D xO)).isSome = false → showsTr untr s x xO c →
      Claims.Flock.check (deployZkRec _root_.Flock.Readable.relation).parse xO w = true → StagedPart b.st s x xO →
      ∀ q ∈ pieces (Claims.Recursion.xI D x), ownQ x q = some s →
        ∃ t, ∃ ht : t < p.sts.size, ∃ o g u, o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog) ∧ g < p.sts[t].c.g ∧
          u < p.sts[t].c.upv ∧
          ∀ os last, Bd.final (ms.getD (Bd.DE.V.k (Claims.Recursion.xI D x) - 1) []) = some (os, last) →
            os.length + 1 = Bd.DE.V.k (Claims.Recursion.xI D x) →
            OpenCopy (sideAt (Claims.Recursion.xI D x) (Bd.opened os last) q) q LANES HH
                (_root_.Flock.Readable.copyOf p t ht w o g u) ∧
              OpenTie (sideAt (Claims.Recursion.xI D x) (Bd.opened os last) q) q LANES HH
                (posAt (Claims.Recursion.xI D x) ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) q)
                (_root_.Flock.Readable.copyOf p t ht w o g u)) :
    PiecesComputes Bd D pieces (openHolds sideAt posAt) ownQ s x c ms := by
  obtain ⟨-, xO, w, hck, hT, hc, hs, hh, hp, hm, hrw⟩ := h
  obtain ⟨p, r', sv, hx, ht, hpi, hr', ⟨ss', hss', hsv'⟩, hsz, htab⟩ := stagedReads hs
  have hck' : Claims.Flock.check (_root_.Flock.Readable.parseOf _root_.Flock.Readable.relation readOfO) xO w =
      true := hck
  have hcp := _root_.Flock.Readable.stmtCopies_of_allOuts htab
    (_root_.Flock.Readable.allOuts_of_check hck' hx ht hpi)
  have hA : L.Agrees cmt x s w xO ms := ⟨hh, hp, hm, hrw⟩
  rw [hr] at hr'
  cases hr'
  rw [hss] at hss'
  cases hss'
  have hcps : ∀ sv', ss[s]? = some sv' → _root_.Flock.Readable.StmtCopies sv' p w := by
    intro sv' h'
    rw [hsv'] at h'
    cases h'
    exact hcp
  have hlev : ∀ t (ht : t < p.sts.size) (o g u : ℕ), o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog) →
      g < p.sts[t].c.g → u < p.sts[t].c.upv →
      _root_.Flock.Readable.LevelOuts (_root_.Flock.Readable.copyOf p t ht w o g u) LANES HH := by
    rcases hsv with hsv | hsv <;> rw [hsv] at hsv' <;> cases hsv'
    · exact fun t ht o g u ho hg hu => _root_.Flock.Readable.levelCopies ht ho hg hu hcp
    · exact fun t ht o g u ho hg hu => _root_.Flock.Readable.stackedCopies ht ho hg hu hcp
  intro q hq ho os last hf hl
  obtain ⟨t, ht, o, g, u, hob, hgb, hub, hcq⟩ := hopen xO w p hx hsz hcps hA hT hc hs.1 hck hs.2 q hq ho
  obtain ⟨⟨hLA, hHH, hrow, hsib⟩, hbits, htop⟩ := hcq os last hf hl
  have hout := (hlev t ht o g u hob hgb hub).1
  obtain ⟨hrs, -, -, -, hsl, -⟩ := _root_.Flock.Readable.openIns_reads
    (_root_.Flock.Readable.copyOf p t ht w o g u) LANES HH
  exact queryOpens_of_outs (openOuts_plain hout) hrow.symm hsib (by rw [hrs, hLA]) (by rw [hsl, hHH]) hbits htop

/-! ## The deployed choices -/

/-- Whether a V\* statement is the circuit table's level `ℓ`. -/
def isLevel (ℓ : ℕ) : VStar.Stmt → Bool
  | .recOpen l .. => l == ℓ
  | _ => false

/-- **A piece's owner**: a piece of the inner session's one table (`table = 0`) at level `ℓ` goes to V\*'s statement
`.recOpen ℓ …` (`RecVStar.statements`: the table's levels first, in level order). None at any other table, or where
the statement reads no V\* list. -/
def ownQR (x : Msg) (q : OpenPiece) : Option ℕ :=
  if q.table = 0 then
    (readX x).bind fun r => (_root_.Flock.RecVStar.statements r.shape).toOption.bind fun ss =>
      ss.findIdx? (isLevel q.level)
  else none

/-- **Table `t`'s schedule at the inner statement `x`**, as `repCtx` reads it: `fast100` at the table's `m`. -/
def schAt (x : Msg) (t : ℕ) : Flock.Schedule :=
  match setupOf false x with
  | some v => (fast100 (tabsOf v.2.2)[t]!.m).getD default
  | none => default

/-- The queries of level `ℓ` of the inner schedule at the inner statement `xI`: table 0's level `ℓ`'s `q`. -/
def qAt (xI : Msg) (ℓ : ℕ) : ℕ := ((schAt xI 0).levels[ℓ]!).q

/-- **A piece's copy** at a parsed level statement `p`: rep `r`'s opening `j` is instance `n = r · q + j`
(`RecCoins.checkOpen`'s order), at block `n / g`, VU `n % g`, copy 0 of the part's one table (`rowHidden`). -/
def copyAtR (p : _root_.Flock.Readable.Parsed) (xI : Msg) (q : OpenPiece) : ℕ × ℕ × ℕ :=
  let n := q.rep * qAt xI q.level + q.query
  (n / p.sts[0]!.c.g, n % p.sts[0]!.c.g, 0)

/-- **The level-0 cap a piece opens against**: rep `k = 2 table + rep`'s table's level-0 top's nodes, the nodes of
round 0's rows for the table (`nodeOf`, as `decR` reads its `r₀`; `repCtx`'s `cap0`, which `readRep` holds the proof's
cap to: `FitsRep.cap`). -/
def capR (x : Msg) (ms : List Msg) (q : OpenPiece) : Array ByteArray :=
  match setupOf false x, (rowsOf (ms.headD [])).map (List.map FlockDeploy.nodeOf) with
  | some v, some r₀ => ((repCtx v.2.1 v.2.2 default r₀ [] (2 * q.table + q.rep)).cap0).toArray
  | _, _ => default

/-- **The Ligerito proof a piece opens**: rep `k = 2 table + rep`'s proof as the last message `rd` reads it (its
openings, all `absorbedFrom` keeps of it), with the recursive caps the rep's rounds commit (`topGroups`, which
`readRep` holds the proof's to: `FitsRep.tops`). -/
def ligR (rd : Msg → Option FinalP) (x : Msg) (ms : List Msg) (q : OpenPiece) : Flock.LigeritoProof :=
  match setupOf false x, ms.getLast?.bind rd, rowsOf (ms.headD []), ms.tail.dropLast.mapM rowsOf with
  | some v, some fl, some r₀, some rss =>
    { (fl.ps[2 * q.table + q.rep]!).ligerito with
      recursiveCaps := ((topGroups (repCtx v.2.1 v.2.2 default r₀ [] (2 * q.table + q.rep)).sched
        ((cutLens (repLens v.2.1 v.2.2) rss).getD (2 * q.table + q.rep) [])).map List.toArray).toArray }
  | _, _, _, _ => default

/-- **The side a `rec-open` piece reads, without `readRep`'s choice**: the table's schedule (`schAt`), the level-0
cap (`capR`) and the opened proof (`ligR`). Where `readRep` fits, it keeps exactly these: `FitsRep.cap`,
`FitsRep.tops`, and `absorbedFrom` keeps the last message's openings. -/
def sideAtR (rd : Msg → Option FinalP) (x : Msg) (ms : List Msg) (q : OpenPiece) : OpenSide :=
  ⟨schAt x q.table, capR x ms q, ligR rd x ms q⟩

/-! ## The level's shape, from the verifier's list -/

/-- **The verifier's list at an inner shape with no RecSparse phase**: V\*'s statements at the inner shape's own
`m`, `k_log` and claims (`statementsOf`), then the caps; candidate 3 has none (`candidate3_ok`). -/
theorem statements_dense {i : _root_.Flock.RecVStar.Inner} {ss : Array VStar.Stmt}
    (h : _root_.Flock.RecVStar.statements i = .ok ss) (hsp : i.sparse = none) :
    ∃ A K n, VStar.statementsOf ⟨i.m, i.kLog, i.claims⟩ i.holo = .ok A ∧
      VStar.levelCapStatements ⟨i.m, i.kLog, i.claims⟩ = .ok K ∧ ss = A ++ (K ++ #[.cap 0 n]) := by
  unfold _root_.Flock.RecVStar.statements at h
  dsimp only at h
  split at h
  rotate_left
  · cases h
  split at h
  · cases h
  obtain ⟨ps, -, h⟩ := FlockC3.VStarList.eok_bind h
  obtain ⟨gs, -, h⟩ := FlockC3.VStarList.eok_bind h
  split at h
  · cases h
  split at h
  · cases h
  obtain ⟨K, hK, h⟩ := FlockC3.VStarList.eok_bind h
  obtain ⟨t, -, h⟩ := FlockC3.VStarList.eok_bind h
  obtain ⟨A, hA, h⟩ := FlockC3.VStarList.eok_bind h
  obtain ⟨C, hC, h⟩ := FlockC3.VStarList.eok_bind h
  obtain ⟨K0, hK0, h⟩ := FlockC3.VStarList.eok_bind h
  obtain ⟨n, rfl⟩ := FlockC3.VStarList.capStatement_ok hK0
  obtain ⟨u, -, h⟩ := FlockC3.VStarList.eok_bind h
  obtain ⟨u', -, h⟩ := FlockC3.VStarList.eok_bind h
  obtain ⟨u'', -, h⟩ := FlockC3.VStarList.eok_bind h
  cases h
  obtain rfl := (FlockC3.VStarList.candidate3_ok hC).1 hsp
  exact ⟨A, K, n, hA, hK, by simp⟩

/-- **A level statement of a public dense inner shape is its level's**: where the verifier's list at a public inner
shape with no RecSparse phase has `.recOpen lv LANES HH n` at index `s`, it is V\*'s level `s` (`statements_eq`, the
rest being inner parts and caps), at `fast100`'s schedule at the shape's `m`, over its two reps' queries. -/
theorem recOpen_dense {i : _root_.Flock.RecVStar.Inner} {ss : Array VStar.Stmt}
    (h : _root_.Flock.RecVStar.statements i = .ok ss) (hsp : i.sparse = none) (hh : i.holo = false)
    {s lv LANES HH n : ℕ} (hsv : ss[s]? = some (.recOpen lv LANES HH n)) :
    ∃ sch, fast100 i.m = some sch ∧ lv = s ∧ LANES = 2 ^ (sch.levels[s]!).lanes ∧
      HH = (sch.levels[s]!).d - (sch.levels[s]!).cap ∧ n = 2 * (sch.levels[s]!).q := by
  obtain ⟨A, K, n0, hA, hK, rfl⟩ := statements_dense h hsp
  rw [hh] at hA
  obtain ⟨sch, sp, ps, hsch, -, -, rfl⟩ := FlockVBridge.Family.statementsOf_eq hA
  refine ⟨sch, FlockVBridge.Family.sched_fast100 hsch, ?_⟩
  by_cases h1 : s < ((List.range sch.levels.size).map fun l => VStar.Stmt.recOpen l (2 ^ (sch.levels[l]!).lanes)
      ((sch.levels[l]!).d - (sch.levels[l]!).cap) (2 * (sch.levels[l]!).q)).toArray.size
  · rw [Array.append_assoc, Array.getElem?_append_left h1] at hsv
    have h1' : s < sch.levels.size := by simpa using h1
    simp only [List.getElem?_toArray, List.getElem?_map, List.getElem?_range h1', Option.map_some,
      Option.some.injEq, VStar.Stmt.recOpen.injEq] at hsv
    obtain ⟨rfl, rfl, rfl, rfl⟩ := hsv
    exact ⟨rfl, rfl, rfl, rfl⟩
  · exfalso
    rw [Array.append_assoc, Array.getElem?_append_right (Nat.le_of_not_lt h1)] at hsv
    rcases Array.mem_append.1 (Array.mem_of_getElem? hsv) with hm | hm
    · simp at hm
    · obtain ⟨l, N, hlN⟩ := FlockC3.VStarList.caps_cap hK n0 _ hm
      cases hlN

/-- **The inner table's `m` is the inner shape's**, at a statement the reader reads: Flock's plain verifier sets the
inner statement up as one table (`innerSetup_files`), whose `m` the setup's checks hold to the shape's
(`checkShape_files`). -/
theorem inner_m {x : Msg} {st : Claims.Recursion.Stmt} (hst : parse x = some st) {r : XR} (hr : readX x = some r) :
    ∃ s spec t, setupOf false (xIOf x) = some (s, spec, #[t]) ∧ t.m = r.shape.m := by
  obtain ⟨r', hr', ha, rfl⟩ := parse_read hst
  rw [hr] at hr'
  cases hr'
  obtain ⟨f, spec', st', s, spec, t, hfs, hstm, hset, hm, -, -, -, -, -⟩ := innerSetup_files hr ha
  unfold RecScope.accepts at ha
  simp only [Bool.and_eq_true, RecScope.isOk_iff] at ha
  obtain ⟨⟨⟨⟨ss, -⟩, -⟩, ⟨u, hsh⟩⟩, -⟩ := ha
  have hsh : RecSetup.checkShape (some r.shape) (filesOf r) = .ok u := hsh
  rw [hfs] at hsh
  obtain ⟨spec'', st'', hstm', e1, -, -, -, -⟩ := checkShape_files hsh
  rw [hstm] at hstm'
  cases hstm'
  have hx : xIOf x = r.w.inner := by unfold xIOf; rw [hst]; rfl
  exact ⟨s, spec, t, by rw [hx]; exact hset, by rw [hm, ← e1]⟩

/-- **`OpenCopy`'s level shape at a dense statement**: where the reader reads `x` at the dense form and its V\* list
has `.recOpen lv LANES HH n` at `s`, the statement is level `s`'s, `LANES`, `HH` are table 0's schedule's (`schAt`)
level `s`'s lanes and depth, and `n` is its two reps' queries (`recOpen_dense`, `inner_m`). -/
theorem levelMatch {x : Msg} {st : Claims.Recursion.Stmt} (hst : parse x = some st)
    (hd : Claims.Recursion.Form.Dense st.form) {r : XR} (hr : readX x = some r) {ss : Array VStar.Stmt}
    (hss : _root_.Flock.RecVStar.statements r.shape = .ok ss) {s lv LANES HH n : ℕ}
    (hsv : ss[s]? = some (.recOpen lv LANES HH n)) :
    lv = s ∧ LANES = 2 ^ ((schAt (xIOf x) 0).levels[s]!).lanes ∧
      HH = ((schAt (xIOf x) 0).levels[s]!).d - ((schAt (xIOf x) 0).levels[s]!).cap ∧
      n = 2 * ((schAt (xIOf x) 0).levels[s]!).q := by
  obtain ⟨r', hr', -, rfl⟩ := parse_read hst
  rw [hr] at hr'
  cases hr'
  have hh : r.shape.holo = false := hd.1
  have hsp : r.shape.sparse = none := by
    have h2 : (r.shape.sparse.map fun sp => (sp.nu, sp.ell, sp.sLog, sp.tLog)) = none := hd.2
    cases hs : r.shape.sparse with
    | none => rfl
    | some _ => simp [hs] at h2
  obtain ⟨sch, hsch, rfl, hL, hH, hn⟩ := recOpen_dense hss hsp hh hsv
  obtain ⟨s', spec, t, hset, htm⟩ := inner_m hst hr
  have e : schAt (xIOf x) 0 = sch := by
    unfold schAt
    rw [hset]
    have htabs : tabsOf #[t] = #[_root_.Flock.Setup.ofCircuit t] := by simp [tabsOf]
    have h0 : (tabsOf #[t])[0]!.m = t.m := by
      rw [htabs]
      rfl
    dsimp only
    rw [h0, htm, hsch]
    rfl
  rw [e]
  exact ⟨rfl, hL, hH, hn⟩

/-- **`ownQR`'s owner of a piece is its level's statement**: where `ownQR` gives piece `q` part `s`, the piece is of
table 0 and part `s`'s statement is a `.recOpen` at the piece's level. -/
theorem ownQR_some {x : Msg} {q : OpenPiece} {s : ℕ} (ho : ownQR x q = some s) :
    q.table = 0 ∧ ∃ r ss, readX x = some r ∧ _root_.Flock.RecVStar.statements r.shape = .ok ss ∧
      ∃ LANES HH n, ss[s]? = some (.recOpen q.level LANES HH n) := by
  unfold ownQR at ho
  split at ho
  · rename_i h0
    obtain ⟨r, hr, ho⟩ := Option.bind_eq_some_iff.1 ho
    obtain ⟨ss, hss, hfi⟩ := Option.bind_eq_some_iff.1 ho
    have hss' : _root_.Flock.RecVStar.statements r.shape = .ok ss := by
      cases h : _root_.Flock.RecVStar.statements r.shape with
      | error e => rw [h] at hss; cases hss
      | ok v => rw [h] at hss; cases hss; rfl
    obtain ⟨hsz, hlev, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hfi
    refine ⟨h0, r, ss, hr, hss', ?_⟩
    rw [Array.getElem?_eq_getElem hsz]
    revert hlev
    cases ss[s] with
    | recOpen l L' H' n' =>
      intro h
      have hl : l = q.level := by simpa [isLevel] using h
      exact ⟨L', H', n', by rw [hl]⟩
    | _ => intro h; simp [isLevel] at h
  · cases ho

/-! ## The two facts at the deployed layout -/

section Deployed

variable (H : Msg → Msg) (D : Claims.Recursion.Deploy Coin) (shape : Claims.Recursion.Form → Prop)
  (untr : TrReader) (b : Brings) (cmt : Msg → ℕ → Msg → Msg) (L : LayoutB)
  (Bd : Claims.Recursion.Bound Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace)
  (side : Msg → List Msg → OpenPiece → OpenSide)

/-- **Part `s` accepts at its staged statement `xO` and witness `w`**, at messages `ms`: what `partAccepts_staged`
names at a level part `sv`, with the outer check and the staging, that `openPieces_computesA`'s `hopen` is given. -/
def StagedAt (s : ℕ) (x : Msg) (c : ℕ → Coin) (ms : List Msg) (sv : VStar.Stmt) (xO : Msg) (w : Claims.Flock.Wit)
    (p : _root_.Flock.Readable.Parsed) : Prop :=
  _root_.Flock.Readable.parseBytes readOfO xO = some p ∧ 0 < p.sts.size ∧ _root_.Flock.Readable.StmtCopies sv p w ∧
    L.Agrees cmt x s w xO ms ∧ (∀ e ∈ L.calls x s w, H e.1 = e.2) ∧
    (Claims.findColl (L.calls x s w ++ vcallsAt H D xO)).isSome = false ∧ showsTr untr s x xO c ∧
    Claims.Flock.check (deployZkRec _root_.Flock.Readable.relation).parse xO w = true ∧ StagedPart b.st s x xO

/-- **The context of a piece's facts at the deployed layout**: at a statement the record reads at the shape, at the
parts' own messages with every part accepting (`Joint`), a level part `s` (`.recOpen lv LANES HH n`), its staged
statement and witness, and a piece `ownQR` gives it. `Fact` is asked there, at the piece's copy `copyAtR`. -/
abbrev AtPieces (pieces : Msg → List OpenPiece)
    (Fact : Msg → (ℕ → Coin) → List Claims.Flock.Wit → ℕ → ℕ → ℕ → (xO : Msg) → Claims.Flock.Wit →
      (p : _root_.Flock.Readable.Parsed) → 0 < p.sts.size → OpenPiece → Prop) : Prop :=
  ∀ (x : Msg) (st : Claims.Recursion.Stmt), D.parse x = some st → shape st.form →
    ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit),
    Joint H D L (PartAccepts H D (deployZkRec _root_.Flock.Readable.relation) L shape (showsPart untr b) cmt) x c ws →
    ∀ (r : XR) (ss : Array VStar.Stmt) (s lv LANES HH n : ℕ), readX x = some r →
      _root_.Flock.RecVStar.statements r.shape = .ok ss → ss[s]? = some (.recOpen lv LANES HH n) →
    ∀ (xO : Msg) (w : Claims.Flock.Wit) (p : _root_.Flock.Readable.Parsed),
      StagedAt H D untr b cmt L s x c (L.msgs x ws) (.recOpen lv LANES HH n) xO w p →
    ∀ q ∈ pieces (Claims.Recursion.xI D x), ownQR x q = some s → ∀ ht : 0 < p.sts.size,
      Fact x c ws LANES HH s xO w p ht q

/-- The copy `copyAtR` of piece `q` at the parsed statement `p`, at the witness `w`. -/
abbrev copyR (xI : Msg) (w : Claims.Flock.Wit) (p : _root_.Flock.Readable.Parsed) (ht : 0 < p.sts.size)
    (q : OpenPiece) : ℕ → Bool :=
  _root_.Flock.Readable.copyOf p 0 ht w (copyAtR p xI q).1 (copyAtR p xI q).2.1 (copyAtR p xI q).2.2

/-- What a piece's fact reads at every reading of the last message: the opened messages, at `Bd`. -/
abbrev AtOpened (x : Msg) (ms : List Msg) (G : List Msg → Prop) : Prop :=
  ∀ os last, Bd.final (ms.getD (Bd.DE.V.k (Claims.Recursion.xI D x) - 1) []) = some (os, last) →
    os.length + 1 = Bd.DE.V.k (Claims.Recursion.xI D x) → G (Bd.opened os last)

/-- **The copy is one of the statement's** (`CopyFitsR`): its block, VU and copy are in range. -/
def CopyFitsR (pieces : Msg → List OpenPiece) : Prop :=
  AtPieces H D shape untr b cmt L pieces fun x _ _ _ _ _ _ _ p ht q =>
    (copyAtR p (Claims.Recursion.xI D x) q).1 < 2 ^ (p.sts[0].m - p.sts[0].c.kLog) ∧
      (copyAtR p (Claims.Recursion.xI D x) q).2.1 < p.sts[0].c.g ∧
      (copyAtR p (Claims.Recursion.xI D x) q).2.2 < p.sts[0].c.upv

/-- **The copy reads the piece's opened row and siblings** (`CopyOpensR`, `OpenCopy`'s last two conjuncts), at
`side` of every reading of the last message. -/
def CopyOpensR (pieces : Msg → List OpenPiece) : Prop :=
  AtPieces H D shape untr b cmt L pieces fun x _ ws LANES HH _ _ w p ht q =>
    AtOpened D Bd x (L.msgs x ws) fun ms' =>
      _root_.Flock.Readable.openRow (copyR (Claims.Recursion.xI D x) w p ht q) LANES =
          (openAt (side (Claims.Recursion.xI D x) ms' q).lig
            ((side (Claims.Recursion.xI D x) ms' q).sch.levels.size - 1) q.level).1[q.query]! ∧
        _root_.Flock.sibsOf (openAt (side (Claims.Recursion.xI D x) ms' q).lig
            ((side (Claims.Recursion.xI D x) ms' q).sch.levels.size - 1) q.level).2.1
            (((side (Claims.Recursion.xI D x) ms' q).sch.levels[q.level]!).d -
              ((side (Claims.Recursion.xI D x) ms' q).sch.levels[q.level]!).cap) q.query =
          _root_.Flock.Readable.openSibs (copyR (Claims.Recursion.xI D x) w p ht q) LANES HH

/-- **The copy's position bits are the coins'** (`CopyBitsR`, `OpenTie`'s first conjunct). -/
def CopyBitsR (pieces : Msg → List OpenPiece) (posAt : Msg → List Coin → OpenPiece → ℕ) : Prop :=
  AtPieces H D shape untr b cmt L pieces fun x c ws LANES HH _ _ w p ht q =>
    AtOpened D Bd x (L.msgs x ws) fun ms' =>
      ∀ h < ((side (Claims.Recursion.xI D x) ms' q).sch.levels[q.level]!).d -
          ((side (Claims.Recursion.xI D x) ms' q).sch.levels[q.level]!).cap,
        (_root_.Flock.Readable.openIdx (copyR (Claims.Recursion.xI D x) w p ht q) LANES HH).testBit h =
          (posAt (Claims.Recursion.xI D x) ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) q).testBit h

/-- **The copy's top is the cap's entry at the position** (`CopyTopR`, `OpenTie`'s second conjunct). -/
def CopyTopR (pieces : Msg → List OpenPiece) (posAt : Msg → List Coin → OpenPiece → ℕ) : Prop :=
  AtPieces H D shape untr b cmt L pieces fun x c ws LANES HH _ _ w p ht q =>
    AtOpened D Bd x (L.msgs x ws) fun ms' =>
      bytesAt (copyR (Claims.Recursion.xI D x) w p ht q) (outCol LANES HH) 64 =
        (FlockSoundness.Refine.capAt (side (Claims.Recursion.xI D x) ms' q).cap
            (side (Claims.Recursion.xI D x) ms' q).lig q.level).getD
          ((posAt (Claims.Recursion.xI D x) ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) q) >>>
            (((side (Claims.Recursion.xI D x) ms' q).sch.levels[q.level]!).d -
              ((side (Claims.Recursion.xI D x) ms' q).sch.levels[q.level]!).cap)) ByteArray.empty

/-- **The layout fact at the deployed layout** (U13's first half): the copy `copyAtR` is in range, and at every
reading of the last message it is the level's (`LANES`, `HH`) and reads the piece's opened row and siblings at
`side` (`OpenCopy`). -/
def OpenCopyR (pieces : Msg → List OpenPiece) : Prop :=
  AtPieces H D shape untr b cmt L pieces fun x c ws LANES HH _ _ w p ht q =>
    (copyAtR p (Claims.Recursion.xI D x) q).1 < 2 ^ (p.sts[0].m - p.sts[0].c.kLog) ∧
      (copyAtR p (Claims.Recursion.xI D x) q).2.1 < p.sts[0].c.g ∧
      (copyAtR p (Claims.Recursion.xI D x) q).2.2 < p.sts[0].c.upv ∧
      AtOpened D Bd x (L.msgs x ws) fun ms' =>
        OpenCopy (side (Claims.Recursion.xI D x) ms' q) q LANES HH (copyR (Claims.Recursion.xI D x) w p ht q)

/-- **The registration ties at the deployed layout** (U13's second half): at the copy `copyAtR`, the position bits it
reads are `posAt`'s and its climb's top is the cap's entry there, at `side` (`OpenTie`). -/
def OpenTieR (pieces : Msg → List OpenPiece) (posAt : Msg → List Coin → OpenPiece → ℕ) : Prop :=
  AtPieces H D shape untr b cmt L pieces fun x c ws LANES HH _ _ w p ht q =>
    AtOpened D Bd x (L.msgs x ws) fun ms' =>
      OpenTie (side (Claims.Recursion.xI D x) ms' q) q LANES HH
        (posAt (Claims.Recursion.xI D x) ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) q)
        (copyR (Claims.Recursion.xI D x) w p ht q)

/-- **The side reads the table's schedule** (`SideSchR`) at every reading of the last message: `sideAtR`'s always
(`sideSchR_sideAtR`), `RecDeploy.Rows.sideAt`'s wherever its rep's proof fits. -/
def SideSchR (pieces : Msg → List OpenPiece) : Prop :=
  AtPieces H D shape untr b cmt L pieces fun x _ ws _ _ _ _ _ _ _ q =>
    AtOpened D Bd x (L.msgs x ws) fun ms' =>
      (side (Claims.Recursion.xI D x) ms' q).sch = schAt (Claims.Recursion.xI D x) q.table

theorem sideSchR_sideAtR (rd : Msg → Option FinalP) (pieces : Msg → List OpenPiece) :
    SideSchR H D shape untr b cmt L Bd (sideAtR rd) pieces := by
  unfold SideSchR AtPieces AtOpened
  intros
  intros
  exact rfl

/-- **`OpenTieR` is its two halves.** -/
theorem openTieR_of (pieces : Msg → List OpenPiece) (posAt : Msg → List Coin → OpenPiece → ℕ)
    (hB : CopyBitsR H D shape untr b cmt L Bd side pieces posAt) (hT : CopyTopR H D shape untr b cmt L Bd side pieces posAt) :
    OpenTieR H D shape untr b cmt L Bd side pieces posAt :=
  fun x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO w p hst q hq ho ht os last hfin hl =>
    And.intro (hB x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO w p hst q hq ho ht os last hfin hl)
      (hT x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO w p hst q hq ho ht os last hfin hl)

/-- **The level a piece's side reads is its statement's**: at a dense statement the deployed reader reads, where
`ownQR` gives piece `q` the level part `s` (`.recOpen lv LANES HH n`), the side's schedule's level `q.level` has
`LANES` lanes and `HH` siblings below its cap (`levelMatch`), wherever the side reads the table's schedule. -/
theorem sideLevel (hP : ∀ x, D.parse x = parse x) (hS : ∀ f, shape f → Claims.Recursion.Form.Dense f)
    {x : Msg} {st : Claims.Recursion.Stmt} (hx : D.parse x = some st) (hf : shape st.form) {r : XR}
    {ss : Array VStar.Stmt} {s lv LANES HH n : ℕ} (hr : readX x = some r)
    (hss : _root_.Flock.RecVStar.statements r.shape = .ok ss) (hsv : ss[s]? = some (.recOpen lv LANES HH n))
    {q : OpenPiece} (ho : ownQR x q = some s) {ms : List Msg}
    (hsch : (side (Claims.Recursion.xI D x) ms q).sch = schAt (Claims.Recursion.xI D x) q.table) :
    LANES = 2 ^ ((side (Claims.Recursion.xI D x) ms q).sch.levels[q.level]!).lanes ∧
      HH = ((side (Claims.Recursion.xI D x) ms q).sch.levels[q.level]!).d -
        ((side (Claims.Recursion.xI D x) ms q).sch.levels[q.level]!).cap := by
  have hx' : parse x = some st := (hP x).symm.trans hx
  have hxI : Claims.Recursion.xI D x = xIOf x := by
    unfold Claims.Recursion.xI xIOf
    rw [hP x]
  obtain ⟨h0, r', ss', hr', hss', L', H', n', hsv'⟩ := ownQR_some ho
  rw [hr] at hr'
  cases hr'
  rw [hss] at hss'
  cases hss'
  rw [hsv] at hsv'
  cases hsv'
  obtain ⟨hls, hL, hH, -⟩ := levelMatch hx' (hS _ hf) hr hss hsv
  rw [hsch, hxI, h0, hls]
  exact ⟨hL, hH⟩

/-- **`OpenCopyR` from the copy's range and its opened row and siblings**, at the deployed reader's statements at
the dense form: the level's shape is the statement's (`sideLevel`). -/
theorem openCopyR_of (pieces : Msg → List OpenPiece) (hP : ∀ x, D.parse x = parse x)
    (hS : ∀ f, shape f → Claims.Recursion.Form.Dense f)
    (hSch : SideSchR H D shape untr b cmt L Bd side pieces) (hF : CopyFitsR H D shape untr b cmt L pieces)
    (hO : CopyOpensR H D shape untr b cmt L Bd side pieces) :
    OpenCopyR H D shape untr b cmt L Bd side pieces := by
  intro x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO w p hst q hq ho ht
  obtain ⟨hob, hgb, hub⟩ := hF x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO w p hst q hq ho ht
  refine ⟨hob, hgb, hub, fun os last hfin hl => ?_⟩
  obtain ⟨hrow, hsib⟩ := hO x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO w p hst q hq ho ht os last hfin hl
  obtain ⟨hL, hH⟩ := sideLevel D shape side hP hS hx hf hr hss hsv ho
    (hSch x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO w p hst q hq ho ht os last hfin hl)
  unfold OpenCopy
  exact ⟨hL, hH, hrow, hsib⟩

/-- **The pieces are in range** (`PiecesInRangeR`): each is of rep 0 or 1 and of one of its level's queries, at its
table's schedule. `RecDeploy.Rows.rowDPieces`'s are (`mem_rowDPiecesOf`, where `Rows.schAt v.2.2` is `schAt`). -/
def PiecesInRangeR (pieces : Msg → List OpenPiece) : Prop :=
  ∀ xI, ∀ q ∈ pieces xI, q.rep < 2 ∧ q.query < ((schAt xI q.table).levels[q.level]!).q

theorem piecesInRangeR_rowDPieces (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) :
    PiecesInRangeR (RecDeploy.Rows.rowDPieces key) := by
  intro xI q hq
  unfold RecDeploy.Rows.rowDPieces at hq
  unfold schAt
  cases h : setupOf false xI with
  | none => rw [h] at hq; cases hq
  | some v =>
    rw [h] at hq
    obtain ⟨-, hr, -, hj, -⟩ := RecDeploy.Rows.mem_rowDPiecesOf.1 hq
    exact ⟨hr, hj⟩

/-- **The staged level circuit holds its statement's instances** (`StagedFitsR`): at a level part `.recOpen lv LANES
HH n` and its staged statement, the parsed table has a VU and a copy, and its blocks' VUs hold all `n` instances. -/
def StagedFitsR : Prop :=
  ∀ (x : Msg) (st : Claims.Recursion.Stmt), D.parse x = some st → shape st.form →
    ∀ (c : ℕ → Coin) (ms : List Msg) (r : XR) (ss : Array VStar.Stmt) (s lv LANES HH n : ℕ), readX x = some r →
      _root_.Flock.RecVStar.statements r.shape = .ok ss → ss[s]? = some (.recOpen lv LANES HH n) →
    ∀ (xO : Msg) (w : Claims.Flock.Wit) (p : _root_.Flock.Readable.Parsed),
      StagedAt H D untr b cmt L s x c ms (.recOpen lv LANES HH n) xO w p → ∀ ht : 0 < p.sts.size,
      0 < p.sts[0].c.g ∧ 0 < p.sts[0].c.upv ∧ n ≤ p.sts[0].c.g * 2 ^ (p.sts[0].m - p.sts[0].c.kLog)

/-- **A staged level circuit holds its statement's instances**: the setup's check makes the table's population the
statement's `n` (`staged_scope`) and its instances at least that (`oneFull_staged`); V\*'s parse puts one unit in a VU
(`checkParsed_ok`); and the setup has a VU and `2 ^ nbl` blocks of `g` VUs for its instances (`table_rows`), at
`m = k_log + nbl` (`Layout.m`). -/
theorem stagedFitsR : StagedFitsR H D shape untr b cmt L := by
  intro x st hx hf c ms r ss s lv LANES HH n hr hss hsv xO w p hst ht
  obtain ⟨hxO, -, -, -, -, -, -, hck, hS⟩ := hst
  have hS' := hS
  obtain ⟨-, i, spec, sts, -, hsu, -⟩ := hS'
  have hp : p = { i.base.read with sts } := Option.some.inj (hxO.symm.trans (parse_of_setupO hsu))
  have h0 : 0 < sts.size := by subst hp; exact ht
  have e0 : p.sts[0] = sts[0] := by subst hp; rfl
  have hle : _root_.Flock.RecVStar.population p.sts[0] ≤ p.sts[0].n := by
    rw [e0]
    exact (oneFull_staged hS hsu).2 0 h0
  obtain ⟨hscope, r', ss', sv', hr', hss', hsv', hn⟩ := staged_scope hS hxO ht
  rw [hr] at hr'
  cases hr'
  rw [hss] at hss'
  cases hss'
  rw [hsv] at hsv'
  cases hsv'
  have hn' : _root_.Flock.RecVStar.population p.sts[0] = n := hn
  obtain ⟨p', _, _, hx', htyp, hpi, -, -, -, htab⟩ := stagedTables hS
  rw [hxO] at hx'
  cases hx'
  obtain ⟨_, -, hupv, -, -, -⟩ := FlockSoundness.Discharge.Recursion.checkParsed_ok (htab 0 ht).2.2
  obtain ⟨-, hts⟩ := tablesOf_of_parse hxO
  obtain ⟨_, hset0⟩ := _root_.Flock.Readable.setup_of_tablesOf hts
  unfold _root_.Flock.Readable.Read.setup at hset0
  rw [hpi] at hset0
  obtain ⟨-, hall⟩ := FlockSoundness.Discharge.HiddenExec.setupTables_setupOne hset0
  obtain ⟨d, hd, -⟩ := hall 0 ht
  have htt : (FlockSoundness.Discharge.Exec.tagsAt p.tags p.I.count 0).typed = false :=
    (FlockSoundness.Discharge.Exec.tagsAt_typed _ _ _).trans htyp
  have hv1 := FlockSoundness.Discharge.Layout.scopeOk_ports
    ((FlockSoundness.Discharge.FailClosed.scopeOk_eq _).symm.trans hscope)
  obtain ⟨Lt, regs, hz⟩ := _root_.Flock.Readable.layout_of_check hck hxO htyp hpi ht
  obtain ⟨-, hg, hnbl, -⟩ := table_rows htt hd (by omega) hv1 Lt regs hz
  have hm : p.sts[0].m - p.sts[0].c.kLog = p.sts[0].nbl := by
    have := Lt.m
    omega
  refine ⟨hg, by omega, ?_⟩
  rw [hm]
  have hN : p.sts[0].n ≤ p.sts[0].c.g * 2 ^ p.sts[0].nbl := by
    rcases Nat.eq_zero_or_pos p.sts[0].n with h | h
    · rw [h]
      exact Nat.zero_le _
    · have hd' := div_lt_blocks hg (show p.sts[0].n - 1 < p.sts[0].n by omega) hnbl
      have hdm := Nat.div_add_mod (p.sts[0].n - 1) p.sts[0].c.g
      have hml := Nat.mod_lt (p.sts[0].n - 1) hg
      have hmul := Nat.mul_le_mul_left p.sts[0].c.g (Nat.succ_le_of_lt hd')
      rw [Nat.mul_succ] at hmul
      omega
  omega

/-- **`CopyFitsR` from the pieces' range and the staged circuit's capacity**: instance `rep · q + query` of a level
part is below its statement's `n = 2 q` (`levelMatch`), which the staged table holds. -/
theorem copyFitsR_of (pieces : Msg → List OpenPiece) (hP : ∀ x, D.parse x = parse x)
    (hS : ∀ f, shape f → Claims.Recursion.Form.Dense f) (hR : PiecesInRangeR pieces)
    (hC : StagedFitsR H D shape untr b cmt L) : CopyFitsR H D shape untr b cmt L pieces := by
  intro x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO w p hst q hq ho ht
  obtain ⟨hg, hu, hcap⟩ := hC x st hx hf c (L.msgs x ws) r ss s lv LANES HH n hr hss hsv xO w p hst ht
  obtain ⟨hrep, hqry⟩ := hR _ q hq
  have hx' : parse x = some st := (hP x).symm.trans hx
  have hxI : Claims.Recursion.xI D x = xIOf x := by
    unfold Claims.Recursion.xI xIOf
    rw [hP x]
  obtain ⟨h0, r', ss', hr', hss', L', H', n', hsv'⟩ := ownQR_some ho
  rw [hr] at hr'
  cases hr'
  rw [hss] at hss'
  cases hss'
  rw [hsv] at hsv'
  cases hsv'
  obtain ⟨hls, -, -, hn⟩ := levelMatch hx' (hS _ hf) hr hss hsv
  subst hls
  rw [h0, hxI] at hqry
  simp only [copyAtR, qAt]
  rw [getElem!_pos p.sts 0 ht, hxI]
  generalize ((schAt (xIOf x) 0).levels[q.level]!).q = Q at hqry hn ⊢
  have h1 : q.rep * Q ≤ 1 * Q := Nat.mul_le_mul_right _ (by omega)
  have hidx : q.rep * Q + q.query < p.sts[0].c.g * 2 ^ (p.sts[0].m - p.sts[0].c.kLog) := by omega
  refine ⟨?_, Nat.mod_lt _ hg, hu⟩
  rw [Nat.div_lt_iff_lt_mul hg]
  exact lt_of_lt_of_eq hidx (Nat.mul_comm _ _)

/-- **`CopyFitsR` at row D's pieces**, outright: `copyFitsR_of` at `piecesInRangeR_rowDPieces` and `stagedFitsR`. -/
theorem copyFitsR_rowD (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) (hP : ∀ x, D.parse x = parse x)
    (hS : ∀ f, shape f → Claims.Recursion.Form.Dense f) :
    CopyFitsR H D shape untr b cmt L (RecDeploy.Rows.rowDPieces key) :=
  copyFitsR_of H D shape untr b cmt L _ hP hS (piecesInRangeR_rowDPieces key) (stagedFitsR H D shape untr b cmt L)

/-- The column of a rec-open copy's position bit `h`, where `openIdx` reads it. -/
def idxCol (LANES HH h : ℕ) : ℕ := 128 * LANES + 1024 * max 1 ((HH + 1) / 2) + 1024 + 256 * LANES + h

theorem openIdx_testBit {v : ℕ → Bool} {LANES HH h : ℕ} (hh : h < HH) :
    (_root_.Flock.Readable.openIdx v LANES HH).testBit h = v (idxCol LANES HH h) := by
  simp [_root_.Flock.Readable.openIdx, idxCol, FlockVBridge.Family.testBit_natOfBits, hh]

/-- **Check 5's `dirs` tie at a copy**: the copy's position columns hold `pos`'s `dirs` row (`RecCoins.dirsRow`, which
`RecCoins.checkOpen` registers at each opening), 16 bits a word. -/
def DirsTie (v : ℕ → Bool) (LANES HH pos : ℕ) : Prop :=
  ∀ h < HH, v (idxCol LANES HH h) = ((_root_.Flock.RecCoins.dirsRow pos HH)[h / 16]!).testBit (h % 16)

/-- **The `dirs` tie gives the position's bits** (`bits_of_dirsRow`), at `HH ≤ 32`. -/
theorem bits_of_dirsTie {v : ℕ → Bool} {LANES HH pos : ℕ} (hH : HH ≤ 32) (ht : DirsTie v LANES HH pos) :
    ∀ h < HH, (_root_.Flock.Readable.openIdx v LANES HH).testBit h = pos.testBit h :=
  bits_of_dirsRow hH fun k hk => (openIdx_testBit hk).trans (ht k hk)

/-- **Check 5's `dirs` tie at the deployed layout** (`CopyDirsR`): the level's `HH` is at most 32 and the piece's copy
holds the `dirs` row of the position `posAt` reads off the coins. -/
def CopyDirsR (pieces : Msg → List OpenPiece) (posAt : Msg → List Coin → OpenPiece → ℕ) : Prop :=
  AtPieces H D shape untr b cmt L pieces fun x c _ LANES HH _ _ w p ht q =>
    HH ≤ 32 ∧ DirsTie (copyR (Claims.Recursion.xI D x) w p ht q) LANES HH
      (posAt (Claims.Recursion.xI D x) ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) q)

/-- **A level part's `HH` is at most 32**: its staged table's unit is `rec-open/l<LANES>-h<HH>` (`stagedTables`),
which the parser's unit check (`ParseFacts.recNet`) accepts only at a valid `LANES` and `HH` (`RecOpen.check_ok`). -/
theorem hh_le_staged {st : Stage} {x : Msg} {r : XR} {ss : Array VStar.Stmt} {s lv LANES HH n : ℕ}
    (hr : readX x = some r) (hss : _root_.Flock.RecVStar.statements r.shape = .ok ss)
    (hsv : ss[s]? = some (.recOpen lv LANES HH n)) {xO : Msg} {p : _root_.Flock.Readable.Parsed}
    (hxO : _root_.Flock.Readable.parseBytes readOfO xO = some p) (hS : StagedPart st s x xO) (ht : 0 < p.sts.size) :
    HH ≤ 32 := by
  obtain ⟨p', r', sv', hx', htyp, hpi, hr', ⟨ss', hss', hsv'⟩, -, htab⟩ := stagedTables hS
  rw [hxO] at hx'
  cases hx'
  rw [hr] at hr'
  cases hr'
  rw [hss] at hss'
  cases hss'
  rw [hsv] at hsv'
  cases hsv'
  obtain ⟨_, P⟩ := _root_.Flock.Readable.parseFacts_of_parse hxO htyp hpi ht
  have hn : p.sts[0].c.unit.name = _root_.Flock.RecOpen.unitName LANES HH := (htab 0 ht).1
  obtain ⟨hv, -⟩ := _root_.Flock.RecOpen.check_ok P.recNet (FlockVBridge.params_of_check P.recNet hn)
  simp only [_root_.Flock.RecOpen.valid, Bool.and_eq_true, decide_eq_true_eq] at hv
  exact hv.2

/-- **Check 5's `dirs` tie at the copies** (`CopyDirsTieR`): `CopyDirsR` without `HH ≤ 32`, which the staged
table's unit check gives (`hh_le_staged`). -/
def CopyDirsTieR (pieces : Msg → List OpenPiece) (posAt : Msg → List Coin → OpenPiece → ℕ) : Prop :=
  AtPieces H D shape untr b cmt L pieces fun x c _ LANES HH _ _ w p ht q =>
    DirsTie (copyR (Claims.Recursion.xI D x) w p ht q) LANES HH
      (posAt (Claims.Recursion.xI D x) ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) q)

theorem copyDirsR_of (pieces : Msg → List OpenPiece) (posAt : Msg → List Coin → OpenPiece → ℕ)
    (hD : CopyDirsTieR H D shape untr b cmt L Bd pieces posAt) : CopyDirsR H D shape untr b cmt L Bd pieces posAt := by
  intro x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO w p hst q hq ho ht
  exact ⟨hh_le_staged hr hss hsv hst.1 hst.2.2.2.2.2.2.2.2 ht,
    hD x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO w p hst q hq ho ht⟩

/-- **`CopyBitsR` from check 5's `dirs` tie**, at the deployed reader's statements at the dense form: the side's
`d − cap` is the statement's `HH` (`sideLevel`), and the tie spells the position's low `HH` bits (`bits_of_dirsTie`). -/
theorem copyBitsR_of (pieces : Msg → List OpenPiece) (posAt : Msg → List Coin → OpenPiece → ℕ)
    (hP : ∀ x, D.parse x = parse x) (hS : ∀ f, shape f → Claims.Recursion.Form.Dense f)
    (hSch : SideSchR H D shape untr b cmt L Bd side pieces) (hD : CopyDirsR H D shape untr b cmt L Bd pieces posAt) :
    CopyBitsR H D shape untr b cmt L Bd side pieces posAt := by
  intro x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO w p hst q hq ho ht os last hfin hl
  obtain ⟨hH32, htie⟩ := hD x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO w p hst q hq ho ht
  obtain ⟨-, hH⟩ := sideLevel D shape side hP hS hx hf hr hss hsv ho
    (hSch x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO w p hst q hq ho ht os last hfin hl)
  intro h hh
  exact bits_of_dirsTie hH32 htie h (by rw [hH]; exact hh)

/-- **The `rec-open` pieces' share at the deployed layout**: at the parts' own messages, every part accepting, each
piece `ownQR` gives part `s` holds (`openHolds` at `side` and `posAt`), from `OpenCopyR` and `OpenTieR`. A
part that is no level part is given no piece. -/
theorem openPieces_deployed (pieces : Msg → List OpenPiece) (posAt : Msg → List Coin → OpenPiece → ℕ)
    (hC : OpenCopyR H D shape untr b cmt L Bd side pieces) (hT : OpenTieR H D shape untr b cmt L Bd side pieces posAt)
    {x : Msg} {st : Claims.Recursion.Stmt} (hx : D.parse x = some st) (hf : shape st.form) {c : ℕ → Coin}
    {ws : List Claims.Flock.Wit}
    (hJ : Joint H D L (PartAccepts H D (deployZkRec _root_.Flock.Readable.relation) L shape (showsPart untr b) cmt)
      x c ws) {s : ℕ} (hs : s < D.nS x) :
    PiecesComputes Bd D pieces (openHolds side posAt) ownQR s x c (L.msgs x ws) := by
  intro q hq ho
  obtain ⟨-, r, ss, hr, hss, LANES, HH, n, hsv⟩ := ownQR_some ho
  refine openPieces_computesA (hJ.2.2.2 s hs) hr hss (Or.inl hsv) Bd pieces ownQR side posAt
    (fun xO w p hxO hsz hcp hA hT' hc hsh hck hsg q' hq' ho' => ?_) q hq ho
  have hst : StagedAt H D untr b cmt L s x c (L.msgs x ws) (.recOpen q.level LANES HH n) xO w p :=
    ⟨hxO, hsz, hcp _ hsv, hA, hT', hc, hsh, hck, hsg⟩
  obtain ⟨hob, hgb, hub, hcp'⟩ :=
    hC x st hx hf c ws hJ r ss s q.level LANES HH n hr hss hsv xO w p hst q' hq' ho' hsz
  exact ⟨0, hsz, _, _, _, hob, hgb, hub, fun os last hfin hl =>
    ⟨hcp' os last hfin hl, hT x st hx hf c ws hJ r ss s q.level LANES HH n hr hss hsv xO w p hst q' hq' ho' hsz os
      last hfin hl⟩⟩

end Deployed

end RecDeploy.OpenLower
