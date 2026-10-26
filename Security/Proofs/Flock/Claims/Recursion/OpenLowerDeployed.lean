import Proofs.Flock.Claims.Recursion.OpenPieces
import Proofs.Flock.Claims.Recursion.DeployedR

/-!
# The `rec-open` pieces' lowering at the deployed layout (`OpenCopy`, `OpenTie`)

`openPieces_computes` reads a part's `rec-open` pieces off `PartAccepts`, from the layout fact `hcopy` (`OpenCopy`) and
the registration ties `htie` (`OpenTie`). At the deployed layout `layR elem` neither holds in the form it takes them:

* `hcopy` is handed only `L.RowsRead`, which says nothing at a layout whose messages carry no shared row (every
  deployed one: `rowIn = none`). The copy's row and siblings are tied to the last message only by the part's pieces of
  it (`LayoutB.Agrees`' second conjunct), which `hcopy` isn't given.
* Both are asked at every `ms` agreeing with part `s` alone, but the side a piece opens against is read off the whole
  last message (`finalR`'s trailer, `rdR`'s framing) and every opened round (`readRep`'s fit). Other parts' bytes,
  their constant ones included, are free in such an `ms`. `DecompB` reads the parts' checks only at the parts' own
  messages `L.msgs x ws`, every part accepting there (`decompB_of_joint`), where every piece's constants are in place.

So this file states the lowering in the form the deployed layout meets:

* `Joint`, `decompB_of_joint`: `DecompB` at `chk'` from `DecompB` at `chk`, wherever every part's `chk'` at the parts'
  own messages gives every part's `chk` there.
* `openPieces_computesA`: `openPieces_computes` with one hypothesis `hopen`, given the part's whole agreement
  (`LayoutB.Agrees`) and naming the piece's copy, at which `OpenCopy` and `OpenTie` hold.
* The deployed choices: `ownQR` (a piece of table 0 at level `ℓ` to V\*'s statement `.recOpen ℓ …`), `copyAtR` (rep
  `r`'s opening `j`, instance `r q + j`, at block `n / g`, VU `n % g`, copy 0 of the part's one table), and `sideAtR`
  (the side `RecDeploy.Rows.sideAt` reads, without `readRep`'s choice: the openings of `rdR`'s proof, the cap of round
  0's rows and the recursive caps the rep's rounds commit, which is what `readRep` keeps wherever it fits).
* `OpenCopyR`, `OpenTieR`: the two facts at the deployed layout, at the parts' own messages with every part accepting.
  `openPieces_deployed` is `PiecesComputes` at them.
-/

namespace RecDeploy.OpenLower

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP setupOf tabsOf repCtx repLens cutLens rowsOf topGroups)
open Claims.Recursion.VStarB
open FlockSoundness.Refine (openAt)
open FlockVBridge.Family (bytesAt outCol plain)
open RecDeploy.OpenedPlain (QueryOpens queryOpens_of_outs)
open RecDeploy.DeployedR (rdR finalR bdR layR)

/-! ## `DecompB` reads the parts' checks only at their own messages -/

section Joint

variable {Coin' : Type} [Fintype Coin'] [Nonempty Coin']

/-- **Every part's `chk` at the parts' own messages**: `nS` witnesses, their recorded calls true, and every part's
check at the messages the layout reads off them. -/
def Joint (H : Msg → Msg) (D : Claims.Recursion.Deploy Coin') (L : LayoutB)
    (chk : ℕ → Msg → (ℕ → Coin') → List Msg → Prop) (x : Msg) (c : ℕ → Coin') (ws : List Claims.Flock.Wit) : Prop :=
  ws.length = D.nS x ∧ (∀ e ∈ L.allCalls x ws, H e.1 = e.2) ∧ ∀ s < D.nS x, chk s x c (L.msgs x ws)

/-- **`DecompB` is antitone in the parts' checks at their own messages**: it holds at `chk'` wherever it holds at
`chk` and every part's `chk'` at the parts' own messages gives every part's `chk` there. -/
theorem decompB_of_joint {H : Msg → Msg} {D : Claims.Recursion.Deploy Coin'} {IV : Claims.Verifier Coin'}
    {L : LayoutB} {shape : Claims.Recursion.Form → Prop} {chk chk' : ℕ → Msg → (ℕ → Coin') → List Msg → Prop}
    (hle : ∀ (x : Msg) (st : Claims.Recursion.Stmt), D.parse x = some st → shape st.form →
      ∀ (c : ℕ → Coin') (ws : List Claims.Flock.Wit), Joint H D L chk' x c ws →
        ∀ s < D.nS x, chk s x c (L.msgs x ws))
    (hD : DecompB H D IV L shape chk) : DecompB H D IV L shape chk' :=
  fun x st hx hf c ws hl hT hC => hD x st hx hf c ws hl hT (hle x st hx hf c ws ⟨hl, hT, hC⟩)

end Joint

/-! ## `openPieces_computes` at the part's whole agreement -/

/-- **The `rec-open` pieces' share at a level part, from the part's whole agreement**: `openPieces_computes` with its
two hypotheses as one, `hopen`, which is given everything `PartAccepts` names at the part (the parsed statement, the
agreement `LayoutB.Agrees`, the true and collision-free calls, the shown coins) and names, for each piece `ownQ` gives
the part, the copy at which the piece's `OpenCopy` and `OpenTie` hold. -/
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
      _root_.Flock.Readable.parseBytes readOfO xO = some p → L.Agrees cmt x s w xO ms →
      (∀ e ∈ L.calls x s w, H e.1 = e.2) →
      (Claims.findColl (L.calls x s w ++ vcallsAt H D xO)).isSome = false → showsTr untr s x xO c →
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
  intro q hq ho os last hf hl
  obtain ⟨t, ht, o, g, u, hob, hgb, hub, hcq⟩ := hopen xO w p hx hA hT hc hs q hq ho
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

/-- The queries of level `ℓ` of the inner schedule at the inner statement `xI`: its table 0's `fast100` level's `q`. -/
def qAt (xI : Msg) (ℓ : ℕ) : ℕ :=
  match setupOf false xI with
  | some v => (((fast100 (tabsOf v.2.2)[0]!.m).getD default).levels[ℓ]!).q
  | none => 0

/-- **A piece's copy** at a parsed level statement `p`: rep `r`'s opening `j` is instance `n = r · q + j`
(`RecCoins.checkOpen`'s order), at block `n / g`, VU `n % g`, copy 0 of the part's one table (`rowHidden`). -/
def copyAtR (p : _root_.Flock.Readable.Parsed) (xI : Msg) (q : OpenPiece) : ℕ × ℕ × ℕ :=
  let n := q.rep * qAt xI q.level + q.query
  (n / p.sts[0]!.c.g, n % p.sts[0]!.c.g, 0)

/-- **The side a `rec-open` piece reads, without `readRep`'s choice**: at a statement that sets up and messages whose
last one `rd` reads, with round 0's and the later rounds' rows read back, rep `k = 2 table + rep`'s table's `fast100`
schedule, the level-0 cap round 0's rows give it (`repCtx`'s `cap0`), and the last message's `k`-th Ligerito proof with
its recursive caps the tops rep `k`'s rounds commit (`topGroups`). Where `readRep` fits, it keeps exactly these:
`FitsRep.cap`, `FitsRep.tops`, and `absorbedFrom` keeps the last message's openings. -/
def sideAtR (rd : Msg → Option FinalP) (x : Msg) (ms : List Msg) (q : OpenPiece) : OpenSide :=
  match setupOf false x, ms.getLast?.bind rd, rowsOf (ms.headD []), ms.tail.dropLast.mapM rowsOf with
  | some v, some fl, some r₀, some rss =>
    ⟨(fast100 (tabsOf v.2.2)[q.table]!.m).getD default,
      ((repCtx v.2.1 v.2.2 default r₀ [] (2 * q.table + q.rep)).cap0).toArray,
      { (fl.ps[2 * q.table + q.rep]!).ligerito with
        recursiveCaps := ((topGroups (repCtx v.2.1 v.2.2 default r₀ [] (2 * q.table + q.rep)).sched
          ((cutLens (repLens v.2.1 v.2.2) rss).getD (2 * q.table + q.rep) [])).map List.toArray).toArray }⟩
  | _, _, _, _ => ⟨default, default, default⟩

/-! ## The two facts at the deployed layout -/

section Deployed

variable (H : Msg → Msg) (D : Claims.Recursion.Deploy Coin) (shape : Claims.Recursion.Form → Prop)
  (untr : TrReader) (b : Brings) (cmt : Msg → ℕ → Msg → Msg)
  (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))

/-- **Part `s` accepts at its staged statement `xO` and witness `w`**, at the parts' own messages `ms`: what
`partAccepts_staged` names, that `openPieces_computesA`'s `hopen` is given. -/
def StagedAt (s : ℕ) (x : Msg) (c : ℕ → Coin) (ms : List Msg) (xO : Msg) (w : Claims.Flock.Wit)
    (p : _root_.Flock.Readable.Parsed) : Prop :=
  _root_.Flock.Readable.parseBytes readOfO xO = some p ∧ (layR elem).Agrees cmt x s w xO ms ∧
    (∀ e ∈ (layR elem).calls x s w, H e.1 = e.2) ∧
    (Claims.findColl ((layR elem).calls x s w ++ vcallsAt H D xO)).isSome = false ∧ showsTr untr s x xO c

/-- **The layout fact at the deployed layout** (U13's first half): at the parts' own messages, every part accepting,
at a level part `s` (`.recOpen ℓ LANES HH n`), its staged statement and witness, each piece `ownQR` gives the part
(of table 0, a rep `< 2`, a level and query of the inner schedule) has its copy `copyAtR` among the statement's, and at
every reading of the last message the copy is the level's (`LANES`, `HH`) and reads the piece's opened row and
siblings at `sideAtR rdR` (`OpenCopy`). -/
def OpenCopyR (pieces : Msg → List OpenPiece) : Prop :=
  ∀ (x : Msg) (c : ℕ → Coin) (ws : List Claims.Flock.Wit),
    Joint H D (layR elem) (PartAccepts H D (deployZkRec _root_.Flock.Readable.relation) (layR elem) shape
      (showsPart untr b) cmt) x c ws →
    ∀ (r : XR) (ss : Array VStar.Stmt) (s lv LANES HH n : ℕ), readX x = some r →
      _root_.Flock.RecVStar.statements r.shape = .ok ss → ss[s]? = some (.recOpen lv LANES HH n) →
    ∀ (xO : Msg) (w : Claims.Flock.Wit) (p : _root_.Flock.Readable.Parsed),
      StagedAt H D untr cmt elem s x c ((layR elem).msgs x ws) xO w p →
    ∀ q ∈ pieces (Claims.Recursion.xI D x), ownQR x q = some s →
      ∃ ht : 0 < p.sts.size,
        (copyAtR p (Claims.Recursion.xI D x) q).1 < 2 ^ (p.sts[0].m - p.sts[0].c.kLog) ∧
        (copyAtR p (Claims.Recursion.xI D x) q).2.1 < p.sts[0].c.g ∧
        (copyAtR p (Claims.Recursion.xI D x) q).2.2 < p.sts[0].c.upv ∧
        ∀ os last, bdR.final (((layR elem).msgs x ws).getD (bdR.DE.V.k (Claims.Recursion.xI D x) - 1) []) =
            some (os, last) → os.length + 1 = bdR.DE.V.k (Claims.Recursion.xI D x) →
          OpenCopy (sideAtR rdR (Claims.Recursion.xI D x) (bdR.opened os last) q) q LANES HH
            (_root_.Flock.Readable.copyOf p 0 ht w (copyAtR p (Claims.Recursion.xI D x) q).1
              (copyAtR p (Claims.Recursion.xI D x) q).2.1 (copyAtR p (Claims.Recursion.xI D x) q).2.2)

/-- **The registration ties at the deployed layout** (U13's second half): in `OpenCopyR`'s context, at the copy
`copyAtR`, the position bits it reads are `posAt`'s and its climb's top is the cap's entry there, at `sideAtR rdR`
(`OpenTie`). -/
def OpenTieR (pieces : Msg → List OpenPiece) (posAt : Msg → List Coin → OpenPiece → ℕ) : Prop :=
  ∀ (x : Msg) (c : ℕ → Coin) (ws : List Claims.Flock.Wit),
    Joint H D (layR elem) (PartAccepts H D (deployZkRec _root_.Flock.Readable.relation) (layR elem) shape
      (showsPart untr b) cmt) x c ws →
    ∀ (r : XR) (ss : Array VStar.Stmt) (s lv LANES HH n : ℕ), readX x = some r →
      _root_.Flock.RecVStar.statements r.shape = .ok ss → ss[s]? = some (.recOpen lv LANES HH n) →
    ∀ (xO : Msg) (w : Claims.Flock.Wit) (p : _root_.Flock.Readable.Parsed),
      StagedAt H D untr cmt elem s x c ((layR elem).msgs x ws) xO w p →
    ∀ q ∈ pieces (Claims.Recursion.xI D x), ownQR x q = some s → ∀ ht : 0 < p.sts.size,
      ∀ os last, bdR.final (((layR elem).msgs x ws).getD (bdR.DE.V.k (Claims.Recursion.xI D x) - 1) []) =
          some (os, last) → os.length + 1 = bdR.DE.V.k (Claims.Recursion.xI D x) →
        OpenTie (sideAtR rdR (Claims.Recursion.xI D x) (bdR.opened os last) q) q LANES HH
          (posAt (Claims.Recursion.xI D x) ((List.range (bdR.DE.V.k (Claims.Recursion.xI D x))).map c) q)
          (_root_.Flock.Readable.copyOf p 0 ht w (copyAtR p (Claims.Recursion.xI D x) q).1
            (copyAtR p (Claims.Recursion.xI D x) q).2.1 (copyAtR p (Claims.Recursion.xI D x) q).2.2)

/-- **The `rec-open` pieces' share at the deployed layout**: at the parts' own messages, every part accepting, each
piece `ownQR` gives part `s` holds (`openHolds` at `sideAtR rdR` and `posAt`), from `OpenCopyR` and `OpenTieR`. A part
that is no level part is given no piece. -/
theorem openPieces_deployed (pieces : Msg → List OpenPiece) (posAt : Msg → List Coin → OpenPiece → ℕ)
    (hC : OpenCopyR H D shape untr b cmt elem pieces) (hT : OpenTieR H D shape untr b cmt elem pieces posAt)
    {x : Msg} {c : ℕ → Coin} {ws : List Claims.Flock.Wit}
    (hJ : Joint H D (layR elem) (PartAccepts H D (deployZkRec _root_.Flock.Readable.relation) (layR elem) shape
      (showsPart untr b) cmt) x c ws) {s : ℕ} (hs : s < D.nS x) :
    PiecesComputes bdR D pieces (openHolds (sideAtR rdR) posAt) ownQR s x c ((layR elem).msgs x ws) := by
  intro q hq ho
  have ho' := ho
  unfold ownQR at ho'
  split at ho'
  · obtain ⟨r, hr, ho'⟩ := Option.bind_eq_some_iff.1 ho'
    obtain ⟨ss, hss, hfi⟩ := Option.bind_eq_some_iff.1 ho'
    have hss' : _root_.Flock.RecVStar.statements r.shape = .ok ss := by
      cases h : _root_.Flock.RecVStar.statements r.shape with
      | error e => rw [h] at hss; cases hss
      | ok v => rw [h] at hss; cases hss; rfl
    obtain ⟨hsz, hlev, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hfi
    have hsv : ∃ lv LANES HH n, ss[s]? = some (.recOpen lv LANES HH n) := by
      rw [Array.getElem?_eq_getElem hsz]
      revert hlev
      cases ss[s] with
      | recOpen l L' H' n' => exact fun _ => ⟨l, L', H', n', rfl⟩
      | _ => intro h; simp [isLevel] at h
    obtain ⟨lv, LANES, HH, n, hsv⟩ := hsv
    refine openPieces_computesA (hJ.2.2 s hs) hr hss' (Or.inl hsv) bdR pieces ownQR (sideAtR rdR) posAt
      (fun xO w p hx hA hT' hc hsh q hq ho => ?_) q hq ho
    have hst : StagedAt H D untr cmt elem s x c ((layR elem).msgs x ws) xO w p := ⟨hx, hA, hT', hc, hsh⟩
    obtain ⟨ht, hob, hgb, hub, hcp⟩ := hC x c ws hJ r ss s lv LANES HH n hr hss' hsv xO w p hst q hq ho
    exact ⟨0, ht, _, _, _, hob, hgb, hub, fun os last hf hl =>
      ⟨hcp os last hf hl, hT x c ws hJ r ss s lv LANES HH n hr hss' hsv xO w p hst q hq ho ht os last hf hl⟩⟩
  · cases ho'

end Deployed

end RecDeploy.OpenLower
