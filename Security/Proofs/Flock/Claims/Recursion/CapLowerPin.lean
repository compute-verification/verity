import Proofs.Flock.Claims.Recursion.CapLower
import Proofs.Flock.Claims.Recursion.PartAcceptsPin

/-!
# The `rec-open` pieces' facts with each part's check at its own witness (`AtPiecesP`, `OutBindRP`)

`AtPieces` asks a piece's fact at the parts' own messages with every part accepting (`Joint` at `PartAccepts`), at
some staged witness `w` of part `s`. That `w` need not be `ws[s]`, so its calls need not be among `L.allCalls x ws`,
which `Joint` has true and free of collisions: `OutBindR`, which binds a level part's `out` slot to RecCap's `top` slot
across parts through those calls (`out_top_rows`), can't be met there. `AtPiecesP` asks it under `JointP` (each part's
check at `ws[s]`, `PartAcceptsPin`), at `w = ws[s]`.

* `AtPiecesP`, `atPiecesP_of`: every fact at `AtPieces` holds at `AtPiecesP` (`joint_of_jointP`), so only the facts
  that read across parts are restated: `CopyTopRP`, `OpenTieRP` (`openTieRP_of`).
* `openPieces_computesP`, `openPieces_deployedP`: `openPieces_computesA` and `openPieces_deployed` with the check at
  `ws[s]`, so `PiecesComputes` under `JointP` from `OpenCopyR` and `OpenTieRP`.
* `OutBindRP`, `capTop_ofP`, `openTieR_deployedP`: the binding at `AtPiecesP`, and the cap tie from it and the
  reader (`SideCapR`, unpinned).
* `OutTopTie`, `outBindRP_of`: at the layout of record `DeployedR.layR`, the binding from `out_top_rows`, wherever the
  copy's `out` columns are its opening's `out` row's node, the two slots' commit strings are one and their ports share
  a row prefix (`OutTopTie`).
-/

namespace RecDeploy.OpenLower

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP)
open Claims.Recursion.VStarB
open FlockVBridge.Family (bytesAt outCol plain)
open RecDeploy.OpenedPlain (QueryOpens queryOpens_of_outs)

/-! ## `openPieces_computesA` at the part's own witness -/

/-- **`openPieces_computesA` at a given witness**: the part's check at `w` (`PartAcceptsP`), `hopen` asked at `w`. -/
theorem openPieces_computesP {H : Msg → Msg} {D : Claims.Recursion.Deploy Coin} {L : LayoutB}
    {shape : Claims.Recursion.Form → Prop} {untr : TrReader} {b : Brings} {cmt : Msg → ℕ → Msg → Msg} {s : ℕ}
    {x : Msg} {c : ℕ → Coin} {w : Claims.Flock.Wit} {ms : List Msg}
    (h : PartAcceptsP H D (deployZkRec _root_.Flock.Readable.relation) L shape (showsPart untr b) cmt s x c w ms)
    {r : XR} (hr : readX x = some r) {ss : Array VStar.Stmt} (hss : _root_.Flock.RecVStar.statements r.shape = .ok ss)
    {lv LANES HH n : ℕ}
    (hsv : ss[s]? = some (.recOpen lv LANES HH n) ∨ ss[s]? = some (.recOpenStacked lv LANES HH n))
    (Bd : Claims.Recursion.Bound Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace)
    (pieces : Msg → List OpenPiece) (ownQ : Msg → OpenPiece → Option ℕ)
    (sideAt : Msg → List Msg → OpenPiece → OpenSide) (posAt : Msg → List Coin → OpenPiece → ℕ)
    (hopen : ∀ (xO : Msg) (p : _root_.Flock.Readable.Parsed),
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
  obtain ⟨-, xO, hck, hT, hc, hs, hh, hp, hm, hrw⟩ := h
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
  obtain ⟨t, ht, o, g, u, hob, hgb, hub, hcq⟩ := hopen xO p hx hsz hcps hA hT hc hs.1 hck hs.2 q hq ho
  obtain ⟨⟨hLA, hHH, hrow, hsib⟩, hbits, htop⟩ := hcq os last hf hl
  have hout := (hlev t ht o g u hob hgb hub).1
  obtain ⟨hrs, -, -, -, hsl, -⟩ := _root_.Flock.Readable.openIns_reads
    (_root_.Flock.Readable.copyOf p t ht w o g u) LANES HH
  exact queryOpens_of_outs (openOuts_plain hout) hrow.symm hsib (by rw [hrs, hLA]) (by rw [hsl, hHH]) hbits htop

/-! ## The facts at the parts' own witnesses -/

section Deployed

variable (H : Msg → Msg) (D : Claims.Recursion.Deploy Coin) (shape : Claims.Recursion.Form → Prop)
  (untr : TrReader) (b : Brings) (cmt : Msg → ℕ → Msg → Msg) (L : LayoutB)
  (Bd : Claims.Recursion.Bound Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace)
  (side : Msg → List Msg → OpenPiece → OpenSide)

/-- **`AtPieces` with each part's check at its own witness**: at the parts' own messages with every part's check at
`ws[s]` (`JointP` at `PartAcceptsP`), a level part `s`, its staged statement and its witness `ws[s]`, and a piece
`ownQR` gives it. -/
abbrev AtPiecesP (pieces : Msg → List OpenPiece)
    (Fact : Msg → (ℕ → Coin) → List Claims.Flock.Wit → ℕ → ℕ → ℕ → (xO : Msg) → Claims.Flock.Wit →
      (p : _root_.Flock.Readable.Parsed) → 0 < p.sts.size → OpenPiece → Prop) : Prop :=
  ∀ (x : Msg) (st : Claims.Recursion.Stmt), D.parse x = some st → shape st.form →
    ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit),
    JointP H D L (PartAcceptsP H D (deployZkRec _root_.Flock.Readable.relation) L shape (showsPart untr b) cmt)
      x c ws →
    ∀ (r : XR) (ss : Array VStar.Stmt) (s lv LANES HH n : ℕ), readX x = some r →
      _root_.Flock.RecVStar.statements r.shape = .ok ss → ss[s]? = some (.recOpen lv LANES HH n) →
    ∀ (xO : Msg) (w : Claims.Flock.Wit) (p : _root_.Flock.Readable.Parsed), ws[s]? = some w →
      StagedAt H D untr b cmt L s x c (L.msgs x ws) (.recOpen lv LANES HH n) xO w p →
    ∀ q ∈ pieces (Claims.Recursion.xI D x), ownQR x q = some s → ∀ ht : 0 < p.sts.size,
      Fact x c ws LANES HH s xO w p ht q

/-- **A fact at `AtPieces` holds at `AtPiecesP`**: `JointP` gives `Joint` (`joint_of_jointP`). -/
theorem atPiecesP_of {pieces : Msg → List OpenPiece}
    {Fact : Msg → (ℕ → Coin) → List Claims.Flock.Wit → ℕ → ℕ → ℕ → (xO : Msg) → Claims.Flock.Wit →
      (p : _root_.Flock.Readable.Parsed) → 0 < p.sts.size → OpenPiece → Prop}
    (h : AtPieces H D shape untr b cmt L pieces Fact) : AtPiecesP H D shape untr b cmt L pieces Fact :=
  fun x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO w p _ hst q hq ho ht =>
    h x st hx hf c ws (joint_of_jointP hJ) r ss s lv LANES HH n hr hss hsv xO w p hst q hq ho ht

/-- **`CopyTopR` at the parts' own witnesses.** -/
def CopyTopRP (pieces : Msg → List OpenPiece) (posAt : Msg → List Coin → OpenPiece → ℕ) : Prop :=
  AtPiecesP H D shape untr b cmt L pieces fun x c ws LANES HH _ _ w p ht q =>
    AtOpened D Bd x (L.msgs x ws) fun ms' =>
      bytesAt (copyR (Claims.Recursion.xI D x) w p ht q) (outCol LANES HH) 64 =
        (FlockSoundness.Refine.capAt (side (Claims.Recursion.xI D x) ms' q).cap
            (side (Claims.Recursion.xI D x) ms' q).lig q.level).getD
          ((posAt (Claims.Recursion.xI D x) ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) q) >>>
            (((side (Claims.Recursion.xI D x) ms' q).sch.levels[q.level]!).d -
              ((side (Claims.Recursion.xI D x) ms' q).sch.levels[q.level]!).cap)) ByteArray.empty

/-- **`OpenTieR` at the parts' own witnesses.** -/
def OpenTieRP (pieces : Msg → List OpenPiece) (posAt : Msg → List Coin → OpenPiece → ℕ) : Prop :=
  AtPiecesP H D shape untr b cmt L pieces fun x c ws LANES HH _ _ w p ht q =>
    AtOpened D Bd x (L.msgs x ws) fun ms' =>
      OpenTie (side (Claims.Recursion.xI D x) ms' q) q LANES HH
        (posAt (Claims.Recursion.xI D x) ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) q)
        (copyR (Claims.Recursion.xI D x) w p ht q)

/-- **`OpenTieRP` is its two halves**, the position bits unpinned (`CopyBitsR`, `atPiecesP_of`). -/
theorem openTieRP_of (pieces : Msg → List OpenPiece) (posAt : Msg → List Coin → OpenPiece → ℕ)
    (hB : CopyBitsR H D shape untr b cmt L Bd side pieces posAt)
    (hT : CopyTopRP H D shape untr b cmt L Bd side pieces posAt) :
    OpenTieRP H D shape untr b cmt L Bd side pieces posAt :=
  fun x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO w p hw hst q hq ho ht os last hfin hl =>
    And.intro
      (atPiecesP_of H D shape untr b cmt L hB x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO w p hw hst q hq ho
        ht os last hfin hl)
      (hT x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO w p hw hst q hq ho ht os last hfin hl)

/-- **The `rec-open` pieces' share under `JointP`**: `openPieces_deployed` with part `s`'s check at `ws[s]`
(`openPieces_computesP`), from `OpenCopyR` (unpinned) and `OpenTieRP`. -/
theorem openPieces_deployedP (pieces : Msg → List OpenPiece) (posAt : Msg → List Coin → OpenPiece → ℕ)
    (hC : OpenCopyR H D shape untr b cmt L Bd side pieces)
    (hT : OpenTieRP H D shape untr b cmt L Bd side pieces posAt)
    {x : Msg} {st : Claims.Recursion.Stmt} (hx : D.parse x = some st) (hf : shape st.form) {c : ℕ → Coin}
    {ws : List Claims.Flock.Wit}
    (hJ : JointP H D L (PartAcceptsP H D (deployZkRec _root_.Flock.Readable.relation) L shape (showsPart untr b) cmt)
      x c ws) {s : ℕ} (hs : s < D.nS x) :
    PiecesComputes Bd D pieces (openHolds side posAt) ownQR s x c (L.msgs x ws) := by
  have hsw : s < ws.length := lt_of_lt_of_eq hs hJ.1.symm
  have hw : ws[s]? = some ws[s] := List.getElem?_eq_getElem hsw
  intro q hq ho
  obtain ⟨-, r, ss, hr, hss, LANES, HH, n, hsv⟩ := ownQR_some ho
  refine openPieces_computesP (hJ.2.2.2 s hsw) hr hss (Or.inl hsv) Bd pieces ownQR side posAt
    (fun xO p hxO hsz hcp hA hT' hc hsh hck hsg q' hq' ho' => ?_) q hq ho
  have hst : StagedAt H D untr b cmt L s x c (L.msgs x ws) (.recOpen q.level LANES HH n) xO ws[s] p :=
    ⟨hxO, hsz, hcp _ hsv, hA, hT', hc, hsh, hck, hsg⟩
  obtain ⟨hob, hgb, hub, hcp'⟩ :=
    hC x st hx hf c ws (joint_of_jointP hJ) r ss s q.level LANES HH n hr hss hsv xO ws[s] p hst q' hq' ho' hsz
  exact ⟨0, hsz, _, _, _, hob, hgb, hub, fun os last hfin hl =>
    ⟨hcp' os last hfin hl, hT x st hx hf c ws hJ r ss s q.level LANES HH n hr hss hsv xO ws[s] p hw hst q' hq' ho' hsz
      os last hfin hl⟩⟩

end Deployed

end RecDeploy.OpenLower

namespace RecDeploy.CapLower

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP nodeOf)
open Claims.Recursion.VStarB
open FlockVBridge.Family (bytesAt outCol)
open RecDeploy.OpenLower (AtPieces AtPiecesP atPiecesP_of AtOpened copyR schAt sideAtR sideLevel CopyBitsR CopyTopRP
  OpenTieRP openTieRP_of)

/-! ## The level part's `out` port and RecCap's `top` port are 64-word rows -/

/-- **The port found by name is a parsed row of that name**: where the circuit's row ports are `R` (by name, words and
bits), the port `findIdx?` finds by `nm` is an entry of `R` named `nm`. -/
theorem port_mem_rows {c : Circuit} {R : List (String × ℕ × ℕ)}
    (hr : c.ports.toList.map (fun p => (p.name, p.words, p.bits)) = R) {nm : String} {q : ℕ}
    (hq : c.ports.findIdx? (·.name == nm) = some q) :
    ∃ hq' : q < c.ports.size, (nm, c.ports[q].words, c.ports[q].bits) ∈ R := by
  obtain ⟨hqs, hname, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hq
  have hn : c.ports[q].name = nm := by simpa using hname
  refine ⟨hqs, ?_⟩
  rw [← hr, ← hn]
  exact List.mem_map.2 ⟨c.ports[q], Array.getElem_mem_toList hqs, rfl⟩

/-- **A parsed row's name is some port's**: where the circuit's row ports are `R`, `findIdx?` finds a port by the
name of any entry of `R`. -/
theorem port_of_rows {c : Circuit} {R : List (String × ℕ × ℕ)}
    (hr : c.ports.toList.map (fun p => (p.name, p.words, p.bits)) = R) {e : String × ℕ × ℕ} (he : e ∈ R) :
    ∃ q, c.ports.findIdx? (·.name == e.1) = some q := by
  cases h : c.ports.findIdx? (·.name == e.1) with
  | some q => exact ⟨q, rfl⟩
  | none =>
    rw [← hr] at he
    obtain ⟨p, hp, rfl⟩ := List.mem_map.1 he
    have := Array.findIdx?_eq_none_iff.1 h p (Array.mem_toList_iff.1 hp)
    simp at this

/-- A port's row prefix is its word count's. -/
theorem rowPfx_of_words {c : Circuit} {q k : ℕ} (hq : q < c.ports.size) (hk : c.ports[q].words = k) :
    rowPfx c q = (HmRow.rowPrefix k).data.toList := by
  unfold rowPfx
  rw [Array.getElem?_eq_getElem hq, Option.map_some, Option.getD_some, hk]

/-- **A staged part's circuit has V\*'s row ports**: V\*'s parse check at the part's table (`stagedTables`,
`checkParsed_ok`), whose circuit is the part's (`partCircuit_staged`). -/
theorem rows_staged {st : Stage} {s : ℕ} {x xO : Msg} (hS : StagedPart st s x xO) {r : XR}
    (hr : readX x = some r) {ss : Array VStar.Stmt} (hss : _root_.Flock.RecVStar.statements r.shape = .ok ss)
    {sv : VStar.Stmt} (hsv : ss[s]? = some sv) {c : Circuit} (hc : r.w.parts[s]?.bind partCircuit = some c) :
    ∃ u, sv.unit = .ok u ∧
      c.ports.toList.map (fun p => (p.name, p.words, p.bits)) = _root_.Flock.RecVStar.vstarRows sv.sub u := by
  obtain ⟨p, r', sv', hx, ht, hpi, hr', ⟨ss', hss', hsv'⟩, h0, htab⟩ := stagedTables hS
  rw [hr] at hr'
  cases hr'
  rw [hss] at hss'
  cases hss'
  rw [hsv] at hsv'
  cases hsv'
  have hc0 := partCircuit_staged hS hx ht hpi hr hc h0
  obtain ⟨u, hu, -, -, -, hrows⟩ := FlockSoundness.Discharge.Recursion.checkParsed_ok (htab 0 h0).2.2
  rw [hc0] at hrows
  exact ⟨u, hu, hrows⟩

/-- **A level part's `out` port is a 64-word row**: V\*'s rows of `rec-open` (`vstarRows`) name `out` once, an output
of `OUT_WORDS = 64` words (`recOpenUnit_ok`). -/
theorem out_port_recOpen {lv LANES HH n : ℕ} {u : VStar.UnitC}
    (hu : (VStar.Stmt.recOpen lv LANES HH n).unit = .ok u) {c : Circuit}
    (hr : c.ports.toList.map (fun p => (p.name, p.words, p.bits)) =
      _root_.Flock.RecVStar.vstarRows (VStar.Stmt.recOpen lv LANES HH n).sub u) :
    ∃ q, c.ports.findIdx? (·.name == "out") = some q ∧ ∃ hq : q < c.ports.size, c.ports[q].words = 64 := by
  obtain ⟨-, -, -, hout⟩ := FlockVBridge.OfVStar.recOpenUnit_ok (show VStar.recOpenUnit LANES HH = .ok u from hu)
  have hm : (("out" : String), (64 : ℕ), (1024 : ℕ)) ∈
      _root_.Flock.RecVStar.vstarRows (VStar.Stmt.recOpen lv LANES HH n).sub u := by
    unfold _root_.Flock.RecVStar.vstarRows
    rw [hout]
    refine List.mem_append_right _ (List.mem_map.2 ⟨("out", VStar.OUT_WORDS), ?_, rfl⟩)
    show ("out", VStar.OUT_WORDS) ∈ [("out", VStar.OUT_WORDS), ("acc_out", VStar.ACC_WORDS)]
    exact List.mem_cons_self
  obtain ⟨q, hq⟩ := port_of_rows hr hm
  have hq0 : c.ports.findIdx? (·.name == "out") = some q := hq
  obtain ⟨hq', hm'⟩ := port_mem_rows hr hq0
  refine ⟨q, hq0, hq', ?_⟩
  unfold _root_.Flock.RecVStar.vstarRows at hm'
  rw [hout] at hm'
  rcases List.mem_append.1 hm' with h | h
  · obtain ⟨⟨a, k⟩, hy, he⟩ := List.mem_map.1 h
    have ha : a = "out" := congrArg Prod.fst he
    subst ha
    have hin := (List.of_mem_zip hy).1
    exact absurd hin (show "out" ∉ (["row", "aux", "acc", "coef", "dirs"] : List String) by decide)
  · obtain ⟨⟨a, k⟩, hy, he⟩ := List.mem_map.1 h
    have hy' : (a, k) ∈ [("out", VStar.OUT_WORDS), ("acc_out", VStar.ACC_WORDS)] := hy
    rcases List.mem_cons.1 hy' with h1 | h1
    · cases h1
      have h2 : (VStar.OUT_WORDS + 63) / 64 * 64 = c.ports[q].words := congrArg (fun e => e.2.1) he
      rw [← h2]
      rfl
    · rw [List.mem_singleton] at h1
      cases h1
      exact absurd (show ("acc_out" : String) = "out" from congrArg Prod.fst he) (by decide)

/-- **RecCap's `top` port is a 64-word row**: V\*'s rows of `rec-cap` (`vstarRows`) name `top` once, its one input of
`CAP_TOP_WORDS = 64` words (`capUnit`). -/
theorem top_port_cap {l N : ℕ} {u : VStar.UnitC} (hu : (VStar.Stmt.cap l N).unit = .ok u) {c : Circuit}
    (hr : c.ports.toList.map (fun p => (p.name, p.words, p.bits)) =
      _root_.Flock.RecVStar.vstarRows (VStar.Stmt.cap l N).sub u) :
    ∃ q, capPort c = some q ∧ ∃ hq : q < c.ports.size, c.ports[q].words = 64 := by
  have hu' : (Except.ok (VStar.capUnit l N) : Except String VStar.UnitC) = .ok u := hu
  obtain rfl := Except.ok.inj hu'
  have hm : (("top" : String), (64 : ℕ), (1024 : ℕ)) ∈
      _root_.Flock.RecVStar.vstarRows (VStar.Stmt.cap l N).sub (VStar.capUnit l N) := by
    unfold _root_.Flock.RecVStar.vstarRows
    refine List.mem_append_left _ (List.mem_map.2 ⟨("top", VStar.CAP_TOP_WORDS), ?_, rfl⟩)
    show ("top", VStar.CAP_TOP_WORDS) ∈ [("top", VStar.CAP_TOP_WORDS)]
    exact List.mem_cons_self
  obtain ⟨q, hq⟩ := port_of_rows hr hm
  have hq0 : c.ports.findIdx? (·.name == "top") = some q := hq
  obtain ⟨hq', hm'⟩ := port_mem_rows hr hq0
  refine ⟨q, hq0, hq', ?_⟩
  unfold _root_.Flock.RecVStar.vstarRows at hm'
  rcases List.mem_append.1 hm' with h | h
  · obtain ⟨⟨a, k⟩, hy, he⟩ := List.mem_map.1 h
    have hy' : (a, k) ∈ [("top", VStar.CAP_TOP_WORDS)] := hy
    rw [List.mem_singleton] at hy'
    cases hy'
    have h2 : VStar.CAP_TOP_WORDS = c.ports[q].words := congrArg (fun e => e.2.1) he
    rw [← h2]
    rfl
  · obtain ⟨⟨a, k⟩, hy, he⟩ := List.mem_map.1 h
    have hy' : (a, k) ∈ [("out", VStar.CAP_OUT_WORDS)] := hy
    rw [List.mem_singleton] at hy'
    cases hy'
    exact absurd (show ("out" : String) = "top" from congrArg Prod.fst he) (by decide)

section Deployed

variable (H : Msg → Msg) (D : Claims.Recursion.Deploy Coin) (shape : Claims.Recursion.Form → Prop)
  (untr : TrReader) (b : Brings) (cmt : Msg → ℕ → Msg → Msg) (L : LayoutB)
  (Bd : Claims.Recursion.Bound Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace) (rd : Msg → Option FinalP)

/-- **The binding at the parts' own witnesses** (`OutBindR` at `AtPiecesP`): at a level part's piece, its witness
`ws[s]`, at every RecCap part of the piece's level holding the row the opening climbs to (`capIdx`), the copy's
climb's top is the node RecCap's `top` row carries there. -/
def OutBindRP (pieces : Msg → List OpenPiece) (posAt : Msg → List Coin → OpenPiece → ℕ) : Prop :=
  AtPiecesP H D shape untr b cmt L pieces fun x c ws LANES HH _ _ w p ht q =>
    ∀ (r : XR) (ss : Array VStar.Stmt), readX x = some r →
      _root_.Flock.RecVStar.statements r.shape = .ok ss →
    ∀ (s' N' : ℕ) (c' : Circuit) (w' : Claims.Flock.Wit), ss[s']? = some (.cap q.level N') →
      r.w.parts[s']?.bind partCircuit = some c' → ws[s']? = some w' →
      capIdx (Claims.Recursion.xI D x) q
          (posAt (Claims.Recursion.xI D x) ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) q) HH < N' →
      bytesAt (copyR (Claims.Recursion.xI D x) w p ht q) (outCol LANES HH) 64 =
        nodeOf (capRow c' w' (capIdx (Claims.Recursion.xI D x) q
          (posAt (Claims.Recursion.xI D x) ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) q) HH))

/-- **The cap tie at the parts' own witnesses**, from the binding there and the reader (`capTop_of`'s argument). -/
theorem capTop_ofP (pieces : Msg → List OpenPiece) (posAt : Msg → List Coin → OpenPiece → ℕ)
    (hP : ∀ x, D.parse x = parse x) (hD : ∀ f, shape f → Claims.Recursion.Form.Dense f)
    (hO : OutBindRP H D shape untr b cmt L Bd pieces posAt) (hS : SideCapR H D shape untr b cmt L Bd rd pieces posAt) :
    CopyTopRP H D shape untr b cmt L Bd (sideAtR rd) pieces posAt := by
  intro x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO w p hw hst q hq ho ht os last hfin hl
  obtain ⟨r', ss', hr', hss', s', N', c', w', hs', hc', hw', hlt, hcap⟩ :=
    atPiecesP_of H D shape untr b cmt L hS x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO w p hw hst q hq ho ht
      os last hfin hl
  obtain ⟨-, hHH⟩ := sideLevel D shape (sideAtR rd) hP hD hx hf hr hss hsv ho (ms := Bd.opened os last) rfl
  rw [← hHH]
  exact (hO x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO w p hw hst q hq ho ht r' ss' hr' hss' s' N' c' w'
    hs' hc' hw' hlt).trans hcap.symm

/-- **The registration ties at the parts' own witnesses**, at `sideAtR rd`: the position bits unpinned, the cap half
from the pinned binding and the reader (`capTop_ofP`). -/
theorem openTieR_deployedP (pieces : Msg → List OpenPiece) (posAt : Msg → List Coin → OpenPiece → ℕ)
    (hP : ∀ x, D.parse x = parse x) (hD : ∀ f, shape f → Claims.Recursion.Form.Dense f)
    (hB : CopyBitsR H D shape untr b cmt L Bd (sideAtR rd) pieces posAt)
    (hO : OutBindRP H D shape untr b cmt L Bd pieces posAt) (hS : SideCapR H D shape untr b cmt L Bd rd pieces posAt) :
    OpenTieRP H D shape untr b cmt L Bd (sideAtR rd) pieces posAt :=
  openTieRP_of H D shape untr b cmt L Bd (sideAtR rd) pieces posAt hB
    (capTop_ofP H D shape untr b cmt L Bd rd pieces posAt hP hD hO hS)

/-- **The two slots of a binding agree** (what `out_top_rows` is given): at a level part's piece and its witness
`ws[s]`, and a RecCap part of the piece's level holding the row the opening climbs to (`capIdx`), some opening `i`
of the level part's `out` port is the copy's, its row's node the copy's `out` columns; its slot and RecCap's `top`
slot at that row carry one commit string; and the two ports share a row prefix. -/
def OutTopTie (pieces : Msg → List OpenPiece) (posAt : Msg → List Coin → OpenPiece → ℕ) : Prop :=
  AtPiecesP H D shape untr b cmt L pieces fun x c ws LANES HH s _ w p ht q =>
    ∀ (r : XR) (ss : Array VStar.Stmt), readX x = some r →
      _root_.Flock.RecVStar.statements r.shape = .ok ss →
    ∀ (s' N' : ℕ) (c' : Circuit) (w' : Claims.Flock.Wit), ss[s']? = some (.cap q.level N') →
      r.w.parts[s']?.bind partCircuit = some c' → ws[s']? = some w' →
      capIdx (Claims.Recursion.xI D x) q
          (posAt (Claims.Recursion.xI D x) ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) q) HH < N' →
      ∃ (lv n : ℕ) (cc : Circuit) (qo i qc : ℕ), ss[s]? = some (.recOpen lv LANES HH n) ∧
        r.w.parts[s]?.bind partCircuit = some cc ∧ cc.ports.findIdx? (·.name == "out") = some qo ∧ i < n ∧
        capPort c' = some qc ∧ rowPfx cc qo = rowPfx c' qc ∧
        readForms w (hmCommit cc qo i) = readForms w' (hmCommit c' qc (capIdx (Claims.Recursion.xI D x) q
          (posAt (Claims.Recursion.xI D x) ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) q) HH)) ∧
        bytesAt (copyR (Claims.Recursion.xI D x) w p ht q) (outCol LANES HH) 64 =
          nodeOf ⟨(readForms w (DeployedR.outRowForms cc qo i)).toArray⟩

/-- **The binding at the layout of record**: where the two slots agree (`OutTopTie`), `layR` records both slots'
calls, which `JointP` has true and free of collisions, so the opening's `out` row is RecCap's `top` row
(`out_top_rows`) and the copy's top is that row's node. -/
theorem outBindRP_of (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (pieces : Msg → List OpenPiece) (posAt : Msg → List Coin → OpenPiece → ℕ)
    (hT : OutTopTie FlockDeploy.sha512 D shape untr b cmt (DeployedR.layR elem) Bd pieces posAt) :
    OutBindRP FlockDeploy.sha512 D shape untr b cmt (DeployedR.layR elem) Bd pieces posAt := by
  intro x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO w p hw hst q hq ho ht r' ss' hr' hss' s' N' c' w'
    hs' hc' hw' hlt
  obtain ⟨lv', n', cc, qo, i, qc, hsv', hcc, hqo, hi, hqc, hpfx, hcm, hcopy⟩ :=
    hT x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO w p hw hst q hq ho ht r' ss' hr' hss' s' N' c' w' hs'
      hc' hw' hlt
  have hsvO : (_root_.Flock.RecVStar.statements r'.shape).toOption.bind (·[s]?) = some (.recOpen lv' LANES HH n') := by
    rw [hss']
    exact hsv'
  have hsvC : (_root_.Flock.RecVStar.statements r'.shape).toOption.bind (·[s']?) = some (.cap q.level N') := by
    rw [hss']
    exact hs'
  have hrow := out_top_rows elem hJ.2.1 hJ.2.2.1 hr' hsvO hcc hqo hi hw hsvC hc' hqc hlt hw' hpfx hcm
  rw [hcopy, hrow]
  rfl

/-- **The level part's `out` slot at the copy's opening** (`OutTopTie` past its ports): at a level part's piece and its
witness `ws[s]`, its circuit and `out` port, and a RecCap part of the piece's level holding the row the opening
climbs to (`capIdx`), with its circuit and `top` port, some opening `i < n` of the `out` port is the copy's (its
`outRowForms` row's node is the copy's `out` columns), and its slot and RecCap's `top` slot there carry one commit
string. -/
def OutSlotTie (pieces : Msg → List OpenPiece) (posAt : Msg → List Coin → OpenPiece → ℕ) : Prop :=
  AtPiecesP H D shape untr b cmt L pieces fun x c ws LANES HH s _ w p ht q =>
    ∀ (r : XR) (ss : Array VStar.Stmt), readX x = some r →
      _root_.Flock.RecVStar.statements r.shape = .ok ss →
    ∀ (lv n : ℕ) (cc : Circuit) (qo : ℕ), ss[s]? = some (.recOpen lv LANES HH n) →
      r.w.parts[s]?.bind partCircuit = some cc → cc.ports.findIdx? (·.name == "out") = some qo →
    ∀ (s' N' : ℕ) (c' : Circuit) (w' : Claims.Flock.Wit) (qc : ℕ), ss[s']? = some (.cap q.level N') →
      r.w.parts[s']?.bind partCircuit = some c' → ws[s']? = some w' → capPort c' = some qc →
      capIdx (Claims.Recursion.xI D x) q
          (posAt (Claims.Recursion.xI D x) ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) q) HH < N' →
      ∃ i < n,
        readForms w (hmCommit cc qo i) = readForms w' (hmCommit c' qc (capIdx (Claims.Recursion.xI D x) q
          (posAt (Claims.Recursion.xI D x) ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) q) HH)) ∧
        bytesAt (copyR (Claims.Recursion.xI D x) w p ht q) (outCol LANES HH) 64 =
          nodeOf ⟨(readForms w (DeployedR.outRowForms cc qo i)).toArray⟩

/-- **The two slots' ports, from V\*'s parse check**: both parts are staged (`StagedAt`, and `JointP` at `s'`), so their
circuits have V\*'s row ports (`rows_staged`): the level part's `out` and RecCap's `top` are 64-word rows
(`out_port_recOpen`, `top_port_cap`), with one row prefix. The rest is `OutSlotTie`. -/
theorem outTopTie_of (pieces : Msg → List OpenPiece) (posAt : Msg → List Coin → OpenPiece → ℕ)
    (hT : OutSlotTie H D shape untr b cmt L Bd pieces posAt) : OutTopTie H D shape untr b cmt L Bd pieces posAt := by
  intro x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO w p hw hst q hq ho ht r' ss' hr' hss' s' N' c' w'
    hs' hc' hw' hlt
  have e1 : r' = r := Option.some.inj (hr'.symm.trans hr)
  rw [e1] at hss' hc' ⊢
  have e2 : ss' = ss := Except.ok.inj (hss'.symm.trans hss)
  rw [e2] at hs' ⊢
  have hSP := hst.2.2.2.2.2.2.2.2
  obtain ⟨cc, hcc⟩ := staged_circuit hSP hr
  obtain ⟨u, hu, hrows⟩ := rows_staged hSP hr hss hsv hcc
  obtain ⟨qo, hqo, hqo', hwo⟩ := out_port_recOpen hu hrows
  obtain ⟨hsl, -⟩ := List.getElem?_eq_some_iff.1 hw'
  obtain ⟨-, xO', -, -, -, hsh, -⟩ := hJ.2.2.2 s' hsl
  obtain ⟨u', hu', hrows'⟩ := rows_staged hsh.2 hr hss hs' hc'
  obtain ⟨qc, hqc, hqc', hwc⟩ := top_port_cap hu' hrows'
  obtain ⟨i, hi, hcm, hcopy⟩ :=
    hT x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO w p hw hst q hq ho ht r ss hr hss lv n cc qo hsv hcc hqo s' N' c' w' qc hs' hc' hw' hqc hlt
  exact ⟨lv, n, cc, qo, i, qc, hsv, hcc, hqo, hi, hqc,
    (rowPfx_of_words hqo' hwo).trans (rowPfx_of_words hqc' hwc).symm, hcm, hcopy⟩

/-- **The binding at the layout of record, past the ports**: `outBindRP_of` from `OutSlotTie` (`outTopTie_of`). -/
theorem outBindRP_of_slot (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (pieces : Msg → List OpenPiece) (posAt : Msg → List Coin → OpenPiece → ℕ)
    (hT : OutSlotTie FlockDeploy.sha512 D shape untr b cmt (DeployedR.layR elem) Bd pieces posAt) :
    OutBindRP FlockDeploy.sha512 D shape untr b cmt (DeployedR.layR elem) Bd pieces posAt :=
  outBindRP_of D shape untr b cmt Bd elem pieces posAt
    (outTopTie_of FlockDeploy.sha512 D shape untr b cmt (DeployedR.layR elem) Bd pieces posAt hT)

end Deployed

end RecDeploy.CapLower
