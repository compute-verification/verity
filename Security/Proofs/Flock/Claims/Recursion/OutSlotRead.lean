import Proofs.Flock.Claims.Recursion.VerdictRowEBytes
import Proofs.Flock.Claims.Recursion.CapLowerPin
import Proofs.Flock.Claims.Recursion.ResLowerDeployed

/-!
# A level part's copy's `out` columns are its instance's `out` row (`OutSlotTie`'s column read)

`OutSlotTie` asks, at a level part's piece `q`, for an opening `i < n` of the part's `out` port whose row's node is
what the copy (`copyAtR`, instance `N = rep · q_l + query`) holds at `Family.outCol L H`, and whose slot carries
RecCap's `top` commit string at `capIdx`. The instance is the copy's own, `N`:

* `colsHold_out`: at a circuit `RecOpen.check` accepts at `rec-open/l<L>-h<H>`, `out` is output row 0 (`nIn = 5`, its
  rows' names), at the copy's columns from `Family.outCol L H` (`colsHold_outRow` at row 0);
* `outBytes_staged`: at a staged level part, its output rows one port each (`staged_hidden`, `outRowPorts_hidden`),
  the copy's 64 bytes at `outCol` are the node of instance `m`'s `out` row (`bytesAt_node`, `bytesAt_of_cols`);
* `outSlotTie_of`: `OutSlotTie` from the two facts the column read doesn't give, `OutIdxLt` (`N < n`) and
  `OutCommitTie` (the two slots' commit strings at `N` and `capIdx`), and `outBindRP_of_leaves` the binding at
  `layR` from them.

`OutCommitTie` reads across two parts' setups: the level's `out` reads `rec-top-L<l>` at each opening's node
(`RecCoins.checkLink`) and RecCap's `top` reads it at every node in order (`RecCoins.checkCap`), so the two slots are
openings of one registration at one position, which bind only up to a collision among the setups' own evaluations
(`setupCalls`), outside `JointP`'s escape.
-/

namespace RecDeploy

open Flock
open Claims (Msg)
open Claims.Recursion.VStarB
open FlockVBridge (ColsHold)
open RecDeploy.ResLower (staged_hidden)

/-- **A rec-open copy's `out` columns hold its unit's `out` row**: at a circuit `RecOpen.check` accepts at
`rec-open/l<L>-h<H>`, of V*'s seven rows, one output port per output row, `out` is output row 0 (`colsHold_accOut`'s
argument at row 0), at the copy's columns from `Family.outCol L H`. -/
theorem colsHold_out {p : Flock.Readable.Parsed} (h1 : p.sts.size = 1) (h0 : 0 < p.sts.size) {c : Circuit}
    (hc0 : p.sts[0].c = c) {tags : Flock.Tags} (P : FlockSoundness.ParseFacts tags c) (ho : c.outNet = c.unitNet)
    {LANES HH : ℕ} (hn : c.unit.name = Flock.RecOpen.unitName LANES HH) (hsz : c.ports.size = 7)
    (hr : c.outRowPorts = Array.replicate c.outPorts.size 1) {qo : ℕ}
    (hqo : c.ports.findIdx? (·.name == "out") = some qo) (w : Claims.Flock.Wit) (n : ℕ) :
    ColsHold (Flock.Readable.copyOf p 0 h0 w (n / c.g) (n % c.g) 0) (FlockVBridge.Family.outCol LANES HH)
        ⟨(readForms w (accOutForms c qo n)).toArray⟩ ∧ (readForms w (accOutForms c qo n)).length = 128 := by
  have hrec := P.recNet
  have hp := FlockVBridge.params_of_check hrec hn
  obtain ⟨ls, hls⟩ := P.check
  have C := FlockSoundness.check_facts P.typed hls
  have hlo0 : 0 < c.leavesOut.size := by
    rw [(FlockSoundness.Discharge.Layout.check_out P.typed ho hls).2.1]
    exact FlockSoundness.Discharge.Hm.upv_pos C ho
  have hlaid := FlockC3.RegLevels.outsLaid_of_outsOk (Flock.RecOpen.outsOk_of_check hrec hp) hlo0
  have hU := FlockSoundness.unit_built P C
  obtain ⟨hB, -, hE⟩ := FlockC3.RegLevels.outCols_of_laid (hU := hU) (Flock.RecOpen.check_ok hrec hp).2
  have hB' : (c.unit.outGroups.getD 0 default).col * Flock.WORD = FlockVBridge.Family.outCol LANES HH := hB
  rw [FlockVBridge.foldl_wordPorts, show (FlockSoundness.Rows.ofNet hU).nIn + (FlockSoundness.Rows.ofNet hU).nComp =
    c.unit.constPos from FlockSoundness.Rows.ofNet_one hU] at hE
  have hE' : FlockVBridge.Family.outCol LANES HH + 2048 ≤ c.unit.constPos := by omega
  obtain ⟨hrows, -⟩ := FlockC3.RegLevels.check_rows hrec (.inl hp)
  have gd : ∀ k (hk : k < c.ports.size), c.ports.getD k default = c.ports[k] := fun k hk => by
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hk, Option.getD_some]
  have hnm : ∀ k < 5, (c.ports.getD k default).name ≠ "out" := by
    intro k hk
    obtain ⟨hk', hnk, -, -⟩ := FlockC3.RegLevels.rowsOk_get hrows hk
    rw [gd k hk', hnk]
    rcases (by omega : k = 0 ∨ k = 1 ∨ k = 2 ∨ k = 3 ∨ k = 4) with rfl | rfl | rfl | rfl | rfl
    · rw [show ((Flock.RecOpen.rows LANES HH)[0]!).1 = "row" from rfl]; decide
    · rw [show ((Flock.RecOpen.rows LANES HH)[1]!).1 = "aux" from rfl]; decide
    · rw [show ((Flock.RecOpen.rows LANES HH)[2]!).1 = "acc" from rfl]; decide
    · rw [show ((Flock.RecOpen.rows LANES HH)[3]!).1 = "coef" from rfl]; decide
    · rw [show ((Flock.RecOpen.rows LANES HH)[4]!).1 = "dirs" from rfl]; decide
  obtain ⟨-, -, hq, hno, -, -, -⟩ := id hlaid
  have hn5 : HmOut.nIn c = 5 := by
    rcases Nat.lt_or_ge (HmOut.nIn c) 5 with h | h
    · have e : (c.ports.getD (HmOut.nIn c) default).name = "out" := by
        rw [gd (HmOut.nIn c) (by omega)]; exact hno
      exact absurd e (hnm _ h)
    · omega
  have hq5 : qo = HmOut.nIn c := by
    obtain ⟨hqs, hqn, hlt⟩ := Array.findIdx?_eq_some_iff_getElem.1 hqo
    have hqn' : (c.ports.getD qo default).name = "out" := by rw [gd qo hqs]; simpa using hqn
    have hle : qo ≤ HmOut.nIn c := by
      by_contra hgt
      exact hlt (HmOut.nIn c) (by omega) (by simp [hno])
    rcases Nat.lt_or_ge qo 5 with h | h
    · exact absurd hqn' (hnm qo h)
    · omega
  subst hq5
  have := colsHold_outRow h1 h0 hc0 P ho hlaid hr hB' hE' w n (jo := 0) (by omega)
  simp only [Nat.mul_zero, Nat.add_zero] at this
  exact this

/-- The first 64 of 128 bytes a copy holds are their node (`nodeOf`). -/
theorem bytesAt_node (f : ℕ → Bool) (base : ℕ) :
    FlockVBridge.Family.bytesAt f base 64 = FlockDeploy.nodeOf (FlockVBridge.Family.bytesAt f base 128) := by
  unfold FlockDeploy.nodeOf FlockVBridge.Family.bytesAt
  simp only [List.size_toArray, List.length_map, List.length_range]
  rw [← List.map_take, List.take_range]
  rfl

/-- **The copy's `out` columns at a staged level part**: at a staged `.recOpen lv L H n` part with circuit `c`, whose
output rows are one port each (hidden outputs, `outRowPorts_hidden`), copy `m` of its one table holds at
`Family.outCol L H` the node of instance `m`'s `out` row. -/
theorem outBytes_staged {st : Stage} {s : ℕ} {x xO : Msg} (hS : StagedPart st s x xO) {p : Flock.Readable.Parsed}
    (hx : Flock.Readable.parseBytes readOfO xO = some p) (h0 : 0 < p.sts.size) {r : XR} (hr : readX x = some r)
    {ss : Array VStar.Stmt} (hss : Flock.RecVStar.statements r.shape = .ok ss) {lv LANES HH n : ℕ}
    (hsv : ss[s]? = some (.recOpen lv LANES HH n)) {c : Circuit} (hc : r.w.parts[s]?.bind partCircuit = some c)
    {qo : ℕ} (hqo : c.ports.findIdx? (·.name == "out") = some qo) (w : Claims.Flock.Wit) (m : ℕ) :
    FlockVBridge.Family.bytesAt (Flock.Readable.copyOf p 0 h0 w (m / p.sts[0]!.c.g) (m % p.sts[0]!.c.g) 0)
        (FlockVBridge.Family.outCol LANES HH) 64 =
      FlockDeploy.nodeOf ⟨(readForms w (DeployedR.outRowForms c qo m)).toArray⟩ := by
  obtain ⟨h1, hc0, ⟨tags, P⟩, hn, ho, h7, hv1, ht, hpi⟩ := stagedRecOpen hS hx h0 hr hss hsv hc
  have hh := staged_hidden hS hx h0
  have hor := Flock.Readable.outRowPorts_hidden hx ht hpi hh h0 (by rw [hc0]; exact hv1)
  rw [hc0] at hor
  have hg : p.sts[0]!.c = c := by rw [getElem!_pos p.sts 0 h0]; exact hc0
  rw [hg]
  obtain ⟨hcs, hlen⟩ := colsHold_out h1 h0 hc0 P ho hn h7 hor hqo w m
  have hb := FlockVBridge.Family.bytesAt_of_cols hcs
  have hsz : (⟨(readForms w (accOutForms c qo m)).toArray⟩ : ByteArray).size = 128 := by
    show (readForms w (accOutForms c qo m)).toArray.size = 128
    rw [List.size_toArray, hlen]
  rw [hsz] at hb
  rw [bytesAt_node, hb]
  rfl

end RecDeploy

namespace RecDeploy.CapLower

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP nodeOf)
open Claims.Recursion.VStarB
open FlockVBridge.Family (bytesAt outCol)
open RecDeploy.OpenLower (AtPiecesP copyR qAt schAt PiecesInRangeR ownQR_some levelMatch)

section Deployed

variable (H : Msg → Msg) (D : Claims.Recursion.Deploy Coin) (shape : Claims.Recursion.Form → Prop)
  (untr : TrReader) (b : Brings) (cmt : Msg → ℕ → Msg → Msg) (L : LayoutB)
  (Bd : Claims.Recursion.Bound Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace)

/-- **The piece's copy is one of the level statement's instances**: at a level part's piece, its instance
`rep · q_l + query` (`copyAtR`'s) is below the statement's `n`. -/
def OutIdxLt (pieces : Msg → List OpenPiece) : Prop :=
  AtPiecesP H D shape untr b cmt L pieces fun x _ _ LANES HH s _ _ _ _ q =>
    ∀ (r : XR) (ss : Array VStar.Stmt), readX x = some r → _root_.Flock.RecVStar.statements r.shape = .ok ss →
    ∀ (lv n : ℕ), ss[s]? = some (.recOpen lv LANES HH n) →
      q.rep * qAt (Claims.Recursion.xI D x) q.level + q.query < n

/-- **The two slots carry one commit string** (`OutSlotTie`'s tie, at the copy's instance): the level part's `out`
slot at the piece's instance `rep · q_l + query` and RecCap's `top` slot at `capIdx`. -/
def OutCommitTie (pieces : Msg → List OpenPiece) (posAt : Msg → List Coin → OpenPiece → ℕ) : Prop :=
  AtPiecesP H D shape untr b cmt L pieces fun x c ws LANES HH s _ w _ _ q =>
    ∀ (r : XR) (ss : Array VStar.Stmt), readX x = some r →
      _root_.Flock.RecVStar.statements r.shape = .ok ss →
    ∀ (lv n : ℕ) (cc : Circuit) (qo : ℕ), ss[s]? = some (.recOpen lv LANES HH n) →
      r.w.parts[s]?.bind partCircuit = some cc → cc.ports.findIdx? (·.name == "out") = some qo →
    ∀ (s' N' : ℕ) (c' : Circuit) (w' : Claims.Flock.Wit) (qc : ℕ), ss[s']? = some (.cap q.level N') →
      r.w.parts[s']?.bind partCircuit = some c' → ws[s']? = some w' → capPort c' = some qc →
      capIdx (Claims.Recursion.xI D x) q
          (posAt (Claims.Recursion.xI D x) ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) q) HH < N' →
        readForms w (hmCommit cc qo (q.rep * qAt (Claims.Recursion.xI D x) q.level + q.query)) =
          readForms w' (hmCommit c' qc (capIdx (Claims.Recursion.xI D x) q
            (posAt (Claims.Recursion.xI D x) ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) q) HH))

/-- **`OutSlotTie` from its tie**: the opening is the copy's own instance, below `n` (`OutIdxLt`), whose slot carries
RecCap's commit string (`OutCommitTie`); the copy's `out` columns are its row's node at every staged part
(`outBytes_staged`). -/
theorem outSlotTie_of (pieces : Msg → List OpenPiece) (posAt : Msg → List Coin → OpenPiece → ℕ)
    (hI : OutIdxLt H D shape untr b cmt L pieces) (hC : OutCommitTie H D shape untr b cmt L Bd pieces posAt) :
    OutSlotTie H D shape untr b cmt L Bd pieces posAt := by
  intro x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO w p hw hst q hq ho ht r' ss' hr' hss' lv' n' cc qo hsv'
    hcc hqo s' N' c' w' qc hs' hc' hw' hqc hlt
  refine ⟨_,
    hI x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO w p hw hst q hq ho ht r' ss' hr' hss' lv' n' hsv',
    hC x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO w p hw hst q hq ho ht r' ss' hr' hss' lv' n' cc qo hsv' hcc
      hqo s' N' c' w' qc hs' hc' hw' hqc hlt, ?_⟩
  exact outBytes_staged hst.2.2.2.2.2.2.2.2 hst.1 ht hr' hss' hsv' hcc hqo w _

/-- **`OutIdxLt` from the pieces' range**: a piece is of rep 0 or 1 and one of its level's `q_l` queries
(`PiecesInRangeR`), and its level statement has `n = 2 q_l` (`levelMatch`), as in `copyFitsR_of`. -/
theorem outIdxLt_of (pieces : Msg → List OpenPiece) (hP : ∀ x, D.parse x = parse x)
    (hS : ∀ f, shape f → Claims.Recursion.Form.Dense f) (hR : PiecesInRangeR pieces) :
    OutIdxLt H D shape untr b cmt L pieces := by
  intro x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO w p hw hst q hq ho ht r' ss' hr' hss' lv' n' hsv'
  obtain ⟨hrep, hqry⟩ := hR _ q hq
  have hx' : parse x = some st := (hP x).symm.trans hx
  have hxI : Claims.Recursion.xI D x = xIOf x := by
    unfold Claims.Recursion.xI xIOf
    rw [hP x]
  obtain ⟨h0, r'', ss'', hr'', hss'', L', H', n'', hsv''⟩ := ownQR_some ho
  rw [hr'] at hr''
  cases hr''
  rw [hss'] at hss''
  cases hss''
  rw [hsv'] at hsv''
  cases hsv''
  obtain ⟨hls, -, -, hn⟩ := levelMatch hx' (hS _ hf) hr' hss' hsv'
  subst hls
  rw [h0, hxI] at hqry
  simp only [qAt]
  rw [hxI]
  generalize ((schAt (xIOf x) 0).levels[q.level]!).q = Q at hqry hn ⊢
  have h1 : q.rep * Q ≤ 1 * Q := Nat.mul_le_mul_right _ (by omega)
  omega

/-- **The binding at the layout of record, from the tie**: `outBindRP_of_slot` off `outSlotTie_of`. -/
theorem outBindRP_of_leaves (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (pieces : Msg → List OpenPiece) (posAt : Msg → List Coin → OpenPiece → ℕ)
    (hI : OutIdxLt FlockDeploy.sha512 D shape untr b cmt (DeployedR.layR elem) pieces)
    (hC : OutCommitTie FlockDeploy.sha512 D shape untr b cmt (DeployedR.layR elem) Bd pieces posAt) :
    OutBindRP FlockDeploy.sha512 D shape untr b cmt (DeployedR.layR elem) Bd pieces posAt :=
  outBindRP_of_slot D shape untr b cmt Bd elem pieces posAt
    (outSlotTie_of FlockDeploy.sha512 D shape untr b cmt (DeployedR.layR elem) Bd pieces posAt hI hC)

/-- **The binding at the layout of record, from the commit tie alone**, at pieces in range (row D's are,
`piecesInRangeR_rowDPieces`). -/
theorem outBindRP_of_tie (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (pieces : Msg → List OpenPiece) (posAt : Msg → List Coin → OpenPiece → ℕ)
    (hP : ∀ x, D.parse x = parse x) (hS : ∀ f, shape f → Claims.Recursion.Form.Dense f)
    (hR : PiecesInRangeR pieces)
    (hC : OutCommitTie FlockDeploy.sha512 D shape untr b cmt (DeployedR.layR elem) Bd pieces posAt) :
    OutBindRP FlockDeploy.sha512 D shape untr b cmt (DeployedR.layR elem) Bd pieces posAt :=
  outBindRP_of_leaves D shape untr b cmt Bd elem pieces posAt
    (outIdxLt_of FlockDeploy.sha512 D shape untr b cmt (DeployedR.layR elem) pieces hP hS hR) hC

end Deployed

end RecDeploy.CapLower
