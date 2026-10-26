import Proofs.Flock.Claims.Recursion.CapLower
import Proofs.Flock.Claims.Recursion.RestFinal

/-!
# The reader's cap at level 0 (`SideCapR`), at the deployed `finalR` and `DeployedR.layR`

`SideCapR` asks that the side `sideAtR rd` read at the opened messages have, at a piece's opening, the node a RecCap
row carries (`CapEntryR`). At level 0 the side's cap is `capR`: the nodes of the first opened message's rows
(`rowsOf`, `nodeOf`), the table's level-0 top (`repCtx`'s `cap0`). That message is round 0's rows joined
(`Bound.opened`, `joinR`), and at the deployed last message round 0 is RecCap's `top` rows:

* the last message off the witnesses starts with round 0's openings, at RecCap's part over its `N = level0Nodes`
  instances, each its row then its salt (`lastAt_parts`, `openPieces`, `capOpen`);
* `finalR` reads the trailer off the end and cuts the rest at `openLens` (`finalR_trailer`), so its first round is
  those rows, as RecCap's part's witness reads them (`cutRound_openSegs`, `cutRounds_head`);
* so the side's level-0 cap entry `i` is the node of RecCap's `top` row at instance `i` (`capRow`), for `i < N`
  (`capEntry_level0`). RecCap's part has a circuit, the one `readX` checked for its port `top` (`capPorts`).

`sideCapR_of` gives `SideCapR` at `layR elem` and the deployed bound record at `finalR` from two premises: the opened
position at level 0 is below the cap's `N` nodes, at one of the table's two reps (`PosCap0`), and the levels above
0 (`SideCapUp`), whose caps are the stream rounds' tops (`topGroups` of the `capOpenAt` openings).
-/

namespace RecDeploy.CapLower

open Flock
open Claims (Msg Out)
open FlockDeploy (Coin FinalP nodeOf)
open Claims.Recursion.VStarB
open RecDeploy.OpenLower (AtPieces AtOpened sideAtR capR ligR ownQR)
open RecDeploy.RestRead (DenseX)

/-! ## Round 0's cut -/

/-- **`cutRound` reads a cap's openings back**: at the openings' row lengths, the bytes read off the witness `w`
are the rows, each then its salt, and what follows is left. -/
theorem cutRound_openSegs (w : Claims.Flock.Wit) (c : Circuit) (nm : String) (q : ℕ) :
    ∀ (ns : List ℕ) (more : Msg), ∃ ps : List (Msg × Salt),
      DeployedR.cutRound (DeployedR.rowLens (ns.flatMap (openSegs c nm q)))
          (readSegs w (ns.flatMap (openSegs c nm q)) ++ more) = some (ps, more) ∧
        ps.map Prod.fst = ns.map fun n => readForms w (((rowHidden c nm n).getD []).map HmNets.Lin.var)
  | [], more => ⟨[], rfl, rfl⟩
  | n :: ns, more => by
    obtain ⟨ps, hc, hp⟩ := cutRound_openSegs w c nm q ns more
    have hR : (readForms w (((rowHidden c nm n).getD []).map HmNets.Lin.var)).length =
        pieceLen [(([] : Msg), ((rowHidden c nm n).getD []).map HmNets.Lin.var)] := by
      rw [readForms_length]
      simp only [pieceLen, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, List.length_nil, Nat.zero_add,
        Nat.add_zero]
    have hS : (readForms w (hmSalt c q n)).length = Hm96.sha512.saltBytes := by
      rw [← RestRead.pieceLen_salt c q n, readForms_length]
      simp only [pieceLen, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, List.length_nil, Nat.zero_add,
        Nat.add_zero]
    have e : (n :: ns).flatMap (openSegs c nm q) =
        (([] : Msg), ((rowHidden c nm n).getD []).map HmNets.Lin.var) :: (([] : Msg), hmSalt c q n) ::
          ns.flatMap (openSegs c nm q) := rfl
    have em : readSegs w ((n :: ns).flatMap (openSegs c nm q)) ++ more =
        readForms w (((rowHidden c nm n).getD []).map HmNets.Lin.var) ++
          (readForms w (hmSalt c q n) ++ (readSegs w (ns.flatMap (openSegs c nm q)) ++ more)) := by
      rw [e]
      simp only [readSegs, List.flatMap_cons, List.nil_append, List.append_assoc]
    have er : DeployedR.rowLens ((n :: ns).flatMap (openSegs c nm q)) =
        pieceLen [(([] : Msg), ((rowHidden c nm n).getD []).map HmNets.Lin.var)] ::
          DeployedR.rowLens (ns.flatMap (openSegs c nm q)) := rfl
    have hle : pieceLen [(([] : Msg), ((rowHidden c nm n).getD []).map HmNets.Lin.var)] + Hm96.sha512.saltBytes ≤
        (readForms w (((rowHidden c nm n).getD []).map HmNets.Lin.var) ++
          (readForms w (hmSalt c q n) ++ (readSegs w (ns.flatMap (openSegs c nm q)) ++ more))).length := by
      simp only [List.length_append]
      omega
    refine ⟨(readForms w (((rowHidden c nm n).getD []).map HmNets.Lin.var),
      DeployedR.saltOf (readForms w (hmSalt c q n))) :: ps, ?_, ?_⟩
    · rw [er, em]
      simp only [DeployedR.cutRound, if_pos hle, List.drop_left' hR, List.drop_left' hS, hc, List.take_left' hR,
        List.take_left' hS, Option.map_some]
    · rw [List.map_cons, hp, List.map_cons]

/-- **The first round `cutRounds` reads is its first cut.** -/
theorem cutRounds_head {l : List ℕ} {ls : List (List ℕ)} {m : Msg} {ps : List (Msg × Salt)} {rest : Msg}
    {os : List (List (Msg × Salt))} {last : Msg} (h1 : DeployedR.cutRound l m = some (ps, rest))
    (h : DeployedR.cutRounds (l :: ls) m = some (os, last)) : ∃ os', os = ps :: os' := by
  simp only [DeployedR.cutRounds, h1, Option.bind_some] at h
  obtain ⟨⟨os', l'⟩, -, he⟩ := Option.map_eq_some_iff.1 h
  exact ⟨os', (Prod.mk.inj he).1.symm⟩

/-! ## The level-0 cap -/

/-- A layout's messages don't read its calls. -/
theorem msgs_shaAt (L : LayoutB) (f : Msg → ℕ → List (Msg × List HmNets.Lin × List HmNets.Lin)) :
    ({ L with shaAt := f } : LayoutB).msgs = L.msgs := rfl

/-- The layout of record's messages are `layoutBTops`' at the last round's pieces: `layR` adds only calls. -/
theorem msgs_layR (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) :
    (DeployedR.layR elem).msgs = (RestRead.layAt (DeployedR.restR elem)).msgs :=
  msgs_shaAt (RestRead.layAt (DeployedR.restR elem)) _

/-- **The side's level-0 cap entry is RecCap's `top` row's node**, at every reading of the deployed last message: at
a dense compiled statement and witnesses one a part, `finalR`'s first round is RecCap's part's `top` rows over its
`N = level0Nodes` instances (`cutRound_openSegs`), so the side's cap (`capR`, table 0's chunk of their nodes) has
at each opened node `i < N` the node of RecCap's row `i`, at the list's last statement `.cap 0 N`. -/
theorem capEntry_level0 (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (rd : Msg → Option FinalP) (posAt : Msg → List Coin → OpenPiece → ℕ) {x : Msg} (hx : DenseX x)
    {ws : List Claims.Flock.Wit} (hl : ws.length = nS x) {cs : List Coin} {HH : ℕ} {q : OpenPiece}
    (h0 : q.level = 0) (ht : q.table = 0) (hrep : q.rep < 2)
    (hpos : ∀ (r : XR) (i N : ℕ), readX x = some r → capAt r = some (i, N) → posAt (xIOf x) cs q >>> HH < N)
    {os : List (List (Msg × Salt))} {last : Msg}
    (hf : DeployedR.finalR (RestRead.lastAt (DeployedR.restR elem) x ws) = some (os, last)) :
    CapEntryR rd posAt x (xIOf x) cs ws HH q (os.map (fun o => FlockDeploy.joinR (o.map Prod.fst)) ++ [last]) := by
  obtain ⟨r, rs, s0, spec, st, hr, hrs, hset, hm, -, -⟩ := RestRead.schedBridge hx
  obtain ⟨r', hr', hlast, -⟩ := RestRead.lastAt_parts elem hx hl
  rw [hr] at hr'
  have e : r = r' := Option.some.inj hr'
  subst e
  have hn : nS x = r.w.parts.length := by simp [nS, hr]
  have hc0 := RestRead.capAt_level0 hr hm
  obtain ⟨vs, hvs, hvl⟩ := readX_statements hr
  have hb := RestRead.statements_cap hvs
  obtain ⟨-, -, hp0⟩ := capAt_readX hr
  obtain ⟨st0, hst0, hd0⟩ := hx
  obtain ⟨c', hc', -⟩ := capPorts Claims.Recursion.Form.Dense x st0 hst0 hd0 r hr
  have hlt : nS x - 1 < ws.length := by omega
  obtain ⟨w', hw⟩ : ∃ w', ws[nS x - 1]? = some w' := ⟨_, List.getElem?_eq_getElem hlt⟩
  have hcc : (r.w.parts[nS x - 1]?.bind partCircuit).getD default = c' := by rw [hc', Option.getD_some]
  have hop : openPieces r = (nS x - 1, capOpen r (nS x - 1) (List.range (level0Nodes st))) ::
      (List.range (2 * rs.size)).flatMap (roundOpen r rs) := by
    simp [openPieces, hc0, hrs]
  have hol : DeployedR.openLens r = DeployedR.rowLens (capOpen r (nS x - 1) (List.range (level0Nodes st))) ::
      (List.range (2 * rs.size)).map fun j => (roundOpen r rs j).flatMap fun p => DeployedR.rowLens p.2 := by
    simp [DeployedR.openLens, hc0, hrs]
  have hco : capOpen r (nS x - 1) (List.range (level0Nodes st)) =
      (List.range (level0Nodes st)).flatMap (openSegs c' "top" ((capPort c').getD 0)) := by
    simp only [capOpen, hcc]
  have hrp : DeployedR.readPieces (openPieces r) ws =
      readSegs w' (capOpen r (nS x - 1) (List.range (level0Nodes st))) ++
        DeployedR.readPieces ((List.range (2 * rs.size)).flatMap (roundOpen r rs)) ws := by
    rw [hop]
    simp only [DeployedR.readPieces, List.flatMap_cons, hw, Option.map_some, Option.getD_some]
  rw [hlast, DeployedR.finalR_trailer, hol, hrp, hco, List.append_assoc] at hf
  obtain ⟨ps, hps, hpf⟩ := cutRound_openSegs w' c' "top" ((capPort c').getD 0) (List.range (level0Nodes st))
    (DeployedR.readPieces ((List.range (2 * rs.size)).flatMap (roundOpen r rs)) ws ++
      DeployedR.readPieces ((elem r).map fun p => (p.1.val, p.2)) ws)
  obtain ⟨os', rfl⟩ := cutRounds_head hps hf
  have hi := hpos r _ _ hr hc0
  have hcr : ((List.range (level0Nodes st)).map fun n =>
      readForms w' (((rowHidden c' "top" n).getD []).map HmNets.Lin.var)).map (fun r => (⟨r.toArray⟩ : ByteArray)) =
      (List.range (level0Nodes st)).map (capRow c' w') :=
    List.map_map.trans (List.map_congr_left fun _ _ => rfl)
  have hhd : (((ps :: os').map fun o => FlockDeploy.joinR (o.map Prod.fst)) ++ [last]).headD [] =
      FlockDeploy.joinR (ps.map Prod.fst) := rfl
  have hrows : (FlockDeploy.rowsOf ((((ps :: os').map fun o => FlockDeploy.joinR (o.map Prod.fst)) ++
      [last]).headD [])).map (List.map nodeOf) = some (((List.range (level0Nodes st)).map (capRow c' w')).map nodeOf) := by
    rw [hhd, hpf, FlockDeploy.rowsOf_joinR', hcr, Option.map_some]
  have hcap : capR (xIOf x) (((ps :: os').map fun o => FlockDeploy.joinR (o.map Prod.fst)) ++ [last]) q =
      (((List.range (level0Nodes st)).map (capRow c' w')).map nodeOf).toArray := by
    unfold capR
    rw [hset, hrows]
    show ((FlockDeploy.cutLens (FlockDeploy.level0Lens #[st]) _).getD ((2 * q.table + q.rep) / 2) []).toArray = _
    rw [ht, show (2 * 0 + q.rep) / 2 = 0 by omega]
    show ((FlockDeploy.cutLens [level0Nodes st] _).getD 0 []).toArray = _
    simp only [FlockDeploy.cutLens, List.getD_cons_zero]
    rw [List.take_of_length_le (l := ((List.range (level0Nodes st)).map (capRow c' w')).map nodeOf) (by simp)]
  refine ⟨r, vs, hr, hvs, nS x - 1, level0Nodes st, c', w', ?_, hc', hw, ?_, ?_⟩
  · rw [h0, show nS x - 1 = vs.size - 1 by omega, level0Nodes, hm]
    exact hb
  · rw [capIdx, capFirst, if_pos h0, Nat.zero_add]
    exact hi
  · rw [capIdx, capFirst, if_pos h0, Nat.zero_add]
    show (FlockSoundness.Refine.capAt (capR (xIOf x) _ q) (ligR rd (xIOf x) _ q) q.level).getD _ ByteArray.empty = _
    rw [FlockSoundness.Refine.capAt, if_pos h0, hcap, Array.getD_eq_getD_getElem?, List.getElem?_toArray,
      List.getElem?_map, List.getElem?_map, List.getElem?_range hi, Option.map_some, Option.map_some,
      Option.getD_some]

/-! ## `SideCapR` from level 0 and the levels above -/

section Deployed

variable (H : Msg → Msg) (D : Claims.Recursion.Deploy Coin) (shape : Claims.Recursion.Form → Prop)
  (untr : TrReader) (b : Brings) (cmt : Msg → ℕ → Msg → Msg) (L : LayoutB)
  (Bd : Claims.Recursion.Bound Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace) (rd : Msg → Option FinalP)

/-- **The opened position at level 0 is one of the cap's nodes**: at a level-0 piece, one of its table's two reps,
the opening's node (`posAt`, below `HH` siblings) is below RecCap's node count `N`. -/
def PosCap0 (pieces : Msg → List OpenPiece) (posAt : Msg → List Coin → OpenPiece → ℕ) : Prop :=
  AtPieces H D shape untr b cmt L pieces fun x c _ _ HH _ _ _ _ _ q => q.level = 0 →
    q.rep < 2 ∧ ∀ (r : XR) (i N : ℕ), readX x = some r → capAt r = some (i, N) →
      posAt (Claims.Recursion.xI D x) ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) q >>> HH < N

/-- **The reader above level 0**: `SideCapR` at the pieces of the levels whose caps are the stream rounds' tops. -/
def SideCapUp (pieces : Msg → List OpenPiece) (posAt : Msg → List Coin → OpenPiece → ℕ) : Prop :=
  AtPieces H D shape untr b cmt L pieces fun x c ws _ HH _ _ _ _ _ q => q.level ≠ 0 →
    AtOpened D Bd x (L.msgs x ws) (CapEntryR rd posAt x (Claims.Recursion.xI D x)
      ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) ws HH q)

/-- **The reader at the deployed layout and last message**, from level 0's position range and the levels above:
at `layR elem` and the bound record at `finalR` with rows joined by `joinR`, a level-0 piece's cap entry is RecCap's
`top` row's node (`capEntry_level0`); a piece is table 0's (`ownQR`), and the record reads the statement as the
deployed reader does, with one witness a part. -/
theorem sideCapR_of (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (nextE : Msg → List Msg → List Coin → Claims.OC Out) (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128))
    (wE : Msg → ℝ) (rowIn : Msg → Msg) (w : Msg → ℝ)
    (pieces : Msg → List OpenPiece) (posAt : Msg → List Coin → OpenPiece → ℕ)
    (hP : ∀ x, D.parse x = parse x) (hN : ∀ x, D.nS x = nS x) (hD : ∀ f, shape f → Claims.Recursion.Form.Dense f)
    (h0 : PosCap0 H D shape untr b cmt (DeployedR.layR elem)
      (bdDeployed rel nextE colsE wE FlockDeploy.joinR rowIn DeployedR.finalR w) pieces posAt)
    (hup : SideCapUp H D shape untr b cmt (DeployedR.layR elem)
      (bdDeployed rel nextE colsE wE FlockDeploy.joinR rowIn DeployedR.finalR w) rd pieces posAt) :
    SideCapR H D shape untr b cmt (DeployedR.layR elem)
      (bdDeployed rel nextE colsE wE FlockDeploy.joinR rowIn DeployedR.finalR w) rd pieces posAt := by
  intro x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO w' p hst q hq ho ht
  by_cases hl0 : q.level = 0
  · obtain ⟨hrep, hpos⟩ := h0 x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO w' p hst q hq ho ht hl0
    intro os last hfin _
    have hxi : Claims.Recursion.xI D x = xIOf x := by
      unfold Claims.Recursion.xI xIOf
      rw [hP]
    have hDx : DenseX x := by
      unfold DenseX
      exact ⟨st, by rw [← hP]; exact hx, hD _ hf⟩
    have hl : ws.length = nS x := by rw [← hN]; exact hJ.1
    have htab : q.table = 0 := by
      unfold ownQR at ho
      by_contra hne
      rw [if_neg hne] at ho
      cases ho
    rw [hxi] at hpos hfin ⊢
    rw [msgs_layR] at hfin
    exact capEntry_level0 elem rd posAt hDx hl hl0 htab hrep hpos hfin
  · exact hup x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO w' p hst q hq ho ht hl0

end Deployed

end RecDeploy.CapLower
