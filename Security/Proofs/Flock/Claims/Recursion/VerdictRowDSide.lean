import Proofs.Flock.Claims.Recursion.VerdictRowDPieces
import Proofs.Flock.Claims.Recursion.VerdictRowG
import Proofs.Flock.VBridge.Family.Structure

/-!
# Row D's layout tie: the side a `rec-open` piece reads

`DecAt.rowD_of_pieces` takes a `sideAt` that reads each `rec-open` piece's side (the table's schedule, the rep's cap
and Ligerito proof) off the statement and the messages, and asks it to be the decided one (`hside`). This file fixes
that reader and proves the tie.

* `sideAt rd`: at a statement that sets up and messages whose last one `rd` reads, rep `k = 2 table + rep`'s proof read
  from its opened rounds (`readRep`) as `decR` reads it, but at a context that takes no coin; its cap and Ligerito
  proof, with the table's `fast100` schedule.
* `readRep_side`: what `readRep` reads of the cap and Ligerito proof doesn't depend on the context's link coin or `z`:
  `FitsRep` reads neither, the extra-claim count is one per region and link point at any coin
  (`extraCount_eq`), and `absorbedFrom` reads the context only in `zerocheck.finalC`.
* `DecAt.sideAt_eq`, `DecAt.hside`: at a decision, a piece of table `i` and rep `r` reads `D.side cls i r`, in
  `runs_assembled`'s form.
-/

namespace RecDeploy.Rows

open Flock
open FlockDeploy (Coin FinalP DecR decR RepCtx repCtx repLens cutLens rowsOf nodeOf FitsRep readRep readPs linkR tabsOf
  pointsOf setupOf Ins recJ mapM_some_get)
open FlockSoundness.Discharge.Recursion (HClass)

noncomputable section

/-! ## The extra-claim count reads no link point -/

theorem extraClaims_ok_of {st : Stmt} {P P' : Array (Array F128)} {e : Array Flock.Claim}
    (h : st.extraClaims P = .ok e) (hs : P.size = P'.size) : ∃ e', st.extraClaims P' = .ok e' := by
  rcases Nat.eq_zero_or_pos P.size with h0 | h0
  · have hP : P = #[] := Array.eq_empty_of_size_eq_zero h0
    have hP' : P' = #[] := Array.eq_empty_of_size_eq_zero (hs ▸ h0)
    exact ⟨e, by rw [hP', ← hP]; exact h⟩
  · exact FlockVBridge.Family.extraClaims_exists st P' (FlockVBridge.Family.regionBytes_of_extraClaims h h0)

/-- **The extra-claim count is the same at any two link point arrays of one size**: one claim per region and point
where `extraClaims` returns, and where it fails at one it fails at the other. -/
theorem extraCount_eq (st : Stmt) {P P' : Array (Array F128)} (hs : P.size = P'.size) :
    (match st.extraClaims P with
      | .ok e => e.size
      | .error _ => 0) =
    (match st.extraClaims P' with
      | .ok e => e.size
      | .error _ => 0) := by
  cases h : st.extraClaims P with
  | error a =>
    cases h' : st.extraClaims P' with
    | error b => rfl
    | ok e' =>
      obtain ⟨_, he⟩ := extraClaims_ok_of h' hs.symm
      rw [h] at he
      cases he
  | ok e =>
    cases h' : st.extraClaims P' with
    | error b =>
      obtain ⟨_, he⟩ := extraClaims_ok_of h hs
      rw [h'] at he
      cases he
    | ok e' =>
      show e.size = e'.size
      rw [FlockDeploy.extraClaims_size h, FlockDeploy.extraClaims_size h', hs]

theorem pointsOf_size (spec : SessionSpec) (c : Coin) : (pointsOf spec c).size = spec.points := by
  simp [FlockDeploy.pointsOf]

theorem tabsOf_get {sts : Array Stmt} {j : ℕ} (hj : j < sts.size) : (tabsOf sts)[j]! = Setup.ofCircuit sts[j] := by
  unfold FlockDeploy.tabsOf
  rw [getElem!_pos (sts.map Setup.ofCircuit) j (by simpa using hj), Array.getElem_map]

/-- **A rep's claim count reads no coin**: at any link coin, rep `k`'s context counts the same claims. -/
theorem repCtx_n {spec : SessionSpec} {sts : Array Stmt} {k : ℕ} (hk : k / 2 < sts.size) (c₁ c₁' : Coin)
    (r₀ r₀' : List ByteArray) (cs cs' : List Coin) :
    (repCtx spec sts c₁ r₀ cs k).n = (repCtx spec sts c₁' r₀' cs' k).n := by
  show 2 + _ = 2 + _
  rw [tabsOf_get hk]
  exact congrArg (2 + ·) (extraCount_eq sts[k / 2] (by rw [pointsOf_size, pointsOf_size]))

/-! ## What `readRep` reads of the openings -/

/-- **`FitsRep` reads no `z`**: it holds at a context with the same setup, schedule, domain, claim count, rounds'
schedule and level-0 nodes. -/
theorem fitsRep_of {c c' : RepCtx} (hst : c.st = c'.st) (hsch : c.sch = c'.sch) (hdom : c.dom = c'.dom)
    (hn : c.n = c'.n) (hsched : c.sched = c'.sched) (hcap0 : c.cap0 = c'.cap0) {rows : List (List ByteArray)}
    {p : Flock.Proof} {T : F128} (h : FitsRep c rows p T) : FitsRep c' rows p T := by
  obtain ⟨st, sch, dom, n, sched, cap0, z⟩ := c
  obtain ⟨st', sch', dom', n', sched', cap0', z'⟩ := c'
  dsimp only at hst hsch hdom hn hsched hcap0
  subst hst hsch hdom hn hsched hcap0
  exact ⟨h.rounds, h.sched, h.cap, h.tops, h.nonces, h.sizes, h.levels, h.elems⟩

theorem choose_congr {α : Type} {P P' : α → Prop} (hP : P = P') (h : ∃ a, P a) (h' : ∃ a, P' a) :
    Classical.choose h = Classical.choose h' := by
  subst hP
  rfl

/-- The cap and Ligerito proof of what `readRep` reads. -/
def sideOf (c : RepCtx) (rows : List (List ByteArray)) (q : Flock.Proof) :
    Option (Array ByteArray × Flock.LigeritoProof) :=
  (readRep c rows q).map fun pm => (pm.1.cap, pm.1.ligerito)

/-- **The cap and Ligerito proof `readRep` reads don't depend on `z`**: at two contexts that agree but in `z`, the same
proofs fit, and `absorbedFrom` reads the context only in `zerocheck.finalC`. -/
theorem readRep_side {c c' : RepCtx} (hst : c.st = c'.st) (hsch : c.sch = c'.sch) (hdom : c.dom = c'.dom)
    (hn : c.n = c'.n) (hsched : c.sched = c'.sched) (hcap0 : c.cap0 = c'.cap0) (rows : List (List ByteArray))
    (q : Flock.Proof) : sideOf c rows q = sideOf c' rows q := by
  have hP : (fun pT : Flock.Proof × F128 => FitsRep c rows pT.1 pT.2) = fun pT => FitsRep c' rows pT.1 pT.2 :=
    funext fun pT => propext ⟨fitsRep_of hst hsch hdom hn hsched hcap0,
      fitsRep_of hst.symm hsch.symm hdom.symm hn.symm hsched.symm hcap0.symm⟩
  unfold sideOf readRep
  by_cases hx : ∃ pT : Flock.Proof × F128, FitsRep c rows pT.1 pT.2
  · have hx' : ∃ pT : Flock.Proof × F128, FitsRep c' rows pT.1 pT.2 := hP ▸ hx
    rw [dif_pos hx, dif_pos hx', choose_congr hP hx hx']
    rfl
  · have hx' : ¬ ∃ pT : Flock.Proof × F128, FitsRep c' rows pT.1 pT.2 := hP ▸ hx
    rw [dif_neg hx, dif_neg hx']

/-- **Rep `k`'s openings read at no coin**: at any link coin and coins after the link, rep `k`'s context gives the
cap and Ligerito proof it gives at the default coin and none after. -/
theorem sideOf_repCtx {spec : SessionSpec} {sts : Array Stmt} {k : ℕ} (hk : k / 2 < sts.size) (c₁ : Coin)
    (r₀ : List ByteArray) (cs : List Coin) (rows : List (List ByteArray)) (q : Flock.Proof) :
    sideOf (repCtx spec sts c₁ r₀ cs k) rows q = sideOf (repCtx spec sts default r₀ [] k) rows q :=
  readRep_side (c := repCtx spec sts c₁ r₀ cs k) (c' := repCtx spec sts default r₀ [] k) rfl rfl rfl
    (repCtx_n hk _ _ _ _ _ _) rfl rfl rows q

/-- **`readPs` reads each rep with `readRep`**: rep `k`'s proof is the one `readRep` reads from its block of rounds,
with the last message's `k`-th proof. -/
theorem readPs_get {spec : SessionSpec} {sts : Array Stmt} {c₁ : Coin} {r₀ : List ByteArray}
    {rss : List (List ByteArray)} {cs : List Coin} {fl : FinalP} {pf : Array Flock.Proof × List Claims.Msg}
    (h : readPs spec sts c₁ r₀ rss cs fl = some pf) {k : ℕ} (hk : k < 2 * sts.size) :
    ∃ ms', readRep (repCtx spec sts c₁ r₀ cs k) ((cutLens (repLens spec sts) rss).getD k []) fl.ps[k]! =
      some (pf.1[k]!, ms') := by
  unfold readPs at h
  by_cases hc : fl.ps.size = 2 * sts.size ∧ (repLens spec sts).sum = rss.length
  · rw [if_pos hc] at h
    obtain ⟨rs, hrs, rfl⟩ := Option.map_eq_some_iff.1 h
    obtain ⟨hlen, hget⟩ := mapM_some_get hrs
    rw [List.length_range] at hlen
    have hk' : k < rs.length := by omega
    have hg := hget k (by rw [List.length_range]; exact hk) hk'
    rw [List.getElem_range] at hg
    refine ⟨rs[k].2, ?_⟩
    have e : ((rs.map Prod.fst).toArray)[k]! = rs[k].1 := by
      rw [getElem!_pos ((rs.map Prod.fst).toArray) k (by simpa using hk')]
      simp
    rw [hg, e]
  · rw [if_neg hc] at h
    cases h

/-- `decR`'s reading, with the last message's proof. -/
theorem decR_last {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {d : DecR} (h : decR rd x as cs = some d) :
    ∃ (v : Ins × SessionSpec × Array Stmt) (c₁ : Coin) (r₀ : List ByteArray) (rss : List (List ByteArray))
      (fl : FinalP) (link : List ByteArray × List ByteArray) (pf : Array Flock.Proof × List Claims.Msg),
      setupOf false x = some v ∧ as.getLast?.bind rd = some fl ∧ cs.head? = some c₁ ∧
      (rowsOf (as.headD [])).map (List.map nodeOf) = some r₀ ∧ as.tail.dropLast.mapM rowsOf = some rss ∧
      linkR v.2.2 r₀ = some link ∧ readPs v.2.1 v.2.2 c₁ r₀ rss cs.tail.dropLast fl = some pf ∧
      d = ⟨v.1, v.2.1, v.2.2, link, c₁, pf.2.zip cs.tail.dropLast,
        ⟨x, recJ v.2.1 v.2.2 link c₁ pf.2 cs.tail.dropLast pf.1, pf.1⟩⟩ := by
  simp only [decR, Option.bind_eq_bind, Option.pure_def] at h
  obtain ⟨v, h1, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨a, h2, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨a₁, h3, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨c₁, h4, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨fl, h5, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨r₀, h6, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨rss, h7, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨link, h8, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨pf, h9, h⟩ := Option.bind_eq_some_iff.1 h
  have ha : as.headD [] = a₁ := by
    cases as with
    | nil => cases h3
    | cons b l => exact Option.some.inj h3
  exact ⟨v, c₁, r₀, rss, fl, link, pf, h1, (by rw [h2]; exact (Option.filter_eq_some_iff.1 h5).1), h4,
    (by rw [ha]; exact h6), h7, h8, (Option.filter_eq_some_iff.1 h9).1, (Option.some.inj h).symm⟩

/-! ## The reader -/

/-- Table `t`'s `fast100` schedule at the inner statement `x`, as `repCtx` reads it. -/
def schOf (x : Claims.Msg) (t : ℕ) : Flock.Schedule :=
  match setupOf false x with
  | some v => (fast100 (tabsOf v.2.2)[t]!.m).getD default
  | none => default

/-- The cap and Ligerito proof of a `rec-open` piece's side: its rep's proof as `readRep` reads it (`sideAt`). -/
def capLigAt (rd : Claims.Msg → Option FinalP) (x : Claims.Msg) (ms : List Claims.Msg) (q : OpenPiece) :
    Array ByteArray × Flock.LigeritoProof :=
  match setupOf false x, ms.getLast?.bind rd, (rowsOf (ms.headD [])).map (List.map nodeOf),
    ms.tail.dropLast.mapM rowsOf with
  | some v, some fl, some r₀, some rss =>
    (sideOf (repCtx v.2.1 v.2.2 default r₀ [] (2 * q.table + q.rep))
      ((cutLens (repLens v.2.1 v.2.2) rss).getD (2 * q.table + q.rep) []) fl.ps[2 * q.table + q.rep]!).getD default
  | _, _, _, _ => default

/-- **The side a `rec-open` piece reads off the statement and the messages**: at a statement that sets up and
messages whose last one `rd` reads, with round 0's nodes and the later rounds' rows read back, rep `k = 2 table + rep`'s
proof as `readRep` reads it from the `k`-th block of the later rounds' rows (at the default link coin: what it reads of
the openings takes no coin, `sideOf_repCtx`), with the last message's `k`-th proof; the side is the table's `fast100`
schedule (`schOf`, whatever the messages), and that proof's cap and Ligerito proof (`capLigAt`). -/
def sideAt (rd : Claims.Msg → Option FinalP) (x : Claims.Msg) (ms : List Claims.Msg) (q : OpenPiece) : OpenSide :=
  ⟨schOf x q.table, (capLigAt rd x ms q).1, (capLigAt rd x ms q).2⟩

namespace DecAt

variable {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {ms : List Claims.Msg} {cs : List Coin}
  (D : DecAt rd x ms cs) (cls : HClass) (i : ℕ)

/-- **A piece reads its rep's decided side**: at a decision, a piece of table `i < sts.size` and rep `r` reads the
table's schedule and rep `r`'s decided cap and Ligerito proof. -/
theorem sideAt_eq (hi : i < D.sts.size) (r : Fin 2) (q : OpenPiece) (ht : q.table = i) (hr : q.rep = r.val) :
    sideAt rd x ms q = D.side cls i r := by
  obtain ⟨v, c₁, r₀, rss, fl, link, pf, hv, hfl, -, hr₀, hrss, -, hpf, hd⟩ := decR_last D.dec
  obtain ⟨-, hsts⟩ := D.setup_eq
  have hsts' : D.sts = v.2.2 := by rw [hsts, hd]
  have hps : D.d.f.ps = pf.1 := by rw [hd]
  have hk : 2 * i + r.val < 2 * v.2.2.size := by rw [← hsts']; omega
  have hk2 : (2 * i + r.val) / 2 < v.2.2.size := by rw [← hsts']; omega
  have hkt : (2 * i + r.val) / 2 = i := by omega
  obtain ⟨ms', hrep⟩ := readPs_get hpf hk
  have hs := sideOf_repCtx (spec := v.2.1) hk2 c₁ r₀ cs.tail.dropLast
    ((cutLens (repLens v.2.1 v.2.2) rss).getD (2 * i + r.val) []) fl.ps[2 * i + r.val]!
  rw [sideOf, hrep, Option.map_some] at hs
  unfold sideAt schOf capLigAt
  rw [hv, hfl, hr₀, hrss]
  dsimp only
  rw [ht, hr, ← hs, Option.getD_some]
  unfold DecAt.side DecAt.files
  dsimp only
  rw [hsts', hps]

/-- **`hside`, the layout tie, in `runs_assembled`'s form**: at a decision and a table `i < sts.size`, every piece of
rep `r` of table `i` reads `D.side cls i r` through `sideAt rd`. -/
theorem hside (hi : i < D.sts.size) (pieces : List OpenPiece) :
    ∀ r : Fin 2, ∀ q ∈ pieces, q.table = i → q.rep = r.val → sideAt rd x ms q = D.side cls i r :=
  fun r q _ ht hr => D.sideAt_eq cls i hi r q ht hr

end DecAt

end

end RecDeploy.Rows
