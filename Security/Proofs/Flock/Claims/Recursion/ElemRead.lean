import Proofs.Flock.Claims.Recursion.VerdictRowRDep
import Proofs.Flock.Claims.Recursion.VerdictRowRDeployed
import Proofs.Flock.Claims.Recursion.VerdictRowDSide
import Proofs.Flock.Claims.Recursion.VerdictFramed

/-!
# Row R's `elemAt`: the rep's `w` read off the messages

`ElemCover`'s elements half asks, at every decision `At` admits, for a piece keyed `(p, o, g, u)` with `o + g = r`:
`elemAt x ms (p, o, g, u) k = el (D.rep cls acc i r).w (wIdx (portSizes (D.sh cls i)) gs[p].2.1)[k]`. Rep `r`'s `w`
is its frames' values at `T` (`DecAt.w`), then its four accumulators `acc`.

* `framesRead x ms k`: rep `k`'s frame values off the statement and the messages alone. A proof and `T` that fit the
  rep's opened rounds (`FitsRep`) at a context that takes no coin; each element of the rounds at table 0's shape
  (`shapeNoCoin`, no coin either) read at them (`keyVal`). `FitsRep` reads no coin (`fitsRep_of`, `repCtx_n`), and any
  two proofs that fit one rep's rows agree in every field `keyVal` reads, and in `T` (`FlockDeploy.fitUnique`).
* `DecAt.framesRead_eq`, `DecAt.w_eq`: at a dense decision, `framesRead` is the decided proof's values at the decided
  `T`, and rep `r`'s `w` is `framesRead x ms r` then `acc r`. Every decision, adversarial included: no honest fact.
* `elemRead`: the coin-free `elemAt`, `el (framesRead x ms (o + g))` at group `p`'s row port `k`. `DecAt.elemRead_eq`
  is `ElemCover`'s elements half at every row port that is a frame value (below `(rounds sh).flatten.size`).
* `elemReadC acc`: the same with the coins, reading the accumulator ports at `acc x ms cs`. `DecAt.elemReadC_eq` is
  `ElemCover`'s elements half at every row port, at any accumulators, among them `Top.accRC rd key`.

No coin-free `elemAt` meets the half at the accumulator ports `acc0`, `acc1` (`portSizes`' last two), which Ligerito's
final-check residuals read (`accTerms`). There `w` holds `Top.accRC rd key x ms cs`, `chainC`'s rows: a choice over
`ChainHolds`, which leaves the chain's first row free and steps each row by the coins' `coefAt`. The messages carry
no coin (`jAt` refills the record's), so two coin lists admitted at one `(x, ms)` give different accumulators for
one `elemAt`. `elemReadC` is the reader once `elemAt` takes the coins (`resHolds` already has them).
-/

namespace RecDeploy.Rows

open Flock
open FlockDeploy (Coin FinalP tabsOf pointsOf setupOf repCtx repLens cutLens rowsOf nodeOf FitsRep absorbedFrom)
open Flock.RecAlgebra (structureOf el rounds Key)
open Flock.RecResiduals (portSizes groups)
open FlockVBridge.PartS (wIdx)
open FlockSoundness.Refine (shapeOf)
open FlockSoundness.Discharge.Recursion (HClass)
open FlockSoundness.Discharge.DecodeZL (ProofSizes)
open FlockDecodeTies (keyVal)

noncomputable section

/-! ## Reading the frames with no coin -/

/-- Table 0's shape, its extra claims counted at the default link coin (`repCtx_n`: the count reads no coin). -/
def shapeNoCoin (spec : SessionSpec) (sts : Array Stmt) : Shape :=
  shapeOf (tabsOf sts)[0]! ((fast100 (tabsOf sts)[0]!.m).getD default) ((repCtx spec sts default [] [] 0).n - 2)

open Classical in
/-- **Rep `k`'s frame values off the statement and the messages**: at a statement that sets up, with round 0's and
the later rounds' rows read back, a proof and `T` that fit rep `k`'s opened rounds at the default link coin and no
coin after, read at each element of table 0's rounds; `[]` where none fits. -/
def framesRead (x : Claims.Msg) (ms : List Claims.Msg) (k : ℕ) : List F128 :=
  match setupOf false x, (rowsOf (ms.headD [])).map (List.map nodeOf), ms.tail.dropLast.mapM rowsOf with
  | some v, some r₀, some rss =>
    if h : ∃ pT : Flock.Proof × F128, FitsRep (repCtx v.2.1 v.2.2 default r₀ [] k)
        ((cutLens (repLens v.2.1 v.2.2) rss).getD k []) pT.1 pT.2 then
      (rounds (shapeNoCoin v.2.1 v.2.2)).flatten.toList.map
        (keyVal (Classical.choose h).1 (Classical.choose h).2)
    else []
  | _, _, _ => []

/-- `keyVal` reads only the fields a fitting proof's frames absorb. -/
theorem keyVal_absorbedFrom (c : FlockDeploy.RepCtx) (a q : Flock.Proof) (T : F128) (y : Key) :
    keyVal (absorbedFrom c a q) T y = keyVal a T y := by
  cases y <;> rfl

theorem el_append_left {a b : Array F128} {j : ℕ} (h : j < a.size) : el (a ++ b) j = el a j := by
  unfold el
  rw [Array.getD_eq_getD_getElem?, Array.getD_eq_getD_getElem?, Array.getElem?_append_left h]

/-! ## The row ports -/

/-- Row port `k` of group `p` at a shape: the `w` index S's groups give it, where they run. -/
def portAt (sh : Shape) (p k : ℕ) : Option ℕ :=
  match structureOf sh with
  | .ok S =>
    match groups S (portSizes sh) with
    | .ok gs =>
      match gs[p]? with
      | some g => (wIdx (portSizes sh) g.2.1)[k]?
      | none => none
    | .error _ => none
  | .error _ => none

theorem portAt_eq {sh : Shape} {S : RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)}
    (hS : structureOf sh = .ok S) (hg : groups S (portSizes sh) = .ok gs) {p k : ℕ} (hp : p < gs.size)
    (hk : k < (wIdx (portSizes sh) gs[p].2.1).size) : portAt sh p k = some (wIdx (portSizes sh) gs[p].2.1)[k] := by
  unfold portAt
  simp [hS, hg, hp, hk]

/-- **The coin-free `elemAt`**: at key `(jj, o, g, u)`, rep `o + g`'s frame values (`framesRead`) at group `jj`'s row
port `k`; `0` at a statement that doesn't set up or a port S doesn't have. -/
def elemRead (x : Claims.Msg) (ms : List Claims.Msg) (key : ℕ × ℕ × ℕ × ℕ) (k : ℕ) : F128 :=
  match setupOf false x with
  | some v =>
    match portAt (shapeNoCoin v.2.1 v.2.2) key.1 k with
    | some j => el (framesRead x ms (key.2.1 + key.2.2.1)).toArray j
    | none => 0
  | none => 0

/-- **`elemAt` with the coins**: as `elemRead`, but `w`'s accumulator ports read `acc x ms cs` at the rep. -/
def elemReadC (acc : Claims.Msg → List Claims.Msg → List Coin → ℕ → Array F128) (x : Claims.Msg)
    (ms : List Claims.Msg) (cs : List Coin) (key : ℕ × ℕ × ℕ × ℕ) (k : ℕ) : F128 :=
  match setupOf false x with
  | some v =>
    match portAt (shapeNoCoin v.2.1 v.2.2) key.1 k with
    | some j => el ((framesRead x ms (key.2.1 + key.2.2.1)).toArray ++ acc x ms cs (key.2.1 + key.2.2.1)) j
    | none => 0
  | none => 0

namespace DecAt

variable {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {ms : List Claims.Msg} {cs : List Coin}
  (D : DecAt rd x ms cs) (cls : HClass)

/-- **Table 0's shape reads no coin**: where the setup row holds, the decision's shape is `shapeNoCoin`. -/
theorem sh_eq (acc : ℕ → Array F128) (hG : D.RowG cls acc 0) (h0 : 0 < D.sts.size) :
    D.sh cls 0 = shapeNoCoin D.spec D.sts := by
  have hext : (tabsOf D.sts)[0]!.extra (pointsOf D.spec D.d.c₁) = .ok (D.files cls 0).extras := by
    rw [← D.points]
    exact hG.extra
  have hn : (repCtx D.spec D.sts default [] [] 0).n = 2 + (D.files cls 0).extras.size := by
    rw [← repCtx_n (spec := D.spec) (sts := D.sts) (k := 0) (by simpa using h0) D.d.c₁ default [] [] [] []]
    simp only [repCtx, Nat.zero_div, hext]
  unfold shapeNoCoin
  rw [hn, Nat.add_sub_cancel_left]
  rfl

/-- **The frames read off the messages are the decision's**: at a dense decision, rep `r` of table `i`'s
`framesRead` is the decided proof's values at the decided `T`, element by element of the rounds. -/
theorem framesRead_eq {xo : Claims.Msg} {sto : Claims.Recursion.Stmt} (hst : parse xo = some sto)
    (hf : Claims.Recursion.Form.Dense sto.form) (hx : x = xIOf xo) {i : ℕ} (hi : i < D.sts.size) (r : Fin 2) :
    framesRead x ms (2 * i + r) =
      (rounds (D.sh cls i)).flatten.toList.map (keyVal D.d.f.ps[2 * i + r]! (D.T (2 * i + r))) := by
  obtain ⟨hT1, -⟩ := D.tables_of_dense hst hf hx
  obtain rfl : i = 0 := by omega
  have hG := D.rowG cls (fun _ => #[]) hst hf hx hi
  have hk : 2 * 0 + r.val < D.d.f.ps.size := D.rep_lt hi r
  obtain ⟨v, c₁, r₀, rss, fl, link, pf, hv, -, -, hr₀, hrss, -, -, hd⟩ := decR_last D.dec
  obtain ⟨hsp, hsts⟩ := D.setup_eq
  have hds : D.d.spec = v.2.1 := by rw [hd]
  have hdt : D.d.sts = v.2.2 := by rw [hd]
  have hdc : D.d.c₁ = c₁ := by rw [hd]
  have e1 : D.r₀ = r₀ := by simp only [DecAt.r₀, hr₀, Option.getD_some]
  have e2 : D.rss = rss := by simp only [DecAt.rss, hrss, Option.getD_some]
  have hk2 : (2 * 0 + r.val) / 2 < v.2.2.size := by rw [← hdt, ← hsts]; omega
  have hctx : D.ctx (2 * 0 + r.val) = repCtx v.2.1 v.2.2 c₁ r₀ cs.tail.dropLast (2 * 0 + r.val) := by
    unfold DecAt.ctx
    rw [hds, hdt, hdc, e1]
  have hrows : D.rows (2 * 0 + r.val) = (cutLens (repLens v.2.1 v.2.2) rss).getD (2 * 0 + r.val) [] := by
    unfold DecAt.rows
    rw [hds, hdt, e2]
  have hn' := repCtx_n (spec := v.2.1) hk2 c₁ default r₀ r₀ cs.tail.dropLast []
  have hfit' : FitsRep (repCtx v.2.1 v.2.2 default r₀ [] (2 * 0 + r.val))
      ((cutLens (repLens v.2.1 v.2.2) rss).getD (2 * 0 + r.val) []) D.d.f.ps[2 * 0 + r.val]!
      (D.T (2 * 0 + r.val)) := by
    have h := D.ctx_rows_fit hk
    rw [hctx, hrows] at h
    exact fitsRep_of (c := repCtx v.2.1 v.2.2 c₁ r₀ cs.tail.dropLast (2 * 0 + r.val))
      (c' := repCtx v.2.1 v.2.2 default r₀ [] (2 * 0 + r.val)) rfl rfl rfl hn' rfl rfl h
  have hex : ∃ pT : Flock.Proof × F128, FitsRep (repCtx v.2.1 v.2.2 default r₀ [] (2 * 0 + r.val))
      ((cutLens (repLens v.2.1 v.2.2) rss).getD (2 * 0 + r.val) []) pT.1 pT.2 := by
    generalize D.d.f.ps[2 * 0 + r.val]! = P0 at hfit'
    generalize D.T (2 * 0 + r.val) = T0 at hfit'
    exact ⟨(P0, T0), hfit'⟩
  have hb : FitsRep (D.ctx (2 * 0 + r.val)) (D.rows (2 * 0 + r.val)) (Classical.choose hex).1
      (Classical.choose hex).2 := by
    rw [hctx, hrows]
    exact fitsRep_of (c := repCtx v.2.1 v.2.2 default r₀ [] (2 * 0 + r.val))
      (c' := repCtx v.2.1 v.2.2 c₁ r₀ cs.tail.dropLast (2 * 0 + r.val)) rfl rfl rfl hn'.symm rfl rfl
      (Classical.choose_spec hex)
  have hkl : (D.ctx (2 * 0 + r.val)).st.fold.kLog = (D.ctx (2 * 0 + r.val)).st.kLog := by
    rw [D.ctx_st cls (fun _ => #[]) 0 r hG]
    exact hG.foldKLog
  obtain ⟨hab, hT⟩ := FlockDeploy.fitUnique _ _ _ _ _ _ hkl (D.ctx_rows_fit hk) hb default
  have hkv : keyVal (Classical.choose hex).1 (Classical.choose hex).2 =
      keyVal D.d.f.ps[2 * 0 + r.val]! (D.T (2 * 0 + r.val)) := by
    funext y
    rw [← keyVal_absorbedFrom (D.ctx (2 * 0 + r.val)) (Classical.choose hex).1 default, hab,
      keyVal_absorbedFrom, hT]
  have hsh : D.sh cls 0 = shapeNoCoin v.2.1 v.2.2 := by
    rw [D.sh_eq cls (fun _ => #[]) hG hi, hsp, hsts, hds, hdt]
  unfold framesRead
  rw [hv, hr₀, hrss]
  dsimp only
  rw [dif_pos hex, hkv, hsh]

/-- **Rep `r`'s `w` is `framesRead`, then its accumulators**, at every dense decision. -/
theorem w_eq (acc : ℕ → Array F128) {xo : Claims.Msg} {sto : Claims.Recursion.Stmt} (hst : parse xo = some sto)
    (hf : Claims.Recursion.Form.Dense sto.form) (hx : x = xIOf xo) {i : ℕ} (hi : i < D.sts.size) (r : Fin 2) :
    (D.rep cls acc i r).w = (framesRead x ms (2 * i + r)).toArray ++ acc (2 * i + r) := by
  have hG := D.rowG cls acc hst hf hx hi
  have hk : 2 * i + r.val < D.d.f.ps.size := D.rep_lt hi r
  have hext : (tabsOf D.sts)[i]!.extra (pointsOf D.spec D.d.c₁) = .ok (D.files cls i).extras := by
    rw [← D.points]
    exact hG.extra
  obtain ⟨e, hne, hps⟩ := (D.ctx_rows_fit hk).sizes
  have hn := D.ctx_n cls acc i r hG hext
  have he : e = (D.files cls i).extras.size := by
    have : 2 + e = 2 + (D.files cls i).extras.size := hne.symm.trans hn
    omega
  have hst0 : (D.ctx (2 * i + r)).st = (D.files cls i).st := D.ctx_st cls acc i r hG
  have hsch : (D.ctx (2 * i + r)).sch = (D.files cls i).sch := D.ctx_sch cls acc i r hG
  have hp : ProofSizes (D.sh cls i) ((D.files cls i).proof r) := by
    rw [hst0, hsch, he] at hps
    exact hps
  have hs := D.ctx_sch cls acc i r hG
  have hvals := FlockDecodeTies.vals_repFrames_rounds (sh := D.sh cls i) (st := (D.ctx (2 * i + r)).st)
    (sch := (D.ctx (2 * i + r)).sch) (n := (D.ctx (2 * i + r)).n) (D.ctx (2 * i + r)).dom D.d.f.ps[2 * i + r]!
    (D.T (2 * i + r)) (by rw [D.ctx_st cls acc i r hG]; rfl)
    (by rw [D.ctx_st cls acc i r hG]; exact hG.foldKLog) hn hs rfl hp.r1ab hp.r1c hp.zPartial
    (fun k hk => FlockDecodeTies.rsEach_get hp (by rw [← hn]; exact hk))
    (by rw [hs]; exact FlockDecodeTies.yr_size hp)
  apply Array.toList_inj.1
  show (D.w (2 * i + r) (acc (2 * i + r))).toList = ((framesRead x ms (2 * i + r)).toArray ++ acc (2 * i + r)).toList
  rw [Array.toList_append, List.toList_toArray, D.w_toList, hvals, D.framesRead_eq cls hst hf hx hi r]

/-- `framesRead` has one value per element of the rounds, at a dense decision. -/
theorem framesRead_length {xo : Claims.Msg} {sto : Claims.Recursion.Stmt} (hst : parse xo = some sto)
    (hf : Claims.Recursion.Form.Dense sto.form) (hx : x = xIOf xo) {i : ℕ} (hi : i < D.sts.size) (r : Fin 2) :
    (framesRead x ms (2 * i + r)).length = (rounds (D.sh cls i)).flatten.size := by
  rw [D.framesRead_eq cls hst hf hx hi r, List.length_map, Array.length_toList]

/-- **`ElemCover`'s elements half with the coins**: at a dense decision, a table `i < sts.size`, S and its groups, a
group `p` and a key `(p, o, g, u)` with `o + g = r`, `elemReadC acc` at row port `k` is rep `r`'s `w` there, at the
accumulators `acc x ms cs`. -/
theorem elemReadC_eq (acc : Claims.Msg → List Claims.Msg → List Coin → ℕ → Array F128) {xo : Claims.Msg}
    {sto : Claims.Recursion.Stmt} (hst : parse xo = some sto) (hf : Claims.Recursion.Form.Dense sto.form)
    (hx : x = xIOf xo) {i : ℕ} (hi : i < D.sts.size) {S : RecAlgebra.GfS}
    {gs : Array (Array ℕ × Array ℕ × Array ℕ)} (hS : structureOf (D.sh cls i) = .ok S)
    (hg : groups S (portSizes (D.sh cls i)) = .ok gs) (r : Fin 2) {p : ℕ} (hp : p < gs.size)
    (key : ℕ × ℕ × ℕ × ℕ) (h1 : key.1 = p) (hog : key.2.1 + key.2.2.1 = r.val) (k : ℕ)
    (hk : k < (wIdx (portSizes (D.sh cls i)) gs[p].2.1).size) :
    elemReadC acc x ms cs key k =
      el (D.rep cls (acc x ms cs) i r).w (wIdx (portSizes (D.sh cls i)) gs[p].2.1)[k] := by
  obtain ⟨hT1, -⟩ := D.tables_of_dense hst hf hx
  obtain rfl : i = 0 := by omega
  have hG := D.rowG cls (acc x ms cs) hst hf hx hi
  obtain ⟨v, c₁, r₀, rss, fl, link, pf, hv, -, -, -, -, -, -, hd⟩ := decR_last D.dec
  obtain ⟨hsp, hsts⟩ := D.setup_eq
  have hsh : D.sh cls 0 = shapeNoCoin v.2.1 v.2.2 := by
    rw [D.sh_eq cls _ hG hi, hsp, hsts, hd]
  have hpa := portAt_eq hS hg hp hk
  generalize (wIdx (portSizes (D.sh cls 0)) gs[p].2.1)[k] = j at hpa ⊢
  rw [hsh] at hpa
  obtain ⟨jj, o, g, u⟩ := key
  dsimp only at h1 hog
  subst h1
  rw [D.w_eq cls (acc x ms cs) hst hf hx hi r, show 2 * 0 + r.val = r.val by omega]
  unfold elemReadC
  rw [hv]
  dsimp only
  rw [hpa, hog]

/-- **`ElemCover`'s elements half with no coin, at the frame ports**: as `elemReadC_eq`, for the coin-free `elemRead`,
at every row port whose `w` index is a frame value, at any accumulators. -/
theorem elemRead_eq (acc : ℕ → Array F128) {xo : Claims.Msg} {sto : Claims.Recursion.Stmt}
    (hst : parse xo = some sto) (hf : Claims.Recursion.Form.Dense sto.form) (hx : x = xIOf xo) {i : ℕ}
    (hi : i < D.sts.size) {S : RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)}
    (hS : structureOf (D.sh cls i) = .ok S) (hg : groups S (portSizes (D.sh cls i)) = .ok gs) (r : Fin 2)
    {p : ℕ} (hp : p < gs.size) (key : ℕ × ℕ × ℕ × ℕ) (h1 : key.1 = p) (hog : key.2.1 + key.2.2.1 = r.val) (k : ℕ)
    (hk : k < (wIdx (portSizes (D.sh cls i)) gs[p].2.1).size)
    (hfr : (wIdx (portSizes (D.sh cls i)) gs[p].2.1)[k] < (rounds (D.sh cls i)).flatten.size) :
    elemRead x ms key k = el (D.rep cls acc i r).w (wIdx (portSizes (D.sh cls i)) gs[p].2.1)[k] := by
  obtain ⟨hT1, -⟩ := D.tables_of_dense hst hf hx
  obtain rfl : i = 0 := by omega
  have hG := D.rowG cls acc hst hf hx hi
  obtain ⟨v, c₁, r₀, rss, fl, link, pf, hv, -, -, -, -, -, -, hd⟩ := decR_last D.dec
  obtain ⟨hsp, hsts⟩ := D.setup_eq
  have hsh : D.sh cls 0 = shapeNoCoin v.2.1 v.2.2 := by
    rw [D.sh_eq cls _ hG hi, hsp, hsts, hd]
  have hlen := D.framesRead_length cls hst hf hx hi r
  have hpa := portAt_eq hS hg hp hk
  generalize (wIdx (portSizes (D.sh cls 0)) gs[p].2.1)[k] = j at hpa hfr ⊢
  rw [hsh] at hpa
  obtain ⟨jj, o, g, u⟩ := key
  dsimp only at h1 hog
  subst h1
  have hj : j < (framesRead x ms (2 * 0 + r.val)).toArray.size := by
    rw [List.size_toArray, hlen]
    exact hfr
  rw [D.w_eq cls acc hst hf hx hi r, el_append_left hj, show 2 * 0 + r.val = r.val by omega]
  unfold elemRead
  rw [hv]
  dsimp only
  rw [hpa, hog]

end DecAt

end

end RecDeploy.Rows
