import Proofs.Flock.Claims.Recursion.InnerAtCoins
import Proofs.Flock.Claims.Recursion.InnerSetupSame
import Proofs.Flock.Claims.Bridge.Custody

/-!
# Check 5's extra claims and comb are the deployed table's

`ExtrasR` at a dense compiled statement (`staged_extrasR`), so `InnerCoinsR` (`innerCoinsR_staged`) and `PosTieR`
(`posTieR_dense`) follow from the verifier's checks alone, and the comb check 5 computes is the deployed table's circuit
fold (`staged_comb`). Check 5 takes the inner session's extra claims and comb from its files' statement (`Inner.load`),
which is the deployed one as the algebra reads it (`innerSetup_same`), and the extra claims at its record's link
points, which are `pointsOf` the first coin:

* the record's link points are the coin server's: `RecVStar.innerLink` refuses others (`innerLink_points`, `held_eq`);
* the coin server's record read in its own format holds them (`RecSetup.checkReads`), and the reader reads that record
  back off the first coin (`Drawn`);
* the record decodes at a spec of two points of the files' `m_pts`, whatever full draw it carries (`drawnSpec_shape`),
  as the deployed spec (`innerSetup_same`), so its points are `pointsOf` the first coin (`points_of_flat`).
-/

namespace RecDeploy.InnerExtras

open Flock

/-! ## Lists -/

theorem flatten_getD {α : Type} {d : α} {m : ℕ} : ∀ {L : List (List α)}, (∀ l ∈ L, l.length = m) →
    ∀ {p i : ℕ}, i < m → L.flatten.getD (p * m + i) d = (L.getD p []).getD i d
  | [], _, p, i, _ => by simp
  | l :: L, h, 0, i, hi => by
    have hl := h l List.mem_cons_self
    simp only [List.flatten_cons, Nat.zero_mul, Nat.zero_add, List.getD_cons_zero]
    rw [List.getD_eq_getElem?_getD, List.getElem?_append_left (by rw [hl]; exact hi), ← List.getD_eq_getElem?_getD]
  | l :: L, h, p + 1, i, hi => by
    have hl := h l List.mem_cons_self
    simp only [List.flatten_cons, List.getD_cons_succ]
    rw [List.getD_eq_getElem?_getD, List.getElem?_append_right (by rw [hl, Nat.succ_mul]; omega),
      ← List.getD_eq_getElem?_getD]
    rw [show (p + 1) * m + i - l.length = p * m + i by rw [hl, Nat.succ_mul]; omega]
    exact flatten_getD (fun l' hl' => h l' (List.mem_cons_of_mem _ hl')) hi

theorem length_flatten_const {α : Type} {m : ℕ} : ∀ {L : List (List α)}, (∀ l ∈ L, l.length = m) →
    L.flatten.length = L.length * m
  | [], _ => by simp
  | l :: L, h => by
    rw [List.flatten_cons, List.length_append, h l List.mem_cons_self,
      length_flatten_const (fun l' hl' => h l' (List.mem_cons_of_mem _ hl')), List.length_cons, Nat.succ_mul]
    omega

theorem forall₂_comp {α β γ : Type} {R : α → β → Prop} {S : β → γ → Prop} :
    ∀ {a : List α} {b : List β} {c : List γ}, List.Forall₂ R a b → List.Forall₂ S b c →
      List.Forall₂ (fun x z => ∃ y, R x y ∧ S y z) a c
  | [], [], [], .nil, .nil => .nil
  | _ :: _, _ :: _, _ :: _, .cons h1 t1, .cons h2 t2 => .cons ⟨_, h1, h2⟩ (forall₂_comp t1 t2)

theorem forall₂_unique {α β : Type} {R S : α → β → Prop} (hu : ∀ x y z, R x y → S x z → y = z) :
    ∀ {a : List α} {b c : List β}, List.Forall₂ R a b → List.Forall₂ S a c → b = c
  | [], [], [], .nil, .nil => rfl
  | _ :: _, _ :: _, _ :: _, .cons h1 t1, .cons h2 t2 => by
    rw [hu _ _ _ h1 h2, forall₂_unique hu t1 t2]

theorem headD_range {α : Type} [Inhabited α] (c : ℕ → α) {B : ℕ} (hB : 0 < B) :
    ((List.range B).map c).headD default = c 0 := by
  obtain ⟨B, rfl⟩ : ∃ B', B = B' + 1 := ⟨B - 1, by omega⟩
  simp [List.range_succ_eq_map]

/-- **Points read off a coin, point-major**: rows of `m` elements whose concatenation is the first elements of `f`
are `f`'s, row `p` at `p * m`. -/
theorem points_of_flat {α : Type} (d : α) {P : Array (Array α)} {n m : ℕ} {f : ℕ → α}
    (hsz : P.size = n) (hrow : ∀ p ∈ P.toList, p.size = m)
    (hdr : (P.toList.map (·.toList)).flatten.toArray =
      (Array.range (P.toList.map (·.toList)).flatten.toArray.size).map f) :
    P = (Array.range n).map fun p => (Array.range m).map fun i => f (p * m + i) := by
  have hL : ∀ l ∈ P.toList.map (·.toList), l.length = m := by
    intro l hl
    obtain ⟨a, ha, rfl⟩ := List.mem_map.1 hl
    rw [Array.length_toList]; exact hrow a ha
  have hN := length_flatten_const hL
  apply Array.ext (by simp [hsz])
  intro p hp _
  have hpn : p < P.size := hp
  have hPp : P[p].size = m := hrow _ (Array.getElem_mem_toList hpn)
  apply Array.ext (by simp [hPp])
  intro i hi _
  have him : i < m := by rw [← hPp]; exact hi
  simp only [Array.getElem_map, Array.getElem_range]
  have hj : p * m + i < (P.toList.map (·.toList)).flatten.length := by
    rw [hN, List.length_map, Array.length_toList]
    have : (p + 1) * m ≤ P.size * m := Nat.mul_le_mul_right _ hpn
    rw [Nat.succ_mul] at this
    omega
  have e := flatten_getD (d := d) hL (p := p) him
  have hrowp : (P.toList.map (·.toList)).getD p [] = P[p].toList := by
    rw [List.getD_eq_getElem?_getD, List.getElem?_map, Array.getElem?_toList, Array.getElem?_eq_getElem hpn]
    rfl
  have e1 : P[p].toList.getD i d = P[p][i] := by
    rw [List.getD_eq_getElem?_getD, Array.getElem?_toList, Array.getElem?_eq_getElem hi]; rfl
  have e2 := congrArg (fun a : Array α => a[p * m + i]?) hdr
  simp only [List.getElem?_toArray, Array.getElem?_map, Array.getElem?_range, List.size_toArray] at e2
  rw [ite_cond_eq_true _ _ (eq_true hj)] at e2
  have e3 : (P.toList.map (·.toList)).flatten.getD (p * m + i) d = f (p * m + i) := by
    rw [List.getD_eq_getElem?_getD, e2]; rfl
  rw [hrowp, e1, e3] at e
  exact e.symm

/-! ## The link points, as the record and the coin server's transcript hold them -/

theorem mapError_ok {ε ε' α : Type} {x : Except ε α} {g : ε → ε'} {a : α} (h : x.mapError g = .ok a) :
    x = .ok a := by
  cases x with
  | error e => cases h
  | ok v => cases h; rfl

theorem mapError_forall₂ {g : String → String} {xs : Array String} {ys : Array F128}
    (h : xs.mapM (fun s => (f128OfHex s).mapError g) = .ok ys) :
    List.Forall₂ (fun s y => f128OfHex s = .ok y) xs.toList ys.toList :=
  (Claims.Bridge.Custody.mapM_forall₂ h).imp fun _ _ hs => mapError_ok hs

theorem fld_val {j v : Lean.Json} {k : String} (h : Record.fld j k = .ok v) : j.getObjVal? k = .ok v := by
  unfold Record.fld at h
  cases hv : j.getObjVal? k with
  | error e => rw [hv] at h; cases h
  | ok w => rw [hv] at h; cases h; rfl

theorem arr_val {j : Lean.Json} {k : String} {a : Array Lean.Json} (h : Record.arr j k = .ok a) :
    j.getObjVal? k = .ok (.arr a) := by
  unfold Record.arr Record.fld at h
  cases hv : j.getObjVal? k with
  | error e => simp [hv, bind, Except.bind] at h
  | ok v =>
    simp only [hv, bind, Except.bind] at h
    cases v <;> simp [Lean.Json.getArr?, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
    subst h
    rfl

theorem getArr_ok {q : Lean.Json} {a : Array Lean.Json} (h : q.getArr? = .ok a) : q = .arr a := by
  cases q <;> simp [Lean.Json.getArr?, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
  subst h
  rfl

/-- **The inner session's record's link points are the coin server's**: `RecVStar.innerLink` reads the record's
`link.points` as hex coins and refuses them unless they are the transcript's (`lk.points`). -/
theorem innerLink_points {tr : Lean.Json} {record : String} {lk : RecVStar.InnerLink} {j l : Lean.Json}
    (hj : Lean.Json.parse record = .ok j) (hl : j.getObjVal? "link" = .ok l)
    (h : RecVStar.innerLink tr record = .ok lk) :
    ∃ rp, l.getObjValAs? (Array (Array String)) "points" = .ok rp ∧
      List.Forall₂ (fun xs pt => List.Forall₂ (fun s y => f128OfHex s = .ok y) xs.toList pt.toList)
        rp.toList lk.points.toList := by
  have hb : ∀ {α : Type} [BEq α] [LawfulBEq α] {a b : α}, ¬(a != b) = true → a = b := by
    intro α _ _ a b h; simpa using h
  unfold RecVStar.innerLink at h
  rw [hj] at h
  simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
  rw [hl] at h
  dsimp only at h
  repeat' split at h
  all_goals try (cases h; done)
  cases h
  rename_i hrp _ _ hmap hne
  have e := hb hne
  refine ⟨_, hrp, ?_⟩
  dsimp only
  rw [← e]
  have hmap' : Array.mapM (fun xs : Array String => xs.mapM fun s => (f128OfHex s).mapError
      (s!"rec inner transcript: {"the inner session's record's link points"}: " ++ ·)) _ = .ok _ := hmap
  exact (Claims.Bridge.Custody.mapM_forall₂ hmap').imp fun _ _ hx => mapError_forall₂ hx

/-- **The decoded link points are the record's `link.points`, read as `innerLink` reads them.** -/
theorem held_eq {spec : SessionSpec} {text : String} {sess : Session} {j l : Lean.Json}
    {rp : Array (Array String)} {pts : Array (Array F128)}
    (hdec : Record.decode spec text = .ok sess) (hj : Lean.Json.parse text = .ok j)
    (hl : j.getObjVal? "link" = .ok l) (hrp : l.getObjValAs? (Array (Array String)) "points" = .ok rp)
    (hh : List.Forall₂ (fun xs pt => List.Forall₂ (fun s y => f128OfHex s = .ok y) xs.toList pt.toList)
      rp.toList pts.toList) :
    sess.points = pts := by
  obtain ⟨j', l', -, hj', hl', -, -, ⟨ps, hps, hPF⟩, -⟩ := Claims.Bridge.Custody.decode_ok hdec
  rw [hj] at hj'
  cases hj'
  rw [fld_val hl'] at hl
  cases hl
  rw [RecDeploy.CoinTie.objAs_eq (α := Array (Array String)) (arr_val hps)] at hrp
  have hrp' : ps.mapM (fun q => (Lean.fromJson? q : Except String (Array String))) = .ok rp := hrp
  have h2 := forall₂_comp (Claims.Bridge.Custody.mapM_forall₂ hrp') hh
  refine Array.toList_inj.1 (forall₂_unique ?_ hPF h2)
  intro q pt pt' ⟨a, ha, hf⟩ ⟨xs, hxs, hhex⟩
  have hxs' : a.mapM Lean.Json.getStr? = .ok xs := by
    rw [getArr_ok ha] at hxs
    exact hxs
  have g2 := forall₂_comp (Claims.Bridge.Custody.mapM_forall₂ hxs') hhex
  unfold Record.f128s at hf
  refine Array.toList_inj.1 (forall₂_unique ?_ (Claims.Bridge.Custody.mapM_forall₂ hf) g2)
  intro y z z' hz ⟨str, hs, hz'⟩
  beta_reduce at hz hs hz'
  rw [hs] at hz
  dsimp only at hz
  exact Except.ok.inj (hz.symm.trans hz')

/-! ## The inner session check 5 loads -/

/-- **What `Inner.load` reads its extra claims at**: the files' statement's, at the link points of the record decoded
at the spec its draw gives (`Files.drawnSpec`). -/
theorem load_extras {unit : String} {f : RecCoins.Files} {inn : RecCoins.Inner}
    (h : RecCoins.Inner.load unit f = .ok inn) :
    ∃ spec st specD sess, f.stmt unit = .ok (spec, st) ∧ f.drawnSpec spec st = .ok specD ∧
      Record.decode specD f.record = .ok sess ∧ inn.extras = (Setup.ofCircuit st).extra sess.points := by
  unfold RecCoins.Inner.load at h
  obtain ⟨⟨spec, st⟩, hs, h⟩ := RecDeploy.CoinTie.tie_bind h
  refine ⟨spec, st, ?_⟩
  dsimp only at h
  repeat' split at h
  all_goals first
    | (cases h; done)
    | skip
  all_goals obtain ⟨specD, hsp, h⟩ := RecDeploy.CoinTie.tie_bind h
  all_goals split at h
  all_goals first
    | (cases h; done)
    | skip
  all_goals obtain ⟨sess, hse, h⟩ := RecDeploy.CoinTie.tie_bind h
  all_goals cases h
  all_goals
    cases hse
    cases hsp
    exact ⟨_, _, hs, by assumption, by assumption, rfl⟩

/-- **The record decodes at two link points of the files' `m_pts`**: without a draw at the files' spec, and at a draw
that proves every unit in order (the only one `drawnSpec` takes) at the drawn statement's, whose `nbl` is the files'
(`drawn_same`). -/
theorem drawnSpec_shape {f : RecCoins.Files} {spec' : SessionSpec} {st' : Stmt} {specD : SessionSpec}
    (h : f.drawnSpec spec' st' = .ok specD)
    (hst' : Stmt.setupHidden RecCoins.innerTags f.circuit f.pub Coins.os [] none (some f.partition) none = .ok st')
    (hspec : spec' = st'.spec) : specD.points = 2 ∧ specD.mPts = st'.mPts := by
  unfold RecCoins.Files.drawnSpec RecCoins.drawnSpec at h
  cases hr : RecCoins.recordDraw f.record with
  | none =>
    rw [hr] at h
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h hspec
    exact ⟨rfl, rfl⟩
  | some d =>
    rw [hr] at h
    dsimp only at h
    obtain ⟨D, hD, h⟩ := RecDeploy.CoinTie.tie_bind h
    split at h
    · simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h
    rename_i hp
    try simp only [pure_bind] at h
    try dsimp only at h
    obtain ⟨st'', hst'', h⟩ := RecDeploy.CoinTie.tie_bind h
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    have hp' : D.proved = Array.range st'.n := by simpa using hp
    obtain ⟨-, -, hn, -⟩ := SetupSame.drawn_same hst' hst'' hD hp'
    refine ⟨rfl, ?_⟩
    show st''.mPts = st'.mPts
    rw [(FlockSoundness.Discharge.Hidden.setupHidden_spec' hst'').mPts,
      (FlockSoundness.Discharge.Hidden.setupHidden_spec' hst').mPts, hn]

/-- **Check 5's source at candidate 3's streams is the files, but their inner shape.** -/
theorem atC3_with {g : RecCoins.Files} {inner : Option RecVStar.Inner} {commits : Option RecVStar.InnerCommits}
    {f : RecCoins.Source} (h : RecCoins.Source.atC3 (some (.files g)) inner commits = some f) :
    ∃ inner', f = .files { g with inner := inner' } := by
  unfold RecCoins.Source.atC3 at h
  rcases inner with _ | i <;> rcases commits with _ | cs <;> dsimp only at h <;> cases h
  all_goals first
    | exact ⟨g.inner, rfl⟩
    | (split
       · exact ⟨g.inner, rfl⟩
       · exact ⟨some _, rfl⟩)

end RecDeploy.InnerExtras

namespace RecDeploy.OpenLower

open Flock
open Claims (Msg)
open FlockDeploy (Coin setupOf tabsOf pointsOf)
open Claims.Recursion.VStarB
open RecDeploy.CoinTie
open RecDeploy.Rows (posAt)
open RecDeploy.InnerAt
open RecDeploy.InnerExtras

/-- **`ExtrasR` at a dense compiled statement**, from the verifier's checks: wherever check 5 loads the inner session
at a staged part that shows the coins `c`, its extra claims are the deployed table 0's at `pointsOf` the first coin. -/
theorem staged_extrasR {b : Brings} {Bk : Msg → ℕ} {s : ℕ} {x : Msg} {c : ℕ → Coin} {st0 : Claims.Recursion.CompiledStatement}
    (hst : parse x = some st0) (hd : Claims.Recursion.Form.Dense st0.form) (hB : RestRead.kI x ≤ Bk (xIOf x)) :
    ExtrasR recRead b Bk s x c := by
  intro xO hsh i spec sts hsu _ f inn hf hload v hv
  obtain ⟨r, rs, -, -, -, hr, -, -, -, -, hk⟩ := RestRead.schedBridge ⟨st0, hst, hd⟩
  obtain ⟨r', hr', ha, rfl⟩ := parse_read hst
  rw [hr] at hr'
  cases hr'
  have hins := (setupO_some hsu).1
  obtain ⟨hc, ms, _, _, _, hxO, -, -⟩ := hsh
  subst hxO
  -- the coin server's record, its link points the first coin's
  obtain ⟨xr, hxr, hinner, -⟩ := insO_staged hins
  rw [hr] at hxr
  cases hxr
  obtain ⟨wr, hdw, htr, -, -⟩ := insO_staged_link hins
  obtain ⟨wr', hd', hw⟩ := ins_of_insO hins
  rw [hdw] at hd'
  cases hd'
  have hcm := RecRaw.ins_commits hw
  rw [hinner] at hcm
  obtain ⟨cs, tt, tr, record, -, htx, -, hsrc, hR, -⟩ := commitsOf_tie htr hcm
  obtain ⟨lk, R, -, hlk, hRead, -, -, hRp, -⟩ := FlockC3.VStarList.checkReads_ok hR
  have hb : (b.st.transcript x ms).data.toList = tt.toUTF8.toList := by
    rw [byteArray_toList_eq, toUTF8_of_fromUTF8? htx]
  have hshow : shownOf recRead x (staged b.st x ms s) = some (R.msgs.map fun mc => (mc.1, mc.2.toArray)) := by
    simp only [shownOf, hdw, htr, Option.bind_some, recRead, hb, hRead, Option.map_some]
  obtain ⟨ps, hps, hdrawn⟩ := hc
  rw [hshow] at hps
  cases hps
  have hd0 : Drawn (c 0) R.points.toArray := by
    have hlen : 0 < (R.msgs.map fun mc => (mc.1, mc.2.toArray)).length := by
      simp [Flock.RecRecord.Record.msgs]
    have hd := hdrawn 0 hlen
    have hget : ((R.msgs.map fun mc => (mc.1, mc.2.toArray)).getD 0 ([], #[])).2 = R.points.toArray := by
      simp [Flock.RecRecord.Record.msgs]
    rwa [hget] at hd
  -- the source is the files `x` carries, with the record the transcript was checked against
  obtain ⟨r2, rec, hr2, pj, hpj, hsrc2⟩ := src_staged hins
  rw [hr] at hr2
  cases hr2
  have hf' := hf
  rw [hsrc2] at hf'
  obtain ⟨inner', hfe⟩ := atC3_with hf'
  subst hfe
  have hrec : rec = record := by
    have := atC3_record hf
    rw [hsrc] at this
    exact (Option.some.inj this).symm
  obtain ⟨specL, stL, specD, sess, hstL, hdS, hdec, hext⟩ :=
    load_extras (hload : RecCoins.Inner.load _ _ = .ok inn)
  -- the files' statement, the deployed one as the algebra reads it
  obtain ⟨f0, spec', st', s0, specS, stS, hfs0, hstm, hset', -, -, hex, -, hpts2, hmpts⟩ := innerSetup_same hr ha
  have hf0 : f0 = ⟨r.innerRaw.circuit, r.innerRaw.publicFile, pj, "", none, some (innerFormOf r.innerRaw)⟩ := by
    rw [filesOf, hpj] at hfs0
    simp only [Option.map_some, Option.some.injEq, RecCoins.Source.files.injEq] at hfs0
    exact hfs0.symm
  subst hf0
  have hstL' : RecCoins.Files.stmt "the inner session"
      ⟨r.innerRaw.circuit, r.innerRaw.publicFile, pj, "", none, some (innerFormOf r.innerRaw)⟩ =
        .ok (specL, stL) :=
    stmt_unit (u' := "the inner session") hstL
  rw [hstm] at hstL'
  obtain ⟨hsL, hstLe⟩ := Prod.mk.inj (Except.ok.inj hstL')
  rw [← hsL, ← hstLe] at hdS
  rw [← hstLe] at hext
  obtain ⟨hst', hspec'⟩ := files_stmt hstm
  obtain ⟨hDp, hDm⟩ := drawnSpec_shape hdS hst' hspec'
  -- the decoded link points are the coin server's, so the first coin's
  have hdec' : Record.decode specD record = .ok sess := by rw [← hrec]; exact hdec
  obtain ⟨j, l, -, hj, hl, -⟩ := Claims.Bridge.Custody.decode_ok hdec'
  obtain ⟨rp, hrp, hh⟩ := innerLink_points hj (fld_val hl) hlk
  have hpe : sess.points = lk.points := held_eq hdec' hj (fld_val hl) hrp hh
  obtain ⟨hps1, hps2⟩ := FlockSoundness.Refine.decode_points hdec'
  have hP : sess.points = pointsOf specS (c 0) := by
    show sess.points = (Array.range specS.points).map fun p => (Array.range specS.mPts).map fun i =>
      (c 0).at (p * specS.mPts + i)
    refine points_of_flat (0 : F128) (by rw [hps1, hDp, hpts2]) (fun p hp => by rw [hps2 p hp, hDm, hmpts]) ?_
    have h3 := hd0
    unfold Drawn FlockDeploy.Coin.vals at h3
    rw [hRp, ← hpe] at h3
    exact h3
  -- the deployed table's extra claims at those points
  have hxI : xIOf x = r.w.inner := by unfold xIOf; rw [hst]; rfl
  rw [hxI, hset'] at hv
  cases hv
  have htabs : tabsOf #[stS] = #[Setup.ofCircuit stS] := by simp [tabsOf]
  dsimp only
  rw [headD_range c (B := Bk (xIOf x)) (by omega), ← hP, hext, htabs]
  show st'.extraClaims sess.points = stS.extraClaims sess.points
  rw [hex]

/-- **`InnerCoinsR` at a dense compiled statement**, from the verifier's checks alone. -/
theorem innerCoinsR_staged {b : Brings} {Bk : Msg → ℕ} {s : ℕ} {x : Msg} {c : ℕ → Coin}
    {st0 : Claims.Recursion.CompiledStatement} (hst : parse x = some st0) (hd : Claims.Recursion.Form.Dense st0.form)
    (hB : RestRead.kI x ≤ Bk (xIOf x)) : InnerCoinsR recRead b Bk s x c :=
  innerCoinsR_of hst hd hB (staged_extrasR hst hd hB)

/-- **`PosTieR` at row D's position reader, at a dense compiled statement**, from the verifier's checks and the
pieces' ranges. -/
theorem posTieR_dense {b : Brings} {pieces : Msg → List OpenPiece} {Bk : Msg → ℕ} {s : ℕ} {x : Msg}
    {c : ℕ → Coin} {st0 : Claims.Recursion.CompiledStatement} (hst : parse x = some st0)
    (hd : Claims.Recursion.Form.Dense st0.form) (hR : PiecesInRangeR pieces) (hB : RestRead.kI x ≤ Bk (xIOf x)) :
    PosTieR recRead b pieces posAt Bk s x c :=
  posTieR_staged hst hd hR hB (staged_extrasR hst hd hB)

/-- **Check 5's comb is the deployed table's circuit fold**: wherever check 5 loads the inner session at a staged part,
its comb at any rep's fold coins is table 0's `CircuitFold.folded` at them. -/
theorem staged_comb {b : Brings} {s : ℕ} {x xO : Msg} {c : ℕ → Coin} {st0 : Claims.Recursion.CompiledStatement}
    (hst : parse x = some st0) (hsh : showsPart recRead b s x xO c) {i : RecIns} (hins : insO xO = some i)
    {f : RecCoins.Source} {unit : String} {inn : RecCoins.Inner}
    (hf : RecCoins.Source.atC3 i.src i.inner i.commits = some f) (hload : f.load unit = .ok inn) :
    ∃ v, setupOf false (xIOf x) = some v ∧
      inn.combOf = fun _ info => (tabsOf v.2.2)[0]!.fold.folded info.lc.fold.alpha info.lc.fold.beta
        info.lc.fold.lw info.lc.fold.rhoIn info.lc.fold.ts := by
  obtain ⟨r, hr, ha, rfl⟩ := parse_read hst
  obtain ⟨ms, _, _, _, hxO, -, -⟩ := hsh.2
  subst hxO
  obtain ⟨r2, rec, hr2, pj, hpj, hsrc⟩ := src_staged hins
  rw [hr] at hr2
  cases hr2
  rw [hsrc] at hf
  obtain ⟨f', rfl, hfs⟩ := atC3_files hf
  have hload' : RecCoins.Inner.load unit f' = .ok inn := hload
  obtain ⟨specL, stL, hstL, -, -, hcomb⟩ := load_fields hload'
  obtain ⟨f0, spec', st', s', spec, st, hfs0, hstm, hset, -, -, -, hfold, -, -⟩ := innerSetup_same hr ha
  have hf0 : f0 = ⟨r.innerRaw.circuit, r.innerRaw.publicFile, pj, "", none, some (innerFormOf r.innerRaw)⟩ := by
    rw [filesOf, hpj] at hfs0
    simp only [Option.map_some, Option.some.injEq, RecCoins.Source.files.injEq] at hfs0
    exact hfs0.symm
  subst hf0
  have hstL' : RecCoins.Files.stmt "the inner session"
      ⟨r.innerRaw.circuit, r.innerRaw.publicFile, pj, "", none, some (innerFormOf r.innerRaw)⟩ =
        .ok (specL, stL) :=
    stmt_unit (u := unit) (by rw [← hstL, hfs unit]; rfl)
  rw [hstm] at hstL'
  obtain ⟨-, hstLe⟩ := Prod.mk.inj (Except.ok.inj hstL')
  have hxI : xIOf x = r.w.inner := by unfold xIOf; rw [hst]; rfl
  refine ⟨(s', spec, #[st]), by rw [hxI]; exact hset, ?_⟩
  rw [hcomb]
  have htabs : tabsOf #[st] = #[Setup.ofCircuit st] := by simp [tabsOf]
  dsimp only
  rw [htabs]
  funext u info
  show stL.circuitFold.folded _ _ _ _ _ = st.circuitFold.folded _ _ _ _ _
  rw [hfold, hstLe]

/-- **The inner session check 5 loads is the deployed setup's, as its algebra reads it**: at a dense compiled
statement, wherever check 5 loads it at a staged part that shows the coins `c`, its extra claims are table 0's at
`pointsOf` the first coin (`staged_extrasR`) and its comb is table 0's circuit fold (`staged_comb`). -/
theorem staged_bridges {b : Brings} {Bk : Msg → ℕ} {s : ℕ} {x : Msg} {c : ℕ → Coin} {st0 : Claims.Recursion.CompiledStatement}
    (hst : parse x = some st0) (hd : Claims.Recursion.Form.Dense st0.form) (hB : RestRead.kI x ≤ Bk (xIOf x)) :
    ExtrasR recRead b Bk s x c ∧
      ∀ xO, showsPart recRead b s x xO c → ∀ i, insO xO = some i → ∀ (f : RecCoins.Source) unit inn,
        RecCoins.Source.atC3 i.src i.inner i.commits = some f → f.load unit = .ok inn →
        ∃ v, setupOf false (xIOf x) = some v ∧
          inn.combOf = fun _ info => (tabsOf v.2.2)[0]!.fold.folded info.lc.fold.alpha info.lc.fold.beta
            info.lc.fold.lw info.lc.fold.rhoIn info.lc.fold.ts :=
  ⟨staged_extrasR hst hd hB, fun _ hsh _ hins _ _ _ hf hload => staged_comb hst hsh hins hf hload⟩

end RecDeploy.OpenLower
