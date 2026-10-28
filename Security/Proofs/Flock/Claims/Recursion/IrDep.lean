import Proofs.Flock.Claims.Recursion.VStarTopsOk
import Proofs.Flock.Claims.Recursion.SideCapPos
import Proofs.Flock.Claims.Complete.Honest

/-!
# The deployed reference inner prover (`IrDep`) and its openings' shape (`irShaped_dep`)

`IrShaped Ir x w ρ hist` is what the stager reads of an inner prover's `rest`. `IrDep x` is the deployed one, the inner
recursive prover `rec_live` runs with C-Flock's prover behind it, as V\*'s stager reads it:

* **Its tape** (`DepTape`): C-Flock's proofs at the coins so far (`prove`: each rep's proof and its batched value `T`), the
  running sums' salts (`rec_vstage.acc_salts`) and the fresh salts of the openings' private rows.
* **Its rows** (`rows`): round 0's, each table's level-0 cap nodes (`capRows`, `nodeRow`, `rec_live.top_rows`), then each
  rep's rounds' rows along its schedule (`repRows`), at the proofs at the coins so far. The proofs' frames are causal, so
  round `i`'s rows read only what the coins before round `i` fix.
* **Its rest** (`depRest`, `rec_vstar.queries`, `rec_vstage`, `verifier_rows`): each rep's openings at each level of table
  0's schedule (`schAt`), in rep, level and query order, with the position the verifier's rows sample (`Rows.posAt`),
  the opened row (`openAt`) and its siblings (`sibsOf`), each fitted to its level's `2^lanes` elements and `d - cap`
  siblings as an honest proof has them; each rep's κ, eq(α) at each level's queries and v at its coins
  (`verifierRows`, v with the comb at the lincheck's fold points).

`IrDepSession` evaluates `IrDep` at a deployed inner session, with the statement's setup read once (`setupB`, `rowsV`,
`depRestV`: `setupB_eq`, `irDep_rows`, `irDep_rest`), and `test_irdep_difftest.py` holds it to the deployed builder's
rows, openings, κ, eq(α), v and running sums' salts there.

`irShaped_dep` proves `IrShaped (IrDep x)` at every dense statement, every tape and every history holding both reps'
stream rounds, where the reps' tops differ at each later level (the tape exception, as `hok_honest` takes it):
both reps open level 0's queries (`q_ne_zero`), level `l`'s openings are its `2q` (`levelMatch`,
`depOpenings_level`), and each opening's top node is among the level's tops: its position is below `2^d`
(`posAt_lt`), so past its `d - cap` siblings below `2^cap`, the level's nodes a rep, and the level's tops are both
reps' one tree at level 0 (`topsAt0_fit`) and rep 0's nodes then rep 1's above (`topsAtL_fit`).
-/

namespace RecDeploy.VStarBuild

open _root_.Flock
open Claims (Msg)
open FlockDeploy (Coin setupOf tabsOf pointsOf repCtx capRows repRows nodeRow)

/-! ## The prover -/

/-- An inner round's coin, from the answer the verifier sent (`Coin.bytes`, decoded as `FlockComplete.coinOf`). -/
def coinAt (a : Msg) : Coin := fun i => FlockComplete.coinOf a i.val

/-- The coins of the inner rounds `h`, in order. -/
def innerCoins (h : List (Msg × Msg)) : List Coin := h.map fun p => coinAt p.2

/-- **The deployed inner prover's tape**: C-Flock's proofs at the coins so far, each rep's proof and its batched value
`T` (`prove`), the running sums' salts, and the fresh salts of statement directory `dir`, port `port`, row `i`. -/
structure DepTape where
  prove : List Coin → Array (_root_.Flock.Proof × F128)
  accSalts : Array ByteArray
  fresh : String → String → ℕ → ByteArray

instance instNonemptyDepTape : Nonempty DepTape := ⟨⟨fun _ => #[], #[], fun _ _ _ => ByteArray.empty⟩⟩

/-- Each table's level-0 cap, its first rep's proof's. -/
def depCaps (sts : Array _root_.Flock.Stmt) (ps : Array (_root_.Flock.Proof × F128)) : Array (Array ByteArray) :=
  (Array.range sts.size).map fun t => (ps.getD (2 * t) default).1.cap

/-- **The honest rows, round by round**: round 0's each table's level-0 cap nodes, then each rep's rounds', rep by rep,
at the proofs `ps` and the coins `cs` (the link's first). -/
def depRows (spec : _root_.Flock.SessionSpec) (sts : Array _root_.Flock.Stmt) (ps : Array (_root_.Flock.Proof × F128))
    (cs : List Coin) : List (List ByteArray) :=
  ((capRows (depCaps sts ps)).map nodeRow) :: ((List.range (2 * sts.size)).map fun k =>
    repRows (repCtx spec sts (cs.headD default) (capRows (depCaps sts ps)) cs.tail k) (ps.getD k default).1
      (ps.getD k default).2).flatten

/-- **Rep `k`'s verifier rows** at the inner statement `xI` and the inner rounds' coins `cs` (the link's first): as
`Rows.posAt` runs them, at table `k / 2`'s shape and extra claims, rep `k`'s `canonRounds` and the comb `comb`. -/
def verRowsAt (xI : Msg) (cs : List Coin) (k : ℕ) (comb : Array F128) :
    Option (Array F128 × RecAlgebra.Info RecAlgebra.Lin × Array (Array (Array F128))) :=
  match setupOf false xI with
  | some v =>
    (RecAlgebra.verifierRows
      (FlockSoundness.Refine.shapeOf (tabsOf v.2.2)[k / 2]! ((fast100 (tabsOf v.2.2)[k / 2]!.m).getD default)
        (((tabsOf v.2.2)[k / 2]!.extra (pointsOf v.2.1 (cs.headD default))).toOption.getD #[]).size)
      (Rows.canonRounds v.2.1 v.2.2 cs.tail k)
      (((tabsOf v.2.2)[k / 2]!.extra (pointsOf v.2.1 (cs.headD default))).toOption.getD #[]) comb).toOption
  | none => none

/-- **Rep `k`'s comb**: the lincheck's fold at its fold points (`RecCoins.Inner.comb`); zero where none. -/
def combAt (xI : Msg) (cs : List Coin) (k : ℕ) : Array F128 :=
  (match setupOf false xI, verRowsAt xI cs k RecCoins.ZERO_COMB with
    | some v, some (_, info, _) =>
      ((tabsOf v.2.2)[k / 2]!.fold.folded info.lc.fold.alpha info.lc.fold.beta info.lc.fold.lw info.lc.fold.rhoIn
        info.lc.fold.ts).toOption
    | _, _ => none).getD RecCoins.ZERO_COMB

/-- `n` entries of `a`, `d` past its end. -/
def fitArr {α : Type} (n : ℕ) (d : α) (a : Array α) : Array α := (Array.range n).map fun i => a.getD i d

/-- **Rep `rep`'s opening `j` of level `l`** (`rec_vstar.queries`): the position the verifier's rows sample, the proof's
row and siblings at level `l` (`openAt`, `sibsOf`), fitted to the level's lanes and siblings below its cap. -/
def depOpen (xI : Msg) (cs : List Coin) (ps : Array (_root_.Flock.Proof × F128)) (rep l j : ℕ) :
    _root_.VStar.In.Opening :=
  let lv := (OpenLower.schAt xI 0).levels[l]!
  let o := FlockSoundness.Refine.openAt (ps.getD rep default).1.ligerito ((OpenLower.schAt xI 0).levels.size - 1) l
  { rep, level := l, j,
    pos := Rows.posAt xI (cs ++ [default]) ⟨0, rep, l, j, (0, 0, 0, 0)⟩,
    row := fitArr (2 ^ lv.lanes) 0 (o.1.getD j #[]),
    sibs := fitArr (lv.d - lv.cap) ByteArray.empty (_root_.Flock.sibsOf o.2.1 (lv.d - lv.cap) j).toArray }

/-- **Both reps' openings**, in rep, level and query order. -/
def depOpenings (xI : Msg) (cs : List Coin) (ps : Array (_root_.Flock.Proof × F128)) :
    Array _root_.VStar.In.Opening :=
  ((List.range 2).flatMap fun rep => (List.range (OpenLower.schAt xI 0).levels.size).flatMap fun l =>
    (List.range ((OpenLower.schAt xI 0).levels[l]!).q).map (depOpen xI cs ps rep l)).toArray

/-- **What the stager reads besides the rows**, at the coins `cs`: the openings, the running sums' salts, each rep's
κ and eq(α) by level and its v (`verifierRows`), the fresh salts. The message rounds and tops are `srcOf`'s. -/
def depRest (xI : Msg) (τ : DepTape) (cs : List Coin) : _root_.VStar.In.Src where
  msgs := #[]
  openings := depOpenings xI cs (τ.prove cs)
  tops := #[]
  accSalts := τ.accSalts
  kappa := (Array.range 2).map fun k => ((verRowsAt xI cs k RecCoins.ZERO_COMB).map (·.2.2)).getD #[]
  eqA := (Array.range 2).map fun k =>
    ((verRowsAt xI cs k RecCoins.ZERO_COMB).map fun t =>
      t.2.1.lg.conss.map fun c => (eqTable c.alpha).extract 0 c.pos.size).getD #[]
  vs := (Array.range 2).map fun k => ((verRowsAt xI cs k (combAt xI cs k)).map (·.1)).getD #[]
  fresh := τ.fresh

/-- **The deployed reference inner prover** at the compiled statement `x`: at the inner rounds `h`, the honest rows of
round `h.length` (`depRows`) at the proofs at `h`'s coins, as u16 words, and `depRest` at `h`'s coins. -/
def IrDep (x : Msg) : InnerRec where
  R := DepTape
  rows _ τ h :=
    match setupOf false (xIOf x) with
    | some v => ((depRows v.2.1 v.2.2 (τ.prove (innerCoins h)) (innerCoins h)).getD h.length []).map
        _root_.VStar.In.wordsOf
    | none => []
  rest _ τ h := depRest (xIOf x) τ (innerCoins h)

/-! ## The openings -/

theorem mem_depOpen {xI : Msg} {cs : List Coin} {ps : Array (_root_.Flock.Proof × F128)} {rep l j : ℕ}
    (hrep : rep < 2) (hl : l < (OpenLower.schAt xI 0).levels.size) (hj : j < ((OpenLower.schAt xI 0).levels[l]!).q) :
    depOpen xI cs ps rep l j ∈ depOpenings xI cs ps := by
  rw [depOpenings, List.mem_toArray]
  exact List.mem_flatMap.2 ⟨rep, List.mem_range.2 hrep,
    List.mem_flatMap.2 ⟨l, List.mem_range.2 hl, List.mem_map.2 ⟨j, List.mem_range.2 hj, rfl⟩⟩⟩

/-- **An opening of `depOpenings`** is one of the two reps', at one of the schedule's levels, at the position the
verifier's rows sample, of the level's lanes and siblings below its cap. -/
theorem mem_depOpenings {xI : Msg} {cs : List Coin} {ps : Array (_root_.Flock.Proof × F128)}
    {o : _root_.VStar.In.Opening} (h : o ∈ depOpenings xI cs ps) :
    o.rep < 2 ∧ o.level < (OpenLower.schAt xI 0).levels.size ∧
      o.pos = Rows.posAt xI (cs ++ [default]) ⟨0, o.rep, o.level, o.j, (0, 0, 0, 0)⟩ ∧
      o.row.size = 2 ^ ((OpenLower.schAt xI 0).levels[o.level]!).lanes ∧
      o.sibs.size = ((OpenLower.schAt xI 0).levels[o.level]!).d - ((OpenLower.schAt xI 0).levels[o.level]!).cap := by
  rw [depOpenings, List.mem_toArray] at h
  obtain ⟨rep, hrep, h⟩ := List.mem_flatMap.1 h
  obtain ⟨l, hl, h⟩ := List.mem_flatMap.1 h
  obtain ⟨j, -, rfl⟩ := List.mem_map.1 h
  exact ⟨List.mem_range.1 hrep, List.mem_range.1 hl, rfl, by simp [depOpen, fitArr], by simp [depOpen, fitArr]⟩

theorem length_filter_flatMap {α β : Type} (p : β → Bool) (f : α → List β) :
    ∀ l : List α, ((l.flatMap f).filter p).length = (l.map fun a => ((f a).filter p).length).sum
  | [] => rfl
  | a :: l => by
    rw [List.flatMap_cons, List.filter_append, List.length_append, length_filter_flatMap p f l, List.map_cons,
      List.sum_cons]

theorem sum_range_ite (f : ℕ → ℕ) (l0 : ℕ) :
    ∀ n, ((List.range n).map fun l => if l = l0 then f l else 0).sum = if l0 < n then f l0 else 0
  | 0 => by simp
  | n + 1 => by
    rw [List.range_succ, List.map_append, List.sum_append, sum_range_ite f l0 n]
    by_cases h : l0 < n
    · simp [h, show l0 < n + 1 by omega, show n ≠ l0 by omega]
    · by_cases h' : n = l0
      · subst h'
        simp
      · simp [h, h', show ¬ l0 < n + 1 by omega]

/-- **Level `l`'s openings are its two reps' queries.** -/
theorem depOpenings_level (xI : Msg) (cs : List Coin) (ps : Array (_root_.Flock.Proof × F128)) (l : ℕ) :
    ((depOpenings xI cs ps).filter (·.level == l)).size = 2 * ((OpenLower.schAt xI 0).levels[l]!).q := by
  have hrow : ∀ rep, (((List.range (OpenLower.schAt xI 0).levels.size).flatMap fun l' =>
      (List.range ((OpenLower.schAt xI 0).levels[l']!).q).map (depOpen xI cs ps rep l')).filter
        (·.level == l)).length =
      if l < (OpenLower.schAt xI 0).levels.size then ((OpenLower.schAt xI 0).levels[l]!).q else 0 := by
    intro rep
    rw [length_filter_flatMap]
    refine Eq.trans ?_ (sum_range_ite (fun l' => ((OpenLower.schAt xI 0).levels[l']!).q) l _)
    congr 1
    apply List.map_congr_left
    intro l' _
    rw [List.filter_map, List.length_map]
    by_cases h : l' = l
    · subst h
      rw [if_pos rfl, List.filter_eq_self.2 (fun j _ => by simp [Function.comp_def, depOpen]), List.length_range]
    · rw [if_neg h, List.filter_eq_nil_iff.2 (fun j _ => by simp [Function.comp_def, depOpen, h])]
      rfl
  rw [depOpenings, List.filter_toArray, List.size_toArray, length_filter_flatMap]
  simp only [hrow]
  have e : List.range 2 = [0, 1] := rfl
  rw [e]
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil]
  by_cases hl : l < (OpenLower.schAt xI 0).levels.size
  · rw [if_pos hl]
    omega
  · rw [if_neg hl, getElem!_neg _ l hl]
    rfl

/-! ## The levels' tops at the fitted records -/

/-- **A cap's `capIn` holds its level's rows to its `N`.** -/
theorem capIn_rows {src : _root_.VStar.In.Src} {l N : ℕ} {pin : _root_.VStar.In.PartIn} {T : _root_.VStar.In.Tops}
    (h : _root_.VStar.In.capIn src l N = .ok pin) (hT : _root_.VStar.In.topsAt src l = .ok T) : T.rows.size = N := by
  unfold _root_.VStar.In.capIn at h
  rw [hT] at h
  by_contra hne
  simp [bind, Except.bind, hne] at h

/-- **Level 0's tops at the fitted records**: both reps' are round 0's `N` nodes, one run (`topsAt_same`), so each
rep's first row is `0`, and RecCap holds the level's rows to its `N`. -/
theorem topsAt0_fit {I : InnerRec} {x : Msg} {w : Claims.Flock.Wit} {ρ : (I.fit x).Tape} {hist : List (Msg × Msg)}
    {r : XR} (hr : readX x = some r) {rs : Array RecVStar.SchedRound} (hrs : repSched r.shape = some rs) {N : ℕ}
    (hcap : capAt r = some (nS x - 1, N)) {src : _root_.VStar.In.Src} (hs : srcOf (I.fit x) x w ρ hist = some src)
    {T : _root_.VStar.In.Tops} (hT : _root_.VStar.In.topsAt src 0 = .ok T) : T.first = #[0, 0] ∧ T.rows.size = N := by
  obtain ⟨pin, hpin⟩ := capIn_zero_honest hr hrs hs hcap
  refine ⟨?_, capIn_rows hpin hT⟩
  have hst := RecDeploy.OpenSlots.capAt_stmt hr hcap
  obtain ⟨vs, hvs, -, hall⟩ := readX_partVStar hr
  rw [hvs] at hst
  simp only [Except.toOption, Option.bind_some] at hst
  obtain ⟨hsz, he⟩ := Array.getElem?_eq_some_iff.1 hst
  obtain ⟨-, p, c, -, hp, hc, -, hcp, -⟩ := hall _ hsz
  rw [he] at hcp
  obtain ⟨p0, p1, hpp, h0, hw, -⟩ := cap_ports hcp
  have hrr := roundRows_cap0 (I := I) (w := w) (ρ := ρ) (hist := hist) hr hcap (by rw [hp, Option.bind_some, hc])
    hpp h0 hw
  have hnode : (roundRows (I.fit x) w ρ hist 0).map
        (fun (rw : _root_.VStar.In.Row) => (wordsBytes (rw.words.extract 0 32), rw.salt)) =
      (List.range N).map fun n => (wordsBytes (fitRow 32 ((I.rows w ρ.1 []).getD n #[])), saltBytes (ρ.2 0 n)) := by
    rw [hrr, List.map_map]
    apply List.map_congr_left
    intro n _
    simp only [Function.comp_apply, extract_top]
  have hsrc := srcOf_eq hr hrs hs
  subst hsrc
  have ht2 : (topsOf rs (roundRows (I.fit x) w ρ hist)).size = 2 := by simp [topsOf]
  have ht0 : ((topsOf rs (roundRows (I.fit x) w ρ hist))[0]!)[0]? = some ((List.range N).map fun n =>
      (wordsBytes (fitRow 32 ((I.rows w ρ.1 []).getD n #[])), saltBytes (ρ.2 0 n))).toArray := by
    rw [← hnode]
    rfl
  have ht1 : ((topsOf rs (roundRows (I.fit x) w ρ hist))[1]!)[0]? = some ((List.range N).map fun n =>
      (wordsBytes (fitRow 32 ((I.rows w ρ.1 []).getD n #[])), saltBytes (ρ.2 0 n))).toArray := by
    rw [← hnode]
    rfl
  have ha : ((List.range N).map fun n =>
      (wordsBytes (fitRow 32 ((I.rows w ρ.1 []).getD n #[])), saltBytes (ρ.2 0 n))).toArray.all
        (·.1.size == 64) = true := by
    simp [wordsBytes_size, fitRow_size]
  have h := topsAt_same (src := { (I.fit x).rest w ρ.1 hist with
      msgs := msgsOf rs (roundRows (I.fit x) w ρ hist), tops := topsOf rs (roundRows (I.fit x) w ρ hist) })
    ht2 ht0 ht1 ha
  exact congrArg _root_.VStar.In.Tops.first (Except.ok.inj (hT.symm.trans h))

/-- **Level `g + 1`'s tops at the fitted records**, where the history holds both reps' stream rounds and the reps'
tops there differ: rep 0's `n` nodes, then rep 1's (`topsAt_two`), so rep 1's first row is `n`, and the level's cap
holds the level's rows to its `2n`. -/
theorem topsAtL_fit {I : InnerRec} {x : Msg} {w : Claims.Flock.Wit} {ρ : (I.fit x).Tape} {hist : List (Msg × Msg)}
    {r : XR} (hr : readX x = some r) {rs : Array RecVStar.SchedRound} (hrs : repSched r.shape = some rs)
    (hH : 2 * rs.size ≤ hist.length) {src : _root_.VStar.In.Src} (hs : srcOf (I.fit x) x w ρ hist = some src)
    {g n : ℕ} (hn : (RecVStar.topSizes rs)[g]? = some n)
    (hd : ∀ a b, (src.tops[0]!)[g + 1]? = some a → (src.tops[1]!)[g + 1]? = some b →
      _root_.VStar.In.sameTop a b = false)
    {T : _root_.VStar.In.Tops} (hT : _root_.VStar.In.topsAt src (g + 1) = .ok T) :
    T.first = #[0, n] ∧ T.rows.size = 2 * n := by
  obtain ⟨pin, hpin⟩ := capIn_later_honest hr hrs hH hs hn hd
  refine ⟨?_, capIn_rows hpin hT⟩
  obtain ⟨hg, -⟩ := List.getElem?_eq_some_iff.1 hn
  obtain ⟨k, e, hk, he, rfl⟩ := topSizes_split rs hg
  rw [topSizes_at hk he] at hn
  cases hn
  obtain ⟨c, hc, hcp⟩ := capPart_parsed hr hrs (topSizes_at hk he)
  obtain ⟨p0, p1, hp, h0, hw, -⟩ := cap_ports hcp
  have hmod : ∀ rep, (rep * rs.size + k) % rs.size = k := fun rep => by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hk]
  have hrows : ∀ rep < 2,
      ((roundRows (I.fit x) w ρ hist (rep * rs.size + k + 1)).drop ((rs.getD k default).2.1.toList.take e).sum).take
        ((rs.getD k default).2.1.toList.getD e 0) =
      (List.range (((rs.getD k default).2.1).getD e 0)).map
        (capRow I w ρ hist ((rs.getD k default).2.1.toList.take e).sum (rep * rs.size + k)) := by
    intro rep hrep
    have h := roundRows_cap_take (I := I) (w := w) (ρ := ρ) (hist := hist) (j := rep * rs.size + k)
      (e := e) hr hrs (by
        have : rep * rs.size + k < 2 * rs.size := by
          rcases (show rep = 0 ∨ rep = 1 by omega) with rfl | rfl <;> omega
        omega) (by rw [hmod]; exact he) (by rw [hmod]; simpa only [Nat.add_assoc] using hc) hp h0 hw
    simpa only [hmod] using h
  have htop : ∀ rep < 2,
      ((topsOf rs (roundRows (I.fit x) w ρ hist))[rep]!)[topsBefore rs k + e + 1]? =
        some ((List.range (((rs.getD k default).2.1).getD e 0)).map fun t =>
          (wordsBytes (fitRow 32 ((I.rows w ρ.1 (hist.take (rep * rs.size + k + 1))).getD
            (((rs.getD k default).2.1.toList.take e).sum + t) #[])),
           saltBytes (ρ.2 (rep * rs.size + k + 1) (((rs.getD k default).2.1.toList.take e).sum + t)))).toArray := by
    intro rep hrep
    rw [topsOf_level rs _ hrep hk he, hrows rep hrep, List.map_map]
    refine congrArg (fun l => some l.toArray) (List.map_congr_left fun t _ => ?_)
    simp only [Function.comp_apply, capRow, extract_top]
  have hall : ∀ rep, ((List.range (((rs.getD k default).2.1).getD e 0)).map fun t =>
      (wordsBytes (fitRow 32 ((I.rows w ρ.1 (hist.take (rep * rs.size + k + 1))).getD
        (((rs.getD k default).2.1.toList.take e).sum + t) #[])),
       saltBytes (ρ.2 (rep * rs.size + k + 1)
        (((rs.getD k default).2.1.toList.take e).sum + t)))).toArray.all (·.1.size == 64) = true := by
    intro rep
    simp [wordsBytes_size, fitRow_size]
  have hsrc := srcOf_eq hr hrs hs
  subst hsrc
  have ht2 : (topsOf rs (roundRows (I.fit x) w ρ hist)).size = 2 := by simp [topsOf]
  have h := topsAt_two (src := { (I.fit x).rest w ρ.1 hist with
      msgs := msgsOf rs (roundRows (I.fit x) w ρ hist), tops := topsOf rs (roundRows (I.fit x) w ρ hist) })
    ht2 (htop 0 (by omega)) (htop 1 (by omega)) (hall 0) (hall 1)
    (by rw [sameTop_comm]; exact hd _ _ (htop 0 (by omega)) (htop 1 (by omega)))
  rw [congrArg _root_.VStar.In.Tops.first (Except.ok.inj (hT.symm.trans h))]
  have he' : e < (rs[k]?.getD default).2.1.size := by simpa using he
  simp [Array.getElem?_eq_getElem he']

/-! ## `IrShaped` at the deployed prover -/

theorem pair_getElem!_zero (a b : ℕ) : (#[a, b] : Array ℕ)[0]! = a := rfl

theorem pair_getElem!_one (a b : ℕ) : (#[a, b] : Array ℕ)[1]! = b := rfl

/-- **A dense statement's inner `m` has a `fast100` schedule**, table 0's at the inner statement (`schAt`). -/
theorem fast100_dense {x : Msg} {st : Claims.Recursion.Stmt} (hx : parse x = some st)
    (hf : Claims.Recursion.Form.Dense st.form) {r : XR} (hr : readX x = some r) :
    ∃ sch, fast100 r.shape.m = some sch ∧ OpenLower.schAt (xIOf x) 0 = sch := by
  obtain ⟨r', hr', -, rfl⟩ := parse_read hx
  rw [hr] at hr'
  cases hr'
  obtain ⟨s', spec, t, hset, htm⟩ := OpenLower.inner_m hx hr
  have hh : r.shape.holo = false := hf.1
  have hsp : r.shape.sparse = none := by
    have h2 : (r.shape.sparse.map fun sp => (sp.nu, sp.ell, sp.sLog, sp.tLog)) = none := hf.2
    cases hs : r.shape.sparse with
    | none => rfl
    | some _ => simp [hs] at h2
  obtain ⟨vs, hvs, -⟩ := readX_statements hr
  obtain ⟨A, K, n, hA, -, -⟩ := OpenLower.statements_dense hvs hsp
  rw [hh] at hA
  obtain ⟨sch, sp, ps, hsch, -, -, -⟩ := FlockVBridge.Family.statementsOf_eq hA
  have hfs : fast100 r.shape.m = some sch := FlockVBridge.Family.sched_fast100 hsch
  refine ⟨sch, hfs, ?_⟩
  unfold OpenLower.schAt
  rw [hset]
  have htabs : tabsOf #[t] = #[_root_.Flock.Setup.ofCircuit t] := by simp [tabsOf]
  have h0 : (tabsOf #[t])[0]!.m = t.m := by
    rw [htabs]
    rfl
  dsimp only
  rw [h0, htm, hfs]
  rfl

/-- **The deployed inner prover's openings fit V\*'s levels** (`IrShaped`) at every dense statement, tape and history
holding both reps' stream rounds, where the reps' tops differ at each later level (`hTops`, the tape exception). -/
theorem irShaped_dep {x : Msg} {st : Claims.Recursion.Stmt} (hx : parse x = some st)
    (hf : Claims.Recursion.Form.Dense st.form) {w : Claims.Flock.Wit} {ρ : ((IrDep x).fit x).Tape}
    {hist : List (Msg × Msg)}
    (hH : ∀ r rs, readX x = some r → repSched r.shape = some rs → 2 * rs.size ≤ hist.length)
    (hTops : ∀ src, srcOf ((IrDep x).fit x) x w ρ hist = some src → ∀ l, 0 < l → ∀ a b,
      (src.tops[0]!)[l]? = some a → (src.tops[1]!)[l]? = some b → _root_.VStar.In.sameTop a b = false) :
    IrShaped (IrDep x) x w ρ hist := by
  obtain ⟨r, hr, -⟩ := sparse_none_of_dense hx hf
  obtain ⟨sch, hsch, hsa⟩ := fast100_dense hx hf hr
  obtain ⟨s', spec, t, hset, htm⟩ := OpenLower.inner_m hx hr
  have h2 : 2 ≤ sch.levels.size := (FlockSoundness.Refine.fast100_dims _ _ hsch).1.2.1
  have h0 : 0 < sch.levels.size := by omega
  have hg : sch.levels.getD 0 default = sch.levels[0]! := by
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem h0, Option.getD_some, getElem!_pos sch.levels 0 h0]
  have hq0 : 0 < (sch.levels[0]!).q := by
    have h := FlockDeploy.q_ne_zero hsch 0 h0
    rw [hg] at h
    exact Nat.pos_of_ne_zero h
  have hOp : (((IrDep x).fit x).rest w ρ.1 hist).openings =
      depOpenings (xIOf x) (innerCoins hist) (DepTape.prove ρ.1 (innerCoins hist)) := rfl
  unfold IrShaped
  rw [hOp]
  refine ⟨⟨_, mem_depOpen (rep := 0) (l := 0) (j := 0) (by omega) (by rw [hsa]; exact h0) (by rw [hsa]; exact hq0),
      rfl⟩, ⟨_, mem_depOpen (rep := 1) (l := 0) (j := 0) (by omega) (by rw [hsa]; exact h0)
      (by rw [hsa]; exact hq0), rfl⟩, ?_⟩
  intro s l L H n c hp
  obtain ⟨vs, hvs, -⟩ := readX_statements hr
  have hst := (partAt_some hr hp).1
  rw [hvs] at hst
  simp only [Except.toOption, Option.bind_some] at hst
  obtain ⟨hls, hL, hHH, hn⟩ := OpenLower.levelMatch hx hf hr hvs hst
  rw [hsa] at hL hHH hn
  refine ⟨?_, fun q hq hql => ?_, fun src hs T hT q hq hql => ?_⟩
  · rw [hn, depOpenings_level, hsa, hls]
  · obtain ⟨-, -, -, hrow, hsib⟩ := mem_depOpenings hq
    rw [hsa, hql, hls] at hrow hsib
    exact ⟨hrow.trans hL.symm, hsib.trans hHH.symm⟩
  · obtain ⟨hrep, hlev, hpos, -, -⟩ := mem_depOpenings hq
    rw [hsa] at hlev
    have htabs : tabsOf #[t] = #[_root_.Flock.Setup.ofCircuit t] := by simp [tabsOf]
    have ht0 : (tabsOf #[t])[0]!.m = t.m := by
      rw [htabs]
      rfl
    have hf' : fast100 (tabsOf #[t])[0]!.m = some sch := by rw [ht0, htm, hsch]
    have hp' : q.pos < 2 ^ (sch.levels[s]!).d := by
      rw [hpos, ← hls, ← hql]
      exact CapLower.posAt_lt (v := (s', spec, #[t])) hset hf' hlev
    have hsh : q.pos / 2 ^ H < 2 ^ (sch.levels[s]!).cap := by
      rw [hHH, ← Nat.shiftRight_eq_div_pow]
      exact CapLower.shift_lt hp'
    rw [← hls] at hsh
    obtain ⟨r', rs, hr', hrs⟩ := srcOf_read hs
    rw [hr] at hr'
    cases hr'
    have hq2 : q.rep = 0 ∨ q.rep = 1 := by omega
    generalize q.pos / 2 ^ H = p at hsh ⊢
    rcases recOpenLevel_of_partAt hr hrs hp with rfl | ⟨g, m, hm, rfl⟩
    · obtain ⟨hfirst, hrows⟩ := topsAt0_fit hr hrs (RestRead.capAt_level0 hr htm) hs hT
      have hN : level0Nodes t = 2 ^ (sch.levels[0]!).cap := by
        unfold level0Nodes
        rw [htm, hsch]
        simp only [Option.map_some, Option.getD_some, hg]
      rw [hfirst, hrows, hN]
      rcases hq2 with h | h
      · rw [h, pair_getElem!_zero, Nat.zero_add]
        exact hsh
      · rw [h, pair_getElem!_one, Nat.zero_add]
        exact hsh
    · obtain ⟨hfirst, hrows⟩ := topsAtL_fit hr hrs (hH r rs hr hrs) hs hm (hTops src hs (g + 1) (Nat.succ_pos g)) hT
      obtain ⟨s2, hs2, hm2⟩ := CapLower.topSize_cap hr hrs hm
      have hs2' : s2 = sch := by
        rw [hsch] at hs2
        exact (Option.some.inj hs2).symm
      rw [hs2'] at hm2
      rw [← hm2] at hsh
      rw [hfirst, hrows]
      rcases hq2 with h | h
      · rw [h, pair_getElem!_zero]
        omega
      · rw [h, pair_getElem!_one]
        omega

/-! ## At the inner statement's setup

What a session reader evaluates (`IrDepSession`, the difftest against the deployed builder): `depRest`'s pieces with the
inner statement's setup `v` read once rather than at every opening, each `depRest`'s own wherever the statement sets up
at `v` (`depRest_eq`). -/

/-- **Rep `k`'s verifier rows at the setup `v`**, table `t`'s shape and extra claims at the link coin `c₁`, rep `k`'s
`canonRounds` at the rep rounds' coins `cs`, and the comb `comb`. -/
def rowsV (v : FlockDeploy.Ins × _root_.Flock.SessionSpec × Array _root_.Flock.Stmt) (c₁ : Coin) (cs : List Coin)
    (t k : ℕ) (comb : Array F128) :
    Except String (Array F128 × RecAlgebra.Info RecAlgebra.Lin × Array (Array (Array F128))) :=
  RecAlgebra.verifierRows
    (FlockSoundness.Refine.shapeOf (tabsOf v.2.2)[t]! ((fast100 (tabsOf v.2.2)[t]!.m).getD default)
      (((tabsOf v.2.2)[t]!.extra (pointsOf v.2.1 c₁)).toOption.getD #[]).size)
    (Rows.canonRounds v.2.1 v.2.2 cs k) (((tabsOf v.2.2)[t]!.extra (pointsOf v.2.1 c₁)).toOption.getD #[]) comb

/-- The position verifier rows `r` sample for query `j` of level `l`. -/
def posOf (r : Except String (Array F128 × RecAlgebra.Info RecAlgebra.Lin × Array (Array (Array F128)))) (l j : ℕ) :
    ℕ :=
  match r with
  | .ok (_, info, _) => (info.lg.conss[l]!).pos[j]!
  | .error _ => 0

/-- Table `t`'s schedule at the setup `v`. -/
def schV (v : FlockDeploy.Ins × _root_.Flock.SessionSpec × Array _root_.Flock.Stmt) (t : ℕ) : _root_.Flock.Schedule :=
  (fast100 (tabsOf v.2.2)[t]!.m).getD default

/-- Rep `k`'s comb at the setup `v`. -/
def combV (v : FlockDeploy.Ins × _root_.Flock.SessionSpec × Array _root_.Flock.Stmt) (cs : List Coin) (k : ℕ) :
    Array F128 :=
  (match (rowsV v (cs.headD default) cs.tail (k / 2) k RecCoins.ZERO_COMB).toOption with
    | some (_, info, _) =>
      ((tabsOf v.2.2)[k / 2]!.fold.folded info.lc.fold.alpha info.lc.fold.beta info.lc.fold.lw info.lc.fold.rhoIn
        info.lc.fold.ts).toOption
    | none => none).getD RecCoins.ZERO_COMB

/-- `depOpen` at the schedule `sch` and rep `rep`'s verifier rows `r`, computed once for all its queries. -/
def depOpenV (sch : _root_.Flock.Schedule)
    (r : Except String (Array F128 × RecAlgebra.Info RecAlgebra.Lin × Array (Array (Array F128))))
    (ps : Array (_root_.Flock.Proof × F128)) (rep l j : ℕ) : _root_.VStar.In.Opening :=
  let lv := sch.levels[l]!
  let o := FlockSoundness.Refine.openAt (ps.getD rep default).1.ligerito (sch.levels.size - 1) l
  { rep, level := l, j, pos := posOf r l j,
    row := fitArr (2 ^ lv.lanes) 0 (o.1.getD j #[]),
    sibs := fitArr (lv.d - lv.cap) ByteArray.empty (_root_.Flock.sibsOf o.2.1 (lv.d - lv.cap) j).toArray }

/-- `depOpenings` at the setup `v`. -/
def depOpeningsV (v : FlockDeploy.Ins × _root_.Flock.SessionSpec × Array _root_.Flock.Stmt) (cs : List Coin)
    (ps : Array (_root_.Flock.Proof × F128)) : Array _root_.VStar.In.Opening :=
  let sch := schV v 0
  ((List.range 2).flatMap fun rep =>
    let r := rowsV v (cs.headD default) cs.tail 0 rep #[]
    (List.range sch.levels.size).flatMap fun l => (List.range (sch.levels[l]!).q).map (depOpenV sch r ps rep l)).toArray

/-- `depRest` at the setup `v`. -/
def depRestV (v : FlockDeploy.Ins × _root_.Flock.SessionSpec × Array _root_.Flock.Stmt) (τ : DepTape)
    (cs : List Coin) : _root_.VStar.In.Src where
  msgs := #[]
  openings := depOpeningsV v cs (τ.prove cs)
  tops := #[]
  accSalts := τ.accSalts
  kappa := (Array.range 2).map fun k =>
    (((rowsV v (cs.headD default) cs.tail (k / 2) k RecCoins.ZERO_COMB).toOption).map (·.2.2)).getD #[]
  eqA := (Array.range 2).map fun k =>
    (((rowsV v (cs.headD default) cs.tail (k / 2) k RecCoins.ZERO_COMB).toOption).map fun t =>
      t.2.1.lg.conss.map fun c => (eqTable c.alpha).extract 0 c.pos.size).getD #[]
  vs := (Array.range 2).map fun k =>
    (((rowsV v (cs.headD default) cs.tail (k / 2) k (combV v cs k)).toOption).map (·.1)).getD #[]
  fresh := τ.fresh

section
variable {xI : Msg} {v : FlockDeploy.Ins × _root_.Flock.SessionSpec × Array _root_.Flock.Stmt}

theorem posAt_eq (h : setupOf false xI = some v) (cs : List Coin) (q : OpenPiece) :
    Rows.posAt xI cs q =
      posOf (rowsV v (cs.headD default) cs.tail.dropLast q.table (2 * q.table + q.rep) #[]) q.level q.query := by
  unfold Rows.posAt
  rw [h]
  rfl

theorem schAt_eq (h : setupOf false xI = some v) (t : ℕ) : OpenLower.schAt xI t = schV v t := by
  unfold OpenLower.schAt
  rw [h]
  rfl

theorem verRowsAt_eq (h : setupOf false xI = some v) (cs : List Coin) (k : ℕ) (comb : Array F128) :
    verRowsAt xI cs k comb = (rowsV v (cs.headD default) cs.tail (k / 2) k comb).toOption := by
  unfold verRowsAt
  rw [h]
  rfl

theorem combAt_eq (h : setupOf false xI = some v) (cs : List Coin) (k : ℕ) : combAt xI cs k = combV v cs k := by
  unfold combAt combV
  rw [h, verRowsAt_eq h]
  cases (rowsV v (cs.headD default) cs.tail (k / 2) k RecCoins.ZERO_COMB).toOption with
  | none => rfl
  | some t =>
    obtain ⟨_, info, _⟩ := t
    rfl

theorem depOpen_eq (h : setupOf false xI = some v) (cs : List Coin) (ps : Array (_root_.Flock.Proof × F128))
    (rep l : ℕ) :
    depOpen xI cs ps rep l = depOpenV (schV v 0) (rowsV v (cs.headD default) cs.tail 0 rep #[]) ps rep l := by
  have e1 : (cs ++ [(default : Coin)]).headD default = cs.headD default := by cases cs <;> rfl
  have e2 : (cs ++ [(default : Coin)]).tail.dropLast = cs.tail := by cases cs <;> simp
  funext j
  simp only [depOpen, depOpenV, schAt_eq h, posAt_eq h, e1, e2, Nat.mul_zero, Nat.zero_add]

theorem depOpenings_eq (h : setupOf false xI = some v) (cs : List Coin) (ps : Array (_root_.Flock.Proof × F128)) :
    depOpenings xI cs ps = depOpeningsV v cs ps := by
  simp only [depOpenings, depOpeningsV, schAt_eq h, depOpen_eq h]

/-- **The session reader's rest is the deployed prover's** wherever the inner statement sets up at `v`. -/
theorem depRest_eq (h : setupOf false xI = some v) (τ : DepTape) (cs : List Coin) :
    depRest xI τ cs = depRestV v τ cs := by
  simp only [depRest, depRestV, verRowsAt_eq h, combAt_eq h, depOpenings_eq h]

/-- **`IrDep x`'s rows** where its inner statement sets up at `v`: round `h.length`'s of `depRows` at the coins of `h`. -/
theorem irDep_rows {x : Msg} (h : setupOf false (xIOf x) = some v) (w : Claims.Flock.Wit) (τ : DepTape)
    (hs : List (Msg × Msg)) :
    (IrDep x).rows w τ hs =
      ((depRows v.2.1 v.2.2 (τ.prove (innerCoins hs)) (innerCoins hs)).getD hs.length []).map
        _root_.VStar.In.wordsOf := by
  show (match setupOf false (xIOf x) with
    | some v => ((depRows v.2.1 v.2.2 (τ.prove (innerCoins hs)) (innerCoins hs)).getD hs.length []).map
        _root_.VStar.In.wordsOf
    | none => []) = _
  simp only [h]

/-- **`IrDep x`'s rest** where its inner statement sets up at `v`: `depRestV v` at the coins of `h`. -/
theorem irDep_rest {x : Msg} (h : setupOf false (xIOf x) = some v) (w : Claims.Flock.Wit) (τ : DepTape)
    (hs : List (Msg × Msg)) : (IrDep x).rest w τ hs = depRestV v τ (innerCoins hs) :=
  depRest_eq h τ (innerCoins hs)

end

/-- `FlockDeploy.decode` with the re-encoding compared by `List.beq`, a loop, rather than by `decide`. -/
def decodeB (x : Msg) : Option FlockDeploy.Raw :=
  match FlockDeploy.Raw.read.run x with
  | some (w, []) => if w.encode == x then some w else none
  | _ => none

/-- `FlockDeploy.insOf` at `decodeB`. -/
def insB (zk : Bool) (x : Msg) : Option FlockDeploy.Ins := do
  let w ← decodeB x
  if w.zk ≠ zk then none
  w.ins.toOption

/-- `FlockDeploy.setupOf` at `decodeB`. -/
def setupB (zk : Bool) (x : Msg) : Option (FlockDeploy.Ins × _root_.Flock.SessionSpec × Array _root_.Flock.Stmt) := do
  let s ← insB zk x
  let (spec, sts) ← s.setup.toOption
  pure (s, spec, sts)

theorem decodeB_eq (x : Msg) : decodeB x = FlockDeploy.decode x := by
  unfold decodeB FlockDeploy.decode
  cases FlockDeploy.Raw.read.run x with
  | none => rfl
  | some p =>
    obtain ⟨w, rest⟩ := p
    cases rest with
    | nil => simp [beq_iff_eq]
    | cons _ _ => rfl

/-- **The session reader's setup is the statement's.** -/
theorem setupB_eq (zk : Bool) (x : Msg) : setupB zk x = setupOf zk x := by
  unfold setupB insB
  rw [decodeB_eq]
  rfl

end RecDeploy.VStarBuild
