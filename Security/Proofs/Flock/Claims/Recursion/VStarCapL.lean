import Proofs.Flock.Claims.Recursion.VStarCap0

/-!
# A later level's rows at its cap's port `top` (`portRow_capl`)

Stream round `j` (inner round `j + 1`) of rep `j / R` at round `k = j % R` of the rep schedule commits its round's tops
first: top `e`'s `n` rows, after the round's earlier tops' (`(sizes.take e).sum`), at the cap of part
`capsFrom + topsBefore k + e` (`capSlots`), instance `j / R · n + t` for the `t`-th. At the fitted records each is a
fitted top (`fitSlot` at a 64-word `top` port: a node's 32 words, then 32 zero words).

The stager reads level `topsBefore k + e + 1`'s tops off the same rows (`topsOf`, `topsOf_level`): rep `rep`'s from its
round `rep · R + k + 1`. Where the two reps' tops differ (`sameTop` refuses them, `hd`), the level's tops are rep 0's
`n` nodes, then rep 1's (`topsAt_two`), so the cap's instance `rep · n + t` reads rep `rep`'s `t`-th row back whole
(`topRows_fit`, `capIn_ok`, `portRow_ownRows`), unchanged at the fitted port inputs (`fitPin`, `fitPortRow_top`).
-/

namespace RecDeploy.VStarBuild

open _root_.Flock
open Claims (Msg)

/-! ## Lists -/

theorem flatMap_range_getElem? {α : Type} (F : ℕ → List α) {R k e : ℕ} (hk : k < R) (he : e < (F k).length) :
    ((List.range R).flatMap F)[((List.range k).flatMap F).length + e]? = some (F k)[e] := by
  obtain ⟨m, rfl⟩ : ∃ m, R = k + 1 + m := ⟨R - k - 1, by omega⟩
  rw [List.range_add, List.range_succ]
  simp only [List.flatMap_append, List.flatMap_cons, List.flatMap_nil, List.append_nil, List.append_assoc]
  rw [List.getElem?_append_right (Nat.le_add_right _ _), Nat.add_sub_cancel_left, List.getElem?_append_left he]
  exact List.getElem?_eq_getElem he

theorem take_flatMap_length {α β : Type} [Inhabited α] (rs : Array α) (g : α → List β) {k : ℕ} (hk : k ≤ rs.size) :
    ((rs.toList.take k).flatMap g).length = ((List.range k).map fun i => (g (rs.getD i default)).length).sum := by
  rw [List.length_flatMap]
  congr 1
  apply List.ext_getElem (by simp; omega)
  intro i h1 h2
  simp at h1 h2
  simp [Array.getD, show i < rs.size by omega]

theorem sum_range_getD (a : Array ℕ) {e : ℕ} (he : e ≤ a.size) :
    ((List.range e).map fun i => a.getD i 0).sum = (a.toList.take e).sum := by
  congr 1
  apply List.ext_getElem (by simp; omega)
  intro i h1 h2
  simp at h1 h2
  simp [Array.getD, show i < a.size by omega]

theorem isEmpty_false {α : Type} {a : Array α} {e : ℕ} (he : e < a.size) : a.isEmpty = false := by
  cases h : a.isEmpty
  · rfl
  · rw [Array.isEmpty_iff_size_eq_zero] at h
    omega

theorem toList_getD (a : Array ℕ) (e : ℕ) : a.toList.getD e 0 = a.getD e 0 := by
  simp [Array.getD_eq_getD_getElem?, List.getD_eq_getElem?_getD]

/-! ## A level whose two reps differ -/

theorem sameTop_comm (a b : Array (ByteArray × ByteArray)) : VStar.In.sameTop a b = VStar.In.sameTop b a := by
  unfold VStar.In.sameTop
  by_cases h : a.size = b.size
  · have hab : (a.size == b.size) = true := by simp [h]
    have hba : (b.size == a.size) = true := by simp [h]
    rw [hab, hba, Bool.true_and, Bool.true_and, Bool.eq_iff_iff, Array.all_eq_true, Array.all_eq_true]
    constructor
    · intro H i hi
      have hi' : i < (a.zip b).size := by simp at hi ⊢; omega
      have := H i hi'
      simp only [Array.getElem_zip, Bool.and_eq_true, beq_iff_eq] at this ⊢
      exact ⟨this.1.symm, this.2.symm⟩
    · intro H i hi
      have hi' : i < (b.zip a).size := by simp at hi ⊢; omega
      have := H i hi'
      simp only [Array.getElem_zip, Bool.and_eq_true, beq_iff_eq] at this ⊢
      exact ⟨this.1.symm, this.2.symm⟩
  · have hab : (a.size == b.size) = false := by simp [h]
    have hba : (b.size == a.size) = false := by simp [Ne.symm h]
    rw [hab, hba, Bool.false_and, Bool.false_and]

/-- **A level the two reps don't share**: `topsAt` makes one run a rep, rep 0's then rep 1's. -/
theorem topsAt_two {src : VStar.In.Src} {l : ℕ} {a b : Array (ByteArray × ByteArray)} (h2 : src.tops.size = 2)
    (h0 : (src.tops[0]!)[l]? = some a) (h1 : (src.tops[1]!)[l]? = some b) (ha : a.all (·.1.size == 64) = true)
    (hb : b.all (·.1.size == 64) = true) (hd : VStar.In.sameTop b a = false) :
    VStar.In.topsAt src l = .ok
      { rows := #[a, b].flatMap (fun t => t.flatMap fun (n, _) => (Firewall.topRows n).toArray),
        salts := #[a, b].flatMap (fun t => t.map (·.2)), first := #[0, a.size] } := by
  have hr : List.range' 0 2 1 = [0, 1] := rfl
  have h0' : (src.tops[0]'(by omega))[l]? = some a := by
    rw [← getElem!_pos src.tops 0 (by omega)]
    exact h0
  have h1' : (src.tops[1]'(by omega))[l]? = some b := by
    rw [← getElem!_pos src.tops 1 (by omega)]
    exact h1
  have ha' : ∀ (i : ℕ) (hi : i < a.size), a[i].1.size = 64 := by simpa using ha
  have hb' : ∀ (i : ℕ) (hi : i < b.size), b[i].1.size = 64 := by simpa using hb
  have hf : Array.findIdx? (VStar.In.sameTop b) #[a] = none := by
    simp [hd]
  simp [VStar.In.topsAt, h2, h0', h1', ha', hb', hf, hr]
  rfl

/-! ## The stager's tops at a later level -/

theorem topsBefore_eq (rs : Array RecVStar.SchedRound) {k : ℕ} (hk : k ≤ rs.size) :
    topsBefore rs k = ((List.range k).map fun i => ((rs.getD i default).2.1.toList.length)).sum :=
  take_flatMap_length rs (·.2.1.toList) hk

/-- **Level `topsBefore k + e + 1`'s tops of rep `rep`**: the rows of rep `rep`'s round `k` after its earlier tops,
as nodes. -/
theorem topsOf_level (rs : Array RecVStar.SchedRound) (rr : ℕ → List _root_.VStar.In.Row) {rep k e : ℕ}
    (hrep : rep < 2) (hk : k < rs.size) (he : e < ((rs.getD k default).2.1).size) :
    ((topsOf rs rr)[rep]!)[topsBefore rs k + e + 1]? = some (((((rr (rep * rs.size + k + 1)).drop
      ((rs.getD k default).2.1.toList.take e).sum).take ((rs.getD k default).2.1.toList.getD e 0)).map
        fun (rw : _root_.VStar.In.Row) => (wordsBytes (rw.words.extract 0 32), rw.salt)).toArray) := by
  have hb : topsBefore rs k = ((List.range k).flatMap fun k' => (List.range (rs.getD k' default).2.1.toList.length).map
      fun e => ((rr (rep * rs.size + k' + 1)).drop (((rs.getD k' default).2.1.toList).take e).sum).take
        (((rs.getD k' default).2.1.toList).getD e 0)).length := by
    rw [topsBefore_eq rs (Nat.le_of_lt hk), List.length_flatMap]
    simp
  have hl := flatMap_range_getElem? (fun k' => (List.range (rs.getD k' default).2.1.toList.length).map
      fun e => ((rr (rep * rs.size + k' + 1)).drop (((rs.getD k' default).2.1.toList).take e).sum).take
        (((rs.getD k' default).2.1.toList).getD e 0)) hk (e := e) (by simpa using he)
  rw [← hb] at hl
  have hA : ∀ (f : ℕ → Array (Array (ByteArray × ByteArray))), (((List.range 2).map f).toArray)[rep]! = f rep := by
    intro f
    rw [getElem!_pos _ _ (by simpa using hrep)]
    simp
  unfold topsOf
  simp only []
  rw [hA]
  simp only [List.getElem?_toArray, List.getElem?_map, List.getElem?_cons_succ, hl, Option.map_some,
    List.getElem_map, List.getElem_range]

/-! ## The fitted records at a round that commits tops -/

theorem capSlots_length (r : XR) (rs : Array RecVStar.SchedRound) (j e : ℕ) :
    (capSlots r rs j e).length = ((rs.getD (j % rs.size) default).2.1).getD e 0 := by
  simp [capSlots]

theorem capSlots_before (r : XR) (rs : Array RecVStar.SchedRound) (j : ℕ) {e : ℕ}
    (he : e ≤ ((rs.getD (j % rs.size) default).2.1).size) :
    ((List.range e).flatMap (capSlots r rs j)).length = ((rs.getD (j % rs.size) default).2.1.toList.take e).sum := by
  rw [List.length_flatMap]
  simp only [capSlots_length]
  exact sum_range_getD _ he

/-- **Top `e`'s `t`-th slot of stream round `j`**: its cap's part, port 0 (`top`), instance `j / R · n + t`. -/
theorem slotsAt_cap {x : Msg} {r : XR} {rs : Array RecVStar.SchedRound} {j e t : ℕ} {c : Circuit}
    (hr : readX x = some r) (hrs : repSched r.shape = some rs) (he : e < ((rs.getD (j % rs.size) default).2.1).size)
    (ht : t < ((rs.getD (j % rs.size) default).2.1).getD e 0)
    (hc : r.w.parts[capsFrom r rs + topsBefore rs (j % rs.size) + e]?.bind partCircuit = some c)
    (hq : capPort c = some 0) :
    (slotsAt x (j + 1))[((rs.getD (j % rs.size) default).2.1.toList.take e).sum + t]? =
      some (capsFrom r rs + topsBefore rs (j % rs.size) + e, 0,
        j / rs.size * ((rs.getD (j % rs.size) default).2.1).getD e 0 + t) := by
  have hne := isEmpty_false he
  have hne' : ¬ (rs.getD (j % rs.size) default).2.1 = #[] := by
    intro h
    rw [h] at he
    simp at he
  have hs : slotsAt x (j + 1) =
      (List.range ((rs.getD (j % rs.size) default).2.1).size).flatMap (capSlots r rs j) ++ rowSlot r j := by
    simp only [Array.getD_eq_getD_getElem?] at hne hne' ⊢
    simp [slotsAt, hr, hrs, hne, hne']
  have h1 := flatMap_range_getElem? (capSlots r rs j) he (e := t) (by rw [capSlots_length]; exact ht)
  rw [capSlots_before r rs j (Nat.le_of_lt he)] at h1
  rw [hs, List.getElem?_append_left (List.getElem?_eq_some_iff.1 h1).1, h1]
  simp [capSlots, hc, hq]

theorem roundRows_getElem? (I : InnerRec) (x : Msg) (w : Claims.Flock.Wit) (ρ : (I.fit x).Tape)
    (hist : List (Msg × Msg)) {i m : ℕ} (hH : i ≤ hist.length) :
    (roundRows (I.fit x) w ρ hist i)[m]? = (slotsAt x i)[m]?.map fun sl =>
      (⟨fitSlot x sl ((I.rows w ρ.1 (hist.take i)).getD m #[]), saltBytes (ρ.2 i m)⟩ : _root_.VStar.In.Row) := by
  simp [roundRows, InnerRec.fit, List.getElem?_mapIdx, Nat.min_eq_left hH, Function.comp_def]

theorem fitSlot_cap {x : Msg} {r : XR} {p : ℕ} {c : Circuit} {p0 p1 : _root_.Flock.Port} (hr : readX x = some r)
    (hc : r.w.parts[p]?.bind partCircuit = some c) (hp : c.ports = #[p0, p1]) (h0 : p0.name = "top")
    (hw : p0.words = 64) (m : ℕ) (ws : Array ℕ) : fitSlot x (p, 0, m) ws = fitRow 32 ws ++ Array.replicate 32 0 := by
  have hcx : circAt x p = c := by simp [circAt, hr, hc]
  simp [fitSlot, isTop, widthAt, hcx, hp, h0, hw]

/-- **A fitted cap row**: the records' row at index `o + t` of inner round `j + 1`, fitted to a 64-word `top`, under
its salt. -/
def capRow (I : InnerRec) (w : Claims.Flock.Wit) (ρ : I.Tape) (hist : List (Msg × Msg)) (o j t : ℕ) :
    _root_.VStar.In.Row :=
  ⟨fitRow 32 ((I.rows w ρ.1 (hist.take (j + 1))).getD (o + t) #[]) ++ Array.replicate 32 0, saltBytes (ρ.2 (j + 1) (o + t))⟩

/-- **Top `e`'s rows of stream round `j`** at the fitted records: the fitted cap rows. -/
theorem roundRows_cap {I : InnerRec} {x : Msg} {w : Claims.Flock.Wit} {ρ : (I.fit x).Tape}
    {hist : List (Msg × Msg)} {r : XR} {rs : Array RecVStar.SchedRound} {j e t : ℕ} {c : Circuit}
    {p0 p1 : _root_.Flock.Port} (hr : readX x = some r) (hrs : repSched r.shape = some rs)
    (hH : j + 1 ≤ hist.length) (he : e < ((rs.getD (j % rs.size) default).2.1).size)
    (ht : t < ((rs.getD (j % rs.size) default).2.1).getD e 0)
    (hc : r.w.parts[capsFrom r rs + topsBefore rs (j % rs.size) + e]?.bind partCircuit = some c)
    (hp : c.ports = #[p0, p1]) (h0 : p0.name = "top") (hw : p0.words = 64) :
    (roundRows (I.fit x) w ρ hist (j + 1))[((rs.getD (j % rs.size) default).2.1.toList.take e).sum + t]? =
      some (capRow I w ρ hist ((rs.getD (j % rs.size) default).2.1.toList.take e).sum j t) := by
  rw [roundRows_getElem? I x w ρ hist hH, slotsAt_cap hr hrs he ht hc (capPort_cap hp h0)]
  simp only [Option.map_some]
  rw [fitSlot_cap hr hc hp h0 hw]
  rfl

/-- **Top `e`'s `n` rows of stream round `j`**, as `topsOf` cuts them: the fitted cap rows. -/
theorem roundRows_cap_take {I : InnerRec} {x : Msg} {w : Claims.Flock.Wit} {ρ : (I.fit x).Tape}
    {hist : List (Msg × Msg)} {r : XR} {rs : Array RecVStar.SchedRound} {j e : ℕ} {c : Circuit}
    {p0 p1 : _root_.Flock.Port} (hr : readX x = some r) (hrs : repSched r.shape = some rs)
    (hH : j + 1 ≤ hist.length) (he : e < ((rs.getD (j % rs.size) default).2.1).size)
    (hc : r.w.parts[capsFrom r rs + topsBefore rs (j % rs.size) + e]?.bind partCircuit = some c)
    (hp : c.ports = #[p0, p1]) (h0 : p0.name = "top") (hw : p0.words = 64) :
    ((roundRows (I.fit x) w ρ hist (j + 1)).drop ((rs.getD (j % rs.size) default).2.1.toList.take e).sum).take
        ((rs.getD (j % rs.size) default).2.1.toList.getD e 0) =
      (List.range (((rs.getD (j % rs.size) default).2.1).getD e 0)).map
        (capRow I w ρ hist ((rs.getD (j % rs.size) default).2.1.toList.take e).sum j) := by
  apply List.ext_getElem?
  intro t
  rw [List.getElem?_take, List.getElem?_drop, toList_getD]
  by_cases ht : t < ((rs.getD (j % rs.size) default).2.1).getD e 0
  · simp only [ht, ↓reduceIte]
    rw [roundRows_cap hr hrs hH he ht hc hp h0 hw, List.getElem?_map, List.getElem?_range ht, Option.map_some]
  · simp only [ht, ↓reduceIte]
    rw [List.getElem?_map, List.getElem?_eq_none (by simpa using ht), Option.map_none]

/-! ## The cap's port `top` -/

set_option maxHeartbeats 1000000 in
/-- **Level `topsBefore k + e + 1`'s cap reads, at instance `j / R · n + t`, stream round `j`'s top-`e` row `t`**, at
the fitted port inputs: at stream round `j < 2R` (round `k = j % R` of rep `j / R`) whose `e`-th top has `n` nodes,
where the cap at part `capsFrom + topsBefore k + e` is V\*'s parsed `.cap l N` with `N = 2n`, the history holds both
reps' rounds, and the reps' tops at level `l` differ (`hd`: `sameTop`, which would merge them, refuses them). -/
theorem portRow_capl {I : InnerRec} {x : Msg} {w : Claims.Flock.Wit} {ρ : (I.fit x).Tape}
    {hist : List (Msg × Msg)} {r : XR} {rs : Array RecVStar.SchedRound} {j e N : ℕ} {c : Circuit}
    (hr : readX x = some r) (hrs : repSched r.shape = some rs) (hj : j < 2 * rs.size)
    (hH : 2 * rs.size ≤ hist.length) (he : e < ((rs.getD (j % rs.size) default).2.1).size)
    (hc : r.w.parts[capsFrom r rs + topsBefore rs (j % rs.size) + e]?.bind partCircuit = some c)
    (hcp : RecVStar.checkParsed (.cap (topsBefore rs (j % rs.size) + e + 1) N) c = .ok ())
    (hN : N = 2 * ((rs.getD (j % rs.size) default).2.1).getD e 0)
    (hd : ∀ a b, ((topsOf rs (roundRows (I.fit x) w ρ hist))[0]!)[topsBefore rs (j % rs.size) + e + 1]? = some a →
      ((topsOf rs (roundRows (I.fit x) w ρ hist))[1]!)[topsBefore rs (j % rs.size) + e + 1]? = some b →
      VStar.In.sameTop a b = false) :
    ∃ src pin, srcOf (I.fit x) x w ρ hist = some src ∧
      _root_.VStar.In.partIn src (progOf c) (.cap (topsBefore rs (j % rs.size) + e + 1) N) = .ok pin ∧
      ∀ t < ((rs.getD (j % rs.size) default).2.1).getD e 0,
        portRow (fitPin c pin) "top" (j / rs.size * ((rs.getD (j % rs.size) default).2.1).getD e 0 + t) =
          (roundRows (I.fit x) w ρ hist (j + 1))[((rs.getD (j % rs.size) default).2.1.toList.take e).sum + t]? := by
  obtain ⟨p0, p1, hp, h0, hw, -⟩ := cap_ports hcp
  have hR : 0 < rs.size := by omega
  have hkR : j % rs.size < rs.size := Nat.mod_lt _ hR
  have hjR : j / rs.size < 2 := (Nat.div_lt_iff_lt_mul hR).2 (by omega)
  -- each rep's round `k`, as a stream round
  have hmod : ∀ rep, (rep * rs.size + j % rs.size) % rs.size = j % rs.size := fun rep => by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_mod]
  have hrows : ∀ rep < 2,
      ((roundRows (I.fit x) w ρ hist (rep * rs.size + j % rs.size + 1)).drop
          ((rs.getD (j % rs.size) default).2.1.toList.take e).sum).take
        ((rs.getD (j % rs.size) default).2.1.toList.getD e 0) =
      (List.range (((rs.getD (j % rs.size) default).2.1).getD e 0)).map
        (capRow I w ρ hist ((rs.getD (j % rs.size) default).2.1.toList.take e).sum (rep * rs.size + j % rs.size)) := by
    intro rep hrep
    have h := roundRows_cap_take (I := I) (w := w) (ρ := ρ) (hist := hist) (j := rep * rs.size + j % rs.size)
      (e := e) hr hrs (by
        have : rep * rs.size + j % rs.size < 2 * rs.size := by
          rcases (show rep = 0 ∨ rep = 1 by omega) with rfl | rfl <;> omega
        omega) (by rw [hmod]; exact he) (by rw [hmod]; exact hc) hp h0 hw
    simpa only [hmod] using h
  -- the nodes of each rep at level `l`
  have htop : ∀ rep < 2,
      ((topsOf rs (roundRows (I.fit x) w ρ hist))[rep]!)[topsBefore rs (j % rs.size) + e + 1]? =
        some ((List.range (((rs.getD (j % rs.size) default).2.1).getD e 0)).map fun t =>
          (wordsBytes (fitRow 32 ((I.rows w ρ.1 (hist.take (rep * rs.size + j % rs.size + 1))).getD
            (((rs.getD (j % rs.size) default).2.1.toList.take e).sum + t) #[])),
           saltBytes (ρ.2 (rep * rs.size + j % rs.size + 1) (((rs.getD (j % rs.size) default).2.1.toList.take e).sum + t)))).toArray := by
    intro rep hrep
    rw [topsOf_level rs _ hrep hkR he, hrows rep hrep, List.map_map]
    refine congrArg (fun l => some l.toArray) (List.map_congr_left fun t _ => ?_)
    simp only [Function.comp_apply, capRow, extract_top]
  have hsrc : srcOf (I.fit x) x w ρ hist = some { (I.fit x).rest w ρ.1 hist with
      msgs := msgsOf rs (roundRows (I.fit x) w ρ hist), tops := topsOf rs (roundRows (I.fit x) w ρ hist) } := by
    simp [srcOf, hr, hrs]
  have ht2 : (topsOf rs (roundRows (I.fit x) w ρ hist)).size = 2 := by simp [topsOf]
  have hall : ∀ rep, ((List.range (((rs.getD (j % rs.size) default).2.1).getD e 0)).map fun t =>
      (wordsBytes (fitRow 32 ((I.rows w ρ.1 (hist.take (rep * rs.size + j % rs.size + 1))).getD
        (((rs.getD (j % rs.size) default).2.1.toList.take e).sum + t) #[])),
       saltBytes (ρ.2 (rep * rs.size + j % rs.size + 1)
        (((rs.getD (j % rs.size) default).2.1.toList.take e).sum + t)))).toArray.all (·.1.size == 64) = true := by
    intro rep
    simp [wordsBytes_size, fitRow_size]
  have hT := topsAt_two (src := { (I.fit x).rest w ρ.1 hist with
      msgs := msgsOf rs (roundRows (I.fit x) w ρ hist), tops := topsOf rs (roundRows (I.fit x) w ρ hist) })
    ht2 (htop 0 (by omega)) (htop 1 (by omega)) (hall 0) (hall 1) (by rw [sameTop_comm]; exact hd _ _ (htop 0 (by omega)) (htop 1 (by omega)))
  obtain ⟨⟨Tr, Ts, Tf⟩, hT', hTr, hTs⟩ : ∃ T : _root_.VStar.In.Tops, _root_.VStar.In.topsAt
      { (I.fit x).rest w ρ.1 hist with
        msgs := msgsOf rs (roundRows (I.fit x) w ρ hist), tops := topsOf rs (roundRows (I.fit x) w ρ hist) }
        (topsBefore rs (j % rs.size) + e + 1) = .ok T ∧
      T.rows = (((List.range (((rs.getD (j % rs.size) default).2.1).getD e 0)).map fun t =>
          (capRow I w ρ hist ((rs.getD (j % rs.size) default).2.1.toList.take e).sum (0 * rs.size + j % rs.size) t).words) ++
        ((List.range (((rs.getD (j % rs.size) default).2.1).getD e 0)).map fun t =>
          (capRow I w ρ hist ((rs.getD (j % rs.size) default).2.1.toList.take e).sum (1 * rs.size + j % rs.size) t).words)).toArray ∧
      T.salts = (((List.range (((rs.getD (j % rs.size) default).2.1).getD e 0)).map fun t =>
          (capRow I w ρ hist ((rs.getD (j % rs.size) default).2.1.toList.take e).sum (0 * rs.size + j % rs.size) t).salt) ++
        ((List.range (((rs.getD (j % rs.size) default).2.1).getD e 0)).map fun t =>
          (capRow I w ρ hist ((rs.getD (j % rs.size) default).2.1.toList.take e).sum (1 * rs.size + j % rs.size) t).salt)).toArray :=
    ⟨_, hT, by apply Array.toList_inj.1; simp [List.flatMap_map, topRows_fit, flatMap_single, capRow],
      by apply Array.toList_inj.1; simp [capRow, Function.comp_def]⟩
  subst hTr
  subst hTs
  have hpin := capIn_ok (N := N) hT' (by
    rw [hN, List.size_toArray, List.length_append, List.length_map, List.length_map, List.length_range]
    omega)
  refine ⟨_, _, hsrc, hpin, fun t ht => ?_⟩
  have hpw : portWords c "top" = 64 := by
    have hq : c.ports.findIdx? (·.name == "top") = some 0 := capPort_cap hp h0
    unfold portWords
    rw [hq]
    simp [hp, hw]
  have hH' : j + 1 ≤ hist.length := by omega
  rw [roundRows_cap hr hrs hH' he ht hc hp h0 hw]
  have hj' : j / rs.size * rs.size + j % rs.size = j := Nat.div_add_mod' j rs.size
  have h01 : j / rs.size = 0 ∨ j / rs.size = 1 := by
    obtain ⟨q, hq⟩ : ∃ q, q = j / rs.size := ⟨_, rfl⟩
    rw [← hq] at hjR ⊢
    omega
  rcases h01 with h | h
  · rw [h] at hj'
    have hcr : capRow I w ρ hist ((rs.getD (j % rs.size) default).2.1.toList.take e).sum j t =
        capRow I w ρ hist ((rs.getD (j % rs.size) default).2.1.toList.take e).sum (0 * rs.size + j % rs.size) t := by
      rw [hj']
    have hi : j / rs.size * ((rs.getD (j % rs.size) default).2.1).getD e 0 + t = t := by
      rw [h, Nat.zero_mul, Nat.zero_add]
    rw [hi, hcr]
    rw [portRow_fitPin, pair_append, portRow_ownRows (by simp at ht ⊢; omega) (by simp at ht ⊢; omega), Option.map_some, hpw]
    simp only [List.getElem_toArray]
    rw [List.getElem_append_left (by simpa using ht), List.getElem_append_left (by simpa using ht)]
    simp only [List.getElem_map, List.getElem_range, capRow]
    exact congrArg some (fitPortRow_top _ _)
  · rw [h] at hj'
    have hcr : capRow I w ρ hist ((rs.getD (j % rs.size) default).2.1.toList.take e).sum j t =
        capRow I w ρ hist ((rs.getD (j % rs.size) default).2.1.toList.take e).sum (1 * rs.size + j % rs.size) t := by
      rw [hj']
    have hi : j / rs.size * ((rs.getD (j % rs.size) default).2.1).getD e 0 + t =
        ((rs.getD (j % rs.size) default).2.1).getD e 0 + t := by
      rw [h, Nat.one_mul]
    rw [hi, hcr]
    rw [portRow_fitPin, pair_append, portRow_ownRows (by simp at ht ⊢; omega) (by simp at ht ⊢; omega), Option.map_some, hpw]
    simp only [List.getElem_toArray]
    rw [List.getElem_append_right (by simp), List.getElem_append_right (by simp)]
    simp only [List.getElem_map, List.getElem_range, List.length_map, List.length_range, Nat.add_sub_cancel_left,
      capRow]
    exact congrArg some (fitPortRow_top _ _)

end RecDeploy.VStarBuild
