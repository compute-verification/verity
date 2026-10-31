import Proofs.Flock.Claims.Recursion.ResTieVV
import Proofs.Flock.Claims.Recursion.PastIn

/-!
# A staged `rec-residuals` part's site, from the verifier's checks

`resTie_resVRead` asks, at a staged algebra part, where its circuit's `v` and `out` ports sit, that `v` is an input row
and `out` an output row, their words, two or more instances, `0 < c.g` and the block bound. Each is a check the
verifier ran on the part's table:

* `checkParsed` holds the circuit's ports to V\*'s rows (`vstarRows`): the row ports, `v`, then `out` (`rs_rows_inner`);
  `checkMsgNames` names each port once (`MoreSlots.staged_names`), so `v` and `out` are found there (`rs_vOut`);
* `out` is an output row (`MoreSlots.pastIn_core`). `v` is an input row (`rs_vIn`): were it an output row, its words
  would be whole chunks (`OutRows`), so the unit's input leaves, its input words in order (`checkParsed`), would reach
  past the input rows' words, which `HmOut.checkRows` refuses (`rs_leavesIn_lt`);
* the part has two instances (`staged_scope`), held in one table (`oneFull_staged`) of `pub.n` rows (`setupOne_n`),
  `0 < c.g` and the blocks are `table_rows`', and the block bound `div_lt_blocks_two`'s (`resSite_staged`);
* check 5's rows are two, each `portWordsOf nv` words (`resRows_dense`, at the plain part `staged_plain` gives).

`resTie_dense` is `resTie_resVRead` with all of these discharged; `hidx` stays a hypothesis (`labels_run`).
-/

namespace RecDeploy.ResLower

open Flock
open Claims (Msg)
open FlockDeploy (Coin)
open Claims.Recursion.VStarB
open FlockSoundness.Discharge.Recursion (residualRows residualRowsOf residualGroups)

/-! ## Arithmetic -/

theorem rs_foldl_sum : ∀ (l : List ℕ) (b : ℕ), l.foldl (· + ·) b = b + l.sum
  | [], b => by simp
  | a :: l, b => by rw [List.foldl_cons, rs_foldl_sum l, List.sum_cons]; omega

theorem rs_sum_take_le : ∀ (l : List ℕ) {n m : ℕ}, n ≤ m → (l.take n).sum ≤ (l.take m).sum
  | [], _, _, _ => by simp
  | _ :: _, 0, _, _ => by simp
  | _ :: _, _ + 1, 0, h => by omega
  | a :: l, n + 1, m + 1, h => by
    simp only [List.take_succ_cons, List.sum_cons]
    have := rs_sum_take_le l (n := n) (m := m) (by omega)
    omega

theorem rs_padTo_size (ws : Array ℕ) (n : ℕ) : (RecCoins.padTo ws n).size = ws.size + (n - ws.size) := by
  unfold RecCoins.padTo
  rw [Array.size_append, Array.size_replicate]

theorem rs_words_size (xs : Array _root_.Flock.F128) : (RecCoins.words xs).size = 8 * xs.size := by
  show (RecCoins.words xs).toList.length = 8 * xs.toList.length
  unfold RecCoins.words
  dsimp only
  rw [Array.toList_flatMap]
  generalize xs.toList = l
  induction l with
  | nil => simp
  | cons x l ih =>
    rw [List.flatMap_cons, List.length_append, ih]
    simp only [Array.length_toList, Array.size_append, Array.size_map, Array.size_range, List.length_cons]
    omega

theorem rs_zeroRow_size {n : ℕ} (hn : n < 2) : ((RecCoins.zeroRows 2 RecCoins.OUT_WORDS)[n]!).size = 128 := by
  obtain rfl | rfl : n = 0 ∨ n = 1 := by omega
  all_goals first | rfl | decide | simp [RecCoins.zeroRows, RecCoins.OUT_WORDS]

/-! ## Check 5's rows -/

/-- One of check 5's rows is `portWordsOf vs.size` words: `vs`' elements, 8 words each, padded. -/
theorem rs_row_size {inn : RecCoins.Inner} {sh : _root_.Flock.Shape} {extras : Array _root_.Flock.Claim}
    {vs : Array ℕ} {r : ℕ} {row : Array ℕ}
    (h : (do
      let rep ← inn.rep sh extras r
      let base := rep.v.extract 0 (rep.v.size - 64)
      let comb ← if vs.any (· ≥ base.size) then inn.comb r rep else pure RecCoins.ZERO_COMB
      let full := base ++ comb
      pure (RecCoins.padTo (RecCoins.words (vs.map (full[·]!))) (RecResiduals.portWordsOf vs.size)) :
        _root_.Flock.V (Array ℕ)) = .ok row) :
    row.size = RecResiduals.portWordsOf vs.size := by
  obtain ⟨rep, -, h⟩ := FlockVBridge.PartS.bind_ok h
  obtain ⟨cb, hrow⟩ : ∃ cb, row = RecCoins.padTo
      (RecCoins.words (vs.map ((rep.v.extract 0 (rep.v.size - 64) ++ cb)[·]!))) (RecResiduals.portWordsOf vs.size) := by
    dsimp only at h
    split at h
    · obtain ⟨cb, -, h⟩ := FlockVBridge.PartS.bind_ok h
      simp only [pure, Except.pure, Except.ok.injEq] at h
      exact ⟨cb, h.symm⟩
    · rw [pure_bind] at h
      simp only [pure, Except.pure, Except.ok.injEq] at h
      exact ⟨RecCoins.ZERO_COMB, h.symm⟩
  subst hrow
  rw [rs_padTo_size, rs_words_size, Array.size_map]
  unfold RecResiduals.portWordsOf
  omega

/-- **Check 5's rows are two, each `portWordsOf vs.size` words.** -/
theorem rs_rows_size {inn : RecCoins.Inner} {sh : _root_.Flock.Shape} {extras : Array _root_.Flock.Claim}
    {vs : Array ℕ} {rows : Array (Array ℕ)} (h : residualRowsOf inn sh extras vs = .ok rows) :
    rows.size = 2 ∧ ∀ n < 2, (rows[n]!).size = RecResiduals.portWordsOf vs.size := by
  unfold residualRowsOf at h
  obtain ⟨a, b, ha, hb, rfl⟩ := FlockC3.Opening.mapM_two h
  refine ⟨rfl, fun n hn => ?_⟩
  match n, hn with
  | 0, _ => exact rs_row_size ha
  | 1, _ => exact rs_row_size hb

/-- **Check 5's rows at a dense statement's part are `portWordsOf nv` words**: check 5 loaded the part's inner session
and registered its rows at the part's shape (`residualsShape_of_check`), plain at a dense statement (`staged_plain`), so
group `jj`'s `vs` has the part's `nv` elements (`part_nv`). -/
theorem resRows_dense {b : Brings} {s : ℕ} {x xO : Msg} {cs : ℕ → Coin} {st0 : Claims.Recursion.Stmt}
    (hst : RecDeploy.parse x = some st0) (hd : Claims.Recursion.Form.Dense st0.form)
    (hsh : showsPart recRead b s x xO cs) {i : RecIns} {spec : SessionSpec} {sts : Array Stmt}
    (hsu : setupO xO = some (i, spec, sts)) (h0 : 0 < sts.size)
    {m k cl jj : ℕ} (hres : RecResiduals.params sts[0].c.unit.name = some (m, k, cl, jj))
    {sh : _root_.Flock.Shape} {holo : Bool} {S : _root_.Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ}
    (hshp : RecResiduals.shape? m k cl = some sh) (hholo : RecResiduals.holoOf sts[0].c = .ok holo)
    (hpart : RecResiduals.part { sh with holo } jj = .ok (S, ns, nv))
    {f : RecCoins.Source} (hf : RecCoins.Source.atC3 i.src i.inner i.commits = some f)
    {inn : RecCoins.Inner} {rows : Array (Array ℕ)} (hload : f.load sts[0].c.unit.name = .ok inn)
    (hrows : residualRows inn jj = some rows) :
    rows.size = 2 ∧ ∀ n < 2, (rows[n]!).size = RecResiduals.portWordsOf nv := by
  obtain ⟨-, hset0⟩ := setupO_some hsu
  have hrs : (RecResiduals.params sts[0].c.unit.name).isSome := by simp [hres]
  obtain ⟨own, f', hown, hf', hso, hop, hsp, hro⟩ := CopySite.copySite_staged hsu h0 (.inr hrs)
  have hff : f' = f := Option.some.inj (hf'.symm.trans hf)
  subst f'
  have hcoin := Flock.Readable.array_forM_ok (RecIns.setup_coins hset0) 0 h0
  rw [hown, hf] at hcoin
  obtain ⟨inn', hinn', hcr⟩ :=
    FlockSoundness.Discharge.Recursion.residualsCheck_of_check hcoin hso hop hsp (hro hrs) hres
  have hii : inn' = inn := Except.ok.inj (hinn'.symm.trans hload)
  subst inn'
  obtain ⟨sh', extras, halg, hm', hk', hcl'⟩ := residualsShape_of_check hcr
  obtain ⟨gs, R, P, vs, hgs, hj, hrr⟩ := residualRows_some halg hrows
  obtain ⟨-, hshI⟩ := OpenLower.algebra_ok halg
  obtain ⟨sch, -, hsm, hsk, hscl, -, hsh0⟩ := OpenLower.shape?_ok hshI
  have hshe : sh = sh' := by
    rw [hm', hk', hcl', hsm, hsk, hscl] at hshp
    exact Option.some.inj (hshp.symm.trans hshI)
  subst sh'
  have hh := hholo.symm.trans (staged_plain hst hd hsh hsu 0 h0)
  cases hh
  have hwith : ({ sh with holo := false } : _root_.Flock.Shape) = sh := by
    cases sh
    dsimp only at hsh0
    subst hsh0
    rfl
  rw [hwith] at hpart
  have hnv : nv = vs.size := part_nv hpart hgs hj
  subst hnv
  exact rs_rows_size hrr

/-! ## The part's `v` and `out` ports -/

/-- V\*'s row ports of an algebra part, as `vstarRows` lists them before `v`. -/
def resRowsA (pt : VStar.Part) : List (String × ℕ × ℕ) :=
  ((pt.S.ports.map (·.1)).zip (pt.S.ports.map fun p => VStar.portWordsOf p.2)).map (fun p => (p.1, p.2, 16 * p.2))

theorem resRowsA_length (pt : VStar.Part) : (resRowsA pt).length = pt.S.ports.length := by
  simp [resRowsA]

/-- **V\*'s rows of an algebra part**: its row ports, then `v` of `portWordsOf nv` words, then `out` of 128; the unit's
input words are the row ports' and `v`'s. -/
theorem rs_rows_inner {j : ℕ} {nm : String} {pt : VStar.Part} {c : Circuit} {u : VStar.UnitC}
    (hu : (VStar.Stmt.inner j nm pt).unit = .ok u)
    (hr : c.ports.toList.map (fun p => (p.name, p.words, p.bits)) =
      RecVStar.vstarRows (VStar.Stmt.inner j nm pt).sub u) :
    c.ports.toList.map (fun p => (p.name, p.words, p.bits)) = resRowsA pt ++
        [("v", VStar.portWordsOf pt.S.gf.nv, 16 * VStar.portWordsOf pt.S.gf.nv), ("out", 128, 16 * 128)] ∧
      ((resRowsA pt).map (·.2.1)).sum + VStar.portWordsOf pt.S.gf.nv = u.inWords := by
  have hu' : VStar.innerUnit pt.S nm = .ok u := hu
  obtain ⟨-, -, hin, hout, -⟩ := FlockVBridge.OfVStar.innerUnit_ok hu'
  have hlen : (pt.S.ports.map (·.1)).length = (pt.S.ports.map fun p => VStar.portWordsOf p.2).length := by simp
  refine ⟨?_, ?_⟩
  · rw [hr]
    have e1 : (VStar.Stmt.inner j nm pt).sub.inputs = pt.S.ports.map (·.1) ++ ["v"] := rfl
    have e2 : (VStar.Stmt.inner j nm pt).sub.outputs = ["out"] := rfl
    unfold RecVStar.vstarRows
    rw [e1, e2, hin, hout, FlockVBridge.OfVStar.repPorts, List.zip_append hlen, List.map_append, List.append_assoc]
    rfl
  · have hAw : (resRowsA pt).map (·.2.1) = pt.S.ports.map fun p => VStar.portWordsOf p.2 := by
      unfold resRowsA
      rw [List.map_map]
      exact List.map_snd_zip (le_of_eq hlen.symm)
    rw [hAw]
    unfold VStar.UnitC.inWords
    rw [hin, FlockVBridge.OfVStar.repPorts, rs_foldl_sum]
    simp only [List.sum_append, List.sum_cons, List.sum_nil] <;> omega

theorem rs_findIdx_app : ∀ (L : List String) (nm : String) (rest : List String), nm ∉ L →
    (L ++ nm :: rest).findIdx? (· == nm) = some L.length
  | [], nm, rest, _ => by simp [List.findIdx?_cons]
  | a :: L, nm, rest, h => by
    have ha : a ≠ nm := fun e => h (e ▸ List.mem_cons_self ..)
    have hL : nm ∉ L := fun hm => h (List.mem_cons_of_mem _ hm)
    rw [List.cons_append, List.findIdx?_cons, rs_findIdx_app L nm rest hL]
    simp [ha]

/-- **`v` and `out` are found where V\*'s rows put them**, the ports named once. -/
theorem rs_vOut {C : Circuit} {A : List (String × ℕ × ℕ)} {wv wo : ℕ}
    (hr : C.ports.toList.map (fun p => (p.name, p.words, p.bits)) =
      A ++ [("v", wv, 16 * wv), ("out", wo, 16 * wo)])
    (hnd : (C.ports.toList.map (·.name)).Nodup) :
    C.ports.findIdx? (·.name == "v") = some A.length ∧ C.ports.findIdx? (·.name == "out") = some (A.length + 1) ∧
      ∃ ptv pto, C.ports[A.length]? = some ptv ∧ C.ports[A.length + 1]? = some pto ∧ ptv.words = wv ∧
        pto.words = wo := by
  have hn : C.ports.toList.map (·.name) = A.map Prod.fst ++ ["v", "out"] := by
    have e : (["v", "out"] : List String) = [("v", wv, 16 * wv), ("out", wo, 16 * wo)].map Prod.fst := rfl
    rw [e, ← List.map_append, ← hr, List.map_map]
    try rfl
  have hnd' := hnd
  rw [hn] at hnd'
  obtain ⟨-, -, hdis⟩ := List.nodup_append.1 hnd'
  have hvA : "v" ∉ A.map Prod.fst := fun h => hdis _ h "v" (by simp) rfl
  have hoA : "out" ∉ A.map Prod.fst := fun h => hdis _ h "out" (by simp) rfl
  have hf : ∀ nm, C.ports.findIdx? (·.name == nm) = (A.map Prod.fst ++ ["v", "out"]).findIdx? (· == nm) := by
    intro nm
    rw [← hn, List.findIdx?_map]
    exact List.findIdx?_toArray _ C.ports.toList
  have hnone : ∀ nm, nm ∉ A.map Prod.fst → (A.map Prod.fst).findIdx? (· == nm) = none := by
    intro nm hnm
    refine List.findIdx?_eq_none_iff.2 fun y hy => ?_
    have hne : y ≠ nm := fun he => hnm (he ▸ hy)
    simpa using hne
  have hAl : (A.map Prod.fst).length = A.length := by simp
  have hvo : ("v" == "out") = false := by decide
  have hlen := congrArg List.length hr
  simp only [List.length_map, Array.length_toList, List.length_append, List.length_cons, List.length_nil] at hlen
  have h1 : A.length < C.ports.size := by omega
  have h2 : A.length + 1 < C.ports.size := by omega
  refine ⟨?_, ?_, ?_⟩
  · rw [hf, show A.map Prod.fst ++ ["v", "out"] = A.map Prod.fst ++ "v" :: ["out"] from rfl,
      rs_findIdx_app _ _ _ hvA, hAl]
  · have hoA' : "out" ∉ A.map Prod.fst ++ ["v"] := fun h =>
      (List.mem_append.1 h).elim hoA (fun h' => by simp at h')
    rw [hf, show A.map Prod.fst ++ ["v", "out"] = (A.map Prod.fst ++ ["v"]) ++ "out" :: [] by simp,
      rs_findIdx_app _ _ _ hoA']
    simp
  · have hv := congrArg (·[A.length]?) hr
    have ho := congrArg (·[A.length + 1]?) hr
    simp only [List.getElem?_map, Array.getElem?_toList, Array.getElem?_eq_getElem h1, Option.map_some,
      List.getElem?_append_right (le_refl A.length), Nat.sub_self, List.getElem?_cons_zero,
      Option.some.injEq, Prod.mk.injEq] at hv
    simp only [List.getElem?_map, Array.getElem?_toList, Array.getElem?_eq_getElem h2, Option.map_some,
      List.getElem?_append_right (by omega : A.length ≤ A.length + 1), Nat.add_sub_cancel_left,
      List.getElem?_cons_succ, List.getElem?_cons_zero, Option.some.injEq, Prod.mk.injEq] at ho
    exact ⟨_, _, Array.getElem?_eq_getElem h1, Array.getElem?_eq_getElem h2, hv.2.1, ho.2.1⟩

/-- **`HmOut.checkRows` holds the input leaves to the input rows**: every unit leaf is below `HmOut.inWords`. -/
theorem rs_leavesIn_lt {c : Circuit} (h : HmOut.checkRows c = .ok ()) {l : Array Int} (hl : l ∈ c.leavesIn)
    {y : Int} (hy : y ∈ l) : y < (HmOut.inWords c : Int) := by
  unfold HmOut.checkRows at h
  obtain ⟨k, -, h⟩ := FlockVBridge.PartS.bind_ok h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) names jp at h
  obtain ⟨-, h⟩ := FlockSoundness.ite_throw_ok' h
  simp only [jp] at h
  split at h
  · cases h
  · rename_i hn
    apply Classical.byContradiction
    intro hlt
    apply hn
    have hb : l.any (fun x => decide (x ≥ (HmOut.inWords c : Int))) = true :=
      Array.any_eq_true'.2 ⟨y, hy, decide_eq_true (by omega)⟩
    have hb2 : c.leavesIn.any (fun l => l.any (fun x => decide (x ≥ (HmOut.inWords c : Int)))) = true :=
      Array.any_eq_true'.2 ⟨l, hl, hb⟩
    first
      | (rw [Bool.or_eq_true]; exact Or.inl hb2)
      | simp [hb2]

/-- **The row after V\*'s row ports is an input row**: were it an output row, its words would be whole chunks
(`OutRows`), so the unit's input leaves, past the rows before it, would reach past the input rows' words. -/
theorem rs_vIn {C : Circuit} {A : List (String × ℕ × ℕ)} {a b : String × ℕ × ℕ} {N : ℕ}
    (hr : C.ports.toList.map (fun p => (p.name, p.words, p.bits)) = A ++ [a, b])
    (hcr : HmOut.checkRows C = .ok ()) (hrw : FlockSoundness.Discharge.Hidden.OutRows C)
    (hch : ∀ p ∈ C.ports.toList, 0 < p.chunks)
    (hl : C.leavesIn.getD 0 #[] = (Array.range N).map Int.ofNat)
    (hN : (A.map (·.2.1)).sum + a.2.1 ≤ N) :
    A.length < HmOut.nIn C := by
  have hlen := congrArg List.length hr
  simp only [List.length_map, Array.length_toList, List.length_append, List.length_cons, List.length_nil] at hlen
  have hnI : HmOut.nIn C = C.ports.size - C.outPorts.size := by
    unfold HmOut.nIn
    rw [FlockSoundness.Discharge.Hidden.OutRows.size hrw]
  apply Classical.byContradiction
  intro hle
  have hj : A.length - HmOut.nIn C < C.outPorts.size := by omega
  obtain ⟨-, hw⟩ := hrw.2.2 _ hj
  have he : HmOut.nIn C + (A.length - HmOut.nIn C) = A.length := by omega
  rw [he] at hw
  have hAs : A.length < C.ports.size := by omega
  have hget : C.ports.getD A.length default = C.ports[A.length] := by
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hAs, Option.getD_some]
  rw [hget] at hw
  have hwa : C.ports[A.length].words = a.2.1 := by
    have hg := congrArg (·[A.length]?) hr
    simp only [List.getElem?_map, Array.getElem?_toList, Array.getElem?_eq_getElem hAs, Option.map_some,
      List.getElem?_append_right (le_refl A.length), Nat.sub_self, List.getElem?_cons_zero,
      Option.some.injEq] at hg
    exact congrArg (fun t : String × ℕ × ℕ => t.2.1) hg
  have hc := hch _ (Array.getElem_mem_toList hAs)
  have hiw : HmOut.inWords C ≤ (A.map (·.2.1)).sum := by
    have he2 : HmOut.inWords C = ((C.ports.toList.take (HmOut.nIn C)).map (·.words)).sum :=
      extract_words Port.words C.ports (HmOut.nIn C)
    have h2 : ((C.ports.toList.take A.length).map (·.words)).sum = (A.map (·.2.1)).sum := by
      have h := congrArg (fun l => (l.take A.length).map (fun r : String × ℕ × ℕ => r.2.1)) hr
      simp only [List.take_left] at h
      rw [← h, ← List.map_take, List.map_map]
      rfl
    rw [he2, ← h2, List.map_take, List.map_take]
    exact rs_sum_take_le _ (by omega)
  have hmem : Int.ofNat (N - 1) ∈ C.leavesIn.getD 0 #[] := by
    rw [hl]
    exact Array.mem_map.2 ⟨N - 1, by simp <;> omega, rfl⟩
  have h0 : 0 < C.leavesIn.size := by
    apply Classical.byContradiction
    intro h0
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_none (by omega)] at hmem
    simp at hmem
  have hin : C.leavesIn.getD 0 #[] ∈ C.leavesIn := by
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem h0, Option.getD_some]
    exact Array.getElem_mem h0
  have hlt : ((N - 1 : ℕ) : Int) < (HmOut.inWords C : Int) := rs_leavesIn_lt hcr hin hmem
  omega

/-! ## At a staged part -/

/-- **A staged `rec-residuals` part's site** (`resTie_resVRead`'s site hypotheses, from the verifier's checks on the
part's table): its circuit is table 0's, holding two or more instances, `0 < c.g`, one unit per instance, each instance
`n < 2` in a block; `v` is the port after V\*'s row ports, an input row of `portWordsOf nv` words, and `out` the port after
it, an output row of 128 words. -/
theorem resSite_staged {st : Stage} {s : ℕ} {x xO : Msg} {w : Claims.Flock.Wit} (hS : StagedPart st s x xO)
    (hck : Claims.Flock.check (deployZkRec Flock.Readable.relation).parse xO w = true)
    {r : XR} (hr : readX x = some r) {ss : Array VStar.Stmt} (hss : RecVStar.statements r.shape = .ok ss)
    {j : ℕ} {nm' : String} {pt : VStar.Part} (hsv : ss[s]? = some (.inner j nm' pt)) {c : Circuit}
    (hpc : r.w.parts[s]?.bind partCircuit = some c) {p : Flock.Readable.Parsed}
    (hx : Flock.Readable.parseBytes readOfO xO = some p) :
    ∃ h0 : 0 < p.sts.size, p.sts[0].c = c ∧ RecVStar.checkPart (.inner j nm' pt) c = .ok () ∧
      2 ≤ p.sts[0].pub.n ∧ 0 < c.g ∧ c.upv = 1 ∧ (∀ n < 2, n / c.g < 2 ^ (p.sts[0].m - c.kLog)) ∧
      c.ports.findIdx? (·.name == "v") = some pt.S.ports.length ∧
      c.ports.findIdx? (·.name == "out") = some (pt.S.ports.length + 1) ∧
      pt.S.ports.length < HmOut.nIn c ∧ ¬ pt.S.ports.length + 1 < HmOut.nIn c ∧
      ∃ ptv pto, c.ports[pt.S.ports.length]? = some ptv ∧ c.ports[pt.S.ports.length + 1]? = some pto ∧
        ptv.words = VStar.portWordsOf pt.S.gf.nv ∧ pto.words = 128 := by
  obtain ⟨p', r', sv, hx', ht', hpi, hr', ⟨ss', hss', hsv'⟩, h0, htab⟩ := stagedTables hS
  rw [hx] at hx'
  cases hx'
  rw [hr] at hr'
  cases hr'
  rw [hss] at hss'
  cases hss'
  rw [hsv] at hsv'
  cases hsv'
  have hS' := hS
  obtain ⟨-, i0, spec, sts, -, hsu, -⟩ := hS'
  have hp : p = { i0.base.read with sts } := Option.some.inj (hx.symm.trans (parse_of_setupO hsu))
  have hsz : p.sts.size = 1 := by rw [hp]; exact (oneFull_staged hS hsu).1
  have hpl : Flock.RecVStar.population p.sts[0] ≤ p.sts[0].n := by
    have h2 := (oneFull_staged hS hsu).2
    subst hp
    exact h2 0 h0
  have hc0 := partCircuit_staged hS hx ht' hpi hr hpc h0
  obtain ⟨L, regs, hz⟩ := Flock.Readable.layout_of_check hck hx ht' hpi h0
  have hz1 : (L.model regs).Satisfies (Flock.Readable.zOf 1 w 0 p.sts[0].m) := by
    have e := congrArg (fun J => Flock.Readable.zOf J w 0 p.sts[0].m) hsz
    rw [← e]
    exact hz
  obtain ⟨_, P⟩ := Flock.Readable.parseFacts_of_parse hx ht' hpi h0
    (FlockSoundness.Discharge.Recursion.checkParsed_flat (htab 0 h0).2.2)
  obtain ⟨-, hts⟩ := tablesOf_of_parse hx
  obtain ⟨spec0, hset⟩ := Flock.Readable.setup_of_tablesOf hts
  unfold Flock.Readable.Read.setup at hset
  rw [hpi] at hset
  obtain ⟨-, hall⟩ := FlockSoundness.Discharge.HiddenExec.setupTables_setupOne hset
  obtain ⟨d, hd, -⟩ := hall 0 h0
  have htd : (FlockSoundness.Discharge.Exec.tagsAt p.tags p.I.count 0).typed = false :=
    (FlockSoundness.Discharge.Exec.tagsAt_typed _ _ _).trans ht'
  have hh : (FlockSoundness.Discharge.Exec.tagsAt p.tags p.I.count 0).hiddenOutputs = true := by
    unfold FlockSoundness.Discharge.Exec.tagsAt
    split <;> exact staged_hidden hS hx h0
  obtain ⟨hscope, r2, ss2, sv2, hr2, hss2, hsv2, hpop⟩ := staged_scope hS hx h0
  rw [hr] at hr2
  cases hr2
  rw [hss] at hss2
  cases hss2
  rw [hsv] at hsv2
  cases hsv2
  have hs := (FlockSoundness.Discharge.FailClosed.scopeOk_eq _).symm.trans hscope
  have hv1 := FlockSoundness.Discharge.Layout.scopeOk_ports hs
  have ho : p.sts[0].c.outNet = p.sts[0].c.unitNet := by
    unfold FlockSoundness.Discharge.Layout.scopeOk at hs
    simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hs
    exact hs.1.1.1.2
  have hrw := outRows_setupOne htd hh hd hv1
  obtain ⟨hcr, hrr⟩ := MoreSlots.checkRows_setupOne htd hh hd hv1
  obtain ⟨u, hu, hupv, -, hlin, hrows⟩ := FlockSoundness.Discharge.Recursion.checkParsed_ok (htab 0 h0).2.2
  obtain ⟨u', hu', hlo⟩ := FlockSoundness.Discharge.Recursion.checkParsed_leavesOut (htab 0 h0).2.2
  rw [hu] at hu'
  cases hu'
  obtain ⟨-, hg0, hnbl, -⟩ := table_rows htd hd (by omega) hv1 L regs hz1
  have hblk : ∀ n < 2, n / p.sts[0].c.g < 2 ^ (p.sts[0].m - p.sts[0].c.kLog) := by
    intro n hn
    rw [L.m, Nat.add_sub_cancel_left]
    exact div_lt_blocks_two hn hnbl
  obtain ⟨ls, hls⟩ := P.check
  obtain ⟨-, -, hlv⟩ := FlockSoundness.Discharge.Layout.check_out P.typed ho hls
  have hget0 : p.sts[0].c.leavesOut.getD 0 #[] = p.sts[0].c.leavesOut[0]! := by
    rw [FlockSoundness.Discharge.Hidden.getElem!_eq_getD']; rfl
  have hnd := MoreSlots.staged_names hS hx h0
  have hpub := setupOne_n hd
  have hinst : (VStar.Stmt.inner j nm' pt).instances = 2 := rfl
  have hcp := (htab 0 h0).2.1
  subst hc0
  obtain ⟨hR, hsum⟩ := rs_rows_inner hu hrows
  obtain ⟨hqv, hqo, ptv, pto, hptv, hpto, hwv, hwo⟩ := rs_vOut hR hnd
  have hvin := rs_vIn hR hcr hrw (fun q hq => (hv1 q hq).2.2) hlin (le_of_eq hsum)
  have hm : "out" ∈ (VStar.Stmt.inner j nm' pt).sub.outputs := by
    show "out" ∈ ["out"]
    exact List.mem_singleton.2 rfl
  have hoin := MoreSlots.pastIn_core hrows hnd hcr hrr (fun q hq => (hv1 q hq).2.1) hlo
    (fun o ho => by rw [hget0] at ho; exact (hlv 0 (by omega)).2 o ho) (MoreSlots.unit_outs hu) hm hqo
  rw [resRowsA_length] at hqv hqo hptv hpto hvin hoin
  exact ⟨h0, rfl, hcp, by omega, hg0, hupv, hblk, hqv, hqo, hvin, by omega, ptv, pto, hptv, hpto, hwv, hwo⟩

/-- **`resTie_resVRead` with its site discharged**: at a dense statement's staged `rec-residuals` part, the site
(`resSite_staged`) and check 5's rows (`resRows_dense`) give every hypothesis `resTie_resVRead` takes but the table's
labels (`hidx`), at instance `o + g < 2`. Also the block facts `resPieces_computesAt` asks of the piece at key
`(jj, o, g, 0)`. -/
theorem resTie_dense {b : Brings} {IV : Claims.Verifier Coin}
    {elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))} {s : ℕ} {x xO : Msg}
    {w : Claims.Flock.Wit} {cs : ℕ → Coin} {Bk : Msg → ℕ} {st0 : Claims.Recursion.Stmt}
    (hst : RecDeploy.parse x = some st0) (hd : Claims.Recursion.Form.Dense st0.form)
    (hB : RestRead.kI x ≤ Bk (xIOf x))
    (hT : ∀ e ∈ (DeployedR.layR elem).calls x s w, FlockDeploy.sha512 e.1 = e.2)
    (hnc : (Claims.findColl ((DeployedR.layR elem).calls x s w ++ vcallsAt FlockDeploy.sha512
      (recBound _root_.Flock.Readable.relation b (DeployedR.layR elem) IV) xO)).isSome = false)
    (hsh : showsPart recRead b s x xO cs)
    (hck : Claims.Flock.check (deployZkRec Flock.Readable.relation).parse xO w = true)
    {r : XR} (hr : readX x = some r) {ss : Array VStar.Stmt} (hss : RecVStar.statements r.shape = .ok ss)
    {j : ℕ} {nm' : String} {pt : VStar.Part} (hsv : ss[s]? = some (.inner j nm' pt)) {c : Circuit}
    (hpc : r.w.parts[s]?.bind partCircuit = some c) {p : Flock.Readable.Parsed}
    (hx : Flock.Readable.parseBytes readOfO xO = some p) (ht : 0 < p.sts.size)
    {i : RecIns} {spec : SessionSpec} {sts : Array Stmt} (hsu : setupO xO = some (i, spec, sts)) (h0 : 0 < sts.size)
    (hidx : Registered.unitIndices sts[0].pub = .ok (Array.range sts[0].pub.n))
    {m k cl jj : ℕ} (hres : RecResiduals.params sts[0].c.unit.name = some (m, k, cl, jj))
    (hset : ∀ u ∈ resSetup xO, u ∈ b.setup xO)
    {sh : _root_.Flock.Shape} {holo : Bool} {S : _root_.Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ}
    (hshp : RecResiduals.shape? m k cl = some sh) (hholo : RecResiduals.holoOf sts[0].c = .ok holo)
    (hpart : RecResiduals.part { sh with holo } jj = .ok (S, ns, nv))
    {o g : ℕ} (hn2 : o + g < 2) :
    0 < p.sts[0].c.g ∧ (o + g) / p.sts[0].c.g < 2 ^ (p.sts[0].m - p.sts[0].c.kLog) ∧ 0 < p.sts[0].c.upv ∧
      ResTie S ns nv (Rows.resVRead (xIOf x) ((List.range (Bk (xIOf x))).map cs) (jj, o, g, 0))
        (keyCopy p ht w (jj, o, g, 0)) := by
  obtain ⟨h0', hc0, hcp, hpn, hg0, hupv, hblk, hqv, hqo, hvin, hoout, ptv, pto, hptv, hpto, hwv, hpw⟩ :=
    resSite_staged hsh.2 hck hr hss hsv hpc hx
  have hp : p = { i.base.read with sts } := Option.some.inj (hx.symm.trans (parse_of_setupO hsu))
  have e0 : p.sts[0] = sts[0] := by subst hp; rfl
  have hc : sts[0].c = c := by rw [← e0]; exact hc0
  refine ⟨by rw [hc0]; exact hg0, by rw [hc0]; exact hblk _ hn2, by rw [hc0, hupv]; decide, ?_⟩
  have hres' : RecResiduals.params c.unit.name = some (m, k, cl, jj) := hc ▸ hres
  have hholo' : RecResiduals.holoOf c = .ok holo := hc ▸ hholo
  obtain ⟨-, hnv⟩ := checkPart_ns hcp hres' hshp hholo' hpart
  have hG : 0 < sts[0].c.g := by rw [hc]; exact hg0
  have hpn' : 2 ≤ sts[0].pub.n := by rw [← e0]; exact hpn
  have hnp : o + g < sts[0].pub.n := lt_of_lt_of_le hn2 hpn'
  refine resTie_resVRead (o := o) (g := g) (n := o + g) hst hd hB hT hnc hsh hck hr hss hsv hpc hx ht hsu h0 hidx
    hres hset hshp hholo hpart hG rfl hn2 hnp hqv hqo hvin hoout hptv hpto ?_ ?_
  · intro f hf inn rows hload hrows
    obtain ⟨hsz, hrw⟩ := resRows_dense hst hd hsh hsu h0 hres hshp hholo hpart hf hload hrows
    have h1 := hrw (o + g) hn2
    have e2 : ptv.words = (8 * nv + 63) / 64 * 64 := by rw [hwv, hnv]; rfl
    rw [rs_padTo_size, h1]
    unfold Registered.rowWords RecResiduals.portWordsOf
    refine ⟨by omega, by omega, by omega⟩
  · rw [rs_padTo_size, rs_zeroRow_size hn2, hpw]
    all_goals (unfold Registered.rowWords; omega)

end RecDeploy.ResLower
