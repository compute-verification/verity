module

public import Proofs.Properties.Flock.Circuit.LowerEval.Nested

/-! `GateWords`: a template of `ScanNested` that reaches no word primitive is bit-typed all the way down, so each of its
words is one bit, and its walk on bits is its walk on words, word for bit. -/

@[expose] public section

namespace Flock.Chain

open Flock Flock.Lower Flock.Tmpl Flock.TmplValue Flock.Chain.LowerDef
open Flock.Program hiding Form
open InferenceOnlyVerifier.Properties.SecurityProofs

theorem widths_one_fix {t : Ty} (h : Ty.bitType t = t) : ∀ w ∈ t.widths.toList, w = 1 := by
  intro w hw
  rw [← h, bitTypeWidths_go] at hw
  obtain ⟨u, -, hu⟩ := List.mem_flatMap.1 hw
  exact List.eq_of_mem_replicate hu

theorem widths_one {t : Ty} (h : isBit t = true) : ∀ w ∈ t.widths.toList, w = 1 :=
  widths_one_fix (bitType_of_isBit t h)

theorem ones_of_map {l : List (List Bool)} {ws : List ℕ} (h : l.map List.length = ws) (h1 : ∀ u ∈ ws, u = 1) :
    ∀ w ∈ l, w.length = 1 :=
  fun _ hw => h1 _ (h ▸ List.mem_map_of_mem (f := List.length) hw)

theorem sum_ones : ∀ {l : List ℕ}, (∀ u ∈ l, u = 1) → l.sum = l.length
  | [], _ => rfl
  | u :: l, h => by
    rw [List.sum_cons, List.length_cons, h u (List.mem_cons_self ..), sum_ones fun v hv => h v (List.mem_cons_of_mem _ hv)]
    omega

theorem flatten_ones : ∀ {l : List (List Bool)}, (∀ w ∈ l, w.length = 1) → l.flatten = l.map (·.headD false)
  | [], _ => rfl
  | [b] :: l, h => by
    rw [List.flatten_cons, List.map_cons, flatten_ones fun w hw => h w (List.mem_cons_of_mem _ hw)]
    rfl
  | [] :: _, h => by simp at h
  | (_ :: _ :: _) :: _, h => by simp at h

theorem refVal_ones {z : List (List Bool)} {st : Array ℕ} {outs : Array (List (List Bool))}
    (hz : ∀ w ∈ z, w.length = 1) (ho : ∀ i, ∀ w ∈ outs.getD i [], w.length = 1) (r : Ref) :
    Flock.Program.refVal z.flatten st (outs.map List.flatten) r = (Flock.Program.refVal z st outs r).headD false := by
  unfold Flock.Program.refVal
  rw [getD_map_flatten, flatten_ones hz, flatten_ones (ho r.i)]
  split
  · rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_map]
    cases (z[st.getD r.i 0 + r.j]?) <;> rfl
  · rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_map]
    cases ((outs.getD r.i [])[r.j]?) <;> rfl

theorem argVals_ones {z : List (List Bool)} {st : Array ℕ} {outs : Array (List (List Bool))}
    (hz : ∀ w ∈ z, w.length = 1) (ho : ∀ i, ∀ w ∈ outs.getD i [], w.length = 1) (a : Arg) (cnt : ℕ) (k : ℕ → ℕ) :
    Flock.Program.argVals (Flock.Program.refVal z.flatten st (outs.map List.flatten)) a cnt k =
      (Flock.Program.argVals (Flock.Program.refVal z st outs) a cnt k).map (·.headD false) := by
  simp only [Flock.Program.argVals, List.map_map, Function.comp_def, refVal_ones hz ho]

theorem tyAt_bit {D : Defn} (hb : BitTyped D) {k : ℕ} {r : Ref} (hr : RefIn D k r) : isBit (tyAt D r) = true := by
  unfold tyAt
  split
  · rename_i hp
    have hi := hr.1 hp
    rw [show D.params.getD r.i default = D.params[r.i] from by simp [Array.getD, hi]]
    exact hb.1 _ (Array.getElem_mem hi)
  · rename_i hp
    have hi := (hr.2.1 (by simpa using hp)).2
    rw [show D.body.nodes.getD r.i default = D.body.nodes[r.i] from by simp [Array.getD, hi]]
    exact hb.2 _ (Array.getElem_mem hi)

theorem refVal_len1 {D : Defn} (hb : BitTyped D) {x : List (List Bool)} (hx : x.map List.length = leafWidths D)
    {outs : Array (List (List Bool))} (ho : OutsOk D outs) {r : Ref} (hr : RefIn D outs.size r) :
    (Flock.Program.refVal x (Flock.Program.paramStarts D.params) outs r).length = 1 := by
  rw [(refBits hx ho hr).2]
  apply widths_one (tyAt_bit hb hr)
  rw [getElem!_pos (tyAt D r).widths r.j hr.2.2]
  exact Array.mem_toList_iff.2 (Array.getElem_mem hr.2.2)

theorem leaf_ones {D : Defn} (hb : BitTyped D) : ∀ u ∈ leafWidths D, u = 1 := by
  intro u hu
  obtain ⟨p, hp, hu⟩ := List.mem_flatMap.1 hu
  exact widths_one (hb.1 p (Array.mem_toList_iff.1 hp)) u hu

theorem words_one {D : Defn} (hb : BitTyped D) {x : List (List Bool)} (hx : x.map List.length = leafWidths D) :
    ∀ w ∈ x, w.length = 1 :=
  ones_of_map hx (leaf_ones hb)

theorem outs_one {D : Defn} (hb : BitTyped D) {outs : Array (List (List Bool))} (ho : OutsOk D outs)
    (hsz : outs.size ≤ D.body.nodes.size) : ∀ i, ∀ w ∈ outs.getD i [], w.length = 1 := by
  intro i w hw
  by_cases hi : i < outs.size
  · have h1 := ho i hi
    have hn : D.body.nodes.getD i default ∈ D.body.nodes := by
      rw [show D.body.nodes.getD i default = D.body.nodes[i] from by simp [Array.getD, show i < D.body.nodes.size by omega]]
      exact Array.getElem_mem _
    exact widths_one (hb.2 _ hn) _ (h1 ▸ List.mem_map_of_mem (f := List.length) hw)
  · simp [Array.getD, hi] at hw

theorem batchArg_ones {z : List (List Bool)} {st : Array ℕ} {outs : Array (List (List Bool))}
    (hz : ∀ w ∈ z, w.length = 1) (ho : ∀ i, ∀ w ∈ outs.getD i [], w.length = 1) (C : Defn) (nd : Node) (m i : ℕ) :
    batchArg (Flock.Program.refVal z.flatten st (outs.map List.flatten)) C nd m i =
      (batchArg (Flock.Program.refVal z st outs) C nd m i).map (·.headD false) := by
  unfold batchArg
  split <;> exact argVals_ones hz ho _ _ _

theorem scanArg_ones {z : List (List Bool)} {st : Array ℕ} {outs : Array (List (List Bool))}
    (hz : ∀ w ∈ z, w.length = 1) (ho : ∀ i, ∀ w ∈ outs.getD i [], w.length = 1) (C : Defn) (nd : Node) (m i : ℕ) :
    scanArg (Flock.Program.refVal z.flatten st (outs.map List.flatten)) C nd m i =
      (scanArg (Flock.Program.refVal z st outs) C nd m i).map (·.headD false) := by
  unfold scanArg
  split <;> exact argVals_ones hz ho _ _ _

theorem argIns_one {D : Defn} (hb : BitTyped D) {x : List (List Bool)} (hx : x.map List.length = leafWidths D)
    {outs : Array (List (List Bool))} (ho : OutsOk D outs) {a : Arg}
    (ha : a.seq.count = a.seq.toArray.size ∧ ∀ r ∈ a.seq.toArray.toList, RefIn D outs.size r) :
    ∀ w ∈ Flock.Program.argVals (Flock.Program.refVal x (Flock.Program.paramStarts D.params) outs) a a.seq.count id,
      w.length = 1 := by
  intro w hw
  simp only [Flock.Program.argVals, List.mem_map, List.mem_range, id] at hw
  obtain ⟨t, ht, rfl⟩ := hw
  rw [ha.1] at ht
  have hmem : a.seq.toArray.getD t default ∈ a.seq.toArray.toList := by
    rw [show a.seq.toArray.getD t default = a.seq.toArray[t] from by simp [Array.getD, ht]]
    exact Array.mem_toList_iff.2 (Array.getElem_mem ht)
  exact refVal_len1 hb hx ho (ha.2 _ hmem)

theorem callIns_one {D : Defn} (hb : BitTyped D) {x : List (List Bool)} (hx : x.map List.length = leafWidths D)
    {outs : Array (List (List Bool))} (ho : OutsOk D outs) (nd : Node)
    (ha : ∀ a ∈ nd.args.toList, a.seq.count = a.seq.toArray.size ∧ ∀ r ∈ a.seq.toArray.toList, RefIn D outs.size r) :
    ∀ w ∈ nd.args.toList.flatMap (fun a =>
      Flock.Program.argVals (Flock.Program.refVal x (Flock.Program.paramStarts D.params) outs) a a.seq.count id),
      w.length = 1 := by
  intro w hw
  obtain ⟨a, ha', hw'⟩ := List.mem_flatMap.1 hw
  exact argIns_one hb hx ho (ha a ha') w hw'

theorem batchIns_one {D : Defn} (hb : BitTyped D) {x : List (List Bool)} (hx : x.map List.length = leafWidths D)
    {outs : Array (List (List Bool))} (ho : OutsOk D outs) {n : ℕ} (hsz : outs.size = n) {C : Defn}
    (hargs : ∀ a ∈ (D.body.nodes[n]!).args.toList,
      a.seq.count = a.seq.toArray.size ∧ ∀ r ∈ a.seq.toArray.toList, RefIn D n r)
    (hbat : ∀ i < (D.body.nodes[n]!).args.size,
      (∀ ax, (D.body.nodes[n]!).axes[i]? = some (some ax) →
        ((D.body.nodes[n]!).args.getD i default).ty.removeAxis ax = some ((D.body.nodes[n]!).n, (C.params.getD i default).2) ∧
        ((D.body.nodes[n]!).args.getD i default).dims = ((D.body.nodes[n]!).args.getD i default).ty.axisDims ax ∧
        i < C.params.size ∧
        ((D.body.nodes[n]!).args.getD i default).seq.toArray.toList.map (fun r => (tyAt D r).widths[r.j]!) =
          ((D.body.nodes[n]!).args.getD i default).ty.widths.toList) ∧
      ((∀ ax, (D.body.nodes[n]!).axes[i]? ≠ some (some ax)) → ((D.body.nodes[n]!).args.getD i default).dims = none))
    (hC1 : ∀ i < C.params.size, ∀ u ∈ (C.params.getD i default).2.widths.toList, u = 1)
    {m : ℕ} (hm' : m < (D.body.nodes[n]!).n) :
    ∀ w ∈ (List.range (D.body.nodes[n]!).args.size).flatMap
      (batchArg (Flock.Program.refVal x (Flock.Program.paramStarts D.params) outs) C (D.body.nodes[n]!) m),
      w.length = 1 := by
  intro w hw
  obtain ⟨i, hi, hw'⟩ := List.mem_flatMap.1 hw
  have hi' := List.mem_range.1 hi
  have hmemA : (D.body.nodes[n]!).args.getD i default ∈ (D.body.nodes[n]!).args.toList := by
    simp [Array.getD, hi']
  obtain ⟨hcnt, hrefs⟩ := hargs _ hmemA
  have hrefs' : ∀ r ∈ ((D.body.nodes[n]!).args.getD i default).seq.toArray.toList, RefIn D outs.size r :=
    fun r h => by rw [hsz]; exact hrefs r h
  by_cases hax : ∃ ax, (D.body.nodes[n]!).axes[i]? = some (some ax)
  · obtain ⟨ax, hax⟩ := hax
    obtain ⟨hrm, hdm, hip, hwd⟩ := (hbat i hi').1 ax hax
    obtain ⟨o, F, htw, hEw, hdF, -⟩ := removeAxis_blocks ax hrm
    have hlen : (batchArg (Flock.Program.refVal x (Flock.Program.paramStarts D.params) outs) C (D.body.nodes[n]!) m
        i).map List.length = (C.params.getD i default).2.widths.toList := by
      rw [batchArg_some _ _ _ _ _ (hdm.trans hdF)]
      simp only [Flock.Program.argVals]
      generalize (C.params.getD i default).2 = E at hEw ⊢
      rw [show E.leaves = E.widths.toList.length by rw [Array.length_toList, hb_widths_size],
        show F.leaves = F.widths.toList.length by rw [Array.length_toList, hb_widths_size], hEw,
        length_replicate_flatten]
      exact axisWords_ax hx ho hrefs' hm' (hwd.trans htw)
    exact ones_of_map hlen (hC1 i hip) w hw'
  · have hn0 : ∀ ax, (D.body.nodes[n]!).axes[i]? ≠ some (some ax) := fun ax h => hax ⟨ax, h⟩
    rw [batchArg_none _ _ _ _ _ ((hbat i hi').2 hn0)] at hw'
    exact argIns_one hb hx ho ⟨hcnt, hrefs'⟩ w hw'

/-- `scanStep`'s bit node as the node itself, in a bit-typed Definition `D` walked on bits as it is: the carry width
and the bits its carry and rest arguments read. -/
theorem scanBitArgs {D : Defn} (hb : BitTyped D) {x : List (List Bool)}
    (hx : x.map List.length = leafWidths D) {outs : Array (List (List Bool))} (ho : OutsOk D outs) {n : ℕ}
    (hnD : n < D.body.nodes.size) (hsz : outs.size = n) {C : Defn} (hC1 : ∀ u ∈ leafWidths C, u = 1)
    (hargs : ∀ a ∈ (D.body.nodes[n]).args.toList,
      a.seq.count = a.seq.toArray.size ∧ ∀ r ∈ a.seq.toArray.toList, RefIn D n r)
    (hcs : (D.body.nodes[n]).args.size = C.params.size) (h0 : 0 < C.params.size)
    (hw0 : ((D.body.nodes[n]).args.getD 0 default).seq.toArray.toList.map (fun r => (tyAt D r).widths[r.j]!) =
      (C.params.getD 0 default).2.widths.toList)
    (hw : ∀ i < (D.body.nodes[n]).args.size, 1 ≤ i →
      ((D.body.nodes[n]).args.getD i default).seq.toArray.toList.map (fun r => (tyAt D r).widths[r.j]!) =
        if i ≤ (D.body.nodes[n]).nxs then
          (List.replicate (D.body.nodes[n]).n (C.params.getD i default).2.widths.toList).flatten
        else (C.params.getD i default).2.widths.toList) :
    (C.params.getD 0 default).2.leaves = ((C.params.getD 0 default).2.widths.toList).sum ∧
      Flock.Program.argVals (Flock.Program.refVal x.flatten (Flock.Program.paramStarts D.params)
          (outs.map List.flatten)) ((D.body.nodes[n]).args.getD 0 default) (C.params.getD 0 default).2.leaves id =
        (Flock.Program.argVals (Flock.Program.refVal x (Flock.Program.paramStarts D.params) outs)
          ((D.body.nodes[n]).args.getD 0 default) (C.params.getD 0 default).2.leaves id).flatten ∧
      ∀ m < (D.body.nodes[n]).n,
        ((List.range (D.body.nodes[n]).args.size).drop 1).flatMap
          (scanArg (Flock.Program.refVal x.flatten (Flock.Program.paramStarts D.params) (outs.map List.flatten)) C
            (D.body.nodes[n]) m) =
        (((List.range (D.body.nodes[n]).args.size).drop 1).flatMap
          (scanArg (Flock.Program.refVal x (Flock.Program.paramStarts D.params) outs) C (D.body.nodes[n]) m)).flatten := by
  have hx1 := words_one hb hx
  have ho1 := outs_one hb ho (by omega)
  have ha0 : 0 < (D.body.nodes[n]).args.size := by omega
  have hargs' : ∀ a ∈ (D.body.nodes[n]).args.toList,
      a.seq.count = a.seq.toArray.size ∧ ∀ r ∈ a.seq.toArray.toList, RefIn D outs.size r := fun a ha => by
    obtain ⟨h1, h2⟩ := hargs a ha
    exact ⟨h1, fun r hr' => by rw [hsz]; exact h2 r hr'⟩
  have hmem0 : (D.body.nodes[n]).args.getD 0 default ∈ (D.body.nodes[n]).args.toList := by simp [Array.getD, ha0]
  have hinitW := argSharedWords hx ho (hargs' _ hmem0).2 hw0
  have hE0 : ∀ u ∈ (C.params.getD 0 default).2.widths.toList, u = 1 := fun u hu =>
    hC1 u (by rw [leafWidths_scan C h0]; exact List.mem_append_left _ hu)
  have hcl : (C.params.getD 0 default).2.leaves = ((C.params.getD 0 default).2.widths.toList).length := by
    rw [Array.length_toList, hb_widths_size]
  refine ⟨by rw [hcl, sum_ones hE0],
    by rw [argVals_ones hx1 ho1, flatten_ones (ones_of_map hinitW hE0)], fun m hm => ?_⟩
  have hl := scanRestWords hx ho hargs' hw hm
  rw [flatten_ones (ones_of_map hl fun u hu => hC1 u (by
    rw [leafWidths_scan C h0, ← hcs]; exact List.mem_append_right _ hu)), List.map_flatMap]
  exact List.flatMap_congr fun i _ => scanArg_ones hx1 ho1 C _ m i

/-- **`GateWords` holds**, on `NestedTyped`. -/
theorem gateWords_of (hT : NestedTyped) : GateWords := by
  intro bytes P L h hr hLP c
  induction c using Nat.strong_induction_on with
  | _ c ih =>
  intro hcl hnc hnw hgd z hz
  have hfacts := hT bytes P h hr c hcl hnc hgd
  cases hnc with
  | mk hp1 hltn hnest hgate h5 =>
  have hbt : BitTyped (P.defs[c]!) := h5.resolve_left hnw
  have hnwc : ∀ nd ∈ (P.defs[c]!).body.nodes, ¬ ReachesWord P nd.fn := fun nd hm hw => hnw (.comp hp1 hm hw)
  have hry' : ∀ nd ∈ (P.defs[c]!).body.nodes, Reach P c nd.fn := fun nd h' => .callee .root hp1 h'
  have hPy : P.defs[c]? = some (P.defs[c]!) := by rw [getElem!_pos P.defs c hcl]; exact getElem?_pos P.defs c hcl
  have hLy : L.defs[c]? = some (P.defs[c]!) := by rw [hLP c hcl]; exact hPy
  obtain ⟨hargs, hret, hretw, hord, hout, hcout, hbat, hcarg, hcbat, hscan⟩ := hfacts
  generalize hD : P.defs[c]! = D at hp1 hltn hnest hgate hbt hnwc hry' hPy hLy hargs hret hretw hord hout hcout hbat hcarg hcbat hscan hz ⊢
  have hz1 := words_one hbt hz
  have hstep : ∀ f', c ≤ f' → ∀ n (hn : n < D.body.nodes.toList.length) (outs : Array (List (List Bool))),
      outs.size = n → OutsOk D outs → ∃ w,
        nodeStep wordWires P (wordDef? P c) c z D outs D.body.nodes.toList[n] = some (outs.push w) ∧
        nodeStep Flock.Program.Wires.bool L (Flock.Program.evalDef? L f') c z.flatten D (outs.map List.flatten)
          D.body.nodes.toList[n] = some ((outs.push w).map List.flatten) ∧ OutsOk D (outs.push w) := by
    intro f' hf' n hn outs hsz ho
    have hnD : n < D.body.nodes.size := by simpa using hn
    simp only [Array.getElem_toList]
    have hmem : D.body.nodes[n] ∈ D.body.nodes := Array.getElem_mem hnD
    have hlt := hord _ hmem
    have hfnP : (D.body.nodes[n]).fn < P.defs.size := by omega
    have hcal : P.defs[(D.body.nodes[n]).fn]? = some (P.defs[(D.body.nodes[n]).fn]!) := by
      rw [getElem!_pos P.defs _ hfnP]; exact getElem?_pos P.defs _ hfnP
    have hcalL : L.defs[(D.body.nodes[n]).fn]? = some (P.defs[(D.body.nodes[n]).fn]!) := by
      rw [hLP _ hfnP]; exact hcal
    have hargsn : ∀ a ∈ (D.body.nodes[n]).args.toList,
        a.seq.count = a.seq.toArray.size ∧ ∀ r ∈ a.seq.toArray.toList, RefIn D outs.size r := fun a ha => by
      obtain ⟨h1, h2⟩ := hargs n hnD a ha
      exact ⟨h1, fun r hr' => by rw [hsz]; exact h2 r hr'⟩
    have ho1 := outs_one hbt ho (by omega)
    have hform2 : (D.body.nodes[n]).form ≠ .batch → (D.body.nodes[n]).form ≠ .scan →
        (D.body.nodes[n]).form = .prim ∨ (D.body.nodes[n]).form = .call := by
      intro hb hs
      cases hfm : (D.body.nodes[n]).form with
      | prim => exact Or.inl rfl
      | call => exact Or.inr rfl
      | batch => exact absurd hfm hb
      | scan => exact absurd hfm hs
    have hcallB : (D.body.nodes[n]).args.toList.flatMap (fun a => Flock.Program.argVals
          (Flock.Program.refVal z.flatten (Flock.Program.paramStarts D.params) (outs.map List.flatten)) a a.seq.count id) =
        ((D.body.nodes[n]).args.toList.flatMap fun a => Flock.Program.argVals
          (Flock.Program.refVal z (Flock.Program.paramStarts D.params) outs) a a.seq.count id).flatten := by
      rw [flatten_ones (callIns_one hbt hz ho _ hargsn), List.map_flatMap]
      exact List.flatMap_congr fun a _ => argVals_ones hz1 ho1 a _ _
    have hbatB : ∀ m, (∀ w ∈ (List.range (D.body.nodes[n]).args.size).flatMap
          (batchArg (Flock.Program.refVal z (Flock.Program.paramStarts D.params) outs) (P.defs[(D.body.nodes[n]).fn]!)
            (D.body.nodes[n]) m), w.length = 1) →
        (List.range (D.body.nodes[n]).args.size).flatMap (fun i =>
          batchArg (Flock.Program.refVal z.flatten (Flock.Program.paramStarts D.params) (outs.map List.flatten))
            (P.defs[(D.body.nodes[n]).fn]!) (D.body.nodes[n]) m i) =
        ((List.range (D.body.nodes[n]).args.size).flatMap fun i =>
          batchArg (Flock.Program.refVal z (Flock.Program.paramStarts D.params) outs) (P.defs[(D.body.nodes[n]).fn]!)
            (D.body.nodes[n]) m i).flatten := by
      intro m h1
      rw [flatten_ones h1, List.map_flatMap]
      exact List.flatMap_congr fun i _ => batchArg_ones hz1 ho1 _ _ m i
    by_cases hpc : (P.defs[(D.body.nodes[n]).fn]!).prim = true
    · have hwf : isWordPrim (P.defs[(D.body.nodes[n]).fn]!).prim (P.defs[(D.body.nodes[n]).fn]!).key = false := by
        cases hw : isWordPrim (P.defs[(D.body.nodes[n]).fn]!).prim (P.defs[(D.body.nodes[n]).fn]!).key with
        | false => rfl
        | true => exact absurd (ReachesWord.word hw) (hnwc _ hmem)
      obtain ⟨-, -, -, hpar, g, hgo⟩ := gateCallee_facts h hr hfnP (hgate _ hmem hpc hwf)
      have hB : ∀ y', Flock.Program.evalDef? L f' (D.body.nodes[n]).fn y' =
          some (Prog.evalBool P (D.body.nodes[n]).fn y') := fun y' => by
        obtain ⟨f'', rfl⟩ : ∃ f'', f' = f'' + 1 := ⟨f' - 1, by omega⟩
        rw [evalDef_gate hcalL hpc hgo, evalBool_gate hfnP hpc hgo]
      have hWp : ∀ ws, wordDef? P c (D.body.nodes[n]).fn ws =
          some [Prog.evalBool P (D.body.nodes[n]).fn ws.flatten] := fun ws => by
        obtain ⟨c', hc'⟩ : ∃ c', c = c' + 1 := ⟨c - 1, by omega⟩
        rw [hc']; exact wordDef_prim hfnP hpc ws
      by_cases hb : (D.body.nodes[n]).form = .batch
      · have hW := nodeStep_batch wordWires (fun _ => rfl) P (wordDef? P c) c z D outs (D.body.nodes[n]) hb hlt hcal
        have hBt := nodeStep_batch Flock.Program.Wires.bool (fun _ => rfl) L (Flock.Program.evalDef? L f') c z.flatten D
          (outs.map List.flatten) (D.body.nodes[n]) hb hlt hcalL
        have hC1 : ∀ i < (P.defs[(D.body.nodes[n]).fn]!).params.size,
            ∀ u ∈ ((P.defs[(D.body.nodes[n]).fn]!).params.getD i default).2.widths.toList, u = 1 :=
          fun i hi => widths_one_fix (paramBit hpar hi).symm
        have hn! : D.body.nodes[n]! = D.body.nodes[n] := getElem!_pos D.body.nodes n hnD
        have hY : ∀ m < (D.body.nodes[n]).n, _ := fun m hm' => hbatB m (by
          have := batchIns_one hbt hz ho hsz (n := n) (C := P.defs[(D.body.nodes[n]).fn]!)
            (by rw [hn!]; exact hargs n hnD) (by rw [hn!]; exact hbat n hnD hb) hC1 (by rw [hn!]; exact hm')
          rw [hn!] at this
          exact this)
        refine ⟨(List.range (D.body.nodes[n]).n).map fun m => Prog.evalBool P (D.body.nodes[n]).fn
            ((List.range (D.body.nodes[n]).args.size).flatMap fun i =>
              batchArg (Flock.Program.refVal z (Flock.Program.paramStarts D.params) outs)
                (P.defs[(D.body.nodes[n]).fn]!) (D.body.nodes[n]) m i).flatten, ?_, ?_, ?_⟩
        · rw [hW]
          simp only [hWp]
          rw [mapM_some_map]
          exact congrArg (fun z => some (outs.push z)) (flatten_map_singleton _ _)
        · rw [hBt]
          simp only [hB]
          rw [mapM_some_map, Array.map_push]
          exact congrArg (fun z => some ((outs.map List.flatten).push z))
            (congrArg List.flatten (List.map_congr_left fun m hm => by rw [hY m (List.mem_range.1 hm)]))
        · apply outsOk_push ho
          rw [hsz, show D.body.nodes.getD n default = D.body.nodes[n] from by simp [Array.getD, hnD]]
          rw [hout n hnD hpc ((List.range (D.body.nodes[n]).n).map fun m =>
              ((List.range (D.body.nodes[n]).args.size).flatMap fun i =>
                batchArg (Flock.Program.refVal z (Flock.Program.paramStarts D.params) outs)
                  (P.defs[(D.body.nodes[n]).fn]!) (D.body.nodes[n]) m i).flatten)
            (by rw [List.length_map, List.length_range, members_batch hb])]
          simp only [List.map_map]
          try rfl
      · have hns : (D.body.nodes[n]).form ≠ .scan := fun hs => by
          rw [(hscan n hnD hs).1] at hpc; cases hpc
        have hform2' := hform2 hb hns
        have hout1 : ∀ y', (D.body.nodes[n]).out.widths.toList = [(Prog.evalBool P (D.body.nodes[n]).fn y').length] :=
          fun y' => by rw [hout n hnD hpc [y'] (by rw [members_one hform2']; rfl)]; rfl
        refine ⟨[Prog.evalBool P (D.body.nodes[n]).fn
          ((D.body.nodes[n]).args.toList.flatMap fun a =>
            Flock.Program.argVals (Flock.Program.refVal z (Flock.Program.paramStarts D.params) outs) a a.seq.count
              id).flatten], ?_, ?_, ?_⟩
        · rcases hform2' with hf | hf <;>
            simp [nodeStep, wordWires, hf, hcal, show ¬ c ≤ (D.body.nodes[n]).fn by omega, hWp]
        · rcases hform2' with hf | hf <;>
            simp [nodeStep, Flock.Program.Wires.bool, hf, hcalL, show ¬ c ≤ (D.body.nodes[n]).fn by omega, hcallB, hB,
              Array.map_push]
        · apply outsOk_push ho
          rw [hsz]
          simp only [List.map_cons, List.map_nil, Array.getD, hnD, ↓reduceDIte]
          exact (hout1 _).symm
    · have hpc0 : (P.defs[(D.body.nodes[n]).fn]!).prim = false := by simpa using hpc
      have hnc := hnest _ hmem hpc0
      have hbtc : BitTyped (P.defs[(D.body.nodes[n]).fn]!) := by
        cases hnc with
        | mk _ _ _ _ h5' => exact h5'.resolve_left (hnwc _ hmem)
      have IH := ih _ hlt hfnP hnc (hnwc _ hmem) (fun z' hz' => hgd z' (reach_trans (hry' _ hmem) hz'))
      have hWc : ∀ ws, wordDef? P c (D.body.nodes[n]).fn ws =
          wordDef? P ((D.body.nodes[n]).fn + 1) (D.body.nodes[n]).fn ws := fun ws => wordDef_fuel P _ c ws hlt
      by_cases hs : (D.body.nodes[n]).form = .scan
      · obtain ⟨-, hcs, h0, hw0, hw, yw, hretc, hout'⟩ := hscan n hnD hs
        obtain ⟨hsum, hinitB, hrest⟩ := scanBitArgs hbt hz ho hnD hsz (leaf_ones hbtc) (hargs n hnD) hcs h0 hw0 hw
        obtain ⟨w, g1, g2, g3⟩ := scanStep hz ho hnD hsz hs hs rfl hlt hcal hlt hcalL (hargs n hnD) hcs h0 hw0 hw hretc
          hsum hinitB hrest (wordDef? P c) (Flock.Program.evalDef? L f') (fun ws hws => by
            obtain ⟨vs, hv1, hv2, hv3⟩ := IH ws hws
            exact ⟨vs, by rw [hWc]; exact hv1, hv2, hv3 f' (by omega)⟩)
        refine ⟨w, g1, g2, ?_⟩
        apply outsOk_push ho
        rw [hsz, show D.body.nodes.getD n default = D.body.nodes[n] from by simp [Array.getD, hnD], hout']
        exact g3
      by_cases hb : (D.body.nodes[n]).form = .batch
      · have hcb := hcbat n hnD hpc0 hb
        have hW := nodeStep_batch wordWires (fun _ => rfl) P (wordDef? P c) c z D outs (D.body.nodes[n]) hb hlt hcal
        have hBt := nodeStep_batch Flock.Program.Wires.bool (fun _ => rfl) L (Flock.Program.evalDef? L f') c z.flatten D
          (outs.map List.flatten) (D.body.nodes[n]) hb hlt hcalL
        let Am : ℕ → List (List Bool) := fun m => (List.range (D.body.nodes[n]).args.size).flatMap fun i =>
          batchArg (Flock.Program.refVal z (Flock.Program.paramStarts D.params) outs)
            (P.defs[(D.body.nodes[n]).fn]!) (D.body.nodes[n]) m i
        let vsOf : ℕ → List (List Bool) := fun m =>
          (wordDef? P ((D.body.nodes[n]).fn + 1) (D.body.nodes[n]).fn (Am m)).getD []
        have hvs : ∀ m < (D.body.nodes[n]).n,
            wordDef? P ((D.body.nodes[n]).fn + 1) (D.body.nodes[n]).fn (Am m) = some (vsOf m) ∧
            (vsOf m).map List.length = (P.defs[(D.body.nodes[n]).fn]!).ret.widths.toList ∧
            Flock.Program.evalDef? L f' (D.body.nodes[n]).fn (Am m).flatten = some (vsOf m).flatten ∧
            (∀ w ∈ Am m, w.length = 1) := by
          intro m hm'
          have hl := batchWords_length_ax hz ho hnD hsz (hargs n hnD) (hbat n hnD hb) hcb hm'
          obtain ⟨vs, hv1, hv2, hv3⟩ := IH (Am m) hl
          have hvo : vsOf m = vs := by simp only [vsOf, hv1, Option.getD_some]
          rw [hvo]
          exact ⟨hv1, hv2, hv3 f' (by omega), ones_of_map hl (leaf_ones hbtc)⟩
        have hmsP := mapM_some (fun m => wordDef? P c (D.body.nodes[n]).fn ((List.range (D.body.nodes[n]).args.size).flatMap
            fun i => batchArg (Flock.Program.refVal z (Flock.Program.paramStarts D.params) outs)
              (P.defs[(D.body.nodes[n]).fn]!) (D.body.nodes[n]) m i)) vsOf (List.range (D.body.nodes[n]).n)
          (fun m hm => by rw [hWc]; exact (hvs m (List.mem_range.1 hm)).1)
        have hmsL := mapM_some (fun m => Flock.Program.evalDef? L f' (D.body.nodes[n]).fn
            ((List.range (D.body.nodes[n]).args.size).flatMap fun i =>
              batchArg (Flock.Program.refVal z.flatten (Flock.Program.paramStarts D.params) (outs.map List.flatten))
                (P.defs[(D.body.nodes[n]).fn]!) (D.body.nodes[n]) m i))
          (fun m => (vsOf m).flatten) (List.range (D.body.nodes[n]).n)
          (fun m hm => by
            try dsimp only
            rw [hbatB m (hvs m (List.mem_range.1 hm)).2.2.2]
            exact (hvs m (List.mem_range.1 hm)).2.2.1)
        refine ⟨((List.range (D.body.nodes[n]).n).map vsOf).flatten, ?_, ?_, ?_⟩
        · rw [hW, hmsP]
          rfl
        · rw [hBt, hmsL]
          simp only [Option.bind_eq_bind, Option.bind_some, Option.pure_def, Array.map_push, flatten_map_flatten,
            List.map_map]
          try rfl
        · apply outsOk_push ho
          rw [hsz, show D.body.nodes.getD n default = D.body.nodes[n] from by simp [Array.getD, hnD],
            hcout n hnD hpc0 (by rw [hb]; intro h; cases h), members_batch hb, List.map_flatten, List.map_map]
          have hc : (List.range (D.body.nodes[n]).n).map (List.map List.length ∘ vsOf) =
              (List.range (D.body.nodes[n]).n).map fun _ => (P.defs[(D.body.nodes[n]).fn]!).ret.widths.toList :=
            List.map_congr_left fun m hm => (hvs m (List.mem_range.1 hm)).2.1
          rw [hc, List.map_const', List.length_range]
      · have hform2' := hform2 hb hs
        obtain ⟨vs, hv1, hv2, hv3⟩ := IH _ ((callArgs_length hz ho _ hargsn).trans (hcarg n hnD hpc0 hform2'))
        refine ⟨vs, ?_, ?_, ?_⟩
        · rcases hform2' with hf | hf <;>
            simp [nodeStep, wordWires, hf, hcal, show ¬ c ≤ (D.body.nodes[n]).fn by omega, hWc, hv1]
        · rcases hform2' with hf | hf <;>
            simp [nodeStep, Flock.Program.Wires.bool, hf, hcalL, show ¬ c ≤ (D.body.nodes[n]).fn by omega, hcallB,
              hv3 f' (by omega), Array.map_push]
        · apply outsOk_push ho
          rw [hsz, show D.body.nodes.getD n default = D.body.nodes[n] from by simp [Array.getD, hnD],
            hcout n hnD hpc0 hs, members_one hform2']
          simpa using hv2
  have hsz' : D.body.nodes.toList.length = D.body.nodes.toList.length := rfl
  obtain ⟨o, g1, -, g3, g4⟩ := fold_calls D D.body.nodes.toList D.body.nodes.toList hsz' _ _ (hstep c (Nat.le_refl _))
    D.body.nodes.toList.length 0 #[] (by omega) rfl (fun i hi => by simp at hi)
  simp only [List.drop_zero] at g1
  have hos : o.size = D.body.nodes.size := by simpa using g4
  have hretr : ∀ r ∈ D.body.ret.toArray.toList, RefIn D o.size r := fun r hr' => by rw [hos]; exact hret.2 r hr'
  refine ⟨(List.range D.body.ret.count).map fun t =>
      Flock.Program.refVal z (Flock.Program.paramStarts D.params) o (D.body.ret.toArray.getD t default), ?_, ?_, ?_⟩
  · unfold wordDef?
    simp only [hPy, Option.bind_eq_bind, Option.bind_some, hp1, Bool.false_eq_true, ↓reduceIte]
    rw [walkStep_comp wordWires (fun _ => rfl) P _ c z hPy hp1, g1]
    rfl
  · rw [hret.1, range_getD_map]
    exact (refWords_length hz g3 _ hretr).trans hretw
  · intro f hf
    obtain ⟨f', rfl⟩ : ∃ f', f = f' + 1 := ⟨f - 1, by omega⟩
    obtain ⟨o', g1', g2', -, -⟩ := fold_calls D D.body.nodes.toList D.body.nodes.toList hsz' _ _ (hstep f' (by omega))
      D.body.nodes.toList.length 0 #[] (by omega) rfl (fun i hi => by simp at hi)
    simp only [List.drop_zero, Array.map_empty] at g1' g2'
    obtain rfl : o = o' := Option.some.inj (g1.symm.trans g1')
    unfold Flock.Program.evalDef?
    rw [walkStep_comp Flock.Program.Wires.bool (fun _ => rfl) L _ c z.flatten hLy hp1, g2']
    simp only [Option.bind_eq_bind, Option.bind_some, Option.pure_def]
    have h1 : ∀ w ∈ (List.range D.body.ret.count).map (fun t =>
        Flock.Program.refVal z (Flock.Program.paramStarts D.params) o (D.body.ret.toArray.getD t default)),
        w.length = 1 := by
      intro w hw
      obtain ⟨t, ht, rfl⟩ := List.mem_map.1 hw
      have ht' : t < D.body.ret.toArray.size := by rw [← hret.1]; exact List.mem_range.1 ht
      have hm : D.body.ret.toArray.getD t default ∈ D.body.ret.toArray.toList := by
        rw [show D.body.ret.toArray.getD t default = D.body.ret.toArray[t] from by simp [Array.getD, ht']]
        exact Array.mem_toList_iff.2 (Array.getElem_mem ht')
      exact refVal_len1 hbt hz g3 (hretr _ hm)
    rw [flatten_ones h1, List.map_map]
    exact congrArg some (List.map_congr_left fun t _ => refVal_ones hz1 (outs_one hbt g3 (by omega)) _)

end Flock.Chain
