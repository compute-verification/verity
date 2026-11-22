module

public import Proofs.Properties.Flock.Circuit.LowerEval.Places
public import Proofs.Properties.Flock.Circuit.LowerEval.GateWords
public import Proofs.Core.QSort

/-! `UnitWordComputes` for templates of word-primitive calls and axis-0 batches. -/

@[expose] public section

namespace Flock.Chain

open Flock Flock.Lower Flock.Tmpl Flock.TmplValue Flock.Chain.LowerDef
open Flock.Program hiding Form

section UnitWordProof
open InferenceOnlyVerifier.Properties.SecurityProofs

theorem ofParts_getD (ps : Array Part) {t : ℕ} (ht : t < (Seq.ofParts ps).count) :
    (Seq.ofParts ps).toArray.getD t default = (Seq.ofParts ps).get t := by
  obtain ⟨i, hi, h1, h2, hg⟩ := Walk.seq_get ps ht
  rw [hg]
  have hok : seqOk (Seq.ofParts ps) = true := by simp [seqOk, Seq.ofParts]
  exact toArray_getD_part hok hi h1 h2

theorem refIn_of_decoded {D : Defn} {k : ℕ} {r : Ref} (hpar : D.body.params = D.params)
    (h : BodiesOk.RefIn D.body r) (hk : r.p = false → r.i < k) : RefIn D k r := by
  obtain ⟨h1, h2⟩ := h
  refine ⟨fun hp => hpar ▸ (h1 hp).1, fun hp => ⟨hk hp, (h2 hp).1⟩, ?_⟩
  unfold tyAt
  cases hp : r.p
  · obtain ⟨hi, hj⟩ := h2 hp
    have e : D.body.nodes.getD r.i default = D.body.nodes[r.i]! := by
      rw [getElem!_pos D.body.nodes r.i hi]; simp [Array.getD, hi]
    simp only [Bool.false_eq_true, ite_false]
    rw [e, hb_widths_size]; exact hj
  · obtain ⟨hi, hj⟩ := h1 hp
    rw [hpar] at hi hj
    have e : D.params.getD r.i default = D.params[r.i]! := by
      rw [getElem!_pos D.params r.i hi]; simp [Array.getD, hi]
    simp only [ite_true]
    rw [e, hb_widths_size]; exact hj

theorem refs_of_decoded {D : Defn} {k : ℕ} {ps : Array Part} (hpar : D.body.params = D.params)
    (h : ∀ t < (Seq.ofParts ps).count, BodiesOk.RefIn D.body ((Seq.ofParts ps).get t))
    (hk : ∀ t < (Seq.ofParts ps).count, ((Seq.ofParts ps).get t).p = false → ((Seq.ofParts ps).get t).i < k) :
    (Seq.ofParts ps).count = (Seq.ofParts ps).toArray.size ∧
      ∀ r ∈ (Seq.ofParts ps).toArray.toList, RefIn D k r := by
  refine ⟨ofParts_count ps, fun r hr => ?_⟩
  obtain ⟨t, ht, rfl⟩ := List.getElem_of_mem hr
  rw [Array.length_toList, ← ofParts_count] at ht
  have he : (Seq.ofParts ps).toArray.toList[t] = (Seq.ofParts ps).get t := by
    rw [← ofParts_getD ps ht, Array.getElem_toList, Array.getD, dif_pos (by rw [← ofParts_count]; exact ht)]
    rfl
  rw [he]
  exact refIn_of_decoded hpar (h t ht) (hk t ht)

/-- **A composite's references are typed**, from the decoder: its references' facts (`BodiesOk.BodyFacts`) and its
refusal of a forward reference (`BodiesOk.decode_fwd`). -/
theorem refs_comp {bytes : ByteArray} {P : Prog} (hdec : decode bytes = .ok P)
    (hr : (P.defs[P.root]!).body.recut = #[]) {d : ℕ} (hd : d < P.defs.size) (hp : (P.defs[d]!).prim = false) :
    (∀ k (hk : k < (P.defs[d]!).body.nodes.size), ∀ a ∈ ((P.defs[d]!).body.nodes[k]).args.toList,
        a.seq.count = a.seq.toArray.size ∧ ∀ r ∈ a.seq.toArray.toList, RefIn (P.defs[d]!) k r) ∧
      ((P.defs[d]!).body.ret.count = (P.defs[d]!).body.ret.toArray.size ∧
        ∀ r ∈ (P.defs[d]!).body.ret.toArray.toList, RefIn (P.defs[d]!) (P.defs[d]!).body.nodes.size r) := by
  obtain ⟨st, hP, hinv, -⟩ := BodiesOk.decode_spec bytes P hdec hr
  have hok : ∀ i < P.defs.size, BodiesOk.DefOk P.defs i P.defs[i]! := by rw [hP]; exact hinv.1
  have hfw := BodiesOk.decode_fwd bytes P hdec hr d hd
  obtain ⟨hcal, hnodes, hret, psr, hpsr, -⟩ := hok d hd
  have hpar := (hcal.2 hp).1
  refine ⟨fun k hk a ha => ?_, ?_⟩
  · rw [← getElem!_pos (P.defs[d]!).body.nodes k hk] at ha
    obtain ⟨a', ha', rfl⟩ := List.getElem_of_mem ha
    rw [Array.length_toList] at ha'
    rw [Array.getElem_toList, ← getElem!_pos _ a' ha']
    obtain ⟨-, -, hargs⟩ := hnodes k hk
    obtain ⟨⟨ps, hps, -⟩, hin⟩ := hargs a' ha'
    have hf := hfw k hk a' ha'
    rw [hps] at hin hf ⊢
    exact refs_of_decoded hpar hin hf
  · rw [hpsr] at hret ⊢
    exact refs_of_decoded hpar hret fun t ht hp => ((hret t ht).2 hp).1

/-- **`Batch0Refs`**: `refs_comp`. -/
theorem batch0Refs : Batch0Refs := fun _ _ hdec hr _ hd hB => refs_comp hdec hr hd hB.1

/-- **`Batch0Callees`**, from the decoder: each node calls a Definition decoded before its caller
(`BodiesOk.BodyFacts`). -/
theorem batch0Callees : Batch0Callees := by
  intro bytes P hdec hr d hd _ nd hnd
  obtain ⟨st, hP, hinv, -⟩ := BodiesOk.decode_spec bytes P hdec hr
  have hok : ∀ i < P.defs.size, BodiesOk.DefOk P.defs i P.defs[i]! := by rw [hP]; exact hinv.1
  obtain ⟨k, hk, rfl⟩ := Array.mem_iff_getElem.1 hnd
  rw [← getElem!_pos _ k hk]
  exact ((hok d hd).2.1 k hk).1

/-- A decoded argument's references, read in `D`'s body before node `k`, name leaves of its type's widths, in order. -/
theorem arg_widths {D : Defn} {k : ℕ} {a : Arg} {ps : Array Part} (hpar : D.body.params = D.params)
    (hps : a.seq = Seq.ofParts ps) (hcnt : a.seq.count = a.ty.leaves)
    (hrefs : ∀ r ∈ a.seq.toArray.toList, RefIn D k r)
    (hw : LeafWidths.SeqWide (LeafWidths.srcOf D.body k) a.ty a.seq) :
    a.seq.toArray.toList.map (fun r => (tyAt D r).widths[r.j]!) = a.ty.widths.toList := by
  have hsz : a.seq.toArray.size = a.seq.count := by rw [hps]; exact (ofParts_count ps).symm
  apply List.ext_getElem (by simp [hsz, hcnt, hb_widths_size])
  intro m h1 h2
  simp only [List.length_map, Array.length_toList] at h1
  have hm : m < a.seq.count := hsz ▸ h1
  simp only [List.getElem_map, Array.getElem_toList]
  have hget : a.seq.toArray[m] = a.seq.get m := by
    have := ofParts_getD ps (hps ▸ hm)
    rw [← hps] at this
    rw [← this, Array.getD, dif_pos h1]
    rfl
  have hr : RefIn D k (a.seq.toArray[m]) := hrefs _ (Array.getElem_mem_toList h1)
  rw [hget] at hr ⊢
  obtain ⟨hp1, hp2, hj⟩ := hr
  have hS : (LeafWidths.srcOf D.body k).ty? (a.seq.get m).p (a.seq.get m).i = some (tyAt D (a.seq.get m)) := by
    unfold LeafWidths.srcOf Widths.Src.ty? tyAt
    cases hp : (a.seq.get m).p
    · obtain ⟨hik, hin⟩ := hp2 hp
      simp [hik, hin, Array.getD]
    · have hip := hp1 hp
      simp [hpar, hip, Array.getD]
  rw [← LeafWidths.wAt_widths _ _ (by rw [← hb_widths_size]; exact hj), hw m hm _ hS,
    LeafWidths.wAt_widths _ _ (by rw [← hcnt]; exact hm)]
  exact getElem!_pos _ m (by simpa using h2)

/-- **A composite's batch axes are typed**, from the decoder: `buildNode`'s axis entries (`BodiesOk.AxesOk`), and each
argument's references naming leaves of its type's widths (`LeafWidths.decoded_wide`). -/
theorem axes_comp {bytes : ByteArray} {P : Prog} (hdec : decode bytes = .ok P)
    (hr : (P.defs[P.root]!).body.recut = #[]) {d : ℕ} (hd : d < P.defs.size) (hp : (P.defs[d]!).prim = false) :
    ∀ k (hk : k < (P.defs[d]!).body.nodes.size), ((P.defs[d]!).body.nodes[k]).form = .batch →
      ∀ i < ((P.defs[d]!).body.nodes[k]).args.size,
        (((P.defs[d]!).body.nodes[k]).axes[i]? = some (some 0) → ∃ E : Ty,
          (((P.defs[d]!).body.nodes[k]).args.getD i default).ty = .array ((P.defs[d]!).body.nodes[k]).n E ∧
          (((P.defs[d]!).body.nodes[k]).args.getD i default).dims = some (((P.defs[d]!).body.nodes[k]).n, E.leaves) ∧
          i < (P.defs[((P.defs[d]!).body.nodes[k]).fn]!).params.size ∧
          ((P.defs[((P.defs[d]!).body.nodes[k]).fn]!).params.getD i default).2 = E ∧
          (((P.defs[d]!).body.nodes[k]).args.getD i default).seq.toArray.toList.map
            (fun r => (tyAt (P.defs[d]!) r).widths[r.j]!) =
            (List.replicate ((P.defs[d]!).body.nodes[k]).n E.widths.toList).flatten) ∧
        ((∀ ax, ((P.defs[d]!).body.nodes[k]).axes[i]? ≠ some (some ax)) →
          (((P.defs[d]!).body.nodes[k]).args.getD i default).dims = none) := by
  intro k hk hbat i hi
  obtain ⟨st, hP, hinv, -⟩ := BodiesOk.decode_spec bytes P hdec hr
  have hok : ∀ i < P.defs.size, BodiesOk.DefOk P.defs i P.defs[i]! := by rw [hP]; exact hinv.1
  obtain ⟨hcal, hnodes, -⟩ := hok d hd
  have hpar := (hcal.2 hp).1
  have hR := (refs_comp hdec hr hd hp).1 k hk
  have hW := ((LeafWidths.decoded_wide bytes P hdec hr d hd hp).2).1 k hk
  obtain ⟨-, hno, hargs⟩ := hnodes k hk
  have hk' : (P.defs[d]!).body.nodes[k] = (P.defs[d]!).body.nodes[k]! := (getElem!_pos _ k hk).symm
  rw [hk'] at hbat hi hR ⊢
  generalize (P.defs[d]!).body.nodes[k]! = x at hno hargs hbat hi hR hW ⊢
  obtain ⟨hsz, hcnt, -, -, -, -, hax⟩ := hno
  obtain ⟨haxs, hall⟩ := hax hbat
  obtain ⟨hnone, hsome⟩ := hall i hi
  obtain ⟨⟨ps, hps, -⟩, -⟩ := hargs i hi
  simp only [← Array.getElem!_eq_getD]
  refine ⟨fun h0 => ?_, fun hn => ?_⟩
  · obtain ⟨hrm, hdm⟩ := hsome 0 h0
    have hty : (x.args[i]!).ty = .array x.n ((P.defs[x.fn]!).params[i]!).2 := by
      revert hrm
      rcases (x.args[i]!).ty with w | ⟨n', e⟩ | fs | fs
      all_goals simp [Ty.removeAxis]
    refine ⟨_, hty, by rw [hdm, hty]; rfl, hsz ▸ hi, rfl, ?_⟩
    rw [← widths_array_flat, ← hty]
    exact arg_widths hpar hps (hcnt i hi)
      (hR _ (by rw [getElem!_pos x.args i hi]; exact Array.getElem_mem_toList hi)).2 (hW i hi)
  · cases hx : x.axes[i]? with
    | none => rw [Array.getElem?_eq_none_iff] at hx; omega
    | some o =>
      cases o with
      | none => exact (hnone hx).1
      | some ax => exact absurd hx (hn ax)

/-- **A composite's batch axes are typed, on any axis**: an argument mapped on axis `ax` has its callee's parameter type
once `ax` is removed, its dimensions are its type's on `ax`, and its references name its type's widths, in order. -/
theorem axes_comp_ax {bytes : ByteArray} {P : Prog} (hdec : decode bytes = .ok P)
    (hr : (P.defs[P.root]!).body.recut = #[]) {d : ℕ} (hd : d < P.defs.size) (hp : (P.defs[d]!).prim = false) :
    ∀ k (hk : k < (P.defs[d]!).body.nodes.size), ((P.defs[d]!).body.nodes[k]).form = .batch →
      ∀ i < ((P.defs[d]!).body.nodes[k]).args.size, ∀ ax,
        ((P.defs[d]!).body.nodes[k]).axes[i]? = some (some ax) →
          (((P.defs[d]!).body.nodes[k]).args.getD i default).ty.removeAxis ax =
            some (((P.defs[d]!).body.nodes[k]).n, ((P.defs[((P.defs[d]!).body.nodes[k]).fn]!).params.getD i default).2) ∧
          (((P.defs[d]!).body.nodes[k]).args.getD i default).dims =
            (((P.defs[d]!).body.nodes[k]).args.getD i default).ty.axisDims ax ∧
          i < (P.defs[((P.defs[d]!).body.nodes[k]).fn]!).params.size ∧
          (((P.defs[d]!).body.nodes[k]).args.getD i default).seq.toArray.toList.map
            (fun r => (tyAt (P.defs[d]!) r).widths[r.j]!) =
            (((P.defs[d]!).body.nodes[k]).args.getD i default).ty.widths.toList := by
  intro k hk hbat i hi ax hax
  obtain ⟨st, hP, hinv, -⟩ := BodiesOk.decode_spec bytes P hdec hr
  have hok : ∀ i < P.defs.size, BodiesOk.DefOk P.defs i P.defs[i]! := by rw [hP]; exact hinv.1
  obtain ⟨hcal, hnodes, -⟩ := hok d hd
  have hpar := (hcal.2 hp).1
  have hR := (refs_comp hdec hr hd hp).1 k hk
  have hW := ((LeafWidths.decoded_wide bytes P hdec hr d hd hp).2).1 k hk
  obtain ⟨-, hno, hargs⟩ := hnodes k hk
  have hk' : (P.defs[d]!).body.nodes[k] = (P.defs[d]!).body.nodes[k]! := (getElem!_pos _ k hk).symm
  rw [hk'] at hbat hi hR hax ⊢
  generalize (P.defs[d]!).body.nodes[k]! = x at hno hargs hbat hi hR hW hax ⊢
  obtain ⟨hsz, hcnt, -, -, -, -, haxs⟩ := hno
  obtain ⟨-, hall⟩ := haxs hbat
  obtain ⟨-, hsome⟩ := hall i hi
  obtain ⟨⟨ps, hps, -⟩, -⟩ := hargs i hi
  simp only [← Array.getElem!_eq_getD]
  obtain ⟨hrm, hdm⟩ := hsome ax hax
  exact ⟨hrm, hdm, hsz ▸ hi, arg_widths hpar hps (hcnt i hi)
    (hR _ (by rw [getElem!_pos x.args i hi]; exact Array.getElem_mem_toList hi)).2 (hW i hi)⟩

/-- **A batch member's leaves**, on any axis: with `(n, L)` the argument's dimensions, member `m`'s leaf `l` of its
parameter is the argument's reference `(l / L * n + m) * L + l % L`, of the parameter's width at `l`. -/
theorem axes_comp_member {bytes : ByteArray} {P : Prog} (hdec : decode bytes = .ok P)
    (hr : (P.defs[P.root]!).body.recut = #[]) {d : ℕ} (hd : d < P.defs.size) (hp : (P.defs[d]!).prim = false)
    (k : ℕ) (hk : k < (P.defs[d]!).body.nodes.size) (hbat : ((P.defs[d]!).body.nodes[k]).form = .batch)
    (i : ℕ) (hi : i < ((P.defs[d]!).body.nodes[k]).args.size) (ax : ℕ)
    (hax : ((P.defs[d]!).body.nodes[k]).axes[i]? = some (some ax)) :
    ∃ L, (((P.defs[d]!).body.nodes[k]).args.getD i default).dims = some (((P.defs[d]!).body.nodes[k]).n, L) ∧
      L ∣ ((P.defs[((P.defs[d]!).body.nodes[k]).fn]!).params.getD i default).2.leaves ∧
      ∀ m < ((P.defs[d]!).body.nodes[k]).n,
        ∀ l < ((P.defs[((P.defs[d]!).body.nodes[k]).fn]!).params.getD i default).2.leaves,
          ((((P.defs[d]!).body.nodes[k]).args.getD i default).seq.toArray.toList.map
            (fun r => (tyAt (P.defs[d]!) r).widths[r.j]!))[(l / L * ((P.defs[d]!).body.nodes[k]).n + m) * L + l % L]? =
            ((P.defs[((P.defs[d]!).body.nodes[k]).fn]!).params.getD i default).2.widths.toList[l]? := by
  obtain ⟨hrm, hdm, -, hw⟩ := axes_comp_ax hdec hr hd hp k hk hbat i hi ax hax
  obtain ⟨L, hL, hdvd, -⟩ := axisDims_of_removeAxis ax hrm
  exact ⟨L, hdm.trans hL, hdvd, fun m hm l hl => by rw [hw]; exact removeAxis_widths ax hrm hL m hm l hl⟩

/-- **`Batch0Axes`**: `axes_comp`. -/
theorem batch0Axes : Batch0Axes := fun _ _ hdec hr _ hd hB => axes_comp hdec hr hd hB.1

/-- **`LowerGateBodies` for a template of word-primitive calls and axis-0 batches**, on `LiftCalls`: the lowering of
its lift evaluates through gates alone, and the lift puts each node's Boolean Definition in its place. -/
theorem gateBodies_batch0 (hLC : LiftCalls) {bytes : ByteArray} {P S L : Prog} {units : Array ℕ} {d e : ℕ}
    {partition : Lean.Json} {cuts : List (String × List (String × ℕ))} {a : Spot.OneStageZk.Admit} {t : Tmpl.Tmpl}
    (h : decode bytes = .ok P) (hr : (P.defs[P.root]!).body.recut = #[]) (hs : Spot.Restrict.restrict P units = .ok S)
    (ha : Spot.OneStageZk.admitted partition cuts S = .ok a) (hda : a.d = d)
    (hd : d < P.defs.size) (hco : Batch0 P d) (hne : 0 < (P.defs[d]!).body.nodes.size)
    (hl : Flock.Lift.lift S d = .ok (L, e)) (hbt : Spot.OneStageZk.blockTmpl S a = .ok t) : GateBodies P d := by
  have _ := ha
  obtain ⟨Q, e', hq, hG⟩ := lower_gates hbt
  rw [hda, hl] at hq
  simp only [Except.ok.injEq, Prod.mk.injEq] at hq
  obtain ⟨rfl, rfl⟩ := hq
  obtain ⟨hle, hpre, hbool, hidx, -⟩ := restrictShape P units S hs
  obtain ⟨⟨hmap, hkeys⟩, -⟩ := FlockSoundness.DecodeBoolean.decode_facts bytes P h hr
  have hm : P.BoolMapOk := ⟨hmap, hkeys⟩
  have hbodyS := bodyOf_eq hm hbool hidx
  have hord := batch0Callees bytes P h hr d hd hco
  have hSd : S.defs[d]! = P.defs[d]! := hpre d hd
  have hcoS : PrimNodes S d := by
    refine ⟨by rw [hSd]; exact hco.1, fun nd hnd => ?_⟩
    rw [hSd] at hnd
    have hlt : nd.fn < P.defs.size := by have := hord nd hnd; omega
    rw [hpre _ hlt]
    exact (hco.2 nd hnd).2
  have hcont : ∀ (k v : String), S.boolean[k]? = some v → S.index.contains v = true := by
    intro k v hv
    rw [hbool] at hv
    obtain ⟨w, hw, -, hkey, hword⟩ := hkeys k v hv
    obtain ⟨b, -, hbo, -⟩ := hmap w hw (by rw [hkey]; exact hword)
    rw [hkey] at hbo
    unfold Prog.bodyOf? at hbo
    rw [hv] at hbo
    have hb' : P.index[v]? = some b := hbo
    rw [Std.HashMap.contains_eq_isSome_getElem?, hidx v b hb']
    rfl
  obtain ⟨heS, hLpre, -, fnOf, primOf, gatesOf, hLe, hfn⟩ :=
    hLC S L d e hcont hl (by omega) hcoS (by rw [hSd]; exact hne)
  rw [hSd] at hLe hfn
  have hLP' : ∀ i < P.defs.size, L.defs[i]! = P.defs[i]! := fun i hi => by
    rw [getElem!_def, getElem!_def, hLpre i (by omega), get?_of_pre hle hpre hi]
  have hLe' : L.defs[e]! = Flock.Lift.liftDef (P.defs[d]!) fnOf primOf gatesOf := by simp [getElem!_def, hLe]
  intro nd hnd b hb
  obtain ⟨n, hn, rfl⟩ := Array.mem_iff_getElem.1 hnd
  have hnD' : n < (Flock.Lift.liftDef (P.defs[d]!) fnOf primOf gatesOf).body.nodes.size := by
    rw [liftDef_nodes_size]; exact hn
  obtain ⟨hfn', -, -⟩ := liftDef_node (P.defs[d]!) fnOf primOf gatesOf hn hnD'
  have hfnP : ((P.defs[d]!).body.nodes[n]).fn < P.defs.size := by have := hord _ hnd; omega
  have hbS := hfn _ hnd
  rw [hpre _ hfnP, hbodyS] at hbS
  obtain rfl : b = fnOf ((P.defs[d]!).body.nodes[n]).fn := Option.some.inj (hb.symm.trans hbS)
  cases hG with
  | prim hp _ =>
    rw [hLe', liftDef_prim, hco.1] at hp
    cases hp
  | comp _ _ hnd' =>
    have hmem : (Flock.Lift.liftDef (P.defs[d]!) fnOf primOf gatesOf).body.nodes[n] ∈ (L.defs[e]!).body.nodes := by
      rw [hLe']; exact Array.getElem_mem hnD'
    have hgL := hnd' _ hmem
    rw [hfn'] at hgL
    exact gateDef_prefix hLP' hgL (body_lt hm hbS)

/-- **`LowerGateBodies`** holds: the lowering of the lift evaluates through gates alone (`lower_gates`); the lift places
each composite reached from the template that reaches a word primitive lifted, its nodes calling where the lift places
their callees (`liftPlacesAll`), so each such lifted composite evaluates through gates alone, and a node of it that calls
a word primitive (`Prog.BoolMapOk`, `DecodeKeys.decode_keys`) calls that primitive's Boolean Definition. -/
theorem lowerGateBodies : LowerGateBodies := by
  intro bytes P partition cuts units S a L e t h hr hs ha hd hl hbt y hy hpy nd hnd b hb
  have _ := ha
  obtain ⟨Q, e', hq, hG⟩ := lower_gates hbt
  rw [hl] at hq
  simp only [Except.ok.injEq, Prod.mk.injEq] at hq
  obtain ⟨rfl, rfl⟩ := hq
  obtain ⟨hle, hpre, hbool, hidx, -⟩ := restrictShape P units S hs
  obtain ⟨⟨hmap, hkeys⟩, -⟩ := FlockSoundness.DecodeBoolean.decode_facts bytes P h hr
  have hm : P.BoolMapOk := ⟨hmap, hkeys⟩
  have hbodyS := bodyOf_eq hm hbool hidx
  obtain ⟨st, hP, hinv, -⟩ := BodiesOk.decode_spec bytes P h hr
  have hok : ∀ i < P.defs.size, BodiesOk.DefOk P.defs i P.defs[i]! := by rw [hP]; exact hinv.1
  have hcont : ∀ (k v : String), S.boolean[k]? = some v → S.index.contains v = true := by
    intro k v hv
    rw [hbool] at hv
    obtain ⟨w, hw, -, hkey, hword⟩ := hkeys k v hv
    obtain ⟨b, -, hbo, -⟩ := hmap w hw (by rw [hkey]; exact hword)
    rw [hkey] at hbo
    unfold Prog.bodyOf? at hbo
    rw [hv] at hbo
    have hb' : P.index[v]? = some b := hbo
    rw [Std.HashMap.contains_eq_isSome_getElem?, hidx v b hb']
    rfl
  obtain ⟨pl, he, hA, -, -, hC⟩ := liftPlacesAll hl
  have hG' : ∀ y, Reach S a.d y → (S.defs[y]!).prim = false → ReachesWord S y → GateDef L (pl y) := by
    intro y hy
    induction hy with
    | root => intro _ _; rw [← he]; exact hG
    | @callee y nd hy' hp hn ih =>
      intro _ hrw
      have hrwy : ReachesWord S y := .comp hp hn hrw
      have hgy := ih hp hrwy
      obtain ⟨-, fnOf, primOf, gatesOf, hLy, hfy, -⟩ := hC y hy' hp hrwy
      obtain ⟨n, hn', rfl⟩ := Array.mem_iff_getElem.1 hn
      have hLy' : L.defs[pl y]! = Flock.Lift.liftDef (S.defs[y]!) fnOf primOf gatesOf := by simp [getElem!_def, hLy]
      have hnD' : n < (Flock.Lift.liftDef (S.defs[y]!) fnOf primOf gatesOf).body.nodes.size := by
        rw [liftDef_nodes_size]; exact hn'
      obtain ⟨hfn', -, -⟩ := liftDef_node (S.defs[y]!) fnOf primOf gatesOf hn' hnD'
      cases hgy with
      | prim hpp _ =>
        rw [hLy', liftDef_prim, hp] at hpp
        cases hpp
      | comp _ _ hnd' =>
        have hmem : (Flock.Lift.liftDef (S.defs[y]!) fnOf primOf gatesOf).body.nodes[n] ∈ (L.defs[pl y]!).body.nodes := by
          rw [hLy']; exact Array.getElem_mem hnD'
        have hg := hnd' _ hmem
        rw [hfn', hfy _ (Array.getElem_mem hn')] at hg
        exact hg
  obtain ⟨hyl, hyS⟩ := reach_restrict h hr hpre hd hy
  obtain ⟨n, hn, rfl⟩ := Array.mem_iff_getElem.1 hnd
  have hfnd : ((P.defs[y]!).body.nodes[n]).fn < y := by
    rw [← getElem!_pos _ n hn]; exact ((hok y hyl).2.1 n hn).1
  have hfnP : ((P.defs[y]!).body.nodes[n]).fn < P.defs.size := by omega
  have hword : isWordPrim (P.defs[((P.defs[y]!).body.nodes[n]).fn]!).prim
      (P.defs[((P.defs[y]!).body.nodes[n]).fn]!).key = true := by
    unfold Prog.bodyOf? at hb
    cases hv : P.boolean[(P.defs[((P.defs[y]!).body.nodes[n]).fn]!).key]? with
    | none => rw [hv] at hb; cases hb
    | some v =>
      obtain ⟨w, -, hiw, -, hwd⟩ := hkeys _ v hv
      have hif := (DecodeKeys.decode_keys bytes P h).2 _ hfnP
      rw [hif] at hiw
      obtain rfl := Option.some.inj hiw
      exact hwd
  have hSy : S.defs[y]! = P.defs[y]! := hpre y hyl
  have hSf := hpre _ hfnP
  have hpS : (S.defs[y]!).prim = false := by rw [hSy]; exact hpy
  have hndS : (P.defs[y]!).body.nodes[n] ∈ (S.defs[y]!).body.nodes := by rw [hSy]; exact hnd
  have hwS : isWordPrim (S.defs[((P.defs[y]!).body.nodes[n]).fn]!).prim
      (S.defs[((P.defs[y]!).body.nodes[n]).fn]!).key = true := by rw [hSf]; exact hword
  have hrw : ReachesWord S y := .comp hpS hndS (.word hwS)
  have hgy := hG' y hyS hpS hrw
  obtain ⟨-, fnOf, primOf, gatesOf, hLy, hfy, -⟩ := hC y hyS hpS hrw
  have hbS := hA _ (.callee hyS hpS hndS) hwS
  rw [hSf, hbodyS] at hbS
  obtain rfl : b = pl ((P.defs[y]!).body.nodes[n]).fn := Option.some.inj (hb.symm.trans hbS)
  obtain ⟨-, hLpre, -⟩ := liftKeeps hcont hl
  have hLP' : ∀ i < P.defs.size, L.defs[i]! = P.defs[i]! := fun i hi => by
    rw [getElem!_def, getElem!_def, hLpre i (by omega), get?_of_pre hle hpre hi]
  rw [hSy] at hLy hfy
  have hLy' : L.defs[pl y]! = Flock.Lift.liftDef (P.defs[y]!) fnOf primOf gatesOf := by simp [getElem!_def, hLy]
  have hnD' : n < (Flock.Lift.liftDef (P.defs[y]!) fnOf primOf gatesOf).body.nodes.size := by
    rw [liftDef_nodes_size]; exact hn
  obtain ⟨hfn', -, -⟩ := liftDef_node (P.defs[y]!) fnOf primOf gatesOf hn hnD'
  cases hgy with
  | prim hp _ =>
    rw [hLy', liftDef_prim, hpy] at hp
    cases hp
  | comp _ _ hnd' =>
    have hmem : (Flock.Lift.liftDef (P.defs[y]!) fnOf primOf gatesOf).body.nodes[n] ∈ (L.defs[pl y]!).body.nodes := by
      rw [hLy']; exact Array.getElem_mem hnD'
    have hgL := hnd' _ hmem
    rw [hfn', hfy _ hnd] at hgL
    exact gateDef_prefix hLP' hgL (body_lt hm hbS)

theorem foldlM_inv {α β : Type} (l : List α) (f : β → α → Option β) (I : ℕ → β → Prop)
    (hstep : ∀ n (hn : n < l.length) b, I n b → ∃ b', f b l[n] = some b' ∧ I (n + 1) b') :
    ∀ k n b, k + n = l.length → I n b → ∃ o, (l.drop n).foldlM f b = some o ∧ I l.length o
  | 0, n, b, hk, hI => ⟨b, by simp [List.drop_eq_nil_of_le (by omega : l.length ≤ n)],
      by rw [show l.length = n by omega]; exact hI⟩
  | k + 1, n, b, hk, hI => by
    obtain ⟨b', h1, h2⟩ := hstep n (by omega) b hI
    obtain ⟨o, g1, g2⟩ := foldlM_inv l f I hstep k (n + 1) b' (by omega) h2
    refine ⟨o, ?_, g2⟩
    rw [List.drop_eq_getElem_cons (by omega), List.foldlM_cons, h1]
    exact g1

/-- **The word at a primitive node of a unit**: in member `m` of root node `k`, which calls a composite, the gate of
primitive node `n`, computed on the trace, is its primitive on the words `v` gives its arguments' references, where `v`
gives each reference the word at the gate below it that it resolves to. -/
theorem unit_node_word {P : Prog} (hroot : P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false)
    (hB : ∀ sc, Resolve.Good P sc → Resolve.BodyOk P sc.d) (hL : ∀ d < P.defs.size, Walk.LaidOut P d)
    (hG : ∀ h < InferenceOnlyVerifier.Properties.gates P, InferenceOnlyVerifier.Properties.GateOk P h)
    {k m : ℕ} (hk : k < (P.defs[P.root]!).body.nodes.size) (hm : m < ((P.defs[P.root]!).body.nodes[k]!).members)
    (hc : (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).prim = false)
    {n : ℕ} (hn : n < (P.defs[(P.child P.rootScope k m).d]!).body.nodes.size) {nd : Node}
    (hnd : (P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]! = nd) (hf : nd.form = .prim)
    (hp : (P.defs[nd.fn]!).prim = true) (hg1 : (P.defs[nd.fn]!).gates = 1)
    (hseq : ∀ a ∈ nd.args.toList, ∃ ps, a.seq = Seq.ofParts ps)
    (Y : (Flock.OfProgram.ofProgram P).Trace) {G : ℕ}
    (hGd : (P.child P.rootScope k m).off + (P.defs[(P.child P.rootScope k m).d]!).body.offsets[n]! = G)
    (hGN : G < (Flock.OfProgram.ofProgram P).N) (hkey : isInputId (P.defs[nd.fn]!).key = false)
    (hY : Flock.OfProgram.inputAt P G = false → Y ⟨G, hGN⟩ = (Flock.OfProgram.ofProgram P).op ⟨G, hGN⟩ Y)
    {w0 : ℕ} (hret : (P.defs[nd.fn]!).ret = .value w0) (hw : ∀ y, (Prog.evalBool P nd.fn y).length = w0)
    (v : Ref → List Bool)
    (hv : ∀ i < nd.args.size, ∀ t < (nd.args[i]!).seq.count, ∀ q,
      resolve P (P.child P.rootScope k m) ((nd.args[i]!).seq.get t) = .ok q → q < G →
        v ((nd.args[i]!).seq.get t) = portWord Y ⟨q, 0⟩) (w : ℕ) :
    portWord Y ⟨G, w⟩ =
      Prog.evalBool P nd.fn ((nd.args.toList.flatMap fun a => Flock.Program.argVals v a a.seq.count id).flatten) := by
  subst hnd
  obtain ⟨p, ops, w', ⟨⟨sc', hg⟩, -, -⟩, hlt⟩ := hG G hGN
  have hg' : P.gate ((P.child P.rootScope k m).off + (P.defs[(P.child P.rootScope k m).d]!).body.offsets[n]! + 0) =
      .ok (p, ops, sc') := by rw [Nat.add_zero, hGd]; exact hg
  obtain ⟨rfl, rfl, hops⟩ := unit_gate_at hroot hB hL hk hm hc hn (by rw [members_one (Or.inl hf)]; omega) hp hg1 hg'
  have hga : Flock.OfProgram.gateAt P G = some (((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!).fn, ops) := by
    unfold Flock.OfProgram.gateAt; rw [hg]
  have hlt' : ∀ a ∈ ops.toList, a < G := fun a ha => hlt a (Array.mem_toList_iff.1 ha)
  have hin : Flock.OfProgram.inputAt P G = false := by unfold Flock.OfProgram.inputAt; rw [hga]; exact hkey
  have hwd : ∀ y, (Prog.evalBool P ((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!).fn y).length =
      Flock.OfProgram.widthAt P G := fun y => by
    unfold Flock.OfProgram.widthAt; rw [hga]; simp only [hret]; exact hw y
  rw [word_at_computed Y hGN hga hin hlt' (hY hin) (hwd _)]
  congr 1
  obtain ⟨L, hlen, hflat, hLi⟩ := OperandWidths.operands_lists P _ _ 0 ops hops
  rw [hflat, List.map_flatten, flatMap_flatten_map]
  congr 2
  apply List.ext_getElem (by simp [hlen])
  intro i h1 h2
  have hi : i < ((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!).args.size := by simpa [hlen] using h1
  simp only [List.getElem_map, Array.getElem_toList]
  obtain ⟨hlc, hres⟩ := hLi i (by simpa using h1)
  have hcnt : L[i].length = (((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!).args[i]!).seq.count := by
    rw [hlc]; unfold OperandWidths.cntOf; simp [hf, BodiesOk.beq_prim_batch]
  have hai : ((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!).args[i] =
      ((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!).args[i]! := (getElem!_pos _ i hi).symm
  obtain ⟨ps, hps⟩ := hseq _ (Array.getElem_mem_toList hi)
  rw [hai] at hps ⊢
  unfold Flock.Program.argVals
  apply List.ext_getElem (by simp [hcnt])
  intro t t1 t2
  simp only [List.length_map, List.length_range] at t1 t2
  have ht : t < (((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!).args[i]!).seq.count := by simpa using t2
  simp only [List.getElem_map, List.getElem_range, id]
  have hr := hres t t1
  have hro : OperandWidths.refOf ((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!) 0 i t =
      (((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!).args[i]!).seq.get t := by
    unfold OperandWidths.refOf; simp [hf, BodiesOk.beq_prim_batch]
  rw [hro] at hr
  have hmem : L[i][t] ∈ ops.toList := by
    rw [hflat]; exact List.mem_flatten.2 ⟨L[i], List.getElem_mem _, List.getElem_mem _⟩
  rw [hps, ofParts_getD ps (by rw [← hps]; exact ht), ← hps]
  exact (hv i hi t ht _ hr (hlt' _ hmem)).symm

/-- **A unit's parameter leaf's word**: leaf `j` of parameter `i` of unit `g`'s callee, resolving to gate `q` of C, is
the word at `q` in the unit's arguments (`argWords`), at the leaf's place among the parameters' leaves. -/
theorem unit_param_word {bytes : ByteArray} {P : Prog} (hd : decode bytes = .ok P)
    (hr : (P.defs[P.root]!).body.recut = #[])
    {g k m : ℕ} (hu : Spot.BlockFront.unitAt P g = .ok (k, m)) (hR : UnitResolves P g)
    (Y : (Flock.OfProgram.ofProgram P).Trace) {i j q : ℕ}
    (hi : i < (P.defs[(P.child P.rootScope k m).d]!).params.size)
    (hj : j < ((P.defs[(P.child P.rootScope k m).d]!).params[i]!).2.leaves)
    (hq : resolve P (P.child P.rootScope k m) ⟨true, i, j⟩ = .ok q) (hqN : q < (Flock.OfProgram.ofProgram P).N) :
    (argWords Y g).getD ((Flock.Program.paramStarts (P.defs[(P.child P.rootScope k m).d]!).params).getD i 0 + j)
      default = portWord Y ⟨q, 0⟩ := by
  obtain ⟨k', m', x, hu', hx0, ⟨ops, hops⟩, -⟩ := hR
  rw [hu] at hu'
  simp only [Except.ok.injEq, Prod.mk.injEq] at hu'
  obtain ⟨rfl, rfl⟩ := hu'
  have hx : (P.defs[P.root]!).body.nodes[k]? = some x := by rw [Array.getElem!_eq_getD]; exact hx0
  have hk : k < (P.defs[P.root]!).body.nodes.size := (Array.getElem?_eq_some_iff.1 hx).1
  have hxk : (P.defs[P.root]!).body.nodes[k]! = x := by
    rw [getElem!_pos _ k hk]; exact (Array.getElem?_eq_some_iff.1 hx).2
  have hx0' : (P.defs[P.root]?.getD default).body.nodes[k]? = some x := by
    rw [← Array.getD_eq_getD_getElem?]; exact hx0
  have hua : unitArgs P g = ops.map fun (q, w) => ({ gate := q, width := w } : Spot.BlockFront.Port) := by
    simp [unitArgs, hu, hx0', hops, bind, Except.bind]
  obtain ⟨raw, hraw, rfl⟩ := operandsW_ok hops
  obtain ⟨st, hP, hinv, -⟩ := BodiesOk.decode_full bytes P hd hr
  have hD : ∀ i < P.defs.size, BodiesOk.DefOk P.defs i P.defs[i]! := by rw [hP]; exact hinv.1
  obtain ⟨hrt, -, -⟩ := BodiesOk.decode_root bytes P hd hr
  obtain ⟨hsh, -⟩ := unitAt_shape hu
  have hq' := unit_param_resolve hr hsh hq
  rw [hxk] at hsh hq'
  rw [child_d, hxk] at hi hj
  rw [child_d, hxk]
  obtain ⟨L, hlen, hflat, hLi⟩ := OperandWidths.operands_lists P P.rootScope x m raw hraw
  obtain ⟨-, hnd, -⟩ := (hD P.root hrt).2.1 k hk
  obtain ⟨hsz, hcnt, -, -, hshape, -⟩ := hnd
  rw [hxk] at hsz hcnt hshape
  have hcntL : ∀ i < x.args.size, OperandWidths.cntOf P x i = ((P.defs[x.fn]!).params[i]!).2.leaves := by
    intro i hi
    unfold OperandWidths.cntOf
    split
    · rfl
    · rename_i hnb
      rw [hcnt i hi]
      rcases hsh with hf | hf
      · simp only [hf, BodiesOk.Shape] at hshape
        rw [hshape.2 i hi]
      · simp only [hf, BodiesOk.Shape] at hshape
        rcases hshape.2 i hi with ⟨-, hty⟩ | ⟨ax, mt, hrem, hmt, hdims⟩
        · rw [hty]
        · obtain ⟨L', hax, -⟩ := BodiesOk.axis_spec ax _ _ mt hrem
          exact (hnb (by rw [hf, hdims, hax]; rfl)).elim
  have hiL : i < L.length := by rw [hlen, hsz]; exact hi
  have hLlen : L.map List.length = (P.defs[x.fn]!).params.toList.map (·.2.leaves) := by
    apply List.ext_getElem (by simp [hlen, hsz])
    intro a h1 h2
    simp only [List.getElem_map, Array.getElem_toList]
    have ha : a < x.args.size := by simpa [hlen] using h1
    rw [(hLi a (by simpa using h1)).1, hcntL a ha, getElem!_pos _ a (by rw [← hsz]; exact ha)]
  have hjL : j < L[i].length := by rw [(hLi i hiL).1, hcntL i (by rw [← hlen]; exact hiL)]; exact hj
  have hqL : L[i][j] = q := by
    have h1 := (hLi i hiL).2 j hjL
    rw [hq'] at h1
    exact (Except.ok.inj h1).symm
  have hpre : (Flock.Program.paramStarts (P.defs[x.fn]!).params).getD i 0 = ((L.take i).map List.length).sum := by
    rw [paramStarts_getD _ hi, List.map_take, List.map_take, hLlen]
  rw [hpre]
  have hget := flatten_getElem?_sum L hiL hjL
  rw [hqL, ← hflat] at hget
  have hidx : ((L.take i).map List.length).sum + j < raw.toList.length := (List.getElem?_eq_some_iff.1 hget).1
  have hws : raw.toList.length = (((P.defs[x.fn]!).params.map fun p => p.2.widths).flatten).toList.length := by
    rw [hflat, List.length_flatten, hLlen]
    simp [Array.toList_flatten, List.length_flatten, Function.comp_def]
    congr 1
    apply List.map_congr_left
    intro p _
    rw [← Array.length_toList, Flock.Sigma.widths_length]
  have hz : (raw.toList.zip (((P.defs[x.fn]!).params.map fun p => p.2.widths).flatten).toList)[
      ((L.take i).map List.length).sum + j]? = some (q, ((((P.defs[x.fn]!).params.map fun p => p.2.widths).flatten).toList)[
        ((L.take i).map List.length).sum + j]'(by omega)) :=
    List.getElem?_zip_eq_some.2 ⟨hget, List.getElem?_eq_getElem _⟩
  unfold argWords
  rw [hua, List.getD_eq_getElem?_getD]
  simp only [Array.toList_map, Array.toList_zip, List.map_map, List.getElem?_map, hz, Option.map_some,
    Option.getD_some, Function.comp_apply]
  rw [portWord_lt Y hqN, portWord_lt Y hqN]

theorem foldl_append_value (w : ℕ) : ∀ (l : List ℕ) (a : Array ℕ),
    (l.foldl (fun a _ => a ++ #[w]) a).toList = a.toList ++ List.replicate l.length w
  | [], a => by simp
  | _ :: l, a => by
    rw [List.foldl_cons, foldl_append_value w l, List.length_cons, List.replicate_succ]
    simp

theorem widths_array_value (n w : ℕ) : (Ty.array n (.value w)).widths.toList = List.replicate n w := by
  simp only [Ty.widths]
  rw [← Array.foldl_toList, Array.toList_range, foldl_append_value]
  simp

/-- **The word at a member of a primitive node of a unit**: in member `m` of root node `k`, which calls a composite, the
gate of member `m'` of node `n`, a primitive or batch node of a primitive of one gate, computed on the trace, is its
primitive on the words `v` gives its operands' references, where `v` gives each reference the word at the gate below it
that it resolves to. -/
theorem unit_member_word {P : Prog} (hroot : P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false)
    (hB : ∀ sc, Resolve.Good P sc → Resolve.BodyOk P sc.d) (hL : ∀ d < P.defs.size, Walk.LaidOut P d)
    (hG : ∀ h < InferenceOnlyVerifier.Properties.gates P, InferenceOnlyVerifier.Properties.GateOk P h)
    {k m : ℕ} (hk : k < (P.defs[P.root]!).body.nodes.size) (hm : m < ((P.defs[P.root]!).body.nodes[k]!).members)
    (hc : (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).prim = false)
    {n : ℕ} (hn : n < (P.defs[(P.child P.rootScope k m).d]!).body.nodes.size) {nd : Node}
    (hnd : (P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]! = nd) {m' : ℕ} (hm' : m' < nd.members)
    (hp : (P.defs[nd.fn]!).prim = true) (hg1 : (P.defs[nd.fn]!).gates = 1)
    (Y : (Flock.OfProgram.ofProgram P).Trace) {G : ℕ}
    (hGd : (P.child P.rootScope k m).off + (P.defs[(P.child P.rootScope k m).d]!).body.offsets[n]! + m' = G)
    (hGN : G < (Flock.OfProgram.ofProgram P).N) (hkey : isInputId (P.defs[nd.fn]!).key = false)
    (hY : Flock.OfProgram.inputAt P G = false → Y ⟨G, hGN⟩ = (Flock.OfProgram.ofProgram P).op ⟨G, hGN⟩ Y)
    {w0 : ℕ} (hret : (P.defs[nd.fn]!).ret = .value w0) (hw : ∀ y, (Prog.evalBool P nd.fn y).length = w0)
    (v : ℕ → ℕ → List Bool)
    (hv : ∀ i < nd.args.size, ∀ t < OperandWidths.cntOf P nd i, ∀ q,
      resolve P (P.child P.rootScope k m) (OperandWidths.refOf nd m' i t) = .ok q → q < G → v i t = portWord Y ⟨q, 0⟩)
    (w : ℕ) :
    portWord Y ⟨G, w⟩ = Prog.evalBool P nd.fn
      ((List.range nd.args.size).flatMap fun i => (List.range (OperandWidths.cntOf P nd i)).map (v i)).flatten := by
  subst hnd
  obtain ⟨p, ops, w', ⟨⟨sc', hg⟩, -, -⟩, hlt⟩ := hG G hGN
  have hg' : P.gate ((P.child P.rootScope k m).off + (P.defs[(P.child P.rootScope k m).d]!).body.offsets[n]! + m') =
      .ok (p, ops, sc') := by rw [hGd]; exact hg
  obtain ⟨rfl, rfl, hops⟩ := unit_gate_at hroot hB hL hk hm hc hn hm' hp hg1 hg'
  have hga : Flock.OfProgram.gateAt P G = some (((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!).fn, ops) := by
    unfold Flock.OfProgram.gateAt; rw [hg]
  have hlt' : ∀ a ∈ ops.toList, a < G := fun a ha => hlt a (Array.mem_toList_iff.1 ha)
  have hin : Flock.OfProgram.inputAt P G = false := by unfold Flock.OfProgram.inputAt; rw [hga]; exact hkey
  have hwd : ∀ y, (Prog.evalBool P ((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!).fn y).length =
      Flock.OfProgram.widthAt P G := fun y => by
    unfold Flock.OfProgram.widthAt; rw [hga]; simp only [hret]; exact hw y
  rw [word_at_computed Y hGN hga hin hlt' (hY hin) (hwd _)]
  congr 1
  obtain ⟨L, hlen, hflat, hLi⟩ := OperandWidths.operands_lists P _ _ m' ops hops
  rw [hflat, List.map_flatten, flatMap_flatten_map]
  congr 2
  apply List.ext_getElem (by simp [hlen])
  intro i h1 h2
  have hi : i < ((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!).args.size := by simpa [hlen] using h1
  obtain ⟨hlc, hres⟩ := hLi i (by simpa using h1)
  simp only [List.getElem_map, List.getElem_range]
  apply List.ext_getElem (by simp [hlc])
  intro t t1 t2
  simp only [List.length_map] at t1
  simp only [List.getElem_map, List.getElem_range]
  have hmem : L[i][t] ∈ ops.toList := by
    rw [hflat]; exact List.mem_flatten.2 ⟨L[i], List.getElem_mem _, List.getElem_mem _⟩
  exact (hv i hi t (by rw [← hlc]; exact t1) _ (hres t t1) (hlt' _ hmem)).symm

/-- **The word at a primitive node of an activation**: in an activation the walk reaches, the gate of primitive node
`n`, computed on the trace, is its primitive on the words `v` gives its arguments' references, where `v`
gives each reference the word at the gate below it that it resolves to. -/
theorem scope_node_word {P : Prog} (hroot : P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false)
    (hB : ∀ sc, Resolve.Good P sc → Resolve.BodyOk P sc.d) (hL : ∀ d < P.defs.size, Walk.LaidOut P d)
    (hG : ∀ h < InferenceOnlyVerifier.Properties.gates P, InferenceOnlyVerifier.Properties.GateOk P h)
    (hD : ∀ d < P.defs.size, BodiesOk.DefOk P.defs d P.defs[d]!) {sc : Scope} (hsc : Resolve.Good P sc)
    {n : ℕ} (hn : n < (P.defs[sc.d]!).body.nodes.size) {nd : Node}
    (hnd : (P.defs[sc.d]!).body.nodes[n]! = nd) (hf : nd.form = .prim)
    (hp : (P.defs[nd.fn]!).prim = true)
    (hseq : ∀ a ∈ nd.args.toList, ∃ ps, a.seq = Seq.ofParts ps)
    (Y : (Flock.OfProgram.ofProgram P).Trace) {G : ℕ}
    (hGd : sc.off + (P.defs[sc.d]!).body.offsets[n]! = G)
    (hGN : G < (Flock.OfProgram.ofProgram P).N) (hkey : isInputId (P.defs[nd.fn]!).key = false)
    (hY : Flock.OfProgram.inputAt P G = false → Y ⟨G, hGN⟩ = (Flock.OfProgram.ofProgram P).op ⟨G, hGN⟩ Y)
    {w0 : ℕ} (hret : (P.defs[nd.fn]!).ret = .value w0) (hw : ∀ y, (Prog.evalBool P nd.fn y).length = w0)
    (v : Ref → List Bool)
    (hv : ∀ i < nd.args.size, ∀ t < (nd.args[i]!).seq.count, ∀ q,
      resolve P sc ((nd.args[i]!).seq.get t) = .ok q → q < G →
        v ((nd.args[i]!).seq.get t) = portWord Y ⟨q, 0⟩) (w : ℕ) :
    portWord Y ⟨G, w⟩ =
      Prog.evalBool P nd.fn ((nd.args.toList.flatMap fun a => Flock.Program.argVals v a a.seq.count id).flatten) := by
  subst hnd
  obtain ⟨p, ops, w', ⟨⟨sc', hg⟩, -, -⟩, hlt⟩ := hG G hGN
  have hg' : P.gate (sc.off + (P.defs[sc.d]!).body.offsets[n]! + 0) =
      .ok (p, ops, sc') := by rw [Nat.add_zero, hGd]; exact hg
  obtain ⟨rfl, hsc', hops⟩ := LeafWidths.gate_of_node_ops P hroot hB hL hD sc hsc n 0 hn hp
    (by rw [members_one (Or.inl hf)]; omega) p ops sc' hg'
  subst sc'
  have hga : Flock.OfProgram.gateAt P G = some (((P.defs[sc.d]!).body.nodes[n]!).fn, ops) := by
    unfold Flock.OfProgram.gateAt; rw [hg]
  have hlt' : ∀ a ∈ ops.toList, a < G := fun a ha => hlt a (Array.mem_toList_iff.1 ha)
  have hin : Flock.OfProgram.inputAt P G = false := by unfold Flock.OfProgram.inputAt; rw [hga]; exact hkey
  have hwd : ∀ y, (Prog.evalBool P ((P.defs[sc.d]!).body.nodes[n]!).fn y).length =
      Flock.OfProgram.widthAt P G := fun y => by
    unfold Flock.OfProgram.widthAt; rw [hga]; simp only [hret]; exact hw y
  rw [word_at_computed Y hGN hga hin hlt' (hY hin) (hwd _)]
  congr 1
  obtain ⟨L, hlen, hflat, hLi⟩ := OperandWidths.operands_lists P _ _ 0 ops hops
  rw [hflat, List.map_flatten, flatMap_flatten_map]
  congr 2
  apply List.ext_getElem (by simp [hlen])
  intro i h1 h2
  have hi : i < ((P.defs[sc.d]!).body.nodes[n]!).args.size := by simpa [hlen] using h1
  simp only [List.getElem_map, Array.getElem_toList]
  obtain ⟨hlc, hres⟩ := hLi i (by simpa using h1)
  have hcnt : L[i].length = (((P.defs[sc.d]!).body.nodes[n]!).args[i]!).seq.count := by
    rw [hlc]; unfold OperandWidths.cntOf; simp [hf, BodiesOk.beq_prim_batch]
  have hai : ((P.defs[sc.d]!).body.nodes[n]!).args[i] =
      ((P.defs[sc.d]!).body.nodes[n]!).args[i]! := (getElem!_pos _ i hi).symm
  obtain ⟨ps, hps⟩ := hseq _ (Array.getElem_mem_toList hi)
  rw [hai] at hps ⊢
  unfold Flock.Program.argVals
  apply List.ext_getElem (by simp [hcnt])
  intro t t1 t2
  simp only [List.length_map, List.length_range] at t1 t2
  have ht : t < (((P.defs[sc.d]!).body.nodes[n]!).args[i]!).seq.count := by simpa using t2
  simp only [List.getElem_map, List.getElem_range, id]
  have hr := hres t t1
  have hro : OperandWidths.refOf ((P.defs[sc.d]!).body.nodes[n]!) 0 i t =
      (((P.defs[sc.d]!).body.nodes[n]!).args[i]!).seq.get t := by
    unfold OperandWidths.refOf; simp [hf, BodiesOk.beq_prim_batch]
  rw [hro] at hr
  have hmem : L[i][t] ∈ ops.toList := by
    rw [hflat]; exact List.mem_flatten.2 ⟨L[i], List.getElem_mem _, List.getElem_mem _⟩
  rw [hps, ofParts_getD ps (by rw [← hps]; exact ht), ← hps]
  exact (hv i hi t ht _ hr (hlt' _ hmem)).symm

/-- **The word at a member of a primitive node of an activation**: in an activation the walk reaches, the gate of
member `m'` of node `n`, a primitive or batch node of a primitive of one gate, computed on the trace, is its
primitive on the words `v` gives its operands' references, where `v` gives each reference the word at the gate below it
that it resolves to. -/
theorem scope_member_word {P : Prog} (hroot : P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false)
    (hB : ∀ sc, Resolve.Good P sc → Resolve.BodyOk P sc.d) (hL : ∀ d < P.defs.size, Walk.LaidOut P d)
    (hG : ∀ h < InferenceOnlyVerifier.Properties.gates P, InferenceOnlyVerifier.Properties.GateOk P h)
    (hD : ∀ d < P.defs.size, BodiesOk.DefOk P.defs d P.defs[d]!) {sc : Scope} (hsc : Resolve.Good P sc)
    {n : ℕ} (hn : n < (P.defs[sc.d]!).body.nodes.size) {nd : Node}
    (hnd : (P.defs[sc.d]!).body.nodes[n]! = nd) {m' : ℕ} (hm' : m' < nd.members)
    (hp : (P.defs[nd.fn]!).prim = true)
    (Y : (Flock.OfProgram.ofProgram P).Trace) {G : ℕ}
    (hGd : sc.off + (P.defs[sc.d]!).body.offsets[n]! + m' = G)
    (hGN : G < (Flock.OfProgram.ofProgram P).N) (hkey : isInputId (P.defs[nd.fn]!).key = false)
    (hY : Flock.OfProgram.inputAt P G = false → Y ⟨G, hGN⟩ = (Flock.OfProgram.ofProgram P).op ⟨G, hGN⟩ Y)
    {w0 : ℕ} (hret : (P.defs[nd.fn]!).ret = .value w0) (hw : ∀ y, (Prog.evalBool P nd.fn y).length = w0)
    (v : ℕ → ℕ → List Bool)
    (hv : ∀ i < nd.args.size, ∀ t < OperandWidths.cntOf P nd i, ∀ q,
      resolve P sc (OperandWidths.refOf nd m' i t) = .ok q → q < G → v i t = portWord Y ⟨q, 0⟩)
    (w : ℕ) :
    portWord Y ⟨G, w⟩ = Prog.evalBool P nd.fn
      ((List.range nd.args.size).flatMap fun i => (List.range (OperandWidths.cntOf P nd i)).map (v i)).flatten := by
  subst hnd
  obtain ⟨p, ops, w', ⟨⟨sc', hg⟩, -, -⟩, hlt⟩ := hG G hGN
  have hg' : P.gate (sc.off + (P.defs[sc.d]!).body.offsets[n]! + m') =
      .ok (p, ops, sc') := by rw [hGd]; exact hg
  obtain ⟨rfl, hsc', hops⟩ := LeafWidths.gate_of_node_ops P hroot hB hL hD sc hsc n m' hn hp hm' p ops sc' hg'
  subst sc'
  have hga : Flock.OfProgram.gateAt P G = some (((P.defs[sc.d]!).body.nodes[n]!).fn, ops) := by
    unfold Flock.OfProgram.gateAt; rw [hg]
  have hlt' : ∀ a ∈ ops.toList, a < G := fun a ha => hlt a (Array.mem_toList_iff.1 ha)
  have hin : Flock.OfProgram.inputAt P G = false := by unfold Flock.OfProgram.inputAt; rw [hga]; exact hkey
  have hwd : ∀ y, (Prog.evalBool P ((P.defs[sc.d]!).body.nodes[n]!).fn y).length =
      Flock.OfProgram.widthAt P G := fun y => by
    unfold Flock.OfProgram.widthAt; rw [hga]; simp only [hret]; exact hw y
  rw [word_at_computed Y hGN hga hin hlt' (hY hin) (hwd _)]
  congr 1
  obtain ⟨L, hlen, hflat, hLi⟩ := OperandWidths.operands_lists P _ _ m' ops hops
  rw [hflat, List.map_flatten, flatMap_flatten_map]
  congr 2
  apply List.ext_getElem (by simp [hlen])
  intro i h1 h2
  have hi : i < ((P.defs[sc.d]!).body.nodes[n]!).args.size := by simpa [hlen] using h1
  obtain ⟨hlc, hres⟩ := hLi i (by simpa using h1)
  simp only [List.getElem_map, List.getElem_range]
  apply List.ext_getElem (by simp [hlc])
  intro t t1 t2
  simp only [List.length_map] at t1
  simp only [List.getElem_map, List.getElem_range]
  have hmem : L[i][t] ∈ ops.toList := by
    rw [hflat]; exact List.mem_flatten.2 ⟨L[i], List.getElem_mem _, List.getElem_mem _⟩
  exact (hv i hi t (by rw [← hlc]; exact t1) _ (hres t t1) (hlt' _ hmem)).symm

/-- A primitive or call node's arguments as the walk reads them are its operands' references in order. -/
theorem argVals_seq {W : Type} [Inhabited W] (P : Prog) (v : Ref → W) {nd : Node} (hf : nd.form = .prim ∨ nd.form = .call)
    (hseq : ∀ a ∈ nd.args.toList, ∃ ps, a.seq = Seq.ofParts ps) :
    nd.args.toList.flatMap (fun a => Flock.Program.argVals v a a.seq.count id) =
      (List.range nd.args.size).flatMap fun i =>
        (List.range (OperandWidths.cntOf P nd i)).map fun t => v (OperandWidths.refOf nd 0 i t) := by
  rw [flatMap_flatten_map, flatMap_flatten_map]
  congr 1
  apply List.ext_getElem (by simp)
  intro i h1 h2
  have hi : i < nd.args.size := by simpa using h1
  simp only [List.getElem_map, List.getElem_range, Array.getElem_toList]
  have hai : nd.args[i] = nd.args[i]! := (getElem!_pos _ i hi).symm
  obtain ⟨ps, hps⟩ := hseq _ (Array.getElem_mem_toList hi)
  rw [hai] at hps ⊢
  have hcnt : OperandWidths.cntOf P nd i = (nd.args[i]!).seq.count := by
    unfold OperandWidths.cntOf; rcases hf with hf | hf <;> simp [hf, BodiesOk.beq_prim_batch, BodiesOk.beq_call_batch]
  rw [hcnt]
  unfold Flock.Program.argVals
  apply List.ext_getElem (by simp)
  intro t t1 t2
  simp only [List.length_map, List.length_range] at t1
  simp only [List.getElem_map, List.getElem_range, id]
  have hro : OperandWidths.refOf nd 0 i t = (nd.args[i]!).seq.get t := by
    unfold OperandWidths.refOf; rcases hf with hf | hf <;> simp [hf, BodiesOk.beq_prim_batch, BodiesOk.beq_call_batch]
  rw [hro, hps, ofParts_getD ps (by rw [← hps]; exact t1)]

/-- A batch node's member's arguments as the walk reads them (`batchArg`) are its operands' references in order. -/
theorem batchArgs_eq {W : Type} [Inhabited W] (P : Prog) (v : Ref → W) {nd : Node} (hf : nd.form = .batch)
    (hseq : ∀ a ∈ nd.args.toList, ∃ ps, a.seq = Seq.ofParts ps) (m' : ℕ)
    (hop : ∀ i < nd.args.size, ∀ t < OperandWidths.cntOf P nd i,
      Resolve.memberIdx (nd.args[i]!) m' t < (nd.args[i]!).seq.count) :
    ((List.range nd.args.size).flatMap fun i => batchArg v (P.defs[nd.fn]!) nd m' i) =
      (List.range nd.args.size).flatMap fun i =>
        (List.range (OperandWidths.cntOf P nd i)).map fun t => v (OperandWidths.refOf nd m' i t) := by
  apply List.flatMap_congr
  intro i hi
  have hi' := List.mem_range.1 hi
  have hgd : nd.args.getD i default = nd.args[i]! := Array.getElem!_eq_getD.symm
  obtain ⟨ps, hps⟩ := hseq _ (Array.getElem_mem_toList hi')
  rw [← getElem!_pos _ i hi'] at hps
  have hro : ∀ t, OperandWidths.refOf nd m' i t = (nd.args[i]!).seq.get (Resolve.memberIdx (nd.args[i]!) m' t) := by
    intro t; unfold OperandWidths.refOf; rw [if_pos (by rw [hf]; rfl), Resolve.memberRef_eq]
  cases hd : (nd.args[i]!).dims with
  | none =>
    rw [batchArg_none _ _ _ _ _ (by rw [hgd]; exact hd), hgd]
    have hcnt : OperandWidths.cntOf P nd i = (nd.args[i]!).seq.count := by
      unfold OperandWidths.cntOf; simp [hd]
    rw [hcnt]
    unfold Flock.Program.argVals
    apply List.ext_getElem (by simp)
    intro t t1 t2
    simp only [List.length_map, List.length_range] at t1
    simp only [List.getElem_map, List.getElem_range, id]
    rw [hro, show Resolve.memberIdx (nd.args[i]!) m' t = t by simp only [Resolve.memberIdx, hd], hps,
      ofParts_getD ps (by rw [← hps]; exact t1)]
  | some dl =>
    obtain ⟨dd, L⟩ := dl
    rw [batchArg_some _ _ _ _ _ (by rw [hgd]; exact hd), hgd]
    have hcnt : OperandWidths.cntOf P nd i = ((P.defs[nd.fn]!).params[i]!).2.leaves := by
      unfold OperandWidths.cntOf; rw [if_pos (by rw [hf, hd]; rfl)]
    have hpl : ((P.defs[nd.fn]!).params.getD i default).2.leaves = ((P.defs[nd.fn]!).params[i]!).2.leaves := by
      simp only [Array.getElem!_eq_getD]
    rw [hpl, ← hcnt]
    unfold Flock.Program.argVals
    apply List.ext_getElem (by simp)
    intro t t1 t2
    simp only [List.length_map, List.length_range] at t1
    simp only [List.getElem_map, List.getElem_range]
    have hidx : Resolve.memberIdx (nd.args[i]!) m' t = t / L * dd * L + t % L + m' * L := by
      simp only [Resolve.memberIdx, hd, Nat.add_mul]; exact Nat.add_right_comm _ _ _
    have hb := hop i hi' t t1
    rw [hro, ← hidx, hps, ofParts_getD ps (by rw [← hps]; exact hb)]

/-- **A reference's word in a unit's walk**: where each node before `n` holds the words of its members' gates, a
reference to it at a leaf below its count resolving to its member's gate, and each parameter leaf's word is the gate it
resolves to, so is the word of a reference of the body before node `n`. -/
theorem refVal_word_members {P : Prog} {sc : Scope} {W : ℕ → List Bool} {Q : ℕ → Prop} {x : List (List Bool)} {n : ℕ}
    {outs : Array (List (List Bool))} {M : ℕ → ℕ}
    (hO : ∀ i < n, outs.getD i [] = (List.range (M i)).map fun j => W (sc.off + (P.defs[sc.d]!).body.offsets[i]! + j))
    (hN : ∀ i < n, ∀ j < ((P.defs[sc.d]!).body.nodes[i]!).out.leaves,
      j < M i ∧ resolve P sc ⟨false, i, j⟩ = .ok (sc.off + (P.defs[sc.d]!).body.offsets[i]! + j))
    (hx : ∀ i j q, i < (P.defs[sc.d]!).body.params.size → j < ((P.defs[sc.d]!).body.params[i]!).2.leaves →
      resolve P sc ⟨true, i, j⟩ = .ok q → Q q →
        x.getD ((Flock.Program.paramStarts (P.defs[sc.d]!).params).getD i 0 + j) default = W q)
    {r : Ref} (hr : Resolve.RefOk P sc.d n r) {q : ℕ} (hq : resolve P sc r = .ok q) (hQ : Q q) :
    Flock.Program.refVal x (Flock.Program.paramStarts (P.defs[sc.d]!).params) outs r = W q := by
  obtain ⟨p, i, j⟩ := r
  cases p with
  | true =>
    obtain ⟨h1, h2⟩ := hr.1 rfl
    simp only [Flock.Program.refVal, ite_true]
    exact hx i j q h1 h2 hq hQ
  | false =>
    obtain ⟨h1, h2⟩ := hr.2 rfl
    obtain ⟨hjM, hres⟩ := hN i h1 j h2
    rw [hres] at hq
    obtain rfl := Except.ok.inj hq
    simp [Flock.Program.refVal, hO i h1, List.getD_eq_getElem?_getD, hjM]

/-- An operand's reference of member `m'` of node `n` of a body the walk reaches names a parameter or an earlier node at
a leaf. -/
theorem refOf_refOk {P : Prog} {d n : ℕ} (hB : Resolve.BodyOk P d) (hn : n < (P.defs[d]!).body.nodes.size)
    (hp : (P.defs[((P.defs[d]!).body.nodes[n]!).fn]!).prim = true) {m' i t : ℕ}
    (hi : i < ((P.defs[d]!).body.nodes[n]!).args.size) (hm' : m' < ((P.defs[d]!).body.nodes[n]!).members)
    (ht : t < OperandWidths.cntOf P ((P.defs[d]!).body.nodes[n]!) i) :
    Resolve.RefOk P d n (OperandWidths.refOf ((P.defs[d]!).body.nodes[n]!) m' i t) := by
  have hop := ((hB.1 n hn).2.2.2.2 hp) i hi m' hm' t (by unfold OperandWidths.cntOf at ht; exact ht)
  unfold OperandWidths.refOf
  split
  · rename_i hb
    rw [Resolve.memberRef_eq]
    rw [if_pos hb] at hop
    exact (hB.1 n hn).2.1 i _ hop
  · rename_i hb
    rw [if_neg hb] at hop
    exact (hB.1 n hn).2.1 i t hop

/-- **`UnitWordComputes` for a template of word-primitive calls and axis-0 batches**, on `Batch0Typed`: where C's gates
are located and unit `g` resolves. -/
theorem unitWordComputes_batch0 (hT : Batch0Typed) {bytes : ByteArray} {P : Prog} (hd : decode bytes = .ok P)
    (hr : (P.defs[P.root]!).body.recut = #[])
    (hG : ∀ h < InferenceOnlyVerifier.Properties.gates P, InferenceOnlyVerifier.Properties.GateOk P h)
    {g k m : ℕ} (hu : Spot.BlockFront.unitAt P g = .ok (k, m)) (hR : UnitResolves P g)
    (hb0 : Batch0 P (P.child P.rootScope k m).d) (hgb : GateBodies P (P.child P.rootScope k m).d)
    (Y : (Flock.OfProgram.ofProgram P).Trace)
    (hY : ∀ h : Fin (Flock.OfProgram.ofProgram P).N, InUnit P k m h.val → Flock.OfProgram.inputAt P h = false →
      Y h = (Flock.OfProgram.ofProgram P).op h Y) :
    outWords Y g = wordEval P (P.child P.rootScope k m).d (argWords Y g) := by
  have hR' := hR
  obtain ⟨k', m', x, hu', hx0, -, hres⟩ := hR
  rw [hu] at hu'
  simp only [Except.ok.injEq, Prod.mk.injEq] at hu'
  obtain ⟨rfl, rfl⟩ := hu'
  have hx : (P.defs[P.root]!).body.nodes[k]? = some x := by rw [Array.getElem!_eq_getD]; exact hx0
  have hk : k < (P.defs[P.root]!).body.nodes.size := (Array.getElem?_eq_some_iff.1 hx).1
  have hxk : (P.defs[P.root]!).body.nodes[k]! = x := by
    rw [getElem!_pos _ k hk]; exact (Array.getElem?_eq_some_iff.1 hx).2
  obtain ⟨st0, hP, hinv, -⟩ := BodiesOk.decode_full bytes P hd hr
  have hD : ∀ i < P.defs.size, BodiesOk.DefOk P.defs i P.defs[i]! := by rw [hP]; exact hinv.1
  obtain ⟨hrt, hrp, -⟩ := BodiesOk.decode_root bytes P hd hr
  have hroot : P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false := ⟨hrt, hrp⟩
  have hB : ∀ sc, Resolve.Good P sc → Resolve.BodyOk P sc.d := fun sc _ => BodiesOk.decoded_bodyOk bytes P hd hr sc.d
  have hL := OperandWidths.decoded_laid bytes P hd hr
  obtain ⟨hsh, hmem⟩ := unitAt_shape hu
  have hc0 : (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).prim = false := hb0.1
  have hgood : Resolve.Good P (P.child P.rootScope k m) :=
    Resolve.good_child P (parent := P.rootScope) ⟨rfl, rfl, rfl⟩ hk rfl hc0 hmem rfl
  obtain ⟨hdl, hpc⟩ := Resolve.good_d P hroot hB _ hgood
  obtain ⟨-, hK, hnodes⟩ := hL _ hdl hpc
  -- the unit's gates are C's
  have hbound : (P.child P.rootScope k m).off + (P.defs[(P.child P.rootScope k m).d]!).gates ≤
      (Flock.OfProgram.ofProgram P).N := by
    have hoff : (P.child P.rootScope k m).off = (P.defs[P.root]!).body.offsets[k]! +
        m * (P.defs[(P.child P.rootScope k m).d]!).gates := by
      show 0 + _ + _ = _; rw [Nat.zero_add]; rfl
    obtain ⟨-, hKr, hnr⟩ := hL _ hrt hrp
    have h1 := hnr k hk
    have h2 := Resolve.offsets_mono P P.root (hL _ hrt) hrp (k + 1) _ (by omega) (Nat.le_refl _)
    have h3 := Nat.mul_le_mul_right (P.defs[(P.child P.rootScope k m).d]!).gates
      (show m + 1 ≤ ((P.defs[P.root]!).body.nodes[k]!).members by omega)
    rw [Nat.add_mul, Nat.one_mul] at h3
    have h4 : (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).gates =
        (P.defs[(P.child P.rootScope k m).d]!).gates := rfl
    rw [h4] at h1
    show _ ≤ (P.defs[P.root]!).gates
    omega
  -- the template's nodes: primitive or batch nodes of a word primitive of one gate and one leaf
  have hnodeP : ∀ n (hn : n < (P.defs[(P.child P.rootScope k m).d]!).body.nodes.size),
      (((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!).form = .prim ∨ ((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!).form = .batch) ∧
      ((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!).fn < P.defs.size ∧ ((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!).fn < (P.child P.rootScope k m).d ∧ (P.defs[((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!).fn]!).prim = true ∧
      isWordPrim (P.defs[((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!).fn]!).prim (P.defs[((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!).fn]!).key = true ∧
      (P.defs[((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!).fn]!).gates = 1 ∧ (P.defs[((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!).fn]!).outLeaves = 1 ∧
      (∃ w0, (P.defs[((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!).fn]!).ret = .value w0 ∧ ((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!).out.widths.toList = List.replicate ((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!).members w0) := by
    intro n hn
    have hmemn : ((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!) ∈ (P.defs[(P.child P.rootScope k m).d]!).body.nodes := by
      rw [getElem!_pos _ n hn]; exact Array.getElem_mem hn
    obtain ⟨hform, hwp⟩ := hb0.2 _ hmemn
    have hprim : (P.defs[((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!).fn]!).prim = true := by
      simp only [isWordPrim, Bool.and_eq_true] at hwp
      exact hwp.1.1
    obtain ⟨hfnlt, hnok, -⟩ := (hD _ hdl).2.1 n hn
    obtain ⟨-, -, -, hcallp, hshape, -⟩ := hnok
    have hfnP : ((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!).fn < P.defs.size := by omega
    have hf : ((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!).form = .prim ∨ ((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!).form = .batch := by
      rcases hform with h | h | ⟨h, -⟩
      · exact Or.inl h
      · have := hcallp (Or.inl h); rw [hprim] at this; cases this
      · exact Or.inr h
    obtain ⟨-, hg1, hol, w0, hret⟩ := (hD _ hfnP).1.1 hprim
    refine ⟨hf, hfnP, hfnlt, hprim, hwp, hg1, hol, w0, hret, ?_⟩
    rcases hf with hf | hf
    · rw [hf] at hshape
      rw [hshape.1, hret, members_one (Or.inl hf)]
      simp [Ty.widths]
    · rw [hf] at hshape
      rw [hshape.1, hret, members_batch hf]
      exact widths_array_value _ _
  -- a reference to an earlier node at a leaf below its count resolves to the gate of that member
  have hNres : ∀ i < (P.defs[(P.child P.rootScope k m).d]!).body.nodes.size, ∀ j < ((P.defs[(P.child P.rootScope k m).d]!).body.nodes[i]!).out.leaves,
      j < ((P.defs[(P.child P.rootScope k m).d]!).body.nodes[i]!).members ∧
      resolve P (P.child P.rootScope k m) ⟨false, i, j⟩ =
        .ok ((P.child P.rootScope k m).off + (P.defs[(P.child P.rootScope k m).d]!).body.offsets[i]! + j) := by
    intro i hi j hj
    obtain ⟨hf, -, -, hprim, -, -, hol, w0, -, hwid⟩ := hnodeP i hi
    have hlv : ((P.defs[(P.child P.rootScope k m).d]!).body.nodes[i]!).out.leaves = ((P.defs[(P.child P.rootScope k m).d]!).body.nodes[i]!).members := by
      rw [← Flock.Sigma.widths_length, hwid, List.length_replicate]
    refine ⟨by omega, ?_⟩
    rcases hf with hf | hf
    · have : j = 0 := by rw [members_one (Or.inl hf)] at hlv; omega
      subst this
      rw [Nat.add_zero]
      exact resolve_prim_node hr _ hf
    · exact resolve_batch_prim hr _ hf hprim hol
  have hbp : (P.defs[(P.child P.rootScope k m).d]!).body.params = (P.defs[(P.child P.rootScope k m).d]!).params :=
    ((hD _ hdl).1.2 hpc).1
  have hx' : ∀ i j q, i < (P.defs[(P.child P.rootScope k m).d]!).body.params.size →
      j < ((P.defs[(P.child P.rootScope k m).d]!).body.params[i]!).2.leaves →
      resolve P (P.child P.rootScope k m) ⟨true, i, j⟩ = .ok q → q < (Flock.OfProgram.ofProgram P).N →
        (argWords Y g).getD ((Flock.Program.paramStarts (P.defs[(P.child P.rootScope k m).d]!).params).getD i 0 + j)
          default = portWord Y ⟨q, 0⟩ := by
    intro i j q h1 h2 hq hqN
    rw [hbp] at h1 h2
    exact unit_param_word hd hr hu hR' Y h1 h2 hq hqN
  have hPd : P.defs[(P.child P.rootScope k m).d]? = some (P.defs[(P.child P.rootScope k m).d]!) := by
    rw [getElem!_pos P.defs _ hdl]; exact getElem?_pos P.defs _ hdl
  obtain ⟨-, -, -, hout, -, -⟩ := hT bytes P hd hr _ hdl hb0 hgb
  -- the walk over the template's nodes
  have hstep : ∀ n (hn : n < (P.defs[(P.child P.rootScope k m).d]!).body.nodes.toList.length)
      (outs : Array (List (List Bool))),
      (outs.size = n ∧ ∀ i < n, outs.getD i [] = (List.range ((P.defs[(P.child P.rootScope k m).d]!).body.nodes[i]!).members).map
        fun j => portWord Y ⟨(P.child P.rootScope k m).off + (P.defs[(P.child P.rootScope k m).d]!).body.offsets[i]! + j, 0⟩) →
      ∃ b', nodeStep wordWires P (wordDef? P (2 * P.defs.size + 1)) (P.child P.rootScope k m).d (argWords Y g)
          (P.defs[(P.child P.rootScope k m).d]!) outs (P.defs[(P.child P.rootScope k m).d]!).body.nodes.toList[n] = some b' ∧
        (b'.size = n + 1 ∧ ∀ i < n + 1, b'.getD i [] = (List.range ((P.defs[(P.child P.rootScope k m).d]!).body.nodes[i]!).members).map
          fun j => portWord Y ⟨(P.child P.rootScope k m).off + (P.defs[(P.child P.rootScope k m).d]!).body.offsets[i]! + j, 0⟩) := by
    intro n hn outs ⟨hs, hO⟩
    have hn' : n < (P.defs[(P.child P.rootScope k m).d]!).body.nodes.size := by simpa using hn
    have hnn : (P.defs[(P.child P.rootScope k m).d]!).body.nodes.toList[n] = ((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!) := by
      rw [Array.getElem_toList, getElem!_pos _ n hn']
    rw [hnn]
    obtain ⟨hf, hfnP, hfnd, hprim, hwp, hg1, -, w0, hret, hwid⟩ := hnodeP n hn'
    have hcal : P.defs[((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!).fn]? = some (P.defs[((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!).fn]!) := by
      rw [getElem!_pos P.defs ((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!).fn hfnP]
      exact getElem?_pos P.defs _ hfnP
    have hstep1 := (hnodes n hn')
    rw [hg1, Nat.mul_one] at hstep1
    have hend := Resolve.offsets_mono P _ (hL _ hdl) hpc (n + 1) _ (by omega) (Nat.le_refl _)
    have hkey := (FlockSoundness.DecodeBoolean.word_ids hwp).1
    have hw : 0 < ((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!).members → ∀ y, (Prog.evalBool P ((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!).fn y).length = w0 := by
      intro hpos y
      have h1 := hout n hn' (List.replicate ((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!).members y) (by rw [List.length_replicate, ← getElem!_pos _ n hn'])
      rw [← getElem!_pos _ n hn', hwid, List.map_replicate] at h1
      have hm : w0 ∈ List.replicate ((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!).members w0 := List.mem_replicate.2 ⟨by omega, rfl⟩
      rw [h1] at hm
      exact (List.mem_replicate.1 hm).2.symm
    have hseq : ∀ a ∈ ((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!).args.toList, ∃ ps, a.seq = Seq.ofParts ps := by
      intro a ha
      obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem ha
      have hi' : i < ((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!).args.size := by simpa using hi
      obtain ⟨-, -, hargsF⟩ := (hD _ hdl).2.1 n hn'
      obtain ⟨⟨ps, hps, -⟩, -⟩ := hargsF i hi'
      exact ⟨ps, by rw [Array.getElem_toList, ← getElem!_pos _ i hi']; exact hps⟩
    have hrv : ∀ (r : Ref), Resolve.RefOk P (P.child P.rootScope k m).d n r → ∀ q, resolve P (P.child P.rootScope k m) r = .ok q →
        q < (Flock.OfProgram.ofProgram P).N →
        Flock.Program.refVal (argWords Y g) (Flock.Program.paramStarts (P.defs[(P.child P.rootScope k m).d]!).params) outs r =
          portWord Y ⟨q, 0⟩ := fun r hr' q hq hqN =>
      refVal_word_members (W := fun q => portWord Y ⟨q, 0⟩) (Q := fun q => q < (Flock.OfProgram.ofProgram P).N)
        (n := n) (M := fun i => ((P.defs[(P.child P.rootScope k m).d]!).body.nodes[i]!).members) hO (fun i hi => hNres i (by omega)) hx'
        hr' hq hqN
    rcases hf with hf | hf
    · rw [members_one (Or.inl hf)] at hstep1
      have hGN : (P.child P.rootScope k m).off + (P.defs[(P.child P.rootScope k m).d]!).body.offsets[n]! < (Flock.OfProgram.ofProgram P).N := by omega
      have hval := unit_node_word hroot hB hL hG hk hmem hc0 hn' rfl hf hprim hg1 hseq Y rfl hGN hkey
        (fun hin => hY ⟨_, hGN⟩ (by unfold InUnit; simp only [hPd, Option.map_some, Option.getD_some]; omega) hin)
        hret (hw (by rw [members_one (Or.inl hf)]; omega))
        (Flock.Program.refVal (argWords Y g) (Flock.Program.paramStarts (P.defs[(P.child P.rootScope k m).d]!).params) outs)
        (fun i hi t ht q hq hqG => hrv _ (((hB _ hgood).1 n hn').2.1 i t ht) q hq (by omega)) 0
      refine ⟨outs.push [Prog.evalBool P ((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!).fn
        ((((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!).args.toList.flatMap fun a =>
          Flock.Program.argVals (Flock.Program.refVal (argWords Y g)
            (Flock.Program.paramStarts (P.defs[(P.child P.rootScope k m).d]!).params) outs) a a.seq.count id).flatten)],
        ?_, by simp [hs], ?_⟩
      · simp [nodeStep, wordWires, hf, hcal, show ¬ (P.child P.rootScope k m).d ≤ ((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!).fn by omega, wordDef_prim hfnP hprim]
      · intro i hi
        by_cases hin : i < n
        · rw [← hO i hin]
          simp [Array.getD, Array.getElem_push, hs, hin, show i < n + 1 by omega]
        · have : i = n := by omega
          subst this
          rw [members_one (Or.inl hf)]
          simp [Array.getD, Array.getElem_push, hs]
          exact hval.symm
    · rw [members_batch hf] at hstep1
      have hGN : ∀ m' < ((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!).n, (P.child P.rootScope k m).off + (P.defs[(P.child P.rootScope k m).d]!).body.offsets[n]! + m' < (Flock.OfProgram.ofProgram P).N :=
        fun m' hm' => by omega
      have hmw : ∀ m' < ((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!).n, Prog.evalBool P ((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!).fn (((List.range ((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!).args.size).flatMap fun i =>
          batchArg (Flock.Program.refVal (argWords Y g) (Flock.Program.paramStarts (P.defs[(P.child P.rootScope k m).d]!).params) outs)
            (P.defs[((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!).fn]!) ((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!) m' i).flatten) = portWord Y ⟨(P.child P.rootScope k m).off + (P.defs[(P.child P.rootScope k m).d]!).body.offsets[n]! + m', 0⟩ := by
        intro m' hm'
        have hm'' : m' < ((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!).members := by rw [members_batch hf]; exact hm'
        have hop := ((hB _ hgood).1 n hn').2.2.2.2 hprim
        rw [batchArgs_eq P _ hf hseq m' (fun i hi t ht => by
          have := hop i hi m' hm'' t (by unfold OperandWidths.cntOf at ht; exact ht)
          rwa [if_pos (by rw [hf]; rfl)] at this)]
        exact (unit_member_word hroot hB hL hG hk hmem hc0 hn' rfl hm'' hprim hg1 Y rfl (hGN m' hm') hkey
          (fun hin => hY ⟨_, hGN m' hm'⟩
            (by unfold InUnit; simp only [hPd, Option.map_some, Option.getD_some]; omega) hin)
          hret (hw (by omega))
          (fun i t => Flock.Program.refVal (argWords Y g) (Flock.Program.paramStarts (P.defs[(P.child P.rootScope k m).d]!).params) outs
            (OperandWidths.refOf ((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!) m' i t))
          (fun i hi t ht q hq hqG => hrv _ (refOf_refOk (hB _ hgood) hn' hprim hi hm'' ht) q hq (by omega)) 0).symm
      obtain ⟨X, hX⟩ : ∃ X, X = (List.range ((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!).n).map fun m' => Prog.evalBool P ((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!).fn
          (((List.range ((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!).args.size).flatMap fun i =>
            batchArg (Flock.Program.refVal (argWords Y g) (Flock.Program.paramStarts (P.defs[(P.child P.rootScope k m).d]!).params) outs)
              (P.defs[((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!).fn]!) ((P.defs[(P.child P.rootScope k m).d]!).body.nodes[n]!) m' i).flatten) := ⟨_, rfl⟩
      refine ⟨outs.push X, ?_, by simp [hs], ?_⟩
      · rw [hX]
        simp only [nodeStep_batch wordWires (fun _ => rfl) P _ _ _ _ _ _ hf hfnd hcal, wordDef_prim hfnP hprim,
          mapM_some_map, Option.bind_eq_bind, Option.bind_some, Option.pure_def, flatten_map_singleton]
      · intro i hi
        by_cases hin : i < n
        · rw [← hO i hin]
          simp [Array.getD, Array.getElem_push, hs, hin, show i < n + 1 by omega]
        · have : i = n := by omega
          subst this
          have hgx : (outs.push X).getD i [] = X := by simp [Array.getD, Array.getElem_push, hs]
          rw [hgx, hX, members_batch hf]
          exact List.map_congr_left fun m' hm' => hmw m' (List.mem_range.1 hm')
  obtain ⟨o, g1, ho1, ho2⟩ := foldlM_inv _ _ _ hstep (P.defs[(P.child P.rootScope k m).d]!).body.nodes.toList.length 0
    #[] (Nat.add_zero _) ⟨rfl, fun i h => absurd h (Nat.not_lt_zero _)⟩
  simp only [List.drop_zero] at g1
  unfold wordEval
  rw [show 2 * P.defs.size + 2 = (2 * P.defs.size + 1) + 1 from rfl]
  unfold wordDef?
  simp only [hPd, Option.bind_eq_bind, Option.bind_some, hpc, Bool.false_eq_true, ↓reduceIte]
  rw [walkStep_comp wordWires (fun _ => rfl) P _ _ _ hPd hpc, g1]
  simp only [Option.bind_eq_bind, Option.bind_some, Option.pure_def, Option.getD_some]
  -- the return leaves
  obtain ⟨-, hcount, houtL⟩ := (hD _ hdl).1.2 hpc
  obtain ⟨ps, hps, -⟩ := (hD _ hdl).2.2.2
  have hcd : (P.child P.rootScope k m).d = x.fn := by rw [child_d, hxk]
  have hc : P.defs.getD x.fn default = P.defs[(P.child P.rootScope k m).d]! := by
    rw [hcd]; exact Array.getElem!_eq_getD.symm
  rw [hc] at hres
  let F : ℕ → V Spot.BlockFront.Port := fun l => do
    let q ← resolve P P.rootScope
      { p := false, i := k, j := m * (P.defs[(P.child P.rootScope k m).d]!).ret.leaves + l }
    pure ({ gate := q, width := (P.defs[(P.child P.rootScope k m).d]!).ret.widths.getD l 0 } : Spot.BlockFront.Port)
  obtain ⟨r, hrm, h2⟩ := mapM_forall₂ F (List.range (P.defs[(P.child P.rootScope k m).d]!).ret.leaves) fun l hl => by
    obtain ⟨q, hq⟩ := hres l (List.mem_range.1 hl)
    exact ⟨_, by simp only [F, hq]; rfl⟩
  have huo : unitOuts P g = r.toArray := by
    unfold unitOuts
    simp only [hu, hx0, hc, bind, Except.bind]
    rw [Array.mapM_eq_mapM_toList, Array.toList_range]
    change (match (Functor.map List.toArray
      (List.mapM F (List.range (P.defs[(P.child P.rootScope k m).d]!).ret.leaves)) : V _) with
      | .ok ps => ps | .error _ => #[]) = _
    rw [hrm]
    rfl
  unfold outWords
  rw [huo, List.toList_toArray]
  have hrlen : r.length = (P.defs[(P.child P.rootScope k m).d]!).ret.leaves := by
    rw [← List.Forall₂.length_eq h2, List.length_range]
  apply List.ext_getElem (by simp [hrlen, hcount])
  intro l h1 h1'
  have hl : l < (P.defs[(P.child P.rootScope k m).d]!).ret.leaves := by rw [← hrlen]; simpa using h1
  simp only [List.getElem_map, List.getElem_range]
  have hF := List.Forall₂.get h2 (by simpa using hl) (by simpa using h1)
  simp only [List.get_eq_getElem, List.getElem_range] at hF
  cases hq : resolve P P.rootScope
      { p := false, i := k, j := m * (P.defs[(P.child P.rootScope k m).d]!).ret.leaves + l } with
  | error e => simp [F, hq, bind, Except.bind] at hF
  | ok q =>
    simp only [F, hq, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at hF
    rw [← hF]
    have hsh' : (((P.defs[P.root]!).body.nodes[k]!).form = .call ∧ m = 0) ∨
        ((P.defs[P.root]!).body.nodes[k]!).form = .batch := by
      rcases hsh with h | h
      · exact Or.inl ⟨h, by have := hmem; rw [members_one (Or.inr h)] at this; omega⟩
      · exact Or.inr h
    have hq0 : resolve P P.rootScope ⟨false, k,
        m * (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).outLeaves + l⟩ = .ok q := by
      have h3 : (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).outLeaves =
          (P.defs[(P.child P.rootScope k m).d]!).ret.leaves := houtL
      rw [h3]; exact hq
    have hretq := unit_ret_resolve hr hc0 hsh' (by
      have h3 : (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).outLeaves =
          (P.defs[(P.child P.rootScope k m).d]!).ret.leaves := houtL
      rw [h3]; exact hl) hq0
    have hlc : l < (P.defs[(P.child P.rootScope k m).d]!).body.ret.count := by rw [hcount]; exact hl
    have hRef := (hB _ hgood).2 l hlc
    have hqN : q < (Flock.OfProgram.ofProgram P).N := by
      have hinv' : Resolve.Inv P ((P.child P.rootScope k m).off +
          (P.defs[(P.child P.rootScope k m).d]!).body.offsets[(P.defs[(P.child P.rootScope k m).d]!).body.nodes.size]!)
          (P.child P.rootScope k m, (P.defs[(P.child P.rootScope k m).d]!).body.ret.get l) := by
        refine ⟨hgood, hRef, fun _ => Nat.le_add_right _ _, fun hp => ?_⟩
        dsimp only at hp ⊢
        have h4 := (hRef.2 hp).1
        have := Resolve.offsets_mono P _ (hL _ hdl) hpc _ _ (show _ + 1 ≤ _ from h4) (Nat.le_refl _)
        omega
      have := Resolve.resolve_lt P hroot hB hL _ _ _ hinv' hretq
      omega
    have hO' : ∀ i < (P.defs[(P.child P.rootScope k m).d]!).body.nodes.size, o.getD i [] =
        (List.range ((P.defs[(P.child P.rootScope k m).d]!).body.nodes[i]!).members).map
          fun j => portWord Y ⟨(P.child P.rootScope k m).off + (P.defs[(P.child P.rootScope k m).d]!).body.offsets[i]! + j, 0⟩ :=
      fun i hi => ho2 i (by simpa using hi)
    have hget : (P.defs[(P.child P.rootScope k m).d]!).body.ret.toArray.getD l default =
        (P.defs[(P.child P.rootScope k m).d]!).body.ret.get l := by
      rw [hps]; exact ofParts_getD ps (by rw [← hps]; exact hlc)
    rw [hget, refVal_word_members (W := fun q => portWord Y ⟨q, 0⟩) (Q := fun q => q < (Flock.OfProgram.ofProgram P).N)
      (M := fun i => ((P.defs[(P.child P.rootScope k m).d]!).body.nodes[i]!).members) hO' hNres hx' hRef hretq hqN]
    rw [portWord_lt Y hqN, portWord_lt Y hqN]

/-- **`UnitWordComputes` for a template of word-primitive calls**, on `Batch0Typed`: where C's gates
are located and unit `g` resolves. -/
theorem unitWordComputes_calls (hT : Batch0Typed) {bytes : ByteArray} {P : Prog} (hd : decode bytes = .ok P)
    (hr : (P.defs[P.root]!).body.recut = #[])
    (hG : ∀ h < InferenceOnlyVerifier.Properties.gates P, InferenceOnlyVerifier.Properties.GateOk P h)
    {g k m : ℕ} (hu : Spot.BlockFront.unitAt P g = .ok (k, m)) (hR : UnitResolves P g)
    (hco : CallsOnly P (P.child P.rootScope k m).d) (hgb : GateBodies P (P.child P.rootScope k m).d) (Y : (Flock.OfProgram.ofProgram P).Trace)
    (hY : ∀ h : Fin (Flock.OfProgram.ofProgram P).N, InUnit P k m h.val → Flock.OfProgram.inputAt P h = false →
      Y h = (Flock.OfProgram.ofProgram P).op h Y) :
    outWords Y g = wordEval P (P.child P.rootScope k m).d (argWords Y g) :=
  unitWordComputes_batch0 hT hd hr hG hu hR (batch0_of_calls hco) hgb Y hY

/-- A word primitive's value on bits is its Boolean Definition's, where that evaluates. -/
theorem evalBool_word {bytes : ByteArray} {P : Prog} (h : decode bytes = .ok P)
    (hr : (P.defs[P.root]!).body.recut = #[]) {f b : ℕ} (hf : f < P.defs.size)
    (hw : isWordPrim (P.defs[f]!).prim (P.defs[f]!).key = true) (hb : P.bodyOf? (P.defs[f]!).key = some b)
    {y v : List Bool} (hv : Flock.Program.evalDef? P (b + 1) b y = some v) : Prog.evalBool P f y = v := by
  obtain ⟨⟨hmap, hkeys⟩, hids⟩ := FlockSoundness.DecodeBoolean.decode_facts bytes P h hr
  have hm : P.BoolMapOk := ⟨hmap, hkeys⟩
  have hbl := body_lt hm hb
  have hbd : P.BoolDef b := by
    obtain ⟨b', -, hbo, hbody', -⟩ := hmap f hf hw
    obtain rfl : b' = b := Option.some.inj (hbo.symm.trans hb)
    exact hbody'
  have hB := FlockSoundness.DecodeBoolean.bool_none hkeys
  have hprim : (P.defs[f]!).prim = true := by
    simp only [isWordPrim, Bool.and_eq_true] at hw
    exact hw.1.1
  obtain ⟨hid, hmis, -⟩ := hids f hf hprim
  obtain ⟨hin, hbp⟩ := FlockSoundness.DecodeBoolean.word_ids hw
  have hg : Flock.Lower.gateOf? (P.defs[f]!).id = none := by
    rw [hid]; exact FlockSoundness.DecodeBoolean.gateOf_none hbp hmis hin
  have hdf : P.defs[f]? = some (P.defs[f]!) := by rw [getElem!_pos P.defs f hf]; exact getElem?_pos P.defs f hf
  have hP : Flock.Program.evalDef? P (2 * P.defs.size + 2) f y = some v := by
    rw [show 2 * P.defs.size + 2 = (2 * P.defs.size + 1) + 1 from rfl]
    unfold Flock.Program.evalDef?
    unfold Flock.Program.walkStep
    simp only [Flock.Program.Wires.bool, hdf, Option.bind_eq_bind, Option.bind_some, hprim, ↓reduceIte, hg, hb]
    rw [evalDef_fuel_at P hB hbd (by omega) y, hv]
    try rfl
  unfold Prog.evalBool
  rw [hP]
  rfl

theorem leaves_bitType_value (w : ℕ) : (Ty.value w).bitType.leaves = w := by
  unfold Ty.bitType
  split
  · rename_i h
    simp only [beq_iff_eq] at h
    subst h
    simp [Ty.leaves]
  · simp [Ty.leaves]

/-- A word primitive whose Boolean Definitions evaluate through gates alone has values of its return's width. -/
theorem evalBool_word_length {bytes : ByteArray} {P : Prog} (h : decode bytes = .ok P)
    (hr : (P.defs[P.root]!).body.recut = #[]) {f b w : ℕ} (hf : f < P.defs.size)
    (hw : isWordPrim (P.defs[f]!).prim (P.defs[f]!).key = true) (hret : (P.defs[f]!).ret = .value w)
    (hb : P.bodyOf? (P.defs[f]!).key = some b) (hG : GateDef P b) (y : List Bool) :
    (Prog.evalBool P f y).length = w := by
  obtain ⟨st, hP, hinv, -⟩ := BodiesOk.decode_spec bytes P h hr
  have hok : ∀ i < P.defs.size, BodiesOk.DefOk P.defs i P.defs[i]! := by rw [hP]; exact hinv.1
  obtain ⟨⟨hmap, hkeys⟩, -⟩ := FlockSoundness.DecodeBoolean.decode_facts bytes P h hr
  have hm : P.BoolMapOk := ⟨hmap, hkeys⟩
  have hbl := body_lt hm hb
  have hrb : (P.defs[b]!).ret = (Ty.value w).bitType := by
    obtain ⟨b', -, hbo, -, -, -, hbret⟩ := hmap f hf hw
    obtain rfl : b' = b := Option.some.inj (hbo.symm.trans hb)
    rw [hbret, hret]
  obtain ⟨v, hv⟩ := Option.isSome_iff_exists.1 (gateDef_total P b hbl hG y)
  rw [evalBool_word h hr hf hw hb hv]
  have hbd : P.defs[b]? = some (P.defs[b]!) := by rw [getElem!_pos P.defs b hbl]; exact getElem?_pos P.defs b hbl
  cases hpb : (P.defs[b]!).prim
  · rw [FlockSoundness.EvalBool.evalDef?_length P hbd hpb hv, ((hok b hbl).1.2 hpb).2.1, hrb, leaves_bitType_value]
  · obtain ⟨-, -, -, w', hw'⟩ := (hok b hbl).1.1 hpb
    have hw1 : w = 1 := by
      rw [hw'] at hrb
      unfold Ty.bitType at hrb
      split at hrb
      · rename_i h1
        simpa using h1
      · cases hrb
    cases hG with
    | prim _ hgt =>
      obtain ⟨g, hg⟩ := Option.isSome_iff_exists.1 hgt
      rw [Flock.Program.evalDef?.eq_2] at hv
      simp [Flock.Program.walkStep, Flock.Program.Wires.bool, hbd, hpb, hg] at hv
      rw [← hv, hw1]
      rfl
    | comp hp _ _ => rw [hpb] at hp; cases hp

/-- A node calling a word primitive outputs its members' words of the primitive's value's width. -/
theorem outWidths_word {bytes : ByteArray} {P : Prog} (h : decode bytes = .ok P)
    (hr : (P.defs[P.root]!).body.recut = #[]) {d : ℕ} (hd : d < P.defs.size) (hgb : GateBodies P d) {k : ℕ}
    (hk : k < (P.defs[d]!).body.nodes.size)
    (hform : ((P.defs[d]!).body.nodes[k]).form = .prim ∨ ((P.defs[d]!).body.nodes[k]).form = .call ∨
      ((P.defs[d]!).body.nodes[k]).form = .batch)
    (hwp : isWordPrim (P.defs[((P.defs[d]!).body.nodes[k]).fn]!).prim
      (P.defs[((P.defs[d]!).body.nodes[k]).fn]!).key = true)
    (ys : List (List Bool)) (hys : ys.length = ((P.defs[d]!).body.nodes[k]).members) :
    ((P.defs[d]!).body.nodes[k]).out.widths.toList =
      ys.map fun y => (Prog.evalBool P ((P.defs[d]!).body.nodes[k]).fn y).length := by
  obtain ⟨st, hP, hinv, -⟩ := BodiesOk.decode_spec bytes P h hr
  have hok : ∀ i < P.defs.size, BodiesOk.DefOk P.defs i P.defs[i]! := by rw [hP]; exact hinv.1
  have hnd : (P.defs[d]!).body.nodes[k] ∈ (P.defs[d]!).body.nodes := Array.getElem_mem hk
  obtain ⟨hlt, hno, -⟩ := (hok d hd).2.1 k hk
  rw [getElem!_pos _ k hk] at hlt hno
  have hGb := hgb _ hnd
  generalize (P.defs[d]!).body.nodes[k] = X at hnd hlt hno hys hGb hform hwp ⊢
  have hfP : X.fn < P.defs.size := by omega
  have hprim : (P.defs[X.fn]!).prim = true := by
    simp only [isWordPrim, Bool.and_eq_true] at hwp
    exact hwp.1.1
  obtain ⟨-, -, -, w, hwv⟩ := (hok X.fn hfP).1.1 hprim
  obtain ⟨⟨hmap, hkeys⟩, -⟩ := FlockSoundness.DecodeBoolean.decode_facts bytes P h hr
  obtain ⟨b, -, hbo, -⟩ := hmap X.fn hfP hwp
  have hlen := evalBool_word_length h hr hfP hwp hwv hbo (hGb b hbo)
  have hmap' : ys.map (fun y => (Prog.evalBool P X.fn y).length) = List.replicate ys.length w := by
    rw [List.map_congr_left (fun y _ => hlen y), List.map_const']
  rw [hmap', hys]
  have hsh := hno.2.2.2.2.1
  rcases hform with hf | hf | hf
  · rw [hf] at hsh
    obtain ⟨ho, -⟩ := hsh
    rw [ho, hwv, show (Ty.value w).widths.toList = [w] by simp [Ty.widths]]
    simp only [Node.members, hf]
    rfl
  · rw [hf] at hsh
    obtain ⟨ho, -⟩ := hsh
    rw [ho, hwv, show (Ty.value w).widths.toList = [w] by simp [Ty.widths]]
    simp only [Node.members, hf]
    rfl
  · rw [hf] at hsh
    obtain ⟨ho, -⟩ := hsh
    rw [ho, hwv, widths_array_value]
    simp only [Node.members, hf]
    rfl

/-- A node calling a gate callee outputs one bit for each member. -/
theorem outWidths_gate {bytes : ByteArray} {P : Prog} (h : decode bytes = .ok P)
    (hr : (P.defs[P.root]!).body.recut = #[]) {d : ℕ} (hd : d < P.defs.size) {k : ℕ}
    (hk : k < (P.defs[d]!).body.nodes.size)
    (hform : ((P.defs[d]!).body.nodes[k]).form = .prim ∨ ((P.defs[d]!).body.nodes[k]).form = .call ∨
      ((P.defs[d]!).body.nodes[k]).form = .batch)
    (hgc : GateCallee P ((P.defs[d]!).body.nodes[k]).fn)
    (ys : List (List Bool)) (hys : ys.length = ((P.defs[d]!).body.nodes[k]).members) :
    ((P.defs[d]!).body.nodes[k]).out.widths.toList =
      ys.map fun y => (Prog.evalBool P ((P.defs[d]!).body.nodes[k]).fn y).length := by
  obtain ⟨st, hP, hinv, -⟩ := BodiesOk.decode_spec bytes P h hr
  have hok : ∀ i < P.defs.size, BodiesOk.DefOk P.defs i P.defs[i]! := by rw [hP]; exact hinv.1
  obtain ⟨hlt, hno, -⟩ := (hok d hd).2.1 k hk
  rw [getElem!_pos _ k hk] at hlt hno
  generalize (P.defs[d]!).body.nodes[k] = X at hlt hno hys hform hgc ⊢
  have hfP : X.fn < P.defs.size := by omega
  obtain ⟨-, -, hret, -, g, hgo⟩ := gateCallee_facts h hr hfP hgc
  have hmap' : ys.map (fun y => (Prog.evalBool P X.fn y).length) = List.replicate ys.length 1 := by
    simp only [evalBool_gate hfP hgc.1 hgo, List.length_singleton, List.map_const']
  rw [hmap', hys]
  have hsh := hno.2.2.2.2.1
  rcases hform with hf | hf | hf
  · rw [hf] at hsh
    obtain ⟨ho, -⟩ := hsh
    rw [ho, hret, show (Ty.value 1).widths.toList = [1] by simp [Ty.widths]]
    simp only [Node.members, hf]
    rfl
  · rw [hf] at hsh
    obtain ⟨ho, -⟩ := hsh
    rw [ho, hret, show (Ty.value 1).widths.toList = [1] by simp [Ty.widths]]
    simp only [Node.members, hf]
    rfl
  · rw [hf] at hsh
    obtain ⟨ho, -⟩ := hsh
    rw [ho, hret, widths_array_value]
    simp only [Node.members, hf]
    rfl

/-- **`Batch0OutWidths`** holds. -/
theorem batch0OutWidths : Batch0OutWidths := fun _ _ h hr _ hd hb hgb _ hk ys hys =>
  outWidths_word h hr hd hgb hk (Or.imp_right (Or.imp_right And.left) (hb.2 _ (Array.getElem_mem hk)).1) (hb.2 _ (Array.getElem_mem hk)).2 ys hys

/-- A composite's return's references name leaves of its return type's widths. -/
theorem ret_widths {bytes : ByteArray} {P : Prog} (h : decode bytes = .ok P) (hr : (P.defs[P.root]!).body.recut = #[])
    {d : ℕ} (hd : d < P.defs.size) (hp : (P.defs[d]!).prim = false) :
    (P.defs[d]!).body.ret.toArray.toList.map (fun r => (tyAt (P.defs[d]!) r).widths[r.j]!) =
      (P.defs[d]!).ret.widths.toList := by
  obtain ⟨st, hP, hinv, -⟩ := BodiesOk.decode_spec bytes P h hr
  have hok : ∀ i < P.defs.size, BodiesOk.DefOk P.defs i P.defs[i]! := by rw [hP]; exact hinv.1
  obtain ⟨hcal, -, -, psr, hpsr, -⟩ := hok d hd
  obtain ⟨hpar, hcnt, -⟩ := hcal.2 hp
  obtain ⟨hrt, -, hW⟩ := LeafWidths.decoded_wide bytes P h hr d hd hp
  rw [hrt] at hW
  exact arg_widths (a := { (default : Arg) with ty := (P.defs[d]!).ret, seq := (P.defs[d]!).body.ret })
    (k := (P.defs[d]!).body.nodes.size) hpar hpsr hcnt (refs_comp h hr hd hp).2.2 hW

/-- A node calling a composite outputs its members' copies of the composite's return. -/
theorem outWidths_comp {bytes : ByteArray} {P : Prog} (h : decode bytes = .ok P)
    (hr : (P.defs[P.root]!).body.recut = #[]) {d : ℕ} (hd : d < P.defs.size) {k : ℕ}
    (hk : k < (P.defs[d]!).body.nodes.size) (hpc : (P.defs[((P.defs[d]!).body.nodes[k]).fn]!).prim = false)
    (hform : ((P.defs[d]!).body.nodes[k]).form = .prim ∨ ((P.defs[d]!).body.nodes[k]).form = .call ∨
      ((P.defs[d]!).body.nodes[k]).form = .batch) :
    ((P.defs[d]!).body.nodes[k]).out.widths.toList =
      (List.replicate ((P.defs[d]!).body.nodes[k]).members
        (P.defs[((P.defs[d]!).body.nodes[k]).fn]!).ret.widths.toList).flatten := by
  obtain ⟨st, hP, hinv, -⟩ := BodiesOk.decode_spec bytes P h hr
  have hok : ∀ i < P.defs.size, BodiesOk.DefOk P.defs i P.defs[i]! := by rw [hP]; exact hinv.1
  obtain ⟨-, hno, -⟩ := (hok d hd).2.1 k hk
  rw [getElem!_pos _ k hk] at hno
  generalize (P.defs[d]!).body.nodes[k] = X at hno hform hpc ⊢
  have hsh := hno.2.2.2.2.1
  rcases hform with hf | hf | hf
  · rw [hno.2.2.1 hf] at hpc; cases hpc
  · rw [hf] at hsh
    rw [hsh.1, members_one (Or.inr hf)]
    simp
  · rw [hf] at hsh
    rw [hsh.1, widths_array_flat, members_batch hf]

/-- A call to a composite passes its parameters' leaves, each at its width. -/
theorem callArgs_widths {bytes : ByteArray} {P : Prog} (h : decode bytes = .ok P)
    (hr : (P.defs[P.root]!).body.recut = #[]) {d : ℕ} (hd : d < P.defs.size) (hp : (P.defs[d]!).prim = false) {k : ℕ}
    (hk : k < (P.defs[d]!).body.nodes.size)
    (hf : ((P.defs[d]!).body.nodes[k]).form = .prim ∨ ((P.defs[d]!).body.nodes[k]).form = .call) :
    (((P.defs[d]!).body.nodes[k]).args.toList.flatMap fun a => a.seq.toArray.toList).map
        (fun r => (tyAt (P.defs[d]!) r).widths[r.j]!) =
      leafWidths (P.defs[((P.defs[d]!).body.nodes[k]).fn]!) := by
  obtain ⟨st, hP, hinv, -⟩ := BodiesOk.decode_spec bytes P h hr
  have hok : ∀ i < P.defs.size, BodiesOk.DefOk P.defs i P.defs[i]! := by rw [hP]; exact hinv.1
  obtain ⟨hcal, hnodes, -⟩ := hok d hd
  have hpar := (hcal.2 hp).1
  have hR := (refs_comp h hr hd hp).1 k hk
  have hW := ((LeafWidths.decoded_wide bytes P h hr d hd hp).2).1 k hk
  obtain ⟨-, hno, hargs⟩ := hnodes k hk
  have hk' : (P.defs[d]!).body.nodes[k] = (P.defs[d]!).body.nodes[k]! := (getElem!_pos _ k hk).symm
  rw [hk'] at hR hf ⊢
  generalize (P.defs[d]!).body.nodes[k]! = x at hno hargs hR hW hf ⊢
  obtain ⟨hsz, hcnt, -, -, hsh, -⟩ := hno
  have hty : ∀ i < x.args.size, (x.args[i]!).ty = ((P.defs[x.fn]!).params[i]!).2 := by
    rcases hf with hf | hf <;> (rw [hf] at hsh; exact hsh.2)
  rw [List.map_flatMap, flatMap_flatten_map, ← range_getD_map, leafWidths_range, flatMap_flatten_map, ← hsz]
  congr 1
  apply List.map_congr_left
  intro i hi
  have hi' : i < x.args.size := List.mem_range.1 hi
  simp only [← Array.getElem!_eq_getD]
  obtain ⟨⟨ps, hps, -⟩, -⟩ := hargs i hi'
  rw [← hty i hi']
  exact arg_widths hpar hps (hcnt i hi')
    (hR _ (by rw [getElem!_pos x.args i hi']; exact Array.getElem_mem_toList hi')).2 (hW i hi')

/-- A batch over a composite passes each argument it does not map whole, as its parameter, each leaf at its width. -/
theorem batchArgs_widths {bytes : ByteArray} {P : Prog} (h : decode bytes = .ok P)
    (hr : (P.defs[P.root]!).body.recut = #[]) {d : ℕ} (hd : d < P.defs.size) (hp : (P.defs[d]!).prim = false) {k : ℕ}
    (hk : k < (P.defs[d]!).body.nodes.size) (hbat : ((P.defs[d]!).body.nodes[k]).form = .batch) :
    ((P.defs[d]!).body.nodes[k]).args.size = (P.defs[((P.defs[d]!).body.nodes[k]).fn]!).params.size ∧
    ∀ i < ((P.defs[d]!).body.nodes[k]).args.size, (∀ ax, ((P.defs[d]!).body.nodes[k]).axes[i]? ≠ some (some ax)) →
      (((P.defs[d]!).body.nodes[k]).args.getD i default).seq.toArray.toList.map
          (fun r => (tyAt (P.defs[d]!) r).widths[r.j]!) =
        ((P.defs[((P.defs[d]!).body.nodes[k]).fn]!).params.getD i default).2.widths.toList := by
  obtain ⟨st, hP, hinv, -⟩ := BodiesOk.decode_spec bytes P h hr
  have hok : ∀ i < P.defs.size, BodiesOk.DefOk P.defs i P.defs[i]! := by rw [hP]; exact hinv.1
  obtain ⟨hcal, hnodes, -⟩ := hok d hd
  have hpar := (hcal.2 hp).1
  have hR := (refs_comp h hr hd hp).1 k hk
  have hW := ((LeafWidths.decoded_wide bytes P h hr d hd hp).2).1 k hk
  obtain ⟨-, hno, hargs⟩ := hnodes k hk
  have hk' : (P.defs[d]!).body.nodes[k] = (P.defs[d]!).body.nodes[k]! := (getElem!_pos _ k hk).symm
  rw [hk'] at hR hbat ⊢
  generalize (P.defs[d]!).body.nodes[k]! = x at hno hargs hR hW hbat ⊢
  obtain ⟨hsz, hcnt, -, -, -, -, hax⟩ := hno
  obtain ⟨haxs, hall⟩ := hax hbat
  refine ⟨hsz, fun i hi hn => ?_⟩
  simp only [← Array.getElem!_eq_getD]
  obtain ⟨hnone, -⟩ := hall i hi
  have hty : (x.args[i]!).ty = ((P.defs[x.fn]!).params[i]!).2 := by
    cases hx : x.axes[i]? with
    | none => rw [Array.getElem?_eq_none_iff] at hx; omega
    | some o =>
      cases o with
      | none => exact (hnone hx).2
      | some ax => exact absurd hx (hn ax)
  obtain ⟨⟨ps, hps, -⟩, -⟩ := hargs i hi
  rw [← hty]
  exact arg_widths hpar hps (hcnt i hi)
    (hR _ (by rw [getElem!_pos x.args i hi]; exact Array.getElem_mem_toList hi)).2 (hW i hi)

theorem form_of_ne_scan {f : Flock.Program.Form} (h : f ≠ .scan) : f = .prim ∨ f = .call ∨ f = .batch := by
  cases f with
  | prim => exact Or.inl rfl
  | call => exact Or.inr (Or.inl rfl)
  | batch => exact Or.inr (Or.inr rfl)
  | scan => exact absurd rfl h

/-- A decoded node calling a primitive is no scan. -/
theorem ne_scan_of_prim {bytes : ByteArray} {P : Prog} (h : decode bytes = .ok P)
    (hr : (P.defs[P.root]!).body.recut = #[]) {d : ℕ} (hd : d < P.defs.size) {k : ℕ}
    (hk : k < (P.defs[d]!).body.nodes.size) (hpt : (P.defs[((P.defs[d]!).body.nodes[k]).fn]!).prim = true) :
    ((P.defs[d]!).body.nodes[k]).form ≠ .scan := by
  obtain ⟨st, hP, hinv, -⟩ := BodiesOk.decode_spec bytes P h hr
  have hok : ∀ i < P.defs.size, BodiesOk.DefOk P.defs i P.defs[i]! := by rw [hP]; exact hinv.1
  obtain ⟨-, hno, -⟩ := (hok d hd).2.1 k hk
  rw [getElem!_pos _ k hk] at hno
  intro hs
  rw [hno.2.2.2.1 (Or.inr hs)] at hpt
  cases hpt

theorem widths_pair (a b : Ty) : (Ty.tuple #[a, b]).widths.toList = a.widths.toList ++ b.widths.toList := by
  simp only [Ty.widths]
  rw [← Array.foldl_toList]
  simp

/-- **A decoded scan's typing**: it calls a composite of one parameter per argument, at least one; its carry argument
has the carry parameter's widths, a sequence argument its parameter's `n` times and any other its parameter's; the
callee returns the carry and then `yw`, and the node outputs the carry and then `yw` `n` times. -/
theorem scanFacts {bytes : ByteArray} {P : Prog} (h : decode bytes = .ok P)
    (hr : (P.defs[P.root]!).body.recut = #[]) {d : ℕ} (hd : d < P.defs.size) (hp : (P.defs[d]!).prim = false) {k : ℕ}
    (hk : k < (P.defs[d]!).body.nodes.size) (hs : ((P.defs[d]!).body.nodes[k]).form = .scan) :
    (P.defs[((P.defs[d]!).body.nodes[k]).fn]!).prim = false ∧
    ((P.defs[d]!).body.nodes[k]).args.size = (P.defs[((P.defs[d]!).body.nodes[k]).fn]!).params.size ∧
    0 < (P.defs[((P.defs[d]!).body.nodes[k]).fn]!).params.size ∧
    (((P.defs[d]!).body.nodes[k]).args.getD 0 default).seq.toArray.toList.map
        (fun r => (tyAt (P.defs[d]!) r).widths[r.j]!) =
      ((P.defs[((P.defs[d]!).body.nodes[k]).fn]!).params.getD 0 default).2.widths.toList ∧
    (∀ i < ((P.defs[d]!).body.nodes[k]).args.size, 1 ≤ i →
      (((P.defs[d]!).body.nodes[k]).args.getD i default).seq.toArray.toList.map
          (fun r => (tyAt (P.defs[d]!) r).widths[r.j]!) =
        if i ≤ ((P.defs[d]!).body.nodes[k]).nxs then
          (List.replicate ((P.defs[d]!).body.nodes[k]).n
            ((P.defs[((P.defs[d]!).body.nodes[k]).fn]!).params.getD i default).2.widths.toList).flatten
        else ((P.defs[((P.defs[d]!).body.nodes[k]).fn]!).params.getD i default).2.widths.toList) ∧
    ∃ yw, (P.defs[((P.defs[d]!).body.nodes[k]).fn]!).ret.widths.toList =
        ((P.defs[((P.defs[d]!).body.nodes[k]).fn]!).params.getD 0 default).2.widths.toList ++ yw ∧
      ((P.defs[d]!).body.nodes[k]).out.widths.toList =
        ((P.defs[((P.defs[d]!).body.nodes[k]).fn]!).params.getD 0 default).2.widths.toList ++
          (List.replicate ((P.defs[d]!).body.nodes[k]).n yw).flatten := by
  obtain ⟨st, hP, hinv, -⟩ := BodiesOk.decode_spec bytes P h hr
  have hok : ∀ i < P.defs.size, BodiesOk.DefOk P.defs i P.defs[i]! := by rw [hP]; exact hinv.1
  obtain ⟨hcal, hnodes, -⟩ := hok d hd
  have hpar := (hcal.2 hp).1
  have hR := (refs_comp h hr hd hp).1 k hk
  have hW := ((LeafWidths.decoded_wide bytes P h hr d hd hp).2).1 k hk
  obtain ⟨-, hno, hargs⟩ := hnodes k hk
  have hk' : (P.defs[d]!).body.nodes[k] = (P.defs[d]!).body.nodes[k]! := (getElem!_pos _ k hk).symm
  rw [hk'] at hR hs ⊢
  generalize (P.defs[d]!).body.nodes[k]! = x at hno hargs hR hW hs ⊢
  obtain ⟨hsz, hcnt, -, hpc, hsh, -⟩ := hno
  rw [hs] at hsh
  obtain ⟨carry, yy, hret, hout, hp0, ha0, hrest, hpos⟩ := hsh
  have hwid : ∀ i < x.args.size, (x.args.getD i default).seq.toArray.toList.map
      (fun r => (tyAt (P.defs[d]!) r).widths[r.j]!) = (x.args[i]!).ty.widths.toList := fun i hi => by
    simp only [← Array.getElem!_eq_getD]
    obtain ⟨⟨ps, hps, -⟩, -⟩ := hargs i hi
    exact arg_widths hpar hps (hcnt i hi)
      (hR _ (by rw [getElem!_pos x.args i hi]; exact Array.getElem_mem_toList hi)).2 (hW i hi)
  have hc0 : ((P.defs[x.fn]!).params.getD 0 default).2 = carry := by
    rw [← Array.getElem!_eq_getD]; exact hp0
  refine ⟨hpc (Or.inr hs), hsz, by omega, ?_, fun i hi hi1 => ?_, yy.widths.toList, ?_, ?_⟩
  · rw [hwid 0 hpos, ha0, hc0]
  · rw [hwid i hi]
    simp only [← Array.getElem!_eq_getD]
    by_cases hx : i ≤ x.nxs
    · rw [if_pos hx]
      obtain ⟨e, hty, he, -⟩ := (hrest i hi (by omega)).1 hx
      rw [hty, widths_array_flat, he]
    · rw [if_neg hx, (hrest i hi (by omega)).2 hx]
  · rw [hret, widths_pair, hc0]
  · rw [hout, widths_pair, widths_array_flat, hc0]

/-- **`NestedTyped`** holds: each fact is the decoder's (`BodiesOk.DefOk`, `LeafWidths.decoded_wide`), and a node's
word-primitive callee's Boolean body evaluates through gates alone (`GateBodiesDeep` at the template). -/
theorem nestedTyped : NestedTyped := by
  intro bytes P h hr d hd hn hgd
  cases hn with
  | mk hp _ _ hgate _ =>
  obtain ⟨st, hP, hinv, -⟩ := BodiesOk.decode_spec bytes P h hr
  have hok : ∀ i < P.defs.size, BodiesOk.DefOk P.defs i P.defs[i]! := by rw [hP]; exact hinv.1
  refine ⟨(refs_comp h hr hd hp).1, (refs_comp h hr hd hp).2, ret_widths h hr hd hp, fun nd hnd => ?_,
    fun k hk hpt ys hys => ?_, fun k hk hpc hs => outWidths_comp h hr hd hk hpc (form_of_ne_scan hs),
    fun k hk hb i hi => ⟨axes_comp_ax h hr hd hp k hk hb i hi, (axes_comp h hr hd hp k hk hb i hi).2⟩,
    fun k hk _ hf => callArgs_widths h hr hd hp hk hf,
    fun k hk _ hb => batchArgs_widths h hr hd hp hk hb,
    fun k hk hs => scanFacts h hr hd hp hk hs⟩
  · obtain ⟨k, hk, rfl⟩ := Array.mem_iff_getElem.1 hnd
    rw [← getElem!_pos _ k hk]
    exact ((hok d hd).2.1 k hk).1
  · have hf3 := form_of_ne_scan (ne_scan_of_prim h hr hd hk hpt)
    by_cases hwp : isWordPrim (P.defs[((P.defs[d]!).body.nodes[k]).fn]!).prim
        (P.defs[((P.defs[d]!).body.nodes[k]).fn]!).key = true
    · exact outWidths_word h hr hd (hgd d .root hp) hk hf3 hwp ys hys
    · exact outWidths_gate h hr hd hk hf3 (hgate _ (Array.getElem_mem hk) hpt (by simpa using hwp)) ys hys

/-- **`LiftWordEval` for a template of batches on any axis and scans over word primitives, gates and earlier such templates**: the lift of a
template of a C `decode` accepts whose root has no recut, each of whose reached Boolean Definitions evaluates through gates
alone, computes its words. -/
theorem liftWordEval_nested :
    ∀ (bytes : ByteArray) (P : Prog) (units : Array ℕ) (S : Prog) (d : ℕ) (L : Prog) (e : ℕ),
      decode bytes = .ok P → (P.defs[P.root]!).body.recut = #[] → Spot.Restrict.restrict P units = .ok S →
      d < P.defs.size → ScanNested P d → GateBodiesDeep P d → Flock.Lift.lift S d = .ok (L, e) →
      ∀ x : List (List Bool), x.map List.length = leafWidths (P.defs[d]!) →
        Prog.evalBool L e x.flatten = (wordEval P d x).flatten :=
  liftWordEval_scanNested_of nestedTyped (gateWords_of nestedTyped)

/-- **`Batch0Typed`** holds: a template of word-primitive calls and axis-0 batches in a C `decode` accepts whose root has
no recut, whose nodes' Boolean bodies evaluate through gates alone, has its references, callees, output widths,
evaluations and axes typed. -/
theorem batch0Typed : Batch0Typed :=
  batch0Typed_of batch0Refs (batch0Calls_of batch0Callees batch0OutWidths batch0BoolEval batch0Axes)

/-- The walk's node values so far in activation `sc`, `outs`: one per node before `n`, each leaf of node `i` the word at
the gate it resolves to. -/
def ScopeOuts (P : Prog) (Y : (Flock.OfProgram.ofProgram P).Trace) (sc : Scope) (n : ℕ)
    (outs : Array (List (List Bool))) : Prop :=
  outs.size = n ∧ ∀ i < n, ∀ j < ((P.defs[sc.d]!).body.nodes[i]!).out.leaves, ∀ q, resolve P sc ⟨false, i, j⟩ = .ok q →
    q < (Flock.OfProgram.ofProgram P).N → (outs.getD i []).getD j default = portWord Y ⟨q, 0⟩

theorem scopeOuts_push {P : Prog} {Y : (Flock.OfProgram.ofProgram P).Trace} {sc : Scope} {n : ℕ}
    {outs : Array (List (List Bool))} (h : ScopeOuts P Y sc n outs) {w : List (List Bool)}
    (hw : ∀ j < ((P.defs[sc.d]!).body.nodes[n]!).out.leaves, ∀ q, resolve P sc ⟨false, n, j⟩ = .ok q → q < (Flock.OfProgram.ofProgram P).N →
      w.getD j default = portWord Y ⟨q, 0⟩) :
    ScopeOuts P Y sc (n + 1) (outs.push w) := by
  refine ⟨by simp [h.1], fun i hi j hj q hq hqN => ?_⟩
  by_cases hin : i < n
  · have e : (outs.push w).getD i [] = outs.getD i [] := by
      simp [Array.getD, Array.getElem_push, h.1, hin, show i < n + 1 by omega]
    rw [e]; exact h.2 i hin j hj q hq hqN
  · obtain rfl : i = n := by omega
    have e : (outs.push w).getD i [] = w := by simp [Array.getD, Array.getElem_push, h.1]
    rw [e]; exact hw j hj q hq hqN

/-- **A reference's word in an activation's walk**, where each earlier node's leaves and each parameter leaf hold the
words at the gates they resolve to. -/
theorem refVal_word_res {P : Prog} {Y : (Flock.OfProgram.ofProgram P).Trace} {sc : Scope} {x : List (List Bool)}
    {n : ℕ} {outs : Array (List (List Bool))} (hO : ScopeOuts P Y sc n outs)
    (hx : ∀ i j q, i < (P.defs[sc.d]!).body.params.size → j < ((P.defs[sc.d]!).body.params[i]!).2.leaves →
      resolve P sc ⟨true, i, j⟩ = .ok q → q < (Flock.OfProgram.ofProgram P).N →
        x.getD ((Flock.Program.paramStarts (P.defs[sc.d]!).params).getD i 0 + j) default = portWord Y ⟨q, 0⟩)
    {r : Ref} (hr : Resolve.RefOk P sc.d n r) {q : ℕ} (hq : resolve P sc r = .ok q) (hqN : q < (Flock.OfProgram.ofProgram P).N) :
    Flock.Program.refVal x (Flock.Program.paramStarts (P.defs[sc.d]!).params) outs r = portWord Y ⟨q, 0⟩ := by
  obtain ⟨p, i, j⟩ := r
  cases p with
  | true =>
    obtain ⟨h1, h2⟩ := hr.1 rfl
    simp only [Flock.Program.refVal, ite_true]
    exact hx i j q h1 h2 hq hqN
  | false =>
    obtain ⟨h1, h2⟩ := hr.2 rfl
    simp only [Flock.Program.refVal, Bool.false_eq_true, ite_false]
    exact hO.2 i h1 j h2 q hq hqN

theorem flatten_getD_div {α : Type} [Inhabited α] (ms : List (List α)) {L j : ℕ} (hl : ∀ l ∈ ms, l.length = L)
    (hq : j / L < ms.length) (hr : j % L < L) :
    ms.flatten.getD j default = (ms.getD (j / L) []).getD (j % L) default := by
  have hsum : ((ms.take (j / L)).map List.length).sum = j / L * L := by
    have : (ms.take (j / L)).map List.length = List.replicate (j / L) L := by
      refine List.eq_replicate_iff.2 ⟨by simp; omega, fun a ha => ?_⟩
      obtain ⟨l, hl', rfl⟩ := List.mem_map.1 ha
      exact hl l (List.mem_of_mem_take hl')
    rw [this, List.sum_replicate, smul_eq_mul]
  have hlen : j % L < (ms[j / L]).length := by rw [hl _ (List.getElem_mem hq)]; exact hr
  have h := flatten_getElem?_sum ms hq hlen
  rw [hsum, show j / L * L + j % L = j by rw [Nat.mul_comm]; exact Nat.div_add_mod j L] at h
  simp [List.getD_eq_getElem?_getD, h, List.getElem?_eq_getElem hq, List.getElem?_eq_getElem hlen]

/-- Leaf `j` of parameter `i` among the parameters' leaves, laid out one parameter after another. -/
theorem flat_params_getD {α : Type} [Inhabited α] (ps : Array (String × Ty)) (cnt : ℕ → ℕ) (g : ℕ → ℕ → α)
    (hc : ∀ i < ps.size, cnt i = (ps[i]!).2.leaves) {i j : ℕ} (hi : i < ps.size) (hj : j < (ps[i]!).2.leaves) :
    ((List.range ps.size).flatMap fun i => (List.range (cnt i)).map (g i)).getD
      ((Flock.Program.paramStarts ps).getD i 0 + j) default = g i j := by
  rw [flatMap_flatten_map]
  have hLi : i < ((List.range ps.size).map fun i => (List.range (cnt i)).map (g i)).length := by simp [hi]
  have hj' : j < (((List.range ps.size).map fun i => (List.range (cnt i)).map (g i))[i]).length := by
    simp [hc i hi, hj]
  have hpre : (Flock.Program.paramStarts ps).getD i 0 =
      ((((List.range ps.size).map fun i => (List.range (cnt i)).map (g i)).take i).map List.length).sum := by
    rw [paramStarts_getD _ hi, List.map_take, List.map_take]
    congr 2
    apply List.ext_getElem (by simp)
    intro a h1 h2
    have ha : a < ps.size := by simpa using h1
    simp only [List.getElem_map, List.getElem_range, Array.getElem_toList]
    rw [hc a ha, getElem!_pos ps a ha]
    simp
  rw [hpre, List.getD_eq_getElem?_getD, flatten_getElem?_sum _ hLi hj']
  simp

theorem cntOf_params {P : Prog} {nd : Node} (hno : BodiesOk.NodeOk (P.defs[nd.fn]!) nd)
    (hf : nd.form = .call ∨ nd.form = .batch) :
    ∀ i < nd.args.size, OperandWidths.cntOf P nd i = ((P.defs[nd.fn]!).params[i]!).2.leaves := by
  obtain ⟨-, hcnt, -, -, hshape, -⟩ := hno
  intro i hi
  unfold OperandWidths.cntOf
  split
  · rfl
  · rename_i hnb
    rw [hcnt i hi]
    rcases hf with hf | hf
    · simp only [hf, BodiesOk.Shape] at hshape
      rw [hshape.2 i hi]
    · simp only [hf, BodiesOk.Shape] at hshape
      rcases hshape.2 i hi with ⟨-, hty⟩ | ⟨ax, mt, hrem, -, hdims⟩
      · rw [hty]
      · obtain ⟨L', hax, -⟩ := BodiesOk.axis_spec ax _ _ mt hrem
        exact (hnb (by rw [hf, hdims, hax]; rfl)).elim

/-- An operand's reference of member `m'` of a call or batch node `n` of a composite, at a leaf of its callee's
parameter, names a parameter or an earlier node at a leaf. -/
theorem refOf_refOk_comp {P : Prog} {d n : ℕ} (hB : Resolve.BodyOk P d) (hn : n < (P.defs[d]!).body.nodes.size)
    (hf : ((P.defs[d]!).body.nodes[n]!).form = .call ∨ ((P.defs[d]!).body.nodes[n]!).form = .batch)
    (hbp : (P.defs[((P.defs[d]!).body.nodes[n]!).fn]!).body.params =
      (P.defs[((P.defs[d]!).body.nodes[n]!).fn]!).params)
    {m' i j : ℕ} (hi : i < (P.defs[((P.defs[d]!).body.nodes[n]!).fn]!).params.size)
    (hj : j < ((P.defs[((P.defs[d]!).body.nodes[n]!).fn]!).params[i]!).2.leaves)
    (hm' : m' < ((P.defs[d]!).body.nodes[n]!).members) :
    Resolve.RefOk P d n (OperandWidths.refOf ((P.defs[d]!).body.nodes[n]!) m' i j) := by
  have hup := (hB.1 n hn).2.2.1 i (by rw [hbp]; exact hi) j (by rw [hbp]; exact hj) m' hm'
  unfold OperandWidths.refOf
  rcases hf with hf | hf
  · rw [if_neg (by rw [hf]; decide)]
    rw [hf] at hup
    exact (hB.1 n hn).2.1 i j hup
  · rw [if_pos (by rw [hf]; rfl), Resolve.memberRef_eq]
    rw [hf] at hup
    exact (hB.1 n hn).2.1 i _ hup

/-- **A scan's fold, iteration by iteration**: where iteration `m`'s carry leaves satisfy `Cin m`, its callee returns
`c + yL` words satisfying `Out m`, and iteration `m`'s carry outputs satisfy `Cin (m + 1)`, the fold returns the last
iteration's carry outputs and, after one another, each iteration's other outputs. -/
theorem scanFold_ports {W : Type} [Inhabited W] (E : List W → Option (List W)) (rest : ℕ → List W) (c yL N : ℕ)
    (Cin Out : ℕ → ℕ → W → Prop) (init : List W) (hi0 : init.length = c)
    (hinit : ∀ j < c, Cin 0 j (init.getD j default))
    (hiter : ∀ m < N, ∀ cr : List W, cr.length = c → (∀ j < c, Cin m j (cr.getD j default)) →
      ∃ ws, E (cr ++ rest m) = some ws ∧ ws.length = c + yL ∧ ∀ l < c + yL, Out m l (ws.getD l default))
    (hlink : ∀ m, m + 1 < N → ∀ j < c, ∀ w, Out m j w → Cin (m + 1) j w) :
    ∃ cr ys, (List.range N).foldlM (fun (st : List W × List W) m => do
        let o ← E (st.1 ++ rest m)
        pure (o.take c, st.2 ++ o.drop c)) (init, []) = some (cr, ys) ∧
      cr.length = c ∧ ys.length = N * yL ∧ (0 < N → ∀ j < c, Out (N - 1) j (cr.getD j default)) ∧
      ∀ k < N, ∀ jj < yL, Out k (c + jj) (ys.getD (k * yL + jj) default) := by
  suffices h : ∀ M ≤ N, ∃ cr ys, (List.range M).foldlM (fun (st : List W × List W) m => do
        let o ← E (st.1 ++ rest m)
        pure (o.take c, st.2 ++ o.drop c)) (init, []) = some (cr, ys) ∧
      cr.length = c ∧ ys.length = M * yL ∧ (M = 0 → cr = init) ∧ (0 < M → ∀ j < c, Out (M - 1) j (cr.getD j default)) ∧
      ∀ k < M, ∀ jj < yL, Out k (c + jj) (ys.getD (k * yL + jj) default) by
    obtain ⟨cr, ys, h1, h2, h3, -, h5, h6⟩ := h N (Nat.le_refl _)
    exact ⟨cr, ys, h1, h2, h3, h5, h6⟩
  intro M
  induction M with
  | zero => exact fun _ => ⟨init, [], rfl, hi0, by simp, fun _ => rfl, fun h => absurd h (Nat.lt_irrefl _),
      fun k hk => absurd hk (Nat.not_lt_zero _)⟩
  | succ M ih =>
  intro hM
  obtain ⟨cr, ys, hf, hl, hyl, h0, hlast, hys⟩ := ih (by omega)
  have hcin : ∀ j < c, Cin M j (cr.getD j default) := by
    rcases Nat.eq_zero_or_pos M with rfl | hpos
    · rw [h0 rfl]; exact hinit
    · intro j hj
      have := hlink (M - 1) (by omega) j hj _ (hlast hpos j hj)
      rwa [show M - 1 + 1 = M by omega] at this
  obtain ⟨ws, hE, hwl, hout⟩ := hiter M (by omega) cr hl hcin
  refine ⟨ws.take c, ys ++ ws.drop c, ?_, by simp [hwl], by simp [hyl, hwl, Nat.add_mul], fun h => absurd h (by omega),
    fun _ j hj => ?_, fun k hk jj hjj => ?_⟩
  · rw [List.range_succ, List.foldlM_append, hf]
    simp [hE]
  · rw [show M + 1 - 1 = M by omega, List.getD_eq_getElem?_getD, List.getElem?_take, if_pos hj,
      ← List.getD_eq_getElem?_getD]
    exact hout j (by omega)
  · have hle : M * yL ≤ ys.length := by rw [hyl]
    by_cases hkM : k < M
    · have hlt : k * yL + jj < ys.length := by
        have := Nat.mul_le_mul_right yL (show k + 1 ≤ M by omega)
        rw [Nat.add_mul, Nat.one_mul] at this
        omega
      rw [List.getD_eq_getElem?_getD, List.getElem?_append_left hlt, ← List.getD_eq_getElem?_getD]
      exact hys k hkM jj hjj
    · have e : M = k := by omega
      subst e
      rw [List.getD_eq_getElem?_getD, List.getElem?_append_right (by omega), List.getElem?_drop, hyl,
        show c + (M * yL + jj - M * yL) = c + jj by omega, ← List.getD_eq_getElem?_getD]
      exact hout (c + jj) (by omega)

/-- Leaf `j` of parameter `i` of a scan iteration's input: the carry's, and then each other argument's in order. -/
theorem scanInput_getD {W : Type} [Inhabited W] (ps : Array (String × Ty)) (h0 : 0 < ps.size) (cr : List W)
    (hc : cr.length = (ps[0]!).2.leaves) (g : ℕ → List W) (hg : ∀ i < ps.size, (g i).length = (ps[i]!).2.leaves)
    {i j : ℕ} (hi : i < ps.size) (hj : j < (ps[i]!).2.leaves) :
    (cr ++ ((List.range ps.size).drop 1).flatMap g).getD ((Flock.Program.paramStarts ps).getD i 0 + j) default =
      if i = 0 then cr.getD j default else (g i).getD j default := by
  have hl : ∀ l : List W, (List.range l.length).map (fun t => l.getD t default) = l := fun l => by
    apply List.ext_getElem (by simp)
    intro t h1 h2
    simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h2]
  have he : cr ++ ((List.range ps.size).drop 1).flatMap g = (List.range ps.size).flatMap fun i =>
      (List.range (ps[i]!).2.leaves).map fun t => if i = 0 then cr.getD t default else (g i).getD t default := by
    obtain ⟨k, hk⟩ : ∃ k, ps.size = k + 1 := ⟨ps.size - 1, by omega⟩
    rw [hk, List.range_succ_eq_map, List.flatMap_cons, List.drop_one, List.tail_cons]
    simp only [↓reduceIte]
    rw [← hc, hl]
    congr 1
    apply List.flatMap_congr
    intro i hi'
    obtain ⟨i', hi'', rfl⟩ := List.mem_map.1 hi'
    simp only [Nat.succ_ne_zero, ↓reduceIte]
    rw [← hg (i' + 1) (by rw [List.mem_range] at hi''; omega), hl]
  rw [he]
  exact flat_params_getD ps (fun i => (ps[i]!).2.leaves)
    (fun i t => if i = 0 then cr.getD t default else (g i).getD t default) (fun _ _ => rfl) hi hj

/-- A resolution that ends takes a step. -/
theorem resolve_step_ok {P : Prog} (hrc : (P.defs[P.root]!).body.recut = #[]) {sc : Scope} {r : Ref} {g : ℕ}
    (h : resolve P sc r = .ok g) : ∃ x, resolveStep P (sc, r) = .ok x := by
  rw [Flock.Program.resolve_eq hrc] at h
  obtain ⟨n, hn⟩ : ∃ n, Flock.Steps.FUEL = n + 1 := ⟨Flock.Steps.FUEL - 1, by unfold Flock.Steps.FUEL; omega⟩
  rw [hn, Flock.Steps.run.eq_2] at h
  rcases hs : resolveStep P (sc, r) with e | x
  · simp [hs, bind, Except.bind] at h
  · exact ⟨x, rfl⟩

/-- **A scan node's words in an activation's walk**: where each iteration's callee, given at each parameter leaf the
word at the gate it resolves to, returns at each return leaf the word at the gate that leaf resolves to, the walk's step
over a scan node of a composite pushes at each output leaf the word at the gate the leaf resolves to. -/
theorem scope_scan {P : Prog} (hr : (P.defs[P.root]!).body.recut = #[]) {Y : (Flock.OfProgram.ofProgram P).Trace}
    {sc : Scope} (hBd : Resolve.BodyOk P sc.d) {n : ℕ} (hn' : n < (P.defs[sc.d]!).body.nodes.size)
    (hf : ((P.defs[sc.d]!).body.nodes[n]!).form = .scan)
    (hfnlt : ((P.defs[sc.d]!).body.nodes[n]!).fn < sc.d)
    (hcal : P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]? = some (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!))
    (hnok : BodiesOk.NodeOk (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!) ((P.defs[sc.d]!).body.nodes[n]!))
    (hseq : ∀ a ∈ ((P.defs[sc.d]!).body.nodes[n]!).args.toList, ∃ ps, a.seq = Seq.ofParts ps)
    (hcntc : (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).body.ret.count =
      (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).ret.leaves)
    {x : List (List Bool)} {outs : Array (List (List Bool))} (hO : ScopeOuts P Y sc n outs)
    (hv : ∀ r, Resolve.RefOk P sc.d n r → ∀ q, resolve P sc r = .ok q → q < (Flock.OfProgram.ofProgram P).N →
      Flock.Program.refVal x (Flock.Program.paramStarts (P.defs[sc.d]!).params) outs r = portWord Y ⟨q, 0⟩)
    (hih : ∀ m < ((P.defs[sc.d]!).body.nodes[n]!).n, ∀ xm : List (List Bool),
      (∀ i j q, i < (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).params.size →
        j < ((P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).params[i]!).2.leaves →
        resolve P (Scope.child sc ((P.defs[sc.d]!).body.nodes[n]!).fn (sc.off + (P.defs[sc.d]!).body.offsets[n]! +
          m * (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).gates) n m) ⟨true, i, j⟩ = .ok q →
        q < (Flock.OfProgram.ofProgram P).N →
        xm.getD ((Flock.Program.paramStarts (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).params).getD i 0 + j) default =
          portWord Y ⟨q, 0⟩) →
      ∃ ws, wordDef? P sc.d ((P.defs[sc.d]!).body.nodes[n]!).fn xm = some ws ∧
        ws.length = (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).body.ret.count ∧
        ∀ l < (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).body.ret.count, ∀ q,
          resolve P (Scope.child sc ((P.defs[sc.d]!).body.nodes[n]!).fn (sc.off + (P.defs[sc.d]!).body.offsets[n]! +
            m * (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).gates) n m)
            ((P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).body.ret.get l) = .ok q →
          q < (Flock.OfProgram.ofProgram P).N → ws.getD l default = portWord Y ⟨q, 0⟩) :
    ∃ b', nodeStep wordWires P (wordDef? P sc.d) sc.d x (P.defs[sc.d]!) outs ((P.defs[sc.d]!).body.nodes[n]!) = some b' ∧
      ScopeOuts P Y sc (n + 1) b' := by
  obtain ⟨hsz, hcnt, -, -, hsh, -, -⟩ := hnok
  simp only [hf, BodiesOk.Shape] at hsh
  obtain ⟨carry, yy, hret, hout, hp0, ha0, hrest, hpos⟩ := hsh
  have hc0 : ((P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).params.getD 0 default).2 = carry := by
    rw [← Array.getElem!_eq_getD]; exact hp0
  have hrefs := (hBd.1 n hn').2.1
  have hmem : ∀ i < ((P.defs[sc.d]!).body.nodes[n]!).args.size,
      ∃ ps, (((P.defs[sc.d]!).body.nodes[n]!).args[i]!).seq = Seq.ofParts ps := fun i hi =>
    hseq _ (by rw [getElem!_pos _ i hi]; exact Array.getElem_mem_toList hi)
  have had : ∀ i, ((P.defs[sc.d]!).body.nodes[n]!).args.getD i default = ((P.defs[sc.d]!).body.nodes[n]!).args[i]! :=
    fun _ => Array.getElem!_eq_getD.symm
  have hpd : ∀ i, (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).params.getD i default =
      (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).params[i]! := fun _ => Array.getElem!_eq_getD.symm
  have hval : ∀ i < ((P.defs[sc.d]!).body.nodes[n]!).args.size, ∀ t < (((P.defs[sc.d]!).body.nodes[n]!).args[i]!).seq.count,
      ∀ q, resolve P sc ((((P.defs[sc.d]!).body.nodes[n]!).args[i]!).seq.get t) = .ok q → q < (Flock.OfProgram.ofProgram P).N →
      Flock.Program.refVal x (Flock.Program.paramStarts (P.defs[sc.d]!).params) outs
        ((((P.defs[sc.d]!).body.nodes[n]!).args.getD i default).seq.toArray.getD t default) = portWord Y ⟨q, 0⟩ := by
    intro i hi t ht q hq hqN
    obtain ⟨ps, hps⟩ := hmem i hi
    rw [had i, hps, ofParts_getD ps (by rw [← hps]; exact ht), ← hps]
    exact hv _ (hrefs i t ht) q hq hqN
  rw [nodeStep_scan wordWires (fun _ => rfl) P _ _ _ _ _ _ hf hfnlt hcal, hc0]
  obtain ⟨cr, ys, hfold, hcl, hyl, hlast, hys⟩ := scanFold_ports (wordDef? P sc.d ((P.defs[sc.d]!).body.nodes[n]!).fn)
    (fun m => ((List.range ((P.defs[sc.d]!).body.nodes[n]!).args.size).drop 1).flatMap
      (scanArg (Flock.Program.refVal x (Flock.Program.paramStarts (P.defs[sc.d]!).params) outs)
        (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!) ((P.defs[sc.d]!).body.nodes[n]!) m))
    carry.leaves yy.leaves ((P.defs[sc.d]!).body.nodes[n]!).n
    (fun m j w => ∀ q, resolve P (Scope.child sc ((P.defs[sc.d]!).body.nodes[n]!).fn (sc.off + (P.defs[sc.d]!).body.offsets[n]! +
      m * (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).gates) n m) ⟨true, 0, j⟩ = .ok q →
      q < (Flock.OfProgram.ofProgram P).N → w = portWord Y ⟨q, 0⟩)
    (fun m l w => ∀ q, resolve P (Scope.child sc ((P.defs[sc.d]!).body.nodes[n]!).fn (sc.off + (P.defs[sc.d]!).body.offsets[n]! +
      m * (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).gates) n m)
      ((P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).body.ret.get l) = .ok q →
      q < (Flock.OfProgram.ofProgram P).N → w = portWord Y ⟨q, 0⟩)
    (Flock.Program.argVals (Flock.Program.refVal x (Flock.Program.paramStarts (P.defs[sc.d]!).params) outs)
      (((P.defs[sc.d]!).body.nodes[n]!).args.getD 0 default) carry.leaves id)
    (by simp [Flock.Program.argVals])
    (fun j hj q hq hqN => by
      have hs := Resolve.rs_up_scan0 P sc ((P.defs[sc.d]!).body.nodes[n]!).fn
        (sc.off + (P.defs[sc.d]!).body.offsets[n]! + 0 * (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).gates) n 0 j hf
      rw [if_pos rfl] at hs
      have h1 := resolve_step_up hr hs hq
      have hjc : j < (((P.defs[sc.d]!).body.nodes[n]!).args[0]!).seq.count := by rw [hcnt 0 hpos, ha0]; exact hj
      unfold Flock.Program.argVals
      rw [List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hj, Option.map_some, Option.getD_some]
      exact hval 0 hpos j hjc q h1 hqN)
    (fun m hm cr hcrl hcin => by
      have hlen : ∀ i < (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).params.size,
          (if i = 0 then cr else scanArg (Flock.Program.refVal x (Flock.Program.paramStarts (P.defs[sc.d]!).params) outs)
            (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!) ((P.defs[sc.d]!).body.nodes[n]!) m i).length =
          ((P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).params[i]!).2.leaves := by
        intro i hi
        by_cases i0 : i = 0
        · subst i0; rw [if_pos rfl, hcrl, hp0]
        · rw [if_neg i0]; unfold scanArg; split <;> simp only [Flock.Program.argVals, List.length_map, List.length_range, hpd i] <;>
            exact congrArg (fun p : String × Ty => p.2.leaves) (getElem!_pos (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).params i hi).symm
      have hsz' : ((P.defs[sc.d]!).body.nodes[n]!).args.size = (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).params.size := hsz
      have hpos' : 0 < (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).params.size := by omega
      have hflat : ((List.range ((P.defs[sc.d]!).body.nodes[n]!).args.size).drop 1).flatMap
          (scanArg (Flock.Program.refVal x (Flock.Program.paramStarts (P.defs[sc.d]!).params) outs)
            (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!) ((P.defs[sc.d]!).body.nodes[n]!) m) =
          ((List.range (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).params.size).drop 1).flatMap
          (fun i => if i = 0 then cr else scanArg (Flock.Program.refVal x (Flock.Program.paramStarts (P.defs[sc.d]!).params) outs)
            (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!) ((P.defs[sc.d]!).body.nodes[n]!) m i) := by
        rw [hsz']
        apply List.flatMap_congr
        intro i hi
        have : i ≠ 0 := by
          obtain ⟨k, hk⟩ : ∃ k, (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).params.size = k + 1 := ⟨_, (Nat.succ_pred_eq_of_pos hpos').symm⟩
          rw [hk, List.range_succ_eq_map, List.drop_one, List.tail_cons] at hi
          obtain ⟨i', -, rfl⟩ := List.mem_map.1 hi
          exact Nat.succ_ne_zero _
        rw [if_neg this]
      obtain ⟨ws, hws, hwl, hwr⟩ := hih m hm (cr ++ ((List.range ((P.defs[sc.d]!).body.nodes[n]!).args.size).drop 1).flatMap
          (scanArg (Flock.Program.refVal x (Flock.Program.paramStarts (P.defs[sc.d]!).params) outs)
            (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!) ((P.defs[sc.d]!).body.nodes[n]!) m)) (fun i j q hi hj hq hqN => by
        rw [hflat, scanInput_getD _ hpos' cr (by rw [hcrl, hp0]) _ hlen hi hj]
        by_cases i0 : i = 0
        · subst i0
          rw [if_pos rfl]
          exact hcin j (by rw [hp0] at hj; exact hj) q hq hqN
        · rw [if_neg i0, if_neg i0]
          have hia : i < ((P.defs[sc.d]!).body.nodes[n]!).args.size := by rw [hsz']; exact hi
          have h1 := resolve_step_up hr (Resolve.rs_up_scan P sc ((P.defs[sc.d]!).body.nodes[n]!).fn _ n m i j i0 hf) hq
          obtain ⟨hsq, hns⟩ := hrest i hia i0
          unfold scanArg Flock.Program.argVals
          rw [hpd i]
          by_cases hx : i ≤ ((P.defs[sc.d]!).body.nodes[n]!).nxs
          · obtain ⟨e, hty, he, hdims⟩ := hsq hx
            rw [if_pos hx] at h1 ⊢
            rw [hdims] at h1
            simp only [Option.map_some, Option.getD_some] at h1
            rw [List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hj, Option.map_some, Option.getD_some]
            have hidx : m * e.leaves + j < (((P.defs[sc.d]!).body.nodes[n]!).args[i]!).seq.count := by
              rw [hcnt i hia, hty, BodiesOk.leaves_array]
              have := Nat.mul_le_mul_right e.leaves (show m + 1 ≤ ((P.defs[sc.d]!).body.nodes[n]!).n by omega)
              rw [Nat.add_mul, Nat.one_mul] at this
              rw [← he] at hj
              omega
            have hv' := hval i hia _ hidx q h1 hqN
            rw [show m * e.leaves + j = j + m * e.leaves by omega, he] at hv'
            exact hv'
          · rw [if_neg hx] at h1 ⊢
            rw [List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hj, Option.map_some, Option.getD_some]
            have hidx : j < (((P.defs[sc.d]!).body.nodes[n]!).args[i]!).seq.count := by
              rw [hcnt i hia, hns hx]; exact hj
            exact hval i hia _ hidx q h1 hqN)
      refine ⟨ws, hws, by rw [hwl, hcntc, hret, BodiesOk.leaves_tuple2], fun l hl q hq hqN => hwr l ?_ q hq hqN⟩
      rw [hcntc, hret, BodiesOk.leaves_tuple2]; exact hl)
    (fun m hm j hj w hw q hq hqN => by
      have hs := Resolve.rs_up_scan0 P sc ((P.defs[sc.d]!).body.nodes[n]!).fn
        (sc.off + (P.defs[sc.d]!).body.offsets[n]! + (m + 1) * (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).gates) n (m + 1) j hf
      rw [if_neg (Nat.succ_ne_zero m), Nat.add_sub_cancel,
        show sc.off + (P.defs[sc.d]!).body.offsets[n]! + (m + 1) * (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).gates -
          (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).gates =
          sc.off + (P.defs[sc.d]!).body.offsets[n]! + m * (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).gates by
        rw [Nat.add_mul, Nat.one_mul]; omega] at hs
      exact hw q (resolve_step_up hr hs hq) hqN)
  refine ⟨outs.push (cr ++ ys), by rw [hfold]; rfl, scopeOuts_push hO ?_⟩
  intro j hj q hq hqN
  rw [hout, BodiesOk.leaves_tuple2, BodiesOk.leaves_array] at hj
  have hN0 : ((P.defs[sc.d]!).body.nodes[n]!).n ≠ 0 := by
    intro h0
    obtain ⟨_, hx⟩ := resolve_step_ok hr hq
    exact Resolve.rs_scan0 P sc n j _ hf h0 hx
  have hs := Resolve.rs_scan P sc n j carry yy hf hN0 hret
  by_cases hjc : j < carry.leaves
  · simp only [hjc, ↓reduceIte] at hs
    rw [List.getD_eq_getElem?_getD, List.getElem?_append_left (by omega), ← List.getD_eq_getElem?_getD]
    exact hlast (by omega) j hjc q (resolve_step_up hr hs hq) hqN
  · simp only [hjc, ↓reduceIte] at hs
    have hyL : 0 < yy.leaves := by
      rcases Nat.eq_zero_or_pos yy.leaves with h | h
      · rw [h, Nat.mul_zero] at hj; omega
      · exact h
    have hm := (Nat.div_lt_iff_lt_mul hyL).2 (show j - carry.leaves < ((P.defs[sc.d]!).body.nodes[n]!).n * yy.leaves by omega)
    have hdm := Nat.div_add_mod (j - carry.leaves) yy.leaves
    rw [List.getD_eq_getElem?_getD, List.getElem?_append_right (by omega), hcl,
      show j - carry.leaves = (j - carry.leaves) / yy.leaves * yy.leaves + (j - carry.leaves) % yy.leaves by
        rw [Nat.mul_comm]; omega, ← List.getD_eq_getElem?_getD]
    exact hys _ hm _ (Nat.mod_lt _ hyL) q (resolve_step_up hr hs hq) hqN

/-- **The words of an activation's walk**: in activation `sc` of a template of batches on any axis and scans over word primitives, gates and
earlier such templates, each of whose reached Boolean Definitions evaluates through gates alone, and whose gates the
trace computes, the walk on words, each the word at the gate its parameter leaf resolves to, returns at each return leaf
the word at the gate that leaf resolves to. -/
theorem scope_words {bytes : ByteArray} {P : Prog} (hd : decode bytes = .ok P)
    (hr : (P.defs[P.root]!).body.recut = #[]) (hroot : P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false)
    (hB : ∀ sc, Resolve.Good P sc → Resolve.BodyOk P sc.d) (hL : ∀ d < P.defs.size, Walk.LaidOut P d)
    (hG : ∀ h < InferenceOnlyVerifier.Properties.gates P, InferenceOnlyVerifier.Properties.GateOk P h)
    (hD : ∀ d < P.defs.size, BodiesOk.DefOk P.defs d P.defs[d]!)
    (Y : (Flock.OfProgram.ofProgram P).Trace) {lo hi : ℕ} (hiN : hi ≤ (Flock.OfProgram.ofProgram P).N)
    (hY : ∀ h : Fin (Flock.OfProgram.ofProgram P).N, lo ≤ h.val → h.val < hi → Flock.OfProgram.inputAt P h = false →
      Y h = (Flock.OfProgram.ofProgram P).op h Y) :
    ∀ d (sc : Scope), sc.d = d → Resolve.Good P sc → ScanNested P d → GateBodiesDeep P d → lo ≤ sc.off →
      sc.off + (P.defs[d]!).gates ≤ hi → ∀ x : List (List Bool),
      (∀ i j q, i < (P.defs[d]!).params.size → j < ((P.defs[d]!).params[i]!).2.leaves →
        resolve P sc ⟨true, i, j⟩ = .ok q → q < (Flock.OfProgram.ofProgram P).N →
          x.getD ((Flock.Program.paramStarts (P.defs[d]!).params).getD i 0 + j) default = portWord Y ⟨q, 0⟩) →
      ∃ ws, wordDef? P (d + 1) d x = some ws ∧ ws.length = (P.defs[d]!).body.ret.count ∧
        ∀ l < (P.defs[d]!).body.ret.count, ∀ q, resolve P sc ((P.defs[d]!).body.ret.get l) = .ok q → q < (Flock.OfProgram.ofProgram P).N →
          ws.getD l default = portWord Y ⟨q, 0⟩ := by
  intro d
  induction d using Nat.strong_induction_on with
  | _ d ih =>
  intro sc hscd hsc hN0 hgd hlo hhi x hx
  cases hN0 with
  | mk _ _ hnest hgate _ =>
  subst hscd
  obtain ⟨hdl, hp⟩ := Resolve.good_d P hroot hB sc hsc
  obtain ⟨-, hK, hnodes⟩ := hL _ hdl hp
  have hbp : (P.defs[sc.d]!).body.params = (P.defs[sc.d]!).params := ((hD _ hdl).1.2 hp).1
  have hPd : P.defs[sc.d]? = some (P.defs[sc.d]!) := by rw [getElem!_pos P.defs _ hdl]; exact getElem?_pos P.defs _ hdl
  have hgb : GateBodies P sc.d := hgd _ .root hp
  obtain ⟨⟨hmap, -⟩, -⟩ := FlockSoundness.DecodeBoolean.decode_facts bytes P hd hr
  have hx' : ∀ i j q, i < (P.defs[sc.d]!).body.params.size → j < ((P.defs[sc.d]!).body.params[i]!).2.leaves →
      resolve P sc ⟨true, i, j⟩ = .ok q → q < (Flock.OfProgram.ofProgram P).N →
        x.getD ((Flock.Program.paramStarts (P.defs[sc.d]!).params).getD i 0 + j) default = portWord Y ⟨q, 0⟩ := by
    intro i j q h1 h2
    rw [hbp] at h1 h2
    exact hx i j q h1 h2
  have hstep : ∀ n (hn : n < (P.defs[sc.d]!).body.nodes.toList.length) (outs : Array (List (List Bool))),
      ScopeOuts P Y sc n outs →
      ∃ b', nodeStep wordWires P (wordDef? P sc.d) sc.d x (P.defs[sc.d]!) outs (P.defs[sc.d]!).body.nodes.toList[n] = some b' ∧
        ScopeOuts P Y sc (n + 1) b' := by
    intro n hn outs hO
    have hn' : n < (P.defs[sc.d]!).body.nodes.size := by simpa using hn
    have hnn : (P.defs[sc.d]!).body.nodes.toList[n] = ((P.defs[sc.d]!).body.nodes[n]!) := by rw [Array.getElem_toList, getElem!_pos _ n hn']
    rw [hnn]
    have hmemn : ((P.defs[sc.d]!).body.nodes[n]!) ∈ (P.defs[sc.d]!).body.nodes := by rw [getElem!_pos _ n hn']; exact Array.getElem_mem hn'
    obtain ⟨hfnlt, hnok, hargsF⟩ := (hD _ hdl).2.1 n hn'
    have hfnP : ((P.defs[sc.d]!).body.nodes[n]!).fn < P.defs.size := by omega
    have hcal : P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]? = some (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!) := by
      rw [getElem!_pos P.defs ((P.defs[sc.d]!).body.nodes[n]!).fn hfnP]; exact getElem?_pos P.defs _ hfnP
    have hseq : ∀ a ∈ ((P.defs[sc.d]!).body.nodes[n]!).args.toList, ∃ ps, a.seq = Seq.ofParts ps := by
      intro a ha
      obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem ha
      have hi' : i < ((P.defs[sc.d]!).body.nodes[n]!).args.size := by simpa using hi
      obtain ⟨⟨ps, hps, -⟩, -⟩ := hargsF i hi'
      exact ⟨ps, by rw [Array.getElem_toList, ← getElem!_pos _ i hi']; exact hps⟩
    have hBn := (hB sc hsc).1 n hn'
    have hstep1 := hnodes n hn'
    have hend := Resolve.offsets_mono P _ (hL _ hdl) hp (n + 1) _ (by omega) (Nat.le_refl _)
    have hv : ∀ r, Resolve.RefOk P sc.d n r → ∀ q, resolve P sc r = .ok q → q < (Flock.OfProgram.ofProgram P).N →
        Flock.Program.refVal x (Flock.Program.paramStarts (P.defs[sc.d]!).params) outs r = portWord Y ⟨q, 0⟩ :=
      fun r hr' q hq hqN => refVal_word_res hO hx' hr' hq hqN
    have hcntL := fun hf => cntOf_params (P := P) (nd := ((P.defs[sc.d]!).body.nodes[n]!)) hnok hf
    have hnok' := hnok
    obtain ⟨hsz, -, hprimp, hcallp, hsh, -, -⟩ := hnok
    cases hpf : (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).prim
    · have hnc : ScanNested P ((P.defs[sc.d]!).body.nodes[n]!).fn := hnest _ hmemn hpf
      have hgdc : GateBodiesDeep P ((P.defs[sc.d]!).body.nodes[n]!).fn := fun y hy => hgd y (reach_trans (.callee .root hp hmemn) hy)
      obtain ⟨hbpc, hcntc, holc⟩ := (hD _ hfnP).1.2 hpf
      by_cases hsf : ((P.defs[sc.d]!).body.nodes[n]!).form = .scan
      · have hmm : ((P.defs[sc.d]!).body.nodes[n]!).members = ((P.defs[sc.d]!).body.nodes[n]!).n := by
          simp only [Node.members, hsf]; rfl
        rw [hmm] at hstep1
        refine scope_scan hr (hB sc hsc) hn' hsf hfnlt hcal hnok' hseq hcntc hO hv fun m hm xm hxm => ?_
        have h3 := Nat.mul_le_mul_right (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).gates
          (show m + 1 ≤ ((P.defs[sc.d]!).body.nodes[n]!).n by omega)
        rw [Nat.add_mul, Nat.one_mul] at h3
        have hgood' : Resolve.Good P (Scope.child sc ((P.defs[sc.d]!).body.nodes[n]!).fn (sc.off + (P.defs[sc.d]!).body.offsets[n]! +
            m * (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).gates) n m) :=
          Resolve.good_child P hsc hn' rfl hpf (by rw [hmm]; exact hm) rfl
        obtain ⟨ws, hws, hwl, hwr⟩ := ih ((P.defs[sc.d]!).body.nodes[n]!).fn hfnlt _ rfl hgood' hnc hgdc
          (by show lo ≤ sc.off + (P.defs[sc.d]!).body.offsets[n]! + m * (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).gates; omega)
          (by show sc.off + (P.defs[sc.d]!).body.offsets[n]! + m * (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).gates +
            (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).gates ≤ hi; omega) xm hxm
        exact ⟨ws, by rw [wordDef_fuel P _ _ _ hfnlt]; exact hws, hwl, hwr⟩
      have hf : ((P.defs[sc.d]!).body.nodes[n]!).form = .call ∨ ((P.defs[sc.d]!).body.nodes[n]!).form = .batch := by
        rcases form_of_ne_scan hsf with h | h | h
        · have := hprimp h; rw [hpf] at this; cases this
        · exact Or.inl h
        · exact Or.inr h
      have hcL := hcntL hf
      have hrefs : ∀ m' < ((P.defs[sc.d]!).body.nodes[n]!).members, ∀ i j q, i < (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).params.size → j < ((P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).params[i]!).2.leaves →
          resolve P sc (OperandWidths.refOf ((P.defs[sc.d]!).body.nodes[n]!) m' i j) = .ok q → q < (Flock.OfProgram.ofProgram P).N →
          ((List.range ((P.defs[sc.d]!).body.nodes[n]!).args.size).flatMap fun i => (List.range (OperandWidths.cntOf P ((P.defs[sc.d]!).body.nodes[n]!) i)).map fun t =>
            Flock.Program.refVal x (Flock.Program.paramStarts (P.defs[sc.d]!).params) outs (OperandWidths.refOf ((P.defs[sc.d]!).body.nodes[n]!) m' i t)).getD
            ((Flock.Program.paramStarts (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).params).getD i 0 + j) default = portWord Y ⟨q, 0⟩ := by
        intro m' hm' i j q hi hj hq hqN
        rw [hsz]
        refine (flat_params_getD (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).params (OperandWidths.cntOf P ((P.defs[sc.d]!).body.nodes[n]!))
          (fun i t => Flock.Program.refVal x (Flock.Program.paramStarts (P.defs[sc.d]!).params) outs (OperandWidths.refOf ((P.defs[sc.d]!).body.nodes[n]!) m' i t))
          (fun i hi => hcL i (by rw [hsz]; exact hi)) hi hj).trans ?_
        exact hv _ (refOf_refOk_comp (hB sc hsc) hn' hf hbpc hi hj hm') q hq hqN
      rcases hf with hf | hf
      · have hm0 : ((P.defs[sc.d]!).body.nodes[n]!).members = 1 := members_one (Or.inr hf)
        rw [hm0, Nat.one_mul] at hstep1
        have hgood' : Resolve.Good P (Scope.child sc ((P.defs[sc.d]!).body.nodes[n]!).fn (sc.off + (P.defs[sc.d]!).body.offsets[n]!) n 0) :=
          Resolve.good_child P hsc hn' rfl hpf (by rw [hm0]; omega) (by simp)
        have hargs := argVals_seq P (Flock.Program.refVal x (Flock.Program.paramStarts (P.defs[sc.d]!).params) outs) (Or.inr hf) hseq
        obtain ⟨ws, hws, -, hwr⟩ := ih ((P.defs[sc.d]!).body.nodes[n]!).fn hfnlt (Scope.child sc ((P.defs[sc.d]!).body.nodes[n]!).fn (sc.off + (P.defs[sc.d]!).body.offsets[n]!) n 0) rfl hgood' hnc hgdc
          (by show lo ≤ sc.off + (P.defs[sc.d]!).body.offsets[n]!; omega)
          (by show sc.off + (P.defs[sc.d]!).body.offsets[n]! + (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).gates ≤ hi; omega)
          (((P.defs[sc.d]!).body.nodes[n]!).args.toList.flatMap fun a =>
            Flock.Program.argVals (Flock.Program.refVal x (Flock.Program.paramStarts (P.defs[sc.d]!).params) outs) a a.seq.count id)
          (fun i j q hi hj hq hqN => by
            rw [hargs]
            refine hrefs 0 (by rw [hm0]; omega) i j q hi hj ?_ hqN
            have h1 := resolve_step_up hr (Resolve.rs_up P sc ((P.defs[sc.d]!).body.nodes[n]!).fn _ n 0 i j (Or.inr hf)) hq
            unfold OperandWidths.refOf
            rw [if_neg (by rw [hf]; decide)]
            exact h1)
        have hev : wordDef? P sc.d ((P.defs[sc.d]!).body.nodes[n]!).fn (((P.defs[sc.d]!).body.nodes[n]!).args.toList.flatMap fun a =>
            Flock.Program.argVals (Flock.Program.refVal x (Flock.Program.paramStarts (P.defs[sc.d]!).params) outs) a a.seq.count id) =
            some ws := by
          rw [wordDef_fuel P _ _ _ hfnlt]; exact hws
        refine ⟨outs.push ws, ?_, scopeOuts_push hO ?_⟩
        · simp [nodeStep, wordWires, hf, hcal, show ¬ sc.d ≤ ((P.defs[sc.d]!).body.nodes[n]!).fn by omega, hev]
        · intro j hj q hq hqN
          have hdn : (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).prim = false ∧ j < (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).body.ret.count := by
            have := hBn.2.2.2.1 j hj; rw [hf] at this; exact this
          exact hwr j hdn.2 q (resolve_step_up hr (Resolve.rs_call P sc n j hf) hq) hqN
      · have hmb : ((P.defs[sc.d]!).body.nodes[n]!).members = ((P.defs[sc.d]!).body.nodes[n]!).n := members_batch hf
        rw [hmb] at hstep1
        have hop : ∀ m' < ((P.defs[sc.d]!).body.nodes[n]!).n, ∀ i < ((P.defs[sc.d]!).body.nodes[n]!).args.size, ∀ t < OperandWidths.cntOf P ((P.defs[sc.d]!).body.nodes[n]!) i,
            Resolve.memberIdx (((P.defs[sc.d]!).body.nodes[n]!).args[i]!) m' t < (((P.defs[sc.d]!).body.nodes[n]!).args[i]!).seq.count := by
          intro m' hm' i hi t ht
          rw [hcL i hi] at ht
          have hup := hBn.2.2.1 i (by rw [hbpc, ← hsz]; exact hi) t (by rw [hbpc]; exact ht) m' (by rw [hmb]; exact hm')
          rw [hf] at hup
          exact hup
        have hmem' : ∀ m' < ((P.defs[sc.d]!).body.nodes[n]!).n, ∃ ws, wordDef? P sc.d ((P.defs[sc.d]!).body.nodes[n]!).fn ((List.range ((P.defs[sc.d]!).body.nodes[n]!).args.size).flatMap fun i =>
            batchArg (Flock.Program.refVal x (Flock.Program.paramStarts (P.defs[sc.d]!).params) outs) (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!) ((P.defs[sc.d]!).body.nodes[n]!) m' i) = some ws ∧
            ws.length = (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).body.ret.count ∧ ∀ l < (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).body.ret.count, ∀ q,
              resolve P (Scope.child sc ((P.defs[sc.d]!).body.nodes[n]!).fn (sc.off + (P.defs[sc.d]!).body.offsets[n]! + m' * (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).gates) n m') ((P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).body.ret.get l) = .ok q → q < (Flock.OfProgram.ofProgram P).N → ws.getD l default = portWord Y ⟨q, 0⟩ := by
          intro m' hm'
          have h3 := Nat.mul_le_mul_right (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).gates (show m' + 1 ≤ ((P.defs[sc.d]!).body.nodes[n]!).n by omega)
          rw [Nat.add_mul, Nat.one_mul] at h3
          have hgood' : Resolve.Good P (Scope.child sc ((P.defs[sc.d]!).body.nodes[n]!).fn (sc.off + (P.defs[sc.d]!).body.offsets[n]! + m' * (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).gates) n m') := Resolve.good_child P hsc hn' rfl hpf (by rw [hmb]; exact hm') rfl
          obtain ⟨ws, hws, hwl, hwr⟩ := ih ((P.defs[sc.d]!).body.nodes[n]!).fn hfnlt (Scope.child sc ((P.defs[sc.d]!).body.nodes[n]!).fn (sc.off + (P.defs[sc.d]!).body.offsets[n]! + m' * (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).gates) n m') rfl hgood' hnc hgdc
            (by show lo ≤ sc.off + (P.defs[sc.d]!).body.offsets[n]! + m' * (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).gates; omega)
            (by show sc.off + (P.defs[sc.d]!).body.offsets[n]! + m' * (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).gates + (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).gates ≤ hi; omega)
            ((List.range ((P.defs[sc.d]!).body.nodes[n]!).args.size).flatMap fun i =>
              batchArg (Flock.Program.refVal x (Flock.Program.paramStarts (P.defs[sc.d]!).params) outs) (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!) ((P.defs[sc.d]!).body.nodes[n]!) m' i)
            (fun i j q hi hj hq hqN => by
              rw [batchArgs_eq P _ hf hseq m' (hop m' hm')]
              refine hrefs m' (by rw [hmb]; exact hm') i j q hi hj ?_ hqN
              have h1 := resolve_step_up hr (Resolve.rs_up_batch P sc ((P.defs[sc.d]!).body.nodes[n]!).fn _ n m' i j hf) hq
              unfold OperandWidths.refOf
              rw [if_pos (by rw [hf]; rfl)]
              exact h1)
          exact ⟨ws, by rw [wordDef_fuel P _ _ _ hfnlt]; exact hws, hwl, hwr⟩
        obtain ⟨ms, hms⟩ : ∃ ms, ms = (List.range ((P.defs[sc.d]!).body.nodes[n]!).n).map fun m' => (wordDef? P sc.d ((P.defs[sc.d]!).body.nodes[n]!).fn
            ((List.range ((P.defs[sc.d]!).body.nodes[n]!).args.size).flatMap fun i =>
              batchArg (Flock.Program.refVal x (Flock.Program.paramStarts (P.defs[sc.d]!).params) outs) (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!) ((P.defs[sc.d]!).body.nodes[n]!) m' i)).getD [] :=
          ⟨_, rfl⟩
        have hmM : (List.range ((P.defs[sc.d]!).body.nodes[n]!).n).mapM (fun m' => wordDef? P sc.d ((P.defs[sc.d]!).body.nodes[n]!).fn ((List.range ((P.defs[sc.d]!).body.nodes[n]!).args.size).flatMap fun i =>
            batchArg (Flock.Program.refVal x (Flock.Program.paramStarts (P.defs[sc.d]!).params) outs) (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!) ((P.defs[sc.d]!).body.nodes[n]!) m' i)) = some ms := by
          rw [hms]
          exact mapM_some _ _ _ fun m' hm' => by
            obtain ⟨ws, hws, -⟩ := hmem' m' (List.mem_range.1 hm')
            rw [hws]; rfl
        have hLc : (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).outLeaves = (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).body.ret.count := by rw [holc, hcntc]
        refine ⟨outs.push ms.flatten, ?_, scopeOuts_push hO ?_⟩
        · rw [nodeStep_batch wordWires (fun _ => rfl) P _ _ _ _ _ _ hf hfnlt hcal, hmM]
          rfl
        · intro j hj q hq hqN
          have hdn : j / (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).outLeaves < ((P.defs[sc.d]!).body.nodes[n]!).n ∧ ((P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).prim = true → (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).gates = 1) ∧
              ((P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).prim = false → j % (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).outLeaves < (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).body.ret.count) := by
            have := hBn.2.2.2.1 j hj; rw [hf] at this; exact this
          have hjr := hdn.2.2 hpf
          have hrs := Resolve.rs_batch P sc n j hf
          simp only [hpf, Bool.false_eq_true, ↓reduceIte] at hrs
          have h1 := resolve_step_up hr hrs hq
          obtain ⟨ws, hws, -, hwr⟩ := hmem' _ hdn.1
          have hfl : ms.flatten.getD j default = (ms.getD (j / (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).outLeaves) []).getD (j % (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).outLeaves) default := by
            refine flatten_getD_div ms (fun l hl => ?_) (by rw [hms]; simpa using hdn.1) (by have := hjr; rw [← hLc] at this; exact Nat.mod_lt _ (by omega))
            rw [hms] at hl
            obtain ⟨m', hm', rfl⟩ := List.mem_map.1 hl
            obtain ⟨ws', hws', hwl', -⟩ := hmem' m' (List.mem_range.1 hm')
            rw [hws', Option.getD_some, hwl', hLc]
          have hmsj : ms.getD (j / (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).outLeaves) [] = ws := by
            rw [hms]; simp [List.getD_eq_getElem?_getD, hdn.1, hws]
          rw [hfl, hmsj]
          exact hwr _ hjr q h1 hqN
    · obtain ⟨-, hg1, hol, w0, hret⟩ := (hD _ hfnP).1.1 hpf
      have hkw : isInputId (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).key = false ∧
          ∀ y, (Prog.evalBool P ((P.defs[sc.d]!).body.nodes[n]!).fn y).length = w0 := by
        by_cases hwp : isWordPrim (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).prim
            (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!).key = true
        · obtain ⟨b, -, hbo, -⟩ := hmap ((P.defs[sc.d]!).body.nodes[n]!).fn hfnP hwp
          exact ⟨(FlockSoundness.DecodeBoolean.word_ids hwp).1,
            evalBool_word_length hd hr hfnP hwp hret hbo (hgb _ hmemn b hbo)⟩
        · obtain ⟨hin, -, hret1, -, g, hgo⟩ := gateCallee_facts hd hr hfnP (hgate _ hmemn hpf (by simpa using hwp))
          have hw1 : w0 = 1 := by
            have h1 := hret.symm.trans hret1
            injection h1
          subst hw1
          exact ⟨hin, fun y => by rw [evalBool_gate hfnP hpf hgo]; rfl⟩
      obtain ⟨hkey, hw⟩ := hkw
      have hev : ∀ y, wordDef? P sc.d ((P.defs[sc.d]!).body.nodes[n]!).fn y = some [Prog.evalBool P ((P.defs[sc.d]!).body.nodes[n]!).fn y.flatten] := fun y => by
        rw [wordDef_fuel P _ _ y hfnlt]; exact wordDef_prim hfnP hpf y
      have hf : ((P.defs[sc.d]!).body.nodes[n]!).form = .prim ∨ ((P.defs[sc.d]!).body.nodes[n]!).form = .batch := by
        by_cases hsf : ((P.defs[sc.d]!).body.nodes[n]!).form = .scan
        · have := hcallp (Or.inr hsf); rw [hpf] at this; cases this
        rcases form_of_ne_scan hsf with h | h | h
        · exact Or.inl h
        · have := hcallp (Or.inl h); rw [hpf] at this; cases this
        · exact Or.inr h
      rw [hg1, Nat.mul_one] at hstep1
      rcases hf with hf | hf
      · rw [members_one (Or.inl hf)] at hstep1
        have hGN : sc.off + (P.defs[sc.d]!).body.offsets[n]! < (Flock.OfProgram.ofProgram P).N := by omega
        have hval := scope_node_word hroot hB hL hG hD hsc hn' rfl hf hpf hseq Y rfl hGN hkey
          (fun hin => hY ⟨_, hGN⟩ (by show lo ≤ sc.off + (P.defs[sc.d]!).body.offsets[n]!; omega)
            (by show sc.off + (P.defs[sc.d]!).body.offsets[n]! < hi; omega) hin) hret hw
          (Flock.Program.refVal x (Flock.Program.paramStarts (P.defs[sc.d]!).params) outs)
          (fun i hi t ht q hq hqG => hv _ (hBn.2.1 i t ht) q hq (by omega)) 0
        refine ⟨outs.push [Prog.evalBool P ((P.defs[sc.d]!).body.nodes[n]!).fn ((((P.defs[sc.d]!).body.nodes[n]!).args.toList.flatMap fun a =>
          Flock.Program.argVals (Flock.Program.refVal x (Flock.Program.paramStarts (P.defs[sc.d]!).params) outs) a a.seq.count
            id).flatten)], ?_, scopeOuts_push hO ?_⟩
        · simp [nodeStep, wordWires, hf, hcal, show ¬ sc.d ≤ ((P.defs[sc.d]!).body.nodes[n]!).fn by omega, hev]
        · intro j hj q hq hqN
          rw [hf] at hsh
          rw [hsh.1, hret, ← Flock.Sigma.widths_length] at hj
          simp [Ty.widths] at hj
          subst hj
          rw [resolve_prim_node hr sc hf] at hq
          obtain rfl := Except.ok.inj hq
          simpa using hval.symm
      · rw [members_batch hf] at hstep1
        have hGN : ∀ m' < ((P.defs[sc.d]!).body.nodes[n]!).n, sc.off + (P.defs[sc.d]!).body.offsets[n]! + m' < (Flock.OfProgram.ofProgram P).N := fun m' hm' => by omega
        have hop := hBn.2.2.2.2 hpf
        have hmw : ∀ m' < ((P.defs[sc.d]!).body.nodes[n]!).n, Prog.evalBool P ((P.defs[sc.d]!).body.nodes[n]!).fn (((List.range ((P.defs[sc.d]!).body.nodes[n]!).args.size).flatMap fun i =>
            batchArg (Flock.Program.refVal x (Flock.Program.paramStarts (P.defs[sc.d]!).params) outs) (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!) ((P.defs[sc.d]!).body.nodes[n]!) m' i).flatten) =
            portWord Y ⟨sc.off + (P.defs[sc.d]!).body.offsets[n]! + m', 0⟩ := by
          intro m' hm'
          have hm'' : m' < ((P.defs[sc.d]!).body.nodes[n]!).members := by rw [members_batch hf]; exact hm'
          rw [batchArgs_eq P _ hf hseq m' (fun i hi t ht => by
            have := hop i hi m' hm'' t (by unfold OperandWidths.cntOf at ht; exact ht)
            rwa [if_pos (by rw [hf]; rfl)] at this)]
          exact (scope_member_word hroot hB hL hG hD hsc hn' rfl hm'' hpf Y rfl (hGN m' hm') hkey
            (fun hin => hY ⟨_, hGN m' hm'⟩ (by show lo ≤ sc.off + (P.defs[sc.d]!).body.offsets[n]! + m'; omega)
              (by show sc.off + (P.defs[sc.d]!).body.offsets[n]! + m' < hi; omega) hin) hret hw
            (fun i t => Flock.Program.refVal x (Flock.Program.paramStarts (P.defs[sc.d]!).params) outs (OperandWidths.refOf ((P.defs[sc.d]!).body.nodes[n]!) m' i t))
            (fun i hi t ht q hq hqG => hv _ (refOf_refOk (hB _ hsc) hn' hpf hi hm'' ht) q hq (by omega)) 0).symm
        obtain ⟨X, hX⟩ : ∃ X, X = (List.range ((P.defs[sc.d]!).body.nodes[n]!).n).map fun m' => Prog.evalBool P ((P.defs[sc.d]!).body.nodes[n]!).fn
            (((List.range ((P.defs[sc.d]!).body.nodes[n]!).args.size).flatMap fun i =>
              batchArg (Flock.Program.refVal x (Flock.Program.paramStarts (P.defs[sc.d]!).params) outs) (P.defs[((P.defs[sc.d]!).body.nodes[n]!).fn]!) ((P.defs[sc.d]!).body.nodes[n]!) m' i).flatten) :=
          ⟨_, rfl⟩
        refine ⟨outs.push X, ?_, scopeOuts_push hO ?_⟩
        · rw [hX]
          simp only [nodeStep_batch wordWires (fun _ => rfl) P _ _ _ _ _ _ hf hfnlt hcal, hev,
            mapM_some_map, Option.bind_eq_bind, Option.bind_some, Option.pure_def, flatten_map_singleton]
        · intro j hj q hq hqN
          rw [hf] at hsh
          rw [hsh.1, hret, ← Flock.Sigma.widths_length, widths_array_value, List.length_replicate] at hj
          rw [resolve_batch_prim hr sc hf hpf hol] at hq
          obtain rfl := Except.ok.inj hq
          simp only [hX, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hj, Option.map_some,
            Option.getD_some]
          exact hmw j hj
  obtain ⟨o, g1, ho⟩ := foldlM_inv _ _ (fun n b => ScopeOuts P Y sc n b) hstep (P.defs[sc.d]!).body.nodes.toList.length 0 #[]
    (Nat.add_zero _) ⟨rfl, fun i h => absurd h (Nat.not_lt_zero _)⟩
  simp only [List.drop_zero] at g1
  rw [Array.length_toList] at ho
  obtain ⟨ps, hps, -⟩ := (hD _ hdl).2.2.2
  refine ⟨(List.range (P.defs[sc.d]!).body.ret.count).map fun t =>
      Flock.Program.refVal x (Flock.Program.paramStarts (P.defs[sc.d]!).params) o ((P.defs[sc.d]!).body.ret.toArray.getD t default), ?_,
    by simp, ?_⟩
  · unfold wordDef?
    simp only [hPd, Option.bind_eq_bind, Option.bind_some, hp, Bool.false_eq_true, ↓reduceIte]
    rw [walkStep_comp wordWires (fun _ => rfl) P _ _ _ hPd hp, g1]
    rfl
  · intro l hl q hq hqN
    have hget : (P.defs[sc.d]!).body.ret.toArray.getD l default = (P.defs[sc.d]!).body.ret.get l := by
      rw [hps]; exact ofParts_getD ps (by rw [← hps]; exact hl)
    simp only [List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hl, Option.map_some,
      Option.getD_some]
    rw [hget]
    exact refVal_word_res ho hx' ((hB sc hsc).2 l hl) hq hqN

/-- **`UnitWordComputes` for a template of batches on any axis and scans over word primitives, gates and earlier such templates**: where C is
topological, its gates are located, unit `g` resolves and each Boolean Definition its callee's reached composites lead to
evaluates through gates alone. -/
theorem unitWordComputes_nested {bytes : ByteArray} {P : Prog} (hd : decode bytes = .ok P)
    (hr : (P.defs[P.root]!).body.recut = #[])
    (hG : ∀ h < InferenceOnlyVerifier.Properties.gates P, InferenceOnlyVerifier.Properties.GateOk P h)
    {g k m : ℕ} (hu : Spot.BlockFront.unitAt P g = .ok (k, m)) (hR : UnitResolves P g)
    (hn : ScanNested P (P.child P.rootScope k m).d) (hgd : GateBodiesDeep P (P.child P.rootScope k m).d)
    (Y : (Flock.OfProgram.ofProgram P).Trace)
    (hY : ∀ h : Fin (Flock.OfProgram.ofProgram P).N, InUnit P k m h.val → Flock.OfProgram.inputAt P h = false →
      Y h = (Flock.OfProgram.ofProgram P).op h Y) :
    outWords Y g = wordEval P (P.child P.rootScope k m).d (argWords Y g) := by
  have hR' := hR
  obtain ⟨k', m', x, hu', hx0, -, hres⟩ := hR
  rw [hu] at hu'
  simp only [Except.ok.injEq, Prod.mk.injEq] at hu'
  obtain ⟨rfl, rfl⟩ := hu'
  have hx : (P.defs[P.root]!).body.nodes[k]? = some x := by rw [Array.getElem!_eq_getD]; exact hx0
  have hk : k < (P.defs[P.root]!).body.nodes.size := (Array.getElem?_eq_some_iff.1 hx).1
  have hxk : (P.defs[P.root]!).body.nodes[k]! = x := by
    rw [getElem!_pos _ k hk]; exact (Array.getElem?_eq_some_iff.1 hx).2
  obtain ⟨st0, hP, hinv, -⟩ := BodiesOk.decode_full bytes P hd hr
  have hD : ∀ i < P.defs.size, BodiesOk.DefOk P.defs i P.defs[i]! := by rw [hP]; exact hinv.1
  obtain ⟨hrt, hrp, -⟩ := BodiesOk.decode_root bytes P hd hr
  have hroot : P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false := ⟨hrt, hrp⟩
  have hB : ∀ sc, Resolve.Good P sc → Resolve.BodyOk P sc.d := fun sc _ => BodiesOk.decoded_bodyOk bytes P hd hr sc.d
  have hL := OperandWidths.decoded_laid bytes P hd hr
  obtain ⟨hsh, hmem⟩ := unitAt_shape hu
  have hc0 : (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).prim = false := by
    cases hn with
    | mk hp _ _ _ _ => exact hp
  have hgood : Resolve.Good P (P.child P.rootScope k m) :=
    Resolve.good_child P (parent := P.rootScope) ⟨rfl, rfl, rfl⟩ hk rfl hc0 hmem rfl
  obtain ⟨hdl, hpc⟩ := Resolve.good_d P hroot hB _ hgood
  have hbound : (P.child P.rootScope k m).off + (P.defs[(P.child P.rootScope k m).d]!).gates ≤ (Flock.OfProgram.ofProgram P).N := by
    have hoff : (P.child P.rootScope k m).off = (P.defs[P.root]!).body.offsets[k]! +
        m * (P.defs[(P.child P.rootScope k m).d]!).gates := by
      show 0 + _ + _ = _; rw [Nat.zero_add]; rfl
    obtain ⟨-, hKr, hnr⟩ := hL _ hrt hrp
    have h1 := hnr k hk
    have h2 := Resolve.offsets_mono P P.root (hL _ hrt) hrp (k + 1) _ (by omega) (Nat.le_refl _)
    have h3 := Nat.mul_le_mul_right (P.defs[(P.child P.rootScope k m).d]!).gates
      (show m + 1 ≤ ((P.defs[P.root]!).body.nodes[k]!).members by omega)
    rw [Nat.add_mul, Nat.one_mul] at h3
    have h4 : (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).gates =
        (P.defs[(P.child P.rootScope k m).d]!).gates := rfl
    rw [h4] at h1
    show _ ≤ (P.defs[P.root]!).gates
    omega
  have hPd : P.defs[(P.child P.rootScope k m).d]? = some (P.defs[(P.child P.rootScope k m).d]!) := by
    rw [getElem!_pos P.defs _ hdl]; exact getElem?_pos P.defs _ hdl
  obtain ⟨ws, hws, hwl, hwr⟩ := scope_words hd hr hroot hB hL hG hD Y hbound
    (fun h h1 h2 hin => hY h (by unfold InUnit; simp only [hPd, Option.map_some, Option.getD_some]; exact ⟨h1, h2⟩) hin)
    _ _ rfl hgood hn hgd (Nat.le_refl _) (Nat.le_refl _) (argWords Y g)
    (fun i j q hi hj hq hqN => unit_param_word hd hr hu hR' Y hi hj hq hqN)
  have hwe : wordEval P (P.child P.rootScope k m).d (argWords Y g) = ws := by
    unfold wordEval
    rw [wordDef_fuel P _ _ _ (by omega), hws]
    rfl
  rw [hwe]
  obtain ⟨-, hcount, houtL⟩ := (hD _ hdl).1.2 hpc
  have hcd : (P.child P.rootScope k m).d = x.fn := by rw [child_d, hxk]
  have hc : P.defs.getD x.fn default = P.defs[(P.child P.rootScope k m).d]! := by
    rw [hcd]; exact Array.getElem!_eq_getD.symm
  rw [hc] at hres
  let F : ℕ → V Spot.BlockFront.Port := fun l => do
    let q ← resolve P P.rootScope
      { p := false, i := k, j := m * (P.defs[(P.child P.rootScope k m).d]!).ret.leaves + l }
    pure ({ gate := q, width := (P.defs[(P.child P.rootScope k m).d]!).ret.widths.getD l 0 } : Spot.BlockFront.Port)
  obtain ⟨r, hrm, h2⟩ := mapM_forall₂ F (List.range (P.defs[(P.child P.rootScope k m).d]!).ret.leaves) fun l hl => by
    obtain ⟨q, hq⟩ := hres l (List.mem_range.1 hl)
    exact ⟨_, by simp only [F, hq]; rfl⟩
  have huo : unitOuts P g = r.toArray := by
    unfold unitOuts
    simp only [hu, hx0, hc, bind, Except.bind]
    rw [Array.mapM_eq_mapM_toList, Array.toList_range]
    change (match (Functor.map List.toArray
      (List.mapM F (List.range (P.defs[(P.child P.rootScope k m).d]!).ret.leaves)) : V _) with
      | .ok ps => ps | .error _ => #[]) = _
    rw [hrm]
    rfl
  unfold outWords
  rw [huo, List.toList_toArray]
  have hrlen : r.length = (P.defs[(P.child P.rootScope k m).d]!).ret.leaves := by
    rw [← List.Forall₂.length_eq h2, List.length_range]
  apply List.ext_getElem (by simp [hrlen, hcount, hwl])
  intro l h1 h1'
  have hl : l < (P.defs[(P.child P.rootScope k m).d]!).ret.leaves := by rw [← hrlen]; simpa using h1
  simp only [List.getElem_map]
  have hF := List.Forall₂.get h2 (by simpa using hl) (by simpa using h1)
  simp only [List.get_eq_getElem, List.getElem_range] at hF
  cases hq : resolve P P.rootScope
      { p := false, i := k, j := m * (P.defs[(P.child P.rootScope k m).d]!).ret.leaves + l } with
  | error e => simp [F, hq, bind, Except.bind] at hF
  | ok q =>
    simp only [F, hq, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at hF
    rw [← hF]
    have hsh' : (((P.defs[P.root]!).body.nodes[k]!).form = .call ∧ m = 0) ∨
        ((P.defs[P.root]!).body.nodes[k]!).form = .batch := by
      rcases hsh with h | h
      · exact Or.inl ⟨h, by have := hmem; rw [members_one (Or.inr h)] at this; omega⟩
      · exact Or.inr h
    have h3 : (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).outLeaves =
        (P.defs[(P.child P.rootScope k m).d]!).ret.leaves := houtL
    have hq0 : resolve P P.rootScope ⟨false, k,
        m * (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).outLeaves + l⟩ = .ok q := by
      rw [h3]; exact hq
    have hretq := unit_ret_resolve hr hc0 hsh' (by rw [h3]; exact hl) hq0
    have hlc : l < (P.defs[(P.child P.rootScope k m).d]!).body.ret.count := by rw [hcount]; exact hl
    have hRef := (hB _ hgood).2 l hlc
    have hqN : q < (Flock.OfProgram.ofProgram P).N := by
      have hinv' : Resolve.Inv P ((P.child P.rootScope k m).off +
          (P.defs[(P.child P.rootScope k m).d]!).body.offsets[(P.defs[(P.child P.rootScope k m).d]!).body.nodes.size]!)
          (P.child P.rootScope k m, (P.defs[(P.child P.rootScope k m).d]!).body.ret.get l) := by
        refine ⟨hgood, hRef, fun _ => Nat.le_add_right _ _, fun hp => ?_⟩
        dsimp only at hp ⊢
        have h4 := (hRef.2 hp).1
        have := Resolve.offsets_mono P _ (hL _ hdl) hpc _ _ (show _ + 1 ≤ _ from h4) (Nat.le_refl _)
        omega
      have := Resolve.resolve_lt P hroot hB hL _ _ _ hinv' hretq
      obtain ⟨-, hK, -⟩ := hL _ hdl hpc
      omega
    have hwsl : ws[l] = ws.getD l default := by
      simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h1']
    rw [hwsl, hwr l hlc q hretq hqN, portWord_lt Y hqN, portWord_lt Y hqN]

/-- **`UnitComputes` for a template of batches on any axis and scans over word primitives, gates and earlier such templates**: each unit of the
block computes, bit for bit, the lift of the template `admitted` gives, where that template is in `ScanNested`. The
Boolean bodies' fact is the lowering's (`lowerGateBodies`). -/
theorem unitComputes_nested (bytes : ByteArray) (P : Prog) (partition : Lean.Json)
    (cuts : List (String × List (String × ℕ))) (units : Array ℕ) (S : Prog) (a : Spot.OneStageZk.Admit) (L : Prog)
    (e : ℕ) (t : Tmpl.Tmpl) (hd : decode bytes = .ok P) (hr : (P.defs[P.root]!).body.recut = #[])
    (hs : Spot.Restrict.restrict P units = .ok S) (ha : Spot.OneStageZk.admitted partition cuts S = .ok a)
    (hab : Spot.OneStageZk.admittedBlock P units a.key = .ok ()) (hl : Flock.Lift.lift S a.d = .ok (L, e))
    (hbt : Spot.OneStageZk.blockTmpl S a = .ok t)
    (hG : ∀ h < InferenceOnlyVerifier.Properties.gates P, InferenceOnlyVerifier.Properties.GateOk P h)
    (hRs : ∀ g ∈ units, UnitResolves P g) (hn : ScanNested P a.d)
    (g : ℕ) (hg : g ∈ units) (k m : ℕ) (hu : Spot.BlockFront.unitAt P g = .ok (k, m))
    (hda : (P.child P.rootScope k m).d = a.d) (Y : (Flock.OfProgram.ofProgram P).Trace)
    (hY : ∀ h : Fin (Flock.OfProgram.ofProgram P).N, InUnit P k m h.val → Flock.OfProgram.inputAt P h = false →
      Y h = (Flock.OfProgram.ofProgram P).op h Y) :
    outBits Y g = Prog.evalBool L e (argBits Y g) := by
  obtain ⟨x, f, h1, h2, -⟩ := admittedBlock_callee hab g hg k m hu
  have hdx : (P.child P.rootScope k m).d = x.fn := by
    rw [child_d]
    generalize P.defs[P.root]! = R at h1 ⊢
    simp [getElem!_def, h1]
  have hlt : a.d < P.defs.size := by
    rw [← hda, hdx]
    exact (Array.getElem?_eq_some_iff.1 h2).1
  have hgb := lowerGateBodies bytes P partition cuts units S a L e t hd hr hs ha hlt hl hbt
  have ho := unitWordComputes_nested hd hr hG hu (hRs g hg) (by rw [hda]; exact hn) (by rw [hda]; exact hgb) Y hY
  have hw : (argWords Y g).map List.length = leafWidths (P.defs[(P.child P.rootScope k m).d]!) := by
    rw [← (unitWidths bytes P hd hr hG g k m hu (hRs g hg)).1]
    simp [argWords, Function.comp_def, portWord_length]
  rw [hda] at hw ho
  unfold outBits argBits
  rw [portBits_eq, portBits_eq]
  change (outWords Y g).flatten = Prog.evalBool L e (argWords Y g).flatten
  rw [ho, liftWordEval_nested bytes P units S a.d L e hd hr hs hlt hn hgb hl _ hw]

/-- **`BlockEvalDoor` for a block whose template is in `ScanNested`**: `BlockEvalDoor`'s conclusion, under its
hypotheses and that the template `admitted` gives is of batches on any axis and scans over word primitives, gates and earlier such templates. -/
theorem blockEvalDoor_nested (bytes : ByteArray) (P : Prog) (partition : Lean.Json)
    (cuts : List (String × List (String × ℕ))) (units : Array ℕ) (S : Prog) (a : Spot.OneStageZk.Admit)
    (fr : Spot.BlockFront.Front) (L : Prog) (e : ℕ) (t : Tmpl.Tmpl) (hd : decode bytes = .ok P)
    (hr : (P.defs[P.root]!).body.recut = #[]) (hs : Spot.Restrict.restrict P units = .ok S)
    (ha : Spot.OneStageZk.admitted partition cuts S = .ok a)
    (hab : Spot.OneStageZk.admittedBlock P units a.key = .ok ()) (hf : Spot.BlockFront.front P units = .ok fr)
    (_ : units.toList.Pairwise (· < ·)) (hl : Flock.Lift.lift S a.d = .ok (L, e))
    (hbt : Spot.OneStageZk.blockTmpl S a = .ok t)
    (hG : ∀ h < InferenceOnlyVerifier.Properties.gates P, InferenceOnlyVerifier.Properties.GateOk P h)
    (hRs : ∀ g ∈ units, UnitResolves P g) (hn : ScanNested P a.d)
    (blocks : Array (Array (ℕ × ℕ))) (u : Fin blocks.size) (hB : BlockIs P blocks u fr)
    (X : (Flock.OfProgram.ofProgram P).Trace) (g : ℕ) (hg : g ∈ units) :
    let Y := (Flock.OfProgram.partOf P blocks).localEval X u.castSucc
    Prog.evalBool L e (argBits Y g) = outBits Y g := by
  intro Y
  obtain ⟨k, m, hu, hin⟩ := frontUnitRanges P units fr hf g hg
  have hY : ∀ h : Fin (Flock.OfProgram.ofProgram P).N, InUnit P k m h.val → Flock.OfProgram.inputAt P h = false →
      Y h = (Flock.OfProgram.ofProgram P).op h Y :=
    fun h h1 h2 => Verity.Partition.localEval_of_mem _ X ((hB h).2 ⟨hin h.val h1, h2⟩)
  exact (unitComputes_nested bytes P partition cuts units S a L e t hd hr hs ha hab hl hbt hG hRs hn g hg k m hu
    (admitCallee bytes P partition cuts units S a hd hs ha hab g hg k m hu) Y hY).symm

end UnitWordProof

/-- **The lift where the door lowers a block**: `lowerBlock` accepts only past `admitted`, whose `on-bits` lifts the
template. -/
theorem lowerBlock_lift {CJ partition : Lean.Json} {cuts : List (String × List (String × ℕ))} {S : Prog}
    {formed : String → V Unit} {file : ByteArray}
    (h : Spot.OneStageZk.lowerBlock CJ partition cuts S formed = .ok file) :
    ∃ a L e, Spot.OneStageZk.admitted partition cuts S = .ok a ∧ Flock.Lift.lift S a.d = .ok (L, e) := by
  unfold Spot.OneStageZk.lowerBlock at h
  cases ha : Spot.OneStageZk.admitted partition cuts S with
  | error e => rw [ha] at h; exact absurd h (by intro h; cases h)
  | ok a =>
    obtain ⟨_, L, dl, _, _, hl, -⟩ := admitted_walk ha
    exact ⟨a, L, dl, rfl, hl⟩

/-- **The front where the door restricts C**: `restrict` accepts only past the front of its units, sorted. -/
theorem restrict_front {P S : Prog} {units : Array ℕ} (h : Spot.Restrict.restrict P units = .ok S) :
    ∃ fr, Spot.BlockFront.front P (units.qsort (· < ·)) = .ok fr := by
  unfold Spot.Restrict.restrict at h
  dsimp only at h
  cases hf : Spot.BlockFront.front P (units.qsort (· < ·)) with
  | error e => rw [hf] at h; exact absurd h (by intro h; cases h)
  | ok fr => exact ⟨fr, rfl⟩

/-- Units in increasing order are their own sort. -/
theorem qsort_ascending {units : Array ℕ} (h : units.toList.Pairwise (· < ·)) : units.qsort (· < ·) = units := by
  refine Verity.QSort.qsort_sorted units _ fun i hi j hj => ?_
  have hp := List.pairwise_iff_getElem.1 h
  rcases Nat.lt_trichotomy i j with hij | rfl | hij
  · have := hp i j (by simpa using hi) (by simpa using hj) hij
    simp only [Array.getElem_toList] at this
    simp [hij, this]
  · simp
  · have := hp j i (by simpa using hj) (by simpa using hi) hij
    simp only [Array.getElem_toList] at this
    simp [Nat.not_lt.2 (Nat.le_of_lt hij), Nat.not_lt.2 (Nat.le_of_lt this)]

/-- `front` returns at a block's units, given in increasing order, wherever `restrict` returns. -/
theorem restrict_front_ascending {P S : Prog} {units : Array ℕ} (h : Spot.Restrict.restrict P units = .ok S)
    (ha : units.toList.Pairwise (· < ·)) : ∃ fr, Spot.BlockFront.front P units = .ok fr := by
  have := restrict_front h
  rwa [qsort_ascending ha] at this

end Flock.Chain
