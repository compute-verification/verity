module

public import Proofs.Properties.InferenceOnly.DigestReads
import Proofs.Properties.InferenceOnly.BodiesOk.Node
import Batteries.Tactic.OpenPrivate
import Circuits.Program
import all Lean.Data.Json.Basic
import all Lean.Data.Json.FromToJson.Basic
import all Init.Data.Ord.String
import Std.Data.TreeMap.Raw.Lemmas

/-!
# `ReadsPin`, proved

Each side's canonical entry shows its Definition: `comp_view` reads a composite's id, signature, nodes and return out of
its canonical entry (`CompView`, node by node `NodeView`), with each argument's flat parts standing for its decoded
parts (`rows_before`, by induction over the nodes through `flat_of_parse`). Two such views of equivalent entries give
the same fields one by one (`node_same`, `same_comp`): the flat parts parse to the same parts (`written_parts`), and
`buildNode` reads only its callee's key, parameters and return type (`buildNode_congr`), which `Same` callees share. A
primitive's canonical entry is its entry (`same_prim`), and no primitive's entry is equivalent to a composite's.
-/

public section

namespace InferenceOnlyVerifier.Properties.Record.SecurityProofs

open Lean (Json)
open Flock.Program
open Claims.Json (Wf)
open FlockDeploy (JsonEquiv jsonForm jsonEquiv_arr_iff)
open InferenceOnlyVerifier.DefinitionDigest (canonicalEntry flatParts written isComposite)
open private natJ? intJ? paramsOf at_ buildNode from Circuits.Program
open InferenceOnlyVerifier.Properties.SecurityProofs.BodiesOk (array_mapM_size array_mapM_get buildNode_spec)

namespace ReadsPinProof

/-- An object over a well-formed map: what `setJ` writes a field into. -/
def IsObj (j : Json) : Prop := ∃ m : Std.TreeMap.Raw String Json compare, j = .obj m ∧ m.WF

theorem isObj_setJ {j : Json} (h : IsObj j) (k : String) (v : Json) : IsObj (setJ j k v) := by
  obtain ⟨m, rfl, hm⟩ := h
  exact ⟨_, rfl, hm.insert⟩

theorem isObj_of_wf {j : Json} (hj : Wf j) {k : String} {v : Json} (h : j.getObjVal? k = .ok v) : IsObj j := by
  cases hj with
  | obj hm _ => exact ⟨_, rfl, hm⟩
  | _ => simp [Json.getObjVal?, throw, throwThe, MonadExceptOf.throw] at h

theorem getObjVal?_setJ_ne {j : Json} (hj : IsObj j) {k k' : String} (v : Json) (h : k ≠ k') :
    (setJ j k v).getObjVal? k' = j.getObjVal? k' := by
  obtain ⟨m, rfl, hm⟩ := hj
  simp only [setJ, Json.getObjVal?, Std.TreeMap.Raw.get?_eq_getElem?, Std.TreeMap.Raw.getElem?_insert hm,
    Std.compare_eq_iff_eq, h, ↓reduceIte]

theorem getObjVal?_setJ_eq {j : Json} (hj : IsObj j) (k : String) (v : Json) :
    (setJ j k v).getObjVal? k = .ok v := by
  obtain ⟨m, rfl, hm⟩ := hj
  simp only [setJ, Json.getObjVal?, Std.TreeMap.Raw.get?_eq_getElem?, Std.TreeMap.Raw.getElem?_insert_self hm]
  rfl

theorem getObjValD_setJ_ne {j : Json} (hj : IsObj j) {k k' : String} (v : Json) (h : k ≠ k') :
    (setJ j k v).getObjValD k' = j.getObjValD k' := by
  simp only [Json.getObjValD, getObjVal?_setJ_ne hj v h]

theorem getObjValD_setJ_eq {j : Json} (hj : IsObj j) (k : String) (v : Json) :
    (setJ j k v).getObjValD k = v := by
  simp only [Json.getObjValD, getObjVal?_setJ_eq hj]
  rfl

theorem getObjValD_of {j v : Json} {k : String} (h : j.getObjVal? k = .ok v) : j.getObjValD k = v := by
  simp only [Json.getObjValD, h]
  rfl

theorem getObjValD_eq (j : Json) (k : String) : (j.getObjVal? k).toOption.getD Json.null = j.getObjValD k := by
  unfold Json.getObjValD; rfl

/-- A string field read with a default other than `""` names a present string. -/
theorem str_of_field {e : Json} {k s : String} (h : (e.getObjValAs? String k).toOption.getD "" = s) (hs : s ≠ "") :
    e.getObjValD k = .str s := by
  simp only [Json.getObjValAs?] at h
  generalize e.getObjValD k = x at h ⊢
  cases x <;> first | (exact absurd h.symm hs) | skip
  case str t => cases h; rfl

theorem canonicalEntry_of_not {cs ts : Array Json} {e : Json} (h : isComposite e = false) :
    canonicalEntry cs ts e = .ok e := by
  unfold canonicalEntry
  simp only [bind, Except.bind, h]
  rfl

theorem canonicalEntry_comp {cs ts : Array Json} {e ce : Json} {nds : Array Json} (hc : isComposite e = true)
    (hn : (e.getObjValD "body").getObjValD "nodes" = .arr nds) (h : canonicalEntry cs ts e = .ok ce) :
    ∃ rows out ret, Steps cs ts #[] #[] nds.toList rows out ∧
      flatParts rows Flock.Steps.FUEL ((e.getObjValD "body").getObjValD "ret") = .ok ret ∧
      ce = setJ e "body" (setJ (setJ (e.getObjValD "body") "nodes" (.arr out)) "ret" (written ret)) := by
  unfold canonicalEntry at h
  simp only [bind, Except.bind] at h
  split at h
  · split at h
    · rename_i nodes hn'
      rw [hn] at hn'
      cases hn'
      split at h
      · cases h
      rename_i r hr
      split at h
      · cases h
      rename_i ret hret
      simp only [pure, Except.pure, Except.ok.injEq] at h
      refine ⟨r.1, r.2, ret, ?_, hret, h.symm⟩
      rw [← Array.forIn_toList] at hr
      clear h hret hn
      generalize (#[] : Array (Array (Array Json))) = rows0 at hr
      generalize (#[] : Array Json) = out0 at hr
      generalize nds.toList = l at hr ⊢
      induction l generalizing rows0 out0 with
      | nil =>
        simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at hr
        subst hr
        exact .nil _ _
      | cons n l ih =>
        rw [List.forIn_cons] at hr
        simp only [bind, Except.bind] at hr
        split at hr
        · cases hr
        rename_i s hs
        split at hs
        · rename_i args ha
          split at hs
          · cases hs
          rename_i row hrow
          simp only [pure, Except.pure, Except.ok.injEq] at hs
          subst hs
          exact .cons args row ha hrow (ih _ _ hr)
        · rename_i x hx
          simp only [throw, throwThe, MonadExceptOf.throw] at hs
          cases hs
    · rename_i hn'
      exact absurd hn (hn' nds)
  · rename_i hc'
    simp_all

theorem push_get_lt {α : Type} [Inhabited α] (a : Array α) (x : α) {i : Nat} (h : i < a.size) :
    (a.push x)[i]! = a[i]! := by
  rw [getElem!_pos (a.push x) i (by simp; omega), getElem!_pos a i h, Array.getElem_push_lt]

theorem push_get_eq {α : Type} [Inhabited α] (a : Array α) (x : α) : (a.push x)[a.size]! = x := by
  rw [getElem!_pos (a.push x) a.size (by simp), Array.getElem_push_eq]

theorem steps_index {cs ts : Array Json} {rows0 : Array (Array (Array Json))} {out0 : Array Json} {l : List Json}
    {rows : Array (Array (Array Json))} {out : Array Json} (h : Steps cs ts rows0 out0 l rows out) :
    rows.size = rows0.size + l.length ∧ out.size = out0.size + l.length ∧
    (∀ k < rows0.size, rows[k]! = rows0[k]!) ∧ (∀ k < out0.size, out[k]! = out0[k]!) ∧
    ∀ k < l.length, ∃ (R : Array (Array (Array Json))) (args : Array Json),
      R.size = rows0.size + k ∧ (∀ k2 < R.size, R[k2]! = rows[k2]!) ∧
      (l[k]!).getObjValD "args" = .arr args ∧
      args.mapM (fun a => flatParts R Flock.Steps.FUEL (a.getObjValD "refs")) = .ok rows[rows0.size + k]! ∧
      out[out0.size + k]! = canonNode cs ts l[k]! args rows[rows0.size + k]! := by
  induction h with
  | nil rows out => exact ⟨by simp, by simp, fun _ _ => rfl, fun _ _ => rfl, fun k hk => by simp at hk⟩
  | @cons rows0 out0 n l rows out args row ha hrow _ ih =>
    obtain ⟨hs, ho, hp, hq, hk⟩ := ih
    simp only [Array.size_push, List.length_cons] at hs ho hk ⊢
    have hp' : ∀ k < rows0.size, rows[k]! = rows0[k]! := fun k hk' => by
      rw [hp k (by simp; omega), push_get_lt _ _ hk']
    have hq' : ∀ k < out0.size, out[k]! = out0[k]! := fun k hk' => by
      rw [hq k (by simp; omega), push_get_lt _ _ hk']
    refine ⟨by omega, by omega, hp', hq', fun k hk' => ?_⟩
    rcases k with _ | k
    · refine ⟨rows0, args, rfl, fun k2 h2 => (hp' k2 h2).symm, by simpa using ha, ?_, ?_⟩
      · rw [Nat.add_zero, hp rows0.size (by simp), push_get_eq]; exact hrow
      · rw [Nat.add_zero, Nat.add_zero, hq out0.size (by simp), push_get_eq, hp rows0.size (by simp), push_get_eq]
        rfl
    · obtain ⟨R, a, hR, hRp, hal, hm, hout⟩ := hk k (by omega)
      refine ⟨R, a, by omega, hRp, by simpa using hal, ?_, ?_⟩
      · rw [show rows0.size + (k + 1) = rows0.size + 1 + k by omega]; exact hm
      · rw [show out0.size + (k + 1) = out0.size + 1 + k by omega,
          show rows0.size + (k + 1) = rows0.size + 1 + k by omega]
        simpa using hout

theorem at_ok {α : Type} [Inhabited α] {table : Array α} {x : Json} {what : String} {t : α}
    (h : at_ table x what = .ok t) : ∃ i : Int, intJ? x = some i ∧ 0 ≤ i ∧ i < table.size ∧ t = table[i.toNat]! := by
  unfold at_ at h
  split at h
  · rename_i i hi
    split at h
    · rename_i hc
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
      cases h
      exact ⟨i, hi, hc.1, hc.2, rfl⟩
    · cases h
  · cases h

theorem getNat_of_intJ {x : Json} {i : Int} (h : intJ? x = some i) (h0 : 0 ≤ i) : x.getNat? = .ok i.toNat := by
  unfold intJ? at h
  split at h
  · rename_i n
    split at h
    · rename_i hc
      cases h
      obtain ⟨m, e⟩ := n
      simp only [beq_iff_eq] at hc
      subst hc
      obtain ⟨m', rfl⟩ := Int.eq_ofNat_of_zero_le h0
      rfl
    · cases h
  · cases h

theorem atJ_of_getNat {xs : Array Json} {x : Json} {n : Nat} (h : x.getNat? = .ok n) : atJ xs x = xs.getD n .null := by
  unfold atJ; rw [h]

theorem ofJson_atJ {ts : Array Json} {types : Array Ty} (ht : ts.mapM Ty.ofJson = .ok types) {x : Json} {n : Nat}
    (h : x.getNat? = .ok n) (hn : n < types.size) : Ty.ofJson (atJ ts x) = .ok types[n]! := by
  have hs := array_mapM_size _ _ _ ht
  rw [atJ_of_getNat h, Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem (by omega), Option.getD_some,
    getElem!_pos types n hn]
  exact array_mapM_get _ _ _ ht n (by omega) hn

/-- A table read the decoder makes (`at_`) and the canonical entry's (`atJ`) of one index give one type. -/
theorem ofJson_atJ_of_at {ts : Array Json} {types : Array Ty} (ht : ts.mapM Ty.ofJson = .ok types) {x : Json}
    {what : String} {t : Ty} (h : at_ types x what = .ok t) : Ty.ofJson (atJ ts x) = .ok t := by
  obtain ⟨i, hi, h0, hlt, rfl⟩ := at_ok h
  exact ofJson_atJ ht (getNat_of_intJ hi h0) (by omega)

theorem canonNode_fn {cs ts : Array Json} {n : Json} (hn : IsObj n) (args : Array Json) (row : Array (Array Json)) :
    (canonNode cs ts n args row).getObjValD "fn" = atJ cs (n.getObjValD "fn") := by
  unfold canonNode
  rw [getObjValD_setJ_ne (isObj_setJ (isObj_setJ hn _ _) _ _) _ (by decide),
    getObjValD_setJ_ne (isObj_setJ hn _ _) _ (by decide), getObjValD_setJ_eq hn]

theorem canonNode_out {cs ts : Array Json} {n : Json} (hn : IsObj n) (args : Array Json) (row : Array (Array Json)) :
    (canonNode cs ts n args row).getObjValD "out" = atJ ts (n.getObjValD "out") := by
  unfold canonNode
  rw [getObjValD_setJ_ne (isObj_setJ (isObj_setJ hn _ _) _ _) _ (by decide),
    getObjValD_setJ_eq (isObj_setJ hn _ _)]

theorem canonNode_args {cs ts : Array Json} {n : Json} (hn : IsObj n) (args : Array Json) (row : Array (Array Json)) :
    (canonNode cs ts n args row).getObjValD "args" = .arr ((args.zip row).map (canonArg ts)) := by
  unfold canonNode
  rw [getObjValD_setJ_eq (isObj_setJ (isObj_setJ hn _ _) _ _)]

theorem canonNode_other {cs ts : Array Json} {n : Json} (hn : IsObj n) (args : Array Json) (row : Array (Array Json))
    {k : String} (h1 : "fn" ≠ k) (h2 : "out" ≠ k) (h3 : "args" ≠ k) :
    (canonNode cs ts n args row).getObjVal? k = n.getObjVal? k := by
  unfold canonNode
  rw [getObjVal?_setJ_ne (isObj_setJ (isObj_setJ hn _ _) _ _) _ h3, getObjVal?_setJ_ne (isObj_setJ hn _ _) _ h2,
    getObjVal?_setJ_ne hn _ h1]

theorem canonArg_type {ts : Array Json} {a : Json} (ha : IsObj a) (ps : Array Json) :
    (canonArg ts (a, ps)).getObjValD "type" = atJ ts (a.getObjValD "type") := by
  unfold canonArg
  rw [getObjValD_setJ_ne (isObj_setJ ha _ _) _ (by decide), getObjValD_setJ_eq ha]

theorem canonArg_refs {ts : Array Json} {a : Json} (ha : IsObj a) (ps : Array Json) :
    (canonArg ts (a, ps)).getObjValD "refs" = written ps := by
  unfold canonArg
  rw [getObjValD_setJ_eq (isObj_setJ ha _ _)]

/-- The decoder's earlier arguments at node `k`. -/
def earlier (d : Defn) (k : Nat) : Nat → Nat → Option Seq :=
  fun k2 x => if k2 < k then (d.body.nodes[k2]!).args[x]?.map (·.seq) else none

/-- The rows of the nodes before `k` stand for their decoded arguments. -/
def RowsBefore (L : Bool → Nat → Option Nat) (d : Defn) (rows : Array (Array (Array Json))) (k : Nat) : Prop :=
  ∀ k2 < k, ∀ x < (d.body.nodes[k2]!).args.size, ∃ ps, (rows[k2]!)[x]? = some ps ∧
    PartsRel L ps.toList ((d.body.nodes[k2]!).args[x]!.seq.parts).toList

theorem nodeReads_unfold {P : Prog} {types : Array Ty} {d : Defn} {outs : Array Ty} {k : Nat} {nd : Json}
    (h : NodeReads P types d outs k nd) :
    ∃ (ajs : Array Json) (raw : Array Arg), nd.getObjVal? "args" = .ok (.arr ajs) ∧ raw.size = ajs.size ∧
      ((d.body.nodes[k]!).form = .scan → 0 < ajs.size) ∧
      (∀ a < ajs.size,
        (∃ t : Int, intJ? (((ajs[a]!).getObjVal? "type").toOption.getD Json.null) = some t ∧ 0 ≤ t ∧
          t < types.size ∧ (raw[a]!).ty = types[t.toNat]!) ∧
        ∃ qs, parseParts (((ajs[a]!).getObjVal? "refs").toOption.getD Json.null) (leavesOf d outs)
            (earlier d k) = .ok qs ∧
          (raw[a]!).seq = Seq.ofParts qs ∧ (raw[a]!).dims = none) ∧
      buildNode d.key k (d.body.nodes[k]!).form (P.defs[(d.body.nodes[k]!).fn]!) raw (d.body.nodes[k]!).axes
        (d.body.nodes[k]!).n (d.body.nodes[k]!).nxs = .ok ((d.body.nodes[k]!).out, (d.body.nodes[k]!).args) := by
  unfold NodeReads at h
  obtain ⟨-, -, -, -, -, h, -⟩ := h
  exact h

/-- Node `k` as one side reads it: its arguments' entries, the arguments `buildNode` read, and each argument's
parts, which the canonical row's flat parts stand for. -/
theorem node_core {P : Prog} {d : Defn} {cs ts : Array Json} {types outs : Array Ty} {nds : Array Json}
    {rows : Array (Array (Array Json))} {out : Array Json}
    (hR : ∀ k < nds.size, NodeReads P types d outs k nds[k]!)
    (hS : Steps cs ts #[] #[] nds.toList rows out) (k : Nat) (hk : k < nds.size)
    (hQ : RowsBefore (leavesOf d outs) d rows k) :
    ∃ (ajs : Array Json) (raw : Array Arg), nds[k]!.getObjVal? "args" = .ok (.arr ajs) ∧ raw.size = ajs.size ∧
      (rows[k]!).size = ajs.size ∧ (d.body.nodes[k]!).args.size = ajs.size ∧
      out[k]! = canonNode cs ts nds[k]! ajs rows[k]! ∧
      buildNode d.key k (d.body.nodes[k]!).form (P.defs[(d.body.nodes[k]!).fn]!) raw (d.body.nodes[k]!).axes
        (d.body.nodes[k]!).n (d.body.nodes[k]!).nxs = .ok ((d.body.nodes[k]!).out, (d.body.nodes[k]!).args) ∧
      ∀ a < ajs.size,
        (∃ t : Int, intJ? ((ajs[a]!).getObjValD "type") = some t ∧ 0 ≤ t ∧
          t < types.size ∧ (raw[a]!).ty = types[t.toNat]!) ∧
        ∃ qs, PartsRel (leavesOf d outs) ((rows[k]!)[a]!).toList qs.toList ∧
          (raw[a]!).seq = Seq.ofParts qs ∧ ((d.body.nodes[k]!).args[a]!).seq = Seq.ofParts qs ∧
          (raw[a]!).dims = none := by
  obtain ⟨-, -, -, -, hidx⟩ := steps_index hS
  obtain ⟨R, args, hRs, hRp, hargs, hmap, hout⟩ := hidx k (by simpa using hk)
  simp only [Array.size_empty, Nat.zero_add] at hRs hmap hout
  have hl : nds.toList[k]! = nds[k]! := by
    rw [getElem!_pos nds k hk, getElem!_pos nds.toList k (by simpa using hk), Array.getElem_toList]
  rw [hl] at hargs hout
  obtain ⟨ajs, raw, hajs, hraw, hscan, hA, hB⟩ := nodeReads_unfold (hR k hk)
  rw [getObjValD_of hajs] at hargs
  cases hargs
  have hrs := array_mapM_size _ _ _ hmap
  have H : RowsArgs (leavesOf d outs) R (earlier d k) := by
    intro k2 x s hs
    unfold earlier at hs
    split at hs
    · rename_i hk2
      obtain ⟨y, hy, rfl⟩ := Option.map_eq_some_iff.1 hs
      have hx : x < (d.body.nodes[k2]!).args.size := (Array.getElem?_eq_some_iff.1 hy).1
      have hy' : y = (d.body.nodes[k2]!).args[x]! := by
        rw [getElem!_pos _ x hx]; exact ((Array.getElem?_eq_some_iff.1 hy).2).symm
      obtain ⟨ps, hps, hrel⟩ := hQ k2 hk2 x hx
      refine ⟨ps, ?_, by rw [hy']; exact hrel⟩
      rw [Array.getElem?_eq_getElem (by omega), Option.bind_some, ← getElem!_pos R k2 (by omega), hRp k2 (by omega)]
      exact hps
    · cases hs
  have hscan' : (d.body.nodes[k]!).form = .scan → 0 < raw.size := fun h => by rw [hraw]; exact hscan h
  have hdims : ∀ i < raw.size, (raw[i]!).dims = none := fun i hi => ((hA i (by omega)).2.choose_spec).2.2
  obtain ⟨hsz, hseq, -⟩ := buildNode_spec _ _ _ _ _ _ _ _ _ _ hB hscan' hdims
  refine ⟨args, raw, hajs, hraw, hrs, by omega, hout, hB, fun a ha => ?_⟩
  obtain ⟨⟨t, ht⟩, qs, hqs, hseqa, hdim⟩ := hA a ha
  refine ⟨⟨t, by rw [← getObjValD_eq]; exact ht.1, ht.2⟩, qs, ?_, hseqa, by rw [(hseq a (by omega)).2]; exact hseqa,
    hdim⟩
  unfold parseParts at hqs
  obtain ⟨fps, hf, hrel⟩ := flat_of_parse H _ _ _ hqs
  have hg := array_mapM_get _ _ _ hmap a ha (by omega)
  rw [getObjValD_eq] at hf
  rw [← getElem!_pos args a ha, hf] at hg
  rw [getElem!_pos rows[k]! a (by omega), ← Except.ok.inj hg]
  exact hrel

/-- Every node's rows stand for its decoded arguments. -/
theorem rows_before {P : Prog} {d : Defn} {cs ts : Array Json} {types outs : Array Ty} {nds : Array Json}
    {rows : Array (Array (Array Json))} {out : Array Json}
    (hR : ∀ k < nds.size, NodeReads P types d outs k nds[k]!)
    (hS : Steps cs ts #[] #[] nds.toList rows out) : ∀ k ≤ nds.size, RowsBefore (leavesOf d outs) d rows k := by
  intro k
  induction k with
  | zero => intro _ k2 hk2; omega
  | succ k ih =>
    intro hk k2 hk2 x hx
    rcases Nat.lt_succ_iff_lt_or_eq.1 hk2 with hk2 | rfl
    · exact ih (by omega) k2 hk2 x hx
    obtain ⟨ajs, raw, -, hraw, hrs, hns, -, -, hA⟩ := node_core hR hS k2 (by omega) (ih (by omega))
    obtain ⟨-, qs, hrel, -, hseq, -⟩ := hA x (by omega)
    refine ⟨(rows[k2]!)[x]!, by rw [getElem!_pos _ x (by omega)]; exact Array.getElem?_eq_getElem _, ?_⟩
    rw [hseq]
    exact hrel

theorem partsRel_concat {L : Bool → Nat → Option Nat} {ps : List Json} {qs : List Part} (h : PartsRel L ps qs) :
    ∀ p ∈ ps, (p.getObjVal? "concat").toOption = none := by
  induction h with
  | nil => intro p hp; cases hp
  | cons _ hc _ _ ih =>
    intro p hp
    rcases List.mem_cons.1 hp with rfl | hp
    · exact hc
    · exact ih p hp

/-- Flat parts equivalent one by one parse to the same parts. -/
theorem partsRel_unique {L : Bool → Nat → Option Nat} : ∀ {ps : List Json} {qs : List Part}, PartsRel L ps qs →
    ∀ {ps' : List Json} {qs' : List Part}, PartsRel L ps' qs' → ps.length = ps'.length →
    (∀ i (h1 : i < ps.length) (h2 : i < ps'.length), Wf ps[i] ∧ Wf ps'[i] ∧ JsonEquiv ps[i] ps'[i]) → qs = qs'
  | _, _, .nil, _, _, .nil, _, _ => rfl
  | _, _, .nil, _, _, .cons .., h, _ => by simp at h
  | _, _, .cons .., _, _, .nil, h, _ => by simp at h
  | _, _, .cons ha hc hq hr, _, _, .cons _ _ hq' hr', hl, he => by
    obtain ⟨w, w', e⟩ := he 0 (by simp) (by simp)
    simp only [List.getElem_cons_zero] at w w' e
    unfold parsePart1 at hq hq'
    have h1 := part_jsonEquiv (n := 0) (m := 0) (args' := fun _ _ => none) w w' e ha hc hq
    rw [hq'] at h1
    cases h1
    rw [partsRel_unique hr hr' (by simpa using hl) fun i h1 h2 => by
      have := he (i + 1) (by simpa using h1) (by simpa using h2)
      simp only [List.getElem_cons_succ] at this
      exact this]

theorem buildNode_congr {fid : String} {k : Nat} {form : Flock.Program.Form} {c c' : Defn} {args : Array Arg}
    {axes : Array (Option Nat)} {n nxs : Nat} (h1 : c.key = c'.key) (h2 : c.params = c'.params) (h3 : c.ret = c'.ret) :
    buildNode fid k form c args axes n nxs = buildNode fid k form c' args axes n nxs := by
  obtain ⟨key, id, prim, params, ret, binding, body, gates, pl, ol⟩ := c
  obtain ⟨key', id', prim', params', ret', binding', body', gates', pl', ol'⟩ := c'
  simp only at h1 h2 h3
  subst h1 h2 h3
  rfl

theorem getObjVal_of_D {j v : Json} {k : String} (h : j.getObjValD k = v) (hv : v ≠ .null) :
    j.getObjVal? k = .ok v := by
  unfold Json.getObjValD at h
  cases hj : j.getObjVal? k with
  | ok w => rw [hj] at h; exact congrArg _ h
  | error e => rw [hj] at h; exact absurd h.symm hv

theorem getObjValAs_eq {j j' : Json} {k : String} (h : j.getObjValD k = j'.getObjValD k) :
    j.getObjValAs? String k = j'.getObjValAs? String k := by
  unfold Json.getObjValAs?; rw [h]

theorem wf_arr_get {a : Array Json} (h : Wf (.arr a)) {i : Nat} (hi : i < a.size) : Wf a[i]! := by
  cases h with
  | arr h => rw [getElem!_pos a i hi]; exact h _ (Array.getElem_mem hi)

/-- Node `k` of a composite as one side's canonical entry shows it. -/
structure NodeView (P : Prog) (d : Defn) (outs : Array Ty) (k : Nat) (cn : Json) : Prop where
  form : formOf ((cn.getObjValAs? String "form").toOption.getD "") = some (d.body.nodes[k]!).form
  axes : if (d.body.nodes[k]!).form == .batch then
      ∃ xs, cn.getObjVal? "axes" = .ok (.arr xs) ∧ xs.mapM axisOf = some (d.body.nodes[k]!).axes
    else (d.body.nodes[k]!).axes = #[]
  n : if (d.body.nodes[k]!).form == .batch || (d.body.nodes[k]!).form == .scan then
      (cn.getObjVal? "n").toOption.bind natJ? = some (d.body.nodes[k]!).n
    else (d.body.nodes[k]!).n = 1
  nxs : if (d.body.nodes[k]!).form == .scan then (cn.getObjVal? "n_xs").toOption.bind natJ? = some (d.body.nodes[k]!).nxs
    else (d.body.nodes[k]!).nxs = 0
  out : Ty.ofJson (cn.getObjValD "out") = .ok outs[k]!
  fn : cn.getObjValD "fn" = .str (P.defs[(d.body.nodes[k]!).fn]!).key
  args : ∃ (cargs : Array Json) (raw : Array Arg), cn.getObjValD "args" = .arr cargs ∧ raw.size = cargs.size ∧
    buildNode d.key k (d.body.nodes[k]!).form (P.defs[(d.body.nodes[k]!).fn]!) raw (d.body.nodes[k]!).axes
      (d.body.nodes[k]!).n (d.body.nodes[k]!).nxs = .ok ((d.body.nodes[k]!).out, (d.body.nodes[k]!).args) ∧
    ∀ a < cargs.size, Ty.ofJson ((cargs[a]!).getObjValD "type") = .ok (raw[a]!).ty ∧ (raw[a]!).dims = none ∧
      ∃ ps qs, (cargs[a]!).getObjValD "refs" = written ps ∧ (∀ p ∈ ps, Wf p) ∧
        PartsRel (leavesOf d outs) ps.toList qs.toList ∧ (raw[a]!).seq = Seq.ofParts qs
  offset : d.body.offsets[k + 1]! = d.body.offsets[k]! +
    (if (d.body.nodes[k]!).form == .prim then 1 else (d.body.nodes[k]!).members * (P.defs[(d.body.nodes[k]!).fn]!).gates)
  wf : Wf cn

/-- A composite Definition as one side's canonical entry shows it. -/
structure CompView (P : Prog) (d : Defn) (ce : Json) : Prop where
  prim : d.prim = false
  binding : d.binding = none
  form : ce.getObjValD "form" = .str "composite"
  id : d.id = (ce.getObjValAs? String "id").toOption.getD ""
  params : paramsOf ((ce.getObjValD "body").getObjValD "params") s!"definition '{d.key}'" = .ok d.params
  bparams : d.body.params = d.params
  ret : Ty.ofJson ((ce.getObjValD "body").getObjValD "ret_type") = .ok d.ret
  bret : d.body.retType = d.ret
  pleaves : d.paramLeaves = d.params.foldl (· + ·.2.leaves) 0
  oleaves : d.outLeaves = d.ret.leaves
  gates : d.gates = d.body.offsets.back!
  body : ∃ (outs : Array Ty) (cn : Array Json), (ce.getObjValD "body").getObjValD "nodes" = .arr cn ∧
    d.body.nodes.size = cn.size ∧ outs.size = cn.size ∧ d.body.offsets.size = cn.size + 1 ∧ d.body.offsets[0]! = 0 ∧
    (∀ k < cn.size, NodeView P d outs k cn[k]!) ∧
    ∃ ps qs, (ce.getObjValD "body").getObjValD "ret" = written ps ∧ (∀ p ∈ ps, Wf p) ∧
      PartsRel (leavesOf d outs) ps.toList qs.toList ∧ d.body.ret = Seq.ofParts qs

theorem canonNode_otherD {cs ts : Array Json} {n : Json} (hn : IsObj n) (args : Array Json)
    (row : Array (Array Json)) {k : String} (h1 : "fn" ≠ k) (h2 : "out" ≠ k) (h3 : "args" ≠ k) :
    (canonNode cs ts n args row).getObjValD k = n.getObjValD k := by
  unfold Json.getObjValD; rw [canonNode_other hn args row h1 h2 h3]

theorem getElem!_mem {α : Type} [Inhabited α] {a : Array α} {i : Nat} (h : i < a.size) : a[i]! ∈ a := by
  rw [getElem!_pos a i h]; exact Array.getElem_mem h

theorem mapM_get! {α β ε : Type} [Inhabited α] [Inhabited β] {f : α → Except ε β} {xs : Array α} {ys : Array β}
    (h : xs.mapM f = .ok ys) {i : Nat} (hi : i < xs.size) : f xs[i]! = .ok ys[i]! := by
  have hs := array_mapM_size _ _ _ h
  rw [getElem!_pos xs i hi, getElem!_pos ys i (by omega)]
  exact array_mapM_get _ _ _ h i hi (by omega)

theorem intJ_null : intJ? Json.null = none := rfl

theorem node_view {P : Prog} {d : Defn} {cs ts : Array Json} {types outs : Array Ty} {nds : Array Json}
    {rows : Array (Array (Array Json))} {out : Array Json} (htypes : ts.mapM Ty.ofJson = .ok types)
    (houts : nds.mapM (fun nd => at_ types (nd.getObjValD "out") "types") = .ok outs) (hnw : ∀ x ∈ nds, Wf x)
    (hR : ∀ k < nds.size, NodeReads P types d outs k nds[k]!)
    (hcal : ∀ k < nds.size, calleeAt cs nds[k]! = some (.str (P.defs[(d.body.nodes[k]!).fn]!).key))
    (hS : Steps cs ts #[] #[] nds.toList rows out) (hrw : RowsWf rows) (how : ∀ o ∈ out, Wf o)
    (k : Nat) (hk : k < nds.size) : NodeView P d outs k out[k]! := by
  obtain ⟨hrs0, hos0, -⟩ := steps_index hS
  simp only [Array.size_empty, Nat.zero_add, Array.length_toList] at hrs0 hos0
  obtain ⟨ajs, raw, hajs, hraw, hrs, hns, hout, hB, hA⟩ :=
    node_core hR hS k hk (rows_before hR hS k (Nat.le_of_lt hk))
  have hnd : Wf nds[k]! := hnw _ (getElem!_mem hk)
  have hno : IsObj nds[k]! := isObj_of_wf hnd hajs
  have hR' := hR k hk
  unfold NodeReads at hR'
  obtain ⟨hf, -, hax, hn, hnx, -, hoff⟩ := hR'
  have hwo : Wf out[k]! := how _ (getElem!_mem (by omega))
  rw [hout] at hwo ⊢
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, hoff, hwo⟩
  · rw [getObjValAs_eq (canonNode_otherD hno _ _ (by decide) (by decide) (by decide))]
    exact hf
  · rw [canonNode_other hno _ _ (k := "axes") (by decide) (by decide) (by decide)]
    exact hax
  · rw [canonNode_other hno _ _ (k := "n") (by decide) (by decide) (by decide)]
    exact hn
  · rw [canonNode_other hno _ _ (k := "n_xs") (by decide) (by decide) (by decide)]
    exact hnx
  · rw [canonNode_out hno]
    have := mapM_get! houts hk
    exact ofJson_atJ_of_at htypes this
  · rw [canonNode_fn hno]
    have hc := hcal k hk
    unfold calleeAt at hc
    unfold atJ
    cases hx : (nds[k]!.getObjValD "fn").getNat? with
    | error e => rw [hx] at hc; cases hc
    | ok x =>
      rw [hx] at hc
      simp only [Except.toOption, Option.bind_some] at hc
      simp only [Array.getD_eq_getD_getElem?, hc, Option.getD_some]
  · have hsz : ((ajs.zip rows[k]!).map (canonArg ts)).size = ajs.size := by simp [hrs]
    refine ⟨(ajs.zip rows[k]!).map (canonArg ts), raw, canonNode_args hno _ _, by rw [hsz, hraw], hB, fun a ha => ?_⟩
    rw [hsz] at ha
    have hel : ((ajs.zip rows[k]!).map (canonArg ts))[a]! = canonArg ts (ajs[a]!, (rows[k]!)[a]!) := by
      rw [getElem!_pos _ a (by rw [hsz]; exact ha), getElem!_pos ajs a ha, getElem!_pos rows[k]! a (by omega)]
      simp
    rw [hel]
    obtain ⟨⟨t, ht, h0, hlt, hty⟩, qs, hrel, hseq, -, hdim⟩ := hA a ha
    have hwa : Wf (nds[k]!.getObjValD "args") := wf_getObjValD hnd _
    rw [getObjValD_of hajs] at hwa
    have hao : IsObj ajs[a]! := isObj_of_wf (wf_arr_get hwa ha) (getObjVal_of_D rfl (by
      intro h; rw [h, intJ_null] at ht; cases ht))
    refine ⟨?_, hdim, (rows[k]!)[a]!, qs, canonArg_refs hao _, ?_, hrel, hseq⟩
    · rw [canonArg_type hao, hty]
      exact ofJson_atJ htypes (getNat_of_intJ ht h0) (by omega)
    · exact hrw _ (getElem!_mem (by omega)) _ (getElem!_mem (by omega))

theorem isComposite_of {e : Json} (h : e.getObjValD "form" = .str "composite") : isComposite e = true := by
  unfold isComposite; rw [h]; rfl

theorem comp_view {j : Json} {P : Prog} {i : Nat} {d : Defn} {ce : Json} (hj : Wf j) (hc : CompReads j P i d)
    (hce : canonicalEntry (tab j "callees") (tab j "types") (entry j d.key) = .ok ce) : CompView P d ce := by
  unfold CompReads at hc
  obtain ⟨hprim, hbind, hform, hid, hparams, hbp, hret, hbr, hpl, hol, hg, nds, types, outs, hnds, -, htypes, houts,
    hN, hOs, hO0, hnodes, qsr, hqr, hbret⟩ := hc
  simp only [getObjValD_eq] at hparams hret hnds hqr houts
  have he : Wf (entry j d.key) := wf_entry hj _
  have heo : IsObj (entry j d.key) := isObj_of_wf he (getObjVal_of_D hform (by intro h; cases h))
  have hb : Wf ((entry j d.key).getObjValD "body") := wf_getObjValD he _
  have hbo : IsObj ((entry j d.key).getObjValD "body") := isObj_of_wf hb hnds
  have hn := getObjValD_of hnds
  obtain ⟨rows, out, ret, hS, hret', rfl⟩ := canonicalEntry_comp (isComposite_of hform) hn hce
  have hnw : ∀ x ∈ nds, Wf x := by
    have := wf_getObjValD hb "nodes"
    rw [hn] at this
    cases this with | arr h => exact h
  have hcs := wf_tab hj "callees"
  have hts := wf_tab hj "types"
  obtain ⟨hrw, how⟩ := wf_steps hcs hts hS (fun r hr => by simp at hr) (fun o ho => by simp at ho)
    (fun n hn => hnw n (by simpa using hn))
  obtain ⟨hrs0, hos0, -⟩ := steps_index hS
  simp only [Array.size_empty, Nat.zero_add, Array.length_toList] at hrs0 hos0
  have hR : ∀ k < nds.size, NodeReads P types d outs k nds[k]! := fun k hk => (hnodes k hk).2.2
  have hcal : ∀ k < nds.size, calleeAt (tab j "callees") nds[k]! = some (.str (P.defs[(d.body.nodes[k]!).fn]!).key) :=
    fun k hk => (hnodes k hk).2.1
  have hbody : (setJ (entry j d.key) "body" (setJ (setJ ((entry j d.key).getObjValD "body") "nodes" (.arr out)) "ret"
      (written ret))).getObjValD "body" = setJ (setJ ((entry j d.key).getObjValD "body") "nodes" (.arr out)) "ret"
      (written ret) := getObjValD_setJ_eq heo _ _
  have hbo' := isObj_setJ hbo "nodes" (.arr out)
  refine ⟨hprim, hbind, ?_, ?_, ?_, hbp, ?_, hbr, hpl, hol, hg, outs, out, ?_, by omega,
    by rw [array_mapM_size _ _ _ houts]; omega, by omega, hO0, fun k hk => ?_, ret, qsr, ?_, ?_, ?_, hbret⟩
  · rw [getObjValD_setJ_ne heo _ (by decide)]; exact hform
  · rw [getObjValAs_eq (getObjValD_setJ_ne heo _ (by decide))]; exact hid
  · rw [hbody, getObjValD_setJ_ne hbo' _ (by decide), getObjValD_setJ_ne hbo _ (by decide)]; exact hparams
  · rw [hbody, getObjValD_setJ_ne hbo' _ (by decide), getObjValD_setJ_ne hbo _ (by decide)]; exact hret
  · rw [hbody, getObjValD_setJ_ne hbo' _ (by decide), getObjValD_setJ_eq hbo]
  · exact node_view htypes houts hnw hR hcal hS hrw how k (by omega)
  · rw [hbody, getObjValD_setJ_eq hbo']
  · exact wf_flatParts hrw _ _ _ (wf_getObjValD hb _) hret'
  · have hRB := rows_before hR hS nds.size (Nat.le_refl _)
    have H : RowsArgs (leavesOf d outs) rows (fun k2 x => d.body.nodes[k2]?.bind fun y => y.args[x]?.map (·.seq)) := by
      intro k2 x s hs
      obtain ⟨y, hy, hs⟩ := Option.bind_eq_some_iff.1 hs
      obtain ⟨z, hz, rfl⟩ := Option.map_eq_some_iff.1 hs
      have hk2 : k2 < d.body.nodes.size := (Array.getElem?_eq_some_iff.1 hy).1
      have hy' : y = d.body.nodes[k2]! := by
        rw [getElem!_pos _ k2 hk2]; exact ((Array.getElem?_eq_some_iff.1 hy).2).symm
      subst hy'
      have hx : x < (d.body.nodes[k2]!).args.size := (Array.getElem?_eq_some_iff.1 hz).1
      have hz' : z = (d.body.nodes[k2]!).args[x]! := by
        rw [getElem!_pos _ x hx]; exact ((Array.getElem?_eq_some_iff.1 hz).2).symm
      subst hz'
      obtain ⟨ps, hps, hrel⟩ := hRB k2 (by omega) x hx
      refine ⟨ps, ?_, hrel⟩
      rw [Array.getElem?_eq_getElem (by omega), Option.bind_some, ← getElem!_pos rows k2 (by omega)]
      exact hps
    unfold parseParts at hqr
    obtain ⟨fps, hf, hrel⟩ := flat_of_parse H _ _ _ hqr
    rw [hret'] at hf
    cases hf
    exact hrel

theorem same_fields {P P' : Prog} {a b : Nat} (h : Same P P' a b) :
    (P.defs[a]!).key = (P'.defs[b]!).key ∧ (P.defs[a]!).params = (P'.defs[b]!).params ∧
      (P.defs[a]!).ret = (P'.defs[b]!).ret ∧ (P.defs[a]!).gates = (P'.defs[b]!).gates := by
  cases h with
  | intro _ _ d d' h1 h2 hkey _ _ hpar hret _ hg =>
    rw [getElem!_def, h1, getElem!_def, h2]
    exact ⟨hkey, hpar, hret, hg⟩

theorem optEquiv_natJ {x y : Option Json} (h : OptEquiv x y) : x.bind natJ? = y.bind natJ? := by
  cases x <;> cases y <;> first | rfl | exact h.elim | exact natJ_jsonEquiv h

theorem jsonEquiv_null {b : Json} (h : JsonEquiv .null b) : b = .null := by
  cases b <;> simp_all [JsonEquiv, jsonForm]

theorem axisOf_jsonEquiv {a b : Json} (h : JsonEquiv a b) : axisOf a = axisOf b := by
  cases a with
  | null => rw [jsonEquiv_null h]
  | _ =>
    cases b with
    | null => exact absurd (jsonEquiv_null h.symm) (by intro h; cases h)
    | _ => simp only [axisOf]; rw [natJ_jsonEquiv h]

theorem mapM_axisOf {xs xs' : Array Json} (h : JsonEquiv (.arr xs) (.arr xs')) : xs.mapM axisOf = xs'.mapM axisOf := by
  obtain ⟨hs, he⟩ := (jsonEquiv_arr_iff xs xs').1 h
  rw [Array.mapM_eq_mapM_toList, Array.mapM_eq_mapM_toList]
  congr 1
  clear h
  obtain ⟨l⟩ := xs
  obtain ⟨l'⟩ := xs'
  simp only [List.size_toArray] at hs he ⊢
  induction l generalizing l' with
  | nil => cases l' with | nil => rfl | cons => simp at hs
  | cons a l ih =>
    cases l' with
    | nil => simp at hs
    | cons a' l' =>
      simp only [List.mapM_cons]
      rw [axisOf_jsonEquiv (by simpa using he 0 (by simp) (by simp)),
        ih l' (by simpa using hs) fun i h1 h2 => by simpa using he (i + 1) (by simpa using h1) (by simpa using h2)]

theorem arr_ext! {α : Type} [Inhabited α] {a b : Array α} (hs : a.size = b.size) (h : ∀ i < a.size, a[i]! = b[i]!) :
    a = b := by
  refine Array.ext hs fun i h1 h2 => ?_
  have := h i h1
  rwa [getElem!_pos a i h1, getElem!_pos b i h2] at this

theorem jsonEquiv_arr_get {xs xs' : Array Json} (h : JsonEquiv (.arr xs) (.arr xs')) :
    xs.size = xs'.size ∧ ∀ i < xs.size, JsonEquiv xs[i]! xs'[i]! := by
  obtain ⟨hs, he⟩ := (jsonEquiv_arr_iff xs xs').1 h
  refine ⟨hs, fun i hi => ?_⟩
  rw [getElem!_pos xs i hi, getElem!_pos xs' i (by omega)]
  exact he i hi (by omega)

/-- Equivalent flat sequences of parts related to `qs` and `qs'` give one sequence. -/
theorem written_parts {L : Bool → Nat → Option Nat} {ps ps' : Array Json} {qs qs' : Array Part}
    (hw : ∀ p ∈ ps, Wf p) (hw' : ∀ p ∈ ps', Wf p) (hr : PartsRel L ps.toList qs.toList)
    (hr' : PartsRel L ps'.toList qs'.toList) (h : JsonEquiv (written ps) (written ps')) : qs = qs' := by
  obtain ⟨hs, he⟩ := written_jsonEquiv hw hw'
    (fun p hp => partsRel_concat hr p (Array.mem_toList_iff.2 hp))
    (fun p hp => partsRel_concat hr' p (Array.mem_toList_iff.2 hp)) h
  have := partsRel_unique hr hr' (by simpa using hs) fun i h1 h2 => by
    simp only [Array.getElem_toList]
    have h1' : i < ps.size := by simpa using h1
    have h2' : i < ps'.size := by simpa using h2
    exact ⟨hw _ (Array.getElem_mem h1'), hw' _ (Array.getElem_mem h2'), he i h1' h2'⟩
  exact Array.toList_inj.1 this

theorem arg_ext {x y : Arg} (h1 : x.ty = y.ty) (h2 : x.seq = y.seq) (h3 : x.dims = y.dims) : x = y := by
  cases x; cases y; simp_all

theorem node_out_eq {P P' : Prog} {d d' : Defn} {outs outs' : Array Ty} {k : Nat} {cn cn' : Json}
    (V : NodeView P d outs k cn) (V' : NodeView P' d' outs' k cn') (he : JsonEquiv cn cn') : outs[k]! = outs'[k]! := by
  have := ofJson_jsonEquiv V.out (fieldD_jsonEquiv V.wf V'.wf he "out")
  rw [V'.out] at this
  exact (Except.ok.inj this).symm

theorem node_same {P P' : Prog} {d d' : Defn} {outs outs' : Array Ty} {k : Nat} {cn cn' : Json}
    (hkey : d.key = d'.key) (hL : leavesOf d outs = leavesOf d' outs')
    (V : NodeView P d outs k cn) (V' : NodeView P' d' outs' k cn') (he : JsonEquiv cn cn')
    (hsame : (P.defs[(d.body.nodes[k]!).fn]!).key = (P'.defs[(d'.body.nodes[k]!).fn]!).key →
      Same P P' (d.body.nodes[k]!).fn (d'.body.nodes[k]!).fn) :
    ({ d.body.nodes[k]! with fn := 0 } : Node) = { d'.body.nodes[k]! with fn := 0 } ∧
      Same P P' (d.body.nodes[k]!).fn (d'.body.nodes[k]!).fn ∧
      (if (d.body.nodes[k]!).form == .prim then 1 else (d.body.nodes[k]!).members * (P.defs[(d.body.nodes[k]!).fn]!).gates)
        = (if (d'.body.nodes[k]!).form == .prim then 1
          else (d'.body.nodes[k]!).members * (P'.defs[(d'.body.nodes[k]!).fn]!).gates) := by
  have hw := V.wf
  have hw' := V'.wf
  -- the callee
  have hfn := fieldD_jsonEquiv hw hw' he "fn"
  rw [V.fn, V'.fn] at hfn
  have hck := (str_of_jsonEquiv hfn).symm
  simp only [Json.str.injEq] at hck
  have hS := hsame hck
  obtain ⟨-, hpar, hret, hgates⟩ := same_fields hS
  -- the form
  have hform : (d.body.nodes[k]!).form = (d'.body.nodes[k]!).form := by
    have h1 := V.form
    have h2 := V'.form
    rw [fieldStr_eq hw hw' he "form", h2] at h1
    exact (Option.some.inj h1).symm
  -- the counts and axes
  have hn : (d.body.nodes[k]!).n = (d'.body.nodes[k]!).n := by
    have h1 := V.n
    have h2 := V'.n
    rw [optEquiv_natJ (field_jsonEquiv hw hw' he "n")] at h1
    rw [← hform] at h2
    split at h1 <;> rename_i hc <;> simp only [hc, ↓reduceIte] at h2
    · rw [h1] at h2; exact Option.some.inj h2
    · rw [h1, h2]
  have hnxs : (d.body.nodes[k]!).nxs = (d'.body.nodes[k]!).nxs := by
    have h1 := V.nxs
    have h2 := V'.nxs
    rw [optEquiv_natJ (field_jsonEquiv hw hw' he "n_xs")] at h1
    rw [← hform] at h2
    split at h1 <;> rename_i hc <;> simp only [hc, ↓reduceIte] at h2
    · rw [h1] at h2; exact Option.some.inj h2
    · rw [h1, h2]
  have hax : (d.body.nodes[k]!).axes = (d'.body.nodes[k]!).axes := by
    have h1 := V.axes
    have h2 := V'.axes
    rw [← hform] at h2
    split at h1 <;> rename_i hc <;> simp only [hc, ↓reduceIte] at h2
    · obtain ⟨xs, hx, hm⟩ := h1
      obtain ⟨xs', hx', hm'⟩ := h2
      have := field_jsonEquiv hw hw' he "axes"
      rw [hx, hx'] at this
      rw [mapM_axisOf this, hm'] at hm
      exact (Option.some.inj hm).symm
    · rw [h1, h2]
  -- the arguments
  obtain ⟨cargs, raw, hca, hraw, hB, hA⟩ := V.args
  obtain ⟨cargs', raw', hca', hraw', hB', hA'⟩ := V'.args
  have hargs := fieldD_jsonEquiv hw hw' he "args"
  rw [hca, hca'] at hargs
  obtain ⟨hcs, hce⟩ := jsonEquiv_arr_get hargs
  have hwa : ∀ a < cargs.size, Wf cargs[a]! := fun a ha => by
    have := wf_getObjValD hw "args"; rw [hca] at this; exact wf_arr_get this ha
  have hwa' : ∀ a < cargs'.size, Wf cargs'[a]! := fun a ha => by
    have := wf_getObjValD hw' "args"; rw [hca'] at this; exact wf_arr_get this ha
  have hraweq : raw = raw' := by
    refine arr_ext! (by omega) fun a ha => ?_
    obtain ⟨ht, hd, ps, qs, hr, hpw, hrel, hseq⟩ := hA a (by omega)
    obtain ⟨ht', hd', ps', qs', hr', hpw', hrel', hseq'⟩ := hA' a (by omega)
    have e := hce a (by omega)
    have hty := ofJson_jsonEquiv ht (fieldD_jsonEquiv (hwa a (by omega)) (hwa' a (by omega)) e "type")
    rw [ht'] at hty
    have hty := (Except.ok.inj hty).symm
    have hrefs := fieldD_jsonEquiv (hwa a (by omega)) (hwa' a (by omega)) e "refs"
    rw [hr, hr'] at hrefs
    rw [hL] at hrel
    have hq := written_parts hpw hpw' hrel hrel' hrefs
    subst hq
    exact arg_ext hty (hseq.trans hseq'.symm) (hd.trans hd'.symm)
  have hB'' : buildNode d.key k (d.body.nodes[k]!).form (P.defs[(d.body.nodes[k]!).fn]!) raw (d.body.nodes[k]!).axes
      (d.body.nodes[k]!).n (d.body.nodes[k]!).nxs = .ok ((d'.body.nodes[k]!).out, (d'.body.nodes[k]!).args) := by
    rw [buildNode_congr hck hpar hret, hkey, hform, hax, hn, hnxs, hraweq]
    exact hB'
  rw [hB] at hB''
  obtain ⟨hout, hargs⟩ := Prod.mk.inj (Except.ok.inj hB'')
  refine ⟨?_, hS, ?_⟩
  · simp only [hform, hax, hn, hnxs, hout, hargs]
  · simp only [Node.members, hform, hn, hgates]

theorem same_comp {P P' : Prog} {i i' : Nat} {d d' : Defn} {ce ce' : Json}
    (hd : P.defs[i]? = some d) (hd' : P'.defs[i']? = some d') (hkey : d.key = d'.key)
    (V : CompView P d ce) (V' : CompView P' d' ce') (hw : Wf ce) (hw' : Wf ce') (he : JsonEquiv ce ce')
    (hcal : ∀ n < d.body.nodes.size, n < d'.body.nodes.size →
      (P.defs[(d.body.nodes[n]!).fn]!).key = (P'.defs[(d'.body.nodes[n]!).fn]!).key →
      Same P P' (d.body.nodes[n]!).fn (d'.body.nodes[n]!).fn) : Same P P' i i' := by
  have hid : d.id = d'.id := by rw [V.id, V'.id, fieldStr_eq hw hw' he "id"]
  have hb := fieldD_jsonEquiv hw hw' he "body"
  have hbw := wf_getObjValD hw "body"
  have hbw' := wf_getObjValD hw' "body"
  have hpar : d.params = d'.params := by
    have := paramsOf_jsonEquiv V.params (fieldD_jsonEquiv hbw hbw' hb "params")
    rw [hkey, V'.params] at this
    exact (Except.ok.inj this).symm
  have hret : d.ret = d'.ret := by
    have := ofJson_jsonEquiv V.ret (fieldD_jsonEquiv hbw hbw' hb "ret_type")
    rw [V'.ret] at this
    exact (Except.ok.inj this).symm
  obtain ⟨outs, cn, hcn, hN, hO, hOs, hO0, hV, ps, qs, hr, hpw, hrel, hbret⟩ := V.body
  obtain ⟨outs', cn', hcn', hN', hO', hOs', hO0', hV', ps', qs', hr', hpw', hrel', hbret'⟩ := V'.body
  have hnodes := fieldD_jsonEquiv hbw hbw' hb "nodes"
  rw [hcn, hcn'] at hnodes
  obtain ⟨hcs, hce⟩ := jsonEquiv_arr_get hnodes
  have houts : outs = outs' :=
    arr_ext! (by omega) fun k hk => node_out_eq (hV k (by omega)) (hV' k (by omega)) (hce k (by omega))
  subst houts
  have hL : leavesOf d outs = leavesOf d' outs := by unfold leavesOf; rw [hpar]
  have hns := fun k (hk : k < cn.size) =>
    node_same hkey hL (hV k hk) (hV' k (by omega)) (hce k hk) (hcal k (by omega) (by omega))
  have hoff : d.body.offsets = d'.body.offsets := by
    have : ∀ k ≤ cn.size, d.body.offsets[k]! = d'.body.offsets[k]! := by
      intro k
      induction k with
      | zero => intro _; rw [hO0, hO0']
      | succ k ih =>
        intro hk
        rw [(hV k (by omega)).offset, (hV' k (by omega)).offset, ih (by omega), (hns k (by omega)).2.2]
    exact arr_ext! (by omega) fun k hk => this k (by omega)
  have hbr : d.body.ret = d'.body.ret := by
    have hrets := fieldD_jsonEquiv hbw hbw' hb "ret"
    rw [hr, hr'] at hrets
    rw [hL] at hrel
    rw [hbret, hbret', written_parts hpw hpw' hrel hrel' hrets]
  exact .intro i i' d d' hd hd' hkey hid (V.prim.trans V'.prim.symm) hpar hret (V.binding.trans V'.binding.symm)
    (by rw [V.gates, V'.gates, hoff]) (by rw [V.pleaves, V'.pleaves, hpar]) (by rw [V.oleaves, V'.oleaves, hret])
    (by rw [V.bparams, V'.bparams, hpar]) (by rw [V.bret, V'.bret, hret]) hbr hoff (by omega)
    (fun n hn => (hns n (by omega)).1) (fun n hn => (hns n (by omega)).2.1)

theorem form_of_prim {j : Json} {d : Defn} (hp : PrimReads j d) :
    (entry j d.key).getObjValD "form" = .str "primitive" := by
  unfold PrimReads at hp
  exact str_of_field hp.2.2.1 (by decide)

theorem same_prim {j j' : Json} {P P' : Prog} {i i' : Nat} {d d' : Defn} (hj : Wf j) (hj' : Wf j')
    (hd : P.defs[i]? = some d) (hd' : P'.defs[i']? = some d') (hkey : d.key = d'.key)
    (hp : PrimReads j d) (hp' : PrimReads j' d') (he : JsonEquiv (entry j d.key) (entry j' d'.key)) :
    Same P P' i i' := by
  have hw := wf_entry hj d.key
  have hw' := wf_entry hj' d'.key
  unfold PrimReads at hp hp'
  obtain ⟨hprim, hbody, -, hid, hpar, hret, hbind, hg, hpl, hol⟩ := hp
  obtain ⟨hprim', hbody', -, hid', hpar', hret', hbind', hg', hpl', hol'⟩ := hp'
  simp only [getObjValD_eq] at hpar hret hpar' hret'
  have hpar2 : d.params = d'.params := by
    have := paramsOf_jsonEquiv hpar (fieldD_jsonEquiv hw hw' he "params")
    rw [hkey, hpar'] at this
    exact (Except.ok.inj this).symm
  have hret2 : d.ret = d'.ret := by
    have := ofJson_jsonEquiv hret (fieldD_jsonEquiv hw hw' he "ret")
    rw [hret'] at this
    exact (Except.ok.inj this).symm
  exact .intro i i' d d' hd hd' hkey (by rw [hid, hid', fieldStr_eq hw hw' he "id"]) (hprim.trans hprim'.symm) hpar2
    hret2 (by rw [hbind, hbind', hkey]) (hg.trans hg'.symm) (by rw [hpl, hpl', hpar2]) (hol.trans hol'.symm)
    (by rw [hbody, hbody']) (by rw [hbody, hbody']) (by rw [hbody, hbody']) (by rw [hbody, hbody'])
    (by rw [hbody, hbody']) (fun n hn => by rw [hbody] at hn; exact absurd hn (Nat.not_lt_zero _))
    (fun n hn => by rw [hbody] at hn; exact absurd hn (Nat.not_lt_zero _))

theorem not_prim_comp {j j' : Json} {P' : Prog} {i' : Nat} {d d' : Defn} {ce' : Json} (hj : Wf j) (hj' : Wf j')
    (hp : PrimReads j d) (hc' : CompReads j' P' i' d')
    (hce' : canonicalEntry (tab j' "callees") (tab j' "types") (entry j' d'.key) = .ok ce')
    (he : JsonEquiv (entry j d.key) ce') : False := by
  have V' := comp_view hj' hc' hce'
  have hw' := wf_canonicalEntry (wf_tab hj' _) (wf_tab hj' _) (wf_entry hj' _) hce'
  have := fieldD_jsonEquiv (wf_entry hj _) hw' he "form"
  rw [form_of_prim hp, V'.form] at this
  have := str_of_jsonEquiv this
  simp only [Json.str.injEq] at this
  exact absurd this (by decide)

end ReadsPinProof

open ReadsPinProof in
/-- **`ReadsPin` holds**: equivalent canonical entries of one key, each read as `DefReads` says, with callees of one
key one Definition, give one Definition. -/
theorem readsPin : ReadsPin := by
  intro j j' P P' i i' ce ce' hj hj' hi hi' hkey hr hr' hce hce' he hcal
  have hd : P.defs[i]? = some P.defs[i]! := by rw [getElem!_pos _ i hi]; exact Array.getElem?_eq_getElem hi
  have hd' : P'.defs[i']? = some P'.defs[i']! := by rw [getElem!_pos _ i' hi']; exact Array.getElem?_eq_getElem hi'
  have hw := wf_canonicalEntry (wf_tab hj _) (wf_tab hj _) (wf_entry hj _) hce
  have hw' := wf_canonicalEntry (wf_tab hj' _) (wf_tab hj' _) (wf_entry hj' _) hce'
  rcases hr.2 with hp | hc <;> rcases hr'.2 with hp' | hc'
  · rw [canonicalEntry_of_not (by unfold isComposite; rw [form_of_prim hp]; rfl)] at hce
    rw [canonicalEntry_of_not (by unfold isComposite; rw [form_of_prim hp']; rfl)] at hce'
    cases hce
    cases hce'
    exact same_prim hj hj' hd hd' hkey hp hp' he
  · rw [canonicalEntry_of_not (by unfold isComposite; rw [form_of_prim hp]; rfl)] at hce
    cases hce
    exact (not_prim_comp hj hj' hp hc' hce' he).elim
  · rw [canonicalEntry_of_not (by unfold isComposite; rw [form_of_prim hp']; rfl)] at hce'
    cases hce'
    exact (not_prim_comp hj' hj hp' hc hce he.symm).elim
  · exact same_comp hd hd' hkey (comp_view hj hc hce) (comp_view hj' hc' hce') hw hw' he hcal

end InferenceOnlyVerifier.Properties.Record.SecurityProofs

end
