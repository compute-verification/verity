module

public import Proofs.Properties.Flock.Circuit.LowerEval.Axes

/-! `LiftWordEval` for a template of batches on any axes over word primitives, gates and earlier such templates
(`BatchNested`): by strong induction on the reached composite, the lift's copy of each evaluates on bits, at every fuel
past its place, as the composite does on words. -/

@[expose] public section

namespace Flock.Chain

open Flock Flock.Lower Flock.Tmpl Flock.TmplValue Flock.Chain.LowerDef
open Flock.Program hiding Form
open InferenceOnlyVerifier.Properties.SecurityProofs

/-- A Definition reached from `d` in a decoded C is placed in C, and reached from `d` in a Program that keeps C's
Definitions. -/
theorem reach_restrict {bytes : ByteArray} {P S : Prog} (h : decode bytes = .ok P) (hr : (P.defs[P.root]!).body.recut = #[])
    (hpre : ∀ i < P.defs.size, S.defs[i]! = P.defs[i]!) {d y : ℕ} (hd : d < P.defs.size) (hy : Reach P d y) :
    y < P.defs.size ∧ Reach S d y := by
  obtain ⟨st, hP, hinv, -⟩ := BodiesOk.decode_spec bytes P h hr
  have hok : ∀ i < P.defs.size, BodiesOk.DefOk P.defs i P.defs[i]! := by rw [hP]; exact hinv.1
  induction hy with
  | root => exact ⟨hd, .root⟩
  | @callee y nd _ hp hn ih =>
    obtain ⟨hyl, hS⟩ := ih
    obtain ⟨n, hn', rfl⟩ := Array.mem_iff_getElem.1 hn
    have hlt : ((P.defs[y]!).body.nodes[n]).fn < y := by
      rw [← getElem!_pos _ n hn']; exact ((hok y hyl).2.1 n hn').1
    exact ⟨by omega, .callee hS (by rw [hpre y hyl]; exact hp) (by rw [hpre y hyl]; exact Array.getElem_mem hn')⟩

/-- A Definition reached from `d` in a Program that keeps a decoded C's Definitions is placed in C and reached from `d`
in C: the walk over the keeping Program reaches no Definition C's walk does not. -/
theorem reach_of_restrict {bytes : ByteArray} {P S : Prog} (h : decode bytes = .ok P)
    (hr : (P.defs[P.root]!).body.recut = #[]) (hpre : ∀ i < P.defs.size, S.defs[i]! = P.defs[i]!) {d y : ℕ}
    (hd : d < P.defs.size) (hy : Reach S d y) : y < P.defs.size ∧ Reach P d y := by
  obtain ⟨st, hP, hinv, -⟩ := BodiesOk.decode_spec bytes P h hr
  have hok : ∀ i < P.defs.size, BodiesOk.DefOk P.defs i P.defs[i]! := by rw [hP]; exact hinv.1
  induction hy with
  | root => exact ⟨hd, .root⟩
  | @callee y nd _ hp hn ih =>
    obtain ⟨hyl, hR⟩ := ih
    rw [hpre y hyl] at hp hn
    obtain ⟨n, hn', rfl⟩ := Array.mem_iff_getElem.1 hn
    have hlt : ((P.defs[y]!).body.nodes[n]).fn < y := by
      rw [← getElem!_pos _ n hn']; exact ((hok y hyl).2.1 n hn').1
    exact ⟨by omega, .callee hR hp (Array.getElem_mem hn')⟩

theorem reach_trans {S : Prog} {d y z : ℕ} (h1 : Reach S d y) (h2 : Reach S y z) : Reach S d z := by
  induction h2 with
  | root => exact h1
  | callee _ hp hn ih => exact .callee ih hp hn

/-- **The typing facts a nested template's walk reads**, for composite `d` of C: `Batch0Typed`'s, its return's leaf
widths, and for a node whose callee is a composite, its output the callee's return repeated over its members and its
arguments of the callee's parameter leaf widths. -/
def NestedFacts (P : Prog) (d : ℕ) (D : Defn) : Prop :=
  (∀ k (hk : k < D.body.nodes.size), ∀ a ∈ (D.body.nodes[k]).args.toList,
    a.seq.count = a.seq.toArray.size ∧ ∀ r ∈ a.seq.toArray.toList, RefIn D k r) ∧
  (D.body.ret.count = D.body.ret.toArray.size ∧ ∀ r ∈ D.body.ret.toArray.toList, RefIn D D.body.nodes.size r) ∧
  D.body.ret.toArray.toList.map (fun r => (tyAt D r).widths[r.j]!) = D.ret.widths.toList ∧
  (∀ nd ∈ D.body.nodes, nd.fn < d) ∧
  (∀ k (hk : k < D.body.nodes.size), (P.defs[(D.body.nodes[k]).fn]!).prim = true → ∀ ys : List (List Bool),
    ys.length = (D.body.nodes[k]).members →
    (D.body.nodes[k]).out.widths.toList = ys.map fun y => (Prog.evalBool P (D.body.nodes[k]).fn y).length) ∧
  (∀ k (hk : k < D.body.nodes.size), (P.defs[(D.body.nodes[k]).fn]!).prim = false → (D.body.nodes[k]).form ≠ .scan →
    (D.body.nodes[k]).out.widths.toList =
      (List.replicate (D.body.nodes[k]).members (P.defs[(D.body.nodes[k]).fn]!).ret.widths.toList).flatten) ∧
  (∀ k (hk : k < D.body.nodes.size), (D.body.nodes[k]).form = .batch →
    ∀ i < (D.body.nodes[k]).args.size,
      (∀ ax, (D.body.nodes[k]).axes[i]? = some (some ax) →
        ((D.body.nodes[k]).args.getD i default).ty.removeAxis ax =
          some ((D.body.nodes[k]).n, ((P.defs[(D.body.nodes[k]).fn]!).params.getD i default).2) ∧
        ((D.body.nodes[k]).args.getD i default).dims = ((D.body.nodes[k]).args.getD i default).ty.axisDims ax ∧
        i < (P.defs[(D.body.nodes[k]).fn]!).params.size ∧
        ((D.body.nodes[k]).args.getD i default).seq.toArray.toList.map (fun r => (tyAt D r).widths[r.j]!) =
          ((D.body.nodes[k]).args.getD i default).ty.widths.toList) ∧
      ((∀ ax, (D.body.nodes[k]).axes[i]? ≠ some (some ax)) → ((D.body.nodes[k]).args.getD i default).dims = none)) ∧
  (∀ k (hk : k < D.body.nodes.size), (P.defs[(D.body.nodes[k]).fn]!).prim = false →
    ((D.body.nodes[k]).form = .prim ∨ (D.body.nodes[k]).form = .call) →
    ((D.body.nodes[k]).args.toList.flatMap fun a => a.seq.toArray.toList).map (fun r => (tyAt D r).widths[r.j]!) =
      leafWidths (P.defs[(D.body.nodes[k]).fn]!)) ∧
  (∀ k (hk : k < D.body.nodes.size), (P.defs[(D.body.nodes[k]).fn]!).prim = false → (D.body.nodes[k]).form = .batch →
    (D.body.nodes[k]).args.size = (P.defs[(D.body.nodes[k]).fn]!).params.size ∧
    ∀ i < (D.body.nodes[k]).args.size, (∀ ax, (D.body.nodes[k]).axes[i]? ≠ some (some ax)) →
      ((D.body.nodes[k]).args.getD i default).seq.toArray.toList.map (fun r => (tyAt D r).widths[r.j]!) =
        ((P.defs[(D.body.nodes[k]).fn]!).params.getD i default).2.widths.toList) ∧
  (∀ k (hk : k < D.body.nodes.size), (D.body.nodes[k]).form = .scan →
    (P.defs[(D.body.nodes[k]).fn]!).prim = false ∧
    (D.body.nodes[k]).args.size = (P.defs[(D.body.nodes[k]).fn]!).params.size ∧
    0 < (P.defs[(D.body.nodes[k]).fn]!).params.size ∧
    ((D.body.nodes[k]).args.getD 0 default).seq.toArray.toList.map (fun r => (tyAt D r).widths[r.j]!) =
      ((P.defs[(D.body.nodes[k]).fn]!).params.getD 0 default).2.widths.toList ∧
    (∀ i < (D.body.nodes[k]).args.size, 1 ≤ i →
      ((D.body.nodes[k]).args.getD i default).seq.toArray.toList.map (fun r => (tyAt D r).widths[r.j]!) =
        if i ≤ (D.body.nodes[k]).nxs then
          (List.replicate (D.body.nodes[k]).n ((P.defs[(D.body.nodes[k]).fn]!).params.getD i default).2.widths.toList).flatten
        else ((P.defs[(D.body.nodes[k]).fn]!).params.getD i default).2.widths.toList) ∧
    ∃ yw, (P.defs[(D.body.nodes[k]).fn]!).ret.widths.toList =
        ((P.defs[(D.body.nodes[k]).fn]!).params.getD 0 default).2.widths.toList ++ yw ∧
      (D.body.nodes[k]).out.widths.toList =
        ((P.defs[(D.body.nodes[k]).fn]!).params.getD 0 default).2.widths.toList ++
          (List.replicate (D.body.nodes[k]).n yw).flatten)

/-- **`NestedTyped`**: in a C `decode` accepts whose root has no recut, each template of batches and scans over word
primitives and earlier such templates (`ScanNested`), each of whose reached Boolean Definitions evaluates through gates
alone, has `NestedFacts`. -/
def NestedTyped : Prop :=
  ∀ (bytes : ByteArray) (P : Prog), decode bytes = .ok P → (P.defs[P.root]!).body.recut = #[] →
    ∀ d < P.defs.size, ScanNested P d → GateBodiesDeep P d → NestedFacts P d (P.defs[d]!)

/-- `boolCall` at any fuel past the Boolean Definition's place. -/
theorem boolCall_at {bytes : ByteArray} {P L : Prog} (h : decode bytes = .ok P) (hr : (P.defs[P.root]!).body.recut = #[])
    (hle : P.defs.size ≤ L.defs.size) (hpre : ∀ i < P.defs.size, L.defs[i]! = P.defs[i]!)
    (hbody : ∀ k, L.bodyOf? k = P.bodyOf? k) {f b : ℕ} (hf : f < P.defs.size)
    (hw : isWordPrim (P.defs[f]!).prim (P.defs[f]!).key = true) (hb : P.bodyOf? (P.defs[f]!).key = some b)
    (hs : ∀ y, (Flock.Program.evalDef? P (b + 1) b y).isSome = true) {m : ℕ} (hm : b < m) (y : List Bool) :
    Flock.Program.evalDef? L m b y = some (Prog.evalBool P f y) := by
  obtain ⟨⟨hmap, hkeys⟩, -⟩ := FlockSoundness.DecodeBoolean.decode_facts bytes P h hr
  have hm' : P.BoolMapOk := ⟨hmap, hkeys⟩
  have hbl := body_lt hm' hb
  have hbd : P.BoolDef b := by
    obtain ⟨b', -, hbo, hbody', -⟩ := hmap f hf hw
    obtain rfl : b' = b := Option.some.inj (hbo.symm.trans hb)
    exact hbody'
  have hB := FlockSoundness.DecodeBoolean.bool_none hkeys
  have hBL : ∀ k, isBooleanPrim k = true → L.bodyOf? k = none := fun k hk => by rw [hbody]; exact hB k hk
  have hbdL : L.BoolDef b := boolDef_lift hpre hbd hbl
  rw [evalDef_fuel_at L hBL hbdL hm y, ← evalDef_fuel_at L hBL hbdL (show b < 2 * L.defs.size + 1 by omega) y]
  exact boolCall h hr hle hpre hbody hf hw hb hs y

theorem refWords_length {D : Defn} {x : List (List Bool)} (hx : x.map List.length = leafWidths D)
    {outs : Array (List (List Bool))} (ho : OutsOk D outs) (rs : List Ref) (hr : ∀ r ∈ rs, RefIn D outs.size r) :
    (rs.map (Flock.Program.refVal x (Flock.Program.paramStarts D.params) outs)).map List.length =
      rs.map fun r => (tyAt D r).widths[r.j]! := by
  rw [List.map_map]
  exact List.map_congr_left fun r h => (refBits hx ho (hr r h)).2

theorem argVals_count {W : Type} [Inhabited W] (v : Ref → W) (a : Arg) (hc : a.seq.count = a.seq.toArray.size) :
    Flock.Program.argVals v a a.seq.count id = a.seq.toArray.toList.map v := by
  simp only [Flock.Program.argVals, id, hc]
  exact range_getD_map _ _

/-- A call's arguments' words have the widths of its references. -/
theorem callArgs_length {D : Defn} {x : List (List Bool)} (hx : x.map List.length = leafWidths D)
    {outs : Array (List (List Bool))} (ho : OutsOk D outs) (nd : Node)
    (ha : ∀ a ∈ nd.args.toList, a.seq.count = a.seq.toArray.size ∧ ∀ r ∈ a.seq.toArray.toList, RefIn D outs.size r) :
    (nd.args.toList.flatMap fun a =>
        Flock.Program.argVals (Flock.Program.refVal x (Flock.Program.paramStarts D.params) outs) a a.seq.count id).map
      List.length = (nd.args.toList.flatMap fun a => a.seq.toArray.toList).map fun r => (tyAt D r).widths[r.j]! := by
  rw [List.map_flatMap, List.map_flatMap]
  apply List.flatMap_congr
  intro a h
  rw [argVals_count _ a (ha a h).1]
  exact refWords_length hx ho _ (ha a h).2

/-- Member `m` of an argument batched on axis 0, whose references' widths repeat `ew` `n` times, reads words of the
widths `ew`. -/
theorem axisWords_length {D : Defn} {x : List (List Bool)} (hx : x.map List.length = leafWidths D)
    {outs : Array (List (List Bool))} (ho : OutsOk D outs) {s : Seq}
    (hr : ∀ r ∈ s.toArray.toList, RefIn D outs.size r) {n m : ℕ} (hm : m < n) {ew : List ℕ}
    (hw : s.toArray.toList.map (fun r => (tyAt D r).widths[r.j]!) = (List.replicate n ew).flatten) :
    ((List.range ew.length).map fun t => Flock.Program.refVal x (Flock.Program.paramStarts D.params) outs
        (s.toArray.getD (t / ew.length * n * ew.length + t % ew.length + m * ew.length) default)).map List.length =
      ew := by
  have hsz : s.toArray.toList.length = n * ew.length := by
    have := congrArg List.length hw
    rw [List.length_map, length_replicate_flatten] at this
    exact this
  apply List.ext_getElem (by simp)
  intro t h1 h2
  simp only [List.length_map, List.length_range] at h1
  simp only [List.getElem_map, List.getElem_range]
  rw [Nat.div_eq_of_lt h1, Nat.mod_eq_of_lt h1, Nat.zero_mul, Nat.zero_mul, Nat.zero_add]
  have hmn := Nat.mul_le_mul_right ew.length (show m + 1 ≤ n by omega)
  rw [Nat.succ_mul] at hmn
  have hlt : t + m * ew.length < s.toArray.toList.length := by rw [hsz]; omega
  have hA : s.toArray.getD (t + m * ew.length) default = s.toArray.toList[t + m * ew.length] := by
    rw [Array.getD_eq_getD_getElem?, ← Array.getElem?_toList, List.getElem?_eq_getElem hlt, Option.getD_some]
  rw [hA, (refBits hx ho (hr _ (List.getElem_mem hlt))).2]
  have h3 := congrArg (·[t + m * ew.length]?) hw
  simp only [List.getElem?_map, List.getElem?_eq_getElem hlt, Option.map_some] at h3
  rw [replicate_flatten_getElem? ew hm h1, List.getElem?_eq_getElem h1] at h3
  exact Option.some.inj h3

theorem leafWidths_range (C : Defn) :
    leafWidths C = (List.range C.params.size).flatMap fun i => (C.params.getD i default).2.widths.toList := by
  unfold leafWidths
  rw [flatMap_flatten_map, flatMap_flatten_map]
  congr 1
  rw [← Array.length_toList, ← list_range_getD_map C.params.toList (fun p => p.2.widths.toList)]
  apply List.map_congr_left
  intro i hi
  have hi' : i < C.params.size := by simpa using List.mem_range.1 hi
  simp [Array.getD, List.getD_eq_getElem?_getD, hi']

/-- Member `m` of an argument batched on any axis, whose references' widths are `o` blocks of `n` copies of `V`, reads
words of the widths `o` copies of `V`. -/
theorem axisWords_ax {D : Defn} {x : List (List Bool)} (hx : x.map List.length = leafWidths D)
    {outs : Array (List (List Bool))} (ho : OutsOk D outs) {s : Seq}
    (hr : ∀ r ∈ s.toArray.toList, RefIn D outs.size r) {o n m : ℕ} (hm : m < n) {V : List ℕ}
    (hw : s.toArray.toList.map (fun r => (tyAt D r).widths[r.j]!) =
      (List.replicate o (List.replicate n V).flatten).flatten) :
    ((List.range (o * V.length)).map fun t => Flock.Program.refVal x (Flock.Program.paramStarts D.params) outs
        (s.toArray.getD (t / V.length * n * V.length + t % V.length + m * V.length) default)).map List.length =
      (List.replicate o V).flatten := by
  rw [map_range_mul, List.map_flatMap, show (List.replicate o V).flatten = (List.range o).flatMap fun _ => V by
    rw [flatMap_flatten_map, List.map_const', List.length_range]]
  apply List.flatMap_congr
  intro a ha
  have hb := axisWords_length hx ho hr (BodiesOk.lt_mul_of a o n m (List.mem_range.1 ha) hm)
    (hw.trans (flatten_replicate_replicate V o n))
  refine Eq.trans ?_ hb
  congr 1
  apply List.map_congr_left
  intro c hc
  have hc' := List.mem_range.1 hc
  rw [block_index n m hc', Nat.div_eq_of_lt hc', Nat.mod_eq_of_lt hc', Nat.zero_mul, Nat.zero_mul, Nat.zero_add]

/-- A batch member's arguments' words, for a composite callee `C` whose parameters the batch's arguments match on any
axes, have `C`'s parameter leaf widths. -/
theorem batchWords_length_ax {D : Defn} {x : List (List Bool)} (hx : x.map List.length = leafWidths D)
    {outs : Array (List (List Bool))} (ho : OutsOk D outs) {n : ℕ} (hnD : n < D.body.nodes.size)
    (hsz : outs.size = n) {C : Defn}
    (hargs : ∀ a ∈ (D.body.nodes[n]).args.toList,
      a.seq.count = a.seq.toArray.size ∧ ∀ r ∈ a.seq.toArray.toList, RefIn D n r)
    (hbat : ∀ i < (D.body.nodes[n]).args.size,
      (∀ ax, (D.body.nodes[n]).axes[i]? = some (some ax) →
        ((D.body.nodes[n]).args.getD i default).ty.removeAxis ax = some ((D.body.nodes[n]).n, (C.params.getD i default).2) ∧
        ((D.body.nodes[n]).args.getD i default).dims = ((D.body.nodes[n]).args.getD i default).ty.axisDims ax ∧
        i < C.params.size ∧
        ((D.body.nodes[n]).args.getD i default).seq.toArray.toList.map (fun r => (tyAt D r).widths[r.j]!) =
          ((D.body.nodes[n]).args.getD i default).ty.widths.toList) ∧
      ((∀ ax, (D.body.nodes[n]).axes[i]? ≠ some (some ax)) → ((D.body.nodes[n]).args.getD i default).dims = none))
    (hcb : (D.body.nodes[n]).args.size = C.params.size ∧ ∀ i < (D.body.nodes[n]).args.size,
      (∀ ax, (D.body.nodes[n]).axes[i]? ≠ some (some ax)) →
      ((D.body.nodes[n]).args.getD i default).seq.toArray.toList.map (fun r => (tyAt D r).widths[r.j]!) =
        (C.params.getD i default).2.widths.toList)
    {m : ℕ} (hm' : m < (D.body.nodes[n]).n) :
    ((List.range (D.body.nodes[n]).args.size).flatMap fun i =>
        batchArg (Flock.Program.refVal x (Flock.Program.paramStarts D.params) outs) C (D.body.nodes[n]) m i).map
      List.length = leafWidths C := by
  rw [List.map_flatMap, leafWidths_range, ← hcb.1]
  apply List.flatMap_congr
  intro i hi
  have hi' := List.mem_range.1 hi
  have hmemA : (D.body.nodes[n]).args.getD i default ∈ (D.body.nodes[n]).args.toList := by
    simp [Array.getD, hi']
  obtain ⟨hcnt, hrefs⟩ := hargs _ hmemA
  have hrefs' : ∀ r ∈ ((D.body.nodes[n]).args.getD i default).seq.toArray.toList, RefIn D outs.size r :=
    fun r h => by rw [hsz]; exact hrefs r h
  by_cases hax : ∃ ax, (D.body.nodes[n]).axes[i]? = some (some ax)
  · obtain ⟨ax, hax⟩ := hax
    obtain ⟨hrm, hdm, -, hw⟩ := (hbat i hi').1 ax hax
    obtain ⟨o, F, htw, hEw, hdF, -⟩ := removeAxis_blocks ax hrm
    rw [batchArg_some _ _ _ _ _ (hdm.trans hdF)]
    simp only [Flock.Program.argVals]
    generalize (C.params.getD i default).2 = E at hEw ⊢
    rw [show E.leaves = E.widths.toList.length by rw [Array.length_toList, hb_widths_size],
      show F.leaves = F.widths.toList.length by rw [Array.length_toList, hb_widths_size], hEw,
      length_replicate_flatten]
    exact axisWords_ax hx ho hrefs' hm' (hw.trans htw)
  · have hn0 : ∀ ax, (D.body.nodes[n]).axes[i]? ≠ some (some ax) := fun ax h => hax ⟨ax, h⟩
    rw [batchArg_none _ _ _ _ _ ((hbat i hi').2 hn0), argVals_count _ _ hcnt, refWords_length hx ho _ hrefs']
    exact hcb.2 i hi' hn0

/-- Slice `m` of an argument whose references' widths repeat `E`'s `n` times reads words of `E`'s widths. -/
theorem argSliceWords {D : Defn} {x : List (List Bool)} (hx : x.map List.length = leafWidths D)
    {outs : Array (List (List Bool))} (ho : OutsOk D outs) {a : Arg}
    (hr : ∀ r ∈ a.seq.toArray.toList, RefIn D outs.size r) {n m : ℕ} (hm : m < n) {E : Ty}
    (hw : a.seq.toArray.toList.map (fun r => (tyAt D r).widths[r.j]!) = (List.replicate n E.widths.toList).flatten) :
    (Flock.Program.argVals (Flock.Program.refVal x (Flock.Program.paramStarts D.params) outs) a E.leaves
      (fun t => t + m * E.leaves)).map List.length = E.widths.toList := by
  have key : ∀ L t : ℕ, t < L → t / L * n * L + t % L + m * L = t + m * L := fun L t h => by
    rw [Nat.div_eq_of_lt h, Nat.mod_eq_of_lt h, Nat.zero_mul, Nat.zero_mul, Nat.zero_add]
  simp only [Flock.Program.argVals]
  rw [show E.leaves = E.widths.toList.length by rw [Array.length_toList, hb_widths_size]]
  refine Eq.trans ?_ (axisWords_length hx ho hr hm hw)
  congr 1
  exact List.map_congr_left fun t ht => by simp only [key _ t (List.mem_range.1 ht)]

theorem argSharedWords {D : Defn} {x : List (List Bool)} (hx : x.map List.length = leafWidths D)
    {outs : Array (List (List Bool))} (ho : OutsOk D outs) {a : Arg}
    (hr : ∀ r ∈ a.seq.toArray.toList, RefIn D outs.size r) {E : Ty}
    (hw : a.seq.toArray.toList.map (fun r => (tyAt D r).widths[r.j]!) = E.widths.toList) :
    (Flock.Program.argVals (Flock.Program.refVal x (Flock.Program.paramStarts D.params) outs) a E.leaves
      id).map List.length = E.widths.toList := by
  have h := argSliceWords hx ho hr (n := 1) (m := 0) Nat.one_pos (E := E) (by rw [hw]; simp)
  simp only [Nat.zero_mul, Nat.add_zero] at h
  exact h

theorem scanArgWords {D : Defn} {x : List (List Bool)} (hx : x.map List.length = leafWidths D)
    {outs : Array (List (List Bool))} (ho : OutsOk D outs) {C : Defn} {nd : Node} {i : ℕ}
    (hr : ∀ r ∈ (nd.args.getD i default).seq.toArray.toList, RefIn D outs.size r)
    (hw : (nd.args.getD i default).seq.toArray.toList.map (fun r => (tyAt D r).widths[r.j]!) =
      if i ≤ nd.nxs then (List.replicate nd.n (C.params.getD i default).2.widths.toList).flatten
      else (C.params.getD i default).2.widths.toList)
    {m : ℕ} (hm : m < nd.n) :
    (scanArg (Flock.Program.refVal x (Flock.Program.paramStarts D.params) outs) C nd m i).map List.length =
      (C.params.getD i default).2.widths.toList := by
  unfold scanArg
  by_cases hle : i ≤ nd.nxs
  · simp only [hle, ↓reduceIte] at hw ⊢
    exact argSliceWords hx ho hr hm hw
  · simp only [hle, ↓reduceIte] at hw ⊢
    exact argSharedWords hx ho hr hw

theorem scanRestWords {D : Defn} {x : List (List Bool)} (hx : x.map List.length = leafWidths D)
    {outs : Array (List (List Bool))} (ho : OutsOk D outs) {C : Defn} {nd : Node}
    (hargs : ∀ a ∈ nd.args.toList, a.seq.count = a.seq.toArray.size ∧ ∀ r ∈ a.seq.toArray.toList, RefIn D outs.size r)
    (hw : ∀ i < nd.args.size, 1 ≤ i →
      (nd.args.getD i default).seq.toArray.toList.map (fun r => (tyAt D r).widths[r.j]!) =
        if i ≤ nd.nxs then (List.replicate nd.n (C.params.getD i default).2.widths.toList).flatten
        else (C.params.getD i default).2.widths.toList)
    {m : ℕ} (hm : m < nd.n) :
    (((List.range nd.args.size).drop 1).flatMap
        (scanArg (Flock.Program.refVal x (Flock.Program.paramStarts D.params) outs) C nd m)).map List.length =
      ((List.range nd.args.size).drop 1).flatMap fun i => (C.params.getD i default).2.widths.toList := by
  rw [List.map_flatMap]
  apply List.flatMap_congr
  intro i hi
  obtain ⟨h1, h2⟩ := mem_drop_range hi
  have hmemA : nd.args.getD i default ∈ nd.args.toList := by simp [Array.getD, h2]
  exact scanArgWords hx ho (hargs _ hmemA).2 (hw i h2 h1) hm

theorem leafWidths_scan (C : Defn) (h0 : 0 < C.params.size) :
    leafWidths C = (C.params.getD 0 default).2.widths.toList ++
      ((List.range C.params.size).drop 1).flatMap fun i => (C.params.getD i default).2.widths.toList := by
  rw [leafWidths_range]
  obtain ⟨k, hk⟩ : ∃ k, C.params.size = k + 1 := ⟨C.params.size - 1, by omega⟩
  rw [hk, List.range_succ_eq_map]
  simp [List.flatMap_cons]

/-- **A scan node's step on words and on bits** (step 4's node): the word node `n` of `D` with callee `C`, and a bit
node `nd'` of `D'` with callee `C'` (the lifted node, or the node itself in a bit-typed `D`) whose carry and rest
arguments on `x.flatten` are the bits of the word node's on `x`, for any callee whose evaluation on words of its
parameter leaf widths returns words of its return's widths and on their bits those words' bits: the bit node pushes
the bits of what the word node pushes, the callee's carry widths and then its outputs' widths `n` times. -/
theorem scanStep {D D' : Defn} {x : List (List Bool)}
    (hx : x.map List.length = leafWidths D) {outs : Array (List (List Bool))} (ho : OutsOk D outs) {n : ℕ}
    (hnD : n < D.body.nodes.size) (hsz : outs.size = n) (hf : (D.body.nodes[n]).form = .scan) {nd' : Node}
    (hf' : nd'.form = .scan) (hn' : nd'.n = (D.body.nodes[n]).n) {P L : Prog} {y y' : ℕ} {C C' : Defn}
    (hlt : (D.body.nodes[n]).fn < y) (hc : P.defs[(D.body.nodes[n]).fn]? = some C)
    (hlt' : nd'.fn < y') (hc' : L.defs[nd'.fn]? = some C')
    (hargs : ∀ a ∈ (D.body.nodes[n]).args.toList,
      a.seq.count = a.seq.toArray.size ∧ ∀ r ∈ a.seq.toArray.toList, RefIn D n r)
    (hcs : (D.body.nodes[n]).args.size = C.params.size) (h0 : 0 < C.params.size)
    (hw0 : ((D.body.nodes[n]).args.getD 0 default).seq.toArray.toList.map (fun r => (tyAt D r).widths[r.j]!) =
      (C.params.getD 0 default).2.widths.toList)
    (hw : ∀ i < (D.body.nodes[n]).args.size, 1 ≤ i →
      ((D.body.nodes[n]).args.getD i default).seq.toArray.toList.map (fun r => (tyAt D r).widths[r.j]!) =
        if i ≤ (D.body.nodes[n]).nxs then
          (List.replicate (D.body.nodes[n]).n (C.params.getD i default).2.widths.toList).flatten
        else (C.params.getD i default).2.widths.toList)
    {yw : List ℕ} (hret : C.ret.widths.toList = (C.params.getD 0 default).2.widths.toList ++ yw)
    (hsum : (C'.params.getD 0 default).2.leaves = ((C.params.getD 0 default).2.widths.toList).sum)
    (hinitB : Flock.Program.argVals (Flock.Program.refVal x.flatten (Flock.Program.paramStarts D'.params)
        (outs.map List.flatten)) (nd'.args.getD 0 default) (C'.params.getD 0 default).2.leaves id =
      (Flock.Program.argVals (Flock.Program.refVal x (Flock.Program.paramStarts D.params) outs)
        ((D.body.nodes[n]).args.getD 0 default) (C.params.getD 0 default).2.leaves id).flatten)
    (hrest : ∀ m < (D.body.nodes[n]).n,
      ((List.range nd'.args.size).drop 1).flatMap
        (scanArg (Flock.Program.refVal x.flatten (Flock.Program.paramStarts D'.params) (outs.map List.flatten)) C'
          nd' m) =
      (((List.range (D.body.nodes[n]).args.size).drop 1).flatMap
        (scanArg (Flock.Program.refVal x (Flock.Program.paramStarts D.params) outs) C (D.body.nodes[n]) m)).flatten)
    (EW : ℕ → List (List Bool) → Option (List (List Bool))) (EB : ℕ → List Bool → Option (List Bool))
    (hev : ∀ ws : List (List Bool), ws.map List.length = leafWidths C → ∃ o,
      EW (D.body.nodes[n]).fn ws = some o ∧ o.map List.length = C.ret.widths.toList ∧
      EB nd'.fn ws.flatten = some o.flatten) :
    ∃ w, nodeStep wordWires P EW y x D outs (D.body.nodes[n]) = some (outs.push w) ∧
      nodeStep Flock.Program.Wires.bool L EB y' x.flatten D' (outs.map List.flatten) nd' =
        some ((outs.push w).map List.flatten) ∧
      w.map List.length = (C.params.getD 0 default).2.widths.toList ++
        (List.replicate (D.body.nodes[n]).n yw).flatten := by
  have ha0 : 0 < (D.body.nodes[n]).args.size := by omega
  have hargs' : ∀ a ∈ (D.body.nodes[n]).args.toList,
      a.seq.count = a.seq.toArray.size ∧ ∀ r ∈ a.seq.toArray.toList, RefIn D outs.size r := fun a ha => by
    obtain ⟨h1, h2⟩ := hargs a ha
    exact ⟨h1, fun r hr' => by rw [hsz]; exact h2 r hr'⟩
  have hmem0 : (D.body.nodes[n]).args.getD 0 default ∈ (D.body.nodes[n]).args.toList := by simp [Array.getD, ha0]
  have hinitW := argSharedWords hx ho (hargs' _ hmem0).2 hw0
  have hcl : (C.params.getD 0 default).2.leaves = ((C.params.getD 0 default).2.widths.toList).length := by
    rw [Array.length_toList, hb_widths_size]
  rw [nodeStep_scan wordWires (fun _ => rfl) P EW y x D outs _ hf hlt hc,
    nodeStep_scan Flock.Program.Wires.bool (fun _ => rfl) L EB y' x.flatten _ _ _ hf' hlt' hc', hinitB, hn']
  have hk : (C.ret.widths.toList).take ((C.params.getD 0 default).2.widths.toList).length =
      (C.params.getD 0 default).2.widths.toList := by
    simp [hret]
  obtain ⟨cr, ys', g1, g2, g4, g3⟩ := scanFold ((C.params.getD 0 default).2.widths.toList) C.ret.widths.toList hk
    (D.body.nodes[n]).n
    (fun carry m => EW (D.body.nodes[n]).fn (carry ++ ((List.range (D.body.nodes[n]).args.size).drop 1).flatMap
      (scanArg (Flock.Program.refVal x (Flock.Program.paramStarts D.params) outs) C (D.body.nodes[n]) m)))
    (fun cb m => EB nd'.fn (cb ++ ((List.range nd'.args.size).drop 1).flatMap
      (scanArg (Flock.Program.refVal x.flatten (Flock.Program.paramStarts D'.params) (outs.map List.flatten)) C'
        nd' m)))
    (fun carry m hm hcar => by
      obtain ⟨o, h1, h2, h3⟩ := hev (carry ++ ((List.range (D.body.nodes[n]).args.size).drop 1).flatMap
          (scanArg (Flock.Program.refVal x (Flock.Program.paramStarts D.params) outs) C (D.body.nodes[n]) m)) (by
        rw [List.map_append, hcar, scanRestWords hx ho hargs' hw hm, leafWidths_scan C h0, hcs])
      refine ⟨o, h1, h2, ?_⟩
      rw [hrest m hm, ← List.flatten_append]
      exact h3)
    (List.range (D.body.nodes[n]).n) _ [] (fun m hm => List.mem_range.1 hm) hinitW
  rw [← hcl] at g1
  rw [← hsum] at g3
  simp only [List.flatten_nil, List.map_nil, List.nil_append, List.length_range] at g3 g4
  refine ⟨cr ++ ys', ?_, ?_, ?_⟩
  · rw [g1]
    rfl
  · rw [g3]
    simp [Array.map_push, List.flatten_append]
  · rw [List.map_append, g2, g4]
    simp [hret]

/-- `scanStep`'s bit node as the lifted node of `D`: its form, length, carry width and the bits its carry and rest
arguments read. -/
theorem scanLiftArgs {D : Defn} {fnOf : ℕ → ℕ} {primOf : ℕ → Bool} {gatesOf : ℕ → ℕ} {x : List (List Bool)}
    (hx : x.map List.length = leafWidths D) {outs : Array (List (List Bool))} (ho : OutsOk D outs) {n : ℕ}
    (hnD : n < D.body.nodes.size) (hnD' : n < (Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes.size)
    (hsz : outs.size = n) (hf : (D.body.nodes[n]).form = .scan) {C C' : Defn}
    (hpar : C'.params.map (·.2) = C.params.map (·.2.bitType))
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
    ((Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes[n]).form = .scan ∧
      ((Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes[n]).n = (D.body.nodes[n]).n ∧
      (C'.params.getD 0 default).2.leaves = ((C.params.getD 0 default).2.widths.toList).sum ∧
      Flock.Program.argVals (Flock.Program.refVal x.flatten
          (Flock.Program.paramStarts (Flock.Lift.liftDef D fnOf primOf gatesOf).params) (outs.map List.flatten))
          (((Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes[n]).args.getD 0 default)
          (C'.params.getD 0 default).2.leaves id =
        (Flock.Program.argVals (Flock.Program.refVal x (Flock.Program.paramStarts D.params) outs)
          ((D.body.nodes[n]).args.getD 0 default) (C.params.getD 0 default).2.leaves id).flatten ∧
      ∀ m < (D.body.nodes[n]).n,
        ((List.range ((Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes[n]).args.size).drop 1).flatMap
          (scanArg (Flock.Program.refVal x.flatten
            (Flock.Program.paramStarts (Flock.Lift.liftDef D fnOf primOf gatesOf).params) (outs.map List.flatten)) C'
            ((Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes[n]) m) =
        (((List.range (D.body.nodes[n]).args.size).drop 1).flatMap
          (scanArg (Flock.Program.refVal x (Flock.Program.paramStarts D.params) outs) C (D.body.nodes[n]) m)).flatten := by
  have ha0 : 0 < (D.body.nodes[n]).args.size := by omega
  obtain ⟨hseq0, -, -, -⟩ := liftDef_arg D fnOf primOf gatesOf hnD hnD' ha0
  have hmem0 : (D.body.nodes[n]).args.getD 0 default ∈ (D.body.nodes[n]).args.toList := by simp [Array.getD, ha0]
  have hr0 : ∀ r ∈ ((D.body.nodes[n]).args.getD 0 default).seq.toArray.toList, RefIn D outs.size r :=
    fun r hr' => by rw [hsz]; exact (hargs _ hmem0).2 r hr'
  have hinitB := argSharedBits hx ho hseq0 hr0 hw0
  rw [← paramBit hpar h0] at hinitB
  refine ⟨?_, by simp [Flock.Lift.liftDef], by rw [paramBit hpar h0, ← sum_widths], by rw [liftDef_params]; exact hinitB,
    fun m hm => ?_⟩
  · simp only [Flock.Lift.liftDef, Array.getElem_map, hf]; rfl
  · rw [liftDef_params]
    exact scanRestBits hx ho hnD hnD' hsz hpar hargs hcs hw hm

theorem flatten_map_flatten {α : Type} (l : List (List (List α))) : l.flatten.flatten = (l.map List.flatten).flatten := by
  induction l with
  | nil => rfl
  | cons a l ih => rw [List.flatten_cons, List.flatten_append, ih, List.map_cons, List.flatten_cons]

/-- **`GateWords`**: in a C `decode` accepts whose root has no recut, a template of `ScanNested` that reaches no word
primitive evaluates on its words' bits, in any `L` that keeps C's Definitions and at every fuel past it, as it does on
words. -/
def GateWords : Prop :=
  ∀ (bytes : ByteArray) (P L : Prog), decode bytes = .ok P → (P.defs[P.root]!).body.recut = #[] →
    (∀ i < P.defs.size, L.defs[i]? = P.defs[i]?) →
    ∀ c < P.defs.size, ScanNested P c → ¬ ReachesWord P c → GateBodiesDeep P c →
      ∀ z : List (List Bool), z.map List.length = leafWidths (P.defs[c]!) →
        ∃ ws, wordDef? P (c + 1) c z = some ws ∧ ws.map List.length = (P.defs[c]!).ret.widths.toList ∧
          ∀ f, c < f → Flock.Program.evalDef? L f c z.flatten = some ws.flatten

/-- **`LiftWordEval` for a template of batches on any axes and scans over word primitives, gates and earlier such
templates**, on `NestedTyped` and `GateWords`. -/
theorem liftWordEval_scanNested_of (hT : NestedTyped) (hGW : GateWords) :
    ∀ (bytes : ByteArray) (P : Prog) (units : Array ℕ) (S : Prog) (d : ℕ) (L : Prog) (e : ℕ),
      decode bytes = .ok P → (P.defs[P.root]!).body.recut = #[] → Spot.Restrict.restrict P units = .ok S →
      d < P.defs.size → ScanNested P d → GateBodiesDeep P d → Flock.Lift.lift S d = .ok (L, e) →
      ∀ x : List (List Bool), x.map List.length = leafWidths (P.defs[d]!) →
        Prog.evalBool L e x.flatten = (wordEval P d x).flatten := by
  intro bytes P units S d L e h hr hs hd hnd hgd hl x hx
  obtain ⟨hle, hpre, hbool, hidx, -⟩ := restrictShape P units S hs
  obtain ⟨⟨hmap, hkeys⟩, -⟩ := FlockSoundness.DecodeBoolean.decode_facts bytes P h hr
  have hm : P.BoolMapOk := ⟨hmap, hkeys⟩
  have hbodyS := bodyOf_eq hm hbool hidx
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
  obtain ⟨hSL, hLpre, hLbody⟩ := liftKeeps hcont hl
  have hLP : ∀ i < P.defs.size, L.defs[i]? = P.defs[i]? := fun i hi => by
    rw [hLpre i (by omega), get?_of_pre hle hpre hi]
  have hLP' : ∀ i < P.defs.size, L.defs[i]! = P.defs[i]! := fun i hi => by
    rw [getElem!_def, getElem!_def, hLP i hi]
  have hLbP : ∀ k, L.bodyOf? k = P.bodyOf? k := fun k => by rw [hLbody, hbodyS]
  obtain ⟨pl, he, hA, hG, hN, hC⟩ := liftPlacesAll hl
  have hnS : ∀ y, y < P.defs.size → ScanNested P y → ¬ ReachesWord P y → ¬ ReachesWord S y :=
    fun y hy hn hw hs => hw (reachesWord_unrestrict hpre hn hy hs)
  have key : ∀ y, y < P.defs.size → ScanNested P y → Reach P d y →
      ∀ z : List (List Bool), z.map List.length = leafWidths (P.defs[y]!) →
        ∃ ws, wordDef? P (y + 1) y z = some ws ∧ ws.map List.length = (P.defs[y]!).ret.widths.toList ∧
          ∀ f, pl y < f → Flock.Program.evalDef? L f (pl y) z.flatten = some ws.flatten := by
    intro y
    induction y using Nat.strong_induction_on with
    | _ y ih =>
    intro hyl hny hry z hz
    have hryS : Reach S d y := (reach_restrict h hr hpre hd hry).2
    have hgy : GateBodiesDeep P y := fun z' hz' => hgd z' (reach_trans hry hz')
    by_cases hrwP : ReachesWord P y
    swap
    · have hply : pl y = y :=
        hN y hryS (by rw [hpre y hyl]; cases hny; assumption) (hnS y hyl hny hrwP)
      obtain ⟨ws, g1, g2, g3⟩ := hGW bytes P L h hr hLP y hyl hny hrwP hgy z hz
      exact ⟨ws, g1, g2, fun f hf => by rw [hply]; exact g3 f (by omega)⟩
    have hrwS : ReachesWord S y := reachesWord_restrict hpre hrwP
    have hfacts := hT bytes P h hr y hyl hny hgy
    cases hny with
    | mk hp1 hltn hnest hgate hex =>
    have hgb : GateBodies P y := hgy y .root hp1
    have hpS : (S.defs[y]!).prim = false := by rw [hpre y hyl]; exact hp1
    obtain ⟨heS, fnOf, primOf, gatesOf, hLy, hfy, hlty⟩ := hC y hryS hpS hrwS
    have hmemS : ∀ nd ∈ (P.defs[y]!).body.nodes, nd ∈ (S.defs[y]!).body.nodes := fun nd h' => by
      rw [hpre y hyl]; exact h'
    rw [hpre y hyl] at hLy hfy hlty
    have hPy : P.defs[y]? = some (P.defs[y]!) := by rw [getElem!_pos P.defs y hyl]; exact getElem?_pos P.defs y hyl
    have hry' : ∀ nd ∈ (P.defs[y]!).body.nodes, Reach P d nd.fn := fun nd h' => .callee hry hp1 h'
    obtain ⟨hargs, hret, hretw, hord, hout, hcout, hbat, hcarg, hcbat, hscan⟩ := hfacts
    generalize hD : P.defs[y]! = D at hp1 hltn hnest hgate hex hargs hret hretw hord hout hcout hbat hcarg hcbat hscan hLy hfy hlty hz hPy hmemS hry' ⊢
    have hD'p : (Flock.Lift.liftDef D fnOf primOf gatesOf).prim = false := by rw [liftDef_prim]; exact hp1
    have hsz' : (Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes.toList.length = D.body.nodes.toList.length := by
      simp [liftDef_nodes_size]
    have hstep : ∀ f', pl y ≤ f' → ∀ n (hn : n < D.body.nodes.toList.length) (outs : Array (List (List Bool))),
        outs.size = n → OutsOk D outs → ∃ w,
          nodeStep wordWires P (wordDef? P y) y z D outs D.body.nodes.toList[n] = some (outs.push w) ∧
          nodeStep Flock.Program.Wires.bool L (Flock.Program.evalDef? L f') (pl y) z.flatten
              (Flock.Lift.liftDef D fnOf primOf gatesOf) (outs.map List.flatten)
              ((Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes.toList[n]'(by omega)) =
            some ((outs.push w).map List.flatten) ∧ OutsOk D (outs.push w) := by
      intro f' hf' n hn outs hsz ho
      have hnD : n < D.body.nodes.size := by simpa using hn
      have hnD' : n < (Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes.size := by
        rw [liftDef_nodes_size]; exact hnD
      simp only [Array.getElem_toList]
      have hmem : D.body.nodes[n] ∈ D.body.nodes := Array.getElem_mem hnD
      have hlt := hord _ hmem
      have hfnP : (D.body.nodes[n]).fn < P.defs.size := by omega
      obtain ⟨hfn', hform', hseq⟩ := liftDef_node D fnOf primOf gatesOf hnD hnD'
      have hcal : P.defs[(D.body.nodes[n]).fn]? = some (P.defs[(D.body.nodes[n]).fn]!) := by
        rw [getElem!_pos P.defs _ hfnP]; exact getElem?_pos P.defs _ hfnP
      have hargsn : ∀ a ∈ (D.body.nodes[n]).args.toList,
          a.seq.count = a.seq.toArray.size ∧ ∀ r ∈ a.seq.toArray.toList, RefIn D outs.size r := fun a ha => by
        obtain ⟨h1, h2⟩ := hargs n hnD a ha
        exact ⟨h1, fun r hr' => by rw [hsz]; exact h2 r hr'⟩
      have hform2 : (D.body.nodes[n]).form ≠ .batch → (D.body.nodes[n]).form ≠ .scan →
          (D.body.nodes[n]).form = .prim ∨ (D.body.nodes[n]).form = .call := by
        intro hb hs
        cases hfm : (D.body.nodes[n]).form with
        | prim => exact Or.inl rfl
        | call => exact Or.inr rfl
        | batch => exact absurd hfm hb
        | scan => exact absurd hfm hs
      have hns : (P.defs[(D.body.nodes[n]).fn]!).prim = true → (D.body.nodes[n]).form ≠ .scan := fun hpt hs => by
        rw [(hscan n hnD hs).1] at hpt; cases hpt
      have hf2 : (D.body.nodes[n]).form = .batch →
          ((Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes[n]).form = .batch := by
        intro hf; simp only [Flock.Lift.liftDef, Array.getElem_map, hf]; rfl
      have hn2 : ((Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes[n]).n = (D.body.nodes[n]).n := by
        simp [Flock.Lift.liftDef]
      by_cases hw : isWordPrim (P.defs[(D.body.nodes[n]).fn]!).prim (P.defs[(D.body.nodes[n]).fn]!).key = true
      · have hwS : isWordPrim (S.defs[(D.body.nodes[n]).fn]!).prim (S.defs[(D.body.nodes[n]).fn]!).key = true := by
          rw [hpre _ hfnP]; exact hw
        have hbS := hA _ (Reach.callee hryS hpS (hmemS _ hmem)) hwS
        rw [hpre _ hfnP, hbodyS, ← hfy _ hmem] at hbS
        have hbl : fnOf (D.body.nodes[n]).fn < P.defs.size := body_lt hm hbS
        have hprim : (P.defs[(D.body.nodes[n]).fn]!).prim = true := by
          simp only [isWordPrim, Bool.and_eq_true] at hw
          exact hw.1.1
        have hcalL : L.defs[fnOf (D.body.nodes[n]).fn]? = some (P.defs[fnOf (D.body.nodes[n]).fn]!) := by
          rw [hLP _ hbl, getElem!_pos P.defs _ hbl]; exact getElem?_pos P.defs _ hbl
        have hsome : ∀ y', (Flock.Program.evalDef? P (fnOf (D.body.nodes[n]).fn + 1) (fnOf (D.body.nodes[n]).fn) y').isSome =
            true := gateDef_total P _ hbl (hgb _ (by rw [hD]; exact hmem) _ hbS)
        have hB : ∀ y', Flock.Program.evalDef? L f' (fnOf (D.body.nodes[n]).fn) y' =
            some (Prog.evalBool P (D.body.nodes[n]).fn y') :=
          boolCall_at h hr (by omega) hLP' hLbP hfnP hw hbS hsome (by omega)
        have hWp : ∀ ws, wordDef? P y (D.body.nodes[n]).fn ws =
            some [Prog.evalBool P (D.body.nodes[n]).fn ws.flatten] := fun ws => by
          obtain ⟨y', hy'⟩ : ∃ y', y = y' + 1 := ⟨y - 1, by omega⟩
          rw [hy']; exact wordDef_prim hfnP hprim ws
        have hfL : ¬ pl y ≤ fnOf (D.body.nodes[n]).fn := by omega
        by_cases hb : (D.body.nodes[n]).form = .batch
        · have hW := nodeStep_batch wordWires (fun _ => rfl) P (wordDef? P y) y z D outs (D.body.nodes[n]) hb hlt hcal
          have hBt := nodeStep_batch Flock.Program.Wires.bool (fun _ => rfl) L
            (Flock.Program.evalDef? L f') (pl y) z.flatten (Flock.Lift.liftDef D fnOf primOf gatesOf)
            (outs.map List.flatten) ((Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes[n]) (hf2 hb)
            (by rw [hfn']; omega) (by rw [hfn']; exact hcalL)
          rw [liftDef_params] at hBt
          obtain ⟨b', -, hbo, -, -, hpar, -⟩ := hmap _ hfnP hw
          obtain rfl : b' = fnOf (D.body.nodes[n]).fn := Option.some.inj (hbo.symm.trans hbS)
          have hY := fun m (hm' : m < (D.body.nodes[n]).n) => batchBits_ax hz ho hnD hnD' hsz hb hpar (hargs n hnD) (hbat n hnD hb) hm'
          refine ⟨(List.range (D.body.nodes[n]).n).map fun m => Prog.evalBool P (D.body.nodes[n]).fn
              ((List.range (D.body.nodes[n]).args.size).flatMap fun i =>
                batchArg (Flock.Program.refVal z (Flock.Program.paramStarts D.params) outs)
                  (P.defs[(D.body.nodes[n]).fn]!) (D.body.nodes[n]) m i).flatten, ?_, ?_, ?_⟩
          · rw [hW]
            simp only [hWp]
            rw [mapM_some_map]
            exact congrArg (fun z => some (outs.push z)) (flatten_map_singleton _ _)
          · rw [hBt, hfn']
            simp only [hB]
            rw [mapM_some_map, Array.map_push, hn2]
            exact congrArg (fun z => some ((outs.map List.flatten).push z))
              (congrArg List.flatten (List.map_congr_left fun m hm => by rw [hY m (List.mem_range.1 hm)]))
          · apply outsOk_push ho
            rw [hsz, show D.body.nodes.getD n default = D.body.nodes[n] from by simp [Array.getD, hnD]]
            rw [hout n hnD hprim ((List.range (D.body.nodes[n]).n).map fun m =>
                ((List.range (D.body.nodes[n]).args.size).flatMap fun i =>
                  batchArg (Flock.Program.refVal z (Flock.Program.paramStarts D.params) outs)
                    (P.defs[(D.body.nodes[n]).fn]!) (D.body.nodes[n]) m i).flatten)
              (by rw [List.length_map, List.length_range, members_batch hb])]
            simp only [List.map_map]
            try rfl
        · have hform2' := hform2 hb (hns hprim)
          have hout1 : ∀ y', (D.body.nodes[n]).out.widths.toList = [(Prog.evalBool P (D.body.nodes[n]).fn y').length] :=
            fun y' => by rw [hout n hnD hprim [y'] (by rw [members_one hform2']; rfl)]; rfl
          have hargsB := argsBits hz ho hseq hargsn
          rw [← liftDef_params D fnOf primOf gatesOf] at hargsB
          refine ⟨[Prog.evalBool P (D.body.nodes[n]).fn
            ((D.body.nodes[n]).args.toList.flatMap fun a =>
              Flock.Program.argVals (Flock.Program.refVal z (Flock.Program.paramStarts D.params) outs) a a.seq.count
                id).flatten], ?_, ?_, ?_⟩
          · rcases hform2' with hf | hf <;>
              simp [nodeStep, wordWires, hf, hcal, show ¬ y ≤ (D.body.nodes[n]).fn by omega, hWp]
          · rcases hform' hform2' with hf | hf <;>
              simp [nodeStep, Flock.Program.Wires.bool, hf, hfn', hcalL, hfL, hargsB, hB, Array.map_push]
          · apply outsOk_push ho
            rw [hsz]
            simp only [List.map_cons, List.map_nil, Array.getD, hnD, ↓reduceDIte]
            exact (hout1 _).symm
      · by_cases hpc0 : (P.defs[(D.body.nodes[n]).fn]!).prim = false
        · have hpc := hpc0
          have hnc := hnest _ hmem hpc
          have hryc := hry' _ hmem
          have hpcS : (S.defs[(D.body.nodes[n]).fn]!).prim = false := by rw [hpre _ hfnP]; exact hpc
          have hrycS := (reach_restrict h hr hpre hd hryc).2
          obtain ⟨C', hLc, hplc, hpar, IH⟩ : ∃ C' : Defn, L.defs[pl (D.body.nodes[n]).fn]? = some C' ∧
              pl (D.body.nodes[n]).fn < pl y ∧
              C'.params.map (·.2) = (P.defs[(D.body.nodes[n]).fn]!).params.map (·.2.bitType) ∧
              ∀ z : List (List Bool), z.map List.length = leafWidths (P.defs[(D.body.nodes[n]).fn]!) →
                ∃ ws, wordDef? P ((D.body.nodes[n]).fn + 1) (D.body.nodes[n]).fn z = some ws ∧
                  ws.map List.length = (P.defs[(D.body.nodes[n]).fn]!).ret.widths.toList ∧
                  ∀ f, pl (D.body.nodes[n]).fn < f →
                    Flock.Program.evalDef? L f (pl (D.body.nodes[n]).fn) z.flatten = some ws.flatten := by
            by_cases hrwc : ReachesWord P (D.body.nodes[n]).fn
            · have hrwcS : ReachesWord S (D.body.nodes[n]).fn := reachesWord_restrict hpre hrwc
              obtain ⟨-, fnOf2, primOf2, gatesOf2, hLc, -, -⟩ := hC _ hrycS hpcS hrwcS
              rw [hpre _ hfnP] at hLc
              refine ⟨_, hLc, hlty _ hmem hpcS hrwcS, ?_, ih _ hlt hfnP hnc hryc⟩
              rw [liftDef_params]; simp [Array.map_map, Function.comp_def]
            · have hplc0 : pl (D.body.nodes[n]).fn = (D.body.nodes[n]).fn :=
                hN _ hrycS hpcS (hnS _ hfnP hnc hrwc)
              have hbt : BitTyped (P.defs[(D.body.nodes[n]).fn]!) := by
                cases hnc with
                | mk _ _ _ _ h5 => exact h5.resolve_left hrwc
              refine ⟨P.defs[(D.body.nodes[n]).fn]!, ?_, ?_, ?_, ?_⟩
              · rw [hplc0, hLP _ hfnP, getElem!_pos P.defs _ hfnP]; exact getElem?_pos P.defs _ hfnP
              · rw [hplc0]; have := hle; omega
              · exact Array.map_congr_left fun p hp' => (bitType_of_isBit _ (hbt.1 p hp')).symm
              · intro z' hz'
                obtain ⟨ws, g1, g2, g3⟩ := hGW bytes P L h hr hLP _ hfnP hnc hrwc
                  (fun z'' hz'' => hgd z'' (reach_trans hryc hz'')) z' hz'
                exact ⟨ws, g1, g2, fun f hf => by rw [hplc0]; exact g3 f (by omega)⟩
          have hfc := hfy _ hmem
          have hWc : ∀ ws, wordDef? P y (D.body.nodes[n]).fn ws =
              wordDef? P ((D.body.nodes[n]).fn + 1) (D.body.nodes[n]).fn ws := fun ws => wordDef_fuel P _ y ws hlt
          have hfL : ¬ pl y ≤ pl (D.body.nodes[n]).fn := by omega
          by_cases hs : (D.body.nodes[n]).form = .scan
          · obtain ⟨-, hcs, h0, hw0, hw, yw, hretc, hout'⟩ := hscan n hnD hs
            obtain ⟨hf', hn', hsum, hinitB, hrest⟩ := scanLiftArgs hz ho hnD hnD' hsz hs hpar (hargs n hnD) hcs h0 hw0 hw
            obtain ⟨w, g1, g2, g3⟩ := scanStep (P := P) (L := L) (y := y) (y' := pl y) hz ho hnD hsz hs hf' hn' hlt hcal
              (by rw [hfn', hfc]; exact hplc) (by rw [hfn', hfc]; exact hLc) (hargs n hnD) hcs h0 hw0 hw hretc hsum hinitB
              hrest (wordDef? P y) (Flock.Program.evalDef? L f') (fun ws hws => by
                obtain ⟨vs, hv1, hv2, hv3⟩ := IH ws hws
                exact ⟨vs, by rw [hWc]; exact hv1, hv2, by rw [hfn', hfc]; exact hv3 f' (by omega)⟩)
            refine ⟨w, g1, g2, ?_⟩
            apply outsOk_push ho
            rw [hsz, show D.body.nodes.getD n default = D.body.nodes[n] from by simp [Array.getD, hnD], hout']
            exact g3
          by_cases hb : (D.body.nodes[n]).form = .batch
          · have hcb := hcbat n hnD hpc hb
            have hW := nodeStep_batch wordWires (fun _ => rfl) P (wordDef? P y) y z D outs (D.body.nodes[n]) hb hlt hcal
            have hBt := nodeStep_batch Flock.Program.Wires.bool (fun _ => rfl) L
              (Flock.Program.evalDef? L f') (pl y) z.flatten (Flock.Lift.liftDef D fnOf primOf gatesOf)
              (outs.map List.flatten) ((Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes[n]) (hf2 hb)
              (by rw [hfn', hfc]; omega) (by rw [hfn', hfc]; exact hLc)
            rw [liftDef_params, hfn', hfc, hn2] at hBt
            let Am : ℕ → List (List Bool) := fun m => (List.range (D.body.nodes[n]).args.size).flatMap fun i =>
              batchArg (Flock.Program.refVal z (Flock.Program.paramStarts D.params) outs)
                (P.defs[(D.body.nodes[n]).fn]!) (D.body.nodes[n]) m i
            let vsOf : ℕ → List (List Bool) := fun m =>
              (wordDef? P ((D.body.nodes[n]).fn + 1) (D.body.nodes[n]).fn (Am m)).getD []
            have hvs : ∀ m < (D.body.nodes[n]).n,
                wordDef? P ((D.body.nodes[n]).fn + 1) (D.body.nodes[n]).fn (Am m) = some (vsOf m) ∧
                (vsOf m).map List.length = (P.defs[(D.body.nodes[n]).fn]!).ret.widths.toList ∧
                Flock.Program.evalDef? L f' (pl (D.body.nodes[n]).fn) (Am m).flatten = some (vsOf m).flatten := by
              intro m hm'
              obtain ⟨vs, hv1, hv2, hv3⟩ :=
                IH (Am m) (batchWords_length_ax hz ho hnD hsz (hargs n hnD) (hbat n hnD hb) hcb hm')
              have hvo : vsOf m = vs := by simp only [vsOf, hv1, Option.getD_some]
              rw [hvo]
              exact ⟨hv1, hv2, hv3 f' (by omega)⟩
            have hmsP := mapM_some (fun m => wordDef? P y (D.body.nodes[n]).fn ((List.range (D.body.nodes[n]).args.size).flatMap
                fun i => batchArg (Flock.Program.refVal z (Flock.Program.paramStarts D.params) outs)
                  (P.defs[(D.body.nodes[n]).fn]!) (D.body.nodes[n]) m i)) vsOf (List.range (D.body.nodes[n]).n)
              (fun m hm => by rw [hWc]; exact (hvs m (List.mem_range.1 hm)).1)
            have hmsL := mapM_some (fun m => Flock.Program.evalDef? L f' (pl (D.body.nodes[n]).fn)
                ((List.range ((Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes[n]).args.size).flatMap fun i =>
                  batchArg (Flock.Program.refVal z.flatten
                    (Flock.Program.paramStarts (D.params.map fun (n, t) => (n, Ty.bitType t))) (outs.map List.flatten))
                    C'
                    ((Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes[n]) m i))
              (fun m => (vsOf m).flatten) (List.range (D.body.nodes[n]).n)
              (fun m hm => by
                try dsimp only
                rw [batchBits_ax hz ho hnD hnD' hsz hb hpar (hargs n hnD) (hbat n hnD hb) (List.mem_range.1 hm)]
                exact (hvs m (List.mem_range.1 hm)).2.2)
            refine ⟨((List.range (D.body.nodes[n]).n).map vsOf).flatten, ?_, ?_, ?_⟩
            · rw [hW, hmsP]
              rfl
            · rw [hBt, hmsL]
              simp only [Option.bind_eq_bind, Option.bind_some, Option.pure_def, Array.map_push, flatten_map_flatten,
                List.map_map]
              try rfl
            · apply outsOk_push ho
              rw [hsz, show D.body.nodes.getD n default = D.body.nodes[n] from by simp [Array.getD, hnD],
                hcout n hnD hpc (by rw [hb]; intro h; cases h), members_batch hb, List.map_flatten, List.map_map]
              have hc : (List.range (D.body.nodes[n]).n).map (List.map List.length ∘ vsOf) =
                  (List.range (D.body.nodes[n]).n).map fun _ => (P.defs[(D.body.nodes[n]).fn]!).ret.widths.toList :=
                List.map_congr_left fun m hm => (hvs m (List.mem_range.1 hm)).2.1
              rw [hc, List.map_const', List.length_range]
          · have hform2' := hform2 hb hs
            have hargsB := argsBits hz ho hseq hargsn
            rw [← liftDef_params D fnOf primOf gatesOf] at hargsB
            obtain ⟨vs, hv1, hv2, hv3⟩ := IH _ ((callArgs_length hz ho _ hargsn).trans (hcarg n hnD hpc hform2'))
            refine ⟨vs, ?_, ?_, ?_⟩
            · rcases hform2' with hf | hf <;>
                simp [nodeStep, wordWires, hf, hcal, show ¬ y ≤ (D.body.nodes[n]).fn by omega, hWc, hv1]
            · rcases hform' hform2' with hf | hf <;>
                simp [nodeStep, Flock.Program.Wires.bool, hf, hfn', hfc, hLc, hfL, hargsB, hv3 f' (by omega),
                  Array.map_push]
            · apply outsOk_push ho
              rw [hsz, show D.body.nodes.getD n default = D.body.nodes[n] from by simp [Array.getD, hnD],
                hcout n hnD hpc hs, members_one hform2']
              simpa using hv2
        · have hprim : (P.defs[(D.body.nodes[n]).fn]!).prim = true := by simpa using hpc0
          have hwf : isWordPrim (P.defs[(D.body.nodes[n]).fn]!).prim (P.defs[(D.body.nodes[n]).fn]!).key = false := by
            simpa using hw
          obtain ⟨-, -, -, hpar, g, hgo⟩ := gateCallee_facts h hr hfnP (hgate _ hmem hprim hwf)
          have hplf : pl (D.body.nodes[n]).fn = (D.body.nodes[n]).fn :=
            hG _ (Reach.callee hryS hpS (hmemS _ hmem)) (by rw [hpre _ hfnP]; exact hprim) (by rw [hpre _ hfnP]; exact hwf)
          have hfn'' : ((Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes[n]).fn = (D.body.nodes[n]).fn := by
            rw [hfn', hfy _ hmem, hplf]
          have hcalL : L.defs[(D.body.nodes[n]).fn]? = some (P.defs[(D.body.nodes[n]).fn]!) := by
            rw [hLP _ hfnP, getElem!_pos P.defs _ hfnP]; exact getElem?_pos P.defs _ hfnP
          have hB : ∀ y', Flock.Program.evalDef? L f' (D.body.nodes[n]).fn y' =
              some (Prog.evalBool P (D.body.nodes[n]).fn y') := fun y' => by
            obtain ⟨f'', rfl⟩ : ∃ f'', f' = f'' + 1 := ⟨f' - 1, by omega⟩
            rw [evalDef_gate hcalL hprim hgo, evalBool_gate hfnP hprim hgo]
          have hWp : ∀ ws, wordDef? P y (D.body.nodes[n]).fn ws =
              some [Prog.evalBool P (D.body.nodes[n]).fn ws.flatten] := fun ws => by
            obtain ⟨y', hy'⟩ : ∃ y', y = y' + 1 := ⟨y - 1, by omega⟩
            rw [hy']; exact wordDef_prim hfnP hprim ws
          have hfL : ¬ pl y ≤ (D.body.nodes[n]).fn := by omega
          by_cases hb : (D.body.nodes[n]).form = .batch
          · have hW := nodeStep_batch wordWires (fun _ => rfl) P (wordDef? P y) y z D outs (D.body.nodes[n]) hb hlt hcal
            have hBt := nodeStep_batch Flock.Program.Wires.bool (fun _ => rfl) L
              (Flock.Program.evalDef? L f') (pl y) z.flatten (Flock.Lift.liftDef D fnOf primOf gatesOf)
              (outs.map List.flatten) ((Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes[n]) (hf2 hb)
              (by rw [hfn'']; omega) (by rw [hfn'']; exact hcalL)
            rw [liftDef_params] at hBt
            have hY := fun m (hm' : m < (D.body.nodes[n]).n) => batchBits_ax hz ho hnD hnD' hsz hb hpar (hargs n hnD) (hbat n hnD hb) hm'
            refine ⟨(List.range (D.body.nodes[n]).n).map fun m => Prog.evalBool P (D.body.nodes[n]).fn
                ((List.range (D.body.nodes[n]).args.size).flatMap fun i =>
                  batchArg (Flock.Program.refVal z (Flock.Program.paramStarts D.params) outs)
                    (P.defs[(D.body.nodes[n]).fn]!) (D.body.nodes[n]) m i).flatten, ?_, ?_, ?_⟩
            · rw [hW]
              simp only [hWp]
              rw [mapM_some_map]
              exact congrArg (fun z => some (outs.push z)) (flatten_map_singleton _ _)
            · rw [hBt, hfn'']
              simp only [hB]
              rw [mapM_some_map, Array.map_push, hn2]
              exact congrArg (fun z => some ((outs.map List.flatten).push z))
                (congrArg List.flatten (List.map_congr_left fun m hm => by rw [hY m (List.mem_range.1 hm)]))
            · apply outsOk_push ho
              rw [hsz, show D.body.nodes.getD n default = D.body.nodes[n] from by simp [Array.getD, hnD]]
              rw [hout n hnD hprim ((List.range (D.body.nodes[n]).n).map fun m =>
                  ((List.range (D.body.nodes[n]).args.size).flatMap fun i =>
                    batchArg (Flock.Program.refVal z (Flock.Program.paramStarts D.params) outs)
                      (P.defs[(D.body.nodes[n]).fn]!) (D.body.nodes[n]) m i).flatten)
                (by rw [List.length_map, List.length_range, members_batch hb])]
              simp only [List.map_map]
              try rfl
          · have hform2' := hform2 hb (hns hprim)
            have hout1 : ∀ y', (D.body.nodes[n]).out.widths.toList = [(Prog.evalBool P (D.body.nodes[n]).fn y').length] :=
              fun y' => by rw [hout n hnD hprim [y'] (by rw [members_one hform2']; rfl)]; rfl
            have hargsB := argsBits hz ho hseq hargsn
            rw [← liftDef_params D fnOf primOf gatesOf] at hargsB
            refine ⟨[Prog.evalBool P (D.body.nodes[n]).fn
              ((D.body.nodes[n]).args.toList.flatMap fun a =>
                Flock.Program.argVals (Flock.Program.refVal z (Flock.Program.paramStarts D.params) outs) a a.seq.count
                  id).flatten], ?_, ?_, ?_⟩
            · rcases hform2' with hf | hf <;>
                simp [nodeStep, wordWires, hf, hcal, show ¬ y ≤ (D.body.nodes[n]).fn by omega, hWp]
            · rcases hform' hform2' with hf | hf <;>
                simp [nodeStep, Flock.Program.Wires.bool, hf, hfn'', hcalL, hfL, hargsB, hB, Array.map_push]
            · apply outsOk_push ho
              rw [hsz]
              simp only [List.map_cons, List.map_nil, Array.getD, hnD, ↓reduceDIte]
              exact (hout1 _).symm
    obtain ⟨o, g1, -, g3, g4⟩ := fold_calls D D.body.nodes.toList
      (Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes.toList hsz' _ _ (hstep (pl y) (Nat.le_refl _))
      D.body.nodes.toList.length 0 #[] (by omega) rfl (fun i hi => by simp at hi)
    simp only [List.drop_zero] at g1
    have hos : o.size = D.body.nodes.size := by simpa using g4
    have hretr : ∀ r ∈ D.body.ret.toArray.toList, RefIn D o.size r := fun r hr' => by rw [hos]; exact hret.2 r hr'
    refine ⟨(List.range D.body.ret.count).map fun t =>
        Flock.Program.refVal z (Flock.Program.paramStarts D.params) o (D.body.ret.toArray.getD t default), ?_, ?_, ?_⟩
    · unfold wordDef?
      simp only [hPy, Option.bind_eq_bind, Option.bind_some, hp1, Bool.false_eq_true, ↓reduceIte]
      rw [walkStep_comp wordWires (fun _ => rfl) P _ y z hPy hp1, g1]
      rfl
    · rw [hret.1, range_getD_map]
      exact (refWords_length hz g3 _ hretr).trans hretw
    · intro f hf
      obtain ⟨f', rfl⟩ : ∃ f', f = f' + 1 := ⟨f - 1, by omega⟩
      obtain ⟨o', g1', g2', -, -⟩ := fold_calls D D.body.nodes.toList
        (Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes.toList hsz' _ _ (hstep f' (by omega))
        D.body.nodes.toList.length 0 #[] (by omega) rfl (fun i hi => by simp at hi)
      simp only [List.drop_zero, Array.map_empty] at g1' g2'
      obtain rfl : o' = o := Option.some.inj (g1'.symm.trans g1)
      unfold Flock.Program.evalDef?
      rw [walkStep_comp Flock.Program.Wires.bool (fun _ => rfl) L _ (pl y) z.flatten hLy hD'p, g2']
      simp only [Option.bind_eq_bind, Option.bind_some, Option.pure_def]
      rw [liftDef_ret, liftDef_params]
      exact congrArg some (seqBits hz g3 hret.1 hretr)
  have hpd : (P.defs[d]!).prim = false := by cases hnd; assumption
  have hpdS : (S.defs[d]!).prim = false := by rw [hpre d hd]; exact hpd
  have hpl : pl d < L.defs.size := by
    by_cases hrwd : ReachesWord P d
    · obtain ⟨-, _f, _p, _g, hLd, -⟩ := hC d .root hpdS (reachesWord_restrict hpre hrwd)
      exact (Array.getElem?_eq_some_iff.1 hLd).1
    · rw [hN d .root hpdS (hnS d hd hnd hrwd)]; omega
  obtain ⟨ws, hw1, -, hw3⟩ := key d hd hnd .root x hx
  unfold wordEval Prog.evalBool
  rw [wordDef_fuel P d _ x (by omega), hw1, he, hw3 _ (by omega)]
  rfl

end Flock.Chain
