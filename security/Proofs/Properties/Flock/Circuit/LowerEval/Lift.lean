module

public import Proofs.Properties.Flock.Circuit.LowerEval.Keeps
public import Proofs.Properties.Flock.Circuit.LowerEval.Gates

/-! `UnitWidths` and `UnitComputes` from their parts, and `LiftWordEval` for templates of word-primitive calls and axis-0
batches. -/

@[expose] public section

namespace Flock.Chain

open Flock Flock.Lower Flock.Tmpl Flock.TmplValue Flock.Chain.LowerDef
open Flock.Program hiding Form

/-- **`UnitWidths`** holds. -/
theorem unitWidths : UnitWidths := fun bytes P hd hr hG g k m hu hR =>
  ⟨unitArgs_widths hd hr hG hu hR, unitOuts_widths hd hr hG hu hR⟩

/-- **`LowerGateBodies`**: where the block's lowering (`blockTmpl`) succeeds, each Boolean Definition the nodes of a
composite reached from the template `admitted` gives lead to evaluates through gates alone. -/
def LowerGateBodies : Prop :=
  ∀ (bytes : ByteArray) (P : Prog) (partition : Lean.Json) (cuts : List (String × List (String × ℕ)))
    (units : Array ℕ) (S : Prog) (a : Spot.OneStageZk.Admit) (L : Prog) (e : ℕ) (t : Tmpl.Tmpl),
    decode bytes = .ok P → (P.defs[P.root]!).body.recut = #[] →
    Spot.Restrict.restrict P units = .ok S → Spot.OneStageZk.admitted partition cuts S = .ok a →
    a.d < P.defs.size → Flock.Lift.lift S a.d = .ok (L, e) → Spot.OneStageZk.blockTmpl S a = .ok t →
    GateBodiesDeep P a.d

/-- **`UnitComputes`**, on `UnitWordComputes`, `LiftWordEval` and `LowerGateBodies`. -/
theorem unitComputes_of (hW : UnitWordComputes) (hL : LiftWordEval) (hLG : LowerGateBodies) : UnitComputes := by
  intro bytes P partition cuts units S a L e t hd hr hs ha hab hl hbt hG hRs g hg k m hu hda Y hY
  obtain ⟨x, f, h1, h2, -⟩ := admittedBlock_callee hab g hg k m hu
  have hdx : (P.child P.rootScope k m).d = x.fn := by
    rw [child_d]
    generalize P.defs[P.root]! = R at h1 ⊢
    simp [getElem!_def, h1]
  have hlt : a.d < P.defs.size := by
    rw [← hda, hdx]
    exact (Array.getElem?_eq_some_iff.1 h2).1
  have hgb := hLG bytes P partition cuts units S a L e t hd hr hs ha hlt hl hbt
  have ho := hW bytes P hd hr hG g k m hu (hRs g hg) (by rw [hda]; exact hgb) Y hY
  have hw : (argWords Y g).map List.length = leafWidths (P.defs[(P.child P.rootScope k m).d]!) := by
    rw [← (unitWidths bytes P hd hr hG g k m hu (hRs g hg)).1]
    simp [argWords, Function.comp_def, portWord_length]
  rw [hda] at hw ho
  unfold outBits argBits
  rw [portBits_eq, portBits_eq]
  change (outWords Y g).flatten = Prog.evalBool L e (argWords Y g).flatten
  rw [ho, hL bytes P partition cuts units S a L e hd hs ha hlt hl hgb _ hw]

/-- **`BlockEvalDoor`**, on `UnitComputes`. -/
theorem blockEvalDoor_of (hU : UnitComputes) : BlockEvalDoor := by
  intro bytes P partition cuts units S a fr L e t hd hr hs ha hab hf _ hl hbt hG hRs blocks u hB X g hg Y
  obtain ⟨k, m, hu, hin⟩ := frontUnitRanges P units fr hf g hg
  have hY : ∀ h : Fin (Flock.OfProgram.ofProgram P).N, InUnit P k m h.val → Flock.OfProgram.inputAt P h = false →
      Y h = (Flock.OfProgram.ofProgram P).op h Y :=
    fun h h1 h2 => Verity.Partition.localEval_of_mem _ X ((hB h).2 ⟨hin h.val h1, h2⟩)
  exact (hU bytes P partition cuts units S a L e t hd hr hs ha hab hl hbt hG hRs g hg k m hu
    (admitCallee bytes P partition cuts units S a hd hs ha hab g hg k m hu) Y hY).symm

/-- **`DoorLengths`** holds. -/
theorem doorLengths : DoorLengths := by
  intro bytes P partition cuts units S a fr hd hr hs ha hab hf _ hG hRs X g hg
  obtain ⟨k, m, hu, -⟩ := frontUnitRanges P units fr hf g hg
  have hda := admitCallee bytes P partition cuts units S a hd hs ha hab g hg k m hu
  obtain ⟨hwa, hwo⟩ := unitWidths bytes P hd hr hG g k m hu (hRs g hg)
  obtain ⟨x, f, h1, h2, -⟩ := admittedBlock_callee hab g hg k m hu
  have hdx : (P.child P.rootScope k m).d = x.fn := by
    rw [child_d]
    generalize P.defs[P.root]! = R at h1 ⊢
    simp [getElem!_def, h1]
  have hlt : a.d < P.defs.size := by
    rw [← hda, hdx]
    exact (Array.getElem?_eq_some_iff.1 h2).1
  rw [hda] at hwa hwo
  obtain ⟨-, hpre, -⟩ := restrictShape P units S hs
  obtain ⟨fn, L, dl, l, hfn, hlift, hl, hoff, -⟩ := admitted_walk ha
  obtain ⟨hpi, hpo⟩ := admitLeaves ha hlift
  obtain ⟨hp, hr'⟩ := leavesOff_none hoff
  have hg' : L.defs.getD dl default = l := by
    rw [Array.getD_eq_getD_getElem?, hl]; rfl
  have hfnP : P.defs[a.d]! = fn := by
    rw [← hpre _ hlt]
    simp [getElem!_def, hfn]
  rw [hfnP] at hwa hwo
  have key : ∀ (ps : List (String × Ty)) (acc : ℕ),
      ps.foldl (fun a x => a + x.2.widths.foldl (· + ·) 0) acc = acc + (ps.flatMap fun p => p.2.widths.toList).sum := by
    intro ps
    induction ps with
    | nil => simp
    | cons x ps ih =>
      intro acc
      simp only [List.foldl_cons]
      rw [ih]
      simp only [List.flatMap_cons, List.sum_append, sum_foldl_add]
      omega
  unfold argBits outBits
  rw [portBits_length, portBits_length, hwa, hwo, hpi, hpo, hg', ← hp, ← hr', sum_foldl_add, ← Array.foldl_toList,
    key]
  simp [leafWidths]

/-! ## The lift's bit layout, which `LiftWordEval` reads -/

/-- **`BitAtSpec`**: word leaf `j` of `t` starts in `Ty.bitType t` at the sum of the widths before it, with its own
width (`Flock.Lift.bitAt`). -/
def BitAtSpec : Prop :=
  ∀ (t : Ty) (j : ℕ), j < t.widths.size → Flock.Lift.bitAt t j = ((t.widths.toList.take j).sum, t.widths[j]!)

/-- **`LiftSeqRefs`**: `liftSeq` puts in place of each word reference of a body the bits it occupies, `at_`'s run of
them, in order, wherever `at_` agrees with `homo`'s one width. -/
def LiftSeqRefs : Prop :=
  ∀ (at_ : Bool → ℕ → ℕ → ℕ × ℕ) (homo : Bool → ℕ → Option ℕ) (s : Seq),
    (∀ r ∈ s.toArray.toList, ∀ w, homo r.p r.i = some w → ∀ j, at_ r.p r.i j = (j * w, w)) →
    (Flock.Lift.liftSeq at_ homo s).toArray.toList = s.toArray.toList.flatMap fun r =>
      (List.range (at_ r.p r.i r.j).2).map fun t => ({ p := r.p, i := r.i, j := (at_ r.p r.i r.j).1 + t } : Ref)

/-- **`BitTypeWidths`**: `Ty.bitType t` has a one-bit leaf for each bit of each of `t`'s leaves, in leaf order. -/
def BitTypeWidths : Prop :=
  ∀ t : Ty, (Ty.bitType t).widths.toList = t.widths.toList.flatMap fun w => List.replicate w 1

/-- **`BitTypeAxis`**: a batch argument's axis of `t` is the same axis of `Ty.bitType t`, of the same length, the leaves
below it the bits of the word leaves below it. -/
def BitTypeAxis : Prop :=
  ∀ (t : Ty) (ax n L : ℕ), t.axisDims ax = some (n, L) →
    ∃ e : Ty, e.leaves = L ∧ (Ty.bitType t).axisDims ax = some (n, (Ty.bitType e).leaves) ∧
      (Ty.bitType e).leaves = (e.widths.foldl (· + ·) 0)

/-- **`ScanCarryBits`**: a scan node's lift keeps its sequences (`nxs`), and each sequence argument's length; its carries
are its carries' bits. -/
def ScanCarryBits : Prop :=
  ∀ (d : Defn) (fnOf : ℕ → ℕ) (primOf : ℕ → Bool) (gatesOf : ℕ → ℕ) (k : ℕ) (x : Node),
    d.body.nodes[k]? = some x → x.form = .scan →
    ∃ y, (Flock.Lift.liftDef d fnOf primOf gatesOf).body.nodes[k]? = some y ∧ y.nxs = x.nxs ∧ y.members = x.members ∧
      y.args.size = x.args.size ∧ ∀ i < x.args.size, (y.args[i]!).ty = Ty.bitType (x.args[i]!).ty

/-- **`RestrictWordEval`**: C's Definitions keep their value on words in the block's Program `restrict P units`, for a
C `decode` accepts whose root has no recut. -/
def RestrictWordEval : Prop :=
  ∀ (bytes : ByteArray) (P S : Prog) (units : Array ℕ), decode bytes = .ok P → (P.defs[P.root]!).body.recut = #[] →
    Spot.Restrict.restrict P units = .ok S → ∀ d < P.defs.size, ∀ x, wordEval S d x = wordEval P d x

theorem widths_homo_all {t : Ty} {w : ℕ} (h : Widths.homo t = some w) : ∀ x ∈ t.widths.toList, x = w := by
  intro x hx
  have hr := hb_mem_range t x (Array.mem_def.2 hx)
  unfold Widths.homo at h
  split at h
  · rename_i hw
    cases h
    rw [hw] at hr
    simpa using hr
  · cases h

theorem prefix_sums (l : List ℕ) : ∀ (acc : Array ℕ), 0 < acc.size →
    (l.foldl (fun a w => a.push (a.back! + w)) acc).toList =
      acc.toList ++ (List.range l.length).map fun i => acc.back! + (l.take (i + 1)).sum := by
  induction l with
  | nil => intro acc _; simp
  | cons y l ih =>
    intro acc _
    rw [List.foldl_cons, ih _ (by simp)]
    have hb : (acc.push (acc.back! + y)).back! = acc.back! + y := by simp [Array.back!]
    rw [hb, List.length_cons, List.range_succ_eq_map]
    simp [List.take_succ_cons, Nat.add_assoc]

theorem bitAtSpec : BitAtSpec := by
  intro t j hj
  unfold Flock.Lift.bitAt
  cases hw : Widths.homo t with
  | some w =>
    simp only
    have hall := widths_homo_all hw
    have hjw : t.widths[j]! = w := by
      rw [getElem!_pos t.widths j hj]
      exact hall _ (Array.getElem_mem_toList hj)
    rw [hjw]
    have hs : ∀ (l : List ℕ), (∀ x ∈ l, x = w) → (l.take j).sum = (l.take j).length * w := by
      intro l hl
      have : ∀ x ∈ l.take j, x = w := fun x hx => hl x (List.mem_of_mem_take hx)
      rw [List.sum_eq_card_nsmul _ w this, smul_eq_mul]
    rw [hs _ hall, List.length_take, Array.length_toList, Nat.min_eq_left (Nat.le_of_lt hj)]
  | none =>
    simp only
    have hp := prefix_sums t.widths.toList #[0] (by simp)
    rw [Array.foldl_toList] at hp
    congr 1
    rw [getElem!_def, ← Array.getElem?_toList, hp]
    cases j with
    | zero => simp
    | succ j =>
      have hj' : j < t.widths.toList.length := by simp; omega
      simp [Array.back!, List.getElem?_range, show j < t.widths.size by omega]

theorem flatMap_range_one : ∀ n : ℕ, (List.range n).flatMap (fun _ => [1]) = List.replicate n 1
  | 0 => rfl
  | n + 1 => by
    rw [List.range_succ, List.flatMap_append, flatMap_range_one n, List.replicate_succ']
    simp

theorem bitTypeWidths_go : ∀ t : Ty,
    (Ty.bitType t).widths.toList = t.widths.toList.flatMap fun w => List.replicate w 1
  | .value w => by
    by_cases h : w = 1
    · subst h; simp [Ty.bitType, Ty.widths]
    · have hb : (w == 1) = false := by simpa using h
      simp only [Ty.bitType, hb, Bool.false_eq_true, ↓reduceIte, Ty.widths]
      rw [← Array.foldl_toList, foldl_append_toList]
      simp [flatMap_range_one]
  | .array n e => by
    have ih := bitTypeWidths_go e
    simp only [Ty.bitType, Ty.widths]
    rw [← Array.foldl_toList, foldl_append_toList, ← Array.foldl_toList, foldl_append_toList]
    simp only [Array.toList_empty, List.nil_append, ih, List.flatMap_assoc]
  | .tuple fs => by
    have ih : ∀ f ∈ fs.toList, (Ty.bitType f).widths.toList = f.widths.toList.flatMap fun w => List.replicate w 1 :=
      fun f hf => bitTypeWidths_go f
    simp only [Ty.bitType, Ty.widths, Array.attach_map_val]
    rw [← Array.foldl_toList, foldl_append_toList, ← Array.foldl_toList, foldl_append_toList]
    simp only [Array.toList_empty, List.nil_append, Array.toList_map, List.flatMap_map, List.flatMap_assoc]
    exact List.flatMap_congr ih
  | .record fs => by
    have ih : ∀ f ∈ fs.toList,
        (Ty.bitType f.2).widths.toList = f.2.widths.toList.flatMap fun w => List.replicate w 1 :=
      fun f hf => bitTypeWidths_go f.2
    simp only [Ty.bitType, Ty.widths]
    rw [Array.attach_map_val fs fun p => (p.1, Ty.bitType p.2)]
    rw [← Array.foldl_toList, foldl_append_toList, ← Array.foldl_toList, foldl_append_toList]
    simp only [Array.toList_empty, List.nil_append, Array.toList_map, List.flatMap_map, List.flatMap_assoc]
    exact List.flatMap_congr ih
termination_by t => t
decreasing_by
  all_goals simp_wf
  all_goals first
    | omega
    | (have := hb_lt_mem (by assumption); omega)
    | (have := hb_lt_mem2 (by assumption); omega)

theorem bitTypeWidths : BitTypeWidths := bitTypeWidths_go

theorem sum_flatMap_replicate (l : List ℕ) : (l.flatMap fun w => List.replicate w 1).length = l.sum := by
  induction l with
  | nil => rfl
  | cons w l ih => simp [List.flatMap_cons, ih]

/-- `Ty.bitType t` has as many leaves as `t` has bits. -/
theorem bitType_leaves (t : Ty) : (Ty.bitType t).leaves = t.widths.foldl (· + ·) 0 := by
  rw [← hb_widths_size, ← Array.length_toList, bitTypeWidths, sum_flatMap_replicate, sum_foldl_add]

theorem bitTypeAxis : BitTypeAxis := by
  intro t ax
  induction ax generalizing t with
  | zero =>
    intro n L h
    cases t with
    | array m e =>
      simp only [Ty.axisDims, Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      exact ⟨e, rfl, by simp [Ty.bitType, Ty.axisDims], bitType_leaves e⟩
    | _ => simp [Ty.axisDims] at h
  | succ ax ih =>
    intro n L h
    cases t with
    | array m e =>
      simp only [Ty.axisDims] at h
      obtain ⟨e', h1, h2, h3⟩ := ih e n L h
      exact ⟨e', h1, by simpa [Ty.bitType, Ty.axisDims] using h2, h3⟩
    | _ => simp [Ty.axisDims] at h

/-- A sequence's references (`Seq.toArray`), part by part. -/
theorem seq_toArray_toList (s : Seq) :
    s.toArray.toList = s.parts.toList.flatMap fun q => (List.range q.count).map q.get := by
  unfold Seq.toArray
  rw [← Array.foldl_toList]
  suffices h : ∀ (ps : List Part) (acc : Array Ref),
      (ps.foldl (fun acc q => (Array.range q.count).foldl (fun acc k => acc.push (q.get k)) acc) acc).toList =
        acc.toList ++ ps.flatMap fun q => (List.range q.count).map q.get by
    simpa using h s.parts.toList #[]
  intro ps
  induction ps with
  | nil => simp
  | cons q ps ih =>
    intro acc
    rw [List.foldl_cons, ih, List.flatMap_cons, ← List.append_assoc]
    congr 1
    rw [← Array.foldl_toList]
    suffices h' : ∀ (ks : List ℕ) (a : Array Ref), (ks.foldl (fun acc k => acc.push (q.get k)) a).toList =
        a.toList ++ ks.map q.get by simpa using h' _ acc
    intro ks
    induction ks with
    | nil => simp
    | cons k ks ihk => intro a; simp [ihk]

theorem range_mul_map {β : Type} (g : ℕ → β) (w : ℕ) : ∀ c : ℕ,
    (List.range (c * w)).map g = (List.range c).flatMap fun k => (List.range w).map fun t => g (k * w + t)
  | 0 => by simp
  | c + 1 => by
    rw [Nat.succ_mul, List.range_add, List.map_append, range_mul_map g w c, List.range_succ, List.flatMap_append]
    simp [Function.comp_def]

theorem ofParts_parts (ps : Array Part) : (Seq.ofParts ps).parts = ps := rfl

theorem explicit_refs (A : Array Ref) : (List.range (Part.explicit A).count).map (Part.explicit A).get = A.toList := by
  apply List.ext_getElem
  · simp [Part.count]
  · intro n h1 h2
    simp only [Part.count, List.length_map, List.length_range] at h1
    simp [Flock.Program.Part.get, getElem!_pos A n h1]

theorem foldl_append_of {β : Type} {f : Array Ref → β → Array Ref} {g : β → Array Ref}
    (h : ∀ a k, f a k = a ++ g k) (l : List β) (acc : Array Ref) :
    (l.foldl f acc).toList = acc.toList ++ l.flatMap fun k => (g k).toList := by
  induction l generalizing acc with
  | nil => simp
  | cons k l ih => rw [List.foldl_cons, ih, h, List.flatMap_cons]; simp

theorem liftSeqRefs : LiftSeqRefs := by
  intro at_ homo s hh
  let G : Ref → List Ref := fun r =>
    (List.range (at_ r.p r.i r.j).2).map fun t => ({ p := r.p, i := r.i, j := (at_ r.p r.i r.j).1 + t } : Ref)
  have hG : ∀ q : Part, ((Array.range q.count).foldl (fun a k => a ++
      (Array.range (at_ (q.get k).p (q.get k).i (q.get k).j).2).map fun t =>
        ({ p := (q.get k).p, i := (q.get k).i, j := (at_ (q.get k).p (q.get k).i (q.get k).j).1 + t } : Ref)) #[]).toList =
      ((List.range q.count).map q.get).flatMap G := by
    intro q
    rw [← Array.foldl_toList, foldl_append_of (fun a k => rfl)]
    simp [G, List.flatMap_map]
  unfold Flock.Lift.liftSeq
  rw [seq_toArray_toList, seq_toArray_toList, ofParts_parts, Array.toList_map, List.flatMap_map, List.flatMap_assoc]
  apply List.flatMap_congr
  intro q hq
  rcases q with ⟨p, i, b, c⟩ | ⟨p, i, b, inner, os, is, c⟩ | ⟨it⟩ | ⟨run, lanes⟩
  · dsimp only
    cases hw : homo p i with
    | none =>
      dsimp only
      rw [explicit_refs]
      exact hG _
    | some w =>
      dsimp only
      simp only [Part.count]
      rw [range_mul_map, List.flatMap_map]
      apply List.flatMap_congr
      intro k hk
      have hmem : (⟨p, i, b + k⟩ : Ref) ∈ s.toArray.toList := by
        rw [seq_toArray_toList]
        exact List.mem_flatMap.2 ⟨_, hq, List.mem_map.2 ⟨k, hk, rfl⟩⟩
      have hat := hh ⟨p, i, b + k⟩ hmem w hw (b + k)
      dsimp only at hat
      show List.map (fun t => (⟨p, i, b * w + (k * w + t)⟩ : Ref)) (List.range w) =
        List.map (fun t => ({ p := p, i := i, j := (at_ p i (b + k)).1 + t } : Ref)) (List.range (at_ p i (b + k)).2)
      rw [hat]
      apply List.map_congr_left
      intro t _
      simp [Nat.add_mul, Nat.add_assoc]
  all_goals
    dsimp only
    rw [explicit_refs]
    exact hG _

/-! ## `RestrictWordEval`, proved -/

/-- The walk's one call reads Definition `d`, the Definitions below it and, for a primitive with no gate, its Boolean
Definition: two Programs and two recursive calls that agree there give the same value. -/
theorem walkStep_congr {W : Type} [Inhabited W] (I : Flock.Program.Wires Option W) (hI : ∀ {α : Type} (o : Option α), I.lift o = o)
    {P Q : Prog} {ev ev' : ℕ → List W → Option (List W)} {d : ℕ} (x : List W)
    (hd : P.defs[d]? = Q.defs[d]?) (hlow : ∀ i < d, P.defs[i]? = Q.defs[i]?)
    (hev : ∀ e < d, ∀ ins, ev e ins = ev' e ins)
    (hprim : ∀ defn, P.defs[d]? = some defn → defn.prim = true → P.bodyOf? defn.key = Q.bodyOf? defn.key ∧
      ∀ b, P.bodyOf? defn.key = some b → ∀ ins, ev b ins = ev' b ins) :
    Flock.Program.walkStep I P ev d x = Flock.Program.walkStep I Q ev' d x := by
  unfold Flock.Program.walkStep
  rw [← hd]
  cases hD : P.defs[d]? with
  | none => simp [hI]
  | some defn =>
    simp only [hI, Option.bind_eq_bind, Option.bind_some]
    by_cases hp : defn.prim = true
    · obtain ⟨hb, hbe⟩ := hprim defn hD hp
      simp only [hp, ↓reduceIte, hb]
      split
      · rfl
      · cases hB : Q.bodyOf? defn.key with
        | none => rfl
        | some b =>
          have := hbe b (by rw [hb, hB])
          simp [this]
    · simp only [hp, Bool.false_eq_true, ↓reduceIte]
      congr 1
      congr 1
      funext outs nd
      by_cases h : d ≤ nd.fn
      · simp only [h, ↓reduceIte]
        rfl
      · simp only [h, ↓reduceIte, hlow nd.fn (by omega), hev nd.fn (by omega)]

theorem get?_of_pre {P S : Prog} (hle : P.defs.size ≤ S.defs.size) (hpre : ∀ i < P.defs.size, S.defs[i]! = P.defs[i]!)
    {i : ℕ} (hi : i < P.defs.size) : S.defs[i]? = P.defs[i]? := by
  have h1 : i < S.defs.size := by omega
  rw [getElem?_pos S.defs i h1, getElem?_pos P.defs i hi]
  congr 1
  have := hpre i hi
  rwa [getElem!_pos S.defs i h1, getElem!_pos P.defs i hi] at this

theorem body_lt {P : Prog} (hm : P.BoolMapOk) {k : String} {b : ℕ} (h : P.bodyOf? k = some b) : b < P.defs.size := by
  have h0 := h
  unfold Prog.bodyOf? at h0
  cases hv : P.boolean[k]? with
  | none => rw [hv] at h0; cases h0
  | some v =>
    obtain ⟨w, hw, -, hkey, hword⟩ := hm.2 k v hv
    obtain ⟨b', hb', hbo, -⟩ := hm.1 w hw (by rw [hkey]; exact hword)
    rw [hkey] at hbo
    obtain rfl : b' = b := Option.some.inj (hbo.symm.trans h)
    exact hb'

/-- The walk on bits in `S` is the walk in `P` at `P`'s Definitions, where `S` keeps them and their Boolean
Definitions. -/
theorem evalDef_restrict {P S : Prog} (hle : P.defs.size ≤ S.defs.size)
    (hpre : ∀ i < P.defs.size, S.defs[i]! = P.defs[i]!) (hbody : ∀ k, S.bodyOf? k = P.bodyOf? k)
    (hblt : ∀ k b, P.bodyOf? k = some b → b < P.defs.size) :
    ∀ f e ins, e < P.defs.size → Flock.Program.evalDef? S f e ins = Flock.Program.evalDef? P f e ins
  | 0, _, _, _ => rfl
  | f + 1, e, ins, he => by
    unfold Flock.Program.evalDef?
    apply walkStep_congr _ (fun _ => rfl) _ (get?_of_pre hle hpre he)
      (fun i hi => get?_of_pre hle hpre (by omega))
      (fun e' he' ins' => evalDef_restrict hle hpre hbody hblt f e' ins' (by omega))
    intro defn _ _
    refine ⟨hbody _, fun b hb ins' => evalDef_restrict hle hpre hbody hblt f b ins' ?_⟩
    rw [hbody] at hb
    exact hblt _ _ hb

theorem evalDef_fuel_iter (P : Prog) (hB : ∀ k, isBooleanPrim k = true → P.bodyOf? k = none) {b : ℕ}
    (hb : P.BoolDef b) (x : List Bool) :
    ∀ k, Flock.Program.evalDef? P (b + 1 + k) b x = Flock.Program.evalDef? P (b + 1) b x
  | 0 => rfl
  | k + 1 => by
    rw [show b + 1 + (k + 1) = (b + 1 + k) + 1 from by omega,
      FlockSoundness.DecodeBoolean.fuel_stable P hB hb (by omega) x, evalDef_fuel_iter P hB hb x k]

theorem evalDef_fuel_at (P : Prog) (hB : ∀ k, isBooleanPrim k = true → P.bodyOf? k = none) {b m : ℕ}
    (hb : P.BoolDef b) (hm : b < m) (x : List Bool) :
    Flock.Program.evalDef? P m b x = Flock.Program.evalDef? P (b + 1) b x := by
  obtain ⟨k, rfl⟩ : ∃ k, m = b + 1 + k := ⟨m - (b + 1), by omega⟩
  exact evalDef_fuel_iter P hB hb x k

/-- `restrict` keeps every primitive of C evaluating as in C. -/
theorem evalBool_restrict {bytes : ByteArray} {P S : Prog} {units : Array ℕ} (h : decode bytes = .ok P)
    (hr : (P.defs[P.root]!).body.recut = #[]) (hs : Spot.Restrict.restrict P units = .ok S) {p : ℕ}
    (hp : p < P.defs.size) (hprim : (P.defs[p]!).prim = true) (y : List Bool) :
    Prog.evalBool S p y = Prog.evalBool P p y := by
  obtain ⟨hle, hpre, hbool, hidx, -⟩ := restrictShape P units S hs
  obtain ⟨⟨hmap, hkeys⟩, -⟩ := FlockSoundness.DecodeBoolean.decode_facts bytes P h hr
  have hm : P.BoolMapOk := ⟨hmap, hkeys⟩
  have hbody := bodyOf_eq hm hbool hidx
  have hB := FlockSoundness.DecodeBoolean.bool_none hkeys
  have hBS : ∀ k, isBooleanPrim k = true → S.bodyOf? k = none := fun k hk => by rw [hbody]; exact hB k hk
  have hdp : S.defs[p]? = some (P.defs[p]!) := by
    rw [get?_of_pre hle hpre hp, getElem!_pos P.defs p hp]; exact getElem?_pos P.defs p hp
  have hdP : P.defs[p]? = some (P.defs[p]!) := by rw [getElem!_pos P.defs p hp]; exact getElem?_pos P.defs p hp
  unfold Prog.evalBool
  rw [show 2 * S.defs.size + 2 = (2 * S.defs.size + 1) + 1 from rfl,
    show 2 * P.defs.size + 2 = (2 * P.defs.size + 1) + 1 from rfl]
  unfold Flock.Program.evalDef?
  unfold Flock.Program.walkStep
  rw [hdp, hdP]
  simp only [Flock.Program.Wires.bool, Option.pure_def, Option.bind_eq_bind, Option.bind_some, hprim, ↓reduceIte]
  split
  · rfl
  · rw [hbody]
    cases hb : P.bodyOf? (P.defs[p]!).key with
    | none => rfl
    | some b =>
      have hbl := body_lt hm hb
      have hbd : P.BoolDef b := by
        have h0 := hb
        unfold Prog.bodyOf? at h0
        cases hv : P.boolean[(P.defs[p]!).key]? with
        | none => rw [hv] at h0; cases h0
        | some v =>
          obtain ⟨w, hw, -, hkey, hword⟩ := hkeys _ v hv
          obtain ⟨b', -, hbo, hbody', -⟩ := hmap w hw (by rw [hkey]; exact hword)
          rw [hkey] at hbo
          obtain rfl : b' = b := Option.some.inj (hbo.symm.trans hb)
          exact hbody'
      have hbdS : S.BoolDef b := boolDef_lift hpre hbd hbl
      simp only [Option.bind_some]
      rw [evalDef_fuel_at S hBS hbdS (by omega) y, evalDef_fuel_at P hB hbd (by omega) y,
        evalDef_restrict hle hpre hbody (fun k b' h' => body_lt hm h') (b + 1) b y hbl]

theorem wordDef_restrict {P S : Prog} (hle : P.defs.size ≤ S.defs.size)
    (hpre : ∀ i < P.defs.size, S.defs[i]! = P.defs[i]!)
    (hpe : ∀ p < P.defs.size, (P.defs[p]!).prim = true → ∀ y, Prog.evalBool S p y = Prog.evalBool P p y) :
    ∀ f d x, d < P.defs.size → wordDef? S f d x = wordDef? P f d x
  | 0, _, _, _ => rfl
  | f + 1, d, x, hd => by
    unfold wordDef?
    rw [get?_of_pre hle hpre hd]
    cases hD : P.defs[d]? with
    | none => rfl
    | some defn =>
      have hdn : P.defs[d]! = defn := by rw [getElem!_def, hD]
      simp only [Option.bind_eq_bind, Option.bind_some]
      by_cases hp : defn.prim = true
      · simp only [hp, ↓reduceIte, hpe d hd (by rw [hdn]; exact hp)]
      · simp only [hp, Bool.false_eq_true, ↓reduceIte]
        apply walkStep_congr _ (fun _ => rfl) _ (get?_of_pre hle hpre hd)
          (fun i hi => get?_of_pre hle hpre (by omega))
          (fun e he ins => wordDef_restrict hle hpre hpe f e ins (by omega))
        intro defn' h' hp'
        rw [get?_of_pre hle hpre hd, hD] at h'
        cases h'
        exact absurd hp' hp

theorem wordDef_fuel (P : Prog) : ∀ d f x, d < f → wordDef? P f d x = wordDef? P (d + 1) d x := by
  intro d
  induction d using Nat.strong_induction_on with
  | _ d ih =>
    intro f x hf
    obtain ⟨f', rfl⟩ : ∃ f', f = f' + 1 := ⟨f - 1, by omega⟩
    unfold wordDef?
    cases hD : P.defs[d]? with
    | none => rfl
    | some defn =>
      simp only [Option.bind_eq_bind, Option.bind_some]
      by_cases hp : defn.prim = true
      · simp only [hp, ↓reduceIte]
      · simp only [hp, Bool.false_eq_true, ↓reduceIte]
        apply walkStep_congr _ (fun _ => rfl) _ rfl (fun _ _ => rfl)
          (fun e he ins => by rw [ih e he f' ins (by omega), ih e he d ins he])
        intro defn' h' hp'
        rw [hD] at h'
        cases h'
        exact absurd hp' hp

theorem restrictWordEval : RestrictWordEval := by
  intro bytes P S units h hr hs d hd x
  obtain ⟨hle, hpre, -⟩ := restrictShape P units S hs
  unfold wordEval
  rw [wordDef_fuel S d _ x (by omega), wordDef_fuel P d _ x (by omega),
    wordDef_restrict hle hpre (fun p hp hprim y => evalBool_restrict h hr hs hp hprim y) (d + 1) d x hd]

/-! ## A reference's bits in the lift -/

theorem ofParts_count (ps : Array Part) : (Seq.ofParts ps).count = (Seq.ofParts ps).toArray.size := by
  have h : ∀ (l : List Part) (a : Array ℕ) (s : ℕ),
      (l.foldl (fun (acc : Array ℕ × ℕ) q => (acc.1.push acc.2, acc.2 + q.count)) (a, s)).2 = s + (l.map Part.count).sum := by
    intro l
    induction l with
    | nil => simp
    | cons q l ih => intro a s; simp only [List.foldl_cons, ih, List.map_cons, List.sum_cons]; omega
  rw [← Array.length_toList, seq_toArray_toList]
  simp only [Seq.ofParts, ofParts_parts]
  rw [← Array.foldl_toList, h]
  simp [List.length_flatMap]

theorem range_getD_map {α β : Type} [Inhabited α] (A : Array α) (f : α → β) :
    (List.range A.size).map (fun t => f (A.getD t default)) = A.toList.map f := by
  apply List.ext_getElem (by simp)
  intro t h1 _
  simp only [List.length_map, List.length_range] at h1
  simp [Array.getD, h1]

theorem flatten_at : ∀ (x : List (List Bool)) (n : ℕ) (hn : n < x.length),
    (List.range (x[n]).length).map (fun t => x.flatten.getD (((x.map List.length).take n).sum + t) false) = x[n]
  | [], n, hn => by simp at hn
  | a :: xs, 0, _ => by
    apply List.ext_getElem (by simp)
    intro t h1 _
    simp only [List.length_map, List.length_range, List.getElem_cons_zero] at h1
    simp [List.getD_eq_getElem?_getD, List.getElem?_append_left h1, h1]
  | a :: xs, n + 1, hn => by
    have ih := flatten_at xs n (by simpa using hn)
    simp only [List.getElem_cons_succ, List.map_cons, List.take_succ_cons, List.sum_cons, List.flatten_cons]
    conv_rhs => rw [← ih]
    apply List.map_congr_left
    intro t _
    rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, Nat.add_assoc,
      List.getElem?_append_right (by omega)]
    simp

theorem flatMap_split {α β : Type} (ps : List α) (f : α → List β) {i : ℕ} (hi : i < ps.length) :
    ps.flatMap f = (ps.take i).flatMap f ++ (f ps[i] ++ (ps.drop (i + 1)).flatMap f) := by
  conv_lhs => rw [← List.take_append_drop i ps, List.drop_eq_getElem_cons hi, List.flatMap_append, List.flatMap_cons]

theorem paramStarts_toList (ps : Array (String × Ty)) :
    (Flock.Program.paramStarts ps).toList =
      (List.range ps.size).map fun i => ((ps.toList.take i).map (·.2.leaves)).sum := by
  have h : ∀ (l : List (String × Ty)) (a : Array ℕ) (s : ℕ),
      (l.foldl (fun (acc : Array ℕ × ℕ) p => (acc.1.push acc.2, acc.2 + p.2.leaves)) (a, s)).1.toList =
        a.toList ++ (List.range l.length).map fun i => s + ((l.take i).map (·.2.leaves)).sum := by
    intro l
    induction l with
    | nil => simp
    | cons p l ih =>
      intro a s
      simp only [List.foldl_cons, ih, Array.toList_push, List.length_cons, List.range_succ_eq_map, List.map_cons,
        List.map_map, List.append_assoc, List.take_zero, List.map_nil, List.sum_nil, Nat.add_zero, List.cons_append,
        List.nil_append]
      congr 2
      apply List.map_congr_left
      intro i _
      simp [Nat.add_assoc]
  unfold Flock.Program.paramStarts
  rw [← Array.foldl_toList, h]
  simp

theorem paramStarts_getD (ps : Array (String × Ty)) {i : ℕ} (hi : i < ps.size) :
    (Flock.Program.paramStarts ps).getD i 0 = ((ps.toList.take i).map (·.2.leaves)).sum := by
  have hs : (Flock.Program.paramStarts ps).size = ps.size := by
    rw [← Array.length_toList, paramStarts_toList]; simp
  rw [Array.getD_eq_getD_getElem?, ← Array.getElem?_toList, paramStarts_toList]
  simp [hi]

theorem sum_widths (t : Ty) : t.widths.toList.sum = (Ty.bitType t).leaves := by
  rw [bitType_leaves, sum_foldl_add]

/-- Reference `r`'s collection's type in Definition `D`'s body: its parameter's or node's. -/
def tyAt (D : Defn) (r : Ref) : Ty :=
  if r.p then (D.params.getD r.i default).2 else (D.body.nodes.getD r.i default).out

/-- Reference `r` names, in `D`'s body before node `k`, a leaf of `D` that exists. -/
def RefIn (D : Defn) (k : ℕ) (r : Ref) : Prop :=
  (r.p = true → r.i < D.params.size) ∧ (r.p = false → r.i < k ∧ r.i < D.body.nodes.size) ∧
    r.j < (tyAt D r).widths.size

/-- The words of `D`'s body's node values so far, `outs`, have their nodes' leaf widths. -/
def OutsOk (D : Defn) (outs : Array (List (List Bool))) : Prop :=
  ∀ i < outs.size, (outs.getD i []).map List.length = (D.body.nodes.getD i default).out.widths.toList

theorem getD_map_flatten (outs : Array (List (List Bool))) (i : ℕ) :
    (outs.map List.flatten).getD i [] = (outs.getD i []).flatten := by
  by_cases h : i < outs.size
  · simp [Array.getD, h]
  · simp [Array.getD, h]

/-- **A reference's bits**: in `D`'s lift, the bits `bitAt` gives word reference `r` read its word, bit by bit. -/
theorem refBits {D : Defn} {x : List (List Bool)} (hx : x.map List.length = leafWidths D)
    {outs : Array (List (List Bool))} (ho : OutsOk D outs) {r : Ref} (hr : RefIn D outs.size r) :
    (List.range (Flock.Lift.bitAt (tyAt D r) r.j).2).map (fun t =>
      Flock.Program.refVal x.flatten (Flock.Program.paramStarts (D.params.map fun (n, t) => (n, Ty.bitType t)))
        (outs.map List.flatten) ⟨r.p, r.i, (Flock.Lift.bitAt (tyAt D r) r.j).1 + t⟩) =
      Flock.Program.refVal x (Flock.Program.paramStarts D.params) outs r ∧
    (Flock.Program.refVal x (Flock.Program.paramStarts D.params) outs r).length = (tyAt D r).widths[r.j]! := by
  obtain ⟨hp, hn, hj⟩ := hr
  rw [bitAtSpec _ _ hj]
  dsimp only
  cases hrp : r.p
  · obtain ⟨hk, -⟩ := hn hrp
    have hty : tyAt D r = (D.body.nodes.getD r.i default).out := by simp [tyAt, hrp]
    have hw := ho r.i hk
    rw [hty] at hj ⊢
    have hjl : r.j < (outs.getD r.i []).length := by
      have := congrArg List.length hw
      simp only [List.length_map, Array.length_toList] at this
      omega
    have hf := flatten_at (outs.getD r.i []) r.j hjl
    rw [hw] at hf
    have hlen : ((outs.getD r.i [])[r.j]).length = (D.body.nodes.getD r.i default).out.widths[r.j]! := by
      rw [getElem!_pos (D.body.nodes.getD r.i default).out.widths r.j hj]
      have := List.getElem_of_eq hw (by simpa using hjl)
      simpa using this
    simp only [Flock.Program.refVal, hrp, Bool.false_eq_true, ↓reduceIte, getD_map_flatten]
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hjl, Option.getD_some]
    refine ⟨?_, hlen⟩
    rw [← hlen]
    exact hf
  · have hi := hp hrp
    have hty : tyAt D r = (D.params.getD r.i default).2 := by simp [tyAt, hrp]
    rw [hty] at hj ⊢
    have hil : r.i < D.params.toList.length := by simpa using hi
    have hsplit := flatMap_split D.params.toList (fun p => p.2.widths.toList) hil
    have hgi : D.params.toList[r.i] = D.params.getD r.i default := by simp [Array.getD, hi]
    rw [hgi] at hsplit
    set A := (D.params.toList.take r.i).flatMap (fun p => p.2.widths.toList)
    set Wd := (D.params.getD r.i default).2.widths.toList
    have hAlen : A.length = (Flock.Program.paramStarts D.params).getD r.i 0 := by
      rw [paramStarts_getD _ hi]
      simp only [A, List.length_flatMap, Array.length_toList]
      congr 1
      apply List.map_congr_left
      intro p _
      exact hb_widths_size p.2
    have hAsum : A.sum = (Flock.Program.paramStarts (D.params.map fun (n, t) => (n, Ty.bitType t))).getD r.i 0 := by
      rw [paramStarts_getD _ (by simpa using hi)]
      simp only [A]
      rw [← sum_map_flatMap]
      have hm : (D.params.map fun (n, t) => (n, Ty.bitType t)).toList.take r.i =
          (D.params.toList.take r.i).map (fun (n, t) => (n, Ty.bitType t)) := by
        first | (rw [Array.toList_map, List.map_take]) | (rw [Array.toList_map, ← List.map_take]) | simp
      rw [hm, List.map_map]
      congr 1
      apply List.map_congr_left
      intro p _
      first
        | exact sum_widths p.2
        | (simp only [Function.comp_def]; exact sum_widths p.2)
        | (rw [← sum_widths]; simp)
        | (simp only [Function.comp_def]; rw [← sum_widths]; simp)
    have hxl : x.map List.length =
        A ++ (Wd ++ (D.params.toList.drop (r.i + 1)).flatMap fun p => p.2.widths.toList) := by
      rw [hx]; exact hsplit
    set N := (Flock.Program.paramStarts D.params).getD r.i 0 + r.j
    have hjW : r.j < Wd.length := by simpa [Wd] using hj
    have hNlt : N < x.length := by
      have := congrArg List.length hxl
      simp only [List.length_map, List.length_append] at this
      omega
    have hNw : (x.map List.length)[N]'(by simpa using hNlt) = Wd[r.j] := by
      simp only [hxl, N, ← hAlen]
      rw [List.getElem_append_right (by omega), List.getElem_append_left (by simp; omega)]
      congr 1; omega
    have hN : N = A.length + r.j := by simp only [N, hAlen]
    have htake : ((x.map List.length).take N).sum = A.sum + (Wd.take r.j).sum := by
      rw [hN, hxl, List.take_append, List.take_of_length_le (by omega : A.length ≤ A.length + r.j),
        Nat.add_sub_cancel_left, List.take_append_of_le_length (by omega : r.j ≤ Wd.length), List.sum_append]
    have hf := flatten_at x N hNlt
    rw [htake] at hf
    have hlen : (x[N]).length = (D.params.getD r.i default).2.widths[r.j]! := by
      rw [getElem!_pos (D.params.getD r.i default).2.widths r.j hj]
      simp only [List.getElem_map] at hNw
      rw [hNw]
      simp [Wd]
    simp only [Flock.Program.refVal, hrp, ↓reduceIte]
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hNlt, Option.getD_some]
    refine ⟨?_, hlen⟩
    rw [← hAsum]
    conv_rhs => rw [← hf]
    rw [← hlen]
    apply List.map_congr_left
    intro t _
    simp [Nat.add_assoc]

/-! ## The lift of a template of word-primitive calls -/

/-- One node of the walk's fold (`Flock.Program.walkStep`), on `Option`. -/
def nodeStep {W : Type} [Inhabited W] (I : Flock.Program.Wires Option W) (P : Prog)
    (ev : ℕ → List W → Option (List W)) (d : ℕ) (x : List W) (defn : Defn) (outs : Array (List W)) (nd : Node) :
    Option (Array (List W)) := do
  let v := Flock.Program.refVal x (Flock.Program.paramStarts defn.params) outs
  if d ≤ nd.fn then failure
  let callee ← I.lift P.defs[nd.fn]?
  let leaves (i : Nat) : Nat := ((callee.params.getD i default).2).leaves
  match nd.form with
  | .prim | .call => outs.push <$> ev nd.fn (nd.args.toList.flatMap fun a => Flock.Program.argVals v a a.seq.count id)
  | .batch =>
    let ms ← (List.range nd.n).mapM fun m => ev nd.fn ((List.range nd.args.size).flatMap fun i =>
      let a := nd.args.getD i default
      match a.dims with
      | none => Flock.Program.argVals v a a.seq.count id
      | some (dd, L) => Flock.Program.argVals v a (leaves i) fun t => (t / L * dd) * L + t % L + m * L)
    return outs.push ms.flatten
  | .scan =>
    let c := leaves 0
    let init := Flock.Program.argVals v (nd.args.getD 0 default) c id
    let (carry, ys) ← (List.range nd.n).foldlM (init := (init, ([] : List W))) fun (carry, ys) m => do
      let ins := carry ++ ((List.range nd.args.size).drop 1).flatMap fun i =>
        let a := nd.args.getD i default
        if i ≤ nd.nxs then Flock.Program.argVals v a (leaves i) fun t => t + m * leaves i
        else Flock.Program.argVals v a (leaves i) id
      let o ← ev nd.fn ins
      return (o.take c, ys ++ o.drop c)
    return outs.push (carry ++ ys)

theorem walkStep_comp {W : Type} [Inhabited W] (I : Flock.Program.Wires Option W)
    (hI : ∀ {α : Type} (o : Option α), I.lift o = o) (P : Prog) (ev : ℕ → List W → Option (List W)) (d : ℕ)
    (x : List W) {defn : Defn} (hD : P.defs[d]? = some defn) (hp : defn.prim = false) :
    Flock.Program.walkStep I P ev d x = (defn.body.nodes.toList.foldlM (nodeStep I P ev d x defn) #[]) >>= fun outs =>
      pure ((List.range defn.body.ret.count).map fun t =>
        Flock.Program.refVal x (Flock.Program.paramStarts defn.params) outs (defn.body.ret.toArray.getD t default)) := by
  unfold Flock.Program.walkStep
  rw [show I.lift P.defs[d]? = some defn from by rw [hI, hD]]
  simp only [Option.bind_eq_bind, Option.bind_some, hp, Bool.false_eq_true, ↓reduceIte]
  rfl

/-- Definition `D`'s leaves' bits, as `Flock.Lift.liftDef` places them. -/
def atOf (D : Defn) : Bool → ℕ → ℕ → ℕ × ℕ := fun p i =>
  ((if p then (D.params.map fun (_, t) => Flock.Lift.bitAt t)[i]? else
    (D.body.nodes.map fun x => Flock.Lift.bitAt x.out)[i]?).getD fun _ => (0, 0))

/-- Definition `D`'s collections' one width, as `Flock.Lift.liftDef` reads it. -/
def homoOf (D : Defn) : Bool → ℕ → Option ℕ := fun p i =>
  Widths.homo (if p then (D.params[i]?.map (·.2)).getD (.value 1) else (D.body.nodes[i]?.map (·.out)).getD (.value 1))

theorem liftDef_ret (D : Defn) (fnOf : ℕ → ℕ) (primOf : ℕ → Bool) (gatesOf : ℕ → ℕ) :
    (Flock.Lift.liftDef D fnOf primOf gatesOf).body.ret = Flock.Lift.liftSeq (atOf D) (homoOf D) D.body.ret := rfl

theorem liftDef_params (D : Defn) (fnOf : ℕ → ℕ) (primOf : ℕ → Bool) (gatesOf : ℕ → ℕ) :
    (Flock.Lift.liftDef D fnOf primOf gatesOf).params = D.params.map fun (n, t) => (n, Ty.bitType t) := rfl

theorem liftDef_prim (D : Defn) (fnOf : ℕ → ℕ) (primOf : ℕ → Bool) (gatesOf : ℕ → ℕ) :
    (Flock.Lift.liftDef D fnOf primOf gatesOf).prim = D.prim := rfl

theorem liftDef_nodes_size (D : Defn) (fnOf : ℕ → ℕ) (primOf : ℕ → Bool) (gatesOf : ℕ → ℕ) :
    (Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes.size = D.body.nodes.size := by
  simp [Flock.Lift.liftDef]

theorem mapIdx_seqs (xs : Array Arg) (f : ℕ → Arg → Arg) (h : Seq → Seq) (hf : ∀ i a, (f i a).seq = h a.seq) :
    (xs.mapIdx f).toList.map (·.seq) = xs.toList.map fun a => h a.seq := by
  apply List.ext_getElem
  · simp
  · intro n h1 h2
    simp [hf]

theorem liftDef_node (D : Defn) (fnOf : ℕ → ℕ) (primOf : ℕ → Bool) (gatesOf : ℕ → ℕ) {k : ℕ}
    (hk : k < D.body.nodes.size) (hk' : k < (Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes.size) :
    ((Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes[k]).fn = fnOf (D.body.nodes[k]).fn ∧
      (((D.body.nodes[k]).form = .prim ∨ (D.body.nodes[k]).form = .call) →
        ((Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes[k]).form = .prim ∨
          ((Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes[k]).form = .call) ∧
      ((Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes[k]).args.toList.map (·.seq) =
        (D.body.nodes[k]).args.toList.map fun a => Flock.Lift.liftSeq (atOf D) (homoOf D) a.seq := by
  refine ⟨?_, ?_, ?_⟩
  · simp [Flock.Lift.liftDef]
  · intro hf
    simp only [Flock.Lift.liftDef, Array.getElem_map]
    split <;> simp_all
  · simp only [Flock.Lift.liftDef, Array.getElem_map]
    exact mapIdx_seqs _ _ _ (fun _ _ => rfl)

theorem bitAt_homo {t : Ty} {w : ℕ} (h : Widths.homo t = some w) (j : ℕ) : Flock.Lift.bitAt t j = (j * w, w) := by
  unfold Flock.Lift.bitAt
  simp only [h]

theorem atOf_in {D : Defn} {k : ℕ} {r : Ref} (hr : RefIn D k r) :
    atOf D r.p r.i = Flock.Lift.bitAt (tyAt D r) ∧ homoOf D r.p r.i = Widths.homo (tyAt D r) := by
  obtain ⟨hp, hn, -⟩ := hr
  cases hrp : r.p
  · have hi := (hn hrp).2
    simp [atOf, homoOf, tyAt, hrp, Array.getD, hi]
  · have hi := hp hrp
    simp [atOf, homoOf, tyAt, hrp, Array.getD, hi]

/-- **A sequence's bits**: the lift's sequence of a body's word references reads, on bits, the bits of the words they
read. -/
theorem seqBits {D : Defn} {x : List (List Bool)} (hx : x.map List.length = leafWidths D)
    {outs : Array (List (List Bool))} (ho : OutsOk D outs) {s : Seq} (hc : s.count = s.toArray.size)
    (hr : ∀ r ∈ s.toArray.toList, RefIn D outs.size r) :
    (List.range (Flock.Lift.liftSeq (atOf D) (homoOf D) s).count).map (fun t =>
      Flock.Program.refVal x.flatten (Flock.Program.paramStarts (D.params.map fun (n, t) => (n, Ty.bitType t)))
        (outs.map List.flatten) ((Flock.Lift.liftSeq (atOf D) (homoOf D) s).toArray.getD t default)) =
      ((List.range s.count).map fun t =>
        Flock.Program.refVal x (Flock.Program.paramStarts D.params) outs (s.toArray.getD t default)).flatten := by
  have hc' : (Flock.Lift.liftSeq (atOf D) (homoOf D) s).count =
      (Flock.Lift.liftSeq (atOf D) (homoOf D) s).toArray.size := by
    unfold Flock.Lift.liftSeq; exact ofParts_count _
  rw [hc', hc, range_getD_map, range_getD_map]
  rw [liftSeqRefs _ _ s (fun r h w hw j => by
    rw [(atOf_in (hr r h)).1]; rw [(atOf_in (hr r h)).2] at hw; exact bitAt_homo hw j)]
  rw [List.map_flatMap]
  show _ = s.toArray.toList.flatMap _
  apply List.flatMap_congr
  intro r hr'
  rw [List.map_map, (atOf_in (hr r hr')).1]
  exact (refBits hx ho (hr r hr')).1

theorem flatten_flatMap {α β : Type} (l : List α) (f : α → List (List β)) :
    (l.flatMap f).flatten = l.flatMap fun a => (f a).flatten := by
  induction l with
  | nil => rfl
  | cons a l ih => simp [List.flatMap_cons, List.flatten_append, ih]

/-- **A call's arguments' bits**: a lifted node's arguments read, on bits, the bits of its word node's arguments. -/
theorem argsBits {D : Defn} {x : List (List Bool)} (hx : x.map List.length = leafWidths D)
    {outs : Array (List (List Bool))} (ho : OutsOk D outs) {nd nd' : Node}
    (hseq : nd'.args.toList.map (·.seq) = nd.args.toList.map fun a => Flock.Lift.liftSeq (atOf D) (homoOf D) a.seq)
    (ha : ∀ a ∈ nd.args.toList, a.seq.count = a.seq.toArray.size ∧ ∀ r ∈ a.seq.toArray.toList, RefIn D outs.size r) :
    nd'.args.toList.flatMap (fun a => Flock.Program.argVals (Flock.Program.refVal x.flatten
        (Flock.Program.paramStarts (D.params.map fun (n, t) => (n, Ty.bitType t))) (outs.map List.flatten)) a
        a.seq.count id) =
      (nd.args.toList.flatMap fun a =>
        Flock.Program.argVals (Flock.Program.refVal x (Flock.Program.paramStarts D.params) outs) a a.seq.count id).flatten := by
  have e1 : ∀ {W : Type} [Inhabited W] (V : Ref → W) (l : List Arg),
      l.flatMap (fun a => Flock.Program.argVals V a a.seq.count id) =
        (l.map (·.seq)).flatMap (fun s => (List.range s.count).map fun t => V (s.toArray.getD t default)) := by
    intro W _ V l
    rw [List.flatMap_map]
    rfl
  rw [e1, hseq, List.flatMap_map, e1, List.flatMap_map, flatten_flatMap]
  apply List.flatMap_congr
  intro a hmem
  exact seqBits hx ho (ha a hmem).1 (ha a hmem).2

theorem outsOk_push {D : Defn} {outs : Array (List (List Bool))} {w : List (List Bool)} (ho : OutsOk D outs)
    (hw : w.map List.length = (D.body.nodes.getD outs.size default).out.widths.toList) : OutsOk D (outs.push w) := by
  intro i hi
  rw [Array.size_push] at hi
  by_cases h : i < outs.size
  · have := ho i h
    simpa [Array.getD, h, Array.getElem_push, hi] using this
  · obtain rfl : i = outs.size := by omega
    simpa [Array.getD] using hw

theorem fold_calls (D : Defn) (l l' : List Node) (hl : l'.length = l.length)
    (f : Array (List (List Bool)) → Node → Option (Array (List (List Bool))))
    (f' : Array (List Bool) → Node → Option (Array (List Bool)))
    (hstep : ∀ n (hn : n < l.length) (outs : Array (List (List Bool))), outs.size = n → OutsOk D outs →
      ∃ w, f outs l[n] = some (outs.push w) ∧
        f' (outs.map List.flatten) (l'[n]'(by omega)) = some ((outs.push w).map List.flatten) ∧ OutsOk D (outs.push w)) :
    ∀ k n (outs : Array (List (List Bool))), k + n = l.length → outs.size = n → OutsOk D outs →
      ∃ o, (l.drop n).foldlM f outs = some o ∧ (l'.drop n).foldlM f' (outs.map List.flatten) = some (o.map List.flatten) ∧
        OutsOk D o ∧ o.size = l.length
  | 0, n, outs, hk, hs, ho => ⟨outs, by simp [List.drop_eq_nil_of_le (by omega : l.length ≤ n)],
      by simp [List.drop_eq_nil_of_le (by omega : l'.length ≤ n)], ho, by omega⟩
  | k + 1, n, outs, hk, hs, ho => by
    obtain ⟨w, h1, h2, h3⟩ := hstep n (by omega) outs hs ho
    obtain ⟨o, g1, g2, g3, g4⟩ := fold_calls D l l' hl f f' hstep k (n + 1) (outs.push w) (by omega)
      (by simp [hs]) h3
    refine ⟨o, ?_, ?_, g3, g4⟩
    · rw [List.drop_eq_getElem_cons (by omega), List.foldlM_cons, h1]; exact g1
    · rw [List.drop_eq_getElem_cons (by omega : n < l'.length), List.foldlM_cons, h2]; exact g2

/-- **A template of word-primitive calls**: a composite whose every node is a primitive or call node of a word
primitive. -/
def CallsOnly (P : Prog) (d : ℕ) : Prop :=
  (P.defs[d]!).prim = false ∧ ∀ nd ∈ (P.defs[d]!).body.nodes,
    (nd.form = .prim ∨ nd.form = .call) ∧ isWordPrim (P.defs[nd.fn]!).prim (P.defs[nd.fn]!).key = true

/-- **A template of word-primitive nodes**: a composite whose every node calls a word primitive. -/
def PrimNodes (P : Prog) (d : ℕ) : Prop :=
  (P.defs[d]!).prim = false ∧ ∀ nd ∈ (P.defs[d]!).body.nodes, isWordPrim (P.defs[nd.fn]!).prim (P.defs[nd.fn]!).key = true

/-- **A template of word-primitive calls and axis-0 batches**: a composite whose every node calls a word primitive, as a
primitive or call node, or as a batch node each of whose arguments is batched on axis 0 or not batched. -/
def Batch0 (P : Prog) (d : ℕ) : Prop :=
  (P.defs[d]!).prim = false ∧ ∀ nd ∈ (P.defs[d]!).body.nodes,
    (nd.form = .prim ∨ nd.form = .call ∨
      (nd.form = .batch ∧ ∀ i < nd.args.size, nd.axes[i]? = some (some 0) ∨ ∀ ax, nd.axes[i]? ≠ some (some ax))) ∧
    isWordPrim (P.defs[nd.fn]!).prim (P.defs[nd.fn]!).key = true

/-- **A template of axis-0 batches over word primitives, gates and earlier such templates**: `Batch0`, but a node may
also call, as a primitive, call or batch node, a gate or a constant bit (`GateCallee`), or an earlier Definition (a
smaller index) that is itself such a template; some node calls a word primitive or such a template. -/
inductive Batch0Nested (P : Prog) : ℕ → Prop
  | mk {d : ℕ} : (P.defs[d]!).prim = false →
      (∀ nd ∈ (P.defs[d]!).body.nodes,
        nd.form = .prim ∨ nd.form = .call ∨
          (nd.form = .batch ∧ ∀ i < nd.args.size, nd.axes[i]? = some (some 0) ∨ ∀ ax, nd.axes[i]? ≠ some (some ax))) →
      (∀ nd ∈ (P.defs[d]!).body.nodes, (P.defs[nd.fn]!).prim = false → nd.fn < d) →
      (∀ nd ∈ (P.defs[d]!).body.nodes, (P.defs[nd.fn]!).prim = false → Batch0Nested P nd.fn) →
      (∀ nd ∈ (P.defs[d]!).body.nodes, (P.defs[nd.fn]!).prim = true →
        isWordPrim (P.defs[nd.fn]!).prim (P.defs[nd.fn]!).key = false → GateCallee P nd.fn) →
      (∃ nd ∈ (P.defs[d]!).body.nodes,
        (P.defs[nd.fn]!).prim = false ∨ isWordPrim (P.defs[nd.fn]!).prim (P.defs[nd.fn]!).key = true) →
      Batch0Nested P d

theorem reach_batch0 {P : Prog} {d y : ℕ} (h : Batch0 P d) (hy : Reach P d y) : y = d ∨ (P.defs[y]!).prim = true := by
  induction hy with
  | root => exact Or.inl rfl
  | callee _ hp hn ih =>
    rcases ih with rfl | hpt
    · have := (h.2 _ hn).2
      simp only [isWordPrim, Bool.and_eq_true] at this
      exact Or.inr this.1.1
    · rw [hpt] at hp; cases hp

/-- For a template of word-primitive calls and axis-0 batches, `GateBodiesDeep` is `GateBodies`: the only composite
reached from it is itself. -/
theorem gateBodiesDeep_batch0 {P : Prog} {d : ℕ} (h : Batch0 P d) : GateBodiesDeep P d ↔ GateBodies P d :=
  ⟨gateBodies_of_deep h.1, fun hg y hy hp => by
    rcases reach_batch0 h hy with rfl | hpt
    · exact hg
    · rw [hpt] at hp; cases hp⟩

theorem batch0Nested_of_batch0 {P : Prog} {d : ℕ} (h : Batch0 P d) (hn : 0 < (P.defs[d]!).body.nodes.size) :
    Batch0Nested P d :=
  .mk h.1 (fun nd hnd => (h.2 nd hnd).1)
    (fun nd hnd hp => absurd (h.2 nd hnd).2 (by simp [isWordPrim, hp]))
    (fun nd hnd hp => absurd (h.2 nd hnd).2 (by simp [isWordPrim, hp]))
    (fun nd hnd _ hw => absurd (h.2 nd hnd).2 (by rw [hw]; decide))
    ⟨_, Array.getElem_mem hn, Or.inr (h.2 _ (Array.getElem_mem hn)).2⟩

/-- Definition `x` reaches a word primitive: it is one, or it is a composite with a node whose callee reaches one. -/
inductive ReachesWord (S : Prog) : ℕ → Prop
  | word {x : ℕ} : isWordPrim (S.defs[x]!).prim (S.defs[x]!).key = true → ReachesWord S x
  | comp {x : ℕ} {nd : Node} : (S.defs[x]!).prim = false → nd ∈ (S.defs[x]!).body.nodes →
      ReachesWord S nd.fn → ReachesWord S x

/-- `ReachesWord` decided at fuel `f`: false at fuel 0. -/
def reachesWordF (P : Prog) : ℕ → ℕ → Bool
  | 0, _ => false
  | f + 1, d => isWordPrim (P.defs[d]!).prim (P.defs[d]!).key ||
      (!(P.defs[d]!).prim && (P.defs[d]!).body.nodes.any fun nd => reachesWordF P f nd.fn)

theorem reachesWord_of_F (P : Prog) : ∀ f d, reachesWordF P f d = true → ReachesWord P d
  | 0, _, h => by simp [reachesWordF] at h
  | f + 1, d, h => by
    simp only [reachesWordF, Bool.or_eq_true, Bool.and_eq_true, Bool.not_eq_true', Array.any_eq_true'] at h
    rcases h with h | ⟨hp, nd, hmem, hr⟩
    · exact .word h
    · exact .comp hp hmem (reachesWord_of_F P f _ hr)

/-- A type whose every leaf is one bit: `Value<1>`, or an Array of such. -/
def isBit : Ty → Bool
  | .value w => w == 1
  | .array _ e => isBit e
  | _ => false

theorem bitType_of_isBit : ∀ t : Ty, isBit t = true → Ty.bitType t = t
  | .value w, h => by
    simp only [isBit, beq_iff_eq] at h
    subst h
    simp [Ty.bitType]
  | .array n e, h => by
    rw [Ty.bitType, bitType_of_isBit e h]
  | .tuple _, h => by simp [isBit] at h
  | .record _, h => by simp [isBit] at h

/-- Every parameter and node output of `D` is a type of one-bit leaves (`isBit`): its words are its bits. -/
def BitTyped (D : Defn) : Prop :=
  (∀ p ∈ D.params, isBit p.2 = true) ∧ ∀ nd ∈ D.body.nodes, isBit nd.out = true

def bitTypedB (D : Defn) : Bool :=
  D.params.all (fun p => isBit p.2) && D.body.nodes.all (fun nd => isBit nd.out)

theorem bitTyped_of_B {D : Defn} (h : bitTypedB D = true) : BitTyped D := by
  simp only [bitTypedB, Bool.and_eq_true, Array.all_eq_true'] at h
  exact h

/-- **A template of batches on any axes over word primitives, gates and earlier such templates**: `Batch0Nested`, but
a batch node may map each argument along any axis of its type. -/
inductive BatchNested (P : Prog) : ℕ → Prop
  | mk {d : ℕ} : (P.defs[d]!).prim = false →
      (∀ nd ∈ (P.defs[d]!).body.nodes, nd.form = .prim ∨ nd.form = .call ∨ nd.form = .batch) →
      (∀ nd ∈ (P.defs[d]!).body.nodes, (P.defs[nd.fn]!).prim = false → nd.fn < d) →
      (∀ nd ∈ (P.defs[d]!).body.nodes, (P.defs[nd.fn]!).prim = false → BatchNested P nd.fn) →
      (∀ nd ∈ (P.defs[d]!).body.nodes, (P.defs[nd.fn]!).prim = true →
        isWordPrim (P.defs[nd.fn]!).prim (P.defs[nd.fn]!).key = false → GateCallee P nd.fn) →
      (∃ nd ∈ (P.defs[d]!).body.nodes,
        (P.defs[nd.fn]!).prim = false ∨ isWordPrim (P.defs[nd.fn]!).prim (P.defs[nd.fn]!).key = true) →
      BatchNested P d

def noScan : Flock.Program.Form → Bool
  | .scan => false
  | _ => true

theorem noScan_form {f : Flock.Program.Form} (h : noScan f = true) : f = .prim ∨ f = .call ∨ f = .batch := by
  cases f <;> simp_all [noScan]

/-- **`ScanNested`**: `BatchNested` with a node of any form, a scan's step among the composite callees: the Definition
is composite, each composite callee is earlier and in `ScanNested`, each primitive callee no word primitive is a gate
(`GateCallee`), and the Definition reaches a word primitive or is bit-typed. -/
inductive ScanNested (P : Prog) : ℕ → Prop
  | mk {d : ℕ} : (P.defs[d]!).prim = false →
      (∀ nd ∈ (P.defs[d]!).body.nodes, (P.defs[nd.fn]!).prim = false → nd.fn < d) →
      (∀ nd ∈ (P.defs[d]!).body.nodes, (P.defs[nd.fn]!).prim = false → ScanNested P nd.fn) →
      (∀ nd ∈ (P.defs[d]!).body.nodes, (P.defs[nd.fn]!).prim = true →
        isWordPrim (P.defs[nd.fn]!).prim (P.defs[nd.fn]!).key = false → GateCallee P nd.fn) →
      (ReachesWord P d ∨ BitTyped (P.defs[d]!)) →
      ScanNested P d

theorem scanNested_of_batchNested {P : Prog} {d : ℕ} (h : BatchNested P d) : ScanNested P d ∧ ReachesWord P d := by
  induction h with
  | @mk d hp _ hlt _ hgate hex ih =>
    have hrw : ReachesWord P d := by
      obtain ⟨nd, hmem, hc | hw⟩ := hex
      · exact .comp hp hmem (ih nd hmem hc).2
      · exact .comp hp hmem (.word hw)
    exact ⟨.mk hp hlt (fun nd h hc => (ih nd h hc).1) hgate (.inl hrw), hrw⟩

/-- Some node of `d` calls a composite or a word primitive: `BatchNested`'s last clause, decided. -/
def anyWordB (P : Prog) (d : ℕ) : Bool :=
  (P.defs[d]!).body.nodes.any fun nd => !(P.defs[nd.fn]!).prim || isWordPrim (P.defs[nd.fn]!).prim (P.defs[nd.fn]!).key

/-- `ScanNested`'s last clause, decided: `d` reaches a word primitive or is bit-typed. -/
def wordOrBitB (P : Prog) (d : ℕ) : Bool := reachesWordF P (d + 1) d || bitTypedB (P.defs[d]!)

/-- The nested class's clauses at fuel `f`, each node's form passing `ok`, each composite callee's at fuel `f - 1`, the
last clause `ex`; false at fuel 0. -/
def nestedF (ok : Flock.Program.Form → Bool) (ex : Prog → ℕ → Bool) (P : Prog) : ℕ → ℕ → Bool
  | 0, _ => false
  | f + 1, d =>
    !(P.defs[d]!).prim &&
    (P.defs[d]!).body.nodes.all (fun nd => ok nd.form &&
      if (P.defs[nd.fn]!).prim then
        isWordPrim (P.defs[nd.fn]!).prim (P.defs[nd.fn]!).key || (gateKind? (P.defs[nd.fn]!).key).isSome
      else decide (nd.fn < d) && nestedF ok ex P f nd.fn) &&
    ex P d

/-- `BatchNested` decided, at fuel the Definition's index plus one: a composite callee is earlier. -/
def batchNestedB (P : Prog) (d : ℕ) : Bool := nestedF noScan anyWordB P (d + 1) d

/-- `ScanNested` decided, at fuel the Definition's index plus one. -/
def scanNestedB (P : Prog) (d : ℕ) : Bool := nestedF (fun _ => true) wordOrBitB P (d + 1) d

theorem nestedF_succ {ok : Flock.Program.Form → Bool} {ex : Prog → ℕ → Bool} {P : Prog} {f d : ℕ}
    (h : nestedF ok ex P (f + 1) d = true) :
    (P.defs[d]!).prim = false ∧
    (∀ nd ∈ (P.defs[d]!).body.nodes, ok nd.form = true ∧
      ((P.defs[nd.fn]!).prim = false → nd.fn < d ∧ nestedF ok ex P f nd.fn = true) ∧
      ((P.defs[nd.fn]!).prim = true → isWordPrim (P.defs[nd.fn]!).prim (P.defs[nd.fn]!).key = false →
        GateCallee P nd.fn)) ∧
    ex P d = true := by
  simp only [nestedF, Bool.and_eq_true, Bool.not_eq_true', Array.all_eq_true'] at h
  obtain ⟨⟨hp, hall⟩, hex⟩ := h
  refine ⟨hp, fun nd h => ⟨(hall nd h).1, fun hp' => ?_, fun hp' hw => ?_⟩, hex⟩
  · have := (hall nd h).2
    simp only [hp', Bool.false_eq_true, ↓reduceIte, Bool.and_eq_true, decide_eq_true_eq] at this
    exact this
  · have := (hall nd h).2
    rw [hp'] at hw
    simp only [hp', ↓reduceIte, hw, Bool.false_or] at this
    exact ⟨hp', this⟩

theorem nestedF_zero {ok : Flock.Program.Form → Bool} {ex : Prog → ℕ → Bool} {P : Prog} {d : ℕ} :
    nestedF ok ex P 0 d = false := rfl

theorem batchNested_of_F (P : Prog) : ∀ f d, nestedF noScan anyWordB P f d = true → BatchNested P d
  | 0, _, h => by simp [nestedF_zero] at h
  | f + 1, _, h => by
    obtain ⟨hp, hall, hex⟩ := nestedF_succ h
    simp only [anyWordB, Array.any_eq_true'] at hex
    obtain ⟨nd, hmem, hx⟩ := hex
    exact .mk hp (fun nd h => noScan_form (hall nd h).1) (fun nd h hp' => ((hall nd h).2.1 hp').1)
      (fun nd h hp' => batchNested_of_F P f _ ((hall nd h).2.1 hp').2) (fun nd h => (hall nd h).2.2)
      ⟨nd, hmem, by simpa using hx⟩

theorem scanNested_of_F (P : Prog) : ∀ f d, nestedF (fun _ => true) wordOrBitB P f d = true → ScanNested P d
  | 0, _, h => by simp [nestedF_zero] at h
  | f + 1, d, h => by
    obtain ⟨hp, hall, hex⟩ := nestedF_succ h
    simp only [wordOrBitB, Bool.or_eq_true] at hex
    exact .mk hp (fun nd h hp' => ((hall nd h).2.1 hp').1)
      (fun nd h hp' => scanNested_of_F P f _ ((hall nd h).2.1 hp').2) (fun nd h => (hall nd h).2.2)
      (hex.imp (reachesWord_of_F P _ d) bitTyped_of_B)

theorem batchNested_of_B {P : Prog} {d : ℕ} (h : batchNestedB P d = true) : BatchNested P d :=
  batchNested_of_F P (d + 1) d h

theorem scanNested_of_B {P : Prog} {d : ℕ} (h : scanNestedB P d = true) : ScanNested P d :=
  scanNested_of_F P (d + 1) d h

/-- **`LiftCalls`**: the lift to bits of a template of word-primitive nodes, at least one, keeps `S`'s
Definitions and Boolean Definitions and places Definition `d` lifted (`Flock.Lift.liftDef`) after them, each node
calling its primitive's Boolean Definition, where every value of `S`'s Boolean map is a key of its index. A template of
no node reaches no word primitive, so `Flock.Lift.closure` lifts nothing and `lift` returns `d` itself. -/
def LiftCalls : Prop :=
  ∀ (S L : Prog) (d e : ℕ), (∀ (k v : String), S.boolean[k]? = some v → S.index.contains v = true) →
    Flock.Lift.lift S d = .ok (L, e) → d < S.defs.size → PrimNodes S d → 0 < (S.defs[d]!).body.nodes.size →
    S.defs.size ≤ e ∧ (∀ i < S.defs.size, L.defs[i]? = S.defs[i]?) ∧ (∀ k, L.bodyOf? k = S.bodyOf? k) ∧
    ∃ fnOf primOf gatesOf, L.defs[e]? = some (Flock.Lift.liftDef (S.defs[d]!) fnOf primOf gatesOf) ∧
      ∀ nd ∈ (S.defs[d]!).body.nodes, S.bodyOf? (S.defs[nd.fn]!).key = some (fnOf nd.fn)

/-- **`LiftPlaces`** (open): `LiftCalls` but for what `liftKeeps` gives: the lift to bits of a template of
word-primitive nodes, at least one, places Definition `d` lifted after `S`'s Definitions, each node calling its
primitive's Boolean Definition. -/
def LiftPlaces : Prop :=
  ∀ (S L : Prog) (d e : ℕ), (∀ (k v : String), S.boolean[k]? = some v → S.index.contains v = true) →
    Flock.Lift.lift S d = .ok (L, e) → d < S.defs.size → PrimNodes S d → 0 < (S.defs[d]!).body.nodes.size →
    S.defs.size ≤ e ∧ ∃ fnOf primOf gatesOf, L.defs[e]? = some (Flock.Lift.liftDef (S.defs[d]!) fnOf primOf gatesOf) ∧
      ∀ nd ∈ (S.defs[d]!).body.nodes, S.bodyOf? (S.defs[nd.fn]!).key = some (fnOf nd.fn)

/-- **`LiftCalls`**, on `LiftPlaces`. -/
theorem liftCalls_of (h : LiftPlaces) : LiftCalls := fun S L d e hc hl hd hp hn => by
  obtain ⟨he, hex⟩ := h S L d e hc hl hd hp hn
  obtain ⟨-, hpre, hb⟩ := liftKeeps hc hl
  exact ⟨he, hpre, hb, hex⟩

/-- **`Batch0Refs`**: `Batch0Typed`'s references: in a C `decode` accepts whose root has no recut, a template of
word-primitive calls and axis-0 batches reads each of its parameters' and earlier nodes' leaves that exist, in sequences
whose count is their references'. -/
def Batch0Refs : Prop :=
  ∀ (bytes : ByteArray) (P : Prog), decode bytes = .ok P → (P.defs[P.root]!).body.recut = #[] →
    ∀ d < P.defs.size, Batch0 P d →
      (∀ k (hk : k < (P.defs[d]!).body.nodes.size), ∀ a ∈ ((P.defs[d]!).body.nodes[k]).args.toList,
        a.seq.count = a.seq.toArray.size ∧ ∀ r ∈ a.seq.toArray.toList, RefIn (P.defs[d]!) k r) ∧
      ((P.defs[d]!).body.ret.count = (P.defs[d]!).body.ret.toArray.size ∧
        ∀ r ∈ (P.defs[d]!).body.ret.toArray.toList, RefIn (P.defs[d]!) (P.defs[d]!).body.nodes.size r)

/-- **`Batch0Callees`**: in a C `decode` accepts whose root has no recut, in a template of word-primitive calls and axis-0
batches, every node of the template calls a Definition before it. -/
def Batch0Callees : Prop :=
  ∀ (bytes : ByteArray) (P : Prog), decode bytes = .ok P → (P.defs[P.root]!).body.recut = #[] →
    ∀ d < P.defs.size, Batch0 P d →
      (∀ nd ∈ (P.defs[d]!).body.nodes, nd.fn < d)

/-- **`Batch0OutWidths`**: in a C `decode` accepts whose root has no recut, in a template of word-primitive calls and axis-0
batches, each node's output is its members' words of its callee's value's width. -/
def Batch0OutWidths : Prop :=
  ∀ (bytes : ByteArray) (P : Prog), decode bytes = .ok P → (P.defs[P.root]!).body.recut = #[] →
    ∀ d < P.defs.size, Batch0 P d → GateBodies P d →
      (∀ k (hk : k < (P.defs[d]!).body.nodes.size), ∀ ys : List (List Bool),
        ys.length = ((P.defs[d]!).body.nodes[k]).members →
        ((P.defs[d]!).body.nodes[k]).out.widths.toList =
          ys.map fun y => (Prog.evalBool P ((P.defs[d]!).body.nodes[k]).fn y).length)

/-- **`Batch0BoolEval`**: in a C `decode` accepts whose root has no recut, in a template of word-primitive calls and axis-0
batches whose primitives' Boolean Definitions evaluate through gates alone (`GateBodies`), each of them evaluates on
every input. -/
def Batch0BoolEval : Prop :=
  ∀ (bytes : ByteArray) (P : Prog), decode bytes = .ok P → (P.defs[P.root]!).body.recut = #[] →
    ∀ d < P.defs.size, Batch0 P d → GateBodies P d →
      (∀ nd ∈ (P.defs[d]!).body.nodes, ∀ b, P.bodyOf? (P.defs[nd.fn]!).key = some b →
        ∀ y, (Flock.Program.evalDef? P (b + 1) b y).isSome = true)

/-- **`Batch0Axes`**: in a C `decode` accepts whose root has no recut, in a template of word-primitive calls and axis-0
batches, each axis-0 batch argument is an array of the batch's length of its callee's parameter, its references of
that parameter's leaf widths repeated, and any other batch argument is not batched. -/
def Batch0Axes : Prop :=
  ∀ (bytes : ByteArray) (P : Prog), decode bytes = .ok P → (P.defs[P.root]!).body.recut = #[] →
    ∀ d < P.defs.size, Batch0 P d →
      (∀ k (hk : k < (P.defs[d]!).body.nodes.size), ((P.defs[d]!).body.nodes[k]).form = .batch →
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
            (((P.defs[d]!).body.nodes[k]).args.getD i default).dims = none))

/-- **`Batch0Calls`**: `Batch0Typed` but for its references (`Batch0Refs`). -/
def Batch0Calls : Prop :=
  ∀ (bytes : ByteArray) (P : Prog), decode bytes = .ok P → (P.defs[P.root]!).body.recut = #[] →
    ∀ d < P.defs.size, Batch0 P d → GateBodies P d →
      (∀ nd ∈ (P.defs[d]!).body.nodes, nd.fn < d) ∧
      (∀ k (hk : k < (P.defs[d]!).body.nodes.size), ∀ ys : List (List Bool),
        ys.length = ((P.defs[d]!).body.nodes[k]).members →
        ((P.defs[d]!).body.nodes[k]).out.widths.toList =
          ys.map fun y => (Prog.evalBool P ((P.defs[d]!).body.nodes[k]).fn y).length) ∧
      (∀ nd ∈ (P.defs[d]!).body.nodes, ∀ b, P.bodyOf? (P.defs[nd.fn]!).key = some b →
        ∀ y, (Flock.Program.evalDef? P (b + 1) b y).isSome = true) ∧
      (∀ k (hk : k < (P.defs[d]!).body.nodes.size), ((P.defs[d]!).body.nodes[k]).form = .batch →
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
            (((P.defs[d]!).body.nodes[k]).args.getD i default).dims = none))

/-- **`Batch0Calls`**, on its four clauses. -/
theorem batch0Calls_of (h1 : Batch0Callees) (h2 : Batch0OutWidths) (h3 : Batch0BoolEval) (h4 : Batch0Axes) :
    Batch0Calls := fun bytes P h hr d hd hb hgb =>
  ⟨h1 bytes P h hr d hd hb, h2 bytes P h hr d hd hb hgb, h3 bytes P h hr d hd hb hgb, h4 bytes P h hr d hd hb⟩

/-- **`Batch0Typed`**: in a C `decode` accepts whose root has no recut, a template of word-primitive calls and
axis-0 batches reads each of its parameters' and earlier nodes' leaves that exist, in sequences whose count is their
references', calls only Definitions before it, each node's output its members' words of its primitive's value's width,
each Boolean Definition of its primitives evaluates, and each axis-0 batch argument is an array of the batch's length of
its primitive's parameter, its references of that parameter's leaf widths repeated, and any other batch argument is not
batched. -/
def Batch0Typed : Prop :=
  ∀ (bytes : ByteArray) (P : Prog), decode bytes = .ok P → (P.defs[P.root]!).body.recut = #[] →
    ∀ d < P.defs.size, Batch0 P d → GateBodies P d →
      (∀ k (hk : k < (P.defs[d]!).body.nodes.size), ∀ a ∈ ((P.defs[d]!).body.nodes[k]).args.toList,
        a.seq.count = a.seq.toArray.size ∧ ∀ r ∈ a.seq.toArray.toList, RefIn (P.defs[d]!) k r) ∧
      ((P.defs[d]!).body.ret.count = (P.defs[d]!).body.ret.toArray.size ∧
        ∀ r ∈ (P.defs[d]!).body.ret.toArray.toList, RefIn (P.defs[d]!) (P.defs[d]!).body.nodes.size r) ∧
      (∀ nd ∈ (P.defs[d]!).body.nodes, nd.fn < d) ∧
      (∀ k (hk : k < (P.defs[d]!).body.nodes.size), ∀ ys : List (List Bool),
        ys.length = ((P.defs[d]!).body.nodes[k]).members →
        ((P.defs[d]!).body.nodes[k]).out.widths.toList =
          ys.map fun y => (Prog.evalBool P ((P.defs[d]!).body.nodes[k]).fn y).length) ∧
      (∀ nd ∈ (P.defs[d]!).body.nodes, ∀ b, P.bodyOf? (P.defs[nd.fn]!).key = some b →
        ∀ y, (Flock.Program.evalDef? P (b + 1) b y).isSome = true) ∧
      (∀ k (hk : k < (P.defs[d]!).body.nodes.size), ((P.defs[d]!).body.nodes[k]).form = .batch →
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
            (((P.defs[d]!).body.nodes[k]).args.getD i default).dims = none))

/-- **`Batch0Typed`**, on `Batch0Refs` and `Batch0Calls`. -/
theorem batch0Typed_of (hR : Batch0Refs) (hC : Batch0Calls) : Batch0Typed := fun bytes P h hr d hd hb hgb =>
  ⟨(hR bytes P h hr d hd hb).1, (hR bytes P h hr d hd hb).2, hC bytes P h hr d hd hb hgb⟩

/-- **`Batch0BoolEval`** holds. -/
theorem batch0BoolEval : Batch0BoolEval := fun bytes P h hr _ _ _ hgb nd hn b hb y => by
  obtain ⟨⟨hmap, hkeys⟩, -⟩ := FlockSoundness.DecodeBoolean.decode_facts bytes P h hr
  exact gateDef_total P b (body_lt ⟨hmap, hkeys⟩ hb) (hgb nd hn b hb) y

theorem boolCall {bytes : ByteArray} {P L : Prog} (h : decode bytes = .ok P) (hr : (P.defs[P.root]!).body.recut = #[])
    (hle : P.defs.size ≤ L.defs.size) (hpre : ∀ i < P.defs.size, L.defs[i]! = P.defs[i]!)
    (hbody : ∀ k, L.bodyOf? k = P.bodyOf? k) {f b : ℕ} (hf : f < P.defs.size)
    (hw : isWordPrim (P.defs[f]!).prim (P.defs[f]!).key = true) (hb : P.bodyOf? (P.defs[f]!).key = some b)
    (hs : ∀ y, (Flock.Program.evalDef? P (b + 1) b y).isSome = true) (y : List Bool) :
    Flock.Program.evalDef? L (2 * L.defs.size + 1) b y = some (Prog.evalBool P f y) := by
  obtain ⟨⟨hmap, hkeys⟩, hids⟩ := FlockSoundness.DecodeBoolean.decode_facts bytes P h hr
  have hm : P.BoolMapOk := ⟨hmap, hkeys⟩
  have hbl := body_lt hm hb
  have hbd : P.BoolDef b := by
    obtain ⟨b', -, hbo, hbody', -⟩ := hmap f hf hw
    obtain rfl : b' = b := Option.some.inj (hbo.symm.trans hb)
    exact hbody'
  have hB := FlockSoundness.DecodeBoolean.bool_none hkeys
  have hBL : ∀ k, isBooleanPrim k = true → L.bodyOf? k = none := fun k hk => by rw [hbody]; exact hB k hk
  have hbdL : L.BoolDef b := boolDef_lift hpre hbd hbl
  obtain ⟨v, hv⟩ := Option.isSome_iff_exists.1 (hs y)
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
  have hE : Prog.evalBool P f y = v := by
    unfold Prog.evalBool
    rw [hP]
    rfl
  rw [hE, evalDef_fuel_at L hBL hbdL (by omega) y,
    evalDef_restrict hle hpre hbody (fun k b' h' => body_lt hm h') (b + 1) b y hbl, hv]

theorem wordDef_prim {P : Prog} {f fuel : ℕ} (hf : f < P.defs.size) (hprim : (P.defs[f]!).prim = true)
    (ws : List (List Bool)) : wordDef? P (fuel + 1) f ws = some [Prog.evalBool P f ws.flatten] := by
  have hdf : P.defs[f]? = some (P.defs[f]!) := by rw [getElem!_pos P.defs f hf]; exact getElem?_pos P.defs f hf
  unfold wordDef?
  simp [hdf, hprim]

theorem scanCarryBits : ScanCarryBits := by
  intro d fnOf primOf gatesOf k x hk hs
  obtain ⟨hk', hx⟩ := Array.getElem?_eq_some_iff.1 hk
  have hk'' : k < (Flock.Lift.liftDef d fnOf primOf gatesOf).body.nodes.size := by
    rw [liftDef_nodes_size]; exact hk'
  refine ⟨(Flock.Lift.liftDef d fnOf primOf gatesOf).body.nodes[k], by simp, ?_, ?_, ?_, ?_⟩
  · simp [Flock.Lift.liftDef, hx]
  · have e1 : (Form.scan == Form.prim) = false := rfl
    have e2 : (Form.scan == Form.scan) = true := rfl
    have e3 : (Form.scan == Form.batch) = false := rfl
    simp [Flock.Lift.liftDef, hx, hs, Node.members, e1, e2, e3]
  · simp [Flock.Lift.liftDef, hx]
  · intro i hi
    have hi' : i < ((Flock.Lift.liftDef d fnOf primOf gatesOf).body.nodes[k]).args.size := by
      simp [Flock.Lift.liftDef, hx, hi]
    rw [getElem!_pos _ i hi', getElem!_pos x.args i hi]
    simp [Flock.Lift.liftDef, hx]
theorem list_range_getD_map {α β : Type} [Inhabited α] (l : List α) (f : α → β) :
    (List.range l.length).map (fun t => f (l.getD t default)) = l.map f := by
  apply List.ext_getElem (by simp)
  intro t h1 _
  simp only [List.length_map, List.length_range] at h1
  simp [List.getD_eq_getElem?_getD, h1]

theorem replicate_flatten_split {α : Type} (ew : List α) {m n : ℕ} (hm : m ≤ n) :
    (List.replicate n ew).flatten = (List.replicate m ew).flatten ++ (List.replicate (n - m) ew).flatten := by
  conv_lhs => rw [show n = m + (n - m) by omega, List.replicate_add, List.flatten_append]

theorem length_replicate_flatten {α : Type} (ew : List α) (m : ℕ) :
    (List.replicate m ew).flatten.length = m * ew.length := by
  induction m with
  | zero => simp
  | succ m ih => rw [List.replicate_succ, List.flatten_cons, List.length_append, ih, Nat.succ_mul]; omega

theorem sum_replicate_flatten (ew : List ℕ) (m : ℕ) : (List.replicate m ew).flatten.sum = m * ew.sum := by
  induction m with
  | zero => simp
  | succ m ih => rw [List.replicate_succ, List.flatten_cons, List.sum_append, ih, Nat.succ_mul]; omega

theorem length_flatMap_wid {α : Type} (G : α → List α) (wid : α → ℕ) :
    ∀ l : List α, (∀ r ∈ l, (G r).length = wid r) → (l.flatMap G).length = (l.map wid).sum
  | [], _ => rfl
  | r :: l, hl => by
    rw [List.flatMap_cons, List.length_append, List.map_cons, List.sum_cons, hl r (by simp),
      length_flatMap_wid G wid l (fun q h => hl q (by simp [h]))]

theorem flatMap_flatten_map {α β : Type} (l : List α) (f : α → List β) : l.flatMap f = (l.map f).flatten := by
  induction l with
  | nil => rfl
  | cons a l ih => rw [List.flatMap_cons, ih]; rfl

/-- **An axis-0 slice's bits**: in a sequence of references whose widths repeat `ew` `n` times, the bits of slice `m`
of the bit sequence (each reference `r` put as its bits `G r`) are the words of slice `m`, flattened. -/
theorem block_bits {α β : Type} [Inhabited α] (refs : List α) (G : α → List α) (Vb : α → β) (Vw : α → List β)
    (wid : α → ℕ) (ew : List ℕ) (n m : ℕ) (hm : m < n)
    (hG : ∀ r ∈ refs, (G r).map Vb = Vw r ∧ (Vw r).length = wid r)
    (hw : refs.map wid = (List.replicate n ew).flatten) :
    (List.range ew.sum).map (fun t => Vb ((refs.flatMap G).getD (t + m * ew.sum) default)) =
      ((List.range ew.length).map fun t => Vw (refs.getD (t + m * ew.length) default)).flatten := by
  have hGl : ∀ r ∈ refs, (G r).length = wid r := fun r h => by
    rw [← (hG r h).2, ← (hG r h).1, List.length_map]
  have hsplit := replicate_flatten_split ew (Nat.le_of_lt hm)
  obtain ⟨k, hk⟩ : ∃ k, n - m = k + 1 := ⟨n - m - 1, by omega⟩
  have hA : (refs.take (m * ew.length)).map wid = (List.replicate m ew).flatten := by
    rw [List.map_take, hw, hsplit, List.take_left' (length_replicate_flatten ew m)]
  have hR : (refs.drop (m * ew.length)).map wid = ew ++ (List.replicate k ew).flatten := by
    rw [List.map_drop, hw, hsplit, List.drop_left' (length_replicate_flatten ew m), hk, List.replicate_succ,
      List.flatten_cons]
  have hB : ((refs.drop (m * ew.length)).take ew.length).map wid = ew := by
    rw [List.map_take, hR, List.take_left' rfl]
  have hAlen : (refs.take (m * ew.length)).length = m * ew.length := by
    have := congrArg List.length hA
    rw [List.length_map, length_replicate_flatten] at this
    exact this
  have hBlen : ((refs.drop (m * ew.length)).take ew.length).length = ew.length := by
    have := congrArg List.length hB
    rw [List.length_map] at this
    exact this
  have hmA : ∀ r ∈ refs.take (m * ew.length), (G r).length = wid r := fun r h => hGl r (List.mem_of_mem_take h)
  have hmB : ∀ r ∈ (refs.drop (m * ew.length)).take ew.length, r ∈ refs :=
    fun r h => List.mem_of_mem_drop (List.mem_of_mem_take h)
  have hAfl : ((refs.take (m * ew.length)).flatMap G).length = m * ew.sum := by
    rw [length_flatMap_wid G wid _ hmA, hA, sum_replicate_flatten]
  have hBfl : (((refs.drop (m * ew.length)).take ew.length).flatMap G).length = ew.sum := by
    rw [length_flatMap_wid G wid _ (fun r h => hGl r (hmB r h)), hB]
  have hrefs : refs = refs.take (m * ew.length) ++
      (((refs.drop (m * ew.length)).take ew.length) ++ ((refs.drop (m * ew.length)).drop ew.length)) := by
    rw [List.take_append_drop, List.take_append_drop]
  have e1 : ∀ t < ew.sum, (refs.flatMap G).getD (t + m * ew.sum) default =
      (((refs.drop (m * ew.length)).take ew.length).flatMap G).getD t default := by
    intro t ht
    conv_lhs => rw [hrefs]
    rw [List.flatMap_append, List.flatMap_append, List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD,
      List.getElem?_append_right (by omega), hAfl, show t + m * ew.sum - m * ew.sum = t by omega,
      List.getElem?_append_left (by omega)]
  have e2 : ∀ t < ew.length, refs.getD (t + m * ew.length) default =
      ((refs.drop (m * ew.length)).take ew.length).getD t default := by
    intro t ht
    conv_lhs => rw [hrefs]
    rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD,
      List.getElem?_append_right (by omega), hAlen, show t + m * ew.length - m * ew.length = t by omega,
      List.getElem?_append_left (by omega)]
  calc (List.range ew.sum).map (fun t => Vb ((refs.flatMap G).getD (t + m * ew.sum) default))
      = (List.range ew.sum).map (fun t => Vb ((((refs.drop (m * ew.length)).take ew.length).flatMap G).getD t
          default)) := List.map_congr_left (fun t ht => by rw [e1 t (List.mem_range.1 ht)])
    _ = (((refs.drop (m * ew.length)).take ew.length).flatMap G).map Vb := by
        conv_lhs => rw [← hBfl]
        exact list_range_getD_map _ _
    _ = (((refs.drop (m * ew.length)).take ew.length).map Vw).flatten := by
        rw [List.map_flatMap]
        rw [show (((refs.drop (m * ew.length)).take ew.length).flatMap fun a => (G a).map Vb) =
            ((refs.drop (m * ew.length)).take ew.length).flatMap Vw from
          List.flatMap_congr (fun r h => (hG r (hmB r h)).1)]
        exact flatMap_flatten_map _ _
    _ = _ := by
        congr 1
        conv_rhs => rw [← hBlen]
        rw [← list_range_getD_map]
        rw [hBlen]
        exact List.map_congr_left (fun t ht => by rw [e2 t (List.mem_range.1 ht)])
theorem members_one {nd : Node} (h : nd.form = .prim ∨ nd.form = .call) : nd.members = 1 := by
  rcases h with h | h <;> simp only [Node.members, h] <;> rfl

theorem members_batch {nd : Node} (h : nd.form = .batch) : nd.members = nd.n := by
  simp only [Node.members, h]; rfl

theorem mapM_some {α β : Type} (f : α → Option β) (g : α → β) :
    ∀ l : List α, (∀ a ∈ l, f a = some (g a)) → l.mapM f = some (l.map g)
  | [], _ => rfl
  | a :: l, h => by
    rw [List.mapM_cons, h a (by simp), mapM_some f g l (fun b hb => h b (by simp [hb]))]
    rfl

theorem flatten_map_singleton {α β : Type} (l : List α) (g : α → β) : (l.map fun a => [g a]).flatten = l.map g := by
  induction l with
  | nil => rfl
  | cons a l ih => rw [List.map_cons, List.flatten_cons, ih]; rfl

/-- Argument `i` of member `m` of batch node `nd` calling `callee`, as the walk reads it. -/
def batchArg {W : Type} [Inhabited W] (v : Ref → W) (callee : Defn) (nd : Node) (m i : ℕ) : List W :=
  match (nd.args.getD i default).dims with
  | none => Flock.Program.argVals v (nd.args.getD i default) (nd.args.getD i default).seq.count id
  | some (dd, L) => Flock.Program.argVals v (nd.args.getD i default) ((callee.params.getD i default).2).leaves
      fun t => (t / L * dd) * L + t % L + m * L

theorem batchArg_none {W : Type} [Inhabited W] (v : Ref → W) (callee : Defn) (nd : Node) (m i : ℕ)
    (h : (nd.args.getD i default).dims = none) :
    batchArg v callee nd m i = Flock.Program.argVals v (nd.args.getD i default) (nd.args.getD i default).seq.count id := by
  simp only [batchArg, h]

theorem batchArg_some {W : Type} [Inhabited W] (v : Ref → W) (callee : Defn) (nd : Node) (m i : ℕ) {dd L : ℕ}
    (h : (nd.args.getD i default).dims = some (dd, L)) :
    batchArg v callee nd m i = Flock.Program.argVals v (nd.args.getD i default) ((callee.params.getD i default).2).leaves
      fun t => (t / L * dd) * L + t % L + m * L := by
  simp only [batchArg, h]

theorem nodeStep_batch {W : Type} [Inhabited W] (I : Flock.Program.Wires Option W)
    (hI : ∀ {α : Type} (o : Option α), I.lift o = o) (P : Prog) (ev : ℕ → List W → Option (List W)) (d : ℕ)
    (x : List W) (defn : Defn) (outs : Array (List W)) (nd : Node) {callee : Defn} (hf : nd.form = .batch)
    (hlt : nd.fn < d) (hc : P.defs[nd.fn]? = some callee) :
    nodeStep I P ev d x defn outs nd = ((List.range nd.n).mapM fun m => ev nd.fn ((List.range nd.args.size).flatMap
      fun i => batchArg (Flock.Program.refVal x (Flock.Program.paramStarts defn.params) outs) callee nd m i)) >>=
        fun ms => pure (outs.push ms.flatten) := by
  unfold nodeStep
  rw [if_neg (by omega), show I.lift P.defs[nd.fn]? = some callee from by rw [hI, hc]]
  simp only [Option.bind_eq_bind, Option.bind_some, hf]
  rfl

theorem liftDef_arg (D : Defn) (fnOf : ℕ → ℕ) (primOf : ℕ → Bool) (gatesOf : ℕ → ℕ) {k i : ℕ}
    (hk : k < D.body.nodes.size) (hk' : k < (Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes.size)
    (hi : i < (D.body.nodes[k]).args.size) :
    (((Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes[k]).args.getD i default).seq =
        Flock.Lift.liftSeq (atOf D) (homoOf D) ((D.body.nodes[k]).args.getD i default).seq ∧
      ((D.body.nodes[k]).form = .batch → (D.body.nodes[k]).axes[i]? = some (some 0) →
        (((Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes[k]).args.getD i default).dims =
          (Ty.bitType ((D.body.nodes[k]).args.getD i default).ty).axisDims 0) ∧
      ((D.body.nodes[k]).form = .batch → (∀ ax, (D.body.nodes[k]).axes[i]? ≠ some (some ax)) →
        (((Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes[k]).args.getD i default).dims = none) ∧
      ((Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes[k]).args.size = (D.body.nodes[k]).args.size := by
  have hsz : ((Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes[k]).args.size = (D.body.nodes[k]).args.size := by
    simp [Flock.Lift.liftDef]
  have hi' : i < ((Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes[k]).args.size := by rw [hsz]; exact hi
  have hg : ∀ (A : Array Arg) (h : i < A.size), A.getD i default = A[i] := fun A h => by simp [Array.getD, h]
  rw [hg _ hi', hg _ hi]
  refine ⟨?_, ?_, ?_, hsz⟩
  · simp only [Flock.Lift.liftDef, Array.getElem_map, Array.getElem_mapIdx]
    rfl
  · intro hf hax
    simp [Flock.Lift.liftDef, hf, hax]
  · intro hf hax
    match h : (D.body.nodes[k]).axes[i]? with
    | none => simp [Flock.Lift.liftDef, hf, h]
    | some none => simp [Flock.Lift.liftDef, hf, h]
    | some (some ax) => exact absurd h (hax ax)

/-- **An axis-0 batch argument's bits**: member `m` of an argument batched on axis 0, whose references' widths repeat
`ew` `n` times, reads on bits the bits of the words it reads. -/
theorem axisBits {D : Defn} {x : List (List Bool)} (hx : x.map List.length = leafWidths D)
    {outs : Array (List (List Bool))} (ho : OutsOk D outs) {s : Seq}
    (hr : ∀ r ∈ s.toArray.toList, RefIn D outs.size r) {n m : ℕ} (hm : m < n) {ew : List ℕ}
    (hw : s.toArray.toList.map (fun r => (tyAt D r).widths[r.j]!) = (List.replicate n ew).flatten) :
    (List.range ew.sum).map (fun t =>
      Flock.Program.refVal x.flatten (Flock.Program.paramStarts (D.params.map fun (n, t) => (n, Ty.bitType t)))
        (outs.map List.flatten)
        ((Flock.Lift.liftSeq (atOf D) (homoOf D) s).toArray.getD (t / ew.sum * n * ew.sum + t % ew.sum + m * ew.sum)
          default)) =
      ((List.range ew.length).map fun t =>
        Flock.Program.refVal x (Flock.Program.paramStarts D.params) outs
          (s.toArray.getD (t / ew.length * n * ew.length + t % ew.length + m * ew.length) default)).flatten := by
  have hA : ∀ (A : Array Ref) (i : ℕ), A.getD i default = A.toList.getD i default := by
    intro A i
    rw [Array.getD_eq_getD_getElem?, ← Array.getElem?_toList, List.getD_eq_getElem?_getD]
  have i1 : ∀ (L t : ℕ), t < L → t / L * n * L + t % L + m * L = t + m * L := by
    intro L t ht
    rw [Nat.div_eq_of_lt ht, Nat.mod_eq_of_lt ht]; simp
  have hG : ∀ r ∈ s.toArray.toList,
      ((List.range (atOf D r.p r.i r.j).2).map fun t => ({ p := r.p, i := r.i, j := (atOf D r.p r.i r.j).1 + t } : Ref)).map
        (Flock.Program.refVal x.flatten (Flock.Program.paramStarts (D.params.map fun (n, t) => (n, Ty.bitType t)))
          (outs.map List.flatten)) = Flock.Program.refVal x (Flock.Program.paramStarts D.params) outs r ∧
      (Flock.Program.refVal x (Flock.Program.paramStarts D.params) outs r).length = (tyAt D r).widths[r.j]! := by
    intro r h
    rw [List.map_map, (atOf_in (hr r h)).1]
    exact refBits hx ho (hr r h)
  have hls := liftSeqRefs (atOf D) (homoOf D) s (fun r h w hw j => by
    rw [(atOf_in (hr r h)).1]; rw [(atOf_in (hr r h)).2] at hw; exact bitAt_homo hw j)
  have key := block_bits s.toArray.toList _ _ _ (fun r => (tyAt D r).widths[r.j]!) ew n m hm hG hw
  rw [← hls] at key
  refine (List.map_congr_left (fun t ht => ?_)).trans (key.trans ?_)
  · rw [i1 _ t (List.mem_range.1 ht), hA]
  · congr 1
    exact List.map_congr_left (fun t ht => by rw [i1 _ t (List.mem_range.1 ht), hA])

theorem mapM_some_map {α β : Type} (g : α → β) (l : List α) : l.mapM (fun a => some (g a)) = some (l.map g) :=
  mapM_some _ g l (fun _ _ => rfl)

/-- **A batch member's arguments' bits**, `C` its callee on words and `C'` on bits: member `m` of a lifted batch node
reads, on bits, the bits of the words its word node's member reads. -/
theorem batchBits {D : Defn} {fnOf : ℕ → ℕ} {primOf : ℕ → Bool} {gatesOf : ℕ → ℕ} {x : List (List Bool)}
    (hx : x.map List.length = leafWidths D) {outs : Array (List (List Bool))} (ho : OutsOk D outs) {n : ℕ}
    (hnD : n < D.body.nodes.size) (hnD' : n < (Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes.size)
    (hsz : outs.size = n) (hf : (D.body.nodes[n]).form = .batch) {C C' : Defn}
    (hpar : C'.params.map (·.2) = C.params.map (·.2.bitType))
    (hargs : ∀ a ∈ (D.body.nodes[n]).args.toList,
      a.seq.count = a.seq.toArray.size ∧ ∀ r ∈ a.seq.toArray.toList, RefIn D n r)
    (hax : ∀ i < (D.body.nodes[n]).args.size,
      (D.body.nodes[n]).axes[i]? = some (some 0) ∨ ∀ ax, (D.body.nodes[n]).axes[i]? ≠ some (some ax))
    (hbat : ∀ i < (D.body.nodes[n]).args.size,
      ((D.body.nodes[n]).axes[i]? = some (some 0) → ∃ E : Ty,
        ((D.body.nodes[n]).args.getD i default).ty = .array (D.body.nodes[n]).n E ∧
        ((D.body.nodes[n]).args.getD i default).dims = some ((D.body.nodes[n]).n, E.leaves) ∧
        i < C.params.size ∧ (C.params.getD i default).2 = E ∧
        ((D.body.nodes[n]).args.getD i default).seq.toArray.toList.map (fun r => (tyAt D r).widths[r.j]!) =
          (List.replicate (D.body.nodes[n]).n E.widths.toList).flatten) ∧
      ((∀ ax, (D.body.nodes[n]).axes[i]? ≠ some (some ax)) → ((D.body.nodes[n]).args.getD i default).dims = none))
    {m : ℕ} (hm' : m < (D.body.nodes[n]).n) :
    ((List.range ((Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes[n]).args.size).flatMap fun i =>
        batchArg (Flock.Program.refVal x.flatten
          (Flock.Program.paramStarts (D.params.map fun (n, t) => (n, Ty.bitType t))) (outs.map List.flatten))
          C' ((Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes[n]) m i) =
      ((List.range (D.body.nodes[n]).args.size).flatMap fun i =>
          batchArg (Flock.Program.refVal x (Flock.Program.paramStarts D.params) outs) C (D.body.nodes[n]) m i).flatten := by
  have hasz : ((Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes[n]).args.size = (D.body.nodes[n]).args.size := by
    simp [Flock.Lift.liftDef]
  rw [hasz, flatten_flatMap]
  apply List.flatMap_congr
  intro i hi
  have hi' := List.mem_range.1 hi
  obtain ⟨hseqi, hax0, haxn, -⟩ := liftDef_arg D fnOf primOf gatesOf hnD hnD' hi'
  have hmemA : (D.body.nodes[n]).args.getD i default ∈ (D.body.nodes[n]).args.toList := by
    simp [Array.getD, hi']
  obtain ⟨hcnt, hrefs⟩ := hargs _ hmemA
  have hrefs' : ∀ r ∈ ((D.body.nodes[n]).args.getD i default).seq.toArray.toList, RefIn D outs.size r :=
    fun r h => by rw [hsz]; exact hrefs r h
  rcases hax i hi' with h0 | hn0
  · obtain ⟨E, hty, hdims, hip, hpe, hw⟩ := (hbat i hi').1 h0
    have hd' : (((Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes[n]).args.getD i default).dims =
        some ((D.body.nodes[n]).n, (Ty.bitType E).leaves) := by
      rw [hax0 hf h0, hty]; simp [Ty.bitType, Ty.axisDims]
    have hsb : C'.params.size = C.params.size := by simpa using congrArg Array.size hpar
    have hib : i < C'.params.size := by rw [hsb]; exact hip
    have hpb : (C'.params.getD i default).2 = Ty.bitType E := by
      have h1 := congrArg (·[i]?) hpar
      simp only [Array.getElem?_map] at h1
      rw [Array.getElem?_eq_getElem hib, Array.getElem?_eq_getElem hip] at h1
      rw [← hpe, Array.getD_eq_getD_getElem?, Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hib,
        Array.getElem?_eq_getElem hip]
      simpa using h1
    rw [batchArg_some _ _ _ _ _ hd', batchArg_some _ _ _ _ _ hdims, hpb, hpe]
    simp only [Flock.Program.argVals]
    rw [hseqi, ← sum_widths E, show E.leaves = E.widths.toList.length by
      rw [Array.length_toList, hb_widths_size]]
    exact axisBits hx ho hrefs' hm' hw
  · rw [batchArg_none _ _ _ _ _ (haxn hf hn0), batchArg_none _ _ _ _ _ ((hbat i hi').2 hn0)]
    simp only [Flock.Program.argVals]
    rw [hseqi]
    exact seqBits hx ho hcnt hrefs'

/-- **`LiftWordEval` for a template of word-primitive calls and axis-0 batches**, at least one node, on `LiftCalls` and
`Batch0Typed`. -/
theorem liftWordEval_batch0 (hLC : LiftCalls) (hT : Batch0Typed) :
    ∀ (bytes : ByteArray) (P : Prog) (units : Array ℕ) (S : Prog) (d : ℕ) (L : Prog) (e : ℕ),
      decode bytes = .ok P → (P.defs[P.root]!).body.recut = #[] → Spot.Restrict.restrict P units = .ok S →
      d < P.defs.size → Batch0 P d → GateBodies P d → 0 < (P.defs[d]!).body.nodes.size →
      Flock.Lift.lift S d = .ok (L, e) →
      ∀ x : List (List Bool), x.map List.length = leafWidths (P.defs[d]!) →
        Prog.evalBool L e x.flatten = (wordEval P d x).flatten := by
  intro bytes P units S d L e h hr hs hd hco hgb hne hl x hx
  obtain ⟨hle, hpre, hbool, hidx, -⟩ := restrictShape P units S hs
  obtain ⟨⟨hmap, hkeys⟩, -⟩ := FlockSoundness.DecodeBoolean.decode_facts bytes P h hr
  have hm : P.BoolMapOk := ⟨hmap, hkeys⟩
  have hbodyS := bodyOf_eq hm hbool hidx
  obtain ⟨hargs, hret, hord, hout, hsome, hbat⟩ := hT bytes P h hr d hd hco hgb
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
  obtain ⟨heS, hLpre, hLbody, fnOf, primOf, gatesOf, hLe, hfn⟩ := hLC S L d e hcont hl (by omega) hcoS (by rw [hSd]; exact hne)
  rw [hSd] at hLe hfn
  have heL : e < L.defs.size := (Array.getElem?_eq_some_iff.1 hLe).1
  have hLP : ∀ i < P.defs.size, L.defs[i]? = P.defs[i]? := fun i hi => by
    rw [hLpre i (by omega), get?_of_pre hle hpre hi]
  have hLP' : ∀ i < P.defs.size, L.defs[i]! = P.defs[i]! := fun i hi => by
    rw [getElem!_def, getElem!_def, hLP i hi]
  have hLbP : ∀ k, L.bodyOf? k = P.bodyOf? k := fun k => by rw [hLbody, hbodyS]
  have hPd : P.defs[d]? = some (P.defs[d]!) := by rw [getElem!_pos P.defs d hd]; exact getElem?_pos P.defs d hd
  obtain ⟨hco1, hco2⟩ := hco
  generalize hD : P.defs[d]! = D at hPd hco1 hco2 hx hargs hret hord hout hsome hbat hLe hfn
  have hD'p : (Flock.Lift.liftDef D fnOf primOf gatesOf).prim = false := by rw [liftDef_prim]; exact hco1
  have hsz' : (Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes.toList.length = D.body.nodes.toList.length := by
    simp [liftDef_nodes_size]
  have hstep : ∀ n (hn : n < D.body.nodes.toList.length) (outs : Array (List (List Bool))), outs.size = n →
      OutsOk D outs → ∃ w,
        nodeStep wordWires P (wordDef? P (2 * P.defs.size + 1)) d x D outs D.body.nodes.toList[n] = some (outs.push w) ∧
        nodeStep Flock.Program.Wires.bool L (Flock.Program.evalDef? L (2 * L.defs.size + 1)) e x.flatten
            (Flock.Lift.liftDef D fnOf primOf gatesOf) (outs.map List.flatten)
            ((Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes.toList[n]'(by omega)) =
          some ((outs.push w).map List.flatten) ∧ OutsOk D (outs.push w) := by
    intro n hn outs hsz ho
    have hnD : n < D.body.nodes.size := by simpa using hn
    have hnD' : n < (Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes.size := by
      rw [liftDef_nodes_size]; exact hnD
    simp only [Array.getElem_toList]
    have hmem : D.body.nodes[n] ∈ D.body.nodes := Array.getElem_mem hnD
    obtain ⟨hform, hwp⟩ := hco2 _ hmem
    have hlt := hord _ hmem
    have hfnP : (D.body.nodes[n]).fn < P.defs.size := by omega
    obtain ⟨hfn', hform', hseq⟩ := liftDef_node D fnOf primOf gatesOf hnD hnD'
    have hbS := hfn _ hmem
    rw [hpre _ hfnP, hbodyS] at hbS
    have hbl : fnOf (D.body.nodes[n]).fn < P.defs.size := body_lt hm hbS
    have hprim : (P.defs[(D.body.nodes[n]).fn]!).prim = true := by
      simp only [isWordPrim, Bool.and_eq_true] at hwp
      exact hwp.1.1
    have hcal : P.defs[(D.body.nodes[n]).fn]? = some (P.defs[(D.body.nodes[n]).fn]!) := by
      rw [getElem!_pos P.defs _ hfnP]; exact getElem?_pos P.defs _ hfnP
    have hcalL : L.defs[fnOf (D.body.nodes[n]).fn]? = some (P.defs[fnOf (D.body.nodes[n]).fn]!) := by
      rw [hLP _ hbl, getElem!_pos P.defs _ hbl]; exact getElem?_pos P.defs _ hbl
    have hB := boolCall h hr (by omega) hLP' hLbP hfnP hwp hbS (hsome _ hmem _ hbS)
    by_cases hb : (D.body.nodes[n]).form = .batch
    · have hax : ∀ i < (D.body.nodes[n]).args.size,
          (D.body.nodes[n]).axes[i]? = some (some 0) ∨ ∀ ax, (D.body.nodes[n]).axes[i]? ≠ some (some ax) := by
        rcases hform with hf' | hf' | ⟨-, hax⟩
        · rw [hf'] at hb; cases hb
        · rw [hf'] at hb; cases hb
        · exact hax
      have hf := hb
      have hf2 : ((Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes[n]).form = .batch := by
        simp only [Flock.Lift.liftDef, Array.getElem_map, hf]; rfl
      have hn2 : ((Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes[n]).n = (D.body.nodes[n]).n := by
        simp [Flock.Lift.liftDef]
      have hW := nodeStep_batch wordWires (fun _ => rfl) P (wordDef? P (2 * P.defs.size + 1)) d x D outs
        (D.body.nodes[n]) hf hlt hcal
      have hBt := nodeStep_batch Flock.Program.Wires.bool (fun _ => rfl) L
        (Flock.Program.evalDef? L (2 * L.defs.size + 1)) e x.flatten (Flock.Lift.liftDef D fnOf primOf gatesOf)
        (outs.map List.flatten) ((Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes[n]) hf2
        (by rw [hfn']; omega) (by rw [hfn']; exact hcalL)
      rw [liftDef_params] at hBt
      obtain ⟨b', -, hbo, -, -, hpar, -⟩ := hmap _ hfnP hwp
      obtain rfl : b' = fnOf (D.body.nodes[n]).fn := Option.some.inj (hbo.symm.trans hbS)
      have hY := fun m (hm' : m < (D.body.nodes[n]).n) => batchBits hx ho hnD hnD' hsz hf hpar (hargs n hnD) hax
        (hbat n hnD hf) hm'
      refine ⟨(List.range (D.body.nodes[n]).n).map fun m => Prog.evalBool P (D.body.nodes[n]).fn ((List.range (D.body.nodes[n]).args.size).flatMap fun i =>
              batchArg (Flock.Program.refVal x (Flock.Program.paramStarts D.params) outs)
                (P.defs[(D.body.nodes[n]).fn]!) (D.body.nodes[n]) m i).flatten,
        ?_, ?_, ?_⟩
      · rw [hW]
        simp only [wordDef_prim hfnP hprim]
        rw [mapM_some_map]
        exact congrArg (fun z => some (outs.push z)) (flatten_map_singleton _ _)
      · rw [hBt, hfn']
        simp only [hB]
        rw [mapM_some_map, Array.map_push, hn2]
        exact congrArg (fun z => some ((outs.map List.flatten).push z))
          (congrArg List.flatten (List.map_congr_left fun m hm => by rw [hY m (List.mem_range.1 hm)]))
      · apply outsOk_push ho
        rw [hsz, show D.body.nodes.getD n default = D.body.nodes[n] from by simp [Array.getD, hnD]]
        rw [hout n hnD ((List.range (D.body.nodes[n]).n).map fun m => ((List.range (D.body.nodes[n]).args.size).flatMap fun i =>
              batchArg (Flock.Program.refVal x (Flock.Program.paramStarts D.params) outs)
                (P.defs[(D.body.nodes[n]).fn]!) (D.body.nodes[n]) m i).flatten)
          (by rw [List.length_map, List.length_range, members_batch hf])]
        simp only [List.map_map]
        try rfl
    have hform2 : (D.body.nodes[n]).form = .prim ∨ (D.body.nodes[n]).form = .call := by
      rcases hform with hf | hf | ⟨hf, -⟩
      · exact Or.inl hf
      · exact Or.inr hf
      · exact absurd hf hb
    have hout1 : ∀ y, (D.body.nodes[n]).out.widths.toList = [(Prog.evalBool P (D.body.nodes[n]).fn y).length] :=
      fun y => by rw [hout n hnD [y] (by rw [members_one hform2]; rfl)]; rfl
    have hargsB := argsBits hx ho hseq (fun a ha => by
      obtain ⟨h1, h2⟩ := hargs n hnD a ha
      exact ⟨h1, fun r hr' => by rw [hsz]; exact h2 r hr'⟩)
    rw [← liftDef_params D fnOf primOf gatesOf] at hargsB
    refine ⟨[Prog.evalBool P (D.body.nodes[n]).fn
      ((D.body.nodes[n]).args.toList.flatMap fun a =>
        Flock.Program.argVals (Flock.Program.refVal x (Flock.Program.paramStarts D.params) outs) a a.seq.count id).flatten],
      ?_, ?_, ?_⟩
    · rcases hform2 with hf | hf <;>
        simp [nodeStep, wordWires, hf, hcal, show ¬ d ≤ (D.body.nodes[n]).fn by omega, wordDef_prim hfnP hprim]
    · rcases hform' hform2 with hf | hf <;>
        simp [nodeStep, Flock.Program.Wires.bool, hf, hfn', hcalL, show ¬ e ≤ fnOf (D.body.nodes[n]).fn by omega,
          hargsB, hB, Array.map_push]
    · apply outsOk_push ho
      rw [hsz]
      simp only [List.map_cons, List.map_nil, Array.getD, hnD, ↓reduceDIte]
      exact (hout1 _).symm
  obtain ⟨o, g1, g2, g3, g4⟩ := fold_calls D D.body.nodes.toList
    (Flock.Lift.liftDef D fnOf primOf gatesOf).body.nodes.toList hsz' _ _ hstep D.body.nodes.toList.length 0 #[]
    (by omega) rfl (fun i hi => by simp at hi)
  simp only [List.drop_zero, Array.map_empty] at g1 g2
  unfold wordEval Prog.evalBool
  rw [show 2 * P.defs.size + 2 = (2 * P.defs.size + 1) + 1 from rfl,
    show 2 * L.defs.size + 2 = (2 * L.defs.size + 1) + 1 from rfl]
  unfold wordDef? Flock.Program.evalDef?
  simp only [hPd, Option.bind_eq_bind, Option.bind_some, hco1, Bool.false_eq_true, ↓reduceIte]
  rw [walkStep_comp wordWires (fun _ => rfl) P _ d x hPd hco1,
    walkStep_comp Flock.Program.Wires.bool (fun _ => rfl) L _ e x.flatten hLe hD'p, g1, g2]
  simp only [Option.bind_eq_bind, Option.bind_some, Option.pure_def, Option.getD_some]
  rw [liftDef_ret, liftDef_params]
  have hos : o.size = D.body.nodes.size := by simpa using g4
  exact seqBits hx g3 hret.1 (fun r hr' => by rw [hos]; exact hret.2 r hr')


theorem batch0_of_calls {P : Prog} {d : ℕ} (h : CallsOnly P d) : Batch0 P d :=
  ⟨h.1, fun nd hnd => ⟨(h.2 nd hnd).1.elim Or.inl (fun h' => Or.inr (Or.inl h')), (h.2 nd hnd).2⟩⟩

/-- **`LiftWordEval` for a template of word-primitive calls**, at least one node, on `LiftCalls` and `Batch0Typed`. -/
theorem liftWordEval_calls (hLC : LiftCalls) (hT : Batch0Typed) :
    ∀ (bytes : ByteArray) (P : Prog) (units : Array ℕ) (S : Prog) (d : ℕ) (L : Prog) (e : ℕ),
      decode bytes = .ok P → (P.defs[P.root]!).body.recut = #[] → Spot.Restrict.restrict P units = .ok S →
      d < P.defs.size → CallsOnly P d → GateBodies P d → 0 < (P.defs[d]!).body.nodes.size →
      Flock.Lift.lift S d = .ok (L, e) →
      ∀ x : List (List Bool), x.map List.length = leafWidths (P.defs[d]!) →
        Prog.evalBool L e x.flatten = (wordEval P d x).flatten :=
  fun bytes P units S d L e h hr hs hd hco =>
    liftWordEval_batch0 hLC hT bytes P units S d L e h hr hs hd (batch0_of_calls hco)

end Flock.Chain
