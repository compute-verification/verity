/-
The claims' slots over the verifier's partition (`Claims.Sampling.Partition` and `Deploy.checked`): the partition the
verifier evaluates, as the claims read it, and the proof units a draw checks, each as its replay unit's slot and its index
there.

* **Classes.** Two, the population and then the inputs. The population's strata are the query's (`Partition.strata`:
  one per template or Definition, in the order of its first unit), each stratum's replay units its units in ascending
  order, each unit one proof unit, its gates (`Partition.unitGates`): an instance's, a `Q_call` unit's interval, or the
  span of the gates a `Q_word` unit owns, which another unit's gates may interleave. The input class is one stratum (none
  when the root has no input) of every gate of the root's input nodes, in gate order, each a replay unit of one proof
  unit; it is drawn `all`, as `partition_object`'s input unit is always checked and never drawn.
* **Slots.** A replay unit's slot is its index in class, stratum and replay-unit order (`Partition.rus`). Where no two
  strata's units interleave, unit `i`'s slot is `i`.
* **Checked.** `checked` lists the proof units of every replay unit the draw proves (`Select.select`'s proved units: its
  units and a work draw's closure) and of every input slot, each as `(slot, index)`, in slot order. The rules a stream
  runs make every replay unit one proof unit, so each index is `0`.
* **Population order.** `popOk`, which the run checks, is stated on the layout's population; it runs as `popOkRanges`
  (`popOk_eq_popOkRanges`), which reads the strata's segments and the units' gates and builds no replay unit and no input
  gate.
-/
import SampledProofs.Select

namespace SampledProofs.Slots

open Lean (Json)
open Flock (V)
open Flock.TemplateQuery (Desc)

/-- A replay unit: the population's unit it is (`none` for an input gate), and its proof units' gates, each `[lo, hi)`. -/
structure RU where
  unit : Option Nat
  pus : Array (Nat × Nat)
  deriving BEq, Inhabited, Repr

/-- The claims' partition over the verifier's: its classes, each its strata, each its replay units, in slot order. -/
structure Layout where
  classes : Array (Array (Array RU))
  deriving BEq, Inhabited, Repr

/-- The replay units, slot by slot. -/
def Layout.rus (l : Layout) : Array RU := l.classes.flatten.flatten

/-- Unit `u`'s slot: the replay unit that is it. -/
def Layout.slotOf (l : Layout) (u : Nat) : Option Nat := l.rus.findIdx? (·.unit == some u)

/-- The gates of the root's input nodes (`TemplateQuery.isInput`), in gate order. -/
def inputGates (d : Desc) : Array Nat := Id.run do
  let some root := d.def? d.root | return #[]
  let mut out := #[]
  let mut at_ := 0
  for x in root.nodes do
    let tid := d.callees.getD x.fn ""
    let n := x.members * d.gates tid
    if Flock.TemplateQuery.isInput tid then out := out ++ (Array.range n).map (at_ + ·)
    at_ := at_ + n
  return out

/-- `inputGates` with each root callee's gates read off `Partition.gTable`, computed once. -/
def inputGatesFast (d : Desc) : Array Nat := Id.run do
  let some root := d.def? d.root | return #[]
  let G := Partition.gTable d
  let mut out := #[]
  let mut at_ := 0
  for x in root.nodes do
    let tid := d.callees.getD x.fn ""
    let n := x.members * Partition.look d G tid
    if Flock.TemplateQuery.isInput tid then out := out ++ (Array.range n).map (at_ + ·)
    at_ := at_ + n
  return out

@[csimp] theorem inputGates_eq_inputGatesFast : @inputGates = @inputGatesFast := by
  funext d
  simp only [inputGates, inputGatesFast, Partition.look_gTable]

/-- The claims' partition over the verifier's own program `p` under `query` (`Partition.evaluate`): the population's
strata, then the input class. An error where the query doesn't apply. -/
def layout (p : Program) (query : Json) : V Layout := do
  let rs ← Partition.evaluate p query
  let gs ← Partition.unitGates p query
  let strata := Flock.Draw.strataOf (rs.map fun r => (r.template, r.first, r.count))
  let pop ← strata.mapM fun (_, ranges) =>
    (ranges.flatMap fun (lo, hi) => (Array.range (hi - lo)).map (lo + ·)).mapM fun u => do
      let some g := gs[u]? | throw s!"unit {u} has no gates"
      pure ({ unit := some u, pus := #[g] } : RU)
  let inputs := (inputGates p.desc).map fun g => ({ unit := none, pus := #[(g, g + 1)] } : RU)
  return { classes := #[pop, if inputs.isEmpty then #[] else #[inputs]] }

/-- A layout's population: its first class's replay units, in slot order. -/
def popOf (l : Layout) : List RU := ((l.classes.toList.headD #[]).toList.map Array.toList).flatten

/-- **The population is the registration's `n` units, in order**: the layout of `p` under `query` is defined, and its
population is `n` replay units with slot `u` unit `u`. The run refuses a registration whose program and query fail it
(`Run.run`, `population-order`): the sampled audit's knowledge soundness reads slot `u` as the game's unit `u`. -/
def popOk (p : Program) (query : Json) (n : Nat) : Bool :=
  match layout p query with
  | .error _ => false
  | .ok l => decide ((popOf l).map (·.unit) = (List.range n).map some)

/-- Segments `[lo, hi)`, read in order, are `a, a + 1, …, b - 1`: each nonempty one starts where the last ended, and the
last ends at `b`. -/
def chain : List (Nat × Nat) → Nat → Nat → Bool
  | [], a, b => a == b
  | (lo, hi) :: ss, a, b => if hi ≤ lo then chain ss a b else lo == a && chain ss hi b

/-- The units of segments `[lo, hi)`, in order. -/
def segUnits (ss : List (Nat × Nat)) : List Nat := ss.flatMap fun s => List.range' s.1 (s.2 - s.1)

/-- The population's segments, in slot order: each stratum's ranges (`Draw.strataOf`), stratum by stratum. -/
def segsOf (rs : Array Flock.TemplateQuery.TRange) : List (Nat × Nat) :=
  (Flock.Draw.strataOf (rs.map fun r => (r.template, r.first, r.count))).toList.flatMap (·.2.toList)

/-- `popOk` from the ranges, with no replay unit and no input gate built: the query's ranges and unit gates are
defined, its strata's segments chain `0, 1, …, n - 1`, and every one of those units has its gates. -/
def popOkRanges (p : Program) (query : Json) (n : Nat) : Bool :=
  match Partition.evaluate p query with
  | .error _ => false
  | .ok rs =>
    match Partition.unitGates p query with
    | .error _ => false
    | .ok gs => chain (segsOf rs) 0 n && decide (n ≤ gs.size)

theorem chain_le : ∀ (ss : List (Nat × Nat)) (a b : Nat), chain ss a b = true → a ≤ b := by
  intro ss
  induction ss with
  | nil => intro a b h; simp [chain] at h; omega
  | cons s ss ih =>
    intro a b h
    obtain ⟨lo, hi⟩ := s
    simp only [chain] at h
    split at h
    · exact ih a b h
    · simp only [Bool.and_eq_true, beq_iff_eq] at h
      have := ih hi b h.2
      omega

theorem chain_iff : ∀ (ss : List (Nat × Nat)) (a m : Nat),
    chain ss a (a + m) = true ↔ segUnits ss = List.range' a m := by
  intro ss
  induction ss with
  | nil =>
    intro a m
    simp only [chain, segUnits, List.flatMap_nil, beq_iff_eq]
    constructor
    · intro h; have : m = 0 := by omega
      subst this; rfl
    · intro h
      have := congrArg List.length h
      simp at this
      omega
  | cons s ss ih =>
    intro a m
    obtain ⟨lo, hi⟩ := s
    simp only [chain, segUnits, List.flatMap_cons] at ih ⊢
    split
    · rename_i hle
      have : hi - lo = 0 := by omega
      rw [this, List.range'_zero, List.nil_append]
      exact ih a m
    · rename_i hlt
      simp only [Bool.and_eq_true, beq_iff_eq]
      constructor
      · rintro ⟨hlo, hc⟩
        subst hlo
        have hle := chain_le ss hi (lo + m) hc
        have e : lo + m = hi + (lo + m - hi) := by omega
        rw [e] at hc
        rw [(ih hi (lo + m - hi)).1 hc]
        have := @List.range'_append lo (hi - lo) (lo + m - hi) 1
        simp only [Nat.one_mul] at this
        rw [show lo + (hi - lo) = hi by omega, show hi - lo + (lo + m - hi) = m by omega] at this
        exact this
      · intro h
        have hk : 0 < hi - lo := by omega
        have hlen := congrArg List.length h
        simp only [List.length_append, List.length_range'] at hlen
        have hm : hi - lo ≤ m := by omega
        have split := @List.range'_append a (hi - lo) (m - (hi - lo)) 1
        simp only [Nat.one_mul] at split
        rw [show hi - lo + (m - (hi - lo)) = m by omega] at split
        rw [← split] at h
        have hlo : lo = a := by
          obtain ⟨k, hk'⟩ : ∃ k, hi - lo = k + 1 := ⟨hi - lo - 1, by omega⟩
          rw [hk', List.range'_succ, List.range'_succ] at h
          simp only [List.cons_append, List.cons.injEq] at h
          exact h.1
        subst hlo
        refine ⟨rfl, ?_⟩
        have hrest := List.append_cancel_left h
        rw [show lo + (hi - lo) = hi by omega] at hrest
        rw [show lo + m = hi + (m - (hi - lo)) by omega]
        exact (ih hi (m - (hi - lo))).2 hrest

theorem mapM_ok {α β ε : Type} (f : α → Except ε β) (v : α → β) :
    ∀ (l : List α), (∀ x ∈ l, f x = .ok (v x)) → l.mapM f = .ok (l.map v) := by
  intro l
  induction l with
  | nil => intro _; rfl
  | cons a l ih =>
    intro h
    rw [List.mapM_cons, h a (by simp), ih (fun x hx => h x (by simp [hx]))]
    rfl

theorem mapM_err {α β ε : Type} (f : α → Except ε β) :
    ∀ (l : List α), (∃ x ∈ l, ∃ e, f x = .error e) → ∃ e, l.mapM f = .error e := by
  intro l
  induction l with
  | nil => intro ⟨x, hx, _⟩; simp at hx
  | cons a l ih =>
    intro ⟨x, hx, e, he⟩
    rw [List.mapM_cons]
    cases hfa : f a with
    | error e' => exact ⟨e', rfl⟩
    | ok b =>
      have hx' : x ∈ l := by
        rcases List.mem_cons.1 hx with rfl | h
        · rw [hfa] at he; cases he
        · exact h
      obtain ⟨e'', he''⟩ := ih ⟨x, hx', e, he⟩
      refine ⟨e'', ?_⟩
      show (do let bs ← List.mapM f l; pure (b :: bs)) = Except.error e''
      rw [he'']; rfl

theorem flat_toList (ranges : Array (Nat × Nat)) :
    (Array.flatMap (fun x => Array.map (fun x_1 => x.fst + x_1) (Array.range (x.snd - x.fst))) ranges).toList =
      segUnits ranges.toList := by
  simp only [Array.toList_flatMap, Array.toList_map, Array.toList_range, segUnits, List.range'_eq_map_range]

theorem strata_ok (G : Nat → V RU) (gv : Nat → RU) (S : Array (String × Array (Nat × Nat)))
    (hG : ∀ x ∈ S.toList, ∀ u ∈ segUnits x.2.toList, G u = .ok (gv u)) :
    S.mapM (fun x => Array.mapM G
      (Array.flatMap (fun x => Array.map (fun x_1 => x.fst + x_1) (Array.range (x.snd - x.fst))) x.snd)) =
      .ok (S.map fun x => ((segUnits x.2.toList).map gv).toArray) := by
  rw [Array.mapM_eq_mapM_toList, mapM_ok _ (fun x => ((segUnits x.2.toList).map gv).toArray) S.toList
    (fun x hx => by rw [Array.mapM_eq_mapM_toList, flat_toList, mapM_ok G gv _ (hG x hx)]; rfl)]
  simp only [Functor.map, Except.map]
  congr 1
  apply Array.ext'
  simp

theorem strata_err (G : Nat → V RU) (S : Array (String × Array (Nat × Nat)))
    (hG : ∃ x ∈ S.toList, ∃ u ∈ segUnits x.2.toList, ∃ e, G u = .error e) :
    ∃ e, S.mapM (fun x => Array.mapM G
      (Array.flatMap (fun x => Array.map (fun x_1 => x.fst + x_1) (Array.range (x.snd - x.fst))) x.snd)) =
      .error e := by
  obtain ⟨x, hx, u, hu, e, he⟩ := hG
  rw [Array.mapM_eq_mapM_toList]
  have : ∃ e, (fun x => Array.mapM G
      (Array.flatMap (fun x => Array.map (fun x_1 => x.fst + x_1) (Array.range (x.snd - x.fst))) x.snd)) x =
      Except.error e := by
    obtain ⟨e', he'⟩ := mapM_err G (segUnits x.2.toList) ⟨u, hu, e, he⟩
    refine ⟨e', ?_⟩
    simp only
    rw [Array.mapM_eq_mapM_toList, flat_toList, he']
    rfl
  obtain ⟨e', he'⟩ := this
  obtain ⟨e'', he''⟩ := mapM_err (fun x => Array.mapM G
      (Array.flatMap (fun x => Array.map (fun x_1 => x.fst + x_1) (Array.range (x.snd - x.fst))) x.snd))
    S.toList ⟨x, hx, e', he'⟩
  exact ⟨e'', by rw [he'']; rfl⟩

theorem popOf_mk (v w : Array (Array RU)) : popOf { classes := #[v, w] } = (v.toList.map Array.toList).flatten := rfl

theorem popOk_of (p : Program) (query : Json) (n : Nat) (rs : Array Flock.TemplateQuery.TRange)
    (gs : Array (Nat × Nat)) (h1 : Partition.evaluate p query = .ok rs) (h2 : Partition.unitGates p query = .ok gs) :
    popOk p query n = (chain (segsOf rs) 0 n && decide (n ≤ gs.size)) := by
  let g : Nat → V RU := fun u => do
    let some g := gs[u]? | throw s!"unit {u} has no gates"
    pure ({ unit := some u, pus := #[g] } : RU)
  let gv : Nat → RU := fun u => { unit := some u, pus := #[gs[u]!] }
  let S := Flock.Draw.strataOf (rs.map fun r => (r.template, r.first, r.count))
  let U := segUnits (segsOf rs)
  have hU : U = S.toList.flatMap (fun x => segUnits x.2.toList) := by
    simp only [U, segsOf, segUnits, List.flatMap_assoc, S]
  have hc : chain (segsOf rs) 0 n = decide (U = List.range n) := by
    have := chain_iff (segsOf rs) 0 n
    rw [Nat.zero_add, ← List.range_eq_range'] at this
    by_cases h : U = List.range n
    · simp only [U] at h; simp [this.2 h, U, h]
    · have : chain (segsOf rs) 0 n = false := by
        cases hh : chain (segsOf rs) 0 n
        · rfl
        · exact absurd (this.1 hh) h
      rw [this]; simp [U, h]
  rw [hc]
  unfold popOk layout
  rw [h1, h2]
  simp only [bind, Except.bind]
  by_cases hall : ∀ u ∈ U, u < gs.size
  · have hG : ∀ x ∈ S.toList, ∀ u ∈ segUnits x.2.toList, g u = .ok (gv u) := by
      intro x hx u hu
      have hlt : u < gs.size := hall u (by rw [hU]; exact List.mem_flatMap.2 ⟨x, hx, hu⟩)
      simp only [g, gv, Array.getElem?_eq_getElem hlt, getElem!_pos gs u hlt]
      rfl
    rw [strata_ok g gv S hG]
    simp only [pure, Except.pure]
    rw [Bool.eq_iff_iff]
    simp only [decide_eq_true_eq, Bool.and_eq_true]
    rw [popOf_mk]
    have hm : ((Array.map (fun x => (List.map gv (segUnits x.snd.toList)).toArray) S).toList.map Array.toList).flatten =
        U.map gv := by
      rw [hU, List.map_flatMap, List.flatten_eq_flatMap]
      simp [List.flatMap_map]
    rw [hm, List.map_map]
    have hu : (fun x : RU => x.unit) ∘ gv = some := rfl
    rw [hu, List.map_inj_right (fun _ _ h => Option.some.inj h)]
    constructor
    · intro h
      refine ⟨h, ?_⟩
      cases n with
      | zero => exact Nat.zero_le _
      | succ k => exact hall k (by rw [h]; simp)
    · exact fun h => h.1
  · obtain ⟨u, hu, hge⟩ : ∃ u ∈ U, gs.size ≤ u := by
      simp only [Classical.not_forall, Nat.not_lt] at hall
      obtain ⟨u, hu, h⟩ := hall
      exact ⟨u, hu, h⟩
    have hu' := hu
    rw [hU] at hu'
    obtain ⟨x, hx, hux⟩ := List.mem_flatMap.1 hu'
    have hgu : g u = .error s!"unit {u} has no gates" := by
      simp only [g, Array.getElem?_eq_none hge]
      rfl
    obtain ⟨e, he⟩ := strata_err g S ⟨x, hx, u, hux, _, hgu⟩
    rw [he]
    symm
    simp only [Bool.and_eq_false_iff, decide_eq_false_iff_not]
    by_cases h : U = List.range n
    · right
      have : u < n := by rw [h] at hu; simpa using hu
      omega
    · left; exact h

/-- `popOk` runs as `popOkRanges`: the layout's population is defined and its slot `u` is unit `u` exactly when the
query's strata chain `0, …, n - 1` and every such unit has gates. The input class, which `popOk` never reads, is never
built. -/
@[csimp] theorem popOk_eq_popOkRanges : @popOk = @popOkRanges := by
  funext p query n
  unfold popOkRanges
  cases h1 : Partition.evaluate p query with
  | error e => unfold popOk layout; rw [h1]; rfl
  | ok rs =>
    cases h2 : Partition.unitGates p query with
    | error e => unfold popOk layout; rw [h1, h2]; rfl
    | ok gs => exact popOk_of p query n rs gs h1 h2

/-- **The checked proof units** (`Deploy.checked`) of a draw that proves the units `proved`: each proof unit of every
replay unit among them and of every input slot, as `(slot, index)`, in slot order. -/
def checked (l : Layout) (proved : Array Nat) : Array (Nat × Nat) :=
  (l.rus.zipIdx.filter fun (r, _) => r.unit.all proved.contains).flatMap fun (r, v) =>
    (Array.range r.pus.size).map (v, ·)

/-- **The challenger's draw in the claims' slots**: the layout of `p` under `query`, and the proof units of the units
`Select.select`'s draw on the coins `src` proves. An error where either is. -/
def select (rule : Select.Rule) (p : Program) (query : Json) (table : Option Json) (cm : Option (Array (Int × Array Int)))
    (src : ByteArray) : V (Layout × Array (Nat × Nat)) := do
  let l ← layout p query
  let (_, proved, _) ← Select.select rule p query table cm src
  return (l, checked l proved)

def RU.toJson (r : RU) : Json :=
  Json.mkObj [("unit", (r.unit.map Lean.toJson).getD .null),
    ("pus", Json.arr (r.pus.map fun (lo, hi) => Json.arr #[Lean.toJson lo, Lean.toJson hi]))]

def Layout.toJson (l : Layout) : Json := Json.arr (l.classes.map fun c => Json.arr (c.map fun s => Json.arr (s.map (·.toJson))))

end SampledProofs.Slots
