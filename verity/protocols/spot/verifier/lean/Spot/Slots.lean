/-
The claims' slots over the verifier's partition (`Claims.Sampling.Partition` and `Deploy.checked`): the partition the
verifier evaluates, as the claims read it, and the slots of the proof blocks a sample checks.

* **Classes.** Two, the population and then the inputs. The population's strata are the query's (`Partition.strata`:
  one per template or Definition, in the order of its first unit), each stratum's proof blocks its units in ascending
  order, each unit's gates (`Partition.unitGates`): an instance's, a `Q_call` unit's interval, or the span of the gates a
  `Q_word` unit owns, which another unit's gates may interleave. The input class is one stratum (none when the root has
  no input) of every gate of the root's input nodes, in gate order, each a proof block of its own; the claims' design
  reads it `all`, as `partition_object`'s input unit is never sampled. No run opens an input row on its own: `Run.run`
  proves the sampled proof blocks (`Draw.proved`), whose session (`HmRow.drawn`) opens only the rows its units read, and
  registration opens none. So an input row is opened only through a sampled proof block that reads it.
* **Slots.** A proof block's slot is its index in class, stratum and block order (`Partition.blocks`). Where no two
  strata's units interleave, unit `i`'s slot is `i`.
* **Checked.** `checked` lists the slot of every proof block the sample proves (`Select.select`'s proved units: its units
  and a work sample's closure) and of every input slot, in slot order. It is the claims' reading of a sample; `Run.run`
  doesn't call it, and the session proves no input slot of a unit outside the sample.
* **JSON.** The layout's JSON keeps a block's keys `unit` and `pus` (its gates as a list of one `[lo, hi]`), and
  `checked` its `[slot, 0]` pairs, until a format version: the difftest's Python side reads them.
-/
module

public import Spot.Select

@[expose] public section

namespace Spot.Slots

open Lean (Json)
open Flock (V)
open Flock.TemplateQuery (Desc)

/-- A proof block: the population's unit it is (`none` for an input gate), and its gates, `[lo, hi)`. -/
structure ProofBlock where
  unit : Option Nat
  gates : Nat × Nat
  deriving BEq, Inhabited, Repr

/-- The claims' partition over the verifier's: its classes, each its strata, each its proof blocks, in slot order. -/
structure Layout where
  classes : Array (Array (Array ProofBlock))
  deriving BEq, Inhabited, Repr

/-- The proof blocks, slot by slot. -/
def Layout.blocks (l : Layout) : Array ProofBlock := l.classes.flatten.flatten

/-- Unit `u`'s slot: the proof block that is it. -/
def Layout.slotOf (l : Layout) (u : Nat) : Option Nat := l.blocks.findIdx? (·.unit == some u)

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

/-- The claims' partition over the verifier's own program `p` under `query` (`Partition.evaluate`): the population's
strata, then the input class. An error where the query doesn't apply. -/
def layout (p : Program) (query : Json) : V Layout := do
  let rs ← Partition.evaluate p query
  let gs ← Partition.unitGates p query
  let strata := Flock.Draw.strataOf (rs.map fun r => (r.template, r.first, r.count))
  let pop ← strata.mapM fun (_, ranges) =>
    (ranges.flatMap fun (lo, hi) => (Array.range (hi - lo)).map (lo + ·)).mapM fun u => do
      let some g := gs[u]? | throw s!"unit {u} has no gates"
      pure ({ unit := some u, gates := g } : ProofBlock)
  let inputs := (inputGates p.desc).map fun g => ({ unit := none, gates := (g, g + 1) } : ProofBlock)
  return { classes := #[pop, if inputs.isEmpty then #[] else #[inputs]] }

/-- A layout's population: its first class's proof blocks, in slot order. -/
def popOf (l : Layout) : List ProofBlock := ((l.classes.toList.headD #[]).toList.map Array.toList).flatten

/-- **The population is the registration's `n` units, in order**: the layout of `p` under `query` is defined, and its
population is `n` proof blocks with slot `u` unit `u`. The run refuses a registration whose program and query fail it
(`Run.run`, `population-order`): the sampled audit's knowledge soundness reads slot `u` as the game's unit `u`. -/
def popOk (p : Program) (query : Json) (n : Nat) : Bool :=
  match layout p query with
  | .error _ => false
  | .ok l => decide ((popOf l).map (·.unit) = (List.range n).map some)

/-- A stratum's units in slot order: its ranges `[lo, hi)`, each in ascending order, as `layout` lays them. -/
def stratumUnits (s : String × Array (Nat × Nat)) : Array Nat :=
  s.2.flatMap fun r => (Array.range (r.2 - r.1)).map (r.1 + ·)

/-- The population's units in slot order: its strata's (`stratumUnits`), stratum by stratum. -/
def popUnits (strata : Array (String × Array (Nat × Nat))) : List Nat :=
  strata.toList.flatMap fun s => (stratumUnits s).toList

/-- `popOk` without laying out a proof block: the query's strata read in slot order are units `0, …, n − 1`, each with
its gates (`Partition.unitGates`). `layout` holds a proof block per input gate of the root as well, which the
population's order doesn't read; the compiler runs this for `popOk` (`popOk_eq_popOkRanges`). -/
def popOkRanges (p : Program) (query : Json) (n : Nat) : Bool :=
  match Partition.evaluate p query, Partition.unitGates p query with
  | .ok rs, .ok gs =>
    let us := popUnits (Flock.Draw.strataOf (rs.map fun r => (r.template, r.first, r.count)))
    decide (us = List.range n) && us.all (· < gs.size)
  | _, _ => false

private theorem mapM_ok_map {ε α β γ : Type} (f : α → Except ε β) (g : β → γ) (k : α → γ)
    (h : ∀ a b, f a = .ok b → g b = k a) : ∀ (l : List α) (bs : List β), l.mapM f = .ok bs → bs.map g = l.map k
  | [], bs, hm => by
    simp only [List.mapM_nil] at hm
    cases hm; rfl
  | a :: l, bs, hm => by
    simp only [List.mapM_cons] at hm
    cases hfa : f a with
    | error e => rw [hfa] at hm; cases hm
    | ok b =>
      rw [hfa] at hm
      cases hl : l.mapM f with
      | error e => rw [hl] at hm; cases hm
      | ok bs' =>
        rw [hl] at hm
        cases hm
        simp [h a b hfa, mapM_ok_map f g k h l bs' hl]

private theorem mapM_ok_iff {ε α β : Type} (f : α → Except ε β) :
    ∀ (l : List α), (∃ bs, l.mapM f = .ok bs) ↔ ∀ a ∈ l, ∃ b, f a = .ok b
  | [] => by simp [List.mapM_nil]; exact ⟨_, rfl⟩
  | a :: l => by
    simp only [List.mapM_cons, List.mem_cons, forall_eq_or_imp]
    rw [← mapM_ok_iff f l]
    cases hfa : f a with
    | error e => simp [bind, Except.bind]
    | ok b =>
      cases hl : l.mapM f with
      | error e => simp [bind, Except.bind]
      | ok bs => simp [bind, Except.bind, pure, Except.pure]

private theorem arr_mapM_ok_map {ε α β γ : Type} (f : α → Except ε β) (g : β → γ) (k : α → γ)
    (h : ∀ a b, f a = .ok b → g b = k a) (xs : Array α) (bs : Array β) (hm : xs.mapM f = .ok bs) :
    bs.toList.map g = xs.toList.map k := by
  rw [Array.mapM_eq_mapM_toList] at hm
  cases hl : xs.toList.mapM f with
  | error e => rw [hl] at hm; cases hm
  | ok l =>
    rw [hl] at hm
    cases hm
    exact mapM_ok_map f g k h _ _ hl

private theorem arr_mapM_ok_iff {ε α β : Type} (f : α → Except ε β) (xs : Array α) :
    (∃ bs, xs.mapM f = .ok bs) ↔ ∀ a ∈ xs.toList, ∃ b, f a = .ok b := by
  rw [Array.mapM_eq_mapM_toList, ← mapM_ok_iff]
  cases hl : xs.toList.mapM f with
  | error e => simp [Functor.map, Except.map]
  | ok l => simp [Functor.map, Except.map]

private theorem strata_mapM_ok_iff {σ β : Type} (strata : Array σ) (A : σ → Array Nat) (G : σ → Except String β)
    (size : Nat) (hG' : ∀ s, (∃ bs, G s = .ok bs) ↔ ∀ u ∈ (A s).toList, u < size) :
    (∃ pop, strata.mapM G = .ok pop) ↔ (strata.toList.flatMap fun s => (A s).toList).all (· < size) = true := by
  rw [arr_mapM_ok_iff]
  simp only [List.all_eq_true, List.mem_flatMap, decide_eq_true_eq]
  constructor
  · rintro h u ⟨s, hs, hu⟩
    exact (hG' s).1 (h s hs) u hu
  · intro h s hs
    exact (hG' s).2 fun u hu => h u ⟨s, hs, hu⟩

private theorem popOk_units_ok {σ : Type} {strata : Array σ} {A : σ → Array Nat}
    {G : σ → Except String (Array ProofBlock)} {pop : Array (Array ProofBlock)} {size : Nat} (n : Nat)
    (ins : Array (Array ProofBlock)) (hm : strata.mapM G = .ok pop)
    (hG : ∀ s bs, G s = .ok bs → bs.toList.map (·.unit) = (A s).toList.map some)
    (hG' : ∀ s, (∃ bs, G s = .ok bs) ↔ ∀ u ∈ (A s).toList, u < size) :
    decide ((popOf { classes := #[pop, ins] }).map (·.unit) = (List.range n).map some) =
      (decide ((strata.toList.flatMap fun s => (A s).toList) = List.range n) &&
        (strata.toList.flatMap fun s => (A s).toList).all (· < size)) := by
  rw [(strata_mapM_ok_iff strata A G size hG').1 ⟨pop, hm⟩, Bool.and_true]
  have := arr_mapM_ok_map G (fun bs => bs.toList.map (·.unit)) (fun s => (A s).toList.map some) hG _ _ hm
  have hu : (popOf { classes := #[pop, ins] }).map (·.unit) =
      (strata.toList.flatMap fun s => (A s).toList).map some := by
    simp only [popOf, List.map_flatten, List.map_map]
    change (pop.toList.map fun bs => bs.toList.map (·.unit)).flatten = _
    rw [this, List.flatMap_def, List.map_flatten, List.map_map]
    rfl
  rw [hu]
  exact decide_eq_decide.2 (List.map_inj_right fun _ _ h => Option.some.inj h)

private theorem popOk_units_error {σ β : Type} {strata : Array σ} {A : σ → Array Nat} {G : σ → Except String β}
    {e : String} {size : Nat} (n : Nat) (hm : strata.mapM G = .error e)
    (hG' : ∀ s, (∃ bs, G s = .ok bs) ↔ ∀ u ∈ (A s).toList, u < size) :
    false = (decide ((strata.toList.flatMap fun s => (A s).toList) = List.range n) &&
      (strata.toList.flatMap fun s => (A s).toList).all (· < size)) := by
  cases hb : (strata.toList.flatMap fun s => (A s).toList).all (· < size)
  · simp
  · obtain ⟨pop, hp⟩ := (strata_mapM_ok_iff strata A G size hG').2 hb
    rw [hm] at hp; cases hp

/-- **`popOk` is `popOkRanges`**: a unit of the strata has a proof block exactly where it has gates, and the population's
proof blocks are the strata's units in slot order, whatever the input class holds. -/
@[csimp] theorem popOk_eq_popOkRanges : @popOk = @popOkRanges := by
  funext p query n
  unfold popOk popOkRanges popUnits layout
  cases he : Partition.evaluate p query with
  | error e => rfl
  | ok rs =>
    cases hg : Partition.unitGates p query with
    | error e => rfl
    | ok gs =>
      simp only [bind, Except.bind, pure, Except.pure]
      have hF : ∀ u, (∃ b, (match gs[u]? with
          | some g => Except.ok ({ unit := some u, gates := g } : ProofBlock)
          | _ => (throw (toString "unit " ++ toString u ++ toString " has no gates") : Except String ProofBlock)) = .ok b) ↔
          u < gs.size := by
        intro u
        constructor
        · rintro ⟨b, h⟩
          split at h
          · rename_i g hu; exact (Array.getElem?_eq_some_iff.1 hu).1
          · cases h
        · intro hu
          simp only [Array.getElem?_eq_getElem hu]
          exact ⟨_, rfl⟩
      split
      next e heq =>
        split at heq
        next err hm =>
          refine popOk_units_error (A := stratumUnits) n hm fun s => ?_
          rw [arr_mapM_ok_iff]
          exact forall_congr' fun u => forall_congr' fun _ => hF u
        next => cases heq
      next v heq =>
        split at heq
        next => cases heq
        next pop hm =>
          cases heq
          refine popOk_units_ok (A := stratumUnits) n _ hm ?_ fun s => ?_
          rotate_left
          · rw [arr_mapM_ok_iff]
            exact forall_congr' fun u => forall_congr' fun _ => hF u
          intro s bs h
          refine arr_mapM_ok_map _ (·.unit) some ?_ _ _ h
          intro u b h
          split at h
          · cases h; rfl
          · cases h

/-- **The checked slots** (`Deploy.checked`) of a sample that proves the units `proved`: every proof block among them and
every input slot, in slot order. An input slot listed here is opened only where a sampled proof block reads its row. -/
def checked (l : Layout) (proved : Array Nat) : Array Nat :=
  (l.blocks.zipIdx.filter fun (b, _) => b.unit.all proved.contains).map (·.2)

/-- **The challenger's sample in the claims' slots**: the layout of `p` under `query`, and the slots of the units
`Select.select`'s sample on the coins `src` proves. An error where either is. -/
def select (rule : Select.Rule) (p : Program) (query : Json) (table : Option Json) (cm : Option (Array (Int × Array Int)))
    (src : ByteArray) : V (Layout × Array Nat) := do
  let l ← layout p query
  let (_, proved, _) ← Select.select rule p query table cm src
  return (l, checked l proved)

def ProofBlock.toJson (b : ProofBlock) : Json :=
  Json.mkObj [("unit", (b.unit.map Lean.toJson).getD .null),
    ("pus", Json.arr #[Json.arr #[Lean.toJson b.gates.1, Lean.toJson b.gates.2]])]

def Layout.toJson (l : Layout) : Json := Json.arr (l.classes.map fun c => Json.arr (c.map fun s => Json.arr (s.map (·.toJson))))

end Spot.Slots
