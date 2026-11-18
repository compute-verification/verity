/-
A window circuit's wires, checked at the size of their description (`verity.commitments.periods.interface.Wire`).

A window's circuit is a population of template instances (`TemplatePopulations_v1`), whose inputs are all Input gates: the
program holds no dataflow between units. The wires are that dataflow. A wire states, once for many units, that reader
template `reader`'s units read rows of their parameter `port` whole as the words of row `out` of a unit of template
`writer` from its word `offset + Σ_k d_k · ostep_k` (`Wire.offsetAt`):

* **The reader rows a wire covers.** A reader port is cut into `k` rows (`cuts`, default 1): row `j` is the run of the
  parameter's outer elements `[j·n/k, (j+1)·n/k)`, `k` dividing the outer length `n` and the port's words, each its own
  hm96 row. A wire covers rows `cut0 … cut0 + cuts − 1` of units `at …` of the reader: reader index `i < units` (the
  product of `radix`) is unit `at + i / cuts`, row `cut0 + i % cuts`, and when `cuts > 1` the last digit of `radix` is
  `cuts`, so the row is that digit. With no cut the wire is a whole-port wire.
* **Its writers.** Reader index `i`'s writer is unit `base + Σ_k d_k · step_k` of `writer`, `d` the digits of `i` in mixed
  radix `radix` (the first digit the slowest); a step of 0 is a fan-out. Each index counts its own template's units.

`check` decides a circuit's wires in work proportional to its templates and wires, never to the units they cover (AGENTS.md
"Design rules"): each wire's writers lie in `span`, the least and greatest writer in closed
form; and two wires wire no reader row twice exactly when they read different ports or their rectangles (reader units ×
rows) are disjoint (`pairwiseDisjoint`, every pair; compiled as `pairwiseDisjointFast`, the wires sorted by reader, port
and first unit and each compared only with the earlier wires of its port whose units reach it). Its summary gives each reader row's writer (`Summary.writerOf`, by the
wires alone) and the wired rows and bits multiplied through (`wiredRows`, `wiredBits`), the wired term of the count of a
window's free inputs.

`checkRef` is the reference it is held to, every covered reader row expanded (`Wire.entries`): the difftest (the `wires`
and `wires_unrolled` pieces) holds both to the same Python reference. The verifier never runs it.

The refusals, in the order checked (`Code`): `cuts` (a cut names no template or port, or `k` is 0 or doesn't divide the
port's outer length and words), then wire by wire `template` (its reader or writer is no template), `port` (its port is no
input of the reader, or `out` no output of the writer), `cut` (its rows are not among the port's, or the last digit isn't
`cuts`, or `cuts` doesn't divide the wire's reader rows), `width` (a reader row's words from some index's offset run
past the writer row's, or an offset is negative), `bounds` (a reader unit or a writer outside its
template), then `overlap` (some reader row wired twice), then the order (the section of that name): `order` (the circuit's
order names no template, or its ranges of a template don't tile the template's units; or, naming it, the first wire the
order doesn't put after its writers) or, with no order stated, `cycle` (the templates' wires have a cycle; naming a wire
on it).
-/
module

public import Lean.Data.Json
public import Circuits.Wire
public import Std.Data.HashMap

@[expose] public section

namespace Spot.Wires

open Lean (Json JsonNumber toJson)
open Circuits

export Circuits (Wire)
export Circuits.Wire (prodR accum lowOf highOf boxLow boxHigh)
export Circuits.Json (int? nat? keysOf fields field natAt natOf intOf strOf arrOf sameKeys)
export Circuits (WIRE_KEYS)

namespace Wire
export Circuits.Wire (ofJson)
end Wire

/-- A reader input port: its u16 words, and its outer length (the parameter's outermost array length, 1 when it is not
an array), which its cut divides. -/
structure Port where
  words : Nat
  outer : Nat
  deriving Repr, BEq

/-- A template of the circuit: its units, its input ports and its output rows' words. -/
structure Template where
  units : Nat
  rows : List (String × Port)
  outs : List (String × Nat)
  deriving Repr

/-- A window circuit's wiring, the dataflow between its proof blocks: the templates by descriptor id, the cuts
(`{tid: {port: k}}`), the wires, and the order of its units' ranges when it states one (`UnitRange`). -/
structure Wiring where
  templates : List (String × Template)
  cuts : List (String × List (String × Nat))
  wires : List Wire
  order : Option (List (String × Nat × Nat)) := none
  deriving Repr

/-- The number of rows template `tid`'s port `p` is cut into (absent: 1). -/
def Wiring.cutsOf (c : Wiring) (tid p : String) : Nat := (((c.cuts.lookup tid).getD []).lookup p).getD 1

/-! ## Refusals -/

inductive Code where
  | cuts | template | port | cut | width | bounds | overlap | order | cycle
  deriving Repr, BEq, DecidableEq

def Code.name : Code → String
  | .cuts => "cuts" | .template => "template" | .port => "port" | .cut => "cut" | .width => "width"
  | .bounds => "bounds" | .overlap => "overlap" | .order => "order" | .cycle => "cycle"

/-- A refusal: its code, and for a wire's own refusal the wire's index. -/
structure Refusal where
  code : Code
  wire : Option Nat
  deriving Repr, BEq

/-- The first malformed cut: a template or port it names that isn't one, or `k` that is 0 or doesn't divide the port's outer
length and words. -/
def cutsOk (c : Wiring) : Bool :=
  c.cuts.all fun (tid, kv) => match c.templates.lookup tid with
    | some t => kv.all fun (p, k) => match t.rows.lookup p with
      | some q => 0 < k && q.outer % k == 0 && q.words % k == 0
      | none => false
    | none => false

/-- The wire's own refusal, before its bounds: its templates, ports, rows and widths. With them, the reader and writer
templates, the reader row's words and the reader's cut. -/
def shape (c : Wiring) (w : Wire) : Except Code (Template × Template) :=
  match c.templates.lookup w.reader, c.templates.lookup w.writer with
  | some rt, some wt =>
    match rt.rows.lookup w.port, wt.outs.lookup w.out with
    | some p, some ow =>
      let k := c.cutsOf w.reader w.port
      if !(1 ≤ w.cuts && w.cut0 + w.cuts ≤ k && (w.cuts == 1 || w.radix.getLast? == some w.cuts) &&
          w.units % w.cuts == 0) then .error .cut
      else if (w.offset : Int) + Circuits.Wire.lowOf w.opairs < 0 ||
          (w.offset : Int) + Circuits.Wire.highOf w.opairs + (p.words / k : Nat) > ow then .error .width
      else .ok (rt, wt)
    | _, _ => .error .port
  | _, _ => .error .template

/-- The bounds in closed form: reader units `[start, start + readerUnits)` among the reader's, the span among the writer's. -/
def boundsOk (w : Wire) (rt wt : Template) : Bool :=
  w.start + w.readerUnits ≤ rt.units && 0 ≤ w.span.1 && w.span.2 < (wt.units : Int)

/-- Two wires of one reader port wire disjoint rectangles: their reader units or their rows are disjoint. -/
def disjoint (a b : Wire) : Bool :=
  a.reader != b.reader || a.port != b.port ||
  a.start + a.readerUnits ≤ b.start || b.start + b.readerUnits ≤ a.start ||
  a.cut0 + a.cuts ≤ b.cut0 || b.cut0 + b.cuts ≤ a.cut0

/-- Every pair of `ws` disjoint. -/
def pairwiseDisjoint : List Wire → Bool
  | [] => true
  | w :: ws => ws.all (disjoint w) && pairwiseDisjoint ws

/-- `a` sorts no later than `b`: by reader, then port, then first reader unit. -/
def before (a b : Wire) : Bool :=
  a.reader < b.reader || (a.reader == b.reader && (a.port < b.port || (a.port == b.port && a.start ≤ b.start)))

/-- Each two neighbours of the list in `before` order. -/
def sortedBy : List Wire → Bool
  | a :: b :: ws => before a b && sortedBy (b :: ws)
  | _ => true

/-- Each wire against the live wires before it, those of its reader port whose units reach past its first unit; the
others are disjoint from it and from every wire after it when the list is sorted (`sortedBy`). -/
def sweep : List Wire → List Wire → Bool
  | _, [] => true
  | live, w :: ws =>
    let live := live.filter fun a => a.reader == w.reader && a.port == w.port && w.start < a.start + a.readerUnits
    live.all (disjoint w) && sweep (w :: live) ws

/-- `pairwiseDisjoint` sorted and swept, in work `O(W log W)` beside the live wires each wire meets: the code compiled in
its place (`pairwiseDisjoint_eq_fast`). -/
def pairwiseDisjointFast (ws : List Wire) : Bool :=
  let s := ws.mergeSort before
  if sortedBy s then sweep [] s else pairwiseDisjoint ws

private def Prec (a b : Wire) : Prop :=
  a.reader < b.reader ∨ (a.reader = b.reader ∧ (a.port < b.port ∨ (a.port = b.port ∧ a.start ≤ b.start)))

private theorem before_iff {a b : Wire} : before a b = true ↔ Prec a b := by
  simp [before, Prec]

private theorem prec_refl (a : Wire) : Prec a a := Or.inr ⟨rfl, Or.inr ⟨rfl, Nat.le_refl _⟩⟩

private theorem prec_trans {a b c : Wire} (h₁ : Prec a b) (h₂ : Prec b c) : Prec a c := by
  rcases h₁ with h₁ | ⟨e₁, h₁ | ⟨f₁, s₁⟩⟩ <;> rcases h₂ with h₂ | ⟨e₂, h₂ | ⟨f₂, s₂⟩⟩
  · exact Or.inl (String.lt_trans h₁ h₂)
  all_goals first
    | exact Or.inl (by rw [← e₂]; exact h₁)
    | exact Or.inl (by rw [e₁]; exact h₂)
    | exact Or.inr ⟨e₁.trans e₂, Or.inl (String.lt_trans h₁ h₂)⟩
    | exact Or.inr ⟨e₁.trans e₂, Or.inl (by rw [← f₂]; exact h₁)⟩
    | exact Or.inr ⟨e₁.trans e₂, Or.inl (by rw [f₁]; exact h₂)⟩
    | exact Or.inr ⟨e₁.trans e₂, Or.inr ⟨f₁.trans f₂, Nat.le_trans s₁ s₂⟩⟩

private theorem disjoint_symm {a b : Wire} (h : disjoint a b = true) : disjoint b a = true := by
  simp only [disjoint, Bool.or_eq_true, bne_iff_ne, ne_eq, decide_eq_true_eq] at h ⊢
  rcases h with ((((h | h) | h) | h) | h) | h
  · exact Or.inl (Or.inl (Or.inl (Or.inl (Or.inl fun e => h e.symm))))
  · exact Or.inl (Or.inl (Or.inl (Or.inl (Or.inr fun e => h e.symm))))
  · exact Or.inl (Or.inl (Or.inr h))
  · exact Or.inl (Or.inl (Or.inl (Or.inr h)))
  · exact Or.inr h
  · exact Or.inl (Or.inr h)

private theorem squeeze {x y z : String} (h₁ : x < y ∨ x = y) (h₂ : y < z ∨ y = z) (h : x = z) : x = y ∧ y = z := by
  subst h
  rcases h₁ with h₁ | h₁
  · rcases h₂ with h₂ | h₂
    · exact absurd (String.lt_trans h₁ h₂) (String.lt_irrefl _)
    · exact absurd (h₂ ▸ h₁) (String.lt_irrefl _)
  · exact ⟨h₁, h₁.symm⟩

private theorem pruned {a w v : Wire} (haw : Prec a w)
    (hk : ¬ (a.reader == w.reader && a.port == w.port && decide (w.start < a.start + a.readerUnits)) = true)
    (hwv : Prec w v) : disjoint a v = true := by
  simp only [disjoint, Bool.or_eq_true, bne_iff_ne, ne_eq, decide_eq_true_eq]
  simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq, not_and, Nat.not_lt] at hk
  by_cases hr : a.reader = v.reader
  · obtain ⟨r₁, r₂⟩ := squeeze (haw.imp_right (·.1)) (hwv.imp_right (·.1)) hr
    have haw' := haw.resolve_left (fun h => String.lt_irrefl _ (r₁ ▸ h)) |>.2
    have hwv' := hwv.resolve_left (fun h => String.lt_irrefl _ (r₂ ▸ h)) |>.2
    by_cases hp : a.port = v.port
    · obtain ⟨p₁, p₂⟩ := squeeze (haw'.imp_right (·.1)) (hwv'.imp_right (·.1)) hp
      have s₁ := haw'.resolve_left (fun h => String.lt_irrefl _ (p₁ ▸ h)) |>.2
      have s₂ := hwv'.resolve_left (fun h => String.lt_irrefl _ (p₂ ▸ h)) |>.2
      exact Or.inl (Or.inl (Or.inl (Or.inr (Nat.le_trans (hk ⟨r₁, p₁⟩) s₂))))
    · exact Or.inl (Or.inl (Or.inl (Or.inl (Or.inr hp))))
  · exact Or.inl (Or.inl (Or.inl (Or.inl (Or.inl hr))))

private theorem sweep_complete : ∀ (live ws : List Wire), (∀ a ∈ live, ∀ v ∈ ws, disjoint a v = true) →
    ws.Pairwise (fun a b => disjoint a b = true) → sweep live ws = true
  | _, [], _, _ => rfl
  | live, w :: ws, h₁, h₂ => by
    rw [List.pairwise_cons] at h₂
    simp only [sweep, Bool.and_eq_true, List.all_eq_true]
    refine ⟨fun a ha => disjoint_symm (h₁ a (List.mem_filter.1 ha).1 w (List.mem_cons_self ..)),
      sweep_complete _ ws (fun a ha v hv => ?_) h₂.2⟩
    rcases List.mem_cons.1 ha with e | e
    · exact e ▸ h₂.1 v hv
    · exact h₁ a (List.mem_filter.1 e).1 v (List.mem_cons_of_mem _ hv)

private theorem sweep_sound : ∀ (live ws : List Wire), ws.Pairwise Prec → (∀ a ∈ live, ∀ v ∈ ws, Prec a v) →
    sweep live ws = true →
    (∀ a ∈ live, ∀ v ∈ ws, disjoint a v = true) ∧ ws.Pairwise (fun a b => disjoint a b = true)
  | _, [], _, _, _ => ⟨by simp, List.Pairwise.nil⟩
  | live, w :: ws, hp, hl, h => by
    simp only [sweep, Bool.and_eq_true, List.all_eq_true] at h
    obtain ⟨hall, hrest⟩ := h
    rw [List.pairwise_cons] at hp
    have ih := sweep_sound _ ws hp.2 (fun a ha v hv => by
      rcases List.mem_cons.1 ha with e | e
      · exact e ▸ hp.1 v hv
      · exact hl a (List.mem_filter.1 e).1 v (List.mem_cons_of_mem _ hv)) hrest
    refine ⟨fun a ha v hv => ?_, List.pairwise_cons.2 ⟨fun v hv => ih.1 w (List.mem_cons_self ..) v hv, ih.2⟩⟩
    by_cases hk : (a.reader == w.reader && a.port == w.port && decide (w.start < a.start + a.readerUnits)) = true
    · have hf : a ∈ live.filter fun a =>
          a.reader == w.reader && a.port == w.port && decide (w.start < a.start + a.readerUnits) :=
        List.mem_filter.2 ⟨ha, hk⟩
      rcases List.mem_cons.1 hv with e | e
      · exact e ▸ disjoint_symm (hall a hf)
      · exact ih.1 a (List.mem_cons_of_mem _ hf) v e
    · refine pruned (hl a ha w (List.mem_cons_self ..)) hk ?_
      rcases List.mem_cons.1 hv with e | e
      · exact e ▸ prec_refl w
      · exact hp.1 v e

private theorem sortedBy_pairwise : ∀ s : List Wire, sortedBy s = true → s.Pairwise Prec
  | [], _ => .nil
  | [_], _ => List.pairwise_singleton _ _
  | a :: b :: ws, h => by
    simp only [sortedBy, Bool.and_eq_true] at h
    have ih := sortedBy_pairwise (b :: ws) h.2
    refine List.pairwise_cons.2 ⟨fun v hv => ?_, ih⟩
    rcases List.mem_cons.1 hv with e | e
    · exact e ▸ before_iff.1 h.1
    · exact prec_trans (before_iff.1 h.1) ((List.pairwise_cons.1 ih).1 v e)

private theorem pairwiseDisjoint_pairwise : ∀ ws : List Wire,
    pairwiseDisjoint ws = true ↔ ws.Pairwise (fun a b => disjoint a b = true)
  | [] => by simp [pairwiseDisjoint]
  | w :: ws => by
    simp [pairwiseDisjoint, List.pairwise_cons, List.all_eq_true, pairwiseDisjoint_pairwise ws]

@[csimp] theorem pairwiseDisjoint_eq_fast : @pairwiseDisjoint = @pairwiseDisjointFast := by
  funext ws
  simp only [pairwiseDisjointFast]
  split
  · rename_i hs
    apply Bool.eq_iff_iff.2
    rw [pairwiseDisjoint_pairwise, ← (List.mergeSort_perm ws before).pairwise_iff disjoint_symm]
    exact ⟨sweep_complete [] _ (by simp), fun h => (sweep_sound [] _ (sortedBy_pairwise _ hs) (by simp) h).2⟩
  · rfl

/-- What `check` establishes: the wires, in order. -/
structure Summary where
  wires : List Wire
  rowWords : List Nat
  deriving Repr

/-- The first wire refusal, by index. -/
def firstWire (c : Wiring) (f : Wire → Option Code) : Option Refusal :=
  (c.wires.zipIdx.find? fun (w, _) => (f w).isSome).bind fun (w, i) => (f w).map fun e => ⟨e, some i⟩

/-- The wire's refusal, with the closed-form bounds. -/
def wireCode (c : Wiring) (w : Wire) : Option Code :=
  match shape c w with
  | .error e => some e
  | .ok (rt, wt) => if boundsOk w rt wt then none else some .bounds

/-- The reader row's words of a wire whose shape passed. -/
def rowWords (c : Wiring) (w : Wire) : Nat :=
  match c.templates.lookup w.reader with
  | some rt => match rt.rows.lookup w.port with
    | some p => p.words / c.cutsOf w.reader w.port
    | none => 0
  | none => 0

/-! ## The order: no unit reads itself

A unit's check holds its output to its inputs, so wires with a cycle among units pass every unit's check on a fixed point
the prover chose: the wires must be acyclic as a graph on units, every reader unit after its writers. An order ranks
ranges of units: each entry `(tid, at, n)` is template `tid`'s units `[at, at + n)`, its rank its place in the order, and
each template's ranges tile its units (`orderOk`), in any order of ranks. A wire is in order when, for every range meeting
its reader units, every range meeting the bounds on the writers it reads there (`Wire.box` of the range's reader indices)
ranks below it (`orderedWire`): every unit a reader unit reads then ranks below it (`check_acyclic`). The order is the
circuit's own when it states one, else the templates in a topological order by their wires between templates, each cut at
the reader units of the wires it reads itself by (`kahn`). -/

/-- A range of an order: template `tid`'s units `[at, at + n)`. -/
abbrev UnitRange := String × Nat × Nat

/-- The ranks of the order's ranges of template `t` that meet units `[lo, hi)`. -/
def ranksIn (o : List UnitRange) (t : String) (lo hi : Nat) : List Nat :=
  (o.zipIdx.filter fun ((s, p0, n), _) => s == t && decide (max p0 lo < min (p0 + n) hi)).map (·.2)

/-- The rank of the first range holding template `t`'s unit `u`. -/
def rangeOf (o : List UnitRange) (t : String) (u : Nat) : Option Nat := (ranksIn o t u (u + 1)).head?

/-- Ranges `(at, n)`, sorted by `at`, tile units `[cur, units)`. -/
def tiles : List (Nat × Nat) → Nat → Nat → Bool
  | [], cur, units => cur == units
  | (p0, n) :: bs, cur, units => p0 == cur && 0 < n && tiles bs (cur + n) units

/-- The order is well formed: each range a template's, and each template's ranges tile its units. -/
def orderOk (c : Wiring) (o : List UnitRange) : Bool :=
  o.all (fun b => (c.templates.lookup b.1).isSome) &&
    c.templates.all fun (t, tt) =>
      tiles (((o.filter (·.1 == t)).map (·.2)).mergeSort fun a b => a.1 ≤ b.1) 0 tt.units

/-- From template `t`, `n` steps back along the first wire (by index) of each template from a template not `done`
(`ins`: each template's wires, as their index and writer). -/
def kahnWalk (ins : Array (List (Nat × Nat))) (done : Array Bool) : Nat → Nat → Nat
  | 0, t => t
  | n + 1, t => match (ins[t]!).find? (fun (_, wr) => !done[wr]!) with
    | some (_, wr) => kahnWalk ins done n wr
    | none => t

/-- Template `t`'s `n` units cut at the first and the end reader unit of each wire it reads itself by (within `[0, n]`), in
unit order; one range `(t, 0, n)` where that leaves none. -/
def selfRanges (ws : List Wire) (t : String) (n : Nat) : List UnitRange :=
  let cs := ((0 :: n :: ws.flatMap fun w =>
    if w.reader == t && w.writer == t then [min w.start n, min (w.start + w.readerUnits) n] else []).mergeSort
      (· ≤ ·)).eraseDups
  match (cs.zip cs.tail).map fun (a, b) => (t, a, b - a) with
  | [] => [(t, 0, n)]
  | rs => rs

/-- `kahn`'s loop, `fuel` the templates not `done` (`ids` in id order, `ranges` each template's ranges in unit order,
`outs` each template's readers wire by wire, `indeg` each template's wires from templates not done). -/
def kahnLoop (ids : Array String) (ranges : String → List UnitRange) (outs : Array (List Nat))
    (ins : Array (List (Nat × Nat))) : Nat → Array Bool → Array Nat → List UnitRange → Except Refusal (List UnitRange)
  | 0, _, _, acc => .ok acc.reverse
  | fuel + 1, done, indeg, acc =>
    match (List.range ids.size).find? (fun i => !done[i]! && indeg[i]! == 0) with
    | some i => kahnLoop ids ranges outs ins fuel (done.set! i true) ((outs[i]!).foldl (fun a r => a.modify r (· - 1)) indeg)
        ((ranges ids[i]!).reverse ++ acc)
    | none =>
      let rest := (List.range ids.size).filter (!done[·]!)
      let t := kahnWalk ins done rest.length (rest.headD 0)
      .error ⟨.cycle, ((ins[t]!).find? (fun (_, wr) => !done[wr]!)).map (·.1)⟩

/-- **The default order**: the templates in a topological order by their wires between templates, the next at each step
the first template by id that reads none not yet placed, each one's units cut at the reader units of the wires it reads
itself by (`selfRanges`), so that such a wire is in order where each of its reader ranges reads only earlier ranges.
Where every template not placed reads one, they hold a cycle: walking back from the first of them along each one's first
wire from one not placed, as many steps as there are of them, ends on the cycle, and the refusal names the first such
wire of the template it ends on. In work `O(templates² + wires)` beside the cuts. -/
def kahn (c : Wiring) : Except Refusal (List UnitRange) :=
  let ids := ((c.templates.map (·.1)).mergeSort (· ≤ ·)).toArray
  let idx : Std.HashMap String Nat := ids.zipIdx.foldl (fun m (t, i) => m.insert t i) {}
  let es := c.wires.zipIdx.filterMap fun (w, j) =>
    if w.writer == w.reader then none else do pure (j, ← idx[w.writer]?, ← idx[w.reader]?)
  let indeg := es.foldl (fun a (_, _, r) => a.modify r (· + 1)) (Array.replicate ids.size 0)
  let outs := es.foldr (fun (_, wr, r) a => a.modify wr (r :: ·)) (Array.replicate ids.size [])
  let ins := es.foldr (fun (j, wr, r) a => a.modify r ((j, wr) :: ·)) (Array.replicate ids.size [])
  let selfWs : Std.HashMap String (List Wire) :=
    c.wires.foldr (fun w m => if w.reader == w.writer then m.insert w.reader (w :: m.getD w.reader []) else m) {}
  let units : Std.HashMap String Nat := c.templates.foldr (fun (t, tt) m => m.insert t tt.units) {}
  kahnLoop ids (fun t => selfRanges (selfWs.getD t []) t (units.getD t 0)) outs ins ids.size
    (Array.replicate ids.size false) indeg []

/-- The order the circuit states, else `kahn`'s. -/
def statedOrder (c : Wiring) : Except Refusal (List UnitRange) :=
  match c.order with
  | some o => .ok o
  | none => kahn c

/-- The order the wires are held to (`statedOrder`), refused (`order`) unless well formed. -/
def orderOf (c : Wiring) : Except Refusal (List UnitRange) :=
  match statedOrder c with
  | .ok o => if orderOk c o then .ok o else .error ⟨.order, none⟩
  | .error r => .error r

/-- The wire is in order `o`, in closed form: for each range meeting its reader units, every range meeting the bounds on
the writers of its reader indices there ranks below it. -/
def orderedWire (o : List UnitRange) (w : Wire) : Bool :=
  o.zipIdx.all fun (b, a) =>
    !w.readsIn b ||
      let s := w.box (w.indicesIn b.2.1 b.2.2).1 (w.indicesIn b.2.1 b.2.2).2
      (ranksIn o w.writer s.1.toNat (s.2.toNat + 1)).all (· < a)

/-- The order's refusal (`orderOf`), else the first wire `f` doesn't put in it, by index. -/
def orderCode (c : Wiring) (f : List UnitRange → Wire → Bool) : Option Refusal :=
  match orderOf c with
  | .error r => some r
  | .ok o => firstWire c fun w => if f o w then none else some .order

/-- **The check at the size of the description**: the refusals in the module docstring's order, else the summary. -/
def check (c : Wiring) : Except Refusal Summary :=
  if !cutsOk c then .error ⟨.cuts, none⟩
  else match firstWire c (wireCode c) with
    | some r => .error r
    | none =>
      if !pairwiseDisjoint c.wires then .error ⟨.overlap, none⟩
      else match orderCode c orderedWire with
        | some r => .error r
        | none => .ok ⟨c.wires, c.wires.map (rowWords c)⟩

/-- The writer `(template, unit, out, offset)` of reader row `(tid, u, port, row)`, by the wires alone: the reader row is
that unit's row `out` from its word `offset` (`Wire.offsetAt`); none when no wire covers it. -/
def Summary.writerOf (s : Summary) (tid : String) (u : Nat) (port : String) (row : Nat) :
    Option (String × Int × String × Nat) :=
  s.wires.findSome? fun w => (w.indexOf tid u port row).map fun i => (w.writer, w.writerAt i, w.out, w.offsetAt i)

/-- The reader rows wired, multiplied through. -/
def Summary.wiredRows (s : Summary) : Nat := (s.wires.map Wire.units).sum

/-- The bits of the reader rows wired, multiplied through. -/
def Summary.wiredBits (s : Summary) : Nat := ((s.wires.zip s.rowWords).map fun (w, n) => w.units * (16 * n)).sum

/-! ## The reference: every covered reader row expanded

The verifier never runs it: it is what `check` is held to, by the difftest and the agreement lemmas. -/

/-- A reader row `(template, unit, port, row)` and its writer `(template, unit, out, offset)`. -/
abbrev Entry := (String × Nat × String × Nat) × (String × Int × String × Nat)

/-- The bounds entry by entry. -/
def boundsRef (w : Wire) (rt wt : Template) : Bool :=
  w.entries.all fun ((_, u, _, _), (_, v, _, _)) => u < rt.units && 0 ≤ v && v < (wt.units : Int)

def wireCodeRef (c : Wiring) (w : Wire) : Option Code :=
  match shape c w with
  | .error e => some e
  | .ok (rt, wt) => if boundsRef w rt wt then none else some .bounds

/-- No reader row twice among `es` (quadratic in the entries). -/
def distinctRows (es : List Entry) : Bool := decide (es.map (·.1)).Nodup

/-- The wire is in order `o`, unit by unit: for each range meeting its reader units, each range holding a unit within
the bounds on the writers of its reader indices there ranks below it. -/
def orderedWireRef (o : List UnitRange) (w : Wire) : Bool :=
  o.zipIdx.all fun (b, a) =>
    !w.readsIn b ||
      let s := w.box (w.indicesIn b.2.1 b.2.2).1 (w.indicesIn b.2.1 b.2.2).2
      (List.range' s.1.toNat (s.2.toNat + 1 - s.1.toNat)).all fun v => (ranksIn o w.writer v (v + 1)).all (· < a)

/-- **The unrolled reference**: `check`'s refusals with the bounds, the overlap and the order taken entry by entry, else
every entry. -/
def checkRef (c : Wiring) : Except Refusal (List Entry) :=
  if !cutsOk c then .error ⟨.cuts, none⟩
  else match firstWire c (wireCodeRef c) with
    | some r => .error r
    | none =>
      let es := c.wires.flatMap Wire.entries
      if !distinctRows es then .error ⟨.overlap, none⟩
      else match orderCode c orderedWireRef with
        | some r => .error r
        | none => .ok es

/-! ## JSON: the difftest's pieces

An instance is `{"templates", "cuts", "wires", "queries"}`, and optionally `"order"`:

* `templates`: `{tid: {"units", "rows": {port: [words, outer]}, "outs": {out: words}}}`, every number a natural, `units`
  and `words` at least 1;
* `cuts`: `{tid: {port: k}}`;
* `wires`: V3a's wire records, `{reader, port, writer, out, at, radix, base, step}` and optionally `cut: [c0, C]`,
  `offset` and `ostep`: strings, `at` and `offset` naturals, `radix` at least one natural each at least 1, `base` an
  integer, `step` and `ostep` one integer a digit, and `C` at least 1;
* `queries`: reader rows `[tid, unit, port, row]` to look up (a sampled proof block's);
* `order`: ranges `[tid, at, n]`, a string and two naturals each (`UnitRange`).

Anything else is refused as malformed. The output is `{"ok": false, "code", "wire"}` (`wire` null but for a wire's own
refusal) or `{"ok": true, "wired_rows", "wired_bits", "spans", "lookups"}`: per wire its least and greatest writer, and per
query its writer `[tid, unit, out, offset]` or null. -/

def Template.ofJson (j : Json) : Except String Template := do
  sameKeys j ["units", "rows", "outs"] "a template"
  let units ← natAt j "units"
  if units < 1 then throw "a template has at least one unit"
  let rows ← (← fields (← field j "rows")).mapM fun (p, v) => do
    match ← arrOf "a row" v with
    | [a, b] =>
      let (words, outer) := (← natOf "words" a, ← natOf "outer" b)
      if words < 1 || outer < 1 then throw s!"row {p}: words and outer at least 1"
      pure (p, ({ words, outer } : Port))
    | _ => throw s!"row {p} is [words, outer]"
  let outs ← (← fields (← field j "outs")).mapM fun (o, v) => do
    let n ← natOf "an output's words" v
    if n < 1 then throw s!"output {o}: at least one word"
    pure (o, n)
  return { units, rows, outs }

def Wiring.ofJson (j : Json) : Except String Wiring := do
  let ts ← (← fields (← field j "templates")).mapM fun (tid, v) => do pure (tid, ← Template.ofJson v)
  let cs ← (← fields (← field j "cuts")).mapM fun (tid, v) => do
    pure (tid, ← (← fields v).mapM fun (p, k) => do pure (p, ← natOf "a cut" k))
  let wires ← (← arrOf "wires" (← field j "wires")).mapM Wire.ofJson
  let order ← match j.getObjVal? "order" with
    | .ok oj => do
      let bs ← (← arrOf "order" oj).mapM fun b => do
        match ← arrOf "a range" b with
        | [t, a, n] => pure (← strOf "a range's template" t, ← natOf "a range's first unit" a, ← natOf "a range's units" n)
        | _ => throw "a range is [template, at, units]"
      pure (some bs)
    | .error _ => pure none
  return { templates := ts, cuts := cs, wires, order }

/-- An instance's keys are `ks`, and `order` when it states one. -/
def instanceKeys (ks : List String) (what : String) (j : Json) : Except String Unit :=
  sameKeys j (if ((keysOf j).getD []).contains "order" then "order" :: ks else ks) what

def queriesOf (j : Json) : Except String (List (String × Nat × String × Nat)) := do
  (← arrOf "queries" (← field j "queries")).mapM fun q => do
    match ← arrOf "a query" q with
    | [a, b, c, d] => pure (← strOf "a query's template" a, ← natOf "a query's unit" b, ← strOf "a query's port" c,
                            ← natOf "a query's row" d)
    | _ => throw "a query is [template, unit, port, row]"

def refusalJson (r : Refusal) : Json :=
  Json.mkObj [("ok", false), ("code", r.code.name), ("wire", match r.wire with | some i => toJson i | none => Json.null)]

def writerJson : Option (String × Int × String × Nat) → Json
  | some (t, v, o, f) => Json.arr #[toJson t, toJson v, toJson o, toJson f]
  | none => Json.null

def instanceCheck (sameKeysOk : Json → Except String Unit) (j : Json) : Except String (Wiring × List (String × Nat × String × Nat)) := do
  sameKeysOk j
  return (← Wiring.ofJson j, ← queriesOf j)

/-- The `wires` piece: `check` on an instance (the section's docstring). -/
def piece (j : Json) : Except String Json := do
  let (c, qs) ← instanceCheck (instanceKeys ["templates", "cuts", "wires", "queries"] "an instance") j
  match check c with
  | .error r => return refusalJson r
  | .ok s =>
    return Json.mkObj [("ok", true), ("wired_rows", toJson s.wiredRows), ("wired_bits", toJson s.wiredBits),
      ("spans", Json.arr (s.wires.map fun w => Json.arr #[toJson w.span.1, toJson w.span.2]).toArray),
      ("lookups", Json.arr (qs.map fun (t, u, p, r) => writerJson (s.writerOf t u p r)).toArray)]

/-- The `wires_unrolled` piece: `checkRef` on an instance, its summary read off the entries. -/
def pieceRef (j : Json) : Except String Json := do
  let (c, qs) ← instanceCheck (instanceKeys ["templates", "cuts", "wires", "queries"] "an instance") j
  match checkRef c with
  | .error r => return refusalJson r
  | .ok es =>
    let spans := c.wires.map fun w =>
      let vs := w.entries.map (·.2.2.1)
      Json.arr #[toJson (vs.foldl min (vs.headD 0)), toJson (vs.foldl max (vs.headD 0))]
    let bits := (c.wires.map fun w => (w.entries.map fun _ => 16 * rowWords c w).sum).sum
    return Json.mkObj [("ok", true), ("wired_rows", toJson es.length), ("wired_bits", toJson bits),
      ("spans", Json.arr spans.toArray),
      ("lookups", Json.arr (qs.map fun (t, u, p, r) =>
        writerJson ((es.find? (·.1 == (t, u, p, r))).map (·.2))).toArray)]

end Spot.Wires
