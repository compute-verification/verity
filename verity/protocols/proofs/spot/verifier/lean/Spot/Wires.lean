/-
A window circuit's wires, checked at the size of their description (`verity.ml.windows.interface.Wire`).

A window's circuit is a population of template instances (`TemplatePopulations_v1`), whose inputs are all Input gates: the
program holds no dataflow between units. The wires are that dataflow. A wire states, once for many units, that reader
template `reader`'s units read rows of their parameter `port` whole as row `out` of a unit of template `writer`:

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
rows) are disjoint, stated pair by pair (`pairwiseDisjoint`) and run as a sweep over the wires sorted by reader, port and
first unit (`overlapFree`), each wire against the wires of its port that start inside its reader units; and the wires
are in the order, stated range by range (`orderedWire`) and run on the order indexed once by template and first unit
(`checkFast`), each wire against the ranges it meets, found by bisection. Its summary gives each reader row's writer
(`Summary.writerOf`, by the wires alone) and the wired rows and bits multiplied through (`wiredRows`, `wiredBits`), the wired term of the count of a
window's free inputs.

`checkRef` is the reference it is held to, every covered reader row expanded (`Wire.entries`): the difftest (the `wires`
and `wires_unrolled` pieces) holds both to the same Python reference. The verifier never runs it.

The refusals, in the order checked (`Code`): `cuts` (a cut names no template or port, or `k` is 0 or doesn't divide the
port's outer length and words), then wire by wire `template` (its reader or writer is no template), `port` (its port is no
input of the reader, or `out` no output of the writer), `cut` (its rows are not among the port's, or the last digit isn't
`cuts`, or `cuts` doesn't divide the wire's reader rows), `width` (a reader row's words are not the writer row's), `bounds` (a reader unit or a writer outside its
template), then `overlap` (some reader row wired twice), then the order (the section of that name): `order` (the circuit's
order names no template, or its ranges of a template don't tile the template's units; or, naming it, the first wire the
order doesn't put after its writers) or, with no order stated, `cycle` (the templates' wires have a cycle; naming a wire
on it).
-/
module

public import Lean.Data.Json
public import Std.Data.HashMap

@[expose] public section

namespace Spot.Wires

open Lean (Json JsonNumber toJson)

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

/-- A wire (the module docstring); `start` is the record's `at`. -/
structure Wire where
  reader : String
  port : String
  writer : String
  out : String
  start : Nat
  radix : List Nat
  base : Int
  step : List Int
  cut0 : Nat
  cuts : Nat
  deriving Repr, BEq

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

/-- The digits' radices and steps, the fastest digit first. -/
def Wire.pairs (w : Wire) : List (Nat × Int) := (w.radix.zip w.step).reverse

/-- The product of the radices of `ps`. -/
def prodR (ps : List (Nat × Int)) : Nat := ps.foldr (fun p n => p.1 * n) 1

/-- The reader rows the wire covers: the product of its radices. -/
def Wire.units (w : Wire) : Nat := prodR w.pairs

/-- `Σ_k d_k · step_k` for index `i`, `ps` the radices and steps fastest first (`Wire.pairs`). -/
def accum : List (Nat × Int) → Nat → Int
  | [], _ => 0
  | (r, s) :: ps, i => ((i % r : Nat) : Int) * s + accum ps (i / r)

/-- Reader index `i`'s writer unit. -/
def Wire.writerAt (w : Wire) (i : Nat) : Int := w.base + accum w.pairs i

/-- `Σ_k min 0 (step_k · (radix_k − 1))`. -/
def lowOf : List (Nat × Int) → Int
  | [] => 0
  | (r, s) :: ps => min 0 (s * ((r : Int) - 1)) + lowOf ps

/-- `Σ_k max 0 (step_k · (radix_k − 1))`. -/
def highOf : List (Nat × Int) → Int
  | [] => 0
  | (r, s) :: ps => max 0 (s * ((r : Int) - 1)) + highOf ps

/-- The least and the greatest writer the wire reads, in closed form. -/
def Wire.span (w : Wire) : Int × Int := (w.base + lowOf w.pairs, w.base + highOf w.pairs)

/-- A lower bound on `accum ps i` over indices `lo ≤ i ≤ hi`, digit by digit fastest first: where every such index has
the same higher digits, its digit's least term and theirs; else the digit's least term over all its values and the bound
on the higher digits. -/
def boxLow : List (Nat × Int) → Nat → Nat → Int
  | [], _, _ => 0
  | (r, s) :: ps, lo, hi =>
    if lo / r = hi / r then min (((lo % r : Nat) : Int) * s) (((hi % r : Nat) : Int) * s) + accum ps (lo / r)
    else min 0 (s * ((r : Int) - 1)) + boxLow ps (lo / r) (hi / r)

/-- The upper bound `boxLow` mirrors. -/
def boxHigh : List (Nat × Int) → Nat → Nat → Int
  | [], _, _ => 0
  | (r, s) :: ps, lo, hi =>
    if lo / r = hi / r then max (((lo % r : Nat) : Int) * s) (((hi % r : Nat) : Int) * s) + accum ps (lo / r)
    else max 0 (s * ((r : Int) - 1)) + boxHigh ps (lo / r) (hi / r)

/-- Bounds on the writers of reader indices `lo … hi`, in closed form (`Wire.writerAt_mem_box`). -/
def Wire.box (w : Wire) (lo hi : Nat) : Int × Int := (w.base + boxLow w.pairs lo hi, w.base + boxHigh w.pairs lo hi)

/-- Reader index `i`'s reader unit and row. -/
def Wire.readerAt (w : Wire) (i : Nat) : Nat × Nat := (w.start + i / w.cuts, w.cut0 + i % w.cuts)

/-- The reader units the wire covers. -/
def Wire.readerUnits (w : Wire) : Nat := w.units / w.cuts

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
      else if p.words / k != ow then .error .width
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

/-! ## The overlap check in `O(n log n)`

`pairwiseDisjoint` compares every pair. Two wires that meet share their reader and port, and each starts before the
other's reader units end; so with the wires sorted by reader, port and first reader unit (`sweepLe`), a wire need only be
compared with the wires after it of its port that start before its reader units end (`sweepFrom`): every wire between
two that meet is of their port and starts before the earlier one ends. `overlapFree` is that sweep, and the compiled code
runs it wherever `pairwiseDisjoint` is called (`pairwiseDisjoint_eq_overlapFree`). Its comparisons are the sort's and,
per wire, the wires of its port that start inside its reader units. -/

/-- The sweep's order: by reader, then port, then first reader unit. -/
def sweepLe (a b : Wire) : Bool :=
  a.reader < b.reader || (a.reader == b.reader && (a.port < b.port || (a.port == b.port && a.start ≤ b.start)))

/-- `w` against the wires after it of its port that start before its reader units end. -/
def sweepFrom (w : Wire) : List Wire → Bool
  | [] => true
  | v :: vs =>
    if v.reader == w.reader && v.port == w.port && v.start < w.start + w.readerUnits then disjoint w v && sweepFrom w vs
    else true

/-- Each wire of a list sorted by `sweepLe` against the wires `sweepFrom` reaches. -/
def sweep : List Wire → Bool
  | [] => true
  | w :: ws => sweepFrom w ws && sweep ws

/-- **The overlap check as a sweep**: `pairwiseDisjoint` (`overlapFree_eq`). -/
def overlapFree (ws : List Wire) : Bool := sweep (ws.mergeSort sweepLe)

theorem pairwiseDisjoint_iff : ∀ ws : List Wire, pairwiseDisjoint ws = true ↔ ws.Pairwise (fun a b => disjoint a b = true)
  | [] => by simp [pairwiseDisjoint]
  | w :: ws => by
    simp [pairwiseDisjoint, List.pairwise_cons, List.all_eq_true, pairwiseDisjoint_iff ws]

theorem disjoint_comm (a b : Wire) : disjoint a b = disjoint b a := by
  rw [Bool.eq_iff_iff]
  simp only [disjoint, Bool.or_eq_true, bne_iff_ne, ne_eq, decide_eq_true_eq]
  constructor <;> rintro (((((h | h) | h) | h) | h) | h)
  all_goals first
    | exact .inl (.inl (.inl (.inl (.inl (Ne.symm h)))))
    | exact .inl (.inl (.inl (.inl (.inr (Ne.symm h)))))
    | exact .inl (.inl (.inl (.inr h)))
    | exact .inl (.inl (.inr h))
    | exact .inl (.inr h)
    | exact .inr h

theorem sweepLe_iff {a b : Wire} : sweepLe a b = true ↔
    a.reader < b.reader ∨ (a.reader = b.reader ∧ (a.port < b.port ∨ (a.port = b.port ∧ a.start ≤ b.start))) := by
  simp [sweepLe]

theorem sweepLe_trans (a b c : Wire) (h1 : sweepLe a b = true) (h2 : sweepLe b c = true) : sweepLe a c = true := by
  rw [sweepLe_iff] at *
  rcases h1 with h1 | ⟨e1, h1 | ⟨f1, h1⟩⟩ <;> rcases h2 with h2 | ⟨e2, h2 | ⟨f2, h2⟩⟩
  · exact .inl (String.lt_trans h1 h2)
  · exact .inl (e2 ▸ h1)
  · exact .inl (e2 ▸ h1)
  · exact .inl (e1 ▸ h2)
  · exact .inr ⟨e1.trans e2, .inl (String.lt_trans h1 h2)⟩
  · exact .inr ⟨e1.trans e2, .inl (f2 ▸ h1)⟩
  · exact .inl (e1 ▸ h2)
  · exact .inr ⟨e1.trans e2, .inl (f1 ▸ h2)⟩
  · exact .inr ⟨e1.trans e2, .inr ⟨f1.trans f2, Nat.le_trans h1 h2⟩⟩

theorem string_lt_of_le_of_ne {s t : String} (h : s ≤ t) (e : s ≠ t) : s < t :=
  Decidable.byContradiction fun h' => e (String.le_antisymm h (String.not_lt.1 h'))

theorem sweepLe_total (a b : Wire) : (sweepLe a b || sweepLe b a) = true := by
  rw [Bool.or_eq_true, sweepLe_iff, sweepLe_iff]
  by_cases e : a.reader = b.reader
  · by_cases f : a.port = b.port
    · rcases Nat.le_total a.start b.start with k | k
      · exact .inl (.inr ⟨e, .inr ⟨f, k⟩⟩)
      · exact .inr (.inr ⟨e.symm, .inr ⟨f.symm, k⟩⟩)
    · rcases String.le_total a.port b.port with g | g
      · exact .inl (.inr ⟨e, .inl (string_lt_of_le_of_ne g f)⟩)
      · exact .inr (.inr ⟨e.symm, .inl (string_lt_of_le_of_ne g (Ne.symm f))⟩)
  · rcases String.le_total a.reader b.reader with g | g
    · exact .inl (.inl (string_lt_of_le_of_ne g e))
    · exact .inr (.inl (string_lt_of_le_of_ne g (Ne.symm e)))

/-- Past a wire `sweepFrom` stops at, in the sweep's order, no wire of `w`'s port starts inside `w`'s reader units. -/
theorem sweepLe_far {w v u : Wire} (hwv : sweepLe w v = true) (hvu : sweepLe v u = true)
    (hv : ¬ (v.reader = w.reader ∧ v.port = w.port ∧ v.start < w.start + w.readerUnits)) :
    ¬ (u.reader = w.reader ∧ u.port = w.port ∧ u.start < w.start + w.readerUnits) := by
  rintro ⟨hr, hp, hs⟩
  apply hv
  rw [sweepLe_iff] at hwv hvu
  rw [hr, hp] at hvu
  rcases hwv with h1 | ⟨e1, h1 | ⟨f1, h1⟩⟩ <;> rcases hvu with h2 | ⟨e2, h2 | ⟨f2, h2⟩⟩
  · exact absurd (String.lt_trans h1 h2) (String.lt_irrefl _)
  · rw [e2] at h1; exact absurd h1 (String.lt_irrefl _)
  · rw [e2] at h1; exact absurd h1 (String.lt_irrefl _)
  · rw [e1] at h2; exact absurd h2 (String.lt_irrefl _)
  · exact absurd (String.lt_trans h1 h2) (String.lt_irrefl _)
  · rw [f2] at h1; exact absurd h1 (String.lt_irrefl _)
  · rw [e1] at h2; exact absurd h2 (String.lt_irrefl _)
  · rw [f1] at h2; exact absurd h2 (String.lt_irrefl _)
  · exact ⟨e2, f2, by omega⟩

theorem disjoint_of_far {w v : Wire} (h : ¬ (v.reader = w.reader ∧ v.port = w.port ∧ v.start < w.start + w.readerUnits)) :
    disjoint w v = true := by
  simp only [disjoint, Bool.or_eq_true, bne_iff_ne, ne_eq, decide_eq_true_eq]
  by_cases hr : v.reader = w.reader
  · by_cases hp : v.port = w.port
    · have : ¬ v.start < w.start + w.readerUnits := fun hs => h ⟨hr, hp, hs⟩
      exact .inl (.inl (.inl (.inr (by omega))))
    · exact .inl (.inl (.inl (.inl (.inr (Ne.symm hp)))))
  · exact .inl (.inl (.inl (.inl (.inl (Ne.symm hr)))))

theorem sweepFrom_iff {w : Wire} : ∀ {l : List Wire}, (∀ v ∈ l, sweepLe w v = true) → l.Pairwise (sweepLe · · = true) →
    (sweepFrom w l = true ↔ ∀ v ∈ l, disjoint w v = true)
  | [], _, _ => by simp [sweepFrom]
  | v :: vs, hw, hl => by
    have ih := sweepFrom_iff (l := vs) (fun u hu => hw u (List.mem_cons_of_mem _ hu)) (List.pairwise_cons.1 hl).2
    unfold sweepFrom
    split
    · rename_i hc
      rw [Bool.and_eq_true, ih, List.forall_mem_cons]
    · rename_i hc
      have hc' : ¬ (v.reader = w.reader ∧ v.port = w.port ∧ v.start < w.start + w.readerUnits) := by
        simpa [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq, and_assoc] using hc
      simp only [true_iff, List.forall_mem_cons]
      refine ⟨disjoint_of_far hc', fun u hu => disjoint_of_far ?_⟩
      exact sweepLe_far (hw v List.mem_cons_self) ((List.pairwise_cons.1 hl).1 u hu) hc'

theorem sweep_iff : ∀ {l : List Wire}, l.Pairwise (sweepLe · · = true) →
    (sweep l = true ↔ l.Pairwise (fun a b => disjoint a b = true))
  | [], _ => by simp [sweep]
  | w :: ws, hl => by
    rw [sweep, Bool.and_eq_true, sweepFrom_iff (List.pairwise_cons.1 hl).1 (List.pairwise_cons.1 hl).2,
      sweep_iff (List.pairwise_cons.1 hl).2, List.pairwise_cons]

/-- **The sweep is the pairwise check.** -/
theorem overlapFree_eq (ws : List Wire) : overlapFree ws = pairwiseDisjoint ws := by
  rw [Bool.eq_iff_iff, overlapFree, sweep_iff (List.pairwise_mergeSort sweepLe_trans sweepLe_total ws),
    pairwiseDisjoint_iff, (List.mergeSort_perm ws sweepLe).pairwise_iff fun {a b} h => (disjoint_comm b a).trans h]

/-- The compiled code runs `overlapFree` for `pairwiseDisjoint`, here above `check` so that `check` is compiled with it,
and in every module that imports this one. -/
@[csimp] theorem pairwiseDisjoint_eq_overlapFree : @pairwiseDisjoint = @overlapFree :=
  funext fun ws => (overlapFree_eq ws).symm

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
  kahnLoop ids (fun t => selfRanges c.wires t (((c.templates.lookup t).map (·.units)).getD 0)) outs ins ids.size
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

/-- The first and the last reader index of the wire's reader units in units `[p0, p0 + n)` of its reader. -/
def Wire.indicesIn (w : Wire) (p0 n : Nat) : Nat × Nat :=
  ((max p0 w.start - w.start) * w.cuts, (min (p0 + n) (w.start + w.readerUnits) - w.start) * w.cuts - 1)

/-- Units `(t, at, n)` meet the wire's reader units. -/
def Wire.readsIn (w : Wire) (b : UnitRange) : Bool :=
  b.1 == w.reader && decide (max b.2.1 w.start < min (b.2.1 + b.2.2) (w.start + w.readerUnits))

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

/-! ## The order check by bisection

`orderedWire` scans the whole order for each wire, and again for each range the wire meets. `Order.index` sorts the
order's ranges once, by template and then first unit, each with its rank and its `reach`, the greatest end among its
template's ranges up to it. Template `t`'s ranges meeting units `[lo, hi)` then lie between the first entry past `t`'s
ranges that end by `lo` and the first past those that start below `hi`, each found by bisection (`Order.allMeeting`,
`Order.allMeeting_iff`), for any order. `orderedIdx` is `orderedWire` on that index (`orderedIdx_eq`): per wire only the
ranges it meets, and per such range only the ranges the bounds on its writers meet, as Python's `check_order` does.
`checkFast` is `check` with the order indexed once, and the compiled code runs it wherever `check` is called
(`check_eq_checkFast`). -/

namespace Order

/-- A range of the order in its index: template `tid`'s units `[p0, p0 + n)`, its rank, and the greatest end among its
template's ranges up to it in the index. -/
structure Ranged where
  tid : String
  p0 : Nat
  n : Nat
  rank : Nat
  reach : Nat

/-- The index's order: by template, then first unit. -/
def keyLe (a b : String × Nat × Nat × Nat) : Bool := a.1 < b.1 || (a.1 == b.1 && a.2.1 ≤ b.2.1)

/-- Each range with its `reach`, `prev` the template and reach of the range before it. -/
def withReach : Option (String × Nat) → List (String × Nat × Nat × Nat) → List Ranged
  | _, [] => []
  | prev, (t, p0, n, k) :: rs =>
    let r := match prev with
      | some (s, m) => if s == t then max m (p0 + n) else p0 + n
      | none => p0 + n
    ⟨t, p0, n, k, r⟩ :: withReach (some (t, r)) rs

/-- The order's ranges sorted by template and first unit, each with its rank and reach. -/
def index (o : List UnitRange) : Array Ranged :=
  (withReach none ((o.zipIdx.map fun ((t, p0, n), k) => (t, p0, n, k)).mergeSort keyLe)).toArray

/-- The least `i` in `[lo, hi]` with `q i`, by bisection, for `q` monotone, false below `lo` and true at `hi`. -/
def firstTrue (q : Nat → Bool) (lo hi : Nat) : Nat :=
  if lo < hi then
    if q ((lo + hi) / 2) then firstTrue q lo ((lo + hi) / 2) else firstTrue q ((lo + hi) / 2 + 1) hi
  else lo
termination_by hi - lo

/-- `p` at entry `i` of `a`, true past its end. -/
def pAt (a : Array Ranged) (p : Ranged → Bool) (i : Nat) : Bool := (a[i]?.map p).getD true

/-- The range is template `t`'s and meets units `[lo, hi)`. -/
def meets (t : String) (lo hi : Nat) (e : Ranged) : Bool := e.tid == t && decide (max e.p0 lo < min (e.p0 + e.n) hi)

/-- The range is past template `t`'s ranges that end by `lo`. -/
def pastLo (t : String) (lo : Nat) (e : Ranged) : Bool := decide (t < e.tid) || (e.tid == t && decide (lo < e.reach))

/-- The range is past template `t`'s ranges that start below `hi`. -/
def pastHi (t : String) (hi : Nat) (e : Ranged) : Bool := decide (t < e.tid) || (e.tid == t && decide (hi ≤ e.p0))

/-- `f` holds at every range of template `t` meeting units `[lo, hi)`: the ranges between the first past those ending
by `lo` and the first past those starting below `hi`, each found by bisection. -/
def allMeeting (a : Array Ranged) (t : String) (lo hi : Nat) (f : Ranged → Bool) : Bool :=
  let i := firstTrue (pAt a (pastLo t lo)) 0 a.size
  let j := firstTrue (pAt a (pastHi t hi)) 0 a.size
  (List.range' i (j - i)).all fun k => (a[k]?.map fun e => !meets t lo hi e || f e).getD true

end Order

/-- `orderedWire` on the order's index (`orderedIdx_eq`): for each range meeting the wire's reader units, every range
meeting the bounds on the writers of its reader indices there ranks below it. -/
def orderedIdx (a : Array Order.Ranged) (w : Wire) : Bool :=
  Order.allMeeting a w.reader w.start (w.start + w.readerUnits) fun b =>
    let s := w.box (w.indicesIn b.p0 b.n).1 (w.indicesIn b.p0 b.n).2
    Order.allMeeting a w.writer s.1.toNat (s.2.toNat + 1) (·.rank < b.rank)

/-- `orderCode c orderedWire` with the order indexed once (`orderCodeFast_eq`). -/
def orderCodeFast (c : Wiring) : Option Refusal :=
  match orderOf c with
  | .error r => some r
  | .ok o =>
    let a := Order.index o
    firstWire c fun w => if orderedIdx a w then none else some .order

/-- `check` with the order indexed once (`check_eq_checkFast`). -/
def checkFast (c : Wiring) : Except Refusal Summary :=
  if !cutsOk c then .error ⟨.cuts, none⟩
  else match firstWire c (wireCode c) with
    | some r => .error r
    | none =>
      if !pairwiseDisjoint c.wires then .error ⟨.overlap, none⟩
      else match orderCodeFast c with
        | some r => .error r
        | none => .ok ⟨c.wires, c.wires.map (rowWords c)⟩

namespace Order

theorem firstTrue_spec {q : Nat → Bool} (hm : ∀ i j, i ≤ j → q i = true → q j = true) :
    ∀ lo hi, lo ≤ hi → q hi = true → (∀ i < lo, q i = false) →
      (∀ i < firstTrue q lo hi, q i = false) ∧ ∀ i, firstTrue q lo hi ≤ i → q i = true := by
  intro lo hi
  induction lo, hi using firstTrue.induct q with
  | case1 lo hi hlt hq ih =>
    intro hle hhi hlo
    rw [firstTrue]
    simp only [hlt, hq, ↓reduceIte]
    exact ih (by omega) hq hlo
  | case2 lo hi hlt hq ih =>
    intro hle hhi hlo
    rw [firstTrue]
    simp only [hlt, hq, ↓reduceIte, Bool.false_eq_true]
    refine ih (by omega) hhi fun i hi' => ?_
    cases e : q i
    · rfl
    · exact absurd (hm _ _ (by omega) e) hq
  | case3 lo hi hlt =>
    intro hle hhi hlo
    rw [firstTrue]
    simp only [hlt, ↓reduceIte]
    have : lo = hi := by omega
    subst this
    exact ⟨hlo, fun i hi' => hm _ _ hi' hhi⟩

/-- A range as the order's tuple and rank. -/
def Ranged.key (e : Ranged) : String × Nat × Nat × Nat := (e.tid, e.p0, e.n, e.rank)

theorem withReach_key : ∀ (prev : Option (String × Nat)) (l : List (String × Nat × Nat × Nat)),
    (withReach prev l).map Ranged.key = l
  | _, [] => rfl
  | _, (t, p0, n, k) :: rs => by
    simp only [withReach, List.map_cons, Ranged.key, withReach_key _ rs]

theorem le_reach : ∀ {prev : Option (String × Nat)} {l : List (String × Nat × Nat × Nat)},
    ∀ e ∈ withReach prev l, e.p0 + e.n ≤ e.reach
  | _, [] => by simp [withReach]
  | prev, (t, p0, n, k) :: rs => by
    simp only [withReach, List.mem_cons, forall_eq_or_imp]
    refine ⟨?_, le_reach⟩
    split <;> (try split) <;> omega

theorem keyLe_iff {a b : String × Nat × Nat × Nat} : keyLe a b = true ↔ a.1 < b.1 ∨ (a.1 = b.1 ∧ a.2.1 ≤ b.2.1) := by
  simp [keyLe]

theorem string_le_of_keyLe {a b : String × Nat × Nat × Nat} (h : keyLe a b = true) : a.1 ≤ b.1 := by
  rcases keyLe_iff.1 h with h | ⟨h, _⟩
  · exact Std.le_of_not_ge fun a => a h
  · exact h ▸ String.le_refl _

theorem reach_mono_of : ∀ {s : String} {m : Nat} {l : List (String × Nat × Nat × Nat)},
    l.Pairwise (keyLe · · = true) → (∀ x ∈ l, s ≤ x.1) → ∀ e ∈ withReach (some (s, m)) l, e.tid = s → m ≤ e.reach
  | _, _, [], _, _ => by simp [withReach]
  | s, m, (t, p0, n, k) :: rs, hl, hs => by
    have hrs := List.pairwise_cons.1 hl
    have ht : ∀ x ∈ rs, t ≤ x.1 := fun x hx => string_le_of_keyLe (hrs.1 x hx)
    simp only [withReach, List.mem_cons, forall_eq_or_imp]
    by_cases e : s = t
    · subst e
      simp only [beq_self_eq_true, ↓reduceIte]
      refine ⟨fun _ => Nat.le_max_left _ _, fun e he het => ?_⟩
      exact Nat.le_trans (Nat.le_max_left _ _) (reach_mono_of hrs.2 ht e he het)
    · refine ⟨fun h => absurd h.symm e, fun x hx hxs => absurd ?_ e⟩
      have hx' : x.key ∈ rs := withReach_key (some (t, _)) rs ▸ List.mem_map_of_mem hx
      have h1 : t ≤ x.tid := ht _ hx'
      have h2 : s ≤ t := hs (t, p0, n, k) List.mem_cons_self
      exact String.le_antisymm h2 (hxs ▸ h1)

theorem reach_pairwise : ∀ {prev : Option (String × Nat)} {l : List (String × Nat × Nat × Nat)},
    l.Pairwise (keyLe · · = true) → (withReach prev l).Pairwise fun x y => x.tid = y.tid → x.reach ≤ y.reach
  | _, [], _ => by simp [withReach]
  | prev, (t, p0, n, k) :: rs, hl => by
    have hrs := List.pairwise_cons.1 hl
    simp only [withReach, List.pairwise_cons]
    exact ⟨fun e he het => reach_mono_of hrs.2 (fun x hx => string_le_of_keyLe (hrs.1 x hx)) e he het.symm,
      reach_pairwise hrs.2⟩

theorem keyLe_trans (a b c : String × Nat × Nat × Nat) (h1 : keyLe a b = true) (h2 : keyLe b c = true) :
    keyLe a c = true := by
  rw [keyLe_iff] at *
  rcases h1 with h1 | ⟨e1, h1⟩ <;> rcases h2 with h2 | ⟨e2, h2⟩
  · exact .inl (String.lt_trans h1 h2)
  · exact .inl (e2 ▸ h1)
  · exact .inl (e1 ▸ h2)
  · exact .inr ⟨e1.trans e2, Nat.le_trans h1 h2⟩

theorem keyLe_total (a b : String × Nat × Nat × Nat) : (keyLe a b || keyLe b a) = true := by
  rw [Bool.or_eq_true, keyLe_iff, keyLe_iff]
  by_cases e : a.1 = b.1
  · rcases Nat.le_total a.2.1 b.2.1 with k | k
    · exact .inl (.inr ⟨e, k⟩)
    · exact .inr (.inr ⟨e.symm, k⟩)
  · rcases String.le_total a.1 b.1 with g | g
    · exact .inl (.inl (string_lt_of_le_of_ne g e))
    · exact .inr (.inl (string_lt_of_le_of_ne g (Ne.symm e)))

/-- The index in order: by template, then first unit, and within a template each range's reach at least the reach of
those before it. -/
theorem index_pairwise (o : List UnitRange) : (index o).toList.Pairwise fun x y =>
    keyLe x.key y.key = true ∧ (x.tid = y.tid → x.reach ≤ y.reach) := by
  have hs := List.pairwise_mergeSort keyLe_trans keyLe_total (o.zipIdx.map fun ((t, p0, n), k) => (t, p0, n, k))
  have hk := hs
  rw [← withReach_key none (List.mergeSort _ keyLe)] at hk
  rw [index, List.toList_toArray]
  exact (List.pairwise_map.1 hk).and (reach_pairwise hs)

/-- The index holds the order's ranges, each with its rank. -/
theorem mem_index {o : List UnitRange} {t : String} {p0 n k : Nat} :
    (∃ e ∈ (index o).toList, e.key = (t, p0, n, k)) ↔ ((t, p0, n), k) ∈ o.zipIdx := by
  rw [index, List.toList_toArray, ← List.mem_map, withReach_key, List.mem_mergeSort, List.mem_map]
  constructor
  · rintro ⟨⟨⟨t', p0', n'⟩, k'⟩, h, he⟩
    simp only [Prod.mk.injEq] at he
    obtain ⟨rfl, rfl, rfl, rfl⟩ := he
    exact h
  · exact fun h => ⟨_, h, rfl⟩

theorem le_reach_index (o : List UnitRange) : ∀ e ∈ (index o).toList, e.p0 + e.n ≤ e.reach := by
  rw [index, List.toList_toArray]
  exact le_reach

theorem pAt_mono {a : Array Ranged} {p : Ranged → Bool} {R : Ranged → Ranged → Prop} (ha : a.toList.Pairwise R)
    (hR : ∀ x y, R x y → p x = true → p y = true) : ∀ i j, i ≤ j → pAt a p i = true → pAt a p j = true := by
  intro i j hij hi
  unfold pAt at *
  by_cases hj : j < a.size
  · have hi' : i < a.size := by omega
    rw [Array.getElem?_eq_getElem hj]
    rw [Array.getElem?_eq_getElem hi'] at hi
    rcases Nat.lt_or_eq_of_le hij with h | rfl
    · have := List.pairwise_iff_getElem.1 ha i j (by simpa using hi') (by simpa using hj) h
      simp only [Array.getElem_toList] at this
      exact hR _ _ this hi
    · exact hi
  · rw [Array.getElem?_eq_none (by omega)]
    rfl

theorem pAt_size (a : Array Ranged) (p : Ranged → Bool) : pAt a p a.size = true := by
  simp [pAt]

/-- **The bisection finds every range meeting the units**: `allMeeting` is `f` at each of template `t`'s ranges meeting
`[lo, hi)`, on an index in order whose reaches bound their ends. -/
theorem allMeeting_iff {a : Array Ranged}
    (ha : a.toList.Pairwise fun x y => keyLe x.key y.key = true ∧ (x.tid = y.tid → x.reach ≤ y.reach))
    (hr : ∀ e ∈ a.toList, e.p0 + e.n ≤ e.reach) {t : String} {lo hi : Nat} {f : Ranged → Bool} :
    allMeeting a t lo hi f = true ↔ ∀ e ∈ a.toList, meets t lo hi e = true → f e = true := by
  have hlo := firstTrue_spec (pAt_mono ha fun x y hxy (h : pastLo t lo x = true) => by
      simp only [pastLo, Bool.or_eq_true, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at h ⊢
      rcases keyLe_iff.1 hxy.1 with k | ⟨k, k'⟩ <;> simp only [Ranged.key] at k
      · rcases h with h | ⟨rfl, h⟩
        · exact .inl (String.lt_trans h k)
        · exact .inl k
      · rcases h with h | ⟨rfl, h⟩
        · exact .inl (k ▸ h)
        · exact .inr ⟨k.symm, Nat.lt_of_lt_of_le h (hxy.2 k)⟩)
    0 a.size (Nat.zero_le _) (pAt_size _ _) (fun _ h => absurd h (Nat.not_lt_zero _))
  have hhi := firstTrue_spec (pAt_mono ha fun x y hxy (h : pastHi t hi x = true) => by
      simp only [pastHi, Bool.or_eq_true, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at h ⊢
      rcases keyLe_iff.1 hxy.1 with k | ⟨k, k'⟩
      · simp only [Ranged.key] at k
        rcases h with h | ⟨rfl, h⟩
        · exact .inl (String.lt_trans h k)
        · exact .inl k
      · simp only [Ranged.key] at k k'
        rcases h with h | ⟨rfl, h⟩
        · exact .inl (k ▸ h)
        · exact .inr ⟨k.symm, Nat.le_trans h k'⟩)
    0 a.size (Nat.zero_le _) (pAt_size _ _) (fun _ h => absurd h (Nat.not_lt_zero _))
  unfold allMeeting
  generalize firstTrue (pAt a (pastLo t lo)) 0 a.size = i at hlo
  generalize firstTrue (pAt a (pastHi t hi)) 0 a.size = j at hhi
  simp only [List.all_eq_true, List.mem_range'_1]
  constructor
  · intro h e he hm
    obtain ⟨k, hk, rfl⟩ := List.getElem_of_mem he
    simp only [Array.length_toList] at hk
    have hrk := hr _ he
    simp only [Array.getElem_toList] at hm hrk ⊢
    simp only [meets, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hm
    have hki : i ≤ k := by
      refine Nat.le_of_not_lt fun hki => ?_
      have := hlo.1 k hki
      simp only [pAt, Array.getElem?_eq_getElem hk, Option.map_some, Option.getD_some, pastLo, Bool.or_eq_false_iff,
        Bool.and_eq_false_imp, beq_iff_eq, decide_eq_false_iff_not] at this
      have := this.2 hm.1
      omega
    have hkj : k < j := by
      refine Nat.lt_of_not_le fun hkj => ?_
      have := hhi.2 k hkj
      simp only [pAt, Array.getElem?_eq_getElem hk, Option.map_some, Option.getD_some, pastHi, Bool.or_eq_true,
        Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at this
      rcases this with h | ⟨_, h⟩
      · rw [hm.1] at h; exact String.lt_irrefl _ h
      · omega
    have := h k ⟨hki, by omega⟩
    simp only [Array.getElem?_eq_getElem hk, Option.map_some, Option.getD_some, Bool.or_eq_true, Bool.not_eq_true',
      meets, Bool.and_eq_false_imp, beq_iff_eq, decide_eq_false_iff_not] at this
    rcases this with h' | h'
    · exact absurd hm.2 (h' hm.1)
    · exact h'
  · intro h k _
    cases hk : a[k]? with
    | none => rfl
    | some e =>
      simp only [Option.map_some, Option.getD_some, Bool.or_eq_true, Bool.not_eq_true']
      cases hm : meets t lo hi e
      · exact .inl rfl
      · exact .inr (h e (Array.mem_toList_iff.2 (Array.mem_of_getElem? hk)) hm)

end Order

theorem mem_ranksIn' {o : List UnitRange} {t : String} {lo hi k : Nat} :
    k ∈ ranksIn o t lo hi ↔ ∃ p0 n, ((t, p0, n), k) ∈ o.zipIdx ∧ max p0 lo < min (p0 + n) hi := by
  simp only [ranksIn, List.mem_map, List.mem_filter, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq]
  constructor
  · rintro ⟨⟨⟨s, p0, n⟩, k'⟩, ⟨hm, rfl, h⟩, rfl⟩
    exact ⟨p0, n, hm, h⟩
  · rintro ⟨p0, n, hm, h⟩
    exact ⟨((t, p0, n), k), ⟨hm, rfl, h⟩, rfl⟩

theorem readsIn_eq_meets (w : Wire) (e : Order.Ranged) :
    w.readsIn (e.tid, e.p0, e.n) = Order.meets w.reader w.start (w.start + w.readerUnits) e := rfl

/-- **The order's index decides `orderedWire`.** -/
theorem orderedIdx_eq (o : List UnitRange) (w : Wire) : orderedIdx (Order.index o) w = orderedWire o w := by
  have hall := @Order.allMeeting_iff _ (Order.index_pairwise o) (Order.le_reach_index o)
  rw [Bool.eq_iff_iff, orderedIdx, hall, orderedWire, List.all_eq_true]
  simp only [hall]
  constructor
  · rintro h ⟨⟨t, p0, n⟩, k⟩ hx
    obtain ⟨e, he, hek⟩ := Order.mem_index.2 hx
    simp only [Order.Ranged.key, Prod.mk.injEq] at hek
    obtain ⟨rfl, rfl, rfl, rfl⟩ := hek
    simp only [Bool.or_eq_true, Bool.not_eq_true', List.all_eq_true, decide_eq_true_eq]
    cases hr : w.readsIn (e.tid, e.p0, e.n)
    · exact .inl rfl
    refine .inr fun k' hk' => ?_
    obtain ⟨p0', n', hy, hm'⟩ := mem_ranksIn'.1 hk'
    obtain ⟨e', he', hek'⟩ := Order.mem_index.2 hy
    simp only [Order.Ranged.key, Prod.mk.injEq] at hek'
    obtain ⟨h1, rfl, rfl, rfl⟩ := hek'
    refine of_decide_eq_true (h e he ((readsIn_eq_meets w e).symm.trans hr) e' he' ?_)
    simpa [Order.meets, h1] using hm'
  · intro h e he hm e' he' hm'
    have hx := Order.mem_index.1 ⟨e, he, rfl⟩
    have hx' := Order.mem_index.1 ⟨e', he', rfl⟩
    have := h _ hx
    simp only [Bool.or_eq_true, Bool.not_eq_true', List.all_eq_true, decide_eq_true_eq] at this
    rcases this with hr | hr
    · rw [readsIn_eq_meets] at hr
      rw [hr] at hm
      cases hm
    · simp only [Order.meets, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hm'
      exact decide_eq_true (hr _ (mem_ranksIn'.2 ⟨_, _, hm'.1 ▸ hx', hm'.2⟩))

theorem orderCodeFast_eq (c : Wiring) : orderCodeFast c = orderCode c orderedWire := by
  unfold orderCodeFast orderCode
  cases orderOf c <;> simp only [orderedIdx_eq]

/-- The compiled code runs `checkFast` for `check` in every module that imports this one. -/
@[csimp] theorem check_eq_checkFast : @check = @checkFast := by
  funext c
  unfold check checkFast
  rw [orderCodeFast_eq]

/-- Whether wire `w` covers reader row `(tid, u, port, row)`, and if so its index. -/
def Wire.indexOf (w : Wire) (tid : String) (u : Nat) (port : String) (row : Nat) : Option Nat :=
  if w.reader == tid && w.port == port && w.start ≤ u && u < w.start + w.readerUnits && w.cut0 ≤ row &&
     row < w.cut0 + w.cuts then some ((u - w.start) * w.cuts + (row - w.cut0)) else none

/-- The writer `(template, unit, out)` of reader row `(tid, u, port, row)`, by the wires alone; none when no wire covers it. -/
def Summary.writerOf (s : Summary) (tid : String) (u : Nat) (port : String) (row : Nat) : Option (String × Int × String) :=
  s.wires.findSome? fun w => (w.indexOf tid u port row).map fun i => (w.writer, w.writerAt i, w.out)

/-- The reader rows wired, multiplied through. -/
def Summary.wiredRows (s : Summary) : Nat := (s.wires.map Wire.units).sum

/-- The bits of the reader rows wired, multiplied through. -/
def Summary.wiredBits (s : Summary) : Nat := ((s.wires.zip s.rowWords).map fun (w, n) => w.units * (16 * n)).sum

/-! ## The reference: every covered reader row expanded

The verifier never runs it: it is what `check` is held to, by the difftest and the agreement lemmas. -/

/-- A reader row `(template, unit, port, row)` and its writer `(template, unit, out)`. -/
abbrev Entry := (String × Nat × String × Nat) × (String × Int × String)

/-- Every reader row the wire covers, with its writer, index by index. -/
def Wire.entries (w : Wire) : List Entry :=
  (List.range w.units).map fun i =>
    ((w.reader, (w.readerAt i).1, w.port, (w.readerAt i).2), (w.writer, w.writerAt i, w.out))

/-- The bounds entry by entry. -/
def boundsRef (w : Wire) (rt wt : Template) : Bool :=
  w.entries.all fun ((_, u, _, _), (_, v, _)) => u < rt.units && 0 ≤ v && v < (wt.units : Int)

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
* `wires`: V3a's wire records, `{reader, port, writer, out, at, radix, base, step}` and optionally `cut: [c0, C]`: strings,
  `at` a natural, `radix` at least one natural each at least 1, `base` an integer, `step` one integer a digit, and `C` at
  least 1;
* `queries`: reader rows `[tid, unit, port, row]` to look up (a sampled proof block's);
* `order`: ranges `[tid, at, n]`, a string and two naturals each (`UnitRange`).

Anything else is refused as malformed. The output is `{"ok": false, "code", "wire"}` (`wire` null but for a wire's own
refusal) or `{"ok": true, "wired_rows", "wired_bits", "spans", "lookups"}`: per wire its least and greatest writer, and per
query its writer `[tid, unit, out]` or null. -/

def int? : Json → Option Int
  | .num n => if n.exponent == 0 then some n.mantissa else none
  | _ => none

def nat? (j : Json) : Option Nat := (int? j).bind fun i => if 0 ≤ i then some i.toNat else none

def keysOf : Json → Option (List String)
  | .obj kvs => some (kvs.toArray.map (·.1)).toList
  | _ => none

def fields : Json → Except String (List (String × Json))
  | .obj kvs => .ok (kvs.toArray.map fun ⟨k, v⟩ => (k, v)).toList
  | _ => .error "not an object"

def field (j : Json) (k : String) : Except String Json :=
  match j.getObjVal? k with
  | .ok v => .ok v
  | .error _ => .error s!"no {k}"

def natAt (j : Json) (k : String) : Except String Nat := do
  match nat? (← field j k) with
  | some n => .ok n
  | none => .error s!"{k} is not a natural"

def natOf (what : String) (j : Json) : Except String Nat :=
  match nat? j with
  | some n => .ok n
  | none => .error s!"{what} is not a natural"

def intOf (what : String) (j : Json) : Except String Int :=
  match int? j with
  | some n => .ok n
  | none => .error s!"{what} is not an integer"

def strOf (what : String) : Json → Except String String
  | .str s => .ok s
  | _ => .error s!"{what} is not a string"

def arrOf (what : String) : Json → Except String (List Json)
  | .arr a => .ok a.toList
  | _ => .error s!"{what} is not a list"

def sameKeys (j : Json) (ks : List String) (what : String) : Except String Unit :=
  match keysOf j with
  | some got => if got.mergeSort (· ≤ ·) == ks.mergeSort (· ≤ ·) then .ok () else .error s!"{what}: keys {got}, not {ks}"
  | none => .error s!"{what} is not an object"

def WIRE_KEYS : List String := ["reader", "port", "writer", "out", "at", "radix", "base", "step"]

def Wire.ofJson (j : Json) : Except String Wire := do
  let ks := (keysOf j).getD []
  if ks.contains "cut" then sameKeys j ("cut" :: WIRE_KEYS) "a wire" else sameKeys j WIRE_KEYS "a wire"
  let s := fun k => do strOf k (← field j k)
  let radix ← (← arrOf "radix" (← field j "radix")).mapM (natOf "a radix")
  let step ← (← arrOf "step" (← field j "step")).mapM (intOf "a step")
  if radix.isEmpty || radix.length != step.length || radix.any (· < 1) then
    throw "a wire is at least one digit of radix 1 or more, and one step a digit"
  let (cut0, cuts) ← match j.getObjVal? "cut" with
    | .ok cj => do
      let xs ← (← arrOf "cut" cj).mapM (natOf "a cut")
      match xs with
      | [a, b] => if 1 ≤ b then pure (a, b) else throw "a wire's cut is [c0, C], C at least 1"
      | _ => throw "a wire's cut is [c0, C]"
    | .error _ => pure (0, 1)
  return { reader := ← s "reader", port := ← s "port", writer := ← s "writer", out := ← s "out",
           start := ← natAt j "at", radix, base := ← intOf "base" (← field j "base"), step, cut0, cuts }

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

def writerJson : Option (String × Int × String) → Json
  | some (t, v, o) => Json.arr #[toJson t, toJson v, toJson o]
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
