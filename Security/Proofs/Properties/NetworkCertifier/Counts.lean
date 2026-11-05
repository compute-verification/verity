import Properties.NetworkCertifier.Wired
import Proofs.Properties.NetworkScan.Lowered
import Proofs.Properties.NetworkScan.Query
import Proofs.Properties.NetworkScan.OfProg
import Proofs.Properties.NetworkCertifier.Slots
import Proofs.Properties.NetworkCertifier.Sources
import Proofs.Properties.NetworkCertifier.Record
import Proofs.Properties.NetworkCertifier.Rects

/-!
# An accepted circuit record's free count bounds its circuit's free inputs

The count is over slots (`Slot`): a template, one of its instances and a bit of its parameters' words, the record's
templates holding `inputs` of them (`slots`, `card_slots_le`). Each input gate of the members' joint program is a slot's
claims position (`slotPos`, at `SampledProofs.Bits.inBit`) or the source gate of an input column no port holds at an
instance (`pads`, `padPos`): `cover`. A slot a wire or a held declaration fills (`filled`, `rectBox`) has its gate wired
(`recordWires`) or registered (`regGates`): `covered`. Those slots are disjoint and number at least `wired` and
`registered`, so the free inputs outside the registered rows number at most `free` and the padding columns' bits
(`padBits`): `counts_le`.

* `recordWires_eq_none`, `recordWires_ne_none`: a gate no wire pair reads, and one a pair reads.
* `portGates_eq`: a resolved port's bits are its slots' claims positions.
* `cover`: each input gate is a slot's or a padding column's.
* `covered`: a filled slot's gate is wired or registered.
* `counts_le_of`, `counts_le`: the bound, where `P`'s Definitions' keys are distinct.
-/

namespace Properties.NetworkCertifier.Counts

open Lean (Json)
open Flock.Program (Prog Ty)
open NetworkCheck.Period (templateNodes bitsOf)
open NetworkCheck.Circuit
open SampledProofs.Bits (Members inBit keyIdx keyProg prog consts base)
open Properties.NetworkScan (jointOf FilesOf MembersOf instPos)
open Properties.NetworkScan.Lowered (srcs exists_member getElem?_jointOf bgateAt_input prog_input_iff posOf_input)
open Properties.NetworkScan.Query (evaluate_inv locate_any_iff unit_unique instPos_eq list_getD_of_lt
  array_getD_of_lt tiles_mem def?_ofProg_root templateNode_lt templateNode_fn countAt_ofProg)

/-! ## Slots at the members' gates -/

/-- **A slot's claims position**: template `s.1`'s instance `s.2.1`'s input bit `s.2.2`, its word's bit, at copy 0 of
the unit the code locates at the instance (`instPos`, `SampledProofs.Bits.inBit`). -/
def slotPos (P : Prog) (rs : Array Flock.TemplateQuery.TRange) (ms : Members) (s : Slot) : ℕ :=
  instPos P rs ms s.1 s.2.1 fun c m v => inBit c m v 0 (s.2.2 / 16) (s.2.2 % 16)

/-- **A padding slot's gate**: template `s.1`'s instance `s.2.1`'s source gate at input column `s.2.2`. -/
def padPos (P : Prog) (rs : Array Flock.TemplateQuery.TRange) (ms : Members) (s : Slot) : ℕ :=
  instPos P rs ms s.1 s.2.1 fun c _ u => (keyIdx c (srcs c) s.2.2).map (2 + u * srcs c + ·)

/-- **The padding slots**: each template's instances at its file's input columns no port holds (`unheld`). -/
def pads (P : Prog) (fs : List Flock.Circuit) : Finset Slot :=
  bigU ((List.range (templateNodes P).length).map fun i =>
    pbox i ((P.defs[P.root]!).body.nodes[(templateNodes P).getD i 0]!).n (unheld (fs.getD i default)))

/-- **The padding's bits**: each template's instances times its file's input columns no port holds. -/
def padBits (P : Prog) (fs : List Flock.Circuit) : ℕ :=
  ((List.range (templateNodes P).length).map fun i =>
    ((P.defs[P.root]!).body.nodes[(templateNodes P).getD i 0]!).n * (unheld (fs.getD i default)).card).sum

theorem recordWires_eq_none {P : Prog} {rs : Array Flock.TemplateQuery.TRange} {ms : Members} {r : Stated} {g : ℕ}
    (h : recordWires P rs ms r g = none) : ∀ q ∈ wirePairs P rs ms r, q.1 ≠ g := by
  intro q hq hqg
  unfold recordWires at h
  rw [Option.map_eq_none_iff, List.find?_eq_none] at h
  exact h q hq (by simp [hqg])

theorem recordWires_ne_none {P : Prog} {rs : Array Flock.TemplateQuery.TRange} {ms : Members} {r : Stated} {g y : ℕ}
    (h : (g, y) ∈ wirePairs P rs ms r) : recordWires P rs ms r g ≠ none :=
  fun hn => recordWires_eq_none hn (g, y) h rfl

/-- **A resolved port's bits are its slots' claims positions.** -/
theorem portGates_eq {P : Prog} {rs : Array Flock.TemplateQuery.TRange} {ms : Members} {r : Stated}
    {tid port : String} {i k π : ℕ} {ty : Ty} (h : resolve P r tid port = some (i, k, π, ty)) (u a n : ℕ) :
    portGates P rs ms r tid port u a n = (List.range n).map fun β => slotPos P rs ms (i, u, at0 P k π + (a + β)) := by
  obtain ⟨h1, h2, h3⟩ := resolve_some h
  simp only [portGates, h1, h2, h3]
  try rfl

/-! ## The filled slots' gates -/

theorem row_slot {a b c0 cuts rb : ℕ} (h1 : a + c0 * rb ≤ b) (h2 : b < a + (c0 + cuts) * rb) :
    ∃ ρ < cuts, ∃ β < rb, b = a + ((c0 + ρ) * rb + β) := by
  have hrb : 0 < rb := Nat.pos_of_ne_zero fun h0 => by subst h0; omega
  have hq1 : c0 ≤ (b - a) / rb := (Nat.le_div_iff_mul_le hrb).2 (by omega)
  have hq2 : (b - a) / rb < c0 + cuts := (Nat.div_lt_iff_lt_mul hrb).2 (by omega)
  refine ⟨(b - a) / rb - c0, by omega, (b - a) % rb, Nat.mod_lt _ hrb, ?_⟩
  have e : c0 + ((b - a) / rb - c0) = (b - a) / rb := by omega
  rw [e]
  have := Nat.div_add_mod' (b - a) rb
  omega

/-- **A wire's reader index at a unit and a row**: unit `u` of its reader units and row `cut0 + ρ` of its rows. -/
theorem wire_index {w : Wire} {u ρ : ℕ} (h1 : w.start ≤ u) (h2 : u < w.start + w.readerUnits) (hρ : ρ < w.cuts) :
    ρ + (u - w.start) * w.cuts < w.units ∧ w.readerAt (ρ + (u - w.start) * w.cuts) = (u, w.cut0 + ρ) := by
  have hc : 0 < w.cuts := by omega
  unfold SampledProofs.Wires.Wire.readerUnits at h2
  have h3 : (u - w.start + 1) * w.cuts ≤ w.units := (Nat.le_div_iff_mul_le hc).1 (by omega)
  rw [Nat.add_mul, Nat.one_mul] at h3
  refine ⟨by omega, ?_⟩
  show (w.start + (ρ + (u - w.start) * w.cuts) / w.cuts, w.cut0 + (ρ + (u - w.start) * w.cuts) % w.cuts) = _
  rw [Nat.add_mul_div_right _ _ hc, Nat.div_eq_of_lt hρ, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hρ,
    Nat.zero_add, Nat.add_sub_of_le h1]

/-- **A wire's slot is a reader row's bit**: at one of its reader indices, a bit of its row. -/
theorem wire_slot {P : Prog} {r : Stated} {w : Wire} {i k π : ℕ} {ty : Ty} {s : Slot}
    (hres : resolve P r w.reader w.port = some (i, k, π, ty)) (hs : s ∈ rectBox P r (wireRect w)) :
    ∃ ι < w.units, ∃ β < bitsOf ty / r.wiring.cutsOf w.reader w.port,
      s = (i, (w.readerAt ι).1,
        at0 P k π + ((w.readerAt ι).2 * (bitsOf ty / r.wiring.cutsOf w.reader w.port) + β)) := by
  rw [rectBox_some (R := wireRect w) hres, mem_box] at hs
  have hs' : s.1 = i ∧ w.start ≤ s.2.1 ∧ s.2.1 < w.start + w.readerUnits ∧
      at0 P k π + w.cut0 * (bitsOf ty / r.wiring.cutsOf w.reader w.port) ≤ s.2.2 ∧
      s.2.2 < at0 P k π + (w.cut0 + w.cuts) * (bitsOf ty / r.wiring.cutsOf w.reader w.port) := hs
  obtain ⟨hs1, hu1, hu2, hb1, hb2⟩ := hs'
  obtain ⟨ρ, hρ, β, hβ, hb⟩ := row_slot hb1 hb2
  obtain ⟨hι, hrd⟩ := wire_index hu1 hu2 hρ
  refine ⟨ρ + (s.2.1 - w.start) * w.cuts, hι, β, hβ, ?_⟩
  rw [hrd]
  exact Prod.ext hs1 (Prod.ext rfl hb)

/-- **A whole port's slot**: at one of its units, a bit of its port. -/
theorem whole_slot {P : Prog} {r : Stated} {R : Rect} {i k π : ℕ} {ty : Ty} {s : Slot}
    (hres : resolve P r R.reader R.port = some (i, k, π, ty)) (h0 : R.r0 = 0)
    (h1 : R.r1 = r.wiring.cutsOf R.reader R.port) (hs : s ∈ rectBox P r R) :
    ∃ v < R.hi - R.lo, ∃ β < bitsOf ty, s = (i, R.lo + v, at0 P k π + (0 + β)) := by
  rw [rectBox_some hres, mem_box, h0, h1, Nat.zero_mul, Nat.add_zero] at hs
  obtain ⟨hs1, hv1, hv2, hb1, hb2⟩ := hs
  have hle := Nat.mul_div_le (bitsOf ty) (r.wiring.cutsOf R.reader R.port)
  refine ⟨s.2.1 - R.lo, by omega, s.2.2 - at0 P k π, by omega, ?_⟩
  refine Prod.ext hs1 (Prod.ext ?_ ?_)
  · show s.2.1 = R.lo + (s.2.1 - R.lo)
    omega
  · show s.2.2 = at0 P k π + (0 + (s.2.2 - at0 P k π))
    omega

theorem mem_zip_range {α β : Type} (f : ℕ → α) {j n : ℕ} {B : List β} (hj : j < n) (hB : B.length = n) :
    ∃ y, (f j, y) ∈ ((List.range n).map f).zip B := by
  have hl : j < (((List.range n).map f).zip B).length := by
    simp only [List.length_zip, List.length_map, List.length_range, hB, Nat.min_self]
    exact hj
  refine ⟨B[j]'(by omega), ?_⟩
  have h := List.getElem_mem hl
  simp only [List.getElem_zip, List.getElem_map, List.getElem_range] at h
  exact h

theorem outGates_length {P : Prog} {rs : Array Flock.TemplateQuery.TRange} {ms : Members} {r : Stated}
    {tid out : String} {kw : ℕ} (hk : nodeOf P r tid = some kw)
    (hf : (fieldOf (templateDef P kw).ret out).isSome = true) (u n : ℕ) :
    (outGates P rs ms r tid out u n).length = n := by
  obtain ⟨i, hi, _, hik, -⟩ := nodeOf_some hk
  have hj : (templateNodes P).findIdx? (· == kw) = some i := by
    rw [← hik]
    exact findIdx?_getElem (nodup_templateNodes P) hi
  obtain ⟨⟨fty, f0, j⟩, hfj⟩ := Option.isSome_iff_exists.1 hf
  simp only [outGates, hk, hfj, hj, List.length_map, List.length_range]

theorem mem_wirePairs {P : Prog} {rs : Array Flock.TemplateQuery.TRange} {ms : Members} {r : Stated} {w : Wire}
    {ι : ℕ} {q : ℕ × ℕ} (hw : w ∈ r.wiring.wires) (hι : ι < w.units)
    (h : q ∈ (portGates P rs ms r w.reader w.port (w.readerAt ι).1
      ((w.readerAt ι).2 * (portWidth P r w.reader w.port / r.wiring.cutsOf w.reader w.port))
      (portWidth P r w.reader w.port / r.wiring.cutsOf w.reader w.port)).zip
      (outGates P rs ms r w.writer w.out (w.writerAt ι).toNat
        (portWidth P r w.reader w.port / r.wiring.cutsOf w.reader w.port))) :
    q ∈ wirePairs P rs ms r := by
  unfold wirePairs
  exact List.mem_flatMap.2 ⟨w, hw, List.mem_flatMap.2 ⟨ι, List.mem_range.2 hι, h⟩⟩

/-- **A wire's slot is wired**: its claims position is a wire pair's reader, where the leaves pass. -/
theorem wired_of_wire {P : Prog} {rs : Array Flock.TemplateQuery.TRange} {ms : Members} {r : Stated} {w : Wire}
    {s : Slot} (hlv : leavesOk P r = true) (hw : w ∈ r.wiring.wires) (hs : s ∈ rectBox P r (wireRect w)) :
    recordWires P rs ms r (slotPos P rs ms s) ≠ none := by
  cases hres : resolve P r w.reader w.port with
  | none =>
    rw [rectBox_none (R := wireRect w) hres] at hs
    exact absurd hs (Finset.notMem_empty s)
  | some x =>
    obtain ⟨i, k, π, ty⟩ := x
    obtain ⟨ι, hι, β, hβ, rfl⟩ := wire_slot hres hs
    obtain ⟨kw, hkw, hf⟩ := leavesOk_ok hlv hw
    obtain ⟨y, hy⟩ := mem_zip_range (fun β => slotPos P rs ms (i, (w.readerAt ι).1,
      at0 P k π + ((w.readerAt ι).2 * (bitsOf ty / r.wiring.cutsOf w.reader w.port) + β))) hβ
      (outGates_length (rs := rs) (ms := ms) hkw hf (w.writerAt ι).toNat
        (bitsOf ty / r.wiring.cutsOf w.reader w.port))
    refine recordWires_ne_none (y := y) (mem_wirePairs hw hι ?_)
    rw [portWidth_eq hres, portGates_eq hres]
    exact hy

/-- **A held read's slot is registered.** -/
theorem reg_of_read {P : Prog} {rs : Array Flock.TemplateQuery.TRange} {ms : Members} {r : Stated}
    {held : List String} {x : Read} {s : Slot} (hx : x ∈ r.registered.reads) (hh : held.contains x.param = true)
    (hs : s ∈ rectBox P r (wireRect x.pat)) : slotPos P rs ms s ∈ regGates P rs ms r held := by
  cases hres : resolve P r x.pat.reader x.pat.port with
  | none =>
    rw [rectBox_none (R := wireRect x.pat) hres] at hs
    exact absurd hs (Finset.notMem_empty s)
  | some y =>
    obtain ⟨i, k, π, ty⟩ := y
    obtain ⟨ι, hι, β, hβ, rfl⟩ := wire_slot hres hs
    unfold regGates
    try dsimp only
    refine List.mem_toFinset.2 (List.mem_append_left _ (List.mem_append_left _ ?_))
    refine List.mem_flatMap.2 ⟨x, List.mem_filter.2 ⟨hx, hh⟩, List.mem_flatMap.2 ⟨ι, List.mem_range.2 hι, ?_⟩⟩
    try dsimp only
    rw [portWidth_eq hres, portGates_eq hres]
    exact List.mem_map.2 ⟨β, List.mem_range.2 hβ, rfl⟩

/-- The tables over held parameters, as `countsOf` and `regGates` pin them. -/
def pinnedOf (r : Stated) (held : List String) : List String :=
  (r.registered.tables.filter (held.contains ·.param)).map (·.name)

/-- **A pinned table read's slot is registered.** -/
theorem reg_of_root {P : Prog} {rs : Array Flock.TemplateQuery.TRange} {ms : Members} {r : Stated}
    {held : List String} {x : TableRead} {s : Slot} (hx : x ∈ r.registered.roots)
    (hp : (pinnedOf r held).contains x.table = true) (hs : s ∈ rectBox P r (rootRect r.wiring x)) :
    slotPos P rs ms s ∈ regGates P rs ms r held := by
  cases hres : resolve P r x.reader x.port with
  | none =>
    rw [rectBox_none (R := rootRect r.wiring x) hres] at hs
    exact absurd hs (Finset.notMem_empty s)
  | some y =>
    obtain ⟨i, k, π, ty⟩ := y
    obtain ⟨v, hv, β, hβ, rfl⟩ := whole_slot (R := rootRect r.wiring x) hres rfl rfl hs
    have hv' : v < x.units := by
      have : v < x.start + x.units - x.start := hv
      omega
    unfold regGates
    try dsimp only
    refine List.mem_toFinset.2 (List.mem_append_left _ (List.mem_append_right _ ?_))
    refine List.mem_flatMap.2 ⟨x, List.mem_filter.2 ⟨hx, hp⟩, List.mem_flatMap.2 ⟨v, List.mem_range.2 hv', ?_⟩⟩
    try dsimp only
    rw [portWidth_eq hres, portGates_eq hres]
    exact List.mem_map.2 ⟨β, List.mem_range.2 hβ, rfl⟩

/-- An opening's port `p`'s rows: its units, every row of the port (`openRects`'). -/
def openRect (c : Wiring) (o : TableOpening) (p : String) : Rect :=
  ⟨o.reader, p, o.start, o.start + o.units, 0, c.cutsOf o.reader p⟩

/-- **A pinned opening's slot is registered.** -/
theorem reg_of_open {P : Prog} {rs : Array Flock.TemplateQuery.TRange} {ms : Members} {r : Stated}
    {held : List String} {o : TableOpening} {s : Slot} (ho : o ∈ r.registered.openings)
    (hp : (pinnedOf r held).contains o.table = true) {p : String} (hpo : p ∈ o.ports)
    (hs : s ∈ rectBox P r (openRect r.wiring o p)) : slotPos P rs ms s ∈ regGates P rs ms r held := by
  cases hres : resolve P r o.reader p with
  | none =>
    rw [rectBox_none (R := openRect r.wiring o p) hres] at hs
    exact absurd hs (Finset.notMem_empty s)
  | some y =>
    obtain ⟨i, k, π, ty⟩ := y
    obtain ⟨v, hv, β, hβ, rfl⟩ := whole_slot (R := openRect r.wiring o p) hres rfl rfl hs
    have hv' : v < o.units := by
      have : v < o.start + o.units - o.start := hv
      omega
    unfold regGates
    try dsimp only
    refine List.mem_toFinset.2 (List.mem_append_right _ ?_)
    refine List.mem_flatMap.2 ⟨o, List.mem_filter.2 ⟨ho, hp⟩,
      List.mem_flatMap.2 ⟨p, hpo, List.mem_flatMap.2 ⟨v, List.mem_range.2 hv', ?_⟩⟩⟩
    try dsimp only
    rw [portWidth_eq hres, portGates_eq hres]
    exact List.mem_map.2 ⟨β, List.mem_range.2 hβ, rfl⟩

/-- The rows the registered reads of held parameters fill. -/
def readRects (r : Stated) (held : List String) : List Rect :=
  (r.registered.reads.filter (held.contains ·.param)).map fun x => wireRect x.pat

/-- The rows the table reads of tables over held parameters fill. -/
def rootRects (r : Stated) (held : List String) : List Rect :=
  (r.registered.roots.filter ((pinnedOf r held).contains ·.table)).map (rootRect r.wiring)

/-- The rows the openings of tables over held parameters fill. -/
def openingRects (r : Stated) (held : List String) : List Rect :=
  ((r.registered.openings.filter ((pinnedOf r held).contains ·.table)).map (openRects r.wiring)).flatten

/-- The rows the held declarations fill. -/
def heldRects (r : Stated) (held : List String) : List Rect :=
  readRects r held ++ rootRects r held ++ openingRects r held

/-- **The filled rows**: the wires' and the held declarations'. -/
def filled (r : Stated) (held : List String) : List Rect := r.wiring.wires.map wireRect ++ heldRects r held

/-- **A filled slot's gate is wired or registered.** -/
theorem covered {P : Prog} {rs : Array Flock.TemplateQuery.TRange} {ms : Members} {r : Stated} {held : List String}
    {s : Slot} (hlv : leavesOk P r = true) (hs : s ∈ bigU ((filled r held).map (rectBox P r))) :
    recordWires P rs ms r (slotPos P rs ms s) ≠ none ∨ slotPos P rs ms s ∈ regGates P rs ms r held := by
  obtain ⟨A, hAm, hsA⟩ := mem_bigU.1 hs
  obtain ⟨R, hR, rfl⟩ := List.mem_map.1 hAm
  unfold filled heldRects readRects rootRects openingRects at hR
  rcases List.mem_append.1 hR with hR | hR
  · obtain ⟨w, hw, rfl⟩ := List.mem_map.1 hR
    exact .inl (wired_of_wire hlv hw hsA)
  rcases List.mem_append.1 hR with hR | hR
  · rcases List.mem_append.1 hR with hR | hR
    · obtain ⟨x, hx, rfl⟩ := List.mem_map.1 hR
      obtain ⟨hx, hh⟩ := List.mem_filter.1 hx
      exact .inr (reg_of_read hx hh hsA)
    · obtain ⟨x, hx, rfl⟩ := List.mem_map.1 hR
      obtain ⟨hx, hp⟩ := List.mem_filter.1 hx
      exact .inr (reg_of_root hx hp hsA)
  · obtain ⟨Rs, hRs, hR⟩ := List.mem_flatten.1 hR
    obtain ⟨o, ho, rfl⟩ := List.mem_map.1 hRs
    obtain ⟨ho, hp⟩ := List.mem_filter.1 ho
    unfold openRects at hR
    obtain ⟨p, hpo, rfl⟩ := List.mem_map.1 hR
    exact .inr (reg_of_open ho hp hpo hsA)

/-! ## The filled slots are disjoint slots -/

theorem unitsOk_le {c : Wiring} {tid : String} {start n : ℕ} {t : Template} (h : unitsOk c tid start n = true)
    (ht : c.templates.lookup tid = some t) : start + n ≤ t.units := by
  simp only [unitsOk, ht, Bool.and_eq_true, decide_eq_true_eq] at h
  exact h.2

theorem filled_valid {r : Stated} {held : List String} {sm : SampledProofs.Wires.Summary} {rects : List Rect}
    (hw : SampledProofs.Wires.check r.wiring = .ok sm) (hd : declRects r.wiring r.registered = .ok rects) :
    ∀ R ∈ filled r held, Valid r R := by
  obtain ⟨-, hcode, -⟩ := wires_ok hw
  obtain ⟨-, hreads, hroots, hopens⟩ := declRects_ok hd
  intro R hR
  unfold filled heldRects readRects rootRects openingRects at hR
  rcases List.mem_append.1 hR with hR | hR
  · obtain ⟨w, hw', rfl⟩ := List.mem_map.1 hR
    obtain ⟨hcut, hunits⟩ := wireCode_none (hcode w hw')
    exact ⟨fun _ _ _ _ => hcut, hunits⟩
  rcases List.mem_append.1 hR with hR | hR
  · rcases List.mem_append.1 hR with hR | hR
    · obtain ⟨x, hx, rfl⟩ := List.mem_map.1 hR
      have hok := hreads x (List.mem_filter.1 hx).1
      exact ⟨fun t q ht hq => hok.cut t q ht hq, fun t ht => unitsOk_le hok.units ht⟩
    · obtain ⟨x, hx, rfl⟩ := List.mem_map.1 hR
      have hu := hroots x (List.mem_filter.1 hx).1
      exact ⟨fun _ _ _ _ => le_refl _, fun t ht => unitsOk_le hu ht⟩
  · obtain ⟨Rs, hRs, hR⟩ := List.mem_flatten.1 hR
    obtain ⟨o, ho, rfl⟩ := List.mem_map.1 hRs
    have hu := hopens o (List.mem_filter.1 ho).1
    unfold openRects at hR
    obtain ⟨p, -, rfl⟩ := List.mem_map.1 hR
    exact ⟨fun _ _ _ _ => le_refl _, fun t ht => unitsOk_le hu ht⟩

theorem flatten_filter_sublist {α β : Type} (p : α → Bool) (g : α → List β) :
    ∀ l : List α, List.Sublist ((l.filter p).map g).flatten (l.map g).flatten
  | [] => List.Sublist.refl _
  | a :: l => by
    rw [List.filter_cons]
    split
    · simp only [List.map_cons, List.flatten_cons]
      exact (flatten_filter_sublist p g l).append_left _
    · simp only [List.map_cons, List.flatten_cons]
      exact (flatten_filter_sublist p g l).trans (List.sublist_append_right _ _)

theorem filled_pairwise {r : Stated} {held : List String} {sm : SampledProofs.Wires.Summary} {rects : List Rect}
    (hw : SampledProofs.Wires.check r.wiring = .ok sm) (hd : declRects r.wiring r.registered = .ok rects)
    (hap : apart rects (r.wiring.wires.map wireRect) = true) : (filled r held).Pairwise RDisj := by
  obtain ⟨-, -, hpd⟩ := wires_ok hw
  obtain ⟨hrects, -, -, -⟩ := declRects_ok hd
  obtain ⟨hX, hR⟩ := apart_ok rects _ hap
  have hfull := List.pairwise_append.2 ⟨pairwise_wires hpd, hR, fun a ha b hb => rdisj_symm (hX b hb a ha)⟩
  unfold filled
  refine hfull.sublist (List.Sublist.append (List.Sublist.refl _) ?_)
  rw [hrects]
  unfold heldRects readRects rootRects openingRects
  exact ((List.Sublist.map _ List.filter_sublist).append (List.Sublist.map _ List.filter_sublist)).append
    (flatten_filter_sublist _ _ _)

/-! ## The counts -/

/-- Template `tid`'s port `p`'s words, 0 where it has none, as `countsOf` reads them. -/
def wordsAt (c : Wiring) (tid p : String) : ℕ :=
  (((c.templates.lookup tid).bind (·.rows.lookup p)).map (·.words)).getD 0

/-- The slots of each of `L`, summed. -/
def boxCard (P : Prog) (r : Stated) (L : List Rect) : ℕ := ((L.map (rectBox P r)).map Finset.card).sum

/-- `countsOf`'s `inputs`. -/
def inputBits (r : Stated) : ℕ :=
  (r.wiring.templates.map fun x => 16 * (x.2.rows.map (·.2.words)).sum * x.2.units).sum

/-- `countsOf`'s `registered`: its reads' term. -/
def readBits (r : Stated) (held : List String) : ℕ :=
  ((r.registered.reads.filter (held.contains ·.param)).map fun x =>
    x.pat.units * (16 * (wordsAt r.wiring x.pat.reader x.pat.port / r.wiring.cutsOf x.pat.reader x.pat.port))).sum

/-- `countsOf`'s `registered`: its table reads' term. -/
def rootBits (r : Stated) (held : List String) : ℕ :=
  ((r.registered.roots.filter ((pinnedOf r held).contains ·.table)).map fun x =>
    x.units * 16 * wordsAt r.wiring x.reader x.port).sum

/-- `countsOf`'s `registered`: its openings' term. -/
def openBits (r : Stated) (held : List String) : ℕ :=
  ((r.registered.openings.filter ((pinnedOf r held).contains ·.table)).map fun o =>
    (o.ports.map fun p => o.units * 16 * wordsAt r.wiring o.reader p).sum).sum

theorem sum_map_flatten {α : Type} (f : α → ℕ) : ∀ L : List (List α),
    (L.flatten.map f).sum = (L.map fun l => (l.map f).sum).sum
  | [] => rfl
  | l :: L => by
    rw [List.flatten_cons, List.map_append, List.sum_append, sum_map_flatten f L, List.map_cons, List.sum_cons]

theorem wiredBits_map (g : Wire → ℕ) (l : List Wire) :
    SampledProofs.Wires.Summary.wiredBits ⟨l, l.map g⟩ = (l.map fun w => w.units * (16 * g w)).sum := by
  show ((l.zip (l.map g)).map fun x => x.1.units * (16 * x.2)).sum = _
  induction l with
  | nil => rfl
  | cons w l ih => simp only [List.map_cons, List.zip_cons_cons, List.sum_cons, ih]

theorem rowWords_eq (c : Wiring) (w : Wire) :
    SampledProofs.Wires.rowWords c w = wordsAt c w.reader w.port / c.cutsOf w.reader w.port := by
  unfold SampledProofs.Wires.rowWords wordsAt
  cases ht : c.templates.lookup w.reader with
  | none => simp
  | some t =>
    cases hq : t.rows.lookup w.port with
    | none => simp [hq]
    | some q => simp [hq]

theorem sixteen_div_le (b K : ℕ) : 16 * (b / 16 / K) ≤ b / K := by
  rcases Nat.eq_zero_or_pos K with hK | hK
  · subst hK
    simp
  · rw [Nat.div_div_eq_div_mul]
    refine (Nat.le_div_iff_mul_le hK).2 ?_
    have e : 16 * (b / (16 * K)) * K = 16 * K * (b / (16 * K)) := by
      rw [Nat.mul_assoc, Nat.mul_comm (b / (16 * K)) K, ← Nat.mul_assoc]
    rw [e]
    exact Nat.mul_div_le b (16 * K)

theorem wordsAt_eq {c : Wiring} {tid p : String} {t : Template} {q : SampledProofs.Wires.Port}
    (ht : c.templates.lookup tid = some t) (hq : t.rows.lookup p = some q) : wordsAt c tid p = q.words := by
  simp only [wordsAt, ht, Option.bind_some, hq, Option.map_some, Option.getD_some]

theorem wordsAt_none {P : Prog} {r : Stated} {tid p : String} (hA : Aligned P r) (h : resolve P r tid p = none) :
    wordsAt r.wiring tid p = 0 := by
  unfold wordsAt
  cases ht : r.wiring.templates.lookup tid with
  | none => rfl
  | some t =>
    cases hq : t.rows.lookup p with
    | none => simp [hq]
    | some q =>
      obtain ⟨i, k, π, ty, hr, -⟩ := resolve_of_lookup hA ht hq
      rw [h] at hr
      cases hr

theorem wordsAt_resolve {P : Prog} {r : Stated} {tid p : String} {i k π : ℕ} {ty : Ty} (hA : Aligned P r)
    (h : resolve P r tid p = some (i, k, π, ty)) : wordsAt r.wiring tid p = bitsOf ty / 16 := by
  obtain ⟨-, -, -, t, ht, hq, -⟩ := resolve_entry hA h
  exact wordsAt_eq ht hq

theorem card_wireRect {P : Prog} {r : Stated} {w : Wire} {i k π : ℕ} {ty : Ty}
    (h : resolve P r w.reader w.port = some (i, k, π, ty)) (hd : w.units % w.cuts = 0) :
    (rectBox P r (wireRect w)).card = w.units * (bitsOf ty / r.wiring.cutsOf w.reader w.port) := by
  rw [card_rectBox (R := wireRect w) h]
  show (w.start + w.readerUnits - w.start) * ((w.cut0 + w.cuts) * (bitsOf ty / r.wiring.cutsOf w.reader w.port) -
    w.cut0 * (bitsOf ty / r.wiring.cutsOf w.reader w.port)) = _
  rw [Nat.add_sub_cancel_left, Nat.add_mul, Nat.add_sub_cancel_left, ← Nat.mul_assoc]
  unfold SampledProofs.Wires.Wire.readerUnits
  rw [Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hd)]

/-- **A wire's or a held read's count is at most its slots.** -/
theorem cut_card {P : Prog} {r : Stated} {w : Wire} (hA : Aligned P r) (hd : w.units % w.cuts = 0) :
    w.units * (16 * (wordsAt r.wiring w.reader w.port / r.wiring.cutsOf w.reader w.port)) ≤
      (rectBox P r (wireRect w)).card := by
  cases h : resolve P r w.reader w.port with
  | none =>
    rw [wordsAt_none hA h]
    simp
  | some x =>
    obtain ⟨i, k, π, ty⟩ := x
    rw [card_wireRect h hd, wordsAt_resolve hA h]
    exact Nat.mul_le_mul_left _ (sixteen_div_le _ _)

theorem resolve_rowOk {P : Prog} {r : Stated} {tid p : String} {i k π : ℕ} {ty : Ty} (hA : Aligned P r)
    (h : resolve P r tid p = some (i, k, π, ty)) : bitsOf ty % 16 = 0 := by
  obtain ⟨hi, hk, hπ, -⟩ := resolve_entry hA h
  obtain ⟨_, _, -, -, htmpl, -⟩ := hA.entry hi
  obtain ⟨-, hall, -⟩ := templateOf_some htmpl
  have hk' : (templateNodes P)[i] = k := (list_getD_of_lt 0 hi).symm.trans hk
  rw [hk'] at hall
  have h' := Array.all_eq_true'.1 hall (p, ty) (Array.mem_def.2 (List.mem_of_getElem? hπ))
  simp only [rowOk, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at h'
  exact h'.1.2

theorem whole_card {P : Prog} {r : Stated} {R : Rect} (hA : Aligned P r)
    (hcut : SampledProofs.Wires.cutsOk r.wiring = true) (h0 : R.r0 = 0)
    (h1 : R.r1 = r.wiring.cutsOf R.reader R.port) :
    (R.hi - R.lo) * 16 * wordsAt r.wiring R.reader R.port ≤ (rectBox P r R).card := by
  cases h : resolve P r R.reader R.port with
  | none =>
    rw [wordsAt_none hA h]
    simp
  | some x =>
    obtain ⟨i, k, π, ty⟩ := x
    obtain ⟨-, -, -, t, ht, hq, -⟩ := resolve_entry hA h
    obtain ⟨-, hdiv⟩ := cutsOf_ok hcut ht hq
    have hrow := resolve_rowOk hA h
    rw [card_rectBox h, wordsAt_resolve hA h, h0, h1, Nat.zero_mul, Nat.sub_zero]
    have e1 : 16 * (bitsOf ty / 16) = bitsOf ty := Nat.mul_div_cancel' (Nat.dvd_of_mod_eq_zero hrow)
    have e2 : r.wiring.cutsOf R.reader R.port * (bitsOf ty / r.wiring.cutsOf R.reader R.port) = bitsOf ty :=
      Nat.mul_div_cancel' (Nat.dvd_trans (Nat.dvd_of_mod_eq_zero hdiv)
        (Nat.div_dvd_of_dvd (Nat.dvd_of_mod_eq_zero hrow)))
    exact Nat.le_of_eq (by rw [Nat.mul_assoc, e1, e2])

theorem root_card {P : Prog} {r : Stated} (hA : Aligned P r) (hcut : SampledProofs.Wires.cutsOk r.wiring = true)
    (x : TableRead) :
    x.units * 16 * wordsAt r.wiring x.reader x.port ≤ (rectBox P r (rootRect r.wiring x)).card := by
  have h : (x.start + x.units - x.start) * 16 * wordsAt r.wiring x.reader x.port ≤
      (rectBox P r (rootRect r.wiring x)).card := whole_card (R := rootRect r.wiring x) hA hcut rfl rfl
  rwa [Nat.add_sub_cancel_left] at h

theorem open_card {P : Prog} {r : Stated} (hA : Aligned P r) (hcut : SampledProofs.Wires.cutsOk r.wiring = true)
    (o : TableOpening) (p : String) :
    o.units * 16 * wordsAt r.wiring o.reader p ≤ (rectBox P r (openRect r.wiring o p)).card := by
  have h : (o.start + o.units - o.start) * 16 * wordsAt r.wiring o.reader p ≤
      (rectBox P r (openRect r.wiring o p)).card := whole_card (R := openRect r.wiring o p) hA hcut rfl rfl
  rwa [Nat.add_sub_cancel_left] at h

/-! ## The slots and the padding -/

theorem templateOf_rows {d : Flock.Program.Defn} {n : ℕ} {t : Template} (h : templateOf d n = some t) :
    t.rows = d.params.toList.map fun q => (q.1, (⟨bitsOf q.2 / 16, outerOf q.2⟩ : SampledProofs.Wires.Port)) := by
  unfold templateOf at h
  try dsimp only at h
  split at h
  · injection h with h
    subst h
    rfl
  · cases h

theorem at0_length (P : Prog) (k : ℕ) :
    at0 P k (psOf P k).length = 16 * ((templateDef P k).params.toList.map fun q => (bitsOf q.2 + 15) / 16).sum := by
  unfold at0
  rw [List.take_length]
  unfold psOf
  rw [List.map_map]
  try rfl

theorem words_eq {d : Flock.Program.Defn} (hall : d.params.all (rowOk ·.2) = true) :
    (d.params.toList.map fun q => (bitsOf q.2 + 15) / 16) = d.params.toList.map fun q => bitsOf q.2 / 16 := by
  refine List.map_congr_left fun q hq => ?_
  have h := Array.all_eq_true'.1 hall q (Array.mem_def.2 hq)
  simp only [rowOk, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at h
  obtain ⟨⟨-, h16⟩, -⟩ := h
  show (bitsOf q.2 + 15) / 16 = bitsOf q.2 / 16
  omega

/-- **A template's slots are its input bits**: its instances times its parameters' words, 16 bits a word. -/
theorem slot_card {P : Prog} {r : Stated} (hA : Aligned P r) {i : ℕ} (hi : i < r.wiring.templates.length) :
    ((P.defs[P.root]!).body.nodes[(templateNodes P).getD i 0]!).n *
        at0 P ((templateNodes P).getD i 0) (psOf P ((templateNodes P).getD i 0)).length =
      16 * (r.wiring.templates[i].2.rows.map (·.2.words)).sum * r.wiring.templates[i].2.units := by
  have hiT : i < (templateNodes P).length := by rw [← hA.length]; exact hi
  obtain ⟨_, _, -, -, htmpl, hn⟩ := hA.entry hiT
  obtain ⟨hunits, hall, -⟩ := templateOf_some htmpl
  have hrows : r.wiring.templates[i].2.rows.map (·.2.words) =
      (templateDef P (templateNodes P)[i]).params.toList.map fun q => bitsOf q.2 / 16 := by
    rw [templateOf_rows htmpl, List.map_map]
    try rfl
  rw [list_getD_of_lt 0 hiT, at0_length, hrows, hunits, ← hn, words_eq hall]
  exact Nat.mul_comm _ _

/-- **The slots number the record's `inputs`.** -/
theorem card_slots_le {P : Prog} {r : Stated} (hA : Aligned P r) : (slots P).card ≤ inputBits r := by
  unfold slots inputBits
  refine (card_bigU_le _).trans (le_of_eq ?_)
  rw [List.map_map, ← hA.length]
  exact sum_range_eq _ _ _ fun i hi =>
    (card_box _ _ _ _ _).trans (by rw [Nat.sub_zero, Nat.sub_zero]; exact slot_card hA hi)

theorem card_pads_le (P : Prog) (fs : List Flock.Circuit) : (pads P fs).card ≤ padBits P fs := by
  unfold pads padBits
  refine (card_bigU_le _).trans (le_of_eq ?_)
  rw [List.map_map]
  exact congrArg List.sum (List.map_congr_left fun i _ => card_pbox _ _ _)

theorem mem_slots {P : Prog} {i u b : ℕ} (hi : i < (templateNodes P).length)
    (hu : u < ((P.defs[P.root]!).body.nodes[(templateNodes P).getD i 0]!).n)
    (hb : b < at0 P ((templateNodes P).getD i 0) (psOf P ((templateNodes P).getD i 0)).length) :
    ((i, u, b) : Slot) ∈ slots P := by
  unfold slots
  refine mem_bigU.2 ⟨_, List.mem_map.2 ⟨i, List.mem_range.2 hi, rfl⟩, ?_⟩
  exact mem_box.2 ⟨rfl, Nat.zero_le _, hu, Nat.zero_le _, hb⟩

theorem mem_pads {P : Prog} {fs : List Flock.Circuit} {i u col : ℕ} (hi : i < (templateNodes P).length)
    (hu : u < ((P.defs[P.root]!).body.nodes[(templateNodes P).getD i 0]!).n)
    (hc : col ∈ unheld (fs.getD i default)) : ((i, u, col) : Slot) ∈ pads P fs := by
  unfold pads
  refine mem_bigU.2 ⟨_, List.mem_map.2 ⟨i, List.mem_range.2 hi, rfl⟩, ?_⟩
  exact mem_pbox.2 ⟨rfl, hu, hc⟩

theorem root_lt {P : Prog} {k : ℕ} (hk : k < (P.defs[P.root]!).body.nodes.size) : P.root < P.defs.size := by
  refine (Nat.lt_or_ge P.root P.defs.size).elim id fun hn => ?_
  rw [getElem!_neg P.defs P.root (by omega)] at hk
  have h0 : (default : Flock.Program.Defn).body.nodes.size = 0 := rfl
  omega

/-- **A template node's file's Definition** is the node's, where `P`'s Definitions' keys are distinct. -/
theorem templateDef_find {P : Prog} {k : ℕ} {d : Flock.Program.Defn} (hk : (P.defs.toList.map (·.key)).Nodup)
    (hkT : k ∈ templateNodes P) (h : P.defs.find? (·.key == (templateDef P k).key) = some d) :
    d = templateDef P k := by
  have hfn := templateNode_fn hkT
  have e : templateDef P k = P.defs[(Properties.NetworkScan.Fits.node P k).fn] := getElem!_pos P.defs _ hfn
  have hp := Array.find?_some h
  simp only [beq_iff_eq] at hp
  rw [e] at hp ⊢
  exact List.inj_on_of_nodup_map hk (Array.mem_def.1 (Array.mem_of_find?_eq_some h))
    (Array.mem_def.1 (Array.getElem_mem hfn)) hp

theorem at0_full {P : Prog} {k : ℕ} {c : Flock.Circuit} (hF : FileOk c (templateDef P k)) :
    at0 P k (psOf P k).length = 16 * (Flock.portCols c.unit.inGroups).size := by
  rw [at0_length, hF.ports]
  try rfl

/-- **Each input gate is a slot's or a padding column's**: an input gate of the members' joint program is the claims
position of a slot of its member's template at its instance (`slotPos`), or the source gate of an input column no port
holds there (`padPos`). -/
theorem cover {P : Prog} {fs : List Flock.Circuit} {tids : Array String} {rs : Array Flock.TemplateQuery.TRange}
    {ms : Members} {r : Stated} {g : ℕ} (hkeys : (P.defs.toList.map (·.key)).Nodup) (hA : Aligned P r)
    (hF : FilesOf P fs) (hQ : Flock.TemplateQuery.evaluate (.ofProg P) tids = .ok rs) (hM : MembersOf P fs rs ms)
    (hg : (jointOf ms).gates[g]? = some .input) :
    (∃ s ∈ slots P, slotPos P rs ms s = g) ∨ ∃ d ∈ pads P fs, padPos P rs ms d = g := by
  obtain ⟨hglt, -⟩ := List.getElem?_eq_some_iff.1 hg
  obtain ⟨m, hm, x, hx, rfl⟩ := exists_member hglt
  rw [getElem?_jointOf ms hm hx, Option.map_eq_some_iff] at hg
  obtain ⟨G, hG, hGi⟩ := hg
  rw [bgateAt_input] at hGi
  subst hGi
  have hG' : (prog (keyProg (ms.getD m default).1 (ms.getD m default).2.size)
      (consts (ms.getD m default).1 (ms.getD m default).2.size)).1[x]? = some .input := hG
  have hxin := ((prog_input_iff _ _ x).1 hG').1
  obtain ⟨u, hu, col, hcol, b, hb, rfl⟩ := input_src hG'
  obtain ⟨i, hi, hci, hloc⟩ := hM.2 m hm
  have hiT : i < (templateNodes P).length := by rw [← hF.1]; exact hi
  have hkT : (templateNodes P).getD i 0 ∈ templateNodes P := by
    rw [list_getD_of_lt 0 hiT]
    exact List.getElem_mem hiT
  obtain ⟨v, hv⟩ : ∃ v, (ms.getD m default).2.getD u 0 = v := ⟨_, rfl⟩
  have hvmem : v ∈ (ms.getD m default).2.toList := by
    rw [← hv, array_getD_of_lt 0 hu]
    exact Array.getElem_mem_toList hu
  obtain ⟨l, hl, hl1⟩ := Option.map_eq_some_iff.1 (hloc _ hvmem)
  unfold Flock.TemplateQuery.locate at hl
  obtain ⟨R, hf, hRl⟩ := Option.map_eq_some_iff.1 hl
  have hRmem := Array.mem_of_find?_eq_some hf
  have hp := Array.find?_some hf
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hp
  obtain ⟨hp1, hp2⟩ := hp
  have hnode : R.node = (templateNodes P).getD i 0 := by
    subst hRl
    exact hl1
  obtain ⟨root, hroot, hP, hR, -⟩ := evaluate_inv hQ
  rw [def?_ofProg_root hkeys (root_lt (templateNode_lt hkT))] at hroot
  obtain rfl := Option.some.inj hroot
  have hcount : R.count = ((P.defs[P.root]!).body.nodes[(templateNodes P).getD i 0]!).n := by
    rw [(hR R (Array.mem_def.1 hRmem)).2.2, hnode]
    exact countAt_ofProg (templateNode_lt hkT) (batch_of_argsOk (hA.args _ hkT))
  have hT := Properties.SampledProofs.SecurityProofs.evaluate_tiles _ _ _ hQ
  rw [Array.toList_map] at hT
  have hN : (rs.toList.map (·.node)).Nodup := List.Pairwise.imp (fun h => Nat.ne_of_lt h) hP
  have htt : v - R.first < R.count := by omega
  have hq : ∀ w, ((Flock.TemplateQuery.locate rs w).any fun l => l.1 == (templateNodes P).getD i 0 &&
      l.2.1 == v - R.first) = true ↔ w = v := by
    intro w
    have h := locate_any_iff hT hN hRmem htt w
    rw [hnode, show R.first + (v - R.first) = v by omega] at h
    exact h
  have hvpop : v < Flock.TemplateQuery.population rs := by
    have := (tiles_mem _ _ _ hT R (Array.mem_def.1 hRmem)).2
    omega
  have hnd := hM.1.nodup_iff.2 List.nodup_range
  have hinst : ∀ pos : Flock.Circuit → ℕ → ℕ → Option ℕ, instPos P rs ms i (v - R.first) pos =
      ((pos (ms.getD m default).1 (ms.getD m default).2.size u).map (base ms m + ·)).getD
        (jointOf ms).gates.length := by
    intro pos
    obtain ⟨m', hm', u', hu', he, hpos⟩ := instPos_eq (pos := pos) hM.1 hvpop hq
    obtain ⟨rfl, rfl⟩ := unit_unique hnd hm' hm hu' hu (he.trans hv.symm)
    exact hpos
  have hFk : FileOk (ms.getD m default).1 (templateDef P ((templateNodes P).getD i 0)) := by
    obtain ⟨d, hd, hFd⟩ := fileOk (hF.2 i hi)
    rw [hci, ← templateDef_find hkeys hkT hd]
    exact hFd
  rcases col_slot hFk hu hcol hb with ⟨p, hp', t, ht, hin⟩ | hpad
  · refine .inl ⟨(i, v - R.first, 16 * p + t), mem_slots hiT (lt_of_lt_of_eq htt hcount) ?_, ?_⟩
    · rw [at0_full hFk]
      omega
    · show instPos P rs ms i (v - R.first)
        (fun c' n' u' => inBit c' n' u' 0 ((16 * p + t) / 16) ((16 * p + t) % 16)) = _
      rw [hinst, show (16 * p + t) / 16 = p by omega, show (16 * p + t) % 16 = t by omega]
      simp only [inBit, hin, Option.map_some, Option.getD_some]
      rw [posOf_input _ hxin]
  · refine .inr ⟨(i, v - R.first, col), mem_pads hiT (lt_of_lt_of_eq htt hcount) (by rw [← hci]; exact hpad), ?_⟩
    show instPos P rs ms i (v - R.first) (fun c' _ u' => (keyIdx c' (srcs c') col).map (2 + u' * srcs c' + ·)) = _
    rw [hinst]
    simp only [hb, Option.map_some, Option.getD_some]

/-! ## The bound -/

/-- **The count's bound** at a record `checked` accepts: the members' joint program's free inputs outside the
registered rows number at most the record's `free` at the held parameters and the padding's bits, where `P`'s
Definitions' keys are distinct. -/
theorem counts_le_of {P : Prog} {record order : Json} {r : Stated} {sm : SampledProofs.Wires.Summary}
    {held : List String} {fs : List Flock.Circuit} {tids : Array String} {rs : Array Flock.TemplateQuery.TRange}
    {ms : Members} {L S : ℕ} {lay : NetTiming.BoundaryGates L S} {k : ℕ}
    (hkeys : (P.defs.toList.map (·.key)).Nodup) (hck : checked P record order = .ok (r, sm)) (hF : FilesOf P fs)
    (hQ : Flock.TemplateQuery.evaluate (.ofProg P) tids = .ok rs) (hM : MembersOf P fs rs ms)
    (hlay : lay.wires = recordWires P rs ms r) :
    (NetTiming.freeInputs (jointOf ms) lay k \ regGates P rs ms r held).card ≤
      (countsOf r sm held).free + padBits P fs := by
  obtain ⟨hrec, hpop, hw, hlv, rects, hd, hap⟩ := checked_ok hck
  have hA := aligned hrec hpop
  obtain ⟨hcut, -, -⟩ := wires_ok hw
  have hdivs := SampledProofs.Wires.check_wires hw
  have hsm := SampledProofs.Wires.check_summary hw
  obtain ⟨-, hreads, -, -⟩ := declRects_ok hd
  have hval := filled_valid (held := held) hw hd
  have hpw := filled_pairwise (held := held) hw hd hap
  have hdisj : ((filled r held).map (rectBox P r)).Pairwise Disjoint :=
    List.pairwise_map.2 (hpw.imp_of_mem fun {a b} ha hb hab => rectBox_disjoint hA (hval a ha) (hval b hb) hab)
  have hCS : bigU ((filled r held).map (rectBox P r)) ⊆ slots P := bigU_subset fun A hA' => by
    obtain ⟨R, hR, rfl⟩ := List.mem_map.1 hA'
    exact rectBox_subset hA (hval R hR)
  have hcov : ∀ y ∈ NetTiming.freeInputs (jointOf ms) lay k \ regGates P rs ms r held,
      (∃ s ∈ slots P, s ∉ bigU ((filled r held).map (rectBox P r)) ∧ slotPos P rs ms s = y) ∨
        ∃ d ∈ pads P fs, padPos P rs ms d = y := by
    intro y hy
    obtain ⟨hyF, hyR⟩ := Finset.mem_sdiff.1 hy
    obtain ⟨-, hyin, -, hyw⟩ := Finset.mem_filter.1 hyF
    rcases cover hkeys hA hF hQ hM hyin with ⟨s, hs, hsy⟩ | ⟨d, hd', hdy⟩
    · refine .inl ⟨s, hs, fun hC => ?_, hsy⟩
      rcases covered (rs := rs) (ms := ms) hlv hC with h | h
      · apply h
        rw [hsy, ← hlay]
        exact hyw
      · exact hyR (by rw [← hsy]; exact h)
    · exact .inr ⟨d, hd', hdy⟩
  have hX := card_le_of_cover (slotPos P rs ms) (padPos P rs ms) hCS hcov
  have hS := card_slots_le hA
  have hD := card_pads_le P fs
  have hC : (bigU ((filled r held).map (rectBox P r))).card = boxCard P r (r.wiring.wires.map wireRect) +
      boxCard P r (readRects r held) + boxCard P r (rootRects r held) + boxCard P r (openingRects r held) := by
    rw [card_bigU hdisj]
    simp only [filled, heldRects, boxCard, List.map_append, List.sum_append]
    omega
  have hWb : sm.wiredBits ≤ boxCard P r (r.wiring.wires.map wireRect) := by
    rw [hsm, wiredBits_map]
    unfold boxCard
    rw [List.map_map, List.map_map]
    refine List.sum_le_sum fun w hw' => ?_
    show w.units * (16 * SampledProofs.Wires.rowWords r.wiring w) ≤ (rectBox P r (wireRect w)).card
    rw [rowWords_eq]
    exact cut_card hA (hdivs w hw').2
  have hRd : readBits r held ≤ boxCard P r (readRects r held) := by
    unfold readBits boxCard readRects
    rw [List.map_map, List.map_map]
    refine List.sum_le_sum fun x hx => ?_
    show x.pat.units * (16 * (wordsAt r.wiring x.pat.reader x.pat.port / r.wiring.cutsOf x.pat.reader x.pat.port)) ≤
      (rectBox P r (wireRect x.pat)).card
    exact cut_card hA (hreads x (List.mem_filter.1 hx).1).div
  have hRt : rootBits r held ≤ boxCard P r (rootRects r held) := by
    unfold rootBits boxCard rootRects
    rw [List.map_map, List.map_map]
    exact List.sum_le_sum fun x _ => root_card hA hcut x
  have hOp : openBits r held ≤ boxCard P r (openingRects r held) := by
    unfold openBits boxCard openingRects
    rw [List.map_map, sum_map_flatten, List.map_map]
    refine List.sum_le_sum fun o _ => ?_
    show (o.ports.map fun p => o.units * 16 * wordsAt r.wiring o.reader p).sum ≤
      ((openRects r.wiring o).map fun R => (rectBox P r R).card).sum
    unfold openRects
    rw [List.map_map]
    exact List.sum_le_sum fun p _ => open_card hA hcut o p
  have hfree : (countsOf r sm held).free =
      inputBits r - sm.wiredBits - (readBits r held + rootBits r held + openBits r held) := rfl
  rw [hfree]
  omega

/-- **An accepted record's free count and its padding bound its circuit's free inputs**: `counts_le_of` at the count
`NetworkCheck.Circuit.check` returns, the record `recordOf` reads and the parameters `heldOf` reads off the pins. -/
theorem counts_le {P : Prog} {record order pins : Json} {c : Counts} {r : Stated} {held : List String}
    {fs : List Flock.Circuit} {tids : Array String} {rs : Array Flock.TemplateQuery.TRange} {ms : Members} {L S : ℕ}
    {lay : NetTiming.BoundaryGates L S} {k : ℕ} (hkeys : (P.defs.toList.map (·.key)).Nodup)
    (hc : NetworkCheck.Circuit.check P record order pins = .ok c) (hrec : recordOf P record = .ok r)
    (hheld : heldOf pins = .ok held) (hF : FilesOf P fs) (hQ : Flock.TemplateQuery.evaluate (.ofProg P) tids = .ok rs)
    (hM : MembersOf P fs rs ms) (hlay : lay.wires = recordWires P rs ms r) :
    (NetTiming.freeInputs (jointOf ms) lay k \ regGates P rs ms r held).card ≤ c.free + padBits P fs := by
  obtain ⟨r', sm, held', hck, hheld', hc'⟩ := check_ok hc
  have er : r' = r := Except.ok.inj ((checked_ok hck).1.symm.trans hrec)
  have eh : held' = held := Except.ok.inj (hheld'.symm.trans hheld)
  rw [er] at hck
  rw [hc', er, eh]
  exact counts_le_of hkeys hck hF hQ hM hlay

end Properties.NetworkCertifier.Counts
