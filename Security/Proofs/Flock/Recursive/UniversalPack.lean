import Proofs.Flock.Recursive.UniversalLower

/-!
# Recursion: a circuit type packed onto 16-bit leaf ports, and the universal unit's packed type

A statement's unit has 16-bit leaf ports: `HmRow.check`, run on every circuit file, refuses any other. So a unit whose ports
are bit vectors is staged packed. `class_statement.class_lowering` lays each row port's bits contiguously into 16-bit
words, padded to whole 64-word SHA-512 blocks, and each output word's bits into the low bits of its own 16-bit words, the
rest zero; `typed_statement.unit_type` of that lowering is the type. This file states that packing for every type, so it
carries over to any unit class, and proves what the packed type computes.

* **The packing** (`pack P T`): `P` gives the input runs and the output words, each `(bits, words)`. Every wire moves by
  one map, `segPos P.ins`: input bit `i` to its run's first word times 16 plus its place in the run, and AND wire `N + k`
  to `16·words + k`. Each output form is mapped the same way, and padded with zero forms to its words (`padL`).
* **Re-indexing** (`flatOk_pack`, `eval_pack`, `evalT_pack`): when the runs fit their words, `segPos` is strictly
  monotone, so the packed type is flat whenever `T` is, and on inputs `x` it computes `T`'s outputs at `x ∘ segPos`, laid
  into the output words with zeros.
* **The unit's rows** (`pack_unit_sound`): when `deriveChecked` accepts a statement's layouts and its unit's type is
  `pack P T`, the unit's rows as the block holds them carry on output `o`'s packed column `T`'s output `o` on the packed
  input columns (`Dag.unit_sound`, `Dag.evalT_flat`).
* **The universal unit** (`uuTypeW p`, `evalT_uuTypeW`, `uuW_unit_sound`): `uuType p` packed as `rec_inner.program_unit` stages it in
  argument order (6692c8703): the description's `width` bits one row, `x` the next, each `⌈bits/16⌉` words padded to
  whole blocks (`blockWords`), and each output its own word. Its unit rows carry `Desc.eval` of description bit `j` at
  input column `j` on data bit `a` at input column `16·blockWords width + a`, at output column `16·i`.
* **The tie to Python** (the `#guard`s at the end): the packed type's digest is the one `typed_statement.unit_type` gives
  for that class lowering (`class_statement.class_lowering(…, build=universal_stage.build_of(G, N_IN, N_OUT))`), at the
  six classes `UniversalLower` pins `uuType` at.
-/

namespace FlockSoundness.Discharge.Recursion.UniversalPack

open Flock Flock.Derive Audit FlockSoundness.Types FlockSoundness.Discharge.Recursion.UniversalClass
  FlockSoundness.Discharge.Recursion.UniversalLower

/-! ## Runs of bits in words -/

/-- Runs of bits, each `(bits, words)`: `bits` laid into `words` 16-bit words. -/
abbrev Segs := List (ℕ × ℕ)

/-- The runs' bits. -/
def bitsOf (s : Segs) : ℕ := (s.map Prod.fst).sum

/-- The runs' words. -/
def wordsOf (s : Segs) : ℕ := (s.map Prod.snd).sum

/-- Every run fits its words. -/
def Fits (s : Segs) : Prop := ∀ q ∈ s, q.1 ≤ 16 * q.2

@[simp] theorem bitsOf_nil : bitsOf [] = 0 := rfl
@[simp] theorem wordsOf_nil : wordsOf [] = 0 := rfl
@[simp] theorem bitsOf_cons (b w : ℕ) (s : Segs) : bitsOf ((b, w) :: s) = b + bitsOf s := by simp [bitsOf]
@[simp] theorem wordsOf_cons (b w : ℕ) (s : Segs) : wordsOf ((b, w) :: s) = w + wordsOf s := by simp [wordsOf]

theorem Fits.head {b w : ℕ} {s : Segs} (h : Fits ((b, w) :: s)) : b ≤ 16 * w := h (b, w) (List.mem_cons_self ..)

theorem Fits.tail {q : ℕ × ℕ} {s : Segs} (h : Fits (q :: s)) : Fits s := fun r hr => h r (List.mem_cons_of_mem _ hr)

/-- **Bit `i`'s place** once each run is laid into its words; past the runs, shifted by the padding. -/
def segPos : Segs → ℕ → ℕ
  | [], i => i
  | (b, w) :: rest, i => if i < b then i else 16 * w + segPos rest (i - b)

theorem segPos_ge : ∀ (s : Segs) {i : ℕ}, bitsOf s ≤ i → segPos s i = 16 * wordsOf s + (i - bitsOf s)
  | [], i, _ => by simp [segPos]
  | (b, w) :: rest, i, h => by
    simp only [bitsOf_cons, wordsOf_cons] at h ⊢
    simp only [segPos, show ¬ i < b by omega, ↓reduceIte]
    rw [segPos_ge rest (by omega)]
    omega

theorem segPos_lt : ∀ (s : Segs), Fits s → ∀ {i : ℕ}, i < bitsOf s → segPos s i < 16 * wordsOf s
  | [], _, i, h => by simp at h
  | (b, w) :: rest, hf, i, h => by
    have hb := hf.head
    simp only [bitsOf_cons, wordsOf_cons] at h ⊢
    rw [segPos]
    split_ifs with hi
    · omega
    · have := segPos_lt rest hf.tail (i := i - b) (by omega)
      omega

theorem segPos_strictMono : ∀ (s : Segs), Fits s → StrictMono (segPos s)
  | [], _ => fun _ _ h => h
  | (b, w) :: rest, hf => by
    have hb := hf.head
    have ih := segPos_strictMono rest hf.tail
    intro i j hij
    simp only [segPos]
    split_ifs with hi hj hj
    · exact hij
    · omega
    · omega
    · have := ih (show i - b < j - b by omega)
      omega

/-- **A list cut into runs**, each run's entries and then `z` up to its `16 · words` places. -/
def padL {α : Type} (z : α) : Segs → List α → List α
  | [], _ => []
  | (b, w) :: rest, l => l.take b ++ List.replicate (16 * w - b) z ++ padL z rest (l.drop b)

theorem length_padL {α : Type} (z : α) : ∀ (s : Segs) (l : List α), Fits s → bitsOf s ≤ l.length →
    (padL z s l).length = 16 * wordsOf s
  | [], _, _, _ => by simp [padL]
  | (b, w) :: rest, l, hf, hl => by
    have hb := hf.head
    simp only [bitsOf_cons, wordsOf_cons] at hl ⊢
    simp only [padL, List.length_append, List.length_take, List.length_replicate]
    rw [length_padL z rest (l.drop b) hf.tail (by simp only [List.length_drop]; omega)]
    omega

theorem map_padL {α β : Type} (f : α → β) (z : α) : ∀ (s : Segs) (l : List α),
    (padL z s l).map f = padL (f z) s (l.map f)
  | [], _ => rfl
  | (b, w) :: rest, l => by
    simp only [padL, List.map_append, List.map_take, List.map_replicate, List.map_drop, map_padL f z rest]

theorem mem_padL {α : Type} {z x : α} : ∀ {s : Segs} {l : List α}, x ∈ padL z s l → x = z ∨ x ∈ l
  | [], _, h => by simp [padL] at h
  | (b, w) :: rest, l, h => by
    simp only [padL, List.mem_append] at h
    rcases h with (h | h) | h
    · exact Or.inr (List.mem_of_mem_take h)
    · exact Or.inl (List.eq_of_mem_replicate h)
    · rcases mem_padL h with h | h
      · exact Or.inl h
      · exact Or.inr (List.mem_of_mem_drop h)

/-- **Entry `o` sits at `segPos o`** once the list is cut into fitting runs. -/
theorem getD_padL {α : Type} (z d : α) : ∀ (s : Segs) (l : List α), Fits s → bitsOf s ≤ l.length →
    ∀ {o : ℕ}, o < bitsOf s → (padL z s l).getD (segPos s o) d = l.getD o d
  | [], _, _, _, o, h => by simp at h
  | (b, w) :: rest, l, hf, hl, o, ho => by
    have hb := hf.head
    simp only [bitsOf_cons] at hl ho
    simp only [padL, segPos]
    split_ifs with h
    · rw [List.getD_eq_getElem?_getD, List.append_assoc, List.getElem?_append_left (by simp; omega),
        List.getElem?_take_of_lt h, ← List.getD_eq_getElem?_getD]
    · have hpre : (l.take b ++ List.replicate (16 * w - b) z).length = 16 * w := by simp; omega
      rw [List.getD_eq_getElem?_getD, List.getElem?_append_right (by omega), hpre, Nat.add_sub_cancel_left,
        ← List.getD_eq_getElem?_getD, getD_padL z d rest (l.drop b) hf.tail (by simp; omega) (by omega),
        List.getD_eq_getElem?_getD, List.getElem?_drop, Nat.add_sub_cancel' (by omega), ← List.getD_eq_getElem?_getD]

/-! ## The packing -/

/-- A form with each wire renamed by `φ`. -/
def mapForm (φ : ℕ → ℕ) (f : Form) : Form := ⟨f.ws.map φ, f.c⟩

/-- An item with each form's wires renamed by `φ`. -/
def mapItem (φ : ℕ → ℕ) : Item → Item
  | .and a b => .and (mapForm φ a) (mapForm φ b)
  | .call k fs => .call k (fs.map (mapForm φ))
  | .read t fs => .read t (fs.map (mapForm φ))
  | .bad op => .bad op

/-- The constant 0 (`gf2.ZERO`), a padding output. -/
def zeroForm : Form := ⟨#[], 0⟩

/-- **A packing onto 16-bit leaves**: the input runs (one a row port) and the output words (one an output word). -/
structure Packing where
  ins : Segs
  outs : Segs

/-- **`P` packs `T`**: its runs cover `T`'s input and output bits, each fitting its words. -/
structure Packing.Packs (P : Packing) (T : CircuitType) : Prop where
  ins_fit : Fits P.ins
  outs_fit : Fits P.outs
  ins_bits : bitsOf P.ins = T.inBits
  outs_bits : bitsOf P.outs = T.outBits

/-- `n` ports of one 16-bit word each. -/
def wordPorts (n : ℕ) : Array (ℕ × ℕ) := (List.replicate n (1, 16)).toArray

/-- **`T` packed by `P`**: its ports 16-bit words (`P`'s), its items and outputs over the wires renamed by `segPos P.ins`,
and each output word padded with zero forms. -/
def pack (P : Packing) (T : CircuitType) : CircuitType where
  label := T.label
  ins := wordPorts (wordsOf P.ins)
  outs := wordPorts (wordsOf P.outs)
  calls := T.calls
  tables := T.tables
  items := T.items.map (mapItem (segPos P.ins))
  outputs := (padL zeroForm P.outs (T.outputs.toList.map (mapForm (segPos P.ins)))).toArray

theorem portBits_wordPorts (n : ℕ) : CircuitType.portBits (wordPorts n) = 16 * n := by
  unfold CircuitType.portBits wordPorts
  simp only [List.foldl_toArray']
  suffices ∀ a, List.foldl (fun a (p : ℕ × ℕ) => a + p.1 * p.2) a (List.replicate n ((1 : ℕ), (16 : ℕ))) = a + 16 * n by
    simpa using this 0
  induction n with
  | zero => simp
  | succ n ih => intro a; rw [List.replicate_succ, List.foldl_cons, ih]; ring

theorem pack_inBits (P : Packing) (T : CircuitType) : (pack P T).inBits = 16 * wordsOf P.ins :=
  portBits_wordPorts _

theorem pack_outBits (P : Packing) (T : CircuitType) : (pack P T).outBits = 16 * wordsOf P.outs :=
  portBits_wordPorts _

section Packed

variable {P : Packing} {T : CircuitType}

/-- `segPos` sends every wire below `T.inBits + k` below the packed type's `inBits + k`. -/
theorem segPos_wire (hP : P.Packs T) {k x : ℕ} (hx : x < T.inBits + k) :
    segPos P.ins x < 16 * wordsOf P.ins + k := by
  by_cases hi : x < T.inBits
  · have := segPos_lt P.ins hP.ins_fit (i := x) (by rw [hP.ins_bits]; exact hi)
    omega
  · rw [segPos_ge P.ins (by rw [hP.ins_bits]; omega), hP.ins_bits]
    omega

theorem segPos_and (hP : P.Packs T) (k : ℕ) : segPos P.ins (T.inBits + k) = 16 * wordsOf P.ins + k := by
  rw [segPos_ge P.ins (by rw [hP.ins_bits]; omega), hP.ins_bits]
  omega

theorem formOk_mapForm {φ : ℕ → ℕ} (hφ : StrictMono φ) {n n' : ℕ} (hn : ∀ x < n, φ x < n') {f : Form}
    (h : CircuitType.formOk n f = true) : CircuitType.formOk n' (mapForm φ f) = true := by
  rw [show mapForm φ f = ⟨f.ws.map φ, f.c⟩ from rfl]
  unfold CircuitType.formOk at h ⊢
  simp only [Bool.and_eq_true, decide_eq_true_eq, Array.all_eq_true, List.all_eq_true, List.mem_range] at h ⊢
  obtain ⟨⟨hc, hw⟩, ha⟩ := h
  refine ⟨⟨hc, fun i hi => ?_⟩, fun i hi => ?_⟩
  · simp only [Array.getElem_map]
    exact hn _ (hw i (by simpa using hi))
  · simp only [Array.size_map] at hi
    have h1 : i < f.ws.size := by omega
    have h2 : i + 1 < f.ws.size := by omega
    have := ha i hi
    rw [getElem!_pos f.ws i h1, getElem!_pos f.ws (i + 1) h2] at this
    rw [getElem!_pos (f.ws.map φ) i (by simpa using h1), getElem!_pos (f.ws.map φ) (i + 1) (by simpa using h2),
      Array.getElem_map, Array.getElem_map]
    exact hφ this

theorem formOk_zeroForm (n : ℕ) : CircuitType.formOk n zeroForm = true := by
  simp [CircuitType.formOk, zeroForm]

theorem pack_outputs_size (hP : P.Packs T) (hT : flatOk T = true) :
    (pack P T).outputs.size = 16 * wordsOf P.outs := by
  simp only [pack, List.size_toArray]
  exact length_padL _ _ _ hP.outs_fit (by simp [hP.outs_bits, (flatOk_outputs hT).1])

/-- **The packed type is flat** whenever `T` is. -/
theorem flatOk_pack (hP : P.Packs T) (hT : flatOk T = true) : flatOk (pack P T) = true := by
  have hφ := segPos_strictMono P.ins hP.ins_fit
  have hT' := hT
  unfold flatOk at hT' ⊢
  simp only [Bool.and_eq_true, List.all_eq_true, List.mem_range, beq_iff_eq] at hT' ⊢
  obtain ⟨⟨hI, -⟩, hO⟩ := hT'
  rw [pack_inBits]
  refine ⟨⟨fun i hi => ?_, ?_⟩, ?_⟩
  · have hi' : i < T.items.size := by simpa [pack] using hi
    have := hI i hi'
    rw [Array.getElem?_eq_getElem hi'] at this
    have he : (pack P T).items[i]? = some (mapItem (segPos P.ins) T.items[i]) := by
      simp [pack, Array.getElem?_map, Array.getElem?_eq_getElem hi']
    rw [he]
    revert this
    cases T.items[i] with
    | and a b =>
      simp only [mapItem, Bool.and_eq_true]
      exact fun ⟨ha, hb⟩ => ⟨formOk_mapForm hφ (fun x hx => segPos_wire hP hx) ha,
        formOk_mapForm hφ (fun x hx => segPos_wire hP hx) hb⟩
    | call _ _ => simp
    | read _ _ => simp
    | bad _ => simp
  · rw [pack_outputs_size hP hT, pack_outBits]
  · have hsz : (pack P T).items.size = T.items.size := by simp [pack]
    rw [hsz, Array.all_eq_true']
    intro f hf
    simp only [pack, List.mem_toArray] at hf
    rcases mem_padL hf with rfl | hf
    · exact formOk_zeroForm _
    · obtain ⟨g, hg, rfl⟩ := List.mem_map.1 hf
      exact formOk_mapForm hφ (fun x hx => segPos_wire hP hx) (Array.all_eq_true'.mp hO g (by simpa using hg))

theorem formVal_mapForm (φ : ℕ → ℕ) (v : ℕ → Bool) (f : Form) : formVal v (mapForm φ f) = formVal (v ∘ φ) f := by
  unfold formVal mapForm xorSum
  simp [Array.toList_map, List.map_map]

theorem pack_items_getElem (k : ℕ) (hk : k < T.items.size) :
    (pack P T).items.toList[k]'(by simpa [pack] using hk) = mapItem (segPos P.ins) T.items[k] := by
  simp [pack]

/-- **What the packed type computes**: `T`'s outputs on the inputs at `segPos`, laid into the output words with zeros. -/
theorem eval_pack (hP : P.Packs T) (hT : flatOk T = true) (x : ℕ → Bool) :
    eval (pack P T) x = padL false P.outs (eval T (x ∘ segPos P.ins)) := by
  have hflat := flatOk_pack hP hT
  set V := run (x ∘ segPos P.ins) T.inBits T.items.toList with hV
  set V' := run x (pack P T).inBits (pack P T).items.toList with hV'
  have hw : ∀ w, w < T.inBits + T.items.size → V' (segPos P.ins w) = V w := by
    intro w
    induction w using Nat.strong_induction_on with
    | _ w ih =>
      intro hwlt
      by_cases hwi : w < T.inBits
      · have hlt := segPos_lt P.ins hP.ins_fit (i := w) (by rw [hP.ins_bits]; exact hwi)
        rw [hV', run_lt _ _ _ _ (by rw [pack_inBits]; exact hlt), hV, run_lt _ _ _ _ hwi]
        rfl
      · obtain ⟨k, rfl⟩ : ∃ k, w = T.inBits + k := ⟨w - T.inBits, by omega⟩
        have hk : k < T.items.size := by omega
        obtain ⟨a, b, hab, ha, hb⟩ := flatOk_items hT k hk
        rw [segPos_and hP k, ← pack_inBits, hV', run_item _ _ _ (flatOk_reads hflat) k (by simpa [pack] using hk),
          pack_items_getElem k hk, ← hV', hV, run_item _ _ _ (flatOk_reads hT) k (by simpa using hk), ← hV]
        simp only [Array.getElem_toList, hab, mapItem, itemVal, formVal_mapForm]
        rw [formVal_congr (v := V' ∘ segPos P.ins) (v' := V)
            (fun y hy => ih y (by have := ha y hy; omega) (by have := ha y hy; omega)),
          formVal_congr (v := V' ∘ segPos P.ins) (v' := V)
            (fun y hy => ih y (by have := hb y hy; omega) (by have := hb y hy; omega))]
  unfold eval
  rw [← hV', show (pack P T).outputs.toList = padL zeroForm P.outs (T.outputs.toList.map (mapForm (segPos P.ins))) from
    by simp [pack], map_padL, List.map_map]
  have hz : formVal V' zeroForm = false := by simp [formVal, zeroForm, xorSum]
  rw [hz]
  congr 1
  refine List.map_congr_left fun f hf => ?_
  rw [Function.comp, formVal_mapForm]
  obtain ⟨o, ho, rfl⟩ := List.getElem_of_mem hf
  exact formVal_congr fun y hy =>
    hw y ((flatOk_outputs hT).2 o (by simpa using ho) y (by simpa using hy))

theorem length_eval (T : CircuitType) (x : ℕ → Bool) : (eval T x).length = T.outputs.size := by
  simp [eval]

/-- **`pack P T` computes `T` re-indexed**, at any fuel: its output `segPos P.outs o` on `x` is `T`'s output `o` on
`x ∘ segPos P.ins`. -/
theorem evalT_pack (types : String → Option CircuitType) (words : String → Option (Array ℕ)) (hP : P.Packs T)
    (hT : flatOk T = true) (F : ℕ) (x : ℕ → Bool) {o : ℕ} (ho : o < T.outputs.size) :
    (Dag.evalT types words (F + 1) (pack P T) x).getD (segPos P.outs o) false =
      (eval T (x ∘ segPos P.ins)).getD o false := by
  have hob : o < bitsOf P.outs := by rw [hP.outs_bits, ← (flatOk_outputs hT).1]; exact ho
  rw [Dag.evalT_flat types words (flatOk_pack hP hT) F, eval_pack hP hT]
  exact getD_padL _ _ _ _ hP.outs_fit (le_of_eq (by rw [length_eval, hP.outs_bits, (flatOk_outputs hT).1])) hob

/-- **The packed unit's rows compute `T`.** When `deriveChecked` accepts a statement's layouts and its unit's type is
`pack P T`, column values that satisfy the unit's rows as the block holds them (Δ applied), with the block's `one` at 1,
carry on output `o`'s packed column `T`'s output `o` on the packed input columns. -/
theorem pack_unit_sound {types : String → Option CircuitType} {ls : Array Flock.Layout}
    {info : String → Option TableInfo} {words : String → Option (Array ℕ)} {unit : ℕ} {done : Array DeriveAll.D}
    (hd : DeriveAll.deriveChecked types ls info words unit = .ok done) (hP : P.Packs T) (hT : flatOk T = true)
    (ht : (ls.getD unit default).type.bind types = some (pack P T)) (one : ℕ) (z : ℕ → Bool)
    (hz : Dag.SatAt (DeriveAll.blockRow words (done.getD unit default) one) z) (h1 : z one = true) {o : ℕ}
    (ho : o < T.outputs.size) :
    z ((done.getD unit default).outCols.getD (segPos P.outs o) 0) =
      (eval T fun i => z ((done.getD unit default).inCols.getD (segPos P.ins i) 0)).getD o false := by
  have hob : o < bitsOf P.outs := by rw [hP.outs_bits, ← (flatOk_outputs hT).1]; exact ho
  have hs : segPos P.outs o < (pack P T).outputs.size := by
    rw [pack_outputs_size hP hT]; exact segPos_lt P.outs hP.outs_fit hob
  rw [Dag.unit_sound hd ht one z hz h1 (unit + 1) (Nat.lt_succ_self unit) _ hs]
  exact evalT_pack types words hP hT unit _ ho

end Packed

/-! ## The universal unit, packed -/

/-- `⌈b/16⌉` words, padded to whole 64-word SHA-512 blocks, at least one (`class_statement._pad(_words(b), 64)`). -/
def blockWords (b : ℕ) : ℕ := max 64 (((b + 15) / 16 + 63) / 64 * 64)

theorem le_blockWords (b : ℕ) : b ≤ 16 * blockWords b := by
  unfold blockWords
  have := le_max_right 64 (((b + 15) / 16 + 63) / 64 * 64)
  omega

variable (p : Params)

/-- **`rec_inner.program_unit`'s packing in argument order**: the description's bits one row, `x`'s the next, each in
whole blocks, and each output its own word. -/
def uuPacking : Packing where
  ins := [(p.width, blockWords p.width), (p.nIn, blockWords p.nIn)]
  outs := List.replicate p.nOut (1, 1)

/-- **`UniversalUnit_v1{G, N_IN, N_OUT}` on 16-bit leaves**: `uuType p` packed as the class is staged. -/
def uuTypeW : CircuitType := pack (uuPacking p) (uuType p)

theorem bitsOf_replicate (n b w : ℕ) : bitsOf (List.replicate n (b, w)) = n * b := by
  induction n with
  | zero => simp
  | succ n ih => rw [List.replicate_succ, bitsOf_cons, ih]; ring

theorem uuPacking_packs : (uuPacking p).Packs (uuType p) where
  ins_fit := by
    intro q hq
    simp only [uuPacking, List.mem_cons, List.not_mem_nil, or_false] at hq
    rcases hq with rfl | rfl <;> exact le_blockWords _
  outs_fit := by
    intro q hq
    simp only [uuPacking] at hq
    rw [List.eq_of_mem_replicate hq]
    omega
  ins_bits := by
    rw [uuType_inBits]
    simp [uuPacking, nIns]
  outs_bits := by
    rw [uuPacking, bitsOf_replicate]
    simp [uuType, CircuitType.outBits, CircuitType.portBits]

theorem segPos_desc {j : ℕ} (hj : j < p.width) : segPos (uuPacking p).ins j = j := by
  simp [uuPacking, segPos, hj]

theorem segPos_data {a : ℕ} (ha : a < p.nIn) :
    segPos (uuPacking p).ins (p.width + a) = 16 * blockWords p.width + a := by
  simp [uuPacking, segPos, ha]

theorem segPos_replicate : ∀ (n : ℕ) {i : ℕ}, i < n → segPos (List.replicate n (1, 1)) i = 16 * i
  | 0, i, hi => absurd hi (Nat.not_lt_zero i)
  | n + 1, i, hi => by
    rw [List.replicate_succ, segPos]
    split_ifs with h
    · omega
    · rw [segPos_replicate n (by omega)]
      omega

theorem segPos_out {i : ℕ} (hi : i < p.nOut) : segPos (uuPacking p).outs i = 16 * i := segPos_replicate p.nOut hi

theorem wordsOf_replicate (n b w : ℕ) : wordsOf (List.replicate n (b, w)) = n * w := by
  induction n with
  | zero => simp
  | succ n ih => rw [List.replicate_succ, wordsOf_cons, ih]; ring

theorem size_outputs_uuTypeW : (uuTypeW p).outputs.size = 16 * p.nOut := by
  rw [uuTypeW, pack_outputs_size (uuPacking_packs p) (flatOk_uuType (p := p))]
  simp [uuPacking, wordsOf_replicate]

theorem eval_uuType_packed (x : ℕ → Bool) (i : Fin p.nOut) :
    (eval (uuType p) (x ∘ segPos (uuPacking p).ins)).getD i false =
      Desc.eval (fun j : Fin p.width => x j) (fun a => x (16 * blockWords p.width + a)) i := by
  rw [eval_uuType, getD_ofFn, dite_eq_left i.isLt]
  congr 1
  · funext j; simp [descOf, segPos_desc p j.isLt]
  · funext a; simp [dataOf, segPos_data p a.isLt]

/-- **`uuTypeW p` computes the class's interpretation**, at any fuel: its output word `i` on `x` is `Desc.eval` of
description bit `j` at input `j` on data bit `a` at input `16·blockWords width + a`. -/
theorem evalT_uuTypeW (types : String → Option CircuitType) (words : String → Option (Array ℕ)) (F : ℕ)
    (x : ℕ → Bool) (i : Fin p.nOut) :
    (Dag.evalT types words (F + 1) (uuTypeW p) x).getD (16 * i.val) false =
      Desc.eval (fun j : Fin p.width => x j) (fun a => x (16 * blockWords p.width + a)) i := by
  rw [← segPos_out p i.isLt]
  exact (evalT_pack types words (uuPacking_packs p) (flatOk_uuType (p := p)) F x
    (by rw [size_outputs]; exact i.isLt)).trans (eval_uuType_packed p x i)

/-- **On a statement's unit rows, packed.** When `deriveChecked` accepts a statement's layouts and its unit's type is
`uuTypeW p`, column values that satisfy the unit's rows as the block holds them (Δ applied), with the block's `one` at 1,
carry on output word `i`'s column `Desc.eval` of description bit `j` at input column `j` on data bit `a` at input column
`16·blockWords width + a`. -/
theorem uuW_unit_sound {types : String → Option CircuitType} {ls : Array Flock.Layout}
    {info : String → Option TableInfo} {words : String → Option (Array ℕ)} {unit : ℕ} {done : Array DeriveAll.D}
    (hd : DeriveAll.deriveChecked types ls info words unit = .ok done)
    (ht : (ls.getD unit default).type.bind types = some (uuTypeW p)) (one : ℕ) (z : ℕ → Bool)
    (hz : Dag.SatAt (DeriveAll.blockRow words (done.getD unit default) one) z) (h1 : z one = true) (i : Fin p.nOut) :
    z ((done.getD unit default).outCols.getD (16 * i.val) 0) =
      Desc.eval (fun j : Fin p.width => z ((done.getD unit default).inCols.getD j 0))
        (fun a => z ((done.getD unit default).inCols.getD (16 * blockWords p.width + a) 0)) i := by
  have h := pack_unit_sound hd (uuPacking_packs p) (flatOk_uuType (p := p)) ht one z hz h1 (o := i.val)
    (by rw [size_outputs]; exact i.isLt)
  rw [segPos_out p i.isLt] at h
  exact h.trans (eval_uuType_packed p (fun j => z ((done.getD unit default).inCols.getD j 0)) i)

/-! ## The tie to Python

`typed_statement.unit_type(class_statement.class_lowering(act, uc, name, groups, True,
build=universal_stage.build_of(G, N_IN, N_OUT)))` over `rec_inner.program_unit(bind(UniversalUnit, G, N_IN, N_OUT))`'s
one class in argument order (`test_universal_stage`'s lowering at 6692c8703), its digest: 128 input ports and `N_OUT`
output ports of one 16-bit word, `and_count` ANDs, and 15 zero forms an output word. -/

#guard ([⟨1, 1, 1⟩, ⟨2, 2, 1⟩, ⟨3, 2, 2⟩, ⟨5, 3, 4⟩, ⟨7, 9, 3⟩, ⟨16, 16, 8⟩] : List Params).all (accepted ∘ uuTypeW)

#guard digestIs (uuTypeW ⟨1, 1, 1⟩)
  "e3033e964a195631320d7d1051152555ecd6aa337aa58adf42c16bbc47b361568f95d2fc45556f6c85fa1d0cb52257153d0870ca1d3785d3f363f7ead0d96814"
#guard digestIs (uuTypeW ⟨2, 2, 1⟩)
  "979dcd86e37ad023fa204c7dbedcc1f2befcee77535dd388a758dd8faf001bd4b42cf19a4f7b8d4f1f666ef692aed925df9cc5840fa40cad36b100bff55d5df1"
#guard digestIs (uuTypeW ⟨3, 2, 2⟩)
  "f88391cc30ac7b495f033cd03d140c08540272d8664ff535018d24e9ddf33b5856afa5e1076535f0c02ea0e6c557e63ef53b6ca94c074cff44e9c63c6c424283"
#guard digestIs (uuTypeW ⟨5, 3, 4⟩)
  "ba481e89f757d37aaab93de7d7c46d4e901484ba5fc325b893383aa315404989355011253262be37bfbdb22b9e50cbaac1ec2e937de6e8d0c26856093c3306c6"
#guard digestIs (uuTypeW ⟨7, 9, 3⟩)
  "dc9b1037bfba7828898705988fe08b5fd2a1d8120e9d465891c07bd83e79b88d6d657584403120d4dec10091ad05e93abd72b1dda71b3584f9fb8db8e22038c2"
#guard digestIs (uuTypeW ⟨16, 16, 8⟩)
  "5bf9542cacbedbe06b1c25082e9b87de6a59193100c16c98ded6ea370b4585a0e49f3266e51f21c904c754e2a0ea4021d7edd02cf5b961cc7e05139fbd6b8a54"

end FlockSoundness.Discharge.Recursion.UniversalPack
