import Proofs.Flock.Soundness.Types.Dag

/-!
# A type DAG's Boolean circuit (phase 1 S4a: `flatten`, `flatten_eval`)

`flatten` lays a circuit type out as a Boolean circuit, its calls inlined to a depth: an array of gates (`input`, `and`,
`xor`, `not`), each over earlier gates by index, evaluated in order (`evalG`, `gval`):
* gates `0 … inBits − 1` are the type's inputs, gate `inBits` the input `one` (fed 1), gate `inBits + 1` its NOT, `zero`;
* a form is its wires' XOR chain, with a NOT when its constant is 1 (`formGate`): the Boolean export's convention;
* an AND item is an AND gate over its forms' gates; a call inlines its callee on its input forms' gates (at the depth
  `evalT` runs it); a read is `table/v2`'s gates (`readGates`): its index forms' gates split at `loOf n` into two
  decoders (`decodeGates`, one-hot), and each value bit the XOR of its products, a high minterm AND the XOR of the low
  minterms whose word has the bit set (`bitsGates`), every value bit gated.

**`flatten_eval`**: on inputs `x`, output `o`'s gate carries `evalT`'s output `o`, for every type, table and depth. So the
type semantics the rows' theorems are stated in (`Types.Dag.evalT`) is the Boolean circuit's.

`GateRows.Gate` is RoPE's earlier gate model; this one keeps the constants as gates so that a callee's gates embed
unchanged.
-/

namespace FlockSoundness.Types.Flatten

open Flock Flock.DeriveAll Audit Types Types.Dag

/-! ## Gates -/

/-- A gate over earlier gates, by index; an input gate reads the input at its own index. -/
inductive Gate where
  | input
  | and (a b : Nat)
  | xor (a b : Nat)
  | not (a : Nat)
  deriving Inhabited

/-- A gate's value, after the gates before it took the values `vals`. -/
def Gate.step (inp : Nat → Bool) (vals : Array Bool) : Gate → Bool
  | .input => inp vals.size
  | .and a b => vals.getD a false && vals.getD b false
  | .xor a b => Bool.xor (vals.getD a false) (vals.getD b false)
  | .not a => !vals.getD a false

/-- Every gate's value, in order. -/
def evalG (inp : Nat → Bool) (gs : Array Gate) : Array Bool :=
  gs.foldl (fun vals g => vals.push (g.step inp vals)) #[]

theorem evalG_push (inp : Nat → Bool) (gs : Array Gate) (g : Gate) :
    evalG inp (gs.push g) = (evalG inp gs).push (g.step inp (evalG inp gs)) := by
  simp [evalG, Array.foldl_push]

theorem foldl_step_size (inp : Nat → Bool) : ∀ (l : List Gate) (acc : Array Bool),
    (l.foldl (fun vals g => vals.push (g.step inp vals)) acc).size = acc.size + l.length
  | [], acc => rfl
  | g :: l, acc => by
    rw [List.foldl_cons, foldl_step_size inp l, Array.size_push, List.length_cons]
    omega

theorem evalG_size (inp : Nat → Bool) (gs : Array Gate) : (evalG inp gs).size = gs.size := by
  unfold evalG
  rw [← Array.foldl_toList, foldl_step_size]
  simp

/-- The value of gate `i`. -/
def gval (inp : Nat → Bool) (gs : Array Gate) (i : Nat) : Bool := (evalG inp gs).getD i false

theorem gval_push_lt (inp : Nat → Bool) (gs : Array Gate) (g : Gate) {i : Nat} (hi : i < gs.size) :
    gval inp (gs.push g) i = gval inp gs i := by
  unfold gval
  rw [evalG_push]
  have : i < (evalG inp gs).size := by rw [evalG_size]; exact hi
  simp [Array.getD, this, Nat.lt_succ_of_lt this, Array.getElem_push_lt this]

theorem gval_push_new (inp : Nat → Bool) (gs : Array Gate) (g : Gate) :
    gval inp (gs.push g) gs.size = g.step inp (evalG inp gs) := by
  unfold gval
  rw [evalG_push, ← evalG_size inp gs]
  simp [Array.getD]

/-- The gates a gate reads. -/
def Gate.args : Gate → List Nat
  | .input => []
  | .and a b => [a, b]
  | .xor a b => [a, b]
  | .not a => [a]

theorem foldl_push_size : ∀ (more : List Gate) (gs : Array Gate), (more.foldl Array.push gs).size = gs.size + more.length
  | [], gs => rfl
  | g :: more, gs => by rw [List.foldl_cons, foldl_push_size more, Array.size_push, List.length_cons]; omega

/-- `gs'` extends `gs`: the same gates first, so the same values there, then gates that each read only gates before
them. -/
def Ext (gs gs' : Array Gate) : Prop :=
  ∃ more : List Gate, gs' = more.foldl Array.push gs ∧ ∀ k (hk : k < more.length), ∀ a ∈ more[k].args, a < gs.size + k

theorem Ext.refl (gs : Array Gate) : Ext gs gs := ⟨[], rfl, fun k hk => absurd hk (Nat.not_lt_zero _)⟩

theorem Ext.push {gs gs' : Array Gate} (h : Ext gs gs') (g : Gate) (hg : ∀ a ∈ g.args, a < gs'.size) :
    Ext gs (gs'.push g) := by
  obtain ⟨more, rfl, hm⟩ := h
  refine ⟨more ++ [g], by simp [List.foldl_append], fun k hk a ha => ?_⟩
  by_cases hk' : k < more.length
  · rw [List.getElem_append_left hk'] at ha
    exact hm k hk' a ha
  · have : k = more.length := by simp at hk; omega
    subst this
    simp only [List.getElem_append_right (Nat.le_refl _), Nat.sub_self, List.getElem_cons_zero] at ha
    have := hg a ha
    rwa [foldl_push_size] at this

theorem Ext.trans {a b c : Array Gate} (h1 : Ext a b) (h2 : Ext b c) : Ext a c := by
  obtain ⟨m1, rfl, hm1⟩ := h1
  obtain ⟨m2, rfl, hm2⟩ := h2
  refine ⟨m1 ++ m2, by simp [List.foldl_append], fun k hk x hx => ?_⟩
  by_cases hk' : k < m1.length
  · rw [List.getElem_append_left hk'] at hx
    exact hm1 k hk' x hx
  · rw [List.getElem_append_right (by omega)] at hx
    have := hm2 (k - m1.length) (by simp at hk; omega) x hx
    rw [foldl_push_size] at this
    omega

theorem Ext.size_le {gs gs' : Array Gate} (h : Ext gs gs') : gs.size ≤ gs'.size := by
  obtain ⟨more, rfl, -⟩ := h
  rw [foldl_push_size]; omega

theorem Ext.gval {gs gs' : Array Gate} (h : Ext gs gs') (inp : Nat → Bool) {i : Nat} (hi : i < gs.size) :
    gval inp gs' i = gval inp gs i := by
  obtain ⟨more, rfl, -⟩ := h
  induction more generalizing gs with
  | nil => rfl
  | cons g more ih =>
    simp only [List.foldl_cons]
    rw [ih (gs := gs.push g) (by simp; omega), gval_push_lt inp gs g hi]

theorem args_and {a b n : Nat} (ha : a < n) (hb : b < n) : ∀ x ∈ (Gate.and a b).args, x < n := by
  intro x hx
  simp only [Gate.args, List.mem_cons, List.not_mem_nil, or_false] at hx
  rcases hx with rfl | rfl <;> assumption

/-- Every gate reads only gates before it. -/
def Topo (gs : Array Gate) : Prop := ∀ i (hi : i < gs.size), ∀ a ∈ gs[i].args, a < i

theorem Topo.ext {gs gs' : Array Gate} (h : Topo gs) (he : Ext gs gs') : Topo gs' := by
  obtain ⟨more, rfl, hm⟩ := he
  induction more generalizing gs with
  | nil => exact h
  | cons g more ih =>
    simp only [List.foldl_cons]
    refine ih (gs := gs.push g) ?_ (fun k hk a ha => ?_)
    · intro i hi a ha
      by_cases hi' : i < gs.size
      · rw [Array.getElem_push_lt hi'] at ha; exact h i hi' a ha
      · have : i = gs.size := by simp at hi; omega
        subst this
        rw [Array.getElem_push_eq] at ha
        simpa using hm 0 (by simp) a (by simpa using ha)
    · have := hm (k + 1) (by simp; omega) a (by simpa using ha)
      simp; omega

/-! ## Forms as gates -/

/-- An XOR chain from gate `acc` over gates `rest`. -/
def xorChain : Array Gate → Nat → List Nat → Array Gate × Nat
  | gs, acc, [] => (gs, acc)
  | gs, acc, g :: rest => xorChain (gs.push (.xor acc g)) gs.size rest

/-- A form over the wires' gates `wg`, as a gate: its wires' XOR chain, with a NOT when its constant is 1. A form of no
wires is gate `one` or gate `zero`; a wire past `wg` reads `zero`. -/
def formGate (one zero : Nat) (wg : Array Nat) (gs : Array Gate) (f : Form) : Array Gate × Nat :=
  match f.ws.toList.map (wg.getD · zero) with
  | [] => if f.c = 1 then (gs, one) else (gs, zero)
  | g :: rest =>
    let r := xorChain gs g rest
    if f.c = 1 then (r.1.push (.not r.2), r.1.size) else r

theorem xorChain_spec (inp : Nat → Bool) :
    ∀ (rest : List Nat) (gs : Array Gate) (acc : Nat), acc < gs.size → (∀ g ∈ rest, g < gs.size) →
      Ext gs (xorChain gs acc rest).1 ∧ (xorChain gs acc rest).2 < (xorChain gs acc rest).1.size ∧
        gval inp (xorChain gs acc rest).1 (xorChain gs acc rest).2 =
          Bool.xor (gval inp gs acc) (xorSum rest (gval inp gs))
  | [], gs, acc, hacc, _ => ⟨Ext.refl gs, hacc, by simp [xorChain, xorSum]⟩
  | g :: rest, gs, acc, hacc, hrest => by
    have hg : g < gs.size := hrest g (List.mem_cons_self ..)
    obtain ⟨h1, h2, h3⟩ := xorChain_spec inp rest (gs.push (.xor acc g)) gs.size (by simp)
      (fun g' hg' => by have := hrest g' (List.mem_cons_of_mem _ hg'); simp; omega)
    refine ⟨(Ext.refl gs).push _ (by simp [Gate.args]; omega) |>.trans h1, h2, ?_⟩
    simp only [xorChain]
    rw [h3, gval_push_new, xorSum_cons]
    simp only [Gate.step]
    have e1 : (evalG inp gs).getD acc false = gval inp gs acc := rfl
    have e2 : (evalG inp gs).getD g false = gval inp gs g := rfl
    rw [e1, e2, xorSum_congr (fun g' hg' => gval_push_lt inp gs _ (hrest g' (List.mem_cons_of_mem _ hg'))),
      Bool.xor_assoc]

/-- The wires' gates carry the wire values: wire `w`'s gate (`zero` past the wires) is a gate whose value is wire `w`'s,
gate `one` is 1 and gate `zero` is 0. -/
structure WInv (inp : Nat → Bool) (one zero : Nat) (gs : Array Gate) (wg : Array Nat) (vals : List Bool) : Prop where
  size : wg.size = vals.length
  lt : ∀ w, wg.getD w zero < gs.size
  val : ∀ w, gval inp gs (wg.getD w zero) = vals.getD w false
  one_lt : one < gs.size
  one_val : gval inp gs one = true
  zero_val : gval inp gs zero = false

theorem WInv.zero_lt {inp : Nat → Bool} {one zero : Nat} {gs : Array Gate} {wg : Array Nat} {vals : List Bool}
    (h : WInv inp one zero gs wg vals) : zero < gs.size := by
  have := h.lt wg.size
  simpa [Array.getD] using this

theorem WInv.ext {inp : Nat → Bool} {one zero : Nat} {gs gs' : Array Gate} {wg : Array Nat} {vals : List Bool}
    (h : WInv inp one zero gs wg vals) (he : Ext gs gs') : WInv inp one zero gs' wg vals where
  size := h.size
  lt w := Nat.lt_of_lt_of_le (h.lt w) he.size_le
  val w := by rw [he.gval inp (h.lt w), h.val w]
  one_lt := Nat.lt_of_lt_of_le h.one_lt he.size_le
  one_val := by rw [he.gval inp h.one_lt, h.one_val]
  zero_val := by rw [he.gval inp h.zero_lt, h.zero_val]

theorem formGate_spec (inp : Nat → Bool) {one zero : Nat} {wg : Array Nat} {gs : Array Gate} {vals : List Bool}
    (hI : WInv inp one zero gs wg vals) (f : Form) :
    Ext gs (formGate one zero wg gs f).1 ∧ (formGate one zero wg gs f).2 < (formGate one zero wg gs f).1.size ∧
      gval inp (formGate one zero wg gs f).1 (formGate one zero wg gs f).2 = formVal (vals.getD · false) f := by
  have hsum : xorSum (f.ws.toList.map (wg.getD · zero)) (gval inp gs) = xorSum f.ws.toList (vals.getD · false) := by
    rw [xorSum_map]
    exact xorSum_congr fun w _ => hI.val w
  have hlt : ∀ g ∈ f.ws.toList.map (wg.getD · zero), g < gs.size := by
    intro g hg
    obtain ⟨w, -, rfl⟩ := List.mem_map.mp hg
    exact hI.lt w
  unfold formGate
  generalize f.ws.toList.map (wg.getD · zero) = L at hsum hlt ⊢
  cases L with
  | nil =>
    have hv : formVal (vals.getD · false) f = decide (f.c = 1) := by
      unfold formVal; rw [← hsum]; simp [xorSum]
    rw [hv]
    by_cases hc : f.c = 1
    · simp only [hc, ↓reduceIte, decide_true]
      exact ⟨Ext.refl gs, hI.one_lt, hI.one_val⟩
    · simp only [hc, ↓reduceIte, decide_false]
      exact ⟨Ext.refl gs, hI.zero_lt, hI.zero_val⟩
  | cons g rest =>
    obtain ⟨e1, e2, e3⟩ := xorChain_spec inp rest gs g (hlt g (List.mem_cons_self ..))
      (fun g' hg' => hlt g' (List.mem_cons_of_mem _ hg'))
    have hv : Bool.xor (gval inp gs g) (xorSum rest (gval inp gs)) = xorSum f.ws.toList (vals.getD · false) := by
      rw [← hsum, xorSum_cons]
    by_cases hc : f.c = 1
    · simp only [hc, ↓reduceIte]
      refine ⟨e1.push _ (by simp [Gate.args]; omega), by simp, ?_⟩
      rw [gval_push_new]
      show (!gval inp (xorChain gs g rest).1 (xorChain gs g rest).2) = formVal (vals.getD · false) f
      rw [e3, hv]
      unfold formVal
      simp [hc]
    · simp only [hc, ↓reduceIte]
      refine ⟨e1, e2, ?_⟩
      rw [e3, hv]
      unfold formVal
      simp [hc]

/-- The gates of forms `fs`, pushed onto `acc`. -/
def formsGates (one zero : Nat) (wg : Array Nat) : Array Gate → List Form → Array Nat → Array Gate × Array Nat
  | gs, [], acc => (gs, acc)
  | gs, f :: rest, acc =>
    let r := formGate one zero wg gs f
    formsGates one zero wg r.1 rest (acc.push r.2)

theorem formsGates_spec (inp : Nat → Bool) {one zero : Nat} {wg : Array Nat} {vals : List Bool} :
    ∀ (fs : List Form) (gs : Array Gate) (acc : Array Nat), WInv inp one zero gs wg vals → (∀ a ∈ acc, a < gs.size) →
      Ext gs (formsGates one zero wg gs fs acc).1 ∧ (formsGates one zero wg gs fs acc).2.size = acc.size + fs.length ∧
        (∀ m < acc.size, (formsGates one zero wg gs fs acc).2.getD m zero = acc.getD m zero) ∧
        (∀ a ∈ (formsGates one zero wg gs fs acc).2, a < (formsGates one zero wg gs fs acc).1.size) ∧
        ∀ m < fs.length, gval inp (formsGates one zero wg gs fs acc).1
          ((formsGates one zero wg gs fs acc).2.getD (acc.size + m) zero) = formVal (vals.getD · false) (fs.getD m default)
  | [], gs, acc, _, hacc => ⟨Ext.refl gs, by simp [formsGates], fun _ _ => rfl, hacc, fun m hm => absurd hm (by simp)⟩
  | f :: rest, gs, acc, hI, hacc => by
    obtain ⟨g1, g2, g3⟩ := formGate_spec inp hI f
    obtain ⟨h1, h2, h3, h4, h5⟩ := formsGates_spec inp rest _ (acc.push (formGate one zero wg gs f).2) (hI.ext g1)
      (fun a ha => by
        rcases Array.mem_push.mp ha with ha | rfl
        · exact Nat.lt_of_lt_of_le (hacc a ha) g1.size_le
        · exact g2)
    simp only [formsGates]
    refine ⟨g1.trans h1, by rw [h2]; simp; omega, fun m hm => ?_, h4, fun m hm => ?_⟩
    · rw [h3 m (by simp; omega)]
      simp [Array.getD, hm, Nat.lt_succ_of_lt hm, Array.getElem_push_lt hm]
    · cases m with
      | zero =>
        rw [Nat.add_zero, h3 acc.size (by simp)]
        have hx : (acc.push (formGate one zero wg gs f).2).getD acc.size zero = (formGate one zero wg gs f).2 := by
          simp [Array.getD]
        rw [hx, h1.gval inp g2, g3]
        rfl
      | succ m =>
        have := h5 m (by simp at hm; omega)
        rw [show acc.size + (m + 1) = (acc.push (formGate one zero wg gs f).2).size + m by simp; omega, this]
        rfl

/-! ## Reads as gates: the decoders -/

/-- Minterms `i, i + 1, …`: minterm `i` is `lo[i mod 2^h] AND hi[i / 2^h]`. -/
def minterms (lo hi : Array Nat) (h : Nat) : Array Gate → Nat → Nat → Array Nat → Array Gate × Array Nat
  | gs, 0, _, acc => (gs, acc)
  | gs, cnt + 1, i, acc =>
    minterms lo hi h (gs.push (.and (lo.getD (i % 2 ^ h) 0) (hi.getD (i / 2 ^ h) 0))) cnt (i + 1) (acc.push gs.size)

/-- One-hot minterms of the leaf gates `bs` (least significant first), as `table/v2`'s decoders lay them out: no leaf
gives gate `one`; one leaf its NOT and itself; more the halves' minterms combined (`minterms`), `h` half the leaves. -/
def decodeGates (one : Nat) (gs : Array Gate) (bs : Array Nat) : Array Gate × Array Nat :=
  if _hs : bs.size ≤ 1 then
    if bs.size = 0 then (gs, #[one]) else (gs.push (.not (bs.getD 0 0)), #[gs.size, bs.getD 0 0])
  else
    let h := bs.size / 2
    let rl := decodeGates one gs (bs.extract 0 h)
    let rh := decodeGates one rl.1 (bs.extract h bs.size)
    minterms rl.2 rh.2 h rh.1 (2 ^ bs.size) 0 #[]
termination_by bs.size
decreasing_by all_goals simp only [Array.size_extract]; omega

theorem minterms_spec (inp : Nat → Bool) (lo hi : Array Nat) (h : Nat) :
    ∀ cnt (gs : Array Gate) i acc, (∀ a, lo.getD a 0 < gs.size) → (∀ a, hi.getD a 0 < gs.size) →
      (∀ a ∈ acc, a < gs.size) →
      Ext gs (minterms lo hi h gs cnt i acc).1 ∧ (minterms lo hi h gs cnt i acc).2.size = acc.size + cnt ∧
        (∀ m < acc.size, (minterms lo hi h gs cnt i acc).2.getD m 0 = acc.getD m 0) ∧
        (∀ a ∈ (minterms lo hi h gs cnt i acc).2, a < (minterms lo hi h gs cnt i acc).1.size) ∧
        ∀ m < cnt, gval inp (minterms lo hi h gs cnt i acc).1 ((minterms lo hi h gs cnt i acc).2.getD (acc.size + m) 0) =
          (gval inp gs (lo.getD ((i + m) % 2 ^ h) 0) && gval inp gs (hi.getD ((i + m) / 2 ^ h) 0))
  | 0, gs, i, acc, _, _, hacc => ⟨Ext.refl gs, by simp [minterms], fun _ _ => rfl, hacc, fun m hm => absurd hm (by omega)⟩
  | cnt + 1, gs, i, acc, hlo, hhi, hacc => by
    obtain ⟨h1, h2, h3, h4, h5⟩ := minterms_spec inp lo hi h cnt
      (gs.push (.and (lo.getD (i % 2 ^ h) 0) (hi.getD (i / 2 ^ h) 0))) (i + 1) (acc.push gs.size)
      (fun a => Nat.lt_of_lt_of_le (hlo a) (by simp)) (fun a => Nat.lt_of_lt_of_le (hhi a) (by simp))
      (fun a ha => by
        rcases Array.mem_push.mp ha with ha | rfl
        · exact Nat.lt_of_lt_of_le (hacc a ha) (by simp)
        · simp)
    simp only [minterms]
    refine ⟨(Ext.refl gs).push _ (args_and (hlo _) (hhi _)) |>.trans h1, by rw [h2]; simp; omega,
      fun m hm => ?_, h4, fun m hm => ?_⟩
    · rw [h3 m (by simp; omega)]
      simp [Array.getD, hm, Nat.lt_succ_of_lt hm, Array.getElem_push_lt hm]
    · cases m with
      | zero =>
        simp only [Nat.add_zero]
        rw [h3 acc.size (by simp)]
        have hx : (acc.push gs.size).getD acc.size 0 = gs.size := by simp [Array.getD]
        rw [hx, h1.gval inp (by simp), gval_push_new]
        rfl
      | succ m =>
        have := h5 m (by omega)
        rw [show acc.size + (m + 1) = (acc.push gs.size).size + m by simp; omega, this,
          gval_push_lt inp gs _ (hlo _), gval_push_lt inp gs _ (hhi _), show i + 1 + m = i + (m + 1) by omega]

theorem toList_split_nat (a : Array Nat) (h : Nat) :
    a.toList = (a.extract 0 h).toList ++ (a.extract h a.size).toList := by
  simp only [Array.toList_extract, List.extract, Nat.sub_zero, List.drop_zero]
  rw [List.take_of_length_le (l := List.drop h a.toList) (by simp)]
  exact (List.take_append_drop h a.toList).symm

theorem getD_extract_nat (a : Array Nat) (s e i : Nat) (hi : i < e - s) (he : e ≤ a.size) :
    (a.extract s e).getD i 0 = a.getD (s + i) 0 := by
  have h2 : s + i < a.size := by omega
  have h3 : i < min e a.size - s := by omega
  simp [Array.getD, h2, h3, Array.getElem_extract]

theorem getD_extract_lt (a : Array Nat) (s e : Nat) (he : e ≤ a.size) {n : Nat} (h : ∀ x, a.getD x 0 < n) :
    ∀ x, (a.extract s e).getD x 0 < n := by
  intro x
  by_cases hx : x < e - s
  · rw [getD_extract_nat a s e x hx he]; exact h _
  · have : ¬ x < min e a.size - s := by omega
    have h0 := h a.size
    simp only [Array.getD, Nat.lt_irrefl, dite_false] at h0
    simp [Array.getD, this, h0]

theorem getD_pair0 (a b d : Nat) : (#[a, b] : Array Nat).getD 0 d = a := rfl
theorem getD_pair1 (a b d : Nat) : (#[a, b] : Array Nat).getD 1 d = b := rfl

/-- **The decoders' minterms are one-hot** at the index their leaf gates spell. -/
theorem decodeGates_spec (inp : Nat → Bool) (one : Nat) :
    ∀ (n : Nat) (bs : Array Nat) (gs : Array Gate), bs.size = n → (∀ a, bs.getD a 0 < gs.size) → one < gs.size →
      gval inp gs one = true →
      Ext gs (decodeGates one gs bs).1 ∧ (decodeGates one gs bs).2.size = 2 ^ bs.size ∧
        (∀ a, (decodeGates one gs bs).2.getD a 0 < (decodeGates one gs bs).1.size) ∧
        ∀ i < 2 ^ bs.size, gval inp (decodeGates one gs bs).1 ((decodeGates one gs bs).2.getD i 0) =
          decide (bitsIdx (bs.toList.map (gval inp gs)) = i) := by
  intro n
  induction n using Nat.strongRecOn with
  | ind n ih =>
  intro bs gs hn hbs hone h1
  have h0 : 0 < gs.size := by
    have := hbs bs.size
    simpa [Array.getD] using this
  unfold decodeGates
  split
  · rename_i hs
    split
    · rename_i hz
      have hl : bs.toList = [] := by simpa using hz
      refine ⟨Ext.refl gs, by simp [hz], fun a => ?_, fun i hi => ?_⟩
      · rcases a with _ | a
        · simpa [Array.getD] using hone
        · simpa [Array.getD] using h0
      · rw [hz] at hi
        have : i = 0 := by simpa using hi
        subst this
        simp [Array.getD, hl, bitsIdx, h1]
    · rename_i hz
      have h1s : bs.size = 1 := by omega
      obtain ⟨x, hx⟩ := List.length_eq_one_iff.mp (by simpa using h1s : bs.toList.length = 1)
      have hb : bs.getD 0 0 = x := by
        rw [Array.getD_eq_getD_getElem?, ← Array.getElem?_toList, hx]; rfl
      have hxs : x < gs.size := by rw [← hb]; exact hbs 0
      rw [hb]
      refine ⟨(Ext.refl gs).push _ (by simp [Gate.args]; exact hxs), by simp [h1s], fun a => ?_, fun i hi => ?_⟩
      · rcases a with _ | _ | a
        · rw [getD_pair0]; simp
        · rw [getD_pair1]; simp; omega
        · have : ¬ (a + 1 + 1 < 2) := by omega
          simp [Array.getD, this]
      · rw [h1s] at hi
        rw [hx]
        rcases (show i = 0 ∨ i = 1 by omega) with rfl | rfl
        · rw [getD_pair0, gval_push_new]
          simp only [Gate.step, List.map_cons, List.map_nil, bitsIdx]
          show (!gval inp gs x) = _
          cases gval inp gs x <;> simp
        · rw [getD_pair1, gval_push_lt inp gs _ hxs]
          simp only [List.map_cons, List.map_nil, bitsIdx]
          cases gval inp gs x <;> simp
  · rename_i hs
    dsimp only
    have hh1 : 1 ≤ bs.size / 2 := by omega
    have hh2 : bs.size / 2 < bs.size := by omega
    have hlsz : (bs.extract 0 (bs.size / 2)).size = bs.size / 2 := by simp [Array.size_extract]; omega
    have hhsz : (bs.extract (bs.size / 2) bs.size).size = bs.size - bs.size / 2 := by simp [Array.size_extract]
    obtain ⟨l1, l2, l3, l4⟩ := ih (bs.size / 2) (by omega) _ gs hlsz
      (getD_extract_lt bs 0 _ (by omega) hbs) hone h1
    obtain ⟨r1, r2, r3, r4⟩ := ih (bs.size - bs.size / 2) (by omega) _ (decodeGates one gs (bs.extract 0 (bs.size / 2))).1
      hhsz (fun a => Nat.lt_of_lt_of_le (getD_extract_lt bs _ bs.size (Nat.le_refl _) hbs a) l1.size_le)
      (Nat.lt_of_lt_of_le hone l1.size_le) (by rw [l1.gval inp hone, h1])
    generalize hRL : decodeGates one gs (bs.extract 0 (bs.size / 2)) = RL at *
    generalize hRH : decodeGates one RL.1 (bs.extract (bs.size / 2) bs.size) = RH at *
    obtain ⟨m1, m2, -, m4, m5⟩ := minterms_spec inp RL.2 RH.2 (bs.size / 2) (2 ^ bs.size) RH.1 0 #[]
      (fun a => Nat.lt_of_lt_of_le (l3 a) r1.size_le) r3 (by simp)
    simp only [Array.size_empty, Nat.zero_add] at m2 m5
    refine ⟨l1.trans (r1.trans m1), m2, fun a => ?_, fun i hi => ?_⟩
    · by_cases ha : a < (minterms RL.2 RH.2 (bs.size / 2) RH.1 (2 ^ bs.size) 0 #[]).2.size
      · exact m4 _ (by simp [Array.getD, ha])
      · have := Nat.lt_of_lt_of_le h0 (l1.trans (r1.trans m1)).size_le
        simpa [Array.getD, ha] using this
    · rw [m5 i hi]
      rw [hlsz] at l2 l4
      rw [hhsz] at r2 r4
      have hq : i / 2 ^ (bs.size / 2) < 2 ^ (bs.size - bs.size / 2) := by
        have hpow : 2 ^ bs.size = 2 ^ (bs.size / 2) * 2 ^ (bs.size - bs.size / 2) := by
          rw [← Nat.pow_add]; congr 1; omega
        exact (Nat.div_lt_iff_lt_mul (Nat.two_pow_pos _)).mpr (by rw [Nat.mul_comm, ← hpow]; exact hi)
      have hlo := l4 (i % 2 ^ (bs.size / 2)) (Nat.mod_lt _ (Nat.two_pow_pos _))
      rw [r1.gval inp (l3 _), hlo, r4 _ hq]
      have hrl : (bs.extract (bs.size / 2) bs.size).toList.map (gval inp RL.1) =
          (bs.extract (bs.size / 2) bs.size).toList.map (gval inp gs) := by
        apply List.map_congr_left
        intro x hx
        obtain ⟨k, hk, rfl⟩ := List.getElem_of_mem hx
        have hk' : k < bs.size - bs.size / 2 := by simpa using hk
        have := getD_extract_lt bs (bs.size / 2) bs.size (Nat.le_refl _) hbs k
        rw [show (bs.extract (bs.size / 2) bs.size).toList[k] = (bs.extract (bs.size / 2) bs.size).getD k 0 by
          simp [Array.getD, hhsz, hk']]
        exact l1.gval inp this
      rw [hrl]
      have hlt := bitsIdx_lt ((bs.extract 0 (bs.size / 2)).toList.map (gval inp gs))
      simp only [List.length_map, Array.length_toList, hlsz] at hlt
      rw [split_idx (Nat.two_pow_pos _) hlt, toList_split_nat bs (bs.size / 2), List.map_append, bitsIdx_append]
      simp [Nat.min_eq_left (Nat.le_of_lt hh2)]

/-! ## Reads as gates: the value bits -/

/-- The XOR of gates `xs` as one gate: `zero` for none. -/
def xorGates (zero : Nat) (gs : Array Gate) : List Nat → Array Gate × Nat
  | [] => (gs, zero)
  | g :: rest => xorChain gs g rest

/-- The low minterms whose word, at high minterm `h`, has bit `j` set. -/
def sel (ws : Array Nat) (lo nl h j : Nat) : List Nat :=
  (List.range nl).filter fun l => (ws.getD (h * 2 ^ lo + l) 0).testBit j

/-- Value bit `j`'s products, high minterms `h, h + 1, …`: the AND of `hi[h]` with the XOR of `sel`'s low minterms, when
`sel` is not empty. -/
def prodGates (zero : Nat) (ws : Array Nat) (lo : Nat) (loM hiM : Array Nat) (j : Nat) :
    Array Gate → Nat → Nat → List Nat → Array Gate × List Nat
  | gs, 0, _, acc => (gs, acc)
  | gs, cnt + 1, h, acc =>
    match sel ws lo loM.size h j with
    | [] => prodGates zero ws lo loM hiM j gs cnt (h + 1) acc
    | l :: ls =>
      let r := xorChain gs (loM.getD l 0) (ls.map (loM.getD · 0))
      prodGates zero ws lo loM hiM j (r.1.push (.and (hiM.getD h 0) r.2)) cnt (h + 1) (acc ++ [r.1.size])

/-- Value bits `j, j + 1, …`: each the XOR of its products. -/
def bitsGates (zero : Nat) (ws : Array Nat) (lo : Nat) (loM hiM : Array Nat) :
    Array Gate → Nat → Nat → Array Nat → Array Gate × Array Nat
  | gs, 0, _, acc => (gs, acc)
  | gs, cnt + 1, j, acc =>
    let rp := prodGates zero ws lo loM hiM j gs hiM.size 0 []
    let rx := xorGates zero rp.1 rp.2
    bitsGates zero ws lo loM hiM rx.1 cnt (j + 1) (acc.push rx.2)

theorem xorGates_spec (inp : Nat → Bool) (zero : Nat) (gs : Array Gate) (xs : List Nat) (hz : zero < gs.size)
    (hzv : gval inp gs zero = false) (hxs : ∀ x ∈ xs, x < gs.size) :
    Ext gs (xorGates zero gs xs).1 ∧ (xorGates zero gs xs).2 < (xorGates zero gs xs).1.size ∧
      gval inp (xorGates zero gs xs).1 (xorGates zero gs xs).2 = xorSum xs (gval inp gs) := by
  cases xs with
  | nil => exact ⟨Ext.refl gs, hz, by simpa [xorGates, xorSum] using hzv⟩
  | cons g rest =>
    obtain ⟨h1, h2, h3⟩ := xorChain_spec inp rest gs g (hxs g (List.mem_cons_self ..))
      (fun x hx => hxs x (List.mem_cons_of_mem _ hx))
    exact ⟨h1, h2, by simp only [xorGates]; rw [h3, xorSum_cons]⟩

theorem prodGates_spec (inp : Nat → Bool) (zero : Nat) (ws : Array Nat) (lo : Nat) (loM hiM : Array Nat) (j : Nat) :
    ∀ cnt (gs : Array Gate) h acc, (∀ a, loM.getD a 0 < gs.size) → (∀ a, hiM.getD a 0 < gs.size) →
      (∀ a ∈ acc, a < gs.size) →
      Ext gs (prodGates zero ws lo loM hiM j gs cnt h acc).1 ∧
        (∀ a ∈ (prodGates zero ws lo loM hiM j gs cnt h acc).2, a < (prodGates zero ws lo loM hiM j gs cnt h acc).1.size) ∧
        xorSum (prodGates zero ws lo loM hiM j gs cnt h acc).2 (gval inp (prodGates zero ws lo loM hiM j gs cnt h acc).1) =
          Bool.xor (xorSum acc (gval inp gs)) (xorSum (List.range' h cnt) fun h' =>
            gval inp gs (hiM.getD h' 0) && xorSum (sel ws lo loM.size h' j) fun l => gval inp gs (loM.getD l 0))
  | 0, gs, h, acc, _, _, hacc => ⟨Ext.refl gs, hacc, by simp [prodGates, xorSum]⟩
  | cnt + 1, gs, h, acc, hlo, hhi, hacc => by
    simp only [prodGates]
    split
    · rename_i hs
      obtain ⟨h1, h2, h3⟩ := prodGates_spec inp zero ws lo loM hiM j cnt gs (h + 1) acc hlo hhi hacc
      refine ⟨h1, h2, ?_⟩
      rw [h3, List.range'_succ, xorSum_cons, hs, xorSum_nil, Bool.and_false, Bool.false_xor]
    · rename_i l ls hs
      have hsel : ∀ x ∈ (l :: ls).map (loM.getD · 0), x < gs.size := by
        intro x hx
        obtain ⟨y, -, rfl⟩ := List.mem_map.mp hx
        exact hlo y
      obtain ⟨c1, c2, c3⟩ := xorChain_spec inp (ls.map (loM.getD · 0)) gs (loM.getD l 0) (hlo l)
        (fun x hx => hsel x (List.mem_cons_of_mem _ hx))
      generalize hR : xorChain gs (loM.getD l 0) (ls.map (loM.getD · 0)) = R at c1 c2 c3
      have e0 : Ext gs (R.1.push (.and (hiM.getD h 0) R.2)) :=
        c1.push _ (args_and (Nat.lt_of_lt_of_le (hhi h) c1.size_le) c2)
      obtain ⟨h1, h2, h3⟩ := prodGates_spec inp zero ws lo loM hiM j cnt (R.1.push (.and (hiM.getD h 0) R.2)) (h + 1)
        (acc ++ [R.1.size]) (fun a => Nat.lt_of_lt_of_le (hlo a) e0.size_le)
        (fun a => Nat.lt_of_lt_of_le (hhi a) e0.size_le)
        (fun a ha => by
          rcases List.mem_append.mp ha with ha | ha
          · exact Nat.lt_of_lt_of_le (hacc a ha) e0.size_le
          · rw [List.mem_singleton] at ha; subst ha; simp)
      refine ⟨e0.trans h1, h2, ?_⟩
      rw [h3]
      have hnew : gval inp (R.1.push (.and (hiM.getD h 0) R.2)) R.1.size =
          (gval inp gs (hiM.getD h 0) && xorSum (sel ws lo loM.size h j) fun l => gval inp gs (loM.getD l 0)) := by
        rw [gval_push_new]
        show (gval inp R.1 (hiM.getD h 0) && gval inp R.1 R.2) = _
        rw [c1.gval inp (hhi h), c3, hs, xorSum_cons, xorSum_map]
        rfl
      have hacc' : xorSum (acc ++ [R.1.size]) (gval inp (R.1.push (.and (hiM.getD h 0) R.2))) =
          Bool.xor (xorSum acc (gval inp gs)) (gval inp gs (hiM.getD h 0) &&
            xorSum (sel ws lo loM.size h j) fun l => gval inp gs (loM.getD l 0)) := by
        rw [xorSum_append, xorSum_cons, xorSum_nil, Bool.xor_false, hnew,
          xorSum_congr (fun a ha => e0.gval inp (hacc a ha))]
      rw [hacc', List.range'_succ, xorSum_cons, Bool.xor_assoc]
      congr 2
      apply xorSum_congr
      intro h' hh'
      have hsel' : ∀ l ∈ sel ws lo loM.size h' j, gval inp (R.1.push (.and (hiM.getD h 0) R.2)) (loM.getD l 0) =
          gval inp gs (loM.getD l 0) := fun l _ => e0.gval inp (hlo l)
      rw [e0.gval inp (hhi h'), xorSum_congr hsel']

/-- **A read's value bits**: with the decoders' minterms one-hot at `loIdx` and `hiIdx`, bit `j + m`'s gate carries bit
`j + m` of word `hiIdx · 2^lo + loIdx`. -/
theorem bitsGates_spec (inp : Nat → Bool) (zero : Nat) (ws : Array Nat) (lo : Nat) (loM hiM : Array Nat) {loIdx hiIdx : Nat}
    (hloI : loIdx < loM.size) (hhiI : hiIdx < hiM.size) :
    ∀ cnt (gs : Array Gate) j acc, (∀ a, loM.getD a 0 < gs.size) → (∀ a, hiM.getD a 0 < gs.size) → zero < gs.size →
      gval inp gs zero = false → (∀ a ∈ acc, a < gs.size) →
      (∀ l < loM.size, gval inp gs (loM.getD l 0) = decide (loIdx = l)) →
      (∀ h < hiM.size, gval inp gs (hiM.getD h 0) = decide (hiIdx = h)) →
      Ext gs (bitsGates zero ws lo loM hiM gs cnt j acc).1 ∧ (bitsGates zero ws lo loM hiM gs cnt j acc).2.size = acc.size + cnt ∧
        (∀ m < acc.size, (bitsGates zero ws lo loM hiM gs cnt j acc).2.getD m zero = acc.getD m zero) ∧
        (∀ a ∈ (bitsGates zero ws lo loM hiM gs cnt j acc).2, a < (bitsGates zero ws lo loM hiM gs cnt j acc).1.size) ∧
        ∀ m < cnt, gval inp (bitsGates zero ws lo loM hiM gs cnt j acc).1
          ((bitsGates zero ws lo loM hiM gs cnt j acc).2.getD (acc.size + m) zero) =
            (ws.getD (hiIdx * 2 ^ lo + loIdx) 0).testBit (j + m)
  | 0, gs, j, acc, _, _, _, _, hacc, _, _ =>
    ⟨Ext.refl gs, by simp [bitsGates], fun _ _ => rfl, hacc, fun m hm => absurd hm (by omega)⟩
  | cnt + 1, gs, j, acc, hlo, hhi, hz, hzv, hacc, hlov, hhiv => by
    obtain ⟨p1, p2, p3⟩ := prodGates_spec inp zero ws lo loM hiM j hiM.size gs 0 [] hlo hhi (by simp)
    generalize hP : prodGates zero ws lo loM hiM j gs hiM.size 0 [] = P at p1 p2 p3
    obtain ⟨x1, x2, x3⟩ := xorGates_spec inp zero P.1 P.2 (Nat.lt_of_lt_of_le hz p1.size_le)
      (by rw [p1.gval inp hz, hzv]) p2
    generalize hX : xorGates zero P.1 P.2 = X at x1 x2 x3
    have e : Ext gs X.1 := p1.trans x1
    -- bit `j`'s value
    have hbit : gval inp X.1 X.2 = (ws.getD (hiIdx * 2 ^ lo + loIdx) 0).testBit j := by
      rw [x3, p3, xorSum_nil, Bool.false_xor, List.range'_eq_map_range, xorSum_map]
      simp only [Nat.zero_add]
      have hsel : ∀ h, xorSum (sel ws lo loM.size h j) (fun l => gval inp gs (loM.getD l 0)) =
          decide (loIdx ∈ sel ws lo loM.size h j) := by
        intro h
        have hnd : (sel ws lo loM.size h j).Nodup := List.nodup_range.filter _
        have hlt : ∀ l ∈ sel ws lo loM.size h j, l < loM.size := fun l hl =>
          List.mem_range.mp (List.mem_filter.mp hl).1
        rw [xorSum_congr (w := fun l => decide (loIdx = l)) (fun l hl => hlov l (hlt l hl))]
        exact xorSum_onehot (fun _ => rfl) _ hnd
      have hfun : ((fun h' => gval inp gs (hiM.getD h' 0) && xorSum (sel ws lo loM.size h' j)
          fun l => gval inp gs (loM.getD l 0)) ∘ fun x => x) = fun h' =>
          gval inp gs (hiM.getD h' 0) && decide (loIdx ∈ sel ws lo loM.size h' j) := by
        funext h'; simp only [Function.comp, hsel]
      rw [hfun, xorSum_congr (w := fun h => decide (hiIdx = h) && decide (loIdx ∈ sel ws lo loM.size h j))
        (fun h hh => by rw [hhiv h (List.mem_range.mp hh)]), xorSum_select hhiI]
      simp [sel, hloI]
    obtain ⟨h1, h2, h3, h4, h5⟩ := bitsGates_spec inp zero ws lo loM hiM hloI hhiI cnt X.1 (j + 1) (acc.push X.2)
      (fun a => Nat.lt_of_lt_of_le (hlo a) e.size_le) (fun a => Nat.lt_of_lt_of_le (hhi a) e.size_le)
      (Nat.lt_of_lt_of_le hz e.size_le) (by rw [e.gval inp hz, hzv])
      (fun a ha => by
        rcases Array.mem_push.mp ha with ha | rfl
        · exact Nat.lt_of_lt_of_le (hacc a ha) e.size_le
        · exact x2)
      (fun l hl => by rw [e.gval inp (hlo l), hlov l hl]) (fun h hh => by rw [e.gval inp (hhi h), hhiv h hh])
    simp only [bitsGates, hP, hX]
    refine ⟨e.trans h1, by rw [h2]; simp; omega, fun m hm => ?_, h4, fun m hm => ?_⟩
    · rw [h3 m (by simp; omega)]
      simp [Array.getD, hm, Nat.lt_succ_of_lt hm, Array.getElem_push_lt hm]
    · cases m with
      | zero =>
        simp only [Nat.add_zero]
        rw [h3 acc.size (by simp)]
        have hx : (acc.push X.2).getD acc.size zero = X.2 := by simp [Array.getD]
        rw [hx, h1.gval inp x2, hbit]
      | succ m =>
        have := h5 m (by omega)
        rw [show acc.size + (m + 1) = (acc.push X.2).size + m by simp; omega, this, show j + 1 + m = j + (m + 1) by omega]

/-- A read's gates: its index forms' gates, split at `lo` into the two decoders, then its value bits `0 … vb − 1`. -/
def readGates (one zero : Nat) (wg : Array Nat) (gs : Array Gate) (idx : Array Form) (ws : Array Nat) (lo vb : Nat) :
    Array Gate × Array Nat :=
  let ri := formsGates one zero wg gs idx.toList #[]
  let rl := decodeGates one ri.1 (ri.2.extract 0 lo)
  let rh := decodeGates one rl.1 (ri.2.extract lo ri.2.size)
  bitsGates zero ws (ri.2.extract 0 lo).size rl.2 rh.2 rh.1 vb 0 #[]

theorem getD_default_irrel {a : Array Nat} {m : Nat} (hm : m < a.size) (d d' : Nat) : a.getD m d = a.getD m d' := by
  simp [Array.getD, hm]

theorem getD_extract_lt' (a : Array Nat) (s e : Nat) {n : Nat} (h : ∀ x, a.getD x 0 < n) :
    ∀ x, (a.extract s e).getD x 0 < n := by
  intro x
  by_cases hx : x < (a.extract s e).size
  · have hx' : s + x < a.size := by simp [Array.size_extract] at hx; omega
    have := h (s + x)
    simp only [Array.getD, hx, dite_true, Array.getElem_extract]
    simpa [Array.getD, hx'] using this
  · have hx2 : ¬ x < min e a.size - s := by simpa [Array.size_extract] using hx
    have h0 := h a.size
    simp only [Array.getD, Nat.lt_irrefl, dite_false] at h0
    simp only [Array.getD, Array.size_extract, hx2, dite_false]
    exact h0

/-- **A read's gates carry the table's bits** at the index its forms spell. -/
theorem readGates_spec (inp : Nat → Bool) {one zero : Nat} {wg : Array Nat} {gs : Array Gate} {vals : List Bool}
    (hI : WInv inp one zero gs wg vals) (idx : Array Form) (ws : Array Nat) (lo vb : Nat) :
    Ext gs (readGates one zero wg gs idx ws lo vb).1 ∧ (readGates one zero wg gs idx ws lo vb).2.size = vb ∧
      (∀ a ∈ (readGates one zero wg gs idx ws lo vb).2, a < (readGates one zero wg gs idx ws lo vb).1.size) ∧
      ∀ j < vb, gval inp (readGates one zero wg gs idx ws lo vb).1 ((readGates one zero wg gs idx ws lo vb).2.getD j zero) =
        (ws.getD (bitsIdx (idx.toList.map (formVal (vals.getD · false)))) 0).testBit j := by
  unfold readGates
  dsimp only
  obtain ⟨f1, f2, -, f4, f5⟩ := formsGates_spec inp idx.toList gs #[] hI (by simp)
  generalize hRI : formsGates one zero wg gs idx.toList #[] = RI at f1 f2 f4 f5 ⊢
  simp only [Array.size_empty, Nat.zero_add, Array.length_toList] at f2 f5
  have hI1 := hI.ext f1
  -- the index bits' gates, as leaves
  have hleaf : ∀ a, RI.2.getD a 0 < RI.1.size := by
    intro a
    by_cases ha : a < RI.2.size
    · exact f4 _ (by simp [Array.getD, ha])
    · have h0 : 0 < RI.1.size := by have := hI1.lt 0; omega
      simpa [Array.getD, ha] using h0
  have hvals : RI.2.toList.map (gval inp RI.1) = idx.toList.map (formVal (vals.getD · false)) := by
    apply List.ext_getElem (by simp [f2])
    intro m h1 h2
    simp only [List.getElem_map, Array.getElem_toList]
    have hm : m < idx.size := by simpa using h2
    have := f5 m (by simpa using hm)
    rw [show RI.2[m] = RI.2.getD m zero by simp [Array.getD, show m < RI.2.size by rw [f2]; exact hm], this]
    simp [List.getD_eq_getElem?_getD, hm]
  obtain ⟨l1, l2, l3, l4⟩ := decodeGates_spec inp one _ (RI.2.extract 0 lo) RI.1 rfl
    (getD_extract_lt' RI.2 0 lo hleaf) hI1.one_lt hI1.one_val
  generalize hRL : decodeGates one RI.1 (RI.2.extract 0 lo) = RL at l1 l2 l3 l4 ⊢
  have hI2 := hI1.ext l1
  obtain ⟨r1, r2, r3, r4⟩ := decodeGates_spec inp one _ (RI.2.extract lo RI.2.size) RL.1 rfl
    (fun a => Nat.lt_of_lt_of_le (getD_extract_lt' RI.2 lo RI.2.size hleaf a) l1.size_le) hI2.one_lt hI2.one_val
  generalize hRH : decodeGates one RL.1 (RI.2.extract lo RI.2.size) = RH at r1 r2 r3 r4 ⊢
  have hI3 := hI2.ext r1
  -- the two halves' indices
  obtain ⟨loIdx, hloDef⟩ : ∃ x, bitsIdx ((RI.2.extract 0 lo).toList.map (gval inp RI.1)) = x := ⟨_, rfl⟩
  obtain ⟨hiIdx, hhiDef⟩ : ∃ x, bitsIdx ((RI.2.extract lo RI.2.size).toList.map (gval inp RI.1)) = x := ⟨_, rfl⟩
  have hloI : loIdx < RL.2.size := by
    rw [← hloDef, l2]; simpa using bitsIdx_lt ((RI.2.extract 0 lo).toList.map (gval inp RI.1))
  have hvh : (RI.2.extract lo RI.2.size).toList.map (gval inp RL.1) =
      (RI.2.extract lo RI.2.size).toList.map (gval inp RI.1) := by
    apply List.map_congr_left
    intro x hx
    obtain ⟨k, hk, rfl⟩ := List.getElem_of_mem hx
    have hk' : k < (RI.2.extract lo RI.2.size).size := by simpa using hk
    have := getD_extract_lt' RI.2 lo RI.2.size hleaf k
    have e : (RI.2.extract lo RI.2.size).getD k 0 = (RI.2.extract lo RI.2.size).toList[k] := by
      rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hk', Option.getD_some, Array.getElem_toList]
    rw [← e]
    exact l1.gval inp this
  have hhiI : hiIdx < RH.2.size := by
    rw [← hhiDef, r2]; simpa using bitsIdx_lt ((RI.2.extract lo RI.2.size).toList.map (gval inp RI.1))
  obtain ⟨b1, b2, -, b4, b5⟩ := bitsGates_spec inp zero ws (RI.2.extract 0 lo).size RL.2 RH.2 hloI hhiI vb RH.1 0 #[]
    (fun a => Nat.lt_of_lt_of_le (l3 a) r1.size_le) r3 hI3.zero_lt hI3.zero_val (by simp)
    (fun l hl => by rw [r1.gval inp (l3 l), l4 l (by rw [l2] at hl; exact hl), hloDef])
    (fun h hh => by rw [r4 h (by rw [r2] at hh; exact hh), hvh, hhiDef])
  simp only [Array.size_empty, Nat.zero_add] at b2 b5
  have hidx : bitsIdx (idx.toList.map (formVal (vals.getD · false))) = hiIdx * 2 ^ (RI.2.extract 0 lo).size + loIdx := by
    rw [← hvals, toList_split_nat RI.2 lo, List.map_append, bitsIdx_append, hloDef, hhiDef]
    simp only [List.length_map, Array.length_toList]
    rw [Nat.mul_comm, Nat.add_comm]
  refine ⟨f1.trans (l1.trans (r1.trans b1)), b2, b4, fun j hj => ?_⟩
  rw [b5 j hj, hidx]

/-! ## Items as gates -/

theorem getD_append_nat (a b : Array Nat) (w d : Nat) :
    (a ++ b).getD w d = if w < a.size then a.getD w d else b.getD (w - a.size) d := by
  simp only [Array.getD_eq_getD_getElem?]
  by_cases h1 : w < a.size
  · rw [if_pos h1, Array.getElem?_append_left h1]
  · rw [if_neg h1, Array.getElem?_append_right (by omega)]

theorem list_getD_append (l m : List Bool) (w : Nat) :
    (l ++ m).getD w false = if w < l.length then l.getD w false else m.getD (w - l.length) false := by
  simp only [List.getD_eq_getElem?_getD]
  by_cases h : w < l.length
  · rw [if_pos h, List.getElem?_append_left h]
  · rw [if_neg h, List.getElem?_append_right (by omega)]

theorem WInv.append {inp : Nat → Bool} {one zero : Nat} {gs : Array Gate} {wg : Array Nat} {vals : List Bool}
    (h : WInv inp one zero gs wg vals) {X : Array Nat} {Y : List Bool} (hs : X.size = Y.length)
    (hlt : ∀ a ∈ X, a < gs.size) (hX : ∀ m < X.size, gval inp gs (X.getD m zero) = Y.getD m false) :
    WInv inp one zero gs (wg ++ X) (vals ++ Y) where
  size := by simp [h.size, hs]
  lt w := by
    rw [getD_append_nat]
    by_cases hw : w < wg.size
    · rw [if_pos hw]; exact h.lt w
    · rw [if_neg hw]
      by_cases hx : w - wg.size < X.size
      · exact hlt _ (by simp only [Array.getD, hx, dite_true]; exact Array.getElem_mem hx)
      · rw [show X.getD (w - wg.size) zero = zero by simp [Array.getD, hx]]; exact h.zero_lt
  val w := by
    rw [getD_append_nat, list_getD_append, ← h.size]
    by_cases hw : w < wg.size
    · rw [if_pos hw, if_pos hw]; exact h.val w
    · rw [if_neg hw, if_neg hw]
      by_cases hx : w - wg.size < X.size
      · exact hX _ hx
      · rw [show X.getD (w - wg.size) zero = zero by simp [Array.getD, hx], h.zero_val,
          List.getD_eq_getElem?_getD, List.getElem?_eq_none (by omega)]
        rfl
  one_lt := h.one_lt
  one_val := h.one_val
  zero_val := h.zero_val

theorem WInv.push {inp : Nat → Bool} {one zero : Nat} {gs : Array Gate} {wg : Array Nat} {vals : List Bool}
    (h : WInv inp one zero gs wg vals) {g : Nat} (hg : g < gs.size) {b : Bool} (hb : gval inp gs g = b) :
    WInv inp one zero gs (wg.push g) (vals ++ [b]) := by
  have := h.append (X := #[g]) (Y := [b]) (by simp) (by simpa using hg) (fun m hm => by
    have : m = 0 := by simpa using hm
    subst this; simpa [Array.getD] using hb)
  simpa using this

theorem formVal_default (v : Nat → Bool) : formVal v default = false := rfl

/-- A callee's output `o` as `evalT` gives it: its output form's value, false past its outputs. -/
theorem evalT_getD_outputs (types : String → Option CircuitType) (words : String → Option (Array Nat)) (F : Nat)
    (u : CircuitType) (x : Nat → Bool) (o : Nat) :
    (evalT types words (F + 1) u x).getD o false = formVal ((runT types words (fun u y => evalT types words F u y) u
      ((List.range u.inBits).map x) u.items.toList).getD · false) (u.outputs.getD o default) := by
  simp only [evalT, List.getD_eq_getElem?_getD, List.getElem?_map, Array.getElem?_toList]
  by_cases ho : o < u.outputs.size
  · simp [ho, Array.getD]
  · simp [ho, Array.getD, formVal_default]

/-- Items of type `t` as gates, from the wires' gates `wg`: an AND's gate over its forms' gates; a call's callee inlined on
its input forms' gates (at depth `F`, as `evalT` runs it: none left gives zero outputs); a read's `readGates`, split at
`loOf n` for an `n`-bit index. Returns the gates and the wires' gates. -/
def flatItems (types : String → Option CircuitType) (words : String → Option (Array Nat)) (loOf : Nat → Nat)
    (one zero : Nat) : Nat → CircuitType → Array Gate → Array Nat → List Item → Array Gate × Array Nat
  | _, _, gs, wg, [] => (gs, wg)
  | F, t, gs, wg, .and a b :: rest =>
    let ra := formGate one zero wg gs a
    let rb := formGate one zero wg ra.1 b
    flatItems types words loOf one zero F t (rb.1.push (.and ra.2 rb.2)) (wg.push rb.1.size) rest
  | F, t, gs, wg, .call k args :: rest =>
    match (t.calls[k]?).bind types with
    | none => flatItems types words loOf one zero F t gs wg rest
    | some u =>
      match F with
      | 0 => flatItems types words loOf one zero 0 t gs (wg ++ Array.replicate u.outBits zero) rest
      | F' + 1 =>
        let ri := formsGates one zero wg gs ((List.range u.inBits).map (args.getD · default)) #[]
        let rc := flatItems types words loOf one zero F' u ri.1 ri.2 u.items.toList
        let ro := formsGates one zero rc.2 rc.1 ((List.range u.outBits).map (u.outputs.getD · default)) #[]
        flatItems types words loOf one zero (F' + 1) t ro.1 (wg ++ ro.2) rest
  | F, t, gs, wg, .read k idx :: rest =>
    match t.tables[k]? with
    | none => flatItems types words loOf one zero F t gs wg rest
    | some (sha, n, vb) =>
      let r := readGates one zero wg gs idx ((words sha).getD #[]) (loOf n) vb
      flatItems types words loOf one zero F t r.1 (wg ++ r.2) rest
  | F, t, gs, wg, .bad _ :: rest => flatItems types words loOf one zero F t gs wg rest
termination_by F _ _ _ items => (F, items.length)

theorem flatItems_spec (types : String → Option CircuitType) (words : String → Option (Array Nat)) (loOf : Nat → Nat)
    (inp : Nat → Bool) (one zero : Nat) :
    ∀ F t (items : List Item) (gs : Array Gate) (wg : Array Nat) (vals : List Bool), WInv inp one zero gs wg vals →
      Ext gs (flatItems types words loOf one zero F t gs wg items).1 ∧
        WInv inp one zero (flatItems types words loOf one zero F t gs wg items).1
          (flatItems types words loOf one zero F t gs wg items).2
          (runT types words (fun u y => evalT types words F u y) t vals items) := by
  intro F
  induction F using Nat.strongRecOn with
  | ind F ihF =>
  intro t items
  induction items with
  | nil =>
    intro gs wg vals hI
    simp only [flatItems, runT]
    exact ⟨Ext.refl gs, hI⟩
  | cons it rest ih =>
    intro gs wg vals hI
    cases it with
    | and a b =>
      obtain ⟨a1, a2, a3⟩ := formGate_spec inp hI a
      generalize hA : formGate one zero wg gs a = A at a1 a2 a3
      obtain ⟨b1, b2, b3⟩ := formGate_spec inp (hI.ext a1) b
      generalize hB : formGate one zero wg A.1 b = B at b1 b2 b3
      have e : Ext gs (B.1.push (.and A.2 B.2)) :=
        (a1.trans b1).push _ (args_and (Nat.lt_of_lt_of_le a2 b1.size_le) b2)
      have hI' : WInv inp one zero (B.1.push (.and A.2 B.2)) (wg.push B.1.size)
          (vals ++ [formVal (vals.getD · false) a && formVal (vals.getD · false) b]) := by
        refine (hI.ext e).push (by simp) ?_
        rw [gval_push_new]
        show (gval inp B.1 A.2 && gval inp B.1 B.2) = _
        rw [b1.gval inp a2, a3, b3]
      obtain ⟨r1, r2⟩ := ih _ _ _ hI'
      simp only [flatItems, hA, hB, runT, itemOut]
      exact ⟨e.trans r1, r2⟩
    | call k args =>
      cases hu : (t.calls[k]?).bind types with
      | none =>
        obtain ⟨r1, r2⟩ := ih gs wg vals hI
        simp only [flatItems, hu, runT, itemOut, List.append_nil]
        exact ⟨r1, r2⟩
      | some u =>
        cases F with
        | zero =>
          have hI' : WInv inp one zero gs (wg ++ Array.replicate u.outBits zero)
              (vals ++ (List.range u.outBits).map fun o =>
                (evalT types words 0 u fun i => formVal (vals.getD · false) (args.getD i default)).getD o false) := by
            refine hI.append (by simp) (fun a ha => ?_) (fun m hm => ?_)
            · rw [(Array.mem_replicate.mp ha).2]; exact hI.zero_lt
            · simp only [Array.size_replicate] at hm
              simp [Array.getD, hm, evalT, List.getD_eq_getElem?_getD, hI.zero_val]
          obtain ⟨r1, r2⟩ := ih gs _ _ hI'
          simp only [flatItems, hu, runT, itemOut]
          exact ⟨r1, r2⟩
        | succ F' =>
          obtain ⟨i1, i2, -, i4, i5⟩ := formsGates_spec inp ((List.range u.inBits).map (args.getD · default)) gs #[] hI
            (by simp)
          generalize hRI : formsGates one zero wg gs ((List.range u.inBits).map (args.getD · default)) #[] = RI at i1 i2 i4 i5
          simp only [Array.size_empty, Nat.zero_add, List.length_map, List.length_range] at i2 i5
          have hI1 := hI.ext i1
          have hIc : WInv inp one zero RI.1 RI.2
              ((List.range u.inBits).map fun i => formVal (vals.getD · false) (args.getD i default)) :=
            { size := by simp [i2]
              lt := fun w => by
                by_cases hw : w < RI.2.size
                · exact i4 _ (by simp [Array.getD, hw])
                · simpa [Array.getD, hw] using hI1.zero_lt
              val := fun w => by
                by_cases hw : w < u.inBits
                · rw [i5 w hw]
                  simp [List.getD_eq_getElem?_getD, hw]
                · have hw' : ¬ w < RI.2.size := by omega
                  have hnone : ((List.range u.inBits).map fun i => formVal (vals.getD · false) (args.getD i default)).getD
                      w false = false := by simp [List.getD_eq_getElem?_getD, hw]
                  rw [hnone, show RI.2.getD w zero = zero by simp [Array.getD, hw']]
                  exact hI1.zero_val
              one_lt := hI1.one_lt
              one_val := hI1.one_val
              zero_val := hI1.zero_val }
          obtain ⟨c1, c2⟩ := ihF F' (by omega) u u.items.toList RI.1 RI.2 _ hIc
          generalize hRC : flatItems types words loOf one zero F' u RI.1 RI.2 u.items.toList = RC at c1 c2
          obtain ⟨o1, o2, -, o4, o5⟩ := formsGates_spec inp ((List.range u.outBits).map (u.outputs.getD · default)) RC.1
            #[] c2 (by simp)
          generalize hRO : formsGates one zero RC.2 RC.1 ((List.range u.outBits).map (u.outputs.getD · default)) #[] = RO
            at o1 o2 o4 o5
          simp only [Array.size_empty, Nat.zero_add, List.length_map, List.length_range] at o2 o5
          have e : Ext gs RO.1 := i1.trans (c1.trans o1)
          have hI' : WInv inp one zero RO.1 (wg ++ RO.2) (vals ++ (List.range u.outBits).map fun o =>
              (evalT types words (F' + 1) u fun i => formVal (vals.getD · false) (args.getD i default)).getD o false) := by
            refine (hI.ext e).append (by simp [o2]) o4 (fun m hm => ?_)
            have hm' : m < u.outBits := by omega
            rw [o5 m hm', list_getD_map_range, if_pos hm', evalT_getD_outputs]
            congr 1
            simp [List.getD_eq_getElem?_getD, List.getElem?_range hm']
          obtain ⟨r1, r2⟩ := ih RO.1 (wg ++ RO.2) _ hI'
          simp only [flatItems, hu, hRI, hRC, hRO, runT, itemOut]
          exact ⟨e.trans r1, r2⟩
    | read k idx =>
      cases ht : t.tables[k]? with
      | none =>
        obtain ⟨r1, r2⟩ := ih gs wg vals hI
        simp only [flatItems, ht, runT, itemOut, List.append_nil]
        exact ⟨r1, r2⟩
      | some e =>
        obtain ⟨sha, n, vb⟩ := e
        obtain ⟨g1, g2, g3, g4⟩ := readGates_spec inp hI idx ((words sha).getD #[]) (loOf n) vb
        generalize hR : readGates one zero wg gs idx ((words sha).getD #[]) (loOf n) vb = R at g1 g2 g3 g4
        have hI' : WInv inp one zero R.1 (wg ++ R.2) (vals ++ (List.range vb).map fun j =>
            (((words sha).getD #[]).getD (bitsIdx (idx.toList.map (formVal (vals.getD · false)))) 0).testBit j) := by
          refine (hI.ext g1).append (by simp [g2]) g3 (fun m hm => ?_)
          rw [g4 m (by omega)]
          simp [List.getD_eq_getElem?_getD, show m < vb by omega]
        obtain ⟨r1, r2⟩ := ih R.1 (wg ++ R.2) _ hI'
        simp only [flatItems, ht, hR, runT, itemOut]
        exact ⟨g1.trans r1, r2⟩
    | bad x =>
      obtain ⟨r1, r2⟩ := ih gs wg vals hI
      simp only [flatItems, runT, itemOut, List.append_nil]
      exact ⟨r1, r2⟩

/-! ## The flattened circuit -/

/-- **Type `t`'s Boolean circuit**, calls inlined to depth `F` (`evalT`'s fuel): gates `0 … inBits − 1` are its inputs,
gate `inBits` the input `one` (fed 1) and gate `inBits + 1` its NOT, `zero`; then its items' gates. Returns the gates and
its outputs' gates. -/
def flatten (types : String → Option CircuitType) (words : String → Option (Array Nat)) (loOf : Nat → Nat) (F : Nat)
    (t : CircuitType) : Array Gate × Array Nat :=
  let gs0 := (Array.replicate (t.inBits + 1) Gate.input).push (.not t.inBits)
  let rc := flatItems types words loOf t.inBits (t.inBits + 1) F t gs0 (Array.range t.inBits) t.items.toList
  formsGates t.inBits (t.inBits + 1) rc.2 rc.1 t.outputs.toList #[]

theorem gval_inputs (inp : Nat → Bool) : ∀ m i, i < m → gval inp (Array.replicate m Gate.input) i = inp i
  | 0, i, hi => absurd hi (Nat.not_lt_zero _)
  | m + 1, i, hi => by
    rw [Array.replicate_succ]
    by_cases him : i < m
    · rw [gval_push_lt inp _ _ (by simp; exact him), gval_inputs inp m i him]
    · have : i = m := by omega
      subst this
      have := gval_push_new inp (Array.replicate i Gate.input) Gate.input
      simp only [Array.size_replicate] at this
      rw [this]
      simp [Gate.step, evalG_size]

/-- **`flatten_eval`: the flattened circuit computes the type.** On inputs `x`, with `one` fed 1, output `o`'s gate carries
`evalT`'s output `o`, at any depth: the program's Boolean circuit and its types say the same thing. -/
theorem flatten_eval (types : String → Option CircuitType) (words : String → Option (Array Nat)) (loOf : Nat → Nat)
    (F : Nat) (t : CircuitType) (x : Nat → Bool) (o : Nat) (ho : o < t.outputs.size) :
    gval (fun i => if i < t.inBits then x i else true) (flatten types words loOf F t).1
      ((flatten types words loOf F t).2.getD o 0) = (evalT types words (F + 1) t x).getD o false := by
  generalize hinp : (fun i => if i < t.inBits then x i else true) = inp
  have hin : ∀ i < t.inBits + 1, gval inp (Array.replicate (t.inBits + 1) Gate.input) i = inp i := fun i hi =>
    gval_inputs inp _ i hi
  have hone : gval inp ((Array.replicate (t.inBits + 1) Gate.input).push (.not t.inBits)) t.inBits = true := by
    rw [gval_push_lt inp _ _ (by simp), hin _ (by omega), ← hinp]; simp
  have hzero : gval inp ((Array.replicate (t.inBits + 1) Gate.input).push (.not t.inBits)) (t.inBits + 1) = false := by
    have := gval_push_new inp (Array.replicate (t.inBits + 1) Gate.input) (.not t.inBits)
    simp only [Array.size_replicate] at this
    rw [this]
    show (!gval inp (Array.replicate (t.inBits + 1) Gate.input) t.inBits) = false
    rw [hin _ (by omega), ← hinp]; simp
  have hI0 : WInv inp t.inBits (t.inBits + 1) ((Array.replicate (t.inBits + 1) Gate.input).push (.not t.inBits))
      (Array.range t.inBits) ((List.range t.inBits).map x) :=
    { size := by simp
      lt := fun w => by
        by_cases hw : w < t.inBits
        · simp [Array.getD, hw]; omega
        · simp [Array.getD, hw]
      val := fun w => by
        by_cases hw : w < t.inBits
        · have hr : (Array.range t.inBits).getD w (t.inBits + 1) = w := by simp [Array.getD, hw]
          rw [hr, gval_push_lt inp _ _ (by simp; omega), hin w (by omega), ← hinp]
          simp [hw, List.getD_eq_getElem?_getD]
        · have hr : (Array.range t.inBits).getD w (t.inBits + 1) = t.inBits + 1 := by simp [Array.getD, hw]
          rw [hr, hzero]
          simp [List.getD_eq_getElem?_getD, hw]
      one_lt := by simp only [Array.size_push, Array.size_replicate]; omega
      one_val := hone
      zero_val := hzero }
  obtain ⟨c1, c2⟩ := flatItems_spec types words loOf inp t.inBits (t.inBits + 1) F t t.items.toList _ _ _ hI0
  obtain ⟨-, f2, -, f4, f5⟩ := formsGates_spec inp t.outputs.toList _ #[] c2 (by simp)
  simp only [Array.size_empty, Nat.zero_add, Array.length_toList] at f2 f5
  unfold flatten
  simp only
  rw [getD_default_irrel (by rw [f2]; exact ho) 0 (t.inBits + 1), f5 o ho]
  simp only [evalT, List.getD_eq_getElem?_getD, List.getElem?_map, Array.getElem?_toList]
  simp [ho, Array.getD]

theorem Topo.prefix_ext {gs : Array Gate} (h : Topo gs) {k : Nat} :
    ∀ m, k ≤ m → m ≤ gs.size → Ext (gs.extract 0 k) (gs.extract 0 m)
  | 0, hkm, _ => by
    have : k = 0 := by omega
    subst this; exact Ext.refl _
  | m + 1, hkm, hm => by
    by_cases hkm' : k = m + 1
    · subst hkm'; exact Ext.refl _
    · have e := Topo.prefix_ext h (k := k) m (by omega) (by omega)
      rw [Array.extract_succ_right (as := gs) (i := 0) (j := m) (by omega) (by omega)]
      refine e.push _ (fun a ha => ?_)
      have := h m (by omega) a ha
      simp [Array.size_extract]; omega

theorem Topo.ext_whole {gs : Array Gate} (h : Topo gs) {k : Nat} (hk : k ≤ gs.size) : Ext (gs.extract 0 k) gs := by
  have := h.prefix_ext gs.size hk (Nat.le_refl _)
  rwa [Array.extract_size] at this

/-- In a circuit whose gates read only earlier gates, gate `i`'s value is its step on the gates before it. -/
theorem gval_step {gs : Array Gate} (h : Topo gs) (inp : Nat → Bool) {i : Nat} (hi : i < gs.size) :
    gval inp gs i = gs[i].step inp (evalG inp (gs.extract 0 i)) := by
  have e := h.ext_whole (k := i + 1) (by omega)
  rw [e.gval inp (by simp [Array.size_extract]; omega), Array.extract_succ_right (as := gs) (i := 0) (j := i) (by omega) hi]
  have hs : (gs.extract 0 i).size = i := by simp [Array.size_extract]; omega
  have := gval_push_new inp (gs.extract 0 i) gs[i]
  rw [hs] at this
  exact this

theorem gval_prefix {gs : Array Gate} (h : Topo gs) (inp : Nat → Bool) {i a : Nat} (hi : i ≤ gs.size) (ha : a < i) :
    (evalG inp (gs.extract 0 i)).getD a false = gval inp gs a := by
  rw [(h.ext_whole hi).gval inp (by simp [Array.size_extract]; omega)]
  rfl

/-- **The flattened circuit's gates read only earlier gates**, and its outputs are gates of it. -/
theorem flatten_topo (types : String → Option CircuitType) (words : String → Option (Array Nat)) (loOf : Nat → Nat)
    (F : Nat) (t : CircuitType) :
    Topo (flatten types words loOf F t).1 ∧ ∀ o, (flatten types words loOf F t).2.getD o 0 < (flatten types words loOf F t).1.size := by
  have h0 : Topo ((Array.replicate (t.inBits + 1) Gate.input).push (.not t.inBits)) := by
    intro i hi a ha
    by_cases hi' : i < t.inBits + 1
    · rw [Array.getElem_push_lt (by simpa using hi')] at ha
      simp [Gate.args] at ha
    · have : i = t.inBits + 1 := by simp at hi; omega
      subst this
      simp [Array.getElem_push, Gate.args] at ha
      omega
  -- any valuation gives the invariant, and so the extension
  let inp : Nat → Bool := fun i => if i < t.inBits then false else true
  have hin : ∀ i < t.inBits + 1, gval inp (Array.replicate (t.inBits + 1) Gate.input) i = inp i := fun i hi =>
    gval_inputs inp _ i hi
  have hone : gval inp ((Array.replicate (t.inBits + 1) Gate.input).push (.not t.inBits)) t.inBits = true := by
    rw [gval_push_lt inp _ _ (by simp), hin _ (by omega)]; simp [inp]
  have hzero : gval inp ((Array.replicate (t.inBits + 1) Gate.input).push (.not t.inBits)) (t.inBits + 1) = false := by
    have := gval_push_new inp (Array.replicate (t.inBits + 1) Gate.input) (.not t.inBits)
    simp only [Array.size_replicate] at this
    rw [this]
    show (!gval inp (Array.replicate (t.inBits + 1) Gate.input) t.inBits) = false
    rw [hin _ (by omega)]; simp [inp]
  have hI0 : WInv inp t.inBits (t.inBits + 1) ((Array.replicate (t.inBits + 1) Gate.input).push (.not t.inBits))
      (Array.range t.inBits) ((List.range t.inBits).map fun _ => false) :=
    { size := by simp
      lt := fun w => by
        by_cases hw : w < t.inBits
        · simp [Array.getD, hw]; omega
        · simp [Array.getD, hw]
      val := fun w => by
        by_cases hw : w < t.inBits
        · have hr : (Array.range t.inBits).getD w (t.inBits + 1) = w := by simp [Array.getD, hw]
          rw [hr, gval_push_lt inp _ _ (by simp; omega), hin w (by omega)]
          simp [inp, hw, List.getD_eq_getElem?_getD]
        · have hr : (Array.range t.inBits).getD w (t.inBits + 1) = t.inBits + 1 := by simp [Array.getD, hw]
          rw [hr, hzero]
          simp [List.getD_eq_getElem?_getD, hw]
      one_lt := by simp only [Array.size_push, Array.size_replicate]; omega
      one_val := hone
      zero_val := hzero }
  obtain ⟨c1, c2⟩ := flatItems_spec types words loOf inp t.inBits (t.inBits + 1) F t t.items.toList _ _ _ hI0
  obtain ⟨f1, -, -, f4, -⟩ := formsGates_spec inp t.outputs.toList _ #[] c2 (by simp)
  unfold flatten
  simp only
  refine ⟨h0.ext (c1.trans f1), fun o => ?_⟩
  by_cases ho : o < (formsGates t.inBits (t.inBits + 1)
      (flatItems types words loOf t.inBits (t.inBits + 1) F t ((Array.replicate (t.inBits + 1) Gate.input).push
        (.not t.inBits)) (Array.range t.inBits) t.items.toList).2
      (flatItems types words loOf t.inBits (t.inBits + 1) F t ((Array.replicate (t.inBits + 1) Gate.input).push
        (.not t.inBits)) (Array.range t.inBits) t.items.toList).1 t.outputs.toList #[]).2.size
  · exact f4 _ (by simp [Array.getD, ho])
  · have := Nat.lt_of_lt_of_le (show 0 < ((Array.replicate (t.inBits + 1) Gate.input).push (.not t.inBits)).size by simp)
      (c1.trans f1).size_le
    simpa [Array.getD, ho] using this

/-! ## As an `Audit.Circuit` -/

/-- Gate `g` as an `Audit.Op`: its arguments as gates of the array (below it, when `Topo` holds; an argument past the
array reads `g` itself, which `Topo` rules out). -/
def toOp (gs : Array Gate) (g : Fin gs.size) : Op gs.size :=
  let fin := fun a : ℕ => if ha : a < gs.size then (⟨a, ha⟩ : Fin gs.size) else g
  match gs[g.val]'g.isLt with
  | .input => .input
  | .and a b => .and (fin a) (fin b)
  | .xor a b => .xor (fin a) (fin b)
  | .not a => .not (fin a)

theorem toOp_input {gs : Array Gate} {n : ℕ} (hn : n < gs.size) (hg : gs[n] = .input) :
    toOp gs ⟨n, hn⟩ = .input := by
  simp [toOp, hg]

theorem toOp_and {gs : Array Gate} {n x y : ℕ} (hn : n < gs.size) (hg : gs[n] = .and x y) (hx : x < gs.size)
    (hy : y < gs.size) : toOp gs ⟨n, hn⟩ = .and ⟨x, hx⟩ ⟨y, hy⟩ := by
  simp [toOp, hg, hx, hy]

theorem toOp_xor {gs : Array Gate} {n x y : ℕ} (hn : n < gs.size) (hg : gs[n] = .xor x y) (hx : x < gs.size)
    (hy : y < gs.size) : toOp gs ⟨n, hn⟩ = .xor ⟨x, hx⟩ ⟨y, hy⟩ := by
  simp [toOp, hg, hx, hy]

theorem toOp_not {gs : Array Gate} {n x : ℕ} (hn : n < gs.size) (hg : gs[n] = .not x) (hx : x < gs.size) :
    toOp gs ⟨n, hn⟩ = .not ⟨x, hx⟩ := by
  simp [toOp, hg, hx]

theorem toOp_args {gs : Array Gate} (h : Topo gs) (g : Fin gs.size) : ∀ a ∈ (toOp gs g).args, a < g := by
  obtain ⟨n, hn⟩ := g
  intro a ha
  have hlt : ∀ x ∈ gs[n].args, x < n := h n hn
  cases hg : gs[n] with
  | input => rw [toOp_input hn hg] at ha; simp [Op.args] at ha
  | and x y =>
    have hx := hlt x (by rw [hg]; simp [Gate.args])
    have hy := hlt y (by rw [hg]; simp [Gate.args])
    rw [toOp_and hn hg (by omega) (by omega)] at ha
    simp only [Op.args, List.mem_cons, List.not_mem_nil, or_false] at ha
    rcases ha with rfl | rfl
    · exact hx
    · exact hy
  | xor x y =>
    have hx := hlt x (by rw [hg]; simp [Gate.args])
    have hy := hlt y (by rw [hg]; simp [Gate.args])
    rw [toOp_xor hn hg (by omega) (by omega)] at ha
    simp only [Op.args, List.mem_cons, List.not_mem_nil, or_false] at ha
    rcases ha with rfl | rfl
    · exact hx
    · exact hy
  | not x =>
    have hx := hlt x (by rw [hg]; simp [Gate.args])
    rw [toOp_not hn hg (by omega)] at ha
    simp only [Op.args, List.mem_cons, List.not_mem_nil, or_false] at ha
    subst ha
    exact hx

/-- **A gate array as an `Audit.Circuit`**: the same gates, each over gates below it (`Topo`). -/
def toCircuit (gs : Array Gate) (h : Topo gs) (outs : Finset (Fin gs.size)) : Circuit where
  N := gs.size
  op := toOp gs
  topo := toOp_args h
  outputs := outs

/-- **The circuit computes the gate array**: on inputs `inp`, gate `g`'s value in `Audit.Circuit.eval` is `gval`'s. -/
theorem toCircuit_eval (gs : Array Gate) (h : Topo gs) (outs : Finset (Fin gs.size)) (inp : ℕ → Bool)
    (g : Fin gs.size) : (toCircuit gs h outs).eval (fun g => inp g.val) g = gval inp gs g.val := by
  suffices ∀ n (hn : n < gs.size), (toCircuit gs h outs).eval (fun g => inp g.val) ⟨n, hn⟩ = gval inp gs n from
    this g.val g.isLt
  intro n
  induction n using Nat.strongRecOn with
  | ind n ih =>
  intro hn
  rw [gval_step h inp hn]
  have hsz : (evalG inp (gs.extract 0 n)).size = n := by rw [evalG_size]; simp [Array.size_extract]; omega
  have hlt : ∀ x ∈ gs[n].args, x < n := h n hn
  cases hg : gs[n] with
  | input =>
    have hop : (toCircuit gs h outs).op ⟨n, hn⟩ = .input := toOp_input hn hg
    rw [Circuit.eval_input _ _ (by rw [hop]; rfl)]
    simp only [Gate.step, hsz]
  | and x y =>
    have hx := hlt x (by rw [hg]; simp [Gate.args])
    have hy := hlt y (by rw [hg]; simp [Gate.args])
    have hop : (toCircuit gs h outs).op ⟨n, hn⟩ = .and ⟨x, show x < gs.size by omega⟩ ⟨y, show y < gs.size by omega⟩ :=
      toOp_and hn hg _ _
    rw [Circuit.eval_op _ _ (by rw [hop]; rfl), hop]
    simp only [Op.apply, Gate.step]
    rw [ih x hx (by omega), ih y hy (by omega), gval_prefix h inp (Nat.le_of_lt hn) hx,
      gval_prefix h inp (Nat.le_of_lt hn) hy]
  | xor x y =>
    have hx := hlt x (by rw [hg]; simp [Gate.args])
    have hy := hlt y (by rw [hg]; simp [Gate.args])
    have hop : (toCircuit gs h outs).op ⟨n, hn⟩ = .xor ⟨x, show x < gs.size by omega⟩ ⟨y, show y < gs.size by omega⟩ :=
      toOp_xor hn hg _ _
    rw [Circuit.eval_op _ _ (by rw [hop]; rfl), hop]
    simp only [Op.apply, Gate.step]
    rw [ih x hx (by omega), ih y hy (by omega), gval_prefix h inp (Nat.le_of_lt hn) hx,
      gval_prefix h inp (Nat.le_of_lt hn) hy]
  | not x =>
    have hx := hlt x (by rw [hg]; simp [Gate.args])
    have hop : (toCircuit gs h outs).op ⟨n, hn⟩ = .not ⟨x, show x < gs.size by omega⟩ := toOp_not hn hg _
    rw [Circuit.eval_op _ _ (by rw [hop]; rfl), hop]
    simp only [Op.apply, Gate.step]
    rw [ih x hx (by omega), gval_prefix h inp (Nat.le_of_lt hn) hx]

/-- **Type `t`'s Boolean circuit**, as an `Audit.Circuit`: `flatten`'s gates, its outputs' gates as outputs. -/
def flatCircuit (types : String → Option CircuitType) (words : String → Option (Array ℕ)) (loOf : ℕ → ℕ) (F : ℕ)
    (t : CircuitType) : Circuit :=
  toCircuit (flatten types words loOf F t).1 (flatten_topo types words loOf F t).1
    (Finset.univ.filter fun g => g.val ∈ (flatten types words loOf F t).2)

/-- **`flatCircuit_eval`**: on inputs `x` (and `one` fed 1), the gate of type `t`'s output `o` carries `evalT`'s
output `o`, in `Audit.Circuit.eval`. -/
theorem flatCircuit_eval (types : String → Option CircuitType) (words : String → Option (Array ℕ)) (loOf : ℕ → ℕ)
    (F : ℕ) (t : CircuitType) (x : ℕ → Bool) (o : ℕ) (ho : o < t.outputs.size) :
    (flatCircuit types words loOf F t).eval (fun g => if g.val < t.inBits then x g.val else true)
      ⟨(flatten types words loOf F t).2.getD o 0, (flatten_topo types words loOf F t).2 o⟩ =
      (evalT types words (F + 1) t x).getD o false :=
  (toCircuit_eval _ _ _ (fun i => if i < t.inBits then x i else true) _).trans (flatten_eval types words loOf F t x o ho)

end FlockSoundness.Types.Flatten
