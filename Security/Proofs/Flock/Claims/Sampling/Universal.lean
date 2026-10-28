import Definitions.Claims.Sampling

/-!
# The universal unit's gates run its private circuit

`universalProg c`: the universal unit of the size class `c` (`verity.ml.private_circuits.universal`, `build`) as a
program of the Boolean profile. Its inputs are the private circuit's `(op, src, out)` arguments and then `x`, in leaf
order (`PrivateCircuit.args`); after them, the constant `0` that fills every multiplexer tree. Each source and each
output is a multiplexer tree over the wires it may read (`gather.gather`), selecting by the index's bits least
significant first, a word without a partner meeting the fill; each of the private circuit's gates is
`s ⊕ t·(a·b ⊕ s) ⊕ n` with `s = a ⊕ b`, `t` and `n` its op's bits.

* `universal_correct`: **the unit runs its private circuit**: on a private circuit of class `c` and inputs of its
  width, `universalProg c` returns `PrivateCircuit.eval`.
* `UniversalIs D`: the deployed unit computes what `universalProg` does, at every class and input. It is the code
  side's: the compiled unit's binding to its Definition.
* `Universal_of`: `UniversalIs D` gives the claim `Universal D`.
-/

namespace Claims.Sampling

namespace Univ

/-! ## The builders -/

/-- `s ? hi : lo` on wires, `lo ⊕ s·(hi ⊕ lo)` (`forms.mux`): three gates after `gs`, the last its value. -/
def mux (gs : List BGate) (s hi lo : ℕ) : List BGate × ℕ :=
  (gs ++ [.xor [hi, lo], .and [s, gs.length], .xor [lo, gs.length + 1]], gs.length + 2)

/-- One level of the multiplexer tree at the index bit `s`: words `2i` and `2i + 1` meet in a `mux`, and a word without
a partner meets the fill wire `z`. -/
def level (s z : ℕ) : List BGate → List ℕ → List BGate × List ℕ
  | gs, [] => (gs, [])
  | gs, [a] => ((mux gs s z a).1, [(mux gs s z a).2])
  | gs, a :: b :: r => ((level s z (mux gs s b a).1 r).1, (mux gs s b a).2 :: (level s z (mux gs s b a).1 r).2)

/-- **The multiplexer tree** (`gather.gather`): the word the index bits `idx` select among `words`, least significant
bit first; past the end, the fill wire `z`. -/
def gather (z : ℕ) : List ℕ → List BGate → List ℕ → List BGate × ℕ
  | [], gs, words => (gs, words.headD z)
  | s :: ss, gs, words => gather z ss (level s z gs words).1 (level s z gs words).2

/-- A gate of the private circuit on the wires `a` and `b`, its op's bits `t` (AND over XOR) and `n` (negate):
`s ⊕ t·(a·b ⊕ s) ⊕ n` with `s = a ⊕ b` (`build`). -/
def gate (gs : List BGate) (t n a b : ℕ) : List BGate × ℕ :=
  (gs ++ [.xor [a, b], .and [a, b], .xor [gs.length + 1, gs.length], .and [t, gs.length + 2],
    .xor [gs.length, gs.length + 3, n]], gs.length + 4)

/-! ## The layout -/

/-- Where gate `g`'s source `j` starts among the inputs: after the `2G` op bits, `K` bits per source. -/
def srcAt (c : SizeClass) (g j : ℕ) : ℕ := 2 * c.G + c.indexBits * (2 * g + j)

/-- Where output `o`'s index starts: after the sources. -/
def outAt (c : SizeClass) (o : ℕ) : ℕ := 2 * c.G + c.indexBits * (2 * c.G + o)

/-- Where `x` starts: after the outputs. -/
def xAt (c : SizeClass) : ℕ := 2 * c.G + c.indexBits * (2 * c.G + c.nOut)

/-- The inputs: the arguments and `x`. -/
def nArgs (c : SizeClass) : ℕ := xAt c + c.nIn

/-- `K` wires from `base`. -/
def bitWires (base K : ℕ) : List ℕ := (List.range K).map (base + ·)

/-- Gate `g` of the private circuit: its two sources' trees over the wires so far, its function, and its wire. -/
def gStep (c : SizeClass) (st : List BGate × List ℕ) (g : ℕ) : List BGate × List ℕ :=
  let a := gather (nArgs c) (bitWires (srcAt c g 0) c.indexBits) st.1 st.2
  let b := gather (nArgs c) (bitWires (srcAt c g 1) c.indexBits) a.1 st.2
  let o := gate b.1 (2 * g) (2 * g + 1) a.2 b.2
  (o.1, st.2 ++ [o.2])

/-- Output `o`: its tree over every wire `ws`. -/
def oStep (c : SizeClass) (ws : List ℕ) (st : List BGate × List ℕ) (o : ℕ) : List BGate × List ℕ :=
  let r := gather (nArgs c) (bitWires (outAt c o) c.indexBits) st.1 ws
  (r.1, st.2 ++ [r.2])

/-- The inputs and the fill, and `x`'s wires. -/
def init (c : SizeClass) : List BGate × List ℕ :=
  (List.replicate (nArgs c) .input ++ [.const false], bitWires (xAt c) c.nIn)

/-- After every gate of the private circuit: the gates and its wires. -/
def body (c : SizeClass) : List BGate × List ℕ := (List.range c.G).foldl (gStep c) (init c)

/-- After every output: the gates and the returned wires. -/
def outs (c : SizeClass) : List BGate × List ℕ := (List.range c.nOut).foldl (oStep c (body c).2) ((body c).1, [])

end Univ

/-- **The universal unit of class `c`** (`UniversalUnit_v1{G, N_IN, N_OUT}`), as a program. -/
def universalProg (c : SizeClass) : Prog := ⟨(Univ.outs c).1, (Univ.outs c).2⟩

/-- **The deployed universal unit is the Definition's**: at every class, it computes what `universalProg` does. -/
def UniversalIs {Coin Col T : Type} (D : Deploy Coin Col T) : Prop :=
  ∀ c y, (D.universal c).eval y = (universalProg c).eval y

namespace Univ

/-! ## Wires -/

/-- Wire `i`'s value when `gs` runs on `y`. -/
def V (gs : List BGate) (y : List Bool) (i : ℕ) : Bool := (Prog.wires ⟨gs, []⟩ y).getD i false

theorem wires_outs (gs : List BGate) (o : List ℕ) (y : List Bool) :
    Prog.wires ⟨gs, o⟩ y = Prog.wires ⟨gs, []⟩ y := rfl

theorem foldl_fst_append {α : Type} (f : List Bool × ℕ → α → List Bool × ℕ)
    (hf : ∀ st g, ∃ v, (f st g).1 = st.1 ++ [v]) :
    ∀ (l : List α) (st : List Bool × ℕ), ∃ t, (l.foldl f st).1 = st.1 ++ t ∧ t.length = l.length
  | [], st => ⟨[], by simp⟩
  | g :: l, st => by
    obtain ⟨v, hv⟩ := hf st g
    obtain ⟨t, ht, hl⟩ := foldl_fst_append f hf l (f st g)
    exact ⟨v :: t, by rw [List.foldl_cons, ht, hv]; simp, by simp [hl]⟩

theorem wires_append (gs more : List BGate) (y : List Bool) :
    ∃ t, Prog.wires ⟨gs ++ more, []⟩ y = Prog.wires ⟨gs, []⟩ y ++ t ∧ t.length = more.length := by
  unfold Prog.wires
  rw [List.foldl_append]
  exact foldl_fst_append _ (fun st g => by cases g <;> exact ⟨_, rfl⟩) more _

theorem wires_length (gs : List BGate) (y : List Bool) : (Prog.wires ⟨gs, []⟩ y).length = gs.length := by
  obtain ⟨t, ht, hl⟩ := wires_append [] gs y
  simp only [List.nil_append] at ht
  have h0 : Prog.wires ⟨[], []⟩ y = [] := rfl
  rw [ht, h0, List.nil_append, hl]

theorem V_append {gs : List BGate} (more : List BGate) (y : List Bool) {i : ℕ} (h : i < gs.length) :
    V (gs ++ more) y i = V gs y i := by
  obtain ⟨t, ht, -⟩ := wires_append gs more y
  unfold V
  rw [ht, List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD,
    List.getElem?_append_left (by rw [wires_length]; exact h)]

theorem wires_snoc (gs : List BGate) (g : BGate) (hg : g ≠ .input) (y : List Bool) :
    Prog.wires ⟨gs ++ [g], []⟩ y =
      Prog.wires ⟨gs, []⟩ y ++ [(g.val (fun i => some (V gs y i)) none).getD false] := by
  unfold Prog.wires V Prog.wires
  rw [List.foldl_append, List.foldl_cons, List.foldl_nil]
  cases g with
  | input => exact absurd rfl hg
  | _ => rfl

theorem V_snoc (gs : List BGate) (g : BGate) (hg : g ≠ .input) (y : List Bool) :
    V (gs ++ [g]) y gs.length = (g.val (fun i => some (V gs y i)) none).getD false := by
  rw [V, wires_snoc gs g hg, List.getD_eq_getElem?_getD,
    List.getElem?_append_right (by rw [wires_length]), wires_length]
  simp

theorem mapM_some {α β : Type} (f : α → β) : ∀ l : List α, l.mapM (fun a => some (f a)) = some (l.map f)
  | [] => rfl
  | a :: l => by
    rw [List.mapM_cons, mapM_some f l]
    rfl

theorem V_xor (gs : List BGate) (ws : List ℕ) (y : List Bool) :
    V (gs ++ [.xor ws]) y gs.length = (ws.map (V gs y)).foldl Bool.xor false := by
  rw [V_snoc gs _ (by simp) y]
  simp only [BGate.val, mapM_some, Option.map_some, Option.getD_some]

theorem V_and (gs : List BGate) (ws : List ℕ) (y : List Bool) :
    V (gs ++ [.and ws]) y gs.length = (ws.map (V gs y)).all id := by
  rw [V_snoc gs _ (by simp) y]
  simp only [BGate.val, mapM_some, Option.map_some, Option.getD_some]

theorem V_const (gs : List BGate) (b : Bool) (y : List Bool) : V (gs ++ [.const b]) y gs.length = b := by
  rw [V_snoc gs _ (by simp) y]
  rfl

theorem V_xor' {gs : List BGate} (ws : List ℕ) (y : List Bool) {n : ℕ} (hn : gs.length = n) :
    V (gs ++ [.xor ws]) y n = (ws.map (V gs y)).foldl Bool.xor false := hn ▸ V_xor gs ws y

theorem V_and' {gs : List BGate} (ws : List ℕ) (y : List Bool) {n : ℕ} (hn : gs.length = n) :
    V (gs ++ [.and ws]) y n = (ws.map (V gs y)).all id := hn ▸ V_and gs ws y

/-- `gs'` extends `gs`. -/
def Ext (gs gs' : List BGate) : Prop := ∃ more, gs' = gs ++ more

theorem Ext.refl (gs : List BGate) : Ext gs gs := ⟨[], by simp⟩

theorem Ext.trans {a b c : List BGate} (h1 : Ext a b) (h2 : Ext b c) : Ext a c := by
  obtain ⟨m1, rfl⟩ := h1
  obtain ⟨m2, rfl⟩ := h2
  exact ⟨m1 ++ m2, by simp⟩

theorem Ext.le {a b : List BGate} (h : Ext a b) : a.length ≤ b.length := by
  obtain ⟨m, rfl⟩ := h
  simp

theorem Ext.V {a b : List BGate} (h : Ext a b) (y : List Bool) {i : ℕ} (hi : i < a.length) : V b y i = V a y i := by
  obtain ⟨m, rfl⟩ := h
  exact V_append m y hi

/-! ## The multiplexer and its tree -/

theorem mux_spec (gs : List BGate) (y : List Bool) {s hi lo : ℕ} (hs : s < gs.length) (hl : lo < gs.length) :
    Ext gs (mux gs s hi lo).1 ∧ (mux gs s hi lo).2 < (mux gs s hi lo).1.length ∧
      V (mux gs s hi lo).1 y (mux gs s hi lo).2 = if V gs y s then V gs y hi else V gs y lo := by
  refine ⟨⟨_, rfl⟩, by simp [mux], ?_⟩
  have e : (mux gs s hi lo).1 = ((gs ++ [.xor [hi, lo]]) ++ [.and [s, gs.length]]) ++ [.xor [lo, gs.length + 1]] := by
    simp [mux]
  have v1 : V (gs ++ [.xor [hi, lo]]) y gs.length = (V gs y hi ^^ V gs y lo) := by
    rw [V_xor]
    simp
  have v2 : V ((gs ++ [.xor [hi, lo]]) ++ [.and [s, gs.length]]) y (gs.length + 1) =
      (V gs y s && (V gs y hi ^^ V gs y lo)) := by
    rw [V_and' _ y (by simp)]
    simp only [List.map_cons, List.map_nil, List.all_cons, List.all_nil, id, Bool.and_true]
    rw [V_append _ y hs, v1]
  have v3 : V (((gs ++ [.xor [hi, lo]]) ++ [.and [s, gs.length]]) ++ [.xor [lo, gs.length + 1]]) y (gs.length + 2) =
      (V gs y lo ^^ (V gs y s && (V gs y hi ^^ V gs y lo))) := by
    rw [V_xor' _ y (by simp)]
    simp only [List.map_cons, List.map_nil, List.foldl_cons, List.foldl_nil, Bool.false_xor]
    rw [v2, V_append _ y (by simp; omega), V_append _ y hl]
  rw [e, show (mux gs s hi lo).2 = gs.length + 2 from rfl, v3]
  cases V gs y s <;> cases V gs y hi <;> cases V gs y lo <;> rfl

/-- One level of the tree on values: words `2i` and `2i + 1` by the bit `b`, a word without a partner against `0`. -/
def selLevel (b : Bool) : List Bool → List Bool
  | [] => []
  | [a] => [if b then false else a]
  | a :: c :: r => (if b then c else a) :: selLevel b r

theorem selLevel_getD (b : Bool) : ∀ (l : List Bool) (i : ℕ),
    (selLevel b l).getD i false = l.getD (2 * i + b.toNat) false
  | [], i => by simp [selLevel]
  | [a], i => by
    cases i with
    | zero => cases b <;> simp [selLevel]
    | succ i => cases b <;> simp [selLevel]
  | a :: c :: r, i => by
    cases i with
    | zero => cases b <;> simp [selLevel]
    | succ i =>
      have := selLevel_getD b r i
      simp only [List.getD_eq_getElem?_getD] at this ⊢
      simp only [selLevel, List.getElem?_cons_succ]
      rw [this, show 2 * (i + 1) + b.toNat = (2 * i + b.toNat) + 1 + 1 by omega]
      simp

theorem selLevel_length (b : Bool) : ∀ l : List Bool, (selLevel b l).length = (l.length + 1) / 2
  | [] => rfl
  | [_] => by simp [selLevel]
  | _ :: _ :: r => by
    simp only [selLevel, List.length_cons, selLevel_length b r]
    omega

theorem level_spec (y : List Bool) {s z : ℕ} :
    ∀ (gs : List BGate) (words : List ℕ), s < gs.length → z < gs.length → V gs y z = false →
      (∀ a ∈ words, a < gs.length) →
      Ext gs (level s z gs words).1 ∧ (∀ a ∈ (level s z gs words).2, a < (level s z gs words).1.length) ∧
        (level s z gs words).2.map (V (level s z gs words).1 y) = selLevel (V gs y s) (words.map (V gs y))
  | gs, [], _, _, _, _ => ⟨Ext.refl gs, by simp [level], by simp [level, selLevel]⟩
  | gs, [a], hs, hz, hz0, hw => by
    obtain ⟨he, hlt, hv⟩ := mux_spec (hi := z) gs y hs (hw a (by simp))
    refine ⟨he, by simpa [level] using hlt, ?_⟩
    simp only [level, List.map_cons, List.map_nil, selLevel, hv, hz0]
  | gs, a :: b :: r, hs, hz, hz0, hw => by
    obtain ⟨he, hlt, hv⟩ := mux_spec (s := s) (hi := b) gs y hs (hw a (by simp))
    have hs' := lt_of_lt_of_le hs he.le
    have hz' := lt_of_lt_of_le hz he.le
    have hr : ∀ a' ∈ r, a' < (mux gs s b a).1.length := fun a' ha' =>
      lt_of_lt_of_le (hw a' (by simp [ha'])) he.le
    obtain ⟨he2, hlt2, hv2⟩ := level_spec y (mux gs s b a).1 r hs' hz' (by rw [he.V y hz]; exact hz0) hr
    refine ⟨he.trans he2, ?_, ?_⟩
    · intro a' ha'
      simp only [level, List.mem_cons] at ha'
      rcases ha' with rfl | ha'
      · exact lt_of_lt_of_le hlt he2.le
      · exact hlt2 a' ha'
    · simp only [level, List.map_cons, selLevel]
      rw [hv2, he2.V y hlt, hv, he.V y hs]
      congr 2
      exact List.map_congr_left fun a' ha' => he.V y (hw a' (by simp [ha']))

/-- A bit string's value, least significant bit first. -/
def bitsVal : List Bool → ℕ
  | [] => 0
  | b :: bs => b.toNat + 2 * bitsVal bs

theorem gather_spec (y : List Bool) {z : ℕ} :
    ∀ (ss : List ℕ) (gs : List BGate) (words : List ℕ), (∀ s ∈ ss, s < gs.length) → z < gs.length →
      V gs y z = false → (∀ a ∈ words, a < gs.length) → words.length ≤ 2 ^ ss.length →
      Ext gs (gather z ss gs words).1 ∧ (gather z ss gs words).2 < (gather z ss gs words).1.length ∧
        V (gather z ss gs words).1 y (gather z ss gs words).2 =
          (words.map (V gs y)).getD (bitsVal (ss.map (V gs y))) false
  | [], gs, words, _, hz, hz0, hw, hlen => by
    refine ⟨Ext.refl gs, ?_, ?_⟩
    · cases words with
      | nil => simpa [gather] using hz
      | cons a _ => simpa [gather] using hw a (by simp)
    · match words, hlen with
      | [], _ => simpa [gather, bitsVal] using hz0
      | [a], _ => simp [gather, bitsVal, V]
      | _ :: _ :: _, h => simp at h
  | s :: ss, gs, words, hss, hz, hz0, hw, hlen => by
    have hs : s < gs.length := hss s (by simp)
    obtain ⟨he, hlt, hv⟩ := level_spec y gs words hs hz hz0 hw
    have hlen' : (level s z gs words).2.length ≤ 2 ^ ss.length := by
      have := congrArg List.length hv
      rw [List.length_map, selLevel_length, List.length_map] at this
      rw [this]
      rw [List.length_cons, pow_succ] at hlen
      omega
    obtain ⟨he2, hlt2, hv2⟩ := gather_spec y ss (level s z gs words).1 (level s z gs words).2
      (fun s' hs' => lt_of_lt_of_le (hss s' (by simp [hs'])) he.le) (lt_of_lt_of_le hz he.le)
      (by rw [he.V y hz]; exact hz0) hlt hlen'
    refine ⟨he.trans he2, hlt2, ?_⟩
    simp only [gather]
    rw [hv2, hv, selLevel_getD]
    have hm : ss.map (V (level s z gs words).1 y) = ss.map (V gs y) :=
      List.map_congr_left fun s' hs' => he.V y (hss s' (by simp [hs']))
    rw [hm]
    simp only [List.map_cons, bitsVal]
    congr 1
    omega

theorem bitsVal_bitsLE : ∀ (K v : ℕ), bitsVal (bitsLE K v) = v % 2 ^ K
  | 0, v => by simp [bitsLE, bitsVal, Nat.mod_one]
  | K + 1, v => by
    have h : bitsLE (K + 1) v = v.testBit 0 :: bitsLE K (v / 2) := by
      simp only [bitsLE, List.range_succ_eq_map, List.map_cons, List.map_map]
      congr 1
      refine List.map_congr_left fun i _ => ?_
      simp [Function.comp, Nat.testBit_succ]
    have h0 : (v.testBit 0).toNat = v % 2 := by
      rw [Nat.testBit_zero]
      rcases Nat.mod_two_eq_zero_or_one v with h | h <;> simp [h]
    rw [h, bitsVal, bitsVal_bitsLE K (v / 2), Nat.pow_succ, Nat.mul_comm (2 ^ K) 2, Nat.mod_mul, h0]

/-! ## The private circuit's gates -/

theorem gate_spec (gs : List BGate) (y : List Bool) {t nb a b : ℕ} (ht : t < gs.length) (hn : nb < gs.length)
    (ha : a < gs.length) (hb : b < gs.length) :
    Ext gs (gate gs t nb a b).1 ∧ (gate gs t nb a b).2 < (gate gs t nb a b).1.length ∧
      V (gate gs t nb a b).1 y (gate gs t nb a b).2 =
        xor (if V gs y t then V gs y a && V gs y b else xor (V gs y a) (V gs y b)) (V gs y nb) := by
  refine ⟨⟨_, rfl⟩, by simp [gate], ?_⟩
  have e1 : Ext gs (gs ++ [.xor [a, b]]) := ⟨_, rfl⟩
  have e2 : Ext gs ((gs ++ [.xor [a, b]]) ++ [.and [a, b]]) := e1.trans ⟨_, rfl⟩
  have e3 : Ext gs (((gs ++ [.xor [a, b]]) ++ [.and [a, b]]) ++ [.xor [gs.length + 1, gs.length]]) := e2.trans ⟨_, rfl⟩
  have e4 : Ext gs ((((gs ++ [.xor [a, b]]) ++ [.and [a, b]]) ++ [.xor [gs.length + 1, gs.length]]) ++
      [.and [t, gs.length + 2]]) := e3.trans ⟨_, rfl⟩
  have e : (gate gs t nb a b).1 = ((((gs ++ [.xor [a, b]]) ++ [.and [a, b]]) ++ [.xor [gs.length + 1, gs.length]]) ++
      [.and [t, gs.length + 2]]) ++ [.xor [gs.length, gs.length + 3, nb]] := by
    simp [gate]
  have v1 : V (gs ++ [.xor [a, b]]) y gs.length = (V gs y a ^^ V gs y b) := by
    rw [V_xor]
    simp
  have v2 : V ((gs ++ [.xor [a, b]]) ++ [.and [a, b]]) y (gs.length + 1) = (V gs y a && V gs y b) := by
    rw [V_and' _ y (by simp)]
    simp only [List.map_cons, List.map_nil, List.all_cons, List.all_nil, id, Bool.and_true]
    rw [e1.V y ha, e1.V y hb]
  have v3 : V (((gs ++ [.xor [a, b]]) ++ [.and [a, b]]) ++ [.xor [gs.length + 1, gs.length]]) y (gs.length + 2) =
      ((V gs y a && V gs y b) ^^ (V gs y a ^^ V gs y b)) := by
    rw [V_xor' _ y (by simp)]
    simp only [List.map_cons, List.map_nil, List.foldl_cons, List.foldl_nil, Bool.false_xor]
    rw [v2, V_append _ y (by simp), v1]
  have v4 : V ((((gs ++ [.xor [a, b]]) ++ [.and [a, b]]) ++ [.xor [gs.length + 1, gs.length]]) ++
      [.and [t, gs.length + 2]]) y (gs.length + 3) =
      (V gs y t && ((V gs y a && V gs y b) ^^ (V gs y a ^^ V gs y b))) := by
    rw [V_and' _ y (by simp)]
    simp only [List.map_cons, List.map_nil, List.all_cons, List.all_nil, id, Bool.and_true]
    rw [v3, e3.V y ht]
  have v5 : V ((((gs ++ [.xor [a, b]]) ++ [.and [a, b]]) ++ [.xor [gs.length + 1, gs.length]]) ++
      [.and [t, gs.length + 2]] ++ [.xor [gs.length, gs.length + 3, nb]]) y (gs.length + 4) =
      ((V gs y a ^^ V gs y b) ^^ (V gs y t && ((V gs y a && V gs y b) ^^ (V gs y a ^^ V gs y b))) ^^ V gs y nb) := by
    rw [V_xor' _ y (by simp)]
    simp only [List.map_cons, List.map_nil, List.foldl_cons, List.foldl_nil, Bool.false_xor]
    rw [v4, e4.V y hn, V_append _ y (by simp), V_append _ y (by simp), V_append _ y (by simp), v1]
  rw [e, show (gate gs t nb a b).2 = gs.length + 4 from rfl, v5]
  cases V gs y t <;> cases V gs y a <;> cases V gs y b <;> cases V gs y nb <;> rfl

/-! ## The inputs -/

theorem foldl_inputs {f : List Bool × ℕ → BGate → List Bool × ℕ} (y : List Bool)
    (hf : ∀ st, f st .input = (st.1 ++ [y.getD st.2 false], st.2 + 1)) :
    ∀ n, (List.replicate n BGate.input).foldl f ([], 0) = ((List.range n).map fun i => y.getD i false, n)
  | 0 => rfl
  | n + 1 => by
    rw [List.replicate_succ', List.foldl_append, foldl_inputs y hf n, List.foldl_cons, List.foldl_nil, hf,
      List.range_succ, List.map_append]
    rfl

theorem V_inputs (n : ℕ) (y : List Bool) {i : ℕ} (hi : i < n) : V (List.replicate n .input) y i = y.getD i false := by
  unfold V Prog.wires
  rw [foldl_inputs y ?_ n]
  · simp [hi]
  · intro st
    rfl

theorem init_length (c : SizeClass) : (init c).1.length = nArgs c + 1 := by
  simp [init]

theorem V_init (c : SizeClass) (y : List Bool) {i : ℕ} (hi : i < nArgs c) : V (init c).1 y i = y.getD i false := by
  show V (List.replicate (nArgs c) .input ++ [.const false]) y i = _
  rw [V_append _ y (by simpa using hi), V_inputs _ y hi]

theorem V_init_fill (c : SizeClass) (y : List Bool) : V (init c).1 y (nArgs c) = false := by
  have h := V_const (List.replicate (nArgs c) .input) false y
  rwa [List.length_replicate] at h

/-! ## The arguments -/

theorem length_flatMap_uniform {α : Type} (f : α → List Bool) {m : ℕ} (hf : ∀ a, (f a).length = m) :
    ∀ l : List α, (l.flatMap f).length = l.length * m
  | [] => by simp
  | a :: l => by
    rw [List.flatMap_cons, List.length_append, hf, length_flatMap_uniform f hf l, List.length_cons, Nat.succ_mul,
      Nat.add_comm]

theorem getD_flatMap {α : Type} (f : α → List Bool) {m : ℕ} (hf : ∀ a, (f a).length = m) :
    ∀ (l : List α) {g : ℕ} {a : α}, l[g]? = some a → ∀ {i : ℕ}, i < m →
      (l.flatMap f).getD (m * g + i) false = (f a).getD i false
  | [], _, _, h, _, _ => by simp at h
  | b :: l, 0, a, h, i, hi => by
    simp only [List.getElem?_cons_zero, Option.some.injEq] at h
    subst h
    rw [List.flatMap_cons, List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, Nat.mul_zero, Nat.zero_add,
      List.getElem?_append_left (by rw [hf]; exact hi)]
  | b :: l, g + 1, a, h, i, hi => by
    simp only [List.getElem?_cons_succ] at h
    have hm : m * (g + 1) + i - m = m * g + i := by rw [Nat.mul_succ]; omega
    rw [List.flatMap_cons, List.getD_eq_getElem?_getD, List.getElem?_append_right (by rw [hf, Nat.mul_succ]; omega),
      hf, hm, ← List.getD_eq_getElem?_getD]
    exact getD_flatMap f hf l h hi

theorem bitsLE_length (K v : ℕ) : (bitsLE K v).length = K := by simp [bitsLE]

theorem bitsLE_getD {K v i : ℕ} (hi : i < K) : (bitsLE K v).getD i false = v.testBit i := by
  simp [bitsLE, hi]

theorem lenA (l : List (Fin 4 × ℕ × ℕ)) : (l.flatMap fun g => bitsLE 2 g.1.val).length = 2 * l.length := by
  rw [length_flatMap_uniform (fun g : Fin 4 × ℕ × ℕ => bitsLE 2 g.1.val) (fun g => bitsLE_length 2 _), Nat.mul_comm]

theorem lenB_uniform (K : ℕ) (g : Fin 4 × ℕ × ℕ) : (bitsLE K g.2.1 ++ bitsLE K g.2.2).length = 2 * K := by
  rw [List.length_append, bitsLE_length, bitsLE_length]
  ring

theorem lenB (K : ℕ) (l : List (Fin 4 × ℕ × ℕ)) :
    (l.flatMap fun g => bitsLE K g.2.1 ++ bitsLE K g.2.2).length = 2 * K * l.length := by
  rw [length_flatMap_uniform (fun g : Fin 4 × ℕ × ℕ => bitsLE K g.2.1 ++ bitsLE K g.2.2) (lenB_uniform K),
    Nat.mul_comm]

theorem lenC (K : ℕ) (l : List ℕ) : (l.flatMap (bitsLE K)).length = K * l.length := by
  rw [length_flatMap_uniform (bitsLE K) (bitsLE_length K), Nat.mul_comm]

theorem xAt_eq (c : SizeClass) : xAt c = 2 * c.G + 2 * c.indexBits * c.G + c.indexBits * c.nOut := by
  unfold xAt
  ring

section Args

variable {c : SizeClass} {n : PrivateCircuit} {x y : List Bool}

theorem args_length (hfit : n.Fits c) : (n.args c).length = xAt c := by
  unfold PrivateCircuit.args
  rw [List.length_append, List.length_append, lenA, lenB, lenC, hfit.2.1, hfit.2.2.1, xAt_eq]

theorem y_args (hfit : n.Fits c) (hy : y = n.args c ++ x) {i : ℕ} (hi : i < xAt c) :
    y.getD i false = (n.args c).getD i false := by
  rw [hy, List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD,
    List.getElem?_append_left (by rw [args_length hfit]; exact hi)]

theorem y_x (hfit : n.Fits c) (hy : y = n.args c ++ x) (i : ℕ) : y.getD (xAt c + i) false = x.getD i false := by
  rw [hy, List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD,
    List.getElem?_append_right (by rw [args_length hfit]; omega), args_length hfit, Nat.add_sub_cancel_left]

theorem y_op (hfit : n.Fits c) (hy : y = n.args c ++ x) {g : ℕ} {gt : Fin 4 × ℕ × ℕ} (hg : n.gates[g]? = some gt)
    {j : ℕ} (hj : j < 2) : y.getD (2 * g + j) false = gt.1.val.testBit j := by
  have hgl : g < c.G := hfit.2.1 ▸ (List.getElem?_eq_some_iff.1 hg).1
  have hx := xAt_eq c
  rw [y_args hfit hy (by omega), PrivateCircuit.args, List.append_assoc, List.getD_eq_getElem?_getD,
    List.getElem?_append_left (by rw [lenA, hfit.2.1]; omega), ← List.getD_eq_getElem?_getD,
    getD_flatMap (fun g : Fin 4 × ℕ × ℕ => bitsLE 2 g.1.val) (fun g => bitsLE_length 2 _) _ hg hj, bitsLE_getD hj]

theorem y_src (hfit : n.Fits c) (hy : y = n.args c ++ x) {g : ℕ} {gt : Fin 4 × ℕ × ℕ} (hg : n.gates[g]? = some gt)
    {i : ℕ} (hi : i < c.indexBits) :
    y.getD (srcAt c g 0 + i) false = gt.2.1.testBit i ∧ y.getD (srcAt c g 1 + i) false = gt.2.2.testBit i := by
  have hgl : g < c.G := hfit.2.1 ▸ (List.getElem?_eq_some_iff.1 hg).1
  have hidx : ∀ j, srcAt c g j + i = 2 * c.G + (2 * c.indexBits * g + (c.indexBits * j + i)) := fun j => by
    unfold srcAt
    ring
  have hjK : ∀ j, j < 2 → c.indexBits * j + i < 2 * c.indexBits := fun j hj => by
    interval_cases j <;> omega
  have hgK : 2 * c.indexBits * g + 2 * c.indexBits ≤ 2 * c.indexBits * c.G := by
    have := Nat.mul_le_mul_left (2 * c.indexBits) (show g + 1 ≤ c.G by omega)
    rwa [Nat.mul_succ] at this
  have hx := xAt_eq c
  have key : ∀ j, j < 2 → y.getD (srcAt c g j + i) false =
      (bitsLE c.indexBits gt.2.1 ++ bitsLE c.indexBits gt.2.2).getD (c.indexBits * j + i) false := fun j hj => by
    have := hjK j hj
    rw [y_args hfit hy (by rw [hidx]; omega), hidx, PrivateCircuit.args, List.getD_eq_getElem?_getD,
      List.getElem?_append_left (by rw [List.length_append, lenA, lenB, hfit.2.1]; omega),
      List.getElem?_append_right (by rw [lenA, hfit.2.1]; omega), lenA, hfit.2.1, Nat.add_sub_cancel_left,
      ← List.getD_eq_getElem?_getD]
    exact getD_flatMap (fun g : Fin 4 × ℕ × ℕ => bitsLE c.indexBits g.2.1 ++ bitsLE c.indexBits g.2.2)
      (lenB_uniform c.indexBits) _ hg this
  refine ⟨?_, ?_⟩
  · rw [key 0 (by omega), Nat.mul_zero, Nat.zero_add, List.getD_eq_getElem?_getD,
      List.getElem?_append_left (by rw [bitsLE_length]; exact hi), ← List.getD_eq_getElem?_getD, bitsLE_getD hi]
  · rw [key 1 (by omega), Nat.mul_one, List.getD_eq_getElem?_getD,
      List.getElem?_append_right (by rw [bitsLE_length]; omega), bitsLE_length, Nat.add_sub_cancel_left,
      ← List.getD_eq_getElem?_getD, bitsLE_getD hi]

theorem y_out (hfit : n.Fits c) (hy : y = n.args c ++ x) {o v : ℕ} (ho : n.outs[o]? = some v) {i : ℕ}
    (hi : i < c.indexBits) : y.getD (outAt c o + i) false = v.testBit i := by
  have hol : o < c.nOut := hfit.2.2.1 ▸ (List.getElem?_eq_some_iff.1 ho).1
  have hK : c.indexBits * o + i < c.indexBits * c.nOut := by
    have : c.indexBits * (o + 1) ≤ c.indexBits * c.nOut := Nat.mul_le_mul_left _ (by omega)
    rw [Nat.mul_succ] at this
    omega
  have hidx : outAt c o + i = (2 * c.G + 2 * c.indexBits * c.G) + (c.indexBits * o + i) := by
    unfold outAt
    ring
  have hx := xAt_eq c
  rw [y_args hfit hy (by rw [hidx]; omega), hidx, PrivateCircuit.args, List.getD_eq_getElem?_getD,
    List.getElem?_append_right (by rw [List.length_append, lenA, lenB, hfit.2.1]; omega), List.length_append, lenA,
    lenB, hfit.2.1, Nat.add_sub_cancel_left, ← List.getD_eq_getElem?_getD,
    getD_flatMap (bitsLE c.indexBits) (bitsLE_length c.indexBits) _ ho hi, bitsLE_getD hi]

end Args

/-! ## The private circuit's wires -/

/-- The private circuit's step on its wires (`PrivateCircuit.eval`). -/
def nStep (ws : List Bool) (g : Fin 4 × ℕ × ℕ) : List Bool :=
  ws ++ [xor (if g.1.val % 2 = 1 then ws.getD g.2.1 false && ws.getD g.2.2 false
    else xor (ws.getD g.2.1 false) (ws.getD g.2.2 false)) (decide (g.1.val / 2 = 1))]

theorem eval_eq (n : PrivateCircuit) (x : List Bool) :
    n.eval x = n.outs.map fun o => (n.gates.foldl nStep x).getD o false := rfl

theorem foldl_nStep_length : ∀ (l : List (Fin 4 × ℕ × ℕ)) (ws : List Bool),
    (l.foldl nStep ws).length = ws.length + l.length
  | [], ws => by simp
  | g :: l, ws => by
    rw [List.foldl_cons, foldl_nStep_length l]
    simp only [nStep, List.length_append, List.length_cons, List.length_nil]
    omega

theorem indexBits_ge (c : SizeClass) : c.nIn + c.G ≤ 2 ^ c.indexBits := by
  unfold SizeClass.indexBits
  have := Nat.lt_log2_self (n := c.nIn + c.G - 1)
  omega

theorem op_bits (op : Fin 4) : op.val.testBit 0 = decide (op.val % 2 = 1) ∧ op.val.testBit 1 = decide (op.val / 2 = 1) := by
  revert op
  decide

theorem bitWires_map (gs : List BGate) (y : List Bool) (base K v : ℕ)
    (h : ∀ i < K, V gs y (base + i) = v.testBit i) : (bitWires base K).map (V gs y) = bitsLE K v := by
  unfold bitWires bitsLE
  rw [List.map_map]
  exact List.map_congr_left fun i hi => h i (List.mem_range.1 hi)

/-- The private circuit's wires after its first `g` gates. -/
def nWires (n : PrivateCircuit) (x : List Bool) (g : ℕ) : List Bool := (n.gates.take g).foldl nStep x

/-- After `g` gates of the private circuit: the gates extend the inputs and the fill, and the wires so far carry the
private circuit's. -/
def BodyInv (c : SizeClass) (n : PrivateCircuit) (x y : List Bool) (g : ℕ) (st : List BGate × List ℕ) : Prop :=
  Ext (init c).1 st.1 ∧ (∀ w ∈ st.2, w < st.1.length) ∧ st.2.map (V st.1 y) = nWires n x g

section Body

variable {c : SizeClass} {n : PrivateCircuit} {x y : List Bool}

theorem body_init (hfit : n.Fits c) (hx : x.length = c.nIn) (hy : y = n.args c ++ x) : BodyInv c n x y 0 (init c) := by
  refine ⟨Ext.refl _, fun w hw => ?_, ?_⟩
  · simp only [init, bitWires, List.mem_map, List.mem_range] at hw
    obtain ⟨i, hi, rfl⟩ := hw
    rw [init_length]
    unfold nArgs
    omega
  · rw [nWires, List.take_zero, List.foldl_nil]
    show (bitWires (xAt c) c.nIn).map (V (init c).1 y) = x
    unfold bitWires
    rw [List.map_map]
    refine List.ext_getElem (by simp [hx]) fun i h1 h2 => ?_
    simp only [List.length_map, List.length_range] at h1
    rw [List.getElem_map, List.getElem_range, Function.comp_apply, V_init c y (by unfold nArgs; omega), y_x hfit hy,
      List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h2, Option.getD_some]

theorem nWires_succ {n : PrivateCircuit} {x : List Bool} {g : ℕ} {gt : Fin 4 × ℕ × ℕ} (hg : n.gates[g]? = some gt) :
    nWires n x (g + 1) = nStep (nWires n x g) gt := by
  rw [nWires, nWires, List.take_add_one, hg, Option.toList_some, List.foldl_append, List.foldl_cons, List.foldl_nil]

theorem gather_at {gs : List BGate} (hext : Ext (init c).1 gs) (ws : List ℕ) (hw : ∀ w ∈ ws, w < gs.length)
    (hlen : ws.length ≤ 2 ^ c.indexBits) (base v : ℕ) (hb : base + c.indexBits ≤ nArgs c)
    (hv : ∀ i < c.indexBits, y.getD (base + i) false = v.testBit i) (hvK : v < 2 ^ c.indexBits) :
    Ext gs (gather (nArgs c) (bitWires base c.indexBits) gs ws).1 ∧
      (gather (nArgs c) (bitWires base c.indexBits) gs ws).2 < (gather (nArgs c) (bitWires base c.indexBits) gs ws).1.length ∧
      V (gather (nArgs c) (bitWires base c.indexBits) gs ws).1 y (gather (nArgs c) (bitWires base c.indexBits) gs ws).2 =
        (ws.map (V gs y)).getD v false := by
  have hgs := hext.le
  rw [init_length] at hgs
  have hVi : ∀ i < c.indexBits, V gs y (base + i) = v.testBit i := fun i hi => by
    rw [hext.V y (by rw [init_length]; omega), V_init c y (by omega), hv i hi]
  obtain ⟨he, hlt, hval⟩ := gather_spec y (z := nArgs c) (bitWires base c.indexBits) gs ws
    (fun s hs => by
      simp only [bitWires, List.mem_map, List.mem_range] at hs
      obtain ⟨i, hi, rfl⟩ := hs
      omega)
    (by omega) (by rw [hext.V y (by rw [init_length]; omega), V_init_fill]) hw
    (by rw [bitWires, List.length_map, List.length_range]; exact hlen)
  refine ⟨he, hlt, ?_⟩
  rw [hval, bitWires_map gs y base _ v hVi, bitsVal_bitsLE, Nat.mod_eq_of_lt hvK]

theorem body_step (hfit : n.Fits c) (hx : x.length = c.nIn) (hy : y = n.args c ++ x) {g : ℕ} (hg : g < c.G)
    {st : List BGate × List ℕ} (hinv : BodyInv c n x y g st) : BodyInv c n x y (g + 1) (gStep c st g) := by
  obtain ⟨hext, hw, hv⟩ := hinv
  obtain ⟨gt, hgt⟩ : ∃ gt, n.gates[g]? = some gt := ⟨_, List.getElem?_eq_getElem (by rw [hfit.2.1]; exact hg)⟩
  have hgm : gt ∈ n.gates := List.mem_of_getElem? hgt
  have hlen : st.2.length ≤ 2 ^ c.indexBits := by
    have := congrArg List.length hv
    rw [List.length_map, nWires, foldl_nStep_length, List.length_take, hx] at this
    have := indexBits_ge c
    have : min g n.gates.length ≤ c.G := by rw [hfit.2.1]; omega
    omega
  have hK : ∀ j, j < 2 → srcAt c g j + c.indexBits ≤ nArgs c := fun j hj => by
    have : c.indexBits * (2 * g + j + 1) ≤ c.indexBits * (2 * c.G) := Nat.mul_le_mul_left _ (by omega)
    rw [Nat.mul_succ] at this
    unfold srcAt nArgs xAt
    rw [Nat.mul_add c.indexBits (2 * c.G)]
    omega
  have hgs := hext.le
  rw [init_length] at hgs
  obtain ⟨hea, hlta, hva⟩ := gather_at hext st.2 hw hlen (srcAt c g 0) gt.2.1 (hK 0 (by omega))
    (fun i hi => (y_src hfit hy hgt hi).1) (hfit.2.2.2.1 gt hgm).1
  have hext' := hext.trans hea
  have hw' : ∀ w ∈ st.2, w < (gather (nArgs c) (bitWires (srcAt c g 0) c.indexBits) st.1 st.2).1.length :=
    fun w h => lt_of_lt_of_le (hw w h) hea.le
  obtain ⟨heb, hltb, hvb⟩ := gather_at hext' st.2 hw' hlen (srcAt c g 1) gt.2.2 (hK 1 (by omega))
    (fun i hi => (y_src hfit hy hgt hi).2) (hfit.2.2.2.1 gt hgm).2
  have hgs' := hext'.trans heb |>.le
  rw [init_length] at hgs'
  have hop : 2 * g + 1 < nArgs c := by unfold nArgs xAt; omega
  obtain ⟨heo, hlto, hvo⟩ := gate_spec _ y (t := 2 * g) (nb := 2 * g + 1) (by omega) (by omega)
    (lt_of_lt_of_le hlta heb.le) hltb
  have hall := (hea.trans heb).trans heo
  have hmap : st.2.map (V (gather (nArgs c) (bitWires (srcAt c g 0) c.indexBits) st.1 st.2).1 y) = st.2.map (V st.1 y) :=
    List.map_congr_left fun w h => hea.V y (hw w h)
  refine ⟨hext.trans hall, fun w h => ?_, ?_⟩
  · simp only [gStep, List.mem_append, List.mem_singleton] at h
    rcases h with h | rfl
    · exact lt_of_lt_of_le (hw w h) hall.le
    · exact hlto
  · simp only [gStep, List.map_append, List.map_cons, List.map_nil]
    have h0 := y_op hfit hy hgt (j := 0) (by omega)
    rw [Nat.add_zero] at h0
    rw [List.map_congr_left fun w h => hall.V y (hw w h), hv, hvo, heb.V y hlta, hva, hvb, hmap, hv,
      (hea.trans heb).V y (by omega), (hea.trans heb).V y (by omega), hext.V y (by rw [init_length]; omega),
      hext.V y (by rw [init_length]; omega), V_init c y (by omega), V_init c y hop, h0,
      y_op hfit hy hgt (j := 1) (by omega), (op_bits gt.1).1, (op_bits gt.1).2, nWires_succ hgt, nStep]
    simp only [decide_eq_true_eq]

theorem body_inv (hfit : n.Fits c) (hx : x.length = c.nIn) (hy : y = n.args c ++ x) :
    ∀ g ≤ c.G, BodyInv c n x y g ((List.range g).foldl (gStep c) (init c))
  | 0, _ => body_init hfit hx hy
  | g + 1, hg => by
    rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
    exact body_step hfit hx hy (by omega) (body_inv hfit hx hy g (by omega))

end Body

section Outs

variable {c : SizeClass} {n : PrivateCircuit} {x y : List Bool}

/-- After `o` outputs: the gates extend the body's, and the outputs so far carry the private circuit's. -/
def OutInv (c : SizeClass) (n : PrivateCircuit) (x y : List Bool) (o : ℕ) (st : List BGate × List ℕ) : Prop :=
  Ext (body c).1 st.1 ∧ (∀ w ∈ st.2, w < st.1.length) ∧
    st.2.map (V st.1 y) = (n.outs.take o).map fun v => (n.gates.foldl nStep x).getD v false

theorem outs_inv (hfit : n.Fits c) (hx : x.length = c.nIn) (hy : y = n.args c ++ x) :
    ∀ o ≤ c.nOut, OutInv c n x y o ((List.range o).foldl (oStep c (body c).2) ((body c).1, []))
  | 0, _ => ⟨Ext.refl _, by simp, by simp⟩
  | o + 1, ho => by
    rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
    obtain ⟨hext, hw, hv⟩ := outs_inv hfit hx hy o (by omega)
    have hb : BodyInv c n x y c.G (body c) := body_inv hfit hx hy c.G le_rfl
    obtain ⟨hbe, hbw, hbv⟩ := hb
    have hfull : nWires n x c.G = n.gates.foldl nStep x := by
      rw [nWires, List.take_of_length_le (by rw [hfit.2.1])]
    obtain ⟨v, hov⟩ : ∃ v, n.outs[o]? = some v := ⟨_, List.getElem?_eq_getElem (by rw [hfit.2.2.1]; omega)⟩
    have hlen : (body c).2.length ≤ 2 ^ c.indexBits := by
      have := congrArg List.length hbv
      rw [List.length_map, hfull, foldl_nStep_length, hx, hfit.2.1] at this
      have := indexBits_ge c
      omega
    have hK : outAt c o + c.indexBits ≤ nArgs c := by
      have : c.indexBits * (o + 1) ≤ c.indexBits * c.nOut := Nat.mul_le_mul_left _ (by omega)
      rw [Nat.mul_succ] at this
      unfold outAt nArgs xAt
      rw [Nat.mul_add c.indexBits (2 * c.G), Nat.mul_add c.indexBits (2 * c.G)]
      omega
    obtain ⟨her, hltr, hvr⟩ := gather_at (hbe.trans hext) (body c).2 (fun w h => lt_of_lt_of_le (hbw w h) hext.le) hlen
      (outAt c o) v hK (fun i hi => y_out hfit hy hov hi) (hfit.2.2.2.2 v (List.mem_of_getElem? hov))
    refine ⟨hext.trans her, fun w h => ?_, ?_⟩
    · simp only [oStep, List.mem_append, List.mem_singleton] at h
      rcases h with h | rfl
      · exact lt_of_lt_of_le (hw w h) her.le
      · exact hltr
    simp only [oStep, List.map_append, List.map_cons, List.map_nil]
    rw [List.map_congr_left fun w h => her.V y (hw w h), hv, hvr,
      List.map_congr_left fun w h => hext.V y (hbw w h), hbv, hfull, List.take_add_one, hov, Option.toList_some,
      List.map_append, List.map_cons, List.map_nil]

end Outs

/-- **The universal unit runs its private circuit**: on a private circuit of class `c`, as its arguments, and inputs of
its width, `universalProg c` returns the private circuit's outputs. -/
theorem universal_correct (c : SizeClass) (n : PrivateCircuit) (x : List Bool) (hfit : n.Fits c) (hx : x.length = c.nIn) :
    (universalProg c).eval (n.args c ++ x) = n.eval x := by
  have ho : OutInv c n x (n.args c ++ x) c.nOut (outs c) := outs_inv hfit hx rfl c.nOut le_rfl
  obtain ⟨-, -, hv⟩ := ho
  rw [eval_eq, Prog.eval, universalProg, wires_outs]
  change (outs c).2.map (V (outs c).1 (n.args c ++ x)) = _
  rw [hv, List.take_of_length_le (by rw [hfit.2.2.1])]

end Univ

/-- **`Universal` from the unit's Definition**: a deployment whose universal unit computes what `universalProg` does
runs every private circuit of its class. -/
theorem Universal_of {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
    {D : Deploy Coin Col T} (h : UniversalIs D) : Universal D :=
  fun c n x hfit hx => (h c _).trans (Univ.universal_correct c n x hfit hx)

end Claims.Sampling
