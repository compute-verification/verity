import Proofs.Core.Boolean
import Proofs.Flock.Claims.Scope.Readable

/-!
# Block R1CS systems over GF(2), as the readable relation

A `Sys` is `J` tables of block R1CS over GF(2), each block `N` bits: table `c`'s row `i` reads the columns `A c i` and
`B c i` select, its pin is `pin c`, and each `slots` entry `(c, a, v)` fixes bit `a` of table `c`'s block `b` to `v b`.
`Sys.rel` is that system as `Relation`'s fields. The block circuit `circ` is a Boolean circuit: its input gates are every
table's block; then one row gate per row, the AND of the XOR of its table's bits that `A c i` selects and the XOR of those
`B c i` selects (`xsum`); then a copy of each pin's bit and of each fixed bit. Every gate after the inputs is an output.
Block `b`'s input gates read the hidden bits of every table's block `b` (table `c`'s bit `a` of block `b` is hidden value
`(b·N + a)·J + c`: the tables' bits interleaved, so where table `c`'s bits sit depends on no other table's); its row
gates' outputs are those hidden bits again, its pins' the public `1`, and its fixed bits' their public values.

`relS_iff`: a witness satisfies `Claims.Flock.RelS` of a statement with these fields exactly when, in every block, every
table's rows hold (`Sys.Holds`).
-/

namespace Flock.Readable

open Claims

/-- The XOR of `A r && x r` over `r < n`. -/
def xsum (A x : ℕ → Bool) (n : ℕ) : Bool := (List.range n).foldl (fun acc r => xor acc (A r && x r)) false

theorem xsum_succ (A x : ℕ → Bool) (n : ℕ) : xsum A x (n + 1) = xor (xsum A x n) (A n && x n) := by
  simp [xsum, List.range_succ, List.foldl_append]

/-- `J` tables of block R1CS over GF(2) with `N`-bit blocks and `nB` blocks: rows `A c i`, `B c i`, pins, and fixed bits
`(table, address, value in each block)`. -/
structure Sys where
  J : ℕ
  N : ℕ
  nB : ℕ
  A : ℕ → ℕ → ℕ → Bool
  B : ℕ → ℕ → ℕ → Bool
  pin : ℕ → ℕ
  slots : List (ℕ × ℕ × (ℕ → Bool))

namespace Sys

variable (S : Sys)

/-- The block circuit's gates: the `J·N` input bits, a row gate for each of the `J·N` rows, and a copy of each pin's bit
and of each fixed bit. -/
def size : ℕ := 2 * (S.J * S.N) + S.J + S.slots.length

/-- The input gates among `ts`: those below `J·N`. -/
def ins (ts : List ℕ) : List (Fin S.size) :=
  ts.filterMap fun t => if h : t < S.J * S.N then some ⟨t, by unfold size; omega⟩ else none

/-- Row `q = c·N + i`'s bits that `A c i` selects, in its table's block. -/
def sel (A : ℕ → ℕ → ℕ → Bool) (q : ℕ) : List ℕ :=
  ((List.range S.N).filter (A (q / S.N) (q % S.N))).map fun r => q / S.N * S.N + r

/-- The bit copy gate `p` copies: pin `p`'s bit for `p < J`, else fixed bit `p − J`'s. -/
def copied (p : ℕ) : List ℕ :=
  if p < S.J then [p * S.N + S.pin p] else (S.slots[p - S.J]?.map fun s => s.1 * S.N + s.2.1).toList

/-- Gate `g` of the block circuit: an input bit, row `g − J·N`'s AND of XORs, or a copy, the AND of one bit with
itself. -/
def gate (g : Fin S.size) : FlockSoundness.Audit.Op S.size :=
  if g.val < S.J * S.N then .input
  else if g.val < 2 * (S.J * S.N) then
    .row (S.ins (S.sel S.A (g.val - S.J * S.N))) (S.ins (S.sel S.B (g.val - S.J * S.N)))
  else .row (S.ins (S.copied (g.val - 2 * (S.J * S.N)))) (S.ins (S.copied (g.val - 2 * (S.J * S.N))))

theorem ins_lt {ts : List ℕ} {a : Fin S.size} (h : a ∈ S.ins ts) : a.val < S.J * S.N := by
  obtain ⟨t, -, e⟩ := List.mem_filterMap.1 h
  by_cases ht : t < S.J * S.N
  · rw [dite_eq_left ht] at e
    obtain rfl := Option.some.inj e
    exact ht
  · rw [dite_eq_right ht] at e
    cases e

theorem gate_lt (g : Fin S.size) : ∀ a ∈ (S.gate g).args, a < g := by
  intro a ha
  show a.val < g.val
  unfold gate at ha
  split_ifs at ha with h1 h2
  · simp [FlockSoundness.Audit.Op.args] at ha
  · simp only [FlockSoundness.Audit.Op.args, List.mem_append] at ha
    rcases ha with ha | ha <;> have := S.ins_lt ha <;> omega
  · simp only [FlockSoundness.Audit.Op.args, List.mem_append] at ha
    rcases ha with ha | ha <;> have := S.ins_lt ha <;> omega

/-- **The block circuit**: gates `0, …, J·N − 1` the input bits, then gate `J·N + q` for row `q = c·N + i`, the AND of the
XOR of its table's bits that `A c i` selects and the XOR of those `B c i` selects, then a copy of each pin's bit and of
each fixed bit. Every gate after the inputs is an output. -/
@[reducible] def circ : FlockSoundness.Audit.Circuit where
  N := S.size
  op := S.gate
  topo := S.gate_lt
  outputs := Finset.univ.filter fun g => S.J * S.N ≤ g.val

/-- Hidden value of table `q / N`'s bit `q % N` of block `b`. -/
def idx (b q : ℕ) : ℕ := (b * S.N + q % S.N) * S.J + q / S.N

/-- The public values: `1`, then each block's fixed bits (value `1 + b·L + k` is block `b`'s fixed bit `k`). -/
def pubs : List Bool :=
  true :: (List.range (S.nB * S.slots.length)).map fun j =>
    (S.slots[j % S.slots.length]?.map fun s => s.2.2 (j / S.slots.length)).getD false

/-- **The system as the readable relation's fields.** In block `b`, input gate `q` reads hidden value `idx b q`; row gate
`J·N + q` outputs that hidden value again, each pin's copy the public `1`, and fixed bit `k`'s copy public value
`1 + b·L + k`. -/
def rel : Relation where
  C := S.circ
  inp b q := .hid (S.idx b q)
  out b g :=
    if g < 2 * (S.J * S.N) then .hid (S.idx b (g - S.J * S.N))
    else if g < 2 * (S.J * S.N) + S.J then .pub 0
    else .pub (1 + b * S.slots.length + (g - 2 * (S.J * S.N) - S.J))
  pub := S.pubs

/-- Table `c`'s bit `a` of block `b` in `w`. -/
def bit (w : Claims.Flock.Wit) (c b a : ℕ) : Bool := w.hid.getD ((b * S.N + a) * S.J + c) false

/-- **The system holds in block `b`**: every table's rows, pins and fixed bits. -/
def Holds (w : Claims.Flock.Wit) (b : ℕ) : Prop :=
  (∀ c < S.J, ∀ i < S.N, (xsum (S.A c i) (S.bit w c b) S.N && xsum (S.B c i) (S.bit w c b) S.N) = S.bit w c b i) ∧
    (∀ c < S.J, S.bit w c b (S.pin c) = true) ∧ ∀ s ∈ S.slots, S.bit w s.1 b s.2.1 = s.2.2 b

/-- The system's shape: nonempty blocks, every pin and fixed bit in its table's block. -/
structure WF : Prop where
  N_pos : 0 < S.N
  pin_lt : ∀ c < S.J, S.pin c < S.N
  slot_lt : ∀ s ∈ S.slots, s.1 < S.J ∧ s.2.1 < S.N

end Sys

namespace Sys

variable {S : Sys}

/-- Row `q`'s AND on inputs `x`. -/
def res (S : Sys) (x : ℕ → Bool) (q : ℕ) : Bool :=
  xsum (S.A (q / S.N) (q % S.N)) (fun r => x (q / S.N * S.N + r)) S.N &&
    xsum (S.B (q / S.N) (q % S.N)) (fun r => x (q / S.N * S.N + r)) S.N

theorem table_le {Q : ℕ} (hN : 0 < S.N) (hQ : Q < S.J * S.N) : Q / S.N * S.N + S.N ≤ S.J * S.N := by
  have h1 : Q / S.N < S.J := (Nat.div_lt_iff_lt_mul hN).2 hQ
  have := Nat.mul_le_mul_right S.N (show Q / S.N + 1 ≤ S.J by omega)
  rw [Nat.succ_mul] at this
  exact this

theorem xsum_congr (A : ℕ → Bool) {x y : ℕ → Bool} {n : ℕ} (h : ∀ r < n, x r = y r) : xsum A x n = xsum A y n := by
  induction n with
  | zero => rfl
  | succ n ih => rw [xsum_succ, xsum_succ, ih fun r hr => h r (by omega), h n (by omega)]

theorem foldl_filter (p f : ℕ → Bool) (l : List ℕ) (c : Bool) :
    l.foldl (fun acc r => xor acc (p r && f r)) c = xor c (((l.filter p).map f).foldr xor false) := by
  induction l generalizing c with
  | nil => simp
  | cons r l ih =>
    rw [List.foldl_cons, ih]
    cases hp : p r <;> simp [hp]

section Eval

variable {a : S.circ.toCircuit.Transcript} {x : ℕ → Bool}

theorem bit_input {g : Fin S.size} (hg : g.val < S.J * S.N) :
    (S.circ.toCircuit.eval a g).getLsbD 0 = (a g).getLsbD 0 := by
  rw [Verity.Circuit.eval_input _ a (g := g)
    (by show (S.gate g).isInput = true; simp [gate, hg, FlockSoundness.Audit.Op.isInput])]

theorem bit_op {g : Fin S.size} (h : (S.gate g).isInput = false) :
    (S.circ.toCircuit.eval a g).getLsbD 0 = (S.gate g).apply fun y : Fin S.size => (S.circ.toCircuit.eval a y).getLsbD 0 :=
  FlockSoundness.Audit.eval_toCircuit S.circ a h

theorem xorSum_ins {v : Fin S.size → Bool} (hv : ∀ g : Fin S.size, g.val < S.J * S.N → v g = x g.val) {ts : List ℕ}
    (hts : ∀ t ∈ ts, t < S.J * S.N) : FlockSoundness.Audit.xorSum (S.ins ts) v = (ts.map x).foldr xor false := by
  induction ts with
  | nil => rfl
  | cons t ts ih =>
    have ht := hts t (by simp)
    have e : S.ins (t :: ts) = ⟨t, by unfold size; omega⟩ :: S.ins ts := by simp [ins, ht]
    rw [e]
    simp only [FlockSoundness.Audit.xorSum, List.map_cons, List.foldr_cons] at ih ⊢
    rw [ih fun t' h => hts t' (by simp [h]), hv _ ht]

theorem xorSum_sel (hN : 0 < S.N) {v : Fin S.size → Bool} (hv : ∀ g : Fin S.size, g.val < S.J * S.N → v g = x g.val)
    (A : ℕ → ℕ → ℕ → Bool) {q : ℕ} (hq : q < S.J * S.N) :
    FlockSoundness.Audit.xorSum (S.ins (S.sel A q)) v =
      xsum (A (q / S.N) (q % S.N)) (fun r => x (q / S.N * S.N + r)) S.N := by
  have hle := table_le hN hq
  rw [xorSum_ins hv fun t ht => ?_]
  · rw [xsum, foldl_filter, Bool.false_xor, sel, List.map_map]
    rfl
  · simp only [sel, List.mem_map, List.mem_filter, List.mem_range] at ht
    obtain ⟨r, ⟨hr, -⟩, rfl⟩ := ht
    omega

variable (hxa : ∀ g : Fin S.size, g.val < S.J * S.N → (a g).getLsbD 0 = x g.val)
include hxa

theorem bit_row (hN : 0 < S.N) {g : Fin S.size} (h1 : S.J * S.N ≤ g.val) (h2 : g.val < 2 * (S.J * S.N)) :
    (S.circ.toCircuit.eval a g).getLsbD 0 = S.res x (g.val - S.J * S.N) := by
  have hop : S.gate g = .row (S.ins (S.sel S.A (g.val - S.J * S.N))) (S.ins (S.sel S.B (g.val - S.J * S.N))) := by
    rw [gate, ite_eq_right (by omega), ite_eq_left h2]
  have hv : ∀ g' : Fin S.size, g'.val < S.J * S.N → (S.circ.toCircuit.eval a g').getLsbD 0 = x g'.val :=
    fun g' h => (bit_input h).trans (hxa g' h)
  rw [bit_op (by rw [hop]; rfl), hop]
  simp only [FlockSoundness.Audit.Op.apply]
  rw [xorSum_sel hN hv _ (by omega), xorSum_sel hN hv _ (by omega)]
  rfl

theorem bit_copy {g : Fin S.size} (h2 : 2 * (S.J * S.N) ≤ g.val)
    (hc : ∀ t ∈ S.copied (g.val - 2 * (S.J * S.N)), t < S.J * S.N) :
    (S.circ.toCircuit.eval a g).getLsbD 0 = ((S.copied (g.val - 2 * (S.J * S.N))).map x).foldr xor false := by
  have hop : S.gate g = .row (S.ins (S.copied (g.val - 2 * (S.J * S.N)))) (S.ins (S.copied (g.val - 2 * (S.J * S.N)))) := by
    rw [gate, ite_eq_right (by omega), ite_eq_right (by omega)]
  have hv : ∀ g' : Fin S.size, g'.val < S.J * S.N → (S.circ.toCircuit.eval a g').getLsbD 0 = x g'.val :=
    fun g' h => (bit_input h).trans (hxa g' h)
  rw [bit_op (by rw [hop]; rfl), hop]
  simp only [FlockSoundness.Audit.Op.apply]
  rw [xorSum_ins hv hc, Bool.and_self]

end Eval

theorem pubs_zero : S.pubs.getD 0 false = true := rfl

theorem pubs_slot {b k : ℕ} (hb : b < S.nB) (hk : k < S.slots.length) :
    S.pubs.getD (1 + b * S.slots.length + k) false = S.slots[k].2.2 b := by
  have hj : b * S.slots.length + k < S.nB * S.slots.length := by
    have := Nat.mul_le_mul_right S.slots.length (show b + 1 ≤ S.nB by omega)
    rw [Nat.succ_mul] at this
    omega
  have hL : 0 < S.slots.length := by omega
  rw [show 1 + b * S.slots.length + k = (b * S.slots.length + k) + 1 by omega, pubs, List.getD_cons_succ,
    List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hj]
  simp only [Option.map_some, Option.getD_some]
  rw [show b * S.slots.length + k = k + b * S.slots.length by omega, Nat.add_mul_mod_self_right,
    Nat.mod_eq_of_lt hk, Nat.add_mul_div_right _ _ hL, Nat.div_eq_of_lt hk, Nat.zero_add, List.getElem?_eq_getElem hk]
  rfl

theorem forall_rows (hN : 0 < S.N) (F : ℕ → ℕ → Prop) :
    (∀ q < S.J * S.N, F (q / S.N) (q % S.N)) ↔ ∀ c < S.J, ∀ i < S.N, F c i := by
  constructor
  · intro h c hc i hi
    have hq : c * S.N + i < S.J * S.N := by
      have := Nat.mul_le_mul_right S.N (show c + 1 ≤ S.J by omega)
      rw [Nat.succ_mul] at this
      omega
    have := h _ hq
    rwa [show c * S.N + i = i + c * S.N by omega, Nat.add_mul_div_right _ _ hN, Nat.div_eq_of_lt hi, Nat.zero_add,
      Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hi] at this
  · intro h q hq
    exact h _ ((Nat.div_lt_iff_lt_mul hN).2 hq) _ (Nat.mod_lt _ hN)

/-- The output gates, by kind: the row gates, the pins' copies and the fixed bits' copies. -/
theorem forall_outputs {P : Fin S.size → Prop} :
    (∀ o ∈ S.circ.outputs, P o) ↔ (∀ q (h : q < S.J * S.N), P ⟨S.J * S.N + q, by unfold size; omega⟩) ∧
      (∀ c (h : c < S.J), P ⟨2 * (S.J * S.N) + c, by unfold size; omega⟩) ∧
      ∀ k (h : k < S.slots.length), P ⟨2 * (S.J * S.N) + S.J + k, by unfold size; omega⟩ := by
  have hmem : ∀ o : Fin S.circ.N, o ∈ S.circ.outputs ↔ S.J * S.N ≤ o.val := fun o =>
    Finset.mem_filter.trans ⟨fun h => h.2, fun h => ⟨Finset.mem_univ _, h⟩⟩
  constructor
  · intro h
    exact ⟨fun q _ => h _ ((hmem _).2 (Nat.le_add_right _ _)), fun c _ => h _ ((hmem _).2 (by dsimp only; omega)),
      fun k _ => h _ ((hmem _).2 (by dsimp only; omega))⟩
  · rintro ⟨h1, h2, h3⟩ o ho
    have hlo := (hmem o).1 ho
    have hhi : o.val < S.size := o.isLt
    unfold size at hhi
    by_cases e1 : o.val < 2 * (S.J * S.N)
    · have := h1 (o.val - S.J * S.N) (by omega)
      rwa [show (⟨S.J * S.N + (o.val - S.J * S.N), _⟩ : Fin S.size) = o from Fin.ext (by dsimp only; omega)] at this
    · by_cases e2 : o.val < 2 * (S.J * S.N) + S.J
      · have := h2 (o.val - 2 * (S.J * S.N)) (by omega)
        rwa [show (⟨2 * (S.J * S.N) + (o.val - 2 * (S.J * S.N)), _⟩ : Fin S.size) = o from Fin.ext (by dsimp only; omega)]
          at this
      · have := h3 (o.val - 2 * (S.J * S.N) - S.J) (by omega)
        rwa [show (⟨2 * (S.J * S.N) + S.J + (o.val - 2 * (S.J * S.N) - S.J), _⟩ : Fin S.size) = o from
          Fin.ext (by dsimp only; omega)] at this

/-- **The block circuit's outputs**: on block `b`'s inputs, every output gate holds the value its output reads exactly
when the system holds in block `b`. -/
theorem outputs_iff (hw : S.WF) (w : Claims.Flock.Wit) {b : ℕ} (hb : b < S.nB) (a : S.circ.toCircuit.Transcript)
    (ha : ∀ g : Fin S.size, (a g).getLsbD 0 = w.hid.getD (S.idx b g.val) false) (val : Claims.Flock.Src → Bool)
    (hhid : ∀ i, val (.hid i) = w.hid.getD i false) (hpub : ∀ i, val (.pub i) = S.pubs.getD i false) :
    (∀ o ∈ S.circ.outputs, (S.circ.toCircuit.eval a o).getLsbD 0 = val (S.rel.out b o.val)) ↔ S.Holds w b := by
  have hN := hw.N_pos
  let x : ℕ → Bool := fun q => w.hid.getD (S.idx b q) false
  have hxa : ∀ g : Fin S.size, g.val < S.J * S.N → (a g).getLsbD 0 = x g.val := fun g _ => ha g
  have hx : ∀ c r, r < S.N → x (c * S.N + r) = S.bit w c b r := by
    intro c r hr
    simp only [x, idx, bit]
    rw [show c * S.N + r = r + c * S.N by omega, Nat.add_mul_div_right _ _ hN, Nat.div_eq_of_lt hr, Nat.zero_add,
      Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hr]
  have hcopy : ∀ c < S.J, ∀ r < S.N, c * S.N + r < S.J * S.N := fun c hc r hr => by
    have := Nat.mul_le_mul_right S.N (show c + 1 ≤ S.J by omega)
    rw [Nat.succ_mul] at this
    omega
  refine (forall_outputs (P := fun o => (S.circ.toCircuit.eval a o).getLsbD 0 = val (S.rel.out b o.val))).trans ?_
  unfold Holds
  refine and_congr ?_ (and_congr ?_ ?_)
  · rw [← forall_rows hN (fun c i => (xsum (S.A c i) (S.bit w c b) S.N && xsum (S.B c i) (S.bit w c b) S.N) =
      S.bit w c b i)]
    refine forall₂_congr fun q hq => ?_
    have hout : S.rel.out b (S.J * S.N + q) = .hid (S.idx b q) := by
      simp only [rel, ite_eq_left (show S.J * S.N + q < 2 * (S.J * S.N) by omega), Nat.add_sub_cancel_left]
    have hq' : x q = S.bit w (q / S.N) b (q % S.N) := by
      rw [← hx _ _ (Nat.mod_lt _ hN), Nat.div_add_mod']
    simp only []
    rw [bit_row hxa hN (Nat.le_add_right _ _) (by simp only []; omega), hout, hhid]
    simp only [Nat.add_sub_cancel_left]
    change S.res x q = x q ↔ _
    rw [hq', res, xsum_congr _ (fun r hr => hx (q / S.N) r hr), xsum_congr (S.B _ _) (fun r hr => hx (q / S.N) r hr)]
  · refine forall₂_congr fun c hc => ?_
    have hout : S.rel.out b (2 * (S.J * S.N) + c) = .pub 0 := by
      simp only [rel, ite_eq_right (show ¬ 2 * (S.J * S.N) + c < 2 * (S.J * S.N) by omega),
        ite_eq_left (show 2 * (S.J * S.N) + c < 2 * (S.J * S.N) + S.J by omega)]
    have hcp : S.copied c = [c * S.N + S.pin c] := ite_eq_left hc
    have hlt : ∀ t ∈ S.copied c, t < S.J * S.N := by
      rw [hcp]
      intro t ht
      rw [List.mem_singleton.1 ht]
      exact hcopy c hc _ (hw.pin_lt c hc)
    dsimp only
    rw [bit_copy hxa (g := ⟨2 * (S.J * S.N) + c, by unfold size; omega⟩) (Nat.le_add_right _ _)
      (by dsimp only; rw [Nat.add_sub_cancel_left]; exact hlt), hout, hpub, pubs_zero]
    dsimp only
    rw [Nat.add_sub_cancel_left, hcp]
    simp only [List.map_cons, List.map_nil, List.foldr_cons, List.foldr_nil, Bool.xor_false]
    rw [hx c _ (hw.pin_lt c hc)]
  · rw [List.forall_mem_iff_getElem]
    refine forall₂_congr fun k hk => ?_
    obtain ⟨h1, h2⟩ := hw.slot_lt _ (List.getElem_mem hk)
    have hout : S.rel.out b (2 * (S.J * S.N) + S.J + k) = .pub (1 + b * S.slots.length + k) := by
      simp only [rel, ite_eq_right (show ¬ 2 * (S.J * S.N) + S.J + k < 2 * (S.J * S.N) by omega),
        ite_eq_right (show ¬ 2 * (S.J * S.N) + S.J + k < 2 * (S.J * S.N) + S.J by omega)]
      congr 2
      omega
    have hcp : S.copied (2 * (S.J * S.N) + S.J + k - 2 * (S.J * S.N)) =
        [S.slots[k].1 * S.N + S.slots[k].2.1] := by
      rw [show 2 * (S.J * S.N) + S.J + k - 2 * (S.J * S.N) = S.J + k by omega, copied,
        ite_eq_right (show ¬ S.J + k < S.J by omega), Nat.add_sub_cancel_left, List.getElem?_eq_getElem hk]
      rfl
    have hlt : ∀ t ∈ S.copied (2 * (S.J * S.N) + S.J + k - 2 * (S.J * S.N)), t < S.J * S.N := by
      rw [hcp]
      intro t ht
      rw [List.mem_singleton.1 ht]
      exact hcopy _ h1 _ h2
    dsimp only
    rw [bit_copy hxa (g := ⟨2 * (S.J * S.N) + S.J + k, by unfold size; omega⟩) (by dsimp only; omega)
      (by dsimp only; exact hlt), hout, hpub, pubs_slot hb hk]
    dsimp only
    rw [hcp]
    simp only [List.map_cons, List.map_nil, List.foldr_cons, List.foldr_nil, Bool.xor_false]
    rw [hx _ _ h2]

theorem ofBool_getLsbD_zero (v : Bool) : (BitVec.ofBool v).getLsbD 0 = v := by
  cases v <;> rfl

/-- **One block**: the block circuit maps block `b`'s inputs to its outputs exactly when the system holds in block
`b`. -/
theorem block_iff (hw : S.WF) (s : Claims.Flock.Stmt) (w : Claims.Flock.Wit) (hC : s.C = S.circ)
    (hinp : s.inp = S.rel.inp) (hout : s.out = S.rel.out) (hpub : s.pub = S.pubs) {b : ℕ} (hb : b < S.nB) :
    (∀ o ∈ s.C.outputs, (s.C.toCircuit.eval (fun g => BitVec.ofBool (Claims.Flock.val s w (s.inp b g.val))) o).getLsbD 0 =
      Claims.Flock.val s w (s.out b o.val)) ↔ S.Holds w b := by
  obtain ⟨m, kLog, C, inp, out, pub, form⟩ := s
  obtain rfl : C = S.circ := hC
  obtain rfl : inp = S.rel.inp := hinp
  obtain rfl : out = S.rel.out := hout
  obtain rfl : pub = S.pubs := hpub
  exact outputs_iff hw w hb _ (fun _ => ofBool_getLsbD_zero _) _ (fun _ => rfl) (fun _ => rfl)

/-- **The relation**: a witness satisfies `RelS` of a statement with the system's fields and `nB` blocks exactly when
the system holds in every block. -/
theorem relS_iff (hw : S.WF) (s : Claims.Flock.Stmt) (w : Claims.Flock.Wit) (hC : s.C = S.circ)
    (hinp : s.inp = S.rel.inp) (hout : s.out = S.rel.out) (hpub : s.pub = S.pubs) (hnb : s.nBlocks = S.nB) :
    Claims.Flock.RelS s w ↔ ∀ b < S.nB, S.Holds w b := by
  unfold Claims.Flock.RelS
  rw [hnb]
  exact forall₂_congr fun b hb => block_iff hw s w hC hinp hout hpub hb

end Sys

end Flock.Readable
