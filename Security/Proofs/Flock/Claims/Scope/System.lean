import Proofs.Flock.Claims.Scope.Block
import Proofs.Flock.Claims.Scope.Readable

/-!
# Block R1CS systems over GF(2), as the readable relation

A `Sys` is `J` tables of block R1CS over GF(2), each block `N` bits: table `c`'s row `i` reads the columns `A c i` and
`B c i` select, its pin is `pin c`, and each `slots` entry `(c, a, v)` fixes bit `a` of table `c`'s block `b` to `v b`.
`Sys.rel` is that system as `Relation`'s fields: the block circuit `circ` recomputes every row of every table's block
(`rowGates`) and reads every pin and fixed bit; block `b`'s inputs and outputs are the hidden bits of every table's block
`b` (table `c`'s bit `a` of block `b` is hidden value `c·T + b·N + a`), then the public `1` for each pin and the public
value of each fixed bit. No commit strings.

`relS_iff`: a witness satisfies `Claims.Flock.RelS` of a statement with these fields exactly when, in every block, every
table's rows hold (`Sys.Holds`).
-/

namespace Flock.Readable

open Claims

/-- `J` tables of block R1CS over GF(2) with `N`-bit blocks, `T` hidden bits per table and `nB` blocks: rows `A c i`,
`B c i`, pins, and fixed bits `(table, address, value in each block)`. -/
structure Sys where
  J : ℕ
  N : ℕ
  T : ℕ
  nB : ℕ
  A : ℕ → ℕ → ℕ → Bool
  B : ℕ → ℕ → ℕ → Bool
  pin : ℕ → ℕ
  slots : List (ℕ × ℕ × (ℕ → Bool))

namespace Sys

variable (S : Sys)

/-- Row `q = c·N + i`'s first gate. -/
def base (q : ℕ) : ℕ := S.J * S.N + 1 + q * (2 * S.N + 1)

/-- The rows' gates, table by table and row by row. -/
def rows (q : ℕ) : List Claims.Gate :=
  rowGates (S.J * S.N) (q / S.N * S.N) (S.base q) S.N (S.A (q / S.N) (q % S.N)) (S.B (q / S.N) (q % S.N))

/-- **The block circuit**: the zero wire, every table's rows, and as outputs every row's AND, every pin and every fixed
bit. -/
def circ : Claims.Circuit where
  nIn := S.J * S.N
  gates := Claims.Gate.xor 0 0 :: (List.range (S.J * S.N)).flatMap S.rows
  outs := (List.range (S.J * S.N)).map (fun q => S.base q + 2 * S.N) ++ (List.range S.J).map (fun c => c * S.N + S.pin c) ++
    S.slots.map fun s => s.1 * S.N + s.2.1

/-- Hidden value of table `q / N`'s bit `q % N` of block `b`. -/
def idx (b q : ℕ) : ℕ := q / S.N * S.T + b * S.N + q % S.N

/-- The public values: `1`, then each block's fixed bits (value `1 + b·L + k` is block `b`'s fixed bit `k`). -/
def pubs : List Bool :=
  true :: (List.range (S.nB * S.slots.length)).map fun j =>
    (S.slots[j % S.slots.length]?.map fun s => s.2.2 (j / S.slots.length)).getD false

/-- **The system as the readable relation's fields.** -/
def rel : Relation where
  C := S.circ
  inp b := (List.range (S.J * S.N)).map fun q => .hid (S.idx b q)
  out b := (List.range (S.J * S.N)).map (fun q => .hid (S.idx b q)) ++ (List.range S.J).map (fun _ => .pub 0) ++
    (List.range S.slots.length).map fun k => .pub (1 + b * S.slots.length + k)
  pub := S.pubs
  cs := []
  grp := fun _ => []

/-- Table `c`'s bit `a` of block `b` in `w`. -/
def bit (w : Claims.Flock.Wit) (c b a : ℕ) : Bool := w.hid.getD (c * S.T + b * S.N + a) false

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

/-- The wires after the zero wire and rows `q < Q`, on inputs `x`. -/
def wsAt (S : Sys) (x : ℕ → Bool) (Q : ℕ) : List Bool :=
  ((List.range Q).flatMap S.rows).foldl step ((List.range (S.J * S.N)).map x ++ [false])

/-- Row `q`'s AND on inputs `x`. -/
def res (S : Sys) (x : ℕ → Bool) (q : ℕ) : Bool :=
  xsum (S.A (q / S.N) (q % S.N)) (fun r => x (q / S.N * S.N + r)) S.N &&
    xsum (S.B (q / S.N) (q % S.N)) (fun r => x (q / S.N * S.N + r)) S.N

theorem base_lt {q Q : ℕ} (h : q < Q) : S.base q + 2 * S.N < S.base Q := by
  have := Nat.mul_le_mul_right (2 * S.N + 1) (show q + 1 ≤ Q by omega)
  rw [Nat.succ_mul] at this
  simp only [base]
  omega

theorem table_le {Q : ℕ} (hN : 0 < S.N) (hQ : Q < S.J * S.N) : Q / S.N * S.N + S.N ≤ S.J * S.N := by
  have h1 : Q / S.N < S.J := (Nat.div_lt_iff_lt_mul hN).2 hQ
  have := Nat.mul_le_mul_right S.N (show Q / S.N + 1 ≤ S.J by omega)
  rw [Nat.succ_mul] at this
  exact this

theorem wsAt_spec (hN : 0 < S.N) (x : ℕ → Bool) : ∀ Q ≤ S.J * S.N,
    (S.wsAt x Q).length = S.base Q ∧ (∀ k < S.J * S.N, (S.wsAt x Q).getD k false = x k) ∧
      (S.wsAt x Q).getD (S.J * S.N) false = false ∧
      ∀ q < Q, (S.wsAt x Q).getD (S.base q + 2 * S.N) false = S.res x q
  | 0, _ => by
    refine ⟨by simp [wsAt, base], fun k hk => ?_, ?_, fun q hq => absurd hq (Nat.not_lt_zero _)⟩
    · simp [wsAt, List.getD_eq_getElem?_getD, List.getElem?_append_left (show k < ((List.range (S.J * S.N)).map x).length
        by simpa using hk), hk]
    · simp [wsAt, List.getD_eq_getElem?_getD]
  | Q + 1, hQ => by
    obtain ⟨hl, hx, hz, hr⟩ := wsAt_spec hN x Q (by omega)
    have hstep : S.wsAt x (Q + 1) = (S.rows Q).foldl step (S.wsAt x Q) := by
      simp [wsAt, List.range_succ, List.flatMap_append, List.foldl_append]
    have hle := table_le hN (show Q < S.J * S.N by omega)
    have hb0 : S.J * S.N + 1 ≤ S.base Q := by simp only [base]; omega
    obtain ⟨hl', hpre, hrow⟩ := row_wires (A := S.A (Q / S.N) (Q % S.N)) (B := S.B (Q / S.N) (Q % S.N))
      (x := fun r => x (Q / S.N * S.N + r)) hN hl (by omega) hz (by omega)
      (fun r hr => hx _ (by omega))
    rw [hstep]
    refine ⟨?_, fun k hk => ?_, ?_, fun q hq => ?_⟩
    · simp only [rows] at hl' ⊢
      rw [hl']
      have e : (Q + 1) * (2 * S.N + 1) = Q * (2 * S.N + 1) + (2 * S.N + 1) := Nat.succ_mul _ _
      simp only [base, e]
      omega
    · simp only [rows]
      rw [hpre k (by omega)]
      exact hx k hk
    · simp only [rows]
      rw [hpre _ (by omega)]
      exact hz
    · simp only [rows]
      rcases Nat.lt_succ_iff_lt_or_eq.1 hq with hq | rfl
      · rw [hpre _ (base_lt hq)]
        exact hr q hq
      · exact hrow

/-- **The block circuit on inputs `x`**: every row's AND, every pin's bit and every fixed bit's. -/
theorem circ_eval (hw : S.WF) (x : ℕ → Bool) :
    S.circ.eval ((List.range (S.J * S.N)).map x) =
      (List.range (S.J * S.N)).map (S.res x) ++ (List.range S.J).map (fun c => x (c * S.N + S.pin c)) ++
        S.slots.map fun s => x (s.1 * S.N + s.2.1) := by
  have hfin := wsAt_spec hw.N_pos x (S.J * S.N) le_rfl
  have hw0 : S.circ.wires ((List.range (S.J * S.N)).map x) = S.wsAt x (S.J * S.N) := by
    rw [wires_eq]
    simp only [circ, List.foldl_cons, wsAt]
    congr 1
    simp [step, gv]
  unfold Circuit.eval
  rw [hw0]
  simp only [circ, List.map_append, List.map_map]
  congr 1
  · congr 1
    · apply List.map_congr_left
      intro q hq
      exact hfin.2.2.2 q (List.mem_range.1 hq)
    · apply List.map_congr_left
      intro c hc
      have := hw.pin_lt c (List.mem_range.1 hc)
      have := table_le hw.N_pos (show c * S.N < S.J * S.N from
        Nat.mul_lt_mul_of_pos_right (List.mem_range.1 hc) hw.N_pos)
      simp only [Function.comp]
      apply hfin.2.1
      rw [Nat.mul_div_cancel _ hw.N_pos] at this
      omega
  · apply List.map_congr_left
    intro s hs
    obtain ⟨h1, h2⟩ := hw.slot_lt s hs
    have := Nat.mul_le_mul_right S.N (show s.1 + 1 ≤ S.J by omega)
    rw [Nat.succ_mul] at this
    simp only [Function.comp]
    exact hfin.2.1 _ (by omega)

theorem xsum_congr (A : ℕ → Bool) {x y : ℕ → Bool} {n : ℕ} (h : ∀ r < n, x r = y r) : xsum A x n = xsum A y n := by
  induction n with
  | zero => rfl
  | succ n ih => rw [xsum_succ, xsum_succ, ih fun r hr => h r (by omega), h n (by omega)]

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

theorem append_eq_iff {α : Type} {a b c d : List α} (h : a.length = c.length) : a ++ b = c ++ d ↔ a = c ∧ b = d :=
  ⟨fun e => List.append_inj e h, fun ⟨h1, h2⟩ => h1 ▸ h2 ▸ rfl⟩

theorem map_eq_range_iff {α β : Type} (l : List α) (f : α → β) (g : ℕ → β) :
    l.map f = (List.range l.length).map g ↔ ∀ k (hk : k < l.length), f l[k] = g k := by
  rw [List.ext_getElem_iff]
  simp

theorem forall_rows (hN : 0 < S.N) (F : ℕ → ℕ → Prop) :
    (∀ q ∈ List.range (S.J * S.N), F (q / S.N) (q % S.N)) ↔ ∀ c < S.J, ∀ i < S.N, F c i := by
  constructor
  · intro h c hc i hi
    have hq : c * S.N + i < S.J * S.N := by
      have := Nat.mul_le_mul_right S.N (show c + 1 ≤ S.J by omega)
      rw [Nat.succ_mul] at this
      omega
    have := h _ (List.mem_range.2 hq)
    rwa [show c * S.N + i = i + c * S.N by omega, Nat.add_mul_div_right _ _ hN, Nat.div_eq_of_lt hi, Nat.zero_add,
      Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hi] at this
  · intro h q hq
    exact h _ ((Nat.div_lt_iff_lt_mul hN).2 (List.mem_range.1 hq)) _ (Nat.mod_lt _ hN)

/-- **One block**: the block circuit maps block `b`'s inputs to its outputs exactly when the system holds in block
`b`. -/
theorem block_iff (hw : S.WF) (s : Claims.Flock.Stmt) (w : Claims.Flock.Wit) (hC : s.C = S.circ)
    (hinp : s.inp = S.rel.inp) (hout : s.out = S.rel.out) (hpub : s.pub = S.pubs) {b : ℕ} (hb : b < S.nB) :
    s.C.eval ((s.inp b).map (Claims.Flock.val s w)) = (s.out b).map (Claims.Flock.val s w) ↔ S.Holds w b := by
  have hN := hw.N_pos
  let x : ℕ → Bool := fun q => w.hid.getD (S.idx b q) false
  have hx : ∀ c r, r < S.N → x (c * S.N + r) = S.bit w c b r := by
    intro c r hr
    simp only [x, idx, bit]
    rw [show c * S.N + r = r + c * S.N by omega, Nat.add_mul_div_right _ _ hN, Nat.div_eq_of_lt hr, Nat.zero_add,
      Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hr]
  have hin : (s.inp b).map (Claims.Flock.val s w) = (List.range (S.J * S.N)).map x := by
    rw [hinp]
    simp only [rel, List.map_map]
    rfl
  rw [hC, hin, circ_eval hw x, hout]
  simp only [rel, List.map_append, List.map_map]
  rw [append_eq_iff (by simp), append_eq_iff (by simp)]
  have hv : ∀ i, Claims.Flock.val s w (.pub i) = S.pubs.getD i false := fun i => by
    simp only [Claims.Flock.val, hpub]
  simp only [Function.comp_def, hv, pubs_zero]
  rw [List.map_inj_left, List.map_inj_left, map_eq_range_iff]
  simp only [Claims.Flock.val]
  rw [and_assoc]
  refine and_congr ?_ (and_congr ?_ ?_)
  · rw [← forall_rows hN (fun c i => (xsum (S.A c i) (S.bit w c b) S.N && xsum (S.B c i) (S.bit w c b) S.N) =
      S.bit w c b i)]
    refine forall₂_congr fun q hq => ?_
    have hq' : q / S.N < S.J := (Nat.div_lt_iff_lt_mul hN).2 (List.mem_range.1 hq)
    simp only [res]
    rw [xsum_congr _ (fun r hr => hx (q / S.N) r hr), xsum_congr (S.B _ _) (fun r hr => hx (q / S.N) r hr)]
    rfl
  · simp only [List.mem_range]
    refine forall₂_congr fun c hc => ?_
    rw [hx c _ (hw.pin_lt c hc)]
  · rw [List.forall_mem_iff_getElem]
    refine forall₂_congr fun k hk => ?_
    obtain ⟨h1, h2⟩ := hw.slot_lt _ (List.getElem_mem hk)
    rw [hx _ _ h2, pubs_slot hb hk]

/-- **The relation**: a witness satisfies `RelS` of a statement with the system's fields and `nB` blocks exactly when
the system holds in every block. -/
theorem relS_iff (H : Claims.Msg → Claims.Msg) (hw : S.WF) (s : Claims.Flock.Stmt) (w : Claims.Flock.Wit)
    (hC : s.C = S.circ) (hinp : s.inp = S.rel.inp) (hout : s.out = S.rel.out) (hpub : s.pub = S.pubs)
    (hcs : s.cs = []) (hnb : s.nBlocks = S.nB) :
    Claims.Flock.RelS H s w ↔ ∀ b < S.nB, S.Holds w b := by
  unfold Claims.Flock.RelS
  rw [hcs, hnb]
  simp only [List.length_nil, Nat.not_lt_zero, IsEmpty.forall_iff, implies_true, true_and]
  exact forall₂_congr fun b hb => block_iff hw s w hC hinp hout hpub hb

end Sys

end Flock.Readable
