import Proofs.Flock.VBridge.RecOpen
import Proofs.Flock.Soundness.Discharge.Hm.Laid

/-!
# V*'s opening as one laid-out net

`recOpenNet`: V*'s outer circuit `RecOpen_v3` or `RecOpen_v4` as one circuit of the verifier's builder
(`Flock.HmNets`), its ports `row, aux, acc, coef, dirs` the input bits `0, 1, …` in that order, then `recOpen`; its
outputs `out ‖ acc_out`. `Ports` gives the ports' widths, so it is either version: v4's `dirs` is 64 words where v3's
is 8, and `recOpen` reads only its low bits.
`Flock.HmNets.layout` lays it out as one slot type (`recOpenLaid`): the input bits in one group from column 0, then every
AND's row in creation order, then the outputs in one group from `outStart`.

`laid_recOpen`: whenever the laid-out net's rows hold at the column values `v` (`LaidRows`), if the input columns hold
the bytes of the row's elements, of `acc`, of the coefficients and of 64-byte siblings (`ColsHold`), and the direction
columns hold the position's low bits, then the output columns hold the bytes `sound_recOpen` gives: the verifier of
record's climb of the row's SHA-512 leaf and 64 zero bytes, then the consistency term and the rest of `acc`. It needs no
freshness hypothesis: every builder `recOpen` runs adds rows only as `fresh` does (`fresh_recOpen`), so `laid_rows` applies.

`colsHold_words`: columns that hold the bits of 16-bit words, bit `q % 16` of word `q / 16`, hold the bytes of the
elements those words are the limbs of: bit `k` of element `i` is bit `k % 16` of word `8 i + k / 16` (`bitOf_limb`).
-/

namespace FlockVBridge

open Flock.HmNets hiding and
open Flock FlockSoundness.Discharge.Hm

/-! ## Freshness -/

theorem fresh_leafSums : ∀ (ps : List (List Lin × List Lin)) (sums : List Lin), Fresh (leafSums ps sums)
  | [], _ => Fresh.pure' _
  | _ :: ps, _ => Fresh.bind' (Fresh.mapM' _ (fun _ => fresh_and _ _) _) fun _ => fresh_leafSums ps _

theorem fresh_residual (S : GfS) (W V : List Lin) (r : Res) : Fresh (residual S W V r) := by
  unfold residual
  exact Fresh.ite' (Fresh.pure' _)
    (Fresh.bind' (fresh_leafSums _ _) fun _ => Fresh.bind' (fresh_commitW _) fun _ => Fresh.pure' _)

theorem fresh_residualsGo (S : GfS) (W V : List Lin) : ∀ rs : List Res, Fresh (residualsGo S W V rs)
  | [] => Fresh.pure' _
  | r :: rs => Fresh.bind' (fresh_residual S W V r) fun _ =>
      Fresh.bind' (fresh_residualsGo S W V rs) fun _ => Fresh.pure' _

theorem fresh_shaBlock (st blk : List Lin) : Fresh (shaBlock st blk) :=
  Fresh.bind' (fresh_compress _ _) fun _ => Fresh.pure' _

theorem fresh_shaBlocks (msg : List Lin) : ∀ (n k : ℕ) (st : List Lin), Fresh (shaBlocks msg n k st)
  | 0, _, _ => Fresh.pure' _
  | n + 1, k, _ => Fresh.bind' (fresh_shaBlock _ _) fun st' => fresh_shaBlocks msg n (k + 1) st'

theorem fresh_shaFrom (cv : List (BitVec 64)) (bits : List Lin) (pre : ℕ) : Fresh (shaFrom cv bits pre) :=
  Fresh.bind' (fresh_shaBlocks _ _ _ _) fun _ => Fresh.pure' _

theorem fresh_swap (d : Lin) (a s : List Lin) : Fresh (swap d a s) :=
  Fresh.bind' (Fresh.mapM' _ (fun _ => fresh_and _ _) _) fun _ => Fresh.pure' _

theorem fresh_climbFrom (d : List Lin) : ∀ (h : ℕ) (cur : List Lin) (ss : List (List Lin)), Fresh (climbFrom d h cur ss)
  | _, _, [] => Fresh.pure' _
  | h, _, _ :: rest => Fresh.bind' (fresh_swap _ _ _) fun _ =>
      Fresh.bind' (fresh_shaFrom _ _ _) fun cur' => fresh_climbFrom d (h + 1) cur' rest

theorem fresh_climbRow (H : ℕ) (row aux d : List Lin) : Fresh (climbRow H row aux d) :=
  Fresh.bind' (fresh_shaFrom _ _ _) fun _ => Fresh.bind' (fresh_climbFrom _ _ _ _) fun _ => Fresh.pure' _

/-- **`recOpen` adds rows only as `fresh` does**: after the input bits, its rows output the next bits in creation order. -/
theorem fresh_recOpen (LANES H : ℕ) (row aux acc coef dirs : List Lin) :
    Fresh (recOpen LANES H row aux acc coef dirs) :=
  Fresh.bind' (fresh_residualsGo _ _ _ _) fun _ => Fresh.bind' (fresh_climbRow _ _ _ _) fun _ => Fresh.pure' _

/-! ## The net -/

/-- V*'s input ports' widths in bits, in port order. -/
structure Ports where
  row : ℕ
  aux : ℕ
  acc : ℕ
  coef : ℕ
  dirs : ℕ

/-- The input ports as `layout` takes them. -/
def Ports.bits (p : Ports) : Array ℕ := #[p.row, p.aux, p.acc, p.coef, p.dirs]

theorem foldl_bits (p : Ports) : p.bits.foldl (· + ·) 0 = p.row + p.aux + p.acc + p.coef + p.dirs := by
  rw [Ports.bits, ← Array.foldl_toList]
  show 0 + p.row + p.aux + p.acc + p.coef + p.dirs = _
  rw [Nat.zero_add]

/-- **V*'s outer circuit as one net**: the ports `row, aux, acc, coef, dirs` are the input bits `0, 1, …` in that order,
then `recOpen`. -/
def recOpenNet (LANES H : ℕ) (p : Ports) : M (List Lin × List Lin) := do
  let row ← input p.row
  let aux ← input p.aux
  let acc ← input p.acc
  let coef ← input p.coef
  let dirs ← input p.dirs
  recOpen LANES H row aux acc coef dirs

/-- `recOpenNet`'s outputs from the empty state: `out ‖ acc_out`. -/
def recOpenOuts (LANES H : ℕ) (p : Ports) : List Lin :=
  ((recOpenNet LANES H p).run St.empty).1.1 ++ ((recOpenNet LANES H p).run St.empty).1.2

/-- `recOpenNet` laid out as one slot type, with output ports `outBits`. -/
def recOpenLaid (LANES H : ℕ) (p : Ports) (outBits : Array ℕ) : Laid :=
  layout ((recOpenNet LANES H p).run St.empty).2 p.bits outBits (recOpenOuts LANES H p).toArray

theorem run_inputs5 {β : Type} (a b c d e : ℕ) (g : List Lin → List Lin → List Lin → List Lin → List Lin → M β) (s : St) :
    (input a >>= fun x₁ => input b >>= fun x₂ => input c >>= fun x₃ => input d >>= fun x₄ => input e >>= fun x₅ =>
      g x₁ x₂ x₃ x₄ x₅).run s =
      (g (vars s.next a) (vars (s.next + a) b) (vars (s.next + a + b) c) (vars (s.next + a + b + c) d)
        (vars (s.next + a + b + c + d) e)).run ⟨s.next + a + b + c + d + e, s.rows⟩ := rfl

theorem recOpenNet_run (LANES H : ℕ) (p : Ports) : (recOpenNet LANES H p).run St.empty =
    (recOpen LANES H (vars 0 p.row) (vars p.row p.aux) (vars (p.row + p.aux) p.acc) (vars (p.row + p.aux + p.acc) p.coef)
      (vars (p.row + p.aux + p.acc + p.coef) p.dirs)).run ⟨p.row + p.aux + p.acc + p.coef + p.dirs, []⟩ := by
  have h := run_inputs5 p.row p.aux p.acc p.coef p.dirs (recOpen LANES H) ⟨0, []⟩
  simp only [Nat.zero_add] at h
  exact h

/-! ## Columns and forms -/

/-- The columns from `base` hold the bytes `b`: column `base + t` holds bit `t % 8` of byte `t / 8`. -/
def ColsHold (v : ℕ → Bool) (base : ℕ) (b : ByteArray) : Prop :=
  ∀ t < 8 * b.size, v (base + t) = bitL b.data.toList t

theorem carries_vars {v : ℕ → Bool} {nIn inRows base n : ℕ} {b : ByteArray} (hn : n = 8 * b.size)
    (hle : base + n ≤ nIn) (h : ColsHold v base b) : Carries (fun x => v (col nIn inRows x)) (vars base n) b := by
  refine ⟨by rw [length_vars, hn], fun t ht => ?_⟩
  rw [length_vars] at ht
  rw [bit_vars _ _ _ ht]
  show v (col nIn inRows (base + t)) = _
  simp only [col, show base + t < nIn by omega, ↓reduceIte]
  exact h t (by omega)

theorem vars_drop_take (k n a m : ℕ) (h : a + m ≤ n) : ((vars k n).drop a).take m = vars (k + a) m := by
  apply List.ext_getElem
  · simp only [List.length_take, List.length_drop, length_vars]
    omega
  · intro j _ _
    simp [vars, Nat.add_assoc]

theorem forall₂_sibs {v : ℕ → Bool} {nIn inRows base n H : ℕ} (ss : List ByteArray) (hss : ss.length = H)
    (hn : 512 * H ≤ n) (hle : base + n ≤ nIn)
    (hs : ∀ i (hi : i < ss.length), ss[i].size = 64 ∧ ColsHold v (base + 512 * i) ss[i]) :
    List.Forall₂ (fun w b => b.size = 64 ∧ Carries (fun x => v (col nIn inRows x)) w b) (sibForms (vars base n) H) ss := by
  rw [List.forall₂_iff_get]
  refine ⟨by simp [sibForms, hss], fun i h1 h2 => ?_⟩
  have hi : i < H := by simpa [sibForms] using h1
  obtain ⟨hsz, hcol⟩ := hs i h2
  simp only [List.get_eq_getElem, sibForms, List.getElem_map, List.getElem_range]
  rw [vars_drop_take base n (512 * i) 512 (by omega)]
  exact ⟨hsz, carries_vars (by rw [hsz]) (by omega) hcol⟩

theorem ev_vars_getD {v : ℕ → Bool} {nIn inRows base n h : ℕ} (hh : h < n) (hle : base + n ≤ nIn) :
    ev (fun x => v (col nIn inRows x)) ((vars base n).getD h Lin.zero) = v (base + h) := by
  rw [List.getD_eq_getElem _ _ (by rw [length_vars]; exact hh)]
  simp only [vars, List.getElem_map, List.getElem_range, ev_var]
  simp only [col, show base + h < nIn by omega, ↓reduceIte]

/-- A laid-out net's output columns hold the bytes its output forms carry. -/
theorem colsHold_outs {st : St} {inBits outBits : Array ℕ} {w : List Lin} {v : ℕ → Bool} {B : ByteArray}
    (hout : w.length ≤ outBits.foldl (· + ·) 0) (h : LaidRows (layout st inBits outBits w.toArray) (inRowsOf inBits) v)
    (hc : Carries (fun x => v (col (inBits.foldl (· + ·) 0) (inRowsOf inBits) x)) w B) :
    ColsHold v (outStart st inBits outBits) B := by
  intro t ht
  have ht' : t < w.length := by rw [hc.1]; exact ht
  rw [laid_outs st inBits outBits w.toArray v (by simpa using hout) h t (by simpa using ht'), ← hc.2 t ht']
  simp [bit, ht']

/-! ## The laid-out net -/

/-- **V*'s laid-out opening computes the verifier of record's climb and the opening's consistency term.** For a Merkle
scheme `ms` whose node is `SHA-512(l ‖ r)`, `H ≤ 32` levels and input ports `p` laid out as `row, aux, acc, coef, dirs`,
whenever `recOpenLaid`'s rows hold at the column values `v` (`LaidRows`): if the `row` columns hold the bytes of the
`LANES` elements `r`, the `acc` columns those of two elements `a` and then `rest`, the `coef` columns those of `c`, the
`aux` columns the 64-byte siblings `ss` one after another, and direction column `h < H` bit `h` of the position `idx`,
then the output columns from `outStart` hold `Flock.climb ms (SHA-512(rowBytes r)) idx ss` and 64 zero bytes
(`rec_open.expected` of that node), then the bytes of `consTerm` at `k = 0, 1` (`rec_open.acc_ref`) and `rest`. -/
theorem laid_recOpen (ms : MerkleScheme) (hms : ∀ l r, ms.node l r = Sha512.hash (l ++ r)) (LANES H : ℕ) (hH : H ≤ 32)
    (p : Ports) (outBits : Array ℕ) (v : ℕ → Bool) (hout : (recOpenOuts LANES H p).length ≤ outBits.foldl (· + ·) 0)
    (h : LaidRows (recOpenLaid LANES H p outBits) (inRowsOf p.bits) v)
    (r a c : Array F128) (rest : ByteArray) (idx : ℕ) (ss : List ByteArray)
    (hr : r.size = LANES) (ha : a.size = 2) (hss : ss.length = H)
    (hrowN : p.row = 8 * (MerkleScheme.rowBytes r).size) (hauxN : 512 * H ≤ p.aux)
    (haccN : p.acc = 8 * (MerkleScheme.rowBytes a ++ rest).size)
    (hcoefN : p.coef = 8 * (MerkleScheme.rowBytes c).size) (hdirsN : H ≤ p.dirs)
    (hrow : ColsHold v 0 (MerkleScheme.rowBytes r))
    (haux : ∀ i (hi : i < ss.length), ss[i].size = 64 ∧ ColsHold v (p.row + 512 * i) ss[i])
    (hacc : ColsHold v (p.row + p.aux) (MerkleScheme.rowBytes a ++ rest))
    (hcoef : ColsHold v (p.row + p.aux + p.acc) (MerkleScheme.rowBytes c))
    (hdir : ∀ h < H, v (p.row + p.aux + p.acc + p.coef + h) = idx.testBit h) :
    ColsHold v (outStart ((recOpenNet LANES H p).run St.empty).2 p.bits outBits)
      ((Flock.climb ms (Sha512.hash (MerkleScheme.rowBytes r)) idx ss ++ ⟨Array.replicate 64 0⟩) ++
        (MerkleScheme.rowBytes #[consTerm LANES r a c 0, consTerm LANES r a c 1] ++ rest)) := by
  have hf := foldl_bits p
  have hg : Good (p.bits.foldl (· + ·) 0) ((recOpenNet LANES H p).run St.empty).2 := by
    rw [recOpenNet_run, hf]
    exact fresh_recOpen LANES H _ _ _ _ _ _ ⟨p.row + p.aux + p.acc + p.coef + p.dirs, []⟩
      ⟨by simp, fun k r h => by simp at h⟩
  have hst := laid_rows ((recOpenNet LANES H p).run St.empty).2 p.bits outBits (recOpenOuts LANES H p).toArray v hg h
  rw [recOpenNet_run] at hst
  have hP := (sound_recOpen ms hms LANES H hH (vars 0 p.row) (vars p.row p.aux) (vars (p.row + p.aux) p.acc)
    (vars (p.row + p.aux + p.acc) p.coef) (vars (p.row + p.aux + p.acc + p.coef) p.dirs)
    ⟨p.row + p.aux + p.acc + p.coef + p.dirs, []⟩).2 _ hst
  obtain ⟨h1, h2⟩ := hP r a c rest idx ss hr ha
    (carries_vars hrowN (by rw [hf]; omega) hrow)
    (carries_vars haccN (by rw [hf]; omega) hacc)
    (carries_vars hcoefN (by rw [hf]; omega) hcoef)
    (forall₂_sibs ss hss hauxN (by rw [hf]; omega) haux)
    (fun j hj => (ev_vars_getD (by omega) (by rw [hf])).trans (hdir j hj))
  refine colsHold_outs (st := ((recOpenNet LANES H p).run St.empty).2) (w := recOpenOuts LANES H p) hout h ?_
  show Carries _ (((recOpenNet LANES H p).run St.empty).1.1 ++ ((recOpenNet LANES H p).run St.empty).1.2) _
  rw [recOpenNet_run]
  exact carries_append h1 h2

/-! ## Words -/

/-- Element `x`'s 16-bit word `j < 8`: its bits `16 j … 16 j + 15`. -/
def limb (x : F128) (j : ℕ) : ℕ := FlockLevel3.w128 x.lo x.hi / 2 ^ (16 * j) % 2 ^ 16

/-- Bit `k` of an element is bit `k % 16` of its word `k / 16`. -/
theorem bitOf_limb (x : F128) (k : ℕ) : bitOf x k = (limb x (k / 16)).testBit (k % 16) := by
  rw [bitOf, limb, Nat.testBit_mod_two_pow, ← Nat.shiftRight_eq_div_pow, Nat.testBit_shiftRight,
    show 16 * (k / 16) + k % 16 = k by omega]
  simp [show k % 16 < 16 by omega]

/-- **Columns that hold 16-bit words hold the elements those words are the limbs of**: if column `base + q` holds bit
`q % 16` of word `w (q / 16)` and word `8 i + j` is element `i`'s word `j`, the columns hold the row's bytes
(`MerkleScheme.rowBytes`): bit `k` of element `i` is bit `k % 16` of word `8 i + k / 16`. -/
theorem colsHold_words {v : ℕ → Bool} {base : ℕ} (r : Array F128) (w : ℕ → ℕ)
    (hw : ∀ i < r.size, ∀ j < 8, w (8 * i + j) = limb r[i]! j)
    (hv : ∀ q < 128 * r.size, v (base + q) = (w (q / 16)).testBit (q % 16)) :
    ColsHold v base (MerkleScheme.rowBytes r) := by
  intro t ht
  rw [size_rowBytes] at ht
  have hi : t / 128 < r.size := by omega
  rw [hv t (by omega), bitL_rowBytes]
  simp only [FlockVBridge.wbit]
  rw [getD_toList, bitOf_limb, show t % 128 % 16 = t % 16 by omega,
    show t / 16 = 8 * (t / 128) + t % 128 / 16 by omega, hw _ hi _ (by omega)]

end FlockVBridge
