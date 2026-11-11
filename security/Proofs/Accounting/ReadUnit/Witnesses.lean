module

public import Properties.Flock.ReadNet
public import Proofs.Accounting.ReadUnit.Digests
public import Proofs.Accounting.ReadUnit.Guarantees
public import Proofs.Flock.Soundness.Refine.Walk
import Proofs.Core.Circuit
import Proofs.Flock.Soundness.Discharge.Placed.Keyed
import Proofs.Flock.Soundness.Discharge.Placed.Wiring
import Proofs.Pouw.PearlC.WholeCircuit
import all Init.Data.Ord.Basic
import all Init.Data.String.Defs
import all Init.Data.String.Basic
import all Init.Data.Array.Basic
import all Init.Data.Array.DecidableEq
import all Init.Data.List.Control
import all Init.Data.Range.Basic
import all Init.Data.Nat.Fold
import all Init.Data.Array.Attach
import all Init.WF

/-!
# `TableReadBinding` says something

* **Honest** (`TableReadBinding_honest`): the check accepts a template at the served expert-row read's shape, 256 rows
  of 3072 words (the one the read's text reads back to), from the one fact witnessed outside Lean
  (`ReadUnit.ExpertReadAccepted`).
* **Nontrivial** (`TableReadBinding_nontrivial`): where the check doesn't fix the read's gates, a read needn't bind: some
  unit's reads don't. The unit net `net₀` has 5 input words (640 columns), its constant on column 640 and no computed
  row, so a read at the template's columns (`portsAt`, at one row of no words) holds its index, root and the domain
  id's first 96 bits on input gates, and the rest of its domain id, its salt and its verdict on the constant. At the
  evaluation of the input that puts a one-row table's root on the root's gates, every hypothesis of `ReadUnit.BindsAt`
  holds at no error and no cost: the unit is correct, its rows hold, the table, a row of 16 bits, is committed under
  that root, and SHA-512 is collision resistant at the reads' finder, since what the read's check and its table's
  hash has pairwise different digests (`Witness.no_collision₀`). The read returns no bits.
-/

@[expose] public section

namespace Accounting.ReadUnit.Witness

open FlockSoundness.Discharge

/-! ## A digest's bits in leaf order -/

/-- Byte `i` of `d`, its index taken mod 64. -/
def dbyte (d : Verity.Digest) (i : ℕ) : UInt8 := d ⟨i % 64, Nat.mod_lt _ (by norm_num)⟩

/-- Bit `k` of `d` where `bytesOf` reads it: bit `k % 8` of byte `8 (k / 64) + 7 − (k % 64) / 8`. -/
def dbit (d : Verity.Digest) (k : ℕ) : Bool := (dbyte d (8 * (k / 64) + (7 - k % 64 / 8))).toNat.testBit (k % 8)

/-- A digest's 512 bits in leaf order, which `digestOf` reads back (`digestOf_digestBits`). -/
def digestBits (d : Verity.Digest) : List Bool := List.ofFn fun k : Fin 512 => dbit d k.val

theorem num_byte : ∀ b < 256, num (List.ofFn fun t : Fin 8 => b.testBit t.val) = b := by
  decide +kernel

theorem ofNat_num_byte (x : UInt8) :
    UInt8.ofNat (num (List.ofFn fun t : Fin 8 => x.toNat.testBit t.val)) = x := by
  rw [num_byte x.toNat x.toNat_lt_size]
  exact UInt8.ofNat_toNat

theorem window (d : Verity.Digest) {w m : ℕ} (hw : w < 8) (hm : m < 8) :
    ((digestBits d).drop (64 * w + 8 * (7 - m))).take 8 =
      List.ofFn fun t : Fin 8 => (dbyte d (8 * w + m)).toNat.testBit t.val := by
  apply List.ext_getElem
  · simp only [List.length_take, List.length_drop, digestBits, List.length_ofFn]
    omega
  · intro t h1 h2
    have ht : t < 8 := by simpa using h2
    have e1 : 8 * ((64 * w + 8 * (7 - m) + t) / 64) + (7 - (64 * w + 8 * (7 - m) + t) % 64 / 8) = 8 * w + m := by
      omega
    have e2 : (64 * w + 8 * (7 - m) + t) % 8 = t := by omega
    simp only [List.getElem_take, List.getElem_drop, digestBits, List.getElem_ofFn, dbit, e1, e2]

theorem getD_bytesOf {bs : List Bool} (hl : bs.length = 512) (i : Fin 64) :
    (bytesOf bs).getD i.val 0 = UInt8.ofNat (num ((bs.drop (64 * (i.val / 8) + 8 * (7 - i.val % 8))).take 8)) := by
  unfold bytesOf
  rw [hl]
  fin_cases i <;> rfl

theorem digestOf_digestBits (d : Verity.Digest) : digestOf (digestBits d) = d := by
  funext i
  have hi := i.isLt
  unfold digestOf
  rw [getD_bytesOf (by simp only [digestBits, List.length_ofFn]) i, window d (w := i.val / 8) (m := i.val % 8) (by omega) (by omega),
    ofNat_num_byte]
  unfold dbyte
  congr 1
  ext
  show (8 * (i.val / 8) + i.val % 8) % 64 = i.val
  omega

theorem num_ofFn_false : ∀ k, num (List.ofFn fun _ : Fin k => false) = 0
  | 0 => rfl
  | k + 1 => by
    rw [List.ofFn_succ]
    simp only [num]
    rw [num_ofFn_false k]
    rfl

/-! ## A unit net the check doesn't accept -/

/-- No input row, the constant on column 640, no computed row. -/
def rows₀ : Array (Array ℕ) := (Array.replicate 640 #[]).push #[640]

/-- 5 input words, as two aligned groups. -/
def inG₀ : Array Flock.Group := #[⟨0, 4, #[]⟩, ⟨4, 1, #[]⟩]

/-- One output word, at word 5: column 640, the constant's. -/
def outG₀ : Array Flock.Group := #[⟨5, 1, #[]⟩]

/-- The net `Net.ofRows` builds from them. Irreducible, so that unification never runs `Net.ofRows`. -/
@[irreducible] def net₀ : Flock.Net :=
  match Flock.Net.ofRows "nt" 641 640 inG₀ outG₀ rows₀ rows₀ with
  | .ok net => net
  | .error _ => default

theorem net₀_spec : Flock.Net.ofRows "nt" 641 640 inG₀ outG₀ rows₀ rows₀ = .ok net₀ := by
  kernel_rfl

theorem net₀_eqs : net₀.constPos = 640 ∧ net₀.inGroups = inG₀ ∧ net₀.outGroups = outG₀ ∧ net₀.inWords = 5 := by
  obtain ⟨-, -, -, -, hc, hi, ho, -, -, -, -, -, -, hw, -⟩ := Flock.Net.ofRows_ok net₀_spec
  exact ⟨hc, hi, ho, by rw [hw]; rfl⟩

/-- The circuit of one copy of `net₀` a unit. -/
def c₀ : Flock.Circuit := { (default : Flock.Circuit) with nets := #[("unit", net₀)], upv := 1, unitNet := 0 }

abbrev D₀ : Flock.Desc := .single c₀

theorem hU₀ : FlockSoundness.Built D₀.unroll.unit := ⟨"nt", 641, 640, inG₀, outG₀, rows₀, rows₀, net₀_spec⟩

theorem hj₀ : 0 < D₀.unroll.upv := Nat.one_pos

noncomputable abbrev Rw : FlockSoundness.Rows := FlockSoundness.Rows.ofNet hU₀

theorem nIn₀ : Rw.nIn = 640 := by
  show min (net₀.inWords * Flock.WORD) net₀.constPos = 640
  rw [net₀_eqs.2.2.2, net₀_eqs.1]
  rfl

theorem nComp₀ : Rw.nComp = 0 := by
  show net₀.constPos - min (net₀.inWords * Flock.WORD) net₀.constPos = 0
  rw [net₀_eqs.2.2.2, net₀_eqs.1]
  rfl

theorem w₀ : Rw.w = 641 := by
  show Rw.nIn + Rw.nComp + 1 = 641
  rw [nIn₀, nComp₀]

theorem one₀ : Rw.one.val = 640 := by
  show Rw.nIn + Rw.nComp = 640
  rw [nIn₀, nComp₀]

theorem colAt_lt {c : ℕ} (hc : c < 640) : (colAt Rw c).val = c := by
  unfold colAt
  rw [dite_eq_left (by rw [w₀]; omega)]

theorem colAt_ge {c : ℕ} (hc : 640 ≤ c) : colAt Rw c = Rw.one := by
  unfold colAt
  split
  · rename_i h
    have h' : c < 641 := by rw [← w₀]; exact h
    exact Fin.ext (by rw [one₀]; show c = 640; omega)
  · rfl

/-! ## Its gates -/

theorem srcKey₀ (x : ℕ) : Aliased.srcKey D₀.unroll x = .pad (x / 640) (x % 640) := by
  have hW : D₀.unroll.unit.inWords * Flock.WORD = 640 := by
    show net₀.inWords * Flock.WORD = 640
    rw [net₀_eqs.2.2.2]
    rfl
  have hp : FlockSoundness.portBits D₀.unroll.unit.inGroups = [] := by
    show FlockSoundness.portBits net₀.inGroups = []
    rw [net₀_eqs.2.1]
    decide +kernel
  unfold Aliased.srcKey
  rw [hW, hp]
  simp

theorem keyIdx₀ {B : ℕ} (x : Fin B) : Aliased.keyIdx D₀.unroll B x = some x := by
  unfold Aliased.keyIdx
  rw [ite_eq_right (by rw [srcKey₀]; simp)]
  obtain ⟨b, hb⟩ : ∃ b, (List.finRange B).find?
      (fun i => decide (Aliased.srcKey D₀.unroll i.val = Aliased.srcKey D₀.unroll x.val)) = some b :=
    Option.isSome_iff_exists.1 (List.find?_isSome.2 ⟨x, List.mem_finRange x, by simp⟩)
  rw [hb]
  have hbx := List.find?_some hb
  simp only [decide_eq_true_eq, srcKey₀, Aliased.Src.pad.injEq] at hbx
  have := b.isLt
  have := x.isLt
  congr 1
  ext
  omega

/-- Copy 0's gate on column `c` of unit 0. -/
noncomputable abbrev g₀ (c : Fin Rw.w) : Fin (Flock.Bridge.circuitOf D₀ hU₀ 1).N :=
  Flock.Bridge.copyGate D₀ hU₀ 1 0 hj₀ c

theorem keyB₀ : Layout.keyB D₀.unroll hU₀ = 640 := by
  show 1 * Rw.nIn = 640
  rw [nIn₀]

/-- Copy 0's stacked column `c`, as a column of the circuit's template instance rows. -/
noncomputable def sc₀ (c : Fin Rw.w) : Fin (Placed.circuitRows D₀.unroll hU₀).w := Rw.stackCol D₀.unroll.upv hj₀ c

theorem stackCol₀ (c : Fin Rw.w) (hc : c.val < 640) : (Rw.stackCol D₀.unroll.upv hj₀ c).val = c.val := by
  unfold FlockSoundness.Rows.stackCol
  rw [dite_eq_left (show c.val < Rw.nIn by rw [nIn₀]; exact hc)]
  simp

theorem sc₀_val (c : Fin Rw.w) (hc : c.val < 640) : (sc₀ c).val = c.val := stackCol₀ c hc

theorem copy_val (c : Fin Rw.w) (hc : c.val < 640) : (g₀ c).val = 2 + c.val := by
  have hs := sc₀_val c hc
  have hw : g₀ c = (Const.Prog.rowsWireC (Placed.keyProg D₀.unroll hU₀ 1) 0).wire
      (Fin.cast (congrArg FlockSoundness.Rows.w (Flock.Bridge.rows_keyProg D₀ hU₀ 1 0)).symm (sc₀ c)) :=
    Flock.Bridge.unitInst_wire D₀ hU₀ 1 0 (sc₀ c)
  have e := congrArg FlockSoundness.Rows.nIn (Flock.Bridge.rows_keyProg D₀ hU₀ 1 0)
  have hx : (Fin.cast (congrArg FlockSoundness.Rows.w (Flock.Bridge.rows_keyProg D₀ hU₀ 1 0)).symm (sc₀ c)).val <
      ((Placed.keyProg D₀.unroll hU₀ 1).rows 0).nIn := by
    rw [Fin.val_cast, hs, e]
    show c.val < 1 * Rw.nIn
    rw [nIn₀]
    omega
  have hxB : (Fin.cast (congrArg FlockSoundness.Rows.w (Flock.Bridge.rows_keyProg D₀ hU₀ 1 0)).symm (sc₀ c)).val <
      Layout.keyB D₀.unroll hU₀ := by
    rw [Fin.val_cast, hs, keyB₀]
    exact hc
  have hv : (Const.Prog.insAt (Placed.keyProg D₀.unroll hU₀ 1) 0
      (Fin.cast (congrArg FlockSoundness.Rows.w (Flock.Bridge.rows_keyProg D₀ hU₀ 1 0)).symm (sc₀ c)).val).val =
      (Placed.insOf 1 (Layout.keyB D₀.unroll hU₀) (Aliased.keyIdx D₀.unroll (Layout.keyB D₀.unroll hU₀)) (0 : Fin 1).val
        ⟨_, hxB⟩).val :=
    Placed.insAt_uniform 1 0 _ hxB
  rw [hw, Const.Prog.rowsWireC_wire_input _ _ _ hx, hv, Placed.insOf_of_lt (nU := 1) (B := Layout.keyB D₀.unroll hU₀)
    (src := Aliased.keyIdx D₀.unroll (Layout.keyB D₀.unroll hU₀)) (0 : Fin 1).isLt,
    keyIdx₀ (B := Layout.keyB D₀.unroll hU₀)]
  simp [Placed.srcGate, hs]

theorem copy_input (c : Fin Rw.w) (hc : c.val < 640) :
    (Flock.Bridge.circuitOf D₀ hU₀ 1).isInput (g₀ c) = true := by
  have hc' : c.val < Rw.nIn := lt_of_lt_of_eq hc nIn₀.symm
  have hne : (Flock.Bridge.partsOf D₀ hU₀ 1).unit (g₀ c) ≠ some 0 :=
    Flock.Bridge.copyGate_input D₀ hU₀ 1 0 hj₀ c hc'
  apply ((Flock.Bridge.partsOf D₀ hU₀ 1).input_iff _).1
  cases h : (Flock.Bridge.partsOf D₀ hU₀ 1).unit (g₀ c) with
  | none => rfl
  | some v => exact absurd (h.trans (by rw [Subsingleton.elim v 0])) hne

theorem const_input :
    (Flock.Bridge.circuitOf D₀ hU₀ 1).isInput (Flock.Bridge.constGate D₀ hU₀ 1 0) = true := by
  apply ((Flock.Bridge.partsOf D₀ hU₀ 1).input_iff _).1
  show (Placed.keyProg D₀.unroll hU₀ 1).unit (Flock.Bridge.constGate D₀ hU₀ 1 0) = none
  exact Layout.unit_uniform_input 1 _ (by rw [Flock.Bridge.constGate_val]; show 0 < 2 + _; omega)

/-! ## The read and its table -/

/-- The hm96 key the binding reads under. -/
abbrev key₀ : Hash.Msg := Flock.Hm96.Default512.key.data.toList

/-- The read at copy 0 of unit 0, at one row of no words. -/
noncomputable abbrev q₀ : Ports (Flock.Bridge.circuitOf D₀ hU₀ 1) := portsAt D₀ hU₀ 1 0 hj₀ 1 0

attribute [local irreducible] d₀

/-- The table's root: its one leaf's digest. Irreducible, so that unification never computes SHA-512. -/
@[irreducible] noncomputable def r₀ : Verity.Digest :=
  Verity.sha512 (Hash.RowLeaf.leafMsg Verity.sha512 Verity.sha512 (q₀.verifier key₀ d₀) 0 o₀)

/-- The input: the constant set, the index's gates (2 to 33) clear, the root's bits on gates 34 to 545 (columns 32 to
543), and every other gate set. -/
noncomputable def xs₀ (g : ℕ) : Bool :=
  if g = 0 then true else if g < 34 then false else if g < 546 then dbit r₀ (g - 34) else true

/-- The input as a transcript of the circuit. -/
noncomputable def in₀ : (Flock.Bridge.circuitOf D₀ hU₀ 1).Trace := fun g => BitVec.ofBool (xs₀ g.val)

noncomputable def tr₀ : (Flock.Bridge.circuitOf D₀ hU₀ 1).Trace := (Flock.Bridge.circuitOf D₀ hU₀ 1).eval in₀

theorem localEval₀ : (Flock.Bridge.partsOf D₀ hU₀ 1).localEval tr₀ 0 = tr₀ :=
  Pouw.PearlC.WholeCircuit.localEval_eval _ _ _

theorem bit_input {g : Fin (Flock.Bridge.circuitOf D₀ hU₀ 1).N}
    (h : (Flock.Bridge.circuitOf D₀ hU₀ 1).isInput g = true) : bit tr₀ g = xs₀ g.val := by
  unfold bit tr₀
  rw [Verity.Circuit.eval_input (h := h)]
  show (BitVec.ofBool (xs₀ g.val)).getLsbD 0 = xs₀ g.val
  cases xs₀ g.val <;> decide

theorem bit_const : bit tr₀ (Flock.Bridge.constGate D₀ hU₀ 1 0) = true := by
  rw [bit_input const_input, Flock.Bridge.constGate_val]
  simp [xs₀]

theorem bit_one : bit tr₀ (g₀ Rw.one) = true := by
  obtain ⟨z, -, hone, hz⟩ := Flock.Bridge.rows_localEval D₀ hU₀ 1 tr₀ 0 hj₀
  have h1 : z Rw.one.val = (tr₀ (g₀ Rw.one)).getLsbD 0 :=
    (hz Rw.one).trans (congrArg (fun (tx : (Flock.Bridge.circuitOf D₀ hU₀ 1).Trace) => (tx (g₀ Rw.one)).getLsbD 0)
      localEval₀)
  have h2 : (tr₀ (Flock.Bridge.constGate D₀ hU₀ 1 0)).getLsbD 0 = true := bit_const
  exact h1.symm.trans (hone.trans h2)

theorem bit_col_lt {c : ℕ} (hc : c < 640) : bit tr₀ (g₀ (colAt Rw c)) = xs₀ (2 + c) := by
  have hv := colAt_lt hc
  rw [bit_input (copy_input _ (by rw [hv]; exact hc)), copy_val _ (by rw [hv]; exact hc), hv]

theorem bit_col_ge {c : ℕ} (hc : 640 ≤ c) : bit tr₀ (g₀ (colAt Rw c)) = true := by
  rw [colAt_ge hc]
  exact bit_one

theorem tokOf₀ : q₀.tokOf tr₀ = 0 := by
  have e : bits tr₀ q₀.tok = List.ofFn fun _ : Fin 32 => false := by
    unfold bits
    congr 1
    funext i
    have := i.isLt
    show bit tr₀ (g₀ (colAt Rw i.val)) = false
    rw [bit_col_lt (by omega)]
    unfold xs₀
    rw [ite_eq_right (by omega), ite_eq_left (by omega)]
  unfold Ports.tokOf
  rw [e, num_ofFn_false]

theorem rootOf₀ : q₀.rootOf tr₀ = r₀ := by
  have e : bits tr₀ q₀.root = digestBits r₀ := by
    unfold bits digestBits
    congr 1
    funext i
    have := i.isLt
    show bit tr₀ (g₀ (colAt Rw (32 + i.val))) = dbit r₀ i.val
    rw [bit_col_lt (by omega)]
    unfold xs₀
    rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_left (by omega)]
    congr 1
    omega
  unfold Ports.rootOf
  rw [e, digestOf_digestBits]

theorem domainOf₀ : q₀.domainOf tr₀ = d₀ := by
  have e : bits tr₀ q₀.domain = List.ofFn fun _ : Fin 512 => true := by
    unfold bits
    congr 1
    funext i
    show bit tr₀ (g₀ (colAt Rw (544 + i.val))) = true
    by_cases hi : 544 + i.val < 640
    · rw [bit_col_lt hi]
      unfold xs₀
      rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega)]
    · exact bit_col_ge (by omega)
  unfold Ports.domainOf d₀
  rw [e]

theorem okOf₀ : q₀.okOf tr₀ = true := by
  have ho : Layout.outCol D₀.unroll 0 0 = 640 := by
    show (net₀.outGroups.getD 0 default).col * Flock.WORD + (16 * 0 + 0) = 640
    rw [net₀_eqs.2.2.1]
    rfl
  show bit tr₀ (g₀ (colAt Rw (Layout.outCol D₀.unroll 0 0))) = true
  rw [ho]
  exact bit_col_ge le_rfl

theorem hOf₀ : q₀.hOf tr₀ = [] := rfl

/-- The opening the read's gates hold: no row bits, the salt of ones on the constant, no siblings. -/
theorem opening₀ : q₀.opening tr₀ = op₀ := by
  have hr : bits tr₀ q₀.row = [] := List.ofFn_eq_nil_iff.2 rfl
  have hs : bits tr₀ q₀.salt = List.ofFn fun _ : Fin (8 * Hash.Hm96.SALT_BYTES) => true := by
    unfold bits
    congr 1
    funext i
    show bit tr₀ (g₀ (colAt Rw (Flock.ReadNet.saltAt 0 + i.val))) = true
    exact bit_col_ge (by unfold Flock.ReadNet.saltAt; omega)
  have hp : (List.ofFn fun j => digestOf (bits tr₀ (q₀.path j))) = [] :=
    List.ofFn_eq_nil_iff.2 (show Hash.Tree.depth 1 = 0 by decide)
  have e : q₀.opening tr₀ =
      ⟨bits tr₀ q₀.row, bytesOf (bits tr₀ q₀.salt), List.ofFn fun j => digestOf (bits tr₀ (q₀.path j))⟩ := rfl
  rw [e, hr, hs, salt_ones, hp]
  rfl

theorem verifier₀ : q₀.verifier key₀ d₀ = V₀ := rfl

/-- The registered constants: unit 0's constant gate at 1. -/
noncomputable def consts₀ (g : Fin (Flock.Bridge.circuitOf D₀ hU₀ 1).N) : Option Bool :=
  if g = Flock.Bridge.constGate D₀ hU₀ 1 0 then some true else none

theorem registered₀ : RegisteredRows consts₀ (Flock.Bridge.partsOf D₀ hU₀ 1) 0 q₀ r₀ d₀ tr₀ where
  root_eq := rootOf₀
  domain_eq := domainOf₀
  ok_eq := okOf₀
  consts_hold g _ b h := by
    unfold consts₀ at h
    split at h
    · rename_i hg
      subst hg
      cases h
      exact bit_const
    · cases h

/-- The one read: at copy 0 of unit 0, its table the one row `o₀` under the root `r₀` and the domain id `d₀`. -/
noncomputable def R₀ : Reads (Flock.Bridge.circuitOf D₀ hU₀ 1) 1 Unit where
  reads := {()}
  unit _ := 0
  port _ := q₀
  root _ := r₀
  domain _ := d₀
  table _ _ := o₀

theorem mem₀ : () ∈ R₀.reads := Finset.mem_singleton_self _
theorem R₀_unit (i : Unit) : R₀.unit i = 0 := rfl
theorem R₀_port (i : Unit) : R₀.port i = q₀ := rfl
theorem R₀_root (i : Unit) : R₀.root i = r₀ := rfl
theorem R₀_domain (i : Unit) : R₀.domain i = d₀ := rfl
theorem R₀_table (i : Unit) (k : ℕ) : R₀.table i k = o₀ := rfl

theorem committed₀ : R₀.Committed Verity.sha512 Verity.sha512 key₀ := by
  intro i _ k hk
  simp only [R₀_port, R₀_domain, R₀_root, R₀_table] at hk ⊢
  have hk0 : k = 0 := by
    have : k < 1 := hk
    omega
  subst hk0
  refine ⟨?_, rfl, ?_, rfl, ?_⟩
  · show (16 : ℕ) < 2 ^ 64
    norm_num
  · show 0 < 1
    omega
  · rw [show o₀.sibs = [] from rfl, Hash.Tree.climb]
    unfold r₀
    rfl

theorem pr_unit_of_not {p : Unit → Prop} (h : ¬ p ()) : Claims.Pr p = 0 := by
  classical
  unfold Claims.Pr Claims.prUniform
  rw [Finset.filter_eq_empty_iff.2 fun u _ => by cases u; exact h]
  simp

theorem pr_unit_of {p : Unit → Prop} (h : p ()) : Claims.Pr p = 1 := by
  classical
  unfold Claims.Pr Claims.prUniform
  rw [Finset.filter_true_of_mem fun u _ => by cases u; exact h]
  simp

theorem correct₀ (i : Unit) : (Flock.Bridge.partsOf D₀ hU₀ 1).Correct tr₀ (R₀.unit i) := by
  rw [R₀_unit]
  exact Pouw.PearlC.WholeCircuit.correct_eval _ _ _

theorem registered_R₀ (i : Unit) :
    RegisteredRows consts₀ (Flock.Bridge.partsOf D₀ hU₀ 1) (R₀.unit i) (R₀.port i) (R₀.root i) (R₀.domain i) tr₀ := by
  rw [R₀_unit, R₀_port, R₀_root, R₀_domain]
  exact registered₀

/-- The read's index and returned row at the counterexample: index 0, no bits. -/
theorem index₀ (i : Unit) :
    R₀.index i ((Flock.Bridge.partsOf D₀ hU₀ 1).localEval tr₀ (R₀.unit i)) < (R₀.port i).count := by
  unfold Reads.index
  rw [R₀_unit, R₀_port, localEval₀, tokOf₀]
  show 0 < 1
  omega

theorem returns₀ (i : Unit) :
    (R₀.port i).hOf ((Flock.Bridge.partsOf D₀ hU₀ 1).localEval tr₀ (R₀.unit i)) ≠
      (R₀.table i (R₀.index i ((Flock.Bridge.partsOf D₀ hU₀ 1).localEval tr₀ (R₀.unit i)))).row := by
  rw [R₀_table, R₀_unit, R₀_port, localEval₀, hOf₀]
  decide

/-- What the drawn read hashed at the counterexample: `L₀`, its opening's and its table's checks. -/
theorem drawnHashed₀ (ω : Unit) :
    R₀.drawnHashed (Flock.Bridge.partsOf D₀ hU₀ 1) Verity.sha512 key₀ (fun _ : Unit => Finset.univ) (fun _ => tr₀) ω =
      L₀ := by
  have hr : R₀.reads = {()} := rfl
  unfold Reads.drawnHashed
  simp only [hr, Finset.mem_univ, Finset.filter_true, Finset.toList_singleton, List.flatMap_cons, List.flatMap_nil,
    List.append_nil]
  unfold Reads.hashedAt Reads.index
  rw [R₀_unit, R₀_port, R₀_domain, localEval₀, tokOf₀, opening₀, R₀_table, verifier₀]
  rfl

/-- The reads' collision finder at the counterexample: every unit drawn, the trace `tr₀`, no cost. -/
noncomputable abbrev finder₀ : Claims.Finder :=
  R₀.finder (Flock.Bridge.partsOf D₀ hU₀ 1) Verity.sha512 key₀ (fun _ : Unit => Finset.univ) (fun _ => tr₀) fun _ => 0

theorem out₀ (ω : finder₀.S) : finder₀.out ω = none := by
  have h := drawnHashed₀ ω
  exact dif_neg (by rw [h]; exact no_collision₀)

theorem cost₀ (ω : finder₀.S) : finder₀.cost ω = 0 := rfl

/-- SHA-512 is collision resistant at the reads' finder: it finds no collision, at no cost. -/
theorem hcr₀ : FlockSoundness.Assumptions.SHA512CRExpected (fun m => (Verity.sha512 m).bytes)
    (Verity.Game.draw finder₀.S) (fun _ => PUnit.unit) finder₀.out fun s => (finder₀.cost s : ℝ) := by
  unfold FlockSoundness.Assumptions.SHA512CRExpected
  simp [Verity.Game.draw, Verity.Game.prob, Verity.Game.expect, out₀, cost₀]

end Accounting.ReadUnit.Witness

namespace Accounting.SecurityProofs

open Accounting.ReadUnit Accounting.ReadUnit.Witness

/-- **`TableReadBinding`'s honest lemma**: the check accepts a template at the served expert-row read's shape, 256 rows
of 3072 words (the one its text reads back to), from the fact witnessed outside Lean (`ExpertReadAccepted`). -/
theorem TableReadBinding_honest (h : ExpertReadAccepted) : ∃ t, Flock.ReadNet.check 256 3072 t = true :=
  let ⟨t, _, ht⟩ := h
  ⟨t, ht⟩

/-- **`TableReadBinding`'s nontrivial lemma**: at a unit the check doesn't fix, the reads needn't bind. At `Witness.D₀`,
a net whose read columns are inputs and the constant, every hypothesis of `ReadUnit.BindsAt` holds at no error and no
cost, SHA-512's collision resistance at the reads' finder included (`Witness.hcr₀`), and the read returns no bits
where its table's row has 16. -/
theorem TableReadBinding_nontrivial :
    ∃ (D : Flock.Desc) (hU : FlockSoundness.Built D.unroll.unit), ¬ ReadUnit.BindsAt D hU 1 0 := by
  refine ⟨D₀, hU₀, fun hB => ?_⟩
  have hle := hB 1 consts₀ (fun u => by rw [Subsingleton.elim u 0]; simp [consts₀]) R₀
    (fun i _ => ⟨0, hj₀, by rw [R₀_port, R₀_unit]⟩) (fun _ : Unit => True) (fun _ => Finset.univ) (fun _ => tr₀)
    (fun _ => 0) 0 committed₀ (le_of_eq (pr_unit_of_not (by
      rintro ⟨-, i, -, -, h⟩
      exact h ⟨correct₀ i, registered_R₀ i⟩))) hcr₀
  rw [pr_unit_of ?ev] at hle
  · norm_num [Claims.Ex] at hle
  · exact ⟨trivial, (), mem₀, Finset.mem_univ _, index₀ (), returns₀ ()⟩

end Accounting.SecurityProofs
