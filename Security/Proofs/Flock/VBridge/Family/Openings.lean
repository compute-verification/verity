import Proofs.Flock.VBridge.Family.Rep

/-!
# VBridge G at V*'s family: the openings' rows, `α` and `rec-acc`'s running sums

`vbridge_of_reps` asks every opening's unit for `OpensAt`, whose input facts are `RegBit`: each input column a
registered read whose registrant value is the opening's. V*'s openings read two private rows, the opened row and its
siblings `aux`, under fresh salts (`rec_outer.stage`, rec_outer.py:8-10, 239-242: only `out`, `acc`, `coef`, `dirs` and
`acc_out` are registered, `row` only with `link_rows`), so `OpensAt`'s `aux` conjunct fails at V*'s registration
whenever a level climbs (`H ≥ 1`). This file reads the openings from the layers themselves instead.

* `OpensAtL`: copy `j` of unit `u` in the layer `bits`, a unit the verifier parsed as `rec-open/l<LANES>-h<H>`
  carrying 1 on its constant, holds the row `r`, `acc`'s first two elements `a` and the coefficients `cf` on its input
  columns and the bytes of `A` on the first 256 columns of its `acc_out` output, at the verifier's layout
  (`keyProg_parse`). The siblings and the position are whatever the columns hold. Then `A`'s first two elements are the
  opening's consistency terms (`unitSum_of_opensAtL`), with nothing asked of the registrant; `unitSums_of_opensAtL`
  assembles `UnitSums`, and `vbridge_of_family` composes `vbridge_of_reps`' per-rep lemmas at `HUL` (`OpensAtL`) and
  `HResL` (`ResZeroAtL`, `algebraHolds_of_decodeL`).
* The family's openings, read off the layers (`rowsOf`, `accOf`) and the verifier's coins (`alphaOf`): opening `j` of
  level `ℓ` of rep `r` is copy 0 of unit `unO ℓ r j`; its row the first `levelLanes` elements of its columns, the chain
  (`chainOf`, `rec_outer.chain_order`: rep-major, then level, then query) position `p`'s row of `rec-acc` the `acc`
  input of the opening at `p`, or, past the last, the `acc_out` output of the last; `α` the one `verifierRows`' Ligerito
  run draws for the level (`ConsCtx.alpha`, `RecAlgebra.queryFold`).
* `hUL_of`: `HUL` at those definitions, from the openings' units (`HOpenUnits`, `HOneOpen`), the registered `coef` read in
  the layer being the verifier's (`HCoefCols`) and the openings reading and writing one function `V`, the rows of the
  registration's `rec-acc` (`HAccRows`). `hReads_of`: `HReads`, from `HAccRows` and the algebra reading `V` at the reps'
  ends (`HEnds`, which `readsEnds_of_regBits` gives).
-/

namespace FlockVBridge.Family

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Recursion FlockSoundness.Refine FlockSoundness.Model FlockSoundness.Audit
open Flock FlockLevel3 FlockVBridge.Compiled FlockVBridge.InnerAccepts FlockVBridge.Stages
open Flock.RecResiduals (portSizes groups)
open Flock.RecAlgebra (structureOf verifierRows el valOf)
open FlockVBridge.AccChain (Openings CoefsAre ConsAre ReadsEnds UnitSums levelLanes)
open FlockSoundness.Discharge.Hm (bitL)

/-! ## Columns as elements and bytes -/

/-- The `n` elements whose element `i` has bit `t` at column `base + 128 i + t` (`rowBytes`' order). -/
noncomputable def elems (f : ℕ → Bool) (base n : ℕ) : Array F128 :=
  Array.ofFn (n := n) fun i => ofBits fun t => f (base + 128 * i.val + t)

/-- The number whose bit `i < n` is `f i`. -/
def natOfBits (f : ℕ → Bool) : ℕ → ℕ
  | 0 => 0
  | n + 1 => Nat.bit (f 0) (natOfBits (fun i => f (i + 1)) n)

/-- The `N` bytes whose byte `i` has bit `t` at column `base + 8 i + t` (`bitL`'s order). -/
def bytesAt (f : ℕ → Bool) (base N : ℕ) : ByteArray :=
  ⟨((List.range N).map fun i => UInt8.ofNat (natOfBits (fun t => f (base + 8 * i + t)) 8)).toArray⟩

theorem size_elems (f : ℕ → Bool) (base n : ℕ) : (elems f base n).size = n := by
  simp [elems]

theorem elems_get (f : ℕ → Bool) (base : ℕ) {n i : ℕ} (hi : i < n) :
    (elems f base n)[i]! = ofBits fun t => f (base + 128 * i + t) := by
  rw [getElem!_pos (elems f base n) i (by rw [size_elems]; exact hi)]
  simp [elems]

theorem testBit_natOfBits : ∀ (f : ℕ → Bool) (n i : ℕ), (natOfBits f n).testBit i = (decide (i < n) && f i)
  | f, 0, i => by simp [natOfBits]
  | f, n + 1, 0 => by simp [natOfBits]
  | f, n + 1, i + 1 => by
    rw [natOfBits, Nat.testBit_bit_succ, testBit_natOfBits _ n i]
    simp

theorem natOfBits_lt : ∀ (f : ℕ → Bool) (n : ℕ), natOfBits f n < 2 ^ n
  | f, 0 => by simp [natOfBits]
  | f, n + 1 => by
    have := natOfBits_lt (fun i => f (i + 1)) n
    rw [natOfBits, Nat.bit_val, pow_succ]
    cases f 0 <;> simp <;> omega

theorem size_bytesAt (f : ℕ → Bool) (base N : ℕ) : (bytesAt f base N).size = N := by
  simp [bytesAt, ByteArray.size]

theorem get_pair (A : Array F128) : ∀ k < 2, (#[A[0]!, A[1]!] : Array F128)[k]! = A[k]! := by
  intro k hk
  rcases (by omega : k = 0 ∨ k = 1) with rfl | rfl <;> rfl

/-- **A row's bytes are its elements' bits**: rows agreeing on their first `n` elements agree on their first `128 n`
bits. -/
theorem bitL_rowBytes_congr {B C : Array F128} {n t : ℕ} (h : ∀ k < n, B[k]! = C[k]!) (ht : t < 128 * n) :
    bitL (MerkleScheme.rowBytes B).data.toList t = bitL (MerkleScheme.rowBytes C).data.toList t := by
  simp only [bitL_rowBytes, wbit, getD_toList]
  rw [h _ (by omega)]

theorem colsHold_elems (f : ℕ → Bool) (base n : ℕ) : ColsHold f base (MerkleScheme.rowBytes (elems f base n)) := by
  intro t ht
  rw [size_rowBytes, size_elems] at ht
  simp only [bitL_rowBytes, wbit, getD_toList]
  rw [elems_get f base (show t / 128 < n by omega), bitOf_ofBits _ (Nat.mod_lt _ (by norm_num))]
  congr 1
  omega

theorem colsHold_bytesAt (f : ℕ → Bool) (base N : ℕ) : ColsHold f base (bytesAt f base N) := by
  intro t ht
  rw [size_bytesAt] at ht
  have hl : (bytesAt f base N).data.toList.getD (t / 8) 0 =
      UInt8.ofNat (natOfBits (fun s => f (base + 8 * (t / 8) + s)) 8) := by
    simp [bytesAt, show t / 8 < N by omega]
  rw [bitL, hl, UInt8.toNat_ofNat', Nat.mod_eq_of_lt (natOfBits_lt _ 8), testBit_natOfBits]
  simp only [Nat.mod_lt t (by norm_num : 8 > 0), decide_true, Bool.true_and]
  congr 1
  omega

theorem colsHold_append {f : ℕ → Bool} {base : ℕ} {a b : ByteArray} (ha : ColsHold f base a)
    (hb : ColsHold f (base + 8 * a.size) b) : ColsHold f base (a ++ b) := by
  intro t ht
  have hl : a.data.toList.length = a.size := Array.length_toList
  rw [ByteArray.toList_data_append]
  by_cases h : t < 8 * a.size
  · rw [bitL_append_left _ (by omega)]
    exact ha t h
  · rw [bitL_append_right _ (by omega), hl, ← hb (t - 8 * a.size) (by rw [ByteArray.size_append] at ht; omega)]
    congr 1
    omega

/-! ## One opening's unit, in the layer -/

/-- Copy 0's column `col` of unit `u` in the layer `bits`. -/
noncomputable def colOf (Q : VStmt) (bits : Q.Bits) (u : Fin Q.nv) (col : ℕ) : Bool :=
  if h : 0 < Q.c.upv then copyVal (keyInst Q.hU Q.nv (Layout.outsOf Q.c Q.hU Q.nv) u) bits h col else false

theorem colOf_eq (Q : VStmt) (bits : Q.Bits) (u : Fin Q.nv) (h : 0 < Q.c.upv) :
    colOf Q bits u = copyVal (keyInst Q.hU Q.nv (Layout.outsOf Q.c Q.hU Q.nv) u) bits h :=
  funext fun _ => dite_eq_left h

/-- The first column of `acc` in the verifier's opening unit (`RecOpen.ports`: `row`, then `aux`, then `acc`). -/
abbrev accCol (LANES H : ℕ) : ℕ := 128 * LANES + 1024 * max 1 ((H + 1) / 2)

/-- The first output column of the verifier's opening unit (`unit_check`). -/
noncomputable abbrev outCol (LANES H : ℕ) : ℕ :=
  Hm.outStart ((Flock.RecOpen.recOpenNet LANES H (Flock.RecOpen.ports LANES H)).run Flock.HmNets.St.empty).2
    (Flock.RecOpen.wordPorts (unitIn LANES H)) (Flock.RecOpen.wordPorts 2048)

/-- **Copy `j` of unit `u` in the layer `bits` opens the row `r` with running sums `a` before and `A` after, at the
coefficients `cf`**: the unit is one the verifier parsed as `rec-open/l<LANES>-h<H>` (`RecOpen.params`), the layer
carries 1 on its constant, and at the verifier's layout the copy's columns hold `r`'s bytes from column 0, the bytes of
`a`'s first two elements from `accCol`, `cf`'s bytes 1,024 columns on, and the bytes of `A`'s first two elements from
the `acc_out` half of its outputs. Its siblings, the rest of its `acc` and its position bits are what the columns hold.
`OpensAt` without the registrant: every fact is about the layer. -/
def OpensAtL (Q : VStmt) (bits : Q.Bits) (u : Fin Q.nv) (j LANES : ℕ) (r a cf A : Array F128) : Prop :=
  ∃ (hj : j < Q.c.upv), bits (Const.Prog.oneAt (Placed.keyProg Q.c Q.hU Q.nv) u) = true ∧
    Q.c.unit.unitLog ≤ 32 ∧ ∃ H, Flock.RecOpen.params Q.c.unit.name = some (LANES, H) ∧
    r.size = LANES ∧ cf.size = 2 * LANES ∧
    ColsHold (copyVal (keyInst Q.hU Q.nv (Layout.outsOf Q.c Q.hU Q.nv) u) bits hj) 0 (MerkleScheme.rowBytes r) ∧
    (∀ t < 256, copyVal (keyInst Q.hU Q.nv (Layout.outsOf Q.c Q.hU Q.nv) u) bits hj (accCol LANES H + t) =
      bitL (MerkleScheme.rowBytes a).data.toList t) ∧
    ColsHold (copyVal (keyInst Q.hU Q.nv (Layout.outsOf Q.c Q.hU Q.nv) u) bits hj) (accCol LANES H + 1024)
      (MerkleScheme.rowBytes cf) ∧
    ∀ t < 256, copyVal (keyInst Q.hU Q.nv (Layout.outsOf Q.c Q.hU Q.nv) u) bits hj (outCol LANES H + 1024 + t) =
      bitL (MerkleScheme.rowBytes A).data.toList t

/-- **An opening's running sum, from its unit's columns in the layer** (`keyProg_parse`, `unitSum_of_cols`): `A`'s
first two elements are the opening's consistency terms. The session's circuit is the verifier's parse of its file
(`ZkOuter.hc`); nothing else of the session is read. -/
theorem unitSum_of_opensAtL {Q : VStmt} {Reg' : Type} (zo : Q.Outer Reg') {bits : Q.Bits} {u : Fin Q.nv}
    {j LANES : ℕ} {r a cf A : Array F128} (h : OpensAtL Q bits u j LANES r a cf A) :
    ∀ k < 2, A[k]! = consTerm LANES r a cf k := by
  obtain ⟨hj, hone, hu, H, hp, hr, hcf, hrow, hacc, hcoef, hout⟩ := h
  have ha' := get_pair a
  have hO := keyProg_parse zo.hc Q.hU Q.nv (Layout.outsOf Q.c Q.hU Q.nv) bits u hj hone (MerkleScheme.sha512 "")
    (fun _ _ => rfl) hp hu r #[a[0]!, a[1]!] cf
    (bytesAt (copyVal (keyInst Q.hU Q.nv (Layout.outsOf Q.c Q.hU Q.nv) u) bits hj) (accCol LANES H + 256) 96)
    (natOfBits (fun h => copyVal (keyInst Q.hU Q.nv (Layout.outsOf Q.c Q.hU Q.nv) u) bits hj
      (accCol LANES H + 1024 + 256 * LANES + h)) H)
    ((List.range H).map fun i =>
      bytesAt (copyVal (keyInst Q.hU Q.nv (Layout.outsOf Q.c Q.hU Q.nv) u) bits hj) (128 * LANES + 512 * i) 64)
    hr rfl hcf (size_bytesAt _ _ _) (by simp) hrow
    (fun i hi => by
      simp only [List.getElem_map, List.getElem_range]
      exact ⟨size_bytesAt _ _ _, colsHold_bytesAt _ _ _⟩)
    (colsHold_append
      (fun t ht => by
        rw [size_rowBytes] at ht
        exact (hacc t (by simpa using ht)).trans (bitL_rowBytes_congr (t := t) ha' (by simpa using ht)).symm)
      (by
        rw [size_rowBytes]
        exact colsHold_bytesAt _ _ _))
    hcoef
    (fun h hh => by
      rw [testBit_natOfBits]
      simp [hh])
  have hX : (Flock.climb (MerkleScheme.sha512 "") (Flock.Sha512.hash (MerkleScheme.rowBytes r))
      (natOfBits (fun h => copyVal (keyInst Q.hU Q.nv (Layout.outsOf Q.c Q.hU Q.nv) u) bits hj
        (accCol LANES H + 1024 + 256 * LANES + h)) H)
      ((List.range H).map fun i =>
        bytesAt (copyVal (keyInst Q.hU Q.nv (Layout.outsOf Q.c Q.hU Q.nv) u) bits hj) (128 * LANES + 512 * i) 64) ++
      ⟨Array.replicate 64 0⟩).size = 128 := by
    rw [ByteArray.size_append,
      AccChain.climb_size_of (MerkleScheme.sha512 "") (fun _ _ => rfl) _ _ _ (FlockSoundness.Refine.sha512_size _)]
    rfl
  intro k hk
  rw [AccChain.unitSum_of_cols hX hO hout k hk]
  exact AccChain.consTerm_acc _ _ _ _ _ (ha' k hk)

/-- **`UnitSums` from each opening's unit in the layers**: `unitSums_of_decode` with `OpensAtL` in place of `OpensAt`. -/
theorem unitSums_of_opensAtL {S : ℕ} {Vs : Fin S → VStmt} {Reg' : Type} (zs : (s : Fin S) → (Vs s).Outer Reg')
    (L : (s : Fin S) → (Vs s).Bits) (sh : Shape) (acc : ℕ → Array F128) (o : Openings) (sess : ℕ → Fin S)
    (un : (ℓ : ℕ) → ℕ → Fin (Vs (sess ℓ)).nv) (cp : ℕ → ℕ → ℕ)
    (h : ∀ ℓ < sh.sched.levels.size, ∀ j < o.q ℓ,
      OpensAtL (Vs (sess ℓ)) (L (sess ℓ)) (un ℓ j) (cp ℓ j) (levelLanes sh ℓ) (o.rows ℓ)[j]!
        #[(acc (o.start ℓ + j))[0]!, (acc (o.start ℓ + j))[1]!] (o.coef ℓ j) (acc (o.start ℓ + j + 1))) :
    UnitSums sh acc o := by
  intro ℓ hℓ j hj k hk
  rw [unitSum_of_opensAtL (zs (sess ℓ)) (h ℓ hℓ j hj) k hk]
  refine AccChain.consTerm_acc _ _ _ _ _ ?_
  rcases (by omega : k = 0 ∨ k = 1) with rfl | rfl <;> rfl

/-! ## `rec-acc`'s rows at an opening, from the registrant -/

section Reg

variable {Q : VStmt} {Reg' : Type} (zo : Q.Outer Reg')

/-- **An opening's `acc_out`, from its registered output columns** (`copyVal_regOut`): in a `Good` layer, when the
first 256 columns of the `acc_out` half of copy 0's outputs are registered bits of the bytes of `A` (`RegOut`), the
output's first two elements are `A`'s. -/
theorem accOut_of_regOut {bits : Q.Bits} (hG : zo.Good bits) {u : Fin Q.nv} (hj : 0 < Q.c.upv) {LANES H : ℕ}
    (hp : Flock.RecOpen.params Q.c.unit.name = some (LANES, H)) {A : Array F128}
    (h : ∀ t < 256, AccChain.RegOut zo u hj (outCol LANES H + 1024 + t)
      (bitL (MerkleScheme.rowBytes A).data.toList t)) :
    ∀ i < 2, (elems (colOf Q bits u) (outCol LANES H + 1024) 2)[i]! = A[i]! := by
  have hnIn : (Rows.ofNet Q.hU).nIn ≤ outCol LANES H := by
    rw [Layout.ofNet_nIn_eq, inWords_unit (Flock.RecOpen.check_ok (parse_facts zo.hc).recNet hp).2]
    exact Decode.inRows_le_outStart _ _ _
  intro i hi
  refine AccChain.getElem!_of_bitL fun s hs => ?_
  rw [← colsHold_elems (colOf Q bits u) (outCol LANES H + 1024) 2 (128 * i + s)
    (by rw [size_rowBytes, size_elems]; omega), colOf_eq Q bits u hj]
  exact AccChain.copyVal_regOut zo hG.2 (keyProg_correct Q.hU Q.nv hG.1 u) hj (by omega) (h _ (by omega))

/-- **An opening's `acc`, from its registered input columns** (`copyVal_regBit`): in a layer agreeing with the
registrant, when the 256 columns from `base` of copy 0 are registered bits of the bytes of `A` (`RegBit`) on wires the
unit reads, the elements there are `A`'s first two. -/
theorem accIn_of_regBit {bits : Q.Bits} (hA : zo.RegAgree bits) {u : Fin Q.nv} (hj : 0 < Q.c.upv) {base : ℕ}
    {A : Array F128}
    (h : ∀ t < 256, Decode.RegBit zo u 0 (base + t) (bitL (MerkleScheme.rowBytes A).data.toList t))
    (hio : ∀ t < 256, ∀ hc : base + t < (Rows.ofNet Q.hU).w,
      (keyInst Q.hU Q.nv (Layout.outsOf Q.c Q.hU Q.nv) u).wire
          ((Rows.ofNet Q.hU).stackCol Q.c.upv hj ⟨base + t, hc⟩) ∈
        ((Placed.keyProg Q.c Q.hU Q.nv).partition (Layout.outsOf Q.c Q.hU Q.nv)).io u) :
    ∀ i < 2, (elems (colOf Q bits u) base 2)[i]! = A[i]! := by
  intro i hi
  refine AccChain.getElem!_of_bitL fun s hs => ?_
  have ht : 128 * i + s < 256 := by omega
  obtain ⟨hcol, -⟩ := h _ ht
  have e1 := copyVal_input (keyInst Q.hU Q.nv (Layout.outsOf Q.c Q.hU Q.nv) u) bits hj hcol
  have e2 := copyVal_input (keyInst Q.hU Q.nv (Layout.outsOf Q.c Q.hU Q.nv) u) (Decode.fill zo bits) hj hcol
  have e3 := Decode.fill_io zo hA (hio _ ht (by simp only [Rows.w]; omega))
  rw [← colsHold_elems (colOf Q bits u) base 2 (128 * i + s) (by rw [size_rowBytes, size_elems]; omega),
    colOf_eq Q bits u hj, e1, ← e3, ← e2]
  exact Decode.copyVal_regBit zo bits hj (h _ ht)

end Reg

/-! ## The chain's positions -/

/-- Level `ℓ`'s openings `(ℓ, j)` for `ℓ < nL`, by level, then query. -/
def block (q : ℕ → ℕ) (nL : ℕ) : List (ℕ × ℕ) :=
  (List.range nL).flatMap fun ℓ => (List.range (q ℓ)).map fun j => (ℓ, j)

theorem block_succ (q : ℕ → ℕ) (nL : ℕ) :
    block q (nL + 1) = block q nL ++ (List.range (q nL)).map fun j => (nL, j) := by
  simp [block, List.range_succ, List.flatMap_append]

theorem sum_range_succ' (q : ℕ → ℕ) (ℓ : ℕ) :
    ((List.range (ℓ + 1)).map q).sum = ((List.range ℓ).map q).sum + q ℓ := by
  simp [List.range_succ, List.sum_append]

theorem sum_range_mono (q : ℕ → ℕ) {a : ℕ} : ∀ {b : ℕ}, a ≤ b →
    ((List.range a).map q).sum ≤ ((List.range b).map q).sum
  | 0, h => by rw [Nat.le_zero.mp h]
  | b + 1, h => by
    rcases Nat.lt_or_eq_of_le h with h | rfl
    · rw [sum_range_succ']
      exact le_trans (sum_range_mono q (by omega)) (Nat.le_add_right _ _)
    · exact le_rfl

theorem length_block (q : ℕ → ℕ) : ∀ nL, (block q nL).length = ((List.range nL).map q).sum
  | 0 => rfl
  | nL + 1 => by
    rw [block_succ, List.length_append, length_block q nL, sum_range_succ', List.length_map, List.length_range]

theorem getElem?_block (q : ℕ → ℕ) : ∀ {nL ℓ j : ℕ}, ℓ < nL → j < q ℓ →
    (block q nL)[((List.range ℓ).map q).sum + j]? = some (ℓ, j)
  | 0, _, _, hℓ, _ => absurd hℓ (Nat.not_lt_zero _)
  | nL + 1, ℓ, j, hℓ, hj => by
    rw [block_succ]
    rcases Nat.lt_or_eq_of_le (Nat.le_of_lt_succ hℓ) with hl | rfl
    · have hm := sum_range_mono q (show ℓ + 1 ≤ nL by omega)
      rw [sum_range_succ'] at hm
      rw [List.getElem?_append_left (by rw [length_block]; omega), getElem?_block q hl hj]
    · rw [List.getElem?_append_right (by rw [length_block]; omega), length_block, Nat.add_sub_cancel_left,
        List.getElem?_map, List.getElem?_range hj]
      rfl

theorem block_inv (q : ℕ → ℕ) : ∀ {nL p ℓ j : ℕ}, (block q nL)[p]? = some (ℓ, j) →
    ℓ < nL ∧ j < q ℓ ∧ p = ((List.range ℓ).map q).sum + j
  | 0, p, ℓ, j, h => by simp [block] at h
  | nL + 1, p, ℓ, j, h => by
    rw [block_succ] at h
    by_cases hp : p < (block q nL).length
    · rw [List.getElem?_append_left hp] at h
      obtain ⟨h1, h2, h3⟩ := block_inv q h
      exact ⟨by omega, h2, h3⟩
    · rw [List.getElem?_append_right (by omega), List.getElem?_map] at h
      rw [length_block] at hp h
      cases hr : (List.range (q nL))[p - ((List.range nL).map q).sum]? with
      | none =>
        rw [hr] at h
        cases h
      | some m =>
        rw [hr] at h
        injection h with h
        injection h with h1 h2
        subst h1 h2
        obtain ⟨hlt, he⟩ := List.getElem?_eq_some_iff.mp hr
        rw [List.getElem_range] at he
        rw [List.length_range] at hlt
        exact ⟨by omega, by omega, by omega⟩

section family

variable {nT nR : ℕ} {T : Fin nT → Type} {CT : Fin nR → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
variable {cls : HClass} {regs : List (Region cls.kLog (cls.m - cls.kLog))} {sch : Flock.Schedule}
variable {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => flockInnerCL T CT execArith H512 enc512 InnerAccepts.leafPlain cls regs (schedOf sch)

namespace VStarFamily

variable (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂)

/-! ## The openings, read off the layers -/

/-- The openings of both reps in chain order (`rec_outer.chain_order`): rep-major, then level, then query. -/
def chainOf : List (Fin 2 × ℕ × ℕ) :=
  (block F.q F.sh.sched.levels.size).map (fun x => ((0 : Fin 2), x)) ++
    (block F.q F.sh.sched.levels.size).map (fun x => ((1 : Fin 2), x))

theorem length_chainOf : F.chainOf.length = 2 * F.nOpen := by
  simp only [chainOf, List.length_append, List.length_map, length_block, nOpen]
  omega

/-- Opening `j` of level `ℓ` of rep `r` is at chain position `r · nOpen + Σ_{ℓ' < ℓ} q ℓ' + j`
(`Openings.start`). -/
theorem getElem?_chainOf (r : Fin 2) {ℓ j : ℕ} (hℓ : ℓ < F.sh.sched.levels.size) (hj : j < F.q ℓ) :
    F.chainOf[r.val * F.nOpen + ((List.range ℓ).map F.q).sum + j]? = some (r, ℓ, j) := by
  have hb := getElem?_block F.q hℓ hj
  have hm := sum_range_mono F.q (show ℓ + 1 ≤ F.sh.sched.levels.size by omega)
  rw [sum_range_succ'] at hm
  have hlen : (block F.q F.sh.sched.levels.size).length = F.nOpen := length_block _ _
  rcases r with ⟨_ | _ | r, hr⟩
  · rw [chainOf, Nat.zero_mul, Nat.zero_add,
      List.getElem?_append_left (by rw [List.length_map, hlen]; unfold nOpen; omega), List.getElem?_map, hb]
    rfl
  · rw [chainOf, Nat.one_mul, List.getElem?_append_right (by rw [List.length_map, hlen]; omega), List.length_map,
      hlen, Nat.add_assoc, Nat.add_sub_cancel_left, List.getElem?_map, hb]
    rfl
  · omega

/-- The chain's position `p` is opening `j` of level `ℓ` of rep `r` only at `getElem?_chainOf`'s position. -/
theorem chainOf_inv {p : ℕ} {x : Fin 2 × ℕ × ℕ} (h : F.chainOf[p]? = some x) :
    x.2.1 < F.sh.sched.levels.size ∧ x.2.2 < F.q x.2.1 ∧
      p = x.1.val * F.nOpen + ((List.range x.2.1).map F.q).sum + x.2.2 := by
  have hlen : (block F.q F.sh.sched.levels.size).length = F.nOpen := length_block _ _
  obtain ⟨r, ℓ, j⟩ := x
  unfold chainOf at h
  by_cases hp : p < F.nOpen
  · rw [List.getElem?_append_left (by rw [List.length_map, hlen]; exact hp), List.getElem?_map] at h
    cases hb : (block F.q F.sh.sched.levels.size)[p]? with
    | none =>
      rw [hb] at h
      cases h
    | some y =>
      rw [hb] at h
      injection h with h
      injection h with h1 h2
      subst h1 h2
      obtain ⟨h1, h2, h3⟩ := block_inv F.q hb
      exact ⟨h1, h2, by simp only [Fin.val_zero, Nat.zero_mul, Nat.zero_add]; exact h3⟩
  · rw [List.getElem?_append_right (by rw [List.length_map, hlen]; omega), List.length_map, hlen,
      List.getElem?_map] at h
    cases hb : (block F.q F.sh.sched.levels.size)[p - F.nOpen]? with
    | none =>
      rw [hb] at h
      cases h
    | some y =>
      rw [hb] at h
      injection h with h
      injection h with h1 h2
      subst h1 h2
      obtain ⟨h1, h2, h3⟩ := block_inv F.q hb
      refine ⟨h1, h2, ?_⟩
      show p = 1 * F.nOpen + ((List.range ℓ).map F.q).sum + j
      omega

/-- Opening `j` of level `ℓ` of rep `r`: copy 0's columns of unit `unO ℓ r j` in the level's session. -/
noncomputable def openCol (l : (j : Fin F.stmts.size) → (F.Vs j).Bits) (r : Fin 2) (ℓ j : ℕ) : ℕ → Bool :=
  colOf (F.Vs (F.sessO ℓ)) (l (F.sessO ℓ)) (F.unO ℓ r j)

/-- Level `ℓ`'s climb, `H` of its unit's name `rec-open/l<LANES>-h<H>`. -/
def depth (ℓ : ℕ) : ℕ := ((Flock.RecOpen.params (F.Vs (F.sessO ℓ)).c.unit.name).getD (0, 0)).2

/-- **Each rep's opened rows of each level**: opening `j`'s row is the `levelLanes` elements of its `row` columns. -/
noncomputable def rowsOf (l : (j : Fin F.stmts.size) → (F.Vs j).Bits) (r : Fin 2) (ℓ : ℕ) : Array (Array F128) :=
  (Array.range (F.q ℓ)).map fun j => elems (F.openCol l r ℓ j) 0 (levelLanes F.sh ℓ)

/-- The first two elements of an opening's `acc` input. -/
noncomputable def accIn (l : (j : Fin F.stmts.size) → (F.Vs j).Bits) (x : Fin 2 × ℕ × ℕ) : Array F128 :=
  elems (F.openCol l x.1 x.2.1 x.2.2) (accCol (levelLanes F.sh x.2.1) (F.depth x.2.1)) 2

/-- The first two elements of an opening's `acc_out` output. -/
noncomputable def accOut (l : (j : Fin F.stmts.size) → (F.Vs j).Bits) (x : Fin 2 × ℕ × ℕ) : Array F128 :=
  elems (F.openCol l x.1 x.2.1 x.2.2) (outCol (levelLanes F.sh x.2.1) (F.depth x.2.1) + 1024) 2

/-- The `coef` an opening's unit reads, in the layer. -/
noncomputable def coefCols (l : (j : Fin F.stmts.size) → (F.Vs j).Bits) (r : Fin 2) (ℓ j : ℕ) : Array F128 :=
  elems (F.openCol l r ℓ j) (accCol (levelLanes F.sh ℓ) (F.depth ℓ) + 1024) (2 * levelLanes F.sh ℓ)

/-- **`rec-acc`'s rows** (`rec_outer.Chain`): row `p` is the `acc` the opening at chain position `p` reads (rec_outer.py:
31: `acc` at row `p`), and the row after the last opening its `acc_out`. -/
noncomputable def accOf (l : (j : Fin F.stmts.size) → (F.Vs j).Bits) (p : ℕ) : Array F128 :=
  match F.chainOf[p]? with
  | some x => F.accIn l x
  | none =>
    match F.chainOf[p - 1]? with
    | some x => F.accOut l x
    | none => #[]

/-- **Each rep's `α` of each level** (`rec_outer.coefficients`' `α`): the one the verifier's Ligerito run draws at the
level's queries (`RecAlgebra.queryFold`, `levelsLoop`: one `ConsCtx` per level, in level order), at the rep's coins. -/
def alphaOf (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin)) (l : (j : Fin F.stmts.size) → (F.Vs j).Bits) (r : Fin 2)
    (ℓ : ℕ) : Array F128 :=
  match verifierRows F.sh (F.rep₀ R ω t l r).rounds (F.rep₀ R ω t l r).extras (F.rep₀ R ω t l r).comb with
  | .ok (_, info, _) => (info.lg.conss[ℓ]!).alpha
  | .error _ => #[]

theorem alphaOf_ok (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin)) (l : (j : Fin F.stmts.size) → (F.Vs j).Bits)
    (r : Fin 2) {v info kappa}
    (h : verifierRows F.sh (F.rep₀ R ω t l r).rounds (F.rep₀ R ω t l r).extras (F.rep₀ R ω t l r).comb =
      .ok (v, info, kappa)) (ℓ : ℕ) : F.alphaOf R ω t l r ℓ = (info.lg.conss[ℓ]!).alpha := by
  unfold alphaOf
  rw [h]

theorem rowsOf_get (l : (j : Fin F.stmts.size) → (F.Vs j).Bits) (r : Fin 2) {ℓ j : ℕ} (hj : j < F.q ℓ) :
    (F.rowsOf l r ℓ)[j]! = elems (F.openCol l r ℓ j) 0 (levelLanes F.sh ℓ) := by
  rw [rowsOf, getElem!_pos _ j (by simpa using hj)]
  simp

theorem accOf_of_some (l : (j : Fin F.stmts.size) → (F.Vs j).Bits) {p : ℕ} {x : Fin 2 × ℕ × ℕ}
    (h : F.chainOf[p]? = some x) : F.accOf l p = F.accIn l x := by
  unfold accOf
  rw [h]

theorem accOf_of_none (l : (j : Fin F.stmts.size) → (F.Vs j).Bits) {p : ℕ} {x : Fin 2 × ℕ × ℕ}
    (h : F.chainOf[p]? = none) (h' : F.chainOf[p - 1]? = some x) : F.accOf l p = F.accOut l x := by
  unfold accOf
  rw [h, h']

/-- **`rec-acc`'s rows are one function `W`'s**, on their first two elements, through the chain's last row, when every
opening reads `W` at its position and writes it at the next. -/
theorem accOf_eq (l : (j : Fin F.stmts.size) → (F.Vs j).Bits) (W : ℕ → Array F128)
    (hW : ∀ p x, F.chainOf[p]? = some x → ∀ k < 2,
      (F.accIn l x)[k]! = (W p)[k]! ∧ (F.accOut l x)[k]! = (W (p + 1))[k]!)
    (hn : 0 < F.nOpen) {p : ℕ} (hp : p ≤ 2 * F.nOpen) : ∀ k < 2, (F.accOf l p)[k]! = (W p)[k]! := by
  intro k hk
  rcases Nat.lt_or_eq_of_le hp with hp | rfl
  · have hlt : p < F.chainOf.length := by rw [length_chainOf]; exact hp
    rw [F.accOf_of_some l (List.getElem?_eq_getElem hlt)]
    exact (hW p _ (List.getElem?_eq_getElem hlt) k hk).1
  · have hlt : 2 * F.nOpen - 1 < F.chainOf.length := by rw [length_chainOf]; omega
    rw [F.accOf_of_none l (List.getElem?_eq_none (by rw [length_chainOf])) (List.getElem?_eq_getElem hlt)]
    have := (hW _ _ (List.getElem?_eq_getElem hlt) k hk).2
    rwa [show 2 * F.nOpen - 1 + 1 = 2 * F.nOpen by omega] at this

/-- `HAccRows`' facts at a chain position: the opening there reads `W` at the position and writes it at the next. -/
theorem accRows_pos (l : (j : Fin F.stmts.size) → (F.Vs j).Bits) (W : ℕ → Array F128)
    (hW : ∀ r : Fin 2, ∀ ℓ < F.sh.sched.levels.size, ∀ j < F.q ℓ, ∀ i < 2,
      (F.accIn l (r, ℓ, j))[i]! = (W (r.val * F.nOpen + ((List.range ℓ).map F.q).sum + j))[i]! ∧
        (F.accOut l (r, ℓ, j))[i]! = (W (r.val * F.nOpen + ((List.range ℓ).map F.q).sum + j + 1))[i]!) :
    ∀ p x, F.chainOf[p]? = some x → ∀ i < 2,
      (F.accIn l x)[i]! = (W p)[i]! ∧ (F.accOut l x)[i]! = (W (p + 1))[i]! := by
  rintro p ⟨r, ℓ, j⟩ hx i hi
  obtain ⟨h1, h2, h3⟩ := F.chainOf_inv hx
  dsimp only at h1 h2 h3
  subst h3
  exact hW r ℓ h1 j h2 i hi

end VStarFamily

/-! ## The obligations, restated -/

namespace VStarFamily

variable (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ)

/-- **`hu` at the layers** (`OpensAtL` in place of `OpensAt`): at every accepted outer proof, rep `r`'s instance of
opening `j` of level `ℓ` opens the rep's row at the verifier's coefficients, from `rec-acc`'s row at its chain position
to the next, in the layer. -/
def HUL : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ r, ∀ ℓ < F.sh.sched.levels.size,
    ∀ j < (F.ops R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).q ℓ,
      OpensAtL (F.Vs (F.sessO ℓ)) (layers F.zs dg N k Rw R ω t R₂ τ (F.sessO ℓ)) (F.unO ℓ r j) (cpO ℓ r j)
        (levelLanes F.sh ℓ) ((F.ops R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).rows ℓ)[j]!
        #[(F.acc (layers F.zs dg N k Rw R ω t R₂ τ)
            ((F.ops R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).start ℓ + j))[0]!,
          (F.acc (layers F.zs dg N k Rw R ω t R₂ τ)
            ((F.ops R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).start ℓ + j))[1]!]
        ((F.ops R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).coef ℓ j)
        (F.acc (layers F.zs dg N k Rw R ω t R₂ τ)
          ((F.ops R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).start ℓ + j + 1))

/-- **The openings' units** (`Stage.lean:56-66`, `Compose.lean:224`): level `ℓ`'s statement is a unit the verifier
parsed as `rec-open/l<2^lanes>-h<H>` (`RecOpen.params`), with a copy per instance and `unit_log ≤ 32`. -/
def HOpenUnits : Prop :=
  ∀ ℓ < F.sh.sched.levels.size, 0 < (F.Vs (F.sessO ℓ)).c.upv ∧ (F.Vs (F.sessO ℓ)).c.unit.unitLog ≤ 32 ∧
    ∃ H, Flock.RecOpen.params (F.Vs (F.sessO ℓ)).c.unit.name = some (levelLanes F.sh ℓ, H)

/-- **The openings' constants**: at every accepted outer proof, each opening's unit carries 1 on its constant gate. -/
def HOneOpen : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ r, ∀ ℓ < F.sh.sched.levels.size,
    ∀ j < F.q ℓ, layers F.zs dg N k Rw R ω t R₂ τ (F.sessO ℓ)
      (Const.Prog.oneAt (Placed.keyProg (F.Vs (F.sessO ℓ)).c (F.Vs (F.sessO ℓ)).hU (F.Vs (F.sessO ℓ)).nv)
        (F.unO ℓ r j)) = true

/-- Obligation: discharged by a verifier check that the registered `coef` is its own (`rec_outer.coefficients`).
**The registered `coef` is the verifier's** (`rec_outer.coefficients`, rec_outer.py:159-161): at every accepted
outer proof, the `coef` each opening's unit reads in the layer is `coef`, computed from the rep's coins. -/
def HCoefCols : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ r, ∀ ℓ < F.sh.sched.levels.size,
    ∀ j < F.q ℓ, F.coefCols (layers F.zs dg N k Rw R ω t R₂ τ) r ℓ j =
      F.coef R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r ℓ j

/-- Obligation: discharged by the openings' reads and writes of `rec-acc`'s registered rows (rec_outer.py:31, 237-238).
**The openings read and write `rec-acc`'s rows `V`** (rec_outer.py:31, 237-238): at every accepted outer proof,
opening `j` of level `ℓ` of rep `r`, at chain position `p` (`getElem?_chainOf`), reads row `p` as its `acc` and writes
row `p + 1` as its `acc_out`, on their first two elements, in the layer. `V` is the rows the registration registers as
`rec-acc`; `accIn_of_regBit` and `accOut_of_regOut` give each fact from the opening's registered columns. -/
def HAccRows (V : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → ℕ → Array F128) : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ r : Fin 2, ∀ ℓ < F.sh.sched.levels.size,
    ∀ j < F.q ℓ, ∀ i < 2,
      (F.accIn (layers F.zs dg N k Rw R ω t R₂ τ) (r, ℓ, j))[i]! =
          (V R ω t R₂ (r.val * F.nOpen + ((List.range ℓ).map F.q).sum + j))[i]! ∧
        (F.accOut (layers F.zs dg N k Rw R ω t R₂ τ) (r, ℓ, j))[i]! =
          (V R ω t R₂ (r.val * F.nOpen + ((List.range ℓ).map F.q).sum + j + 1))[i]!

/-- Obligation: discharged by V*'s algebra's reads of `rec-acc`'s registered rows at each rep's ends.
**V*'s algebra reads `rec-acc`'s rows `V` at each rep's ends** (`Chain.rep_ends`; `readsEnds_of_regBits` gives it
from the algebra unit's registered bits). -/
def HEnds (V : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → ℕ → Array F128) : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ r,
    ReadsEnds F.sh (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r) (V R ω t R₂)
      (F.ops R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r)

/-- **`HUL` at the family's openings read off the layers**: when `rows` and `acc` are `rowsOf` and `accOf`, every
opening's unit is the verifier's (`HOpenUnits`) and carries 1 on its constant (`HOneOpen`), reads the verifier's `coef`
(`HCoefCols`), and the openings read and write one function `V` (`HAccRows`), `HUL` holds. -/
theorem hUL_of (hrows : F.rows = F.rowsOf) (hacc : F.acc = F.accOf) (hU : F.HOpenUnits)
    (hone : F.HOneOpen dg N k Rw) (hcoef : F.HCoefCols dg N k Rw)
    (V : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → ℕ → Array F128) (hV : F.HAccRows dg N k Rw V) :
    F.HUL dg N k Rw := by
  intro R ω t R₂ τ o h r ℓ hℓ j hj
  have hj' : j < F.q ℓ := hj
  have hV' := F.accRows_pos _ _ (hV R ω t R₂ τ o h)
  obtain ⟨hup, hu, H, hp⟩ := hU ℓ hℓ
  have hD : F.depth ℓ = H := by
    unfold depth
    rw [hp]
    rfl
  have hcol : F.openCol (layers F.zs dg N k Rw R ω t R₂ τ) r ℓ j =
      copyVal (keyInst (F.Vs (F.sessO ℓ)).hU (F.Vs (F.sessO ℓ)).nv
        (Layout.outsOf (F.Vs (F.sessO ℓ)).c (F.Vs (F.sessO ℓ)).hU (F.Vs (F.sessO ℓ)).nv) (F.unO ℓ r j))
        (layers F.zs dg N k Rw R ω t R₂ τ (F.sessO ℓ)) (j := cpO ℓ r j) hup :=
    colOf_eq _ _ _ hup
  have hrow : (F.ops R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).rows ℓ =
      F.rowsOf (layers F.zs dg N k Rw R ω t R₂ τ) r ℓ := by
    show F.rows _ r ℓ = _
    rw [hrows]
  have hpos := F.getElem?_chainOf r hℓ hj'
  have hst : (F.ops R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).start ℓ + j =
      r.val * F.nOpen + ((List.range ℓ).map F.q).sum + j := rfl
  have hin : F.acc (layers F.zs dg N k Rw R ω t R₂ τ)
      ((F.ops R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).start ℓ + j) =
      F.accIn (layers F.zs dg N k Rw R ω t R₂ τ) (r, ℓ, j) := by
    rw [hacc, hst]
    exact F.accOf_of_some _ hpos
  have hout : ∀ i < 2, (F.acc (layers F.zs dg N k Rw R ω t R₂ τ)
      ((F.ops R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).start ℓ + j + 1))[i]! =
      (F.accOut (layers F.zs dg N k Rw R ω t R₂ τ) (r, ℓ, j))[i]! := by
    intro i hi
    rw [hacc, hst]
    cases hnx : F.chainOf[r.val * F.nOpen + ((List.range ℓ).map F.q).sum + j + 1]? with
    | some y =>
      rw [F.accOf_of_some _ hnx, (hV' _ y hnx i hi).1, (hV' _ _ hpos i hi).2]
    | none =>
      rw [F.accOf_of_none _ hnx (by rw [Nat.add_sub_cancel]; exact hpos)]
  have hcf : (F.ops R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).coef ℓ j =
      F.coefCols (layers F.zs dg N k Rw R ω t R₂ τ) r ℓ j :=
    (hcoef R ω t R₂ τ o h r ℓ hℓ j hj').symm
  refine ⟨hup, hone R ω t R₂ τ o h r ℓ hℓ j hj', hu, H, hp, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hrow, F.rowsOf_get _ r hj', size_elems]
  · rw [hcf, coefCols, size_elems]
  · rw [hrow, F.rowsOf_get _ r hj', ← hcol]
    exact colsHold_elems _ _ _
  · intro b hb
    rw [hin, bitL_rowBytes_congr (t := b) (get_pair _) (by omega), accIn, ← hcol, hD]
    exact colsHold_elems _ _ _ b (by rw [size_rowBytes, size_elems]; omega)
  · rw [hcf, coefCols, ← hcol, hD]
    exact colsHold_elems _ _ _
  · intro b hb
    rw [bitL_rowBytes_congr (t := b) hout (by omega), accOut, ← hcol, hD]
    exact colsHold_elems _ _ _ b (by rw [size_rowBytes, size_elems]; omega)

theorem readsEnds_congr {sh : Shape} {d : RepData} {A B : ℕ → Array F128} {o : Openings} (h : ReadsEnds sh d A o)
    (h0 : ∀ i < 2, (A o.p0)[i]! = (B o.p0)[i]!)
    (h1 : ∀ i < 2, (A (o.start sh.sched.levels.size))[i]! = (B (o.start sh.sched.levels.size))[i]!) :
    ReadsEnds sh d B o :=
  fun r hr => ⟨(h r hr).1.trans (h0 r hr), (h r hr).2.trans (h1 r hr)⟩

/-- **`HReads` at `accOf`**: the openings read and write `V` (`HAccRows`) and the algebra reads `V` at the reps' ends
(`HEnds`), so it reads `accOf` there; each rep has an opening. -/
theorem hReads_of (hacc : F.acc = F.accOf) (hn : 0 < F.nOpen)
    (V : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → ℕ → Array F128) (hV : F.HAccRows dg N k Rw V)
    (hE : F.HEnds dg N k Rw V) : F.HReads dg N k Rw := by
  intro R ω t R₂ τ o h r
  rw [hacc]
  have hW := fun p (hp : p ≤ 2 * F.nOpen) =>
    F.accOf_eq (layers F.zs dg N k Rw R ω t R₂ τ) (V R ω t R₂) (F.accRows_pos _ _ (hV R ω t R₂ τ o h)) hn hp
  have hr := r.isLt
  refine readsEnds_congr (hE R ω t R₂ τ o h r) (fun i hi => (hW _ ?_ i hi).symm) (fun i hi => (hW _ ?_ i hi).symm)
  · show r.val * F.nOpen ≤ 2 * F.nOpen
    exact Nat.mul_le_mul_right _ (by omega)
  · show r.val * F.nOpen + ((List.range F.sh.sched.levels.size).map F.q).sum ≤ 2 * F.nOpen
    have := Nat.mul_le_mul_right F.nOpen (show r.val + 1 ≤ 2 by omega)
    rw [Nat.add_mul, Nat.one_mul] at this
    exact this

/-- **`HAccRows` from the openings' registered columns** (`accIn_of_regBit`, `accOut_of_regOut`): at every accepted
outer proof, each opening's first 256 `acc` columns are registered bits of `V`'s row at its chain position on wires its
unit reads, and the first 256 columns of its `acc_out` half registered bits of the next row. -/
theorem hAccRows_of (hU : F.HOpenUnits) (V : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → ℕ → Array F128)
    (hin : ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ r : Fin 2,
      ∀ ℓ < F.sh.sched.levels.size, ∀ j < F.q ℓ,
        (∀ b < 256, Decode.RegBit (F.zs R ω t R₂ (F.sessO ℓ)) (F.unO ℓ r j) 0
          (accCol (levelLanes F.sh ℓ) (F.depth ℓ) + b)
          (bitL (MerkleScheme.rowBytes
            (V R ω t R₂ (r.val * F.nOpen + ((List.range ℓ).map F.q).sum + j))).data.toList b)) ∧
        ∀ hj : 0 < (F.Vs (F.sessO ℓ)).c.upv, ∀ b < 256,
          ∀ hc : accCol (levelLanes F.sh ℓ) (F.depth ℓ) + b < (Rows.ofNet (F.Vs (F.sessO ℓ)).hU).w,
            (keyInst (F.Vs (F.sessO ℓ)).hU (F.Vs (F.sessO ℓ)).nv
                (Layout.outsOf (F.Vs (F.sessO ℓ)).c (F.Vs (F.sessO ℓ)).hU (F.Vs (F.sessO ℓ)).nv) (F.unO ℓ r j)).wire
              ((Rows.ofNet (F.Vs (F.sessO ℓ)).hU).stackCol (F.Vs (F.sessO ℓ)).c.upv hj
                ⟨accCol (levelLanes F.sh ℓ) (F.depth ℓ) + b, hc⟩) ∈
            ((Placed.keyProg (F.Vs (F.sessO ℓ)).c (F.Vs (F.sessO ℓ)).hU (F.Vs (F.sessO ℓ)).nv).partition
              (Layout.outsOf (F.Vs (F.sessO ℓ)).c (F.Vs (F.sessO ℓ)).hU (F.Vs (F.sessO ℓ)).nv)).io (F.unO ℓ r j))
    (hout : ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ r : Fin 2,
      ∀ ℓ < F.sh.sched.levels.size, ∀ j < F.q ℓ, ∀ hj : 0 < (F.Vs (F.sessO ℓ)).c.upv, ∀ b < 256,
        AccChain.RegOut (F.zs R ω t R₂ (F.sessO ℓ)) (F.unO ℓ r j) hj
          (outCol (levelLanes F.sh ℓ) (F.depth ℓ) + 1024 + b)
          (bitL (MerkleScheme.rowBytes
            (V R ω t R₂ (r.val * F.nOpen + ((List.range ℓ).map F.q).sum + j + 1))).data.toList b)) :
    F.HAccRows dg N k Rw V := by
  intro R ω t R₂ τ o h r ℓ hℓ j hj i hi
  obtain ⟨hup, -, H, hp⟩ := hU ℓ hℓ
  have hp' : Flock.RecOpen.params (F.Vs (F.sessO ℓ)).c.unit.name = some (levelLanes F.sh ℓ, F.depth ℓ) := by
    unfold depth
    rw [hp]
    rfl
  obtain ⟨hreg, hio⟩ := hin R ω t R₂ τ o h r ℓ hℓ j hj
  exact ⟨accIn_of_regBit (F.zs R ω t R₂ (F.sessO ℓ)) (h.1 (F.sessO ℓ)).2.2 hup hreg (hio hup) i hi,
    accOut_of_regOut (F.zs R ω t R₂ (F.sessO ℓ)) (h.1 (F.sessO ℓ)).2 hup hp' (hout R ω t R₂ τ o h r ℓ hℓ j hj hup) i hi⟩

/-- **`HEnds` from the algebra unit's registered bits** (`readsEnds_of_regBits`): at every accepted outer proof, rep
`r`'s algebra unit `uA r` of session `sA r` holds bit `b` of the rep's `acc0 i` and `acc1 i` at the columns
`col0 r i b`, `col1 r i b` as registered bits, and the registrant's bits there are those of `V`'s rows at the rep's ends
of the chain. -/
theorem hEnds_of (V : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → ℕ → Array F128) (sA : Fin 2 → Fin F.stmts.size)
    (uA : (r : Fin 2) → Fin (F.Vs (sA r)).nv) (col0 col1 : Fin 2 → ℕ → ℕ → ℕ)
    (hd0 : ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ r, ∀ i < 2, ∀ b < 128,
      Decode.RegBit (F.zs R ω t R₂ (sA r)) (uA r) 0 (col0 r i b)
        (bitOf (valOf F.sh (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).w
          (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).comb
          (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).cons (.acc0 i)) b))
    (hr0 : ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ r : Fin 2, ∀ i < 2, ∀ b < 128,
      Decode.RegBit (F.zs R ω t R₂ (sA r)) (uA r) 0 (col0 r i b)
        (bitL (MerkleScheme.rowBytes (V R ω t R₂ (r.val * F.nOpen))).data.toList (128 * i + b)))
    (hd1 : ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ r, ∀ i < 2, ∀ b < 128,
      Decode.RegBit (F.zs R ω t R₂ (sA r)) (uA r) 0 (col1 r i b)
        (bitOf (valOf F.sh (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).w
          (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).comb
          (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).cons (.acc1 i)) b))
    (hr1 : ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ r : Fin 2, ∀ i < 2, ∀ b < 128,
      Decode.RegBit (F.zs R ω t R₂ (sA r)) (uA r) 0 (col1 r i b)
        (bitL (MerkleScheme.rowBytes (V R ω t R₂ (r.val * F.nOpen + F.nOpen))).data.toList (128 * i + b))) :
    F.HEnds dg N k Rw V := fun R ω t R₂ τ o h r =>
  AccChain.readsEnds_of_regBits (F.zs R ω t R₂ (sA r)) (uA r) 0 (col0 r) (col1 r) (hd0 R ω t R₂ τ o h r)
    (hr0 R ω t R₂ τ o h r) (hd1 R ω t R₂ τ o h r) (hr1 R ω t R₂ τ o h r)

end VStarFamily

/-- **VBridge G at V*'s statement family, the parts' and openings' outputs read in the layers**:
`Flock.Assumptions.VBridge` at the family's sessions, statement, messages, commit strings and digests, from its
obligations (`vbridge_of_reps`' composition at `HResL`, `algebraHolds_of_decodeL`, and `HUL`, `unitSums_of_opensAtL`;
`CoefsAre` and `ConsAre` hold by definition) and `VFullR`, which V*'s full draw gives (`vFullR_of_read`). -/
theorem vbridge_of_family (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray)
    (N k Rw : ℕ) {GS : Flock.RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)} (hst : F.HStruct GS gs)
    (hsh : F.HShape dg N k Rw GS) (hres : F.HResL dg N k Rw gs) (hu : F.HUL dg N k Rw) (hrd : F.HReads dg N k Rw)
    (hsch : F.HSch) (hfiles : F.HFiles dg N k Rw) (hruns : F.HRuns dg N k Rw) (hcm : F.HCommits)
    (hdec : F.HDecodes) (hfull : VFullR (ℐ).r F.Sk) :
    Flock.Assumptions.VBridge (I := ℐ) F.zs dg N k Rw F.x F.wmsg F.Sk F.cmt F.msg :=
  ⟨fun R ω t R₂ hc τ o hout h₀ =>
    have h : Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o := ⟨h₀, hc, hout⟩
    (F.files R ω t (layers F.zs dg N k Rw R ω t R₂ τ)).accepts regs (F.x R ω) T CT sch (hsch R ω t _) _
      (hfiles R ω t R₂ τ o h) (hruns R ω t R₂ τ o h fun r => by
        obtain ⟨⟨v, info, kappa, hv, hvs⟩, hw, hc⟩ := hsh R ω t R₂ τ o h r
        exact ⟨FlockVBridge.OfVStar.algebraHolds_of_decodeL (F.zs R ω t R₂) (layers F.zs dg N k Rw R ω t R₂ τ)
            (fun s => (h.1 s).2) hst.1 (Flock.RecAlgebra.structureOf_bitsOk hst.1) hv hw hc hvs hst.2 F.sess
            (fun p => F.un p r) (fun p => cp p r) (hres R ω t R₂ τ o h r),
          FlockVBridge.AccChain.accHolds_of_units
            (unitSums_of_opensAtL (F.zs R ω t R₂) (layers F.zs dg N k Rw R ω t R₂ τ) F.sh _ _ F.sessO
              (fun ℓ j => F.unO ℓ r j) (fun ℓ j => cpO ℓ r j) (hu R ω t R₂ τ o h r))
            (F.coefsAre R ω t _ r) (F.consAre R ω t _ r) (hrd R ω t R₂ τ o h r)⟩),
    hcm, hdec, hfull⟩

/-- **VBridge G at V*'s statement family whose openings are read off the layers**: `vbridge_of_family` at `rows`
and `acc` the layers' (`rowsOf`, `accOf`), `HUL` from `hUL_of` and `HReads` from `hReads_of`. -/
theorem vbridge_of_openings (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray)
    (N k Rw : ℕ) {GS : Flock.RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)}
    (hrows : F.rows = F.rowsOf) (hacc : F.acc = F.accOf) (hst : F.HStruct GS gs)
    (hsh : F.HShape dg N k Rw GS) (hres : F.HResL dg N k Rw gs) (hU : F.HOpenUnits) (hone : F.HOneOpen dg N k Rw)
    (hcoef : F.HCoefCols dg N k Rw) (hn : 0 < F.nOpen)
    (V : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → ℕ → Array F128) (hV : F.HAccRows dg N k Rw V)
    (hE : F.HEnds dg N k Rw V) (hsch : F.HSch) (hfiles : F.HFiles dg N k Rw) (hruns : F.HRuns dg N k Rw)
    (hcm : F.HCommits) (hdec : F.HDecodes) (hfull : VFullR (ℐ).r F.Sk) :
    Flock.Assumptions.VBridge (I := ℐ) F.zs dg N k Rw F.x F.wmsg F.Sk F.cmt F.msg :=
  vbridge_of_family F dg N k Rw hst hsh hres (F.hUL_of dg N k Rw hrows hacc hU hone hcoef V hV)
    (F.hReads_of dg N k Rw hacc hn V hV hE) hsch hfiles hruns hcm hdec hfull

end family

end FlockVBridge.Family
