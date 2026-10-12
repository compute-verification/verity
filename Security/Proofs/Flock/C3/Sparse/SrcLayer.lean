import Proofs.Flock.C3.Sparse.SrcTie
import Proofs.Flock.VBridge.ResLayer

/-!
# The sparse tie at V*'s layer

`SparseTie` states a RecSparse unit's rows as `LaidRows` of `RecSparse.unitLaid` at some columns `v`. Here `v` is read
off V*'s layer, as `ResLayer` does for `RecResiduals`: a copy's columns of the unit `check_sparse` accepted
(`RecSparse.unitNet sh = .ok c.unit`), with each message column no computed row reads (`ReadCol`) set to its bit of `w`.

* `inWords_sparse`, `rowCol_lt_nInS`: the laid-out unit's input rows are `sparseInBits`', and the message ports' columns
  are input columns of the verifier's unit. `nw_le_msum`, `nv_le`: `w` and the verifier row have no more elements than
  the message ports and `v`'s port hold.
* `laid_sparseR`: if the unit's rows hold at `z` with 1 at the constant, and `z` holds `w` at the message columns some
  computed row reads, `vv` on `v`'s and the slots' types on `tau`'s, then `z` with each unread message column set to its
  bit of `w` gives `LaidRows` and `SparseIn`, and agrees with `z` from `sparseBase` on (`sat_of_read`).
* `decode_sparseR`: the same at copy `j` of unit `u` in V*'s layer, the inputs registered bits (`RegBit`) and the
  outputs read in the layer (`copyVal`).
* `SparseTieL`, `sparseTie_of_layer`: `SparseTie` with the unit a copy in a good layer, its inputs registered and its
  residuals' output columns 0 there, in place of `LaidRows`; it gives `SparseTie`.
-/

namespace FlockVBridge.Sparse

open Flock.HmNets hiding and
open Flock (F128)
open FlockSoundness FlockSoundness.Audit FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open FlockSoundness.Discharge.Hm
open Flock.RecSparse (messagePorts offset)
open FlockVBridge.OfVStar (ReadCol sat_of_read rowCol rowCol_inj rowCol_lt foldl_replicate)

theorem inWords_sparse {sh : Flock.RecSparse.Shape} {name : String} {net : Flock.Net}
    (hn : Flock.HmNets.netOf name (Flock.RecSparse.unitLaid sh) = .ok net) :
    net.inWords * Flock.WORD = inRowsOf (sparseInBits sh) := by
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, hw, -⟩ := Flock.Net.ofRows_ok hn
  rw [hw]
  simp only [Flock.RecSparse.unitLaid, FlockSoundness.Discharge.Hm.layout_inGroups]
  simp [inRowsOf, sparseInBits]

theorem offset_last_le (nm : String) (n : ℕ) : ∀ l : List (String × ℕ),
    offset (l ++ [(nm, n)]) nm + n ≤ ((l ++ [(nm, n)]).map (·.2)).sum
  | [] => by simp [offset_cons_self]
  | (a, k) :: l => by
    have ih := offset_last_le nm n l
    simp only [List.cons_append, List.map_cons, List.sum_cons]
    by_cases ha : a = nm
    · rw [ha, offset_cons_self]
      omega
    · rw [offset_cons_ne _ _ ha]
      omega

/-- `w`'s elements are at most the message ports'. -/
theorem nw_le_msum (sh : Flock.RecSparse.Shape) : nw (shOf sh) ≤ ((messagePorts sh).map (·.2)).sum := by
  obtain ⟨l, hl⟩ : ∃ l, messagePorts sh = l ++ [("claim", 1)] := ⟨_, rfl⟩
  have h := offset_last_le "claim" 1 l
  rw [← hl, (mp_offsets sh).2.2.2.2.2.2] at h
  exact h

/-- The verifier row's elements are at most `v`'s port's. -/
theorem nv_le (sh : Flock.RecSparse.Shape) : FlockVBridge.Sparse.nv (shOf sh) ≤ Flock.RecSparse.nv sh := by
  rw [nv_eq sh, sum_degs sh]
  simp only [FlockVBridge.Sparse.nv, vEqp, vUr, vAy, vAx, vAlpha, pLog_shOf, shOf_ell, shOf_sLog, shOf_tLog,
    Flock.RecSparse.Shape.PP, Flock.RecSparse.Shape.S, Flock.RecSparse.Shape.T]
  by_cases ht : sh.tLog = 0
  · have hst : sh.stacked = false := by simp [Flock.RecSparse.Shape.stacked, ht]
    rw [ite_eq_left ht]
    simp only [hst, Bool.false_eq_true, ↓reduceIte]
    omega
  · have hst : sh.stacked = true := by
      simp only [Flock.RecSparse.Shape.stacked, decide_eq_true_eq]
      omega
    rw [ite_eq_right ht]
    simp only [hst, ↓reduceIte]
    omega

/-- **The message ports' columns are input columns** of the verifier's `rec-sparse` unit. -/
theorem rowCol_lt_nInS {c : Flock.Circuit} (hU : Built c.unit) {sh : Flock.RecSparse.Shape} {name : String}
    (hn : Flock.HmNets.netOf name (Flock.RecSparse.unitLaid sh) = .ok c.unit) {q : ℕ}
    (hq : q < 128 * ((messagePorts sh).map (·.2)).sum) : rowCol (messagePorts sh) 0 q < (Rows.ofNet hU).nIn := by
  have hlt := rowCol_lt (messagePorts sh) 0 q hq
  have h16 := inWords_16 sh
  have hvb : tauBase sh ≤ vBaseS sh := Nat.le_add_right _ _
  have htb : tauBase sh = 16 * ((messagePorts sh).map fun p => VStar.portWordsOf p.2).sum := rfl
  have hN : (sparseInBits sh).foldl (· + ·) 0 = 16 * Flock.RecSparse.inWords sh := by
    rw [sparseInBits, foldl_replicate]
    exact Nat.mul_comm _ _
  rw [Layout.ofNet_nIn_eq, inWords_sparse hn]
  refine lt_of_lt_of_le ?_ (Flock.HmNets.le_groupWords _)
  rw [hN]
  omega

/-- **The verifier's laid-out `rec-sparse` unit at its read columns** (`laid_partR`'s construction): when its rows hold
at `z` with 1 at the constant, and `z` holds `w` at the message columns some computed row reads, `vv` on `v`'s and with
`T > 1` the slots' types on `tau`'s, `z` with each unread message column set to its bit of `w` still satisfies the rows
(`sat_of_read`), so it gives `LaidRows` and `SparseIn`, and it agrees with `z` past the inputs. -/
theorem laid_sparseR {net : Flock.Net} (hB : Built net) (sh : Flock.RecSparse.Shape) {name : String}
    (hn : Flock.HmNets.netOf name (Flock.RecSparse.unitLaid sh) = .ok net) (hu : net.unitLog ≤ 32)
    (hlt : ∀ q < 128 * ((messagePorts sh).map (·.2)).sum, rowCol (messagePorts sh) 0 q < (Rows.ofNet hB).nIn)
    {z : Fin (Rows.ofNet hB).w → Bool} (hz : (Rows.ofNet hB).Sat z) (hone : z (Rows.ofNet hB).one = true)
    {w vv : ℕ → F128} {τ : ℕ → ℕ}
    (hw : ∀ q < 128 * ((messagePorts sh).map (·.2)).sum, ReadCol (Rows.ofNet hB) (rowCol (messagePorts sh) 0 q) →
      colVal z (rowCol (messagePorts sh) 0 q) = wbit w q)
    (hv : ∀ q < 128 * Flock.RecSparse.nv sh, colVal z (vBaseS sh + q) = wbit vv q)
    (htau : sh.stacked = true → ∀ s < sh.S, ∀ b < 16, colVal z (tauBase sh + (16 * s + b)) = (τ s).testBit b)
    (hw0 : ∀ e, ((messagePorts sh).map (·.2)).sum ≤ e → w e = 0) (hv0 : ∀ e, Flock.RecSparse.nv sh ≤ e → vv e = 0) :
    ∃ v, LaidRows (Flock.RecSparse.unitLaid sh) (inRowsOf (sparseInBits sh)) v ∧ SparseIn sh w vv τ v ∧
      ∀ t, v (sparseBase sh + t) = colVal z (sparseBase sh + t) := by
  classical
  obtain ⟨Q, hQ⟩ : ∃ Q : ℕ → Prop, ∀ x, Q x ↔ x < (Rows.ofNet hB).nIn ∧ ¬ ReadCol (Rows.ofNet hB) x ∧
      ∃ q, q < 128 * ((messagePorts sh).map (·.2)).sum ∧ rowCol (messagePorts sh) 0 q = x := ⟨_, fun _ => Iff.rfl⟩
  obtain ⟨z', hz'⟩ : ∃ z' : Fin (Rows.ofNet hB).w → Bool, ∀ x, z' x =
      if h : Q x.val then wbit w (Classical.choose ((hQ _).1 h).2.2) else z x := ⟨_, fun _ => rfl⟩
  have hzz : ∀ x : Fin (Rows.ofNet hB).w, ¬ Q x.val → z' x = z x := fun x h => by rw [hz', dite_eq_right h]
  have hzq : ∀ (x : Fin (Rows.ofNet hB).w) (h : Q x.val), z' x = wbit w (Classical.choose ((hQ _).1 h).2.2) :=
    fun x h => by rw [hz', dite_eq_left h]
  have hcol : ∀ x, ¬ Q x → colVal z' x = colVal z x := by
    intro x hx
    unfold colVal
    split_ifs with h
    · exact hzz _ hx
    · rfl
  have hs' : (Rows.ofNet hB).Sat z' := sat_of_read hz fun x hx => hzz x fun hq =>
    hx.elim ((hQ _).1 hq).2.1 fun h => absurd ((hQ _).1 hq).1 (Nat.not_lt.2 h)
  have hone' : z' (Rows.ofNet hB).one = true := by
    rw [hzz (Rows.ofNet hB).one fun hq => absurd ((hQ _).1 hq).1 (by simp [Rows.one])]
    exact hone
  have hiw := inWords_sparse hn
  have hbase : (Rows.ofNet hB).nIn ≤ sparseBase sh := by
    rw [Rows.ofNet_nIn, hiw]
    exact (min_le_left _ _).trans (Decode.inRows_le_outStart _ _ _)
  have hout : ∀ t, ¬ Q (sparseBase sh + t) := fun t hq =>
    absurd ((hQ _).1 hq).1 (Nat.not_lt.2 (hbase.trans (Nat.le_add_right _ _)))
  have hlo : ∀ x, tauBase sh ≤ x → ¬ Q x := fun x hx hq => by
    obtain ⟨-, -, q, hq, he⟩ := (hQ _).1 hq
    have := rowCol_lt (messagePorts sh) 0 q hq
    have htb : tauBase sh = 16 * ((messagePorts sh).map fun p => VStar.portWordsOf p.2).sum := rfl
    omega
  have hvb : tauBase sh ≤ vBaseS sh := Nat.le_add_right _ _
  have hL := laidRows_ofNet hB hn hu hs' hone'
  rw [hiw] at hL
  refine ⟨colVal z', hL, ⟨fun q hq => ?_, fun q hq => ?_, fun hst s hs b hb => ?_, hw0, hv0⟩,
    fun t => hcol _ (hout t)⟩
  · by_cases hr : ReadCol (Rows.ofNet hB) (rowCol (messagePorts sh) 0 q)
    · rw [hcol (rowCol (messagePorts sh) 0 q) fun hq' => ((hQ _).1 hq').2.1 hr]
      exact hw q hq hr
    · have hq' : Q (rowCol (messagePorts sh) 0 q) := (hQ _).2 ⟨hlt q hq, hr, q, hq, rfl⟩
      have hxw : rowCol (messagePorts sh) 0 q < (Rows.ofNet hB).w := by
        have := hlt q hq
        simp only [Rows.w]
        omega
      unfold colVal
      rw [dite_eq_left hxw, hzq ⟨_, hxw⟩ hq']
      obtain ⟨hq0, he⟩ := Classical.choose_spec ((hQ _).1 hq').2.2
      exact congrArg (wbit w) (rowCol_inj _ 0 _ _ hq0 hq he)
  · rw [hcol (vBaseS sh + q) (hlo _ (by omega))]
    exact hv q hq
  · rw [hcol (tauBase sh + (16 * s + b)) (hlo _ (Nat.le_add_right _ _))]
    exact htau hst s hs b hb

section Decode

open FlockVBridge.Decode

variable {mPts nU J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit} {hscope : Layout.scopeOk c = true}
  {Lv : Law nU} (zo : ZkOuter mPts nU J Reg' c hU hscope Lv)

/-- **Copy `j` of the verifier's `rec-sparse` unit in V*'s layer** (`laid_sparseR` at `sat_copy` of `fill`): the
message columns some computed row reads registered bits of `w`, `v`'s of `vv` and `tau`'s of the slots' types give
`LaidRows` and `SparseIn` at columns that agree with the copy's in the layer from `sparseBase` on. -/
theorem decode_sparseR {bits : Fin (Layout.keyN c hU nU) → Bool} (hA : zo.RegAgree bits) (u : Fin nU) {j : ℕ}
    (hj : j < c.upv) (hone : bits (Const.Prog.oneAt (Placed.keyProg c hU nU) u) = true)
    (sh : Flock.RecSparse.Shape) (hn : Flock.RecSparse.unitNet sh = .ok c.unit) (hu : c.unit.unitLog ≤ 32)
    {w vv : ℕ → F128} {τ : ℕ → ℕ}
    (hw : ∀ q < 128 * ((messagePorts sh).map (·.2)).sum, ReadCol (Rows.ofNet hU) (rowCol (messagePorts sh) 0 q) →
      RegBit zo u j (rowCol (messagePorts sh) 0 q) (wbit w q))
    (hv : ∀ q < 128 * Flock.RecSparse.nv sh, RegBit zo u j (vBaseS sh + q) (wbit vv q))
    (htau : sh.stacked = true → ∀ s < sh.S, ∀ b < 16, RegBit zo u j (tauBase sh + (16 * s + b)) ((τ s).testBit b))
    (hw0 : ∀ e, ((messagePorts sh).map (·.2)).sum ≤ e → w e = 0) (hv0 : ∀ e, Flock.RecSparse.nv sh ≤ e → vv e = 0) :
    ∃ v, LaidRows (Flock.RecSparse.unitLaid sh) (inRowsOf (sparseInBits sh)) v ∧ SparseIn sh w vv τ v ∧
      ∀ t, v (sparseBase sh + t) = copyVal (keyInst hU nU (Layout.outsOf c hU nU) u) bits hj (sparseBase sh + t) := by
  have hn' : Flock.HmNets.netOf (Flock.RecSparse.unitName sh) (Flock.RecSparse.unitLaid sh) = .ok c.unit := hn
  have h1 : fill zo bits ((keyInst hU nU (Layout.outsOf c hU nU) u).wire ((Rows.ofNet hU).stack c.upv).one) = true := by
    rw [fill_one zo hA _ hj, keyInst_one]
    exact hone
  obtain ⟨hsat, h1'⟩ := sat_copy (keyInst hU nU (Layout.outsOf c hU nU) u) (fill zo bits) hj
  obtain ⟨v, hL, hin, hout⟩ := laid_sparseR hU sh hn' hu (fun q hq => rowCol_lt_nInS hU hn' hq) hsat (h1'.trans h1)
    (fun q hq hrd => copyVal_regBit zo bits hj (hw q hq hrd)) (fun q hq => copyVal_regBit zo bits hj (hv q hq))
    (fun hst s hs b hb => copyVal_regBit zo bits hj (htau hst s hs b hb)) hw0 hv0
  have hbase : (Rows.ofNet hU).nIn ≤ sparseBase sh := by
    rw [Layout.ofNet_nIn_eq, inWords_sparse hn']
    exact inRows_le_outStart _ _ _
  exact ⟨v, hL, hin, fun t =>
    (hout t).trans (copyVal_fill zo hA (keyInst hU nU (Layout.outsOf c hU nU) u) hj (by omega))⟩

end Decode

end FlockVBridge.Sparse

namespace FlockC3.Sparse

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.Compose
open FlockVBridge (wbit copyVal keyInst)
open FlockVBridge.Decode (RegBit)
open FlockVBridge.OfVStar (ReadCol rowCol)
open FlockVBridge.Sparse (SCoins nw nc verifierRow tauBase vBaseS sparseBase check_sparse nw_le_msum nv_le
  decode_sparseR)

section Layer

variable {mPts nU J : ℕ} {Reg' : Type} {cc : Flock.Circuit} {hU : FlockSoundness.Built cc.unit}
  {hscope : FlockSoundness.Discharge.Layout.scopeOk cc = true} {Lv : FlockSoundness.Audit.Law nU}
  (zo : FlockSoundness.Discharge.Recursion.ZkOuter mPts nU J Reg' cc hU hscope Lv)

/-- **(B)'s tie at V*'s layer**: `SparseTie`'s message words `w` and coins `c`, with the unit copy `j` of unit `un` of
V*'s layer `zo` at `bits`: its circuit passes `RecSparse.check` at the session's shape, the copy's message columns some
computed row reads are registered bits of `w`, `v`'s of the verifier row at `c` and with `T > 1` `tau`'s of the slots'
types, and its residuals' output columns are 0 in the layer. -/
def SparseTieL (bits : Fin (FlockSoundness.Discharge.Layout.keyN cc hU nU) → Bool) (e : ExecHolo) (u : C3Run e)
    (i : Fin 2) : Prop :=
  ∃ (w : Array Flock.F128) (c : SCoins) (un : Fin nU) (j : ℕ) (hj : j < cc.upv),
    w.size = nw e.cls.shape ∧ nc e.cls.shape ≤ c.cs.size ∧
    FlockVBridge.Sparse.pubOf e.cls.shape c =
      holoPub (e.cls.pubStatement e.regs) e.cls.shape e.cls.units (u.rep i).1 (u.rep i).2 ∧
    (∀ s : Fin 64, fOf w[s.val]! = (u.rep i).2.h s) ∧ u.sp i = stepsOf e.cls.shape w c.cs ∧
    Flock.RecSparse.check cc = .ok () ∧ Flock.RecSparse.params cc.unit.name = some (unitShape e) ∧
    bits (FlockSoundness.Discharge.Const.Prog.oneAt (FlockSoundness.Discharge.Placed.keyProg cc hU nU) un) = true ∧
    cc.unit.unitLog ≤ 32 ∧
    (∀ q < 128 * ((Flock.RecSparse.messagePorts (unitShape e)).map (·.2)).sum,
      ReadCol (FlockSoundness.Rows.ofNet hU) (rowCol (Flock.RecSparse.messagePorts (unitShape e)) 0 q) →
        RegBit zo un j (rowCol (Flock.RecSparse.messagePorts (unitShape e)) 0 q) (wbit (w[·]!) q)) ∧
    (∀ q < 128 * Flock.RecSparse.nv (unitShape e),
      RegBit zo un j (vBaseS (unitShape e) + q) (wbit ((verifierRow e.cls.shape c)[·]!) q)) ∧
    ((unitShape e).stacked = true → ∀ s < (unitShape e).S, ∀ b < 16,
      RegBit zo un j (tauBase (unitShape e) + (16 * s + b)) (((tauArr e)[s]!).testBit b)) ∧
    ∀ t < 128 * (Flock.RecSparse.residualNames (unitShape e)).length,
      copyVal (keyInst hU nU (FlockSoundness.Discharge.Layout.outsOf cc hU nU) un) bits hj
        (sparseBase (unitShape e) + t) = false

/-- **The tie at a good layer is the tie** (`decode_sparseR`): `w` and the verifier row are 0 past their sizes, which
the message ports and `v`'s port cover (`nw_le_msum`, `nv_le`). -/
theorem sparseTie_of_layer {bits : Fin (FlockSoundness.Discharge.Layout.keyN cc hU nU) → Bool} (hG : zo.Good bits)
    {e : ExecHolo} {u : C3Run e} {i : Fin 2} (h : SparseTieL zo bits e u i) : SparseTie e u i := by
  obtain ⟨w, c, un, j, hj, hw, hc, hpub, hh, hs, hck, hpar, hone, hu, hrw, hrv, hrt, h0⟩ := h
  obtain ⟨-, -, -, hn⟩ := check_sparse hck hpar
  have hnw : nw e.cls.shape ≤ ((Flock.RecSparse.messagePorts (unitShape e)).map (·.2)).sum :=
    nw_le_msum (unitShape e)
  have hnv : FlockVBridge.Sparse.nv e.cls.shape ≤ Flock.RecSparse.nv (unitShape e) := nv_le (unitShape e)
  have hvs : (verifierRow e.cls.shape c).size = FlockVBridge.Sparse.nv e.cls.shape := Array.size_ofFn
  have hw0 : ∀ k, ((Flock.RecSparse.messagePorts (unitShape e)).map (·.2)).sum ≤ k → w[k]! = 0 :=
    fun k hk => (getElem!_neg w k (by omega)).trans rfl
  have hv0 : ∀ k, Flock.RecSparse.nv (unitShape e) ≤ k → (verifierRow e.cls.shape c)[k]! = 0 :=
    fun k hk => (getElem!_neg (verifierRow e.cls.shape c) k (by omega)).trans rfl
  obtain ⟨v, hL, hin, hout⟩ := decode_sparseR zo hG.2 un hj hone (unitShape e) hn hu
    (w := (w[·]!)) (vv := ((verifierRow e.cls.shape c)[·]!)) (τ := ((tauArr e)[·]!)) hrw hrv hrt hw0 hv0
  exact ⟨w, c, cc, v, hw, hc, hpub, hh, hs, hck, hpar, hL, hin, fun t ht => (hout t).trans (h0 t ht)⟩

end Layer

end

end FlockC3.Sparse
