import Proofs.Flock.VBridge.Family.Registered

/-!
# VBridge G at V*'s family: `rec-acc`'s rows at the openings and at the reps' ends

`HAccRows` and `HEnds` read `rec-acc`'s rows `V` (rec_outer.py:29-35: the prover registers the chain's `N + 1` rows as
`rec-acc`; opening `p` reads row `p` as `acc` and row `p + 1` as `acc_out`, rec_outer.py:236-238; V*'s algebra reads a
rep's first and last rows as `acc0`, `acc1`, `rec_residuals.py`'s row ports). Each read of a row is a registered bit of
the registrant's row at `rec-acc`'s root, so each comes from three facts:

* `ColKeyed`: the reading column's Δ key is bit `k` of a row port the session's reads file names, at `rec-acc`'s root
  and the row's position (the circuit's keys and the reads file: `EndKeys`, `OpenKeys`);
* `RowsRegistered`: the rows `own` names are registered reads;
* `AccRowsAre`: the registrant's rows at `rec-acc`'s root are `V`'s.

`regBit_of_colKeyed` composes them. `hEnds_of_regs` gives `HEnds` from them and `hd0`, `hd1`; `hEnds_of_part` gives
`hd0`, `hd1` from `HResL`'s row ports at the part holding `acc0`, `acc1` (`EndsInPart`, `regBit_of_hResL`), so `HEnds`
from `HResL` and the three facts. `hAccRows_of_keys` gives `HAccRows`' `acc` half from them and takes its `acc_out` half
as `AccOutL`, the openings' outputs in the layer.
-/

namespace FlockVBridge.Family

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Recursion FlockSoundness.Refine FlockSoundness.Model FlockSoundness.Audit
open Flock FlockLevel3 FlockVBridge.Compiled FlockVBridge.InnerAccepts FlockVBridge.Stages
open Flock.RecResiduals (portSizes groups)
open Flock.RecAlgebra (structureOf verifierRows el valOf wi)
open FlockVBridge.OfVStar (ReadCol rowCol partRep)
open FlockVBridge.Decode (RegBit)
open FlockVBridge.PartS (wIdx)
open FlockVBridge.AccChain (levelLanes)
open FlockSoundness.Discharge.Hm (bitL)

/-! ## A column keyed by a registered row -/

section Keyed

variable {mPts nv J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit} {hscope : Layout.scopeOk c = true}
  {Lv : Law nv} (zo : ZkOuter mPts nv J Reg' c hU hscope Lv)

/-- **Copy 0's input column `col` of unit `u` is keyed by bit `k` of a row `own` names, at the registered root `pa` and
position `pos`**: its Δ key (`Aliased.srcKey`) is bit `k` of row port `q`, the session's reads file names `q`, and
unit `u`'s row `q` is read at `pa`, position `pos`. -/
def ColKeyed (u : Fin nv) (col k : ℕ) (pa : Flock.Registered.Port) (pos : ℕ) : Prop :=
  ∃ (_ : col < (Rows.ofNet hU).nIn) (q : Fin c.ports.size) (_ : k < Layout.portRowW c q),
    Aliased.srcKey c (0 * (Rows.ofNet hU).nIn + col) = .msg q.val (k / 8) (k % 8) ∧
      (∃ pt ∈ zo.own, pt.name = c.ports[q].name) ∧ zo.rd.port (.inl (u, q)) = pa ∧ zo.rd.readAt (.inl (u, q)) = pos

/-- **A keyed column is a registered bit of the registrant's row**: when the rows `own` names are registered reads at
`u` and the registrant's rows at `pa` are `row`'s below bit `K`, the column is a registered bit, bit `k` of `row pos`. -/
theorem regBit_of_colKeyed {u : Fin nv} {col k : ℕ} {pa : Flock.Registered.Port} {pos : ℕ} {K : ℕ} (hkK : k < K)
    (hread : ∀ q : Fin c.ports.size, (∃ pt ∈ zo.own, pt.name = c.ports[q].name) → zo.rd.isRead (.inl (u, q)))
    (row : ℕ → ℕ → Bool)
    (hval : ∀ q : Fin c.ports.size, zo.rd.isRead (.inl (u, q)) → zo.rd.port (.inl (u, q)) = pa →
      ∀ b < K, ∀ (hb : b < Layout.portRowW c q),
        (Decode.regVal zo (.inl (u, q))).2
            ⟨b, lt_of_lt_of_le hb ((Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _))⟩ =
          row (zo.rd.readAt (.inl (u, q))) b)
    (h : ColKeyed zo u col k pa pos) : RegBit zo u 0 col (row pos k) := by
  obtain ⟨hc, q, hk, hkey, hown, hport, hat⟩ := h
  have hr := hread q hown
  refine ⟨hc, q, k, hk, hkey, hr, ?_⟩
  rw [hval q hr hport k hkK hk, hat]

end Keyed

theorem valOf_acc0 {sh : Flock.Shape} {w comb : Array F128} {cons : ℕ → ℕ → F128} {i j : ℕ}
    (h : wi sh (.acc0 i) = some j) : valOf sh w comb cons (.acc0 i) = el w j := by
  simp only [valOf, h, Flock.RecAlgebra.getElem!_el]

theorem valOf_acc1 {sh : Flock.Shape} {w comb : Array F128} {cons : ℕ → ℕ → F128} {i j : ℕ}
    (h : wi sh (.acc1 i) = some j) : valOf sh w comb cons (.acc1 i) = el w j := by
  simp only [valOf, h, Flock.RecAlgebra.getElem!_el]

section family

variable {nT nR : ℕ} {T : Fin nT → Type} {CT : Fin nR → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
variable {cls : HClass} {regs : List (Region cls.kLog (cls.m - cls.kLog))} {sch : Flock.Schedule}
variable {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => flockInnerCL T CT execArith H512 enc512 InnerAccepts.leafPlain cls regs (schedOf sch)

namespace VStarFamily

variable (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ)

/-- **The registrant's rows at `rec-acc`'s root are `V`'s.** At every accepted outer proof, in every session, the
registrant's row at a registered read at the root `pa`, on its first 256 bits (the two elements the openings and the
algebra read), is the bytes of `V`'s row at the read's position
(`rec_outer.Chain`: the rows the prover registers as `rec-acc`, `ACC_VALUE`, rec_outer.py:166-181). It holds at
`V := accReg pa` from `RowsShared` (`accRowsAre_of_shared`). -/
def AccRowsAre (V : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → ℕ → Array F128)
    (pa : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Flock.Registered.Port) : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ j (u : Fin (F.Vs j).nv)
    (q : Fin (F.Vs j).c.ports.size), (F.zs R ω t R₂ j).rd.isRead (.inl (u, q)) →
      (F.zs R ω t R₂ j).rd.port (.inl (u, q)) = pa R ω t R₂ → ∀ b < 256, ∀ (hb : b < Layout.portRowW (F.Vs j).c q),
        (Decode.regVal (F.zs R ω t R₂ j) (.inl (u, q))).2
            ⟨b, lt_of_lt_of_le hb ((Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _))⟩ =
          bitL (MerkleScheme.rowBytes (V R ω t R₂ ((F.zs R ω t R₂ j).rd.readAt (.inl (u, q))))).data.toList b

/-- A registered read at the root `pa` and position `p`, in some session, whose row has bit `b`. -/
def AccRead (pa : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Flock.Registered.Port) (R : Reg) (ω : L.Ω)
    (t : List (Cm × (ℐ).Coin)) (R₂ : Reg₂) (p b : ℕ)
    (s : Σ j : Fin F.stmts.size, Fin (F.Vs j).nv × Fin (F.Vs j).c.ports.size) : Prop :=
  (F.zs R ω t R₂ s.1).rd.isRead (.inl (s.2.1, s.2.2)) ∧ (F.zs R ω t R₂ s.1).rd.port (.inl (s.2.1, s.2.2)) = pa R ω t R₂ ∧
    (F.zs R ω t R₂ s.1).rd.readAt (.inl (s.2.1, s.2.2)) = p ∧ b < Layout.portRowW (F.Vs s.1).c s.2.2

open Classical in
/-- Bit `b` of the registrant's row at the root `pa` and position `p`: its bit at some registered read there
(`AccRead`), `false` if there is none. -/
noncomputable def accBit (pa : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Flock.Registered.Port) (R : Reg) (ω : L.Ω)
    (t : List (Cm × (ℐ).Coin)) (R₂ : Reg₂) (p b : ℕ) : Bool :=
  if h : ∃ s, F.AccRead pa R ω t R₂ p b s then
    (Decode.regVal (F.zs R ω t R₂ (Classical.choose h).1)
      (.inl ((Classical.choose h).2.1, (Classical.choose h).2.2))).2
        ⟨b, lt_of_lt_of_le (Classical.choose_spec h).2.2.2 ((Finset.le_sup (Finset.mem_univ _)).trans (Nat.le_succ _))⟩
  else false

/-- **`rec-acc`'s rows as registered**: row `p`'s first two elements are the registrant's at the root `pa` and
position `p` (`accBit`). -/
noncomputable def accReg (pa : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Flock.Registered.Port) (R : Reg) (ω : L.Ω)
    (t : List (Cm × (ℐ).Coin)) (R₂ : Reg₂) (p : ℕ) : Array F128 :=
  #[ofBits fun e => F.accBit pa R ω t R₂ p e, ofBits fun e => F.accBit pa R ω t R₂ p (128 + e)]

theorem bitL_accReg (pa : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Flock.Registered.Port) (R : Reg) (ω : L.Ω)
    (t : List (Cm × (ℐ).Coin)) (R₂ : Reg₂) (p : ℕ) {b : ℕ} (hb : b < 256) :
    bitL (MerkleScheme.rowBytes (F.accReg pa R ω t R₂ p)).data.toList b = F.accBit pa R ω t R₂ p b := by
  have hl : (F.accReg pa R ω t R₂ p).toList =
      [ofBits fun e => F.accBit pa R ω t R₂ p e, ofBits fun e => F.accBit pa R ω t R₂ p (128 + e)] := rfl
  have hm : b % 128 < 128 := Nat.mod_lt _ (by norm_num)
  rw [bitL_rowBytes, hl]
  simp only [FlockVBridge.wbit]
  rcases (show b / 128 = 0 ∨ b / 128 = 1 by omega) with h0 | h0
  · simp only [h0, List.getD_cons_zero]
    rw [bitOf_ofBits _ hm, show b % 128 = b by omega]
  · simp only [h0, List.getD_cons_succ, List.getD_cons_zero]
    rw [bitOf_ofBits _ hm, show 128 + b % 128 = b by omega]

/-- **`AccRowsAre` at `rec-acc`'s rows as registered** (`accReg`), from one registrant row per registered root and
position (`RowsShared`): every registered read at the root and position carries the bits of the one `accBit` reads. -/
theorem accRowsAre_of_shared (hshared : F.RowsShared dg N k Rw)
    (pa : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Flock.Registered.Port) :
    F.AccRowsAre dg N k Rw (F.accReg pa) pa := by
  intro R ω t R₂ τ o h j u q hread hport b hb256 hb
  rw [F.bitL_accReg pa R ω t R₂ _ hb256]
  have hex : ∃ s, F.AccRead pa R ω t R₂ ((F.zs R ω t R₂ j).rd.readAt (.inl (u, q))) b s :=
    ⟨⟨j, (u, q)⟩, hread, hport, rfl, hb⟩
  unfold accBit
  rw [dite_eq_left hex]
  have hs := Classical.choose_spec hex
  exact (hshared R ω t R₂ τ o h _ j _ u _ q hs.1 hread (hs.2.1.trans hport.symm) hs.2.2.1 b hs.2.2.2 hb).symm

/-- Obligation: discharged by the parse of the part's circuit (its keys) and the session's reads file.
**The algebra unit's end columns are keyed by `rec-acc`'s rows at the rep's ends.** At every accepted outer proof,
bit `b` of `acc0 i` (`col0 r i b`) and of `acc1 i` (`col1 r i b`) at rep `r`'s algebra unit is bit `128 i + b` of a row
`own` names, read at `rec-acc`'s root `pa` at the rep's first row `r · nOpen` and its last `r · nOpen + nOpen`
(`rec_residuals.py`: `acc0`, `acc1`, rows of the prover's `rec-acc`). -/
def EndKeys (pa : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Flock.Registered.Port) (sA : Fin 2 → Fin F.stmts.size)
    (uA : (r : Fin 2) → Fin (F.Vs (sA r)).nv) (col0 col1 : Fin 2 → ℕ → ℕ → ℕ) : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ r : Fin 2, ∀ i < 2, ∀ b < 128,
    ColKeyed (F.zs R ω t R₂ (sA r)) (uA r) (col0 r i b) (128 * i + b) (pa R ω t R₂) (r.val * F.nOpen) ∧
      ColKeyed (F.zs R ω t R₂ (sA r)) (uA r) (col1 r i b) (128 * i + b) (pa R ω t R₂) (r.val * F.nOpen + F.nOpen)

/-- **`HEnds` from the registrant's rows** (`hEnds_of` at `regBit_of_colKeyed`): `hr0`, `hr1` from the end columns'
keys (`EndKeys`), the registered reads (`RowsRegistered`) and `V` the registrant's rows (`AccRowsAre`). -/
theorem hEnds_of_regs (V : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → ℕ → Array F128)
    (pa : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Flock.Registered.Port) (sA : Fin 2 → Fin F.stmts.size)
    (uA : (r : Fin 2) → Fin (F.Vs (sA r)).nv) (col0 col1 : Fin 2 → ℕ → ℕ → ℕ)
    (hd0 : ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ r, ∀ i < 2, ∀ b < 128,
      Decode.RegBit (F.zs R ω t R₂ (sA r)) (uA r) 0 (col0 r i b)
        (bitOf (valOf F.sh (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).w
          (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).comb
          (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).cons (.acc0 i)) b))
    (hd1 : ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ r, ∀ i < 2, ∀ b < 128,
      Decode.RegBit (F.zs R ω t R₂ (sA r)) (uA r) 0 (col1 r i b)
        (bitOf (valOf F.sh (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).w
          (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).comb
          (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).cons (.acc1 i)) b))
    (hreg : F.RowsRegistered dg N k Rw) (hacc : F.AccRowsAre dg N k Rw V pa)
    (hkeys : F.EndKeys dg N k Rw pa sA uA col0 col1) : F.HEnds dg N k Rw V :=
  F.hEnds_of dg N k Rw V sA uA col0 col1 hd0
    (fun R ω t R₂ τ o h r i hi b hb => regBit_of_colKeyed _ (K := 256) (by omega) (hreg R ω t R₂ τ o h (sA r) (uA r))
      (fun p b => bitL (MerkleScheme.rowBytes (V R ω t R₂ p)).data.toList b) (hacc R ω t R₂ τ o h (sA r) (uA r))
      (hkeys R ω t R₂ τ o h r i hi b hb).1)
    hd1
    (fun R ω t R₂ τ o h r i hi b hb => regBit_of_colKeyed _ (K := 256) (by omega) (hreg R ω t R₂ τ o h (sA r) (uA r))
      (fun p b => bitL (MerkleScheme.rowBytes (V R ω t R₂ p)).data.toList b) (hacc R ω t R₂ τ o h (sA r) (uA r))
      (hkeys R ω t R₂ τ o h r i hi b hb).2)

/-- **A part's row-port column a computed row reads holds the rep's element** (`HResL`'s row ports): at an accepted outer
proof, rep `r`'s instance of part `p` holds, at the column of bit `b` of its `a`-th element, a registered bit, bit `b`
of the rep's `w` at that element's index. -/
theorem regBit_of_hResL {GS : Flock.RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)}
    (hst : F.HStruct GS gs) (hres : F.HResL dg N k Rw gs) {R : Reg} {ω : L.Ω} {t : List (Cm × (ℐ).Coin)} {R₂ : Reg₂}
    {τ o} (h : Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o) (r : Fin 2) {p : ℕ} (hp : p < (partsOf F.sh).size)
    {a b : ℕ} (ha : a < (wIOf F.sh p).size) (hb : b < 128)
    (hrd : ReadCol (Rows.ofNet (F.Vs (F.sess p)).hU) (rowColOf F.sh p (128 * a + b))) :
    RegBit (F.zs R ω t R₂ (F.sess p)) (F.un p r) 0 (rowColOf F.sh p (128 * a + b))
      (bitOf (el (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).w (wIOf F.sh p)[a]) b) := by
  have hP : partsOf F.sh = gs := partsOf_eq hst.1 hst.2
  have hpg : p < gs.size := by rw [← hP]; exact hp
  obtain ⟨hpg', -, -, -, -, -, -, -, -, -, -, S, ns, nv, hpart, hrow, -, -⟩ := hres R ω t R₂ τ o h r p hpg
  obtain ⟨-, hns, hnv⟩ := FlockVBridge.OfVStar.part_eq hst.1 hst.2 hpg hpart
  have hgp : (partsOf F.sh)[p]! = gs[p] := by rw [hP, getElem!_pos gs p hpg]
  have hnsp : nsOf F.sh p = ns := by rw [nsOf, hgp, hns]
  have hnvp : nvOf F.sh p = nv := by rw [nvOf, hgp, hnv]
  have hwI : wIOf F.sh p = wIdx (portSizes F.sh) gs[p].2.1 := by rw [wIOf, hgp]
  have hsz : (wIOf F.sh p).size = ns.sum := by rw [hwI, FlockVBridge.OfVStar.wIdx_size, hns]
  have hq : 128 * a + b < 128 * ns.sum := by rw [← hsz]; omega
  rw [rowColOf, hnsp, hnvp] at hrd ⊢
  have hr := hrow (128 * a + b) hq hrd
  rw [← hwI] at hr
  have hdiv : (128 * a + b) / 128 = a := by omega
  have hmod : (128 * a + b) % 128 = b := by omega
  unfold FlockVBridge.wbit at hr
  rw [hdiv, hmod, FlockVBridge.Decode.sub_lt _ _ _ ha] at hr
  exact hr

/-- Obligation: discharged by V*'s shape (`wi`) and the parse of the part's circuit (`ReadCol`).
**The rep's ends are elements part `pA` holds, at columns a computed row reads**: `acc0 i` is the element at
position `a0 i` of the part's row ports and `acc1 i` at `a1 i` (`wi`; `rec_residuals.py`'s row ports `acc0`, `acc1`),
and some computed row of the part's unit reads each of their bits' columns. -/
def EndsInPart (pA : ℕ) (a0 a1 : ℕ → ℕ) : Prop :=
  pA < (partsOf F.sh).size ∧ ∀ i < 2,
    (∃ h : a0 i < (wIOf F.sh pA).size, wi F.sh (.acc0 i) = some (wIOf F.sh pA)[a0 i]) ∧
      (∃ h : a1 i < (wIOf F.sh pA).size, wi F.sh (.acc1 i) = some (wIOf F.sh pA)[a1 i]) ∧
      ∀ b < 128, ReadCol (Rows.ofNet (F.Vs (F.sess pA)).hU) (rowColOf F.sh pA (128 * a0 i + b)) ∧
        ReadCol (Rows.ofNet (F.Vs (F.sess pA)).hU) (rowColOf F.sh pA (128 * a1 i + b))

/-- **`HEnds` from `HResL` and the registrant's rows**: the algebra unit is rep `r`'s instance of the part holding the
rep's ends (`EndsInPart`), whose row ports give `hd0`, `hd1` (`regBit_of_hResL`), and `hEnds_of_regs` the rest. -/
theorem hEnds_of_part {GS : Flock.RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)}
    (hst : F.HStruct GS gs) (hres : F.HResL dg N k Rw gs) (V : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → ℕ → Array F128)
    (pa : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Flock.Registered.Port) {pA : ℕ} {a0 a1 : ℕ → ℕ}
    (hin : F.EndsInPart pA a0 a1) (hreg : F.RowsRegistered dg N k Rw) (hacc : F.AccRowsAre dg N k Rw V pa)
    (hkeys : F.EndKeys dg N k Rw pa (fun _ => F.sess pA) (F.un pA) (fun _ i b => rowColOf F.sh pA (128 * a0 i + b))
      (fun _ i b => rowColOf F.sh pA (128 * a1 i + b))) :
    F.HEnds dg N k Rw V :=
  F.hEnds_of_regs dg N k Rw V pa (fun _ => F.sess pA) (F.un pA) (fun _ i b => rowColOf F.sh pA (128 * a0 i + b))
    (fun _ i b => rowColOf F.sh pA (128 * a1 i + b))
    (fun R ω t R₂ τ o h r i hi b hb => by
      obtain ⟨⟨ha, hwi⟩, -, hrd⟩ := hin.2 i hi
      rw [valOf_acc0 hwi]
      exact F.regBit_of_hResL dg N k Rw hst hres h r hin.1 ha hb (hrd b hb).1)
    (fun R ω t R₂ τ o h r i hi b hb => by
      obtain ⟨-, ⟨ha, hwi⟩, hrd⟩ := hin.2 i hi
      rw [valOf_acc1 hwi]
      exact F.regBit_of_hResL dg N k Rw hst hres h r hin.1 ha hb (hrd b hb).2)
    hreg hacc hkeys

/-- Obligation: discharged by the parse of the open circuit (its keys, `Flock.RecOpen`) and the level's reads file.
**Each opening's `acc` columns are keyed by `rec-acc`'s row at its chain position.** At every accepted outer proof,
bit `b` of opening `j` of level `ℓ` of rep `r`'s `acc` (`accCol + b`) is bit `b` of a row `own` names, read at
`rec-acc`'s root `pa` at the opening's chain position (rec_outer.py:236: `acc` reads row `p`), and a computed row of
the opening's unit reads the column. -/
def OpenKeys (pa : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Flock.Registered.Port) : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ r : Fin 2, ∀ ℓ < F.sh.sched.levels.size,
    ∀ j < F.q ℓ, ∀ b < 256,
      ColKeyed (F.zs R ω t R₂ (F.sessO ℓ)) (F.unO ℓ r j) (accCol (levelLanes F.sh ℓ) (F.depth ℓ) + b) b (pa R ω t R₂)
          (r.val * F.nOpen + ((List.range ℓ).map F.q).sum + j) ∧
        ReadCol (Rows.ofNet (F.Vs (F.sessO ℓ)).hU) (accCol (levelLanes F.sh ℓ) (F.depth ℓ) + b)

/-- Obligation: to be discharged by acceptance, now that the layout keys a returned output gate to its output row
(`HiddenExec.rowOfH`), which makes `RegOut` hold at `acc_out`'s columns where they are registered reads.
**The openings' `acc_out` in the layers are `V`'s next rows** (`HAccRows`' second half): at every accepted outer
proof, the first two elements of opening `j` of level `ℓ` of rep `r`'s `acc_out` output, in the layer, are those of
`V`'s row after the opening's chain position. V*'s level sessions register `acc_out` as a read of `rec-acc`'s row
`p + 1` (rec_outer.py:238, 242); acceptance reaches the layer's outputs only through those reads (`RegAgree`). -/
def AccOutL (V : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → ℕ → Array F128) : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ r : Fin 2, ∀ ℓ < F.sh.sched.levels.size,
    ∀ j < F.q ℓ, ∀ i < 2,
      (F.accOut (layers F.zs dg N k Rw R ω t R₂ τ) (r, ℓ, j))[i]! =
        (V R ω t R₂ (r.val * F.nOpen + ((List.range ℓ).map F.q).sum + j + 1))[i]!

/-- **`HAccRows` from the registrant's rows and the openings' outputs**: each opening's `acc` from its keyed columns
(`OpenKeys`, `RowsRegistered`, `AccRowsAre`, `regBit_of_colKeyed`; `ioColAt_of_readCol` puts them among the unit's
wires, `accIn_of_regBit`), its `acc_out` from `AccOutL`. -/
theorem hAccRows_of_keys (hU : F.HOpenUnits) (V : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → ℕ → Array F128)
    (pa : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Flock.Registered.Port) (hreg : F.RowsRegistered dg N k Rw)
    (hacc : F.AccRowsAre dg N k Rw V pa) (hk : F.OpenKeys dg N k Rw pa) (hout : F.AccOutL dg N k Rw V) :
    F.HAccRows dg N k Rw V := by
  intro R ω t R₂ τ o h r ℓ hℓ j hj i hi
  obtain ⟨hup, -⟩ := hU ℓ hℓ
  refine ⟨accIn_of_regBit (F.zs R ω t R₂ (F.sessO ℓ)) (h.1 (F.sessO ℓ)).2.2.1 (u := F.unO ℓ r j) hup
    (base := accCol (levelLanes F.sh ℓ) (F.depth ℓ))
    (A := V R ω t R₂ (r.val * F.nOpen + ((List.range ℓ).map F.q).sum + j)) (fun b hb => ?_) (fun b hb hc => ?_) i hi,
    hout R ω t R₂ τ o h r ℓ hℓ j hj i hi⟩
  · exact regBit_of_colKeyed _ (K := 256) hb (hreg R ω t R₂ τ o h _ _)
      (fun p b => bitL (MerkleScheme.rowBytes (V R ω t R₂ p)).data.toList b) (hacc R ω t R₂ τ o h _ _)
      (hk R ω t R₂ τ o h r ℓ hℓ j hj b hb).1
  · obtain ⟨⟨hc', -⟩, hrd⟩ := hk R ω t R₂ τ o h r ℓ hℓ j hj b hb
    exact ioColAt_of_readCol (F.unO ℓ r j) hc' hrd hup hc

end VStarFamily

end family

end FlockVBridge.Family
