module

public import Proofs.Flock.Claims.ZK.Split
public import Proofs.Flock.Claims.ZK.CutEq
import Flock.With.SimVerify
import Flock.With.SimWalk
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.Complete.Zk.InnerMerkle
import Proofs.Flock.Claims.Complete.ZkTables.Scope
import Proofs.Flock.Claims.Complete.ZkTables.Table
import Proofs.Flock.Claims.DeployK
import Proofs.Flock.Claims.Scope.Check
import Proofs.Flock.Claims.Scope.Sizes
import Proofs.Flock.Soundness.Discharge.Composed.Setup
import Proofs.Flock.Soundness.Discharge.ZkExec.RunsSeams

/-!
# The deployed round plan reads the coins before each round only (`CausalExe`)

`QE`'s round `n ≥ 1` is body `n` of `sessionCut` on the cut's sources at the coins `c`, answered by `asksCut`, which
returns coin `k` right after body `k` and throws once body `n` is sent. So body `n` reads the coins before `n` through
the sends (`asksCut_congr`), and the sources through what the session reads before it sends body `n`. Every source
value the session reads while it has sent `s` bodies reads only the coins before `s` (its fields at `restAt`'s slots,
and the guards before them), and `s ≤ n` until the throw, so body `n` is the same at any coins agreeing before `n`.

* **Equal runs** and **the phases at sources that agree where read** (`RepAgree`): `CutEq`.
* **The cut's sources agree where read** (`repAgree_czExe`): at coins agreeing before `n`, each field `repSrcCut` reads
  is the same wherever the session reads it before body `n`.
-/

@[expose] public section

/-! ## The pads trees by tree position

`Round1Pads` (`Claims/ZK/Round1Pads.lean`), here under `Cx`: a rep's pads leaf at tree position `p`
commits a column that reads no coin (`Cx.digOf_pads`), and a cut's pads leaf at `q` is the leaf at `q`
(`Cx.padLeafCut_cutAt`). `treeEquiv` is irreducible here: unfolding it decides its `∃ e, …` over the tree's size.
-/

namespace FlockSoundness.Discharge.Composed.Cx

open Game Audit Exec Refine Flock Model ZkSession ZkLink ZkExec DecodeZL

local notation "𝔽" => FlockLevel3.GF128

variable (P : LigProver) {mPts : ℕ}

attribute [local irreducible] treeEquiv

/-- **Each rep's pads tree places every position at its index** at every coin vector of a `--zk` table whose pads code
has `Q ≤ n ≤ 2^63` positions: the opened ones at `padPos`, the others at `unPos`. -/
theorem coinedOf_sP_val (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (coinedOf P Tz st).Rest)
    (hQ : Zk.Q ≤ padNOf Tz st) (hN : padNOf Tz st ≤ 2 ^ 63) (r : Fin 2)
    (x : Fin (coinedOf P Tz st).z.nQ ⊕ Fin (coinedOf P Tz st).z.nPU) :
    ((((coinedOf P Tz st).T c).sP r) x).val =
      Sum.elim (padPos (max (min (padNOf Tz st) (2 ^ 63)) Zk.Q) Zk.Q ((bodyOf Tz st).padQueries r c.rest))
        (fun u => unPos (padNOf Tz st)
          (padPos (max (min (padNOf Tz st) (2 ^ 63)) Zk.Q) Zk.Q ((bodyOf Tz st).padQueries r c.rest)) u.val) x := by
  have hL : max (min (padNOf Tz st) (2 ^ 63)) Zk.Q = padNOf Tz st := by
    rw [min_eq_left hN, max_eq_left hQ]
  have hQ' : Zk.Q ≤ max (min (padNOf Tz st) (2 ^ 63)) Zk.Q := le_max_right _ _
  have hsP : ((coinedOf P Tz st).T c).sP r = treeEquiv (a := (coinedOf P Tz st).z.nQ) (b := (coinedOf P Tz st).z.nPU)
      (Sum.elim (padPos (max (min (padNOf Tz st) (2 ^ 63)) Zk.Q) Zk.Q ((bodyOf Tz st).padQueries r c.rest))
        (fun u => unPos (padNOf Tz st)
          (padPos (max (min (padNOf Tz st) (2 ^ 63)) Zk.Q) Zk.Q ((bodyOf Tz st).padQueries r c.rest)) u.val)) := rfl
  rw [hsP]
  exact treeEquiv_val_pos (t := Zk.Q) (N := padNOf Tz st) _ (padPos_injective _ hQ')
    (fun i => lt_of_lt_of_eq (padPos_lt _ hQ' i) hL) hQ x

/-- **A `--zk` table's exposed pad `h_fa`** (`coinedOf`'s `ia`), the same at every coin vector. -/
theorem coinedOf_ia (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (coinedOf P Tz st).Rest) :
    ((coinedOf P Tz st).T c).ia = ⟨(Zk.PadLayout.new st.m st.c.kLog Tz.sch.k0).fa, Nat.lt_of_succ_lt (fa_lt Tz st)⟩ :=
  rfl

/-- **A `--zk` table's exposed pad `h_fb`** (`coinedOf`'s `ib`), the same at every coin vector. -/
theorem coinedOf_ib (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (coinedOf P Tz st).Rest) :
    ((coinedOf P Tz st).T c).ib = ⟨(Zk.PadLayout.new st.m st.c.kLog Tz.sch.k0).fb, fa_lt Tz st⟩ :=
  rfl

/-- **A rep's pads at its `h`** (`ZK.Table.hFull` at `ia`, `ib`): `h`'s entries, then `h_x`, `h_y`, `h_ia·h_ib`,
`h_x·h_y`. -/
noncomputable def padH {nE : ℕ} (ia ib : Fin nE) (h : (Fin nE → 𝔽) × 𝔽 × 𝔽) : ZK.PadIdx nE → 𝔽
  | .inl i => h.1 i
  | .inr j => ![h.2.1, h.2.2, h.1 ia * h.1 ib, h.2.1 * h.2.2] j

/-- **Rep `r`'s pads column at the code symbol of position `p`**: `(C_h, C_μ)` at `X = p + 1` (`innerPt`), pad `i`
(`padH` at `ia`, `ib`) at `X^(padIdx i)` and padding symbol `j` at `X^(len + j)`. -/
noncomputable def padColAt {Hi : Type} {z : ZkView.Sizes} (ia ib : Fin z.nE) (padLen : ℕ) (b : ZkView.Base z Hi)
    (r : Fin 2) (p : ℕ) : 𝔽 × 𝔽 :=
  (∑ i, innerPt p ^ padIdx i * padH ia ib (b.2.2.1 r) i + ∑ j : Fin z.nPad, innerPt p ^ (padLen + j.val) * b.2.2.2.1 r j,
    ∑ i, innerPt p ^ padIdx i * b.2.2.2.2.2.1 r i + ∑ j : Fin z.nPad, innerPt p ^ (padLen + j.val) * b.2.2.2.2.1 r j)

/-- **Rep `r`'s pads column at tree position `p` is `padColAt` at `X = p + 1`**, at every coin vector of a `--zk` table
whose pads code has `Q ≤ n ≤ 2^63` positions (`coinedOf_sP_val`): an opened column and a hidden one are both the code
there. -/
theorem coinedOf_colPAt (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (coinedOf P Tz st).Rest)
    (hQ : Zk.Q ≤ padNOf Tz st) (hN : padNOf Tz st ≤ 2 ^ 63)
    (w : Model.Witness (coinedOf P Tz st).S.m) (b : ZkView.Base (coinedOf P Tz st).z (coinedOf P Tz st).Hi) (r : Fin 2)
    (p : Fin ((coinedOf P Tz st).z.nQ + (coinedOf P Tz st).z.nPU)) :
    ((coinedOf P Tz st).T c).colPAt w b r p =
      padColAt (z := (coinedOf P Tz st).z)
        ⟨(Zk.PadLayout.new st.m st.c.kLog Tz.sch.k0).fa, Nat.lt_of_succ_lt (fa_lt Tz st)⟩
        ⟨(Zk.PadLayout.new st.m st.c.kLog Tz.sch.k0).fb, fa_lt Tz st⟩ (padLenOf Tz st) b r p.val := by
  have hs := coinedOf_sP_val P Tz st c hQ hN r
  have hH : ∀ (h : (Fin (coinedOf P Tz st).z.nE → 𝔽) × 𝔽 × 𝔽) i, ((coinedOf P Tz st).T c).tab.hFull h i =
      padH (nE := (coinedOf P Tz st).z.nE)
        ⟨(Zk.PadLayout.new st.m st.c.kLog Tz.sch.k0).fa, Nat.lt_of_succ_lt (fa_lt Tz st)⟩
        ⟨(Zk.PadLayout.new st.m st.c.kLog Tz.sch.k0).fb, fa_lt Tz st⟩ h i := fun h i => by
    rcases i with i | i <;> rfl
  have hH' := funext (hH (b.2.2.1 r))
  unfold ZkView.Coined.colPAt
  rcases h : (((coinedOf P Tz st).T c).sP r).symm p with k | u
  · have hk : padPos (max (min (padNOf Tz st) (2 ^ 63)) Zk.Q) Zk.Q ((bodyOf Tz st).padQueries r c.rest) k = p.val := by
      have e : ((coinedOf P Tz st).T c).sP r (.inl k) = p := by rw [← h, Equiv.apply_symm_apply]
      have := hs (.inl k)
      rw [e] at this
      exact this.symm
    have hC : ∀ i, ((coinedOf P Tz st).T c).Ch r k i = innerPt p.val ^ padIdx i := fun i => by
      show fOf ⟨(padPos (max (min (padNOf Tz st) (2 ^ 63)) Zk.Q) Zk.Q ((bodyOf Tz st).padQueries r c.rest) k + 1).toUInt64,
        0⟩ ^ padIdx i = _
      rw [hk]
      rfl
    have hP : ∀ j, ((coinedOf P Tz st).T c).Cp r k j = innerPt p.val ^ (padLenOf Tz st + j.val) := fun j => by
      show fOf ⟨(padPos (max (min (padNOf Tz st) (2 ^ 63)) Zk.Q) Zk.Q ((bodyOf Tz st).padQueries r c.rest) k + 1).toUInt64,
        0⟩ ^ (padLenOf Tz st + j.val) = _
      rw [hk]
      rfl
    refine Prod.ext ?_ ?_
    · show ∑ i, ((coinedOf P Tz st).T c).Ch r k i * ((coinedOf P Tz st).T c).tab.hFull (b.2.2.1 r) i +
          ∑ j, ((coinedOf P Tz st).T c).Cp r k j * b.2.2.2.1 r j = _
      simp only [hC, hP, hH']
      rfl
    · show ∑ i, ((coinedOf P Tz st).T c).Ch r k i * b.2.2.2.2.2.1 r i +
          ∑ j, ((coinedOf P Tz st).T c).Cp r k j * b.2.2.2.2.1 r j = _
      simp only [hC, hP]
      rfl
  · have hu : unPos (padNOf Tz st)
        (padPos (max (min (padNOf Tz st) (2 ^ 63)) Zk.Q) Zk.Q ((bodyOf Tz st).padQueries r c.rest)) u.val = p.val := by
      have e : ((coinedOf P Tz st).T c).sP r (.inr u) = p := by rw [← h, Equiv.apply_symm_apply]
      have := hs (.inr u)
      rw [e] at this
      exact this.symm
    have hC : ∀ i, ((coinedOf P Tz st).T c).ChU r u i = innerPt p.val ^ padIdx i := fun i => by
      show fOf ⟨(unPos (padNOf Tz st) (padPos (max (min (padNOf Tz st) (2 ^ 63)) Zk.Q) Zk.Q
        ((bodyOf Tz st).padQueries r c.rest)) u.val + 1).toUInt64, 0⟩ ^ padIdx i = _
      rw [hu]
      rfl
    have hP : ∀ j, ((coinedOf P Tz st).T c).CpU r u j = innerPt p.val ^ (padLenOf Tz st + j.val) := fun j => by
      show fOf ⟨(unPos (padNOf Tz st) (padPos (max (min (padNOf Tz st) (2 ^ 63)) Zk.Q) Zk.Q
        ((bodyOf Tz st).padQueries r c.rest)) u.val + 1).toUInt64, 0⟩ ^ (padLenOf Tz st + j.val) = _
      rw [hu]
      rfl
    refine Prod.ext ?_ ?_
    · show ∑ i, ((coinedOf P Tz st).T c).ChU r u i * ((coinedOf P Tz st).T c).tab.hFull (b.2.2.1 r) i +
          ∑ j, ((coinedOf P Tz st).T c).CpU r u j * b.2.2.2.1 r j = _
      simp only [hC, hP, hH']
      rfl
    · show ∑ i, ((coinedOf P Tz st).T c).ChU r u i * b.2.2.2.2.2.1 r i +
          ∑ j, ((coinedOf P Tz st).T c).CpU r u j * b.2.2.2.2.1 r j = _
      simp only [hC, hP]
      rfl

end FlockSoundness.Discharge.Composed.Cx

namespace Claims.ZKMal.Masked.Cx

open Claims FlockSoundness FlockSoundness.Refine FlockSoundness.Discharge

attribute [local irreducible] Composed.treeEquiv

/-- **A live table's pads code has `Q ≤ n ≤ 2^63` positions**: `n = innerN`, a power of two below `2^64` holding the
`len + K_PAD ≥ Q` symbols. -/
theorem padN_bounds (x : Msg) (j : Fin (jOf x)) :
    _root_.Flock.Zk.Q ≤ Composed.padNOf (zkTab x j).table (stAt x j) ∧
      Composed.padNOf (zkTab x j).table (stAt x j) ≤ 2 ^ 63 := by
  have hN : Composed.padNOf (zkTab x j).table (stAt x j) = ZkExec.innerN (ZkLink.layOf (stAt x j) (zkTab x j).sch) := rfl
  have hL := (zkTab x j).hL
  have h64 := (zkTab x j).hN
  rw [hN]
  refine ⟨le_trans (by simp [_root_.Flock.Zk.Q, _root_.Flock.Zk.K_PAD]) hL, ?_⟩
  have hp := FlockComplete.innerCols_pow ((ZkLink.layOf (stAt x j) (zkTab x j).sch).len + _root_.Flock.Zk.K_PAD)
  unfold ZkExec.innerN at h64 ⊢
  rw [hp] at h64 ⊢
  exact Nat.pow_le_pow_right (by norm_num) (by
    have := (Nat.pow_lt_pow_iff_right (by norm_num : 1 < 2)).1 h64
    omega)

/-- **Table `j`'s rep `r` pads column at tree position `p`** is `padColAt` at `X = p + 1`, at every coin vector. -/
theorem colPAt_tabC (P : Composed.LigProver) (x : Msg) (j : Fin (jOf x)) (c : Composed.CoinsZK (tabC P x j).Rest)
    (w : Model.Witness (tabC P x j).S.m) (b : ZkView.Base (tabC P x j).z (tabC P x j).Hi) (r : Fin 2)
    (p : Fin ((tabC P x j).z.nQ + (tabC P x j).z.nPU)) :
    ((tabC P x j).T c).colPAt w b r p =
      Composed.Cx.padColAt (z := (tabC P x j).z)
        ⟨(_root_.Flock.Zk.PadLayout.new (stAt x j).m (stAt x j).c.kLog (zkTab x j).table.sch.k0).fa,
          Nat.lt_of_succ_lt (Composed.fa_lt (zkTab x j).table (stAt x j))⟩
        ⟨(_root_.Flock.Zk.PadLayout.new (stAt x j).m (stAt x j).c.kLog (zkTab x j).table.sch.k0).fb,
          Composed.fa_lt (zkTab x j).table (stAt x j)⟩
        (Composed.padLenOf (zkTab x j).table (stAt x j)) b r p.val :=
  Composed.Cx.coinedOf_colPAt P (zkTab x j).table (stAt x j) c (padN_bounds x j).1 (padN_bounds x j).2 w b r p

/-- **A cut's pads leaf at tree position `q` is the leaf at `q`**, whatever the coins' tree. -/
theorem padLeafCut_cutAt (P : Composed.LigProver) (x : Msg) (cz : ∀ j, Composed.CoinsZK (tabC P x j).Rest)
    (wt : ∀ j, Model.Witness (tabC P x j).S.m) (ω : ∀ j, ZkView.Base (tabC P x j).z (tabC P x j).Hi)
    (lv : ZkView.Leaves (fun j => (tabC P x j).z) → ByteArray) (j : Fin (jOf x)) (r : Fin 2) {q : ℕ}
    (hq : q < (tabC P x j).z.nQ + (tabC P x j).z.nPU) :
    padLeafCut P x cz (cutAt (tabC P x) cz wt ω lv) j r q = lv ⟨j, .inr (.inl (r, ⟨q, hq⟩))⟩ := by
  unfold padLeafCut
  rw [dite_eq_left_of_eq_true (eq_true hq)]
  show (match (((tabC P x j).T (cz j)).sP r).symm ⟨q, hq⟩ with
    | .inl a => lv ⟨j, .inr (.inl (r, ((tabC P x j).T (cz j)).sP r (.inl a)))⟩
    | .inr u => lv ⟨j, .inr (.inl (r, ((tabC P x j).T (cz j)).sP r (.inr u)))⟩) = _
  rcases h : (((tabC P x j).T (cz j)).sP r).symm ⟨q, hq⟩ with a | u
  · have e : ((tabC P x j).T (cz j)).sP r (.inl a) = ⟨q, hq⟩ := by rw [← h, Equiv.apply_symm_apply]
    simp only [e]
  · have e : ((tabC P x j).T (cz j)).sP r (.inr u) = ⟨q, hq⟩ := by rw [← h, Equiv.apply_symm_apply]
    simp only [e]

/-- **A pads leaf's digest is the same at every coin vector** (`colPAt_tabC`). -/
theorem digOf_pads (P : Composed.LigProver) (x : Msg) (cz cz' : ∀ j, Composed.CoinsZK (tabC P x j).Rest)
    (wt : ∀ j, Model.Witness (tabC P x j).S.m) (ω : ∀ j, ZkView.Base (tabC P x j).z (tabC P x j).Hi) (j : Fin (jOf x))
    (r : Fin 2) (p : Fin ((tabC P x j).z.nQ + (tabC P x j).z.nPU)) :
    digOf (tabC P x) cz wt ω ⟨j, .inr (.inl (r, p))⟩ = digOf (tabC P x) cz' wt ω ⟨j, .inr (.inl (r, p))⟩ := by
  unfold digOf
  simp only [ZkView.Coined.content]
  rw [colPAt_tabC, colPAt_tabC]

end Claims.ZKMal.Masked.Cx

/-! ## The honest clear values read their coins in order -/

namespace FlockSoundness.Discharge.Composed.Cx

open Flock Model ZkSession ZkView ZK Refine

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

theorem foldN_congr {ρ ρ' : ℕ → 𝔽} (u : ℕ → 𝔽) :
    ∀ {j : ℕ}, (∀ i < j, ρ i = ρ' i) → Zerocheck.foldN ρ j u = Zerocheck.foldN ρ' j u
  | 0, _ => rfl
  | j + 1, h => by
    simp only [Zerocheck.foldN]
    rw [foldN_congr u fun i hi => h i (by omega), h j (by omega)]

theorem ofFn_get!_congr {α : Type} [Inhabited α] {n : ℕ} {f f' : Fin n → α} {i : ℕ}
    (h : ∀ a : Fin n, a.val < i → f a = f' a) : ∀ i' < i, (Array.ofFn f)[i']! = (Array.ofFn f')[i']! := by
  intro i' hi'
  by_cases hn : i' < n
  · rw [getElem!_pos _ i' (by simpa using hn), getElem!_pos _ i' (by simpa using hn), Array.getElem_ofFn,
      Array.getElem_ofFn]
    exact h ⟨i', hn⟩ hi'
  · rw [getElem!_neg _ i' (by simpa using hn), getElem!_neg _ i' (by simpa using hn)]

theorem range_map_get! {α : Type} [Inhabited α] (n i : ℕ) {g g' : ℕ → α} (h : i < n → g i = g' i) :
    ((Array.range n).map g)[i]! = ((Array.range n).map g')[i]! := by
  by_cases hi : i < n
  · rw [getElem!_pos _ i (by simpa using hi), getElem!_pos _ i (by simpa using hi)]
    simp [h hi]
  · rw [getElem!_neg _ i (by simpa using hi), getElem!_neg _ i (by simpa using hi)]

section Honest

variable {mPts : ℕ} (Tz : TabZK 𝔽 mPts) (st : Stmt) (c c' : CoinsZK (RestOf Tz st)) (r : Fin 2)
  (zw : Witness Tz.S.m) {L : ℕ} (R : Fin 2 → Fin L → 𝔽)

theorem zcH_r1 (hO : c.rest.rOuter r = c'.rest.rOuter r) :
    (zcH Tz st c r zw).round1ab = (zcH Tz st c' r zw).round1ab ∧
      (zcH Tz st c r zw).round1c = (zcH Tz st c' r zw).round1c := by
  simp only [zcH, Zerocheck.honest, rF, hO, and_self]

theorem zcH_round (hO : c.rest.rOuter r = c'.rest.rOuter r) (hz : c.rest.zc r = c'.rest.zc r) {i : ℕ}
    (hρ : ∀ a : Fin (st.m - 6), a.val < i → c.rest.rho r a = c'.rest.rho r a) :
    (zcH Tz st c r zw).rounds[i]! = (zcH Tz st c' r zw).rounds[i]! := by
  simp only [zcH, Zerocheck.honest, rF, zcF, hO, hz]
  refine range_map_get! _ _ fun _ => ?_
  have h := ofFn_get!_congr hρ
  have hc : ∀ i' < i, Zerocheck.coin (rhoF Tz st c r) i' = Zerocheck.coin (rhoF Tz st c' r) i' :=
    fun i' hi' => congrArg FlockLevel3.GF128.of (h i' hi')
  simp only [Zerocheck.roundMsg, foldN_congr _ hc]

theorem zcH_final (hO : c.rest.rOuter r = c'.rest.rOuter r) (hz : c.rest.zc r = c'.rest.zc r)
    (hρ : c.rest.rho r = c'.rest.rho r) :
    (zcH Tz st c r zw).finalA = (zcH Tz st c' r zw).finalA ∧ (zcH Tz st c r zw).finalB = (zcH Tz st c' r zw).finalB ∧
      (zcH Tz st c r zw).finalC = (zcH Tz st c' r zw).finalC := by
  simp only [zcH, Zerocheck.honest, rF, zcF, rhoF, hO, hz, hρ, and_self]

theorem take_ofFn_congr {n : ℕ} {f f' : Fin n → F128} (g : F128 → 𝔽) {j : ℕ}
    (h : ∀ a : Fin n, a.val < j → f a = f' a) :
    ((Array.ofFn f).toList.map g).take j = ((Array.ofFn f').toList.map g).take j := by
  apply List.ext_getElem (by simp)
  intro i h1 h2
  simp only [List.getElem_take, List.getElem_map, Array.getElem_toList, Array.getElem_ofFn]
  simp only [List.length_take, List.length_map, Array.length_toList, Array.size_ofFn] at h1
  rw [h ⟨i, by omega⟩ (by simp only; omega)]

theorem lay_fa : (layOf Tz st).fa = 128 + 2 * (st.m - 6) := rfl
theorem lay_fb : (layOf Tz st).fb = 129 + 2 * (st.m - 6) := by
  show 128 + 2 * (st.m - 6) + 1 = _
  omega
theorem lay_fc : (layOf Tz st).fc = 130 + 2 * (st.m - 6) := by
  show 128 + 2 * (st.m - 6) + 2 = _
  omega
theorem lay_zp (i : ℕ) : (layOf Tz st).zp i = 131 + 2 * (st.m - 6) + 2 * (st.c.kLog - 6) + i := by
  show 128 + 2 * (st.m - 6) + 2 + 1 + 2 * (st.c.kLog - 6) + 0 + i = _
  omega
theorem lay_lgMsg (j t : ℕ) :
    (layOf Tz st).lgMsg j t = 196 + 2 * (st.m - 6) + 2 * (st.c.kLog - 6) + 4 * j + t := by
  show 128 + 2 * (st.m - 6) + 2 + 1 + 2 * (st.c.kLog - 6) + 0 + 64 + 1 + 4 * j + t = _
  omega
theorem lay_lgE (i t : ℕ) :
    (layOf Tz st).lgE i t = 196 + 2 * (st.m - 6) + 2 * (st.c.kLog - 6) + 4 * Tz.sch.k0 + 2 * i + t := by
  show 128 + 2 * (st.m - 6) + 2 + 1 + 2 * (st.c.kLog - 6) + 0 + 64 + 1 + 4 * Tz.sch.k0 + 2 * i + t = _
  omega
theorem lay_hab : (layOf Tz st).hab = 200 + 2 * (st.m - 6) + 2 * (st.c.kLog - 6) + 4 * Tz.sch.k0 := by
  show 128 + 2 * (st.m - 6) + 2 + 1 + 2 * (st.c.kLog - 6) + 0 + 64 + 1 + 4 * Tz.sch.k0 + 4 = _
  omega

theorem honestAt_r1 (hO : c.rest.rOuter r = c'.rest.rOuter r) {k : ℕ} (hk : k < 128) :
    honestAt Tz st c r zw R k = honestAt Tz st c' r zw R k := by
  obtain ⟨h1, h2⟩ := zcH_r1 Tz st c c' r zw hO
  unfold honestAt
  by_cases h64 : k < 64
  · simp only [ite_eq_left h64, h1]
  · simp only [ite_eq_right h64, ite_eq_left hk, h2]

theorem honestAt_ml (hO : c.rest.rOuter r = c'.rest.rOuter r) (hz : c.rest.zc r = c'.rest.zc r) {i : ℕ}
    (hρ : ∀ a : Fin (st.m - 6), a.val < i → c.rest.rho r a = c'.rest.rho r a) (hi : i < st.m - 6) {k : ℕ}
    (hk1 : 128 + 2 * i ≤ k) (hk2 : k < 130 + 2 * i) :
    honestAt Tz st c r zw R k = honestAt Tz st c' r zw R k := by
  have h := zcH_round Tz st c c' r zw hO hz hρ
  have e : (k - 128) / 2 = i := by omega
  have := lay_fa Tz st
  unfold honestAt
  simp only [ite_eq_right (show ¬ k < 64 by omega), ite_eq_right (show ¬ k < 128 by omega),
    ite_eq_left (show k < (layOf Tz st).fa by omega), e, h]

theorem honestAt_f (hO : c.rest.rOuter r = c'.rest.rOuter r) (hz : c.rest.zc r = c'.rest.zc r)
    (hρ : c.rest.rho r = c'.rest.rho r) {k : ℕ} (hk1 : 128 + 2 * (st.m - 6) ≤ k) (hk2 : k < 131 + 2 * (st.m - 6)) :
    honestAt Tz st c r zw R k = honestAt Tz st c' r zw R k := by
  obtain ⟨h1, h2, h3⟩ := zcH_final Tz st c c' r zw hO hz hρ
  have := lay_fa Tz st
  have := lay_fb Tz st
  have := lay_fc Tz st
  unfold honestAt
  simp only [ite_eq_right (show ¬ k < 64 by omega), ite_eq_right (show ¬ k < 128 by omega),
    ite_eq_right (show ¬ k < (layOf Tz st).fa by omega)]
  by_cases ha : k = (layOf Tz st).fa
  · simp only [ite_eq_left ha, h1]
  by_cases hb : k = (layOf Tz st).fb
  · simp only [ite_eq_right ha, ite_eq_left hb, h2]
  simp only [ite_eq_right ha, ite_eq_right hb, ite_eq_left (show k = (layOf Tz st).fc by omega), h3]

/-- Past the zerocheck: the lincheck's and later regions. -/
theorem honestAt_past (k : ℕ) (hk : 131 + 2 * (st.m - 6) ≤ k) :
    honestAt Tz st c r zw R k =
      let lay := layOf Tz st
      if k < lay.zp 0 then
        (if (k - lay.fc - 1) % 2 = 0 then (lcH Tz st c r zw ((k - lay.fc - 1) / 2)).1
          else (lcH Tz st c r zw ((k - lay.fc - 1) / 2)).2)
      else if k < lay.zp 64 then zpH Tz st c r zw (k - lay.zp 0)
      else if k < lay.lgMsg 0 0 then
        ∑ y ∈ Finset.range (2 ^ (Tz.S.m - 7)), eqAt (oodPoint Tz st c r (k - lay.lgOod 0)) y * wordOf zw y
      else if k < lay.lgE 0 0 then
        let j := (k - lay.lgMsg 0 0) / 4
        let t := (k - lay.lgMsg 0 0) % 4
        let m := laneMsg Tz st c r zw j
        let x := if t < 2 then m.1 else m.2
        if t % 2 = 0 then (coords execArith x).1 else (coords execArith x).2
      else if k < lay.hab then
        let i := (k - lay.lgE 0 0) / 2
        let e := ∑ col : Fin L, execArith.embed (seqOf R i col) * bFold Tz st c r col.val
        if (k - lay.lgE 0 0) % 2 = 0 then (coords execArith e).1 else (coords execArith e).2
      else 0 := by
  have := lay_fa Tz st
  have := lay_fb Tz st
  have := lay_fc Tz st
  unfold honestAt
  simp only [ite_eq_right (show ¬ k < 64 by omega), ite_eq_right (show ¬ k < 128 by omega),
    ite_eq_right (show ¬ k < (layOf Tz st).fa by omega), ite_eq_right (show ¬ k = (layOf Tz st).fa by omega),
    ite_eq_right (show ¬ k = (layOf Tz st).fb by omega), ite_eq_right (show ¬ k = (layOf Tz st).fc by omega)]

theorem honestAt_lc (hz : c.rest.zc r = c'.rest.zc r) (ha : c.rest.lcAlpha r = c'.rest.lcAlpha r)
    (hb : c.rest.lcBeta r = c'.rest.lcBeta r) (hρ : c.rest.rho r = c'.rest.rho r) {j : ℕ}
    (ht : ∀ a : Fin (st.c.kLog - 6), a.val < j → c.rest.ts r a = c'.rest.ts r a) (hj : j < st.c.kLog - 6) {k : ℕ}
    (hk1 : 131 + 2 * (st.m - 6) + 2 * j ≤ k) (hk2 : k < 133 + 2 * (st.m - 6) + 2 * j) :
    honestAt Tz st c r zw R k = honestAt Tz st c' r zw R k := by
  have h : lcH Tz st c r zw j = lcH Tz st c' r zw j := by
    simp only [lcH, Lincheck.lcMsg, zcF, rhoF, hz, ha, hb, hρ, tsF, take_ofFn_congr fOf ht]
  have := lay_fc Tz st
  have := lay_zp Tz st 0
  have e : (k - (layOf Tz st).fc - 1) / 2 = j := by omega
  rw [honestAt_past Tz st c r zw R k (by omega), honestAt_past Tz st c' r zw R k (by omega)]
  simp only [ite_eq_left (show k < (layOf Tz st).zp 0 by omega), e, h]

theorem honestAt_zp (hρ : c.rest.rho r = c'.rest.rho r) (ht : c.rest.ts r = c'.rest.ts r) {k : ℕ}
    (hk1 : 131 + 2 * (st.m - 6) + 2 * (st.c.kLog - 6) ≤ k) (hk2 : k < 195 + 2 * (st.m - 6) + 2 * (st.c.kLog - 6)) :
    honestAt Tz st c r zw R k = honestAt Tz st c' r zw R k := by
  have h : zpH Tz st c r zw = zpH Tz st c' r zw := funext fun i => by simp only [zpH, tsF, rhoF, hρ, ht]
  have := lay_zp Tz st 0
  have := lay_zp Tz st 64
  rw [honestAt_past Tz st c r zw R k (by omega), honestAt_past Tz st c' r zw R k (by omega)]
  simp only [ite_eq_right (show ¬ k < (layOf Tz st).zp 0 by omega), ite_eq_left (show k < (layOf Tz st).zp 64 by omega), h]

theorem honestAt_ood (hp : c.rest.oodPt r = c'.rest.oodPt r) {k : ℕ}
    (hk : k = 195 + 2 * (st.m - 6) + 2 * (st.c.kLog - 6)) :
    honestAt Tz st c r zw R k = honestAt Tz st c' r zw R k := by
  have h : ∀ i, oodPoint Tz st c r i = oodPoint Tz st c' r i := fun i => by unfold oodPoint; rw [hp]
  have := lay_zp Tz st 0
  have := lay_zp Tz st 64
  have := lay_lgMsg Tz st 0 0
  rw [honestAt_past Tz st c r zw R k (by omega), honestAt_past Tz st c' r zw R k (by omega)]
  simp only [ite_eq_right (show ¬ k < (layOf Tz st).zp 0 by omega), ite_eq_right (show ¬ k < (layOf Tz st).zp 64 by omega),
    ite_eq_left (show k < (layOf Tz st).lgMsg 0 0 by omega), h]

/-- What level 0's basis reads: the ring switch's coins, the claims' points and the out-of-domain sample. -/
structure B0Agree : Prop where
  link : c.rest.link = c'.rest.link
  rOuter : c.rest.rOuter r = c'.rest.rOuter r
  rho : c.rest.rho r = c'.rest.rho r
  ts : c.rest.ts r = c'.rest.ts r
  rpp : c.rest.rpp r = c'.rest.rpp r
  rsGamma : c.rest.rsGamma r = c'.rest.rsGamma r
  oodPt : c.rest.oodPt r = c'.rest.oodPt r
  ood : c.rest.ood r = c'.rest.ood r

theorem laneB0_congr (h : B0Agree Tz st c c' r) : laneB0 Tz st c r = laneB0 Tz st c' r := by
  have hx : ∀ k, claimX Tz st c r k = claimX Tz st c' r k := fun k => by
    simp only [claimX, abX, lcT, zcRho, zcReq, h.ts, h.rho, h.rOuter, h.link]
  have hp : ∀ i, oodPoint Tz st c r i = oodPoint Tz st c' r i := fun i => by unfold oodPoint; rw [h.oodPt]
  funext y
  simp only [laneB0, openB, hx, hp, h.rpp, h.rsGamma, h.ood]

theorem laneF_congr {j : ℕ} (hl : ∀ i < j, laneCh Tz st c r i = laneCh Tz st c' r i) :
    laneF Tz st c r zw j = laneF Tz st c' r zw j := by
  induction j with
  | zero => rfl
  | succ j ih => simp only [laneF, ih fun i hi => hl i (by omega), hl j (by omega)]

theorem laneB_congr (h0 : laneB0 Tz st c r = laneB0 Tz st c' r) {j : ℕ}
    (hl : ∀ i < j, laneCh Tz st c r i = laneCh Tz st c' r i) : laneB Tz st c r j = laneB Tz st c' r j := by
  induction j with
  | zero => exact h0
  | succ j ih => simp only [laneB, ih fun i hi => hl i (by omega), hl j (by omega)]

theorem laneCh_congr {j : ℕ} (hl : ∀ a : Fin Tz.sch.k0, a.val < j → c.rest.lane r a = c'.rest.lane r a) :
    ∀ i < j, laneCh Tz st c r i = laneCh Tz st c' r i := fun i hi => by
  unfold laneCh seqOf
  split_ifs with h
  · rw [hl ⟨i, h⟩ hi]
  · rfl

theorem honestAt_msg (h0 : B0Agree Tz st c c' r) {j : ℕ}
    (hl : ∀ a : Fin Tz.sch.k0, a.val < j → c.rest.lane r a = c'.rest.lane r a) (hj : j < Tz.sch.k0) {k : ℕ}
    (hk1 : 196 + 2 * (st.m - 6) + 2 * (st.c.kLog - 6) + 4 * j ≤ k)
    (hk2 : k < 200 + 2 * (st.m - 6) + 2 * (st.c.kLog - 6) + 4 * j) :
    honestAt Tz st c r zw R k = honestAt Tz st c' r zw R k := by
  have hc := laneCh_congr Tz st c c' r hl
  have h : laneMsg Tz st c r zw j = laneMsg Tz st c' r zw j := by
    simp only [laneMsg, laneF_congr Tz st c c' r zw hc, laneB_congr Tz st c c' r (laneB0_congr Tz st c c' r h0) hc]
  have := lay_zp Tz st 0
  have := lay_zp Tz st 64
  have := lay_lgMsg Tz st 0 0
  have := lay_lgE Tz st 0 0
  have e : (k - (layOf Tz st).lgMsg 0 0) / 4 = j := by omega
  rw [honestAt_past Tz st c r zw R k (by omega), honestAt_past Tz st c' r zw R k (by omega)]
  simp only [ite_eq_right (show ¬ k < (layOf Tz st).zp 0 by omega), ite_eq_right (show ¬ k < (layOf Tz st).zp 64 by omega),
    ite_eq_right (show ¬ k < (layOf Tz st).lgMsg 0 0 by omega), ite_eq_left (show k < (layOf Tz st).lgE 0 0 by omega), e, h]

theorem honestAt_E (h0 : B0Agree Tz st c c' r) (hl : c.rest.lane r = c'.rest.lane r) {k : ℕ}
    (hk1 : 196 + 2 * (st.m - 6) + 2 * (st.c.kLog - 6) + 4 * Tz.sch.k0 ≤ k)
    (hk2 : k < 200 + 2 * (st.m - 6) + 2 * (st.c.kLog - 6) + 4 * Tz.sch.k0) :
    honestAt Tz st c r zw R k = honestAt Tz st c' r zw R k := by
  have hc := laneCh_congr Tz st c c' r (j := Tz.sch.k0) fun a _ => congrFun hl a
  have h : bFold Tz st c r = bFold Tz st c' r := by
    funext col
    simp only [bFold, laneB_congr Tz st c c' r (laneB0_congr Tz st c c' r h0) hc]
  have := lay_zp Tz st 0
  have := lay_zp Tz st 64
  have := lay_lgMsg Tz st 0 0
  have := lay_lgE Tz st 0 0
  have := lay_hab Tz st
  rw [honestAt_past Tz st c r zw R k (by omega), honestAt_past Tz st c' r zw R k (by omega)]
  simp only [ite_eq_right (show ¬ k < (layOf Tz st).zp 0 by omega), ite_eq_right (show ¬ k < (layOf Tz st).zp 64 by omega),
    ite_eq_right (show ¬ k < (layOf Tz st).lgMsg 0 0 by omega), ite_eq_right (show ¬ k < (layOf Tz st).lgE 0 0 by omega),
    ite_eq_left (show k < (layOf Tz st).hab by omega), h]

theorem bFold_congr (h0 : B0Agree Tz st c c' r) (hl : c.rest.lane r = c'.rest.lane r) :
    bFold Tz st c r = bFold Tz st c' r := by
  have hc := laneCh_congr Tz st c c' r (j := Tz.sch.k0) fun a _ => congrFun hl a
  funext col
  simp only [bFold, laneB_congr Tz st c c' r (laneB0_congr Tz st c c' r h0) hc]

theorem honestAt_end {k : ℕ} (hk : 200 + 2 * (st.m - 6) + 2 * (st.c.kLog - 6) + 4 * Tz.sch.k0 ≤ k) :
    honestAt Tz st c r zw R k = 0 := by
  have := lay_zp Tz st 0
  have := lay_zp Tz st 64
  have := lay_lgMsg Tz st 0 0
  have := lay_lgE Tz st 0 0
  have := lay_hab Tz st
  rw [honestAt_past Tz st c r zw R k (by omega)]
  simp only [ite_eq_right (show ¬ k < (layOf Tz st).zp 0 by omega), ite_eq_right (show ¬ k < (layOf Tz st).zp 64 by omega),
    ite_eq_right (show ¬ k < (layOf Tz st).lgMsg 0 0 by omega), ite_eq_right (show ¬ k < (layOf Tz st).lgE 0 0 by omega),
    ite_eq_right (show ¬ k < (layOf Tz st).hab by omega)]

/-- **Every masked value of rep `r` reads only the coins of its PIOP and level 0's lane phase.** -/
theorem honestAt_all (h0 : B0Agree Tz st c c' r) (hz : c.rest.zc r = c'.rest.zc r)
    (ha : c.rest.lcAlpha r = c'.rest.lcAlpha r) (hb : c.rest.lcBeta r = c'.rest.lcBeta r)
    (hl : c.rest.lane r = c'.rest.lane r) (k : ℕ) :
    honestAt Tz st c r zw R k = honestAt Tz st c' r zw R k := by
  by_cases h1 : k < 128
  · exact honestAt_r1 Tz st c c' r zw R h0.rOuter h1
  by_cases h2 : k < 128 + 2 * (st.m - 6)
  · exact honestAt_ml Tz st c c' r zw R h0.rOuter hz (i := (k - 128) / 2) (fun a _ => congrFun h0.rho a)
      (by omega) (by omega) (by omega)
  by_cases h3 : k < 131 + 2 * (st.m - 6)
  · exact honestAt_f Tz st c c' r zw R h0.rOuter hz h0.rho (by omega) h3
  by_cases h4 : k < 131 + 2 * (st.m - 6) + 2 * (st.c.kLog - 6)
  · exact honestAt_lc Tz st c c' r zw R hz ha hb h0.rho (j := (k - (131 + 2 * (st.m - 6))) / 2)
      (fun a _ => congrFun h0.ts a) (by omega) (by omega) (by omega)
  by_cases h5 : k < 195 + 2 * (st.m - 6) + 2 * (st.c.kLog - 6)
  · exact honestAt_zp Tz st c c' r zw R h0.rho h0.ts (by omega) h5
  by_cases h6 : k < 196 + 2 * (st.m - 6) + 2 * (st.c.kLog - 6)
  · exact honestAt_ood Tz st c c' r zw R h0.oodPt (by omega)
  by_cases h7 : k < 196 + 2 * (st.m - 6) + 2 * (st.c.kLog - 6) + 4 * Tz.sch.k0
  · exact honestAt_msg Tz st c c' r zw R h0 (j := (k - (196 + 2 * (st.m - 6) + 2 * (st.c.kLog - 6))) / 4)
      (fun a _ => congrFun hl a) (by omega) (by omega) (by omega)
  by_cases h8 : k < 200 + 2 * (st.m - 6) + 2 * (st.c.kLog - 6) + 4 * Tz.sch.k0
  · exact honestAt_E Tz st c c' r zw R h0 hl (by omega) h8
  · exact (honestAt_end Tz st c r zw R (by omega)).trans (honestAt_end Tz st c' r zw R (by omega)).symm

/-- What rep `r`'s inner proof reads of its coins: level 0's basis, the PIOP's, the lanes, the restart's pair, `ε`
and the `γ`s. -/
structure InAgree extends B0Agree Tz st c c' r : Prop where
  zc : c.rest.zc r = c'.rest.zc r
  lcAlpha : c.rest.lcAlpha r = c'.rest.lcAlpha r
  lcBeta : c.rest.lcBeta r = c'.rest.lcBeta r
  rPrime : c.rest.rPrime r = c'.rest.rPrime r
  lane : c.rest.lane r = c'.rest.lane r
  rhoP : c.rho r = c'.rho r
  zeta : c.rest.zeta r = c'.rest.zeta r
  gammas : c.rest.gammas r = c'.rest.gammas r

/-- **Rep `r`'s ten constraints read its coins and its two masked claims' `ŝ`.** -/
theorem rowsOf_congr {nE nC : ℕ} (v : Fin nE → 𝔽) (sh sh' : Fin nC → Fin 128 → 𝔽) (T' : 𝕂) (ρσ : 𝔽 × 𝔽)
    (hA : InAgree Tz st c c' r) (hs0 : sHatAt sh (2 * r.val) = sHatAt sh' (2 * r.val))
    (hs1 : sHatAt sh (2 * r.val + 1) = sHatAt sh' (2 * r.val + 1)) :
    rowsOf Tz st c r v sh T' ρσ = rowsOf Tz st c' r v sh' T' ρσ := by
  have e0 : shArr sh (2 * r.val) = shArr sh' (2 * r.val) := by unfold shArr; rw [hs0]
  have e1 : shArr sh (2 * r.val + 1) = shArr sh' (2 * r.val + 1) := by unfold shArr; rw [hs1]
  have hsw : ∀ k, swAt Tz st c r sh k = swAt Tz st c' r sh' k := fun k => by
    unfold swAt
    rw [hs0, hs1, hA.link]
  have hT : openT Tz st c r sh = openT Tz st c' r sh' := by
    unfold openT
    simp only [hsw, hA.rsGamma, hA.rpp]
  have hrun : ∀ j, laneRun Tz st c r v sh j = laneRun Tz st c' r v sh' j := by
    intro j
    induction j with
    | zero => simp only [laneRun, hT, hA.ood]
    | succ j ih => simp only [laneRun, ih, laneCh, hA.lane]
  simp only [rowsOf, rowCEval, rowAB, rowLC, rowRsAb, rowRsC, laneRow, abClaim, cClaim, combvOf, zcF, rF, rhoF, tsF,
    innerEps, rhoK, e0, e1, hrun, hA.zc, hA.rOuter, hA.rho, hA.lcAlpha, hA.lcBeta, hA.ts, hA.rPrime, hA.rhoP, hA.zeta]

/-- **The inner proof's batched coefficients read rep `r`'s masked values, its two masked claims' `ŝ`, the restart's
`T'`, the triple's `(ρ, σ)` and its coins.** -/
theorem lamOf_congr {z : Sizes} (pre pre' : PreOf z) (hA : InAgree Tz st c c' r) (hv : pre.2.1 r = pre'.2.1 r)
    (hs0 : sHatAt pre.1 (2 * r.val) = sHatAt pre'.1 (2 * r.val))
    (hs1 : sHatAt pre.1 (2 * r.val + 1) = sHatAt pre'.1 (2 * r.val + 1))
    (hT : (pre.2.2.2.2.2 r).1 = (pre'.2.2.2.2.2 r).1) (hρσ : pre.2.2.2.2.1 r = pre'.2.2.2.2.1 r) :
    lamOf Tz st z c r pre = lamOf Tz st z c' r pre' := by
  funext i
  unfold lamOf
  have hg : innerGamma Tz st c r = innerGamma Tz st c' r := funext fun k => by simp only [innerGamma, hA.gammas]
  rw [hv, hT, hρσ, rowsOf_congr Tz st c c' r _ _ _ _ _ hA hs0 hs1, hg]

end Honest

section Fields

variable (P : LigProver) {mPts : ℕ} (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (RestOf Tz st))

theorem coinedOf_nu (w q v) : ((coinedOf P Tz st).T c).tab.nu w q v = ∑ y : Fin (2 ^ (Tz.S.m - 7)),
    if y.val ∈ maskWordsOf st then 0
    else fOf (eqWord (pointsOf c.rest q) y.val) * ZK.Table.castF (F := 𝔽) (wordBit w y.val v.val) := rfl

theorem coinedOf_eqw (q y) :
    ((coinedOf P Tz st).T c).tab.eqw q y = fOf (eqWord (pointsOf c.rest q) ((maskWordsOf st).getD y.val 0)) := rfl

theorem coinedOf_sReg (w q v) : ((coinedOf P Tz st).T c).tab.sReg w q v = ∑ y : Fin (2 ^ (Tz.S.m - 7)),
    fOf (eqWord (regionPtOf st c.rest.link q) y.val) * ZK.Table.castF (F := 𝔽) (wordBit w y.val v.val) := rfl

theorem coinedOf_rho (r i) : ((coinedOf P Tz st).T c).tab.ρ r i =
    if i = 0 then FlockLevel3.GF256.of (c.rho r).1 else FlockLevel3.GF256.of (c.rho r).2 := rfl

theorem coinedOf_beta (r) : ((coinedOf P Tz st).T c).tab.β r = fOf (c.beta r) := rfl

theorem coinedOf_rep (k) : ((coinedOf P Tz st).T c).rep k = if k.val < 2 then 0 else 1 := rfl

theorem coinedOf_embed : ((coinedOf P Tz st).T c).tab.embed = FlockLevel3.embed := rfl

theorem coined_embed : ((coinedOf P Tz st).T c).embed = FlockLevel3.embed := rfl

variable (c' : CoinsZK (RestOf Tz st))

theorem claimsOf_congr (r : Fin 2) (h1 : c.rest.zc r = c'.rest.zc r) (h2 : c.rest.rho r = c'.rest.rho r)
    (h3 : c.rest.rOuter r = c'.rest.rOuter r) (h4 : c.rest.rPrime r = c'.rest.rPrime r)
    (h5 : c.rest.ts r = c'.rest.ts r) : claimsOf c.rest r = claimsOf c'.rest r := by
  simp only [claimsOf, h1, h2, h3, h4, h5]

theorem coined_rankOn {K W Hi VL : Type} [Field K] {z : ZkView.Sizes} (T T' : ZkView.Coined K W Hi VL z)
    (Pr : Fin z.nC → Prop) (h : ∀ k, Pr k → T.eqw k = T'.eqw k) : T.tab.RankOn Pr ↔ T'.tab.RankOn Pr := by
  unfold ZK.Table.RankOn
  have e : (fun (u : Fin z.nMask → ZMod 2) (k : {k // Pr k}) => ∑ y, T.eqw k.1 y * ZK.Table.castF (F := 𝔽) (u y)) =
      fun u k => ∑ y, T'.eqw k.1 y * ZK.Table.castF (F := 𝔽) (u y) := by
    funext u k
    rw [h k.1 k.2]
  exact Iff.of_eq (congrArg Function.Surjective e)

theorem eqw_congr (q) (h : pointsOf c.rest q = pointsOf c'.rest q) :
    ((coinedOf P Tz st).T c).eqw q = ((coinedOf P Tz st).T c').eqw q := by
  funext y
  show fOf (eqWord (pointsOf c.rest q) ((maskWordsOf st).getD y.val 0)) =
    fOf (eqWord (pointsOf c'.rest q) ((maskWordsOf st).getD y.val 0))
  rw [h]

theorem nu_congr (w) (q : Fin (coinedOf P Tz st).z.nC) (h : pointsOf c.rest q = pointsOf c'.rest q) :
    ((coinedOf P Tz st).T c).nu w q = ((coinedOf P Tz st).T c').nu w q := by
  funext v
  show (∑ y : Fin (2 ^ (Tz.S.m - 7)), if y.val ∈ maskWordsOf st then 0
      else fOf (eqWord (pointsOf c.rest q) y.val) * ZK.Table.castF (F := 𝔽) (wordBit w y.val v.val)) =
    ∑ y : Fin (2 ^ (Tz.S.m - 7)), if y.val ∈ maskWordsOf st then 0
      else fOf (eqWord (pointsOf c'.rest q) y.val) * ZK.Table.castF (F := 𝔽) (wordBit w y.val v.val)
  rw [h]

theorem sReg_congr (w) (q : Fin (coinedOf P Tz st).z.nR) (h : c.rest.link = c'.rest.link) :
    ((coinedOf P Tz st).T c).sReg w q = ((coinedOf P Tz st).T c').sReg w q := by
  funext v
  show (∑ y : Fin (2 ^ (Tz.S.m - 7)),
      fOf (eqWord (regionPtOf st c.rest.link q) y.val) * ZK.Table.castF (F := 𝔽) (wordBit w y.val v.val)) =
    ∑ y : Fin (2 ^ (Tz.S.m - 7)),
      fOf (eqWord (regionPtOf st c'.rest.link q) y.val) * ZK.Table.castF (F := 𝔽) (wordBit w y.val v.val)
  rw [h]

theorem coined_val_congr {K W Hi VL : Type} [Field K] {z : ZkView.Sizes} (T T' : ZkView.Coined K W Hi VL z) (w : W)
    (q : Fin z.nC) (u : Fin z.nMask → Fin 128 → ZMod 2) (h1 : T.nu w q = T'.nu w q) (h2 : T.eqw q = T'.eqw q) :
    (fun v => T.tab.nu w q v + ∑ y, T.tab.eqw q y * ZK.Table.castF (F := 𝔽) (u y v)) =
      fun v => T'.tab.nu w q v + ∑ y, T'.tab.eqw q y * ZK.Table.castF (F := 𝔽) (u y v) := by
  show (fun v => T.nu w q v + ∑ y, T.eqw q y * ZK.Table.castF (F := 𝔽) (u y v)) =
    fun v => T'.nu w q v + ∑ y, T'.eqw q y * ZK.Table.castF (F := 𝔽) (u y v)
  rw [h1, h2]

/-- **`τ` reads its batched coefficients** (and the base's `μ`). -/
theorem coined_tau_congr {K W Hi VL : Type} [Field K] {z : ZkView.Sizes} (T T' : ZkView.Coined K W Hi VL z) (w : W)
    (b : ZkView.Base z Hi) (r : Fin 2)
    (h : T.lam r (T.tab.view w (T.randOf b 0)).pre = T'.lam r (T'.tab.view w (T'.randOf b 0)).pre) :
    (T.tab.view w (T.randOf b 0)).tau r = (T'.tab.view w (T'.randOf b 0)).tau r := by
  show ∑ i, T.lam r (T.tab.view w (T.randOf b 0)).pre i * b.2.2.2.2.2.1 r i =
    ∑ i, T'.lam r (T'.tab.view w (T'.randOf b 0)).pre i * b.2.2.2.2.2.1 r i
  rw [h]

/-- **The rank check reads its claims' points** (of the reps up to `r`). -/
theorem rankOn_congr (r : Fin 2)
    (h : ∀ k, ((coinedOf P Tz st).T c).rep k ≤ r → pointsOf c.rest k = pointsOf c'.rest k) :
    ((coinedOf P Tz st).T c).tab.RankOn (fun k => ((coinedOf P Tz st).T c).rep k ≤ r) ↔
      ((coinedOf P Tz st).T c').tab.RankOn (fun k => ((coinedOf P Tz st).T c').rep k ≤ r) := by
  show _ ↔ ((coinedOf P Tz st).T c').tab.RankOn (fun k => ((coinedOf P Tz st).T c).rep k ≤ r)
  exact coined_rankOn _ _ _ fun k hk => eqw_congr P Tz st c c' k (h k hk)

/-- **The restart's check reads its pair coins.** -/
theorem wr_congr (r : Fin 2) (h : c.rho r = c'.rho r) :
    Function.Bijective (fun p : 𝔽 × 𝔽 => ((coinedOf P Tz st).T c).tab.Wr r p.1 p.2) ↔
      Function.Bijective (fun p : 𝔽 × 𝔽 => ((coinedOf P Tz st).T c').tab.Wr r p.1 p.2) := by
  have e : (fun p : 𝔽 × 𝔽 => ((coinedOf P Tz st).T c).tab.Wr r p.1 p.2) =
      fun p => ((coinedOf P Tz st).T c').tab.Wr r p.1 p.2 := by
    funext p
    simp only [ZK.Table.Wr, coinedOf_rho, coinedOf_embed, h]
  rw [e]

/-- **The inner proof's check reads its `β`.** -/
theorem beta_congr (r : Fin 2) (h : c.beta r = c'.beta r) :
    ((coinedOf P Tz st).T c).tab.β r ≠ 0 ↔ ((coinedOf P Tz st).T c').tab.β r ≠ 0 := by
  rw [coinedOf_beta, coinedOf_beta, h]

theorem rbar_congr (r : Fin 2) (h : c.rest.lane r = c'.rest.lane r) :
    ((coinedOf P Tz st).T c).rbar r = ((coinedOf P Tz st).T c').rbar r := by
  funext l
  show eqLane (c.rest.lane r) l.val = eqLane (c'.rest.lane r) l.val
  rw [h]

theorem ρ_congr (r : Fin 2) (h : c.rho r = c'.rho r) :
    ((coinedOf P Tz st).T c).ρ r = ((coinedOf P Tz st).T c').ρ r := by
  funext i
  show (if i = 0 then FlockLevel3.GF256.of (c.rho r).1 else FlockLevel3.GF256.of (c.rho r).2) =
    if i = 0 then FlockLevel3.GF256.of (c'.rho r).1 else FlockLevel3.GF256.of (c'.rho r).2
  rw [h]

theorem embed_congr : ((coinedOf P Tz st).T c).embed = ((coinedOf P Tz st).T c').embed :=
  (coined_embed P Tz st c).trans (coined_embed P Tz st c').symm

theorem lanes_congr : ((coinedOf P Tz st).T c).lanes = ((coinedOf P Tz st).T c').lanes := by
  funext w u l col
  show wordOf (zWith st w u) (l.val + (coinedOf P Tz st).z.nL * col.val) =
    wordOf (zWith st w u) (l.val + (coinedOf P Tz st).z.nL * col.val)
  rfl

theorem β_congr (r : Fin 2) (h : c.beta r = c'.beta r) :
    ((coinedOf P Tz st).T c).β r = ((coinedOf P Tz st).T c').β r := by
  show fOf (c.beta r) = fOf (c'.beta r)
  rw [h]

theorem ζ_congr (r : Fin 2) (h : c.rest.zeta r = c'.rest.zeta r) :
    ((coinedOf P Tz st).T c).ζ r = ((coinedOf P Tz st).T c').ζ r := by
  show fOf (c.rest.zeta r) = fOf (c'.rest.zeta r)
  rw [h]

theorem ia_congr : ((coinedOf P Tz st).T c).ia = ((coinedOf P Tz st).T c').ia :=
  Fin.ext (show (Zk.PadLayout.new st.m st.c.kLog Tz.sch.k0).fa = (Zk.PadLayout.new st.m st.c.kLog Tz.sch.k0).fa from rfl)

theorem ib_congr : ((coinedOf P Tz st).T c).ib = ((coinedOf P Tz st).T c').ib :=
  Fin.ext (show (Zk.PadLayout.new st.m st.c.kLog Tz.sch.k0).fb = (Zk.PadLayout.new st.m st.c.kLog Tz.sch.k0).fb from rfl)

end Fields

end FlockSoundness.Discharge.Composed.Cx

namespace FlockComplete.ZkTables

open FlockDeploy FlockSoundness FlockSoundness.Discharge

/-- **A masked session in scope links at two points.** -/
theorem points_of_setup {x : Claims.Msg} {s : Ins} {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    (hscope : Claims.Flock.Scope (deployZk _root_.Flock.Readable.relation) x)
    (hsetup : setupOf true x = some (s, spec, sts)) : spec.points = 2 := by
  obtain ⟨p, hp, hsc⟩ := (_root_.Flock.Readable.scope_iff (deployZk _root_.Flock.Readable.relation)
    _root_.Flock.Readable.relation (readOf true) rfl (fun _ _ h => readOf_zk h) x).1 hscope
  have hi := (setupOf_some hsetup).1
  unfold _root_.Flock.Readable.parseBytes readOf at hp
  rw [hi] at hp
  have ht := _root_.Flock.Readable.toRead_of_parse (r := s.read) hp
  obtain ⟨hc, -⟩ := (_root_.Flock.Readable.scope_ok_iff p).1 hsc
  have hR := ((_root_.Flock.Readable.check_ok_iff _ _ _ _ _ _ _ _).1 hc).1
  have hpi : p.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simp [hR.2.2.2.2.2])
  have hset := Ins.setup_read (setupOf_some hsetup).2
  rw [← ht] at hset
  unfold _root_.Flock.Readable.Read.setup at hset
  rw [hpi] at hset
  exact setupTables_points hset

end FlockComplete.ZkTables

namespace Claims.ZKMal.Masked

open Claims FlockSoundness.ZK FlockSoundness.Discharge CoinLayer
open FlockComplete.CutEq

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

/-- **The cut at the coins `c`** on the simulator's leaves `H(dig ⊕ b, c)` (`RoundPlan.Causal`'s). -/
noncomputable abbrev cutC (x : Msg) (wt : ∀ j, FlockSoundness.Model.Witness (tabC PE x j).S.m)
    (ω : ∀ j, ZkView.Base (tabC PE x j).z (tabC PE x j).Hi)
    (p : ZkView.Leaves (fun j => (tabC PE x j).z) → ZkView.Bits × ByteArray) (c : Fin (refK x) → FlockDeploy.Coin) :
    ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Cut :=
  cutAt (tabC PE x) (czExe PE x c) wt ω fun l =>
    ZkView.leafH (digOf (tabC PE x) (czExe PE x c) wt ω l + (p l).1, (p l).2)

theorem stsOf_of {x : Msg} {s spec sts} (h : FlockDeploy.setupOf true x = some (s, spec, sts)) : stsOf x = sts := by
  unfold stsOf
  rw [h]

/-! ## The commitments read no coin -/

theorem commit0Cut_congr (ms : _root_.Flock.MerkleScheme) (lv : _root_.Flock.Level) {src src' : FlockComplete.TabSrc}
    (h0 : src.leaf0 = src'.leaf0) (h1 : src.row0 = src'.row0) (h2 : src.salt0 = src'.salt0) :
    FlockComplete.commit0Cut hsE ms lv src = FlockComplete.commit0Cut hsE ms lv src' := by
  unfold FlockComplete.commit0Cut
  rw [h0, h1, h2]

theorem padsTreeCut_congr (N : ℕ) {src src' : FlockComplete.RepSrc} (h0 : src.padLeaf = src'.padLeaf)
    (h1 : src.ch = src'.ch) (h2 : src.cm = src'.cm) (h3 : src.padSalt = src'.padSalt) :
    FlockComplete.padsTreeCut hsE N src = FlockComplete.padsTreeCut hsE N src' := by
  unfold FlockComplete.padsTreeCut
  rw [h0, h1, h2, h3]

section Proj

variable {x : Msg} (cz : ∀ j, Composed.CoinsZK (tabC PE x j).Rest) (cut : ∀ j, ((tabC PE x j).T (cz j)).tab.Cut)

theorem tabSrcCut_live {i : ℕ} (h : i < jOf x) : tabSrcCut PE x cz cut i =
    { leaf0 := leafAt PE x cz cut ⟨i, h⟩, row0 := fun _ => #[], salt0 := fun _ => ByteArray.empty,
      rep := fun r => if hr : r < 2 then repSrcCut PE x cz cut ⟨i, h⟩ ⟨r, hr⟩ else default } := by
  unfold tabSrcCut
  rw [dif_pos h]

theorem tabSrcCut_leaf0 {i : ℕ} (h : i < jOf x) : (tabSrcCut PE x cz cut i).leaf0 = leafAt PE x cz cut ⟨i, h⟩ := by
  unfold tabSrcCut
  rw [dif_pos h]

theorem tabSrcCut_rep {i r : ℕ} (h : i < jOf x) (hr : r < 2) :
    (tabSrcCut PE x cz cut i).rep r = repSrcCut PE x cz cut ⟨i, h⟩ ⟨r, hr⟩ := by
  unfold tabSrcCut
  rw [dif_pos h]
  exact dif_pos hr

theorem tabSrcCut_rep_dead {i r : ℕ} (h : i < jOf x) (hr : ¬ r < 2) : (tabSrcCut PE x cz cut i).rep r = default := by
  unfold tabSrcCut
  rw [dif_pos h]
  exact dif_neg hr

theorem tabSrcCut_dead {i : ℕ} (h : ¬ i < jOf x) : tabSrcCut PE x cz cut i = default := by
  unfold tabSrcCut
  rw [dif_neg h]

theorem repSrcCut_padLeaf (j : Fin (jOf x)) (r : Fin 2) :
    (repSrcCut PE x cz cut j r).padLeaf = padLeafCut PE x cz cut j r := by
  simp only [repSrcCut]

end Proj

section Agree

variable (x : Msg) (wt : ∀ j, FlockSoundness.Model.Witness (tabC PE x j).S.m)
  (ω : ∀ j, ZkView.Base (tabC PE x j).z (tabC PE x j).Hi)
  (p : ZkView.Leaves (fun j => (tabC PE x j).z) → ZkView.Bits × ByteArray)

theorem leaf0_agree (c c' : Fin (refK x) → FlockDeploy.Coin) (j : Fin (jOf x)) :
    leafAt PE x (czExe PE x c) (cutC x wt ω p c) j = leafAt PE x (czExe PE x c') (cutC x wt ω p c') j :=
  funext fun q => by
    by_cases hq : q < (tabC PE x j).z.nS + (tabC PE x j).z.nU
    · rw [leafAt_cutAt PE x _ wt ω _ j hq, leafAt_cutAt PE x _ wt ω _ j hq,
        digOf_inl PE x (czExe PE x c) (czExe PE x c') wt ω j]
    · simp only [leafAt, hq, ↓reduceDIte]

theorem padLeaf_agree (c c' : Fin (refK x) → FlockDeploy.Coin) (j : Fin (jOf x)) (r : Fin 2) :
    padLeafCut PE x (czExe PE x c) (cutC x wt ω p c) j r = padLeafCut PE x (czExe PE x c') (cutC x wt ω p c') j r :=
  funext fun q => by
    by_cases hq : q < (tabC PE x j).z.nQ + (tabC PE x j).z.nPU
    · rw [Cx.padLeafCut_cutAt PE x _ wt ω _ j r hq, Cx.padLeafCut_cutAt PE x _ wt ω _ j r hq,
        Cx.digOf_pads PE x (czExe PE x c) (czExe PE x c') wt ω j r]
    · simp only [padLeafCut, hq, ↓reduceDIte]

end Agree

/-! ## The reps' bookkeeping -/

/-- **Each rep's sources agree where read**, at coins agreeing below `n`: table `j`'s rep `r`, from its first body
`repStartZk (2j + r)`, with its region claims. -/
def RepOk : Prop :=
  ∀ x, Live x → ∀ wt ω p (c c' : Fin (refK x) → FlockDeploy.Coin) (n : ℕ),
    (∀ j : Fin (refK x), j.val < n → c j = c' j) → ∀ s spec, FlockDeploy.setupOf true x = some (s, spec, stsOf x) →
      ∀ (j : Fin (jOf x)) (r : Fin 2) (e : Array _root_.Flock.Claim), e.size = (stAt x j).regions.size * spec.points →
        RepAgree n (FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val)) (_root_.Flock.Setup.ofCircuit (stAt x j))
          (zkTab x j).sch e (repSrcCut PE x (czExe PE x c) (cutC x wt ω p c) j r)
          (repSrcCut PE x (czExe PE x c') (cutC x wt ω p c') j r)

theorem schOk_of {st : _root_.Flock.Stmt} (y : ZkHidden.ZkTab st) : SchOk (_root_.Flock.Setup.ofCircuit st) y.sch := by
  obtain ⟨h1, -, h3, h4, -, ⟨hn, hL, -⟩, hlv, -⟩ := FlockSoundness.Discharge.ZkExec.zk_schedule_seams _ _ _ y.hz
  exact ⟨show 14 ≤ st.m by omega, by omega, h4, hL, fun j hj => (hlv j hj).q_ne_zero⟩

section Generic

variable {x : Msg} (cut : ∀ c : Fin (refK x) → FlockDeploy.Coin, ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Cut)
  {c c' : Fin (refK x) → FlockDeploy.Coin}

/-- **The commitments of cuts whose level-0 and pads leaves agree are the same.** -/
theorem commitsCut_agree
    (hl : ∀ j, leafAt PE x (czExe PE x c) (cut c) j = leafAt PE x (czExe PE x c') (cut c') j)
    (hpl : ∀ j r, padLeafCut PE x (czExe PE x c) (cut c) j r = padLeafCut PE x (czExe PE x c') (cut c') j r) :
    (∀ ms lv i, FlockComplete.commit0Cut hsE ms lv (tabSrcCut PE x (czExe PE x c) (cut c) i) =
        FlockComplete.commit0Cut hsE ms lv (tabSrcCut PE x (czExe PE x c') (cut c') i)) ∧
      (∀ N i r, FlockComplete.padsTreeCut hsE N ((tabSrcCut PE x (czExe PE x c) (cut c) i).rep r) =
        FlockComplete.padsTreeCut hsE N ((tabSrcCut PE x (czExe PE x c') (cut c') i).rep r)) := by
  refine ⟨fun ms lv i => ?_, fun N i r => ?_⟩
  · by_cases h : i < jOf x
    · rw [tabSrcCut_live _ _ h, tabSrcCut_live _ _ h]
      exact commit0Cut_congr ms lv (hl ⟨i, h⟩) rfl rfl
    · rw [tabSrcCut_dead _ _ h, tabSrcCut_dead _ _ h]
  · by_cases h : i < jOf x
    · by_cases hr : r < 2
      · rw [tabSrcCut_rep _ _ h hr, tabSrcCut_rep _ _ h hr]
        refine padsTreeCut_congr N ?_ ?_ ?_ ?_
        · rw [repSrcCut_padLeaf, repSrcCut_padLeaf]
          exact hpl ⟨i, h⟩ ⟨r, hr⟩
        all_goals simp only [repSrcCut]
      · rw [tabSrcCut_rep_dead _ _ h hr, tabSrcCut_rep_dead _ _ h hr]
    · rw [tabSrcCut_dead _ _ h, tabSrcCut_dead _ _ h]

/-- **Each rep stream's sources agree where read from its first body** at cuts whose reps' sources do, table `j`'s
rep `r` from `repStartZk (2j + r)`. -/
theorem repStreams_agree (hx : Live x) {n : ℕ} {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec}
    (hsx : FlockDeploy.setupOf true x = some (s, spec, stsOf x))
    (hR : ∀ (j : Fin (jOf x)) (r : Fin 2) (e : Array _root_.Flock.Claim),
      e.size = (stAt x j).regions.size * spec.points →
      RepAgree n (FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val))
        (_root_.Flock.Setup.ofCircuit (stAt x j)) (zkTab x j).sch e
        (repSrcCut PE x (czExe PE x c) (cut c) j r) (repSrcCut PE x (czExe PE x c') (cut c') j r)) :
    ∀ schs, ((FlockDeploy.tabsOf (stsOf x)).mapM fun st =>
          (monadLift (_root_.Flock.Zk.schedule st.m) : _root_.Flock.V (_root_.Flock.Schedule × ℕ))) = .ok schs →
        ∀ k (hk : k < spec.repStreams.size) i r w, spec.repStreams[k] = (i, r, w) → ∀ extras,
          (FlockDeploy.tabsOf (stsOf x))[i]!.extra (ptsOf (fun k i => slotAt c k i) spec) = .ok extras →
          SchOk (FlockDeploy.tabsOf (stsOf x))[i]! schs[i]!.1 ∧
            FlockComplete.repStartZk spec (stsOf x) (k + 1) = FlockComplete.repStartZk spec (stsOf x) k +
              repSends (FlockDeploy.tabsOf (stsOf x))[i]! schs[i]!.1 extras ∧
            RepAgree n (FlockComplete.repStartZk spec (stsOf x) k) (FlockDeploy.tabsOf (stsOf x))[i]! schs[i]!.1
              extras ((tabSrcCut PE x (czExe PE x c) (cut c) i).rep r)
              ((tabSrcCut PE x (czExe PE x c') (cut c') i).rep r) := by
  intro schs hschs k hk i r w hkw extras hext
  obtain ⟨-, -, hspec, -⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  rw [FlockDeploy.repStreams_getElem spec hk] at hkw
  simp only [Prod.mk.injEq] at hkw
  obtain ⟨rfl, rfl, -⟩ := hkw
  rw [FlockDeploy.repStreams_size] at hk
  have hj : k / 2 < jOf x := by unfold jOf; rw [if_pos hx]; omega
  have hr : k % 2 < 2 := Nat.mod_lt _ (by norm_num)
  have hst : (stsOf x)[k / 2]! = stAt x ⟨k / 2, hj⟩ := getElem!_pos (stsOf x) (k / 2) (lt_of_jOf ⟨k / 2, hj⟩)
  have htab : (FlockDeploy.tabsOf (stsOf x))[k / 2]! = _root_.Flock.Setup.ofCircuit (stAt x ⟨k / 2, hj⟩) := by
    rw [← hst]
    unfold FlockDeploy.tabsOf
    rw [getElem!_pos _ (k / 2) (by rw [Array.size_map]; exact lt_of_jOf ⟨k / 2, hj⟩),
      getElem!_pos _ (k / 2) (lt_of_jOf ⟨k / 2, hj⟩), Array.getElem_map]
  have hsch : schs[k / 2]!.1 = (zkTab x ⟨k / 2, hj⟩).sch := by
    obtain ⟨hsz, hget⟩ := FlockSoundness.Discharge.Exec.array_mapM_ok_getElem hschs
    have hlt : k / 2 < (FlockDeploy.tabsOf (stsOf x)).size := by
      rw [FlockDeploy.tabsOf, Array.size_map]; exact lt_of_jOf ⟨k / 2, hj⟩
    have h1 := hget (k / 2) hlt (hsz ▸ hlt)
    rw [← getElem!_pos (FlockDeploy.tabsOf (stsOf x)) (k / 2) hlt, htab] at h1
    have h2 : (_root_.Flock.Zk.schedule (stAt x ⟨k / 2, hj⟩).m : _root_.Flock.V _) = .ok schs[k / 2] := h1
    rw [(zkTab x ⟨k / 2, hj⟩).hz] at h2
    rw [getElem!_pos schs (k / 2) (hsz ▸ hlt), ← Except.ok.inj h2]
  rw [htab] at hext
  have hes : extras.size = (stAt x ⟨k / 2, hj⟩).regions.size * spec.points := by
    rw [FlockDeploy.extraClaims_size (st := stAt x ⟨k / 2, hj⟩) hext]
    simp
  rw [htab, hsch, tabSrcCut_rep _ _ hj hr, tabSrcCut_rep _ _ hj hr]
  refine ⟨schOk_of _, ?_, ?_⟩
  · have hk6 := (zkTab x ⟨k / 2, hj⟩).wf.1.hk6
    have hm := (FlockSoundness.Discharge.ZkExec.zk_schedule_seams _ _ _ (zkTab x ⟨k / 2, hj⟩).hz).1
    rw [FlockComplete.ZkTables.repStart_succ, hst]
    unfold FlockDeploy.repRounds
    simp only [↓reduceIte, (zkTab x ⟨k / 2, hj⟩).hz]
    simp only [repSends, innerAt]
    show _ = _ + (3 + ((stAt x ⟨k / 2, hj⟩).m - 6) + (3 + ((stAt x ⟨k / 2, hj⟩).c.kLog - 6)) + _ + 1 + _ + 4)
    rw [hes.trans (Nat.mul_comm _ _)]
    omega
  · have := hR ⟨k / 2, hj⟩ ⟨k % 2, hr⟩ extras hes
    rwa [show 2 * (⟨k / 2, hj⟩ : Fin (jOf x)).val + (⟨k % 2, hr⟩ : Fin 2).val = k by simp only; omega] at this

/-- **The deployed round encodings of cuts agree at coins agreeing below the round** where the cuts' level-0 and pads
leaves agree and, at a live statement, each rep's sources agree where read. -/
theorem encExe_congr {n : ℕ} (hcc : ∀ k : Fin (refK x), k.val < n → c k = c' k)
    (hl : ∀ j, leafAt PE x (czExe PE x c) (cut c) j = leafAt PE x (czExe PE x c') (cut c') j)
    (hpl : ∀ j r, padLeafCut PE x (czExe PE x c) (cut c) j r = padLeafCut PE x (czExe PE x c') (cut c') j r)
    (hR : Live x → ∀ s spec, FlockDeploy.setupOf true x = some (s, spec, stsOf x) → ∀ (j : Fin (jOf x)) (r : Fin 2)
      (e : Array _root_.Flock.Claim), e.size = (stAt x j).regions.size * spec.points →
      RepAgree n (FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val))
        (_root_.Flock.Setup.ofCircuit (stAt x j)) (zkTab x j).sch e
        (repSrcCut PE x (czExe PE x c) (cut c) j r) (repSrcCut PE x (czExe PE x c') (cut c') j r)) :
    encExe PE x c n (cut c) = encExe PE x c' n (cut c') := by
  unfold encExe
  by_cases hn : n = 0
  · simp only [hn, ↓reduceIte]
    unfold enc0
    simp only [hl]
  rw [if_neg hn, if_neg hn]
  rcases hsx : FlockDeploy.setupOf true x with _ | ⟨s, spec, sts⟩
  · rfl
  dsimp only
  have hs : stsOf x = sts := stsOf_of hsx
  have hrun : runCut x c n spec (tabSrcCut PE x (czExe PE x c) (cut c)) =
      runCut x c' n spec (tabSrcCut PE x (czExe PE x c') (cut c')) := by
    unfold runCut
    rw [FlockComplete.asksCut_congr (coin := fun k i => slotAt c' k i) (coin' := fun k i => slotAt c k i)
      (fun k hk => funext fun e => (slotAt_congr hcc hk e).symm)]
    by_cases hx : Live x
    · have hsx' : FlockDeploy.setupOf true x = some (s, spec, stsOf x) := by rw [hs]; exact hsx
      obtain ⟨hcm, hic⟩ := commitsCut_agree cut hl hpl
      exact sessionCut_eq _ n spec _ _ _ _ (by omega) hcm hic _ (by simp [FlockComplete.repStartZk])
        (repStreams_agree cut hx hsx' (hR hx s spec hsx'))
    · have hj : jOf x = 0 := by
        unfold jOf
        simp [hx]
      have hd : ∀ c₀ : Fin (refK x) → FlockDeploy.Coin,
          tabSrcCut PE x (czExe PE x c₀) (cut c₀) = fun _ => default :=
        fun c₀ => funext fun i => tabSrcCut_dead _ _ (by omega)
      rw [hd, hd]
  rw [hrun]

end Generic

/-- **The deployed plan is causal given each rep's sources agree where read** (`encExe_congr` at `cutC`). -/
theorem causalExe_of_rep (h : RepOk) : CausalExe := by
  intro x wt ω p c c' n hcc
  show encExe PE x c n (cutC x wt ω p c) = encExe PE x c' n (cutC x wt ω p c')
  exact encExe_congr (cutC x wt ω p) hcc (leaf0_agree x wt ω p c c') (padLeaf_agree x wt ω p c c')
    fun hx s spec hsx => h x hx wt ω p c c' n hcc s spec hsx

/-! ## The pads layout's positions -/

section Layout

variable (m kl k0 : ℕ)

theorem pl_r1ab (i : ℕ) : (_root_.Flock.Zk.PadLayout.new m kl k0).r1ab i = i := rfl
theorem pl_r1c (i : ℕ) : (_root_.Flock.Zk.PadLayout.new m kl k0).r1c i = 64 + i := rfl
theorem pl_ml (i t : ℕ) : (_root_.Flock.Zk.PadLayout.new m kl k0).ml i t = 128 + 2 * i + t := rfl
theorem pl_fa : (_root_.Flock.Zk.PadLayout.new m kl k0).fa = 128 + 2 * (m - 6) := rfl
theorem pl_fb : (_root_.Flock.Zk.PadLayout.new m kl k0).fb = 129 + 2 * (m - 6) := by
  show 128 + 2 * (m - 6) + 1 = _
  omega
theorem pl_fc : (_root_.Flock.Zk.PadLayout.new m kl k0).fc = 130 + 2 * (m - 6) := by
  show 128 + 2 * (m - 6) + 2 = _
  omega
theorem pl_lc (i t : ℕ) : (_root_.Flock.Zk.PadLayout.new m kl k0).lc i t = 131 + 2 * (m - 6) + 2 * i + t := by
  show 128 + 2 * (m - 6) + 2 + 1 + 2 * i + t = _
  omega
theorem pl_zp (i : ℕ) : (_root_.Flock.Zk.PadLayout.new m kl k0).zp i = 131 + 2 * (m - 6) + 2 * (kl - 6) + i := by
  show 128 + 2 * (m - 6) + 2 + 1 + 2 * (kl - 6) + 0 + i = _
  omega
theorem pl_lgOod (i : ℕ) :
    (_root_.Flock.Zk.PadLayout.new m kl k0).lgOod i = 195 + 2 * (m - 6) + 2 * (kl - 6) + i := by
  show 128 + 2 * (m - 6) + 2 + 1 + 2 * (kl - 6) + 0 + 64 + i = _
  omega
theorem pl_lgMsg (i t : ℕ) :
    (_root_.Flock.Zk.PadLayout.new m kl k0).lgMsg i t = 196 + 2 * (m - 6) + 2 * (kl - 6) + 4 * i + t := by
  show 128 + 2 * (m - 6) + 2 + 1 + 2 * (kl - 6) + 0 + 64 + 1 + 4 * i + t = _
  omega
theorem pl_lgE (i t : ℕ) :
    (_root_.Flock.Zk.PadLayout.new m kl k0).lgE i t = 196 + 2 * (m - 6) + 2 * (kl - 6) + 4 * k0 + 2 * i + t := by
  show 128 + 2 * (m - 6) + 2 + 1 + 2 * (kl - 6) + 0 + 64 + 1 + 4 * k0 + 2 * i + t = _
  omega

end Layout

/-! ## The cut's fields -/

/-- A rep's sent `y₁` (`repSrcCut`'s `y1`). -/
def y1Arr {L : ℕ} : Option (Fin L → 𝕂) → Array _root_.Flock.F256
  | some f => Array.ofFn fun c => kVal (f c)
  | none => #[]

/-- A rep's sent `Y` (`repSrcCut`'s `y`). -/
def yArr {nE nPad : ℕ} (Y2 : Fin nPad → 𝔽) : Option (PadIdx nE → 𝔽) → Array _root_.Flock.F128
  | some Y1 => Array.ofFn (yAt Y1 Y2)
  | none => #[]

section CutFields

variable {J : ℕ} (C : Fin J → Composed.CoinedSetup) (cz : ∀ j, Composed.CoinsZK (C j).Rest)
  (wt : ∀ j, FlockSoundness.Model.Witness (C j).S.m) (ω : ∀ j, ZkView.Base (C j).z (C j).Hi)
  (lv : ZkView.Leaves (fun j => (C j).z) → ByteArray) (j : Fin J) (r : Fin 2)

theorem cutAt_masked (i) : (cutAt C cz wt ω lv j).masked r i =
    ((C j).T (cz j)).tab.vals r (wt j) (ω j).1 (ω j).2.1 i + ((ω j).2.2.1 r).1 i := rfl

theorem cutAt_sReg : (cutAt C cz wt ω lv j).sReg = ((C j).T (cz j)).tab.sReg (wt j) := rfl

open Classical in
theorem cutAt_Y1 : (cutAt C cz wt ω lv j).Y1 r =
    ((if (famSent (fun j => ((C j).T (cz j)).tab) (fun j => ((C j).T (cz j)).rep) j).fin r then
      some (fun i : PadIdx (C j).z.nE =>
        ((C j).T (cz j)).tab.hFull ((ω j).2.2.1 r) i + ((C j).T (cz j)).β r * (ω j).2.2.2.2.2.1 r i)
    else none) : Option (PadIdx (C j).z.nE → 𝔽)) := rfl

theorem cutAt_Y2 : (cutAt C cz wt ω lv j).Y2 r =
    fun i : Fin (C j).z.nPad => (ω j).2.2.2.1 r i + ((C j).T (cz j)).β r * (ω j).2.2.2.2.1 r i := rfl

open Classical in
theorem cutAt_y1 : (cutAt C cz wt ω lv j).y1 r =
    if (famSent (fun j => ((C j).T (cz j)).tab) (fun j => ((C j).T (cz j)).rep) j).y1 r then
      some (((C j).T (cz j)).tab.y1' (wt j) (ω j).1 (ω j).2.1.2.1 r)
    else none := rfl

theorem cutAt_ybar : (cutAt C cz wt ω lv j).ybar r = ((C j).T (cz j)).tab.ybar (ω j).2.1.1 (ω j).2.1.2.2 r := rfl

open Classical in
theorem cutAt_lig : (cutAt C cz wt ω lv j).lig r =
    if (famSent (fun j => ((C j).T (cz j)).tab) (fun j => ((C j).T (cz j)).rep) j).y1 r then
      some (((C j).T (cz j)).tab.lig r (((C j).T (cz j)).tab.y1' (wt j) (ω j).1 (ω j).2.1.2.1 r)
        ((ω j).2.2.2.2.2.2.2.2.2 r))
    else none := rfl

variable (cz' : ∀ j, Composed.CoinsZK (C j).Rest) (lv' : ZkView.Leaves (fun j => (C j).z) → ByteArray)

theorem cutAt_inner_congr (hζ : ((C j).T (cz j)).ζ r = ((C j).T (cz' j)).ζ r)
    (ha : ((C j).T (cz j)).ia = ((C j).T (cz' j)).ia) (hb : ((C j).T (cz j)).ib = ((C j).T (cz' j)).ib) :
    (cutAt C cz wt ω lv j).inner r = (cutAt C cz' wt ω lv' j).inner r := by
  show (((C j).T (cz j)).ζ r * ((ω j).2.2.1 r).1 ((C j).T (cz j)).ia + ((ω j).2.2.1 r).2.1,
      ((ω j).2.2.1 r).1 ((C j).T (cz j)).ib + ((ω j).2.2.1 r).2.2) =
    (((C j).T (cz' j)).ζ r * ((ω j).2.2.1 r).1 ((C j).T (cz' j)).ia + ((ω j).2.2.1 r).2.1,
      ((ω j).2.2.1 r).1 ((C j).T (cz' j)).ib + ((ω j).2.2.1 r).2.2)
  rw [hζ, ha, hb]

theorem hFull_congr (ha : ((C j).T (cz j)).ia = ((C j).T (cz' j)).ia)
    (hb : ((C j).T (cz j)).ib = ((C j).T (cz' j)).ib) (h : (Fin (C j).z.nE → 𝔽) × 𝔽 × 𝔽) :
    ((C j).T (cz j)).tab.hFull h = ((C j).T (cz' j)).tab.hFull h := by
  funext i
  rcases i with i | i
  · rfl
  · show ![h.2.1, h.2.2, h.1 ((C j).T (cz j)).ia * h.1 ((C j).T (cz j)).ib, h.2.1 * h.2.2] i =
      ![h.2.1, h.2.2, h.1 ((C j).T (cz' j)).ia * h.1 ((C j).T (cz' j)).ib, h.2.1 * h.2.2] i
    rw [ha, hb]

theorem cutAt_y_congr
    (hfin : (famSent (fun j => ((C j).T (cz j)).tab) (fun j => ((C j).T (cz j)).rep) j).fin r ↔
      (famSent (fun j => ((C j).T (cz' j)).tab) (fun j => ((C j).T (cz' j)).rep) j).fin r)
    (hβ : ((C j).T (cz j)).β r = ((C j).T (cz' j)).β r)
    (ha : ((C j).T (cz j)).ia = ((C j).T (cz' j)).ia) (hb : ((C j).T (cz j)).ib = ((C j).T (cz' j)).ib) :
    yArr (nE := (C j).z.nE) (nPad := (C j).z.nPad) ((cutAt C cz wt ω lv j).Y2 r) ((cutAt C cz wt ω lv j).Y1 r) =
      yArr (nE := (C j).z.nE) (nPad := (C j).z.nPad) ((cutAt C cz' wt ω lv' j).Y2 r)
        ((cutAt C cz' wt ω lv' j).Y1 r) := by
  rw [cutAt_Y1, cutAt_Y1, cutAt_Y2, cutAt_Y2]
  simp only [hβ]
  split_ifs with h1 h2 h2
  · rw [hFull_congr (C := C) (cz := cz) (cz' := cz') (j := j) ha hb]
  · exact absurd (hfin.1 h1) h2
  · exact absurd (hfin.2 h2) h1
  · rfl

theorem y1'_congr (hrbar : ((C j).T (cz j)).rbar r = ((C j).T (cz' j)).rbar r)
    (hρ : ((C j).T (cz j)).ρ r = ((C j).T (cz' j)).ρ r) (he : ((C j).T (cz j)).embed = ((C j).T (cz' j)).embed)
    (hl : ((C j).T (cz j)).lanes = ((C j).T (cz' j)).lanes) :
    ((C j).T (cz j)).tab.y1' (wt j) (ω j).1 (ω j).2.1.2.1 r = ((C j).T (cz' j)).tab.y1' (wt j) (ω j).1 (ω j).2.1.2.1 r := by
  funext col
  show (∑ l, ((C j).T (cz j)).rbar r l * ((C j).T (cz j)).embed (((C j).T (cz j)).lanes (wt j) (ω j).1 l col)) +
      (((C j).T (cz j)).ρ r 0 * ((C j).T (cz j)).embed ((ω j).2.1.2.1 r 0 col) +
        ((C j).T (cz j)).ρ r 1 * ((C j).T (cz j)).embed ((ω j).2.1.2.1 r 1 col)) =
    (∑ l, ((C j).T (cz' j)).rbar r l * ((C j).T (cz' j)).embed (((C j).T (cz' j)).lanes (wt j) (ω j).1 l col)) +
      (((C j).T (cz' j)).ρ r 0 * ((C j).T (cz' j)).embed ((ω j).2.1.2.1 r 0 col) +
        ((C j).T (cz' j)).ρ r 1 * ((C j).T (cz' j)).embed ((ω j).2.1.2.1 r 1 col))
  rw [hrbar, hρ, he, hl]

theorem cutAt_y1_congr
    (hy : (famSent (fun j => ((C j).T (cz j)).tab) (fun j => ((C j).T (cz j)).rep) j).y1 r ↔
      (famSent (fun j => ((C j).T (cz' j)).tab) (fun j => ((C j).T (cz' j)).rep) j).y1 r)
    (hrbar : ((C j).T (cz j)).rbar r = ((C j).T (cz' j)).rbar r)
    (hρ : ((C j).T (cz j)).ρ r = ((C j).T (cz' j)).ρ r) (he : ((C j).T (cz j)).embed = ((C j).T (cz' j)).embed)
    (hl : ((C j).T (cz j)).lanes = ((C j).T (cz' j)).lanes) :
    y1Arr (L := (C j).z.L) ((cutAt C cz wt ω lv j).y1 r) = y1Arr (L := (C j).z.L) ((cutAt C cz' wt ω lv' j).y1 r) := by
  rw [cutAt_y1, cutAt_y1]
  split_ifs with h1 h2 h2
  · rw [y1'_congr C cz wt ω j r cz' hrbar hρ he hl]
    exact rfl
  · exact absurd (hy.1 h1) h2
  · exact absurd (hy.2 h2) h1
  · rfl

theorem cutAt_ybar_congr (hrbar : ((C j).T (cz j)).rbar r = ((C j).T (cz' j)).rbar r)
    (hρ : ((C j).T (cz j)).ρ r = ((C j).T (cz' j)).ρ r) (he : ((C j).T (cz j)).embed = ((C j).T (cz' j)).embed) :
    (Array.ofFn fun i => kVal ((cutAt C cz wt ω lv j).ybar r i)) =
      Array.ofFn fun i => kVal ((cutAt C cz' wt ω lv' j).ybar r i) := by
  rw [cutAt_ybar, cutAt_ybar]
  show (Array.ofFn fun i => kVal ((∑ l, ((C j).T (cz j)).rbar r l * ((C j).T (cz j)).embed ((ω j).2.1.1 l i)) +
      (((C j).T (cz j)).ρ r 0 * ((C j).T (cz j)).embed ((ω j).2.1.2.2 r 0 i) +
        ((C j).T (cz j)).ρ r 1 * ((C j).T (cz j)).embed ((ω j).2.1.2.2 r 1 i)))) =
    Array.ofFn fun i => kVal ((∑ l, ((C j).T (cz' j)).rbar r l * ((C j).T (cz' j)).embed ((ω j).2.1.1 l i)) +
      (((C j).T (cz' j)).ρ r 0 * ((C j).T (cz' j)).embed ((ω j).2.1.2.2 r 0 i) +
        ((C j).T (cz' j)).ρ r 1 * ((C j).T (cz' j)).embed ((ω j).2.1.2.2 r 1 i)))
  rw [hrbar, hρ, he]

/-- **A rep's `τ` commitment reads its `τ`** (at the executable's leaves). -/
theorem cutC_tauCom_congr (p : ZkView.Leaves (fun j => (C j).z) → ZkView.Bits × ByteArray)
    (htau : (((C j).T (cz j)).tab.view (wt j) (((C j).T (cz j)).randOf (ω j) 0)).tau r =
      (((C j).T (cz' j)).tab.view (wt j) (((C j).T (cz' j)).randOf (ω j) 0)).tau r) :
    (cutAt C cz wt ω (fun l => ZkView.leafH (digOf C cz wt ω l + (p l).1, (p l).2)) j).tauCom r =
      (cutAt C cz' wt ω (fun l => ZkView.leafH (digOf C cz' wt ω l + (p l).1, (p l).2)) j).tauCom r := by
  show ZkView.leafH (ZkView.bitsOf (_root_.Flock.Sha512.hash (_root_.Flock.Zk.TAU_TAG ++ _root_.Flock.F128.toBytes
        ((((C j).T (cz j)).tab.view (wt j) (((C j).T (cz j)).randOf (ω j) 0)).tau r))) + (p ⟨j, .inr (.inr r)⟩).1,
      (p ⟨j, .inr (.inr r)⟩).2) =
    ZkView.leafH (ZkView.bitsOf (_root_.Flock.Sha512.hash (_root_.Flock.Zk.TAU_TAG ++ _root_.Flock.F128.toBytes
        ((((C j).T (cz' j)).tab.view (wt j) (((C j).T (cz' j)).randOf (ω j) 0)).tau r))) + (p ⟨j, .inr (.inr r)⟩).1,
      (p ⟨j, .inr (.inr r)⟩).2)
  rw [htau]

end CutFields

section Masked

variable (x : Msg) (cz cz' : ∀ j, Composed.CoinsZK (tabC PE x j).Rest)
  (wt : ∀ j, FlockSoundness.Model.Witness (tabC PE x j).S.m) (ω : ∀ j, ZkView.Base (tabC PE x j).z (tabC PE x j).Hi)
  (lv lv' : ZkView.Leaves (fun j => (tabC PE x j).z) → ByteArray) (j : Fin (jOf x)) (r : Fin 2)

theorem repSrcCut_masked (cut : ∀ j, ((tabC PE x j).T (cz j)).tab.Cut) (k : ℕ) :
    (repSrcCut PE x cz cut j r).masked k =
      if h : k < (tabC PE x j).z.nE then FlockLevel3.GF128.val ((cut j).masked r ⟨k, h⟩) else 0 := by
  simp only [repSrcCut]
  rfl

theorem tabC_vals (c : Composed.CoinsZK (tabC PE x j).Rest) (w u V i) :
    ((tabC PE x j).T c).tab.vals r w u V i =
      Composed.honestAt (zkTab x j).table (stAt x j) c r (Composed.zWith (stAt x j) w u) (V.2.1 r) i.val := rfl

theorem cutTab_masked (i) : (cutAt (tabC PE x) cz wt ω lv j).masked r i =
    Composed.honestAt (zkTab x j).table (stAt x j) (cz j) r (Composed.zWith (stAt x j) (wt j) (ω j).1)
      ((ω j).2.1.2.1 r) i.val + ((ω j).2.2.1 r).1 i :=
  (cutAt_masked _ _ _ _ _ _ _ _).trans (congrArg (· + _) (tabC_vals x j r _ _ _ _ _))

theorem masked_congr (k : ℕ)
    (h : ∀ (zw : FlockSoundness.Model.Witness (zkTab x j).table.S.m) (R : Fin 2 → Fin (tabC PE x j).z.L → 𝔽),
      Composed.honestAt (zkTab x j).table (stAt x j) (cz j) r zw R k =
        Composed.honestAt (zkTab x j).table (stAt x j) (cz' j) r zw R k) :
    (repSrcCut PE x cz (cutAt (tabC PE x) cz wt ω lv) j r).masked k =
      (repSrcCut PE x cz' (cutAt (tabC PE x) cz' wt ω lv') j r).masked k := by
  rw [repSrcCut_masked, repSrcCut_masked]
  by_cases hk : k < (tabC PE x j).z.nE
  · rw [dite_eq_left hk, dite_eq_left hk, cutTab_masked, cutTab_masked, h]
    rfl
  · rw [dite_eq_right hk, dite_eq_right hk]

open Classical in
theorem cutTab_sHat (q) : (cutAt (tabC PE x) cz wt ω lv j).sHat q =
    if (famSent (fun j => ((tabC PE x j).T (cz j)).tab) (fun j => ((tabC PE x j).T (cz j)).rep) j).cl q then
      some (fun v => ((tabC PE x j).T (cz j)).tab.nu (wt j) q v +
        ∑ y, ((tabC PE x j).T (cz j)).tab.eqw q y * FlockSoundness.ZK.Table.castF ((ω j).1 y v))
    else none := rfl


theorem sv_congr (cut : ∀ j, ((tabC PE x j).T (cz j)).tab.Cut) (cut' : ∀ j, ((tabC PE x j).T (cz' j)).tab.Cut)
    (k : ℕ) (h1 : k < 2 → ∀ h : 2 * r.val + k < (tabC PE x j).z.nC,
      (cut j).sHat ⟨2 * r.val + k, h⟩ = (cut' j).sHat ⟨2 * r.val + k, h⟩)
    (h2 : ¬ k < 2 → ∀ h : k - 2 < (tabC PE x j).z.nR, (cut j).sReg ⟨k - 2, h⟩ = (cut' j).sReg ⟨k - 2, h⟩) :
    (repSrcCut PE x cz cut j r).sv k = (repSrcCut PE x cz' cut' j r).sv k := by
  simp only [repSrcCut]
  by_cases hk : k < 2
  · rw [ite_eq_left hk, ite_eq_left hk]
    by_cases hc : 2 * r.val + k < (tabC PE x j).z.nC
    · rw [dite_eq_left (show 2 * r.val + k < ((tabC PE x j).T (cz j)).tab.nC from hc),
        dite_eq_left (show 2 * r.val + k < ((tabC PE x j).T (cz' j)).tab.nC from hc), h1 hk hc]
    · rw [dite_eq_right (show ¬ 2 * r.val + k < ((tabC PE x j).T (cz j)).tab.nC from hc),
        dite_eq_right (show ¬ 2 * r.val + k < ((tabC PE x j).T (cz' j)).tab.nC from hc)]
  · rw [ite_eq_right hk, ite_eq_right hk]
    by_cases hc : k - 2 < (tabC PE x j).z.nR
    · rw [dite_eq_left (show k - 2 < ((tabC PE x j).T (cz j)).tab.nR from hc),
        dite_eq_left (show k - 2 < ((tabC PE x j).T (cz' j)).tab.nR from hc), h2 hk hc]
    · rw [dite_eq_right (show ¬ k - 2 < ((tabC PE x j).T (cz j)).tab.nR from hc),
        dite_eq_right (show ¬ k - 2 < ((tabC PE x j).T (cz' j)).tab.nR from hc)]

theorem tabC_lig_snd (c : Composed.CoinsZK (tabC PE x j).Rest) (y hi) :
    (((tabC PE x j).T c).tab.lig r y hi : Composed.VLOf).2 =
      saltBlob (zkTab x j).table.sch.levels (hiSalt (zkTab x j).table.sch.levels hi) := rfl

theorem hiSalt_congr (cut : ∀ j, ((tabC PE x j).T (cz j)).tab.Cut) (cut' : ∀ j, ((tabC PE x j).T (cz' j)).tab.Cut)
    (h : (((cut j).lig r : Option Composed.VLOf)).map Prod.snd = (((cut' j).lig r : Option Composed.VLOf)).map Prod.snd) :
    (repSrcCut PE x cz cut j r).hiSalt = (repSrcCut PE x cz' cut' j r).hiSalt := by
  show blobSalt (zkTab x j).table.sch.levels
      ((((cut j).lig r : Option Composed.VLOf)).map Prod.snd |>.getD ByteArray.empty) =
    blobSalt (zkTab x j).table.sch.levels ((((cut' j).lig r : Option Composed.VLOf)).map Prod.snd |>.getD ByteArray.empty)
  rw [h]

theorem cutTab_lig_congr
    (hy : (famSent (fun j => ((tabC PE x j).T (cz j)).tab) (fun j => ((tabC PE x j).T (cz j)).rep) j).y1 r ↔
      (famSent (fun j => ((tabC PE x j).T (cz' j)).tab) (fun j => ((tabC PE x j).T (cz' j)).rep) j).y1 r) :
    (((cutAt (tabC PE x) cz wt ω lv j).lig r : Option Composed.VLOf)).map Prod.snd =
      (((cutAt (tabC PE x) cz' wt ω lv' j).lig r : Option Composed.VLOf)).map Prod.snd := by
  rw [cutAt_lig, cutAt_lig]
  split_ifs with h1 h2 h2
  · exact congrArg some ((tabC_lig_snd x j r _ _ _).trans (tabC_lig_snd x j r _ _ _).symm)
  · exact absurd (hy.1 h1) h2
  · exact absurd (hy.2 h2) h1
  · rfl

theorem repSrcCut_tau (cut : ∀ j, ((tabC PE x j).T (cz j)).tab.Cut) :
    (repSrcCut PE x cz cut j r).tau = 0 ∧ (repSrcCut PE x cz cut j r).tauSalt = ByteArray.empty := ⟨rfl, rfl⟩

theorem repSrcCut_y1 (cut : ∀ j, ((tabC PE x j).T (cz j)).tab.Cut) :
    (repSrcCut PE x cz cut j r).y1 = y1Arr (L := (tabC PE x j).z.L) ((cut j).y1 r) := by
  simp only [repSrcCut]
  cases (cut j).y1 r <;> rfl

theorem repSrcCut_y (cut : ∀ j, ((tabC PE x j).T (cz j)).tab.Cut) :
    (repSrcCut PE x cz cut j r).y =
      yArr (nE := (tabC PE x j).z.nE) (nPad := (tabC PE x j).z.nPad) ((cut j).Y2 r) ((cut j).Y1 r) := by
  simp only [repSrcCut]
  cases (cut j).Y1 r <;> rfl

theorem rep_y1_congr
    (hy : (famSent (fun j => ((tabC PE x j).T (cz j)).tab) (fun j => ((tabC PE x j).T (cz j)).rep) j).y1 r ↔
      (famSent (fun j => ((tabC PE x j).T (cz' j)).tab) (fun j => ((tabC PE x j).T (cz' j)).rep) j).y1 r)
    (hrbar : ((tabC PE x j).T (cz j)).rbar r = ((tabC PE x j).T (cz' j)).rbar r)
    (hρ : ((tabC PE x j).T (cz j)).ρ r = ((tabC PE x j).T (cz' j)).ρ r)
    (he : ((tabC PE x j).T (cz j)).embed = ((tabC PE x j).T (cz' j)).embed)
    (hl : ((tabC PE x j).T (cz j)).lanes = ((tabC PE x j).T (cz' j)).lanes) :
    (repSrcCut PE x cz (cutAt (tabC PE x) cz wt ω lv) j r).y1 =
      (repSrcCut PE x cz' (cutAt (tabC PE x) cz' wt ω lv') j r).y1 :=
  (repSrcCut_y1 x cz j r _).trans
    ((cutAt_y1_congr (tabC PE x) cz wt ω lv j r cz' lv' hy hrbar hρ he hl).trans (repSrcCut_y1 x cz' j r _).symm)

theorem rep_ybar_congr (hrbar : ((tabC PE x j).T (cz j)).rbar r = ((tabC PE x j).T (cz' j)).rbar r)
    (hρ : ((tabC PE x j).T (cz j)).ρ r = ((tabC PE x j).T (cz' j)).ρ r)
    (he : ((tabC PE x j).T (cz j)).embed = ((tabC PE x j).T (cz' j)).embed) :
    (repSrcCut PE x cz (cutAt (tabC PE x) cz wt ω lv) j r).ybar =
      (repSrcCut PE x cz' (cutAt (tabC PE x) cz' wt ω lv') j r).ybar :=
  cutAt_ybar_congr (tabC PE x) cz wt ω lv j r cz' lv' hrbar hρ he

theorem rep_inner_congr (hζ : ((tabC PE x j).T (cz j)).ζ r = ((tabC PE x j).T (cz' j)).ζ r)
    (ha : ((tabC PE x j).T (cz j)).ia = ((tabC PE x j).T (cz' j)).ia)
    (hb : ((tabC PE x j).T (cz j)).ib = ((tabC PE x j).T (cz' j)).ib) :
    (repSrcCut PE x cz (cutAt (tabC PE x) cz wt ω lv) j r).rhoIn =
        (repSrcCut PE x cz' (cutAt (tabC PE x) cz' wt ω lv') j r).rhoIn ∧
      (repSrcCut PE x cz (cutAt (tabC PE x) cz wt ω lv) j r).sigmaIn =
        (repSrcCut PE x cz' (cutAt (tabC PE x) cz' wt ω lv') j r).sigmaIn := by
  have h := cutAt_inner_congr (tabC PE x) cz wt ω lv j r cz' lv' hζ ha hb
  constructor <;> (simp only [repSrcCut]; rw [h])

theorem rep_y_congr
    (hfin : (famSent (fun j => ((tabC PE x j).T (cz j)).tab) (fun j => ((tabC PE x j).T (cz j)).rep) j).fin r ↔
      (famSent (fun j => ((tabC PE x j).T (cz' j)).tab) (fun j => ((tabC PE x j).T (cz' j)).rep) j).fin r)
    (hβ : ((tabC PE x j).T (cz j)).β r = ((tabC PE x j).T (cz' j)).β r)
    (ha : ((tabC PE x j).T (cz j)).ia = ((tabC PE x j).T (cz' j)).ia)
    (hb : ((tabC PE x j).T (cz j)).ib = ((tabC PE x j).T (cz' j)).ib) :
    (repSrcCut PE x cz (cutAt (tabC PE x) cz wt ω lv) j r).y =
      (repSrcCut PE x cz' (cutAt (tabC PE x) cz' wt ω lv') j r).y :=
  (repSrcCut_y x cz j r _).trans
    ((cutAt_y_congr (tabC PE x) cz wt ω lv j r cz' lv' hfin hβ ha hb).trans (repSrcCut_y x cz' j r _).symm)

/-- **Rep `r`'s `τ` reads only its coins below the inner proof's `β`** and its own masked claims' points. -/
theorem view_tau_congr (hA : Composed.Cx.InAgree (zkTab x j).table (stAt x j) (cz j) (cz' j) r)
    (hpt : ∀ k (h : k < (tabC PE x j).z.nC), (k = 2 * r.val ∨ k = 2 * r.val + 1) →
      Composed.pointsOf ((cz j).rest : Composed.RestOf (zkTab x j).table (stAt x j)) ⟨k, h⟩ =
        Composed.pointsOf ((cz' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)) ⟨k, h⟩) :
    (((tabC PE x j).T (cz j)).tab.view (wt j) (((tabC PE x j).T (cz j)).randOf (ω j) 0)).tau r =
      (((tabC PE x j).T (cz' j)).tab.view (wt j) (((tabC PE x j).T (cz' j)).randOf (ω j) 0)).tau r := by
  have hrbar := Composed.Cx.rbar_congr PE (zkTab x j).table (stAt x j) (cz j) (cz' j) r hA.lane
  have hρ := Composed.Cx.ρ_congr PE (zkTab x j).table (stAt x j) (cz j) (cz' j) r hA.rhoP
  have he := Composed.Cx.embed_congr PE (zkTab x j).table (stAt x j) (cz j) (cz' j)
  have hla := Composed.Cx.lanes_congr PE (zkTab x j).table (stAt x j) (cz j) (cz' j)
  have hζ : ((tabC PE x j).T (cz j)).ζ r = ((tabC PE x j).T (cz' j)).ζ r :=
    Composed.Cx.ζ_congr PE (zkTab x j).table (stAt x j) (cz j) (cz' j) r hA.zeta
  have hia : ((tabC PE x j).T (cz j)).ia = ((tabC PE x j).T (cz' j)).ia :=
    Composed.Cx.ia_congr PE (zkTab x j).table (stAt x j) (cz j) (cz' j)
  have hib : ((tabC PE x j).T (cz j)).ib = ((tabC PE x j).T (cz' j)).ib :=
    Composed.Cx.ib_congr PE (zkTab x j).table (stAt x j) (cz j) (cz' j)
  refine Composed.Cx.coined_tau_congr ((tabC PE x j).T (cz j)) ((tabC PE x j).T (cz' j)) (wt j) (ω j) r
    (Composed.Cx.lamOf_congr (zkTab x j).table (stAt x j) (cz j) (cz' j) r _ _ hA ?_ ?_ ?_ ?_ ?_)
  · funext i
    show Composed.honestAt (zkTab x j).table (stAt x j) (cz j) r (Composed.zWith (stAt x j) (wt j) (ω j).1)
        ((ω j).2.1.2.1 r) i.val + ((ω j).2.2.1 r).1 i =
      Composed.honestAt (zkTab x j).table (stAt x j) (cz' j) r (Composed.zWith (stAt x j) (wt j) (ω j).1)
        ((ω j).2.1.2.1 r) i.val + ((ω j).2.2.1 r).1 i
    rw [Composed.Cx.honestAt_all _ _ _ _ r _ _ hA.toB0Agree hA.zc hA.lcAlpha hA.lcBeta hA.lane]
  · unfold Composed.sHatAt
    split_ifs with h
    · exact Composed.Cx.coined_val_congr ((tabC PE x j).T (cz j)) ((tabC PE x j).T (cz' j)) (wt j)
        ⟨2 * r.val, h⟩ (ω j).1 (Composed.Cx.nu_congr PE _ _ _ _ (wt j) ⟨2 * r.val, h⟩ (hpt _ h (.inl rfl)))
        (Composed.Cx.eqw_congr PE _ _ _ _ ⟨2 * r.val, h⟩ (hpt _ h (.inl rfl)))
    · rfl
  · unfold Composed.sHatAt
    split_ifs with h
    · exact Composed.Cx.coined_val_congr ((tabC PE x j).T (cz j)) ((tabC PE x j).T (cz' j)) (wt j)
        ⟨2 * r.val + 1, h⟩ (ω j).1 (Composed.Cx.nu_congr PE _ _ _ _ (wt j) ⟨2 * r.val + 1, h⟩ (hpt _ h (.inr rfl)))
        (Composed.Cx.eqw_congr PE _ _ _ _ ⟨2 * r.val + 1, h⟩ (hpt _ h (.inr rfl)))
    · rfl
  · show ∑ col : Fin (tabC PE x j).z.L, ((tabC PE x j).T (cz j)).tab.y1' (wt j) (ω j).1 (ω j).2.1.2.1 r col *
        Composed.bFold (zkTab x j).table (stAt x j) (cz j) r col.val =
      ∑ col : Fin (tabC PE x j).z.L, ((tabC PE x j).T (cz' j)).tab.y1' (wt j) (ω j).1 (ω j).2.1.2.1 r col *
        Composed.bFold (zkTab x j).table (stAt x j) (cz' j) r col.val
    rw [y1'_congr (tabC PE x) cz wt ω j r cz' hrbar hρ he hla,
      Composed.Cx.bFold_congr _ _ _ _ r hA.toB0Agree hA.lane]
  · show (((tabC PE x j).T (cz j)).ζ r * ((ω j).2.2.1 r).1 ((tabC PE x j).T (cz j)).ia + ((ω j).2.2.1 r).2.1,
        ((ω j).2.2.1 r).1 ((tabC PE x j).T (cz j)).ib + ((ω j).2.2.1 r).2.2) =
      (((tabC PE x j).T (cz' j)).ζ r * ((ω j).2.2.1 r).1 ((tabC PE x j).T (cz' j)).ia + ((ω j).2.2.1 r).2.1,
        ((ω j).2.2.1 r).1 ((tabC PE x j).T (cz' j)).ib + ((ω j).2.2.1 r).2.2)
    rw [hζ, hia, hib]

theorem rep_tauCom_congr (p : ZkView.Leaves (fun j => (tabC PE x j).z) → ZkView.Bits × ByteArray)
    (htau : (((tabC PE x j).T (cz j)).tab.view (wt j) (((tabC PE x j).T (cz j)).randOf (ω j) 0)).tau r =
      (((tabC PE x j).T (cz' j)).tab.view (wt j) (((tabC PE x j).T (cz' j)).randOf (ω j) 0)).tau r) :
    (repSrcCut PE x cz (cutAt (tabC PE x) cz wt ω fun l =>
        ZkView.leafH (digOf (tabC PE x) cz wt ω l + (p l).1, (p l).2)) j r).tauCom =
      (repSrcCut PE x cz' (cutAt (tabC PE x) cz' wt ω fun l =>
        ZkView.leafH (digOf (tabC PE x) cz' wt ω l + (p l).1, (p l).2)) j r).tauCom :=
  cutC_tauCom_congr (tabC PE x) cz wt ω j r cz' p htau

end Masked

/-! ## Each coin at its round -/

/-- **The rounds a check reads below**: table `j`'s rep `r`'s rank check, after its lincheck's last coin; its restart,
after the lane phase's pair; its inner check, after `β`. -/
noncomputable def gateAt (spec : _root_.Flock.SessionSpec) (x : Msg) (j : Fin (jOf x)) (r : Fin 2) :
    FlockSoundness.ZK.Session.Phase → ℕ
  | .opening => FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val) + 6 + ((stAt x j).m - 6) +
      ((stAt x j).c.kLog - 6)
  | .restart => FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val) +
      FlockComplete.piopRoundsZk spec (stAt x j) + (zkTab x j).sch.k0 + 4
  | .inner => FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val) +
      FlockComplete.piopRoundsZk spec (stAt x j) + innerOff 2 (zkTab x j).sch + 1

theorem rep_two (r : Fin 2) {k : ℕ} (hk : k < 2) : (if 2 * r.val + k < 2 then (0 : Fin 2) else 1) = r := by
  rcases r with ⟨_ | _ | _, hr⟩
  · simp only [Nat.mul_zero, Nat.zero_add, ite_eq_left hk]; rfl
  · simp only [ite_eq_right (show ¬ 2 + k < 2 by omega)]; rfl
  · omega

section Slots

variable {x : Msg} {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec}
  (hsx : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) (j : Fin (jOf x)) (r : Fin 2)

theorem cxStsOf_half : (stsOf x)[(2 * j.val + r.val) / 2]! = stAt x j := by
  rw [show (2 * j.val + r.val) / 2 = j.val by omega]
  exact getElem!_pos (stsOf x) j.val (lt_of_jOf j)

theorem repStart_next : FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val + 1) =
    FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val) + FlockComplete.piopRoundsZk spec (stAt x j) +
      innerOff 0 (zkTab x j).sch + 4 := by
  have hk6 : 6 ≤ (stAt x j).c.kLog := (zkTab x j).wf.1.hk6
  rw [FlockComplete.ZkTables.repStart_succ, cxStsOf_half]
  unfold FlockDeploy.repRounds FlockComplete.piopRoundsZk innerOff
  simp only [↓reduceIte, (zkTab x j).hz]
  omega

theorem gate_le_next (p : FlockSoundness.ZK.Session.Phase) :
    gateAt spec x j r p ≤ FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val + 1) := by
  have hk6 : 6 ≤ (stAt x j).c.kLog := (zkTab x j).wf.1.hk6
  have hm : 14 ≤ (stAt x j).m := (schOk_of (zkTab x j)).m
  rw [repStart_next]
  cases p <;> simp only [gateAt] <;> unfold FlockComplete.piopRoundsZk innerOff <;> omega

theorem gate_le_start (p : FlockSoundness.ZK.Session.Phase) :
    FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val) ≤ gateAt spec x j r p := by
  cases p <;> simp only [gateAt] <;> omega

theorem gate_mono {j' : Fin (jOf x)} {r' : Fin 2} {p p' : FlockSoundness.ZK.Session.Phase}
    (h : FlockSoundness.ZK.Session.pos j' r' p' ≤ FlockSoundness.ZK.Session.pos j r p) :
    gateAt spec x j' r' p' ≤ gateAt spec x j r p := by
  unfold FlockSoundness.ZK.Session.pos at h
  have hi : ∀ p : FlockSoundness.ZK.Session.Phase, p.idx ≤ 2 := fun p => by
    cases p <;> simp only [FlockSoundness.ZK.Session.Phase.idx] <;> omega
  by_cases hq : 2 * j'.val + r'.val < 2 * j.val + r.val
  · calc gateAt spec x j' r' p' ≤ FlockComplete.repStartZk spec (stsOf x) (2 * j'.val + r'.val + 1) :=
          gate_le_next j' r' p'
      _ ≤ FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val) :=
          FlockComplete.ZkTables.repStart_mono spec (stsOf x) (by omega)
      _ ≤ gateAt spec x j r p := gate_le_start j r p
  · have h1 := hi p
    have h2 := hi p'
    have hj : j' = j := Fin.ext (by omega)
    have hr : r' = r := Fin.ext (by omega)
    subst hj hr
    have hk6 : 6 ≤ (stAt x j').c.kLog := (zkTab x j').wf.1.hk6
    have hm : 14 ≤ (stAt x j').m := (schOk_of (zkTab x j')).m
    cases p <;> cases p' <;> simp only [FlockSoundness.ZK.Session.Phase.idx] at h <;>
      simp only [gateAt, FlockComplete.piopRoundsZk, innerOff] <;> omega

include hsx in
theorem repSlot_at (f : _root_.Flock.Schedule → ℕ) :
    repSlot x (2 * j.val + r.val) f = FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val) +
      FlockComplete.piopRoundsZk spec (stAt x j) + f (zkTab x j).sch := by
  rw [repSlot_eq hsx (sch := (zkTab x j).sch) (tPad := (zkTab x j).tPad)
    (by rw [cxStsOf_half]; exact (zkTab x j).hz) f, cxStsOf_half]

include hsx in
theorem outerSlot_at : outerSlot x j.val r.val = FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val) + 1 := by
  rw [outerSlot_eq hsx, repBase_eq hsx]

variable {c c' : Fin (refK x) → FlockDeploy.Coin} {n : ℕ} (hcc : ∀ k : Fin (refK x), k.val < n → c k = c' k)

include hcc

theorem outerAt_agree {j r : ℕ} (h : outerSlot x j r < n) (i : ℕ) : outerAt c j r i = outerAt c' j r i := by
  unfold outerAt
  split_ifs with hK
  · rw [hcc ⟨_, hK⟩ h]
  · rfl

theorem link_agree (h : 0 < n) : ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).link =
    ((czExe PE x c' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).link :=
  funext fun _ => funext fun _ => slotAt_congr hcc h _

theorem rOuter_agree (h : outerSlot x j.val r.val < n) :
    ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).rOuter r =
      ((czExe PE x c' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).rOuter r :=
  funext fun _ => outerAt_agree hcc h _

theorem zc_agree (h : repBase x (2 * j.val + r.val) + 2 < n) :
    ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).zc r =
      ((czExe PE x c' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).zc r :=
  slotAt_congr hcc h _

theorem rho_agree_lt {i : ℕ} (h : repBase x (2 * j.val + r.val) + 3 + i ≤ n) :
    ∀ a : Fin ((stAt x j).m - 6), a.val < i →
      ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).rho r a =
        ((czExe PE x c' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).rho r a :=
  fun _ ha => slotAt_congr hcc (by omega) _

theorem rho_agree (h : repBase x (2 * j.val + r.val) + 3 + ((stAt x j).m - 6) ≤ n) :
    ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).rho r =
      ((czExe PE x c' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).rho r :=
  funext fun a => rho_agree_lt j r hcc h a a.isLt

theorem lcAlpha_agree (h : repBase x (2 * j.val + r.val) + 3 + ((stAt x j).m - 6) < n) :
    ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).lcAlpha r =
      ((czExe PE x c' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).lcAlpha r :=
  slotAt_congr hcc (by omega) _

theorem lcBeta_agree (h : repBase x (2 * j.val + r.val) + 4 + ((stAt x j).m - 6) < n) :
    ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).lcBeta r =
      ((czExe PE x c' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).lcBeta r :=
  slotAt_congr hcc (by omega) _

theorem ts_agree_lt {i : ℕ} (h : repBase x (2 * j.val + r.val) + 5 + ((stAt x j).m - 6) + i ≤ n) :
    ∀ a : Fin ((stAt x j).c.kLog - 6), a.val < i →
      ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).ts r a =
        ((czExe PE x c' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).ts r a :=
  fun _ ha => slotAt_congr hcc (by omega) _

theorem ts_agree (h : repBase x (2 * j.val + r.val) + 5 + ((stAt x j).m - 6) + ((stAt x j).c.kLog - 6) ≤ n) :
    ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).ts r =
      ((czExe PE x c' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).ts r :=
  funext fun a => ts_agree_lt j r hcc h a a.isLt

theorem rPrime_agree (h : repBase x (2 * j.val + r.val) + 5 + ((stAt x j).m - 6) + ((stAt x j).c.kLog - 6) < n) :
    ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).rPrime r =
      ((czExe PE x c' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).rPrime r :=
  slotAt_congr hcc (by omega) _

theorem rpp_agree (h : repBase x (2 * j.val + r.val) + 6 + ((stAt x j).m - 6) + ((stAt x j).c.kLog - 6) +
    (2 + 2 * (stAt x j).regions.size) ≤ n) :
    ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).rpp r =
      ((czExe PE x c' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).rpp r :=
  funext fun k => funext fun _ => slotAt_congr hcc (by have := k.isLt; omega) _

theorem kAt_agree {k : ℕ} (h : k < n) : kAt c k = kAt c' k := by
  unfold kAt
  rw [slotAt_congr hcc h, slotAt_congr hcc h]

include hsx

theorem rsGamma_agree (h : FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val) +
    FlockComplete.piopRoundsZk spec (stAt x j) - 1 < n) :
    ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).rsGamma r =
      ((czExe PE x c' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).rsGamma r :=
  funext fun _ => slotAt_congr hcc (by rw [repSlot_at hsx]; exact h) _

theorem oodPt_agree (h : FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val) +
    FlockComplete.piopRoundsZk spec (stAt x j) < n) :
    ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).oodPt r =
      ((czExe PE x c' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).oodPt r :=
  funext fun _ => funext fun _ => slotAt_congr hcc (by rw [repSlot_at hsx]; exact h) _

theorem ood_agree (h : FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val) +
    FlockComplete.piopRoundsZk spec (stAt x j) + 1 < n) :
    ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).ood r =
      ((czExe PE x c' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).ood r :=
  funext fun _ => slotAt_congr hcc (by rw [repSlot_at hsx]; exact h) _

theorem lane_agree_lt {i : ℕ} (h : FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val) +
    FlockComplete.piopRoundsZk spec (stAt x j) + 2 + i ≤ n) :
    ∀ a : Fin (zkTab x j).sch.k0, a.val < i →
      ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).lane r a =
        ((czExe PE x c' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).lane r a :=
  fun a ha => kAt_agree hcc (by rw [repSlot_at hsx]; omega)

theorem rhoPair_agree (h : FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val) +
    FlockComplete.piopRoundsZk spec (stAt x j) + (zkTab x j).sch.k0 + 3 < n) :
    (czExe PE x c j).rho r = (czExe PE x c' j).rho r := by
  have h0 : rhoSlot x j.val r.val < n := by unfold rhoSlot; rw [repSlot_at hsx]; omega
  show rhoAt c j.val r.val = rhoAt c' j.val r.val
  unfold rhoAt
  rw [slotAt_congr hcc h0, slotAt_congr hcc h0, slotAt_congr hcc (show rhoSlot x j.val r.val + 1 < n by
    unfold rhoSlot; rw [repSlot_at hsx]; omega), slotAt_congr hcc (show rhoSlot x j.val r.val + 1 < n by
    unfold rhoSlot; rw [repSlot_at hsx]; omega)]

theorem zeta_agree (h : FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val) +
    FlockComplete.piopRoundsZk spec (stAt x j) + innerOff 0 (zkTab x j).sch < n) :
    ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).zeta r =
      ((czExe PE x c' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).zeta r :=
  slotAt_congr hcc (by rw [repSlot_at hsx]; exact h) _

theorem gammas_agree (h : FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val) +
    FlockComplete.piopRoundsZk spec (stAt x j) + innerOff 1 (zkTab x j).sch < n) :
    ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).gammas r =
      ((czExe PE x c' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).gammas r :=
  funext fun _ => slotAt_congr hcc (by rw [repSlot_at hsx]; exact h) _

theorem beta_agree (h : FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val) +
    FlockComplete.piopRoundsZk spec (stAt x j) + innerOff 2 (zkTab x j).sch < n) :
    (czExe PE x c j).beta r = (czExe PE x c' j).beta r :=
  slotAt_congr hcc (by unfold betaSlot; rw [repSlot_at hsx]; unfold innerOff at h; omega) _

theorem claims_agree (h : FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val) + 6 + ((stAt x j).m - 6) +
      ((stAt x j).c.kLog - 6) ≤ n) :
    Composed.claimsOf ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)) r =
      Composed.claimsOf ((czExe PE x c' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)) r := by
  have hb := repBase_eq hsx (2 * j.val + r.val)
  have ho := outerSlot_at hsx j r
  exact Composed.Cx.claimsOf_congr (zkTab x j).table (stAt x j) (czExe PE x c j) (czExe PE x c' j) r
    (zc_agree j r hcc (by omega)) (rho_agree j r hcc (by omega)) (rOuter_agree j r hcc (by omega))
    (rPrime_agree j r hcc (by omega)) (ts_agree j r hcc (by omega))

theorem points_agree (k : Fin _root_.Flock.Zk.RANK_POINTS) (hk : (if k.val < 2 then (0 : Fin 2) else 1) ≤ r)
    (h : gateAt spec x j r .opening ≤ n) :
    Composed.pointsOf ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)) k =
      Composed.pointsOf ((czExe PE x c' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)) k := by
  have hle : (if k.val < 2 then (0 : Fin 2) else 1).val ≤ r.val := hk
  have hm := FlockComplete.ZkTables.repStart_mono spec (stsOf x)
    (show 2 * j.val + (if k.val < 2 then (0 : Fin 2) else 1).val ≤ 2 * j.val + r.val by omega)
  simp only [gateAt] at h
  unfold Composed.pointsOf
  rw [claims_agree hsx j _ hcc (by omega)]

theorem guard_agree (p : Session.Phase) (h : gateAt spec x j r p ≤ n) :
    famGuard (fun j => ((tabC PE x j).T (czExe PE x c j)).tab) (fun j => ((tabC PE x j).T (czExe PE x c j)).rep)
        j r p ↔
      famGuard (fun j => ((tabC PE x j).T (czExe PE x c' j)).tab) (fun j => ((tabC PE x j).T (czExe PE x c' j)).rep)
        j r p := by
  cases p with
  | opening =>
    exact Composed.Cx.rankOn_congr PE (zkTab x j).table (stAt x j) (czExe PE x c j) (czExe PE x c' j) r
      fun k hk => points_agree hsx j r hcc k hk h
  | restart =>
    simp only [gateAt] at h
    exact Composed.Cx.wr_congr PE (zkTab x j).table (stAt x j) (czExe PE x c j) (czExe PE x c' j) r
      (rhoPair_agree hsx j r hcc (by omega))
  | inner =>
    simp only [gateAt] at h
    exact Composed.Cx.beta_congr PE (zkTab x j).table (stAt x j) (czExe PE x c j) (czExe PE x c' j) r
      (beta_agree hsx j r hcc (by omega))

/-- **Whether the session reaches a check reads only the coins below its gate.** -/
theorem passes_agree (p : Session.Phase) (h : gateAt spec x j r p ≤ n) :
    famPasses (fun j => ((tabC PE x j).T (czExe PE x c j)).tab) (fun j => ((tabC PE x j).T (czExe PE x c j)).rep)
        j r p ↔
      famPasses (fun j => ((tabC PE x j).T (czExe PE x c' j)).tab) (fun j => ((tabC PE x j).T (czExe PE x c' j)).rep)
        j r p :=
  forall_congr' fun j' => forall_congr' fun r' => forall_congr' fun p' => imp_congr_right fun hp =>
    guard_agree hsx j' r' hcc p' (le_trans (gate_mono j r hp) h)

theorem cl_agree (q : Fin (tabC PE x j).z.nC) (hq : (if q.val < 2 then (0 : Fin 2) else 1) = r)
    (h : gateAt spec x j r .opening ≤ n) :
    (famSent (fun j => ((tabC PE x j).T (czExe PE x c j)).tab) (fun j => ((tabC PE x j).T (czExe PE x c j)).rep)
        j).cl q ↔
      (famSent (fun j => ((tabC PE x j).T (czExe PE x c' j)).tab)
        (fun j => ((tabC PE x j).T (czExe PE x c' j)).rep) j).cl q :=
  (Iff.of_eq (congrArg (fun r' => famPasses (fun j => ((tabC PE x j).T (czExe PE x c j)).tab)
      (fun j => ((tabC PE x j).T (czExe PE x c j)).rep) j r' .opening) hq)).trans
    ((passes_agree hsx j r hcc .opening h).trans
      (Iff.of_eq (congrArg (fun r' => famPasses (fun j => ((tabC PE x j).T (czExe PE x c' j)).tab)
        (fun j => ((tabC PE x j).T (czExe PE x c' j)).rep) j r' .opening) hq)).symm)

end Slots

theorem repOk : RepOk := by
  intro x hx wt ω p c c' n hcc s spec hsx j r e he
  have hp2 := FlockComplete.ZkTables.points_of_setup hx.1 hsx
  have hm14 : 14 ≤ (stAt x j).m := (schOk_of (zkTab x j)).m
  have hk6 : 6 ≤ (stAt x j).c.kLog := (zkTab x j).wf.1.hk6
  have hbase := repBase_eq hsx (2 * j.val + r.val)
  have hout := outerSlot_at hsx j r
  have hlig : FlockComplete.CutEq.ligAt (FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val))
      (_root_.Flock.Setup.ofCircuit (stAt x j)) e = FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val) +
        FlockComplete.piopRoundsZk spec (stAt x j) := by
    unfold FlockComplete.CutEq.ligAt FlockComplete.piopRoundsZk
    rw [he, hp2]
    show _ + 3 + ((stAt x j).m - 6) + (3 + ((stAt x j).c.kLog - 6)) + _ + 1 = _
    omega
  generalize hB : FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val) = B at hbase hout hlig ⊢
  have em : (_root_.Flock.Setup.ofCircuit (stAt x j)).m = (stAt x j).m := rfl
  have ef : (_root_.Flock.Setup.ofCircuit (stAt x j)).fold.kLog = (stAt x j).c.kLog := rfl
  have ek : (_root_.Flock.Setup.ofCircuit (stAt x j)).kLog = (stAt x j).c.kLog := rfl
  have hO : B + 2 ≤ n → ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).rOuter r =
      ((czExe PE x c' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).rOuter r := fun hn =>
    rOuter_agree j r hcc (by omega)
  have hz : B + 3 ≤ n → ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).zc r =
      ((czExe PE x c' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).zc r := fun hn =>
    zc_agree j r hcc (by omega)
  have hpiop : FlockComplete.piopRoundsZk spec (stAt x j) =
      (stAt x j).m + (stAt x j).c.kLog - 6 + (2 + 2 * (stAt x j).regions.size + 1) := by
    unfold FlockComplete.piopRoundsZk
    rw [hp2]
  have h0 : B + FlockComplete.piopRoundsZk spec (stAt x j) + 2 ≤ n →
      Composed.Cx.B0Agree (zkTab x j).table (stAt x j) (czExe PE x c j) (czExe PE x c' j) r := fun hn =>
    ⟨link_agree j hcc (by omega), hO (by omega), rho_agree j r hcc (by omega), ts_agree j r hcc (by omega),
      rpp_agree j r hcc (by omega), rsGamma_agree hsx j r hcc (by omega), oodPt_agree hsx j r hcc (by omega),
      ood_agree hsx j r hcc (by omega)⟩
  refine ⟨fun hn i hi => ?_, fun i hi hn => ?_, fun hn => ?_, fun i hi hn => ?_, fun hn i hi => ?_, ?_,
    fun hn => ?_, fun i hi hn t ht => ?_, fun hn a ha t ht => ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp only [layOf, pl_r1ab, pl_r1c]
    exact ⟨masked_congr x _ _ wt ω _ _ j r _ fun zw R =>
        Composed.Cx.honestAt_r1 _ _ _ _ r zw R (hO hn) (by omega),
      masked_congr x _ _ wt ω _ _ j r _ fun zw R =>
        Composed.Cx.honestAt_r1 _ _ _ _ r zw R (hO hn) (by omega)⟩
  · simp only [layOf, pl_ml, em] at hi ⊢
    exact ⟨masked_congr x _ _ wt ω _ _ j r _ fun zw R =>
        Composed.Cx.honestAt_ml _ _ _ _ r zw R (hO (by omega)) (hz (by omega))
          (rho_agree_lt j r hcc (by omega)) hi (by omega) (by omega),
      masked_congr x _ _ wt ω _ _ j r _ fun zw R =>
        Composed.Cx.honestAt_ml _ _ _ _ r zw R (hO (by omega)) (hz (by omega))
          (rho_agree_lt j r hcc (by omega)) hi (by omega) (by omega)⟩
  · simp only [layOf, pl_fa, pl_fb, pl_fc, em] at hn ⊢
    have hρ := rho_agree j r hcc (by omega)
    exact ⟨masked_congr x _ _ wt ω _ _ j r _ fun zw R =>
        Composed.Cx.honestAt_f _ _ _ _ r zw R (hO (by omega)) (hz (by omega)) hρ (by omega) (by omega),
      masked_congr x _ _ wt ω _ _ j r _ fun zw R =>
        Composed.Cx.honestAt_f _ _ _ _ r zw R (hO (by omega)) (hz (by omega)) hρ (by omega) (by omega),
      masked_congr x _ _ wt ω _ _ j r _ fun zw R =>
        Composed.Cx.honestAt_f _ _ _ _ r zw R (hO (by omega)) (hz (by omega)) hρ (by omega) (by omega)⟩
  · simp only [layOf, pl_lc, em, ek, ef] at hi hn ⊢
    have hρ := rho_agree j r hcc (by omega)
    have ha := lcAlpha_agree j r hcc (by omega)
    have hb := lcBeta_agree j r hcc (by omega)
    exact ⟨masked_congr x _ _ wt ω _ _ j r _ fun zw R =>
        Composed.Cx.honestAt_lc _ _ _ _ r zw R (hz (by omega)) ha hb hρ (ts_agree_lt j r hcc (by omega)) hi
          (by omega) (by omega),
      masked_congr x _ _ wt ω _ _ j r _ fun zw R =>
        Composed.Cx.honestAt_lc _ _ _ _ r zw R (hz (by omega)) ha hb hρ (ts_agree_lt j r hcc (by omega)) hi
          (by omega) (by omega)⟩
  · simp only [layOf, pl_zp, em, ek, ef] at hn ⊢
    exact masked_congr x _ _ wt ω _ _ j r _ fun zw R =>
      Composed.Cx.honestAt_zp _ _ _ _ r zw R (rho_agree j r hcc (by omega)) (ts_agree j r hcc (by omega))
        (by omega) (by omega)
  · intro k hk hn
    simp only [em, ef] at hn
    have hg : gateAt spec x j r .opening ≤ n := by simp only [gateAt]; rw [hB]; omega
    refine sv_congr x _ _ j r _ _ k (fun hk2 hc => ?_) (fun hk2 hc => ?_)
    · have hq := rep_two r hk2
      have hpt := points_agree hsx j r hcc ⟨2 * r.val + k, hc⟩ (le_of_eq hq) hg
      have hcl := cl_agree hsx j r hcc ⟨2 * r.val + k, hc⟩ hq hg
      rw [cutTab_sHat, cutTab_sHat]
      split_ifs with h1 h2 h2
      · exact congrArg some (Composed.Cx.coined_val_congr _ _ (wt j) ⟨2 * r.val + k, hc⟩ (ω j).1
          (Composed.Cx.nu_congr PE _ _ _ _ (wt j) ⟨2 * r.val + k, hc⟩ hpt)
          (Composed.Cx.eqw_congr PE _ _ _ _ ⟨2 * r.val + k, hc⟩ hpt))
      · exact absurd (hcl.1 h1) h2
      · exact absurd (hcl.2 h2) h1
      · rfl
    · rw [cutAt_sReg, cutAt_sReg]
      exact Composed.Cx.sReg_congr PE _ _ _ _ (wt j) ⟨k - 2, hc⟩ (link_agree j hcc (by omega))
  · simp only [layOf, pl_lgOod, em, ek] at hn ⊢
    exact masked_congr x _ _ wt ω _ _ j r _ fun zw R =>
      Composed.Cx.honestAt_ood _ _ _ _ r zw R (oodPt_agree hsx j r hcc (by omega)) (by omega)
  · simp only [layOf, pl_lgMsg, em, ek] at hn ⊢
    exact masked_congr x _ _ wt ω _ _ j r _ fun zw R =>
      Composed.Cx.honestAt_msg _ _ _ _ r zw R (h0 (by omega)) (lane_agree_lt hsx j r hcc (by omega)) hi
        (by omega) (by omega)
  · simp only [layOf, pl_lgE, em, ek] at hn ⊢
    have ek0 : (zkTab x j).table.sch.k0 = (zkTab x j).sch.k0 := rfl
    exact masked_congr x _ _ wt ω _ _ j r _ fun zw R =>
      Composed.Cx.honestAt_E _ _ _ _ r zw R (h0 (by omega))
        (funext fun a => lane_agree_lt hsx j r hcc (i := (zkTab x j).sch.k0) (by omega) a a.isLt) (by omega)
        (by omega)
  · intro hn
    have ek0 : (zkTab x j).table.sch.k0 = (zkTab x j).sch.k0 := rfl
    have hg : gateAt spec x j r .restart ≤ n := by simp only [gateAt]; rw [hB]; omega
    have hy := passes_agree hsx j r hcc .restart hg
    have hl : ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).lane r =
        ((czExe PE x c' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).lane r :=
      funext fun a => lane_agree_lt hsx j r hcc (i := (zkTab x j).sch.k0) (by omega) a a.isLt
    have hrbar := Composed.Cx.rbar_congr PE (zkTab x j).table (stAt x j) (czExe PE x c j) (czExe PE x c' j) r hl
    have hρ := Composed.Cx.ρ_congr PE (zkTab x j).table (stAt x j) (czExe PE x c j) (czExe PE x c' j) r
      (rhoPair_agree hsx j r hcc (by omega))
    have he := Composed.Cx.embed_congr PE (zkTab x j).table (stAt x j) (czExe PE x c j) (czExe PE x c' j)
    have hla := Composed.Cx.lanes_congr PE (zkTab x j).table (stAt x j) (czExe PE x c j) (czExe PE x c' j)
    exact ⟨rep_y1_congr x _ _ wt ω _ _ j r hy hrbar hρ he hla, rep_ybar_congr x _ _ wt ω _ _ j r hrbar hρ he,
      hiSalt_congr x _ _ j r _ _ (cutTab_lig_congr x _ _ wt ω _ _ j r hy)⟩
  · intro hn
    simp only [innerAt] at hn
    have hζ := zeta_agree hsx j r hcc (by simp only [innerOff]; omega)
    exact rep_inner_congr x _ _ wt ω _ _ j r
      (Composed.Cx.ζ_congr PE (zkTab x j).table (stAt x j) (czExe PE x c j) (czExe PE x c' j) r hζ)
      (Composed.Cx.ia_congr PE (zkTab x j).table (stAt x j) (czExe PE x c j) (czExe PE x c' j))
      (Composed.Cx.ib_congr PE (zkTab x j).table (stAt x j) (czExe PE x c j) (czExe PE x c' j))
  · intro hn
    simp only [innerAt] at hn
    have ek0 : (zkTab x j).table.sch.k0 = (zkTab x j).sch.k0 := rfl
    have hg : gateAt spec x j r .opening ≤ n := by simp only [gateAt]; rw [hB]; omega
    have hl : ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).lane r =
        ((czExe PE x c' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).lane r :=
      funext fun a => lane_agree_lt hsx j r hcc (i := (zkTab x j).sch.k0) (by omega) a a.isLt
    have hA : Composed.Cx.InAgree (zkTab x j).table (stAt x j) (czExe PE x c j) (czExe PE x c' j) r :=
      { toB0Agree := h0 (by omega)
        zc := hz (by omega)
        lcAlpha := lcAlpha_agree j r hcc (by omega)
        lcBeta := lcBeta_agree j r hcc (by omega)
        rPrime := rPrime_agree j r hcc (by omega)
        lane := hl
        rhoP := rhoPair_agree hsx j r hcc (by omega)
        zeta := zeta_agree hsx j r hcc (by simp only [innerOff]; omega)
        gammas := gammas_agree hsx j r hcc (by simp only [innerOff]; omega) }
    have hrep : ∀ k, (k = 2 * r.val ∨ k = 2 * r.val + 1) → (if k < 2 then (0 : Fin 2) else 1) ≤ r := by
      intro k hk
      rcases hk with rfl | rfl
      · exact le_of_eq (rep_two r (k := 0) (by omega))
      · exact le_of_eq (rep_two r (k := 1) (by omega))
    exact rep_tauCom_congr x _ _ wt ω j r p (view_tau_congr x _ _ wt ω j r hA fun k hk hk2 =>
      points_agree hsx j r hcc ⟨k, hk⟩ (hrep k hk2) hg)
  · intro hn
    simp only [innerAt] at hn
    have hg : gateAt spec x j r .inner ≤ n := by simp only [gateAt, innerOff]; rw [hB]; omega
    exact rep_y_congr x _ _ wt ω _ _ j r (passes_agree hsx j r hcc .inner hg)
      (Composed.Cx.β_congr PE (zkTab x j).table (stAt x j) (czExe PE x c j) (czExe PE x c' j) r
        (beta_agree hsx j r hcc (by simp only [innerOff]; omega)))
      (Composed.Cx.ia_congr PE (zkTab x j).table (stAt x j) (czExe PE x c j) (czExe PE x c' j))
      (Composed.Cx.ib_congr PE (zkTab x j).table (stAt x j) (czExe PE x c j) (czExe PE x c' j))
  · exact fun _ => ⟨(repSrcCut_tau x _ j r _).1.trans (repSrcCut_tau x _ j r _).1.symm,
      (repSrcCut_tau x _ j r _).2.trans (repSrcCut_tau x _ j r _).2.symm⟩

/-- **Sent's `hc`**: every value the executable sends reads only the coins flipped before it. -/
theorem causalExe : CausalExe := causalExe_of_rep repOk

end Claims.ZKMal.Masked
