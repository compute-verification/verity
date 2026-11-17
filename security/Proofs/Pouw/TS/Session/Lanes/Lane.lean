module

public import Definitions.Pouw.H1Zy.Chain
public import Definitions.Pouw.H1Zy.LaneBits
public import Proofs.Pouw.TS.Session.Lanes.Combine
public import Proofs.Pouw.TS.Session.Lanes.Decode
public import Proofs.Pouw.H1Zy.LaneBits
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
import Mathlib.RingTheory.WittVector.IsPoly
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
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init
import Proofs.Pouw.H1Zy.Lane
import Std.Tactic.BVDecide.Normalize.Bool

/-! A lane's part of `LeafChecks`: its Definition's `ok = 1` means `LaneOk` at every input and a `K` its tile binds at,
which leaves `LaneComputes`, equivalent to `LaneLowers`. -/

@[expose] public section

namespace ResourceAccounting.TS.Lanes

open ResourceAccounting.TS ResourceAccounting.TS.Checked Flock.SoundAtBridge PouwWindow.Lanes

/-- **A lane's inputs as words**, from a reader of its ports' row bits, each field at the bits `decLane` reads it from:
codes, scales, record words and partial words their fields' bits as unsigned numbers, the admit bits, the public inputs
as `decLane` reads them, and the index's 16 bits. -/
noncomputable def laneIn (K nw : ℕ) (T : Tile) (rd : String → ℕ → Option Bool) :
    Pouw.H1Zy.LaneIn K nw LanePub where
  a s k := BitVec.ofNat 8 (bitsN (rd s!"a{s.val}") (fieldBit (lanePorts T 0) s!"a[{s.val}].codes" + 8 * k.val) 8)
  sa s := BitVec.ofNat 32 (bitsN (rd s!"a{s.val}") (fieldBit (lanePorts T 0) s!"a[{s.val}].scale") 32)
  b t k := BitVec.ofNat 8 (bitsN (rd s!"b{t.val}") (fieldBit (lanePorts T 0) s!"b[{t.val}].codes" + 8 * k.val) 8)
  sw t := BitVec.ofNat 32 (bitsN (rd s!"b{t.val}") (fieldBit (lanePorts T 0) s!"b[{t.val}].scale") 32)
  admit i := rd "admit" (fieldBit (lanePorts T 0) "admit" + i.val) == some true
  zr i j := BitVec.ofNat 32 (bitsN (rd "zy") (fieldBit (lanePorts T 0) "zy.z" + 32 * (i.val * T.c + j.val)) 32)
  yr i j := BitVec.ofNat 16 (bitsN (rd "zy") (fieldBit (lanePorts T 0) "zy.y" + 16 * (i.val * T.c + j.val)) 16)
  partials w := BitVec.ofNat 32 (bitsN (rd "partials") (fieldBit (lanePorts T 0) "partials" + 32 * w.val) 32)
  pub := (decLane K nw T rd).pub
  lane t := rd "pub" (fieldBit (lanePorts T 0) "lane" + t.val) == some true

/-- **`LeafChecks`' `dq` at the dequant's word view** `dqW`: `dqW` on z's s32 word and the scales' f32 encodings, its
bf16 encoding as a number. -/
def dqOf (dqW : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 16) : List ℤ → ℤ
  | [z, sa, sw] => ((dqW (BitVec.ofInt 32 z) (BitVec.ofInt 32 sa) (BitVec.ofInt 32 sw)).toNat : ℤ)
  | _ => 0

/-- **`LeafChecks`' `part` at the lane's chain words** `chain`: at the public inputs `P`, lane `l` and its rows, its
chain's words on the rows' codes, each masked by the admit bits' OR (`PouwMask32`). -/
def partOf {K nw : ℕ}
    (chain : LanePub → Fin 32 → (Fin 4 → Fin K → BitVec 8) → (Fin 8 → Fin K → BitVec 8) → Fin nw → BitVec 32) :
    LanePub → Fin 32 → (Fin 4 → Row K) → (Fin 8 → Row K) → Fin nw → BitVec 32 :=
  fun P l a b w => Pouw.H1Zy.mask32
    (chain P l (fun s k => BitVec.ofInt 8 ((a s).codes k)) (fun t k => BitVec.ofInt 8 ((b t).codes k)) w)
    (Pouw.H1Zy.tree Pouw.H1Zy.orBit false (List.ofFn P.admit))

/-- **The lane's chain words** at contraction `K`: `Pouw.H1Zy.h1Chain K`'s words (`h1tile._chain`) on lane `l`'s rows'
codes, its seeds the salt, position and roots of the `pub` row `P`. -/
def laneChain (K nw : ℕ) :
    LanePub → Fin 32 → (Fin 4 → Fin K → BitVec 8) → (Fin 8 → Fin K → BitVec 8) → Fin nw → BitVec 32 :=
  fun P l a b => Pouw.H1Zy.chainOf (Pouw.H1Zy.h1Chain K nw) ⟨P.salt, P.position, P.roots⟩ l a b

theorem sgnBits_eq_toInt (m n : ℕ) (hm : 0 < m) (h : n < 2 ^ m) : sgnBits m n = (BitVec.ofNat m n).toInt := by
  obtain ⟨m, rfl⟩ : ∃ m', m = m' + 1 := ⟨m - 1, by omega⟩
  unfold sgnBits
  rw [BitVec.toInt_eq_toNat_cond, BitVec.toNat_ofNat, Nat.mod_eq_of_lt h]
  simp only [Nat.add_sub_cancel, pow_succ] at h ⊢
  by_cases hc : n < 2 ^ m
  · have h2 : 2 * n < 2 ^ m * 2 := by omega
    simp only [hc, h2, ↓reduceIte]
  · have h2 : ¬ 2 * n < 2 ^ m * 2 := by omega
    simp only [hc, h2, ↓reduceIte]
    push_cast
    ring

theorem bitsN_add (rd : ℕ → Option Bool) (lo m n : ℕ) :
    bitsN rd lo (m + n) = bitsN rd lo m + 2 ^ m * bitsN rd (lo + m) n := by
  unfold bitsN
  rw [Finset.sum_range_add, Finset.mul_sum]
  congr 1
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [← add_assoc, pow_add]
  split_ifs <;> simp

theorem bitsN_eq_bitsVal (rd : ℕ → Option Bool) : ∀ n lo,
    bitsN rd lo n = Pouw.H1Zy.bitsVal (List.ofFn fun t : Fin n => rd (lo + t.val) == some true)
  | 0, lo => by simp [bitsN, Pouw.H1Zy.bitsVal]
  | n + 1, lo => by
      rw [show bitsN rd lo (n + 1) = bitsN rd lo (1 + n) by rw [add_comm], bitsN_add,
        bitsN_eq_bitsVal rd n (lo + 1), List.ofFn_succ]
      simp only [Pouw.H1Zy.bitsVal, Fin.val_zero, add_zero]
      have e : (List.ofFn fun i : Fin n => rd (lo + (i.succ : Fin (n + 1)).val) == some true) =
          List.ofFn fun t : Fin n => rd (lo + 1 + t.val) == some true := by
        congr 1
        funext i
        rw [show lo + (i.succ : Fin (n + 1)).val = lo + 1 + i.val by simp only [Fin.val_succ]; omega]
      rw [e]
      congr 1
      simp only [bitsN, Finset.range_one, Finset.sum_singleton, add_zero, pow_zero]
      cases rd lo <;> simp

theorem decLane_lane {K nw : ℕ} (T : Tile) (rd : String → ℕ → Option Bool) :
    (decLane K nw T rd).lane = (laneIn K nw T rd).l := by
  apply Fin.ext
  have h16 := bitsN_add (rd "pub") (fieldBit (lanePorts T 0) "lane") 5 11
  have h5 := bitsN_lt (rd "pub") (fieldBit (lanePorts T 0) "lane") 5
  have hv := bitsN_eq_bitsVal (rd "pub") 5 (fieldBit (lanePorts T 0) "lane")
  show bitsN (rd "pub") (fieldBit (lanePorts T 0) "lane") laneBits % 32 = Pouw.H1Zy.bitsVal (laneIn K nw T rd).sel
  rw [show laneBits = 5 + 11 from rfl, h16]
  have : (bitsN (rd "pub") (fieldBit (lanePorts T 0) "lane") 5 + 2 ^ 5 * bitsN (rd "pub")
      (fieldBit (lanePorts T 0) "lane" + 5) 11) % 32 = bitsN (rd "pub") (fieldBit (lanePorts T 0) "lane") 5 := by
    omega
  rw [this, hv]
  rfl

theorem laneOk_of_def {K nw : ℕ} (T : Tile)
    (chain : LanePub → Fin 32 → (Fin 4 → Fin K → BitVec 8) → (Fin 8 → Fin K → BitVec 8) → Fin nw → BitVec 32)
    (dqW : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 16) (rd : String → ℕ → Option Bool) (hK32 : 32 ∣ K)
    (hK : K < 2 ^ 17) (h : Pouw.H1Zy.laneOk chain dqW (laneIn K nw T rd) = true) :
    LaneOk (dqOf dqW) (partOf chain) (decLane K nw T rd) := by
  obtain ⟨-, hz, hp⟩ := Pouw.H1Zy.laneOk_spec chain dqW _ hK32 hK h
  have hl := decLane_lane (K := K) (nw := nw) T rd
  have ha : ∀ s k, ((decLane K nw T rd).a s).codes k = ((laneIn K nw T rd).a s k).toInt := fun s k =>
    sgnBits_eq_toInt 8 _ (by norm_num) (bitsN_lt _ _ _)
  have hb : ∀ t k, ((decLane K nw T rd).b t).codes k = ((laneIn K nw T rd).b t k).toInt := fun t k =>
    sgnBits_eq_toInt 8 _ (by norm_num) (bitsN_lt _ _ _)
  have hsa : ∀ s, BitVec.ofInt 32 ((decLane K nw T rd).a s).scale = (laneIn K nw T rd).sa s := fun s =>
    BitVec.ofInt_natCast _ _
  have hsw : ∀ t, BitVec.ofInt 32 ((decLane K nw T rd).b t).scale = (laneIn K nw T rd).sw t := fun t =>
    BitVec.ofInt_natCast _ _
  have hzv : ∀ i j, (decLane K nw T rd).z i j = ((laneIn K nw T rd).zr i j).toInt := fun i j => by
    have e1 : (decLane K nw T rd).z i j =
        sgnBits 32 (bitsN (rd "zy") (fieldBit (lanePorts T 0) "zy.z" + 32 * (i.val * T.c + j.val)) 32) := by
      simp only [decLane]
    have e2 : (laneIn K nw T rd).zr i j =
        BitVec.ofNat 32 (bitsN (rd "zy") (fieldBit (lanePorts T 0) "zy.z" + 32 * (i.val * T.c + j.val)) 32) := by
      simp only [laneIn]
    rw [e1, e2]
    exact sgnBits_eq_toInt 32 _ (by norm_num) (bitsN_lt _ _ _)
  have hyv : ∀ i j, (decLane K nw T rd).y i j = (((laneIn K nw T rd).yr i j).toNat : ℤ) := fun i j => by
    show ((bitsN _ _ 16 : ℕ) : ℤ) = ((BitVec.ofNat 16 (bitsN _ _ 16)).toNat : ℤ)
    rw [BitVec.toNat_ofNat, Nat.mod_eq_of_lt (bitsN_lt _ _ _)]
  have hdot : ∀ s t, ∑ k, ((decLane K nw T rd).a s).codes k * ((decLane K nw T rd).b t).codes k =
      ∑ k, ((laneIn K nw T rd).a s k).toInt * ((laneIn K nw T rd).b t k).toInt := fun s t =>
    Finset.sum_congr rfl fun k _ => by rw [ha, hb]
  refine ⟨?_, fun s t => ?_⟩
  · funext w
    have e : (decLane K nw T rd).part w = (laneIn K nw T rd).partials w := rfl
    rw [e, hp w, hl]
    unfold partOf
    congr 2
    · funext s k
      rw [ha, BitVec.ofInt_toInt]
    · funext t k
      rw [hb, BitVec.ofInt_toInt]
  · rw [hl]
    have hc := hz s t
    have er : rowOf (laneIn K nw T rd).l s = Pouw.H1Zy.laneRow (laneIn K nw T rd).l s := rfl
    have ec : colOf (laneIn K nw T rd).l t = Pouw.H1Zy.laneCol (laneIn K nw T rd).l t := rfl
    have hzc : (decLane K nw T rd).z (rowOf (laneIn K nw T rd).l s) (colOf (laneIn K nw T rd).l t) =
        ∑ k, ((laneIn K nw T rd).a s k).toInt * ((laneIn K nw T rd).b t k).toInt := by
      rw [hzv, er, ec, hc.1]
      exact Pouw.H1Zy.toInt_ofInt_dot _ _ hK32 hK
    refine ⟨by rw [hzc, hdot], ?_⟩
    rw [hyv, er, ec, hc.2, ← er, ← ec, hzc]
    show _ = ((dqW _ _ _).toNat : ℤ)
    rw [hsa, hsw]

/-- Claims 1 and 2 of `LeafChecks`' lane field: the input wires sit where `decL` reads them, and the circuit computes
`PouwH1LaneZy_v1`. -/
def LaneComputes {K nw : ℕ} (T : Tile)
    (chain : LanePub → Fin 32 → (Fin 4 → Fin K → BitVec 8) → (Fin 8 → Fin K → BitVec 8) → Fin nw → BitVec 32)
    (dqW : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 16) (sess : Option (Fin 32) → RowsSession) : Prop :=
  ∀ l (vals : Fin (sess (some l)).Circ.N → Bool),
    (∀ u, ∀ g ∈ (sess (some l)).P.io u, (sess (some l)).Circ.eval vals g = vals g) →
    (decL K nw T (sess (some l)) (portRead (sess (some l)) vals)).ok =
      Pouw.H1Zy.laneOk chain dqW (laneIn K nw T (portBit (sess (some l)) (portRead (sess (some l)) vals)))

/-- **Each lane's circuit computes its pieces' composition at its port rows**: at lane `l`'s session `sess (some l)`, on
every wire assignment the key program's evaluation agrees with on every unit's wires, the `ok` word `decL` reads from the
port rows is `PouwH1LaneZy_v1{PouwH1Tile_v1}`'s body on bits (`Pouw.H1Zy.laneBits`) on the bits of the words `laneIn`
reads, at the dequant's word view `dqW` and the chain's words `chain` on bits (`Pouw.H1Zy.liftDq`, `liftChain`). What
it rests on is circuit-check's lowering of `PouwH1LaneZy_v1`, which composes those pieces. -/
def LaneLowers {K nw : ℕ} (T : Tile)
    (chain : LanePub → Fin 32 → (Fin 4 → Fin K → BitVec 8) → (Fin 8 → Fin K → BitVec 8) → Fin nw → BitVec 32)
    (dqW : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 16) (sess : Option (Fin 32) → RowsSession) : Prop :=
  ∀ l (vals : Fin (sess (some l)).Circ.N → Bool),
    (∀ u, ∀ g ∈ (sess (some l)).P.io u, (sess (some l)).Circ.eval vals g = vals g) →
    (decL K nw T (sess (some l)) (portRead (sess (some l)) vals)).ok =
      Pouw.H1Zy.laneBits (Pouw.H1Zy.liftChain chain) (Pouw.H1Zy.liftDq dqW)
        (laneIn K nw T (portBit (sess (some l)) (portRead (sess (some l)) vals))).bits

theorem laneLowers_iff {K nw : ℕ} (T : Tile)
    (chain : LanePub → Fin 32 → (Fin 4 → Fin K → BitVec 8) → (Fin 8 → Fin K → BitVec 8) → Fin nw → BitVec 32)
    (dqW : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 16) (sess : Option (Fin 32) → RowsSession) :
    LaneLowers T chain dqW sess ↔ LaneComputes T chain dqW sess := by
  unfold LaneLowers LaneComputes
  simp only [Pouw.H1Zy.laneBits_bv]

theorem laneComputes_of_lowers {K nw : ℕ} {T : Tile}
    {chain : LanePub → Fin 32 → (Fin 4 → Fin K → BitVec 8) → (Fin 8 → Fin K → BitVec 8) → Fin nw → BitVec 32}
    {dqW : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 16} {sess : Option (Fin 32) → RowsSession}
    (h : LaneLowers T chain dqW sess) : LaneComputes T chain dqW sess :=
  (laneLowers_iff T chain dqW sess).1 h

theorem leafChecks_of_computes {K nw : ℕ} (T : Tile)
    (chain : LanePub → Fin 32 → (Fin 4 → Fin K → BitVec 8) → (Fin 8 → Fin K → BitVec 8) → Fin nw → BitVec 32)
    (dqW : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 16) (sess : Option (Fin 32) → RowsSession)
    (hL : LaneComputes T chain dqW sess) (hC : CombineComputes sess (decC nw T)) :
    ResourceAccounting.Assumptions.TS.LeafChecks (dqOf dqW) (partOf chain) sess (decL K nw T) (decC nw T) :=
  leafChecks_of_combineComputes _ _ sess _ _
    (fun hK32 hK l vals hv hok => laneOk_of_def T chain dqW _ hK32 hK ((hL l vals hv).symm.trans hok)) hC

end ResourceAccounting.TS.Lanes
