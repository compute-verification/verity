import Proofs.Accounting.TS.Session.Lanes.Combine
import Proofs.Pouw.H1Zy.Lane
import Proofs.Pouw.H1Zy.LaneBits
import Definitions.Pouw.H1Zy.Chain

/-!
# A lane's part of `LeafChecks` from its Definition

`Accounting.Assumptions.TS.LeafChecks`' lane field bundles three claims: lane `l`'s circuit's input wires sit where
`decL` reads them, the circuit computes `PouwH1LaneZy_v1`, and that Definition's `ok = 1` means `LaneOk`. The third is
proved here for every input, at a `K` the lane's tile binds at (`32 ∣ K`, `K < 2^17`, `GemmAccumulatorS8`'s bind), with
`LeafChecks`' `dq` and `part` the Definition's own (`dqOf`, `partOf`):

* `laneIn K nw T rd`: the lane's inputs as words, read from the same port bits `decLane` reads (each code, scale, record
  word and partial word its field's bits; the index its 16 bits; the public inputs `decLane`'s);
* `dqOf dqW`: `LeafChecks`' `dq` at the dequant's word view `dqW` (`PouwZyDequant_v1`), on z's s32 word and the scales'
  f32 encodings; `partOf chain`: its `part` at the lane's chain words `chain` (`h1tile._chain`), masked by the admit bits'
  OR, as `h1zy.lane_partial`'s zero words for a dead leaf;
* `laneOk_of_def`: where `Pouw.H1Zy.laneOk chain dqW` is `true` on `laneIn`, `decLane`'s view meets `LaneOk`
  (`Pouw.H1Zy.laneOk_spec`: the index's high bits are `0`, so the view's index is the lane the Definition picked; every
  cell's z is the exact dot and y its dequant; the partials are the masked chain).

What stays of the field is the first two claims, `LaneComputes`: on every wire assignment the session's evaluation
agrees with, the decoded `ok` is the Definition's on `laneIn`. `leafChecks_of_computes` gives `LeafChecks` from it and
`CombineComputes`. Everything here is generic in `dqW` and `chain`; `Accounting.Guarantees.TS.LeafLanesSound` takes
`dqW` to be the Definition's, `Pouw.H1Zy.zyDequant` (`PouwZyDequant_v1`), and `chain` to be `laneChain K nw`, the words
of `h1tile._chain` (`Pouw.H1Zy.h1Chain K`: the fx1 chain over the noise from σ_J, keyed from σ_R) at the seeds the
`pub` row holds.

`LaneComputes` is in turn split at the lane's pieces: `LaneLowers` says the circuit computes the body on bits, its pieces
at their lowerings (`Pouw.H1Zy.laneBits`: the s8 multiply's Baugh–Wooley steps and clamp, `PouwZyCell_v2`,
`PouwMuxBit_v2`, `PouwMask32_v2`, `PouwEq32_v2`), the dequant and the chain's words at their word views on bits. That
each restated piece's lowering computes its word view, and so their composition the body on words, is proved
(`Pouw.H1Zy.laneBits_bv`). The two are equivalent (`laneLowers_iff`).
-/

namespace Accounting.TS.Lanes

open Accounting.TS Accounting.TS.Checked Flock.SoundAtBridge PouwWindow.Lanes

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

/-! ### Fields' bits as words -/

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

/-- The index `decLane` reads is the lane the Definition's selector names, wherever the index's high bits are `0`. -/
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

/-! ### Claim 3 at a lane -/

/-- **Claim 3 of `LeafChecks` at a lane**: at a `K` its tile binds at, where `PouwH1LaneZy_v1{PouwH1Tile_v1}`'s `ok` on
the lane's words (`laneIn`) is `true`, `decLane`'s view meets `LaneOk` at `dqOf dqW` and `partOf chain`. -/
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

/-! ### What stays of `LeafChecks` -/

/-- **Each lane's circuit computes its Definition at its port rows** (claims 1 and 2 of `LeafChecks`' lane field): at
lane `l`'s session `sess (some l)`, on every wire assignment the key program's evaluation agrees with on every unit's
wires, the `ok` word `decL` reads from the port rows (`portRead`) is `PouwH1LaneZy_v1{PouwH1Tile_v1}`'s on the words
`laneIn` reads from them, at the dequant `dqW` and the chain's words `chain`. -/
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

/-- **`LaneComputes` is `LaneLowers`**: the pieces' lowerings compose to the body on words (`Pouw.H1Zy.laneBits_bv`). -/
theorem laneLowers_iff {K nw : ℕ} (T : Tile)
    (chain : LanePub → Fin 32 → (Fin 4 → Fin K → BitVec 8) → (Fin 8 → Fin K → BitVec 8) → Fin nw → BitVec 32)
    (dqW : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 16) (sess : Option (Fin 32) → RowsSession) :
    LaneLowers T chain dqW sess ↔ LaneComputes T chain dqW sess := by
  unfold LaneLowers LaneComputes
  simp only [Pouw.H1Zy.laneBits_bv]

/-- **`LaneComputes` from `LaneLowers`.** -/
theorem laneComputes_of_lowers {K nw : ℕ} {T : Tile}
    {chain : LanePub → Fin 32 → (Fin 4 → Fin K → BitVec 8) → (Fin 8 → Fin K → BitVec 8) → Fin nw → BitVec 32}
    {dqW : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 16} {sess : Option (Fin 32) → RowsSession}
    (h : LaneLowers T chain dqW sess) : LaneComputes T chain dqW sess :=
  (laneLowers_iff T chain dqW sess).1 h

/-- **`LeafChecks` from the circuits computing their Definitions**: with `dq` and `part` the Definition's (`dqOf dqW`,
`partOf chain`), `LaneComputes` and `CombineComputes` give `Accounting.Assumptions.TS.LeafChecks` at the layout's
decodes, whose lane field asks only at a `K` the lane's tile binds at; that each Definition's `ok = 1` is `LaneOk` or
`CombineOk` is `laneOk_of_def` and `combineOk_of_def`. -/
theorem leafChecks_of_computes {K nw : ℕ} (T : Tile)
    (chain : LanePub → Fin 32 → (Fin 4 → Fin K → BitVec 8) → (Fin 8 → Fin K → BitVec 8) → Fin nw → BitVec 32)
    (dqW : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 16) (sess : Option (Fin 32) → RowsSession)
    (hL : LaneComputes T chain dqW sess) (hC : CombineComputes sess (decC nw T)) :
    Accounting.Assumptions.TS.LeafChecks (dqOf dqW) (partOf chain) sess (decL K nw T) (decC nw T) :=
  leafChecks_of_combineComputes _ _ sess _ _
    (fun hK32 hK l vals hv hok => laneOk_of_def T chain dqW _ hK32 hK ((hL l vals hv).symm.trans hok)) hC

end Accounting.TS.Lanes
