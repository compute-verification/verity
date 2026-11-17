module

public import Proofs.Pouw.TS.Session.Lanes.KnownFields
public import Proofs.Pouw.TS.Session.Lanes.Layout
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

/-! The leaf and the receipt the 33 sessions' committed rows decode to, each read where a statement reads it, so no
premise says which leaf and receipt the rows hold. -/

@[expose] public section

namespace ResourceAccounting.TS.Lanes

open PouwWindow.Lanes Flock.SoundAtBridge ResourceAccounting.TS.Checked

/-- The lane that reads A row `r`: lane `4 (r mod 8)`. -/
def aLane (r : Fin 32) : Fin 32 := ⟨4 * (r.val % 8), by omega⟩

/-- `aLane r`'s row slot of A row `r`: `⌊r/8⌋`. -/
def aSlot (r : Fin 32) : Fin 4 := ⟨r.val / 8, by omega⟩

/-- The lane that reads B row `j`: lane `⌊(j mod 8)/2⌋`. -/
def bLane (j : Fin 32) : Fin 32 := ⟨(j.val % 8) / 2, by omega⟩

/-- `bLane j`'s column slot of B row `j`: `2⌊j/8⌋ + (j mod 2)`. -/
def bSlot (j : Fin 32) : Fin 8 := ⟨2 * (j.val / 8) + j.val % 2, by omega⟩

/-- The lane whose cells cell `(i, j)` is among: lane `4 (i mod 8) + ⌊(j mod 8)/2⌋`. -/
def cellLane (i j : Fin 32) : Fin 32 := ⟨4 * (i.val % 8) + (j.val % 8) / 2, by omega⟩

/-- **The leaf the lanes' committed rows decode to**: A row `r` as lane `aLane r` decodes it at row slot `aSlot r`, B
row `j` as lane `bLane j` decodes it at column slot `bSlot j`, and cell `(i, j)` of the record as lane `cellLane i j`
decodes it, each at the registrants' commit-time openings (`openedReg`). -/
noncomputable def leafOf (K : ℕ) (T : Tile) (sess : Option (Fin 32) → RowsSession) : Leaf K :=
  let d := fun l : Fin 32 => decL K 0 T (sess (some l)) (openedReg (sess (some l)))
  ⟨fun r => (d (aLane r)).a (aSlot r), fun j => (d (bLane j)).b (bSlot j), fun i j => (d (cellLane i j)).z i j,
    fun i j => (d (cellLane i j)).y i j⟩

/-- **The receipt the combine's committed row decodes to**, at its registrant's commit-time openings. -/
noncomputable def receiptOf (nw : ℕ) (T : Tile) (sess : Option (Fin 32) → RowsSession) : Fin nw → BitVec 32 :=
  (decC nw T (sess none) (openedReg (sess none))).receipt

/-- Lane `4⌊l/4⌋`, which reads lane `l`'s A rows at the same slots. -/
def aHome (l : Fin 32) : Fin 32 := ⟨4 * (l.val / 4), by omega⟩

/-- Lane `l mod 4`, which reads lane `l`'s B rows at the same slots. -/
def bHome (l : Fin 32) : Fin 32 := ⟨l.val % 4, by omega⟩

theorem aLane_rowOf (l : Fin 32) (s : Fin 4) : aLane (rowOf l s) = aHome l := by
  apply Fin.ext
  simp only [aLane, rowOf, aHome]
  omega

theorem aSlot_rowOf (l : Fin 32) (s : Fin 4) : aSlot (rowOf l s) = s := by
  apply Fin.ext
  simp only [aSlot, rowOf]
  omega

theorem rowOf_aHome (l : Fin 32) (s : Fin 4) : rowOf (aHome l) s = rowOf l s := by
  apply Fin.ext
  simp only [rowOf, aHome]
  omega

theorem bLane_colOf (l : Fin 32) (t : Fin 8) : bLane (colOf l t) = bHome l := by
  apply Fin.ext
  simp only [bLane, colOf, bHome]
  omega

theorem bSlot_colOf (l : Fin 32) (t : Fin 8) : bSlot (colOf l t) = t := by
  apply Fin.ext
  simp only [bSlot, colOf]
  omega

theorem colOf_bHome (l : Fin 32) (t : Fin 8) : colOf (bHome l) t = colOf l t := by
  apply Fin.ext
  simp only [colOf, bHome]
  omega

theorem cellLane_rowOf_colOf (l : Fin 32) (s : Fin 4) (t : Fin 8) : cellLane (rowOf l s) (colOf l t) = l := by
  apply Fin.ext
  simp only [cellLane, rowOf, colOf]
  omega

structure LaneAgrees {K nw : ℕ} {Pub : Type} (v : LaneView K nw Pub) (L : Leaf K) (pub : Pub) (p : Fin nw → BitVec 32)
    (l : Fin 32) : Prop where
  a : v.a = laneA L l
  b : v.b = laneB L l
  z : ∀ s t, v.z (rowOf l s) (colOf l t) = L.z (rowOf l s) (colOf l t)
  y : ∀ s t, v.y (rowOf l s) (colOf l t) = L.y (rowOf l s) (colOf l t)
  part : v.part = p
  pub : v.pub = pub
  lane : v.lane = l
  ok : v.ok = true

theorem LaneAgrees.laneOk {K nw : ℕ} {Pub : Type} {dq : List ℤ → ℤ}
    {part : Pub → Fin 32 → (Fin 4 → Row K) → (Fin 8 → Row K) → Fin nw → BitVec 32} {v : LaneView K nw Pub}
    {L : Leaf K} {pub : Pub} {p : Fin nw → BitVec 32} {l : Fin 32} (h : LaneAgrees v L pub p l)
    (hv : LaneOk dq part v) : LaneOk dq part (laneView L pub p l) := by
  obtain ⟨a, b, z, y, p', pub', l', ok⟩ := v
  obtain ⟨ha, hb, hz, hy, hp, hpub, hl, -⟩ := h
  dsimp only at ha hb hz hy hp hpub hl
  subst ha hb hp hpub hl
  obtain ⟨hpart, hc⟩ := hv
  refine ⟨hpart, fun s t => ?_⟩
  obtain ⟨h1, h2⟩ := hc s t
  dsimp only at h1 h2
  dsimp only [laneView]
  rw [← hz s t, ← hy s t]
  exact ⟨h1, h2⟩

theorem LaneKnownHolds.pub_eq {K nw : ℕ} {T : Tile} {rd : String → ℕ → Option Bool} {P : LanePub} {l : Fin 32}
    (h : LaneKnownHolds T rd P l) : (decLane K nw T rd).pub = P := by
  obtain ⟨admit, salt, position, roots, hs, hp, hr⟩ := P
  show (⟨_, _, _, _, _, _, _⟩ : LanePub) = _
  rw [LanePub.mk.injEq]
  refine ⟨?_, ?_, ?_, ?_⟩
  · funext i
    rw [h.admit i]
    dsimp only
    cases admit i <;> rfl
  · exact bitsN_eq h.salt hs
  · exact bitsN_eq h.position hp
  · exact bitsN_eq h.roots hr

theorem LaneKnownHolds.lane_eq {K nw : ℕ} {T : Tile} {rd : String → ℕ → Option Bool} {P : LanePub} {l : Fin 32}
    (h : LaneKnownHolds T rd P l) : (decLane K nw T rd).lane = l := by
  have hl : l.val < 2 ^ laneBits := by
    have := l.isLt
    simp only [laneBits]
    omega
  apply Fin.ext
  show bitsN (rd "pub") (fieldBit (lanePorts T 0) "lane") laneBits % 32 = l.val
  rw [bitsN_eq h.lane hl]
  exact Nat.mod_eq_of_lt l.isLt

theorem LaneKnownHolds.ok_eq {K nw : ℕ} {T : Tile} {rd : String → ℕ → Option Bool} {P : LanePub} {l : Fin 32}
    (h : LaneKnownHolds T rd P l) : (decLane K nw T rd).ok = true := by
  show (rd "o0" 0 == some true) = true
  rw [h.ok]
  rfl

theorem decLane_a {K nw nw' : ℕ} {T : Tile} {rd rd' : String → ℕ → Option Bool} {s : Fin 4}
    (h : rd s!"a{s.val}" = rd' s!"a{s.val}") : (decLane K nw T rd).a s = (decLane K nw' T rd').a s := by
  first
  | (simp only [decLane, h]; done)
  | (unfold decLane; dsimp only; rw [h])

theorem decLane_b {K nw nw' : ℕ} {T : Tile} {rd rd' : String → ℕ → Option Bool} {t : Fin 8}
    (h : rd s!"b{t.val}" = rd' s!"b{t.val}") : (decLane K nw T rd).b t = (decLane K nw' T rd').b t := by
  first
  | (simp only [decLane, h]; done)
  | (unfold decLane; dsimp only; rw [h])

theorem CombineView.eta_ok {nw : ℕ} (v : CombineView nw) (h : v.ok = true) : v = ⟨v.parts, v.receipt, true⟩ := by
  obtain ⟨parts, receipt, ok⟩ := v
  dsimp only at h
  subst h
  rfl

theorem a_read (K : ℕ) (l : Fin 32) (s : Fin 4) :
    (s!"a{s.val}", "a", (rowOf l s).val) ∈ regReads (servedTile K) (some l.val) :=
  List.mem_filterMap.2 ⟨_, lanePorts_a K l s, rfl⟩

theorem b_read (K : ℕ) (l : Fin 32) (t : Fin 8) :
    (s!"b{t.val}", "b", (colOf l t).val) ∈ regReads (servedTile K) (some l.val) :=
  List.mem_filterMap.2 ⟨_, lanePorts_b K l t, rfl⟩

theorem fullValue_a : fullValue "a" = true := by
  first | rfl | decide | simp [fullValue]

theorem fullValue_b : fullValue "b" = true := by
  first | rfl | decide | simp [fullValue]

theorem decLane_z {K nw nw' : ℕ} {T : Tile} {rd : String → ℕ → Option Bool} (i j : Fin 32) :
    (decLane K nw T rd).z i j = (decLane K nw' T rd).z i j :=
  rfl

theorem decLane_y {K nw nw' : ℕ} {T : Tile} {rd : String → ℕ → Option Bool} (i j : Fin 32) :
    (decLane K nw T rd).y i j = (decLane K nw' T rd).y i j :=
  rfl

theorem decComb_ok {nw : ℕ} {T : Tile} {rd : String → ℕ → Option Bool} :
    (decComb nw T rd).ok = (rd "o0" 0 == some true) :=
  rfl

theorem laneAgrees_of {K nw : ℕ} {T : Tile} (sess : Option (Fin 32) → RowsSession) {P : LanePub}
    {p : Fin nw → BitVec 32} (l : Fin 32)
    (ha : ∀ s : Fin 4, portBit (sess (some l)) (openedReg (sess (some l))) s!"a{s.val}" =
      portBit (sess (some (aHome l))) (openedReg (sess (some (aHome l)))) s!"a{s.val}")
    (hb : ∀ t : Fin 8, portBit (sess (some l)) (openedReg (sess (some l))) s!"b{t.val}" =
      portBit (sess (some (bHome l))) (openedReg (sess (some (bHome l)))) s!"b{t.val}")
    (hp : (decL K nw T (sess (some l)) (openedReg (sess (some l)))).part = p)
    (hk : LaneKnownHolds T (portBit (sess (some l)) (openedReg (sess (some l)))) P l) :
    LaneAgrees (decL K nw T (sess (some l)) (openedReg (sess (some l)))) (leafOf K T sess) P p l := by
  refine ⟨funext fun s => ?_, funext fun t => ?_, fun s t => ?_, fun s t => ?_, hp, hk.pub_eq, hk.lane_eq,
    hk.ok_eq⟩
  · simp only [laneA, leafOf, decL]
    rw [aLane_rowOf, aSlot_rowOf]
    exact decLane_a (ha s)
  · simp only [laneB, leafOf, decL]
    rw [bLane_colOf, bSlot_colOf]
    exact decLane_b (hb t)
  · simp only [leafOf, decL]
    rw [cellLane_rowOf_colOf]
    exact decLane_z _ _
  · simp only [leafOf, decL]
    rw [cellLane_rowOf_colOf]
    exact decLane_y _ _

end ResourceAccounting.TS.Lanes
