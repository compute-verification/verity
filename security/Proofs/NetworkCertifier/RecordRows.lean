module

public import Definitions.NetworkCertifier.RecordRows
public import Definitions.NetworkCertifier.Tap
public import Mathlib.Data.Fin.VecNotation
public import Proofs.NetworkCertifier.Warden
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Data.List.OfFn
meta import Definitions.NetworkCertifier.RecordRows

/-!
# The network certifier's rows read back its lanes

`Definitions.NetworkCertifier.RecordRows`. On a schedule that fits the rows format, the lanes read back from the record
of the cells the network certifier's code sends are those cells (`RowsFormat.read_record`), so the rows format is a
`RecordFormat` (`RowsFormat.recordFormat`). A lane of `k` cells has cell `j` in bucket `t j` (`certifierLane_fst`); among the
cells before `j`, those in `j`'s bucket number `j mod r` (`countP_range_div`), so `j` is the `(j mod r)`-th of its
bucket's (`getElem?_filter_countP_take`); a row's frames are its connections' in turn, so a connection's are its own
(`filterMap_bucketFrames`), and the record's slots hold a row's frames in order when the row fits (`frames_record`).

The deployed format is well formed under 2^32 connections and `B`, the frame's bytes, under 2^32 (`cellRows_wf`): a
u32 holds each connection's ordinal and each payload's length, and so a frame names its connection, its kind and its
payload (`cellFrame_inj`).

The `#guard`s hold the rows format to the Python network certifier's egress records (the test harness of
`verity/core/protocols/certifier/tests/test_network_certifier_warden.py`, `params`: B = 64, r = 2, D = 3, two lanes,
T = 24): three connections on two lanes, the third admitted when the second's END frees its lane (`fxQueue`, the
traffic of `test_network_certifier_warden_rows.py`); a byte late at its cell (`fxLate`); and a server past
`egress_cells` (`fxLimit`). Each frame is its ordinal, kind and payload in hex; a lane's arrivals are its DATA cells'
and its END's payloads (`true` for the END), every cell in time except a failed lane's next one.
-/

@[expose] public section

namespace NetTiming

variable {κ Cell : Type}

/-- A connection's frames in a row are its own frames in the bucket. -/
theorem filterMap_bucketFrames [DecidableEq κ] {conns : List κ} (hnd : conns.Nodup) {c : κ} (hc : c ∈ conns)
    (s : CertifierSchedule κ) (v : κ → List (ℕ × CellView Cell)) (b : ℕ) :
    (bucketFrames conns s v b).filterMap (fun f => if f.1 = c then some f.2 else none) = laneFrames s v c b := by
  have : ∀ d, ((laneFrames s v d b).map (d, ·)).filterMap (fun f => if f.1 = c then some f.2 else none) =
      if d = c then laneFrames s v c b else [] := by
    intro d
    by_cases h : d = c
    · subst h; simp [List.filterMap_map]
    · simp [List.filterMap_map, h]
  rw [bucketFrames, List.filterMap_flatMap]
  simp only [this]
  exact filterMap_flatMap_ite _ hnd hc

/-- **The lanes read back from the rows of the code's cells are the code's cells.** -/
theorem lanesOfRows_bucketFrames [DecidableEq κ] {conns : List κ} (hnd : conns.Nodup) (hall : ∀ c, c ∈ conns)
    (s : CertifierSchedule κ) (a : κ → Arrivals Cell) (c : κ) :
    lanesOfRows s (bucketFrames conns s (certifierCode s a)) c = certifierCode s a c := by
  have hlen : (certifierCode s a c).length = s.cells c := certifierLane_length _ _ _ _
  have hfst : ∀ j (hj : j < (certifierCode s a c).length), ((certifierCode s a c)[j]).1 = s.bucket c j := by
    intro j hj
    have h := congrArg (fun l => l[j]?) (certifierCode_fst s a c)
    simp only [List.getElem?_map, List.getElem?_eq_getElem hj, List.getElem?_range (hlen ▸ hj),
      Option.map_some, Option.some.injEq] at h
    exact h
  apply List.ext_getElem (by simp [lanesOfRows, hlen])
  intro j h1 h2
  have hj : j < s.cells c := by simpa [lanesOfRows] using h1
  simp only [lanesOfRows, List.getElem_map, List.getElem_range]
  rw [filterMap_bucketFrames hnd (hall c)]
  have hpre : ¬(s.adm c ≤ s.bucket c j ∧ s.bucket c j < s.adm c + s.D) := fun h =>
    absurd h.2 (Nat.not_lt.2 (Nat.le_add_right _ _))
  simp only [laneFrames, hpre, ite_false, List.nil_append]
  have key := getElem?_filter_countP_take (fun x : ℕ × CellView Cell => x.1 == s.bucket c j) (certifierCode s a c) j h2
    (by simp [hfst j h2])
  have hcount : ((certifierCode s a c).take j).countP (fun x => x.1 == s.bucket c j) = j % s.r := by
    have := congrArg (fun l => (l.take j).countP (· == s.bucket c j)) (certifierCode_fst s a c)
    simp only [← List.map_take, List.countP_map, List.take_range, Nat.min_eq_left hj.le] at this
    rw [show (fun x : ℕ × CellView Cell => x.1 == s.bucket c j) = (· == s.bucket c j) ∘ Prod.fst from rfl, this,
      ← countP_range_div s.r j]
    congr 1
    funext i
    simp [CertifierSchedule.bucket]
  rw [hcount] at key
  rw [List.getElem?_map, key]
  exact Prod.ext (hfst j h2).symm rfl

/-- A row of the code's cells holds each connection's load in the bucket. -/
theorem length_bucketFrames (conns : List κ) (s : CertifierSchedule κ) (a : κ → Arrivals Cell) (b : ℕ) :
    (bucketFrames conns s (certifierCode s a) b).length = (conns.map fun c => laneLoad s c b).sum := by
  simp only [bucketFrames, List.length_flatMap, List.length_map, laneFrames, List.length_append, laneLoad]
  congr 1
  refine List.map_congr_left fun c _ => ?_
  congr 1
  · split_ifs <;> simp
  · rw [← List.countP_eq_length_filter, ← List.countP_eq_length_filter,
      show (fun x : ℕ × CellView Cell => x.1 == b) = (· == b) ∘ Prod.fst from rfl, ← List.countP_map,
      certifierCode_fst, List.countP_map]
    rfl

theorem filterMap_range_getElem? {α : Type} (l : List α) :
    ∀ S, (List.range S).filterMap (fun i => l[i]?) = l.take S
  | 0 => by simp
  | S + 1 => by
    rw [List.range_succ, List.filterMap_append, filterMap_range_getElem? l S, List.take_add_one]
    cases h : l[S]? <;> simp [h]

theorem filterMap_finRange_getElem? {α : Type} (l : List α) {S : ℕ} (h : l.length ≤ S) :
    (List.finRange S).filterMap (fun σ : Fin S => l[σ.val]?) = l := by
  have : (List.finRange S).filterMap (fun σ : Fin S => l[σ.val]?) =
      ((List.finRange S).map Fin.val).filterMap (fun i => l[i]?) := by
    rw [List.filterMap_map]; rfl
  rw [this, List.map_coe_finRange_eq_range, filterMap_range_getElem?, List.take_of_length_le h]

namespace RowsFormat

variable {L K : ℕ}

/-- **The record's slots hold a row's frames in order** at a bucket of the monitoring period whose frames fit. -/
theorem frames_record (F : RowsFormat κ Cell L K) (hdec : ∀ x, F.dec (F.enc x) = some x) (S n : ℕ)
    (s : CertifierSchedule κ) (a : κ → Arrivals Cell) (v : κ → List (ℕ × CellView Cell)) {b : ℕ} (hb : b < n * F.T)
    (hS : (bucketFrames F.conns s v b).length ≤ S) :
    F.frames (F.record S n s a v) b = bucketFrames F.conns s v b := by
  have hT : 0 < F.T := Nat.pos_of_ne_zero fun h => by simp [h] at hb
  have hi : b / F.T < n := (Nat.div_lt_iff_lt_mul hT).2 hb
  have ht : b % F.T < F.T := Nat.mod_lt _ hT
  have hbt : b / F.T * F.T + b % F.T = b := Nat.div_add_mod' b F.T
  have hslot : ∀ σ : Fin S, (((bucketFrames F.conns s v b)[σ.val]?).map F.enc).bind F.dec =
      (bucketFrames F.conns s v b)[σ.val]? := fun σ => by
    cases (bucketFrames F.conns s v b)[σ.val]? <;> simp [hdec]
  simp only [frames, hi, ↓reduceDIte, record, List.getElem?_map, List.getElem?_range ht, Option.map_some, hbt,
    ↓reduceIte, hslot]
  exact filterMap_finRange_getElem? _ hS

/-- **The rows format reads back the code's cells** on every schedule that fits it. -/
theorem read_record [DecidableEq κ] (F : RowsFormat κ Cell L K) (hF : F.Wf) (S n : ℕ) (s : CertifierSchedule κ)
    (hs : F.Fits S n s) (a : κ → Arrivals Cell) :
    F.read s (F.record S n s a (certifierCode s a)) = certifierCode s a := by
  obtain ⟨hnd, hall, hdec⟩ := hF
  funext c
  have : F.read s (F.record S n s a (certifierCode s a)) c = lanesOfRows s (bucketFrames F.conns s (certifierCode s a)) c := by
    simp only [read, lanesOfRows]
    refine List.map_congr_left fun j hj => ?_
    rw [frames_record F hdec S n s a _ (hs.1 c j (List.mem_range.1 hj))
      ((length_bucketFrames _ _ _ _).trans_le (hs.2 _))]
  rw [this, lanesOfRows_bucketFrames hnd hall s a c]

/-- **The rows format is a record format of the network certifier**: on the schedules that fit it, its record of the
cells the code sends reads back to them. -/
def recordFormat [DecidableEq κ] (F : RowsFormat κ Cell L K) (hF : F.Wf) (S n : ℕ) :
    RecordFormat κ Cell (Fin n → List (Obs L K S)) where
  record := F.record S n
  read := F.read
  fits := F.Fits S n
  read_record s hs a := F.read_record hF S n s hs a

end RowsFormat

/-! ### The deployed rows format is well formed -/

theorem length_bitsBE (w x : ℕ) : (bitsBE w x).length = w :=
  List.length_ofFn

theorem bitsBE_inj {w x y : ℕ} (hx : x < 2 ^ w) (hy : y < 2 ^ w) (h : bitsBE w x = bitsBE w y) : x = y := by
  unfold bitsBE at h
  refine Nat.eq_of_testBit_eq fun i => ?_
  by_cases hi : i < w
  · have := congrFun (List.ofFn_injective h) ⟨w - 1 - i, by omega⟩
    simp only at this
    rwa [show w - 1 - (w - 1 - i) = i by omega] at this
  · have hw : 2 ^ w ≤ 2 ^ i := Nat.pow_le_pow_right (by decide) (by omega)
    rw [Nat.testBit_lt_two_pow (Nat.lt_of_lt_of_le hx hw), Nat.testBit_lt_two_pow (Nat.lt_of_lt_of_le hy hw)]

theorem length_flatMap_bytes (p : List (Fin 256)) : (p.flatMap fun b => bitsBE 8 b.val).length = 8 * p.length := by
  induction p with
  | nil => rfl
  | cons b p ih => simp only [List.flatMap_cons, List.length_append, length_bitsBE, ih, List.length_cons]; omega

theorem flatMap_bytes_inj {p q : List (Fin 256)} (hl : p.length = q.length)
    (h : (p.flatMap fun b => bitsBE 8 b.val) = q.flatMap fun b => bitsBE 8 b.val) : p = q := by
  induction p generalizing q with
  | nil => cases q with
    | nil => rfl
    | cons => simp only [List.length_nil, List.length_cons] at hl; omega
  | cons a p ih => cases q with
    | nil => simp only [List.length_nil, List.length_cons] at hl; omega
    | cons b q =>
      simp only [List.flatMap_cons] at h
      obtain ⟨hab, hpq⟩ := List.append_inj h (by simp only [length_bitsBE])
      rw [Fin.ext (bitsBE_inj (Nat.lt_of_lt_of_le a.isLt (by decide)) (Nat.lt_of_lt_of_le b.isLt (by decide)) hab),
        ih (by simpa using hl) hpq]

theorem frameKind_fst_lt {P : ℕ} (v : CellView (FrameCell P)) : (frameKind v).1 < 2 ^ 8 := by
  cases v with
  | data x => simp only [frameKind]; split <;> simp
  | pad => simp [frameKind]
  | fail => simp [frameKind]

theorem frameKind_inj {P : ℕ} {u v : CellView (FrameCell P)} (h : frameKind u = frameKind v) : u = v := by
  rcases u with ⟨⟨_ | _, _, _⟩⟩ | _ | _ <;> rcases v with ⟨⟨_ | _, _, _⟩⟩ | _ | _ <;> simp_all [frameKind]

/-- **A frame names its connection and its cell**: under 2^32 connections and `B`-byte frames, `B` under 2^32, the
network certifier's frames of two cells are equal only when they are the same connection's same cell. -/
theorem cellFrame_inj {B N : ℕ} (hB : B < 2 ^ 32) (hN : N < 2 ^ 32) {x y : Fin N × CellView (FrameCell (B - 9))}
    (h : cellFrame B x = cellFrame B y) : x = y := by
  have hlen : ∀ v : CellView (FrameCell (B - 9)), (frameKind v).2.length < 2 ^ 32 := fun v => by
    cases v with
    | data d => exact Nat.lt_of_le_of_lt (Nat.le_trans d.fits (Nat.sub_le _ _)) hB
    | pad => simp [frameKind]
    | fail => simp [frameKind]
  have hord : ∀ c : Fin N, c.val + 1 < 2 ^ 32 := fun c => Nat.lt_of_le_of_lt (Nat.succ_le_of_lt c.isLt) hN
  simp only [cellFrame, List.append_assoc] at h
  obtain ⟨hk, h⟩ := List.append_inj h (by simp only [length_bitsBE])
  obtain ⟨ho, h⟩ := List.append_inj h (by simp only [length_bitsBE])
  obtain ⟨hl, h⟩ := List.append_inj h (by simp only [length_bitsBE])
  have hl' := bitsBE_inj (hlen x.2) (hlen y.2) hl
  obtain ⟨hp, -⟩ := List.append_inj h (by simp only [length_flatMap_bytes, hl'])
  exact Prod.ext (Fin.ext (Nat.add_right_cancel (bitsBE_inj (hord x.1) (hord y.1) ho)))
    (frameKind_inj (Prod.ext (bitsBE_inj (frameKind_fst_lt _) (frameKind_fst_lt _) hk) (flatMap_bytes_inj hl' hp)))

/-- **The deployed rows format is well formed** under 2^32 connections and `B`-byte frames, `B` under 2^32: its
connections are each once in ordinal order, and a frame reads back as the connection and cell it encodes. -/
theorem cellRows_wf {L K B N : ℕ} (hB : B < 2 ^ 32) (hN : N < 2 ^ 32) (link : Fin L) (T : ℕ)
    (jitter : CertifierSchedule (Fin N) → (Fin N → Arrivals (FrameCell (B - 9))) → ℕ → ℕ)
    (status : CertifierSchedule (Fin N) → (Fin N → Arrivals (FrameCell (B - 9))) → ℕ → Fin K → Status) :
    (cellRows B N link T jitter status).Wf := by
  refine ⟨List.nodup_finRange N, List.mem_finRange, fun x => ?_⟩
  have h : ∃ y, cellFrame B y = cellFrame B x := ⟨x, rfl⟩
  simp only [cellRows]
  rw [dite_eq_left h]
  exact congrArg some (cellFrame_inj hB hN (Classical.choose_spec h))

/-! ### The Python network certifier's egress records -/

section Fixtures

/-- A frame as the Python record holds it: its connection's ordinal (from 1), its kind and its payload in hex. -/
def pyFrame {n : ℕ} : Fin n × CellView (Bool × String) → ℕ × String × String
  | (c, .data (false, h)) => (c.val + 1, "data", h)
  | (c, .data (true, h)) => (c.val + 1, "end", h)
  | (c, .pad) => (c.val + 1, "pad", "")
  | (c, .fail) => (c.val + 1, "fail", "")

/-- The frame a Python record's frame is. -/
def ofPyFrame {n : ℕ} : ℕ × String × String → Option (Fin n × CellView (Bool × String))
  | (q, k, h) =>
    if hq : 0 < q ∧ q - 1 < n then
      let c : Fin n := ⟨q - 1, hq.2⟩
      if k = "data" then some (c, .data (false, h)) else if k = "end" then some (c, .data (true, h))
      else if k = "pad" then some (c, .pad) else if k = "fail" then some (c, .fail) else none
    else none

/-- A cell of a lane, its bucket and the kind and payload its frame holds. -/
def pyCell : ℕ × CellView (Bool × String) → ℕ × String × String
  | (b, x) => (b, (pyFrame ((0 : Fin 1), x)).2)

/-- The rows format's frames of the code's cells are the record's, bucket by bucket; the lanes read back from the
record's frames are the code's cells; and no bucket holds more than the `S = 4` slots. -/
def fixtureAgrees {n : ℕ} (s : CertifierSchedule (Fin n)) (arr : Fin n → Arrivals (Bool × String))
    (rows : List (List (ℕ × String × String))) : Bool :=
  (List.range (rows.length + 8)).all (fun b => (bucketFrames (List.finRange n) s (certifierCode s arr) b).map pyFrame ==
      rows.getD b []) &&
    (List.finRange n).all (fun c => (lanesOfRows s (fun b => (rows.getD b []).filterMap ofPyFrame) c).map pyCell ==
      (certifierCode s arr c).map pyCell) &&
    (List.range (rows.length + 8)).all fun b => (bucketFrames (List.finRange n) s (certifierCode s arr) b).length ≤ 4

/-- `fxQueue`'s arrivals, on its schedule `fxQueueSched` (`Definitions.NetworkCertifier.RecordRows`). -/
def fxQueueArr : Fin 3 → Arrivals (Bool × String) := ![
  { stream := [(false, "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f30313233343536"), (false, "3738393a3b3c3d3e3f404142434445464748494a4b4c4d4e4f505152535455565758595a5b5c5d5e5f606162636465666768696a6b6c6d"), (false, "6e6f707172737475767778797a7b7c7d7e7f808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9fa0a1a2a3a4"), (false, "a5a6a7a8a9aaabacadaeafb0b1b2b3b4b5b6b7b8b9babbbcbdbebfc0c1c2c3c4c5c6c7c8c9cacbcccdcecfd0d1d2d3d4d5d6d7d8d9dadb"), (true, "dcdddedfe0e1e2e3e4e5e6e7e8e9eaebecedeeeff0f1f2f3f4f5f6f7f8f9fafbfcfdfeff")], ready := fun _ => 0 },
  { stream := [(true, "")], ready := fun _ => 0 },
  { stream := [(false, "7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a"), (true, "7a")], ready := fun _ => 0 }]

/-- `fxQueue`'s egress record: each bucket's held slots, in slot order. -/
def fxQueueRows : List (List (ℕ × String × String)) := [
  [],
  [(1, "pad", ""), (1, "pad", "")],
  [(1, "pad", ""), (1, "pad", ""), (2, "pad", ""), (2, "pad", "")],
  [(1, "pad", ""), (1, "pad", ""), (2, "pad", ""), (2, "pad", "")],
  [(1, "data", "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f30313233343536"), (1, "data", "3738393a3b3c3d3e3f404142434445464748494a4b4c4d4e4f505152535455565758595a5b5c5d5e5f606162636465666768696a6b6c6d"), (2, "pad", ""), (2, "pad", "")],
  [(1, "data", "6e6f707172737475767778797a7b7c7d7e7f808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9fa0a1a2a3a4"), (1, "data", "a5a6a7a8a9aaabacadaeafb0b1b2b3b4b5b6b7b8b9babbbcbdbebfc0c1c2c3c4c5c6c7c8c9cacbcccdcecfd0d1d2d3d4d5d6d7d8d9dadb"), (2, "end", ""), (2, "pad", "")],
  [(1, "end", "dcdddedfe0e1e2e3e4e5e6e7e8e9eaebecedeeeff0f1f2f3f4f5f6f7f8f9fafbfcfdfeff"), (1, "pad", ""), (3, "pad", ""), (3, "pad", "")],
  [(3, "pad", ""), (3, "pad", "")],
  [(3, "pad", ""), (3, "pad", "")],
  [(3, "data", "7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a"), (3, "end", "7a")]]

#guard fixtureAgrees fxQueueSched fxQueueArr fxQueueRows

/-- `fxLate`: 1 connections; the Python network certifier's egress record, buckets 0 … 5 (every later bucket empty). -/
def fxLateSched : CertifierSchedule (Fin 1) :=
  { r := 2, D := 3, adm := ![1], cells := ![4] }

/-- `fxLate`'s arrivals. -/
def fxLateArr : Fin 1 → Arrivals (Bool × String) := ![
  { stream := [(false, "61616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161"), (false, "61616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161")], ready := fun j => if j < 2 then 0 else 1000000 }]

/-- `fxLate`'s egress record: each bucket's held slots, in slot order. -/
def fxLateRows : List (List (ℕ × String × String)) := [
  [],
  [(1, "pad", ""), (1, "pad", "")],
  [(1, "pad", ""), (1, "pad", "")],
  [(1, "pad", ""), (1, "pad", "")],
  [(1, "data", "61616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161"), (1, "data", "61616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161")],
  [(1, "fail", ""), (1, "pad", "")]]

#guard fixtureAgrees fxLateSched fxLateArr fxLateRows

/-- `fxLimit`: 1 connections; the Python network certifier's egress record, buckets 0 … 8 (every later bucket empty). -/
def fxLimitSched : CertifierSchedule (Fin 1) :=
  { r := 2, D := 3, adm := ![0], cells := ![12] }

/-- `fxLimit`'s arrivals. -/
def fxLimitArr : Fin 1 → Arrivals (Bool × String) := ![
  { stream := [(false, "71717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171"), (false, "71717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171"), (false, "71717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171"), (false, "71717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171"), (false, "71717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171"), (false, "71717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171"), (false, "71717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171"), (false, "71717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171"), (false, "71717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171"), (false, "71717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171"), (false, "71717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171")], ready := fun j => if j < 11 then 0 else 1000000 }]

/-- `fxLimit`'s egress record: each bucket's held slots, in slot order. -/
def fxLimitRows : List (List (ℕ × String × String)) := [
  [(1, "pad", ""), (1, "pad", "")],
  [(1, "pad", ""), (1, "pad", "")],
  [(1, "pad", ""), (1, "pad", "")],
  [(1, "data", "71717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171"), (1, "data", "71717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171")],
  [(1, "data", "71717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171"), (1, "data", "71717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171")],
  [(1, "data", "71717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171"), (1, "data", "71717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171")],
  [(1, "data", "71717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171"), (1, "data", "71717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171")],
  [(1, "data", "71717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171"), (1, "data", "71717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171")],
  [(1, "data", "71717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171717171"), (1, "fail", "")]]

#guard fixtureAgrees fxLimitSched fxLimitArr fxLimitRows

end Fixtures

end NetTiming
