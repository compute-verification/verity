import Definitions.NetworkCertifier.RecordRows

/-!
# The network certifier's rows read back its lanes

`Definitions.NetworkCertifier.RecordRows`. On a schedule that fits the rows format, the lanes read back from the record
of the cells the network certifier's code sends are those cells (`RowsFormat.read_record`), so the rows format is a
`TapRecord` (`RowsFormat.tapRecord`). A lane of `k` cells has cell `j` in bucket `t j` (`wardenLane_fst`); among the
cells before `j`, those in `j`'s bucket number `j mod r` (`countP_range_div`), so `j` is the `(j mod r)`-th of its
bucket's (`getElem?_filter_countP_take`); a row's frames are its connections' in turn, so a connection's are its own
(`filterMap_bucketFrames`), and the record's slots hold a row's frames in order when the row fits (`frames_record`).

The `#guard`s hold the rows format to the Python network certifier's egress records (the test harness of
`verity/core/protocols/certifier/tests/test_network_certifier_warden.py`, `params`: B = 64, r = 2, D = 3, two lanes,
T = 24): three connections on two lanes, the third admitted when the second's END frees its lane (`fxQueue`, the
traffic of `test_network_certifier_warden_rows.py`); a byte late at its cell (`fxLate`); and a server past
`egress_cells` (`fxLimit`). Each frame is its ordinal, kind and payload in hex; a lane's arrivals are its DATA cells'
and its END's payloads (`true` for the END), every cell in time except a failed lane's next one.
-/

namespace NetTiming

variable {κ Cell : Type}

theorem wardenLane_length (a : Arrivals Cell) (t : ℕ → ℕ) : ∀ j k, (wardenLane a t j k).length = k
  | _, 0 => rfl
  | j, k + 1 => by
    simp only [wardenLane]
    split_ifs <;> simp [wardenLane_length a t (j + 1) k]

/-- A lane's cell `i` is in bucket `t i`. -/
theorem wardenLane_fst (a : Arrivals Cell) (t : ℕ → ℕ) :
    ∀ j k, (wardenLane a t j k).map Prod.fst = (List.range' j k).map t
  | _, 0 => rfl
  | j, k + 1 => by
    simp only [wardenLane]
    split_ifs <;> simp [wardenLane_fst a t (j + 1) k, List.range'_succ, Function.comp_def]

theorem wardenCode_fst (s : WardenSchedule κ) (a : κ → Arrivals Cell) (c : κ) :
    (wardenCode s a c).map Prod.fst = (List.range (s.cells c)).map (s.bucket c) := by
  rw [List.range_eq_range']
  exact wardenLane_fst _ _ 0 _

/-- An element `j` that `p` keeps is the `n`-th that `p` keeps, `n` the number it keeps before `j`. -/
theorem getElem?_filter_countP_take {α : Type} (p : α → Bool) :
    ∀ (L : List α) (j : ℕ) (hj : j < L.length), p L[j] = true → (L.filter p)[(L.take j).countP p]? = some L[j]
  | x :: xs, 0, _, hp => by simp_all
  | x :: xs, j + 1, hj, hp => by
    have ih := getElem?_filter_countP_take p xs j (by simpa using hj) (by simpa using hp)
    by_cases hx : p x = true
    · simpa [List.filter_cons, hx, List.countP_cons] using ih
    · simpa [List.filter_cons, hx, List.countP_cons] using ih

/-- Of the `i < j`, those with `⌊i/r⌋ = ⌊j/r⌋` number `j mod r`. -/
theorem countP_range_div (r j : ℕ) : (List.range j).countP (fun i => i / r == j / r) = j % r := by
  rcases Nat.eq_zero_or_pos r with rfl | hr
  · simp
  · obtain ⟨q, m, hm, rfl⟩ : ∃ q m, m < r ∧ j = r * q + m := ⟨j / r, j % r, Nat.mod_lt _ hr, (Nat.div_add_mod j r).symm⟩
    have hq : (r * q + m) / r = q := by rw [Nat.mul_add_div hr, Nat.div_eq_of_lt hm, Nat.add_zero]
    rw [hq, Nat.mul_add_mod_self_left, Nat.mod_eq_of_lt hm, List.range_add, List.countP_append, List.countP_map]
    have h1 : (List.range (r * q)).countP (fun i => i / r == q) = 0 := by
      refine List.countP_eq_zero.2 fun i hi => ?_
      have : i / r < q := (Nat.div_lt_iff_lt_mul hr).2 (by rw [Nat.mul_comm]; exact List.mem_range.1 hi)
      simp [Nat.ne_of_lt this]
    have h2 : (List.range m).countP ((fun i => i / r == q) ∘ (r * q + ·)) = m := by
      rw [List.countP_eq_length.2 fun i hi => ?_, List.length_range]
      simp [Nat.mul_add_div hr, Nat.div_eq_of_lt ((List.mem_range.1 hi).trans hm)]
    rw [h1, h2, Nat.zero_add]

theorem filterMap_flatMap_ite [DecidableEq κ] {β : Type} (X : List β) {c : κ} :
    ∀ {conns : List κ}, conns.Nodup → c ∈ conns →
      (conns.flatMap fun d => if d = c then X else []) = X
  | d :: ds, hnd, hc => by
    obtain ⟨hd, hds⟩ := List.nodup_cons.1 hnd
    by_cases hdc : d = c
    · subst hdc
      have : (ds.flatMap fun d' => if d' = d then X else []) = [] :=
        List.flatMap_eq_nil_iff.2 fun d' hd' => by simp [show d' ≠ d from fun h => hd (h ▸ hd')]
      simp [this]
    · have hc' : c ∈ ds := (List.mem_cons.1 hc).resolve_left (Ne.symm hdc)
      simp [hdc, filterMap_flatMap_ite X hds hc']

/-- A connection's frames in a row are its own frames in the bucket. -/
theorem filterMap_bucketFrames [DecidableEq κ] {conns : List κ} (hnd : conns.Nodup) {c : κ} (hc : c ∈ conns)
    (s : WardenSchedule κ) (v : κ → List (ℕ × CellView Cell)) (b : ℕ) :
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
    (s : WardenSchedule κ) (a : κ → Arrivals Cell) (c : κ) :
    lanesOfRows s (bucketFrames conns s (wardenCode s a)) c = wardenCode s a c := by
  have hlen : (wardenCode s a c).length = s.cells c := wardenLane_length _ _ _ _
  have hfst : ∀ j (hj : j < (wardenCode s a c).length), ((wardenCode s a c)[j]).1 = s.bucket c j := by
    intro j hj
    have h := congrArg (fun l => l[j]?) (wardenCode_fst s a c)
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
  have key := getElem?_filter_countP_take (fun x : ℕ × CellView Cell => x.1 == s.bucket c j) (wardenCode s a c) j h2
    (by simp [hfst j h2])
  have hcount : ((wardenCode s a c).take j).countP (fun x => x.1 == s.bucket c j) = j % s.r := by
    have := congrArg (fun l => (l.take j).countP (· == s.bucket c j)) (wardenCode_fst s a c)
    simp only [← List.map_take, List.countP_map, List.take_range, Nat.min_eq_left hj.le] at this
    rw [show (fun x : ℕ × CellView Cell => x.1 == s.bucket c j) = (· == s.bucket c j) ∘ Prod.fst from rfl, this,
      ← countP_range_div s.r j]
    congr 1
    funext i
    simp [WardenSchedule.bucket]
  rw [hcount] at key
  rw [List.getElem?_map, key]
  exact Prod.ext (hfst j h2).symm rfl

/-- A row of the code's cells holds each connection's load in the bucket. -/
theorem length_bucketFrames (conns : List κ) (s : WardenSchedule κ) (a : κ → Arrivals Cell) (b : ℕ) :
    (bucketFrames conns s (wardenCode s a) b).length = (conns.map fun c => laneLoad s c b).sum := by
  simp only [bucketFrames, List.length_flatMap, List.length_map, laneFrames, List.length_append, laneLoad]
  congr 1
  refine List.map_congr_left fun c _ => ?_
  congr 1
  · split_ifs <;> simp
  · rw [← List.countP_eq_length_filter, ← List.countP_eq_length_filter,
      show (fun x : ℕ × CellView Cell => x.1 == b) = (· == b) ∘ Prod.fst from rfl, ← List.countP_map,
      wardenCode_fst, List.countP_map]
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
    (s : WardenSchedule κ) (a : κ → Arrivals Cell) (v : κ → List (ℕ × CellView Cell)) {b : ℕ} (hb : b < n * F.T)
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
theorem read_record [DecidableEq κ] (F : RowsFormat κ Cell L K) (hF : F.Wf) (S n : ℕ) (s : WardenSchedule κ)
    (hs : F.Fits S n s) (a : κ → Arrivals Cell) :
    F.read s (F.record S n s a (wardenCode s a)) = wardenCode s a := by
  obtain ⟨hnd, hall, hdec⟩ := hF
  funext c
  have : F.read s (F.record S n s a (wardenCode s a)) c = lanesOfRows s (bucketFrames F.conns s (wardenCode s a)) c := by
    simp only [read, lanesOfRows]
    refine List.map_congr_left fun j hj => ?_
    rw [frames_record F hdec S n s a _ (hs.1 c j (List.mem_range.1 hj))
      ((length_bucketFrames _ _ _ _).trans_le (hs.2 _))]
  rw [this, lanesOfRows_bucketFrames hnd hall s a c]

/-- **The rows format is a record format of the network certifier**: on the schedules that fit it, its record of the
cells the code sends reads back to them. -/
def tapRecord [DecidableEq κ] (F : RowsFormat κ Cell L K) (hF : F.Wf) (S n : ℕ) :
    TapRecord κ Cell (Fin n → List (Obs L K S)) where
  record := F.record S n
  read := F.read
  fits := F.Fits S n
  read_record s hs a := F.read_record hF S n s hs a

end RowsFormat

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
def fixtureAgrees {n : ℕ} (s : WardenSchedule (Fin n)) (arr : Fin n → Arrivals (Bool × String))
    (rows : List (List (ℕ × String × String))) : Bool :=
  (List.range (rows.length + 8)).all (fun b => (bucketFrames (List.finRange n) s (wardenCode s arr) b).map pyFrame ==
      rows.getD b []) &&
    (List.finRange n).all (fun c => (lanesOfRows s (fun b => (rows.getD b []).filterMap ofPyFrame) c).map pyCell ==
      (wardenCode s arr c).map pyCell) &&
    (List.range (rows.length + 8)).all fun b => (bucketFrames (List.finRange n) s (wardenCode s arr) b).length ≤ 4

/-- `fxQueue`: 3 connections; the Python network certifier's egress record, buckets 0 … 9 (every later bucket empty). -/
def fxQueueSched : WardenSchedule (Fin 3) :=
  { r := 2, D := 3, adm := ![1, 2, 6], cells := ![6, 2, 2] }

/-- `fxQueue`'s arrivals. -/
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
def fxLateSched : WardenSchedule (Fin 1) :=
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
def fxLimitSched : WardenSchedule (Fin 1) :=
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
