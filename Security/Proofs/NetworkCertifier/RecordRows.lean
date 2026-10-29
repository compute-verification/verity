import Definitions.NetworkCertifier.RecordRows

/-!
# The network certifier's rows read back its lanes

`Definitions.NetworkCertifier.RecordRows`. On a schedule that fits the rows format, the lanes read back from the record
of the cells the network certifier's code sends are those cells (`RowsFormat.read_record`), with no schedule given, so
the rows format is a `TapRecord` (`RowsFormat.tapRecord`). A row's frames are its connections' in turn, so a
connection's are its own (`filterMap_bucketFrames`), and the record's slots hold a row's frames in order when the row
fits (`frames_record`). A connection's frames in a bucket are its padding, in the buckets from its admission to
`adm c + D`, then its lane's cells in the bucket; a lane of `k` cells has cell `j` in bucket `t j` (`wardenLane_fst`),
so its cells are in buckets from `adm c + D` on, in order (`wardenCode_sorted`). In bucket order, then, a connection's
frames are its `r·D` padding frames and then its lane, cell by cell (`drop_flatMap_laneFrames`).

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

theorem mem_wardenCode_fst (s : WardenSchedule κ) (a : κ → Arrivals Cell) (c : κ) {x : ℕ × CellView Cell}
    (hx : x ∈ wardenCode s a c) : ∃ j < s.cells c, s.bucket c j = x.1 := by
  have h := List.mem_map_of_mem (f := Prod.fst) hx
  rw [wardenCode_fst] at h
  obtain ⟨j, hj, e⟩ := List.mem_map.1 h
  exact ⟨j, List.mem_range.1 hj, e⟩

/-- A lane's cells are in bucket order. -/
theorem wardenCode_sorted (s : WardenSchedule κ) (a : κ → Arrivals Cell) (c : κ) :
    (wardenCode s a c).Pairwise (fun x y => x.1 ≤ y.1) := by
  have h : ((List.range (s.cells c)).map (s.bucket c)).Pairwise (· ≤ ·) :=
    List.pairwise_map.2 (List.pairwise_lt_range.imp fun hij =>
      Nat.add_le_add_left (Nat.div_le_div_right (Nat.le_of_lt hij)) _)
  rw [← wardenCode_fst s a c] at h
  exact List.pairwise_map.1 h

/-- A bucket's cells of a lane, each tagged with the bucket, are those cells. -/
theorem map_pair_filter_fst {β : Type} (b : ℕ) (l : List (ℕ × β)) :
    ((l.filter fun x => x.1 == b).map Prod.snd).map (b, ·) = l.filter fun x => x.1 == b := by
  rw [List.map_map]
  refine (List.map_congr_left fun x hx => ?_).trans (List.map_id _)
  have : x.1 = b := by simpa using (List.mem_filter.1 hx).2
  exact Prod.ext this.symm rfl

/-- Over the buckets, frames that end before bucket `m` and frames that start at it come apart. -/
theorem flatMap_append_sep {β : Type} (f g : ℕ → List β) (m : ℕ) (hf : ∀ b, m ≤ b → f b = [])
    (hg : ∀ b, b < m → g b = []) :
    ∀ N, ((List.range N).flatMap fun b => f b ++ g b) = (List.range N).flatMap f ++ (List.range N).flatMap g
  | 0 => by simp
  | N + 1 => by
    simp only [List.range_succ, List.flatMap_append, List.flatMap_singleton]
    rw [flatMap_append_sep f g m hf hg N]
    by_cases h : m ≤ N
    · rw [hf N h]
      simp
    · have hA : (List.range N).flatMap g = [] :=
        List.flatMap_eq_nil_iff.2 fun b hb => hg b (by have := List.mem_range.1 hb; omega)
      rw [hg N (by omega), hA]
      simp

/-- A list in key order with every key at most `N`: those below `N`, then those at `N`. -/
theorem filter_lt_append_filter_eq {β : Type} (N : ℕ) :
    ∀ l : List (ℕ × β), l.Pairwise (fun x y => x.1 ≤ y.1) → (∀ x ∈ l, x.1 ≤ N) →
      l.filter (fun x => decide (x.1 < N)) ++ l.filter (fun x => x.1 == N) = l
  | [], _, _ => rfl
  | x :: xs, hp, hN => by
    obtain ⟨hx, hxs⟩ := List.pairwise_cons.1 hp
    have hxN := hN x (by simp)
    rcases Nat.lt_or_eq_of_le hxN with h | h
    · have ih := filter_lt_append_filter_eq N xs hxs fun y hy => hN y (List.mem_cons_of_mem _ hy)
      simp [h, Nat.ne_of_lt h, ih]
    · have h1 : xs.filter (fun x => decide (x.1 < N)) = [] :=
        List.filter_eq_nil_iff.2 fun y hy => by
          have := hx y hy
          simp only [decide_eq_true_eq]
          omega
      have h2 : xs.filter (fun x => x.1 == N) = xs :=
        List.filter_eq_self.2 fun y hy => by
          have := hx y hy
          have := hN y (List.mem_cons_of_mem _ hy)
          exact beq_iff_eq.2 (by omega)
      simp [h, h1, h2]

/-- **A list in key order, bucket by bucket, is the list**, when every key is a bucket before `N`. -/
theorem flatMap_filter_fst {β : Type} :
    ∀ (N : ℕ) (l : List (ℕ × β)), l.Pairwise (fun x y => x.1 ≤ y.1) → (∀ x ∈ l, x.1 < N) →
      ((List.range N).flatMap fun b => l.filter fun x => x.1 == b) = l
  | 0, [], _, _ => by simp
  | 0, x :: _, _, hN => absurd (hN x (by simp)) (Nat.not_lt_zero _)
  | N + 1, l, hp, hN => by
    have hsub : ∀ b ∈ List.range N, (l.filter fun x => x.1 == b) =
        (l.filter fun x => decide (x.1 < N)).filter fun x => x.1 == b := by
      intro b hb
      have hbN := List.mem_range.1 hb
      rw [List.filter_filter]
      refine List.filter_congr fun x _ => ?_
      show (x.1 == b) = ((x.1 == b) && decide (x.1 < N))
      cases hxb : (x.1 == b)
      · rfl
      · have hx : x.1 = b := by simpa using hxb
        exact (decide_eq_true (by omega)).symm
    simp only [List.range_succ, List.flatMap_append, List.flatMap_singleton]
    rw [List.flatMap_congr hsub, flatMap_filter_fst N (l.filter fun x => decide (x.1 < N))
      (hp.sublist List.filter_sublist) fun x hx => by simpa using (List.mem_filter.1 hx).2]
    exact filter_lt_append_filter_eq N l hp fun x hx => Nat.lt_succ_iff.1 (hN x hx)

/-- The padding frames a connection sends from its admission to `adm + D`, `r` a bucket, number `r` a bucket of
those before `N`. -/
theorem length_pre {β : Type} (r D adm : ℕ) (x : β) :
    ∀ N, ((List.range N).flatMap fun b =>
      (if adm ≤ b ∧ b < adm + D then List.replicate r x else []).map (b, ·)).length = r * (min N (adm + D) - adm)
  | 0 => by simp
  | N + 1 => by
    simp only [List.range_succ, List.flatMap_append, List.flatMap_singleton, List.length_append,
      length_pre r D adm x N]
    split_ifs with h
    · have : min (N + 1) (adm + D) - adm = min N (adm + D) - adm + 1 := by omega
      simp only [List.length_map, List.length_replicate]
      rw [this]
      exact (Nat.mul_succ r _).symm
    · have : min (N + 1) (adm + D) - adm = min N (adm + D) - adm := by omega
      simp only [List.length_map, List.length_nil, Nat.add_zero]
      rw [this]

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

/-- **A connection's frames, in bucket order past its padding, are its lane's cells**: over buckets that hold its lane,
its frames, each with its bucket, are its `r·D` padding frames and then the cells the code sends on its lane. -/
theorem drop_flatMap_laneFrames (s : WardenSchedule κ) (a : κ → Arrivals Cell) (c : κ) (N : ℕ)
    (hN : ∀ j, j < s.cells c → s.bucket c j < N) :
    ((List.range N).flatMap fun b => (laneFrames s (wardenCode s a) c b).map (b, ·)).drop (s.r * s.D) =
      wardenCode s a c := by
  have hpad : ∀ b, s.adm c + s.D ≤ b →
      (if s.adm c ≤ b ∧ b < s.adm c + s.D then List.replicate s.r (CellView.pad : CellView Cell) else []).map
        (b, ·) = [] := by
    intro b hb
    have : ¬(s.adm c ≤ b ∧ b < s.adm c + s.D) := fun h => absurd h.2 (Nat.not_lt.2 hb)
    rw [ite_eq_right this]
    rfl
  have hcel : ∀ b, b < s.adm c + s.D → (wardenCode s a c).filter (fun x => x.1 == b) = [] := by
    intro b hb
    refine List.filter_eq_nil_iff.2 fun x hx => ?_
    obtain ⟨j, -, hj⟩ := mem_wardenCode_fst s a c hx
    rw [← hj]
    simp only [beq_iff_eq, WardenSchedule.bucket]
    have := Nat.le_add_right (s.adm c + s.D) (j / s.r)
    omega
  have hkey : ∀ x ∈ wardenCode s a c, x.1 < N := fun x hx => by
    obtain ⟨j, hj, e⟩ := mem_wardenCode_fst s a c hx
    rw [← e]
    exact hN j hj
  have hsplit : ((List.range N).flatMap fun b => (laneFrames s (wardenCode s a) c b).map (b, ·)) =
      ((List.range N).flatMap fun b =>
        (if s.adm c ≤ b ∧ b < s.adm c + s.D then List.replicate s.r (CellView.pad : CellView Cell) else []).map
          (b, ·)) ++ wardenCode s a c := by
    simp only [laneFrames, List.map_append, map_pair_filter_fst]
    rw [flatMap_append_sep _ _ (s.adm c + s.D) hpad hcel N,
      flatMap_filter_fst N (wardenCode s a c) (wardenCode_sorted s a c) hkey]
  rw [hsplit]
  have hlen := length_pre s.r s.D (s.adm c) (CellView.pad : CellView Cell) N
  rcases Nat.eq_zero_or_pos (s.cells c) with hc0 | hpos
  · have hnil : wardenCode s a c = [] :=
      List.eq_nil_of_length_eq_zero ((wardenLane_length (a c) (s.bucket c) 0 (s.cells c)).trans hc0)
    rw [hnil, List.append_nil]
    exact List.drop_eq_nil_of_le (by rw [hlen]; exact Nat.mul_le_mul_left _ (by omega))
  · have h0 := hN 0 hpos
    simp only [WardenSchedule.bucket, Nat.zero_div, Nat.add_zero] at h0
    exact List.drop_left' (by rw [hlen, show min N (s.adm c + s.D) - s.adm c = s.D by omega])

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

/-- **The rows format reads back the code's cells** on every schedule that fits it, from the record alone. -/
theorem read_record [DecidableEq κ] (F : RowsFormat κ Cell L K) (hF : F.Wf) (S n : ℕ) (s : WardenSchedule κ)
    (hs : F.Fits S n s) (a : κ → Arrivals Cell) :
    F.read (F.record S n s a (wardenCode s a)) = wardenCode s a := by
  obtain ⟨hnd, hall, hdec⟩ := hF
  obtain ⟨hr, hD, hin, hS⟩ := hs
  funext c
  have hrow : ∀ b ∈ List.range (n * F.T),
      ((F.frames (F.record S n s a (wardenCode s a)) b).filterMap fun f => if f.1 = c then some f.2 else none).map
        (b, ·) = (laneFrames s (wardenCode s a) c b).map (b, ·) := fun b hb => by
    rw [frames_record F hdec S n s a _ (List.mem_range.1 hb) ((length_bucketFrames _ _ _ _).trans_le (hS _)),
      filterMap_bucketFrames hnd (hall c)]
  have : F.read (F.record S n s a (wardenCode s a)) c =
      ((List.range (n * F.T)).flatMap fun b => (laneFrames s (wardenCode s a) c b).map (b, ·)).drop (s.r * s.D) := by
    simp only [read, lanesOfRows, hr, hD]
    congr 1
    exact List.flatMap_congr hrow
  rw [this]
  exact drop_flatMap_laneFrames s a c _ (hin c)

/-- **The rows format is a record format of the network certifier**: on the schedules that fit it, its record of the
cells the code sends reads back to them, with no schedule given. -/
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
    (List.finRange n).all (fun c =>
      (lanesOfRows s.r s.D (rows.length + 8) (fun b => (rows.getD b []).filterMap ofPyFrame) c).map pyCell ==
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
