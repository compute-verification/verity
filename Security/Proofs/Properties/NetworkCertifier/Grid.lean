import Proofs.Properties.NetworkCertifier.Tree

/-!
# A record head determines its grid

`record_head` refuses a row that isn't left-packed, so a left-packed row is its frames followed by empty slots, and a
grid's rows are determined by their real frames and their counts (`grid_inj`). With the frame tree's binding
(`Tree.framesRoot_inj`), two grids with one head are one (`recordHead_grid_inj`).
-/

open NetworkCheck hiding Params
open NetworkCheck.Check
open Flock
open NetworkCertifier (ClockSync LogicalFrame expectedRow)

namespace Properties.NetworkCertifier.GridProofs

open Properties.NetworkCertifier Properties.NetworkCertifier.Bytes Properties.NetworkCertifier.Tree

/-- A row whose first `m` slots, `m` its count, are filled is its frames and then empty slots. -/
theorem packed {α : Type} : ∀ (l : List (Option α)) (m : Nat), (l.take m).all Option.isSome = true →
    m = (l.filter Option.isSome).length →
    l = (l.filterMap id).map some ++ List.replicate (l.length - m) none
  | [], m, _, _ => by simp
  | some a :: t, m, h, hm => by
    simp only [List.filter_cons, Option.isSome_some, ite_true, List.length_cons] at hm
    obtain ⟨m, rfl⟩ : ∃ k, m = k + 1 := ⟨m - 1, by omega⟩
    rw [List.take_succ_cons, List.all_cons] at h
    simp only [Option.isSome_some, Bool.true_and] at h
    have ih := packed t m h (by omega)
    rw [show (some a :: t).filterMap id = a :: t.filterMap id from rfl, List.map_cons, List.cons_append,
      List.length_cons, show t.length + 1 - (m + 1) = t.length - m by omega, ← ih]
  | none :: t, m, h, hm => by
    cases m with
    | zero =>
      simp only [List.filter_cons, Option.isSome_none, Bool.false_eq_true, ite_false] at hm
      have ih := packed t 0 (by simp) hm
      have hlen := congrArg List.length ih
      simp only [List.length_append, List.length_map, List.length_replicate, Nat.sub_zero] at hlen
      have hf : t.filterMap id = [] := List.eq_nil_of_length_eq_zero (by omega)
      rw [hf] at ih
      rw [show (none :: t).filterMap id = t.filterMap id from rfl, hf]
      simp only [List.map_nil, List.nil_append, Nat.sub_zero, List.length_cons, List.replicate_succ] at ih ⊢
      exact congrArg (List.cons none) ih
    | succ m => simp at h

theorem filterMap_length {α : Type} (l : List (Option α)) :
    (l.filterMap id).length = (l.filter Option.isSome).length := by
  induction l with
  | nil => rfl
  | cons a t ih =>
    cases a with
    | none => exact ih
    | some _ => exact congrArg Nat.succ ih

/-- A row's count: its filled slots. -/
def count (row : Array (Option ByteArray)) : Nat := (row.filter Option.isSome).size

theorem count_eq (row : Array (Option ByteArray)) : count row = (row.toList.filter Option.isSome).length := by
  have := congrArg List.length (@Array.toList_filter _ Option.isSome row)
  rw [Array.length_toList] at this
  exact this

theorem rowFrames_cons {r : Nat} {row : Array (Option ByteArray)} {rows : List (Array (Option ByteArray))}
    {F : Array ByteArray} (h : rowFrames r (row :: rows) = .ok F) :
    row.size = r ∧ (row.extract 0 (count row)).all Option.isSome = true ∧
      ∃ F₁, rowFrames r rows = .ok F₁ ∧ F = row.filterMap id ++ F₁ := by
  unfold rowFrames at h
  by_cases hs : row.size ≠ r
  · rw [ite_eq_left hs] at h
    exact (error_ne_ok h).elim
  rw [ite_eq_right hs] at h
  by_cases hp : (row.extract 0 (row.filter Option.isSome).size).all Option.isSome = true
  · rw [ite_eq_left hp] at h
    refine ⟨by omega, hp, ?_⟩
    cases hr : rowFrames r rows with
    | error _ => rw [hr] at h; exact (error_ne_ok h).elim
    | ok F₁ => rw [hr] at h; exact ⟨F₁, rfl, (Except.ok.inj h).symm⟩
  · rw [ite_eq_right hp] at h
    exact (error_ne_ok h).elim

theorem rowFrames_nil {r : Nat} {F : Array ByteArray} (h : rowFrames r [] = .ok F) : F = #[] :=
  (Except.ok.inj h).symm

/-- A left-packed row of `r` slots is its frames and then empty slots. -/
theorem row_eq {row : Array (Option ByteArray)} (hp : (row.extract 0 (count row)).all Option.isSome = true) :
    row.toList = (row.filterMap id).toList.map some ++ List.replicate (row.size - count row) none := by
  have := packed row.toList (count row) (by
    rw [← Array.all_toList, Array.toList_extract] at hp
    simpa [List.extract] using hp) (count_eq row)
  rw [Array.toList_filterMap, ← Array.length_toList]
  exact this

theorem rowFrames_inj {r : Nat} : ∀ {rows rows' : List (Array (Option ByteArray))} {F : Array ByteArray},
    rowFrames r rows = .ok F → rowFrames r rows' = .ok F → rows.map count = rows'.map count → rows = rows'
  | [], [], _, _, _, _ => rfl
  | [], _ :: _, _, _, _, hc => by simp at hc
  | _ :: _, [], _, _, _, hc => by simp at hc
  | row :: rows, row' :: rows', F, h, h', hc => by
    obtain ⟨hs, hp, F₁, h₁, rfl⟩ := rowFrames_cons h
    obtain ⟨hs', hp', F₁', h₁', e⟩ := rowFrames_cons h'
    simp only [List.map_cons, List.cons.injEq] at hc
    have hl : (row.filterMap id).toList.length = (row'.filterMap id).toList.length := by
      rw [Array.toList_filterMap, Array.toList_filterMap, filterMap_length, filterMap_length]
      rw [← count_eq, ← count_eq]
      exact hc.1
    have e2 := congrArg Array.toList e
    simp only [Array.toList_append] at e2
    obtain ⟨ef, eF⟩ := List.append_inj e2 hl
    have er : row = row' := by
      apply Array.toList_inj.mp
      rw [row_eq hp, row_eq hp', ef, hs, hs', hc.1]
    rw [er, rowFrames_inj h₁ (by rw [Array.toList_inj.mp eF]; exact h₁') hc.2]

theorem realFrames_ok {grid : Array (Array (Option ByteArray))} {g : Grid} {F : Array ByteArray}
    (h : realFrames grid g = .ok F) : grid.size = g.buckets ∧ rowFrames g.allocation grid.toList = .ok F := by
  unfold realFrames at h
  by_cases hs : grid.size ≠ g.buckets
  · rw [ite_eq_left hs] at h
    exact (error_ne_ok h).elim
  rw [ite_eq_right hs] at h
  exact ⟨by omega, h⟩

/-- **A grid is its real frames and its counts.** -/
theorem grid_inj {grid grid' : Array (Array (Option ByteArray))} {g : Grid} {F : Array ByteArray}
    (h : realFrames grid g = .ok F) (h' : realFrames grid' g = .ok F)
    (hc : grid.map (fun row => ((row.filter Option.isSome).size : Int)) =
      grid'.map (fun row => ((row.filter Option.isSome).size : Int))) : grid = grid' := by
  apply Array.toList_inj.mp
  refine rowFrames_inj (realFrames_ok h).2 (realFrames_ok h').2 ?_
  have := congrArg Array.toList hc
  simp only [Array.toList_map] at this
  have e := congrArg (List.map Int.toNat) this
  simp only [List.map_map, Function.comp_def, Int.toNat_natCast] at e
  exact e

/-- What `record_head` makes of a grid. -/
theorem recordHead_ok {l : String} {d : Direction} {w : Int} {grid : Array (Array (Option ByteArray))}
    {st : Option String} {g : Grid} {par : ByteArray} {dl : Bool} {h : RecordHead}
    (e : recordHead l d w grid st g par dl = .ok h) :
    ∃ F, realFrames grid g = .ok F ∧ framesRoot F g = .ok h.root ∧
      h.counts = grid.map (fun row => ((row.filter Option.isSome).size : Int)) ∧ h.status = st := by
  unfold recordHead at e
  cases hF : realFrames grid g with
  | error _ => rw [hF] at e; exact (error_ne_ok e).elim
  | ok F =>
    cases hr : framesRoot F g with
    | error _ =>
      simp only [hF, hr, bind, Except.bind] at e
      exact (error_ne_ok e).elim
    | ok rt =>
      simp only [hF, hr, bind, Except.bind, pure, Except.pure] at e
      cases Except.ok.inj e
      exact ⟨F, rfl, hr, rfl, rfl⟩

/-- The fields `record_head` copies into the head. -/
theorem recordHead_head {l : String} {d : Direction} {w : Int} {grid : Array (Array (Option ByteArray))}
    {st : Option String} {g : Grid} {par : ByteArray} {dl : Bool} {h : RecordHead}
    (e : recordHead l d w grid st g par dl = .ok h) :
    h.link = l ∧ h.direction = d ∧ h.window = w ∧ h.status = st ∧ h.deadline = dl ∧ h.parameters = par := by
  unfold recordHead at e
  cases hF : realFrames grid g with
  | error _ => rw [hF] at e; exact (error_ne_ok e).elim
  | ok F =>
    cases hr : framesRoot F g with
    | error _ =>
      simp only [hF, hr, bind, Except.bind] at e
      exact (error_ne_ok e).elim
    | ok rt =>
      simp only [hF, hr, bind, Except.bind, pure, Except.pure] at e
      cases Except.ok.inj e
      exact ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- **One head, one grid**: two grids `record_head` makes one head of, where SHA-512 has no collision on the strings
both frame trees hash, are one, and so are their statuses. -/
theorem recordHead_grid_inj {X : ByteArray → Prop} (hX : CF X) {l l' : String} {d d' : Direction} {w w' : Int}
    {grid grid' : Array (Array (Option ByteArray))} {st st' : Option String} {g : Grid} {par par' : ByteArray}
    {dl dl' : Bool} {h : RecordHead}
    (e : recordHead l d w grid st g par dl = .ok h) (e' : recordHead l' d' w' grid' st' g par' dl' = .ok h)
    (hs : ∀ x, TreeString g grid x → X x) (hs' : ∀ x, TreeString g grid' x → X x) : grid = grid' ∧ st = st' := by
  obtain ⟨F, hF, hr, hc, rfl⟩ := recordHead_ok e
  obtain ⟨F', hF', hr', hc', hst⟩ := recordHead_ok e'
  have eF := framesRoot_inj hX hr hr' (fun x hx => hs x ⟨F, hF, hx⟩) (fun x hx => hs' x ⟨F', hF', hx⟩)
  subst eF
  exact ⟨grid_inj hF hF' (hc.symm.trans hc'), hst⟩

/-! ## From grids to rows -/

/-- Two rows of bits with one `bytesGrid`, every slot whole bytes, are one at every bucket of the window. -/
theorem bytesGrid_inj {r T : Nat} {row row' : Nat → Fin r → Option (List Bool)} {w : Nat}
    (h : bytesGrid T row w = bytesGrid T row' w)
    (hb : ∀ t s f, row t s = some f → f.length % 8 = 0) (hb' : ∀ t s f, row' t s = some f → f.length % 8 = 0) :
    ∀ i < T, row (w * T + i) = row' (w * T + i) := by
  intro i hi
  funext s
  have e := congrArg (fun G : Array (Array (Option ByteArray)) => (G[i]?).bind (·[s.1]?)) h
  simp only [bytesGrid, Array.getElem?_map, List.getElem?_toArray, List.getElem?_range hi, Option.map_some,
    Option.bind_some, List.getElem?_ofFn, s.2, dite_true] at e
  simp only [Fin.eta, Option.some.injEq] at e
  cases hx : row (w * T + i) s with
  | none =>
    cases hy : row' (w * T + i) s with
    | none => rfl
    | some y => rw [hx, hy] at e; cases e
  | some x =>
    cases hy : row' (w * T + i) s with
    | none => rw [hx, hy] at e; cases e
    | some y =>
      rw [hx, hy] at e
      simp only [Option.map_some, Option.some.injEq] at e
      rw [bytesOfBits_inj (hb _ _ _ hx) (hb' _ _ _ hy) e]

/-- A slot of the Program's row holds a payload of the link's frames, or nothing. -/
theorem expectedRow_slot {S T W : Nat} {syncs : Fin W → ClockSync} {frames : List LogicalFrame} {t : Nat}
    {s : Fin S} {f : List Bool} (h : expectedRow S T W syncs frames t s = some f) :
    ∃ k, f = NetworkCertifier.payloads frames k := by
  unfold expectedRow NetworkCertifier.fifoGrid NetworkCertifier.fill at h
  split at h
  · exact ⟨_, (Option.some.inj h).symm⟩
  · cases h

theorem payloads_length {frames : List LogicalFrame} {n : Nat} (hf : ∀ f ∈ frames, f.payload.length = n) (k : Nat) :
    (NetworkCertifier.payloads frames k).length = n ∨ NetworkCertifier.payloads frames k = [] := by
  unfold NetworkCertifier.payloads
  cases hk : frames[k]? with
  | none => right; rfl
  | some f => left; exact hf f (List.mem_of_getElem? hk)

end Properties.NetworkCertifier.GridProofs
