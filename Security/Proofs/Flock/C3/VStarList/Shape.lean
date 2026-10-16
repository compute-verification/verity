import Proofs.Flock.Recursive.HoloExec
import Proofs.Flock.VBridge.Family.Structure

/-!
# Piece (h): the list's R at `mReg`, and `hp` as the inner shape's `sparse` entry

The verifier's list (`RecVStar.statements`) is V*'s statements (`VStar.statements`, `rec-open` and `rec-residuals`),
then candidate 3's (`candidate3Statements`): RecSparse at the inner shape's `sparse` entry, then R's levels and the
parts of R's opening at META's `stacked`. `checkCandidate3` refuses a list with an opening off R's shape, and
`candidate3Statements` RecSparse without R and an R off the two claims or off the bucket's `table_m`. So:

* `SparseIs ish hp`: `hp` is the verifier's own, the inner shape's `sparse` entry (`[NU, ELL, S_LOG, T_LOG]`).
* `sparse_of_list`: the list holds RecSparse at that entry; `stacked_of_list`: with R, it holds each of R's levels at
  `fast100(m)` (`recOpenStacked`); `stacked_of_candidate3`: and the inner shape names R at `(table_m, STACKED_CLAIMS)`.
* `checkInnerStmt_ok`: where check 5 passes `checkInnerStmt`, the inner shape names a sparse entry and its `m`,
  `k_log` and region count are the inner statement's.
* `mReg_of_list`: each of the list's `rec-opening` statements is at `m = mReg` and `STACKED_CLAIMS` claims, where
  `mReg` is the model's (`(holoAt st hp).cls.shape.mReg`, padded to `PAD_TO` as the verifier's `tableM` is) and `ℓ` is
  in the class's range.
-/

namespace FlockC3.VStarList

theorem eok_bind {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : (x >>= f) = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error e => cases h
  | ok a => exact ⟨a, rfl, h⟩

/-- **The verifier's list**: V*'s statements at the inner shape, then candidate 3's, and `checkCandidate3` passes. -/
theorem statements_ok {i : Flock.RecVStar.Inner} {ss : Array VStar.Stmt} (h : Flock.RecVStar.statements i = .ok ss) :
    ∃ vsh A C, VStar.statements vsh = .ok A ∧ Flock.RecVStar.candidate3Statements i = .ok C ∧ ss = A ++ C ∧
      Flock.RecVStar.checkCandidate3 i (A ++ C) = .ok () := by
  unfold Flock.RecVStar.statements at h
  dsimp only at h
  split at h
  · cases h
  split at h
  rotate_left
  · cases h
  split at h
  · cases h
  obtain ⟨ps, hps, h⟩ := eok_bind h
  obtain ⟨gs, hgs, h⟩ := eok_bind h
  split at h
  · cases h
  obtain ⟨A, hA, h⟩ := eok_bind h
  obtain ⟨C, hC, h⟩ := eok_bind h
  obtain ⟨u, hu, h⟩ := eok_bind h
  cases h
  exact ⟨_, A, C, hA, hC, rfl, hu⟩

/-- **Candidate 3's statements**: none without a `sparse` entry; with one, RecSparse at it first, and R at the two
claims and the bucket's `table_m` (which a `sparse` entry needs, `stacked_of_candidate3`). -/
theorem candidate3_ok {i : Flock.RecVStar.Inner} {C : Array VStar.Stmt}
    (h : Flock.RecVStar.candidate3Statements i = .ok C) :
    (i.sparse = none → C = #[]) ∧ ∀ sp, i.sparse = some sp →
      ∃ rest, C = #[.sparse sp (Flock.RecVStar.hRound i.m i.kLog)] ++ rest ∧ (i.stacked = none → rest = #[]) ∧
        ∀ r, i.stacked = some r → r.claims = Flock.RecVStar.STACKED_CLAIMS ∧ r.m = Flock.RecVStar.tableM sp := by
  unfold Flock.RecVStar.candidate3Statements at h
  constructor
  · intro hn
    rw [hn] at h
    cases hr : i.stacked with
    | none => rw [hr] at h; cases h; rfl
    | some r => rw [hr] at h; cases h
  · intro sp hsp
    rw [hsp] at h
    dsimp only at h
    split at h
    · cases h
    cases hr : i.stacked with
    | none =>
      rw [hr] at h
      cases h
    | some r =>
      rw [hr] at h
      dsimp only at h
      split at h
      · cases h
      rename_i hc
      split at h
      · cases h
      rename_i hm
      split at h
      rotate_left
      · cases h
      obtain ⟨x, -, h⟩ := eok_bind h
      cases h
      refine ⟨_, Array.append_assoc, (fun hn => nomatch hn), fun r' hr' => ?_⟩
      cases hr'
      exact ⟨by simpa using hc, by simpa using hm⟩

/-- **A `sparse` entry comes with R**: `candidate3Statements` refuses RecSparse without R's shape, and R off the two
claims or the bucket's `table_m`, so where it passes R is META's `stacked` entry `(table_m, STACKED_CLAIMS)`. -/
theorem stacked_of_candidate3 {i : Flock.RecVStar.Inner} {C : Array VStar.Stmt}
    (h : Flock.RecVStar.candidate3Statements i = .ok C) {sp : Flock.RecSparse.Shape} (hsp : i.sparse = some sp) :
    i.stacked = some ⟨Flock.RecVStar.tableM sp, Flock.RecVStar.STACKED_CLAIMS⟩ := by
  cases hr : i.stacked with
  | none =>
    unfold Flock.RecVStar.candidate3Statements at h
    rw [hsp] at h
    dsimp only at h
    split at h
    · cases h
    rw [hr] at h
    cases h
  | some r =>
    obtain ⟨-, -, -, hst⟩ := (candidate3_ok h).2 sp hsp
    obtain ⟨e1, e2⟩ := hst r hr
    obtain ⟨rm, rc⟩ := r
    simp only at e1 e2
    subst e1 e2
    rfl

/-- **The inner shape is the inner statement's** where `checkInnerStmt` passes (check 5 at candidate 3's session,
`RecCoins.Inner.load`): it names a RecSparse shape, and its `m`, `k_log` and region count are the statement's. -/
theorem checkInnerStmt_ok {i : Flock.RecVStar.Inner} {st : Flock.Stmt}
    (h : Flock.RecVStar.checkInnerStmt i st = .ok ()) :
    i.sparse.isSome = true ∧ i.m = st.m ∧ i.kLog = st.c.kLog ∧ i.regions = st.regions.size := by
  unfold Flock.RecVStar.checkInnerStmt Flock.RecVStar.checkInnerShape at h
  dsimp only at h
  cases hs : i.sparse with
  | none =>
    rw [hs] at h
    cases h
  | some sp =>
    rw [hs] at h
    dsimp only at h
    split at h
    · cases h
    split at h
    · cases h
    rename_i _ hshape
    simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, Decidable.not_not] at hshape
    exact ⟨rfl, hshape.1.1, hshape.1.2, hshape.2⟩

/-- **An accepted candidate-3 shape has one sparse type**: `checkInnerStmt` refuses nonzero `T_LOG`, so the private
type row is uniquely zero and needs no extra verifier input. -/
theorem tLog_zero_of_checkInnerStmt {i : Flock.RecVStar.Inner} {st : Flock.Stmt}
    (h : Flock.RecVStar.checkInnerStmt i st = .ok ()) {sp : Flock.RecSparse.Shape}
    (hs : i.sparse = some sp) : sp.tLog = 0 := by
  unfold Flock.RecVStar.checkInnerStmt Flock.RecVStar.checkInnerShape at h
  rw [hs] at h
  dsimp only at h
  by_cases ht : sp.tLog = 0
  · exact ht
  · have hne : (sp.tLog != 0) = true := by simp [ht]
    simp only [hne, ↓reduceIte] at h
    cases h

/-- **A loop that runs through passes each element**: where each step that succeeds continues and holds `P` of its
element, a loop that succeeds holds `P` of every element. -/
theorem forIn_ok_all {α ε : Type} (P : α → Prop) {f : α → Unit → Except ε (ForInStep Unit)}
    (hf : ∀ a r, f a () = .ok r → P a ∧ r = .yield ()) :
    ∀ (l : List α) {u : Unit}, forIn l () f = .ok u → ∀ a ∈ l, P a
  | [], _, _, a, ha => by cases ha
  | b :: l, u, h, a, ha => by
    rw [List.forIn_cons] at h
    obtain ⟨st, hst, h⟩ := eok_bind h
    obtain ⟨hb, rfl⟩ := hf b st hst
    cases ha with
    | head => exact hb
    | tail _ ha => exact forIn_ok_all P hf l h a ha

/-- **`checkCandidate3` holds every opening at R's shape**: with META's `stacked` entry `r`, each `rec-opening` statement
of a list it passes is at `r`'s `m` and claims. -/
theorem check_openings {i : Flock.RecVStar.Inner} {ss : Array VStar.Stmt} {r : Flock.RecVStar.StackedShape}
    (h : Flock.RecVStar.checkCandidate3 i ss = .ok ()) (hr : i.stacked = some r) {m c j : ℕ}
    {S : Flock.RecOpen.GfS} {ports : List (String × ℕ)} {nv : ℕ}
    (hs : VStar.Stmt.opening m c j S ports nv ∈ ss) : m = r.m ∧ c = r.claims := by
  unfold Flock.RecVStar.checkCandidate3 at h
  rw [hr] at h
  dsimp only at h
  split at h
  split at h
  cases h
  all_goals
    split at h
    rotate_left
    · cases h
    obtain ⟨_, -, h⟩ := eok_bind h
    split at h
    · cases h
    obtain ⟨_, hfor, -⟩ := eok_bind h
    rw [← Array.forIn_toList] at hfor
    refine forIn_ok_all (fun s => ∀ m c j S ports nv, s = .opening m c j S ports nv → m = r.m ∧ c = r.claims) ?_ _ hfor
      _ (Array.mem_toList_iff.2 hs) m c j S ports nv rfl
    intro a st hst
    cases a
    all_goals dsimp only at hst
    case opening m' c' j' S' ports' nv' =>
      split at hst
      · cases hst
      rename_i hne
      cases hst
      refine ⟨fun m c j S ports nv he => ?_, rfl⟩
      cases he
      simpa [Bool.or_eq_true, bne_iff_ne] using hne
    all_goals exact ⟨(fun _ _ _ _ _ _ he => nomatch he), by cases hst; rfl⟩

/-- **The list's openings are R's**: each `rec-opening` statement of the verifier's list is at the bucket's `table_m`
of the inner shape's `sparse` entry, and at the two reps' claims. -/
theorem opening_of_list {i : Flock.RecVStar.Inner} {ss : Array VStar.Stmt} (h : Flock.RecVStar.statements i = .ok ss)
    {m c j : ℕ} {S : Flock.RecOpen.GfS} {ports : List (String × ℕ)} {nv : ℕ}
    (hs : VStar.Stmt.opening m c j S ports nv ∈ ss) :
    ∃ sp, i.sparse = some sp ∧ m = Flock.RecVStar.tableM sp ∧ c = Flock.RecVStar.STACKED_CLAIMS := by
  obtain ⟨vsh, A, C, hA, hC, rfl, hck⟩ := statements_ok h
  have hAo : VStar.Stmt.opening m c j S ports nv ∉ A := by
    obtain ⟨s, sp, ps, -, -, -, rfl⟩ := FlockVBridge.Family.statements_eq hA
    simp
  have hCo : VStar.Stmt.opening m c j S ports nv ∈ C := by
    rcases Array.mem_append.1 hs with h' | h'
    · exact absurd h' hAo
    · exact h'
  obtain ⟨hnone, hsome⟩ := candidate3_ok hC
  cases hsp : i.sparse with
  | none =>
    rw [hnone hsp] at hCo
    simp at hCo
  | some sp =>
    obtain ⟨rest, hCe, hst0, hst⟩ := hsome sp hsp
    cases hr : i.stacked with
    | none =>
      rw [hCe, hst0 hr] at hCo
      simp at hCo
    | some r =>
      obtain ⟨hcl, hm⟩ := hst r hr
      obtain ⟨e1, e2⟩ := check_openings hck hr hs
      exact ⟨sp, rfl, e1.trans hm, e2.trans hcl⟩

/-- **The list holds RecSparse at the inner shape's `sparse` entry**, its `h` the table's round `hRound`. -/
theorem sparse_of_list {i : Flock.RecVStar.Inner} {ss : Array VStar.Stmt} (h : Flock.RecVStar.statements i = .ok ss)
    {sp : Flock.RecSparse.Shape} (hsp : i.sparse = some sp) :
    VStar.Stmt.sparse sp (Flock.RecVStar.hRound i.m i.kLog) ∈ ss := by
  obtain ⟨vsh, A, C, -, hC, rfl, -⟩ := statements_ok h
  obtain ⟨rest, hCe, -⟩ := (candidate3_ok hC).2 sp hsp
  rw [hCe]
  simp

/-- **The list holds R's levels**: with R (`i.stacked`), each level `l` of `fast100`'s schedule at R's `m`, as
`recOpenStacked l (2^lanes) (d − cap) q`. -/
theorem stacked_of_list {i : Flock.RecVStar.Inner} {ss : Array VStar.Stmt} (h : Flock.RecVStar.statements i = .ok ss)
    {r : Flock.RecVStar.StackedShape} (hr : i.stacked = some r) :
    ∃ sch, Flock.fast100 r.m = some sch ∧ ∀ l < sch.levels.size,
      VStar.Stmt.recOpenStacked l (2 ^ (sch.levels[l]!).lanes) ((sch.levels[l]!).d - (sch.levels[l]!).cap)
        (sch.levels[l]!).q ∈ ss := by
  obtain ⟨vsh, A, C, -, hC, rfl, -⟩ := statements_ok h
  unfold Flock.RecVStar.candidate3Statements at hC
  cases hsp : i.sparse with
  | none =>
    rw [hsp, hr] at hC
    cases hC
  | some sp =>
    rw [hsp] at hC
    dsimp only at hC
    split at hC
    · cases hC
    rw [hr] at hC
    dsimp only at hC
    split at hC
    · cases hC
    split at hC
    · cases hC
    split at hC
    rotate_left
    · cases hC
    rename_i sch hsch
    obtain ⟨x, -, hC⟩ := eok_bind hC
    cases hC
    exact ⟨sch, hsch, fun l hl => Array.mem_append.2 (.inr (Array.mem_append.2 (.inl (Array.mem_append.2
      (.inr (Array.mem_map.2 ⟨l, List.mem_toArray.2 (List.mem_range.2 hl), rfl⟩))))))⟩

/-- **`table_m`**: the cells' `NU + P_LOG + T_LOG`, with `P_LOG = ⌊log₂ (2 ELL + 1)⌋ + 1`, at least `PAD_TO`. -/
theorem tableM_eq (nu ell sLog tLog : ℕ) :
    Flock.RecVStar.tableM { nu := nu, ell := ell, sLog := sLog, tLog := tLog } =
      max Flock.RecVStar.PAD_TO (nu + (Nat.log2 (2 * ell + 1) + 1) + tLog) := by
  have e : 2 * ell + 2 - 1 = 2 * ell + 1 := by omega
  show max Flock.RecVStar.PAD_TO (nu + (Nat.log2 (2 * ell + 2 - 1) + 1) + tLog) = _
  rw [e]

open FlockSoundness.Discharge.Recursion in
/-- **`hp` is the verifier's own**: the inner shape's `sparse` entry (`RecVStar.Inner.sparse`, `[NU, ELL, S_LOG,
T_LOG]`, the verifier's `--sparse`) is the sparse phase's `ν`, `ℓ`, slots' bits and types' bits. -/
def SparseIs (ish : Flock.RecVStar.Inner) (hp : HoloParams) : Prop :=
  ish.sparse = some { nu := hp.nu, ell := hp.ell, sLog := hp.sLog, tLog := hp.tLog }

open FlockSoundness.Discharge.Recursion in
/-- **The model's `mReg`** where `ℓ` is in the class's range: `ν + pLog + tLog` at the template's `ℓ`, padded to
`PAD_TO` (`SparseShape.mReg`). -/
theorem mReg_holoAt {st : Flock.Stmt} {hp : HoloParams} (hell : hp.EllOk (execClassAt st).cls.kLog) :
    (holoAt st hp).cls.shape.mReg =
      max Flock.RecVStar.PAD_TO (hp.nu + (Nat.log2 (2 * hp.ell + 1) + 1) + hp.tLog) := by
  have hs : (holoAt st hp).cls.slotLog = hp.ell := ite_eq_left hell
  rw [show (holoAt st hp).cls.shape.mReg = max Flock.RecVStar.PAD_TO
    (hp.nu + (Nat.log2 (2 * (holoAt st hp).cls.slotLog + 1) + 1) + hp.tLog) from rfl, hs]

open FlockSoundness.Discharge.Recursion in
/-- **The list's openings are at `mReg`** (gap 2's `m`): at `hp` the inner shape's `sparse` entry and `ℓ` in the
class's range, each `rec-opening` statement of the verifier's list is at `m = (holoAt st hp).cls.shape.mReg`, the
model's `mReg` padded to `PAD_TO` as the verifier's `tableM` is, and at `STACKED_CLAIMS` claims. -/
theorem mReg_of_list {st : Flock.Stmt} {hp : HoloParams} {ish : Flock.RecVStar.Inner} {ss : Array VStar.Stmt}
    (hss : Flock.RecVStar.statements ish = .ok ss) (hsp : SparseIs ish hp) (hell : hp.EllOk (execClassAt st).cls.kLog)
    {m c j : ℕ} {S : Flock.RecOpen.GfS} {ports : List (String × ℕ)} {nv : ℕ}
    (hs : VStar.Stmt.opening m c j S ports nv ∈ ss) :
    m = (holoAt st hp).cls.shape.mReg ∧ c = Flock.RecVStar.STACKED_CLAIMS := by
  obtain ⟨sp, hsp', hm, hc⟩ := opening_of_list hss hs
  obtain rfl := Option.some.inj ((show ish.sparse = _ from hsp).symm.trans hsp')
  refine ⟨?_, hc⟩
  rw [hm, tableM_eq, mReg_holoAt hell]

end FlockC3.VStarList
