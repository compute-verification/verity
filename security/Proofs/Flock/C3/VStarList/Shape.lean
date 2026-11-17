module

public import Proofs.Flock.Recursive.HoloExec
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
import Proofs.Flock.VBridge.Family.Structure

/-! The list's R at `mReg`, and `hp` as the inner shape's `sparse` entry. -/

@[expose] public section

namespace FlockC3.VStarList

theorem eok_bind {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : (x >>= f) = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error e => cases h
  | ok a => exact ⟨a, rfl, h⟩

theorem capStatement_ok {vsh : VStar.Algebra.Shape} {K : VStar.Stmt} (h : VStar.capStatement vsh = .ok K) :
    ∃ n, K = .cap 0 n := by
  unfold VStar.capStatement at h
  obtain ⟨s, -, h⟩ := eok_bind h
  split at h <;> cases h
  exact ⟨_, rfl⟩

theorem levelCaps_cap {vsh : VStar.Algebra.Shape} {K : Array VStar.Stmt} (h : VStar.levelCapStatements vsh = .ok K) :
    ∀ s ∈ K, ∃ l N, 1 ≤ l ∧ s = .cap l N := by
  unfold VStar.levelCapStatements at h
  obtain ⟨sc, -, h⟩ := eok_bind h
  cases h
  intro s hs
  obtain ⟨j, -, rfl⟩ := Array.mem_map.1 hs
  exact ⟨j + 1, _, by omega, rfl⟩

theorem caps_cap {vsh : VStar.Algebra.Shape} {K : Array VStar.Stmt} (h : VStar.levelCapStatements vsh = .ok K)
    (n : ℕ) : ∀ s ∈ K ++ #[.cap 0 n], ∃ l N, s = .cap l N := by
  intro s hs
  rcases Array.mem_append.1 hs with hs | hs
  · obtain ⟨l, N, -, rfl⟩ := levelCaps_cap h s hs
    exact ⟨l, N, rfl⟩
  · simp only [List.mem_toArray, List.mem_singleton] at hs
    exact ⟨0, n, hs⟩

theorem linkStatements_ok {i : Flock.RecVStar.Inner} {L : Array VStar.Stmt}
    (h : Flock.RecVStar.linkStatements i = .ok L) :
    (i.links = none ∧ L = #[]) ∨ ∃ l, i.links = some l ∧ L = #[.link l.UB l.PB l.rows] := by
  unfold Flock.RecVStar.linkStatements at h
  cases hl : i.links with
  | none =>
    rw [hl] at h
    cases h
    exact .inl ⟨rfl, rfl⟩
  | some l =>
    rw [hl] at h
    dsimp only at h
    obtain ⟨⟨⟩, -, h⟩ := eok_bind h
    refine .inr ⟨l, rfl, ?_⟩
    try dsimp only at h
    split at h
    all_goals first | (cases h; done) | (cases h; rfl)

theorem tail_cases {vsh : VStar.Algebra.Shape} {K : Array VStar.Stmt} (hK : VStar.levelCapStatements vsh = .ok K)
    {i : Flock.RecVStar.Inner} {L : Array VStar.Stmt} (hL : Flock.RecVStar.linkStatements i = .ok L) (n : ℕ) :
    ∀ s ∈ K ++ L ++ #[.cap 0 n], (∃ l N, s = .cap l N) ∨ ∃ UB PB N, s = .link UB PB N := by
  intro s hs
  rcases Array.mem_append.1 hs with hs | hs
  · rcases Array.mem_append.1 hs with hs | hs
    · obtain ⟨l, N, -, rfl⟩ := levelCaps_cap hK s hs
      exact .inl ⟨l, N, rfl⟩
    · rcases linkStatements_ok hL with ⟨-, rfl⟩ | ⟨l, -, rfl⟩
      · simp at hs
      · simp only [List.mem_toArray, List.mem_singleton] at hs
        exact .inr ⟨_, _, _, hs⟩
  · simp only [List.mem_toArray, List.mem_singleton] at hs
    exact .inl ⟨0, n, hs⟩

theorem statements_ok {i : Flock.RecVStar.Inner} {ss : Array VStar.Stmt} (h : Flock.RecVStar.statements i = .ok ss) :
    ∃ vsh A C K L n, VStar.statementsOf vsh i.holo = .ok A ∧ Flock.RecVStar.candidate3Statements i = .ok C ∧
      VStar.levelCapStatements vsh = .ok K ∧ Flock.RecVStar.checkTopsListed i K = .ok () ∧
      Flock.RecVStar.linkStatements i = .ok L ∧
      VStar.capStatement vsh = .ok (.cap 0 n) ∧ ss = A ++ C ++ (K ++ L ++ #[.cap 0 n]) ∧
      Flock.RecVStar.checkCandidate3 i (A ++ C ++ (K ++ L ++ #[.cap 0 n])) = .ok () := by
  unfold Flock.RecVStar.statements at h
  dsimp only at h
  split at h
  rotate_left
  · cases h
  split at h
  · cases h
  obtain ⟨ps, hps, h⟩ := eok_bind h
  obtain ⟨gs, hgs, h⟩ := eok_bind h
  split at h
  · cases h
  split at h
  · cases h
  obtain ⟨K, hK, h⟩ := eok_bind h
  obtain ⟨t, ht, h⟩ := eok_bind h
  obtain ⟨A, hA, h⟩ := eok_bind h
  obtain ⟨C, hC, h⟩ := eok_bind h
  obtain ⟨L, hL, h⟩ := eok_bind h
  obtain ⟨K0, hK0, h⟩ := eok_bind h
  obtain ⟨n, rfl⟩ := capStatement_ok hK0
  obtain ⟨u, hu, h⟩ := eok_bind h
  obtain ⟨u', -, h⟩ := eok_bind h
  obtain ⟨u'', -, h⟩ := eok_bind h
  cases h
  have e : A ++ C ++ K ++ L ++ #[VStar.Stmt.cap 0 n] = A ++ C ++ (K ++ L ++ #[.cap 0 n]) := by
    simp only [Array.append_assoc]
  rw [e] at hu
  exact ⟨_, A, C, K, L, n, hA, hC, hK, ht, hL, hK0, e, hu⟩

theorem statements_holo {i : Flock.RecVStar.Inner} {ss : Array VStar.Stmt} (h : Flock.RecVStar.statements i = .ok ss)
    (hh : i.holo = true) :
    ∃ vsh A C K L n, VStar.statementsHolo vsh = .ok A ∧ Flock.RecVStar.candidate3Statements i = .ok C ∧
      VStar.levelCapStatements vsh = .ok K ∧ Flock.RecVStar.checkTopsListed i K = .ok () ∧
      Flock.RecVStar.linkStatements i = .ok L ∧
      VStar.capStatement vsh = .ok (.cap 0 n) ∧ ss = A ++ C ++ (K ++ L ++ #[.cap 0 n]) ∧
      Flock.RecVStar.checkCandidate3 i (A ++ C ++ (K ++ L ++ #[.cap 0 n])) = .ok () := by
  obtain ⟨vsh, A, C, K, L, n, hA, hC, hK, ht, hL, hK0, e, hu⟩ := statements_ok h
  rw [hh] at hA
  exact ⟨vsh, A, C, K, L, n, hA, hC, hK, ht, hL, hK0, e, hu⟩

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

theorem check_openings {i : Flock.RecVStar.Inner} {ss : Array VStar.Stmt} {r : Flock.RecVStar.StackedShape}
    (h : Flock.RecVStar.checkCandidate3 i ss = .ok ()) (hr : i.stacked = some r) {m c j : ℕ}
    {S : Flock.RecOpen.GfS} {ports : List (String × ℕ)} {nv : ℕ}
    (hs : VStar.Stmt.opening m c j S ports nv ∈ ss) : m = r.m ∧ c = r.claims := by
  unfold Flock.RecVStar.checkCandidate3 at h
  obtain ⟨⟨⟩, -, h⟩ := eok_bind h
  have h : Flock.RecVStar.checkCandidate3Lists i ss = .ok () := h
  unfold Flock.RecVStar.checkCandidate3Lists at h
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

theorem holo_of_list {i : Flock.RecVStar.Inner} {ss : Array VStar.Stmt} (h : Flock.RecVStar.statements i = .ok ss)
    (hs : i.sparse.isSome = true) : i.holo = true := by
  obtain ⟨_, _, _, _, _, _, -, -, -, -, -, -, -, hck⟩ := statements_ok h
  unfold Flock.RecVStar.checkCandidate3 at hck
  obtain ⟨⟨⟩, h1, -⟩ := eok_bind hck
  unfold Flock.RecVStar.checkSparseHolo at h1
  cases hsp : i.sparse with
  | none => rw [hsp] at hs; cases hs
  | some sp =>
    rw [hsp] at h1
    dsimp only at h1
    split at h1
    · assumption
    · cases h1

theorem opening_of_list {i : Flock.RecVStar.Inner} {ss : Array VStar.Stmt} (h : Flock.RecVStar.statements i = .ok ss)
    {m c j : ℕ} {S : Flock.RecOpen.GfS} {ports : List (String × ℕ)} {nv : ℕ}
    (hs : VStar.Stmt.opening m c j S ports nv ∈ ss) :
    ∃ sp, i.sparse = some sp ∧ m = Flock.RecVStar.tableM sp ∧ c = Flock.RecVStar.STACKED_CLAIMS := by
  obtain ⟨vsh, A, C, K, L, n, hA, hC, hK, -, hL, -, rfl, hck⟩ := statements_ok h
  have hAo : VStar.Stmt.opening m c j S ports nv ∉ A := by
    obtain ⟨s, sp, ps, -, -, -, rfl⟩ := FlockVBridge.Family.statementsOf_eq hA
    simp
  have hCo : VStar.Stmt.opening m c j S ports nv ∈ C := by
    rcases Array.mem_append.1 hs with h' | h'
    · rcases Array.mem_append.1 h' with h' | h'
      · exact absurd h' hAo
      · exact h'
    · rcases tail_cases hK hL n _ h' with ⟨_, _, he⟩ | ⟨_, _, _, he⟩ <;> cases he
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

theorem sparse_of_list {i : Flock.RecVStar.Inner} {ss : Array VStar.Stmt} (h : Flock.RecVStar.statements i = .ok ss)
    {sp : Flock.RecSparse.Shape} (hsp : i.sparse = some sp) :
    VStar.Stmt.sparse sp (Flock.RecVStar.hRound i.m i.kLog) ∈ ss := by
  obtain ⟨vsh, A, C, K, L, n, -, hC, -, -, -, -, rfl, -⟩ := statements_ok h
  obtain ⟨rest, hCe, -⟩ := (candidate3_ok hC).2 sp hsp
  rw [hCe]
  simp

theorem stacked_of_list {i : Flock.RecVStar.Inner} {ss : Array VStar.Stmt} (h : Flock.RecVStar.statements i = .ok ss)
    {r : Flock.RecVStar.StackedShape} (hr : i.stacked = some r) :
    ∃ sch, Flock.fast100 r.m = some sch ∧ ∀ l < sch.levels.size,
      VStar.Stmt.recOpenStacked l (2 ^ (sch.levels[l]!).lanes) ((sch.levels[l]!).d - (sch.levels[l]!).cap)
        (sch.levels[l]!).q ∈ ss := by
  obtain ⟨vsh, A, C, K, L, n, -, hC, -, -, -, -, rfl, -⟩ := statements_ok h
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
    exact ⟨sch, hsch, fun l hl => Array.mem_append.2 (.inl (Array.mem_append.2 (.inr (Array.mem_append.2 (.inl
      (Array.mem_append.2 (.inr (Array.mem_map.2 ⟨l, List.mem_toArray.2 (List.mem_range.2 hl), rfl⟩))))))))⟩

theorem tableM_eq (nu ell sLog tLog : ℕ) :
    Flock.RecVStar.tableM { nu := nu, ell := ell, sLog := sLog, tLog := tLog } =
      max Flock.RecVStar.PAD_TO (nu + (Nat.log2 (2 * ell + 1) + 1) + tLog) := by
  have e : 2 * ell + 2 - 1 = 2 * ell + 1 := by omega
  show max Flock.RecVStar.PAD_TO (nu + (Nat.log2 (2 * ell + 2 - 1) + 1) + tLog) = _
  rw [e]

open FlockSoundness.Discharge.Recursion in
def SparseIs (ish : Flock.RecVStar.Inner) (hp : HoloParams) : Prop :=
  ish.sparse = some { nu := hp.nu, ell := hp.ell, sLog := hp.sLog, tLog := hp.tLog }

open FlockSoundness.Discharge.Recursion in
theorem mReg_holoAt {st : Flock.Stmt} {hp : HoloParams} (hell : hp.EllOk (execClassAt st).cls.kLog) :
    (holoAt st hp).cls.shape.mReg =
      max Flock.RecVStar.PAD_TO (hp.nu + (Nat.log2 (2 * hp.ell + 1) + 1) + hp.tLog) := by
  have hs : (holoAt st hp).cls.slotLog = hp.ell := ite_eq_left hell
  rw [show (holoAt st hp).cls.shape.mReg = max Flock.RecVStar.PAD_TO
    (hp.nu + (Nat.log2 (2 * (holoAt st hp).cls.slotLog + 1) + 1) + hp.tLog) from rfl, hs]

open FlockSoundness.Discharge.Recursion in
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
