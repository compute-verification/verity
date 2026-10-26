import Proofs.Flock.Claims.Recursion.VerdictRowEJoint

/-!
# Row E's chain at any statement, messages and coins (`Top.chainHolds_exists`, `Top.rowEJoint_holds`)

`ChainHolds` asks one chain per table whose row past each row D piece's position is `consTerm` of the piece's opened
row, the chain's row at that position and the piece's `coef`, each read off the statement, the messages and the
coins. Within a table the pieces sit at distinct positions (`chainPos_inj`: rep-major, then by level and by query,
each below the rep's `nOpen`), so a chain built row by row from any start (`recChain`) carries every piece's running
sum. So `Top.RowEJoint` holds at every layout, check and witnesses (`rowEJoint_holds`), with no tie to the parts'
`rec-acc` slots.
-/

namespace RecDeploy.Rows

open Flock

theorem levSum_succ (a : ℕ → ℕ) (ℓ : ℕ) :
    ((List.range (ℓ + 1)).map a).sum = ((List.range ℓ).map a).sum + a ℓ := by
  rw [List.range_succ, List.map_append, List.sum_append]
  simp

theorem levSum_mono (a : ℕ → ℕ) {ℓ ℓ' : ℕ} (h : ℓ ≤ ℓ') :
    ((List.range ℓ).map a).sum ≤ ((List.range ℓ').map a).sum := by
  induction h with
  | refl => exact le_rfl
  | step _ ih => rw [levSum_succ]; omega

theorem levSum_inj (a : ℕ → ℕ) {ℓ ℓ' j j' : ℕ} (hj : j < a ℓ) (hj' : j' < a ℓ')
    (h : ((List.range ℓ).map a).sum + j = ((List.range ℓ').map a).sum + j') : ℓ = ℓ' ∧ j = j' := by
  rcases Nat.lt_trichotomy ℓ ℓ' with hl | rfl | hl
  · have h1 := levSum_mono a (show ℓ + 1 ≤ ℓ' by omega)
    rw [levSum_succ] at h1
    omega
  · exact ⟨rfl, by omega⟩
  · have h1 := levSum_mono a (show ℓ' + 1 ≤ ℓ by omega)
    rw [levSum_succ] at h1
    omega

/-- **Distinct openings of one schedule sit at distinct chain positions**: at reps `< 2`, levels of the schedule and
queries of their level. -/
theorem chainPos_inj {sch : Schedule} {r r' ℓ ℓ' j j' : ℕ} (hr : r < 2) (hr' : r' < 2)
    (hℓ : ℓ < sch.levels.size) (hℓ' : ℓ' < sch.levels.size) (hj : j < (sch.levels[ℓ]!).q)
    (hj' : j' < (sch.levels[ℓ']!).q) (h : chainPos sch r ℓ j = chainPos sch r' ℓ' j') :
    r = r' ∧ ℓ = ℓ' ∧ j = j' := by
  have hg : ∀ i (hi : i < sch.levels.size), (sch.levels[i]!).q = (sch.levels.getD i default).q := fun i hi => by
    rw [getElem!_pos sch.levels i hi, Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hi]
    rfl
  rw [hg ℓ hℓ] at hj
  rw [hg ℓ' hℓ'] at hj'
  have hu : ∀ {ℓ j : ℕ}, ℓ < sch.levels.size → j < (sch.levels.getD ℓ default).q →
      ((List.range ℓ).map fun ℓ' => (sch.levels.getD ℓ' default).q).sum + j < nOpen sch := fun {ℓ j} hl hjl => by
    have h1 := levSum_mono (fun ℓ' => (sch.levels.getD ℓ' default).q) (show ℓ + 1 ≤ sch.levels.size by omega)
    rw [levSum_succ] at h1
    try dsimp only at h1
    unfold nOpen
    omega
  have hu1 := hu hℓ hj
  have hu2 := hu hℓ' hj'
  unfold chainPos at h
  have hrr : r = r' := by
    obtain rfl | rfl : r = 0 ∨ r = 1 := by omega
    all_goals obtain rfl | rfl : r' = 0 ∨ r' = 1 := by omega
    all_goals simp only [zero_mul, one_mul, zero_add] at h
    all_goals omega
  subst hrr
  generalize r * nOpen sch = m at h
  have h' : ((List.range ℓ).map fun ℓ' => (sch.levels.getD ℓ' default).q).sum + j =
      ((List.range ℓ').map fun ℓ' => (sch.levels.getD ℓ' default).q).sum + j' := by omega
  exact ⟨rfl, levSum_inj (fun ℓ' => (sch.levels.getD ℓ' default).q) hj hj' h'⟩

open Classical in
/-- **A chain built row by row**: row `p + 1` of table `t` is `step q` of row `p` at the piece `q` of `pcs` at table
`t` and position `p`, where there is one, and row `p` again where there is none. -/
noncomputable def recChain (pcs : List OpenPiece) (pos : OpenPiece → ℕ)
    (step : OpenPiece → Array F128 → Array F128) (t : ℕ) : ℕ → Array F128
  | 0 => #[]
  | p + 1 =>
    if h : ∃ q ∈ pcs, q.table = t ∧ pos q = p then step (Classical.choose h) (recChain pcs pos step t p)
    else recChain pcs pos step t p

/-- **The chain steps at each piece**, where no two pieces of one table share a position. -/
theorem recChain_step {pcs : List OpenPiece} {pos : OpenPiece → ℕ} {step : OpenPiece → Array F128 → Array F128}
    (hinj : ∀ q ∈ pcs, ∀ q' ∈ pcs, q.table = q'.table → pos q = pos q' → q = q') {q : OpenPiece} (hq : q ∈ pcs) :
    recChain pcs pos step q.table (pos q + 1) = step q (recChain pcs pos step q.table (pos q)) := by
  have h : ∃ q' ∈ pcs, q'.table = q.table ∧ pos q' = pos q := ⟨q, hq, rfl, rfl⟩
  obtain ⟨h1, h2, h3⟩ := Classical.choose_spec h
  have he : Classical.choose h = q := hinj _ h1 _ hq h2 h3
  rw [recChain, dif_pos h, he]

end RecDeploy.Rows

namespace RecDeploy.Top

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP)
open Claims.Recursion.VStarB
open RecDeploy.Rows (accHolds sideAt coefRead rowDPieces chainPos recChain)

variable (rd : Msg → Option FinalP) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ)

/-- **The row past a piece's position** that its running sum asks, from the row `a` at its position. -/
noncomputable def accStep (x : Msg) (ms : List Msg) (cs : List Coin) (q : OpenPiece) (a : Array F128) :
    Array F128 :=
  #[FlockVBridge.consTerm (2 ^ ((sideAt rd x ms q).sch.levels[q.level]!).lanes)
      (FlockSoundness.Refine.openAt (sideAt rd x ms q).lig ((sideAt rd x ms q).sch.levels.size - 1) q.level).1[q.query]!
      a (coefRead x cs q) 0,
    FlockVBridge.consTerm (2 ^ ((sideAt rd x ms q).sch.levels[q.level]!).lanes)
      (FlockSoundness.Refine.openAt (sideAt rd x ms q).lig ((sideAt rd x ms q).sch.levels.size - 1) q.level).1[q.query]!
      a (coefRead x cs q) 1]

/-- **Row D's pieces of one table sit at distinct chain positions**: two pieces of `rowDPieces key x` at one table and
one position are one piece (`chainPos_inj` at the table's schedule, and the key is `key`'s). -/
theorem rowDPieces_pos_inj (x : Msg) (ms : List Msg) :
    ∀ q ∈ rowDPieces key x, ∀ q' ∈ rowDPieces key x, q.table = q'.table →
      chainPos (sideAt rd x ms q).sch q.rep q.level q.query =
        chainPos (sideAt rd x ms q').sch q'.rep q'.level q'.query → q = q' := by
  intro q hq q' hq' ht hp
  unfold rowDPieces at hq hq'
  cases hv : FlockDeploy.setupOf false x with
  | none => rw [hv] at hq; simp at hq
  | some v =>
    rw [hv] at hq hq'
    obtain ⟨-, hr, hl, hj, hk⟩ := Rows.mem_rowDPiecesOf.1 hq
    obtain ⟨-, hr', hl', hj', hk'⟩ := Rows.mem_rowDPiecesOf.1 hq'
    have hs : ∀ q'' : OpenPiece, (sideAt rd x ms q'').sch = Rows.schAt v.2.2 q''.table := fun q'' => by
      show Rows.schOf x q''.table = _
      unfold Rows.schOf
      rw [hv]
      all_goals rfl
    rw [hs, hs, ← ht] at hp
    rw [← ht] at hl' hj'
    obtain ⟨h1, h2, h3⟩ := Rows.chainPos_inj hr hr' hl hl' hj hj' hp
    obtain ⟨t, r, ℓ, j, k⟩ := q
    obtain ⟨t', r', ℓ', j', k'⟩ := q'
    simp only at ht h1 h2 h3 hk hk'
    subst ht h1 h2 h3
    rw [hk, hk']

/-- **Some chain carries row D's running sums** at every statement, messages and coins: `recChain` at
`accStep`. -/
theorem chainHolds_exists (x : Msg) (ms : List Msg) (cs : List Coin) : ∃ ch, ChainHolds rd key x ms cs ch := by
  refine ⟨recChain (rowDPieces key x) (fun q => chainPos (sideAt rd x ms q).sch q.rep q.level q.query)
    (accStep rd x ms cs), ?_⟩
  intro q hq k hk
  dsimp only
  rw [Rows.recChain_step (step := accStep rd x ms cs) (rowDPieces_pos_inj rd key x ms) hq]
  obtain rfl | rfl : k = 0 ∨ k = 1 := by omega
  all_goals first | rfl | simp [accStep]

/-- **Row E's leaf holds** at every layout, check and witnesses: `chainHolds_exists` at the messages the last
message opens and the coins. -/
theorem rowEJoint_holds (H : Msg → Msg) (D : Claims.Recursion.Deploy Coin)
    (Bd : Claims.Recursion.Bound Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace) (L : LayoutB)
    (shape : Claims.Recursion.Form → Prop)
    (chk : ℕ → Msg → (ℕ → Coin) → Msg → Claims.Flock.Wit → List Msg → Prop) :
    RowEJoint rd key H D Bd L shape chk :=
  fun x _ _ _ c _ _ _ os last _ _ => chainHolds_exists rd key _ _ _

end RecDeploy.Top
