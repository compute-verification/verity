import Proofs.Flock.Claims.Recursion.CapOpens
import Proofs.Flock.Recursive.ValueInner
import Proofs.Flock.Soundness.Discharge.ZkReg.Checked

/-!
# Check 5's site at a staged `rec-open` or `rec-residuals` part

What `ResLower.resTie_copyOf` and `OpenDirs.dirsTie_staged` take of check 5 beyond the staging, from the recursive
setup's own checks (`RecIns.setup`):

* `params_open`, `params_res`: a `rec-open/…` or `rec-residuals/…` name is no other V\* unit's, so check 5 runs
  `checkOpen` or `checkResiduals` at it (`hso`, `hop`, `hsp`, and `ho` at `rec-residuals`).
* `copySite_staged`: check 5 there holds the verifier's registered reads (`hown`) and the inner session's source (`hf`):
  it refuses either missing (`RecCoins.check`).
* `idx_staged`: the table labels its instances `0` to `n − 1` in order (`hidx`), where its labels ascend
  (`LabelsAscend`) and check 5's value rows are no more than its instances: the setup's registered check
  (`checkRegistered`) reads row `idx[i]` of check 5's rows at instance `i`, so every label is a row (`idx_lt_rows`).
-/

namespace RecDeploy.CopySite

open Flock
open Claims (Msg)

/-! ## The names -/

/-- A `rec-open/l<L>-h<H>` unit is no other V\* unit check 5 reads: no R level, R opening, RecSparse or `rec-residuals`
unit. -/
theorem params_open (L H : ℕ) :
    RecCoins.stackedParams (RecOpen.unitName L H) = none ∧ RecOpening.params (RecOpen.unitName L H) = none ∧
      RecSparse.params (RecOpen.unitName L H) = none ∧ RecResiduals.params (RecOpen.unitName L H) = none := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · show RecOpen.stackedParams _ = none
    cases hq : RecOpen.stackedParams (RecOpen.unitName L H) with
    | none => rfl
    | some p =>
      obtain ⟨L', H'⟩ := p
      exact absurd (RecOpen.stackedName_of_params hq).symm (RecOpen.unitName_ne_stackedName L H L' H')
  · cases hq : RecOpening.params (RecOpen.unitName L H) with
    | none => rfl
    | some p =>
      obtain ⟨m, c, j⟩ := p
      exact absurd (FlockC3.Compose.openingName_params hq) ((FlockC3.Compose.openingName_ne m c j).1 L H)
  · cases hq : RecSparse.params (RecOpen.unitName L H) with
    | none => rfl
    | some sh => exact absurd (FlockVBridge.Sparse.unitName_params hq) ((FlockC3.PubRows.sparse_ne_names sh).1 L H)
  · cases hq : RecResiduals.params (RecOpen.unitName L H) with
    | none => rfl
    | some p =>
      obtain ⟨m, k, c, j⟩ := p
      exact absurd (residualsName_of_params hq) ((FlockC3.Reps.resName_ne m k c j).1 L H)

/-- A `rec-residuals/m<m>-k<k>-c<c>-p<j>` unit is no other V\* unit check 5 reads: no R level, R opening, RecSparse or
`rec-open` unit. -/
theorem params_res (m k c j : ℕ) :
    RecCoins.stackedParams (RecResiduals.unitName m k c j) = none ∧
      RecOpening.params (RecResiduals.unitName m k c j) = none ∧
      RecSparse.params (RecResiduals.unitName m k c j) = none ∧
      RecOpen.params (RecResiduals.unitName m k c j) = none := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · show RecOpen.stackedParams _ = none
    cases hq : RecOpen.stackedParams (RecResiduals.unitName m k c j) with
    | none => rfl
    | some p =>
      obtain ⟨L, H⟩ := p
      exact absurd (RecOpen.stackedName_of_params hq).symm ((FlockC3.Reps.resName_ne m k c j).2.1 L H)
  · cases hq : RecOpening.params (RecResiduals.unitName m k c j) with
    | none => rfl
    | some p =>
      obtain ⟨m', c', j'⟩ := p
      exact absurd (FlockC3.Compose.openingName_params hq) ((FlockC3.Compose.openingName_ne m' c' j').2.1 m k c j)
  · cases hq : RecSparse.params (RecResiduals.unitName m k c j) with
    | none => rfl
    | some sh =>
      exact absurd (FlockVBridge.Sparse.unitName_params hq) ((FlockC3.PubRows.sparse_ne_names sh).2.1 m k c j)
  · cases hq : RecOpen.params (RecResiduals.unitName m k c j) with
    | none => rfl
    | some p =>
      obtain ⟨L, H⟩ := p
      exact absurd (FlockVBridge.unitName_params hq).symm ((FlockC3.Reps.resName_ne m k c j).1 L H)

/-! ## Check 5's inputs -/

/-- **Check 5 refuses a `rec-open` or `rec-residuals` unit without the verifier's registered reads or the inner
session.** -/
theorem site_of_check {st : Stmt} {own : Option (Array Registered.Port)} {f : Option RecCoins.Source}
    {c3 : Option RecCoins.C3} (h : RecCoins.check st own f c3 = .ok ())
    (hso : RecCoins.stackedParams st.c.unit.name = none) (hop : RecOpening.params st.c.unit.name = none)
    (hn : (RecOpen.params st.c.unit.name).isSome ∨ (RecResiduals.params st.c.unit.name).isSome) :
    ∃ own' f', own = some own' ∧ f = some f' := by
  unfold RecCoins.check at h
  rcases hn with hn | hn
  all_goals
    obtain ⟨v, hv⟩ := Option.isSome_iff_exists.1 hn
    simp only [hv, hso, hop, Option.isNone_some, Bool.false_and, Bool.and_false, Bool.false_eq_true,
      ↓reduceIte] at h
    cases own with
    | none => cases h
    | some own =>
      cases f with
      | none => cases h
      | some f => exact ⟨own, f, rfl, rfl⟩

/-- **Check 5's site at a staged `rec-open` or `rec-residuals` part** (`resTie_copyOf`'s and `dirsTie_staged`'s
`hown`, `hf`, `hso`, `hop`, `hsp`, and `ho` at `rec-residuals`): the setup ran check 5 at the table (`setup_coins`),
which refuses either unit without the verifier's registered reads or the inner session (`site_of_check`), and the
unit's name is no other V\* unit's (`params_open`, `params_res`). -/
theorem copySite_staged {i : RecIns} {spec : SessionSpec} {sts : Array Stmt} {xO : Msg}
    (hsu : setupO xO = some (i, spec, sts)) (h0 : 0 < sts.size)
    (hnm : (RecOpen.params sts[0].c.unit.name).isSome ∨ (RecResiduals.params sts[0].c.unit.name).isSome) :
    ∃ (own : Array Registered.Port) (f : RecCoins.Source), i.base.registered = some own ∧
      RecCoins.Source.atC3 i.src i.inner i.commits = some f ∧
      RecCoins.stackedParams sts[0].c.unit.name = none ∧ RecOpening.params sts[0].c.unit.name = none ∧
      RecSparse.params sts[0].c.unit.name = none ∧
      ((RecResiduals.params sts[0].c.unit.name).isSome → RecOpen.params sts[0].c.unit.name = none) := by
  have hnames : RecCoins.stackedParams sts[0].c.unit.name = none ∧ RecOpening.params sts[0].c.unit.name = none ∧
      RecSparse.params sts[0].c.unit.name = none ∧
      ((RecResiduals.params sts[0].c.unit.name).isSome → RecOpen.params sts[0].c.unit.name = none) := by
    rcases hnm with hn | hn
    · obtain ⟨⟨L, H⟩, hv⟩ := Option.isSome_iff_exists.1 hn
      rw [← FlockVBridge.unitName_params hv]
      obtain ⟨h1, h2, h3, h4⟩ := params_open L H
      exact ⟨h1, h2, h3, fun h => by rw [h4] at h; cases h⟩
    · obtain ⟨⟨m, k, c, j⟩, hv⟩ := Option.isSome_iff_exists.1 hn
      rw [← residualsName_of_params hv]
      obtain ⟨h1, h2, h3, h4⟩ := params_res m k c j
      exact ⟨h1, h2, h3, fun _ => h4⟩
  obtain ⟨hso, hop, hsp, ho⟩ := hnames
  obtain ⟨-, hset⟩ := setupO_some hsu
  have hcoin := Flock.Readable.array_forM_ok (RecIns.setup_coins hset) 0 h0
  obtain ⟨own, f, hown, hf⟩ := site_of_check hcoin hso hop hnm
  exact ⟨own, f, hown, hf, hso, hop, hsp, ho⟩

/-! ## The table's labels -/

/-- What `checkRows` holds of the reads file: its port `port` reads row `i` of the verifier's rows in unit `i`. -/
theorem checkRows_reads {salt : String → ℕ → ByteArray} {c : Circuit} {pub : Public}
    {own : Array Registered.Port} {port value : String} {rows : Array (Array ℕ)} {what : ℕ → String}
    (h : RecCoins.checkRows salt c pub own port value rows what = .ok ()) :
    ∃ reg, own.find? (·.name == port) = some reg ∧ reg.reads = (Array.range rows.size).map some := by
  unfold RecCoins.checkRows at h
  dsimp only at h
  split at h
  · rename_i reg hf
    split at h
    · cases h
    split at h
    · cases h
    split at h
    · cases h
    rename_i hr
    simp only [bne_iff_ne, ne_eq, not_not] at hr
    exact ⟨reg, hf, hr⟩
  · cases h

/-- **Every label is one of check 5's rows**: the registered check reads, at instance `i`, the reads file's entry at its
label `idx[i]` (`checkPort`), and the entries `checkRows` holds are the rows' (`checkRows_reads`). -/
theorem idx_lt_rows {st : Stmt} {own : Array Registered.Port} (hreg : checkRegistered st (some own) = .ok ())
    {salt : String → ℕ → ByteArray} {port value : String} {rows : Array (Array ℕ)} {what : ℕ → String}
    (hrows : RecCoins.checkRows salt st.c st.pub own port value rows what = .ok ()) :
    ∃ idx, Registered.unitIndices st.pub = .ok idx ∧ ∀ j < st.pub.n, idx.getD j 0 < rows.size := by
  obtain ⟨reg, hf, hr⟩ := checkRows_reads hrows
  obtain ⟨t, idx, -, hidx, hports⟩ :=
    FlockSoundness.Discharge.ZkReg.check_tables (FlockSoundness.Discharge.ZkReg.checkRegistered_ok hreg)
  refine ⟨idx, hidx, fun j hj => ?_⟩
  obtain ⟨p, positions, paths, -, -, hall⟩ :=
    FlockSoundness.Discharge.ZkReg.checkPort_at (hports reg (Array.mem_of_find?_eq_some hf))
  obtain ⟨-, hrd, -⟩ := hall j hj
  rw [hr] at hrd
  by_contra hge
  rw [Array.getElem?_eq_none (by simp only [Array.size_map, Array.size_range]; omega)] at hrd
  cases hrd

/-- An ascending array of `N` naturals below `N` is `0, …, N − 1`. -/
theorem eq_range_of_lt {a : Array ℕ} {N : ℕ} (hs : a.size = N)
    (hasc : ∀ j, 0 < j → j < N → a[j - 1]! < a[j]!) (hlt : ∀ j < N, a[j]! < N) : a = Array.range N := by
  have hup : ∀ j, j < N → j ≤ a[j]! := by
    intro j
    induction j with
    | zero => intro _; exact Nat.zero_le _
    | succ k ih =>
      intro hk
      have h1 := hasc (k + 1) (Nat.succ_pos k) hk
      simp only [Nat.add_sub_cancel] at h1
      have h2 := ih (by omega)
      omega
  have hdown : ∀ k, k < N → a[N - 1 - k]! ≤ N - 1 - k := by
    intro k
    induction k with
    | zero =>
      intro h
      have := hlt (N - 1) (by omega)
      simp only [Nat.sub_zero]
      omega
    | succ k ih =>
      intro hk
      have h1 := ih (by omega)
      have h2 := hasc (N - 1 - k) (by omega) (by omega)
      rw [show N - 1 - k - 1 = N - 1 - (k + 1) by omega] at h2
      omega
  refine Array.ext (by simp [hs]) fun j h1 h2 => ?_
  have hj : j < N := by omega
  have e1 := hup j hj
  have e2 := hdown (N - 1 - j) (by omega)
  rw [show N - 1 - (N - 1 - j) = j by omega] at e2
  rw [Array.getElem_range, ← getElem!_pos a j h1]
  omega

theorem getD_of_lt {a : Array ℕ} {j : ℕ} (hj : j < a.size) : a.getD j 0 = a[j]! := by
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hj, Option.getD_some, getElem!_pos a j hj]

/-- **The table's labels ascend, one per instance** (`loadPublic`'s order, kept by a full draw). -/
def LabelsAscend (pub : Public) : Prop :=
  ∃ idx, Registered.unitIndices pub = .ok idx ∧ idx.size = pub.n ∧ ∀ j, 0 < j → j < pub.n → idx[j - 1]! < idx[j]!

/-- **`hidx` at a staged table** (`resTie_copyOf`'s and `dirsTie_staged`'s): where its labels ascend and check 5's value
rows at some port are no more than its instances, every label is a row (`idx_lt_rows`), so the labels are
`0, …, n − 1` (`eq_range_of_lt`). -/
theorem idx_staged {i : RecIns} {spec : SessionSpec} {sts : Array Stmt} {xO : Msg}
    (hsu : setupO xO = some (i, spec, sts)) (h0 : 0 < sts.size) {own : Array Registered.Port}
    (hown : i.base.registered = some own) {salt : String → ℕ → ByteArray} {port value : String}
    {rows : Array (Array ℕ)} {what : ℕ → String}
    (hrows : RecCoins.checkRows salt sts[0].c sts[0].pub own port value rows what = .ok ())
    (hle : rows.size ≤ sts[0].pub.n) (hasc : LabelsAscend sts[0].pub) :
    Registered.unitIndices sts[0].pub = .ok (Array.range sts[0].pub.n) := by
  obtain ⟨-, hset⟩ := setupO_some hsu
  have hreg := RecIns.setup_registered hset sts[0] (Array.getElem_mem h0)
  rw [hown] at hreg
  obtain ⟨idx, hidx, hlt⟩ := idx_lt_rows hreg hrows
  obtain ⟨idx', hidx', hsz, hup⟩ := hasc
  rw [hidx] at hidx'
  cases hidx'
  rw [hidx, eq_range_of_lt hsz hup fun j hj => by
    have := hlt j hj
    rw [getD_of_lt (by omega)] at this
    omega]

end RecDeploy.CopySite
