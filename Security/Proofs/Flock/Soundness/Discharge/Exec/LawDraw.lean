import Proofs.Flock.Soundness.ExecDrawOS
import Proofs.Flock.Soundness.Discharge.Exec.Pts

/-!
# How many units the live law draws

`Law.execOS`'s draw at `ω` is `univ.filter (· ∈ d.units)` when `runOS` returns `d`, and every unit when the run would
read past the source's bytes. This file counts the first case: under `ExecStrata σ k S`, a stratified draw that succeeds
holds exactly `∑ s, k s` units (`execOS_draw_card`), so every draw of the live stratified law has `∑ s, k s` units or
is every unit (`execOS_draw_card_or`).

* `uniform_lt`, `loopP_lt`, `subset_spec`: a `subset k n` draw that succeeds is `k` distinct indices below `n`, the
  values below `k` of the loop's permutation, which keeps `[0, n)`.
* `stratified_sub`: every unit of a successful stratified draw comes from some stratum's local draw (the converse of
  `Law.stratified_units`).
* `stratified_card`: stratum `s`'s units in the draw are its local draw under `(S s).unit`, which is injective on the
  stratum by `ExecStrata`, so the draw has `k s` units in stratum `s`.
* `setupH_mPts_execOS`, `setupH_mPts_execOS_or`: with `Pts.lean`, when the draw file written for a draw proves exactly
  its units (no closure), `setupH` at the law's draw sets up at `mPtsOf c.g (∑ s, k s)` on every run that succeeds, and
  at `mPtsOf c.g n` on the others. So one `mPts` serves every draw of the law when those two agree; otherwise the runs
  that read past the source's bytes set up at another.
-/

namespace FlockSoundness.Discharge.Exec

open FlockSoundness.Audit FlockSoundness.Audit.Law Finset

/-- **`uniform m` returns a value below `m`.** -/
theorem uniform_lt {m : ℕ} {src : ByteArray} : ∀ {fuel pos r p : ℕ},
    Flock.Draw.uniform m src fuel pos = some (r, p) → r < m := by
  intro fuel
  induction fuel with
  | zero => intro pos r p h; simp [Flock.Draw.uniform] at h
  | succ fuel ih =>
    intro pos r p h
    unfold Flock.Draw.uniform at h
    dsimp only at h
    split at h
    · cases h
    · split at h
      · rename_i hx
        obtain ⟨rfl, -⟩ := Prod.mk.inj (Option.some.inj h)
        rcases Nat.eq_zero_or_pos m with rfl | hm
        · simp at hx
        · exact Nat.mod_lt _ hm
      · exact ih h

/-- **The loop's permutation keeps `[0, n)`**, and a loop of `c > 0` steps from `i` that finishes has `i + c ≤ n`. -/
theorem loopP_lt (n : ℕ) (src : ByteArray) : ∀ (c i : ℕ) (π π' : Equiv.Perm ℕ) (pos : ℕ),
    DrawExec.loopP n src (List.range' i c) π pos = some π' → (∀ t < n, π t < n) →
      (c = 0 ∨ i + c ≤ n) ∧ ∀ t < n, π' t < n := by
  intro c
  induction c with
  | zero =>
    intro i π π' pos h hπ
    simp only [List.range'_zero, DrawExec.loopP, Option.some.injEq] at h
    subst h
    exact ⟨Or.inl rfl, hπ⟩
  | succ c ih =>
    intro i π π' pos h hπ
    rw [List.range'_succ] at h
    simp only [DrawExec.loopP] at h
    split at h
    · cases h
    · rename_i r p hu
      have hr := uniform_lt hu
      obtain ⟨hc, hπ'⟩ := ih (i + 1) _ π' p h (DrawExec.swap_trans_lt (by omega) π hπ)
      exact ⟨Or.inr (by omega), hπ'⟩

/-- **A `subset` draw that succeeds is `k ≤ n` distinct indices below `n`**: the values on `[0, k)` of a permutation
that keeps `[0, n)`. -/
theorem subset_spec {k n : ℕ} {src : ByteArray} {D : Array ℕ} (h : Flock.Draw.subset k n src = some D) :
    ∃ π : Equiv.Perm ℕ, k ≤ n ∧ (∀ t < n, π t < n) ∧ ∀ x, x ∈ D ↔ ∃ i < k, π i = x := by
  rw [DrawExec.subset_eq] at h
  cases hl : DrawExec.loopSw n src (List.range' 0 k) ∅ 0 with
  | none => rw [hl] at h; exact absurd h (by simp)
  | some sw =>
    rw [hl, Option.map_some, Option.some.injEq] at h
    obtain ⟨π, hπ, hsw⟩ := DrawExec.loopSw_loopP n src _ ∅ (Equiv.refl ℕ) 0 (fun t => by simp) sw hl
    obtain ⟨hk, hlt⟩ := loopP_lt n src k 0 _ π 0 hπ (fun t ht => ht)
    refine ⟨π, by omega, hlt, fun x => ?_⟩
    rw [← h, DrawExec.mem_qsort, Array.mem_map]
    constructor
    · rintro ⟨i, hi, rfl⟩; exact ⟨i, Array.mem_range.1 hi, (hsw i).symm⟩
    · rintro ⟨i, hi, rfl⟩; exact ⟨i, Array.mem_range.2 hi, hsw i⟩

/-- The converse of `Law.forIn_append`: every value the loop returns was in `out` or appended by some step. -/
theorem forIn_append_sub (g : ℕ → Option (Array ℕ)) (u : ℕ → ℕ → ℕ)
    (F : ℕ → Array ℕ → Option (ForInStep (Array ℕ)))
    (hF : ∀ i out, F i out = (g i).bind fun l => some (ForInStep.yield (out ++ l.map (u i)))) :
    ∀ (l : List ℕ) (out res : Array ℕ), forIn l out F = some res →
      ∀ x ∈ res, x ∈ out ∨ ∃ i ∈ l, ∃ li, g i = some li ∧ ∃ j ∈ li, u i j = x := by
  intro l
  induction l with
  | nil =>
    intro out res h x hx
    obtain rfl : out = res := by simpa using h
    exact Or.inl hx
  | cons i is ih =>
    intro out res h x hx
    rw [List.forIn_cons, hF] at h
    cases hg : g i with
    | none => rw [hg] at h; exact absurd h (by simp)
    | some li =>
      rw [hg] at h
      simp only [Option.bind_some] at h
      rcases ih _ res h x hx with hx' | ⟨i', hi', li', hg', j, hj, hx'⟩
      · rcases Array.mem_append.1 hx' with hx'' | hx''
        · exact Or.inl hx''
        · obtain ⟨j, hj, rfl⟩ := Array.mem_map.1 hx''
          exact Or.inr ⟨i, List.mem_cons.2 (Or.inl rfl), li, hg, j, hj, rfl⟩
      · exact Or.inr ⟨i', List.mem_cons.2 (Or.inr hi'), li', hg', j, hj, hx'⟩

/-- **Every unit of a successful stratified draw is a stratum's**: the unit of an index of that stratum's local draw
(the converse of `Law.stratified_units`). -/
theorem stratified_sub (strata : Array Flock.Draw.Stratum) (streams : Array ByteArray) (D : Array ℕ)
    (h : Flock.Draw.stratified strata streams = some D) :
    ∀ x ∈ D, ∃ i < strata.size, ∃ li, Flock.Draw.subset strata[i]!.k strata[i]!.size
      (streams.getD i ByteArray.empty) = some li ∧ ∃ j ∈ li, strata[i]!.unit j = x := by
  unfold Flock.Draw.stratified at h
  dsimp only at h
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at h
  have hsize : ([:strata.size] : Std.Legacy.Range).size = strata.size := by
    show (strata.size - 0 + 1 - 1) / 1 = _
    omega
  rw [hsize] at h
  obtain ⟨res, hres, hD⟩ := Option.bind_eq_some_iff.1 h
  have key := forIn_append_sub (fun i => Flock.Draw.subset strata[i]!.k strata[i]!.size
    (streams.getD i ByteArray.empty)) (fun i => strata[i]!.unit) _ (fun i out => rfl) _ _ _ hres
  intro x hx
  have hD' : res.qsort (fun x1 x2 => decide (x1 < x2)) = D := Option.some.inj hD
  rw [← hD', DrawExec.mem_qsort] at hx
  rcases key x hx with h0 | ⟨i, hi, li, hli, j, hj, hx'⟩
  · simp at h0
  · have hi' : i ∈ List.range' 0 strata.size 1 := hi
    rw [List.mem_range'_1] at hi'
    exact ⟨i, by omega, li, hli, j, hj, hx'⟩

/-- **A successful stratified draw has `∑ s, k s` units**, `k s` of them in stratum `s`, when the executable strata are
`σ`'s (`ExecStrata`). -/
theorem stratified_card {n m : ℕ} (σ : Fin n → Fin m) (k : Fin m → ℕ) (S : Fin m → Flock.Draw.Stratum)
    (hS : ExecStrata σ k S) {streams : Array ByteArray} {D : Array ℕ}
    (h : Flock.Draw.stratified (Array.ofFn S) streams = some D) :
    (univ.filter fun u : Fin n => u.val ∈ D).card = ∑ s, k s := by
  classical
  have hSs : ∀ s : Fin m, (Array.ofFn S)[s.val]! = S s := fun s => by
    rw [getElem!_pos (Array.ofFn S) s.val (by simp)]; simp
  have hloc : ∀ s : Fin m, ∃ li, Flock.Draw.subset (S s).k (S s).size (streams.getD s.val ByteArray.empty) = some li :=
    fun s => by
      obtain ⟨li, hli, -⟩ := stratified_units _ _ D h s.val (by simp)
      rw [hSs] at hli
      exact ⟨li, hli⟩
  choose li hli using hloc
  -- a local index of stratum `s`'s draw is below the stratum's size, and its unit is a unit of `σ`'s stratum `s`
  have hlt : ∀ s, ∀ j ∈ li s, j < (S s).size := fun s j hj => by
    obtain ⟨π, hk, hπ, hmem⟩ := subset_spec (hli s)
    obtain ⟨i, hi, rfl⟩ := (hmem j).1 hj
    exact hπ i (by omega)
  have hunit : ∀ s, ∀ j < (S s).size, ∃ v : Fin n, σ v = s ∧ v.val = (S s).unit j := fun s j hj => by
    have : (S s).unit j ∈ (stratum σ s).map Fin.valEmbedding := by
      rw [← (hS s).2.2]; exact mem_image_of_mem _ (mem_range.2 hj)
    obtain ⟨v, hv, hve⟩ := mem_map.1 this
    exact ⟨v, mem_stratum.1 hv, hve⟩
  -- a unit is in the draw exactly when it is the unit of an index of its own stratum's local draw
  have hiff : ∀ u : Fin n, u.val ∈ D ↔ ∃ j ∈ li (σ u), (S (σ u)).unit j = u.val := by
    intro u
    constructor
    · intro hu
      obtain ⟨i, hi, li', hli', j, hj, hx⟩ := stratified_sub _ _ D h u.val hu
      have hi' : i < m := by simpa using hi
      set s : Fin m := ⟨i, hi'⟩ with hs
      have hSi : (Array.ofFn S)[i]! = S s := hSs s
      rw [hSi] at hli' hx
      have : li' = li s := Option.some.inj (hli'.symm.trans (hli s))
      subst this
      obtain ⟨v, hv, hve⟩ := hunit s j (hlt s j hj)
      obtain rfl : v = u := Fin.ext (hve.trans hx)
      rw [hv]
      exact ⟨j, hj, hx⟩
    · rintro ⟨j, hj, hx⟩
      obtain ⟨li', hli', hmem⟩ := stratified_units _ _ D h (σ u).val (by simp)
      rw [hSs] at hli' hmem
      have : li' = li (σ u) := Option.some.inj (hli'.symm.trans (hli (σ u)))
      subst this
      exact hx ▸ hmem j hj
  rw [card_eq_sum_card_fiberwise (f := σ) (t := univ) (fun _ _ => mem_univ _)]
  refine Finset.sum_congr rfl fun s _ => ?_
  -- stratum `s`'s units in the draw, as numbers, are its local draw's image under `(S s).unit`
  set Ls : Finset ℕ := (li s).toList.toFinset with hLs
  have hLs_card : Ls.card = k s := by
    obtain ⟨π, hk, hπ, hmem⟩ := subset_spec (hli s)
    have : Ls = (range (S s).k).image π := by
      ext x
      simp only [hLs, List.mem_toFinset, Array.mem_toList_iff, mem_image, mem_range]
      exact hmem x
    rw [this, card_image_of_injective _ π.injective, card_range, (hS s).1]
  have hinj : Set.InjOn (S s).unit (range (S s).size) := by
    refine card_image_iff.1 ?_
    rw [(hS s).2.2, card_map, card_range, (hS s).2.1]
  have hsub : Ls ⊆ range (S s).size := fun j hj => by
    simp only [hLs, List.mem_toFinset, Array.mem_toList_iff] at hj
    exact mem_range.2 (hlt s j hj)
  have hmap : ((univ.filter fun u : Fin n => u.val ∈ D).filter (σ · = s)).map Fin.valEmbedding =
      Ls.image (S s).unit := by
    ext x
    simp only [mem_map, mem_filter, mem_univ, true_and, Fin.valEmbedding_apply, mem_image, hLs,
      List.mem_toFinset, Array.mem_toList_iff]
    constructor
    · rintro ⟨u, ⟨hu, rfl⟩, rfl⟩
      obtain ⟨j, hj, hx⟩ := (hiff u).1 hu
      exact ⟨j, hj, hx⟩
    · rintro ⟨j, hj, rfl⟩
      obtain ⟨v, hv, hve⟩ := hunit s j (hlt s j hj)
      refine ⟨v, ⟨(hiff v).2 ⟨j, hv ▸ hj, by rw [hv]; exact hve.symm⟩, hv⟩, hve⟩
  rw [← card_map Fin.valEmbedding, hmap, card_image_of_injOn (hinj.mono (by exact_mod_cast hsub)), hLs_card]

variable {n m : ℕ}

/-- **A draw of the live stratified law from a run that succeeds has `∑ s, k s` units.** -/
theorem execOS_draw_card (σ : Fin n → Fin m) (k : Fin m → ℕ) (S : Fin m → Flock.Draw.Stratum) (hS : ExecStrata σ k S)
    (K pop : ℕ) {Ω : Type} [Fintype Ω] [Nonempty Ω] {L : Fin m → ℕ} (t : Ω → (i : Fin m) → Fin (L i) → Fin 256)
    (ω : Ω) {d : Flock.Draw.UnitDraw} (h : runOS (t ω) (.stratified K (Array.ofFn S)) pop = some d) :
    ((execOS (n := n) t (.stratified K (Array.ofFn S)) pop).draw ω).card = ∑ s, k s := by
  have hd := runOS_strata (t ω) _ _ pop d (fun get s => ⟨_, fun _ => rfl, drawWith_stratified_run get K pop _ s⟩) h
  show (match runOS (t ω) (.stratified K (Array.ofFn S)) pop with
    | some d => univ.filter fun u : Fin n => u.val ∈ d.units
    | none => univ).card = _
  rw [h]
  exact stratified_card σ k S hS hd

/-- **Every draw of the live stratified law has `∑ s, k s` units, or is every unit** (a run that would read past the
source's bytes). -/
theorem execOS_draw_card_or (σ : Fin n → Fin m) (k : Fin m → ℕ) (S : Fin m → Flock.Draw.Stratum)
    (hS : ExecStrata σ k S) (K pop : ℕ) {Ω : Type} [Fintype Ω] [Nonempty Ω] {L : Fin m → ℕ}
    (t : Ω → (i : Fin m) → Fin (L i) → Fin 256) (ω : Ω) :
    ((execOS (n := n) t (.stratified K (Array.ofFn S)) pop).draw ω).card = ∑ s, k s ∨
      (execOS (n := n) t (.stratified K (Array.ofFn S)) pop).draw ω = univ := by
  cases h : runOS (t ω) (.stratified K (Array.ofFn S)) pop with
  | some d => exact Or.inl (execOS_draw_card σ k S hS K pop t ω h)
  | none =>
    refine Or.inr ?_
    show (match runOS (t ω) (.stratified K (Array.ofFn S)) pop with
      | some d => univ.filter fun u : Fin n => u.val ∈ d.units
      | none => univ) = _
    rw [h]

open Flock in
/-- **`m_pts` at the live law's draw, on a run that succeeds**: when the draw file `dj B` written for a draw `B` proves
exactly `B`'s units (`hdj`: no closure beyond them), `setupH` at the law's draw sets up at `mPtsOf c.g (∑ s, k s)`, the
same at every such run (`execOS_draw_card`, `setupH_mPts_of_proved`). -/
theorem setupH_mPts_execOS {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt} {c : Flock.Circuit}
    (hc : (if tags.typed then HmRow.parseTyped tags cf tables else HmRow.parse tags cf tables) = .ok c)
    (σ : Fin n → Fin m) (k : Fin m → ℕ) (S : Fin m → Flock.Draw.Stratum) (hS : ExecStrata σ k S)
    (K pop : ℕ) {Ω : Type} [Fintype Ω] [Nonempty Ω] {L : Fin m → ℕ} (t : Ω → (i : Fin m) → Fin (L i) → Fin 256)
    (ω : Ω) {d : Flock.Draw.UnitDraw} (hrun : runOS (t ω) (.stratified K (Array.ofFn S)) pop = some d)
    (dj : Finset (Fin n) → Lean.Json)
    (hdj : ∀ B D, Flock.Draw.UnitDraw.ofJson (dj B) = .ok D → D.proved.size = B.card)
    (h : Stmt.setupH tags cf pf coins tables
      (some (dj ((execOS (n := n) t (.stratified K (Array.ofFn S)) pop).draw ω))) partition program = .ok st) :
    st.mPts = mPtsOf c.g (∑ s, k s) := by
  rw [← execOS_draw_card σ k S hS K pop t ω hrun]
  exact setupH_mPts_of_proved hc (fun D hD => hdj _ D hD) h

open Flock in
/-- **`m_pts` at every draw of the live law**, under the same `hdj`: `mPtsOf c.g (∑ s, k s)`, or `mPtsOf c.g n` at a
run that would read past the source's bytes, whose draw is every unit (`execOS_draw_card_or`). -/
theorem setupH_mPts_execOS_or {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt} {c : Flock.Circuit}
    (hc : (if tags.typed then HmRow.parseTyped tags cf tables else HmRow.parse tags cf tables) = .ok c)
    (σ : Fin n → Fin m) (k : Fin m → ℕ) (S : Fin m → Flock.Draw.Stratum) (hS : ExecStrata σ k S)
    (K pop : ℕ) {Ω : Type} [Fintype Ω] [Nonempty Ω] {L : Fin m → ℕ} (t : Ω → (i : Fin m) → Fin (L i) → Fin 256)
    (ω : Ω) (dj : Finset (Fin n) → Lean.Json)
    (hdj : ∀ B D, Flock.Draw.UnitDraw.ofJson (dj B) = .ok D → D.proved.size = B.card)
    (h : Stmt.setupH tags cf pf coins tables
      (some (dj ((execOS (n := n) t (.stratified K (Array.ofFn S)) pop).draw ω))) partition program = .ok st) :
    st.mPts = mPtsOf c.g (∑ s, k s) ∨ st.mPts = mPtsOf c.g n := by
  have e := setupH_mPts_of_proved hc (fun D hD => hdj _ D hD) h
  rcases execOS_draw_card_or σ k S hS K pop t ω with hd | hd
  · exact Or.inl (hd ▸ e)
  · rw [hd, card_univ, Fintype.card_fin] at e
    exact Or.inr e

end FlockSoundness.Discharge.Exec

