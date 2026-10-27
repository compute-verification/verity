import Definitions.Flock.ZK.Cut
import Proofs.Flock.Soundness.ZK.Blocks
import Proofs.Flock.Soundness.ZK.Dist
import Proofs.Flock.Soundness.ZK.Session

/-!
# Zero knowledge: Lemma A at every coin vector, up to and including a refusal

`table_prefinal_indep` is Lemma A (`docs/zk-proof-public.md` §4.2) at coins the prover refuses nothing at. Here it holds at
every coin vector, for what the prover sends before it stops (`Cut`):

* **The translation is the same.** Each check guards exactly the messages whose matching needs it: the rank check the
  sent `ŝ` (the mask slot's shift `Δ` solves `Φ_k(Δ) = ν_k(w) − ν_k(w')` for the sent claims only), `W^r` invertible the
  restarted table `y₁'^r` and levels ≥ 1, `β^r ≠ 0` the inner proof's `Y₁`. Everything else matches whatever the coins:
  the masked values and `ρ_in`, `σ_in` through the pads, `ȳ^r`, `Y₂` and every ideal leaf because their randomness isn't
  moved, the region `ŝ` by H_reg. The paper's translation for a refused coin keeps the moved block: `R' := R` and
  `μ' := μ`. Here the shift of a block whose matching isn't needed is whatever the general formula gives (in Lean
  `x / 0 = 0`, so `μ' = μ` at `β = 0`), since nothing sent reads it.
* `table_prefinal_translate_cut`, `table_prefinal_indep_cut`: for one table, at coins that admit what it sent (`Admits`).
* `Session.sent_admits`: **the schedule.** At every coin vector, what each table sent before the session's first failing
  check is admitted. The opening's check is cumulative over a table's reps, so the claims whose `ŝ` were sent are exactly
  the claims of the last opening that passed, which its check covers.
* `Session.session_prefinal_indep_cut`: **Lemma A at every coin vector, for a session of `J` tables**: the pre-final
  views on two tuples of witnesses carrying the public region data, cut where the session stops, are equally distributed.
-/

namespace FlockSoundness.ZK

open Finset

-- As in `SHVZK`: the randomness and the view are products of up to fifteen factors.
set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Table

variable {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] (T : Table F K W Salt Seed D Hi VL) [CharP F 2]

/-- The rank check over fewer claims passes. -/
theorem RankOn.mono {P Q : Fin T.nC → Prop} (hQP : ∀ k, Q k → P k) (h : T.RankOn P) : T.RankOn Q := by
  classical
  intro t
  obtain ⟨c, hc⟩ := h fun k => if hk : Q k.1 then t ⟨k.1, hk⟩ else 0
  refine ⟨c, funext fun k => ?_⟩
  have e := congrFun hc ⟨k.1, hQP k.1 k.2⟩
  simp only [k.2, ↓reduceDIte] at e
  exact e

/-- The rank check over no claim passes. -/
theorem rankOn_of_forall_not {Q : Fin T.nC → Prop} (hQ : ∀ k, ¬ Q k) : T.RankOn Q :=
  fun _ => ⟨0, funext fun k => (hQ k.1 k.2).elim⟩

/-- `Rank` is the rank check over every claim. -/
theorem Rank.rankOn (h : T.Rank) (P : Fin T.nC → Prop) : T.RankOn P := by
  classical
  intro t
  obtain ⟨c, hc⟩ := h fun k => if hk : P k then t ⟨k, hk⟩ else 0
  refine ⟨c, funext fun k => ?_⟩
  have e := congrFun hc k.1
  simp only [k.2, ↓reduceDIte] at e
  exact e

/-- The mask slot reaches every vector of values on the claims the rank check covers. -/
theorem maskHom_surjOn {P : Fin T.nC → Prop} (h : T.RankOn P) (s : Fin T.nC → Fin 128 → F) :
    ∃ Δ : T.Mask, ∀ k, P k → T.maskHom Δ k = s k := by
  choose c hc using fun v : Fin 128 => h fun k => s k.1 v
  exact ⟨fun x v => c v x, fun k hk => funext fun v => congrFun (hc v) ⟨k, hk⟩⟩

variable [AddCommGroup Salt] [AddCommGroup Seed] [AddCommGroup Hi]

/-- **Lemma A's translation at every coin vector** (§4.2), for one table: for witnesses `w`, `w'` that carry the public
region data, and messages `S` the coins admit, `table_prefinal_translate`'s translation of the prover's randomness maps
`w`'s pre-final view to `w'`'s, cut to `S`. The mask slot's shift solves the claims of `S` only. -/
theorem table_prefinal_translate_cut (S : T.Sent) (hS : T.Admits S) (w w' : W) (hreg : T.sReg w = T.pubReg)
    (hreg' : T.sReg w' = T.pubReg) :
    ∃ (d₀ : T.Mask) (d₁ : T.Mask → (Fin T.nL → Fin T.tpad → F) × T.XLanes × T.XPad)
      (d₂ : T.Mask → (Fin T.nL → Fin T.tpad → F) × T.XLanes × T.XPad → Fin 2 → (Fin T.nE → F) × F × F)
      (d₃ : T.Mask → (Fin T.nL → Fin T.tpad → F) × T.XLanes × T.XPad → (Fin 2 → (Fin T.nE → F) × F × F) →
        (Fin 2 → Fin T.nPad → F) × (Fin 2 → Fin T.nPad → F) × (Fin 2 → PadIdx T.nE → F) × T.Seeds),
      ∀ ω, T.cut S (T.preView w ω) = T.cut S (T.preView w' (ω.1 + d₀, ω.2.1 + d₁ ω.1, ω.2.2.1 + d₂ ω.1 ω.2.1,
        ω.2.2.2 + d₃ ω.1 ω.2.1 ω.2.2.1)) := by
  obtain ⟨hrank, hW, hβ⟩ := hS
  obtain ⟨Δ, hΔ⟩ := T.maskHom_surjOn hrank (T.nu w - T.nu w')
  let dR : T.Mask → T.XLanes := fun u r i c =>
    ![(T.Winv r (T.y1 w u r c - T.y1 w' (u + Δ) r c)).1, (T.Winv r (T.y1 w u r c - T.y1 w' (u + Δ) r c)).2] i
  let d₁ : T.Mask → (Fin T.nL → Fin T.tpad → F) × T.XLanes × T.XPad := fun u => (0, dR u, 0)
  let dv : T.Mask → (Fin T.nL → Fin T.tpad → F) × T.XLanes × T.XPad → Fin 2 → Fin T.nE → F := fun u b r =>
    T.vals r w u b.2.1 - T.vals r w' (u + Δ) (b.2.1 + dR u)
  let d₂ : T.Mask → (Fin T.nL → Fin T.tpad → F) × T.XLanes × T.XPad → Fin 2 → (Fin T.nE → F) × F × F :=
    fun u b r => (dv u b r, -(T.ζ r * dv u b r T.ia), -(dv u b r T.ib))
  let d₃ : T.Mask → (Fin T.nL → Fin T.tpad → F) × T.XLanes × T.XPad → (Fin 2 → (Fin T.nE → F) × F × F) →
      (Fin 2 → Fin T.nPad → F) × (Fin 2 → Fin T.nPad → F) × (Fin 2 → PadIdx T.nE → F) × T.Seeds :=
    fun u b h => (0, 0, fun r i => (T.hFull (h r) i - T.hFull ((h + d₂ u b) r) i) / T.β r, 0)
  refine ⟨Δ, d₁, d₂, d₃, fun ω => ?_⟩
  obtain ⟨u, ⟨μ0, R, μR⟩, h, ⟨padh, padμ, μ, s⟩⟩ := ω
  have hy : ∀ r, S.y1 r → T.y1' w u R r = T.y1' w' (u + Δ) (R + dR u) r := by
    intro r hr
    funext c
    have e : T.Wr r (dR u r 0 c) (dR u r 1 c) = T.y1 w u r c - T.y1 w' (u + Δ) r c := T.Wr_Winv (hW r hr) _
    simp only [y1', Pi.add_apply, Wr_add, e]
    ring
  apply Cut.ext
  all_goals simp only [cut, preView, view, d₁, d₂, d₃, dv, Prod.mk_add_mk, Pi.add_apply, add_zero]
  case sHat =>
    funext k
    by_cases hk : S.cl k
    · simp only [hk, ↓reduceIte, map_add, Pi.add_apply, hΔ k hk, Pi.sub_apply]
      congr 1
      abel
    · simp only [hk, ↓reduceIte]
  case sReg => rw [hreg, hreg']
  case masked =>
    funext r i
    simp only [Prod.fst_add, Pi.add_apply, Pi.sub_apply]
    ring
  case y1 =>
    funext r
    by_cases hr : S.y1 r
    · simp only [hr, ↓reduceIte, hy r hr]
    · simp only [hr, ↓reduceIte]
  case lig =>
    funext r
    by_cases hr : S.y1 r
    · simp only [hr, ↓reduceIte, hy r hr]
    · simp only [hr, ↓reduceIte]
  case inner =>
    funext r
    simp only [innerMsg, Prod.fst_add, Prod.snd_add, Pi.add_apply, Pi.sub_apply]
    refine Prod.ext ?_ ?_ <;> simp only [] <;> ring
  case Y1 =>
    funext r
    by_cases hr : S.fin r
    · simp only [hr, ↓reduceIte]
      congr 1
      funext i
      field_simp [hβ r hr]
      ring
    · simp only [hr, ↓reduceIte]

variable [Fintype F] [DecidableEq F] [Fintype Salt] [DecidableEq Salt] [Fintype Seed] [DecidableEq Seed] [Fintype Hi]
  [DecidableEq Hi]

/-- **Lemma A at every coin vector** (§4.2, `card_fiber_eq_of_triShift` applied to its translation), for one table: in
world `W₀`, the pre-final views on any two witnesses carrying the public region data, cut to messages the coins admit,
take every value equally often. -/
theorem table_prefinal_indep_cut (S : T.Sent) (hS : T.Admits S) (w w' : W) (hreg : T.sReg w = T.pubReg)
    (hreg' : T.sReg w' = T.pubReg) :
    SameDist (fun ω => T.cut S (T.preView w ω)) (fun ω => T.cut S (T.preView w' ω)) := by
  classical
  obtain ⟨d₀, d₁, d₂, d₃, h⟩ := T.table_prefinal_translate_cut S hS w w' hreg hreg'
  exact SameDist.of_card_fiber_eq (card_fiber_eq_of_triShift4 _ _ d₀ d₁ d₂ d₃ h)

end Table

namespace Session

variable {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] {J : ℕ} (T : Fin J → Table F K W Salt Seed D Hi VL)
  [CharP F 2]

theorem passes_guard {rep : ∀ j, Fin (T j).nC → Fin 2} {j : Fin J} {r : Fin 2} {p : Phase}
    (h : passes T rep j r p) : guard T rep j r p :=
  h j r p le_rfl

theorem passes_of_le {rep : ∀ j, Fin (T j).nC → Fin 2} {j j' : Fin J} {r r' : Fin 2} {p p' : Phase}
    (hle : pos j' r' p' ≤ pos j r p) (h : passes T rep j r p) : passes T rep j' r' p' :=
  fun a b c hc => h a b c (hc.trans hle)

/-- **The schedule** (Lemma C's input): at every coin vector, what each table sent before the session's first failing
check is admitted. The claims whose `ŝ` were sent are those of the table's reps whose opening passed: all of them if rep
1's did, which its cumulative check covers; rep 0's if only rep 0's did, which its check covers; none otherwise. -/
theorem sent_admits (rep : ∀ j, Fin (T j).nC → Fin 2) (j : Fin J) : (T j).Admits (sent T rep j) := by
  refine ⟨?_, fun r hr => passes_guard T hr, fun r hr => passes_guard T hr⟩
  by_cases h1 : passes T rep j 1 .opening
  · refine Table.RankOn.mono (T j) (fun k _ => ?_) (passes_guard T h1)
    have := (rep j k).isLt
    rw [Fin.le_iff_val_le_val]
    simp only [Fin.isValue, Fin.val_one]
    omega
  · by_cases h0 : passes T rep j 0 .opening
    · refine Table.RankOn.mono (T j) (fun k hk => ?_) (passes_guard T h0)
      have := (rep j k).isLt
      rw [Fin.le_iff_val_le_val]
      by_contra hne
      have e : rep j k = 1 := Fin.ext (by simp only [Fin.isValue, Fin.val_zero, Fin.val_one] at hne ⊢; omega)
      exact h1 (e ▸ hk)
    · refine Table.rankOn_of_forall_not (T j) fun k hk => h0 (passes_of_le T ?_ hk)
      simp only [pos, Phase.idx, Fin.isValue, Fin.val_zero]
      omega

variable [AddCommGroup Salt] [AddCommGroup Seed] [AddCommGroup Hi] [Fintype F] [DecidableEq F] [Fintype Salt]
  [DecidableEq Salt] [Fintype Seed] [DecidableEq Seed] [Fintype Hi] [DecidableEq Hi]

/-- **Lemma A at every coin vector, for a session of `J` tables** (§4.2): the session's pre-final views in world `W₀` on
two tuples of witnesses carrying the public region data, each table's cut to what it sent before the session's first
failing check, are equally distributed. -/
theorem session_prefinal_indep_cut (rep : ∀ j, Fin (T j).nC → Fin 2) (w w' : Fin J → W)
    (hreg : ∀ j, (T j).sReg (w j) = (T j).pubReg) (hreg' : ∀ j, (T j).sReg (w' j) = (T j).pubReg) :
    SameDist (fun ω : ∀ j, (T j).Rand => fun j => (T j).cut (sent T rep j) ((T j).preView (w j) (ω j)))
      (fun ω : ∀ j, (T j).Rand => fun j => (T j).cut (sent T rep j) ((T j).preView (w' j) (ω j))) :=
  SameDist.pi (fun j => fun ω => (T j).cut (sent T rep j) ((T j).preView (w j) ω))
    (fun j => fun ω => (T j).cut (sent T rep j) ((T j).preView (w' j) ω))
    fun j => (T j).table_prefinal_indep_cut (sent T rep j) (sent_admits T rep j) (w j) (w' j) (hreg j) (hreg' j)

end Session

end FlockSoundness.ZK
