import Proofs.Flock.SoundAtBridge.Statement

/-!
# `SoundAtBridge` from `EndToEndRegisteredRows`

* `RowsSession.prob_badU_le` (gap 3): at a session with one registrant's opening per unit, the per-unit event has
  probability at most `n` times `EndToEndRegisteredRows`' bound: that guarantee at each unit's opening, and the union
  bound over the units (`game_prob_exists_le`);
* `openedPlur_zero` (gap 2 at `δo = 0`): a layer whose accepted opening is the value layer's reading meets
  `OpenedPlurAt` at `0`;
* `soundAtBridge_of_rows`: `SoundAtBridge` from `EndToEndRegisteredRows`, through `soundAt_of_sessions`.
-/

namespace Flock.SoundAtBridge

open FlockSoundness FlockSoundness.Discharge.ZkReg Accounting.TS
open Pouw.RelativeSecurity (Paths)
open scoped ENNReal

namespace RowsSession

/-- **The per-unit link, by a union bound over the units** (gap 3): `EndToEndRegisteredRows` at each unit's opening
`rdU u` bounds its event by the session's `bound`; a drawn unit wrong or off at its own reads is in its own opening's
event, so the per-unit event is at most `n · bound`. -/
theorem prob_badU_le (hE : Flock.Guarantees.EndToEndRegisteredRows) (s : RowsSession) (hs : s.CR) :
    Game.prob s.BadU s.G s.τ ≤ (s.n : ℝ≥0∞) * s.bound := by
  obtain ⟨hCR, hA2, hRows, hReg⟩ := hs
  have hU : ∀ u : Fin s.n, Game.prob (s.EvAt u) s.G s.τ ≤ s.bound := fun u =>
    hE s.n s.Reg s.own s.I s.ht s.D s.hD s.hU s.pub s.hpub s.hscope s.kd s.hdiv s.dj s.S₀ s.d₀ s.y₀ (s.rdU u) s.wr
      s.dg s.N s.Ω s.Lb s.src s.hTags s.hCust s.k s.Rw s.M s.hk s.hRw s.hM s.ρ s.hρ s.hr s.t' s.ht' s.τ s.qF s.qS
      s.qR hCR hA2 (hRows u) (hReg u)
  have hsub : ∀ o, s.BadU o → ∃ u ∈ (Finset.univ : Finset (Fin s.n)), s.EvAt u o := by
    rintro o ⟨hA, u, hu, h⟩
    exact ⟨u, Finset.mem_univ u, hA, u, hu, h⟩
  calc Game.prob s.BadU s.G s.τ
      ≤ Game.prob (fun o => ∃ u ∈ (Finset.univ : Finset (Fin s.n)), s.EvAt u o) s.G s.τ :=
        Game.prob_mono hsub _ _
    _ ≤ ∑ u : Fin s.n, Game.prob (s.EvAt u) s.G s.τ := game_prob_exists_le _ _ _ _
    _ ≤ ∑ _u : Fin s.n, s.bound := Finset.sum_le_sum fun u _ => hU u
    _ = (s.n : ℝ≥0∞) * s.bound := by rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]

end RowsSession

variable {Ω : Type} [Fintype Ω] {S nπ : ℕ} {V : Type}

/-- **Gap 2 at `δo = 0`**: a layer whose accepted opening is its session's value layer read as a layer value meets
`OpenedPlurAt` at `0`. -/
theorem openedPlur_zero (pf : ProofLayer Ω S nπ V) (sess : Paths Ω S → Fin S → Fin nπ → RowsSession)
    (rows : ∀ s : RowsSession, (Fin s.Circ.N → Bool) → V)
    (h : ∀ x w i, pf.accepted x w i → pf.opened x w i = rows (sess x w i) (sess x w i).Xv) :
    OpenedPlurAt pf sess rows 0 := by
  intro w i
  have h0 : (fun x => pf.accepted x w i ∧ pf.opened x w i ≠ rows (sess x w i) (sess x w i).Xv) = fun _ => False :=
    funext fun x => propext ⟨fun ⟨ha, hne⟩ => hne (h x w i ha), False.elim⟩
  rw [h0]
  unfold Pouw.RelativeSecurity.prob
  simp

/-- **The proof layer a family of sessions gives**: proof `i` of round `w` is accepted exactly where flock-verify
accepts its session's outcome, its opening is the session's value layer read as a layer value, and its relation holds
there when no drawn unit of the session is wrong or off at its own reads. -/
noncomputable def rowsLayer (sess : Paths Ω S → Fin S → Fin nπ → RowsSession) (out : ∀ x w i, (sess x w i).Out)
    (rows : ∀ s : RowsSession, (Fin s.Circ.N → Bool) → V) (right : Paths Ω S → (Fin S → Fin nπ → V) → Prop) :
    ProofLayer Ω S nπ V where
  accepted x w i := LiveAcceptsZKR (sess x w i).I (sess x w i).own (out x w i)
  opened x w i := rows (sess x w i) (sess x w i).Xv
  rel x w i _ := ∀ u ∈ (out x w i).2.1, ¬ (sess x w i).WrongAt u ∧ (sess x w i).RowsAt (out x w i) u
  right := right

/-- **The layer's premises hold together, at flock-verify's acceptance** (non-vacuity of `AcceptedLive`,
`OpenedPlurAt` and `RelOfRows`): at any family of sessions, `rowsLayer` accepts exactly where flock-verify accepts, and
meets `AcceptedLive`, `OpenedPlurAt` at `δo = 0` and `RelOfRows`. `WindowLaw` and `RowsSession.CR` are premises of the
sessions, not of the layer. -/
theorem rowsLayer_premises (sess : Paths Ω S → Fin S → Fin nπ → RowsSession) (out : ∀ x w i, (sess x w i).Out)
    (rows : ∀ s : RowsSession, (Fin s.Circ.N → Bool) → V) (right : Paths Ω S → (Fin S → Fin nπ → V) → Prop) :
    (∀ x w i, (rowsLayer sess out rows right).accepted x w i ↔
      LiveAcceptsZKR (sess x w i).I (sess x w i).own (out x w i)) ∧
    AcceptedLive (rowsLayer sess out rows right) sess out ∧ OpenedPlurAt (rowsLayer sess out rows right) sess rows 0 ∧
    RelOfRows (rowsLayer sess out rows right) sess out rows :=
  ⟨fun _ _ _ => Iff.rfl, fun _ _ _ h => h, openedPlur_zero _ sess rows fun _ _ _ _ => rfl, fun _ _ _ h => h⟩

/-- **`SoundAtBridge` from `EndToEndRegisteredRows`**: an unsound proof whose opening is the value layer's reading
fails its relation there, so by `RelOfRows` some drawn unit of its session is wrong or off at its own reads, and by
`AcceptedLive` flock-verify accepted: the per-unit event, at most `n · bound ≤ δ` in every session
(`RowsSession.prob_badU_le`); one whose opening isn't is in `OpenedPlurAt`'s event, at most `δo`. `WindowLaw` carries
the sessions' bound to the window's paths (`soundAt_of_sessions`). -/
theorem soundAtBridge_of_rows (hE : Flock.Guarantees.EndToEndRegisteredRows) : Flock.Guarantees.SoundAtBridge := by
  intro Ω _ S nπ V pf sess out rows δo δ hCR hLaw hAcc hOpen hRel hδ hδ0
  refine soundAt_of_sessions pf (Out := fun s : RowsSession => s.Out) (fun s => s.G) (fun s => s.τ)
    (fun s => s.BadU) sess out
    (fun x w i => pf.accepted x w i ∧ pf.opened x w i ≠ rows (sess x w i) (sess x w i).Xv) δo δ hLaw hOpen
    (fun x w i => (RowsSession.prob_badU_le hE _ (hCR x w i)).trans (hδ x w i)) hδ0 ?_
  rintro x w i ⟨hA, hR⟩
  by_cases hO : pf.opened x w i = rows (sess x w i) (sess x w i).Xv
  · refine Or.inr ⟨hAcc x w i hA, ?_⟩
    by_contra hno
    have hrel := hRel x w i fun u hu =>
      ⟨fun hw => hno ⟨u, hu, Or.inl hw⟩, Classical.byContradiction fun hr => hno ⟨u, hu, Or.inr hr⟩⟩
    exact hR (by rw [hO]; exact hrel)
  · exact Or.inl ⟨hA, hO⟩

end Flock.SoundAtBridge
