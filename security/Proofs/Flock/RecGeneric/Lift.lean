module

public import Proofs.Flock.RecGeneric.Coupling

/-!
# A run finder at `P_O`'s aborting runs, read at the composite's reruns

A plan makes the same steps in two models of runs that name runs alike through a map, show the same coins, and accept
alike at every run it asks about (`Plan.exec_map`). So where the map is uniform and a collision among the first
model's runs' evaluations is one among their images', a run finder finds a collision in the first model at most as
often as in the second, plus as often as its plan there asks about a run where the two may accept differently
(`RunFinder.PrAt_le_map`).

`P_O`'s aborting runs (`runModel` at `Comp.innerPO` and `Comp.keepAO`) map so to the composite's reruns
(`Comp.innerModel`) by `extendDraw` (`Proofs.Flock.RecGeneric.Coupling`), the runs where they may accept differently
being reruns that accept at a sample `P_O`'s runs don't show (`Comp.BadO`) or at one where `E_O` gives witnesses outside
`R_O` (`Comp.OutRO`): `Comp.PrAt_le_lift`.

Port of `Claims.Recursion.Bound.PrFC_le_FCI` and `Bound.PrF0_le_F0I` to any run finder, compiled verifier, outer
extraction and server.
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion

section Map

variable {Coin Ω C₁ C₂ : Type} {n₁ n₂ : ℕ}

/-- **A plan makes the same steps in two models** that name runs alike through `f` (`hnamed`), show the same coins
(`hcoins`) and accept alike off the runs where `q` holds (`hacc`), where it asks about no run where `q` holds. -/
theorem Plan.exec_map (m₁ : RunModel Coin Ω n₁ C₁) (m₂ : RunModel Coin Ω n₂ C₂)
    (f : (Fin n₁ → Coin) × C₁ → (Fin n₂ → Coin) × C₂) (q : Ω → (Fin n₂ → Coin) × C₂ → Prop) (ω : Ω)
    (us : ℕ → (Fin n₁ → Coin) × C₁) (hnamed : ∀ e, f (m₁.named us e) = m₂.named (fun i => f (us i)) e)
    (hcoins : ∀ u, m₂.coins (f u) = m₁.coins u) (hacc : ∀ u, ¬ q ω (f u) → (m₁.acc ω u ↔ m₂.acc ω (f u))) :
    ∀ p : Plan Coin, (∀ e ∈ (p.exec m₂ ω fun i => f (us i)).1, ¬ q ω (m₂.named (fun i => f (us i)) e)) →
      p.exec m₁ ω us = p.exec m₂ ω fun i => f (us i) := by
  intro p
  induction p with
  | done l => exact fun _ => rfl
  | ask u k ih =>
    intro hq
    simp only [Plan.exec, List.mem_cons, forall_eq_or_imp] at hq ⊢
    have hu : ¬ q ω (f (m₁.named us u)) := by
      rw [hnamed]
      exact hq.1
    rw [propext (hacc _ hu), hnamed, ih _ hq.2]
  | look u k ih =>
    intro hq
    simp only [Plan.exec] at hq ⊢
    rw [← hcoins, hnamed, ih _ hq]

variable [Fintype Coin] [Nonempty Coin] [Fintype Ω] [Nonempty Ω] [Fintype C₁] [Nonempty C₁] [Fintype C₂]
  [Nonempty C₂]

/-- **A run finder lifts along a uniform map of runs**: where `f` names runs alike, shows the same coins, makes the two
models accept alike off the runs where `q` holds, and a collision among runs' evaluations in `m₁` is one among their
images' in `m₂` (`hlog`), `F` finds a collision in `m₁` at most as often as in `m₂`, plus as often as its plan in `m₂`
asks about a run where `q` holds: off those, its plan makes the same steps in both (`Plan.exec_map`), and its coins
map uniformly. -/
theorem RunFinder.PrAt_le_map (H : Msg → Msg) (F : RunFinder Coin) (m₁ : RunModel Coin Ω n₁ C₁)
    (m₂ : RunModel Coin Ω n₂ C₂) (f : (Fin n₁ → Coin) × C₁ → (Fin n₂ → Coin) × C₂) (hf : UnifMap f)
    (q : Ω → (Fin n₂ → Coin) × C₂ → Prop) (x : Msg)
    (hnamed : ∀ us e, f (m₁.named us e) = m₂.named (fun i => f (us i)) e)
    (hcoins : ∀ u, m₂.coins (f u) = m₁.coins u) (hacc : ∀ ω u, ¬ q ω (f u) → (m₁.acc ω u ↔ m₂.acc ω (f u)))
    (hlog : ∀ (ω : Ω) (l : List RunExp) (g : RunExp → (Fin n₁ → Coin) × C₁),
      (findColl (l.flatMap fun e => m₁.log ω (g e))).isSome →
        ∃ a b, findColl (l.flatMap fun e => m₂.log ω (f (g e))) = some (a, b) ∧ Collision H a b) :
    F.PrAt H m₁ x ≤ F.PrAt H m₂ x + F.PrAsk m₂ x q := by
  classical
  unfold RunFinder.PrAt RunFinder.PrAsk
  by_cases hs : 0 < F.sel x
  swap
  · simp [hs]
  simp only [hs, ↓reduceDIte, ↓reduceIte]
  unfold Finder.Pr
  have key : ∀ s : Fin (F.sel x) × Ω × (Fin (F.slots x + 1) → (Fin n₁ → Coin) × C₁),
      (∃ a b, (F.finder m₁ x hs).out s = some (a, b) ∧ Collision H a b) →
      (∃ a b, (F.finder m₂ x hs).out (s.1, s.2.1, fun i => f (s.2.2 i)) = some (a, b) ∧ Collision H a b) ∨
        ∃ e ∈ ((F.plan x s.1).exec m₂ s.2.1 (slotRuns fun i => f (s.2.2 i))).1,
          q s.2.1 (m₂.named (slotRuns fun i => f (s.2.2 i)) e) := by
    intro s hs'
    obtain ⟨a, b, hout, -⟩ := hs'
    show (∃ a b, findColl ((((F.plan x s.1).exec m₂ s.2.1 fun i => f (slotRuns s.2.2 i)).2.2.flatMap fun e =>
        m₂.log s.2.1 (m₂.named (fun i => f (slotRuns s.2.2 i)) e))) = some (a, b) ∧ Collision H a b) ∨
      ∃ e ∈ ((F.plan x s.1).exec m₂ s.2.1 fun i => f (slotRuns s.2.2 i)).1,
        q s.2.1 (m₂.named (fun i => f (slotRuns s.2.2 i)) e)
    by_cases hB : ∃ e ∈ ((F.plan x s.1).exec m₂ s.2.1 fun i => f (slotRuns s.2.2 i)).1,
        q s.2.1 (m₂.named (fun i => f (slotRuns s.2.2 i)) e)
    · exact Or.inr hB
    refine Or.inl ?_
    have hE := Plan.exec_map m₁ m₂ f q s.2.1 (slotRuns s.2.2) (hnamed _) hcoins (hacc s.2.1) (F.plan x s.1)
      fun e he hqe => hB ⟨e, he, hqe⟩
    have hout' : findColl ((((F.plan x s.1).exec m₁ s.2.1 (slotRuns s.2.2)).2.2.flatMap fun e =>
        m₁.log s.2.1 (m₁.named (slotRuns s.2.2) e))) = some (a, b) := hout
    rw [hE] at hout'
    have hfn : (fun e => m₂.log s.2.1 (m₂.named (fun i => f (slotRuns s.2.2 i)) e)) =
        fun e => m₂.log s.2.1 (f (m₁.named (slotRuns s.2.2) e)) :=
      funext fun e => by rw [hnamed]
    rw [hfn]
    exact hlog s.2.1 _ (m₁.named (slotRuns s.2.2)) (Option.isSome_iff_exists.2 ⟨_, hout'⟩)
  have hΦ : UnifMap (fun s : Fin (F.sel x) × Ω × (Fin (F.slots x + 1) → (Fin n₁ → Coin) × C₁) =>
      ((s.1, s.2.1, fun i => f (s.2.2 i)) : Fin (F.sel x) × Ω × (Fin (F.slots x + 1) → (Fin n₂ → Coin) × C₂))) :=
    UnifMap.self.prod (UnifMap.self.prod (UnifMap.pi hf _))
  exact (Pr_mono key).trans ((Pr_or_le _ _).trans (add_le_add
    (le_of_eq (hΦ.Pr fun s' => ∃ a b, (F.finder m₂ x hs).out s' = some (a, b) ∧ Collision H a b))
    (le_of_eq (hΦ.Pr fun s' => ∃ e ∈ ((F.plan x s'.1).exec m₂ s'.2.1 (slotRuns s'.2.2)).1,
      q s'.2.1 (m₂.named (slotRuns s'.2.2) e)))))

end Map

namespace Comp

variable {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {K : Comp Coin} {I : Proto Coin W}
  {S : Server} {H : Msg → Msg} {Ω : Type} {P : Prover Ω} {x : Msg}

/-- **A run finder at `P_O`'s aborting runs finds a collision at most as often as at the composite's reruns, plus as
often as its plan there asks about a rerun that accepts at a bad sample or at one where `E_O` gives witnesses outside
`R_O`**: runs map to reruns uniformly (`unifMap_extendDraw`), named and with coins alike (`named_extend`,
`coins_extend`), accepting alike off those reruns (`accAt_iff`), and a collision among runs' calls is one among their
reruns' searched evaluations (`findColl_cover`), which are true (`rerunCalls_true`). -/
theorem PrAt_le_lift (F : RunFinder Coin) (hCF : I.LastCoinFree H) (decC : Msg → Coin)
    (hdec : ∀ c, decC (I.ans c) = c) (hn : I.V.k (K.xI x) ≤ K.V.k x) (hRp : K.Replays H x) (hT : K.CallsTrue H x)
    (hL : K.CallsLogColl H (S.IVStar I.V I.ans) x) [Fintype Ω] [Nonempty Ω] :
    F.PrAt H (runModel H I.V (K.innerPO I S H P x decC) (K.xI x) (K.keepAO I S H P x)) (K.xI x) ≤
      F.PrAt H (K.innerModel (S.IVStar I.V I.ans) H P x) (K.xI x) +
        F.PrAsk (K.innerModel (S.IVStar I.V I.ans) H P x) (K.xI x) fun ω u =>
          (K.BadO I S H P x ω u ∨ K.OutRO H P x ω u) ∧ (K.innerModel (S.IVStar I.V I.ans) H P x).acc ω u := by
  refine F.PrAt_le_map H _ _ extendDraw (unifMap_extendDraw hn) _ (K.xI x) (named_extend decC hn)
    (coins_extend decC hn) (fun ω u hu => accAt_iff hCF decC hdec hn ω u hu) fun ω l g h => ?_
  refine findColl_cover (P := P) hCF decC hdec hRp hL ω (fun p hp => ?_) (fun p hp => ?_) h
  · obtain ⟨e, -, hpe⟩ := List.mem_flatMap.1 hp
    exact rerunCalls_true (P := P) hRp hT ω (extendDraw (g e)) p hpe
  · obtain ⟨e, he, hpe⟩ := List.mem_flatMap.1 hp
    refine ⟨extendDraw (g e), ?_, fun p' hp' => List.mem_flatMap.2 ⟨e, he, hp'⟩⟩
    rw [← abortRun_eqO decC hn]
    exact hpe

end Comp

end Claims.RecGeneric
