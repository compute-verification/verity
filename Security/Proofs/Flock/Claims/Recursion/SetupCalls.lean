import Proofs.Flock.Claims.Recursion.BoundRouteCols
import Proofs.Flock.Claims.Recursion.KSLin2

/-!
# The inner verifier's calls at the draw's coins

The deployed record's `innerCalls` are the inner verifier's own calls on the inner messages the outer witnesses give, at
the coins it is handed (`innerCallsOf`), so at a draw's coins they are its inner run's calls, each with `H`'s answer:
every call of an accepted inner run is among those `Deploy.callsAt` lists, by construction, at any steps `nextE`
(`callsLogCollAt_deployRec`).

* `callsLogCollAt_deployRec`: `CallsLogCollAt` at `deployRec`, from nothing.
* `callsLogCollAt_C`, `forksBindAt_C_pick_setup`, `forksBindAt_C_setup`: `callsLogColl_C` and `forksBindAt_C_pick`
  with the inner verifier's calls searched at each draw's coins, the forks binding over the draws' evaluations
  (`ForksBindAtV`), where an inner view's rejection escapes to a collision with the compiled verifier's own.
* `innerAcceptsCollAtV_C`: `innerAcceptsCollAt_C` with that escape.
-/

namespace RecDeploy

open Claims (Msg Out Prover run coinSeq replay findColl Collision)
open FlockDeploy (Coin)
open Claims.Recursion (Bound Inner)

variable {W Col T : Type}

section Readers

variable (H : Msg → Msg) (I : Claims.Recursion.Inner Coin W) (ivScope : Msg → Bool) (b : Brings)
  (DO : Claims.Flock.Deploy Coin Col T)

/-- The inner view's run at `deployRec` is the inner verifier's run of the readers' messages at the draw's coins. -/
theorem innerRun_deployRec (x : Msg) (r : Fin ((deployRec I.V ivScope b).V.k x) → Coin) (ws : List Claims.Flock.Wit) :
    Claims.Recursion.innerRun H (deployRec I.V ivScope b) I x r ws =
      run H (replay (b.rd.msgs x ws)) I.V (xIOf x) () fun i => coinSeq r i.val := by
  show run H (replay ((innerView H I.V b.rd x r ws).msgs.map Prod.fst)) I.V (xIOf x) ()
    (fun i => coinSeq r i.val) = _
  unfold innerView
  rw [run_replay_seen]

/-- **`CallsLogCollAt` at `deployRec`, by construction**: an inner run's calls are the inner verifier's own at the
draw's coins (`innerCallsOf`), each with `H`'s answer (`Claims.run_logs`). -/
theorem callsLogCollAt_deployRec : Claims.Recursion.CallsLogCollAt H (deployRec I.V ivScope b) I DO := by
  intro decO x Ω P ω r c ws _
  left
  intro e he
  rw [innerRun_deployRec] at he
  refine List.mem_append_right _ (List.mem_map.2 ⟨e.1, List.mem_append_right _ (List.mem_map.2 ⟨e, he, rfl⟩), ?_⟩)
  exact Prod.ext rfl ((Claims.run_logs H _ _ _ _ _).2 e he)

end Readers

section Cols

/-- **`callsLogColl_C` at each draw's coins**: the representative's calls are the deployed record's as a set, and the
calls searched are the same at both records. -/
theorem callsLogCollAt_C {Col' T' : Type} [DecidableEq Col] (H : Msg → Msg) (Bd : Bound Coin Col T)
    (decI : List (ℕ × Col) → List T) (ivScope : Msg → Bool) (b : Brings) (DO : Claims.Flock.Deploy Coin Col' T')
    (hCF : Bd.LastCoinFree H)
    (hL : Claims.Recursion.CallsLogCollAt H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO) :
    Claims.Recursion.CallsLogCollAt H (deployRecC Bd ivScope b) (Inner.ofFlock Bd.deploy decI) DO := by
  intro decO x Ω P ω r c ws hg
  rcases hL decO x Ω P ω r c ws hg with h | h
  · left
    intro e he
    have he' : e ∈ (run H (replay (msgsC Bd b H x r ws)) Bd.IVStar (xIOf x) () fun i => coinSeq r i.val).vLog :=
      (congrArg (fun t => e ∈ t.vLog) (innerRun_C H Bd decI ivScope b x r ws)).mp he
    apply h
    have hv : (run H (replay (msgsC Bd b H x r ws)) Bd.IVStar (xIOf x) () fun i => coinSeq r i.val).vLog.toFinset =
        (Claims.Recursion.innerRun H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) x r
          ws).vLog.toFinset :=
      (Bd.run_nfC H hCF (xIOf x) (b.rd.msgs x ws) fun i => coinSeq r i.val).2.1.trans
        (congrArg (fun t => t.vLog.toFinset) (innerRun_bound H Bd decI ivScope b x r ws)).symm
    exact List.mem_toFinset.1 (hv ▸ List.mem_toFinset.2 he')
  · exact Or.inr h

/-- **`innerAcceptsCollAt_C` over the compiled verifier's evaluations**: the representative's view accepts where the
deployed record's does, and its calls and V\* statements are the deployed record's. -/
theorem innerAcceptsCollAtV_C {Col' T' : Type} [DecidableEq Col] (H : Msg → Msg) (Bd : Bound Coin Col T)
    (decI : List (ℕ × Col) → List T) (ivScope : Msg → Bool) (b : Brings) (DO : Claims.Flock.Deploy Coin Col' T')
    (hCF : Bd.LastCoinFree H) (x : Msg)
    (hA : Claims.Recursion.InnerAcceptsCollAtV H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO
      x) :
    Claims.Recursion.InnerAcceptsCollAtV H (deployRecC Bd ivScope b) (Inner.ofFlock Bd.deploy decI) DO x := by
  intro decO Ω P ω r c ws hg
  rcases hA decO Ω P ω r c ws hg with h | h
  · exact Or.inl ((innerView_C H Bd ivScope b hCF x r ws).1.trans h)
  · exact Or.inr h

/-- **`forksBindAt_C_pick` with the inner verifier's calls searched at each draw's coins**: its argument, with
`IVStar`'s calls among those `callsAt` lists (`hL`), and the collisions it finds among the recorded ones, or among
them and the compiled verifier's own evaluations (`hA`), found among the draws' evaluations (`forkCollV`). -/
theorem forksBindAt_C_pick_setup {Col' T' : Type} [DecidableEq Col] [DecidableEq Col'] (H : Msg → Msg)
    (Bd : Bound Coin Col T)
    (decI : List (ℕ × Col) → List T) (ivScope : Msg → Bool) (b : Brings) (DO : Claims.Flock.Deploy Coin Col' T')
    (pk : Msg → ℕ → Msg → Msg) (hEC : Bd.LastBindsEC H) (hCF : Bd.LastCoinFree H)
    (hT : Claims.Recursion.CallsTrue H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO)
    (hL : Claims.Recursion.CallsLogCollAt H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO)
    (x : Msg) (hP : ∀ (decO : List (ℕ × Col') → List T') (Ω : Type) (P : Prover Ω) (ω : Ω)
      (r : Fin ((deployRec Bd.IVStar ivScope b).V.k x) → Coin)
      (c : (Claims.Recursion.EO (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO).C x)
      (ws : List Claims.Flock.Wit),
      Claims.Recursion.EOGives H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO P x ω r c
        ws →
      (∀ j, j + 1 < Bd.DE.V.k (xIOf x) → (b.rd.msgs x ws).getD j [] =
          pk x j (((run H P (deployRec Bd.IVStar ivScope b).V x ω r).msgs.map Prod.fst).getD j [])) ∨
        ∃ u v, findColl ((deployRec Bd.IVStar ivScope b).calls x ws) = some (u, v) ∧ Collision H u v)
    (hA : Claims.Recursion.InnerAcceptsCollAtV H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO
      x) :
    Claims.Recursion.ForksBindAtV H (deployRecC Bd ivScope b) (Inner.ofFlock Bd.deploy decI) DO x := by
  intro decO Ω P j t hod
  obtain ⟨ws, ws', hg₁, hg₂, hne⟩ := hod
  have hg₁' : Claims.Recursion.EOGives H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO P x
      t.1 t.2.1 t.2.2.2.1 ws := hg₁
  have hg₂' : Claims.Recursion.EOGives H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO P x
      t.1 (Claims.Recursion.fork j t.2.1 t.2.2.1) t.2.2.2.2 ws' := hg₂
  have htrue : ∀ p ∈ (deployRec Bd.IVStar ivScope b).callsAt H x t.2.1 ws ++
      (deployRec Bd.IVStar ivScope b).callsAt H x (Claims.Recursion.fork j t.2.1 t.2.2.1) ws', H p.1 = p.2 := fun p hp =>
    (List.mem_append.1 hp).elim (Claims.Recursion.callsAt_true_of (hT decO x Ω P _ _ _ ws hg₁') p)
      (Claims.Recursion.callsAt_true_of (hT decO x Ω P _ _ _ ws' hg₂') p)
  have hD₁ := Claims.Recursion.drawCalls_eq hg₁
  have hD₂ := Claims.Recursion.drawCalls_eq hg₂
  have htrueV : ∀ p ∈ Claims.Recursion.drawCalls H (deployRecC Bd ivScope b)
      (Inner.ofFlock Bd.deploy decI) DO decO P x t.1 (t.2.1, t.2.2.2.1) ++
      Claims.Recursion.drawCalls H (deployRecC Bd ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO P x
        t.1 (Claims.Recursion.fork j t.2.1 t.2.2.1, t.2.2.2.2), H p.1 = p.2 := by
    rw [hD₁, hD₂]
    intro p hp
    rcases List.mem_append.1 hp with hp | hp <;> rcases List.mem_append.1 hp with hp | hp
    · exact Claims.Recursion.callsAt_true_of (hT decO x Ω P _ _ _ ws hg₁') p hp
    · exact Claims.Recursion.setupCalls_true H _ _ p hp
    · exact Claims.Recursion.callsAt_true_of (hT decO x Ω P _ _ _ ws' hg₂') p hp
    · exact Claims.Recursion.setupCalls_true H _ _ p hp
  have hliftV : ∀ l : List (Msg × Msg),
      (∀ p ∈ l, p ∈ Claims.Recursion.drawCalls H (deployRecC Bd ivScope b)
        (Inner.ofFlock Bd.deploy decI) DO decO P x t.1 (t.2.1, t.2.2.2.1) ++
        Claims.Recursion.drawCalls H (deployRecC Bd ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO
          P x t.1 (Claims.Recursion.fork j t.2.1 t.2.2.1, t.2.2.2.2)) →
      (∃ u v, findColl l = some (u, v) ∧ Collision H u v) →
      Claims.Recursion.forkCollV H (deployRecC Bd ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO P x
        j t := by
    rintro l hsub ⟨u, v, huv, -⟩
    unfold Claims.Recursion.forkCollV
    exact Claims.Recursion.VStar.coll_of_sub htrueV hsub (by simp [huv])
  have hlift : ∀ l : List (Msg × Msg),
      (∀ p ∈ l, p ∈ (deployRec Bd.IVStar ivScope b).callsAt H x t.2.1 ws ++
        (deployRec Bd.IVStar ivScope b).callsAt H x (Claims.Recursion.fork j t.2.1 t.2.2.1) ws') →
      (∃ u v, findColl l = some (u, v) ∧ Collision H u v) →
      Claims.Recursion.forkCollV H (deployRecC Bd ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO P x
        j t := by
    intro l hsub
    refine hliftV l fun p hp => ?_
    rw [hD₁, hD₂]
    rcases List.mem_append.1 (hsub p hp) with h | h
    · exact List.mem_append_left _ (List.mem_append_left _ h)
    · exact List.mem_append_right _ (List.mem_append_left _ h)
  have hleft := hlift ((deployRec Bd.IVStar ivScope b).calls x ws) fun p hp =>
    List.mem_append_left _ (Claims.Recursion.mem_callsAt hp)
  have hright := hlift ((deployRec Bd.IVStar ivScope b).calls x ws') fun p hp =>
    List.mem_append_right _ (Claims.Recursion.mem_callsAt hp)
  have hleftS := hlift ((deployRec Bd.IVStar ivScope b).callsAt H x t.2.1 ws) fun p hp =>
    List.mem_append_left _ hp
  have hrightS := hlift ((deployRec Bd.IVStar ivScope b).callsAt H x (Claims.Recursion.fork j t.2.1 t.2.2.1) ws') fun p hp =>
    List.mem_append_right _ hp
  have hv : ∀ (r₀ : Fin ((deployRecC Bd ivScope b).V.k x) → Coin) (ws₀ : List Claims.Flock.Wit),
      (((deployRecC Bd ivScope b).innerView H x r₀ ws₀).msgs.map Prod.fst).getD j [] =
        if j < Bd.DE.V.k (xIOf x) then (msgsC Bd b H x r₀ ws₀).getD j [] else [] :=
    fun r₀ ws₀ => Bd.run_ivstar_fst H (xIOf x) (msgsC Bd b H x r₀ ws₀) (fun i => coinSeq r₀ i.val) j
  have hne' : (if j < Bd.DE.V.k (xIOf x) then (msgsC Bd b H x t.2.1 ws).getD j [] else []) ≠
      (if j < Bd.DE.V.k (xIOf x) then (msgsC Bd b H x (Claims.Recursion.fork j t.2.1 t.2.2.1) ws').getD j []
        else []) :=
    fun h => hne ((hv t.2.1 ws).trans (h.trans (hv (Claims.Recursion.fork j t.2.1 t.2.2.1) ws').symm))
  by_cases hjK : j < Bd.DE.V.k (xIOf x)
  swap
  · exact absurd (by simp [hjK]) hne'
  simp only [hjK, ↓reduceIte] at hne'
  rcases hP decO Ω P t.1 t.2.1 t.2.2.2.1 ws hg₁' with h₁ | h₁
  swap
  · exact hleft h₁
  rcases hP decO Ω P t.1 (Claims.Recursion.fork j t.2.1 t.2.2.1) t.2.2.2.2 ws' hg₂' with h₂ | h₂
  swap
  · exact hright h₂
  have hag : ∀ i, i + 1 < Bd.DE.V.k (xIOf x) → i ≤ j → (b.rd.msgs x ws).getD i [] = (b.rd.msgs x ws').getD i [] :=
    fun i hi hij => (h₁ i hi).trans ((congrArg (pk x i) (Claims.Recursion.run_fst_getD H P _ x t.1 i fun i' hi' =>
      Claims.Recursion.fork_agree j t.2.1 t.2.2.1 i' (by omega))).trans (h₂ i hi).symm)
  unfold msgsC at hne'
  by_cases hj1 : j + 1 < Bd.DE.V.k (xIOf x)
  · rw [Bd.nfMsgsC_getD_lt _ _ _ _ hj1, Bd.nfMsgsC_getD_lt _ _ _ _ hj1] at hne'
    exact absurd (hag j hj1 le_rfl) hne'
  have hk0 : 0 < Bd.DE.V.k (xIOf x) := by omega
  have hj : j + 1 = Bd.DE.V.k (xIOf x) := by omega
  rw [Bd.nfMsgsC_getD_eq H (xIOf x) _ _ hj, Bd.nfMsgsC_getD_eq H (xIOf x) _ _ hj] at hne'
  have hcs : ∀ i, i < j → coinSeq (fun i : Fin (Bd.IVStar.k (xIOf x)) => coinSeq t.2.1 i.val) i =
      coinSeq (fun i : Fin (Bd.IVStar.k (xIOf x)) => coinSeq (Claims.Recursion.fork j t.2.1 t.2.2.1) i.val) i :=
    fun i hi => Claims.Recursion.coinSeq_congr fun _ =>
      Claims.Recursion.coinSeq_congr fun h => Claims.Recursion.fork_agree j t.2.1 t.2.2.1 ⟨i, h⟩ hi
  have hcsd : (coinsC Bd x t.2.1).dropLast = (coinsC Bd x (Claims.Recursion.fork j t.2.1 t.2.2.1)).dropLast := by
    unfold coinsC
    rw [Claims.Recursion.range_map_dropLast, Claims.Recursion.range_map_dropLast]
    exact List.map_congr_left fun i hi => hcs i (by rw [List.mem_range] at hi; omega)
  have hR := Bd.readL_ne_of_repL hCF hk0 (by simp [coinsC]) (by simp [coinsC]) (fun i hi => hag i hi (by omega))
    hcsd hne'
  rcases hA decO Ω P t.1 t.2.1 t.2.2.2.1 ws hg₁' with ha₁ | ha₁
  swap
  · refine hliftV _ (fun p hp => ?_) ha₁
    rw [hD₁, hD₂]
    rcases List.mem_append.1 hp with h | h
    · exact List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ h))
    · exact List.mem_append_left _ (List.mem_append_right _ h)
  rcases hA decO Ω P t.1 (Claims.Recursion.fork j t.2.1 t.2.2.1) t.2.2.2.2 ws' hg₂' with ha₂ | ha₂
  swap
  · refine hliftV _ (fun p hp => ?_) ha₂
    rw [hD₁, hD₂]
    rcases List.mem_append.1 hp with h | h
    · exact List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ h))
    · exact List.mem_append_right _ (List.mem_append_right _ h)
  rcases hL decO x Ω P t.1 t.2.1 t.2.2.2.1 ws hg₁ with hl₁ | hl₁
  swap
  · exact hleftS hl₁
  rcases hL decO x Ω P t.1 (Claims.Recursion.fork j t.2.1 t.2.2.1) t.2.2.2.2 ws' hg₂ with hl₂ | hl₂
  swap
  · exact hrightS hl₂
  have hacc : ∀ (r₀ : Fin ((deployRec Bd.IVStar ivScope b).V.k x) → Coin) (ws₀ : List Claims.Flock.Wit),
      ((deployRec Bd.IVStar ivScope b).innerView H x r₀ ws₀).out = .acc →
      (Bd.decAt H (xIOf x) (b.rd.msgs x ws₀) fun i => coinSeq r₀ i.val).1 = .acc := by
    intro r₀ ws₀ h
    change (run H (replay (b.rd.msgs x ws₀)) Bd.IVStar (xIOf x) () fun i => coinSeq r₀ i.val).out = .acc at h
    rw [Bd.run_ivstar H (xIOf x) _ _ hk0] at h
    exact h
  have hlog : ∀ (r₀ : Fin ((deployRec Bd.IVStar ivScope b).V.k x) → Coin) (ws₀ : List Claims.Flock.Wit),
      (∀ e ∈ (Claims.Recursion.innerRun H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) x r₀
        ws₀).vLog, e ∈ (deployRec Bd.IVStar ivScope b).callsAt H x r₀ ws₀) →
      ∀ e ∈ (Bd.decAt H (xIOf x) (b.rd.msgs x ws₀) fun i => coinSeq r₀ i.val).2,
        e ∈ (deployRec Bd.IVStar ivScope b).callsAt H x r₀ ws₀ := by
    intro r₀ ws₀ h e he
    apply h
    rw [innerRun_bound, Bd.run_ivstar H (xIOf x) _ _ hk0]
    exact he
  have hacc₁ := hacc _ _ ha₁
  have hacc₂ := hacc _ _ ha₂
  have hlog₁ := hlog _ _ hl₁
  have hlog₂ := hlog _ _ hl₂
  obtain ⟨q, hq, q', hq', hc⟩ := hEC (xIOf x)
    ((List.range (Bd.DE.V.k (xIOf x))).map fun i => (b.rd.msgs x ws).getD i [])
    ((List.range (Bd.DE.V.k (xIOf x))).map fun i => (b.rd.msgs x ws').getD i [])
    (coinsC Bd x t.2.1) (coinsC Bd x (Claims.Recursion.fork j t.2.1 t.2.2.1))
    (by simp) (by simp) (by simp [coinsC]) (by simp [coinsC])
    (by
      rw [Claims.Recursion.range_map_dropLast, Claims.Recursion.range_map_dropLast]
      exact List.map_congr_left fun i hi => by
        rw [List.mem_range] at hi
        exact hag i (by omega) (by omega))
    hcsd (Bd.readL_ne hR (hacc₁.trans hacc₂.symm)) hacc₁ hacc₂
  have hsub : ∀ e ∈ (Bd.decAt H (xIOf x) (b.rd.msgs x ws) fun i => coinSeq t.2.1 i.val).2 ++
      (Bd.decAt H (xIOf x) (b.rd.msgs x ws') fun i => coinSeq (Claims.Recursion.fork j t.2.1 t.2.2.1) i.val).2,
      e ∈ (deployRec Bd.IVStar ivScope b).callsAt H x t.2.1 ws ++
        (deployRec Bd.IVStar ivScope b).callsAt H x (Claims.Recursion.fork j t.2.1 t.2.2.1) ws' := fun e he =>
    (List.mem_append.1 he).elim (fun h => List.mem_append_left _ (hlog₁ e h))
      (fun h => List.mem_append_right _ (hlog₂ e h))
  have hmem : ∀ y ∈ ((Bd.decAt H (xIOf x) (b.rd.msgs x ws) fun i => coinSeq t.2.1 i.val).2 ++
      (Bd.decAt H (xIOf x) (b.rd.msgs x ws') fun i =>
        coinSeq (Claims.Recursion.fork j t.2.1 t.2.2.1) i.val).2).map Prod.fst,
      y ∈ ((deployRec Bd.IVStar ivScope b).callsAt H x t.2.1 ws ++
        (deployRec Bd.IVStar ivScope b).callsAt H x (Claims.Recursion.fork j t.2.1 t.2.2.1) ws').map Prod.fst := fun y hy => by
    obtain ⟨e, he, rfl⟩ := List.mem_map.1 hy
    exact List.mem_map.2 ⟨e, hsub e he, rfl⟩
  exact hlift _ (fun p hp => hp) (Claims.Recursion.findColl_of_coll H htrue (hmem q hq) (hmem q' hq') hc)

/-- `forksBindAt_C_pick_setup` with the readers' rounds before the last the runs' own (`pk x j m = m`). -/
theorem forksBindAt_C_setup {Col' T' : Type} [DecidableEq Col] [DecidableEq Col'] (H : Msg → Msg)
    (Bd : Bound Coin Col T)
    (decI : List (ℕ × Col) → List T) (ivScope : Msg → Bool) (b : Brings) (DO : Claims.Flock.Deploy Coin Col' T')
    (hEC : Bd.LastBindsEC H) (hCF : Bd.LastCoinFree H)
    (hT : Claims.Recursion.CallsTrue H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO)
    (hL : Claims.Recursion.CallsLogCollAt H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO)
    (hP : ∀ (decO : List (ℕ × Col') → List T') (x : Msg) (Ω : Type) (P : Prover Ω) (ω : Ω)
      (r : Fin ((deployRec Bd.IVStar ivScope b).V.k x) → Coin)
      (c : (Claims.Recursion.EO (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO).C x)
      (ws : List Claims.Flock.Wit),
      Claims.Recursion.EOGives H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO P x ω r c
        ws →
      (∀ j, j + 1 < Bd.DE.V.k (xIOf x) → (b.rd.msgs x ws).getD j [] =
          ((run H P (deployRec Bd.IVStar ivScope b).V x ω r).msgs.map Prod.fst).getD j []) ∨
        ∃ u v, findColl ((deployRec Bd.IVStar ivScope b).calls x ws) = some (u, v) ∧ Collision H u v)
    (x : Msg)
    (hA : Claims.Recursion.InnerAcceptsCollAtV H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO
      x) :
    Claims.Recursion.ForksBindAtV H (deployRecC Bd ivScope b) (Inner.ofFlock Bd.deploy decI) DO x :=
  forksBindAt_C_pick_setup H Bd decI ivScope b DO (fun _ _ m => m) hEC hCF hT hL x (fun decO => hP decO x) hA

end Cols

end RecDeploy
