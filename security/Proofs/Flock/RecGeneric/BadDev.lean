module

public import Proofs.Flock.RecGeneric.Coupling
public import Proofs.Flock.Claims.Recursion.PlurLin
public import Proofs.Flock.Claims.Recursion.Hybrid
public import Proofs.Flock.Claims.Recursion.Gap

/-!
# `P_O`'s bad samples, linearly, through the gated link

A sample `P_O`'s runs don't show (`Comp.BadO`) misses the plurality of what `P_O` reads (`Comp.fAt`) at some inner round
`j`. By `Pr_plur_fork_lin`, at each `j` that costs at most the link at the fork at `j` plus `1/(M+1)`, the link sampling
`M` more forks at `j` until the first where `R_O` holds. The gated link (`linkPlan` at `verModel`) asks only whether
`E_O` gives witnesses, so its first giving fork is in `R_O`, where two forks at `j` that read differently collide
(`Comp.pair_coll`), or outside it, which over the loop costs at most `PrOut` (`Pr_outLoop_le`, Wald's bound on the
loop's first giving sample). Two forks that read differently collide: before the last inner round their openings differ,
so either their round messages differ (`ForksBind`) or one commit string opens to two rows (`Server.Binds`, through
`binds_tied`, the openings' calls among the inner verifier's and so among `calls`, `CallsLogColl`); at the last round
their classes differ, so their views differ at some round (`Server.run_ivstar_lastCoin`, from `Proto.LastCoinFree`),
and `ForksBind` there. So `Pr[BadO] ≤ k_Π·(Pr[link] + 1/(M+1) + PrOut)` (`Comp.Pr_badO_le_link`), at any compiled
verifier, outer extraction and server.

A judged rerun whose witnesses are in `R_O` and whose inner view rejects shows a collision among its own evaluations
(`accepts`), which `forkPlan` searches at its first run: `Pr[InnerRejectsO] ≤ Pr[B_I]`
(`Comp.Pr_innerRejects_le_fork`).
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion

/-! ## `Open(V, S)` reads no last coin -/

namespace Server

theorem eval_bind_congr (H : Msg → Msg) {α β : Type} {p p' : OC α} {f f' : α → OC β}
    (hp : p.eval H = p'.eval H) (hf : ∀ a, (f a).eval H = (f' a).eval H) :
    (bindOC p f).eval H = (bindOC p' f').eval H := by
  rw [eval_bindOC_full, eval_bindOC_full, hp, hf]

theorem eval_allOC_congr (H : Msg → Msg) {ι : Type} (f g : ι → OC Bool) :
    ∀ l : List ι, (∀ j ∈ l, (f j).eval H = (g j).eval H) → (allOC (l.map f)).eval H = (allOC (l.map g)).eval H
  | [], _ => rfl
  | j :: l, h => by
    simp only [List.map_cons, allOC]
    refine eval_bind_congr H (h j (by simp)) fun b => ?_
    cases b
    · rfl
    · exact eval_allOC_congr H f g l fun i hi => h i (List.mem_cons_of_mem _ hi)

theorem take_eq_of_dropLast {α : Type} {l l' : List α} (hl : l.length = l'.length) (hd : l.dropLast = l'.dropLast)
    {n : ℕ} (hn : n < l.length) : l.take n = l'.take n := by
  have h1 : ∀ m : List α, n < m.length → m.take n = m.dropLast.take n := fun m hm => by
    rw [List.dropLast_eq_take, List.take_take]
    congr 1
    omega
  rw [h1 l hn, h1 l' (by omega), hd]

/-- The inner verifier accepts the messages `ms` alike at two coin lists that differ only in the last coin. -/
theorem agrees_lastCoin {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {H : Msg → Msg}
    {I : Proto Coin W} (hCF : I.LastCoinFree H) {xI : Msg} {ms : List Msg} {cs cs' : List Coin}
    (hk : ms.length = I.V.k xI) (hc : cs.length = ms.length) (hc' : cs'.length = ms.length)
    (hd : cs.dropLast = cs'.dropLast) :
    (Server.agrees I.V I.ans xI ms cs).eval H = (Server.agrees I.V I.ans xI ms cs').eval H := by
  unfold Server.agrees
  refine eval_allOC_congr H _ _ _ fun j hj => ?_
  rw [List.mem_range] at hj
  by_cases hj1 : j + 1 < ms.length
  · have ht : cs.take (j + 1) = cs'.take (j + 1) := take_eq_of_dropLast (hc.trans hc'.symm) hd (by omega)
    have hg : cs.getD j default = cs'.getD j default := by
      rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD,
        ← List.getElem?_take_of_lt (show j < j + 1 by omega), ht, List.getElem?_take_of_lt (show j < j + 1 by omega)]
    rw [ht, hg]
  · rw [ite_eq_right hj1, ite_eq_right hj1, List.take_of_length_le (show ms.length ≤ j + 1 by omega),
      List.take_of_length_le (show cs.length ≤ j + 1 by omega),
      List.take_of_length_le (show cs'.length ≤ j + 1 by omega)]
    exact hCF xI ms cs cs' hk hc hc' hd

/-- `Open(V, S)` decides alike at two coin lists that differ only in the last coin. -/
theorem ivstar_lastCoin {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {H : Msg → Msg}
    {I : Proto Coin W} (hCF : I.LastCoinFree H) (S : Server) (xI : Msg) (as : List Msg) (cs cs' : List Coin)
    (hk : as.length = I.V.k xI) (hc : cs.length = as.length) (hc' : cs'.length = as.length)
    (hd : cs.dropLast = cs'.dropLast) :
    ((S.IVStar I.V I.ans).next xI as cs).eval H = ((S.IVStar I.V I.ans).next xI as cs').eval H := by
  have hlt : ¬ as.length < I.V.k xI := by omega
  simp only [Server.IVStar, hlt, ↓reduceIte]
  cases hfin : S.final (as.getLastD []) with
  | none => rfl
  | some p =>
    obtain ⟨os, last⟩ := p
    dsimp only
    by_cases hcnd : os.length + 1 = as.length ∧ ∀ j < os.length, (os.getD j []).length = S.nAt xI j
    · rw [ite_eq_left hcnd, ite_eq_left hcnd]
      refine eval_bind_congr H rfl fun ok => ?_
      cases ok
      · rfl
      · have hlen : (S.opened os last).length = as.length := by
          simp only [Server.opened, List.length_append, List.length_map, List.length_singleton]
          exact hcnd.1
        exact eval_bind_congr H (agrees_lastCoin hCF (hlen.trans hk) (hc.trans hlen.symm) (hc'.trans hlen.symm) hd)
          fun _ => rfl
    · rw [ite_eq_right hcnd, ite_eq_right hcnd]

/-- **`Open(V, S)`'s runs read no last coin**: on messages alike at its rounds and coins alike before its last round,
its runs are one. -/
theorem run_ivstar_lastCoin {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {H : Msg → Msg}
    {I : Proto Coin W} (hCF : I.LastCoinFree H) (S : Server) (xI : Msg) (ms ms' : List Msg)
    (r r' : Fin ((S.IVStar I.V I.ans).k xI) → Coin) (hms : ∀ i < I.V.k xI, ms.getD i [] = ms'.getD i [])
    (hr : ∀ i, i + 1 < I.V.k xI → coinSeq r i = coinSeq r' i) :
    run H (replay ms) (S.IVStar I.V I.ans) xI () r = run H (replay ms') (S.IVStar I.V I.ans) xI () r' := by
  by_cases hk : 0 < I.V.k xI
  · have hm : (List.range (I.V.k xI)).map (fun i => ms.getD i []) =
        (List.range (I.V.k xI)).map (fun i => ms'.getD i []) :=
      List.map_congr_left fun i hi => hms i (List.mem_range.1 hi)
    have hdec : S.decAt I.V I.ans H xI ms r = S.decAt I.V I.ans H xI ms' r' := by
      unfold Server.decAt
      rw [hm]
      refine ivstar_lastCoin hCF S xI _ _ _ (by simp) (by simp) (by simp) ?_
      rw [List.dropLast_eq_take, List.dropLast_eq_take]
      simp only [List.length_map, List.length_range]
      rw [take_range_map _ (Nat.sub_le _ _), take_range_map _ (Nat.sub_le _ _)]
      exact List.map_congr_left fun i hi => hr i (by rw [List.mem_range] at hi; omega)
    rw [S.run_ivstar I.V I.ans H xI ms r hk, S.run_ivstar I.V I.ans H xI ms' r' hk, hdec]
    have hA : (List.range (I.V.k xI - 1)).map (fun i => (ms.getD i [], Out.msg (I.ans (coinSeq r i)))) =
        (List.range (I.V.k xI - 1)).map (fun i => (ms'.getD i [], Out.msg (I.ans (coinSeq r' i)))) :=
      List.map_congr_left fun i hi => by
        rw [List.mem_range] at hi
        rw [hms i (by omega), hr i (by omega)]
    rw [hA, hms (I.V.k xI - 1) (by omega)]
  · rw [S.run_ivstar_zero I.V I.ans H xI ms r (by omega), S.run_ivstar_zero I.V I.ans H xI ms' r' (by omega)]

variable {Coin : Type} [Inhabited Coin]

/-- `Open(V, S)`'s run on replayed messages shows them at its rounds. -/
theorem run_ivstar_raw (S : Server) (V : Verifier Coin) (ans : Coin → Msg) (H : Msg → Msg) (x : Msg)
    (ms : List Msg) (r : Fin ((S.IVStar V ans).k x) → Coin) {i : ℕ} (hi : i < V.k x) :
    ((run H (replay ms) (S.IVStar V ans) x () r).msgs.map Prod.fst).getD i [] = ms.getD i [] := by
  have hk : 0 < V.k x := by omega
  have hr : List.range (V.k x) = List.range (V.k x - 1) ++ [V.k x - 1] := by
    obtain ⟨n, hn⟩ : ∃ n, V.k x = n + 1 := ⟨V.k x - 1, by omega⟩
    rw [hn, Nat.add_sub_cancel, List.range_succ]
  have e : (run H (replay ms) (S.IVStar V ans) x () r).msgs.map Prod.fst =
      (List.range (V.k x)).map fun i => ms.getD i [] := by
    rw [S.run_ivstar V ans H x ms r hk, hr]
    simp only [List.map_append, List.map_map, List.map_singleton, Function.comp_def]
  rw [e, getD_range_map _ _ hi]

end Server

/-! ## Collisions from two openings, and the link's loop -/

theorem fork_of_le {Coin : Type} {k i j : ℕ} {r r' : Fin k → Coin} (h : i ≤ j) (hf : fork j r r' = r') :
    fork i r r' = r' := by
  funext t
  have ht := congrFun hf t
  simp only [fork] at ht ⊢
  by_cases h1 : t.val < i
  · rw [ite_eq_left h1]
    rw [ite_eq_left (show t.val < j by omega)] at ht
    exact ht
  · exact ite_eq_right h1

theorem coll_of_isSome {H : Msg → Msg} {L : List (Msg × Msg)} (hT : ∀ p ∈ L, H p.1 = p.2)
    (h : (findColl L).isSome = true) : ∃ a b, findColl L = some (a, b) ∧ Collision H a b := by
  obtain ⟨⟨a, b⟩, hab⟩ := Option.isSome_iff_exists.1 h
  exact ⟨a, b, hab, findColl_spec H L hT a b hab⟩

/-- **One round message opened two ways collides** where the server binds: two openings of one round message to rows
that differ, each passing, show a collision among their calls. -/
theorem binds_tied {H : Msg → Msg} {S : Server} (hB : S.Binds) {a : Msg} {o₁ o₂ : List (Msg × S.Salt)}
    {L : List (Msg × Msg)} (h₁ : ((S.opensRound a o₁).eval H).1 = true) (h₂ : ((S.opensRound a o₂).eval H).1 = true)
    (hl : o₁.length = o₂.length) (hne : o₁.map Prod.fst ≠ o₂.map Prod.fst)
    (hs₁ : ∀ e ∈ ((S.opensRound a o₁).eval H).2, e ∈ L) (hs₂ : ∀ e ∈ ((S.opensRound a o₂).eval H).2, e ∈ L) :
    TiedColl H L := by
  obtain ⟨i, hi₁, hi₂, hdi⟩ : ∃ i, ∃ (hi₁ : i < o₁.length) (hi₂ : i < o₂.length), o₁[i].1 ≠ o₂[i].1 := by
    by_contra hc
    push Not at hc
    exact hne (List.ext_getElem (by simp [hl]) fun i h1 h2 => by
      simp only [List.getElem_map]
      exact hc i (by simpa using h1) (by simpa using h2))
  obtain ⟨-, ho₁⟩ := S.opensRound_true H h₁
  obtain ⟨-, ho₂⟩ := S.opensRound_true H h₂
  obtain ⟨hp₁, hl₁⟩ := ho₁ i hi₁
  obtain ⟨hp₂, hl₂⟩ := ho₂ i hi₂
  obtain ⟨q, hq, q', hq', hc⟩ := hB H _ _ _ _ _ hdi hp₁ hp₂
  have hm : ∀ q ∈ queries H (S.opens ((a.drop (S.width * i)).take S.width) o₁[i].1 o₁[i].2) ++
      queries H (S.opens ((a.drop (S.width * i)).take S.width) o₂[i].1 o₂[i].2), q ∈ L.map Prod.fst := by
    intro q hq
    simp only [queries, List.mem_append, List.mem_map] at hq
    rcases hq with ⟨e, he, rfl⟩ | ⟨e, he, rfl⟩
    · exact List.mem_map.2 ⟨e, hs₁ e (hl₁ e he), rfl⟩
    · exact List.mem_map.2 ⟨e, hs₂ e (hl₂ e he), rfl⟩
  exact ⟨q, q', hc, hm q hq, hm q' hq'⟩

/-- **The link's loop finds the first accepted fork**: where `firstPick` finds `v` among the loop's forks, the loop
searches its first run and a run named as the fork of the shared prefix with `v`. -/
theorem linkLoop_exec {Coin Ω C : Type} {n : ℕ} (m : RunModel Coin Ω n C) (ω : Ω) (us : ℕ → (Fin n → Coin) × C)
    (j : ℕ) (first : RunExp) : ∀ (N i : ℕ) (v : (Fin n → Coin) × C),
      firstPick (fun u : (Fin n → Coin) × C => m.acc ω (fork (m.cut j) (us 0).1 u.1, u.2)) N
          (fun t : Fin N => us (i + t.val)) = some v →
        ∃ e, ((linkLoop j first N i).exec m ω us).2.2 = [first, e] ∧
          m.named us e = (fork (m.cut j) (us 0).1 v.1, v.2)
  | 0 => fun i v h => by simp [firstPick] at h
  | N + 1 => fun i v h => by
    have htail : Fin.tail (fun t : Fin (N + 1) => us (i + t.val)) = fun t : Fin N => us (i + 1 + t.val) :=
      funext fun t => congrArg us (by simp only [Fin.val_succ]; omega)
    simp only [firstPick, Fin.val_zero, Nat.add_zero, htail] at h
    by_cases ha : m.acc ω (fork (m.cut j) (us 0).1 (us i).1, (us i).2)
    · simp only [ha, ↓reduceIte] at h
      have hv := Option.some_injective _ h
      subst hv
      refine ⟨.fork j (.fresh 0) (.fresh i), ?_, rfl⟩
      simp only [linkLoop, Plan.exec, RunModel.named, decide_eq_true_eq, ha, ↓reduceIte]
    · simp only [ha, ↓reduceIte] at h
      obtain ⟨e, he, hn⟩ := linkLoop_exec m ω us j first N (i + 1) v h
      refine ⟨e, ?_, hn⟩
      simp only [linkLoop, Plan.exec, RunModel.named, decide_eq_true_eq, ha, ↓reduceIte]
      exact he

/-- The link's slots, read as its first run's coins, its first fork and the loop's samples, are uniform. -/
theorem unifMap_link {A B : Type} [Fintype A] [Fintype B] [Nonempty B] (M : ℕ) :
    UnifMap (fun sl : Fin (M + 1 + 1) → A × B => ((sl 0).1, Fin.tail sl 0, Fin.tail (Fin.tail sl))) := by
  have h1 : UnifMap (fun sl : Fin (M + 1 + 1) → A × B => (sl 0, Fin.tail sl)) :=
    UnifMap.equiv (splitFirst (U := A × B) (M + 1))
  have h2 : UnifMap (fun p : (A × B) × (Fin (M + 1) → A × B) => (p.1.1, p.2 0, Fin.tail p.2)) :=
    UnifMap.prod (φ := fun q : A × B => q.1) (ψ := splitFirst (U := A × B) M) UnifMap.fst (UnifMap.equiv _)
  exact h1.comp h2

open Classical in
/-- **A loop over `gR`'s samples, read through `gv ⊇ gR`**: where the first `gR` sample differs, the first `gv` sample is a `gR`
sample that differs, or a `gv` sample outside `gR`, there in `o`. -/
theorem firstDiffers_split {U : Type} {gR gv o : U → Prop} {f : U → Msg} {a : Msg} (hg : ∀ u, gR u → gv u)
    (hgo : ∀ u, gv u → ¬ gR u → o u) : ∀ M (us : Fin M → U), firstDiffers gR f a M us →
      (∃ v, firstPick gv M us = some v ∧ gR v ∧ f v ≠ a) ∨
        firstDiffers gv (fun u => if o u then [0] else []) [] M us
  | 0, _, h => h.elim
  | M + 1, us, h => by
    by_cases hv : gv (us 0)
    · have hp : firstPick gv (M + 1) us = some (us 0) := by simp only [firstPick, hv, ↓reduceIte]
      rcases h with ⟨hr, hf⟩ | ⟨hr, -⟩
      · exact Or.inl ⟨us 0, hp, hr, hf⟩
      · exact Or.inr (Or.inl ⟨hv, by simp [hgo _ hv hr]⟩)
    · have hp : firstPick gv (M + 1) us = firstPick gv M (Fin.tail us) := by
        simp only [firstPick, hv, ↓reduceIte]
      rcases h with ⟨hr, -⟩ | ⟨-, h⟩
      · exact absurd (hg _ hr) hv
      · rcases firstDiffers_split hg hgo M (Fin.tail us) h with ⟨v, hv', hr, hf⟩ | h'
        · exact Or.inl ⟨v, hp.trans hv', hr, hf⟩
        · exact Or.inr (Or.inr ⟨hv, h'⟩)

open Classical in
/-- **Wald's bound on the loop's first `gv` sample**: a `gR` sample, then a loop whose first `gv` sample is in `o`, has
probability at most `Pr[o]`: `Pr[gR]·Pr[gv ∧ o]·G_M ≤ Pr[o]·(1 − (1 − Pr[gv])^M)`. -/
theorem Pr_firstLoop_le {U : Type} [Fintype U] [Nonempty U] (gR gv o : U → Prop) (hR : ∀ u, gR u → gv u) (M : ℕ) :
    Pr (fun p : U × (Fin M → U) => gR p.1 ∧ firstDiffers gv (fun u => if o u then [0] else []) [] M p.2) ≤
      Pr o := by
  have hO0 : 0 ≤ Pr o := Pr_nonneg _
  have hq0 : 0 ≤ Pr gv := Pr_nonneg _
  have hq1 : Pr gv ≤ 1 := Pr_le_one _
  have hS0 : 0 ≤ geomS (1 - Pr gv) M := geomS_nonneg (by linarith) M
  have ht : 0 ≤ (1 - Pr gv) ^ M := pow_nonneg (by linarith) M
  have hfd : Pr (firstDiffers gv (fun u => if o u then [0] else []) [] M) ≤ Pr o * geomS (1 - Pr gv) M := by
    rw [Pr_firstDiffers]
    refine mul_le_mul_of_nonneg_right (Pr_mono fun u h => ?_) hS0
    by_contra hn
    exact h.2 (ite_eq_right hn)
  have hL : Pr (fun p : U × (Fin M → U) => gR p.1 ∧
      firstDiffers gv (fun u => if o u then [0] else []) [] M p.2) =
      Pr (firstDiffers gv (fun u => if o u then [0] else []) [] M) * Pr gR := by
    rw [Claims.Recursion.Pr_prod, ← Ex_if_zero]
    refine congrArg Ex (funext fun u => ?_)
    by_cases hu : gR u
    · simp only [hu, true_and, ↓reduceIte]
    · simp only [hu, false_and, ↓reduceIte, Pr_false]
  have hgm := geomS_mul (Pr gv) M
  rw [hL]
  calc Pr (firstDiffers gv (fun u => if o u then [0] else []) [] M) * Pr gR
      ≤ (Pr o * geomS (1 - Pr gv) M) * Pr gv := mul_le_mul hfd (Pr_mono hR) (Pr_nonneg _) (mul_nonneg hO0 hS0)
    _ = Pr o - Pr o * (1 - Pr gv) ^ M := by
        have e : Pr o * geomS (1 - Pr gv) M * Pr gv = Pr o * (Pr gv * geomS (1 - Pr gv) M) := by ring
        rw [e, hgm]
        ring
    _ ≤ Pr o := by linarith [mul_nonneg hO0 ht]

section Loop

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] {Ω C : Type} [Fintype Ω] [Nonempty Ω] [Fintype C]
  [Nonempty C] {N : ℕ}

omit [Nonempty Ω] in
open Classical in
/-- **The loop's first giving fork in `o` costs `Pr[o]`**: over a tape, a prefix, a `gR` fork and `M` more forks at `j`,
the first `gv` fork in `o` has probability at most `Pr[o]` over one sample. -/
theorem Pr_outLoop_le (j M : ℕ) (gR gv o : Ω → (Fin N → Coin) × C → Prop) (hR : ∀ ω d, gR ω d → gv ω d) :
    Pr (fun t : Ω × (Fin N → Coin) × ((Fin N → Coin) × C) × (Fin M → (Fin N → Coin) × C) =>
      gR t.1 (fork j t.2.1 t.2.2.1.1, t.2.2.1.2) ∧
        firstDiffers (fun u : (Fin N → Coin) × C => gv t.1 (fork j t.2.1 u.1, u.2))
          (fun u => if o t.1 (fork j t.2.1 u.1, u.2) then [0] else []) [] M t.2.2.2) ≤
      Pr (fun t : Ω × (Fin N → Coin) × C => o t.1 t.2) := by
  have hoff : PrefixDet (k := N) (Coin := Coin) fun _ => j := fun _ _ _ => rfl
  let a : Ω → (Fin N → Coin) → ℝ := fun ω r => Pr fun u : (Fin N → Coin) × C => o ω (fork j r u.1, u.2)
  let b : Ω → (Fin N → Coin) → ℝ := fun ω r =>
    Pr fun p : ((Fin N → Coin) × C) × (Fin M → (Fin N → Coin) × C) =>
      gR ω (fork j r p.1.1, p.1.2) ∧
        firstDiffers (fun u : (Fin N → Coin) × C => gv ω (fork j r u.1, u.2))
          (fun u => if o ω (fork j r u.1, u.2) then [0] else []) [] M p.2
  have hb : ∀ ω r, b ω r ≤ a ω r := fun ω r =>
    Pr_firstLoop_le (U := (Fin N → Coin) × C) (fun u => gR ω (fork j r u.1, u.2)) (fun u => gv ω (fork j r u.1, u.2))
      (fun u => o ω (fork j r u.1, u.2)) (fun u => hR ω _) M
  have hL : Pr (fun t : Ω × (Fin N → Coin) × ((Fin N → Coin) × C) × (Fin M → (Fin N → Coin) × C) =>
      gR t.1 (fork j t.2.1 t.2.2.1.1, t.2.2.1.2) ∧
        firstDiffers (fun u : (Fin N → Coin) × C => gv t.1 (fork j t.2.1 u.1, u.2))
          (fun u => if o t.1 (fork j t.2.1 u.1, u.2) then [0] else []) [] M t.2.2.2) =
      Ex fun ω => Ex fun r => b ω r := by
    rw [Claims.Recursion.Pr_prod]
    refine congrArg Ex (funext fun ω => ?_)
    rw [Claims.Recursion.Pr_prod]
  have hRt : Pr (fun t : Ω × (Fin N → Coin) × C => o t.1 t.2) = Ex fun ω => Ex fun r => a ω r := by
    rw [Claims.Recursion.Pr_prod]
    refine congrArg Ex (funext fun ω => ?_)
    rw [Claims.Recursion.Pr_prod, ← Ex_fork hoff (fun r => Pr fun c : C => o ω (r, c)), Ex_prod]
    refine congrArg Ex (funext fun r => ?_)
    show _ = Pr (fun u : (Fin N → Coin) × C => o ω (fork j r u.1, u.2))
    rw [Claims.Recursion.Pr_prod (fun u : (Fin N → Coin) × C => o ω (fork j r u.1, u.2))]
  rw [hL, hRt]
  exact Ex_mono fun ω => Ex_mono fun r => hb ω r

omit [Nonempty Coin] [Nonempty Ω] [Nonempty C] in
open Classical in
/-- **The link through `gv ⊇ gR`**: the link's success at `j` splits into its first `gv` fork being a `gR` fork that
reads differently, or a `gv` fork outside `gR`, there in `o`. -/
theorem Pr_link_split (j M : ℕ) (gR gv o : Ω → (Fin N → Coin) × C → Prop) (f : Ω → (Fin N → Coin) × C → Msg)
    (hg : ∀ ω d, gR ω d → gv ω d) (hgo : ∀ ω d, gv ω d → ¬ gR ω d → o ω d) :
    Pr (fun t : Ω × (Fin N → Coin) × ((Fin N → Coin) × C) × (Fin M → (Fin N → Coin) × C) =>
      gR t.1 (fork j t.2.1 t.2.2.1.1, t.2.2.1.2) ∧
        firstDiffers (fun u : (Fin N → Coin) × C => gR t.1 (fork j t.2.1 u.1, u.2))
          (fun u => f t.1 (fork j t.2.1 u.1, u.2)) (f t.1 (fork j t.2.1 t.2.2.1.1, t.2.2.1.2)) M t.2.2.2) ≤
      Pr (fun t : Ω × (Fin N → Coin) × ((Fin N → Coin) × C) × (Fin M → (Fin N → Coin) × C) =>
        gR t.1 (fork j t.2.1 t.2.2.1.1, t.2.2.1.2) ∧
          ∃ v, firstPick (fun u : (Fin N → Coin) × C => gv t.1 (fork j t.2.1 u.1, u.2)) M t.2.2.2 = some v ∧
            gR t.1 (fork j t.2.1 v.1, v.2) ∧
            f t.1 (fork j t.2.1 v.1, v.2) ≠ f t.1 (fork j t.2.1 t.2.2.1.1, t.2.2.1.2)) +
      Pr (fun t : Ω × (Fin N → Coin) × ((Fin N → Coin) × C) × (Fin M → (Fin N → Coin) × C) =>
        gR t.1 (fork j t.2.1 t.2.2.1.1, t.2.2.1.2) ∧
          firstDiffers (fun u : (Fin N → Coin) × C => gv t.1 (fork j t.2.1 u.1, u.2))
            (fun u => if o t.1 (fork j t.2.1 u.1, u.2) then [0] else []) [] M t.2.2.2) := by
  refine (Pr_mono fun t ht => ?_).trans (Claims.Pr_or_le _ _)
  rcases firstDiffers_split (gR := fun u => gR t.1 (fork j t.2.1 u.1, u.2))
      (gv := fun u => gv t.1 (fork j t.2.1 u.1, u.2)) (o := fun u => o t.1 (fork j t.2.1 u.1, u.2))
      (fun u => hg _ _) (fun u => hgo _ _) M t.2.2.2 ht.2 with h | h
  · exact Or.inl ⟨ht.1, h⟩
  · exact Or.inr ⟨ht.1, h⟩

end Loop

/-! ## `P_O`'s bad samples -/

namespace Comp

variable {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
variable {K : Comp Coin} {I : Proto Coin W} {S : Server} {H : Msg → Msg} {Ω : Type} {P : Prover Ω} {x : Msg}

omit [Fintype Coin] [Nonempty Coin] in
theorem rerunCalls_some {ω : Ω} {d : (Fin (K.V.k x) → Coin) × K.EO.C x} {w : K.WO}
    (hw : K.wits H P x ω d = some w) : K.rerunCalls H P x ω d = K.calls H x d.1 w := by
  unfold Comp.rerunCalls
  rw [hw]
  rfl

omit [Fintype Coin] [Nonempty Coin] in
theorem isSome_of_givesRO {ω : Ω} {d : (Fin (K.V.k x) → Coin) × K.EO.C x} (h : K.GivesRO H P x ω d) :
    (K.wits H P x ω d).isSome = true := by
  obtain ⟨w, hw, -⟩ := h
  exact Option.isSome_iff_exists.2 ⟨w, hw⟩

omit [Fintype Coin] [Nonempty Coin] in
theorem outRO_of_isSome {ω : Ω} {d : (Fin (K.V.k x) → Coin) × K.EO.C x} (h : (K.wits H P x ω d).isSome = true)
    (hn : ¬ K.GivesRO H P x ω d) : K.OutRO H P x ω d := by
  obtain ⟨w, hw⟩ := Option.isSome_iff_exists.1 h
  exact ⟨w, hw, fun hR => hn ⟨w, hw, hR⟩⟩

omit [Fintype Coin] [Nonempty Coin] in
theorem view_raw {r : Fin (K.V.k x) → Coin} {w : K.WO} {i : ℕ} (hi : i < I.V.k (K.xI x)) :
    ((K.view (S.IVStar I.V I.ans) H x r w).msgs.map Prod.fst).getD i [] = (K.msgs x w).getD i [] :=
  Server.run_ivstar_raw S I.V I.ans H (K.xI x) (K.msgs x w) (fun i => coinSeq r i.val) hi

/-- **Two forks that `P_O` reads differently collide**: two samples whose witnesses are in `R_O`, the second forked from
the first at the inner round `j`, where `fAt` differs, show a collision in any true list holding both samples'
evaluations. -/
theorem pair_core (hB : S.Binds) (hCF : I.LastCoinFree H)
    (hK : K.Contract H (S.IVStar I.V I.ans) (fun v => I.opens (S.openView v)) x) {j : ℕ}
    (hj : j < I.V.k (K.xI x)) (ω : Ω) {d₁ d₂ : (Fin (K.V.k x) → Coin) × K.EO.C x} {w₁ w₂ : K.WO}
    (hf : fork j d₁.1 d₂.1 = d₂.1) (hw₁ : K.wits H P x ω d₁ = some w₁) (hw₂ : K.wits H P x ω d₂ = some w₂)
    (hR₁ : K.RO x w₁) (hR₂ : K.RO x w₂) (hG₁ : K.Gives H P x ω d₁.1 d₁.2 w₁)
    (hG₂ : K.Gives H P x ω d₂.1 d₂.2 w₂) (hne : K.fAt I S H P x ω j d₁ ≠ K.fAt I S H P x ω j d₂)
    {L : List (Msg × Msg)} (hT : ∀ p ∈ L, H p.1 = p.2)
    (hm₁ : ∀ p ∈ K.calls H x d₁.1 w₁ ++ K.own H x (run H P K.V x ω d₁.1).msgs, p ∈ L)
    (hm₂ : ∀ p ∈ K.calls H x d₂.1 w₂ ++ K.own H x (run H P K.V x ω d₂.1).msgs, p ∈ L) :
    (findColl L).isSome = true := by
  have hAcc : AccAt H K.V x := ⟨Ω, P, ω, d₁.1, hG₁.1⟩
  have hFB := hK.forks hAcc
  have hsub : ∀ {l : List (Msg × Msg)}, (∃ u v, findColl l = some (u, v) ∧ Collision H u v) →
      (∀ p ∈ l, p ∈ L) → (findColl L).isSome = true := by
    intro l h hl
    obtain ⟨u, v, huv, -⟩ := h
    exact findColl_isSome_of_sub H hT hl (Option.isSome_iff_exists.2 ⟨_, huv⟩)
  by_cases hj1 : j + 1 < I.V.k (K.xI x)
  · rcases hK.accepts hAcc Ω P ω d₁.1 d₁.2 w₁ hG₁ hR₁ with hv₁ | h
    swap
    · exact hsub h hm₁
    rcases hK.accepts hAcc Ω P ω d₂.1 d₂.2 w₂ hG₂ hR₂ with hv₂ | h
    swap
    · exact hsub h hm₂
    rcases hK.callsLog Ω P ω d₁.1 d₁.2 w₁ hG₁ hR₁ with hIL₁ | h
    swap
    · exact hsub h fun p hp => hm₁ p (List.mem_append_left _ hp)
    rcases hK.callsLog Ω P ω d₂.1 d₂.2 w₂ hG₂ hR₂ with hIL₂ | h
    swap
    · exact hsub h fun p hp => hm₂ p (List.mem_append_left _ hp)
    obtain ⟨-, os₁, last₁, -, hlen₁, hrd₁, -, hov₁⟩ :=
      S.accFacts I.V I.ans H (K.xI x) (replay (K.msgs x w₁)) () (fun i => coinSeq d₁.1 i.val) hv₁
    obtain ⟨-, os₂, last₂, -, hlen₂, hrd₂, -, hov₂⟩ :=
      S.accFacts I.V I.ans H (K.xI x) (replay (K.msgs x w₂)) () (fun i => coinSeq d₂.1 i.val) hv₂
    have hO₁ : ∀ i < I.V.k (K.xI x), K.oMsg I S H P x ω d₁ i = (S.opened os₁ last₁).getD i [] := fun i hi => by
      rw [oMsg_eq hw₁ hov₁ i, List.map_map]
      exact getD_range_map _ _ hi
    have hO₂ : ∀ i < I.V.k (K.xI x), K.oMsg I S H P x ω d₂ i = (S.opened os₂ last₂).getD i [] := fun i hi => by
      rw [oMsg_eq hw₂ hov₂ i, List.map_map]
      exact getD_range_map _ _ hi
    have hjo₁ : j < os₁.length := by omega
    have hjo₂ : j < os₂.length := by omega
    have hrows : (os₁.getD j []).map Prod.fst ≠ (os₂.getD j []).map Prod.fst := fun h => hne (by
      rw [fAt_lt hj1, fAt_lt hj1, hO₁ j hj, hO₂ j hj, S.opened_getD_lt hjo₁, S.opened_getD_lt hjo₂, h])
    by_cases hraw : ((K.view (S.IVStar I.V I.ans) H x d₁.1 w₁).msgs.map Prod.fst).getD j [] =
        ((K.view (S.IVStar I.V I.ans) H x d₂.1 w₂).msgs.map Prod.fst).getD j []
    · obtain ⟨hn₁, hop₁, hl₁⟩ := hrd₁ j hjo₁
      obtain ⟨hn₂, hop₂, hl₂⟩ := hrd₂ j hjo₂
      have hraw' : ((run H (replay (K.msgs x w₁)) (S.IVStar I.V I.ans) (K.xI x) ()
            (fun i => coinSeq d₁.1 i.val)).msgs.map Prod.fst).getD j [] =
          ((run H (replay (K.msgs x w₂)) (S.IVStar I.V I.ans) (K.xI x) ()
            (fun i => coinSeq d₂.1 i.val)).msgs.map Prod.fst).getD j [] := hraw
      rw [← hraw'] at hop₂ hl₂
      exact Claims.findColl_isSome H hT (binds_tied hB hop₁ hop₂ (hn₁.trans hn₂.symm) hrows
        (fun e he => hm₁ e (List.mem_append_left _ (hIL₁ e (hl₁ e he))))
        (fun e he => hm₂ e (List.mem_append_left _ (hIL₂ e (hl₂ e he)))))
    · have hG₂' : K.Gives H P x ω (fork j d₁.1 d₂.1) d₂.2 w₂ := by rw [hf]; exact hG₂
      obtain ⟨a, b, hab, -⟩ := hFB Ω P j ω d₁.1 d₂.1 d₁.2 d₂.2 w₁ w₂ hj hG₁ hG₂' hR₁ hR₂
        (by rw [hf]; exact hraw) (Or.inl hj1)
      rw [hf] at hab
      exact findColl_isSome_of_sub H hT (fun p hp => (List.mem_append.1 hp).elim (hm₁ p) (hm₂ p))
        (Option.isSome_iff_exists.2 ⟨_, hab⟩)
  · have hcs : ∀ i < j, coinSeq d₁.1 i = coinSeq d₂.1 i := fun i hi => by
      have e := congrArg (fun r => coinSeq r i) hf
      simp only [coinSeq_fork, hi, ↓reduceIte] at e
      exact e
    have hpre : (List.range (I.V.k (K.xI x) - 1)).map (coinSeq d₁.1) =
        (List.range (I.V.k (K.xI x) - 1)).map (coinSeq d₂.1) :=
      List.map_congr_left fun i hi => hcs i (by rw [List.mem_range] at hi; omega)
    have hcls : K.clsO I S H P x ω d₁ ≠ K.clsO I S H P x ω d₂ := fun hc => hne (by
      unfold Comp.fAt
      rw [ite_eq_right hj1, ite_eq_right hj1, hpre, hc])
    have hc₁ : K.clsO I S H P x ω d₁ = (I.opens (S.openView (K.view (S.IVStar I.V I.ans) H x d₁.1 w₁)),
        (K.innerLog (S.IVStar I.V I.ans) H x d₁.1 w₁).toFinset) := by
      unfold Comp.clsO Comp.viewCalls
      rw [hw₁]
      rfl
    have hc₂ : K.clsO I S H P x ω d₂ = (I.opens (S.openView (K.view (S.IVStar I.V I.ans) H x d₂.1 w₂)),
        (K.innerLog (S.IVStar I.V I.ans) H x d₂.1 w₂).toFinset) := by
      unfold Comp.clsO Comp.viewCalls
      rw [hw₂]
      rfl
    by_cases hall : ∀ i < I.V.k (K.xI x), ((K.view (S.IVStar I.V I.ans) H x d₁.1 w₁).msgs.map Prod.fst).getD i [] =
        ((K.view (S.IVStar I.V I.ans) H x d₂.1 w₂).msgs.map Prod.fst).getD i []
    · exfalso
      have hrun := Server.run_ivstar_lastCoin hCF S (K.xI x) (K.msgs x w₁) (K.msgs x w₂)
        (fun i => coinSeq d₁.1 i.val) (fun i => coinSeq d₂.1 i.val)
        (fun i hi => (view_raw hi).symm.trans ((hall i hi).trans (view_raw hi)))
        (fun i hi => (coinSeq_firstCoins d₁.1 (m := I.V.k (K.xI x)) (by omega)).trans
          ((hcs i (by omega)).trans (coinSeq_firstCoins d₂.1 (m := I.V.k (K.xI x)) (by omega)).symm))
      apply hcls
      rw [hc₁, hc₂]
      exact congrArg (fun t : Transcript => (I.opens (S.openView t.seen), t.vLog.toFinset)) hrun
    · push Not at hall
      obtain ⟨i, hi, hri⟩ := hall
      have hfi : fork i d₁.1 d₂.1 = d₂.1 := fork_of_le (by omega) hf
      have hG₂' : K.Gives H P x ω (fork i d₁.1 d₂.1) d₂.2 w₂ := by rw [hfi]; exact hG₂
      have hor : i + 1 < (S.IVStar I.V I.ans).k (K.xI x) ∨
          I.opens (S.openView (K.view (S.IVStar I.V I.ans) H x d₁.1 w₁)) ≠
            I.opens (S.openView (K.view (S.IVStar I.V I.ans) H x (fork i d₁.1 d₂.1) w₂)) ∨
          (K.innerLog (S.IVStar I.V I.ans) H x d₁.1 w₁).toFinset ≠
            (K.innerLog (S.IVStar I.V I.ans) H x (fork i d₁.1 d₂.1) w₂).toFinset := by
        rw [hfi]
        by_cases hi1 : i + 1 < I.V.k (K.xI x)
        · exact Or.inl hi1
        · exact Or.inr (not_and_or.1 fun hc => hcls (by rw [hc₁, hc₂, hc.1, hc.2]))
      obtain ⟨a, b, hab, -⟩ := hFB Ω P i ω d₁.1 d₂.1 d₁.2 d₂.2 w₁ w₂ hi hG₁ hG₂' hR₁ hR₂
        (by rw [hfi]; exact hri) hor
      rw [hfi] at hab
      exact findColl_isSome_of_sub H hT (fun p hp => (List.mem_append.1 hp).elim (hm₁ p) (hm₂ p))
        (Option.isSome_iff_exists.2 ⟨_, hab⟩)

/-- **Two forks that `P_O` reads differently collide**, among the two reruns' evaluations as the link searches them. -/
theorem pair_coll (hB : S.Binds) (hCF : I.LastCoinFree H)
    (hK : K.Contract H (S.IVStar I.V I.ans) (fun v => I.opens (S.openView v)) x) {j : ℕ}
    (hj : j < I.V.k (K.xI x)) (ω : Ω) {d₁ d₂ : (Fin (K.V.k x) → Coin) × K.EO.C x} (hf : fork j d₁.1 d₂.1 = d₂.1)
    (hg₁ : K.GivesRO H P x ω d₁) (hg₂ : K.GivesRO H P x ω d₂)
    (hne : K.fAt I S H P x ω j d₁ ≠ K.fAt I S H P x ω j d₂) :
    ∃ a b, findColl (K.rerunCalls H P x ω d₁ ++ K.ownAt H P x ω d₁ ++
      (K.rerunCalls H P x ω d₂ ++ K.ownAt H P x ω d₂)) = some (a, b) ∧ Collision H a b := by
  obtain ⟨w₁, hw₁, hR₁⟩ := hg₁
  obtain ⟨w₂, hw₂, hR₂⟩ := hg₂
  have hG₁ : K.Gives H P x ω d₁.1 d₁.2 w₁ := ⟨hK.replays Ω P ω d₁.1 d₁.2 w₁ hw₁, hw₁⟩
  have hG₂ : K.Gives H P x ω d₂.1 d₂.2 w₂ := ⟨hK.replays Ω P ω d₂.1 d₂.2 w₂ hw₂, hw₂⟩
  rw [rerunCalls_some hw₁, rerunCalls_some hw₂]
  unfold Comp.ownAt
  have hT : ∀ p ∈ K.calls H x d₁.1 w₁ ++ K.own H x (run H P K.V x ω d₁.1).msgs ++
      (K.calls H x d₂.1 w₂ ++ K.own H x (run H P K.V x ω d₂.1).msgs), H p.1 = p.2 := by
    intro p hp
    simp only [List.mem_append] at hp
    rcases hp with (hp | hp) | (hp | hp)
    · exact hK.callsTrue Ω P ω d₁.1 d₁.2 w₁ hG₁ p hp
    · exact hK.ownTrue Ω P ω d₁.1 p hp
    · exact hK.callsTrue Ω P ω d₂.1 d₂.2 w₂ hG₂ p hp
    · exact hK.ownTrue Ω P ω d₂.1 p hp
  exact coll_of_isSome hT (pair_core hB hCF hK hj ω hf hw₁ hw₂ hR₁ hR₂ hG₁ hG₂ hne hT
    (fun p hp => List.mem_append_left _ hp) (fun p hp => List.mem_append_right _ hp))

/-- **The gated link at `j` finds the pair**: where its first fork at `j` gives witnesses in `R_O`, and its loop's first
giving fork is in `R_O` and read differently, the link's search shows a collision. -/
theorem link_out (hB : S.Binds) (hCF : I.LastCoinFree H)
    (hK : K.Contract H (S.IVStar I.V I.ans) (fun v => I.opens (S.openView v)) x) {M : ℕ}
    (j : Fin (I.V.k (K.xI x))) (ω : Ω) (us : ℕ → (Fin (K.V.k x) → Coin) × K.EO.C x)
    (hg : K.GivesRO H P x ω (fork j.val (us 0).1 (us 1).1, (us 1).2)) {v : (Fin (K.V.k x) → Coin) × K.EO.C x}
    (hv : firstPick (fun u : (Fin (K.V.k x) → Coin) × K.EO.C x =>
        (K.wits H P x ω (fork j.val (us 0).1 u.1, u.2)).isSome = true) M (fun t : Fin M => us (2 + t.val)) = some v)
    (hgv : K.GivesRO H P x ω (fork j.val (us 0).1 v.1, v.2))
    (hne : K.fAt I S H P x ω j.val (fork j.val (us 0).1 v.1, v.2) ≠
      K.fAt I S H P x ω j.val (fork j.val (us 0).1 (us 1).1, (us 1).2)) :
    ∃ a b, findColl ((((linkPlan (Coin := Coin) I.V.k M).plan (K.xI x) j).exec
      (K.verModel (S.IVStar I.V I.ans) H P x) ω us).2.2.flatMap fun e =>
        (K.verModel (S.IVStar I.V I.ans) H P x).log ω ((K.verModel (S.IVStar I.V I.ans) H P x).named us e)) =
      some (a, b) ∧ Collision H a b := by
  have hcut : (K.verModel (S.IVStar I.V I.ans) H P x).cut j.val = j.val := Nat.min_eq_left j.isLt.le
  have hv' : firstPick (fun u : (Fin (K.V.k x) → Coin) × K.EO.C x => (K.verModel (S.IVStar I.V I.ans) H P x).acc ω
      (fork ((K.verModel (S.IVStar I.V I.ans) H P x).cut j.val) (us 0).1 u.1, u.2)) M
      (fun t : Fin M => us (2 + t.val)) = some v := by
    rw [hcut]
    exact hv
  obtain ⟨e, he, hne'⟩ := linkLoop_exec (K.verModel (S.IVStar I.V I.ans) H P x) ω us j.val
    (.fork j.val (.fresh 0) (.fresh 1)) M 2 v hv'
  have hacc : (K.verModel (S.IVStar I.V I.ans) H P x).acc ω
      ((K.verModel (S.IVStar I.V I.ans) H P x).named us (.fork j.val (.fresh 0) (.fresh 1))) := by
    show (K.wits H P x ω (fork (min j.val (I.V.k (K.xI x))) (us 0).1 (us 1).1, (us 1).2)).isSome = true
    rw [Nat.min_eq_left j.isLt.le]
    exact isSome_of_givesRO hg
  have hexec : (((linkPlan (Coin := Coin) I.V.k M).plan (K.xI x) j).exec
      (K.verModel (S.IVStar I.V I.ans) H P x) ω us).2.2 =
      ((linkLoop j.val (.fork j.val (.fresh 0) (.fresh 1)) M 2).exec
        (K.verModel (S.IVStar I.V I.ans) H P x) ω us).2.2 := by
    show ((Plan.ask (.fork j.val (.fresh 0) (.fresh 1)) fun b =>
      if b then linkLoop j.val (.fork j.val (.fresh 0) (.fresh 1)) M 2 else .done []).exec
        (K.verModel (S.IVStar I.V I.ans) H P x) ω us).2.2 = _
    simp only [Plan.exec, decide_eq_true_eq, hacc, ↓reduceIte]
  rw [hexec, he, List.flatMap_cons, List.flatMap_cons, List.flatMap_nil, List.append_nil]
  have hn1 : (K.verModel (S.IVStar I.V I.ans) H P x).named us (.fork j.val (.fresh 0) (.fresh 1)) =
      (fork j.val (us 0).1 (us 1).1, (us 1).2) := by
    show (fork ((K.verModel (S.IVStar I.V I.ans) H P x).cut j.val) (us 0).1 (us 1).1, (us 1).2) = _
    rw [hcut]
  rw [hn1, hne', hcut]
  have hf : fork j.val (fork j.val (us 0).1 (us 1).1) (fork j.val (us 0).1 v.1) = fork j.val (us 0).1 v.1 := by
    rw [fork_fork, fork_fork_le le_rfl]
  exact pair_coll (d₁ := (fork j.val (us 0).1 (us 1).1, (us 1).2)) (d₂ := (fork j.val (us 0).1 v.1, v.2))
    hB hCF hK j.isLt ω hf hg hgv (Ne.symm hne)

/-- **The link's pick at `j` is the gated link's success at `j`**, over its uniform slots. -/
theorem Pr_linkPick_le (hB : S.Binds) (hCF : I.LastCoinFree H)
    (hK : K.Contract H (S.IVStar I.V I.ans) (fun v => I.opens (S.openView v)) x) [Fintype Ω] [Nonempty Ω] (M : ℕ)
    (hk : 0 < (linkPlan (Coin := Coin) I.V.k M).sel (K.xI x)) (j : Fin (I.V.k (K.xI x))) :
    Pr (fun t : Ω × (Fin (K.V.k x) → Coin) × ((Fin (K.V.k x) → Coin) × K.EO.C x) ×
        (Fin M → (Fin (K.V.k x) → Coin) × K.EO.C x) =>
      K.GivesRO H P x t.1 (fork j t.2.1 t.2.2.1.1, t.2.2.1.2) ∧
        ∃ v, firstPick (fun u : (Fin (K.V.k x) → Coin) × K.EO.C x =>
            (K.wits H P x t.1 (fork j t.2.1 u.1, u.2)).isSome = true) M t.2.2.2 = some v ∧
          K.GivesRO H P x t.1 (fork j t.2.1 v.1, v.2) ∧
          K.fAt I S H P x t.1 j (fork j t.2.1 v.1, v.2) ≠ K.fAt I S H P x t.1 j (fork j t.2.1 t.2.2.1.1, t.2.2.1.2)) ≤
      Pr fun s : Ω × (Fin (M + 1 + 1) → (Fin (K.V.k x) → Coin) × K.EO.C x) =>
        ∃ a b, ((linkPlan (Coin := Coin) I.V.k M).finder (K.verModel (S.IVStar I.V I.ans) H P x) (K.xI x) hk).out
          (j, s) = some (a, b) ∧ Collision H a b := by
  have hU : UnifMap (fun s : Ω × (Fin (M + 1 + 1) → (Fin (K.V.k x) → Coin) × K.EO.C x) =>
      (s.1, ((s.2 0).1, Fin.tail s.2 0, Fin.tail (Fin.tail s.2)))) :=
    UnifMap.prod UnifMap.self (unifMap_link M)
  refine (hU.Pr _).symm.trans_le (Pr_mono fun s hs => ?_)
  obtain ⟨hg, v, hv, hgv, hne⟩ := hs
  have h0 : slotRuns s.2 0 = s.2 0 := congrArg s.2 (Fin.ext (by simp))
  have h1 : slotRuns s.2 1 = Fin.tail s.2 0 := congrArg s.2 (Fin.ext (by simp))
  have h2 : (fun t : Fin M => slotRuns s.2 (2 + t.val)) = Fin.tail (Fin.tail s.2) :=
    funext fun t => congrArg s.2 (Fin.ext (by have := t.isLt; simp; omega))
  exact link_out hB hCF hK j s.1 (slotRuns s.2) (by rw [h0, h1]; exact hg) (by rw [h2, h0]; exact hv)
    (by rw [h0]; exact hgv) (by rw [h0, h1]; exact hne)

/-- **`P_O`'s bad samples, linearly**: a sample where `E_O` gives witnesses in `R_O` and `P_O`'s runs don't show it has
probability at most `k_Π·(Pr[link] + 1/(M+1) + PrOut)`, the link the gated link on the composite's reruns. -/
theorem Pr_badO_le_link (hB : S.Binds) (hCF : I.LastCoinFree H)
    (hK : K.Contract H (S.IVStar I.V I.ans) (fun v => I.opens (S.openView v)) x) [Fintype Ω] [Nonempty Ω] (M : ℕ) :
    Pr (fun t : Ω × (Fin (K.V.k x) → Coin) × K.EO.C x => K.BadO I S H P x t.1 t.2) ≤
      I.V.k (K.xI x) * ((linkPlan (Coin := Coin) I.V.k M).PrAt H (K.verModel (S.IVStar I.V I.ans) H P x) (K.xI x) +
        1 / ((M : ℝ) + 1) + K.PrOut H P x) := by
  by_cases hk : 0 < I.V.k (K.xI x)
  swap
  · have h0 : Pr (fun t : Ω × (Fin (K.V.k x) → Coin) × K.EO.C x => K.BadO I S H P x t.1 t.2) = 0 := by
      refine le_antisymm ?_ (Pr_nonneg _)
      refine (Pr_mono fun t ht => ?_).trans_eq Pr_false
      exact ht.2 fun j hj => absurd hj (by omega)
    have hk0 : (I.V.k (K.xI x) : ℝ) = 0 := by exact_mod_cast (show I.V.k (K.xI x) = 0 by omega)
    rw [h0, hk0, zero_mul]
  have hk' : 0 < (linkPlan (Coin := Coin) I.V.k M).sel (K.xI x) := hk
  have h1 : Pr (fun t : Ω × (Fin (K.V.k x) → Coin) × K.EO.C x => K.BadO I S H P x t.1 t.2) ≤
      ∑ j : Fin (I.V.k (K.xI x)), Pr (fun t : Ω × (Fin (K.V.k x) → Coin) × K.EO.C x =>
        K.GivesRO H P x t.1 t.2 ∧
          K.fAt I S H P x t.1 j t.2 ≠ K.opAtO I S H P x t.1 j ((List.range j).map (coinSeq t.2.1))) := by
    refine (Pr_mono fun t ht => ?_).trans (Claims.Pr_exists_le _)
    obtain ⟨hs, hna⟩ := ht
    simp only [Comp.AgreesO, not_forall] at hna
    obtain ⟨j, hj, hne⟩ := hna
    exact ⟨⟨j, hj⟩, hs, hne⟩
  have h2 : ∀ j : Fin (I.V.k (K.xI x)), Pr (fun t : Ω × (Fin (K.V.k x) → Coin) × K.EO.C x =>
      K.GivesRO H P x t.1 t.2 ∧
        K.fAt I S H P x t.1 j t.2 ≠ K.opAtO I S H P x t.1 j ((List.range j).map (coinSeq t.2.1))) ≤
      (Pr fun s : Ω × (Fin (M + 1 + 1) → (Fin (K.V.k x) → Coin) × K.EO.C x) =>
        ∃ a b, ((linkPlan (Coin := Coin) I.V.k M).finder (K.verModel (S.IVStar I.V I.ans) H P x) (K.xI x) hk').out
          (j, s) = some (a, b) ∧ Collision H a b) + 1 / ((M : ℝ) + 1) + K.PrOut H P x := by
    intro j
    have hp := Pr_plur_fork_lin (N := K.V.k x) j.val (fun ω d => K.GivesRO H P x ω d)
      (fun ω d => K.fAt I S H P x ω j.val d) M
    have hs := Pr_link_split j.val M (fun ω d => K.GivesRO H P x ω d)
      (fun ω d => (K.wits H P x ω d).isSome = true) (fun ω d => K.OutRO H P x ω d)
      (fun ω d => K.fAt I S H P x ω j.val d) (fun _ _ h => isSome_of_givesRO h) (fun _ _ h hn => outRO_of_isSome h hn)
    have ho' := (Pr_outLoop_le j.val M (fun ω d => K.GivesRO H P x ω d)
      (fun ω d => (K.wits H P x ω d).isSome = true) (fun ω d => K.OutRO H P x ω d)
      (fun _ _ h => isSome_of_givesRO h)).trans_eq
        (show Pr (fun t : Ω × (Fin (K.V.k x) → Coin) × K.EO.C x => K.OutRO H P x t.1 t.2) = K.PrOut H P x from rfl)
    have hl := Pr_linkPick_le (Ω := Ω) (P := P) hB hCF hK M hk' j
    refine (le_of_eq (Pr_congr fun t => ?_)).trans (hp.trans ?_)
    · simp only [Comp.opAtO, fork_preCoins]
    · exact (add_le_add (hs.trans (add_le_add hl ho')) le_rfl).trans_eq (by ring)
  have hP : (linkPlan (Coin := Coin) I.V.k M).PrAt H (K.verModel (S.IVStar I.V I.ans) H P x) (K.xI x) =
      Ex fun j : Fin (I.V.k (K.xI x)) => Pr fun s : Ω × (Fin (M + 1 + 1) → (Fin (K.V.k x) → Coin) × K.EO.C x) =>
        ∃ a b, ((linkPlan (Coin := Coin) I.V.k M).finder (K.verModel (S.IVStar I.V I.ans) H P x) (K.xI x) hk').out
          (j, s) = some (a, b) ∧ Collision H a b := by
    rw [RunFinder.PrAt, dite_eq_left hk']
    exact Claims.Recursion.Pr_prod _
  have hk0 : (I.V.k (K.xI x) : ℝ) ≠ 0 := by exact_mod_cast hk.ne'
  have hF : (I.V.k (K.xI x) : ℝ) *
      (linkPlan (Coin := Coin) I.V.k M).PrAt H (K.verModel (S.IVStar I.V I.ans) H P x) (K.xI x) =
      ∑ j : Fin (I.V.k (K.xI x)), Pr fun s : Ω × (Fin (M + 1 + 1) → (Fin (K.V.k x) → Coin) × K.EO.C x) =>
        ∃ a b, ((linkPlan (Coin := Coin) I.V.k M).finder (K.verModel (S.IVStar I.V I.ans) H P x) (K.xI x) hk').out
          (j, s) = some (a, b) ∧ Collision H a b := by
    rw [hP]
    unfold Ex
    rw [Fintype.card_fin, mul_div_assoc', mul_div_cancel_left₀ _ hk0]
  calc Pr (fun t : Ω × (Fin (K.V.k x) → Coin) × K.EO.C x => K.BadO I S H P x t.1 t.2)
      ≤ ∑ j : Fin (I.V.k (K.xI x)), Pr (fun t : Ω × (Fin (K.V.k x) → Coin) × K.EO.C x =>
        K.GivesRO H P x t.1 t.2 ∧
          K.fAt I S H P x t.1 j t.2 ≠ K.opAtO I S H P x t.1 j ((List.range j).map (coinSeq t.2.1))) := h1
    _ ≤ ∑ j : Fin (I.V.k (K.xI x)), ((Pr fun s : Ω × (Fin (M + 1 + 1) → (Fin (K.V.k x) → Coin) × K.EO.C x) =>
        ∃ a b, ((linkPlan (Coin := Coin) I.V.k M).finder (K.verModel (S.IVStar I.V I.ans) H P x) (K.xI x) hk').out
          (j, s) = some (a, b) ∧ Collision H a b) + 1 / ((M : ℝ) + 1) + K.PrOut H P x) :=
        Finset.sum_le_sum fun j _ => h2 j
    _ = I.V.k (K.xI x) * ((linkPlan (Coin := Coin) I.V.k M).PrAt H (K.verModel (S.IVStar I.V I.ans) H P x)
        (K.xI x) + 1 / ((M : ℝ) + 1) + K.PrOut H P x) := by
        rw [Finset.sum_add_distrib, Finset.sum_add_distrib, ← hF, Finset.sum_const, Finset.sum_const,
          Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, nsmul_eq_mul]
        ring

/-! ## A rejecting judged rerun -/

omit [Fintype Coin] [Nonempty Coin] in
/-- Every evaluation `B_I` and the link search at a rerun is true. -/
theorem verModel_log_true (hK : K.Contract H (S.IVStar I.V I.ans) (fun v => I.opens (S.openView v)) x) (ω : Ω)
    (u : (Fin (K.V.k x) → Coin) × K.EO.C x) :
    ∀ p ∈ (K.verModel (S.IVStar I.V I.ans) H P x).log ω u, H p.1 = p.2 := by
  intro p hp
  rcases List.mem_append.1 hp with hp | hp
  · exact rerunCalls_true hK.replays hK.callsTrue ω u p hp
  · exact hK.ownTrue Ω P ω u.1 p hp

/-- **A rejecting judged rerun is `B_I`'s success**: where `E_O` gives witnesses in `R_O` on an accepted run and the
inner view rejects, the rerun's evaluations collide (`accepts`), and `forkPlan` searches them at its first run, at
every round it guesses. -/
theorem Pr_innerRejects_le_fork
    (hK : K.Contract H (S.IVStar I.V I.ans) (fun v => I.opens (S.openView v)) x) [Fintype Ω] [Nonempty Ω] :
    Pr (K.InnerRejectsO I S H P x) ≤
      (forkPlan (Coin := Coin) I.V.k).PrAt H (K.verModel (S.IVStar I.V I.ans) H P x) (K.xI x) := by
  by_cases hk : 0 < I.V.k (K.xI x)
  swap
  · have h0 : Pr (K.InnerRejectsO I S H P x) = 0 := by
      refine le_antisymm ((Pr_mono fun t ht => ?_).trans_eq Pr_false) (Pr_nonneg _)
      obtain ⟨w, hG, -, -⟩ := ht
      exact hk (hK.margin Ω P t.1 t.2.1 hG.1).1
    have hk' : ¬ 0 < (forkPlan (Coin := Coin) I.V.k).sel (K.xI x) := hk
    have hP0 : (forkPlan (Coin := Coin) I.V.k).PrAt H (K.verModel (S.IVStar I.V I.ans) H P x) (K.xI x) = 0 := by
      rw [RunFinder.PrAt, dite_eq_right hk']
    exact h0.trans_le hP0.symm.le
  have hk' : 0 < (forkPlan (Coin := Coin) I.V.k).sel (K.xI x) := hk
  have hP : (forkPlan (Coin := Coin) I.V.k).PrAt H (K.verModel (S.IVStar I.V I.ans) H P x) (K.xI x) =
      Ex fun j : Fin (I.V.k (K.xI x)) => Pr fun s : Ω × (Fin (1 + 1) → (Fin (K.V.k x) → Coin) × K.EO.C x) =>
        ∃ a b, ((forkPlan (Coin := Coin) I.V.k).finder (K.verModel (S.IVStar I.V I.ans) H P x) (K.xI x) hk').out
          (j, s) = some (a, b) ∧ Collision H a b := by
    rw [RunFinder.PrAt, dite_eq_left hk']
    exact Claims.Recursion.Pr_prod _
  have : Nonempty (Fin (I.V.k (K.xI x))) := ⟨⟨0, hk⟩⟩
  rw [hP]
  refine (Ex_const (α := Fin (I.V.k (K.xI x))) _).symm.trans_le (Ex_mono fun j => ?_)
  have hU : UnifMap (fun s : Ω × (Fin (1 + 1) → (Fin (K.V.k x) → Coin) × K.EO.C x) => (s.1, s.2 0)) :=
    UnifMap.prod UnifMap.self
      ((UnifMap.equiv (splitFirst (U := (Fin (K.V.k x) → Coin) × K.EO.C x) 1)).comp UnifMap.fst)
  refine (hU.Pr _).symm.trans_le (Pr_mono fun s hs => ?_)
  obtain ⟨w, hG, hR, hrej⟩ := hs
  have hAcc : AccAt H K.V x := ⟨Ω, P, s.1, (s.2 0).1, hG.1⟩
  obtain ⟨a, b, hab, -⟩ := (hK.accepts hAcc Ω P s.1 (s.2 0).1 (s.2 0).2 w hG hR).resolve_left hrej
  have hw : K.wits H P x s.1 (s.2 0) = some w := hG.2
  have hu0 : slotRuns s.2 0 = s.2 0 := congrArg s.2 (Fin.ext (by simp))
  have hT : ∀ p ∈ ((((forkPlan (Coin := Coin) I.V.k).plan (K.xI x) j).exec (K.verModel (S.IVStar I.V I.ans) H P x)
      s.1 (slotRuns s.2)).2.2.flatMap fun e =>
        (K.verModel (S.IVStar I.V I.ans) H P x).log s.1 ((K.verModel (S.IVStar I.V I.ans) H P x).named (slotRuns s.2) e)),
      H p.1 = p.2 := by
    intro p hp
    obtain ⟨e, -, hpe⟩ := List.mem_flatMap.1 hp
    exact verModel_log_true hK s.1 _ p hpe
  have hsub : ∀ p ∈ K.calls H x (s.2 0).1 w ++ K.own H x (run H P K.V x s.1 (s.2 0).1).msgs,
      p ∈ ((((forkPlan (Coin := Coin) I.V.k).plan (K.xI x) j).exec (K.verModel (S.IVStar I.V I.ans) H P x)
        s.1 (slotRuns s.2)).2.2.flatMap fun e =>
          (K.verModel (S.IVStar I.V I.ans) H P x).log s.1
            ((K.verModel (S.IVStar I.V I.ans) H P x).named (slotRuns s.2) e)) := by
    intro p hp
    refine List.mem_flatMap.2 ⟨.fresh 0, List.mem_cons.2 (Or.inl rfl), ?_⟩
    show p ∈ K.rerunCalls H P x s.1 (slotRuns s.2 0) ++ K.ownAt H P x s.1 (slotRuns s.2 0)
    rw [hu0, rerunCalls_some hw]
    exact hp
  exact coll_of_isSome hT (findColl_isSome_of_sub H hT hsub (Option.isSome_iff_exists.2 ⟨_, hab⟩))

end Comp

end Claims.RecGeneric
