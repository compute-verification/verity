import Proofs.Flock.Claims.Recursion.Inner
import Proofs.Flock.Claims.Recursion.Abort

/-!
# The inner stage's KS from the inner protocol's KS against provers that may abort

`InnerKS` from `Inner.KSAbort` (`innerKS_of_KSAbort`), and with it the recursion's KS from Flock's KS at the outer
deployment, `Inner.KSAbort` and code facts (`KS_of_flockKS`).

The inner prover `P_I` has the compiled prover's tape. At inner round `j`, on the inner coins so far (which the inner
verifier's answers show, `PublicCoin`), it sends the plurality of the inner view's message `j` over `E_O`'s extractions
at the draws with that coin prefix (`opAt`). Each inner run aborts unless `E_O` succeeds at the draw and every inner
message of its view is `P_I`'s (`keepD`); the abort coins are the rest of the draw, which the inner verifier doesn't
read. Where every rerun of the inner `E^rep` agrees, `E*` sees what `E^rep` sees against `P_I`'s aborting runs
(`RP.eval_congr`, `servedRun_seen_eq`), so `E*`'s inner stage fails at most as `Inner.KSAbort` bounds, plus where some
rerun disagrees. That is at most `N_I` times one draw's disagreement (`Pr_exists_rerun_le`), at most
`Σ_j √(Pr[two extractions open inner round j differently])` (`Pr_plur_fork`), so `√(k_I·k(x)·Pr[Be])`
(`sum_opensDiffer_le`). `B` at `P_I`'s aborting runs finds collisions `B_I` finds (`PrBAbort_le_PrBI`).
-/

namespace Claims.Recursion

open Claims

/-! ## Reruns that look alike, runs that agree -/

/-- **Runs that look alike give one result**: a program against two runs that show the same at every draw it reruns at
returns the same. -/
theorem RP.eval_congr {R α : Type} (H : Msg → Msg) (run₁ run₂ : R → Trace) (p : RP R α)
    (h : ∀ r ∈ p.rerunsAt H run₁, (run₁ r).seen = (run₂ r).seen) : (p.eval H run₁).1 = (p.eval H run₂).1 := by
  induction p with
  | ret a => rfl
  | hash q k ih => exact ih (H q) h
  | rerun r k ih =>
    simp only [RP.eval]
    rw [← h r List.mem_cons_self]
    exact ih _ fun r' hr' => h r' (List.mem_cons_of_mem _ hr')

/-- **Two provers that agree on every history the run reaches give one run**: while an invariant `G` of the rounds
left, the history and the messages holds, the provers' next messages agree, and each verifier message keeps it. -/
theorem stepWith_congr {Ω₁ Ω₂ : Type} (H : Msg → Msg) (P₁ : Prover Ω₁) (P₂ : Prover Ω₂) (x : Msg) (ω₁ : Ω₁)
    (ω₂ : Ω₂) (vn : List Msg → OC Out) (G : ℕ → List (Msg × Msg) → List Msg → Prop)
    (hP : ∀ M hist as, G (M + 1) hist as → (P₁.next x hist ω₁).eval H = (P₂.next x hist ω₂).eval H)
    (hG : ∀ M hist as m, G (M + 1) hist as →
      ((vn (as ++ [((P₁.next x hist ω₁).eval H).1])).eval H).1 = .msg m →
        G M (hist ++ [(((P₁.next x hist ω₁).eval H).1, m)]) (as ++ [((P₁.next x hist ω₁).eval H).1])) :
    ∀ M hist as t, G M hist as → stepWith H P₁ x ω₁ vn M hist as t = stepWith H P₂ x ω₂ vn M hist as t
  | 0, _, _, _, _ => rfl
  | M + 1, hist, as, t, h => by
    simp only [stepWith]
    rw [← hP M hist as h]
    rcases hv : ((vn (as ++ [((P₁.next x hist ω₁).eval H).1])).eval H).1 with m | _ | _
    · exact stepWith_congr H P₁ P₂ x ω₁ ω₂ vn G hP hG M _ _ _ (hG M hist as m h hv)
    · rfl
    · rfl

/-! ## The plurality -/

open Classical in
/-- **The plurality value**: the value `f` takes most often where `g` holds, `[]` where `g` never does. -/
noncomputable def plur {U : Type} [Fintype U] (g : U → Prop) (f : U → Msg) : Msg :=
  if h : (Finset.univ.filter g).Nonempty then
    f (Classical.choose (Finset.exists_max_image (Finset.univ.filter g)
      (fun u => (Finset.univ.filter fun u' => g u' ∧ f u' = f u).card) h))
  else []

theorem Pr_eq_card {α : Type} [Fintype α] (p : α → Prop) [DecidablePred p] :
    Pr p = ((Finset.univ.filter p).card : ℝ) / Fintype.card α := by
  unfold Pr
  congr

/-- **The plurality is a value `f` takes where `g` holds, and no value is taken more often**, when `g` ever holds. -/
theorem plur_spec {U : Type} [Fintype U] (g : U → Prop) (f : U → Msg) (hne : ∃ u, g u) :
    ∃ u₀, g u₀ ∧ plur g f = f u₀ ∧
      ∀ u, g u → Pr (fun u' => g u' ∧ f u' = f u) ≤ Pr (fun u' => g u' ∧ f u' = f u₀) := by
  classical
  have hne' : (Finset.univ.filter g).Nonempty := by
    obtain ⟨u, hu⟩ := hne
    exact ⟨u, Finset.mem_filter.2 ⟨Finset.mem_univ _, hu⟩⟩
  obtain ⟨hu₀, hmax⟩ := Classical.choose_spec (Finset.exists_max_image (Finset.univ.filter g)
    (fun u => (Finset.univ.filter fun u' => g u' ∧ f u' = f u).card) hne')
  refine ⟨_, (Finset.mem_filter.1 hu₀).2, ?_, fun u hu => ?_⟩
  · unfold plur
    simp only [hne', ↓reduceDIte]
  · have hc := hmax u (Finset.mem_filter.2 ⟨Finset.mem_univ _, hu⟩)
    rw [Pr_eq_card, Pr_eq_card]
    exact div_le_div_of_nonneg_right (Nat.cast_le.2 (by convert hc)) (Nat.cast_nonneg _)

theorem Pr_split {U : Type} [Fintype U] (g : U → Prop) (f : U → Msg) (w : Msg) :
    Pr g = Pr (fun u => g u ∧ f u = w) + Pr (fun u => g u ∧ f u ≠ w) := by
  classical
  rw [← Ex_ind, ← Ex_ind, ← Ex_ind, ← Ex_add]
  refine congrArg Ex (funext fun u => ?_)
  by_cases hg : g u <;> by_cases hf : f u = w <;> simp [hg, hf]

/-- **Off the plurality, squared, is at most two disagreeing draws**: `Pr[g ∧ f ≠ plur]² ≤ Pr[g ∧ g' ∧ f ≠ f']` over
two independent draws, since each draw where `g` holds disagrees with a fresh one at least as often as the plurality
is missed. -/
theorem Pr_plur_sq_le {U : Type} [Fintype U] (g : U → Prop) (f : U → Msg) :
    Pr (fun u => g u ∧ f u ≠ plur g f) ^ 2 ≤ Pr (fun p : U × U => g p.1 ∧ g p.2 ∧ f p.1 ≠ f p.2) := by
  classical
  by_cases hne : ∃ u, g u
  · obtain ⟨u₀, -, hp, hmax⟩ := plur_spec g f hne
    rw [hp]
    have ha0 : 0 ≤ Pr (fun u => g u ∧ f u ≠ f u₀) := Pr_nonneg _
    have hpt : ∀ u, (if g u then Pr (fun u => g u ∧ f u ≠ f u₀) else 0) ≤
        Pr (fun u' => g u ∧ g u' ∧ f u ≠ f u') := by
      intro u
      by_cases hu : g u
      · simp only [hu, ↓reduceIte, true_and]
        rw [Pr_congr fun u' => show (g u' ∧ f u ≠ f u') ↔ (g u' ∧ f u' ≠ f u) from
          ⟨fun h => ⟨h.1, Ne.symm h.2⟩, fun h => ⟨h.1, Ne.symm h.2⟩⟩]
        have h1 := Pr_split g f (f u)
        have h2 := Pr_split g f (f u₀)
        linarith [hmax u hu]
      · simp only [hu, ↓reduceIte]
        exact Pr_nonneg _
    have hEx : Ex (fun u => if g u then Pr (fun u => g u ∧ f u ≠ f u₀) else 0) =
        Pr (fun u => g u ∧ f u ≠ f u₀) * Pr g := by
      rw [← Ex_ind g, ← Ex_mul_left]
      congr 1
      funext u
      by_cases hu : g u <;> simp [hu]
    have hag : Pr (fun u => g u ∧ f u ≠ f u₀) ≤ Pr g := Pr_mono fun u h => h.1
    calc Pr (fun u => g u ∧ f u ≠ f u₀) ^ 2 = Pr (fun u => g u ∧ f u ≠ f u₀) * Pr (fun u => g u ∧ f u ≠ f u₀) :=
          sq _
      _ ≤ Pr (fun u => g u ∧ f u ≠ f u₀) * Pr g := mul_le_mul_of_nonneg_left hag ha0
      _ = Ex (fun u => if g u then Pr (fun u => g u ∧ f u ≠ f u₀) else 0) := hEx.symm
      _ ≤ Ex fun u => Pr (fun u' => g u ∧ g u' ∧ f u ≠ f u') := Ex_mono hpt
      _ = Pr (fun p : U × U => g p.1 ∧ g p.2 ∧ f p.1 ≠ f p.2) :=
          (Pr_prod (fun p : U × U => g p.1 ∧ g p.2 ∧ f p.1 ≠ f p.2)).symm
  · have h0 : Pr (fun u => g u ∧ f u ≠ plur g f) = 0 := by
      refine le_antisymm ((Pr_mono fun u h => ?_).trans_eq Pr_false) (Pr_nonneg _)
      exact hne ⟨u, h.1⟩
    rw [h0]
    simpa using Pr_nonneg (fun p : U × U => g p.1 ∧ g p.2 ∧ f p.1 ≠ f p.2)

/-- **Off the plurality at each coin prefix, at most the root of two disagreeing forks**: over a tape and a draw, the
draw's `f` misses the plurality over the draws with its first `j` coins (`plur` at the fork at `j`) with probability at
most `√(Pr[two draws forked at j, where g holds at both, disagree])`: `Pr_plur_sq_le` at each prefix, and Jensen. -/
theorem Pr_plur_fork {Coin : Type} [Fintype Coin] [Nonempty Coin] {Ω C : Type} [Fintype Ω] [Fintype C] [Nonempty C]
    {N : ℕ} (j : ℕ) (g : Ω → (Fin N → Coin) × C → Prop) (f : Ω → (Fin N → Coin) × C → Msg) :
    Pr (fun s : Ω × (Fin N → Coin) × C => g s.1 s.2 ∧
      f s.1 s.2 ≠ plur (fun u : (Fin N → Coin) × C => g s.1 (fork j s.2.1 u.1, u.2))
        (fun u => f s.1 (fork j s.2.1 u.1, u.2))) ≤
      Real.sqrt (Pr (fun t : Ω × (Fin N → Coin) × (Fin N → Coin) × C × C => g t.1 (t.2.1, t.2.2.2.1) ∧
        g t.1 (fork j t.2.1 t.2.2.1, t.2.2.2.2) ∧ f t.1 (t.2.1, t.2.2.2.1) ≠ f t.1 (fork j t.2.1 t.2.2.1, t.2.2.2.2))) := by
  have hoff : PrefixDet (k := N) (Coin := Coin) fun _ => j := fun _ _ _ => rfl
  let Gr : Ω → (Fin N → Coin) → (Fin N → Coin) × C → Prop := fun ω r u => g ω (fork j r u.1, u.2)
  let Fr : Ω → (Fin N → Coin) → (Fin N → Coin) × C → Msg := fun ω r u => f ω (fork j r u.1, u.2)
  let a : Ω → (Fin N → Coin) → ℝ := fun ω r => Pr fun u => Gr ω r u ∧ Fr ω r u ≠ plur (Gr ω r) (Fr ω r)
  let b : Ω → (Fin N → Coin) → ℝ := fun ω r =>
    Pr fun p : ((Fin N → Coin) × C) × ((Fin N → Coin) × C) => Gr ω r p.1 ∧ Gr ω r p.2 ∧ Fr ω r p.1 ≠ Fr ω r p.2
  have hGr : ∀ ω r u, Gr ω (fork j r u) = Gr ω r := fun ω r u => funext fun v => by simp only [Gr, fork_fork]
  have hFr : ∀ ω r u, Fr ω (fork j r u) = Fr ω r := fun ω r u => funext fun v => by simp only [Fr, fork_fork]
  have hL : Pr (fun s : Ω × (Fin N → Coin) × C => g s.1 s.2 ∧
      f s.1 s.2 ≠ plur (fun u : (Fin N → Coin) × C => g s.1 (fork j s.2.1 u.1, u.2))
        (fun u => f s.1 (fork j s.2.1 u.1, u.2))) = Ex fun ω => Ex fun r => a ω r := by
    rw [Pr_prod]
    refine congrArg Ex (funext fun ω => ?_)
    rw [Pr_prod]
    rw [← Ex_fork hoff (fun r => Pr fun c : C => g ω (r, c) ∧ f ω (r, c) ≠ plur (Gr ω r) (Fr ω r)), Ex_prod]
    refine congrArg Ex (funext fun r => ?_)
    simp only [hGr, hFr]
    show _ = Pr (fun u : (Fin N → Coin) × C => g ω (fork j r u.1, u.2) ∧ f ω (fork j r u.1, u.2) ≠ plur (Gr ω r) (Fr ω r))
    rw [Pr_prod (fun u : (Fin N → Coin) × C => g ω (fork j r u.1, u.2) ∧ f ω (fork j r u.1, u.2) ≠ plur (Gr ω r) (Fr ω r))]
  have hR : Pr (fun t : Ω × (Fin N → Coin) × (Fin N → Coin) × C × C => g t.1 (t.2.1, t.2.2.2.1) ∧
      g t.1 (fork j t.2.1 t.2.2.1, t.2.2.2.2) ∧ f t.1 (t.2.1, t.2.2.2.1) ≠ f t.1 (fork j t.2.1 t.2.2.1, t.2.2.2.2)) =
      Ex fun ω => Ex fun r => b ω r := by
    rw [Pr_prod]
    refine congrArg Ex (funext fun ω => ?_)
    rw [Pr_prod]
    rw [← Ex_fork hoff (fun r => Pr fun t : (Fin N → Coin) × C × C => g ω (r, t.2.1) ∧
      g ω (fork j r t.1, t.2.2) ∧ f ω (r, t.2.1) ≠ f ω (fork j r t.1, t.2.2)), Ex_prod]
    refine congrArg Ex (funext fun r => ?_)
    let e : (Fin N → Coin) × (Fin N → Coin) × C × C ≃ ((Fin N → Coin) × C) × ((Fin N → Coin) × C) :=
      { toFun := fun t => ((t.1, t.2.2.1), (t.2.1, t.2.2.2))
        invFun := fun p => (p.1.1, p.2.1, p.1.2, p.2.2)
        left_inv := fun _ => rfl
        right_inv := fun _ => rfl }
    show _ = Pr fun p : ((Fin N → Coin) × C) × ((Fin N → Coin) × C) =>
      g ω (fork j r p.1.1, p.1.2) ∧ g ω (fork j r p.2.1, p.2.2) ∧ f ω (fork j r p.1.1, p.1.2) ≠ f ω (fork j r p.2.1, p.2.2)
    rw [← Pr_equiv e, Pr_prod]
    simp only [fork_fork]
    rfl
  have hab : ∀ ω r, a ω r ≤ Real.sqrt (b ω r) := fun ω r =>
    Real.le_sqrt_of_sq_le (Pr_plur_sq_le (Gr ω r) (Fr ω r))
  rw [hL, hR]
  calc (Ex fun ω => Ex fun r => a ω r) ≤ Ex fun ω => Ex fun r => Real.sqrt (b ω r) :=
        Ex_mono fun ω => Ex_mono fun r => hab ω r
    _ = Ex fun p : Ω × (Fin N → Coin) => Real.sqrt (b p.1 p.2) :=
        (Ex_prod fun p : Ω × (Fin N → Coin) => Real.sqrt (b p.1 p.2)).symm
    _ ≤ Real.sqrt (Ex fun p : Ω × (Fin N → Coin) => b p.1 p.2) :=
        Ex_sqrt_le (fun p : Ω × (Fin N → Coin) => b p.1 p.2) fun p => Pr_nonneg _
    _ = Real.sqrt (Ex fun ω => Ex fun r => b ω r) := by rw [Ex_prod fun p : Ω × (Fin N → Coin) => b p.1 p.2]

/-! ## Coins -/

/-- Coins `l` as the first of `N`. -/
def preCoins {Coin : Type} [Inhabited Coin] {N : ℕ} (l : List Coin) : Fin N → Coin := fun i => l.getD i.val default

theorem fork_preCoins {Coin : Type} [Inhabited Coin] {N : ℕ} (j : ℕ) (r u : Fin N → Coin) :
    fork j (preCoins ((List.range j).map (coinSeq r))) u = fork j r u := by
  funext i
  unfold fork preCoins
  split_ifs with hi
  · simp [List.getD_eq_getElem?_getD, hi, coinSeq]
  · rfl

/-- **The inner coins `ρ`, then the coins `r` past them**: the compiled verifier's coins of an inner run at `ρ`. -/
def extend {Coin : Type} [Inhabited Coin] {n N : ℕ} (ρ : Fin n → Coin) (r : Fin N → Coin) : Fin N → Coin :=
  fork n (fun i => coinSeq ρ i.val) r

/-- An inner run's coins with its abort coins, as a draw of the compiled verifier's coins and `E_O`'s. -/
def extendDraw {Coin C : Type} [Inhabited Coin] {n N : ℕ} (t : (Fin n → Coin) × (Fin N → Coin) × C) :
    (Fin N → Coin) × C :=
  (extend t.1 t.2.1, t.2.2)

theorem extend_forkCoins {Coin : Type} [Inhabited Coin] {n N : ℕ} (j : Fin n) (ρ ρ' : Fin n → Coin)
    (r r' : Fin N → Coin) : fork j.val (extend ρ r) (extend ρ' r') = extend (forkCoins j ρ ρ') r' := by
  funext i
  have hj := j.isLt
  simp only [fork, extend, forkCoins, coinSeq]
  split_ifs <;> first | rfl | omega

theorem UnifMap.self {A : Type} [Fintype A] : UnifMap (fun a : A => a) := fun _ => rfl

/-- **The inner coins and the coins past them are the compiled verifier's, uniform** where the inner rounds are among
the compiled verifier's: the inner coins are its first `n` (`UnifMap.comp_emb`), and a fork at `n` keeps coins uniform
(`Ex_fork`). -/
theorem unifMap_extend {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {n N : ℕ} (h : n ≤ N) :
    UnifMap (fun p : (Fin n → Coin) × (Fin N → Coin) => extend p.1 p.2) := fun G => by
  have hres : UnifMap (fun f : Fin N → Coin => f ∘ Fin.castLEEmb h) := UnifMap.comp_emb (Fin.castLEEmb h)
  have h1 : ∀ r' r : Fin N → Coin, extend (r' ∘ Fin.castLEEmb h) r = fork n r' r := fun r' r => by
    funext i
    unfold extend fork
    split_ifs with hi
    · simp [coinSeq, hi]
    · rfl
  have hoff : PrefixDet (k := N) (Coin := Coin) fun _ => n := fun _ _ _ => rfl
  calc Ex (fun p : (Fin n → Coin) × (Fin N → Coin) => G (extend p.1 p.2))
      = Ex (fun p : (Fin N → Coin) × (Fin N → Coin) => G (extend (p.1 ∘ Fin.castLEEmb h) p.2)) :=
        ((hres.prod UnifMap.self) fun p : (Fin n → Coin) × (Fin N → Coin) => G (extend p.1 p.2)).symm
    _ = Ex (fun p : (Fin N → Coin) × (Fin N → Coin) => G (fork n p.1 p.2)) := by simp only [h1]
    _ = Ex G := Ex_fork hoff G

theorem unifMap_extendDraw {Coin C : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [Fintype C] {n N : ℕ}
    (h : n ≤ N) : UnifMap (fun t : (Fin n → Coin) × (Fin N → Coin) × C => extendDraw t) := by
  let e : (Fin n → Coin) × (Fin N → Coin) × C ≃ ((Fin n → Coin) × (Fin N → Coin)) × C :=
    { toFun := fun t => ((t.1, t.2.1), t.2.2)
      invFun := fun p => (p.1.1, p.1.2, p.2)
      left_inv := fun _ => rfl
      right_inv := fun _ => rfl }
  exact (UnifMap.equiv e).comp ((unifMap_extend h).prod UnifMap.self)

/-- `B`'s coins at aborting runs, read as `B_I`'s: each run's inner coins extended by its abort coins. -/
theorem unifMap_bAbort {Coin Ω C : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [Fintype Ω] [Fintype C]
    {n N : ℕ} (hn : n ≤ N) :
    UnifMap (fun s : (Fin n × Ω × (Fin n → Coin) × (Fin n → Coin)) × ((Fin N → Coin) × C) × ((Fin N → Coin) × C) =>
      (s.1.1, s.1.2.1, extend s.1.2.2.1 s.2.1.1, extend s.1.2.2.2 s.2.2.1, s.2.1.2, s.2.2.2)) := by
  let e : (Fin n × Ω × (Fin n → Coin) × (Fin n → Coin)) × ((Fin N → Coin) × C) × ((Fin N → Coin) × C) ≃
      Fin n × Ω × ((Fin n → Coin) × (Fin N → Coin)) × ((Fin n → Coin) × (Fin N → Coin)) × C × C :=
    { toFun := fun s => (s.1.1, s.1.2.1, (s.1.2.2.1, s.2.1.1), (s.1.2.2.2, s.2.2.1), s.2.1.2, s.2.2.2)
      invFun := fun t => ((t.1, t.2.1, t.2.2.1.1, t.2.2.2.1.1), (t.2.2.1.2, t.2.2.2.2.1), (t.2.2.2.1.2, t.2.2.2.2.2))
      left_inv := fun _ => rfl
      right_inv := fun _ => rfl }
  exact (UnifMap.equiv e).comp (UnifMap.self.prod (UnifMap.self.prod ((unifMap_extend hn).prod
    ((unifMap_extend hn).prod UnifMap.self))))

/-! ## The inner prover -/

section Coupling

variable {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- `E_O`'s output at a draw: the run's coins and `E_O`'s. -/
noncomputable def eoOut (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (d : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) : Option (List Flock.Wit) :=
  ((EO D I DO decO).eval H P x ω d.1 d.2).1

/-- The inner view's message `j` at a draw, as the inner stage sees it. -/
noncomputable def eoMsg (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (d : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) (j : ℕ) : Msg :=
  ((dec H D x d.1 (eoOut H D I DO decO P x ω d)).msgs.map Prod.fst).getD j []

/-- **The inner prover's message at inner round `j`**, on the inner coins `l` so far: the inner view's message `j`
that extractions most often open over the draws with those coins first, where `E_O` gives witnesses (`plur`). -/
noncomputable def opAt (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (j : ℕ) (l : List Coin) : Msg :=
  plur (fun u : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x =>
      (eoOut H D I DO decO P x ω (fork j (preCoins l) u.1, u.2)).isSome = true)
    (fun u => eoMsg H D I DO decO P x ω (fork j (preCoins l) u.1, u.2) j)

/-- **`P_I`, the inner prover**: at the compiled prover's tape, it reads the inner coins so far off the inner
verifier's answers (`decC`) and sends `opAt` there. -/
noncomputable def innerP (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) {Ω : Type} (P : Prover Ω) (x : Msg) (decC : Msg → Msg → Coin) : Prover Ω :=
  ⟨fun _ hist ω => .ret (opAt H D I DO decO P x ω hist.length (hist.map fun h => decC (xI D x) h.2))⟩

/-- The draw's inner view agrees with `P_I` at every inner round. -/
def Agrees (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (d : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) : Prop :=
  ∀ j < I.V.k (xI D x), eoMsg H D I DO decO P x ω d j = opAt H D I DO decO P x ω j ((List.range j).map (coinSeq d.1))

/-- **A draw that `P_I`'s runs don't show**: `E_O` gives witnesses there, and the inner view disagrees with `P_I`. -/
def Bad (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (d : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) : Prop :=
  (eoOut H D I DO decO P x ω d).isSome = true ∧ ¬ Agrees H D I DO decO P x ω d

open Classical in
/-- `P_I`'s run at a draw is kept where `E_O` gives witnesses and the inner view agrees with `P_I`. -/
noncomputable def keepD (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (d : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) : Bool :=
  decide ((eoOut H D I DO decO P x ω d).isSome = true ∧ Agrees H D I DO decO P x ω d)

/-- **`P_I`'s run at a draw, as it aborts**: at the draw's first `k_I` coins, unless `keepD` refuses it. -/
noncomputable def innerF (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) {Ω : Type} (P : Prover Ω) (x : Msg) (decC : Msg → Msg → Coin) (ω : Ω)
    (d : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) : Trace :=
  if keepD H D I DO decO P x ω d then run H (innerP H D I DO decO P x decC) I.V (xI D x) ω (innerCoins D I x d.1)
  else abortTrace

/-- **`P_I`'s aborts**: an inner run at the inner coins `ρ` and the abort coins `a` (the compiled verifier's other
coins and `E_O`'s) is kept as `keepD` keeps the draw they extend to. -/
noncomputable def keepA (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (ρ : Fin (I.V.k (xI D x)) → Coin) (a : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) : Bool :=
  keepD H D I DO decO P x ω (extendDraw (ρ, a))

/-! ## `E*`'s view is `P_I`'s aborting runs' wherever every rerun agrees -/

theorem EO_some_acc (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (r : Fin (D.V.k x) → Coin)
    (c : (EO D I DO decO).C x) {ws : List Flock.Wit} (h : ((EO D I DO decO).eval H P x ω r c).1 = some ws) :
    (run H P D.V x ω r).out = .acc := by
  by_contra hne
  simp only [Extractor.eval, EO, RP.eval, Trace.seen, hne, ↓reduceIte] at h
  cases h

theorem eoGives_of_some {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {decO : List (ℕ × Col) → List T} {Ω : Type} {P : Prover Ω} {x : Msg} {ω : Ω}
    {d : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x} {ws : List Flock.Wit}
    (h : eoOut H D I DO decO P x ω d = some ws) : EOGives H D I DO decO P x ω d.1 d.2 ws :=
  ⟨EO_some_acc H D I DO decO P x ω d.1 d.2 h, h⟩

omit [Fintype Coin] [Nonempty Coin] in
theorem coinSeq_innerCoins (D : Deploy Coin) (I : Inner Coin W) (x : Msg) (r : Fin (D.V.k x) → Coin) {i : ℕ}
    (hi : i < I.V.k (xI D x)) : coinSeq (innerCoins D I x r) i = coinSeq r i := by
  simp only [coinSeq, hi, ↓reduceDIte, innerCoins]

omit [Fintype Coin] [Nonempty Coin] in
theorem innerCoins_extend (D : Deploy Coin) (I : Inner Coin W) (x : Msg) (hn : I.V.k (xI D x) ≤ D.V.k x)
    (ρ : Fin (I.V.k (xI D x)) → Coin) (r : Fin (D.V.k x) → Coin) : innerCoins D I x (extend ρ r) = ρ := by
  funext i
  have hi : i.val < D.V.k x := lt_of_lt_of_le i.isLt hn
  simp only [innerCoins, coinSeq, hi, ↓reduceDIte, extend, fork, i.isLt, ↓reduceIte, Fin.eta]

/-- **Where the inner view agrees with `P_I`, `P_I`'s run is the inner view's run**: the inner verifier's answers show
its coins (`PublicCoin`), so at each round `P_I` reads the inner coins so far and sends the inner view's message. -/
theorem innerF_eq_innerRun {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {decO : List (ℕ × Col) → List T} {Ω : Type} {P : Prover Ω} {x : Msg} {decC : Msg → Msg → Coin}
    (hpc : PublicCoin H I decC) {ω : Ω} {d : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x} {ws : List Flock.Wit}
    (h : eoOut H D I DO decO P x ω d = some ws) (ha : Agrees H D I DO decO P x ω d) :
    innerF H D I DO decO P x decC ω d = innerRun H D I x d.1 ws := by
  have hk : keepD H D I DO decO P x ω d = true := by
    simp only [keepD, decide_eq_true_eq]
    exact ⟨by rw [h]; rfl, ha⟩
  unfold innerF innerRun run
  simp only [hk, ↓reduceIte]
  have step : ∀ (hist : List (Msg × Msg)) (as : List Msg) (a m : Msg), as.length = hist.length →
      hist.map (fun h => decC (xI D x) h.2) = (List.range hist.length).map (coinSeq (innerCoins D I x d.1)) →
      ((I.V.next (xI D x) (as ++ [a]) ((List.range (as ++ [a]).length).map
        (coinSeq (innerCoins D I x d.1)))).eval H).1 = .msg m →
      (hist ++ [(a, m)]).map (fun h => decC (xI D x) h.2) =
        (List.range (hist ++ [(a, m)]).length).map (coinSeq (innerCoins D I x d.1)) := by
    intro hist as a m hl hd hv
    have hlen : (as ++ [a]).length = hist.length + 1 := by simp [hl]
    rw [hlen, List.range_succ, List.map_append, List.map_singleton] at hv
    have hc := hpc (xI D x) (as ++ [a]) _ _ m (by simp [hl]) hv
    rw [List.map_append, hd, List.length_append, List.length_singleton, List.range_succ, List.map_append,
      List.map_singleton, List.map_singleton, hc]
  refine stepWith_congr H (innerP H D I DO decO P x decC) (replay ((D.innerView H x d.1 ws).msgs.map Prod.fst)) (xI D x)
    ω () _ (fun M hist as => hist.length + M = I.V.k (xI D x) ∧ as.length = hist.length ∧
      hist.map (fun h => decC (xI D x) h.2) = (List.range hist.length).map (coinSeq (innerCoins D I x d.1)))
    ?_ ?_ _ _ _ _ ⟨by simp, rfl, rfl⟩
  · intro M hist as hG
    have hL : hist.length < I.V.k (xI D x) := by omega
    have hc : (List.range hist.length).map (coinSeq (innerCoins D I x d.1)) =
        (List.range hist.length).map (coinSeq d.1) :=
      List.map_congr_left fun i hi => coinSeq_innerCoins D I x d.1 (by rw [List.mem_range] at hi; omega)
    simp only [innerP, replay, OC.eval, hG.2.2, hc]
    rw [← ha _ hL]
    simp [eoMsg, h, dec]
  · intro M hist as m hG hv
    exact ⟨by simp only [List.length_append, List.length_singleton]; omega,
      by simp only [List.length_append, List.length_singleton, hG.2.1], step hist as _ m hG.2.1 hG.2.2 hv⟩

/-- **Where a draw isn't bad, the inner stage sees there what it sees of `P_I`'s aborting run**: a rejected run where
`E_O` gives up, and the inner view's run, which is `P_I`'s, where the inner view agrees with `P_I`. -/
theorem servedRun_seen_eq {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {decO : List (ℕ × Col) → List T} {Ω : Type} {P : Prover Ω} {x : Msg} {decC : Msg → Msg → Coin}
    (hV : InnerViewRun H D I DO) (hpc : PublicCoin H I decC) {ω : Ω}
    {d : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x} (hb : ¬ Bad H D I DO decO P x ω d) :
    (servedRun H (Claims.outerStage (EO D I DO decO) (dec H D) x) (run H P D.V x ω) d).seen =
      (innerF H D I DO decO P x decC ω d).seen := by
  rw [servedRun_seen, outerStage_eval]
  change dec H D x d.1 (eoOut H D I DO decO P x ω d) = _
  rcases ho : eoOut H D I DO decO P x ω d with _ | ws
  · have hk : keepD H D I DO decO P x ω d = false := by
      unfold keepD
      simp [ho]
    simp only [innerF, hk]
    rfl
  · have ha : Agrees H D I DO decO P x ω d := by
      by_contra hn
      exact hb ⟨by rw [ho]; rfl, hn⟩
    rw [innerF_eq_innerRun hpc ho ha]
    exact hV decO x Ω P ω d.1 d.2 ws (eoGives_of_some ho)

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem innerR_ext (D : Deploy Coin) (I : Inner Coin W) (x : Msg) (o : Option W) :
    (innerR D I).ext x o ↔ I.R.ext (xI D x) o := by
  cases o <;> exact Iff.rfl

theorem abortRun_eq {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {decO : List (ℕ × Col) → List T} {Ω : Type} {P : Prover Ω} {x : Msg} {decC : Msg → Msg → Coin}
    (hn : I.V.k (xI D x) ≤ D.V.k x) (ω : Ω)
    (t : (Fin (I.V.k (xI D x)) → Coin) × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) :
    abortRun H (innerP H D I DO decO P x decC) I.V (xI D x) (keepA H D I DO decO P x) ω t =
      innerF H D I DO decO P x decC ω (extendDraw t) := by
  simp only [abortRun, innerF, keepA, extendDraw, innerCoins_extend D I x hn]
  rfl

/-- `P_I`'s aborting run records only calls that `E_O`'s witnesses there record, or those witnesses' calls collide
(`CallsLogColl`). -/
theorem innerF_vLog_sub {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {decO : List (ℕ × Col) → List T} {Ω : Type} {P : Prover Ω} {x : Msg} {decC : Msg → Msg → Coin}
    (hL : CallsLogColl H D I DO) (hpc : PublicCoin H I decC) (ω : Ω)
    (d : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) :
    (∀ e ∈ (innerF H D I DO decO P x decC ω d).vLog, e ∈ D.calls x ((eoOut H D I DO decO P x ω d).getD [])) ∨
      ∃ u v, findColl (D.calls x ((eoOut H D I DO decO P x ω d).getD [])) = some (u, v) ∧ Collision H u v := by
  by_cases hk : keepD H D I DO decO P x ω d = true
  · have hk' : (eoOut H D I DO decO P x ω d).isSome = true ∧ Agrees H D I DO decO P x ω d := by
      simpa only [keepD, decide_eq_true_eq] using hk
    obtain ⟨ws, hws⟩ := Option.isSome_iff_exists.1 hk'.1
    rw [innerF_eq_innerRun hpc hws hk'.2, hws]
    exact hL decO x Ω P ω d.1 d.2 ws (eoGives_of_some hws)
  · exact Or.inl fun e he => by simp [innerF, hk, abortTrace] at he

/-- The calls `Be` reads at a draw are true evaluations: `E_O`'s witnesses' where it gives them (`CallsTrue`), none's
where it gives up (`CallsNil`). -/
theorem calls_true {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {decO : List (ℕ × Col) → List T} {Ω : Type} {P : Prover Ω} {x : Msg} (hT : CallsTrue H D I DO)
    (hN0 : CallsNil H D) (ω : Ω) (d : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) :
    ∀ e ∈ D.calls x ((eoOut H D I DO decO P x ω d).getD []), H e.1 = e.2 := by
  rcases hws : eoOut H D I DO decO P x ω d with _ | ws
  · exact hN0 x
  · exact hT decO x Ω P ω d.1 d.2 ws (eoGives_of_some hws)

/-! ## The inner stage's failure -/

/-- **The inner stage fails where `E^rep` fails against `P_I`'s aborting runs, or at a bad draw it reruns at**: where no
draw it reruns at is bad, it sees what `E^rep` sees of `P_I`'s aborting runs (`RP.eval_congr`); the judged draw is
`P_I`'s accepted run, since `E_O` succeeds there and the inner view accepts. A bad rerun costs `N_I` times one draw's
(`Pr_exists_rerun_le`), and `E^rep` at `P_I`'s aborting runs over their own coins is `Inner.PrBadAbort`'s (the inner
coins and the abort coins extend to uniform draws, `unifMap_extendDraw`). -/
theorem PrBadGood_le_abort {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {decC : Msg → Msg → Coin} (hV : InnerViewRun H D I DO) (hAcc : InnerAccepts H D I DO)
    (hpc : PublicCoin H I decC) (hR : InnerRounds D I) (decO : List (ℕ × Col) → List T) (x : Msg) {Ω : Type}
    [Fintype Ω] (P : Prover Ω) :
    PrBadGood H D.V (innerR D I) outR (EO D I DO decO) (dec H D) (fun x => I.waits (xI D x))
        (fun x => I.k (xI D x)) (fun x => I.cap (xI D x)) I.opens (fun x => I.assemble (xI D x))
        (fun x => I.check (xI D x)) x P ≤
      I.PrBadAbort H (xI D x) (innerP H D I DO decO P x decC) (keepA H D I DO decO P x) +
        NI D I x * Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x =>
          Bad H D I DO decO P x t.1 t.2) := by
  let Y : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x ×
      (Fin (I.waits (xI D x)) → Fin (I.cap (xI D x)) → (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) → Prop :=
    fun s => (innerF H D I DO decO P x decC s.1 (s.2.1, s.2.2.1)).out = .acc ∧
      ¬ I.R.ext (xI D x) ((erepProg (I.waits (xI D x)) (I.k (xI D x)) (I.cap (xI D x)) I.opens
        (I.assemble (xI D x)) (I.check (xI D x)) (s.2.1, s.2.2.1) s.2.2.2).eval H
          (innerF H D I DO decO P x decC s.1)).1
  let Z : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x ×
      (Fin (I.waits (xI D x)) → Fin (I.cap (xI D x)) → (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) → Prop :=
    fun s => ∃ d ∈ (erepProg (I.waits (xI D x)) (I.k (xI D x)) (I.cap (xI D x)) I.opens (I.assemble (xI D x))
      (I.check (xI D x)) (s.2.1, s.2.2.1) s.2.2.2).rerunsAt H
        (servedRun H (Claims.outerStage (EO D I DO decO) (dec H D) x) (run H P D.V x s.1)), Bad H D I DO decO P x s.1 d
  have hYZ : PrBadGood H D.V (innerR D I) outR (EO D I DO decO) (dec H D) (fun x => I.waits (xI D x))
      (fun x => I.k (xI D x)) (fun x => I.cap (xI D x)) I.opens (fun x => I.assemble (xI D x))
      (fun x => I.check (xI D x)) x P ≤ Pr Y + Pr Z := by
    refine (Pr_mono fun s hs => ?_).trans (Pr_or_le Y Z)
    by_cases hz : Z s
    · exact Or.inr hz
    · left
      have hnb : ∀ d ∈ (erepProg (I.waits (xI D x)) (I.k (xI D x)) (I.cap (xI D x)) I.opens (I.assemble (xI D x))
          (I.check (xI D x)) (s.2.1, s.2.2.1) s.2.2.2).rerunsAt H
            (servedRun H (Claims.outerStage (EO D I DO decO) (dec H D) x) (run H P D.V x s.1)),
          ¬ Bad H D I DO decO P x s.1 d := fun d hd hb => hz ⟨d, hd, hb⟩
      have heq := RP.eval_congr H _ (innerF H D I DO decO P x decC s.1) _ fun d hd =>
        servedRun_seen_eq hV hpc (hnb d hd)
      have hd₀ : (s.2.1, s.2.2.1) ∈ (erepProg (I.waits (xI D x)) (I.k (xI D x)) (I.cap (xI D x)) I.opens
          (I.assemble (xI D x)) (I.check (xI D x)) (s.2.1, s.2.2.1) s.2.2.2).rerunsAt H
            (servedRun H (Claims.outerStage (EO D I DO decO) (dec H D) x) (run H P D.V x s.1)) := List.mem_cons_self
      have hgood := hs.2.1 _ hd₀
      obtain ⟨ws, hws⟩ : ∃ ws, eoOut H D I DO decO P x s.1 (s.2.1, s.2.2.1) = some ws := by
        rcases ho : eoOut H D I DO decO P x s.1 (s.2.1, s.2.2.1) with _ | ws
        · refine absurd ⟨hs.1, fun h => ?_⟩ hgood
          rw [show ((EO D I DO decO).eval H P x s.1 s.2.1 s.2.2.1).1 = none from ho] at h
          exact h
        · exact ⟨ws, rfl⟩
      have ha : Agrees H D I DO decO P x s.1 (s.2.1, s.2.2.1) := by
        by_contra hn
        exact hnb _ hd₀ ⟨by rw [hws]; rfl, hn⟩
      refine ⟨?_, fun hI => hs.2.2 ?_⟩
      · rw [innerF_eq_innerRun hpc hws ha]
        have hacc := hAcc decO x Ω P s.1 s.2.1 s.2.2.1 ws (eoGives_of_some hws)
        rw [hV decO x Ω P s.1 s.2.1 s.2.2.1 ws (eoGives_of_some hws)] at hacc
        exact hacc
      · show (innerR D I).ext x (((erepProg (I.waits (xI D x)) (I.k (xI D x)) (I.cap (xI D x)) I.opens
          (I.assemble (xI D x)) (I.check (xI D x)) (s.2.1, s.2.2.1) s.2.2.2).serve
            (Claims.outerStage (EO D I DO decO) (dec H D) x)).eval H (run H P D.V x s.1)).1
        rw [(RP.serve_eval H _ _ _).1, heq]
        exact (innerR_ext D I x _).2 hI
  have hY : Pr Y = I.PrBadAbort H (xI D x) (innerP H D I DO decO P x decC) (keepA H D I DO decO P x) := by
    have hΦ : UnifMap (fun s : Ω × ((Fin (I.V.k (xI D x)) → Coin) × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) ×
        (Fin (I.waits (xI D x)) → Fin (I.cap (xI D x)) →
          (Fin (I.V.k (xI D x)) → Coin) × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) =>
        (s.1, (extendDraw s.2.1).1, (extendDraw s.2.1).2, fun i j => extendDraw (s.2.2 i j))) := by
      let e : Ω × ((Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) ×
          (Fin (I.waits (xI D x)) → Fin (I.cap (xI D x)) → (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) ≃
          Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x ×
          (Fin (I.waits (xI D x)) → Fin (I.cap (xI D x)) → (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) :=
        { toFun := fun s => (s.1, s.2.1.1, s.2.1.2, s.2.2)
          invFun := fun s => (s.1, (s.2.1, s.2.2.1), s.2.2.2)
          left_inv := fun _ => rfl
          right_inv := fun _ => rfl }
      exact (UnifMap.self.prod ((unifMap_extendDraw (hR x)).prod
        (UnifMap.pi (UnifMap.pi (unifMap_extendDraw (hR x)) _) _))).comp (UnifMap.equiv e)
    rw [← hΦ.Pr Y]
    refine Pr_congr fun s => ?_
    show (innerF H D I DO decO P x decC s.1 (extendDraw s.2.1)).out = .acc ∧
      ¬ I.R.ext (xI D x) ((erepProg (I.waits (xI D x)) (I.k (xI D x)) (I.cap (xI D x)) I.opens
        (I.assemble (xI D x)) (I.check (xI D x)) (extendDraw s.2.1) (fun i j => extendDraw (s.2.2 i j))).eval H
          (innerF H D I DO decO P x decC s.1)).1 ↔ _
    have hf : (fun d => innerF H D I DO decO P x decC s.1 (extendDraw d)) =
        abortRun H (innerP H D I DO decO P x decC) I.V (xI D x) (keepA H D I DO decO P x) s.1 :=
      funext fun d => (abortRun_eq (hR x) s.1 d).symm
    rw [erepProg_eval_comp, hf, ← abortRun_eq (hR x)]
    rfl
  have hZ : Pr Z ≤ NI D I x * Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x =>
      Bad H D I DO decO P x t.1 t.2) := by
    have hN : (NI D I x : ℝ) = 1 + (I.waits (xI D x) : ℝ) * (I.k (xI D x) : ℝ) := by
      simp only [NI]; push_cast; ring
    rw [hN]
    exact Pr_exists_rerun_le H D.V (EO D I DO decO) (dec H D) (fun x => I.waits (xI D x)) (fun x => I.k (xI D x))
      (fun x => I.cap (xI D x)) I.opens (fun x => I.assemble (xI D x)) (fun x => I.check (xI D x)) x P
      (Bad H D I DO decO P x)
  linarith

/-- **A draw is bad at most as `√(k_I·Σ_{j<k_I} Pr[OpensDiffer j])`**: it misses `P_I` at some inner round `j`, at
most `√(Pr[two extractions forked at j open round j differently])` (`Pr_plur_fork`), and the rounds' roots sum under
one (Cauchy–Schwarz). Each `Pr_bad_le` form bounds the sum. -/
theorem Pr_bad_le_opens (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) :
    Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x => Bad H D I DO decO P x t.1 t.2) ≤
      Real.sqrt (I.V.k (xI D x) * ∑ j : Fin (I.V.k (xI D x)), Pr (OpensDiffer H D I DO decO P x j)) := by
  have h1 : Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x => Bad H D I DO decO P x t.1 t.2) ≤
      ∑ j : Fin (I.V.k (xI D x)), Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x =>
        (eoOut H D I DO decO P x t.1 t.2).isSome = true ∧
          eoMsg H D I DO decO P x t.1 t.2 j ≠ opAt H D I DO decO P x t.1 j ((List.range j).map (coinSeq t.2.1))) := by
    refine (Pr_mono fun t ht => ?_).trans (Pr_exists_le _)
    obtain ⟨hs, hna⟩ := ht
    simp only [Agrees, not_forall] at hna
    obtain ⟨j, hj, hne⟩ := hna
    exact ⟨⟨j, hj⟩, hs, hne⟩
  have h2 : ∀ j : Fin (I.V.k (xI D x)), Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x =>
      (eoOut H D I DO decO P x t.1 t.2).isSome = true ∧
        eoMsg H D I DO decO P x t.1 t.2 j ≠ opAt H D I DO decO P x t.1 j ((List.range j).map (coinSeq t.2.1))) ≤
      Real.sqrt (Pr (OpensDiffer H D I DO decO P x j)) := fun j => by
    have hp := Pr_plur_fork j (fun ω d => (eoOut H D I DO decO P x ω d).isSome = true)
      (fun ω d => eoMsg H D I DO decO P x ω d j)
    refine (le_of_eq (Pr_congr fun t => ?_)).trans (hp.trans (Real.sqrt_le_sqrt (Pr_mono fun t ht => ?_)))
    · simp only [opAt, fork_preCoins]
    · obtain ⟨hg₁, hg₂, hf⟩ := ht
      obtain ⟨ws, hws⟩ := Option.isSome_iff_exists.1 hg₁
      obtain ⟨ws', hws'⟩ := Option.isSome_iff_exists.1 hg₂
      refine ⟨ws, ws', eoGives_of_some hws, eoGives_of_some hws', ?_⟩
      simpa [eoMsg, hws, hws', dec] using hf
  have hcs := Real.sum_sqrt_mul_sqrt_le (Finset.univ : Finset (Fin (I.V.k (xI D x))))
    (f := fun _ : Fin (I.V.k (xI D x)) => (1 : ℝ)) (g := fun j => Pr (OpensDiffer H D I DO decO P x j))
    (fun _ => zero_le_one) (fun j => Pr_nonneg _)
  simp only [Real.sqrt_one, one_mul, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
    mul_one] at hcs
  calc Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x => Bad H D I DO decO P x t.1 t.2)
      ≤ ∑ j : Fin (I.V.k (xI D x)), Real.sqrt (Pr (OpensDiffer H D I DO decO P x j)) :=
        h1.trans (Finset.sum_le_sum fun j _ => h2 j)
    _ ≤ Real.sqrt (I.V.k (xI D x)) * Real.sqrt (∑ j : Fin (I.V.k (xI D x)), Pr (OpensDiffer H D I DO decO P x j)) :=
        hcs
    _ = Real.sqrt (I.V.k (xI D x) * ∑ j : Fin (I.V.k (xI D x)), Pr (OpensDiffer H D I DO decO P x j)) :=
        (Real.sqrt_mul (Nat.cast_nonneg _) _).symm

/-- **A draw is bad at most as `√(k_I·k(x)·Pr[Be])`**: it misses `P_I` at some inner round `j`, at most
`√(Pr[two extractions forked at j open round j differently])` (`Pr_plur_fork`), which is a collision `Be` finds; the
rounds' roots sum under one (Cauchy–Schwarz) to at most `√(k_I·Σ_j Pr[…])`, and `Σ_j ≤ k(x)·Pr[Be]`
(`sum_opensDiffer_le`, from the played rounds' openings). -/
theorem Pr_bad_le {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {opening : Opening} (hV : InnerViewRun H D I DO) (hO : CallsOpenPlayed H D I DO opening)
    (hT : CallsTrue H D I DO) (hR : InnerRounds D I)
    (decO : List (ℕ × Col) → List T) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) :
    Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x => Bad H D I DO decO P x t.1 t.2) ≤
      Real.sqrt (I.V.k (xI D x) * D.V.k x * PrBe H D I DO decO P x) := by
  rw [mul_assoc]
  exact (Pr_bad_le_opens H D I DO decO x P).trans
    (Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left (sum_opensDiffer_le hV hO hT hR decO x P) (Nat.cast_nonneg _)))

/-- **`B` at `P_I`'s aborting runs finds only collisions `B_I` finds**: each kept run's calls are recorded by `E_O`'s
witnesses at the draw it extends to, or those recorded calls collide (`CallsLogColl`), which `B_I` finds as well; an
aborted run makes none, and `B`'s fork of the inner coins extends to `B_I`'s fork of the compiled verifier's
(`extend_forkCoins`, `unifMap_bAbort`). -/
theorem PrBAbort_le_PrBI {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {decC : Msg → Msg → Coin} (hL : CallsLogColl H D I DO) (hT : CallsTrue H D I DO) (hN0 : CallsNil H D)
    (hpc : PublicCoin H I decC) (hR : InnerRounds D I) (decO : List (ℕ × Col) → List T) (x : Msg) {Ω : Type}
    [Fintype Ω] [Nonempty Ω] (P : Prover Ω) :
    PrBAbort H I.V (innerP H D I DO decO P x decC) (xI D x) (keepA H D I DO decO P x) ≤
      PrBI H D I DO decO P x := by
  have hcoll : ∀ (j : Fin (I.V.k (xI D x))) (ω : Ω) (ρ ρ' : Fin (I.V.k (xI D x)) → Coin)
      (a a' : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x),
      (findColl ((abortRun H (innerP H D I DO decO P x decC) I.V (xI D x) (keepA H D I DO decO P x) ω (ρ, a)).vLog ++
        (abortRun H (innerP H D I DO decO P x decC) I.V (xI D x) (keepA H D I DO decO P x) ω
          (forkCoins j ρ ρ', a')).vLog)).isSome →
      ∃ u v, findColl (D.calls x ((eoOut H D I DO decO P x ω (extend ρ a.1, a.2)).getD []) ++
        D.calls x ((eoOut H D I DO decO P x ω (fork j.val (extend ρ a.1) (extend ρ' a'.1), a'.2)).getD [])) =
          some (u, v) ∧ Collision H u v := by
    intro j ω ρ ρ' a a' h
    rw [abortRun_eq (hR x), abortRun_eq (hR x)] at h
    rw [extend_forkCoins]
    have htrue : ∀ p ∈ D.calls x ((eoOut H D I DO decO P x ω (extend ρ a.1, a.2)).getD []) ++
        D.calls x ((eoOut H D I DO decO P x ω (extend (forkCoins j ρ ρ') a'.1, a'.2)).getD []), H p.1 = p.2 :=
      fun p hp => (List.mem_append.1 hp).elim (calls_true hT hN0 ω _ p) (calls_true hT hN0 ω _ p)
    have hsome : (findColl (D.calls x ((eoOut H D I DO decO P x ω (extend ρ a.1, a.2)).getD []) ++
        D.calls x ((eoOut H D I DO decO P x ω (extend (forkCoins j ρ ρ') a'.1, a'.2)).getD []))).isSome := by
      rcases innerF_vLog_sub (decC := decC) hL hpc ω (extendDraw (ρ, a)) with h1 | ⟨u, v, huv, -⟩
      · rcases innerF_vLog_sub (decC := decC) hL hpc ω (extendDraw (forkCoins j ρ ρ', a')) with
          h2 | ⟨u, v, huv, -⟩
        · exact findColl_isSome_of_sub H htrue (fun p hp => (List.mem_append.1 hp).elim
            (fun hp => List.mem_append_left _ (h1 p hp)) fun hp => List.mem_append_right _ (h2 p hp)) h
        · exact findColl_isSome_of_sub H htrue (fun p hp => List.mem_append_right _ hp)
            (Option.isSome_iff_exists.2 ⟨_, huv⟩)
      · exact findColl_isSome_of_sub H htrue (fun p hp => List.mem_append_left _ hp)
          (Option.isSome_iff_exists.2 ⟨_, huv⟩)
    obtain ⟨⟨u, v⟩, huv⟩ := Option.isSome_iff_exists.1 hsome
    exact ⟨u, v, huv, findColl_spec H _ htrue u v huv⟩
  unfold PrBAbort PrBI
  by_cases hk : 0 < I.V.k (xI D x)
  · simp only [hk, ↓reduceDIte]
    unfold Finder.Pr
    refine (Pr_mono fun s hs => ?_).trans_eq
      ((unifMap_bAbort (Ω := Ω) (C := (EO D I DO decO).C x) (hR x)).Pr _)
    obtain ⟨u, v, huv, -⟩ := hs
    exact hcoll s.1.1 s.1.2.1 s.1.2.2.1 s.1.2.2.2 s.2.1 s.2.2 (Option.isSome_iff_exists.2 ⟨(u, v), huv⟩)
  · simp only [hk, ↓reduceDIte, le_refl]

/-- **The inner protocol's KS, carried to the served runs, from its KS against provers that may abort**: `E*`'s inner
stage fails where `E^rep` fails against `P_I`'s aborting runs, at most `κ_I + √(γ_I·Pr[B at P_I's aborting runs])`
(`Inner.KSAbort`), with `Pr[B at P_I's aborting runs] ≤ Pr[B_I]`, plus at a bad rerun, at most
`N_I·√(k_I·k(x)·Pr[Be])`. -/
theorem innerKS_of_KSAbort [DecidableEq Col] [Zero Col] {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W}
    {DO : Flock.Deploy Coin Col T} {opening : Opening} {decC : Msg → Msg → Coin} (hKS : I.KSAbort H)
    (hR : InnerRounds D I) (hV : InnerViewRun H D I DO) (hT : CallsTrue H D I DO) (hL : CallsLogColl H D I DO)
    (hO : CallsOpenPlayed H D I DO opening) (hAcc : InnerAccepts H D I DO) (hpc : PublicCoin H I decC)
    (hN0 : CallsNil H D) : InnerKS H D I DO := by
  intro decO _ x Ω _ _ P
  have h1 := PrBadGood_le_abort hV hAcc hpc hR decO x P
  have h2 := hKS (xI D x) Ω (innerP H D I DO decO P x decC) ((Fin (D.V.k x) → Coin) × (EO D I DO decO).C x)
    (keepA H D I DO decO P x)
  have h3 := PrBAbort_le_PrBI hL hT hN0 hpc hR decO x P
  have h4 := Pr_bad_le hV hO hT hR decO x P
  have hBA0 : 0 ≤ PrBAbort H I.V (innerP H D I DO decO P x decC) (xI D x) (keepA H D I DO decO P x) := by
    unfold PrBAbort; split
    · exact Pr_nonneg _
    · exact le_rfl
  have h5 : Real.sqrt (I.γ (xI D x) * PrBAbort H I.V (innerP H D I DO decO P x decC) (xI D x)
      (keepA H D I DO decO P x)) ≤ Real.sqrt (I.γ (xI D x) * PrBI H D I DO decO P x) := by
    by_cases hγ : 0 ≤ I.γ (xI D x)
    · exact Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left h3 hγ)
    · rw [Real.sqrt_eq_zero'.2 (mul_nonpos_of_nonpos_of_nonneg (not_le.1 hγ).le hBA0)]
      exact Real.sqrt_nonneg _
  calc PrBadGood H D.V (innerR D I) outR (EO D I DO decO) (dec H D) (fun x => I.waits (xI D x))
        (fun x => I.k (xI D x)) (fun x => I.cap (xI D x)) I.opens (fun x => I.assemble (xI D x))
        (fun x => I.check (xI D x)) x P
      ≤ I.PrBadAbort H (xI D x) (innerP H D I DO decO P x decC) (keepA H D I DO decO P x) +
        NI D I x * Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x =>
          Bad H D I DO decO P x t.1 t.2) := h1
    _ ≤ I.κ (xI D x) + Real.sqrt (I.γ (xI D x) * PrBI H D I DO decO P x) +
        NI D I x * Real.sqrt (I.V.k (xI D x) * D.V.k x * PrBe H D I DO decO P x) :=
        add_le_add (h2.trans (add_le_add le_rfl h5)) (mul_le_mul_of_nonneg_left h4 (Nat.cast_nonneg _))

end Coupling

/-! ## The recursion's KS -/

section RecursionKS

variable {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

/-- **The recursion's KS from Flock's KS at the outer deployment and the inner protocol's KS against provers that may
abort**: `KS_of`, its `OuterKS` from `outerKS_of_flockKS` and its `InnerKS` from `innerKS_of_KSAbort`. It rests on
`Claims.Flock.KS` at `DO`, `Inner.KSAbort` and code facts only. -/
theorem KS_of_flockKS [Zero Col] (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (opening : Opening) (decC : Msg → Msg → Coin) (hF : Flock.KS H DO) (hKS : I.KSAbort H)
    (hRef : Refuses H D) (hRounds : InnerRounds D I) (hN : SessionNums D DO) (hε : TableErrNonneg D)
    (hRd : SessionRounds D I DO) (hS : SessionAccepts D) (hR : SessionRuns H D I DO) (hV : InnerViewRun H D I DO)
    (hT : CallsTrue H D I DO) (hL : CallsLogColl H D I DO) (hO : CallsOpenPlayed H D I DO opening)
    (hAcc : InnerAccepts H D I DO) (hpc : PublicCoin H I decC) (hN0 : CallsNil H D) : KSA H D I DO :=
  KS_of H D I DO hRef hRounds (outerKS_of_flockKS hF hN hε hRd hS hR)
    (innerKS_of_KSAbort hKS hRounds hV hT hL hO hAcc hpc hN0)

end RecursionKS

end Claims.Recursion
