import Proofs.Flock.Claims.Prob

/-!
# The fork's coin coupling at coins that are uniform functions of the live history

A simulator whose deployed coin `j` takes its first `n j` entries from the live coin and the rest from a uniform
function `f j` on its tape, read at a key of the live coins (`fnC`, `fnAt`), has coins of uniform law; its fork, which
redraws the live coins past round `t`, is the deployed fork until the redrawn live coin agrees with the first
branch's on the entries the key reads. That is identical-until-bad (Bellare–Rogaway), used multiplicatively: on the
bad event the simulated fork at round `t` is the simulated fork at `t + 1`.

* `PrefixKey`: the key at round `j` reads, and tells apart, the live coins' revealed entries through round `j`;
  `histKey`, the revealed history itself, is one (`prefixKey_histKey`).
* `fnC_coupling`: for every pair event `E`, the simulated fork at round `t` (`simF`) is at most the deployed fork there
  (`depF`) and `δ_t` times the simulated fork at `t + 1`, with `δ_t` the chance that two coins agree on their first
  `n t` entries (`δAt`, which is `|X|^{-min (n t) M}`, `δAt_eq`), and past the last round it is the one-run event
  `E s s`.
  - On agreement at `t`, the forks at `t` and `t + 1` are the same coins (`fnC_fork_succ`), and agreement is
    independent of the rest with chance `δ_t` whatever the first coin (`Pr_agree`, `Pr_and_agree`).
  - Otherwise the keys differ from round `t` on and agree before it, so the tape's values at the two keys are
    independent and uniform (`Pr_evalAt₂`), and the pair is the deployed fork's, `Pr_mergeC` (`Pr_disagree_le`).
* `fnAt_coupling`: the same at `fnAt`, the live coins indexed by `Fin L`, `L = K`, in the shape of
  `Claims.Flock.ForkCoinCoupling` at `simForkPr`, with the deployed fork written by `Claims.forkCoins`.

`mergeC`, `fnC` and `fnAt` are `Claims.Bridge.Live`'s, verbatim, here so that this file needs only `Prob`.
-/

namespace Claims.Flock.FnFork

open Claims

/-! ## Uniform averages -/

section Prob

open Classical in
/-- The indicator of an event. -/
noncomputable def ind {α : Type} (p : α → Prop) (a : α) : ℝ := if p a then 1 else 0

open Classical in
theorem Pr_eq_Ex {α : Type} [Fintype α] (p : α → Prop) : Pr p = Ex (ind p) := by
  rw [Pr_eq_sum]
  rfl

theorem Ex_equiv {α γ : Type} [Fintype α] [Fintype γ] (e : α ≃ γ) (f : γ → ℝ) :
    Ex (fun a => f (e a)) = Ex f := by
  unfold Ex
  rw [Fintype.sum_equiv e (fun a => f (e a)) f (fun _ => rfl), Fintype.card_congr e]

theorem Ex_prod {α γ : Type} [Fintype α] [Fintype γ] (f : α × γ → ℝ) :
    Ex f = Ex fun a => Ex fun c => f (a, c) := by
  unfold Ex
  rw [Fintype.sum_prod_type, Fintype.card_prod, Finset.sum_div]
  push_cast
  rw [Finset.sum_div]
  refine Finset.sum_congr rfl fun a _ => ?_
  rw [div_div, mul_comm]

theorem Ex_mul_const {α : Type} [Fintype α] (f : α → ℝ) (c : ℝ) : Ex (fun a => f a * c) = Ex f * c := by
  unfold Ex
  rw [← Finset.sum_mul]
  ring

theorem Pr_equiv {α β : Type} [Fintype α] [Fintype β] (e : α ≃ β) (p : β → Prop) :
    Pr (fun a => p (e a)) = Pr p := by
  rw [Pr_eq_Ex, Pr_eq_Ex]
  exact Ex_equiv e (ind p)

theorem Pr_prod {α β : Type} [Fintype α] [Fintype β] (P : α → β → Prop) :
    Pr (fun p : α × β => P p.1 p.2) = Ex fun a => Pr (P a) := by
  rw [Pr_eq_Ex, Ex_prod]
  simp only [Pr_eq_Ex]
  rfl

/-- `Pr[p] = Pr[p ∧ q] + Pr[p ∧ ¬q]`. -/
theorem Pr_split {α : Type} [Fintype α] (p q : α → Prop) :
    Pr p = Pr (fun a => p a ∧ q a) + Pr (fun a => p a ∧ ¬ q a) := by
  rw [Pr_eq_Ex, Pr_eq_Ex, Pr_eq_Ex, ← Ex_add]
  congr 1
  funext a
  unfold ind
  by_cases hp : p a <;> by_cases hq : q a <;> simp [hp, hq]

end Prob

/-! ## Coins from a tape and live coins -/

section Coins

variable {K M : ℕ} {X D : Type}

/-- **Coordinate `j`'s first `n j` entries from `r`, the rest from `u`** (`Claims.Bridge.Live.mergeC`). -/
def mergeC (n : Fin K → ℕ) (u r : Fin K → Fin M → X) : Fin K → Fin M → X :=
  fun j i => if i.val < n j then r j i else u j i

/-- Swapping coordinate `j`'s first `n j` entries between two coin vectors. -/
def swapC (X : Type) (n : Fin K → ℕ) :
    (Fin K → Fin M → X) × (Fin K → Fin M → X) ≃ (Fin K → Fin M → X) × (Fin K → Fin M → X) where
  toFun p := (mergeC n p.1 p.2, mergeC n p.2 p.1)
  invFun p := (mergeC n p.1 p.2, mergeC n p.2 p.1)
  left_inv p := by
    refine Prod.ext (funext fun j => funext fun i => ?_) (funext fun j => funext fun i => ?_) <;>
      simp only [mergeC] <;> split_ifs <;> rfl
  right_inv p := by
    refine Prod.ext (funext fun j => funext fun i => ?_) (funext fun j => funext fun i => ?_) <;>
      simp only [mergeC] <;> split_ifs <;> rfl

/-- **The merged coins are uniform**, as an average. -/
theorem Ex_mergeC [Fintype X] [Nonempty X] (n : Fin K → ℕ) (g : (Fin K → Fin M → X) → ℝ) :
    Ex (fun p : (Fin K → Fin M → X) × (Fin K → Fin M → X) => g (mergeC n p.1 p.2)) = Ex g := by
  refine (Ex_equiv (swapC X n) (fun q => g q.1)).trans ?_
  rw [Ex_prod]
  exact congrArg Ex (funext fun a => Ex_const (g a))

/-- **The merged coins are uniform**: `Pr_{u, r}[E (mergeC n u r)] = Pr[E]`. -/
theorem Pr_mergeC [Fintype X] [Nonempty X] (n : Fin K → ℕ) (E : (Fin K → Fin M → X) → Prop) :
    Pr (fun p : (Fin K → Fin M → X) × (Fin K → Fin M → X) => E (mergeC n p.1 p.2)) = Pr E := by
  rw [Pr_eq_Ex, Pr_eq_Ex]
  exact Ex_mergeC n (ind E)

/-- **Independent uniform functions at fixed points are uniform**, on domains that may vary with the coordinate:
`Pr_f[G (j ↦ f j (d j))] = Pr[G]`. -/
theorem Pr_evalAtDep {Dj : Fin K → Type} [∀ j, Fintype (Dj j)] [∀ j, DecidableEq (Dj j)] {Y : Type} [Fintype Y]
    [Nonempty Y] (d : (j : Fin K) → Dj j) (G : (Fin K → Y) → Prop) :
    Pr (fun f : (j : Fin K) → Dj j → Y => G fun j => f j (d j)) = Pr G := by
  refine (Pr_equiv ((Equiv.piCongrRight fun j => Equiv.funSplitAt (d j) Y).trans
    (Equiv.arrowProdEquivProdArrow (Fin K) (fun _ => Y) fun j => {z // z ≠ d j} → Y)) fun q => G q.1).trans ?_
  rw [Pr_eq_Ex, Ex_prod, Pr_eq_Ex]
  exact congrArg Ex (funext fun a => Ex_const (ind G a))

/-- **Independent uniform functions at fixed points are uniform**: `Pr_f[G (j ↦ f j (d j))] = Pr[G]`. -/
theorem Pr_evalAt {Y : Type} [Fintype D] [DecidableEq D] [Fintype Y] [Nonempty Y] (d : Fin K → D)
    (G : (Fin K → Y) → Prop) : Pr (fun f : Fin K → D → Y => G fun j => f j (d j)) = Pr G :=
  Pr_evalAtDep (Dj := fun _ => D) d G

/-- **Independent uniform functions at two different points each are two independent uniform vectors**:
`Pr_f[G (j ↦ f j (a j)) (j ↦ f j (b j))] = Pr_{u, u'}[G u u']` where `a j ≠ b j` for every `j`: the values at `a`
are uniform and independent of the rest (`funSplitAt`), and the rest at `b` is uniform (`Pr_evalAtDep`). -/
theorem Pr_evalAt₂ {Y : Type} [Fintype D] [DecidableEq D] [Fintype Y] [Nonempty Y] (a b : Fin K → D)
    (hab : ∀ j, b j ≠ a j) (G : (Fin K → Y) → (Fin K → Y) → Prop) :
    Pr (fun f : Fin K → D → Y => G (fun j => f j (a j)) (fun j => f j (b j))) =
      Pr (fun p : (Fin K → Y) × (Fin K → Y) => G p.1 p.2) := by
  refine (Pr_equiv ((Equiv.piCongrRight fun j => Equiv.funSplitAt (a j) Y).trans
    (Equiv.arrowProdEquivProdArrow (Fin K) (fun _ => Y) fun j => {z // z ≠ a j} → Y))
    (fun q => G q.1 (fun j => q.2 j ⟨b j, hab j⟩))).trans ?_
  rw [Pr_prod (α := Fin K → Y) (β := (j : Fin K) → {z // z ≠ a j} → Y)
      (fun u w => G u (fun j => w j ⟨b j, hab j⟩)),
    Pr_prod (α := Fin K → Y) (β := Fin K → Y) (fun u u' => G u u')]
  exact congrArg Ex (funext fun u =>
    Pr_evalAtDep (Dj := fun j => {z // z ≠ a j}) (fun j => ⟨b j, hab j⟩) (G u))

/-- **Coordinate `j`'s first `n j` entries from `r`, the rest from `f j` at `key j r`** (`Claims.Bridge.Live.fnC`). -/
def fnC (n : Fin K → ℕ) (key : Fin K → (Fin K → Fin M → X) → D) (f : Fin K → D → Fin M → X)
    (r : Fin K → Fin M → X) : Fin K → Fin M → X :=
  mergeC n (fun j => f j (key j r)) r

/-- **Those coins are uniform**: `Pr_{f, r}[E (fnC n key f r)] = Pr[E]` at uniform functions `f`, whatever the keys. -/
theorem Pr_fnC [Fintype X] [Nonempty X] [Fintype D] [DecidableEq D] (n : Fin K → ℕ)
    (key : Fin K → (Fin K → Fin M → X) → D) (E : (Fin K → Fin M → X) → Prop) :
    Pr (fun p : (Fin K → D → Fin M → X) × (Fin K → Fin M → X) => E (fnC n key p.1 p.2)) = Pr E := by
  calc Pr (fun p : (Fin K → D → Fin M → X) × (Fin K → Fin M → X) => E (fnC n key p.1 p.2))
      = Pr (fun q : (Fin K → Fin M → X) × (Fin K → D → Fin M → X) => E (fnC n key q.2 q.1)) :=
        Pr_equiv (Equiv.prodComm _ _) fun q => E (fnC n key q.2 q.1)
    _ = Ex fun r => Pr fun f : Fin K → D → Fin M → X => E (fnC n key f r) :=
        Pr_prod fun r f => E (fnC n key f r)
    _ = Ex fun r => Pr fun u : Fin K → Fin M → X => E (mergeC n u r) :=
        congrArg Ex (funext fun r => Pr_evalAt (fun j => key j r) fun u => E (mergeC n u r))
    _ = Pr (fun q : (Fin K → Fin M → X) × (Fin K → Fin M → X) => E (mergeC n q.2 q.1)) :=
        (Pr_prod fun r u => E (mergeC n u r)).symm
    _ = Pr (fun p : (Fin K → Fin M → X) × (Fin K → Fin M → X) => E (mergeC n p.1 p.2)) :=
        Pr_equiv (Equiv.prodComm _ _) fun p => E (mergeC n p.1 p.2)
    _ = Pr E := Pr_mergeC n E

/-- `fnC` with the coins `r` indexed by `Fin L`, where `L = K` (`Claims.Bridge.Live.fnAt`). -/
def fnAt {L : ℕ} (h : L = K) (n : Fin K → ℕ) (key : Fin K → (Fin K → Fin M → X) → D)
    (f : Fin K → D → Fin M → X) (r : Fin L → Fin M → X) : Fin K → Fin M → X :=
  fnC n key f fun j => r (Fin.cast h.symm j)

/-- **The fork at round `t`**: the first `t` coordinates from `r`, the rest from `r'`; at `t < K` it is
`Claims.forkCoins ⟨t, _⟩ r r'`. -/
def forkN {Y : Type} {K : ℕ} (t : ℕ) (r r' : Fin K → Y) : Fin K → Y := fun i => if i.val < t then r i else r' i

/-- **Two coins agree on their first `k` entries.** -/
def AgreeAt (k : ℕ) (a b : Fin M → X) : Prop := ∀ m : Fin M, m.val < k → a m = b m

/-- **The chance that two coins agree on their first `k` entries.** -/
noncomputable def δAt (X : Type) [Fintype X] (M k : ℕ) : ℝ :=
  Pr fun p : (Fin M → X) × (Fin M → X) => AgreeAt k p.1 p.2

/-- **A key that reads the revealed history**: the key at round `j` depends on the coins only through their first
`n i` entries at the rounds `i ≤ j` (`read`), and tells two such histories apart (`sep`). -/
structure PrefixKey (n : Fin K → ℕ) (key : Fin K → (Fin K → Fin M → X) → D) : Prop where
  read : ∀ j r r', (∀ i, i ≤ j → AgreeAt (n i) (r i) (r' i)) → key j r = key j r'
  sep : ∀ j r r', key j r = key j r' → ∀ i, i ≤ j → AgreeAt (n i) (r i) (r' i)

/-- **The revealed history through round `j`**: the first `n i` entries of the coins at rounds `i ≤ j`. -/
def histKey (n : Fin K → ℕ) (j : Fin K) (r : Fin K → Fin M → X) : Fin K → Fin M → Option X :=
  fun i m => if i ≤ j ∧ m.val < n i then some (r i m) else none

theorem prefixKey_histKey (n : Fin K → ℕ) : PrefixKey n (histKey (M := M) (X := X) n) where
  read j r r' h := funext fun i => funext fun m => by
    unfold histKey
    split_ifs with hc
    · rw [h i hc.1 m hc.2]
    · rfl
  sep j r r' h i hi m hm := by
    have e := congrFun (congrFun h i) m
    simp only [histKey, hi, hm, and_self, ↓reduceIte] at e
    exact Option.some.inj e

end Coins

/-! ## The coupling -/

section Coupling

variable {K M : ℕ} {X D : Type}

/-- **The agreement at one coordinate has the same chance whatever the first coin.** -/
theorem Pr_agree [Fintype X] [DecidableEq X] [Nonempty X] (k : ℕ) (c : Fin M → X) :
    Pr (fun b : Fin M → X => AgreeAt k c b) = δAt X M k := by
  have hswap : ∀ c c' : Fin M → X, Pr (AgreeAt k c) = Pr (AgreeAt k c') := by
    intro c c'
    let σ : (Fin M → X) ≃ (Fin M → X) := Equiv.piCongrRight fun m => Equiv.swap (c m) (c' m)
    rw [← Pr_equiv σ (AgreeAt k c')]
    refine Pr_congr fun b => forall_congr' fun m => imp_congr_right fun _ => ?_
    show c m = b m ↔ c' m = Equiv.swap (c m) (c' m) (b m)
    rw [eq_comm (a := c' m), Equiv.swap_apply_eq_iff, Equiv.swap_apply_right, eq_comm]
  unfold δAt
  rw [Pr_prod (fun a b => AgreeAt k a b)]
  exact (Ex_const (α := Fin M → X) _).symm.trans (congrArg Ex (funext fun a => hswap c a))

theorem prod_range_ite (c : ℝ) (k : ℕ) :
    ∀ N : ℕ, (∏ i ∈ Finset.range N, if i < k then (1 : ℝ) else c) = c ^ (N - min k N)
  | 0 => by simp
  | N + 1 => by
    rw [Finset.prod_range_succ, prod_range_ite c k N]
    by_cases h : N < k
    · rw [ite_eq_left h, mul_one, show N + 1 - min k (N + 1) = N - min k N by omega]
    · rw [ite_eq_right h, ← pow_succ, show N + 1 - min k (N + 1) = N - min k N + 1 by omega]

/-- **The agreement's chance in closed form**: two uniform coins agree on their first `k` entries with chance
`|X|^{-min k M}`. -/
theorem δAt_eq [Fintype X] [DecidableEq X] [Nonempty X] (k : ℕ) :
    δAt X M k = (Fintype.card X : ℝ)⁻¹ ^ min k M := by
  classical
  obtain ⟨x₀⟩ := ‹Nonempty X›
  rw [← Pr_agree k (fun _ : Fin M => x₀), Pr_eq_sum]
  have hX : (0 : ℝ) < Fintype.card X := by exact_mod_cast Fintype.card_pos
  have hsum : (∑ b : Fin M → X, if AgreeAt k (fun _ => x₀) b then (1 : ℝ) else 0) =
      (Fintype.card X : ℝ) ^ (M - min k M) := by
    have e : ∀ b : Fin M → X, (if AgreeAt k (fun _ => x₀) b then (1 : ℝ) else 0) =
        ∏ m : Fin M, if (m.val < k → x₀ = b m) then (1 : ℝ) else 0 := fun b => by
      rw [Finset.prod_boole]
      split_ifs with h1 h2 h2
      · rfl
      · exact absurd (fun m _ => h1 m) h2
      · exact absurd (fun m => h2 m (Finset.mem_univ m)) h1
      · rfl
    rw [Finset.sum_congr rfl fun b _ => e b,
      ← Fintype.prod_sum (fun (m : Fin M) (y : X) => if (m.val < k → x₀ = y) then (1 : ℝ) else 0)]
    have f : ∀ m : Fin M, (∑ y : X, if (m.val < k → x₀ = y) then (1 : ℝ) else 0) =
        if m.val < k then 1 else (Fintype.card X : ℝ) := fun m => by
      by_cases hm : m.val < k <;> simp [hm]
    rw [Finset.prod_congr rfl fun m _ => f m,
      Fin.prod_univ_eq_prod_range (fun i => if i < k then (1 : ℝ) else (Fintype.card X : ℝ)) M, prod_range_ite]
  rw [hsum, Fintype.card_fun, Fintype.card_fin]
  push_cast
  rw [show (Fintype.card X : ℝ) ^ M = (Fintype.card X : ℝ) ^ (M - min k M) * (Fintype.card X : ℝ) ^ min k M by
      rw [← pow_add, Nat.sub_add_cancel (min_le_right k M)],
    div_mul_cancel_left₀ (pow_ne_zero _ hX.ne'), inv_pow]

/-- **On agreement at round `t`, the forks at `t` and `t + 1` are the same coins**: they differ only at round `t`,
where the key reads only the agreeing entries and the merge takes only those from the live coin. -/
theorem fnC_fork_succ {n : Fin K → ℕ} {key : Fin K → (Fin K → Fin M → X) → D} (hk : PrefixKey n key)
    (f : Fin K → D → Fin M → X) (r r'' : Fin K → Fin M → X) (t : Fin K) (hag : AgreeAt (n t) (r t) (r'' t)) :
    fnC n key f (forkN t.val r r'') = fnC n key f (forkN (t.val + 1) r r'') := by
  have hA : ∀ i, AgreeAt (n i) (forkN t.val r r'' i) (forkN (t.val + 1) r r'' i) := by
    intro i m hm
    unfold forkN
    by_cases h1 : i.val < t.val
    · rw [ite_eq_left h1, ite_eq_left (show i.val < t.val + 1 by omega)]
    · by_cases h2 : i.val < t.val + 1
      · obtain rfl : i = t := Fin.ext (by omega)
        rw [ite_eq_right h1, ite_eq_left h2]
        exact (hag m hm).symm
      · rw [ite_eq_right h1, ite_eq_right h2]
  have hkey : ∀ j, key j (forkN t.val r r'') = key j (forkN (t.val + 1) r r'') :=
    fun j => hk.read j _ _ fun i _ => hA i
  funext j m
  show (if m.val < n j then forkN t.val r r'' j m else f j (key j (forkN t.val r r'')) m) =
    (if m.val < n j then forkN (t.val + 1) r r'' j m else f j (key j (forkN (t.val + 1) r r'')) m)
  rw [hkey j]
  split_ifs with hm
  · exact hA j m hm
  · rfl

/-- The merge of two forks is the fork of the two merges. -/
theorem mergeC_forkN (n : Fin K → ℕ) (t : ℕ) (u u' r r'' : Fin K → Fin M → X) :
    mergeC n (forkN t u u') (forkN t r r'') = forkN t (mergeC n u r) (mergeC n u' r'') := by
  funext i m
  unfold mergeC forkN
  by_cases h1 : i.val < t <;> by_cases h2 : m.val < n i <;> simp [h1, h2]

/-- **The fork's coins at round `t`, from the tape at two keys**: before round `t` the fork's keys are the first
branch's, from round `t` on they are `b`'s. -/
theorem fnC_forkN_eq {n : Fin K → ℕ} {key : Fin K → (Fin K → Fin M → X) → D} (hk : PrefixKey n key)
    (f : Fin K → D → Fin M → X) (r r'' : Fin K → Fin M → X) (t : ℕ) (b : Fin K → D)
    (hb : ∀ i : Fin K, t ≤ i.val → b i = key i (forkN t r r'')) :
    fnC n key f (forkN t r r'') =
      mergeC n (forkN t (fun i => f i (key i r)) (fun i => f i (b i))) (forkN t r r'') := by
  unfold fnC
  congr 1
  funext i
  by_cases h : i.val < t
  · have e : forkN t (fun i => f i (key i r)) (fun i => f i (b i)) i = f i (key i r) := ite_eq_left h
    rw [e]
    congr 1
    refine hk.read i _ _ fun i' hi' m _ => ?_
    show (if i'.val < t then r i' else r'' i') m = r i' m
    rw [ite_eq_left (show i'.val < t from lt_of_le_of_lt (Fin.le_def.1 hi') h)]
  · have e : forkN t (fun i => f i (key i r)) (fun i => f i (b i)) i = f i (b i) := ite_eq_right h
    rw [e, hb i (by omega)]

variable [Fintype X] [Nonempty X] [Fintype D] [DecidableEq D]

/-- **Off agreement, the simulated fork over the tape is the deployed fork's law at the live coins**: the keys differ
at every round from `t` on (`sep`), so the tape's values at the two keys are independent and uniform
(`Pr_evalAt₂`, with any other point before round `t`), and the second branch's coins are the fork of the two merges. -/
theorem Pr_disagree [_root_.Nontrivial D] {n : Fin K → ℕ} {key : Fin K → (Fin K → Fin M → X) → D}
    (hk : PrefixKey n key) (E : (Fin K → Fin M → X) → (Fin K → Fin M → X) → Prop) (t : Fin K)
    (r r'' : Fin K → Fin M → X) (hne : ¬ AgreeAt (n t) (r t) (r'' t)) :
    Pr (fun f : Fin K → D → Fin M → X => E (fnC n key f r) (fnC n key f (forkN t.val r r''))) =
      Pr (fun p : (Fin K → Fin M → X) × (Fin K → Fin M → X) =>
        E (mergeC n p.1 r) (forkN t.val (mergeC n p.1 r) (mergeC n p.2 r''))) := by
  classical
  obtain ⟨b, hb, hab⟩ : ∃ b : Fin K → D, (∀ i : Fin K, t.val ≤ i.val → b i = key i (forkN t.val r r'')) ∧
      ∀ i, b i ≠ key i r := by
    refine ⟨fun i => if t.val ≤ i.val then key i (forkN t.val r r'') else (exists_ne (key i r)).choose,
      fun i hi => ite_eq_left hi, fun i => ?_⟩
    dsimp only
    by_cases hi : t.val ≤ i.val
    · rw [ite_eq_left hi]
      intro heq
      have hA := hk.sep i _ _ heq t (Fin.le_def.2 hi)
      refine hne fun m hm => ?_
      have h' : forkN t.val r r'' t m = r t m := hA m hm
      rw [show forkN t.val r r'' t = r'' t from ite_eq_right (lt_irrefl _)] at h'
      exact h'.symm
    · rw [ite_eq_right hi]
      exact (exists_ne (key i r)).choose_spec
  have hG : ∀ f : Fin K → D → Fin M → X,
      E (fnC n key f r) (fnC n key f (forkN t.val r r'')) ↔
        E (mergeC n (fun i => f i (key i r)) r)
          (forkN t.val (mergeC n (fun i => f i (key i r)) r) (mergeC n (fun i => f i (b i)) r'')) := by
    intro f
    rw [fnC_forkN_eq hk f r r'' t.val b hb, mergeC_forkN]
    exact Iff.rfl
  refine (Pr_congr hG).trans ?_
  exact Pr_evalAt₂ (fun i => key i r) b hab
    (fun u u' => E (mergeC n u r) (forkN t.val (mergeC n u r) (mergeC n u' r'')))

/-- **The simulated fork at round `t`**: the two branches' coins at the tape `f`, the live coins `r` and the live
fork at `t` of `r` and `r''`. -/
noncomputable def simF (n : Fin K → ℕ) (key : Fin K → (Fin K → Fin M → X) → D)
    (E : (Fin K → Fin M → X) → (Fin K → Fin M → X) → Prop) (t : ℕ) : ℝ :=
  Pr fun p : (Fin K → D → Fin M → X) × (Fin K → Fin M → X) × (Fin K → Fin M → X) =>
    E (fnC n key p.1 p.2.1) (fnC n key p.1 (forkN t p.2.1 p.2.2))

/-- **The deployed fork at round `t`.** -/
noncomputable def depF (E : (Fin K → Fin M → X) → (Fin K → Fin M → X) → Prop) (t : ℕ) : ℝ :=
  Pr fun s : (Fin K → Fin M → X) × (Fin K → Fin M → X) => E s.1 (forkN t s.1 s.2)

/-- `((r, r''), (u, u')) ↦ ((u, r), (u', r''))`. -/
def shuffle (α : Type) : (α × α) × (α × α) ≃ (α × α) × (α × α) where
  toFun q := ((q.2.1, q.1.1), (q.2.2, q.1.2))
  invFun q := ((q.1.2, q.2.2), (q.1.1, q.2.1))
  left_inv _ := rfl
  right_inv _ := rfl

/-- **The deployed fork at the merges of uniform coins is the deployed fork**: averaged over the live coins, the
fork of the merges of uniform coins with them has the deployed fork's probability (`Pr_mergeC`, `Ex_mergeC`). -/
theorem Ex_fork_merge (n : Fin K → ℕ) (E : (Fin K → Fin M → X) → (Fin K → Fin M → X) → Prop) (t : ℕ) :
    (Ex fun rr : (Fin K → Fin M → X) × (Fin K → Fin M → X) =>
      Pr fun p : (Fin K → Fin M → X) × (Fin K → Fin M → X) =>
        E (mergeC n p.1 rr.1) (forkN t (mergeC n p.1 rr.1) (mergeC n p.2 rr.2))) = depF E t := by
  refine (Pr_prod (α := (Fin K → Fin M → X) × (Fin K → Fin M → X))
    (β := (Fin K → Fin M → X) × (Fin K → Fin M → X))
    (fun rr p => E (mergeC n p.1 rr.1) (forkN t (mergeC n p.1 rr.1) (mergeC n p.2 rr.2)))).symm.trans ?_
  refine (Pr_equiv (shuffle (Fin K → Fin M → X))
    (fun y => E (mergeC n y.1.1 y.1.2) (forkN t (mergeC n y.1.1 y.1.2) (mergeC n y.2.1 y.2.2)))).trans ?_
  rw [Pr_prod (α := (Fin K → Fin M → X) × (Fin K → Fin M → X))
    (β := (Fin K → Fin M → X) × (Fin K → Fin M → X))
    (fun y₁ y₂ => E (mergeC n y₁.1 y₁.2) (forkN t (mergeC n y₁.1 y₁.2) (mergeC n y₂.1 y₂.2)))]
  have hin : ∀ y₁ : (Fin K → Fin M → X) × (Fin K → Fin M → X),
      (Pr fun y₂ : (Fin K → Fin M → X) × (Fin K → Fin M → X) =>
        E (mergeC n y₁.1 y₁.2) (forkN t (mergeC n y₁.1 y₁.2) (mergeC n y₂.1 y₂.2))) =
      (fun s : Fin K → Fin M → X => Pr fun s' : Fin K → Fin M → X => E s (forkN t s s')) (mergeC n y₁.1 y₁.2) :=
    fun y₁ => Pr_mergeC n fun s' => E (mergeC n y₁.1 y₁.2) (forkN t (mergeC n y₁.1 y₁.2) s')
  rw [congrArg Ex (funext hin), Ex_mergeC n fun s => Pr fun s' : Fin K → Fin M → X => E s (forkN t s s')]
  exact (Pr_prod (α := Fin K → Fin M → X) (β := Fin K → Fin M → X) fun s s' => E s (forkN t s s')).symm

/-- **Off agreement the simulated fork is at most the deployed one**: at each pair of live coins that disagree at
round `t`, the tape's chance is the deployed fork's at those coins (`Pr_disagree`), whose average is the deployed
fork's (`Ex_fork_merge`). -/
theorem Pr_disagree_le [_root_.Nontrivial D] {n : Fin K → ℕ} {key : Fin K → (Fin K → Fin M → X) → D}
    (hk : PrefixKey n key) (E : (Fin K → Fin M → X) → (Fin K → Fin M → X) → Prop) (t : Fin K) :
    Pr (fun p : (Fin K → D → Fin M → X) × (Fin K → Fin M → X) × (Fin K → Fin M → X) =>
        E (fnC n key p.1 p.2.1) (fnC n key p.1 (forkN t.val p.2.1 p.2.2)) ∧
          ¬ AgreeAt (n t) (p.2.1 t) (p.2.2 t)) ≤ depF E t.val := by
  classical
  refine (Pr_equiv (Equiv.prodComm ((Fin K → Fin M → X) × (Fin K → Fin M → X)) (Fin K → D → Fin M → X))
    (fun p : (Fin K → D → Fin M → X) × (Fin K → Fin M → X) × (Fin K → Fin M → X) =>
      E (fnC n key p.1 p.2.1) (fnC n key p.1 (forkN t.val p.2.1 p.2.2)) ∧
        ¬ AgreeAt (n t) (p.2.1 t) (p.2.2 t))).symm.le.trans ?_
  refine (Pr_prod (α := (Fin K → Fin M → X) × (Fin K → Fin M → X)) (β := Fin K → D → Fin M → X)
    (fun rr f => E (fnC n key f rr.1) (fnC n key f (forkN t.val rr.1 rr.2)) ∧
      ¬ AgreeAt (n t) (rr.1 t) (rr.2 t))).le.trans ?_
  refine le_of_le_of_eq ?_ (Ex_fork_merge n E t.val)
  refine Ex_mono fun rr => ?_
  by_cases hag : AgreeAt (n t) (rr.1 t) (rr.2 t)
  · calc _ ≤ Pr (fun _ : Fin K → D → Fin M → X => False) := Pr_mono fun _ hf => hf.2 hag
      _ = 0 := Pr_false
      _ ≤ _ := Pr_nonneg _
  · rw [Pr_congr fun f => and_iff_left hag]
    exact (Pr_disagree hk E t rr.1 rr.2 hag).le

/-- **Round `t`'s coin from the rest and its own**: `(funSplitAt t _).symm (b, rest)`. -/
def glue (t : Fin K) (rest : {i // i ≠ t} → Fin M → X) (b : Fin M → X) : Fin K → Fin M → X :=
  (Equiv.funSplitAt t (Fin M → X)).symm (b, rest)

omit [Fintype X] [Nonempty X] [Fintype D] [DecidableEq D] in
theorem glue_at (t : Fin K) (rest : {i // i ≠ t} → Fin M → X) (b : Fin M → X) : glue t rest b t = b := by
  simp [glue]

omit [Fintype X] [Nonempty X] [Fintype D] [DecidableEq D] in
/-- The fork at `t + 1` does not read the second coins' round `t`. -/
theorem forkN_succ_glue (t : Fin K) (r : Fin K → Fin M → X) (rest : {i // i ≠ t} → Fin M → X) (b b' : Fin M → X) :
    forkN (t.val + 1) r (glue t rest b) = forkN (t.val + 1) r (glue t rest b') := by
  funext i
  unfold forkN
  split_ifs with h
  · rfl
  · have hi : i ≠ t := fun e => h (by rw [e]; omega)
    simp [glue, hi]

/-- The tape, the first coins, the second coins but round `t`'s, and round `t`'s: `(a, rest, b) ↦ (a.1, a.2, glue)`. -/
def splitΨ (t : Fin K) :
    ((Fin K → D → Fin M → X) × (Fin K → Fin M → X)) × ({i // i ≠ t} → Fin M → X) × (Fin M → X) ≃
      (Fin K → D → Fin M → X) × (Fin K → Fin M → X) × (Fin K → Fin M → X) :=
  ((Equiv.refl _).prodCongr ((Equiv.prodComm _ _).trans (Equiv.funSplitAt t (Fin M → X)).symm)).trans
    (Equiv.prodAssoc _ _ _)

omit [Fintype X] [Nonempty X] [Fintype D] [DecidableEq D] in
theorem splitΨ_apply (t : Fin K)
    (q : ((Fin K → D → Fin M → X) × (Fin K → Fin M → X)) × ({i // i ≠ t} → Fin M → X) × (Fin M → X)) :
    splitΨ t q = (q.1.1, q.1.2, glue t q.2.1 q.2.2) := rfl

/-- **On agreement the simulated fork is `δ_t` times the next one**: there the fork at `t` is the fork at `t + 1`
(`fnC_fork_succ`), which does not read the redrawn coin at round `t` (`forkN_succ_glue`), and agreement reads only
that coin and the first branch's, with chance `δ_t` whatever the latter (`Pr_agree`, `Pr_indep`). -/
theorem Pr_and_agree [DecidableEq X] {n : Fin K → ℕ} {key : Fin K → (Fin K → Fin M → X) → D}
    (hk : PrefixKey n key) (E : (Fin K → Fin M → X) → (Fin K → Fin M → X) → Prop) (t : Fin K) :
    Pr (fun p : (Fin K → D → Fin M → X) × (Fin K → Fin M → X) × (Fin K → Fin M → X) =>
        E (fnC n key p.1 p.2.1) (fnC n key p.1 (forkN t.val p.2.1 p.2.2)) ∧
          AgreeAt (n t) (p.2.1 t) (p.2.2 t)) = δAt X M (n t) * simF n key E (t.val + 1) := by
  classical
  let pE : (Fin K → D → Fin M → X) × (Fin K → Fin M → X) → ({i // i ≠ t} → Fin M → X) → Prop := fun a rest =>
    E (fnC n key a.1 a.2) (fnC n key a.1 (forkN (t.val + 1) a.2 (glue t rest (a.2 t))))
  have hsucc : simF n key E (t.val + 1) = Ex fun a => Pr (pE a) := by
    unfold simF
    rw [← Pr_equiv (splitΨ t)]
    calc Pr (fun q : ((Fin K → D → Fin M → X) × (Fin K → Fin M → X)) × ({i // i ≠ t} → Fin M → X) ×
            (Fin M → X) =>
          E (fnC n key (splitΨ t q).1 (splitΨ t q).2.1)
            (fnC n key (splitΨ t q).1 (forkN (t.val + 1) (splitΨ t q).2.1 (splitΨ t q).2.2)))
        = Pr (fun q : ((Fin K → D → Fin M → X) × (Fin K → Fin M → X)) × ({i // i ≠ t} → Fin M → X) ×
            (Fin M → X) => pE q.1 q.2.1) :=
          Pr_congr fun q => by
            rw [splitΨ_apply]
            show _ ↔ E (fnC n key q.1.1 q.1.2) (fnC n key q.1.1 (forkN (t.val + 1) q.1.2 (glue t q.2.1 (q.1.2 t))))
            rw [forkN_succ_glue t q.1.2 q.2.1 q.2.2 (q.1.2 t)]
      _ = Pr (fun q : ((Fin K → D → Fin M → X) × (Fin K → Fin M → X)) × ({i // i ≠ t} → Fin M → X) =>
            pE q.1 q.2) := Pr_drop pE
      _ = Ex fun a => Pr (pE a) := Pr_prod pE
  rw [← Pr_equiv (splitΨ t)]
  calc Pr (fun q : ((Fin K → D → Fin M → X) × (Fin K → Fin M → X)) × ({i // i ≠ t} → Fin M → X) ×
          (Fin M → X) =>
        E (fnC n key (splitΨ t q).1 (splitΨ t q).2.1)
            (fnC n key (splitΨ t q).1 (forkN t.val (splitΨ t q).2.1 (splitΨ t q).2.2)) ∧
          AgreeAt (n t) ((splitΨ t q).2.1 t) ((splitΨ t q).2.2 t))
      = Pr (fun q : ((Fin K → D → Fin M → X) × (Fin K → Fin M → X)) × ({i // i ≠ t} → Fin M → X) ×
          (Fin M → X) => pE q.1 q.2.1 ∧ AgreeAt (n t) (q.1.2 t) q.2.2) :=
        Pr_congr fun q => by
          rw [splitΨ_apply]
          show E (fnC n key q.1.1 q.1.2) (fnC n key q.1.1 (forkN t.val q.1.2 (glue t q.2.1 q.2.2))) ∧
              AgreeAt (n t) (q.1.2 t) (glue t q.2.1 q.2.2 t) ↔
            E (fnC n key q.1.1 q.1.2) (fnC n key q.1.1 (forkN (t.val + 1) q.1.2 (glue t q.2.1 (q.1.2 t)))) ∧
              AgreeAt (n t) (q.1.2 t) q.2.2
          rw [glue_at]
          refine and_congr_left fun hag => ?_
          rw [fnC_fork_succ hk q.1.1 q.1.2 (glue t q.2.1 q.2.2) t (by rwa [glue_at]),
            forkN_succ_glue t q.1.2 q.2.1 q.2.2 (q.1.2 t)]
    _ = Ex fun a => Pr (pE a) * Pr (fun b : Fin M → X => AgreeAt (n t) (a.2 t) b) :=
        Pr_indep pE fun a b => AgreeAt (n t) (a.2 t) b
    _ = Ex fun a => Pr (pE a) * δAt X M (n t) :=
        congrArg Ex (funext fun a => by rw [Pr_agree])
    _ = δAt X M (n t) * simF n key E (t.val + 1) := by
        rw [Ex_mul_const, hsucc, mul_comm]

/-- **The coupling's step**: the simulated fork at round `t` is at most the deployed fork there and `δ_t` times the
simulated fork at `t + 1`. -/
theorem simF_step [DecidableEq X] [_root_.Nontrivial D] {n : Fin K → ℕ}
    {key : Fin K → (Fin K → Fin M → X) → D} (hk : PrefixKey n key)
    (E : (Fin K → Fin M → X) → (Fin K → Fin M → X) → Prop) (t : Fin K) :
    simF n key E t.val ≤ depF E t.val + δAt X M (n t) * simF n key E (t.val + 1) := by
  rw [show simF n key E t.val = _ from
      Pr_split (fun p : (Fin K → D → Fin M → X) × (Fin K → Fin M → X) × (Fin K → Fin M → X) =>
        E (fnC n key p.1 p.2.1) (fnC n key p.1 (forkN t.val p.2.1 p.2.2)))
        (fun p => AgreeAt (n t) (p.2.1 t) (p.2.2 t)),
    Pr_and_agree hk E t]
  linarith [Pr_disagree_le hk E t]

/-- **Past the last round the simulated fork is one run**: the fork at `K` is the first branch, of uniform law
(`Pr_fnC`). -/
theorem simF_last {n : Fin K → ℕ} {key : Fin K → (Fin K → Fin M → X) → D}
    (E : (Fin K → Fin M → X) → (Fin K → Fin M → X) → Prop) :
    simF n key E K = Pr fun s : Fin K → Fin M → X => E s s := by
  have hK : ∀ r r'' : Fin K → Fin M → X, forkN K r r'' = r := fun r r'' => funext fun i => by
    unfold forkN
    rw [ite_eq_left i.isLt]
  unfold simF
  simp only [hK]
  exact (Pr_drop fun f r => E (fnC n key f r) (fnC n key f r)).trans (Pr_fnC n key fun s => E s s)

/-- **The fork's coin coupling at `fnC`**: at a key that reads the revealed history, the simulated forks are at most
the deployed fork at each round and `δ_t` times the next simulated fork, and past the last round are the one run. -/
theorem fnC_coupling [DecidableEq X] [_root_.Nontrivial D] {n : Fin K → ℕ}
    {key : Fin K → (Fin K → Fin M → X) → D} (hk : PrefixKey n key)
    (E : (Fin K → Fin M → X) → (Fin K → Fin M → X) → Prop) :
    (∀ t : Fin K, simF n key E t.val ≤ depF E t.val + δAt X M (n t) * simF n key E (t.val + 1)) ∧
      simF n key E K = Pr fun s : Fin K → Fin M → X => E s s :=
  ⟨simF_step hk E, simF_last E⟩

omit [Nonempty X] [Fintype D] [DecidableEq D] in
/-- **The chances by round**: `δ_t` at the rounds `t < K`, and `0` past them. -/
noncomputable def δN (X : Type) [Fintype X] (M : ℕ) (n : Fin K → ℕ) (t : ℕ) : ℝ :=
  if ht : t < K then δAt X M (n ⟨t, ht⟩) else 0

omit [Nonempty X] [Fintype D] [DecidableEq D] in
theorem δN_nonneg (n : Fin K → ℕ) (t : ℕ) : 0 ≤ δN X M n t := by
  unfold δN
  split_ifs
  · exact Pr_nonneg _
  · exact le_rfl

/-- **The fork's coin coupling at `fnAt`, in `ForkCoinCoupling`'s shape**: with the live coins indexed by `Fin L`,
`L = K`, the simulated forks `simForkPr`-style at the live fork `forkN` are at most the deployed fork
`Claims.forkCoins ⟨t, _⟩` at each round `t < K` and `δN X M n t` times the next, and past the last round at most any
event `O` the diagonal implies. At `E := pairEv sha512 V P x ω` and `O := oneEv sha512 V P x ω` (whose diagonal step
is `TiedColl (l ++ l) → TiedColl l`), the two clauses are `ForkCoinCoupling sha512 V P x (δN X M n) sim ω` once
`forkPrAt` is unfolded at `t < V.k x` and at `V.k x`. -/
theorem fnAt_coupling [DecidableEq X] [_root_.Nontrivial D] {L : ℕ} (h : L = K) {n : Fin K → ℕ}
    {key : Fin K → (Fin K → Fin M → X) → D} (hk : PrefixKey n key)
    (E : (Fin K → Fin M → X) → (Fin K → Fin M → X) → Prop) (O : (Fin K → Fin M → X) → Prop)
    (hEO : ∀ s, E s s → O s) :
    (∀ t (ht : t < K),
      Pr (fun p : (Fin K → D → Fin M → X) × (Fin L → Fin M → X) × (Fin L → Fin M → X) =>
          E (fnAt h n key p.1 p.2.1) (fnAt h n key p.1 (forkN t p.2.1 p.2.2))) ≤
        Pr (fun s : (Fin K → Fin M → X) × (Fin K → Fin M → X) => E s.1 (Claims.forkCoins ⟨t, ht⟩ s.1 s.2)) +
          δN X M n t * Pr (fun p : (Fin K → D → Fin M → X) × (Fin L → Fin M → X) × (Fin L → Fin M → X) =>
            E (fnAt h n key p.1 p.2.1) (fnAt h n key p.1 (forkN (t + 1) p.2.1 p.2.2)))) ∧
      Pr (fun p : (Fin K → D → Fin M → X) × (Fin L → Fin M → X) × (Fin L → Fin M → X) =>
          E (fnAt h n key p.1 p.2.1) (fnAt h n key p.1 (forkN K p.2.1 p.2.2))) ≤ Pr O := by
  subst h
  obtain ⟨hstep, hlast⟩ := fnC_coupling hk E
  refine ⟨fun t ht => ?_, ?_⟩
  · have := hstep ⟨t, ht⟩
    unfold δN
    rw [dite_eq_left ht]
    exact this
  · have := hlast
    unfold simF at this
    exact this.trans_le (Pr_mono fun s hs => hEO s hs)

end Coupling

end Claims.Flock.FnFork
