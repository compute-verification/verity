import Proofs.Flock.Claims.KS.Radius
import Proofs.Flock.Claims.Bridge.Transport
import Proofs.Flock.Soundness.ExtractAcc

/-!
# One wait's bound at the filled word, from a fixed table's bound

The old theorems' extraction step (`Extract.joint_acc_le`) is stated over a game; a verifier's runs at one tape are one
(`runGame`: one coin node, the round coins), so it applies at the deployed verifier itself, with no simulation:

* **The run's law** (`expect_runGame`, `expectN_runGame`): `k` independent runs at one tape are `expectN` of `runGame`;
  so `accPr` and `accFail` are its `expect` and `expectN` (`accPr_eq`, `accFail_eq`), and one wait's left side,
  `ε · Q/ε^k`, is the product form's `ε · expectAcc(¬G)` (`oneWait_lhs_eq`).
* **The table** (`opθ`, `firstTable_opθ`): an accepted run's opening at a position of the word is its first opened
  column there (`lookCol`), a rejected run's none; on accepted reruns the extractor's table (`firstTable`) is the filled
  word (`fillWord`, `fillWord_eq_ofFn`).
* **The two obligations** at the verifier:
  * `TableBoundθ`: at every statement and prover, escapes that fit the weight with one more fork event, such that at
    every tape and every table at the word's positions that does not extract (`GoodTable`), a run accepts with
    probability at most `ε_table + escapes(ω)` beyond that of accepting and opening a position off the table. This is
    the old theorems' fixed-table bound (`table_sound_compiled_of_table`: `Pr[accepted ∧ T uncommitted] ≤ ε_c⁻ +
    Pr[off T]`) at the deployed verifier.
  * `ConflictTied`: two accepted runs at one tape that open one position of the word to two columns collide among the
    verifier's calls.
* **One wait** (`oneWaitθ_of_tableBound`): `joint_acc_le` at `runGame`, with the conflict term an escape at round `0`
  (two independent runs, `forkCoins_zero`) of weight `k` (`Escapes.addFork`), gives `OneWaitθ` when the word is no
  longer than `N₀`. At the deployed records (`plain_oneWaitθ_of_tableBound`, `zk_oneWaitθ_of_tableBound`) the word is
  `N₀` long, and KS at both instances asks only the two obligations (`plain_KSθ_of_tableBound`,
  `zk_KSθ_of_tableBound`).
-/

namespace Claims.Flock

open FlockSoundness FlockSoundness.Game FlockSoundness.Extract FlockSoundness.Rewinding

/-! ## A verifier's runs as a game -/

section RunGame

variable {Coin : Type} [Fintype Coin] [Nonempty Coin]

/-- **A run's law as a game**: one coin node, the `n` round coins. -/
def runGame (n : ℕ) : Game (Fin n → Coin) := .coin (Fin n → Coin) .ret

/-- The game's one strategy. -/
def runS (n : ℕ) : Strategy (runGame (Coin := Coin) n) := fun _ => PUnit.unit

theorem expect_runGame (n : ℕ) (f : (Fin n → Coin) → ℝ) : expect f (runGame n) (runS n) = Ex f := rfl

/-- **`k` independent runs** are `expectN` of `runGame`, first run first. -/
theorem expectN_runGame (n : ℕ) : ∀ (k : ℕ) (F : List (Fin n → Coin) → ℝ),
    expectN (runGame n) (runS n) k F = Ex fun σ : Fin k → Fin n → Coin => F ((List.finRange k).map σ)
  | 0, F => by
    simp only [expectN, List.finRange_zero, List.map_nil]
    exact (Ex_const _).symm
  | k + 1, F => by
    rw [expectN, expect_runGame, Claims.Bridge.Ex_cons]
    refine congrArg Ex (funext fun c => ?_)
    rw [expectN_runGame n k]
    refine congrArg Ex (funext fun r => ?_)
    simp [List.finRange_succ, List.map_cons, List.map_map, Function.comp_def]

open Classical in
theorem prod_map_ind {α : Type} (p : α → Prop) :
    ∀ l : List α, (l.map (ind p)).prod = if ∀ a ∈ l, p a then 1 else 0
  | [] => by simp
  | a :: l => by
    rw [List.map_cons, List.prod_cons, prod_map_ind p l]
    unfold ind
    by_cases ha : p a <;> by_cases hl : ∀ b ∈ l, p b <;> simp [ha, hl]

end RunGame

section Laws

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- A run at tape `ω` accepts. -/
def Acc (H : Msg → Msg) (V : Verifier Coin) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (r : Fin (V.k x) → Coin) : Prop :=
  (run H P V x ω r).out = .acc

/-- The views of runs at tape `ω`. -/
def seenOf (H : Msg → Msg) (V : Verifier Coin) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (bs : List (Fin (V.k x) → Coin)) : List Seen :=
  bs.map fun r => (run H P V x ω r).seen

theorem accPr_eq (H : Msg → Msg) (V : Verifier Coin) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) :
    accPr H V P x ω = expect (ind (Acc H V P x ω)) (runGame (V.k x)) (runS _) := by
  rw [expect_runGame]
  exact Claims.Bridge.Pr_eq_Ex _

/-- **`k` accepted runs that fail `G`**, on the run's law. -/
theorem accFail_eq (H : Msg → Msg) (V : Verifier Coin) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (k : ℕ)
    (G : List Seen → Prop) :
    accFail H V P x ω k G = expectN (runGame (V.k x)) (runS _) k fun bs =>
      (bs.map (ind (Acc H V P x ω))).prod * ind (fun bs => ¬ G (seenOf H V P x ω bs)) bs := by
  classical
  rw [expectN_runGame]
  unfold accFail
  rw [Claims.Bridge.Pr_eq_Ex]
  refine congrArg Ex (funext fun σ => ?_)
  rw [prod_map_ind]
  have hall : (∀ a ∈ (List.finRange k).map σ, Acc H V P x ω a) ↔ ∀ i, (run H P V x ω (σ i)).out = .acc := by
    simp [Acc]
  have hseen : seenOf H V P x ω ((List.finRange k).map σ) =
      (List.finRange k).map fun i => (run H P V x ω (σ i)).seen := by
    simp [seenOf, List.map_map, Function.comp_def]
  unfold ind
  beta_reduce
  rw [hseen]
  by_cases h1 : ∀ i, (run H P V x ω (σ i)).out = .acc <;>
    by_cases h2 : G ((List.finRange k).map fun i => (run H P V x ω (σ i)).seen) <;> simp [Acc, h1, h2]

/-- **One wait's left side is the product form's**: `ε · Q/ε^k = ε · expectAcc(¬G)` on the run's law. -/
theorem oneWait_lhs_eq (H : Msg → Msg) (V : Verifier Coin) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (k : ℕ)
    (G : List Seen → Prop) :
    accPr H V P x ω * (accFail H V P x ω k G / accPr H V P x ω ^ k) =
      expect (ind (Acc H V P x ω)) (runGame (V.k x)) (runS _) *
        expectAcc (runGame (V.k x)) (runS _) (Acc H V P x ω) k
          (ind fun bs => ¬ G (seenOf H V P x ω bs)) := by
  rw [accFail_eq, accPr_eq]
  rfl

end Laws

/-! ## The table the accepted reruns fill -/

section Table

variable {Col T : Type} [DecidableEq Col]

/-- The column a list of openings opens at `q`: its first there. -/
def lookCol (os : List (ℕ × Col)) (q : ℕ) : Option Col := (os.find? fun p => decide (p.1 = q)).map Prod.snd

omit [DecidableEq Col] in
theorem fillAt_eq_lookCol (z : Col) (os : List (ℕ × Col)) (q : ℕ) : fillAt z os q = (lookCol os q).getD z := rfl

omit [DecidableEq Col] in
theorem lookCol_append (os os' : List (ℕ × Col)) (q : ℕ) :
    lookCol (os ++ os') q = (lookCol os q).or (lookCol os' q) := by
  unfold lookCol
  rw [List.find?_append]
  cases os.find? fun p => decide (p.1 = q) <;> rfl

omit [DecidableEq Col] in
theorem fillWord_eq_ofFn (z : Col) (N : ℕ) (os : List (ℕ × Col)) :
    fillWord z N os = List.ofFn fun p : Fin N => fillAt z os p := by
  apply List.ext_getElem
  · simp [fillWord]
  · intro i h₁ h₂
    simp [fillWord]

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

open Classical in
/-- **An accepted run's openings** at the word's positions, its first column at each; none on a rejected run. -/
noncomputable def opθ (H : Msg → Msg) (D : Deploy Coin Col T) (Rd : Radius Col T) {Ω : Type} (P : Prover Ω)
    (x : Msg) (ω : Ω) (r : Fin (D.V.k x) → Coin) (p : Fin (Rd.N x)) : Option Col :=
  if (run H P D.V x ω r).out = .acc then lookCol (D.cols (run H P D.V x ω r).seen) p.val else none

omit [DecidableEq Col] [Fintype Coin] [Nonempty Coin] in
theorem findSome?_lookCol {β : Type} (f : β → Option Col) (cs : β → List (ℕ × Col)) (q : ℕ) :
    ∀ bs : List β, (∀ b ∈ bs, f b = lookCol (cs b) q) → bs.findSome? f = lookCol (bs.flatMap cs) q
  | [], _ => rfl
  | b :: bs, h => by
    rw [List.flatMap_cons, lookCol_append, List.findSome?_cons, h b List.mem_cons_self,
      findSome?_lookCol f cs q bs fun b' hb' => h b' (List.mem_cons_of_mem _ hb')]
    cases lookCol (cs b) q <;> rfl

omit [DecidableEq Col] [Fintype Coin] [Nonempty Coin] in
/-- **On accepted reruns the extractor's table is the filled word.** -/
theorem firstTable_opθ (H : Msg → Msg) (D : Deploy Coin Col T) (Rd : Radius Col T) {Ω : Type} (P : Prover Ω)
    (x : Msg) (ω : Ω) (bs : List (Fin (D.V.k x) → Coin)) (hacc : ∀ b ∈ bs, Acc H D.V P x ω b) (p : Fin (Rd.N x)) :
    firstTable (opθ H D Rd P x ω) (fun _ => Rd.zero x) bs p =
      fillAt (Rd.zero x) ((seenOf H D.V P x ω bs).flatMap D.cols) p.val := by
  unfold firstTable
  rw [fillAt_eq_lookCol, seenOf, List.flatMap_map,
    findSome?_lookCol _ (fun r => D.cols (run H P D.V x ω r).seen) p.val bs fun b hb => by
      unfold opθ
      simp only [show (run H P D.V x ω b).out = .acc from hacc b hb, ↓reduceIte]]

/-- **A table at the word's positions extracts**: it is within `θ·N` of the encoding of a trace of the word's length
that reads as a witness of the parsed statement. -/
def GoodTable (H : Msg → Msg) (D : Deploy Coin Col T) (Rd : Radius Col T) (x : Msg) (C : Fin (Rd.N x) → Col) : Prop :=
  ∃ (M : T) (s : Stmt), D.parse x = some s ∧ RelS H s (D.read M) ∧ Rd.N x = Rd.len M ∧
    (hdist (List.ofFn C) (D.enc M) : ℝ) ≤ Rd.θ M * Rd.len M

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem extractsFill_of_good (H : Msg → Msg) (D : Deploy Coin Col T) (Rd : Radius Col T) (x : Msg)
    (vs : List Seen) (h : GoodTable H D Rd x fun p => fillAt (Rd.zero x) (vs.flatMap D.cols) p.val) :
    ExtractsFill H D Rd x vs := by
  obtain ⟨M, s, hs, hrel, hN, hd⟩ := h
  exact ⟨M, s, hs, hrel, hN, by rwa [fillWord_eq_ofFn]⟩

end Table

/-! ## Escapes with the conflict fork -/

section Fork

variable {Ω Coin : Type} [Inhabited Coin] {H : Msg → Msg} {V : Verifier Coin} {P : Prover Ω} {x : Msg}

/-- **Escapes with one more fork event**: `e`'s, and `ev` at round `j₀` with weight `c`. -/
def _root_.Claims.Escapes.addFork (e : Escapes H V P x) (c : ℝ) (hc : 0 ≤ c) (j₀ : Fin (V.k x))
    (ev : Ω → (Fin (V.k x) → Coin) → (Fin (V.k x) → Coin) → Prop)
    (tied : ∀ ω r r', ev ω r r' →
      TiedColl H ((run H P V x ω r).vLog ++ (run H P V x ω (forkCoins j₀ r r')).vLog)) : Escapes H V P x where
  I := Option e.I
  c i := i.elim c e.c
  j i := i.elim j₀ e.j
  ev i := i.elim ev e.ev
  L := e.L
  β := e.β
  ev₁ := e.ev₁
  c_nonneg i := by cases i; exacts [hc, e.c_nonneg _]
  β_nonneg := e.β_nonneg
  tied i := by cases i; exacts [tied, e.tied _]
  tied₁ := e.tied₁

/-- **The escapes fit `C` with one more fork event** of weight `c` at round `0`: at most `mult` events at one round
counting it, `mult · (c + Σ_i c_i) ≤ 2C`, and `Σ_l β_l ≤ k(x)`. -/
def _root_.Claims.Escapes.FitsFork (e : Escapes H V P x) (C c : ℝ) : Prop :=
  ∃ mult : ℕ, (∀ r : Fin (V.k x), (Finset.univ.filter fun i => e.j i = r).card + (if r.val = 0 then 1 else 0) ≤ mult) ∧
    mult * (c + ∑ i, e.c i) ≤ 2 * C ∧ ∑ l, e.β l ≤ V.k x

omit [Inhabited Coin] in
theorem sum_option_any {α M : Type} [AddCommMonoid M] [Fintype α] (inst : Fintype (Option α))
    (f : Option α → M) : @Finset.sum _ _ _ (@Finset.univ _ inst) f = f none + ∑ a, f (some a) := by
  rw [Subsingleton.elim inst instFintypeOption]
  exact Fintype.sum_option f

omit [Inhabited Coin] in
theorem card_filter_option {α β : Type} [Fintype α] [DecidableEq β] (inst : Fintype (Option α))
    (g : Option α → β) (r : β) :
    ((@Finset.univ _ inst).filter fun i => g i = r).card =
      (if g none = r then 1 else 0) + (Finset.univ.filter fun a => g (some a) = r).card := by
  rw [Subsingleton.elim inst instFintypeOption, Finset.card_filter, Fintype.sum_option, Finset.card_filter]

variable [Fintype Coin]

theorem _root_.Claims.Escapes.term_addFork (e : Escapes H V P x) (c : ℝ) (hc : 0 ≤ c) (j₀ : Fin (V.k x))
    (ev : Ω → (Fin (V.k x) → Coin) → (Fin (V.k x) → Coin) → Prop) (tied) (ω : Ω) :
    (e.addFork c hc j₀ ev tied).term ω =
      Real.sqrt (c * Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => ev ω s.1 s.2) + e.term ω := by
  have h1 : (∑ i : (e.addFork c hc j₀ ev tied).I, Real.sqrt ((e.addFork c hc j₀ ev tied).c i *
        Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => (e.addFork c hc j₀ ev tied).ev i ω s.1 s.2)) =
      Real.sqrt (c * Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => ev ω s.1 s.2) +
        ∑ i, Real.sqrt (e.c i * Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => e.ev i ω s.1 s.2) :=
    sum_option_any _ fun i : Option e.I => Real.sqrt (i.elim c e.c *
      Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => (i.elim ev e.ev) ω s.1 s.2)
  unfold Escapes.term
  rw [h1, add_assoc]
  rfl

omit [Fintype Coin] in
theorem _root_.Claims.Escapes.fits_addFork (e : Escapes H V P x) (C c : ℝ) (hc : 0 ≤ c) (j₀ : Fin (V.k x)) (hj₀ : j₀.val = 0)
    (ev : Ω → (Fin (V.k x) → Coin) → (Fin (V.k x) → Coin) → Prop) (tied) (h : e.FitsFork C c) :
    (e.addFork c hc j₀ ev tied).Fits C := by
  classical
  obtain ⟨mult, hm, hmc, hβ⟩ := h
  refine ⟨mult, fun r => ?_, ?_, hβ⟩
  · refine le_trans (le_of_eq ?_) (hm r)
    refine (card_filter_option _ (fun i : Option e.I => i.elim j₀ e.j) r).trans ?_
    by_cases h0 : r.val = 0
    · rw [ite_eq_left (show (none : Option e.I).elim j₀ e.j = r from Fin.ext (hj₀.trans h0.symm)),
        ite_eq_left h0]
      exact Nat.add_comm _ _
    · rw [ite_eq_right (show ¬ (none : Option e.I).elim j₀ e.j = r from fun h => h0 (by
          rw [← (h : j₀ = r)]; exact hj₀)), ite_eq_right h0]
      exact (Nat.zero_add _).trans (Nat.add_zero _).symm
  · show mult * ∑ i : Option e.I, i.elim c e.c ≤ 2 * C
    rw [sum_option_any]
    exact hmc

end Fork

/-! ## One wait, from the table's bound -/

section OneWait

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

/-- **The fixed-table bound at the filled word**, the old theorems' `table_sound_compiled_of_table` at the deployed
verifier: at every statement and prover, escapes that fit the weight with one more fork event of weight `k`, such that
at every tape and every table at the word's positions that does not extract, a run accepts with probability at most
`ε_table + escapes(ω)` beyond that of accepting and opening a position of the word off the table. -/
def TableBoundθ (H : Msg → Msg) (D : Deploy Coin Col T) (Rd : Radius Col T) : Prop :=
  ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω), ∃ e : Escapes H D.V P x,
    e.FitsFork (D.weight x) (D.k x) ∧ ∀ (ω : Ω) (C : Fin (Rd.N x) → Col), ¬ GoodTable H D Rd x C →
      accPr H D.V P x ω ≤ D.εtable x + e.term ω +
        Pr fun r : Fin (D.V.k x) → Coin => (run H P D.V x ω r).out = .acc ∧
          ∃ (p : Fin (Rd.N x)) (u : Col), lookCol (D.cols (run H P D.V x ω r).seen) p.val = some u ∧ u ≠ C p

/-- **Two accepted runs at one tape open each position of the word to one column**, or the verifier's calls in the two
runs collide. -/
def ConflictTied (H : Msg → Msg) (D : Deploy Coin Col T) (Rd : Radius Col T) : Prop :=
  ∀ (x : Msg) (Ω : Type) (P : Prover Ω) (ω : Ω) (r r' : Fin (D.V.k x) → Coin) (q : ℕ) (u v : Col),
    (run H P D.V x ω r).out = .acc → (run H P D.V x ω r').out = .acc → q < Rd.N x →
    lookCol (D.cols (run H P D.V x ω r).seen) q = some u → lookCol (D.cols (run H P D.V x ω r').seen) q = some v →
    u ≠ v → TiedColl H ((run H P D.V x ω r).vLog ++ (run H P D.V x ω r').vLog)

/-- **Flock's one-wait bound at the filled word, from the table's bound**: when the word is no longer than `N₀`,
`ε_table ≥ 0`, the verifier takes a round, and `k ≥ 1` wherever it accepts some run. -/
theorem oneWaitθ_of_tableBound (H : Msg → Msg) (D : Deploy Coin Col T) (Rd : Radius Col T)
    (hN : ∀ x, (Rd.N x : ℝ) ≤ D.N₀ x) (hεt : ∀ x, 0 ≤ D.εtable x) (hVk : ∀ x, 0 < D.V.k x)
    (hk : ∀ x, 1 ≤ D.k x ∨ NoAcc H D.V x) (hconf : ConflictTied H D Rd) (htab : TableBoundθ H D Rd) :
    OneWaitθ H D Rd := by
  classical
  intro x Ω _ _ P
  obtain ⟨e₀, hfit, hb⟩ := htab x Ω P
  set j₀ : Fin (D.V.k x) := ⟨0, hVk x⟩
  set ev : Ω → (Fin (D.V.k x) → Coin) → (Fin (D.V.k x) → Coin) → Prop :=
    fun ω r r' => ∃ p, ConflictAt (opθ H D Rd P x ω) (r, r') p with hev
  have tied : ∀ ω r r', ev ω r r' →
      TiedColl H ((run H P D.V x ω r).vLog ++ (run H P D.V x ω (forkCoins j₀ r r')).vLog) := by
    intro ω r r' ⟨p, h1, h2, h3⟩
    rw [Claims.Bridge.forkCoins_zero j₀ rfl]
    simp only [opθ] at h1 h2 h3
    split_ifs at h1 h2 h3 with ha ha'
    · obtain ⟨u, hu⟩ := Option.isSome_iff_exists.1 h1
      obtain ⟨v, hv⟩ := Option.isSome_iff_exists.1 h2
      exact hconf x Ω P ω r r' p.val u v ha ha' p.isLt hu hv fun huv => h3 (by rw [hu, hv, huv])
    all_goals simp at h1 h2
  refine ⟨e₀.addFork (D.k x) (Nat.cast_nonneg _) j₀ ev tied,
    e₀.fits_addFork (D.weight x) (D.k x) (Nat.cast_nonneg _) j₀ rfl ev tied hfit, fun ω hε => ?_⟩
  have hk1 : 1 ≤ D.k x := (hk x).resolve_right fun hno => by
    have : accPr H D.V P x ω = 0 := by
      unfold accPr
      rw [Pr_congr (q := fun _ => False) fun r => ⟨fun h => hno Ω P ω r h, False.elim⟩, Pr_false]
    linarith
  set Xt : List (Fin (D.V.k x) → Coin) → Prop := fun bs =>
    (∀ b ∈ bs, Acc H D.V P x ω b) → ExtractsFill H D Rd x (seenOf H D.V P x ω bs) with hXt
  have hXb : ∀ bs, ¬ Xt bs → expect (ind (Acc H D.V P x ω)) (runGame (D.V.k x)) (runS _) ≤
      (D.εtable x + e₀.term ω) + expect (ind fun r => Acc H D.V P x ω r ∧
        ∃ p, Off (opθ H D Rd P x ω) (firstTable (opθ H D Rd P x ω) (fun _ => Rd.zero x) bs) r p)
        (runGame (D.V.k x)) (runS _) := by
    intro bs hnX
    simp only [hXt, Classical.not_imp] at hnX
    obtain ⟨hall, hnot⟩ := hnX
    have hgood : ¬ GoodTable H D Rd x (firstTable (opθ H D Rd P x ω) (fun _ => Rd.zero x) bs) := by
      intro hg
      refine hnot (extractsFill_of_good H D Rd x _ ?_)
      have : firstTable (opθ H D Rd P x ω) (fun _ => Rd.zero x) bs =
          fun p => fillAt (Rd.zero x) ((seenOf H D.V P x ω bs).flatMap D.cols) p.val :=
        funext (firstTable_opθ H D Rd P x ω bs hall)
      rwa [this] at hg
    rw [← accPr_eq, expect_runGame, ← Claims.Bridge.Pr_eq_Ex]
    refine (hb ω _ hgood).trans (le_of_eq ?_)
    congr 1
    refine Pr_congr fun r => ?_
    unfold Acc Off opθ
    constructor
    · rintro ⟨ha, p, u, hu, hne⟩
      refine ⟨ha, p, ?_⟩
      rw [ite_eq_left ha, hu]
      exact ⟨rfl, fun h => hne (Option.some_injective _ h)⟩
    · rintro ⟨ha, p, h1, h2⟩
      rw [ite_eq_left ha] at h1 h2
      obtain ⟨u, hu⟩ := Option.isSome_iff_exists.1 h1
      exact ⟨ha, p, u, hu, fun h => h2 (by rw [hu, h])⟩
  have hεc : 0 ≤ D.εtable x + e₀.term ω := add_nonneg (hεt x) (e₀.term_nonneg ω)
  have hj := joint_acc_le (opθ H D Rd P x ω) (fun _ => Rd.zero x) (runGame (D.V.k x)) (runS _)
    (Acc H D.V P x ω) Xt _ hεc hXb (D.k x) hk1
  have hlhs : accPr H D.V P x ω * (accFail H D.V P x ω (D.k x) (ExtractsFill H D Rd x) /
        accPr H D.V P x ω ^ D.k x) =
      expect (ind (Acc H D.V P x ω)) (runGame (D.V.k x)) (runS _) *
        expectAcc (runGame (D.V.k x)) (runS _) (Acc H D.V P x ω) (D.k x) (ind fun bs => ¬ Xt bs) := by
    rw [oneWait_lhs_eq]
    congr 1
    unfold expectAcc
    congr 1
    refine congrArg (expectN _ _ _) (funext fun bs => ?_)
    rw [prod_map_ind]
    by_cases hall : ∀ b ∈ bs, Acc H D.V P x ω b
    · congr 1
      unfold ind
      have : (¬ Xt bs) ↔ ¬ ExtractsFill H D Rd x (seenOf H D.V P x ω bs) := by
        rw [hXt]
        exact not_congr ⟨fun h => h hall, fun h _ => h⟩
      beta_reduce
      split_ifs with h1 h2 h2 <;> first | rfl | (exfalso; tauto)
    · rw [ite_eq_right hall, zero_mul, zero_mul]
  have hE₂ : expect₂ (fun a a' => ind (fun y : (Fin (D.V.k x) → Coin) × (Fin (D.V.k x) → Coin) =>
        ∃ p, ConflictAt (opθ H D Rd P x ω) y p) (a, a')) (runGame (D.V.k x)) (runS _) =
      Pr fun s : (Fin (D.V.k x) → Coin) × (Fin (D.V.k x) → Coin) => ev ω s.1 s.2 := by
    unfold expect₂
    simp only [expect_runGame]
    rw [Claims.Bridge.Pr_eq_Ex, Claims.Bridge.Ex_prod]
  rw [hE₂, ← accPr_eq] at hj
  rw [hlhs, ← accPr_eq, Escapes.term_addFork]
  have hrate : (Fintype.card (Fin (Rd.N x)) : ℝ) * accPr H D.V P x ω / (Real.exp 1 * D.k x) ≤
      D.N₀ x / (Real.exp 1 * D.k x) * accPr H D.V P x ω := by
    rw [Fintype.card_fin, div_mul_eq_mul_div]
    exact div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_right (hN x) hε.le) (by positivity)
  linarith

end OneWait

end Claims.Flock

/-! ## At the deployed records -/

namespace FlockDeploy.KS

open Flock

theorem wordLen_eq_N₀ (zk : Bool) (x : Claims.Msg) : (wordLen zk x : ℝ) = N₀ zk x := by
  unfold wordLen N₀
  cases (tables zk x).head? with
  | none => simp
  | some p => obtain ⟨_, _⟩ := p; simp

theorem V_k_pos (zk : Bool) (x : Claims.Msg) : 0 < (V zk).k x := by
  simp only [V]
  split <;> omega

theorem k_one_plain (rel : Readable.Parsed → Readable.Relation) (x : Claims.Msg) :
    1 ≤ (deployPlain rel).k x ∨ Claims.Flock.NoAcc sha512 (deployPlain rel).V x :=
  (k_range_plain rel x).imp_left fun h => by omega

theorem k_one_zk (rel : Readable.Parsed → Readable.Relation) (x : Claims.Msg) :
    1 ≤ (deployZk rel).k x ∨ Claims.Flock.NoAcc sha512 (deployZk rel).V x :=
  (k_range_zk rel x).imp_left fun h => by omega

/-- **The deployed plain verifier: one wait's bound at the filled word from the table's bound**, at SHA-512. -/
theorem plain_oneWaitθ_of_tableBound (rel : Readable.Parsed → Readable.Relation)
    (hconf : Claims.Flock.ConflictTied sha512 (deployPlain rel) plainRadius)
    (htab : Claims.Flock.TableBoundθ sha512 (deployPlain rel) plainRadius) :
    Claims.Flock.OneWaitθ sha512 (deployPlain rel) plainRadius :=
  Claims.Flock.oneWaitθ_of_tableBound sha512 (deployPlain rel) plainRadius (fun x => (wordLen_eq_N₀ false x).le)
    (fun x => εtable_nonneg false x) (V_k_pos false) (k_one_plain rel) hconf htab

/-- **The deployed masked (`--zk`) verifier: one wait's bound at the filled word from the table's bound**, at
SHA-512. -/
theorem zk_oneWaitθ_of_tableBound (rel : Readable.Parsed → Readable.Relation)
    (hconf : Claims.Flock.ConflictTied sha512 (deployZk rel) zkRadius)
    (htab : Claims.Flock.TableBoundθ sha512 (deployZk rel) zkRadius) :
    Claims.Flock.OneWaitθ sha512 (deployZk rel) zkRadius :=
  Claims.Flock.oneWaitθ_of_tableBound sha512 (deployZk rel) zkRadius (fun x => (wordLen_eq_N₀ true x).le)
    (fun x => εtable_nonneg true x) (V_k_pos true) (k_one_zk rel) hconf htab

/-- **The deployed plain verifier: KS at the filled word from the table's bound**, at SHA-512. -/
theorem plain_KSθ_of_tableBound (rel : Readable.Parsed → Readable.Relation)
    (hconf : Claims.Flock.ConflictTied sha512 (deployPlain rel) plainRadius)
    (htab : Claims.Flock.TableBoundθ sha512 (deployPlain rel) plainRadius) :
    Claims.Flock.KSθ sha512 (deployPlain rel) plainRadius :=
  plain_of_oneWaitθ_sha512 rel (plain_oneWaitθ_of_tableBound rel hconf htab)

/-- **The deployed masked (`--zk`) verifier: KS at the filled word from the table's bound**, at SHA-512. -/
theorem zk_KSθ_of_tableBound (rel : Readable.Parsed → Readable.Relation)
    (hconf : Claims.Flock.ConflictTied sha512 (deployZk rel) zkRadius)
    (htab : Claims.Flock.TableBoundθ sha512 (deployZk rel) zkRadius) :
    Claims.Flock.KSθ sha512 (deployZk rel) zkRadius :=
  zk_of_oneWaitθ_sha512 rel (zk_oneWaitθ_of_tableBound rel hconf htab)

end FlockDeploy.KS
