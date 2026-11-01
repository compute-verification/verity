import Definitions.Pouw.Assumptions.PearlC.TTOutW1
import Definitions.Pouw.PearlC.W1Stop
import Proofs.Pouw.Prob

/-!
# A cheap winning W1 program refutes TT_OUT at W1, and W1 charges only the branch a run takes

`TTOutW1` (`Definitions.Pouw.Assumptions.PearlC.TTOutW1`), `Gγ` and `GγPerCoin` bound every adversary by its cost and
its oracle calls. W1 itself bounds a program's text (`W1.CodeImage.bounded`: at every layout `2^64` code nodes and `2^64`
constants), and nothing else bounds an adversary. These lemmas make that exact, as attack tools for
the TT family restated at `W1.w1Priced`.

- `exec_ite_untaken`: an `ite` runs one branch, so a run is the same whatever its untaken branch is. That branch may be
  any `Code` (in a program, any within the text bound) at no cost: W1 charges only the path a run takes.
- `exec_tree`: a complete tree of `ite`s of depth `n` (`tree`, `2^n` leaves) costs `n` INT32 instructions and the one
  leaf the memory reaches (`pathOf`), whatever the other leaves are. At `n = 256` on the salt's bits, that would be a
  table of one leaf per salt, selected for 256 INT32.
- `size_tree`, `tree_not_code`: a tree of depth `n` has at least `2^n` nodes, so no program's code is a tree of depth
  above 64. The table `tree leaf 256 []` is not a W1 program.
- `not_ttOutW1_of_win`, `not_gγ_of_win`, `not_gγPerCoin_of_win`: one adversary within `T` and `q ≤ 2^64` calls that
  wins with probability above the error refutes each form. They are the forms read at that adversary.
- `schema_not_ttOutW1` (and `schema_not_gγ`, `schema_not_gγPerCoin`): the forms as schemas carry no guard of their own.
  For a protocol whose reference a program reproduces at no cost (`tabP`, whose words are the zeros a fresh memory
  holds), the empty program wins on every coin, so the schema fails at it. What makes a named form true is its
  protocol's reference being expensive to produce, which the form does not state.

What these do not do: refute a named form at the deployed protocol. At `NoiseSha512.sha512Sem` with the game's salt the
reference reads no oracle. Without the text bound a program could tabulate it per salt in `tree` at depth 256 and pay
for one path (`exec_tree`). The rest of that program (the salt's bits, the word tables, the leaf's writes) and its
transcript are arithmetic. Under
the bound it is not a program (`tree_not_code`).

**What the bound leaves, an argument and not a theorem here.**
- A run reads precomputed data only as words of memory: the constant bank (at most `2^64` words), and the preprocessed
  state, the activations and the weights at the addresses its instructions name. An address is fixed in the code (an
  instruction names at most a fragment's spans), or it is `(rg, w32 (M r) + off)` for an `ld` with `rg` and `off`
  compiled in, at most `2^32` addresses per `ld` node. So all of a program's runs together read below `2^97` distinct
  words of data.
- The code can have far more paths than nodes, since an `ite`'s continuation follows both branches. But a path holds
  no data of its own beyond those words and its instructions' few immediates, so the count is of words, not paths.
- A table that answers a salt holds that salt's reference words, which differ from salt to salt as SHA-512's outputs
  do. So a program that reads its answers from a table has them for at most about `2^97` of the `2^256` salts. Its
  success is then at most about `2^-159`, below `εPearlC`'s `2^-128` per unit (W1-RT estimates about `2^-150`).

"Tabulates" isn't defined in Lean, and this counting bound isn't proved. Whatever else a bounded program can do, such
as compute the reference more cheaply than the honest program, is TT_OUT's own conjecture.
-/

namespace Pouw.PearlC.W1Tabulate

open Pouw Pouw.PearlC Pouw.PearlC.W1 Pouw.PearlC.Assumptions

variable {Q R S : Type}

/-! ## W1 charges only the branch a run takes -/

/-- **A run doesn't read an `ite`'s untaken branch.** When the word is zero the run takes `b`, so replacing the `then`
branch `a` by any `a'` leaves the run, its memory, fetches and cost, unchanged. As a `Code`, `a'` may be of any size;
a program's is within W1's text bound. -/
theorem exec_ite_untaken (dv : W1Device) (t : Bool) (p : Addr) (a a' b k : W1.Code) (st : St)
    (h : w32 (st.mem p) = 0) :
    exec dv t (.ite p a b k) st = exec dv t (.ite p a' b k) st := by
  simp [exec, h]

/-- A complete binary tree of `ite`s of depth `n` on the words at `(10, n − 1)`, …, `(10, 0)`, with `leaf path` at the
end of each of its `2^n` paths (`path` lists the branches taken, nonzero as `true`, the last one first). -/
def tree (leaf : List Bool → W1.Code) : ℕ → List Bool → W1.Code
  | 0, path => leaf path
  | n + 1, path => .ite (10, n) (tree leaf n (true :: path)) (tree leaf n (false :: path)) .done

/-- The leaf a memory reaches in `tree leaf n path`. -/
def pathOf (M : Mem) : ℕ → List Bool → List Bool
  | 0, path => path
  | n + 1, path => pathOf M n (decide (w32 (M (10, n)) ≠ 0) :: path)

/-- **A run through a tree of `ite`s of depth `n` pays `n` INT32 instructions and its one leaf**, whatever the other
`2^n − 1` leaves are. The branches fetch nothing (region 10 is not pre-salt) and leave memory as it was, and the leaf
runs under the taint rule. -/
theorem exec_tree (dv : W1Device) (leaf : List Bool → W1.Code) :
    ∀ (n : ℕ) (path : List Bool) (st : St),
      exec dv true (tree leaf n path) st =
        exec dv true (leaf (pathOf st.mem n path)) ⟨st.mem, st.fetched, st.cost + n * dv.prices.int32⟩
  | 0, path, st => by cases st; simp [tree, pathOf]
  | n + 1, path, st => by
      have hf : ([((10 : ℕ), n)] : List Addr).filter preSalt = [] := by simp [preSalt]
      have hc : st.cost + dv.prices.int32 + (n : ℚ) * dv.prices.int32 =
          st.cost + ((n + 1 : ℕ) : ℚ) * dv.prices.int32 := by
        push_cast; ring
      by_cases h : w32 (st.mem (10, n)) = 0
      · have e : exec dv true (tree leaf (n + 1) path) st =
            exec dv true (tree leaf n (false :: path)) ⟨st.mem, st.fetched, st.cost + dv.prices.int32⟩ := by
          simp [tree, exec, h, hf]
        have hp : pathOf st.mem (n + 1) path = pathOf st.mem n (false :: path) := by simp [pathOf, h]
        rw [e, exec_tree dv leaf n]
        try dsimp only
        rw [hc, hp]
      · have e : exec dv true (tree leaf (n + 1) path) st =
            exec dv true (tree leaf n (true :: path)) ⟨st.mem, st.fetched, st.cost + dv.prices.int32⟩ := by
          simp [tree, exec, h, hf]
        have hp : pathOf st.mem (n + 1) path = pathOf st.mem n (true :: path) := by simp [pathOf, h]
        rw [e, exec_tree dv leaf n]
        try dsimp only
        rw [hc, hp]

/-- A tree run from an untainted start runs as from a tainted one: its root is an `ite`, which taints both branches,
and nothing follows it. -/
theorem exec_tree_untainted (dv : W1Device) (leaf : List Bool → W1.Code) (n : ℕ) (path : List Bool) (st : St) :
    exec dv false (tree leaf (n + 1) path) st = exec dv true (tree leaf (n + 1) path) st := by
  simp [tree, exec]

/-! ## W1's text bound leaves the tree out (F2) -/

/-- Every code tree has a node. -/
theorem one_le_codeSize (c : W1.Code) : 1 ≤ c.size := by
  cases c <;> simp only [W1.Code.size] <;> omega

/-- **A tree of `ite`s of depth `n` has at least `2^n` nodes**: each of its `2^n` paths ends in a leaf of at least one
node. -/
theorem size_tree (leaf : List Bool → W1.Code) : ∀ (n : ℕ) (path : List Bool), 2 ^ n ≤ (tree leaf n path).size
  | 0, path => by simpa [tree] using one_le_codeSize (leaf path)
  | n + 1, path => by
      have h1 := size_tree leaf n (true :: path)
      have h2 := size_tree leaf n (false :: path)
      have h3 : 2 ^ (n + 1) = 2 ^ n + 2 ^ n := by ring
      simp only [tree, W1.Code.size]
      omega

/-- **No W1 program's code is a tree of `ite`s of depth above 64**, at any layout and whatever the tree's leaves: the
tree has at least `2^n` nodes (`size_tree`), and W1's text bound (`CodeImage.bounded`) allows `2^64`. The table of one
leaf per salt, `tree leaf 256 []` on the salt's 256 bits, is one of them. -/
theorem tree_not_code (leaf : List Bool → W1.Code) {n : ℕ} (hn : 64 < n) (path : List Bool) (pr : CodeImage)
    (L : Layout) : pr.code L ≠ tree leaf n path := by
  intro h
  have h1 := (pr.bounded L).1
  rw [h] at h1
  have h2 : 2 ^ 65 ≤ 2 ^ n := Nat.pow_le_pow_right (by norm_num) (by omega)
  exact absurd (h2.trans ((size_tree leaf n path).trans h1)) (by norm_num)

/-! ## A cheap winning program refutes each form -/

/-- **TT_OUT at W1 fails at a winning adversary.** One adversary within `T` and `q ≤ 2^64` calls whose correct units'
credit exceeds `T / (1 − γ₀)` with probability above `ε q N` refutes `TTOutW1`. -/
theorem not_ttOutW1_of_win [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] {dv : W1Device}
    {salt : S → List ℕ} {P : Protocol Q R S} {D : Layout → Prop} {γ₀ : ℝ} {ε : ℕ → ℕ → ℝ}
    (U : Workload) (hU : U.InDomain P D) (T : ℝ) (hT : 0 ≤ T) (q : ℕ) (hq : q ≤ 2 ^ 64)
    (A : Adversary (W1.w1Priced dv salt : CostModel Q R S)) (hA : A.Bounded U T q)
    (hwin : ε q U.layout.N < pr (fun x : (Q → R) × S =>
      T / (1 - γ₀) < ∑ u ∈ correctSet P U x.1 x.2 (A.transcript U x.1 x.2),
        P.credit x.1 x.2 U u (actOf (A.transcript U x.1 x.2) u))) :
    ¬ TTOutW1 dv salt P D γ₀ ε := fun h =>
  absurd (h U hU T hT q hq A hA) (not_le.mpr hwin)

/-- **The γ-gap property fails at a winning adversary**, over any cost model (`W1.w1Priced` among them). -/
theorem not_gγ_of_win [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] {CM : CostModel Q R S}
    {P : Protocol Q R S} {D : Layout → Prop} {γ : ℝ} {η : ℕ → ℕ → ℝ}
    (U : Workload) (hU : U.InDomain P D) (T : ℝ) (hT : 0 ≤ T) (q : ℕ) (hq : q ≤ 2 ^ 64)
    (A : Adversary CM) (hA : A.Bounded U T q)
    (hwin : η q U.layout.N < pr (fun x : (Q → R) × S => Wins P U γ T (A.transcript U x.1 x.2) x.1 x.2)) :
    ¬ Gγ CM P D γ η := fun h =>
  absurd (h U hU T hT q hq A hA) (not_le.mpr hwin)

/-- **The γ-gap property per coin fails at a winning adversary**, over any cost model (the compute audit's
`W1.w1Priced` among them). -/
theorem not_gγPerCoin_of_win [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] {CM : CostModel Q R S}
    {P : Protocol Q R S} {D : Layout → Prop} {γ : ℝ} {η : ℕ → ℕ → ℝ}
    (U : Workload) (hU : U.InDomain P D) (T : ℝ) (hT : 0 ≤ T) (q : ℕ) (hq : q ≤ 2 ^ 64)
    (A : Adversary CM)
    (hcalls : ∀ (H : Q → R) (s : S), A.pre.calls H + CM.calls A.online H U A.act (A.pre.eval H) s ≤ q)
    (hwin : η q U.layout.N < pr (fun x : (Q → R) × S =>
      CM.cost A.online x.1 U A.act (A.pre.eval x.1) x.2 ≤ T ∧ Wins P U γ T (A.transcript U x.1 x.2) x.1 x.2)) :
    ¬ GγPerCoin CM P D γ η := fun h =>
  absurd (h U hU T hT q hq A hcalls) (not_le.mpr hwin)

/-! ## The forms as schemas carry no guard -/

/-- One unit of shape `1 × 1 × 1`, weight 0, all zero. -/
def tabU : Workload := ⟨⟨1, fun _ => ⟨1, 1, 1⟩, fun _ => 0⟩, fun _ _ _ => 0⟩

/-- A protocol whose reference words are `0`, the words a fresh memory holds, whose rules and cap always pass, and
which credits each unit `1` and charges it `1` of honest work. -/
def tabP : Protocol Unit Unit Unit where
  ticket := fun _ _ _ _ _ _ _ => Fp8Atom.wordQ 0
  useful := fun _ _ _ _ _ _ _ => Fp8Atom.wordQ 0
  actOK := fun _ _ => True
  weightOK := fun _ _ _ => True
  credit := fun _ _ _ _ _ => 1
  capOK := fun _ _ _ _ _ => True
  Wref := fun _ _ => 1

/-- The empty program: one node of text, within W1's bound. -/
def emptyProg : CodeImage := ⟨fun _ => [], fun _ => .done, fun _ => by norm_num [W1.Code.size]⟩

/-- The adversary that runs the empty program on zero activations, with no preprocessing. -/
def emptyAdv (dv : W1Device) : Adversary (W1.w1Priced dv (fun _ : Unit => []) : CostModel Unit Unit Unit) :=
  ⟨.done [], emptyProg, fun _ _ _ => 0⟩

theorem emptyAdv_cost (dv : W1Device) (H : Unit → Unit) (s : Unit) :
    (W1.w1Priced dv (fun _ : Unit => []) : CostModel Unit Unit Unit).cost (emptyAdv dv).online H tabU
      (emptyAdv dv).act ((emptyAdv dv).pre.eval H) s = 0 := by
  simp [W1.w1Priced, w1Read, final, emptyAdv, emptyProg, exec, QTree.eval]

theorem emptyAdv_bounded (dv : W1Device) : (emptyAdv dv).Bounded tabU 0 0 := by
  intro H s
  refine ⟨(emptyAdv_cost dv H s).le, ?_⟩
  simp [emptyAdv, W1.w1Priced, QTree.calls]

/-- Unit 0 of the empty program's transcript is correct, on every coin. -/
theorem emptyAdv_mem (dv : W1Device) (H : Unit → Unit) (s : Unit) :
    0 ∈ correctSet tabP tabU H s ((emptyAdv dv).transcript tabU H s) := by
  simp [correctSet, Correct, Adversary.transcript, W1.w1Priced, final, readout, emptyAdv, emptyProg, exec, init,
    tabP, tabU, ticketAt, usefulAt, QTree.eval]

/-- The empty program's correct units are unit 0 alone, on every coin. -/
theorem emptyAdv_correctSet (dv : W1Device) (H : Unit → Unit) (s : Unit) :
    correctSet tabP tabU H s ((emptyAdv dv).transcript tabU H s) = {0} := by
  ext u
  rw [Finset.mem_singleton]
  constructor
  · intro hu
    simp only [correctSet, Finset.mem_filter, Finset.mem_range] at hu
    have h1 : u < 1 := hu.1
    omega
  · rintro rfl
    exact emptyAdv_mem dv H s

theorem tabU_inDomain : tabU.InDomain tabP (fun _ => True) := by
  simp [Workload.InDomain, tabU]

/-- **TT_OUT at W1, as a schema, fails**: at `tabP` the empty program wins on every coin at cost `0`, so at every `γ₀`
no error with `ε 0 1 < 1` (`εPearlC 0 1 = 2^-128` among them) makes `TTOutW1` hold. -/
theorem schema_not_ttOutW1 (dv : W1Device) (γ₀ : ℝ) {ε : ℕ → ℕ → ℝ} (hε : ε 0 1 < 1) :
    ¬ TTOutW1 dv (fun _ : Unit => []) tabP (fun _ => True) γ₀ ε := by
  refine not_ttOutW1_of_win tabU tabU_inDomain 0 le_rfl 0 (by norm_num) (emptyAdv dv) (emptyAdv_bounded dv) ?_
  rw [pr_eq_one]
  · exact hε
  · intro x
    simp only [emptyAdv_correctSet]
    simp [tabP]

/-- **The γ-gap property at W1, as a schema, fails** at `tabP`, at every `γ` and error with `η 0 1 < 1`. -/
theorem schema_not_gγ (dv : W1Device) (γ : ℝ) {η : ℕ → ℕ → ℝ} (hη : η 0 1 < 1) :
    ¬ Gγ (W1.w1Priced dv (fun _ : Unit => []) : CostModel Unit Unit Unit) tabP (fun _ => True) γ η := by
  refine not_gγ_of_win tabU tabU_inDomain 0 le_rfl 0 (by norm_num) (emptyAdv dv) (emptyAdv_bounded dv) ?_
  rw [pr_eq_one]
  · exact hη
  · intro x
    simp only [Wins, emptyAdv_correctSet]
    simp [tabP]

/-- **The γ-gap property per coin at W1, as a schema, fails** at `tabP`, at every `γ` and error with `η 0 1 < 1`. -/
theorem schema_not_gγPerCoin (dv : W1Device) (γ : ℝ) {η : ℕ → ℕ → ℝ} (hη : η 0 1 < 1) :
    ¬ GγPerCoin (W1.w1Priced dv (fun _ : Unit => []) : CostModel Unit Unit Unit) tabP (fun _ => True) γ η := by
  refine not_gγPerCoin_of_win tabU tabU_inDomain 0 le_rfl 0 (by norm_num) (emptyAdv dv) ?_ ?_
  · intro H s
    simp [emptyAdv, W1.w1Priced, QTree.calls]
  · rw [pr_eq_one]
    · exact hη
    · intro x
      refine ⟨(emptyAdv_cost dv x.1 x.2).le, ?_⟩
      simp only [Wins, emptyAdv_correctSet]
      simp [tabP]

end Pouw.PearlC.W1Tabulate
