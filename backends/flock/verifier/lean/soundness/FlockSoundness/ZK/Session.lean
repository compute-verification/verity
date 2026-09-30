import FlockSoundness.ZK.RealLeaves

/-!
# Zero knowledge of one session of `J` masked tables

The paper's Lemmas A and B are per session, with `J` tables (`docs/zk-proof-public.md` §4.2, §4.3), and the Lean ones of
`SHVZK` and `RealLeaves` are per table. The step between them is a product:

* the session's tables use disjoint randomness (§2.2: table `j` reads its own ranges of the session's two ChaCha20 streams,
  `purpose | j << 32`, and its own salt trees), which the PRG step makes independent uniform draws per table;
* no claim or message reads two tables' witnesses or masks, so at fixed coins the session's view is the tuple of its
  tables' views, each a function of its own table's randomness. The tables' witnesses may overlap (two drawn units in
  different tables can read one row); they are fixed.

So the session's statements follow from the tables' by two facts about independent components: equal distributions give
equal distributions of the tuples (`SameDist.pi`), and closeness adds up (`prCoin_pi_close`, one table at a time).

* `Session.session_shvzk`: Lemma B in world `W₁` for the session: its view and the tuple of `S_shvzk`'s are equally
  distributed.
* `Session.session_prefinal_indep`: Lemma A at a non-degenerate coin vector, for the session.
* `Session.session_shvzk_hm96`: Lemma B with real hm96 leaves, within `Σ_j 2·|Hid_j|·δ₁`.
-/

namespace FlockSoundness.ZK

open Finset Game
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **Independent components.** Tuples of independent components have equal distributions when each component's do. -/
theorem SameDist.pi {J : ℕ} {Ω Ω' V : Fin J → Type} [∀ j, Fintype (Ω j)] [∀ j, Fintype (Ω' j)]
    (f : ∀ j, Ω j → V j) (g : ∀ j, Ω' j → V j) (h : ∀ j, SameDist (f j) (g j)) :
    SameDist (fun ω : ∀ j, Ω j => fun j => f j (ω j)) (fun σ : ∀ j, Ω' j => fun j => g j (σ j)) := by
  sorry

/-- **Closeness adds up over independent components**: if for every `j` and every event, component `j`'s probability under
`f j` is at most its probability under `g j` plus `ε j`, then for every event on the tuple, its probability under the
`f`s is at most its probability under the `g`s plus `Σ_j ε j`. One component at a time, with the others part of the
experiment. -/
theorem prCoin_pi_close {J : ℕ} {Ω Ω' V : Fin J → Type} [∀ j, Fintype (Ω j)] [∀ j, Nonempty (Ω j)]
    [∀ j, Fintype (Ω' j)] [∀ j, Nonempty (Ω' j)] (f : ∀ j, Ω j → V j) (g : ∀ j, Ω' j → V j) (ε : Fin J → ℝ≥0∞)
    (h : ∀ j (E : V j → Prop), prCoin (fun ω => E (f j ω)) ≤ prCoin (fun σ => E (g j σ)) + ε j)
    (E : (∀ j, V j) → Prop) :
    prCoin (fun ω : ∀ j, Ω j => E fun j => f j (ω j)) ≤ prCoin (fun σ : ∀ j, Ω' j => E fun j => g j (σ j)) + ∑ j, ε j := by
  sorry

namespace Session

variable {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] {J : ℕ} (T : Fin J → Table F K W Salt Seed D Hi VL)

variable [CharP F 2]

/-- **The session's view** at fixed coins: each table's view, on its witness `w j` and its own randomness `ω j`. -/
noncomputable def view (w : Fin J → W) (ω : ∀ j, (T j).Rand) : ∀ j, (T j).View := fun j => (T j).view (w j) (ω j)

/-- **`S_shvzk` for the session**: each table's simulator, on its dummy witness `w₀ j` and its own randomness. -/
noncomputable def sim (w0 : Fin J → W) (σ : ∀ j, (T j).SimRand) : ∀ j, (T j).View := fun j => (T j).sim (w0 j) (σ j)

/-- The session's pre-final view in world `W₀`. -/
noncomputable def preView (w : Fin J → W) (ω : ∀ j, (T j).Rand) : ∀ j, (T j).PreView :=
  fun j => (T j).preView (w j) (ω j)

variable [AddCommGroup Salt] [AddCommGroup Seed] [AddCommGroup Hi] [Fintype F] [DecidableEq F] [Fintype K]
  [DecidableEq K] [Fintype Salt] [DecidableEq Salt] [Fintype Seed] [DecidableEq Seed] [Fintype Hi] [DecidableEq Hi]

/-- **Lemma B in world `W₁` for a session of `J` tables** (§4.3): at fixed coins that are non-degenerate for every table,
with each table's witness satisfying H_reg and `InnerHolds`, the session's view and the tuple of the tables' `S_shvzk`
outputs have the same distribution. -/
theorem session_shvzk (hrank : ∀ j, (T j).Rank) (hpad : ∀ j, (T j).PadOnto)
    (hW : ∀ j r, Function.Bijective fun p : F × F => (T j).Wr r p.1 p.2) (hpads : ∀ j, (T j).PadsOnto)
    (hβ : ∀ j r, (T j).β r ≠ 0) (w : Fin J → W) (hreg : ∀ j, (T j).sReg (w j) = (T j).pubReg)
    (hinner : ∀ j, (T j).InnerHolds (w j)) (w0 : Fin J → W) :
    SameDist (view T w) (sim T w0) := by
  sorry

/-- **Lemma A at a non-degenerate coin vector, for a session of `J` tables** (§4.2): the session's pre-final views on two
tuples of witnesses carrying the public region data are equally distributed. -/
theorem session_prefinal_indep (hrank : ∀ j, (T j).Rank)
    (hW : ∀ j r, Function.Bijective fun p : F × F => (T j).Wr r p.1 p.2) (hβ : ∀ j r, (T j).β r ≠ 0)
    (w w' : Fin J → W) (hreg : ∀ j, (T j).sReg (w j) = (T j).pubReg) (hreg' : ∀ j, (T j).sReg (w' j) = (T j).pubReg) :
    SameDist (preView T w) (preView T w') := by
  sorry

end Session

namespace Session

variable {F K W Salt B D Hi VL : Type} [Field F] [Field K] [AddCommGroup B] {J : ℕ}
  (T : Fin J → Table F K W Salt (B × Salt) D Hi VL) [CharP F 2]
  [AddCommGroup Salt] [AddCommGroup Hi] [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K]
  [Fintype Salt] [DecidableEq Salt] [Fintype B] [DecidableEq B] [Fintype Hi] [DecidableEq Hi]

/-- **Lemma B with real hm96 leaves for a session of `J` tables** (§4.3): under each table's hypotheses of
`table_shvzk_hm96`, every event on the session's view has probabilities within `Σ_j 2·|Hid_j|·δ₁` of each other under the
real prover (every table's hidden leaves real hm96 leaves of what they commit) and under the tuple of the tables'
`S_shvzk` with real leaves, in both directions. `|Hid_j|` counts table `j`'s unopened leaves, so the sum is at most the
paper's `2·δ₁·N_hid` with `N_hid` summed over the session's tables. -/
theorem session_shvzk_hm96 {C : Type} (M : ∀ j, (T j).Hm96 C) {δ₁ : ℝ≥0∞}
    (hT1 : ∀ j, Assumptions.Hm96Hiding (M j).My (M j).cs δ₁) (hrank : ∀ j, (T j).Rank) (hpad : ∀ j, (T j).PadOnto)
    (hW : ∀ j r, Function.Bijective fun p : F × F => (T j).Wr r p.1 p.2) (hpads : ∀ j, (T j).PadsOnto)
    (hβ : ∀ j r, (T j).β r ≠ 0) (w : Fin J → W) (hreg : ∀ j, (T j).sReg (w j) = (T j).pubReg)
    (hinner : ∀ j, (T j).InnerHolds (w j)) (w0 : Fin J → W) (E : (∀ j, (T j).View) → Prop) :
    prCoin (fun p : ∀ j, (T j).Rand × ((T j).Hid → Salt) => E fun j => (M j).viewR (w j) (p j).1 (p j).2)
        ≤ prCoin (fun p : ∀ j, (T j).SimRand × ((T j).Hid → Salt) => E fun j => (M j).simR (w0 j) (p j).1 (p j).2)
          + ∑ j, 2 * Fintype.card (T j).Hid * δ₁
      ∧ prCoin (fun p : ∀ j, (T j).SimRand × ((T j).Hid → Salt) => E fun j => (M j).simR (w0 j) (p j).1 (p j).2)
        ≤ prCoin (fun p : ∀ j, (T j).Rand × ((T j).Hid → Salt) => E fun j => (M j).viewR (w j) (p j).1 (p j).2)
          + ∑ j, 2 * Fintype.card (T j).Hid * δ₁ := by
  sorry

end Session

end FlockSoundness.ZK
