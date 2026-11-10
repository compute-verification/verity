module

public import Proofs.Flock.ZkAny.Tables
public import Proofs.Flock.Soundness.ZK.Abort
public import Definitions.Flock.ZK.RealView

/-!
# The ZK server's release with real hm96 leaves, at any values

`ZkAny/Table.lean` and `ZkAny/Tables.lean` bound the release behind the server's checks with every unopened leaf ideal.
Here every leaf is real, on both branches: where the check passes, the view's hidden leaves are hm96 leaves under fresh
salts (`Hm96.viewR`); where it fails, the same run's pre-final messages leave (`Hm96.preR`), and every one of their leaves
is an hm96 leaf of what it commits, `τ`'s commitment included (`Hm96.TauLeaf`), since nothing is opened. Each branch
costs `2·δ₁` a leaf it leaves unopened: T1 over its leaves (`ideal_leaves_swap_fintype`, `real_leaves_swap_fintype`, the
commitment to any trace against the ideal one at `δ₁` a leaf, of which `hm96_commit_ideal_close` is the form at one
trace), once on each side, and nothing reads the values. These are lemmas for the deployed level's statements, not
guarantees of their own:

* `Table.PreLeaves`, `Table.preOf`, `Hm96.preR`, `Hm96.releaseR`: every leaf of the pre-final messages, a view's
  pre-final messages, the real prover's, and one table's release behind the check with real leaves.
* `Session.releaseR`, `Session.simR`, `Session.unopened`: a session's release at an outcome, its simulator, and how many
  leaves the release at an outcome leaves unopened; `Session.WitnessesFirst`: the witnesses read none of the server's
  samples for the session.
* `SameDist.prCoin_prod_eq`: equal distributions stay equal beside an independent coin.
* `Hm96.derived_close`, where the check passes: the view is the derived view with real hidden leaves (`viewR_derived`).
  Its hidden leaves go ideal, at each fixed vector of ideal leaves the derived view is distributed as `S_shvzk(x, e)`
  (`derived_sim`), and the leaves go real again on `S_shvzk`'s side: `2·|Hid|·δ₁` at any witness.
* `Hm96.preR_close`, where it fails: the real pre-final messages read the prover's salts at the opened positions only in
  those leaves, so the salts move to a fresh vector over every leaf (`toPre_viewR`, `prCoin_salts`); every leaf goes ideal,
  which gives the pre-final view (`prCoin_preLeaves_ideal`), Lemma A (`table_prefinal_indep`) moves it to the dummy, and
  the leaves go real again there: `2·|PreLeaves|·δ₁` at any witness carrying the public region data.
* `Hm96.release_hm96`: one table, per outcome of the check, by `gate_perOutcome`.
* `Session.release_hm96`: a session at fixed witnesses, per outcome, by `prCoin_pi_close` and `perOutcome_of_split`;
  `Session.release_hm96_first`: the same at witnesses fixed before the server's samples.
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace FlockSoundness.ZK

open Verity.Game
open scoped ENNReal

namespace Table

variable {F K W Salt B D Hi VL : Type} [Field F] [Field K] [AddCommGroup B] [CharP F 2]
  (T : Table F K W Salt (B × Salt) D Hi VL)

/-- **Every leaf of the pre-final messages**: those the final message opens (level 0's at the opened positions, each
rep's pads tree's at its opened positions, and each rep's commitment to `τ`), then those nobody opens (`Hid`). -/
abbrev PreLeaves := (Fin T.nS ⊕ (Fin 2 × Fin T.nQ) ⊕ Fin 2) ⊕ T.Hid

/-- **A view's pre-final messages**: everything before the final message, with the leaves the view has. -/
def preOf (v : T.View) : T.PreView :=
  { sHat := v.sHat, sReg := v.sReg, masked := v.masked, y1 := v.y1, ybar := v.ybar, lig := v.lig, inner := v.inner
    Y1 := v.Y1, Y2 := v.Y2, leaves0 := v.leaves0, tauCom := v.tauCom, leavesP := v.leavesP }

variable {T} {C : Type}

/-- **The real prover's pre-final messages**: the pre-final part of its run with real leaves (`viewR`). Every leaf is an
hm96 leaf of what it commits: level 0's and the pads trees' at the opened positions under their salts, the others under
fresh salts, and `τ`'s commitment under `y_τ`. -/
noncomputable def Hm96.preR (M : T.Hm96 C) (w : W) (ω : T.Rand) (ys : T.Hid → Salt) : T.PreView :=
  T.preOf (M.viewR w ω ys)

/-- **`τ`'s commitment is an hm96 leaf** of `τ`'s digest `digT τ`, as level 0's and the pads trees' leaves are. -/
def Hm96.TauLeaf (M : T.Hm96 C) (digT : F → B) : Prop :=
  ∀ t y, T.leafT t y = M.H (digT t + M.My y, M.cs y)

/-- **What the ZK server releases behind the check `go`, with real leaves**: the view where `go` passes, the same run's
pre-final messages where it doesn't. -/
noncomputable def Hm96.releaseR (M : T.Hm96 C) (go : T.Rand → Bool) (w : W) (ω : T.Rand) (ys : T.Hid → Salt) :
    T.View ⊕ T.PreView :=
  if go ω then .inl (M.viewR w ω ys) else .inr (M.preR w ω ys)

end Table

namespace Session

variable {F K W Salt B D Hi VL : Type} [Field F] [Field K] [AddCommGroup B] [CharP F 2] {J : ℕ}
  (T : Fin J → Table F K W Salt (B × Salt) D Hi VL) {C : Type} (M : ∀ j, (T j).Hm96 C) {St : Type}

/-- **What the ZK server releases from a session at the outcome `s`, with real leaves**: table `j`'s view where its file
left (`full s j`), the same run's pre-final messages elsewhere, each on its table's randomness and fresh salts. -/
noncomputable def releaseR (full : St → Fin J → Bool) (w : Fin J → W) (s : St)
    (p : ∀ j, (T j).Rand × ((T j).Hid → Salt)) : ∀ j, (T j).View ⊕ (T j).PreView :=
  fun j => if full s j then .inl ((M j).viewR (w j) (p j).1 (p j).2) else .inr ((M j).preR (w j) (p j).1 (p j).2)

/-- **The simulator at the outcome `s`, with real leaves**: `S_shvzk(x, e)` on the dummy with real leaves where table
`j`'s file left, the dummy's real pre-final messages elsewhere, each on its own randomness. -/
noncomputable def simR (full : St → Fin J → Bool) (w₀ : Fin J → W) (s : St)
    (σ : ∀ j, ((T j).SimRand × ((T j).Hid → Salt)) × ((T j).Rand × ((T j).Hid → Salt))) :
    ∀ j, (T j).View ⊕ (T j).PreView :=
  fun j => if full s j then .inl ((M j).simR (w₀ j) (σ j).1.1 (σ j).1.2)
    else .inr ((M j).preR (w₀ j) (σ j).2.1 (σ j).2.2)

/-- **The leaves the release at `s` leaves unopened**: a table's hidden leaves where its file left, every leaf of its
pre-final messages elsewhere. -/
def unopened (full : St → Fin J → Bool) (s : St) : ℕ :=
  ∑ j, if full s j then Fintype.card (T j).Hid else Fintype.card (T j).PreLeaves

/-- **The witnesses are fixed before the server's randomness**: the session's witnesses, as a function `wit` of every
sample the server makes for it (each table's level-0 randomness, pads and salts), read none of those samples. A server that
answers `level0` only once every table's witness is in gives it. At `Table.onVals` the witness carries the worker's
values, so it fixes them too. -/
def WitnessesFirst {Ω : Type} (wit : Ω → Fin J → W) : Prop :=
  ∀ p p', wit p = wit p'

end Session

/-- **Equal distributions beside an independent coin**: an event of `f ω` and an independent uniform `u` has the same
probability as the event of `g σ` and `u`. -/
theorem SameDist.prCoin_prod_eq {Ω Ω' U V : Type} [Fintype Ω] [Fintype Ω'] [Fintype U] [Nonempty Ω] [Nonempty Ω']
    {f : Ω → V} {g : Ω' → V} (h : SameDist f g) (G : V → U → Prop) :
    prCoin (fun p : Ω × U => G (f p.1) p.2) = prCoin (fun p : Ω' × U => G (g p.1) p.2) := by
  calc prCoin (fun p : Ω × U => G (f p.1) p.2)
      = prCoin (fun q : U × Ω => G (f q.2) q.1) :=
        (prCoin_congr_equiv (Equiv.prodComm U Ω) (fun p : Ω × U => G (f p.1) p.2)).symm
    _ = avg fun u => prCoin fun ω => G (f ω) u := prCoin_prod_split fun (u : U) ω => G (f ω) u
    _ = avg fun u => prCoin fun σ => G (g σ) u := congrArg avg (funext fun u => h.prCoin_eq fun v => G v u)
    _ = prCoin (fun q : U × Ω' => G (g q.2) q.1) := (prCoin_prod_split fun (u : U) σ => G (g σ) u).symm
    _ = prCoin (fun p : Ω' × U => G (g p.1) p.2) :=
        prCoin_congr_equiv (Equiv.prodComm U Ω') (fun p : Ω' × U => G (g p.1) p.2)

namespace Table

variable {F K W Salt B D Hi VL : Type} [Field F] [Field K] [CharP F 2] [Fintype F] [DecidableEq F] [Fintype K]
  [DecidableEq K] [Fintype Salt] [DecidableEq Salt] [Fintype B] [DecidableEq B] [Fintype Hi] [DecidableEq Hi]
  [AddCommGroup Salt] [AddCommGroup B] [AddCommGroup Hi] (T : Table F K W Salt (B × Salt) D Hi VL) {C : Type}

/-- A view with its hidden leaves `L`: `viewWith` and `simWith` are it at the prover's view and at `S_shvzk`'s. -/
def hidWith (v : T.View) (L : T.Hid → D) : T.View :=
  { v with
    leaves0 := (v.leaves0.1, fun q => L (.inl q))
    leavesP := fun r => ((v.leavesP r).1, fun k => L (.inr (r, k))) }

variable {T}

/-- **The derived view with real hidden leaves is within `2·|Hid|·δ₁` of `S_shvzk(x, e)` with real ones**, at any
witness: its hidden leaves go ideal, the derived view is distributed as `S_shvzk(x, e)` at each fixed vector of ideal
leaves (`derived_sim`), and the leaves go real again on `S_shvzk`'s side. -/
theorem Hm96.derived_close (M : T.Hm96 C) {δ₁ : ℝ≥0∞} (hT1 : Claims.Hm96Hiding M.My M.cs δ₁) (hrank : T.Rank)
    (hpad : T.PadOnto) (hW : ∀ r, Function.Bijective fun p : F × F => T.Wr r p.1 p.2) (hpads : T.PadsOnto)
    (hβ : ∀ r, T.β r ≠ 0) (w w₀ : W) (E : T.View → Prop) :
    prCoin (fun p : T.Rand × (T.Hid → Salt) =>
        E (T.hidWith (T.psi (T.realCore w p.1)) fun i => M.H (M.dig w p.1 i + M.My (p.2 i), M.cs (p.2 i)))) ≤
      prCoin (fun σ : T.SimRand × (T.Hid → Salt) => E (M.simR w₀ σ.1 σ.2)) + 2 * Fintype.card T.Hid * δ₁ := by
  have : Nonempty Salt := ⟨0⟩
  have : Nonempty T.Rand := ⟨0⟩
  have : Nonempty T.SimRand := ⟨0⟩
  have h1 := ideal_leaves_swap_fintype M.My M.cs hT1 M.H (M.dig w)
    fun ω L => E (T.hidWith (T.psi (T.realCore w ω)) L)
  have hm := (T.derived_sim hrank hpad hW hpads hβ w w₀).prCoin_prod_eq
    fun v (u : T.Hid → B × Salt) => E (T.hidWith v fun i => M.H ((u i).1, M.cs (u i).2))
  have h2 := real_leaves_swap_fintype M.My M.cs hT1 M.H (M.digS w₀)
    fun σ L => E (T.hidWith (T.sim w₀ σ) L)
  calc _ ≤ _ := h1
    _ = _ := congrArg (· + _) hm
    _ ≤ (prCoin (fun σ : T.SimRand × (T.Hid → Salt) => E (M.simR w₀ σ.1 σ.2)) + Fintype.card T.Hid * δ₁) +
          Fintype.card T.Hid * δ₁ := add_le_add h2 le_rfl
    _ = _ := by rw [add_assoc, ← two_mul, ← mul_assoc]

omit [Fintype K] [DecidableEq K] in
/-- **The real pre-final messages are within `2·|PreLeaves|·δ₁` of the dummy's**, at any witness carrying the public
region data: the opened leaves' salts move to a fresh vector (`toPre_viewR`, `prCoin_salts`), every leaf goes ideal,
which gives the pre-final view (`prCoin_preLeaves_ideal`), Lemma A moves it to the dummy (`table_prefinal_indep`), and
the leaves go real again there. -/
theorem Hm96.preR_close (M : T.Hm96 C) {digT : F → B} (hT : M.TauLeaf digT) {δ₁ : ℝ≥0∞}
    (hT1 : Claims.Hm96Hiding M.My M.cs δ₁) (hrank : T.Rank)
    (hW : ∀ r, Function.Bijective fun p : F × F => T.Wr r p.1 p.2) (hβ : ∀ r, T.β r ≠ 0) (w w₀ : W)
    (hreg : T.sReg w = T.pubReg) (hreg₀ : T.sReg w₀ = T.pubReg) (E : T.PreView → Prop) :
    prCoin (fun p : T.Rand × (T.Hid → Salt) => E (M.preR w p.1 p.2)) ≤
      prCoin (fun p : T.Rand × (T.Hid → Salt) => E (M.preR w₀ p.1 p.2)) + 2 * Fintype.card T.PreLeaves * δ₁ := by
  have : Nonempty Salt := ⟨0⟩
  have : Nonempty T.Rand := ⟨0⟩
  have hr : ∀ w', prCoin (fun p : T.Rand × (T.Hid → Salt) => E (M.preR w' p.1 p.2)) =
      prCoin (fun p : T.Rand × (T.PreLeaves → Salt) =>
        E (T.preLeaves w' p.1 fun i => M.H (M.digAll digT w' p.1 i + M.My (p.2 i), M.cs (p.2 i)))) := fun w' =>
    (congrArg prCoin (funext fun p : T.Rand × (T.Hid → Salt) =>
      congrArg E (T.toPre_viewR M digT hT w' p.1 p.2))).trans (T.prCoin_salts M digT w' E)
  have hi : ∀ w', prCoin (fun p : T.Rand × (T.PreLeaves → B × Salt) =>
        E (T.preLeaves w' p.1 fun i => M.H ((p.2 i).1, M.cs (p.2 i).2))) =
      prCoin (fun ω => E (T.preView w' ω)) := fun w' =>
    (congrArg prCoin (funext fun p : T.Rand × (T.PreLeaves → B × Salt) =>
      congrArg (fun L => E (T.preLeaves w' p.1 L))
        (funext fun i => (M.ideal_eq (p.2 i).1 (p.2 i).2).symm))).trans (T.prCoin_preLeaves_ideal w' E)
  have h1 := ideal_leaves_swap_fintype (ι := T.PreLeaves) M.My M.cs hT1 M.H (M.digAll digT w)
    fun ω L => E (T.preLeaves w ω L)
  have h2 := real_leaves_swap_fintype (ι := T.PreLeaves) M.My M.cs hT1 M.H (M.digAll digT w₀)
    fun ω L => E (T.preLeaves w₀ ω L)
  have hA := (T.table_prefinal_indep hrank hW hβ w w₀ hreg hreg₀).prCoin_eq E
  calc prCoin (fun p : T.Rand × (T.Hid → Salt) => E (M.preR w p.1 p.2))
      = _ := hr w
    _ ≤ _ := h1
    _ = prCoin (fun ω => E (T.preView w₀ ω)) + Fintype.card T.PreLeaves * δ₁ :=
        congrArg (· + Fintype.card T.PreLeaves * δ₁) ((hi w).trans hA)
    _ = prCoin (fun p : T.Rand × (T.PreLeaves → B × Salt) =>
          E (T.preLeaves w₀ p.1 fun i => M.H ((p.2 i).1, M.cs (p.2 i).2))) + Fintype.card T.PreLeaves * δ₁ :=
        congrArg (· + Fintype.card T.PreLeaves * δ₁) (hi w₀).symm
    _ ≤ (prCoin (fun p : T.Rand × (T.PreLeaves → Salt) =>
          E (T.preLeaves w₀ p.1 fun i => M.H (M.digAll digT w₀ p.1 i + M.My (p.2 i), M.cs (p.2 i)))) +
          Fintype.card T.PreLeaves * δ₁) + Fintype.card T.PreLeaves * δ₁ := add_le_add h2 le_rfl
    _ = _ := by rw [← hr w₀, add_assoc, ← two_mul, ← mul_assoc]

omit [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K] [Fintype Salt] [DecidableEq Salt] [Fintype B]
  [DecidableEq B] [Fintype Hi] [DecidableEq Hi] in
/-- **Where the check holds, the real view is the derived view with real hidden leaves** (`psi_realCore_at`). -/
theorem Hm96.viewR_derived (M : T.Hm96 C) (hW : ∀ r, Function.Bijective fun p : F × F => T.Wr r p.1 p.2)
    (hβ : ∀ r, T.β r ≠ 0) (w : W) (hreg : T.sReg w = T.pubReg) (ω : T.Rand) (hω : T.HoldsAt w ω)
    (ys : T.Hid → Salt) :
    M.viewR w ω ys = T.hidWith (T.psi (T.realCore w ω)) fun i => M.H (M.dig w ω i + M.My (ys i), M.cs (ys i)) :=
  congrArg (fun v => T.hidWith v fun i => M.H (M.dig w ω i + M.My (ys i), M.cs (ys i)))
    (T.psi_realCore_at hW hβ w hreg ω hω).symm

/-- **One table's release behind the check, at any values, with real leaves**, per outcome of the check: where it
passes, within `2·δ₁` a hidden leaf of `S_shvzk(x, e)` with real leaves; where it fails, within `2·δ₁` a leaf of its
pre-final messages of the dummy's real ones. -/
theorem Hm96.release_hm96 (M : T.Hm96 C) {digT : F → B} (hT : M.TauLeaf digT) {δ₁ : ℝ≥0∞}
    (hT1 : Claims.Hm96Hiding M.My M.cs δ₁) (hrank : T.Rank) (hpad : T.PadOnto)
    (hW : ∀ r, Function.Bijective fun p : F × F => T.Wr r p.1 p.2) (hpads : T.PadsOnto) (hβ : ∀ r, T.β r ≠ 0)
    (go : T.Rand → Bool) (w w₀ : W) (hreg : T.sReg w = T.pubReg) (hreg₀ : T.sReg w₀ = T.pubReg)
    (hgo : T.Gated go w) :
    PerOutcome (fun p : T.Rand × (T.Hid → Salt) => go p.1) (fun p => M.releaseR go w p.1 p.2)
      (gatedSim (fun σ : T.SimRand × (T.Hid → Salt) => M.simR w₀ σ.1 σ.2)
        fun p : T.Rand × (T.Hid → Salt) => M.preR w₀ p.1 p.2)
      fun b => if b then 2 * Fintype.card T.Hid * δ₁ else 2 * Fintype.card T.PreLeaves * δ₁ := by
  have : Nonempty Salt := ⟨0⟩
  have : Nonempty T.Rand := ⟨0⟩
  have : Nonempty T.SimRand := ⟨0⟩
  exact gate_perOutcome (view := fun p : T.Rand × (T.Hid → Salt) => M.viewR w p.1 p.2)
    (d := fun p => T.hidWith (T.psi (T.realCore w p.1)) fun i => M.H (M.dig w p.1 i + M.My (p.2 i), M.cs (p.2 i)))
    (cut := fun p => M.preR w p.1 p.2)
    (fun E' => M.preR_close hT hT1 hrank hW hβ w w₀ hreg hreg₀ E')
    (fun E' => M.derived_close hT1 hrank hpad hW hpads hβ w w₀ E')
    (fun p hp => M.viewR_derived hW hβ w hreg p.1 (hgo p.1 hp) p.2)

end Table

namespace Session

variable {F K W Salt B D Hi VL : Type} [Field F] [Field K] [CharP F 2] [Fintype F] [DecidableEq F] [Fintype K]
  [DecidableEq K] [Fintype Salt] [DecidableEq Salt] [Fintype B] [DecidableEq B] [Fintype Hi] [DecidableEq Hi]
  [AddCommGroup Salt] [AddCommGroup B] [AddCommGroup Hi] {J : ℕ}

/-- **Table `j`'s branch at the outcome `s`, with real leaves**: the derived view with real hidden leaves where its file
left, the real pre-final messages elsewhere. -/
noncomputable def branchR (T : Fin J → Table F K W Salt (B × Salt) D Hi VL) {C : Type} (M : ∀ j, (T j).Hm96 C)
    {St : Type} (full : St → Fin J → Bool) (w : Fin J → W) (s : St) (j : Fin J)
    (x : (T j).Rand × ((T j).Hid → Salt)) : (T j).View ⊕ (T j).PreView :=
  if full s j then
    .inl ((T j).hidWith ((T j).psi ((T j).realCore (w j) x.1)) fun i =>
      (M j).H ((M j).dig (w j) x.1 i + (M j).My (x.2 i), (M j).cs (x.2 i)))
  else .inr ((M j).preR (w j) x.1 x.2)

/-- **Each table's branch is within `2·δ₁` a leaf it leaves unopened of its simulator**: `Hm96.derived_close` where its
file left, `Hm96.preR_close` elsewhere. -/
theorem branchR_le (T : Fin J → Table F K W Salt (B × Salt) D Hi VL) {C : Type} (M : ∀ j, (T j).Hm96 C)
    {digT : Fin J → F → B} (hT : ∀ j, (M j).TauLeaf (digT j)) {δ₁ : ℝ≥0∞}
    (hT1 : ∀ j, Claims.Hm96Hiding (M j).My (M j).cs δ₁) (hrank : ∀ j, (T j).Rank) (hpad : ∀ j, (T j).PadOnto)
    (hW : ∀ j r, Function.Bijective fun p : F × F => (T j).Wr r p.1 p.2) (hpads : ∀ j, (T j).PadsOnto)
    (hβ : ∀ j r, (T j).β r ≠ 0) {St : Type} (full : St → Fin J → Bool) (w w₀ : Fin J → W)
    (hreg : ∀ j, (T j).sReg (w j) = (T j).pubReg) (hreg₀ : ∀ j, (T j).sReg (w₀ j) = (T j).pubReg) (s : St)
    (j : Fin J) (E : (T j).View ⊕ (T j).PreView → Prop) :
    prCoin (fun x => E (branchR T M full w s j x)) ≤
      prCoin (fun σ : ((T j).SimRand × ((T j).Hid → Salt)) × ((T j).Rand × ((T j).Hid → Salt)) =>
        E (if full s j then .inl ((M j).simR (w₀ j) σ.1.1 σ.1.2) else .inr ((M j).preR (w₀ j) σ.2.1 σ.2.2))) +
      (if full s j then 2 * Fintype.card (T j).Hid * δ₁ else 2 * Fintype.card (T j).PreLeaves * δ₁) := by
  have : Nonempty Salt := ⟨0⟩
  have : Nonempty (T j).Rand := ⟨0⟩
  have : Nonempty (T j).SimRand := ⟨0⟩
  unfold branchR
  cases hf : full s j
  · show prCoin (fun x : (T j).Rand × ((T j).Hid → Salt) => E (.inr ((M j).preR (w j) x.1 x.2))) ≤
      prCoin (fun σ : ((T j).SimRand × ((T j).Hid → Salt)) × ((T j).Rand × ((T j).Hid → Salt)) =>
        E (.inr ((M j).preR (w₀ j) σ.2.1 σ.2.2))) + 2 * Fintype.card (T j).PreLeaves * δ₁
    rw [prCoin_snd (A := (T j).SimRand × ((T j).Hid → Salt))
      (fun x : (T j).Rand × ((T j).Hid → Salt) => E (.inr ((M j).preR (w₀ j) x.1 x.2)))]
    exact (M j).preR_close (hT j) (hT1 j) (hrank j) (hW j) (hβ j) (w j) (w₀ j) (hreg j) (hreg₀ j)
      fun v => E (.inr v)
  · show prCoin (fun x : (T j).Rand × ((T j).Hid → Salt) =>
        E (.inl ((T j).hidWith ((T j).psi ((T j).realCore (w j) x.1)) fun i =>
          (M j).H ((M j).dig (w j) x.1 i + (M j).My (x.2 i), (M j).cs (x.2 i))))) ≤
      prCoin (fun σ : ((T j).SimRand × ((T j).Hid → Salt)) × ((T j).Rand × ((T j).Hid → Salt)) =>
        E (.inl ((M j).simR (w₀ j) σ.1.1 σ.1.2))) + 2 * Fintype.card (T j).Hid * δ₁
    rw [prCoin_fst (E := (T j).Rand × ((T j).Hid → Salt))
      (fun σ : (T j).SimRand × ((T j).Hid → Salt) => E (.inl ((M j).simR (w₀ j) σ.1 σ.2)))]
    exact (M j).derived_close (hT1 j) (hrank j) (hpad j) (hW j) (hpads j) (hβ j) (w j) (w₀ j) fun v => E (.inl v)

/-- **A session's release behind the ZK server's checks, at any values, with real leaves**, per outcome: at each
outcome, within `2·δ₁` a leaf it leaves unopened of the simulator at that outcome. -/
theorem release_hm96 (T : Fin J → Table F K W Salt (B × Salt) D Hi VL) {C : Type} (M : ∀ j, (T j).Hm96 C)
    {digT : Fin J → F → B} (hT : ∀ j, (M j).TauLeaf (digT j)) {δ₁ : ℝ≥0∞}
    (hT1 : ∀ j, Claims.Hm96Hiding (M j).My (M j).cs δ₁) (hrank : ∀ j, (T j).Rank) (hpad : ∀ j, (T j).PadOnto)
    (hW : ∀ j r, Function.Bijective fun p : F × F => (T j).Wr r p.1 p.2) (hpads : ∀ j, (T j).PadsOnto)
    (hβ : ∀ j r, (T j).β r ≠ 0) {St : Type} (out : (∀ j, (T j).Rand) → St)
    (full : St → Fin J → Bool) (w w₀ : Fin J → W) (hreg : ∀ j, (T j).sReg (w j) = (T j).pubReg)
    (hreg₀ : ∀ j, (T j).sReg (w₀ j) = (T j).pubReg) (hgo : Release.Gated T out full w) :
    PerOutcome (fun p : ∀ j, (T j).Rand × ((T j).Hid → Salt) => out fun j => (p j).1)
      (fun p => releaseR T M full w (out fun j => (p j).1) p) (simR T M full w₀)
      fun s => 2 * unopened T full s * δ₁ := by
  have : Nonempty Salt := ⟨0⟩
  have : ∀ j, Nonempty ((T j).Rand × ((T j).Hid → Salt)) := fun _ => ⟨(0, fun _ => 0)⟩
  have : ∀ j, Nonempty (((T j).SimRand × ((T j).Hid → Salt)) × ((T j).Rand × ((T j).Hid → Salt))) :=
    fun _ => ⟨((0, fun _ => 0), (0, fun _ => 0))⟩
  let ε : St → Fin J → ℝ≥0∞ := fun s j =>
    if full s j then 2 * Fintype.card (T j).Hid * δ₁ else 2 * Fintype.card (T j).PreLeaves * δ₁
  have hε : ∀ s, ∑ j, ε s j = 2 * unopened T full s * δ₁ := fun s => by
    simp only [ε, unopened, Nat.cast_sum, Finset.mul_sum, Finset.sum_mul]
    refine Finset.sum_congr rfl fun j _ => ?_
    split_ifs <;> rfl
  have hrel : Splits (fun p : ∀ j, (T j).Rand × ((T j).Hid → Salt) => releaseR T M full w (out fun j => (p j).1) p)
      (fun p => out fun j => (p j).1) (fun s p => fun j => branchR T M full w s j (p j)) :=
    fun p => funext fun j => by
      simp only [releaseR, branchR]
      by_cases hf : full (out fun j => (p j).1) j = true
      · rw [ite_eq_left hf, ite_eq_left hf,
          (M j).viewR_derived (hW j) (hβ j) (w j) (hreg j) (p j).1 (hgo (fun j => (p j).1) j hf) (p j).2]
      · rw [ite_eq_right hf, ite_eq_right hf]
  refine perOutcome_of_split hrel fun s E' => ?_
  have h := prCoin_pi_close (Ω := fun j => (T j).Rand × ((T j).Hid → Salt))
    (Ω' := fun j => ((T j).SimRand × ((T j).Hid → Salt)) × ((T j).Rand × ((T j).Hid → Salt)))
    (V := fun j => (T j).View ⊕ (T j).PreView) (fun j => branchR T M full w s j)
    (fun j σ => if full s j then .inl ((M j).simR (w₀ j) σ.1.1 σ.1.2) else .inr ((M j).preR (w₀ j) σ.2.1 σ.2.2))
    (ε s) (fun j E'' => branchR_le T M hT hT1 hrank hpad hW hpads hβ full w w₀ hreg hreg₀ s j E'') E'
  rw [hε s] at h
  exact h

/-- **A session's release behind the ZK server's checks, at any values, with real leaves, at witnesses fixed before the
server's samples** (`WitnessesFirst`): `release_hm96` at the one witness they take. At each outcome `s` the release is
within `2·δ₁` a leaf it leaves unopened of the simulator given `s`, which reads the dummies `w₀` and not the
witnesses. -/
theorem release_hm96_first (T : Fin J → Table F K W Salt (B × Salt) D Hi VL) {C : Type} (M : ∀ j, (T j).Hm96 C)
    {digT : Fin J → F → B} (hT : ∀ j, (M j).TauLeaf (digT j)) {δ₁ : ℝ≥0∞}
    (hT1 : ∀ j, Claims.Hm96Hiding (M j).My (M j).cs δ₁) (hsim : ∀ j, (T j).Simulable) {St : Type}
    (out : (∀ j, (T j).Rand) → St) (full : St → Fin J → Bool)
    (wit : (∀ j, (T j).Rand × ((T j).Hid → Salt)) → Fin J → W) (w₀ : Fin J → W) (hfirst : WitnessesFirst wit)
    (hreg : ∀ p j, (T j).RegPublic (wit p j)) (hreg₀ : ∀ j, (T j).RegPublic (w₀ j))
    (hgo : ∀ p, Release.Gated T out full (wit p)) :
    PerOutcome (fun p : ∀ j, (T j).Rand × ((T j).Hid → Salt) => out fun j => (p j).1)
      (fun p => releaseR T M full (wit p) (out fun j => (p j).1) p) (simR T M full w₀)
      fun s => 2 * unopened T full s * δ₁ := by
  have : Nonempty Salt := ⟨0⟩
  let p₀ : ∀ j, (T j).Rand × ((T j).Hid → Salt) := fun _ => (0, fun _ => 0)
  obtain ⟨w, rfl⟩ : ∃ w, wit = fun _ => w := ⟨wit p₀, funext fun p => hfirst p p₀⟩
  exact release_hm96 T M hT hT1 (fun j => (hsim j).1) (fun j => (hsim j).2.1) (fun j => (hsim j).2.2.1)
    (fun j => (hsim j).2.2.2.1) (fun j => (hsim j).2.2.2.2) out full w w₀ (hreg p₀) hreg₀ (hgo p₀)

end Session

end FlockSoundness.ZK
