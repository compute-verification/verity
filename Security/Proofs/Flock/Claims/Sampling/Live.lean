import Proofs.Flock.Claims.Sampling.Draws
import Proofs.Flock.Claims.Sampling.ExecBernoulli

/-!
# The live draw's miss bounds, as the claims read probabilities

The claims' verifier draws from uniform coins (`Claims.Pr`). The deployed verifier draws with `Flock.Draw.drawWith`,
reading each stream's bytes and extending and rerunning a stream that runs out; `FlockSoundness.Audit.Law.runOS f law pop`
is that draw on the bytes `f` (`none` when it would read past them). Taking the claims' coins to be those bytes, the
identity source is uniform by definition (`uniformBytes_id`), so these statements need no assumption: A3
(`Assumptions.UniformRandomBytes`) is what puts `IO.getRandomBytes` there, and it is not read here.

* `LiveMiss law pop B f`: the live draw on `f` returns a draw that misses `B`.
* `live_le_model`: the live draw misses `B` no more often than a model law whose escape bounds it.
* `live_subset_miss_le`, `live_bernoulli_miss_le`, `live_stratified_miss_le`, `live_work_miss_le`: each registered law's
  live draw misses `t ≤ |B|` wrong regions with probability at most its bound (`Claims.Sampling` in `Draws.lean`), for
  every number of bytes `L` read.
* `live_stratified_miss_le_count`: the stratified live draw, at most its model's count curve at `t`.
-/

namespace Claims.Sampling

open FlockSoundness FlockSoundness.Audit FlockSoundness.Game Finset
open scoped ENNReal

/-- **The claims' coins as the live draw's bytes**: the identity source is uniform. -/
theorem uniformBytes_id {m : ℕ} (L : Fin m → ℕ) :
    Assumptions.UniformRandomBytes L (fun f : (i : Fin m) → Fin (L i) → Fin 256 => f) := fun _ => rfl

/-- The live draw on the bytes `f` returns a draw that misses `B`. -/
def LiveMiss {m n : ℕ} {L : Fin m → ℕ} (law : Flock.Draw.Law) (pop : ℕ) (B : Finset (Fin n))
    (f : (i : Fin m) → Fin (L i) → Fin 256) : Prop :=
  ∃ d, Law.runOS f law pop = some d ∧ ∀ u ∈ B, u.val ∉ d.units

/-- **The live draw misses no more often than a model law** whose escape bounds its `prCoin`. -/
theorem live_le_model {m n : ℕ} {L : Fin m → ℕ} (law : Flock.Draw.Law) (pop : ℕ) (M : Law n) (B : Finset (Fin n))
    (h : prCoin (LiveMiss (L := L) law pop B) ≤ M.escape B) :
    Claims.Pr (LiveMiss (L := L) law pop B) ≤ Claims.Pr (fun ω : M.Ω => Disjoint (M.draw ω) B) := by
  rw [Pr_eq_toReal_prCoin, Pr_miss_eq]
  exact ENNReal.toReal_mono (ne_top_of_le_ne_top ENNReal.one_ne_top (DrawExec.prCoin_le_one _)) h

/-- **The live uniform `k`-subset misses `t` wrong regions** with probability at most `C(n − t, k)/C(n, k)`. -/
theorem live_subset_miss_le (n k : ℕ) (hk : k ≤ n) (L : Fin 1 → ℕ) {B : Finset (Fin n)} {t : ℕ} (ht : t ≤ B.card) :
    Claims.Pr (LiveMiss (L := L) (.subset k) n B) ≤ ((n - t).choose k : ℝ) / (n.choose k : ℝ) :=
  (live_le_model _ n (Law.subset n k hk) B
    (Law.drawOS_subset_escape_le n k hk L _ (uniformBytes_id L) B)).trans (subset_miss_le hk ht)

/-- **The live Bernoulli draw misses `t` wrong regions** with probability at most `((den − num)/den)^t`. -/
theorem live_bernoulli_miss_le (n num den : ℕ) [NeZero den] (h : num ≤ den) (L : Fin 1 → ℕ) {B : Finset (Fin n)}
    {t : ℕ} (ht : t ≤ B.card) :
    Claims.Pr (LiveMiss (L := L) (.bernoulli num den) n B) ≤ (((den - num : ℕ) : ℝ) / den) ^ t :=
  (live_le_model _ n (Law.bernoulli n num den) B
    (Law.drawOS_bernoulli_escape_le n num den h L _ (uniformBytes_id L) B)).trans (bernoulli_miss_le h ht)

/-- **The live stratified draw misses `t` wrong regions** with probability at most `(1 − π)^t`, when its strata `S` are
the model's (`Law.ExecStrata`) and every non-empty stratum draws at least a fraction `π ≤ 1` of its units. -/
theorem live_stratified_miss_le {n m : ℕ} (σ : Fin n → Fin m) (k : Fin m → ℕ)
    (hk : ∀ s, k s ≤ (Law.stratum σ s).card) (S : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σ k S) (K pop : ℕ)
    (L : Fin m → ℕ) {π : ℝ} (hπ0 : 0 ≤ π) (hπ1 : π ≤ 1)
    (hπ : ∀ s, 0 < (Law.stratum σ s).card → π * (Law.stratum σ s).card ≤ k s) {B : Finset (Fin n)} {t : ℕ}
    (ht : t ≤ B.card) :
    Claims.Pr (LiveMiss (L := L) (.stratified K (Array.ofFn S)) pop B) ≤ (1 - π) ^ t :=
  (live_le_model _ pop (Law.stratified σ k hk) B
    (Law.drawOS_stratified_escape_le σ k hk S hS K pop L _ (uniformBytes_id L) B)).trans
    (stratified_miss_le σ k hk hπ0 hπ1 hπ ht)

/-- **The live stratified draw misses `t` wrong regions** with probability at most its model's count curve at `t`. -/
theorem live_stratified_miss_le_count {n m : ℕ} (σ : Fin n → Fin m) (k : Fin m → ℕ)
    (hk : ∀ s, k s ≤ (Law.stratum σ s).card) (S : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σ k S) (K pop : ℕ)
    (L : Fin m → ℕ) {B : Finset (Fin n)} {t : ℕ} (ht : t ≤ B.card) :
    Claims.Pr (LiveMiss (L := L) (.stratified K (Array.ofFn S)) pop B) ≤ ((Law.stratified σ k hk).miss t).toReal :=
  (live_le_model _ pop (Law.stratified σ k hk) B
    (Law.drawOS_stratified_escape_le σ k hk S hS K pop L _ (uniformBytes_id L) B)).trans (miss_le_count _ ht)

/-- **The live work draw misses units doing at least `T` of the work** `W` with probability at most `((W − T)/W)^K`,
when its strata `Ws` are the work law's (`Law.ExecStrata` at `Law.workK`). -/
theorem live_work_miss_le {n m : ℕ} (σ : Fin n → Fin m) (w : Fin m → ℕ) (f : Fin m → ℕ) (K : ℕ) (hf : ∀ s, 1 ≤ f s)
    (hW : 0 < Law.totalWork σ w) (S : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σ (Law.workK σ w f K) S)
    (K' pop : ℕ) (Ws : Array Flock.Draw.WorkStratum) (hWs : Ws.map (·.toStratum) = Array.ofFn S) (L : Fin m → ℕ)
    {B : Finset (Fin n)} {T : ℕ} (hT : T ≤ Law.workOf σ w B) :
    Claims.Pr (LiveMiss (L := L) (.work K' Ws) pop B) ≤
      (((Law.totalWork σ w - T : ℕ) : ℝ) / (Law.totalWork σ w : ℝ)) ^ K :=
  (live_le_model _ pop (Law.work σ w f K) B
    (Law.drawOS_work_escape_le σ _ (Law.workK_le σ w f K) S hS K' pop Ws hWs L _ (uniformBytes_id L) B)).trans
    (work_miss_le σ w f K hf hW hT)

end Claims.Sampling
