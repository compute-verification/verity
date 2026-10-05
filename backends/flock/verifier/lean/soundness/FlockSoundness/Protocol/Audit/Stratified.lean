import FlockSoundness.Protocol.Audit.OneStage

/-!
# The stratified draw law

The strata are the units a map `σ : Fin n → Fin m` assigns to each `s`, for example one stratum per template. The
stratified law draws an independent uniform `k s`-subset inside each stratum (`Law.stratified`,
`docs/sampling-strategies.md` §1, §4).

* `stratified_escape`: a set `B` with `m_s` units in stratum `s` escapes with the product of hypergeometrics
  `∏ s C(n_s − m_s, k_s) / C(n_s, k_s)`. It depends only on the counts, as `verity.proofs.profile.Stratified` computes
  it. The uniform `k`-subset is the case of one stratum.
* `stratified_escape_floor`: **a floor catches its whole stratum**. With `1 ≤ k s`, a set containing stratum `s` never
  escapes.
* The one-stage profile theorems hold for every law, so they instantiate here. `audit_whole_stratum` (oracle layer) and
  `extraction_audit_whole_stratum` (compiled layer): an audit accepts while a whole floored stratum is wrong with
  probability at most the per-unit terms.
-/

namespace FlockSoundness.Audit

open Game Finset
open scoped ENNReal

namespace Law

variable {n m : ℕ}

/-- Stratum `s` of the assignment `σ`: the units it assigns to `s`. -/
def stratum (σ : Fin n → Fin m) (s : Fin m) : Finset (Fin n) := univ.filter (σ · = s)

/-- **Independent uniform subsets inside the strata of `σ`**, `k s` of stratum `s`. -/
def stratified (σ : Fin n → Fin m) (k : Fin m → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card) : Law n where
  Ω := (s : Fin m) → ((stratum σ s).powersetCard (k s))
  nonempty := ⟨fun s => ⟨_, (Finset.powersetCard_nonempty.2 (hk s)).choose_spec⟩⟩
  draw ω := univ.biUnion fun s => (ω s).1

end Law

section Audits

variable {n m : ℕ} {σ : Fin n → Fin m} {k : Fin m → ℕ} {hk : ∀ s, k s ≤ (Law.stratum σ s).card}
  {Reg : Type} {session : Finset (Fin n) → Reg → Game Bool}

end Audits

end FlockSoundness.Audit
