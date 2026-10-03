import FlockSoundness.Discharge.ZkTree.Hybrid
import FlockSoundness.Discharge.Exec.Live

/-!
# The tree-coin run of a one-stage audit

`treeAudit_le_model`: in a one-stage audit whose session is a committed interaction `g` of the model session `G`
(`ErD g G`, no round opened twice), the tree-coin run (`prTree`: the prover registers, the units are drawn, the prover
sends its nonce, then the server's key, blocks and salts) is, on every event of the audit's outcome, within
`|I|·2^-256` of a mixture of strategies of the model audit `Exec.auditLive` that register what the tree prover
registers. The mixture does not depend on the event.

`erD_liveTableC`, `distinct_liveTableC`: the live table session on the coin tree is `Refine.liveTable` with coins read
off committed rounds, and opens each round at most once when its three streams are distinct.
-/

namespace FlockSoundness.Discharge.ZkTree

open FlockSoundness Game Discharge.ZkCoins Discharge.Exec Refine
open scoped ENNReal

/-- **The tree-coin run of a one-stage audit against the model audit.** -/
theorem treeAudit_le_model {n : ℕ} {E₂ I Blk Cs D Nonce Reg β : Type} [Field E₂] [Fintype E₂] [DecidableEq E₂]
    (hE : Fintype.card E₂ = 2) [Fintype I] [DecidableEq I] [Fintype Blk] [Nonempty Blk] [DecidableEq Blk]
    [DecidableEq Cs] (L : Audit.Law n) {g : TGame I Blk β} {G : Game β} (hgG : ErD g G) (hg : g.Distinct)
    (out : Finset (Fin n) → Reg → β → Transcript)
    (dig : Nonce → I → Blk → (Fin 512 → E₂)) (cs : Nonce → Salt E₂ → Cs)
    (hcs : ∀ ν, (Finset.univ.image (cs ν)).card ≤ 2 ^ 512) (H : Nonce → Key E₂ → (Fin 512 → E₂) × Cs → D)
    (τ : TreeProver L Reg Nonce E₂ D g) :
    ∃ M : Mix (auditLive L Reg fun S R => G.map (out S R)) PUnit, (∀ ρ, (M.σ ρ).1 = τ.1) ∧
      ∀ E, prTree L g out dig cs H τ E ≤ M.pr (fun _ => E) + Fintype.card I * ENNReal.ofReal (1 / 2 ^ 256) := by
  sorry

local notation "𝔽" => FlockLevel3.GF128

variable {Blk : Type} (dg : ByteArray → ByteArray) {nS : ℕ} (rl : ℕ) (i0 i1 : Fin nS)
  (dr : (n : ℕ) → Option (Dec Blk (Fin n → 𝔽))) (iS : Fin nS) {mPts : ℕ}
  (dl : Option (Dec Blk (Fin 2 → Fin mPts → 𝔽))) (N : ℕ)

/-- **The live table session on the coin tree is the live table session**, with its coins read off committed
rounds. -/
noncomputable def erD_liveTableC : ErD (liveTableC dg rl i0 i1 dr iS dl N) (liveTable dg mPts N) := by
  sorry

/-- **The live table session on the coin tree opens each round at most once.** -/
theorem distinct_liveTableC (hS0 : iS ≠ i0) (hS1 : iS ≠ i1) (h01 : i0 ≠ i1) :
    (liveTableC dg rl i0 i1 dr iS dl N).Distinct := by
  sorry

end FlockSoundness.Discharge.ZkTree
