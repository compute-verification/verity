import Proofs.Pouw.PearlC.HiddenSystem
import Proofs.Pouw.Draft.Unambiguous

/-!
# The honest prover in its class, from decisive tile games

`HiddenSystem.honestProver_bounded` asks that the drawn tiles' proof games read as no other list of games
(`Unambiguous`). That holds when each tile's proof game is decisive on its own (`Decisive`: every message type
inhabited, some run accepting and some rejecting), by `Pouw.Draft.Game.unambiguous_of_decisive`:
`HiddenSystem.unambiguous_of_decisive` gives the hypothesis, and `HiddenSystem.honestProver_bounded_of_decisive` is
`honestProver_bounded` with it. `HiddenSystem.decisive_of_sys_decisive` gives that lemma's `hDec` from one fact per kind
of tile check: its game, relabelled to its verdict, is decisive on every root. Decisiveness is existential (some run
accepts, for messages and coins of its choosing), not completeness.
-/

namespace Pouw.PearlC

open Pouw.Draft.SampledProofs

variable {Q R S : Type}

/-- **Decisive tile games make every draw unambiguous.** When the proof game of every tile of the layout, on the honest
commitment, is decisive, the drawn tiles' proof games read as no other list, whatever the draw. -/
theorem HiddenSystem.unambiguous_of_decisive (HS : HiddenSystem Q R S) (TR : TileRules Q R S) (U : Workload)
    (τ : (Q → R) → S → Transcript) (salts : (Q → R) → S → ℕ → HS.O) (w : ℕ → ℕ) (t : ℕ)
    (hDec : ∀ (H : Q → R) (s : S) (j : ℕ), j < (TR.tiling U.layout).NT →
      Verity.Game.Decisive (HS.game H s U (HS.root U (τ H s) (HS.honestOpening U salts H s)) j)) :
    ∀ x : (Q → R) × S × (Fin t → Ticket w (TR.tiling U.layout).NT),
      Pouw.Draft.Game.Unambiguous (List.ofFn fun i =>
        HS.game x.1 x.2.1 U (HS.root U (τ x.1 x.2.1) (HS.honestOpening U salts x.1 x.2.1)) (x.2.2 i).1) := by
  intro x
  refine Pouw.Draft.Game.unambiguous_of_decisive _ fun g hg => ?_
  obtain ⟨i, rfl⟩ := List.mem_ofFn.1 hg
  exact hDec x.1 x.2.1 _ (x.2.2 i).1.isLt

/-- **The honest prover is in the class, for decisive tile games.** `HiddenSystem.honestProver_bounded` with its
unambiguity hypothesis replaced by one on each tile's proof game on its own: on the honest commitment, every message
type it asks for is inhabited, and some run accepts and some run rejects. -/
theorem HiddenSystem.honestProver_bounded_of_decisive (HS : HiddenSystem Q R S) (hHon : ∀ k, HonestIn (HS.sys k))
    (TR : TileRules Q R S) (U : Workload) (τ : (Q → R) → S → Transcript)
    (salts : (Q → R) → S → ℕ → HS.O) (w : ℕ → ℕ) (t q : ℕ)
    (hD : ∀ H s, HS.Distinct U H s (τ H s) (HS.honestOpening U salts H s))
    (hDec : ∀ (H : Q → R) (s : S) (j : ℕ), j < (TR.tiling U.layout).NT →
      Verity.Game.Decisive (HS.game H s U (HS.root U (τ H s) (HS.honestOpening U salts H s)) j)) :
    (HiddenAudit.ofSystem HS).ProverBounded (HS.honestProver TR U τ salts w t) q :=
  HS.honestProver_bounded hHon TR U τ salts w t q hD (HS.unambiguous_of_decisive TR U τ salts w t hDec)

/-- **Decisive tile checks give decisive tile games.** When every kind of tile check's game, relabelled to its verdict,
is decisive on every root, so is every tile's proof game on the honest commitment, which names the verifier's oracle,
salt and workload: `honestProver_bounded_of_decisive`'s `hDec`. -/
theorem HiddenSystem.decisive_of_sys_decisive (HS : HiddenSystem Q R S) (TR : TileRules Q R S) (U : Workload)
    (τ : (Q → R) → S → Transcript) (salts : (Q → R) → S → ℕ → HS.O)
    (hSys : ∀ (k : HS.K) (r : HS.Rt), Verity.Game.Decisive (((HS.sys k).game r).map (HS.sys k).accept)) :
    ∀ (H : Q → R) (s : S) (j : ℕ), j < (TR.tiling U.layout).NT →
      Verity.Game.Decisive (HS.game H s U (HS.root U (τ H s) (HS.honestOpening U salts H s)) j) := by
  intro H s j _
  have e : HS.game H s U (HS.root U (τ H s) (HS.honestOpening U salts H s)) j =
      ((HS.sys (HS.kind U j)).game (HS.root U (τ H s) (HS.honestOpening U salts H s) H s j).2).map
        (HS.sys (HS.kind U j)).accept := ite_eq_left rfl
  rw [e]
  exact hSys _ _

end Pouw.PearlC
