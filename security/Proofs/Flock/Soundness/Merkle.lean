module

public import Definitions.Flock.Merkle
public import Mathlib.Tactic.Basic

/-!
# Merkle trees: the facts the binding rests on

`append_inj`: a node's input determines its two children, since all digests have one length. `verifiesL_plain`: an
opening verifies under the unsalted leaf scheme (`Leaf.plain`) exactly when it verifies (`Verifies`).

**Binding** is in `Proofs.Flock.Soundness.MerklePair`: two openings of one position that both verify against one tree
top (`Definitions.Flock.Merkle`) but carry different columns give two byte strings that collide under `H`
(`openCollPair_spec`, and `openCollPairL_spec` for any leaf scheme, salted ones included).
-/

@[expose] public section

namespace FlockSoundness.Merkle

variable {Col D : Type}

variable (H : List UInt8 → D) (E : Enc Col D)

/-- `up` is `H(acc ‖ sib)` at an even index and `H(sib ‖ acc)` at an odd one. -/
theorem up_eq (idx : ℕ) (acc sib : D) :
    up H E idx acc sib = if idx % 2 = 0 then H (E.dig acc ++ E.dig sib) else H (E.dig sib ++ E.dig acc) := by
  unfold up upIn
  exact apply_ite H _ _ _

theorem append_inj {a b a' b' : D} (h : E.dig a ++ E.dig b = E.dig a' ++ E.dig b') :
    a = a' ∧ b = b' := by
  have hl : (E.dig a).length = (E.dig a').length := E.dig_len a a'
  obtain ⟨h1, h2⟩ := List.append_inj h hl
  exact ⟨E.dig_inj h1, E.dig_inj h2⟩

variable {S : Type}

theorem verifiesL_plain (cap : ℕ → D) (depth pos : ℕ) (column : Col) (sibs : List D) :
    VerifiesL H E (Leaf.plain H E) cap depth pos column () sibs ↔ Verifies H E cap depth pos column sibs :=
  Iff.rfl

end FlockSoundness.Merkle
