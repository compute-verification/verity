import Properties.Pous
import Definitions.Pous.P2.Cost

/-!
# Proofs of the PoUS setup check's properties (`Properties.Pous`), and what is not a claim

Proved: `deployed : Deployed`, the registered point's numbers, by reduction (`rfl`, `decide`). `AcceptsDecodes` and
`PublishesRoot` are proved in `Proofs.Properties.Pous.Setup`.

Stated here, open, and claims of no one:

* `AcceptsComplete`, the completeness half that pairs with `AcceptsDecodes`: at valid parameters, a `C*` that decodes to
  the committed `W` is accepted. Its honest case (`HonestSetup`) is stated as written. Its concrete instance, the setup
  vector `SetupProtocol`'s guards accept (`p2-6B`, `W = 01 02 … 0c`), is not given yet: `Commit.commit` reads the
  manifest through `Flock.canonUtf8`, well-founded recursion the kernel does not evaluate.
* `DecodeIsModel`, a lemma statement: the code's `P2.decodeValue p K t c` is the trusted model's block decoder
  `Pous.decBlock` (P2 in the ring model `ringRO`) run at the identity labeling with block keys `(K, t)`, on every
  canonical `c < p`, read mod `2^b`. With `Decodes` (which reads `decodeValue`), it would tie what the check accepts to
  `p2ring`'s `dec`, which the model's audit statements read. Nothing cites it as proved.
-/

namespace PousCheck.Properties

/-- The honest case of setup: valid parameters, and a `C*` that decodes to the committed `W`. -/
def HonestSetup (q : P2.Params) (c : SetupProtocol.Commitment) (C : Array (Array ByteArray)) : Prop :=
  q.validate = .ok q ∧ DecodesToCommitment q c C

/-- Completeness: the check accepts every honest setup. -/
def AcceptsComplete : Prop :=
  ∀ (q : P2.Params) (c : SetupProtocol.Commitment) (C : Array (Array ByteArray)),
    HonestSetup q c C → SetupProtocol.protocol.Accepts q c C

/-- The identity labeling `φ = id` of `Z/p` (the model's `nonemptyΩ` witness), at which `p2ring` is
`decode_segment` for given keys. -/
def idLabeling (p : ℕ) [NeZero p] : Pous.OddLabeling p :=
  ⟨(ZMod.finEquiv p).symm.toEquiv, fun x => map_neg (ZMod.finEquiv p).symm x⟩

/-- At `(B, w, p, b)`: for every block `j` whose keys are the mask `K < 2^w` and the offset `t < p`, and every canonical
`c < p`, the model's `decBlock` run at the identity labeling reads `P2.decodeValue p K t c mod 2^b`. -/
def DecodeIsModelAt (B w p b : ℕ) [NeZero p] : Prop :=
  ∀ (keys : Fin B → Pous.Bits w × Fin p) (j : Fin B) (K c : ℕ),
    K < 2 ^ w → c < p → (keys j).1 = Pous.bitsOfNat w K →
      Pous.bitsVal ((Pous.decBlock B w p b j (Pous.bitsOfNat w c)).run
          ((Pous.ringRO B w p).answer (idLabeling p, keys)))
        = P2.decodeValue p K (keys j).2 c % 2 ^ b

def DecodeIsModel : Prop := ∀ (B w p b : ℕ) [NeZero p], p ≤ 2 ^ w → DecodeIsModelAt B w p b

end PousCheck.Properties

namespace PousCheck.Properties.SecurityProofs

set_option exponentiation.threshold 20000 in
theorem deployed : Deployed := by
  unfold Deployed
  exact ⟨rfl, rfl, rfl, by decide, by decide, by decide, rfl, rfl, by decide⟩

end PousCheck.Properties.SecurityProofs
