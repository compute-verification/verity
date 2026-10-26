import Properties.Verity.Soundness
import MemoryChallenger.WipeProtocol

/-!
# The wipe, the sixth subprotocol, in the causal game

Onsite is made of six subprotocols: sampled proofs, PoUW, PoUS, the bandwidth audit, the network certifier and the
wipe. `Properties.Verity.Game` states the first five at onsite's compiled components; this module adds the wipe, whose
compiled component is `MemoryChallenger.WipeProtocol.protocol` (`verity/wipe/v0`), in the same game (`Vocab.game`):

* the wipe's verdict (`WipeVocab.Verdict`) is its component's acceptance of its own case in what the prover sent
  (`WipeAccepts`), with every release its check read in the device's log;
* its claim is per server, in the physical model (Daniel, 26 Oct 12:16 PM PDT: "We only wipe all but 5 GB of the assumed
  memory capacity of the server"): each server `s` has a public assumed memory capacity `capacity θ s` (M_s, over every
  tier), and at the wipe tick that opens the period a server whose tiers hold at most M_s keeps at most `keepBits θ`
  bits of prior content (5 GB, 4 × 10^10 bits, at the operating point) plus its egress (`WipeVocab.Holds`);
* its target (`WipeVocab.WipeSound`, MEM-93) is that claim's bound per server, at every parameters, commitment and
  strategy, with error `delta θ s`.

`Vocab.SoundnessWithWipe` is `Vocab.Soundness` with the wipe's claim among the period's claims and its errors, summed
over the servers, in the bound. The wipe's claim is counted where the wipe's case is accepted: onsite's components
don't include `verity/wipe/v0` yet, so until `Core.Onsite.components` composes it, `WipeAccepts` holds of nothing and
the wipe's term bounds nothing.

The names the wipe defines are `WipeVocab`'s fields: `coinsRead` (ONE-4), what each server holds, keeps and sends in
each play (the memory challenger's world), and the capacities, kept bits and errors the product publishes, from the
parameters.
-/

namespace Verity

open Accounting Core
open scoped ENNReal

/-- The wipe's component accepts its own case in what the prover sent. -/
def WipeAccepts (s : Sent) : Prop := CaseAccepts MemoryChallenger.WipeProtocol.protocol s

/-- What one server's tiers do across the wipe tick that opens the period: the bits they hold, the bits of prior content
they keep across the tick, and the bits the server sent out. -/
structure ServerWipe where
  mem : ℕ
  kept : ℕ
  egress : ℕ

/-- **The names the wipe defines**, over the servers `Srv` and the rounds of the subprotocols' names `V`: the releases the
wipe's check read, decoded from the device's signed log entries in what the prover sent (ONE-4); what each server holds,
keeps and sends in each play; and each server's assumed capacity M_s, the bits a wipe leaves (5 GB at the operating
point) and the error per server, from the parameters alone. -/
structure WipeVocab {ι : Type} {d : Device ι} {α : Type} [Fintype α] (V : Vocab d α) (Srv : Type) where
  coinsRead : Sent → List Release
  world : (θ c : ByteArray) → Game.Strategy (V.game θ c) → Outcome → Srv → ServerWipe
  capacity : ByteArray → Srv → ℕ
  keepBits : ByteArray → ℕ
  delta : ByteArray → Srv → ℝ≥0∞

namespace WipeVocab

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] {V : Vocab d α} {Srv : Type} (W : WipeVocab V Srv)

/-- **The wipe's verdict**: its component accepts what the prover sent, and every release its check read is in the
device's log. -/
def Verdict (o : Outcome) : Prop := WipeAccepts o.sent ∧ ∀ r ∈ W.coinsRead o.sent, r ∈ o.log

/-- **The wipe's claim at server `s`**: if the server's tiers hold at most its assumed capacity M_s, it keeps at most
`keepBits` bits of prior content across the wipe tick, plus its egress. -/
def Holds (θ c : ByteArray) (σ : Game.Strategy (V.game θ c)) (s : Srv) (o : Outcome) : Prop :=
  (W.world θ c σ o s).mem ≤ W.capacity θ s →
    (W.world θ c σ o s).kept ≤ W.keepBits θ + (W.world θ c σ o s).egress

/-- **The wipe's break**: its component accepts on a release missing from the device's log, a signature under the
device's key on a message the device never signed. -/
def Break (o : Outcome) : Prop := WipeAccepts o.sent ∧ ∃ r ∈ W.coinsRead o.sent, r ∉ o.log

/-- **The wipe's target (MEM-93)**: at all parameters and commitments, for every strategy and every server, the
probability that the wipe accepts while that server's claim fails, with no break, is at most its published error. -/
def WipeSound : Prop :=
  ∀ (θ c : ByteArray) (σ : Game.Strategy (V.game θ c)) (s : Srv),
    Game.prob (fun o => W.Verdict o ∧ ¬ W.Holds θ c σ s o ∧ ¬ W.Break o) (V.game θ c) σ ≤ W.delta θ s

end WipeVocab

namespace Vocab

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] {Srv : Type} [Fintype Srv]

/-- **Soundness over the six subprotocols, in the causal game.** At all parameters and commitments, and for every
strategy, the probability that onsite's compiled verifier accepts while one of the period's claims fails (one of
`Vocab.Soundness`'s five, or the wipe's case is accepted while some server's claim fails), with no break, is at most
the published errors summed, the wipe's summed over the servers. -/
def SoundnessWithWipe (V : Vocab d α) (W : WipeVocab V Srv) : Prop :=
  ∀ (θ c : ByteArray) (σ : Game.Strategy (V.game θ c)),
    Game.prob (fun o => Accepts o.sent ∧
        ((∃ a, ¬ V.Holds θ c σ a o) ∨ (WipeAccepts o.sent ∧ ∃ s, ¬ W.Holds θ c σ s o)) ∧
        ¬ V.Break θ c o ∧ ¬ W.Break o) (V.game θ c) σ
      ≤ (∑ a, V.delta θ a) + ∑ s, W.delta θ s

end Vocab

end Verity
