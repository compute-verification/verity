import Properties.Verity.Soundness
import MemoryChallenger.WipeProtocol
import Definitions.Pous.Guarantees.StreamedWipeServer

/-!
# The wipe, the sixth subprotocol, in the causal game

Onsite is made of six subprotocols: sampled proofs, PoUW, PoUS, the bandwidth audit, the network certifier and the
wipe. `Properties.Verity.Game` states the first five at onsite's compiled components; this module adds the wipe, whose
compiled component is `MemoryChallenger.WipeProtocol.protocol` (`verity/wipe/v0`), in the same game (`Vocab.game`):

* the wipe's verdict (`WipeVocab.Verdict`) is the commit phase's file check, NCI's acceptance of the registration sampled
  proofs committed to (`RegistrationAccepts`), and its component's acceptance of its own case in what the prover sent
  (`WipeAccepts`);
* its claim is per server, in the physical model, at the **registered** wipe: each server `s` publishes its wipe (`n`
  labels, ratio `r`, `k` reads, `R` bits outside the fill, public parameters of `θ`), so its assumed memory capacity is
  `M_s = 512 · n + R` over every tier (`capacity`), and at the wipe tick that opens the period a server whose tiers hold
  at most M_s keeps at most `B(n, r, k, R)` bits of prior content (`keepBound`, `StreamedWipe.serverBound`) plus its
  egress (`WipeVocab.Holds`);
* its target (`WipeVocab.WipeSound`, MEM-93) is that claim's bound per server, at every parameters, commitment and
  strategy, with error `StreamedWipeMeetsServer`'s bound at the wipe model's prover built from the strategy (`2^-128`
  plus the PRF's advantage against the distinguisher built from it) plus the forged-release finder's loss
  (`WipeVocab.error`). It follows from `Pous.Guarantees.StreamedWipeMeetsServer` (memory-accounting, MEM-95) and the
  coupling of the wipe model's game into this one (`WipeVocab.Couples`): `Verity.WipeVocab.wipeSound_of_streamed`.
  The 5 GB form (Daniel, 26 Oct 12:16 PM PDT: "We only wipe all but 5 GB of the assumed memory capacity of the server")
  is a corollary where the registered wipe is at that point (`AtOperatingPoint`, `Holds5GB`,
  `wipeSound5GB_of_wipeSound`), never a premise of the theorem: today's fill leaves `R` above it.

`Vocab.SoundnessWithWipe` is `Vocab.Soundness` with the wipe's claim among the period's claims and its errors, summed
over the servers, in the bound. Onsite composes the wipe (`verity/wipe/v0`), so what onsite accepts, the wipe's case
accepts (`WipeAccepts`).

The names the wipe defines are `WipeVocab`'s fields: `coinsRead` (ONE-4), what each server holds, keeps and sends in
each play (the memory challenger's world), each server's published wipe and egress bound, the PRF, the wipe model's
prover built from the strategy, and the forged-release finder and its loss.
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
keeps and sends in each play; each server's wipe as published (the fill's `n` labels, the ratio `r`, the read count
`k`, the bits `R` outside the fill, so its assumed capacity is `M_s = 512 · n + R`) and the egress bound `E` its network
certifier enforces, with the side conditions `StreamedWipeMeetsServer` reads of the published `n`, `r` and `k`
(`n_pos`, `r_pos`, `reads`: public parameters, which the registration fixes); the PRF the labels come from; the wipe model's prover built from the strategy at each server
(`prover`, a fixed construction, whose distinguisher `StreamedWipe.wipeDistinguisherBy` is the wipe's finder term); the
forged-release finder per server and its loss. -/
structure WipeVocab {ι : Type} {d : Device ι} {α : Type} [Fintype α] (V : Vocab d α) (Srv : Type) where
  coinsRead : Sent → List Release
  world : (θ F c : ByteArray) → Game.Strategy (V.game θ F c) → Outcome → Srv → ServerWipe
  n : ByteArray → Srv → ℕ
  r : ByteArray → Srv → ℕ
  k : ByteArray → Srv → ℕ
  R : ByteArray → Srv → ℕ
  egressBound : ByteArray → Srv → ℕ
  κ : ByteArray → Srv → ℕ
  n_pos : ∀ θ s, 0 < n θ s
  r_pos : ∀ θ s, 0 < r θ s
  reads : ∀ θ s, (r θ s - 1) ^ k θ s * 2 ^ 129 ≤ r θ s ^ k θ s
  prf : (θ : ByteArray) → (s : Srv) → Pous.Bits (κ θ s) → Fin (n θ s) → Pous.Bits 512
  prover : (θ F c : ByteArray) → Game.Strategy (V.game θ F c) → (s : Srv) →
    Pous.StreamedWipe.Prover (n θ s) (k θ s) 512
      (Pous.StreamedWipe.serverBound (n θ s) (r θ s) (k θ s) (R θ s) + egressBound θ s) (512 * n θ s + R θ s)
      (egressBound θ s)
  finder : (θ F c : ByteArray) → Game.Strategy (V.game θ F c) → Srv → Claims.Finder
  loss : ByteArray → Srv → ℝ≥0∞ → ℝ≥0∞

namespace WipeVocab

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] {V : Vocab d α} {Srv : Type} (W : WipeVocab V Srv)

/-- **The wipe's verdict**: the circuit file passed the commit phase's check, NCI accepted the registration sampled
proofs committed to, and the wipe's component accepts what the prover sent. -/
def Verdict (V : Vocab d α) (o : Outcome) : Prop :=
  V.Registered o.params o.file ∧ RegistrationAccepts o.sent ∧ WipeAccepts o.sent

/-- Every release the wipe's check read is in the device's log; outside it, the release is a forgery of the device's
LMS signature, which the wipe's finder turns into a collision. -/
def Logged (o : Outcome) : Prop := ∀ r ∈ W.coinsRead o.sent, r ∈ o.log

/-- Server `s`'s assumed memory capacity, every tier counted: `M_s = 512 · n + R`. -/
def capacity (θ : ByteArray) (s : Srv) : ℕ := 512 * W.n θ s + W.R θ s

/-- Server `s`'s keep bound at its registered wipe: `B(n, r, k, R)` bits (`StreamedWipe.serverBound`). -/
def keepBound (θ : ByteArray) (s : Srv) : ℕ := Pous.StreamedWipe.serverBound (W.n θ s) (W.r θ s) (W.k θ s) (W.R θ s)

/-- **The wipe's claim at server `s`**: if the server's tiers hold at most its assumed capacity M_s, it keeps at most
`B(n, r, k, R)` bits of prior content across the wipe tick, plus its egress. -/
def Holds (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (s : Srv) (o : Outcome) : Prop :=
  (W.world θ F c σ o s).mem ≤ W.capacity θ s →
    (W.world θ F c σ o s).kept ≤ W.keepBound θ s + (W.world θ F c σ o s).egress

/-- **The wipe's claim at server `s` in the 5 GB form**: if the server's tiers hold at most M_s, it keeps at most 5 GB
(`4 · 10^10` bits) of prior content across the wipe tick, plus its egress. -/
def Holds5GB (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (s : Srv) (o : Outcome) : Prop :=
  (W.world θ F c σ o s).mem ≤ W.capacity θ s →
    (W.world θ F c σ o s).kept ≤ 40000000000 + (W.world θ F c σ o s).egress

/-- **The wipe's error at server `s` and a strategy**: `StreamedWipeMeetsServer`'s bound at the prover built from the
strategy (`2^-128` plus the PRF's advantage against the distinguisher built from it), plus the forged-release finder's
loss at its probability of a SHA-512 collision. -/
noncomputable def error (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (s : Srv) : ℝ≥0∞ :=
  (2⁻¹ : ℝ≥0∞) ^ 128 +
      Pous.StreamedWipe.prfAdvantage (W.prf θ s)
        (Pous.StreamedWipe.wipeDistinguisherBy Pous.StreamedWipe.Accepts (W.prover θ F c σ s)) +
    W.loss θ s (ENNReal.ofReal ((W.finder θ F c σ s).Pr sha512Msg))

/-- **The wipe's target (MEM-93)**: at all parameters and commitments, for every strategy and every server, the
probability that the wipe accepts while that server's claim fails is at most its error at that strategy. -/
def WipeSound : Prop :=
  ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (s : Srv),
    Game.prob (fun o => Verdict V o ∧ ¬ W.Holds θ F c σ s o) (V.game θ F c) σ ≤ W.error θ F c σ s

/-- **The wipe model's game realizes the causal game's wipe** at each server: the probability that the wipe accepts
while server `s` keeps more than its keep bound plus its egress is at most the probability that the wipe model accepts
the prover built from the strategy while it keeps its `B(n, r, k, R) + E` bits (`StreamedWipe.winRealBy` at `Accepts`),
plus the forged-release finder's loss. It is the coupling the memory challenger's piece proves (MEM-93), by
construction of `prover` from the strategy, as `NetworkCouples` is the certifier's. -/
def Couples : Prop :=
  ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (s : Srv),
    Game.prob (fun o => Verdict V o ∧ ¬ W.Holds θ F c σ s o) (V.game θ F c) σ ≤
      Pous.StreamedWipe.winRealBy Pous.StreamedWipe.Accepts (W.prf θ s) (W.prover θ F c σ s) +
        W.loss θ s (ENNReal.ofReal ((W.finder θ F c σ s).Pr sha512Msg))

/-- **Every server's registered wipe is at the 5 GB point** (`StreamedWipe.OperatingPoint`, `B(n, r, k, R) ≤ 4 · 10^10`).
The hypothesis of the 5 GB corollary only, never of `WipeSound`. -/
def AtOperatingPoint : Prop :=
  ∀ θ s, Pous.StreamedWipe.OperatingPoint (W.n θ s) (W.r θ s) (W.k θ s) (W.R θ s)

/-- **The wipe's honest lemma at the deployment**: at registered files, the deployment's honest prover meets every
server's wipe claim on almost every play (memory-accounting's honest wipe at the deployed parameters, through
`Couples`). -/
def HonestHolds : Prop :=
  ∀ θ F c : ByteArray, V.Registered θ F → ∀ s,
    Game.prob (fun o => ¬ W.Holds θ F c (V.honest θ F c) s o) (V.game θ F c) (V.honest θ F c) = 0

end WipeVocab

namespace Vocab

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] {Srv : Type} [Fintype Srv]

/-- **Soundness over the six subprotocols for the acceptance `acc`, in the causal game.** At all parameters and
commitments, and for every strategy, the probability that `acc` holds of what the prover sent for files that passed the
commit phase's check while one of the period's claims fails (one of `Vocab.Soundness`'s five, or some server's wipe
claim) is at most the errors at that strategy summed, the wipe's summed over the
servers. -/
def SoundnessWithWipeFor (V : Vocab d α) (W : WipeVocab V Srv) (acc : Sent → Prop) : Prop :=
  ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)),
    Game.prob (fun o => acc o.sent ∧ V.Registered o.params o.file ∧
        ((∃ a, ¬ V.Holds θ F c σ a o) ∨ ∃ s, ¬ W.Holds θ F c σ s o)) (V.game θ F c) σ
      ≤ (∑ a, V.error a θ F c σ) + ∑ s, W.error θ F c σ s

/-- **Soundness over the six subprotocols at onsite's compiled verifier** (`Accepts`). -/
def SoundnessWithWipe (V : Vocab d α) (W : WipeVocab V Srv) : Prop := V.SoundnessWithWipeFor W Accepts

end Vocab

end Verity
