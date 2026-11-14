module

public import Properties.Pous
public import Definitions.Core.InProofs
public import Properties.Verity.Sha512

/-!
# PoUS's setup inside proofs: accepted proofs leave at most `⌊B/100⌋` blocks that don't decode

In PoUS's setup checked inside proofs, `C*` and `W` never reach the verifier. Each of the `B` blocks is a proof block
(`u`, block `u % blocksPerSegment` of segment `u / blocksPerSegment`): it reads its block of `C*` and its slice of `W`
(the registered weights' row it reads, carried), and checks today's per-block check, `BlockDecodes`. The proof system
proves a live sample of `8,828` of them (`rate:8828/B`).

`SetupGives` states what an accepted setup gives, in `Verity.InProofs`'s shape at `K₀ = ⌊B/100⌋ + 1`: at most `⌊B/100⌋`
blocks of `C*` fail to decode to their slice of `W`, or the run exhibits a SHA-512 collision, except with probability `ε`.
That is `P2SlackFamilyFreeBlocks`' premise, at most `⌊B/100⌋` free blocks, so the space bound at `ρ_eff = 18/19 · 99/100`
follows by that guarantee, unchanged. Its premises, all named and all open:

* the proof system's half (`Verity.InProofs.Gives` at `K₀ = ⌊B/100⌋ + 1`): `SampledKSLiveV5` at `rate:8828/B`, whose
  sampling term is the chance a uniform `8,828`-subset misses `⌊B/100⌋ + 1` wrong blocks, at most `0.99^8828`; its
  condition is that each proof block's output is registered as known content `1`;
* each proof block refines `BlockDecodes` (`SetupUnitsRefine`);
* the carried slices (`CarriedW`): an accepted run's proof blocks open the registered `W`'s slices, or the run's SHA-512
  inputs hold a collision.
-/

@[expose] public section

namespace PousCheck.Properties.InProofs

open PousCheck.Properties Verity.InProofs Verity.Game
open scoped ENNReal

/-- The number of blocks, `B`. -/
def blocks (q : P2.Params) : Nat := q.segments * q.blocksPerSegment

/-- Block `u`'s slice of `W`: its `payloadBytes` bytes at offset `u · payloadBytes`, as `Decodes` reads it. -/
def slice (q : P2.Params) (W : ByteArray) (u : Nat) : ByteArray :=
  W.extract (u * q.payloadBytes) (u * q.payloadBytes + q.payloadBytes)

/-- **Today's per-block check on the values a proof block opens**: its block (`v.1`) decodes canonically under `salt` to
the payload it opens (`v.2`). -/
def BlockOk (q : P2.Params) (salt : ByteArray) (u : Fin (blocks q)) (v : ByteArray × ByteArray) : Prop :=
  BlockDecodes q salt (u.val / q.blocksPerSegment) (u.val % q.blocksPerSegment) v.1 v.2

/-- **The setup's proof blocks refine today's check**: every value a proof block's circuit accepts (`right u`) meets
`BlockDecodes`. -/
def SetupUnitsRefine (q : P2.Params) (salt : ByteArray)
    (right : Fin (blocks q) → ByteArray × ByteArray → Prop) : Prop :=
  Refines (V := fun _ => ByteArray × ByteArray) right (BlockOk q salt)

/-- **The carried slices open to `W`'s, or the run carries a collision**: whenever the verifier accepts, every proof
block opens its slice of the registered `W`, or the run's SHA-512 inputs (`hashInputs`) hold a collision. -/
def CarriedW {Ω : Type} (q : P2.Params) (W : ByteArray) (r : Run Ω (blocks q) fun _ => ByteArray × ByteArray)
    (hashInputs : Ω → List Hash.Msg) : Prop :=
  ∀ ω, r.accepts ω → (∀ u, (r.ports u ω).2 = slice q W u.val) ∨ Hash.CollisionIn Verity.sha512 (hashInputs ω)

open Classical in
/-- The blocks of `C*` the run opens that do not decode to their slice of `W`. -/
noncomputable def wrong {Ω : Type} (q : P2.Params) (salt W : ByteArray)
    (r : Run Ω (blocks q) fun _ => ByteArray × ByteArray) (ω : Ω) : Finset (Fin (blocks q)) :=
  Finset.univ.filter fun u => ¬ BlockOk q salt u ((r.ports u ω).1, slice q W u.val)

/-- **An accepted setup inside proofs leaves at most `⌊B/100⌋` blocks that don't decode, except `ε`.** For parameters
`q`, the server's salt, the registered weights `W` and a run `r` of the setup's `B` proof blocks: if the proof system
gives that an accepted run leaves fewer than `⌊B/100⌋ + 1` proof blocks whose values their circuit (`right`) rejects,
except with probability `ε`, the proof blocks refine `BlockDecodes`, and the carried slices open to `W`'s or the run
carries a collision, then the verifier accepts the run while more than `⌊B/100⌋` blocks of `C*` fail to decode to their
slice of `W` and the run's SHA-512 inputs hold no collision, with probability at most `ε`. -/
def SetupGives : Prop :=
  ∀ (q : P2.Params) (salt W : ByteArray) (Ω : Type) (μ : PMF Ω)
    (r : Run Ω (blocks q) fun _ => ByteArray × ByteArray) (right : Fin (blocks q) → ByteArray × ByteArray → Prop)
    (hashInputs : Ω → List Hash.Msg) (ε : ℝ≥0∞),
    Gives μ r right (blocks q / 100 + 1) ε →
    SetupUnitsRefine q salt right →
    CarriedW q W r hashInputs →
    prPMF μ (fun ω => r.accepts ω ∧ blocks q / 100 + 1 ≤ (wrong q salt W r ω).card ∧
        ¬ Hash.CollisionIn Verity.sha512 (hashInputs ω)) ≤ ε

end PousCheck.Properties.InProofs
