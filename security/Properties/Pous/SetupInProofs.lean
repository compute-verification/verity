module

public import Properties.Pous
public import Definitions.Core.InProofs
public import Properties.Verity.Sha512
public import Properties.Pous.InProofs

/-!
# PoUS's setup inside proofs: an accepted run leaves at most `⌊B/100⌋` blocks that don't decode

`SetupGives` holds for any run of the setup's proof blocks that meets its three hypotheses (`Gives` at the relation
`right`, `right ≤ BlockOk` and `CarriedW`); it says nothing of the deployed run until `SampledKSLiveV5` and proofs'
carried form give `Gives` and `CarriedW` there.

In PoUS's setup checked inside proofs, `C*` and `W` never reach the verifier. Each of the `B = q.blocks` blocks is a proof
block (`u`, block `u % blocksPerSegment` of segment `u / blocksPerSegment`): it reads its block of `C*` and its slice of `W`
(the registered weights' row it reads, carried), and checks today's per-block check, `BlockDecodes`. The proof system
proves a live sample of `8,828` of them (`rate:8828/B`).

`SetupGives`, in `Verity.InProofs`'s shape at `K₀ = ⌊B/100⌋ + 1`: if an accepted run leaves `⌊B/100⌋ + 1` proof blocks
whose values `right` rejects with probability at most `ε`, `right ≤ BlockOk`, and an accepted run's proof blocks hold the
registered `W`'s slices or its SHA-512 inputs hold a collision (`CarriedW`), then the verifier accepts while more than
`⌊B/100⌋` blocks of `C*` fail to decode to their slice of `W` and no collision is in `r.hashInputs`, with probability at
most `ε`. The guarantee that instantiates it builds `r.ports` from the counted extractor the sampled audit reads and
`r.hashInputs` from the run, sets `right` to the setup unit's circuit, and takes `Gives` from `SampledKSLiveV5` at
`rate:8828/B` under its condition that every proof block's output is registered as known content `1`; its sampling term,
the chance a uniform `8,828`-subset misses `⌊B/100⌋ + 1` wrong blocks, is at most `0.99^8828`.

Its gap to the space bound: at most `⌊B/100⌋` blocks that don't decode is `P2SlackFamilyFreeBlocks`' premise of at most
`⌊B/100⌋` free blocks only in that guarantee's model, the abstract `m1p` scheme (`Definitions/Pous/Guarantees/P2.lean`),
not the deployed P2 this lemma reads; nothing here states the step from one to the other.
-/

@[expose] public section

namespace PousCheck.Properties.InProofs

open PousCheck.Properties Verity.InProofs Verity.Game
open scoped ENNReal

/-- Block `u`'s slice of `W`: its `payloadBytes` bytes at offset `u · payloadBytes`, as `Decodes` reads it. -/
def slice (q : P2.Params) (W : ByteArray) (u : Nat) : ByteArray :=
  W.extract (u * q.payloadBytes) (u * q.payloadBytes + q.payloadBytes)

/-- **Today's per-block check on a proof block's values**: its block (`v.1`) decodes canonically under `salt` to the
payload `v.2`. -/
def BlockOk (q : P2.Params) (salt : ByteArray) (u : Fin q.blocks) (v : ByteArray × ByteArray) : Prop :=
  BlockDecodes q salt (u.val / q.blocksPerSegment) (u.val % q.blocksPerSegment) v.1 v.2

/-- **The carried slices are `W`'s, or the run carries a collision**: whenever the verifier accepts, every proof block
holds its slice of the registered `W`, or the run's SHA-512 inputs hold a collision. -/
def CarriedW {Ω : Type} (q : P2.Params) (W : ByteArray) (r : Run Ω q.blocks fun _ => ByteArray × ByteArray) : Prop :=
  ∀ ω, r.accepts ω → (∀ u, (r.ports u ω).2 = slice q W u.val) ∨ Hash.CollisionIn Verity.sha512 (r.hashInputs ω)

open Classical in
/-- The blocks of `C*` the run holds that do not decode to their slice of `W`. -/
noncomputable def wrong {Ω : Type} (q : P2.Params) (salt W : ByteArray)
    (r : Run Ω q.blocks fun _ => ByteArray × ByteArray) (ω : Ω) : Finset (Fin q.blocks) :=
  Finset.univ.filter fun u => ¬ BlockOk q salt u ((r.ports u ω).1, slice q W u.val)

/-- **An accepted setup leaves at most `⌊B/100⌋` blocks that don't decode, except `ε`, for any run meeting its
hypotheses**; it says nothing of the deployed run until `SampledKSLiveV5` and proofs' carried form give `Gives` and
`CarriedW` there. For parameters `q`, the server's salt, the registered weights `W` and a run `r` of the setup's
`q.blocks` proof blocks: if an accepted run leaves `⌊B/100⌋ + 1` proof blocks whose values `right` rejects with
probability at most `ε`, every value `right` accepts meets `BlockOk`, and `CarriedW` holds, then the verifier accepts the
run while more than `⌊B/100⌋` blocks of `C*` fail to decode to their slice of `W` and no collision is in `r.hashInputs`,
with probability at most `ε`. -/
def SetupGives : Prop :=
  ∀ (q : P2.Params) (salt W : ByteArray) (Ω : Type) (μ : PMF Ω)
    (r : Run Ω q.blocks fun _ => ByteArray × ByteArray) (right : Fin q.blocks → ByteArray × ByteArray → Prop)
    (ε : ℝ≥0∞),
    Gives μ r right (q.blocks / 100 + 1) ε →
    right ≤ BlockOk q salt →
    CarriedW q W r →
    prPMF μ (fun ω => r.accepts ω ∧ q.blocks / 100 + 1 ≤ (wrong q salt W r ω).card ∧
        ¬ Hash.CollisionIn Verity.sha512 (r.hashInputs ω)) ≤ ε

/-- **`SetupGives`' conclusion fails without its hypotheses**: at the registered point, a run that always accepts, holds
an empty block at each proof block and hashes nothing leaves every block wrong, more than `⌊B/100⌋`, so the event has
probability `1` and the bound fails at `ε = 0`. -/
def SetupGivesNontrivial : Prop :=
  ∃ (salt W : ByteArray) (v : Fin PousCheck.P2.Params.deployed.blocks → ByteArray × ByteArray),
    ¬ prPMF (PMF.pure ()) (fun ω => (MemoryChallenger.Properties.InProofs.constRun v).accepts ω ∧
        PousCheck.P2.Params.deployed.blocks / 100 + 1 ≤ (wrong PousCheck.P2.Params.deployed salt W (MemoryChallenger.Properties.InProofs.constRun v) ω).card ∧
        ¬ Hash.CollisionIn Verity.sha512 ((MemoryChallenger.Properties.InProofs.constRun v).hashInputs ω)) ≤ 0

end PousCheck.Properties.InProofs
