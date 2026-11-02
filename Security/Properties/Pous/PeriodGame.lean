import Properties.Pous.PeriodAt
import Definitions.Hash.Assumptions
import Definitions.Pous.Guarantees.Obligation

/-!
# PoUS over one period in the shared coin game

`PeriodHeld` and `PeriodHeldAt` bound the period's event in the ring model: over the primitive `ω`, the memory
challenger's hm96 coins and its streams. The shared coin game (`Verity.Game`) has no primitive: its coins are uniform
bytes. The event never reads `ω` once the session, the device record and the schedule don't (`ReadsNoPrimitive`: the
run comes from the coin game, which has no `ω`), so `ω` averages out exactly, and the game reads the bound at any `ω`
with no premise. The certificate's fit (`Fits`) and the 105-round bound (`Pous.Guarantees.RoundsBound`) are
`PeriodHeld`'s and `PeriodHeldAt`'s, named. What the game's coins are decides the rest:

* `PeriodHeldCoins`, `PeriodHeldCoinsAt`: the coins are the challenger's hm96 coins and its streams, uniform bytes. The
  event's probability over them is at most `δ + ε_max`, the ring model's bound, with no premise on SHA-512.
* `PeriodHeldSha512`, `PeriodHeldSha512At`: the coins are the challenger's hm96 coins and its 64-byte secret source, and
  its streams are the deployed derivation's (`challengerStreams`: `derive(source, COIN_DOMAIN, {session, blocks, k})`,
  then `stream(key, ("u", j), DRAW_BYTES)` for each round `j`, `MemoryChallenger.drawIndices`'s). Under
  `Hash.Assumptions.Sha512Prf` (`prf/sha-512`) at the two tests the event gives, the key derivation (one input) and the
  streams' 105 · 64 blocks, the bound is `δ + ε_max + ε_prf(t, 1) + ε_prf(t, 6720)`.

The session id is fixed for the period (`Audits.sid`, the one the auditor designated), before the source. The deployed
auditor derives it from the same coins (`pous.period`, `challenger_party`): reading that would make the coin key's input
depend on an earlier answer, an adaptive query, which `Sha512Prf` (non-adaptive) doesn't cover.
-/

namespace Pous.Period

open Pous.EndToEnd PousCheck PousCheck.Properties MemoryChallenger

open scoped ENNReal

set_option exponentiation.threshold 20000

/-! ## The memory challenger's derivation, on SHA-512 -/

/-- SHA-512 of a message, as a digest: the verifier's FIPS 180-4 SHA-512 (`Flock.Sha512.hash`) on its bytes. -/
def sha512Digest (m : Hash.Msg) : Hash.Digest := fun i => (Flock.Sha512.hash ⟨m.toArray⟩).data.getD i.val 0

/-- The memory challenger's coin context at `N` blocks for the session `sid` (`drawIndices`'s). -/
def coinContext (N : ℕ) (sid : ByteArray) : List (String × FlockSoundness.Randomness.Part) :=
  [("session", .bytes sid), ("blocks", .int N), ("k", .int K)]

/-- What `derive` hashes before a 64-byte source: its tag and the source's frame header, 37 bytes. -/
def derivePre : Hash.Msg :=
  (FlockSoundness.Randomness.DERIVE ++ ⟨#[98]⟩ ++ FlockSoundness.Randomness.be8 64).data.toList

/-- What `derive` hashes after the source for the memory challenger's coin key at `N` blocks and session `sid`: the
domain's frame and the context's, sorted by key (`derive`'s `ctx`). -/
def deriveTail (N : ℕ) (sid : ByteArray) : Hash.Msg :=
  let ctx := (((coinContext N sid).toArray.qsort fun a b => a.1 < b.1).toList.foldl
    (fun acc kv => acc ++ FlockSoundness.Randomness.frame (.str kv.1) ++ FlockSoundness.Randomness.frame kv.2)
    ByteArray.empty)
  (FlockSoundness.Randomness.frame (.str COIN_DOMAIN) ++ FlockSoundness.Randomness.frame (.bytes ctx)).data.toList

/-- The key `derive` returns from its digest: the first 32 bytes. -/
def keyOf (d : Hash.Digest) : ByteArray := (⟨Array.ofFn d⟩ : ByteArray).extract 0 32

/-- The memory challenger's streams under `key`: stream `j` is `stream(key, ("u", j), DRAW_BYTES)`, the one its
`uniform(blocks, j)` reads (`Randomness.uniform`). -/
def streamsOf (key : ByteArray) : Streams := fun j b =>
  ((FlockSoundness.Randomness.stream key [.str "u", .int j.val] DRAW_BYTES).data.getD b.val 0).toFin

/-- The memory challenger's streams from its source `src` at `N` blocks for the session `sid`: the key
`derive(src, COIN_DOMAIN, coinContext N sid)`, and its streams (`drawIndices`). -/
def challengerStreams (N : ℕ) (sid : ByteArray) (src : Bytes 64) : Streams :=
  streamsOf (FlockSoundness.Randomness.derive (bytesOf src) COIN_DOMAIN (coinContext N sid))

/-- What block `c` of stream `j` hashes after the key: the index's frames and the counter, 37 bytes (`Randomness.block`
at the index `("u", j)`). -/
def streamInputs (jc : Fin 105 × Fin 64) : Hash.Msg :=
  (FlockSoundness.Randomness.frames [.str "u", .int jc.1.val] ++ FlockSoundness.Randomness.be8 jc.2.val).data.toList

/-- Streams laid out from blocks in counter mode: byte `b` of stream `j` is byte `b mod 64` of block `(j, b / 64)`. -/
def streamsOfBlocks (U : Fin 105 × Fin 64 → Hash.Digest) : Streams := fun j b =>
  (U (j, ⟨b.val / 64, by have := b.isLt; unfold DRAW_BYTES at this; omega⟩) ⟨b.val % 64, by omega⟩).toFin

/-! ## The premises the coin game adds, named -/

/-- **The certificate fits the state bound `S₀`**: `S · len + S_chip + ∑ cap_i · round ≤ S₀`, the capacity account's
`S` bits a tick over a window of `len` ticks beside the device's on-chip state and its side channels' capacity over
`round` ticks (`PeriodHeld`'s fit). -/
def Fits {ι : Type} [Fintype ι] (d : Accounting.Device ι) (len round : ℕ) (S S₀ : ℝ) : Prop :=
  S * len + d.chip + ((∑ i, d.cap i * round : ℕ) : ℝ) ≤ S₀

/-- **`f` doesn't read the primitive** at the other audits' coins `o`: its value at `(ω, y)` is its value at `(ω', y)`
for every two primitives `ω, ω'`, as in a run of the shared coin game, which has none. -/
def ReadsNoPrimitive {ΩW ΩQ Ω Y X : Type} (f : ΩW × ΩQ × (Ω × Y) → X) (o : ΩW × ΩQ) : Prop :=
  ∀ (ω ω' : Ω) (y : Y), f (o.1, o.2, (ω, y)) = f (o.1, o.2, (ω', y))

/-! ## The event, at the deployed point and at any point -/

/-- `PeriodHeld`'s event at the coins `x`: the verifier's whole decision accepts with no break in the period's own data
while the capacity account held fewer than `S · len` bit-ticks over the capacity window. -/
def AcceptedShort {ι : Type} [Fintype ι] {d : Accounting.Device ι} {α ΩW ΩQ : Type} [Fintype α]
    (r : Accounting.Run d α ΩW ΩQ ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams))
    (P : Audits ΩW ΩQ) (S : ℝ) (x : ΩW × ΩQ × ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams)) :
    Prop :=
  P.Accepts x ∧ ¬ P.Break x ∧ (r.held x : ℝ) < S * (r.win .capacity).len

/-- `PeriodHeldAt`'s event at the point `pt` and the coins `x`. -/
def AcceptedShortAt (pt : Run.Point) (N : ℕ) [NeZero N] {ι : Type} [Fintype ι] {d : Accounting.Device ι}
    {α ΩW ΩQ : Type} [Fintype α]
    (r : Accounting.Run d α ΩW ΩQ ((ringRO N 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams))
    (P : AuditsAt N ΩW ΩQ) (S : ℝ) (x : ΩW × ΩQ × ((ringRO N 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams)) :
    Prop :=
  P.Accepts pt x ∧ ¬ P.Break pt x ∧ (r.held x : ℝ) < S * (r.win .capacity).len

/-! ## The coin game's bounds -/

/-- **PoUS's per-period guarantee in the coin game, at uniform streams.** Under `PeriodHeld`'s premises (its fit
`Fits`), at every value `o` of the other audits' coins where the session, the device record and the schedule don't read
the primitive (`ReadsNoPrimitive`: a run of the coin game, which has none), at every primitive `ω`, the verifier's whole
decision accepts the period with no break while the capacity account held fewer than `S · len` bit-ticks with
probability at most `δ + ε_max` over the memory challenger's hm96 coins and its streams, uniform. -/
def PeriodHeldCoins : Prop :=
  ∀ D Q L : ℕ, Pous.P2.Assumptions.P2RingAuditSecure D Q L →
    ∀ (ι α ΩW ΩQ : Type) [Fintype ι] [Fintype α] (d : Accounting.Device ι)
      (r : Accounting.Run d α ΩW ΩQ ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams))
      (P : Audits ΩW ΩQ) (τ : Strategy ΩW ΩQ) (round : ℕ) (S : ℝ),
      DeviceModel r P τ D Q L round → Instantiates P τ → Fits d (r.win .capacity).len round S Pous.EndToEnd.S →
        ∀ o : ΩW × ΩQ, ReadsNoPrimitive P.session o → ReadsNoPrimitive P.device o → ReadsNoPrimitive r.schedule o →
          ∀ ω : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω,
            Verity.Game.prCoin (fun y : HmCoins × Streams => AcceptedShort r P S (o.1, o.2, (ω, y))) ≤
              ENNReal.ofReal Params.δ + Params.εMax

/-- **PoUS's per-period guarantee in the coin game, at the deployed challenger's SHA-512 streams.** `PeriodHeldCoins`
with the memory challenger's streams derived from its 64-byte secret source (`challengerStreams` at `2^19` blocks and
the designated session): given `Sha512Prf` (`prf/sha-512`) with advantage `ε` at the test's time `t`, at its two tests
(the coin key, `derive`'s one input after the source; the streams' blocks, 105 · 64 inputs after the key), the verifier's
whole decision accepts the period with no break while the capacity account held fewer than `S · len` bit-ticks with
probability at most `δ + ε_max + ε(t, 1) + ε(t, 6720)` over the source and the hm96 coins. -/
def PeriodHeldSha512 : Prop :=
  ∀ D Q L : ℕ, Pous.P2.Assumptions.P2RingAuditSecure D Q L →
    ∀ (ι α ΩW ΩQ : Type) [Fintype ι] [Fintype α] (d : Accounting.Device ι)
      (r : Accounting.Run d α ΩW ΩQ ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams))
      (P : Audits ΩW ΩQ) (τ : Strategy ΩW ΩQ) (round : ℕ) (S : ℝ),
      DeviceModel r P τ D Q L round → Instantiates P τ → Fits d (r.win .capacity).len round S Pous.EndToEnd.S →
        ∀ o : ΩW × ΩQ, ReadsNoPrimitive P.session o → ReadsNoPrimitive P.device o → ReadsNoPrimitive r.schedule o →
          ∀ (ω : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω) (ε : ℕ → ℕ → ℝ≥0∞) (t : ℕ),
            Hash.Assumptions.Sha512Prf sha512Digest ε derivePre 64 (fun _ : Unit => deriveTail (2 ^ 19) P.sid)
              (fun (c : HmCoins) k => AcceptedShort r P S (o.1, o.2, (ω, c, streamsOf (keyOf (k ()))))) t →
            Hash.Assumptions.Sha512Prf sha512Digest ε FlockSoundness.Randomness.STREAM.data.toList 32
              streamInputs (fun (c : HmCoins) U => AcceptedShort r P S (o.1, o.2, (ω, c, streamsOfBlocks U))) t →
              Verity.Game.prCoin (fun z : Bytes 64 × HmCoins =>
                  AcceptedShort r P S (o.1, o.2, (ω, z.2, challengerStreams (2 ^ 19) P.sid z.1))) ≤
                ENNReal.ofReal Params.δ + Params.εMax + ε t 1 + ε t (105 * 64)

/-- **`PeriodHeldAt` in the coin game, at uniform streams**: `PeriodHeldCoins` at every point `pt` and block count
`2^19 = 524,288 ≤ N ≤ 134,217,728 = 2^27`, under `PeriodHeldAt`'s premises (its 105-round bound
`Pous.Guarantees.RoundsBound`). -/
def PeriodHeldCoinsAt : Prop :=
  ∀ (p ε₀ : ℝ≥0∞) (D Q L : ℕ),
    Pous.Guarantees.RoundsBound p ε₀ 105 (ENNReal.ofReal Params.δ + ((2⁻¹ : ℝ≥0∞) ^ 129 + (2⁻¹ : ℝ≥0∞) ^ 130)) →
    Pous.P2.Assumptions.P2RingRoundSecureWide p ε₀ D Q L →
      ∀ (pt : Run.Point) (N : ℕ) [NeZero N], 524288 ≤ N → N ≤ 134217728 →
        ∀ (ι α ΩW ΩQ : Type) [Fintype ι] [Fintype α] (d : Accounting.Device ι)
          (r : Accounting.Run d α ΩW ΩQ ((ringRO N 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams))
          (P : AuditsAt N ΩW ΩQ) (round : ℕ) (S : ℝ),
          RingDeviceAt N pt r P D Q L round → Fits d (r.win .capacity).len round S (SAt N) →
            ∀ o : ΩW × ΩQ, ReadsNoPrimitive P.session o → ReadsNoPrimitive P.device o → ReadsNoPrimitive r.schedule o →
              ∀ ω : (ringRO N 16448 (2 ^ 16448 - 21065)).Ω,
                Verity.Game.prCoin (fun y : HmCoins × Streams => AcceptedShortAt pt N r P S (o.1, o.2, (ω, y))) ≤
                  ENNReal.ofReal Params.δ + Params.εMax

/-- **`PeriodHeldAt` in the coin game, at the deployed challenger's SHA-512 streams**: `PeriodHeldSha512` at every point
`pt` and block count `2^19 = 524,288 ≤ N ≤ 134,217,728 = 2^27`, the streams derived at `N` blocks, under
`PeriodHeldAt`'s premises, with bound `δ + ε_max + ε(t, 1) + ε(t, 6720)`. -/
def PeriodHeldSha512At : Prop :=
  ∀ (p ε₀ : ℝ≥0∞) (D Q L : ℕ),
    Pous.Guarantees.RoundsBound p ε₀ 105 (ENNReal.ofReal Params.δ + ((2⁻¹ : ℝ≥0∞) ^ 129 + (2⁻¹ : ℝ≥0∞) ^ 130)) →
    Pous.P2.Assumptions.P2RingRoundSecureWide p ε₀ D Q L →
      ∀ (pt : Run.Point) (N : ℕ) [NeZero N], 524288 ≤ N → N ≤ 134217728 →
        ∀ (ι α ΩW ΩQ : Type) [Fintype ι] [Fintype α] (d : Accounting.Device ι)
          (r : Accounting.Run d α ΩW ΩQ ((ringRO N 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams))
          (P : AuditsAt N ΩW ΩQ) (round : ℕ) (S : ℝ),
          RingDeviceAt N pt r P D Q L round → Fits d (r.win .capacity).len round S (SAt N) →
            ∀ o : ΩW × ΩQ, ReadsNoPrimitive P.session o → ReadsNoPrimitive P.device o → ReadsNoPrimitive r.schedule o →
              ∀ (ω : (ringRO N 16448 (2 ^ 16448 - 21065)).Ω) (ε : ℕ → ℕ → ℝ≥0∞) (t : ℕ),
                Hash.Assumptions.Sha512Prf sha512Digest ε derivePre 64 (fun _ : Unit => deriveTail N P.sid)
                  (fun (c : HmCoins) k => AcceptedShortAt pt N r P S (o.1, o.2, (ω, c, streamsOf (keyOf (k ()))))) t →
                Hash.Assumptions.Sha512Prf sha512Digest ε FlockSoundness.Randomness.STREAM.data.toList 32
                  streamInputs
                  (fun (c : HmCoins) U => AcceptedShortAt pt N r P S (o.1, o.2, (ω, c, streamsOfBlocks U))) t →
                  Verity.Game.prCoin (fun z : Bytes 64 × HmCoins =>
                      AcceptedShortAt pt N r P S (o.1, o.2, (ω, z.2, challengerStreams N P.sid z.1))) ≤
                    ENNReal.ofReal Params.δ + Params.εMax + ε t 1 + ε t (105 * 64)

end Pous.Period
