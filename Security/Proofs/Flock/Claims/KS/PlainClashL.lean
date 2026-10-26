import Proofs.Flock.Claims.KS.PlainSimW
import Proofs.Flock.Claims.Recursion.PlainK

/-!
# Self-clashes past level 0

`SelfClashL` counts, past level 0, two verifying openings of one position to different rows against a rep's recorded
cap at a later level. On the deployed side such a clash is two `Opening`s of one position to different rows against
one root in the run's own log (`PlainLevelClash`), which collide there as at level 0 (`plainLevelClash_tied`, by
`tiedColl_of_openings`). So `PlainClashRun` follows from a raw clash at level 0 or at any later level
(`PlainRawClashRunL`, `plainClashRun_of_rawL`), of which `PlainRawClashRun` is the level-0 case
(`plainRawClashRunL_of_raw`).

What remains for a later level is the level-`ℓ` form of `plain_opening_raw`: a row an accepted run's rep opens at level
`ℓ ≥ 1` (`recursiveProofs[ℓ - 1]`'s rows, at the positions of that level's query round) is an `Opening` of level `ℓ`
against the rep's level-`ℓ` cap (`recursiveCaps[ℓ]`, the cap the custody check records) in the run's log; with it a
live self-clash at a later level gives `PlainRawClashRunL`'s middle case.
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Discharge.Tables Claims.Bridge.Live

/-- **Two openings of one position to different rows, against one root, in one log**, under a scheme `ms` at a level
`lv`. -/
def PlainLevelClash (ms : MerkleScheme) (lv : Flock.Level) (lanes : ℕ) (L : List (Claims.Msg × Claims.Msg)) : Prop :=
  ∃ (root : ByteArray) (q : ℕ) (row row' : Array F128), row ≠ row' ∧ Opening ms lv lanes root L q row ∧
    Opening ms lv lanes root L q row'

/-- **A clash at any level collides in the log.** -/
theorem plainLevelClash_tied {ms : MerkleScheme} {lv : Flock.Level} {lanes : ℕ} {L : List (Claims.Msg × Claims.Msg)}
    (h : PlainLevelClash ms lv lanes L) : Claims.TiedColl sha512 L := by
  obtain ⟨_, _, _, _, hne, h₁, h₂⟩ := h
  exact tiedColl_of_append_self (tiedColl_of_openings h₁ h₂ hne)

/-- **A live event at `P'`'s outcome gives an accepting run of `P` with a raw clash at some level, or a collision**:
two different columns its reps open at one level-0 position, two openings of one position to different rows against
one root at a later level, or a collision among the verifier's calls. -/
def PlainRawClashRunL {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (mPts : ℕ) (N : Claims.Msg → ℕ)
    (dg : ByteArray → ByteArray) (w : Claims.Msg → Wire 65536 1 mPts) {τ : Type} (P' : Claims.Prover (Ω × τ))
    (Φ : τ × (Fin (N x + 2) → LCoin 65536) → Fin ((V false).k x) → Coin) (pL : LiveOutJ 1 mPts → Prop) : Prop :=
  ∀ u r', pL (liveOut (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x (ω, u) r') →
    (Claims.run sha512 P (V false) x ω (Φ (u, r'))).out = .acc ∧
      ((∃ (q : ℕ) (a b : List FlockLevel3.GF128),
        (q, a) ∈ plainRawCols (Claims.run sha512 P (V false) x ω (Φ (u, r'))).seen ∧
        (q, b) ∈ plainRawCols (Claims.run sha512 P (V false) x ω (Φ (u, r'))).seen ∧ a ≠ b) ∨
        (∃ (ms : MerkleScheme) (lv : Flock.Level) (lanes : ℕ),
          PlainLevelClash ms lv lanes (Claims.run sha512 P (V false) x ω (Φ (u, r'))).vLog) ∨
        Claims.TiedColl sha512 (Claims.run sha512 P (V false) x ω (Φ (u, r'))).vLog)

/-- **A live event at `P'`'s outcome gives an accepting run of `P` with a raw clash at some level, or a collision**:
two different columns its reps open at one level-0 position, two openings of one position to different rows against
one root at a later level, or a collision among the verifier's calls. -/
def PlainRawClashRunLK (keep : Bool → Bool) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (mPts : ℕ) (N : Claims.Msg → ℕ)
    (dg : ByteArray → ByteArray) (w : Claims.Msg → Wire 65536 1 mPts) {τ : Type} (P' : Claims.Prover (Ω × τ))
    (Φ : τ × (Fin (N x + 2) → LCoin 65536) → Fin ((V false).k x) → Coin) (pL : LiveOutJ 1 mPts → Prop) : Prop :=
  ∀ u r', pL (liveOut (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x (ω, u) r') →
    (Claims.run sha512 P (VK keep) x ω (Φ (u, r'))).out = .acc ∧
      ((∃ (q : ℕ) (a b : List FlockLevel3.GF128),
        (q, a) ∈ plainRawCols (Claims.run sha512 P (VK keep) x ω (Φ (u, r'))).seen ∧
        (q, b) ∈ plainRawCols (Claims.run sha512 P (VK keep) x ω (Φ (u, r'))).seen ∧ a ≠ b) ∨
        (∃ (ms : MerkleScheme) (lv : Flock.Level) (lanes : ℕ),
          PlainLevelClash ms lv lanes (Claims.run sha512 P (VK keep) x ω (Φ (u, r'))).vLog) ∨
        Claims.TiedColl sha512 (Claims.run sha512 P (VK keep) x ω (Φ (u, r'))).vLog)

/-- **The clash at any level gives the collision**, so `PlainRawClashRunL` gives `PlainClashRun`. -/
theorem plainClashRun_of_rawLK (keep : Bool → Bool) {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {mPts : ℕ}
    {N : Claims.Msg → ℕ} {dg : ByteArray → ByteArray} {w : Claims.Msg → Wire 65536 1 mPts} {τ : Type}
    {P' : Claims.Prover (Ω × τ)} {Φ : τ × (Fin (N x + 2) → LCoin 65536) → Fin ((V false).k x) → Coin}
    {pL : LiveOutJ 1 mPts → Prop} (h : PlainRawClashRunLK keep P x ω mPts N dg w P' Φ pL) :
    PlainClashRunK keep P x ω mPts N dg w P' Φ pL := fun u r' hp => by
  obtain ⟨hacc, ⟨q, a, b, ha, hb, hab⟩ | ⟨_, _, _, hc⟩ | ht⟩ := h u r' hp
  · exact ⟨hacc, plainK_selfClash_tied keep hacc ha hb hab⟩
  · exact ⟨hacc, plainLevelClash_tied hc⟩
  · exact ⟨hacc, ht⟩

/-- `plainClashRun_of_rawLK` at the deployed plain verifier (`keep := id`). -/
theorem plainClashRun_of_rawL {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {mPts : ℕ}
    {N : Claims.Msg → ℕ} {dg : ByteArray → ByteArray} {w : Claims.Msg → Wire 65536 1 mPts} {τ : Type}
    {P' : Claims.Prover (Ω × τ)} {Φ : τ × (Fin (N x + 2) → LCoin 65536) → Fin ((V false).k x) → Coin}
    {pL : LiveOutJ 1 mPts → Prop} (h : PlainRawClashRunL P x ω mPts N dg w P' Φ pL) :
    PlainClashRun P x ω mPts N dg w P' Φ pL :=
  plainClashRun_of_rawLK id (Ω := Ω) (P := P) (x := x) (ω := ω) (mPts := mPts) (N := N) (dg := dg) (w := w) (τ := τ) (P' := P') (Φ := Φ) (pL := pL) (h := h)

/-- **The level-0 raw clash is a raw clash at some level.** -/
theorem plainRawClashRunL_of_raw {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {mPts : ℕ}
    {N : Claims.Msg → ℕ} {dg : ByteArray → ByteArray} {w : Claims.Msg → Wire 65536 1 mPts} {τ : Type}
    {P' : Claims.Prover (Ω × τ)} {Φ : τ × (Fin (N x + 2) → LCoin 65536) → Fin ((V false).k x) → Coin}
    {pL : LiveOutJ 1 mPts → Prop} (h : PlainRawClashRun P x ω mPts N dg w P' Φ pL) :
    PlainRawClashRunL P x ω mPts N dg w P' Φ pL := fun u r' hp => by
  obtain ⟨hacc, hc | ht⟩ := h u r' hp
  · exact ⟨hacc, .inl hc⟩
  · exact ⟨hacc, .inr (.inr ht)⟩

end FlockDeploy.KS
