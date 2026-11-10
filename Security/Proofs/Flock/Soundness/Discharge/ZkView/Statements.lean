module

public import Definitions.Flock.ZK.SHVZK
public import Proofs.Flock.Soundness.Discharge.ZkView.Defs
public import Proofs.Flock.Soundness.Discharge.ZkView.Wire
import ArkLib.Data.CodingTheory.JohnsonBound.Family
import ArkLib.Data.CodingTheory.ReedSolomon.MutualCorrelatedAgreement.Johnson.Probability
import Batteries.Data.UInt
import Flock.Sizes
import Mathlib.Algebra.Order.Floor.Semifield
import Mathlib.Algebra.Polynomial.SpecificDegree
import Mathlib.Data.Nat.BitIndices
import Mathlib.Data.Nat.Size
import Mathlib.FieldTheory.Finite.Extension
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Proofs.Flock.Level3
import Proofs.Flock.Soundness.Accounting.Fast100
import Proofs.Flock.Soundness.Accounting.Numbers
import Proofs.Flock.Soundness.Audit.Flock
import Proofs.Flock.Soundness.Binding.E2E
import Proofs.Flock.Soundness.Binding.Weights
import Proofs.Flock.Soundness.CROnly
import Proofs.Flock.Soundness.CompiledAlign
import Proofs.Flock.Soundness.Defs
import Proofs.Flock.Soundness.Discharge.Hm.Mask
import Proofs.Flock.Soundness.Discharge.Hm.Slot
import Proofs.Flock.Soundness.Extract
import Proofs.Flock.Soundness.Game
import Proofs.Core.Game.Poly
import Proofs.Flock.Soundness.Knowledge
import Proofs.Flock.Soundness.LigeritoCoins
import Proofs.Flock.Soundness.LigeritoWeights
import Proofs.Flock.Soundness.LincheckPhase
import Proofs.Flock.Soundness.Lowering
import Proofs.Flock.Soundness.Merkle
import Proofs.Flock.Soundness.Model.Compiled
import Proofs.Flock.Soundness.OpeningPhase
import Proofs.Flock.Soundness.Refine.Arith
import Proofs.Flock.Soundness.Refine.Basis
import Proofs.Flock.Soundness.Refine.Final
import Proofs.Flock.Soundness.Refine.Lincheck
import Proofs.Flock.Soundness.Refine.Opening
import Proofs.Flock.Soundness.Refine.Plays
import Proofs.Flock.Soundness.Refine.Seam
import Proofs.Flock.Soundness.Refine.Zerocheck
import Proofs.Flock.Soundness.Registered.Binding
import Proofs.Flock.Soundness.Registered.E2E
import Proofs.Flock.Soundness.RepSound
import Proofs.Flock.Soundness.Rewinding
import Proofs.Flock.Soundness.Rounds
import Proofs.Flock.Soundness.SessionAcc
import Proofs.Flock.Soundness.Soundness
import Proofs.Flock.Soundness.SparseRows
import Proofs.Flock.Soundness.TmplFoldDef
import Proofs.Flock.Soundness.TmplFoldLayout
import Proofs.Flock.Soundness.TmplFoldPos
import Proofs.Flock.Soundness.TmplFoldRef
import Proofs.Flock.Soundness.TmplLayOk
import Proofs.Flock.Soundness.TmplSplitEq
import Proofs.Flock.Soundness.TmplSplitWalk
import Proofs.Flock.Soundness.TmplUnit
import Proofs.Flock.Soundness.TmplUnrollAt
import Proofs.Flock.Soundness.TmplWf
import Proofs.Flock.Soundness.Types.ProgramE2E
import Proofs.Flock.Soundness.ZK.Blocks
import Proofs.Flock.Soundness.ZK.Hiding
import Proofs.Flock.Soundness.ZK.RealLeaves
import Proofs.Flock.Soundness.ZK.Refusal
import Proofs.Flock.Soundness.ZK.SHVZK
import Proofs.Flock.Verifier.Transcript
import Std.Data.HashSet
import Std.Data.TreeMap.Raw.Lemmas
import VCVio.Prelude.Core

/-!
# Zero-knowledge at the `--zk` wire: the target statements

* `LeafSplit`: the executable's hm96-sha512 leaf of a 64-byte digest `x` under salt `y` is `H(x ⊕ M·y, c(y))`, with `M` the
  pinned key's Hankel matrix (`My`) and `c` the salt hash (`saltHash`). This is `Table.Hm96`'s structure at the executable's
  leaves, so Lemma B's and Lemma C's `Hm96Hiding` is `CROnly`'s, from `hash-derived-key` and `cr/sha-512` only.
* `HonestMsg`: the honest prover's messages carry the executable's pre-final leaves.
* `ZkWireFixed`: Lemma B at the executable's leaves, for any encoding `enc` of the session's views (the proof files): the
  real views' encoding and `S_shvzk`'s are within `∑ⱼ 2·|Hidⱼ|·hidingGap(My, saltHash)` both ways.
* `ZkWirePrefinal`: Lemma C at the executable's leaves: against any adaptive verifier, the honest prover's pre-final
  messages at witness `w` and at `w'` are within `2·|Leaves|·hidingGap(My, saltHash)`. Its `hP`, `he` and `hI` are this
  file's to discharge: `hP` by `pairProver`'s definition, `he` because `randOf` is a bijection, `hI` because
  `preWith` at the ideal leaves is `Table.preView`.
* `DecodeEncode`: `flock-verify verify --zk`'s parse inverts the file writer: `Zk.decode sh (encZk sh p) = .ok p` for every
  decoded proof of the shape.
-/

@[expose] public section

namespace FlockSoundness.Discharge.ZkView

open Flock ZK
open FlockSoundness.Game (prCoin)
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

/-- **The executable's leaf splits**: for a 64-byte digest `x` and salt `y`,
`Default512.leaf x y = SHA-512(lp ‖ (x ⊕ M·y) ‖ SHA-512(sp ‖ y))`. -/
def LeafSplit : Prop :=
  ∀ (x : ByteArray) (y : Salt), x.size = 64 →
    Hm96.Default512.leaf x (saltBytes y) = leafH (bitsOf x + My y, saltHash y)

/-- **The honest prover's messages are the executable's**: at randomness `x.1` and salts `x.2`, message `i` at answers `c`
is the round encoding of the session's pre-final view cut at `c`'s coins, every pre-final leaf the executable's leaf of
what it commits under its own salt. -/
def HonestMsg : Prop :=
  ∀ {K W Hi VL : Type} [Field K] {J n : ℕ} {Cn : Type} {zs : Fin J → Sizes}
    (T : (Fin n → Cn) → ∀ j, Coined K W Hi VL (zs j)) (out : ∀ c, (∀ j, (T c j).tab.Cut) → Fin n → ByteArray)
    (hc : Causal T out) (w : Fin J → W) (x : (∀ j, Base (zs j) Hi) × (Leaves zs → Salt)) (c : Fin n → Cn),
    (honest T out hc w).msg x c = msgsAt T out w x.1 c fun l => (T c l.1).exeLeaf (w l.1) (x.1 l.1) l.2 (x.2 l)

/-- **Lemma B at the `--zk` wire**: for the session's tables `T` at fixed coins, any encoding `enc` of their views (the
proof files the prover writes) and any event `E`, the encoding of the real views (every hidden leaf the executable's leaf
of its column under a uniform salt) and of `S_shvzk`'s output are within `∑ⱼ 2·|Hidⱼ|·hidingGap(My, saltHash)`, both ways. -/
def ZkWireFixed : Prop :=
  ∀ {K W Hi VL : Type} [Field K] [Fintype K] [DecidableEq K] [AddCommGroup Hi] [Fintype Hi] [DecidableEq Hi] {J : ℕ}
    (zs : Fin J → Sizes) (T : ∀ j, Coined K W Hi VL (zs j)) {Out : Type} (enc : (∀ j, (T j).tab.View) → Out)
    (_hrank : ∀ j, (T j).tab.Rank) (_hpad : ∀ j, (T j).tab.PadOnto)
    (_hW : ∀ j r, Function.Bijective fun p : 𝔽 × 𝔽 => (T j).tab.Wr r p.1 p.2) (_hpads : ∀ j, (T j).tab.PadsOnto)
    (_hβ : ∀ j r, (T j).tab.β r ≠ 0) (w : Fin J → W) (_hreg : ∀ j, (T j).tab.sReg (w j) = (T j).tab.pubReg)
    (_hinner : ∀ j, (T j).tab.InnerHolds (w j)) (w0 : Fin J → W) (E : Out → Prop),
    prCoin (fun p : ∀ j, (T j).tab.Rand × ((T j).tab.Hid → Salt) =>
        E (enc fun j => (T j).viewExe (w j) (p j).1 (p j).2))
        ≤ prCoin (fun p : ∀ j, (T j).tab.SimRand × ((T j).tab.Hid → Salt) =>
            E (enc fun j => (T j).simExe (w0 j) (p j).1 (p j).2))
          + ∑ j, 2 * Fintype.card (T j).tab.Hid * ENNReal.ofReal (CROnly.hidingGap My saltHash)
      ∧ prCoin (fun p : ∀ j, (T j).tab.SimRand × ((T j).tab.Hid → Salt) =>
          E (enc fun j => (T j).simExe (w0 j) (p j).1 (p j).2))
        ≤ prCoin (fun p : ∀ j, (T j).tab.Rand × ((T j).tab.Hid → Salt) =>
            E (enc fun j => (T j).viewExe (w j) (p j).1 (p j).2))
          + ∑ j, 2 * Fintype.card (T j).tab.Hid * ENNReal.ofReal (CROnly.hidingGap My saltHash)

/-- **Lemma C at the `--zk` wire**: the session's tables `T c` at the verifier's answers `c` (their sizes fixed), a round
encoding `out` that reads the answers before each round only, and any adaptive verifier `V` on a uniform tape: for any event
`G`, the honest prover's pre-final messages at witnesses `w` and `w'` (both satisfying the registered values) are within
`2·|Leaves|·hidingGap(My, saltHash)`. -/
def ZkWirePrefinal : Prop :=
  ∀ {K W Hi VL : Type} [Field K] [AddCommGroup Hi] [Fintype Hi] [DecidableEq Hi] {J n : ℕ} {Cn Tp : Type}
    [Fintype Tp] [Nonempty Tp] (zs : Fin J → Sizes) (T : (Fin n → Cn) → ∀ j, Coined K W Hi VL (zs j))
    (out : ∀ c, (∀ j, (T c j).tab.Cut) → Fin n → ByteArray) (hc : Causal T out) (w w' : Fin J → W)
    (_hreg : ∀ c j, (T c j).tab.sReg (w j) = (T c j).tab.pubReg)
    (_hreg' : ∀ c j, (T c j).tab.sReg (w' j) = (T c j).tab.pubReg)
    (V : Tp → Responder ByteArray Cn n) (G : Tp → (Fin n → ByteArray) → Prop),
    prCoin (fun q : Tp × ((∀ j, Base (zs j) Hi) × (Leaves zs → Salt)) => G q.1 (run (honest T out hc w) (V q.1) q.2))
      ≤ prCoin (fun q : Tp × ((∀ j, Base (zs j) Hi) × (Leaves zs → Salt)) =>
          G q.1 (run (honest T out hc w') (V q.1) q.2))
        + 2 * Fintype.card (Leaves zs) * ENNReal.ofReal (CROnly.hidingGap My saltHash)

/-- **The verifier reads what the prover writes**: `Zk.decode sh` of the file `encZk sh p` is `p`, for every decoded proof
`p` with the lengths the shape requires. -/
def DecodeEncode : Prop :=
  ∀ (sh : Shape) (p : Zk.ZkProof), Shaped sh p → Zk.decode sh (encZk sh p) = .ok p

end FlockSoundness.Discharge.ZkView
