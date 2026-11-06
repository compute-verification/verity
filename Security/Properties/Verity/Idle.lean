module

public import Properties.Verity.Game
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
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
import Mathlib.Topology.Sheaves.Init

/-!
# An idle deployment: two instances of the names the subprotocols define, built by hand

The witnesses of `Verity.Soundness`, `Verity.RemoteSoundness` and `Verity.SoundnessOf` read these: a small hand-built
descriptor, not a parsed file. The device has no on-chip state, no memory, no ops and no bandwidth (`device`); the
circuit file is read as empty files under the netlist tags (`fileRead`); the coin commitment has no families, the device
draws nothing at the close and signs nothing (`signer`), the honest prover sends an empty request and closes the device
at once, no check reads its log, and there are no commitment openings and no gates computed;
the world charges nothing over an empty window (`world`); the schedule is one
empty block (`blocks`), and its wipe fills nothing and accepts nothing (`wipe`). Every finder and forger outputs nothing
on its one coin, and no bound reads a further finder. The circuit file's and the public file's digests in the parameters
are both the parameters' own SHA-512, so the committed files pass the commit phase's check exactly when one of them is
the parameters.

`vocab δ W` publishes the error `δ` for every subprotocol, the work certificate `W`, and no collision, forgery or loss
term. At `unbounded` (`δ = ⊤`) every subprotocol's error is `⊤`, so every target holds. At `exact` (`δ = 0`, `W = 1`)
every error is `0` while PoUW's claim, a gate computed, fails on every play.
-/

@[expose] public section

namespace Verity.Idle

open Accounting
open scoped ENNReal

noncomputable section

/-- **The device**: no on-chip state, no memory, no ops, no bandwidth, and one side channel that carries nothing. -/
def device : Device Unit := ⟨0, 0, fun _ => 0, 0, fun _ => 0⟩

/-- **The committed files as v2 reads them**: empty files under the netlist tags. -/
def fileRead : FileRead := ⟨Flock.Tags.netlistV1, .empty, [], .empty⟩

/-- The step that charges nothing. -/
def step : Step Unit Unit := ⟨fun _ _ => 0, fun _ => 0, fun _ _ => 0, fun _ => 0, fun _ => 0⟩

/-- **The world**: every resource on the one account, an empty window, the schedule that charges nothing, and every
bound on what the network certifiers saw met. -/
def world : World device Unit where
  acct _ := ()
  win := ⟨0, 0⟩
  schedule _ := step
  fits _ := by simp [Step.Fits, step, device]
  commWithin _ := True

/-- **The schedule**: one block, with no ticks and no gates. -/
def blocks : Verity.Blocks fileRead.circuit 1 where
  block _ := ⟨⟨0, 0⟩, ∅⟩
  ordered _ _ _ := Nat.le_refl 0
  disjoint _ _ _ := Finset.disjoint_empty_left _

/-- **The wipe at the one boundary**: it fills nothing, nothing leaves the device, and the challenger accepts
nothing. -/
def wipe : Verity.Wipe device blocks 0 Unit Unit where
  fill := 0
  fill_le := Nat.zero_le _
  egress := 0
  carry _ _ := 0
  filled σ _ := (σ, fun i => i.elim0)
  accepts _ _ _ := False
  after σ _ _ := σ

/-- **A run read at the boundary as the wipe's game**: no foreign content, the one seed and the one spot read. -/
def ofRun {O : Type} (G : Game O) : wipe.OfRun G where
  S_φ := 0
  hold _ := ⟨fun _ => false, fun _ => false⟩
  keep _ _ _ i := i.elim0
  view _ := (fun i => i.elim0, (), ())

/-- The collision finder that outputs nothing, on one coin, with no SHA-512 call. -/
def finder : Claims.Finder where
  S := Unit
  out _ := none
  cost _ := 0

/-- **The device's signer**: it signs nothing and draws nothing at the close. -/
def signer : Signer := ⟨fun _ => [], 0, fun _ _ _ _ => [], fun _ _ => []⟩

/-- The forger that saw no signature and outputs nothing, on one coin, with no SHA-512 call. -/
def forger : Hash.Assumptions.LmsForger where
  S := Unit
  key := []
  signed _ := []
  out _ := none
  cost _ := 0

/-- **The idle deployment's names**, publishing the error `δ` for every subprotocol and the work certificate `W`. -/
noncomputable def vocab (δ : ℝ≥0∞) (W : ℝ) : Vocab device Unit where
  read _ _ := fileRead
  place _ _ _ _ := none
  programDigest θ := some (sha512 θ.data.toList)
  publicDigest θ := some (sha512 θ.data.toList)
  registered θ := #[θ]
  families _ := []
  signer _ := signer
  coinsRead _ _ := none
  extract _ _ _ _ _ := 0
  finder _ _ _ _ _ := finder
  cost _ _ _ _ := 0
  callsMul _ _ := 0
  callsAdd _ _ := 0
  finder_calls _ _ _ _ _ _ := Nat.zero_le _
  finders _ _ _ _ := []
  forger _ _ _ _ _ := forger
  forger_calls _ _ _ _ _ _ := Nat.zero_le _
  signatures _ := 0
  signs_le _ _ _ _ _ := Nat.le_refl 0
  forger_signed _ _ _ _ _ _ := Nat.zero_le _
  computed _ _ _ _ _ := ∅
  world _ _ _ _ _ := world
  window _ := ⟨0, 0⟩
  capacity _ := ()
  world_window _ _ _ _ _ := rfl
  world_capacity _ _ _ _ _ := rfl
  theta _ _ := ∅
  theta_input _ _ _ h := absurd h (Finset.notMem_empty _)
  gates _ := 0
  fresh _ := 0
  spans _ _ := 1
  schedule _ _ := blocks
  boundary _ _ := 0
  Seed := Unit
  Spots := Unit
  wipe _ _ := wipe
  wipeRun θ F c _ := ofRun (play [] signer θ F c)
  foreignBits _ := 0
  honest θ F c := requestThen [] signer θ F c .empty ⟨.empty, fun _ => ⟨(.empty, #[]), ⟨⟩⟩⟩
  certOf _ := ⟨0, W, 0, 0, 0⟩
  crBound _ _ _ := 0
  delta _ _ _ := δ
  epsForge _ _ _ := 0
  loss _ _ _ := 0

/-- **The deployment at which every target holds**: every subprotocol's published error is `⊤`. -/
noncomputable def unbounded : Vocab device Unit := vocab ⊤ 1

/-- **The deployment whose errors are all `0`**, publishing a work certificate of one gate, which no play computes. -/
noncomputable def exact : Vocab device Unit := vocab 0 1

end

end Verity.Idle
