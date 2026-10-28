import Properties.Verity.Game

/-!
# An idle deployment: two instances of the names the subprotocols define, built by hand

The witnesses of `Verity.Soundness`, `Verity.RemoteSoundness` and `Verity.SoundnessOf` read these: a small hand-built
descriptor, not a parsed file. The device has no on-chip state, no memory, no ops and no bandwidth (`device`); the
committed blob is read as empty files under the netlist tags (`fileRead`); there are no device rounds, no releases, no
commitment openings and no gates computed; the world charges nothing over an empty window (`world`); the schedule is one
empty block (`blocks`), and its wipe fills nothing and accepts nothing (`wipe`). Every finder and forger outputs nothing
on its one coin. The circuit file's digest in the parameters is the parameters' own SHA-512, so the files pass the commit
phase's check exactly when they are the parameters.

`vocab δ W` publishes the error `δ` for every subprotocol, the work certificate `W`, and no collision, forgery or loss
term. At `unbounded` (`δ = ⊤`) every subprotocol's error is `⊤`, so every target holds. At `exact` (`δ = 0`, `W = 1`)
every error is `0` while PoUW's claim, a gate computed, fails on every play.
-/

namespace Verity.Idle

open Accounting
open scoped ENNReal

/-- **The device**: no on-chip state, no memory, no ops, no bandwidth, and one side channel that carries nothing. -/
def device : Device Unit := ⟨0, 0, fun _ => 0, 0, fun _ => 0⟩

/-- **The committed blob as v2 reads it**: empty files under the netlist tags. -/
def fileRead : FileRead := ⟨Flock.netlistV1, .empty, [], .empty⟩

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

/-- The forger that saw no signature and outputs nothing, on one coin, with no SHA-512 call. -/
def forger : Hash.Assumptions.LmsForger where
  S := Unit
  key := []
  signed := []
  out _ := none
  cost _ := 0

/-- **The idle deployment's names**, publishing the error `δ` for every subprotocol and the work certificate `W`. -/
noncomputable def vocab (δ : ℝ≥0∞) (W : ℝ) : Vocab device Unit where
  read _ := fileRead
  place _ _ _ _ := none
  programDigest θ := some (sha512 θ.data.toList)
  registered θ := θ
  rounds _ := []
  coinsRead _ _ := []
  extract _ _ _ _ _ := 0
  finder _ _ _ _ _ := finder
  cost _ _ _ _ := 0
  callsMul _ _ := 0
  callsAdd _ _ := 0
  finder_calls _ _ _ _ _ _ := Nat.zero_le _
  forger _ _ _ _ _ := forger
  forger_calls _ _ _ _ _ _ := Nat.zero_le _
  signatures _ := 0
  forger_signed _ _ _ _ _ := Nat.zero_le _
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
  spans _ := 1
  schedule _ := blocks
  boundary _ := 0
  Seed := Unit
  Spots := Unit
  wipe _ _ := wipe
  wipeRun θ F c _ := ofRun (play [] θ F c)
  foreignBits _ := 0
  honest _ _ _ := ⟨.empty, ⟨⟩⟩
  certOf _ := ⟨0, W, 0, 0, 0⟩
  crBound _ _ _ := 0
  delta _ _ _ := δ
  epsForge _ _ _ := 0
  loss _ _ _ := 0

/-- **The deployment at which every target holds**: every subprotocol's published error is `⊤`. -/
noncomputable def unbounded : Vocab device Unit := vocab ⊤ 1

/-- **The deployment whose errors are all `0`**, publishing a work certificate of one gate, which no play computes. -/
noncomputable def exact : Vocab device Unit := vocab 0 1

end Verity.Idle
