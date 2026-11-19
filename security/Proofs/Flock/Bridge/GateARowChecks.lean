module

public import Proofs.Flock.Bridge.GateARow
public import Proofs.Flock.SpotFlock.InstanceStmt.Scope
public import Proofs.Flock.SpotFlock.InstanceStmt.SetupAt
public import Proofs.Flock.Soundness.Discharge.Hidden.Built
public import Proofs.Properties.Flock.Nets

/-!
# `GateARow` at a file that parses

The recursion's instance statement (`SpotFlock.InstanceStmt`): its circuit file at the verifier's two fixed nets and its
public file at the plain run whose three strings are empty parse (`parsedOf_x`), to one unit a VU over two units
(`n_x`), each with at least one row (`m_x`). So `GateARow` is not vacuous: at these files unit `0` of the game's
circuit has as many gates as the file has rows a unit, and at least one (`gateARow_x`).
-/

@[expose] public section

namespace Flock.Bridge.GateARowChecks

open Flock Flock.ZkServer RecDeploy.VStarNets SpotFlock.InstanceStmt
open Properties.Flock.Nets (fixedNetsBuild)

/-- The plain run whose three strings are empty. -/
def run0 : Run := ⟨"", "", ""⟩

theorem run0_plain : run0.Plain where
  set := by simp [run0, plainStr]
  bindI := by simp [run0, plainStr]
  bindO := by simp [run0, plainStr]
  size := by decide

/-- The instance statement's circuit file at the fixed nets. -/
abbrev cfX : ByteArray := bytesOf (HmNets.shaLaid ()) (HmNets.hmLaid ())

/-- Its public file at `run0`. -/
abbrev pfX : ByteArray := pubOf run0 (hexOf circuitHash) (hexOf classHash)

/-- What the public file loads to. -/
noncomputable abbrev pubX :=
  pubO run0 (HmNets.shaLaid ()) (HmNets.hmLaid ()) circuitHash classHash AF BF AF' BF' s1X (s3X (hexOf classHash))
    (s4X run0 (hexOf circuitHash) (hexOf classHash))

theorem parse_x : Flock.Desc.parse tz cfX [] = .ok (.single cF) :=
  Flock.Desc.parse_single (parseOut_x fixedNetsBuild)

theorem load_x : Flock.HmOut.loadPublic tz (Flock.Desc.single cF).unroll pfX = .ok pubX := by
  have hCe : classHash.isEmpty = false := classOfB_isEmpty (HmNets.shaLaid ()) (HmNets.hmLaid ())
  exact load_pubO run0 (HmNets.shaLaid ()) (HmNets.hmLaid ()) circuitHash classHash AF BF AF' BF' s1X
    (s3X (hexOf classHash)) (s4X run0 (hexOf circuitHash) (hexOf classHash)) run0_plain hCe (hu1_of _)
    (hu2_of _ _ _ _ _ _ _ _ _ hCe) (hd_of _ _ _)

theorem built_x : FlockSoundness.Built (Flock.Desc.single cF).unroll.unit :=
  FlockSoundness.Discharge.Hidden.built_of_parseStmt (cf := cfX) (tables := []) (by
    unfold Flock.Desc.parseStmt
    rewrite [ite_eq_right (by rw [tmpl_tzX]; decide)]
    exact parse_x) typed_tzX

/-- **The instance statement's files parse.** -/
theorem parsedOf_x : parsedOf tz cfX [] pfX = some ⟨.single cF, built_x, pubX.n⟩ := by
  unfold parsedOf
  rw [parse_x]
  dsimp only
  rw [load_x]
  dsimp only
  rw [dif_pos built_x]

theorem n_x : pubX.n = 2 := by decide +kernel

theorem m_x : 0 < (Spot.FilePart.ofCircuit (Flock.Desc.single cF).unroll pubX.n).m := by
  have h1 : cF.upv = 1 := cO_upv _ _ _ _ _ _ _ _ _
  show 0 < cF.upv * _
  rw [h1]
  omega

/-- **`GateARow` says something**: at the instance statement's files, which parse to two units, unit `0` of the game's
circuit has as many gates as the file has rows a unit, and that is at least one. -/
theorem gateARow_x :
    ∃ u : Fin (unitsOfFile tz cfX [] pfX),
      (circuitOfFile tz cfX [] pfX).size ((partsOfFile tz cfX [] pfX).gates u) =
        (Spot.FilePart.ofCircuit (Flock.Desc.single cF).unroll pubX.n).m ∧
      0 < (circuitOfFile tz cfX [] pfX).size ((partsOfFile tz cfX [] pfX).gates u) := by
  have hn : 0 < unitsOfFile tz cfX [] pfX := by
    show 0 < Parsed.units (parsedOf tz cfX [] pfX)
    rw [parsedOf_x]
    show 0 < pubX.n
    rw [n_x]
    decide
  have h := gateARow tz cfX [] pfX ⟨0, hn⟩
  rw [parsedOf_x] at h
  refine ⟨⟨0, hn⟩, h, ?_⟩
  rw [h]
  exact m_x

end Flock.Bridge.GateARowChecks
