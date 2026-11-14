module

public import Proofs.Properties.Verity.Sampled.HonestSession.Data.Public
public import VStarRecursive.Sampled

/-!
# The honest session's parameters

The composite's parameters `θ₀`: one component, sampled proofs at the one-stage form, whose parameters name the program,
its query and partition object, the subset law of one unit, the challenger's deployed key and family, and the circuit
and public files' SHA-512s (`proofs`), each spliced as a term.

Piece P7 owns this file's values: it fixes the members to the ones `Spot.OneStageZk.paramsOfJson` reads.
-/

@[expose] public section

namespace Verity.Sampled.Instance.HonestSession

open Lean (Json)
open Flock

/-- The challenger family the parameters name. -/
def label₀ : String := "honest-session"

/-- The challenger's deployed key's public part: 80 zero bytes. -/
def keyPub : ByteArray := ByteArray.mk (Array.replicate 80 0)

/-- The challenger's deployed key. -/
def key₀ : Spot.Challenger.ChallengerKey := { params := _root_.Lms.deployed, pub := keyPub }

/-- The sampled component's parameters. -/
def paramsJ : Json := Json.mkObj [
  ("anchors", .arr #[]),
  ("challenger", Json.mkObj [("key", Json.mkObj [("H", jn 20), ("W", jn 1), ("public", .str (hexOf keyPub))]),
    ("label", .str label₀)]),
  ("delta", .str "1/2"),
  ("form", .str "one-stage"),
  ("law", law),
  ("leaf_layer", .null),
  ("log", .arr #[]),
  ("parameters", .arr #[]),
  ("partition", partitionObj),
  ("ports", Json.mkObj []),
  ("program", programJ),
  ("proofs", Json.mkObj [("circuit", .str (hexOf circuitSha)), ("public", .str (hexOf (Sha512.hash publicFile))),
    ("registered", .null)]),
  ("query", query),
  ("registered", Json.mkObj []),
  ("scheme", .str "frame-v3-sha512/hm96-sha512"),
  ("window_root", .str (hexOf (ByteArray.mk (Array.replicate 64 0)))),
  ("work_table", .null)]

/-- The composite's parameters as JSON: the one sampled component. -/
def θJ : Json := Json.mkObj [("components", .arr #[Json.mkObj [("id", .str VStar.Recursive.Sampled.PROTOCOL),
  ("params", paramsJ)]])]

/-- The composite's parameters, `θJ`'s text. -/
def θ₀ : ByteArray := (Firewall.printJson θJ).toUTF8

end Verity.Sampled.Instance.HonestSession
