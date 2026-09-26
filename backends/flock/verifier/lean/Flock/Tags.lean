/-
The byte tags of the circuit statement (PROTOCOL.md §16.5), as a parameter: every string here is hashed into Σ or the
statement digest, or matched byte for byte against the pinned files, so each version of PR #83's statement is one value.
-/
import Lean.Data.Json
import Flock.Hm96

namespace Flock

open Lean (Json)

structure Tags where
  /-- how the verifier's configuration names this version: the statement id, `@<PR #83 commit>` for a superseded format -/
  name : String
  /-- the statement: the digest's tag, META `statement`, the public file's `statement`, the identity's `statement` -/
  statement : String
  /-- the circuit file's line-1 prefix and META `format` -/
  fileFormat : String
  /-- the keyword opening each expanded circuit's block of lines -/
  blockKeyword : String
  /-- the public file's `format` -/
  publicFormat : String
  /-- the public header's key holding the circuit file's SHA-256 -/
  publicCircuitKey : String
  sigmaTag : String
  /-- `domain(r) = domainPrefix ‖ decimal(r)` -/
  domainPrefix : String
  table : String
  /-- the leaf-commitment scheme of the proof's own Merkle trees (META `leaf_scheme.id`) -/
  merkleLeaf : String
  /-- the backend identity object the statement digest binds (canonical JSON) -/
  identity : Json
  /-- `Hello`'s `coins` in seed mode: the step-0 coin scheme (§7.2) -/
  coinScheme : String
  /-- when set, META `leaf_scheme` must be this object (canonical JSON), not only carry the id -/
  pinnedLeafScheme : Option Json := none
  deriving Inhabited

namespace Tags

def domain (t : Tags) (r : Nat) : String := t.domainPrefix ++ toString r

private def s (x : String) : Json := Json.str x
private def n (x : Nat) : Json := Lean.toJson x

def hooksNoMask : Json :=
  Json.mkObj [("zk", Json.bool false), ("t_pad", n 0), ("masks", s "none"), ("sumcheck_mask", s "none"),
    ("claim_permutation", s "identity")]

/-- `verity/flock-netlist/v1`, PR #83 at `9294e161` (the recorded sessions of 2026-09-26 up to 20:40Z). -/
def netlistV1 : Tags :=
  { name := "verity/flock-netlist/v1", statement := "verity/flock-netlist/v1", fileFormat := "flock-netlist/v1", blockKeyword := "NET",
    publicFormat := "flock-netlist-inst/v1", publicCircuitKey := "netlist_sha256", sigmaTag := "verity/flock-netlist/sigma/v1",
    domainPrefix := "flock-netlist-v1/fast100x2/rep", table := "nl", merkleLeaf := "flock-leaf/sha256-unsalted/v0",
    identity := Json.mkObj [("statement", s "verity/flock-netlist/v1"), ("backend", s "C-interactive"),
      ("proof_class", s "NON_ZK_PROOF"), ("profile", s "fast100"), ("reps", n 2), ("commitments", s "one root, both reps (R1)"),
      ("flavor", s "rs"), ("merkle_hash", s "sha256"), ("leaf_scheme", s "flock-leaf/sha256-unsalted/v0"),
      ("hooks", hooksNoMask)],
    coinScheme := "coin-commit/sha256/v1" }

/-- `verity/flock-circuit`, PR #83 at `19c7269a` (renamed, unversioned; the identity names every commitment hash). -/
def circuit19c7269a : Tags :=
  { name := "verity/flock-circuit@19c7269a", statement := "verity/flock-circuit", fileFormat := "flock-circuit", blockKeyword := "CIRCUIT",
    publicFormat := "flock-circuit-inputs", publicCircuitKey := "circuit_sha256", sigmaTag := "verity/flock-circuit/sigma",
    domainPrefix := "flock-circuit/fast100x2/rep", table := "circuit", merkleLeaf := "flock-leaf/sha256-unsalted",
    identity := Json.mkObj [("statement", s "verity/flock-circuit"), ("backend", s "C-interactive"),
      ("proof_class", s "NON_ZK_PROOF"), ("profile", s "fast100"), ("reps", n 2), ("commitments", s "one root, both reps (R1)"),
      ("flavor", s "rs"), ("leaf_scheme", s "flock-leaf/sha256-unsalted"),
      ("hashes", Json.mkObj [("merkle", s "sha256"), ("round_digest", s "sha256"), ("statement_digest", s "sha256"),
        ("sigma", s "sha256"), ("coin_commitment", s "sha256"), ("coin_derivation", s "sha256 (verity.randomness)"),
        ("frame_v3_tree", s "sha256"), ("transcript", s "none (live verifier coins)"),
        ("row_leaf_in_circuit", s "blake3-keyed/row/v2")]),
      ("hooks", hooksNoMask)],
    coinScheme := "coin-commit/sha256" }

/-- META `leaf_scheme` at PR #83 `fd02e847`: the unsalted SHA-256 leaf, reserving `hm96-sha256/v1` with its pinned key
(`key_sha256` is SHA-256 of `Hm96.sha256`'s key from its label, so it is checked against core's derivation). -/
def leafSchemeFd02e847 : Json :=
  Json.mkObj [("id", s "flock-leaf/sha256-unsalted"), ("salt_bytes", n 0),
    ("target", Json.mkObj [("scheme", s "hm96-sha256/v1"), ("inner", s "sha256 column digest"), ("salt_bytes_per_leaf", n 128),
      ("salt_source", s "os generator, one draw per leaf"), ("key", s "DEFAULT_KEY, a statement constant for every tree"),
      ("key_sha256", s (hexOf (Sha256.hash Hm96.sha256.keyFromLabel)))])]

/-- `verity/flock-circuit`, final format (PR #83 at `fd02e847`): the tags of `19c7269a`; META `leaf_scheme` is the pinned
object, and the identity carries it. -/
def circuit : Tags :=
  { circuit19c7269a with
    name := "verity/flock-circuit",
    identity := Json.mkObj [("statement", s "verity/flock-circuit"), ("backend", s "C-interactive"),
      ("proof_class", s "NON_ZK_PROOF"), ("profile", s "fast100"), ("reps", n 2), ("commitments", s "one root, both reps (R1)"),
      ("flavor", s "rs"), ("leaf_scheme", leafSchemeFd02e847),
      ("hashes", Json.mkObj [("merkle", s "sha256"), ("round_digest", s "sha256"), ("statement_digest", s "sha256"),
        ("sigma", s "sha256"), ("coin_commitment", s "sha256"), ("coin_derivation", s "sha256 (verity.randomness)"),
        ("frame_v3_tree", s "sha256"), ("transcript", s "none (live verifier coins)"),
        ("row_leaf_in_circuit", s "blake3-keyed/row/v2")]),
      ("hooks", hooksNoMask)],
    pinnedLeafScheme := some leafSchemeFd02e847 }

/-- The versions this verifier knows, by name. -/
def all : List Tags := [netlistV1, circuit19c7269a, circuit]

def find? (name : String) : Option Tags := all.find? (·.name == name)

end Tags

end Flock
