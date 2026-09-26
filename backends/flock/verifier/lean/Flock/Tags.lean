/-
The byte tags of the circuit statement (PROTOCOL.md §16.5), as a parameter: every string here is hashed into Σ or the
statement digest, or matched byte for byte against the pinned files, so each version of PR #83's statement is one value.
-/
import Lean.Data.Json

namespace Flock

open Lean (Json)

structure Tags where
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
  { statement := "verity/flock-netlist/v1", fileFormat := "flock-netlist/v1", blockKeyword := "NET",
    publicFormat := "flock-netlist-inst/v1", publicCircuitKey := "netlist_sha256", sigmaTag := "verity/flock-netlist/sigma/v1",
    domainPrefix := "flock-netlist-v1/fast100x2/rep", table := "nl", merkleLeaf := "flock-leaf/sha256-unsalted/v0",
    identity := Json.mkObj [("statement", s "verity/flock-netlist/v1"), ("backend", s "C-interactive"),
      ("proof_class", s "NON_ZK_PROOF"), ("profile", s "fast100"), ("reps", n 2), ("commitments", s "one root, both reps (R1)"),
      ("flavor", s "rs"), ("merkle_hash", s "sha256"), ("leaf_scheme", s "flock-leaf/sha256-unsalted/v0"),
      ("hooks", hooksNoMask)],
    coinScheme := "coin-commit/sha256/v1" }

/-- `verity/flock-circuit`, PR #83 at `19c7269a` (renamed, unversioned; the identity names every commitment hash). -/
def circuit : Tags :=
  { statement := "verity/flock-circuit", fileFormat := "flock-circuit", blockKeyword := "CIRCUIT",
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

/-- The versions this verifier knows, by statement id. -/
def all : List Tags := [netlistV1, circuit]

def find? (statement : String) : Option Tags := all.find? (·.statement == statement)

end Tags

end Flock
