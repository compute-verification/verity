/-
The byte tags of the circuit statement (PROTOCOL.md §16.5), as a parameter: every string here is hashed into Σ or the
statement digest, or matched byte for byte against the pinned files, so each version of PR #83's statement is one value.
-/
import Lean.Data.Json
import Flock.Hm96

namespace Flock

open Lean (Json)

/-- Sessions of several tables (upstream's `session_tables`, `--session-tables J`): the session Σ's tag, and the rep domains'
prefix before `{table}/rep{r}`. -/
structure TablesTags where
  sigmaTag : String
  domainPrefix : String
  deriving Inhabited

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
  /-- the coin server retains every round's bytes (`Hello`'s `rounds`; every recorded round carries `msg`, §6.2) -/
  retainRounds : Bool := false
  /-- PR #83 from `eb90718f`: serving rows are `hm96-sha512/row/v1` leaves proved in the circuit (`Flock.HmRow`), and every
  identity and binding digest is SHA-512 (the circuit pin, the public digest, the statement digest, Σ, the record's) -/
  hm96Rows : Bool := false
  /-- PR #83 from `e51e2b86`: META binds `program_sha512` and a partition (`verity/partition/v1`, or the provisional unit cover),
  and the public header's `units` carries every instance's class (`Flock.HmRow.classOf`) -/
  bindings : Bool := false
  /-- PR #83 from `967b8d06`: a public file may be a shared-row file (header `shared_rows`, `Flock.HmRow.sharedRows`) -/
  sharedRows : Bool := false
  /-- #306: a session may prove its units in several tables; `none`: one table per session -/
  manyTables : Option TablesTags := none
  deriving Inhabited

namespace Tags

def domain (t : Tags) (r : Nat) : String := t.domainPrefix ++ toString r

/-- The most tables in one session; their names sort in table order (`circuit-00` to `circuit-63`). -/
def MAX_TABLES : Nat := 64

/-- A session's table names: `table` alone, or `table-00`, `table-01`, … -/
def tableNames (t : Tags) (count : Nat) : Array String :=
  if count ≤ 1 then #[t.table]
  else (Array.range count).map fun j => s!"{t.table}-{if j < 10 then "0" else ""}{j}"

/-- A rep stream's domain in a session of `count` tables: `domain r` for one, the table's name in it for several. -/
def tableDomain (t : Tags) (count : Nat) (table : String) (r : Nat) : String :=
  match t.manyTables with
  | some tt => if count ≤ 1 then t.domain r else s!"{tt.domainPrefix}{table}/rep{r}"
  | none => t.domain r

/-- Table `j`'s identity in a session of `count > 1` tables: the statement's, naming the session's rules. -/
def tableIdentity (t : Tags) (count j : Nat) : Json :=
  t.identity.setObjVal! "session" (Json.mkObj [("tables", Lean.toJson count), ("table", Lean.toJson j),
    ("units", Json.str "the draw's units (without a draw, the population's) in equal consecutive parts; table j proves part j as a subset draw of the population")])

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

/-- `verity/flock-circuit`, PR #83 at `fd02e847`: the tags of `19c7269a`; META `leaf_scheme` is the pinned object, and the
identity carries it. -/
def circuitFd02e847 : Tags :=
  { circuit19c7269a with
    name := "verity/flock-circuit@fd02e847",
    identity := Json.mkObj [("statement", s "verity/flock-circuit"), ("backend", s "C-interactive"),
      ("proof_class", s "NON_ZK_PROOF"), ("profile", s "fast100"), ("reps", n 2), ("commitments", s "one root, both reps (R1)"),
      ("flavor", s "rs"), ("leaf_scheme", leafSchemeFd02e847),
      ("hashes", Json.mkObj [("merkle", s "sha256"), ("round_digest", s "sha256"), ("statement_digest", s "sha256"),
        ("sigma", s "sha256"), ("coin_commitment", s "sha256"), ("coin_derivation", s "sha256 (verity.randomness)"),
        ("frame_v3_tree", s "sha256"), ("transcript", s "none (live verifier coins)"),
        ("row_leaf_in_circuit", s "blake3-keyed/row/v2")]),
      ("hooks", hooksNoMask)],
    pinnedLeafScheme := some leafSchemeFd02e847 }

/-- META `leaf_scheme` at PR #83 `b3baabd8`: the unsalted SHA-512 leaf, reserving `hm96-sha512/v1` (PR #93) with its pinned
key (`key_sha512`, from `Hm96.sha512`'s key label). -/
def leafSchemeSha512 : Json :=
  Json.mkObj [("id", s "flock-leaf/sha512-unsalted"), ("salt_bytes", n 0),
    ("target", Json.mkObj [("scheme", s "hm96-sha512/v1"), ("inner", s "sha512 column digest"), ("salt_bytes_per_leaf", n 192),
      ("salt_source", s "os generator, one draw per leaf"), ("key", s "default_key, a statement constant for every tree"),
      ("key_sha512", s (hexOf (Sha512.hash Hm96.sha512.keyFromLabel)))])]

/-- `verity/flock-circuit` on SHA-512 trees (PR #83 at `631567f7`, Daniel's SHA-512 decision): every Ligerito level's Merkle
tree on SHA-512 (64-byte digests), the SHA-512 coin commitment, and the coin server retaining every round's bytes. -/
def circuit631567f7 : Tags :=
  { circuit19c7269a with
    name := "verity/flock-circuit@631567f7", merkleLeaf := "flock-leaf/sha512-unsalted", coinScheme := "coin-commit/sha512",
    identity := Json.mkObj [("statement", s "verity/flock-circuit"), ("backend", s "C-interactive"),
      ("proof_class", s "NON_ZK_PROOF"), ("profile", s "fast100"), ("reps", n 2), ("commitments", s "one root, both reps (R1)"),
      ("flavor", s "rs"), ("leaf_scheme", leafSchemeSha512),
      ("hashes", Json.mkObj [("merkle", s "sha512 (leaves and nodes, every Ligerito level)"),
        ("round_digest", s "sha256 (framing: the coin server retains every round's bytes and the replay compares them)"),
        ("statement_digest", s "sha256"), ("sigma", s "sha256"), ("coin_commitment", s "sha512"),
        ("coin_derivation", s "sha256 (verity.randomness)"), ("frame_v3_tree", s "sha256"),
        ("transcript", s "none (live verifier coins)"), ("row_leaf_in_circuit", s "blake3-keyed/row/v2")]),
      ("hooks", hooksNoMask)],
    pinnedLeafScheme := some leafSchemeSha512, retainRounds := true }

/-- META `leaf_scheme` from PR #83 `b32e1a7b`: `hm96-sha512/v1` leaves at every Ligerito level, salts a ChaCha20 expansion of
a per-proof OS key (`key_sha512`: SHA-512 of the default key, from its label). -/
def leafSchemeHm96 : Json :=
  Json.mkObj [("id", s "hm96-sha512/v1"), ("inner", s "sha512 column digest"), ("salt_bytes", n 192),
    ("salt_source", s "chacha20 expansion of a fresh 256-bit os key per proof, never reused or stored"),
    ("key", s "default_key, a statement constant for every tree"), ("key_sha512", s (hexOf (Sha512.hash Hm96.Default512.key)))]

/-- The identity of PR #83's statement from `eb90718f`; `seedInjection` is true only in the selftest harness's build. From
`68ae79f2` (`sha512Text`) its text names the round digest, the statement digest and Σ as SHA-512, as they are since
`eb90718f`. -/
def identityHm96 (seedInjection : Bool) (sha512Text : Bool := false) : Json :=
  let h := if sha512Text then "sha512" else "sha256"
  Json.mkObj [("statement", s "verity/flock-circuit"), ("backend", s "C-interactive"),
    ("proof_class", s "NON_ZK_PROOF"), ("profile", s "fast100"), ("reps", n 2), ("commitments", s "one root, both reps (R1)"),
    ("flavor", s "rs"), ("leaf_scheme", leafSchemeHm96), ("seed_injection", Json.bool seedInjection),
    ("hashes", Json.mkObj [("merkle", s "hm96-sha512 (every Ligerito level)"),
      ("round_digest", s s!"{h} (framing: the coin server retains every round's bytes and the replay compares them)"),
      ("statement_digest", s h), ("sigma", s h), ("coin_commitment", s "sha512"),
      ("coin_derivation", s "sha256 (verity.randomness)"), ("frame_v3_tree", s "sha512 (frame-v3-sha512)"),
      ("transcript", s "none (live verifier coins)"), ("row_leaf_in_circuit", s "hm96-sha512/row/v1 over sha512/row/v1")]),
    ("hooks", hooksNoMask)]

/-- `verity/flock-circuit` (PR #83 from `eb90718f`): serving rows are `hm96-sha512/row/v1` leaves proved in the circuit,
every identity and binding digest is SHA-512, the proof's Merkle leaves are `hm96-sha512/v1`, and the unit draw is live. -/
def circuitEb90718f : Tags :=
  { circuit631567f7 with
    name := "verity/flock-circuit@eb90718f", publicCircuitKey := "circuit_sha512", merkleLeaf := "hm96-sha512/v1",
    identity := identityHm96 false, pinnedLeafScheme := some leafSchemeHm96, hm96Rows := true }

/-- `eb90718f` as the selftest harness's build pins it (`seed_injection: true`): M0's selftest records only. -/
def circuitEb90718fSelftest : Tags :=
  { circuitEb90718f with name := "verity/flock-circuit@eb90718f+seed-injection", identity := identityHm96 true }

/-- `verity/flock-circuit` (PR #83 from `e51e2b86`): `eb90718f`'s statement with the e2e bindings: META `program_sha512` and a
partition (a `verity/partition/v1` digest the verifier checks against its own copy of the partition object), and each
instance's class in the public header. -/
def circuit : Tags :=
  { circuitEb90718f with name := "verity/flock-circuit", bindings := true }

/-- The same statement as the selftest harness's build pins it (`seed_injection: true`): M0's selftest records only. -/
def circuitSelftest : Tags :=
  { circuit with name := "verity/flock-circuit+seed-injection", identity := identityHm96 true }

/-- `verity/flock-circuit` as of PR #83 `967b8d06`: `e51e2b86`'s statement with the identity text of `68ae79f2` (the round
digest, the statement digest and Σ named SHA-512), and shared-row public files (`967b8d06`). From #306 a session may prove
its units in several tables. -/
def circuit967b8d06 : Tags :=
  { circuit with
    name := "verity/flock-circuit@967b8d06", identity := identityHm96 false true, sharedRows := true,
    manyTables := some { sigmaTag := "verity/flock-circuit/sigma-tables", domainPrefix := "flock-circuit/fast100x2/" } }

def circuit967b8d06Selftest : Tags :=
  { circuit967b8d06 with name := "verity/flock-circuit@967b8d06+seed-injection", identity := identityHm96 true true }

/-- The versions this verifier knows, by name. -/
def all : List Tags := [netlistV1, circuit19c7269a, circuitFd02e847, circuit631567f7, circuitEb90718f, circuitEb90718fSelftest,
  circuit, circuitSelftest, circuit967b8d06, circuit967b8d06Selftest]

def find? (name : String) : Option Tags := all.find? (·.name == name)

end Tags

end Flock
