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
  /-- item 1e (#225): the unit is circuit types and a layout, not rows, and the verifier derives its rows (`Flock.Typed`) -/
  typed : Bool := false
  /-- #306: a session may prove its units in several tables; `none`: one table per session -/
  manyTables : Option TablesTags := none
  /-- every output is a hidden row (PROTOCOL.md §16.13, `Flock.HmOut`): the rows are the input rows, then one
  `hm96-sha512/row/v1` row per output port hashed in the circuit, and the public file has no output word -/
  hiddenOutputs : Bool := false
  /-- the prover of main from `804be92b4`: on per-round OS coins (`FC_COINS=os`) its identity names the coin mode and no coin
  commitment or derivation (`liveOsIdentity`) -/
  liveOs : Bool := false
  /-- the prover after verity `cec3e347` (Daniel's decision 4): every random value it uses is the OS's bytes, which the
  `--zk` identity's `hooks` name (`Flock.Zk.hooks`); before, the pads and every table's draws were ChaCha20 streams of the
  prover's seed -/
  osMasks : Bool := false
  /-- the plain proof files are upstream's form, which end after `consistency_batch_grinding_nonces` (`flock-verify
  --upstream-plain`: the agreement corpus's sessions, from upstream's builds); by default they are the deployed prover's,
  the zk patch's build (`pod/60-circuit.sh`), with its `zk_level0 = None` tag after, the one form `decodeProof` accepts;
  the verifier decodes upstream's as the file then that tag, and S17 hashes the file (`verifyFiles (plainBytes true)`;
  PROTOCOL.md §8.1, D17) -/
  upstreamPlain : Bool := false
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

/-- `id` on per-round OS coins, as upstream's `identity()` writes it under `FC_COINS=os`: `coins` names the mode, and
`hashes` names no coin commitment or derivation. -/
def liveOsIdentity (id : Json) : Json :=
  let hashes := (id.getObjValD "hashes").setObjVal! "coin_commitment" (s "none (FC_COINS=os)")
  let hashes := hashes.setObjVal! "coin_derivation" (s "none (FC_COINS=os)")
  (id.setObjVal! "hashes" hashes).setObjVal! "coins" (s "live-os (every round's coins the verifier's OS draws: no seed, no PRF)")

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

/-- META `leaf_scheme` from PR #83 `b32e1a7b` up to verity `b3727bd9`: `hm96-sha512/v1` leaves at every Ligerito level, salts
a ChaCha20 expansion of a per-proof OS key (`key_sha512`: SHA-512 of the default key, from its label). -/
def leafSchemeHm96 : Json :=
  Json.mkObj [("id", s "hm96-sha512/v1"), ("inner", s "sha512 column digest"), ("salt_bytes", n 192),
    ("salt_source", s "chacha20 expansion of a fresh 256-bit os key per proof, never reused or stored"),
    ("key", s "default_key, a statement constant for every tree"), ("key_sha512", s (hexOf (Sha512.hash Hm96.Default512.key)))]

/-- META `leaf_scheme` after verity #1654 (`b3727bd9`, Daniel's decision 4(a)): `leafSchemeHm96` with its salts named as
the prover draws them, the OS's own bytes, every leaf's 192 read with its tree's (`zk_hooks::LeafSalts`). -/
def leafSchemeHm96Os : Json :=
  Json.mkObj [("id", s "hm96-sha512/v1"), ("inner", s "sha512 column digest"), ("salt_bytes", n 192),
    ("salt_source", s "os bytes (getrandom), 192 per leaf, each tree's read once, never reused or stored"),
    ("key", s "default_key, a statement constant for every tree"), ("key_sha512", s (hexOf (Sha512.hash Hm96.Default512.key)))]

/-- The identity of PR #83's statement from `eb90718f`; `seedInjection` is true only in the selftest harness's build. From
`68ae79f2` (`sha512Text`) its text names the round digest, the statement digest and Σ as SHA-512, as they are since
`eb90718f`. `coinDerivation` names `verity.primitives.randomness`'s hash: SHA-256 up to verity `f5df3bbc`, SHA-512 (its v2)
after. `leafScheme` is META's: `leafSchemeHm96` up to verity `b3727bd9`, `leafSchemeHm96Os` after. -/
def identityHm96 (seedInjection : Bool) (sha512Text : Bool := false) (statement : String := "verity/flock-circuit")
    (coinDerivation : String := "sha256 (verity.randomness)") (leafScheme : Json := leafSchemeHm96) : Json :=
  let h := if sha512Text then "sha512" else "sha256"
  Json.mkObj [("statement", s statement), ("backend", s "C-interactive"),
    ("proof_class", s "NON_ZK_PROOF"), ("profile", s "fast100"), ("reps", n 2), ("commitments", s "one root, both reps (R1)"),
    ("flavor", s "rs"), ("leaf_scheme", leafScheme), ("seed_injection", Json.bool seedInjection),
    ("hashes", Json.mkObj [("merkle", s "hm96-sha512 (every Ligerito level)"),
      ("round_digest", s s!"{h} (framing: the coin server retains every round's bytes and the replay compares them)"),
      ("statement_digest", s h), ("sigma", s h), ("coin_commitment", s "sha512"),
      ("coin_derivation", s coinDerivation), ("frame_v3_tree", s "sha512 (frame-v3-sha512)"),
      ("transcript", s "none (live verifier coins)"), ("row_leaf_in_circuit", s "hm96-sha512/row/v1 over sha512/row/v1")]),
    ("hooks", hooksNoMask)]

/-- A hidden-output identity's `outputs`, its output rows `leaf` rows. -/
def hiddenOutputsText (leaf : String) : String :=
  s!"hidden: each output port an {leaf} row hashed in the circuit; the public file has no output word"

/-- A bit-row statement's identity (META `row_schemas`): `row_leaf_in_circuit`, and a hidden-output identity's `outputs`,
name `hm96-sha512/row/v2`, as the prover's do; a `sha512/row/v1` statement keeps its identity unchanged. -/
def rowV2Identity (id : Json) : Json :=
  let id := id.setObjVal! "hashes" ((id.getObjValD "hashes").setObjVal! "row_leaf_in_circuit" (s "hm96-sha512/row/v2 over sha512/row/v2"))
  if (id.getObjVal? "outputs").toOption.isSome then id.setObjVal! "outputs" (s (hiddenOutputsText "hm96-sha512/row/v2")) else id

/-- A statement with a segmented row (`hm96-sha512/row-seg/v1`, a port of META `row_schemas`): `row_leaf_in_circuit` names
both bit-row leaves, as the prover's does; a statement of `v2` rows alone keeps its identity. -/
def rowSegIdentity (id : Json) : Json :=
  id.setObjVal! "hashes" ((id.getObjValD "hashes").setObjVal! "row_leaf_in_circuit"
    (s "hm96-sha512/row/v2 over sha512/row/v2, hm96-sha512/row-seg/v1 over sha512/row-seg/v1, per port"))

/-- A hidden-output statement with runs of output ports (META `out_row_ports`): `outputs` names their segmented rows, as the
prover's does; a statement of one output row per port keeps its identity. -/
def outRowsIdentity (id : Json) : Json :=
  if (id.getObjVal? "outputs").toOption.isSome then
    id.setObjVal! "outputs" (s ("hidden: each output port an hm96-sha512/row/v2 row, or each run of output ports in META " ++
      "out_row_ports one hm96-sha512/row-seg/v1 row of their words, hashed in the circuit; the public file has no output word"))
  else id

/-- `outRowsIdentity` with a run committed as an `hm96-sha512/row/v2` row: `outputs` names both leaves, as the prover's does. -/
def outRowsV2Identity (id : Json) : Json :=
  if (id.getObjVal? "outputs").toOption.isSome then
    id.setObjVal! "outputs" (s ("hidden: each output port an hm96-sha512/row/v2 row, or each run of output ports in META " ++
      "out_row_ports one hm96-sha512/row/v2 or hm96-sha512/row-seg/v1 row of their words (its row_schemas entry), hashed in " ++
      "the circuit; the public file has no output word"))
  else id

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
def circuitE51e2b86 : Tags :=
  { circuitEb90718f with name := "verity/flock-circuit@e51e2b86", bindings := true }

/-- The same statement as the selftest harness's build pins it (`seed_injection: true`): M0's selftest records only. -/
def circuitE51e2b86Selftest : Tags :=
  { circuitE51e2b86 with name := "verity/flock-circuit@e51e2b86+seed-injection", identity := identityHm96 true }

/-- `verity/flock-circuit` as of PR #83 `967b8d06`: `e51e2b86`'s statement with the identity text of `68ae79f2` (the round
digest, the statement digest and Σ named SHA-512), and shared-row public files (`967b8d06`). From #306 a session may prove
its units in several tables. -/
def circuit967b8d06 : Tags :=
  { circuitE51e2b86 with
    name := "verity/flock-circuit@967b8d06", identity := identityHm96 false true, sharedRows := true,
    manyTables := some { sigmaTag := "verity/flock-circuit/sigma-tables", domainPrefix := "flock-circuit/fast100x2/" },
    liveOs := true }

/-- A selftest build always seeds, so its statement has no live-OS identity. -/
def circuit967b8d06Selftest : Tags :=
  { circuit967b8d06 with
    name := "verity/flock-circuit@967b8d06+seed-injection", identity := identityHm96 true true, liveOs := false }

/-- `verity/flock-circuit/types` (item 1e, #225, #273) up to verity `210d32e1`: `verity/flock-circuit`'s statement (with
`967b8d06`'s identity text) with its unit as circuit types and a layout; the verifier derives the unit's rows
(`Flock.Typed`). The identity and the statement digest's leading tag name the typed statement, as the prover's do, and it has
its own Σ tag and domain (a new id gets new tags). -/
def circuitTypes210d32e1 : Tags :=
  { circuit967b8d06 with
    name := "verity/flock-circuit/types@210d32e1", statement := "verity/flock-circuit/types", fileFormat := "flock-circuit/types",
    sigmaTag := "verity/flock-circuit/types/sigma", domainPrefix := "flock-circuit-types/fast100x2/rep",
    identity := identityHm96 false true "verity/flock-circuit/types", typed := true, manyTables := none }

def circuitTypes210d32e1Selftest : Tags :=
  { circuitTypes210d32e1 with
    name := "verity/flock-circuit/types@210d32e1+seed-injection", identity := identityHm96 true true "verity/flock-circuit/types",
    liveOs := false }

/-- The identity of a statement with hidden outputs: `967b8d06`'s text (or the typed statement's), naming them, the coins'
derivation and META's leaf scheme (`identityHm96`'s `coinDerivation` and `leafScheme`). -/
def identityHidden (seedInjection : Bool) (statement : String := "verity/flock-circuit")
    (coinDerivation : String := "sha512 (verity.randomness)") (leafScheme : Json := leafSchemeHm96) : Json :=
  (identityHm96 seedInjection true statement coinDerivation leafScheme).setObjVal! "outputs"
    (s (hiddenOutputsText "hm96-sha512/row/v1"))

/-- `verity/flock-circuit` (PROTOCOL.md §16.13) up to verity `f5df3bbc`: `967b8d06`'s statement with every output hidden, an
`hm96-sha512/row/v1` row hashed in the circuit after the input rows; the public file holds every row's `b ‖ c` and no output
word. Its coins derive through `verity.primitives.randomness` v1 (SHA-256), and the identity says so. -/
def circuitF5df3bbc : Tags :=
  { circuit967b8d06 with
    name := "verity/flock-circuit@f5df3bbc", identity := identityHidden false (coinDerivation := "sha256 (verity.randomness)"),
    hiddenOutputs := true }

/-- The same statement as the selftest harness's build pins it (`seed_injection: true`): M0's selftest records only. -/
def circuitF5df3bbcSelftest : Tags :=
  { circuitF5df3bbc with
    name := "verity/flock-circuit@f5df3bbc+seed-injection",
    identity := identityHidden true (coinDerivation := "sha256 (verity.randomness)"), liveOs := false }

/-- `verity/flock-circuit/types` (PROTOCOL.md §16.13) up to verity `f5df3bbc`: `210d32e1`'s typed statement with hidden outputs,
its coins from `verity.primitives.randomness` v1 (SHA-256). -/
def circuitTypesF5df3bbc : Tags :=
  { circuitTypes210d32e1 with
    name := "verity/flock-circuit/types@f5df3bbc",
    identity := identityHidden false "verity/flock-circuit/types" "sha256 (verity.randomness)", hiddenOutputs := true }

def circuitTypesF5df3bbcSelftest : Tags :=
  { circuitTypesF5df3bbc with
    name := "verity/flock-circuit/types@f5df3bbc+seed-injection",
    identity := identityHidden true "verity/flock-circuit/types" "sha256 (verity.randomness)", liveOs := false }

/-- `verity/flock-circuit` (PROTOCOL.md §16.13) up to verity `b3727bd9`: `f5df3bbc`'s statement with its coins from
`verity.primitives.randomness` v2 (SHA-512), as the identity says, and META's leaf salts named a ChaCha20 expansion of a
per-proof OS key (`leafSchemeHm96`). -/
def circuitB3727bd9 : Tags :=
  { circuitF5df3bbc with name := "verity/flock-circuit@b3727bd9", identity := identityHidden false }

/-- `verity/flock-circuit/types` (PROTOCOL.md §16.13) up to verity `b3727bd9`: `f5df3bbc`'s typed statement with its coins
from `verity.primitives.randomness` v2 (SHA-512), and META's `leafSchemeHm96`. -/
def circuitTypesB3727bd9 : Tags :=
  { circuitTypesF5df3bbc with
    name := "verity/flock-circuit/types@b3727bd9", identity := identityHidden false "verity/flock-circuit/types" }

def circuitTypesB3727bd9Selftest : Tags :=
  { circuitTypesB3727bd9 with
    name := "verity/flock-circuit/types@b3727bd9+seed-injection", identity := identityHidden true "verity/flock-circuit/types",
    liveOs := false }

/-- `verity/flock-circuit` (PROTOCOL.md §16.13) up to verity `cec3e347`: `b3727bd9`'s statement with META's leaf salts
named as the prover draws them (`leafSchemeHm96Os`), in the identity and the pinned leaf scheme; its `--zk` identity names
the pads and every table's draws ChaCha20 streams of the prover's seed. -/
def circuitCec3e347 : Tags :=
  { circuitB3727bd9 with
    name := "verity/flock-circuit@cec3e347", identity := identityHidden false (leafScheme := leafSchemeHm96Os),
    pinnedLeafScheme := some leafSchemeHm96Os }

/-- `verity/flock-circuit/types` (PROTOCOL.md §16.13) up to verity `cec3e347`: `b3727bd9`'s typed statement with META's
`leafSchemeHm96Os`. -/
def circuitTypesCec3e347 : Tags :=
  { circuitTypesB3727bd9 with
    name := "verity/flock-circuit/types@cec3e347",
    identity := identityHidden false "verity/flock-circuit/types" (leafScheme := leafSchemeHm96Os),
    pinnedLeafScheme := some leafSchemeHm96Os }

def circuitTypesCec3e347Selftest : Tags :=
  { circuitTypesCec3e347 with
    name := "verity/flock-circuit/types@cec3e347+seed-injection",
    identity := identityHidden true "verity/flock-circuit/types" (leafScheme := leafSchemeHm96Os), liveOs := false }

/-- `verity/flock-circuit` (PROTOCOL.md §16.13): `cec3e347`'s statement, its prover's every random value the OS's bytes,
as its `--zk` identity's `hooks` say (`osMasks`). -/
def circuit : Tags := { circuitCec3e347 with name := "verity/flock-circuit", osMasks := true }

/-- The same statement as the selftest harness's build pins it (`seed_injection: true`): the selftest's records only, which
`verify` refuses once every check has passed (`ProvedScope.servedForm`). -/
def circuitSelftest : Tags :=
  { circuit with
    name := "verity/flock-circuit+seed-injection", identity := identityHidden true (leafScheme := leafSchemeHm96Os),
    liveOs := false }

/-- `verity/flock-circuit/types` (PROTOCOL.md §16.13): `cec3e347`'s typed statement with `osMasks`. -/
def circuitTypes : Tags := { circuitTypesCec3e347 with name := "verity/flock-circuit/types", osMasks := true }

def circuitTypesSelftest : Tags :=
  { circuitTypesCec3e347Selftest with name := "verity/flock-circuit/types+seed-injection", osMasks := true }

/-- META `leaf_scheme` of a recursive session's inner statement (upstream's `leaf_scheme_plain`,
`verity_flock.circuit.LEAF_SCHEME_PLAIN`): plain SHA-512 leaves and no salt, since the firewall commits every tree top under
a salt of its own. -/
def leafSchemePlain : Json :=
  Json.mkObj [("id", s "flock-leaf/sha512-unsalted"), ("inner", s "sha512 column digest"), ("salt_bytes", n 0),
    ("use", s "a recursive session's inner statement: the gateway commits every tree top under a salt, ZK off")]

/-- `t` with plain SHA-512 leaves (`leafSchemePlain`), as upstream proves a statement whose META `leaf_scheme` is that
object: its identity names the leaf scheme and the Merkle hash (`flock-circuit`'s `identity`, from `merkle(c)`). -/
def withPlainLeaves (t : Tags) : Tags :=
  let hashes := (t.identity.getObjValD "hashes").setObjVal! "merkle" (s "sha512 (every Ligerito level)")
  let identity := (t.identity.setObjVal! "leaf_scheme" leafSchemePlain).setObjVal! "hashes" hashes
  { t with merkleLeaf := "flock-leaf/sha512-unsalted", pinnedLeafScheme := some leafSchemePlain, identity }

/-- A statement whose trees have plain SHA-512 leaves (leafSchemePlain), ZK off: verify refuses it under --zk, as upstream's
--zk does, and without --zk takes it where `ProvedScope.leafOk` covers it (hidden outputs). -/
def plainLeaves (t : Tags) : Bool := t.pinnedLeafScheme == some leafSchemePlain

/-- A recursive session's inner statement (PROTOCOL.md §16.16): the default statement with plain SHA-512 leaves. -/
def circuitPlainLeaves : Tags := { circuit.withPlainLeaves with name := "verity/flock-circuit+plain-leaves" }

/-- The versions this verifier knows, by name. `verify` accepts a session only at `circuit` or `circuitPlainLeaves`
(`ProvedScope.served`); it sets every other version up and runs every check, so the agreement corpus and the recorded
fixtures replay, and then refuses the session. -/
def all : List Tags := [netlistV1, circuit19c7269a, circuitFd02e847, circuit631567f7, circuitEb90718f, circuitEb90718fSelftest,
  circuitE51e2b86, circuitE51e2b86Selftest, circuit967b8d06, circuit967b8d06Selftest, circuitTypes210d32e1,
  circuitTypes210d32e1Selftest, circuitF5df3bbc, circuitF5df3bbcSelftest, circuitTypesF5df3bbc, circuitTypesF5df3bbcSelftest,
  circuitB3727bd9, circuitTypesB3727bd9, circuitTypesB3727bd9Selftest,
  circuitCec3e347, circuitTypesCec3e347, circuitTypesCec3e347Selftest,
  circuit, circuitSelftest, circuitTypes, circuitTypesSelftest, circuitPlainLeaves]

def find? (name : String) : Option Tags := all.find? (·.name == name)

/-- `verity/flock-circuit/tmpl` (PRF-133): `verity/flock-circuit`'s statement with its unit a `flock-ir-tmpl/v2` template
and a layout mode in META (`HmOut.parseTmpl`), which the verifier unrolls and folds from the template
(`Stmt.partialRangesT`). The identity and the statement digest's leading tag name it, with its own Σ tag and domain. -/
def circuitTmpl : Tags :=
  { circuit with
    name := "verity/flock-circuit/tmpl", statement := "verity/flock-circuit/tmpl", fileFormat := "flock-circuit/tmpl",
    sigmaTag := "verity/flock-circuit/tmpl/sigma", domainPrefix := "flock-circuit-tmpl/fast100x2/rep",
    identity := identityHidden false "verity/flock-circuit/tmpl" (leafScheme := leafSchemeHm96Os) }

/-- The templated statements, which no soundness theorem covers yet: `verify` runs every check of a session and then
refuses it, so they are outside `all` and `find?`, which the proofs read. -/
def templated : List Tags := [circuitTmpl]

end Tags

end Flock
