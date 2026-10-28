import Difftest
import PousCheck.Setup

/-!
# The setup difftest: `PousCheck`'s functions behind `Difftest.serve`

`pous-setup-difftest` serves `table` to the core difftest harness (`verity/core/difftest`), whose Python half
(`tests/test_setup_difftest.py`) draws inputs and holds each function's output to `verity.core.protocols.pous`
(`schemes/p2.py`, `setup.py`) and `verity.ml.commitments.identity`. The functions:

* `maps`: `P2.d`, `P2.sigma` and `P2.decodeValue` on canonical values (Python's `_Field.sq`, `sigma`);
* `keys`: `Keys.packagePrefix` and `Expander.keys` (`expand_prefix`, `sha512_keys`): K, t and which candidate t is, or
  the salt's refusal;
* `decode`: `Setup.decodeSegment` (`decode_segment`): the payload, or the kind of refusal;
* `commit`: `Commit.commit` and `Commit.manifest` (`setup.commit`, `canonical_json_bytes(params.manifest())`), the
  commitment over `Identity.taggedState` (the identity hash itself is `core_identity`'s, difftested there);
* `vector` and `sigma`: an entry of the vectors file (`benchmarks/pous/p2_v1/vectors.json`, schema `p2-test-vectors/v4`):
  its own `sha512` (of its canonical JSON without it) and whether `ENTRY_SHA512` pins it, its K, t, rejection index and `D(c)` under the
  spec's prefix (`Keys.specPrefix`); a σ rejection case's `p`, `r ⊕ K` and `σ_K(r)`. The test holds these to the file's
  recorded values, and the file to `reference.VECTORS_SHA512`.

The setup check itself is the protocol `pous-setup` serves (`SetupProtocol`), which the harness drives as a protocol.
-/

namespace PousCheck.SetupDifftest

open Lean PousCheck

abbrev V := Except String

def str (j : Json) (k : String) : V String := Flock.J.str j k "instance"
def nat (j : Json) (k : String) : V Nat := Flock.J.nat j k "instance"
def bytes (j : Json) (k : String) : V ByteArray := do Flock.unhex (← str j k)
def natHex (j : Json) (k : String) : V Nat := do return Prim.natOfBE (← bytes j k)
def hexW (x n : Nat) : Json := .str (Flock.hexOf (Prim.beOfNat x n))
def hex (b : ByteArray) : Json := .str (Flock.hexOf b)
def err (kind : String) : Json := Json.mkObj [("error", .str kind)]

def params (j : Json) : V P2.Params := do
  let p ← Flock.J.get j "params" "instance"
  P2.Params.validate ⟨← nat p "w", ← nat p "c0", ← nat p "blocks_per_segment", ← nat p "segments"⟩

def blockList (j : Json) : V (Array ByteArray) := do (← Flock.J.arr j "blocks" "instance").mapM fun b => do
  Flock.unhex (← Flock.J.asStr b "block")

def maps (i : Json) : V Json := do
  let q ← P2.Params.validate ⟨← nat i "w", ← nat i "c0", 1, 1⟩
  let (p, n) := (q.p, q.blockBytes)
  let (z, r, k, t) := (← natHex i "z", ← natHex i "r", ← natHex i "k", ← natHex i "t")
  if z ≥ p || r ≥ p || t ≥ p then throw "maps: a value is not canonical"
  return Json.mkObj [("d", hexW (P2.d p z) n), ("sigma", hexW (P2.sigma p r k) n),
    ("decode", hexW (P2.decodeValue p k t z) n)]

def keys (i : Json) : V Json := do
  let q ← P2.Params.validate ⟨← nat i "w", ← nat i "c0", 2 ^ 16, 2 ^ 20⟩
  match Keys.packagePrefix q (← bytes i "salt") with
  | .error _ => return err "salt"
  | .ok pre =>
    let (k, t, a) ← (Keys.Expander.ofPrefix pre).keys q (← nat i "s") (← nat i "j")
    return Json.mkObj [("K", hexW k q.blockBytes), ("t", hexW t q.blockBytes), ("t_index", .num a)]

def decode (i : Json) : V Json := do
  match Setup.decodeSegment (← params i) (← bytes i "salt") (← nat i "s") (← blockList i) with
  | .ok payload => return Json.mkObj [("payload", hex payload)]
  | .error r => return err r.kind

def commit (i : Json) : V Json := do
  let q ← params i
  return Json.mkObj [("commitment", hex (← Commit.commit q (← bytes i "w"))),
    ("manifest", .str (String.fromUTF8! (← Commit.manifest q)))]

def without (keys : List String) : Json → Json
  | .obj kvs => .obj (keys.foldl (fun m k => m.erase k) kvs)
  | j => j

/-- The vectors' entry hashes, by width (`reference.ENTRY_SHA512`). -/
def ENTRY_SHA512 : List (Nat × String) := [
  (64, "c34350fa0e0b8ae7c521cd7df833c160df1498adc938a99feb6166c2f37b3e6a2ca07239b9ab71fdd979b02efc8c8951d68f32d108abbde5cc44bac3137241e1"),
  (96, "df8941e52d73848baa7862cd3f20ea6f7adc1da79149db154b0be9e0d28be8905b95aac079cd98e14b56f8077ba6cfb173e0472df7e83b40138c6f16a871ff1f"),
  (128, "adf7cd300e2d4ea93643e94d563b02c80e29d88d65a34a55269f9e37da6b71bde6dce1c9e4c04c561dd103a8dc4c3955ee83ecabef20ad717de4672e8014fcae"),
  (16448, "2a227a512683d67624bca51ba087ec05a4dda85c5e93b67c50c03349a7823f50f84b98f4f2f9e92d9c87ddc41ab0e5fd8d2211472709a05093e7c0121a86811c")]

/-- One vector of `vectors.json`: its entry's hash and whether `ENTRY_SHA512` pins it, and its K, t, rejection index and `D(c)` under the spec's
prefix, with `c < p` and the payload's bound. -/
def vector (v : Json) : V Json := do
  let q : P2.Params := ⟨← nat v "w", ← nat v "c0", 2 ^ 16, 8⟩
  let entry := Flock.hexOf (Flock.Sha512.hash (← Flock.canon (without ["sha512"] v)).toUTF8)
  let e := Keys.Expander.ofPrefix (Keys.specPrefix q (← bytes v "model_commitment") (← bytes v "setup_salt"))
  let (k, t, a) ← e.keys q (← nat v "segment") (← nat v "block")
  let b := if q.w == W_BITS then 16432 else 8 * (q.blockBytes - 1)
  let c ← natHex v "c"
  let m := P2.decodeValue q.p k t c
  return Json.mkObj [("sha512", .str entry), ("pinned", .bool (ENTRY_SHA512.lookup q.w == some entry)),
    ("K", hexW k q.blockBytes), ("t", hexW t q.blockBytes), ("t_rejection_index", .num a),
    ("canonical", .bool (c < q.p)), ("decoded_m", hexW m q.blockBytes), ("fits", .bool (m < 2 ^ b))]

/-- One σ rejection case: `p`, `r ⊕ K`, whether it is at least `p`, and `σ_K(r)`. -/
def sigma (v : Json) : V Json := do
  let q : P2.Params := ⟨← nat v "w", ← nat v "c0", 1, 1⟩
  let (r, k) := (← natHex v "r", ← natHex v "K")
  return Json.mkObj [("p", hexW q.p q.blockBytes), ("r_xor_K", hexW (r ^^^ k) q.blockBytes),
    ("rejects", .bool (r ^^^ k ≥ q.p)), ("sigma", hexW (P2.sigma q.p r k) q.blockBytes)]

def table : String → Option Difftest.Fn :=
  Difftest.table [("maps", maps), ("keys", keys), ("decode", decode), ("commit", commit), ("vector", vector),
    ("sigma", sigma)]

end PousCheck.SetupDifftest
