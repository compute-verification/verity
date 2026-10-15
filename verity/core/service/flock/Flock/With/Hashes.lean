/-
The verifier's hashes as an oracle. `Hashes M` is SHA-512, SHA-256 and BLAKE3 as actions of a verdict monad `M`, and the
`…With` copies of the verifier (`Flock.With.Verify`, `Flock.With.Zk`) hash only through it: run in `V` with
`Hashes.ofSha512 Sha512.hash` they are `Flock.verify` and `Flock.Zk.verify` (`Flock.With.Eq`), and run in a monad that
records each call they give the verifier's log of hash calls. The one hashing the oracle does not cover is the
`hm96-sha512/v1` key's: `Hm96.Default512.key`, its mask tables and its leaf prefix are the statement's constants, derived
once from the key label with SHA-512 itself.

Here: the oracle, and §14's Merkle checks over it. A registered scheme's node and leaf go through the oracle
(`MerkleScheme.opsWith`); a scheme the registry doesn't name keeps its own closures.
-/
import Flock.Merkle

namespace Flock

/-- The hash functions the verifier calls, each an action of `M`. -/
structure Hashes (M : Type → Type) where
  sha512 : ByteArray → M ByteArray
  sha256 : ByteArray → M ByteArray
  blake3 : ByteArray → M ByteArray
  blake3Keyed : ByteArray → ByteArray → M ByteArray

/-- A Merkle scheme's node and leaf as actions of `M`. -/
structure MerkleOps (M : Type → Type) where
  node : ByteArray → ByteArray → M ByteArray
  leaf : Array F128 → ByteArray → M ByteArray

variable {M : Type → Type} [Monad M] [MonadExceptOf String M]

/-- The deployed hashes with SHA-512 replaced by `h`. -/
def Hashes.ofSha512 (h : ByteArray → ByteArray) : Hashes M :=
  { sha512 := fun b => pure (h b), sha256 := fun b => pure (Sha256.hash b), blake3 := fun b => pure (Blake3.hash b),
    blake3Keyed := fun k b => pure (Blake3.keyedHash k b) }

/-- The scheme's own closures. -/
def MerkleOps.ofScheme (ms : MerkleScheme) : MerkleOps M :=
  ⟨fun l r => pure (ms.node l r), fun row salt => pure (ms.leaf row salt)⟩

/-- `Hm96.Default512.leaf x salt` through the oracle: `SHA-512(leaf_prefix ‖ b ‖ c)` for `c = SHA-512(salt_prefix ‖ salt)`
and `b = x ⊕ M·salt`. -/
def Hm96.Default512.leafWith (hs : Hashes M) (x salt : ByteArray) : M ByteArray := do
  let m := mask salt
  let b := ByteArray.mk ((Array.range 64).map fun i => x.get! i ^^^ m.get! i)
  let c ← hs.sha512 (saltPrefix ++ salt)
  hs.sha512 (leafPrefix ++ (b ++ c))

/-- The registry's scheme of `ms`'s id, its node and leaf through the oracle, by its `merkle_hash` index; `ms`'s own
closures for an id the registry doesn't name. -/
def MerkleScheme.opsWith (hs : Hashes M) (ms : MerkleScheme) : MerkleOps M :=
  match MerkleScheme.find? ms.id with
  | some s =>
    if s.pcsIndex = 0 then ⟨fun l r => hs.sha256 (l ++ r), fun row _ => hs.sha256 (rowBytes row)⟩
    else if s.pcsIndex = 2 then ⟨fun l r => hs.sha512 (l ++ r), fun row _ => hs.sha512 (rowBytes row)⟩
    else if s.pcsIndex = 3 then
      ⟨fun l r => hs.sha512 (l ++ r), fun row salt => do Hm96.Default512.leafWith hs (← hs.sha512 (rowBytes row)) salt⟩
    else MerkleOps.ofScheme ms
  | none => MerkleOps.ofScheme ms

/-- `ms` is the registry's scheme of its id, or the registry names no scheme of its id. -/
def MerkleScheme.Named (ms : MerkleScheme) : Prop := ∀ s, MerkleScheme.find? ms.id = some s → s = ms

/-- `climb` over `ops`. -/
def climbWith (ops : MerkleOps M) : ByteArray → Nat → List ByteArray → M ByteArray
  | acc, _, [] => pure acc
  | acc, idx, s :: rest => do climbWith ops (← if idx % 2 = 0 then ops.node acc s else ops.node s acc) (idx / 2) rest

/-- `opens` over `ops`: the leaf is hashed only once the opening has its shape. -/
def opensWith (ops : MerkleOps M) (ms : MerkleScheme) (cap : Array ByteArray) (c d lanes pos : Nat) (row : Array F128)
    (salt : ByteArray) (sibs : List ByteArray) : M Bool :=
  if decide (row.size = lanes) && decide (sibs.length = d - c) && decide (salt.size = ms.saltBytes) then do
    let top ← climbWith ops (← ops.leaf row salt) pos sibs
    pure (decide (top = cap.getD (pos >>> (d - c)) ByteArray.empty))
  else pure false

/-- `merkleCheck` over `ops`. -/
def merkleCheckWith (ops : MerkleOps M) (ms : MerkleScheme) (cap : Array ByteArray) (c d : Nat) (positions : Array Nat)
    (rows : Array (Array F128)) (siblings : Array ByteArray) (salts : Array ByteArray) (lanes : Nat) : M Unit :=
  if cap.size != 2 ^ c || rows.size != positions.size || siblings.size != positions.size * (d - c)
      || salts.size != saltCount ms positions.size then
    throw "merkle: shape"
  else if cap.any (·.size != ms.digestLen) || siblings.any (·.size != ms.digestLen) then
    throw "merkle: digest width"
  else do
    let bad ← (List.range positions.size).findM? fun j => do
      let o ← opensWith ops ms cap c d lanes positions[j]! rows[j]! (salts.getD j ByteArray.empty) (sibsOf siblings (d - c) j)
      pure !o
    match bad with
    | some j => throw s!"merkle: query {j} does not open to the cap"
    | none => pure ()

end Flock
