import Flock.Hm96

/-!
SHA-512 in counter mode and `Hm96.Default512.leaf` with their words unboxed, for the store binding's payload root
(`Store.expectedRoot`), which salts and hides every block of a whole node's store. The counter mode is `Codec.xof`'s,
but the blocks every output shares are compressed once, not once an output; Flock's leaf keeps the mask's eight limbs
in an `Array UInt64`, which boxes each word a salt byte writes, and here they are a loop's arguments.
`lean/tests/test_binding.py` holds them to Python: `store_salt` to `store.store_salt` (`codec.xof`), and the expected
root to `audit.payload_tree`.
-/

namespace RemoteAudit.Unboxed

open Flock

/-- `x`'s eight bytes, little-endian, after `out`. -/
@[inline] def pushLane (out : ByteArray) (x : UInt64) : ByteArray :=
  (((((((out.push x.toUInt8).push (x >>> 8).toUInt8).push (x >>> 16).toUInt8).push (x >>> 24).toUInt8).push
    (x >>> 32).toUInt8).push (x >>> 40).toUInt8).push (x >>> 48).toUInt8).push (x >>> 56).toUInt8

/-- `x`'s eight bytes, big-endian, after `out`. -/
@[inline] def pushWord (out : ByteArray) (x : UInt64) : ByteArray :=
  (((((((out.push (x >>> 56).toUInt8).push (x >>> 48).toUInt8).push (x >>> 40).toUInt8).push (x >>> 32).toUInt8).push
    (x >>> 24).toUInt8).push (x >>> 16).toUInt8).push (x >>> 8).toUInt8).push x.toUInt8

/-- `Codec.xof msg len`: the first `len` bytes of `SHA512(msg ‖ u32be 0) ‖ SHA512(msg ‖ u32be 1) ‖ …`, over one padded
message whose counter is rewritten for each output. With at most 256 outputs the counter's first three bytes are zero
in every one, so the blocks before the one holding its last byte are every output's, compressed once. -/
def xof (msg : ByteArray) (len : Nat) : ByteArray := Id.run do
  let outputs := (len + 63) / 64
  let bits := (msg.size + 4) * 8
  let mut p := (msg ++ ByteArray.mk #[0, 0, 0, 0]).push 0x80
  for _ in [0:(112 + 128 - p.size % 128) % 128] do p := p.push 0
  for i in [0:16] do p := p.push ((bits >>> (8 * (15 - i))) % 256).toUInt8
  let shared := if outputs ≤ 256 then (msg.size + 3) / 128 else msg.size / 128
  let mut mid := Sha512.iv
  for b in [0:shared] do mid := Sha512.compress mid p (128 * b)
  let mut out := ByteArray.emptyWithCapacity (64 * outputs)
  for k in [0:outputs] do
    for i in [0:4] do p := p.set! (msg.size + i) ((k >>> (8 * (3 - i))) % 256).toUInt8
    let mut h := mid
    for b in [shared:p.size / 128] do h := Sha512.compress h p (128 * b)
    for w in h do out := pushWord out w
  return if out.size == len then out else out.extract 0 len

/-- `Hm96.Default512.mask`'s limbs from salt byte `p` on, `n` bytes of it left: limb `k` XORs in
`tables[8 (256 p + salt[p]) + k]` for each. -/
def maskFrom (salt : ByteArray) :
    Nat → Nat → UInt64 → UInt64 → UInt64 → UInt64 → UInt64 → UInt64 → UInt64 → UInt64 → ByteArray
  | 0, _, a0, a1, a2, a3, a4, a5, a6, a7 =>
    let out := ByteArray.emptyWithCapacity 64
    pushLane (pushLane (pushLane (pushLane (pushLane (pushLane (pushLane (pushLane out a0) a1) a2) a3) a4) a5) a6) a7
  | n + 1, p, a0, a1, a2, a3, a4, a5, a6, a7 =>
    let t := 8 * (256 * p + (salt.get! p).toNat)
    let T := Hm96.Default512.tables
    maskFrom salt n (p + 1) (a0 ^^^ T[t]!) (a1 ^^^ T[t + 1]!) (a2 ^^^ T[t + 2]!) (a3 ^^^ T[t + 3]!) (a4 ^^^ T[t + 4]!)
      (a5 ^^^ T[t + 5]!) (a6 ^^^ T[t + 6]!) (a7 ^^^ T[t + 7]!)

/-- `Hm96.Default512.mask`: `M·salt` as 64 bytes, over the salt's 192 bytes. -/
def mask (salt : ByteArray) : ByteArray := maskFrom salt 192 0 0 0 0 0 0 0 0 0

/-- `Hm96.Default512.leaf x salt`, `receipts.hidden_digest`: the hiding leaf of the digest `x` under `salt`. -/
def hiddenDigest (x salt : ByteArray) : ByteArray :=
  let m := mask salt
  let b := ByteArray.mk ((Array.range 64).map fun i => x.get! i ^^^ m.get! i)
  Sha512.hash (Hm96.Default512.leafPrefix ++ (b ++ Sha512.hash (Hm96.Default512.saltPrefix ++ salt)))

end RemoteAudit.Unboxed
