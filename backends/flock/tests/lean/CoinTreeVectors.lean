import Lean.Data.Json
import FlockSoundness.Discharge.ZkTree.Layout

/-!
The soundness package's layout of the live coin tree (`FlockSoundness/Discharge/ZkTree/Layout.lean`) on the tapes of
`tests/fixtures/coin_tree_rust.json` (the live crate's `CoinTree::draw`, `live/examples/coin_tree_lean.rs`), for
`tests/fixtures/coin_tree_lean.json`.

* `draws`: for each splitmix64 draw, the key (`keyBits` of the tape's first 256 bytes, written back with bit 2047 zero),
  and for each opening `(stream, round, n)` the coins `drLive block n` reads off the block at `tapeAt` and the salt
  (`saltBits` written back), and the link points `dlLive block m` reads off the session stream's round 0, each through
  its decoder's equivalence (`blockEquiv`, `ptsEquiv`).
* `mask`: `maskBytes` of the §9 key and salt.

Regenerate with `tests/lean/coin_tree.sh`.
-/

open Lean FlockSoundness.Discharge.ZkTree

def hexOf (b : ByteArray) : String :=
  let d := "0123456789abcdef".toList.toArray
  String.ofList (b.toList.flatMap fun x => [d[(x.toNat / 16)]!, d[(x.toNat % 16)]!])

def unhex (s : String) : ByteArray :=
  let v := fun (c : Char) => if c.isDigit then c.toNat - 48 else c.toNat - 87
  let cs := s.toList.toArray
  ⟨(Array.range (cs.size / 2)).map fun i => (16 * v cs[2 * i]! + v cs[2 * i + 1]!).toUInt8⟩

/-- splitmix64's `i`-th output from `seed`. -/
def sm (seed : UInt64) (i : Nat) : UInt64 :=
  let s := seed + (i.toUInt64 + 1) * 0x9e3779b97f4a7c15
  let z := (s ^^^ (s >>> 30)) * 0xbf58476d1ce4e5b9
  let z := (z ^^^ (z >>> 27)) * 0x94d049bb133111eb
  z ^^^ (z >>> 31)

/-- The tape's bytes `a .. a + len`. -/
def window (seed : UInt64) (a len : Nat) : ByteArray :=
  ⟨(Array.range len).map fun j => ((sm seed ((a + j) / 8)) >>> (8 * ((a + j) % 8)).toUInt64).toUInt8⟩

/-- Bits written back as bytes, bit `t` bit `t % 8` of byte `t / 8`. -/
def bytesOf (nb : Nat) (bit : Nat → Bool) : ByteArray :=
  ⟨(Array.range nb).map fun i => ((List.range 8).foldl (fun acc j => acc + if bit (8 * i + j) then 2 ^ j else 0) 0).toUInt8⟩

def wordHex (x : FlockLevel3.GF128) : String :=
  let v : Flock.F128 := x.val
  hexOf ⟨(Array.range 8).map (fun j => (v.lo >>> (8 * j).toUInt64).toUInt8) ++
    (Array.range 8).map (fun j => (v.hi >>> (8 * j).toUInt64).toUInt8)⟩

/-- `drLive block n`'s coins (its equivalence `blockEquiv`: the `Dec`'s `Fintype` of the rest is not evaluated). -/
def coinsHex (block n : Nat) (b : LiveBlk block) : Json :=
  if h : n ≤ block then Json.arr ((Array.finRange n).map fun i => Json.str (wordHex (((blockEquiv h).symm b).1 i)))
  else Json.null

/-- `dlLive block m`'s points (its equivalence `ptsEquiv`). -/
def pointsHex (block m : Nat) (b : LiveBlk block) : Json :=
  if h : 2 * m ≤ block then Json.arr ((Array.finRange 2).map fun k => Json.arr ((Array.finRange m).map fun t =>
      Json.str (wordHex (((ptsEquiv h).symm b).1 k t))))
  else Json.null

def num (j : Json) (k : String) : Nat := (j.getObjValAs? Nat k).toOption.getD 0

def drawOf (j : Json) : Json := Id.run do
  let seed := (num j "seed").toUInt64
  let sp := (j.getObjVal? "spec").toOption.getD Json.null
  let (rl, block) := (num sp "rounds_log", num sp "block")
  let kb := window seed 0 256
  let key := bytesOf 256 fun t => t < 2047 && (keyBits kb 0 ⟨t % 2047, Nat.mod_lt _ (by decide)⟩ = 1)
  let mut opens := #[]
  for o in ((j.getObjValAs? (Array Json) "openings").toOption.getD #[]) do
    let (s, k, n) := (num o "stream", num o "round", num o "n")
    let w := window seed (tapeAt rl block s k) (16 * block + 192)
    let salt := bytesOf 192 fun t => t < 1536 && (saltBits w (16 * block) ⟨t % 1536, Nat.mod_lt _ (by decide)⟩ = 1)
    opens := opens.push (Json.mkObj [("stream", s), ("round", k), ("n", n),
      ("coins", coinsHex block n (blockAt w 0 block)), ("salt", hexOf salt)])
  let m := num ((j.getObjVal? "points").toOption.getD Json.null) "m"
  let w0 := window seed (tapeAt rl block 0 0) (16 * block)
  return Json.mkObj [("name", (j.getObjVal? "name").toOption.getD Json.null), ("key", hexOf key),
    ("openings", Json.arr opens), ("points", Json.mkObj [("m", m), ("points", pointsHex block m (blockAt w0 0 block))])]

def main (args : List String) : IO Unit := do
  let rust ← IO.ofExcept (Json.parse (← IO.FS.readFile args[0]!))
  let draws := ((rust.getObjValAs? (Array Json) "draws").toOption.getD #[]).map drawOf
  let mk := (rust.getObjVal? "mask").toOption.getD Json.null
  let str := fun (k : String) => (mk.getObjValAs? String k).toOption.getD ""
  let mask := maskBytes (keyBits (unhex (str "key")) 0) (saltBits (unhex (str "salt")) 0)
  IO.println (Json.mkObj [("draws", Json.arr draws), ("mask", Json.mkObj [("mask", hexOf mask)])]).pretty
