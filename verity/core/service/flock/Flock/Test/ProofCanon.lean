/-
A proof file is the one encoding of what it decodes to (§8.1, D17 and D18 of §17.1): `decodeProof` and `Zk.decode` accept a
file written field by field at a small shape (two levels, one query each, one-byte digests, no salts). `decodeProof`
accepts the plain file in the deployed form, with the zk patch's `zk_level0 = None` after the pair; `verify` reads a file
through its statement's pin (`plainBytes`), so upstream's form, without the tag, passes under `--upstream-plain` alone.
Each form is refused where the other is pinned, and so is every other byte after the end, a plain file whose
`matrix_evals` holds one pair, and a query, claim or consistency grinding nonce other than 0 (R4), each with its reason.
-/
import Flock.ZkProof

namespace Flock.Test.ProofCanon

open Flock

def zeros (n : Nat) : ByteArray := ⟨Array.replicate n 0⟩

def u64 (n : Nat) : ByteArray := le64 n.toUInt64

/-- A `Vec` of `n` elements of `w` zero bytes each. -/
def vec (n w : Nat) : ByteArray := u64 n ++ zeros (n * w)

def cat (bs : List ByteArray) : ByteArray := bs.foldl (· ++ ·) ByteArray.empty

/-- A level's openings: `q` rows of `lanes` elements, their siblings, no salts. -/
def level (sh : Shape) (lv : Level) (lanes : Nat) : ByteArray :=
  u64 lv.q ++ cat (List.replicate lv.q (vec lanes 16)) ++ vec (lv.q * (lv.d - lv.cap)) sh.hashLen

/-- §8.2's pair at `sh`, every value zero: level 0's rows `l0Lanes` wide, `matrix_evals` holding `me` pairs. -/
def pair (sh : Shape) (l0Lanes me : Nat) : ByteArray :=
  let s := sh.sched
  let R := s.levels.size - 1
  let lvl := fun (i : Nat) => s.levels.getD i default
  vec (2 ^ (lvl 0).cap) sh.hashLen ++ zeros (3 * 8 + 4 + 1 + 4) ++
    vec 64 16 ++ vec 64 16 ++ vec (sh.m - 6) 32 ++ zeros 48 ++ vec 0 8 ++
    vec (sh.kLog - 6) 32 ++ vec 64 16 ++ vec me 32 ++ vec 0 8 ++
    u64 sh.claims ++ cat (List.replicate sh.claims (vec 128 16 ++ u64 0)) ++ vec 0 8 ++
    vec (2 ^ (lvl 0).cap) sh.hashLen ++ level sh (lvl 0) l0Lanes ++
    u64 R ++ cat ((List.range R).map fun i => vec (2 ^ (lvl (i + 1)).cap) sh.hashLen) ++
    u64 (R - 1) ++ cat ((List.range (R - 1)).map fun i => level sh (lvl (i + 1)) 16) ++
    vec (2 ^ (s.ext + 1)) 16 ++ level sh (lvl R) 16 ++
    vec 0 32 ++ vec (s.k0 + 8 + 7 * (R - 1)) 64 ++ vec (R + 1) 8 ++ vec (1 + 2 * R) 16 ++ vec 0 8 ++
    vec (3 * R + 2) 8 ++ vec (R + 1) 8

/-- A `--zk` file at `sh`: the pair with level 0's extra lanes, level 0's hiding messages, then the inner proof. -/
def zkFile (sh : Shape) : ByteArray :=
  pair sh (2 ^ (sh.sched.levels.getD 0 default).lanes + Zk.EXTRA_LANES) 0 ++
    ByteArray.mk #[1] ++ zeros (3 * 32) ++ vec 0 32 ++
    vec 64 1 ++ zeros 32 ++ vec 64 1 ++ vec 192 1 ++ zeros 16 ++ vec 0 16 ++ vec 0 32 ++ vec 0 192 ++ vec 0 8

def lv (lanes : Nat) : Level := { rate := 0, q := 1, cols := 0, d := 1, strata := #[0], cap := 0, lanes }

def sh : Shape :=
  { m := 7, kLog := 6, claims := 1, hashLen := 1,
    sched := { m := 7, n := 0, k0 := 0, levels := #[lv 1, lv 4], ext := 0 } }

def why {α} : V α → String
  | .error e => e
  | .ok _ => "accepted"

def canon : ByteArray := pair sh 2 0

def tagged : ByteArray := canon ++ zeros 1

/-- What the verifier decodes from `file` by default (`up = false`) and under `--upstream-plain` (`up = true`). -/
def read (up : Bool) (file : ByteArray) : V Proof := decodeProof sh (plainBytes up file)

#guard why (read false tagged) == "accepted"
#guard why (read false canon) ==
  "zk_level0: the file ends without the `None` tag every deployed prover writes after a plain proof (upstream's form: --upstream-plain)"
#guard why (read true canon) == "accepted"
#guard why (read true tagged) ==
  s!"proof: the file has {canon.size + 2} bytes and its encoding ends at byte {canon.size + 1}"

#guard (read false tagged).toOption.map (Wire.proofFile sh) == some tagged
#guard (read true canon).toOption.map (Wire.proofFile sh) == some tagged

#guard why (read false (canon ++ ByteArray.mk #[1])) ==
  "zk_level0: tag 1, want 0: a plain proof carries no hiding messages"
#guard why (read false (canon ++ ByteArray.mk #[1] ++ zeros 99)) ==
  "zk_level0: tag 1, want 0: a plain proof carries no hiding messages"
#guard [false, true].all fun up => why (read up (canon ++ ByteArray.mk #[1])) != "accepted"
#guard why (read false (canon ++ zeros 2)) ==
  s!"proof: the file has {canon.size + 2} bytes and its encoding ends at byte {canon.size + 1}"
#guard [false, true].all fun up => why (read up (canon ++ zeros 64)) ==
  s!"proof: the file has {canon.size + 64 + (if up then 1 else 0)} bytes and its encoding ends at byte {canon.size + 1}"
#guard why (read false (pair sh 2 1 ++ zeros 1)) == "matrix_evals: length 1, want 0"
#guard why (read true (pair sh 2 1)) == "matrix_evals: length 1, want 0"

#guard why (Zk.decode sh (zkFile sh)) == "accepted"
#guard why (Zk.decode sh (zkFile sh ++ zeros 1)) ==
  s!"zk proof: the file has {(zkFile sh).size + 1} bytes and its encoding ends at byte {(zkFile sh).size}"

/-- `file` with the byte `k` before `stop` set to 1. In the pair's last 152 bytes (at `R = 1`), `k = 152`, `64` and `16`
are the first query, claim and consistency nonces' low bytes. -/
def flip (file : ByteArray) (stop k : Nat) : ByteArray := file.set! (stop - k) 1

def zkStop : Nat := (pair sh (2 ^ (sh.sched.levels.getD 0 default).lanes + Zk.EXTRA_LANES) 0).size

-- R4: every grinding nonce is 0, and any other is refused, in both decoders.
#guard [false, true].all fun up => why (read up (flip (if up then canon else tagged) canon.size 152)) ==
  "grinding_nonce ≠ 0 (R4)"
#guard [false, true].all fun up => why (read up (flip (if up then canon else tagged) canon.size 64)) ==
  "claim_batch_grinding_nonce ≠ 0 (R4)"
#guard [false, true].all fun up => why (read up (flip (if up then canon else tagged) canon.size 16)) ==
  "consistency_batch_grinding_nonce ≠ 0 (R4)"
#guard why (Zk.decode sh (flip (zkFile sh) zkStop 152)) == "grinding_nonce ≠ 0 (R4)"
#guard why (Zk.decode sh (flip (zkFile sh) zkStop 64)) == "claim_batch_grinding_nonce ≠ 0 (R4)"
#guard why (Zk.decode sh (flip (zkFile sh) zkStop 16)) == "consistency_batch_grinding_nonce ≠ 0 (R4)"

end Flock.Test.ProofCanon
