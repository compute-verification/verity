/-
A `--zk` statement's digest hashes its identity as upstream's `Stmt::new` does, `serde_json::to_vec(&sorted(identity))`
(`Flock.identityBytes`): strings in UTF-8. The identity's `hooks` name `veil (ePrint 2026/683 §4.1, …)`, so `§` enters
the digest as its bytes `c2 a7`. Canonical JSON's escape (`canon`) would give another digest and Σ, and the prover's
`Hello` would fail R7. `HOOKS` is upstream's `zk_identity` at `mask_log` 14 (`flock-circuit.rs`), as `serde_json`
writes it.
-/
module

public import Flock.Zk
meta import Flock.Zk

@[expose] public section

namespace Flock.Test.ZkIdentity

open Lean (Json)

def HOOKS : String := String.join [
  "{\"compiler\":\"veil (ePrint 2026/683 §4.1, Remark 4.4) over gf(2^128)\",\"inner_proof\":\"one ",
  "batched <c, h> = t by a masked dot product (tau = <c, mu> committed by an hm96 leaf, opened after the ",
  "last coin; Y = h + beta mu)\",\"ligerito_level0\":\"veil figs. 8-9: every lane padded by (W_n + c) mu ",
  "with t_pad = 2 q_0 uniform coefficients (c = x); four padded uniform lanes, rep r folding its own ",
  "pair 2r, 2r + 1 after its lane rounds (rho_0, rho_1 F128-independent); the ood value, the lane-round ",
  "messages and the pair's sums padded; T' and ybar sent (uniform)\",\"ligerito_level0_queries\":\"q_0 = ",
  "277, 242, 229 at m = 25, 26, 27, else the least q_0 >= fast100's with q_0 log2(1/(sqrt(rho') + 1/50)) ",
  ">= 100; rho' = (L + 2 q_0) / 2L; refused below 100 bits and at m <= ",
  "24\",\"not_yet_masked\":[],\"pads\":\"zerocheck and lincheck: every sent value + a pad (os bytes, ",
  "getrandom); pads committed before the zerocheck's first coin\",\"pads_commitment\":\"rs over ",
  "gf(2^128), 192 padding symbols, rate 1/8, 168 queries, hm96-sha512/v1 leaves, sha512 ",
  "nodes\",\"product\":\"h_ab = h_a h_b committed; sacrificed against a committed triple (x, y, x ",
  "y)\",\"refused_coins\":[\"beta = 0\",\"mask-slot weights of rank < 128 k over the claims so ",
  "far\",\"rho_0, rho_1 dependent over F128\",\"a zerocheck outer coin r_j = ",
  "1\"],\"ring_switch\":\"non-region s_hat_v hidden by the mask slot's uniform words (2^14 bits per ",
  "block); the prover refuses coins whose weights do not span\",\"stage\":\"m2-prototype (committed ",
  "verifier coins, CPU)\",\"verifier_coins\":\"committed before the prover's first message ",
  "(coin-tree/hm96-sha512/v2): the prover's fresh nonce in hello; the root and the verifier's session ",
  "key in its answer; every coin opened against them, or the prover stops\",\"zk\":true}"]

def bytesOf (j : Json) : ByteArray := (identityBytes j).toOption.getD .empty

def identity : Json :=
  (Zk.tags Tags.circuit 14 1).identity

#guard HOOKS.toUTF8.size == 1720
#guard "§".toUTF8.data == #[0xc2, 0xa7] && (HOOKS.splitOn "2026/683 §4.1").length == 2
#guard (HOOKS.splitOn "\\u00a7").length == 1
#guard (bytesOf (Zk.hooks 14 false true)).data == HOOKS.toUTF8.data
#guard (String.fromUTF8? (bytesOf identity)).map (fun s => (s.splitOn ("\"hooks\":" ++ HOOKS)).length) == some 2
#guard (canon (Zk.hooks 14 false true)).toOption != some HOOKS

/-! ## The wide-leaf statement

Its `--zk` schedule is `fast100`'s up to `m = 31` and holds level 0 at `2^19` symbols from 32 to 35. Its identity and its
`Hello` are its own at every `m`, so a session of one version is refused as one of the other even where the two
schedules are one: the statement digest hashes the identity, Σ the digest, and `Hello` carries Σ and the profile, which
S2 compares with the record's byte for byte. -/

/-- A string field of a JSON object, `""` when absent. -/
def strOf (j : Json) (k : String) : String := (j.getObjValAs? String k).toOption.getD ""

def wide : Json := (Zk.tags Tags.circuitWideLeaf 14 1).identity

-- today's version reads the schedule it read
#guard (List.range 40).all fun m => reprStr (Zk.schedule false m) ==
  reprStr (match Zk.queries0 false m, fast100 m with
    | .ok q0, some sch =>
      let l0 := sch.levels[0]!
      let strata := decompose q0 l0.d
      (.ok ({ sch with levels := sch.levels.set! 0 { l0 with q := q0, strata, cap := strata.getD 0 0 } }, 2 * q0) :
        V (Schedule × Nat))
    | .error e, _ => .error e
    | _, none => .error s!"m = {m}: no fast100 schedule")
#guard [32, 33, 34, 35].all fun m => (Zk.queries0 false m).toOption == some 218
-- up to m = 31 the two versions' schedules are one
#guard (List.range 32).all fun m => reprStr (Zk.schedule true m) == reprStr (Zk.schedule false m)
-- 32 to 35: level 0 of 2^19 symbols with m − 25 lanes and 219 queries, the levels after it those of m = 31
#guard [32, 33, 34, 35].all fun m => match Zk.schedule true m with
  | .ok (s, tPad) => tPad == 438 && s.k0 == m - 25 && s.levels.map (·.q) == #[219, 106, 71, 53, 43, 36] &&
      s.levels.map (·.d) == #[19, 17, 15, 13, 11, 9] && s.levels[0]!.lanes == m - 25 &&
      s.levels[0]!.strata == decompose 219 19 && s.levels[0]!.cap == 7
  | .error _ => false
#guard [36, 37].all fun m => (Zk.schedule true m).toOption.isNone && (Zk.schedule false m).toOption.isNone
#guard (List.range 25).all fun m => (Zk.schedule true m).toOption.isNone
-- `queries0`'s rule at `L = 2^18`: 219 is the least count at 100 bits
#guard [218, 219].map (fun q : Nat =>
  let rho := (2 ^ 18 + 2 * q).toFloat / (2 ^ 19 : Nat).toFloat
  decide (q.toFloat * Float.log2 (1 / (Float.sqrt rho + 0.02)) ≥ 100)) == [false, true]
-- the identity: the profile and the query rule's text are the wide leaf's, and nothing else of the hooks differs
#guard strOf identity "profile" == "fast100" && strOf wide "profile" == Tags.WIDE_LEAF_PROFILE
#guard strOf (Zk.hooks 14 false true true) "ligerito_level0_queries" ==
  strOf (Zk.hooks 14 false true) "ligerito_level0_queries" ++
    "; wide leaf: level 0 of 2^19 symbols at m = 32..35 (k_0 = m - 25, L = 2^18), where q_0 = 219"
#guard bytesOf wide != bytesOf identity
#guard (bytesOf ((Zk.hooks 14 false true true).setObjVal! "ligerito_level0_queries" Json.null)).data ==
  (bytesOf ((Zk.hooks 14 false true).setObjVal! "ligerito_level0_queries" Json.null)).data
-- `Hello`: the two versions' differ at the same Σ, link size and coins
#guard (helloOf Tags.circuitWideLeaf ByteArray.empty 20 .seed).toOption != (helloOf Tags.circuit ByteArray.empty 20 .seed).toOption
#guard ((helloOf Tags.circuitWideLeaf ByteArray.empty 20 .seed).toOption.map
  fun h => (h.splitOn "\"profile\":\"fast100-wide-leaf\"").length) == some 2
#guard ((helloOf Tags.circuit ByteArray.empty 20 .seed).toOption.map
  fun h => (h.splitOn "\"profile\":\"fast100\"").length) == some 2

end Flock.Test.ZkIdentity
