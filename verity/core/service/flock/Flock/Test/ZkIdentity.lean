/-
A `--zk` statement's digest hashes its identity as upstream's `Stmt::new` does, `serde_json::to_vec(&sorted(identity))`
(`Flock.identityBytes`): strings in UTF-8. The identity's `hooks` name `veil (ePrint 2026/683 §4.1, …)`, so `§` enters
the digest as its bytes `c2 a7`. Canonical JSON's escape (`canon`) would give another digest and Σ, and the prover's
`Hello` would fail R7. `HOOKS` is upstream's `zk_identity` at `mask_log` 14 (`flock-circuit.rs`), as `serde_json`
writes it.
-/
import Flock.Zk

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
  (Zk.tags (Tags.circuit.withCoins (.tree (Zk.coinSpec (Tags.circuit.tableNames 1)))) 14 1).identity

#guard HOOKS.toUTF8.size == 1720
#guard "§".toUTF8.data == #[0xc2, 0xa7] && (HOOKS.splitOn "2026/683 §4.1").length == 2
#guard (HOOKS.splitOn "\\u00a7").length == 1
#guard (bytesOf (Zk.hooks 14 false true)).data == HOOKS.toUTF8.data
#guard (String.fromUTF8? (bytesOf identity)).map (fun s => (s.splitOn ("\"hooks\":" ++ HOOKS)).length) == some 2
#guard (canon (Zk.hooks 14 false true)).toOption != some HOOKS

end Flock.Test.ZkIdentity
