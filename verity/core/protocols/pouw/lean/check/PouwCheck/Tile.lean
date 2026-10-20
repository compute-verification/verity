/-!
# The tile templates the verifier pins

`tiles`: each tile template a window's check item may name, with its work per tile, its check query's SHA-512 (one-stage's
`query_digest` of the template's instance query), and the one circuit, by SHA-512, a unit's tile proof of it is over: the
`circuit_sha512` of the proof's public file, which `flock-verify verify --zk` binds to the circuit it verifies against.
`paramsOf` refuses a check item or query that isn't one of these, and `opening` a proof over any other circuit, so this
table fixes the circuit a window's units are proved over, not the code that builds it (`verity.ml.circuits`, lowered by
C-Flock). A change to a template's Definition or to C-Flock's lowering changes its circuit's digest, which then has to
change here before the verifier accepts its proofs.

A salted template (`salted`, `core_check.Salted`) is one whose unit reads the window's certified salt: its pinned circuit's
first ports are R A rows, C B rows and the public row `pub`, and only a certified window opens its units (`opening`).
-/

namespace PouwCheck

/-- A salted template's unit: R A rows and C B rows, then its public row, each an `hm96-sha512/row/v2` row. -/
structure Salted where
  r : Nat
  c : Nat
  deriving DecidableEq, Repr

/-- A tile template the verifier accepts: its work per tile, its check query's and circuit's SHA-512, hex, and, for a
salted template, its unit's rows. -/
structure Tile where
  template : String
  tileWork : Nat
  query : String
  circuit : String
  salted : Option Salted := none
  deriving DecidableEq, Repr

/-- The served window's template (`served_commit.template`) at K = 4096 and at the test window's K = 128, the two
templates of the windows recorded before it, which the replay of those windows reads, and the salted H1 tile unit at
K = 64 (R = 4, C = 8: 32 dot products of K per tile). -/
def tiles : List Tile := [
  ⟨"Pc8TileSide_v1{K=4096,TM=1,TN=1,DEV='sm120'}", 4096,
    "491c9160f3c0d0c95ef14575e0966bfa9958de58886c3cc46e8008c767118926582acc2199a49dae85a32656cd2b11f90bffc1105aa9d660e98f66f75ccac4af",
    "39d540c10ed1148526fcdf9d01561fcf68e33427f28f55d2dc2f453c09f68696978a530b2c14054f4720af1710afee8c74331f45b4c9a7c0511bfc1e02017c12", none⟩,
  ⟨"Pc8TileSide_v1{K=128,TM=1,TN=1,DEV='sm120'}", 128,
    "63cd98cb86598a2d67169d1a5ea949957c941b3e11aad61aeb14e99fc1a2d7c7f5b7b19e7349bcae68d0ee7400cfbe89b8a6dd20f14319666eb7a7d81a0a6343",
    "84315c5940d5c61a4938182d41330c2f4febbf049ca9cdadd58ec9f1f9dd92a13d9e43501b37e9e0f32914e698667907d768ce49c2ba8be026454bdff78baf24", none⟩,
  ⟨"Pc8TileHidden_v1{K=4096,TM=1,TN=1,DEV='sm120'}", 4096,
    "f48d5a1f247cb17063bcea693733db3d26e91be34967645486ea20df1336b4d9ae14aaf242ba4437f0d2a0123aff81f7ab31d18d437f4abe73363a1ac7bf65fb",
    "5ec6002f654337db2051fd6a98b7ec122a9f33e2d06180012177195674233d7c811c2ffd8ef49bc9a586dcf8e61ee1b16914b9f36e1bd378f9e53d0d40f4926d", none⟩,
  ⟨"Pc8TileLeafH3_v1{K=4096,DEV='sm120'}", 4096,
    "0f93d0dce725a5269061b91a1eb39ad294a1369c6f47814880acc74cef30aad734b0067d1cc7f5347bf9db4279a6519eb0abfa000c581759b14fd0ed5a7f9647",
    "c4fb8b6c5ce91ad0bc750f315700491911f678ad219cd370baf4359dc751954c9955590eb50304a166e1cc103ffd0d431de31f03c29ad5b6d119a33502b004af", none⟩,
  { template := "PouwH1Tile_v1{EXACT=GemmAccumulatorS8_v2{K=64,DOT=BlackwellS8ImmaDot32_v2},R=4,C=8,PAIRS=((0,0),(0,1)," ++
      "(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(1,0),(1,1),(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(2,0),(2,1),(2,2),(2,3)," ++
      "(2,4),(2,5),(2,6),(2,7),(3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(3,6),(3,7))}"
    tileWork := 64 * 32
    query := "1083a2eb9d13f77afdb936ecac9e780aa63b7f993795a8ff59d9d05ca7259c455c1220a833979cd975f907e97d9045033c5427950b7953dd33ac26cfdef44631"
    circuit := "571e56a77368c64505e46a2affd6360d869cf17d272f86926bbb6d8974f2c39177b351e15c20bbf8fd78cbb87a4951f9e3a1062edab8e4a57905eb99e6074b2f"
    salted := some ⟨4, 8⟩ }]

/-- The pinned tile of `template`, if any. -/
def tile? (template : String) : Option Tile := tiles.find? (·.template == template)

end PouwCheck
