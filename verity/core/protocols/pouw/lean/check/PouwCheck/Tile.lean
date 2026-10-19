/-!
# The tile templates the verifier pins

`tiles`: each tile template a window's check item may name, with its work per tile, its check query's SHA-512 (one-stage's
`query_digest` of the template's instance query), and the one circuit, by SHA-512, a unit's tile proof of it is over: the
`circuit_sha512` of the proof's public file, which `flock-verify verify --zk` binds to the circuit it verifies against.
`paramsOf` refuses a check item or query that isn't one of these, and `opening` a proof over any other circuit, so this
table fixes the circuit a window's units are proved over, not the code that builds it (`verity.ml.circuits`, lowered by
C-Flock). A change to a template's Definition or to C-Flock's lowering changes its circuit's digest, which then has to
change here before the verifier accepts its proofs.
-/

namespace PouwCheck

/-- A tile template the verifier accepts: its work per tile, and its check query's and circuit's SHA-512, hex. -/
structure Tile where
  template : String
  tileWork : Nat
  query : String
  circuit : String
  deriving DecidableEq, Repr

/-- The served window's template (`served_commit.template`) at K = 4096 and at the test window's K = 128, and the two
templates of the windows recorded before it, which the replay of those windows reads. -/
def tiles : List Tile := [
  ⟨"Pc8TileSide_v1{K=4096,TM=1,TN=1,DEV='sm120'}", 4096,
    "491c9160f3c0d0c95ef14575e0966bfa9958de58886c3cc46e8008c767118926582acc2199a49dae85a32656cd2b11f90bffc1105aa9d660e98f66f75ccac4af",
    "39d540c10ed1148526fcdf9d01561fcf68e33427f28f55d2dc2f453c09f68696978a530b2c14054f4720af1710afee8c74331f45b4c9a7c0511bfc1e02017c12"⟩,
  ⟨"Pc8TileSide_v1{K=128,TM=1,TN=1,DEV='sm120'}", 128,
    "63cd98cb86598a2d67169d1a5ea949957c941b3e11aad61aeb14e99fc1a2d7c7f5b7b19e7349bcae68d0ee7400cfbe89b8a6dd20f14319666eb7a7d81a0a6343",
    "84315c5940d5c61a4938182d41330c2f4febbf049ca9cdadd58ec9f1f9dd92a13d9e43501b37e9e0f32914e698667907d768ce49c2ba8be026454bdff78baf24"⟩,
  ⟨"Pc8TileHidden_v1{K=4096,TM=1,TN=1,DEV='sm120'}", 4096,
    "f48d5a1f247cb17063bcea693733db3d26e91be34967645486ea20df1336b4d9ae14aaf242ba4437f0d2a0123aff81f7ab31d18d437f4abe73363a1ac7bf65fb",
    "5ec6002f654337db2051fd6a98b7ec122a9f33e2d06180012177195674233d7c811c2ffd8ef49bc9a586dcf8e61ee1b16914b9f36e1bd378f9e53d0d40f4926d"⟩,
  ⟨"Pc8TileLeafH3_v1{K=4096,DEV='sm120'}", 4096,
    "0f93d0dce725a5269061b91a1eb39ad294a1369c6f47814880acc74cef30aad734b0067d1cc7f5347bf9db4279a6519eb0abfa000c581759b14fd0ed5a7f9647",
    "c4fb8b6c5ce91ad0bc750f315700491911f678ad219cd370baf4359dc751954c9955590eb50304a166e1cc103ffd0d431de31f03c29ad5b6d119a33502b004af"⟩]

/-- The pinned tile of `template`, if any. -/
def tile? (template : String) : Option Tile := tiles.find? (·.template == template)

end PouwCheck
