import Flock
open Flock

#eval hexOf (Keccak.shake256 ByteArray.empty 32) == "46b9dd2b0ba88d13233b3feb743eeb243fcd52ea62b81b82b50c27646ed5762f"
#eval hexOf (RecCoins.verifierSalt "abc" "rec-coef" 3) == "4bd670368acaed941fec4749d855274022e5a746f3bf0b7ab4ccc27dd10dbc352be30d3fe9d4f0713d00de80e33615be237b063c14b65778d6653d229b563cd22c36cf8e9331846da6280731c3639af591083a347a7ae853ff222e27481b2640945cd1febc6d5e21c81dbc9161b4cc03576e4340c96505a191066af296deb0638a656552aa169cb90eccc62db5fc505db214b3b597f862358f8fe9b4a06954699426f121ef723c19aa50ea4b0003f74c885e9b0c0c381ae8aa173e74ee67bd52"
#eval (hexOf (Keccak.shake256 (ByteArray.mk (Array.replicate 135 120)) 8), hexOf (Keccak.shake256 (ByteArray.mk (Array.replicate 136 120)) 8), hexOf (Keccak.shake256 (ByteArray.mk (Array.replicate 300 120)) 8))
