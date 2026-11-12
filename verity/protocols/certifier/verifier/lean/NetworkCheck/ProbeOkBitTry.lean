/-
A probe, not for landing: `NetworkCheck.ProbeOkBit`'s rule tried on registrations, each held by its three forms,
with the salt rule and the root held to the Python driver's (`scan_zk.ok_salt`, `scan_zk.ok_root`).
-/
module

public import Flock.RegKnown
public import NetworkCheck.ProbeOkBit
public import NetworkCheck.Period
meta import Flock.RegKnown
meta import NetworkCheck.ProbeOkBit
meta import NetworkCheck.Period

@[expose] public section

namespace NetworkCheck.ProbeOkBit.Try

open Lean (Json)
open Flock

def hex (s : String) : ByteArray := (unhex s).toOption.getD ByteArray.empty

#guard hexOf ((okSalt 0).extract 0 8) == "f61f80289735b3b6"
#guard (okSalt 7).size == 192 && hexOf ((okSalt 7).extract 184 192) == "27e9fc9788b62bae"
#guard hexOf (okRoot 1 (hex "41f8dbf969654a4f0d5e4303c9a801ff5900462ed8cd344254fb57df69d4bd1bf142775bdc066e44c75f9edadc2453bb9afa66743914bb4201b3bfd17a4cf9e3")) ==
  "dd4ddeca0ca930e5c8438b4bccef1ea42fafeba05d3e58a0aa7618d451933ca838b1fbd9075379c688ac7ca3b5e5c99c006395ce32a89caeec20958d78f9fdf7"
#guard hexOf (okRoot 3 (hex "76c0ae7dc6cb99888aaf31cd85191ca00e123c10d0aed3fe8433f5daa281225084ed79279c158dc0b982761ffcc2a62bd9dfa962d802cfc674d68e1148a2375e")) ==
  "1aeb5a4ac1c8eda62ec1447d7246b182db03a334cabe175115ffb4b879814ab8265532c9d15fb4bc518a3b47ef6c11503c322825b8d537f77892036b7a414ab3"
#guard okRootCalls 1 == 7 && okRootCalls 14 == 115 && okRootCalls 16 == 127 && okRootCalls 30 == 243 && okRootCalls 32 == 255

/-- A registration held by the three forms, with the period check's entries ports left out: a line each. -/
def tried (name text : String) (T : Nat) : IO Unit := do
  let .ok reg := Json.parse text | IO.println s!"TRY {name}: not JSON"
  let entriesPort := NetworkCheck.Period.entriesPortName
  IO.println s!"TRY {name} T={T} walk: {verdict (okWalk reg T entriesPort)}"
  match okCached reg T entriesPort with
  | .error e => IO.println s!"TRY {name} T={T} cached: REFUSE {e}"
  | .ok cs =>
    IO.println s!"TRY {name} T={T} cached: ACCEPT, {cs.size} constants of leaves {cs.map (·.leaves)}"
    IO.println s!"TRY {name} T={T} carried, the table: {verdict (checkCarried cs)}"
    IO.println s!"TRY {name} T={T} carried, the comparison: {verdict (okCarried cs reg T entriesPort)}"

def reg0 : String :=
  "{\"program\": \"synthetic\", \"registered\": [], \"roots\": [{\"domain\": \"839e420b73e81bd94ccce19ef927249d0ee83b30cdfda644c535b21b0f19a6262117619d26c8046e53091db0ce11ee05b85a449b2cc1282a70c557d0319788e5\", \"leaves\": 1, \"port\": \"first/o0\", \"root\": \"35945fd71f3e042e68bb3f2f83459df3208b1727286a9421658c2ecfdee6f52841049e2b4dd580718eefab85a3068ae3add472beac34bcf40f03b20940d5a9a0\"}, {\"domain\": \"2cddd8495fe330352dcd4431892a9f64cf5e44881678e7963295ed4c288815d78e8b39c1504891add525b9be62af2138f1202e8470848fd0479682401f4de233\", \"leaves\": 1, \"port\": \"first/state\", \"root\": \"3faed7aa16def8d3045fe59d4b46a8987b3fe590a77158cf7c907a6ffee98b8b3b8dcec738cb53a761bcba47887d606ec1f23d2756d5f71adce1313af50731b5\"}, {\"domain\": \"6022e52a75ffa4c2281255ceee9ba8f3995f43e38e7858e32693db06d0da282cd06472aeea5690113c21826bec854002dab7e9fcd7093b7360735a1c993705be\", \"leaves\": 14, \"port\": \"ingress-0/o0\", \"root\": \"a850ba5b9641078619e24a48b4c2a12adb1a6f7f7ec6f2a98848da3c8a4a0253a3c97eed446d9f96980a981fa3b85f57402b7520cea54730ce541ee7363f9742\"}, {\"domain\": \"0b63ac5d5aa342d23286bfbce7a5928624e418d57ce48dfeb9fe28a8b4b23144b2180cc7f5c2aa7669cedd4c3ddc287cd8589f06c494bb467da63f5204f0285d\", \"leaves\": 14, \"port\": \"ingress-0/state\", \"root\": \"182fad726636425946daa6734c05eae90a44fb2d7d45b50a3e240319cf1cc2bab960a2f46a4fbcb15f09df304873c5661d4182798a0dbac785ebec3bc2e8c2b4\"}, {\"domain\": \"a67d3867166e1e3c688ddbdb7b23c4caf2c75d28f508184f1aa16e9118a9c097f29c3e5b2f4d71b19f486d6ace2f957014a978cf7a58712d8f1daf0ea8147b4f\", \"leaves\": 1, \"port\": \"last/o0\", \"root\": \"2d59d90bc6e950c9697062234176f8430a574103f17a0fe3511b9bb2725b4a870572fc12a953d15313cf65129729b25fa769466f3b255bc3dbafbac7b027dbde\"}, {\"domain\": \"d7214af54df2a2458926d76b09945f9c8ac8b41f088776aec8fe5b8b9065c155047e2749fafd7711e89c1219e67c3c9024ae97c333812ef8a31ab85e660a2a59\", \"leaves\": 1, \"port\": \"last/state\", \"root\": \"8c83e3cdd4121c799c92a4e38f2836f0eb5593910f5b3a758ecf42e73a429a0233118218178e3c5af9d8aa7b2a15784455e9f73a596acce224b434b4b40ddec1\"}, {\"domain\": \"dbf7d065a4d17512c629fbc9016deb37f983d9e51c62d37e7c3dda463c0976110dd0b3a3a4fff06b1acbf430cfe0d3b1fa92545a821a3206dd79f86fb455c775\", \"leaves\": 16, \"port\": \"egress-0/o0\", \"root\": \"d38417207773c4131997eca3bc1f4d8d67e186af101910208a2b09012c35b351e5c7822efd1c579a0391bc4dea6bd7d4371c4bd5412cdf37ebad8000fdd05578\"}, {\"domain\": \"cd12c315137f23aed7d3174d7db635725b976d717681738c5b93ace869232068d34a0e7b173e412b552141fb88434413121a1f5329dd6dd9e94e7f9a1396062e\", \"leaves\": 16, \"port\": \"egress-0/state\", \"root\": \"9d0e1794c89511360f23b37ae0e02c8f6664b20b638a249eee51fa8f4397790701242f5516a6f8b900ac36a79b9a5084574fc164a8d83a162ae3fdfcab92ef02\"}, {\"domain\": \"3c551410be5c9aa8e9bb9d520c1147bf18356cb6e5a874e7373ecdf736461237d7d709d29a00fc87c35cc2817bcfd143559906cccde748dc7093377c95d6e878\", \"leaves\": 16, \"port\": \"scan/state\", \"root\": \"698e245e63a99fd551cf9b79527dfa328eb2f2824584319017ec65cfca308d7937d179be69055a8507c3b8e3b78a035ab9c9894a0f988ba05f7256ff7f890012\"}, {\"domain\": \"a3a85ca104611a254285aef4bbb52321948a2436a8e8579256e749592b3b135b6e1adc7f9dfb55cfbdd74d33adc6ee85344c856d916906b1a6ad3f52d02bdb11\", \"leaves\": 16, \"port\": \"scan/prog\", \"root\": \"8c7748684f1cc801d25ab772363ab837fc5e8867acd77db8ea629fbcf499bdffcac2f5ecc7c9185db22dcfd1f4ba18382403f1574d4641f474a1c332f994de55\"}, {\"domain\": \"ab82200d2acc3c5e98a8798e481b2ccb9138a88ad67c9b4c6926d4f16d130b78e3401da4fbbf906a8ace17b9230a78ede3529d54704c4c45978fd7a06d558eb8\", \"leaves\": 16, \"port\": \"scan/view\", \"root\": \"dd8a0781c0f160e7836bc6b8eb246ba4eabf00c4a10f36c66b53899447369888b42f263c76d86f2a828d522b95e64c1e0efddfea68f7e9153a7f83692af08c3c\"}]}"

#eval tried "synthetic-T16" reg0 16

def reg1 : String :=
  "{\"program\": \"synthetic\", \"registered\": [], \"roots\": [{\"domain\": \"839e420b73e81bd94ccce19ef927249d0ee83b30cdfda644c535b21b0f19a6262117619d26c8046e53091db0ce11ee05b85a449b2cc1282a70c557d0319788e5\", \"leaves\": 1, \"port\": \"first/o0\", \"root\": \"35945fd71f3e042e68bb3f2f83459df3208b1727286a9421658c2ecfdee6f52841049e2b4dd580718eefab85a3068ae3add472beac34bcf40f03b20940d5a9a0\"}, {\"domain\": \"2cddd8495fe330352dcd4431892a9f64cf5e44881678e7963295ed4c288815d78e8b39c1504891add525b9be62af2138f1202e8470848fd0479682401f4de233\", \"leaves\": 1, \"port\": \"first/state\", \"root\": \"3faed7aa16def8d3045fe59d4b46a8987b3fe590a77158cf7c907a6ffee98b8b3b8dcec738cb53a761bcba47887d606ec1f23d2756d5f71adce1313af50731b5\"}, {\"domain\": \"6022e52a75ffa4c2281255ceee9ba8f3995f43e38e7858e32693db06d0da282cd06472aeea5690113c21826bec854002dab7e9fcd7093b7360735a1c993705be\", \"leaves\": 14, \"port\": \"ingress-0/o0\", \"root\": \"306ad2ea37615621ae9c60c200d3fb94c88f7699719acc87b6b41a2e62c8f18c8669031e9e6e45890a1f99585c6d4bedde8fe9f767d33c27e12eee8e17088562\"}, {\"domain\": \"0b63ac5d5aa342d23286bfbce7a5928624e418d57ce48dfeb9fe28a8b4b23144b2180cc7f5c2aa7669cedd4c3ddc287cd8589f06c494bb467da63f5204f0285d\", \"leaves\": 14, \"port\": \"ingress-0/state\", \"root\": \"182fad726636425946daa6734c05eae90a44fb2d7d45b50a3e240319cf1cc2bab960a2f46a4fbcb15f09df304873c5661d4182798a0dbac785ebec3bc2e8c2b4\"}, {\"domain\": \"a67d3867166e1e3c688ddbdb7b23c4caf2c75d28f508184f1aa16e9118a9c097f29c3e5b2f4d71b19f486d6ace2f957014a978cf7a58712d8f1daf0ea8147b4f\", \"leaves\": 1, \"port\": \"last/o0\", \"root\": \"2d59d90bc6e950c9697062234176f8430a574103f17a0fe3511b9bb2725b4a870572fc12a953d15313cf65129729b25fa769466f3b255bc3dbafbac7b027dbde\"}, {\"domain\": \"d7214af54df2a2458926d76b09945f9c8ac8b41f088776aec8fe5b8b9065c155047e2749fafd7711e89c1219e67c3c9024ae97c333812ef8a31ab85e660a2a59\", \"leaves\": 1, \"port\": \"last/state\", \"root\": \"8c83e3cdd4121c799c92a4e38f2836f0eb5593910f5b3a758ecf42e73a429a0233118218178e3c5af9d8aa7b2a15784455e9f73a596acce224b434b4b40ddec1\"}, {\"domain\": \"dbf7d065a4d17512c629fbc9016deb37f983d9e51c62d37e7c3dda463c0976110dd0b3a3a4fff06b1acbf430cfe0d3b1fa92545a821a3206dd79f86fb455c775\", \"leaves\": 16, \"port\": \"egress-0/o0\", \"root\": \"d38417207773c4131997eca3bc1f4d8d67e186af101910208a2b09012c35b351e5c7822efd1c579a0391bc4dea6bd7d4371c4bd5412cdf37ebad8000fdd05578\"}, {\"domain\": \"cd12c315137f23aed7d3174d7db635725b976d717681738c5b93ace869232068d34a0e7b173e412b552141fb88434413121a1f5329dd6dd9e94e7f9a1396062e\", \"leaves\": 16, \"port\": \"egress-0/state\", \"root\": \"9d0e1794c89511360f23b37ae0e02c8f6664b20b638a249eee51fa8f4397790701242f5516a6f8b900ac36a79b9a5084574fc164a8d83a162ae3fdfcab92ef02\"}, {\"domain\": \"3c551410be5c9aa8e9bb9d520c1147bf18356cb6e5a874e7373ecdf736461237d7d709d29a00fc87c35cc2817bcfd143559906cccde748dc7093377c95d6e878\", \"leaves\": 16, \"port\": \"scan/state\", \"root\": \"698e245e63a99fd551cf9b79527dfa328eb2f2824584319017ec65cfca308d7937d179be69055a8507c3b8e3b78a035ab9c9894a0f988ba05f7256ff7f890012\"}, {\"domain\": \"a3a85ca104611a254285aef4bbb52321948a2436a8e8579256e749592b3b135b6e1adc7f9dfb55cfbdd74d33adc6ee85344c856d916906b1a6ad3f52d02bdb11\", \"leaves\": 16, \"port\": \"scan/prog\", \"root\": \"8c7748684f1cc801d25ab772363ab837fc5e8867acd77db8ea629fbcf499bdffcac2f5ecc7c9185db22dcfd1f4ba18382403f1574d4641f474a1c332f994de55\"}, {\"domain\": \"ab82200d2acc3c5e98a8798e481b2ccb9138a88ad67c9b4c6926d4f16d130b78e3401da4fbbf906a8ace17b9230a78ede3529d54704c4c45978fd7a06d558eb8\", \"leaves\": 16, \"port\": \"scan/view\", \"root\": \"dd8a0781c0f160e7836bc6b8eb246ba4eabf00c4a10f36c66b53899447369888b42f263c76d86f2a828d522b95e64c1e0efddfea68f7e9153a7f83692af08c3c\"}]}"

#eval tried "synthetic-T16-other-root" reg1 16

end NetworkCheck.ProbeOkBit.Try
