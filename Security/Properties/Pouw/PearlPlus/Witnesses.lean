import PouwPearlPlus

/-!
# The instances Pearl+'s witnesses are stated at

`Proofs.Properties.Pouw.PearlPlus.Witnesses` proves each Pearl+ property's `honest` and `nontrivial` witnesses
(`lean-audit.json`'s `witnesses`); the instances they read are here, so their statements read only this tree and the code.
-/

namespace Properties.Pouw.PearlPlus

open PouwCommit PouwCheck.PearlPlus

/-- The hm96 tile tree of one position that `rowsDomain` builds under a zero binding, owner −1 (a transcript's). -/
def witnessDomain : RowsDomain :=
  match rowsDomain ⟨Array.replicate 64 0⟩ (-1) 1 HM96 true with
  | .ok d => d
  | .error _ => ⟨⟨.sha512, 0, .empty⟩, true, true⟩

/-- The opening of position 0 of a tile tree whose value is the 64-byte digest `b, 0, …, 0` and a zero salt. -/
def witnessOpening (b : UInt8) : LeafOpening := ⟨0, ⟨#[b] ++ Array.replicate 63 0⟩ ++ ⟨Array.replicate 192 0⟩, []⟩

/-- A y check that accepts every served word `y`, whatever its U and α: `Rule.yOf` at the verifier that accepts
everything. -/
def acceptAllYOf (y _u _alphaA _alphaB : Nat) : Option Nat := some y

/-- The coin commitment of epoch 0 under an empty root. -/
def witnessCoins : CoinCommitment := ⟨0, .empty⟩

/-- A run that retains no pass. -/
def emptyRun : Input := { coins := witnessCoins, manifests := [], draws := 1 }

/-- `plan`'s refusal of a run that retains no pass. -/
def emptyRefusal : Refusal := { reason := "the run holds no pass with calls" }

/-- A prover that opens no tile. -/
def noOpening (_pass _tile : Nat) : Except String TileOpening := .error "no opening"

/-- A run verdict that accepts every run, with no plan: `check` at the verifier that accepts everything. -/
def acceptAllCheck (_i : Input) (_opener : Nat → Nat → Except String TileOpening) : RunAnswer := { accept := true }

/-- The plan that a `plan` accepting every run returns: no passes and an empty root. -/
def acceptAllPlan : Plan := ⟨[], .empty, .empty, [], 0⟩

/-- A record of the pass that a `load` accepting every manifest returns: one 1 × 1 × 1 call with two voluntary A rows. -/
def acceptAllRecord : UnitRecord := { matmul := ⟨0, 1, 1, 1, 0⟩, realRows := 1, kKept := 1, voluntaryA := [0, 0] }

/-- A pass with no calls, so no tiles. -/
def noCallsPass : Pass := default

/-- A test instance of the scheme, as a manifest's `instance` sets it: `k_min` 128 and 1 × 1 tiles. -/
def capScheme : Scheme := { kMin := 128, tileM := 1, tileN := 1 }

/-- One 1 × 128 × 1 call, its one tile `[0] × [0]`: its A row real, both rows kept whole and credited. -/
def capRecord : UnitRecord := { matmul := ⟨0, 1, 128, 1, 0⟩, realRows := 1, kKept := 128 }

/-- One formed row of 128 words: each word 1.0 and each code E4M3 1.0, with ρ = α = 1.0. -/
def capSide : Words.Formed :=
  ⟨#[Array.replicate 128 0x38], #[0x3F800000], #[0x3F800000], #[#[]], #[Array.replicate 128 0x3F800000]⟩

/-- `capRecord`'s tile formed: both rows `capSide`. -/
def capOps : Words.Ops := ⟨capSide, capSide, #[], #[]⟩

/-- `capSide` with 32 codes, less than one promotion group of 128, so the debit replay refuses its words. -/
def shortSide : Words.Formed := { capSide with codes := #[Array.replicate 32 0x38] }

/-- `capRecord`'s tile with `shortSide`'s codes. -/
def shortOps : Words.Ops := ⟨shortSide, shortSide, #[], #[]⟩

/-! ## One honest pass

One pass of the deployed scheme (`pearl-c-h100-v1-h3`, `k_min` 1,024, 64 × 64 tiles): one call of A (16 × 1,024) against
its weight's B (64 × 1,024), each element a BF16 value drawn from a standard normal, proved by the honest prover
(`clear_audit.Prover`) and declared by it (`clear_work.declare`: no row excluded), whose one tile `clear_audit.Verifier`
and `clear_work.check_opened` pass. Every hm96 salt in it is zero, the challenger's (which its entropy may give) and the
prover's rows' and leaf's. Its epoch's coins are the challenger's of epoch 0, the salt coin 0x11…11 and the draw coin
0x22…22. The digests below are the Python's for these values, and the kernel computes the Lean's in each witness. -/

/-- Bytes from hex, as `bytes.fromhex` reads them. -/
def hexBytes (s : String) : ByteArray := (pyFromHex s).toOption.getD .empty

/-- The root of the epoch's coin tree. -/
def honestCoinsRoot : String :=
  "16c3f3b6cefac407b74009f3250c4499d9649728c25a0038a8b3011b28d43d7d62a8848ca05dd9c8a50bde90124cffc7867492046358d2e6ff0143557a382b19"

/-- The draw coin's position's value in the coin tree, the salt coin's opening's one sibling. -/
def honestCoinSibling : String :=
  "19cf1c75456f465a808221091bdc5f6d5d4eec4f20b304492f839d6e7b8f42c0ff5d82ad71b4be1bab4ae67f69d0f1729666bca17b98ad35a05e737f8d5657c5"

/-- The root of weight 0's B rows. -/
def honestRootB : String :=
  "746b809ce5f8479b57f8f05fbc396e247942bea80d1cdd8a09d290eeb61201548284e97e60bd17f7e728033b2a4955681301c51f7c9e68701abba2b03765fea9"

/-- The weights' root over weight 0's. -/
def honestWeightsRoot : String :=
  "5808746767aa035e79a2708d7ee82f8abf8c9c3b41f9f6cabfa2715fbde6e12457c01a11ffda028221a3130aeb25cc796531bbc0cd32f2264ca082dbde9336d1"

/-- The epoch's salt from its salt coin. -/
def honestSalt : String := "83fb499dc6f904193523a178e2000946cf1cbcca5f0d3d7a231078c8753f67c6"

/-- The root of the call's A rows. -/
def honestRootA : String :=
  "238023299830e5b31dd5922657d88d468864a466ab240391aa57488c78e4a3b8ba5bda45486cacc72798db71cf4a2104903d6c01b88ae852fd3580587485a946"

/-- The records root the pass published. -/
def honestRecordsRoot : String :=
  "2eaf3bc2cd4601c7b5aa503c47e42a9c98c46677b55e81b845020f8f6d1663abea0d23029e591470d09d1074fd9d406cd6b2e55b46d82019e67e2d7aa725201b"

/-- The root of the pass's transcript tree, over its one tile's leaf. -/
def honestTranscriptRoot : String :=
  "b7c34fb15b6e08ef6a689eb67becfbcea9ccbf8a212f710b40e13ca678809dc7a3283764b46e5204b1d3f979ec81d3ad4d160cb9fdd65786bd69694669b17761"

/-- The epoch's coin commitment, the verifier's copy. -/
def honestCoins : CoinCommitment := ⟨0, hexBytes honestCoinsRoot⟩

/-- The pass's manifest: what `load` reads of it. -/
def honestManifest : Lean.Json := Lean.Json.mkObj [
  ("schema", .str MANIFEST_SCHEMA),
  ("scheme", .str NAME),
  ("coins", Lean.Json.mkObj [("scheme", .str SCHEME), ("epoch", .num 0), ("root", .str honestCoinsRoot)]),
  ("salt_opening", Lean.Json.mkObj [("epoch", .num 0), ("label", .str "salt"), ("after", .str honestWeightsRoot),
    ("coin", .str (hex ⟨Array.replicate 32 0x11⟩)), ("salt", .str (hex ⟨Array.replicate 192 0⟩)),
    ("path", .arr #[.str honestCoinSibling])]),
  ("salt", .str honestSalt),
  ("weights", Lean.Json.mkObj [("0", Lean.Json.mkObj [("root", .str honestRootB), ("k", .num 1024), ("excluded_b", .arr #[])])]),
  ("matmuls", .arr #[Lean.Json.mkObj [("index", .num 0), ("m", .num 16), ("k", .num 1024), ("n", .num 64), ("weight", .num 0),
    ("real_rows", .num 16), ("k_kept", .num 1024), ("excluded_a", .arr #[]), ("root_a", .str honestRootA)]]),
  ("records_root", .str honestRecordsRoot),
  ("commitment", Lean.Json.mkObj [("root", .str honestTranscriptRoot), ("count", .num 1)])]

/-- The pass's one record: every row real and credited, the call's whole k kept. -/
def honestRecord : UnitRecord := { matmul := ⟨0, 16, 1024, 64, 0⟩, realRows := 16, kKept := 1024 }

/-- The pass `load` reads from `honestManifest`. -/
def honestPass : Pass where
  scheme := {}
  salt := hexBytes honestSalt
  weightRoots := [(0, hexBytes honestRootB)]
  weightsRoot := hexBytes honestWeightsRoot
  records := [honestRecord]
  rootsA := [hexBytes honestRootA]
  transcriptRoot := hexBytes honestTranscriptRoot
  count := 1
  recordsRoot := hexBytes honestRecordsRoot
  yRoots := [none]

/-- The verifier's 32-byte source of the draw (the Pearl+ difftest's). -/
def honestSource : ByteArray := "verifierverifierverifierverifier".toUTF8

/-- A run of the one honest pass at `draws` draws, on the verifier's source. -/
def honestRun (draws : Nat) : Input := { coins := honestCoins, manifests := [honestManifest], draws, source := some honestSource }

end Properties.Pouw.PearlPlus
