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

end Properties.Pouw.PearlPlus
