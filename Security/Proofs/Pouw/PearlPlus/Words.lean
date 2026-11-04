import PouwPearlPlus.Words
import Definitions.Pouw.PearlC.DeviceRev1
import Definitions.Pouw.PearlC.RowSem
import Definitions.Pouw.PearlC.NoiseSha512
import Proofs.Pouw.PearlPlus.Noise

/-!
# Pearl+'s bridge, part (b): a tile's words under -h3's noise are the semantics'

`PouwCheck.PearlPlus.Words.encode` and `checked` form a tile's operands from its opened rows and compute its checked words
C̃ and U. The game reads the reference words of `pearlCProtocolDevRev1 devH100 sem ρ` at the deployed semantics `semH3`:
Pearl-C at `pc8`'s row arithmetic on core's primitives and the H100 atom, with -h3's noise from SHA-512
(`sha512Sem`). `WordsAreTheSemantics` says the check's words are those values on every word of a credited row.

It composes the noise (`Noise.lean`), the row statistics (`StatsAreTheRow`), the forming, the chain (the H100 chain is
`chainWordP? hopperE4m3K32 4`, whose value `TileValues.chainWordP_value` gives) and the peel (`peelExactP` at the H100).
The check reads a row's raw words and the semantics their values re-encoded (`wordsVal`, `encRow`), so a `−0` word and
an A′ or B̃ code `0x80` agree with the semantics' `+0` in value only; every statement here is on values (`wordQ`).
-/

namespace Pouw.PearlPlus.Bridge

open Pouw.PearlC Pouw.Fp8Atom PouwCheck.PearlPlus

/-- Row `i` of a matrix of FP32 words with `k` columns as the bytes a row opening carries (`NoiseSha512.rowBytes`). -/
def rowBytesOf (W : Codes) (k i : ℕ) : ByteArray := ⟨((NoiseSha512.rowBytes W k i).map Nat.toUInt8).toArray⟩

/-- **Pearl+'s semantics**: Pearl-C at `pc8`'s row arithmetic on core's primitives and the H100 atom, with -h3's noise
from SHA-512 at the salt's bytes and the weights' roots. -/
def semH3 {Q R S : Type} (saltOf : S → List ℕ) (rootOf : ℕ → List ℕ) : PearlCSem Q R S :=
  sha512Sem (RowCircuit.pc8Sem RowCircuit.coreOps hopperE4m3K32
    fun _ _ _ _ _ => ⟨fun _ _ => 0, fun _ _ => 0, fun _ _ => 0, fun _ _ => 0⟩) saltOf rootOf

/-- Pearl+'s protocol at the H100 record under rev1, at the deployed semantics and cap `ρ`. -/
noncomputable def protocolH3 {Q R S : Type} (saltOf : S → List ℕ) (rootOf : ℕ → List ℕ) (ρ : ℚ) : Protocol Q R S :=
  pearlCProtocolDevRev1 devH100 (semH3 saltOf rootOf) ρ

/-- Pearl+'s audit tiles at the H100 record under rev1, at the deployed semantics and cap `ρ`. -/
noncomputable def tilesH3 {Q R S : Type} (saltOf : S → List ℕ) (rootOf : ℕ → List ℕ) (ρ : ℚ) : TileRules Q R S :=
  pearlCTilesDevRev1 devH100 (semH3 saltOf rootOf) ρ

/-- A row of FP32 words as values, the way `wordsVal` reads a committed row. -/
def rowVals (k : ℕ) (x : Array ℕ) : Fin k → ℚ := fun l => (wordQ x[l.val]!).getD 0

/-- **(b) The row statistics are `pc8`'s**: on a row of `k` finite FP32 words, `s5_stats`' ρ, α and β are `pc8`'s
`rowRho`, `alpha8` and `beta8` of the row's values re-encoded. -/
def StatsAreTheRow : Prop :=
  ∀ (k : ℕ) (x : Array ℕ) (s ρw αw βw : ℕ), x.size = k → 0 < k → (∀ l < k, x[l]! < 2 ^ 32 ∧ (wordQ x[l]!).isSome) →
    Words.s5Stats x = .ok (s, ρw, αw, βw) →
      ρw = RowCircuit.rowRho RowCircuit.coreOps .id (RowCircuit.encRow k (rowVals k x)) k ∧
        αw = RowCircuit.alpha8 RowCircuit.coreOps (RowCircuit.encRow k (rowVals k x)) k ∧
        βw = RowCircuit.beta8 RowCircuit.coreOps (RowCircuit.encRow k (rowVals k x)) k

/-- **(b) A tile's words are the semantics'**: for a unit `u` of `U` with salt and weight root the semantics' bytes, a
tile's rows and columns in the unit, opened rows that are the commitment's activation rows and the weight's rows, and
the check's noise, operands and checked words values (no rejection): each word of a row that passes the rules has the
reference ticket and useful values. -/
def WordsAreTheSemantics : Prop :=
  ∀ {Q R S : Type} (saltOf : S → List ℕ) (rootOf : ℕ → List ℕ) (ρ : ℚ) (H : Q → R) (s : S) (U : Workload) (u : ℕ)
    (act : Codes) (salt rootB : ByteArray) (wn : Words.FixedNoise) (rows cols : List ℕ) (ops : Words.Ops)
    (ch : Words.Checked),
    bytesNat salt = saltOf s → bytesNat rootB = rootOf (U.layout.wid u) →
    (∀ i ∈ rows, i < (U.layout.shape u).m) → (∀ j ∈ cols, j < (U.layout.shape u).n) →
    (∀ i ∈ rows, ∀ l < (U.layout.shape u).k, act i l < 2 ^ 32) →
    (∀ j ∈ cols, ∀ l < (U.layout.shape u).k, U.weight (U.layout.wid u) j l < 2 ^ 32) →
    Words.fixedNoise salt rootB (U.layout.shape u).k = .ok wn →
    Words.encode salt rootB wn u (U.layout.shape u).k rows (rows.map (rowBytesOf act (U.layout.shape u).k)) cols
      (cols.map (rowBytesOf (U.weight (U.layout.wid u)) (U.layout.shape u).k)) = .ok ops →
    Words.checked ops = .ok ch →
    ∀ x < rows.length, (semH3 saltOf rootOf : PearlCSem Q R S).RowOK (U.layout.shape u).m (U.layout.shape u).k act
        rows[x]! →
      ∀ z < cols.length,
        wordQ ch.c[x]![z]! = (protocolH3 saltOf rootOf ρ : Protocol Q R S).ticket H s U u act rows[x]! cols[z]! ∧
          wordQ ch.u[x]![z]! = (protocolH3 saltOf rootOf ρ : Protocol Q R S).useful H s U u act rows[x]! cols[z]!

end Pouw.PearlPlus.Bridge
