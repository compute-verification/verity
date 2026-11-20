import BandChainModel
import BandEncDec
import BandPubEncoder

/-!
# Lean-evaluated vectors for `verity.protocols.accounting.space.pous.band`

Evaluates `chainBand`'s own definitions (`lean/submissions/band-chain/BandChainModel.lean`) on small cases and prints
them as JSON (`tests/fixtures/lean_band_vectors.json`): the band's parents newest first (`parentsDesc (bandDAG B d)`),
the encoder (`labelG`) and decoder formula (`absorbedG`, `ovKey`) with the toy `Π₂` of `PousDenseCheck.lean` and
its 4-bit `domSmall`, and encode after decode on codes that are not encodings (`labelG_dec`, the lemma behind
`chainBand_encDec` in `BandEncDec.lean`).  The `rfl` theorems tie `labelG` and the decoder's formula to `chainBand`'s
`enc` and `dec`; `encDec_12` and `pub_12` pin the secure-first point's `EncDec` and public-encoder guarantee
(`band_pub_meets_64_d12`, given `ChainExPostFactoG`).
Compiled `ovStep` recomputes its call on every bit read, so the label cases are small.  Regenerate with
`tests/lean/generate_band.sh`.
-/

namespace PousBandCheck

open Pous PousSponge PousBandChain

theorem enc_eq (B d ℓ c : ℕ) [NeZero B] (dm : ℕ → Bits c) (W : Bits (B * ℓ)) (π : Equiv.Perm (Bits (ℓ + (ℓ + c)))) :
    ((chainBand B d ℓ c dm).enc W π ()).2 = labelG (bandDAG B d) (fun x => π x) dm (Dense.blocks B ℓ W) := rfl

theorem dec_eq (B d ℓ c : ℕ) [NeZero B] (dm : ℕ → Bits c) (pp : Bits 0) (C : Fin B → Bits ℓ)
    (f : (idealPerm (ℓ + (ℓ + c))).Qry → Bits (ℓ + (ℓ + c))) :
    (chainBand B d ℓ c dm).dec pp C f =
      codeEquiv B ℓ fun i => Labelling.bxor (C i) (ovKey (fun x => f (.inl x)) (dm i) (absorbedG (bandDAG B d) C i)) :=
  rfl

theorem encDec_12 (salt : Bits 256) : (chainBand (2 ^ 9) 12 524288 512 (PousSponge.dom salt)).EncDec :=
  chainBand_encDec _ _ _ _ _

theorem pub_12 (hepf : ChainExPostFactoG) : ChainBandPubMeets64 12 := band_pub_meets_64_d12 hepf

def bitsNat {w : ℕ} (b : Bits w) : ℕ := (List.finRange w).foldl (fun a i => if b i then a + 2 ^ i.val else a) 0

def natBits (w n : ℕ) : Bits w := fun i => n.testBit i.val

def toyF (w : ℕ) (x : Bits w) : Bits w := natBits w ((11400714819323198485 * bitsNat x + 12345 + w) % 2 ^ w)

def domSmall (salt : Bits 2) (i : ℕ) : Bits 4 := Fin.append salt fun j : Fin 2 => i.testBit j

def wOf (B ℓ seed : ℕ) : Fin B → Bits ℓ := fun i j => (i.val * 13 + j.val * 5 + seed) % 7 < 3

def natsJson (xs : List ℕ) : String := "[" ++ ", ".intercalate (xs.map fun x => s!"\"{x}\"") ++ "]"

def idxJson (xs : List ℕ) : String := "[" ++ ", ".intercalate (xs.map toString) ++ "]"

def parentsCase (B d : ℕ) : String :=
  let rows := (List.finRange B).map fun v => idxJson ((parentsDesc (bandDAG B d) v).map (·.val))
  "{" ++ s!"\"B\": {B}, \"d\": {d}, \"desc\": [{", ".intercalate rows}]" ++ "}"

def labelCase (B d ℓ seed : ℕ) : String :=
  let salt : Bits 2 := fun i => (i.val + seed) % 3 = 0
  let dm := domSmall salt
  let w := ℓ + (ℓ + 4)
  let G := bandDAG B d
  let Wb := wOf B ℓ seed
  let C := labelG G (toyF w) dm Wb
  let dec := codeEquiv B ℓ fun i => Labelling.bxor (C i) (ovKey (toyF w) (dm i) (absorbedG G C i))
  let blocks := (List.finRange B).map fun i => bitsNat (C i)
  "{" ++ s!"\"B\": {B}, \"d\": {d}, \"l\": {ℓ}, \"seed\": {seed}, \"salt\": \"{bitsNat salt}\", " ++
    s!"\"w\": \"{bitsNat (codeEquiv B ℓ Wb)}\", \"blocks\": {natsJson blocks}, \"decode\": \"{bitsNat dec}\"" ++ "}"

/-- A code `C` that is not an encoding: its decoding `W'`, and the labels of `W'`, which `labelG_dec` says are `C`. -/
def encDecCase (B d ℓ seed : ℕ) : String :=
  let salt : Bits 2 := fun i => (i.val + seed) % 3 = 0
  let dm := domSmall salt
  let w := ℓ + (ℓ + 4)
  let G := bandDAG B d
  let C : Fin B → Bits ℓ := fun i j => (i.val * 3 + j.val * 7 + seed) % 5 < 2
  let W' : Fin B → Bits ℓ := fun i => Labelling.bxor (C i) (ovKey (toyF w) (dm i) (absorbedG G C i))
  let again := labelG G (toyF w) dm W'
  let nats (X : Fin B → Bits ℓ) := natsJson ((List.finRange B).map fun i => bitsNat (X i))
  "{" ++ s!"\"B\": {B}, \"d\": {d}, \"l\": {ℓ}, \"seed\": {seed}, \"salt\": \"{bitsNat salt}\", " ++
    s!"\"code\": {nats C}, \"w\": \"{bitsNat (codeEquiv B ℓ W')}\", \"reencode\": {nats again}" ++ "}"

end PousBandCheck

open PousBandCheck in
def main : IO Unit := do
  let parents := [parentsCase 6 1, parentsCase 8 3, parentsCase 16 5, parentsCase 20 12, parentsCase 4 12]
  let labels := [labelCase 3 1 3 1, labelCase 4 1 2 2, labelCase 3 1 8 3]
  let encdec := [encDecCase 3 1 3 5, encDecCase 4 1 2 6, encDecCase 3 1 8 7]
  IO.println ("{\"parents\": [\n" ++ ",\n".intercalate parents ++ "\n], \"labels\": [\n" ++ ",\n".intercalate labels ++
    "\n], \"encdec\": [\n" ++ ",\n".intercalate encdec ++ "\n]}")
