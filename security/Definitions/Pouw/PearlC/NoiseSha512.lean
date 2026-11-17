module

public import Definitions.Pouw.PearlC.Instance
public import Definitions.Pouw.PearlC.Sha512

/-! Pearl-C's noise as h3's SHA-512 hashing samples it (`verity/protocols/resource_accounting/compute/pouw/prover/schemes/pearl_c.py`), with the salt's bytes and the weight's commitment root as parameters. -/

@[expose] public section

namespace Pouw.PearlC

open Pouw.Fp8Atom Pouw.PearlC.Sha512

namespace NoiseSha512

abbrev Salt32 : Type := Fin 32 → Fin 256

def Salt32.bytes (s : Salt32) : List ℕ := List.ofFn fun i => (s i).val

def label (role : String) : List ℕ := padTo 128 (ascii ("pearl-c/h3s/" ++ role) ++ [0])

def s512 (role : String) (msg : List ℕ) : List ℕ := (digest (label role ++ msg)).take 32

def rowSegs (len : ℕ) : List ℕ := List.replicate (len / 1024) 1024 ++ (if len % 1024 = 0 then [] else [len % 1024])

def rowDigest (row : List ℕ) : List ℕ :=
  let segs := rowSegs row.length
  let s := segs.length
  let q := padTo 128 (ascii "verity/sha512-row-seg/v1" ++ [0] ++ beBytes 8 (8 * row.length) ++ beBytes 8 s ++ beBytes 4 1024)
  let d := (List.range s).flatMap fun j =>
    digest (padTo 128 (ascii "verity/sha512-row-seg/v1/segment" ++ [0] ++ beBytes 8 j ++ beBytes 8 s ++
      beBytes 8 (8 * segs.getD j 0)) ++ (row.drop (1024 * j)).take (segs.getD j 0))
  digest (q ++ d)

/-- Each word little-endian. -/
def rowBytes (W : Codes) (k i : ℕ) : List ℕ := (List.range k).flatMap fun l => leBytes 4 (W i l)

def seedB (salt rootB : List ℕ) : List ℕ := s512 "seed-B" (salt ++ rootB)

def rowSeed (salt row rootB : List ℕ) (u i : ℕ) : List ℕ :=
  s512 "seed-A" (salt ++ rowDigest row ++ rootB ++ leBytes 8 (u * 2 ^ 32 + i))

def lineInts (raw : List ℕ) : List ℤ :=
  raw.map fun b => (if b / 128 % 2 = 1 then -1 else 1) * (((b % 128 : ℕ) : ℤ) + 1)

def lineScale (norm : ℚ) (xs : List ℤ) : ℚ :=
  bf16Rn (rn (bf16Rn (norm * 32) / bf16Rn (Nat.sqrt ((xs.map fun x => x.natAbs ^ 2).sum * 1024) : ℚ)))

def lineList (seed : List ℕ) (side factor line : ℕ) (norm : ℚ) : List Code :=
  let xs := lineInts (s512 "line" (s512 "noise-line" seed ++ [side, factor] ++ leBytes 4 line))
  let sc := lineScale norm xs
  xs.map fun x => e4m3 (bf16Rn (rn (bf16Rn x * sc)))

def sampleLine (seed : List ℕ) (side factor line : ℕ) (norm : ℚ) (t : Fin Params.pi.r) : Code :=
  (lineList seed side factor line norm).getD t.val 0

def unitNoise (salt rootB : List ℕ) (sh : Shape) (u : ℕ) (act : Codes) : Noise sh.m sh.n sh.k Params.pi.r where
  EA := fun i t => sampleLine (rowSeed salt (rowBytes act sh.k i) rootB u i) 0 0 0 256 t
  FA := fun l t => sampleLine (seedB salt rootB) 0 1 l 16 t
  EB := fun j t => sampleLine (seedB salt rootB) 1 0 j 256 t
  FB := fun l t => sampleLine (seedB salt rootB) 1 1 l 16 t

end NoiseSha512

open NoiseSha512

variable {Q R S : Type}

/-- The oracle is unused. -/
def sha512Sem (base : PearlCSem Q R S) (salt : S → List ℕ) (rootB : ℕ → List ℕ) : PearlCSem Q R S :=
  { base with noise := fun _ s U u act => unitNoise (salt s) (rootB (U.layout.fixedId u)) (U.layout.shape u) u act }

end Pouw.PearlC
