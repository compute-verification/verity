import PebblingMeets.P3Meets

/-!
# Lean-evaluated vectors for `verity.protocols.accounting.space.pous`

Evaluates the model's own definitions on small cases and prints them as JSON (`tests/fixtures/lean_vectors.json`):
`topLabel'`/`c1'`/`c2'` (`pebbling/ColumnAwareDraft.lean`), `transpose`, `labXor`, `decodeLower`
(`Definitions.Pous.Model.Invertible`), `tagOf`, the decoder, `bandGraph`'s parents, and `colRound`.  The primitive is a
computable toy `ω = (toyH, toyP)`; `tests/test_pous_lean.py` plugs the same toy into the Python.  `decodeW'`, `Wlab'`
and `answer'` are `P3Meets.decodeW`, `P3Meets.Wlab` and `(p3Model' n wc).answer` for every `n` (checked by `rfl` at
`n = 220` below), since those are fixed at `220` or noncomputable.  Regenerate with `tests/lean/generate.sh`.
-/

namespace PousRefCheck

open Pous Pous.Invertible PousColumnAware PousP3Column

variable {n wc : ℕ}

def answer' (ω : Omega' n wc) : (p3Model' n wc).Qry → Lab n wc
  | .inl x => ω.1 x
  | .inr (.inl y) => ω.2 y
  | .inr (.inr z) => ω.2.symm z

theorem answer'_eq (ω : Omega' n wc) : answer' ω = (p3Model' n wc).answer ω := by
  funext q; rcases q with q | q | q <;> rfl

def parentsOf' (G : Labelling.TopoDAG n) (L : Fin n → Lab n wc) (v : Fin n) : Fin n → Option (Lab n wc) :=
  fun u => if _h : u ∈ G.par v then some (L u) else none

def decodeW' (G : Labelling.TopoDAG n) (tag : Bits 256) (c2 : Fin n → Lab n wc)
    (f : (p3Model' n wc).Qry → (p3Model' n wc).Ans) : Fin n → Lab n wc :=
  let x2 : Fin n → Lab n wc := fun v => labXor (f (.inr (.inr (c2 v)))) (f (.inl (tag, 1, v, parentsOf' G c2 v)))
  let c1 := transpose x2
  fun u => labXor (f (.inr (.inr (c1 u)))) (f (.inl (tag, 0, u, parentsOf' G c1 u)))

def Wlab' (n wc : ℕ) (W : Bits (n * (n * wc))) : Fin n → Lab n wc :=
  fun u => (codeEquiv n wc).symm ((codeEquiv n (n * wc)).symm W u)

theorem decodeW'_eq : decodeW' (n := 220) (wc := 37) PousP3Meets.G = PousP3Meets.decodeW := rfl
theorem Wlab'_eq : Wlab' 220 37 = PousP3Meets.Wlab := rfl
theorem enc_eq (W : Bits (220 * (220 * 37))) (ω : Omega' 220 37) (salt : Bits 128) :
    (PousP3Meets.p3Scheme.enc W ω salt) =
      (salt, fun i => codeEquiv 220 37 (topLabel' (PousP3Meets.tagOf salt) (bandGraph 220 12) (Wlab' 220 37 W) ω i)) :=
  rfl
theorem dec_eq (pp : Bits 128) (C : Fin 220 → Bits (220 * 37)) (f : (p3Model' 220 37).Qry → Lab 220 37) :
    PousP3Meets.p3Scheme.dec pp C f =
      PousP3Meets.Wbits (decodeW' (bandGraph 220 12) (PousP3Meets.tagOf pp) (fun v => (codeEquiv 220 37).symm (C v)) f) :=
  rfl

/-! ### The toy primitive -/

def P61 : ℕ := 2305843009213693951

def mix (acc x : ℕ) : ℕ := (acc * 1000003 + x + 1) % P61

def bitsMix {L : ℕ} (acc : ℕ) (b : Bits L) : ℕ := (List.finRange L).foldl (fun a i => mix a (if b i then 1 else 0)) acc

def labMix (acc : ℕ) (L : Lab n wc) : ℕ := (List.finRange n).foldl (fun a c => bitsMix a (L c)) acc

def toySeed (q : KQ' n wc) : ℕ :=
  let a := mix (mix (bitsMix 7 q.1) q.2.1.val) q.2.2.1.val
  (List.finRange n).foldl (fun a u => match q.2.2.2 u with
    | none => mix a 2
    | some L => labMix (mix a 3) L) a

def toyH (q : KQ' n wc) : Lab n wc :=
  let s := toySeed q
  fun c j => (mix s (c.val * wc + j.val + 5) * 11400714819323198485 / 2 ^ 33) % 2 == 1

def toyConst (n wc : ℕ) : Lab n wc := fun c j => (c.val * 7 + j.val * 3) % 5 < 2

def xorPerm (k : Lab n wc) : Equiv.Perm (Lab n wc) where
  toFun a := labXor a k
  invFun a := labXor a k
  left_inv a := by funext v j; simp only [labXor, Labelling.bxor]; cases a v j <;> cases k v j <;> rfl
  right_inv a := by funext v j; simp only [labXor, Labelling.bxor]; cases a v j <;> cases k v j <;> rfl

def rotPerm (n wc : ℕ) : Equiv.Perm (Lab n wc) :=
  Equiv.arrowCongr (finRotate n) (Equiv.arrowCongr (finRotate wc) (Equiv.refl Bool))

/-- `P x = rotate (x ⊕ toyConst)`: component `v`, bit `j` of the output is bit `j - 1` of component `v - 1`. -/
def toyP (n wc : ℕ) : Equiv.Perm (Lab n wc) := (xorPerm (toyConst n wc)).trans (rotPerm n wc)

def toyΩ (n wc : ℕ) : Omega' n wc := (toyH, toyP n wc)

/-! ### Rendering -/

def bitsNat {L : ℕ} (b : Bits L) : ℕ := (List.finRange L).foldl (fun a i => if b i then a + 2 ^ i.val else a) 0

def labNat (L : Lab n wc) : ℕ := bitsNat (codeEquiv n wc L)

def natsJson (xs : List ℕ) : String := "[" ++ ", ".intercalate (xs.map fun x => s!"\"{x}\"") ++ "]"

def idxJson (xs : List ℕ) : String := "[" ++ ", ".intercalate (xs.map toString) ++ "]"

def finList (n : ℕ) (p : Fin n → Bool) : List ℕ := ((List.finRange n).filter p).map (·.val)

/-! ### The cases -/

def saltOf (seed : ℕ) : Bits 128 := fun i => (i.val * 11 + seed) % 5 < 2

def wOf (n wc seed : ℕ) : Fin n → Lab n wc := fun u c j => (u.val * 13 + c.val * 5 + j.val * 3 + seed) % 7 < 3

def wBits (n wc : ℕ) (L : Fin n → Lab n wc) : Bits (n * (n * wc)) := codeEquiv n (n * wc) fun u => codeEquiv n wc (L u)

def encodeCase (n wc k seed : ℕ) (hn : 0 < n) : String :=
  let G := bandGraph n k
  let salt := saltOf seed
  let tag := PousP3Meets.tagOf salt
  let ω := toyΩ n wc
  let W := wOf n wc seed
  let c1 := c1' tag G ω.1 ω.2 W
  let c2 := fun v => topLabel' tag G W ω v
  let x2 := transpose c1
  let dec := decodeW' G tag c2 (answer' ω)
  let lower := decodeLower ω.2 c2 (fun v => ω.1 (tag, 1, v, parentsOf' G c2 v))
  let fin := List.finRange n
  let last : Fin n := ⟨n - 1, by omega⟩
  let first : Fin n := ⟨0, hn⟩
  let parents := ", ".intercalate (fin.map fun v => idxJson (finList n fun u => decide (u ∈ G.par v)))
  let hSample := labNat (toyH (tag, 1, last, parentsOf' G c2 last))
  "{" ++ s!"\"n\": {n}, \"wc\": {wc}, \"k\": {k}, \"seed\": {seed}, " ++
    s!"\"salt\": \"{bitsNat salt}\", \"tag\": \"{bitsNat tag}\", \"w\": \"{bitsNat (wBits n wc W)}\", " ++
    s!"\"parents\": [{parents}], " ++
    s!"\"lower\": {natsJson (fin.map fun v => labNat (c1 v))}, " ++
    s!"\"transpose\": {natsJson (fin.map fun v => labNat (x2 v))}, " ++
    s!"\"top\": {natsJson (fin.map fun v => labNat (c2 v))}, " ++
    s!"\"decode_lower\": {natsJson (fin.map fun v => labNat (lower v))}, " ++
    s!"\"decode\": \"{bitsNat (wBits n wc dec)}\", \"h_sample\": \"{hSample}\", " ++
    s!"\"p_sample\": \"{labNat (toyP n wc (W first))}\", " ++
    s!"\"p_inverse_sample\": \"{labNat ((toyP n wc).symm (W first))}\"" ++ "}"

def holdingOf (n seed : ℕ) : ColState n where
  lowerWhole := Finset.univ.filter fun w : Fin n => (seed / 2 ^ w.val) % 5 = 0
  lowerCol := Finset.univ.filter fun p : Fin n × Fin n => (seed / 3 ^ (p.1.val * n + p.2.val)) % 3 = 0
  topWhole := Finset.univ.filter fun u : Fin n => (seed / 2 ^ (u.val + n)) % 3 ≠ 0

def stateJson (c : ColState n) : String :=
  let pairs := ", ".intercalate ((List.finRange n).flatMap fun w => (List.finRange n).filterMap fun u =>
    if (w, u) ∈ c.lowerCol then some s!"[{w.val}, {u.val}]" else none)
  "{" ++ s!"\"lower_whole\": {idxJson (finList n fun w => decide (w ∈ c.lowerWhole))}, " ++
    s!"\"lower_col\": [{pairs}], " ++
    s!"\"top_whole\": {idxJson (finList n fun u => decide (u ∈ c.topWhole))}, \"size\": {c.size}" ++ "}"

def gameCase (n k seed rounds : ℕ) : String :=
  let G := bandGraph n k
  let c0 := holdingOf n seed
  "{" ++ s!"\"n\": {n}, \"k\": {k}, \"seed\": {seed}, \"reach\": [" ++
    ", ".intercalate ((List.range (rounds + 1)).map fun r => stateJson (colReach G c0 r)) ++ "]}"

def encodeCases : List (ℕ × ℕ × ℕ × ℕ) := [(4, 3, 2, 1), (5, 2, 1, 2), (3, 4, 0, 3), (6, 2, 3, 4), (2, 5, 12, 5),
  (5, 3, 2, 6), (1, 3, 1, 7)]

def gameCases : List (ℕ × ℕ × ℕ × ℕ) := [(4, 1, 123456789, 5), (4, 2, 987654321, 5), (5, 2, 31415926535, 6),
  (5, 1, 27182818284, 6), (6, 3, 16180339887, 6), (3, 0, 1234567, 4)]

end PousRefCheck

open PousRefCheck in
def main : IO Unit := do
  IO.println "{\"encode\": ["
  IO.println (",\n".intercalate (encodeCases.map fun (n, wc, k, seed) =>
    if h : 0 < n then encodeCase n wc k seed h else ""))
  IO.println "], \"game\": ["
  IO.println (",\n".intercalate (gameCases.map fun (n, k, seed, r) => gameCase n k seed r))
  IO.println "]}"
