import VStar.Unit

/-!
# `InnerRepCheck{S}`: one part of a rep's inner algebraic checks (`verity_flock.rec_residuals`)

`GfResiduals{S.gf}` on the row ports' first elements (`w`, in port order) and `v`'s first `S.gf.nv`, its residuals as u16
words, then zero words up to 128. Each row port of `n` elements is `P(n) = 64 ⌈8 n / 64⌉` words (whole 64-word M0 rows),
`v` is `P(nv)`.
-/

namespace VStar

open Flock.HmNets

/-- `InnerRepCheck`'s S: `GfResiduals`' S and the row ports, `[name, elements]`. -/
structure RepS where
  gf : GfS
  ports : List (String × Nat)
  deriving Inhabited

def RES_OUT_WORDS : Nat := 128

/-- `rec_residuals.port_words`: a port of `n` elements, in whole 64-word rows. -/
def portWordsOf (n : Nat) : Nat := (8 * n + 63) / 64 * 64

/-- `rec_residuals._check(S, whole)`: one statement's S (`whole` false) also holds at most `RES_OUT_WORDS / 8` residuals,
which a whole algebra's S (`whole`, `parts`' input) need not. -/
def checkRepS (S : RepS) (whole : Bool := false) : Except String Unit := do
  let names := S.ports.map (·.1)
  if names.eraseDups.length != names.length || names.contains "v" || S.ports.any (·.2 < 1) then
    throw "InnerRepCheck's row ports are distinct names, not v, each of at least one element"
  let n := (S.ports.map (·.2)).foldl (· + ·) 0
  if n != S.gf.nw then throw s!"row ports of {n} elements for a w of {S.gf.nw}"
  if !whole && 8 * S.gf.res.length > RES_OUT_WORDS then
    throw s!"{S.gf.res.length} residuals: more than {RES_OUT_WORDS / 8}"

/-- `rec_residuals.lowering(S, name)`'s unit: `rec_residuals.unit_circuit`. -/
def innerUnit (S : RepS) (name : String) : Except String UnitC := do
  checkRepS S
  checkS S.gf
  let pw := S.ports.map (fun p => portWordsOf p.2) ++ [portWordsOf S.gf.nv]
  let build : M (List Lin) := do
    let flat ← input (16 * pw.foldl (· + ·) 0)
    let parts := slices flat pw
    let w := (S.ports.zip parts).flatMap fun (p, bits) => bits.take (128 * p.2)
    let v := ((parts.drop S.ports.length).headD []).take (128 * S.gf.nv)
    let bits ← residualForms S.gf w.toArray v.toArray
    pure (bits ++ List.replicate (16 * RES_OUT_WORDS - bits.length) Lin.zero)
  let r := build.run St.empty
  pure { name, inPorts := pw, outPorts := [RES_OUT_WORDS], st := r.2, outs := r.1.toArray }

end VStar
