import Flock.Canon
import Flock.Hash
import Flock.Bytes

/-!
# property/v1's table (`verity/stacking-property` v1)

The closed free set and the budgeted classes, by primitive id, as `verity.ml.circuits.stacking` states them (its module
docstring is the prose): `CLASSES`, the constants (`CONST_ID`), the s8 family's core steps (`FAMILY`), the affine maps
(`AFFINE`), the gathers and their index bits (`GATHERS`), the families built on demand (`FAMILIES`), the lifts (`LIFTS`)
and the gates whose range a rule derives (`RANGES`). The table is Lean's `Accounting.Gates.classOf` id for id where that
lists the id (`verity/ml/circuits/tests/test_stacking.py` holds the Python table to it, and the difftest holds this one to
the Python by `digest`, the SHA-512 of the table as canonical data).

The patterns are kept as data, for the digest, and matched by the parsers below, which the difftest holds to Python's
`re.fullmatch` of them (`class_of`).
-/

namespace RemoteAudit.Property

open Lean (Json toJson)

inductive Cls where
  | const | family | affine | roundings | compareSelect | gathers | folds
  deriving BEq, Repr, Inhabited, Hashable

def Cls.name : Cls → String
  | .const => "const" | .family => "family" | .affine => "affine" | .roundings => "roundings"
  | .compareSelect => "compare_select" | .gathers => "gathers" | .folds => "folds"

def TAG : String := "verity/stacking-property"
def VERSION : Nat := 1
def SCHEMA : String := s!"{TAG}/v{VERSION}"

/-- The budgeted classes, in their order. -/
def BUDGETED : Array Cls := #[.roundings, .compareSelect, .gathers, .folds]
def REFUSED : Array String := #["unlisted", "reinterpretation"]

def S8 : Int × Int := (-128, 127)
def U8 : Int × Int := (0, 255)
def S16 : Int × Int := (-32768, 32767)
def S32 : Int × Int := (-2147483648, 2147483647)
def R_IN : Nat := 100000
def OPERAND_BITS : Nat := 128

/-- An s8-family core step: k products a step, a and b in their formats, D in s32. -/
structure Step where
  k : Nat
  a : Int × Int
  b : Int × Int

def FAMILY : Array (String × Step) := #[
  ("BlackwellS8ImmaDot32_v1", ⟨32, S8, S8⟩), ("BlackwellS8ImmaDot32_v2", ⟨32, S8, S8⟩), ("PouwSegDot_v1", ⟨32, S8, S8⟩),
  ("BlackwellU8S8ImmaDot32_v1", ⟨32, U8, S8⟩), ("BlackwellU8S8ImmaDot32_v2", ⟨32, U8, S8⟩),
  ("PouwDot16U8S8_v1", ⟨16, U8, S8⟩)]

def family? (fid : String) : Option Step := (FAMILY.find? (·.1 == fid)).map (·.2)

def AFFINE : Array String := #["I32Add_v1", "I32Sub_v1", "I32Mul_v1", "Bf16ToF32_v1"]

/-- `CLASSES`, sorted by id. -/
def CLASSES : Array (String × Cls) := #[
  ("AmpereBF16TcDot16_v2", .roundings),
  ("And2_v1", .compareSelect),
  ("Bf16AddF2fp_v1", .roundings),
  ("Bf16Add_v1", .roundings),
  ("Bf16GtStrict_v1", .compareSelect),
  ("Bf16MulBf16_v1", .roundings),
  ("Bf16MulF32_v1", .roundings),
  ("Bf16ToF32_v1", .affine),
  ("BitAnd_v1", .compareSelect),
  ("BitNot_v1", .compareSelect),
  ("BitOr_v1", .compareSelect),
  ("BitXor_v1", .compareSelect),
  ("BlackwellE4m3QmmaDot32_v1", .roundings),
  ("BlackwellE5m2QmmaDot32_v1", .roundings),
  ("BlackwellMxf4OmmaDot64_v1", .roundings),
  ("BlackwellNvf4OmmaDot64_v1", .roundings),
  ("BlackwellS8ImmaDot32_v1", .family),
  ("BlackwellS8ImmaDot32_v2", .family),
  ("BlackwellU8S8ImmaDot32_v1", .family),
  ("BlackwellU8S8ImmaDot32_v2", .family),
  ("DivFullRcp_v1", .roundings),
  ("DivFullScaleA_v1", .roundings),
  ("DivFullScaleA_v2", .roundings),
  ("E2m1ToBf16_v1", .roundings),
  ("E4m3ToF32_v1", .roundings),
  ("E8m0DequantBf16_v1", .roundings),
  ("E8m0ShlBf16_v1", .roundings),
  ("E8m0ToBf16_v1", .roundings),
  ("F2fpBf16_v1", .roundings),
  ("F32AddFtz_v1", .roundings),
  ("F32AddFtz_v2", .roundings),
  ("F32Add_v1", .roundings),
  ("F32Add_v2", .roundings),
  ("F32BitsShl23_v1", .roundings),
  ("F32CeilFtz_v1", .roundings),
  ("F32Div_v1", .roundings),
  ("F32Div_v2", .roundings),
  ("F32Eq_v1", .compareSelect),
  ("F32ErfRn_v1", .roundings),
  ("F32ExpRn_v1", .roundings),
  ("F32Fabs_v1", .roundings),
  ("F32FmaFtz_v1", .roundings),
  ("F32FmaFtz_v2", .roundings),
  ("F32FmaRm_v1", .roundings),
  ("F32FmaRm_v2", .roundings),
  ("F32FmaSubFtz_v1", .roundings),
  ("F32FmaSubFtz_v2", .roundings),
  ("F32Fma_v1", .roundings),
  ("F32Fma_v2", .roundings),
  ("F32Fmaxf_v1", .compareSelect),
  ("F32Fminf_v1", .compareSelect),
  ("F32Geu_v1", .compareSelect),
  ("F32GtStrict_v1", .compareSelect),
  ("F32IsFinite_v1", .compareSelect),
  ("F32MaxRef_v1", .compareSelect),
  ("F32Max_v1", .compareSelect),
  ("F32MulFtz_v1", .roundings),
  ("F32MulFtz_v2", .roundings),
  ("F32MulWide_v1", .roundings),
  ("F32Mul_v1", .roundings),
  ("F32Mul_v2", .roundings),
  ("F32Neg_v1", .roundings),
  ("F32Rint_v1", .roundings),
  ("F32Sat_v1", .compareSelect),
  ("F32Sqrt_v1", .roundings),
  ("F32SubFtz_v1", .roundings),
  ("F32SubFtz_v2", .roundings),
  ("F32Sub_v1", .roundings),
  ("F32Sub_v2", .roundings),
  ("F32TanhRn_v1", .roundings),
  ("F32ToBf16Rn_v1", .roundings),
  ("F32ToE2m1Sat_v1", .roundings),
  ("F32ToE4m3Sat_v1", .roundings),
  ("F32ToS8Rzi_v1", .roundings),
  ("F32ToU32RzByte_v1", .roundings),
  ("F64Mul_v1", .roundings),
  ("F64ToU32RniSat_v1", .roundings),
  ("Fa2InvSum_v1", .roundings),
  ("Fa3InvSum_v1", .roundings),
  ("GeluErfBf16_v1", .roundings),
  ("GeluTanhMulBf16_v1", .roundings),
  ("GuardNegInfZero_v1", .compareSelect),
  ("GumbelNoiseLane_v1", .roundings),
  ("GumbelNoiseLane_v2", .roundings),
  ("GumbelStreamKey_v1", .roundings),
  ("HopperBF16WgmmaDot16_v1", .roundings),
  ("HopperE4m3MmaV2Dot32_v1", .roundings),
  ("HopperE4m3QgmmaDot32_v1", .roundings),
  ("HopperTf32WgmmaDot8_v1", .roundings),
  ("I32Add_v1", .affine),
  ("I32Div_v1", .roundings),
  ("I32Eq_v1", .compareSelect),
  ("I32Le_v1", .compareSelect),
  ("I32Mul_v1", .roundings),
  ("I32Rem_v1", .roundings),
  ("I32Sub_v1", .affine),
  ("I32ToF32_v1", .roundings),
  ("I64ToI32_v1", .roundings),
  ("LutIndex_v1", .roundings),
  ("MufuEx2Ftz_v1", .roundings),
  ("MufuLg2Mantissa_v1", .roundings),
  ("MufuRcpFtz_v1", .roundings),
  ("MufuSqrtFtz_v1", .roundings),
  ("MufuTanh_v1", .roundings),
  ("NcpAbsMax_v1", .compareSelect),
  ("NcpDot16I7_v1", .roundings),
  ("NcpRneClampI7_v1", .roundings),
  ("Not_v1", .compareSelect),
  ("NvLogf_v1", .roundings),
  ("NvLogf_v2", .roundings),
  ("PouwAdd64Hi_v1", .roundings),
  ("PouwAdd64Lo_v1", .roundings),
  ("PouwAdd8_v1", .roundings),
  ("PouwAddSub8_v1", .roundings),
  ("PouwBf16Neg_v1", .roundings),
  ("PouwBlackwellE4m3SaltDead_v1", .compareSelect),
  ("PouwBlackwellE4m3Window_v1", .roundings),
  ("PouwCh32_v1", .roundings),
  ("PouwChi32_v1", .roundings),
  ("PouwDot16U8S8_v1", .family),
  ("PouwDuoAbsorb_v1", .folds),
  ("PouwETilde_v1", .roundings),
  ("PouwF32SqrtRn_v1", .roundings),
  ("PouwFormX3_v1", .roundings),
  ("PouwFormX_v1", .roundings),
  ("PouwHopperE4m3SaltDead_v1", .compareSelect),
  ("PouwHopperE4m3Window_v1", .roundings),
  ("PouwI32ToF32Rn_v1", .roundings),
  ("PouwMaj32_v1", .roundings),
  ("PouwNegBetaSum16_v1", .roundings),
  ("PouwNoiseByte_v1", .roundings),
  ("PouwPackBE4_v1", .roundings),
  ("PouwRotl64Hi_v1", .roundings),
  ("PouwRotl64Lo_v1", .roundings),
  ("PouwSegDot_v1", .family),
  ("PouwSel32_v1", .compareSelect),
  ("PouwSha512BigSigma0Hi_v1", .roundings),
  ("PouwSha512BigSigma0Lo_v1", .roundings),
  ("PouwSha512BigSigma1Hi_v1", .roundings),
  ("PouwSha512BigSigma1Lo_v1", .roundings),
  ("PouwSha512SmallSigma0Hi_v1", .roundings),
  ("PouwSha512SmallSigma0Lo_v1", .roundings),
  ("PouwSha512SmallSigma1Hi_v1", .roundings),
  ("PouwSha512SmallSigma1Lo_v1", .roundings),
  ("PouwSqAcc_v1", .roundings),
  ("PouwSqCeil_v1", .roundings),
  ("PouwSqGe_v1", .compareSelect),
  ("PouwSub8_v1", .roundings),
  ("PouwWinOk_v1", .compareSelect),
  ("PouwWindowWord_v1", .roundings),
  ("PouwXor32_v1", .roundings),
  ("RopeOutAdd_v1", .roundings),
  ("RopeOut_v1", .roundings),
  ("RsqrtApprox_v1", .roundings),
  ("S32ByteCarry_v1", .roundings),
  ("S32ByteCarry_v2", .roundings),
  ("S32ClampRest_v1", .roundings),
  ("S32ClampRest_v2", .roundings),
  ("S32ClampS8_v1", .compareSelect),
  ("S32ClampS8_v2", .compareSelect),
  ("S32LowByte_v1", .roundings),
  ("S32LowByte_v2", .roundings),
  ("S32ToF32Rn_v1", .roundings),
  ("S8ToF32_v1", .roundings),
  ("SelectBf16_v1", .compareSelect),
  ("SelectBit_v1", .compareSelect),
  ("SelectF32_v1", .compareSelect),
  ("SelectI32_v1", .compareSelect),
  ("SiluMulBf16_v1", .roundings),
  ("SiluMulBf16_v2", .roundings),
  ("TanhF32Rn_v1", .roundings),
  ("TopPNumkeep_v1", .roundings),
  ("U32And_v1", .roundings),
  ("U32Ge_v1", .compareSelect),
  ("U32Xor_v1", .roundings),
  ("Ue8m0CeilByte_v1", .roundings),
  ("Ue8m0InvScale_v1", .roundings),
  ("Xor2_v1", .compareSelect)
]

def CONST_PATTERN : String := "^Const([1-9][0-9]*)\\[0x([0-9a-f]+)\\]_v1$"

inductive GKind where
  | range | low | lifted
  deriving BEq, Inhabited

def GKind.name : GKind → String
  | .range => "range" | .low => "low" | .lifted => "lifted"

/-- `GATHERS`: pattern, index operand, kind, fixed n. -/
def GATHERS : Array (String × Nat × GKind × Option Nat) := #[
  ("^Gather(?:Bf16|F32|I32|E4m3|E2m1|E8m0)x([1-9][0-9]*)_v1$", 0, .range, none),
  ("^TableRead([1-9][0-9]*)x[1-9][0-9]*_v1$", 0, .low, none),
  ("^LGatherRow[1-9][0-9]*x([1-9][0-9]*)(?:o[0-9a-f]+)?_v1$", 0, .lifted, none),
  ("^Exp2Lut127_v1$", 0, .low, some 8)]

def FAMILIES : Array (String × Cls) := #[
  ("^BitAtx[1-9][0-9]*_v1$", .roundings),
  ("^(?:And|Xor)(?:[2-9]|[1-9][0-9]+)_v1$", .compareSelect),
  ("^TopPMaskWordx[1-9][0-9]*_v1$", .compareSelect),
  ("^L(?:Select[1-9][0-9]*|FixedSelect[1-9][0-9]*x[1-9][0-9]*|IsActive[1-9][0-9]*|IsEOS[1-9][0-9]*|StepLive[1-9][0-9]*|Enable|EmitNext\\[H=[1-9][0-9]*\\])_v1$", .compareSelect),
  ("^L?SplitsForSMS[1-9][0-9]*_v1$", .roundings),
  ("^LiveCount[1-9][0-9]*_v1$", .roundings),
  ("^LBot[1-9][0-9]*_v1$", .const)]

def LIFTS : Array String := #["^Lifted\\[(.+)\\]_v[12]$", "^(.+)\\{ORD=[0-9]+\\}$"]

/-- `_LIFTED`: a lift's class when the lifted primitive is free. -/
def lifted (c : Cls) : Cls :=
  match c with
  | .const => .compareSelect
  | .affine | .family => .roundings
  | c => c

def RANGES : Array String := #["Exp2Lut127_v1", "F32Fmaxf_v1", "F32Fminf_v1", "F32ToU32RzByte_v1"]

/-! ### The patterns' parsers, over characters -/

private def isDigit (c : Char) : Bool := '0' ≤ c && c ≤ '9'
private def isHex (c : Char) : Bool := isDigit c || ('a' ≤ c && c ≤ 'f')

/-- `[1-9][0-9]*`, greedy: the digits and the rest. -/
private def nat19 (cs : List Char) : Option (List Char × List Char) :=
  match cs with
  | c :: _ => if '1' ≤ c && c ≤ '9' then some (cs.takeWhile isDigit, cs.dropWhile isDigit) else none
  | [] => none

private def lit (s : String) (cs : List Char) : Option (List Char) :=
  if s.toList.isPrefixOf cs then some (cs.drop s.length) else none

private def digitsNat (ds : List Char) : Nat := ds.foldl (fun n c => 10 * n + (c.toNat - '0'.toNat)) 0

/-- `[1-9][0-9]*` then exactly `tail`. -/
private def numThen (tail : String) (cs : List Char) : Option Nat := do
  let (ds, rest) ← nat19 cs
  if rest == tail.toList then some (digitsNat ds) else none

private def matchesNumThen (tail : String) (cs : List Char) : Bool := (numThen tail cs).isSome

/-- A constant's width and value: `Const<w>[0x<hex>]_v1`. -/
def constOf? (fid : String) : Option (Nat × Nat) := do
  let cs ← lit "Const" fid.toList
  let (ds, rest) ← nat19 cs
  let rest ← lit "[0x" rest
  let hex := rest.takeWhile isHex
  if hex.isEmpty || rest.dropWhile isHex != "]_v1".toList then none
  let v := hex.foldl (fun n c => 16 * n + (if isDigit c then c.toNat - '0'.toNat else c.toNat - 'a'.toNat + 10)) 0
  some (digitsNat ds, v)

/-- `_unlift`: the primitive a lift wraps, innermost, and whether `fid` is one. Each unwrapping drops at least four
characters, so `fid.length + 1` is fuel enough. -/
def unlift (fid : String) (lifted : Bool := false) (fuel : Nat := fid.length + 1) : String × Bool :=
  match fuel with
  | 0 => (fid, lifted)
  | fuel + 1 =>
  let cs := fid.toList
  let inner? : Option (List Char) :=
    match lit "Lifted[" cs with
    | some rest =>
      if rest.length ≥ 5 && (rest.drop (rest.length - 4) == "]_v1".toList || rest.drop (rest.length - 4) == "]_v2".toList)
      then some (rest.take (rest.length - 4)) else none
    | none => none
  let inner? := inner?.orElse fun _ =>
    -- `(.+)\{ORD=[0-9]+\}`: the last `{ORD=` followed by digits and the closing brace
    if cs.getLast? != some '}' then none else
    let body := cs.dropLast
    let ds := (body.reverse.takeWhile isDigit).reverse
    let pre := body.take (body.length - ds.length)
    if ds.isEmpty || pre.length < 6 || pre.drop (pre.length - 5) != "{ORD=".toList then none
    else some (pre.take (pre.length - 5))
  match inner? with
  | some inner => if inner.isEmpty then (fid, lifted) else unlift (String.ofList inner) true fuel
  | none => (fid, lifted)

private def gatherBase? (cs : List Char) : Option (Nat × GKind) :=
  let types := ["Bf16", "F32", "I32", "E4m3", "E2m1", "E8m0"]
  match lit "Gather" cs with
  | some rest => types.findSome? fun t => do
      let r ← lit t rest
      let r ← lit "x" r
      let n ← numThen "_v1" r
      some (n, .range)
  | none =>
  match lit "TableRead" cs with
  | some rest => do
      let (ds, r) ← nat19 rest
      let r ← lit "x" r
      if matchesNumThen "_v1" r then some (digitsNat ds, .low) else none
  | none =>
  match lit "LGatherRow" cs with
  | some rest => do
      let (_, r) ← nat19 rest
      let r ← lit "x" r
      let (ds, r) ← nat19 r
      if r == "_v1".toList then some (digitsNat ds, .lifted)
      else
        let r ← lit "o" r
        let hex := r.takeWhile isHex
        if !hex.isEmpty && r.dropWhile isHex == "_v1".toList then some (digitsNat ds, .lifted) else none
  | none => if cs == "Exp2Lut127_v1".toList then some (8, .low) else none

/-- A gather's index bits for an index of n of its kind. -/
def GKind.bits (k : GKind) (n : Nat) : Nat :=
  match k with
  | .range => Nat.log2 n + (if n == 0 then 0 else 1)
  | .low => n
  | .lifted => Nat.log2 (n + 2) + 1

/-- `gather`: a gather's index operand, its index bits and kind, and whether it is lifted. -/
def gather? (fid : String) : Option (Nat × Nat × GKind × Bool) := do
  let (inner, lifted) := unlift fid
  let (n, kind) ← gatherBase? inner.toList
  some (0, kind.bits n, kind, lifted || kind == .lifted)

/-- `_column`: an unlifted gather, whose operand bits are the column its index bits pay for. -/
def column (fid : String) : Bool := match gather? fid with
  | some (_, _, _, l) => !l
  | none => false

private def familyOf? (cs : List Char) : Option Cls :=
  let anyNum := fun (pre : String) => match lit pre cs with
    | some r => matchesNumThen "_v1" r
    | none => false
  if anyNum "BitAtx" then some .roundings
  else if (match (lit "And" cs).orElse (fun _ => lit "Xor" cs) with
      | some r => match r with
        | c :: rest => let ds := (c :: rest).takeWhile isDigit
                       c != '0' && isDigit c && (ds.length ≥ 2 || c ≥ '2') && (c :: rest).dropWhile isDigit == "_v1".toList
        | [] => false
      | none => false) then some .compareSelect
  else if anyNum "TopPMaskWordx" then some .compareSelect
  else if (match lit "L" cs with
      | some r =>
        let n := fun (pre : String) => match lit pre r with
          | some q => matchesNumThen "_v1" q
          | none => false
        n "Select" || n "IsActive" || n "IsEOS" || n "StepLive" || r == "Enable_v1".toList
        || (match lit "FixedSelect" r with
            | some q => match nat19 q with
              | some (_, q) => match lit "x" q with
                | some q => matchesNumThen "_v1" q
                | none => false
              | none => false
            | none => false)
        || (match lit "EmitNext[H=" r with
            | some q => matchesNumThen "]_v1" q
            | none => false)
      | none => false) then some .compareSelect
  else if anyNum "SplitsForSMS" || anyNum "LSplitsForSMS" then some .roundings
  else if anyNum "LiveCount" then some .roundings
  else if anyNum "LBot" then some .const
  else none

/-- `class_of`: `CLASSES`, else a lift's (the lifted primitive's class, a free one's budgeted), else ``const``, ``gathers``
or a `FAMILIES` class, else none (unlisted, refused). -/
def classOf (fid : String) : Option Cls :=
  match CLASSES.find? (·.1 == fid) with
  | some (_, c) => some c
  | none =>
    let (inner, isLift) := unlift fid
    let base := fun (f : String) =>
      match CLASSES.find? (·.1 == f) with
      | some (_, c) => some c
      | none =>
        if (constOf? f).isSome then some .const
        else if (gatherBase? f.toList).isSome then some .gathers
        else familyOf? f.toList
    if isLift then (base inner).map lifted else base fid

/-! ### The table as data, and its digest -/

private def pairJ (r : Int × Int) : Json := .arr #[toJson r.1, toJson r.2]

/-- `spec()`: the table as canonical data. -/
def spec : Json :=
  let sorted := CLASSES.qsort (fun a b => a.1 < b.1)
  Json.mkObj [
    ("format", .str TAG), ("version", toJson VERSION),
    ("budgeted", .arr (BUDGETED.map (.str ·.name))), ("refused", .arr (REFUSED.map .str)),
    ("classes", Json.mkObj (sorted.toList.map fun (k, c) => (k, .str c.name))),
    ("const", .str CONST_PATTERN), ("affine", .arr (AFFINE.map .str)),
    ("gathers", .arr (GATHERS.map fun (p, i, k, n) => .arr #[.str p, toJson i, .str k.name, match n with
      | some n => toJson n
      | none => .null])),
    ("families", .arr (FAMILIES.map fun (p, c) => .arr #[.str p, .str c.name])),
    ("lifts", .arr (LIFTS.map .str)),
    ("lifted", Json.mkObj [("affine", .str "roundings"), ("const", .str "compare_select"), ("family", .str "roundings")]),
    ("ranges", .arr ((RANGES.qsort (· < ·)).map .str)),
    ("family", Json.mkObj ((FAMILY.qsort (fun a b => a.1 < b.1)).toList.map fun (f, s) =>
      (f, Json.mkObj [("k", toJson s.k), ("a", pairJ s.a), ("b", pairJ s.b), ("acc", pairJ S32)])))]

/-- `digest()`: `SHA-512(TAG || "/v" || version || 0 || canon(spec))`, hex. -/
def digest : String :=
  match Flock.canon spec with
  | .ok s => Flock.hexOf (Flock.Sha512.hash ((s!"{TAG}/v{VERSION}".toUTF8.push 0) ++ s.toUTF8))
  | .error e => s!"unreachable: {e}"

end RemoteAudit.Property
