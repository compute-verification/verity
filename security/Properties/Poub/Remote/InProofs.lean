module

public import Poub.Span
public import Definitions.Core.Boolean
public import Definitions.Core.Interaction
public import Properties.Poub.Remote
public import Properties.Verity.Sha512

/-! PoUB's read session proved in zero knowledge, as deployed (`SpanInProofsDeployed`): stated and not proved. -/

@[expose] public section

namespace Properties.Poub.SpanDeployed

open _root_.Poub _root_.Poub.Span

/-- **One opening unit's values, as its proof binds them**: the opening it checks, the words of its `pub` and `signs`
rows (the supplied rows R7 holds to the verifier's), its `o0` verdict word and the salt `o0` is committed under. -/
structure UnitVals where
  opening : Opening
  pub : Array Nat
  signs : Array Nat
  verdict : Nat
  verdictSalt : ByteArray
deriving Inhabited

def instances (sp : Span) : Array Instance := sp.statements.foldl (fun acc st => acc ++ st.instances) #[]

/-- **The SHA-512 input of a row's `sha512/row/v2` digest**, the inner digest `rowBc` commits: the tag and the row's bit
count, padded to 128 bytes, then the row of `bits` bits its words hold (u16, low bit first, zero words to fill it). -/
def rowInput (bits : Nat) (words : Array Nat) : ByteArray :=
  let ws := words ++ Array.replicate ((bits + 15) / 16 - words.size) 0
  pad128 (tag0 "verity/sha512-row/v2" ++ be 8 bits) ++
    (ws.foldl (fun acc x => acc ++ le 2 x) ByteArray.empty).extract 0 ((bits + 7) / 8)

def saltInput (salt : ByteArray) : ByteArray := Flock.Hm96.Default512.saltPrefix ++ salt

/-- **The premise for the opened rows**: two rows of `bits` bits that open to one `b ‖ c` (`rowBc`) have one SHA-512
input under one salt, or the inputs their openings read hold a collision. -/
def RowsBind : Prop :=
  ∀ (bits : Nat) (u v : Array Nat) (s t bc : ByteArray), rowBc bits u s = some bc → rowBc bits v t = some bc →
    (rowInput bits u = rowInput bits v ∧ s = t) ∨
      Hash.CollisionIn Verity.sha512 [(rowInput bits u).toList, (rowInput bits v).toList, (saltInput s).toList,
        (saltInput t).toList]

/-- **The SHA-512 inputs a span's `o0` openings read**, at the units `us`: for each sample, the `o0` row's digest input
(of the bits the auditor's layout gives) at the unit's verdict word and at the word the prover sends, and the two salts'. -/
def spanInputs (a : Auditor) (sp : Span) (us : Array UnitVals) : List Hash.Msg :=
  let bits := ((a.layout.port? "o0").map (·.2)).getD 0
  (List.range us.size).flatMap fun k =>
    let u := us[k]!
    let ins := (instances sp)[k]!
    [rowInput bits #[u.verdict], rowInput bits #[ins.verdict], saltInput u.verdictSalt,
      saltInput ins.verdictSalt].map ByteArray.toList

def wordBits (words : Array Nat) : List Bool := words.toList.flatMap fun x => (List.range 16).map x.testBit

def bitsWord (bs : List Bool) : Nat := bs.foldr (fun b acc => 2 * acc + if b then 1 else 0) 0

/-- **Named assumption: the registered opening circuit computes `checkOpening`**: an assignment of `read` carrying
the verifier's `pub` and `signs` rows for sample (i, p) whose `o0` word passes decodes to an opening
`checkOpening` passes at (i, p). This tree names no reading `read`, so it holds of one that registers nothing. -/
def OpeningComputesCheck (a : Auditor) (reg : Registration) (w : WindowRecord) (sampled : Array (Nat × Nat))
    (read : Layout → Option ((C : Verity.Boolean.Circuit) × (String → List (Fin C.N)) × List (Fin C.N)))
    (decode : (String → List Bool) → Opening) : Prop :=
  ∀ r ∈ read a.layout, ∀ (k : Nat) (hk : k < sampled.size) (x : Fin r.1.N → Bool),
    (∀ j (h : j < (r.2.1 "pub").length),
      x ((r.2.1 "pub").get ⟨j, h⟩) = (wordBits (publicWords reg w sampled[k].1 sampled[k].2)).getD j false) →
    (∀ j (h : j < (r.2.1 "signs").length),
      x ((r.2.1 "signs").get ⟨j, h⟩) = (wordBits (signWords reg w sampled[k].1)).getD j false) →
    verdictFailure (bitsWord (r.2.2.map (r.1.eval x))) = none →
    checkOpening reg reg.digest #[] w (Properties.Poub.receipterOf reg w sampled[k].1) sampled[k].1 sampled[k].2
      (decode fun name => (r.2.1 name).map (r.1.eval x)) = none

/-- **PoUB's read session proved in zero knowledge, as deployed: stated and unproved.** Under `RowsBind` and
`OpeningComputesCheck`, a fixed extractor recovers one unit a sample whose openings `checkWindow` accepts, except
when the `o0` openings' SHA-512 inputs collide, at `n · (εKs + εBind)` and `Claims.KS`'s collision term at `γ`. -/
def SpanInProofsDeployed (H : Claims.Msg → Claims.Msg) (εKs εBind γ : ℝ) : Prop :=
  ∀ (a : Auditor) (reg : Registration) (w : WindowRecord) (sampled : Array (Nat × Nat)) (n : ℕ)
    {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (V : Claims.Verifier Coin)
    (spanOf : List Claims.Msg → Option Span)
    (read : Layout → Option ((C : Verity.Boolean.Circuit) × (String → List (Fin C.N)) × List (Fin C.N)))
    (decode : (String → List Bool) → Opening),
    (∀ (x : Claims.Msg) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (r : Fin (V.k x) → Coin),
      (Claims.run H P V x ω r).out = .acc →
        ∃ sp, spanOf ((Claims.run H P V x ω r).msgs.map Prod.fst) = some sp ∧ sp.statements.size ≤ n ∧
          checkSpan a reg w sampled sp = .accept) →
    RowsBind →
    OpeningComputesCheck a reg w sampled read decode →
    ∃ E : Claims.Extractor Coin (Array UnitVals) V,
      ∀ (x : Claims.Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω),
        Claims.Pr (fun s : Ω × (Fin (V.k x) → Coin) × E.C x =>
          let t := Claims.run H P V x s.1 s.2.1
          t.out = .acc ∧ ∃ sp, spanOf (t.msgs.map Prod.fst) = some sp ∧
            match (E.eval H P x s.1 s.2.1 s.2.2).1 with
            | none => True
            | some us => checkWindow reg reg.digest #[] w sampled (us.map (·.opening)) ≠ .accept ∧
                ¬ Hash.CollisionIn Verity.sha512 (spanInputs a sp us)) ≤
          n * (εKs + εBind) + Real.sqrt (γ * Claims.PrB H V P x)

end Properties.Poub.SpanDeployed
