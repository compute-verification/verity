module

public import Poub.Span
public import Definitions.Core.Interaction
public import Properties.Verity.Sha512

/-!
# PoUB's read session proved in zero knowledge, as deployed: stated, unproved

The memory-read session once its openings stay inside proofs. The auditor's session decides window `w` by
`Poub.Span.checkSpan` on the span the prover sends: the sampled proof system's accepted one-stage statements, one
opening unit a sample. `checkSpan` reads three rows of each unit, `pub`, `signs` and `o0`, none in the clear: it computes
the `pub` and `signs` rows itself (`publicWords`, `signWords`) and takes the verdict word the prover sends, and holds
each to the unit's `b ‖ c` in the accepted public file by opening it under a salt the prover sends (`rowBc`: hm96-sha512
over the row's `sha512/row/v2` digest). So the statement reads those rows as committed rows opened by salt, as the
session's and the setup's statements read their carried rows: it takes a premise for the opened rows (`RowsBind`: two
rows that open to one `b ‖ c` are one row under one salt, or their SHA-512 inputs collide), and its bad event excludes a
collision among the SHA-512 inputs those openings read (`spanInputs`).

`SpanInProofsDeployed` says that a fixed extractor, rerunning any prover, recovers each sample's unit (its opening, and its
rows and salts as the proof binds them) such that `checkWindow` accepts the openings, except when those SHA-512 inputs
hold a collision, `n · (ε_ks + ε_bind)` over the span's `n` statements and the collision term `Claims.KS` carries.

It is stated and not proved. What closes it is proofs' period form of V5 with every unit proved:
`Flock.Guarantees.IntegrityAllProvedCarried` at `I` the window's statements (each proves all its units, `kd = n`, and no
leaf is carried), whose conclusion is `Verity.InProofs.Gives` at `K₀ = 1` for the run whose failing units are the game's
wrong ones. `Properties.Poub.InProofs.SecurityProofs.spanGives` (`Proofs.Properties.Compose.Remote.InProofs`)
then carries it from the opening unit's circuit to `checkWindow`, once `RowsBind` holds each unit's bound `pub` and `signs`
rows to the verifier's own.
-/

@[expose] public section

namespace Properties.Poub.SpanDeployed

open _root_.Poub _root_.Poub.Span

/-- **One opening unit's values, as its proof binds them**: the opening it checks, the words of its `pub`, `signs` and
`o0` rows, and the salts those rows are committed under. -/
structure UnitVals where
  opening : Opening
  pub : Array Nat
  signs : Array Nat
  verdict : Nat
  pubSalt : ByteArray
  signsSalt : ByteArray
  verdictSalt : ByteArray
deriving Inhabited

/-- The span's instances, its statements' in order: the window's samples, one each. -/
def instances (sp : Span) : Array Instance := sp.statements.foldl (fun acc st => acc ++ st.instances) #[]

/-- **The SHA-512 input of a row's `sha512/row/v2` digest**, the inner digest `rowBc` commits: the tag and the row's bit
count, padded to 128 bytes, then the row of `bits` bits its words hold (u16, low bit first, zero words to fill it). -/
def rowInput (bits : Nat) (words : Array Nat) : ByteArray :=
  let ws := words ++ Array.replicate ((bits + 15) / 16 - words.size) 0
  pad128 (tag0 "verity/sha512-row/v2" ++ be 8 bits) ++
    (ws.foldl (fun acc x => acc ++ le 2 x) ByteArray.empty).extract 0 ((bits + 7) / 8)

/-- The SHA-512 input of a salt's half `c` of `b ‖ c`. -/
def saltInput (salt : ByteArray) : ByteArray := Flock.Hm96.Default512.saltPrefix ++ salt

/-- **The premise for the opened rows**: two rows of `bits` bits that open to one `b ‖ c` (`rowBc`) have one SHA-512
input under one salt, or the inputs their openings read hold a collision. -/
def RowsBind : Prop :=
  ∀ (bits : Nat) (u v : Array Nat) (s t bc : ByteArray), rowBc bits u s = some bc → rowBc bits v t = some bc →
    (rowInput bits u = rowInput bits v ∧ s = t) ∨
      Hash.CollisionIn Verity.sha512 [(rowInput bits u).toList, (rowInput bits v).toList, (saltInput s).toList,
        (saltInput t).toList]

/-- **The SHA-512 inputs a span's openings read**, at the units `us`: for each sample (i, p) and each of its `pub`,
`signs` and `o0` rows (of the bits the auditor's layout gives), the unit's row and the row `checkSpan` holds it to (the
verifier's own, or the verdict word sent), and the two salts. -/
def spanInputs (a : Auditor) (reg : Registration) (w : WindowRecord) (drawn : Array (Nat × Nat)) (sp : Span)
    (us : Array UnitVals) : List Hash.Msg :=
  let bitsOf name := ((a.layout.port? name).map (·.2)).getD 0
  (List.range drawn.size).flatMap fun k =>
    let (i, p) := drawn[k]!
    let u := us[k]!
    let ins := (instances sp)[k]!
    [rowInput (bitsOf "pub") u.pub, rowInput (bitsOf "pub") (publicWords reg w i p),
      rowInput (bitsOf "signs") u.signs, rowInput (bitsOf "signs") (signWords reg w i),
      rowInput (bitsOf "o0") #[u.verdict], rowInput (bitsOf "o0") #[ins.verdict],
      saltInput u.pubSalt, saltInput ins.pubSalt, saltInput u.signsSalt, saltInput ins.signsSalt,
      saltInput u.verdictSalt, saltInput ins.verdictSalt].map ByteArray.toList

/-- **PoUB's read session proved in zero knowledge, as deployed: stated and unproved**; proofs'
`IntegrityAllProvedCarried` (the period form of V5, every unit proved) closes it through `spanGives`. For the auditor `a`,
the registration `reg`, window `w` and its samples `drawn`, a session verifier `V` that accepts only a span (`spanOf`) of
at most `n` statements that `checkSpan` accepts, and the premise `RowsBind` for the rows it opens: a fixed extractor `E`
recovers, by rerunning any prover, one unit a sample such that `checkWindow` accepts their openings, except when the
SHA-512 inputs the span's openings read at those units hold a collision, `n · (εKs + εBind)` and `Claims.KS`'s collision
term at `γ`. -/
def SpanInProofsDeployed (H : Claims.Msg → Claims.Msg) (εKs εBind γ : ℝ) : Prop :=
  ∀ (a : Auditor) (reg : Registration) (w : WindowRecord) (drawn : Array (Nat × Nat)) (n : ℕ)
    {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (V : Claims.Verifier Coin)
    (spanOf : List Claims.Msg → Option Span),
    (∀ (x : Claims.Msg) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (r : Fin (V.k x) → Coin),
      (Claims.run H P V x ω r).out = .acc →
        ∃ sp, spanOf ((Claims.run H P V x ω r).msgs.map Prod.fst) = some sp ∧ sp.statements.size ≤ n ∧
          checkSpan a reg w drawn sp = .accept) →
    RowsBind →
    ∃ E : Claims.Extractor Coin (Array UnitVals) V,
      ∀ (x : Claims.Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω),
        Claims.Pr (fun s : Ω × (Fin (V.k x) → Coin) × E.C x =>
          let t := Claims.run H P V x s.1 s.2.1
          t.out = .acc ∧ ∃ sp, spanOf (t.msgs.map Prod.fst) = some sp ∧
            match (E.eval H P x s.1 s.2.1 s.2.2).1 with
            | none => True
            | some us => checkWindow reg reg.digest #[] w drawn (us.map (·.opening)) ≠ .accept ∧
                ¬ Hash.CollisionIn Verity.sha512 (spanInputs a reg w drawn sp us)) ≤
          n * (εKs + εBind) + Real.sqrt (γ * Claims.PrB H V P x)

end Properties.Poub.SpanDeployed
