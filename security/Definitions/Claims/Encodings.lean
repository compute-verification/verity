module

public import Definitions.Claims.SamplingGeneric

/-!
# Claims: registered encodings, the weights' commitment a proof opens by linear claims

The registered weights stay hm96 rows (the deployed `Claims.Sampling.CommitmentScheme`), the one commitment of record. A
link at registration ties them to an encoding of the same values, a code whose columns are committed under a root, which
a proof block's session opens by linear claims, so no proof block hashes its weight rows in-circuit. The sampled audit's
statement, witness, relation and claims (`Claims.Sampling`, `Claims.SamplingGeneric`) are read as they stand.

1. **The sum by position** (`byPos`): one `CommitmentScheme`, the encoding's at the positions the statement's roots
   encode and the record's elsewhere; no new opening kind. It binds when its parts do (`ByPosBinds`).
2. **The encoding** (`CodeScheme`): its opening, as the extractor checks it (`CodeScheme.scheme`), is a message and at
   least `agree` of its columns, at distinct positions below `len`, each passing its check under the roots (`colOK`,
   the column's tree path), with the message's encoding equal to the opened column at each of them and the opened value
   the message's bit at the position. It binds when its distance exceeds twice the columns an opening may leave out
   (`Distance`) and the column check binds (`ColBinds`): `CodeBinds`, decoding within the unique radius, a collision
   reduction with no assumption.
3. **The link** (`RegLink`): a proof system for proof blocks (`Arg`) run once at registration on its statement `x`, whose
   witness reveals, at each audit statement it names, pairs of a record opening and an encoding opening (`reveal`). Its
   contract (`RegLink.Opens`): a witness in its relation reveals, at every encoded position, a pair whose openings pass
   and agree (`Linked`, `linkedAt`). Then an encoding opening and a record opening at a linked position agree, or the
   checks collide (`LinkBinds`); and a trace in the sampling relation under the sum is in it under the record once its
   encoded openings are replaced by the link's record openings (`relink`), or the checks collide (`RelinkR`).
4. **What the composed statements read**, on a compiled sampled verifier whose commitment is the sum and a link that
   names its statements, with the audit's prover reading the link's run: for knowledge soundness at the record, the
   linked bad event (`PrBadLinked`), the link's error (`RegLink.ε`, a term of the registration's one run, which every
   audit naming it shares) and the binding term `B_link` (`Blink`); for the committed form, the committed trace's
   incorrect set (`FregL` at `tableL`: the link's record value at an encoded position and `Compiled.regVal` elsewhere,
   fixed by the registration and the prover's tape before the window's coins) and its error, `ε_reg`
   (`Compiled.εreg`), the link's error and `B_cross` (`Bcross`: the audit's extraction and the link's revealing one
   position two ways) (`εregLinked`). The two statements (`CompilerKSLinked`, `CompilerKSRegLinked`) are closed, over
   every proof system, compiled verifier and link, and unproved, so they are stated in `Proofs.Flock.Encodings`, not as
   claims.
5. **Assumptions**: none new. The encoding binds by a collision reduction at a distance its instance proves, the link's
   error is its proof system's knowledge error (`Arg.KS`), and `ExpectedCR` reads `B_link` and `B_cross` as it reads
   `B_ext` and `B_conf`.

Statements only: definitions, no proofs. Zero knowledge with a registered encoding's columns opened across sessions is
not stated here.
-/

@[expose] public section

namespace Claims.Encodings

open Claims
open Claims.Sampling (AuditStatement Partition CommitmentScheme Opening LogColl ownLog)
open Claims.SamplingGeneric (Arg Compiled)
open Claims.Recursion (allOC)

/-! ## 1. The sum by position -/

/-- **The sum of two commitment schemes by position**: at a position the roots encode (`enc roots pos`), the encoding's
opening check (`Ce`), and elsewhere the record's (`Cm`); its published layer is the record's leaves of the openings it
checks and the encoding's of the rest. -/
def byPos {α : Type} (enc : List Msg → ℕ → Bool) (Ce Cm : CommitmentScheme α) : CommitmentScheme α where
  opens roots o := if enc roots o.pos then Ce.opens roots o else Cm.opens roots o
  layer H roots os :=
    Cm.layer H roots (os.filter fun o => !enc roots o.pos) ++ Ce.layer H roots (os.filter fun o => enc roots o.pos)

/-- **The sum binds when its parts do.** -/
def ByPosBinds : Prop :=
  ∀ (α : Type) (H : Msg → Msg) (enc : List Msg → ℕ → Bool) (Ce Cm : CommitmentScheme α),
    Ce.Binds H → Cm.Binds H → (byPos enc Ce Cm).Binds H

/-! ## 2. The encoding -/

/-- **An encoding of committed values**, as the extractor opens it: under `roots`, codewords of `len` columns
(`encode`, of a message `M`, its pad among it at a masked code), the value a message holds at a position (`bit`), the
opening's data (`parse`: a message and opened columns, each its position, its column and its proof) and one column's
check under the roots (`colOK`: its tree path to the encoding's root), and the columns an opening must agree on
(`agree`). -/
structure CodeScheme (M Col Pf : Type) where
  len : List Msg → ℕ
  agree : List Msg → ℕ
  encode : List Msg → M → ℕ → Col
  bit : List Msg → M → ℕ → Option Bool
  parse : Msg → Option (M × List (ℕ × Col × Pf))
  colOK : List Msg → ℕ → Col → Pf → OC Bool

namespace CodeScheme

variable {M Col Pf : Type} [DecidableEq Col]

/-- Every opened column passes its check, in order, up to the first that fails. -/
def colsOK (C : CodeScheme M Col Pf) (roots : List Msg) (cs : List (ℕ × Col × Pf)) : OC Bool :=
  allOC (cs.map fun q => C.colOK roots q.1 q.2.1 q.2.2)

/-- The opened columns `cs` fit the message `m` and the opening `o`: at distinct positions below `len`, at least
`agree` of them equal to `m`'s encoding, and `o`'s value `m`'s bit at `o`'s position. -/
def fits (C : CodeScheme M Col Pf) (roots : List Msg) (m : M) (cs : List (ℕ × Col × Pf)) (o : Opening Bool) : Bool :=
  decide (cs.map (·.1)).Nodup && cs.all (fun q => decide (q.1 < C.len roots)) &&
    decide (C.agree roots ≤ (cs.filter fun q => decide (C.encode roots m q.1 = q.2.1)).length) &&
    decide (C.bit roots m o.pos = some o.val)

/-- **The encoding as a commitment scheme**: an opening passes when its data parses, each opened column passes its check,
and the columns fit the message and the opening. It publishes no leaves of its own: the roots are its public part. -/
def scheme (C : CodeScheme M Col Pf) : CommitmentScheme Bool where
  opens roots o :=
    match C.parse o.data with
    | none => .ret false
    | some (m, cs) => OC.bind (C.colsOK roots cs) fun ok => .ret (ok && C.fits roots m cs o)
  layer _ _ _ := []

/-- **The code's distance**, at least `d`: two messages whose codewords differ at fewer than `d` of the `len` columns hold
the same values. -/
def Distance (C : CodeScheme M Col Pf) (d : List Msg → ℕ) : Prop :=
  ∀ (roots : List Msg) (m m' : M),
    ((Finset.range (C.len roots)).filter fun i => C.encode roots m i ≠ C.encode roots m' i).card < d roots →
      ∀ g, C.bit roots m g = C.bit roots m' g

/-- **The column check binds, as a collision reduction**: two different columns that pass at one position under the
same roots make `H` calls that collide. -/
def ColBinds (H : Msg → Msg) (C : CodeScheme M Col Pf) : Prop :=
  ∀ (roots : List Msg) (i : ℕ) (c c' : Col) (pf pf' : Pf), c ≠ c' →
    ((C.colOK roots i c pf).eval H).1 = true → ((C.colOK roots i c' pf').eval H).1 = true →
    LogColl (((C.colOK roots i c pf).eval H).2 ++ ((C.colOK roots i c' pf').eval H).2)

end CodeScheme

/-- **The encoding binds**: within the unique decoding radius, `2·(len − agree) < d`, at a distance `d` and a binding
column check, two openings at one position of different values make `H` calls that collide. -/
def CodeBinds : Prop :=
  ∀ (M Col Pf : Type) [DecidableEq Col] (H : Msg → Msg) (C : CodeScheme M Col Pf) (d : List Msg → ℕ),
    C.Distance d → (∀ roots, 2 * (C.len roots - C.agree roots) < d roots) → C.ColBinds H → C.scheme.Binds H

/-! ## 3. The link at registration -/

/-- **The record's value the link ties at `g`**: the value of the first pair of `ℓ` at `g` whose record opening (`Cm`)
and encoding opening (`Ce`) pass under `roots` and agree; `none` where no pair does. -/
def linkedAt (H : Msg → Msg) (Cm Ce : CommitmentScheme Bool) (roots : List Msg)
    (ℓ : List (Opening Bool × Opening Bool)) (g : ℕ) : Option Bool :=
  (ℓ.find? fun p => p.1.pos == g && p.2.pos == g && p.1.val == p.2.val && Cm.Opens H roots p.1 &&
    Ce.Opens H roots p.2).map (·.1.val)

/-- **The pairs `ℓ` link the positions `ps`**: at each, a pair whose openings pass and agree. -/
def Linked (H : Msg → Msg) (Cm Ce : CommitmentScheme Bool) (roots : List Msg) (ps : ℕ → Prop)
    (ℓ : List (Opening Bool × Opening Bool)) : Prop :=
  ∀ g, ps g → linkedAt H Cm Ce roots ℓ g ≠ none

/-- The `H` calls of checking the pairs `ℓ`, each opening under its scheme, in order. -/
def linkLog (H : Msg → Msg) (Cm Ce : CommitmentScheme Bool) (roots : List Msg)
    (ℓ : List (Opening Bool × Opening Bool)) : List (Msg × Msg) :=
  ℓ.flatMap fun p => ((Cm.opens roots p.1).eval H).2 ++ ((Ce.opens roots p.2).eval H).2

/-- **The link binds the record to the encoding**: at a position the pairs `ℓ` link, a record opening and an encoding
opening that pass, of different values, make `H` calls that collide, among theirs and the pairs' checks. -/
def LinkBinds : Prop :=
  ∀ (H : Msg → Msg) (Cm Ce : CommitmentScheme Bool) (roots : List Msg) (ℓ : List (Opening Bool × Opening Bool))
    (a e : Opening Bool), Cm.Binds H → Ce.Binds H → a.pos = e.pos → linkedAt H Cm Ce roots ℓ a.pos ≠ none →
    Cm.Opens H roots a = true → Ce.Opens H roots e = true → a.val ≠ e.val →
    LogColl (((Cm.opens roots a).eval H).2 ++ ((Ce.opens roots e).eval H).2 ++ linkLog H Cm Ce roots ℓ)

/-- **A trace read at the record**: `w`'s openings at positions the roots don't encode, and the record openings of the
pairs `ℓ` at those they do. -/
def relink (enc : List Msg → ℕ → Bool) (roots : List Msg) (w : Sampling.Wit) (ℓ : List (Opening Bool × Opening Bool)) :
    Sampling.Wit where
  vals := w.vals.filter (fun o => !enc roots o.pos) ++ (ℓ.map Prod.fst).filter fun o => enc roots o.pos

/-- **The relation carries over the link**: a trace in the sampling relation at `δ` under the sum, relinked by pairs that
link every encoded position, is in it under the record, or the checks of its openings under the sum and of the pairs
collide. -/
def RelinkR : Prop :=
  ∀ (H : Msg → Msg) (enc : List Msg → ℕ → Bool) (Ce Cm : CommitmentScheme Bool) (δ : ℝ) (s : AuditStatement)
    (P : Partition) (w : Sampling.Wit) (ℓ : List (Opening Bool × Opening Bool)),
    Ce.Binds H → Cm.Binds H → Linked H Cm Ce s.roots (fun g => enc s.roots g = true) ℓ →
    Sampling.RelS H (byPos enc Ce Cm) δ s P w →
    Sampling.RelS H Cm δ s P (relink enc s.roots w ℓ) ∨
      LogColl ((w.vals.flatMap fun o => (((byPos enc Ce Cm).opens s.roots o).eval H).2) ++
        linkLog H Cm Ce s.roots ℓ)

/-- **A registration's link**: a proof system for proof blocks (`A`), run once on the registration's statement `x`, the
audit statements it names (`names`), and the pairs of a record opening and an encoding opening its witness reveals at
each (`reveal`), at the statement's positions and under its roots. -/
structure RegLink (Coin : Type) where
  A : Arg Coin
  x : Msg
  names : AuditStatement → Prop
  reveal : AuditStatement → A.W → List (Opening Bool × Opening Bool)

/-- **The link's contract**: a witness in its relation reveals, at each audit statement it names, pairs that link every
position the statement's roots encode. -/
def RegLink.Opens {Coin : Type} (H : Msg → Msg) (Lk : RegLink Coin) (enc : List Msg → ℕ → Bool)
    (Cm Ce : CommitmentScheme Bool) : Prop :=
  ∀ w, Lk.A.R Lk.x w → ∀ s, Lk.names s → Linked H Cm Ce s.roots (fun g => enc s.roots g = true) (Lk.reveal s w)

/-! ## 4. What the composed statements read -/

section Composed

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **The pairs the link's extraction reveals** at `s`, at the link's run `(ω, r)` and its extractor's coins `c`; none
when it gives up. -/
noncomputable def RegLink.pairs (H : Msg → Msg) (Lk : RegLink Coin) {Ω : Type} (Pℓ : Prover Ω) (s : AuditStatement)
    (ω : Ω) (r : Fin (Lk.A.V.k Lk.x) → Coin) (c : Lk.A.E.C Lk.x) : List (Opening Bool × Opening Bool) :=
  match (Lk.A.E.eval H Pℓ Lk.x ω r c).1 with
  | some w => Lk.reveal s w
  | none => []

/-- **The link's error**: its proof system's knowledge error at its statement, `κ + √(γ·Pr[B])`. -/
noncomputable def RegLink.ε (H : Msg → Msg) (Lk : RegLink Coin) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (Pℓ : Prover Ω) : ℝ :=
  Lk.A.κ Lk.x + Real.sqrt (Lk.A.γ Lk.x * PrB H Lk.A.V Pℓ Lk.x)

/-- **The linked bad event**: the link's run (`Pℓ` on the tape `ω`) and the audit's (`Pa` on `ω` and the link's coins)
accept, and the audit's extraction relinked by the link's is not in the sampling relation at `δ` under the record
(`Cm`), or an extraction gives up. -/
noncomputable def PrBadLinked (H : Msg → Msg) {A : Arg Coin} (L : Compiled Coin A) (Lk : RegLink Coin)
    (enc : List Msg → ℕ → Bool) (Cm : CommitmentScheme Bool) (δ : ℝ) (x : Msg) {Ω : Type} [Fintype Ω]
    (Pℓ : Prover Ω) (Pa : Prover (Ω × (Fin (Lk.A.V.k Lk.x) → Coin))) : ℝ :=
  Pr fun q : Ω × (Fin (Lk.A.V.k Lk.x) → Coin) × Lk.A.E.C Lk.x × (Fin (L.V.k x) → Coin) × L.E.C x =>
    (run H Pℓ Lk.A.V Lk.x q.1 q.2.1).out = .acc ∧ (run H Pa L.V x (q.1, q.2.1) q.2.2.2.1).out = .acc ∧
      ¬ ∃ s wa wℓ, L.parse x = some s ∧ (L.E.eval H Pa x (q.1, q.2.1) q.2.2.2.1 q.2.2.2.2).1 = some wa ∧
        (Lk.A.E.eval H Pℓ Lk.x q.1 q.2.1 q.2.2.1).1 = some wℓ ∧
        Sampling.R H Cm L.parse L.partition δ x (relink enc s.roots wa (Lk.reveal s wℓ))

/-- **`B_link`** (expected time): the tape, the link's run and extraction, the audit's run and extraction, and the first
collision among the calls the audit's extraction makes itself and the checks of the pairs the link's reveals. Its cost
is both extractions' SHA-512 calls, their reruns' included, and the pairs' checks. -/
noncomputable def Blink (H : Msg → Msg) {A : Arg Coin} (L : Compiled Coin A) (Lk : RegLink Coin)
    (Cm Ce : CommitmentScheme Bool) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω] (Pℓ : Prover Ω)
    (Pa : Prover (Ω × (Fin (Lk.A.V.k Lk.x) → Coin))) : Finder :=
  match L.parse x with
  | none => { S := Unit, out := fun _ => none, cost := fun _ => 0 }
  | some s =>
    { S := Ω × (Fin (Lk.A.V.k Lk.x) → Coin) × Lk.A.E.C Lk.x × (Fin (L.V.k x) → Coin) × L.E.C x
      out := fun q =>
        findColl (ownLog H (run H Pa L.V x (q.1, q.2.1)) (L.E.ext x q.2.2.2.1 q.2.2.2.2) ++
          linkLog H Cm Ce s.roots (Lk.pairs H Pℓ s q.1 q.2.1 q.2.2.1))
      cost := fun q =>
        (L.E.eval H Pa x (q.1, q.2.1) q.2.2.2.1 q.2.2.2.2).2.1 + (Lk.A.E.eval H Pℓ Lk.x q.1 q.2.1 q.2.2.1).2.1 +
          (linkLog H Cm Ce s.roots (Lk.pairs H Pℓ s q.1 q.2.1 q.2.2.1)).length }

open Classical in
/-- **The incorrect set at a table** `T` of the statement: the proof blocks that no values equal to `T` on the program's
gates, and no interior values, make correct (`Compiled.Freg`'s form at any table); empty on a statement that doesn't
parse. -/
noncomputable def FregAt {A : Arg Coin} (L : Compiled Coin A) (x : Msg) (T : AuditStatement → ℕ → Option Bool) :
    Finset ℕ :=
  match L.parse x with
  | none => ∅
  | some s => (Finset.range (L.part s).blocks.length).filter fun v =>
      ¬ ∃ b, (L.part s).blocks[v]? = some b ∧ ∃ (val : ℕ → Option Bool) (ι : ℕ → Bool),
        (∀ g < s.prog.gates.length, val g = T s g) ∧ s.prog.Correct (s.prog.refIn b val ι) b

/-- **The committed trace through the link**, a definition nobody runs: at an encoded position, the record value the
link's extraction ties there (`linkedAt` of its pairs, at the link's run `(ω, r)` and its extractor's coins `c`);
elsewhere the committed trace of the audit's prover at the tape `ω` and the link's coins (`Compiled.regVal`). It reads
the registration, the provers and the tape, so it is fixed before any of the window's coins. -/
noncomputable def tableL (H : Msg → Msg) {A : Arg Coin} (L : Compiled Coin A) (Lk : RegLink Coin)
    (enc : List Msg → ℕ → Bool) (Cm Ce : CommitmentScheme Bool) (x : Msg) {Ω : Type} (Pℓ : Prover Ω)
    (Pa : Prover (Ω × (Fin (Lk.A.V.k Lk.x) → Coin))) (ω : Ω) (r : Fin (Lk.A.V.k Lk.x) → Coin) (c : Lk.A.E.C Lk.x)
    (s : AuditStatement) (g : ℕ) : Option Bool :=
  if enc s.roots g then linkedAt H Cm Ce s.roots (Lk.pairs H Pℓ s ω r c) g else L.regVal H Pa x (ω, r) g

/-- **The committed trace's incorrect set through the link** (`FregAt` at `tableL`). -/
noncomputable def FregL (H : Msg → Msg) {A : Arg Coin} (L : Compiled Coin A) (Lk : RegLink Coin)
    (enc : List Msg → ℕ → Bool) (Cm Ce : CommitmentScheme Bool) (x : Msg) {Ω : Type} (Pℓ : Prover Ω)
    (Pa : Prover (Ω × (Fin (Lk.A.V.k Lk.x) → Coin))) (ω : Ω) (r : Fin (Lk.A.V.k Lk.x) → Coin) (c : Lk.A.E.C Lk.x) :
    Finset ℕ :=
  FregAt L x (tableL H L Lk enc Cm Ce x Pℓ Pa ω r c)

/-- **`B_cross`**: the tape, the link's run and extraction, one audit run with the extractor's coins there, and the
first collision among the calls that run's extractions make themselves (`Compiled.runExt`) and the checks of the pairs
the link's extraction reveals. An audit extraction and the link's that reveal one encoded position two ways give one
(`CommitmentScheme.Binds` of the encoding). Its cost is the audit run's SHA-512 calls, both parties', its extractions'
and the link's extraction's, their reruns' included, and the pairs' checks. -/
noncomputable def Bcross (H : Msg → Msg) {A : Arg Coin} (L : Compiled Coin A) (Lk : RegLink Coin)
    (Cm Ce : CommitmentScheme Bool) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω] (Pℓ : Prover Ω)
    (Pa : Prover (Ω × (Fin (Lk.A.V.k Lk.x) → Coin))) : Finder :=
  match L.parse x with
  | none => { S := Unit, out := fun _ => none, cost := fun _ => 0 }
  | some s =>
    { S := Ω × (Fin (Lk.A.V.k Lk.x) → Coin) × Lk.A.E.C Lk.x × (Fin (L.V.k x) → Coin) × L.SlotEC x
      out := fun q =>
        findColl (ownLog H (run H Pa L.V x (q.1, q.2.1))
            (L.runExt x s q.2.2.2.1 (run H Pa L.V x (q.1, q.2.1) q.2.2.2.1).seen q.2.2.2.2) ++
          linkLog H Cm Ce s.roots (Lk.pairs H Pℓ s q.1 q.2.1 q.2.2.1))
      cost := fun q =>
        (run H Pa L.V x (q.1, q.2.1) q.2.2.2.1).pLog.length + (run H Pa L.V x (q.1, q.2.1) q.2.2.2.1).vLog.length +
          ((L.runExt x s q.2.2.2.1 (run H Pa L.V x (q.1, q.2.1) q.2.2.2.1).seen q.2.2.2.2).eval H
            (run H Pa L.V x (q.1, q.2.1))).2.1 +
          (Lk.A.E.eval H Pℓ Lk.x q.1 q.2.1 q.2.2.1).2.1 +
          (linkLog H Cm Ce s.roots (Lk.pairs H Pℓ s q.1 q.2.1 q.2.2.1)).length }

/-- **The committed form's error through the link**: `ε_reg + ε_link + Pr[B_cross]`. -/
noncomputable def εregLinked (H : Msg → Msg) {A : Arg Coin} (L : Compiled Coin A) (Lk : RegLink Coin)
    (Cm Ce : CommitmentScheme Bool) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω] (Pℓ : Prover Ω)
    (Pa : Prover (Ω × (Fin (Lk.A.V.k Lk.x) → Coin))) : ℝ :=
  L.εreg H x Pa + Lk.ε H Pℓ + (Bcross H L Lk Cm Ce x Pℓ Pa).Pr H

end Composed

end Claims.Encodings
