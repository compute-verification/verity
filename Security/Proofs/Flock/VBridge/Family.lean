import Proofs.Flock.VBridge.Top
import VStar.Stage
import VStar.Compose

/-!
# VBridge G at V*'s statement family

`vbridge_of_vstar` (`Top.lean`) takes V*'s statements as free objects. `VStarFamily` ties them to V*'s executable Lean
(`verity/core/service/vstar`):

* the statements are `VStar.statements` at the inner session's shape (`Stage.lean:56-66`: each Ligerito level's
  `RecOpen{LANES, H}` over `2 q` instances, then each part's `InnerRepCheck{S}` over two), the verifier's shape the one
  `RecResiduals.shape?` gives at that shape's `m`, `k_log` and claims, as `RecResiduals.check` derives it from a part's
  unit name;
* session `j`'s circuit is the one the verifier parses from the circuit file it reads (`ZkOuter.hc`), and that file is
  the one `vstar stage` writes for statement `j` (`circuitFile`, `Cli.lean:43-53`);
* part `p` is computed by the statement `statements` tags `.inner p`, rep `r`'s values by its instance `r`
  (`rec_vstage.py:409-412`: row `r` of each read, `v` of rep `r`); opening `j` of level `ℓ` of rep `r` by the statement
  tagged `.recOpen ℓ`, its instance `r q + j` (`rec_outer.levels`: rep, then query); every instance is copy 0 of its unit
  (`units_per_vu` 1, `Compose.lean:224`);
* the openings' query counts, their chain positions (`rec_outer.chain_order`: rep-major, row 0 before rep 0's first),
  their coefficients (`rec_outer.coefficients`: `κ[ℓ][r][l] · eq(α)[j]` at the rep's coins) and each rep's consistency
  sums (`rec_algebra.consistency`, from the opened rows) are defined from the rest, so `CoefsAre` and `ConsAre` hold by
  definition (`coefsAre`, `consAre`).

Top's hypotheses name one unit and copy per part (`un p`, `cp p`) and per opening (`unO ℓ j`, `cpO ℓ j`) for both reps.
V* computes the two reps in two instances, and a registered bit has one value (`AccChain.regBit_unique`), so at those
maps `hres` and `hu` would force the reps' values equal. Top's `rep`, `ops` and `files` are functions of the layers
alone, but each rep's coins, extra claims and comb, each level's `α` and the table files' streams are the verifier's,
drawn from the history `t`, and V*'s wires register only what is computed from them (`v`, `coef`, `dirs`). So
`vbridge_of_reps` is `vbridge_of_vstar` with the unit maps indexed by the rep and `rep`, `acc`, `ops`, `files` read at the
history as well, from the same per-rep lemmas. Its `ResZeroAt` and `OpensAt` take the parts' and openings' outputs as
registered reads (`RegOut`). At the family the parts' and openings' outputs are read in the layers instead, and
`vbridge_of_family` (`Openings.lean`) composes the same per-rep lemmas at `HResL` (`Rep.lean`) and `HUL`:
`Flock.Assumptions.VBridge` from one named `Prop` per remaining hypothesis.
-/

namespace FlockVBridge.Family

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Recursion FlockSoundness.Refine FlockSoundness.Model FlockSoundness.Audit
open Flock FlockLevel3 FlockVBridge.Compiled FlockVBridge.InnerAccepts FlockVBridge.Stages
open Flock.RecResiduals (portSizes groups)
open Flock.RecAlgebra (structureOf verifierRows el)
open FlockVBridge.OfVStar (ResZeroAt)
open FlockVBridge.AccChain (Openings OpensAt CoefsAre ConsAre ReadsEnds levelLanes)

/-- **V*'s circuit file of a statement**: the bytes `vstar stage` writes to `<dir>/circuit.txt` (`VStar.Cli.write`,
`Cli.lean:43-53`): the statement's unit, laid out, emitted with its subcircuit and the statement's bindings
(`VStar.emitCircuit`). -/
def circuitFile (s : VStar.Stmt) (b : VStar.Bindings) : Except String ByteArray := do
  let u ← s.unit
  let l ← u.laid
  let e ← VStar.emitCircuit (m := StateM String) (fun x => modify (· ++ x)) u l s.sub b
  pure (Id.run (e.run "")).2.toUTF8

/-- Every part's instance is copy 0 of its unit (`units_per_vu` 1, `Compose.lean:224`). -/
def cp (_ : ℕ) (_ : Fin 2) : ℕ := 0

/-- Every opening's instance is copy 0 of its unit. -/
def cpO (_ : ℕ) (_ : Fin 2) (_ : ℕ) : ℕ := 0

/-! ## Top's composition, each unit map indexed by the rep -/

section reps

variable {n : ℕ} {L : Law n} {Reg : Type} {Cm : Type} {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type}
variable {nT nR : ℕ} (T : Fin nT → Type) (CT : Fin nR → Type) [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
variable (cls : HClass) (regs : List (Region cls.kLog (cls.m - cls.kLog))) (sch : Flock.Schedule)

local notation "ℐ" => flockInnerCL T CT execArith H512 enc512 InnerAccepts.leafPlain cls regs (schedOf sch)

/-- **`vbridge_of_vstar` with a unit and copy per rep, and the reps' values at the history**: part `p` of rep `r` is
copy `cp p r` of unit `un p r` of session `sess p`, opening `j` of level `ℓ` of rep `r` copy `cpO ℓ r j` of unit
`unO ℓ r j` of session `sessO ℓ`; each rep's values, its openings, `rec-acc`'s rows and the table files are read at the
history `(R, ω, t)` as well as the layers, and `VFull` is taken whole (`vFull_of_whole` gives it). The proof is `vbridge_compiled`'s: `TableFiles.accepts`, its runs from the
per-rep lemmas (`algebraHolds_of_decode`, `accHolds_of_units` on `unitSums_of_decode`) at each rep's maps. -/
theorem vbridge_of_reps
    (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (x : Reg → L.Ω → Hidden cls)
    (wmsg : ((j : Fin S) → (Vs j).Bits) → List (ℐ).Msg)
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → (ℐ).Msg)
    (sh : Shape) (rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData)
    {GS : Flock.RecAlgebra.GfS} (hS : structureOf sh = .ok GS)
    {gs : Array (Array ℕ × Array ℕ × Array ℕ)} (hg : groups GS (portSizes sh) = .ok gs)
    (sess : ℕ → Fin S) (un : (p : ℕ) → Fin 2 → Fin (Vs (sess p)).nv) (cp : ℕ → Fin 2 → ℕ)
    (hvr : ∀ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ r, ∃ v info kappa,
      verifierRows sh (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).rounds
        (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).extras (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).comb =
          .ok (v, info, kappa) ∧ v.size = GS.nv)
    (hsz : ∀ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ r,
      (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).w.size = GS.nw ∧
        (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).comb.size = 64)
    (hres : ∀ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ r, ∀ p < gs.size,
      ResZeroAt (zs R ω t R₂ (sess p)) (layers zs dg N k Rw R ω t R₂ τ (sess p)) (un p r) (cp p r) sh gs p
        (el (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).w) (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).rounds
        (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).extras (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).comb)
    (acc : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → ℕ → Array F128)
    (ops : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → Openings)
    (sessO : ℕ → Fin S) (unO : (ℓ : ℕ) → Fin 2 → ℕ → Fin (Vs (sessO ℓ)).nv) (cpO : ℕ → Fin 2 → ℕ → ℕ)
    (hu : ∀ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ r, ∀ ℓ < sh.sched.levels.size,
      ∀ j < (ops R ω t (layers zs dg N k Rw R ω t R₂ τ) r).q ℓ,
        OpensAt (zs R ω t R₂ (sessO ℓ)) (layers zs dg N k Rw R ω t R₂ τ (sessO ℓ)) (unO ℓ r j) (cpO ℓ r j)
          (levelLanes sh ℓ) ((ops R ω t (layers zs dg N k Rw R ω t R₂ τ) r).rows ℓ)[j]!
          #[(acc R ω t (layers zs dg N k Rw R ω t R₂ τ)
              ((ops R ω t (layers zs dg N k Rw R ω t R₂ τ) r).start ℓ + j))[0]!,
            (acc R ω t (layers zs dg N k Rw R ω t R₂ τ)
              ((ops R ω t (layers zs dg N k Rw R ω t R₂ τ) r).start ℓ + j))[1]!]
          ((ops R ω t (layers zs dg N k Rw R ω t R₂ τ) r).coef ℓ j)
          (acc R ω t (layers zs dg N k Rw R ω t R₂ τ)
            ((ops R ω t (layers zs dg N k Rw R ω t R₂ τ) r).start ℓ + j + 1)))
    (hcoef : ∀ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ r,
      CoefsAre sh (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r) (ops R ω t (layers zs dg N k Rw R ω t R₂ τ) r))
    (hcons : ∀ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ r,
      ConsAre sh (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r) (ops R ω t (layers zs dg N k Rw R ω t R₂ τ) r))
    (hrd : ∀ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ r,
      ReadsEnds sh (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r) (acc R ω t (layers zs dg N k Rw R ω t R₂ τ))
        (ops R ω t (layers zs dg N k Rw R ω t R₂ τ) r))
    (files : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → TableFiles cls)
    (hsch : ∀ R ω t l, (files R ω t l).sch = sch)
    (hfiles : ∀ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
      (files R ω t (layers zs dg N k Rw R ω t R₂ τ)).Valid regs (x R ω) T CT
        ((wmsg (layers zs dg N k Rw R ω t R₂ τ)).zip (t.map Prod.snd)))
    (hruns : ∀ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
      (∀ r, AlgebraHolds sh (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r) ∧
        AccHolds sh (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r)) →
      (files R ω t (layers zs dg N k Rw R ω t R₂ τ)).Runs)
    (hcm : VCommits zs Sk cmt) (hdec : VDecodesR wmsg (fun _ _ a b => a = b) Sk msg) (hfull : VFullR (ℐ).r Sk) :
    Flock.Assumptions.VBridge (I := ℐ) zs dg N k Rw x wmsg Sk cmt msg :=
  ⟨fun R ω t R₂ hc τ o h₀ =>
    have h : Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o := ⟨h₀, hc⟩
    (files R ω t (layers zs dg N k Rw R ω t R₂ τ)).accepts regs (x R ω) T CT sch (hsch R ω t _) _
      (hfiles R ω t R₂ τ o h) (hruns R ω t R₂ τ o h fun r => by
        obtain ⟨v, info, kappa, hv, hvs⟩ := hvr R ω t R₂ τ o h r
        obtain ⟨hw, hc⟩ := hsz R ω t R₂ τ o h r
        exact ⟨FlockVBridge.OfVStar.algebraHolds_of_decode (zs R ω t R₂) (layers zs dg N k Rw R ω t R₂ τ)
            (fun s => (h.1 s).2) hS (Flock.RecAlgebra.structureOf_bitsOk hS) hv hw hc hvs hg sess (fun p => un p r)
            (fun p => cp p r) (hres R ω t R₂ τ o h r),
          FlockVBridge.AccChain.accHolds_of_units
            (FlockVBridge.AccChain.unitSums_of_decode (zs R ω t R₂) (layers zs dg N k Rw R ω t R₂ τ)
              (fun s => (h.1 s).2) sh _ _ sessO (fun ℓ j => unO ℓ r j) (fun ℓ j => cpO ℓ r j) (hu R ω t R₂ τ o h r))
            (hcoef R ω t R₂ τ o h r) (hcons R ω t R₂ τ o h r) (hrd R ω t R₂ τ o h r)⟩),
    hcm, hdec, hfull⟩

end reps

/-! ## The family -/

section family

variable {nT nR : ℕ} (T : Fin nT → Type) (CT : Fin nR → Type) [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
variable (cls : HClass) (regs : List (Region cls.kLog (cls.m - cls.kLog))) (sch : Flock.Schedule)
variable {n : ℕ} (L : Law n) (Reg Cm Reg' Reg₂ : Type)

local notation "ℐ" => flockInnerCL T CT execArith H512 enc512 InnerAccepts.leafPlain cls regs (schedOf sch)

/-- **V*'s statement family over one inner session.** The objects `vbridge_of_vstar` reads, each tied to V*'s code by
an equation: the statements `VStar.statements` lists at V*'s shape `vsh` (`hstmts`), the verifier's shape `sh` the one
`RecResiduals.shape?` gives at `vsh` (`hsh`), each session reading the circuit file `vstar stage` writes for its
statement at its bindings (`hfile`), whose parse carries V*'s META (`hparsed`), over one unit per instance (`hnv`), and
the units of each part and opening
(`hsess`, `hun`, `hsessO`, `hunO`). The values the stages read off the layers (`rep₀`, `rows`, `alpha`, `acc`, `wmsg`)
and V*'s registered data (`files`, `Sk`, `cmt`, `msg`) are the remaining fields. -/
structure VStarFamily where
  /-- The inner session's shape as V* takes it: `m`, `k_log` and claims (`rec_vstage.shape`). -/
  vsh : VStar.Algebra.Shape
  /-- The verifier's shape. -/
  sh : Shape
  hsh : RecResiduals.shape? vsh.m vsh.kLog vsh.claims = some sh
  /-- V*'s statements (`Stage.lean:56-66`). -/
  stmts : Array VStar.Stmt
  hstmts : VStar.statements vsh = .ok stmts
  /-- Each statement's bindings: its partition's digest, its Program's SHA-512 and `program_digests`. -/
  bind : Fin stmts.size → VStar.Bindings
  Vs : Fin stmts.size → VStmt
  /-- V*'s sessions at each history and registration. -/
  zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin stmts.size) → (Vs j).Outer Reg'
  hfile : ∀ R ω t R₂ j, circuitFile stmts[j] (bind j) = .ok (zs R ω t R₂ j).I.circuitFile
  /-- Each statement's circuit carries V*'s META (`RecVStar.checkParsed`, which `checkCircuit` runs on every session it
  accepts after `checkFile`, `checkCircuit_ok`): what `HmRow.parse` reads through `Lean.Json.parse`, which no proof
  unfolds, stated as the check's verdict. -/
  hparsed : ∀ j, RecVStar.checkParsed stmts[j] (Vs j).c = .ok ()
  hnv : ∀ j, (Vs j).nv = stmts[j].instances
  /-- Each statement's circuit is over plain parts: its META carries no `holo` (`VStar.metaHead` writes none, so
  `hfile`'s bytes say so, but no META parse is stated here). -/
  hplain : ∀ j, RecResiduals.holoOf (Vs j).c = .ok false
  /-- The statement of part `p`. -/
  sess : ℕ → Fin stmts.size
  hsess : ∀ p (j : Fin stmts.size), (∃ nm pt, stmts[j] = .inner p nm pt) → sess p = j
  /-- Rep `r`'s instance of part `p`. -/
  un : (p : ℕ) → Fin 2 → Fin (Vs (sess p)).nv
  hun : ∀ p r, r.val < (Vs (sess p)).nv → (un p r).val = r.val
  /-- The statement of level `ℓ`'s openings. -/
  sessO : ℕ → Fin stmts.size
  hsessO : ∀ ℓ (j : Fin stmts.size), (∃ LANES H m, stmts[j] = .recOpen ℓ LANES H m) → sessO ℓ = j
  /-- Rep `r`'s opening `j` of level `ℓ`. -/
  unO : (ℓ : ℕ) → Fin 2 → ℕ → Fin (Vs (sessO ℓ)).nv
  hunO : ∀ ℓ r j, r.val * (sh.sched.levels[ℓ]!).q + j < (Vs (sessO ℓ)).nv →
    (unO ℓ r j).val = r.val * (sh.sched.levels[ℓ]!).q + j
  /-- The hidden circuit's statement at each history. -/
  x : Reg → L.Ω → Hidden cls
  /-- Each rep's message elements in V*'s layers, its coins, extra claims and comb the verifier's at the history (its
  consistency sums are `rep`'s). -/
  rep₀ : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin stmts.size) → (Vs j).Bits) → Fin 2 → RepData
  /-- Each rep's opened rows of each level. -/
  rows : ((j : Fin stmts.size) → (Vs j).Bits) → Fin 2 → ℕ → Array (Array F128)
  /-- Each rep's `α` of each level at the history, whose `eq` weighs its openings. -/
  alpha : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin stmts.size) → (Vs j).Bits) → Fin 2 → ℕ → Array F128
  /-- The rows of `rec-acc`, the running sums. -/
  acc : ((j : Fin stmts.size) → (Vs j).Bits) → ℕ → Array F128
  /-- The inner transcript's messages V*'s layers hold. -/
  wmsg : ((j : Fin stmts.size) → (Vs j).Bits) → List (ℐ).Msg
  /-- The inner session's table files at the history (its streams' coins are the verifier's). -/
  files : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin stmts.size) → (Vs j).Bits) → TableFiles cls
  /-- Each stage's commit strings in each session. -/
  Sk : ℕ → (j : Fin stmts.size) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv)
  /-- The digests a commitment registers at each commit string. -/
  cmt : (j : Fin stmts.size) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg
  /-- The message each stage's commit strings decode to. -/
  msg : ℕ → ((j : Fin stmts.size) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → (ℐ).Msg

variable {T CT cls regs sch L Reg Cm Reg' Reg₂}

namespace VStarFamily

variable (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂)

/-- Level `ℓ`'s queries of each rep. -/
def q (ℓ : ℕ) : ℕ := (F.sh.sched.levels[ℓ]!).q

/-- Each rep's openings. -/
def nOpen : ℕ := ((List.range F.sh.sched.levels.size).map F.q).sum

/-- The `coef` the verifier registers for opening `j` of level `ℓ` of rep `r` (`rec_outer.coefficients`): `κ[ℓ]`'s
two rows, each scaled by `eq(α)[j]`, at the `kappa` `verifierRows` gives at the rep's coins. -/
def coef (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin)) (l : (j : Fin F.stmts.size) → (F.Vs j).Bits) (r : Fin 2)
    (ℓ j : ℕ) : Array F128 :=
  match verifierRows F.sh (F.rep₀ R ω t l r).rounds (F.rep₀ R ω t l r).extras (F.rep₀ R ω t l r).comb with
  | .ok (_, _, kappa) => coefAt (kappa[ℓ]!) (levelLanes F.sh ℓ) (eqTable (F.alpha R ω t l r ℓ))[j]!
  | .error _ => #[]

/-- **Rep `r`'s openings**: level `ℓ`'s `q` queries, from chain row `r · nOpen` (`rec_outer.chain_order`, rep-major),
the rows and `α` in the layers, and the verifier's coefficients. -/
def ops (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin)) (l : (j : Fin F.stmts.size) → (F.Vs j).Bits) (r : Fin 2) :
    Openings :=
  ⟨r.val * F.nOpen, F.q, F.rows l r, F.alpha R ω t l r, F.coef R ω t l r⟩

/-- **Rep `r`'s values**: `rep₀`'s, with the levels' consistency sums of the opened rows (`rec_algebra.consistency`). -/
def rep (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin)) (l : (j : Fin F.stmts.size) → (F.Vs j).Bits) (r : Fin 2) :
    RepData :=
  { F.rep₀ R ω t l r with
    cons := fun ℓ i => Flock.RecAlgebra.consAt (F.rows l r ℓ) (F.q ℓ) (F.alpha R ω t l r ℓ) i }

/-- **The verifier's coefficients are the openings'**, by definition of `coef`. -/
theorem coefsAre (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin)) (l : (j : Fin F.stmts.size) → (F.Vs j).Bits)
    (r : Fin 2) : CoefsAre F.sh (F.rep R ω t l r) (F.ops R ω t l r) := by
  intro v info kappa h ℓ _ j _
  have h' : verifierRows F.sh (F.rep₀ R ω t l r).rounds (F.rep₀ R ω t l r).extras (F.rep₀ R ω t l r).comb =
      .ok (v, info, kappa) := h
  show F.coef R ω t l r ℓ j = coefAt (kappa[ℓ]!) (levelLanes F.sh ℓ) (eqTable (F.alpha R ω t l r ℓ))[j]!
  unfold coef
  rw [h']

/-- **The rep's consistency sums are the opened rows'**, by definition of `rep`. -/
theorem consAre (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin)) (l : (j : Fin F.stmts.size) → (F.Vs j).Bits)
    (r : Fin 2) : ConsAre F.sh (F.rep R ω t l r) (F.ops R ω t l r) :=
  fun _ _ _ _ => rfl

/-- **Session `j`'s circuit is the parse of V*'s file for statement `j`**: the file `vstar stage` writes is the one the
session reads (`hfile`), and the verifier parses its circuit from it (`ZkOuter.hc`). -/
theorem parse_circuitFile (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin)) (R₂ : Reg₂) (j : Fin F.stmts.size)
    {file : ByteArray} (h : circuitFile F.stmts[j] (F.bind j) = .ok file) :
    HmRow.parse (ZkExec.zkTags (F.zs R ω t R₂ j).I) file (F.zs R ω t R₂ j).I.tables = .ok (F.Vs j).c := by
  have e := F.hfile R ω t R₂ j
  rw [h] at e
  injection e with e
  rw [e]
  exact (F.zs R ω t R₂ j).hc

end VStarFamily

/-! ## The obligations -/

namespace VStarFamily

variable (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ)

/-- **`hS`, `hg`**: the verifier's structure at `sh` and its parts' groups. -/
def HStruct (GS : Flock.RecAlgebra.GfS) (gs : Array (Array ℕ × Array ℕ × Array ℕ)) : Prop :=
  structureOf F.sh = .ok GS ∧ groups GS (portSizes F.sh) = .ok gs

/-- **`HStruct` from the verifier's check of one part**: a circuit `RecResiduals.check` passes (`flock-verify` runs it on
every circuit it reads), its unit named part `p` at a shape `RecResiduals.shape?` gives as `sh`, has the structure and
the groups return at `sh` (`RecResiduals.part` computes both), its parts plain (`holoOf`). -/
theorem hStruct_of_check {c : Flock.Circuit} (hc : RecResiduals.check c = .ok ()) {m k cl p : ℕ}
    (hp : RecResiduals.params c.unit.name = some (m, k, cl, p)) (hsh : RecResiduals.shape? m k cl = some F.sh)
    (hh : RecResiduals.holoOf c = .ok false) : ∃ GS gs, F.HStruct GS gs := by
  obtain ⟨S, ns, nv, hpart, -⟩ := FlockVBridge.OfVStar.check_part hc hp hsh hh
  unfold RecResiduals.part at hpart
  obtain ⟨GS, hGS, hpart⟩ := FlockVBridge.PartS.bind_ok hpart
  dsimp only at hpart
  split at hpart
  · cases hpart
  · obtain ⟨gs, hgs, -⟩ := FlockVBridge.PartS.bind_ok hpart
    exact ⟨GS, gs, hGS, hgs⟩

/-- **`hvr`, `hsz`**: at every accepted outer proof, the verifier rows at each rep's coins, `GS`'s `nv` of them, and the
rep's `w` and comb of `GS`'s `nw` and 64 elements. -/
def HShape (GS : Flock.RecAlgebra.GfS) : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ r,
    (∃ v info kappa, verifierRows F.sh (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).rounds
      (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).extras
        (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).comb = .ok (v, info, kappa) ∧ v.size = GS.nv) ∧
    (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).w.size = GS.nw ∧
      (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).comb.size = 64

/-- **`hrd`**: at every accepted outer proof, V*'s algebra reads `rec-acc` at each rep's ends of the chain. -/
def HReads : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ r,
    ReadsEnds F.sh (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r) (F.acc (layers F.zs dg N k Rw R ω t R₂ τ))
      (F.ops R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r)

/-- **`hsch`**: the table files are the inner schedule's. -/
def HSch : Prop := ∀ R ω t l, (F.files R ω t l).sch = sch

/-- **`hfiles`**: at every accepted outer proof, the table files are valid for the transcript V*'s layers hold at the
rounds' coins. -/
def HFiles : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o →
    (F.files R ω t (layers F.zs dg N k Rw R ω t R₂ τ)).Valid regs (F.x R ω) T CT
      ((F.wmsg (layers F.zs dg N k Rw R ω t R₂ τ)).zip (t.map Prod.snd))

/-- **`hruns`**: at every accepted outer proof whose reps meet E's preconditions, both reps' runs of the table files. -/
def HRuns : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o →
    (∀ r, AlgebraHolds F.sh (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r) ∧
      AccHolds F.sh (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r)) →
    (F.files R ω t (layers F.zs dg N k Rw R ω t R₂ τ)).Runs

/-- **`hcm`**: V*'s public files register the stages' commitments. -/
def HCommits : Prop := VCommits F.zs F.Sk F.cmt

/-- **`hdec`**: V*'s circuits decode each stage's message from the read wires of its commit strings. -/
def HDecodes : Prop := VDecodesR F.wmsg (fun _ _ a b => a = b) F.Sk F.msg

/-- **`hio`**: each stage's commit string is a wire of a unit. -/
def HFull : Prop :=
  ∀ i < (ℐ).r, ∀ j g, posK (F.Vs j).c (F.Vs j).hU (F.Vs j).nv g ∈ F.Sk i j →
    ∃ u, g ∈ ((Placed.keyProg (F.Vs j).c (F.Vs j).hU (F.Vs j).nv).partition
      (Layout.outsOf (F.Vs j).c (F.Vs j).hU (F.Vs j).nv)).io u

/-- `VFull` under the full draw (`vFull_of_whole`). -/
theorem vFull_of_hFull (h : F.HFull) : VFull (ℐ).r F.Sk := FlockVBridge.Full.vFull_of_whole _ F.Sk h

end VStarFamily

end family

end FlockVBridge.Family
