import Proofs.Flock.C3.Compose.Honest
import Proofs.Flock.VBridge.Family.Registered

/-!
# The rows `own` names are registered reads, at candidate 3's family

`RowsRegisteredC3` is VBridge's `RowsRegistered` (`VBridge/Family/Registered.lean`) at `C3Family`: at every accepted
outer proof, every row of a port the session's reads file `own` names is a registered read of the session's record `rd`
at every unit. At the checked family (`C3Family.checked`) the sessions' records are the prover's: `checkedC3` keeps
`custodied`'s `rd`, which `ZkOuterR.build` takes from `zr` (`Recursive/Raw.lean`), and `RecursiveAuditC3At` quantifies
`zr`. So no proof from acceptance exists: the prover's `zr` with `rd.isRead := False` meets `hown` and `ReadsX` vacuously,
`Accepted` reads `rd` only through `RegAgree` and `RegColl`, both vacuous there, and the honest runs stay accepted while
`RowsRegisteredC3` fails.

* `CoversAt`, `RowsCovered`: the record form, over `rd` and `own` alone, with no outcome, layer or transcript;
  `rowsRegisteredC3_of_covered` gives `RowsRegisteredC3` from it at every outcome, reached or not.
* `RegistrantCovers`: the premise on the prover's sessions `zr`, the record form at each typing of each raw session.
* `rowsCovered_checked`: `RowsCovered` at the checked family from `RegistrantCovers zr`; `keepOwn` and `ZkOuter.refuse`
  only narrow `own` to `freshOwn`, which names no port of the circuit (`fresh_not_named`).
-/

namespace FlockC3.Compose

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled

/-- **A session's record covers the rows its reads file names**: every row of a port `own` names is a registered read
of `rd`, at every unit. -/
def CoversAt {Q : VStmt} {Reg' : Type} (z : Q.Outer Reg') : Prop :=
  ∀ (u : Fin Q.nv) (q : Fin Q.c.ports.size), (∃ pt ∈ z.own, pt.name = Q.c.ports[q].name) → z.rd.isRead (.inl (u, q))

/-- Premise, standing in for the one-run extraction of the registrant's openings: no accepted run carries them, since a `--zk` transcript holds a registered row's `b ‖ c` and path only, and
`ReadsX.committed` asks for the row and salt under it. **The registrant's record covers the rows its reads file names.**
At every history and registration, at each typing of V*'s session `j` as the prover supplies it, every row of a port the
session's reads file `own` names is a registered read of the session's record `rd`, at every unit. The records are the
inner session's prover's: it committed each round's rows and salts behind the firewall (each row's `b ‖ c` is the round's
commitment the coin server records, `InnerRecordCustody`), and the registered roots and positions are the verifier's
own (`own`, `--registered`). -/
def RegistrantCovers {S : ℕ} {Vs : Fin S → VStmt} {Reg' A B C D : Type}
    (zr : A → B → C → D → (j : Fin S) → (Vs j).OuterR Reg') : Prop :=
  ∀ R ω t R₂ j ht S₀ d₀ y₀ hc hpub (u : Fin (Vs j).nv) (q : Fin (Vs j).c.ports.size),
    (∃ pt ∈ (zr R ω t R₂ j).own, pt.name = (Vs j).c.ports[q].name) →
      ((zr R ω t R₂ j).rd ht S₀ d₀ y₀ hc hpub).isRead (.inl (u, q))

/-- `freshOwn c` names no port of `c`. -/
theorem fresh_not_named {c : Flock.Circuit} (q : Fin c.ports.size) {pt : Flock.Registered.Port}
    (hpt : pt ∈ freshOwn c) : pt.name ≠ c.ports[q].name := by
  intro h
  have hpt' : pt = { (default : Flock.Registered.Port) with name := freshName c } := Array.mem_singleton.mp hpt
  have hn : pt.name = freshName c := by rw [hpt']
  exact freshName_ne c q.isLt (by rw [getElem!_pos c.ports q.val q.isLt]; exact h.symm.trans hn)

theorem coversAt_keepOwn {Q : VStmt} {Reg' : Type} {z : Q.Outer Reg'}
    {P : FlockSoundness.Discharge.Exec.Inputs → Array Flock.Registered.Port →
      FlockSoundness.Discharge.Exec.Transcript → Prop} (hz : CoversAt z) :
    CoversAt (Flock.Guarantees.keepOwn z P) := by
  intro u q ⟨pt, hpt, hn⟩
  by_cases hp : ∃ t, AcceptsZKR z.I z.own t ∧ P z.I z.own t
  · have ho : (Flock.Guarantees.keepOwn z P).own = z.own := ite_eq_left hp
    rw [ho] at hpt
    exact hz u q ⟨pt, hpt, hn⟩
  · have ho : (Flock.Guarantees.keepOwn z P).own = freshOwn Q.c := ite_eq_right hp
    rw [ho] at hpt
    exact absurd hn (fresh_not_named q hpt)

theorem coversAt_refuse {Q : VStmt} {Reg' : Type} (z : Q.Outer Reg') : CoversAt z.refuse :=
  fun _ q ⟨_, hpt, hn⟩ => absurd hn (fresh_not_named q hpt)

theorem coversAt_zsOf {S : ℕ} {Vs : Fin S → VStmt} {Reg' A B C D : Type}
    {zr : A → B → C → D → (j : Fin S) → (Vs j).OuterR Reg'} (hcov : RegistrantCovers zr)
    (hRec : ∀ R ω t R₂ j, RecordsLiveZKJ (zr R ω t R₂ j).I (zr R ω t R₂ j).dj (fun _ _ => (Vs j).J)
      (zr R ω t R₂ j).wr) (h : Typed zr) (R : A) (ω : B) (t : C) (R₂ : D) (j : Fin S) :
    CoversAt (zsOf zr hRec h R ω t R₂ j) := by
  by_cases hx : (zr R ω t R₂ j).Typable
  · rw [zsOf_pos zr hRec h R ω t R₂ j hx]
    exact fun u q hq => hcov R ω t R₂ j _ _ _ _ _ _ u q hq
  · rw [zsOf_neg zr hRec h R ω t R₂ j hx]
    exact coversAt_refuse _

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

namespace C3Family

variable (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂)

/-- Obligation: `RowsCovered` gives it (`rowsRegisteredC3_of_covered`), and at the checked
family `RegistrantCovers` gives that (`rowsCovered_checked`). **The rows `own` names are registered reads.** At every
accepted outer proof, in each session, the row of every port the session's reads file `own` names is a registered
read at every unit (VBridge's `RowsRegistered` at candidate 3's family). -/
def RowsRegisteredC3 (dg : ByteArray → ByteArray) (N k Rw : ℕ) : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := holoInnerAt T CT st hp) F.zs dg N k Rw R ω t R₂ τ o → ∀ j (u : Fin (F.Vs j).nv)
    (q : Fin (F.Vs j).c.ports.size), (∃ pt ∈ (F.zs R ω t R₂ j).own, pt.name = (F.Vs j).c.ports[q].name) →
      (F.zs R ω t R₂ j).rd.isRead (.inl (u, q))

/-- **Every session's record covers the rows its reads file names** (`CoversAt`), at every history and registration. -/
def RowsCovered : Prop :=
  ∀ R ω t R₂ j, CoversAt (F.zs R ω t R₂ j)

theorem rowsRegisteredC3_of_covered (h : F.RowsCovered) (dg : ByteArray → ByteArray) (N k Rw : ℕ) :
    F.RowsRegisteredC3 dg N k Rw :=
  fun R ω t R₂ _ _ _ j u q hq => h R ω t R₂ j u q hq

end C3Family

/-- **`RowsCovered` at the checked family** from `RegistrantCovers` on the prover's sessions: a checked session is
`keepOwn` of the typed one (`checkedC3`), which is `zr`'s build at a typing, with its `own` and `rd`, or refuses
(`zsOf`). -/
theorem rowsCovered_checked (ish : RecVStar.Inner) {ss : Array VStar.Stmt} (hss : RecVStar.statements ish = .ok ss)
    (bind : Fin (vstarCount ish) → VStar.Bindings) {Vs : Fin (vstarCount ish) → VStmt}
    (hob : VStarParsesScoped Flock.Tags.circuit ish bind Vs)
    (zr : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin (vstarCount ish)) →
      (Vs j).OuterR Reg')
    (hCust : Flock.Premises.RecordCustody zr) (h : Typed zr) (coins : String → Option Flock.RecCoins.Source)
    (rows : Option (Array Flock.Registered.Port))
    (own : Fin (vstarCount ish) → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop)
    (bc : (j : Fin (vstarCount ish)) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → List UInt8)
    (dec : ℕ → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → (holoInnerAt T CT st hp).Msg)
    (wr : List (Cm × (holoInnerAt T CT st hp).Coin) → Lean.Json × String) (reg : Lean.Json)
    (hcov : RegistrantCovers zr) :
    (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg).RowsCovered :=
  fun R ω t R₂ j => by
    refine coversAt_keepOwn ?_
    dsimp only [Flock.Guarantees.custodied]
    exact coversAt_zsOf hcov _ h R ω t R₂ j

end FlockC3.Compose
