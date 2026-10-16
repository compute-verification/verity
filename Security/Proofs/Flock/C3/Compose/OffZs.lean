import Proofs.Flock.C3.Compose.RepsOff
import Proofs.Flock.C3.Compose.LaterHolo
import Proofs.Flock.C3.Compose.PartRead

/-!
# V*'s sessions refusing a bad event

`offZs zs Bad` is V*'s sessions `zs`, each refusing everything at a history and registration where `Bad` holds: its own
list there is `freshOwn`, which accepts nothing (`dropOwn`, `keepOwn`'s pattern, `not_acceptsZKR_fresh`). A session's
game, layer and every read but its own list are `zs`'s, so an outer proof `offZs zs Bad` accepts is one `zs` accepts
off `Bad` (`accepted_offZs`), and back (`offZs_accepted`). Hence:

* a fact at every proof `zs` accepts holds at every proof `offZs zs Bad` accepts (`hLeafC3_offZs`, `atSrc_offZs`,
  `atRep_offZs`, `mapsAt_offZs`, `srcHTie_offZs`, and the reads across two proofs, `repsMsgsRead_offZs`,
  `partMsgsRead_offZs`, `opensReadAt_offZs`, `rsRead_offZs`);
* a fact off `Bad` holds at every proof `offZs zs Bad` accepts (`algebraDecodes_offZs`, `accChains_offZs`);
* a fact at every proof the honest family at `offZs zs Bad` accepts holds off `Bad` at the honest family at `zs`
  (`atAcceptedOff_offZs`).

So a source's obligations need hold only off `Bad`, where the value ties do: `hbr_honest_src` builds `SrcOK` and the
fixed decoder at `offZs zs Bad`, and moves each read fact back to `zs` off `Bad`.
-/

namespace FlockC3.Compose

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.Reps (RepsMsgsRead AtRep)

open Classical in
/-- **A session refusing everything where `p` holds**: `z`, with the own list `freshOwn` where `p` holds. Its game
and every other field are `z`'s. -/
noncomputable def dropOwn {Q : VStmt} {Reg' : Type} (z : Q.Outer Reg') (p : Prop) : Q.Outer Reg' :=
  { z with
    own := if p then freshOwn Q.c else z.own
    hown := fun ω R' o h => by
      by_cases hp : p
      · rw [if_pos hp] at h
        exact (not_acceptsZKR_fresh z.I z.ht z.hTags z.hcH _ h).elim
      · rw [if_neg hp] at h ⊢
        exact z.hown ω R' o h }

open Classical in
theorem own_dropOwn_pos {Q : VStmt} {Reg' : Type} (z : Q.Outer Reg') {p : Prop} (hp : p) :
    (dropOwn z p).own = freshOwn Q.c := by
  show (if p then freshOwn Q.c else z.own) = freshOwn Q.c
  exact if_pos hp

open Classical in
theorem own_dropOwn_neg {Q : VStmt} {Reg' : Type} (z : Q.Outer Reg') {p : Prop} (hp : ¬ p) :
    (dropOwn z p).own = z.own := by
  show (if p then freshOwn Q.c else z.own) = z.own
  exact if_neg hp

/-- **A session refusing `p` accepts nothing where `p` holds.** -/
theorem not_accepts_dropOwn {Q : VStmt} {Reg' : Type} (z : Q.Outer Reg') {p : Prop} (hp : p)
    (tr : FlockSoundness.Discharge.Exec.Transcript) : ¬ AcceptsZKR (dropOwn z p).I (dropOwn z p).own tr := by
  rw [own_dropOwn_pos z hp]
  exact not_acceptsZKR_fresh z.I z.ht z.hTags z.hcH tr

/-- **V*'s sessions refusing `Bad`**: each session refuses everything at a history and registration where `Bad`
holds (`dropOwn`). -/
def offZs {I : Inner} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (Bad : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → Prop) :
    Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg' :=
  fun R ω t R₂ j => dropOwn (zs R ω t R₂ j) (Bad R ω t R₂)

/-- **An outer proof the sessions refusing `Bad` accept is one `zs` accepts off `Bad`**: the first session accepts
nothing where `Bad` holds (V*'s list is not empty, `hS`), and elsewhere every session's own list is `zs`'s. -/
theorem accepted_offZs {I : Inner} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}
    {zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {Bad : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → Prop} (hS : 0 < S) {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {R : Reg} {ω : L.Ω} {t : List (Cm × I.Coin)} {R₂ : Reg₂}
    {τ : (j : Fin S) → Strategy ((zs R ω t R₂ j).game dg N)}
    {o : (j : Fin S) → Reg' × Finset (Fin (Vs j).nv) × FlockSoundness.Discharge.Exec.Transcript}
    (h : Accepted (offZs zs Bad) dg N k Rw R ω t R₂ τ o) : ¬ Bad R ω t R₂ ∧ Accepted zs dg N k Rw R ω t R₂ τ o := by
  have hb : ¬ Bad R ω t R₂ := fun hb =>
    not_accepts_dropOwn (zs R ω t R₂ ⟨0, hS⟩) hb (o ⟨0, hS⟩).2.2 (h.1 ⟨0, hS⟩).1
  refine ⟨hb, fun j => ⟨?_, (h.1 j).2⟩, h.2.1, h.2.2⟩
  have h1 : LiveAcceptsZKR (zs R ω t R₂ j).I (dropOwn (zs R ω t R₂ j) (Bad R ω t R₂)).own (o j) := (h.1 j).1
  rwa [own_dropOwn_neg _ hb] at h1

/-- **An outer proof `zs` accepts off `Bad` is one the sessions refusing `Bad` accept.** -/
theorem offZs_accepted {I : Inner} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}
    {zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {Bad : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → Prop} {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {R : Reg} {ω : L.Ω} {t : List (Cm × I.Coin)} {R₂ : Reg₂}
    {τ : (j : Fin S) → Strategy ((zs R ω t R₂ j).game dg N)}
    {o : (j : Fin S) → Reg' × Finset (Fin (Vs j).nv) × FlockSoundness.Discharge.Exec.Transcript}
    (hb : ¬ Bad R ω t R₂) (h : Accepted zs dg N k Rw R ω t R₂ τ o) :
    Accepted (offZs zs Bad) dg N k Rw R ω t R₂ τ o := by
  refine ⟨fun j => ⟨?_, (h.1 j).2⟩, h.2.1, h.2.2⟩
  show LiveAcceptsZKR (zs R ω t R₂ j).I (dropOwn (zs R ω t R₂ j) (Bad R ω t R₂)).own (o j)
  rw [own_dropOwn_neg _ hb]
  exact (h.1 j).1

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => holoInnerAt T CT st hp

section transport

variable {S : ℕ} {Vs : Fin S → VStmt}

theorem hLeafC3_offZs {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {Bad : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Prop} (hS : 0 < S) {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    (h : HLeafC3 zs dg N k Rw) : HLeafC3 (offZs zs Bad) dg N k Rw :=
  fun R ω t R₂ τ o hA => h R ω t R₂ τ o (accepted_offZs hS hA).2

theorem atSrc_offZs {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {Bad : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Prop} (hS : 0 < S) {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {P : List (Cm × (ℐ).Coin) → C3Run (holoAt st hp) → Prop} (h : AtSrc zs dg N k Rw src P) :
    AtSrc (offZs zs Bad) dg N k Rw src P :=
  fun R ω t R₂ τ o hA => h R ω t R₂ τ o (accepted_offZs hS hA).2

theorem atRep_offZs {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {Bad : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Prop} (hS : 0 < S) {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {sh : Flock.Shape} {rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData}
    {P : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → Prop}
    (h : AtRep zs dg N k Rw sh rep P) : AtRep (offZs zs Bad) dg N k Rw sh rep P :=
  fun R ω t R₂ τ o hA => h R ω t R₂ τ o (accepted_offZs hS hA).2

theorem algebraDecodes_offZs {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {Bad : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Prop} (hS : 0 < S) {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {sh : Flock.Shape} {rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData}
    (h : AlgebraDecodesOff zs dg N k Rw Bad sh rep) : AlgebraDecodes (I := ℐ) (offZs zs Bad) dg N k Rw sh rep :=
  fun R ω t R₂ τ o hA => h R ω t R₂ τ o (accepted_offZs hS hA).2 (accepted_offZs hS hA).1

theorem accChains_offZs {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {Bad : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Prop} (hS : 0 < S) {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {sh : Flock.Shape} {rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData}
    (h : AccChainsOff zs dg N k Rw Bad sh rep) : AccChains (I := ℐ) (offZs zs Bad) dg N k Rw sh rep :=
  fun R ω t R₂ τ o hA => h R ω t R₂ τ o (accepted_offZs hS hA).2 (accepted_offZs hS hA).1

theorem repsMsgsRead_offZs {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {Bad : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Prop} (hS : 0 < S) {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {own : Fin S → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}
    (h : RepsMsgsRead zs dg N k Rw src own sel) : RepsMsgsRead (offZs zs Bad) dg N k Rw src own sel :=
  fun i R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' =>
    h i R ω t R₂ τ o R' ω' t' R₂' τ' o' (accepted_offZs hS hA).2 (accepted_offZs hS hA').2

theorem partMsgsRead_offZs {prev part : C3Run (holoAt st hp) → List Step}
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {Bad : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Prop} (hS : 0 < S) {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {own : Fin S → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}
    (h : PartMsgsRead prev part zs dg N k Rw src own sel) :
    PartMsgsRead prev part (offZs zs Bad) dg N k Rw src own sel :=
  fun i R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' =>
    h i R ω t R₂ τ o R' ω' t' R₂' τ' o' (accepted_offZs hS hA).2 (accepted_offZs hS hA').2

theorem opensReadAt_offZs {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {Bad : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Prop} (hS : 0 < S) {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)} {s : ℕ}
    (h : OpensReadAt zs dg N k Rw src s) : OpensReadAt (offZs zs Bad) dg N k Rw src s :=
  fun R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' =>
    h R ω t R₂ τ o R' ω' t' R₂' τ' o' (accepted_offZs hS hA).2 (accepted_offZs hS hA').2

theorem rsRead_offZs {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {Bad : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Prop} (hS : 0 < S) {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {sh : Flock.Shape} {w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Array F128}
    {own : Fin S → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop} {c₀ : ℕ}
    (h : RsRead zs dg N k Rw sh w own sel c₀) : RsRead (offZs zs Bad) dg N k Rw sh w own sel c₀ :=
  fun i hi R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' =>
    h i hi R ω t R₂ τ o R' ω' t' R₂' τ' o' (accepted_offZs hS hA).2 (accepted_offZs hS hA').2

theorem mapsAt_offZs {stmt : Fin S → VStar.Stmt}
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {Bad : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Prop} (hS : 0 < S) {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {M : FlockC3.Reps.StackedMaps stmt Vs} {sh : Flock.Shape} {unit : String}
    {stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {c3 : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → RecCoins.C3}
    {w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Array F128}
    {kap : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Array (Array (Array F128))}
    {e0 : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → ℕ → ByteArray}
    {sb : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → ℕ → ℕ → ℕ → ByteArray}
    {e : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → ℕ → ℕ → ByteArray} {cap₀ : Cap Dig}
    (h : MapsAt zs dg N k Rw M sh unit stp c3 w kap e0 sb e cap₀) :
    MapsAt (offZs zs Bad) dg N k Rw M sh unit stp c3 w kap e0 sb e cap₀ :=
  fun R ω t R₂ τ o hA => h R ω t R₂ τ o (accepted_offZs hS hA).2

theorem srcHTie_offZs {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {Bad : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Prop} (hS : 0 < S) {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → Array Flock.F128}
    (h : FlockC3.HPort.SrcHTie zs dg N k Rw stp w) : FlockC3.HPort.SrcHTie (offZs zs Bad) dg N k Rw stp w :=
  fun R ω t R₂ τ o hA => h R ω t R₂ τ o (accepted_offZs hS hA).2

end transport

/-- **A fact at every proof the honest family at `offZs zs Bad` accepts holds off `Bad'` at the honest family at
`zs`**, where `Bad` holds only where `Bad'` does: the two families' runs and messages are the same, and a proof `zs`
accepts off `Bad` is one `offZs zs Bad` accepts (`offZs_accepted`). -/
theorem atAcceptedOff_offZs {xR : HoloReg GF128 Dig} {ish : RecVStar.Inner} {ss : Array VStar.Stmt}
    {hss : RecVStar.statements ish = .ok ss} {bind : Fin (vstarCount ish) → VStar.Bindings}
    {Vs : Fin (vstarCount ish) → VStmt} {hob : VStarParsesScoped Flock.Tags.circuit ish bind Vs}
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin (vstarCount ish)) → (Vs j).Outer Reg'}
    {Bad Bad' : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Prop}
    {own : Fin (vstarCount ish) → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}
    {bc : (j : Fin (vstarCount ish)) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → List UInt8}
    {dec : ℕ → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → (ℐ).Msg}
    {wr : List (Cm × (ℐ).Coin) → Lean.Json × String} {reg : Lean.Json} {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {P : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → Prop}
    (hB : ∀ R ω t R₂, ¬ Bad' R ω t R₂ → ¬ Bad R ω t R₂)
    (h : (C3Family.honest (xR := xR) ish hss bind hob (offZs zs Bad) own sel bc dec wr reg).AtAccepted dg N k Rw P) :
    (C3Family.honest (xR := xR) ish hss bind hob zs own sel bc dec wr reg).AtAcceptedOff Bad' dg N k Rw P :=
  fun R ω t R₂ τ o hA hb => h R ω t R₂ τ o (offZs_accepted (hB _ _ _ _ hb) hA)

end

end FlockC3.Compose
