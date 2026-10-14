import Proofs.Flock.C3.Reps.WRead

/-!
# C3's rep decoder with its reads: `RepMaps.withReads`

`RepMaps.withReads` keeps a decoder's unit maps, extras, opened rows and running sums and reads the rest: rep `ρ`'s
rounds are the coins the verifier holds of stream `s ρ` (`heldRounds`), `rep₀`'s `w` is read off the layers (`wRead`)
beside a comb of 64 elements, and `α` is the Ligerito run's at the rounds and extras (`infoAlpha`, `verifierRows`'
info). At it:

* `withReads_roundsHeld`: `RepRoundsHeld`, by definition (`repRoundsHeld_of_rounds`);
* `withReads_alpha`: `α` is the verifier rows' (`infoAlpha_of_rows`);
* `withReads_wDecodes`, `withReads_wColsAt`: `WDecodes` (`wDecodes_of_read`, each session's `upv > 0` from an outer
  proof, `upv_of_outer`), and `WColsAt` at the sent keys from a key-by-key check that `w` holds them (`wKey`,
  `wKeys_sentK`) and the reads file at `wLoc`'s columns (`WSel`);
* `withReads_algebraDecodes`: G items 1–3, from each part's `ResZeroAtLH` at copy 0 at one `w` per rep
  (`resZeroAtLH_wRead`, the layers agreeing with their registrants at an accepted proof), `w`'s size S's
  (`wRead_size`, `nwOf_eq`) and the verifier rows at the reads.
-/

namespace FlockC3.Reps

noncomputable section

open Flock FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockVBridge.OfVStar (ResZeroAtLH)

/-- **Level `ℓ`'s α at a rep's rounds and extras**: the Ligerito run's, in the coin-derived points `residualLins`
returns there (`verifierRows`' info, `infoAlpha_of_rows`); none where it refuses. -/
def infoAlpha (sh : Flock.Shape) (rounds : Array (Array F128)) (extras : Array Flock.Claim) (ℓ : ℕ) : Array F128 :=
  match Flock.RecAlgebra.residualLins sh (Flock.RecAlgebra.Coins.ofRounds rounds) extras with
  | .ok (_, info) => (info.lg.conss[ℓ]!).alpha
  | .error _ => #[]

/-- **`infoAlpha` is the α of the points `verifierRows` returns** at the same rounds and extras. -/
theorem infoAlpha_of_rows {sh : Flock.Shape} {rounds : Array (Array F128)} {extras : Array Flock.Claim}
    {comb v : Array F128} {info : Flock.RecAlgebra.Info Flock.RecAlgebra.Lin} {kappa : Array (Array (Array F128))}
    (h : Flock.RecAlgebra.verifierRows sh rounds extras comb = .ok (v, info, kappa)) (ℓ : ℕ) :
    infoAlpha sh rounds extras ℓ = (info.lg.conss[ℓ]!).alpha := by
  unfold Flock.RecAlgebra.verifierRows at h
  unfold infoAlpha
  cases hl : Flock.RecAlgebra.residualLins sh (Flock.RecAlgebra.Coins.ofRounds rounds) extras with
  | error _ => rw [hl] at h; cases h
  | ok q =>
    obtain ⟨lins, _⟩ := q
    rw [hl] at h
    simp only [Bind.bind, Except.bind] at h
    cases hb : Flock.RecAlgebra.build sh lins with
    | error _ => rw [hb] at h; cases h
    | ok b =>
      rw [hb] at h
      simp only at h
      split at h
      · cases h
      · cases h
        rfl

/-- **Whether `w` holds a key** (`WKeys` at one key, a computable check): a consistency sum, or a key `w` names
(`wi`) that is neither a ring switch's transpose nor a comb element. -/
def wKey (sh : Flock.Shape) : Flock.RecAlgebra.Key → Bool
  | .cons _ _ => true
  | .rsT _ _ => false
  | .comb _ => false
  | x => (Flock.RecAlgebra.wi sh x).isSome

theorem wKey_spec {sh : Flock.Shape} {x : Flock.RecAlgebra.Key} (h : wKey sh x = true)
    (hnc : ∀ ℓ l, x ≠ .cons ℓ l) :
    (∃ i, Flock.RecAlgebra.wi sh x = some i) ∧ (∀ k u, x ≠ .rsT k u) ∧ ∀ l, x ≠ .comb l := by
  cases x with
  | cons ℓ l => exact (hnc ℓ l rfl).elim
  | rsT => cases h
  | comb => cases h
  | _ => exact ⟨Option.isSome_iff_exists.1 h, fun _ _ e => (by cases e), fun _ e => by cases e⟩

/-- **`WKeys` at the sent keys**, from `wKey` at each message's keys. -/
theorem wKeys_sentK {sh : Flock.Shape} {σ : List FlockSoundness.Discharge.Exec.Kind} {c₀ : ℕ}
    {keys : ℕ → List Flock.RecAlgebra.Key} (h : ∀ m, (keys m).all (wKey sh) = true) :
    WKeys sh (sentK σ c₀ keys) := by
  rintro _ _ x ⟨m, -, hx, -⟩ hnc
  exact wKey_spec (List.all_eq_true.1 (h m) x hx) hnc

namespace RepMaps

/-- **The decoder with its reads**: `M`'s unit maps, extras, opened rows and running sums; rep `ρ`'s rounds the coins
the verifier holds of stream `s ρ` in the record `held (wr t)` (`heldRounds`), `rep₀`'s `w` read off the layers
(`wRead`) and its comb `comb`, and `α` the Ligerito run's at the rounds and extras (`infoAlpha`). -/
def withReads {S : ℕ} {stmt : Fin S → VStar.Stmt} {Vs : Fin S → VStmt} {sh : Flock.Shape} {Reg Ω C : Type}
    (M : RepMaps stmt Vs sh Reg Ω C) (wr : List C → Lean.Json × String)
    (held : Lean.Json × String → Except String Flock.RecVStar.InnerCommits) (s : Fin 2 → String)
    (comb : ((j : Fin S) → (Vs j).Bits) → Fin 2 → Fin 64 → F128) : RepMaps stmt Vs sh Reg Ω C :=
  { M with
    rounds := fun _ _ t ρ => heldRounds (held (wr t)) (s ρ),
    rep₀ := fun Lb ρ =>
      { w := wRead Vs sh M.sess M.un Lb ρ, comb := Array.ofFn (comb Lb ρ), cons := fun _ _ => 0, rounds := #[],
        extras := #[] },
    alpha := fun R ω t ρ => infoAlpha sh (heldRounds (held (wr t)) (s ρ)) (M.extras R ω t ρ) }

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}
  {stmt : Fin S → VStar.Stmt} {sh : Flock.Shape}
  {zs : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
  {dg : ByteArray → ByteArray} {N k Rw : ℕ}
  {wr : List (Cm × (holoInnerAt T CT st hp).Coin) → Lean.Json × String}
  {held : Lean.Json × String → Except String Flock.RecVStar.InnerCommits} {s : Fin 2 → String}
  {comb : ((j : Fin S) → (Vs j).Bits) → Fin 2 → Fin 64 → F128}

local notation "ℐ" => holoInnerAt T CT st hp

/-- **The reads' rounds are the record's** (`RepRoundsHeld`, round `q` at record round `q`), by definition. -/
theorem withReads_roundsHeld (M : RepMaps stmt Vs sh Reg L.Ω (Cm × (ℐ).Coin)) :
    RepRoundsHeld zs dg N k Rw sh (M.withReads wr held s comb).rep wr held s fun _ q => q :=
  repRoundsHeld_of_rounds fun _ _ _ _ _ => rfl

/-- **The reads' α is the verifier rows'** at the rep's rounds and extras. -/
theorem withReads_alpha (M : RepMaps stmt Vs sh Reg L.Ω (Cm × (ℐ).Coin)) {R : Reg} {ω : L.Ω}
    {t : List (Cm × (ℐ).Coin)} {ρ : Fin 2} {cb v : Array F128} {info : Flock.RecAlgebra.Info Flock.RecAlgebra.Lin}
    {kappa : Array (Array (Array F128))}
    (h : Flock.RecAlgebra.verifierRows sh (heldRounds (held (wr t)) (s ρ)) (M.extras R ω t ρ) cb =
      .ok (v, info, kappa)) (ℓ : ℕ) :
    (M.withReads wr held s comb).alpha R ω t ρ ℓ = (info.lg.conss[ℓ]!).alpha :=
  infoAlpha_of_rows h ℓ

/-- **`WDecodes` at the reads** (`wDecodes_of_read`): each session's `upv > 0` from its outer proof `zo`. -/
theorem withReads_wDecodes (M : RepMaps stmt Vs sh Reg L.Ω (Cm × (ℐ).Coin)) (zo : (j : Fin S) → (Vs j).Outer Reg') :
    WDecodes (M.withReads wr held s comb) (fun _ => wSes Vs sh M.sess) (wUn Vs sh M.sess M.un)
      (fun _ => wCols Vs sh M.sess) :=
  wDecodes_of_read (M.withReads wr held s comb) (fun j => upv_of_outer (zo j)) fun _ _ => rfl

/-- **`WColsAt` at the reads and the sent keys** (`wColsAt_of_w`): `w` holds every sent key (`wKeys_sentK`), the reads
decode `w` (`withReads_wDecodes`), and the reads file has each sent key's ports at `wLoc`'s columns (`WSel`). -/
theorem withReads_wColsAt (M : RepMaps stmt Vs sh Reg L.Ω (Cm × (ℐ).Coin)) (zo : (j : Fin S) → (Vs j).Outer Reg')
    {own : Fin S → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}
    {σ : List FlockSoundness.Discharge.Exec.Kind} {c₀ : ℕ} {keys : ℕ → List Flock.RecAlgebra.Key}
    (hk : ∀ m, (keys m).all (wKey sh) = true)
    (hs : WSel sh own sel (fun _ => wSes Vs sh M.sess) (wUn Vs sh M.sess M.un) (fun _ => wCols Vs sh M.sess)
      (sentK σ c₀ keys)) :
    WColsAt own sel (M.withReads wr held s comb) (sentK σ c₀ keys) :=
  wColsAt_of_w (wKeys_sentK hk) (M.withReads_wDecodes zo) hs

/-- **G items 1–3 at the reads** (`RepMaps.algebraDecodes`): `w` has S's size (`wRead_size`, `nwOf_eq`) and the comb 64
elements, and each part's `hres` holds at `wRead`'s `w` where at each accepted outer proof each rep's parts hold at
copy 0 at one `w` (`resZeroAtLH_wRead`: an element a part reads is read whole at some part, `ResReadCol`, and each
layer agrees with its registrant, `Good`). -/
theorem withReads_algebraDecodes (M : RepMaps stmt Vs sh Reg L.Ω (Cm × (ℐ).Coin)) {GS : Flock.RecAlgebra.GfS}
    {gs : Array (Array ℕ × Array ℕ × Array ℕ)}
    (hst : Flock.RecAlgebra.structureOf sh = .ok GS ∧
      Flock.RecResiduals.groups GS (Flock.RecResiduals.portSizes sh) = .ok gs)
    (hrc : ResReadCol Vs sh M.sess)
    (hvr : ∀ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ ρ : Fin 2, ∃ v info kappa,
      Flock.RecAlgebra.verifierRows sh (heldRounds (held (wr t)) (s ρ)) (M.extras R ω t ρ)
          (Array.ofFn (comb (layers zs dg N k Rw R ω t R₂ τ) ρ)) = .ok (v, info, kappa) ∧ v.size = GS.nv)
    (hw : ∀ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ ρ : Fin 2, ∃ w : ℕ → F128,
      ∀ p < gs.size, ResZeroAtLH (zs R ω t R₂ (M.sess p)) (layers zs dg N k Rw R ω t R₂ τ (M.sess p)) (M.un p ρ) 0
        sh gs p w (heldRounds (held (wr t)) (s ρ)) (M.extras R ω t ρ)
        (Array.ofFn (comb (layers zs dg N k Rw R ω t R₂ τ) ρ))) :
    AlgebraDecodes zs dg N k Rw sh (M.withReads wr held s comb).rep :=
  (M.withReads wr held s comb).algebraDecodes zs dg N k Rw hst hvr
    (fun R ω t R₂ τ _ _ ρ =>
      ⟨(wRead_size (Vs := Vs) (sh := sh) (sess := M.sess) (un := M.un) (layers zs dg N k Rw R ω t R₂ τ) ρ).trans
          (FlockVBridge.Family.nwOf_eq hst.1),
        Array.size_ofFn (f := comb (layers zs dg N k Rw R ω t R₂ τ) ρ)⟩)
    fun R ω t R₂ τ o hA ρ _ hp => by
      obtain ⟨w, hw⟩ := hw R ω t R₂ τ o hA ρ
      have hAg : ∀ j, (zs R ω t R₂ j).RegAgree (layers zs dg N k Rw R ω t R₂ τ j) := fun j => (hA.1 j).2.2
      exact resZeroAtLH_wRead hrc hAg hst.1 hst.2 ρ hw hp

end RepMaps

end

end FlockC3.Reps
