import Proofs.Flock.C3.Compose.Decoder

/-!
# n7: `SrcOK`'s coin clause and its message types, stage by stage

`SrcOK`'s coin clause, the source's coin steps are the history's (`coinSteps`), follows from the whole history
(`HLeafC3`) and a coin fact per stage (`StageCoins`): the run's link points are the history's first coin, and each
stage's coins are the history's at that stage's rounds (`CoinsAt`), the stages' rounds following one another from
round 1 to the last coin (`srcCoins_of_stages`). `MsgTysC3` follows from each stage's message types (`TysAt`) and the
frame's three, `Cap Dig`, the first interaction's marker `Bool` and the openings, among `T` (`msgTysC3_of_stages`).
`srcOK_of_stages` puts them together: what is left of `SrcOK` is `MsgsReadC3` and the decode fixed point.

The ties (`SparseTie`, `OpenTie`) leave their stages' coins existential, so each tie's discharge adds its stage's
`CoinsAt`: that its unit's coins are the history's at its rounds, which reads the custody record's coins (`ChecksC3`'s
coin check, `RecordCustody`).
-/

namespace FlockC3.Compose

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockVBridge.Family (coinSteps)
open FlockVBridge.InnerAccepts (firstSteps)
open FlockC3.PubRows

variable {nT r : ℕ} {CT : Fin r → Type}

theorem coinSteps_append : ∀ (a b : List ((k : Fin r) → CT k)) (j : ℕ),
    coinSteps CT j (a ++ b) = coinSteps CT j a ++ coinSteps CT (j + a.length) b
  | [], b, j => by simp [coinSteps]
  | ρ :: a, b, j => by
      have e : j + (ρ :: a).length = j + 1 + a.length := by rw [List.length_cons]; omega
      rw [List.cons_append, coinSteps, coinSteps, coinSteps_append a b (j + 1), e, List.append_assoc]

/-- **A stage's coins are the history's at rounds `j` to `j + n - 1`.** -/
def CoinsAt (cs : List ((k : Fin r) → CT k)) (j n : ℕ) (l : List Step) : Prop :=
  l.filterMap Sum.getRight? = coinSteps CT j ((cs.drop j).take n)

theorem coinsAt_append {cs : List ((k : Fin r) → CT k)} {j n j' m : ℕ} {a b : List Step} (hj : j' = j + n)
    (hlen : j' ≤ cs.length) (ha : CoinsAt cs j n a) (hb : CoinsAt cs j' m b) : CoinsAt cs j (n + m) (a ++ b) := by
  subst hj
  unfold CoinsAt at ha hb ⊢
  have hl : ((cs.drop j).take n).length = n := by
    rw [List.length_take, List.length_drop, Nat.min_eq_left (show n ≤ cs.length - j by omega)]
  rw [FlockVBridge.Family.rights_append, ha, hb, List.take_add (l := cs.drop j) (i := n) (j := m),
    coinSteps_append ((cs.drop j).take n) (((cs.drop j).drop n).take m) j, hl, List.drop_drop]

theorem coinsAt_firstSteps {cs : List ((k : Fin r) → CT k)} {j n : ℕ} {ts : List Step} (h : CoinsAt cs j n ts) :
    CoinsAt cs j n (firstSteps ts) := by
  unfold CoinsAt at h ⊢
  rw [FlockVBridge.Family.firstSteps_rights, h]

/-- **A run's coin steps**: its link points, then its stages'. -/
theorem rights_runSteps {e : ExecHolo} (u : C3Run e) : (runSteps u).filterMap Sum.getRight? =
    ⟨Fin 2 → Fin e.cls.mPts → GF128, u.pts⟩ :: (stagesOf u).filterMap Sum.getRight? := by
  rw [runSteps, holoSteps_eq]
  show ⟨Fin 2 → Fin e.cls.mPts → GF128, u.pts⟩ :: (stagesOf u ++ _).filterMap Sum.getRight? = _
  rw [FlockVBridge.Family.rights_append]
  exact congrArg (_ :: ·) (List.append_nil _)

/-- **Each stage's coins are the history's at its rounds**: the link points are the history's first coin, and the
stages' coins are the history's at consecutive rounds from round 1, `w i` rounds for stage `i`, to the last coin. -/
def StageCoins {e : ExecHolo} (cs : List ((k : Fin r) → CT k)) (u : C3Run e) : Prop :=
  ∃ w : Fin 6 → ℕ, 1 + w 0 + w 1 + w 2 + w 3 + w 4 + w 5 = r ∧
    (coinSteps CT 0 cs).head? = some ⟨Fin 2 → Fin e.cls.mPts → GF128, u.pts⟩ ∧
    CoinsAt cs 1 (w 0) (u.ts 0) ∧ CoinsAt cs (1 + w 0) (w 1) (u.ts 1) ∧
    CoinsAt cs (1 + w 0 + w 1) (w 2) (u.sp 0) ∧ CoinsAt cs (1 + w 0 + w 1 + w 2) (w 3) (u.sp 1) ∧
    CoinsAt cs (1 + w 0 + w 1 + w 2 + w 3) (w 4) u.op ∧ CoinsAt cs (1 + w 0 + w 1 + w 2 + w 3 + w 4) (w 5) u.lg

/-- **The coin clause at a history with a coin per round**, from each stage's coins. -/
theorem coinClause_of_stageCoins {e : ExecHolo} {cs : List ((k : Fin r) → CT k)} {u : C3Run e} (hr : r ≤ cs.length)
    (h : StageCoins cs u) : (runSteps u).filterMap Sum.getRight? = coinSteps CT 0 cs := by
  obtain ⟨w, hw, hpts, h0, h1, h2, h3, h4, h5⟩ := h
  have hs : CoinsAt cs 1 (w 0 + w 1 + w 2 + w 3 + w 4 + w 5) (stagesOf u) :=
    coinsAt_append (by omega) (by omega)
      (coinsAt_append (by omega) (by omega)
        (coinsAt_append (by omega) (by omega)
          (coinsAt_append (by omega) (by omega)
            (coinsAt_append (by omega) (by omega) (coinsAt_firstSteps h0) h1) h2) h3) h4) h5
  unfold CoinsAt at hs
  cases cs with
  | nil => rw [List.length_nil] at hr; omega
  | cons c cs =>
      have hr0 : 0 < r := by omega
      have e0 : coinSteps CT 0 (c :: cs) = ⟨CT ⟨0, hr0⟩, c ⟨0, hr0⟩⟩ :: coinSteps CT 1 cs := by
        rw [coinSteps, dif_pos hr0, List.singleton_append, Nat.zero_add]
      rw [e0, List.head?_cons, Option.some.injEq] at hpts
      rw [rights_runSteps, hs, e0, hpts, List.drop_one, List.tail_cons,
        FlockVBridge.Family.coinSteps_take CT cs 1 (w 0 + w 1 + w 2 + w 3 + w 4 + w 5) (by omega)]

/-- **A stage's messages are typed by `T`.** -/
def TysAt (T : Fin nT → Type) (l : List Step) : Prop := ∀ m, Sum.inl m ∈ l → ∃ j, T j = m.1

theorem tysAt_append {T : Fin nT → Type} {a b : List Step} (ha : TysAt T a) (hb : TysAt T b) : TysAt T (a ++ b) :=
  fun m hm => (List.mem_append.1 hm).elim (ha m) (hb m)

theorem tysAt_firstSteps {T : Fin nT → Type} {ts : List Step} (hbool : ∃ j, T j = Bool) (h : TysAt T ts) :
    TysAt T (firstSteps ts) :=
  fun m hm => (FlockVBridge.Family.mem_firstSteps hm).elim (fun he => by subst he; exact hbool) (h m)

/-- **A run's messages are typed by `T`** where its stages' are and `T` holds the frame's three types. -/
theorem tysAt_runSteps {T : Fin nT → Type} {e : ExecHolo} {u : C3Run e} (hcap : ∃ j, T j = Cap Dig)
    (hbool : ∃ j, T j = Bool) (hos : ∃ j, T j = (Opens GF128 Dig × Salts (List UInt8)))
    (h0 : TysAt T (u.ts 0)) (h1 : TysAt T (u.ts 1)) (h2 : TysAt T (u.sp 0)) (h3 : TysAt T (u.sp 1))
    (h4 : TysAt T u.op) (h5 : TysAt T u.lg) : TysAt T (runSteps u) := by
  have hs : TysAt T (stagesOf u) :=
    tysAt_append (tysAt_append (tysAt_append (tysAt_append (tysAt_append (tysAt_firstSteps hbool h0) h1) h2) h3)
      h4) h5
  intro m hm
  rw [runSteps, holoSteps_eq] at hm
  rcases List.mem_cons.1 hm with hm | hm
  · cases hm
    exact hcap
  rcases List.mem_cons.1 hm with hm | hm
  · cases hm
  rcases List.mem_append.1 hm with hm | hm
  · exact hs m hm
  · rw [List.mem_singleton] at hm
    cases hm
    exact hos

section src

variable {T : Fin nT → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)] {st : Flock.Stmt} {hp : HoloParams}
  {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}

local notation "ℐ" => holoInnerAt T CT st hp

/-- **`SrcOK`'s coin clause** at a whole history (`HLeafC3`), from each stage's coins (`StageCoins`). -/
theorem srcCoins_of_stages {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    (hleaf : HLeafC3 zs dg N k Rw) (h : AtSrc zs dg N k Rw src fun t u => StageCoins (CT := CT) (t.map Prod.snd) u) :
    AtSrc zs dg N k Rw src (fun t u => (runSteps u).filterMap Sum.getRight? = coinSteps CT 0 (t.map Prod.snd)) :=
  fun R ω t R₂ τ o hacc => coinClause_of_stageCoins
    (Nat.le_trans (Nat.le_add_right r 1)
      (Nat.le_trans (hleaf R ω t R₂ τ o hacc) (Nat.le_of_eq (List.length_map _).symm)))
    (h R ω t R₂ τ o hacc)

/-- **`MsgTysC3`** from each stage's message types and the frame's three among `T`. -/
theorem msgTysC3_of_stages {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    (hcap : ∃ j, T j = Cap Dig) (hbool : ∃ j, T j = Bool) (hos : ∃ j, T j = (Opens GF128 Dig × Salts (List UInt8)))
    (h : AtSrc zs dg N k Rw src fun _ u => TysAt T (u.ts 0) ∧ TysAt T (u.ts 1) ∧ TysAt T (u.sp 0) ∧
      TysAt T (u.sp 1) ∧ TysAt T u.op ∧ TysAt T u.lg) : MsgTysC3 zs dg N k Rw src := by
  intro R ω t R₂ τ o hacc g hg m hm
  obtain ⟨h0, h1, h2, h3, h4, h5⟩ := h R ω t R₂ τ o hacc
  unfold c3Msgs at hg
  obtain ⟨x, hx, he⟩ := List.mem_filterMap.1 (FlockVBridge.Family.mem_cut _ _ hg m hm)
  obtain ⟨rfl⟩ : x = Sum.inl m := by
    cases x with
    | inl y => cases he; rfl
    | inr y => cases he
  exact tysAt_runSteps hcap hbool hos h0 h1 h2 h3 h4 h5 m hx

/-- **`SrcOK` from the stages**: the whole history, each stage's coins and message types, the frame's types among
`T`, the reads (`MsgsReadC3`) and the decode fixed point. -/
theorem srcOK_of_stages {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {own : Fin S → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}
    (hleaf : HLeafC3 zs dg N k Rw) (hcoins : AtSrc zs dg N k Rw src fun t u => StageCoins (CT := CT) (t.map Prod.snd) u)
    (hread : MsgsReadC3 zs dg N k Rw src own sel)
    (hcap : ∃ j, T j = Cap Dig) (hbool : ∃ j, T j = Bool) (hos : ∃ j, T j = (Opens GF128 Dig × Salts (List UInt8)))
    (htys : AtSrc zs dg N k Rw src fun _ u => TysAt T (u.ts 0) ∧ TysAt T (u.ts 1) ∧ TysAt T (u.sp 0) ∧
      TysAt T (u.sp 1) ∧ TysAt T u.op ∧ TysAt T u.lg)
    (hdec : AtSrc zs dg N k Rw src fun _ u => decodeRun (holoAt st hp) (runSteps u) = u) :
    SrcOK zs dg N k Rw src own sel :=
  ⟨hleaf, srcCoins_of_stages hleaf hcoins, hread, msgTysC3_of_stages hcap hbool hos htys, hdec⟩

end src

end

end FlockC3.Compose
