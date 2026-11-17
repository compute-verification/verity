module

public import Proofs.Flock.VBridge.Family.Rep
public import Proofs.Flock.VBridge.Family.Files
public import Proofs.Flock.VBridge.Family.Tape
public import Proofs.Flock.VBridge.Algebra.Schedule
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Proofs.Flock.Soundness.Defs
import Proofs.Flock.Soundness.Refine.Arith

/-! The family's joins, with the inner coins taken from the history once (`coinFiles`), so that
`rep₀ := repOf coinFiles` is not circular with `filesOf`. -/

@[expose] public section

namespace FlockVBridge.Family

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Recursion FlockSoundness.Discharge.PlaysCoin FlockSoundness.Refine
open FlockSoundness.Model FlockSoundness.Audit
open Flock FlockLevel3 FlockVBridge.Compiled FlockVBridge.InnerAccepts FlockVBridge.Stages
open Flock.RecAlgebra (verifierRows residualLins zeroRounds)

local notation "𝔽" => FlockLevel3.GF128

theorem claim_eq {m : ℕ} {a b : Flock.Claim} {c : Model.Claim 𝔽 m} (ha : ClaimRel m a c) (hb : ClaimRel m b c) :
    a = b := by
  have hv : a.value = b.value := fOf_injective (ha.value.symm.trans hb.value)
  have hs : a.skip = b.skip := by
    refine Array.ext (ha.skip_size.trans hb.skip_size.symm) fun i h₁ h₂ => ?_
    have hi : i < 64 := by have := ha.skip_size; omega
    have e := fOf_injective ((ha.skip ⟨i, hi⟩).symm.trans (hb.skip ⟨i, hi⟩))
    rwa [getElem!_pos a.skip i h₁, getElem!_pos b.skip i h₂] at e
  have hx : a.x = b.x := by
    refine Array.ext (ha.x_size.trans hb.x_size.symm) fun i h₁ h₂ => ?_
    have hi : i < m - 6 := by have := ha.x_size; omega
    have e := fOf_injective ((ha.x ⟨i, hi⟩).symm.trans (hb.x ⟨i, hi⟩))
    rwa [getElem!_pos a.x i h₁, getElem!_pos b.x i h₂] at e
  cases a
  cases b
  simp only at hv hs hx
  subst hv hs hx
  rfl

theorem extras_unique {st : Setup} {S : Model.Statement} {ptLocal mPts : ℕ} {pts : Fin 2 → Fin mPts → 𝔽}
    {P Q : Array (Array F128)} (hP : PointsOK st S ptLocal mPts pts P) (hQ : PointsOK st S ptLocal mPts pts Q) :
    st.extra P = st.extra Q := by
  obtain ⟨e, he, hre, -⟩ := hP
  obtain ⟨f, hf, hrf, -⟩ := hQ
  rw [he, hf]
  obtain ⟨hes, hec⟩ := hre e he
  obtain ⟨hfs, hfc⟩ := hrf f hf
  refine congrArg Except.ok (Array.ext (hes.trans hfs.symm) fun j h₁ h₂ => ?_)
  have hj : j < (extraClaims S ptLocal mPts pts).length := by omega
  have := claim_eq (hec j hj) (hfc j hj)
  rwa [getElem!_pos e j h₁, getElem!_pos f j h₂] at this

theorem TableSrc.extra_points (s : TableSrc) {ptLocal mPts : ℕ} (hset : SetupRealizes s.st s.S s.sch ptLocal mPts)
    (pts : Fin 2 → Fin mPts → 𝔽) {P : Array (Array F128)} (hP : PointsOK s.st s.S ptLocal mPts pts P) :
    s.st.extra P = .ok (s.extrasAt (s.points ptLocal mPts pts)) := by
  obtain ⟨e, he, -⟩ := s.points_ok hset pts
  rw [extras_unique hP (s.points_ok hset pts), he, s.extrasAt_of he]

theorem TableSrc.extrasAt_size (s : TableSrc) {ptLocal mPts : ℕ} (hset : SetupRealizes s.st s.S s.sch ptLocal mPts)
    (pts : Fin 2 → Fin mPts → 𝔽) : (s.extrasAt (s.points ptLocal mPts pts)).size = 2 * s.S.regions.length := by
  obtain ⟨e, he, hext, -⟩ := s.points_ok hset pts
  rw [s.extrasAt_of he, (hext e he).1, FlockSoundness.extraClaims_length]

theorem shape?_claims {m k c : ℕ} {sh : Shape} (h : Flock.RecResiduals.shape? m k c = some sh) : sh.claims = c := by
  unfold Flock.RecResiduals.shape? at h
  cases hf : Flock.fast100 m with
  | none => simp [hf] at h
  | some s =>
    by_cases hk : k < 6
    · simp [hf, hk] at h
    · by_cases hc : c < 2
      · simp [hf, hc] at h
      · simp [hf, hk, hc] at h
        subst h
        rfl

noncomputable def roundSizes (sh : Shape) : List ℕ :=
  Classical.epsilon fun ns : List ℕ => (∀ n ∈ ns, n ≠ 0) ∧ ∀ (rounds : Array (Array F128)) (extras : Array Claim)
    (comb : Array F128), (∃ x, verifierRows sh rounds extras comb = .ok x) ↔
      (rounds.filter (·.size ≠ 0)).toList.map Array.size = ns ∧ sh.claims = 2 + extras.size

theorem roundSizes_spec' {sh : Shape} {S : Flock.RecAlgebra.GfS} (hS : Flock.RecAlgebra.structureOf sh = .ok S) :
    (∀ n ∈ roundSizes sh, n ≠ 0) ∧ ∀ (rounds : Array (Array F128)) (extras : Array Claim) (comb : Array F128),
      (∃ x, verifierRows sh rounds extras comb = .ok x) ↔
        (rounds.filter (·.size ≠ 0)).toList.map Array.size = roundSizes sh ∧ sh.claims = 2 + extras.size :=
  Classical.epsilon_spec (p := fun ns : List ℕ => (∀ n ∈ ns, n ≠ 0) ∧ ∀ (rounds : Array (Array F128))
    (extras : Array Claim) (comb : Array F128), (∃ x, verifierRows sh rounds extras comb = .ok x) ↔
      (rounds.filter (·.size ≠ 0)).toList.map Array.size = ns ∧ sh.claims = 2 + extras.size)
    (Flock.RecAlgebra.verifierRows_schedule hS)

theorem roundSizes_ne {sh : Shape} {S : Flock.RecAlgebra.GfS} (hS : Flock.RecAlgebra.structureOf sh = .ok S) :
    ∀ n ∈ roundSizes sh, n ≠ 0 :=
  (roundSizes_spec' hS).1

theorem roundSizes_spec {sh : Shape} {S : Flock.RecAlgebra.GfS} (hS : Flock.RecAlgebra.structureOf sh = .ok S)
    {rounds : Array (Array F128)} (hne : Flock.RecAlgebra.NoEmpty rounds) (extras : Array Claim) (comb : Array F128) :
    (∃ x, verifierRows sh rounds extras comb = .ok x) ↔
      rounds.toList.map Array.size = roundSizes sh ∧ sh.claims = 2 + extras.size := by
  rw [(roundSizes_spec' hS).2, Flock.RecAlgebra.filter_noEmpty hne]

open Classical in
noncomputable def arrOf (n : ℕ) (m : Option Msg) : Array F128 :=
  if h : ∃ c : Array F128, c.size = n ∧ m = some (coinArr fOf c) then Classical.choose h else Array.replicate n 0

theorem arrOf_size (n : ℕ) (m : Option Msg) : (arrOf n m).size = n := by
  unfold arrOf
  split
  · rename_i h
    exact (Classical.choose_spec h).1
  · simp

noncomputable def coinsOf {nR : ℕ} (CT : Fin nR → Type) (sh : Shape) (cs : List ((k : Fin nR) → CT k)) (r : Fin 2) :
    Array (Array F128) :=
  ((roundSizes sh).mapIdx fun j n => arrOf n (coinSteps CT 0 cs)[1 + r.val * (roundSizes sh).length + j]?).toArray

theorem coinsOf_sizes {nR : ℕ} (CT : Fin nR → Type) (sh : Shape) (cs : List ((k : Fin nR) → CT k)) (r : Fin 2) :
    (coinsOf CT sh cs r).toList.map Array.size = roundSizes sh := by
  apply List.ext_getElem <;> simp [coinsOf, arrOf_size]

noncomputable def ptsC {nR : ℕ} (CT : Fin nR → Type) (mPts : ℕ) (cs : List ((k : Fin nR) → CT k)) :
    Fin 2 → Fin mPts → 𝔽 :=
  @Classical.epsilon _ ⟨fun _ _ => 0⟩ fun P => (coinSteps CT 0 cs).head? = some ⟨Fin 2 → Fin mPts → 𝔽, P⟩

theorem ptsC_eq {nR : ℕ} {CT : Fin nR → Type} {mPts : ℕ} {cs : List ((k : Fin nR) → CT k)} {P : Fin 2 → Fin mPts → 𝔽}
    (h : (coinSteps CT 0 cs).head? = some ⟨Fin 2 → Fin mPts → 𝔽, P⟩) : ptsC CT mPts cs = P := by
  have hs := @Classical.epsilon_spec _ (fun P => (coinSteps CT 0 cs).head? = some ⟨Fin 2 → Fin mPts → 𝔽, P⟩) ⟨P, h⟩
  have e := Option.some.inj (hs.symm.trans h)
  exact eq_of_heq (Sigma.mk.inj e).2

def coinRep (rounds : Array (Array F128)) : RepData := ⟨#[], #[], fun _ _ => 0, rounds, #[]⟩

theorem fin2_val (r : Fin 2) : fin2 r.val = r := by
  rcases r with ⟨_ | _ | n, h⟩
  · rfl
  · rfl
  · omega

theorem sizes_get {a b : Array (Array F128)} (hs : a.toList.map Array.size = b.toList.map Array.size) (i : ℕ) :
    (a[i]!).size = (b[i]!).size := by
  have hl : a.size = b.size := by simpa using congrArg List.length hs
  by_cases hi : i < a.size
  · rw [getElem!_pos a i hi, getElem!_pos b i (by omega)]
    have := List.getElem_of_eq hs (i := i) (by simpa using hi)
    simpa using this
  · rw [getElem!_neg a i hi, getElem!_neg b i (by omega)]

theorem coinsShaped_of {m kLog n L L₀ : ℕ} {a b : Array (Array F128)} (h : CoinsShaped m kLog n L₀ a) (hL : L ≤ L₀)
    (hs : a.toList.map Array.size = b.toList.map Array.size) : CoinsShaped m kLog n L b := by
  have hl : a.size = b.size := by simpa using congrArg List.length hs
  obtain ⟨hc, h0, h1, h2, hz, hlc, hr, hg⟩ := h
  refine ⟨by omega, ?_, ?_, ?_, fun i hi => ?_, fun j hj => ?_, fun k hk => ?_, ?_⟩
  · rw [← sizes_get hs]; exact h0
  · rw [← sizes_get hs]; exact h1
  · rw [← sizes_get hs]; exact h2
  · rw [← sizes_get hs]; exact hz i hi
  · rw [← sizes_get hs]; exact hlc j hj
  · rw [← sizes_get hs]; exact hr k hk
  · rw [← sizes_get hs]; exact hg

theorem zeroRounds_sizes (ns : List ℕ) : (zeroRounds ns).toList.map Array.size = ns := by
  apply List.ext_getElem <;> simp [zeroRounds]

/-- A fact of the inner statement's setup (`Setup.ofCircuit`), as `SetupRealizes` is. -/
def FoldsOk (st : Setup) (sh : Shape) : Prop :=
  ∀ (rounds : Array (Array F128)) (extras : Array Claim) lins info,
    rounds.toList.map Array.size = roundSizes sh →
    residualLins sh (Flock.RecAlgebra.RoundCoins.ofRounds rounds) extras = .ok (lins, info) →
    ∃ c, st.fold.folded info.lc.fold.alpha info.lc.fold.beta info.lc.fold.lw info.lc.fold.rhoIn info.lc.fold.ts =
      .ok c ∧ c.size = 64

/-- A fact of the two programs at the shape, as `ShapeIs` is. -/
def CoinsAgree (st : Setup) (sh : Shape) (sch : Flock.Schedule) (n : ℕ) : Prop :=
  ∃ L₀, CoinsShaped st.m st.fold.kLog n L₀ (zeroRounds (roundSizes sh)) ∧
    (∀ (T : F128) (cap : Array ByteArray) (lig : LigeritoProof), (chunks (ligFrames sch T cap lig)).1.length ≤ L₀) ∧
    ∀ c ∈ roundSizes sh, c ≠ 0

theorem residualLins_of_verifierRows {sh : Shape} {rounds : Array (Array F128)} {extras : Array Claim}
    {comb v : Array F128} {info : Flock.RecAlgebra.Info Flock.RecAlgebra.Lin} {kappa : Array (Array (Array F128))}
    (h : verifierRows sh rounds extras comb = .ok (v, info, kappa)) :
    ∃ lins, residualLins sh (Flock.RecAlgebra.RoundCoins.ofRounds rounds) extras = .ok (lins, info) := by
  unfold verifierRows at h
  cases hl : residualLins sh (Flock.RecAlgebra.RoundCoins.ofRounds rounds) extras with
  | error e => rw [hl] at h; cases h
  | ok q =>
    obtain ⟨lins, info'⟩ := q
    rw [hl] at h
    simp only [Bind.bind, Except.bind] at h
    cases hb : Flock.RecAlgebra.build sh lins with
    | error e => rw [hb] at h; cases h
    | ok b =>
      rw [hb] at h
      simp only at h
      split at h
      · cases h
      · cases h
        exact ⟨lins, rfl⟩

theorem combFolds_of {sh : Shape} {cls : HClass} (TF : TableFiles cls) (d : RepData)
    (hc : d.comb = combOf sh TF d.rounds) (he : d.extras = TF.extras)
    (hns : d.rounds.toList.map Array.size = roundSizes sh)
    (hfold : FoldsOk TF.st sh) : CombFolds TF.st sh d := by
  intro v info kappa h
  obtain ⟨lins, hl⟩ := residualLins_of_verifierRows h
  rw [he] at hl
  obtain ⟨c, hcf, hcs⟩ := hfold d.rounds TF.extras lins info hns hl
  rw [hc, hcf]
  unfold combOf
  rw [hl]
  simp only
  rw [hcf]
  simp [hcs]

section family

variable {nT nR : ℕ} {T : Fin nT → Type} {CT : Fin nR → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
variable {cls : HClass} {regs : List (Region cls.kLog (cls.m - cls.kLog))} {sch : Flock.Schedule}
variable {n : ℕ} {L : SamplingDesign n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => flockInnerCL T CT execArith H512 enc512 InnerAccepts.leafPlain cls regs (schedOf sch)

namespace VStarFamily

variable (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂) (spec : SessionSpec) (st : Reg → L.Ω → Setup)
    (cap : ((j : Fin F.stmts.size) → (F.Vs j).Bits) → Array ByteArray)
    (lig : ((j : Fin F.stmts.size) → (F.Vs j).Bits) → Fin 2 → LigeritoProof)

noncomputable def coinSrc (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin))
    (l : (j : Fin F.stmts.size) → (F.Vs j).Bits) : TableSrc :=
  ⟨spec, st R ω, sch, F.sh, (F.x R ω).toStatement regs, cap l, lig l,
    fun r => coinRep (coinsOf CT F.sh (t.map Prod.snd) r)⟩

noncomputable def coinFiles (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin))
    (l : (j : Fin F.stmts.size) → (F.Vs j).Bits) : TableFiles cls :=
  (F.coinSrc spec st cap lig R ω t l).filesAt cls (ptsC CT cls.mPts (t.map Prod.snd))

theorem repOf_coinFiles_rounds (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin))
    (l : (j : Fin F.stmts.size) → (F.Vs j).Bits) (r : Fin 2) :
    (F.repOf (F.coinFiles spec st cap lig) R ω t l r).rounds = coinsOf CT F.sh (t.map Prod.snd) r := by
  show (streamRounds (coinsOf CT F.sh (t.map Prod.snd) (fin2 r.val)) _).map (·.coins) = _
  rw [streamRounds_coins, fin2_val]

end VStarFamily

namespace VStarFamily

variable (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ)
variable (spec : SessionSpec) (st : Reg → L.Ω → Setup)
    (cap : ((j : Fin F.stmts.size) → (F.Vs j).Bits) → Array ByteArray)
    (lig : ((j : Fin F.stmts.size) → (F.Vs j).Bits) → Fin 2 → LigeritoProof)

def ClaimsIs : Prop := ∀ R ω, F.sh.claims = 2 + 2 * ((F.x R ω).toStatement regs).regions.length

/-- Discharged by the verifier's ClaimsAt check (`verity/rec-inner/v1`'s `regions`), with the setup's
`regs.length = st.regions.size`. -/
def ClaimsAt : Prop := F.vsh.claims = 2 + 2 * regs.length

theorem claimsIs_of (h : F.ClaimsAt) : F.ClaimsIs := fun _ _ => (shape?_claims F.hsh).trans h

theorem rep_rounds (hrep : F.rep₀ = F.repOf (F.coinFiles spec st cap lig)) (R : Reg) (ω : L.Ω)
    (t : List (Cm × (ℐ).Coin)) (l : (j : Fin F.stmts.size) → (F.Vs j).Bits) (r : Fin 2) :
    (F.rep R ω t l r).rounds = coinsOf CT F.sh (t.map Prod.snd) r := by
  show (F.rep₀ R ω t l r).rounds = _
  rw [hrep]
  exact F.repOf_coinFiles_rounds spec st cap lig R ω t l r

theorem rep_extras (hrep : F.rep₀ = F.repOf (F.coinFiles spec st cap lig)) (R : Reg) (ω : L.Ω)
    (t : List (Cm × (ℐ).Coin)) (l : (j : Fin F.stmts.size) → (F.Vs j).Bits) (r : Fin 2) :
    (F.rep R ω t l r).extras = (F.coinFiles spec st cap lig R ω t l).extras := by
  show (F.rep₀ R ω t l r).extras = _
  rw [hrep]
  rfl

theorem rep_comb (hrep : F.rep₀ = F.repOf (F.coinFiles spec st cap lig)) (R : Reg) (ω : L.Ω)
    (t : List (Cm × (ℐ).Coin)) (l : (j : Fin F.stmts.size) → (F.Vs j).Bits) (r : Fin 2) :
    (F.rep R ω t l r).comb = combOf F.sh (F.coinFiles spec st cap lig R ω t l) (F.rep R ω t l r).rounds := by
  have e : (F.rep R ω t l r).rounds = roundsOf (F.coinFiles spec st cap lig R ω t l) r := by
    show (F.rep₀ R ω t l r).rounds = _
    rw [hrep]
    rfl
  rw [e]
  show (F.rep₀ R ω t l r).comb = _
  rw [hrep]
  rfl

theorem extrasAre_of (hrep : F.rep₀ = F.repOf (F.coinFiles spec st cap lig))
    (hset : ∀ R ω, SetupRealizes (st R ω) ((F.x R ω).toStatement regs) sch cls.ptLocal cls.mPts)
    (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin)) (l : (j : Fin F.stmts.size) → (F.Vs j).Bits)
    (hpts : ptsOf (cls := cls) T CT ((F.wmsg l).zip (t.map Prod.snd)) = ptsC CT cls.mPts (t.map Prod.snd)) :
    (F.tableSrc spec st cap lig R ω t l).ExtrasAre T CT cls ((F.wmsg l).zip (t.map Prod.snd)) := by
  intro r P hP
  show (st R ω).extra P = .ok (F.rep R ω t l r).extras
  rw [F.rep_extras spec st cap lig hrep]
  rw [hpts] at hP
  exact (F.coinSrc spec st cap lig R ω t l).extra_points (hset R ω) _ hP

theorem hFit_of (hrep : F.rep₀ = F.repOf (F.coinFiles spec st cap lig)) {GS : Flock.RecAlgebra.GfS}
    {gs : Array (Array ℕ × Array ℕ × Array ℕ)} (hst : F.HStruct GS gs)
    (hset : ∀ R ω, SetupRealizes (st R ω) ((F.x R ω).toStatement regs) sch cls.ptLocal cls.mPts)
    (hcl : F.ClaimsIs) : F.HFit dg N k Rw := by
  intro R ω t R₂ τ _ _ r
  refine (roundSizes_spec hst.1 ?_ _ _).2 ⟨?_, ?_⟩
  · rw [F.rep_rounds spec st cap lig hrep]
    exact Flock.RecAlgebra.noEmpty_of_sizes (coinsOf_sizes CT F.sh _ r) (roundSizes_ne hst.1)
  · rw [F.rep_rounds spec st cap lig hrep]
    exact coinsOf_sizes CT F.sh _ r
  · rw [F.rep_extras spec st cap lig hrep, hcl R ω]
    show _ = 2 + ((F.coinSrc spec st cap lig R ω t _).extrasAt
      ((F.coinSrc spec st cap lig R ω t _).points cls.ptLocal cls.mPts _)).size
    rw [TableSrc.extrasAt_size _ (hset R ω)]
    rfl

theorem combFolds_at (hrep : F.rep₀ = F.repOf (F.coinFiles spec st cap lig))
    (hfold : ∀ R ω, FoldsOk (st R ω) F.sh) (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin))
    (l : (j : Fin F.stmts.size) → (F.Vs j).Bits) (r : Fin 2) : CombFolds (st R ω) F.sh (F.rep R ω t l r) :=
  combFolds_of (F.coinFiles spec st cap lig R ω t l) _ (F.rep_comb spec st cap lig hrep R ω t l r)
    (F.rep_extras spec st cap lig hrep R ω t l r)
    (by rw [F.rep_rounds spec st cap lig hrep]; exact coinsOf_sizes CT F.sh _ r) (hfold R ω)

theorem coins_of (hrep : F.rep₀ = F.repOf (F.coinFiles spec st cap lig))
    (hco : ∀ R ω, CoinsAgree (st R ω) F.sh sch (2 * ((F.x R ω).toStatement regs).regions.length))
    (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin)) (l : (j : Fin F.stmts.size) → (F.Vs j).Bits) :
    (F.tableSrc spec st cap lig R ω t l).Coins := by
  intro r
  obtain ⟨L₀, hL₀, hle, hnz⟩ := hco R ω
  refine ⟨coinsShaped_of hL₀ (hle _ _ _) ?_, ?_⟩
  · show _ = (F.rep R ω t l r).rounds.toList.map Array.size
    rw [zeroRounds_sizes, F.rep_rounds spec st cap lig hrep, coinsOf_sizes]
  · show Flock.RecAlgebra.NoEmpty (F.rep R ω t l r).rounds
    rw [F.rep_rounds spec st cap lig hrep]
    exact Flock.RecAlgebra.noEmpty_of_sizes (coinsOf_sizes CT F.sh _ r) hnz

end VStarFamily

end family

end FlockVBridge.Family
