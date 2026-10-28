import Proofs.Flock.VBridge.Family.Rep
import Proofs.Flock.VBridge.Family.Openings
import Proofs.Flock.VBridge.Family.Files
import Proofs.Flock.VBridge.Family.Commits
import Proofs.Flock.VBridge.Family.Tape
import Proofs.Flock.VBridge.Algebra.Schedule

/-!
# The lanes' joins

Ha's `repOf tf` reads each rep's rounds and extra claims from a table's files `tf`, and hf's `filesOf` streams issue
`F.rep`'s coins, so `files = filesOf …` with `rep₀ = repOf files` is circular. The inner coins are the history's, so
they are taken from it once:

* `coinsOf`: rep `r`'s rounds are the coin steps of the history's coins (`coinSteps`) after the link points and `r`
  reps' rounds, each read as the array of the size `verifierRows` takes there (`roundSizes`, `verifierRows_schedule`);
* `coinFiles`: the table files of a source whose reps carry only those coins (`coinRep`), at the same setup and
  statement as `filesOf`'s and at the link points of the history's first coin (`ptsC`), so they read no message and
  their extra claims are `filesOf`'s wherever the inner transcript's link points are those; `rep₀ := repOf coinFiles`.

At a family whose `rep₀` is that, the joins between the lanes hold:

* `extrasAre_of` (hf's `ExtrasAre`): any well-formed link point arrays give the same extra claims (`extras_unique`:
  `ClaimRel` determines a claim);
* `hFit_of` (ha's `HFit`): the rounds have `verifierRows`' sizes by construction, and the extra claims number
  `2 |regions|` (`ClaimsIs`);
* `combFolds_of` (hf's `CombFolds`): `combOf` is the lincheck's fold at the rep's coins wherever that fold returns 64
  elements (`FoldsOk`);
* `coins_of` (hf's `Coins`): `CoinsShaped` reads only the rounds' sizes, which are `verifierRows`' (`CoinsAgree`).
-/

namespace FlockVBridge.Family

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Recursion FlockSoundness.Discharge.PlaysCoin FlockSoundness.Refine
open FlockSoundness.Model FlockSoundness.Audit
open Flock FlockLevel3 FlockVBridge.Compiled FlockVBridge.InnerAccepts FlockVBridge.Stages
open Flock.RecAlgebra (verifierRows residualLins zeroRounds)

local notation "𝔽" => FlockLevel3.GF128

/-! ## The extra claims are the setup's -/

/-- **`ClaimRel` determines a claim**: its value, its 64 skip weights and its `m − 6` coordinates are the model
claim's, and `fOf` is injective. -/
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

/-- **Well-formed link point arrays give one set of extra claims**: both realize the model's claims at the link
points, claim by claim (`claim_eq`). -/
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

/-- **Every well-formed link point arrays give a table source's extra claims** at those link points. -/
theorem TableSrc.extra_points (s : TableSrc) {ptLocal mPts : ℕ} (hset : SetupRealizes s.st s.S s.sch ptLocal mPts)
    (pts : Fin 2 → Fin mPts → 𝔽) {P : Array (Array F128)} (hP : PointsOK s.st s.S ptLocal mPts pts P) :
    s.st.extra P = .ok (s.extrasAt (s.points ptLocal mPts pts)) := by
  obtain ⟨e, he, -⟩ := s.points_ok hset pts
  rw [extras_unique hP (s.points_ok hset pts), he, s.extrasAt_of he]

/-- A table source's extra claims number `2 |regions|`. -/
theorem TableSrc.extrasAt_size (s : TableSrc) {ptLocal mPts : ℕ} (hset : SetupRealizes s.st s.S s.sch ptLocal mPts)
    (pts : Fin 2 → Fin mPts → 𝔽) : (s.extrasAt (s.points ptLocal mPts pts)).size = 2 * s.S.regions.length := by
  obtain ⟨e, he, hext, -⟩ := s.points_ok hset pts
  rw [s.extrasAt_of he, (hext e he).1, FlockSoundness.extraClaims_length]

/-- `shape?` keeps the claims it is given. -/
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

/-! ## The inner coins, from the history -/

/-- **The rounds' sizes `verifierRows` returns at**, at a shape (`verifierRows_schedule`), none of them 0. -/
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

/-- `roundSizes`' property at a record with no round of no coins. -/
theorem roundSizes_spec {sh : Shape} {S : Flock.RecAlgebra.GfS} (hS : Flock.RecAlgebra.structureOf sh = .ok S)
    {rounds : Array (Array F128)} (hne : Flock.RecAlgebra.NoEmpty rounds) (extras : Array Claim) (comb : Array F128) :
    (∃ x, verifierRows sh rounds extras comb = .ok x) ↔
      rounds.toList.map Array.size = roundSizes sh ∧ sh.claims = 2 + extras.size := by
  rw [(roundSizes_spec' hS).2, Flock.RecAlgebra.filter_noEmpty hne]

open Classical in
/-- **A coin step as an array of `n` words**: the array of `n` words that `coinArr` reads as the step, zeros where there
is none. -/
noncomputable def arrOf (n : ℕ) (m : Option Msg) : Array F128 :=
  if h : ∃ c : Array F128, c.size = n ∧ m = some (coinArr fOf c) then Classical.choose h else Array.replicate n 0

theorem arrOf_size (n : ℕ) (m : Option Msg) : (arrOf n m).size = n := by
  unfold arrOf
  split
  · rename_i h
    exact (Classical.choose_spec h).1
  · simp

/-- **Rep `r`'s coins in the inner transcript's coins `cs`**: its round `j` is the coin step after the link points and
`r` reps' rounds (`coinSteps`; `tableTape` takes rep 0's steps, then rep 1's), as the array of the size
`verifierRows` takes there (`roundSizes`). -/
noncomputable def coinsOf {nR : ℕ} (CT : Fin nR → Type) (sh : Shape) (cs : List ((k : Fin nR) → CT k)) (r : Fin 2) :
    Array (Array F128) :=
  ((roundSizes sh).mapIdx fun j n => arrOf n (coinSteps CT 0 cs)[1 + r.val * (roundSizes sh).length + j]?).toArray

theorem coinsOf_sizes {nR : ℕ} (CT : Fin nR → Type) (sh : Shape) (cs : List ((k : Fin nR) → CT k)) (r : Fin 2) :
    (coinsOf CT sh cs r).toList.map Array.size = roundSizes sh := by
  apply List.ext_getElem <;> simp [coinsOf, arrOf_size]

/-- **The link points of the coins `cs`**: their first coin step (`coinSteps`), where it is one. -/
noncomputable def ptsC {nR : ℕ} (CT : Fin nR → Type) (mPts : ℕ) (cs : List ((k : Fin nR) → CT k)) :
    Fin 2 → Fin mPts → 𝔽 :=
  @Classical.epsilon _ ⟨fun _ _ => 0⟩ fun P => (coinSteps CT 0 cs).head? = some ⟨Fin 2 → Fin mPts → 𝔽, P⟩

theorem ptsC_eq {nR : ℕ} {CT : Fin nR → Type} {mPts : ℕ} {cs : List ((k : Fin nR) → CT k)} {P : Fin 2 → Fin mPts → 𝔽}
    (h : (coinSteps CT 0 cs).head? = some ⟨Fin 2 → Fin mPts → 𝔽, P⟩) : ptsC CT mPts cs = P := by
  have hs := @Classical.epsilon_spec _ (fun P => (coinSteps CT 0 cs).head? = some ⟨Fin 2 → Fin mPts → 𝔽, P⟩) ⟨P, h⟩
  have e := Option.some.inj (hs.symm.trans h)
  exact eq_of_heq (Sigma.mk.inj e).2

/-- A rep that carries only its coins. -/
def coinRep (rounds : Array (Array F128)) : RepData := ⟨#[], #[], fun _ _ => 0, rounds, #[]⟩

theorem fin2_val (r : Fin 2) : fin2 r.val = r := by
  rcases r with ⟨_ | _ | n, h⟩
  · rfl
  · rfl
  · omega

/-! ## `CoinsShaped` reads only the rounds' sizes -/

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

/-! ## The setup's facts the joins take -/

/-- **The lincheck's fold returns 64 elements at the coins `verifierRows` takes** (Piop.lean:143-146): at every record
of `verifierRows`' sizes and every extra claims where the algebra's lincheck has a fold, the setup's `folded` at it
returns 64 elements. A fact of the inner statement's setup (`Setup.ofCircuit`: `Stmt.folded`, Statement.lean:321-340,
whose `foldedPartial` starts from 64 zeros and whose `fold` branch halves `2^k_log` elements `k_log − 6` times), as
`SetupRealizes` is. -/
def FoldsOk (st : Setup) (sh : Shape) : Prop :=
  ∀ (rounds : Array (Array F128)) (extras : Array Claim) lins info,
    rounds.toList.map Array.size = roundSizes sh →
    residualLins sh (Flock.RecAlgebra.CoinSource.ofRounds rounds) extras = .ok (lins, info) →
    ∃ c, st.fold.folded info.lc.fold.alpha info.lc.fold.beta info.lc.fold.lw info.lc.fold.rhoIn info.lc.fold.ts =
      .ok c ∧ c.size = 64

/-- **`verifierRows`' rounds are the ones `verifyRep` draws** (Verify.lean:43-48, RecAlgebra.lean:824-829): the sizes
`verifierRows` takes at `sh`, none of them 0, have the counts `run_of_frames` draws (`CoinsShaped`), for `n` extra
claims and at least as many Ligerito rounds as any proof's frames (`ligFrames`) chunk into. A fact of the two programs
at the shape, as `ShapeIs` is. -/
def CoinsAgree (st : Setup) (sh : Shape) (sch : Flock.Schedule) (n : ℕ) : Prop :=
  ∃ L₀, CoinsShaped st.m st.fold.kLog n L₀ (zeroRounds (roundSizes sh)) ∧
    (∀ (T : F128) (cap : Array ByteArray) (lig : LigeritoProof), (chunks (ligFrames sch T cap lig)).1.length ≤ L₀) ∧
    ∀ c ∈ roundSizes sh, c ≠ 0

/-- `verifierRows` at a lincheck fold returns the fold `residualLins` gives. -/
theorem residualLins_of_verifierRows {sh : Shape} {rounds : Array (Array F128)} {extras : Array Claim}
    {comb v : Array F128} {info : Flock.RecAlgebra.Info Flock.RecAlgebra.Lin} {kappa : Array (Array (Array F128))}
    (h : verifierRows sh rounds extras comb = .ok (v, info, kappa)) :
    ∃ lins, residualLins sh (Flock.RecAlgebra.CoinSource.ofRounds rounds) extras = .ok (lins, info) := by
  unfold verifierRows at h
  cases hl : residualLins sh (Flock.RecAlgebra.CoinSource.ofRounds rounds) extras with
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

/-- **`CombFolds` at `combOf`**: a rep whose comb is `combOf` at its coins and a table's extra claims, its rounds of
`verifierRows`' sizes, has the lincheck's fold as its comb where the fold returns 64 elements (`FoldsOk`). -/
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
variable {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => flockInnerCL T CT execArith H512 enc512 InnerAccepts.leafPlain cls regs (schedOf sch)

namespace VStarFamily

variable (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂) (spec : SessionSpec) (st : Reg → L.Ω → Setup)
    (cap : ((j : Fin F.stmts.size) → (F.Vs j).Bits) → Array ByteArray)
    (lig : ((j : Fin F.stmts.size) → (F.Vs j).Bits) → Fin 2 → LigeritoProof)

/-- **The coins' source** at a history and V*'s layers: `tableSrc`'s spec, setup, schedule, shape, statement, cap and
openings, its reps carrying only the history's coins (`coinsOf`). It reads none of the fields `filesOf` and `repOf`
define. -/
noncomputable def coinSrc (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin))
    (l : (j : Fin F.stmts.size) → (F.Vs j).Bits) : TableSrc :=
  ⟨spec, st R ω, sch, F.sh, (F.x R ω).toStatement regs, cap l, lig l,
    fun r => coinRep (coinsOf CT F.sh (t.map Prod.snd) r)⟩

/-- **The coins' table files**: `coinSrc`'s, at the link points of the history's first coin (`ptsC`). Their streams
issue `coinsOf`; they read no message, so `rep₀ := repOf coinFiles` comes before the messages (`wmsg`) and the
decoder that reads them. -/
noncomputable def coinFiles (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin))
    (l : (j : Fin F.stmts.size) → (F.Vs j).Bits) : TableFiles cls :=
  (F.coinSrc spec st cap lig R ω t l).filesAt cls (ptsC CT cls.mPts (t.map Prod.snd))

theorem repOf_coinFiles_rounds (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin))
    (l : (j : Fin F.stmts.size) → (F.Vs j).Bits) (r : Fin 2) :
    (F.repOf (F.coinFiles spec st cap lig) R ω t l r).rounds = coinsOf CT F.sh (t.map Prod.snd) r := by
  show (streamRounds (coinsOf CT F.sh (t.map Prod.snd) (fin2 r.val)) _).map (·.coins) = _
  rw [streamRounds_coins, fin2_val]

end VStarFamily

/-! ## The joins at a family whose `rep₀` is `repOf coinFiles` -/

namespace VStarFamily

variable (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ)
variable (spec : SessionSpec) (st : Reg → L.Ω → Setup)
    (cap : ((j : Fin F.stmts.size) → (F.Vs j).Bits) → Array ByteArray)
    (lig : ((j : Fin F.stmts.size) → (F.Vs j).Bits) → Fin 2 → LigeritoProof)

/-- The verifier's shape counts the statement's claims: `2` and two per region. -/
def ClaimsIs : Prop := ∀ R ω, F.sh.claims = 2 + 2 * ((F.x R ω).toStatement regs).regions.length

/-- Obligation: discharged by the verifier's ClaimsAt check (`verity/rec-inner/v1`'s `regions`,
`cursor/rec-vstar-checks-741b`), with the setup's `regs.length = st.regions.size`.
**V*'s shape counts the inner statement's claims** (the verifier's gap `vsh`, with `ShapeIs`): `vsh`'s claims are
`2` and two per region of the inner session's `regs` (`rec_vstage.shape`'s claims). -/
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

/-- **hf's `ExtrasAre`**: each rep's extra claims are the coins' files', which are the setup's at every well-formed
link point arrays (`TableSrc.extra_points`). -/
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

/-- **ha's `HFit`**: each rep's rounds have `verifierRows`' sizes (`coinsOf_sizes`), and its extra claims number the
shape's (`ClaimsIs`, `TableSrc.extrasAt_size`), so `verifierRows` returns (`roundSizes_spec`). -/
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

/-- **hf's `CombFolds`**: each rep's comb is `combOf` at its coins and the coins' files' extra claims (`rep_comb`), of
`verifierRows`' sizes, so it is the lincheck's fold where that returns 64 elements (`combFolds_of`). -/
theorem combFolds_at (hrep : F.rep₀ = F.repOf (F.coinFiles spec st cap lig))
    (hfold : ∀ R ω, FoldsOk (st R ω) F.sh) (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin))
    (l : (j : Fin F.stmts.size) → (F.Vs j).Bits) (r : Fin 2) : CombFolds (st R ω) F.sh (F.rep R ω t l r) :=
  combFolds_of (F.coinFiles spec st cap lig R ω t l) _ (F.rep_comb spec st cap lig hrep R ω t l r)
    (F.rep_extras spec st cap lig hrep R ω t l r)
    (by rw [F.rep_rounds spec st cap lig hrep]; exact coinsOf_sizes CT F.sh _ r) (hfold R ω)

/-- **hf's `Coins`**: each rep's rounds have `verifierRows`' sizes, and those have `verifyRep`'s counts
(`CoinsAgree`). -/
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
