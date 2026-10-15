import Proofs.Flock.C3.RegLevels.SrcOpenRel
import Proofs.Flock.C3.RegLevels.SrcUnits
import Proofs.Flock.C3.Opening.SrcCover

/-!
# n7's levels: `hR`'s `OpenRel` from R's opening's run

`openRel_of_plays` asks that the messages' transposes be their ring switches' (`hT`). At the messages w holds
(`decodedO`) they are: ring switch `k` is w's 128 elements at its slots (`decodedO_rs_eq`, `wiO_rs`), and its transpose
is the term `wt` at the first of them (`valO`), `tOf` of those elements (`decodedO_rsT`). `openRel_of_run` is
`openRel_of_plays` where R's opening's run returns zero residuals (ring switching's are among them: `opening_parts`), at
`RecOpening.shape?`'s shape, whose schedule's `m` is `mReg` (`shape_spec`, `fast100_dims`). `hR_of_run` and
`hR_of_units` are `levelTie_of_run`'s and `levelTie_of_units`' (so `levelTie_of_reads`') `hR` with that conjunct given
and the others left; at the units' messages `hT` holds and `OpenTie` is `hmsg`.
-/

namespace FlockC3.RegLevels

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockC3.Compose
open FlockSoundness.Discharge.PlaysCoin (coinArr)

/-- Ring switch `k` of the messages w holds is w's 128 elements from `128 k`. -/
theorem decodedO_rs_eq (sh : Flock.Shape) (w : Array Flock.F128) (cons : ℕ → ℕ → Flock.F128) {k : ℕ}
    (hk : k < sh.claims) :
    (FlockC3.Opening.decodedO sh w cons).rs[k]! = (Array.range 128).map fun v => w[128 * k + v]! := by
  simp only [FlockC3.Opening.decodedO, Flock.RecAlgebra.Messages.map, Flock.RecAlgebra.symbolic]
  rw [getElem!_pos _ k (by simpa using hk)]
  simp only [Array.getElem_map, Array.getElem_range, Array.map_map]
  apply Array.ext
  · simp
  · intro v h1 h2
    have h1' : v < 128 := by simpa using h1
    simp only [Array.getElem_map, Array.getElem_range, Function.comp, Flock.RecAlgebra.Lin.eval_var,
      FlockC3.Opening.valO, FlockC3.Opening.wiO_rs sh hk h1']

/-- **The messages w holds have their ring switches' transposes** (`openRel_of_plays`' `hT`): ring switch `k`'s
transpose is the term `wt` at its first slot of w (`valO`), `tOf` of its 128 elements there. -/
theorem decodedO_rsT (sh : Flock.Shape) (w : Array Flock.F128) (cons : ℕ → ℕ → Flock.F128) {k : ℕ}
    (hk : k < sh.claims) :
    (FlockC3.Opening.decodedO sh w cons).rsT[k]! = tOf (FlockC3.Opening.decodedO sh w cons).rs[k]! := by
  rw [decodedO_rs_eq sh w cons hk]
  simp only [FlockC3.Opening.decodedO, Flock.RecAlgebra.Messages.map, Flock.RecAlgebra.symbolic]
  rw [getElem!_pos _ k (by simpa using hk)]
  simp only [Array.getElem_map, Array.getElem_range, Array.map_map]
  apply Array.ext
  · simp [tOf]
  · intro u h1 h2
    simp only [tOf, Array.getElem_map, Array.getElem_range, Function.comp, Flock.RecAlgebra.Lin.eval_var,
      FlockC3.Opening.valO, FlockC3.Opening.wiO_rs sh hk (show 0 < 128 by omega), Nat.add_zero,
      Flock.RecAlgebra.termValue]
    refine Flock.RecAlgebra.foldl_range_congr (fun b v hv => ?_) 0
    rw [getElem!_pos _ v (by simpa using hv)]
    simp only [Array.getElem_map, Array.getElem_range]

/-- **`hR`'s `OpenRel` from R's opening's run**: `openRel_of_plays` at ring switching's output in a run of R's opening
returning zero residuals, at `RecOpening.shape?`'s shape at `mReg`, whose schedule's `m` is `mReg`. -/
theorem openRel_of_run {e : ExecHolo} {u : C3Run e} {cl : ℕ} {sh : Flock.Shape}
    {M : Flock.RecAlgebra.Messages Flock.F128} {a : Array (Flock.RecAlgebra.AClaim Flock.F128)}
    {c : Array (Array Flock.F128)} {k₀ k₁ k₂ : ℕ} {res rsr : Array Flock.F128} {rs : Flock.RecAlgebra.RsPts Flock.F128}
    (hsh : Flock.RecOpening.shape? e.cls.shape.mReg cl = some sh)
    (hrun : Flock.RecOpening.run sh M a k₀ (Flock.RecAlgebra.Coins.ofRounds c) = .ok (res, k₁))
    (hz : ∀ x ∈ res, x = 0) (hne : Flock.RecAlgebra.NoEmpty c) (ha : a.size = 2)
    (hshape : ∀ k < 2, (M.rs[k]!).size = 128 ∧ ((a.map Flock.RecAlgebra.AClaim.toClaim)[k]!).skip.size = 64 ∧
      ((a.map Flock.RecAlgebra.AClaim.toClaim)[k]!).x.size = e.cls.shape.mReg - 6)
    (hT : ∀ k < 2, M.rsT[k]! = tOf M.rs[k]!)
    (hcl : ∀ i : Fin 2, ClaimRel e.cls.shape.mReg (a.map Flock.RecAlgebra.AClaim.toClaim)[i.val]! (u.claim i).toClaim)
    (hpl : PlaysT (FlockC3.Opening.openingAt u) u.op (some u.opened))
    (hmsg : ∃ ρ γ, u.op = FlockC3.Opening.openSteps (fun i => vecOf 128 M.rs[i.val]!) ρ γ)
    (h₁ : Flock.RecAlgebra.ringSwitch M a k₀ (Flock.RecAlgebra.Coins.ofRounds c) = .ok ((rsr, rs), k₂))
    (hcoin : u.op.filterMap Sum.getRight? = ((List.range' k₀ (k₂ - k₀)).map fun i => c[i]!).map (coinArr fOf)) :
    ∃ nc, OpenRel sh.sched.m nc rs.T rs.sws u.opened := by
  obtain ⟨rsr', rs', k₂', _lgr, _lg, h₁', -, rfl⟩ := opening_parts hrun
  obtain ⟨⟨rfl, rfl⟩, rfl⟩ : (rsr = rsr' ∧ rs = rs') ∧ k₂ = k₂' := by simpa [h₁] using h₁'
  rw [(FlockSoundness.Refine.fast100_dims _ _ (shape_spec hsh).2).2]
  exact openRel_of_plays h₁ (fun x hx => hz x (by simp [hx])) hne ha hshape hT hcl hpl hmsg hcoin

/-- **`levelTie_of_run`'s `hR`** with its `OpenRel` from the run (`openRel_of_run`); left are its `Opened` and the
Ligerito steps' coins (`hR'`), and the opening steps' coins from ring switching's first round (`hcoin`). -/
theorem hR_of_run {e : ExecHolo} {u : C3Run e} {cl : ℕ} {sh : Flock.Shape}
    {M : Flock.RecAlgebra.Messages Flock.F128} {a : Array (Flock.RecAlgebra.AClaim Flock.F128)}
    {c : Array (Array Flock.F128)} {k₀ k₁ : ℕ} {res : Array Flock.F128}
    (hsh : Flock.RecOpening.shape? e.cls.shape.mReg cl = some sh)
    (hrun : Flock.RecOpening.run sh M a k₀ (Flock.RecAlgebra.Coins.ofRounds c) = .ok (res, k₁))
    (hz : ∀ x ∈ res, x = 0) (hne : Flock.RecAlgebra.NoEmpty c) (ha : a.size = 2)
    (hshape : ∀ k < 2, (M.rs[k]!).size = 128 ∧ ((a.map Flock.RecAlgebra.AClaim.toClaim)[k]!).skip.size = 64 ∧
      ((a.map Flock.RecAlgebra.AClaim.toClaim)[k]!).x.size = e.cls.shape.mReg - 6)
    (hT : ∀ k < 2, M.rsT[k]! = tOf M.rs[k]!)
    (hcl : ∀ i : Fin 2, ClaimRel e.cls.shape.mReg (a.map Flock.RecAlgebra.AClaim.toClaim)[i.val]! (u.claim i).toClaim)
    (hpl : PlaysT (FlockC3.Opening.openingAt u) u.op (some u.opened))
    (hmsg : ∃ ρ γ, u.op = FlockC3.Opening.openSteps (fun i => vecOf 128 M.rs[i.val]!) ρ γ)
    {msId : String} {cap : Array ByteArray} {p : Flock.LigeritoProof}
    (hcoin : ∀ (rsr : Array Flock.F128) (rs : Flock.RecAlgebra.RsPts Flock.F128) (k₂ : ℕ),
      Flock.RecAlgebra.ringSwitch M a k₀ (Flock.RecAlgebra.Coins.ofRounds c) = .ok ((rsr, rs), k₂) →
      u.op.filterMap Sum.getRight? = ((List.range' k₀ (k₂ - k₀)).map fun i => c[i]!).map (coinArr fOf))
    (hR' : ∀ (rsr : Array Flock.F128) (rs : Flock.RecAlgebra.RsPts Flock.F128) (k₂ : ℕ) (lgr : Array Flock.F128)
      (lg : Flock.RecAlgebra.LgPts),
      Flock.RecAlgebra.ringSwitch M a k₀ (Flock.RecAlgebra.Coins.ofRounds c) = .ok ((rsr, rs), k₂) →
      Flock.RecAlgebra.ligerito sh.sched M rs.T rs.sws k₂ (Flock.RecAlgebra.Coins.ofRounds c) = .ok ((lgr, lg), k₁) →
      (∀ cs ∈ lg.conss.toList, Flock.RecAlgebra.Opened (Flock.MerkleScheme.sha512 msId) sh.sched cap p M cs) ∧
      u.lg.filterMap Sum.getRight? = ((List.range' k₂ (c.size - k₂)).map (fun i => c[i]!)).map (coinArr fOf)) :
    ∀ (rsr : Array Flock.F128) (rs : Flock.RecAlgebra.RsPts Flock.F128) (k₂ : ℕ) (lgr : Array Flock.F128)
      (lg : Flock.RecAlgebra.LgPts),
      Flock.RecAlgebra.ringSwitch M a k₀ (Flock.RecAlgebra.Coins.ofRounds c) = .ok ((rsr, rs), k₂) →
      Flock.RecAlgebra.ligerito sh.sched M rs.T rs.sws k₂ (Flock.RecAlgebra.Coins.ofRounds c) = .ok ((lgr, lg), k₁) →
      (∀ cs ∈ lg.conss.toList, Flock.RecAlgebra.Opened (Flock.MerkleScheme.sha512 msId) sh.sched cap p M cs) ∧
      (∃ nc, OpenRel sh.sched.m nc rs.T rs.sws u.opened) ∧
      u.lg.filterMap Sum.getRight? = ((List.range' k₂ (c.size - k₂)).map (fun i => c[i]!)).map (coinArr fOf) :=
  fun rsr rs k₂ lgr lg h₁ h₂ => ⟨(hR' rsr rs k₂ lgr lg h₁ h₂).1,
    openRel_of_run hsh hrun hz hne ha hshape hT hcl hpl hmsg h₁ (hcoin rsr rs k₂ h₁), (hR' rsr rs k₂ lgr lg h₁ h₂).2⟩

/-- **`levelTie_of_units`' (and `levelTie_of_reads`') `hR`** with its `OpenRel` from the run (`openRel_of_run`), at
the messages w holds: their transposes are their ring switches' (`decodedO_rsT`) and `OpenTie` is the opening steps'
messages. Left are the openings' `α`, their units and the Ligerito steps' weave (`hR'`), and the opening steps' coins
from ring switching's first round (`hcoin`). -/
theorem hR_of_units {nS : ℕ} {Vs : Fin nS → VStmt} {Reg' : Type} {zs : (s : Fin nS) → (Vs s).Outer Reg'}
    {Lb : (s : Fin nS) → (Vs s).Bits} {e : ExecHolo} {u : C3Run e} {sh : Flock.Shape} {w : Array Flock.F128}
    {ops : FlockVBridge.AccChain.Openings} {a : Array (Flock.RecAlgebra.AClaim Flock.F128)}
    {c : Array (Array Flock.F128)} {k₀ k₁ : ℕ} {res : Array Flock.F128}
    (hsh : Flock.RecOpening.shape? e.cls.shape.mReg 2 = some sh)
    (hrun : Flock.RecOpening.run sh (msgsO sh w ops) a k₀ (Flock.RecAlgebra.Coins.ofRounds c) = .ok (res, k₁))
    (hz : ∀ x ∈ res, x = 0) (hne : Flock.RecAlgebra.NoEmpty c) (ha : a.size = 2)
    (hshape : ∀ k < 2, ((msgsO sh w ops).rs[k]!).size = 128 ∧
      ((a.map Flock.RecAlgebra.AClaim.toClaim)[k]!).skip.size = 64 ∧
      ((a.map Flock.RecAlgebra.AClaim.toClaim)[k]!).x.size = e.cls.shape.mReg - 6)
    (hcl : ∀ i : Fin 2, ClaimRel e.cls.shape.mReg (a.map Flock.RecAlgebra.AClaim.toClaim)[i.val]! (u.claim i).toClaim)
    (hpl : PlaysT (FlockC3.Opening.openingAt u) u.op (some u.opened)) (htie : FlockC3.Opening.OpenTie sh w ops u)
    {sibs : ℕ → Array ByteArray} {cap : Array ByteArray} {p : Flock.LigeritoProof}
    (hcoin : ∀ (rsr : Array Flock.F128) (rs : Flock.RecAlgebra.RsPts Flock.F128) (k₂ : ℕ),
      Flock.RecAlgebra.ringSwitch (msgsO sh w ops) a k₀ (Flock.RecAlgebra.Coins.ofRounds c) = .ok ((rsr, rs), k₂) →
      u.op.filterMap Sum.getRight? = ((List.range' k₀ (k₂ - k₀)).map fun i => c[i]!).map (coinArr fOf))
    (hR' : ∀ (rsr : Array Flock.F128) (rs : Flock.RecAlgebra.RsPts Flock.F128) (k₂ : ℕ) (lgr : Array Flock.F128)
      (lg : Flock.RecAlgebra.LgPts),
      Flock.RecAlgebra.ringSwitch (msgsO sh w ops) a k₀ (Flock.RecAlgebra.Coins.ofRounds c) = .ok ((rsr, rs), k₂) →
      Flock.RecAlgebra.ligerito sh.sched (msgsO sh w ops) rs.T rs.sws k₂ (Flock.RecAlgebra.Coins.ofRounds c) =
        .ok ((lgr, lg), k₁) →
      (∀ ℓ < sh.sched.levels.size, ops.alpha ℓ = (lg.conss[ℓ]!).alpha) ∧ OpensUnits zs Lb sh ops sibs cap p lg ∧
      u.lg = FlockSoundness.Discharge.Exec.weave (ligShape GF128 GF256 Dig e.rsch.sch) (ligMsgs p capDig sh.sched)
        (((List.range' k₂ (c.size - k₂)).map (fun i => c[i]!)).map (coinArr fOf))) :
    ∀ (rsr : Array Flock.F128) (rs : Flock.RecAlgebra.RsPts Flock.F128) (k₂ : ℕ) (lgr : Array Flock.F128)
      (lg : Flock.RecAlgebra.LgPts),
      Flock.RecAlgebra.ringSwitch (msgsO sh w ops) a k₀ (Flock.RecAlgebra.Coins.ofRounds c) = .ok ((rsr, rs), k₂) →
      Flock.RecAlgebra.ligerito sh.sched (msgsO sh w ops) rs.T rs.sws k₂ (Flock.RecAlgebra.Coins.ofRounds c) =
        .ok ((lgr, lg), k₁) →
      (∀ ℓ < sh.sched.levels.size, ops.alpha ℓ = (lg.conss[ℓ]!).alpha) ∧ OpensUnits zs Lb sh ops sibs cap p lg ∧
      (∃ nc, OpenRel sh.sched.m nc rs.T rs.sws u.opened) ∧
      u.lg = FlockSoundness.Discharge.Exec.weave (ligShape GF128 GF256 Dig e.rsch.sch) (ligMsgs p capDig sh.sched)
        (((List.range' k₂ (c.size - k₂)).map (fun i => c[i]!)).map (coinArr fOf)) :=
  fun rsr rs k₂ lgr lg h₁ h₂ => ⟨(hR' rsr rs k₂ lgr lg h₁ h₂).1, (hR' rsr rs k₂ lgr lg h₁ h₂).2.1,
    openRel_of_run hsh hrun hz hne ha hshape
      (fun k hk => decodedO_rsT sh w _ (by rw [FlockC3.Opening.shape_claims hsh]; exact hk)) hcl hpl htie h₁
      (hcoin rsr rs k₂ h₁), (hR' rsr rs k₂ lgr lg h₁ h₂).2.2⟩

end FlockC3.RegLevels
