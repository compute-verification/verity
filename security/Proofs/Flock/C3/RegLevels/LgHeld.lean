module

public import Proofs.Flock.C3.Compose.HoloEncoded
public import Proofs.Flock.C3.Compose.HoloStreamSizes
public import Proofs.Flock.C3.RegLevels.SrcMaps
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
import Mathlib.Tactic.ReduceModChar
import Proofs.Flock.C3.RegLevels.LgCoins
import Proofs.Flock.C3.RegLevels.RRounds
import Proofs.Flock.C3.RegLevels.SrcRun
import Proofs.Flock.C3.Sparse.HeldStages

/-! `StageHeld` at R's stages: `laterCoins_of_held`'s `h4` and `h5`. -/

@[expose] public section

namespace FlockC3.RegLevels

open Flock.RecVStar (Inner InnerCommits InnerStream SchedRound StackedShape STACKED STACKED_CLAIMS stackedRounds)
open Flock.Guarantees (HoloStream heldBefore holoStreams holoRounds sparseRounds)
open FlockSoundness.Discharge.PlaysCoin (coinArr)
open FlockSoundness.Discharge.Recursion (VStmt HoloParams holoAt)
open FlockC3.Reps (nCoins StageHeld)

theorem holoStreams_four {ish : Inner} {rep rs : Array SchedRound} {sp : Flock.RecSparse.Shape} {sh : StackedShape}
    (hrep : holoRounds ish.m ish.kLog ish.claims = some rep) (hsp : ish.sparse = some sp)
    (hsh : ish.stacked = some sh) (hrs : stackedRounds sh = some rs) :
    holoStreams ish.m ish.kLog ish.claims ish.sparse ish.stacked = some ([((.rep 0 : HoloStream), rep), (.rep 1, rep),
      (.sparse 0, sparseRounds sp), (.sparse 1, sparseRounds sp)] ++ [(.stacked, rs)]) := by
  rw [hsp, hsh]
  simp [holoStreams, hrep, hrs]

theorem stageHeld_stacked {r : ℕ} {CT : Fin r → Type} {Cm : Type} {t : List (Cm × ((k : Fin r) → CT k))}
    {wr : List (Cm × ((k : Fin r) → CT k)) → Lean.Json × String}
    {held : Lean.Json × String → Except String InnerCommits} {ts : List FlockSoundness.Game.Step} {ish : Inner}
    {tbl : String} {cs : InnerCommits} {c3 : Flock.RecCoins.C3} {unit : String} {rep rs : Array SchedRound}
    {sp : Flock.RecSparse.Shape} {sh : StackedShape} {J k n : ℕ}
    (hh : held (wr t) = .ok cs) (hc3 : Flock.Guarantees.c3Of ish (some cs) = some c3)
    (hrep : holoRounds ish.m ish.kLog ish.claims = some rep) (hsp : ish.sparse = some sp)
    (hsh : ish.stacked = some sh) (hcl : sh.claims = STACKED_CLAIMS) (hrs : stackedRounds sh = some rs)
    (hcsz : (FlockC3.Opening.rep0 c3 unit).size = rs.size) (htbl : tbl ≠ s!"{STACKED}/")
    (hJ : J = 1 + ([((.rep 0 : HoloStream), rep), (.rep 1, rep), (.sparse 0, sparseRounds sp),
      (.sparse 1, sparseRounds sp)].map fun e => heldBefore e.1 e.2 e.2.size).sum)
    (hts : ts.filterMap Sum.getRight? = ((List.range' k n).map fun i => (FlockC3.Opening.rep0 c3 unit)[i]!).map
      (coinArr FlockSoundness.Refine.fOf))
    (hn : k + n ≤ (FlockC3.Opening.rep0 c3 unit).size) (hr : J + k + n ≤ r) :
    StageHeld t wr held (FlockC3.Compose.holoSelOne ish tbl) (J + k) ts := by
  obtain ⟨L, hS', hL, hf⟩ := holoStreams_stacked hrep hsh hrs
  have hS := holoStreams_four hrep hsp hsh hrs
  obtain rfl := List.append_cancel_right (Option.some.inj (hS'.symm.trans hS))
  have hpos := stackedRounds_coins hcl hrs
  obtain ⟨h1, h2⟩ := heldStage_of_lgCoins (unit := unit) hc3 hts
  refine ⟨List.range' k n, fun q => (FlockC3.Opening.rep0 c3 unit)[q]!,
    s!"{STACKED}/rep0", id, h1, fun q c h => h2 q c (by rw [hh] at h; exact h),
    by rw [List.length_range']; omega, ?_⟩
  intro p q hp
  have hk : heldBefore .stacked rs k = k := by
    have e := heldBefore_stacked (rs := rs) (k₂ := 0) k fun p' hp' => hpos (0 + p') (by omega)
    rw [Nat.zero_add] at e
    rw [e]
    simp [heldBefore]
  have hd := sampled_stacked (n := n) hS hL (hf tbl htbl) (fun p' hp' => hpos (k + p') (by omega)) p q hp
  rw [hk] at hd
  subst hJ
  exact hd

theorem weave_coins_prefix : ∀ σ ms cs,
    (FlockSoundness.Discharge.Exec.weave σ ms cs).filterMap Sum.getRight? <+: cs
  | [], _, _ => List.nil_prefix
  | .msg _ :: _, [], _ => List.nil_prefix
  | .msg _ :: σ, _ :: ms, cs => by
      simpa [FlockSoundness.Discharge.Exec.weave, List.filterMap_cons] using weave_coins_prefix σ ms cs
  | .coin _ :: _, _, [] => List.nil_prefix
  | .coin _ :: σ, ms, _ :: cs => by
      simpa [FlockSoundness.Discharge.Exec.weave, List.cons_prefix_cons] using weave_coins_prefix σ ms cs

theorem eq_range'_of_prefix {α : Type _} {f : ℕ → α} {l : List α} {k N : ℕ} (h : l <+: (List.range' k N).map f) :
    l = (List.range' k l.length).map f ∧ l.length ≤ N := by
  have hle : l.length ≤ N := by simpa using h.length_le
  refine ⟨?_, hle⟩
  conv_lhs => rw [List.prefix_iff_eq_take.1 h]
  rw [← List.map_take, List.take_range'_of_length_ge hle]

theorem lgCoins_of_weave {ts : List FlockSoundness.Game.Step} {σ ms} {c : Array (Array Flock.F128)} {k₂ : ℕ}
    (h : ts = FlockSoundness.Discharge.Exec.weave σ ms
      (((List.range' k₂ (c.size - k₂)).map fun i => (c)[i]!).map (coinArr FlockSoundness.Refine.fOf))) :
    ts.filterMap Sum.getRight? =
        ((List.range' k₂ (nCoins ts)).map fun i => (c)[i]!).map (coinArr FlockSoundness.Refine.fOf) ∧
      nCoins ts ≤ c.size - k₂ := by
  have hp := weave_coins_prefix σ ms
    (((List.range' k₂ (c.size - k₂)).map fun i => (c)[i]!).map (coinArr FlockSoundness.Refine.fOf))
  rw [← h, List.map_map] at hp
  rw [List.map_map]
  exact eq_range'_of_prefix hp

theorem le_size_of_ringSwitch {M : Flock.RecAlgebra.Messages Flock.F128}
    {a : Array (Flock.RecAlgebra.AClaim Flock.F128)} {c : Array (Array Flock.F128)} {rsr : Array Flock.F128}
    {rs : Flock.RecAlgebra.RsPts Flock.F128} {k₂ : ℕ} (hne : Flock.RecAlgebra.NoEmpty c)
    (h : Flock.RecAlgebra.ringSwitch M a 0 (Flock.RecAlgebra.RoundCoins.ofRounds c) = .ok ((rsr, rs), k₂)) :
    k₂ ≤ c.size := by
  unfold Flock.RecAlgebra.ringSwitch at h
  obtain ⟨⟨outs, betas, pts⟩, k₁, h1, h⟩ := Flock.RecAlgebra.cm_bind h
  obtain ⟨gamma, k₃, h2, h⟩ := Flock.RecAlgebra.cm_bind h
  cases Flock.RecAlgebra.cm_pure h
  obtain ⟨-, -, -, hk, -⟩ := Flock.RecAlgebra.rsClaims_spec M a.toList 0 0 k₁ _ _ _ _ h1
  rw [Array.length_toList] at hk
  rcases Flock.RecAlgebra.coinVec_ok h2 with ⟨hn, -, rfl⟩ | ⟨-, hr, rfl⟩
  · omega
  · have := (Array.getElem?_eq_some_iff.1 (Flock.RecAlgebra.ofRounds_get? hne hr).1).1
    omega

theorem stagesHeld_of_run {r : ℕ} {CT : Fin r → Type} {Cm : Type} {t : List (Cm × ((k : Fin r) → CT k))}
    {wr : List (Cm × ((k : Fin r) → CT k)) → Lean.Json × String}
    {held : Lean.Json × String → Except String InnerCommits} {st : Flock.Stmt} {hp : HoloParams}
    {u : FlockC3.Compose.C3Run (holoAt st hp)} {ish : Inner} {tbl : String} {cs : InnerCommits}
    {c3 : Flock.RecCoins.C3} {unit : String} {rep rs : Array SchedRound} {sp : Flock.RecSparse.Shape}
    {sh : StackedShape} {x : InnerStream} {S : ℕ} {stmt : Fin S → VStar.Stmt} {Vs : Fin S → VStmt}
    {Reg' : Type} {M : FlockC3.Reps.StackedMaps stmt Vs} {zs : (s : Fin S) → (Vs s).Outer Reg'}
    {Lb : (s : Fin S) → (Vs s).Bits} {osh : Flock.Shape} {w : Array Flock.F128}
    {kap : Array (Array (Array Flock.F128))} {e0 sb e cap₀} {res : Array Flock.F128} {k₁ : ℕ}
    (hh : held (wr t) = .ok cs) (hc3 : Flock.Guarantees.c3Of ish (some cs) = some c3)
    (hrep : holoRounds ish.m ish.kLog ish.claims = some rep) (hsp : ish.sparse = some sp)
    (hsh : ish.stacked = some sh) (hcl : sh.claims = STACKED_CLAIMS) (hrs : stackedRounds sh = some rs)
    (hx : cs.stream? "stacked/rep0" = some x) (hxs : x.coins.size = rs.size) (htbl : tbl ≠ s!"{STACKED}/")
    (hts : ∀ i : Fin 2, nCoins (u.ts i) = heldBefore (.rep i) rep rep.size)
    (hss : ∀ x : Fin 2, ∃ w cw, u.sp x = FlockC3.Sparse.stepsOf (FlockC3.Compose.sparseOf sp) w cw)
    (hrun : Flock.RecOpening.run osh (msgsO osh w (opsAt (readMaps M osh.sched) osh Lb kap c3 unit w))
      (FlockC3.Opening.claimsO osh w c3 unit) 0 (Flock.RecAlgebra.RoundCoins.ofRounds (FlockC3.Opening.rep0 c3 unit)) =
        .ok (res, k₁))
    (hz : ∀ y ∈ res, y = 0) (hleft : MapsLeft M zs Lb osh unit c3 w kap e0 sb e cap₀ u)
    (htot : 1 + nCoins (u.ts 0) + nCoins (u.ts 1) + nCoins (u.sp 0) + nCoins (u.sp 1) + nCoins u.op + nCoins u.lg ≤ r) :
    StageHeld t wr held (FlockC3.Compose.holoSelOne ish tbl)
        (1 + nCoins (u.ts 0) + nCoins (u.ts 1) + nCoins (u.sp 0) + nCoins (u.sp 1)) u.op ∧
      StageHeld t wr held (FlockC3.Compose.holoSelOne ish tbl)
        (1 + nCoins (u.ts 0) + nCoins (u.ts 1) + nCoins (u.sp 0) + nCoins (u.sp 1) + nCoins u.op) u.lg := by
  have hname : s!"{STACKED}/rep0" = "stacked/rep0" := by decide +kernel
  have hcsz : (FlockC3.Opening.rep0 c3 unit).size = rs.size := by
    rw [FlockC3.Opening.rep0_eq (c := x.coins) (by
      unfold Flock.RecCoins.C3.coins
      rw [commits_c3Of hc3, hname, hx]
      rfl), hxs]
  obtain ⟨rsr, rsp, k₂, lgr, lg, h₁, h₂, rfl⟩ := opening_parts hrun
  have hlast : Flock.RecAlgebra.Val.add
      ((msgsO osh w (opsAt (readMaps M osh.sched) osh Lb kap c3 unit w)).tAtom.getD rsp.T) rsp.T = 0 :=
    hz _ (by simp)
  rw [Flock.RecAlgebra.f128_eq_of_add_eq_zero hlast] at h₂
  obtain ⟨hne, -, hop, hR, -⟩ := hleft
  have hk₂ : k₂ ≤ (FlockC3.Opening.rep0 c3 unit).size := le_size_of_ringSwitch hne h₁
  have hop' := hop rsr rsp k₂ h₁
  obtain ⟨hlg, hn⟩ : _ ∧ nCoins u.lg ≤ (FlockC3.Opening.rep0 c3 unit).size - k₂ :=
    lgCoins_of_weave (hR rsr rsp k₂ lgr lg k₁ h₁ h₂).2
  have hopn : nCoins u.op = k₂ := by
    show (u.op.filterMap Sum.getRight?).length = _
    rw [hop']
    simp
  have hj := (FlockC3.Sparse.stageStarts (u := u) hts hss).2.2.2
  have h4 := stageHeld_stacked (t := t) hh hc3 hrep hsp hsh hcl hrs hcsz htbl hj hop' (by omega) (by omega)
  rw [Nat.add_zero] at h4
  refine ⟨h4, ?_⟩
  rw [hopn]
  exact stageHeld_stacked hh hc3 hrep hsp hsh hcl hrs hcsz htbl hj hlg (by omega) (by omega)

end FlockC3.RegLevels
