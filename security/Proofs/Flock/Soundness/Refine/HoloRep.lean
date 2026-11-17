module

public import Proofs.Flock.Soundness.Refine.Frames
public import Proofs.Flock.VBridge.Algebra.Holo
public import Flock.Verify
import Mathlib.Algebra.Order.Floor.Semifield
import Mathlib.Combinatorics.Matroid.Init
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import VCVio.Prelude.Core

/-! One rep with hidden unit circuits as a transcript program (`holoRepProg`), with `z_partial`'s round of no coins. -/

@[expose] public section

namespace FlockSoundness.Refine

open Flock Transcript

local notation "𝔽" => FlockLevel3.GF128

/-- Round `k` issued no coins; the record keeps the round, so the program consumes it. -/
def closeEmpty : TM Unit := fun t =>
  match t.rounds[t.k]? with
  | none => .error s!"R2: round {t.k}: the verifier issued no such round"
  | some r =>
    if r.n ≠ 0 then .error s!"R2: round {t.k}: {r.n} coins issued where z_partial's round squeezes none"
    else if ¬ r.answers (framed t 2 0) then .error s!"R2: round {t.k}: the proof's messages differ from z_partial's round"
    else .ok ((), { t with pending := ByteArray.empty, k := t.k + 1 })

def holoLincheckT (c : CircuitFold) (zc : ZcOut) (p : LincheckProof) (hpart : Array F128) : TM LcOut := do
  let kLog := c.kLog
  let rhoIn := zc.rho.extract 0 (kLog - 6)
  label "flock-lincheck-v0"
  let alpha ← sample
  let beta ← sample
  let mut claim := alpha * zc.vA + zc.vB + beta
  let mut ts : Array F128 := #[]
  for j in [0:kLog - 6] do
    let (q1, qinf) := p.rounds[j]!
    absorbF q1
    absorbF qinf
    let t ← sample
    ts := ts.push t
    let e0 := claim + q1
    let c1 := e0 + q1 + qinf
    claim := qinf * t * t + c1 * t + e0
  let comb ← match c.folded alpha beta (skipWeights zc.z) rhoIn ts with
    | .ok v => pure v
    | .error err => throw err
  Flock.check (comb.size == 64) "lincheck: the circuit fold has the wrong width"
  absorbSlice p.zPartial
  closeEmpty
  absorbSlice hpart
  Flock.check (dot (RecAlgebra.addH comb hpart) p.zPartial == claim) "lincheck: ConsistencyFailed (sumcheck-final, h)"
  let rPrime ← sample
  let w := dot (skipWeights rPrime) p.zPartial
  return { w, rPrime, rIn := ts.reverse }

def holoRepProg (st : Setup) (sch : Flock.Schedule) (p : Proof) (hpart : Array F128) (extras : Array Claim) :
    TM Unit := do
  let zc ← Flock.bindAndZerocheck st.m st.digest (p.cap.foldl (· ++ ·) ByteArray.empty) p.zerocheck
  let lc ← holoLincheckT st.fold zc p.lincheck hpart
  let (T, sws) ← Flock.ringSwitch st.m (Flock.abcClaims st.m st.kLog zc lc ++ extras) p.ringSwitches
  Flock.ligerito st.merkle sch T p.cap p.ligerito (Flock.basis sws)

def holoLcMsgs (kLog : ℕ) (p : LincheckProof) (hpart : Array F128) : List Game.Msg :=
  lcMsgs kLog p ++ [⟨Fin 64 → 𝔽, vecOf 64 hpart⟩]

def holoRepMsgs {D : Type} (st : Setup) (sch : Flock.Schedule) (nClaims : ℕ) (p : Proof) (hpart : Array F128)
    (capD : Array ByteArray → Model.Cap D) : List Game.Msg :=
  zcMsgs st.m p.zerocheck ++ holoLcMsgs st.kLog p.lincheck hpart ++ opMsgs nClaims p.ringSwitches ++
    ligMsgs p.ligerito capD sch

theorem closeEmpty_ok {t t' : Transcript} (h : closeEmpty t = .ok ((), t')) :
    ∃ r, t.rounds[t.k]? = some r ∧ r.n = 0 ∧ r.answers (framed t 2 0) ∧
      t' = { t with pending := ByteArray.empty, k := t.k + 1 } := by
  simp only [closeEmpty] at h
  split at h
  · exact absurd h (by simp)
  · rename_i r hr
    by_cases h1 : r.n ≠ 0
    · simp [h1] at h
    · by_cases h2 : ¬ r.answers (framed t 2 0)
      · simp [h1, h2] at h
      · simp only [h1, h2, ite_false] at h
        injection h with h
        obtain ⟨-, ht⟩ := Prod.mk.inj h
        exact ⟨r, hr, Decidable.of_not_not h1, Decidable.of_not_not h2, ht.symm⟩

theorem closeEmpty_bind {β : Type} {f : Unit → TM β} {t t' : Transcript} {b : β}
    (h : (closeEmpty >>= f) t = .ok (b, t')) :
    t.k < t.rounds.size ∧ (t.rounds[t.k]!).n = 0 ∧
      ∃ t₁, t₁.rounds = t.rounds ∧ t₁.k = t.k + 1 ∧ f () t₁ = .ok (b, t') := by
  obtain ⟨⟨⟩, t₁, h₁, h₂⟩ := bind_ok h
  obtain ⟨r, hr, hn, -, ht⟩ := closeEmpty_ok h₁
  have hlt : t.k < t.rounds.size := by
    by_contra hge; rw [Array.getElem?_eq_none (by omega)] at hr; cases hr
  have hget : t.rounds[t.k]! = r := by
    rw [getElem!_pos t.rounds t.k hlt]; exact Option.some.inj ((Array.getElem?_eq_getElem hlt).symm.trans hr)
  subst ht
  exact ⟨hlt, by rw [hget]; exact hn, ⟨t.rounds, t.k + 1, ByteArray.empty⟩, rfl, rfl, h₂⟩

theorem closeEmpty_answered {t : Transcript} {L : List UInt8} (hlt : t.k < t.rounds.size)
    (hN : (t.rounds[t.k]!).n = 0) (ha : (t.rounds[t.k]!).answers ⟨L.toArray⟩)
    (hL : t.pending.data.toList ++ (Frame.sq 2 0).enc = L) :
    closeEmpty t = .ok ((), ⟨t.rounds, t.k + 1, ByteArray.empty⟩) := by
  have hc : framed t 2 0 = ⟨L.toArray⟩ := by rw [← hL, ← framed_enc]
  rw [getElem!_pos t.rounds t.k hlt] at hN ha
  rw [← hc] at ha
  simp [closeEmpty, Array.getElem?_eq_getElem hlt, hN, ha]

end FlockSoundness.Refine
