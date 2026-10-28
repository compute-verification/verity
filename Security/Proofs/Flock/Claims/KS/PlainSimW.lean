import Proofs.Flock.Claims.KS.ForkSlack
import Proofs.Flock.Claims.Recursion.PlainK

/-!
# The deployed events a simulator's bounds read

A simulator of a prover `P` is a live prover `P'` with a finite tape and coins `Φ` of `V false` at the tape and the
live coins (`PlainDrawSim`'s `PlainSimWrPAtD`). Its bounds read these events of `P`'s deployed runs:

* `plainRoot`, `plainRoot_of_first`: the roots of `P`'s first message, which the strategies send wherever `P'`'s first
  message is `P`'s and the wire reads roots as the deployed link does;
* `plainEv₁`, `PlainClashRun`: an accepting run with a collision among the verifier's calls, which a live self-clash at
  `P'`'s outcome gives at `Φ` (the third bound, by `liveMix_le_Pr`);
* `plainForkEv`: an accepting fork with a collision at its round `j`, which bounds the fork at `P'`'s strategies up to
  the slack `Σ_{j<i} Δ_{j,i} Pr[F_i] + Δ_{j,K} Pr[F₁]` (`Flock.forkSlack`, the second bound);
* `PlainRawClashRun`, `plainClashRun_of_raw`: where the self-clash is at level 0, the collision in `PlainClashRun`
  follows from two different columns the run's reps open at one position.
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Tables Claims.Bridge.Live
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- **The roots of `P`'s first message**, as `plain_first_of_simD` reads them. -/
def plainRoot {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) : Array ByteArray :=
  ((linkOf ((P.next x [] ω).eval sha512).1).getD default).1.toArray

/-- **A live prover whose first message is `P`'s sends `P`'s roots**, where the wire reads roots as the deployed link
does (`liveStrat_fst`): the strategies' roots `PlainSimWrPAtD` asks for. -/
theorem plainRoot_of_first {Ω τ : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {mPts : ℕ}
    (N : Claims.Msg → ℕ) (dg : ByteArray → ByteArray) {w : Claims.Msg → Wire 65536 1 mPts} {P' : Claims.Prover (Ω × τ)}
    (hw : ∀ a, (w x).root a = ((linkOf a).getD default).1.toArray)
    (h1 : ∀ u, ((P'.next x [] (ω, u)).eval sha512).1 = ((P.next x [] ω).eval sha512).1) (u : τ) :
    (liveStrat (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x (ω, u)).1 = plainRoot P x ω := by
  rw [liveStrat_fst, hw, h1]
  rfl

/-- **An accepting run with a collision among the verifier's calls**: the one-run event the third bound reads. -/
def plainEv₁ {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (s : Fin ((V false).k x) → Coin) : Prop :=
  (Claims.run sha512 P (V false) x ω s).out = .acc ∧ Claims.TiedColl sha512 (Claims.run sha512 P (V false) x ω s).vLog

/-- **An accepting run with a collision among the verifier's calls**: the one-run event the third bound reads. -/
def plainEv₁K (keep : Bool → Bool) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (s : Fin ((V false).k x) → Coin) : Prop :=
  (Claims.run sha512 P (VK keep) x ω s).out = .acc ∧ Claims.TiedColl sha512 (Claims.run sha512 P (VK keep) x ω s).vLog

/-- **An accepting fork with a collision among the verifier's calls in its two runs**, at the fork's coin
`plainJ … r ℓ`: the event the second bound reads. -/
def plainForkEv {x : Claims.Msg} {spec : SessionSpec} {st : Stmt} (h : tables false x = [(spec, st)]) {m k0 : ℕ}
    {l₀ l₁ : Accounting.Level} {rest : List Accounting.Level} {ext : ℕ}
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) {Ω : Type} (P : Claims.Prover Ω)
    (r : Fin 2) (ℓ : Fin (l₀ :: l₁ :: rest).length) (ω : Ω) (s s' : Fin ((V false).k x) → Coin) : Prop :=
  (Claims.run sha512 P (V false) x ω s).out = .acc ∧
    (Claims.run sha512 P (V false) x ω (Claims.forkCoins (plainJ (tables_head_one h) hsch r ℓ) s s')).out = .acc ∧
    Claims.TiedColl sha512 ((Claims.run sha512 P (V false) x ω s).vLog ++
      (Claims.run sha512 P (V false) x ω (Claims.forkCoins (plainJ (tables_head_one h) hsch r ℓ) s s')).vLog)

/-- **An accepting fork with a collision among the verifier's calls in its two runs**, at the fork's coin
`plainJ … r ℓ`: the event the second bound reads. -/
def plainForkEvK (keep : Bool → Bool) {x : Claims.Msg} {spec : SessionSpec} {st : Stmt} (h : tables false x = [(spec, st)]) {m k0 : ℕ}
    {l₀ l₁ : Accounting.Level} {rest : List Accounting.Level} {ext : ℕ}
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) {Ω : Type} (P : Claims.Prover Ω)
    (r : Fin 2) (ℓ : Fin (l₀ :: l₁ :: rest).length) (ω : Ω) (s s' : Fin ((V false).k x) → Coin) : Prop :=
  (Claims.run sha512 P (VK keep) x ω s).out = .acc ∧
    (Claims.run sha512 P (VK keep) x ω (Claims.forkCoins (plainJ (x := x) (tables_head_one h) hsch r ℓ) s s')).out = .acc ∧
    Claims.TiedColl sha512 ((Claims.run sha512 P (VK keep) x ω s).vLog ++
      (Claims.run sha512 P (VK keep) x ω (Claims.forkCoins (plainJ (x := x) (tables_head_one h) hsch r ℓ) s s')).vLog)

/-- **A live event at `P'`'s outcome gives an accepting run of `P` with a collision**: wherever `pL` holds at the live
outcome of `P'` at the tape `(ω, u)` and the live coins `r'`, `V false` accepts `P` at `Φ (u, r')` with a collision
among its calls (`plainEv₁`). -/
def PlainClashRun {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (mPts : ℕ) (N : Claims.Msg → ℕ)
    (dg : ByteArray → ByteArray) (w : Claims.Msg → Wire 65536 1 mPts) {τ : Type} (P' : Claims.Prover (Ω × τ))
    (Φ : τ × (Fin (N x + 2) → LCoin 65536) → Fin ((V false).k x) → Coin) (pL : LiveOutJ 1 mPts → Prop) : Prop :=
  ∀ u r', pL (liveOut (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x (ω, u) r') → plainEv₁ P x ω (Φ (u, r'))

/-- **A live event at `P'`'s outcome gives an accepting run of `P` with a collision**: wherever `pL` holds at the live
outcome of `P'` at the tape `(ω, u)` and the live coins `r'`, `VK keep` accepts `P` at `Φ (u, r')` with a collision
among its calls (`plainEv₁`). -/
def PlainClashRunK (keep : Bool → Bool) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (mPts : ℕ) (N : Claims.Msg → ℕ)
    (dg : ByteArray → ByteArray) (w : Claims.Msg → Wire 65536 1 mPts) {τ : Type} (P' : Claims.Prover (Ω × τ))
    (Φ : τ × (Fin (N x + 2) → LCoin 65536) → Fin ((V false).k x) → Coin) (pL : LiveOutJ 1 mPts → Prop) : Prop :=
  ∀ u r', pL (liveOut (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x (ω, u) r') → plainEv₁K keep P x ω (Φ (u, r'))

/-- **A live event at `P'`'s outcome gives an accepting run of `P` whose reps open two different columns at one
position, or a collision**: `PlainClashRun` with a self-clash at level 0 asked without the hashes (`plainRawCols`).
`SelfClashL` also counts clashes at later levels, against the caps of the reps' pendings; those still need their
collision. -/
def PlainRawClashRun {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (mPts : ℕ) (N : Claims.Msg → ℕ)
    (dg : ByteArray → ByteArray) (w : Claims.Msg → Wire 65536 1 mPts) {τ : Type} (P' : Claims.Prover (Ω × τ))
    (Φ : τ × (Fin (N x + 2) → LCoin 65536) → Fin ((V false).k x) → Coin) (pL : LiveOutJ 1 mPts → Prop) : Prop :=
  ∀ u r', pL (liveOut (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x (ω, u) r') →
    (Claims.run sha512 P (V false) x ω (Φ (u, r'))).out = .acc ∧
      ((∃ (q : ℕ) (a b : List 𝔽),
        (q, a) ∈ plainRawCols (Claims.run sha512 P (V false) x ω (Φ (u, r'))).seen ∧
        (q, b) ∈ plainRawCols (Claims.run sha512 P (V false) x ω (Φ (u, r'))).seen ∧ a ≠ b) ∨
        Claims.TiedColl sha512 (Claims.run sha512 P (V false) x ω (Φ (u, r'))).vLog)

/-- **The clash gives the collision**: two different columns of an accepting run at one position collide in its log
(`plain_selfClash_tied`), so `PlainRawClashRun` gives `PlainClashRun`. -/
theorem plainClashRun_of_raw {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {mPts : ℕ}
    {N : Claims.Msg → ℕ} {dg : ByteArray → ByteArray} {w : Claims.Msg → Wire 65536 1 mPts} {τ : Type}
    {P' : Claims.Prover (Ω × τ)} {Φ : τ × (Fin (N x + 2) → LCoin 65536) → Fin ((V false).k x) → Coin}
    {pL : LiveOutJ 1 mPts → Prop} (h : PlainRawClashRun P x ω mPts N dg w P' Φ pL) :
    PlainClashRun P x ω mPts N dg w P' Φ pL := fun u r' hp => by
  obtain ⟨hacc, ⟨q, a, b, ha, hb, hab⟩ | ht⟩ := h u r' hp
  · exact ⟨hacc, plain_selfClash_tied hacc ha hb hab⟩
  · exact ⟨hacc, ht⟩

end FlockDeploy.KS
