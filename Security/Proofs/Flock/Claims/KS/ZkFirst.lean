import Proofs.Flock.Claims.KS.ZkEvent
import Proofs.Flock.Claims.KS.PlainThird

/-!
# The masked mixture's first and third bounds, from a simulation

`PlainFirst` and `PlainThird` under `--zk`, at the live session at `J` tables and the witness-lane verdict at table
`j` (`AccTxJ`). `zk_first_of_sim` bounds `V true`'s accepting runs whose reader looks up no column of table `j` off
`C` by the average over a tape of the live session's `AccTxJ`, from a simulation alone (`ZkSim`): coins `Φ (u, r')` of
`V true` with the law of uniform coins, and a prover `P'` of the live session whose live outcome at the tape `(ω, u)`
is the run's wherever `V true` accepts at `Φ (u, r')` (`ZkSimRun`, `viewOf`). Unlike the plain bound it reads no
draw in the record: `accTxJ_of_run` reads the setup at the statement's draw. At `zkCoins`, where each deployed coin
takes a live coin's first entries and the rest from the tape, the law holds (`zkSim_of_run`), so `zk_first_of_run`
asks only for `ZkSimRun`, deterministic; and at those coins a deployed event that a live event implies bounds the
mixture, the third bound's form (`zkMix_le_of_run`).
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Tables Claims.Bridge.Custody Claims.Bridge.Live
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- **A prover of the live session at `J` tables simulates `P` on the coins `Φ`**, under `--zk`: wherever `V true`
accepts at `Φ (u, r')`, the coins of `V true` at the tape `u` and the live coins `r'`, the live outcome of `P'` at the
tape `(ω, u)` and `r'` is the run's (`viewOf`). -/
def ZkSimRun {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (J : Claims.Msg → ℕ) (mPts : ℕ) {nMax : ℕ}
    (N : Claims.Msg → ℕ) (dg : ByteArray → ByteArray) (w : (x : Claims.Msg) → Wire nMax (J x) mPts) {τ : Type}
    (P' : Claims.Prover (Ω × τ)) (Φ : τ × (Fin (N x + 2) → LCoin nMax) → Fin ((V true).k x) → Coin) : Prop :=
  ∀ u r', (Claims.run sha512 P (V true) x ω (Φ (u, r'))).out = .acc →
    ∀ s spec sts link c₁ mids f,
      act true x ((Claims.run sha512 P (V true) x ω (Φ (u, r'))).msgs.map Prod.fst)
          ((List.range ((Claims.run sha512 P (V true) x ω (Φ (u, r'))).msgs.map Prod.fst).length).map
            (Claims.coinSeq (Φ (u, r')))) = .decide s spec sts link c₁ mids f →
        liveOut J (fun _ => mPts) N dg w sha512 P' x (ω, u) r' = viewOf (J x) mPts spec link c₁ f

namespace Gen

variable [S : ZkSetup]

/-- **A prover of the live session at `J` tables simulates `P` on the coins `Φ`**, under `--zk`: wherever `V true`
accepts at `Φ (u, r')`, the coins of `V true` at the tape `u` and the live coins `r'`, the live outcome of `P'` at the
tape `(ω, u)` and `r'` is the run's (`viewOf`). -/
def ZkSimRun {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (J : Claims.Msg → ℕ) (mPts : ℕ) {nMax : ℕ}
    (N : Claims.Msg → ℕ) (dg : ByteArray → ByteArray) (w : (x : Claims.Msg) → Wire nMax (J x) mPts) {τ : Type}
    (P' : Claims.Prover (Ω × τ)) (Φ : τ × (Fin (N x + 2) → LCoin nMax) → Fin ((V true).k x) → Coin) : Prop :=
  ∀ u r', (Claims.run sha512 P (V true) x ω (Φ (u, r'))).out = .acc →
    ∀ s spec sts link c₁ mids f,
      act true x ((Claims.run sha512 P (V true) x ω (Φ (u, r'))).msgs.map Prod.fst)
          ((List.range ((Claims.run sha512 P (V true) x ω (Φ (u, r'))).msgs.map Prod.fst).length).map
            (Claims.coinSeq (Φ (u, r')))) = .decide s spec sts link c₁ mids f →
        liveOut J (fun _ => mPts) N dg w sha512 P' x (ω, u) r' = viewOf (J x) mPts spec link c₁ f

end Gen

/-- **A prover of the live session at `J` tables simulates `P`**, under `--zk`: the coins `Φ (u, r')` have the law of
uniform coins, and `P'` simulates `P` on them (`ZkSimRun`). -/
def ZkSim {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (J : Claims.Msg → ℕ) (mPts : ℕ) {nMax : ℕ}
    (N : Claims.Msg → ℕ) (dg : ByteArray → ByteArray) (w : (x : Claims.Msg) → Wire nMax (J x) mPts) {τ : Type}
    [Fintype τ] [Nonempty τ] (P' : Claims.Prover (Ω × τ))
    (Φ : τ × (Fin (N x + 2) → LCoin nMax) → Fin ((V true).k x) → Coin) : Prop :=
  (∀ E : (Fin ((V true).k x) → Coin) → Prop, Claims.Pr (fun p => E (Φ p)) = Claims.Pr E) ∧
    ZkSimRun P x ω J mPts N dg w P' Φ

namespace Gen

variable [S : ZkSetup]

/-- **A prover of the live session at `J` tables simulates `P`**, under `--zk`: the coins `Φ (u, r')` have the law of
uniform coins, and `P'` simulates `P` on them (`ZkSimRun`). -/
def ZkSim {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (J : Claims.Msg → ℕ) (mPts : ℕ) {nMax : ℕ}
    (N : Claims.Msg → ℕ) (dg : ByteArray → ByteArray) (w : (x : Claims.Msg) → Wire nMax (J x) mPts) {τ : Type}
    [Fintype τ] [Nonempty τ] (P' : Claims.Prover (Ω × τ))
    (Φ : τ × (Fin (N x + 2) → LCoin nMax) → Fin ((V true).k x) → Coin) : Prop :=
  (∀ E : (Fin ((V true).k x) → Coin) → Prop, Claims.Pr (fun p => E (Φ p)) = Claims.Pr E) ∧
    ZkSimRun P x ω J mPts N dg w P' Φ

end Gen

/-- **The masked deployed coins at the tape `u` and the live coins `r'`**: coin `j` takes its first `n j` field elements
from the live coin `r' j` and the rest from `u j` (`mergeAt`). -/
def zkCoins {x : Claims.Msg} {N : ℕ} (hk : N + 2 = (V true).k x) (n : Fin ((V true).k x) → ℕ)
    (p : (Fin ((V true).k x) → Coin) × (Fin (N + 2) → LCoin 65536)) : Fin ((V true).k x) → Coin :=
  @mergeAt _ _ _ 𝔽 hk n p.1 p.2

/-- **`P'` simulates `P` on `zkCoins`** where it simulates the runs: the merged coins are uniform (`Pr_mergeAt`). -/
theorem zkSim_of_run {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {J : Claims.Msg → ℕ} {mPts : ℕ}
    {N : Claims.Msg → ℕ} {dg : ByteArray → ByteArray} {w : (x : Claims.Msg) → Wire 65536 (J x) mPts}
    {P' : Claims.Prover (Ω × (Fin ((V true).k x) → Coin))} (hk : N x + 2 = (V true).k x)
    (n : Fin ((V true).k x) → ℕ) (h : ZkSimRun P x ω J mPts N dg w P' (zkCoins hk n)) :
    ZkSim P x ω J mPts N dg w P' (zkCoins hk n) :=
  ⟨fun E => @Pr_mergeAt _ _ _ 𝔽 _ _ hk n E, h⟩

namespace Gen

variable [S : ZkSetup]

open Classical in
/-- **The masked mixture's first bound, from a simulation**: at a masked statement `x` that sets up at `s₀` with inputs
`I` and draw `d`, at `J x ≤` its tables above `j`, if a prover `P'` of the live session at `J` tables simulates `P`
(`ZkSim`), then the accepting runs whose reader looks up no column of table `j` off `C` are no likelier than the
average over `u` of `AccTxJ` at table `j` against `P'`'s strategies, at the root of `P`'s first message
(`Pr_le_liveMix`, `accTxJ_of_run`). -/
theorem zk_first_of_sim {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) {M : ℕ}
    (C : Fin M → List 𝔽) (j : ℕ) (J : Claims.Msg → ℕ) (mPts : ℕ) {s₀ : Ins} {spec₀ : SessionSpec}
    {sts₀ : Array Stmt} {d : Json} (hset : setupOf true x = some (s₀, spec₀, sts₀)) (hd : s₀.read.draw = some d)
    (hJ : J x ≤ sts₀.size) (hj : j < J x) {nMax : ℕ} (N : Claims.Msg → ℕ) (dg : ByteArray → ByteArray)
    (w : (x : Claims.Msg) → Wire nMax (J x) mPts) {τ : Type} [Fintype τ] [Nonempty τ] (P' : Claims.Prover (Ω × τ))
    (Φ : τ × (Fin (N x + 2) → LCoin nMax) → Fin ((V true).k x) → Coin) (hsim : ZkSim P x ω J mPts N dg w P' Φ) :
    Claims.Pr (fun r : Fin ((V true).k x) → Coin => (Claims.run sha512 P (V true) x ω r).out = .acc ∧
        ¬ ∃ (p : Fin M) (v : List 𝔽),
          Claims.Flock.lookCol (cols true (Claims.run sha512 P (V true) x ω r).seen j) p.val = some v ∧
            v ≠ C p) ≤
      (∑ u, (prob (AccTxJ s₀.read.I d j ((linkOf ((P.next x [] ω).eval sha512).1).getD default).1.toArray C)
        (liveSession dg (J x) mPts (N x)) (liveStrat J (fun _ => mPts) N dg w sha512 P' x (ω, u))).toReal) /
        Fintype.card τ := by
  set E : (Fin ((V true).k x) → Coin) → Prop := fun r => (Claims.run sha512 P (V true) x ω r).out = .acc ∧
    ¬ ∃ (p : Fin M) (v : List 𝔽),
      Claims.Flock.lookCol (cols true (Claims.run sha512 P (V true) x ω r).seen j) p.val = some v ∧ v ≠ C p
    with hEdef
  obtain ⟨hΦ, hsim⟩ := hsim
  refine Pr_le_liveMix J (fun _ => mPts) N dg w sha512 P' x ω
    (AccTxJ s₀.read.I d j ((linkOf ((P.next x [] ω).eval sha512).1).getD default).1.toArray C) E Φ (hΦ E)
    fun u r' hE => ?_
  obtain ⟨hacc, hno⟩ := hE
  obtain ⟨s, spec, sts, link, c₁, mids, f, hact, hv⟩ := accTxJ_of_run mPts C j hacc
  obtain ⟨rfl, -, rfl⟩ : s = s₀ ∧ spec = spec₀ ∧ sts = sts₀ := by
    have e := (act_decide hact).symm.trans hset
    simp only [Option.some.injEq, Prod.mk.injEq] at e
    exact e
  have hroot : ((linkOf ((P.next x [] ω).eval sha512).1).getD default).1.toArray = link.1.toArray := by
    obtain ⟨rest, hhead⟩ := run_head sha512 P (V true) x ω (Φ (u, r')) hacc
    obtain ⟨rest', -, hlink⟩ := act_decide_link hact
    rw [hhead] at hlink
    simp only [List.headD_cons] at hlink
    rw [hlink]
    rfl
  rw [hroot, hsim u r' hacc s spec sts link c₁ mids f hact]
  exact hv d hd hJ hj hno

end Gen

open Classical in
/-- **The masked mixture's first bound, from a simulation**: at a masked statement `x` that sets up at `s₀` with inputs
`I` and draw `d`, at `J x ≤` its tables above `j`, if a prover `P'` of the live session at `J` tables simulates `P`
(`ZkSim`), then the accepting runs whose reader looks up no column of table `j` off `C` are no likelier than the
average over `u` of `AccTxJ` at table `j` against `P'`'s strategies, at the root of `P`'s first message
(`Pr_le_liveMix`, `accTxJ_of_run`). -/
theorem zk_first_of_sim {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) {M : ℕ}
    (C : Fin M → List 𝔽) (j : ℕ) (J : Claims.Msg → ℕ) (mPts : ℕ) {s₀ : Ins} {spec₀ : SessionSpec}
    {sts₀ : Array Stmt} {d : Json} (hset : setupOf true x = some (s₀, spec₀, sts₀)) (hd : s₀.read.draw = some d)
    (hJ : J x ≤ sts₀.size) (hj : j < J x) {nMax : ℕ} (N : Claims.Msg → ℕ) (dg : ByteArray → ByteArray)
    (w : (x : Claims.Msg) → Wire nMax (J x) mPts) {τ : Type} [Fintype τ] [Nonempty τ] (P' : Claims.Prover (Ω × τ))
    (Φ : τ × (Fin (N x + 2) → LCoin nMax) → Fin ((V true).k x) → Coin) (hsim : ZkSim P x ω J mPts N dg w P' Φ) :
    Claims.Pr (fun r : Fin ((V true).k x) → Coin => (Claims.run sha512 P (V true) x ω r).out = .acc ∧
        ¬ ∃ (p : Fin M) (v : List 𝔽),
          Claims.Flock.lookCol (cols true (Claims.run sha512 P (V true) x ω r).seen j) p.val = some v ∧
            v ≠ C p) ≤
      (∑ u, (prob (AccTxJ s₀.read.I d j ((linkOf ((P.next x [] ω).eval sha512).1).getD default).1.toArray C)
        (liveSession dg (J x) mPts (N x)) (liveStrat J (fun _ => mPts) N dg w sha512 P' x (ω, u))).toReal) /
        Fintype.card τ :=
  Gen.zk_first_of_sim (S := S₀) (Ω := Ω) (P := P) (x := x) (ω := ω) (M := M) (C := C) (j := j) (J := J) (mPts := mPts)
    (s₀ := s₀) (spec₀ := spec₀) (sts₀ := sts₀) (d := d) (hset := hset) (hd := hd) (hJ := hJ) (hj := hj) (nMax := nMax)
    (N := N) (dg := dg) (w := w) (τ := τ) (P' := P') (Φ := Φ) (hsim := hsim)

open Classical in
/-- **The masked mixture's first bound, from a simulation of the runs**: `zk_first_of_sim` with the deployed coins as
the tape and `zkCoins` as the coins, whose law is uniform (`zkSim_of_run`), so it asks only for `ZkSimRun`,
deterministic. -/
theorem zk_first_of_run {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) {M : ℕ}
    (C : Fin M → List 𝔽) (j : ℕ) (J : Claims.Msg → ℕ) (mPts : ℕ) {s₀ : Ins} {spec₀ : SessionSpec}
    {sts₀ : Array Stmt} {d : Json} (hset : setupOf true x = some (s₀, spec₀, sts₀)) (hd : s₀.read.draw = some d)
    (hJ : J x ≤ sts₀.size) (hj : j < J x) (N : Claims.Msg → ℕ) (dg : ByteArray → ByteArray)
    (w : (x : Claims.Msg) → Wire 65536 (J x) mPts) (P' : Claims.Prover (Ω × (Fin ((V true).k x) → Coin)))
    (hk : N x + 2 = (V true).k x) (n : Fin ((V true).k x) → ℕ) (hrun : ZkSimRun P x ω J mPts N dg w P' (zkCoins hk n)) :
    Claims.Pr (fun r : Fin ((V true).k x) → Coin => (Claims.run sha512 P (V true) x ω r).out = .acc ∧
        ¬ ∃ (p : Fin M) (v : List 𝔽),
          Claims.Flock.lookCol (cols true (Claims.run sha512 P (V true) x ω r).seen j) p.val = some v ∧
            v ≠ C p) ≤
      (∑ u, (prob (AccTxJ s₀.read.I d j ((linkOf ((P.next x [] ω).eval sha512).1).getD default).1.toArray C)
        (liveSession dg (J x) mPts (N x)) (liveStrat J (fun _ => mPts) N dg w sha512 P' x (ω, u))).toReal) /
        Fintype.card (Fin ((V true).k x) → Coin) :=
  zk_first_of_sim P x ω C j J mPts hset hd hJ hj N dg w P' (zkCoins hk n) (zkSim_of_run hk n hrun)

open Classical in
/-- **A deployed masked event bounds a mixture of live strategies at `zkCoins`**: where the live event `pL` at the live
outcome of `P'` at the tape `(ω, u)` and the live coins `r'` gives `E` at `zkCoins (u, r')`, the average over the
deployed coins `u` of the live session's probability of `pL` against `P'`'s strategies is at most `Pr[E]`
(`liveMix_le_Pr`, `Pr_mergeAt`). -/
theorem zkMix_le_of_run {Ω : Type} (x : Claims.Msg) (ω : Ω) (J : Claims.Msg → ℕ) (mPts : ℕ) (N : Claims.Msg → ℕ)
    (dg : ByteArray → ByteArray) (w : (x : Claims.Msg) → Wire 65536 (J x) mPts)
    (P' : Claims.Prover (Ω × (Fin ((V true).k x) → Coin))) (hk : N x + 2 = (V true).k x)
    (n : Fin ((V true).k x) → ℕ) (pL : LiveOutJ (J x) mPts → Prop) (E : (Fin ((V true).k x) → Coin) → Prop)
    (h : ∀ u r', pL (liveOut J (fun _ => mPts) N dg w sha512 P' x (ω, u) r') → E (zkCoins hk n (u, r'))) :
    (∑ u, (prob pL (liveSession dg (J x) mPts (N x))
      (liveStrat J (fun _ => mPts) N dg w sha512 P' x (ω, u))).toReal) /
        Fintype.card (Fin ((V true).k x) → Coin) ≤ Claims.Pr E :=
  liveMix_le_Pr J (fun _ => mPts) N dg w sha512 P' x ω pL E (zkCoins hk n) (@Pr_mergeAt _ _ _ 𝔽 _ _ hk n E) h

end FlockDeploy.KS
