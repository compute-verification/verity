import Proofs.Flock.Claims.Bridge.Live
import Proofs.Flock.Claims.Bridge.Hm96Tree

/-!
# A mixture of live strategies bounds the runs of a verifier that reveals more

`PlainLiveWrPAt`'s first clause bounds `V false`'s runs by an average over a tape `τ` of the live session's
probability against strategies `σL u`. `V false` answers each round with a whole coin, the live session with the `n`
coins the round draws, so a prover that plays the live session keeps the rest of each coin on its tape:

* `mergeC`, `Pr_mergeC`: the coins whose coordinate `j` takes its first `n j` entries from `r` and the rest from `u`
  are uniform when `u` and `r` are: swapping those entries between `u` and `r` is an involution (`swapC`); `mergeAt`,
  `Pr_mergeAt`: the same with `r` indexed by another `Fin L`, `L = K`.
* `fnC`, `Pr_fnC`: the same with the rest of coordinate `j` a uniform function `f j` on the tape read at a key of `r`,
  such as the live coins a prover has seen by round `j`, so a fork that redraws them reads fresh entries; uniform
  functions at fixed points are uniform (`Pr_evalAt`); `fnAt`, `Pr_fnAt`: with `r` indexed by `Fin L`.
* `liveMix_eq`: the average over `u` of the live session's probability of `pL` against `liveStrat` at `(ω, u)` is the
  probability over the tape and the round coins of `pL` at the live outcome.
* `Pr_le_liveMix`: an event `E` of coins whose law is that of `E ∘ Φ` on the tape and the live session's coins, and
  which implies `pL` at the live outcome of a prover `P'` at the tape `(ω, u)`, has probability at most that average;
  `liveMix_le_Pr`: where `pL` implies `E ∘ Φ` instead, the average is at most `Pr[E]`.
* `liveStrat_fst`: `liveStrat`'s roots are the wire's roots of the prover's first message (`fst_eqRec`).
-/

namespace Claims.Bridge.Live

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.Tables

/-- **Coordinate `j`'s first `n j` entries from `r`, the rest from `u`.** -/
def mergeC {K M : ℕ} {X : Type} (n : Fin K → ℕ) (u r : Fin K → Fin M → X) : Fin K → Fin M → X :=
  fun j i => if i.val < n j then r j i else u j i

/-- Swapping coordinate `j`'s first `n j` entries between two coin vectors. -/
def swapC {K M : ℕ} (X : Type) (n : Fin K → ℕ) :
    (Fin K → Fin M → X) × (Fin K → Fin M → X) ≃ (Fin K → Fin M → X) × (Fin K → Fin M → X) where
  toFun p := (mergeC n p.1 p.2, mergeC n p.2 p.1)
  invFun p := (mergeC n p.1 p.2, mergeC n p.2 p.1)
  left_inv p := by
    refine Prod.ext (funext fun j => funext fun i => ?_) (funext fun j => funext fun i => ?_) <;>
      simp only [mergeC] <;> split_ifs <;> rfl
  right_inv p := by
    refine Prod.ext (funext fun j => funext fun i => ?_) (funext fun j => funext fun i => ?_) <;>
      simp only [mergeC] <;> split_ifs <;> rfl

open Classical in
/-- **The merged coins are uniform**: `Pr_{u, r}[E (mergeC n u r)] = Pr[E]` for every event `E`. -/
theorem Pr_mergeC {K M : ℕ} {X : Type} [Fintype X] [Nonempty X] (n : Fin K → ℕ)
    (E : (Fin K → Fin M → X) → Prop) :
    Pr (fun p : (Fin K → Fin M → X) × (Fin K → Fin M → X) => E (mergeC n p.1 p.2)) = Pr E := by
  refine (Pr_equiv (swapC X n) (fun q => E q.1)).trans ?_
  rw [Pr_eq_Ex, Ex_prod, Pr_eq_Ex]
  exact congrArg Ex (funext fun a => Ex_const' (ind E a))

/-- `mergeC` with the coins `r` indexed by `Fin L`, where `L = K`. -/
def mergeAt {K L M : ℕ} {X : Type} (h : L = K) (n : Fin K → ℕ) (u : Fin K → Fin M → X) (r : Fin L → Fin M → X) :
    Fin K → Fin M → X :=
  mergeC n u fun j => r (Fin.cast h.symm j)

open Classical in
/-- **The merged coins are uniform**, with the coins `r` indexed by `Fin L`. -/
theorem Pr_mergeAt {K L M : ℕ} {X : Type} [Fintype X] [Nonempty X] (h : L = K) (n : Fin K → ℕ)
    (E : (Fin K → Fin M → X) → Prop) :
    Pr (fun p : (Fin K → Fin M → X) × (Fin L → Fin M → X) => E (mergeAt h n p.1 p.2)) = Pr E := by
  subst h
  exact Pr_mergeC n E

/-- **Independent uniform functions at fixed points are uniform**: `Pr_f[G (j ↦ f j (d j))] = Pr[G]`. -/
theorem Pr_evalAt {K : ℕ} {D Y : Type} [Fintype D] [DecidableEq D] [Fintype Y] [Nonempty Y] (d : Fin K → D)
    (G : (Fin K → Y) → Prop) : Pr (fun f : Fin K → D → Y => G fun j => f j (d j)) = Pr G := by
  refine (Pr_equiv ((Equiv.piCongrRight fun j => Equiv.funSplitAt (d j) Y).trans
    (Equiv.arrowProdEquivProdArrow (Fin K) (fun _ => Y) fun j => {e // e ≠ d j} → Y)) fun q => G q.1).trans ?_
  rw [Pr_eq_Ex, Ex_prod, Pr_eq_Ex]
  exact congrArg Ex (funext fun a => Ex_const' (ind G a))

/-- **Coordinate `j`'s first `n j` entries from `r`, the rest from `f j` at `key j r`**: `mergeC` with the rest of each
coin a function on the tape read at a key of the coins, such as the entries of `r` a prover sees by round `j`. -/
def fnC {K M : ℕ} {X D : Type} (n : Fin K → ℕ) (key : Fin K → (Fin K → Fin M → X) → D)
    (f : Fin K → D → Fin M → X) (r : Fin K → Fin M → X) : Fin K → Fin M → X :=
  mergeC n (fun j => f j (key j r)) r

/-- **Those coins are uniform**: `Pr_{f, r}[E (fnC n key f r)] = Pr[E]` at uniform functions `f`, whatever the keys:
at each `r` the functions' values at the keys are uniform (`Pr_evalAt`), and the merged coins are (`Pr_mergeC`). -/
theorem Pr_fnC {K M : ℕ} {X D : Type} [Fintype X] [Nonempty X] [Fintype D] [DecidableEq D] (n : Fin K → ℕ)
    (key : Fin K → (Fin K → Fin M → X) → D) (E : (Fin K → Fin M → X) → Prop) :
    Pr (fun p : (Fin K → D → Fin M → X) × (Fin K → Fin M → X) => E (fnC n key p.1 p.2)) = Pr E := by
  calc Pr (fun p : (Fin K → D → Fin M → X) × (Fin K → Fin M → X) => E (fnC n key p.1 p.2))
      = Pr (fun q : (Fin K → Fin M → X) × (Fin K → D → Fin M → X) => E (fnC n key q.2 q.1)) :=
        Pr_equiv (Equiv.prodComm _ _) fun q => E (fnC n key q.2 q.1)
    _ = Ex fun r => Pr fun f : Fin K → D → Fin M → X => E (fnC n key f r) :=
        Pr_prod fun r f => E (fnC n key f r)
    _ = Ex fun r => Pr fun u : Fin K → Fin M → X => E (mergeC n u r) :=
        congrArg Ex (funext fun r => Pr_evalAt (fun j => key j r) fun u => E (mergeC n u r))
    _ = Pr (fun q : (Fin K → Fin M → X) × (Fin K → Fin M → X) => E (mergeC n q.2 q.1)) :=
        (Pr_prod fun r u => E (mergeC n u r)).symm
    _ = Pr (fun p : (Fin K → Fin M → X) × (Fin K → Fin M → X) => E (mergeC n p.1 p.2)) :=
        Pr_equiv (Equiv.prodComm _ _) fun p => E (mergeC n p.1 p.2)
    _ = Pr E := Pr_mergeC n E

/-- `fnC` with the coins `r` indexed by `Fin L`, where `L = K`. -/
def fnAt {K L M : ℕ} {X D : Type} (h : L = K) (n : Fin K → ℕ) (key : Fin K → (Fin K → Fin M → X) → D)
    (f : Fin K → D → Fin M → X) (r : Fin L → Fin M → X) : Fin K → Fin M → X :=
  fnC n key f fun j => r (Fin.cast h.symm j)

/-- **The coins `fnAt` makes are uniform**, with the coins `r` indexed by `Fin L`. -/
theorem Pr_fnAt {K L M : ℕ} {X D : Type} [Fintype X] [Nonempty X] [Fintype D] [DecidableEq D] (h : L = K)
    (n : Fin K → ℕ) (key : Fin K → (Fin K → Fin M → X) → D) (E : (Fin K → Fin M → X) → Prop) :
    Pr (fun p : (Fin K → D → Fin M → X) × (Fin L → Fin M → X) => E (fnAt h n key p.1 p.2)) = Pr E := by
  subst h
  exact Pr_fnC n key E

open Classical in
/-- **A mixture of live strategies is a probability over the tape and the live coins**: the average over `u` of the
live session's probability of `pL` against `liveStrat` at `(ω, u)` is the probability over `u` and the round coins
`r'` of `pL` at the live outcome of `P'` at `(ω, u)` and `r'`. -/
theorem liveMix_eq {nMax : ℕ} (J mPts N : Msg → ℕ) (dg : ByteArray → ByteArray)
    (w : (x : Msg) → Wire nMax (J x) (mPts x)) (H : Msg → Msg) {Ω τ : Type} [Fintype τ] [Nonempty τ]
    (P' : Prover (Ω × τ)) (x : Msg) (ω : Ω) (pL : LiveOutJ (J x) (mPts x) → Prop) :
    (∑ u, (prob pL (liveSession dg (J x) (mPts x) (N x))
      (liveStrat J mPts N dg w H P' x (ω, u))).toReal) / Fintype.card τ =
      Pr (fun p : τ × (Fin (N x + 2) → LCoin nMax) => pL (liveOut J mPts N dg w H P' x (ω, p.1) p.2)) := by
  symm
  calc Pr (fun p : τ × (Fin (N x + 2) → LCoin nMax) => pL (liveOut J mPts N dg w H P' x (ω, p.1) p.2))
      = Ex fun u => Pr fun r' : Fin (N x + 2) → LCoin nMax => pL (liveOut J mPts N dg w H P' x (ω, u) r') :=
        Pr_prod (fun u r' => pL (liveOut J mPts N dg w H P' x (ω, u) r'))
    _ = Ex fun u => (prob pL (liveSession dg (J x) (mPts x) (N x))
          (liveStrat J mPts N dg w H P' x (ω, u))).toReal := by
        refine congrArg Ex (funext fun u => ?_)
        rw [liveStrat, prob_eqRec, prob_toReal, Pr_eq_Ex]
        exact ex_outcome H P' (fun x => N x + 2) (fun x => sessionK dg (J x) (mPts x) (N x))
          (fun x => sessionIf (w x) dg (N x)) x (ω, u) (sessionLe (w x) dg (N x) _) (ind pL)
    _ = _ := rfl

open Classical in
/-- **A mixture of live strategies bounds an event of another verifier's coins**: where the event `E` of coins `R` has
the law of `E ∘ Φ` on the tape `τ` and the live session's round coins, and `E (Φ (u, r'))` gives `pL` at the live
outcome of `P'` at the tape `(ω, u)` and coins `r'`, `Pr[E]` is at most the average over `u` of the live session's
probability of `pL` against the strategy `P'`'s bytes decode to at `(ω, u)` (`liveStrat`). -/
theorem Pr_le_liveMix {nMax : ℕ} (J mPts N : Msg → ℕ) (dg : ByteArray → ByteArray)
    (w : (x : Msg) → Wire nMax (J x) (mPts x)) (H : Msg → Msg) {Ω τ : Type} [Fintype τ] [Nonempty τ]
    (P' : Prover (Ω × τ)) (x : Msg) (ω : Ω) (pL : LiveOutJ (J x) (mPts x) → Prop) {R : Type} [Fintype R]
    (E : R → Prop) (Φ : τ × (Fin (N x + 2) → LCoin nMax) → R) (hΦ : Pr (fun p => E (Φ p)) = Pr E)
    (hsim : ∀ u r', E (Φ (u, r')) → pL (liveOut J mPts N dg w H P' x (ω, u) r')) :
    Pr E ≤ (∑ u, (prob pL (liveSession dg (J x) (mPts x) (N x))
      (liveStrat J mPts N dg w H P' x (ω, u))).toReal) / Fintype.card τ := by
  rw [← hΦ, liveMix_eq J mPts N dg w H P' x ω pL]
  exact Pr_mono fun p hp => hsim p.1 p.2 hp

open Classical in
/-- **An event of another verifier's coins bounds a mixture of live strategies**: where the event `E` of coins `R` has
the law of `E ∘ Φ` on the tape and the round coins, and `pL` at the live outcome of `P'` at `(ω, u)` and `r'` gives
`E (Φ (u, r'))`, the average over `u` of the live session's probability of `pL` against `liveStrat` is at most
`Pr[E]`. -/
theorem liveMix_le_Pr {nMax : ℕ} (J mPts N : Msg → ℕ) (dg : ByteArray → ByteArray)
    (w : (x : Msg) → Wire nMax (J x) (mPts x)) (H : Msg → Msg) {Ω τ : Type} [Fintype τ] [Nonempty τ]
    (P' : Prover (Ω × τ)) (x : Msg) (ω : Ω) (pL : LiveOutJ (J x) (mPts x) → Prop) {R : Type} [Fintype R]
    (E : R → Prop) (Φ : τ × (Fin (N x + 2) → LCoin nMax) → R) (hΦ : Pr (fun p => E (Φ p)) = Pr E)
    (hsim : ∀ u r', pL (liveOut J mPts N dg w H P' x (ω, u) r') → E (Φ (u, r'))) :
    (∑ u, (prob pL (liveSession dg (J x) (mPts x) (N x))
      (liveStrat J mPts N dg w H P' x (ω, u))).toReal) / Fintype.card τ ≤ Pr E := by
  rw [← hΦ, liveMix_eq J mPts N dg w H P' x ω pL]
  exact Pr_mono fun p hp => hsim p.1 p.2 hp

/-- A strategy cast across an equation of two games that send the same type sends the same message. -/
theorem fst_eqRec {α M : Type} {k k' : M → Game α} (h : Game.send M k' = Game.send M k)
    (s : Strategy (Game.send M k')) : (h ▸ s : Strategy (Game.send M k)).1 = s.1 := by
  obtain ⟨-, hk⟩ := Game.send.inj h
  cases eq_of_heq hk
  rfl

/-- **`liveStrat`'s roots are the wire's roots of the prover's first message.** -/
theorem liveStrat_fst {nMax : ℕ} (J mPts N : Msg → ℕ) (dg : ByteArray → ByteArray)
    (w : (x : Msg) → Wire nMax (J x) (mPts x)) (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) :
    (liveStrat J mPts N dg w H P x ω).1 = (w x).root ((P.next x [] ω).eval H).1 :=
  fst_eqRec (liveSession_eq dg (J x) (mPts x) (N x)).symm _

end Claims.Bridge.Live
