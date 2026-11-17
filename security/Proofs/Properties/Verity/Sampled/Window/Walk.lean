module

public import Proofs.Properties.Verity.Sampled.Window
public import Proofs.Properties.Verity.Sampled.Project

/-! SPOT's coupling over a window: the projection through several sessions, in order. -/

@[expose] public section

namespace Verity.Game.Pos

variable {α V ι : Type} {g : Game α} {σ : Strategy g}

theorem toStop_of_stops (q : Pos g σ V ι) (h : (∃ i, (q.Ks i).isHere) ∨ ∃ a, q.g' = .ret a) : q.toStop = q := by
  obtain ⟨g', s', Ks, x, back⟩ := q
  show walk g' s' Ks x back = _
  cases g' with
  | ret a => rfl
  | send M k =>
    rcases h with h | ⟨a, ha⟩
    · rw [walk, ite_eq_left_of_eq_true _ _ (eq_true h)]
    · cases ha
  | coin C k =>
    rcases h with h | ⟨a, ha⟩
    · rw [walk, ite_eq_left_of_eq_true _ _ (eq_true h)]
    · cases ha

theorem toStop_toStop (p : Pos g σ V ι) : p.toStop.toStop = p.toStop := toStop_of_stops _ p.toStop_stops

end Verity.Game.Pos

namespace Verity.Sampled

open Verity.Game
open FlockSoundness.Discharge.Tables (SubmitJ LiveOutJ liveRoundsJ KLJ liveSession addRoundJ)
open FlockSoundness.Refine (roundOf)
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

structure PosDecoders (g : Game Exchange) (σ : Strategy g) (R : ℕ) (Reg : Type) where
  reg : Option Claims.Msg → Reg
  root : (Fin R → Option Claims.Msg) → Array ByteArray
  sub : (J : ℕ) → (Fin R → Option Claims.Msg) → Option (SubmitJ J)
  files : Pos g σ Claims.Msg (Fin R) → Array ByteArray

def Decoders.atEnd {g : Game Exchange} {σ : Strategy g} {R : ℕ} {Reg : Type} (D : Decoders R Reg) :
    PosDecoders g σ R Reg where
  reg := D.reg
  root := D.root
  sub := D.sub
  files q := D.proofs (CoinTape.run g σ q.tape)

section SessionK

variable {g : Game Exchange} {σ : Strategy g} {Ω Reg : Type} {R mPts : ℕ} (D : PosDecoders g σ R Reg)
  (M : CoinMap g σ Ω R mPts) (dg : ByteArray → ByteArray) (J : ℕ) (B : RoundHold) {τ γ : Type}
  (wr : LiveOutJ J mPts → τ) (K : τ → Game γ)

abbrev roundsThen (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽) (N : ℕ)
    (rs : Fin J → Fin 2 → List Flock.Round) : Game γ :=
  (((liveRoundsJ dg J N rs).bind (KLJ rootB pts)).map wr).bind K

noncomputable def stepK (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽) (N : ℕ)
    (rs : Fin J → Fin 2 → List Flock.Round) (q : Pos g σ Claims.Msg (Fin R)) (fin : (a : τ) → Strategy (K a))
    (next : (s : SubmitJ J) → (c : Fin s.n → 𝔽) →
      Strategy (roundsThen dg J wr K rootB pts N (addRoundJ rs s.tab s.rep (roundOf dg s.bytes s.n c)))) :
    Option (SubmitJ J) → Strategy (roundsThen dg J wr K rootB pts (N + 1) rs)
  | none => ⟨none, ⟨D.files q,
      fin (wr ⟨rootB, pts, rs, D.files q⟩)⟩⟩
  | some s => ⟨some s, next s⟩

noncomputable def roundsK (next : Pos g σ Claims.Msg (Fin R) → (a : τ) → Strategy (K a)) (rootB : Array ByteArray)
    (pts : Fin 2 → Fin mPts → 𝔽) :
    (N : ℕ) → (rs : Fin J → Fin 2 → List Flock.Round) → Pos g σ Claims.Msg (Fin R) →
      Strategy (roundsThen dg J wr K rootB pts N rs)
  | 0, rs, p => ⟨D.files p,
      next p (wr ⟨rootB, pts, rs, D.files p⟩)⟩
  | N + 1, rs, p => stepK D dg J wr K rootB pts N rs p.toStop (next p.toStop)
    (fun s c => roundsK next rootB pts N (addRoundJ rs s.tab s.rep (roundOf dg s.bytes s.n c))
      (p.toStop.rewrite (M.round p.toStop J s c)).consume) (D.sub J p.toStop.modes)

noncomputable def sessionK (next : Pos g σ Claims.Msg (Fin R) → (a : τ) → Strategy (K a)) (N : ℕ)
    (p : Pos g σ Claims.Msg (Fin R)) : Strategy (((liveSession dg J mPts N).map wr).bind K) :=
  ⟨D.root p.toStop.modes, fun pts => roundsK D M dg J wr K next (D.root p.toStop.modes) pts N (fun _ _ => [])
    (p.toStop.rewrite (M.link p.toStop pts)).consume⟩

noncomputable def stepWK {ο : Type} (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽)
    (rs : Fin J → Fin 2 → List Flock.Round) (q : Pos g σ Claims.Msg (Fin R)) (fin : τ → Game ο)
    (next : (s : SubmitJ J) → (Fin s.n → 𝔽) → Game ο) : Option (SubmitJ J) → Game ο
  | none => fin (wr ⟨rootB, pts, rs, D.files q⟩)
  | some s => .coin (Fin s.n → 𝔽) (next s)

noncomputable def roundsWK {ο : Type} (next : Pos g σ Claims.Msg (Fin R) → τ → Game ο) (rootB : Array ByteArray)
    (pts : Fin 2 → Fin mPts → 𝔽) :
    (N : ℕ) → (rs : Fin J → Fin 2 → List Flock.Round) → Pos g σ Claims.Msg (Fin R) → Game ο
  | 0, rs, p => next p (wr ⟨rootB, pts, rs, D.files p⟩)
  | N + 1, rs, p => stepWK D J wr rootB pts rs p.toStop (next p.toStop)
    (fun s c => roundsWK next rootB pts N (addRoundJ rs s.tab s.rep (roundOf dg s.bytes s.n c))
      (p.toStop.rewrite (M.round p.toStop J s c)).consume) (D.sub J p.toStop.modes)

noncomputable def sessionWK {ο : Type} (next : Pos g σ Claims.Msg (Fin R) → τ → Game ο) (N : ℕ)
    (p : Pos g σ Claims.Msg (Fin R)) : Game ο :=
  .coin (Fin 2 → Fin mPts → 𝔽) fun pts => roundsWK D M dg J wr next (D.root p.toStop.modes) pts N (fun _ _ => [])
    (p.toStop.rewrite (M.link p.toStop pts)).consume

def stepWKS {ο : Type} (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽)
    (rs : Fin J → Fin 2 → List Flock.Round) (q : Pos g σ Claims.Msg (Fin R)) (fin : τ → Game ο)
    (next : (s : SubmitJ J) → (Fin s.n → 𝔽) → Game ο) (wf : ∀ a, Strategy (fin a))
    (w : ∀ s c, Strategy (next s c)) : (o : Option (SubmitJ J)) → Strategy (stepWK D J wr rootB pts rs q fin next o)
  | none => wf _
  | some s => w s

noncomputable def roundsWKS {ο : Type} (next : Pos g σ Claims.Msg (Fin R) → τ → Game ο)
    (wn : ∀ q a, Strategy (next q a)) (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽) :
    (N : ℕ) → (rs : Fin J → Fin 2 → List Flock.Round) → (p : Pos g σ Claims.Msg (Fin R)) →
      Strategy (roundsWK D M dg J wr next rootB pts N rs p)
  | 0, _, _ => wn _ _
  | N + 1, rs, p => stepWKS D J wr rootB pts rs p.toStop (next p.toStop) _ (wn p.toStop)
    (fun _ _ => roundsWKS next wn rootB pts N _ _) (D.sub J p.toStop.modes)

noncomputable def sessionWKS {ο : Type} (next : Pos g σ Claims.Msg (Fin R) → τ → Game ο)
    (wn : ∀ q a, Strategy (next q a)) (N : ℕ) (p : Pos g σ Claims.Msg (Fin R)) :
    Strategy (sessionWK D M dg J wr next N p) :=
  fun pts => roundsWKS D M dg J wr next wn _ pts N _ _

theorem prob_stepK (E : γ → Prop) (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽) (N : ℕ)
    (rs : Fin J → Fin 2 → List Flock.Round) (q : Pos g σ Claims.Msg (Fin R)) (finP : (a : τ) → Strategy (K a))
    (finW : τ → Game (γ × CoinTape g σ))
    (nextP : (s : SubmitJ J) → (c : Fin s.n → 𝔽) →
      Strategy (roundsThen dg J wr K rootB pts N (addRoundJ rs s.tab s.rep (roundOf dg s.bytes s.n c))))
    (nextW : (s : SubmitJ J) → (Fin s.n → 𝔽) → Game (γ × CoinTape g σ))
    (hf : ∀ a (w : Strategy (finW a)), prob E (K a) (finP a) = prob (fun o => E o.1) (finW a) w)
    (h : ∀ s c (w : Strategy (nextW s c)),
      prob E (roundsThen dg J wr K rootB pts N (addRoundJ rs s.tab s.rep (roundOf dg s.bytes s.n c))) (nextP s c) =
        prob (fun o => E o.1) (nextW s c) w) :
    ∀ (o : Option (SubmitJ J)) (w : Strategy (stepWK D J wr rootB pts rs q finW nextW o)),
      prob E (roundsThen dg J wr K rootB pts (N + 1) rs) (stepK D dg J wr K rootB pts N rs q finP nextP o) =
        prob (fun o => E o.1) (stepWK D J wr rootB pts rs q finW nextW o) w
  | none, w => hf _ w
  | some s, w => Finset.sum_congr rfl fun c _ => congrArg (HMul.hMul _) (h s c (w c))

theorem prob_roundsK (E : γ → Prop) (nextP : Pos g σ Claims.Msg (Fin R) → (a : τ) → Strategy (K a))
    (nextW : Pos g σ Claims.Msg (Fin R) → τ → Game (γ × CoinTape g σ))
    (h : ∀ q a (w : Strategy (nextW q a)), prob E (K a) (nextP q a) = prob (fun o => E o.1) (nextW q a) w)
    (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽) :
    ∀ (N : ℕ) (rs : Fin J → Fin 2 → List Flock.Round) (p : Pos g σ Claims.Msg (Fin R))
      (w : Strategy (roundsWK D M dg J wr nextW rootB pts N rs p)),
      prob E (roundsThen dg J wr K rootB pts N rs) (roundsK D M dg J wr K nextP rootB pts N rs p) =
        prob (fun o => E o.1) (roundsWK D M dg J wr nextW rootB pts N rs p) w
  | 0, _, p, w => h p _ w
  | N + 1, rs, p, w =>
    prob_stepK D dg J wr K E rootB pts N rs p.toStop (nextP p.toStop) (nextW p.toStop) _ _ (h p.toStop)
      (fun _ _ w => prob_roundsK E nextP nextW h rootB pts N _ _ w) (D.sub J p.toStop.modes) w

theorem prob_sessionK (E : γ → Prop) (nextP : Pos g σ Claims.Msg (Fin R) → (a : τ) → Strategy (K a))
    (nextW : Pos g σ Claims.Msg (Fin R) → τ → Game (γ × CoinTape g σ))
    (h : ∀ q a (w : Strategy (nextW q a)), prob E (K a) (nextP q a) = prob (fun o => E o.1) (nextW q a) w)
    (N : ℕ) (p : Pos g σ Claims.Msg (Fin R)) (w : Strategy (sessionWK D M dg J wr nextW N p)) :
    prob E (((liveSession dg J mPts N).map wr).bind K) (sessionK D M dg J wr K nextP N p) =
      prob (fun o => E o.1) (sessionWK D M dg J wr nextW N p) w :=
  Finset.sum_congr rfl fun pts _ => congrArg (HMul.hMul _)
    (prob_roundsK D M dg J wr K E nextP nextW h _ pts N _ _ (w pts))

open Classical in
noncomputable def stopOffK (fin e : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞) (q : Pos g σ Claims.Msg (Fin R)) :
    Option (SubmitJ J) → ℝ≥0∞
  | none => fin q
  | some s => avg fun c : Fin s.n → 𝔽 => if B q.g' J s c then 0 else e (q.rewrite (M.round q J s c)).consume

noncomputable def roundsOffK (e : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞) : ℕ → Pos g σ Claims.Msg (Fin R) → ℝ≥0∞
  | 0, p => e p
  | N + 1, p => stopOffK M J B e (roundsOffK e N) p.toStop (D.sub J p.toStop.modes)

open Classical in
noncomputable def sessionOffK (BL : LinkHold mPts) (e : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞) (N : ℕ)
    (p : Pos g σ Claims.Msg (Fin R)) : ℝ≥0∞ :=
  avg fun pts : Fin 2 → Fin mPts → 𝔽 =>
    if BL p.toStop.g' pts then 0 else roundsOffK D M J B e N (p.toStop.rewrite (M.link p.toStop pts)).consume

inductive RoundsReachK {ο : Type} (next : Pos g σ Claims.Msg (Fin R) → τ → ο → Prop) (rootB : Array ByteArray)
    (pts : Fin 2 → Fin mPts → 𝔽) :
    ℕ → (Fin J → Fin 2 → List Flock.Round) → Pos g σ Claims.Msg (Fin R) → ο → Prop
  | zero (rs : Fin J → Fin 2 → List Flock.Round) (p : Pos g σ Claims.Msg (Fin R)) (o : ο)
      (hn : next p (wr ⟨rootB, pts, rs, D.files p⟩) o) :
      RoundsReachK next rootB pts 0 rs p o
  | stop (N : ℕ) (rs : Fin J → Fin 2 → List Flock.Round) (p : Pos g σ Claims.Msg (Fin R)) (o : ο)
      (h : D.sub J p.toStop.modes = none)
      (hn : next p.toStop (wr ⟨rootB, pts, rs, D.files p.toStop⟩) o) :
      RoundsReachK next rootB pts (N + 1) rs p o
  | round (N : ℕ) (rs : Fin J → Fin 2 → List Flock.Round) (p : Pos g σ Claims.Msg (Fin R)) (s : SubmitJ J)
      (c : Fin s.n → 𝔽) (o : ο) (h : D.sub J p.toStop.modes = some s) (hB : ¬ B p.toStop.g' J s c)
      (ho : RoundsReachK next rootB pts N (addRoundJ rs s.tab s.rep (roundOf dg s.bytes s.n c))
        (p.toStop.rewrite (M.round p.toStop J s c)).consume o) :
      RoundsReachK next rootB pts (N + 1) rs p o

def SessionReachK {ο : Type} (BL : LinkHold mPts) (next : Pos g σ Claims.Msg (Fin R) → τ → ο → Prop) (N : ℕ)
    (p : Pos g σ Claims.Msg (Fin R)) (o : ο) : Prop :=
  ∃ pts : Fin 2 → Fin mPts → 𝔽, ¬ BL p.toStop.g' pts ∧
    RoundsReachK D M dg J B wr next (D.root p.toStop.modes) pts N (fun _ _ => [])
      (p.toStop.rewrite (M.link p.toStop pts)).consume o

open Classical in
variable {D M J B wr} in
theorem stopOffK_le_prob {ο : Type} (E : ο → Prop) (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽)
    (rs : Fin J → Fin 2 → List Flock.Round) (q : Pos g σ Claims.Msg (Fin R))
    (fin e : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞) (finW : τ → Game ο)
    (nextW : (s : SubmitJ J) → (Fin s.n → 𝔽) → Game ο) :
    ∀ (o : Option (SubmitJ J)) (w : Strategy (stepWK D J wr rootB pts rs q finW nextW o)),
      (o = none → ∀ w : Strategy (finW (wr ⟨rootB, pts, rs, D.files q⟩)),
        fin q ≤ prob E (finW (wr ⟨rootB, pts, rs, D.files q⟩)) w) →
      (∀ s, o = some s → ∀ c (w : Strategy (nextW s c)), ¬ B q.g' J s c →
        e (q.rewrite (M.round q J s c)).consume ≤ prob E (nextW s c) w) →
      stopOffK M J B fin e q o ≤ prob E (stepWK D J wr rootB pts rs q finW nextW o) w
  | none, w, h0, _ => h0 rfl w
  | some s, w, _, hs => avg_mono fun c => by
    split_ifs with hB
    · exact zero_le
    · exact hs s rfl c (w c) hB

variable {D M dg J B wr} in
theorem roundsOffK_le_prob {ο : Type} (E H : ο → Prop) (e : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞)
    (nextW : Pos g σ Claims.Msg (Fin R) → τ → Game ο) (nextR : Pos g σ Claims.Msg (Fin R) → τ → ο → Prop)
    (hn : ∀ q a (w : Strategy (nextW q a)), (∀ o, nextR q a o → H o → E o) → e q ≤ prob E (nextW q a) w)
    (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽) :
    ∀ (N : ℕ) (rs : Fin J → Fin 2 → List Flock.Round) (p : Pos g σ Claims.Msg (Fin R))
      (w : Strategy (roundsWK D M dg J wr nextW rootB pts N rs p)),
      (∀ o, RoundsReachK D M dg J B wr nextR rootB pts N rs p o → H o → E o) →
      roundsOffK D M J B e N p ≤ prob E (roundsWK D M dg J wr nextW rootB pts N rs p) w
  | 0, rs, p, w, h => hn p _ w fun o ho => h o (.zero rs p o ho)
  | N + 1, rs, p, w, h =>
    stopOffK_le_prob E rootB pts rs p.toStop e (roundsOffK D M J B e N) (nextW p.toStop) _ (D.sub J p.toStop.modes) w
      (fun h0 w => hn p.toStop _ w fun o ho => h o (.stop N rs p o h0 ho))
      (fun s hs c w hB => roundsOffK_le_prob E H e nextW nextR hn rootB pts N _ _ w
        fun o ho => h o (.round N rs p s c o hs hB ho))

open Classical in
variable {D M dg J B wr} in
theorem sessionOffK_le_prob {ο : Type} (BL : LinkHold mPts) (E H : ο → Prop)
    (e : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞) (nextW : Pos g σ Claims.Msg (Fin R) → τ → Game ο)
    (nextR : Pos g σ Claims.Msg (Fin R) → τ → ο → Prop)
    (hn : ∀ q a (w : Strategy (nextW q a)), (∀ o, nextR q a o → H o → E o) → e q ≤ prob E (nextW q a) w)
    (N : ℕ) (p : Pos g σ Claims.Msg (Fin R)) (w : Strategy (sessionWK D M dg J wr nextW N p))
    (h : ∀ o, SessionReachK D M dg J B wr BL nextR N p o → H o → E o) :
    sessionOffK D M J B BL e N p ≤ prob E (sessionWK D M dg J wr nextW N p) w :=
  avg_mono fun pts => by
    split_ifs with hL
    · exact zero_le
    · exact roundsOffK_le_prob E H e nextW nextR hn _ pts N _ _ (w pts) fun o ho => h o ⟨pts, hL, ho⟩

/-- **An idle session's outcome** at the link points `pts`: no root, no round and no proof file. It is the outcome
the projection gives the session of a member with no proof block in the sample; such a member has no statement, so
nothing reads it. -/
def idleOut (pts : Fin 2 → Fin mPts → 𝔽) : LiveOutJ J mPts :=
  ⟨#[], pts, fun _ _ => [], #[]⟩

/-- **An idle session's rounds, then what follows the session**: no submission and no proof file, then `fin` at the
session's transcript. It reads nothing of the period's play. -/
noncomputable def idleK (fin : (a : τ) → Strategy (K a)) (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽) :
    (N : ℕ) → (rs : Fin J → Fin 2 → List Flock.Round) → Strategy (roundsThen dg J wr K rootB pts N rs)
  | 0, rs => ⟨#[], fin (wr ⟨rootB, pts, rs, #[]⟩)⟩
  | _ + 1, rs => ⟨none, ⟨#[], fin (wr ⟨rootB, pts, rs, #[]⟩)⟩⟩

noncomputable def skipK (fin : (a : τ) → Strategy (K a)) (N : ℕ) :
    Strategy (((liveSession dg J mPts N).map wr).bind K) :=
  ⟨#[], fun pts => idleK dg J wr K fin #[] pts N (fun _ _ => [])⟩

noncomputable def skipWK {ο : Type} (fin : τ → Game ο) : Game ο :=
  .coin (Fin 2 → Fin mPts → 𝔽) fun pts => fin (wr (idleOut J pts))

theorem prob_idleK (E : γ → Prop) (fin : (a : τ) → Strategy (K a)) (rootB : Array ByteArray)
    (pts : Fin 2 → Fin mPts → 𝔽) :
    ∀ (N : ℕ) (rs : Fin J → Fin 2 → List Flock.Round),
      prob E (roundsThen dg J wr K rootB pts N rs) (idleK dg J wr K fin rootB pts N rs) =
        prob E (K (wr ⟨rootB, pts, rs, #[]⟩)) (fin (wr ⟨rootB, pts, rs, #[]⟩))
  | 0, _ => rfl
  | _ + 1, _ => rfl

theorem prob_skipK (E : γ → Prop) (finP : (a : τ) → Strategy (K a)) (finW : τ → Game (γ × CoinTape g σ))
    (h : ∀ a (w : Strategy (finW a)), prob E (K a) (finP a) = prob (fun o => E o.1) (finW a) w) (N : ℕ)
    (w : Strategy (skipWK J wr finW)) :
    prob E (((liveSession dg J mPts N).map wr).bind K) (skipK dg J wr K finP N) =
      prob (fun o => E o.1) (skipWK J wr finW) w :=
  Finset.sum_congr rfl fun pts _ => congrArg (HMul.hMul _)
    ((prob_idleK dg J wr K E finP #[] pts N fun _ _ => []).trans (h (wr (idleOut J pts)) (w pts)))

noncomputable def memberK (next : Pos g σ Claims.Msg (Fin R) → (a : τ) → Strategy (K a)) (N : ℕ)
    (p : Pos g σ Claims.Msg (Fin R)) : Bool → Strategy (((liveSession dg J mPts N).map wr).bind K)
  | true => sessionK D M dg J wr K (fun q => next q.toStop.consume) N p
  | false => skipK dg J wr K (next p) N

noncomputable def memberWK {ο : Type} (next : Pos g σ Claims.Msg (Fin R) → τ → Game ο) (N : ℕ)
    (p : Pos g σ Claims.Msg (Fin R)) : Bool → Game ο
  | true => sessionWK D M dg J wr (fun q => next q.toStop.consume) N p
  | false => skipWK J wr (next p)

noncomputable def memberWKS {ο : Type} (next : Pos g σ Claims.Msg (Fin R) → τ → Game ο)
    (wn : ∀ q a, Strategy (next q a)) (N : ℕ) (p : Pos g σ Claims.Msg (Fin R)) :
    (b : Bool) → Strategy (memberWK D M dg J wr next N p b)
  | true => sessionWKS D M dg J wr (fun q => next q.toStop.consume) (fun q => wn q.toStop.consume) N p
  | false => fun _ => wn p _

theorem prob_memberK (E : γ → Prop) (nextP : Pos g σ Claims.Msg (Fin R) → (a : τ) → Strategy (K a))
    (nextW : Pos g σ Claims.Msg (Fin R) → τ → Game (γ × CoinTape g σ))
    (h : ∀ q a (w : Strategy (nextW q a)), prob E (K a) (nextP q a) = prob (fun o => E o.1) (nextW q a) w)
    (N : ℕ) (p : Pos g σ Claims.Msg (Fin R)) :
    ∀ (b : Bool) (w : Strategy (memberWK D M dg J wr nextW N p b)),
      prob E (((liveSession dg J mPts N).map wr).bind K) (memberK D M dg J wr K nextP N p b) =
        prob (fun o => E o.1) (memberWK D M dg J wr nextW N p b) w
  | true, w => prob_sessionK D M dg J wr K E (fun q => nextP q.toStop.consume) (fun q => nextW q.toStop.consume)
      (fun q => h q.toStop.consume) N p w
  | false, w => prob_skipK dg J wr K E (nextP p) (nextW p) (h p) N w

noncomputable def memberOffK (BL : LinkHold mPts) (e : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞) (N : ℕ)
    (p : Pos g σ Claims.Msg (Fin R)) : Bool → ℝ≥0∞
  | true => sessionOffK D M J B BL (fun q => e q.toStop.consume) N p
  | false => e p

def MemberReachK {ο : Type} (BL : LinkHold mPts) (next : Pos g σ Claims.Msg (Fin R) → τ → ο → Prop) (N : ℕ)
    (p : Pos g σ Claims.Msg (Fin R)) (o : ο) : Bool → Prop
  | true => SessionReachK D M dg J B wr BL (fun q => next q.toStop.consume) N p o
  | false => ∃ pts : Fin 2 → Fin mPts → 𝔽, next p (wr (idleOut J pts)) o

variable {D M dg J B wr} in
theorem memberOffK_le_prob {ο : Type} (BL : LinkHold mPts) (E H : ο → Prop)
    (e : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞) (nextW : Pos g σ Claims.Msg (Fin R) → τ → Game ο)
    (nextR : Pos g σ Claims.Msg (Fin R) → τ → ο → Prop)
    (hn : ∀ q a (w : Strategy (nextW q a)), (∀ o, nextR q a o → H o → E o) → e q ≤ prob E (nextW q a) w)
    (N : ℕ) (p : Pos g σ Claims.Msg (Fin R)) :
    ∀ (b : Bool) (w : Strategy (memberWK D M dg J wr nextW N p b)),
      (∀ o, MemberReachK D M dg J B wr BL nextR N p o b → H o → E o) →
      memberOffK D M J B BL e N p b ≤ prob E (memberWK D M dg J wr nextW N p b) w
  | true, w, h => sessionOffK_le_prob BL E H (fun q => e q.toStop.consume) (fun q => nextW q.toStop.consume)
      (fun q => nextR q.toStop.consume) (fun q => hn q.toStop.consume) N p w h
  | false, w, h =>
    (avg_const (C := Fin 2 → Fin mPts → 𝔽) (e p)).symm.le.trans
      (avg_mono fun pts => hn p (wr (idleOut J pts)) (w pts) fun o ho => h o ⟨pts, ho⟩)

end SessionK

structure ExecutionPhaseCoinMap (g : Game Exchange) (σ : Strategy g) (Ω : Type) (R : ℕ) where
  T : Type
  [fT : Fintype T]
  [nT : Nonempty T]
  start : Ω × T → CoinTape g σ
  link : ℕ → (p : Pos g σ Claims.Msg (Fin R)) → (mPts : ℕ) → (Fin 2 → Fin mPts → 𝔽) → CoinTape p.g' p.s' →
    CoinTape p.g' p.s'
  round : ℕ → (p : Pos g σ Claims.Msg (Fin R)) → (J : ℕ) → (s : SubmitJ J) → (Fin s.n → 𝔽) → CoinTape p.g' p.s' →
    CoinTape p.g' p.s'

attribute [instance] ExecutionPhaseCoinMap.fT ExecutionPhaseCoinMap.nT

def ExecutionPhaseCoinMap.widening {g : Game Exchange} {σ : Strategy g} {Ω : Type} {R : ℕ} (M : ExecutionPhaseCoinMap g σ Ω R) :
    Widening :=
  ⟨M.T⟩

def ExecutionPhaseCoinMap.member {g : Game Exchange} {σ : Strategy g} {Ω : Type} {R : ℕ} (M : ExecutionPhaseCoinMap g σ Ω R)
    (i mPts : ℕ) : CoinMap g σ Ω R mPts where
  T := M.T
  fT := M.fT
  nT := M.nT
  start := M.start
  link p := M.link i p mPts
  round := M.round i

abbrev ExecutionPhaseLinkHold : Type 1 := ℕ → (mPts : ℕ) → LinkHold mPts

abbrev ExecutionPhaseRoundHold : Type 1 := ℕ → RoundHold

section Seq

open FlockSoundness.Discharge.Exec (Transcript mPtsOf)

variable {g : Game Exchange} {σ : Strategy g} {Ω Reg : Type} {R nJ : ℕ} (D : ℕ → PosDecoders g σ R Reg)
  (M : ExecutionPhaseCoinMap g σ Ω R) (BL : ExecutionPhaseLinkHold) (B : ExecutionPhaseRoundHold) (S : Finset (Fin nJ)) (Rg : Reg)

abbrev memberPts (m : Flock.Guarantees.MemberV5 Reg nJ) : ℕ :=
  mPtsOf m.D.unroll.g (m.kd / FlockSoundness.Discharge.Composed.nTab m.I)

noncomputable abbrev memberTabs (m : Flock.Guarantees.MemberV5 Reg nJ) : ℕ :=
  (FlockSoundness.Discharge.ZkHidden.planZHJP m.dj m.y₀ (m.part S) Rg).length

open Classical in
noncomputable def memberSampled (m : Flock.Guarantees.MemberV5 Reg nJ) : Bool :=
  decide (m.part S).Nonempty

open Classical in
variable {S} in
theorem memberSampled_iff {m : Flock.Guarantees.MemberV5 Reg nJ} : memberSampled S m = true ↔ (m.part S).Nonempty :=
  decide_eq_true_iff

variable {S} in
theorem not_nonempty_of_memberSampled {m : Flock.Guarantees.MemberV5 Reg nJ} (h : memberSampled S m = false) :
    ¬ (m.part S).Nonempty :=
  fun hne => Bool.false_ne_true (h.symm.trans (memberSampled_iff.2 hne))

noncomputable def seqP : (i : ℕ) → (l : List (Flock.Guarantees.MemberV5 Reg nJ)) → Pos g σ Claims.Msg (Fin R) →
    Strategy (FlockSoundness.Game.seqG (fun m : Flock.Guarantees.MemberV5 Reg nJ => m.live (m.part S) Rg) l)
  | _, [], _ => PUnit.unit
  | i, m :: l, p =>
    memberK (D i) (M.member i (memberPts m)) m.dg (memberTabs S Rg m) (m.wr (m.part S) Rg)
      (fun a => (FlockSoundness.Game.seqG
        (fun m : Flock.Guarantees.MemberV5 Reg nJ => m.live (m.part S) Rg) l).map (a :: ·))
      (fun q _ => Strategy.map _ _ (seqP (i + 1) l q)) m.N p (memberSampled S m)

noncomputable def seqW : (i : ℕ) → (l : List (Flock.Guarantees.MemberV5 Reg nJ)) → Pos g σ Claims.Msg (Fin R) →
    Game (List Transcript × CoinTape g σ)
  | _, [], p => .ret ([], p.tape)
  | i, m :: l, p =>
    memberWK (D i) (M.member i (memberPts m)) m.dg (memberTabs S Rg m) (m.wr (m.part S) Rg)
      (fun q a => (seqW (i + 1) l q).map fun o => (a :: o.1, o.2)) m.N p (memberSampled S m)

noncomputable def seqOff (Q : CoinTape g σ → Prop) :
    (i : ℕ) → (l : List (Flock.Guarantees.MemberV5 Reg nJ)) → Pos g σ Claims.Msg (Fin R) → ℝ≥0∞
  | _, [], p => tapeInd Q p
  | i, m :: l, p =>
    memberOffK (D i) (M.member i (memberPts m)) (memberTabs S Rg m) (B i) (BL i (memberPts m))
      (fun q => seqOff Q (i + 1) l q) m.N p (memberSampled S m)

def SeqReach : (i : ℕ) → (l : List (Flock.Guarantees.MemberV5 Reg nJ)) → Pos g σ Claims.Msg (Fin R) →
    List Transcript × CoinTape g σ → Prop
  | _, [], p, o => o = ([], p.tape)
  | i, m :: l, p, o =>
    MemberReachK (D i) (M.member i (memberPts m)) m.dg (memberTabs S Rg m) (B i) (m.wr (m.part S) Rg)
      (BL i (memberPts m)) (fun q a o' => ∃ ts, SeqReach (i + 1) l q (ts, o'.2) ∧ o'.1 = a :: ts) m.N p o
      (memberSampled S m)

noncomputable def seqWS : (i : ℕ) → (l : List (Flock.Guarantees.MemberV5 Reg nJ)) →
    (p : Pos g σ Claims.Msg (Fin R)) → Strategy (seqW D M S Rg i l p)
  | _, [], _ => PUnit.unit
  | i, m :: l, p =>
    memberWKS (D i) (M.member i (memberPts m)) m.dg (memberTabs S Rg m) (m.wr (m.part S) Rg)
      (fun q a => (seqW D M S Rg (i + 1) l q).map fun o => (a :: o.1, o.2))
      (fun q _ => Strategy.map _ _ (seqWS (i + 1) l q)) m.N p (memberSampled S m)

theorem prob_seqP : ∀ (l : List (Flock.Guarantees.MemberV5 Reg nJ)) (E : List Transcript → Prop) (i : ℕ)
    (p : Pos g σ Claims.Msg (Fin R)) (w : Strategy (seqW D M S Rg i l p)),
    prob E (FlockSoundness.Game.seqG (fun m : Flock.Guarantees.MemberV5 Reg nJ => m.live (m.part S) Rg) l)
        (seqP D M S Rg i l p) =
      prob (fun o => E o.1) (seqW D M S Rg i l p) w
  | [], _, _, _, _ => rfl
  | m :: l, E, i, p, w =>
    prob_memberK (D i) (M.member i (memberPts m)) m.dg (memberTabs S Rg m) (m.wr (m.part S) Rg)
      (fun a => (FlockSoundness.Game.seqG
        (fun m : Flock.Guarantees.MemberV5 Reg nJ => m.live (m.part S) Rg) l).map (a :: ·)) E
      (fun q _ => Strategy.map _ _ (seqP D M S Rg (i + 1) l q))
      (fun q a => (seqW D M S Rg (i + 1) l q).map fun o => (a :: o.1, o.2))
      (fun q a w => (FlockSoundness.Discharge.Exec.prob_strategy_map _ _ _ _).trans
        ((prob_seqP l (fun ts => E (a :: ts)) (i + 1) q (Strategy.ofMap _ _ w)).trans
          (prob_map (fun o : List Transcript × CoinTape g σ => E o.1)
            (fun o : List Transcript × CoinTape g σ => (a :: o.1, o.2)) _ w).symm))
      m.N p (memberSampled S m) w

open Classical in
theorem seqOff_le_prob (Q : CoinTape g σ → Prop) : ∀ (l : List (Flock.Guarantees.MemberV5 Reg nJ))
    (E : List Transcript → Prop) (i : ℕ) (p : Pos g σ Claims.Msg (Fin R)) (w : Strategy (seqW D M S Rg i l p)),
    (∀ o, SeqReach D M BL B S Rg i l p o → Q o.2 → E o.1) →
    seqOff D M BL B S Rg Q i l p ≤ prob (fun o => E o.1) (seqW D M S Rg i l p) w
  | [], E, _, p, _, h => by
    show tapeInd Q p ≤ if E [] then 1 else 0
    unfold tapeInd
    by_cases hQ : Q p.tape
    · rw [ite_eq_left hQ, ite_eq_left (h ([], p.tape) rfl hQ)]
    · rw [ite_eq_right hQ]
      exact zero_le
  | m :: l, E, i, p, w, h =>
    memberOffK_le_prob (D := D i) (M := M.member i (memberPts m)) (dg := m.dg) (J := memberTabs S Rg m) (B := B i)
      (wr := m.wr (m.part S) Rg) (BL i (memberPts m)) (fun o => E o.1) (fun o => Q o.2)
      (fun q => seqOff D M BL B S Rg Q (i + 1) l q)
      (fun q a => (seqW D M S Rg (i + 1) l q).map fun o => (a :: o.1, o.2))
      (fun q a o' => ∃ ts, SeqReach D M BL B S Rg (i + 1) l q (ts, o'.2) ∧ o'.1 = a :: ts)
      (fun q a w hq => (seqOff_le_prob Q l (fun ts => E (a :: ts)) (i + 1) q (Strategy.ofMap _ _ w)
          fun o ho hQ => hq (a :: o.1, o.2) ⟨o.1, ho, rfl⟩ hQ).trans_eq
        (prob_map (fun o : List Transcript × CoinTape g σ => E o.1)
            (fun o : List Transcript × CoinTape g σ => (a :: o.1, o.2)) _ w).symm)
      m.N p (memberSampled S m) w h

end Seq

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

variable (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)

abbrev ExecutionPhaseDecoders (W : ExecutionPhase) (σ : Strategy (V.game θ F c)) : Type 1 :=
  ℕ → PosDecoders (V.game θ F c) σ W.rounds ByteArray

noncomputable def executionPhaseReg (W : ExecutionPhase) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (D : ExecutionPhaseDecoders V θ F c W σ) : ByteArray :=
  (D 0).reg ((recordCut V θ F c rd).cut.modeHere σ)

noncomputable abbrev executionPhaseOut (W : ExecutionPhase) (t : ℕ) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (D : ExecutionPhaseDecoders V θ F c W σ) (M : ExecutionPhaseCoinMap (V.game θ F c) σ W.Ω W.rounds)
    (ω : (W.extend M.widening).L.Ω) (ts : List FlockSoundness.Discharge.Exec.Transcript) : W.Out :=
  ((executionPhaseReg V θ F c W rd σ D, ((W.extend M.widening).joint t).ended ω), (W.extend M.widening).L.draw ω, ts)

noncomputable def executionPhaseP (W : ExecutionPhase) (t : ℕ) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (D : ExecutionPhaseDecoders V θ F c W σ) (M : ExecutionPhaseCoinMap (V.game θ F c) σ W.Ω W.rounds) :
    FlockSoundness.Game.Strategy ((W.extend M.widening).joint t).game :=
  ⟨executionPhaseReg V θ F c W rd σ D, fun ω =>
    FlockSoundness.Game.Strategy.map _ _
      (seqP D M ((W.extend M.widening).L.draw ω) (executionPhaseReg V θ F c W rd σ D) 0 ((W.extend M.widening).joint t).ms
        (Pos.root (roundCuts V θ F c rd) (M.start ω)))⟩

noncomputable def executionPhaseG (W : ExecutionPhase) (t : ℕ) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (D : ExecutionPhaseDecoders V θ F c W σ) (M : ExecutionPhaseCoinMap (V.game θ F c) σ W.Ω W.rounds) :
    Game (W.Out × CoinTape (V.game θ F c) σ) :=
  .coin (W.extend M.widening).L.Ω fun ω =>
    (seqW D M ((W.extend M.widening).L.draw ω) (executionPhaseReg V θ F c W rd σ D) 0 ((W.extend M.widening).joint t).ms
      (Pos.root (roundCuts V θ F c rd) (M.start ω))).map
      fun o => (executionPhaseOut V θ F c W t rd σ D M ω o.1, o.2)

theorem prob_executionPhaseP (W : ExecutionPhase) (t : ℕ) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (D : ExecutionPhaseDecoders V θ F c W σ) (M : ExecutionPhaseCoinMap (V.game θ F c) σ W.Ω W.rounds) (E : W.Out → Prop)
    (w : Strategy (executionPhaseG V θ F c W t rd σ D M)) :
    FlockSoundness.Game.prob E ((W.extend M.widening).joint t).game (executionPhaseP V θ F c W t rd σ D M) =
      prob (fun o => E o.1) (executionPhaseG V θ F c W t rd σ D M) w := by
  refine Finset.sum_congr rfl fun ω _ => congrArg (HMul.hMul _) ?_
  exact (FlockSoundness.Discharge.Exec.prob_strategy_map _ _ _ _).trans
    ((prob_seqP D M ((W.extend M.widening).L.draw ω) (executionPhaseReg V θ F c W rd σ D)
        ((W.extend M.widening).joint t).ms
        (fun ts => E (executionPhaseOut V θ F c W t rd σ D M ω ts)) 0
        (Pos.root (roundCuts V θ F c rd) (M.start ω)) (Strategy.ofMap _ _ (w ω))).trans
      (prob_map (fun o : W.Out × CoinTape (V.game θ F c) σ => E o.1)
        (fun o : List FlockSoundness.Discharge.Exec.Transcript × CoinTape (V.game θ F c) σ =>
          (executionPhaseOut V θ F c W t rd σ D M ω o.1, o.2)) _ (w ω)).symm)

noncomputable def executionPhaseGS (W : ExecutionPhase) (t : ℕ) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (D : ExecutionPhaseDecoders V θ F c W σ) (M : ExecutionPhaseCoinMap (V.game θ F c) σ W.Ω W.rounds) :
    Strategy (executionPhaseG V θ F c W t rd σ D M) :=
  fun ω => Strategy.map _ _ (seqWS D M ((W.extend M.widening).L.draw ω) (executionPhaseReg V θ F c W rd σ D) 0
    ((W.extend M.widening).joint t).ms (Pos.root (roundCuts V θ F c rd) (M.start ω)))

def ExecutionPhaseReaches (W : ExecutionPhase) (t : ℕ) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (D : ExecutionPhaseDecoders V θ F c W σ) (M : ExecutionPhaseCoinMap (V.game θ F c) σ W.Ω W.rounds) (BL : ExecutionPhaseLinkHold)
    (B : ExecutionPhaseRoundHold) (o : W.Out × CoinTape (V.game θ F c) σ) : Prop :=
  ∃ ω : (W.extend M.widening).L.Ω,
    o.1.1 = (executionPhaseReg V θ F c W rd σ D, ((W.extend M.widening).joint t).ended ω) ∧
    o.1.2.1 = (W.extend M.widening).L.draw ω ∧
    SeqReach D M BL B ((W.extend M.widening).L.draw ω) (executionPhaseReg V θ F c W rd σ D) 0
      ((W.extend M.widening).joint t).ms (Pos.root (roundCuts V θ F c rd) (M.start ω)) (o.1.2.2, o.2)

noncomputable def executionPhaseOff (W : ExecutionPhase) (t : ℕ) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (D : ExecutionPhaseDecoders V θ F c W σ) (M : ExecutionPhaseCoinMap (V.game θ F c) σ W.Ω W.rounds) (BL : ExecutionPhaseLinkHold)
    (B : ExecutionPhaseRoundHold) (Q : CoinTape (V.game θ F c) σ → Prop) : ℝ≥0∞ :=
  avg fun ω : (W.extend M.widening).L.Ω =>
    seqOff D M BL B ((W.extend M.widening).L.draw ω) (executionPhaseReg V θ F c W rd σ D) Q 0
      ((W.extend M.widening).joint t).ms (Pos.root (roundCuts V θ F c rd) (M.start ω))

theorem executionPhaseOff_le_prob (W : ExecutionPhase) (t : ℕ) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (D : ExecutionPhaseDecoders V θ F c W σ) (M : ExecutionPhaseCoinMap (V.game θ F c) σ W.Ω W.rounds) (BL : ExecutionPhaseLinkHold)
    (B : ExecutionPhaseRoundHold) (Q : CoinTape (V.game θ F c) σ → Prop) (E : W.Out → Prop)
    (h : ∀ o, ExecutionPhaseReaches V θ F c W t rd σ D M BL B o → Q o.2 → E o.1)
    (w : Strategy (executionPhaseG V θ F c W t rd σ D M)) :
    executionPhaseOff V θ F c W t rd σ D M BL B Q ≤ prob (fun o => E o.1) (executionPhaseG V θ F c W t rd σ D M) w :=
  avg_mono fun ω =>
    (seqOff_le_prob D M BL B ((W.extend M.widening).L.draw ω) (executionPhaseReg V θ F c W rd σ D) Q
        ((W.extend M.widening).joint t).ms
        (fun ts => E (executionPhaseOut V θ F c W t rd σ D M ω ts)) 0
        (Pos.root (roundCuts V θ F c rd) (M.start ω)) (Strategy.ofMap _ _ (w ω))
        fun o ho hQ => h (executionPhaseOut V θ F c W t rd σ D M ω o.1, o.2)
          ⟨ω, rfl, rfl, ho⟩ hQ).trans_eq
      (prob_map (fun o : W.Out × CoinTape (V.game θ F c) σ => E o.1)
        (fun o : List FlockSoundness.Discharge.Exec.Transcript × CoinTape (V.game θ F c) σ =>
          (executionPhaseOut V θ F c W t rd σ D M ω o.1, o.2)) _ (w ω)).symm

def ExecutionPhaseCouples (W : ExecutionPhase) (t : ℕ) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (D : ExecutionPhaseDecoders V θ F c W σ) (M : ExecutionPhaseCoinMap (V.game θ F c) σ W.Ω W.rounds) (BL : ExecutionPhaseLinkHold)
    (B : ExecutionPhaseRoundHold) : Prop :=
  ∀ Q : CoinTape (V.game θ F c) σ → Prop, ENNReal.ofReal (Claims.Pr Q) ≤
    executionPhaseOff V θ F c W t rd σ D M BL B Q + W.hidingTerm

def ExecutionPhaseDecodes (k : Verity.Registration.Composite) (P : Exchange → Prop) (W : ExecutionPhase) (t : ℕ)
    (rd : Reading W.rounds) (σ : Strategy (V.game θ F c)) (D : ExecutionPhaseDecoders V θ F c W σ)
    (M : ExecutionPhaseCoinMap (V.game θ F c) σ W.Ω W.rounds) (BL : ExecutionPhaseLinkHold) (B : ExecutionPhaseRoundHold) : Prop :=
  ∀ o, ExecutionPhaseReaches V θ F c W t rd σ D M BL B o →
    V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ o.2) ∧ NoMiss V θ F c rd σ o.2 ∧
      P (CoinTape.run (V.game θ F c) σ o.2) →
    ((W.extend M.widening).joint t).Accepts o.1

theorem executionPhaseLogged (k : Verity.Registration.Composite) (P : Exchange → Prop) (W : ExecutionPhase) (t : ℕ)
    (rd : Reading W.rounds) (σ : Strategy (V.game θ F c)) (D : ExecutionPhaseDecoders V θ F c W σ)
    (M : ExecutionPhaseCoinMap (V.game θ F c) σ W.Ω W.rounds) (BL : ExecutionPhaseLinkHold) (B : ExecutionPhaseRoundHold)
    (hC : ExecutionPhaseCouples V θ F c W t rd σ D M BL B) (hD : ExecutionPhaseDecodes V θ F c k P W t rd σ D M BL B) :
    ENNReal.ofReal (Claims.Pr fun x : CoinTape (V.game θ F c) σ =>
        V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) ∧ NoMiss V θ F c rd σ x ∧
          P (CoinTape.run (V.game θ F c) σ x)) ≤
      FlockSoundness.Game.prob (fun o => ((W.extend M.widening).joint t).Accepts o)
          ((W.extend M.widening).joint t).game (executionPhaseP V θ F c W t rd σ D M) +
        W.hidingTerm := by
  have h := (hC fun x => V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) ∧ NoMiss V θ F c rd σ x ∧
      P (CoinTape.run (V.game θ F c) σ x)).trans
    (add_le_add (executionPhaseOff_le_prob V θ F c W t rd σ D M BL B
      (fun x => V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) ∧ NoMiss V θ F c rd σ x ∧
        P (CoinTape.run (V.game θ F c) σ x))
      (fun o => ((W.extend M.widening).joint t).Accepts o) hD (executionPhaseGS V θ F c W t rd σ D M)) le_rfl)
  rw [← prob_executionPhaseP V θ F c W t rd σ D M (fun o => ((W.extend M.widening).joint t).Accepts o)] at h
  exact h

abbrev ExecutionPhaseReadings := (W : ExecutionPhase) → ByteArray → Array ByteArray → ByteArray → Reading W.rounds

noncomputable def executionPhaseMsgFinder (rd : ExecutionPhaseReadings) (W : ExecutionPhase) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Strategy (V.game θ F c)) (j : Fin W.rounds) : Claims.Finder :=
  (roundCut V θ F c (rd W θ F c) j).cut.finder σ (2 * V.cost θ F c σ + 2 * W.h)

noncomputable def executionPhaseRecordFinder (rd : ExecutionPhaseReadings) (W : ExecutionPhase) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Strategy (V.game θ F c)) : Claims.Finder :=
  (recordCut V θ F c (rd W θ F c)).cut.finder σ (2 * V.cost θ F c σ + 2 * W.h)

omit θ F c in
theorem executionPhaseMsgFinder_calls (rd : ExecutionPhaseReadings) : ∀ (W : ExecutionPhase) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Strategy (V.game θ F c)) (j : Fin W.rounds) (x : (executionPhaseMsgFinder V rd W θ F c σ j).S),
    ((executionPhaseMsgFinder V rd W θ F c σ j).cost x : ℝ) ≤ W.qM (V.cost θ F c σ) := fun W θ F c σ j _ => by
  show ((2 * V.cost θ F c σ + 2 * W.h : ℕ) : ℝ) ≤ W.qM (V.cost θ F c σ)
  simp [ExecutionPhase.qM]

omit θ F c in
theorem executionPhaseRecordFinder_calls (rd : ExecutionPhaseReadings) : ∀ (W : ExecutionPhase) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Strategy (V.game θ F c)) (x : (executionPhaseRecordFinder V rd W θ F c σ).S),
    ((executionPhaseRecordFinder V rd W θ F c σ).cost x : ℝ) ≤ W.qM (V.cost θ F c σ) := fun W θ F c σ _ => by
  show ((2 * V.cost θ F c σ + 2 * W.h : ℕ) : ℝ) ≤ W.qM (V.cost θ F c σ)
  simp [ExecutionPhase.qM]

noncomputable def ExecutionPhaseConstructions.withProject (K : ExecutionPhaseConstructions V) (rd : ExecutionPhaseReadings)
    (D : (W : ExecutionPhase) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → (σ : Strategy (V.game θ F c)) →
      ExecutionPhaseDecoders V θ F c W σ)
    (M : (W : ExecutionPhase) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → (σ : Strategy (V.game θ F c)) →
      ExecutionPhaseCoinMap (V.game θ F c) σ W.Ω W.rounds) : ExecutionPhaseConstructions V :=
  { K with
    coins := fun W θ F c σ => (M W θ F c σ).widening
    project := fun W θ F c σ => executionPhaseP V θ F c W (V.cost θ F c σ) (rd W θ F c) σ (D W θ F c σ) (M W θ F c σ)
    msgFinder := executionPhaseMsgFinder V rd
    recordFinder := executionPhaseRecordFinder V rd
    msg_calls := executionPhaseMsgFinder_calls V rd
    record_calls := executionPhaseRecordFinder_calls V rd }

variable (k : Verity.Registration.Composite)

omit θ F c in
theorem prob_le_executionPhaseMessages (rd : ExecutionPhaseReadings) (K : ExecutionPhaseConstructions V)
    (hM : K.msgFinder = executionPhaseMsgFinder V rd) (hR : K.recordFinder = executionPhaseRecordFinder V rd) (W : ExecutionPhase)
    (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Strategy (V.game θ F c)) (E : Exchange → Prop) :
    prob E (V.game θ F c) σ ≤
      ENNReal.ofReal (Claims.Pr fun x : CoinTape (V.game θ F c) σ =>
          E (CoinTape.run (V.game θ F c) σ x) ∧ NoMiss V θ F c (rd W θ F c) σ x) +
        (∑ j, ENNReal.ofReal (Real.sqrt ((K.msgFinder W θ F c σ j).Pr sha512Msg)) +
          ENNReal.ofReal (Real.sqrt ((K.recordFinder W θ F c σ).Pr sha512Msg))) := by
  rw [hM, hR]
  exact prob_le_noMiss_add V θ F c (rd W θ F c) σ _ E

omit θ F c in
theorem executionPhaseCouplingTarget_of_noMiss (rd : ExecutionPhaseReadings) (K : ExecutionPhaseConstructions V)
    (hM : K.msgFinder = executionPhaseMsgFinder V rd) (hR : K.recordFinder = executionPhaseRecordFinder V rd) (W : ExecutionPhase)
    (θ : ByteArray) (F : Array ByteArray)
    (h : ∀ (c : ByteArray) (σ : Strategy (V.game θ F c)),
      ENNReal.ofReal (Claims.Pr fun x : CoinTape (V.game θ F c) σ =>
          V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) ∧ NoMiss V θ F c (rd W θ F c) σ x) ≤
        FlockSoundness.Game.prob (fun o => ((K.live W θ F c σ).joint (V.cost θ F c σ)).Accepts o)
            ((K.live W θ F c σ).joint (V.cost θ F c σ)).game (K.project W θ F c σ) +
          (ENNReal.ofReal (Real.sqrt ((K.bindFinder W θ F c σ).Pr sha512Msg)) + W.hidingTerm +
            ENNReal.ofReal ((K.readFinder θ F c σ).Pr sha512Msg) +
            ENNReal.ofReal ((K.forger θ F c σ).Pr Verity.lmsVerify))) :
    ExecutionPhaseCouplingTarget V k K W θ F := fun c σ =>
  (prob_le_executionPhaseMessages V rd K hM hR W θ F c σ _).trans <| (add_le_add (h c σ) le_rfl).trans_eq (by ring)

omit θ F c in
theorem executionPhaseNoMiss_le_of_logged (rd : ExecutionPhaseReadings) (K : ExecutionPhaseConstructions V) (W : ExecutionPhase)
    (P : Exchange → Prop) (θ : ByteArray) (F : Array ByteArray)
    (Q : (c : ByteArray) → Strategy (V.game θ F c) → Exchange → Prop)
    (ha : ∀ (c : ByteArray) (σ : Strategy (V.game θ F c)),
      prob (fun o => V.Verdict k .sampled o ∧ ¬ P o) (V.game θ F c) σ ≤
        ENNReal.ofReal ((K.readFinder θ F c σ).Pr sha512Msg) +
          ENNReal.ofReal ((K.forger θ F c σ).Pr Verity.lmsVerify))
    (hq : ∀ (c : ByteArray) (σ : Strategy (V.game θ F c)),
      prob (fun o => V.Verdict k .sampled o ∧ ¬ Q c σ o) (V.game θ F c) σ ≤
        ENNReal.ofReal (Real.sqrt ((K.bindFinder W θ F c σ).Pr sha512Msg)))
    (hb : ∀ (c : ByteArray) (σ : Strategy (V.game θ F c)),
      ENNReal.ofReal (Claims.Pr fun x : CoinTape (V.game θ F c) σ =>
          V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) ∧ NoMiss V θ F c (rd W θ F c) σ x ∧
            (P (CoinTape.run (V.game θ F c) σ x) ∧ Q c σ (CoinTape.run (V.game θ F c) σ x))) ≤
        FlockSoundness.Game.prob (fun o => ((K.live W θ F c σ).joint (V.cost θ F c σ)).Accepts o)
            ((K.live W θ F c σ).joint (V.cost θ F c σ)).game (K.project W θ F c σ) +
          W.hidingTerm)
    (c : ByteArray) (σ : Strategy (V.game θ F c)) :
    ENNReal.ofReal (Claims.Pr fun x : CoinTape (V.game θ F c) σ =>
        V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) ∧ NoMiss V θ F c (rd W θ F c) σ x) ≤
      FlockSoundness.Game.prob (fun o => ((K.live W θ F c σ).joint (V.cost θ F c σ)).Accepts o)
          ((K.live W θ F c σ).joint (V.cost θ F c σ)).game (K.project W θ F c σ) +
        (ENNReal.ofReal (Real.sqrt ((K.bindFinder W θ F c σ).Pr sha512Msg)) + W.hidingTerm +
          ENNReal.ofReal ((K.readFinder θ F c σ).Pr sha512Msg) +
          ENNReal.ofReal ((K.forger θ F c σ).Pr Verity.lmsVerify)) := by
  have hsplit : Claims.Pr (fun x : CoinTape (V.game θ F c) σ =>
        V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) ∧ NoMiss V θ F c (rd W θ F c) σ x) ≤
      Claims.Pr (fun x : CoinTape (V.game θ F c) σ =>
          V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) ∧ NoMiss V θ F c (rd W θ F c) σ x ∧
            (P (CoinTape.run (V.game θ F c) σ x) ∧ Q c σ (CoinTape.run (V.game θ F c) σ x))) +
        (Claims.Pr (fun x : CoinTape (V.game θ F c) σ =>
            V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) ∧ ¬ P (CoinTape.run (V.game θ F c) σ x)) +
          Claims.Pr (fun x : CoinTape (V.game θ F c) σ =>
            V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) ∧
              ¬ Q c σ (CoinTape.run (V.game θ F c) σ x))) := by
    refine (Claims.Pr_mono fun x hx => ?_).trans ((pr_or_le _ _).trans (add_le_add le_rfl (pr_or_le _ _)))
    by_cases hp : P (CoinTape.run (V.game θ F c) σ x)
    · by_cases hQ : Q c σ (CoinTape.run (V.game θ F c) σ x)
      · exact Or.inl ⟨hx.1, hx.2, hp, hQ⟩
      · exact Or.inr (Or.inr ⟨hx.1, hQ⟩)
    · exact Or.inr (Or.inl ⟨hx.1, hp⟩)
  have hnot := ha c σ
  rw [prob_eq_ofReal_pr] at hnot
  have hnq := hq c σ
  rw [prob_eq_ofReal_pr] at hnq
  refine ((ENNReal.ofReal_le_ofReal hsplit).trans
    (ENNReal.ofReal_add_le.trans (add_le_add le_rfl ENNReal.ofReal_add_le))).trans ?_
  refine (add_le_add (hb c σ) (add_le_add hnot hnq)).trans_eq ?_
  ring

omit θ F c in
theorem executionPhaseCouplingTarget_of_logged (rd : ExecutionPhaseReadings) (K : ExecutionPhaseConstructions V)
    (D : (W : ExecutionPhase) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → (σ : Strategy (V.game θ F c)) →
      ExecutionPhaseDecoders V θ F c W σ)
    (M : (W : ExecutionPhase) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → (σ : Strategy (V.game θ F c)) →
      ExecutionPhaseCoinMap (V.game θ F c) σ W.Ω W.rounds)
    (W : ExecutionPhase) (P : Exchange → Prop) (θ : ByteArray) (F : Array ByteArray)
    (Q : (c : ByteArray) → Strategy (V.game θ F c) → Exchange → Prop)
    (BL : (c : ByteArray) → Strategy (V.game θ F c) → ExecutionPhaseLinkHold)
    (B : (c : ByteArray) → Strategy (V.game θ F c) → ExecutionPhaseRoundHold)
    (ha : ∀ (c : ByteArray) (σ : Strategy (V.game θ F c)),
      prob (fun o => V.Verdict k .sampled o ∧ ¬ P o) (V.game θ F c) σ ≤
        ENNReal.ofReal ((K.readFinder θ F c σ).Pr sha512Msg) +
          ENNReal.ofReal ((K.forger θ F c σ).Pr Verity.lmsVerify))
    (hq : ∀ (c : ByteArray) (σ : Strategy (V.game θ F c)),
      prob (fun o => V.Verdict k .sampled o ∧ ¬ Q c σ o) (V.game θ F c) σ ≤
        ENNReal.ofReal (Real.sqrt ((K.bindFinder W θ F c σ).Pr sha512Msg)))
    (hC : ∀ c σ, ExecutionPhaseCouples V θ F c W (V.cost θ F c σ) (rd W θ F c) σ (D W θ F c σ) (M W θ F c σ) (BL c σ)
      (B c σ))
    (hD : ∀ c σ, ExecutionPhaseDecodes V θ F c k (fun o => P o ∧ Q c σ o) W (V.cost θ F c σ) (rd W θ F c) σ (D W θ F c σ)
      (M W θ F c σ) (BL c σ) (B c σ)) :
    ExecutionPhaseCouplingTarget V k (K.withProject V rd D M) W θ F :=
  executionPhaseCouplingTarget_of_noMiss V k rd (K.withProject V rd D M) rfl rfl W θ F
    (executionPhaseNoMiss_le_of_logged V k rd (K.withProject V rd D M) W P θ F Q ha hq fun c σ =>
      executionPhaseLogged V θ F c k (fun o => P o ∧ Q c σ o) W (V.cost θ F c σ) (rd W θ F c) σ (D W θ F c σ)
        (M W θ F c σ) (BL c σ) (B c σ) (hC c σ) (hD c σ))

end Verity.Sampled
