module

public import Definitions.Claims.SamplingTable

/-! At a statement of one table, `Flock.KSTable` at table `0` and `Flock.KS` are one bound. -/

@[expose] public section

namespace Claims.Flock

open Claims

theorem mapM_range_one {α : Type} (f : ℕ → Option α) : (List.range 1).mapM f = (f 0).map fun a => [a] := by
  rw [List.range_one]
  cases h : f 0 with
  | none =>
    simp [h]
  | some a =>
    simp [h]

theorem kTable_zero_eq {Coin Col T : Type} (D : Claims.Flock.Deploy Coin Col T) (N : Msg → ℕ → ℕ) (x : Msg)
    (hN : (N x 0 : ℝ) = D.N₀ x) : Claims.Flock.kTable N 0 x = D.k x := by
  unfold Claims.Flock.kTable Claims.Flock.Deploy.k
  rw [hN]

theorem κTable_zero_eq {Coin Col T : Type} (D : Claims.Flock.Deploy Coin Col T) (N : Msg → ℕ → ℕ) (ε : Msg → ℕ → ℝ)
    (x : Msg) (hε : ε x 0 = D.εtable x) (hN : (N x 0 : ℝ) = D.N₀ x) :
    Claims.Flock.κTable D N ε 0 x = Claims.Flock.κ D x := by
  unfold Claims.Flock.κTable Claims.Flock.κ Claims.Flock.kTable Claims.Flock.Deploy.k
  rw [hε, hN]

theorem RTable_zero_iff_R (parse : Msg → Option Stmt) (table : Msg → ℕ → Option Stmt) (x : Msg) {s : Stmt}
    (hs : parse x = some s) (ht : table x 0 = some s) (w : Wit) :
    Claims.Flock.RTable table 0 x w ↔ Claims.Flock.R parse x w := by
  constructor
  · rintro ⟨t, h, hR⟩
    have e : s = t := Option.some.inj (ht.symm.trans h)
    exact ⟨s, hs, e ▸ hR⟩
  · rintro ⟨s', h, hR⟩
    have e : s = s' := Option.some.inj (hs.symm.trans h)
    exact ⟨s, ht, e ▸ hR⟩

theorem checkTable_zero_eq (parse : Msg → Option Stmt) (table : Msg → ℕ → Option Stmt) (x : Msg) {s : Stmt}
    (hs : parse x = some s) (ht : table x 0 = some s) (w : Wit) :
    Claims.Flock.checkTable table 0 x w = Claims.Flock.check parse x w := by
  unfold Claims.Flock.checkTable Claims.Flock.check
  rw [hs, ht]

theorem assembleTable_zero_eq {Col T : Type} (parse : Msg → Option Stmt) (table : Msg → ℕ → Option Stmt)
    (decode : List (ℕ × Col) → List T) (read : List T → Wit) (x : Msg) {s : Stmt} (hs : parse x = some s)
    (h1 : s.tables = 1) (opens : List (ℕ → List (ℕ × Col))) :
    Claims.Flock.assembleTable table decode read 0 x opens = Claims.Flock.assemble parse table decode read x opens := by
  unfold Claims.Flock.assembleTable Claims.Flock.assemble
  rw [hs]
  dsimp only
  rw [h1, mapM_range_one, Option.map_map]
  all_goals rfl

theorem pr_congr {α : Type} [Fintype α] {p q : α → Prop} (h : ∀ a, p a ↔ q a) : Pr p = Pr q := by
  have e : p = q := funext fun a => propext (h a)
  rw [e]

theorem ERep_ext_eq {Coin W Op : Type} [Fintype Coin] [Nonempty Coin] (V : Verifier Coin) (m k K : Msg → ℕ)
    (opens : Seen → List Op) (a : Msg → List Op → Option W) (c : Msg → W → Bool) (x : Msg)
    (r₀ : Fin (V.k x) → Coin) (ρ : Fin (m x) → Fin (K x) → Fin (V.k x) → Coin) :
    (ERep V m k K opens a c).ext x r₀ ρ =
      RP.rerun r₀ fun v₀ =>
        if v₀.out = .acc then
          firstWait (k x) opens (a x) (c x) ((List.finRange (m x)).map fun i => (List.finRange (K x)).map (ρ i))
        else RP.ret none :=
  rfl

theorem ERep_eval_congr {Coin W Op Ω : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg)
    (P : Prover Ω) (V : Verifier Coin) (m k k' K : Msg → ℕ) (opens : Seen → List Op)
    (a a' : Msg → List Op → Option W) (c c' : Msg → W → Bool) (x : Msg) (hk : k x = k' x) (ha : a x = a' x)
    (hc : c x = c' x) (ω : Ω) (r₀ : Fin (V.k x) → Coin) (ρ : Fin (m x) → Fin (K x) → Fin (V.k x) → Coin) :
    (ERep V m k K opens a c).eval H P x ω r₀ ρ = (ERep V m k' K opens a' c').eval H P x ω r₀ ρ := by
  unfold Extractor.eval
  rw [ERep_ext_eq, ERep_ext_eq, hk, ha, hc]

theorem prBad_ERep_congr {Coin W Op : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg)
    (V : Verifier Coin) (R R' : Rel W) (m k k' K : Msg → ℕ) (opens : Seen → List Op)
    (a a' : Msg → List Op → Option W) (c c' : Msg → W → Bool) (x : Msg) (hk : k x = k' x) (ha : a x = a' x)
    (hc : c x = c' x) (hR : ∀ w, R x w ↔ R' x w) {Ω : Type} [Fintype Ω] (P : Prover Ω) :
    PrBad H V R (ERep V m k K opens a c) x P = PrBad H V R' (ERep V m k' K opens a' c') x P := by
  have hext : ∀ o : Option W, R.ext x o ↔ R'.ext x o := by
    intro o
    cases o with
    | none => exact Iff.rfl
    | some w => exact hR w
  unfold PrBad
  refine pr_congr fun s => ?_
  have h := ERep_eval_congr H P V m k k' K opens a a' c c' x hk ha hc s.1 s.2.1 s.2.2
  constructor
  · rintro ⟨hacc, hno⟩
    refine ⟨hacc, fun hw => hno ((hext _).2 ?_)⟩
    exact h ▸ hw
  · rintro ⟨hacc, hno⟩
    refine ⟨hacc, fun hw => hno ((hext _).1 ?_)⟩
    exact h.symm ▸ hw

section OneTable

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

omit [DecidableEq Col] in
theorem prBad_zero_eq (H : Msg → Msg) (D : Claims.Flock.Deploy Coin Col T) (N : Msg → ℕ → ℕ)
    (decode : List (ℕ × Col) → List T) (x : Msg) {s : Stmt} (hs : D.parse x = some s) (h1 : s.tables = 1)
    (ht : D.table x 0 = some s) (hN : (N x 0 : ℝ) = D.N₀ x) {Ω : Type} [Fintype Ω] (P : Prover Ω) :
    PrBad H D.V (Claims.Flock.RTable D.table 0) (Claims.Flock.ETable D N decode 0) x P =
      PrBad H D.V (Claims.Flock.R D.parse) (Claims.Flock.E D decode) x P :=
  prBad_ERep_congr H D.V (Claims.Flock.RTable D.table 0) (Claims.Flock.R D.parse) D.waits (Claims.Flock.kTable N 0) D.k
    D.cap (fun v => [D.cols v]) (Claims.Flock.assembleTable D.table decode D.read 0)
    (Claims.Flock.assemble D.parse D.table decode D.read) (Claims.Flock.checkTable D.table 0)
    (Claims.Flock.check D.parse) x (kTable_zero_eq D N x hN)
    (funext fun opens => assembleTable_zero_eq D.parse D.table decode D.read x hs h1 opens)
    (funext fun w => checkTable_zero_eq D.parse D.table x hs ht w)
    (fun w => RTable_zero_iff_R D.parse D.table x hs ht w) P

theorem ksTable_one_table [Zero Col] (H : Msg → Msg) (D : Claims.Flock.Deploy Coin Col T) (N : Msg → ℕ → ℕ)
    (ε : Msg → ℕ → ℝ) (hKS : Claims.Flock.KSTable H D N ε) (decode : List (ℕ × Col) → List T)
    (hlw : Claims.Flock.ListWithin D.len D.enc D.θ decode) (x : Msg) {s : Stmt} (hs : D.parse x = some s)
    (h1 : s.tables = 1) (ht : D.table x 0 = some s) (hN : (N x 0 : ℝ) = D.N₀ x) (hε : ε x 0 = D.εtable x)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) :
    PrBad H D.V (Claims.Flock.R D.parse) (Claims.Flock.E D decode) x P ≤
      Claims.Flock.κ D x + Real.sqrt (Claims.Flock.γ D x * PrB H D.V P x) := by
  have h : PrBad H D.V (Claims.Flock.RTable D.table 0) (Claims.Flock.ETable D N decode 0) x P ≤
      Claims.Flock.κTable D N ε 0 x + Real.sqrt (Claims.Flock.γ D x * PrB H D.V P x) := hKS decode hlw 0 x Ω P
  rw [prBad_zero_eq H D N decode x hs h1 ht hN P, κTable_zero_eq D N ε x hε hN] at h
  exact h

theorem ksTable_zero_of_ks [Zero Col] (H : Msg → Msg) (D : Claims.Flock.Deploy Coin Col T) (N : Msg → ℕ → ℕ)
    (ε : Msg → ℕ → ℝ) (hKS : Claims.Flock.KS H D) (decode : List (ℕ × Col) → List T)
    (hlw : Claims.Flock.ListWithin D.len D.enc D.θ decode) (x : Msg) {s : Stmt} (hs : D.parse x = some s)
    (h1 : s.tables = 1) (ht : D.table x 0 = some s) (hN : (N x 0 : ℝ) = D.N₀ x) (hε : ε x 0 = D.εtable x)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) :
    PrBad H D.V (Claims.Flock.RTable D.table 0) (Claims.Flock.ETable D N decode 0) x P ≤
      Claims.Flock.κTable D N ε 0 x + Real.sqrt (Claims.Flock.γ D x * PrB H D.V P x) := by
  have h : PrBad H D.V (Claims.Flock.R D.parse) (Claims.Flock.E D decode) x P ≤
      Claims.Flock.κ D x + Real.sqrt (Claims.Flock.γ D x * PrB H D.V P x) := hKS decode hlw x Ω P
  rw [prBad_zero_eq H D N decode x hs h1 ht hN P, κTable_zero_eq D N ε x hε hN]
  exact h

end OneTable

end Claims.Flock
