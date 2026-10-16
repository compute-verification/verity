import Proofs.Flock.Claims.Sampling.Wait
import Proofs.Flock.Claims.Sampling.Escape

/-!
# Sampling: KS

The claims block's `KS` (`Definitions.Claims.Sampling`) at a general deployment `D`: the sampling extractor fails on an
accepted run with probability at most `κ_t + √(γ_t·Pr[B]) + Pr[B_ext]`. An accepted run whose extraction fails has `t`
or more regions wrong under the assembled witness. Pick `t` of them, `B`. Then either

* (a) the wait came up short, or every kept rerun's draw missed `B`; or
* (b) a kept rerun's draw sampled a region of `B` and its session's Flock proof was accepted, but its extraction
  failed; or
* (c) the openings the extractor assembled disagree (`EDisagree`), an own-call collision when the path check binds
  (`KSLin_of_disagree`).

If none of these holds, a kept rerun's session extracted a Flock witness whose openings open the sampled region's rows
and satisfy its check (`SessionOpens`), so the region is right: a contradiction (`split`). (a) and (b) together are
every kept rerun bad for `B` (`badB`), and `wait_bad_le` bounds them at each tape `ω`:
`a(ω)·Pr[(a) or (b)] ≤ 2k/K + e^{−k/8} + C(N, t)^{1/k}·(φ(ω) + miss(t))`, where `a(ω)` is the judged run's acceptance
and `φ(ω)` the session extraction's failure at `ω`. Averaged over the tape, `𝔼φ` is the session's `PrBad`, which
Flock's KS at the sessions bounds by `κ_F + √(γ_F·Pr[B])`. With `γ_t = C(N, t)^{2/k}·γ_F`, that is
`κ_t + √(γ_t·Pr[B])`.

* `sessionE`, `sessionR`, `γF`, `SessionKS`: the session's extractor (Flock's `E^rep` on the session's statement, its
  reruns keeping the judged run's draw coin, as `inner` runs it), Flock's relation at the session's statement, Flock's
  `γ` at the sampled verifier's rounds, and **Flock's KS at the sessions**: `Claims.KS` at these, with Flock's
  `κ` at the inner numbers (`κF`). It is what `Claims.Flock.KS` gives once the session is embedded in the sampled
  verifier's run.
* `DrawLaw`, `Refuses`, `SessionOpens`: what the deployed sampled verifier is taken to do. Its draw (from coin 0) has
  its law's escape, as `Escape.lean` and `Live.lean` prove of the model and live draws. It accepts only parsed
  statements with well-formed laws. An accepted session's Flock witness opens the drawn regions' rows and satisfies
  their checks.
* `KS_of`: **Sampling's KS** at a general `D`, from `PathBinds`, `SessionKS`, `SessionOpens`, `DrawLaw`, `Refuses`, a
  positive truncation `K` and a nonnegative table error. `Bits_of`: the block's `Bits`, adding `OpenCallsLe`.
-/

namespace Claims.Sampling

open Claims Finset

/-! ## Rerun programs, evaluated -/

theorem flatMapRP_eval {R β γ : Type} (H : Msg → Msg) (run : R → Trace) (f : β → RP R (List γ)) :
    ∀ l : List β, ((flatMapRP f l).eval H run).1 = l.flatMap fun b => ((f b).eval H run).1
  | [] => rfl
  | b :: bs => by
    simp only [flatMapRP, List.flatMap_cons]
    rw [(RP.eval_bind H run _ _).1, (RP.eval_bind H run _ _).1, flatMapRP_eval H run f bs]
    rfl

theorem mem_checkOpens {Coin Col T R : Type} (H : Msg → Msg) (run : R → Trace) (D : Deploy Coin Col T) (s : Stmt)
    {os : List Opening} {o : Opening} (ho : o ∈ os) (hok : ((openOC D.path s o).eval H).1 = true) :
    o ∈ ((checkOpens (R := R) D s os).eval H run).1 := by
  unfold checkOpens
  rw [flatMapRP_eval]
  refine List.mem_flatMap.2 ⟨o, ho, ?_⟩
  rw [(RP.eval_bind H run _ _).1, eval_ofOC, hok]
  simp [RP.eval]

theorem openOC_row_lt {path : Msg → ℕ → Msg → Msg → OC Bool} {H : Msg → Msg} {s : Stmt} {o : Opening}
    (h : ((openOC path s o).eval H).1 = true) : o.row < s.rows.length := by
  unfold openOC at h
  cases hr : s.rows[o.row]? with
  | none => simp [hr, OC.eval] at h
  | some row => exact (List.getElem?_eq_some_iff.1 hr).1

theorem mem_zip_of_getElem? {R S : Type} {k : ℕ} {l : List R} (f : R → S) {j : Fin k} {r : R}
    (h : l[j.val]? = some r) : (j, f r) ∈ (List.finRange k).zip (l.map f) := by
  rw [List.mem_iff_getElem?]
  exact ⟨j.val, List.getElem?_zip_eq_some.2 ⟨by simp, by rw [List.getElem?_map, h]; rfl⟩⟩

/-! ## The assembled witness -/

/-- **`assemble` reads an agreeing opening**: when the openings of each row agree on its values, the assembled witness
opens a row that some opening opens, with one of the openings and its values. -/
theorem assemble_opening {s : Stmt} {os : List Opening}
    (hagree : ∀ o ∈ os, ∀ o' ∈ os, o.row = o'.row → o.vals = o'.vals) {o : Opening} (ho : o ∈ os)
    (hr : o.row < s.rows.length) : ∃ o' ∈ os, (assemble s os).opening o.row = o' ∧ o'.vals = o.vals := by
  obtain ⟨o', hf⟩ : ∃ o', os.find? (fun o' => decide (o'.row = o.row)) = some o' :=
    Option.isSome_iff_exists.1 (List.find?_isSome.2 ⟨o, ho, by simp⟩)
  have hmem := List.mem_of_find?_eq_some hf
  have hrow : o'.row = o.row := by simpa using List.find?_some hf
  refine ⟨o', hmem, ?_, hagree o' hmem o ho hrow⟩
  simp only [Wit.opening, assemble, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hr,
    Option.map_some, Option.getD_some, hf]
  cases o'
  simp_all

/-- `t` or more entries of a list that meet `q` sit at `t` indices. -/
theorem filter_set {α : Type} (q : α → Prop) [DecidablePred q] : ∀ (l : List α) (t : ℕ),
    t ≤ (l.filter fun a => decide (q a)).length →
      ∃ B ⊆ range l.length, B.card = t ∧ ∀ i ∈ B, ∀ g, l[i]? = some g → q g
  | [], t, h => ⟨∅, empty_subset _, by simp at h; simp [h], by simp⟩
  | a :: l, t, h => by
    have hsucc : ∀ B ⊆ range l.length, B.map ⟨Nat.succ, Nat.succ_injective⟩ ⊆ range (a :: l).length := by
      intro B hB i hi
      obtain ⟨j, hj, rfl⟩ := mem_map.1 hi
      simpa using hB hj
    by_cases ha : q a
    · rcases t with _ | t
      · exact ⟨∅, empty_subset _, rfl, by simp⟩
      have h' : t ≤ (l.filter fun a => decide (q a)).length := by simpa [List.filter_cons, ha] using h
      obtain ⟨B, hB, hc, hq⟩ := filter_set q l t h'
      refine ⟨cons 0 (B.map ⟨Nat.succ, Nat.succ_injective⟩) (by simp), fun i hi => ?_, by simp [hc],
        fun i hi g hg => ?_⟩
      · rcases mem_cons.1 hi with rfl | hi
        · simp
        · exact hsucc B hB hi
      · rcases mem_cons.1 hi with rfl | hi
        · simp only [List.getElem?_cons_zero, Option.some.injEq] at hg
          exact hg ▸ ha
        · obtain ⟨j, hj, rfl⟩ := mem_map.1 hi
          exact hq j hj g (by simpa using hg)
    · have h' : t ≤ (l.filter fun a => decide (q a)).length := by simpa [List.filter_cons, ha] using h
      obtain ⟨B, hB, hc, hq⟩ := filter_set q l t h'
      refine ⟨B.map ⟨Nat.succ, Nat.succ_injective⟩, hsucc B hB, by simp [hc], fun i hi g hg => ?_⟩
      obtain ⟨j, hj, rfl⟩ := mem_map.1 hi
      exact hq j hj g (by simpa using hg)

/-- `t` or more wrong regions contain `t` of them. -/
theorem wrong_set {H : Msg → Msg} {path : Msg → ℕ → Msg → Msg → OC Bool} {s : Stmt} {w : Wit} {t : ℕ}
    (h : t ≤ wrong H path s w) :
    ∃ B ⊆ range s.regions.length, B.card = t ∧ ∀ i ∈ B, ∀ g, s.regions[i]? = some g → ¬ Right H path s w g := by
  unfold wrong at h
  exact filter_set (fun g => ¬ Right H path s w g) s.regions t (by simpa only [decide_not] using h)

/-! ## The session's extraction -/

section Session

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

/-- **The session's extractor** at a decoder: replay the judged run; if it accepted and holds a session, Flock's
`E^rep` on the session's statement, without its replay, its reruns keeping the judged run's draw coin (as `inner` runs
it); the session's statement with the witness it recovers. -/
noncomputable def sessionE (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) :
    Extractor Coin (Msg × Flock.Wit) D.V where
  C x := Fin (D.innerWaits x) → Fin (D.innerCap x) → Fin (D.V.k x) → Coin
  ext x r ρ := .rerun r fun v =>
    if v.out = .acc then
      match D.session v with
      | none => .ret none
      | some (xS, _) =>
        (firstWait (D.innerK x) D.sessionCols (Flock.assemble H D.flock.parse decode D.flock.read xS)
          (Flock.check H D.flock.parse xS)
          ((List.finRange (D.innerWaits x)).map fun i =>
            (List.finRange (D.innerCap x)).map fun j => atDraw r (ρ i j))).bind
          fun o => .ret (o.map fun wF => (xS, wF))
    else .ret none

/-- Flock's relation at the session's statement. -/
def sessionR (H : Msg → Msg) (D : Deploy Coin Col T) : Rel (Msg × Flock.Wit) :=
  fun _ p => Flock.R H D.flock.parse p.1 p.2

/-- Flock's `γ` at the sampled verifier's rounds: `16·(√(2·k(x)·C) + k(x))²`, so `γ_t = C(N, t)^{2/k}·γ_F`. -/
noncomputable def γF (D : Deploy Coin Col T) (x : Msg) : ℝ :=
  16 * (Real.sqrt (2 * D.V.k x * D.weight x) + D.V.k x) ^ 2

/-- **Flock's KS at the sessions**: at every decoder that meets Flock's list-decoding spec, the session's extraction
fails on an accepted run with probability at most `κ_F + √(γ_F·Pr[B])`, `κ_F` Flock's `κ` at the inner numbers and
`γ_F` Flock's `γ` at the sampled verifier's rounds. `Claims.KS`, as `Claims.Flock.KS` states it of Flock's verifier;
the session's embedding in the sampled verifier's run carries one to the other. -/
def SessionKS (H : Msg → Msg) (D : Deploy Coin Col T) : Prop :=
  ∀ decode, Flock.ListComplete D.flock.enc D.flock.agree D.flock.nMin decode →
    Claims.KS H D.V (sessionR H D) (sessionE H D decode) (κF D) (γF D)

/-- **The draw has its law's escape**: at every parsed statement with a well-formed law, the regions the sampled
verifier draws from its draw coin (coin 0) miss a set `B` of regions with probability at most the law's escape. What
`Escape.lean` and `Live.lean` prove of the model and live draws; binding the sampled verifier's draw to them is the
code side's. -/
def DrawLaw (D : Deploy Coin Col T) (drawn : Msg → Coin → Finset ℕ) : Prop :=
  ∀ x s, D.parse x = some s → s.law.WellFormed s.regions.length → ∀ B ⊆ range s.regions.length,
    Pr (fun c : Coin => Disjoint (drawn x c) B) ≤ s.law.escape s.regions.length B

/-- **The sampled verifier refuses what it can't parse or whose law is ill formed**: every accepted run is of a parsed
statement with a well-formed law. -/
def Refuses (H : Msg → Msg) (D : Deploy Coin Col T) : Prop :=
  ∀ (x : Msg) (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin), (run H P D.V x ω r).out = .acc →
    ∃ s, D.parse x = some s ∧ s.law.WellFormed s.regions.length

/-- **An accepted session's witness opens the drawn regions' rows and satisfies their checks**: on an accepted run of
a parsed statement, a Flock witness the session's relation accepts gives, for each region the run's draw sampled, an
opening of each of its rows that `openOC` accepts, and any witness that agrees with those openings satisfies the
region's check. -/
def SessionOpens (H : Msg → Msg) (D : Deploy Coin Col T) (drawn : Msg → Coin → Finset ℕ) : Prop :=
  ∀ (x : Msg) (s : Stmt) (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin) (xS : Msg) (vS : Seen)
    (wF : Flock.Wit), D.parse x = some s → (run H P D.V x ω r).out = .acc →
    D.session (run H P D.V x ω r).seen = some (xS, vS) → Flock.check H D.flock.parse xS wF = true →
    ∀ i ∈ drawn x (coinSeq r 0), ∀ g, s.regions[i]? = some g →
      (∀ row ∈ g.rows, ∃ o ∈ D.openings s (run H P D.V x ω r).seen wF, o.row = row ∧
        ((openOC D.path s o).eval H).1 = true) ∧
      ∀ w : Wit, (∀ row ∈ g.rows, ∀ o ∈ D.openings s (run H P D.V x ω r).seen wF, o.row = row →
        ((openOC D.path s o).eval H).1 = true → w.vals.getD row [] = o.vals) →
        g.C.eval (g.inp.map (val s w)) = g.out.map (val s w)

omit [DecidableEq Col] in
/-- **The session's extraction is `inner`'s**: when it recovers `(xS, wF)` at a rerun, the rerun accepted with session
statement `xS`, Flock's relation accepts `wF`, and `inner` at that rerun checks `wF`'s openings. -/
theorem sessionE_some (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) {Ω : Type}
    (P : Prover Ω) (x : Msg) (s : Stmt) (ω : Ω) (r : Fin (D.V.k x) → Coin) (ρ : (sessionE H D decode).C x)
    {xS : Msg} {wF : Flock.Wit} (h : ((sessionE H D decode).eval H P x ω r ρ).1 = some (xS, wF)) :
    (run H P D.V x ω r).out = .acc ∧ (∃ vS, D.session (run H P D.V x ω r).seen = some (xS, vS)) ∧
      Flock.check H D.flock.parse xS wF = true ∧
      ((inner H D decode x s r (run H P D.V x ω r).seen ρ).eval H (run H P D.V x ω)).1 =
        ((checkOpens D s (D.openings s (run H P D.V x ω r).seen wF)).eval H (run H P D.V x ω)).1 := by
  simp only [Extractor.eval, sessionE, RP.eval] at h
  by_cases hacc : (run H P D.V x ω r).seen.out = .acc
  · simp only [hacc, ↓reduceIte] at h
    refine ⟨hacc, ?_⟩
    rcases hs : D.session (run H P D.V x ω r).seen with _ | ⟨xS', vS⟩
    · simp [hs, RP.eval] at h
    · simp only [hs] at h
      rw [(RP.eval_bind _ _ _ _).1] at h
      simp only [RP.eval] at h
      obtain ⟨wF', hw, he⟩ := Option.map_eq_some_iff.1 h
      obtain ⟨rfl, rfl⟩ := Prod.mk.injEq _ _ _ _ ▸ he
      refine ⟨⟨vS, rfl⟩, firstWait_some H _ _ _ _ _ _ _ hw, ?_⟩
      unfold inner
      simp only [hs]
      rw [(RP.eval_bind _ _ _ _).1, hw]
  · simp [hacc, RP.eval] at h

end Session

/-! ## The split -/

section Split

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

/-- A kept rerun is **bad for `B`** at its session's coins: the session's extraction fails there, or the rerun's draw
misses `B`. -/
def badB (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) (drawn : Msg → Coin → Finset ℕ)
    {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (B : Finset ℕ) (r : Fin (D.V.k x) → Coin)
    (ρ : (sessionE H D decode).C x) : Prop :=
  ¬ (sessionR H D).ext x ((sessionE H D decode).eval H P x ω r ρ).1 ∨ Disjoint (drawn x (coinSeq r 0)) B

/-- **The sampling extraction's failures at a tape**, other than disagreeing openings: at a parsed statement with a
well-formed law, the wait came up short, or for some `t` regions `B` every kept rerun is bad for `B`. -/
def Fails (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) (drawn : Msg → Coin → Finset ℕ)
    (t : ℕ) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (c : (E H D decode).C x) : Prop :=
  ∃ s, D.parse x = some s ∧ s.law.WellFormed s.regions.length ∧
    ((accList (fun r => (run H P D.V x ω r).out = .acc) (D.k x) ((List.finRange (D.cap x)).map c.1)).length < D.k x ∨
      ∃ B ∈ powersetCard t (range s.regions.length),
        AllBad (fun r => (run H P D.V x ω r).out = .acc) (D.k x) (badB H D decode drawn P x ω B)
          ((List.finRange (D.cap x)).map c.1) (c.2 : Fin (D.k x) → (sessionE H D decode).C x))

omit [DecidableEq Col] in
/-- **The split**: an accepted run whose extraction fails has a failure at its tape (`Fails`), or disagreeing
assembled openings. Otherwise some kept rerun sampled a region of the `t` wrong ones and extracted its session, whose
openings make that region right under the assembled witness. -/
theorem split (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T)
    (drawn : Msg → Coin → Finset ℕ) (t : ℕ) (hopens : SessionOpens H D drawn) (hrefuse : Refuses H D) (x : Msg)
    (Ω : Type) (P : Prover Ω) (s : Ω × (Fin (D.V.k x) → Coin) × (E H D decode).C x)
    (hacc : (run H P D.V x s.1 s.2.1).out = .acc)
    (hR : ¬ (R H D.path D.parse t).ext x ((E H D decode).eval H P x s.1 s.2.1 s.2.2).1) :
    ((run H P D.V x s.1 s.2.1).out = .acc ∧ Fails H D decode drawn t P x s.1 s.2.2) ∨
      EDisagree H D decode P x s.1 s.2.2 := by
  obtain ⟨ω, r₀, c⟩ := s
  simp only at hacc hR ⊢
  obtain ⟨sS, hs, hwf⟩ := hrefuse x Ω P ω r₀ hacc
  by_cases hd : EDisagree H D decode P x ω c
  · exact Or.inr hd
  refine Or.inl ⟨hacc, sS, hs, hwf, ?_⟩
  set os := Eopens H D decode x sS (run H P D.V x ω) c with hos
  have hagree : ∀ o ∈ os, ∀ o' ∈ os, o.row = o'.row → o.vals = o'.vals := by
    intro o ho o' ho' hrow
    by_contra hne
    exact hd ⟨sS, hs, o, ho, o', ho', hrow, hne⟩
  by_cases hlen : (accList (fun r => (run H P D.V x ω r).out = .acc) (D.k x)
      ((List.finRange (D.cap x)).map c.1)).length < D.k x
  · exact Or.inl hlen
  right
  have hlen' := le_antisymm (accList_length_le _ _ _) (not_lt.1 hlen)
  rw [E_eval_acc H D decode P x sS hs ω r₀ c hacc] at hR
  have hwr : t ≤ wrong H D.path sS (assemble sS os) := by
    by_contra hlt
    exact hR ⟨sS, hs, not_le.1 hlt⟩
  obtain ⟨B, hBsub, hBcard, hBw⟩ := wrong_set hwr
  refine ⟨B, mem_powersetCard.2 ⟨hBsub, hBcard⟩, hlen', fun j r hj => ?_⟩
  by_contra hbad
  simp only [badB, not_or, not_not] at hbad
  obtain ⟨hext, hdisj⟩ := hbad
  rcases hres : ((sessionE H D decode).eval H P x ω r (c.2 j)).1 with _ | ⟨xS, wF⟩
  · rw [hres] at hext
    exact hext
  obtain ⟨hacc_r, ⟨vS, hsess⟩, hcheck, hinner⟩ := sessionE_some H D decode P x sS ω r (c.2 j) hres
  obtain ⟨i, hi, hiB⟩ := Finset.not_disjoint_iff.1 hdisj
  have hiN : i < sS.regions.length := mem_range.1 (hBsub hiB)
  obtain ⟨hrows, hcirc⟩ := hopens x sS Ω P ω r xS vS wF hs hacc_r hsess hcheck i hi sS.regions[i]
    (List.getElem?_eq_getElem hiN)
  have hin : ∀ o ∈ D.openings sS (run H P D.V x ω r).seen wF, ((openOC D.path sS o).eval H).1 = true → o ∈ os := by
    intro o ho hok
    rw [hos]
    unfold Eopens
    rw [flatMapRP_eval, waitAccC_eval_nil]
    refine List.mem_flatMap.2 ⟨(j, (r, (run H P D.V x ω r).seen)), mem_zip_of_getElem? _ hj, ?_⟩
    show o ∈ ((inner H D decode x sS r (run H P D.V x ω r).seen (c.2 j)).eval H (run H P D.V x ω)).1
    rw [hinner]
    exact mem_checkOpens H _ D sS ho hok
  refine hBw i hiB sS.regions[i] (List.getElem?_eq_getElem hiN)
    ⟨fun row hrow => ?_, hcirc _ fun row hrow o ho horow hok => ?_⟩
  · obtain ⟨o, ho, horow, hok⟩ := hrows row hrow
    obtain ⟨o', ho', hop, -⟩ := assemble_opening hagree (hin o ho hok) (openOC_row_lt hok)
    rw [← horow, hop]
    exact (Eopens_checked H D decode P x sS hs ω r₀ c hacc o' ho').1
  · obtain ⟨o', -, hop, hv⟩ := assemble_opening hagree (hin o ho hok) (openOC_row_lt hok)
    have := congrArg Opening.vals hop
    simp only [Wit.opening] at this
    rw [← horow, this, hv]

end Split

/-! ## The bound -/

section Bound

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

omit [Nonempty Coin] [Inhabited Coin] in
/-- The draw coin of uniform coins is uniform. -/
theorem Pr_coinSeq_zero [Nonempty Coin] [Inhabited Coin] {n : ℕ} (hn : 0 < n) (p : Coin → Prop) :
    Pr (fun r : Fin n → Coin => p (coinSeq r 0)) = Pr p := by
  classical
  have e : ∀ r : Fin n → Coin, coinSeq r 0 = r ⟨0, hn⟩ := fun r => by simp [coinSeq, hn]
  simp only [e]
  rw [← Ex_ind, ← Ex_ind p]
  exact Ex_pi_eval (⟨0, hn⟩ : Fin n) fun c => if p c then (1 : ℝ) else 0

/-- **The failures' probability**: `Pr[acc ∧ Fails] ≤ κ_t + √(γ_t·Pr[B])`, from Flock's KS at the sessions and the
draw's law. -/
theorem Pr_fails_le (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T)
    (drawn : Msg → Coin → Finset ℕ) (t : ℕ) (hsess : SessionKS H D)
    (hdec : Flock.ListComplete D.flock.enc D.flock.agree D.flock.nMin decode) (hdraw : DrawLaw D drawn)
    (hcap : ∀ x, 0 < D.cap x) (hε : ∀ x, 0 ≤ D.εtable x) (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) :
    Pr (fun s : Ω × (Fin (D.V.k x) → Coin) × (E H D decode).C x =>
      (run H P D.V x s.1 s.2.1).out = .acc ∧ Fails H D decode drawn t P x s.1 s.2.2) ≤
      κ D t x + Real.sqrt (γ D t x * PrB H D.V P x) := by
  have hκF : 0 ≤ κF D x := by unfold κF; have := hε x; positivity
  have hcov : 0 ≤ cover D t x := by unfold cover; positivity
  have hmiss := missAt_nonneg D t x
  have hsq := Real.sqrt_nonneg (γ D t x * PrB H D.V P x)
  have hκ0 : 0 ≤ κ D t x := (κ_pos D t x (hε x)).le
  have hzero : ∀ (Q : Ω × (Fin (D.V.k x) → Coin) × (E H D decode).C x → Prop), (∀ s, ¬ Q s) →
      Pr Q ≤ κ D t x + Real.sqrt (γ D t x * PrB H D.V P x) := fun Q hQ => by
    rw [(Pr_congr fun s => ⟨fun h => hQ s h, False.elim⟩).trans Pr_false]
    linarith
  rcases hp : D.parse x with _ | sS
  · exact hzero _ fun s ⟨_, s', h, _⟩ => by rw [hp] at h; cases h
  by_cases hwf : sS.law.WellFormed sS.regions.length
  swap
  · exact hzero _ fun s ⟨_, s', h, hw, _⟩ => by rw [hp, Option.some.injEq] at h; subst h; exact hwf hw
  by_cases hk : D.k x = 0
  · have h1 : 1 ≤ κ D t x := by
      unfold κ
      rw [hk]
      simp only [Nat.cast_zero, mul_zero, zero_div, neg_zero, Real.exp_zero]
      nlinarith [mul_nonneg hcov (add_nonneg hmiss hκF)]
    exact (Pr_le_one _).trans (by linarith)
  by_cases hn : D.V.k x = 0
  · exact hzero _ fun s ⟨h, _⟩ => by rw [run_out_of_k_eq_zero H P D.V x hn] at h; cases h
  have hn' : 0 < D.V.k x := Nat.pos_of_ne_zero hn
  have hmissS : missAt D t x = sS.law.miss sS.regions.length t := by unfold missAt; rw [hp]
  have hcovS : cover D t x = ((sS.regions.length.choose t : ℕ) : ℝ) ^ (1 / (D.k x : ℝ)) := by
    unfold cover nRegions; rw [hp]
  let φ : Ω → ℝ := fun ω => Pr (fun p : (Fin (D.V.k x) → Coin) × (sessionE H D decode).C x =>
    (run H P D.V x ω p.1).out = .acc ∧ ¬ (sessionR H D).ext x ((sessionE H D decode).eval H P x ω p.1 p.2).1)
  have hEφ : Ex φ = PrBad H D.V (sessionR H D) (sessionE H D decode) x P :=
    (Pr_prod_eq_Ex fun s : Ω × (Fin (D.V.k x) → Coin) × (sessionE H D decode).C x =>
      (run H P D.V x s.1 s.2.1).out = .acc ∧
        ¬ (sessionR H D).ext x ((sessionE H D decode).eval H P x s.1 s.2.1 s.2.2).1).symm
  have hKS : Ex φ ≤ κF D x + Real.sqrt (γF D x * PrB H D.V P x) := hEφ ▸ hsess decode hdec x Ω P
  have hper : ∀ ω, Pr (fun r : Fin (D.V.k x) → Coin => (run H P D.V x ω r).out = .acc) *
      Pr (fun c : (E H D decode).C x => Fails H D decode drawn t P x ω c) ≤
      2 * D.k x / D.cap x + Real.exp (-(D.k x : ℝ) / 8) + cover D t x * (φ ω + missAt D t x) := by
    intro ω
    have hF : Pr (fun c : (E H D decode).C x => Fails H D decode drawn t P x ω c) =
        Pr (fun c : (Fin (D.cap x) → Fin (D.V.k x) → Coin) × (Fin (D.k x) → (sessionE H D decode).C x) =>
          (accList (fun r => (run H P D.V x ω r).out = .acc) (D.k x) ((List.finRange (D.cap x)).map c.1)).length <
              D.k x ∨
            ∃ B ∈ powersetCard t (range sS.regions.length),
              AllBad (fun r => (run H P D.V x ω r).out = .acc) (D.k x) (badB H D decode drawn P x ω B)
                ((List.finRange (D.cap x)).map c.1) c.2) :=
      Pr_congr fun c => ⟨fun ⟨s', h, _, hc⟩ => by rw [hp, Option.some.injEq] at h; subst h; exact hc,
        fun hc => ⟨sS, hp, hwf, hc⟩⟩
    rw [hF, hcovS]
    have hy : ∀ B ∈ powersetCard t (range sS.regions.length),
        Pr (fun p : (Fin (D.V.k x) → Coin) × (sessionE H D decode).C x =>
          (run H P D.V x ω p.1).out = .acc ∧ badB H D decode drawn P x ω B p.1 p.2) ≤ φ ω + missAt D t x := by
      intro B hB
      obtain ⟨hBsub, hBcard⟩ := mem_powersetCard.1 hB
      calc Pr (fun p : (Fin (D.V.k x) → Coin) × (sessionE H D decode).C x =>
            (run H P D.V x ω p.1).out = .acc ∧ badB H D decode drawn P x ω B p.1 p.2)
          ≤ Pr (fun p : (Fin (D.V.k x) → Coin) × (sessionE H D decode).C x =>
            ((run H P D.V x ω p.1).out = .acc ∧
              ¬ (sessionR H D).ext x ((sessionE H D decode).eval H P x ω p.1 p.2).1) ∨
              Disjoint (drawn x (coinSeq p.1 0)) B) :=
            Pr_mono fun p ⟨ha, hb⟩ => hb.imp (fun h => ⟨ha, h⟩) id
        _ ≤ φ ω + Pr (fun p : (Fin (D.V.k x) → Coin) × (sessionE H D decode).C x =>
              Disjoint (drawn x (coinSeq p.1 0)) B) := Pr_union_le _ _
        _ = φ ω + Pr (fun c : Coin => Disjoint (drawn x c) B) := by
            rw [Pr_fst (β := (sessionE H D decode).C x) fun r : Fin (D.V.k x) → Coin =>
              Disjoint (drawn x (coinSeq r 0)) B, Pr_coinSeq_zero hn' fun c => Disjoint (drawn x c) B]
        _ ≤ φ ω + sS.law.escape sS.regions.length B := add_le_add le_rfl (hdraw x sS hp hwf B hBsub)
        _ ≤ φ ω + missAt D t x := add_le_add le_rfl (by rw [hmissS]; exact escape_le_miss _ hwf hBsub hBcard.ge)
    have hw := wait_bad_le (fun r : Fin (D.V.k x) → Coin => (run H P D.V x ω r).out = .acc) (D.k x) (D.cap x)
      (hcap x) (Nat.one_le_iff_ne_zero.2 hk) (powersetCard t (range sS.regions.length))
      (badB H D decode drawn P x ω) (add_nonneg (Pr_nonneg _) hmiss) hy
    rwa [card_powersetCard, card_range] at hw
  have hsqrt : cover D t x * Real.sqrt (γF D x * PrB H D.V P x) = Real.sqrt (γ D t x * PrB H D.V P x) := by
    rw [show γ D t x * PrB H D.V P x = cover D t x ^ 2 * (γF D x * PrB H D.V P x) by unfold γ γF; ring,
      Real.sqrt_mul (sq_nonneg (cover D t x)) (γF D x * PrB H D.V P x), Real.sqrt_sq hcov]
  rw [Pr_indep (fun ω (r : Fin (D.V.k x) → Coin) => (run H P D.V x ω r).out = .acc)
    (fun ω (c : (E H D decode).C x) => Fails H D decode drawn t P x ω c)]
  calc Ex (fun ω => Pr (fun r : Fin (D.V.k x) → Coin => (run H P D.V x ω r).out = .acc) *
        Pr (fun c : (E H D decode).C x => Fails H D decode drawn t P x ω c))
      ≤ Ex (fun ω => (2 * D.k x / D.cap x + Real.exp (-(D.k x : ℝ) / 8)) + cover D t x * (φ ω + missAt D t x)) :=
        Ex_mono hper
    _ = 2 * D.k x / D.cap x + Real.exp (-(D.k x : ℝ) / 8) + cover D t x * (Ex φ + missAt D t x) := by
        rw [Ex_add, Ex_const, Ex_mul_left, Ex_add, Ex_const]
    _ ≤ 2 * D.k x / D.cap x + Real.exp (-(D.k x : ℝ) / 8) +
          cover D t x * (κF D x + Real.sqrt (γF D x * PrB H D.V P x) + missAt D t x) := by
        gcongr
    _ = κ D t x + cover D t x * Real.sqrt (γF D x * PrB H D.V P x) := by unfold κ; ring
    _ = κ D t x + Real.sqrt (γ D t x * PrB H D.V P x) := by rw [hsqrt]

end Bound

/-! ## Sampling's KS -/

section Claim

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

/-- **Sampling: KS**, at a general deployment: the registered path check binds (`PathBinds`), Flock's KS holds at the
sessions (`SessionKS`), an accepted session's witness opens the drawn regions (`SessionOpens`), the draw has its law's
escape (`DrawLaw`), the verifier refuses unparsed statements and ill-formed laws (`Refuses`), the wait has `K > 0`
draws, and the table error is nonnegative. -/
theorem KS_of {H : Msg → Msg} {D : Deploy Coin Col T} (drawn : Msg → Coin → Finset ℕ) (hpath : PathBinds H D.path)
    (hsess : SessionKS H D) (hopens : SessionOpens H D drawn) (hdraw : DrawLaw D drawn) (hrefuse : Refuses H D)
    (hcap : ∀ x, 0 < D.cap x) (hε : ∀ x, 0 ≤ D.εtable x) : KS H D := fun t decode hdec =>
  KSLin_of_disagree hpath decode (R H D.path D.parse t) (κ D t) (γ D t)
    (fun x _ P s => (run H P D.V x s.1 s.2.1).out = .acc ∧ Fails H D decode drawn t P x s.1 s.2.2)
    (fun x Ω P s hacc hR => split H D decode drawn t hopens hrefuse x Ω P s hacc hR)
    (fun x Ω _ _ P => Pr_fails_le H D decode drawn t hsess hdec hdraw hcap hε x Ω P)

/-- **Sampling: bits**, from its KS, with one rerun's openings checks bounded (`OpenCallsLe`). -/
theorem Bits_of {H : Msg → Msg} {D : Deploy Coin Col T} (drawn : Msg → Coin → Finset ℕ) (hpath : PathBinds H D.path)
    (hsess : SessionKS H D) (hopens : SessionOpens H D drawn) (hdraw : DrawLaw D drawn) (hrefuse : Refuses H D)
    (hcap : ∀ x, 0 < D.cap x) (hε : ∀ x, 0 ≤ D.εtable x) (hopen : OpenCallsLe H D) : Bits H D :=
  Bits_of_KS (KS_of drawn hpath hsess hopens hdraw hrefuse hcap hε) hopen hε

end Claim

end Claims.Sampling
