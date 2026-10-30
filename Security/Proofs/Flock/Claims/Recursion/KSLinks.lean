import Proofs.Flock.Claims.Recursion.KSLin2

/-!
# The linked relation: `KSLin2` with link equality, by a union bound

A recursive statement may carry its inner population's link table (`CompiledStatement.links`: the table's root `T` and
row count `N`), whose rows say that input row `p` of unit `r` holds output row `o` of unit `w`. The linked relation `RL`
is the relation `R` and a link clause `Q` at the inner witness and the table's rows, which the extractor reads off the
outer witnesses of its judged draw (`EL`: `E*`, then the rows `rows` reads there).

`PrBad_EL_le` is the union bound: where the linked extractor fails, `E*` fails at `R`, or it recovers a witness in `R`
whose link clause fails. The second event is a collision the link finder `BL` finds, wherever accepting with a witness in
`R` and a failing link clause is a collision among the SHA-512 evaluations of `ins` (`LinkBindsAt`): `ins` reads, at
each draw `E*` may serve, the draw's coins, its run's messages and `E_O`'s outer witnesses there, and the extracted
witness and table rows. So `KSLin2On` at `R` gives `KSLin2LinkOn` at `RL`, its bound one term more, `Pr[B_L]`
(`ksLin2LinkOn_of`), and `KSLin2LinkOn` gives `KSLin2On` back wherever `B_L` finds nothing (`ksLin2On_of_linkOn`).
-/

namespace Claims.Recursion

open Claims

section Links

variable {Coin W Col T Tb : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- `E*`'s coins for `E_O` at its judged draw. -/
def judgedC {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {decO : List (ℕ × Col) → List T} {x : Msg} (c : (E H D I DO decO).C x) : (EO D I DO decO).C x :=
  Prod.fst (β := Fin (I.waits (xI D x)) → Fin (I.cap (xI D x)) → (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) c

/-- `E*`'s draws for its waits: wait `i`'s `j`-th rerun's run coins and `E_O`'s coins. -/
def waitsC {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {decO : List (ℕ × Col) → List T} {x : Msg} (c : (E H D I DO decO).C x) :
    Fin (I.waits (xI D x)) → Fin (I.cap (xI D x)) → (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x :=
  Prod.snd (α := (EO D I DO decO).C x) c

/-- **`E*` with the link table's rows**: `E*`, and where it extracts `w`, `w` with the rows `rows` reads off `E_O`'s
outer witnesses at the judged draw (none where `E_O` gives up there). -/
noncomputable def EL (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) (rows : Msg → List Flock.Wit → Tb) : Extractor Coin (W × Tb) D.V where
  C := (E H D I DO decO).C
  ext x r₀ c := ((E H D I DO decO).ext x r₀ c).bind fun
    | none => .ret none
    | some w => ((EO D I DO decO).ext x r₀ (judgedC c)).bind fun ws => .ret (some (w, rows x (ws.getD [])))

/-- **The linked relation**: `R`, and the link clause `Q` at the inner witness and the table's rows. -/
def RL (D : Deploy Coin) (I : Inner Coin W) (Q : Msg → W → Tb → Prop) : Rel (W × Tb) :=
  fun x p => Recursion.R D I x p.1 ∧ Q x p.1 p.2

/-- The draws `E*` may serve at its coins `c`: the judged draw `(r₀, c.1)`, then every wait's. -/
def drawsOf {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {decO : List (ℕ × Col) → List T} {x : Msg} (r₀ : Fin (D.V.k x) → Coin) (c : (E H D I DO decO).C x) :
    List ((Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) :=
  (r₀, judgedC c) :: (List.finRange (I.waits (xI D x))).flatMap fun i =>
    (List.finRange (I.cap (xI D x))).map (waitsC c i)

/-- What the link finder reads at a draw: the run's coins, the run's messages, and `E_O`'s outer witnesses there (none
where it gives up). -/
noncomputable def drawView (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (d : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) : (ℕ → Coin) × List (Msg × Out) × List Flock.Wit :=
  (coinSeq d.1, (run H P D.V x ω d.1).msgs, ((EO D I DO decO).eval H P x ω d.1 d.2).1.getD [])

/-- The link finder's inputs at one run and `E*`'s coins: `ins` at the draws `E*` may serve, the witness and the rows. -/
noncomputable def linkInputs (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T)
    (ins : (Msg → Msg) → Msg → List ((ℕ → Coin) × List (Msg × Out) × List Flock.Wit) → W → Tb → List Msg)
    {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (r₀ : Fin (D.V.k x) → Coin) (c : (E H D I DO decO).C x) (w : W)
    (rs : Tb) : List Msg :=
  ins H x ((drawsOf r₀ c).map (drawView H D I DO decO P x ω)) w rs

/-- **`B_L`, the link finder** (extraction-based): draw the tape, the judged run's coins and `E*`'s; run `E*` with the
table's rows (`EL`); where it extracts `(w, rows)`, run `E_O` at each draw `E*` may serve and output the first
collision among the SHA-512 evaluations of the inputs `ins` reads there (`linkInputs`). Its cost is the judged run's
calls, both parties', the extraction's, `E_O`'s at every draw, and the inputs'. -/
noncomputable def BL (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) (rows : Msg → List Flock.Wit → Tb)
    (ins : (Msg → Msg) → Msg → List ((ℕ → Coin) × List (Msg × Out) × List Flock.Wit) → W → Tb → List Msg)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg) : Finder where
  S := Ω × (Fin (D.V.k x) → Coin) × (E H D I DO decO).C x
  out s :=
    match ((EL H D I DO decO rows).eval H P x s.1 s.2.1 s.2.2).1 with
    | some p => findColl ((linkInputs H D I DO decO ins P x s.1 s.2.1 s.2.2 p.1 p.2).map fun u => (u, H u))
    | none => none
  cost s :=
    (run H P D.V x s.1 s.2.1).pLog.length + (run H P D.V x s.1 s.2.1).vLog.length +
      ((EL H D I DO decO rows).eval H P x s.1 s.2.1 s.2.2).2.1 +
      ((drawsOf s.2.1 s.2.2).map fun d => ((EO D I DO decO).eval H P x s.1 d.1 d.2).2.1).sum +
      (match ((EL H D I DO decO rows).eval H P x s.1 s.2.1 s.2.2).1 with
        | some p => (linkInputs H D I DO decO ins P x s.1 s.2.1 s.2.2 p.1 p.2).length
        | none => 0)

/-- `Pr[B_L^{P*}(x) outputs a collision]`. -/
noncomputable def PrBL (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) (rows : Msg → List Flock.Wit → Tb)
    (ins : (Msg → Msg) → Msg → List ((ℕ → Coin) × List (Msg × Out) × List Flock.Wit) → W → Tb → List Msg)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg) : ℝ :=
  (BL H D I DO decO rows ins P x).Pr H

/-- **The links bind at `x`**: on every run the verifier accepts where the linked extractor recovers a witness in `R`
whose link clause fails, two of the inputs `ins` reads there collide. -/
def LinkBindsAt (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) (Q : Msg → W → Tb → Prop) (rows : Msg → List Flock.Wit → Tb)
    (ins : (Msg → Msg) → Msg → List ((ℕ → Coin) × List (Msg × Out) × List Flock.Wit) → W → Tb → List Msg)
    (x : Msg) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (ω : Ω) (r₀ : Fin (D.V.k x) → Coin)
    (c : (E H D I DO decO).C x) (w : W) (rs : Tb),
    (run H P D.V x ω r₀).out = .acc → ((EL H D I DO decO rows).eval H P x ω r₀ c).1 = some (w, rs) →
      Recursion.R D I x w → ¬ Q x w rs →
      ∃ q ∈ linkInputs H D I DO decO ins P x ω r₀ c w rs, ∃ q' ∈ linkInputs H D I DO decO ins P x ω r₀ c w rs,
        Collision H q q'

/-- **What the linked extractor returns**: `E*`'s witness, with the rows read off `E_O`'s outer witnesses at the judged
draw. -/
theorem EL_eval_fst (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) (rows : Msg → List Flock.Wit → Tb) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (r₀ : Fin (D.V.k x) → Coin) (c : (E H D I DO decO).C x) :
    ((EL H D I DO decO rows).eval H P x ω r₀ c).1 = ((E H D I DO decO).eval H P x ω r₀ c).1.map fun w =>
      (w, rows x (((EO D I DO decO).eval H P x ω r₀ (judgedC c)).1.getD [])) := by
  unfold Extractor.eval
  show ((((E H D I DO decO).ext x r₀ c).bind _).eval H (run H P D.V x ω)).1 = _
  rw [(RP.eval_bind H (run H P D.V x ω) _ _).1]
  cases (((E H D I DO decO).ext x r₀ c).eval H (run H P D.V x ω)).1 with
  | none => rfl
  | some w =>
    show ((((EO D I DO decO).ext x r₀ (judgedC c)).bind fun ws => RP.ret (some (w, rows x (ws.getD [])))).eval H
        (run H P D.V x ω)).1 =
      some (w, rows x ((((EO D I DO decO).ext x r₀ (judgedC c)).eval H (run H P D.V x ω)).1.getD []))
    rw [(RP.eval_bind H (run H P D.V x ω) _ _).1]
    rfl

/-- **The union bound**: the linked extractor fails where `E*` fails at `R`, or where it recovers a witness in `R` whose
link clause fails, which, where the links bind, `B_L` turns into a collision. -/
theorem PrBad_EL_le (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) (Q : Msg → W → Tb → Prop) (rows : Msg → List Flock.Wit → Tb)
    (ins : (Msg → Msg) → Msg → List ((ℕ → Coin) × List (Msg × Out) × List Flock.Wit) → W → Tb → List Msg)
    (x : Msg) (hL : LinkBindsAt H D I DO decO Q rows ins x) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) :
    PrBad H D.V (RL D I Q) (EL H D I DO decO rows) x P ≤
      PrBad H D.V (Recursion.R D I) (E H D I DO decO) x P + PrBL H D I DO decO rows ins P x := by
  have key : ∀ s : Ω × (Fin (D.V.k x) → Coin) × (E H D I DO decO).C x,
      (run H P D.V x s.1 s.2.1).out = .acc ∧
          ¬ (RL D I Q).ext x ((EL H D I DO decO rows).eval H P x s.1 s.2.1 s.2.2).1 →
        ((run H P D.V x s.1 s.2.1).out = .acc ∧
            ¬ (Recursion.R D I).ext x ((E H D I DO decO).eval H P x s.1 s.2.1 s.2.2).1) ∨
          ∃ u v, (BL H D I DO decO rows ins P x).out s = some (u, v) ∧ Collision H u v := by
    rintro s ⟨hacc, hbad⟩
    have hEL := EL_eval_fst H D I DO decO rows P x s.1 s.2.1 s.2.2
    cases hw : ((E H D I DO decO).eval H P x s.1 s.2.1 s.2.2).1 with
    | none => exact Or.inl ⟨hacc, id⟩
    | some w =>
      rw [hw] at hEL
      rw [hEL] at hbad
      by_cases hR : Recursion.R D I x w
      · right
        have hQ : ¬ Q x w (rows x (((EO D I DO decO).eval H P x s.1 s.2.1 (judgedC s.2.2)).1.getD [])) :=
          fun hQ => hbad (show Recursion.R D I x w ∧ Q x w _ from ⟨hR, hQ⟩)
        obtain ⟨q, hq, q', hq', hc⟩ := hL Ω P s.1 s.2.1 s.2.2 w _ hacc hEL hR hQ
        obtain ⟨u, v, hf, huv⟩ := findColl_of_coll H
          (l := (linkInputs H D I DO decO ins P x s.1 s.2.1 s.2.2 w
            (rows x (((EO D I DO decO).eval H P x s.1 s.2.1 (judgedC s.2.2)).1.getD []))).map fun u => (u, H u))
          (by simp) (by simpa [List.map_map, Function.comp_def] using hq)
          (by simpa [List.map_map, Function.comp_def] using hq') hc
        refine ⟨u, v, ?_, huv⟩
        show (match ((EL H D I DO decO rows).eval H P x s.1 s.2.1 s.2.2).1 with
          | some p => findColl ((linkInputs H D I DO decO ins P x s.1 s.2.1 s.2.2 p.1 p.2).map fun u => (u, H u))
          | none => none) = some (u, v)
        rw [hEL]
        exact hf
      · exact Or.inl ⟨hacc, hR⟩
  unfold PrBad PrBL Finder.Pr
  exact (Pr_mono key).trans (Pr_or_le _ _)

/-- **`R` at `E*` within `RL` at the linked extractor**: where `E*` fails at `R`, the linked extractor fails at `RL`. -/
theorem PrBad_le_PrBad_EL (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) (Q : Msg → W → Tb → Prop) (rows : Msg → List Flock.Wit → Tb) (x : Msg)
    {Ω : Type} [Fintype Ω] (P : Prover Ω) :
    PrBad H D.V (Recursion.R D I) (E H D I DO decO) x P ≤ PrBad H D.V (RL D I Q) (EL H D I DO decO rows) x P := by
  unfold PrBad
  refine Pr_mono ?_
  rintro s ⟨hacc, hbad⟩
  refine ⟨hacc, ?_⟩
  show ¬ (RL D I Q).ext x ((EL H D I DO decO rows).eval H P x s.1 s.2.1 s.2.2).1
  rw [EL_eval_fst]
  cases hw : ((E H D I DO decO).eval H P x s.1 s.2.1 s.2.2).1 with
  | none => exact id
  | some w =>
    rw [hw] at hbad
    exact fun h => hbad (show Recursion.R D I x w ∧ _ from h).1

/-- `B_L` finds nothing where `E*` extracts nothing. -/
theorem prBL_of_E_none (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) (rows : Msg → List Flock.Wit → Tb)
    (ins : (Msg → Msg) → Msg → List ((ℕ → Coin) × List (Msg × Out) × List Flock.Wit) → W → Tb → List Msg)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg)
    (hE : ∀ ω r₀ c, ((E H D I DO decO).eval H P x ω r₀ c).1 = none) : PrBL H D I DO decO rows ins P x = 0 := by
  unfold PrBL Finder.Pr
  refine (Pr_congr fun (s : Ω × (Fin (D.V.k x) → Coin) × (E H D I DO decO).C x) => iff_false_intro ?_).trans Pr_false
  rintro ⟨u, v, h, -⟩
  have he : ((EL H D I DO decO rows).eval H P x s.1 s.2.1 s.2.2).1 = none := by
    rw [EL_eval_fst, hE]
    rfl
  change (match ((EL H D I DO decO rows).eval H P x s.1 s.2.1 s.2.2).1 with
    | some p => findColl ((linkInputs H D I DO decO ins P x s.1 s.2.1 s.2.2 p.1 p.2).map fun u => (u, H u))
    | none => none) = some (u, v) at h
  rw [he] at h
  cases h

end Links

section Scoped

variable {Coin Col T Col' T' Tb : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
  [DecidableEq Col']

/-- **`KSLin2` at the linked relation, at the statements `G` admits**: `KSLin2On`'s bound with `RL` for `R`, the linked
extractor `EL` for `E*`, and one more term, the link finder's `Pr[B_L]`. -/
def KSLin2LinkOn [Zero Col] [Zero Col'] (H : Msg → Msg) (D : RecAt Coin) (Bd : Bound Coin Col T)
    (DO : Flock.Deploy Coin Col' T') (Cp : Caps Coin) (G : Msg → Prop) (Q : Msg → Flock.Wit → Tb → Prop)
    (rows : Msg → List Flock.Wit → Tb)
    (ins : (Msg → Msg) → Msg → List ((ℕ → Coin) × List (Msg × Out) × List Flock.Wit) → Flock.Wit → Tb → List Msg) :
    Prop :=
  ∀ decI, decOK Bd.deploy decI → ∀ decO, decOK DO decO → ∀ (x : Msg), G x → ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (M R : ℕ),
    PrBad H (D.record Bd.IVStar).V (RL (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) Q)
        (EL H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO rows) x P ≤
      κ (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) x +
        Real.sqrt (γ (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) x * PrB H (D.record Bd.IVStar).V P x) +
        (4 * (Inner.ofFlock Bd.deploy decI).V.k (xI (D.record Bd.IVStar) x) + 1) *
          PrBinnerV H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO P x +
        4 * Cp.width (xI (D.record Bd.IVStar) x) *
          (PrFCI H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO Cp P x M +
            1 / ((M : ℝ) + 1)) +
        4 * (PrF0I H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO P x R +
          (((Inner.ofFlock Bd.deploy decI).k (xI (D.record Bd.IVStar) x) : ℝ) + 1) / ((R : ℝ) + 1)) +
        βdev2 (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) Cp x *
          (PrFdevV H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO P x M +
            1 / ((M : ℝ) + 1)) +
        PrBL H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO rows ins P x

/-- **`KSLin2On` at `R` and the links' binding give `KSLin2LinkOn` at `RL`** (`PrBad_EL_le`). -/
theorem ksLin2LinkOn_of [Zero Col] [Zero Col'] {H : Msg → Msg} {D : RecAt Coin} {Bd : Bound Coin Col T}
    {DO : Flock.Deploy Coin Col' T'} {Cp : Caps Coin} {G : Msg → Prop} {Q : Msg → Flock.Wit → Tb → Prop}
    {rows : Msg → List Flock.Wit → Tb}
    {ins : (Msg → Msg) → Msg → List ((ℕ → Coin) × List (Msg × Out) × List Flock.Wit) → Flock.Wit → Tb → List Msg}
    (h : KSLin2On H D Bd DO Cp G)
    (hL : ∀ decI, decOK Bd.deploy decI → ∀ decO, decOK DO decO → ∀ x, G x →
      LinkBindsAt H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO Q rows ins x) :
    KSLin2LinkOn H D Bd DO Cp G Q rows ins := by
  intro decI hdI decO hdO x hx Ω _ _ P M R
  have h1 := PrBad_EL_le H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO Q rows ins x
    (hL decI hdI decO hdO x hx) P
  have h2 := h decI hdI decO hdO x hx Ω P M R
  linarith

/-- **`KSLin2LinkOn` gives `KSLin2On` back** at the statements `G'` within `G` where `B_L` finds nothing
(`PrBad_le_PrBad_EL`). -/
theorem ksLin2On_of_linkOn [Zero Col] [Zero Col'] {H : Msg → Msg} {D : RecAt Coin} {Bd : Bound Coin Col T}
    {DO : Flock.Deploy Coin Col' T'} {Cp : Caps Coin} {G G' : Msg → Prop} {Q : Msg → Flock.Wit → Tb → Prop}
    {rows : Msg → List Flock.Wit → Tb}
    {ins : (Msg → Msg) → Msg → List ((ℕ → Coin) × List (Msg × Out) × List Flock.Wit) → Flock.Wit → Tb → List Msg}
    (h : KSLin2LinkOn H D Bd DO Cp G Q rows ins) (hG : ∀ x, G' x → G x)
    (h0 : ∀ decI, decOK Bd.deploy decI → ∀ decO, decOK DO decO → ∀ x, G' x → ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω]
      (P : Prover Ω), PrBL H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO rows ins P x = 0) :
    KSLin2On H D Bd DO Cp G' := by
  intro decI hdI decO hdO x hx Ω _ _ P M R
  have h1 := PrBad_le_PrBad_EL H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO Q rows x P
  have h2 := h decI hdI decO hdO x (hG x hx) Ω P M R
  rw [h0 decI hdI decO hdO x hx Ω P, add_zero] at h2
  linarith

end Scoped

end Claims.Recursion
