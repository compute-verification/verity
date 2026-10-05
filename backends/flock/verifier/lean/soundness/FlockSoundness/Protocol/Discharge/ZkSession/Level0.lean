import FlockSoundness.Protocol.Discharge.ZkSession.Inner
import FlockSoundness.Protocol.SoundnessPad

/-!
# M1's masked Ligerito level 0 (live/PROTOCOL.md §1, §4, §7)

Under `--zk` the prover sends level 0's out-of-domain value, its lane rounds' messages and the rep's extra-lane sums
`e^r` masked: `v' = v + h` with `h` a pad (§2: one `F` pad for the OOD value, four per lane message
`u₀.c0, u₀.c1, u₂.c0, u₂.c1`, two per `e_i`). The verifier can no longer compute the running claim, so it carries it as a
`K`-valued affine form in the pads (`LinK`, §4's `Lin2` read as one `K` value), receives the restart `T'` from the prover
(after `ρ`, before `ybar`, §1), and continues from `T'`. What it could not check becomes the lane-phase row
`claim_k + Σ_i ρ_i e_i + T'` (§4; written `claim_k + Σ_i ρ_i e_i − T'`, the same row in characteristic 2), two `F` rows by
its coordinates `K = F ⊕ F·u` (`limbRows`), which the inner proof enforces.

`ligeritoPadZK` is `Model.ligeritoPad` with those changes and nothing else; levels 1+ (`ligTail`) are verbatim. The
masked rep (`Algebraic.lean`) runs it after the masked algebraic phase.

**The coupling** (VEIL Lemma 4.3, "`v' − h` is a prover strategy for the unmasked protocol"): for every pad vector `h`,
masked level 0 ends with a true verdict and its row vanishing at `h` with probability at most the clear level 0's
acceptance probability (`ligeritoPadZK_couple`). Unmasking each message at `h` turns a masked strategy into a clear one;
the claim forms evaluate at `h` to the clear claims (`fold_couple`), and when the row vanishes `T'` is the clear
restart, after which the two games are the same game.
-/

namespace FlockSoundness.Discharge.ZkSession

open Game Model Accounting
open scoped ENNReal

variable {F K : Type} [Field F] [Field K]

/-! ## Simulation steps -/

/-! ## Forms over `K` -/

/-- A `K`-valued affine form in the pads `h : Fin np → F`: `Σ_i embed(h_i)·coef_i + const`. -/
abbrev LinK (K : Type) (np : ℕ) := (Fin np → K) × K

/-- A constant form. -/
def cK {np : ℕ} (x : K) : LinK K np := (0, x)

/-- Pad `i` of an `F` value. -/
def padF {np : ℕ} (i : Fin np) : LinK K np := (Pi.single i 1, 0)

/-- The pad of a `K` value with coordinate pads `i₀, i₁`: `embed(h_{i₀}) + embed(h_{i₁})·u`. -/
def padK (A : Arith F K) {np : ℕ} (i₀ i₁ : Fin np) : LinK K np := (Pi.single i₀ 1 + Pi.single i₁ A.u, 0)

section Eval

variable (A : Arith F K) {np : ℕ} (h : Fin np → F)

end Eval

/-! ## The lane-phase rows over `F` -/

/-- A `K` element's coordinates `(c0, c1)` with `x = c0 + c1·u` (§2.4; `K = F ⊕ F·u`). -/
noncomputable def coords (A : Arith F K) (x : K) : F × F :=
  Function.invFun (fun w : F × F => A.ofLimbs w.1 w.2) x

/-- The two `F` rows of a `K` row, coordinate by coordinate (§4's `ligerito lane phase (c0, c1)`). -/
noncomputable def limbRows (A : Arith F K) {np : ℕ} (row : LinK K np) : Fin 2 → Lin F np :=
  ![⟨fun i => (coords A (row.1 i)).1, (coords A row.2).1⟩, ⟨fun i => (coords A (row.1 i)).2, (coords A row.2).2⟩]

variable [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K]

/-! ## The masked lane rounds -/

/-- Where a rep's level-0 pads sit (§2): the OOD value's, lane round `j`'s message coordinates
(`u₀.c0, u₀.c1, u₂.c0, u₂.c1`), and the extra lanes' sums `e_i` (coordinates `c0, c1`). Soundness holds for any
placement; hiding is what needs them distinct. -/
structure Pads0 (np e : ℕ) where
  ood : Fin np
  msg : ℕ → Fin 4 → Fin np
  ev : Fin e → Fin 2 → Fin np

/-- A round polynomial whose coefficients are forms. -/
abbrev QuadL (K : Type) (np : ℕ) := LinK K np × LinK K np × LinK K np

/-- `from(msg, s)` (§13.2) on forms. -/
def fromMsgL {np : ℕ} (m : LinK K np × LinK K np) (s : LinK K np) : QuadL K np := (m.1, s + m.2, m.2)

/-- `q(x) = q₀ + q₁·x + q₂·x²` on forms. -/
def QuadL.eval {np : ℕ} (q : QuadL K np) (x : K) : LinK K np := q.1 + x • q.2.1 + x ^ 2 • q.2.2

/-- Lane round `j`'s message unmasked, as forms: `(u₀' + pad, u₂' + pad)`. -/
def unmask (A : Arith F K) {np e : ℕ} (P : Pads0 np e) (j : ℕ) (m : K × K) : LinK K np × LinK K np :=
  (cK m.1 + padK A (P.msg j 0) (P.msg j 1), cK m.2 + padK A (P.msg j 2) (P.msg j 3))

/-- **The masked lane rounds** (`foldRoundsNL` with masked messages): the claim is a form; round `j`'s message is
unmasked by its pads. -/
def foldRoundsZK (A : Arith F K) {np e : ℕ} (P : Pads0 np e) :
    ℕ → ℕ → LinK K np → QuadL K np → Game (List K × LinK K np)
  | 0, _, s, _ => pure ([], s)
  | 1, _, _, q => do
      let r ← drawK A
      pure ([r], q.eval r)
  | k + 2, j, _, q => do
      let r ← drawK A
      let s' := q.eval r
      let msg ← recv (K × K)
      let (rs, s'') ← foldRoundsZK A P (k + 1) (j + 1) s' (fromMsgL (unmask A P (j + 1) msg) s')
      pure (r :: rs, s'')

/-! ## Level 0 -/

/-- Levels 1+ of `Model.ligeritoPad`, from the restarted claim `s` on: the code-switch message, level 1's commitment,
its OOD samples, level 0's padded queries, and the recursive rounds. -/
def ligTail (A : Arith F K) (sch : Schedule) (t : ℕ) (c₀ : F) (e r : ℕ) (b : ℕ → F) (C₀ : Oracle F)
    (lv₀ : Level) (rest : List Level) (z₀ : Fin (sch.m - 7) → F) (β₀ : F) (initial : List K) (ρ : Fin e → K)
    (ybar : Fin t → K) (s : K) : Game Bool := do
  let n := sch.m - 7
  let msg' ← recv (K × K)
  let q := fromMsg msg' s
  match rest with
  | [] => pure false
  | lv₁ :: _ => do
      let C₁ ← recv (Oracle F)
      let dim := n - sch.k0 + 1
      let (s, q, oods) ← twoOod A dim 0 s q
      let st : LigState F K := ⟨s, q, dim, [], ⟨List.ofFn z₀, β₀, none⟩ :: oods, [], C₀, lv₀⟩
      let st ← queryAndBatchV A lv₀
        (padColumn A lv₀.logLen lv₀.logCols t c₀ initial ρ (extraLane sch.k0 e r) ybar C₀) 0 st
      recursiveRounds A n b initial sch.ext rest { st with prev := C₁, prevLevel := lv₁ }

/-- **M1's masked level 0** for rep `r` (§1, §7): `Model.ligeritoPad` with the OOD value, the lane messages and `e^r`
masked by the pads `P`, the claim carried as a form, and the restart `T'` received after `ρ` and before `ybar`. Returns
the lane-phase row `claim_k + Σ_i ρ_i·e_i − T'` and the verdict of levels 1+, run from `T'`. -/
def ligeritoPadZK (A : Arith F K) {np e : ℕ} (P : Pads0 np e) (sch : Schedule) (t : ℕ) (c₀ : F) (r : ℕ) (T : F)
    (b : ℕ → F) (C₀ : Oracle F) : Game (LinK K np × Bool) := do
  let n := sch.m - 7
  match sch.levels with
  | [] => pure (0, false)
  | lv₀ :: rest => do
      let z₀ ← draw (Fin n → F)
      let y₀ ← recv F
      let β₀ ← draw F
      let s : LinK K np := cK (A.embed T) + A.embed β₀ • (cK (A.embed y₀) + padF P.ood)
      let msg ← recv (K × K)
      let (initial, s) ← foldRoundsZK A P sch.k0 0 s (fromMsgL (unmask A P 0 msg) s)
      let ev ← recv (Fin e → K)
      let ρ ← drawKs A e
      let T' ← recv K
      let ybar ← recv (Fin t → K)
      let row := s + ∑ i, ρ i • (cK (ev i) + padK A (P.ev i 0) (P.ev i 1)) - cK T'
      let acc ← ligTail A sch t c₀ e r b C₀ lv₀ rest z₀ β₀ initial ρ ybar T'
      pure (row, acc)

end FlockSoundness.Discharge.ZkSession
