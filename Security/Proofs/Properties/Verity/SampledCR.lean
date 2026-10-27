import Proofs.Properties.Verity.SampledDeploy
import Proofs.Flock.Claims.GameFinder
import Proofs.Flock.Soundness.CROnly.Link

/-!
# Sampled proofs' finder: C-Flock's collision finders as one `Claims.Finder`

The vocabulary's finder premise (`Vocab.FinderCR .sampled`: the finder `B^τ` outputs a SHA-512 collision with
probability at most the published `ε_cr` at the strategy's calls) is about one finder. `B^τ` is built from the finders
C-Flock's premises name (`TableCRZAt`, `LinkCRZCAt`), and no others:

* each table's knowledge finders (`SessCRZ`'s: the rewinding finders at `qF τ = 2 (t' τ + v)`, the self-clash finder at
  `qS τ = t' τ + v`), and each committed position's link finder (`LinkCRZC`'s), each as a `Claims.Finder` on its
  strategy's coin tape (`Game.tapeFinder`, with the finder's success);
* a table's finder is indexed by the table and its kind (`TabIx lvMax`, level slots past a table's levels finding
  nothing), never by the draw: it draws `ω` with its own coins, uniformly as the law does, and runs that draw's finder
  (`tabMix`, a `Claims.Finder.mix` over the draw). A link finder already reads the draw inside its trial;
* `B^τ` (`LiveAt.crFinder τ extra`) is the uniform mixture of all of them and one more finder `extra`: it picks one of
  the `k = card Ix` uniformly with its own coins and runs it. Each, `extra` included (`extra_le`), succeeds at most `k`
  times as often as `B^τ` (`le_mix`), and `B^τ` costs what they do (`crFinder_cost_le`). `extra` is the vocabulary's
  replay finder in a deployment, so that one finder bounds both the play's collision event and C-Flock's premises;
  `Claims.Finder.null` where nothing else is needed. `k` has no factor of the draws: `card_Ix_le` bounds it by the
  tables, their levels and the committed positions, with no strategy in the bound.

Where `B^τ` succeeds with probability at most `β`, each table finder's success averaged over the draw is at most `k β`
(`tabCRAvg_of_crFinder`, `TabCRAvg`), and each link finder meets `LinkCRZC`'s premise where `k β ≤ t' τ / 2^256`
(`linkCR_of_crFinder`; a link finder plays `τ` at least once, so its expected cost is at least `t' τ`, `linkCost_ge`).
The table premise C-Flock's knowledge bound reads (`LiveAt.CR`'s first part) is per draw; one finder gives it only
averaged over the draw, since bounding any one draw's finder by the mixture costs the number of draws.
-/

namespace Flock.Guarantees.LiveAt

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden FlockSoundness.LinkFinder
open scoped ENNReal

/-- `H512` and the compiled SHA-512 agree on collisions. -/
theorem H512_eq_iff (u v : List UInt8) : H512 u = H512 v ↔ Verity.sha512Msg u = Verity.sha512Msg v := by
  rw [Subtype.ext_iff]
  unfold H512 Verity.sha512Msg
  try dsimp only
  -- `hash` is opaque to the proof: unifying through it unfolds SHA-512 on a symbolic input.
  generalize Flock.Sha512.hash ⟨u.toArray⟩ = a
  generalize Flock.Sha512.hash ⟨v.toArray⟩ = b
  constructor
  · intro h
    rw [h]
  · intro h
    cases a
    cases b
    exact congrArg ByteArray.mk (Array.toList_inj.mp h)

/-- A strict-time finder of C-Flock's on its tape, at the constant cost `⌈q⌉`. -/
noncomputable def strictComp (Fd : StrictCR.Finder StrictCR.Pair) (q : ℝ) : Claims.Finder :=
  Game.tapeFinder Fd.g Fd.s id (fun _ => ⌈q⌉₊)

/-- **A strict-time finder's premise from the threshold**: where its tape finder succeeds with probability at most
`β ≤ q²/2^513`, C-Flock's finder meets `SHA512CRStrict` at the budget `q`. -/
theorem strictCR_of (Fd : StrictCR.Finder StrictCR.Pair) (q : ℝ) {β : ℝ}
    (hB : (strictComp Fd q).Pr Verity.sha512Msg ≤ β) (hb : β ≤ q ^ 2 / 2 ^ (513 : ℝ)) :
    Fd.CR H512 q := by
  unfold StrictCR.Finder.CR Assumptions.SHA512CRStrict
  intro _
  unfold strictComp at hB
  rw [Game.tapeFinder_Pr] at hB
  have hp : (fun b : StrictCR.Pair => ∃ x y, id b = some (x, y) ∧ x ≠ y ∧ H512 x = H512 y) =
      fun b => ∃ u v, id b = some (u, v) ∧ Claims.Collision Verity.sha512Msg u v :=
    funext fun b => by simp only [Claims.Collision, H512_eq_iff]
  rw [hp]
  exact hB.trans hb

/-- **An expected-time finder's premise from the threshold**: where every outcome costs at least `t` and its tape
finder succeeds with probability at most `β ≤ t/2^256`, `SHA512CRExpected` holds. -/
theorem expectedCR_of {γ : Type} (g : Game γ) (s : Strategy g) (out : γ → Option (Claims.Msg × Claims.Msg))
    (cost : γ → ℝ) (cst : γ → ℕ) {t β : ℝ} (hcost : ∀ x, t ≤ cost x)
    (hB : (Game.tapeFinder g s out cst).Pr Verity.sha512Msg ≤ β) (hb : β ≤ t / 2 ^ (256 : ℝ)) :
    Assumptions.SHA512CRExpected H512 g s out cost := by
  unfold Assumptions.SHA512CRExpected
  rw [Game.tapeFinder_Pr] at hB
  have hp : (fun b : γ => ∃ x y, out b = some (x, y) ∧ x ≠ y ∧ H512 x = H512 y) =
      fun b => ∃ u v, out b = some (u, v) ∧ Claims.Collision Verity.sha512Msg u v :=
    funext fun b => by simp only [Claims.Collision, H512_eq_iff]
  rw [hp]
  have ht : t ≤ expect cost g s := by
    have := expect_mono hcost g s
    rwa [expect_const] at this
  exact (hB.trans hb).trans (div_le_div_of_nonneg_right ht (by positivity))

variable {n : ℕ} {D : Flock.Desc} {hU : Built D.unroll.unit} {kd K₀ : ℕ} {desc : Claims.Msg}
  (A : LiveAt n D hU kd K₀ desc)

/-- The plan after draw `ω`, at `τ`. -/
noncomputable abbrev tsAt (τ : Strategy A.G) (ω : A.L.Ω) := planZHJ A.dj A.y₀ (A.L.draw ω) (reg (A.sim τ))

/-- The session strategy after draw `ω`, at `τ`. -/
noncomputable abbrev zcAt (τ : Strategy A.G) (ω : A.L.Ω) :=
  zcStrat execArith hs512 (planZHJ A.dj A.y₀) (cont (A.sim τ) ω)

/-- **A table's knowledge finders**: level `ℓ + 1`'s rewinding finder of rep `r`, the pads' and `τ`'s of rep `r`,
level 0's and the self-clash finder. -/
abbrev TabIx (levels : ℕ) : Type := (Fin 2 × Fin levels) ⊕ Fin 2 ⊕ Fin 2 ⊕ Unit ⊕ Unit

/-- **Table `j`'s knowledge finders after draw `ω`**, each `SessCRZ`'s (through `TableCRZ`). -/
noncomputable def tabComp (τ : Strategy A.G) (ω : A.L.Ω) (j : Fin (A.tsAt τ ω).length) :
    TabIx ((A.tsAt τ ω).get j).sch.levels.length → Claims.Finder
  | .inl (r, ℓ) => strictComp (advRSFinderZ execArith hs512 ((A.tsAt τ ω).get j)
      ((gamesZC execArith hs512 (A.tsAt τ ω) (A.zcAt τ ω).1).take j)
      ((gamesZC execArith hs512 (A.tsAt τ ω) (A.zcAt τ ω).1).drop (j + 1)) ((A.zcAt τ ω).1 j) r.val (ℓ.val + 1)
      (stratZ execArith hs512 (A.tsAt τ ω) j (A.zcAt τ ω))) (A.qF τ)
  | .inr (.inl r) => strictComp (advPadsSFinderZ execArith hs512 ((A.tsAt τ ω).get j)
      ((gamesZC execArith hs512 (A.tsAt τ ω) (A.zcAt τ ω).1).take j)
      ((gamesZC execArith hs512 (A.tsAt τ ω) (A.zcAt τ ω).1).drop (j + 1)) ((A.zcAt τ ω).1 j) r.val
      (stratZ execArith hs512 (A.tsAt τ ω) j (A.zcAt τ ω))) (A.qF τ)
  | .inr (.inr (.inl r)) => strictComp (advTauSFinderZ execArith hs512 ((A.tsAt τ ω).get j)
      ((gamesZC execArith hs512 (A.tsAt τ ω) (A.zcAt τ ω).1).take j)
      ((gamesZC execArith hs512 (A.tsAt τ ω) (A.zcAt τ ω).1).drop (j + 1)) ((A.zcAt τ ω).1 j) r.val
      (stratZ execArith hs512 (A.tsAt τ ω) j (A.zcAt τ ω))) (A.qF τ)
  | .inr (.inr (.inr (.inl _))) => strictComp (adv₀SFinderZ execArith hs512 ((A.tsAt τ ω).get j)
      ((gamesZC execArith hs512 (A.tsAt τ ω) (A.zcAt τ ω).1).take j)
      ((gamesZC execArith hs512 (A.tsAt τ ω) (A.zcAt τ ω).1).drop (j + 1)) ((A.zcAt τ ω).1 j)
      (stratZ execArith hs512 (A.tsAt τ ω) j (A.zcAt τ ω))) (A.qF τ)
  | .inr (.inr (.inr (.inr _))) => strictComp (selfSFinderZ execArith hs512 ((A.tsAt τ ω).get j)
      ((gamesZC execArith hs512 (A.tsAt τ ω) (A.zcAt τ ω).1).take j)
      ((gamesZC execArith hs512 (A.tsAt τ ω) (A.zcAt τ ω).1).drop (j + 1)) ((A.zcAt τ ω).1 j)
      (stratZ execArith hs512 (A.tsAt τ ω) j (A.zcAt τ ω))) (A.qS τ)

/-- The committed positions some draw reads, where a link finder exists. -/
abbrev LinkPos : Type :=
  {c : A.vb.Pos // Nonempty (ΩcZ execArith A.P (planZHJ A.dj A.y₀) (tabZHJ A.dj A.y₀) A.vb A.L c)}

/-- **Position `c`'s link finder**, `LinkCRZC`'s: a trial and up to `M` more, at cost `t' τ` per run of `τ`. -/
noncomputable def linkComp (τ : Strategy A.G) (c : A.LinkPos) : Claims.Finder :=
  haveI := c.2
  (Game.tapeFinder
    (finderG (trialGZC execArith hs512 A.P (planZHJ A.dj A.y₀) (tabZHJ A.dj A.y₀) A.vb (reg (A.sim τ))
      (cont (A.sim τ)) c.1 A.k A.Rw) A.M)
    (finderStrat _ (trialStratZC execArith hs512 A.P (planZHJ A.dj A.y₀) (tabZHJ A.dj A.y₀) A.vb (reg (A.sim τ))
      (cont (A.sim τ)) c.1 A.k A.Rw) A.M)
    (outCZC execArith hs512 A.P (planZHJ A.dj A.y₀) (tabZHJ A.dj A.y₀) A.vb (reg (A.sim τ)) (cont (A.sim τ)) c.1)
    (fun x => ⌈finderCost (rdTZC execArith hs512 A.P (planZHJ A.dj A.y₀) (tabZHJ A.dj A.y₀) A.vb (reg (A.sim τ))
      (cont (A.sim τ)) c.1) (cTZC (planZHJ A.dj A.y₀) (reg (A.sim τ)) (A.t' τ)) x⌉₊))

/-- The most levels a table's schedule has, at any draw and table. -/
noncomputable def lvMax : ℕ :=
  Finset.univ.sup fun p : Finset (Fin n) × Fin (nTab A.I) => (tableZHJ A.dj A.y₀ p.1 p.2).sch.levels.length

theorem tsAt_length (τ : Strategy A.G) (ω : A.L.Ω) : (A.tsAt τ ω).length = nTab A.I :=
  planZHJ_length _ _ _ _

/-- Table `j` of the plan after draw `ω`. -/
abbrev tabOf (τ : Strategy A.G) (ω : A.L.Ω) (j : Fin (nTab A.I)) : Fin (A.tsAt τ ω).length :=
  Fin.cast (A.tsAt_length τ ω).symm j

/-- Slot `s` of the table kinds at `m` levels, at a table of `l` levels: the same kind, where the table has that
level. -/
def slotAt {m : ℕ} (l : ℕ) : TabIx m → Option (TabIx l)
  | .inl (r, ℓ) => if h : ℓ.val < l then some (.inl (r, ⟨ℓ.val, h⟩)) else none
  | .inr x => some (.inr x)

/-- **Slot `s` of table `j` after draw `ω`**: that table's finder of that kind, and `Claims.Finder.null` at a level
slot past the table's levels. -/
noncomputable def tabAt (τ : Strategy A.G) (ω : A.L.Ω) (j : Fin (nTab A.I)) (s : TabIx A.lvMax) : Claims.Finder :=
  match slotAt ((A.tsAt τ ω).get (A.tabOf τ ω j)).sch.levels.length s with
  | some s' => A.tabComp τ ω (A.tabOf τ ω j) s'
  | none => Claims.Finder.null

/-- **Slot `s` of table `j`, the draw inside**: draws `ω` with its own coins, uniformly as the law does, and runs
`tabAt τ ω j s`. -/
noncomputable def tabMix (τ : Strategy A.G) (j : Fin (nTab A.I)) (s : TabIx A.lvMax) : Claims.Finder :=
  Claims.Finder.mix fun ω : A.L.Ω => A.tabAt τ ω j s

theorem tabMix_Pr (τ : Strategy A.G) (j : Fin (nTab A.I)) (s : TabIx A.lvMax) :
    (A.tabMix τ j s).Pr Verity.sha512Msg = Claims.Ex fun ω : A.L.Ω => (A.tabAt τ ω j s).Pr Verity.sha512Msg :=
  Claims.Finder.mix_Pr _ _

/-- **The finders**: each table's finder of each kind (the draw inside), every link finder, and one more (`none`).
None is indexed by the draw. -/
abbrev Ix : Type := Option ((Fin (nTab A.I) × TabIx A.lvMax) ⊕ A.LinkPos)

instance instFiniteIx : Finite A.Ix := by
  classical
  infer_instance

noncomputable instance instFintypeIx : Fintype A.Ix := Fintype.ofFinite _

/-- C-Flock's finders at `τ`, and `extra`. -/
noncomputable def crFam (τ : Strategy A.G) (extra : Claims.Finder) : A.Ix → Claims.Finder
  | none => extra
  | some (.inl (j, s)) => A.tabMix τ j s
  | some (.inr c) => A.linkComp τ c

/-- **Sampled proofs' finder `B^τ`**: one of C-Flock's finders at `τ` and `extra`, picked uniformly with its own coins,
run on its own coins. -/
noncomputable def crFinder (τ : Strategy A.G) (extra : Claims.Finder) : Claims.Finder :=
  Claims.Finder.mix (A.crFam τ extra)

/-- **The number of finders**: where every table's schedule, at every draw, has at most `lv` levels, `k = card Ix` is
at most `2 lv + 6` per table, one per committed position, and one for `extra`. No draw, strategy or count of draws is
in the bound. -/
theorem card_Ix_le {lv : ℕ}
    (hlv : ∀ (S : Finset (Fin n)) (j : Fin (nTab A.I)), (tableZHJ A.dj A.y₀ S j).sch.levels.length ≤ lv) :
    Fintype.card A.Ix ≤ nTab A.I * (2 * lv + 6) + Fintype.card A.vb.Pos + 1 := by
  classical
  have e : Fintype.card A.Ix = Fintype.card (Option ((Fin (nTab A.I) × TabIx A.lvMax) ⊕ A.LinkPos)) :=
    Fintype.card_congr (Equiv.refl _)
  have hm : A.lvMax ≤ lv := Finset.sup_le fun p _ => hlv p.1 p.2
  have hT : Fintype.card (TabIx A.lvMax) = 2 * A.lvMax + 6 := by
    simp only [TabIx, Fintype.card_sum, Fintype.card_prod, Fintype.card_fin, Fintype.card_unit] <;> omega
  rw [e, Fintype.card_option, Fintype.card_sum, Fintype.card_prod, Fintype.card_fin, hT]
  have hL : Fintype.card A.LinkPos ≤ Fintype.card A.vb.Pos := by convert Fintype.card_subtype_le _
  have hN : nTab A.I * (2 * A.lvMax + 6) ≤ nTab A.I * (2 * lv + 6) := Nat.mul_le_mul_left _ (by omega)
  exact Nat.add_le_add_right (Nat.add_le_add hN hL) 1

/-- **`extra` succeeds at most `k` times as often as `B^τ`.** -/
theorem extra_le (τ : Strategy A.G) (extra : Claims.Finder) :
    extra.Pr Verity.sha512Msg ≤ Fintype.card A.Ix * (A.crFinder τ extra).Pr Verity.sha512Msg :=
  Claims.Finder.le_mix Verity.sha512Msg (A.crFam τ extra) none

theorem strictComp_cost_le (Fd : StrictCR.Finder StrictCR.Pair) (q : ℝ) {b : ℕ} (h : ⌈q⌉₊ ≤ b)
    (x : (strictComp Fd q).S) : (strictComp Fd q).cost x ≤ b :=
  Game.tapeFinder_cost_le _ _ _ _ b (fun _ => h) x

theorem tabComp_cost_le (τ : Strategy A.G) (ω : A.L.Ω) (j : Fin (A.tsAt τ ω).length)
    (s : TabIx ((A.tsAt τ ω).get j).sch.levels.length) {b : ℕ} (hF : ⌈A.qF τ⌉₊ ≤ b) (hS : ⌈A.qS τ⌉₊ ≤ b) :
    ∀ x, (A.tabComp τ ω j s).cost x ≤ b := by
  rcases s with ⟨r, ℓ⟩ | r | r | _ | _
  · exact strictComp_cost_le _ _ hF
  · exact strictComp_cost_le _ _ hF
  · exact strictComp_cost_le _ _ hF
  · exact strictComp_cost_le _ _ hF
  · exact strictComp_cost_le _ _ hS

theorem tabAt_cost_le (τ : Strategy A.G) (ω : A.L.Ω) (j : Fin (nTab A.I)) (s : TabIx A.lvMax) {b : ℕ}
    (hF : ⌈A.qF τ⌉₊ ≤ b) (hS : ⌈A.qS τ⌉₊ ≤ b) : ∀ x, (A.tabAt τ ω j s).cost x ≤ b := by
  unfold tabAt
  split
  · exact A.tabComp_cost_le τ ω _ _ hF hS
  · exact fun _ => Nat.zero_le _

/-- **`B^τ`'s cost on every coin** is at most `b` where the rewinding and self-clash budgets are, every link finder's
run is, and `extra`'s cost is. -/
theorem crFinder_cost_le (τ : Strategy A.G) (extra : Claims.Finder) {b : ℕ} (hE : ∀ x, extra.cost x ≤ b) (hF : ⌈A.qF τ⌉₊ ≤ b) (hS : ⌈A.qS τ⌉₊ ≤ b)
    (hL : ∀ (c : A.vb.Pos) (x : TrialOutZC FlockLevel3.GF128 A.L Dig (List UInt8) ×
        List (TrialOutZC FlockLevel3.GF128 A.L Dig (List UInt8))),
      finderCost (rdTZC execArith hs512 A.P (planZHJ A.dj A.y₀) (tabZHJ A.dj A.y₀) A.vb (reg (A.sim τ))
        (cont (A.sim τ)) c) (cTZC (planZHJ A.dj A.y₀) (reg (A.sim τ)) (A.t' τ)) x ≤ b)
    (x : (A.crFinder τ extra).S) : (A.crFinder τ extra).cost x ≤ b := by
  refine Claims.Finder.mix_cost_le _ b (fun i => ?_) x
  rcases i with _ | ⟨j, s⟩ | c
  · exact hE
  · exact Claims.Finder.mix_cost_le _ b fun ω => A.tabAt_cost_le τ ω j s hF hS
  · exact Game.tapeFinder_cost_le _ _ _ _ b (fun _ => Nat.ceil_le.mpr (hL c.1 _))

/-- A link trial plays `τ` at least once: its cost is at least `t' τ`. -/
theorem cTZC_ge (τ : Strategy A.G) (e : TrialOutZC FlockLevel3.GF128 A.L Dig (List UInt8)) :
    A.t' τ ≤ cTZC (planZHJ A.dj A.y₀) (reg (A.sim τ)) (A.t' τ) e := by
  unfold cTZC
  have h0 := A.ht' τ
  split_ifs
  · refine le_mul_of_one_le_right h0 (le_add_of_nonneg_right (List.sum_nonneg ?_))
    simp only [List.mem_map]
    rintro _ ⟨b, -, rfl⟩
    exact Extract.waitCost_nonneg _ b
  · simp

/-- **A link finder's cost is at least one run's**, `t' τ`. -/
theorem linkCost_ge (τ : Strategy A.G) (c : A.vb.Pos) (x : TrialOutZC FlockLevel3.GF128 A.L Dig (List UInt8) ×
    List (TrialOutZC FlockLevel3.GF128 A.L Dig (List UInt8))) :
    A.t' τ ≤ finderCost (rdTZC execArith hs512 A.P (planZHJ A.dj A.y₀) (tabZHJ A.dj A.y₀) A.vb (reg (A.sim τ))
      (cont (A.sim τ)) c) (cTZC (planZHJ A.dj A.y₀) (reg (A.sim τ)) (A.t' τ)) x := by
  have h1 := A.cTZC_ge τ x.1
  unfold finderCost
  split_ifs
  · exact le_add_of_le_of_nonneg h1 (CROnly.loopCost_nonneg _ (fun e => (A.ht' τ).trans (A.cTZC_ge τ e)) _)
  · simpa using h1

/-- **The table finders' premise, averaged over the draw**: each table's finder of each kind outputs a SHA-512 collision
with probability at most `β`, averaged over the draw (uniformly, as the law draws). -/
def TabCRAvg (τ : Strategy A.G) (β : ℝ) : Prop :=
  ∀ j s, Claims.Ex (fun ω : A.L.Ω => (A.tabAt τ ω j s).Pr Verity.sha512Msg) ≤ β

/-- **The link finders' premise** (`LiveAt.CR`'s second part). -/
def LinkCR (τ : Strategy A.G) : Prop :=
  LinkCRZC execArith hs512 A.P (planZHJ A.dj A.y₀) (tabZHJ A.dj A.y₀) A.vb (reg (A.sim τ)) (cont (A.sim τ)) A.k A.Rw
    A.M (A.t' τ)

/-- Each of `B^τ`'s finders succeeds at most `k β` where `B^τ` succeeds at most `β`. -/
theorem crFam_le (τ : Strategy A.G) (extra : Claims.Finder) {β : ℝ}
    (hB : (A.crFinder τ extra).Pr Verity.sha512Msg ≤ β) (i : A.Ix) :
    (A.crFam τ extra i).Pr Verity.sha512Msg ≤ Fintype.card A.Ix * β :=
  (Claims.Finder.le_mix Verity.sha512Msg (A.crFam τ extra) i).trans
    (mul_le_mul_of_nonneg_left hB (Nat.cast_nonneg _))

/-- **The table premise from the threshold, averaged over the draw.** Where `B^τ` (`crFinder τ extra`, any `extra`)
outputs a SHA-512 collision with probability at most `β`, every table finder of every kind does, averaged over the
draw, with probability at most `k β` (`k = card Ix`). -/
theorem tabCRAvg_of_crFinder (τ : Strategy A.G) (extra : Claims.Finder) {β : ℝ}
    (hB : (A.crFinder τ extra).Pr Verity.sha512Msg ≤ β) : A.TabCRAvg τ (Fintype.card A.Ix * β) := fun j s =>
  (A.tabMix_Pr τ j s).symm.trans_le (A.crFam_le τ extra hB (some (.inl (j, s))))

/-- **The link premise from the threshold.** Where `B^τ` outputs a SHA-512 collision with probability at most `β` and
`k β ≤ t' τ / 2^256`, every link finder meets `LinkCRZC`'s premise. -/
theorem linkCR_of_crFinder (τ : Strategy A.G) (extra : Claims.Finder) {β : ℝ}
    (hB : (A.crFinder τ extra).Pr Verity.sha512Msg ≤ β)
    (hbL : Fintype.card A.Ix * β ≤ A.t' τ / 2 ^ (256 : ℝ)) : A.LinkCR τ := by
  unfold LinkCR
  intro c hc
  exact expectedCR_of _ _ _ _ _ (A.linkCost_ge τ c) (A.crFam_le τ extra hB (some (.inr ⟨c, hc⟩))) hbL

end Flock.Guarantees.LiveAt

namespace Verity

open Accounting Core

variable {ι α : Type} [Fintype α] {d : Device ι}

/-- **What the vocabulary's finder premise gives C-Flock.** At a vocabulary whose sampled finder, at registered files
that parse, is `crFinder` at the session's strategy and some `extra` (`hfinder`): every table finder succeeds, averaged
over the draw, at most `k ε_cr` (`k = card Ix`, `ε_cr` the published bound at the strategy's calls), and where
`k ε_cr ≤ t' τ / 2^256` (`hbL`) every link finder meets its premise. -/
theorem SampledDeploy.crAvg_of (V : Vocab d α) (kd : ByteArray → ℕ) (desc : ByteArray → Claims.Msg)
    (A : ∀ (θ F c : ByteArray) (_σ : Game.Strategy (V.game θ F c)) (r : Flock.Bridge.Parsed), V.parsed F = some r →
      Flock.Guarantees.LiveAt r.n r.D r.hU (kd θ) (⌊(V.certOf θ).ε * r.n⌋₊ + 1) (desc θ))
    (τOf : ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (r : Flock.Bridge.Parsed)
      (hr : V.parsed F = some r), FlockSoundness.Game.Strategy (A θ F c σ r hr).G)
    (extra : ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (r : Flock.Bridge.Parsed),
      V.parsed F = some r → Claims.Finder)
    (hfinder : ∀ θ F c σ r hr, V.Registered θ F →
      V.finder .sampled θ F c σ = (A θ F c σ r hr).crFinder (τOf θ F c σ r hr) (extra θ F c σ r hr))
    (hbL : ∀ θ F c σ r hr, Fintype.card (A θ F c σ r hr).Ix * V.crBound θ .sampled (V.cost θ F c σ) ≤
      (A θ F c σ r hr).t' (τOf θ F c σ r hr) / 2 ^ (256 : ℝ)) :
    ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (r : Flock.Bridge.Parsed)
      (hr : V.parsed F = some r), V.Registered θ F → V.FinderCR .sampled θ F c σ →
      (A θ F c σ r hr).TabCRAvg (τOf θ F c σ r hr)
          (Fintype.card (A θ F c σ r hr).Ix * V.crBound θ .sampled (V.cost θ F c σ)) ∧
        (A θ F c σ r hr).LinkCR (τOf θ F c σ r hr) := by
  intro θ F c σ r hr hreg hcr
  have hB : ((A θ F c σ r hr).crFinder (τOf θ F c σ r hr) (extra θ F c σ r hr)).Pr sha512Msg ≤
      V.crBound θ .sampled (V.cost θ F c σ) := by
    rw [← hfinder θ F c σ r hr hreg]
    exact hcr
  exact ⟨(A θ F c σ r hr).tabCRAvg_of_crFinder _ _ hB, (A θ F c σ r hr).linkCR_of_crFinder _ _ hB (hbL θ F c σ r hr)⟩

/-- **`SampledDeploy`'s `finder_ge` field.** At a vocabulary whose sampled finder, at registered files that parse, is
`crFinder` with the play's replay finder of `col` in its spare slot (`hfinder`, `crAvg_of`'s at that `extra`), and
where `kMax θ` is at least the number of C-Flock's finders (`hk`, from `LiveAt.card_Ix_le`), the replay finder succeeds
at most `kMax θ` times as often as the vocabulary's finder (`extra_le`). -/
theorem SampledDeploy.finder_ge_of (V : Vocab d α) (kd : ByteArray → ℕ) (desc : ByteArray → Claims.Msg)
    (A : ∀ (θ F c : ByteArray) (_σ : Game.Strategy (V.game θ F c)) (r : Flock.Bridge.Parsed), V.parsed F = some r →
      Flock.Guarantees.LiveAt r.n r.D r.hU (kd θ) (⌊(V.certOf θ).ε * r.n⌋₊ + 1) (desc θ))
    (τOf : ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (r : Flock.Bridge.Parsed)
      (hr : V.parsed F = some r), FlockSoundness.Game.Strategy (A θ F c σ r hr).G)
    (col : ByteArray → ByteArray → ByteArray → Outcome → Option (Claims.Msg × Claims.Msg)) (kMax : ByteArray → ℕ)
    (hfinder : ∀ θ F c σ r hr, V.Registered θ F →
      V.finder .sampled θ F c σ = (A θ F c σ r hr).crFinder (τOf θ F c σ r hr)
        (V.replayFinder θ F c σ (col θ F c) (V.cost θ F c σ)))
    (hk : ∀ θ F c σ r hr, Fintype.card (A θ F c σ r hr).Ix ≤ kMax θ) :
    ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (r : Flock.Bridge.Parsed),
      V.parsed F = some r → V.Registered θ F →
      (V.replayFinder θ F c σ (col θ F c) (V.cost θ F c σ)).Pr sha512Msg ≤
        kMax θ * (V.finder .sampled θ F c σ).Pr sha512Msg := by
  intro θ F c σ r hr hreg
  rw [hfinder θ F c σ r hr hreg]
  exact ((A θ F c σ r hr).extra_le _ _).trans
    (mul_le_mul_of_nonneg_right (by exact_mod_cast hk θ F c σ r hr) (Claims.Pr_nonneg _))

end Verity
