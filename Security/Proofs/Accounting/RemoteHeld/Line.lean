import Proofs.Accounting.RemoteHeld.Window
import Proofs.Accounting.RemoteHeld.Draw
import Properties.Proofs.RemoteAudit.Line
import Properties.Proofs.RemoteAudit.Pooled

/-!
# The memory line of record, proved (`Properties.RemoteAudit.Line`: `LineHeld`, `LineHeldIdentity`)

At each outcome where the Lean verifier accepts the record's session (so the trace of record accepts, `ofRecord`), the
receipts' sketch holds and the trace's wrong receipts are at most the session's pooled bound at the line's `δ`
(`holdsAnd` of `SketchBound` and `Draw.pooledDraw`, so at most the W the line reads: `LineCoversSession` at the
session's windows, `DrawsOfRecord`), every stretch of the record's line credits what the trace held (`heldAt_of`):

* the arithmetic link (`stretchBound`, `Properties.Proofs.RemoteAudit.Line`) bounds the stretch's credit by the left-hand
  side of `HeldOverTimeLinks` at its paced session `stretchPace`, over its span from `spanStart`;
* `heldOverWindow_of` bounds that by what the trace held over the span: the stretch's paced session is the record's
  (`recordPace`, `Paced`) from group `m·per` on, `spanStart` is its first group's end or coin, and the form's charge is
  the model's rates in its units (`charge_units`);
* the weights' credit follows from the held credit (`weights_le`): a set of `w` registered blocks misses at most
  `N − w` of those in node memory at a tick (`card_le_inter`).
-/

namespace Properties.RemoteAudit.Line.SecurityProofs

open Finset Accounting.Remote.Held Accounting.Remote.Held.Assumptions Accounting.Remote.Draw
  Accounting.Remote.Draw.Assumptions _root_.RemoteAudit.Line Accounting.SecurityProofs.Remote Properties.RemoteAudit.Line

/-- `firstPass` grows with the groups it counts. -/
theorem firstPass_mono {gs : Array Group} {a b : ℕ} (h : a ≤ b) : firstPass gs a ≤ firstPass gs b := by
  induction b, h using Nat.le_induction with
  | base => exact le_rfl
  | succ b _ ih => rw [firstPass_succ]; omega

/-- The registered blocks of a set `R` in node memory at a tick are at least the blocks in node memory there less
those outside `R`. -/
theorem card_le_inter {N : ℕ} (M R : Finset (Fin N)) : (M.card : ℝ) ≤ (M ∩ R).card + (N - R.card) := by
  have h1 : (M \ R).card + (M ∩ R).card = M.card := Finset.card_sdiff_add_card_inter M R
  have h2 : (M \ R).card ≤ (univ \ R).card := Finset.card_le_card (Finset.sdiff_subset_sdiff (subset_univ M) le_rfl)
  rw [Finset.card_univ_sdiff, Fintype.card_fin] at h2
  have hR : R.card ≤ N := by simpa using Finset.card_le_univ R
  have : ((M \ R).card : ℝ) ≤ (N : ℝ) - R.card := by
    rw [← Nat.cast_sub hR]; exact_mod_cast h2
  have h1' : ((M \ R).card : ℝ) + (M ∩ R).card = M.card := by exact_mod_cast h1
  linarith

/-- Each form's charge in the model's units: `X = s/E + β (+ β_l)` and `B = B_r + B_d (+ B_l)`. -/
theorem charge_units (node : Node) :
    ((charges node.rates node.blockBytes).1.rate : ℝ) = node.squarings / node.encode + node.disk + 0 ∧
    ((charges node.rates node.blockBytes).1.burst : ℝ) = node.rebuildBurst + node.diskBurst + 0 ∧
    ((charges node.rates node.blockBytes).2.rate : ℝ) = node.squarings / node.encode + node.disk + node.link ∧
    ((charges node.rates node.blockBytes).2.burst : ℝ) = node.rebuildBurst + node.diskBurst + node.linkBurst := by
  simp only [charges, ns, Node.squarings, Node.encode, Node.disk, Node.link, Node.rebuildBurst, Node.diskBurst,
    Node.linkBurst]
  push_cast
  refine ⟨by ring, by ring, by ring, by ring⟩

/-- The weights' credit from the held credit: a set `R` of at least `w` registered blocks has, summed over a span, at
least `span · (w − lost)` blocks in node memory when all of node memory has at least `span · (N − lost)`. -/
theorem weights_le {Ω : Type} {T : Trace Ω} {ω : Ω} {u v : ℕ} {x : Stretch} {b : ℕ}
    (hwts : x.time.weights = max 0 (x.weights - x.time.lost) * (b : ℚ)) {R : Finset (Fin T.blocks)}
    (hR : (x.weights : ℝ) ≤ R.card) (hheld : (x.spanNs : ℝ) * ((T.blocks : ℝ) - x.time.lost) ≤ T.heldOver ω u v)
    (hspan : x.spanNs = v - u) :
    (x.spanNs : ℝ) * x.time.weights ≤ b * ∑ t ∈ Ico u v, (T.mem ω t ∩ R).card := by
  have hb0 : (0 : ℝ) ≤ b := Nat.cast_nonneg _
  have hsum : (T.heldOver ω u v : ℝ) ≤ ∑ t ∈ Ico u v, ((T.mem ω t ∩ R).card : ℝ) + (x.spanNs : ℝ) * (T.blocks - R.card) := by
    simp only [Trace.heldOver, Nat.cast_sum]
    have : (x.spanNs : ℝ) * (T.blocks - R.card) = ∑ t ∈ Ico u v, ((T.blocks : ℝ) - R.card) := by
      rw [Finset.sum_const, Nat.card_Ico, nsmul_eq_mul, hspan]
    rw [this, ← Finset.sum_add_distrib]
    exact Finset.sum_le_sum fun t _ => card_le_inter _ _
  have hs0 : (0 : ℝ) ≤ x.spanNs := Nat.cast_nonneg _
  rw [hwts]
  push_cast
  rcases le_total (x.weights : ℝ) (x.time.lost : ℝ) with hle | hle
  · rw [max_eq_left (by linarith)]
    have : (0 : ℝ) ≤ ∑ t ∈ Ico u v, ((T.mem ω t ∩ R).card : ℝ) := Finset.sum_nonneg fun _ _ => Nat.cast_nonneg _
    nlinarith
  · rw [max_eq_right (by linarith)]
    have h1 : (x.spanNs : ℝ) * ((x.weights : ℝ) - x.time.lost) ≤ ∑ t ∈ Ico u v, ((T.mem ω t ∩ R).card : ℝ) := by
      have := mul_le_mul_of_nonneg_left hR hs0
      nlinarith
    calc (x.spanNs : ℝ) * (((x.weights : ℝ) - x.time.lost) * b)
        = b * ((x.spanNs : ℝ) * ((x.weights : ℝ) - x.time.lost)) := by ring
      _ ≤ b * ∑ t ∈ Ico u v, ((T.mem ω t ∩ R).card : ℝ) := mul_le_mul_of_nonneg_left h1 hb0

/-- On an accepting outcome where the receipts' sketch holds and the trace's wrong receipts are at most the W the
record's line reads, every stretch of form `i` credits what the trace held over its span. -/
theorem heldAt_of {Ω : Type} [Fintype Ω] {T : Trace Ω} {node : Node} {rec : Ω → Record} (i : ℕ) (hi : i < 2)
    {βl Bl : ℝ} (hβl : 0 ≤ node.rates.linkBytesPerS → 0 ≤ βl) (hl : Links T βl Bl)
    (hcr : (((if i = 0 then (charges node.rates node.blockBytes).1 else (charges node.rates node.blockBytes).2).rate
      : ℚ) : ℝ) = node.squarings / node.encode + node.disk + βl)
    (hcb : (((if i = 0 then (charges node.rates node.blockBytes).1 else (charges node.rates node.blockBytes).2).burst
      : ℚ) : ℝ) = node.rebuildBurst + node.diskBurst + Bl)
    (hsq : SquaringCeiling T node.squarings node.encode node.rebuildBurst) (hd : Disks T node.disk node.diskBurst)
    (hS : ChipCeiling T node.chipBlocks) (hr : Routed T) (hp : Paced T (recordPace node rec)) (hcov : Covers T)
    (hN : Registered T node) (hW : WindowsOfRecord T node rec) {ω : Ω} (ha : T.accepts ω)
    (hk : ∀ p < T.passes, ∀ j ∈ T.correct ω p, ∃ t, T.tick ω p ≤ t ∧ t < T.tick ω (p + 1) ∧ j ∈ T.present ω t)
    (hw : ∀ l, lineOf node (rec ω) = .ok l → (T.wrong ω : ℝ) ≤ l.wrong) :
    HeldAt T ω node (rec ω) i := by
  intro l hline f hf x hx
  obtain ⟨hb, hE, hs0, hd0, hl0, -, -, -, -⟩ := lineChecked _ _ _ _ _ _ _ _ _ _ hline
  obtain ⟨hhi, hne, hspan, hstop, hstart, hch, hbound, hwts⟩ := stretchBound _ _ _ _ _ _ _ _ _ _ hline i hi f hf x hx
  set gs := (rec ω).groups with hgs
  set m := x.stretch with hm
  set per := node.per with hper
  obtain ⟨hmono, -, hfP, hfg, hrel, hre, hpass⟩ := hp ω ha
  have hG : gs.size = node.windows := hW ω ha
  have hC : (recordPace node rec).credited = node.windows - 1 := rfl
  have hlo : ∀ g ≤ per, m * per + g ≤ node.windows - 1 := fun g hg => by
    have : m * per + per = (m + 1) * per := by rw [Nat.succ_mul]
    omega
  let K : Pace Ω := stretchPace (fun ω => (rec ω).groups) per m
  have hKf : ∀ g, K.first ω g = (recordPace node rec).first ω (m * per + g) := fun g => rfl
  have hKr : ∀ g, K.released ω g = (recordPace node rec).released ω (m * per + g) := fun g => by
    show released gs gs.size (m * per + g) = released gs node.windows (m * per + g)
    rw [hG]
  have hKe : ∀ g, K.ended ω g = (recordPace node rec).ended ω (m * per + g) := fun g => rfl
  have hKC : K.credited = per := rfl
  have hfP' : K.first ω (K.credited + 1) ≤ T.passes := by
    rw [hKf, hKC]
    refine le_trans (firstPass_mono ?_) (hC ▸ hfP)
    have : m * per + per = (m + 1) * per := by rw [Nat.succ_mul]
    omega
  have hfg' : ∀ g ≤ K.credited, K.first ω g < K.first ω (g + 1) := fun g hg => by
    rw [hKf, hKf]; exact hfg _ (hlo g hg)
  have hrel' : Monotone (K.released ω) := fun a b hab => by
    rw [hKr, hKr]; exact hrel (by omega)
  have hre' : ∀ g ≤ K.credited, K.released ω (g + 1) ≤ K.ended ω g := fun g hg => by
    rw [hKr, hKe]; exact hre _ (hlo g hg)
  have hpass' : ∀ g ≤ K.credited, ∀ p, K.first ω g ≤ p → p < K.first ω (g + 1) →
      K.released ω g ≤ T.tick ω p ∧ T.tick ω (p + 1) ≤ K.ended ω g := fun g hg p h1 h2 => by
    rw [hKr, hKe]; exact hpass _ (hlo g hg) p h1 h2
  have hre0 : K.released ω 0 ≤ K.ended ω 0 := by
    have h01 := hfg' 0 (Nat.zero_le _)
    obtain ⟨h1, h2⟩ := hpass' 0 (Nat.zero_le _) (K.first ω 0) le_rfl h01
    exact h1.trans ((hmono (Nat.le_succ _)).trans h2)
  have hKs : K.stop ω = spanStop gs per m := hstop.symm
  have hu : K.released ω 0 ≤ spanStart gs per m ∧ spanStart gs per m ≤ K.ended ω 0 := by
    rcases hstart with h | h
    · exact ⟨h ▸ hre0, h ▸ le_rfl⟩
    · exact ⟨h ▸ le_rfl, h ▸ hre0⟩
  have hE' : (0 : ℝ) < node.encode := by simp only [Node.encode]; exact_mod_cast hE
  have hs' : (0 : ℝ) ≤ node.squarings := by simp only [Node.squarings, ns]; push_cast; positivity
  have hβ' : (0 : ℝ) ≤ node.disk := by simp only [Node.disk, ns]; push_cast; positivity
  have hwin := heldOverWindow_of K ω hE' hs' hβ' (hβl hl0) hl hsq hd hS hr hfP' hfg' hrel' hre' hpass' (hcov ω ha) hu.1 hu.2
    hk (hw l hline)
  rw [hKs, ← hspan] at hwin
  have hL : K.longest ω = (stretchPace (fun _ : Unit => gs) per m).longest () := rfl
  have hF : K.fill ω T.blocks l.wrong = (stretchPace (fun _ : Unit => gs) per m).fill () node.blocks l.wrong := by
    rw [hN]; rfl
  rw [hL, hF] at hwin
  rw [hch, hcr, hcb] at hbound
  have hNr : (T.blocks : ℝ) = node.blocks := by rw [hN]
  rw [hNr] at hwin
  have hheld : (x.spanNs : ℝ) * ((T.blocks : ℝ) - x.time.lost) ≤
      T.heldOver ω (spanStart gs per m) (spanStop gs per m) := by
    rw [hNr]
    simp only [Node.chipBlocks] at hwin
    linarith
  exact ⟨hspan, hheld, fun R hR => weights_le hwts hR hheld hspan⟩

/-- Where the record's windows are the session's, the trace's wrong receipts below the session's pooled bound at the
line's `δ` are below the W every line of the record reads (`LineCoversSession`). -/
theorem wrong_le_line {n m : ℕ} {k : Fin n → ℕ} {X : Fin n → Type} {D : Session n m k X} {T : Trace (Coins n m k X)}
    {node : Node} {rec : Coins n m k X → Record} (hD : DrawsOfRecord D T rec) {ω : Coins n m k X} (ha : T.accepts ω)
    (hw : (T.wrong ω : ℝ) ≤ D.pooled node.delta ω) :
    ∀ l, lineOf node (rec ω) = .ok l → (T.wrong ω : ℝ) ≤ l.wrong := fun l hline => by
  unfold lineOf at hline
  rw [hD ω ha] at hline
  have hc := Properties.RemoteAudit.lineCoversSession n m k X D ω node.delta _ _ _ _ _ _ _ l hline
  rw [Rat.cast_nnratCast] at hc
  exact hw.trans hc

/-- The node's `δ` is nonnegative, in the reals. -/
theorem delta_nonneg (node : Node) : (0 : ℝ) ≤ (node.delta : ℝ) := NNRat.cast_nonneg _

/-- **The memory line of record, link-charged** (`LineHeld`), proved. -/
theorem lineHeld : LineHeld := by
  intro n m k X _ D T node rec δs hl hsq hd hS hr hp hcov hk hL hR hC hN hW hD
  have hH := holdsAnd hk (Draw.pooledDraw D (ofRecord T rec) (delta_nonneg node) hL hR hC)
  refine le_trans (Pouw.pr_mono fun ω hω => ⟨hω.1, fun h => hω.2 ?_⟩) hH
  have ha : (ofRecord T rec).accepts ω := hω.1
  have hc := charge_units node
  refine heldAt_of 1 (by omega) ?_ hl ?_ ?_ hsq hd hS hr hp hcov hN hW ha h.1 (wrong_le_line hD ha h.2)
  · intro h0
    simp only [Node.link, ns]
    push_cast
    positivity
  · simpa using hc.2.2.1
  · simpa using hc.2.2.2

/-- **The memory line of record, time-held, under Identity** (`LineHeldIdentity`), proved. -/
theorem lineHeldIdentity : LineHeldIdentity := by
  intro n m k X _ D T node rec δs hI hsq hd hS hr hp hcov hk hL hR hC hN hW hD
  have hH := holdsAnd hk (Draw.pooledDraw D (ofRecord T rec) (delta_nonneg node) hL hR hC)
  refine le_trans (Pouw.pr_mono fun ω hω => ⟨hω.1, fun h => hω.2 ?_⟩) hH
  have ha : (ofRecord T rec).accepts ω := hω.1
  have hc := charge_units node
  exact heldAt_of 0 (by omega) (fun _ => le_rfl) (links_of_identity hI) (by simpa using hc.1) (by simpa using hc.2.1) hsq hd
    hS hr hp hcov hN hW ha h.1 (wrong_le_line hD ha h.2)

end Properties.RemoteAudit.Line.SecurityProofs
