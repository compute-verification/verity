import Proofs.Accounting.RemoteHeld.Window
import Proofs.Accounting.RemoteHeld.Draw
import Proofs.Accounting.RemoteHeld.Outside
import Proofs.Properties.RemoteAudit.Line
import Proofs.Properties.RemoteAudit.Pooled
import Proofs.Properties.RemoteAudit.Session

/-!
# The memory line of record, proved (`Properties.RemoteAudit.Line`: `LineHeld`, `LineHeldIdentity`)

At each outcome where the Lean verifier accepts the record's session (so the trace of record accepts, `ofRecord`), the
receipts' sketch holds and the trace's wrong receipts are at most the session's pooled bound at the line's `δ`
(`holdsAnd` of `SketchBound`, from `SketchBoundAt` and `NoOutsideCompute` by `sketchBound_of_noOutside`, and
`Draw.pooledDraw`, so at most the W the line reads: `LineCoversSession` at the
record's windows, which are the draw's by `SizesOfRecord` and the line's check of their draws, `wrong_le_line`), every
stretch of the line read off the record credits what the trace held (`heldAt_of`), its groups one for each of the
session's windows and its registered blocks the node's (`record_summary`):

* the arithmetic link (`stretchBound`, `Proofs.Properties.RemoteAudit.Line`) bounds the stretch's credit by the left-hand
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

/-- What `ofSummary` reads of a record it accepts: one group a window, and each window's receipts and draws, its draws
`min(s, U_t)`. -/
theorem ofSummary_ok {sm : RemoteAudit.Session.Summary} {samples : ℕ} {seen : Array Seen} {gs : Array Group}
    {ws : Array (ℕ × ℕ)} (h : ofSummary sm samples seen = .ok (gs, ws)) :
    gs.size = sm.windows.size ∧ ws = sm.windows.map fun w => (receipts w, min samples (receipts w)) := by
  unfold ofSummary at h
  split at h
  · cases h
  · split at h
    · cases h
    · rename_i hf
      cases h
      refine ⟨by simp, Array.map_congr_left fun w hw => ?_⟩
      have hd := Array.findIdx?_eq_none_iff.1 hf w hw
      simp only [bne_eq_false_iff_eq] at hd
      simp only [hd]

/-- A line `lineOfRecord` computes is `line` at the record's groups, windows and registered blocks. -/
theorem lineOfRecord_ok {sm : RemoteAudit.Session.Summary} {samples : ℕ} {seen : Array Seen}
    {per blockBytes chipBytes nodeBytes : ℕ} {δ : ℚ} {rates : Rates} {l : Line}
    (h : lineOfRecord sm samples seen per blockBytes chipBytes nodeBytes δ rates = .ok l) :
    ∃ gs ws, ofSummary sm samples seen = .ok (gs, ws) ∧
      line gs per sm.blocks blockBytes chipBytes nodeBytes ws δ rates = .ok l := by
  unfold lineOfRecord at h
  split at h
  · rename_i gs ws hs
    exact ⟨gs, ws, hs, h⟩
  · cases h

/-- A line the record gives is `line` at the groups and windows `ofSummary` reads off the Lean verifier's record of the
session, its registered blocks the record's, and those groups are the record's (`Record.groups`). -/
theorem lineOf_ok {node : Node} {r : Record} {l : Line} (h : lineOf node r = .ok l) :
    ∃ sm gs ws, r.summary node = some sm ∧ ofSummary sm node.params.samples r.seen = .ok (gs, ws) ∧
      r.groups node = gs ∧
      line gs node.per sm.blocks node.blockBytes node.chipBytes node.nodeBytes ws node.delta node.rates = .ok l := by
  unfold lineOf at h
  split at h
  · rename_i sm hsm
    obtain ⟨gs, ws, hs, hl⟩ := lineOfRecord_ok h
    exact ⟨sm, gs, ws, hsm, hs, by simp only [Record.groups, hsm, hs], hl⟩
  · cases h

/-- A state's summary has its windows and its registration's blocks. -/
theorem summary_of {s : RemoteAudit.Session.State} {sm : RemoteAudit.Session.Summary} (h : s.summary = some sm) :
    sm.windows.size = s.windows.size ∧ sm.blocks = s.reg.blocks := by
  unfold RemoteAudit.Session.State.summary at h
  split at h
  · cases h
  · simp only [Option.some.injEq] at h
    subst h
    exact ⟨by simp, rfl⟩

/-- **One group a window**: the Lean verifier's record of a session it accepts has the node's `windows` windows, and
the node's registered blocks. -/
theorem record_summary {node : Node} {r : Record} {sm : RemoteAudit.Session.Summary} (hv : Verified node r)
    (hsm : r.summary node = some sm) : sm.windows.size = node.windows ∧ sm.blocks = node.blocks := by
  obtain ⟨rs, s, hrun, hout⟩ := hv
  have hrun' : RemoteAudit.Session.run (RemoteAudit.Session.State.init node.params node.reg r.coins r.appendCoins
      r.notBeforeNs) r.events = .ok (rs, s) := hrun
  have hs : s.summary = some sm := by
    unfold Record.summary at hsm
    rw [hrun] at hsm
    exact hsm
  obtain ⟨h1, h2⟩ := summary_of hs
  exact ⟨h1.trans (Properties.RemoteAudit.acceptedMinPasses _ _ _ _ _ _ _ _ hrun' hout).1,
    h2.trans (congrArg RemoteAudit.Registration.blocks (Properties.RemoteAudit.inv_run hrun').2.2.1)⟩

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
    (hN : Registered T node) {ω : Ω} (ha : T.accepts ω) (hv : Verified node (rec ω))
    (hk : ∀ p < T.passes, ∀ j ∈ T.correct ω p, ∃ t, T.tick ω p ≤ t ∧ t < T.tick ω (p + 1) ∧ j ∈ T.present ω t)
    (hw : ∀ l, lineOf node (rec ω) = .ok l → (T.wrong ω : ℝ) ≤ l.wrong) :
    HeldAt T ω node (rec ω) i := by
  intro l hline f hf x hx
  obtain ⟨sm, gs₀, ws, hsm, hs, hgs₀, hln⟩ := lineOf_ok hline
  obtain ⟨hsmW, hsmB⟩ := record_summary hv hsm
  subst hgs₀
  obtain ⟨hb, hE, hs0, hd0, hl0, -, -, -, -⟩ := lineChecked _ _ _ _ _ _ _ _ _ _ hln
  obtain ⟨hhi, hne, hspan, hstop, hstart, hch, hbound, hwts⟩ := stretchBound _ _ _ _ _ _ _ _ _ _ hln i hi f hf x hx
  rw [hsmB] at hbound
  set gs := (rec ω).groups node with hgs
  set m := x.stretch with hm
  set per := node.per with hper
  obtain ⟨hmono, -, hfP, hfg, hrel, hre, hpass⟩ := hp ω ha
  have hG : gs.size = node.windows := (ofSummary_ok hs).1.trans hsmW
  have hC : (recordPace node rec).credited = node.windows - 1 := rfl
  have hlo : ∀ g ≤ per, m * per + g ≤ node.windows - 1 := fun g hg => by
    have : m * per + per = (m + 1) * per := by rw [Nat.succ_mul]
    omega
  let K : Pace Ω := stretchPace (fun ω => (rec ω).groups node) per m
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

/-- Where the record's windows' receipts are the draw's sizes, the trace's wrong receipts below the session's pooled
bound at the line's `δ` are below the W every line of the record reads (`LineCoversSession`): the line reads each
window's draws off the record and refuses any but `min(s, U_t)`, the draw's at the session's `samples`. -/
theorem wrong_le_line {node : Node} {n m κ : ℕ} {X : Type} {D : Session n m (fun _ => node.params.samples) κ X}
    {T : Trace (Coins n κ X)} {rec : Coins n κ X → Record} (hU : SizesOfRecord D T node rec) {ω : Coins n κ X}
    (ha : T.accepts ω) (hw : (T.wrong ω : ℝ) ≤ D.pooled node.delta ω) :
    ∀ l, lineOf node (rec ω) = .ok l → (T.wrong ω : ℝ) ≤ l.wrong := fun l hline => by
  obtain ⟨sm, gs, ws, hsm, hs, -, hl⟩ := lineOf_ok hline
  have hR := hU ω ha
  simp only [Record.receipts, hsm] at hR
  have e1 : sm.windows.map (fun w => (receipts w, min node.params.samples (receipts w))) =
      (sm.windows.map receipts).map (fun u => (u, min node.params.samples u)) :=
    (Array.map_map (f := receipts) (g := fun u => (u, min node.params.samples u)) (as := sm.windows)).symm
  have e2 : (Array.ofFn fun t => (D.size t ω : ℕ)).map (fun u => (u, min node.params.samples u)) =
      Array.ofFn fun t => ((D.size t ω : ℕ), D.draws t ω) :=
    Array.map_ofFn (f := fun t => (D.size t ω : ℕ)) (g := fun u => (u, min node.params.samples u))
  rw [(ofSummary_ok hs).2, e1, hR, e2] at hl
  have hc := Properties.RemoteAudit.lineCoversSession n m _ κ X D ω node.delta _ _ _ _ _ _ _ l hl
  rw [Rat.cast_nnratCast] at hc
  exact hw.trans hc

/-- The node's `δ` is nonnegative, in the reals. -/
theorem delta_nonneg (node : Node) : (0 : ℝ) ≤ (node.delta : ℝ) := NNRat.cast_nonneg _

/-- **The memory line of record, link-charged** (`LineHeld`), proved. -/
theorem lineHeld : LineHeld := by
  intro node n m κ X _ prf rest D T rec δs O hl hsq hd hS hr hp hcov hk hO hm hL hR hC hN hU
  have hH := holdsAnd (sketchBound_of_noOutside hk hO)
    (Draw.pooledDraw prf rest D (ofRecord T node rec) (delta_nonneg node) hm hL hR hC)
  refine le_trans (Pouw.pr_mono fun ω hω => ⟨hω.1, fun h => hω.2 ?_⟩) hH
  have ha : (ofRecord T node rec).accepts ω := hω.1
  have hc := charge_units node
  refine heldAt_of 1 (by omega) ?_ hl ?_ ?_ hsq hd hS hr hp hcov hN ha hω.1 h.1 (wrong_le_line hU ha h.2)
  · intro h0
    simp only [Node.link, ns]
    push_cast
    positivity
  · simpa using hc.2.2.1
  · simpa using hc.2.2.2

/-- **The memory line of record, time-held, under Identity** (`LineHeldIdentity`), proved. -/
theorem lineHeldIdentity : LineHeldIdentity := by
  intro node n m κ X _ prf rest D T rec δs O hI hsq hd hS hr hp hcov hk hO hm hL hR hC hN hU
  have hH := holdsAnd (sketchBound_of_noOutside hk hO)
    (Draw.pooledDraw prf rest D (ofRecord T node rec) (delta_nonneg node) hm hL hR hC)
  refine le_trans (Pouw.pr_mono fun ω hω => ⟨hω.1, fun h => hω.2 ?_⟩) hH
  have ha : (ofRecord T node rec).accepts ω := hω.1
  have hc := charge_units node
  exact heldAt_of 0 (by omega) (fun _ => le_rfl) (links_of_identity hI) (by simpa using hc.1) (by simpa using hc.2.1) hsq hd
    hS hr hp hcov hN ha hω.1 h.1 (wrong_le_line hU ha h.2)

end Properties.RemoteAudit.Line.SecurityProofs
