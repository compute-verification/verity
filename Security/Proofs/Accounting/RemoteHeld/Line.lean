import Proofs.Accounting.RemoteHeld.Window
import Proofs.Accounting.RemoteHeld.Draw
import Proofs.Accounting.RemoteHeld.Outside
import Proofs.Properties.RemoteAudit.Line
import Proofs.Properties.RemoteAudit.Pooled
import Proofs.Properties.RemoteAudit.Session
import Proofs.Properties.RemoteAudit.Shared

/-!
# The memory line of record, proved (`Properties.RemoteAudit.Line`: `LineHeld`), and memory held from it

At each outcome where the Lean verifier accepts the record's session (so the trace of record accepts, `ofRecord`) and
the certifiers observed nothing but the verifier's messages entering the node, the receipts' sketch holds and the
trace's wrong receipts are at most the session's pooled bound at the line's `δ` (`holdsAnd` of the sketch bound, from
`SketchBoundAt`, `Covered` and `IngressCounted` by `sketchBound_of_covered`, and `Draw.pooledDraw` at the records read
through the coin commitment, `recordCommitted`, so at most the W the line reads: `LineCoversSession` at the record's
draws, whose receipts on an accepted record are `recordDraw`'s sizes, `receipts_of_line`, and the line's check of their
draws, `wrong_le_line`), and the line read off the record credits no more bits read than its groups' passes read
(`readAt_of`): each pass reads every registered block but its wrong receipts (`read_card`), and the groups are one for
each of the session's windows with the node's registered blocks (`record_summary`: `Shared.run` at any windows a draw
keeps the parameters and registration, `frame_run`, and an accepted state has its parameters' windows,
`windows_of_accepted`), their passes among the credited ones.

Memory held follows from bits read under the device model (`heldAt_of_read`, over the outcomes `pr_held_le_read`): the
blocks the passes read are the correct receipts of the trace of those passes alone, whose wrong receipts `ReadAt` holds
to the line's W, and on it every stretch of the line credits what the trace held (`heldAt_of`):

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

/-- What `ofSummaryAt` reads of a record it accepts: one group a window, and each draw's receipts and draws
(`drawReceipts`, a window's at one window a coin), its draws `min(s, U_c)`, and above one window a coin each draw's
receipts below `RECEIPT_LIMIT`. -/
theorem ofSummaryAt_ok {g : ℕ} {sm : RemoteAudit.Session.Summary} {samples : ℕ} {seen : Array Seen}
    {gs : Array Group} {ws : Array (ℕ × ℕ)} (h : ofSummaryAt g sm samples seen = .ok (gs, ws)) :
    gs.size = sm.windows.size ∧ ws = (drawReceipts g sm.windows).map (fun u => (u, min samples u)) ∧
      (1 < g → ∀ u ∈ drawReceipts g sm.windows, u < RemoteAudit.Session.RECEIPT_LIMIT) := by
  unfold ofSummaryAt at h
  by_cases hg : g ≤ 1
  · rw [ite_eq_left hg] at h
    obtain ⟨h1, h2⟩ := ofSummary_ok h
    refine ⟨h1, ?_, fun _ => absurd hg (by omega)⟩
    rw [h2, drawReceipts, ite_eq_left hg, Array.map_map]
    rfl
  · rw [ite_eq_right hg] at h
    split at h
    · cases h
    · try dsimp only at h
      split at h
      · cases h
      · rename_i hf
        split at h
        · cases h
        · rename_i hl
          cases h
          refine ⟨by simp, ?_, fun _ u hu => ?_⟩
          · rw [drawReceipts, ite_eq_right hg, Array.map_map]
            conv => lhs; rw [← Array.map_id (drawsOf g sm.windows)]
            refine Array.map_congr_left fun p hp => ?_
            obtain ⟨u, d⟩ := p
            have hd := Array.findIdx?_eq_none_iff.1 hf (u, d) hp
            simp only [bne_eq_false_iff_eq] at hd
            simp [hd]
          · rw [drawReceipts, ite_eq_right hg] at hu
            obtain ⟨⟨u', d⟩, hp, rfl⟩ := Array.mem_map.1 hu
            have hd := Array.findIdx?_eq_none_iff.1 hl (u', d) hp
            simpa using hd

/-- A line `lineOfRecordAt` computes is `line` at the record's groups, draws and registered blocks. -/
theorem lineOfRecordAt_ok {g : ℕ} {sm : RemoteAudit.Session.Summary} {samples : ℕ} {seen : Array Seen}
    {per blockBytes chipBytes nodeBytes : ℕ} {δ : ℚ} {rates : Rates} {l : Line}
    (h : lineOfRecordAt g sm samples seen per blockBytes chipBytes nodeBytes δ rates = .ok l) :
    ∃ gs ws, ofSummaryAt g sm samples seen = .ok (gs, ws) ∧
      line gs per sm.blocks blockBytes chipBytes nodeBytes ws δ rates = .ok l := by
  unfold lineOfRecordAt at h
  split at h
  · rename_i gs ws hs
    exact ⟨gs, ws, hs, h⟩
  · cases h

/-- A line the record gives is `line` at the groups and draws `ofSummaryAt` reads off the Lean verifier's record of the
session, its registered blocks the record's, and those groups are the record's (`Record.groups`). -/
theorem lineOf_ok {node : Node} {r : Record} {l : Line} (h : lineOf node r = .ok l) :
    ∃ sm gs ws, r.summary node = some sm ∧
      ofSummaryAt node.groupsPerCoin sm node.params.samples r.seen = .ok (gs, ws) ∧ r.groups node = gs ∧
      line gs node.per sm.blocks node.blockBytes node.chipBytes node.nodeBytes ws node.delta node.rates = .ok l := by
  unfold lineOf at h
  split at h
  · rename_i sm hsm
    obtain ⟨gs, ws, hs, hl⟩ := lineOfRecordAt_ok h
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

/-- A session `Shared.run g` runs from `State.init` keeps the parameters and registration it started with:
`Session.run`'s at g ≤ 1, and above it every `Shared.step` keeps them (`Properties.RemoteAudit.Shared.sstep`). -/
theorem frame_run {g : ℕ} {params : RemoteAudit.Session.Params} {reg : RemoteAudit.Registration}
    {coins appendCoins : Array ByteArray} {notBeforeNs : ℕ} {events : Array RemoteAudit.Session.Event}
    {rs : Array RemoteAudit.Session.Result} {s : RemoteAudit.Session.State}
    (h : _root_.RemoteAudit.Shared.run g (RemoteAudit.Session.State.init params reg coins appendCoins notBeforeNs)
      events = .ok (rs, s)) :
    s.params = params ∧ s.reg = reg := by
  rcases Nat.lt_or_ge g 2 with hg | hg
  · rw [Properties.RemoteAudit.Shared.run_le_one (g := g) (by omega)] at h
    obtain ⟨-, hp, hr, -⟩ := Properties.RemoteAudit.inv_run h
    exact ⟨hp, hr⟩
  · have := Properties.RemoteAudit.Shared.srun_inv g
      (fun s => Properties.RemoteAudit.Inv s ∧ s.params = params ∧ s.reg = reg)
      (fun _ _ _ _ ⟨hs, hp, hr⟩ he =>
        have ⟨hs', ⟨fp, fr, _⟩, _⟩ := Properties.RemoteAudit.Shared.sstep hg hs he
        ⟨hs', fp.trans hp, fr.trans hr⟩)
      _ _ _ _ ⟨Properties.RemoteAudit.inv_init .., rfl, rfl⟩ h
    exact this.2

/-- An accepted state has its parameters' windows (`State.outcome`). -/
theorem windows_of_accepted {s : RemoteAudit.Session.State} (hout : s.outcome = some "accepted") :
    s.windows.size = s.params.windows := by
  unfold RemoteAudit.Session.State.outcome at hout
  split at hout
  · rename_i f _
    have := Option.some.inj hout
    cases hf : f.failure <;> simp [hf, _root_.RemoteAudit.Failure.name] at this
  · split at hout
    · rename_i hc
      simp only [Bool.and_eq_true, beq_iff_eq, Array.all_eq_true] at hc
      exact hc.1
    · cases hout

/-- An accepted run, at any windows a draw, ends with the node's `windows` windows. -/
theorem windows_of_verified {node : Node} {r : Record} {rs : Array RemoteAudit.Session.Result}
    {s : RemoteAudit.Session.State} (hrun : r.run node = .ok (rs, s)) (hout : s.outcome = some "accepted") :
    s.windows.size = node.windows ∧ s.reg = node.reg := by
  obtain ⟨hp, hr⟩ := frame_run (show _root_.RemoteAudit.Shared.run node.groupsPerCoin
    (RemoteAudit.Session.State.init node.params node.reg r.coins r.appendCoins r.notBeforeNs) r.events = .ok (rs, s)
    from hrun)
  exact ⟨(windows_of_accepted hout).trans (congrArg RemoteAudit.Session.Params.windows hp), hr⟩

/-- **One group a window**: the Lean verifier's record of a session it accepts, at any windows a draw, has the node's
`windows` windows, and the node's registered blocks. -/
theorem record_summary {node : Node} {r : Record} {sm : RemoteAudit.Session.Summary} (hv : Verified node r)
    (hsm : r.summary node = some sm) : sm.windows.size = node.windows ∧ sm.blocks = node.blocks := by
  obtain ⟨rs, s, hrun, hout⟩ := hv
  have hs : s.summary = some sm := by
    unfold Record.summary at hsm
    rw [hrun] at hsm
    exact hsm
  obtain ⟨h1, h2⟩ := summary_of hs
  obtain ⟨hW, hr⟩ := windows_of_verified hrun hout
  exact ⟨h1.trans hW, h2.trans (congrArg RemoteAudit.Registration.blocks hr)⟩

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
  have hG : gs.size = node.windows := (ofSummaryAt_ok hs).1.trans hsmW
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

/-- A run at `g` windows a draw that runs every event reaches its last state one event at a time. -/
theorem reach_of_fold (g : ℕ) (l : List RemoteAudit.Session.Event) :
    ∀ (acc : Array RemoteAudit.Session.Result × RemoteAudit.Session.State) out,
      List.foldlM (fun (acc : Array RemoteAudit.Session.Result × RemoteAudit.Session.State) e => do
        let (r, s') ← _root_.RemoteAudit.Shared.step g acc.2 e
        pure (acc.1.push r, s')) acc l = .ok out →
      List.foldlM (reachStep g) acc.2 l = .ok out.2 := by
  induction l with
  | nil => intro acc out h; simp only [List.foldlM_nil] at h ⊢; cases h; rfl
  | cons e l ih =>
    intro acc out h
    rw [List.foldlM_cons] at h
    rw [List.foldlM_cons]
    cases he : _root_.RemoteAudit.Shared.step g acc.2 e with
    | error err => rw [he] at h; cases h
    | ok rs' =>
      rw [he] at h
      obtain ⟨r, s''⟩ := rs'
      have hr : reachStep g acc.2 e = .ok s'' := by simp only [reachStep, he]
      rw [hr]
      exact ih (acc.1.push r, s'') out h

/-- Where the record's session runs every event, the state it reaches is its run's last. -/
theorem reached_of_run {node : Node} {r : Record} {rs : Array RemoteAudit.Session.Result}
    {s : RemoteAudit.Session.State} (h : r.run node = .ok (rs, s)) : r.reached node = s := by
  unfold Record.run _root_.RemoteAudit.Shared.run at h
  rw [← Array.foldlM_toList] at h
  have hk : List.foldlM (reachStep node.groupsPerCoin)
      (RemoteAudit.Session.State.init node.params node.reg r.coins r.appendCoins r.notBeforeNs) r.events.toList =
      .ok s := reach_of_fold _ _ _ _ h
  unfold Record.reached
  rw [← Array.foldlM_toList, hk]

/-- Each window's receipts in a state's summary are its window's. -/
theorem summary_receipts {s : RemoteAudit.Session.State} {sm : RemoteAudit.Session.Summary}
    (h : s.summary = some sm) :
    sm.windows.map receipts = (Array.range s.windows.size).map fun t => s.windows[t]!.receipts := by
  unfold RemoteAudit.Session.State.summary at h
  split at h
  · cases h
  · simp only [Option.some.injEq] at h
    subst h
    apply Array.ext
    · simp
    · intro i h1 h2
      simp only [Array.getElem_map, Array.getElem_range]
      rfl

/-- Each window's receipts in a state's summary are its window's, as arrays. -/
theorem summary_windows {s : RemoteAudit.Session.State} {sm : RemoteAudit.Session.Summary}
    (h : s.summary = some sm) : sm.windows.map receipts = s.windows.map RemoteAudit.Session.Win.receipts := by
  rw [summary_receipts h]
  apply Array.ext
  · simp
  · intro i h1 h2
    have hi : i < s.windows.size := by simpa using h1
    simp only [Array.getElem_map, Array.getElem_range]
    rw [getElem!_pos s.windows i hi]

/-- Two arrays whose maps agree sum alike over any of their slices. -/
theorem foldl_extract_congr {α β : Type} (f : α → ℕ) (g : β → ℕ) {as : Array α} {bs : Array β}
    (h : as.map f = bs.map g) (a b : ℕ) :
    (as.extract a b).foldl (fun x w => x + f w) 0 = (bs.extract a b).foldl (fun x w => x + g w) 0 := by
  rw [← Array.foldl_map (f := f) (g := fun x y => x + y), ← Array.foldl_map (f := g) (g := fun x y => x + y),
    Array.map_extract, Array.map_extract, h]

/-- **The record's draws are the draw's**: on an accepted record whose draws the line reads, each draw's receipts in the
Lean verifier's record are those its windows have in the state the record's session reaches, and below
`RECEIPT_LIMIT`, so `recordDraw`'s sizes: at one window a coin each window's is below it (`ReceiptsBelowLimit`), and
above it the line refuses a draw that isn't (`ofSummaryAt_ok`). -/
theorem receipts_of_line {node : Node} {r : Record} {sm : RemoteAudit.Session.Summary} (hv : Verified node r)
    (hsm : r.summary node = some sm)
    (hlim : 1 < node.groupsPerCoin → ∀ u ∈ drawReceipts node.groupsPerCoin sm.windows,
      u < RemoteAudit.Session.RECEIPT_LIMIT) :
    drawReceipts node.groupsPerCoin sm.windows =
      Array.ofFn fun c : Fin node.draws => min (r.drawReceiptsAt node c) receiptMax := by
  obtain ⟨rs, s, hrun, hout⟩ := hv
  have hs : s.summary = some sm := by
    unfold Record.summary at hsm
    rw [hrun] at hsm
    exact hsm
  have hreach := reached_of_run hrun
  have hW := (windows_of_verified hrun hout).1
  have hS := summary_windows hs
  have hsz := (summary_of hs).1
  by_cases hg : node.groupsPerCoin ≤ 1
  · have hrun' : RemoteAudit.Session.run (RemoteAudit.Session.State.init node.params node.reg r.coins r.appendCoins
        r.notBeforeNs) r.events = .ok (rs, s) := (Properties.RemoteAudit.Shared.run_le_one hg _ _).symm.trans hrun
    have hL := Properties.RemoteAudit.receiptsBelowLimit _ _ _ _ _ _ _ _ hrun'
    have hD : node.draws = node.windows := by unfold Node.draws; rw [ite_eq_left hg]
    rw [drawReceipts, ite_eq_left hg, hS]
    apply Array.ext
    · simp [hD, hW]
    · intro i h1 h2
      have hi : i < s.windows.size := by simpa using h1
      have hat : r.drawReceiptsAt node i = s.windows[i].receipts := by
        rw [Record.drawReceiptsAt, ite_eq_left hg, Record.receiptsAt, hreach, Array.getElem?_eq_getElem hi]
        all_goals rfl
      have hle : r.drawReceiptsAt node i ≤ receiptMax := by
        have := hL i hi
        unfold receiptMax
        omega
      simp only [Array.getElem_map, Array.getElem_ofFn]
      rw [← hat]
      exact (min_eq_left hle).symm
  · have hD : node.draws = (node.windows + node.groupsPerCoin - 1) / node.groupsPerCoin := by
      unfold Node.draws; rw [ite_eq_right hg]
    have hlim' := hlim (by omega)
    rw [drawReceipts, ite_eq_right hg] at hlim' ⊢
    apply Array.ext
    · simp [drawsOf, hD, hsz, hW]
    · intro i h1 h2
      have hu := hlim' _ (Array.getElem_mem h1)
      have hat : r.drawReceiptsAt node i = ((drawsOf node.groupsPerCoin sm.windows).map (·.1))[i] := by
        rw [Record.drawReceiptsAt, ite_eq_right hg, hreach]
        simp only [drawsOf, Array.getElem_map, Array.getElem_range]
        exact (foldl_extract_congr receipts RemoteAudit.Session.Win.receipts hS _ _).symm
      have hle : r.drawReceiptsAt node i ≤ receiptMax := by
        unfold receiptMax
        omega
      simp only [Array.getElem_ofFn]
      rw [← hat]
      exact (min_eq_left hle).symm

/-- `m < 2^448`: each draw's sampler word lies in its stream's first SHA-512 block. -/
theorem receiptMax_lt : receiptMax < 2 ^ 448 := by
  unfold receiptMax RemoteAudit.Session.RECEIPT_LIMIT
  exact Nat.lt_of_lt_of_le (Nat.sub_lt (Nat.two_pow_pos 64) Nat.one_pos)
    (Nat.pow_le_pow_right (by norm_num) (by norm_num))

/-- On an accepted record, the trace's wrong receipts below the record's draw's pooled bound at the line's `δ` are
below the W every line of the record reads (`LineCoversSession`): the line reads each draw's receipts and draws off
the record and refuses any draws but `min(s, U_c)`, the draw's at the session's `samples`, and the receipts are the
draw's sizes (`receipts_of_line`). -/
theorem wrong_le_line {node : Node} {κ : ℕ} {X : Type} {T : Trace (DrawCoins node.draws κ X)}
    {rec : DrawCoins node.draws κ X → Record} {correct : Fin node.draws → DrawCoins node.draws κ X → Finset ℕ}
    {ω : DrawCoins node.draws κ X} (hv : Verified node (rec ω))
    (hw : (T.wrong ω : ℝ) ≤ (recordDraw node rec correct).pooled node.delta ω) :
    ∀ l, lineOf node (rec ω) = .ok l → (T.wrong ω : ℝ) ≤ l.wrong := fun l hline => by
  obtain ⟨sm, gs, ws, hsm, hs, -, hl⟩ := lineOf_ok hline
  obtain ⟨-, hws, hlim⟩ := ofSummaryAt_ok hs
  have hR : drawReceipts node.groupsPerCoin sm.windows =
      Array.ofFn fun c => ((recordDraw node rec correct).size c ω : ℕ) :=
    receipts_of_line hv hsm hlim
  have e2 : (Array.ofFn fun t => ((recordDraw node rec correct).size t ω : ℕ)).map
      (fun u => (u, min node.params.samples u)) =
      Array.ofFn fun t => (((recordDraw node rec correct).size t ω : ℕ), (recordDraw node rec correct).draws t ω) :=
    Array.map_ofFn (f := fun t => ((recordDraw node rec correct).size t ω : ℕ))
      (g := fun u => (u, min node.params.samples u))
  rw [hws, hR, e2] at hl
  have hc := Properties.RemoteAudit.lineCoversSession node.draws receiptMax _ κ X (recordDraw node rec correct) ω
    node.delta _ _ _ _ _ _ _ l hl
  rw [Rat.cast_nnratCast] at hc
  exact hw.trans hc

/-- The node's `δ` is nonnegative, in the reals. -/
theorem delta_nonneg (node : Node) : (0 : ℝ) ≤ (node.delta : ℝ) := NNRat.cast_nonneg _

/-- On an accepting outcome of a trace whose passes plan every registered block, a credited pass's blocks read and its
wrong receipts make up the registered blocks, where its correct receipts' blocks were on the node at a tick of it. -/
theorem read_card {Ω : Type} {T : Trace Ω} (hcov : Covers T) {ω : Ω} (ha : T.accepts ω) {p : ℕ} (hp : p < T.passes)
    (hk : ∀ j ∈ T.correct ω p, ∃ t, T.tick ω p ≤ t ∧ t < T.tick ω (p + 1) ∧ j ∈ T.present ω t) :
    (T.read ω p).card + (T.plan ω p \ T.correct ω p).card = T.blocks := by
  have hr : T.read ω p = T.plan ω p ∩ T.correct ω p := by
    unfold Trace.read
    refine Finset.inter_eq_left.2 fun j hj => ?_
    obtain ⟨t, h1, h2, ht⟩ := hk j (Finset.mem_inter.1 hj).2
    exact Finset.mem_biUnion.2 ⟨t, Finset.mem_Ico.2 ⟨h1, h2⟩, ht⟩
  rw [hr, add_comm, Finset.card_sdiff_add_card_inter, hcov ω ha p hp, Finset.card_univ, Fintype.card_fin]

/-- On an accepting outcome where the receipts' sketch holds and the trace's wrong receipts are at most the W the
record's line reads, the line credits no more bits read than its groups' passes read: each of the `P` passes reads
every registered block but its wrong receipts (`read_card`), and the passes are among the credited ones (`Paced`). -/
theorem readAt_of {Ω : Type} {T : Trace Ω} {node : Node} {rec : Ω → Record} (hp : Paced T (recordPace node rec))
    (hcov : Covers T) (hN : Registered T node) {ω : Ω} (ha : T.accepts ω) (hv : Verified node (rec ω))
    (hk : ∀ p < T.passes, ∀ j ∈ T.correct ω p, ∃ t, T.tick ω p ≤ t ∧ t < T.tick ω (p + 1) ∧ j ∈ T.present ω t)
    (hw : ∀ l, lineOf node (rec ω) = .ok l → (T.wrong ω : ℝ) ≤ l.wrong) :
    ReadAt T ω node (rec ω) := by
  intro l hline
  obtain ⟨sm, gs, ws, hsm, hs, hgs, -⟩ := lineOf_ok hline
  have hG : ((rec ω).groups node).size = node.windows := by
    rw [hgs]; exact (ofSummaryAt_ok hs).1.trans (record_summary hv hsm).1
  have hPle : firstPass ((rec ω).groups node) ((rec ω).groups node).size ≤ T.passes := by
    obtain ⟨-, -, hfP, -⟩ := hp ω ha
    refine le_trans (firstPass_mono ?_) hfP
    show ((rec ω).groups node).size ≤ node.windows - 1 + 1
    omega
  set P := firstPass ((rec ω).groups node) ((rec ω).groups node).size with hP
  have h1 : ∑ p ∈ range P, ((T.read ω p).card + (T.plan ω p \ T.correct ω p).card) = P * T.blocks := by
    rw [Finset.sum_const_nat fun p hp' => read_card hcov ha (lt_of_lt_of_le (Finset.mem_range.1 hp') hPle)
      (hk p (lt_of_lt_of_le (Finset.mem_range.1 hp') hPle)), Finset.card_range]
  have h2 : ∑ p ∈ range P, (T.plan ω p \ T.correct ω p).card ≤ T.wrong ω :=
    Finset.sum_le_sum_of_subset (Finset.range_subset_range.2 hPle)
  have hsum : P * T.blocks ≤ T.reads ω P + T.wrong ω := by
    rw [Finset.sum_add_distrib] at h1
    unfold Trace.reads
    rw [← h1]
    exact Nat.add_le_add_left h2 _
  have hNr : (T.blocks : ℝ) = node.blocks := by rw [hN]
  have hsum' : (P : ℝ) * T.blocks ≤ T.reads ω P + T.wrong ω := by exact_mod_cast hsum
  have hwl := hw l hline
  have hb : (0 : ℝ) ≤ 8 * node.blockBytes := by positivity
  refine mul_le_mul_of_nonneg_left ?_ hb
  push_cast
  rw [← hNr]
  linarith

/-- **The trace of record is over the registration's blocks**: `Registered` holds of it, with no premise. -/
theorem registered_ofRecord {Ω : Type} (T : Trace Ω) (node : Node) (rec : Ω → Record) :
    Registered (ofRecord T node rec) node := rfl

/-- **The memory line of record, in bits read** (`LineHeld`), proved. -/
theorem lineHeld : LineHeld := by
  intro node κ X Cm _ prf rest recs correct com T R δs δnet r F O I hp hcov hk hC hI hL hR hCnt
  have hN := registered_ofRecord T node r
  have hH := holdsAnd (sketchBound_of_covered hk hC hI)
    (Draw.pooledDraw prf rest F com (ofRecord T node r) R (delta_nonneg node) receiptMax_lt hL hR hCnt)
  refine le_trans (Pouw.pr_mono fun ω hω => ⟨hω.1, fun h => hω.2 fun hq => ?_⟩) hH
  exact readAt_of hp hcov hN hω.1 hω.1 (h.1 hq) (wrong_le_line (correct := fun t ω => correct (com ω) t ω) hω.1 h.2)

/-- **Memory held from bits read, under the device model**: on an accepting outcome where the record's line credits no
more bits read than its groups' passes read (`ReadAt`), every stretch of form `i` credits what the trace held over its
span, at the links' ceiling `Links` (`Identity` at `0`) and the rest of the device model. The blocks the passes read
are the correct receipts of the trace of those passes alone, every block planned (`T'`), whose sketch holds by
`Trace.read`'s definition and whose wrong receipts are at most the line's W by `ReadAt`, so `heldAt_of` applies to it;
what a stretch credits reads only the trace's memory, which `T'` shares. -/
theorem heldAt_of_read {Ω : Type} [Fintype Ω] {T : Trace Ω} {node : Node} {rec : Ω → Record} (i : ℕ) (hi : i < 2)
    {βl Bl : ℝ} (hβl : 0 ≤ node.rates.linkBytesPerS → 0 ≤ βl) (hl : Links T βl Bl)
    (hcr : (((if i = 0 then (charges node.rates node.blockBytes).1 else (charges node.rates node.blockBytes).2).rate
      : ℚ) : ℝ) = node.squarings / node.encode + node.disk + βl)
    (hcb : (((if i = 0 then (charges node.rates node.blockBytes).1 else (charges node.rates node.blockBytes).2).burst
      : ℚ) : ℝ) = node.rebuildBurst + node.diskBurst + Bl)
    (hsq : SquaringCeiling T node.squarings node.encode node.rebuildBurst) (hd : Disks T node.disk node.diskBurst)
    (hS : ChipCeiling T node.chipBlocks) (hr : Routed T) (hp : Paced T (recordPace node rec))
    (hN : Registered T node) {ω : Ω} (ha : T.accepts ω) (hv : Verified node (rec ω))
    (hread : ReadAt T ω node (rec ω)) :
    HeldAt T ω node (rec ω) i := by
  intro l hline f hf x hx
  obtain ⟨sm, gs, ws, hsm, hs, hgs, hln⟩ := lineOf_ok hline
  obtain ⟨hb, -, -, -, -, -, -, -, -⟩ := lineChecked _ _ _ _ _ _ _ _ _ _ hln
  have hhi := (stretchBound _ _ _ _ _ _ _ _ _ _ hln i hi f hf x hx).1
  have hG : ((rec ω).groups node).size = node.windows := by
    rw [hgs]; exact (ofSummaryAt_ok hs).1.trans (record_summary hv hsm).1
  have h0 : 0 < gs.size := lt_of_le_of_lt (Nat.zero_le _) hhi
  have hW : 1 ≤ node.windows := by rw [← hG, hgs]; omega
  let P := firstPass ((rec ω).groups node) ((rec ω).groups node).size
  let T' : Trace Ω :=
    { T with accepts := fun ω' => ω' = ω, passes := P, plan := fun _ _ => Finset.univ,
             correct := fun ω' p => T.read ω' p }
  have hp' : Paced T' (recordPace node rec) := fun ω' (e : ω' = ω) => by
    rw [e]
    obtain ⟨hmono, hf0, -, hfg, hrel, hre, hpass⟩ := hp ω ha
    refine ⟨hmono, hf0, ?_, hfg, hrel, hre, hpass⟩
    show firstPass ((rec ω).groups node) (node.windows - 1 + 1) ≤
      firstPass ((rec ω).groups node) ((rec ω).groups node).size
    exact le_of_eq (by rw [hG, Nat.sub_add_cancel hW])
  have hcov' : Covers T' := fun _ _ _ _ => rfl
  have hk' : ∀ p < T'.passes, ∀ j ∈ T'.correct ω p,
      ∃ t, T'.tick ω p ≤ t ∧ t < T'.tick ω (p + 1) ∧ j ∈ T'.present ω t := fun p _ j hj => by
    have hj' : j ∈ T.read ω p := hj
    unfold Trace.read at hj'
    obtain ⟨t, ht, htj⟩ := Finset.mem_biUnion.1 (Finset.mem_inter.1 hj').2
    exact ⟨t, (Finset.mem_Ico.1 ht).1, (Finset.mem_Ico.1 ht).2, htj⟩
  have hNr : (T.blocks : ℝ) = node.blocks := by rw [hN]
  have hw' : ∀ l, lineOf node (rec ω) = .ok l → (T'.wrong ω : ℝ) ≤ l.wrong := fun l' hline' => by
    have h8 : (0 : ℝ) < 8 * node.blockBytes := by
      have : (1 : ℝ) ≤ node.blockBytes := by exact_mod_cast hb
      linarith
    have hR : ((P * node.blocks : ℕ) : ℝ) - l'.wrong ≤ T.reads ω P :=
      le_of_mul_le_mul_left (hread l' hline') h8
    have hp1 : ∀ p ∈ range P, (univ \ T.read ω p).card + (T.read ω p).card = T.blocks := fun p _ => by
      have := Finset.card_sdiff_add_card_inter (univ : Finset (Fin T.blocks)) (T.read ω p)
      rwa [Finset.univ_inter, Finset.card_univ, Fintype.card_fin] at this
    have hsum : T'.wrong ω + T.reads ω P = P * T.blocks := by
      show ∑ p ∈ range P, (univ \ T.read ω p).card + ∑ p ∈ range P, (T.read ω p).card = P * T.blocks
      rw [← Finset.sum_add_distrib, Finset.sum_const_nat hp1, Finset.card_range]
    have hsumR : (T'.wrong ω : ℝ) + T.reads ω P = (P : ℝ) * node.blocks := by
      rw [← hNr]; exact_mod_cast hsum
    push_cast at hR
    linarith
  exact heldAt_of (T := T') i hi hβl hl hcr hcb hsq hd hS hr hp' hcov' hN rfl hv hk' hw' l hline f hf x hx

/-- **Memory held from bits read, over the outcomes**: on the trace of record under the device model, the outcomes
where the Lean verifier accepts, `Q` holds and a stretch of form `i` credits more than the trace held are among those
where it accepts, `Q` holds and the line credits more bits read than the passes read (`heldAt_of_read`), so
`LineHeld`'s bound, at `Q` the certifiers' observation, bounds them. -/
theorem pr_held_le_read {Ω : Type} [Fintype Ω] {T : Trace Ω} {node : Node} {rec : Ω → Record} (i : ℕ) (hi : i < 2)
    {βl Bl : ℝ} (hβl : 0 ≤ node.rates.linkBytesPerS → 0 ≤ βl) (hl : Links (ofRecord T node rec) βl Bl)
    (hcr : (((if i = 0 then (charges node.rates node.blockBytes).1 else (charges node.rates node.blockBytes).2).rate
      : ℚ) : ℝ) = node.squarings / node.encode + node.disk + βl)
    (hcb : (((if i = 0 then (charges node.rates node.blockBytes).1 else (charges node.rates node.blockBytes).2).burst
      : ℚ) : ℝ) = node.rebuildBurst + node.diskBurst + Bl)
    (hsq : SquaringCeiling (ofRecord T node rec) node.squarings node.encode node.rebuildBurst)
    (hd : Disks (ofRecord T node rec) node.disk node.diskBurst) (hS : ChipCeiling (ofRecord T node rec) node.chipBlocks)
    (hr : Routed (ofRecord T node rec)) (hp : Paced (ofRecord T node rec) (recordPace node rec)) (Q : Ω → Prop) :
    Pouw.pr (fun ω => Verified node (rec ω) ∧ ¬ (Q ω → HeldAt (ofRecord T node rec) ω node (rec ω) i)) ≤
      Pouw.pr (fun ω => Verified node (rec ω) ∧ ¬ (Q ω → ReadAt (ofRecord T node rec) ω node (rec ω))) :=
  Pouw.pr_mono fun _ hω => ⟨hω.1, fun h => hω.2 fun hq =>
    heldAt_of_read i hi hβl hl hcr hcb hsq hd hS hr hp (registered_ofRecord T node rec) hω.1 hω.1 (h hq)⟩

end Properties.RemoteAudit.Line.SecurityProofs
