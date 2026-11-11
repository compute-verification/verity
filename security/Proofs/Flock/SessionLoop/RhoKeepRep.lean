module

public import Proofs.Flock.SessionLoop.RhoUnrefused

/-!
# A rep's frame inside level 0, at its coins

Past the opening's binding a rep's frame is in level 0 after each body (`RhoKeep.sched_l0Run`), and the coins it takes
there keep it so (`Masker.answer`): before level 0's last masked value it only marks the round answered, and at it it
also takes `ρ₀`'s words, which neither moves the position nor leaves the phase.

* **`RhoKeep.answer_l0`**: coins the frame takes at level 0, outside the inner proof, keep its phase and position.
* **`RhoKeep.repState_l0`**: the loop's state at a rep's round from level 0's first body to `ρ₀`'s, where it takes the
  round's coins, is in level 0 at the next body's first value.
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Claims.ZKServer.RhoKeep

open _root_.Flock (V F128)
open _root_.Flock.Firewall.Frame (Cfg Masker MaskSlot Draws)
open Loop

/-- Coins the frame takes at level 0, outside the inner proof, keep its phase and position and leave it owing none. -/
theorem answer_l0 {cfg : Cfg} {s m : Masker} {cs : Array F128} (hi : s.inner = none) (hl : s.l0 = true)
    (h : s.answer cfg cs = .ok m) :
    m.on = s.on ∧ m.l0 = true ∧ m.inner = none ∧ m.owed = none ∧ m.pos = s.pos := by
  unfold Masker.answer at h
  cases ho : s.owed with
  | none => rw [ho] at h; cases h
  | some n =>
    by_cases hc : (cs.size != n) = true
    · simp only [ho, hc, ↓reduceIte] at h
      cases h
    simp only [ho, hc, Bool.false_eq_true, ↓reduceIte] at h
    simp only [hi, hl, ↓reduceIte] at h
    by_cases hp : (s.pos != cfg.order.size) = true
    · simp only [hp, ↓reduceIte, pure, Except.pure, Except.ok.injEq] at h
      subst h
      exact ⟨rfl, rfl, rfl, rfl, rfl⟩
    · simp only [hp, Bool.false_eq_true, ↓reduceIte] at h
      obtain ⟨r, -, h⟩ := bindV_ok h
      simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      exact ⟨rfl, rfl, rfl, rfl, rfl⟩

/-- **The loop's state at a rep's round inside level 0**: where the loop's state at a rep's round from level 0's first
body to `ρ₀`'s takes the round's coins, the frame is in level 0's masked phase, outside the inner proof, owing nothing,
at the next body's first value. The state is the frame's after body `i - 1` (`sched_l0Run`), whose coins keep it
(`answer_l0`). -/
theorem repState_l0 {c c'' : Cfg} (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size)
    {t : _root_.Flock.Firewall.Finish.Tab} {rep : ℕ} {dom : ByteArray} {xs : Array (Array F128)}
    {hi : Array ByteArray} {cr0 cr : Array (Array F128)} {ns : ℕ → ℕ} {alg : Array (ByteArray × ℕ)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hm14 : 14 ≤ t.st.m) (hdg : t.st.digest.size < 2 ^ 64)
    (hcap : (_root_.Flock.Firewall.Finish.flat t.cm0.cap).size < 2 ^ 64) (hdom : dom.size < 2 ^ 64)
    (hmk : t.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512) (hx : xs.size + 2 < 2 ^ 64) {tp : ℕ}
    (hz : _root_.Flock.Zk.schedule t.st.m = .ok (t.sch, tp)) {lay : _root_.Flock.Zk.PadLayout} {d : Draws}
    {mk : Option MaskSlot} (hlay : lay = _root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0)
    (hcd : c = Cfg.ofDraws lay d mk) (hord : c''.order.size = c.order.size) {i : ℕ} (hi1 : 1 ≤ i)
    (hik : i ≤ t.sch.k0 + 3) (hcr : cr.size = alg.size + (xs.size + 2) + 1 + i) {cs : Array F128} {m : Masker}
    (ha : (repState c t rep dom xs hi cr0 cr ns (alg.size + (xs.size + 2) + 1 + i)).answer c'' cs = .ok m) :
    m.on = true ∧ m.l0 = true ∧ m.inner = none ∧ m.owed = none ∧ m.pos = c.nPiop + l0Off i := by
  obtain ⟨-, sched, -, hs⟩ := repState_ok ha
  have hpop : cr.pop.size = alg.size + (xs.size + 2) + 1 + (i - 1) := by
    rw [Array.size_pop, hcr]
    omega
  obtain ⟨hsz, hL⟩ := sched_l0Run hc hal hs hm14 hdg hcap hdom hmk hx hz hlay hcd (j := i - 1) (by omega) hpop
    (if rep = 0 then #[] else (cr0.extract 0 alg.size).foldl (· ++ ·) #[])
  rw [show alg.size + (xs.size + 2) + 1 + i = alg.size + (xs.size + 2) + 1 + (i - 1) + 1 by omega,
    repState_at c t rep dom xs hi cr0 cr ns _ hal hs (by omega)] at ha
  dsimp only at ha
  cases h2 : (_root_.Flock.Firewall.Frame.run c {} (schedReqs
      (if rep = 0 then #[] else (cr0.extract 0 alg.size).foldl (· ++ ·) #[]) sched cr
        (alg.size + (xs.size + 2) + 1 + (i - 1)))).2
  · rw [h2] at ha
    simp only [Bool.false_eq_true, ↓reduceIte] at ha
    rcases hL (i - 1) le_rfl (by omega) h2 with ho | ⟨-, -, hon, hl0, -, hin, hpos, -⟩
    · exact absurd ha (answer_owed_none cs ho)
    · obtain ⟨e1, e2, e3, e4, e5⟩ := answer_l0 hin hl0 ha
      refine ⟨e1.trans hon, e2, e3, e4, e5.trans ?_⟩
      rw [hpos, show i - 1 + 1 = i by omega]
  · rw [h2] at ha
    simp only [↓reduceIte] at ha
    exact absurd ha (answer_owed_none cs rfl)

end Claims.ZKServer.RhoKeep
