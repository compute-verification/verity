import Proofs.Flock.Claims.ZK.FramePlan

/-!
# The rep's own table against the placeholder's

The session loop keeps a rep's table as `Finish.tabOfLevel0` of a level 0 that `Level0.ofDraws` drew (`Finish`'s
`level0` request, `level0Of`), and the placeholder is the same at zero draws. So the rep's own table is of the
placeholder's shape.

* `RealTabs`: the tables `real` answers are table `k / 2`'s `tabOfLevel0` at a level 0 `ofDraws` drew at its `m`.
* `sched_bounds`: every `--zk` schedule's pad and level-0 tree positions fit a `u64`'s count, by `decide +kernel`.
* `ofDraws_gsize`: `ofDraws` takes a packed witness of exactly level 0's size.
* `real_tabAlike`: at `RealTabs`, the rep's own table is `TabAlike` the placeholder's (`Sched.tabOfLevel0_alike`).
-/

namespace Claims.ZKServer.Count

open _root_.Flock (V F128)

/-- A `--zk` schedule's pad and tree sizes within a `u64`'s count: what `tabOfLevel0_alike` and `schedSoFar_alike`
read. -/
def schedBnd (m : ℕ) : Bool :=
  match _root_.Flock.Zk.schedule m with
  | .ok (sch, tp) => decide (2 * tp < 2 ^ 64) && sch.levels.all (fun lv => decide (64 * 2 ^ lv.d < 2 ^ 64)) &&
      decide (64 * 2 ^ (default : _root_.Flock.Level).d < 2 ^ 64)
  | .error _ => true

/-- Every `--zk` schedule, `m < 36`, by the kernel's evaluation. -/
theorem schedBnd_lt : ∀ m, m < 36 → schedBnd m = true := by decide +kernel

theorem sched_bounds {m tp : ℕ} {sch : _root_.Flock.Schedule} (h : _root_.Flock.Zk.schedule m = .ok (sch, tp)) :
    2 * tp < 2 ^ 64 ∧ ∀ i, 64 * 2 ^ (sch.levels.getD i default).d < 2 ^ 64 := by
  have hok := schedBnd_lt m (sched_facts h).1
  unfold schedBnd at hok
  rw [h] at hok
  simp only [Bool.and_eq_true, decide_eq_true_eq, Array.all_eq_true] at hok
  obtain ⟨⟨hp, hl⟩, hd⟩ := hok
  refine ⟨hp, fun i => ?_⟩
  rw [Array.getD_eq_getD_getElem?]
  rcases hi : sch.levels[i]? with _ | lv
  · exact hd
  · obtain ⟨hi', rfl⟩ := Array.getElem?_eq_some_iff.1 hi
    exact hl i hi'

theorem ofDraws_gsize {m : ℕ} {g : Array F128} {d : ByteArray} {l : _root_.Flock.Firewall.Level0.Level0}
    (h : _root_.Flock.Firewall.Level0.ofDraws m g d = .ok l) :
    ∃ s tp, _root_.Flock.Zk.schedule m = .ok (s, tp) ∧
      g.size = 2 ^ ((s.levels.getD 0 default).lanes + (s.levels.getD 0 default).cols) := by
  unfold _root_.Flock.Firewall.Level0.ofDraws at h
  cases hz : _root_.Flock.Zk.schedule m with
  | error e => rw [hz] at h; cases h
  | ok p =>
    obtain ⟨s, tPad⟩ := p
    rw [hz] at h
    refine ⟨s, tPad, rfl, ?_⟩
    by_contra hg
    rw [Array.getD_eq_getD_getElem?] at hg
    simp [bind, Except.bind, hg] at h

/-- **The tables `real` answers are the session loop's own**: table `k / 2`'s `tabOfLevel0` at a level 0 that
`Level0.ofDraws` drew at its `m`, as `Finish`'s `level0` request reads it (`level0Of`). -/
def RealTabs (sts : Array _root_.Flock.Stmt) (real : ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray)) :
    Prop :=
  ∀ k tr hr, real k = .ok (tr, hr) → ∃ g d l, _root_.Flock.Firewall.Level0.ofDraws sts[k / 2]!.m g d = .ok l ∧
    _root_.Flock.Firewall.Finish.tabOfLevel0 (_root_.Flock.Setup.ofCircuit sts[k / 2]!) l = .ok tr

/-- **The rep's own table is of the placeholder's shape**, where `real` answers the loop's own tables. -/
theorem real_tabAlike {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt} {k : ℕ} {link : Array F128}
    {c' : _root_.Flock.Firewall.Frame.Cfg} {t : _root_.Flock.Firewall.Finish.Tab} {dom : ByteArray}
    {xs : Array (Array F128)} {hi : Array ByteArray}
    {real : ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray)} (hR : RealTabs sts real)
    (hph : Loop.placeholder spec sts k link = .ok (c', t, dom, xs, hi)) {tr : _root_.Flock.Firewall.Finish.Tab}
    {hr : Array ByteArray} (hre : real k = .ok (tr, hr)) : Sched.TabAlike tr t := by
  have bok : ∀ {α β : Type} {x : V α} {f : α → V β} {b : β}, x >>= f = .ok b → ∃ a, x = .ok a ∧ f a = .ok b := by
    intro α β x f b hb
    cases x with
    | error e => cases hb
    | ok a => exact ⟨a, rfl, hb⟩
  obtain ⟨g, d, l, hl, htr⟩ := hR k tr hr hre
  unfold Loop.placeholder at hph
  obtain ⟨⟨lay, mask⟩, -, hph⟩ := bok hph
  obtain ⟨⟨sch, tPad⟩, -, hph⟩ := bok hph
  obtain ⟨l0, hl0, hph⟩ := bok hph
  obtain ⟨t0, ht0, hph⟩ := bok hph
  cases hph
  obtain ⟨s, tp, hs, hg⟩ := ofDraws_gsize hl
  obtain ⟨s', tp', hs', hg'⟩ := ofDraws_gsize hl0
  rw [hs] at hs'
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj hs')
  obtain ⟨-, -, ⟨s0, hs0, hlv⟩, -⟩ := _root_.Flock.Firewall.Level0.ofDraws_ok hl
  rw [hs] at hs0
  obtain ⟨rfl, -⟩ := Prod.mk.inj (Except.ok.inj hs0)
  have hd : 64 * 2 ^ l.lv.d < 2 ^ 64 := by
    rw [hlv]
    exact (sched_bounds hs).2 0
  have hA := Sched.tabOfLevel0_alike (_root_.Flock.Setup.ofCircuit sts[k / 2]!) hl hl0 (by rw [hg, hg']) hd
  rw [htr, ht0] at hA
  exact hA

theorem tabOfLevel0_sched {st : _root_.Flock.Setup} {l : _root_.Flock.Firewall.Level0.Level0}
    {t : _root_.Flock.Firewall.Finish.Tab} (h : _root_.Flock.Firewall.Finish.tabOfLevel0 st l = .ok t) :
    t.st = st ∧ _root_.Flock.Zk.schedule st.m = .ok (t.sch, t.tPad) := by
  unfold _root_.Flock.Firewall.Finish.tabOfLevel0 at h
  simp only [bind, Except.bind] at h
  split at h
  · cases h
  split at h
  · cases h
  rcases hs : _root_.Flock.Zk.schedule st.m with e | ⟨sch, tPad⟩
  · rw [hs] at h; cases h
  · rw [hs] at h
    simp only at h
    split at h
    · cases h
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨rfl, rfl⟩

/-- **A table of the placeholder's shape is within the bounds `schedSoFar_alike` reads**: its statement's, digest and
`m`, and its `--zk` schedule's (`sched_facts`, `sched_bounds`). -/
theorem real_bounds {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt} {k : ℕ} {link : Array F128}
    {c' : _root_.Flock.Firewall.Frame.Cfg} {t : _root_.Flock.Firewall.Finish.Tab} {dom : ByteArray}
    {xs : Array (Array F128)} {hi : Array ByteArray} {tr : _root_.Flock.Firewall.Finish.Tab}
    (hph : Loop.placeholder spec sts k link = .ok (c', t, dom, xs, hi)) (hT : Sched.TabAlike tr t) :
    tr.st.digest.size < 2 ^ 64 ∧ tr.st.m < 2 ^ 64 ∧ tr.sch.n + 1 < 2 ^ 64 ∧
      (∀ i, (tr.sch.levels.getD i default).q < 2 ^ 64) ∧ 2 * tr.tPad < 2 ^ 64 ∧
      ∀ i, 64 * 2 ^ (tr.sch.levels.getD i default).d < 2 ^ 64 := by
  have bok : ∀ {α β : Type} {x : V α} {f : α → V β} {b : β}, x >>= f = .ok b → ∃ a, x = .ok a ∧ f a = .ok b := by
    intro α β x f b hb
    cases x with
    | error e => cases hb
    | ok a => exact ⟨a, rfl, hb⟩
  unfold Loop.placeholder at hph
  obtain ⟨⟨lay, mask⟩, -, hph⟩ := bok hph
  obtain ⟨⟨sch, tPad⟩, -, hph⟩ := bok hph
  obtain ⟨l0, -, hph⟩ := bok hph
  obtain ⟨t0, ht0, hph⟩ := bok hph
  cases hph
  obtain ⟨hts, hsch⟩ := tabOfLevel0_sched ht0
  obtain ⟨hm, -, hn, hq⟩ := sched_facts hsch
  obtain ⟨hp, hd⟩ := sched_bounds hsch
  rw [hT.1, hT.2.1, hT.2.2.1]
  refine ⟨?_, ?_, Nat.lt_of_lt_of_le (Nat.add_lt_add_right hn 1) (by decide),
    fun i => Nat.lt_trans (hq i) (by decide), hp, hd⟩
  · rw [hts]
    simp only [_root_.Flock.Setup.ofCircuit, ByteArray.size_extract]
    omega
  · rw [hts]
    exact Nat.lt_trans hm (by decide)

end Claims.ZKServer.Count
