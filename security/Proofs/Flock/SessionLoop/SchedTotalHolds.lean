module

public import Proofs.Flock.SessionLoop.ScheduleTotal
public import Proofs.Flock.SessionLoop.SchedTotalWalk
public import Proofs.Flock.SessionLoop.SchedTotalPiop

/-!
# Schedule totality, proved

`Loop.scheduleTotal : ∀ x, Loop.ScheduleTotal x`. At the placeholder's table the opening's program (`Drive.openStep`)
takes the algebraic phase's coins without refusing, each of the skeleton's round's count, the fold returning at every
zerocheck point of the skeleton's sizes (`SchedTotalPiop`), and past them, answered each asked round's count with level
0's pair independent, it does not stop (`SchedTotalWalk`). Fed one more round's coins it is still not stopped, so
`schedSoFar` on them is not refused:

* `Count.openStep_st`: the opening doesn't stop on such coins.
* `Count.feed_not_fail`: fed answers of their rounds' counts, with level 0's pair independent where both are fed, a
  program that doesn't stop is not stopped.
-/

@[expose] public section

namespace Claims.ZKServer.Count

open _root_.Flock (V F128 Transcript TMW liftTM)
open _root_.Flock.Firewall.Drive (Step OpenM stepAsks)

variable {A : Adm} {coins : Got}

theorem os_start {α : Type} {p : TMW OpenM α} (h : OS A (fun g hs => g = #[] ∧ hs = []) (fun _ _ _ => True) p)
    (t : Flock.Transcript) : ST A (fun _ _ => True) ((p.run t).run #[]) [] :=
  h t #[] [] ⟨rfl, rfl⟩

/-- A program that takes its coins from the given ones doesn't stop, nor answer anything. -/
theorem os_of_pt {α : Type} {ns : List ℕ} {Q : α → Prop} {p : TMW OpenM α} (h : PT coins ns Q p) (g0 : Got)
    (h0 : Hist) :
    OS A (fun g hs => g = g0 ∧ Fits coins g0.size ns ∧ hs = h0)
      (fun a g hs => Q a ∧ g.size = g0.size + ns.length ∧ hs = h0) p := by
  rintro t _ _ ⟨rfl, hF, rfl⟩
  obtain ⟨r, e, hq, hs'⟩ := h t _ hF
  rw [e]
  exact ⟨hq, hs', rfl⟩

theorem os_getThe {P : Got → Hist → Prop} : OS A P (fun v g hs => v = g ∧ P g hs) (getThe Got : TMW OpenM Got) :=
  fun _ _ _ hP => ⟨rfl, hP⟩

theorem os_false {α : Type} {P : Got → Hist → Prop} {B : α → Got → Hist → Prop} {p : TMW OpenM α}
    (h : ∀ g hs, P g hs → False) : OS A P B p :=
  fun _ g hs hP => (h g hs hP).elim

theorem os_of_fact {α : Type} {c : Prop} {P : Got → Hist → Prop} {B : α → Got → Hist → Prop} {p : TMW OpenM α}
    (hc : ∀ g hs, P g hs → c) (h : c → OS A P B p) : OS A P B p :=
  fun t g hs hP => h (hc g hs hP) t g hs hP

/-- **The opening doesn't stop** on the algebraic phase's coins `coins`, each of its round's count, at a fold returning
at every zerocheck point, where `A` makes the answer after `ρ₀`'s independent of it. -/
theorem openStep_st (t : _root_.Flock.Firewall.Finish.Tab) (rep : ℕ) (dom : ByteArray) (xs : Array (Array F128))
    (hi : Array ByteArray) (hm : t.st.m < 2 ^ 64) (hF : Fits coins 0 (zcCounts t.st.m ++ lcCounts t.st.fold.kLog))
    (hsz : coins.size = (zcCounts t.st.m ++ lcCounts t.st.fold.kLog).length)
    (hfold : ∀ zc : _root_.Flock.ZcOut, zc.rho.size = t.st.m - 6 → ∀ α, ∃ v,
      t.st.fold.fold α (_root_.Flock.skipWeights zc.z) (zc.rho.extract 0 (t.st.fold.kLog - 6)) = .ok v)
    (hρ : ∀ hs cs,
      hs.length = xs.size + 2 + (sq (xs.size + 2)).length + (sq t.sch.n).length + 1 + t.sch.k0 + 1 → A hs cs →
      _root_.Flock.Zk.rhoIndependent
        (⟨(hs.getD (xs.size + 2 + (sq (xs.size + 2)).length + (sq t.sch.n).length + 1 + t.sch.k0) #[])[0]!,
          (hs.getD (xs.size + 2 + (sq (xs.size + 2)).length + (sq t.sch.n).length + 1 + t.sch.k0) #[])[1]!⟩,
          ⟨cs[0]!, cs[1]!⟩) = true) :
    ST A (fun _ _ => True) (_root_.Flock.Firewall.Drive.openStep t rep dom xs hi coins) [] := by
  unfold _root_.Flock.Firewall.Drive.openStep
  dsimp only
  refine st_run' (R := fun _ => True) (os_start ?_ _)
  refine os_bind (os_mono (os_of_pt (repPiop_pt _ _ _ _ _ _ hm hfold) #[] []) (fun _ _ h => ⟨h.1, hF, h.2⟩)
    (B' := fun a g hs => a.2.2.size = (_root_.Flock.Firewall.Finish.claimsOf xs).size + 2 ∧ g.size = coins.size ∧
      hs = [])
    fun _ _ _ h => ⟨h.1, by rw [h.2.1, hsz, Array.size_empty, Nat.zero_add], h.2.2⟩) fun r => ?_
  obtain ⟨zp, lp, claims⟩ := r
  dsimp only
  refine os_of_fact (fun _ _ h => h.1) fun hc => ?_
  have hc' : claims.size = xs.size + 2 := by
    rw [hc, _root_.Flock.Firewall.Finish.claimsOf, Array.size_map]
  refine os_bind os_getThe fun v => ?_
  split
  · next hne =>
    refine os_false fun _ _ h => ?_
    obtain ⟨rfl, -, hg, -⟩ := h
    simp [hg] at hne
  · refine os_bind (os_mono (repOpening_st _ _ _ _ _ _ _ claims _ _ _ _ (by rw [hc']; exact hρ))
      (fun _ _ h => ⟨Nat.le_of_eq h.2.2.1.symm, by simp [h.2.2.2]⟩) (B' := fun _ _ _ => True)
      fun _ _ _ _ => trivial) fun _ => os_pure _ fun _ _ _ => trivial

/-- **Level 0's pair independent**: the answer after the `R`-th answer independent of it over F128. -/
def RhoAdm (R : ℕ) : Adm := fun hs cs => hs.length = R + 1 →
  _root_.Flock.Zk.rhoIndependent (⟨(hs.getD R #[])[0]!, (hs.getD R #[])[1]!⟩, ⟨cs[0]!, cs[1]!⟩) = true

/-- **Fed answers of its rounds' counts, a program that doesn't stop is not stopped**, the answers `R` and `R + 1`
independent where both are fed. -/
theorem feed_not_fail {S : Step Unit} {L : List (Array F128)} {R : ℕ} (hS : ST (RhoAdm R) (fun _ _ => True) S [])
    (hn : ∀ j (hj : j < L.length) c n, (S.feed L #[]).2[j]? = some (c, n) → L[j].size = n)
    (hρ : ∀ h : R + 1 < L.length,
      _root_.Flock.Zk.rhoIndependent (⟨L[R][0]!, L[R][1]!⟩, ⟨L[R + 1][0]!, L[R + 1][1]!⟩) = true)
    (e : String) : (S.feed L #[]).1 ≠ .fail e := by
  refine st_ne_fail (st_feed S L #[] [] hS fun j hj c n hc =>
    ⟨hn j hj c n (by simpa only [Array.size_empty, Nat.zero_add] using hc), ?_⟩) e
  intro hl
  rw [List.nil_append, List.length_take] at hl
  obtain rfl : j = R + 1 := by omega
  rw [List.nil_append, List.getD_eq_getElem?_getD, List.getElem?_take, ite_eq_left (show R < R + 1 by omega),
    List.getElem?_eq_getElem (show R < L.length by omega), Option.getD_some]
  exact hρ hj

/-- The algebraic phase's round count. -/
theorem counts_length {m kLog : ℕ} (hm : 25 ≤ m) (hk : 6 ≤ kLog) :
    (zcCounts m ++ lcCounts kLog).length = m + kLog - 6 := by
  have e : ∀ n, (List.replicate n [1]).flatten.length = n := fun n => by
    induction n with
    | zero => rfl
    | succ n ih =>
      rw [List.replicate_succ, List.flatten_cons, List.length_append, ih, List.length_singleton]
      omega
  simp only [zcCounts, lcCounts, sq, List.length_append, e, show m - 13 ≠ 0 by omega, ↓reduceIte, List.length_cons,
    List.length_nil]
  omega

end Claims.ZKServer.Count

namespace Claims.ZKServer.Loop

open _root_.Flock (F128)

/-- **Schedule totality holds** at every message. -/
theorem scheduleTotal (x : Msg) : ScheduleTotal x := by
  intro _ spec sts _ k _ link hlink _ t dom xs hi hph cr sched hS hcnt hρ
  by_cases h0 : cr.size = 0
  · have e : cr.pop = cr := by simp [Array.eq_empty_of_size_eq_zero h0]
    rw [e] at hS
    exact ⟨_, hS⟩
  obtain ⟨p, b, rfl⟩ : ∃ p b, cr = Array.push p b := ⟨_, _, Array.eq_push_pop_back!_of_size_ne_zero h0⟩
  rw [Array.pop_push] at hS
  obtain ⟨htst, ⟨tp, hsch⟩, hxs⟩ := Count.placeholder_tab hph
  have hmst : t.st.m = sts[k / 2]!.m := congrArg _root_.Flock.Setup.m htst
  have hkst : t.st.fold.kLog = sts[k / 2]!.c.kLog := congrArg (fun s => s.fold.kLog) htst
  have hm : t.st.m < 2 ^ 64 := by rw [hmst]; exact Nat.lt_trans (Count.sched_facts hsch).1 (by decide)
  have hm25 : 25 ≤ t.st.m := by rw [hmst]; exact Count.schedule_m hsch
  have hn1 : 1 ≤ t.sch.n := (Count.sched_rho hsch).1
  have hxsz : xs.size = spec.points * sts[k / 2]!.regions.size := by
    rw [hxs, Count.extras_size, Count.linkPoints_size, hlink, Nat.mul_comm]
  obtain ⟨alg, hal⟩ := schedSoFar_ok_rounds hS
  unfold schedSoFar at hS ⊢
  rw [hal] at hS ⊢
  simp only [bind, Except.bind] at hS ⊢
  by_cases hlt : (p.push b).size < alg.size
  · simp only [hlt, ↓reduceIte]
    exact ⟨alg, rfl⟩
  simp only [hlt, ↓reduceIte]
  rw [Array.size_push] at hlt
  have hcounts := Count.rounds_counts hm hal
  obtain ⟨zc0, α0, v0, hz0, hv0⟩ := Count.rounds_fold hm hal
  have hv0' : sts[k / 2]!.fold α0 (_root_.Flock.skipWeights zc0.z) (zc0.rho.extract 0 (sts[k / 2]!.c.kLog - 6)) =
      .ok v0 := by
    rw [htst] at hv0
    exact hv0
  have hkl : 6 ≤ t.st.fold.kLog := by
    have := (Count.fold_size hv0').1
    omega
  have halg : alg.size = t.st.m + t.st.fold.kLog - 6 := by
    have e := congrArg List.length hcounts
    rw [List.length_map, Array.length_toList, Count.counts_length hm25 hkl] at e
    exact e
  have hfold : ∀ zc : _root_.Flock.ZcOut, zc.rho.size = t.st.m - 6 → ∀ α, ∃ v,
      t.st.fold.fold α (_root_.Flock.skipWeights zc.z) (zc.rho.extract 0 (t.st.fold.kLog - 6)) = .ok v := by
    intro zc hz α
    have h := Count.fold_ok_of (α' := α) (lw' := _root_.Flock.skipWeights zc.z)
      (ρ' := zc.rho.extract 0 (sts[k / 2]!.c.kLog - 6)) hv0'
      (by rw [FlockSoundness.Refine.skipWeights_size, FlockSoundness.Refine.skipWeights_size])
      (by simp only [Array.size_extract, hz, hz0])
    rw [htst]
    exact h
  obtain ⟨R, hR⟩ : ∃ R, R = xs.size + 2 + (Count.sq (xs.size + 2)).length + (Count.sq t.sch.n).length + 1 + t.sch.k0 :=
    ⟨_, rfl⟩
  have hI : FlockComplete.piopRoundsZk spec sts[k / 2]! + t.sch.k0 + 2 = alg.size + R := by
    rw [hR, halg, Count.sq_length (n := xs.size + 2) (by omega), Count.sq_length (n := t.sch.n) (by omega), hxsz,
      hmst, hkst]
    unfold FlockComplete.piopRoundsZk
    omega
  have hrel : sched = alg ++ ((_root_.Flock.Firewall.Drive.openStep t (k % 2) dom xs hi
      ((p.push b).extract 0 alg.size)).feed ((p.push b).extract alg.size (p.push b).size).toList #[]).2 := by
    by_cases hle : p.size < alg.size
    · simp only [hle, ↓reduceIte, pure, Except.pure, Except.ok.injEq] at hS
      subst hS
      rw [Array.extract_empty_of_size_le_start (xs := p.push b) (start := alg.size) (stop := (p.push b).size)
        (by rw [Array.size_push]; omega), Array.toList_empty, feed_nil]
      exact Array.append_empty.symm
    · simp only [hle, ↓reduceIte] at hS
      have e1 : (p.push b).extract 0 alg.size = p.extract 0 alg.size := by
        rw [Array.extract_push, ite_eq_left (show alg.size ≤ p.size by omega)]
      have e2 : ((p.push b).extract alg.size (p.push b).size).toList = (p.extract alg.size p.size).toList ++ [b] := by
        rw [Array.size_push, Array.extract_push, ite_eq_right (show ¬p.size + 1 ≤ p.size by omega),
          ite_eq_left (show alg.size ≤ p.size by omega), Array.toList_push]
      rw [e1, e2, feed_append]
      revert hS
      generalize (_root_.Flock.Firewall.Drive.openStep t (k % 2) dom xs hi (p.extract 0 alg.size)).feed
        (p.extract alg.size p.size).toList #[] = P
      obtain ⟨S, rs⟩ := P
      intro hS
      cases S with
      | ask c n kk =>
        simp only [pure, Except.pure, Except.ok.injEq] at hS
        subst hS
        show alg ++ rs.push (c, n) = alg ++ ((kk b).feed [] (rs.push (c, n))).2
        rw [feed_nil]
      | done _ =>
        simp only [pure, Except.pure, Except.ok.injEq] at hS
        subst hS
        rfl
      | fail _ => cases hS
  subst hrel
  have hF : Count.Fits ((p.push b).extract 0 alg.size) 0
      (Count.zcCounts t.st.m ++ Count.lcCounts t.st.fold.kLog) := by
    rw [← hcounts]
    intro i hi
    have hia : i < alg.size := by simpa only [List.length_map, Array.length_toList] using hi
    have hic : i < (p.push b).size := by rw [Array.size_push]; omega
    refine ⟨(p.push b)[i], ?_, ?_⟩
    · rw [Nat.zero_add, Array.getElem?_extract, ite_eq_left (show i < min alg.size (p.push b).size - 0 by omega),
        Nat.zero_add]
      exact getElem?_pos (p.push b) i hic
    · have h := hcnt i (by rw [Array.size_append]; omega) hic
      rw [getElem!_pos (p.push b) i hic, Array.getElem_append_left hia] at h
      rw [h, List.getElem_map, Array.getElem_toList]
  have hsz : ((p.push b).extract 0 alg.size).size =
      (Count.zcCounts t.st.m ++ Count.lcCounts t.st.fold.kLog).length := by
    rw [← hcounts, List.length_map, Array.length_toList, Array.size_extract, Array.size_push]
    omega
  have hST := Count.openStep_st (A := Count.RhoAdm R) t (k % 2) dom xs hi hm hF hsz hfold
    fun hs cs hl hA => by subst hR; exact hA hl
  have hLlen : ((p.push b).extract alg.size (p.push b).size).toList.length = (p.push b).size - alg.size := by
    rw [Array.length_toList, Array.size_extract, Nat.min_self]
  have hLget : ∀ j (hj : j < ((p.push b).extract alg.size (p.push b).size).toList.length),
      ((p.push b).extract alg.size (p.push b).size).toList[j] = (p.push b)[alg.size + j]! := by
    intro j hj
    have hj' : alg.size + j < (p.push b).size := by omega
    rw [Array.getElem_toList, Array.getElem_extract, getElem!_pos (p.push b) (alg.size + j) hj']
  generalize _root_.Flock.Firewall.Drive.openStep t (k % 2) dom xs hi ((p.push b).extract 0 alg.size) = S0
    at hST hcnt ⊢
  generalize ((p.push b).extract alg.size (p.push b).size).toList = L at hcnt hLget hLlen ⊢
  have hnf : ∀ e, (S0.feed L #[]).1 ≠ .fail e := by
    refine Count.feed_not_fail hST (fun j hj c n hc => ?_) fun h => ?_
    · obtain ⟨hlt2, hv⟩ := Array.getElem?_eq_some_iff.1
        (show (alg ++ (S0.feed L #[]).2)[alg.size + j]? = some (c, n) by
          rw [Array.getElem?_append_right (Nat.le_add_right _ _), Nat.add_sub_cancel_left]
          exact hc)
      have h := hcnt (alg.size + j) hlt2 (by omega)
      rw [hv] at h
      rw [hLget j hj]
      exact h
    · have h' := hρ (by omega)
      rw [hI] at h'
      rw [hLget R (by omega), hLget (R + 1) h]
      exact h'
  revert hnf
  generalize S0.feed L #[] = P
  obtain ⟨S, rs⟩ := P
  intro hnf
  cases S with
  | ask _ _ _ => exact ⟨_, rfl⟩
  | done _ => exact ⟨_, rfl⟩
  | fail e => exact (hnf e rfl).elim

end Claims.ZKServer.Loop
