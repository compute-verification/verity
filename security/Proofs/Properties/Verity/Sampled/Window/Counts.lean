module

public import Proofs.Properties.Verity.Sampled.Window.Record
public import Proofs.Properties.Verity.Sampled.Window.Trees
public import Proofs.Properties.Verity.Sampled.Window.Writes
public import Proofs.Properties.Verity.Sampled.CountsLayout

/-!
# Sampled proofs' coupling over a window: a session's stops' frames and counts

What the members' steps (`MemberSteps`) read at a stop of a member's session, beside `Window/Record.lean`'s moves:

* **A stop's frame, on any tape through it** (`run_ret_through`, `Pos.Past.through`, `stop_frame_through`): at a
  position that keeps the invariant and where `StopOf` says, the play on a tape through the position ends there with
  the log walked so far, or has next the opening frame the rounds fixed there pick, the cuts at the bytes it answers,
  every description of the opening asking for the frame's count, and the walk from the stop, consumed, stopping
  where `StopOf` says. It is `RoundsReach.stop_frame` with no trace after the stop, so it reads the stop after a
  session's last round too, its closing round's.
* **A trace off another hold** (`RoundsEnd.weaken`, `MemberReach.weaken`): a member's trace off `BL` and `B` is one
  off `BL'` and `B'` wherever its first stop and each stop it answers are off them.
* **A session's rounds but its closing one** (`session_card`, `covers_of_card`): at `per` rounds a session they are at
  most `per - 1`, so picks after as many logs as that are every pick of theirs.
* **A session's slots** (`slotOf_eq_close`, `closeSlot_ne_link`, `slotOf_mid`): the closing round's slot is at the
  last place only, is no link's, and every other slot but the link's is a rep round's of the first table.
* **The session `--zk` sets up at a member** (`Member.spec_at`, `Member.mPts_eq`, `Member.tabs_eq`): at one table
  with hm96-sha512 rows, the member's table, `Σ` a SHA-512 digest, rounds retained, two link points, SHA-512 round
  digests, the member's link points' count at its part of a sample, and one table in the member's plan.
* **The counts** (`SessionRead.link_n`, `session_stop_count`): a session's first opening asks for both link points'
  coordinates, and an answered stop's opening for its message's count.
-/

@[expose] public section

namespace Verity.Sampled

open Verity.Game
open Lean (Json)
open Spot.Challenger
open FlockSoundness.Discharge
open FlockSoundness.Discharge.Tables (LiveOutJ SubmitJ addRoundJ)
open FlockSoundness.Refine (roundOf)

local notation "𝔽" => FlockLevel3.GF128

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-! ## A stop's frame, on any tape through it -/

/-- **A position past `p` has a tape through `p`'s next stop.** -/
theorem _root_.Verity.Game.Pos.Past.through {β W κ : Type} {g : Game β} {σ : Strategy g} {p r : Pos g σ W κ}
    (h : p.Past r) : ∃ w, r.tape = p.toStop.back w := by
  rcases h with h | h
  · rw [h]
    exact ⟨p.toStop.x, (Pos.toStop_tape p).symm⟩
  · exact h.through

section Through

variable {g : Game Exchange} {σ : Strategy g} {R : ℕ}

/-- **The play on a tape through a position at a play's end is that end**, at a position that keeps the invariant
(`run_tape_ret`, on any tape through it). -/
theorem run_ret_through {Ks₀ : Fin R → DigestCut Claims.Msg g} {q : Pos g σ Claims.Msg (Fin R)} (hi : q.Inv Ks₀)
    {y : Exchange} (hy : q.g' = .ret y) (w : CoinTape q.g' q.s') : CoinTape.run g σ (q.back w) = y := by
  have key : ∀ p : Pos g σ Claims.Msg (Fin R), p.g' = .ret y → ∀ t, CoinTape.run p.g' p.s' t = y :=
    fun ⟨_, _, _, _, _⟩ hg t => by
      subst hg
      rfl
  exact (hi.run w).trans (key q hy w)

/-- **A stop's frame, on any tape through it** (`RoundsReach.stop_frame`, with no trace after the stop): at a
position `q` that keeps the invariant at the cuts `Ks₀` and where `StopOf` says from the log `pre₀` at the live rounds
`a`, where each round's reading picks only openings the device serves, the play on a tape `z` through `q` has after
`pre₀` a log `L` none of the live rounds picks in, and then either ends, every cut of `q` past; or has an opening
frame `.opening i s n` that some live round whose cut `q` fixes picks, the rounds whose cuts `q` fixes being the
live ones that pick it, with the cuts `q` fixes at the bytes it answers (`CutsAt`), every description of `q`'s
opening asking for its `n` coins, and the walk from `q`, consumed, on any rewriting of the tape there stopping where
`StopOf` says from the log before that frame, at the live rounds that don't pick it (`stopOf_next`). -/
theorem stop_frame_through {P : Fin R → List Release → Frame → Prop} {rv : Fin R → Exchange → Option Claims.Msg}
    {sg : Signer} {θ : ByteArray} {F : Array ByteArray} {c rq : ByteArray}
    (hP : ∀ j pre fr, P j pre fr → ∃ i s n, fr = .opening i s n)
    (hPS : ∀ j pre i s n, P j pre (.opening i s n) → Served pre i s n) {Ks₀ : Fin R → DigestCut Claims.Msg g}
    {q : Pos g σ Claims.Msg (Fin R)} (hi : q.Inv Ks₀) {pre₀ : List Release} {f : List Release → Game Exchange}
    {a : Fin R → Prop} (hf : ∀ log, f log = finish sg θ F c rq (pre₀ ++ log)) (hS : StopOf P rv pre₀ f a q)
    {z : CoinTape g σ} (hz : ∃ w, z = q.back w) :
    ∃ L, NoPick P a pre₀ L ∧
      (((∀ j, q.Ks j = .skip) ∧ (CoinTape.run g σ z).log = pre₀ ++ L) ∨
        ∃ (i s n : ℕ) (b : ByteArray) (bits : List Bool) (L' : List Release) (f' : List Release → Game Exchange),
          (∃ j, a j ∧ (q.Ks j).isHere = true ∧ P j (pre₀ ++ L) (.opening i s n)) ∧
          (∀ j, (q.Ks j).isHere = true ↔ a j ∧ P j (pre₀ ++ L) (.opening i s n)) ∧
          (CoinTape.run g σ z).log = (pre₀ ++ L) ++ ⟨.opening i s n, b, bits⟩ :: L' ∧
          CutsAt rv q (pre₀ ++ L) (CoinTape.run g σ z).log ∧
          (∀ o' : OpeningAt (finish sg θ F c rq) q.g', o'.n = n) ∧
          (∀ log, f' log = finish sg θ F c rq ((pre₀ ++ L) ++ log)) ∧
          ∀ φ : CoinTape q.g' q.s' → CoinTape q.g' q.s',
            StopOf P rv (pre₀ ++ L) f' (fun j => a j ∧ ¬ P j (pre₀ ++ L) (.opening i s n))
              (q.rewrite φ).consume.toStop) := by
  obtain ⟨L, hNP, h⟩ := hS
  obtain ⟨w, rfl⟩ := hz
  refine ⟨L, hNP, ?_⟩
  rcases h with ⟨hK, y, hy, s₀, x₀, hrun⟩ |
    ⟨k, fs, ts, f', i, s, n, b, hf', hT, ⟨j, haj, hj⟩, s'', Ks, x'', back'', hq, hKs⟩
  · refine Or.inl ⟨hK, ?_⟩
    obtain ⟨s₂, h₂⟩ := playReaches_of_eq (hf L) s₀ (run_reaches _ s₀ x₀)
    rw [hrun] at h₂
    rw [← log_of_finish h₂]
    exact congrArg Exchange.log (run_ret_through hi hy w)
  · refine Or.inr ?_
    have hjK : (q.Ks j).isHere = true := by
      rw [hq]
      exact (isHere_openCuts hKs j).2 ⟨haj, hj⟩
    have hall : ∀ j', (q.Ks j').isHere = true ↔ a j' ∧ P j' (pre₀ ++ L) (.opening i s n) := fun j' => by
      rw [hq]
      exact isHere_openCuts hKs j'
    have hf'' : ∀ log, f' log = finish sg θ F c rq ((pre₀ ++ L) ++ log) := fun log => by
      rw [hf', hf, List.append_assoc]
    have hSv := hPS j _ i s n hj
    subst hq
    obtain ⟨T, hTi, _, _, hp, -⟩ := Served.next hT hSv
    have hc := Served.cap hT hSv
    have hX : CoinTape.run g σ (back'' w) = CoinTape.run (optOpenK k fs ts f' i s n b ts[i]?) s'' w := hi.run w
    obtain ⟨L', hL'⟩ := log_of_opening hf'' hc hTi hp s'' w
    rw [← hX] at hL'
    exact ⟨i, s, n, b, _, L', f', ⟨j, haj, hjK, hj⟩, hall, hL', cutsAt_openCuts rfl hKs hL', fun o' =>
      OpeningAt.n_eq (finish_recordsLog sg θ F c rq) o' (OpeningAt.ofNext (k := k) (b := b) hf'' hc hTi hp), hf'',
      fun φ => stopOf_next hP s'' Ks x'' back'' rfl hT hKs φ⟩

end Through

/-! ## A trace off another hold -/

section Weaken

variable {g : Game Exchange} {σ : Strategy g} {Ω Reg : Type} {R mPts : ℕ} {D : PosDecoders g σ R Reg}
  {M : CoinMap g σ Ω R mPts} {dg : ByteArray → ByteArray} {J : ℕ} {B : RoundHold}

/-- **A trace to its end off `B` is one off `B'`** wherever, at every position `Q` holds of, each stop the trace
answers there, the trace's end a later position than the stop with its coins written in and consumed, is off `B'`,
and `Q` holds of each answered stop's next position (`RoundsReach.weaken`, on a trace to its end). -/
theorem RoundsEnd.weaken {B' : RoundHold} {rootB : Array ByteArray} {pts : Fin 2 → Fin mPts → 𝔽} {N : ℕ}
    {rs : Fin J → Fin 2 → List Flock.Round} {p q : Pos g σ Claims.Msg (Fin R)} {out : LiveOutJ J mPts}
    (Q : Pos g σ Claims.Msg (Fin R) → Prop)
    (hstep : ∀ q (φ : CoinTape q.toStop.g' q.toStop.s' → CoinTape q.toStop.g' q.toStop.s'), Q q →
      Q (q.toStop.rewrite φ).consume)
    (he : RoundsEnd D M dg J B rootB pts N rs p q out)
    (hB : ∀ p' (s : SubmitJ J) (c : Fin s.n → 𝔽), Q p' → D.sub J p'.toStop.modes = some s →
      ¬ B p'.toStop.g' J s c → (p'.toStop.rewrite (M.round p'.toStop J s c)).consume.Le q →
      ¬ B' p'.toStop.g' J s c) (hp : Q p) :
    RoundsEnd D M dg J B' rootB pts N rs p q out := by
  revert hB hp
  induction he with
  | zero rs p => exact fun _ _ => .zero rs p
  | stop N rs p h => exact fun _ _ => .stop N rs p h
  | round N rs p s c q out h hB₀ ho ih =>
    exact fun hB hp => .round N rs p s c q out h (hB p s c hp h hB₀ ho.le) (ih hB (hstep p _ hp))

end Weaken

section WeakenMember

variable {g : Game Exchange} {σ : Strategy g} {Ω Reg : Type} {R nJ : ℕ} {D : ℕ → PosDecoders g σ R Reg}
  {M : WindowCoinMap g σ Ω R} {BL : WindowLinkHold} {B : WindowRoundHold} {S : Finset (Fin nJ)} {Rg : Reg}

/-- **A member's session's trace off `BL` and `B` is one off `BL'` and `B'`** wherever its link points, the trace's
end a later position than its first stop with them written in and consumed, are off `BL'` there, and each stop the
trace answers at a position `Q` holds of is off `B'`, `Q` holding past the first stop and of each answered stop's
next position. -/
theorem MemberReach.weaken {BL' : WindowLinkHold} {B' : WindowRoundHold} {i : ℕ}
    {m : Flock.Guarantees.MemberV5 Reg nJ} {p q : Pos g σ Claims.Msg (Fin R)}
    {out : LiveOutJ (memberTabs S Rg m) (memberPts m)} (Q : Pos g σ Claims.Msg (Fin R) → Prop)
    (hstep : ∀ q (φ : CoinTape q.toStop.g' q.toStop.s' → CoinTape q.toStop.g' q.toStop.s'), Q q →
      Q (q.toStop.rewrite φ).consume)
    (h : MemberReach D M BL B S Rg i m p q out)
    (hBL : ∀ pts, ¬ BL i (memberPts m) p.toStop.g' pts →
      (p.toStop.rewrite ((M.member i (memberPts m)).link p.toStop pts)).consume.Le q →
      ¬ BL' i (memberPts m) p.toStop.g' pts)
    (hQ : ∀ pts, Q (p.toStop.rewrite ((M.member i (memberPts m)).link p.toStop pts)).consume)
    (hB : ∀ p' (s : SubmitJ (memberTabs S Rg m)) (c : Fin s.n → 𝔽), Q p' →
      (D i).sub (memberTabs S Rg m) p'.toStop.modes = some s → ¬ B i p'.toStop.g' (memberTabs S Rg m) s c →
      (p'.toStop.rewrite ((M.member i (memberPts m)).round p'.toStop (memberTabs S Rg m) s c)).consume.Le q →
      ¬ B' i p'.toStop.g' (memberTabs S Rg m) s c) :
    MemberReach D M BL' B' S Rg i m p q out := by
  obtain ⟨pts, hL, he⟩ := h
  exact ⟨pts, hBL pts hL he.le, he.weaken Q hstep hB (hQ pts)⟩

end WeakenMember

/-! ## A session's rounds but its closing one -/

/-- **A session's rounds but its closing one are at most `per - 1`**, at `per` rounds a session. -/
theorem session_card (R : ℕ) {per : ℕ} (hper : 0 < per) (u : ℕ) :
    (Finset.univ.filter fun j : Fin R => j.val / per = u ∧ j.val % per + 1 ≠ per).card ≤ per - 1 := by
  refine (Finset.card_le_card_of_injOn (t := Finset.range (per - 1)) (fun j : Fin R => j.val % per)
    (fun j hj => ?_) (fun j hj j' hj' e => ?_)).trans (Finset.card_range _).le
  · have h := (Finset.mem_filter.1 (Finset.mem_coe.1 hj)).2
    have hlt : j.val % per < per := Nat.mod_lt _ hper
    show j.val % per ∈ (↑(Finset.range (per - 1)) : Set ℕ)
    exact Finset.mem_coe.2 (Finset.mem_range.2 (by omega))
  · have h := (Finset.mem_filter.1 (Finset.mem_coe.1 hj)).2
    have h' := (Finset.mem_filter.1 (Finset.mem_coe.1 hj')).2
    have e' : j.val % per = j'.val % per := e
    exact Fin.ext (by rw [← Nat.div_add_mod j.val per, ← Nat.div_add_mod j'.val per, h.1, h'.1, e'])

/-- **Picks after as many logs as the rounds of `T` but one are every pick of `T`'s** (`covers_of_length`, among the
rounds `T`): where each round picks at most one release of `Lg`, round `j₀` of `T` picks the release after `L₀`, and
some round of `T` picks the release after each of the distinct logs `ms`, each longer than `L₀` and at least as many
as `T`'s rounds but one, every release a round of `T` picks is after `L₀` or after one of `ms`. -/
theorem covers_of_card {R : ℕ} {P : Fin R → List Release → Frame → Prop} {Lg : List Release}
    (hU : PicksOnce P Lg) (T : Finset (Fin R)) {L₀ : List Release} {j₀ : Fin R} (hj₀ : j₀ ∈ T)
    (h₀ : PicksIn P Lg j₀ L₀) {ms : List (List Release)} (hpick : ∀ m ∈ ms, ∃ j, j ∈ T ∧ PicksIn P Lg j m)
    (hgt : ∀ m ∈ ms, L₀.length < m.length) (hnd : ms.Nodup) (hlen : T.card ≤ ms.length + 1) :
    ∀ j ∈ T, ∀ L₁, PicksIn P Lg j L₁ → L₁ = L₀ ∨ L₁ ∈ ms := by
  classical
  intro j hjT L₁ hp
  by_contra hne
  rw [not_or] at hne
  let g : List Release → Fin R := fun m => if h : ∃ j, j ∈ T ∧ PicksIn P Lg j m then h.choose else j₀
  have hg : ∀ m ∈ ms, g m ∈ T ∧ PicksIn P Lg (g m) m := fun m hm => by
    have h : ∃ j, j ∈ T ∧ PicksIn P Lg j m := hpick m hm
    have e : g m = h.choose := dif_pos h
    rw [e]
    exact h.choose_spec
  have hnd' : (j₀ :: j :: ms.map g).Nodup := by
    refine List.nodup_cons.2 ⟨fun h => ?_, List.nodup_cons.2 ⟨fun h => ?_, List.Nodup.map_on
      (fun x hx y hy e => hU (g y) x y (e ▸ (hg x hx).2) (hg y hy).2) hnd⟩⟩
    · rcases List.mem_cons.1 h with h | h
      · exact hne.1 (hU j L₁ L₀ hp (h ▸ h₀))
      · obtain ⟨m, hm, e⟩ := List.mem_map.1 h
        have h' := hU j₀ m L₀ (e ▸ (hg m hm).2) h₀
        have h₂ := hgt m hm
        rw [h'] at h₂
        exact lt_irrefl _ h₂
    · obtain ⟨m, hm, e⟩ := List.mem_map.1 h
      have h' := hU j m L₁ (e ▸ (hg m hm).2) hp
      exact hne.2 (h' ▸ hm)
  have hsub : (j₀ :: j :: ms.map g).toFinset ⊆ T := fun x hx => by
    rcases List.mem_cons.1 (List.mem_toFinset.1 hx) with h | h
    · rw [h]
      exact hj₀
    · rcases List.mem_cons.1 h with h | h
      · rw [h]
        exact hjT
      · obtain ⟨m, hm, e⟩ := List.mem_map.1 h
        rw [← e]
        exact (hg m hm).1
  have hc := Finset.card_le_card hsub
  rw [List.toFinset_card_of_nodup hnd'] at hc
  simp only [List.length_cons, List.length_map] at hc
  omega

/-! ## A session's slots -/

/-- **A session's slot is its closing round's only at the session's last place**, where there is a table. -/
theorem slotOf_eq_close {T per i : ℕ} (hT : 0 < T) : slotOf T per i = closeSlot T ↔ i + 1 = per :=
  ⟨fun h => by
    by_contra hc
    rw [slotOf_of_ne T hc] at h
    exact slotAt_ne_close hT i h, slotOf_last T⟩

/-- A session's closing round's slot is not the link's. -/
theorem closeSlot_ne_link (T : ℕ) : closeSlot T ≠ .link := fun h => absurd h (by simp [closeSlot])

/-- **A session's slot before its last place, not the link's, is a rep round's of the first table.** -/
theorem slotOf_mid {T per i : ℕ} (hl : slotOf T per i ≠ .link) (hc : i + 1 ≠ per) :
    ∃ (r : Fin 2) (k : ℕ), slotOf T per i = .rep 0 r k := by
  rw [slotOf_of_ne T hc] at hl ⊢
  cases i with
  | zero => exact absurd slotAt_zero hl
  | succ i => exact ⟨⟨i % 2, Nat.mod_lt i (by decide)⟩, i / 2, rfl⟩

/-! ## The session `--zk` sets up at a member -/

/-- **The session `--zk` sets up at a member of one table with hm96-sha512 rows**, for any record: the member's one
table, `Σ` a SHA-512 digest, rounds retained, two link points, and SHA-512 for a round's digest (`SetsUp.setup_one`,
`SetsUp.spec_tables`, `SetsUp.spec_dg`, at a member of a window). -/
theorem Member.spec_at {nJ : ℕ} (m : Member nJ) (hc : m.I.count = 1) (hr : m.I.tags.hm96Rows = true)
    {record : String} {spec : Flock.SessionSpec} {sts : Array Flock.Stmt}
    (hs : Exec.setupOfP (ZkExec.zkInputs m.I) m.own record = .ok (spec, sts)) :
    spec.tables = #[m.I.tags.table] ∧ spec.sigma.size = 64 ∧ spec.retainRounds = true ∧ spec.points = 2 ∧
      spec.sha512Digests = true ∧ spec.dg = Flock.Sha512.hash := by
  obtain ⟨-, hσ, hret, hp2⟩ := setup_one m.hTags hc hs
  have h512 := setup_sha512Digests m.hTags.1 hr hc hs
  refine ⟨?_, hσ, hret, hp2, h512, ?_⟩
  · have hs' := hs
    unfold Exec.setupOfP at hs'
    rcases HiddenExec.setupTables_okHP ((ZkExec.zkTags_hiddenOutputs m.I).trans m.hTags.1) hs' with
      ⟨-, st, hst, hsp, -⟩ | ⟨h1, -⟩
    · obtain ⟨_, hP⟩ := Hidden.setupHidden_specP hst
      rw [hsp]
      show #[st.tags.table] = #[m.I.tags.table]
      rw [hP.tagsEq]
      exact congrArg (fun t => #[t]) (zkTags_table m.I)
    · exact absurd h1 (by show ¬ 1 < m.I.count; omega)
  · funext b
    simp [Flock.SessionSpec.dg, h512]

/-- **The session `--zk` sets up at a member's part of a sample has the member's link points' count**: its one
statement is set up at the record's sample, the member's printed part, where it has the link points of as many proof
blocks as the part (`drawOkZKHJP_of_count`), the member's at a part of its session's size (`Window.hcls`). -/
theorem Member.mPts_eq {nJ : ℕ} (m : Member nJ) (t : ℕ) {record : String} {spec : Flock.SessionSpec}
    {sts : Array Flock.Stmt} (hs : Exec.setupOfP (ZkExec.zkInputs m.I) m.own record = .ok (spec, sts))
    {S : Finset (Fin nJ)} (hdr : Exec.recordDraw record = some (printedPart (m.part S)))
    (hcls : Exec.mPtsOf m.D.unroll.g (m.part S).card = Exec.mPtsOf m.D.unroll.g (m.kd / Composed.nTab m.I)) :
    spec.mPts = memberPts (m.toV5 t) := by
  have hpin := ZkHidden.drawOkZKHJP_of_count m.I m.own m.ht (m.toV5 t).hcS m.hcount (fun S => .subset S.card) m.n
    printedPart (m.toV5 t).hCust.2 (m.part S)
  have h1 : m.I.count ≤ 1 := m.hcount
  unfold Exec.setupOfP at hs
  rw [hdr] at hs
  rcases HiddenExec.setupTables_okHP ((ZkExec.zkTags_hiddenOutputs m.I).trans m.hTags.1) hs with
    ⟨-, st, hst, hsp, -⟩ | ⟨h1', -⟩
  · rw [hsp]
    refine Eq.trans ((hpin ⟨0, Composed.nTab_pos m.I⟩).2 st ?_) hcls
    simp only [Exec.tagsAt, Composed.drawJ, if_pos h1]
    exact hst
  · exact absurd h1' (by show ¬ 1 < m.I.count; omega)

/-- A member's plan has one table. -/
theorem Member.tabs_eq {nJ : ℕ} (m : Member nJ) (t : ℕ) (S : Finset (Fin nJ)) (Rg : ByteArray) :
    memberTabs S Rg (m.toV5 t) = 1 := by
  have h1 : m.I.count ≤ 1 := m.hcount
  show (ZkHidden.planZHJP (Reg := ByteArray) (m.toV5 t).dj (m.toV5 t).y₀ ((m.toV5 t).part S) Rg).length = 1
  rw [ZkHidden.planZHJP_length]
  show Composed.nTab m.I = 1
  simp [Composed.nTab, h1]

/-! ## The counts -/

/-- **A session's first opening asks for both link points' coordinates**: where the verifier's read holds the log to
session `u`'s record (`SessionRead`), decoded at a session of two link points, the opening that first opens the
`u`-th tree asks for `spec.mPts + spec.mPts` coins (`SessionRead.linkCount`, `link_count`). -/
theorem SessionRead.link_n {Lg : List Release} {tables : Array String} {u : ℕ} {text : String} {J : Json}
    {sess : Flock.RunRecord} {w : Claims.Msg} (hR : SessionRead Lg tables u text J sess w)
    {spec : Flock.SessionSpec} (hd : Flock.Record.decode spec text = .ok sess) (hp2 : spec.points = 2)
    {L₁ L₂ : List Release} {x : Release} {i s n : ℕ} (he : Lg = L₁ ++ x :: L₂) (hx : x.frame = .opening i s n)
    (hni : i ∉ openedTrees L₁) (hlen : (openedTrees L₁).length = u) : n = spec.mPts + spec.mPts := by
  obtain ⟨-, -, b, hb, -⟩ := hR.link L₁ x L₂ i s n he hx hni hlen
  rw [hR.linkCount L₁ x L₂ i s n he hx hni hlen b hb]
  exact link_count hd hR.parse hp2 hb

/-- **An answered stop's opening asks for its message's count** (`stop_count`, at a session of a window): at
`session_stop_msg`'s stop, the opening frame `.opening i s n` asks for `tailCount w` coins: the verifier's read holds
the frame's count to the record's round's (`SessionRead.roundCount`), `tailCount` of the round's message
(`rep_tailCount_accepted`), which is `w` (`session_stop_msg`). -/
theorem session_stop_count {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray} {R : ℕ}
    {tb : String} (per : ℕ) (ev : ℕ → Exchange → Option Claims.Msg) (record₀ : Exchange → Option Claims.Msg)
    (recordDigest : Claims.Msg) {rd : Reading R}
    (hrd : rd = windowReadingAt #[tb] per (sessionReveal #[tb] ev) record₀ recordDigest)
    {σ : Strategy (V.game θ F c)} {q : Pos (V.game θ F c) σ Claims.Msg (Fin R)}
    (hq : q.Inv (roundCuts V θ F c rd)) {z : CoinTape (V.game θ F c) σ} (hz : ∃ y, z = q.back y)
    (hn : NoMiss V θ F c rd σ z) {j : Fin R} {u : ℕ} {ρ : Fin 2} {k : ℕ}
    (hLj : slotOf (#[tb] : Array String).size per (j.val % per) = .rep 0 ρ k) (hju : j.val / per = u)
    (hj : (q.Ks j).isHere = true)
    {w : Claims.Msg} (hw : q.modes j = some w) {bits : List Bool} {m L' : List Release} {i s n : ℕ}
    {b : ByteArray} (hlog : (CoinTape.run (V.game θ F c) σ z).log = m ++ ⟨.opening i s n, b, bits⟩ :: L')
    (hcut : CutsAt rd.reveal q m (CoinTape.run (V.game θ F c) σ z).log)
    (hτ : (openedTrees m)[u]? = some i) (hk : openedOf m i s = k)
    (hname : ((helloSpecs m)[i]?.bind fun sp => sp.streams[s]?) = some (repStreamName tb ρ.val))
    {text : String} {J : Json} {spec : Flock.SessionSpec} {tabs : Array Flock.Setup} {proofs : Array ByteArray}
    {sess : Flock.RunRecord} {w₀ : Claims.Msg}
    (hrec : sentRecordAt u (CoinTape.run (V.game θ F c) σ z).sent = some text)
    (hR : SessionRead (CoinTape.run (V.game θ F c) σ z).log #[tb] u text J sess w₀)
    (hd : Flock.Record.decode spec text = .ok sess) (hv : Flock.Zk.verify spec tabs text proofs = .ok ())
    (hret : spec.retainRounds = true) (h512 : spec.sha512Digests = true) (htab : spec.tables = #[tb]) :
    n = tailCount w := by
  obtain ⟨st, hst, hkR, hmsg⟩ := session_stop_msg per ev record₀ recordDigest hrd hq hz hn hLj hju hj hw hlog hcut
    hτ hk hname hrec hR hd hv hret h512 htab
  subst hk
  have h0 : tb = spec.tables[0]! := by
    rw [htab]
    all_goals rfl
  subst h0
  have hnT := repStreamName_ne spec.tables[0]! ρ.val
  have hn' := hR.roundCount m _ L' i s n _ hlog rfl hτ hname hnT st hst hkR
  rw [getElem!_pos st.rounds (openedOf m i s) hkR] at hn'
  exact hn'.trans (rep_tailCount_accepted hv hd hret ρ hst hkR hmsg).symm

end Verity.Sampled
