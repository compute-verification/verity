module

public import Proofs.Flock.ZkAny.RelayAlg
public import Proofs.Flock.ZkAny.RelayPlace
public import Proofs.Flock.Soundness.Refine.Walk

/-!
# A rep's frame at its stream's first round, as `Frames.startRep` pushed it

A rep's frame starts fresh (`FreshRep`), and until the rep's first round only updates that keep it come (`Kept`). At
that round the loop schedules the rep's algebraic phase on it (`withAlg`), and from there the frame keeps its schedule
so far (`Loop.RepInv`):

* `frameAt_keptStep_cases`, `freshAt_kept`: an update that keeps a rep's frame leaves it as it was or is a start
  pushing a fresh one, so a run of them keeps a rep's frame absent or fresh (`FreshAt`).
* `fresh_found`, `fresh_repRun`, `fresh_alg`: so the frame a rep's first round finds is fresh, its run on no requests
  (the switch lemmas' `RepRun [] []`), with no scheduled rounds yet (`withAlg_size`'s premise).
* `tower_back`, `cap_root_build`: a level 0 `Level0.build` built has its root its commitment's cap, concatenated.
* `withAlg_keep`, `repInv_withAlg`: `withAlg` sets only the scheduled rounds, so on a fresh frame, at a table whose
  commitment's cap is the level-0 root, the frame it gives keeps its schedule so far.
* `roundPlace_first_of`: at a stream's first round, the readied frame is `placeOf`'s on `withAlg`'s, after rep 0's
  algebraic-phase coins at rep 1.
* **`roundRun_repInv_first`**: so at a `RoundRun` of a stream's first round on a fresh frame, whose round releases and
  whose coins the rep's frame took, the round is in the place the schedule so far gives the stream's first round
  (unless that is the inner proof's own), and the frame the session's next round finds keeps its schedule so far.
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock (F128 Stmt)
open _root_.Flock.Firewall.Session (State Placed byte?)
open _root_.Flock.Firewall.Frames (Sessions Rep Release)
open _root_.Flock.Firewall.Frame (Req Place)

/-- **Session `sid`'s table `i`'s rep `r`'s frame is absent or fresh** in the loop's frames `R`. -/
def FreshAt (R : Sessions) (sid i r : Nat) : Prop := ∀ p, frameAt R sid i r = some p → FreshRep p

/-- **An update that keeps a rep's frame leaves it as it was, or is a start pushing a fresh frame.** -/
theorem frameAt_keptStep_cases {sid i r : Nat} {R R' : Sessions} (h : KeptStep sid i r R R') :
    frameAt R' sid i r = frameAt R sid i r ∨ ∃ q, FreshRep q ∧ R' = startAt R q := by
  cases h with
  | begin => exact Or.inl rfl
  | link => exact Or.inl rfl
  | start q hq => exact Or.inr ⟨q, hq, rfl⟩
  | put p hk => exact Or.inl (frameAt_put_other R p hk)
  | keep =>
    left
    unfold keepOf
    split <;> rfl
  | file root r' b hf =>
    left
    unfold _root_.Flock.Firewall.Frames.Sessions.file at hf
    split at hf
    · cases hf
    · cases hf
      rfl
  | inner root r' j b hf =>
    left
    unfold _root_.Flock.Firewall.Frames.Sessions.innerRound at hf
    simp only at hf
    split at hf
    · cases hf
    · split at hf <;> (cases hf; rfl)
  | close sid' roots hne =>
    left
    unfold frameAt closeOf
    simp only
    rw [Array.find?_filter]
    congr 1
    funext q
    rcases hk : keyIs sid i r q with _ | _
    · simp
    · have hq : q.sid = sid := by
        simp only [keyIs, Bool.and_eq_true, beq_iff_eq] at hk
        exact hk.1.1
      have hb : (q.sid != sid') = true := by simp [hq, Ne.symm hne]
      simp [hb]

/-- **An update that keeps a rep's frame keeps it absent or fresh.** -/
theorem freshAt_keptStep {sid i r : Nat} {R R' : Sessions} (h : KeptStep sid i r R R') (hf : FreshAt R sid i r) :
    FreshAt R' sid i r := by
  intro p hp
  rcases frameAt_keptStep_cases h with he | ⟨q, hq, rfl⟩
  · rw [he] at hp
    exact hf p hp
  · unfold startAt at hp
    split at hp
    · exact hf p hp
    · rcases hfd : frameAt R sid i r with _ | x
      · have hp' : frameAt { R with reps := R.reps.push q } sid i r =
            (if keyIs sid i r q = true then some q else none) := by
          unfold frameAt at hfd ⊢
          simp only
          rw [Array.find?_push, hfd]
          rfl
        rw [hp'] at hp
        split at hp
        · cases hp
          exact hq
        · cases hp
      · have hp' : frameAt { R with reps := R.reps.push q } sid i r = some x := by
          unfold frameAt at hfd ⊢
          simp only
          rw [Array.find?_push, hfd]
          rfl
        rw [hp'] at hp
        exact hf p (hfd.trans hp)

/-- **Updates that keep a rep's frame keep it absent or fresh.** -/
theorem freshAt_kept {sid i r : Nat} {R R' : Sessions} (h : Kept sid i r R R') (hf : FreshAt R sid i r) :
    FreshAt R' sid i r := by
  induction h with
  | refl => exact hf
  | step _ hs ih => exact freshAt_keptStep hs ih

/-- **The frame found where the rep's frame is absent or fresh is fresh.** -/
theorem fresh_found {R : Sessions} {sid i r j : Nat} {p : Rep} (hF : FreshAt R sid i r)
    (hf : R.find sid i r = some (j, p)) : FreshRep p :=
  hF p (by rw [← find_frameAt, hf]; rfl)

/-- **A fresh frame is its run on no requests.** -/
theorem fresh_repRun {p : Rep} (h : FreshRep p) : RepRun [] [] p := repRun_start h.st h.coins

/-- **A fresh frame has no scheduled rounds yet.** -/
theorem fresh_alg {p : Rep} (h : FreshRep p) : p.alg.isEmpty = true := by
  rw [h.alg]
  rfl

/-- **A Merkle tower's last layer is its top.** -/
theorem tower_back (d : Nat) (lf : Nat → ByteArray) :
    ∀ k, (_root_.Flock.Firewall.Level0.tower d lf k).2.back? = some (_root_.Flock.Firewall.Level0.tower d lf k).1
  | 0 => rfl
  | k + 1 => by
    simp only [_root_.Flock.Firewall.Level0.tower, Array.back?_push]

/-- **A built level 0's root is its commitment's cap, concatenated.** -/
theorem cap_root_build {l : _root_.Flock.Firewall.Level0.Level0} {m tPad : Nat} {lv g extra pads salts}
    (h : l = _root_.Flock.Firewall.Level0.build m lv tPad g extra pads salts) :
    _root_.Flock.Firewall.Finish.flat (_root_.Flock.Firewall.Finish.commitOf l).cap = l.root := by
  subst h
  unfold _root_.Flock.Firewall.Finish.commitOf _root_.Flock.Firewall.Level0.build
  simp only [tower_back, Option.getD_some]
  rfl

/-- **`withAlg` sets only the scheduled rounds**: the frame it gives has the given one's key, coins and opening. -/
theorem withAlg_keep {s : Release} {l0 : _root_.Flock.Firewall.Level0.Level0} {name : String} {p q : Rep}
    (h : _root_.Flock.Firewall.Frames.withAlg s l0 name p = .ok q) :
    q.table = p.table ∧ q.rep = p.rep ∧ q.coins = p.coins ∧ q.opening = p.opening ∧ q.opened = p.opened ∧
      q.toOpening = p.toOpening := by
  unfold _root_.Flock.Firewall.Frames.withAlg at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first
    | (cases h; done)
    | (cases h; exact ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩)

/-- **On a fresh frame `withAlg` gives a frame keeping its schedule so far**, at the table its level 0 gives the
frame's table's statement, where that table's commitment's cap is the level-0 root. -/
theorem repInv_withAlg {s : Release} {l0 : _root_.Flock.Firewall.Level0.Level0} {name : String} {p p' : Rep}
    {stm : Stmt} (hp : FreshRep p) (hw : _root_.Flock.Firewall.Frames.withAlg s l0 name p = .ok p')
    (hst : s.sts[p.table]? = some stm) {t : _root_.Flock.Firewall.Finish.Tab}
    (htab : _root_.Flock.Firewall.Finish.tabOfLevel0 (_root_.Flock.Setup.ofCircuit stm) l0 = .ok t)
    (hcap : _root_.Flock.Firewall.Finish.flat t.cm0.cap = l0.root) (xs : Array (Array F128))
    (hi : Array ByteArray) : Loop.RepInv t (s.spec.domain name p.rep).toUTF8 xs hi p' := by
  obtain ⟨sch, tp, hsch, ha⟩ := withAlg_alg hw (fresh_alg hp) hst
  obtain ⟨hts, hsch'⟩ := Count.tabOfLevel0_sched htab
  have e : _root_.Flock.Zk.schedule stm.m = .ok (t.sch, t.tPad) := hsch'
  rw [hsch, Except.ok.injEq, Prod.mk.injEq] at e
  obtain ⟨-, -, hco, hop, hod, hto⟩ := withAlg_keep hw
  refine ⟨?_, by rw [hto]; exact hp.toOpening,
    Or.inl ⟨by rw [hop]; exact hp.opening, by rw [hod]; exact hp.opened, by rw [hco, hp.coins]; exact Nat.zero_le _⟩⟩
  rw [hts, hcap, ← e.1]
  exact ha

set_option maxHeartbeats 1000000 in
/-- **At a stream's first round the readied frame is `placeOf`'s on `withAlg`'s**: on the found frame at its first
round, `withAlg`'s frame, after rep 0's algebraic-phase coins at rep 1 (its state then another), is placed by
`placeOf`, in `placeOf`'s place unless that is the inner proof's own. -/
theorem roundPlace_first_of {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i r j : Nat}
    {dom content : ByteArray} {n : Nat} {p p' q : Rep} {pl : Place} {f : Bool} {nd : Option String}
    (hf : R.find sid i r = some (j, p))
    (h : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n = .ok (q, pl, f, nd))
    (hfst : (p.coins.isEmpty && !p.st.begun) = true) {s : Release} {name : String}
    {l0 : _root_.Flock.Firewall.Level0.Level0} (hs : R.sessions.find? (·.sid == sid) = some s)
    (hn : s.spec.tables[i]? = some name) (hl0 : _root_.Flock.Firewall.Level0.ofSession L0 sid name = some l0)
    (hw : _root_.Flock.Firewall.Frames.withAlg s l0 name p = .ok p') :
    ∃ p'' pl₀ n₀, _root_.Flock.Firewall.Frames.placeOf R.salts l0 s p'' (s.spec.domain name r).toUTF8 =
        .ok (pl₀, n₀, q) ∧ (pl = pl₀ ∨ pl₀ = .own) ∧ (p'' = p' ∨ ∃ st₁, p'' = { p' with st := st₁ }) := by
  unfold _root_.Flock.Firewall.Frames.roundPlace at h
  simp only [hf, bind, Except.bind, pure, Except.pure] at h
  repeat' (first
    | simp only [hs, hn, hl0, hfst, hw, bind, Except.bind, pure, Except.pure, Bool.false_eq_true, ↓reduceIte] at h
    | split at h)
  all_goals first
    | (cases h; done)
    | exact absurd (by assumption) throw_ne_ok
    | (cases h
       exact ⟨_, _, _, by assumption, by first | exact Or.inl rfl | exact Or.inr rfl | exact Or.inr (by assumption),
         by first | exact Or.inl rfl | exact Or.inr ⟨_, rfl⟩⟩)

/-- **A rep's frame after its stream's first round keeps its schedule so far**: at a `RoundRun` of the stream's first
round on a fresh frame, whose round releases and whose coins the rep's frame took off the auditor's answer, at the
table and extras the session gives it, and the salts the loop's salts give it where `withAlg` scheduled no rounds
(`hsalt`, where `placeOf` reads them), where the table's commitment's cap is the level-0 root, the round is in the
place the schedule so far gives the frame's first round, unless that is the inner proof's own, and the frame the
session's next round finds keeps that schedule. -/
theorem roundRun_repInv_first {sid : Nat} {st : State} {pl : Placed} {stream : String} {n i r : Nat}
    {dom content : ByteArray} {e : Env} {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {R₃ : Sessions}
    (hr : RoundRun sid st pl stream n i r dom content e R L0 R₃) (hreq : st.request pl (.round stream n) = none)
    {q p₁ : Rep} {pl' : Place} {f : Bool} {nd : Option String} {o : ByteArray}
    (hrp : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n = .ok (q, pl', f, nd))
    (ho : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom content n).out = .ok o)
    (hput : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom content n).put = some p₁) {a : ByteArray}
    (ha : e.ans = some a) (hb : byte? a 0 = some 2) (hs : (st.answer (.round stream n) a).1.stop = none)
    (hc : e.coins = .ok ()) {pre : List Req} {ev : List (Nat × ByteArray × Place × Array F128)}
    (hR : RepRun pre ev q) {j : Nat} {p : Rep} (hf : R.find sid i r = some (j, p)) (hp : FreshRep p)
    {s : _root_.Flock.Firewall.Frames.Release} {name : String} {l0 : _root_.Flock.Firewall.Level0.Level0}
    (hss : R.sessions.find? (·.sid == sid) = some s) (hn : s.spec.tables[i]? = some name)
    (hl0 : _root_.Flock.Firewall.Level0.ofSession L0 sid name = some l0) {p' : Rep}
    (hw : _root_.Flock.Firewall.Frames.withAlg s l0 name p = .ok p') {t : _root_.Flock.Firewall.Finish.Tab}
    {xs : Array (Array F128)} {hi : Array ByteArray} {stm : Stmt} {c : Array F128} {r0 : ByteArray} {q0 : Nat}
    (hst : s.sts[p.table]? = some stm) (hlink : s.link = some c)
    (hsalt : p'.alg.size = 0 → R.salts.find? (fun x => x.1 == l0.root && x.2.1 == p.rep) = some (r0, q0, hi))
    (htab : _root_.Flock.Firewall.Finish.tabOfLevel0 (_root_.Flock.Setup.ofCircuit stm) l0 = .ok t)
    (hcap : _root_.Flock.Firewall.Finish.flat t.cm0.cap = l0.root)
    (hxs : xs = _root_.Flock.Firewall.Drive.extras stm (_root_.Flock.Firewall.Drive.linkPoints stm c))
    {sched : Array (ByteArray × Nat)}
    (hsc : Loop.schedSoFar t p.rep (s.spec.domain name r).toUTF8 xs hi p.coins = .ok sched) :
    (pl' = (Loop.placeAt sched p.coins.size).1 ∨ (Loop.placeAt sched p.coins.size).1 = .own) ∧
      ∃ p₃, frameAt R₃ sid i r = some p₃ ∧ Loop.RepInv t (s.spec.domain name r).toUTF8 xs hi p₃ := by
  have hrep : p.rep = r := by
    have hk := find_key hf
    simp only [keyOf, Prod.mk.injEq] at hk
    exact hk.2.2
  have hfr : (frameAt R sid i r).isSome := by
    rw [← find_frameAt, hf]
    rfl
  have hfst : (p.coins.isEmpty && !p.st.begun) = true := by
    rw [hp.coins, hp.st]
    rfl
  have hI' := repInv_withAlg hp hw hst htab hcap xs hi
  rw [hrep] at hI'
  obtain ⟨htb, hrp', hco, -, -, -⟩ := withAlg_keep hw
  obtain ⟨p'', pl₀, n₀, hpo, hpl, hp''⟩ := roundPlace_first_of hf hrp hfst hss hn hl0 hw
  have hI'' : Loop.RepInv t (s.spec.domain name r).toUTF8 xs hi p'' ∧ p''.table = p.table ∧ p''.rep = p.rep ∧
      p''.coins = p.coins ∧ p''.alg = p'.alg := by
    rcases hp'' with rfl | ⟨st₁, rfl⟩
    · exact ⟨hI', htb, hrp', hco, rfl⟩
    · exact ⟨⟨hI'.alg, hI'.toOpening, hI'.opening⟩, htb, hrp', hco, rfl⟩
  obtain ⟨hI₂, htb₂, hrp₂, hco₂, hal₂⟩ := hI''
  have hsalt₂ : p''.alg.size ≤ p''.coins.size → p''.opened = false → p''.opening = none →
      R.salts.find? (fun x => x.1 == l0.root && x.2.1 == p''.rep) = some (r0, q0, hi) := by
    intro hle _ _
    rw [hal₂, hco₂, hp.coins, Array.size_empty] at hle
    rw [hrp₂]
    exact hsalt (by omega)
  obtain ⟨q', hpo', hinv⟩ := Loop.placeOf_sched R.salts l0 s p'' (s.spec.domain name r).toUTF8
    (by rw [htb₂]; exact hst) hlink hsalt₂ htab hxs hI₂ (by rw [hrp₂, hco₂]; exact hsc)
  rw [hpo] at hpo'
  simp only [Except.ok.injEq, Prod.mk.injEq] at hpo'
  obtain ⟨hpl0, -, hqq⟩ := hpo'
  rw [hco₂] at hpl0
  subst hqq
  obtain ⟨st₂, -, hat, -⟩ := roundRun_next hr hreq hfr hrp ho hput ha hb hs hc hR
  rcases hres : _root_.Flock.Firewall.Frame.step q.cfg q.st
    (.round n (if nd.isSome then ByteArray.empty else content) pl') with ⟨a', st'⟩
  obtain ⟨v, hv⟩ := roundStep_put hrp hres
  rw [hv, Option.some.injEq] at hput
  subst hput
  refine ⟨?_, _, hat, hinv st' v _ st₂⟩
  rw [← hpl0]
  exact hpl

end Claims.ZKServer.Relay
