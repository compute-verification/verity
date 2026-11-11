module

public import Proofs.Flock.ZkAny.RelayRound

/-!
# A rep's frame is its frame run on the loop's requests to it

Between its start and each round, a rep's frame in the loop's frames is the frame's run from its start (`Frame.run`
from `{}`) on the requests the loop handed it: at rep 1, rep 0's algebraic-phase coins (`prior`), then each round it
took, in its place, and that round's coins (`RepRun`). A round that releases and its coins extend the run by those two
requests:

* `run_snoc`: `Frame.run` on one more request, where the run so far has not ended, is the step on it.
* `evReqs`, `RepRun`: the loop's requests for the rounds a rep took, and the frame its run on them.
* `placeOf_run`, `withAlg_run`, `RepRun.of_eq`: `placeOf` and `withAlg` keep what `RepRun` reads (the frame's
  configuration, state and coins).
* `repRun_start`, `repRun_first0`, `repRun_first1`, `repRun_later`: the frame `roundPlace` readies keeps `RepRun`: on
  no requests at the rep's start and at rep 0's first round, on rep 0's algebraic-phase coins (`repRun_prior`) at rep
  1's, and on the rep's requests so far at a later round.
* `roundStep_out`, `roundStep_check`: a round that releases is the frame's release of it in its place, at a scheduled
  place its check against the scheduled round, after the stream's domain frame at its first round.
* `repRun_round_coins`: where the frame `roundPlace` readies keeps `RepRun` and only updates that keep the rep's frame
  come between the round's read, its put and its coins' read, the round that releases and its coins leave the frame
  their run.
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock (F128)
open _root_.Flock.Firewall.Frames (Sessions Rep Release)
open _root_.Flock.Firewall.Frame (Cfg Req Place)

theorem run_cons (c : Cfg) (st : _root_.Flock.Firewall.Frame.State) (q : Req) (l : List Req) :
    _root_.Flock.Firewall.Frame.run c st (q :: l) =
      if (_root_.Flock.Firewall.Frame.step c st q).1.ends then ((_root_.Flock.Firewall.Frame.step c st q).2, true)
      else _root_.Flock.Firewall.Frame.run c (_root_.Flock.Firewall.Frame.step c st q).2 l := rfl

/-- **`Frame.run` on one more request**, where the run so far has not ended: the step on it at the run's state. -/
theorem run_snoc (c : Cfg) (q : Req) : ∀ (st : _root_.Flock.Firewall.Frame.State) (l : List Req),
    (_root_.Flock.Firewall.Frame.run c st l).2 = false →
    _root_.Flock.Firewall.Frame.run c st (l ++ [q]) =
      ((_root_.Flock.Firewall.Frame.step c (_root_.Flock.Firewall.Frame.run c st l).1 q).2,
        (_root_.Flock.Firewall.Frame.step c (_root_.Flock.Firewall.Frame.run c st l).1 q).1.ends)
  | st, [], _ => by
    rw [List.nil_append, run_cons]
    show _ = ((_root_.Flock.Firewall.Frame.step c st q).2, (_root_.Flock.Firewall.Frame.step c st q).1.ends)
    cases (_root_.Flock.Firewall.Frame.step c st q).1.ends <;> rfl
  | st, q' :: l, h => by
    rw [run_cons] at h
    rw [List.cons_append, run_cons, run_cons]
    cases he : (_root_.Flock.Firewall.Frame.step c st q').1.ends
    · simp only [he, Bool.false_eq_true, ↓reduceIte] at h ⊢
      exact run_snoc c q (_root_.Flock.Firewall.Frame.step c st q').2 l h
    · revert h
      simp [he]

/-- **The loop's requests for the rounds `ev`** a rep took: each its coin count, the device round the frame took, its
place, and its coins. -/
def evReqs (ev : List (Nat × ByteArray × Place × Array F128)) : List Req :=
  ev.flatMap fun e => [.round e.1 e.2.1 e.2.2.1, .coins (.ok e.2.2.2)]

theorem evReqs_snoc (ev : List (Nat × ByteArray × Place × Array F128)) (e : Nat × ByteArray × Place × Array F128) :
    evReqs (ev ++ [e]) = evReqs ev ++ [.round e.1 e.2.1 e.2.2.1, .coins (.ok e.2.2.2)] := by
  simp [evReqs, List.flatMap_append]

/-- **The rep `p`'s frame is its run on the loop's requests**: `pre` (rep 0's algebraic-phase coins at rep 1, else
none), then the rounds `ev`, whose coins are the rep's, from the frame's start, not ended. -/
def RepRun (pre : List Req) (ev : List (Nat × ByteArray × Place × Array F128)) (p : Rep) : Prop :=
  _root_.Flock.Firewall.Frame.run p.cfg {} (pre ++ evReqs ev) = (p.st, false) ∧ p.coins.toList = ev.map (·.2.2.2)

theorem RepRun.of_eq {pre : List Req} {ev : List (Nat × ByteArray × Place × Array F128)} {p q : Rep}
    (h : RepRun pre ev p) (hc : q.cfg = p.cfg) (hs : q.st = p.st) (hk : q.coins = p.coins) : RepRun pre ev q := by
  obtain ⟨h1, h2⟩ := h
  exact ⟨by rw [hc, hs]; exact h1, by rw [hk]; exact h2⟩

/-- `placeOf` keeps a rep's frame configuration, state and coins. -/
theorem placeOf_run {salts : Array (ByteArray × Nat × Array ByteArray)} {l0 : _root_.Flock.Firewall.Level0.Level0}
    {s : Release} {p : Rep} {dom : ByteArray} {pl : Place} {n : Nat} {q : Rep}
    (h : _root_.Flock.Firewall.Frames.placeOf salts l0 s p dom = .ok (pl, n, q)) :
    q.cfg = p.cfg ∧ q.st = p.st ∧ q.coins = p.coins := by
  unfold _root_.Flock.Firewall.Frames.placeOf at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first
    | (cases h; done)
    | (cases h; exact ⟨rfl, rfl, rfl⟩)

/-- `withAlg` keeps a rep's frame configuration, state and coins. -/
theorem withAlg_run {s : Release} {l0 : _root_.Flock.Firewall.Level0.Level0} {name : String} {p q : Rep}
    (h : _root_.Flock.Firewall.Frames.withAlg s l0 name p = .ok q) :
    q.cfg = p.cfg ∧ q.st = p.st ∧ q.coins = p.coins := by
  unfold _root_.Flock.Firewall.Frames.withAlg at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first
    | (cases h; done)
    | (cases h; exact ⟨rfl, rfl, rfl⟩)

/-- **At a rep's start** its frame is its run on no requests. -/
theorem repRun_start {p : Rep} (hs : p.st = {}) (hk : p.coins = #[]) : RepRun [] [] p :=
  ⟨by rw [hs]; all_goals rfl, by rw [hk]; all_goals rfl⟩

/-- **Rep 0's algebraic-phase coins at rep 1's first round** (`Frame.step`'s `prior`, answered `ok`): the frame after
them is its run on them. -/
theorem repRun_prior {p : Rep} {k : Nat} {b : ByteArray} {st₁ : _root_.Flock.Firewall.Frame.State}
    (h : RepRun [] [] p) (hpr : _root_.Flock.Firewall.Frame.step p.cfg p.st (.prior k b) = (.ok, st₁)) :
    RepRun [.prior k b] [] { p with st := st₁ } := by
  obtain ⟨h1, h2⟩ := h
  refine ⟨?_, h2⟩
  have hst : p.st = {} := (congrArg Prod.fst h1).symm
  have hpr' : _root_.Flock.Firewall.Frame.step p.cfg {} (.prior k b) = (.ok, st₁) := by
    rw [← hst]
    exact hpr
  show _root_.Flock.Firewall.Frame.run p.cfg {} [.prior k b] = (st₁, false)
  rw [run_cons, hpr']
  all_goals rfl

/-- **At a stream's first round at rep 0**, the frame `roundPlace` readies is its run on no requests. -/
theorem repRun_first0 {salts : Array (ByteArray × Nat × Array ByteArray)} {l0 : _root_.Flock.Firewall.Level0.Level0}
    {s : Release} {name : String} {dom : ByteArray} {pl : Place} {n : Nat} {p p' q : Rep} (h : RepRun [] [] p)
    (hw : _root_.Flock.Firewall.Frames.withAlg s l0 name p = .ok p')
    (hp : _root_.Flock.Firewall.Frames.placeOf salts l0 s p' dom = .ok (pl, n, q)) : RepRun [] [] q := by
  obtain ⟨c1, s1, k1⟩ := withAlg_run hw
  obtain ⟨c2, s2, k2⟩ := placeOf_run hp
  exact (h.of_eq c1 s1 k1).of_eq c2 s2 k2

/-- **At a stream's first round at rep 1**, the frame `roundPlace` readies is its run on rep 0's algebraic-phase
coins. -/
theorem repRun_first1 {salts : Array (ByteArray × Nat × Array ByteArray)} {l0 : _root_.Flock.Firewall.Level0.Level0}
    {s : Release} {name : String} {dom : ByteArray} {pl : Place} {n k : Nat} {b : ByteArray}
    {st₁ : _root_.Flock.Firewall.Frame.State} {p p' q : Rep} (h : RepRun [] [] p)
    (hw : _root_.Flock.Firewall.Frames.withAlg s l0 name p = .ok p')
    (hpr : _root_.Flock.Firewall.Frame.step p'.cfg p'.st (.prior k b) = (.ok, st₁))
    (hp : _root_.Flock.Firewall.Frames.placeOf salts l0 s { p' with st := st₁ } dom = .ok (pl, n, q)) :
    RepRun [.prior k b] [] q := by
  obtain ⟨c1, s1, k1⟩ := withAlg_run hw
  obtain ⟨c2, s2, k2⟩ := placeOf_run hp
  exact (repRun_prior (h.of_eq c1 s1 k1) hpr).of_eq c2 s2 k2

/-- **At a later round**, the frame `roundPlace` readies is the rep's run. -/
theorem repRun_later {salts : Array (ByteArray × Nat × Array ByteArray)} {l0 : _root_.Flock.Firewall.Level0.Level0}
    {s : Release} {dom : ByteArray} {pl : Place} {n : Nat} {pre : List Req}
    {ev : List (Nat × ByteArray × Place × Array F128)} {p q : Rep} (h : RepRun pre ev p)
    (hp : _root_.Flock.Firewall.Frames.placeOf salts l0 s p dom = .ok (pl, n, q)) : RepRun pre ev q := by
  obtain ⟨c2, s2, k2⟩ := placeOf_run hp
  exact h.of_eq c2 s2 k2

/-- **A round that releases** is the frame's release of the round in its place (the device's, or none where
`roundPlace` takes none), after the stream's domain frame at its first round. -/
theorem roundStep_out {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i r : Nat}
    {dom content : ByteArray} {n : Nat} {q : Rep} {pl : Place} {f : Bool} {nd : Option String} {o : ByteArray}
    (h : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n = .ok (q, pl, f, nd))
    (ho : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom content n).out = .ok o) :
    ∃ out v, (_root_.Flock.Firewall.Frame.step q.cfg q.st
        (.round n (if nd.isSome then ByteArray.empty else content) pl)).1 = .round out v ∧
      o = if f then dom ++ out else out := by
  unfold _root_.Flock.Firewall.Frames.roundStep at ho
  simp only [h] at ho
  revert ho
  generalize _root_.Flock.Firewall.Frame.step q.cfg q.st
    (.round n (if nd.isSome then ByteArray.empty else content) pl) = x
  obtain ⟨a, st'⟩ := x
  cases a
  all_goals intro ho
  all_goals first
    | (cases ho; done)
    | (cases ho; exact ⟨_, _, rfl, rfl⟩)

/-- **A round released at a scheduled place is the frame's check** of the device round (none where `roundPlace` takes
none) against the scheduled round, at the frame's masking state, after the stream's domain frame at its first round. -/
theorem roundStep_check {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i r : Nat}
    {dom content : ByteArray} {n : Nat} {q : Rep} {rr : ByteArray} {f : Bool} {nd : Option String} {o : ByteArray}
    (h : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n = .ok (q, .sched rr, f, nd))
    (ho : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom content n).out = .ok o) :
    ∃ out s' why, _root_.Flock.Firewall.Frame.check q.cfg q.st.m (if nd.isSome then ByteArray.empty else content) rr n =
        .ok (out, s', why) ∧ o = if f then dom ++ out else out := by
  obtain ⟨out, v, hre, rfl⟩ := roundStep_out h ho
  refine ⟨out, ?_⟩
  simp only [_root_.Flock.Firewall.Frame.step] at hre
  split at hre
  · cases hre
  · split at hre
    · cases hre
      exact ⟨_, _, by assumption, rfl⟩
    · split at hre <;> cases hre

/-- **A round that releases and its coins extend the rep's run**: where the frame `q` that `roundPlace` readies is its
run on the requests `pre` and the rounds `ev` (`RepRun`), the round releases, and only updates that keep the rep's
frame come between the round's read (`R`), its put (on `R₁`) and the coins' read (`R₂`), `Frames.coinsStep` puts back
`Loop.coinsOf` of the frame the round put back, which is its run on `ev` and the round in its place with its coins. -/
theorem repRun_round_coins {R R₁ R₂ : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i r : Nat}
    {dom content : ByteArray} {n : Nat} {q p₁ : Rep} {pl : Place} {f : Bool} {nd : Option String} {o : ByteArray}
    {pre : List Req} {ev : List (Nat × ByteArray × Place × Array F128)} (hR : RepRun pre ev q)
    (hfr : (frameAt R sid i r).isSome)
    (hrp : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n = .ok (q, pl, f, nd))
    (ho : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom content n).out = .ok o)
    (hput : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom content n).put = some p₁)
    (h₀₁ : Kept sid i r R R₁) (h₁₂ : Kept sid i r (R₁.put p₁) R₂) {cs : Array F128}
    {st₂ : _root_.Flock.Firewall.Frame.State}
    (hcs : _root_.Flock.Firewall.Frame.step p₁.cfg p₁.st (.coins (.ok cs)) = (.ok, st₂)) :
    _root_.Flock.Firewall.Frames.coinsStep R₂ sid i r cs = .ok (Loop.coinsOf p₁ st₂ cs) ∧
      RepRun pre (ev ++ [(n, if nd.isSome then ByteArray.empty else content, pl, cs)]) (Loop.coinsOf p₁ st₂ cs) := by
  obtain ⟨out, v', hre, _⟩ := roundStep_out hrp ho
  obtain ⟨⟨v, rfl⟩, hc⟩ := round_coins hfr hrp hput h₀₁ h₁₂ hcs
  obtain ⟨hrun, hco⟩ := hR
  refine ⟨hc, ?_, ?_⟩
  · show _root_.Flock.Firewall.Frame.run q.cfg {} (pre ++ evReqs (ev ++ [_])) = (st₂, false)
    rw [evReqs_snoc, show ∀ (a b : List Req) (x y : Req), a ++ (b ++ [x, y]) = (a ++ b ++ [x]) ++ [y] by simp]
    dsimp only
    have h1 := run_snoc q.cfg (.round n (if nd.isSome then ByteArray.empty else content) pl) {} (pre ++ evReqs ev)
      (by simp only [hrun])
    rw [hrun] at h1
    simp only [hre, _root_.Flock.Firewall.Frame.Reply.ends] at h1
    have hcs' : _root_.Flock.Firewall.Frame.step q.cfg (_root_.Flock.Firewall.Frame.step q.cfg q.st
        (.round n (if nd.isSome then ByteArray.empty else content) pl)).2 (.coins (.ok cs)) = (.ok, st₂) := hcs
    rw [run_snoc q.cfg (.coins (.ok cs)) {}
      (pre ++ evReqs ev ++ [Req.round n (if nd.isSome then ByteArray.empty else content) pl]) (by simp only [h1]), h1]
    simp only [hcs', _root_.Flock.Firewall.Frame.Reply.ends]
  · show (q.coins.push cs).toList = _
    simp only [Array.toList_push, hco, List.map_append, List.map_cons, List.map_nil]

end Claims.ZKServer.Relay
