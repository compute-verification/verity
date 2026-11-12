module

public import Proofs.Flock.ZkAny.RelayStay
public import Proofs.Flock.Claims.ZK.FrameRep

/-!
# A rep's round in the loop's frames is at `placeOf`'s place

At a rep's round, `Frames.roundPlace` reads the rep's frame, its session's entry (`sessAt`), its table's statement and
name, its level 0 and, through `Frames.placeOf`, the loop's salts, and changes the frame only as `placeOf` does, after,
at the stream's first round, `Frames.withAlg` and rep 1's `prior`:

* `roundPlace_later`: past the stream's first round, where `placeOf` places the round in a scheduled round's place, or
  fails it, the round's place is that, the frame `placeOf`'s, and the frame takes the device round exactly where the
  session has not stopped (`Frames.Sessions.stopped`: no rep's frame of it stopped).
* `roundPlace_own`: past the opening (`placeOf`'s `own`), once the loop's finish computed the inner proof's round `j`
  for the rep, the round's place is round `j` the frame's own (`Frame.innerSched` on the rounds the finish computed,
  `Sessions.innerOf`), and the frame takes the device round exactly where the session has not stopped. A round `j`
  the finish has not computed has no place: the loop refuses it.
* `roundPlace_first0`, `roundPlace_first1`: at the stream's first round, the place is `placeOf`'s on the frame with its
  algebraic phase, at rep 1 after rep 0's algebraic-phase coins, and the frame takes the device round exactly where
  the session has not stopped.
* `roundStep_put`: the round puts back `roundPlace`'s frame at the frame's state after the round, with some values.
* `round_coins`: where only updates that keep the rep's frame come between the round's read, its put and its coins'
  read, the coins find that frame and `Frames.coinsStep` puts back `Loop.coinsOf` of it.
* `placeOf_sched_alg`: `Loop.placeOf_sched` in the algebraic phase, which reads no salts.
* `repInv_round_coins`: so where the rep keeps its schedule so far (`Loop.RepInv`) and its later round is at a
  scheduled place, the rep after its coins keeps `RepInv`.
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock (Stmt F128)
open _root_.Flock.Firewall.Frames (Sessions Rep Release)

/-- **A later round at a scheduled or failed place** is at `placeOf`'s place, on `placeOf`'s frame, and the frame
takes the device round exactly where the session has not stopped. -/
theorem roundPlace_later {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i r j : Nat}
    {dom content : ByteArray} {n : Nat} {p : Rep} {s : Release} {st : Stmt} {name : String}
    {l0 : _root_.Flock.Firewall.Level0.Level0} {pl : _root_.Flock.Firewall.Frame.Place} {k : Nat} {q : Rep}
    (hf : R.find sid i r = some (j, p)) (hs : sessAt R sid = some s) (hst : s.sts[i]? = some st)
    (hname : s.spec.tables[i]? = some name)
    (hl0 : _root_.Flock.Firewall.Level0.ofSession L0 sid name = some l0)
    (hfst : (p.coins.isEmpty && !p.st.begun) = false)
    (hp : _root_.Flock.Firewall.Frames.placeOf R.salts l0 s p (s.spec.domain name r).toUTF8 = .ok (pl, k, q))
    (hpl : pl ≠ .own) :
    ∃ nd, _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n = .ok (q, pl, false, nd) ∧
      (nd = none ↔ R.stopped sid = false) := by
  unfold sessAt at hs
  unfold _root_.Flock.Firewall.Frames.roundPlace
  rcases pl with c | _ | _
  all_goals first
    | exact absurd rfl hpl
    | (cases hss : R.stopped sid
       all_goals
         simp only [hf, hs, hst, hname, hl0, hfst, hp, bind, Except.bind, pure, Except.pure, Bool.false_eq_true,
           ↓reduceIte]
         exact ⟨_, rfl, by simp⟩)

/-- **A later round past the opening** is, once the loop's finish computed the inner proof's round `j` for the rep,
at round `j` the frame's own, on the rounds the finish computed, and the frame takes the device round exactly where
the session has not stopped. -/
theorem roundPlace_own {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i r j : Nat}
    {dom content : ByteArray} {n : Nat} {p : Rep} {s : Release} {st : Stmt} {name : String}
    {l0 : _root_.Flock.Firewall.Level0.Level0} {k : Nat} {q : Rep}
    (hf : R.find sid i r = some (j, p)) (hs : sessAt R sid = some s) (hst : s.sts[i]? = some st)
    (hname : s.spec.tables[i]? = some name)
    (hl0 : _root_.Flock.Firewall.Level0.ofSession L0 sid name = some l0)
    (hfst : (p.coins.isEmpty && !p.st.begun) = false)
    (hp : _root_.Flock.Firewall.Frames.placeOf R.salts l0 s p (s.spec.domain name r).toUTF8 = .ok (.own, k, q))
    (hj : q.st.m.inner.getD 0 < _root_.Flock.Firewall.Frame.INNER_ROUNDS)
    (hfin : q.st.m.inner.getD 0 < (R.innerOf l0.root r).size) :
    ∃ nd, _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n =
        .ok (q, .sched (_root_.Flock.Firewall.Frame.innerSched q.cfg q.st.m (R.innerOf l0.root r) n
          (q.st.m.inner.getD 0)), false, nd) ∧
      (nd = none ↔ R.stopped sid = false) := by
  have hj' : ¬ _root_.Flock.Firewall.Frame.INNER_ROUNDS ≤ q.st.m.inner.getD 0 := Nat.not_le.mpr hj
  have hfin' : ¬ (R.innerOf l0.root r).size ≤ q.st.m.inner.getD 0 := Nat.not_le.mpr hfin
  unfold sessAt at hs
  unfold _root_.Flock.Firewall.Frames.roundPlace
  simp only [hf, hs, hst, hname, hl0, hfst, hp, hj', hfin', bind, Except.bind, pure, Except.pure, Bool.false_eq_true,
    ↓reduceIte]
  refine ⟨_, rfl, ?_⟩
  cases hss : R.stopped sid <;> simp

/-- **A round puts back `roundPlace`'s frame**, its frame state the frame's after the round in its place (the device's
round, or none where `roundPlace` takes none), with some values. -/
theorem roundStep_put {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i r : Nat}
    {dom content : ByteArray} {n : Nat} {q : Rep} {pl : _root_.Flock.Firewall.Frame.Place} {f : Bool}
    {nd : Option String} {a : _root_.Flock.Firewall.Frame.Reply} {st' : _root_.Flock.Firewall.Frame.State}
    (h : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n = .ok (q, pl, f, nd))
    (hs : _root_.Flock.Firewall.Frame.step q.cfg q.st
      (.round n (if nd.isSome then ByteArray.empty else content) pl) = (a, st')) :
    ∃ v, (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom content n).put =
      some { q with st := st', vals := v } := by
  unfold _root_.Flock.Firewall.Frames.roundStep
  simp only [h, hs]
  cases a <;> exact ⟨_, rfl⟩

/-- **A round's coins find the frame the round put back**: at a round at `roundPlace`'s place on the frame `q`, the
frame the round puts back (`p₁`) is `q` at the frame's state after the round, with some values, and where only updates
that keep the rep's frame come between the round's read (`R`), its put (on `R₁`) and the coins' read (`R₂`),
`Frames.coinsStep` finds it and puts back `Loop.coinsOf` of it. -/
theorem round_coins {R R₁ R₂ : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i r : Nat}
    {dom content : ByteArray} {n : Nat} {q p₁ : Rep} {pl : _root_.Flock.Firewall.Frame.Place} {f : Bool}
    {nd : Option String} (hfr : (frameAt R sid i r).isSome)
    (hrp : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n = .ok (q, pl, f, nd))
    (hput : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom content n).put = some p₁)
    (h₀₁ : Kept sid i r R R₁) (h₁₂ : Kept sid i r (R₁.put p₁) R₂) {cs : Array F128}
    {st₂ : _root_.Flock.Firewall.Frame.State}
    (hcs : _root_.Flock.Firewall.Frame.step p₁.cfg p₁.st (.coins (.ok cs)) = (.ok, st₂)) :
    (∃ v, p₁ = { q with st := (_root_.Flock.Firewall.Frame.step q.cfg q.st
      (.round n (if nd.isSome then ByteArray.empty else content) pl)).2, vals := v }) ∧
      _root_.Flock.Firewall.Frames.coinsStep R₂ sid i r cs = .ok (Loop.coinsOf p₁ st₂ cs) := by
  rcases hres : _root_.Flock.Firewall.Frame.step q.cfg q.st
    (.round n (if nd.isSome then ByteArray.empty else content) pl) with ⟨a, st'⟩
  obtain ⟨v, hv⟩ := roundStep_put hrp hres
  rw [hv, Option.some.injEq] at hput
  subst hput
  refine ⟨⟨v, rfl⟩, ?_⟩
  have hk₀ : keyOf q = (sid, i, r) := roundPlace_key hrp
  have hk₁ : keyOf { q with st := st', vals := v } = (sid, i, r) := hk₀
  have h2 := find_own hfr hk₁ h₀₁ h₁₂
  rcases hx : R₂.find sid i r with _ | ⟨j₂, p₂⟩
  · rw [hx] at h2
    cases h2
  · rw [hx] at h2
    cases h2
    exact Loop.coinsStep_ok R₂ sid i r cs hx hcs

/-- **`Loop.placeOf_sched` in the algebraic phase**, where `placeOf` reads no salts: at the rep `p` keeping
`Loop.RepInv` before its algebraic phase's coins are all in, `placeOf` places its next round at the algebraic phase's
scheduled round, and the rep after that round's coins keeps `RepInv`. -/
theorem placeOf_sched_alg (salts : Array (ByteArray × Nat × Array ByteArray))
    (l0 : _root_.Flock.Firewall.Level0.Level0) (s : Release) (p : Rep) (dom : ByteArray)
    {t : _root_.Flock.Firewall.Finish.Tab} {xs : Array (Array F128)} {hi : Array ByteArray}
    (hI : Loop.RepInv t dom xs hi p) (hlt : p.coins.size < p.alg.size) :
    ∃ q, _root_.Flock.Firewall.Frames.placeOf salts l0 s p dom =
        .ok ((Loop.placeAt p.alg p.coins.size).1, (Loop.placeAt p.alg p.coins.size).2, q) ∧
      ∀ st' v cs st'', Loop.RepInv t dom xs hi (Loop.coinsOf { q with st := st', vals := v } st'' cs) := by
  obtain ⟨hal, hto, hop⟩ := hI
  rcases hop with ⟨hn, ho, _⟩ | ⟨hle, _⟩ | ⟨hle, _⟩
  · refine ⟨p, ?_, fun st' v cs st'' => Loop.repInv_coinsOf_alg t dom xs hi _ st'' cs hal hlt hto hn ho⟩
    rw [Loop.placeOf_alg salts l0 s p dom hlt]
    unfold Loop.placeAt
    rw [Array.getElem?_eq_getElem hlt]
  · omega
  · omega

/-- **A rep's later round and its coins keep its schedule so far**: where `placeOf` places the rep `p`'s round at a
scheduled place of its schedule so far `sched` and the rep after the round's coins keeps `Loop.RepInv`
(`Loop.placeOf_sched`, `placeOf_sched_alg`), the frame the round puts back (`p₁`) is the one its coins find, where only
updates that keep the rep's frame come between the round's read (`R`), its put (on `R₁`) and the coins' read (`R₂`);
`Frames.coinsStep` puts back `Loop.coinsOf` of it, and that keeps `RepInv`. -/
theorem repInv_round_coins {R R₁ R₂ : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i r j : Nat}
    {dom content : ByteArray} {n : Nat} {p : Rep} {s : Release} {st : Stmt} {name : String}
    {l0 : _root_.Flock.Firewall.Level0.Level0} {t : _root_.Flock.Firewall.Finish.Tab} {xs : Array (Array F128)}
    {hi : Array ByteArray} {sched : Array (ByteArray × Nat)}
    (hf : R.find sid i r = some (j, p)) (hs : sessAt R sid = some s) (hst : s.sts[i]? = some st)
    (hname : s.spec.tables[i]? = some name)
    (hl0 : _root_.Flock.Firewall.Level0.ofSession L0 sid name = some l0)
    (hfst : (p.coins.isEmpty && !p.st.begun) = false)
    (hq : ∃ q, _root_.Flock.Firewall.Frames.placeOf R.salts l0 s p (s.spec.domain name r).toUTF8 =
        .ok ((Loop.placeAt sched p.coins.size).1, (Loop.placeAt sched p.coins.size).2, q) ∧
      ∀ st' v cs st'', Loop.RepInv t (s.spec.domain name r).toUTF8 xs hi
        (Loop.coinsOf { q with st := st', vals := v } st'' cs))
    (hpl : (Loop.placeAt sched p.coins.size).1 ≠ .own)
    {p₁ : Rep} (hput : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom content n).put = some p₁)
    (h₀₁ : Kept sid i r R R₁) (h₁₂ : Kept sid i r (R₁.put p₁) R₂) {cs : Array F128}
    {st₂ : _root_.Flock.Firewall.Frame.State}
    (hcs : _root_.Flock.Firewall.Frame.step p₁.cfg p₁.st (.coins (.ok cs)) = (.ok, st₂)) :
    _root_.Flock.Firewall.Frames.coinsStep R₂ sid i r cs = .ok (Loop.coinsOf p₁ st₂ cs) ∧
      Loop.RepInv t (s.spec.domain name r).toUTF8 xs hi (Loop.coinsOf p₁ st₂ cs) := by
  obtain ⟨q, hq, hinv⟩ := hq
  obtain ⟨_, hrp, -⟩ := roundPlace_later (dom := dom) (content := content) (n := n) hf hs hst hname hl0 hfst hq hpl
  have hfr : (frameAt R sid i r).isSome := by
    rw [← find_frameAt, hf]
    rfl
  obtain ⟨⟨v, rfl⟩, hc⟩ := round_coins hfr hrp hput h₀₁ h₁₂ hcs
  exact ⟨hc, hinv _ v cs st₂⟩

/-- **A stream's first round at rep 0** is at `placeOf`'s place on the frame `withAlg` gives its algebraic phase, where
the round opens with the stream's domain frame, and the frame takes the device round exactly where the session has not
stopped. -/
theorem roundPlace_first0 {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i j : Nat}
    {dom content : ByteArray} {n : Nat} {p p' : Rep} {s : Release} {st : Stmt} {name : String}
    {l0 : _root_.Flock.Firewall.Level0.Level0} {pl : _root_.Flock.Firewall.Frame.Place} {k : Nat} {q : Rep}
    (hf : R.find sid i 0 = some (j, p)) (hs : sessAt R sid = some s) (hst : s.sts[i]? = some st)
    (hname : s.spec.tables[i]? = some name)
    (hl0 : _root_.Flock.Firewall.Level0.ofSession L0 sid name = some l0)
    (hfst : (p.coins.isEmpty && !p.st.begun) = true)
    (hdom : ((content.extract 0 dom.size).data != dom.data) = false)
    (hw : _root_.Flock.Firewall.Frames.withAlg s l0 name p = .ok p')
    (hp : _root_.Flock.Firewall.Frames.placeOf R.salts l0 s p' (s.spec.domain name 0).toUTF8 = .ok (pl, k, q))
    (hpl : pl ≠ .own) :
    ∃ nd, _root_.Flock.Firewall.Frames.roundPlace R L0 sid i 0 dom content n = .ok (q, pl, true, nd) ∧
      (nd = none ↔ R.stopped sid = false) := by
  have h01 : ((0 : Nat) == 1) = false := rfl
  unfold sessAt at hs
  unfold _root_.Flock.Firewall.Frames.roundPlace
  rcases pl with c | _ | _
  all_goals first
    | exact absurd rfl hpl
    | (cases hss : R.stopped sid
       all_goals
         simp only [hf, hs, hst, hname, hl0, hfst, hdom, hw, h01, hp, bind, Except.bind, pure, Except.pure,
           Bool.false_eq_true, ↓reduceIte]
         exact ⟨_, rfl, by simp⟩)

/-- **A stream's first round at rep 1** is at `placeOf`'s place on the frame `withAlg` gives its algebraic phase, after
rep 0's algebraic-phase coins (`Frame.step`'s `prior`), where the round opens with the stream's domain frame and rep 0's
algebraic phase ended, and the frame takes the device round exactly where the session has not stopped. -/
theorem roundPlace_first1 {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i j j0 : Nat}
    {dom content : ByteArray} {n : Nat} {p p' p0 : Rep} {s : Release} {st : Stmt} {name : String}
    {l0 : _root_.Flock.Firewall.Level0.Level0} {st₁ : _root_.Flock.Firewall.Frame.State}
    {pl : _root_.Flock.Firewall.Frame.Place} {k : Nat} {q : Rep}
    (hf : R.find sid i 1 = some (j, p)) (hs : sessAt R sid = some s) (hst : s.sts[i]? = some st)
    (hname : s.spec.tables[i]? = some name)
    (hl0 : _root_.Flock.Firewall.Level0.ofSession L0 sid name = some l0)
    (hfst : (p.coins.isEmpty && !p.st.begun) = true)
    (hdom : ((content.extract 0 dom.size).data != dom.data) = false)
    (hw : _root_.Flock.Firewall.Frames.withAlg s l0 name p = .ok p')
    (h0 : R.find sid i 0 = some (j0, p0)) (hle : ¬ p0.coins.size < p'.alg.size)
    (hpr : _root_.Flock.Firewall.Frame.step p'.cfg p'.st
      (.prior ((p0.coins.extract 0 p'.alg.size).foldl (· ++ ·) #[]).size
        (_root_.Flock.Firewall.Frame.wordBytes ((p0.coins.extract 0 p'.alg.size).foldl (· ++ ·) #[]))) = (.ok, st₁))
    (hp : _root_.Flock.Firewall.Frames.placeOf R.salts l0 s { p' with st := st₁ } (s.spec.domain name 1).toUTF8 =
      .ok (pl, k, q))
    (hpl : pl ≠ .own) :
    ∃ nd, _root_.Flock.Firewall.Frames.roundPlace R L0 sid i 1 dom content n = .ok (q, pl, true, nd) ∧
      (nd = none ↔ R.stopped sid = false) := by
  have h11 : ((1 : Nat) == 1) = true := rfl
  unfold sessAt at hs
  unfold _root_.Flock.Firewall.Frames.roundPlace
  rcases pl with c | _ | _
  all_goals first
    | exact absurd rfl hpl
    | (cases hss : R.stopped sid
       all_goals
         simp only [hf, hs, hst, hname, hl0, hfst, hdom, hw, h11, h0, hle, hpr, hp, bind, Except.bind, pure,
           Except.pure, Bool.false_eq_true, ↓reduceIte]
         exact ⟨_, rfl, by simp⟩)

end Claims.ZKServer.Relay
