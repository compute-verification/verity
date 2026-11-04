module

public import Proofs.Flock.ZkAny.RelayStay
public import Proofs.Flock.Claims.ZK.FrameRep

/-!
# A rep's later round in the loop's frames is at `placeOf`'s place

At a rep's round past its stream's first, `Frames.roundPlace` reads the rep's frame, its session's entry (`sessAt`),
its table's statement and name, its level 0 and, through `Frames.placeOf`, the loop's salts, and changes the frame only
as `placeOf` does:

* `roundPlace_later`: where `placeOf` places the round in a scheduled round's place, or fails it, the round's place
  is that, the frame `placeOf`'s, and the frame takes the device round.
* `roundPlace_own`: past the opening (`placeOf`'s `own`), the round's place is the inner proof's round `j` the
  frame's own (`Frame.innerSched` on the rounds the loop's finish computed for the rep, `Sessions.innerOf`), and the
  frame takes the device round exactly where the finish computed round `j`.
* `roundStep_put`: the round puts back `roundPlace`'s frame at the frame's state after the round, with some values.
* `repInv_round_coins`: so where the rep keeps its schedule so far (`Loop.RepInv`) and its round is at a scheduled
  place, and only updates that keep the rep's frame come between the round's read, its put and its coins' read, the
  coins find that frame, `Frames.coinsStep` puts back `Loop.coinsOf` of it, and the rep after keeps `RepInv`.
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock (Stmt F128)
open _root_.Flock.Firewall.Frames (Sessions Rep Release)

/-- **A later round at a scheduled or failed place** is at `placeOf`'s place, on `placeOf`'s frame. -/
theorem roundPlace_later {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i r j : Nat}
    {dom content : ByteArray} {n : Nat} {p : Rep} {s : Release} {st : Stmt} {name : String}
    {l0 : _root_.Flock.Firewall.Level0.Level0} {pl : _root_.Flock.Firewall.Frame.Place} {k : Nat} {q : Rep}
    (hf : R.find sid i r = some (j, p)) (hs : sessAt R sid = some s) (hst : s.sts[i]? = some st)
    (hname : s.spec.tables[i]? = some name)
    (hl0 : _root_.Flock.Firewall.Level0.ofSession L0 sid name = some l0)
    (hfst : (p.coins.isEmpty && !p.st.begun) = false)
    (hp : _root_.Flock.Firewall.Frames.placeOf R.salts l0 s p (s.spec.domain name r).toUTF8 = .ok (pl, k, q))
    (hpl : pl ≠ .own) :
    _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n = .ok (q, pl, false, none) := by
  unfold sessAt at hs
  unfold _root_.Flock.Firewall.Frames.roundPlace
  rcases pl with c | _ | _
  all_goals first
    | exact absurd rfl hpl
    | (simp only [hf, hs, hst, hname, hl0, hfst, hp, bind, Except.bind, pure, Except.pure, Bool.false_eq_true,
        ↓reduceIte]
       all_goals rfl)

/-- **A later round past the opening** is at the inner proof's round `j` the frame's own, on the rounds the loop's
finish computed for the rep, and the frame takes the device round exactly where the finish computed round `j`. -/
theorem roundPlace_own {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i r j : Nat}
    {dom content : ByteArray} {n : Nat} {p : Rep} {s : Release} {st : Stmt} {name : String}
    {l0 : _root_.Flock.Firewall.Level0.Level0} {k : Nat} {q : Rep}
    (hf : R.find sid i r = some (j, p)) (hs : sessAt R sid = some s) (hst : s.sts[i]? = some st)
    (hname : s.spec.tables[i]? = some name)
    (hl0 : _root_.Flock.Firewall.Level0.ofSession L0 sid name = some l0)
    (hfst : (p.coins.isEmpty && !p.st.begun) = false)
    (hp : _root_.Flock.Firewall.Frames.placeOf R.salts l0 s p (s.spec.domain name r).toUTF8 = .ok (.own, k, q))
    (hj : q.st.m.inner.getD 0 < _root_.Flock.Firewall.Frame.INNER_ROUNDS) :
    ∃ nd, _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n =
        .ok (q, .sched (_root_.Flock.Firewall.Frame.innerSched q.cfg q.st.m (R.innerOf l0.root r) n
          (q.st.m.inner.getD 0)), false, nd) ∧
      (nd = none ↔ q.st.m.inner.getD 0 < (R.innerOf l0.root r).size) := by
  have hj' : ¬ _root_.Flock.Firewall.Frame.INNER_ROUNDS ≤ q.st.m.inner.getD 0 := Nat.not_le.mpr hj
  unfold sessAt at hs
  unfold _root_.Flock.Firewall.Frames.roundPlace
  simp only [hf, hs, hst, hname, hl0, hfst, hp, hj', bind, Except.bind, pure, Except.pure, Bool.false_eq_true,
    ↓reduceIte]
  refine ⟨_, rfl, ?_⟩
  split
  · rename_i h
    simp [h]
  · rename_i h
    simp [h]

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

/-- **A rep's later round and its coins keep its schedule so far**: at the rep `p` keeping `Loop.RepInv` on the
table, extras and salts its session and the loop's salts give it, its round at a scheduled place of the schedule so
far, the frame the round puts back (`p₁`) is the one its coins find, where only updates that keep the rep's frame come
between the round's read (`R`), its put (on `R₁`) and the coins' read (`R₂`); `Frames.coinsStep` puts back
`Loop.coinsOf` of it, and that keeps `RepInv`. -/
theorem repInv_round_coins {R R₁ R₂ : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i r j : Nat}
    {dom content : ByteArray} {n : Nat} {p : Rep} {s : Release} {st : Stmt} {name : String}
    {l0 : _root_.Flock.Firewall.Level0.Level0} {t : _root_.Flock.Firewall.Finish.Tab} {xs : Array (Array F128)}
    {hi : Array ByteArray} {c : Array F128} {r0 : ByteArray} {q0 : Nat} {sched : Array (ByteArray × Nat)}
    (hf : R.find sid i r = some (j, p)) (hs : sessAt R sid = some s) (hst : s.sts[i]? = some st)
    (hname : s.spec.tables[i]? = some name)
    (hl0 : _root_.Flock.Firewall.Level0.ofSession L0 sid name = some l0)
    (hfst : (p.coins.isEmpty && !p.st.begun) = false) (hlink : s.link = some c)
    (hsalt : R.salts.find? (fun x => x.1 == l0.root && x.2.1 == p.rep) = some (r0, q0, hi))
    (htab : _root_.Flock.Firewall.Finish.tabOfLevel0 (_root_.Flock.Setup.ofCircuit st) l0 = .ok t)
    (hxs : xs = _root_.Flock.Firewall.Drive.extras st (_root_.Flock.Firewall.Drive.linkPoints st c))
    (hI : Loop.RepInv t (s.spec.domain name r).toUTF8 xs hi p)
    (hsch : Loop.schedSoFar t p.rep (s.spec.domain name r).toUTF8 xs hi p.coins = .ok sched)
    (hpl : (Loop.placeAt sched p.coins.size).1 ≠ .own)
    {p₁ : Rep} (hput : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom content n).put = some p₁)
    (h₀₁ : Kept sid i r R R₁) (h₁₂ : Kept sid i r (R₁.put p₁) R₂) {cs : Array F128}
    {st₂ : _root_.Flock.Firewall.Frame.State}
    (hcs : _root_.Flock.Firewall.Frame.step p₁.cfg p₁.st (.coins (.ok cs)) = (.ok, st₂)) :
    _root_.Flock.Firewall.Frames.coinsStep R₂ sid i r cs = .ok (Loop.coinsOf p₁ st₂ cs) ∧
      Loop.RepInv t (s.spec.domain name r).toUTF8 xs hi (Loop.coinsOf p₁ st₂ cs) := by
  have hk : keyOf p = (sid, i, r) := find_key hf
  have htable : p.table = i := by
    simp only [keyOf, Prod.mk.injEq] at hk
    exact hk.2.1
  have hst' : s.sts[p.table]? = some st := by rw [htable]; exact hst
  obtain ⟨q, hq, hinv⟩ :=
    Loop.placeOf_sched R.salts l0 s p (s.spec.domain name r).toUTF8 hst' hlink hsalt htab hxs hI hsch
  have hrp := roundPlace_later (dom := dom) (content := content) (n := n) hf hs hst hname hl0 hfst hq hpl
  rcases hres : _root_.Flock.Firewall.Frame.step q.cfg q.st
    (.round n (if (none : Option String).isSome then ByteArray.empty else content)
      (Loop.placeAt sched p.coins.size).1) with ⟨a, st'⟩
  obtain ⟨v, hv⟩ := roundStep_put hrp hres
  rw [hv, Option.some.injEq] at hput
  subst hput
  have hk₁ : keyOf { q with st := st', vals := v } = (sid, i, r) := (placeOf_key hq).trans hk
  have hfr : (frameAt R sid i r).isSome := by
    rw [← find_frameAt, hf]
    rfl
  have h2 := find_own hfr hk₁ h₀₁ h₁₂
  rcases hx : R₂.find sid i r with _ | ⟨j₂, p₂⟩
  · rw [hx] at h2
    cases h2
  · rw [hx] at h2
    cases h2
    exact ⟨Loop.coinsStep_ok R₂ sid i r cs hx hcs, hinv st' v cs st₂⟩

end Claims.ZKServer.Relay
