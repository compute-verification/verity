import Proofs.Flock.Claims.ZK.FrameSched
import Flock.Firewall.Frames

/-!
# The loop places a rep's rounds as `Drive.reqs` does

`Frames.placeOf` places a rep's next round on what the loop kept of the rep: its algebraic phase's scheduled rounds,
then its opening (`Drive.openStep`), begun at the first round past them and fed each round's coins (`Frames.coinsStep`).
Where what it kept is the schedule so far on the rep's coins (`RepInv`), the round's place is the schedule's
(`placeAt`: its round there, the inner proof's own past it), as `Drive.reqs` places it, and the rep's frame after the
round's coins keeps the invariant:

* `placeOf_sched`: at a rep keeping `RepInv`, `placeOf` places its next round at `placeAt` of the schedule so far
  (`schedSoFar`) on its coins, and the rep after that round's coins (`coinsOf`) keeps `RepInv`.
* `coinsStep_ok`: what `Frames.coinsStep` puts back is `coinsOf`.
-/

namespace Claims.ZKServer

open _root_.Flock (V F128 Stmt)
open _root_.Flock.Firewall.Finish (Tab flat)

namespace Loop

/-- What the schedule `sched` places at round `k`: its round there and the coins it squeezes, else the inner proof's
own round (and no coins), as `Drive.reqs` places them. -/
def placeAt (sched : Array (ByteArray × ℕ)) (k : ℕ) : _root_.Flock.Firewall.Frame.Place × ℕ :=
  match sched[k]? with
  | some (c, n) => (.sched c, n)
  | none => (.own, 0)

/-- **The rep `p`'s opening fed its coins past the algebraic phase**, at the table `t`, its stream's domain `dom`, the
extra claims `xs` and the salts for levels ≥ 1 `hi`: what it goes on as, and the rounds it sent. -/
def openFed (t : Tab) (dom : ByteArray) (xs : Array (Array F128)) (hi : Array ByteArray)
    (p : _root_.Flock.Firewall.Frames.Rep) :
    _root_.Flock.Firewall.Drive.Step Unit × Array (ByteArray × ℕ) :=
  (_root_.Flock.Firewall.Drive.openStep t p.rep dom xs hi (p.coins.extract 0 p.alg.size)).feed
    (p.coins.extract p.alg.size p.coins.size).toList #[]

/-- **The loop keeps the rep `p`'s schedule so far**, at its next round: its algebraic phase's rounds are
`Skeleton.rounds`', and its opening is not begun while their coins are not all in, then is `openFed`, until it ended
(`opened`). -/
structure RepInv (t : Tab) (dom : ByteArray) (xs : Array (Array F128)) (hi : Array ByteArray)
    (p : _root_.Flock.Firewall.Frames.Rep) : Prop where
  alg : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (flat t.cm0.cap) dom = .ok p.alg
  toOpening : p.toOpening = false
  opening : (p.opening = none ∧ p.opened = false ∧ p.coins.size ≤ p.alg.size) ∨
    (p.alg.size ≤ p.coins.size ∧ p.opened = false ∧ p.opening = some (openFed t dom xs hi p).1) ∨
    (p.alg.size ≤ p.coins.size ∧ p.opened = true ∧ p.opening = none ∧
      ∃ x, (openFed t dom xs hi p).1 = .done x)

/-- **The rep `p` after the coins `cs`**, its frame's state then `st'`: as `Frames.coinsStep` puts it back. -/
def coinsOf (p : _root_.Flock.Firewall.Frames.Rep) (st' : _root_.Flock.Firewall.Frame.State) (cs : Array F128) :
    _root_.Flock.Firewall.Frames.Rep :=
  let opening := match p.toOpening, p.opening with
    | true, some (.ask _ _ k) => some (k cs)
    | _, o => o
  { p with st := st', coins := p.coins.push cs, toOpening := false, opening }

theorem coinsStep_ok (R : _root_.Flock.Firewall.Frames.Sessions) (sid i r : ℕ) (cs : Array F128) {j : ℕ}
    {p : _root_.Flock.Firewall.Frames.Rep} {st' : _root_.Flock.Firewall.Frame.State} (hf : R.find sid i r = some (j, p))
    (hs : _root_.Flock.Firewall.Frame.step p.cfg p.st (.coins (.ok cs)) = (.ok, st')) :
    _root_.Flock.Firewall.Frames.coinsStep R sid i r cs = .ok (coinsOf p st' cs) := by
  unfold _root_.Flock.Firewall.Frames.coinsStep
  simp only [hf, hs, pure, Except.pure]
  rfl

/-- A step fed coins sent at most a round for each. -/
theorem feed_size_le {α : Type} : ∀ (s : _root_.Flock.Firewall.Drive.Step α) (l : List (Array F128))
    (rs : Array (ByteArray × ℕ)), (s.feed l rs).2.size ≤ rs.size + l.length
  | .ask c n k, a :: l, rs => by
    have := feed_size_le (k a) l (rs.push (c, n))
    show ((k a).feed l (rs.push (c, n))).2.size ≤ _
    rw [Array.size_push] at this
    rw [List.length_cons]
    omega
  | .ask _ _ _, [], rs => Nat.le_add_right _ _
  | .done _, _, rs => Nat.le_add_right _ _
  | .fail _, _, rs => Nat.le_add_right _ _

theorem placeOf_alg (salts : Array (ByteArray × ℕ × Array ByteArray)) (l0 : _root_.Flock.Firewall.Level0.Level0)
    (s : _root_.Flock.Firewall.Frames.Session) (p : _root_.Flock.Firewall.Frames.Rep) (dom : ByteArray)
    (h : p.coins.size < p.alg.size) :
    _root_.Flock.Firewall.Frames.placeOf salts l0 s p dom =
      .ok (.sched (p.alg[p.coins.size]'h).1, (p.alg[p.coins.size]'h).2, p) := by
  unfold _root_.Flock.Firewall.Frames.placeOf
  simp only [dif_pos h]
  rfl

theorem placeOf_opened (salts : Array (ByteArray × ℕ × Array ByteArray)) (l0 : _root_.Flock.Firewall.Level0.Level0)
    (s : _root_.Flock.Firewall.Frames.Session) (p : _root_.Flock.Firewall.Frames.Rep) (dom : ByteArray)
    (h : ¬ p.coins.size < p.alg.size) (ho : p.opened = true) :
    _root_.Flock.Firewall.Frames.placeOf salts l0 s p dom = .ok (.own, 0, p) := by
  unfold _root_.Flock.Firewall.Frames.placeOf
  simp only [dif_neg h, ho]
  rfl

/-- What the opening at `S` places, and the rep `q` after: its round, asked for; the inner proof's, once it ended. -/
def stepPlace (S : _root_.Flock.Firewall.Drive.Step Unit) (q : _root_.Flock.Firewall.Frames.Rep) :
    Except String (_root_.Flock.Firewall.Frame.Place × ℕ × _root_.Flock.Firewall.Frames.Rep) :=
  match S with
  | .ask c n _ => .ok (.sched c, n, { q with toOpening := true })
  | .done _ => .ok (.own, 0, { q with opened := true, opening := none })
  | .fail e => .error e

theorem placeOf_some (salts : Array (ByteArray × ℕ × Array ByteArray)) (l0 : _root_.Flock.Firewall.Level0.Level0)
    (s : _root_.Flock.Firewall.Frames.Session) (p : _root_.Flock.Firewall.Frames.Rep) (dom : ByteArray)
    (h : ¬ p.coins.size < p.alg.size) (ho : p.opened = false) {S : _root_.Flock.Firewall.Drive.Step Unit}
    (hS : p.opening = some S) :
    _root_.Flock.Firewall.Frames.placeOf salts l0 s p dom = stepPlace S { p with opening := some S } := by
  unfold _root_.Flock.Firewall.Frames.placeOf
  simp only [dif_neg h, ho, hS, Option.isNone_some, Bool.false_eq_true, ↓reduceIte]
  cases S <;> rfl

theorem placeOf_none (salts : Array (ByteArray × ℕ × Array ByteArray)) (l0 : _root_.Flock.Firewall.Level0.Level0)
    (s : _root_.Flock.Firewall.Frames.Session) (p : _root_.Flock.Firewall.Frames.Rep) (dom : ByteArray)
    (h : ¬ p.coins.size < p.alg.size) (ho : p.opened = false) (hS : p.opening = none) {st : Stmt}
    {c : Array F128} {r0 : ByteArray} {q0 : ℕ} {hi : Array ByteArray} {t : Tab}
    (hst : s.sts[p.table]? = some st) (hlink : s.link = some c)
    (hsalt : salts.find? (fun x => x.1 == l0.root && x.2.1 == p.rep) = some (r0, q0, hi))
    (htab : _root_.Flock.Firewall.Finish.tabOfLevel0 (_root_.Flock.Setup.ofCircuit st) l0 = .ok t) :
    _root_.Flock.Firewall.Frames.placeOf salts l0 s p dom =
      let S := _root_.Flock.Firewall.Drive.openStep t p.rep dom
        (_root_.Flock.Firewall.Drive.extras st (_root_.Flock.Firewall.Drive.linkPoints st c)) hi
        (p.coins.extract 0 p.alg.size)
      stepPlace S { p with opening := some S } := by
  unfold _root_.Flock.Firewall.Frames.placeOf
  simp only [dif_neg h, ho, hS, Option.isNone_none, hst, hlink, hsalt, htab, Bool.false_eq_true, ↓reduceIte, bind,
    Except.bind]
  generalize _root_.Flock.Firewall.Drive.openStep t p.rep dom
    (_root_.Flock.Firewall.Drive.extras st (_root_.Flock.Firewall.Drive.linkPoints st c)) hi
    (p.coins.extract 0 p.alg.size) = S
  cases S <;> rfl

theorem schedSoFar_alg (t : Tab) (dom : ByteArray) (xs : Array (Array F128)) (hi : Array ByteArray)
    (p : _root_.Flock.Firewall.Frames.Rep)
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (flat t.cm0.cap) dom = .ok p.alg)
    (h : p.coins.size < p.alg.size) : schedSoFar t p.rep dom xs hi p.coins = .ok p.alg := by
  unfold schedSoFar
  rw [hal]
  simp only [bind, Except.bind, h, ↓reduceIte, pure, Except.pure]

theorem schedSoFar_open (t : Tab) (dom : ByteArray) (xs : Array (Array F128)) (hi : Array ByteArray)
    (p : _root_.Flock.Firewall.Frames.Rep)
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (flat t.cm0.cap) dom = .ok p.alg)
    (h : ¬ p.coins.size < p.alg.size) :
    schedSoFar t p.rep dom xs hi p.coins = match openFed t dom xs hi p with
      | (.ask c n _, rs) => .ok (p.alg ++ rs.push (c, n))
      | (.done _, rs) => .ok (p.alg ++ rs)
      | (.fail e, _) => .error e := by
  unfold schedSoFar
  rw [hal]
  simp only [bind, Except.bind, h, ↓reduceIte]
  rfl

/-- **The opening on one more round's coins** is the opening so far fed them. -/
theorem openFed_push (t : Tab) (dom : ByteArray) (xs : Array (Array F128)) (hi : Array ByteArray)
    (p q : _root_.Flock.Firewall.Frames.Rep) (cs : Array F128) (hle : p.alg.size ≤ p.coins.size)
    (hq : q.rep = p.rep) (ha : q.alg = p.alg) (hc : q.coins = p.coins.push cs) :
    openFed t dom xs hi q = (openFed t dom xs hi p).1.feed [cs] (openFed t dom xs hi p).2 := by
  unfold openFed
  rw [hq, ha, hc]
  have e1 : (p.coins.push cs).extract 0 p.alg.size = p.coins.extract 0 p.alg.size := by
    rw [Array.extract_push, if_pos hle]
  have e2 : ((p.coins.push cs).extract p.alg.size (p.coins.push cs).size).toList =
      (p.coins.extract p.alg.size p.coins.size).toList ++ [cs] := by
    rw [Array.size_push, Array.extract_push, if_neg (by omega), if_pos hle, Array.toList_push]
  rw [e1, e2, feed_append]

theorem openFed_size (t : Tab) (dom : ByteArray) (xs : Array (Array F128)) (hi : Array ByteArray)
    (p : _root_.Flock.Firewall.Frames.Rep) : (openFed t dom xs hi p).2.size ≤ p.coins.size - p.alg.size := by
  have := feed_size_le (_root_.Flock.Firewall.Drive.openStep t p.rep dom xs hi (p.coins.extract 0 p.alg.size))
    (p.coins.extract p.alg.size p.coins.size).toList #[]
  rw [Array.length_toList, Array.size_extract, Array.size_empty] at this
  unfold openFed
  omega

theorem openFed_ask_size (t : Tab) (dom : ByteArray) (xs : Array (Array F128)) (hi : Array ByteArray)
    (p : _root_.Flock.Firewall.Frames.Rep) {c : ByteArray} {n : ℕ} {k} {rs : Array (ByteArray × ℕ)}
    (h : openFed t dom xs hi p = (.ask c n k, rs)) : rs.size = p.coins.size - p.alg.size := by
  unfold openFed at h
  rw [feed_ask_size _ _ _ h, Array.length_toList, Array.size_extract, Array.size_empty]
  omega

theorem placeAt_none {sched : Array (ByteArray × ℕ)} {k : ℕ} (h : sched.size ≤ k) : placeAt sched k = (.own, 0) := by
  unfold placeAt
  rw [Array.getElem?_eq_none h]

theorem placeAt_ask (alg rs : Array (ByteArray × ℕ)) (c : ByteArray) (n k : ℕ) (hk : k = alg.size + rs.size) :
    placeAt (alg ++ rs.push (c, n)) k = (.sched c, n) := by
  unfold placeAt
  rw [Array.getElem?_append_right (by omega), show k - alg.size = rs.size by omega, Array.getElem?_push_size]

theorem repInv_coinsOf_alg (t : Tab) (dom : ByteArray) (xs : Array (Array F128)) (hi : Array ByteArray)
    (p : _root_.Flock.Firewall.Frames.Rep) (st' : _root_.Flock.Firewall.Frame.State) (cs : Array F128)
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (flat t.cm0.cap) dom = .ok p.alg)
    (hlt : p.coins.size < p.alg.size) (hto : p.toOpening = false) (hn : p.opening = none) (ho : p.opened = false) :
    RepInv t dom xs hi (coinsOf p st' cs) :=
  ⟨hal, rfl, Or.inl ⟨by simp only [coinsOf, hto, hn], ho, by simp only [coinsOf, Array.size_push]; omega⟩⟩

theorem repInv_coinsOf_ask (t : Tab) (dom : ByteArray) (xs : Array (Array F128)) (hi : Array ByteArray)
    (p : _root_.Flock.Firewall.Frames.Rep) (st' : _root_.Flock.Firewall.Frame.State) (cs : Array F128)
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (flat t.cm0.cap) dom = .ok p.alg)
    (hle : p.alg.size ≤ p.coins.size) (hto : p.toOpening = true) (ho : p.opened = false) {c : ByteArray} {n : ℕ} {k}
    {rs : Array (ByteArray × ℕ)} (hF : openFed t dom xs hi p = (.ask c n k, rs)) (hS : p.opening = some (.ask c n k)) :
    RepInv t dom xs hi (coinsOf p st' cs) := by
  refine ⟨hal, rfl, Or.inr (Or.inl ⟨by simp only [coinsOf, Array.size_push]; omega, ho, ?_⟩)⟩
  rw [openFed_push t dom xs hi p (coinsOf p st' cs) cs hle rfl rfl rfl, hF,
    show (_root_.Flock.Firewall.Drive.Step.ask c n k).feed [cs] rs = (k cs).feed [] (rs.push (c, n)) from rfl, feed_nil]
  simp only [coinsOf, hto, hS]

theorem repInv_coinsOf_done (t : Tab) (dom : ByteArray) (xs : Array (Array F128)) (hi : Array ByteArray)
    (p : _root_.Flock.Firewall.Frames.Rep) (st' : _root_.Flock.Firewall.Frame.State) (cs : Array F128)
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (flat t.cm0.cap) dom = .ok p.alg)
    (hle : p.alg.size ≤ p.coins.size) (hto : p.toOpening = false) (hn : p.opening = none) (ho : p.opened = true)
    {x : Unit} {rs : Array (ByteArray × ℕ)} (hF : openFed t dom xs hi p = (.done x, rs)) :
    RepInv t dom xs hi (coinsOf p st' cs) := by
  refine ⟨hal, rfl, Or.inr (Or.inr ⟨by simp only [coinsOf, Array.size_push]; omega, ho,
    by simp only [coinsOf, hto, hn], x, ?_⟩)⟩
  rw [openFed_push t dom xs hi p (coinsOf p st' cs) cs hle rfl rfl rfl, hF]
  rfl

/-- **The loop places a rep's round as `Drive.reqs` does**: at the rep `p` keeping its schedule so far (`RepInv`), on
the table `t`, extras and salts its session and the loop's salts give it, `placeOf` places its next round (round
`p.coins.size`) at the schedule so far on its coins (`placeAt`), and the rep it puts back, after that round's coins
at any frame state and values (`coinsOf`), keeps `RepInv`. -/
theorem placeOf_sched (salts : Array (ByteArray × ℕ × Array ByteArray)) (l0 : _root_.Flock.Firewall.Level0.Level0)
    (s : _root_.Flock.Firewall.Frames.Session) (p : _root_.Flock.Firewall.Frames.Rep) (dom : ByteArray) {t : Tab}
    {xs : Array (Array F128)} {hi : Array ByteArray} {st : Stmt} {c : Array F128} {r0 : ByteArray} {q0 : ℕ}
    (hst : s.sts[p.table]? = some st) (hlink : s.link = some c)
    (hsalt : salts.find? (fun x => x.1 == l0.root && x.2.1 == p.rep) = some (r0, q0, hi))
    (htab : _root_.Flock.Firewall.Finish.tabOfLevel0 (_root_.Flock.Setup.ofCircuit st) l0 = .ok t)
    (hxs : xs = _root_.Flock.Firewall.Drive.extras st (_root_.Flock.Firewall.Drive.linkPoints st c))
    (hI : RepInv t dom xs hi p) {sched : Array (ByteArray × ℕ)}
    (hs : schedSoFar t p.rep dom xs hi p.coins = .ok sched) :
    ∃ q, _root_.Flock.Firewall.Frames.placeOf salts l0 s p dom =
        .ok ((placeAt sched p.coins.size).1, (placeAt sched p.coins.size).2, q) ∧
      ∀ st' v cs st'', RepInv t dom xs hi (coinsOf { q with st := st', vals := v } st'' cs) := by
  obtain ⟨hal, hto, hop⟩ := hI
  by_cases hlt : p.coins.size < p.alg.size
  · rw [schedSoFar_alg t dom xs hi p hal hlt, Except.ok.injEq] at hs
    subst hs
    rcases hop with ⟨hn, ho, _⟩ | ⟨hle, _⟩ | ⟨hle, _⟩
    · refine ⟨p, ?_, fun st' v cs st'' => repInv_coinsOf_alg t dom xs hi _ st'' cs hal hlt hto hn ho⟩
      rw [placeOf_alg salts l0 s p dom hlt]
      unfold placeAt
      rw [Array.getElem?_eq_getElem hlt]
    · omega
    · omega
  · have hs' := hs
    rw [schedSoFar_open t dom xs hi p hal hlt] at hs'
    rcases hop with ⟨hn, ho, hle⟩ | ⟨hle, ho, hS⟩ | ⟨hle, ho, hn, x, hx⟩
    · have hk : p.coins.size = p.alg.size := by omega
      have hF : openFed t dom xs hi p = (_root_.Flock.Firewall.Drive.openStep t p.rep dom xs hi
          (p.coins.extract 0 p.alg.size), #[]) := by
        unfold openFed
        rw [Array.extract_empty_of_size_le_start hle]
        exact feed_nil _ _
      rw [hF] at hs'
      rw [placeOf_none salts l0 s p dom hlt ho hn hst hlink hsalt htab, ← hxs]
      revert hs' hF
      generalize _root_.Flock.Firewall.Drive.openStep t p.rep dom xs hi (p.coins.extract 0 p.alg.size) = S
      intro hs' hF
      cases S with
      | ask c' n k' =>
        dsimp only at hs'
        rw [Except.ok.injEq] at hs'
        subst hs'
        rw [placeAt_ask p.alg #[] c' n _ (by rw [Array.size_empty]; omega)]
        exact ⟨_, rfl, fun st' v cs st'' =>
          repInv_coinsOf_ask t dom xs hi _ st'' cs hal (show p.alg.size ≤ p.coins.size by omega) rfl ho hF rfl⟩
      | done x' =>
        dsimp only at hs'
        rw [Except.ok.injEq] at hs'
        subst hs'
        rw [placeAt_none (by rw [Array.size_append, Array.size_empty]; omega)]
        exact ⟨_, rfl, fun st' v cs st'' =>
          repInv_coinsOf_done t dom xs hi _ st'' cs hal (show p.alg.size ≤ p.coins.size by omega) hto rfl rfl hF⟩
      | fail e => cases hs'
    · rw [placeOf_some salts l0 s p dom hlt ho hS]
      have hsz := openFed_size t dom xs hi p
      revert hs' hS hsz
      rcases hF : openFed t dom xs hi p with ⟨S, rs⟩
      intro hs' hS hsz
      cases S with
      | ask c' n k' =>
        dsimp only at hs' hS hsz
        rw [Except.ok.injEq] at hs'
        subst hs'
        have hr := openFed_ask_size t dom xs hi p hF
        rw [placeAt_ask p.alg rs c' n _ (by omega)]
        exact ⟨_, rfl, fun st' v cs st'' =>
          repInv_coinsOf_ask t dom xs hi _ st'' cs hal hle rfl ho hF rfl⟩
      | done x' =>
        dsimp only at hs' hS hsz
        rw [Except.ok.injEq] at hs'
        subst hs'
        rw [placeAt_none (by rw [Array.size_append]; omega)]
        exact ⟨_, rfl, fun st' v cs st'' =>
          repInv_coinsOf_done t dom xs hi _ st'' cs hal hle hto rfl rfl hF⟩
      | fail e => cases hs'
    · rw [placeOf_opened salts l0 s p dom hlt ho]
      have hsz := openFed_size t dom xs hi p
      revert hs' hx hsz
      rcases hF : openFed t dom xs hi p with ⟨S, rs⟩
      intro hs' hx hsz
      dsimp only at hx hsz
      subst hx
      dsimp only at hs'
      rw [Except.ok.injEq] at hs'
      subst hs'
      rw [placeAt_none (by rw [Array.size_append]; omega)]
      exact ⟨_, rfl, fun st' v cs st'' =>
        repInv_coinsOf_done t dom xs hi _ st'' cs hal hle hto hn ho hF⟩

end Loop

end Claims.ZKServer
