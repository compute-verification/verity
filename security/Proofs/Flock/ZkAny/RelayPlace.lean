module

public import Proofs.Flock.ZkAny.RelayFollow

/-!
# A rep's frame past its stream's first round keeps its schedule so far

`Loop.placeOf_sched` places a rep's next round at its schedule so far on its coins where the frame keeps that schedule
(`Loop.RepInv`), and the frame after the round's coins keeps it. Here the loop's round reads it so:

* `roundPlace_later_of`: past the stream's first round, the frame `roundPlace` readies is `placeOf`'s on the found
  frame, in `placeOf`'s place unless that is the inner proof's own.
* **`roundRun_repInv`**: so at a `RoundRun` past the stream's first round whose round releases and whose coins the
  rep's frame took, where the found frame keeps its schedule so far, the round is in the place that schedule gives the
  frame's next round (unless that is the inner proof's own), and the frame the session's next round finds keeps it.
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock (F128 Stmt)
open _root_.Flock.Firewall.Session (State Placed byte?)
open _root_.Flock.Firewall.Frames (Sessions Rep)
open _root_.Flock.Firewall.Frame (Req Place)

set_option maxHeartbeats 1000000 in
/-- **Past the stream's first round the readied frame is `placeOf`'s**: at the session `s`, its table `name` and its
level 0 `l0`, the frame `roundPlace` readies is the one `placeOf` returns on the found frame, in `placeOf`'s place
unless that is the inner proof's own. -/
theorem roundPlace_later_of {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i r j : Nat}
    {dom content : ByteArray} {n : Nat} {p q : Rep} {pl : Place} {f : Bool} {nd : Option String}
    (hf : R.find sid i r = some (j, p))
    (h : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n = .ok (q, pl, f, nd))
    (hl : f = false) {s : _root_.Flock.Firewall.Frames.Release} {name : String}
    {l0 : _root_.Flock.Firewall.Level0.Level0} (hs : R.sessions.find? (·.sid == sid) = some s)
    (hn : s.spec.tables[i]? = some name) (hl0 : _root_.Flock.Firewall.Level0.ofSession L0 sid name = some l0)
    {v : Place × Nat × Rep}
    (hv : _root_.Flock.Firewall.Frames.placeOf R.salts l0 s p (s.spec.domain name r).toUTF8 = .ok v) :
    q = v.2.2 ∧ (pl = v.1 ∨ v.1 = .own) := by
  have hfst : (p.coins.isEmpty && !p.st.begun) = false := (roundPlace_first hf h).symm.trans hl
  obtain ⟨pl₀, n₀, q₀⟩ := v
  unfold _root_.Flock.Firewall.Frames.roundPlace at h
  simp only [hf, bind, Except.bind, pure, Except.pure] at h
  repeat' (first
    | simp only [hs, hn, hl0, hfst, hv, bind, Except.bind, pure, Except.pure, Bool.false_eq_true, ↓reduceIte] at h
    | split at h)
  all_goals first
    | (cases h; done)
    | exact absurd (by assumption) throw_ne_ok
    | (cases h; exact ⟨rfl, Or.inr rfl⟩)
    | (cases h; exact ⟨rfl, Or.inl rfl⟩)
    | (cases h; exact ⟨rfl, Or.inr (by assumption)⟩)

/-- **A rep's frame past its stream's first round keeps its schedule so far**: at a `RoundRun` past the stream's first
round whose round releases and whose coins the rep's frame took off the auditor's answer, where the found frame keeps
its schedule so far (`Loop.RepInv`) on the table and extras the session gives it and the salts the loop's salts give
it at the opening's first round (`hsalt`, where `placeOf` reads them), the round is in the place that schedule gives
the frame's next round, unless that is the inner proof's own, and the frame the session's next round finds keeps it. -/
theorem roundRun_repInv {sid : Nat} {st : State} {pl : Placed} {stream : String} {n i r : Nat}
    {dom content : ByteArray} {e : Env} {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {R₃ : Sessions}
    (hr : RoundRun sid st pl stream n i r dom content e R L0 R₃) (hreq : st.request pl (.round stream n) = none)
    (hfr : (frameAt R sid i r).isSome) {q p₁ : Rep} {pl' : Place} {f : Bool} {nd : Option String} {o : ByteArray}
    (hrp : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n = .ok (q, pl', f, nd))
    (ho : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom content n).out = .ok o)
    (hput : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom content n).put = some p₁) {a : ByteArray}
    (ha : e.ans = some a) (hb : byte? a 0 = some 2) (hs : (st.answer (.round stream n) a).1.stop = none)
    (hc : e.coins = .ok ()) {pre : List Req} {ev : List (Nat × ByteArray × Place × Array F128)}
    (hR : RepRun pre ev q) {j : Nat} {p : Rep} (hf : R.find sid i r = some (j, p)) (hl : f = false)
    {s : _root_.Flock.Firewall.Frames.Release} {name : String} {l0 : _root_.Flock.Firewall.Level0.Level0}
    (hss : R.sessions.find? (·.sid == sid) = some s) (hn : s.spec.tables[i]? = some name)
    (hl0 : _root_.Flock.Firewall.Level0.ofSession L0 sid name = some l0) {t : _root_.Flock.Firewall.Finish.Tab}
    {xs : Array (Array F128)} {hi : Array ByteArray} {stm : Stmt} {c : Array F128} {r0 : ByteArray} {q0 : Nat}
    (hst : s.sts[p.table]? = some stm) (hlink : s.link = some c)
    (hsalt : p.alg.size ≤ p.coins.size → p.opened = false → p.opening = none →
      R.salts.find? (fun x => x.1 == l0.root && x.2.1 == p.rep) = some (r0, q0, hi))
    (htab : _root_.Flock.Firewall.Finish.tabOfLevel0 (_root_.Flock.Setup.ofCircuit stm) l0 = .ok t)
    (hxs : xs = _root_.Flock.Firewall.Drive.extras stm (_root_.Flock.Firewall.Drive.linkPoints stm c))
    (hI : Loop.RepInv t (s.spec.domain name r).toUTF8 xs hi p) {sched : Array (ByteArray × Nat)}
    (hsc : Loop.schedSoFar t p.rep (s.spec.domain name r).toUTF8 xs hi p.coins = .ok sched) :
    (pl' = (Loop.placeAt sched p.coins.size).1 ∨ (Loop.placeAt sched p.coins.size).1 = .own) ∧
      ∃ p₃, frameAt R₃ sid i r = some p₃ ∧ Loop.RepInv t (s.spec.domain name r).toUTF8 xs hi p₃ := by
  obtain ⟨q', hpo, hinv⟩ :=
    Loop.placeOf_sched R.salts l0 s p (s.spec.domain name r).toUTF8 hst hlink hsalt htab hxs hI hsc
  obtain ⟨hq, hpl⟩ := roundPlace_later_of hf hrp hl hss hn hl0 hpo
  obtain ⟨st₂, -, hat, -⟩ := roundRun_next hr hreq hfr hrp ho hput ha hb hs hc hR
  rcases hres : _root_.Flock.Firewall.Frame.step q.cfg q.st
    (.round n (if nd.isSome then ByteArray.empty else content) pl') with ⟨a', st'⟩
  obtain ⟨v, hv⟩ := roundStep_put hrp hres
  rw [hv, Option.some.injEq] at hput
  subst hput
  have hq' : q = q' := hq
  subst hq'
  exact ⟨hpl, _, hat, hinv st' v _ st₂⟩

end Claims.ZKServer.Relay
