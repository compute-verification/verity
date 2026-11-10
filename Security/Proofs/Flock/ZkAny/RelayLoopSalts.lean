module

public import Proofs.Flock.ZkAny.RelaySame
public import Proofs.Flock.ZkAny.RelayFresh

/-!
# A rep's salts for levels ≥ 1 where the loop's round reads them

The round lemmas take the rep's salts for levels ≥ 1 in the loop's salts at the opening's first round, where `placeOf`
reads them (`hsalt`). `placeOf` begins the opening only on salts it found there, so a round `roundPlace` readied had
them in the frames it read, and a frame keeping its schedule so far before its opening begins keeps it at any salts:

* `placeOf_salts`: where `placeOf` places the opening's first round, it found the rep's salts among the loop's.
* `roundPlace_placeOf`: past the stream's first round, `placeOf` placed the found frame's round.
* `roundPlace_of_out`: a round that releases is one `roundPlace` readied.
* **`roundPlace_salts`**: so past the stream's first round the frames the round read hold the rep's salts wherever
  `placeOf` reads them. **`roundPlace_salts_first`**: at the stream's first round, where `withAlg` scheduled no rounds.
* `repInv_hi`: before its opening begins, a frame keeps its schedule so far at any salts.
* **`sameInv_salts`**: so a frame keeping `SameInv` at a round past its stream's first keeps it at salts the frames
  the round read hold wherever `placeOf` reads them, the round lemmas' `hsalt`.
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock (F128 Stmt)
open _root_.Flock.Firewall.Frames (Sessions Rep Release)
open _root_.Flock.Firewall.Frame (Req Place)

/-- **Where `placeOf` places the opening's first round, it found the rep's salts** among the loop's salts `salts`: it
begins the opening only on them. -/
theorem placeOf_salts {salts : Array (ByteArray × Nat × Array ByteArray)} {l0 : _root_.Flock.Firewall.Level0.Level0}
    {s : Release} {p : Rep} {dom : ByteArray} {v : Place × Nat × Rep}
    (h : _root_.Flock.Firewall.Frames.placeOf salts l0 s p dom = .ok v) (hle : p.alg.size ≤ p.coins.size)
    (ho : p.opened = false) (hop : p.opening = none) :
    ∃ r0 q0 hi, salts.find? (fun x => x.1 == l0.root && x.2.1 == p.rep) = some (r0, q0, hi) := by
  unfold _root_.Flock.Firewall.Frames.placeOf at h
  simp only [Nat.not_lt.mpr hle, ↓reduceDIte, ho, hop, Option.isNone_none, Bool.false_eq_true, ↓reduceIte, bind,
    Except.bind] at h
  repeat' split at h
  all_goals first
    | (cases h; done)
    | exact absurd (by assumption) throw_ne_ok
    | exact ⟨_, _, _, by assumption⟩

set_option maxHeartbeats 1000000 in
/-- **Past the stream's first round `placeOf` placed the found frame's round**, at the session `s`, its table `name`
and its level 0 `l0`. -/
theorem roundPlace_placeOf {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i r j : Nat}
    {dom content : ByteArray} {n : Nat} {p q : Rep} {pl : Place} {f : Bool} {nd : Option String}
    (hf : R.find sid i r = some (j, p))
    (h : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n = .ok (q, pl, f, nd))
    (hfst : (p.coins.isEmpty && !p.st.begun) = false) {s : Release} {name : String}
    {l0 : _root_.Flock.Firewall.Level0.Level0} (hs : R.sessions.find? (·.sid == sid) = some s)
    (hn : s.spec.tables[i]? = some name) (hl0 : _root_.Flock.Firewall.Level0.ofSession L0 sid name = some l0) :
    ∃ v, _root_.Flock.Firewall.Frames.placeOf R.salts l0 s p (s.spec.domain name r).toUTF8 = .ok v := by
  rcases hv : _root_.Flock.Firewall.Frames.placeOf R.salts l0 s p (s.spec.domain name r).toUTF8 with e | v
  · unfold _root_.Flock.Firewall.Frames.roundPlace at h
    simp only [hf, bind, Except.bind, pure, Except.pure] at h
    repeat' (first
      | simp only [hs, hn, hl0, hfst, hv, Bool.false_eq_true, ↓reduceIte] at h
      | split at h)
    all_goals cases h
  · exact ⟨v, rfl⟩

/-- **A round that releases is one `roundPlace` readied.** -/
theorem roundPlace_of_out {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i r : Nat}
    {dom content : ByteArray} {n : Nat} {o : ByteArray}
    (ho : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom content n).out = .ok o) :
    ∃ q pl f nd, _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n = .ok (q, pl, f, nd) := by
  rcases h : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n with e | ⟨q, pl, f, nd⟩
  · have he : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom content n).out = .error e := by
      unfold _root_.Flock.Firewall.Frames.roundStep
      rw [h]
    rw [he] at ho
    cases ho
  · exact ⟨q, pl, f, nd, rfl⟩

/-- **Past the stream's first round the frames the round read hold the rep's salts wherever `placeOf` reads them**:
at the opening's first round, `placeOf` found them. -/
theorem roundPlace_salts {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i r : Nat}
    {dom content : ByteArray} {n : Nat} {p q : Rep} {pl : Place} {f : Bool} {nd : Option String}
    (hfa : frameAt R sid i r = some p) (hk : p.coins.isEmpty = false)
    (hrp : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n = .ok (q, pl, f, nd))
    {s : Release} {name : String} {l0 : _root_.Flock.Firewall.Level0.Level0}
    (hss : R.sessions.find? (·.sid == sid) = some s) (hn : s.spec.tables[i]? = some name)
    (hl0 : _root_.Flock.Firewall.Level0.ofSession L0 sid name = some l0) :
    ∃ r0 q0 hi, p.alg.size ≤ p.coins.size → p.opened = false → p.opening = none →
      R.salts.find? (fun x => x.1 == l0.root && x.2.1 == p.rep) = some (r0, q0, hi) := by
  by_cases hc : p.alg.size ≤ p.coins.size ∧ p.opened = false ∧ p.opening = none
  · obtain ⟨j, hf⟩ := find_of_frameAt hfa
    obtain ⟨v, hv⟩ := roundPlace_placeOf hf hrp (by rw [hk, Bool.false_and]) hss hn hl0
    obtain ⟨r0, q0, hi, hs⟩ := placeOf_salts hv hc.1 hc.2.1 hc.2.2
    exact ⟨r0, q0, hi, fun _ _ _ => hs⟩
  · exact ⟨.empty, 0, #[], fun a b c => absurd ⟨a, b, c⟩ hc⟩

/-- **At the stream's first round the frames the round read hold the rep's salts where `withAlg` scheduled no
rounds**: there the round is the opening's first, which `placeOf` places on the frame `withAlg` gave, after rep 0's
algebraic-phase coins at rep 1. -/
theorem roundPlace_salts_first {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i r j : Nat}
    {dom content : ByteArray} {n : Nat} {p q : Rep} {pl : Place} {f : Bool} {nd : Option String}
    (hf : R.find sid i r = some (j, p)) (hp : FreshRep p)
    (hrp : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n = .ok (q, pl, f, nd))
    {s : Release} {name : String} {l0 : _root_.Flock.Firewall.Level0.Level0}
    (hss : R.sessions.find? (·.sid == sid) = some s) (hn : s.spec.tables[i]? = some name)
    (hl0 : _root_.Flock.Firewall.Level0.ofSession L0 sid name = some l0) {p' : Rep}
    (hw : _root_.Flock.Firewall.Frames.withAlg s l0 name p = .ok p') :
    ∃ r0 q0 hi, p'.alg.size = 0 →
      R.salts.find? (fun x => x.1 == l0.root && x.2.1 == p.rep) = some (r0, q0, hi) := by
  by_cases h0 : p'.alg.size = 0
  · have hfst : (p.coins.isEmpty && !p.st.begun) = true := by
      rw [hp.coins, hp.st]
      rfl
    obtain ⟨-, hrep, -, hop, hod, -⟩ := withAlg_keep hw
    obtain ⟨p'', pl₀, n₀, hpo, -, hp''⟩ := roundPlace_first_of hf hrp hfst hss hn hl0 hw
    have hle : p'.alg.size ≤ p'.coins.size := by omega
    have ho : p'.opened = false := by rw [hod, hp.opened]
    have hop' : p'.opening = none := by rw [hop, hp.opening]
    rw [← hrep]
    rcases hp'' with rfl | ⟨st₁, rfl⟩
    · obtain ⟨r0, q0, hi, hs⟩ := placeOf_salts hpo hle ho hop'
      exact ⟨r0, q0, hi, fun _ => hs⟩
    · obtain ⟨r0, q0, hi, hs⟩ := placeOf_salts hpo hle ho hop'
      exact ⟨r0, q0, hi, fun _ => hs⟩
  · exact ⟨.empty, 0, #[], fun h => absurd h h0⟩

/-- **Before its opening begins, a frame keeps its schedule so far at any salts**: only the opening reads them. -/
theorem repInv_hi {t : _root_.Flock.Firewall.Finish.Tab} {dom : ByteArray} {xs : Array (Array F128)}
    {hi hi' : Array ByteArray} {p : Rep} (h : Loop.RepInv t dom xs hi p) (hop : p.opening = none)
    (ho : p.opened = false) : Loop.RepInv t dom xs hi' p := by
  refine ⟨h.alg, h.toOpening, Or.inl ⟨hop, ho, ?_⟩⟩
  rcases h.opening with h1 | ⟨-, -, h2⟩ | ⟨-, h3, -⟩
  · exact h1.2.2
  · rw [hop] at h2
    cases h2
  · rw [ho] at h3
    cases h3

/-- **A frame keeping `SameInv` at a round past its stream's first keeps it at salts the round lemmas read**: at
salts the frames the round read hold wherever `placeOf` reads them (`hsalt`), the found ones at the opening's first
round, before which nothing reads the salts. -/
theorem sameInv_salts {W Rr : Type} {L : LeanRelease W Rr} {x : Msg} {ρ : Rr} {h : List (Msg × Msg)}
    {cs : List FlockDeploy.Coin} {q₀ j : Nat} {t : _root_.Flock.Firewall.Finish.Tab} {dom' : ByteArray}
    {xs : Array (Array F128)} {hi : Array ByteArray} {pre : List Req}
    {ev : List (Nat × ByteArray × Place × Array F128)} {p : Rep} (hI : SameInv L x ρ h cs q₀ j t dom' xs hi pre ev p)
    {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i r : Nat} {dom content : ByteArray} {n : Nat}
    {q : Rep} {pl : Place} {f : Bool} {nd : Option String} (hfa : frameAt R sid i r = some p)
    (hrp : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n = .ok (q, pl, f, nd))
    {s : Release} {name : String} {l0 : _root_.Flock.Firewall.Level0.Level0}
    (hss : R.sessions.find? (·.sid == sid) = some s) (hn : s.spec.tables[i]? = some name)
    (hl0 : _root_.Flock.Firewall.Level0.ofSession L0 sid name = some l0) :
    ∃ r0 q0 hi', SameInv L x ρ h cs q₀ j t dom' xs hi' pre ev p ∧
      (p.alg.size ≤ p.coins.size → p.opened = false → p.opening = none →
        R.salts.find? (fun x => x.1 == l0.root && x.2.1 == p.rep) = some (r0, q0, hi')) := by
  have hsz : p.coins.size = j + 1 := by rw [hI.coins, Array.size_map, Array.size_range]
  have hk : p.coins.isEmpty = false := by simp [Array.isEmpty, hsz]
  by_cases hc : p.opened = false ∧ p.opening = none
  · obtain ⟨r0, q0, hi', hs⟩ := roundPlace_salts hfa hk hrp hss hn hl0
    exact ⟨r0, q0, hi', ⟨hI.run, hI.cfg, hI.st, hI.coins, repInv_hi hI.inv hc.2 hc.1⟩, hs⟩
  · exact ⟨.empty, 0, hi, hI, fun _ a b => absurd ⟨a, b⟩ hc⟩

end Claims.ZKServer.Relay
