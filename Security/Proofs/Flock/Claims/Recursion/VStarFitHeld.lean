import Proofs.Flock.Claims.Recursion.VStarHh
import Proofs.Flock.Claims.Recursion.VStarCap0
import Proofs.Flock.Claims.Recursion.VStarPortsM
import Proofs.Flock.Claims.Recursion.OpensReads
import Proofs.Flock.C3.Reps.RoundLinks

/-!
# The slots a part holds are fitted to their rounds' rows (`hfit_A0`)

`hh_of_fit`'s `hfit` at the deployed builder (`g := fitPin`, the fitted records `InnerRec.fit x`): each round before
the last that part `s`'s message reads at `layoutBTops` (`msgAt`) is as many hm96 slots as the round has rows, and the
`k`-th slot's port reads, at the fitted port inputs, the round's `k`-th row.

* **Round 0** is RecCap's `N` instances of port `top` (`capHold`), which read round 0's rows in order
  (`portRow_cap0`, `fit_cap0`).
* **Stream round `j + 1`**, held where it has no tops and absorbs a row, is one slot: the port of part `s` that reads
  the row (`heldRows`), an algebra part's `m<k>` at rep `rep`, whose input is message round `k`'s rep-`rep` row
  (`portRow_algebra`), the last row of inner round `msgRound rs rep k + 1 = j + 1` (`msgsOf`). That round's one row is
  fitted to the slot `rowSlot` names, the first part holding it, so it reads unchanged at part `s` where the two ports
  have one word count (`hW`) (`fit_stream`).

Part `s`'s stager accepts (`hok`) and its circuit is the staged one (`hS`, for `checkParsed` and the slots' input
columns, `rowHidden_staged`).
-/

namespace RecDeploy.VStarBuild

open _root_.Flock
open Claims (Msg)

theorem mIndex_top : RecVStar.mIndex "top" = none := by
  unfold RecVStar.mIndex
  exact ite_eq_right_iff.2 fun h => absurd h (by decide +kernel)

theorem claimIndex_top : RecVStar.claimIndex "top" = none := by
  unfold RecVStar.claimIndex
  exact ite_eq_right_iff.2 fun h => absurd h (by decide +kernel)

/-- **A statement's own port is never `top`.** -/
theorem ownPort_ne_top {sv : _root_.VStar.Stmt} {q : String × String × Array (Option ℕ)}
    (hq : q ∈ RecVStar.ownPorts sv) : q.1 ≠ "top" := by
  cases sv with
  | inner _ _ _ =>
    unfold RecVStar.ownPorts at hq
    obtain ⟨⟨nm, _⟩, -, h⟩ := List.mem_filterMap.1 hq
    dsimp only at h
    obtain ⟨k, hk, rfl⟩ := Option.map_eq_some_iff.1 h
    intro he
    rw [show nm = "top" from he, mIndex_top] at hk
    cases hk
  | sparse _ _ =>
    unfold RecVStar.ownPorts at hq
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hq
    rcases hq with rfl | rfl
    · show "h" ≠ "top"
      decide
    · show "claim" ≠ "top"
      decide
  | opening _ _ _ _ _ _ =>
    unfold RecVStar.ownPorts at hq
    obtain ⟨⟨nm, _⟩, -, h⟩ := List.mem_filterMap.1 hq
    dsimp only at h
    intro he
    cases hm : RecVStar.mIndex nm with
    | some k =>
      rw [hm] at h
      cases h
      rw [show nm = "top" from he, mIndex_top] at hm
      cases hm
    | none =>
      cases hc : RecVStar.claimIndex nm with
      | some k =>
        rw [hm, hc] at h
        cases h
        rw [show nm = "top" from he, claimIndex_top] at hc
        cases hc
      | none =>
        rw [hm, hc] at h
        cases h
  | recOpen _ _ _ _ => simp [RecVStar.ownPorts] at hq
  | recOpenStacked _ _ _ _ => simp [RecVStar.ownPorts] at hq
  | cap _ _ => simp [RecVStar.ownPorts] at hq

/-- A round's rows read off a longer history are the same. -/
theorem roundRows_take (I : InnerRec) (w : Claims.Flock.Wit) (ρ : I.Tape) (hist : List (Msg × Msg)) {K i : ℕ}
    (h : i ≤ K) : roundRows I w ρ (hist.take K) i = roundRows I w ρ hist i := by
  have h1 : min i K = i := Nat.min_eq_left h
  have h2 : min K i = i := by omega
  simp only [roundRows, List.take_take, h1, h2]

/-- Part `s`'s statement and circuit, where `partAt` has them. -/
theorem partAt_some {x : Msg} {s : ℕ} {sv : _root_.VStar.Stmt} {c : Circuit} {r : XR} (hr : readX x = some r)
    (hp : partAt x s = some (sv, c)) :
    (RecVStar.statements r.shape).toOption.bind (·[s]?) = some sv ∧ r.w.parts[s]?.bind partCircuit = some c := by
  unfold partAt at hp
  rw [hr] at hp
  cases h1 : (RecVStar.statements r.shape).toOption.bind (·[s]?) with
  | none => simp [h1] at hp
  | some sv' =>
    cases h2 : r.w.parts[s]?.bind partCircuit with
    | none => simp [h1, h2] at hp
    | some c' =>
      simp [h1, h2] at hp
      obtain ⟨rfl, rfl⟩ := hp
      exact ⟨rfl, rfl⟩

/-- **A staged part's circuit parses as its statement's** (`checkParsed`). -/
theorem checkParsed_staged {st : Stage} {s : ℕ} {x xO : Msg} (hS : StagedPart st s x xO) {r : XR} {c : Circuit}
    {sv : _root_.VStar.Stmt} (hr : readX x = some r) (hpc : r.w.parts[s]?.bind partCircuit = some c)
    (hsv : (RecVStar.statements r.shape).toOption.bind (·[s]?) = some sv) :
    RecVStar.checkParsed sv c = .ok () := by
  obtain ⟨p, r', sv', hx, ht, hpi, hr', ⟨ss, hss, hsv'⟩, h0, htab⟩ := stagedTables hS
  rw [hr] at hr'
  cases hr'
  have hc0 := partCircuit_staged hS hx ht hpi hr hpc h0
  rw [hss] at hsv
  have hsv2 : ss[s]? = some sv := hsv
  rw [hsv'] at hsv2
  cases hsv2
  rw [← hc0]
  exact (htab 0 h0).2.2

/-- A part holding a row has a statement. -/
theorem heldRows_stmt {r : XR} {s : ℕ} {h : ℕ × String × ℕ × ℕ} (hh : h ∈ heldRows r s) :
    ∃ sv, (RecVStar.statements r.shape).toOption.bind (·[s]?) = some sv := by
  cases hst : (RecVStar.statements r.shape).toOption.bind (·[s]?) with
  | some sv => exact ⟨sv, rfl⟩
  | none =>
    unfold heldRows at hh
    rw [hst] at hh
    simp at hh

/-- **The absorbed row's slot**, where part `s` holds the row: one slot, the first holder's port and instance. -/
theorem rowSlot_of_held {r : XR} {s j : ℕ} {h : ℕ × String × ℕ × ℕ} {c : Circuit}
    (hf : (heldRows r s).find? (·.1 == j) = some h) (hc : r.w.parts[s]?.bind partCircuit = some c) :
    ∃ s0 c0 h0, rowSlot r j = [(s0, h0.2.2.1, h0.2.2.2)] ∧ (heldRows r s0).find? (·.1 == j) = some h0 ∧
      r.w.parts[s0]?.bind partCircuit = some c0 := by
  unfold rowSlot
  cases hF : (List.range r.w.parts.length).findSome? (fun s => ((heldRows r s).find? (·.1 == j)).bind fun h =>
      (r.w.parts[s]?.bind partCircuit).map fun _ => (s, h.2.2.1, h.2.2.2)) with
  | none =>
    exfalso
    have hs : s < r.w.parts.length := by
      cases hp : r.w.parts[s]? with
      | none => simp [hp] at hc
      | some _ => exact (List.getElem?_eq_some_iff.1 hp).1
    have := List.findSome?_eq_none_iff.1 hF s (List.mem_range.2 hs)
    simp [hf, hc] at this
  | some sl =>
    obtain ⟨s0, -, h1⟩ := List.exists_of_findSome?_eq_some hF
    cases hf0 : (heldRows r s0).find? (·.1 == j) with
    | none => simp [hf0] at h1
    | some h0 =>
      cases hc0 : r.w.parts[s0]?.bind partCircuit with
      | none => simp [hf0, hc0] at h1
      | some c0 =>
        simp [hf0, hc0] at h1
        subst h1
        try rw [hF]
        exact ⟨s0, c0, h0, rfl, hf0, hc0⟩

/-- **Stream round `j + 1`'s slots** where it has no tops and absorbs a row: the absorbed row's. -/
theorem slotsAt_held {x : Msg} {r : XR} {rs : Array RecVStar.SchedRound} {j : ℕ} (hr : readX x = some r)
    (hrs : repSched r.shape = some rs) (hj : j < 2 * rs.size)
    (he : (rs.getD (j % rs.size) default).2.1.isEmpty = true) (ha : (rs.getD (j % rs.size) default).2.2.2 = true) :
    slotsAt x (j + 1) = rowSlot r j := by
  simp only [slotsAt, hr, hrs, Option.map, Option.getD]
  rw [if_pos he, if_pos (by rw [ha, decide_eq_true hj]; rfl)]

/-- **A round of one slot** at the fitted records: its one row, fitted to the slot, under the round's first salt. -/
theorem roundRows_single (I : InnerRec) {x : Msg} (w : Claims.Flock.Wit) (ρ : (I.fit x).Tape)
    {hist : List (Msg × Msg)} {i : ℕ} {sl : ℕ × ℕ × ℕ} (hl : i ≤ hist.length) (hs : slotsAt x i = [sl]) :
    roundRows (I.fit x) w ρ hist i =
      [⟨fitSlot x sl ((I.rows w ρ.1 (hist.take i)).getD 0 #[]), saltBytes (ρ.2 i 0)⟩] := by
  have hlen : (hist.take i).length = i := by
    rw [List.length_take]
    omega
  have hrows : (I.fit x).rows w ρ.1 (hist.take i) = [fitSlot x sl ((I.rows w ρ.1 (hist.take i)).getD 0 #[])] := by
    show (slotsAt x (hist.take i).length).mapIdx _ = _
    rw [hlen, hs]
    simp
  unfold roundRows
  rw [hrows]
  simp

/-- **Message round `k`'s rep-`rep` row** is the last row of inner round `msgRound rs rep k + 1`. -/
theorem msgsOf_rep {rs : Array RecVStar.SchedRound} {rr : ℕ → List _root_.VStar.In.Row} {k rep j : ℕ}
    {mr : Array _root_.VStar.In.Row} (hrep : rep < 2) (hk : (msgsOf rs rr)[k]? = some mr)
    (hj : msgRound rs rep k = some j) :
    mr[rep]? = some (((rr (j + 1)).getLast?).getD ⟨#[], ByteArray.empty⟩) := by
  unfold msgsOf at hk
  rw [List.getElem?_toArray, List.getElem?_map] at hk
  by_cases hkL : k < (RecVStar.commitRounds rs).length
  · rw [List.getElem?_range hkL] at hk
    obtain rfl := Option.some.inj hk
    simp [List.getElem?_range hrep, hj]
  · rw [List.getElem?_eq_none_iff.2 (by simpa using hkL)] at hk
    cases hk

/-- A fitted row at a port that is not a cap's `top` is of the port's words, each 16 bits. -/
theorem fitSlot_notTop {x : Msg} {sl : ℕ × ℕ × ℕ} (ws : Array ℕ) (ht : isTop x sl = false) :
    (fitSlot x sl ws).size = widthAt x sl ∧ ∀ j (h : j < (fitSlot x sl ws).size), (fitSlot x sl ws)[j] < 65536 := by
  have he : fitSlot x sl ws = fitRow (widthAt x sl) ws := by simp [fitSlot, ht]
  rw [he]
  exact ⟨fitRow_size _ _, fitRow_lt _ _⟩

/-- **Round 0's slots are fitted to its rows**: RecCap's `N` instances of port `top` read round 0's rows in order. -/
theorem fit_cap0 (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    {I : InnerRec} {x : Msg} {w : Claims.Flock.Wit} {ρ : (I.fit x).Tape} {hist : List (Msg × Msg)} {s K : ℕ}
    {stg : Stage} {xO : Msg} {r : XR} {rs : Array RecVStar.SchedRound} (hr : readX x = some r)
    (hrs : repSched r.shape = some rs) (hS : StagedPart stg s x xO) {segs : List (Msg × List HmNets.Lin)}
    (hm : (DeployedR.layR elem).msgAt x s 0 = some segs) :
    ∃ (st : _root_.VStar.Stmt) (c : Circuit) (src : _root_.VStar.In.Src) (pin : _root_.VStar.In.PartIn),
      partAt x s = some (st, c) ∧ srcOf (I.fit x) x w ρ (hist.take K) = some src ∧
      _root_.VStar.In.partIn src (progOf c) st = .ok pin ∧
      segs.length = (roundRows (I.fit x) w ρ hist 0).length ∧
      ∀ k (hk : k < segs.length), ∃ (nm : String) (q n : ℕ) (ps : List ℕ) (rw : _root_.VStar.In.Row),
        segs[k] = (([] : Msg), hmCommit c q n) ∧ c.ports.findIdx? (·.name == nm) = some q ∧
        portRow (fitPin c pin) nm n = some rw ∧ rowHidden c nm n = some ps ∧
        (roundRows (I.fit x) w ρ hist 0)[k]? = some rw := by
  have hm0 : capHold r s = some segs := by
    have h : (readX x).bind (fun r => capHold r s) = some segs := hm
    rw [hr, Option.bind_some] at h
    exact h
  unfold capHold at hm0
  obtain ⟨⟨p, N⟩, hcap, hm0⟩ := Option.bind_eq_some_iff.1 hm0
  try dsimp only at hm0
  by_cases hsp : s = p
  · subst hsp
    rw [if_pos rfl] at hm0
    obtain ⟨c, hc, hm0⟩ := Option.bind_eq_some_iff.1 hm0
    obtain ⟨q, hq, rfl⟩ := Option.map_eq_some_iff.1 hm0
    have hpa := partAt_cap0 hr hcap hc
    have hsv := (partAt_some hr hpa).1
    have hcp := checkParsed_staged hS hr hc hsv
    obtain ⟨src, pin, hsrc, hpin, hlen, hrow⟩ :=
      portRow_cap0 (I := I) (w := w) (ρ := ρ) (hist := hist.take K) hr hrs hcap hc hcp
    rw [roundRows_take (I.fit x) w ρ hist (Nat.zero_le K)] at hlen hrow
    have hq' : c.ports.findIdx? (·.name == "top") = some q := hq
    refine ⟨_, c, src, pin, hpa, hsrc, hpin, by rw [hlen]; simp, fun k hk => ?_⟩
    have hkN : k < N := by simpa using hk
    have hkl : k < (roundRows (I.fit x) w ρ hist 0).length := by rw [hlen]; exact hkN
    obtain ⟨ps, hps⟩ := (rowHidden_staged hS hr hc).2 0 N hsv q hq k
    refine ⟨"top", q, k, ps, (roundRows (I.fit x) w ρ hist 0)[k], by simp, hq', ?_, hps,
      List.getElem?_eq_getElem hkl⟩
    rw [hrow k hkN, List.getElem?_eq_getElem hkl]
  · rw [if_neg hsp] at hm0
    cases hm0

/-- **A held stream round's slot is fitted to its row**: at stream round `j + 1`, where part `s` holds the absorbed
row at its port `m<k>`'s instance `rep`, the port reads, at the fitted port inputs, the round's one row, as the slot
`rowSlot` names has the port's word count (`hW`). -/
theorem fit_stream (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    {I : InnerRec} {x : Msg} {w : Claims.Flock.Wit} {ρ : (I.fit x).Tape} {hist : List (Msg × Msg)} {s K j : ℕ}
    {stg : Stage} {xO : Msg} {r : XR} {rs : Array RecVStar.SchedRound} (hr : readX x = some r)
    (hrs : repSched r.shape = some rs) (hS : StagedPart stg s x xO) (hjK : j + 1 ≤ K) (hH : j + 1 ≤ hist.length)
    (hok : ∀ st c src, partAt x s = some (st, c) → srcOf (I.fit x) x w ρ (hist.take K) = some src →
      ∃ pin, _root_.VStar.In.partIn src (progOf c) st = .ok pin)
    (hW : ∀ s₁ c₁ h₁ s₂ c₂ h₂, r.w.parts[s₁]?.bind partCircuit = some c₁ → h₁ ∈ heldRows r s₁ →
      r.w.parts[s₂]?.bind partCircuit = some c₂ → h₂ ∈ heldRows r s₂ → h₁.1 = h₂.1 →
      (c₁.ports[h₁.2.2.1]?).map (·.words) = (c₂.ports[h₂.2.2.1]?).map (·.words))
    {segs : List (Msg × List HmNets.Lin)} (hm : (DeployedR.layR elem).msgAt x s (j + 1) = some segs) :
    ∃ (st : _root_.VStar.Stmt) (c : Circuit) (src : _root_.VStar.In.Src) (pin : _root_.VStar.In.PartIn),
      partAt x s = some (st, c) ∧ srcOf (I.fit x) x w ρ (hist.take K) = some src ∧
      _root_.VStar.In.partIn src (progOf c) st = .ok pin ∧
      segs.length = (roundRows (I.fit x) w ρ hist (j + 1)).length ∧
      ∀ k (hk : k < segs.length), ∃ (nm : String) (q n : ℕ) (ps : List ℕ) (rw : _root_.VStar.In.Row),
        segs[k] = (([] : Msg), hmCommit c q n) ∧ c.ports.findIdx? (·.name == nm) = some q ∧
        portRow (fitPin c pin) nm n = some rw ∧ rowHidden c nm n = some ps ∧
        (roundRows (I.fit x) w ρ hist (j + 1))[k]? = some rw := by
  have hm1 : ((streamRound x j).bind fun sr => if sr.2.1.isEmpty then
      (if sr.2.2.2 then (readX x).bind fun r => (r.w.parts[s]?.bind partCircuit).bind fun c =>
        ((heldRows r s).find? (·.1 == j)).map fun h => [(([] : Msg), hmCommit c h.2.2.1 h.2.2.2)] else none)
      else none) = some segs := hm
  obtain ⟨sr, hsr, hm1⟩ := Option.bind_eq_some_iff.1 hm1
  simp only [streamRound, hr, hrs, Option.bind_some] at hsr
  by_cases hj : j < 2 * rs.size
  swap
  · rw [if_neg hj] at hsr
    cases hsr
  rw [if_pos hj] at hsr
  obtain rfl := Option.some.inj hsr
  by_cases he : (rs.getD (j % rs.size) default).2.1.isEmpty = true
  swap
  · rw [if_neg he] at hm1
    cases hm1
  by_cases ha : (rs.getD (j % rs.size) default).2.2.2 = true
  swap
  · rw [if_pos he, if_neg ha] at hm1
    cases hm1
  rw [if_pos he, if_pos ha, hr, Option.bind_some] at hm1
  obtain ⟨c, hc, hm1⟩ := Option.bind_eq_some_iff.1 hm1
  obtain ⟨h, hf, rfl⟩ := Option.map_eq_some_iff.1 hm1
  have hh : h ∈ heldRows r s := List.mem_of_find?_eq_some hf
  have hhj : h.1 = j := by simpa using List.find?_some hf
  obtain ⟨sv, hsv⟩ := heldRows_stmt hh
  have hpa : partAt x s = some (sv, c) := by simp [partAt, hr, hsv, hc]
  have hsrc : srcOf (I.fit x) x w ρ (hist.take K) = some { (I.fit x).rest w ρ.1 (hist.take K) with
      msgs := msgsOf rs (roundRows (I.fit x) w ρ (hist.take K)),
      tops := topsOf rs (roundRows (I.fit x) w ρ (hist.take K)) } := by
    simp [srcOf, hr, hrs]
  obtain ⟨pin, hpin⟩ := hok sv c _ hpa hsrc
  obtain ⟨q, hq, k, rs', hk, hrep, hnm, hqi, hrs', hmr⟩ := mem_heldRows_round hsv hc hh
  rw [hrs] at hrs'
  rw [← Option.some.inj hrs', hhj] at hmr
  have hrep2 : h.2.2.2 < 2 := (ownPorts_held hq).2 k hk _ hrep
  obtain ⟨s0, c0, h0, hslot, hf0, hc0⟩ := rowSlot_of_held hf hc
  have hsl : slotsAt x (j + 1) = [(s0, h0.2.2.1, h0.2.2.2)] := (slotsAt_held hr hrs hj he ha).trans hslot
  have hRR := roundRows_single I w ρ hH hsl
  have hh0 : h0 ∈ heldRows r s0 := List.mem_of_find?_eq_some hf0
  have hj0 : h0.1 = j := by simpa using List.find?_some hf0
  obtain ⟨sv0, hsv0⟩ := heldRows_stmt hh0
  obtain ⟨q0, hq0, k0, rs0, -, -, -, hqi0, -, -⟩ := mem_heldRows_round hsv0 hc0 hh0
  obtain ⟨hlt0, hname0, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hqi0
  have hn0 : c0.ports[h0.2.2.1].name = q0.1 := by simpa using hname0
  have hcirc : circAt x s0 = c0 := by simp [circAt, hr, hc0]
  have htop : isTop x (s0, h0.2.2.1, h0.2.2.2) = false := by
    simp [isTop, hcirc, Array.getElem?_eq_getElem hlt0, hn0, ownPort_ne_top hq0]
  have hW' := hW s0 c0 h0 s c h hc0 hh0 hc hh (by rw [hhj, hj0])
  cases sv with
  | inner j' name p =>
    unfold RecVStar.ownPorts at hq
    obtain ⟨⟨nm', x'⟩, hmem, hq'⟩ := List.mem_filterMap.1 hq
    dsimp only at hq'
    obtain ⟨k', hk', rfl⟩ := Option.map_eq_some_iff.1 hq'
    have hkk : k = k' := by
      have hc' : RecVStar.cIndex s!"rec-c{k'}" = some k := hk
      rw [FlockC3.Reps.cIndex_c] at hc'
      exact (Option.some.inj hc').symm
    rw [hkk] at hmr
    have hnm' : h.2.1 = nm' := hnm
    have hqi' : c.ports.findIdx? (·.name == nm') = some h.2.2.1 := hqi
    have hmemn : nm' ∈ p.S.ports.map Prod.fst := List.mem_map.2 ⟨(nm', x'), hmem, rfl⟩
    have hal : _root_.VStar.In.algebraIn _ (progOf c) p.S.ports p.v = .ok pin := hpin
    obtain ⟨mr, hmr', -, hpr⟩ := portRow_algebra hal hmemn hk' hrep2
    have hrow := msgsOf_rep (rr := roundRows (I.fit x) w ρ (hist.take K)) hrep2 hmr' hmr
    rw [roundRows_take (I.fit x) w ρ hist hjK, hRR] at hrow
    simp only [List.getLast?_singleton, Option.getD_some] at hrow
    have hwid : widthAt x (s0, h0.2.2.1, h0.2.2.2) = portWords c nm' := by
      simp only [widthAt, portWords, hcirc, hqi', Option.bind_some]
      rw [hW']
    obtain ⟨hsz0, hlt⟩ := fitSlot_notTop ((I.rows w ρ.1 (hist.take (j + 1))).getD 0 #[]) htop
    have hfit : fitPortRow (portWords c nm')
        ⟨fitSlot x (s0, h0.2.2.1, h0.2.2.2) ((I.rows w ρ.1 (hist.take (j + 1))).getD 0 #[]),
          saltBytes (ρ.2 (j + 1) 0)⟩ =
        ⟨fitSlot x (s0, h0.2.2.1, h0.2.2.2) ((I.rows w ρ.1 (hist.take (j + 1))).getD 0 #[]),
          saltBytes (ρ.2 (j + 1) 0)⟩ :=
      fitPortRow_of (by rw [← hwid]; exact hsz0) hlt (saltBytes_size _)
    refine ⟨_, c, _, pin, hpa, hsrc, hpin, by rw [hRR]; rfl, fun k2 hk2 => ?_⟩
    have hk20 : k2 = 0 := by simpa using hk2
    subst hk20
    obtain ⟨ps, hps⟩ := (rowHidden_staged hS hr hc).1 h hh
    refine ⟨nm', h.2.2.1, h.2.2.2, ps,
      ⟨fitSlot x (s0, h0.2.2.1, h0.2.2.2) ((I.rows w ρ.1 (hist.take (j + 1))).getD 0 #[]),
        saltBytes (ρ.2 (j + 1) 0)⟩, rfl, hqi', ?_, by rw [← hnm']; exact hps, by rw [hRR]; rfl⟩
    rw [portRow_fitPin, hpr, hrow]
    exact congrArg some hfit
  | recOpen _ _ _ _ => simp [RecVStar.ownPorts] at hq
  | recOpenStacked _ _ _ _ => simp [RecVStar.ownPorts] at hq
  | cap _ _ => simp [RecVStar.ownPorts] at hq
  | sparse _ _ =>
    first | cases hpin | simp [_root_.VStar.In.partIn, throw, throwThe, MonadExceptOf.throw] at hpin
  | opening _ _ _ _ _ _ =>
    first | cases hpin | simp [_root_.VStar.In.partIn, throw, throwThe, MonadExceptOf.throw] at hpin

/-- **`hh_of_fit`'s `hfit` at the deployed builder** (`g := fitPin`, the fitted records): every round before the last
that part `s` reads (`msgAt`) is as many slots as the round has rows, the `k`-th reading the `k`-th row; round 0 at
RecCap (`fit_cap0`), a held stream round at the part's `m<k>` port (`fit_stream`). -/
theorem hfit_A0 (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    {I : InnerRec} {x : Msg} {w : Claims.Flock.Wit} {ρ : (I.fit x).Tape} {hist : List (Msg × Msg)} {s : ℕ}
    {stg : Stage} {xO : Msg} {r : XR} {rs : Array RecVStar.SchedRound} (hr : readX x = some r)
    (hrs : repSched r.shape = some rs) (hS : StagedPart stg s x xO) (hH : RestRead.kI x - 1 ≤ hist.length)
    (hok : ∀ st c src, partAt x s = some (st, c) →
      srcOf (I.fit x) x w ρ (hist.take (RestRead.kI x - 1)) = some src →
      ∃ pin, _root_.VStar.In.partIn src (progOf c) st = .ok pin)
    (hW : ∀ s₁ c₁ h₁ s₂ c₂ h₂, r.w.parts[s₁]?.bind partCircuit = some c₁ → h₁ ∈ heldRows r s₁ →
      r.w.parts[s₂]?.bind partCircuit = some c₂ → h₂ ∈ heldRows r s₂ → h₁.1 = h₂.1 →
      (c₁.ports[h₁.2.2.1]?).map (·.words) = (c₂.ports[h₂.2.2.1]?).map (·.words)) :
    ∀ i < RestRead.kI x - 1, ∀ segs, (DeployedR.layR elem).msgAt x s i = some segs →
      ∃ (st : _root_.VStar.Stmt) (c : Circuit) (src : _root_.VStar.In.Src) (pin : _root_.VStar.In.PartIn),
        partAt x s = some (st, c) ∧ srcOf (I.fit x) x w ρ (hist.take (RestRead.kI x - 1)) = some src ∧
        _root_.VStar.In.partIn src (progOf c) st = .ok pin ∧
        segs.length = (roundRows (I.fit x) w ρ hist i).length ∧
        ∀ k (hk : k < segs.length), ∃ (nm : String) (q n : ℕ) (ps : List ℕ) (rw : _root_.VStar.In.Row),
          segs[k] = (([] : Msg), hmCommit c q n) ∧ c.ports.findIdx? (·.name == nm) = some q ∧
          portRow (fitPin c pin) nm n = some rw ∧ rowHidden c nm n = some ps ∧
          (roundRows (I.fit x) w ρ hist i)[k]? = some rw := by
  intro i hi segs hm
  cases i with
  | zero => exact fit_cap0 elem hr hrs hS hm
  | succ j => exact fit_stream elem hr hrs hS (by omega) (by omega) hok hW hm

end RecDeploy.VStarBuild
