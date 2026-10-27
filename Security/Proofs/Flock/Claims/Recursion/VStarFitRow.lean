import Proofs.Flock.Claims.Recursion.VStarFitHeld

/-!
# The absorbed row's piece at a round that commits tops (`fit_rowPiece`)

At stream round `j + 1` with tops, the round's pieces at `layR` are each top's at its cap, then the absorbed row's
(`rowPiece`): one hm96 slot, at the first part holding round `j`'s row. The round's slots (`slotsAt`) are the tops'
then the row's (`rowSlot`), so at the fitted records the absorbed row is the round's last row (`roundRows_last`),
which is the row the stager's message rounds read (`msgsOf`, `getLast?`). The holder's port `m<k>` reads it at the
fitted port inputs, its words being the port's (`hW`), as at a held round (`fit_stream`).
-/

namespace RecDeploy.VStarBuild

open _root_.Flock
open Claims (Msg)

/-- **A round's last row** at the fitted records, where its slots end in `sl`: the round's last row, fitted to `sl`,
under the round's last salt. -/
theorem roundRows_last (I : InnerRec) {x : Msg} (w : Claims.Flock.Wit) (ρ : (I.fit x).Tape)
    {hist : List (Msg × Msg)} {i : ℕ} {A : List (ℕ × ℕ × ℕ)} {sl : ℕ × ℕ × ℕ} (hl : i ≤ hist.length)
    (hs : slotsAt x i = A ++ [sl]) :
    (roundRows (I.fit x) w ρ hist i).getLast? =
      some ⟨fitSlot x sl ((I.rows w ρ.1 (hist.take i)).getD A.length #[]), saltBytes (ρ.2 i A.length)⟩ := by
  have hlen : (hist.take i).length = i := by
    rw [List.length_take]
    omega
  have hrows : (I.fit x).rows w ρ.1 (hist.take i) =
      (A ++ [sl]).mapIdx fun k sl => fitSlot x sl ((I.rows w ρ.1 (hist.take i)).getD k #[]) := by
    show (slotsAt x (hist.take i).length).mapIdx _ = _
    rw [hlen, hs]
  unfold roundRows
  rw [hrows, List.getLast?_eq_getElem?]
  simp [List.getElem?_mapIdx]

/-- **Stream round `j + 1`'s slots where it commits tops**: the tops', then the absorbed row's. -/
theorem slotsAt_tops {x : Msg} {r : XR} {rs : Array RecVStar.SchedRound} {j : ℕ} (hr : readX x = some r)
    (hrs : repSched r.shape = some rs) (he : (rs.getD (j % rs.size) default).2.1.isEmpty = false) :
    slotsAt x (j + 1) =
      (List.range (rs.getD (j % rs.size) default).2.1.size).flatMap (capSlots r rs j) ++ rowSlot r j := by
  simp only [slotsAt, hr, hrs, Option.map, Option.getD]
  rw [if_neg (by rw [he]; exact Bool.false_ne_true)]

/-- **The absorbed row's piece is fitted to the round's last row**: at stream round `j + 1` with tops, where the
round's row piece is part `s0`'s segments `segs`, they are one hm96 slot of a port of part `s0`'s circuit that reads,
at the fitted port inputs, the round's last row, as the first holder's slot has the port's word count (`hW`). -/
theorem fit_rowPiece {I : InnerRec} {x : Msg} {w : Claims.Flock.Wit} {ρ : (I.fit x).Tape} {hist : List (Msg × Msg)}
    {s0 K j : ℕ} {stg : Stage} {xO : Msg} {r : XR} {rs : Array RecVStar.SchedRound} (hr : readX x = some r)
    (hrs : repSched r.shape = some rs) (hS : StagedPart stg s0 x xO) (hjK : j + 1 ≤ K) (hH : j + 1 ≤ hist.length)
    (hj : j < 2 * rs.size) (he : (rs.getD (j % rs.size) default).2.1.isEmpty = false)
    (hok : ∀ st c src, partAt x s0 = some (st, c) → srcOf (I.fit x) x w ρ (hist.take K) = some src →
      ∃ pin, _root_.VStar.In.partIn src (progOf c) st = .ok pin)
    (hW : ∀ s₁ c₁ h₁ s₂ c₂ h₂, r.w.parts[s₁]?.bind partCircuit = some c₁ → h₁ ∈ heldRows r s₁ →
      r.w.parts[s₂]?.bind partCircuit = some c₂ → h₂ ∈ heldRows r s₂ → h₁.1 = h₂.1 →
      (c₁.ports[h₁.2.2.1]?).map (·.words) = (c₂.ports[h₂.2.2.1]?).map (·.words))
    {segs : List (Msg × List HmNets.Lin)} (hp : rowPiece r j = [(s0, segs)]) :
    ∃ (st : _root_.VStar.Stmt) (c : Circuit) (src : _root_.VStar.In.Src) (pin : _root_.VStar.In.PartIn)
      (nm : String) (q n : ℕ) (ps : List ℕ) (rw : _root_.VStar.In.Row),
      partAt x s0 = some (st, c) ∧ srcOf (I.fit x) x w ρ (hist.take K) = some src ∧
      _root_.VStar.In.partIn src (progOf c) st = .ok pin ∧ segs = [(([] : Msg), hmCommit c q n)] ∧
      c.ports.findIdx? (·.name == nm) = some q ∧ portRow (fitPin c pin) nm n = some rw ∧
      rowHidden c nm n = some ps ∧ (roundRows (I.fit x) w ρ hist (j + 1)).getLast? = some rw := by
  unfold rowPiece at hp
  cases hF : (List.range r.w.parts.length).findSome? (fun s => ((heldRows r s).find? (·.1 == j)).bind fun h =>
      (r.w.parts[s]?.bind partCircuit).map fun c => (s, [(([] : Msg), hmCommit c h.2.2.1 h.2.2.2)])) with
  | none => rw [hF] at hp; cases hp
  | some pc =>
    rw [hF] at hp
    have hpc : pc = (s0, segs) := by simpa using hp
    subst hpc
    obtain ⟨a, -, h1⟩ := List.exists_of_findSome?_eq_some hF
    try dsimp only at h1
    obtain ⟨h, hf, h1⟩ := Option.bind_eq_some_iff.1 h1
    obtain ⟨c, hc, h1⟩ := Option.map_eq_some_iff.1 h1
    simp only [Prod.mk.injEq] at h1
    obtain ⟨rfl, rfl⟩ := h1
    have hh : h ∈ heldRows r a := List.mem_of_find?_eq_some hf
    have hhj : h.1 = j := by simpa using List.find?_some hf
    obtain ⟨sv, hsv⟩ := heldRows_stmt hh
    have hpa : partAt x a = some (sv, c) := by simp [partAt, hr, hsv, hc]
    have hsrc : srcOf (I.fit x) x w ρ (hist.take K) = some { (I.fit x).rest w ρ.1 (hist.take K) with
        msgs := msgsOf rs (roundRows (I.fit x) w ρ (hist.take K)),
        tops := topsOf rs (roundRows (I.fit x) w ρ (hist.take K)) } := by
      simp [srcOf, hr, hrs]
    obtain ⟨pin, hpin⟩ := hok sv c _ hpa hsrc
    obtain ⟨q, hq, k, rs', hk, hrep, hnm, hqi, hrs', hmr⟩ := mem_heldRows_round hsv hc hh
    rw [hrs] at hrs'
    rw [← Option.some.inj hrs', hhj] at hmr
    have hrep2 : h.2.2.2 < 2 := (ownPorts_held hq).2 k hk _ hrep
    obtain ⟨s1, c1, h0, hslot, hf0, hc0⟩ := rowSlot_of_held hf hc
    have hsl : slotsAt x (j + 1) =
        (List.range (rs.getD (j % rs.size) default).2.1.size).flatMap (capSlots r rs j) ++
          [(s1, h0.2.2.1, h0.2.2.2)] := by
      rw [slotsAt_tops hr hrs he, hslot]
    have hRL := roundRows_last I w ρ hH hsl
    have hh0 : h0 ∈ heldRows r s1 := List.mem_of_find?_eq_some hf0
    have hj0 : h0.1 = j := by simpa using List.find?_some hf0
    obtain ⟨sv0, hsv0⟩ := heldRows_stmt hh0
    obtain ⟨q0, hq0, k0, rs0, -, -, -, hqi0, -, -⟩ := mem_heldRows_round hsv0 hc0 hh0
    obtain ⟨hlt0, hname0, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hqi0
    have hn0 : c1.ports[h0.2.2.1].name = q0.1 := by simpa using hname0
    have hcirc : circAt x s1 = c1 := by simp [circAt, hr, hc0]
    have htop : isTop x (s1, h0.2.2.1, h0.2.2.2) = false := by
      simp [isTop, hcirc, Array.getElem?_eq_getElem hlt0, hn0, ownPort_ne_top hq0]
    have hW' := hW s1 c1 h0 a c h hc0 hh0 hc hh (by rw [hhj, hj0])
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
      rw [roundRows_take (I.fit x) w ρ hist hjK, hRL] at hrow
      simp only [Option.getD_some] at hrow
      have hwid : widthAt x (s1, h0.2.2.1, h0.2.2.2) = portWords c nm' := by
        simp only [widthAt, portWords, hcirc, hqi', Option.bind_some]
        rw [hW']
      generalize hA : ((List.range (rs.getD (j % rs.size) default).2.1.size).flatMap (capSlots r rs j)).length = m
        at hrow hRL
      obtain ⟨hsz0, hlt⟩ := fitSlot_notTop ((I.rows w ρ.1 (hist.take (j + 1))).getD m #[]) htop
      have hfit : fitPortRow (portWords c nm')
          ⟨fitSlot x (s1, h0.2.2.1, h0.2.2.2) ((I.rows w ρ.1 (hist.take (j + 1))).getD m #[]),
            saltBytes (ρ.2 (j + 1) m)⟩ =
          ⟨fitSlot x (s1, h0.2.2.1, h0.2.2.2) ((I.rows w ρ.1 (hist.take (j + 1))).getD m #[]),
            saltBytes (ρ.2 (j + 1) m)⟩ :=
        fitPortRow_of (by rw [← hwid]; exact hsz0) hlt (saltBytes_size _)
      obtain ⟨ps, hps⟩ := (rowHidden_staged hS hr hc).1 h hh
      refine ⟨_, c, _, pin, nm', h.2.2.1, h.2.2.2, ps,
        ⟨fitSlot x (s1, h0.2.2.1, h0.2.2.2) ((I.rows w ρ.1 (hist.take (j + 1))).getD m #[]),
          saltBytes (ρ.2 (j + 1) m)⟩, hpa, hsrc, hpin, rfl, hqi', ?_, by rw [← hnm']; exact hps, hRL⟩
      rw [portRow_fitPin, hpr, hrow]
      exact congrArg some hfit
    | recOpen _ _ _ _ => simp [RecVStar.ownPorts] at hq
    | recOpenStacked _ _ _ _ => simp [RecVStar.ownPorts] at hq
    | cap _ _ => simp [RecVStar.ownPorts] at hq
    | sparse _ _ =>
      first | cases hpin | simp [_root_.VStar.In.partIn, throw, throwThe, MonadExceptOf.throw] at hpin
    | opening _ _ _ _ _ _ =>
      first | cases hpin | simp [_root_.VStar.In.partIn, throw, throwThe, MonadExceptOf.throw] at hpin

end RecDeploy.VStarBuild
