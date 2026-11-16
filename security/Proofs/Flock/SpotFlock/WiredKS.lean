module

public import Proofs.Flock.SpotFlock.WiredCode
public import Proofs.Flock.SamplingGeneric.TwoCount

/-!
# SPOT[Flock]'s two-stage knowledge soundness at the connected program

`ksTwo_spot` (`KS`) is `CompiledTwo.KS` at `code h`, over the members' programs side by side. This file proves it
at `codeWired h ω keep` (`WiredCode`), over the members' programs wired by `ω`, from the same facts of the code:
each is `code h`'s, carried to the wired program by what a map that rewires the members' programs keeps
(`Prog.Rewires`, `cell_cut`, `replayOf_cut`, `scope_rewired`).

* `wired_stmt`: a statement, partition and design the wired code evaluates are `code h`'s, wired.
* `units_wired`, `nPU_wired`, `checked_wired`: the wired code has `code h`'s units and checks `code h`'s.
* `refusesTwo_wired`, `regMatchesTwo_wired`, `replayCap_wired`, `sessionRunsTo_wired`: the code facts.
* `run_checked`: what an accepted run of `V h` that checks a unit and shows its session passed: the session
  statements' check, at the unit's statement.
* `sessionOpensTwo_wired`: the sessions' openings at the wired code, from the session statements' openings there
  (`StmtOpensWired`).
* `ksTwo_wired`: `CompiledTwo.KS` at the wired code, from a map that rewires the members' programs where the design
  evaluates and `StmtOpensWired`.
* `ksTwo_connected`, `exfilTwo_connected`: at the code's own wiring (`codeConnected h`), from SPOT's own frame there
  and the addresses (`BuiltWired`, `AddressesConnected`). What they read of that wiring (`WiresFirst`,
  `SourcesNamed`) the design's evaluation decides.
-/

@[expose] public section

namespace Claims.Sampling

/-- **The proof blocks in sampled replay blocks are those of the partitions without the gates `ω` moves**, where
no gate is in two replay blocks, every proof block has a gate `ω` leaves alone and a replay block. -/
theorem sampleBlocks_cut {Pt Pρ : Partition} (ω : ℕ → ℕ) (hnd : Pρ.gates.Nodup)
    (hne : ∀ u ∈ Pt.blocks, ∃ g ∈ u, ω g = g) (hall : ∀ v < Pt.blocks.length, (replayOf Pt Pρ v).isSome)
    (S : List ℕ) : sampleBlocks (Pt.cut ω) (Pρ.cut ω) S = sampleBlocks Pt Pρ S := by
  unfold sampleBlocks
  rw [Partition.length_blocks_cut]
  refine Finset.filter_congr fun v hv => ?_
  rw [replayOf_cut_eq ω hnd hne (hall v (Finset.mem_range.1 hv))]

end Claims.Sampling

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy Properties.Spot Finset

variable (h : Held) (ω : ℕ → ℕ) (keep : ℕ → Bool)

/-! ## The wired code's statement, units and checked units -/

/-- **What the wired code evaluates is `code h`'s, wired**: a statement the wired code parses, with its partition
and its design, is a statement `code h` parses in the two-stage scope with its design, the program wired by `ω` and
each proof block and replay block without the gates `ω` moves. -/
theorem wired_stmt {x : Msg} {s' : AuditStatement} {Pt' : Partition} {d' : Design.Two}
    (hx : parseWired h ω keep x = some s') (hP : partitionWired h ω s' = some Pt')
    (hd : twoWired h ω s' = some d') :
    ∃ s d, parse h x = some s ∧ two h s = some d ∧ InScopeTwo (membersPart h.ms) d s ∧
      s' = { s with prog := (bitsProgs h.ms).wired ω keep } ∧ Pt' = (membersPart h.ms).cut ω ∧
      d' = { d with part := d.part.cut ω } := by
  obtain ⟨s, hs, rfl⟩ := Option.map_eq_some_iff.1 hx
  have hd' : (two h s).map (fun d : Design.Two => ({ d with part := d.part.cut ω } : Design.Two)) = some d' := hd
  obtain ⟨d, h2, rfl⟩ := Option.map_eq_some_iff.1 hd'
  have hP' : some ((membersPart h.ms).cut ω) = some Pt' := hP
  obtain ⟨l, -, -, -, hsc, -⟩ := two_eq h h2
  exact ⟨s, d, hs, h2, scopeOk_sound hsc s (parse_prog h hs), rfl, (Option.some.inj hP').symm, rfl⟩

/-- **The wired code has `code h`'s units.** -/
theorem units_wired (x : Msg) :
    units ((baseWired h ω keep).atOrder (orderWired h ω keep)) x = units ((base h).atOrder (order h)) x := by
  unfold SamplingDeploy.units
  have hp : ((baseWired h ω keep).atOrder (orderWired h ω keep)).parse x =
      (parse h x).map fun s => { s with prog := (bitsProgs h.ms).wired ω keep } := rfl
  have hq : ((base h).atOrder (order h)).parse x = parse h x := rfl
  rw [hp, hq]
  cases parse h x with
  | none => rfl
  | some s => exact Partition.length_blocks_cut (membersPart h.ms) ω

/-- **The wired code's record has `code h`'s count of proof blocks.** -/
theorem nPU_wired (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (x : Msg) :
    Claims.Sampling.nPU (deploy rel (codeWired h ω keep)) x = Claims.Sampling.nPU (deploy rel (code h)) x := by
  unfold Claims.Sampling.nPU
  have hp : (deploy rel (codeWired h ω keep)).parse x =
      (parse h x).map fun s => { s with prog := (bitsProgs h.ms).wired ω keep } := rfl
  have hq : (deploy rel (code h)).parse x = parse h x := rfl
  rw [hp, hq]
  cases parse h x with
  | none => rfl
  | some s => exact Partition.length_blocks_cut (membersPart h.ms) ω

/-- **The wired code checks the units `code h` checks.** -/
theorem checked_wired (x : Msg) (r : Fin ((code h).V.k x) → FlockDeploy.Coin) :
    (codeWired h ω keep).checked x r = (code h).checked x r := by
  have e := units_wired h ω keep x
  show Option.getD (Option.map _ (reg h x)) [] = Option.getD (Option.map _ (reg h x)) []
  cases reg h x with
  | none => rfl
  | some g =>
    show g.checked (units ((baseWired h ω keep).atOrder (orderWired h ω keep)) x) _ _ = _
    rw [e]
    rfl

/-- A unit the wired code checks at a slot is one `code h` checks at that slot. -/
theorem checked_slot_wired (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) {x : Msg}
    {r : Fin ((code h).V.k x) → FlockDeploy.Coin} {u : ℕ}
    (hu : ∃ p : Fin (Claims.Sampling.nPU (deploy rel (codeWired h ω keep)) x),
      ((deploy rel (codeWired h ω keep)).checked x r)[p.val]? = some u) :
    ∃ p : Fin (Claims.Sampling.nPU (deploy rel (code h)) x),
      ((deploy rel (code h)).checked x r)[p.val]? = some u := by
  obtain ⟨p, hp⟩ := hu
  have hlt : p.val < Claims.Sampling.nPU (deploy rel (code h)) x :=
    Nat.lt_of_lt_of_eq p.isLt (nPU_wired h ω keep rel x)
  have hp' : ((codeWired h ω keep).checked x r)[p.val]? = some u := hp
  rw [checked_wired h ω keep x r] at hp'
  exact ⟨⟨p.val, hlt⟩, hp'⟩

/-! ## The code facts -/

variable {ω keep}

/-- **Refuses at two stages** at the wired code: an accepting run is `V h`'s, whose statement is in the two-stage
scope side by side (`refusesTwo`), and so at the wired program (`scope_rewired`). -/
theorem refusesTwo_wired (hW : ∀ s d, two h s = some d → (bitsProgs h.ms).Rewires (membersPart h.ms) ω keep)
    (H : Msg → Msg) :
    ((baseWired h ω keep).atOrder (orderWired h ω keep)).RefusesTwo H (twoWired h ω) := by
  intro x Ω P w r hacc
  obtain ⟨s, Pt, d, hx, hP, hd, hin⟩ := refusesTwo h H x Ω P w r hacc
  have hx' : parse h x = some s := hx
  obtain rfl : Pt = membersPart h.ms := partition_eq h (s := s) hP
  have e := scope_rewired h hx' rfl (hW s d hd) hin
  rw [parse_prog h hx'] at e
  exact ⟨{ s with prog := (bitsProgs h.ms).wired ω keep }, (membersPart h.ms).cut ω,
    { d with part := d.part.cut ω },
    congrArg (Option.map fun s : AuditStatement =>
      ({ s with prog := (bitsProgs h.ms).wired ω keep } : AuditStatement)) hx', rfl,
    congrArg (Option.map fun d : Design.Two => ({ d with part := d.part.cut ω } : Design.Two)) hd, e⟩

/-- **The registration is the statement's two-stage design** at the wired code (`regMatchesTwo`): the partitions
without the gates `ω` moves have the partitions' shapes, strata and cells. -/
theorem regMatchesTwo_wired
    (hW : ∀ s d, two h s = some d → (bitsProgs h.ms).Rewires (membersPart h.ms) ω keep) :
    SamplingDeploy.RegMatchesTwo ((baseWired h ω keep).atOrder (orderWired h ω keep)) (reg h)
      (twoWired h ω) := by
  intro x s' Pt' d' hx hP hd _
  obtain ⟨s, d, hs, h2, hin, rfl, rfl, rfl⟩ := wired_stmt h ω keep hx hP hd
  have hW := hW s d h2
  obtain ⟨g, l, hg, hl, hshape, hsd, hT, hk, hcl⟩ := regMatchesTwo h x s (membersPart h.ms) d hs rfl h2 hin
  have hlen := Partition.length_strata_cut (membersPart h.ms) ω
  have hcell : ∀ ρ t, cell ((membersPart h.ms).cut ω) (d.part.cut ω) ρ t = cell (membersPart h.ms) d.part ρ t :=
    cell_cut ω hin.2.2.1 hW.stays hin.2.2.2.1
  refine ⟨g, l, hg, hl, (Partition.shape_cut d.part ω).trans hshape, hsd, hT.trans hlen.symm,
    fun t ht => hk t (lt_of_lt_of_eq ht hlen), fun ρ b hb => ?_⟩
  rcases hcl ρ b hb with ⟨hu, hc⟩ | ⟨hu, hc⟩
  · refine Or.inl ⟨hu, fun t ht => ?_⟩
    obtain ⟨hinj, hce⟩ := hc t (lt_of_lt_of_eq ht hlen)
    exact ⟨hinj, (hcell ρ t).trans hce⟩
  · refine Or.inr ⟨hu, fun t v hv => hc t v ?_⟩
    rw [← hcell ρ t]
    exact hv

/-- **The replay cap holds** at the wired code (`replayCap`): the proof blocks in the sampled replay blocks are the
same blocks. -/
theorem replayCap_wired (hW : ∀ s d, two h s = some d → (bitsProgs h.ms).Rewires (membersPart h.ms) ω keep)
    (H : Msg → Msg) :
    ((baseWired h ω keep).atOrder (orderWired h ω keep)).ReplayCap H (twoWired h ω)
      (rsWired h ω keep) (cap h) := by
  intro x s' Pt' d' hx hP hd _ Ω P w r e hacc
  obtain ⟨s, d, hs, h2, hin, rfl, rfl, rfl⟩ := wired_stmt h ω keep hx hP hd
  have hW := hW s d h2
  have hc := replayCap h H x s (membersPart h.ms) d hs rfl h2 hin Ω P w r e hacc
  have hsb : sampleBlocks ((membersPart h.ms).cut ω) (d.part.cut ω) (rs h x r) =
      sampleBlocks (membersPart h.ms) d.part (rs h x r) :=
    sampleBlocks_cut ω hin.2.2.1 hW.stays hin.2.2.2.1 _
  show (sampleBlocks ((membersPart h.ms).cut ω) (d.part.cut ω) (rs h x r)).card ≤ cap h x
  rw [hsb]
  exact hc

variable (ω keep)

/-- **The wired code's sessions are Flock runs** (`sessionRunsTo`): its verifier, its sessions and their rounds are
`code h`'s, and it checks the units `code h` checks. -/
theorem sessionRunsTo_wired (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (H : Msg → Msg) :
    SessionRunsTo H (deploy rel (codeWired h ω keep)) := by
  intro x Ω P w r u hu
  exact sessionRunsTo h rel H x Ω P w r u (checked_slot_wired h ω keep rel hu)

/-- **Every session of the wired code is in its unit's scope** (`sessionsIn_code`): the sessions and their numbers
are `code h`'s. -/
theorem sessionsIn_wired : SamplingDeploy.SessionsIn (codeWired h ω keep) := sessionsIn_at h

/-! ## The sessions' openings -/

/-- **An accepted run that checks a unit and shows its session passed the session statements' check at the unit's
statement**: its transcript is `seenAt` of its messages of rounds 0, 1 and `stage + 1`, round 0's coin and the second
stage's, `stmtOk` held at them, the session's statement is the unit's as round `stage + 1` states it, and the unit is
one the two stages check on those coins. -/
theorem run_checked {Ω : Type} (P : Prover Ω) (x : Msg) (w : Ω)
    (r : Fin ((deploy _root_.Flock.Readable.relation (code h)).V.k x) → FlockDeploy.Coin) {s : AuditStatement}
    (hx : parse h x = some s) {u : ℕ} {xS : Msg} {vS : Seen}
    (hacc : (run FlockDeploy.sha512 P (deploy _root_.Flock.Readable.relation (code h)).V x w r).out = .acc)
    (hu : ∃ p : Fin (Claims.Sampling.nPU (deploy _root_.Flock.Readable.relation (code h)) x),
      ((deploy _root_.Flock.Readable.relation (code h)).checked x r)[p.val]? = some u)
    (hsess : (deploy _root_.Flock.Readable.relation (code h)).session x u
      (run FlockDeploy.sha512 P (deploy _root_.Flock.Readable.relation (code h)).V x w r).seen = some (xS, vS)) :
    ∃ m₀ m₁ m₂ rest,
      (run FlockDeploy.sha512 P (deploy _root_.Flock.Readable.relation (code h)).V x w r).seen =
        seenAt m₀ m₁ m₂ (r ⟨0, k_pos h x⟩) (stageCoins h x r) rest ∧
      stmtOk h x m₁ m₂ (r ⟨0, k_pos h x⟩) (stageCoins h x r) = true ∧ xS = unitStmt h m₂ u ∧
      u ∈ h.reg.checked (units (base h) x) (src (r ⟨0, k_pos h x⟩)) (stageSrc (stageCoins h x r)) := by
  obtain ⟨m₀, m₁, m₂, rest, hseen, hok⟩ := run_seenAt h FlockDeploy.sha512 P x w r hacc
  obtain ⟨ms, rest', hms, hlen, hses⟩ := (seq h _root_.Flock.Readable.relation).session_acc FlockDeploy.sha512 x P w
    r u ((seq h _root_.Flock.Readable.relation).idx_lt x r u hu) hacc
  have h3 : ((ms.take (h.stage + 2)).map Prod.fst).getD (h.stage + 1) [] = m₂ := by
    have hl : h.stage + 2 ≤ ms.length := hlen ▸ stage_le_off h _
    have hms' : (run FlockDeploy.sha512 P (V h) x w r).msgs = ms ++ rest' := hms
    have hm : (run FlockDeploy.sha512 P (V h) x w r).msgs =
        (seenAt m₀ m₁ m₂ (r ⟨0, k_pos h x⟩) (stageCoins h x r) rest).msgs := congrArg Seen.msgs hseen
    rw [← getD_fst_pre h ms rest' (by omega), ← hms', hm]
    exact seenAt_stmts h
  have hxS : xS = unitStmt h m₂ u := by
    have e : (deploy _root_.Flock.Readable.relation (code h)).session x u
        (run FlockDeploy.sha512 P (deploy _root_.Flock.Readable.relation (code h)).V x w r).seen =
          some (stmtAt h (((ms.take (h.stage + 2)).map Prod.fst).getD (h.stage + 1) []) (templateOf h.ms u),
            cutSeen [] (k₀ h (templateOf h.ms u)) rest') := hses
    rw [hsess, h3] at e
    exact (Prod.mk.inj (Option.some.inj e)).1
  obtain ⟨st, hst⟩ : ∃ st, decode x = some st := by
    cases hdx : decode x with
    | none => simp [SpotFlock.parse, hdx] at hx
    | some st => exact ⟨st, rfl⟩
  obtain ⟨p, hp⟩ := hu
  exact ⟨m₀, m₁, m₂, rest, hseen, hok, hxS,
    (congrArg (u ∈ ·) (checked_eq h x hst r)).mp (List.mem_of_getElem? hp)⟩

variable {ω keep}

/-- **The wired code's session opens each checked proof block at its replay block**: `SessionOpensTwo` at the wired
code, from the session statements' openings at the wired program (`StmtOpensWired`). The reveal is the frame's at the
statement side by side, each opening at the gate `ω` sends it to. -/
theorem sessionOpensTwo_wired
    (hW : ∀ s d, two h s = some d → (bitsProgs h.ms).Rewires (membersPart h.ms) ω keep)
    (hs : StmtOpensWired h ω keep) :
    Claims.SamplingGeneric.SessionOpensTwo FlockDeploy.sha512
      (deploy _root_.Flock.Readable.relation (codeWired h ω keep)) (twoWired h ω) iroots := by
  intro x s' Pt' d' Ω P w r u xS vS wF hx hP hd _ hacc hu hsess hck
  obtain ⟨s, d, hx0, h2, hin, rfl, rfl, rfl⟩ := wired_stmt h ω keep hx hP hd
  have hW := hW s d h2
  obtain ⟨m₀, m₁, m₂, rest, hseen, hok, hxS, hum⟩ :=
    run_checked h P x w r hx0 hacc (checked_slot_wired h ω keep _root_.Flock.Readable.relation hu) hsess
  have hseen' : (run FlockDeploy.sha512 P (deploy _root_.Flock.Readable.relation (codeWired h ω keep)).V x w
      r).seen = seenAt m₀ m₁ m₂ (r ⟨0, k_pos h x⟩) (stageCoins h x r) rest := hseen
  rw [hxS] at hck
  obtain ⟨b, ρ, rb, hb, hρ, hrb, hcor⟩ := hs x s (membersPart h.ms) d m₀ m₁ m₂ (r ⟨0, k_pos h x⟩)
    (stageCoins h x r) rest wF u hx0 rfl h2 hin hok hum hck
  refine ⟨cutBlock ω b, ρ, cutBlock ω rb,
    (Partition.cut_getElem? (membersPart h.ms) ω u).trans (congrArg (Option.map (cutBlock ω)) hb),
    replayOf_cut ω hin.2.2.1 hW.stays hρ,
    (Partition.cut_getElem? d.part ω ρ).trans (congrArg (Option.map (cutBlock ω)) hrb), fun val ι hv => ?_⟩
  rw [hseen'] at hv
  have e := hcor val ι hv
  rw [parse_prog h hx0] at e
  exact e

/-! ## Knowledge soundness -/

/-- **Two-stage knowledge soundness at the wired code**: `ksTwo_stage` at `codeWired h ω keep`, from the map's
rewiring the members' programs where the design evaluates and the session statements' openings at the wired
program. -/
theorem ksTwo_wired (hW : ∀ s d, two h s = some d → (bitsProgs h.ms).Rewires (membersPart h.ms) ω keep)
    (hs : StmtOpensWired h ω keep) :
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (codeWired h ω keep) iroots
      (twoWired h ω) (rsWired h ω keep) (cap h)).KS FlockDeploy.sha512 :=
  SamplingDeploy.ksTwo_stage (baseWired h ω keep) (reg h) (twoWired h ω) (fun _ => L)
    (fun _ => h.stage * L) (fun _ => coinByte) (fun _ => h.stage) (stageOrderWired h ω keep)
    (fun _ => stageByte) iroots (cap h) callsUniform (stageUniform h.stage)
    (regMatchesTwo_wired h hW) (refusesTwo_wired h hW _) (replayCap_wired h hW _) (interiorFixed h _)
    (sessionRunsTo_wired h ω keep _root_.Flock.Readable.relation FlockDeploy.sha512) (sessionsIn_wired h ω keep)
    (sessionOpensTwo_wired h hW hs)

/-- **The session statements open every unit they check at the connected program**: `stmtOpensWired` at the code's
own wiring, the wiring map's rewiring read at the run's own design (`rewires_of_two`) and the addresses from
`AddressesConnected h`, whose condition on the source bits the run's own design decides (`sourcesNamed_of_two`). -/
theorem stmtOpens_connected (hB : BuiltWired h (wireMap h) (returned h)) (hA : AddressesConnected h) :
    StmtOpensWired h (wireMap h) (returned h) := by
  intro x s Pt d m₀ m₁ m₂ c₀ c₁ rest wF u hx hP hd
  exact stmtOpensWired h (wireMap h) (returned h) hB (hA (sourcesNamed_of_two h hd))
    (rewires_of_two hd (wiresFirst_of_two h hd)) x s Pt d m₀ m₁ m₂ c₀ c₁ rest wF u hx hP hd

/-- **SPOT[Flock]'s two-stage knowledge soundness at the connected program**: `CompiledTwo.KS` at the code's own
wiring. Where the design evaluates, the wires' writers come first and every source bit is at a port a link or a wire
names (`wiringOk`), so the wiring map rewires the members' programs (`rewires_of_two`), and the sessions open the
wired program from SPOT's own frame there and the addresses (`stmtOpens_connected`). -/
theorem ksTwo_connected (hB : BuiltWired h (wireMap h) (returned h)) (hA : AddressesConnected h) :
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (codeConnected h) iroots
      (twoConnected h) (rsConnected h) (cap h)).KS FlockDeploy.sha512 :=
  ksTwo_wired h (fun _ _ hd => rewires_of_two hd (wiresFirst_of_two h hd)) (stmtOpens_connected h hB hA)

/-- **SPOT[Flock]'s two-stage exfiltration bound at the connected program**: `CompiledTwo.Exfil` from
`ksTwo_connected` (`CompiledTwo.exfil_of_ks`). -/
theorem exfilTwo_connected (hB : BuiltWired h (wireMap h) (returned h)) (hA : AddressesConnected h) :
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (codeConnected h) iroots
      (twoConnected h) (rsConnected h) (cap h)).Exfil FlockDeploy.sha512 :=
  Claims.SamplingGeneric.CompiledTwo.exfil_of_ks (ksTwo_connected h hB hA)

end SpotFlock
