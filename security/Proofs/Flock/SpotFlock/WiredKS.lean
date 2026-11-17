module

public import Proofs.Flock.SpotFlock.WiredCode
public import Proofs.Flock.SamplingGeneric.TwoCount

/-! SPOT[Flock]'s two-stage knowledge soundness at the wired code (`ksTwo_wired`) and at the code's own wiring (`ksTwo_connected`, `exfilTwo_connected`). -/

@[expose] public section

namespace Claims.Sampling

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

variable {ω keep}

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

theorem sessionRunsTo_wired (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (H : Msg → Msg) :
    SessionRunsTo H (deploy rel (codeWired h ω keep)) := by
  intro x Ω P w r u hu
  exact sessionRunsTo h rel H x Ω P w r u (checked_slot_wired h ω keep rel hu)

theorem sessionsIn_wired : SamplingDeploy.SessionsIn (codeWired h ω keep) := sessionsIn_at h

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

theorem stmtOpens_connected (hB : BuiltWired h (wireMap h) (returned h)) (hA : AddressesConnected h) :
    StmtOpensWired h (wireMap h) (returned h) := by
  intro x s Pt d m₀ m₁ m₂ c₀ c₁ rest wF u hx hP hd
  exact stmtOpensWired h (wireMap h) (returned h) hB (hA (sourcesNamed_of_two h hd))
    (rewires_of_two hd (wiresFirst_of_two h hd)) x s Pt d m₀ m₁ m₂ c₀ c₁ rest wF u hx hP hd

theorem ksTwo_connected (hB : BuiltWired h (wireMap h) (returned h)) (hA : AddressesConnected h) :
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (codeConnected h) iroots
      (twoConnected h) (rsConnected h) (cap h)).KS FlockDeploy.sha512 :=
  ksTwo_wired h (fun _ _ hd => rewires_of_two hd (wiresFirst_of_two h hd)) (stmtOpens_connected h hB hA)

theorem exfilTwo_connected (hB : BuiltWired h (wireMap h) (returned h)) (hA : AddressesConnected h) :
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (codeConnected h) iroots
      (twoConnected h) (rsConnected h) (cap h)).Exfil FlockDeploy.sha512 :=
  Claims.SamplingGeneric.CompiledTwo.exfil_of_ks (ksTwo_connected h hB hA)

end SpotFlock
