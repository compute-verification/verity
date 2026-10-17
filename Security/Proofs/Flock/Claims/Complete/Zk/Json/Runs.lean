import Proofs.Flock.Claims.Complete.Zk.Json.Files
import Proofs.Flock.Claims.Complete.Runs

/-!
# The honest masked session runs

`Runs.lean` at the masked session: the goal `SessionRunsZk`, given `JsonSpec` and `StmtOkZk` (the masked record's
`hello` prints) and one named hypothesis on the masked prover's pieces, `ProverRunsZk`: at every good masked case's
table, the regions fit (`RegionsFit`), and each masked rep returns, on the masked level-0 commitment, the extra claims
at the link's points and the tape's pads committed, from any sends. The table's `--zk` schedule returns
(`zkSchedule_ok`), and through `Hashes.ofSha512` so do the masked level-0 commitment (`commitPZk_ok`) and the pads'
commitment (`innerCommitP_ok`).
-/

namespace FlockComplete.JsonRT

open Flock Transcript FlockComplete FlockDeploy

/-- **Level 0's masked commitment returns** through `Hashes.ofSha512`. -/
theorem commitPZk_ok (h : ByteArray → ByteArray) (ms : MerkleScheme) (lv : Level) (tPad : ℕ) (g : Array F128)
    (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ByteArray) :
    ∃ cm, commitPZk (Hashes.ofSha512 h : Hashes V) ms lv tPad g extra lanePad salt = .ok cm := by
  obtain ⟨hn, hl⟩ := opsWith_ok h ms
  unfold commitPZk commitRowsP
  refine ok_bind (g := id) (mapM_array_ok _ (fun _ => hl _ _) _) fun leaves => ?_
  refine ok_bind (g := id) (forIn_range_ok _ _ _ fun _ r => ?_) fun r => ⟨_, rfl⟩
  dsimp only
  exact ok_bind (g := ForInStep.yield) (mapM_array_ok _ (fun _ => hn _ _) _) fun _ => ⟨_, rfl⟩

/-- **The pads' commitment returns** through `Hashes.ofSha512`. -/
theorem innerCommitP_ok (h : ByteArray → ByteArray) (p : PadsZk) (salt : ℕ → ByteArray) :
    ∃ ic, innerCommitP (Hashes.ofSha512 h : Hashes V) p salt = .ok ic := by
  unfold innerCommitP
  refine ok_bind (g := id) (mapM_array_ok _ (fun _ => ⟨_, rfl⟩) _) fun leaves => ?_
  refine ok_bind (g := id) (forIn_range_ok _ _ _ fun _ r => ?_) fun r => ⟨_, rfl⟩
  dsimp only
  exact ok_bind (g := ForInStep.yield) (mapM_array_ok _ (fun _ => ⟨_, rfl⟩) _) fun _ => ⟨_, rfl⟩

variable {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}

/-- **The masked prover's pieces return**, at every good masked case's table with its `--zk` schedule: the table's
regions fit (`RegionsFit`, so its extra claims return), and each rep returns on the masked level-0 commitment of the
trace's packed table under the case's tape, the extra claims at the link's points, and the tape's pads committed, under
any domain, from any sends. -/
def ProverRunsZk (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Prop :=
  ∀ c : CaseZk H rel, ∀ st v, c.sts = #[st] → Zk.schedule (Setup.ofCircuit st).m = .ok v → RegionsFit st ∧
    ∀ cm e, commitPZk (Hashes.ofSha512 (Claims.hOf H) : Hashes V) (Setup.ofCircuit st).merkle
      (v.1.levels.getD 0 default) v.2 (packed (tracesOf c.sts c.w)[0]! (Setup.ofCircuit st).m) (c.ω.extra 0)
      (c.ω.lanePad 0) (c.ω.salt 0 0 0) = .ok cm →
    (Setup.ofCircuit st).extra (pointsOf c.spec (Claims.coinSeq c.r 0)) = .ok e →
    ∀ rep ic, innerCommitP (Hashes.ofSha512 (Claims.hOf H) : Hashes V)
      (padsOf c.ω 0 rep (Zk.PadLayout.new (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog v.1.k0))
      (c.ω.innerSalt 0 rep) = .ok ic →
    ∀ d l, ∃ pt l', ((repPZk (asksS c.resp) (Hashes.ofSha512 (Claims.hOf H)) (Setup.ofCircuit st) v.1 v.2 rep d
      (tracesOf c.sts c.w)[0]! (packed (tracesOf c.sts c.w)[0]! (Setup.ofCircuit st).m) cm e
      (padsOf c.ω 0 rep (Zk.PadLayout.new (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog v.1.k0)) ic
      (c.ω.extra 0) (c.ω.lanePad 0) (c.ω.salt 0 rep) (c.ω.nonce 0 rep)).run (Transcript.init #[] d)).go l = (.ok pt, l')

/-- **The masked session runs**: the goal `SessionRunsZk`. -/
theorem sessionRunsZk_goal (S : JsonSpec) (hS : StmtOkZk H rel) (hR : ProverRunsZk H rel) : SessionRunsZk H rel := by
  intro c
  obtain ⟨st, hst, hsp, hm1, hm2⟩ := c.table
  obtain ⟨-, -, -, m, kvs, hown, -, -, -, hg⟩ := hS c
  obtain ⟨hz, hc⟩ := S.canon_ok _ hg
  obtain ⟨ct, hct⟩ : ∃ ct, c.spec.coinTree = some ct := Option.isSome_iff_exists.1 c.ready_parts.2.2.2.2
  have ht : c.spec.tables = #[st.tags.table] := by rw [hsp]; rfl
  obtain ⟨v, hv⟩ := zkSchedule_ok (m := (Setup.ofCircuit st).m) hm1 hm2
  obtain ⟨hreg, hRep⟩ := hR c st v hst hv
  obtain ⟨e, he⟩ := extraClaims_ok st hreg (pointsOf c.spec (Claims.coinSeq c.r 0))
  replace he : (Setup.ofCircuit st).extra (pointsOf c.spec (Claims.coinSeq c.r 0)) = .ok e := he
  obtain ⟨cm, hcm⟩ := commitPZk_ok (Claims.hOf H) (Setup.ofCircuit st).merkle (v.1.levels.getD 0 default) v.2
    (packed (tracesOf c.sts c.w)[0]! (Setup.ofCircuit st).m) (c.ω.extra 0) (c.ω.lanePad 0) (c.ω.salt 0 0 0)
  rw [hst] at hcm hRep
  suffices h : Runs (sessionPZk (asksS c.resp) (Hashes.ofSha512 (Claims.hOf H)) c.spec #[Setup.ofCircuit st]
      (tracesOf #[st] c.w) c.ω) #[] fun _ _ => True by
    obtain ⟨⟨record, proofs⟩, bodies, e, -⟩ := h
    refine ⟨record, proofs, bodies, ?_⟩
    unfold sessionSZk
    rw [hst, show tabsOf #[st] = #[Setup.ofCircuit st] by simp [tabsOf]]
    exact e
  unfold sessionPZk
  refine Runs.bind (Q := fun schs l => schs = #[v] ∧ l = #[]) (Runs.mapM_singleton _ _ ?_) fun schs l hq => ?_
  · refine ⟨v, #[], ?_, rfl, rfl⟩
    rw [monadLift_go, hv]
  obtain ⟨rfl, rfl⟩ := hq
  dsimp only
  refine Runs.bind (Q := fun cms l => cms = #[cm] ∧ l = #[]) ?_ fun cms l hq => ?_
  · rw [show Array.range (#[Setup.ofCircuit st] : Array Setup).size = #[0] from rfl]
    refine Runs.mapM_singleton _ _ ⟨cm, #[], ?_, rfl, rfl⟩
    rw [commitPZk_pureS]
    simpa using hcm
  obtain ⟨rfl, rfl⟩ := hq
  refine Runs.bind (Q := fun c1 _ => c1 = coinOf (c.resp 0)) (Runs.of_go (link_go _ _ _ _) rfl) fun c1 l hq => ?_
  subst hq
  have he' : (Setup.ofCircuit st).extra (ptsOf c.spec (coinOf (c.resp 0))) = .ok e := by
    rw [show c.resp 0 = (Claims.coinSeq c.r 0).bytes from rfl, ptsOf_coin]
    exact he
  refine Runs.bind (Q := fun _ _ => True) (Runs.forIn_array _ _ (fun _ _ => True) ?_ trivial) fun b l _ => ?_
  · rintro ⟨i, r, name⟩ hmem ⟨ss, ps⟩ l -
    have hi : i = 0 := by
      rw [repStreams_one ht] at hmem
      simp at hmem
      omega
    subst hi
    dsimp only
    refine Runs.bind (Q := fun ex _ => ex = e) ⟨e, l, ?_, rfl⟩ fun ex l hq => ?_
    · rw [monadLift_go]
      show ((Setup.ofCircuit st).extra (ptsOf c.spec (coinOf (c.resp 0))), l) = _
      rw [he']
    subst ex
    obtain ⟨ic, hic⟩ := innerCommitP_ok (Claims.hOf H)
      (padsOf c.ω 0 r (Zk.PadLayout.new (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog v.1.k0)) (c.ω.innerSalt 0 r)
    refine Runs.bind (Q := fun ic' l' => ic' = ic ∧ l' = l) ⟨ic, l, ?_, rfl, rfl⟩ fun ic' l₁ hq => ?_
    · have := innerCommitP_pureS (Claims.hOf H)
        (padsOf c.ω 0 r (Zk.PadLayout.new (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog v.1.k0))
        (c.ω.innerSalt 0 r) l
      rw [hic] at this
      simpa using this
    rw [hq.1, hq.2]
    have h4 : (Array.map (fun i => packed (tracesOf #[st] c.w)[i]! (#[Setup.ofCircuit st] : Array Setup)[i]!.m)
        (Array.range (#[Setup.ofCircuit st] : Array Setup).size))[0]! =
        packed (tracesOf #[st] c.w)[0]! (Setup.ofCircuit st).m := by simp
    rw [h4, show (#[Setup.ofCircuit st] : Array Setup)[0]! = Setup.ofCircuit st from rfl,
      show (#[v] : Array (Schedule × ℕ))[0]! = v from rfl, show (#[cm] : Array Commit)[0]! = cm from rfl]
    refine Runs.bind (Q := fun _ _ => True) ?_ fun pt l _ => ?_
    · obtain ⟨pt, l', e'⟩ := hRep cm e hcm he r ic hic (c.spec.domain c.spec.tables[0]! r).toUTF8 l
      exact ⟨pt, l', e', trivial⟩
    obtain ⟨p, t⟩ := pt
    exact Runs.pure ⟨_, rfl, trivial⟩
  obtain ⟨ss, ps⟩ := b
  dsimp only
  rw [recordJson_eqZk _ hct hown hc]
  simp only [pure_bind]
  exact Runs.pure trivial

end FlockComplete.JsonRT
