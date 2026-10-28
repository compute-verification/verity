import Proofs.Flock.Claims.Complete.Zk.Json.Files
import Proofs.Flock.Claims.Complete.Runs

/-!
# The honest masked session runs

`Runs.lean` at the masked session: the goal `SessionRunsZk`, given `JsonSpec` and `StmtOkZk` (the masked record's
`hello` prints) and one named hypothesis on the masked prover's pieces, `ProverRunsZk`: at every good masked case's
table, on the masked level-0 commitment, the extra claims at the link's points (which return, the table's regions
fitting by `readyZk`) and the tape's pads committed, rep `0` returns from the link's send and rep `1` from where rep `0`
stopped (`repRunZk`), so
each draws the case's coins at the session's rounds, where `¬ BadZk` passes the masked prover's coin checks. The table's
`--zk` schedule returns (`zkSchedule_ok`), and through `Hashes.ofSha512` so do the masked level-0 commitment
(`commitPZk_ok`) and the pads' commitment (`innerCommitP_ok`); the session walks its two rep streams in order
(`Runs.forIn_idx`).
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

/-- Rep `rep` of a good masked case's one table as the session runs it, from its stream's domain: on the masked level-0
commitment `cm`, the extra claims `e` and the pads' commitment `ic`, its rank check at the earlier claim points
`prior` over the table's mask words. -/
def repRunZk (c : CaseZk H rel) (st : Flock.Stmt) (v : Schedule × ℕ) (cm : Commit) (e : Array Claim) (rep : ℕ)
    (ic : InnerCommit) (prior : Array (List F128)) : PS (Zk.ZkProof × Transcript) :=
  (repPZk (asksS c.resp) (Hashes.ofSha512 (Claims.hOf H)) (Setup.ofCircuit st) v.1 v.2 rep
    (c.spec.domain c.spec.tables[0]! rep).toUTF8 (tracesOf c.sts c.w)[0]!
    (packed (tracesOf c.sts c.w)[0]! (Setup.ofCircuit st).m) cm e
    (padsOf c.ω 0 rep (Zk.PadLayout.new (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog v.1.k0)) ic
    (c.ω.extra 0) (c.ω.lanePad 0) (c.ω.salt 0 rep) (c.ω.nonce 0 rep) prior
    (FlockSoundness.Discharge.Composed.maskWordsOf st)).run
    (Transcript.init #[] (c.spec.domain c.spec.tables[0]! rep).toUTF8)

/-- **The masked prover's pieces return**, at every good masked case's table with its `--zk` schedule: on the masked
level-0 commitment of the trace's packed table under the case's tape, the extra claims at the link's points, and the
tape's pads committed, rep `0` returns from the link's send and rep `1` from where rep `0` stopped, its rank check
at rep `0`'s claim points too. The reps' coins are the case's at the session's rounds, where the masked prover's coin
checks (level 0's pair, the inner `β`, the rank checks) pass by `¬ BadZk`. -/
def ProverRunsZk (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Prop :=
  ∀ c : CaseZk H rel, ∀ st v, c.sts = #[st] → Zk.schedule (Setup.ofCircuit st).m = .ok v →
    ∀ cm e, commitPZk (Hashes.ofSha512 (Claims.hOf H) : Hashes V) (Setup.ofCircuit st).merkle
      (v.1.levels.getD 0 default) v.2 (packed (tracesOf c.sts c.w)[0]! (Setup.ofCircuit st).m) (c.ω.extra 0)
      (c.ω.lanePad 0) (c.ω.salt 0 0 0) = .ok cm →
    (Setup.ofCircuit st).extra (pointsOf c.spec (Claims.coinSeq c.r 0)) = .ok e →
    ∀ ic : ℕ → InnerCommit, (∀ rep, innerCommitP (Hashes.ofSha512 (Claims.hOf H) : Hashes V)
      (padsOf c.ω 0 rep (Zk.PadLayout.new (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog v.1.k0))
      (c.ω.innerSalt 0 rep) = .ok (ic rep)) →
    ∀ l₀ : Array Claims.Msg, l₀.size = 1 → ∃ pt₀ l₁, (repRunZk c st v cm e 0 (ic 0) #[]).go l₀ = (.ok pt₀, l₁) ∧
      ∃ pt₁ l₂, (repRunZk c st v cm e 1 (ic 1) (MaskRank.roundPoints st.m st.c.kLog pt₀.2.rounds)).go l₁ =
        (.ok pt₁, l₂)

theorem Runs.weaken {α : Type} {y : PS α} {l : Array Claims.Msg} {P Q : α → Array Claims.Msg → Prop}
    (h : Runs y l P) (hPQ : ∀ a l', P a l' → Q a l') : Runs y l Q := by
  obtain ⟨a, l', e, hp⟩ := h
  exact ⟨a, l', e, hPQ a l' hp⟩

/-- A loop over a list runs when its step at each index `k` runs from the invariant at `j + k` to the invariant at
`j + k + 1`. -/
theorem Runs.forIn_idx {α β : Type} (f : α → β → PS (ForInStep β)) (inv : ℕ → β → Array Claims.Msg → Prop) :
    ∀ (xs : List α) (j : ℕ), (∀ k a, xs[k]? = some a → ∀ b l, inv (j + k) b l →
      Runs (f a b) l fun r l' => ∃ b', r = .yield b' ∧ inv (j + k + 1) b' l') →
    ∀ b l, inv j b l → Runs (forIn xs b f) l (inv (j + xs.length))
  | [], j, _, b, l, h => Runs.pure (by simpa using h)
  | x :: xs, j, hf, b, l, h => by
    rw [List.forIn_cons]
    refine Runs.bind (hf 0 x rfl b l (by simpa using h)) fun r l' hr => ?_
    obtain ⟨b', rfl, hi⟩ := hr
    have ih := Runs.forIn_idx f inv xs (j + 1) (fun k a ha b l hb => by
      have := hf (k + 1) a (by simpa using ha) b l (by rw [show j + (k + 1) = j + 1 + k by omega]; exact hb)
      rw [show j + (k + 1) + 1 = j + 1 + k + 1 by omega] at this
      exact this) b' l' (by simpa using hi)
    rw [show j + (x :: xs).length = j + 1 + xs.length by simp; omega]
    exact ih

/-- **The masked session runs**: the goal `SessionRunsZk`. -/
theorem sessionRunsZk_goal (S : Claims.JsonSpec printJ Flock.canon) (hS : StmtOkZk H rel)
    (hR : ProverRunsZk H rel) : SessionRunsZk H rel := by
  intro c
  obtain ⟨st, hst, hsp, hm1, hm2⟩ := c.table
  obtain ⟨-, -, -, m, kvs, hown, -, -, -, hg, ha, -⟩ := hS c
  obtain ⟨hz, hc⟩ := S.canon_ok _ hg ha
  obtain ⟨ct, hct⟩ : ∃ ct, c.spec.coinTree = some ct := Option.isSome_iff_exists.1 c.ready_parts.2.2.2.2
  have ht : c.spec.tables = #[st.tags.table] := by rw [hsp]; rfl
  obtain ⟨v, hv⟩ := zkSchedule_ok (m := (Setup.ofCircuit st).m) hm1 hm2
  have hRep := hR c st v hst hv
  obtain ⟨e, he⟩ := extraClaims_ok st (c.ready_fits.1 st (by rw [hst]; simp)) (pointsOf c.spec (Claims.coinSeq c.r 0))
  replace he : (Setup.ofCircuit st).extra (pointsOf c.spec (Claims.coinSeq c.r 0)) = .ok e := he
  obtain ⟨cm, hcm⟩ := commitPZk_ok (Claims.hOf H) (Setup.ofCircuit st).merkle (v.1.levels.getD 0 default) v.2
    (packed (tracesOf c.sts c.w)[0]! (Setup.ofCircuit st).m) (c.ω.extra 0) (c.ω.lanePad 0) (c.ω.salt 0 0 0)
  rw [hst] at hcm hRep
  suffices h : Runs (sessionPZk (asksS c.resp) (Hashes.ofSha512 (Claims.hOf H)) c.spec #[Setup.ofCircuit st]
      (MaskRank.masksOf #[st]) (tracesOf #[st] c.w) c.ω) #[] fun _ _ => True by
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
  refine Runs.bind (Q := fun c1 l => c1 = coinOf (c.resp 0) ∧ l.size = 1) (Runs.of_go (link_go _ _ _ _)
    ⟨rfl, rfl⟩) fun c1 l₀ hq => ?_
  obtain ⟨rfl, hl₀⟩ := hq
  have he' : (Setup.ofCircuit st).extra (ptsOf c.spec (coinOf (c.resp 0))) = .ok e := by
    rw [show c.resp 0 = (Claims.coinSeq c.r 0).bytes from rfl, ptsOf_coin]
    exact he
  let ic : ℕ → InnerCommit := fun r => (innerCommitP_ok (Claims.hOf H)
    (padsOf c.ω 0 r (Zk.PadLayout.new (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog v.1.k0))
    (c.ω.innerSalt 0 r)).choose
  have hic : ∀ r, innerCommitP (Hashes.ofSha512 (Claims.hOf H) : Hashes V)
      (padsOf c.ω 0 r (Zk.PadLayout.new (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog v.1.k0))
      (c.ω.innerSalt 0 r) = .ok (ic r) := fun r => (innerCommitP_ok _ _ _).choose_spec
  have hRun := hRep cm e hcm he ic hic
  let inv : ℕ → Array (Array Round) × Array ByteArray → Array Claims.Msg → Prop := fun j b l =>
    (j = 0 → l.size = 1) ∧ (j = 1 → ∃ l₀ pt₀, l₀.size = 1 ∧
      (repRunZk c st v cm e 0 (ic 0) #[]).go l₀ = (.ok pt₀, l) ∧ b.1.back! = pt₀.2.rounds)
  rw [repStreams_one ht, ← Array.forIn_toList]
  refine Runs.bind (Q := fun _ _ => True) ?_ fun b l _ => ?_
  · refine Runs.weaken (Runs.forIn_idx _ inv _ 0 ?_ _ _ ⟨fun _ => hl₀, by omega⟩) fun _ _ _ => trivial
    intro j ⟨i, k, name⟩ ha ⟨ss, ps⟩ l hinv
    have hk : j < 2 ∧ i = 0 ∧ k = j := by
      rcases j with _ | _ | j <;> simp at ha <;> omega
    obtain ⟨hk, rfl, rfl⟩ := hk
    dsimp only
    obtain ⟨pt, l', e', hnext⟩ : ∃ pt l',
        (repRunZk c st v cm e k (ic k) (priorOf (Setup.ofCircuit st) ss k)).go l = (.ok pt, l') ∧
        ∀ ps', inv (0 + k + 1) (ss.push pt.2.rounds, ps') l' := by
      rcases Nat.lt_succ_iff_lt_or_eq.1 hk with h0 | rfl
      · obtain rfl : k = 0 := by omega
        obtain ⟨pt₀, l₁, e₀, -⟩ := hRun l (hinv.1 rfl)
        refine ⟨pt₀, l₁, by simpa [priorOf] using e₀, fun _ => ⟨fun h => absurd h (by decide), fun _ =>
          ⟨l, pt₀, hinv.1 rfl, e₀, by simp⟩⟩⟩
      · obtain ⟨l₀', pt₀, hl₀', e₀, hback⟩ := hinv.2 rfl
        obtain ⟨pt₀', l₁, e₀', pt₁, l₂, e₁⟩ := hRun l₀' hl₀'
        have h01 := e₀.symm.trans e₀'
        simp only [Prod.mk.injEq, Except.ok.injEq] at h01
        rw [← h01.2, ← h01.1] at e₁
        refine ⟨pt₁, l₂, ?_, fun _ => ⟨fun h => absurd h (by decide), fun h => absurd h (by decide)⟩⟩
        have hp : priorOf (Setup.ofCircuit st) ss 1 = MaskRank.roundPoints st.m st.c.kLog pt₀.2.rounds := by
          simp only [priorOf, ite_true]
          rw [hback]
          rfl
        rw [hp]
        exact e₁
    refine Runs.bind (Q := fun ex l'' => ex = e ∧ l'' = l) ⟨e, l, ?_, rfl, rfl⟩ fun ex l'' hq => ?_
    · rw [monadLift_go]
      show ((Setup.ofCircuit st).extra (ptsOf c.spec (coinOf (c.resp 0))), l) = _
      rw [he']
    obtain ⟨rfl, hl⟩ := hq
    subst l''
    refine Runs.bind (Q := fun ic' l'' => ic' = ic k ∧ l'' = l) ⟨ic k, l, ?_, rfl, rfl⟩ fun ic' l₁ hq => ?_
    · have := innerCommitP_pureS (Claims.hOf H)
        (padsOf c.ω 0 k (Zk.PadLayout.new (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog v.1.k0))
        (c.ω.innerSalt 0 k) l
      rw [hic k] at this
      simpa using this
    rw [hq.1, hq.2]
    have h4 : (Array.map (fun i => packed (tracesOf #[st] c.w)[i]! (#[Setup.ofCircuit st] : Array Setup)[i]!.m)
        (Array.range (#[Setup.ofCircuit st] : Array Setup).size))[0]! =
        packed (tracesOf #[st] c.w)[0]! (Setup.ofCircuit st).m := by simp
    rw [h4, show (#[Setup.ofCircuit st] : Array Setup)[0]! = Setup.ofCircuit st from rfl,
      show (#[v] : Array (Schedule × ℕ))[0]! = v from rfl, show (#[cm] : Array Commit)[0]! = cm from rfl]
    rw [show (MaskRank.masksOf #[st])[0]! = FlockSoundness.Discharge.Composed.maskWordsOf st by
      simp [MaskRank.masksOf]]
    refine Runs.bind (Q := fun pt' l'' => pt' = pt ∧ l'' = l') ⟨pt, l', ?_, rfl, rfl⟩ fun pt' l'' hq => ?_
    · unfold repRunZk at e'
      rw [hst] at e'
      exact e'
    obtain ⟨rfl, rfl⟩ := hq
    obtain ⟨p, t⟩ := pt'
    exact Runs.pure ⟨_, rfl, hnext _⟩
  obtain ⟨ss, ps⟩ := b
  dsimp only
  rw [recordJson_eqZk _ hct hown hc]
  simp only [pure_bind]
  exact Runs.pure trivial

end FlockComplete.JsonRT
