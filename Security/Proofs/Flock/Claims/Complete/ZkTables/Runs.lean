import Proofs.Flock.Claims.Complete.ZkTables.Count
import Proofs.Flock.Claims.Complete.Zk.Json.Runs

/-!
# The honest masked session runs, at several tables

`Zk/Json/Runs.lean` at every table: the goal `SessionRunsJ` (the masked session runs at a good case of several
tables), from the masked record's `hello` printing (`StmtOkJ`, `StmtOkZk`'s statement at several tables) and the masked
prover's pieces returning at each table (`ProverRunsJ`, `ProverRunsZk`'s at table `i`, on table `i`'s trace of its
share). Every table's regions fit (`ready`), so its extra claims return (`JsonRT.extraClaims_ok`); its `--zk` schedule
returns (`zkSchedule_ok`), and through `Hashes.ofSha512` so do its masked level-0 commitment (`commitPZk_ok`) and its
pads' (`innerCommitP_ok`).
-/

namespace FlockComplete.ZkTables

open Flock Transcript FlockComplete FlockDeploy JsonRT
open Lean (Json)

variable {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}

theorem mapM_list_ok_mem {α β : Type} (f : α → V β) :
    ∀ xs : List α, (∀ a ∈ xs, ∃ b, f a = .ok b) → ∃ ys, xs.mapM f = .ok ys
  | [], _ => ⟨[], rfl⟩
  | a :: xs, h => by
    obtain ⟨b, hb⟩ := h a List.mem_cons_self
    obtain ⟨ys, hys⟩ := mapM_list_ok_mem f xs fun x hx => h x (List.mem_cons_of_mem _ hx)
    exact ⟨b :: ys, by rw [List.mapM_cons, hb, hys]; rfl⟩

theorem mapM_array_ok_mem {α β : Type} (f : α → V β) (xs : Array α) (hf : ∀ a ∈ xs, ∃ b, f a = .ok b) :
    ∃ ys, xs.mapM f = .ok ys := by
  rw [Array.mapM_eq_mapM_toList]
  obtain ⟨ys, h⟩ := mapM_list_ok_mem f xs.toList fun a ha => hf a (Array.mem_toList_iff.1 ha)
  exact ⟨ys.toArray, by rw [h]; rfl⟩

section Several

variable (H rel)

/-- **The masked session runs** at every good case of several tables. -/
def SessionRunsJ : Prop :=
  ∀ c : Case H rel, 1 < c.sts.size →
    ∃ record proofs bodies, sessionS H c.resp c.spec c.sts c.w c.ω = (.ok (record, proofs), bodies)

/-- **What the masked record needs of the statement** (`StmtOkZk`), at every good case of several tables: a non-empty
`Σ`, a printable draw that decodes to the statement's instances, the statement's draw the one it was set up at, and a
`hello` that parses to an object with a `coins` object and prints. -/
def StmtOkJ : Prop :=
  ∀ c : Case H rel, 1 < c.sts.size → 0 < c.spec.sigma.size ∧ DrawOk c.spec ∧ c.spec.unitDraw = c.s.read.draw ∧
    ∃ m kvs : Std.TreeMap.Raw String Json compare, Json.parse c.spec.hello = .ok (.obj m) ∧ m.WF ∧ kvs.WF ∧
      m.get? "coins" = some (.obj kvs) ∧ Good (helloJ (.obj m))

/-- **The masked prover's pieces return** (`ProverRunsZk`), at every table `i` of every good case of several tables,
with its `--zk` schedule: each rep returns on table `i`'s masked level-0 commitment of its trace's packed table under
the tape's level `i`, its extra claims at the link's points, and the tape's pads committed, under any domain, from any
sends. -/
def ProverRunsJ : Prop :=
  ∀ c : Case H rel, 1 < c.sts.size → ∀ i (_ : i < c.sts.size), ∀ v,
    Zk.schedule (Setup.ofCircuit c.sts[i]).m = .ok v →
    ∀ cm e, commitPZk (Hashes.ofSha512 (Claims.hOf H) : Hashes V) (Setup.ofCircuit c.sts[i]).merkle
      (v.1.levels.getD 0 default) v.2 (packed (traces c.sts c.w)[i]! (Setup.ofCircuit c.sts[i]).m) (c.ω.extra i)
      (c.ω.lanePad i) (c.ω.salt i 0 0) = .ok cm →
    (Setup.ofCircuit c.sts[i]).extra (pointsOf c.spec (Claims.coinSeq c.r 0)) = .ok e →
    ∀ rep ic, innerCommitP (Hashes.ofSha512 (Claims.hOf H) : Hashes V)
      (padsOf c.ω i rep (Zk.PadLayout.new (Setup.ofCircuit c.sts[i]).m (Setup.ofCircuit c.sts[i]).kLog v.1.k0))
      (c.ω.innerSalt i rep) = .ok ic →
    ∀ d l, ∃ pt l', ((repPZk (asksS c.resp) (Hashes.ofSha512 (Claims.hOf H)) (Setup.ofCircuit c.sts[i]) v.1 v.2 rep
      d (traces c.sts c.w)[i]! (packed (traces c.sts c.w)[i]! (Setup.ofCircuit c.sts[i]).m) cm e
      (padsOf c.ω i rep (Zk.PadLayout.new (Setup.ofCircuit c.sts[i]).m (Setup.ofCircuit c.sts[i]).kLog v.1.k0)) ic
      (c.ω.extra i) (c.ω.lanePad i) (c.ω.salt i rep) (c.ω.nonce i rep)).run (Transcript.init #[] d)).go l =
      (.ok pt, l')

end Several

/-- **The masked session runs** at several tables: the goal `SessionRunsJ`. -/
theorem sessionRunsJ_goal (S : JsonSpec) (hS : StmtOkJ H rel) (hR : ProverRunsJ H rel) : SessionRunsJ H rel := by
  intro c hc
  obtain ⟨-, -, -, m, kvs, hown, -, -, -, hg⟩ := hS c hc
  obtain ⟨hz, hcn⟩ := S.canon_ok _ hg
  obtain ⟨ct, hct⟩ : ∃ ct, c.spec.coinTree = some ct := Option.isSome_iff_exists.1 c.ready_parts.2.2.2.2
  have hfit := c.ready_parts.2.2.1
  obtain ⟨-, -, hspec, hall⟩ := c.tables
  have htabs : (tabsOf c.sts).size = c.sts.size := by simp [tabsOf]
  have htab : ∀ i (hi : i < c.sts.size), (tabsOf c.sts)[i]! = Setup.ofCircuit c.sts[i] := fun i hi => by
    rw [getElem!_pos (tabsOf c.sts) i (by rw [htabs]; exact hi)]
    simp [tabsOf]
  have htp : ∀ i (hi : i < c.sts.size), (tablesP (tabsOf c.sts) (traces c.sts c.w))[i]! =
      packed (traces c.sts c.w)[i]! (Setup.ofCircuit c.sts[i]).m := fun i hi => by
    unfold tablesP
    rw [getElem!_pos _ i (by simpa [htabs] using hi)]
    simp only [Array.getElem_map, Array.getElem_range, htab i hi]
  obtain ⟨schs, hschs⟩ : ∃ schs, (schedsPZk (tabsOf c.sts) : V _) = .ok schs := by
    unfold schedsPZk
    refine mapM_array_ok_mem _ _ fun st hst => ?_
    obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.1 hst
    simp only [tabsOf, Array.size_map] at hj
    obtain ⟨h1, h2, -⟩ := hall j hj
    simp only [tabsOf, Array.getElem_map, Setup.ofCircuit, monadLift_self]
    exact zkSchedule_ok h1 h2
  have hsch := hschs
  unfold schedsPZk at hsch
  obtain ⟨hs1, hs2⟩ := FlockSoundness.Discharge.Exec.array_mapM_ok_getElem hsch
  have hv : ∀ i (hi : i < c.sts.size), Zk.schedule (Setup.ofCircuit c.sts[i]).m = .ok schs[i]! := by
    intro j hj
    have hv := hs2 j (by rw [htabs]; exact hj) (by rw [hs1, htabs]; exact hj)
    simp only [tabsOf, Array.getElem_map, monadLift_self] at hv
    rw [getElem!_pos schs j (by rw [hs1, htabs]; exact hj)]
    exact hv
  obtain ⟨cms, hcms⟩ : ∃ cms, (commitsPZk (Hashes.ofSha512 (Claims.hOf H)) (tabsOf c.sts) schs
      (tablesP (tabsOf c.sts) (traces c.sts c.w)) c.ω : V _) = .ok cms := by
    unfold commitsPZk
    exact mapM_array_ok _ (fun i => commitPZk_ok _ _ _ _ _ _ _ _) _
  have hcm := hcms
  unfold commitsPZk at hcm
  obtain ⟨hc1, hc2⟩ := FlockSoundness.Discharge.Exec.range_mapM_ok hcm
  have hcmi : ∀ i (hi : i < c.sts.size), commitPZk (Hashes.ofSha512 (Claims.hOf H) : Hashes V)
      (Setup.ofCircuit c.sts[i]).merkle (schs[i]!.1.levels.getD 0 default) schs[i]!.2
      (packed (traces c.sts c.w)[i]! (Setup.ofCircuit c.sts[i]).m) (c.ω.extra i) (c.ω.lanePad i) (c.ω.salt i 0 0) =
      .ok cms[i]! := by
    intro j hj
    have := hc2 j (by rw [hc1, htabs]; exact hj)
    rw [htab j hj, htp j hj] at this
    rw [getElem!_pos cms j (by rw [hc1, htabs]; exact hj)]
    exact this
  suffices h : Runs (sessionPZk (asksS c.resp) (Hashes.ofSha512 (Claims.hOf H)) c.spec (tabsOf c.sts)
      (traces c.sts c.w) c.ω) #[] fun _ _ => True by
    obtain ⟨⟨record, proofs⟩, bodies, e, -⟩ := h
    exact ⟨record, proofs, bodies, e⟩
  rw [sessionPZk_eq]
  refine Runs.bind (Q := fun s l => s = schs ∧ l = #[])
    ⟨schs, #[], by rw [schedsPZk_quiet _ #[], hschs], rfl, rfl⟩ fun s l hq => ?_
  obtain ⟨hs, hl⟩ := hq
  subst s l
  refine Runs.bind (Q := fun s l => s = cms ∧ l = #[])
    ⟨cms, #[], by rw [commitsPZk_quiet _ _ _ _ _ #[], hcms], rfl, rfl⟩ fun s l hq => ?_
  obtain ⟨hs, hl⟩ := hq
  subst s l
  refine Runs.bind (Q := fun c1 _ => c1 = coinOf (c.resp 0)) (Runs.of_go (link_go _ _ _ _) rfl) fun c1 l hq => ?_
  subst hq
  unfold afterLinkPZk
  refine Runs.bind (Q := fun _ _ => True) (Runs.forIn_array _ _ (fun _ _ => True) ?_ trivial) fun b l _ => ?_
  · rintro ⟨i, r, name⟩ hmem ⟨ss, ps⟩ l -
    have hi : i < c.sts.size := by
      rw [← Array.mem_toList_iff, FlockDeploy.repStreams_toList, List.mem_map] at hmem
      obtain ⟨k, hk, he⟩ := hmem
      rw [List.mem_range, hspec] at hk
      simp only [Prod.mk.injEq] at he
      obtain ⟨rfl, -, -⟩ := he
      omega
    dsimp only
    obtain ⟨e, he⟩ := extraClaims_ok c.sts[i] (hfit _ (Array.getElem_mem hi)) (ptsOf c.spec (coinOf (c.resp 0)))
    rw [htab i hi]
    refine Runs.bind (Q := fun ex _ => ex = e) ⟨e, l, ?_, rfl⟩ fun ex l hq => ?_
    · rw [monadLift_go]
      show ((Setup.ofCircuit c.sts[i]).extra (ptsOf c.spec (coinOf (c.resp 0))), l) = _
      rw [show (Setup.ofCircuit c.sts[i]).extra = c.sts[i].extraClaims from rfl, he]
    subst ex
    obtain ⟨ic, hic⟩ := innerCommitP_ok (Claims.hOf H) (padsOf c.ω i r (Zk.PadLayout.new (Setup.ofCircuit c.sts[i]).m
      (Setup.ofCircuit c.sts[i]).kLog schs[i]!.1.k0)) (c.ω.innerSalt i r)
    refine Runs.bind (Q := fun ic' l' => ic' = ic ∧ l' = l) ⟨ic, l, ?_, rfl, rfl⟩ fun ic' l₁ hq => ?_
    · have := innerCommitP_pureS (Claims.hOf H) (padsOf c.ω i r (Zk.PadLayout.new (Setup.ofCircuit c.sts[i]).m
        (Setup.ofCircuit c.sts[i]).kLog schs[i]!.1.k0)) (c.ω.innerSalt i r) l
      rw [hic] at this
      simpa using this
    rw [hq.1, hq.2, htp i hi]
    refine Runs.bind (Q := fun _ _ => True) ?_ fun pt l _ => ?_
    · have he' : (Setup.ofCircuit c.sts[i]).extra (pointsOf c.spec (Claims.coinSeq c.r 0)) = .ok e := by
        rw [← ptsOf_coin]
        exact he
      obtain ⟨pt, l', e'⟩ := hR c hc i hi schs[i]! (hv i hi) cms[i]! e (hcmi i hi) he' r ic hic
        (c.spec.domain c.spec.tables[i]! r).toUTF8 l
      exact ⟨pt, l', e', trivial⟩
    obtain ⟨p, t⟩ := pt
    exact Runs.pure ⟨_, rfl, trivial⟩
  obtain ⟨ss, ps⟩ := b
  dsimp only
  rw [recordJson_eqZk _ hct hown hcn]
  simp only [pure_bind]
  exact Runs.pure trivial

end FlockComplete.ZkTables
