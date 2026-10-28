import Proofs.Flock.Claims.Complete.ZkTables.Count
import Proofs.Flock.Claims.Complete.Zk.Json.Runs
import Proofs.Flock.Claims.Complete.Json.Canon

/-!
# The honest masked session runs, at several tables

`Zk/Json/Runs.lean` at every table: the masked session runs at a good case (`Case.sessionRuns`; at several tables the
goal `SessionRunsJ`), from the masked record's `hello` printing (`StmtAt`, `StmtOkZk`'s statement at the case; at
every case of several tables `StmtOkJ`) and the masked prover's pieces returning (`ProverRunsAt`, `ProverRunsZk`'s at
the case: on table `i`'s trace of its share, the session's `2·|tables|` reps each return from where the one before
stopped, `repRunJ`; at every case of several tables `ProverRunsJ`). Every table's regions fit
(`ready`), so its extra claims return (`JsonRT.extraClaims_ok`); its `--zk` schedule returns (`zkSchedule_ok`), and
through `Hashes.ofSha512` so do its masked level-0 commitment (`commitPZk_ok`) and its pads' (`innerCommitP_ok`); the
session walks its rep streams in order (`JsonRT.Runs.forIn_idx`).
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

/-- **What the masked record needs of the statement** (`StmtOkZk`), at a good case: a non-empty `Σ`, a printable ASCII
draw that decodes to the statement's instances, the statement's draw the one it was set up at, a `hello` that parses to
an object with a `coins` object and prints, ASCII, and an ASCII coin tree spec. -/
def StmtAt (c : Case H rel) : Prop :=
  0 < c.spec.sigma.size ∧ DrawOk c.spec ∧ c.spec.unitDraw = c.s.read.draw ∧
    ∃ m kvs : Std.TreeMap.Raw String Json compare, Flock.JsonParse.parse c.spec.hello = .ok (.obj m) ∧ m.WF ∧ kvs.WF ∧
      m.get? "coins" = some (.obj kvs) ∧ Good (helloJ (.obj m)) ∧ Ascii (helloJ (.obj m)) ∧
      ∀ ct, c.spec.coinTree = some ct → Ascii ct.toJson

variable (H rel) in
/-- Rep `rep` of table `i` of a good case as the session runs it, from its stream's domain: on the table's `--zk`
schedule `v`, its masked level-0 commitment `cm`, its extra claims `e` and its pads' commitment `ic`, its rank check at
the earlier claim points `prior` over the table's mask words. -/
def repRunJ (c : Case H rel) (i : ℕ) (v : Schedule × ℕ) (cm : Commit) (e : Array Claim) (rep : ℕ)
    (ic : InnerCommit) (prior : Array (List F128)) : PS (Zk.ZkProof × Transcript) :=
  (repPZk (asksS c.resp) (Hashes.ofSha512 (Claims.hOf H)) (Setup.ofCircuit c.sts[i]!) v.1 v.2 rep
    (c.spec.domain c.spec.tables[i]! rep).toUTF8 (traces c.sts c.w)[i]!
    (packed (traces c.sts c.w)[i]! (Setup.ofCircuit c.sts[i]!).m) cm e
    (padsOf c.ω i rep (Zk.PadLayout.new (Setup.ofCircuit c.sts[i]!).m (Setup.ofCircuit c.sts[i]!).kLog v.1.k0)) ic
    (c.ω.extra i) (c.ω.lanePad i) (c.ω.salt i rep) (c.ω.nonce i rep) prior
    (FlockSoundness.Discharge.Composed.maskWordsOf c.sts[i]!)).run
    (Transcript.init #[] (c.spec.domain c.spec.tables[i]! rep).toUTF8)

/-- The claim points the session's `k`th rep checks ahead of its own (`priorOf`), the reps having returned `pts`: at
a table's rep `1`, its rep `0`'s, read off its rounds. -/
def priorJ (c : Case H rel) (pts : ℕ → Zk.ZkProof × Transcript) (k : ℕ) : Array (List F128) :=
  if k % 2 = 1 then MaskRank.roundPoints c.sts[k / 2]!.m c.sts[k / 2]!.c.kLog (pts (k - 1)).2.rounds else #[]

/-- **The masked prover's pieces return** (`ProverRunsZk`), at a good case: on each table `i`'s `--zk` schedule `v i`,
its masked level-0 commitment `cm i` of its trace's packed table under the tape's level `i`, its extra claims `e i` at
the link's points and its pads' commitments `ic i`, the session's `2·|tables|` reps in its rep streams' order (table
`k / 2`'s rep `k % 2`) each return (`pts k`) from where the one before stopped, the first from the link's send, a rep
`1`'s rank check at its rep `0`'s claim points too (`priorJ`). The reps' coins are the case's at the session's rounds,
where the masked prover's coin checks pass by `¬ BadZk`. -/
def ProverRunsAt (c : Case H rel) : Prop :=
  ∀ (v : ℕ → Schedule × ℕ) (cm : ℕ → Commit) (e : ℕ → Array Claim)
    (ic : ℕ → ℕ → InnerCommit), (∀ i < c.sts.size, Zk.schedule (Setup.ofCircuit c.sts[i]!).m = .ok (v i) ∧
      commitPZk (Hashes.ofSha512 (Claims.hOf H) : Hashes V) (Setup.ofCircuit c.sts[i]!).merkle
        ((v i).1.levels.getD 0 default) (v i).2 (packed (traces c.sts c.w)[i]! (Setup.ofCircuit c.sts[i]!).m)
        (c.ω.extra i) (c.ω.lanePad i) (c.ω.salt i 0 0) = .ok (cm i) ∧
      (Setup.ofCircuit c.sts[i]!).extra (pointsOf c.spec (Claims.coinSeq c.r 0)) = .ok (e i) ∧
      ∀ rep, innerCommitP (Hashes.ofSha512 (Claims.hOf H) : Hashes V) (padsOf c.ω i rep
        (Zk.PadLayout.new (Setup.ofCircuit c.sts[i]!).m (Setup.ofCircuit c.sts[i]!).kLog (v i).1.k0))
        (c.ω.innerSalt i rep) = .ok (ic i rep)) →
    ∀ l₀ : Array Claims.Msg, l₀.size = 1 → ∃ (ls : ℕ → Array Claims.Msg) (pts : ℕ → Zk.ZkProof × Transcript),
      ls 0 = l₀ ∧ ∀ k < 2 * c.sts.size,
      (repRunJ H rel c (k / 2) (v (k / 2)) (cm (k / 2)) (e (k / 2)) (k % 2) (ic (k / 2) (k % 2))
        (priorJ c pts k)).go (ls k) = (.ok (pts k), ls (k + 1))

section Several

variable (H rel)

/-- **The masked session runs** at every good case of several tables. -/
def SessionRunsJ : Prop :=
  ∀ c : Case H rel, 1 < c.sts.size →
    ∃ record proofs bodies, sessionS H c.resp c.spec c.sts c.w c.ω = (.ok (record, proofs), bodies)

/-- **What the masked record needs of the statement** (`StmtOkZk`), at every good case of several tables. -/
def StmtOkJ : Prop := ∀ c : Case H rel, 1 < c.sts.size → StmtAt c

/-- **The masked prover's pieces return** (`ProverRunsZk`), at every good case of several tables. -/
def ProverRunsJ : Prop := ∀ c : Case H rel, 1 < c.sts.size → ProverRunsAt c

end Several

/-- **The masked session runs** at a good case where the statement's `hello` prints and the prover's pieces
return. -/
theorem Case.sessionRuns (c : Case H rel) (hS : StmtAt c) (hR : ProverRunsAt c) :
    ∃ record proofs bodies, sessionS H c.resp c.spec c.sts c.w c.ω = (.ok (record, proofs), bodies) := by
  obtain ⟨-, -, -, m, kvs, hown, -, -, -, hg, ha, -⟩ := hS
  obtain ⟨hz, hcn⟩ := FlockComplete.JsonRT.canon_ok _ hg ha
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
      (MaskRank.masksOf c.sts) (traces c.sts c.w) c.ω) #[] fun _ _ => True by
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
  refine Runs.bind (Q := fun c1 l => c1 = coinOf (c.resp 0) ∧ l.size = 1) (Runs.of_go (link_go _ _ _ _)
    ⟨rfl, rfl⟩) fun c1 l₀ hq => ?_
  obtain ⟨rfl, hl₀⟩ := hq
  have hst : ∀ i (hi : i < c.sts.size), c.sts[i]! = c.sts[i] := fun i hi => getElem!_pos c.sts i hi
  have htab' : ∀ i (hi : i < c.sts.size), (tabsOf c.sts)[i]! = Setup.ofCircuit c.sts[i]! := fun i hi => by
    rw [hst i hi]
    exact htab i hi
  have htp' : ∀ i (hi : i < c.sts.size), (tablesP (tabsOf c.sts) (traces c.sts c.w))[i]! =
      packed (traces c.sts c.w)[i]! (Setup.ofCircuit c.sts[i]!).m := fun i hi => by
    rw [hst i hi]
    exact htp i hi
  have hex : ∀ i, ∃ ei, i < c.sts.size →
      (Setup.ofCircuit c.sts[i]!).extra (pointsOf c.spec (Claims.coinSeq c.r 0)) = .ok ei := by
    intro i
    by_cases hi : i < c.sts.size
    · rw [hst i hi]
      obtain ⟨ei, h⟩ := extraClaims_ok c.sts[i] (hfit _ (Array.getElem_mem hi)) (pointsOf c.spec (Claims.coinSeq c.r 0))
      exact ⟨ei, fun _ => h⟩
    · exact ⟨#[], fun h => absurd h hi⟩
  choose e he using hex
  choose ic hic using fun i rep => innerCommitP_ok (Claims.hOf H) (padsOf c.ω i rep
    (Zk.PadLayout.new (Setup.ofCircuit c.sts[i]!).m (Setup.ofCircuit c.sts[i]!).kLog schs[i]!.1.k0))
    (c.ω.innerSalt i rep)
  obtain ⟨ls, pts, hls0, hls⟩ := hR (fun i => schs[i]!) (fun i => cms[i]!) e ic
    (fun i hi => ⟨by rw [hst i hi]; exact hv i hi, by rw [hst i hi]; exact hcmi i hi, he i hi, hic i⟩) l₀ hl₀
  unfold afterLinkPZk
  refine Runs.bind (Q := fun _ _ => True) ?_ fun b l _ => ?_
  · rw [← Array.forIn_toList, FlockDeploy.repStreams_toList, hspec]
    refine Runs.weaken (Runs.forIn_idx _ (fun j b l => l = ls j ∧ (0 < j → b.1.back! = (pts (j - 1)).2.rounds)) _ 0
      ?_ _ _ ⟨hls0.symm, fun h => absurd h (lt_irrefl 0)⟩) fun _ _ _ => trivial
    intro j ⟨i, k, name⟩ ha ⟨ss, ps⟩ l hinv
    obtain ⟨hj, rfl, rfl⟩ : j < 2 * c.sts.size ∧ i = j / 2 ∧ k = j % 2 := by
      rw [List.getElem?_map] at ha
      by_cases hj : j < 2 * c.sts.size
      · rw [List.getElem?_range hj] at ha
        simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq] at ha
        exact ⟨hj, ha.1.symm, ha.2.1.symm⟩
      · rw [List.getElem?_eq_none (by simpa using hj)] at ha
        cases ha
    have hi : j / 2 < c.sts.size := by omega
    have hpt := hls j hj
    simp only [Nat.zero_add] at hinv
    obtain ⟨rfl, hback⟩ := hinv
    dsimp only
    rw [htab' _ hi]
    have hprior : priorOf (Setup.ofCircuit c.sts[j / 2]!) ss (j % 2) = priorJ c pts j := by
      unfold priorOf priorJ
      by_cases h1 : j % 2 = 1
      · rw [if_pos h1, if_pos h1, hback (by omega)]
        rfl
      · rw [if_neg h1, if_neg h1]
    rw [hprior, MaskRank.masksOf_getElem! hi]
    refine Runs.bind (Q := fun ex l' => ex = e (j / 2) ∧ l' = ls j) ⟨e (j / 2), ls j, ?_, rfl, rfl⟩
      fun ex l' hq => ?_
    · rw [monadLift_go]
      show ((Setup.ofCircuit c.sts[j / 2]!).extra (ptsOf c.spec (coinOf (c.resp 0))), ls j) = _
      rw [show c.resp 0 = (Claims.coinSeq c.r 0).bytes from rfl, ptsOf_coin, he _ hi]
    obtain ⟨rfl, rfl⟩ := hq
    refine Runs.bind (Q := fun ic' l'' => ic' = ic (j / 2) (j % 2) ∧ l'' = ls j)
      ⟨ic (j / 2) (j % 2), ls j, ?_, rfl, rfl⟩ fun ic' l₁ hq => ?_
    · have := innerCommitP_pureS (Claims.hOf H) (padsOf c.ω (j / 2) (j % 2) (Zk.PadLayout.new
        (Setup.ofCircuit c.sts[j / 2]!).m (Setup.ofCircuit c.sts[j / 2]!).kLog schs[j / 2]!.1.k0))
        (c.ω.innerSalt (j / 2) (j % 2)) (ls j)
      rw [hic] at this
      simpa using this
    obtain ⟨rfl, rfl⟩ := hq
    rw [htp' _ hi]
    refine Runs.bind (Q := fun pt' l'' => pt' = pts j ∧ l'' = ls (j + 1)) ⟨pts j, ls (j + 1), hpt, rfl, rfl⟩
      fun pt' l'' hq => ?_
    obtain ⟨p, t⟩ := pt'
    exact Runs.pure ⟨_, rfl, by simp only [Nat.zero_add]; exact ⟨hq.2, fun _ => by simp [← hq.1]⟩⟩
  obtain ⟨ss, ps⟩ := b
  dsimp only
  rw [recordJson_eqZk _ hct hown hcn]
  simp only [pure_bind]
  exact Runs.pure trivial

/-- **The masked session runs** at several tables: the goal `SessionRunsJ`. -/
theorem sessionRunsJ_goal (hS : StmtOkJ H rel) (hR : ProverRunsJ H rel) : SessionRunsJ H rel :=
  fun c hc => c.sessionRuns (hS c hc) (hR c hc)

end FlockComplete.ZkTables
