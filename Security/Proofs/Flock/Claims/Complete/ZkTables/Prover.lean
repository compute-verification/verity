import Proofs.Flock.Claims.Complete.ZkTables.Complete

/-!
# What the record needs of the masked prover's pieces, at every table

`Zk/ProverOk.lean` and `Zk/RepRounds.lean` at every table of a good case of any number of tables: the session links at
two points and each table has at most 1,024 regions (`Case.points_regions`), so each table's masked level-0 cap is
non-empty and under `2^64` bytes (`commitPZk_cap`) and each masked rep, on the table's extra claims at the link's
points, records a round, each of at most `65536` coins (`repPZk_grows`): `proverOkJ`.
-/

open Flock FlockDeploy Transcript

namespace FlockComplete.ZkTables

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Exec JsonRT

variable {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}

/-- **At a good case the session links at two points and every table has at most 1,024 regions.** -/
theorem Case.points_regions (c : Case H rel) : c.spec.points = 2 ∧ ∀ st ∈ c.sts, st.regions.size ≤ 1024 := by
  obtain ⟨p, hp, hsc⟩ := (Readable.scope_iff (deployZk rel) rel (readOf true) rfl
    (fun _ _ h => readOf_zk h) c.x).1 c.scope
  have hi := (setupOf_some c.setup).1
  unfold Readable.parseBytes readOf at hp
  rw [hi] at hp
  have ht := Readable.toRead_of_parse (r := c.s.read) hp
  obtain ⟨hck, -⟩ := (Readable.scope_ok_iff p).1 hsc
  have hR := ((Readable.check_ok_iff _ _ _ _ _ _ _ _).1 hck).1
  have hpi : p.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simp [hR.2.2.2.2.2])
  have hset := Ins.setup_read (setupOf_some c.setup).2
  rw [← ht] at hset
  unfold Readable.Read.setup at hset
  rw [hpi] at hset
  refine ⟨setupTables_points hset, fun st hst => ?_⟩
  obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.1 hst
  obtain ⟨d, hd, -⟩ := (HiddenExec.setupTables_setupOne hset).2 j hj
  rcases Bool.eq_false_or_eq_true (tagsAt p.tags p.I.count j).hiddenOutputs with hh | hh
  · rw [HiddenExec.setupOne_hidden hh] at hd
    exact FlockVBridge.Family.spec_regions_le (FlockSoundness.Discharge.Hidden.setupHidden_spec' hd)
  · rw [HiddenExec.setupOne_public hh] at hd
    exact FlockVBridge.Family.setupH_regions_le hd

variable (H rel)

/-- **What the record needs of the masked prover's pieces** (`ProverOkZk`), at every table of every good case: the
masked commitment's cap, at the `--zk` schedule's first level, is non-empty and under `2^64` bytes, and each masked rep,
on the table's extra claims at the link's points, records a round, each of at most `65536` coins. -/
def ProverOkJ : Prop :=
  ∀ c : Case H rel, ∀ st ∈ c.sts,
    (∀ v, Zk.schedule (Setup.ofCircuit st).m = .ok v → ∀ g extra lanePad salt cm,
      commitPZk (Hashes.ofSha512 (Claims.hOf H) : Hashes V) (Setup.ofCircuit st).merkle (v.1.levels.getD 0 default)
        v.2 g extra lanePad salt = .ok cm → 0 < (flat cm.cap).size ∧ (flat cm.cap).size < 2 ^ 64) ∧
    ∀ v, Zk.schedule (Setup.ofCircuit st).m = .ok v → ∀ rep dom tr tb cm₀ extras pads ic extra lanePad salt nonce,
      (Setup.ofCircuit st).extra (ptsOf c.spec (coinOf (c.resp 0))) = .ok extras →
      Ret ((repPZk (asksS c.resp) (Hashes.ofSha512 (Claims.hOf H)) (Setup.ofCircuit st) v.1 v.2 rep dom tr tb cm₀
        extras pads ic extra lanePad salt nonce).run (Transcript.init #[] dom)) fun pt => RepOk pt.2.rounds

variable {H rel}

/-- **The masked prover's pieces are what the record needs**, at every table, at an oracle of 64-byte outputs. -/
theorem proverOkJ (h64 : Out64 H) : ProverOkJ H rel := by
  intro c st hst
  obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.1 hst
  obtain ⟨hm1, hm2, hms⟩ := c.tables.2.2.2 j hj
  refine ⟨fun v hv g extra lanePad salt cm hcm => ?_,
    fun v hv rep dom tr tb cm₀ extras pads ic extra lanePad salt nonce hex => ?_⟩
  · have hm : (Setup.ofCircuit c.sts[j]).merkle = MerkleScheme.hm96Sha512 := hms
    rw [hm] at hcm
    exact commitPZk_cap (hOf_size h64) hm1 hm2 hv hcm
  · have hreg := c.points_regions.2 _ hst
    have hx : extras.size = c.sts[j].regions.size * 2 := by
      rw [FlockDeploy.extraClaims_size hex]
      simp [JsonRT.ptsOf, c.points_regions.1]
    have hm : (Setup.ofCircuit c.sts[j]).m = c.sts[j].m := rfl
    rw [hm] at hv
    obtain ⟨hn, hq⟩ := zkSchedule_small hm1 hm2 hv
    rintro l ⟨p, t'⟩ l' hrun
    obtain ⟨⟨hs, hsz⟩, -⟩ := repPZk_grows c.resp _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ (B := 65536) (by decide)
      (by simp [Setup.ofCircuit]; omega) (by omega) hn hq _ l _ _ _ hrun
    exact ⟨show 0 < t'.rounds.size by simp [Transcript.init] at hsz; omega, hs (by simp [CoinsLe, Transcript.init])⟩

end FlockComplete.ZkTables
