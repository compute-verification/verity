import Proofs.Flock.Claims.Complete.Replay.RepRounds
import Proofs.Flock.Claims.Complete.Replay.LigeritoWalk
import Proofs.Flock.Claims.DeployK
import Proofs.Flock.VBridge.Family.Lig
import Proofs.Flock.Soundness.Discharge.Hidden.Spec
import Proofs.Flock.Soundness.Discharge.Exec.Merkle

/-!
# The prover's pieces at every good case (`ProverOk`)

`proverOk`: at an oracle of 64-byte outputs (`Out64`), every good case's table has what the record needs of the
prover's pieces.

* The commitment's cap at the schedule's first level is non-empty and under `2^64` bytes: the commitment is its Merkle
  tree (`Lg.commitP_tree`), whose cap is `2^cap` digests of the scheme's width (`opsWith_widths`, between `1` and `64`
  bytes for a registered scheme), and `fast100`'s first level has `cap ≤ 32` (`fast100_cap0`).
* Each rep on the table's extra claims records a round, each of at most `65536` coins (`repP_repOk`): the extra claims
  are the regions' at the link's two points (`extraClaims_size`), at most 1,024 regions (`Case.schemeOk`).

`Case.schemeOk`: the case's one table is `setupOne`'s, so its scheme is registered (`MerkleScheme.find?`) and its
regions pass `checkInRange` (`setupH_regions_le`, `spec_regions_le`). So the run's schemes have their widths
(`schemeWidths`, the walk's `SchemeWidths`).
-/

namespace FlockComplete.Replay

open Flock FlockDeploy Transcript

/-- **The case's table has a registered scheme and at most 1,024 regions.** -/
theorem _root_.FlockComplete.Case.schemeOk {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}
    (c : Case H rel) :
    ∃ st, c.sts = #[st] ∧ c.spec = st.spec ∧ st.merkle ∈ MerkleScheme.all ∧ st.regions.size ≤ 1024 := by
  obtain ⟨p, hp, hsc⟩ := (Readable.scope_iff (deployPlain rel) rel (readOf false) rfl
    (fun _ _ h => readOf_zk h) c.x).1 c.scope
  have hi := (setupOf_some c.setup).1
  unfold Readable.parseBytes readOf at hp
  rw [hi] at hp
  have ht := Readable.toRead_of_parse (r := c.s.read) hp
  obtain ⟨hc, -⟩ := (Readable.scope_ok_iff p).1 hsc
  have hR := ((Readable.check_ok_iff _ _ _ _ _ _ _ _).1 hc).1
  have hpi : p.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simp [hR.2.2.2.2.2])
  have hset := Ins.setup_read (setupOf_some c.setup).2
  rw [← ht] at hset
  unfold Readable.Read.setup at hset
  rw [hpi] at hset
  rcases FlockSoundness.Discharge.HiddenExec.setupTables_okAny hset with ⟨-, st, h1, hs, hsts⟩ | ⟨hc1, hsz, -⟩
  · refine ⟨st, hsts, hs, ?_⟩
    rcases Bool.eq_false_or_eq_true p.tags.hiddenOutputs with hh | hh
    · rw [FlockSoundness.Discharge.HiddenExec.setupOne_hidden hh] at h1
      have S := FlockSoundness.Discharge.Hidden.setupHidden_spec' h1
      exact ⟨List.mem_of_find?_eq_some S.merkle, FlockVBridge.Family.spec_regions_le S⟩
    · rw [FlockSoundness.Discharge.HiddenExec.setupOne_public hh] at h1
      exact ⟨List.mem_of_find?_eq_some (FlockSoundness.Discharge.Exec.setupH_find h1).2,
        FlockVBridge.Family.setupH_regions_le h1⟩
  · obtain ⟨st0, hsts0, -⟩ := c.table
    rw [hsts0] at hsz
    simp only [List.size_toArray, List.length_cons, List.length_nil, zero_add] at hsz
    omega

theorem scheme_widths {ms : MerkleScheme} (h : ms ∈ MerkleScheme.all) : 0 < ms.digestLen ∧ ms.digestLen ≤ 64 := by
  simp only [MerkleScheme.all, List.mem_cons, List.not_mem_nil, or_false] at h
  rcases h with rfl | rfl | rfl | rfl <;> simp [MerkleScheme.sha256, MerkleScheme.sha512, MerkleScheme.hm96Sha512]

theorem opsWith_lookup {M : Type → Type} [Monad M] [MonadExceptOf String M] {hs : Hashes M} {ms : MerkleScheme}
    {k : UInt32} (h : MerkleScheme.oracleIds.lookup ms.id = some k) :
    ms.opsWith hs = if k = 0 then MerkleOps.sha256With hs else if k = 2 then MerkleOps.sha512With hs
      else if k = 3 then MerkleOps.hm96With hs else MerkleOps.ofScheme ms := by
  unfold MerkleScheme.opsWith
  rw [h]

/-- **Every registered scheme's digests have its width** through `Hashes.ofSha512` at an oracle of 64-byte outputs. -/
theorem opsWith_widths {H : Claims.Msg → Claims.Msg} (h64 : JsonRT.Out64 H) {ms : MerkleScheme}
    (hms : ms ∈ MerkleScheme.all) :
    (∀ l r v, (ms.opsWith (Hashes.ofSha512 (Claims.hOf H) : Hashes V)).node l r = .ok v → v.size = ms.digestLen) ∧
    ∀ row salt v, (ms.opsWith (Hashes.ofSha512 (Claims.hOf H) : Hashes V)).leaf row salt = .ok v →
      v.size = ms.digestLen := by
  have h512 : ∀ b, (Claims.hOf H b).size = 64 := JsonRT.hOf_size h64
  simp only [MerkleScheme.all, List.mem_cons, List.not_mem_nil, or_false] at hms
  rcases hms with rfl | rfl | rfl | rfl
  · have e : MerkleScheme.oracleIds.lookup (MerkleScheme.sha256 "flock-leaf/sha256-unsalted/v0").id = some 0 := by
      decide +kernel
    rw [opsWith_lookup e, ite_eq_left rfl]
    exact ⟨fun l r v e => by
      simp only [MerkleOps.sha256With, Hashes.ofSha512, pure, Except.pure, Except.ok.injEq] at e
      rw [← e]; exact Sha256.hash_size _, fun row salt v e => by
      simp only [MerkleOps.sha256With, Hashes.ofSha512, pure, Except.pure, Except.ok.injEq] at e
      rw [← e]; exact Sha256.hash_size _⟩
  · have e : MerkleScheme.oracleIds.lookup (MerkleScheme.sha256 "flock-leaf/sha256-unsalted").id = some 0 := by
      decide +kernel
    rw [opsWith_lookup e, ite_eq_left rfl]
    exact ⟨fun l r v e => by
      simp only [MerkleOps.sha256With, Hashes.ofSha512, pure, Except.pure, Except.ok.injEq] at e
      rw [← e]; exact Sha256.hash_size _, fun row salt v e => by
      simp only [MerkleOps.sha256With, Hashes.ofSha512, pure, Except.pure, Except.ok.injEq] at e
      rw [← e]; exact Sha256.hash_size _⟩
  · have e : MerkleScheme.oracleIds.lookup (MerkleScheme.sha512 "flock-leaf/sha512-unsalted").id = some 2 := by
      decide +kernel
    rw [opsWith_lookup e, ite_eq_right (by decide), ite_eq_left rfl]
    exact ⟨fun l r v e => by
      simp only [MerkleOps.sha512With, Hashes.ofSha512, pure, Except.pure, Except.ok.injEq] at e
      rw [← e]; exact h512 _, fun row salt v e => by
      simp only [MerkleOps.sha512With, Hashes.ofSha512, pure, Except.pure, Except.ok.injEq] at e
      rw [← e]; exact h512 _⟩
  · have e : MerkleScheme.oracleIds.lookup MerkleScheme.hm96Sha512.id = some 3 := by decide +kernel
    rw [opsWith_lookup e, ite_eq_right (by decide), ite_eq_right (by decide), ite_eq_left rfl]
    exact ⟨fun l r v e => by
      simp only [MerkleOps.hm96With, Hashes.ofSha512, pure, Except.pure, Except.ok.injEq] at e
      rw [← e]; exact h512 _, fun row salt v e => by
      simp only [MerkleOps.hm96With, Hm96.Default512.leafWith, Hashes.ofSha512, bind, Except.bind, pure, Except.pure,
        Except.ok.injEq] at e
      rw [← e]; exact h512 _⟩

/-- **Each table's Merkle scheme returns digests of its width** (`SchemeWidths`), at an oracle of 64-byte outputs. -/
theorem schemeWidths {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}
    (h64 : JsonRT.Out64 H) : SchemeWidths H rel := by
  intro o i hi
  obtain ⟨st, hst, -, hall, -⟩ := o.toCase.schemeOk
  have htabs : o.tabs = #[Setup.ofCircuit st] := by simp [Run.tabs, tabsOf, hst]
  obtain rfl : i = 0 := by rw [htabs] at hi; simpa using hi
  have h0 : o.tabs[0]! = Setup.ofCircuit st := by rw [htabs]; rfl
  rw [h0]
  exact opsWith_widths h64 hall

/-- `w` bytes per digest: `flat` of `n` of them is `w n` bytes. -/
theorem flat_size_of {cap : Array ByteArray} {w : ℕ} (h : ∀ b ∈ cap, b.size = w) : (flat cap).size = w * cap.size := by
  unfold flat
  rw [← Array.foldl_toList, ← Array.length_toList]
  suffices ∀ (l : List ByteArray) (acc : ByteArray), (∀ b ∈ l, b.size = w) →
      (l.foldl (· ++ ·) acc).size = acc.size + w * l.length by
    simpa using this cap.toList ByteArray.empty fun b hb => h b (Array.mem_toList_iff.1 hb)
  intro l
  induction l with
  | nil => intro acc _; simp
  | cons b l ih =>
    intro acc hl
    rw [List.foldl_cons, ih _ fun x hx => hl x (List.mem_cons_of_mem _ hx), ByteArray.size_append,
      hl b List.mem_cons_self, List.length_cons]
    ring

/-- fast100's first level caps at most 32 levels. -/
theorem fast100_cap0 {m : ℕ} {s : Schedule} (h : fast100 m = some s) : (s.levels.getD 0 default).cap ≤ 32 := by
  have key : ∀ m : Fin 40, ∀ s ∈ fast100 m.val, decide ((s.levels.getD 0 default).cap ≤ 32) = true := by
    decide +kernel
  exact of_decide_eq_true (key ⟨m, fast100_lt h⟩ s h)

/-- **The level-0 commitment's cap is non-empty and under `2^64` bytes**, through a registered scheme at an oracle of
64-byte outputs, at `fast100`'s first level. -/
theorem commitP_cap {H : Claims.Msg → Claims.Msg} (h64 : JsonRT.Out64 H) {ms : MerkleScheme}
    (hms : ms ∈ MerkleScheme.all) {m : ℕ} {sch : Schedule} (hsch : fast100 m = some sch) {g : Array F128}
    {salt : ℕ → ByteArray} {cm : Commit}
    (h : commitP (Hashes.ofSha512 (Claims.hOf H) : Hashes V) ms (sch.levels.getD 0 default) g salt = .ok cm) :
    0 < (flat cm.cap).size ∧ (flat cm.cap).size < 2 ^ 64 := by
  obtain ⟨hn, hlf⟩ := opsWith_widths h64 hms
  obtain ⟨-, hL, -, -, -⟩ := fast100_ligerito hsch
  have hc := Lg.fast100_cap hsch _ (Lg.getD_mem default (show 0 < sch.levels.size by omega))
  have hT := Lg.commitP_tree hn hlf hc h
  have hsz : cm.cap.size = 2 ^ (sch.levels.getD 0 default).cap := by
    rw [hT.cap, hT.layer _ le_rfl]
    congr 1
    omega
  rw [flat_size_of (fun b hb => hT.width _ le_rfl b (hT.cap ▸ hb)), hsz]
  obtain ⟨hw0, hw64⟩ := scheme_widths hms
  have h32 := fast100_cap0 hsch
  refine ⟨Nat.mul_pos hw0 (Nat.two_pow_pos _), ?_⟩
  calc ms.digestLen * 2 ^ (sch.levels.getD 0 default).cap ≤ 64 * 2 ^ 32 :=
        Nat.mul_le_mul hw64 (Nat.pow_le_pow_right (by norm_num) h32)
    _ < 2 ^ 64 := by norm_num

/-- **What the record needs of the prover's pieces holds at every good case** (`ProverOk`), at an oracle of 64-byte
outputs. -/
theorem proverOk {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation} (h64 : JsonRT.Out64 H) :
    JsonRT.ProverOk H rel := by
  intro c st hst
  obtain ⟨st', hst', hspec, hm22, -⟩ := c.table
  obtain ⟨st'', hst'', -, hall, hreg⟩ := c.schemeOk
  rw [hst'] at hst
  obtain rfl : st = st' := by simpa using hst
  rw [hst'] at hst''
  obtain rfl : st = st'' := by simpa using hst''
  refine ⟨fun sch hsch g salt cm hcm => commitP_cap h64 hall hsch hcm, ?_⟩
  intro sch hsch tr tb cm₀ extras salt nonce d hex
  have hx : extras.size = st.regions.size * 2 := by
    rw [extraClaims_size hex]
    simp [JsonRT.ptsOf, hspec, Stmt.spec]
  exact repP_repOk c.resp _ _ sch tr tb cm₀ extras salt nonce (by simp [Setup.ofCircuit]; omega) hsch (by omega) d

end FlockComplete.Replay
