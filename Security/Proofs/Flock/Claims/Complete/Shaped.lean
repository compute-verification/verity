import Proofs.Flock.Claims.Complete.Shaped.Walk
import Proofs.Flock.Claims.Complete.Shaped.Piop
import Proofs.Flock.Claims.Complete.Json.Stmt
import Proofs.Flock.VBridge.Family.Lig
import Proofs.Flock.Soundness.Refine.Regions

/-!
# The honest rep's proof fits its shape

`repShaped` proves the goal `RepShaped` at an oracle of 64-byte outputs (`Claims.Out64`): each proof the honest rep
returns is `ProofShaped` at the shape the verifier decodes it at, so `decodeProof_encodeProof` reads it back.

`repP_shape` composes the phases' lengths (`Shaped.Piop`, and Ligerito's `ligeritoP_shape`) into `Lengths`, at a
statement whose fold returns (`fold_ok`), a schedule the walk follows (`LigSched`, every `fast100` schedule's by
`fast100_lig`) and a level-0 commitment of the shape the opening reads (`commitP_shape`). The encoding's size bound
is `fileBound_lt`: the file of a statement of `m ≤ 35` with at most 2,050 claims (two, and the regions' extra claims,
`extraClaims_size`, at most 1,024 regions by `Case.scheme`) is under `2^64` bytes, since a registered scheme's digests
and salts are within SHA-512's (`scheme_sizes`, `schedPart_mono`) and `fast100`'s file part at those is under `2^32`.
-/

namespace FlockComplete.JsonRT

open Flock Transcript FlockComplete FlockDeploy FlockSoundness.Refine

theorem shapeOf_salt (st : Setup) (sch : Schedule) (n : ℕ) :
    ((shapeOf st sch n).saltLen = none ∧ st.merkle.saltBytes = 0) ∨
      (shapeOf st sch n).saltLen = some st.merkle.saltBytes := by
  unfold shapeOf
  dsimp only
  split
  · next h =>
    simp only [Bool.and_eq_true, decide_eq_true_eq, Bool.not_eq_eq_eq_not, Bool.not_true] at h
    exact Or.inl ⟨rfl, h.1⟩
  · exact Or.inr rfl

theorem repP_shape (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (st : Stmt) (hk : 6 ≤ st.c.kLog)
    (hkm : st.c.kLog ≤ st.m) (hs : ∀ r ∈ st.c.ranges.toList, SlotOk st.c r)
    (hn : ∀ l r v, (st.merkle.opsWith (Hashes.ofSha512 h : Hashes V)).node l r = .ok v → v.size = st.merkle.digestLen)
    (hlf : ∀ row salt v, (st.merkle.opsWith (Hashes.ofSha512 h : Hashes V)).leaf row salt = .ok v →
      v.size = st.merkle.digestLen)
    {sch : Schedule} (hsch : LigSched sch) (tr : Trace) {t : Array F128} (ht : t.size = 2 ^ sch.n) {cm0 : Commit}
    (hc0 : CommitOk st.merkle.digestLen st.merkle.saltBytes (sch.levels.getD 0 default) cm0) (extras : Array Claim)
    (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) :
    totT.T (repP (asksS resp) (Hashes.ofSha512 h) (Setup.ofCircuit st) sch tr t cm0 extras salt nonce)
      (FlockVBridge.Family.Lengths (shapeOf (Setup.ofCircuit st) sch extras.size)) := by
  unfold repP
  refine totT.bind (zerocheckP_shape resp _ _ _ tr) fun zz hz => ?_
  obtain ⟨zp, zc⟩ := zz
  obtain ⟨h1, h2, h3, hrho⟩ := hz
  dsimp only at h1 h2 h3 hrho ⊢
  have hfold : ∀ α, ∃ v, (Setup.ofCircuit st).fold.fold α (skipWeights zc.z)
      (zc.rho.extract 0 ((Setup.ofCircuit st).fold.kLog - 6)) = .ok v := fun α =>
    fold_ok st hk hs α (skipWeights_size _) (by
      simp only [Setup.ofCircuit, Stmt.circuitFold, Array.size_extract, hrho]
      omega)
  refine totT.bind (lincheckP_shape resp _ _ tr zc hfold hk) fun ll hl => ?_
  obtain ⟨lp, lc⟩ := ll
  dsimp only at hl ⊢
  refine totT.bind (ringSwitchP_shape resp tr _) fun rr hr => ?_
  obtain ⟨ss, T, sws⟩ := rr
  dsimp only at hr ⊢
  refine totT.bind (ligeritoP_shape (sh := shapeOf (Setup.ofCircuit st) sch extras.size) resp h hn hlf
    (shapeOf_salt _ _ _) hsch T hc0 ht _ salt nonce) fun lg hlg => ?_
  refine totT.pure _ ⟨⟨hc0.2.2.2.2.2.1, hc0.2.2.2.2.2.2⟩, h1, h2, h3, hl.1, hl.2, ?_, hr.2, hlg⟩
  rw [hr.1, Array.size_append]
  simp [abcClaims, shapeOf]

/-- **A table's setup has a registered scheme and at most 1,024 regions** (`checkInRange`). -/
theorem setupOne_scheme {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}
    (h : FlockSoundness.Discharge.HiddenExec.setupOne tags cf pf coins tables draw partition program = .ok st) :
    st.merkle ∈ MerkleScheme.all ∧ st.regions.size ≤ 1024 := by
  rcases Bool.eq_false_or_eq_true tags.hiddenOutputs with hh | hh
  · rw [FlockSoundness.Discharge.HiddenExec.setupOne_hidden hh] at h
    have S := FlockSoundness.Discharge.Hidden.setupHidden_spec' h
    exact ⟨List.mem_of_find?_eq_some S.merkle, FlockVBridge.Family.spec_regions_le S⟩
  · rw [FlockSoundness.Discharge.HiddenExec.setupOne_public hh] at h
    exact ⟨List.mem_of_find?_eq_some (FlockSoundness.Discharge.Exec.setupH_find h).2,
      FlockVBridge.Family.setupH_regions_le h⟩

theorem scheme_sizes {ms : MerkleScheme} (h : ms ∈ MerkleScheme.all) :
    0 < ms.digestLen ∧ ms.digestLen ≤ 64 ∧ ms.saltBytes ≤ 192 := by
  simp only [MerkleScheme.all, List.mem_cons, List.not_mem_nil, or_false] at h
  rcases h with rfl | rfl | rfl | rfl <;> simp [MerkleScheme.sha256, MerkleScheme.sha512, MerkleScheme.hm96Sha512]

/-- **The walk follows every executable `fast100` schedule.** -/
theorem fast100_lig (m : ℕ) (s : Schedule) (hs : fast100 m = some s) : LigSched s ∧ s.m = m := by
  obtain ⟨⟨-, h2, -, -, hl4, -, -⟩, hm⟩ := fast100_dims m s hs
  have hlv := fast100_level m s hs
  have hkey : 1 ≤ s.k0 ∧ s.n = s.k0 + s.ext + 3 * (s.levels.size - 1) ∧
      (s.levels.all fun lv => decide (lv.cap ≤ lv.d)) = true := by
    by_cases hm40 : m < 40
    · have key : ∀ m : Fin 40, ∀ s ∈ fast100 m.val, decide (1 ≤ s.k0 ∧ s.n = s.k0 + s.ext + 3 * (s.levels.size - 1) ∧
          (s.levels.all fun lv => decide (lv.cap ≤ lv.d)) = true) = true := by
        decide +kernel
      simpa using key ⟨m, hm40⟩ s hs
    · rw [exec_fast100_none m (by omega)] at hs
      cases hs
  have hget : ∀ i (hi : i < s.levels.size), s.levels.getD i default = s.levels[i] := fun i hi => by
    simp [Array.getD, hi]
  refine ⟨⟨hkey.1, by omega, hkey.2.1, fun i hi => ?_, fun i hi => ?_, hl4⟩, hm⟩
  · rw [hget i hi]
    simpa using Array.all_eq_true.1 hkey.2.2 i hi
  · rw [hget i hi]
    exact hlv _ (Array.getElem_mem hi)

theorem packed_size (tr : Trace) (m : ℕ) : (packed tr m).size = 2 ^ (m - 7) := by simp [packed]

/-- A file's schedule part grows with its digests and salts. -/
theorem schedPart_mono {s : Schedule} {H H' : ℕ} {sl sl' : Option ℕ} (hH : H ≤ H')
    (h : ∀ q, FlockVBridge.Family.saltsSize sl q ≤ FlockVBridge.Family.saltsSize sl' q) :
    FlockVBridge.Family.schedPart s H sl ≤ FlockVBridge.Family.schedPart s H' sl' := by
  have hl : ∀ l w, FlockVBridge.Family.levelSize H sl l w ≤ FlockVBridge.Family.levelSize H' sl' l w := fun l w => by
    unfold FlockVBridge.Family.levelSize
    have := h l.q
    have := Nat.mul_le_mul_left (l.q * (l.d - l.cap)) hH
    omega
  have hs := FlockVBridge.Family.sum_map_le (l := List.range (s.levels.size - 1 - 1))
    (f := fun i => FlockVBridge.Family.levelSize H sl (s.levels.getD (i + 1) default) 16)
    (g := fun i => FlockVBridge.Family.levelSize H' sl' (s.levels.getD (i + 1) default) 16) fun i => hl _ _
  have hc := FlockVBridge.Family.sum_map_le (l := List.range (s.levels.size - 1))
    (f := fun i => 8 + 2 ^ (s.levels.getD (i + 1) default).cap * H)
    (g := fun i => 8 + 2 ^ (s.levels.getD (i + 1) default).cap * H') fun i => by
      have := Nat.mul_le_mul_left (2 ^ (s.levels.getD (i + 1) default).cap) hH
      omega
  have h0 := hl (s.levels.getD 0 default) (2 ^ (s.levels.getD 0 default).lanes)
  have hR := hl (s.levels.getD (s.levels.size - 1) default) 16
  have hc0 := Nat.mul_le_mul_left (2 ^ (s.levels.getD 0 default).cap) hH
  simp only [FlockVBridge.Family.schedPart]
  omega

theorem fileBound_lt {sh : Shape} (hs : FlockVBridge.Family.schedPart sh.sched sh.hashLen sh.saltLen ≤ 2 ^ 32)
    (hm : sh.m ≤ 35) (hk : sh.kLog ≤ 35) (hc : sh.claims ≤ 2050) : FlockVBridge.Family.fileBound sh < 2 ^ 64 := by
  unfold FlockVBridge.Family.fileBound
  omega

theorem extraClaims_size {st : Stmt} {pts : Array (Array F128)} {e : Array Claim} (h : st.extraClaims pts = .ok e) :
    e.size = st.regions.size * pts.size := by
  obtain ⟨ess, hess, heq⟩ := FlockSoundness.Refine.extraClaims_ok st pts e h
  have key : ∀ (L : List Region) (ess : List (List Claim)), List.Forall₂ (fun reg es => List.Forall₂
      (fun r e => ∃ v, st.regionValue reg r = .ok v ∧ e = regionClaimOf st reg r v) pts.toList es) L ess →
      ess.flatten.length = L.length * pts.size := by
    intro L ess h
    induction h with
    | nil => simp
    | cons hes _ ih =>
      rw [List.flatten_cons, List.length_append, ih, ← hes.length_eq, List.length_cons, Array.length_toList]
      ring
  rw [← Array.length_toList, heq, key _ _ hess, Array.length_toList]

/-- **The honest rep proof fits its shape**: at every good case, on its table's trace, the trace's packed table
committed at the schedule's first level (`cm`, under any salts), and the table's extra claims at the link's points
(`e`), each proof a rep returns is `ProofShaped` at the shape the verifier decodes it at. -/
def RepShaped (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Prop :=
  ∀ c : Case H rel, ∀ st sch cm e, c.sts = #[st] → fast100 (Setup.ofCircuit st).m = some sch →
    ∀ salt₀, commitP (Hashes.ofSha512 (Claims.hOf H) : Hashes V) (Setup.ofCircuit st).merkle
      (sch.levels.getD 0 default) (packed (tracesOf c.sts c.w)[0]! (Setup.ofCircuit st).m) salt₀ = .ok cm →
    (Setup.ofCircuit st).extra (pointsOf c.spec (Claims.coinSeq c.r 0)) = .ok e →
    ∀ salt nonce d, Ret ((repP (asksS c.resp) (Hashes.ofSha512 (Claims.hOf H)) (Setup.ofCircuit st) sch
      (tracesOf c.sts c.w)[0]! (packed (tracesOf c.sts c.w)[0]! (Setup.ofCircuit st).m) cm e salt nonce).run
      (Transcript.init #[] d)) fun pt => FlockVBridge.Family.ProofShaped (shapeOf (Setup.ofCircuit st) sch e.size) pt.1

/-- **The case's table has a registered scheme and at most 1,024 regions.** -/
theorem _root_.FlockComplete.Case.scheme {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}
    (c : Case H rel) :
    ∃ st, c.sts = #[st] ∧ c.spec = st.spec ∧ st.merkle ∈ MerkleScheme.all ∧ st.regions.size ≤ 1024 := by
  obtain ⟨st, h1, h2, h3⟩ := c.one
  exact ⟨st, h1, h2, setupOne_scheme h3⟩

/-- **The honest rep proof fits its shape**: the goal `RepShaped`, at an oracle of 64-byte outputs. -/
theorem repShaped {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation} (h64 : Claims.Out64 H) :
    RepShaped H rel := by
  intro c st sch cm e hst hsch salt₀ hcm he salt nonce d
  obtain ⟨st', hst', hspec, -, hm35, ⟨tags, file, tables, hty, hcirc⟩, hm, hk⟩ := c.table
  rw [hst] at hst'
  obtain rfl : st = st' := by simpa using hst'
  obtain ⟨st'', hst'', -, hall, hreg⟩ := c.scheme
  rw [hst] at hst''
  obtain rfl : st = st'' := by simpa using hst''
  obtain ⟨ls, hcheck⟩ := circuitOf_check hty hcirc
  obtain ⟨hL, hsm⟩ := fast100_lig _ _ hsch
  obtain ⟨hn, hlf⟩ := opsWith_sizes h64 hall
  have hc0 := commitP_shape hn hlf (hL.cap 0 (by have := hL.two; omega)) hcm
  have ht : (packed (tracesOf c.sts c.w)[0]! (Setup.ofCircuit st).m).size = 2 ^ sch.n := by
    rw [packed_size, (fast100_dims _ _ hsch).1.1, hsm]
  have hT := repP_shape c.resp (Claims.hOf H) st hk (by omega) (hm_check_kinds hcheck) hn hlf hL
    (tracesOf c.sts c.w)[0]! ht hc0 e salt nonce
  have hsc := scheme_sizes hall
  have he2 : e.size = st.regions.size * 2 := by
    rw [extraClaims_size he]
    simp [pointsOf, hspec, Stmt.spec]
  intro l pt l' hrun
  obtain ⟨pt', l'', e', hp⟩ := hT (Transcript.init #[] d) l
  have := hrun.symm.trans e'
  simp only [Prod.mk.injEq, Except.ok.injEq] at this
  obtain ⟨rfl, -⟩ := this
  have hsp : FlockVBridge.Family.schedPart sch st.merkle.digestLen
      (shapeOf (Setup.ofCircuit st) sch e.size).saltLen ≤ FlockVBridge.Family.schedPart sch 64 (some 192) :=
    schedPart_mono hsc.2.1 fun q => by
      rcases shapeOf_salt (Setup.ofCircuit st) sch e.size with ⟨e1, -⟩ | e1 <;> rw [e1] <;>
        simp only [FlockVBridge.Family.saltsSize]
      · omega
      · have := Nat.mul_le_mul_left q hsc.2.2
        dsimp only [Setup.ofCircuit]
        omega
  refine ⟨hsc.1, hL.two, lt_of_le_of_lt (FlockVBridge.Family.encodeProof_size_le _ _ hp)
    (fileBound_lt (le_trans hsp (FlockVBridge.Family.fast100_file _ _ hsch).2.1) hm35 (by show st.c.kLog ≤ 35; omega)
      (by show 2 + e.size ≤ 2050; omega)), hp⟩

end FlockComplete.JsonRT
