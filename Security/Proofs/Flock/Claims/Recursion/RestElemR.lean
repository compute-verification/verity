import Proofs.Flock.Claims.Recursion.RestReadsAt
import Proofs.Flock.Claims.Recursion.OpenRowsDeployed
import Proofs.Flock.Claims.Recursion.VerdictRowG
import Proofs.Flock.Claims.Recursion.NextRTables
import Proofs.Flock.Claims.Complete.Shaped
import Proofs.Flock.Claims.Recursion.NextRFits
import Proofs.Flock.Claims.Recursion.RestFitsLift

/-!
# `ElemFramed` and `OpenedShaped` at the deployed last message `elemR`

`RestRead.readsAt` takes three facts of the element-level last message. At U8's `elemR` two of them hold with no
hypothesis, at every witness and every decision `rdR` reads:

* `elemFramed_elemR`: `rdR` reads `elemR`'s pieces as the statement `xIOf x` and two proofs a table
  (`rdR_elemR`). The statement `elemR` lays is `stmtOf`, which is `xIOf` where the reader parses `x` and `[]` where it
  doesn't (`stmtOf_eq`), and it fits its word since the compiled statement's decoder read its length as 8 bytes
  (`inner_lt`).
* `openedShaped_elemR`: every proof `readPs` reads is `ProofShaped` at its file's shape. Its absorbed fields are a
  proof's that fits the rep (`FitsRep.sizes`, `FitsRep.nonces`, which `readPs` demands), its openings are `elemR`'s,
  whose sizes are its layout's for any witness (`opens_ok`), the inner table commits under plain SHA-512 with no salts
  (`dense_merkle`: the dense inner statement is set up at `RecVStar.innerTags`), and the file fits a `u64` at a
  `fast100` schedule, `m ≤ 35` and at most 2,048 extra claims (`fileBound_lt`, `tablesOk_of_read`). This holds for
  adversarial witnesses too: nothing here reads a witness's values, only `elemR`'s layout.

`readsAt_elemR` is `RestRead.readsAt` at `elemR` with only `OpenedFit` left. `OpenedFit` follows from the rows'
sizes (`openedFit_of`): the honest proof fits its own rows (`repFits`), and a fit lifts to any rows of the same byte
sizes (`RestRead.fitsLift`), so it is enough that the opened rounds have an honest rep's sizes (`OpenedSized`).
`readsAt_elemR'` takes that one.
-/

namespace RecDeploy.RestElem

open Flock
open Claims (Msg)
open RecDeploy.DeployedR
open RecDeploy.OpenRows
open RecDeploy.RestRead (DenseX finalR_last)
open FlockDeploy (Coin Rd rdTake rdNat rdBytes rdStr rdOpt rdList)
open Lean (Json)
open FlockSoundness.Discharge.ZkView renaming LevelOk → ZLevelOk, NoncesZero → ZNoncesZero
open FlockSoundness.Refine renaming shapeOf → shapeR
open FlockSoundness.Discharge.DecodeZL (ProofSizes)

/-! ## The statement fits its word -/

theorem rd_bind {α β : Type} {a : Rd α} {f : α → Rd β} {s : Msg} {r : β × Msg} (h : (a >>= f).run s = some r) :
    ∃ y s₁, a.run s = some (y, s₁) ∧ (f y).run s₁ = some r := by
  rw [StateT.run_bind] at h
  cases ha : a.run s with
  | none => rw [ha] at h; cases h
  | some p => rw [ha] at h; exact ⟨p.1, p.2, rfl, h⟩

theorem rdTake_length {n : ℕ} {s s' : Msg} {a : List UInt8} (h : (rdTake n).run s = some (a, s')) : a.length = n := by
  simp only [rdTake, StateT.run] at h
  split_ifs at h with hle
  obtain ⟨rfl, -⟩ := Prod.mk.inj (Option.some.inj h)
  simp [hle]

theorem foldr_lt : ∀ bs : List UInt8, bs.foldr (fun b acc => b.toNat + 256 * acc) 0 < 256 ^ bs.length
  | [] => by simp
  | b :: bs => by
    have ih := foldr_lt bs
    have hb : b.toNat < 256 := UInt8.toNat_lt b
    simp only [List.foldr_cons, List.length_cons, pow_succ]
    omega

theorem rdNat_lt {s s' : Msg} {n : ℕ} (h : rdNat.run s = some (n, s')) : n < 2 ^ 64 := by
  unfold rdNat at h
  obtain ⟨bs, s₀, hb, h⟩ := rd_bind h
  rw [StateT.run_pure] at h
  obtain ⟨rfl, -⟩ := Prod.mk.inj (Option.some.inj h)
  have hl := rdTake_length hb
  have := foldr_lt bs
  rw [hl] at this
  exact lt_of_lt_of_eq this (by norm_num)

/-- `RecX.read` after its format string. -/
def readTail : Rd RecX := do
  let n ← rdNat
  let inner ← rdTake n
  let recInner ← rdBytes
  let registration ← rdOpt rdBytes
  let parts ← rdList PartX.read
  pure ⟨inner, recInner, registration, parts⟩

theorem read_eq : RecX.read = (do
    if (← rdStr) ≠ XFORMAT then failure
    readTail) := rfl

theorem tail_inner_lt {s s' : Msg} {w : RecX} (h : readTail.run s = some (w, s')) : w.inner.length < 2 ^ 64 := by
  unfold readTail at h
  obtain ⟨n, s₁, hn, h⟩ := rd_bind h
  obtain ⟨inner, s₂, hi, h⟩ := rd_bind h
  obtain ⟨_, s₃, -, h⟩ := rd_bind h
  obtain ⟨_, s₄, -, h⟩ := rd_bind h
  obtain ⟨_, s₅, -, h⟩ := rd_bind h
  rw [StateT.run_pure] at h
  obtain ⟨rfl, -⟩ := Prod.mk.inj (Option.some.inj h)
  show inner.length < 2 ^ 64
  rw [rdTake_length hi]
  exact rdNat_lt hn

theorem decodeX_of_readX {x : Msg} {r : XR} (h : readX x = some r) : decodeX x = some r.w := by
  replace h := readXPre_of_readX h
  unfold readXPre at h
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨w, hw, ri, -, sh, -, pa, -, dims, -, vs, -, h⟩ := h
  split_ifs at h with hl
  · simp at h
  have h' : (⟨w, ri, pa, sh, dims⟩ : XR) = r := by simpa using h
  subst h'
  exact hw

theorem read_of_decodeX {x : Msg} {w : RecX} (h : decodeX x = some w) : RecX.read.run x = some (w, []) := by
  unfold decodeX at h
  split at h
  · split_ifs at h with he
    cases h
    assumption
  · cases h

/-- **The compiled statement's inner statement fits its word**: its decoder reads the length as 8 bytes. -/
theorem inner_lt {x : Msg} {r : XR} (h : readX x = some r) : r.w.inner.length < 2 ^ 64 := by
  have hrun := read_of_decodeX (decodeX_of_readX h)
  rw [read_eq] at hrun
  obtain ⟨t, s₁, -, hrun⟩ := rd_bind hrun
  by_cases hc : t ≠ XFORMAT
  · exfalso
    first
    | (rw [if_pos hc] at hrun; exact Option.noConfusion hrun)
    | (simp only [hc, if_true, ite_true, not_false_eq_true] at hrun; exact Option.noConfusion hrun)
    | (simp [hc] at hrun; done)
  · have h2 : readTail.run s₁ = some (r.w, []) := by
      first
      | (rw [if_neg hc] at hrun; exact hrun)
      | (simpa [hc] using hrun)
      | (simp only [hc, if_false, ite_false] at hrun; exact hrun)
    exact tail_inner_lt h2

theorem stmtOf_eq {x : Msg} {r : XR} (hr : readX x = some r) : stmtOf r = xIOf x := by
  cases hp : RecDeploy.parse x with
  | some st => exact (stmtOf_xIOf hp hr).symm
  | none =>
    have ha : _root_.Flock.RecScope.accepts (readOf r) = false := by
      simpa [RecDeploy.parse, _root_.Flock.RecScope.parseOf, _root_.Flock.RecScope.parseBytes, decodeR, hr] using hp
    simp [stmtOf, ha, xIOf, hp]

theorem stmtR_eq {x : Msg} {r : XR} (hr : readX x = some r) : stmtR r = xIOf x := by
  have hlt : (stmtOf r).length < 2 ^ 64 := by
    unfold stmtOf
    split_ifs
    · exact inner_lt hr
    · simp
  rw [stmtR, if_pos hlt, stmtOf_eq hr]

/-- **`elemR` lays the element-level last message**: `rdR` reads its pieces as the statement and two proofs a table. -/
theorem elemFramed_elemR : ElemFramed elemR := by
  intro x r hr ws hl
  have hpos : 0 < r.w.parts.length := by
    obtain ⟨_, -, h0⟩ := capAt_readX hr
    have hn : nS x = r.w.parts.length := by simp [nS, hr]
    omega
  refine ⟨(((List.range (nProofs r)).map (proofR r ws)).map opensOf).toArray, ?_, ?_⟩
  · rintro ⟨s, spec, sts⟩ hv
    have ht : tablesR r = sts.size := by
      unfold tablesR
      rw [stmtOf_eq hr, hv]
    have h64 := (FlockDeploy.setup_tables hv).1
    have hn : nProofs r = 2 * sts.size := by
      unfold nProofs
      rw [ht, if_pos (by omega)]
    simp [hn]
  · rw [rdR_elemR r hl hpos, stmtR_eq hr]

/-! ## The inner table commits under plain SHA-512 -/

/-- **A dense inner statement's one table commits under plain SHA-512**: it is set up at `RecVStar.innerTags`, whose
leaves are `flock-leaf/sha512-unsalted`, by `setupHidden`. -/
theorem dense_merkle {x : Msg} {s : FlockDeploy.Ins} {spec : SessionSpec} {st : Stmt}
    (hset : FlockDeploy.setupOf false x = some (s, spec, #[st])) (htags : s.read.tags = RecCoins.innerTags)
    (hpi : s.read.publicInputs = none) : st.merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted" := by
  have h := FlockDeploy.Ins.setup_read (FlockDeploy.setupOf_some hset).2
  unfold Readable.Read.setup at h
  have htags' : s.read.tags = RecVStar.innerTags := htags
  rw [htags', hpi] at h
  have hl : RecVStar.innerTags.merkleLeaf = "flock-leaf/sha512-unsalted" := rfl
  rcases FlockSoundness.Discharge.HiddenExec.setupTables_okH
      FlockSoundness.Discharge.Recursion.innerTags_hiddenOutputs h with
    ⟨-, st', hst, -, hsts⟩ | ⟨hc, hsz, -, -, -⟩
  · have e : st = st' := by simpa using hsts
    subst e
    exact FlockSoundness.Discharge.Hidden.setupHidden_plain hl hst
  · have : (#[st] : Array Stmt).size = 1 := rfl
    omega

/-! ## The openings `elemR` lays -/

theorem levelOk_levelProof (r : XR) (ws : List Claims.Flock.Wit) (k ℓ : ℕ) {sh : Flock.Shape} {l : Flock.Level}
    {w : ℕ} (hH : sh.hashLen = 64) (hsl : sh.saltLen = none ∨ sh.saltLen = some 0)
    (hl : (schR r).levels[ℓ]! = l) (hw : 2 ^ l.lanes = w) :
    ZLevelOk sh l w (levelProof r ws k ℓ) := by
  subst hl hw
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · simp [levelProof, lvlRows]
  · intro a ha
    obtain ⟨j, -, rfl⟩ := List.mem_map.1 (List.mem_toArray.1 ha)
    exact FlockVBridge.Family.size_elems _ _ _
  · simp only [levelProof, lvlSibs, List.size_toArray]
    exact length_flatMap_range _ _ _
  · intro b hb
    obtain ⟨j, -, hj⟩ := List.mem_flatMap.1 (List.mem_toArray.1 hb)
    obtain ⟨h, -, rfl⟩ := List.mem_map.1 hj
    rw [FlockVBridge.Family.size_bytesAt, hH]
  · rcases hsl with h | h <;> simp [FlockSoundness.Discharge.ZkView.SaltsOk, h, levelProof]

/-- **Rep `k`'s openings have their levels' shapes**: level 0's, the recursive levels' and the last level's, at a
shape of 64-byte digests and no salts, for any witness. -/
theorem opens_ok (r : XR) (ws : List Claims.Flock.Wit) (k : ℕ) {sh : Flock.Shape} {sch : Flock.Schedule}
    (hsR : schR r = sch) (hH : sh.hashLen = 64) (hsl : sh.saltLen = none ∨ sh.saltLen = some 0)
    (hl4 : ∀ i < sch.levels.size, 1 ≤ i → (sch.levels.getD i default).lanes = 4) (h2 : 2 ≤ sch.levels.size) :
    ZLevelOk sh (sch.levels.getD 0 default) (2 ^ (sch.levels.getD 0 default).lanes)
        (opensOf (proofR r ws k)).ligerito.initialProof ∧
      (opensOf (proofR r ws k)).ligerito.recursiveProofs.size = sch.levels.size - 1 - 1 ∧
      (∀ i (h : i < (opensOf (proofR r ws k)).ligerito.recursiveProofs.size),
        ZLevelOk sh (sch.levels.getD (i + 1) default) 16 (opensOf (proofR r ws k)).ligerito.recursiveProofs[i]) ∧
      ZLevelOk sh (sch.levels.getD (sch.levels.size - 1) default) 16
        ⟨(opensOf (proofR r ws k)).ligerito.finalProof.rows, (opensOf (proofR r ws k)).ligerito.finalProof.siblings,
          (opensOf (proofR r ws k)).ligerito.finalProof.salts⟩ := by
  subst hsR
  have hg : ∀ ℓ, (schR r).levels[ℓ]! = (schR r).levels.getD ℓ default := fun ℓ =>
    FlockSoundness.Discharge.Hidden.getElem!_eq_getD' _ ℓ
  have hsz : (opensOf (proofR r ws k)).ligerito.recursiveProofs.size = (schR r).levels.size - 2 := by
    show (((List.range ((schR r).levels.size - 2)).map fun i => levelProof r ws k (i + 1)).toArray).size = _
    simp
  refine ⟨levelOk_levelProof r ws k 0 hH hsl (hg 0) rfl, by omega, fun i hi => ?_, ?_⟩
  · have hi' : i < (schR r).levels.size - 2 := by omega
    have e : (opensOf (proofR r ws k)).ligerito.recursiveProofs[i] = levelProof r ws k (i + 1) := by
      show (((List.range ((schR r).levels.size - 2)).map fun i => levelProof r ws k (i + 1)).toArray)[i]'(by
        simpa using hi') = _
      simp
    rw [e]
    exact levelOk_levelProof r ws k (i + 1) hH hsl (hg (i + 1)) (by rw [hl4 (i + 1) (by omega) (by omega)] <;> rfl)
  · exact levelOk_levelProof r ws k ((schR r).levels.size - 1) hH hsl (hg _)
      (by rw [hl4 ((schR r).levels.size - 1) (by omega) (by omega)] <;> rfl)

/-! ## A rep's proof is shaped for its file -/

theorem readRep_absorbed {c : FlockDeploy.RepCtx} {rows : List (List ByteArray)} {q q₀ p : Flock.Proof}
    {ms : List Msg} (h : FlockDeploy.readRep c rows q = some (p, ms)) (hq : q = q₀) :
    ∃ a, p = FlockDeploy.absorbedFrom c a q₀ := by
  subst hq
  by_cases hx : ∃ pT : Flock.Proof × F128, FlockDeploy.FitsRep c rows pT.1 pT.2
  · rw [FlockDeploy.readRep, dif_pos hx] at h
    exact ⟨_, (Prod.mk.inj (Option.some.inj h)).1.symm⟩
  · rw [FlockDeploy.readRep, dif_neg hx] at h
    cases h

theorem ps_get {r : XR} {ws : List Claims.Flock.Wit} {k : ℕ} (hk : k < nProofs r) :
    ((((List.range (nProofs r)).map (proofR r ws)).map opensOf).toArray)[k]! = opensOf (proofR r ws k) := by
  rw [getElem!_pos _ k (by simpa using hk)]
  simp

/-- **Rep `k`'s shape is its context's**, at its context's extra claims' count `e`, which is at most 2,048. -/
theorem shapeAt_n {spec : SessionSpec} {sts : Array Stmt} {c₁ : Coin} {r₀ : List ByteArray} {cs : List Coin}
    {k e : ℕ} (hn : (FlockDeploy.repCtx spec sts c₁ r₀ cs k).n = 2 + e)
    (hb : ∀ E, (FlockDeploy.tabsOf sts)[k / 2]!.extra (FlockDeploy.pointsOf spec c₁) = .ok E → E.size ≤ 2048) :
    FlockDeploy.shapeAt (FlockDeploy.tabsOf sts) (FlockDeploy.pointsOf spec c₁) k =
        shapeR (FlockDeploy.repCtx spec sts c₁ r₀ cs k).st
          (FlockDeploy.repCtx spec sts c₁ r₀ cs k).sch e ∧ e ≤ 2048 := by
  cases hx : (FlockDeploy.tabsOf sts)[k / 2]!.extra (FlockDeploy.pointsOf spec c₁) with
  | ok E =>
    have hE : E.size = e := by
      simp only [FlockDeploy.repCtx, hx] at hn
      omega
    subst hE
    refine ⟨?_, hb E hx⟩
    simp only [FlockDeploy.shapeAt, hx]
    first | rfl | skip
  | error _ =>
    have hE : 0 = e := by
      simp only [FlockDeploy.repCtx, hx] at hn
      omega
    subst hE
    refine ⟨?_, by omega⟩
    simp only [FlockDeploy.shapeAt, hx]
    first | rfl | skip

/-- **A proof with a fitting rep's absorbed fields and plain openings of its levels' shapes is shaped for its file**,
at a plain SHA-512 table of a `fast100` schedule with at most 2,048 extra claims. -/
theorem shaped_of {c : FlockDeploy.RepCtx} {st : Stmt} {sch : Flock.Schedule} {e : ℕ} {a q p : Flock.Proof}
    (hp : p = FlockDeploy.absorbedFrom c a q)
    (hms : st.merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted") (hsch : fast100 st.m = some sch)
    (hk : st.c.kLog ≤ 35) (he : e ≤ 2048)
    (hs : ProofSizes (shapeR (Setup.ofCircuit st) sch e) p)
    (hnz : ZNoncesZero p.ligerito)
    (h0 : ZLevelOk (shapeR (Setup.ofCircuit st) sch e) (sch.levels.getD 0 default)
      (2 ^ (sch.levels.getD 0 default).lanes) q.ligerito.initialProof)
    (hR : q.ligerito.recursiveProofs.size = sch.levels.size - 1 - 1)
    (hRi : ∀ i (h : i < q.ligerito.recursiveProofs.size),
      ZLevelOk (shapeR (Setup.ofCircuit st) sch e) (sch.levels.getD (i + 1) default) 16
        q.ligerito.recursiveProofs[i])
    (hF : ZLevelOk (shapeR (Setup.ofCircuit st) sch e)
      (sch.levels.getD (sch.levels.size - 1) default) 16
      ⟨q.ligerito.finalProof.rows, q.ligerito.finalProof.siblings, q.ligerito.finalProof.salts⟩) :
    FlockVBridge.Family.ProofShaped (shapeR (Setup.ofCircuit st) sch e) p := by
  subst hp
  have hms' : (Setup.ofCircuit st).merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted" := hms
  obtain ⟨hH, hsl⟩ := FlockVBridge.Family.shapeOf_plain (sch := sch) (n := e) hms'
  obtain ⟨hf2, hfs, hf35⟩ := FlockVBridge.Family.fast100_file _ _ hsch
  have hL : FlockVBridge.Family.Lengths (shapeR (Setup.ofCircuit st) sch e)
      (FlockDeploy.absorbedFrom c a q) :=
    ⟨⟨hs.cap, hs.capEach⟩, hs.r1ab, hs.r1c, hs.zcRounds, hs.lcRounds, hs.zPartial, hs.rs, hs.rsEach,
      ⟨hs.cap, hs.capEach⟩, h0, hs.rcaps, hs.rcapsEach, hR, hRi, hs.yr, hF, hs.messages, hs.queryNonces,
      hs.oodValues, hs.claimNonces, hs.consistencyNonces, hnz⟩
  refine ⟨by rw [hH]; norm_num, ?_, lt_of_le_of_lt (FlockVBridge.Family.encodeProof_size_le _ _ hL)
    (FlockComplete.JsonRT.fileBound_lt ?_ ?_ ?_ ?_), hL⟩
  · show 1 ≤ sch.levels.size - 1
    omega
  · rw [hH]
    exact le_trans (FlockVBridge.Family.schedPart_plain hsl) hfs
  · exact hf35
  · exact hk
  · show 2 + e ≤ 2050
    omega

/-- **The proofs read off the opened rounds are shaped for their files** at `elemR`, at every witness and every
decision `rdR` reads: `OpenedShaped elemR`. -/
theorem openedShaped_elemR : RestRead.OpenedShaped elemR := by
  intro x hx c ws hl _ os last hf _ v hv fl hfl pf hpf
  obtain ⟨st0, hst0, hd⟩ := id hx
  obtain ⟨s, spec, st, sch, hset, htags, hpi, hsch⟩ := Rows.dense_inner hst0 hd
  rw [hset] at hv
  obtain rfl := Option.some.inj hv
  obtain ⟨r, hr, hlast⟩ := finalR_last elemR hx hl hf
  have hn : nS x = r.w.parts.length := by simp [nS, hr]
  have hpos : 0 < r.w.parts.length := by
    obtain ⟨_, -, h0⟩ := capAt_readX hr
    omega
  rw [hlast, rdR_elemR r (hl.trans hn) hpos] at hfl
  obtain rfl := Option.some.inj hfl
  have hms := dense_merkle hset htags hpi
  have h22 : ∀ t ∈ (#[st] : Array Stmt), 22 ≤ t.m := by
    intro t ht
    obtain rfl : t = st := by simpa using ht
    exact (FlockVBridge.Family.fast100_range hsch).1
  have hty : s.read.tags.typed = false := by rw [htags]; rfl
  obtain ⟨hok, hext⟩ := FlockDeploy.tablesOk_of_read hset hty hpi h22 0 (by simp)
  have hok' : FlockComplete.JsonRT.TableOk st := hok
  have hk35 : st.c.kLog ≤ 35 := by
    have h1 := hok'.m35
    have h2 := hok'.m
    omega
  have hsR : schR r = sch := by
    show OpenLower.schAt (stmtOf r) 0 = sch
    rw [stmtOf_eq hr]
    unfold OpenLower.schAt
    rw [hset]
    show (fast100 (FlockDeploy.tabsOf #[st])[0]!.m).getD default = sch
    rw [FlockDeploy.tabsOf_get! (sts := #[st]) (i := 0) (by simp)]
    show (fast100 st.m).getD default = sch
    simp only [hsch, Option.getD_some]
  obtain ⟨⟨-, h2, -, -, hl4, -, -⟩, -⟩ := FlockSoundness.Refine.fast100_dims _ _ hsch
  unfold FlockDeploy.readPs at hpf
  split at hpf
  next hcond =>
    obtain ⟨rs, hrs, rfl⟩ := Option.map_eq_some_iff.1 hpf
    obtain ⟨hlen, hget⟩ := FlockDeploy.mapM_some_get hrs
    have hnp : nProofs r = 2 := by simpa using hcond.1
    have hrl : rs.length = 2 := by simpa using hlen
    intro k hk
    have hk' : k < rs.length := by simpa using hk
    have hk2 : k < 2 := by omega
    suffices hsh : FlockVBridge.Family.ProofShaped
        (FlockDeploy.shapeAt (FlockDeploy.tabsOf #[st]) (FlockDeploy.pointsOf spec (c 0)) k) rs[k].1 by
      simpa using hsh
    have hget' := hget k (by simp; omega) hk'
    simp only [List.getElem_range] at hget'
    obtain ⟨T, hfit, -⟩ := FlockDeploy.readRep_fits (p := rs[k].1) (ms := rs[k].2) hget'
    obtain ⟨a, ha⟩ := readRep_absorbed (p := rs[k].1) (ms := rs[k].2) hget' (ps_get (by omega))
    have htab : (FlockDeploy.tabsOf #[st])[k / 2]! = Setup.ofCircuit st := by
      rw [show k / 2 = 0 by omega]
      exact FlockDeploy.tabsOf_get! (sts := #[st]) (i := 0) (by simp)
    have hsch' : (fast100 (Setup.ofCircuit st).m).getD default = sch := by
      show (fast100 st.m).getD default = sch
      simp only [hsch, Option.getD_some]
    obtain ⟨e, hne, hsz⟩ := hfit.sizes
    obtain ⟨hshape, hle⟩ := shapeAt_n hne fun E hE => hext (c 0) E (by rw [htab] at hE; exact hE)
    rw [hshape]
    show FlockVBridge.Family.ProofShaped (shapeR (FlockDeploy.tabsOf #[st])[k / 2]!
      ((fast100 (FlockDeploy.tabsOf #[st])[k / 2]!.m).getD default) e) rs[k].1
    rw [htab, hsch']
    have hsz' : ProofSizes
        (shapeR (Setup.ofCircuit st) sch e) rs[k].1 := by
      have h := hsz
      change ProofSizes (shapeR
        (FlockDeploy.tabsOf #[st])[k / 2]! ((fast100 (FlockDeploy.tabsOf #[st])[k / 2]!.m).getD default) e) rs[k].1
        at h
      rw [htab, hsch'] at h
      exact h
    obtain ⟨hH, hsl⟩ := FlockVBridge.Family.shapeOf_plain (st := Setup.ofCircuit st) (sch := sch) (n := e) hms
    obtain ⟨h0, hR, hRi, hF⟩ := opens_ok r ws k hsR hH hsl hl4 h2
    exact shaped_of ha hms hsch hk35 hle hsz' hfit.nonces h0 hR hRi hF
  next => cases hpf

/-- **The last step reads at the deployed readers at `elemR`**, wherever every rep's opened rounds fit
(`OpenedFit elemR`): `ElemFramed` and `OpenedShaped` hold at `elemR` (`elemFramed_elemR`, `openedShaped_elemR`). -/
theorem readsAt_elemR (hfit : RestRead.OpenedFit elemR) : RestRead.ReadsAt rdR finalR (restR elemR) :=
  RestRead.readsAt elemR elemFramed_elemR hfit openedShaped_elemR

/-! ## `OpenedFit` from the rows' sizes -/

/-- **The opened rounds have an honest rep's byte sizes**: at the same statements, witnesses and coins as
`OpenedFit`, each rep's context is its frames' schedule (`SchedOf`), and some honest proof at it (`HonestRep`, its cap
the context's) commits rows of the byte sizes of the rep's block of the opened rounds after the link. -/
def OpenedSized (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) : Prop :=
  ∀ x, DenseX x → ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit), ws.length = nS x →
    (∀ e ∈ (RestRead.layAt (restR elem)).allCalls x ws, FlockDeploy.sha512 e.1 = e.2) →
    ∀ os last, finalR (RestRead.lastAt (restR elem) x ws) = some (os, last) → os.length + 1 = RestRead.kI x →
      ∀ v, FlockDeploy.setupOf false (xIOf x) = some v → ∀ k < 2 * v.2.2.size,
        FlockDeploy.SchedOf (FlockDeploy.repCtx v.2.1 v.2.2 (c 0) (RestRead.nodesR (os.headD []))
            ((List.range (RestRead.kI x)).map c).tail.dropLast k) ∧
          ∃ (p : Flock.Proof) (T : F128),
            FlockDeploy.HonestRep (FlockDeploy.repCtx v.2.1 v.2.2 (c 0) (RestRead.nodesR (os.headD []))
              ((List.range (RestRead.kI x)).map c).tail.dropLast k) p ∧
            p.cap.toList = (FlockDeploy.repCtx v.2.1 v.2.2 (c 0) (RestRead.nodesR (os.headD []))
              ((List.range (RestRead.kI x)).map c).tail.dropLast k).cap0 ∧
            ((FlockDeploy.cutLens (FlockDeploy.repLens v.2.1 v.2.2) (os.tail.map RestRead.rowsR)).getD k []).map
                (·.map ByteArray.size) =
              (FlockDeploy.repRows (FlockDeploy.repCtx v.2.1 v.2.2 (c 0) (RestRead.nodesR (os.headD []))
                ((List.range (RestRead.kI x)).map c).tail.dropLast k) p T).map (·.map ByteArray.size)

/-- **Every rep's opened rounds fit where they have an honest rep's sizes**: the honest proof fits its own rows
(`repFits`), and a fit lifts to rows of the same byte sizes (`fitsLift`). -/
theorem openedFit_of (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (hS : OpenedSized elem) : RestRead.OpenedFit elem := by
  intro x hx c ws hl hT os last hf hos v hv k hk
  obtain ⟨hsc, p, T, hp, hcap, hsz⟩ := hS x hx c ws hl hT os last hf hos v hv k hk
  exact RestRead.fitsLift _ hsc.2.1 _ _ p T (FlockDeploy.repFits _ p T hsc hp hcap) hsz

/-- **The last step reads at the deployed readers at `elemR`** wherever the opened rounds have an honest rep's
sizes. -/
theorem readsAt_elemR' (hS : OpenedSized elemR) : RestRead.ReadsAt rdR finalR (restR elemR) :=
  readsAt_elemR (openedFit_of elemR hS)

end RecDeploy.RestElem
