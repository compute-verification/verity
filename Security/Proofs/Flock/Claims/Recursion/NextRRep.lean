import Proofs.Flock.Claims.Recursion.NextRShaped
import Proofs.Flock.Claims.Complete.Shaped
import Proofs.Flock.Claims.Complete.Discharge
import Proofs.Flock.Claims.Complete.Replay.ZerocheckGoal

/-!
# The honest rep's proof at any table

`FlockComplete.repShaped` proves the honest rep's proof `ProofShaped` at a plain case, which has one table
(`Case.table`). Its proof reads only that table's facts (`TableOk`: `22 ≤ m ≤ 35`, an untyped row circuit the scope
parsed, `m = k_log + nbl`, `6 ≤ k_log`, a registered scheme) and a bound on the extra claims, so it holds at any table
that has them (`repShapedOk`). At rep `k` of a set-up statement, table `k / 2`'s returned proof is then an honest rep of
`repCtx` wherever its `finalC` is at the reader's `z` (`honestRep_of_repP`): its nonces are zero, as the shape has
them, and `repP` sets `params` and `initialCap` as V\* takes them (`repP_unabsorbed`). That is the per-rep hypothesis
`ReadsBack` takes, less the schedule (`schedOf_repCtx`) and the cap.

The plain case's one table is `TableOk` (`Case.tableOk`); a statement of several plain tables has no deployed relation
yet, so there `TableOk` stays a hypothesis per table.
-/

namespace FlockComplete.JsonRT

open Flock Transcript FlockComplete FlockDeploy FlockSoundness.Refine

/-- **A table the honest rep's proof fits at**: what `Case.table` and `Case.scheme` give a plain case's one table. -/
structure TableOk (st : Stmt) : Prop where
  m22 : 22 ≤ st.m
  m35 : st.m ≤ 35
  circ : ∃ tags file tables, tags.typed = false ∧ ProvedScope.circuitOf tags file tables = .ok st.c
  m : st.m = st.c.kLog + st.nbl
  k : 6 ≤ st.c.kLog
  scheme : st.merkle ∈ MerkleScheme.all

/-- **A plain case's one table is `TableOk`.** -/
theorem _root_.FlockComplete.Case.tableOk {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}
    (c : Case H rel) : ∃ st, c.sts = #[st] ∧ TableOk st := by
  obtain ⟨st, hst, -, h22, h35, hc, hm, hk⟩ := c.table
  obtain ⟨st', hst', -, hall, -⟩ := c.scheme
  rw [hst] at hst'
  obtain rfl : st = st' := by simpa using hst'
  exact ⟨st, hst, h22, h35, hc, hm, hk, hall⟩

/-- **The honest rep proof fits its shape at any `TableOk` table**: `repShaped` on the table's facts, on any trace,
with at most 2,048 extra claims. -/
theorem repShapedOk {H : Claims.Msg → Claims.Msg} (h64 : Claims.Out64 H) {st : Stmt} (hok : TableOk st)
    {sch : Schedule} (hsch : fast100 (Setup.ofCircuit st).m = some sch) (resp : ℕ → Claims.Msg) (tr : Trace)
    {salt₀ : ℕ → ByteArray} {cm : Commit}
    (hcm : commitP (Hashes.ofSha512 (Claims.hOf H) : Hashes V) (Setup.ofCircuit st).merkle
      (sch.levels.getD 0 default) (packed tr (Setup.ofCircuit st).m) salt₀ = .ok cm)
    {e : Array Claim} (he : e.size ≤ 2048) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) (d : ByteArray) :
    Ret ((repP (asksS resp) (Hashes.ofSha512 (Claims.hOf H)) (Setup.ofCircuit st) sch tr
      (packed tr (Setup.ofCircuit st).m) cm e salt nonce).run (Transcript.init #[] d))
      fun pt => FlockVBridge.Family.ProofShaped (shapeOf (Setup.ofCircuit st) sch e.size) pt.1 := by
  obtain ⟨-, hm35, ⟨tags, file, tables, hty, hcirc⟩, hm, hk, hall⟩ := hok
  obtain ⟨ls, hcheck⟩ := circuitOf_check hty hcirc
  obtain ⟨hL, hsm⟩ := fast100_lig _ _ hsch
  obtain ⟨hn, hlf⟩ := opsWith_sizes h64 hall
  have hc0 := commitP_shape hn hlf (hL.cap 0 (by have := hL.two; omega)) hcm
  have ht : (packed tr (Setup.ofCircuit st).m).size = 2 ^ sch.n := by
    rw [packed_size, (fast100_dims _ _ hsch).1.1, hsm]
  have hT := repP_shape resp (Claims.hOf H) st hk (by omega) (hm_check_kinds hcheck) hn hlf hL tr ht hc0 e salt
    nonce
  have hsc := scheme_sizes hall
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

/-- **Ligerito's proof starts at the commitment's cap**: `ligeritoP` returns `cm0.cap` as `initialCap`. -/
theorem ligeritoP_initialCap (resp : ℕ → Claims.Msg) (hs : Hashes PS) (ms : MerkleScheme) (sch : Schedule)
    (T : F128) (cm0 : Commit) (t b : Array F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (t₀ : Transcript) :
    Ret ((ligeritoP (asksS resp) hs ms sch T cm0 t b salt nonce).run t₀) fun r => r.1.initialCap = cm0.cap := by
  unfold ligeritoP
  refine Replay.Ret.run_bind (Ret.true _) fun _ _ _ => ?_
  refine Replay.Ret.run_bind (Ret.true _) fun _ _ _ => ?_
  refine Replay.Ret.run_bind (Ret.true _) fun _ _ _ => ?_
  refine Replay.Ret.run_bind (Ret.true _) fun ⟨⟨_, _, _, _⟩, _⟩ _ _ => ?_
  rw [StateT.run_pure]
  exact Ret.pure rfl

/-- **`repP` sets the fields V\* takes unabsorbed**: `params` as `pcsParams`, and the level-0 cap as `initialCap`
(`ligeritoP_initialCap`). -/
theorem repP_unabsorbed (resp : ℕ → Claims.Msg) (hs : Hashes PS) (st : Setup) (sch : Schedule) (tr : Trace)
    (t : Array F128) (cm0 : Commit) (extras : Array Claim) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (t₀ : Transcript) :
    Ret ((repP (asksS resp) hs st sch tr t cm0 extras salt nonce).run t₀) fun pt =>
      pt.1.params = pcsParams st.merkle sch ∧ pt.1.ligerito.initialCap = pt.1.cap := by
  unfold repP
  refine Replay.Ret.run_bind (Ret.true _) fun ⟨_, _⟩ _ _ => ?_
  refine Replay.Ret.run_bind (Ret.true _) fun ⟨_, _⟩ _ _ => ?_
  refine Replay.Ret.run_bind (Ret.true _) fun ⟨_, _, _⟩ _ _ => ?_
  refine Replay.Ret.run_bind (ligeritoP_initialCap resp hs _ _ _ cm0 _ _ salt nonce _) fun _ _ h => ?_
  rw [StateT.run_pure]
  exact Ret.pure ⟨rfl, h⟩

/-- **The honest rep's `finalC` is its zerocheck's at its `z`**: `repP`'s zerocheck is the prover's
(`Replay.repP_zerocheck`), whose `finalC` is `round1c` at the coin of the reply to the rep's third send. What is left
for `HonestRep.finalC` is that the reader's `z` is that coin. -/
theorem repP_finalC (resp : ℕ → Claims.Msg) (hs : Hashes PS) {st : Setup} (sch : Schedule) (tr : Trace)
    (t : Array F128) (cm0 : Commit) (extras : Array Claim) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (hm : 14 ≤ st.m) (t₀ : Transcript) {l l' : Array Claims.Msg} {p : Proof} {t' : Transcript}
    (h : ((repP (asksS resp) hs st sch tr t cm0 extras salt nonce).run t₀).go l = (.ok (p, t'), l')) :
    p.zerocheck.finalC = dot (lagrange nodesL den6Inv (coinOf (resp (l.size + 2)) 0)) p.zerocheck.round1c := by
  rw [(Replay.repP_zerocheck resp hs st sch tr t cm0 extras salt nonce hm t₀ l h).1]
  rfl

end FlockComplete.JsonRT

namespace FlockDeploy

open Flock FlockComplete FlockSoundness.Refine

/-- **Table `k / 2`'s returned proof is an honest rep of `repCtx`** at rep `k` of a set-up statement: at a `TableOk`
table with its `fast100` schedule and at most 2,048 extra claims at the first coin's link points, the proof `repP`
returns on any trace, under any salts, is `ProofShaped` (`repShapedOk`), zero nonces included, with V\*'s `params`
and `initialCap` (`repP_unabsorbed`), so it is an honest rep (`honestRep_of_shaped`) wherever its `finalC` is the
zerocheck's at the reader's `z` (`hfc`, a hypothesis here). -/
theorem honestRep_of_repP {H : Claims.Msg → Claims.Msg} (h64 : Claims.Out64 H) {spec : SessionSpec}
    {sts : Array Stmt} {c₁ : Coin} {r₀ : List ByteArray} {cs : List Coin} {k : ℕ} (hk : k / 2 < sts.size)
    (hok : JsonRT.TableOk sts[k / 2]) {sch : Schedule} (hsch : fast100 sts[k / 2].m = some sch)
    {e : Array Claim} (he : sts[k / 2].extraClaims (pointsOf spec c₁) = .ok e) (he2 : e.size ≤ 2048)
    (resp : ℕ → Claims.Msg) (tr : Trace) {salt₀ : ℕ → ByteArray} {cm : Commit}
    (hcm : commitP (Hashes.ofSha512 (Claims.hOf H) : Hashes Flock.V) (Setup.ofCircuit sts[k / 2]).merkle
      (sch.levels.getD 0 default) (packed tr (Setup.ofCircuit sts[k / 2]).m) salt₀ = .ok cm)
    (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) (d : ByteArray) {l l' : Array Claims.Msg}
    {pt : Flock.Proof × Transcript}
    (hrun : ((repP (asksS resp) (Hashes.ofSha512 (Claims.hOf H)) (Setup.ofCircuit sts[k / 2]) sch tr
      (packed tr (Setup.ofCircuit sts[k / 2]).m) cm e salt nonce).run (Transcript.init #[] d)).go l = (.ok pt, l'))
    (hfc : pt.1.zerocheck.finalC =
      dot (lagrange nodesL den6Inv (repCtx spec sts c₁ r₀ cs k).z) pt.1.zerocheck.round1c) :
    HonestRep (repCtx spec sts c₁ r₀ cs k) pt.1 := by
  have hp := JsonRT.repShapedOk h64 hok hsch resp tr hcm he2 salt nonce d l pt l' hrun
  obtain ⟨hpa, hic⟩ := JsonRT.repP_unabsorbed resp _ _ _ _ _ _ _ salt nonce _ l pt l' hrun
  have hk' : k / 2 < (tabsOf sts).size := by
    unfold tabsOf
    rw [Array.size_map]
    exact hk
  have ht : (tabsOf sts)[k / 2]! = Setup.ofCircuit sts[k / 2] := by
    rw [getElem!_pos (tabsOf sts) (k / 2) hk']
    exact Array.getElem_map ..
  have hst : (repCtx spec sts c₁ r₀ cs k).st = Setup.ofCircuit sts[k / 2] := ht
  have hsch' : (repCtx spec sts c₁ r₀ cs k).sch = sch := by
    show (fast100 (tabsOf sts)[k / 2]!.m).getD default = sch
    rw [ht]
    show (fast100 sts[k / 2].m).getD default = sch
    rw [hsch, Option.getD_some]
  have hn : (repCtx spec sts c₁ r₀ cs k).n = 2 + e.size := by
    show 2 + (match (tabsOf sts)[k / 2]!.extra (pointsOf spec c₁) with
      | .ok e => e.size
      | .error _ => 0) = _
    rw [ht]
    show 2 + (match sts[k / 2].extraClaims (pointsOf spec c₁) with
      | .ok e => e.size
      | .error _ => 0) = _
    rw [he]
  refine honestRep_of_shaped hn ?_ ?_ hic hfc
  · rw [hst, hsch']
    exact hp
  · rw [hpa, hst, hsch']

end FlockDeploy
