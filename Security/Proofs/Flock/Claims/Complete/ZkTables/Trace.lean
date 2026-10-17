import Proofs.Flock.Claims.Complete.ZkTables.Scope
import Proofs.Flock.Claims.Complete.Replay.Trace

/-!
# Each table's honest trace at the deployed relation

`Replay/Trace.lean` per table. At the deployed relation (`Readable.relation`) a witness satisfies a statement of `J`
tables exactly when each table's share (`Readable.tableWit J w j`, its bit `a` the witness's bit `a·J + j`) satisfies
that table alone (`Readable.relS_only`). `traces` runs table `j` on its share from bit `0`, so each table's trace is
a one-table trace of a witness of a one-table statement, and satisfies the block R1CS.

* `Case.parsed`: the statement the verifier parsed at a good case is untyped, has no public input ports, has the case's
  tables, and the witness satisfies it.
* `holds_only`: a witness of a one-table statement satisfies its table's rows in every block.
* `traceOf_r1cs`: on such a witness the trace from bit `0` satisfies the block R1CS.
* `Case.traceR1CS`, `traceR1CS_goal`: each table's trace (`traces`) satisfies its block R1CS, at every number of
  tables (`TraceR1CS` at `traces`).
-/

open Flock FlockDeploy

namespace FlockComplete.ZkTables

variable {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}

/-- **The statement the verifier parsed** at a good case: it parses, is untyped, has no public input ports, its tables
are the case's, and the witness satisfies it. -/
theorem Case.parsed (c : Case H rel) :
    ∃ p : Readable.Parsed, Readable.Parses p ∧ p.tags.typed = false ∧ p.publicInputs = none ∧ p.sts = c.sts ∧
      Claims.Flock.RelS (Readable.toReadable rel p) c.w := by
  obtain ⟨p, hp, hsc⟩ := (Readable.scope_iff (deployZk rel) rel (readOf true) rfl
    (fun _ _ h => readOf_zk h) c.x).1 c.scope
  obtain ⟨s', hs', hR'⟩ := c.rel_w
  change (Readable.parseBytes (readOf true) c.x).map (Readable.toReadable rel) = some s' at hs'
  rw [hp] at hs'
  simp only [Option.map_some, Option.some.injEq] at hs'
  subst hs'
  have hi := (setupOf_some c.setup).1
  unfold Readable.parseBytes readOf at hp
  rw [hi] at hp
  have ht := Readable.toRead_of_parse (r := c.s.read) hp
  have hps := Readable.parses_of_parse hp
  have htab := Readable.tablesOf_of_parses hps
  obtain ⟨hc, -⟩ := (Readable.scope_ok_iff p).1 hsc
  have hR := ((Readable.check_ok_iff _ _ _ _ _ _ _ _).1 hc).1
  have hpi : p.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simp [hR.2.2.2.2.2])
  obtain ⟨spec', hset'⟩ := Readable.setup_of_tablesOf htab
  have hset := Ins.setup_read (setupOf_some c.setup).2
  rw [ht] at hset'
  have hpe : p.sts = c.sts := by
    have := hset'.symm.trans hset
    simp only [Except.ok.injEq, Prod.mk.injEq] at this
    exact this.2
  exact ⟨p, hps, hR.2.1, hpi, hpe, hR'⟩

/-- **A witness of a one-table statement** at the deployed relation: the table fits its blocks, and in every block
`b` its rows hold on the witness's bits from `b·2^kLog`. -/
theorem holds_only {p : Readable.Parsed} {st : Stmt} (hpe : p.sts = #[st]) {w : Claims.Flock.Wit}
    (hR : Claims.Flock.RelS (Readable.toReadable Readable.relation p) w) :
    (Readable.Tab.ofStmt st).Fits st.m st.c.kLog ∧
      ∀ b < 2 ^ (st.m - st.c.kLog), ∀ i < 2 ^ st.c.kLog,
        (Readable.xsum ((Readable.Tab.ofStmt st).a i) (fun j => w.hid.getD (b * 2 ^ st.c.kLog + j) false)
            (2 ^ st.c.kLog) &&
          Readable.xsum ((Readable.Tab.ofStmt st).b i) (fun j => w.hid.getD (b * 2 ^ st.c.kLog + j) false)
            (2 ^ st.c.kLog)) =
          w.hid.getD (b * 2 ^ st.c.kLog + i) false := by
  have hfit := (Readable.relS_relation p w).1 hR
  have hsys := (Readable.Sys.relS_iff (Readable.sysOf_wf _ _ _) (Readable.toReadable Readable.relation p) w rfl rfl
    rfl rfl rfl).1 hR
  have hm : (Readable.toReadable Readable.relation p).m = st.m := by simp [Readable.toReadable, hpe]
  have hk : (Readable.toReadable Readable.relation p).kLog = st.c.kLog := by simp [Readable.toReadable, hpe]
  obtain ⟨hf, -⟩ := hfit 0 (by simp [hpe])
  have h0 : p.sts[0]! = st := by simp [hpe]
  rw [h0, hm, hk] at hf
  refine ⟨hf, fun b hb => ?_⟩
  have e : Readable.sysOf ((p.sts[0]?.map (·.m)).getD 0) ((p.sts[0]?.map (·.c.kLog)).getD 0)
      (p.sts.toList.map Readable.Tab.ofStmt) = Readable.sysOf st.m st.c.kLog [Readable.Tab.ofStmt st] := by
    simp [hpe]
  rw [e] at hsys
  exact (Replay.holds_one hf (hsys b hb)).1

/-- **The trace from bit `0` satisfies the block R1CS** on a witness whose bits satisfy the table's rows in every
block. -/
theorem traceOf_r1cs {st : Stmt} {w : Claims.Flock.Wit} (hf : (Readable.Tab.ofStmt st).Fits st.m st.c.kLog)
    (hb : ∀ b < 2 ^ (st.m - st.c.kLog), ∀ i < 2 ^ st.c.kLog,
      (Readable.xsum ((Readable.Tab.ofStmt st).a i) (fun j => w.hid.getD (b * 2 ^ st.c.kLog + j) false)
          (2 ^ st.c.kLog) &&
        Readable.xsum ((Readable.Tab.ofStmt st).b i) (fun j => w.hid.getD (b * 2 ^ st.c.kLog + j) false)
          (2 ^ st.c.kLog)) =
        w.hid.getD (b * 2 ^ st.c.kLog + i) false)
    {x : ℕ} (hx : x < 2 ^ st.m) :
    ((traceOf st w.hid.toArray 0).az[x]! && (traceOf st w.hid.toArray 0).bz[x]!) =
      (traceOf st w.hid.toArray 0).z[x]! := by
  have hk : st.c.kLog ≤ st.m := hf.2.2.2.1.trans_eq hf.1
  set N := 2 ^ st.c.kLog with hN
  have hN0 : 0 < N := Nat.two_pow_pos _
  have hbl : x / N < 2 ^ (st.m - st.c.kLog) := by
    refine (Nat.div_lt_iff_lt_mul hN0).2 ?_
    rw [hN, ← pow_add, Nat.sub_add_cancel hk]
    exact hx
  have hrow := hb (x / N) hbl
  have hz : ∀ j < N, (traceOf st w.hid.toArray 0).z[x / N * N + j]! = w.hid.getD (x / N * N + j) false :=
    fun j hj => Replay.traceOf_z st _ (Replay.block_lt hk hx hj)
  have hxm : x % N < N := Nat.mod_lt _ hN0
  simp only [traceOf]
  rw [Replay.mulBlocks_get _ _ _ (by simpa using hx), Replay.mulBlocks_get _ _ _ (by simpa using hx)]
  have hz' := Replay.traceOf_z st w.hid hx
  simp only [traceOf] at hz hz'
  rw [Readable.Sys.xsum_congr _ hz, Readable.Sys.xsum_congr _ hz, hz',
    Replay.xsum_congrA fun j hj => Replay.entryA_eq st hxm hj,
    Replay.xsum_congrA fun j hj => Replay.entryB_eq st hxm hj]
  have := hrow (x % N) hxm
  rw [Nat.mul_comm (x / N) N] at this ⊢
  rwa [Nat.div_add_mod] at this

/-- Table `i`'s trace is the one-table trace of its share. -/
theorem traces_getElem (sts : Array Stmt) (w : Claims.Flock.Wit) {i : ℕ} (hi : i < sts.size) :
    (traces sts w)[i]! = traceOf sts[i] (Readable.tableWit sts.size w i).hid.toArray 0 := by
  unfold traces
  rw [getElem!_pos _ i (by simpa using hi)]
  simp [getElem!_pos sts i hi]

/-- **Each table's honest trace satisfies its block R1CS** at the deployed relation, at a good case of any number of
tables: table `i`'s share satisfies table `i` alone (`Readable.relS_only`). -/
theorem Case.traceR1CS (c : Case H Readable.relation) {i : ℕ} (hi : i < c.sts.size) {x : ℕ}
    (hx : x < 2 ^ c.sts[i].m) :
    (((traces c.sts c.w)[i]!).az[x]! && ((traces c.sts c.w)[i]!).bz[x]!) = ((traces c.sts c.w)[i]!).z[x]! := by
  obtain ⟨p, hps, hty, hpi, hpe, hR⟩ := c.parsed
  have hJ := (Readable.relS_only hps hty hpi c.w).1 hR i (by rw [hpe]; exact hi)
  rw [hpe] at hJ
  have hone : (p.only i).sts = #[c.sts[i]] := by
    show #[p.sts[i]!] = _
    rw [hpe, getElem!_pos c.sts i hi]
  obtain ⟨hf, hb⟩ := holds_only hone hJ
  rw [traces_getElem c.sts c.w hi]
  exact traceOf_r1cs hf hb hx

/-- **Each table's honest trace satisfies its block R1CS** (`TraceR1CS` at `traces`), at every honest masked run of
the deployed relation of any number of tables. -/
theorem traceR1CS_goal (H : Claims.Msg → Claims.Msg) :
    ∀ o : Run H Readable.relation, ∀ i < o.tabs.size, ∀ x < 2 ^ o.tabs[i]!.m,
      (((traces o.sts o.w)[i]!).az[x]! && ((traces o.sts o.w)[i]!).bz[x]!) = ((traces o.sts o.w)[i]!).z[x]! := by
  intro o i hi x hx
  have hi' : i < o.sts.size := by simpa [Run.tabs, tabsOf] using hi
  have hm : o.tabs[i]!.m = o.sts[i].m := by
    rw [getElem!_pos o.tabs i hi]
    simp [Run.tabs, tabsOf, Setup.ofCircuit]
  rw [hm] at hx
  exact o.toCase.traceR1CS hi' hx

end FlockComplete.ZkTables
