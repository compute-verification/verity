import Proofs.Flock.Claims.Complete.Replay.RingSwitch
import Proofs.Flock.Claims.Complete.Discharge

/-!
# The honest trace at the deployed relation

At the deployed relation (`Readable.relation`) the readable circuit's inputs in block `b` are every hidden bit of every
table's block `b` (`Sys.rel`), so the witness's hidden bits are the whole trace, and `traceOf` reads table `0`'s bit
`a` off hidden value `a`. `RelS` says the circuit maps those inputs to its outputs: its row gates are every row's AND
(`Sys.bit_row`), equal to the trace bits themselves (`Sys.rel`'s `out`), then `1` at every pin and the public value of
every region bit. So the trace satisfies every row, its pin is `1`, and its region bits are the regions'.

* `Case.parsed`: the statement the verifier parsed at a good case has the case's tables, and the witness satisfies it.
* `Case.holds`: at the deployed relation, every block's rows, pin and region bits hold on the witness's bits.
* `traceR1CS`: each table's trace satisfies the block R1CS (`TraceR1CS`), from `RelS` alone.
* `zerocheck_goal'`, `lincheck_goal'`, `ringSwitch_goal'`: the goals at the deployed relation, without `TraceR1CS`.
-/

namespace FlockComplete.Replay

open Flock FlockDeploy

variable {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}

/-- **The statement the verifier parsed** at a good case: its tables are the case's, and the witness satisfies it. -/
theorem _root_.FlockComplete.Case.parsed (c : Case H rel) :
    ∃ p : Readable.Parsed, Readable.parseBytes (readOf false) c.x = some p ∧ p.sts = c.sts ∧
      Claims.Flock.RelS (Readable.toReadable rel p) c.w := by
  obtain ⟨s, hs, hR⟩ := c.rel_w
  change (Readable.parseBytes (readOf false) c.x).map (Readable.toReadable rel) = some s at hs
  obtain ⟨p, hp, rfl⟩ := Option.map_eq_some_iff.1 hs
  refine ⟨p, hp, ?_, hR⟩
  have hi := (setupOf_some c.setup).1
  unfold Readable.parseBytes readOf at hp
  rw [hi] at hp
  have ht := Readable.toRead_of_parse (r := c.s.read) hp
  have htab := Readable.tablesOf_of_parses (Readable.parses_of_parse hp)
  obtain ⟨spec', hset'⟩ := Readable.setup_of_tablesOf htab
  have hset := Ins.setup_read (setupOf_some c.setup).2
  rw [ht] at hset'
  have := hset'.symm.trans hset
  simp only [Except.ok.injEq, Prod.mk.injEq] at this
  exact this.2

/-- A one-table system in block `b`: the table's rows, pin and region bits on the bits from `b·2^kLog`. -/
theorem holds_one {m kLog : ℕ} {t : Readable.Tab} (hf : t.Fits m kLog) {w : Claims.Flock.Wit} {b : ℕ}
    (h : (Readable.sysOf m kLog [t]).Holds w b) :
    (∀ i < 2 ^ kLog, (Readable.xsum (t.a i) (fun j => w.hid.getD (b * 2 ^ kLog + j) false) (2 ^ kLog) &&
        Readable.xsum (t.b i) (fun j => w.hid.getD (b * 2 ^ kLog + j) false) (2 ^ kLog)) =
      w.hid.getD (b * 2 ^ kLog + i) false) ∧
    w.hid.getD (b * 2 ^ kLog + t.pin) false = true ∧
    ∀ f ∈ t.fixed, w.hid.getD (b * 2 ^ kLog + f.1) false = f.2 b := by
  obtain ⟨hr, hp, hs⟩ := h
  have hb : (Readable.sysOf m kLog [t]).bit w 0 b = fun a => w.hid.getD (b * 2 ^ kLog + a) false := funext fun a => by
    simp [Readable.Sys.bit, Readable.sysOf]
  refine ⟨fun i hi => ?_, ?_, fun f hfm => ?_⟩
  · have := hr 0 (by simp [Readable.sysOf]) i hi
    rw [hb] at this
    exact this
  · have := hp 0 (by simp [Readable.sysOf])
    rw [hb] at this
    have e : (Readable.sysOf m kLog [t]).pin 0 = t.pin := by
      simp only [Readable.sysOf]
      exact ite_eq_left hf
    rwa [e] at this
  · have := hs (0, f.1, f.2) (by
      simp only [Readable.sysOf, List.length_singleton, List.range_one, List.flatMap_cons, List.flatMap_nil,
        List.append_nil, List.getD_cons_zero, Readable.fixedOf, ite_eq_left hf]
      exact List.mem_map.2 ⟨f, hfm, rfl⟩)
    rw [hb] at this
    exact this

/-- **The relation at a good case** of the deployed relation: one table `st`, which fits its blocks, and in every
block its rows, its pin and its region bits hold on the witness's bits from `b·2^kLog` (table `0`'s block `b`). -/
theorem _root_.FlockComplete.Case.holds (c : Case H Readable.relation) :
    ∃ st, c.sts = #[st] ∧ (Readable.Tab.ofStmt st).Fits st.m st.c.kLog ∧
      ∀ b < 2 ^ (st.m - st.c.kLog),
        (∀ i < 2 ^ st.c.kLog,
          (Readable.xsum ((Readable.Tab.ofStmt st).a i) (fun j => c.w.hid.getD (b * 2 ^ st.c.kLog + j) false)
              (2 ^ st.c.kLog) &&
            Readable.xsum ((Readable.Tab.ofStmt st).b i) (fun j => c.w.hid.getD (b * 2 ^ st.c.kLog + j) false)
              (2 ^ st.c.kLog)) =
            c.w.hid.getD (b * 2 ^ st.c.kLog + i) false) ∧
        c.w.hid.getD (b * 2 ^ st.c.kLog + st.pin) false = true ∧
        ∀ f ∈ (Readable.Tab.ofStmt st).fixed, c.w.hid.getD (b * 2 ^ st.c.kLog + f.1) false = f.2 b := by
  obtain ⟨st, hsts, -, -, -⟩ := c.table
  obtain ⟨p, -, hpe, hR⟩ := c.parsed
  rw [hsts] at hpe
  have hfit := (Readable.relS_relation p c.w).1 hR
  have hsys := (Readable.Sys.relS_iff (Readable.sysOf_wf _ _ _) (Readable.toReadable Readable.relation p) c.w rfl rfl
    rfl rfl rfl).1 hR
  have hm : (Readable.toReadable Readable.relation p).m = st.m := by simp [Readable.toReadable, hpe]
  have hk : (Readable.toReadable Readable.relation p).kLog = st.c.kLog := by simp [Readable.toReadable, hpe]
  obtain ⟨hf, -⟩ := hfit 0 (by simp [hpe])
  have h0 : p.sts[0]! = st := by simp [hpe]
  rw [h0, hm, hk] at hf
  refine ⟨st, hsts, hf, fun b hb => ?_⟩
  have e : Readable.sysOf ((p.sts[0]?.map (·.m)).getD 0) ((p.sts[0]?.map (·.c.kLog)).getD 0)
      (p.sts.toList.map Readable.Tab.ofStmt) = Readable.sysOf st.m st.c.kLog [Readable.Tab.ofStmt st] := by
    simp [hpe]
  rw [e] at hsys
  exact holds_one hf (hsys b hb)

/-! ## The trace -/

theorem xsum_congrA {A A' x : ℕ → Bool} {n : ℕ} (h : ∀ r < n, A r = A' r) :
    Readable.xsum A x n = Readable.xsum A' x n := by
  induction n with
  | zero => rfl
  | succ n ih => rw [Readable.xsum_succ, Readable.xsum_succ, ih fun r hr => h r (by omega), h n (by omega)]

/-- A row of `mulBlocks` is its block's XOR. -/
theorem mulBlocks_get (entry : ℕ → ℕ → Bool) (kLog : ℕ) (z : Array Bool) {x : ℕ} (hx : x < z.size) :
    (mulBlocks entry kLog z)[x]! =
      Readable.xsum (entry (x % 2 ^ kLog)) (fun j => z[x / 2 ^ kLog * 2 ^ kLog + j]!) (2 ^ kLog) := by
  unfold mulBlocks Readable.xsum
  rw [getElem!_pos _ x (by simpa using hx)]
  simp only [Array.getElem_map, Array.getElem_range]
  congr 1
  funext acc j
  cases entry (x % 2 ^ kLog) j && z[x / 2 ^ kLog * 2 ^ kLog + j]! <;> cases acc <;> rfl

/-- `entryA` is the table's A side in its block. -/
theorem entryA_eq (st : Stmt) {r j : ℕ} (hr : r < 2 ^ st.c.kLog) (hj : j < 2 ^ st.c.kLog) :
    entryA st r j = (Readable.Tab.ofStmt st).a r j := by
  simp only [Readable.Tab.a, Readable.Tab.ofStmt, hr, hj, and_self, ↓reduceDIte]
  rfl

/-- `entryB` is the table's B side in its block. -/
theorem entryB_eq (st : Stmt) {r j : ℕ} (hr : r < 2 ^ st.c.kLog) (hj : j < 2 ^ st.c.kLog) :
    entryB st r j = (Readable.Tab.ofStmt st).b r j := by
  simp only [Readable.Tab.b, Readable.Tab.ofStmt, hr, hj, and_self, ↓reduceDIte]
  rfl

/-- **The trace of a one-table run**: table `0`'s trace reads the witness's bits from `0`. -/
theorem _root_.FlockComplete.Run.trace_one (o : Run H rel) {st : Stmt} (hsts : o.sts = #[st]) :
    o.trace 0 = traceOf st o.w.hid.toArray 0 := by
  simp [Run.trace, tracesOf, hsts]

/-- Table `0`'s trace bit `a` is the witness's. -/
theorem traceOf_z (st : Stmt) (l : List Bool) {a : ℕ} (ha : a < 2 ^ st.m) :
    (traceOf st l.toArray 0).z[a]! = l.getD a false := by
  simp only [traceOf]
  rw [getElem!_pos _ a (by simpa using ha)]
  simp

theorem block_lt {m k x j : ℕ} (hk : k ≤ m) (hx : x < 2 ^ m) (hj : j < 2 ^ k) : x / 2 ^ k * 2 ^ k + j < 2 ^ m := by
  have e : 2 ^ m = 2 ^ (m - k) * 2 ^ k := by rw [← pow_add, Nat.sub_add_cancel hk]
  have hb : x / 2 ^ k < 2 ^ (m - k) := (Nat.div_lt_iff_lt_mul (Nat.two_pow_pos _)).2 (e ▸ hx)
  have := Nat.mul_le_mul_right (2 ^ k) (show x / 2 ^ k + 1 ≤ 2 ^ (m - k) by omega)
  rw [Nat.succ_mul] at this
  omega

/-- **Each table's trace satisfies the block R1CS** at the deployed relation, from `RelS`: the trace is the witness's
bits, and `RelS` says every row's AND is its bit. -/
theorem traceR1CS (H : Claims.Msg → Claims.Msg) : TraceR1CS H Readable.relation := by
  intro o i hi x hx
  obtain ⟨st, hsts, hf, hb⟩ := o.holds
  have hi0 : i = 0 := by
    have : o.tabs.size = 1 := by simp [Run.tabs, tabsOf, hsts]
    omega
  subst hi0
  have hm : o.tabs[0]!.m = st.m := by simp [Run.tabs, tabsOf, hsts, Setup.ofCircuit]
  rw [hm] at hx
  have hk : st.c.kLog ≤ st.m := hf.2.2.2.1.trans_eq hf.1
  rw [o.trace_one hsts]
  set N := 2 ^ st.c.kLog with hN
  have hN0 : 0 < N := Nat.two_pow_pos _
  have hzs : (traceOf st o.w.hid.toArray 0).z.size = 2 ^ st.m := by simp [traceOf]
  have hbl : x / N < 2 ^ (st.m - st.c.kLog) := by
    refine (Nat.div_lt_iff_lt_mul hN0).2 ?_
    rw [hN, ← pow_add, Nat.sub_add_cancel hk]
    exact hx
  obtain ⟨hrow, -, -⟩ := hb (x / N) hbl
  have hz : ∀ j < N, (traceOf st o.w.hid.toArray 0).z[x / N * N + j]! = o.w.hid.getD (x / N * N + j) false :=
    fun j hj => traceOf_z st _ (block_lt hk hx hj)
  have hxm : x % N < N := Nat.mod_lt _ hN0
  simp only [traceOf]
  rw [mulBlocks_get _ _ _ (by simpa using hx), mulBlocks_get _ _ _ (by simpa using hx)]
  have hz' := traceOf_z st o.w.hid hx
  simp only [traceOf] at hz hz'
  rw [Readable.Sys.xsum_congr _ hz, Readable.Sys.xsum_congr _ hz, hz',
    xsum_congrA fun j hj => entryA_eq st hxm hj, xsum_congrA fun j hj => entryB_eq st hxm hj]
  have := hrow (x % N) hxm
  rw [Nat.mul_comm (x / N) N] at this ⊢
  rwa [Nat.div_add_mod] at this

/-! ## The goals at the deployed relation -/

/-- **The zerocheck goal** at the deployed relation, at every rep the session walk aligns. -/
theorem zerocheck_goal' (H : Claims.Msg → Claims.Msg)
    (hzc : ∀ o : Run H Readable.relation, ∀ i < o.tabs.size, ∀ r < 2, ZcAt o i r) : Zerocheck H Readable.relation :=
  zerocheck_goal H _ (traceR1CS H) hzc

/-- **The lincheck goal** at the deployed relation, at every rep the session walk aligns, given the lincheck's
algebra. -/
theorem lincheck_goal' (H : Claims.Msg → Claims.Msg)
    (hzc : ∀ o : Run H Readable.relation, ∀ i < o.tabs.size, ∀ r < 2, ZcAt o i r)
    (hlc : ∀ o : Run H Readable.relation, ∀ i < o.tabs.size, ∀ r < 2, LcAt o i r)
    (hal : LcAlgebra H Readable.relation) : Lincheck H Readable.relation :=
  lincheck_goal H _ (traceR1CS H) hzc hlc hal

/-- **The ring-switch goal** at the deployed relation, at every rep the session walk aligns, given the lincheck's and
the ring switch's algebra. -/
theorem ringSwitch_goal' (H : Claims.Msg → Claims.Msg)
    (hzc : ∀ o : Run H Readable.relation, ∀ i < o.tabs.size, ∀ r < 2, ZcAt o i r)
    (hlc : ∀ o : Run H Readable.relation, ∀ i < o.tabs.size, ∀ r < 2, LcAt o i r)
    (hal : LcAlgebra H Readable.relation)
    (hrs : ∀ o : Run H Readable.relation, ∀ i < o.tabs.size, ∀ r < 2, RsAt o i r)
    (hra : RsAlgebra H Readable.relation) : RingSwitch H Readable.relation :=
  ringSwitch_goal H _ (traceR1CS H) hzc hlc hal hrs hra

end FlockComplete.Replay
