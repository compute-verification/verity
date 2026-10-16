import Proofs.Flock.Claims.KS.Joint
import Proofs.Flock.Claims.Bridge.Custody
import Proofs.Flock.Soundness.Discharge.DecodeZL.Sizes

/-!
# Accepted runs open full-width columns

`ColsOk` (`KS.Joint`) at both deployed records, at every `H`. An accepted run decides on its last message
(`run_acc_next`, `act_decide_final`), whose statement is the statement (`checks`' first check,
`Bridge.Custody.checks_ok`), so the run's columns are read at the statement's own table 0 (`acc_last`, `runOf_m`).
`decodeProof` (`Zk.decode`) reads each opened level-0 row with exactly `2^lanes` (`2^lanes + EXTRA_LANES`) symbols
(`decodeProof_rows`, `zkDecode_rows`), so each column has the `2^lanes` witness lanes (under `--zk`, a row's first
`2^lanes` symbols), the zero column's width (`plain_colsOk`, `zk_colsOk`). That leaves `TableBoundθ`
and `ConflictTied` for `Flock.KS` at SHA-512 (`plain_KS_sha512`, `zk_KS_sha512`); `KS.ConflictTied` proves the second.
-/

namespace FlockDeploy.KS

open Flock FlockSoundness.Discharge.DecodeZL Reader
open FlockSoundness.Refine (ebind_ok)

local notation "𝔽" => FlockLevel3.GF128

/-! ## The level-0 rows a decoded proof carries -/

/-- **A decoded plain proof's level-0 rows have `2^lanes` symbols** (`opened_rows`' `vecExact`). -/
theorem decodeProof_rows {sh : Shape} {bytes : ByteArray} {p : Proof} (h : decodeProof sh bytes = .ok p) :
    ∀ row ∈ p.ligerito.initialProof.rows, row.size = 2 ^ (sh.sched.levels.getD 0 default).lanes := by
  unfold decodeProof at h
  obtain ⟨⟨p', r'⟩, hrun, hp⟩ := ebind_ok h
  have hp' : p' = p := Except.ok.inj hp
  rw [← hp']
  refine (?_ : Post _ fun p : Proof => ∀ row ∈ p.ligerito.initialProof.rows,
    row.size = 2 ^ (sh.sched.levels.getD 0 default).lanes) p' ⟨_, _, hrun⟩
  dsimp only
  refine post_bind fun _ _ => ?_
  refine post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => ?_
  refine post_bind fun _ _ => ?_
  split
  rotate_left 2
  · exact post_bind_throw
  all_goals
    refine post_bind fun _ _ => post_bind fun _ _ => ?_
    refine post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => ?_
    refine post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => ?_
    refine post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => ?_
    refine post_bind fun _ _ => post_bind fun _ _ => ?_
    refine post_bind fun _ _ => post_bind fun rows hrows => post_bind fun _ _ => post_bind fun _ _ => ?_
    refine post_bind fun _ _ => post_ite (fun _ => post_bind_throw) (fun _ => ?_)
    refine post_bind fun _ _ => ?_
    refine post_bind fun _ _ => post_ite (fun _ => post_bind_throw) (fun _ => ?_)
    refine post_bind fun _ _ => post_bind fun _ _ => ?_
    refine post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => ?_
    refine post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => ?_
    refine post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => ?_
    exact post_pure fun row hrow => (yields_vecExact ((yields_vecExact hrows).2 row hrow)).1

/-- **A decoded masked proof's level-0 rows have `2^lanes + EXTRA_LANES` symbols.** -/
theorem zkDecode_rows {sh : Shape} {bytes : ByteArray} {p : Zk.ZkProof} (h : Zk.decode sh bytes = .ok p) :
    ∀ row ∈ p.proof.ligerito.initialProof.rows,
      row.size = 2 ^ (sh.sched.levels.getD 0 default).lanes + Zk.EXTRA_LANES := by
  unfold Zk.decode at h
  obtain ⟨⟨p', r'⟩, hrun, hp⟩ := ebind_ok h
  have hp' : p' = p := Except.ok.inj hp
  rw [← hp']
  refine (?_ : Post _ fun p : Zk.ZkProof => ∀ row ∈ p.proof.ligerito.initialProof.rows,
    row.size = 2 ^ (sh.sched.levels.getD 0 default).lanes + Zk.EXTRA_LANES) p' ⟨_, _, hrun⟩
  unfold Zk.reader
  dsimp only
  refine post_bind fun _ _ => ?_
  refine post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => ?_
  refine post_bind fun _ _ => ?_
  split
  rotate_left 2
  · exact post_bind_throw
  all_goals
    refine post_bind fun _ _ => post_bind fun _ _ => ?_
    refine post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => ?_
    refine post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => ?_
    refine post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => ?_
    refine post_bind fun _ _ => post_bind fun _ _ => ?_
    refine post_bind fun _ _ => post_bind fun rows hrows => post_bind fun _ _ => post_bind fun _ _ => ?_
    refine post_bind fun _ _ => post_ite (fun _ => post_bind_throw) (fun _ => ?_)
    refine post_bind fun _ _ => ?_
    refine post_bind fun _ _ => post_ite (fun _ => post_bind_throw) (fun _ => ?_)
    refine post_bind fun _ _ => post_bind fun _ _ => ?_
    refine post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => ?_
    refine post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => ?_
    refine post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => ?_
    refine post_bind fun _ _ => ?_
    split
    · refine post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => ?_
      refine post_bind fun _ _ => post_bind fun _ _ => ?_
      exact post_pure fun row hrow => (yields_vecExact ((yields_vecExact hrows).2 row hrow)).1
    · exact post_bind_throw
    · exact post_bind_throw

/-! ## The columns a run opens -/

theorem plainRepColsOf_width {sh : Shape} {rounds : Array Round} {bytes : ByteArray} {p : ℕ × List 𝔽}
    (hp : p ∈ Claims.Flock.plainRepColsOf sh rounds bytes) :
    p.2.length = 2 ^ (sh.sched.levels.getD 0 default).lanes := by
  unfold Claims.Flock.plainRepColsOf at hp
  split at hp
  · rename_i pr hpr
    unfold Claims.Flock.zkRepCols at hp
    obtain ⟨⟨pos, row⟩, hmem, rfl⟩ := List.mem_map.1 hp
    have hrow : row ∈ pr.ligerito.initialProof.rows :=
      Array.mem_toList_iff.1 (List.mem_of_mem_take (List.of_mem_zip hmem).2)
    simp [decodeProof_rows hpr row hrow]
  · simp at hp

theorem zkRepColsOf_width {sh : Shape} {rounds : Array Round} {bytes : ByteArray} {p : ℕ × List 𝔽}
    (hp : p ∈ Claims.Flock.zkRepColsOf sh rounds bytes) :
    p.2.length = 2 ^ (sh.sched.levels.getD 0 default).lanes := by
  unfold Claims.Flock.zkRepColsOf at hp
  split at hp
  · rename_i zp hzp
    obtain ⟨p', hp', rfl⟩ := List.mem_map.1 hp
    unfold Claims.Flock.zkRepCols at hp'
    obtain ⟨⟨pos, row⟩, hmem, rfl⟩ := List.mem_map.1 hp'
    have hrow : row ∈ zp.proof.ligerito.initialProof.rows :=
      Array.mem_toList_iff.1 (List.mem_of_mem_take (List.of_mem_zip hmem).2)
    simp [zkDecode_rows hzp row hrow]
  · simp at hp

theorem mem_of_distinctBelow {d : ℕ} {os : List (ℕ × List 𝔽)} {p : ℕ × List 𝔽}
    (h : p ∈ Claims.Flock.distinctBelow d os) : p ∈ os :=
  (List.mem_filter.1 ((List.pwFilter_sublist _).subset h)).1

/-- **A plain run's columns have level 0's `2^lanes` symbols.** -/
theorem plainRunCols_width {r : Claims.Flock.ZkRun} {p : ℕ × List 𝔽} (hp : p ∈ Claims.Flock.plainRunCols r) :
    p.2.length = 2 ^ (Claims.Flock.plainLevel0 r.m).lanes := by
  obtain ⟨y, -, hy⟩ := List.mem_flatMap.1 (mem_of_distinctBelow hp)
  exact plainRepColsOf_width hy

/-- **A masked run's columns have level 0's `2^lanes` witness symbols.** -/
theorem zkRunCols_width {r : Claims.Flock.ZkRun} {p : ℕ × List 𝔽} (hp : p ∈ Claims.Flock.zkRunCols r) :
    p.2.length = 2 ^ (Claims.Flock.zkLevel0 r.m).lanes := by
  obtain ⟨y, -, hy⟩ := List.mem_flatMap.1 (mem_of_distinctBelow hp)
  exact zkRepColsOf_width hy

theorem exists_of_lookCol {os : List (ℕ × List 𝔽)} {q : ℕ} {u : List 𝔽} (h : Claims.Flock.lookCol os q = some u) :
    ∃ p ∈ os, p.2 = u := by
  unfold Claims.Flock.lookCol at h
  obtain ⟨p, hp, rfl⟩ := Option.map_eq_some_iff.1 h
  exact ⟨p, List.mem_of_find?_eq_some hp, rfl⟩

/-- **The run's reader sits at its last message's table 0**: its `m` is that table's. -/
theorem runOf_m {zk : Bool} {s : Claims.Seen} {r : Claims.Flock.ZkRun} (h : runOf zk s = some r) :
    ∃ f s' spec sts st, lastFinal s = some f ∧ setupOf zk f.stmt = some (s', spec, sts) ∧ sts[0]? = some st ∧
      r.m = st.m := by
  unfold runOf at h
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def, Option.some.injEq] at h
  obtain ⟨f, hf, ⟨s', spec, sts⟩, hs, st, hst, _, _, _, _, rfl⟩ := h
  exact ⟨f, s', spec, sts, st, hf, hs, hst, rfl⟩

/-! ## An accepted run decides on its last message -/

/-- **A run accepts at its own messages**: where the rounds accept, the verifier's step accepts the prover's messages
the trace records. -/
theorem stepWith_acc_msgs {Ω : Type} (H : Claims.Msg → Claims.Msg) (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω)
    (vnext : List Claims.Msg → Claims.OC Claims.Out) :
    ∀ (n : ℕ) (hist : List (Claims.Msg × Claims.Msg)) (as : List Claims.Msg) (t : Claims.Trace),
      t.msgs.map Prod.fst = as → (Claims.stepWith H P x ω vnext n hist as t).out = .acc →
        ((vnext ((Claims.stepWith H P x ω vnext n hist as t).msgs.map Prod.fst)).eval H).1 = .acc := by
  intro n
  induction n with
  | zero => intro hist as t _ h; simp [Claims.stepWith] at h
  | succ n ih =>
    intro hist as t ht h
    simp only [Claims.stepWith] at h ⊢
    split at h
    · rename_i m heq
      simp only [heq] at h ⊢
      exact ih _ _ _ (by simp [ht]) h
    · rename_i heq
      simp only [heq, List.map_append, ht, List.map_cons, List.map_nil]
    · simp at h

theorem run_acc_next {Ω Coin : Type} [Inhabited Coin] (H : Claims.Msg → Claims.Msg) (P : Claims.Prover Ω)
    (V : Claims.Verifier Coin) (x : Claims.Msg) (ω : Ω) (r : Fin (V.k x) → Coin)
    (h : (Claims.run H P V x ω r).out = .acc) :
    ((V.next x ((Claims.run H P V x ω r).msgs.map Prod.fst)
      ((List.range ((Claims.run H P V x ω r).msgs.map Prod.fst).length).map (Claims.coinSeq r))).eval H).1 = .acc :=
  stepWith_acc_msgs H P x ω _ (V.k x) [] [] ⟨[], [], [], .rej⟩ rfl h

/-- **A decision is on the last message.** -/
theorem act_decide_final {zk : Bool} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin} {s : Ins}
    {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : Final} (h : act zk x as cs = .decide s spec sts link c₁ mids f) :
    as.getLast?.bind finalOf = some f := by
  unfold act at h
  split at h
  · cases h
  · repeat' split at h
    all_goals first | (cases h; done) | (cases h; simp_all)

/-- **An accepted run's last message is a final message on the statement**, at a statement that sets up. -/
theorem acc_last {zk : Bool} {H : Claims.Msg → Claims.Msg} {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin ((V zk).k x) → Coin} (h : (Claims.run H P (V zk) x ω r).out = .acc) :
    ∃ s spec sts f, setupOf zk x = some (s, spec, sts) ∧ lastFinal (Claims.run H P (V zk) x ω r).seen = some f ∧
      f.stmt = x := by
  have ha := run_acc_next H P (V zk) x ω r h
  generalize hcs : (List.range _).map (Claims.coinSeq r) = cs at ha
  replace ha : ((next zk x ((Claims.run H P (V zk) x ω r).msgs.map Prod.fst) cs).eval H).1 = .acc := ha
  cases hact : act zk x ((Claims.run H P (V zk) x ω r).msgs.map Prod.fst) cs with
  | rej => rw [next_rej H hact] at ha; cases ha
  | coin c => rw [next_coin H hact] at ha; cases ha
  | decide s spec sts link c₁ mids f =>
    obtain ⟨_, _, -, -, hc⟩ := (next_decide H hact).1 ha
    refine ⟨s, spec, sts, f, act_decide hact, ?_, (Claims.Bridge.Custody.checks_ok hc).1⟩
    have hl := act_decide_final hact
    rw [List.getLast?_map, Option.bind_map] at hl
    exact hl

/-! ## `ColsOk` at both records -/

theorem tables_head {zk : Bool} {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {st : Stmt}
    (hs : setupOf zk x = some (s, spec, sts)) (hst : sts[0]? = some st) : (tables zk x).head? = some (spec, st) := by
  unfold tables
  rw [hs]
  rcases sts with ⟨_ | ⟨a, l⟩⟩
  · simp at hst
  · simp only [List.getElem?_toArray, List.getElem?_cons_zero, Option.some.injEq] at hst
    subst hst
    rfl

/-- **An accepted run opens columns of the reader's table 0**: each has the statement's table 0's level-0 width. -/
theorem acc_cols {zk : Bool} {H : Claims.Msg → Claims.Msg} {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin ((V zk).k x) → Coin} (h : (Claims.run H P (V zk) x ω r).out = .acc) {q : ℕ} {u : List 𝔽}
    (hu : Claims.Flock.lookCol (cols zk (Claims.run H P (V zk) x ω r).seen) q = some u) :
    ∃ spec st, (tables zk x).head? = some (spec, st) ∧
      u.length = 2 ^ (if zk then Claims.Flock.zkLevel0 st.m else Claims.Flock.plainLevel0 st.m).lanes := by
  obtain ⟨s, spec, sts, f, hs, hf, hx⟩ := acc_last h
  unfold cols at hu
  split at hu
  · rename_i rr hrr
    obtain ⟨f', s', spec', sts', st, hf', hs', hst, hm⟩ := runOf_m hrr
    rw [hf] at hf'
    cases hf'
    rw [hx, hs] at hs'
    cases hs'
    refine ⟨spec, st, tables_head hs hst, ?_⟩
    obtain ⟨p, hp, rfl⟩ := exists_of_lookCol hu
    cases zk
    · simp only [Bool.false_eq_true, ↓reduceIte] at hp ⊢
      rw [plainRunCols_width hp, hm]
    · simp only [↓reduceIte] at hp ⊢
      rw [zkRunCols_width hp, hm]
  · simp [Claims.Flock.lookCol] at hu

theorem zeroCol_length {zk : Bool} {x : Claims.Msg} {spec : SessionSpec} {st : Stmt}
    (h : (tables zk x).head? = some (spec, st)) :
    (zeroCol zk x).length =
      2 ^ (if zk then Claims.Flock.zkLevel0 st.m else Claims.Flock.plainLevel0 st.m).lanes := by
  unfold zeroCol
  rw [h]
  cases zk <;> simp

/-- **`ColsOk` at the plain record**: an accepted run opens only columns of the zero column's width. -/
theorem plain_colsOk (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) :
    Claims.Flock.ColsOk H (deployPlain rel) (wordLen false) (wide false) := by
  intro x Ω P ω r q u hacc _ hu
  obtain ⟨spec, st, hh, hl⟩ := acc_cols (zk := false) hacc hu
  unfold wide
  rw [hl, zeroCol_length hh]

/-- **`ColsOk` at the masked record**: an accepted run opens only columns of the zero column's width, the witness
lanes. -/
theorem zk_colsOk (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) :
    Claims.Flock.ColsOk H (deployZk rel) (wordLen true) (wide true) := by
  intro x Ω P ω r q u hacc _ hu
  obtain ⟨spec, st, hh, hl⟩ := acc_cols (zk := true) hacc hu
  unfold wide
  rw [hl, zeroCol_length hh]

/-- **The deployed plain verifier: KS**, at SHA-512, from a fixed table's bound and conflicts tied to collisions. -/
theorem plain_KS_sha512 (rel : Readable.Parsed → Readable.Relation)
    (hconf : Claims.Flock.ConflictTied sha512 (deployPlain rel) (wordLen false))
    (htab : Claims.Flock.TableBoundθ sha512 (deployPlain rel) (wordLen false) (Claims.Flock.orZero (wide false))) :
    Claims.Flock.KS sha512 (deployPlain rel) :=
  plain_KS_of_tableBound rel (plain_colsOk sha512 rel) hconf htab

/-- **The deployed masked (`--zk`) verifier: KS**, at SHA-512, from a fixed table's bound and conflicts tied to
collisions. -/
theorem zk_KS_sha512 (rel : Readable.Parsed → Readable.Relation)
    (hconf : Claims.Flock.ConflictTied sha512 (deployZk rel) (wordLen true))
    (htab : Claims.Flock.TableBoundθ sha512 (deployZk rel) (wordLen true) (Claims.Flock.orZero (wide true))) :
    Claims.Flock.KS sha512 (deployZk rel) :=
  zk_KS_of_tableBound rel (zk_colsOk sha512 rel) hconf htab

end FlockDeploy.KS
