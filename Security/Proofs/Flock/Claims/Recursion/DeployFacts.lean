import Proofs.Flock.Claims.Recursion.Deploy
import Proofs.Flock.Claims.Recursion.DeployScope
import Proofs.Flock.Claims.DeployK
import Proofs.Flock.Soundness.Refine.Paths
import Proofs.Flock.Claims.KS.PlainAcc
import Proofs.Flock.Claims.Recursion.FlockKSRecord
import Proofs.Flock.Claims.Recursion.Gap
import Proofs.Flock.Claims.Recursion.KSLin

/-!
# The compiled verifier's structural code facts

The code facts the recursion's analysis reads of a deployment that follow from `deployRec`'s definition alone, each
the analysis' own Prop (`Refuses`, `InnerRounds`, `SessionNums`, `SessionRounds`, `TableErrNonneg`,
`SessionAccepts`, `InnerViewRun`, `CallsNil`), at `deployRec I.V ivScope b` and the outer record `deployZkRec rel`:

* `refuses`: a run accepts only at a step where the compiled verifier's does (`stepWith_acc`), which is the last
  session's decision, so the statement parses (`DeployScope.acc_parse`) and `parse` reads no circuit commitment;
  `run_acc_inScope`: and what it reads is in `InScope`.
* `innerRounds`, `sessionRounds`: `k(x)` is the inner rounds, the staging round and every session's rounds at `x`'s
  numbers, and every V* statement `vstar` gives has those numbers (`vstar_map_k`).
* `sessionNums`, `tableErrNonneg`: `vstar`'s statements have `nums`' numbers (`numsOf_vstarAt`), which are
  `deployZkRec`'s.
* `sessionAccepts`: `session` reads an accepted run's sessions as accepted.
* `innerViewRun`: the inner view is `I.V`'s run of the readers' messages at the claim's oracle, and a run of replayed
  messages is the run of the messages it shows (`stepWith_replay`), so with no hypothesis on the witnesses.
* `callsNil`: the record's calls are each session's witness's, so no witnesses record none.

`nextO_acc_length` and `next_acc_length`: the outer verifier (at an oracle where its setup has its SHA-512 rounds,
`nextO_acc_length_sha512`) and the compiled verifier accept only at their last rounds, as `FlockDeploy.next_acc_length`
has it for Flock's. `coinOf_bytes`: a coin's bytes determine it, so `coinOf`
reads an inner round's answer back as its coin. `zeroCommit_sha512`: the record's commit string to the zero digest
(`Deploy.zeroCommit` at its salt prefix and mask) is the deployed one at SHA-512.
-/

namespace RecDeploy

open FlockDeploy (Coin)

/-! ## Runs -/

/-- **A run accepts only where its verifier's step does.** -/
theorem stepWith_acc {Ω : Type} (H : Claims.Msg → Claims.Msg) (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω)
    (vnext : List Claims.Msg → Claims.OC Claims.Out) :
    ∀ (n : ℕ) (hist : List (Claims.Msg × Claims.Msg)) (as : List Claims.Msg) (t : Claims.Trace),
      (Claims.stepWith H P x ω vnext n hist as t).out = .acc → ∃ as', ((vnext as').eval H).1 = .acc
  | 0, _, _, _, h => by simp [Claims.stepWith] at h
  | n + 1, hist, as, t, h => by
    simp only [Claims.stepWith] at h
    split at h
    · exact stepWith_acc H P x ω vnext n _ _ _ h
    · exact ⟨_, by assumption⟩
    · simp at h

/-- **A replayed run plays the replayed messages in order, and only those it plays decide it**: from history `hist`,
the run's `m` new messages are the replayed ones from `hist.length` on, and any replay that agrees on those has the same
run. -/
theorem stepWith_replay (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) (vnext : List Claims.Msg → Claims.OC Claims.Out)
    (ms : List Claims.Msg) :
    ∀ (n : ℕ) (hist : List (Claims.Msg × Claims.Msg)) (as : List Claims.Msg) (t : Claims.Trace),
      ∃ m, (Claims.stepWith H (Claims.replay ms) x () vnext n hist as t).msgs.map Prod.fst =
          t.msgs.map Prod.fst ++ (List.range m).map (fun i => ms.getD (hist.length + i) []) ∧
        ∀ ms' : List Claims.Msg, (∀ i < m, ms'.getD (hist.length + i) [] = ms.getD (hist.length + i) []) →
          Claims.stepWith H (Claims.replay ms') x () vnext n hist as t =
            Claims.stepWith H (Claims.replay ms) x () vnext n hist as t
  | 0, hist, as, t => ⟨0, by simp [Claims.stepWith], fun _ _ => rfl⟩
  | n + 1, hist, as, t => by
    rcases hvb : (vnext (as ++ [ms.getD hist.length []])).eval H with ⟨o, log⟩
    cases o with
    | msg m =>
      obtain ⟨k, hk, hag⟩ := stepWith_replay H x vnext ms n (hist ++ [(ms.getD hist.length [], m)])
        (as ++ [ms.getD hist.length []])
        ⟨t.msgs ++ [(ms.getD hist.length [], .msg m)], t.pLog ++ [], t.vLog ++ log, t.out⟩
      refine ⟨k + 1, ?_, fun ms' hms => ?_⟩
      · simp only [Claims.stepWith, Claims.replay, Claims.OC.eval, hvb]
        refine hk.trans ?_
        simp only [List.map_append, List.map_cons, List.map_nil, List.length_append, List.length_cons,
          List.length_nil, List.append_assoc, List.range_succ_eq_map, List.map_cons, List.map_map]
        simp only [List.map_inj_left, Function.comp, List.cons_append, List.nil_append, List.append_cancel_left_eq,
          List.cons.injEq, Nat.add_zero, true_and]
        intro a _
        rw [Nat.add_assoc, Nat.add_comm 1 a]
      · have h0 : ms'.getD hist.length [] = ms.getD hist.length [] := by simpa using hms 0 (by omega)
        simp only [Claims.stepWith, Claims.replay, Claims.OC.eval, h0, hvb]
        exact hag ms' fun i hi => by
          simpa [Nat.add_assoc, Nat.add_comm 1] using hms (i + 1) (by omega)
    | acc =>
      refine ⟨1, ?_, fun ms' hms => ?_⟩
      · simp only [Claims.stepWith, Claims.replay, Claims.OC.eval, hvb]
        simp
      · have h0 : ms'.getD hist.length [] = ms.getD hist.length [] := by simpa using hms 0 (by omega)
        simp only [Claims.stepWith, Claims.replay, Claims.OC.eval, h0, hvb]
    | rej =>
      refine ⟨1, ?_, fun ms' hms => ?_⟩
      · simp only [Claims.stepWith, Claims.replay, Claims.OC.eval, hvb]
        simp
      · have h0 : ms'.getD hist.length [] = ms.getD hist.length [] := by simpa using hms 0 (by omega)
        simp only [Claims.stepWith, Claims.replay, Claims.OC.eval, h0, hvb]

/-- **A replayed run is the run of the messages it shows.** -/
theorem run_replay_seen {Coin' : Type} [Inhabited Coin'] (H : Claims.Msg → Claims.Msg) (V' : Claims.Verifier Coin')
    (x : Claims.Msg) (ms : List Claims.Msg) (c : Fin (V'.k x) → Coin') :
    Claims.run H (Claims.replay ((Claims.run H (Claims.replay ms) V' x () c).seen.msgs.map Prod.fst)) V' x () c =
      Claims.run H (Claims.replay ms) V' x () c := by
  obtain ⟨m, hm, hag⟩ := stepWith_replay H x (fun as => V'.next x as ((List.range as.length).map (Claims.coinSeq c)))
    ms (V'.k x) [] [] ⟨[], [], [], .rej⟩
  refine hag _ fun i hi => ?_
  show (((Claims.run H (Claims.replay ms) V' x () c).msgs.map Prod.fst).getD ([].length + i) []) = _
  unfold Claims.run
  rw [hm]
  simp [List.getD_eq_getElem?_getD, hi]

/-! ## The code facts -/

variable {W : Type}

theorem ract_readX {IV : Claims.Verifier Coin} {st : Stage} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {s j : ℕ} {xO : Claims.Msg} {as' : List Claims.Msg} {cs' : List Coin}
    (h : ract IV st x as cs = .sess s j xO as' cs') : ∃ r, readX x = some r := by
  unfold ract at h
  split at h
  · cases h
  · exact ⟨_, by assumption⟩

/-- **The compiled verifier refuses an accepted run out of scope**: every accepted run is of a statement that `parse`
reads, and what it reads is in `InScope` (`acc_parse`: the last session's decision refuses an inner shape the setup
doesn't accept, and a record that doesn't draw every unit). -/
theorem run_acc_inScope (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (b : Brings)
    (H : Claims.Msg → Claims.Msg) {x : Claims.Msg} {Ω : Type} {P : Claims.Prover Ω} {ω : Ω}
    {r : Fin ((deployRec IV ivScope b).V.k x) → Coin}
    (h : (Claims.run H P (deployRec IV ivScope b).V x ω r).out = .acc) :
    ∃ s, parse x = some s ∧ Claims.Recursion.InScope s := by
  obtain ⟨as', ha⟩ := stepWith_acc H P x ω _ _ _ _ _ h
  obtain ⟨_, -, hp, hin⟩ := acc_parse (IV := IV) (st := b.st) H ha
  exact ⟨_, hp, hin⟩

/-- **`Refuses`**: every accepted run is of a statement that parses, with no circuit commitment (`RecScope.toStmt`
reads none). -/
theorem refuses (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (b : Brings) (H : Claims.Msg → Claims.Msg) :
    Claims.Recursion.Refuses H (deployRec IV ivScope b) := by
  intro x Ω P ω r h
  obtain ⟨as', ha⟩ := stepWith_acc H P x ω _ _ _ _ _ h
  obtain ⟨_, -, hp, -⟩ := acc_parse (IV := IV) (st := b.st) H ha
  exact ⟨_, hp, rfl⟩

/-- **`InnerRounds`**: the compiled verifier takes at least the inner verifier's rounds. -/
theorem innerRounds (I : Claims.Recursion.Inner Coin W) (ivScope : Claims.Msg → Bool) (b : Brings) :
    Claims.Recursion.InnerRounds (deployRec I.V ivScope b) I := fun _ => by
  show I.V.k (xIOf _) ≤ I.V.k (xIOf _) + kS _ + _
  omega

/-- **`SessionNums`**: whatever the inner and staging rounds, part `s`'s V* statement has `nums`' rounds, waits,
truncation, `N₀`, table error and weight under the outer record. -/
theorem sessionNums (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (b : Brings)
    (rel : Flock.Readable.Parsed → Flock.Readable.Relation) :
    Claims.Recursion.SessionNums (deployRec IV ivScope b) (deployZkRec rel) := by
  intro x s ms
  obtain ⟨s, hs⟩ := s
  dsimp only [deployRec, numsZk]
  rw [vstar_getD b.st x ms hs, ← numsOf_vstarAt b.st x ms s]
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩

theorem sum_range_le (f : ℕ → ℕ) {a b : ℕ} (h : a ≤ b) :
    ((List.range a).map f).sum ≤ ((List.range b).map f).sum := by
  induction b, h using Nat.le_induction with
  | base => exact le_rfl
  | succ n _ ih =>
    rw [List.range_succ, List.map_append, List.sum_append]
    omega

/-- **`SessionRounds`**: session `s`'s rounds, after the inner and staging rounds and the earlier sessions', end by
`k(x)`. -/
theorem sessionRounds (I : Claims.Recursion.Inner Coin W) (ivScope : Claims.Msg → Bool) (b : Brings)
    (rel : Flock.Readable.Parsed → Flock.Readable.Relation) :
    Claims.Recursion.SessionRounds (deployRec I.V ivScope b) I (deployZkRec rel) := by
  intro x s ms
  obtain ⟨s, hs⟩ := s
  replace hs : s < nS x := hs
  show I.V.k (xIOf x) + kS x + (((vstar b.st x ms).take s).map VO.k).sum + VO.k ((vstar b.st x ms).getD s []) ≤
    I.V.k (xIOf x) + kS x + ((List.range (nS x)).map fun s => (nums b.st x s).rounds).sum
  have hk : VO.k ((vstar b.st x ms).getD s []) = (nums b.st x s).rounds := by
    rw [vstar_getD b.st x ms hs]
    exact congrArg Claims.Recursion.Nums.rounds (numsOf_vstarAt b.st x ms s)
  rw [List.map_take, vstar_map_k, ← List.map_take, List.take_range, Nat.min_eq_left hs.le, hk]
  have := sum_range_le (fun s => (nums b.st x s).rounds) (Nat.succ_le_of_lt hs)
  rw [List.range_succ, List.map_append, List.sum_append] at this
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, Nat.add_zero] at this
  omega

/-- **`TableErrNonneg`**: every session's table error is nonnegative. -/
theorem tableErrNonneg (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (b : Brings) :
    Claims.Recursion.TableErrNonneg (deployRec IV ivScope b) :=
  fun _ _ => εtableO_nonneg _

/-- **`SessionAccepts`**: an accepted run shows each V* session accepting. -/
theorem sessionAccepts (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (b : Brings) :
    Claims.Recursion.SessionAccepts (deployRec IV ivScope b) := by
  intro x s v _ hv
  simp [deployRec, session, hv]

/-- **`InnerViewRun`, at every oracle and with no hypothesis on the witnesses**: the inner view is `I.V`'s run, at the
compiled verifier's first `k_I` coins, of the messages it shows. -/
theorem innerViewRun {Col T : Type} (H : Claims.Msg → Claims.Msg) (I : Claims.Recursion.Inner Coin W)
    (ivScope : Claims.Msg → Bool) (b : Brings) (DO : Claims.Flock.Deploy Coin Col T) :
    Claims.Recursion.InnerViewRun H (deployRec I.V ivScope b) I DO := by
  intro decO x Ω P ω r c ws _
  show innerView H I.V b.rd x r ws =
    (Claims.run H (Claims.replay ((innerView H I.V b.rd x r ws).msgs.map Prod.fst)) I.V (xIOf x) ()
      fun i => Claims.coinSeq r i.val).seen
  unfold innerView
  rw [run_replay_seen]

/-- **`CallsNil`, at every oracle**: no witnesses record no calls. -/
theorem callsNil (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (b : Brings) (H : Claims.Msg → Claims.Msg) :
    Claims.Recursion.CallsNil H (deployRec IV ivScope b) := by
  simp [Claims.Recursion.CallsNil, deployRec]

/-! ## The outer verifier decides at its last round -/

theorem actO_decide_length {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin} {s : FlockDeploy.Ins}
    {spec : Flock.SessionSpec} {sts : Array Flock.Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : FlockDeploy.Final} (hcs : cs.length = as.length)
    (h : actO x as cs = .decide s spec sts link c₁ mids f) : as.length = mids.length + 2 := by
  unfold actO actAt at h
  repeat' split at h
  all_goals first | (cases h; done) | skip
  rename_i a _ _ _ _ a₁ rest _ crest hfin hlast _ _ _ hlink
  cases h
  cases rest with
  | nil =>
    simp only [List.getLast?_singleton, Option.some.injEq] at hlast
    subst hlast
    have h1 := FlockDeploy.finalOf_head hfin
    rw [FlockDeploy.linkOf_head hlink] at h1
    cases h1
  | cons b rest =>
    simp only [List.length_cons] at hcs
    simp only [List.length_cons, List.length_zip, List.length_map, List.length_dropLast]
    omega

/-- **The outer verifier accepts only at its last round** (`FlockDeploy.next_acc_length` at the recursive setup):
wherever it accepts, against an oracle `H` at which Flock's setup of the statement has the rounds of its recursive setup
(`hr`), and with one coin per message, the prover sent `VO.k x` messages. -/
theorem nextO_acc_length (H : Claims.Msg → Claims.Msg) {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    (hcs : cs.length = as.length)
    (hr : ∀ (i : RecIns) (spec : Flock.SessionSpec) (sts : Array Flock.Stmt) (spec' : Flock.SessionSpec)
      (sts' : Array Flock.Stmt), setupO x = some (i, spec, sts) →
        FlockDeploy.setupWith (Claims.hOf H) i.base.read = .ok (spec', sts') →
          FlockDeploy.rounds true spec' sts' = FlockDeploy.rounds true spec sts)
    (h : ((nextO x as cs).eval H).1 = .acc) : as.length = VO.k x := by
  cases ha : actO x as cs with
  | rej => rw [nextO_rej H ha] at h; cases h
  | coin c => rw [nextO_coin H ha] at h; cases h
  | decide s spec sts link c₁ mids f =>
  obtain ⟨⟨spec', sts', hsu, hv, hc⟩, -⟩ := (nextO_decide H ha).1 h
  obtain ⟨lv, hlv, hcus⟩ := FlockDeploy.checks_custody hc
  obtain ⟨hsz, hk⟩ := FlockDeploy.streams_size true (Flock.Hashes.ofSha512 (Claims.hOf H)) hv hlv
  obtain ⟨i, hi, rfl⟩ := actO_decide ha
  show as.length = (match setupO x with | some (_, spec, sts) => 2 + FlockDeploy.rounds true spec sts | none => 1)
  rw [hi, actO_decide_length hcs ha, ← FlockDeploy.custody_length hcus]
  dsimp only
  rw [← hr i spec sts spec' sts' hi hsu, FlockDeploy.rounds_sum, List.length_flatMap,
    Nat.add_comm _ 2, ← FlockDeploy.sum_pairs (fun i => FlockDeploy.repRounds true spec' sts'[i]!)]
  congr 2
  apply List.ext_getElem
  · simp [hsz]
  · intro k h₁ h₂
    simp only [List.getElem_map, List.getElem_range, Array.getElem_toList, Array.length_toList]
    exact hk k _

/-- **At the SHA-512 oracle the outer verifier accepts only at its last round.** -/
theorem nextO_acc_length_sha512 {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    (hcs : cs.length = as.length) (h : ((nextO x as cs).eval FlockDeploy.sha512).1 = .acc) : as.length = VO.k x :=
  nextO_acc_length FlockDeploy.sha512 hcs (fun i spec sts spec' sts' hi hsu => by
    rw [FlockDeploy.hOf_sha512, FlockDeploy.setupWith_sha512, RecIns.setup_read (setupO_some hi).2] at hsu
    cases hsu
    rfl) h

/-! ## The compiled verifier decides at its last round -/

theorem locate_sum : ∀ {ks : List ℕ} {i s j : ℕ}, locate ks i = some (s, j) → i = (ks.take s).sum + j
  | [], _, _, _, h => by simp [locate] at h
  | k :: ks, i, s, j, h => by
    unfold locate at h
    split_ifs at h with hi
    · cases h
      simp
    · obtain ⟨⟨s', j'⟩, hl, he⟩ := Option.map_eq_some_iff.1 h
      cases he
      have := locate_sum hl
      simp only [List.take_succ_cons, List.sum_cons]
      omega

theorem ract_sess_locate {IV : Claims.Verifier Coin} {st : Stage} {x : Claims.Msg} {as : List Claims.Msg}
    {cs : List Coin} {s j : ℕ} {xO : Claims.Msg} {as' : List Claims.Msg} {cs' : List Coin}
    (h : ract IV st x as cs = .sess s j xO as' cs') :
    IV.k (xIOf x) + kS x < as.length ∧
      locate ((vstar st x (innerPairs (IV.k (xIOf x) + kS x) as cs)).map VO.k)
          (as.length - 1 - (IV.k (xIOf x) + kS x)) = some (s, j) ∧
      xO = (vstar st x (innerPairs (IV.k (xIOf x) + kS x) as cs)).getD s [] := by
  unfold ract at h
  split at h
  · cases h
  split_ifs at h with hk
  · split at h <;> cases h
  dsimp only at h
  split at h
  · cases h
  rename_i s' j' hl
  split_ifs at h with hn hd
  obtain ⟨rfl, rfl, rfl, -, -⟩ := RAct.sess.inj h
  exact ⟨by omega, hl, rfl⟩

/-- **The compiled verifier accepts only at its last round**: wherever it accepts, against any oracle, the prover sent
`k(x)` messages, the inner rounds, the staging round and every V* session's rounds. -/
theorem next_acc_length (H : Claims.Msg → Claims.Msg) {IV : Claims.Verifier Coin} {st : Stage} {x : Claims.Msg}
    {as : List Claims.Msg} {cs : List Coin} (h : ((next IV st x as cs).eval H).1 = .acc) :
    as.length = (V IV st).k x := by
  obtain ⟨s, j, xO, as', cs', hr, hs, hj, -⟩ := next_acc H h
  obtain ⟨hlt, hl, hx⟩ := ract_sess_locate hr
  have hsum := locate_sum hl
  rw [vstar_map_k] at hsum
  have hk : VO.k xO = (nums st x s).rounds := by
    rw [hx, vstar_getD st x _ (by omega)]
    exact congrArg Claims.Recursion.Nums.rounds (numsOf_vstarAt st x _ s)
  show as.length = IV.k (xIOf x) + kS x + ((List.range (nS x)).map fun s => (nums st x s).rounds).sum
  rw [← hs] at hsum ⊢
  rw [List.range_succ, List.map_append, List.take_left' (by simp)] at hsum
  rw [List.range_succ, List.map_append, List.sum_append]
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, Nat.add_zero]
  omega

/-! ## Coins off their bytes -/

theorem flatten_inj_of_length {n : ℕ} :
    ∀ {L₁ L₂ : List (List UInt8)}, L₁.length = L₂.length → (∀ l ∈ L₁, l.length = n) → (∀ l ∈ L₂, l.length = n) →
      L₁.flatten = L₂.flatten → L₁ = L₂
  | [], [], _, _, _, _ => rfl
  | [], _ :: _, hl, _, _, _ => by simp at hl
  | _ :: _, [], hl, _, _, _ => by simp at hl
  | a :: L₁, b :: L₂, hl, h₁, h₂, h => by
    simp only [List.flatten_cons] at h
    obtain ⟨rfl, h'⟩ := List.append_inj h ((h₁ a (by simp)).trans (h₂ b (by simp)).symm)
    rw [flatten_inj_of_length (by simpa using hl) (fun l hl' => h₁ l (by simp [hl']))
      (fun l hl' => h₂ l (by simp [hl'])) h']

/-- **A coin's bytes determine it.** -/
theorem coin_bytes_injective : Function.Injective FlockDeploy.Coin.bytes := by
  intro c d h
  simp only [FlockDeploy.Coin.bytes, List.flatMap] at h
  have hm := flatten_inj_of_length (n := 16) (by simp)
    (fun l hl => by obtain ⟨i, -, rfl⟩ := List.mem_map.1 hl; exact FlockSoundness.Refine.toBytes_length _)
    (fun l hl => by obtain ⟨i, -, rfl⟩ := List.mem_map.1 hl; exact FlockSoundness.Refine.toBytes_length _) h
  funext i
  have hi := (List.map_inj_left.1 hm) i (List.mem_finRange i)
  exact FlockSoundness.Refine.toBytes_inj (ByteArray.ext (Array.ext' hi))

open Classical in
/-- A coin read back off its bytes; `default` off anything else. -/
noncomputable def coinOf (m : Claims.Msg) : Coin :=
  if h : ∃ c : Coin, c.bytes = m then Classical.choose h else default

/-- **`coinOf` inverts a coin's bytes**: the compiled verifier's inner-round answers (`.msg c.bytes`) show their
coins. -/
theorem coinOf_bytes (c : Coin) : coinOf c.bytes = c := by
  unfold coinOf
  split
  · rename_i h
    exact coin_bytes_injective (Classical.choose_spec h)
  · rename_i h
    exact absurd ⟨c, rfl⟩ h

/-! ## The inner rounds' commit strings -/

theorem toMsg_append (a b : ByteArray) : Claims.toMsg (a ++ b) = Claims.toMsg a ++ Claims.toMsg b := by
  simp [Claims.toMsg, ByteArray.data_append]

theorem toMsg_ofMsg (a : Claims.Msg) : Claims.toMsg (Claims.ofMsg a) = a := by
  simp [Claims.toMsg, Claims.ofMsg]

/-- **The record's commit string to the zero digest at SHA-512 is the deployed one**: hm96-sha512's `b ‖ c` at the
pinned key (`Hm96.Default512.commitString`) under the salt. -/
theorem zeroCommit_sha512 (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (b : Brings)
    (y : Fin Flock.Hm96.sha512.saltBytes → UInt8) :
    (((deployRec IV ivScope b).zeroCommit y).eval FlockDeploy.sha512).1 =
      Claims.toMsg (Flock.Hm96.Default512.commitString (ByteArray.mk (Array.replicate 64 0)) ⟨Array.ofFn y⟩) := by
  have hs : Claims.toMsg Flock.Hm96.Default512.saltPrefix ++ List.ofFn y =
      Claims.toMsg (Flock.Hm96.Default512.saltPrefix ++ ⟨Array.ofFn y⟩) := by
    rw [toMsg_append]
    simp [Claims.toMsg]
  simp only [Claims.Recursion.Deploy.zeroCommit, deployRec, Claims.OC.eval, FlockDeploy.sha512,
    Flock.Hm96.Default512.commitString, hs, Claims.ofMsg_toMsg]
  rw [toMsg_append]
  congr 1
  apply List.ext_getElem
  · simp [Claims.toMsg]
  · intro n h1 _
    rw [List.getElem_ofFn]
    simp only [List.length_ofFn] at h1
    have h0 : ({ data := Array.replicate 64 0 } : ByteArray).get! n = 0 := by
      simp [ByteArray.get!, h1]
    simp [Claims.toMsg, h0]

end RecDeploy
