import Proofs.Flock.Claims.Complete.Run
import Flock.With.SimVerify
import Flock.With.Log

/-!
# The honest run as a pure session

`Flock.With.Sim` relates two runs of one definition in two verdict monads; `sim_walk` walks them in step. Here the
honest prover's monad `PV` (its program: oracle queries and sends) is related to a pure one, `PS := StV (Array Msg)`, in
which a send pushes its body onto the state and reads its coin from `resp` at its index, and SHA-512 is `hOf H`.

* **The program** (`PI`) is a lawful monad; its run against `H` and `resp` (`PI.sim`) of a bind is the two runs, the
  second from the first's sends (`PI.sim_bind`).
* **The simulation** (`simS`): from the bodies `l` so far, the program's sends are some bodies `bs`, each of tag `0`, and
  its result is the pure run's, which ends with `l ++ bs`. The sends (`asks`, `asksS`) and the hashes (`Hashes.pi`,
  `Hashes.ofSha512 (hOf H)`) are related (`simS_link`, `simS_round`, `simS_hashes`).
* **Every prover function** is related to its pure run, from the squeeze to the session (`squeezeP_sim`, …,
  `sessionP_sim`).
* **The honest run** (`honestRun_eq`): where the statement sets up and the pure session (`sessionS`) returns the record
  and the proof files, the honest run's sends are the session's bodies with tag `0`, and its last message is
  `Final.encode ⟨x, record, proofs⟩`.
-/

namespace FlockComplete

open Flock FlockDeploy

namespace PI

theorem bind_assoc' {α β γ : Type} (x : PI α) (f : α → PI β) (g : β → PI γ) :
    PI.bind (PI.bind x f) g = PI.bind x fun a => PI.bind (f a) g := by
  induction x with
  | ret a => rfl
  | query q k ih => simp only [PI.bind, ih]
  | ask m k ih => simp only [PI.bind, ih]

theorem bind_ret' {α : Type} (x : PI α) : PI.bind x PI.ret = x := by
  induction x with
  | ret a => rfl
  | query q k ih => simp only [PI.bind, ih]
  | ask m k ih => simp only [PI.bind, ih]

instance instLawfulMonad : LawfulMonad PI := LawfulMonad.mk'
  (id_map := fun x => bind_ret' x)
  (pure_bind := fun _ _ => rfl)
  (bind_assoc := bind_assoc')

theorem sim_bind {α β : Type} (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) (x : PI α) (f : α → PI β) :
    ∀ i, (PI.bind x f).sim H resp i =
      ((x.sim H resp i).1 ++ ((f (x.sim H resp i).2).sim H resp (i + (x.sim H resp i).1.length)).1,
        ((f (x.sim H resp i).2).sim H resp (i + (x.sim H resp i).1.length)).2) := by
  induction x with
  | ret a => intro i; simp [PI.bind, PI.sim]
  | query q k ih => intro i; simp only [PI.bind, PI.sim]; exact ih _ i
  | ask m k ih =>
    intro i
    simp only [PI.bind, PI.sim, ih (resp i) (i + 1), List.cons_append, List.length_cons]
    rw [show i + ((PI.sim H resp (k (resp i)) (i + 1)).1.length + 1) =
      i + 1 + (PI.sim H resp (k (resp i)) (i + 1)).1.length by omega]

end PI

/-- The pure session's monad: the verdict, over the sends' bodies so far, which a rejection keeps. -/
abbrev PS := StV (Array Claims.Msg)

theorem go_bind {α β : Type} (y : PS α) (g : α → PS β) (l : Array Claims.Msg) :
    (y >>= g).go l = match (y.go l) with
      | (.ok a, l') => (g a).go l'
      | (.error e, l') => (.error e, l') := by
  simp only [StV.go, Id.run, ExceptT.run_bind, StateT.run_bind]
  rcases y.run.run l with ⟨r, l'⟩
  cases r <;> rfl

/-- **The simulation**: from the bodies `l`, the program sends some bodies `bs`, each with tag `0`, and returns the pure
run's verdict, which ends with `l ++ bs`. -/
def simS (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) : With.Sim PV PS where
  R x y := ∀ l : Array Claims.Msg, ∃ bs : List Claims.Msg,
    x.run.sim H resp l.size = (bs.map (List.cons 0), (y.go l).1) ∧ (y.go l).2 = l ++ bs.toArray
  pure a l := ⟨[], rfl, by simp [StV.go]⟩
  bind {α β x y f g} hxy hfg l := by
    obtain ⟨bs, hx, hl⟩ := hxy l
    rw [go_bind, ExceptT.run_bind]
    rcases hy : y.go l with ⟨r, l'⟩
    rw [hy] at hl hx
    dsimp only at hl hx
    subst hl
    change ∃ bs', (PI.bind x.run _).sim H resp l.size = _ ∧ _
    rw [PI.sim_bind, hx]
    cases r with
    | error e =>
      refine ⟨bs, ?_, ?_⟩
      · show (_ ++ [], _) = _
        rw [List.append_nil]
        rfl
      · simp
    | ok a =>
      obtain ⟨bs', hx', hl'⟩ := hfg a (l ++ bs.toArray)
      refine ⟨bs ++ bs', ?_, ?_⟩
      · dsimp only
        rw [show l.size + (bs.map (List.cons 0)).length = (l ++ bs.toArray).size by simp, hx']
        simp
      · rw [hl']; simp
  throw e l := ⟨[], rfl, by simp [StV.go]⟩
  lift v l := by
    refine ⟨[], ?_, ?_⟩
    · cases v <;> rfl
    · cases v <;> simp [StV.go] <;> rfl

/-- **The pure session's sends**: each send's body pushed, answered by the coin read back from `resp n`, `n` its index. -/
def asksS (resp : ℕ → Claims.Msg) : Asks PS where
  link rootB publics := do
    let l ← get
    set (l.push (linkBody rootB.toList publics.toList))
    pure (coinOf (resp l.size))
  round c := do
    let l ← get
    set (l.push (Claims.toMsg c))
    pure (coinOf (resp l.size))

theorem simS_link (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) (a b : Array ByteArray) :
    (simS H resp).R (asks.link a b) ((asksS resp).link a b) := fun l =>
  ⟨[linkBody a.toList b.toList], rfl, by simp [StV.go, asksS]; rfl⟩

theorem simS_round (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) (c : ByteArray) :
    (simS H resp).R (asks.round c) ((asksS resp).round c) := fun l =>
  ⟨[Claims.toMsg c], rfl, by simp [StV.go, asksS]; rfl⟩

theorem simS_hashes (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) :
    (simS H resp).Hashes Hashes.pi (Hashes.ofSha512 (Claims.hOf H)) where
  sha512 _ _ := ⟨[], rfl, by simp [StV.go, Hashes.ofSha512]⟩
  sha256 _ _ := ⟨[], rfl, by simp [StV.go, Hashes.ofSha512]⟩
  blake3 _ _ := ⟨[], rfl, by simp [StV.go, Hashes.ofSha512]⟩
  blake3Keyed _ _ _ := ⟨[], rfl, by simp [StV.go, Hashes.ofSha512]⟩

open With

theorem squeezeP_sim (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) (kind : UInt8) (n : ℕ) :
    ((simS H resp).st Transcript).R (squeezeP asks kind n) (squeezeP (asksS resp) kind n) := by
  intro t
  unfold squeezeP
  sim_walk [simS_round H resp]

theorem sampleP_sim (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) :
    ((simS H resp).st Transcript).R (sampleP asks) (sampleP (asksS resp)) := by
  unfold sampleP
  sim_walk [squeezeP_sim H resp]

theorem sampleKP_sim (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) :
    ((simS H resp).st Transcript).R (sampleKP asks) (sampleKP (asksS resp)) := by
  unfold sampleKP
  sim_walk [squeezeP_sim H resp]

theorem zerocheckP_sim (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) (m : ℕ) (digest cap : ByteArray)
    (tr : Trace) :
    ((simS H resp).st Transcript).R (zerocheckP asks m digest cap tr) (zerocheckP (asksS resp) m digest cap tr) := by
  unfold zerocheckP sampleVecP
  sim_walk [squeezeP_sim H resp, sampleP_sim H resp]

theorem lincheckP_sim (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) (c : CircuitFold) (m : ℕ) (tr : Trace)
    (zc : ZcOut) :
    ((simS H resp).st Transcript).R (lincheckP asks c m tr zc) (lincheckP (asksS resp) c m tr zc) := by
  unfold lincheckP
  sim_walk [sampleP_sim H resp]

theorem ringSwitchP_sim (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) (tr : Trace) (claims : Array Claim) :
    ((simS H resp).st Transcript).R (ringSwitchP asks tr claims) (ringSwitchP (asksS resp) tr claims) := by
  unfold ringSwitchP sampleVecP
  sim_walk [squeezeP_sim H resp]

theorem commitP_sim (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) (ms : MerkleScheme) (lv : Level)
    (g : Array F128) (salt : ℕ → ByteArray) :
    (simS H resp).R (commitP Hashes.pi ms lv g salt) (commitP (Hashes.ofSha512 (Claims.hOf H)) ms lv g salt) := by
  have ho := opsWith_sim (simS H resp) (simS_hashes H resp) ms
  unfold commitP
  sim_walk [ho.leaf, ho.node]

theorem liftT_sim (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) {α : Type} {x : TMW PV α} {y : TMW PS α}
    (h : ((simS H resp).st Transcript).R x y) :
    (((simS H resp).st Transcript).st LigState).R (liftT x) (liftT y) :=
  Sim.lift_st _ h

theorem sendMsg_sim (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) (m : F256 × F256) :
    (((simS H resp).st Transcript).st LigState).R (sendMsg m : LP PV Unit) (sendMsg m) := by
  unfold sendMsg
  sim_walk [liftT_sim H resp]

theorem powP_sim (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) (nonce : ℕ → ℕ → UInt64) (kind : ℕ) :
    (((simS H resp).st Transcript).st LigState).R (powP nonce kind : LP PV Unit) (powP nonce kind) := by
  unfold powP
  sim_walk [liftT_sim H resp]

theorem challengeP_sim (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) (split : Bool) :
    (((simS H resp).st Transcript).st LigState).R (challengeP asks split) (challengeP (asksS resp) split) := by
  unfold challengeP
  sim_walk [liftT_sim H resp, sampleKP_sim H resp, sendMsg_sim H resp]

theorem oodIntroP_sim (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) (nonce : ℕ → ℕ → UInt64)
    (g : Array F128) (dim : ℕ) :
    (((simS H resp).st Transcript).st LigState).R (oodIntroP asks nonce g dim) (oodIntroP (asksS resp) nonce g dim) := by
  unfold oodIntroP sampleVecP
  sim_walk [liftT_sim H resp, squeezeP_sim H resp, sampleP_sim H resp, sendMsg_sim H resp, powP_sim H resp]

theorem queryP_sim (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) (nonce : ℕ → ℕ → UInt64) (ms : MerkleScheme)
    (lv : Level) (cm : Commit) :
    (((simS H resp).st Transcript).st LigState).R (queryP asks nonce ms lv cm) (queryP (asksS resp) nonce ms lv cm) := by
  unfold queryP sampleVecP
  sim_walk [liftT_sim H resp, squeezeP_sim H resp, powP_sim H resp]

theorem ligeritoP_sim (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) (ms : MerkleScheme) (sch : Schedule)
    (T : F128) (cm0 : Commit) (t b : Array F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) :
    ((simS H resp).st Transcript).R (ligeritoP asks Hashes.pi ms sch T cm0 t b salt nonce)
      (ligeritoP (asksS resp) (Hashes.ofSha512 (Claims.hOf H)) ms sch T cm0 t b salt nonce) := by
  unfold ligeritoP sampleVecP
  sim_walk [liftT_sim H resp, squeezeP_sim H resp, sampleP_sim H resp, sampleKP_sim H resp, sendMsg_sim H resp,
    powP_sim H resp, challengeP_sim H resp, oodIntroP_sim H resp, queryP_sim H resp, commitP_sim H resp]

theorem repP_sim (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) (st : Setup) (sch : Schedule) (tr : Trace)
    (t : Array F128) (cm0 : Commit) (extras : Array Claim) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) :
    ((simS H resp).st Transcript).R (repP asks Hashes.pi st sch tr t cm0 extras salt nonce)
      (repP (asksS resp) (Hashes.ofSha512 (Claims.hOf H)) st sch tr t cm0 extras salt nonce) := by
  unfold repP
  sim_walk [zerocheckP_sim H resp, lincheckP_sim H resp, ringSwitchP_sim H resp, ligeritoP_sim H resp]

theorem roundJ_sim (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) (spec : SessionSpec) (k : ℕ) (r : Round) :
    (simS H resp).R (roundJ Hashes.pi spec k r) (roundJ (Hashes.ofSha512 (Claims.hOf H)) spec k r) := by
  have hh := simS_hashes H resp
  unfold roundJ SessionSpec.dgWith
  sim_walk

theorem recordJson_sim (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) (spec : SessionSpec)
    (rootB : Array ByteArray) (points : Array (Array F128)) (streams : Array (Array Round)) (proofs : Array ByteArray) :
    (simS H resp).R (recordJson Hashes.pi spec rootB points streams proofs)
      (recordJson (Hashes.ofSha512 (Claims.hOf H)) spec rootB points streams proofs) := by
  have hh := simS_hashes H resp
  unfold recordJson SessionSpec.dgWith
  sim_walk [roundJ_sim H resp, Sim.mapM_list]

theorem sessionP_sim (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) (spec : SessionSpec) (tabs : Array Setup)
    (trs : Array Trace) (tape : Tape) :
    (simS H resp).R (sessionP asks Hashes.pi spec tabs trs tape)
      (sessionP (asksS resp) (Hashes.ofSha512 (Claims.hOf H)) spec tabs trs tape) := by
  unfold sessionP
  sim_walk [simS_link H resp, commitP_sim H resp, repP_sim H resp, recordJson_sim H resp]

/-- **The pure honest session** at the replies `resp`: its verdict (the record and the proof files) and its sends'
bodies. -/
def sessionS (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) (spec : SessionSpec) (sts : Array Flock.Stmt)
    (w : Claims.Flock.Wit) (ω : Tape) : Except String (String × Array ByteArray) × Array Claims.Msg :=
  (sessionP (asksS resp) (Hashes.ofSha512 (Claims.hOf H)) spec (tabsOf sts) (tracesOf sts w) ω).go #[]

/-- **The honest run is the pure session's**: its sends are the session's bodies, each of tag `0`, and its last message
the session's record and proof files after the statement. -/
theorem honestRun_eq (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) (w : Claims.Flock.Wit) (ω : Tape) {k : ℕ}
    (r : Fin k → Coin) {s : Ins} {spec : SessionSpec} {sts : Array Flock.Stmt} (hs : setupOf false x = some (s, spec, sts))
    {rec : String} {pfs : Array ByteArray} {bs : Array Claims.Msg}
    (hS : sessionS H (fun j => (Claims.coinSeq r j).bytes) spec sts w ω = (.ok (rec, pfs), bs)) :
    honestRun H x w ω r = (bs.toList.map (List.cons 0), Final.encode ⟨x, rec, pfs⟩) := by
  obtain ⟨bs', h1, h2⟩ := sessionP_sim H (fun j => (Claims.coinSeq r j).bytes) spec (tabsOf sts) (tracesOf sts w) ω #[]
  unfold sessionS at hS
  rw [hS] at h1 h2
  dsimp only at h1 h2
  unfold honestRun prog
  rw [hs]
  dsimp only
  change (PI.bind (sessionP asks Hashes.pi spec (tabsOf sts) (tracesOf sts w) ω).run _).sim _ _ 0 = _
  rw [PI.sim_bind]
  have h1' := h1
  simp only [Array.size_empty] at h1'
  rw [h1']
  rw [h2, Array.empty_append, List.toList_toArray]
  show (_ ++ [], _) = _
  rw [List.append_nil]
  rfl

end FlockComplete
