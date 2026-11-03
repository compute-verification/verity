import Proofs.Flock.Claims.Complete.Encode
import Proofs.Flock.Claims.Complete.Prover.Session

/-!
# The plain verdict accepts when each of its checks passes

The converse of the soundness side's peeling (`Exec.verify_ok_tables`): each step of the shipped verdict and of B's
checks, as its own hypothesis, composes into acceptance. These are the checks the completeness skeleton (`Skeleton`)
names, one goal each.

* **A rep** (`verifyRepWith_ok`, §9–§14): its stream is recorded (S11), the statement's extra claims evaluate, the proof
  file decodes at the rep's shape, its PCS parameters are fast100's (R7/F2), zerocheck, lincheck, the ring switch and
  Ligerito each replay from the transcript the previous one left, every recorded round is consumed (S15) and the cap is
  the committed `root_B` (S16).
* **The verdict** (`verifyWith_ok`, §3, §16): one statement per table, the record decodes (S1–S14), each table's public
  words, each table's fast100 schedule (`schedsOf`), one proof file per rep stream, the record's proof digests list
  exactly the rep streams and each is its file's (S17), and each table's two reps accept with one cap (S18).
* **B's checks** (`custody_ok`, `checks_ok`): the last message's statement, the record's draw, the record's live session
  (`liveOf`) and its custody (`root_B`, the public words, the link points, the rounds' count and each round's bytes and
  coins), the scope, and a drawn session.
-/

open Flock FlockDeploy Transcript

namespace FlockComplete

theorem ok_bind {ε α β : Type} (a : α) (f : α → Except ε β) : (Except.ok a >>= f) = f a := rfl

/-! ## A rep -/

/-- **A rep accepts** when each of its checks passes, each replay from the transcript the previous one left. -/
theorem verifyRepWith_ok (hs : Flock.Hashes V) (spec : SessionSpec) (st : Setup) (sch : Schedule) (sess : RunRecord)
    (i r : ℕ) (bytes : ByteArray) {stream : StreamRec} {extras : Array Claim} {p : Proof} {zc : ZcOut} {lc : LcOut}
    {T : F128} {sws : Array Switched} {t₁ t₂ t₃ t₄ : Transcript}
    (hS : sess.streams.find? (·.name == s!"{spec.tables[i]!}/rep{r}") = some stream)
    (hE : st.extra sess.points = .ok extras)
    (hP : decodeProof (shapeOf st sch extras.size) bytes = .ok p)
    (hR : (p.params != pcsParams st.merkle sch) = false)
    (hZ : (bindAndZerocheckWith hs st.m st.digest (flat p.cap) p.zerocheck).run
      (Transcript.init stream.rounds (spec.domain spec.tables[i]! r).toUTF8) = .ok (zc, t₁))
    (hL : (lincheckWith hs st.fold zc p.lincheck).run t₁ = .ok (lc, t₂))
    (hW : (ringSwitchWith hs st.m (abcClaims st.m st.kLog zc lc ++ extras) p.ringSwitches).run t₂ = .ok ((T, sws), t₃))
    (hG : (ligeritoWith hs st.merkle sch T p.cap p.ligerito (basis sws)).run t₃ = .ok ((), t₄))
    (hK : (t₄.k != stream.rounds.size) = false)
    (hC : (flat p.cap != sess.rootB[i]!) = false) :
    verifyRepWith hs spec st sch sess i r bytes = .ok (flat p.cap) := by
  unfold verifyRepWith
  simp only [flat] at hZ hC
  unfold shapeOf at hP
  dsimp only
  rw [hS]
  dsimp only
  simp only [liftM, monadLift_self, hE, hP, ok_bind, hR, Bool.false_eq_true, ite_false, StateT.run_bind, hZ, hL, hW,
    hG, hK, hC]
  rfl

/-! ## The verdict -/

theorem forIn_ok {α : Type} : ∀ (l : List α) (f : α → Unit → V (ForInStep Unit)),
    (∀ a ∈ l, f a () = .ok (.yield ())) → forIn l () f = .ok ()
  | [], _, _ => rfl
  | a :: l, f, hf => by
    rw [List.forIn_cons, hf a List.mem_cons_self, ok_bind]
    exact forIn_ok l f fun b hb => hf b (List.mem_cons_of_mem a hb)

theorem list_mapM_ok {α β : Type} (f : α → V β) (g : α → β) :
    ∀ (l : List α), (∀ a ∈ l, f a = .ok (g a)) → l.mapM f = .ok (l.map g)
  | [], _ => rfl
  | a :: l, hf => by
    rw [List.mapM_cons, hf a List.mem_cons_self, ok_bind, list_mapM_ok f g l fun b hb => hf b (List.mem_cons_of_mem a hb)]
    rfl

theorem array_mapM_ok {α β : Type} (f : α → V β) (g : α → β) (xs : Array α) (hf : ∀ a ∈ xs.toList, f a = .ok (g a)) :
    xs.mapM f = .ok (xs.map g) := by
  rw [Array.mapM_eq_mapM_toList, list_mapM_ok f g _ hf]
  show Except.ok (List.map g xs.toList).toArray = _
  congr 1
  apply Array.ext'
  simp

/-- Each table's fast100 schedule. -/
def schedsOf (tabs : Array Setup) : Array Schedule := tabs.map fun st => (fast100 st.m).getD default

/-- The record's digest (`SessionSpec.dgWith`) with SHA-512 replaced by `h`. -/
def dgOf (spec : SessionSpec) (h : ByteArray → ByteArray) (b : ByteArray) : ByteArray :=
  if spec.sha512Digests then h b else Sha256.hash b

theorem dgWith_ofSha512 (spec : SessionSpec) (h : ByteArray → ByteArray) (b : ByteArray) :
    spec.dgWith (Flock.Hashes.ofSha512 h : Flock.Hashes V) b = .ok (dgOf spec h b) := by
  unfold SessionSpec.dgWith dgOf
  cases spec.sha512Digests <;> rfl

/-- **The plain verdict accepts** when each of its checks passes. -/
theorem verifyWith_ok (h : ByteArray → ByteArray) (spec : SessionSpec) (tabs : Array Setup) (record : String)
    (proofs : Array ByteArray) {sess : RunRecord}
    (hT : (tabs.size != spec.tables.size || tabs.size == 0) = false)
    (hD : Record.decodeWith (Flock.Hashes.ofSha512 h : Flock.Hashes V) spec record = Except.ok sess)
    (hU : ∀ q ∈ (tabs.zip sess.publics).toList, q.1.publics q.2 = .ok ())
    (hF : ∀ st ∈ tabs.toList, (fast100 st.m).isSome)
    (hN : (proofs.size != spec.repStreams.size) = false)
    (hL : (sess.proofSha.size != spec.repStreams.size ||
      !spec.repStreams.all fun a => sess.proofSha.any fun x => x.1 == a.2.2) = false)
    (hG : ∀ q ∈ (spec.repStreams.zip proofs).toList,
      ((sess.proofSha.find? (·.1 == q.1.2.2)).map (·.2) != some (hexOf (dgOf spec h q.2))) = false)
    (hR : ∀ i < tabs.size, ∃ c,
      verifyRepWith (Flock.Hashes.ofSha512 h : Flock.Hashes V) spec tabs[i]! (schedsOf tabs)[i]! sess i 0 proofs[2 * i]! =
        Except.ok c ∧
      verifyRepWith (Flock.Hashes.ofSha512 h : Flock.Hashes V) spec tabs[i]! (schedsOf tabs)[i]! sess i 1 proofs[2 * i + 1]! =
        Except.ok c) :
    verifyWith h spec tabs record proofs = .ok () := by
  unfold verifyWith verifyWithM
  have hpub : forIn (tabs.zip sess.publics) PUnit.unit (fun x _ => do
      liftM (x.1.publics x.2)
      pure (ForInStep.yield PUnit.unit) : _ → _ → V _) = Except.ok () := by
    rw [← Array.forIn_toList]
    refine forIn_ok _ _ fun a ha => ?_
    simp only [liftM, monadLift_self, hU a ha]
    rfl
  have h17 : forIn (spec.repStreams.zip proofs) PUnit.unit (fun x _ => do
      let d ← SessionSpec.dgWith (Flock.Hashes.ofSha512 h) spec x.2
      if (Option.map (fun x => x.2) (Array.find? (fun x_1 => x_1.1 == x.1.2.2) sess.proofSha) != some (hexOf d)) = true
      then do
        throw (toString "S17: " ++ toString x.1.2.2 ++ toString ": the proof is not the one the server received")
        pure (ForInStep.yield PUnit.unit)
      else pure (ForInStep.yield PUnit.unit) : _ → _ → V _) = Except.ok () := by
    rw [← Array.forIn_toList]
    refine forIn_ok _ _ fun a ha => ?_
    simp only [dgWith_ofSha512, ok_bind, hG a ha, Bool.false_eq_true, ite_false]
    rfl
  have hreps : forIn [:tabs.size] PUnit.unit (fun i _ => do
      let c0 ← verifyRepWith (Flock.Hashes.ofSha512 h) spec tabs[i]! (schedsOf tabs)[i]! sess i 0 proofs[2 * i]!
      let c1 ← verifyRepWith (Flock.Hashes.ofSha512 h) spec tabs[i]! (schedsOf tabs)[i]! sess i 1 proofs[2 * i + 1]!
      if (c0 != c1) = true then do
        throw (toString "S18/R1: " ++ toString spec.tables[i]! ++ toString ": the reps' caps differ")
        pure (ForInStep.yield PUnit.unit)
      else pure (ForInStep.yield PUnit.unit) : _ → _ → V _) = Except.ok () := by
    rw [Std.Legacy.Range.forIn_eq_forIn_range']
    refine forIn_ok _ _ fun i hi => ?_
    have hi' : i < tabs.size := by
      simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_one_sub_one, Nat.div_one, List.mem_range'] at hi
      omega
    obtain ⟨c, h0, h1⟩ := hR i hi'
    have hcc : (c != c) = false := by
      show (!(c.data == c.data)) = false
      simp
    simp only [h0, h1, ok_bind, hcc, Bool.false_eq_true, ite_false]
    rfl
  simp only [hT, Bool.false_eq_true, ite_false, hD, ok_bind, hpub, hN, hL, h17]
  rw [array_mapM_ok _ (fun st => (fast100 st.m).getD default) tabs ?_]
  · rw [show Array.map (fun st => (fast100 st.m).getD default) tabs = schedsOf tabs from rfl, ok_bind, hreps]
    rfl
  · intro st hst
    have := hF st hst
    split
    · rename_i sch hf
      rw [hf]
      rfl
    · rename_i hf
      rw [hf] at this
      cases this

/-! ## B's checks -/

theorem check_true {why : String} : Record.check true why = .ok () := rfl

/-- **The custody checks pass** when each of their five comparisons holds. -/
theorem custody_ok {spec : SessionSpec} {lv : Live} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)}
    (hB : (lv.rootB.toList == link.1) = true) (hP : (lv.publics.toList == link.2) = true)
    (hX : (lv.points == pointsOf spec c₁) = true)
    (hN : ((lv.streams.toList.flatMap (·.toList)).length == mids.length) = true)
    (hR : ((lv.streams.toList.flatMap (·.toList)).zip mids).all
      (fun (r, a, c) => r.bytes == some ⟨a.toArray⟩ && r.coins == c.vals r.n) = true) :
    custody spec lv link c₁ mids = .ok () := by
  unfold custody
  simp only [hB, hP, hX, hN, hR, check_true, ok_bind]

/-- **B's checks pass** when each does: the statement, the draw, the live session's custody, the scope, a drawn
session, and the record's canonical text (S0). -/
theorem checks_ok {zk : Bool} {s : Ins} {spec : SessionSpec} {x : Claims.Msg} {link : List ByteArray × List ByteArray}
    {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final} {lv : Live}
    (hS : (f.stmt == x) = true)
    (hD : drawEq (FlockSoundness.Discharge.Exec.recordDraw f.record) s.read.draw = true)
    (hL : liveOf spec f.record = .ok lv) (hC : custody spec lv link c₁ mids = .ok ())
    (hQ : s.read.check = .ok ()) (hN : s.read.draw.isNone = false)
    (hT : Flock.RecordText.check f.record = .ok ()) :
    checks zk s spec x link c₁ mids f = .ok () := by
  unfold checks
  simp only [hS, hD, check_true, ok_bind, hL, hC, hQ, hN, Bool.false_eq_true, ite_false]
  exact hT

end FlockComplete
