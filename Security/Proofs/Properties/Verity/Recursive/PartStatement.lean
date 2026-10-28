import Proofs.Flock.Claims.Complete.Encode
import Proofs.Flock.Claims.Recursion.Deploy
import Proofs.Properties.Verity.Recursive.Names
import VStarRecursive

/-!
# Piece C of `acc_of_accepts`: V\* part `s`'s statement is the compiled part's inputs

The model's V\* statement of part `s` (`RecDeploy.staged st x ms s`, the encoding of `RecDeploy.partRaw`) is read by the
outer verifier's readers (`RecDeploy.insO`, `RecRaw.ins`) as exactly the recursive inputs compiled part `s`'s
`VStar.Recursive.partVerdict` sets its session up on (`partIns`): the part's circuit and staged public file, its
partition, program, registered reads, bindings and rows, the verifier's inner shape, check 5's source, the inner
circuit's binding, the inner session's commits and link points, and the part's draw. Piece D2 holds `partIns`' setup to
the compiled `setupOf` and so concludes `setupO (staged st x ms s) = some _`.

* **Round trip** (`decode_encode_fits`, `decodeRec_encode_fits`): a statement whose lengths fit their 8-byte words reads
  back as itself, so `staged` is `partRaw`'s encoding (`staged_eq`) and `insO` reads it as `partRaw`'s inputs
  (`insO_staged`). Every field `partRaw` takes from `x` fits, since `x` decodes (`partRaw_fits`); the staged files and
  the transcript, which come from the run, fit by hypothesis.
* **The readers agree** where they are two pieces of code: the two `metaOf`s (`metaOf_eq`), `jsonOf` whatever it names
  (`jsonOf_what`), `Blobs.get` whatever it names (`blobs_get_what`), and the compiled `registeredOf`'s ports
  (`registeredOf_own`). `innerHeld_ok` reads `innerHeld`'s fields off the blobs.
* **The correspondence** (`insO_staged_partIns`) takes `Names`' clauses (piece B) and `Shows`' as hypotheses by
  name, at `readX`'s reading `r` of `x` (`readXPre_spec` relates it to `Names`' `w` and `ri`), and `Names`'
  `PartForm`, the form compiled part `s` sets up (statement `verity/flock-circuit`, `--zk`, one table, no work,
  stratified, closure or public input words).
* `vstarAt_eq_staged`: where part `s`'s staged numbers are `nums`' (piece C2), its V\* statement is the staged one.
-/

namespace Verity.Recursive.Part

open Flock
open RecDeploy
open FlockDeploy (Raw encNat encBytes encStr encBool encOpt encList Rd rdTake rdNat rdBytes rdStr rdBool rdOpt rdList
  jsonOf)
open Lean (Json)

/-! ## The round trip -/

theorem rdBool_encBool (b : Bool) (rest : List UInt8) : rdBool.run (encBool b ++ rest) = some (b, rest) := by
  cases b <;> rfl

theorem rdOpt_encOpt {α : Type} {r : Rd α} {e : α → Claims.Msg} {a : Option α}
    (hr : ∀ v ∈ a, ∀ rest, r.run (e v ++ rest) = some (v, rest)) (rest : List UInt8) :
    (rdOpt r).run (encOpt e a ++ rest) = some (a, rest) := by
  cases a with
  | none => rfl
  | some v =>
    have h := hr v rfl rest
    show (rdOpt r).run (encBool true ++ (e v ++ rest)) = some (some v, rest)
    simp only [rdOpt, StateT.run_bind, rdBool_encBool, Option.bind_eq_bind, Option.bind_some, ite_true,
      StateT.run_map, h]
    rfl

theorem format_fits : FlockDeploy.FORMAT.toUTF8.size < 2 ^ 64 := by decide

/-- **A statement's lengths and numbers fit their 8-byte words** (`encNat`). -/
def RawFits (w : Raw) : Prop :=
  w.statement.toUTF8.size < 2 ^ 64 ∧ w.circuit.size < 2 ^ 64 ∧ w.publicFile.size < 2 ^ 64 ∧
    w.tables.length < 2 ^ 64 ∧ (∀ p ∈ w.tables, p.1.toUTF8.size < 2 ^ 64 ∧ p.2.size < 2 ^ 64) ∧ w.count < 2 ^ 64 ∧
    (∀ b ∈ w.draw, b.size < 2 ^ 64) ∧ (∀ b ∈ w.partition, b.size < 2 ^ 64) ∧ (∀ b ∈ w.program, b.size < 2 ^ 64) ∧
    (∀ n ∈ w.work, n < 2 ^ 64) ∧ (∀ b ∈ w.workTable, b.size < 2 ^ 64) ∧ (∀ n ∈ w.stratified, n < 2 ^ 64) ∧
    (∀ b ∈ w.closure, b.size < 2 ^ 64) ∧ (∀ b ∈ w.registered, b.size < 2 ^ 64) ∧ (∀ b ∈ w.publicInputs, b.size < 2 ^ 64)

/-- **A V\* statement's lengths and numbers fit their 8-byte words**: Flock's statement's, its encoding's length (it is
one field), and each recursive flag's. -/
def RecRawFits (w : RecRaw) : Prop :=
  RawFits w.base ∧ w.base.encode.length < 2 ^ 64 ∧ (∀ b ∈ w.recInner, b.size < 2 ^ 64) ∧ (∀ n ∈ w.recPart, n < 2 ^ 64) ∧
    (∀ b ∈ w.recBindings, b.size < 2 ^ 64) ∧ (∀ b ∈ w.recRows, b.size < 2 ^ 64) ∧
    (∀ b ∈ w.innerCircuit, b.size < 2 ^ 64) ∧ (∀ b ∈ w.innerPublic, b.size < 2 ^ 64) ∧
    (∀ b ∈ w.innerPartition, b.size < 2 ^ 64) ∧ (∀ b ∈ w.innerCoins, b.size < 2 ^ 64) ∧
    (∀ b ∈ w.innerSession, b.size < 2 ^ 64) ∧ (∀ b ∈ w.transcript, b.size < 2 ^ 64) ∧
    (∀ b ∈ w.registration, b.size < 2 ^ 64) ∧ (∀ t ∈ w.innerStatement, t.toUTF8.size < 2 ^ 64) ∧
    (∀ ts ∈ w.innerTables, ts.length < 2 ^ 64 ∧ ∀ p ∈ ts, p.1.toUTF8.size < 2 ^ 64 ∧ p.2.size < 2 ^ 64) ∧
    (∀ b ∈ w.innerDraw, b.size < 2 ^ 64)

theorem rdOptBytes {a : Option ByteArray} (h : ∀ b ∈ a, b.size < 2 ^ 64) (rest : List UInt8) :
    (rdOpt rdBytes).run (encOpt encBytes a ++ rest) = some (a, rest) :=
  rdOpt_encOpt (fun b hb rest => FlockComplete.rdBytes_encBytes (h b hb) rest) rest

theorem rdOptNat {a : Option Nat} (h : ∀ n ∈ a, n < 2 ^ 64) (rest : List UInt8) :
    (rdOpt rdNat).run (encOpt encNat a ++ rest) = some (a, rest) :=
  rdOpt_encOpt (fun n hn rest => FlockComplete.rdNat_encNat (h n hn) rest) rest

theorem run_bind_of {α β : Type} {x : Rd α} {f : α → Rd β} {s s' : Claims.Msg} {a : α}
    (h : x.run s = some (a, s')) : (x >>= f).run s = (f a).run s' := by
  rw [StateT.run_bind, h]
  rfl

theorem raw_read_encode {w : Raw} (h : RawFits w) (rest : List UInt8) :
    Raw.read.run (w.encode ++ rest) = some (w, rest) := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15⟩ := h
  have ht : ∀ rest, (rdList (do
      let n ← rdStr
      let b ← rdBytes
      pure (n, b))).run (encList (fun p => encStr p.1 ++ encBytes p.2) w.tables ++ rest) = some (w.tables, rest) :=
    fun rest => FlockComplete.rdList_encList (fun p hp rest => by
      rw [List.append_assoc, run_bind_of (FlockComplete.rdStr_encStr (h5 p hp).1 _),
        run_bind_of (FlockComplete.rdBytes_encBytes (h5 p hp).2 _)]
      rfl) h4 rest
  simp only [Raw.encode, List.append_assoc]
  unfold Raw.read
  rw [run_bind_of (FlockComplete.rdStr_encStr format_fits _)]
  dsimp only
  rw [if_neg (fun h => h rfl)]
  rw [run_bind_of (FlockComplete.rdStr_encStr h1 _), run_bind_of (FlockComplete.rdBytes_encBytes h2 _),
    run_bind_of (FlockComplete.rdBytes_encBytes h3 _), run_bind_of (ht _), run_bind_of (rdBool_encBool _ _),
    run_bind_of (rdBool_encBool _ _), run_bind_of (FlockComplete.rdNat_encNat h6 _), run_bind_of (rdOptBytes h7 _),
    run_bind_of (rdOptBytes h8 _), run_bind_of (rdOptBytes h9 _), run_bind_of (rdOptNat h10 _),
    run_bind_of (rdOptBytes h11 _), run_bind_of (rdOptNat h12 _), run_bind_of (rdOptBytes h13 _),
    run_bind_of (rdOptBytes h14 _), run_bind_of (rdOptBytes h15 _)]
  rfl

/-- **Flock's statement reads back as itself** where it fits. -/
theorem decode_encode_fits {w : Raw} (h : RawFits w) : FlockDeploy.decode w.encode = some w := by
  have hr := raw_read_encode h []
  rw [List.append_nil] at hr
  unfold FlockDeploy.decode
  rw [hr]
  simp

theorem rformat_fits : RecDeploy.RFORMAT.toUTF8.size < 2 ^ 64 := by decide

theorem rdTables {ts : List (String × ByteArray)} (h : ts.length < 2 ^ 64 ∧ ∀ p ∈ ts, p.1.toUTF8.size < 2 ^ 64 ∧ p.2.size < 2 ^ 64)
    (rest : List UInt8) : (rdList (do
      let n ← rdStr
      let b ← rdBytes
      pure (n, b))).run (encList (fun p => encStr p.1 ++ encBytes p.2) ts ++ rest) = some (ts, rest) :=
  FlockComplete.rdList_encList (fun p hp rest => by
      rw [List.append_assoc, run_bind_of (FlockComplete.rdStr_encStr (h.2 p hp).1 _),
        run_bind_of (FlockComplete.rdBytes_encBytes (h.2 p hp).2 _)]
      rfl) h.1 rest

theorem recRaw_read_encode {w : RecRaw} (h : RecRawFits w) (rest : List UInt8) :
    RecRaw.read.run (w.encode ++ rest) = some (w, rest) := by
  obtain ⟨h0, hb, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14⟩ := h
  have hbs : (⟨w.base.encode.toArray⟩ : ByteArray).size < 2 ^ 64 := by
    show w.base.encode.toArray.size < 2 ^ 64
    simpa using hb
  have hdec : FlockDeploy.decode (⟨w.base.encode.toArray⟩ : ByteArray).data.toList = some w.base := by
    simpa using decode_encode_fits h0
  simp only [RecRaw.encode, List.append_assoc]
  unfold RecRaw.read
  rw [run_bind_of (FlockComplete.rdStr_encStr rformat_fits _)]
  dsimp only
  rw [if_neg (fun h => h rfl)]
  rw [run_bind_of (FlockComplete.rdBytes_encBytes hbs _)]
  dsimp only
  rw [hdec]
  dsimp only
  rw [run_bind_of (rdOptBytes h1 _), run_bind_of (rdOptNat h2 _), run_bind_of (rdOptBytes h3 _),
    run_bind_of (rdOptBytes h4 _), run_bind_of (rdOptBytes h5 _), run_bind_of (rdOptBytes h6 _),
    run_bind_of (rdOptBytes h7 _), run_bind_of (rdOptBytes h8 _), run_bind_of (rdOptBytes h9 _),
    run_bind_of (rdOptBytes h10 _), run_bind_of (rdOptBytes h11 _),
    run_bind_of (rdOpt_encOpt (fun t ht rest => FlockComplete.rdStr_encStr (h12 t ht) rest) _),
    run_bind_of (rdOpt_encOpt (fun b _ rest => rdBool_encBool b rest) _),
    run_bind_of (rdOpt_encOpt (fun ts hts rest => rdTables (h13 ts hts) rest) _), run_bind_of (rdOptBytes h14 _)]
  rfl

/-- **A V\* statement reads back as itself** where it fits. -/
theorem decodeRec_encode_fits {w : RecRaw} (h : RecRawFits w) : decodeRec w.encode = some w := by
  have hr := recRaw_read_encode h []
  rw [List.append_nil] at hr
  unfold decodeRec
  rw [hr]
  simp

theorem obind_some {α β : Type} {m : Option α} {f : α → Option β} {b : β} (h : (m >>= f) = some b) :
    ∃ a, m = some a ∧ f a = some b := by
  cases m with
  | none => cases h
  | some a => exact ⟨a, rfl, h⟩

theorem toOption_ok {ε α : Type} {e : Except ε α} {a : α} (h : e.toOption = some a) : e = .ok a := by
  cases e with
  | error _ => cases h
  | ok a' => cases h; rfl

/-- **What `readXPre` reads**: `x`'s fields (`decodeX`), the inner statement's (`FlockDeploy.decode`), the verifier's
own inner shape and the inner statement's partition, each as `readXPre` reads it. -/
theorem readXPre_spec {x : Claims.Msg} {r : XR} (h : readXPre x = some r) :
    decodeX x = some r.w ∧ FlockDeploy.decode r.w.inner = some r.innerRaw ∧
      (do RecVStar.Inner.ofJson (← jsonOf "rec-inner" r.w.recInner)) = .ok r.shape ∧
      r.innerRaw.partition.mapM (jsonOf "partition") = .ok r.partition := by
  unfold readXPre at h
  obtain ⟨w, hw, h⟩ := obind_some h
  obtain ⟨ri, hri, h⟩ := obind_some h
  obtain ⟨shape, hshape, h⟩ := obind_some h
  obtain ⟨partition, hpart, h⟩ := obind_some h
  obtain ⟨dims, _, h⟩ := obind_some h
  obtain ⟨vs, _, h⟩ := obind_some h
  split at h
  · cases h
  · cases h
    exact ⟨hw, hri, toOption_ok hshape, toOption_ok hpart⟩

theorem len_encBytes (b : ByteArray) : (encBytes b).length = 8 + b.size := by
  show (encNat b.size ++ b.data.toList).length = _
  rw [List.length_append, FlockComplete.encNat_length, Array.length_toList]
  rfl

theorem len_encStr (t : String) : (encStr t).length = 8 + t.toUTF8.size := len_encBytes _

theorem encOpt_le {o : Option ByteArray} : ∀ b ∈ o, b.size ≤ (encOpt encBytes o).length := by
  intro b hb
  cases o with
  | none => cases hb
  | some b' =>
    cases hb
    simp only [encOpt, List.length_cons, len_encBytes]
    omega

theorem flatMap_le {α : Type} {f : α → List UInt8} : ∀ {l : List α} {a : α}, a ∈ l →
    (f a).length ≤ (l.flatMap f).length
  | [], _, h => by cases h
  | b :: l, a, h => by
    rw [List.flatMap_cons, List.length_append]
    cases h with
    | head => omega
    | tail _ h => have := flatMap_le (f := f) h; omega

theorem length_le_flatMap {α : Type} {f : α → List UInt8} (hf : ∀ a, 1 ≤ (f a).length) :
    ∀ l : List α, l.length ≤ (l.flatMap f).length
  | [] => by simp
  | a :: l => by
    rw [List.flatMap_cons, List.length_append, List.length_cons]
    have := hf a
    have := length_le_flatMap hf l
    omega

theorem raw_len (w : Raw) :
    w.statement.toUTF8.size + w.circuit.size + w.publicFile.size +
      (w.tables.flatMap fun p => encStr p.1 ++ encBytes p.2).length + (encOpt encBytes w.draw).length +
      (encOpt encBytes w.partition).length + (encOpt encBytes w.program).length +
      (encOpt encBytes w.registered).length ≤ w.encode.length := by
  simp only [Raw.encode, encList, List.length_append, len_encStr, len_encBytes]
  omega

theorem raw_pub (w : Raw) (P : ByteArray) :
    ({ w with publicFile := P } : Raw).encode.length + w.publicFile.size = w.encode.length + P.size := by
  simp only [Raw.encode, List.length_append, len_encStr, len_encBytes]
  omega

theorem recx_len (w : RecX) :
    w.inner.length + w.recInner.size + (encOpt encBytes w.registration).length +
      (w.parts.flatMap PartX.encode).length ≤ w.encode.length := by
  simp only [RecX.encode, encList, List.length_append, len_encStr, len_encBytes]
  omega

theorem partx_len (p : PartX) : p.base.encode.length + p.bindings.size ≤ p.encode.length := by
  have hs : (⟨p.base.encode.toArray⟩ : ByteArray).size = p.base.encode.length := by
    show p.base.encode.toArray.size = _
    simp
  simp only [PartX.encode, List.length_append, len_encBytes, hs]
  omega

/-- **Part `s`'s statement fits** where the files the run stages for it do: every field it takes from `x` fits, since
`x` decodes, and Flock's statement with the staged public file is no longer than `x` and that file together. -/
theorem partRaw_fits {x : Claims.Msg} {r : XR} (hr : readX x = some r) {st : Stage}
    {ms : List (Claims.Msg × Claims.Out)} {s : ℕ} {q : PartX} (hq : r.w.parts[s]? = some q) (hP : PartForm q.base)
    (hpub : x.length + (st.pub x ms s).size < 2 ^ 64)
    (hrec : (st.record x ms).size < 2 ^ 64) (htr : (st.transcript x ms).size < 2 ^ 64)
    (hrows : (st.rows x ms).size < 2 ^ 64) :
    RecRawFits (partRaw r st x ms s q) := by
  obtain ⟨hw, hri, -, -⟩ := readXPre_spec (readXPre_of_readX hr)
  have hx := decodeX_encode hw
  have hi := FlockDeploy.decode_encode hri
  obtain ⟨_, _, _, hcnt, htab, hwk, hwt, hstr, hcl, hpi⟩ := hP
  have hmem : q ∈ r.w.parts := List.mem_of_getElem? hq
  have hslt : s < r.w.parts.length := by
    rcases List.getElem?_eq_some_iff.mp hq with ⟨h, _⟩
    exact h
  have h1 := recx_len r.w
  have h2 := flatMap_le (f := PartX.encode) hmem
  have h3 := partx_len q
  have h4 := raw_len q.base
  have h5 := raw_pub q.base (st.pub x ms s)
  have h6 := raw_len r.innerRaw
  have h7 := length_le_flatMap (f := PartX.encode) (fun p => by
    simp only [PartX.encode, List.length_append, len_encBytes]
    omega) r.w.parts
  rw [hx] at h1
  rw [hi] at h6
  have hxl : x.length < 2 ^ 64 := by omega
  refine ⟨⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_,
    ?_, ?_, ?_⟩
  · show q.base.statement.toUTF8.size < 2 ^ 64; omega
  · show q.base.circuit.size < 2 ^ 64; omega
  · show (st.pub x ms s).size < 2 ^ 64; omega
  · show q.base.tables.length < 2 ^ 64; rw [htab]; decide
  · show ∀ p ∈ q.base.tables, _; rw [htab]; intro p hp; cases hp
  · show q.base.count < 2 ^ 64; rw [hcnt]; decide
  · intro b hb; have := encOpt_le (o := q.base.draw) b hb; omega
  · intro b hb; have := encOpt_le (o := q.base.partition) b hb; omega
  · intro b hb; have := encOpt_le (o := q.base.program) b hb; omega
  · show ∀ n ∈ q.base.work, _; rw [hwk]; intro n hn; cases hn
  · show ∀ n ∈ q.base.workTable, _; rw [hwt]; intro n hn; cases hn
  · show ∀ n ∈ q.base.stratified, _; rw [hstr]; intro n hn; cases hn
  · show ∀ n ∈ q.base.closure, _; rw [hcl]; intro n hn; cases hn
  · intro b hb; have := encOpt_le (o := q.base.registered) b hb; omega
  · show ∀ n ∈ q.base.publicInputs, _; rw [hpi]; intro n hn; cases hn
  · have h5' : (partRaw r st x ms s q).base.encode.length + q.base.publicFile.size =
        q.base.encode.length + (st.pub x ms s).size := h5
    omega
  · intro b hb; cases hb; omega
  · intro n hn; cases hn; omega
  · intro b hb; cases hb; omega
  · intro b hb; cases hb; exact hrows
  · intro b hb; cases hb; omega
  · intro b hb; cases hb; omega
  · intro b hb; have := encOpt_le (o := r.innerRaw.partition) b hb; omega
  · intro b hb; cases hb
  · intro b hb; cases hb; exact hrec
  · intro b hb; cases hb; exact htr
  · intro b hb; have := encOpt_le (o := r.w.registration) b hb; omega
  · intro t ht; cases ht; omega
  · intro ts hts
    cases hts
    have h8 := length_le_flatMap (f := fun p : String × ByteArray => encStr p.1 ++ encBytes p.2) (fun p => by
      simp only [List.length_append, len_encStr]; omega) r.innerRaw.tables
    refine ⟨by omega, fun p hp => ?_⟩
    have h9 := flatMap_le (f := fun p : String × ByteArray => encStr p.1 ++ encBytes p.2) hp
    simp only [List.length_append, len_encStr, len_encBytes] at h9
    constructor <;> omega
  · intro b hb; have := encOpt_le (o := r.innerRaw.draw) b hb; omega

/-- **Part `s`'s staged statement is `partRaw`'s encoding** where it fits. -/
theorem staged_eq {x : Claims.Msg} {r : XR} (hr : readX x = some r) {st : Stage} {ms : List (Claims.Msg × Claims.Out)}
    {s : ℕ} {q : PartX} (hq : r.w.parts[s]? = some q) (hfit : RecRawFits (partRaw r st x ms s q)) :
    staged st x ms s = (partRaw r st x ms s q).encode := by
  unfold staged
  rw [hr]
  simp only [hq, decodeRec_encode_fits hfit, if_true]

/-- **The outer verifier reads part `s`'s staged statement as `partRaw`'s inputs** (`RecRaw.ins`), where it fits and is
a `--zk` statement. -/
theorem insO_staged {x : Claims.Msg} {r : XR} (hr : readX x = some r) {st : Stage}
    {ms : List (Claims.Msg × Claims.Out)} {s : ℕ} {q : PartX} (hq : r.w.parts[s]? = some q)
    (hfit : RecRawFits (partRaw r st x ms s q)) (hzk : q.base.zk = true) :
    insO (staged st x ms s) = (partRaw r st x ms s q).ins.toOption := by
  rw [staged_eq hr hq hfit]
  unfold insO
  rw [decodeRec_encode_fits hfit]
  have hz : (partRaw r st x ms s q).base.zk = true := hzk
  simp only [Option.bind_eq_bind, Option.bind_some, hz, Bool.true_eq_false, ite_false]

/-- **Where part `s`'s staged numbers are `nums`', its V\* statement is the staged one.** -/
theorem vstarAt_eq_staged {st : Stage} {x : Claims.Msg} {ms : List (Claims.Msg × Claims.Out)} {s : ℕ}
    (hn : numsOf (staged st x ms s) = nums st x s) : vstarAt st x ms s = staged st x ms s := by
  unfold vstarAt
  rw [if_pos hn]

/-! ## The readers agree -/

/-- `linesAux` splits at the first newline. -/
theorem linesAux_eq : ∀ (l cur : List UInt8), linesAux l cur =
    (cur.reverse ++ l.takeWhile (· != 10)) :: (match l.dropWhile (· != 10) with
      | [] => []
      | _ :: r => linesAux r [])
  | [], cur => by simp [linesAux]
  | b :: bs, cur => by
    by_cases hb : b = 10
    · subst hb
      simp [linesAux]
    · have hb' : (b != 10) = true := by simpa using hb
      rw [linesAux, if_neg hb, linesAux_eq bs (b :: cur), List.takeWhile_cons_of_pos (p := (· != 10)) hb',
        List.dropWhile_cons_of_pos (p := (· != 10)) hb']
      simp

/-- **`lineEnd` is where the line from `i` stops**: the first newline at or after `i`, else the end. -/
theorem lineEnd_spec (b : ByteArray) : ∀ (k i : Nat), b.size - i = k → i ≤ b.size →
    i ≤ VStar.Recursive.lineEnd b i ∧ VStar.Recursive.lineEnd b i ≤ b.size ∧
      (b.data.toList.drop i).takeWhile (· != 10) =
        (b.data.toList.drop i).take (VStar.Recursive.lineEnd b i - i) ∧
      (b.data.toList.drop i).dropWhile (· != 10) = b.data.toList.drop (VStar.Recursive.lineEnd b i)
  | k, i, hk, hi => by
    rw [VStar.Recursive.lineEnd]
    by_cases h : i < b.size
    · have hl : i < b.data.toList.length := by simpa using h
      rw [dif_pos h]
      have hd : b.data.toList.drop i = b[i] :: b.data.toList.drop (i + 1) := by
        rw [List.drop_eq_getElem_cons hl]
        rfl
      by_cases hn : b[i] == 10
      · rw [if_pos hn]
        have hn' : (b[i] != 10) = false := by simpa using hn
        refine ⟨Nat.le_refl _, Nat.le_of_lt h, ?_, ?_⟩
        · rw [hd, List.takeWhile_cons_of_neg (p := (· != 10)) (by simp [hn'])]
          simp
        · rw [hd, List.dropWhile_cons_of_neg (p := (· != 10)) (by simp [hn'])]
      · rw [if_neg hn]
        have hn' : (b[i] != 10) = true := by simpa using hn
        obtain ⟨h1, h2, h3, h4⟩ := lineEnd_spec b (b.size - (i + 1)) (i + 1) rfl (by omega)
        refine ⟨by omega, h2, ?_, ?_⟩
        · rw [hd, List.takeWhile_cons_of_pos (p := (· != 10)) hn', h3]
          have : VStar.Recursive.lineEnd b (i + 1) - i = (VStar.Recursive.lineEnd b (i + 1) - (i + 1)) + 1 := by omega
          rw [this, List.take_succ_cons]
        · rw [hd, List.dropWhile_cons_of_pos (p := (· != 10)) hn', h4]
    · rw [dif_neg h]
      have : i = b.size := by omega
      subst this
      have hn : List.drop b.size b.data.toList = [] := List.drop_eq_nil_of_le (by simp)
      rw [hn]
      exact ⟨Nat.le_refl _, Nat.le_refl _, by simp, by simp⟩
termination_by k => k
decreasing_by omega

/-- The lines from byte `i` on, as the model's `linesAux` splits them; none past the end. -/
def linesFrom (b : ByteArray) (i : Nat) : List (List UInt8) :=
  if i ≤ b.size then linesAux (b.data.toList.drop i) [] else []

theorem linesFrom_lt {b : ByteArray} {i : Nat} (h : i < b.size) :
    linesFrom b i = (b.extract i (VStar.Recursive.lineEnd b i)).data.toList ::
      linesFrom b (VStar.Recursive.lineEnd b i + 1) := by
  obtain ⟨h1, h2, h3, h4⟩ := lineEnd_spec b _ i rfl (Nat.le_of_lt h)
  unfold linesFrom
  rw [if_pos (Nat.le_of_lt h), linesAux_eq, h3, h4]
  rw [List.reverse_nil, List.nil_append, ByteArray.data_extract, Array.toList_extract, List.extract_eq_drop_take]
  congr 1
  by_cases he : VStar.Recursive.lineEnd b i < b.size
  · have hl : VStar.Recursive.lineEnd b i < b.data.toList.length := by simpa using he
    rw [List.drop_eq_getElem_cons hl, if_pos (by omega)]
  · have he' : VStar.Recursive.lineEnd b i = b.size := by omega
    rw [he', if_neg (by omega), List.drop_eq_nil_of_le (by simp)]

theorem ok_bind {ε α β : Type} (a : α) (f : α → Except ε β) : (Except.ok a >>= f) = f a := rfl

/-- The model's reading of the lines `L`: their first `META` line among `n`, as JSON. -/
def metaRead (n : Nat) (L : List (List UInt8)) : V Json := do
  let t ← metaLine n L
  match Json.parse (t.drop 5).toString with
  | .error e => throw s!"rec inner circuit: META is not JSON: {e}"
  | .ok m => pure m

theorem metaRead_nil (n : Nat) : metaRead n [] = .error "rec inner circuit: no META line" := by
  cases n <;> rfl

theorem metaRead_end {b : ByteArray} {i : Nat} (h : b.size ≤ i) (n : Nat) :
    metaRead n (linesFrom b i) = .error "rec inner circuit: no META line" := by
  unfold linesFrom
  by_cases hi : i ≤ b.size
  · have : i = b.size := by omega
    subst this
    rw [if_pos hi]
    simp only [List.drop_eq_nil_of_le (by simp : b.data.toList.length ≤ b.size), linesAux, List.reverse_nil]
    cases n with
    | zero => rfl
    | succ n =>
      show (do let t ← metaLine (n + 1) [[]]; _) = _
      simp only [metaLine]
      have hu : String.fromUTF8? ⟨(([] : List UInt8)).toArray⟩ = some "" := by decide
      rw [hu]
      simp only [show ("".startsWith "META ") = false by decide, Bool.false_eq_true, if_false]
      exact metaRead_nil n
  · rw [if_neg hi]
    exact metaRead_nil n

/-- `VStar.Recursive.metaOf`'s loop body. -/
def metaBody (b : ByteArray) : Nat → Option Json × Nat → V (ForInStep (Option Json × Nat)) := fun _ s =>
  if s.2 < b.size then
    match String.fromUTF8? (b.extract s.2 (VStar.Recursive.lineEnd b s.2)) with
    | some line =>
      if line.startsWith "META " = true then
        match Json.parse (line.drop 5).toString with
        | .ok m => pure (.done (some m, s.2))
        | .error err => throw (toString "rec inner circuit: META is not JSON: " ++ toString err)
      else pure (.yield (none, VStar.Recursive.lineEnd b s.2 + 1))
    | _ => throw "rec inner circuit: not UTF-8"
  else pure (.yield (none, s.2))

def metaPost (s : Option Json × Nat) : V Json :=
  match s.1 with
  | some r => pure r
  | none => throw "rec inner circuit: no META line"

theorem metaOf_loop (b : ByteArray) :
    VStar.Recursive.metaOf b = forIn [:9] ((none : Option Json), 0) (metaBody b) >>= metaPost := by
  unfold VStar.Recursive.metaOf
  simp only [metaPost]
  congr 1
  funext s
  rcases s with ⟨_ | r, n⟩ <;> rfl

theorem loop_eq (b : ByteArray) : ∀ (l : List Nat) (i : Nat),
    forIn l ((none : Option Json), i) (metaBody b) >>= metaPost = metaRead l.length (linesFrom b i)
  | [], i => by
    by_cases h : b.size ≤ i
    · rw [metaRead_end h]; rfl
    · rw [linesFrom_lt (by omega)]; rfl
  | a :: l, i => by
    rw [List.forIn_cons]
    by_cases h : i < b.size
    · rw [linesFrom_lt h]
      simp only [metaBody, if_pos h]
      show _ = (do let t ← metaLine (l.length + 1) _; _)
      simp only [metaLine]
      have he : (⟨(b.extract i (VStar.Recursive.lineEnd b i)).data.toList.toArray⟩ : ByteArray) =
          b.extract i (VStar.Recursive.lineEnd b i) := by rw [Array.toArray_toList]
      rw [he]
      cases hu : String.fromUTF8? (b.extract i (VStar.Recursive.lineEnd b i)) with
      | none => rfl
      | some line =>
        simp only
        by_cases hm : line.startsWith "META " = true
        · simp only [hm, if_true]
          rw [ok_bind]
          cases hp : Json.parse (line.drop 5).toString <;> rfl
        · simp only [hm, Bool.false_eq_true, if_false]
          have := loop_eq b l (VStar.Recursive.lineEnd b i + 1)
          simpa [metaRead] using this
    · simp only [metaBody, if_neg h]
      have ih := loop_eq b l i
      have hb : b.size ≤ i := by omega
      rw [metaRead_end hb] at ih ⊢
      simpa using ih

/-- **The compiled `metaOf` is the model's**: the first `META` line among a circuit file's first nine, as JSON, with the
same refusals. -/
theorem metaOf_eq (b : ByteArray) : VStar.Recursive.metaOf b = RecDeploy.metaOf b := by
  rw [metaOf_loop, Std.Legacy.Range.forIn_eq_forIn_range']
  have := loop_eq b (List.range' 0 (Std.Legacy.Range.size [:9]) 1) 0
  rw [this]
  have hl : (List.range' 0 (Std.Legacy.Range.size [:9]) 1).length = 9 := by decide
  rw [hl]
  unfold linesFrom
  rw [if_pos (Nat.zero_le _), List.drop_zero]
  rfl

/-- `jsonOf`'s reading doesn't depend on what it names. -/
theorem jsonOf_what {a a' : String} {b : ByteArray} {j : Json} (h : jsonOf a b = .ok j) : jsonOf a' b = .ok j := by
  unfold jsonOf at h ⊢
  revert h
  cases String.fromUTF8? b with
  | none => intro h; cases h
  | some t =>
    intro h
    dsimp only at h ⊢
    revert h
    cases Json.parse t with
    | error e => intro h; cases h
    | ok j' => intro h; exact h

theorem bind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : x >>= f = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error => cases h
  | ok a => exact ⟨a, rfl, h⟩

/-- A blob's lookup doesn't depend on what it names. -/
theorem blobs_get_what {bs : Core.Blobs} {d a a' : String} {b : ByteArray} (h : bs.get d a = .ok b) :
    bs.get d a' = .ok b := by
  unfold Core.Blobs.get at h ⊢
  obtain ⟨d', hd, h⟩ := bind_ok h
  rw [hd]
  simp only [bind, Except.bind] at h ⊢
  split at h
  · cases h
  · rw [if_neg (by assumption)]
    simp only [pure, Except.pure] at h ⊢
    cases hf : bs.entries.find? (·.1 == d') with
    | none => rw [hf] at h; cases h
    | some p => rw [hf] at h; exact h

/-- **The compiled `registeredOf`'s ports are `Registered.ofJson`'s**, the model's reading of a registered file. -/
theorem registeredOf_own {j : Json}
    {t : Array Registered.Port × Bool × Array (String × Array (Array Nat × ByteArray)) × Array String}
    (h : VStar.Recursive.registeredOf j = .ok t) : Registered.ofJson j = .ok t.1 := by
  unfold VStar.Recursive.registeredOf at h
  obtain ⟨own, h1, h⟩ := bind_ok h
  obtain ⟨known, _, h⟩ := bind_ok h
  obtain ⟨_, _, h⟩ := bind_ok h
  obtain ⟨c, _, h⟩ := bind_ok h
  obtain ⟨k, _, h⟩ := bind_ok h
  cases h
  exact h1

/-- `fits` passes only a file under `FIELD_MAX` bytes. -/
theorem fits_ok {b : ByteArray} {what : String} (h : VStar.Recursive.fits b what = .ok ()) :
    b.size < VStar.Recursive.FIELD_MAX := by
  unfold VStar.Recursive.fits VStar.Recursive.need at h
  split at h
  · exact of_decide_eq_true ‹_›
  · first | (simp [throw, throwThe, MonadExceptOf.throw] at h; done) | cases h

theorem mapError_ok {ε ε' α : Type} {f : ε → ε'} {x : Except ε α} {a : α} (h : x.mapError f = .ok a) :
    x = .ok a := by
  cases x with
  | error e => cases h
  | ok b => cases h; rfl

/-- **The staged rows as the compiled `innerHeld` reads them are `RecRaw.ins`' `--rec-rows`**, port for port. -/
theorem rows_model {b : ByteArray} {rows : Array Registered.Port}
    (h : ((do
        let some t := String.fromUTF8? b | throw "not UTF-8"
        pure (← VStar.Recursive.registeredOf (← (Json.parse t).mapError (s!"not JSON: " ++ ·))).1) : Flock.V _).mapError
          (s!"rec rows: " ++ ·) = .ok rows) :
    (some b).mapM (fun b => do
      (Registered.ofJson (← jsonOf "rec rows" b)).mapError fun e => s!"rec rows: {e}") = .ok (some rows) := by
  have h := mapError_ok h
  cases hu : String.fromUTF8? b with
  | none =>
    simp only [hu] at h
    first | (simp [throw, throwThe, MonadExceptOf.throw] at h; done) | cases h
  | some t =>
    simp only [hu] at h
    obtain ⟨j, hj, h⟩ := bind_ok h
    obtain ⟨r, hr, h⟩ := bind_ok h
    cases h
    have hjs : jsonOf "rec rows" b = .ok j := by
      simp only [jsonOf, hu, mapError_ok hj]
      try rfl
    simp only [Option.mapM]
    rw [hjs, ok_bind, registeredOf_own hr]
    try rfl

/-- **What `innerHeld` holds**, read off the blobs: the inner circuit, the inner public file, the inner session's
record, the inner statement's draw, the coin server's transcript and the registration, each as text where `innerHeld`
reads text; its shape (`shapeOf`), check 5's source, the inner circuit's binding, the commits and the link points; V*'s
staged shared rows, read as `RecRaw.ins` reads `--rec-rows`; Flock's own checks on the inner statement (`innerOk`);
and the record, transcript and staged rows each under `FIELD_MAX` bytes (`fits`). -/
theorem innerHeld_ok {o : VStar.Recursive.Own} {pr : VStar.Recursive.Proofs} {bs : Core.Blobs}
    {h : VStar.Recursive.InnerHeld} (hh : VStar.Recursive.innerHeld o pr bs = .ok h) :
    ∃ (circuitB pubB recB drawB trB rowsB : ByteArray) (record t : String) (reg : Option String),
      bs.get o.circuit "the inner circuit" = .ok circuitB ∧ bs.get o.pub "the inner public file" = .ok pubB ∧
      VStar.Recursive.shapeOf circuitB pubB o.partition o.program = .ok h.shape ∧
      bs.get pr.record "the inner session's record" = .ok recB ∧ String.fromUTF8? recB = some record ∧
      bs.get o.draw "the inner statement's draw" = .ok drawB ∧
      h.src = .files
        { circuit := circuitB, pub := pubB, partition := o.partition, record := record,
          form := some { statement := VStar.Recursive.INNER_STATEMENT, os := true, tables := [], draw := some drawB } } ∧
      h.bound = (do RecSetup.metaBound (← VStar.Recursive.metaOf circuitB)) ∧
      bs.get pr.transcript "rec inner transcript" = .ok trB ∧ String.fromUTF8? trB = some t ∧
      (o.registration.mapM fun d => do
          let some s := String.fromUTF8? (← bs.get d "rec inner registration")
            | throw "rec inner registration: not UTF-8"
          pure s : Flock.V (Option String)) = .ok reg ∧
      RecSetup.commits (some h.shape) (pure t) (pure reg) h.bound record = .ok h.commits ∧
      (match Json.parse t with
        | .ok j => RecVStar.innerLink j record
        | .error e => throw s!"rec inner transcript: not JSON: {e}") = .ok h.link ∧
      bs.get pr.rows "the staged rows" = .ok rowsB ∧
      (some rowsB).mapM (fun b => do
        (Registered.ofJson (← jsonOf "rec rows" b)).mapError fun e => s!"rec rows: {e}") = .ok (some h.rows) ∧
      VStar.Recursive.innerOk o circuitB pubB drawB = .ok () ∧
      recB.size < VStar.Recursive.FIELD_MAX ∧ trB.size < VStar.Recursive.FIELD_MAX ∧
      rowsB.size < VStar.Recursive.FIELD_MAX := by
  unfold VStar.Recursive.innerHeld at hh
  obtain ⟨circuitB, hc, hh⟩ := bind_ok hh
  obtain ⟨pubB, hp, hh⟩ := bind_ok hh
  obtain ⟨shape, hs, hh⟩ := bind_ok hh
  obtain ⟨recB, hrB, hh⟩ := bind_ok hh
  obtain ⟨⟨⟩, hfr, hh⟩ := bind_ok hh
  cases hu : String.fromUTF8? recB with
  | none => rw [hu] at hh; cases hh
  | some record =>
    rw [hu] at hh
    obtain ⟨drawB, hd, hh⟩ := bind_ok hh
    obtain ⟨⟨⟩, hio, hh⟩ := bind_ok hh
    obtain ⟨trB, htB, hh⟩ := bind_ok hh
    obtain ⟨⟨⟩, hft, hh⟩ := bind_ok hh
    cases ht : String.fromUTF8? trB with
    | none => rw [ht] at hh; cases hh
    | some t =>
      rw [ht] at hh
      obtain ⟨reg, hreg, hh⟩ := bind_ok hh
      obtain ⟨commits, hcm, hh⟩ := bind_ok hh
      simp only at hh
      cases hj : Json.parse t with
      | error e =>
        rw [hj] at hh
        obtain ⟨_, hl, _⟩ := bind_ok hh
        cases hl
      | ok j =>
        rw [hj] at hh
        obtain ⟨link, hl, hh⟩ := bind_ok hh
        obtain ⟨rowsB, hrowsB, hh⟩ := bind_ok hh
        obtain ⟨⟨⟩, hfw, hh⟩ := bind_ok hh
        obtain ⟨rows, hrows, hh⟩ := bind_ok hh
        cases hh
        exact ⟨circuitB, pubB, recB, drawB, trB, rowsB, record, t, reg, hc, hp, hs, hrB, hu, hd, rfl, rfl, htB, ht,
          hreg, hcm, by rw [hj]; exact hl, hrowsB, rows_model hrows, mapError_ok hio, fits_ok hfr, fits_ok hft,
          fits_ok hfw⟩

/-! ## The correspondence -/

/-- **Compiled part `j`'s recursive inputs**, as `partVerdict` hands them to its setup: the statement's tags `tags0`,
the part's circuit `circuitB` and public file `pubB`, its partition and program (`pj`'s), its registered reads, its draw,
the inner session held (`h`: the verifier's inner shape, check 5's source, the inner circuit's binding, the commits and
the link points), `--rec-part j`, the bindings and the rows. -/
def partIns (tags0 : Tags) (circuitB pubB : ByteArray) (pj : VStar.Recursive.PartOwn)
    (registered : Option (Array Registered.Port)) (draw : Option Json) (h : VStar.Recursive.InnerHeld) (j : ℕ)
    (bindings : VStar.Bindings) (rows : Option (Array Registered.Port)) : RecIns where
  base :=
    { stmtTags := tags0
      registered
      read :=
        { I := { tags := tags0.withCoins (partCoins tags0), circuitFile := circuitB, publicFile := pubB,
                 coins := partCoins tags0, count := 1, tables := [], partition := some pj.partition,
                 program := some pj.program, work := none, stratified := none, closure := none }
          zk := true
          draw
          publicInputs := none
          registered := registered.isSome } }
  inner := some h.shape
  part := some j
  bindings := some bindings
  src := some h.src
  bound := h.bound
  commits := some h.commits
  link := some h.link
  rows

/-- `withCoins` changes a statement's tags only for the coin server's coins, never for a keyed coin tree. -/
theorem withCoins_tree (t : Tags) (s : CoinTree.Spec) : t.withCoins (.tree s) = t := by
  unfold Tags.withCoins
  rfl

/-- **The session's tags at `partIns` are the compiled part's**: `verify --zk`'s (`ZkExec.zkTags`) at the keyed coin
tree of one table and the part circuit's mask log, as `partVerdict` sets its session up. -/
theorem partIns_tags (tags0 : Tags) (circuitB pubB : ByteArray) (pj : VStar.Recursive.PartOwn)
    (registered : Option (Array Registered.Port)) (draw : Option Json) (h : VStar.Recursive.InnerHeld) (j : ℕ)
    (bindings : VStar.Bindings) (rows : Option (Array Registered.Port)) :
    (partIns tags0 circuitB pubB pj registered draw h j bindings rows).base.read.tags =
      Zk.tags (tags0.withCoins (partCoins tags0)) (Zk.maskLogOf circuitB) 1 :=
  congrArg (fun t => Zk.tags t (Zk.maskLogOf circuitB) 1) (withCoins_tree _ _)

/-- **What the recursive setup reads of `partIns`** (piece D2's `SetupMatch`, field by field, in its order): `--zk`, the
statement's and the session's tags, the part's circuit and staged public file, no lookup table, one table, the part's
partition and program, no public input words, the registered reads, the draw `draw`, the inner shape, check 5's
source, the commits, the link, `--rec-part j`, the bindings and the rows. -/
theorem partIns_fields (tags0 : Tags) (circuitB pubB : ByteArray) (pj : VStar.Recursive.PartOwn)
    (registered : Option (Array Registered.Port)) (draw : Option Json) (h : VStar.Recursive.InnerHeld) (j : ℕ)
    (bindings : VStar.Bindings) (rows : Option (Array Registered.Port)) :
    let i := partIns tags0 circuitB pubB pj registered draw h j bindings rows
    i.base.read.zk = true ∧ i.base.stmtTags = tags0 ∧
      i.base.read.tags = Zk.tags (tags0.withCoins (partCoins tags0)) (Zk.maskLogOf circuitB) 1 ∧
      i.base.read.I.circuitFile = circuitB ∧ i.base.read.I.publicFile = pubB ∧ i.base.read.I.tables = [] ∧
      i.base.read.I.count = 1 ∧ i.base.read.I.partition = some pj.partition ∧
      i.base.read.I.program = some pj.program ∧ i.base.read.publicInputs = none ∧ i.base.registered = registered ∧
      i.base.read.registered = registered.isSome ∧ i.base.read.draw = draw ∧ i.inner = some h.shape ∧
      i.src = some h.src ∧ i.commits = some h.commits ∧ i.link = some h.link ∧ i.part = some j ∧
      i.bindings = some bindings ∧ i.rows = rows :=
  ⟨rfl, rfl, partIns_tags tags0 circuitB pubB pj registered draw h j bindings rows, rfl, rfl, rfl, rfl, rfl, rfl, rfl,
    rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

theorem mapM_some_ok {ε α β : Type} {f : α → Except ε β} {a : α} {b : β} (h : f a = .ok b) :
    (some a).mapM f = .ok (some b) := by
  simp only [Option.mapM, h]
  rfl

/-- **`RecRaw.ins` from its readers' results.** -/
theorem recIns_of {w : RecRaw} {b : FlockDeploy.Ins} {i : Option RecVStar.Inner} {f : Option VStar.Bindings}
    {src : Option RecCoins.Source} {c : Option RecVStar.InnerCommits} {l : Option RecVStar.InnerLink}
    {rw : Option (Array Registered.Port)} (h1 : w.base.ins = .ok b)
    (h2 : w.recInner.mapM (fun b => do RecVStar.Inner.ofJson (← jsonOf "rec-inner" b)) = .ok i)
    (h3 : w.recBindings.mapM (fun b => do RecVStar.ownBindingsOfJson (← jsonOf "rec-bindings" b)) = .ok f)
    (h4 : sourceOf w = .ok src) (h5 : commitsOf w src (boundOf w.innerCircuit) i = .ok c)
    (h6 : linkInputOf w src = .ok l)
    (h7 : w.recRows.mapM (fun b => do
      (Registered.ofJson (← jsonOf "rec rows" b)).mapError fun e => s!"rec rows: {e}") = .ok rw) :
    w.ins = .ok
      { base := b, inner := i, part := w.recPart, bindings := f, src := src, bound := boundOf w.innerCircuit,
        commits := c, link := l, rows := rw } := by
  unfold RecRaw.ins
  rw [h1]; simp only [ok_bind]
  rw [h2]; simp only [ok_bind]
  rw [h3]; simp only [ok_bind]
  rw [h4]; simp only [ok_bind]
  rw [h5]; simp only [ok_bind]
  rw [h6]; simp only [ok_bind]
  rw [h7]; rfl

/-- The registration `innerHeld` reads off the blobs is the one `Names` puts in the statement, as text. -/
theorem reg_text {bs : Core.Blobs} {rd : Option String} {rw : Option ByteArray} {reg : Option String}
    (hI : (rd.mapM fun d => (bs.get d "rec inner registration").toOption) = some rw)
    (hC : (rd.mapM fun d => do
          let some s := String.fromUTF8? (← bs.get d "rec inner registration")
            | throw "rec inner registration: not UTF-8"
          pure s : Flock.V (Option String)) = .ok reg) :
    rw.mapM (textOf "rec inner registration") = .ok reg := by
  cases rd with
  | none =>
    cases hI; cases hC; rfl
  | some d =>
    simp only [Option.mapM] at hI hC
    simp only [Option.mapM, textOf]
    revert hI hC
    cases bs.get d "rec inner registration" with
    | error e => intro hI; cases hI
    | ok b =>
      intro hI hC
      simp only [Except.toOption] at hI
      cases hI
      dsimp only
      simp only [ok_bind] at hC
      revert hC
      cases String.fromUTF8? b with
      | none => intro hC; cases hC
      | some u => intro hC; cases hC; rfl

/-- **The verifier's readers take part `s`'s statement to compiled part `s`'s inputs**: `RecRaw.ins` on `partRaw`,
under `insO_staged_partIns`' hypotheses. -/
theorem ins_partRaw {r : XR} {st : Stage} {x : Claims.Msg} {ms : List (Claims.Msg × Claims.Out)} {s : ℕ} {q : PartX}
    {o : VStar.Recursive.Own} {pr : VStar.Recursive.Proofs} {bs : Core.Blobs} {h : VStar.Recursive.InnerHeld}
    (hh : VStar.Recursive.innerHeld o pr bs = .ok h)
    {pj : VStar.Recursive.PartOwn} {tags0 : Tags} (htags : Tags.find? VStar.Recursive.STATEMENT = some tags0)
    (hplain : tags0.plainLeaves = false)
    {circuitB pubB : ByteArray} {registered : Option (Array Registered.Port)} {bindings : VStar.Bindings}
    {draw : Option Json}
    (hregC : (pj.registered.mapM VStar.Recursive.registeredOf).map (Option.map (·.1)) = .ok registered)
    (hbindC : RecVStar.ownBindingsOfJson pj.bindings = .ok bindings)
    (hIc : bs.get o.circuit "the inner circuit" = .ok r.innerRaw.circuit)
    (hIp : bs.get o.pub "the inner public file" = .ok r.innerRaw.publicFile)
    (hIstmt : r.innerRaw.statement = VStar.Recursive.INNER_STATEMENT) (hIos : r.innerRaw.osCoins = true)
    (hItab : r.innerRaw.tables = [])
    (hIpart : r.innerRaw.partition.map (jsonOf "partition") = some (.ok o.partition))
    (hIdraw : r.innerRaw.draw = (bs.get o.draw "the inner statement's draw").toOption)
    (hIshape : (do RecVStar.Inner.ofJson (← jsonOf "rec-inner" r.w.recInner)) = .ok h.shape)
    (hIreg : (o.registration.mapM fun d => (bs.get d "rec inner registration").toOption) = some r.w.registration)
    (hPform : PartForm q.base) (hPc : q.base.circuit = circuitB)
    (hPpart : q.base.partition.map (jsonOf "partition") = some (.ok pj.partition))
    (hPprog : q.base.program = some pj.program)
    (hPreg : q.base.registered.map (jsonOf "registered") = pj.registered.map Except.ok)
    (hPbind : jsonOf "rec-bindings" q.bindings = .ok pj.bindings)
    (hPdraw : (q.base.draw.mapM (jsonOf "draw")).map (Option.map Flock.norm) = .ok draw)
    (hSp : st.pub x ms s = pubB)
    (hSr : bs.get pr.record "the inner session's record" = .ok (st.record x ms))
    (hSt : bs.get pr.transcript "rec inner transcript" = .ok (st.transcript x ms))
    (hSrows : bs.get pr.rows "the staged rows" = .ok (st.rows x ms)) :
    (partRaw r st x ms s q).ins = .ok (partIns tags0 circuitB pubB pj registered draw h s bindings (some h.rows)) := by
  obtain ⟨cI, pI, recB, drawB, trB, rowsB, record, t, reg, hc, hp, hs, hrB, hu, hd, hsrc, hbound, htB, ht, hreg, hcm,
    hl, hrowsB, hrows, -⟩ := innerHeld_ok hh
  have ecI : r.innerRaw.circuit = cI := Except.ok.inj (hIc.symm.trans hc)
  have epI : r.innerRaw.publicFile = pI := Except.ok.inj (hIp.symm.trans hp)
  have erec : st.record x ms = recB := Except.ok.inj (hSr.symm.trans hrB)
  have etr : st.transcript x ms = trB := Except.ok.inj (hSt.symm.trans htB)
  have edraw : r.innerRaw.draw = some drawB := by rw [hIdraw, hd]; rfl
  obtain ⟨hst, hzk, hos, hcnt, htab, hwk, hwt, hstr, hcl, hpi⟩ := hPform
  have hpart : ∀ (o' : Option ByteArray) (a : String) (j : Json), o'.map (jsonOf a) = some (.ok j) →
      ∃ pq, o' = some pq ∧ jsonOf a pq = .ok j := by
    intro o' a j h'
    cases o' with
    | none => cases h'
    | some pq => exact ⟨pq, rfl, Option.some.inj h'⟩
  obtain ⟨pq, hpq, hpqj⟩ := hpart _ _ _ hIpart
  have hbase : (partRaw r st x ms s q).base.ins =
      .ok (partIns tags0 circuitB pubB pj registered draw h s bindings (some h.rows)).base := by
    obtain ⟨d, hd1, hd2⟩ : ∃ d, q.base.draw.mapM (jsonOf "draw") = .ok d ∧ d.map Flock.norm = draw := by
      revert hPdraw
      cases q.base.draw.mapM (jsonOf "draw") with
      | error e => intro h'; cases h'
      | ok d => intro h'; cases h'; exact ⟨d, rfl, rfl⟩
    obtain ⟨pb, hpb, hpbj⟩ := hpart _ _ _ hPpart
    have hreg' : q.base.registered.mapM (fun b => do Registered.ofJson (← jsonOf "registered reads" b)) =
        .ok registered := by
      revert hPreg
      revert hregC
      cases q.base.registered <;> cases pj.registered <;> intro hC hP
      · rw [← hC]; rfl
      · cases hP
      · cases hP
      · rename_i b j
        have hj := jsonOf_what (a' := "registered reads") (Option.some.inj hP)
        have hro : ∀ t, VStar.Recursive.registeredOf j = .ok t → Registered.ofJson j = .ok t.1 :=
          fun _ h' => registeredOf_own h'
        simp only [Option.mapM] at hC ⊢
        rw [hj, ok_bind]
        revert hC hro
        cases VStar.Recursive.registeredOf j with
        | error e => intro hC; cases hC
        | ok t =>
          intro hC hro
          rw [hro t rfl]
          cases hC
          rfl
    have hpm : q.base.partition.mapM (jsonOf "partition") = .ok (some pj.partition) := by
      rw [hpb]; exact mapM_some_ok hpbj
    have hnone : ∀ {α β : Type} (f : α → Except String β), (none : Option α).mapM f = .ok none := fun _ => rfl
    subst hSp hd2 hPc
    simp only [FlockDeploy.Raw.ins, partRaw, hst, htags, hzk, hplain, hos, hcnt, htab, hwk, hwt, hcl, hpi, hstr,
      hPprog, hd1, hpm, hreg', hnone, ok_bind, Bool.and_false, Bool.and_true, Bool.false_eq_true, ite_false,
      ite_true]
    rfl
  have h2 : (partRaw r st x ms s q).recInner.mapM (fun b => do RecVStar.Inner.ofJson (← jsonOf "rec-inner" b)) =
      .ok (some h.shape) := mapM_some_ok hIshape
  have h3 : (partRaw r st x ms s q).recBindings.mapM
      (fun b => do RecVStar.ownBindingsOfJson (← jsonOf "rec-bindings" b)) = .ok (some bindings) := by
    apply mapM_some_ok
    rw [hPbind]
    exact hbindC
  have h4 : sourceOf (partRaw r st x ms s q) = .ok (some h.src) := by
    simp only [sourceOf, formOf, partRaw, hpq, hIstmt, hIos, hItab, edraw, ecI, epI, erec, jsonOf_what hpqj, textOf,
      hu, ok_bind, Option.getD_some]
    rw [hsrc]
    rfl
  have hb : boundOf (partRaw r st x ms s q).innerCircuit = h.bound := by
    rw [hbound]
    show boundOf (some r.innerRaw.circuit) = _
    rw [ecI]
    show (do RecSetup.metaBound (← RecDeploy.metaOf cI)) = _
    rw [metaOf_eq]
  have h5 : commitsOf (partRaw r st x ms s q) (some h.src) (boundOf (partRaw r st x ms s q).innerCircuit)
      (some h.shape) = .ok (some h.commits) := by
    have htx : textOf "rec inner transcript" (st.transcript x ms) = .ok t := by
      rw [etr]; unfold textOf; rw [ht]
    have hrg : r.w.registration.mapM (textOf "rec inner registration") = .ok reg :=
      reg_text hIreg hreg
    unfold commitsOf
    rw [hb, hsrc]
    show (if (r.w.registration.isSome && (some r.innerRaw.circuit).isNone) = true then _ else
      some <$> RecSetup.commits (some h.shape) (textOf "rec inner transcript" (st.transcript x ms))
        (r.w.registration.mapM (textOf "rec inner registration")) h.bound record) = _
    rw [htx, hrg, if_neg (by simp)]
    show some <$> RecSetup.commits (some h.shape) (pure t) (pure reg) h.bound record = _
    rw [hcm]
    rfl
  have h6 : linkInputOf (partRaw r st x ms s q) (some h.src) = .ok (some h.link) := by
    have htx : textOf "rec inner transcript" (st.transcript x ms) = .ok t := by
      rw [etr]; unfold textOf; rw [ht]
    unfold linkInputOf
    rw [hsrc]
    show (do match Json.parse (← textOf "rec inner transcript" (st.transcript x ms)) with
      | .ok j => some <$> RecVStar.innerLink j record
      | .error e => .error s!"rec inner transcript: not JSON: {e}") = _
    rw [htx, ok_bind]
    revert hl
    cases Json.parse t with
    | ok j => intro hl; dsimp only at hl ⊢; rw [hl]; rfl
    | error e => intro hl; cases hl
  have h7 : (partRaw r st x ms s q).recRows.mapM (fun b => do
      (Registered.ofJson (← jsonOf "rec rows" b)).mapError fun e => s!"rec rows: {e}") = .ok (some h.rows) := by
    show (some (st.rows x ms)).mapM _ = _
    rw [Except.ok.inj (hSrows.symm.trans hrowsB)]
    exact hrows
  rw [recIns_of hbase h2 h3 h4 h5 h6 h7]
  rw [hb]
  rfl

/-- **Piece C: part `s`'s staged statement is read as compiled part `s`'s inputs.** Take `readX`'s reading `r` of `x`
and its part `s`, `q`; the compiled check's inner session held (`innerHeld`), and part `s`'s own inputs `pj` with what
the compiled `partVerdict` reads of them (`registered`, `bindings`, the statement's tags `tags0`, the part's circuit and
public file; its rows are `innerHeld`'s). Suppose:
* `Names`' clauses: the inner statement's circuit, public file, partition and draw are the blobs' and `p.own`'s, its
  statement and form those `innerHeld` sets up (`INNER_STATEMENT`, the coin server's coins, no tables), its own inner
  shape `innerHeld`'s, the registration the blobs'; part `s`'s circuit, partition, program and registered reads
  `pj`'s, its own bindings `pj`'s, its form `PartForm`, and its draw, read as the verifier reads one, `draw`;
* `Shows`' clauses: part `s`'s staged public file is `pubB`, the inner session's record and the coin server's
  transcript the blobs', and V*'s staged shared rows the evidence's `rows` blob (`hSrows`, `RowsShow`), which
  `innerHeld` reads once for every part;
* `hplain`: the statement's tags `tags0` don't have plain SHA-512 leaves, which `Raw.ins` refuses under `--zk`. Lean
  can't evaluate it (`Tags.plainLeaves` compares `Json`, whose `==` is a `partial def`); compiled `partVerdict`
  refuses it (`need (!tags0.plainLeaves)`);
* and the staged files fit their length words: `x` with part `s`'s public file, the inner session's record, the
  transcript and the staged rows are each under `2^64` bytes (the compiled check refuses a file of `2^56` bytes or
  more, `fits`).

Then the outer verifier reads part `s`'s staged statement (`insO`) as `partIns`: compiled part `s`'s inputs. -/
theorem insO_staged_partIns {x : Claims.Msg} {r : XR} (hr : readX x = some r) {st : Stage}
    {ms : List (Claims.Msg × Claims.Out)} {s : ℕ} {q : PartX} (hq : r.w.parts[s]? = some q)
    {o : VStar.Recursive.Own} {pr : VStar.Recursive.Proofs} {bs : Core.Blobs} {h : VStar.Recursive.InnerHeld}
    (hh : VStar.Recursive.innerHeld o pr bs = .ok h)
    {pj : VStar.Recursive.PartOwn} {tags0 : Tags} (htags : Tags.find? VStar.Recursive.STATEMENT = some tags0)
    (hplain : tags0.plainLeaves = false)
    {circuitB pubB : ByteArray} {registered : Option (Array Registered.Port)} {bindings : VStar.Bindings}
    {draw : Option Json}
    (hregC : (pj.registered.mapM VStar.Recursive.registeredOf).map (Option.map (·.1)) = .ok registered)
    (hbindC : RecVStar.ownBindingsOfJson pj.bindings = .ok bindings)
    (hIc : bs.get o.circuit "the inner circuit" = .ok r.innerRaw.circuit)
    (hIp : bs.get o.pub "the inner public file" = .ok r.innerRaw.publicFile)
    (hIstmt : r.innerRaw.statement = VStar.Recursive.INNER_STATEMENT) (hIos : r.innerRaw.osCoins = true)
    (hItab : r.innerRaw.tables = [])
    (hIpart : r.innerRaw.partition.map (jsonOf "partition") = some (.ok o.partition))
    (hIdraw : r.innerRaw.draw = (bs.get o.draw "the inner statement's draw").toOption)
    (hIshape : (do RecVStar.Inner.ofJson (← jsonOf "rec-inner" r.w.recInner)) = .ok h.shape)
    (hIreg : (o.registration.mapM fun d => (bs.get d "rec inner registration").toOption) = some r.w.registration)
    (hPform : PartForm q.base) (hPc : q.base.circuit = circuitB)
    (hPpart : q.base.partition.map (jsonOf "partition") = some (.ok pj.partition))
    (hPprog : q.base.program = some pj.program)
    (hPreg : q.base.registered.map (jsonOf "registered") = pj.registered.map Except.ok)
    (hPbind : jsonOf "rec-bindings" q.bindings = .ok pj.bindings)
    (hPdraw : (q.base.draw.mapM (jsonOf "draw")).map (Option.map Flock.norm) = .ok draw)
    (hSp : st.pub x ms s = pubB)
    (hSr : bs.get pr.record "the inner session's record" = .ok (st.record x ms))
    (hSt : bs.get pr.transcript "rec inner transcript" = .ok (st.transcript x ms))
    (hSrows : bs.get pr.rows "the staged rows" = .ok (st.rows x ms))
    (hpub : x.length + pubB.size < 2 ^ 64) (hrec : (st.record x ms).size < 2 ^ 64)
    (htr : (st.transcript x ms).size < 2 ^ 64) (hrows : (st.rows x ms).size < 2 ^ 64) :
    insO (staged st x ms s) = some (partIns tags0 circuitB pubB pj registered draw h s bindings (some h.rows)) := by
  have hfit := partRaw_fits hr hq hPform (by rw [hSp]; exact hpub) hrec htr hrows
  rw [insO_staged hr hq hfit hPform.2.1,
    ins_partRaw hh htags hplain hregC hbindC hIc hIp hIstmt hIos hItab hIpart hIdraw hIshape hIreg hPform hPc
      hPpart hPprog hPreg hPbind hPdraw hSp hSr hSt hSrows]
  rfl

end Verity.Recursive.Part
