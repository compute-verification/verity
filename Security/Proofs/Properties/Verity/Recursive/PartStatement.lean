import Proofs.Flock.Claims.Complete.Encode
import Proofs.Flock.Claims.Recursion.Deploy
import VStarRecursive

/-!
# Piece C of `acc_of_accepts` (PRF-202): V\* part `s`'s statement is the compiled part's inputs

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
* **The correspondence** (`insO_staged_partIns`) takes `Names`' clauses (PRF-202, piece B) and `Shows`' as hypotheses by
  name, at `readX`'s reading `r` of `x` (`readXPre_spec` relates it to `Names`' `w` and `ri`), and `PartForm`, the
  form compiled part `s` sets up (statement `verity/flock-circuit`, `--zk`, one table, no work, stratified, closure or
  public input words), which `Names` is to carry.
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

theorem raw_read_encode {w : Raw} (h : RawFits w) (rest : List UInt8) :
    Raw.read.run (w.encode ++ rest) = some (w, rest) := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15⟩ := h
  have ht : ∀ rest, (rdList (do
      let n ← rdStr
      let b ← rdBytes
      pure (n, b))).run (encList (fun p => encStr p.1 ++ encBytes p.2) w.tables ++ rest) = some (w.tables, rest) :=
    fun rest => FlockComplete.rdList_encList (fun p hp rest => by
      simp only [List.append_assoc, StateT.run_bind, FlockComplete.rdStr_encStr (h5 p hp).1,
        FlockComplete.rdBytes_encBytes (h5 p hp).2, Option.bind_eq_bind, Option.bind_some]
      rfl) h4 rest
  simp only [Raw.read, Raw.encode, List.append_assoc, StateT.run_bind, FlockComplete.rdStr_encStr format_fits,
    Option.bind_eq_bind, Option.bind_some, ne_eq, not_true_eq_false, ite_false,
    FlockComplete.rdStr_encStr h1, FlockComplete.rdBytes_encBytes h2, FlockComplete.rdBytes_encBytes h3, ht,
    rdBool_encBool, FlockComplete.rdNat_encNat h6, rdOptBytes h7, rdOptBytes h8, rdOptBytes h9, rdOptNat h10,
    rdOptBytes h11, rdOptNat h12, rdOptBytes h13, rdOptBytes h14, rdOptBytes h15]
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
      simp only [List.append_assoc, StateT.run_bind, FlockComplete.rdStr_encStr (h.2 p hp).1,
        FlockComplete.rdBytes_encBytes (h.2 p hp).2, Option.bind_eq_bind, Option.bind_some]
      rfl) h.1 rest

theorem recRaw_read_encode {w : RecRaw} (h : RecRawFits w) (rest : List UInt8) :
    RecRaw.read.run (w.encode ++ rest) = some (w, rest) := by
  obtain ⟨h0, hb, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14⟩ := h
  have hbs : (⟨w.base.encode.toArray⟩ : ByteArray).size < 2 ^ 64 := by
    show w.base.encode.toArray.size < 2 ^ 64
    simpa using hb
  have hdec : FlockDeploy.decode (⟨w.base.encode.toArray⟩ : ByteArray).data.toList = some w.base := by
    simpa using decode_encode_fits h0
  simp only [RecRaw.read, RecRaw.encode, List.append_assoc, StateT.run_bind, FlockComplete.rdStr_encStr rformat_fits,
    Option.bind_eq_bind, Option.bind_some, ne_eq, not_true_eq_false, ite_false,
    FlockComplete.rdBytes_encBytes hbs, hdec, rdOptBytes h1, rdOptNat h2, rdOptBytes h3, rdOptBytes h4, rdOptBytes h5,
    rdOptBytes h6, rdOptBytes h7, rdOptBytes h8, rdOptBytes h9, rdOptBytes h10, rdOptBytes h11,
    rdOpt_encOpt (fun t ht rest => FlockComplete.rdStr_encStr (h12 t ht) rest),
    rdOpt_encOpt (fun b _ rest => rdBool_encBool b rest),
    rdOpt_encOpt (fun ts hts rest => rdTables (h13 ts hts) rest), rdOptBytes h14]
  rfl

/-- **A V\* statement reads back as itself** where it fits. -/
theorem decodeRec_encode_fits {w : RecRaw} (h : RecRawFits w) : decodeRec w.encode = some w := by
  have hr := recRaw_read_encode h []
  rw [List.append_nil] at hr
  unfold decodeRec
  rw [hr]
  simp

/-- **What `readXPre` reads**: `x`'s fields (`decodeX`), the inner statement's (`FlockDeploy.decode`), the verifier's
own inner shape and the inner statement's partition, each as `readXPre` reads it. -/
theorem readXPre_spec {x : Claims.Msg} {r : XR} (h : readXPre x = some r) :
    decodeX x = some r.w ∧ FlockDeploy.decode r.w.inner = some r.innerRaw ∧
      (do RecVStar.Inner.ofJson (← jsonOf "rec-inner" r.w.recInner)) = .ok r.shape ∧
      r.innerRaw.partition.mapM (jsonOf "partition") = .ok r.partition := by
  sorry

/-- **Part `s`'s statement fits** where the files the run stages for it do: every field it takes from `x` fits, since
`x` decodes, and Flock's statement with the staged public file is no longer than `x` and that file together. -/
theorem partRaw_fits {x : Claims.Msg} {r : XR} (hr : readX x = some r) {st : Stage}
    {ms : List (Claims.Msg × Claims.Out)} {s : ℕ} {q : PartX} (hq : r.w.parts[s]? = some q)
    (hpub : x.length + (st.pub x ms s).size < 2 ^ 64)
    (hrec : (st.record x ms).size < 2 ^ 64) (htr : (st.transcript x ms).size < 2 ^ 64) :
    RecRawFits (partRaw r st x ms s q) := by
  sorry

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
  simp [decodeRec_encode_fits hfit, partRaw, hzk]

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
  cases hu : String.fromUTF8? b with
  | none => rw [hu] at h; cases h
  | some t =>
    rw [hu] at h ⊢
    cases hp : Json.parse t with
    | error e => rw [hp] at h; cases h
    | ok j' => rw [hp] at h ⊢; exact h

theorem bind_ok {ε α β : Type} {m : Except ε α} {f : α → Except ε β} {b : β} (h : (m >>= f) = .ok b) :
    ∃ a, m = .ok a ∧ f a = .ok b := by
  cases m with
  | error e => cases h
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
  obtain ⟨ps, _, h⟩ := bind_ok h
  cases h
  exact h1

/-- **What `innerHeld` holds**, read off the blobs: the inner circuit, the inner public file, the inner session's
record, the inner statement's draw, the coin server's transcript and the registration, each as text where `innerHeld`
reads text; its shape (`shapeOf`), check 5's source, the inner circuit's binding, the commits and the link points. -/
theorem innerHeld_ok {o : VStar.Recursive.Own} {pr : VStar.Recursive.Proofs} {bs : Core.Blobs}
    {h : VStar.Recursive.InnerHeld} (hh : VStar.Recursive.innerHeld o pr bs = .ok h) :
    ∃ (circuitB pubB recB drawB trB : ByteArray) (record t : String) (reg : Option String),
      bs.get o.circuit "the inner circuit" = .ok circuitB ∧ bs.get pr.pub "the inner public file" = .ok pubB ∧
      VStar.Recursive.shapeOf circuitB pubB o.partition o.program = .ok h.shape ∧
      bs.get pr.record "the inner session's record" = .ok recB ∧ String.fromUTF8? recB = some record ∧
      bs.get pr.draw "the inner statement's draw" = .ok drawB ∧
      h.src = .files
        { circuit := circuitB, pub := pubB, partition := o.partition, record := record,
          form := some { statement := VStar.Recursive.INNER_STATEMENT, os := true, tables := [], draw := some drawB } } ∧
      h.bound = (do RecSetup.metaBound (← VStar.Recursive.metaOf circuitB)) ∧
      bs.get pr.transcript "rec inner transcript" = .ok trB ∧ String.fromUTF8? trB = some t ∧
      (pr.registration.mapM fun d => do
          let some s := String.fromUTF8? (← bs.get d "rec inner registration")
            | throw "rec inner registration: not UTF-8"
          pure s : Flock.V (Option String)) = .ok reg ∧
      RecSetup.commits (some h.shape) (pure t) (pure reg) h.bound record = .ok h.commits ∧
      (match Json.parse t with
        | .ok j => RecVStar.innerLink j record
        | .error e => throw s!"rec inner transcript: not JSON: {e}") = .ok h.link := by
  unfold VStar.Recursive.innerHeld at hh
  obtain ⟨circuitB, hc, hh⟩ := bind_ok hh
  obtain ⟨pubB, hp, hh⟩ := bind_ok hh
  obtain ⟨shape, hs, hh⟩ := bind_ok hh
  obtain ⟨recB, hrB, hh⟩ := bind_ok hh
  cases hu : String.fromUTF8? recB with
  | none => rw [hu] at hh; cases hh
  | some record =>
    rw [hu] at hh
    obtain ⟨drawB, hd, hh⟩ := bind_ok hh
    obtain ⟨trB, htB, hh⟩ := bind_ok hh
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
        cases hh
        exact ⟨circuitB, pubB, recB, drawB, trB, record, t, reg, hc, hp, hs, hrB, hu, hd, rfl, rfl, htB, ht, hreg,
          hcm, by rw [hj]; exact hl⟩

/-! ## The correspondence -/

/-- **The form compiled part `s` sets up**: the statement `verity/flock-circuit` (`VStar.Recursive.STATEMENT`) under
`--zk`, its coins the keyed coin tree, one table and no lookup tables, and no `--work`, `--work-table`, `--stratified`,
`--closure` or `--public-inputs`. `Names` is to carry it for each part. -/
def PartForm (b : Raw) : Prop :=
  b.statement = VStar.Recursive.STATEMENT ∧ b.zk = true ∧ b.osCoins = false ∧ b.count = 1 ∧ b.tables = [] ∧
    b.work = none ∧ b.workTable = none ∧ b.stratified = none ∧ b.closure = none ∧ b.publicInputs = none

/-- The coins compiled part `s`'s session is set up on: the keyed coin tree of its one table. -/
def partCoins (tags0 : Tags) : Coins := .tree (Zk.coinSpec (tags0.tableNames 1))

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

/-- **Piece C: part `s`'s staged statement is read as compiled part `s`'s inputs.** Take `readX`'s reading `r` of `x`
and its part `s`, `q`; the compiled check's inner session held (`innerHeld`), and part `s`'s own inputs `pj` with what
the compiled `partVerdict` reads of them (`registered`, `rows`, `bindings`, the statement's tags `tags0`, the part's
circuit and public file). Suppose:
* `Names`' clauses: the inner statement's circuit, public file, partition and draw are the blobs' and `p.own`'s, its
  statement and form those `innerHeld` sets up (`INNER_STATEMENT`, the coin server's coins, no tables), its own inner
  shape `innerHeld`'s, the registration the blobs'; part `s`'s circuit, partition, program, registered reads and rows
  `pj`'s, the bindings `pj`'s, its form `PartForm`, and its draw, read as the verifier reads one, `draw`;
* `Shows`' clauses: part `s`'s staged public file is `pubB`, the inner session's record and the coin server's
  transcript the blobs';
* and part `s`'s statement fits its length words (`partRaw_fits` gives it from the staged files').

Then the outer verifier reads part `s`'s staged statement (`insO`) as `partIns`: compiled part `s`'s inputs. -/
theorem insO_staged_partIns {x : Claims.Msg} {r : XR} (hr : readX x = some r) {st : Stage}
    {ms : List (Claims.Msg × Claims.Out)} {s : ℕ} {q : PartX} (hq : r.w.parts[s]? = some q)
    {o : VStar.Recursive.Own} {pr : VStar.Recursive.Proofs} {bs : Core.Blobs} {h : VStar.Recursive.InnerHeld}
    (hh : VStar.Recursive.innerHeld o pr bs = .ok h)
    {pj : VStar.Recursive.PartOwn} {tags0 : Tags} (htags : Tags.find? VStar.Recursive.STATEMENT = some tags0)
    {circuitB pubB : ByteArray} {registered rows : Option (Array Registered.Port)} {bindings : VStar.Bindings}
    {draw : Option Json}
    (hregC : (pj.registered.mapM VStar.Recursive.registeredOf).map (Option.map (·.1)) = .ok registered)
    (hrowsC : pj.rows.mapM (fun j => ((do pure (← VStar.Recursive.registeredOf j).1 : Flock.V _)).mapError
      (s!"rec rows: " ++ ·)) = .ok rows)
    (hbindC : RecVStar.ownBindingsOfJson pj.bindings = .ok bindings)
    (hIc : bs.get o.circuit "the inner circuit" = .ok r.innerRaw.circuit)
    (hIp : bs.get pr.pub "the inner public file" = .ok r.innerRaw.publicFile)
    (hIstmt : r.innerRaw.statement = VStar.Recursive.INNER_STATEMENT) (hIos : r.innerRaw.osCoins = true)
    (hItab : r.innerRaw.tables = [])
    (hIpart : r.innerRaw.partition.map (jsonOf "partition") = some (.ok o.partition))
    (hIdraw : r.innerRaw.draw = (bs.get pr.draw "the inner statement's draw").toOption)
    (hIshape : (do RecVStar.Inner.ofJson (← jsonOf "rec-inner" r.w.recInner)) = .ok h.shape)
    (hIreg : (pr.registration.mapM fun d => (bs.get d "rec inner registration").toOption) = some r.w.registration)
    (hPform : PartForm q.base) (hPc : q.base.circuit = circuitB)
    (hPpart : q.base.partition.map (jsonOf "partition") = some (.ok pj.partition))
    (hPprog : q.base.program = some pj.program)
    (hPreg : q.base.registered.map (jsonOf "registered") = pj.registered.map Except.ok)
    (hProws : q.rows.map (jsonOf "rec-rows") = pj.rows.map Except.ok)
    (hPbind : jsonOf "rec-bindings" r.w.bindings = .ok pj.bindings)
    (hPdraw : (q.base.draw.mapM (jsonOf "draw")).map (Option.map Flock.norm) = .ok draw)
    (hSp : st.pub x ms s = pubB)
    (hSr : bs.get pr.record "the inner session's record" = .ok (st.record x ms))
    (hSt : bs.get pr.transcript "rec inner transcript" = .ok (st.transcript x ms))
    (hfit : RecRawFits (partRaw r st x ms s q)) :
    insO (staged st x ms s) = some (partIns tags0 circuitB pubB pj registered draw h s bindings rows) := by
  sorry

end Verity.Recursive.Part
