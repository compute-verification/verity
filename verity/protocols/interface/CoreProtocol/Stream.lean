/-
Stream: a `BlobProtocol` whose check also reads its blobs one at a time, so that blobs larger together than memory are
checked holding one of them and what the check keeps of the others.

* `Stream`: the check as a fold over the blobs: their order, named before the first by SHA-512 (`tables`, which may
  come first, each at most once and in their order, then exactly `plan`), a step on each blob and a finish;
* `Stream.feed` and `Stream.close`: that fold with each blob's SHA-512 held to its place, refusing at the first blob
  out of place, and where the plan names more blobs than came; `Stream.run` is the two on a list of blobs;
* `StreamProtocol`: a `BlobProtocol` with a stream for each statement, and `StreamProtocol.Sound`: on a feed in the
  stream's order, its finish is `checkBlobs` on the blobs fed;
* `StreamProtocol.run_eq`: on a sound protocol, `run` refuses or decides what `checkBlobs` decides on the blobs fed,
  in any order they come in, so its accept is `checkBlobs`' (`run_accept`);
* `StreamProtocol.main`: `BlobProtocol.main` with two more commands: `plan`, the blobs' SHA-512 in the order the stream
  reads them, and `check-stream`, which reads the blobs a file lists one at a time, holding one blob in memory.
-/
module

public import CoreProtocol.Blobs

/-!
# Streams: a check that reads its blobs one at a time

`Stream`, with its `feed`, `close` and `run`; `StreamProtocol`, a `BlobProtocol` with a stream for each statement, its
`Sound` and what a sound one's `run` decides (`run_eq`, `run_accept`); and `StreamProtocol.main`, with the commands
`plan` and `check-stream`.
-/

@[expose] public section

namespace Core

open Flock (V)

/-- A check that reads its blobs one at a time, in an order it names before the first, and holds of each blob only what
`step` keeps: `tables`, the SHA-512 of the blobs that may come first, each at most once and in this order; `plan`, the
SHA-512 of every other blob, in the order they are read, or a refusal. `step` takes a blob's SHA-512, which the feed
computes once, beside its bytes. -/
structure Stream (Out : Type) where
  State : Type
  tables : List ByteArray
  plan : V (Array ByteArray)
  init : State
  step : State → ByteArray → ByteArray → State
  finish : State → Out

namespace Stream

variable {Out : Type}

/-- The blobs `bs` as `s`'s feed on its plan `plan`: some of its tables first, by SHA-512 and in their order, then
exactly its plan. -/
def Fed (s : Stream Out) (plan : Array ByteArray) (bs : List ByteArray) : Prop :=
  ∃ pre rest, bs = pre ++ rest ∧ (pre.map Flock.Sha512.hash).Sublist s.tables ∧
    rest.map Flock.Sha512.hash = plan.toList

/-- Where a feed is: the stream's state, the tables that may still come, how many planned blobs came, how many blobs
were read, and the refusal once one was out of place. -/
structure At (s : Stream Out) where
  state : s.State
  tables : List ByteArray
  next : Nat
  read : Nat
  error : Option String

def start (s : Stream Out) : s.At := ⟨s.init, s.tables, 0, 0, none⟩

/-- The state after `step` on each of `bs`, in order. -/
def steps (s : Stream Out) (bs : List ByteArray) : s.State :=
  bs.foldl (fun st b => s.step st (Flock.Sha512.hash b) b) s.init

/-- The list after the first entry `h`, if any. -/
def after (h : ByteArray) : List ByteArray → Option (List ByteArray)
  | [] => none
  | t :: ts => if t = h then some ts else after h ts

theorem after_some {h : ByteArray} : ∀ {l r : List ByteArray}, after h l = some r → ∃ u, l = u ++ h :: r
  | [], _, hl => by simp [after] at hl
  | t :: ts, r, hl => by
    by_cases ht : t = h
    · simp only [after, ht, ite_true, Option.some.injEq] at hl
      exact ⟨[], by simp [ht, hl]⟩
    · simp only [after, ht, ite_false] at hl
      obtain ⟨u, hu⟩ := after_some hl
      exact ⟨t :: u, by simp [hu]⟩

def outOfPlace (read : Nat) (h : ByteArray) : String :=
  s!"blob {read}: its SHA-512 {Flock.hexOf h} is not the next one the check reads"

/-- The table `b` would be, before the first planned blob: what may still come after it. -/
def tableNext (s : Stream Out) (a : s.At) (h : ByteArray) : Option (List ByteArray) :=
  if a.next = 0 then after h a.tables else none

/-- One more blob: stepped where its SHA-512 is a table that may still come, before any planned blob, or else the next
planned blob's; a refusal otherwise, after which nothing more is stepped. -/
def feed (s : Stream Out) (plan : Array ByteArray) (a : s.At) (b : ByteArray) : s.At :=
  match a.error with
  | some _ => a
  | none =>
    let h := Flock.Sha512.hash b
    match s.tableNext a h with
    | some rest => ⟨s.step a.state h b, rest, 0, a.read + 1, none⟩
    | none =>
      if plan[a.next]? = some h then ⟨s.step a.state h b, [], a.next + 1, a.read + 1, none⟩
      else { a with read := a.read + 1, error := some (outOfPlace a.read h) }

def close (s : Stream Out) (refuse : String → Out) (plan : Array ByteArray) (a : s.At) : Out :=
  match a.error with
  | some e => refuse e
  | none =>
    if a.next < plan.size then refuse s!"the check reads {plan.size - a.next} more blobs than came"
    else s.finish a.state

/-- What `s` decides on `bs`, one at a time: its plan's refusal; a refusal at the first blob whose SHA-512 is not in its
place, or where the plan names more blobs than came; else `finish` after `step` on each. -/
def run (s : Stream Out) (refuse : String → Out) (bs : List ByteArray) : Out :=
  match s.plan with
  | .error e => refuse e
  | .ok plan => s.close refuse plan (bs.foldl (s.feed plan) s.start)

/-- What a feed without a refusal holds: its state is the steps on every blob read, and those blobs are some tables
in order, then the plan's first `next`. -/
def Inv (s : Stream Out) (plan : Array ByteArray) (past : List ByteArray) (a : s.At) : Prop :=
  a.error = none → a.state = s.steps past ∧
    ∃ pre rest used, past = pre ++ rest ∧ (pre.map Flock.Sha512.hash).Sublist used ∧ s.tables = used ++ a.tables ∧
      rest.map Flock.Sha512.hash = plan.toList.take a.next

theorem inv_start (s : Stream Out) (plan : Array ByteArray) : s.Inv plan [] s.start := fun _ =>
  ⟨rfl, [], [], [], by simp, List.Sublist.slnil, by simp [start], by simp [start]⟩

theorem inv_feed (s : Stream Out) (plan : Array ByteArray) {past : List ByteArray} {a : s.At} (b : ByteArray)
    (ha : s.Inv plan past a) : s.Inv plan (past ++ [b]) (s.feed plan a b) := by
  intro hn
  unfold feed at hn ⊢
  match he : a.error with
  | some _ => simp [he] at hn
  | none =>
    obtain ⟨hst, pre, rest, used, hp, hsub, ht, hr⟩ := ha he
    simp only [he] at hn ⊢
    match hn' : s.tableNext a (Flock.Sha512.hash b) with
    | some r =>
      simp only [hn'] at hn ⊢
      unfold tableNext at hn'
      by_cases h0 : a.next = 0
      · simp only [h0, ite_true] at hn'
        obtain ⟨u, hu⟩ := after_some hn'
        have hrest : rest = [] := by
          rw [h0, List.take_zero] at hr
          exact List.map_eq_nil_iff.mp hr
        refine ⟨by simp [hst, steps], pre ++ [b], [], used ++ u ++ [Flock.Sha512.hash b], ?_, ?_, ?_, by simp⟩
        · simp [hp, hrest]
        · simpa using (hsub.trans (List.sublist_append_left used u)).append (List.Sublist.refl _)
        · simp [ht, hu]
      · simp [h0] at hn'
    | none =>
      simp only [hn'] at hn ⊢
      by_cases hplan : plan[a.next]? = some (Flock.Sha512.hash b)
      · simp only [hplan, ite_true] at hn ⊢
        refine ⟨by simp [hst, steps], pre, rest ++ [b], used ++ a.tables, by simp [hp], ?_, by simp [ht], ?_⟩
        · exact hsub.trans (List.sublist_append_left used a.tables)
        · rw [List.map_append, hr, List.take_add_one, Array.getElem?_toList, hplan]
          rfl
      · simp [hplan] at hn

theorem inv_fold (s : Stream Out) (plan : Array ByteArray) :
    ∀ (bs past : List ByteArray) (a : s.At), s.Inv plan past a →
      s.Inv plan (past ++ bs) (bs.foldl (s.feed plan) a)
  | [], past, a, h => by simpa using h
  | b :: bs, past, a, h => by
    have := inv_fold s plan bs (past ++ [b]) (s.feed plan a b) (s.inv_feed plan b h)
    simpa using this

/-- `run` refuses, or its plan is a plan, `bs` is fed in its order and `run` is the finish after every step. -/
theorem run_fed (s : Stream Out) (refuse : String → Out) (bs : List ByteArray) :
    (∃ r, s.run refuse bs = refuse r) ∨
      ∃ plan, s.plan = .ok plan ∧ s.Fed plan bs ∧ s.run refuse bs = s.finish (s.steps bs) := by
  unfold run
  match hp : s.plan with
  | .error e => exact .inl ⟨e, rfl⟩
  | .ok plan =>
    simp only
    have hi := s.inv_fold plan bs [] s.start (s.inv_start plan)
    simp only [List.nil_append] at hi
    unfold close
    match he : (bs.foldl (s.feed plan) s.start).error with
    | some e => exact .inl ⟨e, by simp⟩
    | none =>
      simp only
      by_cases hlt : (bs.foldl (s.feed plan) s.start).next < plan.size
      · exact .inl ⟨s!"the check reads {plan.size - (bs.foldl (s.feed plan) s.start).next} more blobs than came",
          by simp [hlt]⟩
      · obtain ⟨hst, pre, rest, used, hb, hsub, ht, hr⟩ := hi he
        rw [List.take_of_length_le (by simpa using Nat.le_of_not_lt hlt)] at hr
        exact .inr ⟨plan, rfl, ⟨pre, rest, hb, ht ▸ hsub.trans (List.sublist_append_left used _), hr⟩,
          by simp [hlt, hst]⟩

/-- The blobs the files `paths` hold, as `run` reads them, holding one blob at a time; the files after a refusal are
not read. This loop and the files' reading are the trusted code, as `BlobProtocol.main`'s reading of its blobs is. -/
def files (s : Stream Out) (refuse : String → Out) (paths : List String) : IO Out := do
  match s.plan with
  | .error e => pure (refuse e)
  | .ok plan =>
    let mut a := s.start
    for f in paths do
      if a.error.isSome then break
      a := s.feed plan a (← IO.FS.readBinFile f)
    pure (s.close refuse plan a)

/-- The stream that keeps every blob whole and decides `f` on them: for a check whose blobs fit in memory. -/
def whole (tables : List ByteArray) (plan : V (Array ByteArray)) (f : Blobs → Out) : Stream Out :=
  { State := Array (ByteArray × ByteArray), tables, plan, init := #[], step := fun S h b => S.push (h, b)
    finish := fun S => f ⟨S⟩ }

theorem whole_fold : ∀ (bs : List ByteArray) (S : Array (ByteArray × ByteArray)),
    (bs.foldl (fun (S : Array (ByteArray × ByteArray)) b => S.push (Flock.Sha512.hash b, b)) S).toList =
      S.toList ++ bs.map fun b => (Flock.Sha512.hash b, b)
  | [], S => by simp
  | b :: bs, S => by
    rw [List.foldl_cons, whole_fold bs]
    simp

/-- **The stream that keeps every blob decides `f` on the blobs fed**, in any order. -/
theorem whole_finish (tables : List ByteArray) (plan : V (Array ByteArray)) (f : Blobs → Out) (bs : List ByteArray) :
    (whole tables plan f).finish ((whole tables plan f).steps bs) = f (Blobs.ofArray bs.toArray) := by
  show f ⟨bs.foldl (fun (S : Array (ByteArray × ByteArray)) b => S.push (Flock.Sha512.hash b, b)) #[]⟩ = _
  unfold Blobs.ofArray
  congr 2
  apply Array.toList_inj.mp
  rw [whole_fold]
  simp

end Stream

structure StreamProtocol extends BlobProtocol where
  stream : Params → Commitment → Evidence → Stream (Outcome Public Profile)

/-- On a feed in its stream's order, the stream's finish after every step is `checkBlobs` on the blobs fed, a
rejection's reason included. -/
def StreamProtocol.Sound (P : StreamProtocol) : Prop :=
  ∀ p c e plan bs, (P.stream p c e).plan = .ok plan → (P.stream p c e).Fed plan bs →
    (P.stream p c e).finish ((P.stream p c e).steps bs) =
      P.checkBlobs p c e (Blobs.ofArray bs.toArray)

/-- **The streamed check refuses, or decides what `checkBlobs` decides on the blobs fed**, in any order they come. -/
theorem StreamProtocol.run_eq (P : StreamProtocol) (hP : P.Sound) (p : P.Params) (c : P.Commitment) (e : P.Evidence)
    (bs : List ByteArray) :
    (∃ r, (P.stream p c e).run .reject bs = .reject r) ∨
      (P.stream p c e).run .reject bs = P.checkBlobs p c e (Blobs.ofArray bs.toArray) := by
  rcases (P.stream p c e).run_fed .reject bs with h | ⟨plan, hp, hf, h⟩
  · exact .inl h
  · exact .inr (h.trans (hP p c e plan bs hp hf))

theorem StreamProtocol.run_accept (P : StreamProtocol) (hP : P.Sound) {p : P.Params} {c : P.Commitment}
    {e : P.Evidence} {bs : List ByteArray} {pub : P.Public} {prof : P.Profile}
    (h : (P.stream p c e).run .reject bs = .accept pub prof) :
    P.checkBlobs p c e (Blobs.ofArray bs.toArray) = .accept pub prof := by
  rcases P.run_eq hP p c e bs with ⟨r, hr⟩ | hr
  · rw [hr] at h; cases h
  · rw [← hr, h]

/-- The parameters, commitment and evidence decoded as `Protocol.answer` decodes them, with its refusals. -/
def StreamProtocol.decode (P : StreamProtocol) (params commitment evidence : ByteArray) :
    V (P.Params × P.Commitment × P.Evidence) := do
  let p ← (canonicalJson params >>= P.params).mapError (s!"params: {·}")
  let c ← (P.commitment commitment).mapError (s!"commitment: {·}")
  let e ← (P.evidence evidence).mapError (s!"evidence: {·}")
  pure (p, c, e)

/-- `BlobProtocol.main` with `plan PARAMS COMMITMENT EVIDENCE`, which prints the stream's tables (`table HEX`) and plan
(`blob HEX`), one SHA-512 a line, and `check-stream PARAMS COMMITMENT EVIDENCE LIST`, which reads the blobs in the
files LIST names, one path a line, as `Stream.files` reads them, and answers as `check` does. -/
def StreamProtocol.main (P : StreamProtocol) (args : List String) : IO UInt32 := do
  match args with
  | ["plan", params, commitment, evidence] =>
    match P.decode (← IO.FS.readBinFile params) (← IO.FS.readBinFile commitment) (← IO.FS.readBinFile evidence) with
    | .error e => IO.eprintln s!"{P.id}: {e}"; return 1
    | .ok (p, c, e) =>
      let s := P.stream p c e
      match s.plan with
      | .error why => IO.eprintln s!"{P.id}: {why}"; return 1
      | .ok plan =>
        let out ← IO.getStdout
        for t in s.tables do out.putStrLn s!"table {Flock.hexOf t}"
        for d in plan do out.putStrLn s!"blob {Flock.hexOf d}"
        out.flush
        return 0
  | ["check-stream", params, commitment, evidence, list] =>
    let (pb, cb, eb) := (← IO.FS.readBinFile params, ← IO.FS.readBinFile commitment, ← IO.FS.readBinFile evidence)
    let outcome ← match P.decode pb cb eb with
      | .error e => pure (Outcome.reject e)
      | .ok (p, c, e) =>
        let paths := ((← IO.FS.readFile list).splitOn "\n").filter (· != "")
        (P.stream p c e).files .reject paths
    let a := ({ P.toProtocol with check := fun _ _ _ => outcome } : Protocol).answer pb cb eb
    match a.canonical with
    | .ok line =>
      let out ← IO.getStdout
      out.putStrLn line
      out.flush
      return (match a.verdict with | .accept => 0 | .reject => 1 | .undecided => 2)
    | .error e =>
      IO.eprintln s!"{P.id}: the outcome is not canonical JSON: {e}"
      return 70
  | _ => P.toBlobProtocol.main args

end Core
