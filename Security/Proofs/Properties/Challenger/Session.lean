module

public import Proofs.Properties.Challenger.Break
public import Proofs.Properties.Challenger.Commitment
public import Proofs.Properties.Pouw.LiveSalt.Ledger
public import Proofs.Properties.Pouw.LiveSalt.Coin
import all Init.Data.String.Defs

/-!
# A session `sessionOf` accepts is a run the device closed, or a break

`sessionOf` accepts a session's files only when its close's LMS signature verifies under the verifier's key over the
close message, which hashes the record's message (itself over the host and the commitment's tagged digest), the
session's id, the log's chain and how the session ended, and only when the release log it read chains to the close's
chain and the session's client closed it. Each of those strings is one the finder hashes (`readHashed`), and so is each
the device hashes as it runs (`deviceHashed`). So the signature is the device's, or the finder outputs a forgery
(`signed_or_break`), and the message it signs is a run's close, since the record's and the ledger's begin with roles of
other lengths (`closeBytes_ne_record`, `closeBytes_ne_ledger`). Equal digests are equal strings, or a collision of
two the finder hashes (`eq_or_break`). A close's string names its record's message, its session's id, its chain and its
end (`closeBytes_inj`), a record's its host and its commitment's tagged digest (`recordBytes_inj`), and a chain its
lines (`chain_bind`), each step's string being the previous digest, 64 bytes, then the line's field.
A commitment's identity bytes name it (`Commitment.canon_inj`). So `sessionOf_run` concludes that the
commitment, the roots and the session id `sessionOf` returns are those of a run of the device under the key whose client
closed it, read from that run's own release log, which is the session's log, or the finder outputs a break
(`sessionOf_bound` without the log).

`sessionOf`'s, `sessionNamed`'s and `drawCoins`' facts are Hoare triples proved by `mvcgen` (`sessionOf_ok`,
`sessionNamed_ok`, `drawCoins_ok`), read back at an `.ok` result by `of_triple`.
-/

@[expose] public section

set_option mvcgen.warning false

namespace Properties.Challenger.Session

open Lean (Json)
open Std.Do
open Spot Spot.Challenger
open Properties.NetworkCertifier.Bytes (bs bs_append bs_length bs_inj split_bytes hash_size be64_size be64_size_inj)
open Properties.Pouw.LiveSalt.Coin (enc enc_length enc_split enc_inj taggedBytes tagged_ok taggedBytes_inj)
open Properties.Pouw.LiveSalt.Ledger (of_triple of_triple_opt throw_spec ok_spec bind_ok except_spec none_spec
  failure_spec toOption_spec ledgerTag_spec nat_spec some_pure_spec recordOfJson_spec mapM_spec)

local instance lawfulBEqByteArraySession : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := by show (a.data == a.data) = true; exact beq_self_eq_true _

variable {dev : History} {key : ChallengerKey} {label : String} {coins : Json}

/-! ## The messages' strings -/

theorem field_bs (b : ByteArray) : bs (field b) = enc b := bs_append _ _

theorem field_size (b : ByteArray) : (field b).size = 8 + b.size := by
  rw [← bs_length, field_bs, enc_length]

theorem close_role_size : CLOSE_ROLE.toUTF8.size = 26 := by decide
theorem record_role_size : RECORD_ROLE.toUTF8.size = 27 := by decide
theorem ledger_role_size : LEDGER_ROLE.toUTF8.size = 35 := by decide
theorem log_tag_size : LOG_TAG.toUTF8.size = 32 := by decide

/-- How a close says its session ended: `0x00` when its client closed it, else `0x01` and why not. -/
def finOf : Option String → ByteArray
  | none => ByteArray.mk #[0]
  | some e => ByteArray.mk #[1] ++ e.toUTF8

/-- What the close of session `session` hashes, over its record's message `r`. -/
def closeBytes (I r : ByteArray) (session : String) (lines : Nat) (chain : ByteArray) (ended : Option String) :
    ByteArray :=
  field CLOSE_ROLE.toUTF8 ++ I ++ field r ++ field session.toUTF8 ++ field (Flock.be64 lines.toUInt64) ++
    field chain ++ field (finOf ended)

/-- What the record made on host `host` hashes, over its commitment's tagged digest `d`. -/
def recordBytes (I : ByteArray) (host : String) (d : ByteArray) : ByteArray :=
  field RECORD_ROLE.toUTF8 ++ I ++ field (field host.toUTF8) ++ field d

/-- What a ledger entry's message hashes, over the entry's tagged digest `d`. -/
def ledgerBytes (I d : ByteArray) : ByteArray := field LEDGER_ROLE.toUTF8 ++ I ++ field d

theorem closeBytes_bs (I r : ByteArray) (session : String) (lines : Nat) (chain : ByteArray) (ended : Option String) :
    bs (closeBytes I r session lines chain ended) = enc CLOSE_ROLE.toUTF8 ++ (bs I ++ (enc r ++
      (enc session.toUTF8 ++ (enc (Flock.be64 lines.toUInt64) ++ (enc chain ++ enc (finOf ended)))))) := by
  simp only [closeBytes, bs_append, field_bs, List.append_assoc]

theorem recordBytes_bs (I : ByteArray) (host : String) (d : ByteArray) :
    bs (recordBytes I host d) = enc RECORD_ROLE.toUTF8 ++ (bs I ++ (enc (field host.toUTF8) ++ enc d)) := by
  simp only [recordBytes, bs_append, field_bs, List.append_assoc]

theorem ledgerBytes_bs (I d : ByteArray) : bs (ledgerBytes I d) = enc LEDGER_ROLE.toUTF8 ++ (bs I ++ enc d) := by
  simp only [ledgerBytes, bs_append, field_bs, List.append_assoc]

/-- Two strings whose first fields have different sizes under `2^64` differ. -/
theorem enc_ne {a a' : ByteArray} {r r' : List UInt8} (ha : a.size < 2 ^ 64) (ha' : a'.size < 2 ^ 64)
    (hne : a.size ≠ a'.size) : enc a ++ r ≠ enc a' ++ r' := by
  intro h
  unfold enc at h
  simp only [List.append_assoc] at h
  exact hne (be64_size_inj ha ha' (bs_inj (List.append_inj h (by rw [bs_length, bs_length, be64_size, be64_size])).1))

theorem closeBytes_ne_record (I r : ByteArray) (session : String) (lines : Nat) (chain : ByteArray)
    (ended : Option String) (I' : ByteArray) (host : String) (d : ByteArray) :
    closeBytes I r session lines chain ended ≠ recordBytes I' host d := by
  intro h
  have hb := congrArg bs h
  rw [closeBytes_bs, recordBytes_bs] at hb
  exact enc_ne (by rw [close_role_size]; decide) (by rw [record_role_size]; decide)
    (by rw [close_role_size, record_role_size]; decide) hb

theorem closeBytes_ne_ledger (I r : ByteArray) (session : String) (lines : Nat) (chain : ByteArray)
    (ended : Option String) (I' d : ByteArray) : closeBytes I r session lines chain ended ≠ ledgerBytes I' d := by
  intro h
  have hb := congrArg bs h
  rw [closeBytes_bs, ledgerBytes_bs] at hb
  exact enc_ne (by rw [close_role_size]; decide) (by rw [ledger_role_size]; decide)
    (by rw [close_role_size, ledger_role_size]; decide) hb

theorem finOf_none_inj {e : Option String} (h : finOf none = finOf e) : e = none := by
  cases e with
  | none => rfl
  | some e =>
    exfalso
    have hb := congrArg bs h
    rw [show finOf (some e) = ByteArray.mk #[1] ++ e.toUTF8 from rfl, bs_append] at hb
    have h01 : (0 : UInt8) = 1 := (List.cons.inj hb).1
    exact absurd h01 (by decide)

/-- **A close's string names its record's message, its session, its chain and its end**, against a close of a session
its client closed, whose session id is at most 512 bytes. -/
theorem closeBytes_inj {I r r' : ByteArray} {s s' : String} {n n' : Nat} {ch ch' : ByteArray} {e' : Option String}
    (hr : r.size = 64) (hr' : r'.size = 64) (hs : s.toUTF8.size ≤ 512) (hch : ch.size = 64) (hch' : ch'.size = 64)
    (h : closeBytes I r s n ch none = closeBytes I r' s' n' ch' e') :
    r = r' ∧ s = s' ∧ ch = ch' ∧ e' = none := by
  have hb := congrArg bs h
  rw [closeBytes_bs, closeBytes_bs] at hb
  have hb2 := List.append_cancel_left (List.append_cancel_left hb)
  obtain ⟨rfl, hb3⟩ := enc_split hb2 (hr.trans hr'.symm)
  have hlen := congrArg List.length hb3
  have hfin : (finOf none).size = 1 := rfl
  simp only [List.length_append, enc_length, be64_size, hch, hch', hfin] at hlen
  have hs' : s'.toUTF8.size < 2 ^ 64 := by omega
  have hss : s.toUTF8.size = s'.toUTF8.size := by
    have e := hb3
    unfold enc at e
    simp only [List.append_assoc] at e
    exact be64_size_inj (by omega) hs'
      (bs_inj (List.append_inj e (by rw [bs_length, bs_length, be64_size, be64_size])).1)
  obtain ⟨hs1, hb4⟩ := enc_split hb3 hss
  obtain ⟨-, hb5⟩ := enc_split hb4 (by rw [be64_size, be64_size])
  obtain ⟨rfl, hb6⟩ := enc_split hb5 (hch.trans hch'.symm)
  exact ⟨rfl, String.toByteArray_inj.1 hs1, rfl, finOf_none_inj (enc_inj hb6)⟩

/-- **A record's string names its host and its commitment's tagged digest.** -/
theorem recordBytes_inj {I : ByteArray} {h h' : String} {d d' : ByteArray} (hd : d.size = 64) (hd' : d'.size = 64)
    (e : recordBytes I h d = recordBytes I h' d') : h = h' ∧ d = d' := by
  have hb := congrArg bs e
  rw [recordBytes_bs, recordBytes_bs] at hb
  have hb2 := List.append_cancel_left (List.append_cancel_left hb)
  have hlen := congrArg List.length hb2
  simp only [List.length_append, enc_length, field_size, hd, hd'] at hlen
  obtain ⟨h1, h2⟩ := enc_split hb2 (by rw [field_size, field_size]; omega)
  have h3 := congrArg bs h1
  rw [field_bs, field_bs] at h3
  exact ⟨String.toByteArray_inj.1 (enc_inj h3), enc_inj h2⟩

/-! ## The release log's chain -/

/-- One step of the release log's chain over line `l`: the digest after it, and the strings hashed so far. -/
def stepF (acc : ByteArray × List ByteArray) (l : String) : ByteArray × List ByteArray :=
  (Flock.Sha512.hash (acc.1 ++ field l.toUTF8), acc.2 ++ [acc.1 ++ field l.toUTF8])

/-- What the chain over `lines` hashes after its start. -/
def chainSteps (lines : List String) : List ByteArray :=
  (lines.foldl stepF (Flock.Sha512.hash (field LOG_TAG.toUTF8), [])).2

theorem logHashed_eq (lines : List String) : logHashed lines =
    [field LOG_TAG.toUTF8] ++ chainSteps lines ++ [(String.join (lines.map (· ++ "\n"))).toUTF8] := rfl

theorem stepF_fst (acc : ByteArray × List ByteArray) (x : String) :
    (stepF acc x).1 = Flock.Sha512.hash (acc.1 ++ field x.toUTF8) := rfl

theorem foldl_fst (l : List String) (acc : ByteArray × List ByteArray) :
    (l.foldl stepF acc).1 = l.foldl (fun h x => Flock.Sha512.hash (h ++ field x.toUTF8)) acc.1 := by
  induction l generalizing acc with
  | nil => rfl
  | cons x l ih => rw [List.foldl_cons, List.foldl_cons, ih, stepF_fst]

theorem logChain_eq (l : List String) :
    logChain l = (l.foldl stepF (Flock.Sha512.hash (field LOG_TAG.toUTF8), [])).1 := by
  rw [foldl_fst]
  rfl

theorem steps_concat (l : List String) (x : String) :
    chainSteps (l ++ [x]) = chainSteps l ++ [logChain l ++ field x.toUTF8] := by
  unfold chainSteps
  rw [List.foldl_append, logChain_eq]
  rfl

theorem logChain_concat (l : List String) (x : String) :
    logChain (l ++ [x]) = Flock.Sha512.hash (logChain l ++ field x.toUTF8) := by
  unfold logChain
  rw [List.foldl_append]
  rfl

theorem nil_or_snoc (l : List String) : l = [] ∨ ∃ l0 x, l = l0 ++ [x] := by
  rcases List.eq_nil_or_concat l with h | ⟨l0, x, h⟩
  · exact .inl h
  · exact .inr ⟨l0, x, by rw [h, List.concat_eq_append]⟩

theorem logChain_size (l : List String) : (logChain l).size = 64 := by
  rcases nil_or_snoc l with rfl | ⟨l0, x, rfl⟩
  · exact hash_size _
  · rw [logChain_concat]
    exact hash_size _

theorem start_ne_step (l : List String) (x : String) : field LOG_TAG.toUTF8 ≠ logChain l ++ field x.toUTF8 := by
  intro h
  have := congrArg ByteArray.size h
  rw [ByteArray.size_append, field_size, field_size, logChain_size, log_tag_size] at this
  omega

theorem mem_steps_concat {l : List String} {x : String} {y : ByteArray} (hy : y ∈ chainSteps l) :
    y ∈ chainSteps (l ++ [x]) := by
  rw [steps_concat]
  exact List.mem_append_left _ hy

theorem last_mem_steps (l : List String) (x : String) : logChain l ++ field x.toUTF8 ∈ chainSteps (l ++ [x]) := by
  rw [steps_concat]
  exact List.mem_append_right _ (List.mem_singleton_self _)

/-- **The chain binds its lines**: two release logs with one chain are one, or the finder outputs a break, where every
string either chain hashes is one it hashes. -/
theorem chain_bind (hU : field LOG_TAG.toUTF8 ∈ readHashed key label coins ++ deviceHashed dev key.pub) :
    ∀ (n : Nat) (l l' : List String), l.length + l'.length ≤ n →
      (∀ y ∈ chainSteps l, y ∈ readHashed key label coins ++ deviceHashed dev key.pub) →
      (∀ y ∈ chainSteps l', y ∈ readHashed key label coins ++ deviceHashed dev key.pub) →
      logChain l = logChain l' → l = l' ∨ Break dev key label coins := by
  intro n
  induction n with
  | zero =>
    intro l l' hn _ _ _
    left
    rw [List.length_eq_zero_iff.1 (by omega : l.length = 0), List.length_eq_zero_iff.1 (by omega : l'.length = 0)]
  | succ n ih =>
    intro l l' hn hl hl' h
    rcases nil_or_snoc l with rfl | ⟨l0, x, rfl⟩ <;> rcases nil_or_snoc l' with rfl | ⟨l0', x', rfl⟩
    · exact .inl rfl
    · rw [logChain_concat] at h
      rcases eq_or_break hU (hl' _ (last_mem_steps l0' x')) h with e | b
      · exact absurd e (start_ne_step l0' x')
      · exact .inr b
    · rw [logChain_concat] at h
      rcases eq_or_break (hl _ (last_mem_steps l0 x)) hU h with e | b
      · exact absurd e.symm (start_ne_step l0 x)
      · exact .inr b
    · rw [logChain_concat, logChain_concat] at h
      rcases eq_or_break (hl _ (last_mem_steps l0 x)) (hl' _ (last_mem_steps l0' x')) h with e | b
      · have hb := congrArg bs e
        rw [bs_append, bs_append, field_bs, field_bs] at hb
        obtain ⟨h1, h2⟩ := split_bytes hb (by rw [logChain_size, logChain_size])
        obtain rfl : x = x' := String.toByteArray_inj.1 (enc_inj h2)
        simp only [List.length_append, List.length_singleton] at hn
        rcases ih l0 l0' (by omega) (fun y hy => hl y (mem_steps_concat hy))
            (fun y hy => hl' y (mem_steps_concat hy)) h1 with e | b
        · exact .inl (by rw [e])
        · exact .inr b
      · exact .inr b

/-! ## What `sessionOf` read when it accepts -/

theorem mapM_str {f : Json → Flock.V String} (hf : ∀ j s, f j = .ok s → j = .str s) :
    ∀ {l : List Json} {xs : List String}, l.mapM f = .ok xs → l = xs.map Json.str
  | [], xs, h => by
    rw [List.mapM_nil] at h
    obtain rfl := Except.ok.inj h
    rfl
  | j :: l, xs, h => by
    rw [List.mapM_cons] at h
    obtain ⟨s, hs, h⟩ := bind_ok.1 h
    obtain ⟨ss, hss, h⟩ := bind_ok.1 h
    obtain rfl := Except.ok.inj h
    rw [List.map_cons, hf j s hs, mapM_str hf hss]

theorem mapM_str_of {f : Json → Flock.V String} {l : List Json} {xs : List String} (h : l.mapM f = .ok xs)
    (hf : ∀ j s, f j = .ok s → j = .str s) : l = xs.map Json.str :=
  mapM_str hf h

theorem none_of_forall {α : Type} {o : Option α} (h : ∀ x, o = some x → False) : o = none := by
  cases o with
  | none => rfl
  | some x => exact absurd rfl (h x)

theorem lms_of_check {key : ChallengerKey} {m sig : ByteArray}
    (h : (match Lms.Signature.decode key.params sig with
      | .ok s => (Lms.verify key.params key.pub m s).toOption == some true
      | .error _ => false) = true) : LmsVerifies key m sig := by
  unfold LmsVerifies lmsVerifies
  revert h
  cases Lms.Signature.decode key.params sig with
  | error e => intro h; simp at h
  | ok s =>
    intro h
    have h' : ((Lms.verify key.params key.pub m s).toOption == some true) = true := h
    show (match Lms.verify key.params key.pub m s with | .ok b => b | .error _ => false) = true
    revert h'
    cases Lms.verify key.params key.pub m s with
    | error e => intro h'; simp [Except.toOption] at h'
    | ok b => intro h'; simpa [Except.toOption] using h'

theorem recordOfJson_ok_spec (j : Json) :
    ⦃⌜True⌝⦄ ChallengerRecord.ofJson j ⦃⇓? a => ⌜ChallengerRecord.ofJson j = .ok a⌝⦄ := ok_spec _

theorem recordReason_spec (key : ChallengerKey) (record : Json) :
    ⦃⌜True⌝⦄ recordReason key record ⦃⇓? _ => ⌜True⌝⦄ := except_spec _

theorem closeMessage_ok_spec (I : ByteArray) (host : String) (c : LiveCommitment) (session : String) (lines : Nat)
    (chain : ByteArray) (ended : Option String) :
    ⦃⌜True⌝⦄ closeMessage I host c session lines chain ended
      ⦃⇓? a => ⌜closeMessage I host c session lines chain ended = .ok a⌝⦄ := ok_spec _

theorem mapM_ok_spec {α β : Type} (f : α → Flock.V β) (l : List α) :
    ⦃⌜True⌝⦄ l.mapM f ⦃⇓? a => ⌜l.mapM f = .ok a⌝⦄ := ok_spec _

theorem releasedRootsOf_ok_spec (c : LiveCommitment) (log : List String) :
    ⦃⌜True⌝⦄ releasedRootsOf c log ⦃⇓? a => ⌜releasedRootsOf c log = .ok a⌝⦄ := ok_spec _

-- `mvcgen` splits `sessionOf` into one goal per check it makes of the record, the close and the log
set_option maxHeartbeats 4000000 in
/-- What `sessionOf` read when it accepts: its record, its close and the close's message, whose signature verifies under
the key, its log's lines, which chain to the close's chain, a session its client closed, and the roots its log shows. -/
theorem sessionOf_read {key : ChallengerKey} {record log close : Json} {res : LiveCommitment × Roots × String}
    (h : sessionOf key record log close = .ok res) :
    ∃ (r : ChallengerRecord) (c : ChallengerClose) (m : ByteArray) (ls : Array Json) (lines : List String),
      ChallengerRecord.ofJson record = .ok r ∧ ChallengerClose.ofJson close = .ok c ∧
      closeMessage (key.pub.extract 0 16) r.host r.commitment c.session c.lines c.chain c.ended = .ok m ∧
      LmsVerifies key m c.signature ∧ log = .arr ls ∧ ls.toList = lines.map Json.str ∧
      logChain lines = c.chain ∧ c.ended = none ∧ releasedRootsOf r.commitment lines = .ok res.2.1 ∧
      res = (r.commitment, res.2.1, c.session) := by
  refine of_triple (Q := fun res => ∃ (r : ChallengerRecord) (c : ChallengerClose) (m : ByteArray) (ls : Array Json)
      (lines : List String),
      ChallengerRecord.ofJson record = .ok r ∧ ChallengerClose.ofJson close = .ok c ∧
      closeMessage (key.pub.extract 0 16) r.host r.commitment c.session c.lines c.chain c.ended = .ok m ∧
      LmsVerifies key m c.signature ∧ log = .arr ls ∧ ls.toList = lines.map Json.str ∧
      logChain lines = c.chain ∧ c.ended = none ∧ releasedRootsOf r.commitment lines = .ok res.2.1 ∧
      res = (r.commitment, res.2.1, c.session)) ?_ h
  clear h
  mvcgen [sessionOf, -Spec.throw_Except, throw_spec, recordReason_spec, recordOfJson_ok_spec, closeMessage_ok_spec,
    mapM_ok_spec, releasedRootsOf_ok_spec]
  all_goals
    refine ⟨_, _, _, _, _, by assumption, by assumption, by assumption, lms_of_check (by assumption),
      by first | rfl | assumption, mapM_str_of (by assumption) (fun j s hj => by cases j <;> cases hj <;> rfl),
      beq_iff_eq.1 (by assumption), by first | assumption | exact none_of_forall (by assumption) |
        exact none_of_forall (by subst_vars; assumption),
      by assumption, rfl⟩

theorem sessionOf_ok_spec (key : ChallengerKey) (record log close : Json) :
    ⦃⌜True⌝⦄ sessionOf key record log close ⦃⇓? a => ⌜sessionOf key record log close = .ok a⌝⦄ := ok_spec _

-- `mvcgen` splits `sessionNamed` into one goal per check it makes of the session's record and files
set_option maxHeartbeats 1000000 in
/-- `sessionNamed` returns what `sessionOf` returns of the session's files. -/
theorem sessionNamed_ok {a : Audited} {key : ChallengerKey} {session : Json} {r : LiveCommitment × Roots}
    (h : sessionNamed a key session = .ok r) :
    ∃ sid, sessionOf key (getD session "record") (getD session "log") (getD session "close") = .ok (r.1, r.2, sid) := by
  refine of_triple (Q := fun r => ∃ sid,
    sessionOf key (getD session "record") (getD session "log") (getD session "close") = .ok (r.1, r.2, sid)) ?_ h
  clear h
  mvcgen [sessionNamed, -Spec.throw_Except, throw_spec, sessionOf_ok_spec, recordOfJson_spec, mapM_spec]
  all_goals exact ⟨_, by assumption⟩

/-- `trustedSessionOf` returns what `sessionNamed` returns of the session the ledger pins. -/
theorem trustedSessionOf_ok {ledger window : Json} {key : ChallengerKey} {pins main session : Json}
    {r : LiveCommitment × Roots} (h : trustedSessionOf ledger window key pins main session = .ok r) :
    ∃ a, pinnedSession ledger window key pins main = .ok a ∧ sessionNamed a key session = .ok r := by
  unfold trustedSessionOf at h
  exact bind_ok.1 h

theorem trustedSessionOf_ok_spec (ledger window : Json) (key : ChallengerKey) (pins main session : Json) :
    ⦃⌜True⌝⦄ trustedSessionOf ledger window key pins main session
      ⦃⇓? a => ⌜trustedSessionOf ledger window key pins main session = .ok a⌝⦄ := ok_spec _

theorem challengerStream_ok_spec (c : LiveCommitment) (segment : Json) (after : ByteArray)
    (roots : Option (List ByteArray)) :
    ⦃⌜True⌝⦄ challengerStream c segment after roots
      ⦃⇓? a => ⌜challengerStream c segment after roots = .ok a⌝⦄ := ok_spec _

-- `mvcgen` splits `drawCoins` into one goal per check it makes of the record's form, the window and the family
set_option maxHeartbeats 1000000 in
/-- What `drawCoins` read when it accepts: the session the ledger opens on the window, and the segment's coins, read
against its commitment and the roots its log shows received for the family. -/
theorem drawCoins_ok {key : ChallengerKey} {w label : String} {after : ByteArray} {coins : Json} {ξ : ByteArray}
    (h : drawCoins key w label after coins = .ok ξ) :
    ∃ c roots, trustedSessionOf (getD coins "ledger") (Json.mkObj [("root", Json.str w)]) key (getD coins "pins")
        (getD coins "main") (getD coins "session") = .ok (c, roots) ∧
      challengerStream c (getD coins "segment") after
        (some (((roots.find? (·.1 == label)).map (·.2)).getD [])) = .ok ξ := by
  refine of_triple (Q := fun ξ => ∃ c roots, trustedSessionOf (getD coins "ledger") (Json.mkObj [("root", Json.str w)])
      key (getD coins "pins") (getD coins "main") (getD coins "session") = .ok (c, roots) ∧
      challengerStream c (getD coins "segment") after
        (some (((roots.find? (·.1 == label)).map (·.2)).getD [])) = .ok ξ) ?_ h
  clear h
  mvcgen [drawCoins, -Spec.throw_Except, throw_spec, trustedSessionOf_ok_spec, challengerStream_ok_spec]
  all_goals first
    | exact ⟨_, _, by assumption, by assumption⟩
    | (intro h; exact ⟨_, _, by assumption, h⟩)
    | (intro h; subst_vars; exact ⟨_, _, by assumption, h⟩)

/-! ## What the ledger's entries sign -/

theorem toOption_some {ε α : Type} {x : Except ε α} {a : α} (h : x.toOption = some a) : x = .ok a := by
  cases x <;> simp_all [Except.toOption]

-- `mvcgen` splits `Core.Record.ofJson` into one goal per field it parses
set_option maxHeartbeats 4000000 in
/-- A record entry is an object, and its schema is the one it names. -/
theorem recordOfJson_schema {j : Json} {r : Core.Record} (h : Core.Record.ofJson j = .ok r) :
    (∃ kvs, j = .obj kvs) ∧ getD j "schema" = .str r.schema := by
  refine of_triple (Q := fun r => (∃ kvs, j = .obj kvs) ∧ getD j "schema" = .str r.schema) ?_ h
  clear h
  mvcgen [Core.Record.ofJson, -Spec.throw_Except, throw_spec]
  all_goals first
    | exact ⟨⟨_, rfl⟩, by assumption⟩
    | exact ⟨⟨_, by assumption⟩, by assumption⟩

-- `mvcgen` splits `ledgerEntry?` into one goal per field it parses
set_option maxHeartbeats 1000000 in
/-- A ledger entry's record decodes from its entry, with its schema. -/
theorem ledgerEntry_record {line : Json} {e : LedgerEntry} (h : ledgerEntry? line = some e) :
    ∃ r, (Core.Record.ofJson e.entry).toOption = some r ∧ r.schema = e.schema := by
  refine of_triple_opt (Q := fun e => ∃ r, (Core.Record.ofJson e.entry).toOption = some r ∧ r.schema = e.schema) ?_ h
  clear h
  mvcgen [ledgerEntry?, guard, -Spec.throw_Option, none_spec, failure_spec, toOption_spec, ledgerTag_spec, nat_spec,
    some_pure_spec]
  all_goals exact ⟨_, by assumption, rfl⟩

theorem ledgerEntry_obj {line : Json} {e : LedgerEntry} (h : ledgerEntry? line = some e) :
    (∃ kvs, e.entry = .obj kvs) ∧ getD e.entry "schema" = .str e.schema := by
  obtain ⟨r, hr, hs⟩ := ledgerEntry_record h
  rw [← hs]
  exact recordOfJson_schema (toOption_some hr)

theorem tagged_ok_spec (tag : String) (j : Json) :
    ⦃⌜True⌝⦄ taggedSha512 tag j ⦃⇓? a => ⌜taggedSha512 tag j = .ok a⌝⦄ := ok_spec _

-- `mvcgen` splits `ledgerMessage` into one goal per check it makes of the identifier and the schema
set_option maxHeartbeats 1000000 in
/-- What a ledger entry's message hashes: the role, the identifier and the entry's tagged digest under its schema's
tag, the entry without its `sig`. -/
theorem ledgerMessage_ok {I : ByteArray} {entry : Json} {m : ByteArray} (h : ledgerMessage I entry = .ok m) :
    ∃ schema tag d, getD entry "schema" = .str schema ∧ ledgerTag? schema = some tag ∧
      taggedSha512 tag (Core.Record.unsigned entry) = .ok d ∧ m = Flock.Sha512.hash (ledgerBytes I d) := by
  refine of_triple (Q := fun m => ∃ schema tag d, getD entry "schema" = .str schema ∧ ledgerTag? schema = some tag ∧
      taggedSha512 tag (Core.Record.unsigned entry) = .ok d ∧ m = Flock.Sha512.hash (ledgerBytes I d)) ?_ h
  clear h
  mvcgen [ledgerMessage, -Spec.throw_Except, throw_spec, tagged_ok_spec]
  all_goals exact ⟨_, _, _, by assumption, by assumption, by assumption, rfl⟩

-- `mvcgen` splits `recordMessage` at its check of the identifier
set_option maxHeartbeats 1000000 in
/-- What a record's message hashes: the role, the identifier, the host and the commitment's tagged digest. -/
theorem recordMessage_ok {I : ByteArray} {host : String} {c : LiveCommitment} {m : ByteArray}
    (h : recordMessage I host c = .ok m) :
    ∃ d, taggedSha512 COMMITMENT_TAG c.toJson = .ok d ∧ m = Flock.Sha512.hash (recordBytes I host d) := by
  refine of_triple (Q := fun m => ∃ d, taggedSha512 COMMITMENT_TAG c.toJson = .ok d ∧
      m = Flock.Sha512.hash (recordBytes I host d)) ?_ h
  clear h
  mvcgen [recordMessage, -Spec.throw_Except, throw_spec, tagged_ok_spec]
  all_goals exact ⟨_, by assumption, rfl⟩

/-- What a close's message hashes: the role, the identifier, its record's message, its session, its line count, its
chain and how it ended. -/
theorem closeMessage_ok {I : ByteArray} {host : String} {c : LiveCommitment} {session : String} {lines : Nat}
    {chain : ByteArray} {ended : Option String} {m : ByteArray}
    (h : closeMessage I host c session lines chain ended = .ok m) :
    ∃ r, recordMessage I host c = .ok r ∧ m = Flock.Sha512.hash (closeBytes I r session lines chain ended) := by
  unfold closeMessage at h
  obtain ⟨r, hr, h⟩ := bind_ok.1 h
  refine ⟨r, hr, ?_⟩
  cases ended <;> exact (Except.ok.inj h).symm

/-! ## The strings the finder hashes -/

theorem filterMap_str {f : Json → Option String} (lines : List String) (hf : ∀ s, f (.str s) = some s) :
    (lines.map Json.str).filterMap f = lines := by
  induction lines with
  | nil => rfl
  | cons l ls ih => rw [List.map_cons, List.filterMap_cons_some (hf l), ih]

theorem mem_readHashed_record {r : ChallengerRecord} {x : ByteArray}
    (hr : ChallengerRecord.ofJson (getD (getD coins "session") "record") = .ok r)
    (hx : x ∈ recordHashed (ident key.pub) r.host r.commitment) : x ∈ readHashed key label coins := by
  unfold readHashed
  simp only [hr, List.mem_append, hx, true_or]

theorem mem_readHashed_close {r : ChallengerRecord} {c : ChallengerClose} {x : ByteArray}
    (hr : ChallengerRecord.ofJson (getD (getD coins "session") "record") = .ok r)
    (hc : ChallengerClose.ofJson (getD (getD coins "session") "close") = .ok c)
    (hx : x ∈ closeHashed (ident key.pub) r.host r.commitment c.session c.lines c.chain c.ended) :
    x ∈ readHashed key label coins := by
  unfold readHashed
  simp only [hr, hc, List.mem_append, hx, true_or, or_true]

theorem mem_readHashed_log {r : ChallengerRecord} {ls : Array Json} {lines : List String} {x : ByteArray}
    (hr : ChallengerRecord.ofJson (getD (getD coins "session") "record") = .ok r)
    (hlog : getD (getD coins "session") "log" = .arr ls) (hls : ls.toList = lines.map Json.str)
    (hx : x ∈ logHashed lines) : x ∈ readHashed key label coins := by
  unfold readHashed
  simp only [hr, hlog, hls]
  rw [filterMap_str]
  · simp only [List.mem_append, hx, true_or, or_true]
  · intro s; rfl

theorem commitment_some {s : Run} {c : LiveCommitment} (h : s.commitment = some c) :
    ∃ dom, liveDomain s.scheme s.epoch s.families = .ok dom := by
  unfold Run.commitment at h
  split at h
  · exact ⟨_, by assumption⟩
  · cases h

theorem mem_deviceHashed_run {pk : ByteArray} {s : Run} {c : LiveCommitment} {x : ByteArray} (hs : s ∈ dev.runs pk)
    (hc : s.commitment = some c)
    (hx : x ∈ closeHashed (ident pk) s.host c s.id s.log.length (logChain s.log) s.ended ∨ x ∈ logHashed s.log) :
    x ∈ deviceHashed dev pk := by
  obtain ⟨dom, hdom⟩ := commitment_some hc
  unfold deviceHashed
  simp only [List.mem_append, List.mem_flatMap]
  refine .inl ⟨s, hs, .inr ?_⟩
  simp only [hdom, hc, List.mem_append]
  rcases hx with hx | hx <;> simp only [hx, true_or, or_true]

theorem mem_deviceHashed_ledger {pk : ByteArray} {line : Json} {x : ByteArray} (hl : line ∈ dev.ledger pk)
    (hx : x ∈ ledgerHashed (ident pk) line) : x ∈ deviceHashed dev pk := by
  unfold deviceHashed
  simp only [List.mem_append, List.mem_flatMap]
  exact .inr ⟨line, hl, hx⟩

theorem tagged_mem {tag : String} {j : Json} {s : String} (h : Flock.canonUtf8 j = .ok s) :
    taggedBytes tag s ∈ taggedMessage tag j := by
  unfold taggedMessage
  rw [h]
  exact List.mem_singleton_self _

theorem record_mem_recordHashed {I : ByteArray} {host : String} {c : LiveCommitment} {d : ByteArray}
    (hd : taggedSha512 COMMITMENT_TAG c.toJson = .ok d) : recordBytes I host d ∈ recordHashed I host c := by
  unfold recordHashed
  rw [hd]
  exact List.mem_append_right _ (List.mem_singleton_self _)

theorem tagged_mem_recordHashed {I : ByteArray} {host : String} {c : LiveCommitment} {s : String}
    (hs : Flock.canonUtf8 c.toJson = .ok s) : taggedBytes COMMITMENT_TAG s ∈ recordHashed I host c := by
  unfold recordHashed
  exact List.mem_append_left _ (tagged_mem hs)

theorem close_mem_closeHashed {I : ByteArray} {host : String} {c : LiveCommitment} {session : String} {lines : Nat}
    {chain : ByteArray} {ended : Option String} {r : ByteArray} (hr : recordMessage I host c = .ok r) :
    closeBytes I r session lines chain ended ∈ closeHashed I host c session lines chain ended := by
  unfold closeHashed
  rw [hr]
  cases ended <;> exact List.mem_append_right _ (List.mem_singleton_self _)

theorem record_mem_closeHashed {I : ByteArray} {host : String} {c : LiveCommitment} {session : String} {lines : Nat}
    {chain : ByteArray} {ended : Option String} {x : ByteArray} (hx : x ∈ recordHashed I host c) :
    x ∈ closeHashed I host c session lines chain ended := by
  unfold closeHashed
  exact List.mem_append_left _ hx

theorem ledger_mem_ledgerHashed {I : ByteArray} {line : Json} {e : LedgerEntry} {m : ByteArray}
    (he : ledgerEntry? line = some e) (hm : ledgerMessage I e.entry = .ok m) :
    ∃ d, m = Flock.Sha512.hash (ledgerBytes I d) ∧ ledgerBytes I d ∈ ledgerHashed I line := by
  obtain ⟨schema, tag, d, hs, ht, hd, rfl⟩ := ledgerMessage_ok hm
  obtain ⟨⟨kvs, hkvs⟩, hs'⟩ := ledgerEntry_obj he
  obtain rfl : schema = e.schema := Json.str.inj (hs.symm.trans hs')
  refine ⟨d, rfl, ?_⟩
  unfold ledgerHashed
  rw [he]
  simp only [ht]
  have hb : Json.mkObj ((entries e.entry).toList.filter (·.1 != "sig")) = Core.Record.unsigned e.entry := by
    rw [hkvs]
    rfl
  rw [hb, hd]
  exact List.mem_append_right _ (List.mem_singleton_self _)

theorem close_mem_signatures {r : ChallengerRecord} {c : ChallengerClose} {m : ByteArray}
    (hr : ChallengerRecord.ofJson (getD (getD coins "session") "record") = .ok r)
    (hc : ChallengerClose.ofJson (getD (getD coins "session") "close") = .ok c)
    (hm : closeMessage (ident key.pub) r.host r.commitment c.session c.lines c.chain c.ended = .ok m) :
    (m, c.signature) ∈ signatures key coins := by
  unfold signatures
  simp only [hr, hc, hm]
  exact List.mem_append_left _ (List.mem_append_right _ (List.mem_singleton_self _))

/-! ## The binding -/

/-- **What the device signed under `pk` that is a close's message is a run's close**, or the finder outputs a break:
the record's and the ledger's messages begin with roles of other lengths. -/
theorem signed_close {r m : ByteArray} {session : String} {lines : Nat} {chain : ByteArray}
    (hsigned : Signed dev key.pub m)
    (hm : m = Flock.Sha512.hash (closeBytes (ident key.pub) r session lines chain none))
    (hmem : closeBytes (ident key.pub) r session lines chain none ∈ readHashed key label coins) :
    (∃ s ∈ dev.runs key.pub, ∃ cs rs, s.commitment = some cs ∧ recordMessage (ident key.pub) s.host cs = .ok rs ∧
      closeBytes (ident key.pub) r session lines chain none =
        closeBytes (ident key.pub) rs s.id s.log.length (logChain s.log) s.ended) ∨ Break dev key label coins := by
  have hU : closeBytes (ident key.pub) r session lines chain none ∈
      readHashed key label coins ++ deviceHashed dev key.pub := List.mem_append_left _ hmem
  unfold Signed signedBy at hsigned
  rcases List.mem_append.1 hsigned with hrun | hled
  · obtain ⟨s, hs, hms⟩ := List.mem_flatMap.1 hrun
    cases hcs : s.commitment with
    | none => rw [hcs] at hms; cases hms
    | some cs =>
      rw [hcs] at hms
      rcases List.mem_append.1 hms with hrec | hcl
      · cases hrm : recordMessage (ident key.pub) s.host cs with
        | error _ => rw [hrm] at hrec; cases hrec
        | ok m' =>
          rw [hrm] at hrec
          obtain rfl : m = m' := List.mem_singleton.1 hrec
          obtain ⟨d, hd, hm'⟩ := recordMessage_ok hrm
          have hV : recordBytes (ident key.pub) s.host d ∈ readHashed key label coins ++ deviceHashed dev key.pub :=
            List.mem_append_right _ (mem_deviceHashed_run hs hcs (.inl (record_mem_closeHashed
              (record_mem_recordHashed hd))))
          rcases eq_or_break hU hV (hm.symm.trans hm') with e | b
          · exact absurd e (closeBytes_ne_record _ _ _ _ _ _ _ _ _)
          · exact .inr b
      · cases hcm : closeMessage (ident key.pub) s.host cs s.id s.log.length (logChain s.log) s.ended with
        | error _ => rw [hcm] at hcl; cases hcl
        | ok m' =>
          rw [hcm] at hcl
          obtain rfl : m = m' := List.mem_singleton.1 hcl
          obtain ⟨rs, hrs, hm'⟩ := closeMessage_ok hcm
          have hV : closeBytes (ident key.pub) rs s.id s.log.length (logChain s.log) s.ended ∈
              readHashed key label coins ++ deviceHashed dev key.pub :=
            List.mem_append_right _ (mem_deviceHashed_run hs hcs (.inl (close_mem_closeHashed hrs)))
          rcases eq_or_break hU hV (hm.symm.trans hm') with e | b
          · exact .inl ⟨s, hs, cs, rs, hcs, hrs, e⟩
          · exact .inr b
  · obtain ⟨line, hline, hlm⟩ := List.mem_filterMap.1 hled
    obtain ⟨e, he, hem⟩ := Option.bind_eq_some_iff.1 hlm
    obtain ⟨d, hd, hdm⟩ := ledger_mem_ledgerHashed he (toOption_some hem)
    have hV : ledgerBytes (ident key.pub) d ∈ readHashed key label coins ++ deviceHashed dev key.pub :=
      List.mem_append_right _ (mem_deviceHashed_ledger hline hdm)
    rcases eq_or_break hU hV (hm.symm.trans hd) with e | b
    · exact absurd e (closeBytes_ne_ledger _ _ _ _ _ _ _ _)
    · exact .inr b

theorem utf8_size_le (s : String) : s.toUTF8.size ≤ 4 * s.length := by
  refine String.push_induction s (fun s => s.toUTF8.size ≤ 4 * s.length) (by simp) ?_
  intro b c ih
  simp only [String.toUTF8_eq_toByteArray, String.size_toByteArray, String.utf8ByteSize_push,
    String.length_push] at ih ⊢
  have := Char.utf8Size_le_four c
  omega

theorem session_size {s : String} (h : isSessionId s = true) : s.toUTF8.size ≤ 512 := by
  have hl : s.length ≤ 128 := by
    unfold isSessionId at h
    simp only [Bool.and_eq_true, decide_eq_true_eq] at h
    exact h.1
  have := utf8_size_le s
  omega

theorem close_form {chain : ByteArray} {lines : Nat} {session : String}
    (h : (chain.size == 64 && 1 ≤ lines && lines ≤ MAX_LOG_LINES && isSessionId session) = true) :
    chain.size = 64 ∧ isSessionId session = true := by
  simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at h
  exact ⟨h.1.1.1, h.2⟩

-- `mvcgen` splits `ChallengerClose.ofJson` into one goal per field it parses
set_option maxHeartbeats 1000000 in
/-- A close's chain is 64 bytes and its session id at most 128 characters. -/
theorem closeOfJson_ok {j : Json} {c : ChallengerClose} (h : ChallengerClose.ofJson j = .ok c) :
    c.chain.size = 64 ∧ isSessionId c.session = true := by
  refine of_triple (Q := fun c => c.chain.size = 64 ∧ isSessionId c.session = true) ?_ h
  clear h
  mvcgen [ChallengerClose.ofJson, -Spec.throw_Except, throw_spec]
  all_goals exact close_form (by assumption)

/-- **A session `sessionOf` accepts from a sample's record is a run the device closed, its log that run's**, or the
finder outputs a break: the commitment, the roots and the session id `sessionOf` returns from the record's session
files are a run's of the device under the key, a run its client closed, the roots the ones that run's own release log
shows, and the session's log is that run's lines. -/
theorem sessionOf_run {c : LiveCommitment} {roots : Roots} {sid : String}
    (h : sessionOf key (getD (getD coins "session") "record") (getD (getD coins "session") "log")
      (getD (getD coins "session") "close") = .ok (c, roots, sid)) :
    (∃ s ∈ dev.runs key.pub, s.commitment = some c ∧ s.id = sid ∧ s.ended = none ∧
      releasedRootsOf c s.log = .ok roots ∧
        ∃ ls : Array Json, getD (getD coins "session") "log" = .arr ls ∧ ls.toList = s.log.map Json.str) ∨
      Break dev key label coins := by
  obtain ⟨r, cl, m, ls, lines, hr, hcl, hm, hv, hlog, hls, hch, hend, hroots, hres⟩ := sessionOf_read h
  have hc : c = r.commitment := congrArg Prod.fst hres
  have hsid0 : sid = cl.session := congrArg (fun x => x.2.2) hres
  subst hc hsid0
  have hm' : closeMessage (ident key.pub) r.host r.commitment cl.session cl.lines cl.chain cl.ended = .ok m := hm
  obtain ⟨rm, hrm, hmh⟩ := closeMessage_ok hm'
  rw [hend] at hmh
  have hmem : closeBytes (ident key.pub) rm cl.session cl.lines cl.chain none ∈ readHashed key label coins := by
    have hx := mem_readHashed_close (label := label) hr hcl
      (close_mem_closeHashed (session := cl.session) (lines := cl.lines) (chain := cl.chain) (ended := cl.ended) hrm)
    rwa [hend] at hx
  obtain ⟨hchs, hsid⟩ := closeOfJson_ok hcl
  rcases signed_or_break (dev := dev) (label := label) (close_mem_signatures hr hcl hm') hv with hs | b
  · rcases signed_close hs hmh hmem with ⟨s, hs, cs, rs, hcs, hrs, he⟩ | b
    · obtain ⟨d, hd, hrmd⟩ := recordMessage_ok hrm
      obtain ⟨ds, hds, hrsd⟩ := recordMessage_ok hrs
      have hrsize : rm.size = 64 := by rw [hrmd]; exact hash_size _
      have hrssize : rs.size = 64 := by rw [hrsd]; exact hash_size _
      obtain ⟨rfl, hsid', hchain, hended⟩ :=
        closeBytes_inj hrsize hrssize (session_size hsid) hchs (logChain_size _) he
      have hRr : recordBytes (ident key.pub) r.host d ∈ readHashed key label coins ++ deviceHashed dev key.pub :=
        List.mem_append_left _ (mem_readHashed_record hr (record_mem_recordHashed hd))
      have hRs : recordBytes (ident key.pub) s.host ds ∈ readHashed key label coins ++ deviceHashed dev key.pub :=
        List.mem_append_right _ (mem_deviceHashed_run hs hcs (.inl (record_mem_closeHashed
          (record_mem_recordHashed hds))))
      obtain ⟨t, ht, hdt⟩ := tagged_ok hd
      obtain ⟨ts, hts, hdts⟩ := tagged_ok hds
      rcases eq_or_break hRr hRs (hrmd.symm.trans hrsd) with e | b
      · obtain ⟨-, rfl⟩ := recordBytes_inj (by rw [hdt]; exact hash_size _) (by rw [hdts]; exact hash_size _) e
        have hTr : taggedBytes COMMITMENT_TAG t ∈ readHashed key label coins ++ deviceHashed dev key.pub :=
          List.mem_append_left _ (mem_readHashed_record hr (tagged_mem_recordHashed ht))
        have hTs : taggedBytes COMMITMENT_TAG ts ∈ readHashed key label coins ++ deviceHashed dev key.pub :=
          List.mem_append_right _ (mem_deviceHashed_run hs hcs (.inl (record_mem_closeHashed
            (tagged_mem_recordHashed hts))))
        rcases eq_or_break hTr hTs (hdt.symm.trans hdts) with e | b
        · obtain rfl := taggedBytes_inj e
          obtain rfl := Commitment.canon_inj _ _ _ ht hts
          have hstart : field LOG_TAG.toUTF8 ∈ readHashed key label coins ++ deviceHashed dev key.pub :=
            List.mem_append_left _ (mem_readHashed_log hr hlog hls (by rw [logHashed_eq]; simp))
          rcases chain_bind hstart (lines.length + s.log.length) lines s.log (Nat.le_refl _)
              (fun y hy => List.mem_append_left _ (mem_readHashed_log hr hlog hls
                (by rw [logHashed_eq]; simp [hy])))
              (fun y hy => List.mem_append_right _ (mem_deviceHashed_run hs hcs
                (.inr (by rw [logHashed_eq]; simp [hy]))))
              (hch.trans hchain) with e | b
          · refine .inl ⟨s, hs, hcs, hsid'.symm, hended, ?_, ls, hlog, ?_⟩
            · rw [← e]
              exact hroots
            · rw [← e]
              exact hls
          · exact .inr b
        · exact .inr b
      · exact .inr b
    · exact .inr b
  · exact .inr b

/-- **A session `sessionOf` accepts from a sample's record is a run the device closed**, or the finder outputs a break:
`sessionOf_run` without the log. -/
theorem sessionOf_bound {c : LiveCommitment} {roots : Roots} {sid : String}
    (h : sessionOf key (getD (getD coins "session") "record") (getD (getD coins "session") "log")
      (getD (getD coins "session") "close") = .ok (c, roots, sid)) :
    (∃ s ∈ dev.runs key.pub, s.commitment = some c ∧ s.id = sid ∧ s.ended = none ∧
      releasedRootsOf c s.log = .ok roots) ∨ Break dev key label coins := by
  rcases sessionOf_run (dev := dev) (label := label) h with ⟨s, hs, hc, hid, he, hr, -⟩ | b
  · exact .inl ⟨s, hs, hc, hid, he, hr⟩
  · exact .inr b

/-- **A sample `drawCoins` accepts reads its coins against a run the device closed**, or the finder outputs a break: the
commitment and the roots its segment's coins are read against are a run's of the device under the key, a run its client
closed, the roots the ones that run's own release log shows. -/
theorem drawCoins_session {w : String} {after ξ : ByteArray} (h : drawCoins key w label after coins = .ok ξ) :
    ∃ c roots, challengerStream c (getD coins "segment") after
        (some (((roots.find? (·.1 == label)).map (·.2)).getD [])) = .ok ξ ∧
      ((∃ s ∈ dev.runs key.pub, s.commitment = some c ∧ s.ended = none ∧ releasedRootsOf c s.log = .ok roots) ∨
        Break dev key label coins) := by
  obtain ⟨c, roots, ht, hs⟩ := drawCoins_ok h
  obtain ⟨a, -, hn⟩ := trustedSessionOf_ok ht
  obtain ⟨sid, ho⟩ := sessionNamed_ok hn
  refine ⟨c, roots, hs, ?_⟩
  rcases sessionOf_bound (dev := dev) (label := label) ho with ⟨s, hs', hc, -, he, hr⟩ | b
  · exact .inl ⟨s, hs', hc, he, hr⟩
  · exact .inr b

end Properties.Challenger.Session
