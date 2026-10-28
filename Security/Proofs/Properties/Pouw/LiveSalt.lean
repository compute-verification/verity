import Properties.Pouw.LiveSalt
import Proofs.Properties.Pouw.LiveSalt.Ledger
import Proofs.Properties.Pouw.Commit
import Proofs.Properties.NetworkRecords.Bytes

/-!
# Proofs of the Pearl-C live salt's properties (`Properties.Pouw.LiveSalt`)

The epoch's root is `PouwCheck.epochOpenRoot`, the port's own (`epochOpenRoot_eq`). Its message is the tag, the 64-byte
weights' root and u64be of the epoch, so two equal messages split at fixed offsets into equal parts (`epochRootBinding`).
A row seed's message ends in u64le of `u`·2^32 + `i`, so two equal messages have one call index and row
(`rowSeedBinding`). The ledger opens a session once (`LedgerShows`' last clause), so two acceptances by one session read
one open and its one root (`oneRoot_of`), and so for `PouwCheck.liveSaltCheck` (`oneRoot`).

`PouwCheck.liveSaltCheck` accepts only what `Accepted` describes (`acceptMeans`): each check it makes is one conjunct,
and the ledger's are `LedgerShows`, from the ledger's invariant (`Ledger.ledgerShows_of`). Two accepted passes of one
epoch are pinned on one root, so they read one session of the ledger, and each one's commitment has the root whose hex
the session's `end` recorded (`Ledger.sessionNamed_root`). So, given `LiveCoinBinding`, their salt coins are one coin
and their salts one salt, or the openings exhibit a collision (`onePair_of_binding`). `LiveSalt.Coin` proves
`LiveCoinBinding` (`liveCoinBinding`), so `onePair` holds with no hypothesis.
-/

set_option mvcgen.warning false

namespace Properties.Pouw.LiveSalt

open Lean (Json)
open Std.Do
open Properties.NetworkCertifier.Bytes (bs bs_append bs_length bs_inj)
open SampledProofs.Challenger (Audited ChallengerKey LedgerEntry LiveCommitment LiveOpening Roots familiesJson liveReason
  pinnedSession sessionNamed)
open Properties.Pouw.LiveSalt.Ledger (bind_ok of_triple ok_spec throw_spec ledgerShows_of sessionNamed_root)

/-- The epoch's root is the port's: `PouwCheck.epochOpenRoot`. -/
theorem epochOpenRoot_eq (w : ByteArray) (k : Nat) : epochOpenRoot w k = PouwCheck.epochOpenRoot w k := rfl

theorem epochRootBinding : EpochRootBinding := by
  intro w w' k k' hw hw' hk hk' h
  by_cases hm : epochMessage w k = epochMessage w' k'
  · left
    have e := congrArg bs hm
    simp only [epochMessage, bs_append, List.append_assoc] at e
    have e1 := List.append_cancel_left e
    obtain ⟨e2, e3⟩ := List.append_inj e1 (by rw [bs_length, bs_length, hw, hw'])
    exact ⟨bs_inj e2, Properties.Pouw.Commit.be8_inj k k' hk hk' e3⟩
  · right
    exact ⟨_, List.mem_singleton_self _, _, List.mem_singleton_self _, hm, h⟩

theorem le64_data (x : Nat) : (le64 x).data.toList =
    [(x >>> 0).toUInt8, (x >>> 8).toUInt8, (x >>> 16).toUInt8, (x >>> 24).toUInt8,
     (x >>> 32).toUInt8, (x >>> 40).toUInt8, (x >>> 48).toUInt8, (x >>> 56).toUInt8] := by
  simp [le64, List.range_succ]

theorem le64_inj (x y : Nat) (hx : x < 2 ^ 64) (hy : y < 2 ^ 64)
    (h : (le64 x).data.toList = (le64 y).data.toList) : x = y := by
  simp only [le64_data, List.cons.injEq] at h
  obtain ⟨h0, h1, h2, h3, h4, h5, h6, h7, -⟩ := h
  simp only [← UInt8.toNat_inj] at h0 h1 h2 h3 h4 h5 h6 h7
  simp [Nat.shiftRight_eq_div_pow] at h0 h1 h2 h3 h4 h5 h6 h7
  omega

theorem le64_length (x : Nat) : (bs (le64 x)).length = 8 := by
  simp [bs, le64_data]

theorem rowSeedBinding : RowSeedBinding := by
  intro salt leaf rootB salt' leaf' rootB' u i u' i' hu hi hu' hi' hne h
  refine ⟨_, List.mem_singleton_self _, _, List.mem_singleton_self _, fun hm => hne ?_, h⟩
  have e := congrArg bs hm
  simp only [rowSeedMessage, bs_append] at e
  have e8 := (List.append_inj' e (by rw [le64_length, le64_length])).2
  have hx := le64_inj _ _ (by omega) (by omega) e8
  obtain ⟨rfl, rfl⟩ : u = u' ∧ i = i' := by omega
  rfl

/-- One session salts one epoch, for any check: the ledger opens a session once, so two acceptances by one session read
one open, and its root's hex names one root. -/
theorem oneRoot_of (check : Check) : OneRoot check := by
  intro _ _ _ _ _ _ _ _ _ _ _ _ w k a _ _ w' k' a' _ _ h h' hs
  obtain ⟨_, _, _, -, -, -, -, -, -, hl, -⟩ := h
  obtain ⟨_, _, _, -, -, -, -, -, -, hl', -⟩ := h'
  obtain ⟨_, op, _, -, hop, -, -, ⟨hk, hss, -, -, -, -, hr⟩, -, huniq⟩ := hl
  obtain ⟨_, op', _, -, hop', -, -, ⟨hk', hss', -, -, -, -, hr'⟩, -, -⟩ := hl'
  have heq : op' = op := huniq op' hop' hk' (Or.inr (hss'.trans hs.symm))
  subst heq
  exact Properties.NetworkRecords.Bytes.hexOf_inj (Json.str.inj (hr.symm.trans hr'))

theorem oneRoot : OneRoot PouwCheck.liveSaltCheck := oneRoot_of _

theorem mapError_spec {ε ε' α : Type} (x : Except ε α) (f : ε → ε') :
    ⦃⌜True⌝⦄ x.mapError f ⦃⇓? a => ⌜x = .ok a⌝⦄ := by
  cases x
  · exact fun _ => True.intro
  · exact fun _ => rfl

theorem workloadOf_spec (m : Json) :
    ⦃⌜True⌝⦄ PouwCheck.workloadOf m ⦃⇓? a => ⌜PouwCheck.workloadOf m = .ok a⌝⦄ := ok_spec _

theorem checkWorkload_spec (us : List PouwCheck.Matmul) (roots : List (Nat × ByteArray)) :
    ⦃⌜True⌝⦄ PouwCheck.checkWorkload us roots ⦃⇓? a => ⌜PouwCheck.checkWorkload us roots = .ok a⌝⦄ := ok_spec _

theorem saltSession_spec (key ledger pins main window session : Json) :
    ⦃⌜True⌝⦄ PouwCheck.saltSession key ledger pins main window session
    ⦃⇓? a => ⌜PouwCheck.saltSession key ledger pins main window session = .ok a⌝⦄ := ok_spec _

theorem port_spec {α : Type} (x : Except String α) : ⦃⌜True⌝⦄ PouwCheck.port x ⦃⇓? a => ⌜x = .ok a⌝⦄ :=
  mapError_spec _ _

theorem throwR_spec {α : Type} (e : String × String) (Q : PostCond α (.except (String × String) .pure)) :
    ⦃Q.2.1 e⦄ (MonadExceptOf.throw e : PouwCheck.R α) ⦃Q⦄ := fun h => h

theorem mapError_ok {ε ε' α : Type} {x : Except ε α} {f : ε → ε'} {a : α} : x.mapError f = .ok a ↔ x = .ok a := by
  cases x <;> simp [Except.mapError]

theorem map_ok {ε α β : Type} {x : Except ε α} {f : α → β} {b : β} :
    (f <$> x) = .ok b ↔ ∃ a, x = .ok a ∧ f a = b := by
  cases x <;> simp [Functor.map, Except.map]

theorem saltSession_ok {key ledger pins main window session : Json} {res : Audited × LiveCommitment × Roots}
    (h : PouwCheck.saltSession key ledger pins main window session = .ok res) :
    ∃ ck, ChallengerKey.ofJson key = .ok ck ∧ pinnedSession ledger window ck pins main = .ok res.1 ∧
      sessionNamed res.1 ck session = .ok res.2 := by
  unfold PouwCheck.saltSession at h
  rw [bind_ok] at h
  obtain ⟨ck, hck, h⟩ := h
  rw [mapError_ok] at hck h
  rw [bind_ok] at h
  obtain ⟨a, ha, h⟩ := h
  rw [map_ok] at h
  obtain ⟨cr, hcr, rfl⟩ := h
  exact ⟨ck, hck, ha, hcr⟩

theorem field_eq (j : Json) (k : String) : PouwWindow.field? j k = SampledProofs.field? j k := by
  cases j <;> rfl

theorem at_eq (j : Json) (k : String) : PouwCheck.at? j k = SampledProofs.getD j k := by
  simp only [PouwCheck.at?, SampledProofs.getD, field_eq]

theorem jsonStrIs_eq {j : Json} {s : String} (h : PouwCheck.jsonStrIs j s = true) : j = .str s := by
  cases j <;> simp_all [PouwCheck.jsonStrIs]

theorem int_of_intOf {j : Json} {n : Int} (h : PouwWindow.intOf? j = some n) : SampledProofs.int? j = some n := by
  cases j with
  | num x =>
    obtain ⟨m, e⟩ := x
    cases e with
    | zero => simpa [PouwWindow.intOf?, SampledProofs.int?] using h
    | succ e => simp [PouwWindow.intOf?] at h
  | _ => simp [PouwWindow.intOf?] at h

theorem sameCommitment_eq {a b : LiveCommitment} (h : PouwCheck.sameCommitment a b = true) : a = b := by
  simp only [PouwCheck.sameCommitment, Bool.and_eq_true, beq_iff_eq] at h
  obtain ⟨⟨⟨h1, h2⟩, h3⟩, h4⟩ := h
  cases a; cases b; simp_all

theorem saltEpoch_of {c : LiveCommitment} {x : Option String} (hx : ∀ why, x = some why → False)
    (h : PouwWindow.saltEpochReason c = x) : SaltEpoch c := by
  subst h
  unfold PouwWindow.saltEpochReason at hx
  by_cases h1 : (c.scheme != PouwWindow.LIVE_SCHEME) = true
  · rw [ite_eq_left h1] at hx; exact absurd rfl (hx _)
  · by_cases h2 : (c.families != PouwWindow.SALT_FAMILIES) = true
    · rw [ite_eq_right h1, ite_eq_left h2] at hx; exact absurd rfl (hx _)
    · simp only [bne_iff_ne, ne_eq, Decidable.not_not] at h1 h2
      exact ⟨h1, h2⟩

theorem filter_some {x : Option Nat} {k : Nat}
    (h : Option.filter (fun x => decide (x < 18446744073709551616)) x = some k) : x = some k ∧ k < 2 ^ 64 := by
  rw [Option.filter_eq_some_iff] at h
  obtain ⟨h1, h2⟩ := h
  exact ⟨h1, by simpa using h2⟩

theorem hexOf_ne {r : ByteArray} (h : r.size = 64) : Flock.hexOf r ≠ "" := by
  intro he
  have e := congrArg String.toList he
  rw [Properties.NetworkRecords.Bytes.hexOf_toList, String.toList_empty] at e
  have e' := List.flatMap_eq_nil_iff.1 e
  have hl : r.data.toList.length = 64 := by rw [Array.length_toList]; exact h
  obtain ⟨x, hx⟩ := List.exists_mem_of_ne_nil r.data.toList (by intro hn; rw [hn] at hl; cases hl)
  have := Properties.NetworkRecords.Bytes.hexOf_length x
  rw [e' x hx] at this
  cases this

theorem none_of {x : Option String} (h : ∀ why, x = some why → False) : x = none := by
  cases x with
  | none => rfl
  | some why => exact absurd rfl (h why)

theorem carries_names {s : Json} {sid : String} {seq : Nat} {root : ByteArray} {cm : String}
    (h : PouwCheck.carries s sid seq root cm = true) :
    SampledProofs.getD s "session" = .str sid ∧ SampledProofs.int? (SampledProofs.getD s "seq") = some (seq : Int) ∧
      SampledProofs.getD s "root" = .str (Flock.hexOf root) ∧ SampledProofs.getD s "commitment" = .str cm ∧
      SampledProofs.field? s "secret" = none := by
  cases s with
  | obj kv =>
    simp only [PouwCheck.carries, Bool.and_eq_true, Option.isNone_iff_eq_none, beq_iff_eq] at h
    obtain ⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, h5⟩ := h
    refine ⟨?_, ?_, ?_, ?_, ?_⟩
    · rw [← at_eq]; exact jsonStrIs_eq h2
    · rw [← at_eq]; exact int_of_intOf h3
    · rw [← at_eq]; exact jsonStrIs_eq h4
    · rw [← at_eq]; exact jsonStrIs_eq h5
    · rw [← field_eq]; exact h1
  | _ => simp [PouwCheck.carries] at h

theorem epochOpenRoot_size (w : ByteArray) (k : Nat) : (PouwCheck.epochOpenRoot w k).size = 64 :=
  Properties.Pouw.Commit.sha512_size _

/-- What `PouwCheck.liveSaltCheck` has checked on the path that returns, in the forms it checks them, is `Accepted`. -/
theorem accepted_of {key ledger pins main session : Json} {kv : Std.TreeMap.Raw String Json compare}
    {us : List PouwCheck.Matmul × List (Nat × ByteArray)} {u : Unit} {k : Nat}
    {res : Audited × LiveCommitment × Roots} {reason : Option String} {rv op en : LedgerEntry}
    {claimed : LiveCommitment} {o : LiveOpening} {lr : Option String} {seed : ByteArray}
    (hschema : PouwCheck.jsonStrIs (PouwCheck.at? (Json.obj kv) "schema") PouwCheck.MANIFEST_SCHEMA = true)
    (hwork : PouwCheck.workloadOf (Json.obj kv) = .ok us)
    (hcw : PouwCheck.checkWorkload us.1 us.2 = .ok u)
    (hk : Option.filter (fun x => decide (x < 18446744073709551616))
      (PouwWindow.natOf? (PouwCheck.at? (Json.obj kv) "pass_index")) = some k)
    (hsess : PouwCheck.saltSession key ledger pins main (Json.mkObj [("root", Json.str (Flock.hexOf
      (PouwCheck.epochOpenRoot (PouwCheck.weightsRootAt 64 "verity/pouw/weights/v1" us.2) k)))]) session = .ok res)
    (hre : PouwWindow.saltEpochReason res.2.1 = reason) (hreason : ∀ why, reason = some why → False)
    (hfind : (List.find? (fun x => x.kind == "reserve") (PouwCheck.sessionEntries ledger res.1.session),
      List.find? (fun x => x.kind == "open") (PouwCheck.sessionEntries ledger res.1.session),
      List.find? (fun x => x.kind == "end") (PouwCheck.sessionEntries ledger res.1.session)) =
        (some rv, some op, some en))
    (hfam : SampledProofs.jeq (SampledProofs.getD rv.body "families") (familiesJson PouwWindow.SALT_FAMILIES) = true)
    (hres : (SampledProofs.int? (SampledProofs.getD op.body "reserve") == some (rv.seq : Int)) = true)
    (hcomp : (PouwCheck.jsonTrue (SampledProofs.getD en.body "complete") &&
      SampledProofs.jeq (SampledProofs.getD en.body "released")
        (Json.mkObj [(PouwWindow.SALT, Lean.toJson (1 : Nat))])) = true)
    (hrecv : (res.2.2 == [(PouwWindow.SALT,
      [PouwCheck.epochOpenRoot (PouwCheck.weightsRootAt 64 "verity/pouw/weights/v1" us.2) k])]) = true)
    (hcoins : LiveCommitment.ofJson (PouwCheck.at? (Json.obj kv) "coins") = .ok claimed)
    (hsame : PouwCheck.sameCommitment claimed res.2.1 = true)
    (hopen : LiveOpening.ofJson (PouwCheck.at? (Json.obj kv) "salt_opening") = .ok o)
    (hlive : liveReason res.2.1 o PouwWindow.SALT 0
      (PouwCheck.epochOpenRoot (PouwCheck.weightsRootAt 64 "verity/pouw/weights/v1" us.2) k) = .ok lr)
    (hlr : ∀ why, lr = some why → False)
    (hseed : PouwCommit.key? o.coin PouwCommit.EPOCH_SALT_DOMAIN
      [("epoch", .int res.2.1.epoch), ("coins", .bytes res.2.1.root),
        ("weights", .bytes (PouwCheck.weightsRootAt 64 "verity/pouw/weights/v1" us.2))] = .ok seed)
    (hsalt : PouwCheck.jsonStrIs (PouwCheck.at? (Json.obj kv) "salt")
      (Flock.hexOf (FlockSoundness.Randomness.stream seed [.str "salt"] 32)) = true)
    (hcarries : PouwCheck.carries (PouwCheck.at? (Json.obj kv) "session") res.1.session op.seq
      (PouwCheck.epochOpenRoot (PouwCheck.weightsRootAt 64 "verity/pouw/weights/v1" us.2) k) res.1.commitment = true) :
    ∃ w k a c o, Accepted key ledger pins main session (Json.obj kv) w k a c o
      (FlockSoundness.Randomness.stream seed [.str "salt"] 32) := by
  obtain ⟨ck, hck, hpin, hnamed⟩ := saltSession_ok hsess
  rw [sameCommitment_eq hsame] at hcoins
  obtain rfl : lr = none := none_of hlr
  obtain ⟨hpi, hk64⟩ := filter_some hk
  exact ⟨_, k, res.1, res.2.1, o, ck, op.seq, res.2.2, hck, jsonStrIs_eq hschema,
    ⟨⟨us.1, us.2, hwork, hcw, rfl⟩, hpi, hk64⟩, hpin, hnamed, ⟨saltEpoch_of hreason hre, beq_iff_eq.1 hrecv⟩,
    ledgerShows_of (hexOf_ne (epochOpenRoot_size _ _)) hpin hfind hfam hres hcomp, carries_names hcarries, hcoins,
    hopen, hlive, by unfold liveSalt; rw [hseed]; rfl, jsonStrIs_eq hsalt⟩

-- `mvcgen` splits `PouwCheck.liveSaltCheck` into one goal per check, and all but one throw
set_option maxHeartbeats 4000000 in
/-- `PouwCheck.liveSaltCheck` accepts only what `Accepted` describes. -/
theorem acceptMeans : AcceptMeans PouwCheck.liveSaltCheck := by
  intro key ledger pins main session m salt h
  refine of_triple (Q := fun salt => ∃ w k a c o, Accepted key ledger pins main session m w k a c o salt) ?_ h
  clear h
  mvcgen [PouwCheck.liveSaltCheck, PouwCheck.manifest, PouwCheck.coin, -Spec.throw_Except, throw_spec, throwR_spec,
    workloadOf_spec, checkWorkload_spec, saltSession_spec, port_spec, mapError_spec]
  apply accepted_of <;> assumption

/-- A manifest names one epoch's root and one pass index. -/
theorem epochRoot_unique {m : Json} {w w' : ByteArray} {k k' : Nat} (h : EpochRoot m w k) (h' : EpochRoot m w' k') :
    w = w' ∧ k = k' := by
  obtain ⟨⟨us, roots, hw, -, rfl⟩, hk, -⟩ := h
  obtain ⟨⟨us', roots', hw', -, rfl⟩, hk', -⟩ := h'
  rw [hw] at hw'
  cases hw'
  exact ⟨rfl, Option.some.inj (hk.symm.trans hk')⟩

/-- `LiveCoinBinding` of two commitments with one root. -/
theorem binding_of_root (hb : LiveCoinBinding) {c c' : LiveCommitment} {o o' : LiveOpening} {label : String}
    {index : Int} {after after' : ByteArray} (hr : c.root = c'.root) (h : liveReason c o label index after = .ok none)
    (h' : liveReason c' o' label index after' = .ok none) :
    (c = c' ∧ o.coin = o'.coin) ∨ Collides (hashedBy c o label index) (hashedBy c' o' label index) := by
  obtain ⟨s, e, fs, r⟩ := c
  obtain ⟨s', e', fs', r'⟩ := c'
  obtain rfl : r = r' := hr
  exact hb s s' e e' fs fs' _ o o' label index after after' h h'

/-- `OnePair` for `PouwCheck.liveSaltCheck`, given `LiveCoinBinding`: two accepted passes of one epoch are pinned on one
root, so they read one session and two commitments with one root; the binding gives one coin, and one coin one salt,
unless the openings exhibit a collision. -/
theorem onePair_of_binding (hb : LiveCoinBinding) : OnePair PouwCheck.liveSaltCheck := by
  intro key ledger pins main session session' m m' salt salt' h h' w k he he'
  obtain ⟨w1, k1, a, c, o, hA⟩ := acceptMeans _ _ _ _ _ _ _ h
  obtain ⟨w2, k2, a', c', o', hA'⟩ := acceptMeans _ _ _ _ _ _ _ h'
  obtain ⟨ck, os, rcv, hck, hsch, hER, hpin, hnamed, hcounts, hshows, hnames, hcoins, hop, hlive, hsalt, hsaltj⟩ := hA
  obtain ⟨ck', os', rcv', hck', hsch', hER', hpin', hnamed', hcounts', hshows', hnames', hcoins', hop', hlive',
    hsalt', hsaltj'⟩ := hA'
  obtain ⟨rfl, rfl⟩ := epochRoot_unique hER he
  obtain ⟨rfl, rfl⟩ := epochRoot_unique hER' he'
  obtain rfl : ck = ck' := Except.ok.inj (hck.symm.trans hck')
  obtain rfl : a = a' := Except.ok.inj (hpin.symm.trans hpin')
  have hroot : c.root = c'.root := Properties.NetworkRecords.Bytes.hexOf_inj
    ((sessionNamed_root hnamed).trans (sessionNamed_root hnamed').symm)
  refine ⟨a, c, c', o, o', ⟨ck, os, rcv, hck, hsch, hER, hpin, hnamed, hcounts, hshows, hnames, hcoins, hop, hlive, hsalt,
    hsaltj⟩, ⟨ck, os', rcv', hck', hsch', hER', hpin', hnamed', hcounts', hshows', hnames', hcoins', hop', hlive',
    hsalt', hsaltj'⟩, ?_⟩
  rcases binding_of_root hb hroot hlive hlive' with ⟨rfl, hcoin⟩ | hcol
  · left
    rw [hcoin] at hsalt
    exact Except.ok.inj (hsalt.symm.trans hsalt')
  · exact .inr hcol

end Properties.Pouw.LiveSalt
